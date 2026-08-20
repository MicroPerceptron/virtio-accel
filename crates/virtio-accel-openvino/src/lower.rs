//! TOSA 1.0 to OpenVINO IR lowering.
//!
//! This module intentionally owns the OpenVINO IR (version 11) XML and weights encoding.
//! Portable crates expose only the verified TOSA model and provider-neutral analysis; no
//! OpenVINO type, header, or dependency crosses the backend boundary, so this encoder compiles
//! and unit-tests on every platform.
//!
//! Emission order is load-bearing: `Parameter` layers are written first in input-slot order and
//! `Result` layers last in output-slot order, because the IR frontend builds its parameter and
//! result vectors in encounter order and those indices drive the runtime's
//! `set_input/output_tensor_by_index` calls. Every layer, tensor, and attribute string written
//! into the document is generated from fixed tables and integer IDs — TOSA-declared names never
//! reach the XML, so no escaping surface exists.

// Builds without a detected OpenVINO runtime type-check and unit-test this backend-local
// encoder, but only the native runtime modules call it from `load_program`.
#![cfg_attr(not(va_openvino), allow(dead_code))]

use std::fmt;
use std::fmt::Write as _;

use virtio_accel_tosa::{
    AnalysisError, AnalyzedValueKind, DType, Error as ParseError, ExtensionSet, Level,
    NanPropagationMode, Op, OpAttributes, ProfileSet, Target, TosaAnalysis, ValueId, Version,
    parse,
};

/// TOSA target currently lowered by the OpenVINO backend.
pub const OPENVINO_TOSA_TARGET: Target = Target::new(
    Version::TOSA_1_0,
    ProfileSet::FLOATING_POINT,
    Level::Level8K,
    ExtensionSet::NONE,
);

/// TOSA integer-profile target lowered with exact INT8 storage and INT32 arithmetic.
pub const OPENVINO_TOSA_INTEGER_TARGET: Target = Target::new(
    Version::TOSA_1_0,
    ProfileSet::INTEGER,
    Level::Level8K,
    ExtensionSet::NONE,
);

/// Weights-blob entries are aligned generously so every element type loads aligned.
const WEIGHTS_ALIGNMENT: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LoweringError {
    Parse(ParseError),
    Analysis(AnalysisError),
    UnsupportedGraph,
    UnsupportedType(DType),
    UnsupportedOperator(Op),
    InvalidConstant,
    ResourceLimit,
}

impl fmt::Display for LoweringError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for LoweringError {}

/// Element kinds this lowering writes into IR documents.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum OvElement {
    F32,
    F16,
    I8,
    I32,
    I64,
    Bool,
}

impl OvElement {
    /// The `element_type` attribute spelling.
    pub(crate) const fn element_type(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::I8 => "i8",
            Self::I32 => "i32",
            Self::I64 => "i64",
            Self::Bool => "boolean",
        }
    }

    /// The port `precision` attribute spelling.
    const fn precision(self) -> &'static str {
        match self {
            Self::F32 => "FP32",
            Self::F16 => "FP16",
            Self::I8 => "I8",
            Self::I32 => "I32",
            Self::I64 => "I64",
            Self::Bool => "BOOL",
        }
    }

    /// Storage bytes per scalar.
    pub(crate) const fn scalar_bytes(self) -> u32 {
        match self {
            Self::F32 | Self::I32 => 4,
            Self::F16 => 2,
            Self::I8 => 1,
            Self::I64 => 8,
            Self::Bool => 1,
        }
    }

    fn for_dtype(dtype: DType) -> Result<Self, LoweringError> {
        match dtype {
            DType::FP32 => Ok(Self::F32),
            DType::FP16 => Ok(Self::F16),
            DType::INT8 => Ok(Self::I8),
            DType::INT32 => Ok(Self::I32),
            DType::BOOL => Ok(Self::Bool),
            _ => Err(LoweringError::UnsupportedType(dtype)),
        }
    }
}

/// A model-boundary element type; I64 stays internal to the graph.
fn boundary_element(dtype: DType) -> Result<OvElement, LoweringError> {
    match dtype {
        DType::FP16 => Ok(OvElement::F16),
        DType::FP32 => Ok(OvElement::F32),
        DType::INT8 => Ok(OvElement::I8),
        DType::INT32 => Ok(OvElement::I32),
        DType::BOOL => Ok(OvElement::Bool),
        _ => Err(LoweringError::UnsupportedType(dtype)),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LoweredFeatureRole {
    Input,
    Output,
}

/// One model-boundary tensor and the binding slot that carries it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct LoweredFeature {
    pub slot: u32,
    pub role: LoweredFeatureRole,
    /// Index within the model's inputs or outputs, for `*_tensor_by_index` access.
    pub io_index: u32,
    pub element: OvElement,
    pub dims: Vec<i64>,
    /// Exact tensor bytes: the required length of a binding over this slot.
    pub byte_len: u64,
}

#[derive(Clone, Debug)]
pub(crate) struct LoweredModel {
    pub xml: Vec<u8>,
    pub weights: Vec<u8>,
    pub features: Vec<LoweredFeature>,
}

/// Whether the initial OpenVINO lowering tier can lower `op` for supported types and attributes.
pub const fn supports_tosa_operator(op: Op) -> bool {
    matches!(
        op,
        Op::ARGMAX
            | Op::MATMUL
            | Op::MAX_POOL2D
            | Op::CLAMP
            | Op::ERF
            | Op::SIGMOID
            | Op::TANH
            | Op::ADD
            | Op::LOGICAL_AND
            | Op::LOGICAL_OR
            | Op::LOGICAL_XOR
            | Op::MAXIMUM
            | Op::MINIMUM
            | Op::MUL
            | Op::POW
            | Op::SUB
            | Op::ABS
            | Op::CEIL
            | Op::COS
            | Op::EXP
            | Op::FLOOR
            | Op::LOG
            | Op::LOGICAL_NOT
            | Op::NEGATE
            | Op::RECIPROCAL
            | Op::RSQRT
            | Op::SIN
            | Op::SELECT
            | Op::EQUAL
            | Op::GREATER
            | Op::GREATER_EQUAL
            | Op::REDUCE_MAX
            | Op::REDUCE_MIN
            | Op::REDUCE_PRODUCT
            | Op::REDUCE_SUM
            | Op::CONCAT
            | Op::RESHAPE
            | Op::REVERSE
            | Op::TRANSPOSE
            | Op::CONST
            | Op::CONST_SHAPE
            | Op::IDENTITY
    )
}

/// Whether this lowering can expose `dtype` at an OpenVINO model boundary.
///
/// Operator-specific and target-specific validation still applies. INT8 is currently limited to
/// exact identity and zero-point-aware MATMUL; other integer operators are rejected instead of
/// silently dequantized.
pub const fn supports_tosa_dtype(dtype: DType) -> bool {
    matches!(
        dtype,
        DType::FP16 | DType::FP32 | DType::INT8 | DType::INT32 | DType::BOOL
    )
}

/// A produced tensor inside the document: one output port of one layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PortRef {
    layer: u32,
    port: u32,
}

/// Incremental IR v11 document builder.
struct IrBuilder {
    layers: String,
    edges: String,
    weights: Vec<u8>,
    next_layer: u32,
    /// The producing port of each analyzed value, indexed by dense `ValueId`.
    sources: Vec<Option<PortRef>>,
}

impl IrBuilder {
    fn new(values: usize) -> Result<Self, LoweringError> {
        let mut sources = Vec::new();
        sources
            .try_reserve_exact(values)
            .map_err(|_| LoweringError::ResourceLimit)?;
        sources.resize(values, None);
        Ok(Self {
            layers: String::new(),
            edges: String::new(),
            weights: Vec::new(),
            next_layer: 0,
            sources,
        })
    }

    fn source(&self, value: ValueId) -> Result<PortRef, LoweringError> {
        self.sources[value.get() as usize].ok_or(LoweringError::UnsupportedGraph)
    }

    fn set_source(&mut self, value: ValueId, port: PortRef) {
        self.sources[value.get() as usize] = Some(port);
    }

    /// Emit one layer; input ports take `0..inputs.len()` and output ports follow.
    ///
    /// `data` is a preformatted attribute list (`key="value" ...`) built exclusively from fixed
    /// tables and integer formatting.
    fn emit_layer(
        &mut self,
        kind: &str,
        version: &str,
        name: &str,
        data: &str,
        inputs: &[(PortRef, &[i64])],
        outputs: &[(OvElement, &[i64])],
    ) -> Vec<PortRef> {
        let layer = self.next_layer;
        self.next_layer += 1;
        let _ = write!(
            self.layers,
            "<layer id=\"{layer}\" name=\"{name}\" type=\"{kind}\" version=\"{version}\">"
        );
        if !data.is_empty() {
            let _ = write!(self.layers, "<data {data}/>");
        }
        if !inputs.is_empty() {
            self.layers.push_str("<input>");
            for (port, (source, dims)) in inputs.iter().enumerate() {
                let port = port as u32;
                let _ = write!(self.layers, "<port id=\"{port}\">");
                Self::write_dims(&mut self.layers, dims);
                self.layers.push_str("</port>");
                let _ = write!(
                    self.edges,
                    "<edge from-layer=\"{}\" from-port=\"{}\" to-layer=\"{layer}\" to-port=\"{port}\"/>",
                    source.layer, source.port
                );
            }
            self.layers.push_str("</input>");
        }
        let mut ports = Vec::with_capacity(outputs.len());
        if !outputs.is_empty() {
            self.layers.push_str("<output>");
            for (index, (element, dims)) in outputs.iter().enumerate() {
                let port = (inputs.len() + index) as u32;
                let _ = write!(
                    self.layers,
                    "<port id=\"{port}\" precision=\"{}\">",
                    element.precision()
                );
                Self::write_dims(&mut self.layers, dims);
                self.layers.push_str("</port>");
                ports.push(PortRef { layer, port });
            }
            self.layers.push_str("</output>");
        }
        self.layers.push_str("</layer>");
        ports
    }

    fn write_dims(target: &mut String, dims: &[i64]) {
        for dim in dims {
            let _ = write!(target, "<dim>{dim}</dim>");
        }
    }

    /// Append raw constant bytes to the weights blob and emit its `Const` layer.
    fn emit_const(
        &mut self,
        name: &str,
        element: OvElement,
        dims: &[i64],
        bytes: &[u8],
    ) -> Result<PortRef, LoweringError> {
        let expected = element_byte_len(element, dims)?;
        if expected != bytes.len() as u64 {
            return Err(LoweringError::InvalidConstant);
        }
        let padding = self.weights.len().next_multiple_of(WEIGHTS_ALIGNMENT) - self.weights.len();
        self.weights
            .try_reserve_exact(padding + bytes.len())
            .map_err(|_| LoweringError::ResourceLimit)?;
        self.weights.resize(self.weights.len() + padding, 0);
        let offset = self.weights.len();
        self.weights.extend_from_slice(bytes);
        let mut shape = String::new();
        for (index, dim) in dims.iter().enumerate() {
            if index > 0 {
                shape.push(',');
            }
            let _ = write!(shape, "{dim}");
        }
        let data = format!(
            "element_type=\"{}\" shape=\"{shape}\" offset=\"{offset}\" size=\"{}\"",
            element.element_type(),
            bytes.len()
        );
        let ports = self.emit_layer("Const", "opset1", name, &data, &[], &[(element, dims)]);
        Ok(ports[0])
    }

    /// Emit an `i64` vector (or rank-0 scalar) constant used as an operator parameter input.
    fn emit_i64_const(
        &mut self,
        name: &str,
        values: &[i64],
        rank0: bool,
    ) -> Result<PortRef, LoweringError> {
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(values.len() * 8)
            .map_err(|_| LoweringError::ResourceLimit)?;
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        let dims = [values.len() as i64];
        let dims: &[i64] = if rank0 { &[] } else { &dims };
        self.emit_const(name, OvElement::I64, dims, &bytes)
    }

    fn finish(self) -> LoweredModel {
        let mut document = String::with_capacity(
            self.layers.len()
                + self.edges.len()
                + "<net name=\"tosa\" version=\"11\"><layers></layers><edges></edges></net>".len()
                + 24,
        );
        document.push_str("<?xml version=\"1.0\"?>");
        document.push_str("<net name=\"tosa\" version=\"11\">");
        document.push_str("<layers>");
        document.push_str(&self.layers);
        document.push_str("</layers>");
        document.push_str("<edges>");
        document.push_str(&self.edges);
        document.push_str("</edges>");
        document.push_str("</net>");
        LoweredModel {
            xml: document.into_bytes(),
            weights: self.weights,
            features: Vec::new(),
        }
    }
}

fn element_byte_len(element: OvElement, dims: &[i64]) -> Result<u64, LoweringError> {
    let mut total = u64::from(element.scalar_bytes());
    for dim in dims {
        let dim = u64::try_from(*dim).map_err(|_| LoweringError::UnsupportedGraph)?;
        total = total
            .checked_mul(dim)
            .ok_or(LoweringError::UnsupportedGraph)?;
    }
    Ok(total)
}

pub(crate) fn lower_tosa(bytes: &[u8], target: Target) -> Result<LoweredModel, LoweringError> {
    if target != OPENVINO_TOSA_TARGET && target != OPENVINO_TOSA_INTEGER_TARGET {
        return Err(LoweringError::UnsupportedGraph);
    }
    let model = parse(bytes).map_err(LoweringError::Parse)?;
    let analysis = model.analyze_for(target).map_err(LoweringError::Analysis)?;
    validate_target_types(&analysis, target)?;
    if analysis.regions().len() != 1
        || analysis.blocks().len() != 1
        || !analysis.conditions().is_empty()
    {
        return Err(LoweringError::UnsupportedGraph);
    }
    let block = analysis.blocks()[0].id();
    let inputs = analysis.block_inputs(block);
    let outputs = analysis.block_outputs(block);
    if inputs.is_empty()
        || outputs.is_empty()
        || inputs.iter().any(|input| outputs.contains(input))
        || inputs.len().checked_add(outputs.len()).is_none()
    {
        return Err(LoweringError::UnsupportedGraph);
    }

    let mut builder = IrBuilder::new(analysis.values().len())?;
    let mut features = Vec::new();
    features
        .try_reserve_exact(inputs.len() + outputs.len())
        .map_err(|_| LoweringError::ResourceLimit)?;

    for (index, value) in inputs.iter().copied().enumerate() {
        let tensor = tensor(&analysis, value)?;
        let element = boundary_element(tensor.dtype())?;
        let dims = feature_dims(tensor)?;
        let byte_len = element_byte_len(element, &dims)?;
        let name = format!("input_{index}");
        let mut shape = String::new();
        for (position, dim) in dims.iter().enumerate() {
            if position > 0 {
                shape.push(',');
            }
            let _ = write!(shape, "{dim}");
        }
        let data = format!(
            "shape=\"{shape}\" element_type=\"{}\"",
            element.element_type()
        );
        let ports = builder.emit_layer(
            "Parameter",
            "opset1",
            &name,
            &data,
            &[],
            &[(element, dims.as_slice())],
        );
        builder.set_source(value, ports[0]);
        features.push(LoweredFeature {
            slot: u32::try_from(index).map_err(|_| LoweringError::ResourceLimit)?,
            role: LoweredFeatureRole::Input,
            io_index: u32::try_from(index).map_err(|_| LoweringError::ResourceLimit)?,
            element,
            dims,
            byte_len,
        });
    }

    for operator in analysis.execution_order(block) {
        encode_operator(&mut builder, &analysis, *operator)?;
    }

    for (index, value) in outputs.iter().copied().enumerate() {
        let tensor = tensor(&analysis, value)?;
        let element = boundary_element(tensor.dtype())?;
        let dims = feature_dims(tensor)?;
        let byte_len = element_byte_len(element, &dims)?;
        let source = builder.source(value)?;
        let name = format!("output_{index}");
        builder.emit_layer(
            "Result",
            "opset1",
            &name,
            "",
            &[(source, dims.as_slice())],
            &[],
        );
        features.push(LoweredFeature {
            slot: u32::try_from(inputs.len() + index).map_err(|_| LoweringError::ResourceLimit)?,
            role: LoweredFeatureRole::Output,
            io_index: u32::try_from(index).map_err(|_| LoweringError::ResourceLimit)?,
            element,
            dims,
            byte_len,
        });
    }

    let mut lowered = builder.finish();
    lowered.features = features;
    Ok(lowered)
}

/// Keep the artifact's declared TOSA profile load-bearing after individual operators gain support
/// for more than one dtype. TOSA profile analysis validates operator legality, but a simple op such
/// as IDENTITY can be legal for several profiles; the provider target still selects exactly one
/// lowering tier and must not be used to smuggle a differently typed graph into it.
fn validate_target_types(analysis: &TosaAnalysis<'_>, target: Target) -> Result<(), LoweringError> {
    for value in analysis.values() {
        let AnalyzedValueKind::Tensor(tensor) = value.kind() else {
            continue;
        };
        let dtype = tensor.dtype();
        let mismatched = if target == OPENVINO_TOSA_TARGET {
            dtype == DType::INT8 && !constant_is_parameter_only(analysis, value.id())
        } else {
            matches!(dtype, DType::FP16 | DType::FP32)
        };
        if mismatched {
            return Err(LoweringError::UnsupportedType(dtype));
        }
    }
    Ok(())
}

fn tensor<'a>(
    analysis: &'a TosaAnalysis<'a>,
    value: ValueId,
) -> Result<virtio_accel_tosa::Tensor<'a>, LoweringError> {
    match analysis.value(value).kind() {
        AnalyzedValueKind::Tensor(tensor) => Ok(tensor),
        AnalyzedValueKind::Shape(_) => Err(LoweringError::UnsupportedGraph),
    }
}

/// Static, positive dimensions of a tensor; empty (rank-0) is allowed only inside the graph.
fn static_dims(tensor: virtio_accel_tosa::Tensor<'_>) -> Result<Vec<i64>, LoweringError> {
    tensor.rank().ok_or(LoweringError::UnsupportedGraph)?;
    let dims = tensor.dimensions().map(i64::from).collect::<Vec<_>>();
    if dims.iter().any(|dimension| *dimension <= 0) {
        return Err(LoweringError::UnsupportedGraph);
    }
    Ok(dims)
}

/// Model-boundary tensors additionally reject rank-0.
fn feature_dims(tensor: virtio_accel_tosa::Tensor<'_>) -> Result<Vec<i64>, LoweringError> {
    let dims = static_dims(tensor)?;
    if dims.is_empty() {
        return Err(LoweringError::UnsupportedGraph);
    }
    Ok(dims)
}

fn value_port_dims(
    builder: &IrBuilder,
    analysis: &TosaAnalysis<'_>,
    value: ValueId,
) -> Result<(PortRef, Vec<i64>), LoweringError> {
    Ok((
        builder.source(value)?,
        static_dims(tensor(analysis, value)?)?,
    ))
}

fn encode_operator(
    builder: &mut IrBuilder,
    analysis: &TosaAnalysis<'_>,
    operator_id: virtio_accel_tosa::OperatorId,
) -> Result<(), LoweringError> {
    let operator = analysis.operator(operator_id);
    let op = operator.op();
    if !supports_tosa_operator(op) {
        return Err(LoweringError::UnsupportedOperator(op));
    }
    let all_inputs = analysis.operator_inputs(operator_id);
    let outputs = analysis.operator_outputs(operator_id);
    if op == Op::MATMUL && tensor(analysis, all_inputs[0])?.dtype() == DType::INT8 {
        return encode_int8_matmul(builder, analysis, operator_id, all_inputs, outputs);
    }
    let inputs = match op {
        Op::MATMUL => {
            for zero_point in &all_inputs[2..4] {
                let bytes = analysis
                    .serialized_constant(*zero_point)
                    .ok_or(LoweringError::UnsupportedGraph)?;
                if !serialized_float_is_zero(tensor(analysis, *zero_point)?.dtype(), bytes) {
                    return Err(LoweringError::UnsupportedGraph);
                }
            }
            &all_inputs[..2]
        }
        Op::MUL => {
            let shift = analysis
                .serialized_constant(all_inputs[2])
                .ok_or(LoweringError::UnsupportedGraph)?;
            if shift.iter().any(|byte| *byte != 0) {
                return Err(LoweringError::UnsupportedGraph);
            }
            &all_inputs[..2]
        }
        Op::NEGATE => {
            for zero_point in &all_inputs[1..3] {
                let bytes = analysis
                    .serialized_constant(*zero_point)
                    .ok_or(LoweringError::UnsupportedGraph)?;
                if bytes.iter().any(|byte| *byte != 0) {
                    return Err(LoweringError::UnsupportedGraph);
                }
            }
            &all_inputs[..1]
        }
        Op::RESHAPE => {
            analysis
                .serialized_constant(all_inputs[1])
                .ok_or(LoweringError::UnsupportedGraph)?;
            &all_inputs[..1]
        }
        _ => all_inputs,
    };

    // Compile-time constants consumed only by a layer parameter are deliberately absent from the
    // OpenVINO graph. They have already been validated by TOSA analysis.
    if op == Op::CONST_SHAPE {
        return Ok(());
    }
    if op == Op::CONST {
        let output = outputs[0];
        if constant_is_parameter_only(analysis, output) {
            return Ok(());
        }
        return encode_tosa_constant(builder, analysis, operator_id, output);
    }

    validate_operator_types(analysis, op, inputs, outputs)?;
    match operator.source().attributes() {
        OpAttributes::Maximum { nan_mode } | OpAttributes::Minimum { nan_mode } => {
            require_propagating_nan(nan_mode)?;
        }
        _ => {}
    }

    let stem = format!("tosa_{}_{}", operator_id.get(), op.name().unwrap_or("op"));
    if op == Op::MAX_POOL2D {
        return encode_max_pool2d(builder, analysis, operator_id, inputs, outputs, &stem);
    }
    if op == Op::ARGMAX {
        return encode_argmax(builder, analysis, operator_id, inputs, outputs, &stem);
    }
    if matches!(op, Op::RECIPROCAL | Op::RSQRT) {
        return encode_negative_power(builder, analysis, op, inputs, outputs, &stem);
    }

    let output_tensor = tensor(analysis, outputs[0])?;
    let output_element = OvElement::for_dtype(output_tensor.dtype())?;
    let output_dims = static_dims(output_tensor)?;

    // (layer type, extra `<data .../>` attributes)
    const NUMPY: &str = "auto_broadcast=\"numpy\"";
    let (kind, data) = match op {
        // A same-type Convert materializes the copy. Identity must not become a bare
        // parameter-to-result edge: the runtime accepts that document but completes without
        // writing a caller-bound output tensor (observed with the 2026.3 CPU plugin), which the
        // output-pointer honesty check cannot detect.
        Op::IDENTITY => (
            "Convert",
            format!("destination_type=\"{}\"", output_element.element_type()),
        ),
        Op::ADD => ("Add", NUMPY.to_owned()),
        Op::SUB => ("Subtract", NUMPY.to_owned()),
        Op::MUL => ("Multiply", NUMPY.to_owned()),
        Op::POW => ("Power", NUMPY.to_owned()),
        Op::MAXIMUM => ("Maximum", NUMPY.to_owned()),
        Op::MINIMUM => ("Minimum", NUMPY.to_owned()),
        Op::EQUAL => ("Equal", NUMPY.to_owned()),
        Op::GREATER => ("Greater", NUMPY.to_owned()),
        Op::GREATER_EQUAL => ("GreaterEqual", NUMPY.to_owned()),
        Op::LOGICAL_AND => ("LogicalAnd", NUMPY.to_owned()),
        Op::LOGICAL_OR => ("LogicalOr", NUMPY.to_owned()),
        Op::LOGICAL_XOR => ("LogicalXor", NUMPY.to_owned()),
        Op::SELECT => ("Select", NUMPY.to_owned()),
        Op::LOGICAL_NOT => ("LogicalNot", String::new()),
        Op::ABS => ("Abs", String::new()),
        Op::CEIL => ("Ceiling", String::new()),
        Op::COS => ("Cos", String::new()),
        Op::ERF => ("Erf", String::new()),
        Op::EXP => ("Exp", String::new()),
        Op::FLOOR => ("Floor", String::new()),
        Op::LOG => ("Log", String::new()),
        Op::NEGATE => ("Negative", String::new()),
        Op::SIN => ("Sin", String::new()),
        Op::SIGMOID => ("Sigmoid", String::new()),
        Op::TANH => ("Tanh", String::new()),
        Op::MATMUL => (
            "MatMul",
            "transpose_a=\"false\" transpose_b=\"false\"".to_owned(),
        ),
        Op::CLAMP => {
            let OpAttributes::Clamp {
                min_val,
                max_val,
                nan_mode,
            } = operator.source().attributes()
            else {
                return Err(LoweringError::UnsupportedGraph);
            };
            require_propagating_nan(nan_mode)?;
            let dtype = tensor(analysis, inputs[0])?.dtype();
            let min = attr_float(decode_float(dtype, min_val)?)?;
            let max = attr_float(decode_float(dtype, max_val)?)?;
            ("Clamp", format!("min=\"{min}\" max=\"{max}\""))
        }
        Op::CONCAT => {
            let OpAttributes::Concat { axis } = operator.source().attributes() else {
                return Err(LoweringError::UnsupportedGraph);
            };
            ("Concat", format!("axis=\"{axis}\""))
        }
        Op::REDUCE_MAX | Op::REDUCE_MIN | Op::REDUCE_PRODUCT | Op::REDUCE_SUM => {
            let axis = match operator.source().attributes() {
                OpAttributes::ReduceMax { axis, nan_mode }
                | OpAttributes::ReduceMin { axis, nan_mode } => {
                    require_propagating_nan(nan_mode)?;
                    axis
                }
                OpAttributes::ReduceProduct { axis } | OpAttributes::ReduceSum { axis } => axis,
                _ => return Err(LoweringError::UnsupportedGraph),
            };
            let axes =
                builder.emit_i64_const(&format!("{stem}_axes"), &[i64::from(axis)], false)?;
            let (source, dims) = value_port_dims(builder, analysis, inputs[0])?;
            let kind = match op {
                Op::REDUCE_MAX => "ReduceMax",
                Op::REDUCE_MIN => "ReduceMin",
                Op::REDUCE_SUM => "ReduceSum",
                _ => "ReduceProd",
            };
            let ports = builder.emit_layer(
                kind,
                "opset1",
                &stem,
                "keep_dims=\"true\"",
                &[(source, dims.as_slice()), (axes, &[1])],
                &[(output_element, output_dims.as_slice())],
            );
            builder.set_source(outputs[0], ports[0]);
            return Ok(());
        }
        Op::RESHAPE => {
            let shape = builder.emit_i64_const(&format!("{stem}_shape"), &output_dims, false)?;
            let (source, dims) = value_port_dims(builder, analysis, inputs[0])?;
            let ports = builder.emit_layer(
                "Reshape",
                "opset1",
                &stem,
                "special_zero=\"false\"",
                &[
                    (source, dims.as_slice()),
                    (shape, &[output_dims.len() as i64]),
                ],
                &[(output_element, output_dims.as_slice())],
            );
            builder.set_source(outputs[0], ports[0]);
            return Ok(());
        }
        Op::REVERSE => {
            let OpAttributes::Reverse { axis } = operator.source().attributes() else {
                return Err(LoweringError::UnsupportedGraph);
            };
            let axes =
                builder.emit_i64_const(&format!("{stem}_axes"), &[i64::from(axis)], false)?;
            let (source, dims) = value_port_dims(builder, analysis, inputs[0])?;
            let ports = builder.emit_layer(
                "Reverse",
                "opset1",
                &stem,
                "mode=\"index\"",
                &[(source, dims.as_slice()), (axes, &[1])],
                &[(output_element, output_dims.as_slice())],
            );
            builder.set_source(outputs[0], ports[0]);
            return Ok(());
        }
        Op::TRANSPOSE => {
            let OpAttributes::Transpose { perms } = operator.source().attributes() else {
                return Err(LoweringError::UnsupportedGraph);
            };
            let perms = perms.iter().map(i64::from).collect::<Vec<_>>();
            let order = builder.emit_i64_const(&format!("{stem}_perms"), &perms, false)?;
            let (source, dims) = value_port_dims(builder, analysis, inputs[0])?;
            let ports = builder.emit_layer(
                "Transpose",
                "opset1",
                &stem,
                "",
                &[(source, dims.as_slice()), (order, &[perms.len() as i64])],
                &[(output_element, output_dims.as_slice())],
            );
            builder.set_source(outputs[0], ports[0]);
            return Ok(());
        }
        _ => return Err(LoweringError::UnsupportedOperator(op)),
    };

    let mut connected = Vec::with_capacity(inputs.len());
    for value in inputs {
        let (source, dims) = value_port_dims(builder, analysis, *value)?;
        connected.push((source, dims));
    }
    let connected = connected
        .iter()
        .map(|(source, dims)| (*source, dims.as_slice()))
        .collect::<Vec<_>>();
    let ports = builder.emit_layer(
        kind,
        "opset1",
        &stem,
        &data,
        &connected,
        &[(output_element, output_dims.as_slice())],
    );
    builder.set_source(outputs[0], ports[0]);
    Ok(())
}

fn encode_max_pool2d(
    builder: &mut IrBuilder,
    analysis: &TosaAnalysis<'_>,
    operator_id: virtio_accel_tosa::OperatorId,
    inputs: &[ValueId],
    outputs: &[ValueId],
    stem: &str,
) -> Result<(), LoweringError> {
    let OpAttributes::MaxPool2d {
        kernel,
        stride,
        pad,
        nan_mode,
    } = analysis.operator(operator_id).source().attributes()
    else {
        return Err(LoweringError::UnsupportedGraph);
    };
    require_propagating_nan(nan_mode)?;
    let kernel = kernel.iter().collect::<Vec<_>>();
    let stride = stride.iter().collect::<Vec<_>>();
    let pad = pad.iter().collect::<Vec<_>>();
    if kernel.len() != 2
        || stride.len() != 2
        || pad.len() != 4
        || kernel.iter().chain(&stride).any(|value| *value <= 0)
        || pad.iter().any(|value| *value != 0)
    {
        return Err(LoweringError::UnsupportedGraph);
    }

    let input_tensor = tensor(analysis, inputs[0])?;
    let element = OvElement::for_dtype(input_tensor.dtype())?;
    let nhwc_in = static_dims(input_tensor)?;
    let nhwc_out = static_dims(tensor(analysis, outputs[0])?)?;
    if nhwc_in.len() != 4 || nhwc_out.len() != 4 {
        return Err(LoweringError::UnsupportedGraph);
    }
    let permute = |dims: &[i64], perms: [usize; 4]| perms.map(|axis| dims[axis]);
    let nchw_in = permute(&nhwc_in, [0, 3, 1, 2]);
    let nchw_out = permute(&nhwc_out, [0, 3, 1, 2]);

    let to_nchw = builder.emit_i64_const(&format!("{stem}_nchw_perms"), &[0, 3, 1, 2], false)?;
    let source = builder.source(inputs[0])?;
    let nchw_input = builder.emit_layer(
        "Transpose",
        "opset1",
        &format!("{stem}_to_nchw"),
        "",
        &[(source, nhwc_in.as_slice()), (to_nchw, &[4])],
        &[(element, nchw_in.as_slice())],
    )[0];

    let data = format!(
        "strides=\"{},{}\" kernel=\"{},{}\" pads_begin=\"0,0\" pads_end=\"0,0\" \
         rounding_type=\"floor\" auto_pad=\"explicit\"",
        stride[0], stride[1], kernel[0], kernel[1]
    );
    let pooled = builder.emit_layer(
        "MaxPool",
        "opset1",
        stem,
        &data,
        &[(nchw_input, nchw_in.as_slice())],
        &[(element, nchw_out.as_slice())],
    )[0];

    let to_nhwc = builder.emit_i64_const(&format!("{stem}_nhwc_perms"), &[0, 2, 3, 1], false)?;
    let restored = builder.emit_layer(
        "Transpose",
        "opset1",
        &format!("{stem}_to_nhwc"),
        "",
        &[(pooled, nchw_out.as_slice()), (to_nhwc, &[4])],
        &[(element, nhwc_out.as_slice())],
    )[0];
    builder.set_source(outputs[0], restored);
    Ok(())
}

/// Lower TOSA INT8 MATMUL without relying on provider-specific implicit quantization.
///
/// TOSA requires `(a - a_zp) * (b - b_zp)` with exact INT32 accumulation. OpenVINO's ordinary
/// MatMul does not carry TOSA zero points, so both operands are widened and adjusted explicitly.
/// This preserves the integer-profile contract on every plugin; a plugin may fuse the pattern when
/// it can prove an equivalent optimized kernel.
fn encode_int8_matmul(
    builder: &mut IrBuilder,
    analysis: &TosaAnalysis<'_>,
    operator_id: virtio_accel_tosa::OperatorId,
    inputs: &[ValueId],
    outputs: &[ValueId],
) -> Result<(), LoweringError> {
    if inputs.len() != 4 || outputs.len() != 1 {
        return Err(LoweringError::UnsupportedGraph);
    }
    let output = tensor(analysis, outputs[0])?;
    if tensor(analysis, inputs[0])?.dtype() != DType::INT8
        || tensor(analysis, inputs[1])?.dtype() != DType::INT8
        || output.dtype() != DType::INT32
    {
        return Err(LoweringError::UnsupportedGraph);
    }
    let read_zero_point = |value| {
        let bytes = analysis
            .serialized_constant(value)
            .ok_or(LoweringError::UnsupportedGraph)?;
        if bytes.len() != 1 || tensor(analysis, value)?.dtype() != DType::INT8 {
            return Err(LoweringError::UnsupportedGraph);
        }
        Ok(i32::from(bytes[0] as i8))
    };
    let zero_points = [read_zero_point(inputs[2])?, read_zero_point(inputs[3])?];
    let stem = format!("tosa_{}_matmul", operator_id.get());
    let mut adjusted = Vec::with_capacity(2);
    for index in 0..2 {
        let dims = static_dims(tensor(analysis, inputs[index])?)?;
        let source = builder.source(inputs[index])?;
        let widened = builder.emit_layer(
            "Convert",
            "opset1",
            &format!("{stem}_widen_{index}"),
            "destination_type=\"i32\"",
            &[(source, dims.as_slice())],
            &[(OvElement::I32, dims.as_slice())],
        )[0];
        let zero_point = builder.emit_const(
            &format!("{stem}_zero_point_{index}"),
            OvElement::I32,
            &[],
            &zero_points[index].to_le_bytes(),
        )?;
        let shifted = builder.emit_layer(
            "Subtract",
            "opset1",
            &format!("{stem}_shift_{index}"),
            "auto_broadcast=\"numpy\"",
            &[(widened, dims.as_slice()), (zero_point, &[])],
            &[(OvElement::I32, dims.as_slice())],
        )[0];
        adjusted.push((shifted, dims));
    }
    let output_dims = static_dims(output)?;
    let result = builder.emit_layer(
        "MatMul",
        "opset1",
        &stem,
        "transpose_a=\"false\" transpose_b=\"false\"",
        &[
            (adjusted[0].0, adjusted[0].1.as_slice()),
            (adjusted[1].0, adjusted[1].1.as_slice()),
        ],
        &[(OvElement::I32, output_dims.as_slice())],
    )[0];
    builder.set_source(outputs[0], result);
    Ok(())
}

/// TOSA `ARGMAX` lowers to `TopK` (k = 1, stable lowest-index ties) plus a `Squeeze` that drops
/// the kept axis; only the TopK indices output is consumed and the values port dangles, which
/// the runtime accepts (pinned by a native unit test).
fn encode_argmax(
    builder: &mut IrBuilder,
    analysis: &TosaAnalysis<'_>,
    operator_id: virtio_accel_tosa::OperatorId,
    inputs: &[ValueId],
    outputs: &[ValueId],
    stem: &str,
) -> Result<(), LoweringError> {
    let OpAttributes::ArgMax { axis, nan_mode } =
        analysis.operator(operator_id).source().attributes()
    else {
        return Err(LoweringError::UnsupportedGraph);
    };
    require_propagating_nan(nan_mode)?;
    let input_tensor = tensor(analysis, inputs[0])?;
    let element = OvElement::for_dtype(input_tensor.dtype())?;
    let input_dims = static_dims(input_tensor)?;
    let axis_index = usize::try_from(axis).map_err(|_| LoweringError::UnsupportedGraph)?;
    if axis_index >= input_dims.len() {
        return Err(LoweringError::UnsupportedGraph);
    }
    let mut kept_dims = input_dims.clone();
    kept_dims[axis_index] = 1;

    let k = builder.emit_i64_const(&format!("{stem}_k"), &[1], true)?;
    let source = builder.source(inputs[0])?;
    let data = format!(
        "axis=\"{axis}\" mode=\"max\" sort=\"value\" stable=\"true\" index_element_type=\"i32\""
    );
    let topk = builder.emit_layer(
        "TopK",
        "opset11",
        &format!("{stem}_topk"),
        &data,
        &[(source, input_dims.as_slice()), (k, &[])],
        &[
            (element, kept_dims.as_slice()),
            (OvElement::I32, kept_dims.as_slice()),
        ],
    );
    let indices = topk[1];

    let output_dims = static_dims(tensor(analysis, outputs[0])?)?;
    let axes = builder.emit_i64_const(&format!("{stem}_axes"), &[i64::from(axis)], false)?;
    let squeezed = builder.emit_layer(
        "Squeeze",
        "opset1",
        &format!("{stem}_squeeze"),
        "",
        &[(indices, kept_dims.as_slice()), (axes, &[1])],
        &[(OvElement::I32, output_dims.as_slice())],
    )[0];
    builder.set_source(outputs[0], squeezed);
    Ok(())
}

/// `RECIPROCAL` and `RSQRT` lower through `Power` with a `-1` exponent.
///
/// `RSQRT` deliberately becomes `Sqrt` followed by `Power(x, -1)` rather than `Power(x, -0.5)`:
/// IEEE `pow(-0.0, -0.5)` is `+inf`, while TOSA `rsqrt(-0.0)` is `-inf`. `sqrt(-0.0) = -0.0`
/// and `pow(-0.0, -1) = -inf` preserve the signed-zero edge, and negative inputs still produce
/// NaN through `Sqrt`.
fn encode_negative_power(
    builder: &mut IrBuilder,
    analysis: &TosaAnalysis<'_>,
    op: Op,
    inputs: &[ValueId],
    outputs: &[ValueId],
    stem: &str,
) -> Result<(), LoweringError> {
    let input_tensor = tensor(analysis, inputs[0])?;
    let element = OvElement::for_dtype(input_tensor.dtype())?;
    let dims = static_dims(input_tensor)?;
    let exponent_bytes: &[u8] = match element {
        OvElement::F32 => &(-1.0f32).to_le_bytes(),
        OvElement::F16 => &0xbc00u16.to_le_bytes(),
        _ => return Err(LoweringError::UnsupportedType(input_tensor.dtype())),
    };
    let exponent = builder.emit_const(&format!("{stem}_exponent"), element, &[], exponent_bytes)?;

    let mut source = builder.source(inputs[0])?;
    if op == Op::RSQRT {
        source = builder.emit_layer(
            "Sqrt",
            "opset1",
            &format!("{stem}_sqrt"),
            "",
            &[(source, dims.as_slice())],
            &[(element, dims.as_slice())],
        )[0];
    }
    let output_dims = static_dims(tensor(analysis, outputs[0])?)?;
    let powered = builder.emit_layer(
        "Power",
        "opset1",
        stem,
        "auto_broadcast=\"numpy\"",
        &[(source, dims.as_slice()), (exponent, &[])],
        &[(element, output_dims.as_slice())],
    )[0];
    builder.set_source(outputs[0], powered);
    Ok(())
}

fn encode_tosa_constant(
    builder: &mut IrBuilder,
    analysis: &TosaAnalysis<'_>,
    operator_id: virtio_accel_tosa::OperatorId,
    output: ValueId,
) -> Result<(), LoweringError> {
    let tensor = tensor(analysis, output)?;
    let dtype = tensor.dtype();
    if !matches!(dtype, DType::FP16 | DType::FP32 | DType::INT8 | DType::BOOL) {
        return Err(LoweringError::UnsupportedType(dtype));
    }
    let element = OvElement::for_dtype(dtype)?;
    let data = analysis
        .serialized_constant(output)
        .ok_or(LoweringError::InvalidConstant)?;
    let dims = static_dims(tensor)?;
    let name = format!("tosa_{}_const", operator_id.get());
    let port = if element == OvElement::Bool {
        // Normalize TOSA bool serialization to strict 0/1 storage bytes.
        let mut normalized = Vec::new();
        normalized
            .try_reserve_exact(data.len())
            .map_err(|_| LoweringError::ResourceLimit)?;
        normalized.extend(data.iter().map(|byte| u8::from(*byte != 0)));
        builder.emit_const(&name, element, &dims, &normalized)?
    } else {
        builder.emit_const(&name, element, &dims, data)?
    };
    builder.set_source(output, port);
    Ok(())
}

fn constant_is_parameter_only(analysis: &TosaAnalysis<'_>, value: ValueId) -> bool {
    let mut consumed = false;
    for operator in analysis.operators() {
        for (index, input) in analysis.operator_inputs(operator.id()).iter().enumerate() {
            if *input != value {
                continue;
            }
            consumed = true;
            if !matches!(
                (operator.op(), index),
                (Op::MATMUL, 2 | 3) | (Op::MUL, 2) | (Op::NEGATE, 1 | 2) | (Op::RESHAPE, 1)
            ) {
                return false;
            }
        }
    }
    consumed
}

fn validate_operator_types(
    analysis: &TosaAnalysis<'_>,
    op: Op,
    inputs: &[ValueId],
    outputs: &[ValueId],
) -> Result<(), LoweringError> {
    let require = |value, predicate: fn(DType) -> bool| {
        let dtype = tensor(analysis, value)?.dtype();
        if predicate(dtype) {
            Ok(())
        } else {
            Err(LoweringError::UnsupportedType(dtype))
        }
    };
    let is_float = |dtype| matches!(dtype, DType::FP16 | DType::FP32);
    let is_float_or_int8 = |dtype| matches!(dtype, DType::FP16 | DType::FP32 | DType::INT8);
    let is_bool = |dtype| dtype == DType::BOOL;
    let is_int32 = |dtype| dtype == DType::INT32;

    match op {
        Op::IDENTITY => {
            for value in inputs.iter().chain(outputs) {
                require(*value, is_float_or_int8)?;
            }
        }
        Op::LOGICAL_AND | Op::LOGICAL_OR | Op::LOGICAL_XOR | Op::LOGICAL_NOT => {
            for value in inputs.iter().chain(outputs) {
                require(*value, is_bool)?;
            }
        }
        Op::EQUAL | Op::GREATER | Op::GREATER_EQUAL => {
            for value in inputs {
                require(*value, is_float)?;
            }
            require(outputs[0], is_bool)?;
        }
        Op::SELECT => {
            require(inputs[0], is_bool)?;
            for value in inputs[1..].iter().chain(outputs) {
                require(*value, is_float)?;
            }
        }
        Op::ARGMAX => {
            require(inputs[0], is_float)?;
            require(outputs[0], is_int32)?;
        }
        _ => {
            for value in inputs.iter().chain(outputs) {
                require(*value, is_float)?;
            }
        }
    }
    Ok(())
}

fn require_propagating_nan(nan_mode: NanPropagationMode) -> Result<(), LoweringError> {
    if nan_mode == NanPropagationMode::PROPAGATE {
        Ok(())
    } else {
        Err(LoweringError::UnsupportedGraph)
    }
}

/// Format a float attribute value; IR attribute text has no NaN spelling this crate relies on.
fn attr_float(value: f32) -> Result<String, LoweringError> {
    if value.is_nan() {
        return Err(LoweringError::UnsupportedGraph);
    }
    Ok(format!("{value}"))
}

fn decode_float(dtype: DType, bytes: &[u8]) -> Result<f32, LoweringError> {
    match dtype {
        DType::FP16 if bytes.len() == 2 => Ok(f16_to_f32(u16::from_le_bytes(
            bytes.try_into().expect("length checked"),
        ))),
        DType::FP32 if bytes.len() == 4 => Ok(f32::from_le_bytes(bytes.try_into().unwrap())),
        _ => Err(LoweringError::UnsupportedType(dtype)),
    }
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exponent = (bits >> 10) & 0x1f;
    let fraction = u32::from(bits & 0x03ff);
    let converted = match exponent {
        0 if fraction == 0 => sign,
        0 => {
            let shift = fraction.leading_zeros() - 21;
            let normalized = fraction << shift;
            sign | ((127 - 15 - shift + 1) << 23) | ((normalized & 0x03ff) << 13)
        }
        0x1f => sign | 0x7f80_0000 | (fraction << 13),
        _ => sign | ((u32::from(exponent) + 127 - 15) << 23) | (fraction << 13),
    };
    f32::from_bits(converted)
}

fn serialized_float_is_zero(dtype: DType, bytes: &[u8]) -> bool {
    match dtype {
        DType::FP16 if bytes.len() == 2 => {
            u16::from_le_bytes(bytes.try_into().expect("length checked")) & 0x7fff == 0
        }
        DType::FP32 if bytes.len() == 4 => {
            u32::from_le_bytes(bytes.try_into().expect("length checked")) & 0x7fff_ffff == 0
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use virtio_accel_conformance::numerics::{
        HEXAGON_LOGICAL_CASES, IDENTITY_EDGES_FP16, IDENTITY_EDGES_FP32, IDENTITY_FP8E4M3,
        IDENTITY_FP8E5M2, IDENTITY_INT4, IDENTITY_INT8, MATMUL_FP16, MATMUL_FP32, MATMUL_INT8,
        MAX_POOL2D_FP16, MAX_POOL2D_FP32, MUL_FP16,
    };

    const IDENTITY_FP32_LOCAL: &[u8] = include_bytes!("../tests/data/identity-fp32-v1.0.0.tosa");

    fn xml_str(lowered: &LoweredModel) -> &str {
        core::str::from_utf8(&lowered.xml).expect("lowered documents are UTF-8")
    }

    #[test]
    fn lowers_a_verified_tosa_model_without_host_dependencies() {
        let lowered = lower_tosa(IDENTITY_FP32_LOCAL, OPENVINO_TOSA_TARGET).unwrap();
        let xml = xml_str(&lowered);
        assert!(xml.starts_with("<?xml version=\"1.0\"?><net name=\"tosa\" version=\"11\">"));
        assert_eq!(xml.matches("type=\"Parameter\"").count(), 1);
        assert_eq!(xml.matches("type=\"Result\"").count(), 1);
        assert_eq!(xml.matches("type=\"Const\"").count(), 0);
        // Identity materializes as a same-type Convert; a bare parameter-to-result edge would
        // complete without writing a caller-bound output tensor.
        assert_eq!(xml.matches("type=\"Convert\"").count(), 1);
        assert!(xml.contains("destination_type=\"f32\""));
        assert!(
            xml.contains("<edge from-layer=\"0\" from-port=\"0\" to-layer=\"1\" to-port=\"0\"/>")
        );
        assert!(
            xml.contains("<edge from-layer=\"1\" from-port=\"1\" to-layer=\"2\" to-port=\"0\"/>")
        );
        assert!(lowered.weights.is_empty());
        assert_eq!(lowered.features.len(), 2);
        assert_eq!(
            (lowered.features[0].slot, lowered.features[0].role),
            (0, LoweredFeatureRole::Input)
        );
        assert_eq!(
            (lowered.features[1].slot, lowered.features[1].role),
            (1, LoweredFeatureRole::Output)
        );
        assert_eq!(lowered.features[0].io_index, 0);
        assert_eq!(lowered.features[1].io_index, 0);
        assert_eq!(lowered.features[0].element, OvElement::F32);
        assert_eq!(lowered.features[0].byte_len, lowered.features[1].byte_len);
    }

    #[test]
    fn rejects_a_different_tosa_target_before_parsing() {
        let integer_target = Target::new(
            Version::TOSA_1_0,
            ProfileSet::INTEGER,
            Level::Level8K,
            ExtensionSet::INT4,
        );
        assert_eq!(
            lower_tosa(IDENTITY_FP32_LOCAL, integer_target).unwrap_err(),
            LoweringError::UnsupportedGraph
        );
    }

    #[test]
    fn reports_the_exact_integer_boundary_independently_of_other_low_precision_types() {
        for dtype in [DType::INT4, DType::FP8E4M3, DType::FP8E5M2] {
            assert!(!supports_tosa_dtype(dtype), "{dtype:?}");
        }
        for dtype in [
            DType::FP16,
            DType::FP32,
            DType::INT8,
            DType::INT32,
            DType::BOOL,
        ] {
            assert!(supports_tosa_dtype(dtype), "{dtype:?}");
        }
    }

    #[test]
    fn lowers_boolean_model_boundaries_as_direct_bytes() {
        let case = HEXAGON_LOGICAL_CASES
            .iter()
            .find(|case| case.name == "logical-or")
            .unwrap();
        let lowered = lower_tosa(case.artifact, OPENVINO_TOSA_TARGET).unwrap();
        let xml = xml_str(&lowered);
        assert_eq!(xml.matches("type=\"Parameter\"").count(), 2);
        assert_eq!(xml.matches("type=\"LogicalOr\"").count(), 1);
        assert_eq!(xml.matches("type=\"Result\"").count(), 1);
        assert!(xml.contains("element_type=\"boolean\""));
        assert!(xml.contains("precision=\"BOOL\""));
        assert_eq!(lowered.features.len(), 3);
        for feature in &lowered.features {
            assert_eq!(feature.element, OvElement::Bool);
            assert_eq!(feature.byte_len, 4);
        }
    }

    #[test]
    fn rejects_unimplemented_low_precision_extensions_at_the_declared_target_boundary() {
        for (case, extensions, profiles) in [
            (IDENTITY_INT4, ExtensionSet::INT4, ProfileSet::INTEGER),
            (
                IDENTITY_FP8E4M3,
                ExtensionSet::FP8E4M3,
                ProfileSet::FLOATING_POINT,
            ),
            (
                IDENTITY_FP8E5M2,
                ExtensionSet::FP8E5M2,
                ProfileSet::FLOATING_POINT,
            ),
        ] {
            let target = Target::new(Version::TOSA_1_0, profiles, Level::Level8K, extensions);
            assert_eq!(
                lower_tosa(case.artifact, target).unwrap_err(),
                LoweringError::UnsupportedGraph,
                "{}",
                case.name
            );
        }
    }

    #[test]
    fn lowers_int8_identity_with_direct_byte_boundaries() {
        let lowered = lower_tosa(IDENTITY_INT8.artifact, OPENVINO_TOSA_INTEGER_TARGET).unwrap();
        let xml = xml_str(&lowered);
        assert!(xml.contains("element_type=\"i8\""));
        assert!(xml.contains("precision=\"I8\""));
        assert!(xml.contains("destination_type=\"i8\""));
        assert_eq!(lowered.features[0].element, OvElement::I8);
        assert_eq!(lowered.features[0].byte_len, 8);
        assert_eq!(lowered.features[1].byte_len, 8);
    }

    #[test]
    fn target_profiles_cannot_admit_the_other_tiers_tensor_types() {
        assert_eq!(
            lower_tosa(IDENTITY_INT8.artifact, OPENVINO_TOSA_TARGET).unwrap_err(),
            LoweringError::UnsupportedType(DType::INT8)
        );
        assert!(matches!(
            lower_tosa(IDENTITY_FP32_LOCAL, OPENVINO_TOSA_INTEGER_TARGET),
            Err(LoweringError::Analysis(_))
        ));
    }

    #[test]
    fn floating_target_admits_only_parameter_only_int8_constants() {
        let lowered = lower_tosa(MUL_FP16.artifact, OPENVINO_TOSA_TARGET).unwrap();
        assert!(xml_str(&lowered).contains("type=\"Multiply\""));

        // A graph-visible INT8 boundary remains an integer-tier program.
        assert_eq!(
            lower_tosa(IDENTITY_INT8.artifact, OPENVINO_TOSA_TARGET).unwrap_err(),
            LoweringError::UnsupportedType(DType::INT8)
        );
    }

    #[test]
    fn lowers_int8_matmul_with_explicit_zero_point_legalization() {
        let lowered = lower_tosa(MATMUL_INT8.artifact, OPENVINO_TOSA_INTEGER_TARGET).unwrap();
        let xml = xml_str(&lowered);
        assert_eq!(xml.matches("type=\"Parameter\"").count(), 2);
        assert_eq!(xml.matches("type=\"Convert\"").count(), 2);
        assert_eq!(xml.matches("type=\"Subtract\"").count(), 2);
        assert_eq!(xml.matches("type=\"MatMul\"").count(), 1);
        assert!(xml.contains("destination_type=\"i32\""));
        assert!(xml.contains("element_type=\"i8\""));
        assert!(xml.contains("element_type=\"i32\""));
        assert_eq!(lowered.features[0].element, OvElement::I8);
        assert_eq!(lowered.features[1].element, OvElement::I8);
        assert_eq!(lowered.features[2].element, OvElement::I32);
        assert_eq!(lowered.features[2].byte_len, 16);
        let zero_points = [
            i32::from_le_bytes(lowered.weights[0..4].try_into().unwrap()),
            i32::from_le_bytes(lowered.weights[64..68].try_into().unwrap()),
        ];
        assert_eq!(zero_points, MATMUL_INT8.zero_points.map(i32::from));
    }

    #[test]
    fn lowers_batched_matmul_without_encoding_parameter_constants() {
        let lowered = lower_tosa(MATMUL_FP32.artifact, OPENVINO_TOSA_TARGET).unwrap();
        let xml = xml_str(&lowered);
        assert_eq!(xml.matches("type=\"Parameter\"").count(), 2);
        assert_eq!(xml.matches("type=\"Result\"").count(), 1);
        assert!(xml.contains("type=\"MatMul\""));
        assert!(xml.contains("transpose_a=\"false\" transpose_b=\"false\""));
        // The zero-point operands are parameter-only constants and never become Const layers.
        assert_eq!(xml.matches("type=\"Const\"").count(), 0);
        assert!(lowered.weights.is_empty());
        assert_eq!(lowered.features.len(), 3);
        assert_eq!(lowered.features[2].io_index, 0);
    }

    #[test]
    fn lowers_the_shared_fp32_edge_identity_artifact() {
        let lowered = lower_tosa(IDENTITY_EDGES_FP32.artifact, OPENVINO_TOSA_TARGET).unwrap();
        assert_eq!(lowered.features.len(), 2);
        assert_eq!(lowered.features[0].byte_len, 32);
    }

    #[test]
    fn lowers_nhwc_max_pool_through_explicit_layout_transposes() {
        let lowered = lower_tosa(MAX_POOL2D_FP32.artifact, OPENVINO_TOSA_TARGET).unwrap();
        let xml = xml_str(&lowered);
        assert_eq!(xml.matches("type=\"Transpose\"").count(), 2);
        assert_eq!(xml.matches("type=\"MaxPool\"").count(), 1);
        assert!(xml.contains(
            "strides=\"2,2\" kernel=\"2,2\" pads_begin=\"0,0\" pads_end=\"0,0\" \
             rounding_type=\"floor\" auto_pad=\"explicit\""
        ));
        // Two i64 permutation constants, each 64-byte aligned in the weights blob.
        assert_eq!(xml.matches("type=\"Const\"").count(), 2);
        assert!(xml.contains("offset=\"0\" size=\"32\""));
        assert!(xml.contains("offset=\"64\" size=\"32\""));
        assert_eq!(lowered.weights.len(), 96);
        let perms = |offset: usize| {
            (0..4)
                .map(|index| {
                    i64::from_le_bytes(
                        lowered.weights[offset + index * 8..offset + index * 8 + 8]
                            .try_into()
                            .unwrap(),
                    )
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(perms(0), [0, 3, 1, 2]);
        assert_eq!(perms(64), [0, 2, 3, 1]);
    }

    #[test]
    fn lowers_every_shared_fp16_numerical_artifact() {
        for (name, artifact) in [
            ("identity", IDENTITY_EDGES_FP16.artifact),
            ("matmul", MATMUL_FP16.artifact),
            ("max_pool2d", MAX_POOL2D_FP16.artifact),
        ] {
            let lowered = lower_tosa(artifact, OPENVINO_TOSA_TARGET).unwrap_or_else(|error| {
                panic!("fp16 {name} failed to lower: {error}");
            });
            let xml = xml_str(&lowered);
            assert!(xml.contains("element_type=\"f16\""), "{name}");
            assert!(xml.contains("precision=\"FP16\""), "{name}");
            assert_eq!(lowered.features[0].element, OvElement::F16, "{name}");
        }
    }

    #[test]
    fn fp16_constant_bytes_are_preserved_bit_exactly() {
        let mut builder = IrBuilder::new(0).unwrap();
        // A quiet NaN with a payload, negative zero, and a subnormal: raw storage must survive.
        let bytes: [u8; 6] = [0x01, 0x7e, 0x00, 0x80, 0x01, 0x00];
        let port = builder
            .emit_const("payload", OvElement::F16, &[3], &bytes)
            .unwrap();
        assert_eq!(port, PortRef { layer: 0, port: 0 });
        assert_eq!(&builder.weights[..6], &bytes);
        // A second constant lands at the next 64-byte boundary.
        builder
            .emit_const("aligned", OvElement::F32, &[1], &1.0f32.to_le_bytes())
            .unwrap();
        assert_eq!(builder.weights.len(), 68);
        assert_eq!(&builder.weights[64..68], &1.0f32.to_le_bytes());
        assert!(builder.layers.contains("offset=\"64\" size=\"4\""));
    }

    #[test]
    fn scalar_constants_are_rank_zero() {
        let mut builder = IrBuilder::new(0).unwrap();
        builder.emit_i64_const("k", &[1], true).unwrap();
        assert!(builder.layers.contains("shape=\"\""));
        assert!(builder.layers.contains("size=\"8\""));
        assert!(!builder.layers.contains("<dim>"));
        assert_eq!(
            builder
                .emit_const("wrong", OvElement::F32, &[2], &[0u8; 4])
                .unwrap_err(),
            LoweringError::InvalidConstant
        );
    }

    #[test]
    fn f16_to_f32_preserves_zero_finite_and_nan_classes() {
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert!(f16_to_f32(0x8000).is_sign_negative());
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xc000), -2.0);
        assert_eq!(f16_to_f32(0x7bff), 65504.0);
        assert_eq!(f16_to_f32(0x0001), 5.960_464_5e-8);
        assert_eq!(f16_to_f32(0x7c00), f32::INFINITY);
        assert_eq!(f16_to_f32(0xfc00), f32::NEG_INFINITY);
        assert!(f16_to_f32(0x7e01).is_nan());
    }

    #[test]
    fn attribute_floats_reject_nan_and_render_infinities() {
        assert_eq!(attr_float(1.5).unwrap(), "1.5");
        assert_eq!(attr_float(f32::INFINITY).unwrap(), "inf");
        assert_eq!(attr_float(f32::NEG_INFINITY).unwrap(), "-inf");
        assert_eq!(
            attr_float(f32::NAN).unwrap_err(),
            LoweringError::UnsupportedGraph
        );
    }

    #[test]
    fn generated_documents_never_contain_escapable_text() {
        for artifact in [
            IDENTITY_FP32_LOCAL,
            MATMUL_FP32.artifact,
            MAX_POOL2D_FP32.artifact,
        ] {
            let lowered = lower_tosa(artifact, OPENVINO_TOSA_TARGET).unwrap();
            let xml = xml_str(&lowered);
            assert!(!xml.contains('&'));
            assert!(!xml.contains('\''));
            assert_eq!(xml.matches('<').count(), xml.matches('>').count());
        }
    }
}
