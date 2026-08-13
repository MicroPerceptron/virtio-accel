//! TOSA 1.0 to Core ML neural-network lowering.
//!
//! This module intentionally owns Core ML's protobuf encoding. Portable crates expose only the
//! verified TOSA model and provider-neutral analysis; no Core ML type, path, or dependency crosses
//! the backend boundary.

// Non-macOS builds type-check and unit-test this backend-local encoder, but only the macOS runtime
// calls it from `load_program`.
#![cfg_attr(not(target_os = "macos"), allow(dead_code))]

use std::fmt;

use virtio_accel_tosa::{
    AnalysisError, AnalyzedValueKind, DType, Error as ParseError, ExtensionSet, Level,
    NanPropagationMode, Op, OpAttributes, ProfileSet, Target, TosaAnalysis, ValueId, Version,
    parse,
};

/// TOSA target currently lowered by the Core ML backend.
pub const COREML_TOSA_TARGET: Target = Target::new(
    Version::TOSA_1_0,
    ProfileSet::FLOATING_POINT,
    Level::Level8K,
    ExtensionSet::NONE,
);

// Float16 MLMultiArray model boundaries require the iOS 16 / macOS 13 format revision. The
// backend itself requires macOS 14, so all production TOSA models can use this version uniformly.
const COREML_SPECIFICATION_VERSION: u64 = 7;
const COREML_FLOAT16: u64 = 65_552;
const COREML_FLOAT32: u64 = 65_568;
const COREML_INT32: u64 = 131_104;

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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LoweredFeatureRole {
    Input,
    Output,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct LoweredFeature {
    pub slot: u32,
    pub role: LoweredFeatureRole,
    pub name: String,
}

#[derive(Clone, Debug)]
pub(crate) struct LoweredModel {
    pub bytes: Vec<u8>,
    pub features: Vec<LoweredFeature>,
}

/// Whether the initial Core ML lowering tier can lower `op` for supported types and attributes.
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

/// Whether this lowering can expose `dtype` at a Core ML model boundary.
///
/// Operator-specific validation still applies. In particular, INT32 is limited to outputs from
/// operators such as `ARGMAX`. Quantized TOSA tensor types require a future ML Program lowering;
/// the current dependency-free NeuralNetwork encoder rejects them during program admission.
pub const fn supports_tosa_dtype(dtype: DType) -> bool {
    matches!(dtype, DType::FP16 | DType::FP32 | DType::INT32)
}

pub(crate) fn lower_tosa(bytes: &[u8], target: Target) -> Result<LoweredModel, LoweringError> {
    if target != COREML_TOSA_TARGET {
        return Err(LoweringError::UnsupportedGraph);
    }
    let model = parse(bytes).map_err(LoweringError::Parse)?;
    let analysis = model.analyze_for(target).map_err(LoweringError::Analysis)?;
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

    let mut names = analysis
        .values()
        .iter()
        .map(|value| format!("v{}", value.id().get()))
        .collect::<Vec<_>>();
    let mut features = Vec::new();
    features
        .try_reserve_exact(inputs.len() + outputs.len())
        .map_err(|_| LoweringError::ResourceLimit)?;
    let mut description = Vec::new();

    for (index, value) in inputs.iter().copied().enumerate() {
        let name = format!("input_{index}");
        names[value.get() as usize] = name.clone();
        let tensor = tensor(&analysis, value)?;
        encode_feature(&mut description, 1, &name, tensor)?;
        features.push(LoweredFeature {
            slot: u32::try_from(index).map_err(|_| LoweringError::ResourceLimit)?,
            role: LoweredFeatureRole::Input,
            name,
        });
    }
    for (index, value) in outputs.iter().copied().enumerate() {
        let name = format!("output_{index}");
        names[value.get() as usize] = name.clone();
        let tensor = tensor(&analysis, value)?;
        encode_feature(&mut description, 10, &name, tensor)?;
        features.push(LoweredFeature {
            slot: u32::try_from(inputs.len() + index).map_err(|_| LoweringError::ResourceLimit)?,
            role: LoweredFeatureRole::Output,
            name,
        });
    }

    let mut network = Vec::new();
    for operator in analysis.execution_order(block) {
        encode_operator(&mut network, &analysis, *operator, &names)?;
    }
    // Exact rank mapping is mandatory for the general-ND layers used by this lowering.
    field_varint(&mut network, 5, 1);

    let mut encoded = Vec::new();
    field_varint(&mut encoded, 1, COREML_SPECIFICATION_VERSION);
    field_message(&mut encoded, 2, &description);
    field_message(&mut encoded, 500, &network);
    Ok(LoweredModel {
        bytes: encoded,
        features,
    })
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

fn encode_feature(
    description: &mut Vec<u8>,
    field: u32,
    name: &str,
    tensor: virtio_accel_tosa::Tensor<'_>,
) -> Result<(), LoweringError> {
    let shape = static_shape(tensor)?;
    if shape.is_empty() {
        return Err(LoweringError::UnsupportedGraph);
    }
    let data_type = coreml_data_type(tensor.dtype())?;
    let mut array = Vec::new();
    field_packed_varints(
        &mut array,
        1,
        shape.iter().copied().map(|value| value as u64),
    );
    field_varint(&mut array, 2, data_type);
    let mut feature_type = Vec::new();
    field_message(&mut feature_type, 5, &array);
    let mut feature = Vec::new();
    field_string(&mut feature, 1, name);
    field_message(&mut feature, 3, &feature_type);
    field_message(description, field, &feature);
    Ok(())
}

fn encode_operator(
    network: &mut Vec<u8>,
    analysis: &TosaAnalysis<'_>,
    operator_id: virtio_accel_tosa::OperatorId,
    names: &[String],
) -> Result<(), LoweringError> {
    let operator = analysis.operator(operator_id);
    let op = operator.op();
    if !supports_tosa_operator(op) {
        return Err(LoweringError::UnsupportedOperator(op));
    }
    let all_inputs = analysis.operator_inputs(operator_id);
    let outputs = analysis.operator_outputs(operator_id);
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

    // CTC constants consumed only by a layer parameter are deliberately absent from the Core ML
    // graph. They have already been validated by TOSA analysis.
    if op == Op::CONST_SHAPE {
        return Ok(());
    }
    if op == Op::CONST {
        let output = outputs[0];
        if constant_is_parameter_only(analysis, output) {
            return Ok(());
        }
        let dtype = tensor(analysis, output)?.dtype();
        if !matches!(dtype, DType::FP16 | DType::FP32 | DType::BOOL) {
            return Err(LoweringError::UnsupportedType(dtype));
        }
    }

    validate_operator_types(analysis, op, inputs, outputs)?;
    match operator.source().attributes() {
        OpAttributes::Maximum { nan_mode } | OpAttributes::Minimum { nan_mode } => {
            require_propagating_nan(nan_mode)?;
        }
        _ => {}
    }

    if op == Op::MAX_POOL2D {
        return encode_max_pool2d(network, analysis, operator_id, inputs, outputs, names);
    }

    let mut layer = Vec::new();
    field_string(
        &mut layer,
        1,
        &format!("tosa_{}_{}", operator_id.get(), op.name().unwrap_or("op")),
    );
    for value in inputs {
        field_string(&mut layer, 2, &names[value.get() as usize]);
    }
    for value in outputs {
        field_string(&mut layer, 3, &names[value.get() as usize]);
    }

    match op {
        Op::IDENTITY => field_message(&mut layer, 600, &[]),
        Op::MATMUL => field_message(&mut layer, 1045, &[]),
        Op::ADD => field_message(&mut layer, 880, &[]),
        Op::SUB => field_message(&mut layer, 905, &[]),
        Op::MUL => field_message(&mut layer, 900, &[]),
        Op::POW => field_message(&mut layer, 885, &[]),
        Op::MAXIMUM => field_message(&mut layer, 875, &[]),
        Op::MINIMUM => field_message(&mut layer, 870, &[]),
        Op::EQUAL => field_message(&mut layer, 815, &[]),
        Op::GREATER => field_message(&mut layer, 830, &[]),
        Op::GREATER_EQUAL => field_message(&mut layer, 832, &[]),
        Op::LOGICAL_OR => field_message(&mut layer, 840, &[]),
        Op::LOGICAL_XOR => field_message(&mut layer, 845, &[]),
        Op::LOGICAL_NOT => field_message(&mut layer, 850, &[]),
        Op::LOGICAL_AND => field_message(&mut layer, 855, &[]),
        Op::SELECT => field_message(&mut layer, 1330, &[]),
        Op::CEIL => field_message(&mut layer, 665, &[]),
        Op::FLOOR => field_message(&mut layer, 670, &[]),
        Op::SIN => field_message(&mut layer, 710, &[]),
        Op::COS => field_message(&mut layer, 715, &[]),
        Op::TANH => field_message(&mut layer, 760, &[]),
        Op::ERF => field_message(&mut layer, 790, &[]),
        Op::SIGMOID => {
            let mut activation = Vec::new();
            field_message(&mut activation, 40, &[]);
            field_message(&mut layer, 130, &activation);
        }
        Op::ABS => encode_unary(&mut layer, 6, None),
        Op::EXP => encode_unary(&mut layer, 4, None),
        Op::LOG => encode_unary(&mut layer, 5, None),
        // Core ML's INVERSE and RSQRT unary modes force a nonzero default epsilon when zero is
        // encoded. POWER avoids that numerical mismatch while retaining one fused unary layer.
        Op::RECIPROCAL => encode_unary(&mut layer, 3, Some(-1.0)),
        Op::RSQRT => encode_unary(&mut layer, 3, Some(-0.5)),
        Op::NEGATE => {
            let mut multiply = Vec::new();
            field_float(&mut multiply, 1, -1.0);
            field_message(&mut layer, 231, &multiply);
        }
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
            let mut clip = Vec::new();
            field_float(&mut clip, 1, decode_float(dtype, min_val)?);
            field_float(&mut clip, 2, decode_float(dtype, max_val)?);
            field_message(&mut layer, 660, &clip);
        }
        Op::ARGMAX => {
            let OpAttributes::ArgMax { axis, nan_mode } = operator.source().attributes() else {
                return Err(LoweringError::UnsupportedGraph);
            };
            require_propagating_nan(nan_mode)?;
            let mut params = Vec::new();
            field_signed(&mut params, 1, i64::from(axis));
            field_varint(&mut params, 2, 1);
            field_message(&mut layer, 1025, &params);
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
            let mut params = Vec::new();
            field_packed_varints(&mut params, 1, [axis as i64 as u64]);
            field_varint(&mut params, 2, 1);
            let field = match op {
                Op::REDUCE_MAX => 1260,
                Op::REDUCE_MIN => 1265,
                Op::REDUCE_SUM => 1270,
                _ => 1275,
            };
            field_message(&mut layer, field, &params);
        }
        Op::CONCAT => {
            let OpAttributes::Concat { axis } = operator.source().attributes() else {
                return Err(LoweringError::UnsupportedGraph);
            };
            let mut params = Vec::new();
            field_signed(&mut params, 1, i64::from(axis));
            field_message(&mut layer, 980, &params);
        }
        Op::RESHAPE => {
            let shape = static_shape(tensor(analysis, outputs[0])?)?;
            let mut params = Vec::new();
            field_packed_varints(
                &mut params,
                1,
                shape.iter().copied().map(|value| value as u64),
            );
            field_message(&mut layer, 1140, &params);
        }
        Op::REVERSE => {
            let OpAttributes::Reverse { axis } = operator.source().attributes() else {
                return Err(LoweringError::UnsupportedGraph);
            };
            let rank = tensor(analysis, inputs[0])?
                .rank()
                .ok_or(LoweringError::UnsupportedGraph)?;
            let axis = usize::try_from(axis).map_err(|_| LoweringError::UnsupportedGraph)?;
            let mut params = Vec::new();
            field_packed_varints(
                &mut params,
                1,
                (0..rank).map(|index| u64::from(index == axis)),
            );
            field_message(&mut layer, 960, &params);
        }
        Op::TRANSPOSE => {
            let OpAttributes::Transpose { perms } = operator.source().attributes() else {
                return Err(LoweringError::UnsupportedGraph);
            };
            let mut params = Vec::new();
            field_packed_varints(&mut params, 1, perms.iter().map(|axis| axis as u64));
            field_message(&mut layer, 985, &params);
        }
        Op::CONST => encode_constant(&mut layer, analysis, outputs[0])?,
        _ => return Err(LoweringError::UnsupportedOperator(op)),
    }
    field_message(network, 1, &layer);
    Ok(())
}

fn encode_max_pool2d(
    network: &mut Vec<u8>,
    analysis: &TosaAnalysis<'_>,
    operator_id: virtio_accel_tosa::OperatorId,
    inputs: &[ValueId],
    outputs: &[ValueId],
    names: &[String],
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

    let stem = format!("tosa_{}_max_pool2d", operator_id.get());
    let nchw_input = format!("{stem}_nchw_input");
    let nchw_output = format!("{stem}_nchw_output");
    encode_transpose_layer(
        network,
        &format!("{stem}_to_nchw"),
        &names[inputs[0].get() as usize],
        &nchw_input,
        [0, 3, 1, 2],
    );

    let mut params = Vec::new();
    field_packed_varints(
        &mut params,
        10,
        kernel.into_iter().map(|value| value as u64),
    );
    field_packed_varints(
        &mut params,
        20,
        stride.into_iter().map(|value| value as u64),
    );
    field_message(&mut params, 30, &[]);
    let mut pooling = Vec::new();
    field_string(&mut pooling, 1, &stem);
    field_string(&mut pooling, 2, &nchw_input);
    field_string(&mut pooling, 3, &nchw_output);
    field_message(&mut pooling, 120, &params);
    field_message(network, 1, &pooling);

    encode_transpose_layer(
        network,
        &format!("{stem}_to_nhwc"),
        &nchw_output,
        &names[outputs[0].get() as usize],
        [0, 2, 3, 1],
    );
    Ok(())
}

fn encode_transpose_layer(
    network: &mut Vec<u8>,
    name: &str,
    input: &str,
    output: &str,
    axes: impl IntoIterator<Item = u64>,
) {
    let mut params = Vec::new();
    field_packed_varints(&mut params, 1, axes);
    let mut layer = Vec::new();
    field_string(&mut layer, 1, name);
    field_string(&mut layer, 2, input);
    field_string(&mut layer, 3, output);
    field_message(&mut layer, 985, &params);
    field_message(network, 1, &layer);
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
    let is_bool = |dtype| dtype == DType::BOOL;
    let is_int32 = |dtype| dtype == DType::INT32;

    match op {
        Op::CONST => {
            require(outputs[0], |dtype| {
                matches!(dtype, DType::FP16 | DType::FP32 | DType::BOOL)
            })?;
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

fn encode_unary(layer: &mut Vec<u8>, operation: u64, alpha: Option<f32>) {
    let mut params = Vec::new();
    field_varint(&mut params, 1, operation);
    if let Some(alpha) = alpha {
        field_float(&mut params, 2, alpha);
    }
    field_message(layer, 220, &params);
}

fn encode_constant(
    layer: &mut Vec<u8>,
    analysis: &TosaAnalysis<'_>,
    output: ValueId,
) -> Result<(), LoweringError> {
    let tensor = tensor(analysis, output)?;
    let data = analysis
        .serialized_constant(output)
        .ok_or(LoweringError::InvalidConstant)?;
    let mut shape = static_shape(tensor)?;
    if shape.is_empty() {
        shape.push(1);
    }
    let mut weights = Vec::new();
    match tensor.dtype() {
        DType::FP32 => {
            if data.len() % 4 != 0 {
                return Err(LoweringError::InvalidConstant);
            }
            field_bytes(&mut weights, 1, data);
        }
        DType::FP16 => field_bytes(&mut weights, 2, data),
        DType::BOOL => {
            let mut floats = Vec::new();
            floats
                .try_reserve_exact(data.len() * 4)
                .map_err(|_| LoweringError::ResourceLimit)?;
            for value in data {
                floats.extend_from_slice(&f32::from(*value != 0).to_le_bytes());
            }
            field_bytes(&mut weights, 1, &floats);
        }
        dtype => return Err(LoweringError::UnsupportedType(dtype)),
    }
    let mut params = Vec::new();
    field_packed_varints(
        &mut params,
        1,
        shape.iter().copied().map(|value| value as u64),
    );
    field_message(&mut params, 2, &weights);
    field_message(layer, 1070, &params);
    Ok(())
}

fn static_shape(tensor: virtio_accel_tosa::Tensor<'_>) -> Result<Vec<i32>, LoweringError> {
    tensor.rank().ok_or(LoweringError::UnsupportedGraph)?;
    let shape = tensor.dimensions().collect::<Vec<_>>();
    if shape.iter().any(|dimension| *dimension <= 0) {
        return Err(LoweringError::UnsupportedGraph);
    }
    Ok(shape)
}

fn coreml_data_type(dtype: DType) -> Result<u64, LoweringError> {
    match dtype {
        DType::FP16 => Ok(COREML_FLOAT16),
        DType::FP32 => Ok(COREML_FLOAT32),
        DType::INT32 => Ok(COREML_INT32),
        _ => Err(LoweringError::UnsupportedType(dtype)),
    }
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

fn field_varint(target: &mut Vec<u8>, field: u32, value: u64) {
    varint(target, u64::from(field) << 3);
    varint(target, value);
}

fn field_signed(target: &mut Vec<u8>, field: u32, value: i64) {
    field_varint(target, field, value as u64);
}

fn field_float(target: &mut Vec<u8>, field: u32, value: f32) {
    varint(target, (u64::from(field) << 3) | 5);
    target.extend_from_slice(&value.to_le_bytes());
}

fn field_string(target: &mut Vec<u8>, field: u32, value: &str) {
    field_bytes(target, field, value.as_bytes());
}

fn field_message(target: &mut Vec<u8>, field: u32, message: &[u8]) {
    field_bytes(target, field, message);
}

fn field_bytes(target: &mut Vec<u8>, field: u32, bytes: &[u8]) {
    varint(target, (u64::from(field) << 3) | 2);
    varint(target, bytes.len() as u64);
    target.extend_from_slice(bytes);
}

fn field_packed_varints(target: &mut Vec<u8>, field: u32, values: impl IntoIterator<Item = u64>) {
    let mut packed = Vec::new();
    for value in values {
        varint(&mut packed, value);
    }
    field_bytes(target, field, &packed);
}

fn varint(target: &mut Vec<u8>, mut value: u64) {
    while value >= 0x80 {
        target.push((value as u8) | 0x80);
        value >>= 7;
    }
    target.push(value as u8);
}

#[cfg(test)]
mod tests {
    use super::*;

    const IDENTITY_FP32: &[u8] = include_bytes!("../tests/data/identity-fp32-v1.0.0.tosa");
    #[test]
    fn lowers_a_verified_tosa_model_without_host_dependencies() {
        let lowered = lower_tosa(IDENTITY_FP32, COREML_TOSA_TARGET).unwrap();

        assert!(!lowered.bytes.is_empty());
        assert_eq!(lowered.features.len(), 2);
        assert_eq!(lowered.features[0].slot, 0);
        assert_eq!(lowered.features[0].role, LoweredFeatureRole::Input);
        assert_eq!(lowered.features[1].slot, 1);
        assert_eq!(lowered.features[1].role, LoweredFeatureRole::Output);
    }

    #[test]
    fn rejects_a_different_tosa_target_before_parsing() {
        let target = Target::new(
            Version::TOSA_1_0,
            ProfileSet::INTEGER,
            Level::Level8K,
            ExtensionSet::NONE,
        );

        assert!(matches!(
            lower_tosa(IDENTITY_FP32, target),
            Err(LoweringError::UnsupportedGraph)
        ));
    }

    #[test]
    fn reports_low_precision_types_as_unsupported_in_the_neural_network_lowering() {
        assert!(supports_tosa_dtype(DType::FP16));
        assert!(supports_tosa_dtype(DType::FP32));
        assert!(supports_tosa_dtype(DType::INT32));
        assert!(!supports_tosa_dtype(DType::INT8));
        assert!(!supports_tosa_dtype(DType::INT4));
        assert!(!supports_tosa_dtype(DType::FP8E4M3));
        assert!(!supports_tosa_dtype(DType::FP8E5M2));
    }

    #[test]
    fn rejects_shared_quantized_artifacts_at_the_declared_target_boundary() {
        use virtio_accel_conformance::numerics::{
            IDENTITY_FP8E4M3, IDENTITY_FP8E5M2, IDENTITY_INT4, IDENTITY_INT8,
        };

        for (case, target) in [
            (
                IDENTITY_INT8,
                Target::new(
                    Version::TOSA_1_0,
                    ProfileSet::INTEGER,
                    Level::Level8K,
                    ExtensionSet::NONE,
                ),
            ),
            (
                IDENTITY_INT4,
                Target::new(
                    Version::TOSA_1_0,
                    ProfileSet::INTEGER,
                    Level::Level8K,
                    ExtensionSet::INT4,
                ),
            ),
            (
                IDENTITY_FP8E4M3,
                Target::new(
                    Version::TOSA_1_0,
                    ProfileSet::FLOATING_POINT,
                    Level::Level8K,
                    ExtensionSet::FP8E4M3,
                ),
            ),
            (
                IDENTITY_FP8E5M2,
                Target::new(
                    Version::TOSA_1_0,
                    ProfileSet::FLOATING_POINT,
                    Level::Level8K,
                    ExtensionSet::FP8E5M2,
                ),
            ),
        ] {
            assert!(matches!(
                lower_tosa(case.artifact, target),
                Err(LoweringError::UnsupportedGraph)
            ));
        }
    }

    #[test]
    fn lowers_batched_matmul_without_encoding_parameter_constants() {
        let lowered = lower_tosa(
            virtio_accel_conformance::numerics::MATMUL_FP32.artifact,
            COREML_TOSA_TARGET,
        )
        .unwrap();

        assert!(!lowered.bytes.is_empty());
        assert_eq!(lowered.features.len(), 3);
        assert_eq!(lowered.features[0].slot, 0);
        assert_eq!(lowered.features[1].slot, 1);
        assert_eq!(lowered.features[2].slot, 2);
        // NeuralNetworkLayer.batchedMatmul is field 1045 (wire key 8362 = 0xaa 0x41).
        assert!(lowered.bytes.windows(2).any(|bytes| bytes == [0xaa, 0x41]));
    }

    #[test]
    fn lowers_the_shared_fp32_edge_identity_artifact() {
        let lowered = lower_tosa(
            virtio_accel_conformance::numerics::IDENTITY_EDGES_FP32.artifact,
            COREML_TOSA_TARGET,
        )
        .unwrap();

        assert_eq!(lowered.features.len(), 2);
        assert!(!lowered.bytes.is_empty());
    }

    #[test]
    fn lowers_nhwc_max_pool_through_explicit_layout_transposes() {
        let lowered = lower_tosa(
            virtio_accel_conformance::numerics::MAX_POOL2D_FP32.artifact,
            COREML_TOSA_TARGET,
        )
        .unwrap();

        // The lowering emits transpose -> pooling -> transpose. Pooling is field 120
        // (wire key 962 = 0xc2 0x07); transpose is field 985 (0xca 0x3d).
        assert_eq!(
            lowered
                .bytes
                .windows(2)
                .filter(|bytes| *bytes == [0xca, 0x3d])
                .count(),
            2
        );
        assert!(lowered.bytes.windows(2).any(|bytes| bytes == [0xc2, 0x07]));
    }

    #[test]
    fn lowers_every_shared_fp16_numerical_artifact() {
        use virtio_accel_conformance::numerics::{
            IDENTITY_EDGES_FP16, MATMUL_FP16, MAX_POOL2D_FP16,
        };

        for case in [IDENTITY_EDGES_FP16, MATMUL_FP16, MAX_POOL2D_FP16] {
            let lowered = lower_tosa(case.artifact, COREML_TOSA_TARGET).unwrap();
            assert!(!lowered.bytes.is_empty(), "{}", case.name);
        }
    }

    #[test]
    fn greater_equal_uses_the_distinct_core_ml_field() {
        assert!(supports_tosa_operator(Op::GREATER_EQUAL));
        let mut layer = Vec::new();
        field_message(&mut layer, 832, &[]);
        assert_eq!(layer, [0x82, 0x34, 0x00]);
    }

    #[test]
    fn fp16_parameters_preserve_zero_finite_and_nan_classes() {
        assert_eq!(decode_float(DType::FP16, &0_u16.to_le_bytes()), Ok(0.0));
        assert_eq!(
            decode_float(DType::FP16, &0x8000_u16.to_le_bytes())
                .unwrap()
                .to_bits(),
            (-0.0_f32).to_bits()
        );
        assert_eq!(
            decode_float(DType::FP16, &0x3c00_u16.to_le_bytes()),
            Ok(1.0)
        );
        assert!(
            decode_float(DType::FP16, &0x7e00_u16.to_le_bytes())
                .unwrap()
                .is_nan()
        );
        assert_eq!(
            decode_float(DType::FP16, &0x0001_u16.to_le_bytes())
                .unwrap()
                .to_bits(),
            (2.0_f32.powi(-24)).to_bits()
        );
    }
}
