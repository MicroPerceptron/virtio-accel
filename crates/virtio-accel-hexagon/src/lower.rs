//! Strict TOSA 1.0 admission and provider-local QNN graph planning.
//!
//! This module deliberately contains no QNN ABI types. It turns a verified TOSA artifact into
//! owned tensor, binding, and operation metadata that the native boundary can translate while the
//! source FlatBuffer is no longer borrowed. It compiles and tests on hosts without QAIRT.

#![cfg_attr(not(va_hexagon), allow(dead_code))]

use std::fmt;

use virtio_accel_tosa::{
    AnalysisError, AnalyzedValueKind, CapabilityDescriptor, DType, DTypeCapability,
    Error as ParseError, ExtensionSet, GraphCapabilities, Level, NanPropagationMode, Op,
    OpAttributes, OperatorCapability, OperatorConstraints, ProfileSet, RuntimeCondition,
    RuntimeConditionSupport, Target, TosaAnalysis, ValueId, ValueRoles, Version, parse,
};

/// TOSA declaration accepted by the first Hexagon tier.
///
/// The target selects the TOSA floating-point profile. Backend admission narrows its tensor types
/// to FP16 until the selected HTP runtime can prove FP32 computation without precision reduction.
pub const HEXAGON_TOSA_TARGET: Target = Target::new(
    Version::TOSA_1_0,
    ProfileSet::FLOATING_POINT,
    Level::Level8K,
    ExtensionSet::NONE,
);

/// TOSA integer-profile target lowered with exact INT8 storage and INT32 accumulation.
pub const HEXAGON_TOSA_INTEGER_TARGET: Target = Target::new(
    Version::TOSA_1_0,
    ProfileSet::INTEGER,
    Level::Level8K,
    ExtensionSet::NONE,
);

const FLOAT_DTYPES: &[DTypeCapability] = &[
    DTypeCapability::new(DType::FP16, ValueRoles::ALL),
    DTypeCapability::new(DType::BOOL, ValueRoles::ALL),
    DTypeCapability::new(
        DType::INT32,
        ValueRoles::OUTPUT
            .union(ValueRoles::CONSTANT)
            .union(ValueRoles::INTERMEDIATE),
    ),
];

const INTEGER_DTYPES: &[DTypeCapability] = &[
    DTypeCapability::new(DType::INT8, ValueRoles::ALL),
    DTypeCapability::new(
        DType::INT32,
        ValueRoles::OUTPUT
            .union(ValueRoles::CONSTANT)
            .union(ValueRoles::INTERMEDIATE),
    ),
];

const FLOAT_OPERATORS: &[OperatorCapability] = &[
    OperatorCapability::constrained(Op::ARGMAX, OperatorConstraints::PROPAGATING_NAN),
    OperatorCapability::constrained(Op::MATMUL, OperatorConstraints::ZERO_ZERO_POINTS),
    OperatorCapability::constrained(
        Op::MAX_POOL2D,
        OperatorConstraints::PROPAGATING_NAN.union(OperatorConstraints::ZERO_PADDING),
    ),
    OperatorCapability::constrained(Op::CLAMP, OperatorConstraints::PROPAGATING_NAN),
    OperatorCapability::new(Op::SIGMOID),
    OperatorCapability::new(Op::TANH),
    OperatorCapability::new(Op::ADD),
    OperatorCapability::new(Op::SUB),
    OperatorCapability::constrained(Op::MUL, OperatorConstraints::ZERO_SHIFT),
    OperatorCapability::new(Op::POW),
    OperatorCapability::constrained(Op::MAXIMUM, OperatorConstraints::PROPAGATING_NAN),
    OperatorCapability::constrained(Op::MINIMUM, OperatorConstraints::PROPAGATING_NAN),
    OperatorCapability::new(Op::LOGICAL_AND),
    OperatorCapability::new(Op::LOGICAL_OR),
    OperatorCapability::new(Op::LOGICAL_XOR),
    OperatorCapability::new(Op::ABS),
    OperatorCapability::new(Op::CEIL),
    OperatorCapability::new(Op::COS),
    OperatorCapability::new(Op::EXP),
    OperatorCapability::new(Op::FLOOR),
    OperatorCapability::new(Op::LOG),
    OperatorCapability::new(Op::LOGICAL_NOT),
    OperatorCapability::constrained(Op::NEGATE, OperatorConstraints::ZERO_ZERO_POINTS),
    OperatorCapability::new(Op::RECIPROCAL),
    OperatorCapability::new(Op::RSQRT),
    OperatorCapability::new(Op::SIN),
    OperatorCapability::new(Op::SELECT),
    OperatorCapability::new(Op::EQUAL),
    OperatorCapability::new(Op::GREATER),
    OperatorCapability::new(Op::GREATER_EQUAL),
    OperatorCapability::constrained(Op::REDUCE_MAX, OperatorConstraints::PROPAGATING_NAN),
    OperatorCapability::constrained(Op::REDUCE_MIN, OperatorConstraints::PROPAGATING_NAN),
    OperatorCapability::new(Op::REDUCE_PRODUCT),
    OperatorCapability::new(Op::REDUCE_SUM),
    OperatorCapability::new(Op::CONCAT),
    OperatorCapability::constrained(Op::RESHAPE, OperatorConstraints::CONSTANT_PARAMETERS),
    OperatorCapability::new(Op::REVERSE),
    OperatorCapability::new(Op::TRANSPOSE),
    OperatorCapability::new(Op::CONST),
    OperatorCapability::new(Op::IDENTITY),
    OperatorCapability::new(Op::CONST_SHAPE),
];

const INTEGER_OPERATORS: &[OperatorCapability] = &[
    OperatorCapability::new(Op::CONST),
    OperatorCapability::new(Op::IDENTITY),
    OperatorCapability::new(Op::MATMUL),
];

/// Conservative floating-profile capability boundary for the validated QNN HTP tier.
pub const HEXAGON_TOSA_CAPABILITY: CapabilityDescriptor = CapabilityDescriptor {
    target: HEXAGON_TOSA_TARGET,
    dtypes: FLOAT_DTYPES,
    operators: FLOAT_OPERATORS,
    graph: GraphCapabilities {
        max_regions: 1,
        max_blocks: 1,
        dynamic_shapes: false,
        runtime_conditions: RuntimeConditionSupport::AdvisoryOnly,
    },
};

/// Conservative exact integer-profile capability boundary for the validated QNN HTP tier.
pub const HEXAGON_TOSA_INTEGER_CAPABILITY: CapabilityDescriptor = CapabilityDescriptor {
    target: HEXAGON_TOSA_INTEGER_TARGET,
    dtypes: INTEGER_DTYPES,
    operators: INTEGER_OPERATORS,
    graph: GraphCapabilities {
        max_regions: 1,
        max_blocks: 1,
        dynamic_shapes: false,
        runtime_conditions: RuntimeConditionSupport::None,
    },
};

/// Failure while validating and planning a graph for QNN HTP.
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

/// Whether the first hardware tier has a QNN lowering for `op`.
pub const fn supports_tosa_operator(op: Op) -> bool {
    HEXAGON_TOSA_CAPABILITY.supports_operator(op)
}

fn supports_operator_for_target(op: Op, integer: bool) -> bool {
    if integer {
        HEXAGON_TOSA_INTEGER_CAPABILITY.supports_operator(op)
    } else {
        supports_tosa_operator(op)
    }
}

/// Whether the first hardware tier may expose `dtype` at a model boundary.
///
/// FP32 remains deliberately rejected because current HTP floating-point execution may use FP16
/// math. Integer and packed low-precision tiers require separate targets and evidence.
pub const fn supports_tosa_dtype(dtype: DType) -> bool {
    HEXAGON_TOSA_CAPABILITY.supports_dtype(dtype, ValueRoles::INPUT)
        || HEXAGON_TOSA_CAPABILITY.supports_dtype(dtype, ValueRoles::OUTPUT)
        || HEXAGON_TOSA_INTEGER_CAPABILITY.supports_dtype(dtype, ValueRoles::INPUT)
        || HEXAGON_TOSA_INTEGER_CAPABILITY.supports_dtype(dtype, ValueRoles::OUTPUT)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Element {
    Bool,
    F16,
    F32,
    I8,
    I32,
}

impl Element {
    pub(crate) const fn scalar_bytes(self) -> u64 {
        match self {
            Self::Bool | Self::I8 => 1,
            Self::F16 => 2,
            Self::F32 | Self::I32 => 4,
        }
    }

    fn for_dtype(dtype: DType) -> Result<Self, LoweringError> {
        match dtype {
            DType::BOOL => Ok(Self::Bool),
            DType::FP16 => Ok(Self::F16),
            DType::FP32 => Ok(Self::F32),
            DType::INT8 => Ok(Self::I8),
            DType::INT32 => Ok(Self::I32),
            _ => Err(LoweringError::UnsupportedType(dtype)),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Quantization {
    pub scale: f32,
    pub offset: i32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FeatureRole {
    Input,
    Output,
}

/// One exact model-boundary binding.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct LoweredFeature {
    pub slot: u32,
    pub role: FeatureRole,
    pub io_index: u32,
    pub value: u32,
    pub dims: Vec<u32>,
    pub byte_len: u64,
}

/// One owned tensor descriptor used while constructing the QNN graph.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct LoweredTensor {
    pub value: u32,
    pub element: Element,
    pub quantization: Option<Quantization>,
    pub dims: Vec<u32>,
    pub data: Option<Vec<u8>>,
}

/// QNN operation selected by portable TOSA lowering.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum NodeKind {
    Identity,
    Transpose,
    Reverse,
    Concat,
    MatMul,
    MaxPool2d,
    Add,
    Subtract,
    Multiply,
    Maximum,
    Minimum,
    Power,
    Abs,
    Ceil,
    Cos,
    Exp,
    Floor,
    Log,
    Negate,
    Reciprocal,
    Rsqrt,
    Sin,
    Sigmoid,
    Tanh,
    Clamp,
    Equal,
    Greater,
    GreaterEqual,
    Select,
    LogicalAnd,
    LogicalOr,
    LogicalXor,
    LogicalNot,
    ArgMax,
    ReduceMax,
    ReduceMin,
    ReduceProduct,
    ReduceSum,
}

/// One owned operation descriptor. Parameter meaning is fixed by `kind` and validated natively.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct LoweredNode {
    pub kind: NodeKind,
    pub inputs: Vec<u32>,
    pub outputs: Vec<u32>,
    pub parameters: Vec<i32>,
}

/// Fully owned graph plan produced before entering the native QNN boundary.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct LoweredModel {
    pub tensors: Vec<LoweredTensor>,
    pub nodes: Vec<LoweredNode>,
    pub features: Vec<LoweredFeature>,
    pub precision: Option<Element>,
}

impl LoweredModel {
    pub(crate) fn boundary(&self, value: u32) -> Option<(FeatureRole, u32)> {
        self.features
            .iter()
            .find(|feature| feature.value == value)
            .map(|feature| (feature.role, feature.io_index))
    }
}

pub(crate) fn lower_tosa(bytes: &[u8], target: Target) -> Result<LoweredModel, LoweringError> {
    let integer = if target == HEXAGON_TOSA_TARGET {
        false
    } else if target == HEXAGON_TOSA_INTEGER_TARGET {
        true
    } else {
        return Err(LoweringError::UnsupportedGraph);
    };
    let model = parse(bytes).map_err(LoweringError::Parse)?;
    let analysis = model.analyze_for(target).map_err(LoweringError::Analysis)?;
    if analysis.regions().len() != 1
        || analysis.blocks().len() != 1
        || analysis
            .conditions()
            .iter()
            .any(|condition| !matches!(condition, RuntimeCondition::PowDomain { .. }))
    {
        return Err(LoweringError::UnsupportedGraph);
    }

    validate_types(&analysis, integer)?;
    let block = analysis.blocks()[0].id();
    let inputs = analysis.block_inputs(block);
    let outputs = analysis.block_outputs(block);
    if inputs.is_empty()
        || outputs.is_empty()
        || inputs.iter().any(|input| outputs.contains(input))
        || inputs
            .iter()
            .chain(outputs)
            .any(|value| !matches!(analysis.value(*value).kind(), AnalyzedValueKind::Tensor(_)))
        || outputs
            .iter()
            .any(|value| analysis.serialized_constant(*value).is_some())
        || inputs.len().checked_add(outputs.len()).is_none()
    {
        return Err(LoweringError::UnsupportedGraph);
    }

    let mut tensors = Vec::new();
    tensors
        .try_reserve_exact(analysis.values().len())
        .map_err(|_| LoweringError::ResourceLimit)?;
    for value in analysis.values() {
        let AnalyzedValueKind::Tensor(tensor) = value.kind() else {
            if analysis.serialized_constant(value.id()).is_none() {
                return Err(LoweringError::UnsupportedGraph);
            }
            continue;
        };
        let element = Element::for_dtype(tensor.dtype())?;
        tensors.push(LoweredTensor {
            value: value.id().get(),
            element,
            quantization: (integer && matches!(element, Element::I8 | Element::I32)).then_some(
                Quantization {
                    scale: 1.0,
                    offset: 0,
                },
            ),
            dims: static_dims(tensor, true)?,
            data: analysis.serialized_constant(value.id()).map(<[u8]>::to_vec),
        });
    }

    let mut features = Vec::new();
    features
        .try_reserve_exact(inputs.len() + outputs.len())
        .map_err(|_| LoweringError::ResourceLimit)?;
    for (index, value) in inputs.iter().copied().enumerate() {
        features.push(lower_feature(
            &analysis,
            value,
            index,
            index,
            FeatureRole::Input,
        )?);
    }
    for (index, value) in outputs.iter().copied().enumerate() {
        features.push(lower_feature(
            &analysis,
            value,
            inputs.len() + index,
            index,
            FeatureRole::Output,
        )?);
    }

    let mut nodes = Vec::new();
    let mut quantization_offsets = Vec::new();
    nodes
        .try_reserve_exact(analysis.execution_order(block).len())
        .map_err(|_| LoweringError::ResourceLimit)?;
    for operator_id in analysis.execution_order(block) {
        let operator = analysis.operator(*operator_id);
        let op = operator.op();
        if !supports_operator_for_target(op, integer) {
            return Err(LoweringError::UnsupportedOperator(op));
        }
        let op_inputs = analysis.operator_inputs(*operator_id);
        let op_outputs = analysis.operator_outputs(*operator_id);
        match op {
            Op::CONST => {
                if op_inputs.is_empty()
                    && op_outputs.len() == 1
                    && analysis.serialized_constant(op_outputs[0]).is_some()
                {
                    continue;
                }
                return Err(LoweringError::InvalidConstant);
            }
            Op::CONST_SHAPE => {
                if op_inputs.is_empty()
                    && op_outputs.len() == 1
                    && analysis.serialized_constant(op_outputs[0]).is_some()
                {
                    continue;
                }
                return Err(LoweringError::InvalidConstant);
            }
            Op::IDENTITY => {
                require_arity(op_inputs, 1, op_outputs, 1)?;
                nodes.push(lowered_node(
                    NodeKind::Identity,
                    op_inputs,
                    op_outputs,
                    Vec::new(),
                ));
            }
            Op::RESHAPE => {
                require_arity(op_inputs, 2, op_outputs, 1)?;
                analysis
                    .serialized_constant(op_inputs[1])
                    .ok_or(LoweringError::InvalidConstant)?;
                nodes.push(lowered_node(
                    NodeKind::Identity,
                    &op_inputs[..1],
                    op_outputs,
                    Vec::new(),
                ));
            }
            Op::TRANSPOSE => {
                require_arity(op_inputs, 1, op_outputs, 1)?;
                let OpAttributes::Transpose { perms } = operator.source().attributes() else {
                    return Err(LoweringError::UnsupportedGraph);
                };
                let parameters = perms.iter().collect::<Vec<_>>();
                if parameters.is_empty() {
                    return Err(LoweringError::UnsupportedGraph);
                }
                nodes.push(lowered_node(
                    NodeKind::Transpose,
                    op_inputs,
                    op_outputs,
                    parameters,
                ));
            }
            Op::REVERSE => {
                require_arity(op_inputs, 1, op_outputs, 1)?;
                let OpAttributes::Reverse { axis } = operator.source().attributes() else {
                    return Err(LoweringError::UnsupportedGraph);
                };
                nodes.push(lowered_node(
                    NodeKind::Reverse,
                    op_inputs,
                    op_outputs,
                    vec![axis],
                ));
            }
            Op::CONCAT => {
                if op_inputs.is_empty() || op_outputs.len() != 1 {
                    return Err(LoweringError::UnsupportedGraph);
                }
                let OpAttributes::Concat { axis } = operator.source().attributes() else {
                    return Err(LoweringError::UnsupportedGraph);
                };
                nodes.push(lowered_node(
                    NodeKind::Concat,
                    op_inputs,
                    op_outputs,
                    vec![axis],
                ));
            }
            Op::MATMUL => {
                require_arity(op_inputs, 4, op_outputs, 1)?;
                let left_zero_point = scalar_zero_point(&analysis, op_inputs[2])?;
                let right_zero_point = scalar_zero_point(&analysis, op_inputs[3])?;
                if integer {
                    set_quantization_offset(
                        &mut tensors,
                        &mut quantization_offsets,
                        op_inputs[0],
                        left_zero_point,
                    )?;
                    set_quantization_offset(
                        &mut tensors,
                        &mut quantization_offsets,
                        op_inputs[1],
                        right_zero_point,
                    )?;
                } else if left_zero_point != 0 || right_zero_point != 0 {
                    return Err(LoweringError::UnsupportedGraph);
                }
                nodes.push(LoweredNode {
                    kind: NodeKind::MatMul,
                    inputs: vec![op_inputs[0].get(), op_inputs[1].get()],
                    outputs: vec![op_outputs[0].get()],
                    parameters: Vec::new(),
                });
            }
            Op::MAX_POOL2D => {
                require_arity(op_inputs, 1, op_outputs, 1)?;
                let OpAttributes::MaxPool2d {
                    kernel,
                    stride,
                    pad,
                    nan_mode,
                } = operator.source().attributes()
                else {
                    return Err(LoweringError::UnsupportedGraph);
                };
                if nan_mode != NanPropagationMode::PROPAGATE {
                    return Err(LoweringError::UnsupportedGraph);
                }
                let kernel = fixed_positive_pair(kernel.iter())?;
                let stride = fixed_positive_pair(stride.iter())?;
                let pad = pad.iter().collect::<Vec<_>>();
                if pad.len() != 4 || pad.iter().any(|value| *value != 0) {
                    return Err(LoweringError::UnsupportedGraph);
                }
                let input_dims = tensor_dims(&analysis, op_inputs[0], false)?;
                let output_dims = tensor_dims(&analysis, op_outputs[0], false)?;
                if input_dims.len() != 4 || output_dims.len() != 4 {
                    return Err(LoweringError::UnsupportedGraph);
                }
                nodes.push(LoweredNode {
                    kind: NodeKind::MaxPool2d,
                    inputs: vec![op_inputs[0].get()],
                    outputs: vec![op_outputs[0].get()],
                    parameters: vec![
                        kernel[0] as i32,
                        kernel[1] as i32,
                        stride[0] as i32,
                        stride[1] as i32,
                    ],
                });
            }
            Op::ADD | Op::SUB | Op::POW | Op::MAXIMUM | Op::MINIMUM => {
                require_arity(op_inputs, 2, op_outputs, 1)?;
                match operator.source().attributes() {
                    OpAttributes::Maximum { nan_mode } | OpAttributes::Minimum { nan_mode }
                        if nan_mode != NanPropagationMode::PROPAGATE =>
                    {
                        return Err(LoweringError::UnsupportedGraph);
                    }
                    _ => {}
                }
                let kind = match op {
                    Op::ADD => NodeKind::Add,
                    Op::SUB => NodeKind::Subtract,
                    Op::POW => NodeKind::Power,
                    Op::MAXIMUM => NodeKind::Maximum,
                    Op::MINIMUM => NodeKind::Minimum,
                    _ => unreachable!(),
                };
                nodes.push(LoweredNode {
                    kind,
                    inputs: op_inputs.iter().map(|value| value.get()).collect(),
                    outputs: vec![op_outputs[0].get()],
                    parameters: Vec::new(),
                });
            }
            Op::MUL => {
                require_arity(op_inputs, 3, op_outputs, 1)?;
                let shift = analysis
                    .serialized_constant(op_inputs[2])
                    .ok_or(LoweringError::InvalidConstant)?;
                if shift.iter().any(|byte| *byte != 0) {
                    return Err(LoweringError::UnsupportedGraph);
                }
                nodes.push(LoweredNode {
                    kind: NodeKind::Multiply,
                    inputs: op_inputs[..2].iter().map(|value| value.get()).collect(),
                    outputs: vec![op_outputs[0].get()],
                    parameters: Vec::new(),
                });
            }
            Op::ABS
            | Op::CEIL
            | Op::COS
            | Op::EXP
            | Op::FLOOR
            | Op::LOG
            | Op::RECIPROCAL
            | Op::RSQRT
            | Op::SIN
            | Op::SIGMOID
            | Op::TANH
            | Op::LOGICAL_NOT => {
                require_arity(op_inputs, 1, op_outputs, 1)?;
                let kind = match op {
                    Op::ABS => NodeKind::Abs,
                    Op::CEIL => NodeKind::Ceil,
                    Op::COS => NodeKind::Cos,
                    Op::EXP => NodeKind::Exp,
                    Op::FLOOR => NodeKind::Floor,
                    Op::LOG => NodeKind::Log,
                    Op::RECIPROCAL => NodeKind::Reciprocal,
                    Op::RSQRT => NodeKind::Rsqrt,
                    Op::SIN => NodeKind::Sin,
                    Op::SIGMOID => NodeKind::Sigmoid,
                    Op::TANH => NodeKind::Tanh,
                    Op::LOGICAL_NOT => NodeKind::LogicalNot,
                    _ => unreachable!(),
                };
                nodes.push(lowered_node(kind, op_inputs, op_outputs, Vec::new()));
            }
            Op::NEGATE => {
                require_arity(op_inputs, 3, op_outputs, 1)?;
                if scalar_zero_point(&analysis, op_inputs[1])? != 0
                    || scalar_zero_point(&analysis, op_inputs[2])? != 0
                {
                    return Err(LoweringError::UnsupportedGraph);
                }
                nodes.push(lowered_node(
                    NodeKind::Negate,
                    &op_inputs[..1],
                    op_outputs,
                    Vec::new(),
                ));
            }
            Op::CLAMP => {
                require_arity(op_inputs, 1, op_outputs, 1)?;
                let OpAttributes::Clamp {
                    min_val,
                    max_val,
                    nan_mode,
                } = operator.source().attributes()
                else {
                    return Err(LoweringError::UnsupportedGraph);
                };
                if nan_mode != NanPropagationMode::PROPAGATE {
                    return Err(LoweringError::UnsupportedGraph);
                }
                let dtype = tensor(&analysis, op_inputs[0])?.dtype();
                let minimum = decode_float(dtype, min_val)?;
                let maximum = decode_float(dtype, max_val)?;
                nodes.push(lowered_node(
                    NodeKind::Clamp,
                    op_inputs,
                    op_outputs,
                    vec![minimum.to_bits() as i32, maximum.to_bits() as i32],
                ));
            }
            Op::EQUAL
            | Op::GREATER
            | Op::GREATER_EQUAL
            | Op::LOGICAL_AND
            | Op::LOGICAL_OR
            | Op::LOGICAL_XOR => {
                require_arity(op_inputs, 2, op_outputs, 1)?;
                let kind = match op {
                    Op::EQUAL => NodeKind::Equal,
                    Op::GREATER => NodeKind::Greater,
                    Op::GREATER_EQUAL => NodeKind::GreaterEqual,
                    Op::LOGICAL_AND => NodeKind::LogicalAnd,
                    Op::LOGICAL_OR => NodeKind::LogicalOr,
                    Op::LOGICAL_XOR => NodeKind::LogicalXor,
                    _ => unreachable!(),
                };
                nodes.push(lowered_node(kind, op_inputs, op_outputs, Vec::new()));
            }
            Op::SELECT => {
                require_arity(op_inputs, 3, op_outputs, 1)?;
                nodes.push(lowered_node(
                    NodeKind::Select,
                    op_inputs,
                    op_outputs,
                    Vec::new(),
                ));
            }
            Op::ARGMAX => {
                require_arity(op_inputs, 1, op_outputs, 1)?;
                let OpAttributes::ArgMax { axis, nan_mode } = operator.source().attributes() else {
                    return Err(LoweringError::UnsupportedGraph);
                };
                if nan_mode != NanPropagationMode::PROPAGATE {
                    return Err(LoweringError::UnsupportedGraph);
                }
                nodes.push(lowered_node(
                    NodeKind::ArgMax,
                    op_inputs,
                    op_outputs,
                    vec![axis],
                ));
            }
            Op::REDUCE_MAX | Op::REDUCE_MIN | Op::REDUCE_PRODUCT | Op::REDUCE_SUM => {
                require_arity(op_inputs, 1, op_outputs, 1)?;
                let (kind, axis) = match operator.source().attributes() {
                    OpAttributes::ReduceMax { axis, nan_mode } => {
                        if nan_mode != NanPropagationMode::PROPAGATE {
                            return Err(LoweringError::UnsupportedGraph);
                        }
                        (NodeKind::ReduceMax, axis)
                    }
                    OpAttributes::ReduceMin { axis, nan_mode } => {
                        if nan_mode != NanPropagationMode::PROPAGATE {
                            return Err(LoweringError::UnsupportedGraph);
                        }
                        (NodeKind::ReduceMin, axis)
                    }
                    OpAttributes::ReduceProduct { axis } => (NodeKind::ReduceProduct, axis),
                    OpAttributes::ReduceSum { axis } => (NodeKind::ReduceSum, axis),
                    _ => return Err(LoweringError::UnsupportedGraph),
                };
                nodes.push(lowered_node(kind, op_inputs, op_outputs, vec![axis]));
            }
            _ => return Err(LoweringError::UnsupportedOperator(op)),
        }
    }

    if nodes.is_empty() {
        return Err(LoweringError::UnsupportedGraph);
    }
    Ok(LoweredModel {
        tensors,
        nodes,
        features,
        precision: if integer {
            None
        } else if analysis.values().iter().any(|value| {
            matches!(value.kind(), AnalyzedValueKind::Tensor(tensor) if tensor.dtype() == DType::FP32)
        }) {
            Some(Element::F32)
        } else {
            Some(Element::F16)
        },
    })
}

fn validate_types(analysis: &TosaAnalysis<'_>, integer: bool) -> Result<(), LoweringError> {
    for value in analysis.values() {
        let AnalyzedValueKind::Tensor(tensor) = value.kind() else {
            if analysis.serialized_constant(value.id()).is_none() {
                return Err(LoweringError::UnsupportedGraph);
            }
            continue;
        };
        let supported = if integer {
            matches!(tensor.dtype(), DType::INT8 | DType::INT32)
        } else {
            matches!(tensor.dtype(), DType::BOOL | DType::FP16 | DType::INT32)
                || (tensor.dtype() == DType::INT8
                    && constant_is_parameter_only(analysis, value.id()))
        };
        if !supported {
            return Err(LoweringError::UnsupportedType(tensor.dtype()));
        }
    }
    Ok(())
}

fn lower_feature(
    analysis: &TosaAnalysis<'_>,
    value: ValueId,
    slot: usize,
    io_index: usize,
    role: FeatureRole,
) -> Result<LoweredFeature, LoweringError> {
    let dims = tensor_dims(analysis, value, false)?;
    let element = Element::for_dtype(tensor(analysis, value)?.dtype())?;
    let byte_len = checked_tensor_byte_len(element, &dims)?;
    Ok(LoweredFeature {
        slot: u32::try_from(slot).map_err(|_| LoweringError::ResourceLimit)?,
        role,
        io_index: u32::try_from(io_index).map_err(|_| LoweringError::ResourceLimit)?,
        value: value.get(),
        dims,
        byte_len,
    })
}

fn checked_tensor_byte_len(element: Element, dims: &[u32]) -> Result<u64, LoweringError> {
    dims.iter().try_fold(element.scalar_bytes(), |bytes, dim| {
        bytes
            .checked_mul(u64::from(*dim))
            .ok_or(LoweringError::ResourceLimit)
    })
}

fn tensor_dims(
    analysis: &TosaAnalysis<'_>,
    value: ValueId,
    allow_scalar: bool,
) -> Result<Vec<u32>, LoweringError> {
    let AnalyzedValueKind::Tensor(tensor) = analysis.value(value).kind() else {
        return Err(LoweringError::UnsupportedGraph);
    };
    static_dims(tensor, allow_scalar)
}

fn static_dims(
    tensor: virtio_accel_tosa::Tensor<'_>,
    allow_scalar: bool,
) -> Result<Vec<u32>, LoweringError> {
    tensor.rank().ok_or(LoweringError::UnsupportedGraph)?;
    let dims = tensor
        .dimensions()
        .map(|dim| u32::try_from(dim).map_err(|_| LoweringError::UnsupportedGraph))
        .collect::<Result<Vec<_>, _>>()?;
    if (!allow_scalar && dims.is_empty()) || dims.contains(&0) {
        return Err(LoweringError::UnsupportedGraph);
    }
    Ok(dims)
}

fn lowered_node(
    kind: NodeKind,
    inputs: &[ValueId],
    outputs: &[ValueId],
    parameters: Vec<i32>,
) -> LoweredNode {
    LoweredNode {
        kind,
        inputs: inputs.iter().map(|value| value.get()).collect(),
        outputs: outputs.iter().map(|value| value.get()).collect(),
        parameters,
    }
}

fn decode_float(dtype: DType, bytes: &[u8]) -> Result<f32, LoweringError> {
    match dtype {
        DType::FP16 if bytes.len() == 2 => Ok(f16_to_f32(u16::from_le_bytes(
            bytes.try_into().expect("length checked"),
        ))),
        DType::FP32 if bytes.len() == 4 => Ok(f32::from_le_bytes(
            bytes.try_into().expect("length checked"),
        )),
        _ => Err(LoweringError::InvalidConstant),
    }
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exponent = (bits >> 10) & 0x1f;
    let fraction = u32::from(bits & 0x03ff);
    let output = match exponent {
        0 if fraction == 0 => sign,
        0 => {
            let leading = 31 - fraction.leading_zeros();
            let normalized_fraction = (fraction << (10 - leading)) & 0x03ff;
            let exponent32 = 127 - 14 - (10 - leading);
            sign | (exponent32 << 23) | (normalized_fraction << 13)
        }
        0x1f => sign | 0x7f80_0000 | (fraction << 13),
        _ => sign | (u32::from(exponent + 112) << 23) | (fraction << 13),
    };
    f32::from_bits(output)
}

fn require_arity(
    inputs: &[ValueId],
    input_count: usize,
    outputs: &[ValueId],
    output_count: usize,
) -> Result<(), LoweringError> {
    if inputs.len() == input_count && outputs.len() == output_count {
        Ok(())
    } else {
        Err(LoweringError::UnsupportedGraph)
    }
}

fn fixed_positive_pair(values: impl Iterator<Item = i32>) -> Result<[u32; 2], LoweringError> {
    let values = values.collect::<Vec<_>>();
    if values.len() != 2 || values.iter().any(|value| *value <= 0) {
        return Err(LoweringError::UnsupportedGraph);
    }
    Ok([
        u32::try_from(values[0]).map_err(|_| LoweringError::UnsupportedGraph)?,
        u32::try_from(values[1]).map_err(|_| LoweringError::UnsupportedGraph)?,
    ])
}

fn tensor<'a>(
    analysis: &'a TosaAnalysis<'a>,
    value: ValueId,
) -> Result<virtio_accel_tosa::Tensor<'a>, LoweringError> {
    let AnalyzedValueKind::Tensor(tensor) = analysis.value(value).kind() else {
        return Err(LoweringError::UnsupportedGraph);
    };
    Ok(tensor)
}

fn scalar_zero_point(analysis: &TosaAnalysis<'_>, value: ValueId) -> Result<i32, LoweringError> {
    let tensor = tensor(analysis, value)?;
    if tensor.rank().is_none() || tensor.dimensions().any(|dimension| dimension != 1) {
        return Err(LoweringError::InvalidConstant);
    }
    let bytes = analysis
        .serialized_constant(value)
        .ok_or(LoweringError::InvalidConstant)?;
    match tensor.dtype() {
        DType::FP16 if bytes.len() == 2 => {
            let bits = u16::from_le_bytes(bytes.try_into().expect("length checked"));
            (bits & 0x7fff == 0)
                .then_some(0)
                .ok_or(LoweringError::UnsupportedGraph)
        }
        DType::FP32 if bytes.len() == 4 => {
            let value = f32::from_le_bytes(bytes.try_into().expect("length checked"));
            (value == 0.0)
                .then_some(0)
                .ok_or(LoweringError::UnsupportedGraph)
        }
        DType::INT8 if bytes.len() == 1 => Ok(i32::from(bytes[0] as i8)),
        _ => Err(LoweringError::InvalidConstant),
    }
}

fn set_quantization_offset(
    tensors: &mut [LoweredTensor],
    assigned: &mut Vec<(u32, i32)>,
    value: ValueId,
    zero_point: i32,
) -> Result<(), LoweringError> {
    let tensor = tensors
        .iter_mut()
        .find(|tensor| tensor.value == value.get())
        .ok_or(LoweringError::UnsupportedGraph)?;
    let quantization = tensor
        .quantization
        .as_mut()
        .ok_or(LoweringError::UnsupportedGraph)?;
    let offset = zero_point
        .checked_neg()
        .ok_or(LoweringError::UnsupportedGraph)?;
    if let Some((_, prior)) = assigned
        .iter()
        .find(|(assigned_value, _)| *assigned_value == value.get())
    {
        if *prior != offset {
            return Err(LoweringError::UnsupportedGraph);
        }
    } else {
        assigned
            .try_reserve(1)
            .map_err(|_| LoweringError::ResourceLimit)?;
        assigned.push((value.get(), offset));
    }
    quantization.offset = offset;
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
            if !matches!((operator.op(), index), (Op::MATMUL, 2 | 3) | (Op::MUL, 2)) {
                return false;
            }
        }
    }
    consumed
}

#[cfg(test)]
mod tests {
    use super::*;
    use virtio_accel_conformance::numerics::{
        ADD_FP16, HEXAGON_LOGICAL_CASES, HEXAGON_MOVEMENT_CASES, HEXAGON_REDUCTION_CASES,
        HEXAGON_UNARY_FP16_CASES, IDENTITY_EDGES_FP16, IDENTITY_EDGES_FP32, IDENTITY_FP8E4M3,
        IDENTITY_FP8E5M2, IDENTITY_INT4, IDENTITY_INT8, MATMUL_FP16, MATMUL_FP32, MATMUL_INT8,
        MAX_POOL2D_FP16, MAX_POOL2D_FP32, MAXIMUM_FP16, MINIMUM_FP16, MUL_FP16, POW_FP16, SUB_FP16,
    };

    #[test]
    fn plans_the_complete_initial_fp16_corpus_without_qairt() {
        let cases = [
            ("identity", IDENTITY_EDGES_FP16.artifact, 1usize, 2usize),
            ("matmul", MATMUL_FP16.artifact, 1, 3),
            ("max_pool2d", MAX_POOL2D_FP16.artifact, 1, 2),
        ];
        for (name, artifact, nodes, features) in cases {
            let lowered = lower_tosa(artifact, HEXAGON_TOSA_TARGET)
                .unwrap_or_else(|error| panic!("{name} failed to lower: {error}"));
            assert_eq!(lowered.nodes.len(), nodes, "{name}");
            assert_eq!(lowered.features.len(), features, "{name}");
            assert!(lowered.features.iter().all(|feature| feature.byte_len > 0));
        }
    }

    #[test]
    fn matmul_discards_only_validated_zero_point_parameters() {
        let lowered = lower_tosa(MATMUL_FP16.artifact, HEXAGON_TOSA_TARGET).unwrap();
        assert_eq!(lowered.nodes[0].kind, NodeKind::MatMul);
        assert_eq!(lowered.features[0].slot, 0);
        assert_eq!(lowered.features[1].slot, 1);
        assert_eq!(lowered.features[2].slot, 2);
        assert_eq!(lowered.features[2].role, FeatureRole::Output);
    }

    #[test]
    fn boundary_indices_do_not_depend_on_tensor_declaration_order() {
        let lowered = LoweredModel {
            tensors: vec![
                LoweredTensor {
                    value: 20,
                    element: Element::F16,
                    quantization: None,
                    dims: vec![1],
                    data: None,
                },
                LoweredTensor {
                    value: 10,
                    element: Element::F16,
                    quantization: None,
                    dims: vec![1],
                    data: None,
                },
                LoweredTensor {
                    value: 30,
                    element: Element::F16,
                    quantization: None,
                    dims: vec![1],
                    data: None,
                },
            ],
            nodes: vec![LoweredNode {
                kind: NodeKind::MatMul,
                inputs: vec![10, 20],
                outputs: vec![30],
                parameters: Vec::new(),
            }],
            features: vec![
                LoweredFeature {
                    slot: 0,
                    role: FeatureRole::Input,
                    io_index: 0,
                    value: 10,
                    dims: vec![1],
                    byte_len: 2,
                },
                LoweredFeature {
                    slot: 1,
                    role: FeatureRole::Input,
                    io_index: 1,
                    value: 20,
                    dims: vec![1],
                    byte_len: 2,
                },
                LoweredFeature {
                    slot: 2,
                    role: FeatureRole::Output,
                    io_index: 0,
                    value: 30,
                    dims: vec![1],
                    byte_len: 2,
                },
            ],
            precision: Some(Element::F16),
        };

        assert_eq!(lowered.tensors[0].value, 20);
        assert_eq!(lowered.boundary(20), Some((FeatureRole::Input, 1)));
        assert_eq!(lowered.boundary(10), Some((FeatureRole::Input, 0)));
        assert_eq!(lowered.boundary(30), Some((FeatureRole::Output, 0)));
    }

    #[test]
    fn max_pool_keeps_nhwc_shapes_and_attributes() {
        let lowered = lower_tosa(MAX_POOL2D_FP16.artifact, HEXAGON_TOSA_TARGET).unwrap();
        assert_eq!(lowered.nodes[0].kind, NodeKind::MaxPool2d);
        assert_eq!(lowered.nodes[0].parameters, [2, 2, 2, 2]);
        assert_eq!(lowered.features[0].dims.len(), 4);
        assert_eq!(lowered.features[1].dims.len(), 4);
    }

    #[test]
    fn rejects_fp32_after_htp_precision_probe_detected_fp16_math() {
        for case in [IDENTITY_EDGES_FP32, MATMUL_FP32, MAX_POOL2D_FP32] {
            assert_eq!(
                lower_tosa(case.artifact, HEXAGON_TOSA_TARGET).unwrap_err(),
                LoweringError::UnsupportedType(DType::FP32),
                "{}",
                case.name,
            );
        }
    }

    #[test]
    fn plans_exact_integer_identity_and_matmul_tier() {
        let identity = lower_tosa(IDENTITY_INT8.artifact, HEXAGON_TOSA_INTEGER_TARGET).unwrap();
        assert_eq!(identity.precision, None);
        assert!(
            identity
                .features
                .iter()
                .all(|feature| feature.byte_len == 8)
        );

        let matmul = lower_tosa(MATMUL_INT8.artifact, HEXAGON_TOSA_INTEGER_TARGET).unwrap();
        assert_eq!(matmul.nodes[0].kind, NodeKind::MatMul);
        assert_eq!(matmul.features[0].byte_len, 6);
        assert_eq!(matmul.features[1].byte_len, 6);
        assert_eq!(matmul.features[2].byte_len, 16);
        assert!(matmul.tensors.iter().any(|tensor| {
            tensor.element == Element::I8
                && tensor
                    .quantization
                    .is_some_and(|quantization| quantization.offset != 0)
        }));
    }

    #[test]
    fn integer_target_operator_surface_is_exact() {
        for op in [Op::CONST, Op::IDENTITY, Op::MATMUL] {
            assert!(supports_operator_for_target(op, true), "{op:?}");
        }
        for raw in Op::ARGMAX.get()..=Op::CONST_SHAPE.get() {
            let op = Op::new(raw);
            if !matches!(op, Op::CONST | Op::IDENTITY | Op::MATMUL) {
                assert!(!supports_operator_for_target(op, true), "{op:?}");
            }
        }
    }

    #[test]
    fn descriptor_exposes_hexagon_pool_and_precision_restrictions() {
        assert!(!HEXAGON_TOSA_CAPABILITY.supports_dtype(DType::FP32, ValueRoles::INPUT));
        assert!(HEXAGON_TOSA_CAPABILITY.supports_dtype(DType::FP16, ValueRoles::INPUT));
        assert!(HEXAGON_TOSA_INTEGER_CAPABILITY.supports_dtype(DType::INT8, ValueRoles::INPUT));
        let pool = HEXAGON_TOSA_CAPABILITY.operator(Op::MAX_POOL2D).unwrap();
        assert!(
            pool.constraints
                .contains(OperatorConstraints::PROPAGATING_NAN)
        );
        assert!(pool.constraints.contains(OperatorConstraints::ZERO_PADDING));
    }

    #[test]
    fn rejects_unadvertised_low_precision_profiles_and_extensions() {
        for (case, target) in [
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
            assert_eq!(
                lower_tosa(case.artifact, target).unwrap_err(),
                LoweringError::UnsupportedGraph,
                "{}",
                case.name
            );
        }
    }

    #[test]
    fn rejects_crossed_floating_and_integer_targets() {
        assert!(lower_tosa(IDENTITY_INT8.artifact, HEXAGON_TOSA_TARGET).is_err());
        assert!(lower_tosa(IDENTITY_EDGES_FP16.artifact, HEXAGON_TOSA_INTEGER_TARGET).is_err());
    }

    #[test]
    fn plans_broadcast_binary_fp16_family() {
        for (case, kind) in [
            (ADD_FP16, NodeKind::Add),
            (SUB_FP16, NodeKind::Subtract),
            (MUL_FP16, NodeKind::Multiply),
            (POW_FP16, NodeKind::Power),
            (MAXIMUM_FP16, NodeKind::Maximum),
            (MINIMUM_FP16, NodeKind::Minimum),
        ] {
            let lowered = lower_tosa(case.artifact, HEXAGON_TOSA_TARGET)
                .unwrap_or_else(|error| panic!("{}: {error:?}", case.name));
            assert_eq!(lowered.nodes.len(), 1, "{}", case.name);
            assert_eq!(lowered.nodes[0].kind, kind, "{}", case.name);
            assert_eq!(lowered.nodes[0].inputs.len(), 2, "{}", case.name);
            assert_eq!(lowered.nodes[0].outputs.len(), 1, "{}", case.name);
        }
    }

    #[test]
    fn plans_every_advertised_operator_family_without_qairt() {
        for case in HEXAGON_UNARY_FP16_CASES
            .iter()
            .chain(HEXAGON_LOGICAL_CASES)
            .chain(HEXAGON_REDUCTION_CASES)
            .chain(HEXAGON_MOVEMENT_CASES)
        {
            let lowered = lower_tosa(case.artifact, HEXAGON_TOSA_TARGET)
                .unwrap_or_else(|error| panic!("{}: {error:?}", case.name));
            assert!(!lowered.nodes.is_empty(), "{}", case.name);
            assert_eq!(
                lowered.features.len(),
                case.inputs.len() + 1,
                "{}",
                case.name
            );
        }
    }

    #[test]
    fn advertised_operator_and_dtype_surface_is_exact() {
        let mut shared_count = 0;
        let mut exceptions = Vec::new();
        for raw in Op::ARGMAX.get()..=Op::CONST_SHAPE.get() {
            let op = Op::new(raw);
            let coreml = virtio_accel_coreml::supports_tosa_operator(op);
            let openvino = virtio_accel_openvino::supports_tosa_operator(op);
            assert_eq!(coreml, openvino, "shared providers disagree on {op:?}");
            if !coreml {
                assert!(
                    !supports_tosa_operator(op),
                    "Hexagon alone advertises {op:?}"
                );
                continue;
            }
            shared_count += 1;
            if !supports_tosa_operator(op) {
                exceptions.push(op);
            }
        }
        assert_eq!(shared_count, 42);
        assert_eq!(exceptions, [Op::ERF]);
        for dtype in [DType::BOOL, DType::FP16, DType::INT8, DType::INT32] {
            assert!(supports_tosa_dtype(dtype), "{dtype:?}");
        }
        for dtype in [DType::FP32, DType::INT4, DType::FP8E4M3, DType::FP8E5M2] {
            assert!(!supports_tosa_dtype(dtype), "{dtype:?}");
        }
    }

    #[test]
    fn converts_binary16_attributes_without_losing_special_values() {
        assert_eq!(f16_to_f32(0x0000).to_bits(), 0x0000_0000);
        assert_eq!(f16_to_f32(0x8000).to_bits(), 0x8000_0000);
        assert_eq!(f16_to_f32(0x0001).to_bits(), 0x3380_0000);
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0x7c00), f32::INFINITY);
        assert_eq!(f16_to_f32(0xfc00), f32::NEG_INFINITY);
        assert!(f16_to_f32(0x7e00).is_nan());
    }

    #[test]
    fn rejects_malformed_artifacts_and_storage_overflow_before_native_work() {
        let truncated = &IDENTITY_EDGES_FP16.artifact[..IDENTITY_EDGES_FP16.artifact.len() / 2];
        assert!(matches!(
            lower_tosa(truncated, HEXAGON_TOSA_TARGET),
            Err(LoweringError::Parse(_))
        ));
        assert_eq!(
            checked_tensor_byte_len(Element::I32, &[u32::MAX, u32::MAX, u32::MAX]),
            Err(LoweringError::ResourceLimit)
        );
    }
}
