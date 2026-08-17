//! Strict TOSA 1.0 admission and provider-local QNN graph planning.
//!
//! This module deliberately contains no QNN ABI types. It turns a verified TOSA artifact into
//! owned tensor, binding, and operation metadata that the native boundary can translate while the
//! source FlatBuffer is no longer borrowed. It compiles and tests on hosts without QAIRT.

#![cfg_attr(not(va_hexagon), allow(dead_code))]

use std::fmt;

use virtio_accel_tosa::{
    AnalysisError, AnalyzedValueKind, DType, Error as ParseError, ExtensionSet, Level,
    NanPropagationMode, Op, OpAttributes, ProfileSet, Target, TosaAnalysis, ValueId, Version,
    parse,
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
    matches!(op, Op::IDENTITY | Op::MATMUL | Op::MAX_POOL2D)
}

/// Whether the first hardware tier may expose `dtype` at a model boundary.
///
/// FP32 remains deliberately rejected because current HTP floating-point execution may use FP16
/// math. Integer and packed low-precision tiers require separate targets and evidence.
pub const fn supports_tosa_dtype(dtype: DType) -> bool {
    matches!(dtype, DType::FP16 | DType::INT8 | DType::INT32)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Element {
    F16,
    F32,
    I8,
    I32,
}

impl Element {
    pub(crate) const fn scalar_bytes(self) -> u64 {
        match self {
            Self::F16 => 2,
            Self::F32 | Self::I32 => 4,
            Self::I8 => 1,
        }
    }

    fn for_dtype(dtype: DType) -> Result<Self, LoweringError> {
        match dtype {
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
}

/// One operation from the accepted initial QNN lowering subset.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum LoweredNode {
    Identity {
        input: u32,
        output: u32,
    },
    MatMul {
        left: u32,
        right: u32,
        output: u32,
    },
    MaxPool2d {
        input: u32,
        output: u32,
        kernel: [u32; 2],
        stride: [u32; 2],
    },
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
        || !analysis.conditions().is_empty()
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
            return Err(LoweringError::UnsupportedGraph);
        };
        let element = Element::for_dtype(tensor.dtype())?;
        tensors.push(LoweredTensor {
            value: value.id().get(),
            element,
            quantization: matches!(element, Element::I8 | Element::I32).then_some(Quantization {
                scale: 1.0,
                offset: 0,
            }),
            dims: static_dims(tensor, true)?,
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
        let op_inputs = analysis.operator_inputs(*operator_id);
        let op_outputs = analysis.operator_outputs(*operator_id);
        match op {
            Op::CONST => {
                if op_outputs.len() != 1 || !constant_is_matmul_zero_point(&analysis, op_outputs[0])
                {
                    return Err(LoweringError::UnsupportedGraph);
                }
            }
            Op::IDENTITY => {
                require_arity(op_inputs, 1, op_outputs, 1)?;
                nodes.push(LoweredNode::Identity {
                    input: op_inputs[0].get(),
                    output: op_outputs[0].get(),
                });
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
                nodes.push(LoweredNode::MatMul {
                    left: op_inputs[0].get(),
                    right: op_inputs[1].get(),
                    output: op_outputs[0].get(),
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
                nodes.push(LoweredNode::MaxPool2d {
                    input: op_inputs[0].get(),
                    output: op_outputs[0].get(),
                    kernel,
                    stride,
                });
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
            return Err(LoweringError::UnsupportedGraph);
        };
        let supported = if integer {
            matches!(tensor.dtype(), DType::INT8 | DType::INT32)
        } else {
            tensor.dtype() == DType::FP16
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
    let mut byte_len = element.scalar_bytes();
    for dim in &dims {
        byte_len = byte_len
            .checked_mul(u64::from(*dim))
            .ok_or(LoweringError::ResourceLimit)?;
    }
    Ok(LoweredFeature {
        slot: u32::try_from(slot).map_err(|_| LoweringError::ResourceLimit)?,
        role,
        io_index: u32::try_from(io_index).map_err(|_| LoweringError::ResourceLimit)?,
        value: value.get(),
        dims,
        byte_len,
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

fn constant_is_matmul_zero_point(analysis: &TosaAnalysis<'_>, value: ValueId) -> bool {
    let mut consumed = false;
    for operator in analysis.operators() {
        for (index, input) in analysis.operator_inputs(operator.id()).iter().enumerate() {
            if *input != value {
                continue;
            }
            consumed = true;
            if operator.op() != Op::MATMUL || !matches!(index, 2 | 3) {
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
        IDENTITY_EDGES_FP16, IDENTITY_EDGES_FP32, IDENTITY_FP8E4M3, IDENTITY_FP8E5M2,
        IDENTITY_INT4, IDENTITY_INT8, MATMUL_FP16, MATMUL_FP32, MATMUL_INT8, MAX_POOL2D_FP16,
        MAX_POOL2D_FP32,
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
        assert!(matches!(
            lowered.nodes.as_slice(),
            [LoweredNode::MatMul { .. }]
        ));
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
                },
                LoweredTensor {
                    value: 10,
                    element: Element::F16,
                    quantization: None,
                    dims: vec![1],
                },
                LoweredTensor {
                    value: 30,
                    element: Element::F16,
                    quantization: None,
                    dims: vec![1],
                },
            ],
            nodes: vec![LoweredNode::MatMul {
                left: 10,
                right: 20,
                output: 30,
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
        assert!(matches!(
            lowered.nodes.as_slice(),
            [LoweredNode::MaxPool2d {
                kernel: [2, 2],
                stride: [2, 2],
                ..
            }]
        ));
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
        assert!(matches!(
            matmul.nodes.as_slice(),
            [LoweredNode::MatMul { .. }]
        ));
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
    fn advertised_operator_and_dtype_surface_is_exact() {
        for op in [Op::IDENTITY, Op::MATMUL, Op::MAX_POOL2D] {
            assert!(supports_tosa_operator(op));
        }
        assert!(!supports_tosa_operator(Op::ADD));
        for dtype in [DType::FP16, DType::INT8, DType::INT32] {
            assert!(supports_tosa_dtype(dtype), "{dtype:?}");
        }
        for dtype in [DType::FP32, DType::INT4, DType::FP8E4M3, DType::FP8E5M2] {
            assert!(!supports_tosa_dtype(dtype), "{dtype:?}");
        }
    }
}
