//! TOSA admission for the Vulkan backend: the advertised targets, the capability descriptor, and
//! the hardware-free lowering of an admitted graph into a kernel plan.
//!
//! This module compiles and unit-tests on every host. It owns every decision about which TOSA
//! graphs the backend executes; `native` only turns an accepted [`ProgramPlan`] into Vulkan
//! objects. The FP32 base tier is the only advertised tier today (ADR 0004): it admits the
//! single-operator IDENTITY graph that ticket 8 of the
//! [Vulkan wayfinder map](https://github.com/MicroPerceptron/virtio-accel/issues/154) proves
//! end-to-end. The provisional integer target constant stays declared but admits nothing until
//! its per-device gating is ratified.

// Builds forced to the placeholder (`VIRTIO_ACCEL_VULKAN=0`, or an OS outside the loader host
// set) still type-check and unit-test this admission path; only the native module calls it.
#![cfg_attr(not(va_vulkan), allow(dead_code))]

use std::fmt;

use virtio_accel_tosa::{
    AnalysisError, AnalyzedValueKind, CapabilityDescriptor, DType, DTypeCapability,
    Error as ParseError, ExtensionSet, GraphCapabilities, Level, Op, OperatorCapability,
    OperatorId, ProfileSet, RuntimeConditionSupport, Target, TosaAnalysis, ValueId, ValueRoles,
    Version, parse,
};

/// The FP32 base tier: TOSA 1.0, floating-point profile, level 8K, no extensions.
pub const VULKAN_TOSA_TARGET: Target = Target::new(
    Version::TOSA_1_0,
    ProfileSet::FLOATING_POINT,
    Level::Level8K,
    ExtensionSet::NONE,
);

/// The provisional integer tier: TOSA 1.0, integer profile, level 8K, no extensions.
///
/// Declared (ADR 0004) but not yet advertised: no capability descriptor names it and admission
/// rejects it until `shaderInt8` gating and the operator subset table close wayfinder ticket 5.
pub const VULKAN_TOSA_INTEGER_TARGET: Target = Target::new(
    Version::TOSA_1_0,
    ProfileSet::INTEGER,
    Level::Level8K,
    ExtensionSet::NONE,
);

const FLOAT_DTYPES: &[DTypeCapability] = &[DTypeCapability::new(DType::FP32, ValueRoles::ALL)];

const FLOAT_OPERATORS: &[OperatorCapability] = &[
    OperatorCapability::new(Op::IDENTITY),
    OperatorCapability::new(Op::MATMUL),
];

/// The FP32 base tier's admitted boundary: exactly what the checked-in shaders execute.
pub const VULKAN_TOSA_CAPABILITY: CapabilityDescriptor = CapabilityDescriptor {
    target: VULKAN_TOSA_TARGET,
    dtypes: FLOAT_DTYPES,
    operators: FLOAT_OPERATORS,
    graph: GraphCapabilities {
        max_regions: 1,
        max_blocks: 1,
        dynamic_shapes: false,
        runtime_conditions: RuntimeConditionSupport::None,
    },
};

/// Whether the FP32 tier admits `op`.
pub const fn supports_tosa_operator(op: Op) -> bool {
    VULKAN_TOSA_CAPABILITY.supports_operator(op)
}

/// Whether the FP32 tier exposes `dtype` at a program boundary.
pub const fn supports_tosa_dtype(dtype: DType) -> bool {
    VULKAN_TOSA_CAPABILITY.supports_dtype(dtype, ValueRoles::INPUT)
        || VULKAN_TOSA_CAPABILITY.supports_dtype(dtype, ValueRoles::OUTPUT)
}

/// Why an artifact was not admitted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LoweringError {
    Parse(ParseError),
    Analysis(AnalysisError),
    /// The target is not one this backend advertises.
    UnsupportedTarget,
    /// The graph shape (regions, blocks, boundary, operator structure) is outside the tier.
    UnsupportedGraph,
    UnsupportedType(DType),
    UnsupportedOperator(Op),
    /// A static shape does not fit the kernel's 32-bit element domain.
    ResourceLimit,
}

impl fmt::Display for LoweringError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for LoweringError {}

/// The checked-in kernel an admitted graph maps onto.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Kernel {
    /// Elementwise 32-bit word copy (`shader::copy_u32_spirv`).
    CopyU32,
    /// FP32 batched matrix multiplication (`shader::matmul_fp32_spirv`).
    MatmulFp32,
}

/// One binding slot of an admitted program.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SlotPlan {
    pub slot: u32,
    pub role: SlotRole,
    /// Exact tensor bytes: the required length of a binding over this slot.
    pub byte_len: u64,
    /// Storage bytes per scalar; a bound range must start scalar-aligned within its buffer.
    pub scalar_bytes: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SlotRole {
    Input,
    Output,
}

/// A hardware-free execution plan for one admitted TOSA graph.
///
/// Slots follow the workspace convention shared with the other TOSA backends: block inputs take
/// slots `0..inputs`, block outputs follow in declared order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ProgramPlan {
    pub kernel: Kernel,
    /// Elements processed by one invocation domain; the shader's specialization constant.
    pub element_count: u32,
    pub slots: Vec<SlotPlan>,
    /// MATMUL geometry: (batch, m, n, k). Zero for non-MATMUL kernels.
    pub matmul: Option<MatmulGeometry>,
}

/// Batched matrix-multiplication dimensions captured from an admitted graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct MatmulGeometry {
    pub batch: u32,
    pub m: u32,
    pub n: u32,
    pub k: u32,
}

#[cfg(test)]
impl ProgramPlan {
    fn slot(&self, slot: u32) -> Option<&SlotPlan> {
        self.slots.iter().find(|plan| plan.slot == slot)
    }
}

/// Admit `bytes` for `target` and produce its plan, or explain the rejection.
pub(crate) fn lower_tosa(bytes: &[u8], target: Target) -> Result<ProgramPlan, LoweringError> {
    if target != VULKAN_TOSA_TARGET {
        return Err(LoweringError::UnsupportedTarget);
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
    let order = analysis.execution_order(block);

    // The tier admits graphs whose execution order carries `CONST` producers for constant
    // parameters plus exactly one boundary-to-boundary compute operator. Everything else is
    // rejected here, before any Vulkan object exists, with the reason a host can act on.
    let compute_operators: Vec<_> = order
        .iter()
        .filter(|operator_id| analysis.operator(**operator_id).op() != Op::CONST)
        .collect();
    let [operator_id] = compute_operators[..] else {
        return Err(LoweringError::UnsupportedGraph);
    };
    let operator = analysis.operator(*operator_id);
    if !supports_tosa_operator(operator.op()) {
        return Err(LoweringError::UnsupportedOperator(operator.op()));
    }
    match operator.op() {
        Op::IDENTITY => lower_identity(&analysis, *operator_id, inputs, outputs),
        Op::MATMUL => lower_matmul(&analysis, *operator_id, inputs, outputs),
        other => Err(LoweringError::UnsupportedOperator(other)),
    }
}

fn lower_identity(
    analysis: &TosaAnalysis<'_>,
    operator_id: OperatorId,
    inputs: &[ValueId],
    outputs: &[ValueId],
) -> Result<ProgramPlan, LoweringError> {
    let ([input], [output]) = (inputs, outputs) else {
        return Err(LoweringError::UnsupportedGraph);
    };
    if analysis.operator_inputs(operator_id) != [*input]
        || analysis.operator_outputs(operator_id) != [*output]
    {
        return Err(LoweringError::UnsupportedGraph);
    }

    let input_tensor = boundary_tensor(analysis, *input)?;
    let output_tensor = boundary_tensor(analysis, *output)?;
    if input_tensor != output_tensor {
        return Err(LoweringError::UnsupportedGraph);
    }
    let element_count =
        u32::try_from(input_tensor.elements).map_err(|_| LoweringError::ResourceLimit)?;
    let byte_len = input_tensor
        .elements
        .checked_mul(input_tensor.scalar_bytes)
        .ok_or(LoweringError::ResourceLimit)?;
    Ok(ProgramPlan {
        kernel: Kernel::CopyU32,
        element_count,
        slots: vec![
            SlotPlan {
                slot: 0,
                role: SlotRole::Input,
                byte_len,
                scalar_bytes: input_tensor.scalar_bytes,
            },
            SlotPlan {
                slot: 1,
                role: SlotRole::Output,
                byte_len,
                scalar_bytes: input_tensor.scalar_bytes,
            },
        ],
        matmul: None,
    })
}

fn lower_matmul(
    analysis: &TosaAnalysis<'_>,
    operator_id: OperatorId,
    inputs: &[ValueId],
    outputs: &[ValueId],
) -> Result<ProgramPlan, LoweringError> {
    let ([lhs, rhs], [output]) = (inputs, outputs) else {
        return Err(LoweringError::UnsupportedGraph);
    };
    let operator_inputs = analysis.operator_inputs(operator_id);
    if operator_inputs.len() != 4
        || operator_inputs[0] != *lhs
        || operator_inputs[1] != *rhs
        || analysis.operator_outputs(operator_id) != [*output]
    {
        return Err(LoweringError::UnsupportedGraph);
    }
    // TOSA 1.0 FP32 MATMUL admits only zero zero-points: the two trailing inputs must be
    // `CONST` tensors whose serialized payload is all-zero (signed zero included).
    for zero_point in &operator_inputs[2..4] {
        let bytes = analysis
            .serialized_constant(*zero_point)
            .ok_or(LoweringError::UnsupportedGraph)?;
        if !serialized_float_is_zero(bytes) {
            return Err(LoweringError::UnsupportedGraph);
        }
    }

    let lhs_tensor = boundary_tensor(analysis, *lhs)?;
    let rhs_tensor = boundary_tensor(analysis, *rhs)?;
    let output_tensor = boundary_tensor(analysis, *output)?;

    let [batch_l, m, k] = lhs_tensor.dims;
    let [batch_r, k_r, n] = rhs_tensor.dims;
    let [batch_o, m_o, n_o] = output_tensor.dims;
    if batch_l != batch_r || batch_l != batch_o || k != k_r || m != m_o || n != n_o {
        return Err(LoweringError::UnsupportedGraph);
    }

    let batch = u32::try_from(batch_l).map_err(|_| LoweringError::ResourceLimit)?;
    let m = u32::try_from(m).map_err(|_| LoweringError::ResourceLimit)?;
    let n = u32::try_from(n).map_err(|_| LoweringError::ResourceLimit)?;
    let k = u32::try_from(k).map_err(|_| LoweringError::ResourceLimit)?;

    let lhs_bytes = lhs_tensor
        .elements
        .checked_mul(lhs_tensor.scalar_bytes)
        .ok_or(LoweringError::ResourceLimit)?;
    let rhs_bytes = rhs_tensor
        .elements
        .checked_mul(rhs_tensor.scalar_bytes)
        .ok_or(LoweringError::ResourceLimit)?;
    let output_bytes = output_tensor
        .elements
        .checked_mul(output_tensor.scalar_bytes)
        .ok_or(LoweringError::ResourceLimit)?;

    Ok(ProgramPlan {
        kernel: Kernel::MatmulFp32,
        element_count: batch
            .checked_mul(m)
            .and_then(|v| v.checked_mul(n))
            .ok_or(LoweringError::ResourceLimit)?,
        slots: vec![
            SlotPlan {
                slot: 0,
                role: SlotRole::Input,
                byte_len: lhs_bytes,
                scalar_bytes: lhs_tensor.scalar_bytes,
            },
            SlotPlan {
                slot: 1,
                role: SlotRole::Input,
                byte_len: rhs_bytes,
                scalar_bytes: rhs_tensor.scalar_bytes,
            },
            SlotPlan {
                slot: 2,
                role: SlotRole::Output,
                byte_len: output_bytes,
                scalar_bytes: output_tensor.scalar_bytes,
            },
        ],
        matmul: Some(MatmulGeometry { batch, m, n, k }),
    })
}

/// Whether a serialized FP32 constant payload is zero (any sign) byte for byte.
fn serialized_float_is_zero(bytes: &[u8]) -> bool {
    bytes.len() == 4
        && u32::from_le_bytes(bytes.try_into().expect("length checked")) & 0x7fff_ffff == 0
}

/// Static shape summary of a boundary tensor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct BoundaryTensor {
    dtype: DType,
    elements: u64,
    scalar_bytes: u64,
    dims: [u64; 3],
}

fn boundary_tensor(
    analysis: &TosaAnalysis<'_>,
    value: ValueId,
) -> Result<BoundaryTensor, LoweringError> {
    let AnalyzedValueKind::Tensor(tensor) = analysis.value(value).kind() else {
        return Err(LoweringError::UnsupportedGraph);
    };
    let dtype = tensor.dtype();
    if !supports_tosa_dtype(dtype) {
        return Err(LoweringError::UnsupportedType(dtype));
    }
    let scalar_bytes = match dtype {
        DType::FP32 => 4,
        other => return Err(LoweringError::UnsupportedType(other)),
    };
    let rank = tensor.rank().ok_or(LoweringError::UnsupportedGraph)?;
    if rank == 0 {
        return Err(LoweringError::UnsupportedGraph);
    }
    let mut elements = 1_u64;
    let mut dims = [1_u64; 3];
    for (index, dimension) in tensor.dimensions().enumerate() {
        if index >= 3 {
            // Rank > 3 has no slot in the checked-in kernels.
            return Err(LoweringError::UnsupportedGraph);
        }
        let dimension = u64::try_from(dimension)
            .ok()
            .filter(|dimension| *dimension > 0)
            .ok_or(LoweringError::UnsupportedGraph)?;
        dims[index] = dimension;
        elements = elements
            .checked_mul(dimension)
            .ok_or(LoweringError::ResourceLimit)?;
    }
    Ok(BoundaryTensor {
        dtype,
        elements,
        scalar_bytes,
        dims,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use virtio_accel_conformance::numerics::{
        IDENTITY_EDGES_FP16, IDENTITY_EDGES_FP32, IDENTITY_INT8, MATMUL_FP32, MAX_POOL2D_FP32,
    };

    const IDENTITY_FP32_LOCAL: &[u8] = include_bytes!("../tests/data/identity-fp32-v1.0.0.tosa");

    #[test]
    fn targets_validate_and_round_trip() {
        for target in [VULKAN_TOSA_TARGET, VULKAN_TOSA_INTEGER_TARGET] {
            assert_eq!(target.validate(), Ok(target));
            assert_eq!(Target::from_identity(target.to_identity()), Ok(target));
        }
        assert_ne!(VULKAN_TOSA_TARGET, VULKAN_TOSA_INTEGER_TARGET);
    }

    #[test]
    fn capability_names_exactly_the_executed_boundary() {
        assert!(supports_tosa_operator(Op::IDENTITY));
        assert!(supports_tosa_operator(Op::MATMUL));
        assert!(!supports_tosa_operator(Op::MAX_POOL2D));
        assert!(supports_tosa_dtype(DType::FP32));
        assert!(!supports_tosa_dtype(DType::FP16));
        assert!(!supports_tosa_dtype(DType::INT8));
        assert_eq!(VULKAN_TOSA_CAPABILITY.target, VULKAN_TOSA_TARGET);
    }

    #[test]
    fn lowers_the_local_fp32_identity_artifact() {
        let plan = lower_tosa(IDENTITY_FP32_LOCAL, VULKAN_TOSA_TARGET).unwrap();
        assert_eq!(plan.kernel, Kernel::CopyU32);
        assert_eq!(plan.element_count, 1);
        assert_eq!(plan.slots.len(), 2);
        assert_eq!(plan.slot(0).unwrap().role, SlotRole::Input);
        assert_eq!(plan.slot(1).unwrap().role, SlotRole::Output);
        assert_eq!(plan.slot(0).unwrap().byte_len, 4);
        assert_eq!(plan.slot(1).unwrap().scalar_bytes, 4);
        assert!(plan.slot(2).is_none());
    }

    #[test]
    fn lowers_the_shared_fp32_edge_identity_artifact() {
        let plan = lower_tosa(IDENTITY_EDGES_FP32.artifact, VULKAN_TOSA_TARGET).unwrap();
        let expected = IDENTITY_EDGES_FP32.inputs[0].values.len();
        assert_eq!(plan.element_count as usize, expected);
        assert_eq!(plan.slot(1).unwrap().byte_len as usize, expected * 4);
    }

    #[test]
    fn rejects_other_targets_before_parsing() {
        assert_eq!(
            lower_tosa(IDENTITY_FP32_LOCAL, VULKAN_TOSA_INTEGER_TARGET),
            Err(LoweringError::UnsupportedTarget)
        );
        assert_eq!(
            lower_tosa(IDENTITY_INT8.artifact, VULKAN_TOSA_INTEGER_TARGET),
            Err(LoweringError::UnsupportedTarget)
        );
    }

    #[test]
    fn rejects_mistyped_identity_graphs_loudly() {
        // FP16 identity: the FP16 tier is deliberately undeclared (ADR 0004).
        assert!(matches!(
            lower_tosa(IDENTITY_EDGES_FP16.artifact, VULKAN_TOSA_TARGET),
            Err(LoweringError::UnsupportedType(DType::FP16) | LoweringError::Analysis(_))
        ));
        // INT8 identity under the floating-point target: never relabeled.
        assert!(matches!(
            lower_tosa(IDENTITY_INT8.artifact, VULKAN_TOSA_TARGET),
            Err(LoweringError::UnsupportedType(DType::INT8) | LoweringError::Analysis(_))
        ));
    }

    #[test]
    fn rejects_operators_outside_the_tier() {
        // MAX_POOL2D is not yet admitted: the FP32 base tier grows one operator at a time.
        assert!(matches!(
            lower_tosa(MAX_POOL2D_FP32.artifact, VULKAN_TOSA_TARGET),
            Err(LoweringError::UnsupportedOperator(Op::MAX_POOL2D))
        ));
    }

    #[test]
    fn lowers_the_shared_fp32_matmul_artifact() {
        let plan = lower_tosa(MATMUL_FP32.artifact, VULKAN_TOSA_TARGET).unwrap();
        assert_eq!(plan.kernel, Kernel::MatmulFp32);
        let geometry = plan.matmul.expect("MATMUL plan carries geometry");
        assert_eq!(geometry.batch, 1);
        assert_eq!(geometry.m, 2);
        assert_eq!(geometry.n, 2);
        assert_eq!(geometry.k, 3);
        assert_eq!(plan.element_count, 4);
        assert_eq!(plan.slots.len(), 3);
        assert_eq!(plan.slot(0).unwrap().role, SlotRole::Input);
        assert_eq!(plan.slot(0).unwrap().byte_len, 6 * 4);
        assert_eq!(plan.slot(1).unwrap().role, SlotRole::Input);
        assert_eq!(plan.slot(1).unwrap().byte_len, 6 * 4);
        assert_eq!(plan.slot(2).unwrap().role, SlotRole::Output);
        assert_eq!(plan.slot(2).unwrap().byte_len, 4 * 4);
        assert_eq!(plan.slot(2).unwrap().scalar_bytes, 4);
        assert!(plan.slot(3).is_none());
    }

    #[test]
    fn rejects_garbage_as_a_parse_error() {
        assert!(matches!(
            lower_tosa(b"not a flatbuffer", VULKAN_TOSA_TARGET),
            Err(LoweringError::Parse(_))
        ));
    }
}
