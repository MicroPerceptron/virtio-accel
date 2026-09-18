//! TOSA admission for the Vulkan backend: the advertised targets, the capability descriptor, and
//! the hardware-free lowering of an admitted graph into a [`ProgramPlan`] of kernel dispatches.
//!
//! This module compiles and unit-tests on every host. It owns every decision about which TOSA
//! graphs the backend executes and every index computation the kernels will perform; `native`
//! only turns an accepted plan into Vulkan objects. Because the kernels address storage buffers
//! with the geometry planned here and no robust-buffer-access mode is relied upon, every shape,
//! axis, permutation, and pooling window is re-derived and checked against the declared tensor
//! shapes before a plan is produced: a graph whose declared shapes disagree with its operator
//! semantics is rejected, never dispatched.
//!
//! The FP32 tier (ADR 0004, ADR 0007) admits static single-block graphs over the 42 operators the
//! Core ML and OpenVINO backends share, with `BOOL` and `INT32` auxiliaries where TOSA defines
//! them; the FP16 tier (ADR 0008) admits the same graphs over binary16 tensors on every device —
//! its conversions are crate-owned integer and binary32 code, so it needs no device feature.
//! `CONST` tensors and intermediates live in a per-program arena; `RESHAPE` and `IDENTITY` of
//! arena tensors are views, not copies. The provisional integer target stays declared but admits
//! nothing until its per-device gating is ratified.

// Builds forced to the placeholder (`VIRTIO_ACCEL_VULKAN=0`, or an OS outside the loader host
// set) still type-check and unit-test this admission path; only the native module calls it.
#![cfg_attr(not(va_vulkan), allow(dead_code))]

use std::collections::HashMap;
use std::fmt;

use virtio_accel_tosa::{
    AnalysisError, AnalyzedValueKind, CapabilityDescriptor, DType, DTypeCapability,
    DTypeConstraints, Error as ParseError, ExtensionSet, GraphCapabilities, Level,
    NanPropagationMode, Op, OpAttributes, OperatorCapability, OperatorConstraints, OperatorId,
    OptimizationHints, ProfileSet, RuntimeCondition, RuntimeConditionSupport, Target, TosaAnalysis,
    ValueId, ValueRoles, Version, parse,
};

use crate::shader::{
    ElementwiseOp, ElementwiseSpec, Fp8Format, MAX_RANK, MoveGeometry, NanMode, Operand,
    PoolGeometry, ReduceOp, Storage, matmul_spec, max_pool_spec, move_spec, reduce_spec,
};

/// The FP32 base tier: TOSA 1.0, floating-point profile, level 8K, no extensions.
pub const VULKAN_TOSA_TARGET: Target = Target::new(
    Version::TOSA_1_0,
    ProfileSet::FLOATING_POINT,
    Level::Level8K,
    ExtensionSet::NONE,
);

/// The FP8 tier's target: TOSA 1.0, floating-point profile, level 8K, both FP8 extensions.
///
/// Unlike the FP16 tier, this cannot share [`VULKAN_TOSA_TARGET`]'s identity. FP16 lives in the
/// base floating-point profile, so narrowing dtypes under one target was enough (ADR 0008); FP8
/// legality is gated on `ExtensionSet::FP8E4M3` / `FP8E5M2`, so the envelope itself differs and
/// the tier needs a target of its own — the same split `virtio-accel-xdna` makes (ADR 0009).
pub const VULKAN_TOSA_FP8_TARGET: Target = Target::new(
    Version::TOSA_1_0,
    ProfileSet::FLOATING_POINT,
    Level::Level8K,
    ExtensionSet::NONE
        .union(ExtensionSet::FP8E4M3)
        .union(ExtensionSet::FP8E5M2),
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

const FLOAT_DTYPES: &[DTypeCapability] = &[
    DTypeCapability::new(DType::FP32, ValueRoles::ALL),
    DTypeCapability::new(DType::BOOL, ValueRoles::ALL),
    DTypeCapability::new(DType::INT32, ValueRoles::ALL),
    // The TOSA 1.0 `MUL` shift operand is an INT8 constant consumed at admission (it must be
    // zero); INT8 never becomes a graph-visible provider tensor.
    DTypeCapability::constrained(
        DType::INT8,
        ValueRoles::CONSTANT,
        DTypeConstraints::PARAMETER_ONLY,
    ),
];

/// The FP16 tier's dtypes: the FP32 tier plus binary16 tensors in every role. FP16 shares the
/// floating-point target identity with FP32 (the tier is dtype-narrowed, exactly the Hexagon
/// pattern), and it is the descriptor every native instance advertises (ADR 0008).
const FLOAT16_DTYPES: &[DTypeCapability] = &[
    DTypeCapability::new(DType::FP32, ValueRoles::ALL),
    DTypeCapability::new(DType::FP16, ValueRoles::ALL),
    DTypeCapability::new(DType::BOOL, ValueRoles::ALL),
    DTypeCapability::new(DType::INT32, ValueRoles::ALL),
    DTypeCapability::constrained(
        DType::INT8,
        ValueRoles::CONSTANT,
        DTypeConstraints::PARAMETER_ONLY,
    ),
];

/// The 42 operators shared with the Core ML and OpenVINO FP32 tiers. NaN modes and pool padding
/// are unconstrained: the kernels implement both `PROPAGATE` and `IGNORE` literally and exclude
/// padded taps from the window.
const FLOAT_OPERATORS: &[OperatorCapability] = &[
    OperatorCapability::new(Op::ARGMAX),
    OperatorCapability::constrained(Op::MATMUL, OperatorConstraints::ZERO_ZERO_POINTS),
    OperatorCapability::new(Op::MAX_POOL2D),
    OperatorCapability::new(Op::CLAMP),
    OperatorCapability::new(Op::ERF),
    OperatorCapability::new(Op::SIGMOID),
    OperatorCapability::new(Op::TANH),
    OperatorCapability::new(Op::ADD),
    OperatorCapability::new(Op::LOGICAL_AND),
    OperatorCapability::new(Op::LOGICAL_OR),
    OperatorCapability::new(Op::LOGICAL_XOR),
    OperatorCapability::new(Op::MAXIMUM),
    OperatorCapability::new(Op::MINIMUM),
    OperatorCapability::constrained(Op::MUL, OperatorConstraints::ZERO_SHIFT),
    OperatorCapability::new(Op::POW),
    OperatorCapability::new(Op::SUB),
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
    OperatorCapability::new(Op::REDUCE_MAX),
    OperatorCapability::new(Op::REDUCE_MIN),
    OperatorCapability::new(Op::REDUCE_PRODUCT),
    OperatorCapability::new(Op::REDUCE_SUM),
    OperatorCapability::new(Op::CONCAT),
    OperatorCapability::constrained(Op::RESHAPE, OperatorConstraints::CONSTANT_PARAMETERS),
    OperatorCapability::new(Op::REVERSE),
    OperatorCapability::new(Op::TRANSPOSE),
    OperatorCapability::new(Op::CONST),
    OperatorCapability::new(Op::CONST_SHAPE),
    OperatorCapability::new(Op::IDENTITY),
];

/// The FP32 tier's admitted boundary: exactly what the generated kernels execute.
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

/// The FP16 tier's admitted boundary (ADR 0008): the same 42 operators and graph envelope with
/// binary16 tensors in every role. The tier needs no device feature — the kernels' binary16
/// conversions are crate-owned integer and binary32 code, the implementation choice TOSA 1.0
/// §1.10.3 names explicitly — so it is advertised on every device the backend opens, with
/// numerics identical to the FP32 tier's everywhere.
pub const VULKAN_TOSA_FP16_CAPABILITY: CapabilityDescriptor = CapabilityDescriptor {
    target: VULKAN_TOSA_TARGET,
    dtypes: FLOAT16_DTYPES,
    operators: FLOAT_OPERATORS,
    graph: GraphCapabilities {
        max_regions: 1,
        max_blocks: 1,
        dynamic_shapes: false,
        runtime_conditions: RuntimeConditionSupport::None,
    },
};

/// The FP8 tier's dtypes: both FP8 encodings in every role, plus the FP16 that TOSA assigns as
/// the result of an FP8 MATMUL, and INT32 for shape operands.
const FLOAT8_DTYPES: &[DTypeCapability] = &[
    DTypeCapability::new(DType::FP8E4M3, ValueRoles::ALL),
    DTypeCapability::new(DType::FP8E5M2, ValueRoles::ALL),
    DTypeCapability::new(DType::FP16, ValueRoles::ALL),
    DTypeCapability::new(DType::INT32, ValueRoles::ALL),
];

/// The FP8 tier's operators: what TOSA admits for FP8 *and* this crate executes. Deliberately a
/// subset, not [`FLOAT_OPERATORS`] — TOSA admits no FP8 elementwise operator at all (no
/// arithmetic, comparison, selection, reduction or transcendental lane takes FP8), so the tier
/// is MATMUL plus the data movement that feeds it. A capability descriptor cannot express
/// per-operator dtype legality, so listing the 42-operator table here would advertise FP8 lanes
/// that admission would then reject.
///
/// `CAST`, `MAX_POOL2D` and `ARGMAX` are admitted by TOSA for FP8 and are deliberately absent:
/// they need kernels this tier does not yet carry.
const FLOAT8_OPERATORS: &[OperatorCapability] = &[
    OperatorCapability::constrained(Op::MATMUL, OperatorConstraints::ZERO_ZERO_POINTS),
    OperatorCapability::new(Op::CONCAT),
    OperatorCapability::constrained(Op::RESHAPE, OperatorConstraints::CONSTANT_PARAMETERS),
    OperatorCapability::new(Op::REVERSE),
    OperatorCapability::new(Op::TRANSPOSE),
    OperatorCapability::new(Op::CONST),
    OperatorCapability::new(Op::CONST_SHAPE),
    OperatorCapability::new(Op::IDENTITY),
];

/// The FP8 tier's admitted boundary (ADR 0009): `(FP8, FP8) -> FP16` MATMUL and exact FP8 data
/// movement. Like the FP16 tier it needs no device feature — the widening is crate-owned integer
/// and binary32 code and nothing writes FP8 except a raw byte copy — so it is advertised on
/// every device the backend opens, with numerics identical everywhere.
pub const VULKAN_TOSA_FP8_CAPABILITY: CapabilityDescriptor = CapabilityDescriptor {
    target: VULKAN_TOSA_FP8_TARGET,
    dtypes: FLOAT8_DTYPES,
    operators: FLOAT8_OPERATORS,
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

/// Whether the advertised FP16 tier exposes `dtype` at a program boundary.
pub const fn supports_tosa_dtype(dtype: DType) -> bool {
    VULKAN_TOSA_FP16_CAPABILITY.supports_dtype(dtype, ValueRoles::INPUT)
        || VULKAN_TOSA_FP16_CAPABILITY.supports_dtype(dtype, ValueRoles::OUTPUT)
        || VULKAN_TOSA_FP8_CAPABILITY.supports_dtype(dtype, ValueRoles::INPUT)
        || VULKAN_TOSA_FP8_CAPABILITY.supports_dtype(dtype, ValueRoles::OUTPUT)
}

/// Why an artifact was not admitted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LoweringError {
    Parse(ParseError),
    Analysis(AnalysisError),
    /// The target is not one this backend advertises.
    UnsupportedTarget,
    /// The graph shape (regions, blocks, boundary, operator structure, attributes, or declared
    /// tensor shapes) is outside the tier.
    UnsupportedGraph,
    UnsupportedType(DType),
    UnsupportedOperator(Op),
    /// A static shape does not fit the kernels' 32-bit element domain, or the program needs more
    /// arena or slots than the plan can express.
    ResourceLimit,
}

impl fmt::Display for LoweringError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for LoweringError {}

/// Arena regions are aligned generously so every tensor start is cache-line aligned and any
/// storage-buffer offset alignment a device reports (at most 256) is satisfied.
pub(crate) const ARENA_ALIGNMENT: u64 = 256;

/// Device-independent kernel selection; `native` maps it onto a [`crate::shader::KernelKey`]
/// once the device's workgroup tuning and descriptor-array length are known.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum KernelSpec {
    Elementwise {
        op: ElementwiseOp,
        float: Storage,
        broadcast: bool,
    },
    Reduce {
        op: ReduceOp,
        float: Storage,
    },
    Matmul {
        input: Storage,
        output: Storage,
    },
    MaxPool {
        nan_mode: NanMode,
        float: Storage,
    },
    Move {
        storage: Storage,
        contiguous: bool,
    },
}

/// Dispatch geometry before device tuning.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Work {
    /// A grid-stride kernel over this many items.
    Linear(u32),
    /// A tiled MATMUL over `m × n` outputs per batch.
    Matmul { m: u32, n: u32, batch: u32 },
}

/// One recorded `vkCmdDispatch`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct DispatchPlan {
    pub kernel: KernelSpec,
    /// Specialization constants in the kernel's declared order.
    pub spec: Vec<u32>,
    pub work: Work,
    /// A compute→compute memory barrier must precede this dispatch: it reads a tensor an
    /// earlier dispatch wrote, or writes memory an earlier dispatch touched.
    pub barrier_before: bool,
}

/// One binding slot of an admitted program.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SlotPlan {
    pub slot: u32,
    pub role: SlotRole,
    /// Exact tensor bytes: the required length of a binding over this slot.
    pub byte_len: u64,
    pub storage: Storage,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SlotRole {
    Input,
    Output,
}

/// A serialized constant to upload into the arena at `load_program`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ConstantPlan {
    pub offset: u64,
    pub bytes: Vec<u8>,
}

/// A hardware-free execution plan for one admitted TOSA graph.
///
/// Slots follow the workspace convention shared with the other TOSA backends: block inputs take
/// slots `0..inputs`, block outputs follow in declared order. Bound slots occupy descriptor
/// array elements `0..slots.len()`; the arena, when present, sits at index `slots.len()`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ProgramPlan {
    pub slots: Vec<SlotPlan>,
    /// Bytes of program-owned storage (constants and intermediates); zero when none is needed.
    pub arena_bytes: u64,
    pub constants: Vec<ConstantPlan>,
    pub dispatches: Vec<DispatchPlan>,
}

impl ProgramPlan {
    /// Descriptor-array element the arena is bound at.
    pub fn arena_buffer_index(&self) -> u32 {
        self.slots.len() as u32
    }

    /// Descriptor-array elements the program addresses.
    pub fn buffer_count(&self) -> u32 {
        self.slots.len() as u32 + u32::from(self.arena_bytes != 0)
    }

    #[cfg(test)]
    fn slot(&self, slot: u32) -> Option<&SlotPlan> {
        self.slots.iter().find(|plan| plan.slot == slot)
    }
}

/// Admit `bytes` for `target` and produce its plan, or explain the rejection.
pub(crate) fn lower_tosa(bytes: &[u8], target: Target) -> Result<ProgramPlan, LoweringError> {
    if target != VULKAN_TOSA_TARGET && target != VULKAN_TOSA_FP8_TARGET {
        return Err(LoweringError::UnsupportedTarget);
    }
    let model = parse(bytes).map_err(LoweringError::Parse)?;
    let analysis = model.analyze_for(target).map_err(LoweringError::Analysis)?;
    Lowering::new(&analysis)?.run()
}

// ---------------------------------------------------------------------------------------------
// Tensor bookkeeping
// ---------------------------------------------------------------------------------------------

/// Static shape summary of a tensor value.
#[derive(Clone, Debug, PartialEq, Eq)]
struct TensorShape {
    dtype: DType,
    dims: Vec<u32>,
    elements: u32,
}

impl TensorShape {
    fn storage(&self) -> Storage {
        storage_of(self.dtype)
    }

    fn byte_len(&self) -> u64 {
        u64::from(self.elements) * scalar_bytes(self.dtype)
    }

    fn rank(&self) -> usize {
        self.dims.len()
    }

    /// Row-major element strides.
    fn strides(&self) -> Vec<u32> {
        let mut strides = vec![1_u32; self.dims.len()];
        let mut acc = 1_u32;
        for d in (0..self.dims.len()).rev() {
            strides[d] = acc;
            acc = acc.wrapping_mul(self.dims[d]);
        }
        strides
    }

    /// Dims padded with leading ones to `MAX_RANK`.
    fn padded_dims(&self) -> [u32; MAX_RANK] {
        pad_leading(&self.dims, 1)
    }
}

fn pad_leading(values: &[u32], fill: u32) -> [u32; MAX_RANK] {
    let mut padded = [fill; MAX_RANK];
    let offset = MAX_RANK - values.len();
    padded[offset..].copy_from_slice(values);
    padded
}

fn storage_of(dtype: DType) -> Storage {
    match dtype {
        DType::BOOL => Storage::Byte,
        DType::FP8E4M3 => Storage::Quarter(Fp8Format::E4M3),
        DType::FP8E5M2 => Storage::Quarter(Fp8Format::E5M2),
        DType::FP16 => Storage::Half,
        _ => Storage::Word,
    }
}

fn scalar_bytes(dtype: DType) -> u64 {
    match dtype {
        DType::BOOL | DType::FP8E4M3 | DType::FP8E5M2 => 1,
        DType::FP16 => 2,
        _ => 4,
    }
}

/// Where a tensor's bytes live during execution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Location {
    Slot(u32),
    /// An arena region (index into `Lowering::regions`).
    Region(usize),
}

/// Memory identity for hazard tracking. Arena tensors are keyed by the bytes they occupy, not by
/// region index: lifetime packing hands a freed region's bytes to later tensors, possibly
/// straddling several earlier regions, and a dispatch touching any of those bytes must be ordered
/// after every earlier dispatch that touched them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MemKey {
    Slot(u32),
    Arena { offset: u64, end: u64 },
}

impl MemKey {
    fn overlaps(self, other: Self) -> bool {
        match (self, other) {
            (Self::Slot(a), Self::Slot(b)) => a == b,
            (
                Self::Arena { offset, end },
                Self::Arena {
                    offset: other_offset,
                    end: other_end,
                },
            ) => offset < other_end && other_offset < end,
            _ => false,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Region {
    offset: u64,
    bytes: u64,
    /// Last execution position at which the region is read; the region is free afterwards.
    live_end: u32,
}

struct Lowering<'a, 'b> {
    analysis: &'b TosaAnalysis<'a>,
    inputs: &'b [ValueId],
    outputs: &'b [ValueId],
    order: &'b [OperatorId],
    shapes: HashMap<ValueId, TensorShape>,
    locations: HashMap<ValueId, Location>,
    /// Last execution position at which each value is consumed by a live operator.
    last_use: HashMap<ValueId, u32>,
    regions: Vec<Region>,
    arena_bytes: u64,
    constants: Vec<ConstantPlan>,
    dispatches: Vec<DispatchPlan>,
    slots: Vec<SlotPlan>,
    /// Hazard tracking since the last barrier: memory written (with the writing value) and read.
    written: Vec<(MemKey, ValueId)>,
    read: Vec<MemKey>,
    position: u32,
}

impl<'a, 'b> Lowering<'a, 'b> {
    fn new(analysis: &'b TosaAnalysis<'a>) -> Result<Self, LoweringError> {
        if analysis.regions().len() != 1 || analysis.blocks().len() != 1 {
            return Err(LoweringError::UnsupportedGraph);
        }
        // `PowDomain` is the IEEE NaN case the POW kernel produces itself; every other runtime
        // condition needs error detection this backend does not perform.
        if analysis
            .conditions()
            .iter()
            .any(|condition| !matches!(condition, RuntimeCondition::PowDomain { .. }))
        {
            return Err(LoweringError::UnsupportedGraph);
        }
        let block = analysis.blocks()[0].id();
        let inputs = analysis.block_inputs(block);
        let outputs = analysis.block_outputs(block);
        let order = analysis.execution_order(block);
        if inputs.is_empty()
            || outputs.is_empty()
            || inputs.iter().any(|input| outputs.contains(input))
            || outputs
                .iter()
                .any(|output| analysis.serialized_constant(*output).is_some())
        {
            return Err(LoweringError::UnsupportedGraph);
        }
        let mut duplicates = inputs.iter().chain(outputs).collect::<Vec<_>>();
        duplicates.sort_unstable_by_key(|value| value.get());
        if duplicates.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(LoweringError::UnsupportedGraph);
        }
        if inputs.len() + outputs.len() > u32::MAX as usize {
            return Err(LoweringError::ResourceLimit);
        }
        Ok(Self {
            analysis,
            inputs,
            outputs,
            order,
            shapes: HashMap::new(),
            locations: HashMap::new(),
            last_use: HashMap::new(),
            regions: Vec::new(),
            arena_bytes: 0,
            constants: Vec::new(),
            dispatches: Vec::new(),
            slots: Vec::new(),
            written: Vec::new(),
            read: Vec::new(),
            position: 0,
        })
    }

    fn run(mut self) -> Result<ProgramPlan, LoweringError> {
        // Boundary slots: inputs first, outputs after, in declared order.
        for (index, value) in self.inputs.iter().chain(self.outputs).enumerate() {
            let slot = index as u32;
            let shape = self.shape(*value)?;
            self.slots.push(SlotPlan {
                slot,
                role: if index < self.inputs.len() {
                    SlotRole::Input
                } else {
                    SlotRole::Output
                },
                byte_len: shape.byte_len(),
                storage: shape.storage(),
            });
            self.locations.insert(*value, Location::Slot(slot));
        }

        // Liveness: the last live consumer of every value.
        for (position, operator_id) in self.order.iter().enumerate() {
            let operator = self.analysis.operator(*operator_id);
            if self.skipped(operator) {
                continue;
            }
            for input in self.analysis.operator_inputs(*operator_id) {
                self.last_use.insert(*input, position as u32);
            }
        }

        for (position, operator_id) in self.order.iter().enumerate() {
            self.position = position as u32;
            let operator = self.analysis.operator(*operator_id);
            if self.skipped(operator) {
                continue;
            }
            let op = operator.op();
            if !supports_tosa_operator(op) {
                return Err(LoweringError::UnsupportedOperator(op));
            }
            let operator_inputs = self.analysis.operator_inputs(*operator_id);
            let operator_outputs = self.analysis.operator_outputs(*operator_id);
            let [output] = operator_outputs else {
                return Err(LoweringError::UnsupportedGraph);
            };
            let output = *output;
            if self.analysis.serialized_constant(output).is_some() {
                return Err(LoweringError::UnsupportedGraph);
            }
            match op {
                Op::IDENTITY => self.lower_copy(operator_inputs, output, 1)?,
                Op::RESHAPE => self.lower_reshape(operator_inputs, output)?,
                Op::TRANSPOSE => self.lower_transpose(operator, operator_inputs, output)?,
                Op::REVERSE => self.lower_reverse(operator, operator_inputs, output)?,
                Op::CONCAT => self.lower_concat(operator, operator_inputs, output)?,
                Op::MATMUL => self.lower_matmul(operator_inputs, output)?,
                Op::MAX_POOL2D => self.lower_max_pool(operator, operator_inputs, output)?,
                Op::ARGMAX
                | Op::REDUCE_MAX
                | Op::REDUCE_MIN
                | Op::REDUCE_PRODUCT
                | Op::REDUCE_SUM => self.lower_reduce(operator, operator_inputs, output)?,
                _ => self.lower_elementwise(operator, operator_inputs, output)?,
            }
        }

        // Every block output must have been produced by a live operator.
        for output in self.outputs {
            if self.analysis.value(*output).producer().is_none() {
                return Err(LoweringError::UnsupportedGraph);
            }
        }
        Ok(ProgramPlan {
            slots: self.slots,
            arena_bytes: self.arena_bytes,
            constants: self.constants,
            dispatches: self.dispatches,
        })
    }

    /// Producers of serialized constants and operators the analysis proved dead emit nothing.
    fn skipped(&self, operator: &virtio_accel_tosa::AnalyzedOperator<'_>) -> bool {
        matches!(operator.op(), Op::CONST | Op::CONST_SHAPE)
            || operator.hints().contains(OptimizationHints::DEAD)
    }

    // -- shapes -------------------------------------------------------------------------------

    fn shape(&mut self, value: ValueId) -> Result<TensorShape, LoweringError> {
        if let Some(shape) = self.shapes.get(&value) {
            return Ok(shape.clone());
        }
        let AnalyzedValueKind::Tensor(tensor) = self.analysis.value(value).kind() else {
            return Err(LoweringError::UnsupportedGraph);
        };
        let dtype = tensor.dtype();
        if !matches!(
            dtype,
            DType::FP32
                | DType::FP16
                | DType::BOOL
                | DType::INT32
                | DType::FP8E4M3
                | DType::FP8E5M2
        ) {
            return Err(LoweringError::UnsupportedType(dtype));
        }
        let rank = tensor.rank().ok_or(LoweringError::UnsupportedGraph)?;
        if rank > MAX_RANK {
            return Err(LoweringError::UnsupportedGraph);
        }
        let mut dims = Vec::with_capacity(rank);
        let mut elements = 1_u64;
        for dimension in tensor.dimensions() {
            let dimension = u32::try_from(dimension)
                .ok()
                .filter(|dimension| *dimension > 0)
                .ok_or(LoweringError::UnsupportedGraph)?;
            dims.push(dimension);
            elements = elements
                .checked_mul(u64::from(dimension))
                .ok_or(LoweringError::ResourceLimit)?;
        }
        let elements = u32::try_from(elements).map_err(|_| LoweringError::ResourceLimit)?;
        let shape = TensorShape {
            dtype,
            dims,
            elements,
        };
        self.shapes.insert(value, shape.clone());
        Ok(shape)
    }

    /// The shape of a floating-point tensor (`FP32`, or `FP16` where the tier is advertised).
    fn float_shape(&mut self, value: ValueId) -> Result<TensorShape, LoweringError> {
        let shape = self.shape(value)?;
        if !matches!(shape.dtype, DType::FP32 | DType::FP16) {
            return Err(LoweringError::UnsupportedType(shape.dtype));
        }
        Ok(shape)
    }

    /// The shape of a MATMUL operand: a float the tier multiplies, which is `FP32`/`FP16` or,
    /// under the FP8 target, either FP8 encoding.
    fn matmul_operand_shape(&mut self, value: ValueId) -> Result<TensorShape, LoweringError> {
        let shape = self.shape(value)?;
        if !matches!(
            shape.dtype,
            DType::FP32 | DType::FP16 | DType::FP8E4M3 | DType::FP8E5M2
        ) {
            return Err(LoweringError::UnsupportedType(shape.dtype));
        }
        Ok(shape)
    }

    fn typed_shape(&mut self, value: ValueId, dtype: DType) -> Result<TensorShape, LoweringError> {
        let shape = self.shape(value)?;
        if shape.dtype != dtype {
            return Err(LoweringError::UnsupportedType(shape.dtype));
        }
        Ok(shape)
    }

    // -- arena --------------------------------------------------------------------------------

    /// Allocate a region of `bytes` live through `live_end` at the lowest offset free of every
    /// overlapping-lifetime region. The region is first written by the dispatch at the current
    /// position, so regions whose lifetime ended earlier are free.
    fn allocate_region(&mut self, bytes: u64, live_end: u32) -> Result<usize, LoweringError> {
        self.allocate_region_written_at(bytes, live_end, self.position)
    }

    /// [`allocate_region`](Self::allocate_region) for bytes first written at `written_at`
    /// rather than at the current position. A region whose lifetime ended before `written_at`
    /// is free; one that ends at or after it is still overlapped, because the dispatch that
    /// writes it runs after the new bytes exist.
    fn allocate_region_written_at(
        &mut self,
        bytes: u64,
        live_end: u32,
        written_at: u32,
    ) -> Result<usize, LoweringError> {
        let bytes = bytes.max(1).div_ceil(ARENA_ALIGNMENT) * ARENA_ALIGNMENT;
        let mut candidates = vec![0_u64];
        let overlapping: Vec<Region> = self
            .regions
            .iter()
            .filter(|region| region.live_end >= written_at)
            .copied()
            .collect();
        for region in &overlapping {
            candidates.push(region.offset + region.bytes);
        }
        candidates.sort_unstable();
        let offset = candidates
            .into_iter()
            .find(|candidate| {
                let end = candidate + bytes;
                overlapping.iter().all(|region| {
                    end <= region.offset || *candidate >= region.offset + region.bytes
                })
            })
            .ok_or(LoweringError::ResourceLimit)?;
        let end = offset
            .checked_add(bytes)
            .ok_or(LoweringError::ResourceLimit)?;
        // Word offsets must fit the kernels' `u32` operand base.
        if end / 4 > u64::from(u32::MAX) {
            return Err(LoweringError::ResourceLimit);
        }
        self.arena_bytes = self.arena_bytes.max(end);
        self.regions.push(Region {
            offset,
            bytes,
            live_end,
        });
        Ok(self.regions.len() - 1)
    }

    fn value_last_use(&self, value: ValueId) -> u32 {
        self.last_use.get(&value).copied().unwrap_or(self.position)
    }

    /// The location of an operator input: a bound slot, an arena intermediate, or a constant
    /// uploaded into the arena on first use.
    fn input_location(&mut self, value: ValueId) -> Result<Location, LoweringError> {
        if let Some(location) = self.locations.get(&value) {
            return Ok(*location);
        }
        let shape = self.shape(value)?;
        let bytes = self
            .analysis
            .serialized_constant(value)
            .ok_or(LoweringError::UnsupportedGraph)?;
        if bytes.len() as u64 != shape.byte_len() {
            return Err(LoweringError::UnsupportedGraph);
        }
        // Constants are uploaded at `load_program`, before the first dispatch, and stay live
        // for the whole program: their bytes are written at position 0, so a region an
        // earlier intermediate has already died in is *not* free — the dispatch that wrote
        // that intermediate runs after the upload and would overwrite the constant.
        let region = self.allocate_region_written_at(shape.byte_len(), u32::MAX, 0)?;
        self.constants.push(ConstantPlan {
            offset: self.regions[region].offset,
            bytes: bytes.to_vec(),
        });
        let location = Location::Region(region);
        self.locations.insert(value, location);
        Ok(location)
    }

    /// The location an operator output is written to: its slot, or a fresh arena region.
    fn output_location(&mut self, value: ValueId) -> Result<Location, LoweringError> {
        if let Some(location) = self.locations.get(&value) {
            return Ok(*location);
        }
        let shape = self.shape(value)?;
        let live_end = self.value_last_use(value);
        let region = self.allocate_region(shape.byte_len(), live_end)?;
        let location = Location::Region(region);
        self.locations.insert(value, location);
        Ok(location)
    }

    fn operand(&self, location: Location) -> Operand {
        match location {
            Location::Slot(slot) => Operand {
                buffer: slot,
                base: 0,
            },
            Location::Region(index) => Operand {
                buffer: self.slots.len() as u32,
                base: (self.regions[index].offset / 4) as u32,
            },
        }
    }

    // -- dispatch recording -------------------------------------------------------------------

    fn mem_key(&self, location: Location) -> MemKey {
        match location {
            Location::Slot(slot) => MemKey::Slot(slot),
            Location::Region(index) => {
                let region = &self.regions[index];
                MemKey::Arena {
                    offset: region.offset,
                    end: region.offset + region.bytes,
                }
            }
        }
    }

    fn dispatch(
        &mut self,
        kernel: KernelSpec,
        spec: Vec<u32>,
        work: Work,
        reads: &[Location],
        writes: Location,
        written_value: ValueId,
    ) {
        let reads: Vec<MemKey> = reads.iter().map(|read| self.mem_key(*read)).collect();
        let writes = self.mem_key(writes);
        let raw = reads.iter().any(|read| {
            self.written
                .iter()
                .any(|(written, _)| written.overlaps(*read))
        });
        let war = self.read.iter().any(|read| read.overlaps(writes));
        // Segments of one `CONCAT` write disjoint parts of one tensor and need no ordering; any
        // other overlap with earlier written bytes does.
        let waw = self
            .written
            .iter()
            .any(|(written, value)| written.overlaps(writes) && *value != written_value);
        let barrier_before = raw || war || waw;
        if barrier_before {
            self.written.clear();
            self.read.clear();
        }
        self.read.extend_from_slice(&reads);
        self.written.push((writes, written_value));
        self.dispatches.push(DispatchPlan {
            kernel,
            spec,
            work,
            barrier_before,
        });
    }

    // -- operators ----------------------------------------------------------------------------

    /// `IDENTITY` (and any copy of `expected_inputs` operands whose first is the source): a
    /// view when both ends live in the arena, otherwise a contiguous copy.
    fn lower_copy(
        &mut self,
        inputs: &[ValueId],
        output: ValueId,
        expected_inputs: usize,
    ) -> Result<(), LoweringError> {
        if inputs.len() != expected_inputs {
            return Err(LoweringError::UnsupportedGraph);
        }
        let source = self.shape(inputs[0])?;
        let target = self.shape(output)?;
        if source.dtype != target.dtype || source.elements != target.elements {
            return Err(LoweringError::UnsupportedGraph);
        }
        let from = self.input_location(inputs[0])?;
        if let Location::Region(region) = from {
            if !self.locations.contains_key(&output) {
                // Arena to arena: alias the region and extend its lifetime over the view.
                let live_end = self.value_last_use(output);
                self.regions[region].live_end = self.regions[region].live_end.max(live_end);
                self.locations.insert(output, from);
                return Ok(());
            }
        }
        let to = self.output_location(output)?;
        let storage = source.storage();
        let geometry = MoveGeometry {
            count: source.elements,
            dims: source.padded_dims(),
            in_strides: pad_leading(&source.strides(), 0),
            in_offset: 0,
            out_strides: pad_leading(&source.strides(), 0),
            out_offset: 0,
        };
        let spec = move_spec(self.operand(from), self.operand(to), geometry, true);
        self.dispatch(
            KernelSpec::Move {
                storage,
                contiguous: true,
            },
            spec,
            Work::Linear(source.elements),
            &[from],
            to,
            output,
        );
        Ok(())
    }

    fn lower_reshape(&mut self, inputs: &[ValueId], output: ValueId) -> Result<(), LoweringError> {
        let [source, shape] = inputs else {
            return Err(LoweringError::UnsupportedGraph);
        };
        let AnalyzedValueKind::Shape(shape) = self.analysis.value(*shape).kind() else {
            return Err(LoweringError::UnsupportedGraph);
        };
        let values = shape.values().ok_or(LoweringError::UnsupportedGraph)?;
        let target = self.shape(output)?;
        let declared: Vec<i64> = values.collect();
        if declared.len() != target.rank()
            || declared
                .iter()
                .zip(&target.dims)
                .any(|(declared, dim)| *declared != i64::from(*dim))
        {
            return Err(LoweringError::UnsupportedGraph);
        }
        self.lower_copy(&[*source], output, 1)
    }

    fn lower_transpose(
        &mut self,
        operator: &virtio_accel_tosa::AnalyzedOperator<'_>,
        inputs: &[ValueId],
        output: ValueId,
    ) -> Result<(), LoweringError> {
        let [input] = inputs else {
            return Err(LoweringError::UnsupportedGraph);
        };
        let OpAttributes::Transpose { perms } = operator.source().attributes() else {
            return Err(LoweringError::UnsupportedGraph);
        };
        let source = self.shape(*input)?;
        let target = self.shape(output)?;
        let perms: Vec<usize> = perms
            .iter()
            .map(|perm| usize::try_from(perm).map_err(|_| LoweringError::UnsupportedGraph))
            .collect::<Result<_, _>>()?;
        if perms.len() != source.rank()
            || target.rank() != source.rank()
            || source.dtype != target.dtype
        {
            return Err(LoweringError::UnsupportedGraph);
        }
        let mut seen = vec![false; perms.len()];
        for (d, perm) in perms.iter().enumerate() {
            if *perm >= perms.len() || seen[*perm] || target.dims[d] != source.dims[*perm] {
                return Err(LoweringError::UnsupportedGraph);
            }
            seen[*perm] = true;
        }
        let source_strides = source.strides();
        let in_strides: Vec<u32> = perms.iter().map(|perm| source_strides[*perm]).collect();
        let geometry = MoveGeometry {
            count: target.elements,
            dims: target.padded_dims(),
            in_strides: pad_leading(&in_strides, 0),
            in_offset: 0,
            out_strides: pad_leading(&target.strides(), 0),
            out_offset: 0,
        };
        self.lower_move(*input, output, source.storage(), geometry)
    }

    fn lower_reverse(
        &mut self,
        operator: &virtio_accel_tosa::AnalyzedOperator<'_>,
        inputs: &[ValueId],
        output: ValueId,
    ) -> Result<(), LoweringError> {
        let [input] = inputs else {
            return Err(LoweringError::UnsupportedGraph);
        };
        let OpAttributes::Reverse { axis } = operator.source().attributes() else {
            return Err(LoweringError::UnsupportedGraph);
        };
        let source = self.shape(*input)?;
        let target = self.shape(output)?;
        if source != target {
            return Err(LoweringError::UnsupportedGraph);
        }
        let axis = usize::try_from(axis)
            .ok()
            .filter(|axis| *axis < source.rank())
            .ok_or(LoweringError::UnsupportedGraph)?;
        let strides = source.strides();
        let mut in_strides = strides.clone();
        in_strides[axis] = strides[axis].wrapping_neg();
        let in_offset = (source.dims[axis] - 1).wrapping_mul(strides[axis]);
        let geometry = MoveGeometry {
            count: source.elements,
            dims: source.padded_dims(),
            in_strides: pad_leading(&in_strides, 0),
            in_offset,
            out_strides: pad_leading(&strides, 0),
            out_offset: 0,
        };
        self.lower_move(*input, output, source.storage(), geometry)
    }

    fn lower_concat(
        &mut self,
        operator: &virtio_accel_tosa::AnalyzedOperator<'_>,
        inputs: &[ValueId],
        output: ValueId,
    ) -> Result<(), LoweringError> {
        if inputs.is_empty() {
            return Err(LoweringError::UnsupportedGraph);
        }
        let OpAttributes::Concat { axis } = operator.source().attributes() else {
            return Err(LoweringError::UnsupportedGraph);
        };
        let target = self.shape(output)?;
        let axis = usize::try_from(axis)
            .ok()
            .filter(|axis| *axis < target.rank())
            .ok_or(LoweringError::UnsupportedGraph)?;
        let mut total = 0_u64;
        let mut sources = Vec::with_capacity(inputs.len());
        for input in inputs {
            let source = self.shape(*input)?;
            if source.dtype != target.dtype
                || source.rank() != target.rank()
                || source
                    .dims
                    .iter()
                    .zip(&target.dims)
                    .enumerate()
                    .any(|(d, (source, target))| d != axis && source != target)
            {
                return Err(LoweringError::UnsupportedGraph);
            }
            total += u64::from(source.dims[axis]);
            sources.push(source);
        }
        if total != u64::from(target.dims[axis]) {
            return Err(LoweringError::UnsupportedGraph);
        }
        let out_strides = target.strides();
        let to = self.output_location(output)?;
        let mut offset = 0_u32;
        for (input, source) in inputs.iter().zip(&sources) {
            let geometry = MoveGeometry {
                count: source.elements,
                dims: source.padded_dims(),
                in_strides: pad_leading(&source.strides(), 0),
                in_offset: 0,
                out_strides: pad_leading(&out_strides, 0),
                out_offset: offset.wrapping_mul(out_strides[axis]),
            };
            offset += source.dims[axis];
            let from = self.input_location(*input)?;
            let spec = move_spec(self.operand(from), self.operand(to), geometry, false);
            self.dispatch(
                KernelSpec::Move {
                    storage: source.storage(),
                    contiguous: false,
                },
                spec,
                Work::Linear(source.elements),
                &[from],
                to,
                output,
            );
        }
        Ok(())
    }

    fn lower_move(
        &mut self,
        input: ValueId,
        output: ValueId,
        storage: Storage,
        geometry: MoveGeometry,
    ) -> Result<(), LoweringError> {
        let from = self.input_location(input)?;
        let to = self.output_location(output)?;
        let spec = move_spec(self.operand(from), self.operand(to), geometry, false);
        self.dispatch(
            KernelSpec::Move {
                storage,
                contiguous: false,
            },
            spec,
            Work::Linear(geometry.count),
            &[from],
            to,
            output,
        );
        Ok(())
    }

    fn lower_matmul(&mut self, inputs: &[ValueId], output: ValueId) -> Result<(), LoweringError> {
        let [lhs, rhs, lhs_zp, rhs_zp] = inputs else {
            return Err(LoweringError::UnsupportedGraph);
        };
        let lhs_shape = self.matmul_operand_shape(*lhs)?;
        let rhs_shape = self.typed_shape(*rhs, lhs_shape.dtype)?;
        // TOSA defines FP8 MATMUL as `(FP8, FP8) -> FP16`; every other admitted operand type
        // keeps its own dtype.
        let result_dtype = match lhs_shape.dtype {
            DType::FP8E4M3 | DType::FP8E5M2 => DType::FP16,
            dtype => dtype,
        };
        let out_shape = self.typed_shape(output, result_dtype)?;
        // TOSA 1.0 floating-point MATMUL admits only zero zero-points: the two trailing inputs
        // must be `CONST` tensors whose serialized payload is all-zero (signed zero included).
        for zero_point in [lhs_zp, rhs_zp] {
            self.require_zero_constant(*zero_point, lhs_shape.dtype)?;
        }
        let ([batch_l, m, k], [batch_r, k_r, n], [batch_o, m_o, n_o]) = (
            lhs_shape.dims.as_slice(),
            rhs_shape.dims.as_slice(),
            out_shape.dims.as_slice(),
        ) else {
            return Err(LoweringError::UnsupportedGraph);
        };
        if batch_l != batch_r || batch_l != batch_o || k != k_r || m != m_o || n != n_o {
            return Err(LoweringError::UnsupportedGraph);
        }
        let (batch, m, n, k) = (*batch_l, *m, *n, *k);
        let from_lhs = self.input_location(*lhs)?;
        let from_rhs = self.input_location(*rhs)?;
        let to = self.output_location(output)?;
        let spec = matmul_spec(
            self.operand(from_lhs),
            self.operand(from_rhs),
            self.operand(to),
            m,
            n,
            k,
            batch,
        );
        self.dispatch(
            KernelSpec::Matmul {
                input: lhs_shape.storage(),
                output: out_shape.storage(),
            },
            spec,
            Work::Matmul { m, n, batch },
            &[from_lhs, from_rhs],
            to,
            output,
        );
        Ok(())
    }

    fn lower_max_pool(
        &mut self,
        operator: &virtio_accel_tosa::AnalyzedOperator<'_>,
        inputs: &[ValueId],
        output: ValueId,
    ) -> Result<(), LoweringError> {
        let [input] = inputs else {
            return Err(LoweringError::UnsupportedGraph);
        };
        let OpAttributes::MaxPool2d {
            kernel,
            stride,
            pad,
            nan_mode,
        } = operator.source().attributes()
        else {
            return Err(LoweringError::UnsupportedGraph);
        };
        let nan_mode = nan_mode_of(nan_mode)?;
        let source = self.float_shape(*input)?;
        let target = self.typed_shape(output, source.dtype)?;
        let ([batch, height, width, channels], [batch_o, out_height, out_width, channels_o]) =
            (source.dims.as_slice(), target.dims.as_slice())
        else {
            return Err(LoweringError::UnsupportedGraph);
        };
        let attribute = |list: virtio_accel_tosa::I32List<'_>, len: usize| {
            let values: Vec<u32> = list
                .iter()
                .map(|value| u32::try_from(value).map_err(|_| LoweringError::UnsupportedGraph))
                .collect::<Result<_, _>>()?;
            if values.len() != len {
                return Err(LoweringError::UnsupportedGraph);
            }
            Ok(values)
        };
        let kernel = attribute(kernel, 2)?;
        let stride = attribute(stride, 2)?;
        let pad = attribute(pad, 4)?;
        if kernel.contains(&0) || stride.contains(&0) {
            return Err(LoweringError::UnsupportedGraph);
        }
        // TOSA: every pad is smaller than its kernel extent, so no window is entirely padding,
        // and the padded extent divides exactly into output positions.
        let [pad_top, pad_bottom, pad_left, pad_right] = pad[..] else {
            return Err(LoweringError::UnsupportedGraph);
        };
        if pad_top >= kernel[0]
            || pad_bottom >= kernel[0]
            || pad_left >= kernel[1]
            || pad_right >= kernel[1]
        {
            return Err(LoweringError::UnsupportedGraph);
        }
        let output_extent = |input: u32, pad_a: u32, pad_b: u32, kernel: u32, stride: u32| {
            let padded = u64::from(input) + u64::from(pad_a) + u64::from(pad_b);
            let span = padded.checked_sub(u64::from(kernel))?;
            if span % u64::from(stride) != 0 {
                return None;
            }
            u32::try_from(span / u64::from(stride) + 1).ok()
        };
        let expected_height = output_extent(*height, pad_top, pad_bottom, kernel[0], stride[0])
            .ok_or(LoweringError::UnsupportedGraph)?;
        let expected_width = output_extent(*width, pad_left, pad_right, kernel[1], stride[1])
            .ok_or(LoweringError::UnsupportedGraph)?;
        if batch != batch_o
            || channels != channels_o
            || expected_height != *out_height
            || expected_width != *out_width
        {
            return Err(LoweringError::UnsupportedGraph);
        }
        // The kernel's window arithmetic stays inside u32.
        let last_row = u64::from(*out_height - 1) * u64::from(stride[0]) + u64::from(kernel[0]);
        let last_col = u64::from(*out_width - 1) * u64::from(stride[1]) + u64::from(kernel[1]);
        if last_row > u64::from(u32::MAX) || last_col > u64::from(u32::MAX) {
            return Err(LoweringError::ResourceLimit);
        }
        let geometry = PoolGeometry {
            batch: *batch,
            height: *height,
            width: *width,
            channels: *channels,
            out_height: *out_height,
            out_width: *out_width,
            kernel: [kernel[0], kernel[1]],
            stride: [stride[0], stride[1]],
            pad_top,
            pad_left,
        };
        let from = self.input_location(*input)?;
        let to = self.output_location(output)?;
        let spec = max_pool_spec(self.operand(from), self.operand(to), geometry);
        self.dispatch(
            KernelSpec::MaxPool {
                nan_mode,
                float: source.storage(),
            },
            spec,
            Work::Linear(target.elements),
            &[from],
            to,
            output,
        );
        Ok(())
    }

    fn lower_reduce(
        &mut self,
        operator: &virtio_accel_tosa::AnalyzedOperator<'_>,
        inputs: &[ValueId],
        output: ValueId,
    ) -> Result<(), LoweringError> {
        let [input] = inputs else {
            return Err(LoweringError::UnsupportedGraph);
        };
        let (axis, op, argmax) = match operator.source().attributes() {
            OpAttributes::ArgMax { axis, nan_mode } => {
                (axis, ReduceOp::ArgMax(nan_mode_of(nan_mode)?), true)
            }
            OpAttributes::ReduceMax { axis, nan_mode } => {
                (axis, ReduceOp::Max(nan_mode_of(nan_mode)?), false)
            }
            OpAttributes::ReduceMin { axis, nan_mode } => {
                (axis, ReduceOp::Min(nan_mode_of(nan_mode)?), false)
            }
            OpAttributes::ReduceProduct { axis } => (axis, ReduceOp::Product, false),
            OpAttributes::ReduceSum { axis } => (axis, ReduceOp::Sum, false),
            _ => return Err(LoweringError::UnsupportedGraph),
        };
        let source = self.float_shape(*input)?;
        let target = self.typed_shape(output, if argmax { DType::INT32 } else { source.dtype })?;
        let axis = usize::try_from(axis)
            .ok()
            .filter(|axis| *axis < source.rank())
            .ok_or(LoweringError::UnsupportedGraph)?;
        let mut expected = source.dims.clone();
        if argmax {
            expected.remove(axis);
        } else {
            expected[axis] = 1;
        }
        if target.dims != expected {
            return Err(LoweringError::UnsupportedGraph);
        }
        let outer = source.dims[..axis].iter().product::<u32>();
        let inner = source.dims[axis + 1..].iter().product::<u32>();
        let from = self.input_location(*input)?;
        let to = self.output_location(output)?;
        let spec = reduce_spec(
            self.operand(from),
            self.operand(to),
            outer,
            source.dims[axis],
            inner,
        );
        self.dispatch(
            KernelSpec::Reduce {
                op,
                float: source.storage(),
            },
            spec,
            Work::Linear(target.elements),
            &[from],
            to,
            output,
        );
        Ok(())
    }

    fn lower_elementwise(
        &mut self,
        operator: &virtio_accel_tosa::AnalyzedOperator<'_>,
        inputs: &[ValueId],
        output: ValueId,
    ) -> Result<(), LoweringError> {
        let op = operator.op();
        let attributes = operator.source().attributes();
        let mut clamp = None;
        let (lane, tensor_inputs): (ElementwiseOp, Vec<ValueId>) = match op {
            Op::ABS => (ElementwiseOp::Abs, inputs.to_vec()),
            Op::CEIL => (ElementwiseOp::Ceil, inputs.to_vec()),
            Op::COS => (ElementwiseOp::Cos, inputs.to_vec()),
            Op::ERF => (ElementwiseOp::Erf, inputs.to_vec()),
            Op::EXP => (ElementwiseOp::Exp, inputs.to_vec()),
            Op::FLOOR => (ElementwiseOp::Floor, inputs.to_vec()),
            Op::LOG => (ElementwiseOp::Log, inputs.to_vec()),
            Op::RECIPROCAL => (ElementwiseOp::Reciprocal, inputs.to_vec()),
            Op::RSQRT => (ElementwiseOp::Rsqrt, inputs.to_vec()),
            Op::SIN => (ElementwiseOp::Sin, inputs.to_vec()),
            Op::SIGMOID => (ElementwiseOp::Sigmoid, inputs.to_vec()),
            Op::TANH => (ElementwiseOp::Tanh, inputs.to_vec()),
            Op::NEGATE => {
                let [value, input_zp, output_zp] = inputs else {
                    return Err(LoweringError::UnsupportedGraph);
                };
                let dtype = self.shape(*value)?.dtype;
                self.require_zero_constant(*input_zp, dtype)?;
                self.require_zero_constant(*output_zp, dtype)?;
                (ElementwiseOp::Negate, vec![*value])
            }
            Op::CLAMP => {
                let OpAttributes::Clamp {
                    min_val,
                    max_val,
                    nan_mode,
                } = attributes
                else {
                    return Err(LoweringError::UnsupportedGraph);
                };
                // Bounds serialize in the tensor dtype: four bytes for FP32, two for FP16. The
                // kernel applies the clamp at binary32 and narrows once (ADR 0008), so the
                // specialization words always carry binary32 bit patterns; a binary16 bound is
                // widened host-side, exactly. The ordering check runs on the same values.
                let bound = |bytes: &[u8]| -> Result<u32, LoweringError> {
                    let value = match bytes.len() {
                        4 => f32::from_le_bytes(
                            bytes
                                .try_into()
                                .map_err(|_| LoweringError::UnsupportedGraph)?,
                        ),
                        2 => crate::shader::f16_to_f32(u16::from_le_bytes(
                            bytes
                                .try_into()
                                .map_err(|_| LoweringError::UnsupportedGraph)?,
                        )),
                        _ => return Err(LoweringError::UnsupportedGraph),
                    };
                    if value.is_nan() {
                        return Err(LoweringError::UnsupportedGraph);
                    }
                    Ok(value.to_bits())
                };
                let lo = bound(min_val)?;
                let hi = bound(max_val)?;
                if f32::from_bits(hi) < f32::from_bits(lo) {
                    return Err(LoweringError::UnsupportedGraph);
                }
                clamp = Some([lo, hi]);
                (
                    ElementwiseOp::Clamp(nan_mode_of(nan_mode)?),
                    inputs.to_vec(),
                )
            }
            Op::ADD => (ElementwiseOp::Add, inputs.to_vec()),
            Op::SUB => (ElementwiseOp::Sub, inputs.to_vec()),
            Op::POW => (ElementwiseOp::Pow, inputs.to_vec()),
            Op::MUL => {
                let [lhs, rhs, shift] = inputs else {
                    return Err(LoweringError::UnsupportedGraph);
                };
                self.require_zero_constant(*shift, DType::INT8)?;
                (ElementwiseOp::Mul, vec![*lhs, *rhs])
            }
            Op::MAXIMUM => {
                let OpAttributes::Maximum { nan_mode } = attributes else {
                    return Err(LoweringError::UnsupportedGraph);
                };
                (
                    ElementwiseOp::Maximum(nan_mode_of(nan_mode)?),
                    inputs.to_vec(),
                )
            }
            Op::MINIMUM => {
                let OpAttributes::Minimum { nan_mode } = attributes else {
                    return Err(LoweringError::UnsupportedGraph);
                };
                (
                    ElementwiseOp::Minimum(nan_mode_of(nan_mode)?),
                    inputs.to_vec(),
                )
            }
            Op::EQUAL => (ElementwiseOp::Equal, inputs.to_vec()),
            Op::GREATER => (ElementwiseOp::Greater, inputs.to_vec()),
            Op::GREATER_EQUAL => (ElementwiseOp::GreaterEqual, inputs.to_vec()),
            Op::LOGICAL_AND => (ElementwiseOp::LogicalAnd, inputs.to_vec()),
            Op::LOGICAL_OR => (ElementwiseOp::LogicalOr, inputs.to_vec()),
            Op::LOGICAL_XOR => (ElementwiseOp::LogicalXor, inputs.to_vec()),
            Op::LOGICAL_NOT => (ElementwiseOp::LogicalNot, inputs.to_vec()),
            Op::SELECT => (ElementwiseOp::Select, inputs.to_vec()),
            other => return Err(LoweringError::UnsupportedOperator(other)),
        };
        self.emit_elementwise(lane, &tensor_inputs, output, clamp)
    }

    fn emit_elementwise(
        &mut self,
        lane: ElementwiseOp,
        inputs: &[ValueId],
        output: ValueId,
        clamp: Option<[u32; 2]>,
    ) -> Result<(), LoweringError> {
        let lanes = lane.inputs();
        if inputs.len() != lanes.len() {
            return Err(LoweringError::UnsupportedGraph);
        }
        let target = self.shape(output)?;
        // Every float lane (`Word` in the lane table) must agree on one float dtype, FP32 or
        // FP16; pure-`BOOL` lanes have no float storage and the kernel's float width is unused.
        let mut float_dtype: Option<DType> = None;
        let mut unify = |dtype: DType| -> Result<(), LoweringError> {
            if !matches!(dtype, DType::FP32 | DType::FP16) {
                return Err(LoweringError::UnsupportedType(dtype));
            }
            match float_dtype {
                Some(existing) if existing != dtype => Err(LoweringError::UnsupportedGraph),
                _ => {
                    float_dtype = Some(dtype);
                    Ok(())
                }
            }
        };
        let mut shapes = Vec::with_capacity(inputs.len());
        for (input, storage) in inputs.iter().zip(lanes) {
            let shape = self.shape(*input)?;
            match storage {
                Storage::Word => unify(shape.dtype)?,
                Storage::Byte if shape.dtype != DType::BOOL => {
                    return Err(LoweringError::UnsupportedType(shape.dtype));
                }
                _ => {}
            }
            if shape.rank() != target.rank()
                || shape
                    .dims
                    .iter()
                    .zip(&target.dims)
                    .any(|(dim, out)| *dim != *out && *dim != 1)
            {
                return Err(LoweringError::UnsupportedGraph);
            }
            shapes.push(shape);
        }
        match lane.output() {
            Storage::Word => unify(target.dtype)?,
            Storage::Byte if target.dtype != DType::BOOL => {
                return Err(LoweringError::UnsupportedType(target.dtype));
            }
            _ => {}
        }
        let float = storage_of(float_dtype.unwrap_or(DType::FP32));
        let broadcast = shapes.iter().any(|shape| shape.dims != target.dims);
        let mut strides = Vec::with_capacity(inputs.len());
        for shape in &shapes {
            let own = shape.strides();
            let mut broadcast_strides = vec![0_u32; shape.rank()];
            for d in 0..shape.rank() {
                broadcast_strides[d] = if shape.dims[d] == 1 && target.dims[d] != 1 {
                    0
                } else {
                    own[d]
                };
            }
            strides.push(pad_leading(&broadcast_strides, 0));
        }
        let mut reads = Vec::with_capacity(inputs.len());
        let mut operands = Vec::with_capacity(inputs.len());
        for input in inputs {
            let location = self.input_location(*input)?;
            reads.push(location);
            operands.push(self.operand(location));
        }
        let to = self.output_location(output)?;
        let spec = ElementwiseSpec {
            count: target.elements,
            inputs: &operands,
            output: self.operand(to),
            dims: target.padded_dims(),
            strides: &strides,
            clamp,
        }
        .words(broadcast);
        self.dispatch(
            KernelSpec::Elementwise {
                op: lane,
                float,
                broadcast,
            },
            spec,
            Work::Linear(target.elements),
            &reads,
            to,
            output,
        );
        Ok(())
    }

    /// A serialized constant of `dtype` whose every scalar is zero (signed zero included).
    fn require_zero_constant(&mut self, value: ValueId, dtype: DType) -> Result<(), LoweringError> {
        let AnalyzedValueKind::Tensor(tensor) = self.analysis.value(value).kind() else {
            return Err(LoweringError::UnsupportedGraph);
        };
        if tensor.dtype() != dtype {
            return Err(LoweringError::UnsupportedGraph);
        }
        let bytes = self
            .analysis
            .serialized_constant(value)
            .ok_or(LoweringError::UnsupportedGraph)?;
        let zero = match dtype {
            DType::FP32 => {
                bytes.len() % 4 == 0
                    && bytes.chunks_exact(4).all(|chunk| {
                        u32::from_le_bytes(chunk.try_into().expect("four bytes")) & 0x7fff_ffff == 0
                    })
            }
            DType::FP16 => {
                bytes.len() % 2 == 0
                    && bytes.chunks_exact(2).all(|chunk| {
                        u16::from_le_bytes(chunk.try_into().expect("two bytes")) & 0x7fff == 0
                    })
            }
            // Both FP8 encodings put the sign in bit 7, so this admits signed zero and nothing
            // else, exactly as the FP32 and FP16 arms do.
            DType::FP8E4M3 | DType::FP8E5M2 => bytes.iter().all(|byte| byte & 0x7f == 0),
            _ => bytes.iter().all(|byte| *byte == 0),
        };
        if bytes.is_empty() || !zero {
            return Err(LoweringError::UnsupportedGraph);
        }
        Ok(())
    }
}

fn nan_mode_of(mode: NanPropagationMode) -> Result<NanMode, LoweringError> {
    if mode == NanPropagationMode::PROPAGATE {
        Ok(NanMode::Propagate)
    } else if mode == NanPropagationMode::IGNORE {
        Ok(NanMode::Ignore)
    } else {
        Err(LoweringError::UnsupportedGraph)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use virtio_accel_conformance::numerics::{
        HEXAGON_UNARY_FP16_CASES, IDENTITY_EDGES_FP16, IDENTITY_EDGES_FP32, IDENTITY_INT8,
        MATMUL_FP16, MATMUL_FP32, MAX_POOL2D_FP16, MAX_POOL2D_FP32,
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
    fn capability_names_the_shared_fp32_operator_set() {
        assert_eq!(FLOAT_OPERATORS.len(), 42);
        for op in [
            Op::IDENTITY,
            Op::MATMUL,
            Op::MAX_POOL2D,
            Op::ARGMAX,
            Op::ERF,
            Op::CONCAT,
            Op::TRANSPOSE,
        ] {
            assert!(supports_tosa_operator(op), "{op:?}");
        }
        for op in [Op::CONV2D, Op::AVG_POOL2D, Op::CAST, Op::RESCALE, Op::PAD] {
            assert!(!supports_tosa_operator(op), "{op:?}");
        }
        assert!(supports_tosa_dtype(DType::FP32));
        assert!(supports_tosa_dtype(DType::FP16));
        assert!(supports_tosa_dtype(DType::BOOL));
        assert!(supports_tosa_dtype(DType::INT32));
        // INT8 is a compile-time parameter only (the `MUL` shift), never a boundary dtype.
        assert!(!supports_tosa_dtype(DType::INT8));
        assert!(VULKAN_TOSA_CAPABILITY.supports_dtype(DType::INT8, ValueRoles::CONSTANT));
        assert!(!VULKAN_TOSA_CAPABILITY.supports_dtype(DType::INT8, ValueRoles::INTERMEDIATE));
        assert_eq!(VULKAN_TOSA_CAPABILITY.target, VULKAN_TOSA_TARGET);
    }

    #[test]
    fn lowers_the_local_fp32_identity_artifact() {
        let plan = lower_tosa(IDENTITY_FP32_LOCAL, VULKAN_TOSA_TARGET).unwrap();
        assert_eq!(plan.slots.len(), 2);
        assert_eq!(plan.slot(0).unwrap().role, SlotRole::Input);
        assert_eq!(plan.slot(1).unwrap().role, SlotRole::Output);
        assert_eq!(plan.slot(0).unwrap().byte_len, 4);
        assert_eq!(plan.slot(1).unwrap().storage, Storage::Word);
        assert!(plan.slot(2).is_none());
        assert_eq!(plan.arena_bytes, 0);
        assert!(plan.constants.is_empty());
        assert_eq!(plan.dispatches.len(), 1);
        let dispatch = &plan.dispatches[0];
        assert_eq!(
            dispatch.kernel,
            KernelSpec::Move {
                storage: Storage::Word,
                contiguous: true
            }
        );
        assert_eq!(dispatch.work, Work::Linear(1));
        assert!(!dispatch.barrier_before);
        // input (buffer 0, base 0), output (buffer 1, base 0), count 1
        assert_eq!(dispatch.spec, vec![0, 0, 1, 0, 1]);
    }

    #[test]
    fn lowers_the_shared_fp32_edge_identity_artifact() {
        let plan = lower_tosa(IDENTITY_EDGES_FP32.artifact, VULKAN_TOSA_TARGET).unwrap();
        let expected = IDENTITY_EDGES_FP32.inputs[0].values.len();
        assert_eq!(plan.dispatches[0].work, Work::Linear(expected as u32));
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
        // INT8 identity under the floating-point target: never relabeled.
        assert!(matches!(
            lower_tosa(IDENTITY_INT8.artifact, VULKAN_TOSA_TARGET),
            Err(LoweringError::UnsupportedType(DType::INT8) | LoweringError::Analysis(_))
        ));
    }

    #[test]
    fn fp16_capability_extends_the_fp32_boundary() {
        for dtype in [DType::FP32, DType::FP16, DType::BOOL, DType::INT32] {
            assert!(
                VULKAN_TOSA_FP16_CAPABILITY.supports_dtype(dtype, ValueRoles::INPUT),
                "{dtype:?}"
            );
            assert!(
                VULKAN_TOSA_FP16_CAPABILITY.supports_dtype(dtype, ValueRoles::OUTPUT),
                "{dtype:?}"
            );
        }
        assert_eq!(VULKAN_TOSA_FP16_CAPABILITY.target, VULKAN_TOSA_TARGET);
        assert_eq!(VULKAN_TOSA_FP16_CAPABILITY.operators, FLOAT_OPERATORS);
        assert!(!VULKAN_TOSA_CAPABILITY.supports_dtype(DType::FP16, ValueRoles::INPUT));
    }

    #[test]
    fn lowers_the_shared_fp16_artifacts() {
        let plan = lower_tosa(IDENTITY_EDGES_FP16.artifact, VULKAN_TOSA_TARGET).unwrap();
        let expected = IDENTITY_EDGES_FP16.inputs[0].bits.len() as u32;
        assert_eq!(plan.slot(0).unwrap().byte_len, u64::from(expected) * 2);
        assert_eq!(plan.slot(0).unwrap().storage, Storage::Half);
        assert_eq!(plan.dispatches.len(), 1);
        assert_eq!(
            plan.dispatches[0].kernel,
            KernelSpec::Move {
                storage: Storage::Half,
                contiguous: true
            }
        );
        assert_eq!(plan.dispatches[0].work, Work::Linear(expected));

        let plan = lower_tosa(MATMUL_FP16.artifact, VULKAN_TOSA_TARGET).unwrap();
        assert_eq!(
            plan.dispatches[0].kernel,
            KernelSpec::Matmul {
                input: Storage::Half,
                output: Storage::Half
            }
        );
        assert_eq!(plan.slot(0).unwrap().byte_len, 6 * 2);
        assert_eq!(plan.slot(2).unwrap().byte_len, 4 * 2);
        assert_eq!(plan.arena_bytes, 0);

        let plan = lower_tosa(MAX_POOL2D_FP16.artifact, VULKAN_TOSA_TARGET).unwrap();
        assert_eq!(
            plan.dispatches[0].kernel,
            KernelSpec::MaxPool {
                nan_mode: NanMode::Propagate,
                float: Storage::Half
            }
        );
    }

    #[test]
    fn fp16_clamp_bounds_arrive_widened_to_binary32() {
        // The shared clamp-fp16 fixture clamps [0.5, 1.0, 2.0, 4.0] to [-1.0, 1.0]; the kernel
        // applies the clamp at binary32 and narrows once (ADR 0008), so the specialization
        // words carry the widened binary32 patterns.
        let clamp = HEXAGON_UNARY_FP16_CASES
            .iter()
            .find(|case| case.name == "clamp-fp16")
            .expect("the shared clamp-fp16 case");
        let plan = lower_tosa(clamp.artifact, VULKAN_TOSA_TARGET).unwrap();
        let dispatch = &plan.dispatches[0];
        assert_eq!(
            dispatch.kernel,
            KernelSpec::Elementwise {
                op: ElementwiseOp::Clamp(NanMode::Propagate),
                float: Storage::Half,
                broadcast: false
            }
        );
        assert_eq!(dispatch.spec[dispatch.spec.len() - 2], (-1.0_f32).to_bits());
        assert_eq!(dispatch.spec[dispatch.spec.len() - 1], 1.0_f32.to_bits());
    }

    #[test]
    fn fp16_negate_zero_points_are_consumed_at_admission() {
        let negate = HEXAGON_UNARY_FP16_CASES
            .iter()
            .find(|case| case.name == "negate-fp16")
            .expect("the shared negate-fp16 case");
        let plan = lower_tosa(negate.artifact, VULKAN_TOSA_TARGET).unwrap();
        assert_eq!(
            plan.dispatches[0].kernel,
            KernelSpec::Elementwise {
                op: ElementwiseOp::Negate,
                float: Storage::Half,
                broadcast: false
            }
        );
        assert_eq!(plan.arena_bytes, 0);
    }

    #[test]
    fn lowers_the_shared_fp32_matmul_artifact() {
        let plan = lower_tosa(MATMUL_FP32.artifact, VULKAN_TOSA_TARGET).unwrap();
        assert_eq!(plan.slots.len(), 3);
        assert_eq!(plan.slot(0).unwrap().byte_len, 6 * 4);
        assert_eq!(plan.slot(1).unwrap().byte_len, 6 * 4);
        assert_eq!(plan.slot(2).unwrap().role, SlotRole::Output);
        assert_eq!(plan.slot(2).unwrap().byte_len, 4 * 4);
        assert_eq!(plan.dispatches.len(), 1);
        let dispatch = &plan.dispatches[0];
        assert_eq!(
            dispatch.kernel,
            KernelSpec::Matmul {
                input: Storage::Word,
                output: Storage::Word
            }
        );
        assert_eq!(
            dispatch.work,
            Work::Matmul {
                m: 2,
                n: 2,
                batch: 1
            }
        );
        // lhs, rhs, out operands then m, n, k, batch.
        assert_eq!(dispatch.spec, vec![0, 0, 1, 0, 2, 0, 2, 2, 3, 1]);
        // The zero-point constants are consumed at admission, never uploaded.
        assert_eq!(plan.arena_bytes, 0);
    }

    #[test]
    fn lowers_the_shared_fp32_max_pool_artifact() {
        let plan = lower_tosa(MAX_POOL2D_FP32.artifact, VULKAN_TOSA_TARGET).unwrap();
        let dispatch = &plan.dispatches[0];
        assert_eq!(
            dispatch.kernel,
            KernelSpec::MaxPool {
                nan_mode: NanMode::Propagate,
                float: Storage::Word
            }
        );
        assert_eq!(dispatch.work, Work::Linear(8));
        // in, out, N, H, W, C, OH, OW, KH, KW, SH, SW, PT, PL
        assert_eq!(
            dispatch.spec,
            vec![0, 0, 1, 0, 1, 4, 4, 2, 2, 2, 2, 2, 2, 2, 0, 0]
        );
    }

    #[test]
    fn rejects_garbage_as_a_parse_error() {
        assert!(matches!(
            lower_tosa(b"not a flatbuffer", VULKAN_TOSA_TARGET),
            Err(LoweringError::Parse(_))
        ));
    }

    #[test]
    fn arena_regions_pack_by_lifetime() {
        // Build a lowering with no graph behind it purely to exercise the allocator.
        let bytes = IDENTITY_FP32_LOCAL;
        let model = parse(bytes).unwrap();
        let analysis = model.analyze_for(VULKAN_TOSA_TARGET).unwrap();
        let mut lowering = Lowering::new(&analysis).unwrap();
        lowering.position = 0;
        let a = lowering.allocate_region(100, 1).unwrap();
        let b = lowering.allocate_region(100, 5).unwrap();
        assert_eq!(lowering.regions[a].offset, 0);
        assert_eq!(lowering.regions[b].offset, ARENA_ALIGNMENT);
        // Position 2: `a` is dead, its space is reused before the arena grows.
        lowering.position = 2;
        let c = lowering.allocate_region(ARENA_ALIGNMENT, 9).unwrap();
        assert_eq!(lowering.regions[c].offset, 0);
        let d = lowering.allocate_region(1, 9).unwrap();
        assert_eq!(lowering.regions[d].offset, 2 * ARENA_ALIGNMENT);
        assert_eq!(lowering.arena_bytes, 3 * ARENA_ALIGNMENT);
    }

    /// A constant first used at position 2 is uploaded before position 0's dispatch runs, so the
    /// bytes of `a` (dead since position 1) are not free for it: that dispatch would overwrite
    /// the constant. An intermediate first written at position 2 may still take them.
    #[test]
    fn constants_never_share_bytes_with_earlier_intermediates() {
        let model = parse(IDENTITY_FP32_LOCAL).unwrap();
        let analysis = model.analyze_for(VULKAN_TOSA_TARGET).unwrap();
        let mut lowering = Lowering::new(&analysis).unwrap();
        lowering.position = 0;
        let a = lowering.allocate_region(64, 1).unwrap();
        assert_eq!(lowering.regions[a].offset, 0);
        lowering.position = 2;
        let constant = lowering
            .allocate_region_written_at(64, u32::MAX, 0)
            .unwrap();
        assert_eq!(lowering.regions[constant].offset, ARENA_ALIGNMENT);
        let intermediate = lowering.allocate_region(64, 3).unwrap();
        assert_eq!(lowering.regions[intermediate].offset, 0);
    }

    /// Region `a` is read at position 1 and dies; at position 2 the packer hands its bytes to a
    /// new tensor `b`. The dispatch writing `b` reads nothing that was written since the last
    /// barrier, so only byte-range hazard tracking can order it after `a`'s reader.
    #[test]
    fn reused_arena_bytes_force_a_barrier_before_the_new_writer() {
        let model = parse(IDENTITY_FP32_LOCAL).unwrap();
        let analysis = model.analyze_for(VULKAN_TOSA_TARGET).unwrap();
        let mut lowering = Lowering::new(&analysis).unwrap();
        let values: Vec<ValueId> = analysis.values().iter().map(|value| value.id()).collect();
        let (a_value, b_value) = (values[0], values[1]);
        let kernel = KernelSpec::Move {
            storage: Storage::Word,
            contiguous: true,
        };

        // Position 0: a producer writes `a`.
        lowering.position = 0;
        let a = lowering.allocate_region(64, 1).unwrap();
        lowering.dispatch(
            kernel,
            Vec::new(),
            Work::Linear(16),
            &[Location::Slot(0)],
            Location::Region(a),
            a_value,
        );
        // Position 1: the last reader of `a` writes a bound slot.
        lowering.position = 1;
        lowering.dispatch(
            kernel,
            Vec::new(),
            Work::Linear(16),
            &[Location::Region(a)],
            Location::Slot(1),
            b_value,
        );
        // Position 2: `a` is dead, so `b` is packed into its bytes and written from a slot.
        lowering.position = 2;
        let b = lowering.allocate_region(64, 3).unwrap();
        assert_eq!(lowering.regions[b].offset, lowering.regions[a].offset);
        assert_ne!(a, b, "a fresh region index over the same bytes");
        lowering.dispatch(
            kernel,
            Vec::new(),
            Work::Linear(16),
            &[Location::Slot(0)],
            Location::Region(b),
            b_value,
        );

        let barriers: Vec<bool> = lowering
            .dispatches
            .iter()
            .map(|dispatch| dispatch.barrier_before)
            .collect();
        assert_eq!(
            barriers,
            [false, true, true],
            "the reader depends on the producer (RAW); the new writer on the reader (WAR)"
        );

        // Disjoint arena bytes carry no hazard: a writer into a fresh region needs no barrier.
        lowering.position = 3;
        let c = lowering.allocate_region(64, 4).unwrap();
        assert_ne!(lowering.regions[c].offset, lowering.regions[b].offset);
        lowering.dispatch(
            kernel,
            Vec::new(),
            Work::Linear(16),
            &[Location::Slot(0)],
            Location::Region(c),
            a_value,
        );
        assert!(!lowering.dispatches[3].barrier_before);
        assert!(
            MemKey::Arena {
                offset: 0,
                end: 256
            }
            .overlaps(MemKey::Arena {
                offset: 255,
                end: 512
            })
        );
        assert!(
            !MemKey::Arena {
                offset: 0,
                end: 256
            }
            .overlaps(MemKey::Arena {
                offset: 256,
                end: 512
            })
        );
        assert!(!MemKey::Slot(0).overlaps(MemKey::Arena {
            offset: 0,
            end: 256
        }));
    }

    #[test]
    fn padding_and_strides_follow_row_major_order() {
        let shape = TensorShape {
            dtype: DType::FP32,
            dims: vec![2, 3, 4],
            elements: 24,
        };
        assert_eq!(shape.strides(), vec![12, 4, 1]);
        assert_eq!(shape.padded_dims(), [1, 1, 1, 2, 3, 4]);
        assert_eq!(pad_leading(&shape.strides(), 0), [0, 0, 0, 12, 4, 1]);
    }
}
