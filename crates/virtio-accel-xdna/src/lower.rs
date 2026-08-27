//! Portable TOSA admission for the XDNA backend.
//!
//! This module compiles on every host (no HRX, no `unsafe`). It declares the backend's advertised
//! `Target` constants (issue #82) and [`admit`]s a TOSA artifact into a [`CompilerSpec`] — the
//! validated, integers-and-enums-only description the compiler helper (issue #84) turns into an
//! amdxdna artifact. Anything outside the advertised subset is rejected here, before any subprocess
//! runs. Graph lowering for compute tiers grows on top of this; the compilable subsets today are
//! the BF16 IDENTITY (a DMA copy), BF16 → FP32 MATMUL (issue #90), BF16 NHWC MAX_POOL2D
//! (issue #91), explicit FP8 → BF16 CAST storage conversion (issue #109), and exact INT8
//! IDENTITY, zero-point-aware INT8 → INT32 MATMUL (issue #144), and exact static
//! INT32 → INT8 RESCALE (issue #147).

use virtio_accel_tosa::{
    AnalyzedValueKind, CapabilityDescriptor, DType, DTypeCapability, ExtensionSet,
    GraphCapabilities, Level, NanPropagationMode, Op, OpAttributes, OperatorCapability,
    OperatorConstraints, ProfileSet, RoundingMode, RuntimeConditionSupport, Target, ValueRoles,
    Version, parse,
};

/// The BF16 floating-point tier: TOSA 1.0, floating-point profile, level 8K, BF16 extension.
///
/// XDNA2 executes BF16 with FP32 accumulation natively; FP32/FP16 have no compute path and are
/// rejected at admission rather than silently run as BF16 (issue #82).
pub const XDNA_TOSA_TARGET: Target = Target::new(
    Version::TOSA_1_0,
    ProfileSet::FLOATING_POINT,
    Level::Level8K,
    ExtensionSet::BF16,
);

/// The FP8 storage tier: graph-visible E4M3/E5M2 inputs explicitly cast to BF16 on the NPU.
///
/// This is separate from [`XDNA_TOSA_TARGET`] so adding the storage tier does not change the
/// identity of the existing BF16 target. FP8 is not advertised as a compute dtype: every admitted
/// graph has an FP8 block input, one explicit CAST, and a BF16 block output.
pub const XDNA_TOSA_FP8_TARGET: Target = Target::new(
    Version::TOSA_1_0,
    ProfileSet::FLOATING_POINT,
    Level::Level8K,
    ExtensionSet::BF16
        .union(ExtensionSet::FP8E4M3)
        .union(ExtensionSet::FP8E5M2),
);

/// The integer tier: TOSA 1.0, integer profile, level 8K, no extensions.
///
/// INT8 identity and zero-point-aware MATMUL with exact INT32 results, kept on a separate target from the
/// floating-point tier exactly as the OpenVINO backend separates its FP and INTEGER targets.
pub const XDNA_TOSA_INTEGER_TARGET: Target = Target::new(
    Version::TOSA_1_0,
    ProfileSet::INTEGER,
    Level::Level8K,
    ExtensionSet::NONE,
);

const BF16_DTYPES: &[DTypeCapability] = &[
    DTypeCapability::new(DType::BF16, ValueRoles::ALL),
    DTypeCapability::new(DType::FP32, ValueRoles::OUTPUT),
];

const BF16_OPERATORS: &[OperatorCapability] = &[
    OperatorCapability::new(Op::CONST),
    OperatorCapability::new(Op::IDENTITY),
    OperatorCapability::constrained(Op::MATMUL, OperatorConstraints::ZERO_ZERO_POINTS),
    OperatorCapability::constrained(
        Op::MAX_POOL2D,
        OperatorConstraints::PROPAGATING_NAN.union(OperatorConstraints::ZERO_PADDING),
    ),
];

/// Conservative capability boundary for the implemented XDNA BF16 execution tier.
pub const XDNA_TOSA_CAPABILITY: CapabilityDescriptor = CapabilityDescriptor {
    target: XDNA_TOSA_TARGET,
    dtypes: BF16_DTYPES,
    operators: BF16_OPERATORS,
    graph: GraphCapabilities {
        max_regions: 1,
        max_blocks: 1,
        dynamic_shapes: false,
        runtime_conditions: RuntimeConditionSupport::None,
    },
};

const FP8_STORAGE_DTYPES: &[DTypeCapability] = &[
    DTypeCapability::new(DType::FP8E4M3, ValueRoles::INPUT),
    DTypeCapability::new(DType::FP8E5M2, ValueRoles::INPUT),
    DTypeCapability::new(DType::BF16, ValueRoles::OUTPUT),
];

const FP8_STORAGE_OPERATORS: &[OperatorCapability] = &[OperatorCapability::new(Op::CAST)];

/// Conservative capability boundary for explicit FP8 storage conversion.
pub const XDNA_TOSA_FP8_CAPABILITY: CapabilityDescriptor = CapabilityDescriptor {
    target: XDNA_TOSA_FP8_TARGET,
    dtypes: FP8_STORAGE_DTYPES,
    operators: FP8_STORAGE_OPERATORS,
    graph: GraphCapabilities {
        max_regions: 1,
        max_blocks: 1,
        dynamic_shapes: false,
        runtime_conditions: RuntimeConditionSupport::None,
    },
};

// This is OpenVINO's semantic surface for its integer target, plus the one role its integer tier
// does not need: OpenVINO only ever *produces* INT32, while this backend also *consumes* it as the
// RESCALE input below. A dtype omitted from `INPUT` is unroutable through the standard
// `INPUT || OUTPUT` capability filter, so the advertised roles have to cover every admitted tier.
// XDNA's admission functions below impose an additional, hardware-specific static-memory envelope.
const INTEGER_DTYPES: &[DTypeCapability] = &[
    DTypeCapability::new(DType::INT8, ValueRoles::ALL),
    DTypeCapability::new(
        DType::INT32,
        ValueRoles::INPUT
            .union(ValueRoles::OUTPUT)
            .union(ValueRoles::CONSTANT)
            .union(ValueRoles::INTERMEDIATE),
    ),
];

const INTEGER_OPERATORS: &[OperatorCapability] = &[
    OperatorCapability::new(Op::CONST),
    OperatorCapability::new(Op::IDENTITY),
    OperatorCapability::new(Op::MATMUL),
    OperatorCapability::new(Op::RESCALE),
];

/// Conservative exact integer-profile capability boundary for XDNA lowering.
///
/// This mirrors OpenVINO's dtype and operator declaration. XDNA additionally admits only bounded,
/// static specializations that fit the proven one-core implementation; this is a hardware
/// envelope, not a semantic fallback.
pub const XDNA_TOSA_INTEGER_CAPABILITY: CapabilityDescriptor = CapabilityDescriptor {
    target: XDNA_TOSA_INTEGER_TARGET,
    dtypes: INTEGER_DTYPES,
    operators: INTEGER_OPERATORS,
    graph: GraphCapabilities {
        max_regions: 1,
        max_blocks: 1,
        dynamic_shapes: false,
        runtime_conditions: RuntimeConditionSupport::None,
    },
};

/// The DMA line size the IDENTITY template transfers in; an admitted element count must be a
/// positive multiple of it. The Rust compiler driver carries this value into the helper spec.
pub(crate) const IDENTITY_LINE_SIZE: usize = 1024;

/// The fixed element tile converted by the FP8 → BF16 kernel.
pub(crate) const FP8_CAST_LINE_SIZE: usize = 1024;

/// Largest direct-DMA line used by the INT8 IDENTITY template.
///
/// Small corpus tensors use one shorter line. Every line is four-byte aligned because the AIE DMA
/// rejects shorter granularity; larger tensors must divide into complete 1,024-byte lines so the
/// runtime sequence never needs a hidden tail copy.
pub(crate) const INT8_IDENTITY_MAX_LINE_SIZE: usize = 1024;

/// The one tested MATMUL compute tile (`m`, `k`, `n`), proven on npu2 (AIE2P).
///
/// The bf16→fp32 kernel's micro-tile is (4, 8, 8); this L1-fitting macro-tile is a multiple of it,
/// and every admitted `(M, K, N)` is a positive multiple of this tile — the single tiling the
/// helper compiles and the hardware tests exercise. Untested shapes are rejected (issue #90).
/// The FP32 output is 4 B/element, so this tile is smaller than a same-shape bf16 tile would be, to
/// keep the double-buffered C tile plus the A/B tiles inside the compute core's ~64 KiB L1. The
/// Rust compiler driver carries this tested tile into the helper spec.
pub(crate) const MATMUL_TILE_M: usize = 32;
pub(crate) const MATMUL_TILE_K: usize = 64;
pub(crate) const MATMUL_TILE_N: usize = 32;

/// Largest admitted MATMUL dimension. The tested tiling generalizes across multiples of the tile,
/// but only within this envelope; larger shapes are a later generalization and are rejected now.
pub(crate) const MATMUL_MAX_DIM: usize = 512;

/// Largest combined input/output footprint for the exact INT8 MATMUL specialization.
///
/// The initial kernel keeps complete A, B, and C tensors in one AIE2P core. Limiting their
/// undoubled footprint to 16 KiB leaves room for depth-two FIFOs, code, and bookkeeping in its
/// roughly 64 KiB local memory. This is stricter than OpenVINO because it is an XDNA memory limit.
pub(crate) const INT8_MATMUL_MAX_TOTAL_BYTES: usize = 16 * 1024;

/// Largest combined direct-bound INT32 input and padded INT8 output for one RESCALE worker.
///
/// Like the first INT8 MATMUL path, the worker keeps complete depth-two objects in one AIE2P
/// core. This bound leaves room for code and bookkeeping and is rejected before compilation.
pub(crate) const INT8_RESCALE_MAX_TOTAL_BYTES: usize = 16 * 1024;

/// Maximum admitted pooling kernel and stride dimensions. Larger windows are unproven on the
/// scalar AIE2P kernel and are rejected before the compiler subprocess runs.
pub(crate) const MAX_POOL_MAX_KERNEL: usize = 8;
pub(crate) const MAX_POOL_MAX_STRIDE: usize = 8;

/// Maximum combined input/output BF16 elements for one pooling specialization.
///
/// The worker keeps depth-two input and output objects in a compute core's roughly 64 KiB local
/// memory. Capping the undoubled footprint at 16 KiB leaves half of L1 for code and bookkeeping.
pub(crate) const MAX_POOL_MAX_TOTAL_ELEMENTS: usize = 8 * 1024;

/// A validated operator specialization ready for the compiler helper. Each variant names its input
/// and output dtypes; the closed shape is integers only, so no guest bytes cross the boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CompilerSpec {
    /// BF16 → BF16 elementwise copy of `elements` values (a positive multiple of
    /// 1,024 values).
    Identity { elements: usize },
    /// Exact INT8 → INT8 elementwise copy. `line_size` is the complete DMA line selected by
    /// admission; no tail staging or dtype conversion is permitted.
    Int8Identity { elements: usize, line_size: usize },
    /// Explicit FP8 storage conversion to BF16. Every finite source value is exactly
    /// representable, so the output is bit-exact except for permitted NaN canonicalization.
    Fp8ToBf16 { format: Fp8Format, elements: usize },
    /// BF16 × BF16 → FP32 matrix multiply `C[M, N] = A[M, K] · B[K, N]` (batch 1). Each of `m`,
    /// `k`, `n` is a positive multiple of the corresponding MATMUL tile dimension and at most
    /// 512. The FP32 output is the TOSA-mandated accumulator (issue #82).
    Matmul { m: usize, k: usize, n: usize },
    /// Exact zero-point-aware INT8 × INT8 → INT32 matrix multiply (batch 1).
    ///
    /// The serialized TOSA zero points are part of the specialization and therefore also part of
    /// the compiler cache key. Arithmetic is `(a - a_zp) * (b - b_zp)` accumulated in INT32.
    Int8Matmul {
        m: usize,
        k: usize,
        n: usize,
        left_zero_point: i8,
        right_zero_point: i8,
    },
    /// Exact signed INT32 → INT8 scale32 RESCALE with one shared multiplier and shift.
    Int32ToInt8Rescale {
        elements: usize,
        multiplier: i32,
        shift: i8,
        output_zero_point: i8,
    },
    /// Batch-1 BF16 NHWC MAX_POOL2D with zero padding and propagating NaNs. The complete static
    /// specialization is carried to the helper so no TOSA bytes cross the subprocess boundary.
    MaxPool2d {
        input_h: usize,
        input_w: usize,
        channels: usize,
        output_h: usize,
        output_w: usize,
        kernel_h: usize,
        kernel_w: usize,
        stride_h: usize,
        stride_w: usize,
    },
}

/// The two TOSA/OCP FP8 storage encodings accepted by the conversion template.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Fp8Format {
    E4M3,
    E5M2,
}

/// Why a TOSA artifact is not admissible to the compilable subset.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdmitError {
    /// The bytes are not a valid TOSA artifact.
    Parse,
    /// The graph is valid TOSA but not for the requested target.
    Analysis,
    /// The graph is admissible TOSA but outside the compilable subset.
    Unsupported,
}

impl core::fmt::Display for AdmitError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for AdmitError {}

/// The one authoritative mapping to wire-level error codes, shared by `load_program` and
/// `compile_artifact` so the two paths can never drift. The codes match the OpenVINO backend's
/// classification of the same failure classes (its `lowering_error`): a malformed or semantically
/// invalid artifact is the guest's mistake (`InvalidArgument`); a valid graph this backend cannot
/// execute is `Unsupported`.
impl From<AdmitError> for virtio_accel_core::BackendError {
    fn from(error: AdmitError) -> Self {
        match error {
            AdmitError::Parse | AdmitError::Analysis => Self::InvalidArgument,
            AdmitError::Unsupported => Self::Unsupported,
        }
    }
}

/// Admit a TOSA artifact for `target`, returning the specialization the helper compiles.
///
/// The BF16 target admits IDENTITY (a DMA copy), BF16 → FP32 MATMUL, and BF16 NHWC MAX_POOL2D.
/// The separate FP8 storage target admits one explicit FP8 → BF16 CAST. The integer target admits
/// exact INT8 IDENTITY, zero-point-aware INT8 → INT32 MATMUL, and static signed INT32 → INT8
/// RESCALE. Everything else is rejected without running the compiler. Each template admits only
/// graphs whose **dataflow** matches what
/// the compiled kernel executes — the IDENTITY
/// template requires every operator to be IDENTITY (no constants: with a single block input, every
/// value then provably carries that input's bytes), and the MATMUL template requires the operator's
/// operands to be exactly the block inputs (constants may exist only as the two zero-points).
/// Without these checks a semantically different graph (say, a constant-output IDENTITY or a
/// constant-weights MATMUL) would compile to a kernel that reads runtime buffers the graph never
/// asked for, returning well-formed but wrong data. Semantic and target validity — including that
/// BF16 MATMUL zero-points are constant zero — is enforced by
/// [`analyze_for`](virtio_accel_tosa::Model::analyze_for) before these structural checks.
pub fn admit(bytes: &[u8], target: Target) -> Result<CompilerSpec, AdmitError> {
    if target != XDNA_TOSA_TARGET
        && target != XDNA_TOSA_FP8_TARGET
        && target != XDNA_TOSA_INTEGER_TARGET
    {
        return Err(AdmitError::Unsupported);
    }
    let model = parse(bytes).map_err(|_| AdmitError::Parse)?;
    let analysis = model
        .analyze_for(target)
        .map_err(|_| AdmitError::Analysis)?;

    if analysis.regions().len() != 1
        || analysis.blocks().len() != 1
        || !analysis.conditions().is_empty()
    {
        return Err(AdmitError::Unsupported);
    }
    let block = analysis.blocks()[0].id();

    // Classify in one pass. IDENTITY and CAST tolerate no other operator kind (not even CONST);
    // MATMUL tolerates exactly one MATMUL plus CONST operators, which `admit_matmul` then pins down
    // to the two zero-points.
    let mut matmul = None;
    let mut max_pool = None;
    let mut cast = None;
    let mut rescale = None;
    let mut identities = 0usize;
    let mut constants = 0usize;
    for operator in analysis.execution_order(block) {
        match analysis.operator(*operator).op() {
            Op::IDENTITY => identities += 1,
            Op::CONST => constants += 1,
            Op::MATMUL if matmul.is_none() => matmul = Some(*operator),
            Op::MAX_POOL2D if max_pool.is_none() => max_pool = Some(*operator),
            Op::CAST if cast.is_none() => cast = Some(*operator),
            Op::RESCALE if rescale.is_none() => rescale = Some(*operator),
            _ => return Err(AdmitError::Unsupported),
        }
    }
    match (
        target, matmul, max_pool, cast, rescale, identities, constants,
    ) {
        // All-IDENTITY (zero operators included: the block output then *is* the block input, and a
        // DMA copy is exact for it).
        (XDNA_TOSA_TARGET, None, None, None, None, _, 0) => admit_identity(&analysis, block),
        (XDNA_TOSA_TARGET, Some(matmul), None, None, None, 0, _) => {
            admit_matmul(&analysis, block, matmul)
        }
        (XDNA_TOSA_TARGET, None, Some(max_pool), None, None, 0, 0) => {
            admit_max_pool2d(&analysis, block, max_pool)
        }
        (XDNA_TOSA_FP8_TARGET, None, None, Some(cast), None, 0, 0) => {
            admit_fp8_to_bf16(&analysis, block, cast)
        }
        (XDNA_TOSA_INTEGER_TARGET, None, None, None, None, _, 0) => {
            admit_int8_identity(&analysis, block)
        }
        (XDNA_TOSA_INTEGER_TARGET, Some(matmul), None, None, None, 0, _) => {
            admit_int8_matmul(&analysis, block, matmul)
        }
        (XDNA_TOSA_INTEGER_TARGET, None, None, None, Some(rescale), 0, 4) => {
            admit_int32_to_int8_rescale(&analysis, block, rescale)
        }
        _ => Err(AdmitError::Unsupported),
    }
}

/// Admit exact INT8 IDENTITY, including the shared eight-byte corpus fixture.
///
/// Unlike BF16's established 1,024-element template, a small INT8 tensor is one complete DMA line.
/// Larger tensors must be an exact multiple of the maximum line so no tail is staged on the host.
fn admit_int8_identity(
    analysis: &virtio_accel_tosa::TosaAnalysis<'_>,
    block: virtio_accel_tosa::BlockId,
) -> Result<CompilerSpec, AdmitError> {
    let inputs = analysis.block_inputs(block);
    let outputs = analysis.block_outputs(block);
    if inputs.len() != 1 || outputs.len() != 1 {
        return Err(AdmitError::Unsupported);
    }
    for value in analysis.values() {
        if let AnalyzedValueKind::Tensor(tensor) = value.kind() {
            if tensor.dtype() != DType::INT8 {
                return Err(AdmitError::Unsupported);
            }
        }
    }

    let elements = tensor_elements(analysis, outputs[0])?;
    let line_size = elements.min(INT8_IDENTITY_MAX_LINE_SIZE);
    if line_size % 4 != 0 || (elements > INT8_IDENTITY_MAX_LINE_SIZE && elements % line_size != 0) {
        return Err(AdmitError::Unsupported);
    }
    Ok(CompilerSpec::Int8Identity {
        elements,
        line_size,
    })
}

/// Admit exact zero-point-aware INT8 MATMUL with the same graph semantics as OpenVINO.
///
/// XDNA specializes the two serialized rank-1 INT8 zero points into the device kernel because the
/// AIE kernel API has no provider graph in which to insert OpenVINO's widen-and-subtract nodes.
/// Both paths compute the same TOSA expression; XDNA never adjusts values on the host.
fn admit_int8_matmul(
    analysis: &virtio_accel_tosa::TosaAnalysis<'_>,
    block: virtio_accel_tosa::BlockId,
    matmul: virtio_accel_tosa::OperatorId,
) -> Result<CompilerSpec, AdmitError> {
    let inputs = analysis.operator_inputs(matmul);
    let outputs = analysis.operator_outputs(matmul);
    if inputs.len() != 4
        || outputs.len() != 1
        // The compiled kernel always declares two independent input slots, so one value feeding
        // both operands would let a caller bind different buffers and compute `A * B` for a graph
        // that says `X * X`.
        || inputs[0] == inputs[1]
        || analysis.block_inputs(block) != [inputs[0], inputs[1]]
        || analysis.block_outputs(block) != [outputs[0]]
    {
        return Err(AdmitError::Unsupported);
    }
    for operator in analysis.execution_order(block) {
        if analysis.operator(*operator).op() != Op::CONST {
            continue;
        }
        for produced in analysis.operator_outputs(*operator) {
            if *produced != inputs[2] && *produced != inputs[3] {
                return Err(AdmitError::Unsupported);
            }
        }
    }

    let lhs = matmul_dims(analysis, inputs[0], DType::INT8)?;
    let rhs = matmul_dims(analysis, inputs[1], DType::INT8)?;
    let out = matmul_dims(analysis, outputs[0], DType::INT32)?;
    let ([1, m, k], [1, k2, n], [1, m2, n2]) = (lhs, rhs, out) else {
        return Err(AdmitError::Unsupported);
    };
    if k != k2 || m != m2 || n != n2 || [m, k, n].iter().any(|dim| *dim > MATMUL_MAX_DIM) {
        return Err(AdmitError::Unsupported);
    }

    // AIE DMA descriptors transfer whole 32-bit words. The direct-binding ABI therefore rounds
    // each INT8 input slot up to four bytes; the kernel ignores those explicit padding bytes.
    let lhs_bytes = align_to_four(m.checked_mul(k).ok_or(AdmitError::Unsupported)?)?;
    let rhs_bytes = align_to_four(k.checked_mul(n).ok_or(AdmitError::Unsupported)?)?;
    let output_bytes = m
        .checked_mul(n)
        .and_then(|elements| elements.checked_mul(4))
        .ok_or(AdmitError::Unsupported)?;
    if lhs_bytes
        .checked_add(rhs_bytes)
        .and_then(|bytes| bytes.checked_add(output_bytes))
        .is_none_or(|bytes| bytes > INT8_MATMUL_MAX_TOTAL_BYTES)
    {
        return Err(AdmitError::Unsupported);
    }

    Ok(CompilerSpec::Int8Matmul {
        m,
        k,
        n,
        left_zero_point: int8_zero_point(analysis, inputs[2])?,
        right_zero_point: int8_zero_point(analysis, inputs[3])?,
    })
}

/// Admit the released signed scale32 RESCALE row without OpenVINO-style implicit conversion.
///
/// OpenVINO does not advertise RESCALE in its conservative integer tier. XDNA deliberately extends
/// the operator surface here, while retaining the same separate target, static lowering boundary,
/// direct binding, and reject-don't-fallback behavior. Per-channel, unsigned, double-round, and
/// inexact-round forms remain unsupported until they have separate kernels and corpus proof.
fn admit_int32_to_int8_rescale(
    analysis: &virtio_accel_tosa::TosaAnalysis<'_>,
    block: virtio_accel_tosa::BlockId,
    rescale: virtio_accel_tosa::OperatorId,
) -> Result<CompilerSpec, AdmitError> {
    let inputs = analysis.operator_inputs(rescale);
    let outputs = analysis.operator_outputs(rescale);
    if inputs.len() != 5
        || outputs.len() != 1
        || analysis.block_inputs(block) != [inputs[0]]
        || analysis.block_outputs(block) != [outputs[0]]
    {
        return Err(AdmitError::Unsupported);
    }
    for operator in analysis.execution_order(block) {
        if analysis.operator(*operator).op() != Op::CONST {
            continue;
        }
        for produced in analysis.operator_outputs(*operator) {
            if !inputs[1..].contains(produced) {
                return Err(AdmitError::Unsupported);
            }
        }
    }

    let AnalyzedValueKind::Tensor(input) = analysis.value(inputs[0]).kind() else {
        return Err(AdmitError::Unsupported);
    };
    let AnalyzedValueKind::Tensor(output) = analysis.value(outputs[0]).kind() else {
        return Err(AdmitError::Unsupported);
    };
    if input.dtype() != DType::INT32
        || output.dtype() != DType::INT8
        || !input.dimensions().eq(output.dimensions())
    {
        return Err(AdmitError::Unsupported);
    }
    let OpAttributes::Rescale {
        scale32,
        rounding_mode,
        per_channel,
        input_unsigned,
        output_unsigned,
    } = analysis.operator(rescale).source().attributes()
    else {
        return Err(AdmitError::Unsupported);
    };
    if !scale32
        || rounding_mode != RoundingMode::SINGLE_ROUND
        || per_channel
        || input_unsigned
        || output_unsigned
    {
        return Err(AdmitError::Unsupported);
    }

    let multiplier = int32_constant(analysis, inputs[1])?;
    let shift = int8_zero_point(analysis, inputs[2])?;
    let input_zero_point = int32_constant(analysis, inputs[3])?;
    let output_zero_point = int8_zero_point(analysis, inputs[4])?;
    if multiplier < 0 || !(2..=62).contains(&shift) || input_zero_point != 0 {
        return Err(AdmitError::Unsupported);
    }

    let elements = tensor_elements(analysis, inputs[0])?;
    let input_bytes = elements.checked_mul(4).ok_or(AdmitError::Unsupported)?;
    let output_bytes = align_to_four(elements)?;
    if input_bytes
        .checked_add(output_bytes)
        .is_none_or(|bytes| bytes > INT8_RESCALE_MAX_TOTAL_BYTES)
    {
        return Err(AdmitError::Unsupported);
    }
    Ok(CompilerSpec::Int32ToInt8Rescale {
        elements,
        multiplier,
        shift,
        output_zero_point,
    })
}

fn align_to_four(bytes: usize) -> Result<usize, AdmitError> {
    bytes
        .checked_add(3)
        .map(|bytes| bytes & !3)
        .ok_or(AdmitError::Unsupported)
}

fn int32_constant(
    analysis: &virtio_accel_tosa::TosaAnalysis<'_>,
    value: virtio_accel_tosa::ValueId,
) -> Result<i32, AdmitError> {
    let AnalyzedValueKind::Tensor(tensor) = analysis.value(value).kind() else {
        return Err(AdmitError::Unsupported);
    };
    if tensor.dtype() != DType::INT32 || tensor.dimensions().ne([1]) {
        return Err(AdmitError::Unsupported);
    }
    let bytes: [u8; 4] = analysis
        .serialized_constant(value)
        .ok_or(AdmitError::Unsupported)?
        .try_into()
        .map_err(|_| AdmitError::Unsupported)?;
    Ok(i32::from_le_bytes(bytes))
}

fn int8_zero_point(
    analysis: &virtio_accel_tosa::TosaAnalysis<'_>,
    value: virtio_accel_tosa::ValueId,
) -> Result<i8, AdmitError> {
    let AnalyzedValueKind::Tensor(tensor) = analysis.value(value).kind() else {
        return Err(AdmitError::Unsupported);
    };
    if tensor.dtype() != DType::INT8 || tensor.dimensions().ne([1]) {
        return Err(AdmitError::Unsupported);
    }
    let [byte] = analysis
        .serialized_constant(value)
        .ok_or(AdmitError::Unsupported)?
    else {
        return Err(AdmitError::Unsupported);
    };
    Ok(*byte as i8)
}

fn tensor_elements(
    analysis: &virtio_accel_tosa::TosaAnalysis<'_>,
    value: virtio_accel_tosa::ValueId,
) -> Result<usize, AdmitError> {
    let AnalyzedValueKind::Tensor(tensor) = analysis.value(value).kind() else {
        return Err(AdmitError::Unsupported);
    };
    let mut elements = 1usize;
    for dimension in tensor.dimensions() {
        let dimension = usize::try_from(dimension).map_err(|_| AdmitError::Unsupported)?;
        if dimension == 0 {
            return Err(AdmitError::Unsupported);
        }
        elements = elements
            .checked_mul(dimension)
            .ok_or(AdmitError::Unsupported)?;
    }
    Ok(elements)
}

/// Admit one explicit FP8 → BF16 CAST directly connecting the block input and output.
///
/// FP8 is storage only: no FP8 arithmetic is inferred or hidden. The guest-visible CAST is the
/// exact point where the NPU expands each element, after which existing BF16 compute kernels can
/// consume the result in a subsequent program.
fn admit_fp8_to_bf16(
    analysis: &virtio_accel_tosa::TosaAnalysis<'_>,
    block: virtio_accel_tosa::BlockId,
    cast: virtio_accel_tosa::OperatorId,
) -> Result<CompilerSpec, AdmitError> {
    let inputs = analysis.operator_inputs(cast);
    let outputs = analysis.operator_outputs(cast);
    if inputs.len() != 1
        || outputs.len() != 1
        || analysis.block_inputs(block) != [inputs[0]]
        || analysis.block_outputs(block) != [outputs[0]]
    {
        return Err(AdmitError::Unsupported);
    }

    let AnalyzedValueKind::Tensor(input) = analysis.value(inputs[0]).kind() else {
        return Err(AdmitError::Unsupported);
    };
    let AnalyzedValueKind::Tensor(output) = analysis.value(outputs[0]).kind() else {
        return Err(AdmitError::Unsupported);
    };
    let format = match input.dtype() {
        DType::FP8E4M3 => Fp8Format::E4M3,
        DType::FP8E5M2 => Fp8Format::E5M2,
        _ => return Err(AdmitError::Unsupported),
    };
    if output.dtype() != DType::BF16 || input.dimensions().ne(output.dimensions()) {
        return Err(AdmitError::Unsupported);
    }

    let mut elements = 1usize;
    for dimension in output.dimensions() {
        let dimension = usize::try_from(dimension).map_err(|_| AdmitError::Unsupported)?;
        if dimension == 0 {
            return Err(AdmitError::Unsupported);
        }
        elements = elements
            .checked_mul(dimension)
            .ok_or(AdmitError::Unsupported)?;
    }
    if elements % FP8_CAST_LINE_SIZE != 0 {
        return Err(AdmitError::Unsupported);
    }

    Ok(CompilerSpec::Fp8ToBf16 { format, elements })
}

/// Admit the BF16 IDENTITY subset: one BF16 input and output, IDENTITY operators only (already
/// established by the caller — with no constants and one block input, every value in the block
/// carries the input's bytes, so the output equals the input under any operator arrangement),
/// every tensor BF16, and an element count that is a positive multiple of [`IDENTITY_LINE_SIZE`].
fn admit_identity(
    analysis: &virtio_accel_tosa::TosaAnalysis<'_>,
    block: virtio_accel_tosa::BlockId,
) -> Result<CompilerSpec, AdmitError> {
    let inputs = analysis.block_inputs(block);
    let outputs = analysis.block_outputs(block);
    if inputs.len() != 1 || outputs.len() != 1 {
        return Err(AdmitError::Unsupported);
    }
    for value in analysis.values() {
        if let AnalyzedValueKind::Tensor(tensor) = value.kind() {
            if tensor.dtype() != DType::BF16 {
                return Err(AdmitError::Unsupported);
            }
        }
    }

    let AnalyzedValueKind::Tensor(output) = analysis.value(outputs[0]).kind() else {
        return Err(AdmitError::Unsupported);
    };
    output.rank().ok_or(AdmitError::Unsupported)?;
    let mut elements: usize = 1;
    for dimension in output.dimensions() {
        // Negative (dynamic) dimensions must not sign-extend into a huge count.
        let dimension = usize::try_from(dimension).map_err(|_| AdmitError::Unsupported)?;
        elements = elements
            .checked_mul(dimension)
            .ok_or(AdmitError::Unsupported)?;
    }
    if elements == 0 || elements % IDENTITY_LINE_SIZE != 0 {
        return Err(AdmitError::Unsupported);
    }

    Ok(CompilerSpec::Identity { elements })
}

/// Admit the BF16 → FP32 MATMUL subset: `lhs`/`rhs` BF16 rank-3 `[1, M, K]`/`[1, K, N]`, output
/// FP32 rank-3 `[1, M, N]`, and each of `M`, `K`, `N` a positive multiple of the tested tile within
/// [`MATMUL_MAX_DIM`]. TOSA's MATMUL carries four inputs — `lhs`, `rhs`, and the two zero-points —
/// and one output; the zero-points' constant-zero requirement was already enforced by analysis.
///
/// The dataflow must be exactly the compiled kernel's: `lhs`/`rhs` are the block inputs (in slot
/// order — the runtime binds A to slot 0 and B to slot 1), the MATMUL output is the block output,
/// and every CONST in the graph produces only the two zero-points. A CONST-produced operand (baked
/// weights) or a CONST-produced block output is a semantically different program and is rejected.
fn admit_matmul(
    analysis: &virtio_accel_tosa::TosaAnalysis<'_>,
    block: virtio_accel_tosa::BlockId,
    matmul: virtio_accel_tosa::OperatorId,
) -> Result<CompilerSpec, AdmitError> {
    let inputs = analysis.operator_inputs(matmul);
    let outputs = analysis.operator_outputs(matmul);
    if inputs.len() != 4 || outputs.len() != 1 {
        return Err(AdmitError::Unsupported);
    }

    // The operator's dataflow must be the block's: lhs/rhs are the block inputs in binding order,
    // and the MATMUL result is the block output. One value feeding both operands is rejected: the
    // compiled kernel declares two independent slots, so a caller could bind different buffers and
    // compute `A * B` for a graph that says `X * X`.
    if inputs[0] == inputs[1]
        || analysis.block_inputs(block) != [inputs[0], inputs[1]]
        || analysis.block_outputs(block) != [outputs[0]]
    {
        return Err(AdmitError::Unsupported);
    }
    // Every CONST feeds only the zero-points (operands 2 and 3); any other constant value would be
    // graph state the compiled kernel cannot reproduce.
    for operator in analysis.execution_order(block) {
        if analysis.operator(*operator).op() != Op::CONST {
            continue;
        }
        for produced in analysis.operator_outputs(*operator) {
            if *produced != inputs[2] && *produced != inputs[3] {
                return Err(AdmitError::Unsupported);
            }
        }
    }

    let lhs = matmul_dims(analysis, inputs[0], DType::BF16)?;
    let rhs = matmul_dims(analysis, inputs[1], DType::BF16)?;
    let out = matmul_dims(analysis, outputs[0], DType::FP32)?;

    // Batch 1 only (the tested tiling), and the shared dimensions must agree: A[1,M,K], B[1,K,N],
    // C[1,M,N].
    let ([1, m, k], [1, k2, n], [1, m2, n2]) = (lhs, rhs, out) else {
        return Err(AdmitError::Unsupported);
    };
    if k != k2 || m != m2 || n != n2 {
        return Err(AdmitError::Unsupported);
    }
    if !tile_admissible(m, MATMUL_TILE_M)
        || !tile_admissible(k, MATMUL_TILE_K)
        || !tile_admissible(n, MATMUL_TILE_N)
    {
        return Err(AdmitError::Unsupported);
    }

    Ok(CompilerSpec::Matmul { m, k, n })
}

/// The rank-3 dimensions of `value`, requiring the given dtype and every dimension statically
/// positive. Dynamic (non-positive) dimensions and non-tensor or wrong-dtype values are rejected.
fn matmul_dims(
    analysis: &virtio_accel_tosa::TosaAnalysis<'_>,
    value: virtio_accel_tosa::ValueId,
    dtype: DType,
) -> Result<[usize; 3], AdmitError> {
    let AnalyzedValueKind::Tensor(tensor) = analysis.value(value).kind() else {
        return Err(AdmitError::Unsupported);
    };
    if tensor.dtype() != dtype || tensor.rank() != Some(3) {
        return Err(AdmitError::Unsupported);
    }
    let mut dims = [0usize; 3];
    for (slot, dimension) in dims.iter_mut().zip(tensor.dimensions()) {
        *slot = usize::try_from(dimension).map_err(|_| AdmitError::Unsupported)?;
        if *slot == 0 {
            return Err(AdmitError::Unsupported);
        }
    }
    Ok(dims)
}

/// A dimension is admissible when it is a positive multiple of the tested tile within the envelope.
fn tile_admissible(dim: usize, tile: usize) -> bool {
    dim > 0 && dim % tile == 0 && dim <= MATMUL_MAX_DIM
}

/// Admit one batch-1 BF16 NHWC MAX_POOL2D directly connecting the block input and output.
///
/// OpenVINO accepts the same propagating-NaN and zero-padding semantic envelope. XDNA narrows it
/// further to bounded static tensors and small positive kernels/strides because each complete
/// tensor is double-buffered in one AIE2P compute core's local memory.
fn admit_max_pool2d(
    analysis: &virtio_accel_tosa::TosaAnalysis<'_>,
    block: virtio_accel_tosa::BlockId,
    max_pool: virtio_accel_tosa::OperatorId,
) -> Result<CompilerSpec, AdmitError> {
    let inputs = analysis.operator_inputs(max_pool);
    let outputs = analysis.operator_outputs(max_pool);
    if inputs.len() != 1
        || outputs.len() != 1
        || analysis.block_inputs(block) != [inputs[0]]
        || analysis.block_outputs(block) != [outputs[0]]
    {
        return Err(AdmitError::Unsupported);
    }

    let OpAttributes::MaxPool2d {
        kernel,
        stride,
        pad,
        nan_mode,
    } = analysis.operator(max_pool).source().attributes()
    else {
        return Err(AdmitError::Unsupported);
    };
    if nan_mode != NanPropagationMode::PROPAGATE {
        return Err(AdmitError::Unsupported);
    }
    let kernel = exact_positive_pair(kernel.iter(), MAX_POOL_MAX_KERNEL)?;
    let stride = exact_positive_pair(stride.iter(), MAX_POOL_MAX_STRIDE)?;
    let pad: Vec<_> = pad.iter().collect();
    if pad != [0, 0, 0, 0] {
        return Err(AdmitError::Unsupported);
    }

    let [batch, input_h, input_w, channels] = pool_dims(analysis, inputs[0])?;
    let [output_batch, output_h, output_w, output_channels] = pool_dims(analysis, outputs[0])?;
    if batch != 1 || output_batch != 1 || channels != output_channels {
        return Err(AdmitError::Unsupported);
    }
    let input_elements = input_h
        .checked_mul(input_w)
        .and_then(|elements| elements.checked_mul(channels))
        .ok_or(AdmitError::Unsupported)?;
    let output_elements = output_h
        .checked_mul(output_w)
        .and_then(|elements| elements.checked_mul(channels))
        .ok_or(AdmitError::Unsupported)?;
    if input_elements
        .checked_add(output_elements)
        .is_none_or(|total| total > MAX_POOL_MAX_TOTAL_ELEMENTS)
    {
        return Err(AdmitError::Unsupported);
    }

    Ok(CompilerSpec::MaxPool2d {
        input_h,
        input_w,
        channels,
        output_h,
        output_w,
        kernel_h: kernel[0],
        kernel_w: kernel[1],
        stride_h: stride[0],
        stride_w: stride[1],
    })
}

fn exact_positive_pair(
    values: impl Iterator<Item = i32>,
    maximum: usize,
) -> Result<[usize; 2], AdmitError> {
    let values: Vec<_> = values.collect();
    let [first, second] = values.as_slice() else {
        return Err(AdmitError::Unsupported);
    };
    let first = usize::try_from(*first).map_err(|_| AdmitError::Unsupported)?;
    let second = usize::try_from(*second).map_err(|_| AdmitError::Unsupported)?;
    if first == 0 || second == 0 || first > maximum || second > maximum {
        return Err(AdmitError::Unsupported);
    }
    Ok([first, second])
}

fn pool_dims(
    analysis: &virtio_accel_tosa::TosaAnalysis<'_>,
    value: virtio_accel_tosa::ValueId,
) -> Result<[usize; 4], AdmitError> {
    let AnalyzedValueKind::Tensor(tensor) = analysis.value(value).kind() else {
        return Err(AdmitError::Unsupported);
    };
    if tensor.dtype() != DType::BF16 || tensor.rank() != Some(4) {
        return Err(AdmitError::Unsupported);
    }
    let mut dims = [0usize; 4];
    for (slot, dimension) in dims.iter_mut().zip(tensor.dimensions()) {
        *slot = usize::try_from(dimension).map_err(|_| AdmitError::Unsupported)?;
        if *slot == 0 {
            return Err(AdmitError::Unsupported);
        }
    }
    Ok(dims)
}

#[cfg(test)]
mod tests {
    use super::*;
    use virtio_accel_conformance::numerics::RESCALE_INT32_TO_INT8;
    use virtio_accel_tosa::Target;
    use virtio_accel_tosa_build::{OperatorKind, OwnedGraph, OwnedOperator, OwnedTensor};

    fn identity_graph(dtype: DType, shape: Vec<i32>) -> OwnedGraph<'static> {
        let mut graph = OwnedGraph::new("main");
        graph.push_tensor(OwnedTensor::new("x", shape.clone(), dtype));
        graph.push_tensor(OwnedTensor::new("y", shape, dtype));
        graph.push_operator(OwnedOperator::new(
            OperatorKind::Identity,
            vec!["x".into()],
            vec!["y".into()],
        ));
        graph.push_input("x");
        graph.push_output("y");
        graph
    }

    fn fp8_cast_graph(input: DType, output: DType, shape: Vec<i32>) -> OwnedGraph<'static> {
        let mut graph = OwnedGraph::new("main");
        graph
            .push_tensor(OwnedTensor::new("x", shape.clone(), input))
            .push_tensor(OwnedTensor::new("y", shape, output))
            .push_operator(OwnedOperator::new(
                OperatorKind::Cast,
                vec!["x".into()],
                vec!["y".into()],
            ))
            .push_input("x")
            .push_output("y");
        graph
    }

    /// A batch-1 MATMUL `C[1,M,N] = A[1,M,K] · B[1,K,N]` with the two constant-zero zero-points.
    fn matmul_graph(
        m: i32,
        k: i32,
        n: i32,
        in_dtype: DType,
        out_dtype: DType,
    ) -> OwnedGraph<'static> {
        matmul_graph_with_zero_points(m, k, n, in_dtype, out_dtype, 0, 0)
    }

    fn matmul_graph_with_zero_points(
        m: i32,
        k: i32,
        n: i32,
        in_dtype: DType,
        out_dtype: DType,
        left_zero_point: i8,
        right_zero_point: i8,
    ) -> OwnedGraph<'static> {
        let zero_point = |dtype: DType, value: i8| match dtype {
            DType::INT8 => vec![value as u8],
            DType::BF16 => vec![0u8; 2],
            DType::FP32 => vec![0u8; 4],
            _ => Vec::new(),
        };
        let mut graph = OwnedGraph::new("main");
        graph
            .push_tensor(OwnedTensor::new("lhs", vec![1, m, k], in_dtype))
            .push_tensor(OwnedTensor::new("rhs", vec![1, k, n], in_dtype))
            .push_tensor(OwnedTensor::constant(
                "lhs_zp",
                vec![1],
                in_dtype,
                zero_point(in_dtype, left_zero_point),
            ))
            .push_tensor(OwnedTensor::constant(
                "rhs_zp",
                vec![1],
                in_dtype,
                zero_point(in_dtype, right_zero_point),
            ))
            .push_tensor(OwnedTensor::new("output", vec![1, m, n], out_dtype))
            .push_operator(OwnedOperator::new(
                OperatorKind::Const,
                vec![],
                vec!["lhs_zp".into()],
            ))
            .push_operator(OwnedOperator::new(
                OperatorKind::Const,
                vec![],
                vec!["rhs_zp".into()],
            ))
            .push_operator(OwnedOperator::new(
                OperatorKind::MatMul,
                vec!["lhs".into(), "rhs".into(), "lhs_zp".into(), "rhs_zp".into()],
                vec!["output".into()],
            ))
            .push_input("lhs")
            .push_input("rhs")
            .push_output("output");
        graph
    }

    fn rescale_graph(
        elements: i32,
        multiplier: i32,
        shift: i8,
        per_channel: bool,
        rounding_mode: RoundingMode,
    ) -> OwnedGraph<'static> {
        let mut graph = OwnedGraph::new("main");
        graph
            .push_tensor(OwnedTensor::new("input", vec![elements], DType::INT32))
            .push_tensor(OwnedTensor::constant(
                "multiplier",
                vec![1],
                DType::INT32,
                multiplier.to_le_bytes().to_vec(),
            ))
            .push_tensor(OwnedTensor::constant(
                "shift",
                vec![1],
                DType::INT8,
                vec![shift as u8],
            ))
            .push_tensor(OwnedTensor::constant(
                "input_zp",
                vec![1],
                DType::INT32,
                0_i32.to_le_bytes().to_vec(),
            ))
            .push_tensor(OwnedTensor::constant(
                "output_zp",
                vec![1],
                DType::INT8,
                vec![(-3_i8) as u8],
            ))
            .push_tensor(OwnedTensor::new("output", vec![elements], DType::INT8));
        for parameter in ["multiplier", "shift", "input_zp", "output_zp"] {
            graph.push_operator(OwnedOperator::new(
                OperatorKind::Const,
                vec![],
                vec![parameter.into()],
            ));
        }
        graph
            .push_operator(OwnedOperator::new(
                OperatorKind::Rescale {
                    scale32: true,
                    rounding_mode,
                    per_channel,
                    input_unsigned: false,
                    output_unsigned: false,
                },
                vec![
                    "input".into(),
                    "multiplier".into(),
                    "shift".into(),
                    "input_zp".into(),
                    "output_zp".into(),
                ],
                vec!["output".into()],
            ))
            .push_input("input")
            .push_output("output");
        graph
    }

    struct MaxPoolCase {
        input: [i32; 3],
        kernel: [i32; 2],
        stride: [i32; 2],
        pad: [i32; 4],
        dtype: DType,
        nan_mode: NanPropagationMode,
    }

    fn max_pool_graph(case: MaxPoolCase) -> OwnedGraph<'static> {
        let [input_h, input_w, channels] = case.input;
        let output_h = (input_h + case.pad[0] + case.pad[1] - case.kernel[0]) / case.stride[0] + 1;
        let output_w = (input_w + case.pad[2] + case.pad[3] - case.kernel[1]) / case.stride[1] + 1;
        let mut graph = OwnedGraph::new("main");
        graph
            .push_tensor(OwnedTensor::new(
                "input",
                vec![1, input_h, input_w, channels],
                case.dtype,
            ))
            .push_tensor(OwnedTensor::new(
                "output",
                vec![1, output_h, output_w, channels],
                case.dtype,
            ))
            .push_operator(OwnedOperator::new(
                OperatorKind::MaxPool2d {
                    kernel: case.kernel,
                    stride: case.stride,
                    pad: case.pad,
                    nan_mode: case.nan_mode,
                },
                vec!["input".into()],
                vec!["output".into()],
            ))
            .push_input("input")
            .push_output("output");
        graph
    }

    #[test]
    fn both_targets_are_coherent_and_distinct() {
        assert_eq!(XDNA_TOSA_TARGET.validate(), Ok(XDNA_TOSA_TARGET));
        assert_eq!(
            XDNA_TOSA_INTEGER_TARGET.validate(),
            Ok(XDNA_TOSA_INTEGER_TARGET)
        );
        assert_ne!(XDNA_TOSA_TARGET, XDNA_TOSA_INTEGER_TARGET);
        for target in [
            XDNA_TOSA_TARGET,
            XDNA_TOSA_FP8_TARGET,
            XDNA_TOSA_INTEGER_TARGET,
        ] {
            assert_eq!(Target::from_identity(target.to_identity()), Ok(target));
        }
    }

    #[test]
    fn admits_bf16_identity() {
        let bytes = identity_graph(DType::BF16, vec![1, 4, 1024])
            .build(XDNA_TOSA_TARGET)
            .expect("build bf16 identity");
        let spec = admit(&bytes, XDNA_TOSA_TARGET).expect("admit");
        assert_eq!(spec, CompilerSpec::Identity { elements: 4 * 1024 });
    }

    #[test]
    fn integer_capability_preserves_the_openvino_base_and_adds_rescale() {
        assert_eq!(
            XDNA_TOSA_INTEGER_CAPABILITY.target,
            XDNA_TOSA_INTEGER_TARGET
        );
        assert_eq!(XDNA_TOSA_INTEGER_CAPABILITY.dtypes, INTEGER_DTYPES);
        assert_eq!(XDNA_TOSA_INTEGER_CAPABILITY.operators, INTEGER_OPERATORS);
        for op in [Op::CONST, Op::IDENTITY, Op::MATMUL, Op::RESCALE] {
            assert!(XDNA_TOSA_INTEGER_CAPABILITY.supports_operator(op));
        }
        assert_eq!(XDNA_TOSA_INTEGER_CAPABILITY.graph.max_regions, 1);
        assert_eq!(XDNA_TOSA_INTEGER_CAPABILITY.graph.max_blocks, 1);
        assert_eq!(
            XDNA_TOSA_INTEGER_CAPABILITY.graph.runtime_conditions,
            RuntimeConditionSupport::None
        );
    }

    #[test]
    fn admits_shared_int8_identity_shape() {
        let bytes = identity_graph(DType::INT8, vec![8])
            .build(XDNA_TOSA_INTEGER_TARGET)
            .expect("build int8 identity");
        assert_eq!(
            admit(&bytes, XDNA_TOSA_INTEGER_TARGET),
            Ok(CompilerSpec::Int8Identity {
                elements: 8,
                line_size: 8,
            })
        );
    }

    #[test]
    fn admits_zero_point_aware_int8_matmul() {
        let bytes = matmul_graph_with_zero_points(2, 3, 2, DType::INT8, DType::INT32, -2, 3)
            .build(XDNA_TOSA_INTEGER_TARGET)
            .expect("build int8 matmul");
        assert_eq!(
            admit(&bytes, XDNA_TOSA_INTEGER_TARGET),
            Ok(CompilerSpec::Int8Matmul {
                m: 2,
                k: 3,
                n: 2,
                left_zero_point: -2,
                right_zero_point: 3,
            })
        );
    }

    #[test]
    fn admits_shared_exact_int32_to_int8_rescale() {
        assert_eq!(
            admit(RESCALE_INT32_TO_INT8.artifact, XDNA_TOSA_INTEGER_TARGET),
            Ok(CompilerSpec::Int32ToInt8Rescale {
                elements: 16,
                multiplier: 1 << 29,
                shift: 30,
                output_zero_point: -3,
            })
        );
    }

    #[test]
    fn rescale_rejects_unimplemented_modes_and_invalid_parameters() {
        let per_channel = rescale_graph(1, 1 << 29, 30, true, RoundingMode::SINGLE_ROUND)
            .build(XDNA_TOSA_INTEGER_TARGET)
            .expect("per-channel one-element RESCALE is valid TOSA");
        assert_eq!(
            admit(&per_channel, XDNA_TOSA_INTEGER_TARGET),
            Err(AdmitError::Unsupported)
        );

        assert!(
            rescale_graph(16, 1 << 29, 1, false, RoundingMode::SINGLE_ROUND)
                .build(XDNA_TOSA_INTEGER_TARGET)
                .is_err(),
            "shift 1 violates the released RESCALE range"
        );

        assert!(
            rescale_graph(16, 1 << 29, 30, false, RoundingMode::DOUBLE_ROUND)
                .build(XDNA_TOSA_INTEGER_TARGET)
                .is_err(),
            "DOUBLE_ROUND requires an extension absent from the integer target"
        );
    }

    #[test]
    fn rejects_int8_shapes_outside_the_one_core_envelope() {
        let non_word_identity = identity_graph(DType::INT8, vec![6])
            .build(XDNA_TOSA_INTEGER_TARGET)
            .expect("build int8 identity");
        assert_eq!(
            admit(&non_word_identity, XDNA_TOSA_INTEGER_TARGET),
            Err(AdmitError::Unsupported)
        );

        let non_divisible_identity = identity_graph(DType::INT8, vec![1025])
            .build(XDNA_TOSA_INTEGER_TARGET)
            .expect("build int8 identity");
        assert_eq!(
            admit(&non_divisible_identity, XDNA_TOSA_INTEGER_TARGET),
            Err(AdmitError::Unsupported)
        );

        let oversized_matmul =
            matmul_graph_with_zero_points(64, 64, 64, DType::INT8, DType::INT32, -2, 3)
                .build(XDNA_TOSA_INTEGER_TARGET)
                .expect("build int8 matmul");
        assert_eq!(
            admit(&oversized_matmul, XDNA_TOSA_INTEGER_TARGET),
            Err(AdmitError::Unsupported)
        );
    }

    #[test]
    fn admits_both_explicit_fp8_to_bf16_casts() {
        for (dtype, format) in [
            (DType::FP8E4M3, Fp8Format::E4M3),
            (DType::FP8E5M2, Fp8Format::E5M2),
        ] {
            let bytes = fp8_cast_graph(dtype, DType::BF16, vec![1, 1, 4096])
                .build(XDNA_TOSA_FP8_TARGET)
                .expect("build fp8 cast");
            assert_eq!(
                admit(&bytes, XDNA_TOSA_FP8_TARGET),
                Ok(CompilerSpec::Fp8ToBf16 {
                    format,
                    elements: 4096,
                })
            );
        }
    }

    #[test]
    fn fp8_storage_tier_rejects_hidden_or_unsupported_conversion() {
        let valid = fp8_cast_graph(DType::FP8E4M3, DType::BF16, vec![1024])
            .build(XDNA_TOSA_FP8_TARGET)
            .expect("build fp8 cast");
        assert_eq!(admit(&valid, XDNA_TOSA_TARGET), Err(AdmitError::Analysis));

        for graph in [
            fp8_cast_graph(DType::FP8E4M3, DType::FP32, vec![1024]),
            fp8_cast_graph(DType::FP8E5M2, DType::BF16, vec![8]),
        ] {
            let bytes = graph
                .build(XDNA_TOSA_FP8_TARGET)
                .expect("build semantically valid cast");
            assert_eq!(
                admit(&bytes, XDNA_TOSA_FP8_TARGET),
                Err(AdmitError::Unsupported)
            );
        }
    }

    #[test]
    fn admits_bf16_matmul_at_tile_multiples() {
        // Single tile, and a larger non-square multiple of the tested tile.
        for (m, k, n) in [(32, 64, 32), (64, 128, 96)] {
            let bytes = matmul_graph(m, k, n, DType::BF16, DType::FP32)
                .build(XDNA_TOSA_TARGET)
                .expect("build bf16 matmul");
            let spec = admit(&bytes, XDNA_TOSA_TARGET).expect("admit");
            assert_eq!(
                spec,
                CompilerSpec::Matmul {
                    m: m as usize,
                    k: k as usize,
                    n: n as usize,
                }
            );
        }
    }

    #[test]
    fn admits_bf16_nhwc_max_pool2d_corpus_shape() {
        let bytes = max_pool_graph(MaxPoolCase {
            input: [4, 4, 2],
            kernel: [2, 2],
            stride: [2, 2],
            pad: [0; 4],
            dtype: DType::BF16,
            nan_mode: NanPropagationMode::PROPAGATE,
        })
        .build(XDNA_TOSA_TARGET)
        .expect("build bf16 max pool2d");
        assert_eq!(
            admit(&bytes, XDNA_TOSA_TARGET),
            Ok(CompilerSpec::MaxPool2d {
                input_h: 4,
                input_w: 4,
                channels: 2,
                output_h: 2,
                output_w: 2,
                kernel_h: 2,
                kernel_w: 2,
                stride_h: 2,
                stride_w: 2,
            })
        );
    }

    #[test]
    fn rejects_max_pool2d_outside_the_proven_envelope() {
        let cases = [
            max_pool_graph(MaxPoolCase {
                input: [4, 4, 2],
                kernel: [2, 2],
                stride: [2, 2],
                pad: [0; 4],
                dtype: DType::FP32,
                nan_mode: NanPropagationMode::PROPAGATE,
            }),
            max_pool_graph(MaxPoolCase {
                input: [4, 4, 2],
                kernel: [2, 2],
                stride: [2, 2],
                pad: [0; 4],
                dtype: DType::BF16,
                nan_mode: NanPropagationMode::IGNORE,
            }),
            max_pool_graph(MaxPoolCase {
                input: [4, 4, 2],
                kernel: [2, 2],
                stride: [2, 2],
                pad: [1; 4],
                dtype: DType::BF16,
                nan_mode: NanPropagationMode::PROPAGATE,
            }),
            max_pool_graph(MaxPoolCase {
                input: [16, 16, 2],
                kernel: [9, 2],
                stride: [1, 1],
                pad: [0; 4],
                dtype: DType::BF16,
                nan_mode: NanPropagationMode::PROPAGATE,
            }),
            max_pool_graph(MaxPoolCase {
                input: [64, 64, 2],
                kernel: [2, 2],
                stride: [2, 2],
                pad: [0; 4],
                dtype: DType::BF16,
                nan_mode: NanPropagationMode::PROPAGATE,
            }),
        ];
        for graph in cases {
            let bytes = graph
                .build(XDNA_TOSA_TARGET)
                .expect("build semantically valid max pool2d");
            assert_eq!(
                admit(&bytes, XDNA_TOSA_TARGET),
                Err(AdmitError::Unsupported)
            );
        }
    }

    #[test]
    fn admits_zero_operator_passthrough() {
        // The block output *is* the block input; a DMA copy is exact for it.
        let mut graph = OwnedGraph::new("main");
        graph.push_tensor(OwnedTensor::new("x", vec![1, 4, 1024], DType::BF16));
        graph.push_input("x");
        graph.push_output("x");
        let bytes = graph.build(XDNA_TOSA_TARGET).expect("build passthrough");
        assert_eq!(
            admit(&bytes, XDNA_TOSA_TARGET),
            Ok(CompilerSpec::Identity { elements: 4 * 1024 })
        );
    }

    #[test]
    fn rejects_constant_output_identity() {
        // TOSA semantics: the output equals the constant. The compiled IDENTITY kernel would copy
        // the runtime input instead — silently wrong results — so the graph must not admit.
        let shape = vec![1i32, 4, 1024];
        let mut graph = OwnedGraph::new("main");
        graph
            .push_tensor(OwnedTensor::new("x", shape.clone(), DType::BF16))
            .push_tensor(OwnedTensor::constant(
                "c",
                shape.clone(),
                DType::BF16,
                vec![0u8; 4 * 1024 * 2],
            ))
            .push_tensor(OwnedTensor::new("y", shape.clone(), DType::BF16))
            .push_tensor(OwnedTensor::new("dead", shape, DType::BF16))
            .push_operator(OwnedOperator::new(
                OperatorKind::Const,
                vec![],
                vec!["c".into()],
            ))
            .push_operator(OwnedOperator::new(
                OperatorKind::Identity,
                vec!["c".into()],
                vec!["y".into()],
            ))
            .push_operator(OwnedOperator::new(
                OperatorKind::Identity,
                vec!["x".into()],
                vec!["dead".into()],
            ))
            .push_input("x")
            .push_output("y");
        let bytes = graph
            .build(XDNA_TOSA_TARGET)
            .expect("build constant identity");
        assert_eq!(
            admit(&bytes, XDNA_TOSA_TARGET),
            Err(AdmitError::Unsupported)
        );
    }

    #[test]
    fn rejects_constant_weights_matmul() {
        // A CONST-produced lhs (baked weights) is a different program from the two-runtime-input
        // kernel the helper compiles; admitting it would matmul against whatever lands in slot 0.
        let (m, k, n) = (32i32, 64i32, 32i32);
        let mut graph = OwnedGraph::new("main");
        graph
            .push_tensor(OwnedTensor::constant(
                "lhs",
                vec![1, m, k],
                DType::BF16,
                vec![0u8; (m * k * 2) as usize],
            ))
            .push_tensor(OwnedTensor::new("rhs", vec![1, k, n], DType::BF16))
            .push_tensor(OwnedTensor::constant(
                "lhs_zp",
                vec![1],
                DType::BF16,
                vec![0u8; 2],
            ))
            .push_tensor(OwnedTensor::constant(
                "rhs_zp",
                vec![1],
                DType::BF16,
                vec![0u8; 2],
            ))
            .push_tensor(OwnedTensor::new("output", vec![1, m, n], DType::FP32))
            .push_operator(OwnedOperator::new(
                OperatorKind::Const,
                vec![],
                vec!["lhs".into()],
            ))
            .push_operator(OwnedOperator::new(
                OperatorKind::Const,
                vec![],
                vec!["lhs_zp".into()],
            ))
            .push_operator(OwnedOperator::new(
                OperatorKind::Const,
                vec![],
                vec!["rhs_zp".into()],
            ))
            .push_operator(OwnedOperator::new(
                OperatorKind::MatMul,
                vec!["lhs".into(), "rhs".into(), "lhs_zp".into(), "rhs_zp".into()],
                vec!["output".into()],
            ))
            .push_input("rhs")
            .push_output("output");
        let bytes = graph
            .build(XDNA_TOSA_TARGET)
            .expect("build constant-weights matmul");
        assert_eq!(
            admit(&bytes, XDNA_TOSA_TARGET),
            Err(AdmitError::Unsupported)
        );
    }

    #[test]
    fn admit_error_maps_to_the_reference_backend_error_codes() {
        use virtio_accel_core::BackendError;
        assert_eq!(
            BackendError::from(AdmitError::Parse),
            BackendError::InvalidArgument
        );
        assert_eq!(
            BackendError::from(AdmitError::Analysis),
            BackendError::InvalidArgument
        );
        assert_eq!(
            BackendError::from(AdmitError::Unsupported),
            BackendError::Unsupported
        );
    }

    /// One value feeding both MATMUL operands is rejected. The compiled design always declares two
    /// independent input slots, so admitting `X * X` would let a caller bind different buffers to
    /// slots 0 and 1 and compute `A * B` — a result no reading of the graph produces.
    #[test]
    fn rejects_matmul_with_one_value_feeding_both_operands() {
        fn aliased_matmul(dim: i32, in_dtype: DType, out_dtype: DType) -> Vec<u8> {
            let zp = match in_dtype {
                DType::INT8 => vec![0u8],
                _ => vec![0u8; 2],
            };
            let mut graph = OwnedGraph::new("main");
            graph
                .push_tensor(OwnedTensor::new("x", vec![1, dim, dim], in_dtype))
                .push_tensor(OwnedTensor::constant(
                    "lhs_zp",
                    vec![1],
                    in_dtype,
                    zp.clone(),
                ))
                .push_tensor(OwnedTensor::constant("rhs_zp", vec![1], in_dtype, zp))
                .push_tensor(OwnedTensor::new("output", vec![1, dim, dim], out_dtype))
                .push_operator(OwnedOperator::new(
                    OperatorKind::Const,
                    vec![],
                    vec!["lhs_zp".into()],
                ))
                .push_operator(OwnedOperator::new(
                    OperatorKind::Const,
                    vec![],
                    vec!["rhs_zp".into()],
                ))
                .push_operator(OwnedOperator::new(
                    OperatorKind::MatMul,
                    vec!["x".into(), "x".into(), "lhs_zp".into(), "rhs_zp".into()],
                    vec!["output".into()],
                ))
                .push_input("x")
                .push_input("x")
                .push_output("output");
            let target = if in_dtype == DType::INT8 {
                XDNA_TOSA_INTEGER_TARGET
            } else {
                XDNA_TOSA_TARGET
            };
            graph.build(target).expect("build aliased matmul")
        }

        let bf16 = aliased_matmul(64, DType::BF16, DType::FP32);
        assert_eq!(admit(&bf16, XDNA_TOSA_TARGET), Err(AdmitError::Unsupported));
        let int8 = aliased_matmul(32, DType::INT8, DType::INT32);
        assert_eq!(
            admit(&int8, XDNA_TOSA_INTEGER_TARGET),
            Err(AdmitError::Unsupported)
        );
    }

    #[test]
    fn rejects_fp32_matmul_inputs() {
        // FP32-input MATMUL is admissible TOSA but has no compute path on this hardware.
        let bytes = matmul_graph(32, 64, 32, DType::FP32, DType::FP32)
            .build(XDNA_TOSA_TARGET)
            .expect("build fp32 matmul");
        assert_eq!(
            admit(&bytes, XDNA_TOSA_TARGET),
            Err(AdmitError::Unsupported)
        );
    }

    #[test]
    fn rejects_matmul_shape_off_the_tested_tiling() {
        // M not a multiple of the tile, and a dimension past the tested envelope.
        for (m, k, n) in [(48, 64, 32), (32, 64, MATMUL_MAX_DIM as i32 + 32)] {
            let bytes = matmul_graph(m, k, n, DType::BF16, DType::FP32)
                .build(XDNA_TOSA_TARGET)
                .expect("build matmul");
            assert_eq!(
                admit(&bytes, XDNA_TOSA_TARGET),
                Err(AdmitError::Unsupported)
            );
        }
    }

    #[test]
    fn rejects_fp32_identity() {
        // FP32 is admissible TOSA under the floating-point profile but outside the BF16 tier.
        let bytes = identity_graph(DType::FP32, vec![1, 4, 1024])
            .build(XDNA_TOSA_TARGET)
            .expect("build fp32 identity");
        assert_eq!(
            admit(&bytes, XDNA_TOSA_TARGET),
            Err(AdmitError::Unsupported)
        );
    }

    #[test]
    fn rejects_non_multiple_of_line_size() {
        let bytes = identity_graph(DType::BF16, vec![1, 1, 100])
            .build(XDNA_TOSA_TARGET)
            .expect("build small identity");
        assert_eq!(
            admit(&bytes, XDNA_TOSA_TARGET),
            Err(AdmitError::Unsupported)
        );
    }

    #[test]
    fn rejects_bf16_artifact_under_integer_target() {
        let bytes = identity_graph(DType::BF16, vec![1, 4, 1024])
            .build(XDNA_TOSA_TARGET)
            .expect("build");
        assert_eq!(
            admit(&bytes, XDNA_TOSA_INTEGER_TARGET),
            Err(AdmitError::Analysis)
        );
    }
}
