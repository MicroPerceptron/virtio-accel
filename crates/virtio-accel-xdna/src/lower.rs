//! Portable TOSA admission for the XDNA backend.
//!
//! This module compiles on every host (no HRX, no `unsafe`). It declares the backend's advertised
//! `Target` constants (issue #82) and [`admit`]s a TOSA artifact into a [`CompilerSpec`] — the
//! validated, integers-and-enums-only description the compiler helper (issue #84) turns into an
//! amdxdna artifact. Anything outside the advertised subset is rejected here, before any subprocess
//! runs. Graph lowering for compute tiers grows on top of this; the compilable subsets today are
//! the BF16 IDENTITY (a DMA copy) and the BF16 → FP32 MATMUL (issue #90).

use virtio_accel_tosa::{
    AnalyzedValueKind, DType, ExtensionSet, Level, Op, ProfileSet, Target, Version, parse,
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

/// The integer tier: TOSA 1.0, integer profile, level 8K, no extensions.
///
/// Native i8/i16 matmul with exact (bit-for-bit) results, kept on a separate target from the
/// floating-point tier exactly as the OpenVINO backend separates its FP and INTEGER targets.
pub const XDNA_TOSA_INTEGER_TARGET: Target = Target::new(
    Version::TOSA_1_0,
    ProfileSet::INTEGER,
    Level::Level8K,
    ExtensionSet::NONE,
);

/// The DMA line size the IDENTITY template transfers in; an admitted element count must be a
/// positive multiple of it. Kept in sync with `compiler/xdna_compile.py`.
pub const IDENTITY_LINE_SIZE: usize = 1024;

/// The one tested MATMUL compute tile (`m`, `k`, `n`), proven on npu2 (AIE2P).
///
/// The bf16→fp32 kernel's micro-tile is (4, 8, 8); this L1-fitting macro-tile is a multiple of it,
/// and every admitted `(M, K, N)` is a positive multiple of this tile — the single tiling the
/// helper compiles and the hardware tests exercise. Untested shapes are rejected (issue #90).
/// The FP32 output is 4 B/element, so this tile is smaller than a same-shape bf16 tile would be, to
/// keep the double-buffered C tile plus the A/B tiles inside the compute core's ~64 KiB L1. Kept in
/// sync with `compiler/xdna_compile.py`.
pub const MATMUL_TILE_M: usize = 32;
pub const MATMUL_TILE_K: usize = 64;
pub const MATMUL_TILE_N: usize = 32;

/// Largest admitted MATMUL dimension. The tested tiling generalizes across multiples of the tile,
/// but only within this envelope; larger shapes are a later generalization and are rejected now.
pub const MATMUL_MAX_DIM: usize = 512;

/// A validated operator specialization ready for the compiler helper. Each variant names its input
/// and output dtypes; the closed shape is integers only, so no guest bytes cross the boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CompilerSpec {
    /// BF16 → BF16 elementwise copy of `elements` values (a positive multiple of
    /// [`IDENTITY_LINE_SIZE`]).
    Identity { elements: usize },
    /// BF16 × BF16 → FP32 matrix multiply `C[M, N] = A[M, K] · B[K, N]` (batch 1). Each of `m`,
    /// `k`, `n` is a positive multiple of the corresponding MATMUL tile dimension and at most
    /// [`MATMUL_MAX_DIM`]. The FP32 output is the TOSA-mandated accumulator (issue #82).
    Matmul { m: usize, k: usize, n: usize },
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
/// Two subsets are compilable today, both on the BF16 target: the BF16 IDENTITY (a DMA copy) and
/// the BF16 → FP32 MATMUL. Everything else is rejected without running the compiler. Each template
/// admits only graphs whose **dataflow** matches what the compiled kernel executes — the IDENTITY
/// template requires every operator to be IDENTITY (no constants: with a single block input, every
/// value then provably carries that input's bytes), and the MATMUL template requires the operator's
/// operands to be exactly the block inputs (constants may exist only as the two zero-points).
/// Without these checks a semantically different graph (say, a constant-output IDENTITY or a
/// constant-weights MATMUL) would compile to a kernel that reads runtime buffers the graph never
/// asked for, returning well-formed but wrong data. Semantic and target validity — including that
/// BF16 MATMUL zero-points are constant zero — is enforced by
/// [`analyze_for`](virtio_accel_tosa::Model::analyze_for) before these structural checks.
pub fn admit(bytes: &[u8], target: Target) -> Result<CompilerSpec, AdmitError> {
    if target != XDNA_TOSA_TARGET {
        // The integer tier and other targets are later tickets.
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

    // Classify in one pass. IDENTITY tolerates no other operator kind (not even CONST); MATMUL
    // tolerates exactly one MATMUL plus CONST operators, which `admit_matmul` then pins down to
    // the two zero-points.
    let mut matmul = None;
    let mut identities = 0usize;
    let mut constants = 0usize;
    for operator in analysis.execution_order(block) {
        match analysis.operator(*operator).op() {
            Op::IDENTITY => identities += 1,
            Op::CONST => constants += 1,
            Op::MATMUL if matmul.is_none() => matmul = Some(*operator),
            _ => return Err(AdmitError::Unsupported),
        }
    }
    match (matmul, identities, constants) {
        // All-IDENTITY (zero operators included: the block output then *is* the block input, and a
        // DMA copy is exact for it).
        (None, _, 0) => admit_identity(&analysis, block),
        (Some(matmul), 0, _) => admit_matmul(&analysis, block, matmul),
        _ => Err(AdmitError::Unsupported),
    }
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
    // and the MATMUL result is the block output.
    if analysis.block_inputs(block) != [inputs[0], inputs[1]]
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

#[cfg(test)]
mod tests {
    use super::*;
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

    /// A batch-1 MATMUL `C[1,M,N] = A[1,M,K] · B[1,K,N]` with the two constant-zero zero-points.
    fn matmul_graph(
        m: i32,
        k: i32,
        n: i32,
        in_dtype: DType,
        out_dtype: DType,
    ) -> OwnedGraph<'static> {
        // A BF16 zero (0x0000) and an FP32 zero (0x0000_0000); the zero-point dtype tracks the
        // input dtype, so two bytes suffice for every dtype used here.
        let zero = |dtype: DType| match dtype {
            DType::FP32 => vec![0u8; 4],
            _ => vec![0u8; 2],
        };
        let mut graph = OwnedGraph::new("main");
        graph
            .push_tensor(OwnedTensor::new("lhs", vec![1, m, k], in_dtype))
            .push_tensor(OwnedTensor::new("rhs", vec![1, k, n], in_dtype))
            .push_tensor(OwnedTensor::constant(
                "lhs_zp",
                vec![1],
                in_dtype,
                zero(in_dtype),
            ))
            .push_tensor(OwnedTensor::constant(
                "rhs_zp",
                vec![1],
                in_dtype,
                zero(in_dtype),
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

    #[test]
    fn both_targets_are_coherent_and_distinct() {
        assert_eq!(XDNA_TOSA_TARGET.validate(), Ok(XDNA_TOSA_TARGET));
        assert_eq!(
            XDNA_TOSA_INTEGER_TARGET.validate(),
            Ok(XDNA_TOSA_INTEGER_TARGET)
        );
        assert_ne!(XDNA_TOSA_TARGET, XDNA_TOSA_INTEGER_TARGET);
        for target in [XDNA_TOSA_TARGET, XDNA_TOSA_INTEGER_TARGET] {
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
    fn rejects_wrong_target() {
        let bytes = identity_graph(DType::BF16, vec![1, 4, 1024])
            .build(XDNA_TOSA_TARGET)
            .expect("build");
        assert_eq!(
            admit(&bytes, XDNA_TOSA_INTEGER_TARGET),
            Err(AdmitError::Unsupported)
        );
    }
}
