//! Portable TOSA admission for the XDNA backend.
//!
//! This module compiles on every host (no HRX, no `unsafe`). It declares the backend's advertised
//! `Target` constants (issue #82) and [`admit`]s a TOSA artifact into a [`CompilerSpec`] — the
//! validated, integers-and-enums-only description the compiler helper (issue #84) turns into an
//! amdxdna artifact. Anything outside the advertised subset is rejected here, before any subprocess
//! runs. Graph lowering for compute tiers grows on top of this; today the compilable op is the
//! BF16 IDENTITY (a DMA copy).

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

/// A validated operator specialization ready for the compiler helper.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CompilerSpec {
    pub op: SpecOp,
    pub dtype: SpecDType,
    /// Total element count (a positive multiple of [`IDENTITY_LINE_SIZE`]).
    pub elements: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SpecOp {
    Identity,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SpecDType {
    Bf16,
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

/// Admit a TOSA artifact for `target`, returning the specialization the helper compiles.
///
/// Only the BF16 IDENTITY subset is compilable today: one region and block, a single BF16 input
/// and output, IDENTITY operators only, every tensor BF16, and an element count that is a positive
/// multiple of [`IDENTITY_LINE_SIZE`]. Everything else is rejected without running the compiler.
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
    let inputs = analysis.block_inputs(block);
    let outputs = analysis.block_outputs(block);
    if inputs.len() != 1 || outputs.len() != 1 {
        return Err(AdmitError::Unsupported);
    }

    // Only IDENTITY operators, and every tensor value BF16.
    for operator in analysis.execution_order(block) {
        if analysis.operator(*operator).op() != Op::IDENTITY {
            return Err(AdmitError::Unsupported);
        }
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
        elements = elements
            .checked_mul(dimension as usize)
            .ok_or(AdmitError::Unsupported)?;
    }
    if elements == 0 || elements % IDENTITY_LINE_SIZE != 0 {
        return Err(AdmitError::Unsupported);
    }

    Ok(CompilerSpec {
        op: SpecOp::Identity,
        dtype: SpecDType::Bf16,
        elements,
    })
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
        assert_eq!(spec.op, SpecOp::Identity);
        assert_eq!(spec.dtype, SpecDType::Bf16);
        assert_eq!(spec.elements, 4 * 1024);
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
