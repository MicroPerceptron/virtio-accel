//! Scaffold-level tests: the advertised targets.
//!
//! Native lifecycle tests arrive with the HRX FFI and hardware tickets. These run on every host.

#[cfg(not(va_xdna))]
use virtio_accel_tosa::TosaCapabilityProvider;
use virtio_accel_tosa::{ExtensionSet, Op, OperatorConstraints, ProfileSet, Target, ValueRoles};
#[cfg(not(va_xdna))]
use virtio_accel_xdna::XdnaAccelerator;
use virtio_accel_xdna::{
    REQUIRED_RESIDENT_BYTES, XDNA_TOSA_CAPABILITY, XDNA_TOSA_FP8_CAPABILITY, XDNA_TOSA_FP8_TARGET,
    XDNA_TOSA_INTEGER_TARGET, XDNA_TOSA_TARGET,
};

#[test]
fn required_resident_bytes_is_maximal() {
    assert_eq!(REQUIRED_RESIDENT_BYTES, u64::MAX);
}

#[test]
fn advertised_targets_are_coherent_and_distinct() {
    assert_eq!(XDNA_TOSA_TARGET.validate(), Ok(XDNA_TOSA_TARGET));
    assert_eq!(
        XDNA_TOSA_INTEGER_TARGET.validate(),
        Ok(XDNA_TOSA_INTEGER_TARGET)
    );
    assert_eq!(XDNA_TOSA_FP8_TARGET.validate(), Ok(XDNA_TOSA_FP8_TARGET));
    assert_ne!(XDNA_TOSA_TARGET, XDNA_TOSA_INTEGER_TARGET);
    assert_ne!(XDNA_TOSA_TARGET, XDNA_TOSA_FP8_TARGET);
}

#[test]
fn bf16_target_declares_the_floating_point_profile_and_bf16_extension() {
    assert!(
        XDNA_TOSA_TARGET
            .profiles
            .contains(ProfileSet::FLOATING_POINT)
    );
    assert!(XDNA_TOSA_TARGET.extensions.contains(ExtensionSet::BF16));
}

#[test]
fn integer_target_declares_the_integer_profile_and_no_extensions() {
    assert!(
        XDNA_TOSA_INTEGER_TARGET
            .profiles
            .contains(ProfileSet::INTEGER)
    );
    assert_eq!(XDNA_TOSA_INTEGER_TARGET.extensions, ExtensionSet::NONE);
}

#[test]
fn fp8_storage_target_requires_bf16_and_both_fp8_extensions() {
    for extension in [
        ExtensionSet::BF16,
        ExtensionSet::FP8E4M3,
        ExtensionSet::FP8E5M2,
    ] {
        assert!(XDNA_TOSA_FP8_TARGET.extensions.contains(extension));
    }
}

#[test]
fn targets_survive_an_identity_round_trip() {
    for target in [
        XDNA_TOSA_TARGET,
        XDNA_TOSA_FP8_TARGET,
        XDNA_TOSA_INTEGER_TARGET,
    ] {
        assert_eq!(Target::from_identity(target.to_identity()), Ok(target));
    }
}

#[test]
fn fp8_capability_is_storage_conversion_only() {
    assert_eq!(XDNA_TOSA_FP8_CAPABILITY.target, XDNA_TOSA_FP8_TARGET);
    assert!(XDNA_TOSA_FP8_CAPABILITY.supports_operator(Op::CAST));
    assert!(!XDNA_TOSA_FP8_CAPABILITY.supports_operator(Op::MATMUL));
    for dtype in [
        virtio_accel_tosa::DType::FP8E4M3,
        virtio_accel_tosa::DType::FP8E5M2,
    ] {
        assert!(XDNA_TOSA_FP8_CAPABILITY.supports_dtype(dtype, ValueRoles::INPUT));
        assert!(!XDNA_TOSA_FP8_CAPABILITY.supports_dtype(dtype, ValueRoles::OUTPUT));
    }
    assert!(
        XDNA_TOSA_FP8_CAPABILITY.supports_dtype(virtio_accel_tosa::DType::BF16, ValueRoles::OUTPUT)
    );
}

#[test]
fn bf16_capability_matches_the_implemented_surface() {
    assert_eq!(XDNA_TOSA_CAPABILITY.target, XDNA_TOSA_TARGET);
    assert!(XDNA_TOSA_CAPABILITY.supports_operator(Op::IDENTITY));
    assert!(XDNA_TOSA_CAPABILITY.supports_operator(Op::MATMUL));
    let pool = XDNA_TOSA_CAPABILITY
        .operator(Op::MAX_POOL2D)
        .expect("MAX_POOL2D capability");
    assert!(
        pool.constraints
            .contains(OperatorConstraints::PROPAGATING_NAN)
    );
    assert!(pool.constraints.contains(OperatorConstraints::ZERO_PADDING));
}

#[cfg(not(va_xdna))]
#[test]
fn placeholder_advertises_no_runtime_capabilities() {
    assert!(XdnaAccelerator.tosa_capabilities().is_empty());
}

/// The offline / catalog-population path works without HRX: `compile_artifact` is available in
/// every unix build (including the no-HRX placeholder), needing only the pinned toolchain at run
/// time. This is the workflow where a build host populates a precompiled catalog for device-less
/// (and compiler-less) serving hosts.
#[cfg(unix)]
#[test]
fn compile_artifact_works_without_hrx() {
    use virtio_accel_tosa_build::{OperatorKind, OwnedGraph, OwnedOperator, OwnedTensor};
    use virtio_accel_xdna::compile_artifact;

    if std::env::var_os("VIRTIO_ACCEL_AMDXDNA_TOOLCHAIN").is_none() {
        eprintln!("no XDNA toolchain configured; skipping offline compile test");
        return;
    }
    const ELEMENTS: usize = 1024;
    let mut graph = OwnedGraph::new("main");
    graph
        .push_tensor(OwnedTensor::new(
            "x",
            vec![1, 1, ELEMENTS as i32],
            virtio_accel_tosa::DType::BF16,
        ))
        .push_tensor(OwnedTensor::new(
            "y",
            vec![1, 1, ELEMENTS as i32],
            virtio_accel_tosa::DType::BF16,
        ))
        .push_operator(OwnedOperator::new(
            OperatorKind::Identity,
            vec!["x".into()],
            vec!["y".into()],
        ))
        .push_input("x")
        .push_output("y");
    let tosa = graph.build(XDNA_TOSA_TARGET).expect("build bf16 identity");

    let container = compile_artifact(&tosa, XDNA_TOSA_TARGET).expect("offline compile");
    let parsed =
        virtio_accel_xdna::PrecompiledArtifact::parse(&container).expect("valid container");
    assert_eq!((parsed.inputs, parsed.outputs), (1, 1));
    assert_eq!(parsed.slot_bytes, [(ELEMENTS * 2) as u64; 2]);
    assert!(parsed.xclbin.starts_with(b"xclbin2"));
}
