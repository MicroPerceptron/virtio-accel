//! Host-independent tests: the advertised targets, the capability descriptors, and the offline
//! compile path.
//!
//! Native lifecycle and on-metal numerics live in `hardware.rs`, which is gated on a detected HRX
//! runtime. Everything here runs on every host.

#[cfg(not(va_xdna))]
use virtio_accel_tosa::TosaCapabilityProvider;
use virtio_accel_tosa::{ExtensionSet, Op, OperatorConstraints, ProfileSet, Target, ValueRoles};
#[cfg(not(va_xdna))]
use virtio_accel_xdna::XdnaAccelerator;
use virtio_accel_xdna::{
    REQUIRED_RESIDENT_BYTES, XDNA_TOSA_CAPABILITY, XDNA_TOSA_FP8_CAPABILITY, XDNA_TOSA_FP8_TARGET,
    XDNA_TOSA_INTEGER_CAPABILITY, XDNA_TOSA_INTEGER_TARGET, XDNA_TOSA_TARGET,
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
fn fp8_target_never_produces_fp8() {
    // FP8 is a *storage* encoding here: it is only ever consumed. Both admitted tiers read FP8 and
    // write a wider type (BF16 for the standalone CAST, FP32 for the fused MATMUL), so no graph on
    // this target may produce an FP8 value. MATMUL joined this surface with the fused tier; see
    // `fused_fp8_matmul_capability_covers_its_graph`.
    assert_eq!(XDNA_TOSA_FP8_CAPABILITY.target, XDNA_TOSA_FP8_TARGET);
    assert!(XDNA_TOSA_FP8_CAPABILITY.supports_operator(Op::CAST));
    for dtype in [
        virtio_accel_tosa::DType::FP8E4M3,
        virtio_accel_tosa::DType::FP8E5M2,
    ] {
        assert!(XDNA_TOSA_FP8_CAPABILITY.supports_dtype(dtype, ValueRoles::INPUT));
        assert!(!XDNA_TOSA_FP8_CAPABILITY.supports_dtype(dtype, ValueRoles::OUTPUT));
        assert!(!XDNA_TOSA_FP8_CAPABILITY.supports_dtype(dtype, ValueRoles::INTERMEDIATE));
    }
    assert!(
        XDNA_TOSA_FP8_CAPABILITY.supports_dtype(virtio_accel_tosa::DType::BF16, ValueRoles::OUTPUT)
    );
    // Nothing on this target may produce or consume FP32 as anything but the fused result.
    assert!(
        !XDNA_TOSA_FP8_CAPABILITY.supports_dtype(virtio_accel_tosa::DType::FP32, ValueRoles::INPUT)
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

#[test]
fn integer_capability_preserves_openvino_and_adds_exact_rescale() {
    use virtio_accel_tosa::DType;

    assert_eq!(
        XDNA_TOSA_INTEGER_CAPABILITY.target,
        XDNA_TOSA_INTEGER_TARGET
    );
    // CONST/IDENTITY/MATMUL are the OpenVINO baseline. RESCALE is the intentional issue-#147
    // expansion, and target separation is unchanged.
    for op in [Op::CONST, Op::IDENTITY, Op::MATMUL, Op::RESCALE] {
        assert!(XDNA_TOSA_INTEGER_CAPABILITY.supports_operator(op));
    }
    assert!(XDNA_TOSA_INTEGER_CAPABILITY.supports_dtype(DType::INT8, ValueRoles::ALL));
    assert!(XDNA_TOSA_INTEGER_CAPABILITY.supports_dtype(DType::INT32, ValueRoles::OUTPUT));
    // The dtype roles *do* diverge from OpenVINO here, and they have to. `ValueRoles::INPUT` means
    // "graph-visible block input"; OpenVINO's integer tier only ever produces INT32, but the
    // RESCALE tier consumes it as the block input, so omitting the role would advertise a surface
    // that contradicts `admit`. `admitted_integer_block_input_dtypes_are_advertised` pins that.
    assert!(XDNA_TOSA_INTEGER_CAPABILITY.supports_dtype(DType::INT32, ValueRoles::INPUT));
}

/// The advertised dtype roles must cover what admission actually accepts. A dtype omitted from
/// `INPUT` is unroutable through the standard `INPUT || OUTPUT` capability filter every sibling
/// backend builds, so an admitted tier whose block input dtype is unadvertised is invisible to a
/// scheduler — the tier would be implemented, tested, and unreachable.
#[test]
fn admitted_integer_block_input_dtypes_are_advertised() {
    use virtio_accel_conformance::numerics::RESCALE_INT32_TO_INT8;
    use virtio_accel_tosa::DType;

    // The shared corpus RESCALE fixture takes INT32 in and INT8 out, both at the block boundary.
    assert!(
        virtio_accel_xdna::admit(RESCALE_INT32_TO_INT8.artifact, XDNA_TOSA_INTEGER_TARGET).is_ok(),
        "the shared RESCALE fixture must stay admissible"
    );
    for (dtype, role) in [
        (DType::INT32, ValueRoles::INPUT),
        (DType::INT8, ValueRoles::OUTPUT),
    ] {
        assert!(
            XDNA_TOSA_INTEGER_CAPABILITY.supports_dtype(dtype, role),
            "admission accepts {dtype:?} at the block boundary but the capability hides it"
        );
    }
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

/// The fused tier's capability must cover the graph it admits, including the interior: a promoted
/// BF16 value that the descriptor does not admit as `INTERMEDIATE` would advertise a surface the
/// backend contradicts.
#[test]
fn fused_fp8_matmul_capability_covers_its_graph() {
    use virtio_accel_tosa::DType;

    for dtype in [DType::FP8E4M3, DType::FP8E5M2] {
        assert!(XDNA_TOSA_FP8_CAPABILITY.supports_dtype(dtype, ValueRoles::INPUT));
    }
    // The promotion is graph-interior for the fused tier and a block output for the CAST tier.
    assert!(XDNA_TOSA_FP8_CAPABILITY.supports_dtype(DType::BF16, ValueRoles::INTERMEDIATE));
    assert!(XDNA_TOSA_FP8_CAPABILITY.supports_dtype(DType::BF16, ValueRoles::OUTPUT));
    // The fused tier ends at the TOSA-mandated FP32 accumulator.
    assert!(XDNA_TOSA_FP8_CAPABILITY.supports_dtype(DType::FP32, ValueRoles::OUTPUT));
    for op in [Op::CAST, Op::CONST, Op::MATMUL] {
        assert!(XDNA_TOSA_FP8_CAPABILITY.supports_operator(op));
    }
    let matmul = XDNA_TOSA_FP8_CAPABILITY
        .operator(Op::MATMUL)
        .expect("MATMUL capability");
    assert!(
        matmul
            .constraints
            .contains(OperatorConstraints::ZERO_ZERO_POINTS)
    );
}

/// The offline path compiles the fused graph to a container that binds FP8 operands directly and
/// never binds a BF16 tensor — the whole point of the tier.
///
/// Unix-only for the same reason as `compile_artifact_works_without_hrx`: the compiler helper is a
/// subprocess in its own process group, so `compile_artifact` is not offered on other platforms.
#[cfg(unix)]
#[test]
fn fused_fp8_matmul_compiles_to_a_wellformed_artifact() {
    use virtio_accel_tosa::DType;
    use virtio_accel_tosa_build::{OperatorKind, OwnedGraph, OwnedOperator, OwnedTensor};
    use virtio_accel_xdna::{PrecompiledArtifact, compile_artifact};

    if std::env::var_os("VIRTIO_ACCEL_AMDXDNA_TOOLCHAIN").is_none() {
        eprintln!("no XDNA toolchain configured; skipping fused offline compile test");
        return;
    }
    let (m, k, n) = (32i32, 64i32, 32i32);
    let mut graph = OwnedGraph::new("main");
    graph
        .push_tensor(OwnedTensor::new("lhs_fp8", vec![1, m, k], DType::FP8E4M3))
        .push_tensor(OwnedTensor::new("rhs_fp8", vec![1, k, n], DType::FP8E4M3))
        .push_tensor(OwnedTensor::new("lhs_bf16", vec![1, m, k], DType::BF16))
        .push_tensor(OwnedTensor::new("rhs_bf16", vec![1, k, n], DType::BF16))
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
            OperatorKind::Cast,
            vec!["lhs_fp8".into()],
            vec!["lhs_bf16".into()],
        ))
        .push_operator(OwnedOperator::new(
            OperatorKind::Cast,
            vec!["rhs_fp8".into()],
            vec!["rhs_bf16".into()],
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
            vec![
                "lhs_bf16".into(),
                "rhs_bf16".into(),
                "lhs_zp".into(),
                "rhs_zp".into(),
            ],
            vec!["output".into()],
        ))
        .push_input("lhs_fp8")
        .push_input("rhs_fp8")
        .push_output("output");
    let tosa = graph
        .build(XDNA_TOSA_FP8_TARGET)
        .expect("build fused fp8 matmul graph");
    let container =
        compile_artifact(&tosa, XDNA_TOSA_FP8_TARGET).expect("compile fused fp8 matmul");
    let parsed = PrecompiledArtifact::parse(&container).expect("valid container");
    assert_eq!(parsed.inputs, 2);
    assert_eq!(parsed.outputs, 1);
    assert_eq!(
        parsed.slot_bytes,
        vec![(m * k) as u64, (k * n) as u64, (m * n * 4) as u64],
        "FP8 operands bind one byte per element and no BF16 tensor is bound"
    );
}
