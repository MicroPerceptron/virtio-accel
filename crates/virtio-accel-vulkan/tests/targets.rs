//! Host-independent tests: the advertised target constants and the capability boundary. These
//! run on every host, native or placeholder.

use virtio_accel_tosa::{DType, ExtensionSet, Op, ProfileSet, Target, ValueRoles};
use virtio_accel_vulkan::{
    REQUIRED_RESIDENT_BYTES, VULKAN_TOSA_CAPABILITY, VULKAN_TOSA_FP16_CAPABILITY,
    VULKAN_TOSA_INTEGER_TARGET, VULKAN_TOSA_TARGET, supports_tosa_dtype, supports_tosa_operator,
};

#[test]
fn required_resident_bytes_is_maximal() {
    assert_eq!(REQUIRED_RESIDENT_BYTES, u64::MAX);
}

#[test]
fn advertised_targets_are_coherent_and_distinct() {
    assert_eq!(VULKAN_TOSA_TARGET.validate(), Ok(VULKAN_TOSA_TARGET));
    assert_eq!(
        VULKAN_TOSA_INTEGER_TARGET.validate(),
        Ok(VULKAN_TOSA_INTEGER_TARGET)
    );
    assert_ne!(VULKAN_TOSA_TARGET, VULKAN_TOSA_INTEGER_TARGET);
}

#[test]
fn fp32_target_declares_the_floating_point_profile_and_no_extensions() {
    assert!(
        VULKAN_TOSA_TARGET
            .profiles
            .contains(ProfileSet::FLOATING_POINT)
    );
    assert_eq!(VULKAN_TOSA_TARGET.extensions, ExtensionSet::NONE);
}

#[test]
fn integer_target_declares_the_integer_profile_and_no_extensions() {
    assert!(
        VULKAN_TOSA_INTEGER_TARGET
            .profiles
            .contains(ProfileSet::INTEGER)
    );
    assert_eq!(VULKAN_TOSA_INTEGER_TARGET.extensions, ExtensionSet::NONE);
}

#[test]
fn targets_survive_an_identity_round_trip() {
    for target in [VULKAN_TOSA_TARGET, VULKAN_TOSA_INTEGER_TARGET] {
        assert_eq!(Target::from_identity(target.to_identity()), Ok(target));
    }
}

#[test]
fn capability_advertises_the_shared_fp32_operator_set() {
    assert_eq!(VULKAN_TOSA_CAPABILITY.target, VULKAN_TOSA_TARGET);
    assert!(VULKAN_TOSA_CAPABILITY.supports_dtype(DType::FP32, ValueRoles::ALL));
    assert!(VULKAN_TOSA_CAPABILITY.supports_dtype(DType::BOOL, ValueRoles::ALL));
    assert!(VULKAN_TOSA_CAPABILITY.supports_dtype(DType::INT32, ValueRoles::ALL));
    assert_eq!(VULKAN_TOSA_CAPABILITY.operators.len(), 42);
    for op in [
        Op::IDENTITY,
        Op::MATMUL,
        Op::MAX_POOL2D,
        Op::ARGMAX,
        Op::CLAMP,
        Op::ERF,
        Op::SIGMOID,
        Op::TANH,
        Op::ADD,
        Op::SUB,
        Op::MUL,
        Op::POW,
        Op::MAXIMUM,
        Op::MINIMUM,
        Op::LOGICAL_AND,
        Op::LOGICAL_OR,
        Op::LOGICAL_XOR,
        Op::LOGICAL_NOT,
        Op::ABS,
        Op::CEIL,
        Op::FLOOR,
        Op::COS,
        Op::SIN,
        Op::EXP,
        Op::LOG,
        Op::NEGATE,
        Op::RECIPROCAL,
        Op::RSQRT,
        Op::SELECT,
        Op::EQUAL,
        Op::GREATER,
        Op::GREATER_EQUAL,
        Op::REDUCE_MAX,
        Op::REDUCE_MIN,
        Op::REDUCE_PRODUCT,
        Op::REDUCE_SUM,
        Op::CONCAT,
        Op::RESHAPE,
        Op::REVERSE,
        Op::TRANSPOSE,
        Op::CONST,
        Op::CONST_SHAPE,
    ] {
        assert!(supports_tosa_operator(op), "{op:?}");
    }
    for op in [
        Op::CONV2D,
        Op::AVG_POOL2D,
        Op::CAST,
        Op::RESCALE,
        Op::PAD,
        Op::GATHER,
    ] {
        assert!(!supports_tosa_operator(op), "{op:?}");
    }
    for accepted in [DType::FP32, DType::FP16, DType::BOOL, DType::INT32] {
        assert!(supports_tosa_dtype(accepted), "{accepted:?}");
    }
    for rejected in [DType::BF16, DType::INT8, DType::INT4, DType::INT16] {
        assert!(!supports_tosa_dtype(rejected), "{rejected:?}");
    }
    // The `MUL` shift is an INT8 constant parameter consumed at admission, so the descriptor
    // admits INT8 in the constant role only — exactly as the Core ML and OpenVINO tiers do.
    assert!(VULKAN_TOSA_CAPABILITY.supports_dtype(DType::INT8, ValueRoles::CONSTANT));
    assert!(!VULKAN_TOSA_CAPABILITY.supports_dtype(DType::INT8, ValueRoles::INPUT));
    assert!(!VULKAN_TOSA_CAPABILITY.supports_dtype(DType::INT8, ValueRoles::OUTPUT));
    assert!(!VULKAN_TOSA_CAPABILITY.supports_dtype(DType::INT8, ValueRoles::INTERMEDIATE));
    assert_eq!(VULKAN_TOSA_CAPABILITY.graph.max_blocks, 1);
}

/// The FP16 tier (ADR 0008): the same target identity, operators, and graph envelope as the
/// FP32 tier, with binary16 in every role; the advertised descriptor is valid on every device
/// because the kernels use crate-owned conversions rather than float16 device features.
#[test]
fn fp16_capability_extends_the_fp32_boundary() {
    assert_eq!(VULKAN_TOSA_FP16_CAPABILITY.target, VULKAN_TOSA_TARGET);
    assert_eq!(
        VULKAN_TOSA_FP16_CAPABILITY.operators,
        VULKAN_TOSA_CAPABILITY.operators
    );
    assert_eq!(
        VULKAN_TOSA_FP16_CAPABILITY.graph,
        VULKAN_TOSA_CAPABILITY.graph
    );
    for accepted in [DType::FP32, DType::FP16, DType::BOOL, DType::INT32] {
        assert!(
            VULKAN_TOSA_FP16_CAPABILITY.supports_dtype(accepted, ValueRoles::ALL),
            "{accepted:?}"
        );
    }
    for rejected in [DType::BF16, DType::INT4, DType::INT16] {
        assert!(
            !VULKAN_TOSA_FP16_CAPABILITY.supports_dtype(rejected, ValueRoles::ALL),
            "{rejected:?}"
        );
    }
    assert!(VULKAN_TOSA_FP16_CAPABILITY.supports_dtype(DType::INT8, ValueRoles::CONSTANT));
    assert!(!VULKAN_TOSA_FP16_CAPABILITY.supports_dtype(DType::INT8, ValueRoles::INPUT));
}

#[test]
fn every_kernel_variant_assembles_to_valid_spirv_headers() {
    use virtio_accel_vulkan::shader::{Fp8Format, KernelKey, NanMode, ReduceOp, Storage};
    let keys = [
        KernelKey::Matmul {
            input: Storage::Word,
            output: Storage::Word,
            tile: 16,
            buffers: 17,
        },
        KernelKey::Matmul {
            input: Storage::Half,
            output: Storage::Half,
            tile: 16,
            buffers: 17,
        },
        // The FP8 tier's `(FP8, FP8) -> FP16` MATMUL, one variant per encoding.
        KernelKey::Matmul {
            input: Storage::Quarter(Fp8Format::E4M3),
            output: Storage::Half,
            tile: 16,
            buffers: 17,
        },
        KernelKey::Matmul {
            input: Storage::Quarter(Fp8Format::E5M2),
            output: Storage::Half,
            tile: 16,
            buffers: 17,
        },
        KernelKey::Cast {
            input: Storage::Quarter(Fp8Format::E4M3),
            output: Storage::Half,
            workgroup: 64,
            buffers: 17,
        },
        KernelKey::Cast {
            input: Storage::Word,
            output: Storage::Quarter(Fp8Format::E5M2),
            workgroup: 64,
            buffers: 17,
        },
        KernelKey::Move {
            storage: Storage::Quarter(Fp8Format::E4M3),
            contiguous: true,
            workgroup: 64,
            buffers: 17,
        },
        KernelKey::Reduce {
            op: ReduceOp::ArgMax(NanMode::Propagate),
            float: Storage::Half,
            workgroup: 256,
            buffers: 17,
        },
        KernelKey::Move {
            storage: Storage::Byte,
            contiguous: false,
            workgroup: 128,
            buffers: 5,
        },
    ];
    for key in keys {
        let words = key.assemble();
        assert_eq!(words[0], 0x0723_0203, "SPIR-V magic for {key:?}");
        assert_eq!(words[1], 0x0001_0300, "SPIR-V 1.3 for {key:?}");
        assert_eq!(key.assemble(), words, "deterministic assembly for {key:?}");
    }
}

/// Every kernel variant passes `spirv-val --target-env vulkan1.3`.
///
/// ADR 0007 and ADR 0008 both cite this validation as evidence, but nothing ran it: a module
/// that no driver accepts is otherwise diagnosed as an opaque `VK_ERROR_UNKNOWN` from pipeline
/// creation, which names neither the instruction nor the reason. The absence rule mirrors
/// `VIRTIO_ACCEL_VULKAN_REQUIRE_DEVICE`: without the tool the sweep skips, and
/// `VIRTIO_ACCEL_VULKAN_REQUIRE_SPIRV_VAL=1` turns that absence into a failure, so a CI lane
/// cannot lose the check by losing the package.
#[test]
fn every_kernel_variant_passes_spirv_val() {
    use std::io::Write;
    use std::process::Command;
    use virtio_accel_vulkan::shader::KernelKey;

    let required =
        std::env::var_os("VIRTIO_ACCEL_VULKAN_REQUIRE_SPIRV_VAL").is_some_and(|value| value == "1");
    let probe = Command::new("spirv-val")
        .arg("--version")
        .output()
        .ok()
        .filter(|output| output.status.success());
    let Some(probe) = probe else {
        assert!(
            !required,
            "VIRTIO_ACCEL_VULKAN_REQUIRE_SPIRV_VAL=1 but spirv-val is not on PATH"
        );
        eprintln!("skipping: spirv-val is not installed (package `spirv-tools`)");
        return;
    };
    eprintln!(
        "validating with {}",
        String::from_utf8_lossy(&probe.stdout).trim()
    );

    let directory = std::env::temp_dir().join(format!("va-spirv-{}", std::process::id()));
    std::fs::create_dir_all(&directory).expect("scratch directory");
    let variants = KernelKey::every_variant();
    assert!(!variants.is_empty(), "no kernel variants to validate");
    let mut failures = Vec::new();
    for (index, key) in variants.iter().enumerate() {
        let words = key.assemble();
        let path = directory.join(format!("{index:04}.spv"));
        let mut file = std::fs::File::create(&path).expect("module file");
        for word in &words {
            file.write_all(&word.to_le_bytes()).expect("module bytes");
        }
        drop(file);
        let output = Command::new("spirv-val")
            .arg("--target-env")
            .arg("vulkan1.3")
            .arg(&path)
            .output()
            .expect("spirv-val runs");
        if !output.status.success() {
            failures.push(format!(
                "{key:?}: {}",
                String::from_utf8_lossy(&output.stderr).trim()
            ));
        }
    }
    let _ = std::fs::remove_dir_all(&directory);
    assert!(
        failures.is_empty(),
        "{} of {} kernel variants are invalid SPIR-V:\n{}",
        failures.len(),
        variants.len(),
        failures.join("\n")
    );
    eprintln!("{} kernel variants validated", variants.len());
}
