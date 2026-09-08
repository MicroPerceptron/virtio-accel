//! Host-independent tests: the advertised target constants and the capability boundary. These
//! run on every host, native or placeholder.

use virtio_accel_tosa::{DType, ExtensionSet, Op, ProfileSet, Target, ValueRoles};
use virtio_accel_vulkan::{
    REQUIRED_RESIDENT_BYTES, VULKAN_TOSA_CAPABILITY, VULKAN_TOSA_INTEGER_TARGET,
    VULKAN_TOSA_TARGET, supports_tosa_dtype, supports_tosa_operator,
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
    for accepted in [DType::FP32, DType::BOOL, DType::INT32] {
        assert!(supports_tosa_dtype(accepted), "{accepted:?}");
    }
    for rejected in [
        DType::FP16,
        DType::BF16,
        DType::INT8,
        DType::INT4,
        DType::INT16,
    ] {
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

#[test]
fn every_kernel_variant_assembles_to_valid_spirv_headers() {
    use virtio_accel_vulkan::shader::{KernelKey, NanMode, ReduceOp, Storage};
    let keys = [
        KernelKey::Matmul {
            tile: 16,
            buffers: 17,
        },
        KernelKey::Reduce {
            op: ReduceOp::ArgMax(NanMode::Propagate),
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
