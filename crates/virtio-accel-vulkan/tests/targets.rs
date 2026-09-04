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
fn capability_advertises_exactly_the_executed_fp32_boundary() {
    assert_eq!(VULKAN_TOSA_CAPABILITY.target, VULKAN_TOSA_TARGET);
    assert!(VULKAN_TOSA_CAPABILITY.supports_dtype(DType::FP32, ValueRoles::ALL));
    assert!(supports_tosa_operator(Op::IDENTITY));
    assert!(supports_tosa_operator(Op::MATMUL));
    assert!(!supports_tosa_operator(Op::MAX_POOL2D));
    assert!(supports_tosa_dtype(DType::FP32));
    for rejected in [
        DType::FP16,
        DType::BF16,
        DType::INT8,
        DType::INT32,
        DType::BOOL,
    ] {
        assert!(!supports_tosa_dtype(rejected), "{rejected:?}");
    }
    assert_eq!(VULKAN_TOSA_CAPABILITY.graph.max_blocks, 1);
}

#[test]
fn checked_in_shader_module_is_stable() {
    let words = virtio_accel_vulkan::shader::copy_u32_spirv();
    assert_eq!(words[0], 0x0723_0203, "SPIR-V magic");
    assert_eq!(words[1], 0x0001_0300, "SPIR-V 1.3");
    assert!(std::ptr::eq(
        words,
        virtio_accel_vulkan::shader::copy_u32_spirv()
    ));
    let matmul = virtio_accel_vulkan::shader::matmul_fp32_spirv();
    assert_eq!(matmul[0], 0x0723_0203, "SPIR-V magic");
    assert_eq!(matmul[1], 0x0001_0300, "SPIR-V 1.3");
    assert!(std::ptr::eq(
        matmul,
        virtio_accel_vulkan::shader::matmul_fp32_spirv()
    ));
}
