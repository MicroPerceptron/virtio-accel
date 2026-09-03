//! Scaffold-level tests: the advertised target constants and the placeholder.
//!
//! Native lifecycle tests arrive with the `ash` FFI and hardware tickets. These run on every host.

use virtio_accel_tosa::{ExtensionSet, ProfileSet, Target};
use virtio_accel_vulkan::{
    REQUIRED_RESIDENT_BYTES, VULKAN_TOSA_INTEGER_TARGET, VULKAN_TOSA_TARGET, VulkanAccelerator,
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
fn placeholder_reports_runtime_unavailable() {
    let error = VulkanAccelerator::new().unwrap_err();
    assert_eq!(error, virtio_accel_vulkan::InitError::RuntimeUnavailable);
}
