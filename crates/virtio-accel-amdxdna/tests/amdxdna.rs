//! Scaffold-level tests: the placeholder surface and the advertised targets.
//!
//! Native lifecycle tests arrive with the HRX FFI and hardware tickets. These run on every host.

use virtio_accel_amdxdna::{
    AMDXDNA_TOSA_INTEGER_TARGET, AMDXDNA_TOSA_TARGET, AmdXdnaAccelerator, InitError,
    REQUIRED_RESIDENT_BYTES,
};
use virtio_accel_tosa::{ExtensionSet, ProfileSet, Target};

#[test]
fn placeholder_reports_runtime_unavailable() {
    assert!(matches!(
        AmdXdnaAccelerator::new(),
        Err(InitError::RuntimeUnavailable)
    ));
}

#[test]
fn required_resident_bytes_is_maximal() {
    assert_eq!(REQUIRED_RESIDENT_BYTES, u64::MAX);
}

#[test]
fn advertised_targets_are_coherent_and_distinct() {
    assert_eq!(AMDXDNA_TOSA_TARGET.validate(), Ok(AMDXDNA_TOSA_TARGET));
    assert_eq!(
        AMDXDNA_TOSA_INTEGER_TARGET.validate(),
        Ok(AMDXDNA_TOSA_INTEGER_TARGET)
    );
    assert_ne!(AMDXDNA_TOSA_TARGET, AMDXDNA_TOSA_INTEGER_TARGET);
}

#[test]
fn bf16_target_declares_the_floating_point_profile_and_bf16_extension() {
    assert!(
        AMDXDNA_TOSA_TARGET
            .profiles
            .contains(ProfileSet::FLOATING_POINT)
    );
    assert!(AMDXDNA_TOSA_TARGET.extensions.contains(ExtensionSet::BF16));
}

#[test]
fn integer_target_declares_the_integer_profile_and_no_extensions() {
    assert!(
        AMDXDNA_TOSA_INTEGER_TARGET
            .profiles
            .contains(ProfileSet::INTEGER)
    );
    assert_eq!(AMDXDNA_TOSA_INTEGER_TARGET.extensions, ExtensionSet::NONE);
}

#[test]
fn targets_survive_an_identity_round_trip() {
    for target in [AMDXDNA_TOSA_TARGET, AMDXDNA_TOSA_INTEGER_TARGET] {
        assert_eq!(Target::from_identity(target.to_identity()), Ok(target));
    }
}
