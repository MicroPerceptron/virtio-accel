//! Portable TOSA admission surface for the XDNA backend.
//!
//! This module compiles on every host (no HRX, no `unsafe`) and declares the backend's public
//! numerical promise: the two `Target` constants decided in the numerical-tier ticket (issue #82).
//! Graph-level admission (target-equality gating, the per-operator dtype sweep, and emission of the
//! compiler-helper input) lands with the TOSA-admission ticket on top of these constants; keeping
//! the targets here now lets the placeholder and the future native path name one authority.

use virtio_accel_tosa::{ExtensionSet, Level, ProfileSet, Target, Version};

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

#[cfg(test)]
mod tests {
    use super::*;
    use virtio_accel_tosa::Target;

    #[test]
    fn both_targets_are_coherent() {
        // `validate` rejects an extension without its base profile; BF16 requires the
        // floating-point profile, so this proves the tier targets are self-consistent.
        assert_eq!(XDNA_TOSA_TARGET.validate(), Ok(XDNA_TOSA_TARGET));
        assert_eq!(
            XDNA_TOSA_INTEGER_TARGET.validate(),
            Ok(XDNA_TOSA_INTEGER_TARGET)
        );
    }

    #[test]
    fn targets_round_trip_through_their_identity() {
        for target in [XDNA_TOSA_TARGET, XDNA_TOSA_INTEGER_TARGET] {
            assert_eq!(Target::from_identity(target.to_identity()), Ok(target));
        }
    }

    #[test]
    fn the_two_tiers_are_distinct() {
        assert_ne!(XDNA_TOSA_TARGET, XDNA_TOSA_INTEGER_TARGET);
        assert_ne!(
            XDNA_TOSA_TARGET.to_identity(),
            XDNA_TOSA_INTEGER_TARGET.to_identity()
        );
    }
}
