//! Planned advertised TOSA tiers for the Vulkan backend.
//!
//! These are *candidates* declared by the scaffold (ADR 0004 in `docs/adr/`): the FP32 base tier
//! plus the provisional INT8 tier. An FP16 tier is deliberately undeclared until per-device
//! `VK_KHR_shader_float_controls` evidence closes wayfinder ticket 5, and FP8 is rejected at
//! admission. The final capability descriptors and operator subset table arrive with ticket 5.

use virtio_accel_tosa::{ExtensionSet, Level, ProfileSet, Target, Version};

/// The FP32 base tier: TOSA 1.0, floating-point profile, level 8K, no extensions.
pub const VULKAN_TOSA_TARGET: Target = Target::new(
    Version::TOSA_1_0,
    ProfileSet::FLOATING_POINT,
    Level::Level8K,
    ExtensionSet::NONE,
);

/// The provisional integer tier: TOSA 1.0, integer profile, level 8K, no extensions.
pub const VULKAN_TOSA_INTEGER_TARGET: Target = Target::new(
    Version::TOSA_1_0,
    ProfileSet::INTEGER,
    Level::Level8K,
    ExtensionSet::NONE,
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn targets_validate() {
        assert_eq!(VULKAN_TOSA_TARGET.validate(), Ok(VULKAN_TOSA_TARGET));
        assert_eq!(
            VULKAN_TOSA_INTEGER_TARGET.validate(),
            Ok(VULKAN_TOSA_INTEGER_TARGET)
        );
    }

    #[test]
    fn targets_survive_an_identity_round_trip() {
        for target in [VULKAN_TOSA_TARGET, VULKAN_TOSA_INTEGER_TARGET] {
            assert_eq!(Target::from_identity(target.to_identity()), Ok(target));
        }
    }
}
