//! Vendor-neutral Vulkan compute host backend for `virtio-accel`.
//!
//! **This is the scaffold.** It ships the always-compiled admission constants (`lower`) and a
//! placeholder; the `ash`-based FFI, the native `Accelerator` implementation, and the advertised
//! numerical tiers land in subsequent tickets of the
//! [Vulkan wayfinder map](https://github.com/MicroPerceptron/virtio-accel/issues/154). The design
//! decisions ratified by this scaffold live in [`docs/adr/`](../../../../docs/adr/) (ADRs 0001–0004).

#![forbid(unsafe_code)]

mod lower;

pub use lower::{VULKAN_TOSA_INTEGER_TARGET, VULKAN_TOSA_TARGET};

/// TOSA capability list: empty until the native path's advertised tiers are ratified (ticket 5)
/// and proven by conformance (ticket 10). Compiled for every build, native or placeholder.
const TOSA_CAPABILITIES: &[virtio_accel_tosa::CapabilityDescriptor] = &[];

/// Providers with no finite residency bound promise the maximal charge; anything less would
/// under-report `ArtifactRef::resident_bytes`.
pub const REQUIRED_RESIDENT_BYTES: u64 = u64::MAX;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InitError {
    /// The native backend or the Vulkan loader could not be initialized.
    RuntimeUnavailable,
}

impl core::fmt::Display for InitError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::RuntimeUnavailable => write!(formatter, "no Vulkan loader initialized"),
        }
    }
}

impl std::error::Error for InitError {}

use virtio_accel_tosa::TosaCapabilityProvider;

/// Placeholder that keeps workspace consumers portable until the native backend lands.
#[derive(Clone, Copy, Debug, Default)]
pub struct VulkanAccelerator;

impl VulkanAccelerator {
    /// Report that the native backend is not available in this build.
    pub fn new() -> Result<Self, InitError> {
        Err(InitError::RuntimeUnavailable)
    }

    /// Enumerate nothing in the scaffold; the native path reports one entry per
    /// `ash`-enumerated physical device.
    pub fn available_devices() -> Result<Vec<String>, InitError> {
        Err(InitError::RuntimeUnavailable)
    }
}

impl TosaCapabilityProvider for VulkanAccelerator {
    fn tosa_capabilities(&self) -> &'static [virtio_accel_tosa::CapabilityDescriptor] {
        TOSA_CAPABILITIES
    }
}
