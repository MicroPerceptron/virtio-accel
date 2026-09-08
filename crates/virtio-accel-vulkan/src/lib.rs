//! Vendor-neutral Vulkan compute host backend for `virtio-accel`.
//!
//! The native path binds Vulkan 1.3 through the pinned [`ash`] crate, loading the platform's
//! Vulkan loader at run time (ADR 0002). It executes device-neutral TOSA 1.0 programs admitted by
//! [`lower`](crate::VULKAN_TOSA_CAPABILITY) — the FP32 operator tier shared with the Core ML and
//! OpenVINO backends, with `BOOL`/`INT32` auxiliaries — on crate-authored SPIR-V compute kernels
//! specialized at `load_program` (ADR 0003, ADR 0007). A whole graph is one submission: constants
//! and intermediates live in a per-program arena and dependent dispatches are separated by compute
//! barriers. Buffers are dedicated `VkDeviceMemory` allocations bound directly as storage buffers;
//! completion is a nonblocking `vkGetFenceStatus` read over a bounded per-context ring of command
//! buffers, fences, and descriptor sets (ADR 0006); no worker thread exists.
//!
//! The native module compiles on the host operating systems enumerated by `build.rs` (`va_vulkan`).
//! Loader absence is a run-time fact reported as [`InitError::RuntimeUnavailable`], never a build
//! probe. `VIRTIO_ACCEL_VULKAN=0` forces the placeholder, `=1` makes an unsupported target a loud
//! build failure. The design decisions live in [`docs/adr/`](../../../docs/adr/) (ADRs 0001–0007).

#![cfg_attr(not(va_vulkan), forbid(unsafe_code))]

mod lower;
pub mod shader;

pub use lower::{
    LoweringError, VULKAN_TOSA_CAPABILITY, VULKAN_TOSA_INTEGER_TARGET, VULKAN_TOSA_TARGET,
    supports_tosa_dtype, supports_tosa_operator,
};

use virtio_accel_tosa::CapabilityDescriptor;
#[cfg(not(va_vulkan))]
use virtio_accel_tosa::TosaCapabilityProvider;

/// TOSA capability list: the FP32 operator tier on native builds, nothing on the placeholder.
#[cfg(va_vulkan)]
const TOSA_CAPABILITIES: &[CapabilityDescriptor] = &[VULKAN_TOSA_CAPABILITY];
#[cfg(not(va_vulkan))]
const TOSA_CAPABILITIES: &[CapabilityDescriptor] = &[];

/// Vulkan publishes no bound on what a driver retains for a compiled pipeline, so the provider
/// promises the maximal charge; anything less would under-report `ArtifactRef::resident_bytes`.
pub const REQUIRED_RESIDENT_BYTES: u64 = u64::MAX;

/// Failure to initialize a Vulkan backend instance.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InitError {
    /// No Vulkan loader could be loaded, or it predates Vulkan 1.3 (or this is a placeholder
    /// build).
    RuntimeUnavailable,
    /// `vkCreateInstance` failed.
    InstanceCreationFailed,
    /// Physical-device enumeration failed inside the loader or an ICD.
    DeviceEnumerationFailed,
    /// No requested or suitable Vulkan 1.3 compute device is available on this host.
    DeviceUnavailable,
    /// `vkCreateDevice` or the device-level setup failed.
    DeviceCreationFailed,
}

impl core::fmt::Display for InitError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::RuntimeUnavailable => write!(formatter, "no Vulkan 1.3 loader initialized"),
            Self::InstanceCreationFailed => write!(formatter, "Vulkan instance creation failed"),
            Self::DeviceEnumerationFailed => {
                write!(formatter, "Vulkan physical-device enumeration failed")
            }
            Self::DeviceUnavailable => write!(formatter, "no suitable Vulkan compute device"),
            Self::DeviceCreationFailed => write!(formatter, "Vulkan device creation failed"),
        }
    }
}

impl std::error::Error for InitError {}

#[cfg(va_vulkan)]
mod native;
#[cfg(va_vulkan)]
pub use native::{
    LiveResources, VulkanAccelerator, VulkanBuffer, VulkanContext, VulkanEvent, VulkanProgram,
    VulkanQueue,
};

/// Placeholder that keeps workspace consumers portable where no Vulkan loader host exists.
#[cfg(not(va_vulkan))]
#[derive(Clone, Copy, Debug, Default)]
pub struct VulkanAccelerator;

#[cfg(not(va_vulkan))]
impl VulkanAccelerator {
    /// Report that the native backend is not available in this build.
    pub fn new() -> Result<Self, InitError> {
        Err(InitError::RuntimeUnavailable)
    }

    /// Report that the native backend is not available in this build.
    pub fn with_device(_device: &str) -> Result<Self, InitError> {
        Err(InitError::RuntimeUnavailable)
    }

    /// Enumerate nothing in the placeholder; the native path reports one entry per physical
    /// device.
    pub fn available_devices() -> Result<Vec<String>, InitError> {
        Err(InitError::RuntimeUnavailable)
    }
}

#[cfg(not(va_vulkan))]
impl TosaCapabilityProvider for VulkanAccelerator {
    fn tosa_capabilities(&self) -> &'static [CapabilityDescriptor] {
        TOSA_CAPABILITIES
    }
}
