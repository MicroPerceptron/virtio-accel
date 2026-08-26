//! Qualcomm Hexagon NPU backend for `virtio-accel`.
//!
//! The portable portion validates TOSA 1.0 artifacts and creates a provider-local, fixed binding
//! and operation plan. Native modules are compiled only for Windows ARM64 when a complete public
//! QAIRT/QNN SDK is detected. SDK-free hosts retain that portable code and a constructor that
//! reports the unavailable runtime without importing Qualcomm dependencies into portable crates.

#![cfg_attr(not(any(va_hexagon, va_hexagon_direct)), forbid(unsafe_code))]

mod lower;

#[allow(unsafe_code)]
#[cfg(va_hexagon_direct)]
mod direct;
#[cfg(va_hexagon_direct)]
pub use direct::{
    DIRECT_HTP_ARTIFACT_FORMAT, DIRECT_HTP_V73_TARGET, DirectHexagonAccelerator,
    DirectHexagonBuffer, DirectHexagonContext, DirectHexagonEvent, DirectHexagonProgram,
    DirectHexagonQueue, DirectHtpArtifact, DirectHtpOperation, DirectHtpRuntimeInfo,
    KerrFrameParameters, KerrTraceParameters, WormholeTraceParameters,
};

pub use lower::{
    HEXAGON_TOSA_CAPABILITY, HEXAGON_TOSA_INTEGER_CAPABILITY, HEXAGON_TOSA_INTEGER_TARGET,
    HEXAGON_TOSA_TARGET, LoweringError, supports_tosa_dtype, supports_tosa_operator,
};

use virtio_accel_tosa::CapabilityDescriptor;
#[cfg(not(va_hexagon))]
use virtio_accel_tosa::TosaCapabilityProvider;

#[cfg(va_hexagon)]
const TOSA_CAPABILITIES: &[CapabilityDescriptor] =
    &[HEXAGON_TOSA_CAPABILITY, HEXAGON_TOSA_INTEGER_CAPABILITY];
#[cfg(not(va_hexagon))]
const TOSA_CAPABILITIES: &[CapabilityDescriptor] = &[];

/// Native QNN runtime and ABI version validated by the initial backend tier.
pub const TESTED_QAIRT_RELEASE: &str = "2.49.0.260730";

/// Conservative program-residency charge until QNN publishes a finite graph allocation bound.
pub const REQUIRED_RESIDENT_BYTES: u64 = u64::MAX;

/// Failure to initialize a Qualcomm Hexagon backend instance.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InitError {
    /// The crate was built without a complete QAIRT/QNN development runtime.
    RuntimeUnavailable,
    /// The loaded QNN interface is outside the validated API range.
    IncompatibleRuntime,
    /// The HTP backend or its device could not be created.
    DeviceUnavailable,
}

/// Failure to initialize the signed direct FastRPC/QFloat32 provider.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DirectInitError {
    RuntimeUnavailable,
    IncompatibleHardware,
    ModuleUnavailable,
}

impl core::fmt::Display for DirectInitError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for DirectInitError {}

impl core::fmt::Display for InitError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for InitError {}

#[allow(unsafe_code)]
#[cfg(va_hexagon)]
mod ffi;
#[allow(unsafe_code)]
#[cfg(va_hexagon)]
mod native;
#[cfg(va_hexagon)]
pub use native::HexagonAccelerator;
#[cfg(va_hexagon)]
pub use native::{
    HexagonBuffer, HexagonContext, HexagonEvent, HexagonProgram, HexagonQueue, QnnRuntimeInfo,
};

/// Placeholder that keeps workspace consumers portable when no complete QNN SDK was detected.
#[cfg(not(va_hexagon))]
#[derive(Clone, Copy, Debug, Default)]
pub struct HexagonAccelerator;

#[cfg(not(va_hexagon))]
impl HexagonAccelerator {
    /// Report that the native QNN build requirements were not detected.
    pub fn new() -> Result<Self, InitError> {
        Err(InitError::RuntimeUnavailable)
    }

    /// Report that the native QNN build requirements were not detected.
    pub fn available_devices() -> Result<Vec<String>, InitError> {
        Err(InitError::RuntimeUnavailable)
    }
}

#[cfg(not(va_hexagon))]
impl TosaCapabilityProvider for HexagonAccelerator {
    fn tosa_capabilities(&self) -> &'static [CapabilityDescriptor] {
        TOSA_CAPABILITIES
    }
}

/// SDK-free placeholder for the provider-local direct HTP path.
#[cfg(not(va_hexagon_direct))]
#[derive(Clone, Copy, Debug, Default)]
pub struct DirectHexagonAccelerator;

#[cfg(not(va_hexagon_direct))]
impl DirectHexagonAccelerator {
    pub fn new() -> Result<Self, DirectInitError> {
        Err(DirectInitError::RuntimeUnavailable)
    }
}

#[cfg(all(test, not(va_hexagon)))]
mod tests {
    use super::*;

    #[test]
    fn sdk_free_constructor_is_explicitly_unavailable() {
        assert!(matches!(
            HexagonAccelerator::new(),
            Err(InitError::RuntimeUnavailable)
        ));
        assert!(matches!(
            HexagonAccelerator::available_devices(),
            Err(InitError::RuntimeUnavailable)
        ));
    }
}
