//! Qualcomm Hexagon NPU backend for `virtio-accel`.
//!
//! The portable portion validates TOSA 1.0 artifacts and creates a provider-local, fixed binding
//! and operation plan. Native modules are compiled only for Windows ARM64 when a complete public
//! QAIRT/QNN SDK is detected. SDK-free hosts retain that portable code and a constructor that
//! reports the unavailable runtime without importing Qualcomm dependencies into portable crates.

#![forbid(unsafe_code)]

mod lower;

pub use lower::{HEXAGON_TOSA_TARGET, LoweringError, supports_tosa_dtype, supports_tosa_operator};

/// Native QNN runtime and ABI version validated by the initial backend tier.
pub const TESTED_QAIRT_RELEASE: &str = "2.48.40.260702";

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

impl core::fmt::Display for InitError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for InitError {}

#[cfg(va_hexagon)]
mod ffi;
#[cfg(va_hexagon)]
mod native;
#[cfg(va_hexagon)]
pub use native::HexagonAccelerator;

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

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(not(va_hexagon))]
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
