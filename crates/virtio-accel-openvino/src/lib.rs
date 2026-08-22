//! Intel OpenVINO host backend for NPU, GPU, and CPU execution.
//!
//! The production path accepts device-neutral TOSA 1.0 FlatBuffers, validates and analyzes them
//! with `virtio-accel-tosa`, and lowers supported static floating-point and exact INT8 graphs to
//! in-memory OpenVINO IR inside this host-native crate. Models are compiled with the `ACCURACY` execution
//! mode hint so plugins may not silently execute a declared-FP32 graph at reduced precision.
//! Program buffers are page-aligned allocations wrapped directly by OpenVINO tensors created
//! from host pointers; output execution is accepted only when the runtime uses the same
//! allocation as its output tensor.
//!
//! The native modules compile only when the build script detects an OpenVINO C runtime
//! (`libopenvino_c`) via pkg-config or the `VIRTIO_ACCEL_OPENVINO_LIB_DIR` override; the
//! `VIRTIO_ACCEL_OPENVINO` variable forces the probe on (`1`, failing loudly when the runtime is
//! missing) or off (`0`). Builds without the runtime keep the portable lowering module and an
//! unsupported-runtime placeholder.

#![cfg_attr(not(va_openvino), forbid(unsafe_code))]

mod lower;

pub use lower::{
    LoweringError, OPENVINO_TOSA_CAPABILITY, OPENVINO_TOSA_INTEGER_CAPABILITY,
    OPENVINO_TOSA_INTEGER_TARGET, OPENVINO_TOSA_TARGET, supports_tosa_dtype,
    supports_tosa_operator,
};

use virtio_accel_tosa::CapabilityDescriptor;
#[cfg(not(va_openvino))]
use virtio_accel_tosa::TosaCapabilityProvider;

#[cfg(va_openvino)]
const TOSA_CAPABILITIES: &[CapabilityDescriptor] =
    &[OPENVINO_TOSA_CAPABILITY, OPENVINO_TOSA_INTEGER_CAPABILITY];
#[cfg(not(va_openvino))]
const TOSA_CAPABILITIES: &[CapabilityDescriptor] = &[];

/// The OpenVINO runtime does not publish a finite upper bound for compiled-model residency.
///
/// Requiring the maximal charge makes the provider promise truthful: a process cannot retain
/// `u64::MAX` bytes for one model. Device integrations must set their aggregate program-residency
/// policy accordingly when admitting an OpenVINO program.
pub const REQUIRED_RESIDENT_BYTES: u64 = u64::MAX;

/// Failure to initialize an OpenVINO backend instance.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InitError {
    /// The crate was built without an OpenVINO C runtime (`libopenvino_c`).
    RuntimeUnavailable,
    /// The OpenVINO core object could not be created.
    CoreCreationFailed,
    /// Device enumeration failed inside the OpenVINO runtime.
    DeviceEnumerationFailed,
    /// No requested or known inference device is available on this host.
    DeviceUnavailable,
}

impl std::fmt::Display for InitError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for InitError {}

#[cfg(va_openvino)]
mod ffi;
#[cfg(va_openvino)]
mod native;
#[cfg(va_openvino)]
pub use native::{
    OpenVinoAccelerator, OpenVinoBuffer, OpenVinoContext, OpenVinoEvent, OpenVinoProgram,
    OpenVinoQueue,
};

/// Placeholder that keeps workspace consumers portable when no OpenVINO runtime was detected.
#[cfg(not(va_openvino))]
#[derive(Clone, Copy, Debug, Default)]
pub struct OpenVinoAccelerator;

#[cfg(not(va_openvino))]
impl OpenVinoAccelerator {
    /// Report that no OpenVINO runtime was detected when this crate was built.
    pub fn new() -> Result<Self, InitError> {
        Err(InitError::RuntimeUnavailable)
    }

    /// Report that no OpenVINO runtime was detected when this crate was built.
    pub fn with_device(_device: &str) -> Result<Self, InitError> {
        Err(InitError::RuntimeUnavailable)
    }

    /// Report that no OpenVINO runtime was detected when this crate was built.
    pub fn available_devices() -> Result<Vec<String>, InitError> {
        Err(InitError::RuntimeUnavailable)
    }
}

#[cfg(not(va_openvino))]
impl TosaCapabilityProvider for OpenVinoAccelerator {
    fn tosa_capabilities(&self) -> &'static [CapabilityDescriptor] {
        TOSA_CAPABILITIES
    }
}
