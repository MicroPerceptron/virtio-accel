//! Core ML host backend for Apple Neural Engine capable Macs.
//!
//! The production path accepts device-neutral TOSA 1.0 FlatBuffers, validates and analyzes them
//! with `virtio-accel-tosa`, and lowers supported static floating-point graphs inside this
//! host-native crate. Core ML models are configured with `CPUAndNeuralEngine`: supported
//! operations may execute on the ANE, while Core ML remains free to place unsupported operations
//! on the CPU. Program buffers are page-aligned allocations wrapped directly by `MLMultiArray`;
//! output execution is accepted only when Core ML uses the same allocation as its output backing.

#![cfg_attr(not(target_os = "macos"), forbid(unsafe_code))]

mod artifact;
mod lower;

pub use artifact::{ArtifactBuildError, CoreMlArtifact, FeatureRole};
pub use lower::{COREML_TOSA_TARGET, LoweringError, supports_tosa_dtype, supports_tosa_operator};

use virtio_accel_core::{ArtifactFormat, TargetIdentity};

/// Provider artifact format for [`CoreMlArtifact`].
pub const ARTIFACT_FORMAT: ArtifactFormat = match ArtifactFormat::new(0x434d_4c50) {
    Some(format) => format,
    None => panic!("Core ML artifact format must be nonzero"),
};

/// Core ML path-artifact ABI v1 targeting CPU plus Apple Neural Engine execution.
pub const TARGET_IDENTITY: TargetIdentity = TargetIdentity([
    0x434f_5245,
    0x4d4c_0001,
    0x414e_4503,
    0x4d41_434f,
    0x000e_0000,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
]);

/// The Core ML runtime does not publish a finite upper bound for model residency.
///
/// Requiring the maximal charge makes the provider promise truthful: a process cannot retain
/// `u64::MAX` bytes for one model. Device integrations must set their aggregate program-residency
/// policy accordingly when admitting a Core ML program.
pub const REQUIRED_RESIDENT_BYTES: u64 = u64::MAX;

/// Failure to initialize a Core ML backend instance.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InitError {
    /// The backend is only executable on macOS 14 or newer.
    UnsupportedPlatform,
    /// The configured model root is missing, not a directory, or not representable as UTF-8.
    InvalidModelRoot,
    /// Core ML does not report an accessible Apple Neural Engine.
    NeuralEngineUnavailable,
}

impl std::fmt::Display for InitError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for InitError {}

#[cfg(target_os = "macos")]
mod macos;
#[cfg(target_os = "macos")]
pub use macos::{
    CoreMlAccelerator, CoreMlBuffer, CoreMlContext, CoreMlEvent, CoreMlProgram, CoreMlQueue,
};

/// Non-macOS placeholder that keeps workspace consumers portable.
#[cfg(not(target_os = "macos"))]
#[derive(Clone, Copy, Debug, Default)]
pub struct CoreMlAccelerator;

#[cfg(not(target_os = "macos"))]
impl CoreMlAccelerator {
    /// Report that Core ML is unavailable on this target.
    pub fn new(_model_root: impl AsRef<std::path::Path>) -> Result<Self, InitError> {
        Err(InitError::UnsupportedPlatform)
    }

    /// Report that the native TOSA-to-Core ML execution path is unavailable on this target.
    pub fn new_tosa() -> Result<Self, InitError> {
        Err(InitError::UnsupportedPlatform)
    }
}
