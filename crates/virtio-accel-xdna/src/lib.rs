//! AMD XDNA (Ryzen AI NPU) host backend for `virtio-accel`.
//!
//! This crate executes device-neutral TOSA 1.0 programs on an AMD XDNA2 NPU through the HRX
//! runtime (`libhrx`), compiling admitted graphs with the pinned aiecc toolchain as a bounded
//! subprocess. The design is recorded across the AMD XDNA wayfinder map (issue #78) and its
//! decision tickets (#82 numerical tier, #83 crate layout, #84 compiler helper, #85 execution
//! model), whose resolution records live on their respective ticket branches.
//!
//! The native modules (`ffi`, `native`) compile only when the build script finds a complete
//! amdxdna-native HRX prefix (`VIRTIO_ACCEL_HRX_DIR`/`HRX_DIR`, or the `VIRTIO_ACCEL_HRX_LIB_DIR`
//! escape hatch) and sets the `va_xdna` cfg; `VIRTIO_ACCEL_XDNA` forces the probe on (`1`, failing
//! loudly) or off (`0`). Hosts without HRX build and unit-test the portable admission surface
//! (`lower`) plus a compile-only placeholder, and compile no `unsafe` at all.
//!
//! **Scope today:** the HRX device/stream owner and `hrx_buffer` primitives (allocate with a
//! persistent host mapping, range flush/invalidate, release). Program loading and dispatch land
//! with the execution path; they currently report `Unsupported`.

#![cfg_attr(not(va_xdna), forbid(unsafe_code))]

mod lower;

pub use lower::{XDNA_TOSA_INTEGER_TARGET, XDNA_TOSA_TARGET};

/// The HRX runtime publishes no finite upper bound for a loaded program's device residency.
///
/// Requiring the maximal charge keeps the provider promise truthful: a process cannot retain
/// `u64::MAX` bytes for one program. Device integrations set their aggregate program-residency
/// policy accordingly when admitting an XDNA program. Mirrors the OpenVINO backend, whose runtime
/// has the same property.
pub const REQUIRED_RESIDENT_BYTES: u64 = u64::MAX;

/// Failure to initialize an XDNA backend instance.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InitError {
    /// The crate was built without a detected HRX runtime (`libhrx`).
    RuntimeUnavailable,
    /// HRX initialized but enumerated no NPU device on this host.
    DeviceUnavailable,
    /// The HRX device or stream could not be initialized.
    Initialization,
}

impl core::fmt::Display for InitError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for InitError {}

#[cfg(va_xdna)]
mod ffi;
#[cfg(va_xdna)]
mod native;
#[cfg(va_xdna)]
pub use native::{
    XDNA_ERROR_DOMAIN, XdnaAccelerator, XdnaBuffer, XdnaContext, XdnaEvent, XdnaProgram, XdnaQueue,
};

/// Placeholder that keeps workspace consumers portable when no HRX runtime was detected.
#[cfg(not(va_xdna))]
#[derive(Clone, Copy, Debug, Default)]
pub struct XdnaAccelerator;

#[cfg(not(va_xdna))]
impl XdnaAccelerator {
    /// Report that no HRX runtime was detected when this crate was built.
    pub fn new() -> Result<Self, InitError> {
        Err(InitError::RuntimeUnavailable)
    }
}
