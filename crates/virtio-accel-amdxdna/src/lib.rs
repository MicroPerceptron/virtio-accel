//! AMD XDNA (Ryzen AI NPU) host backend for `virtio-accel`.
//!
//! This crate will execute device-neutral TOSA 1.0 programs on an AMD XDNA2 NPU through the HRX
//! runtime (`libhrx`), compiling admitted graphs with the pinned aiecc toolchain as a bounded
//! subprocess. The design is recorded in `docs/research/amdxdna-crate-design.md`,
//! `docs/research/amdxdna-compiler-helper-contract.md`, and
//! `docs/research/amdxdna-execution-model.md`; the advertised numerical tier is
//! `docs/adr/0001-amdxdna-first-numerical-tier.md`.
//!
//! **This is the scaffold.** It ships the always-compiled admission surface (`lower`) and a
//! compile-only placeholder; the HRX FFI, native `Accelerator` implementation, and compiler helper
//! land in later tickets. The build script sets the `va_amdxdna` cfg only when it finds a complete
//! amdxdna-native HRX prefix (`VIRTIO_ACCEL_HRX_DIR`/`HRX_DIR`, or the `VIRTIO_ACCEL_HRX_LIB_DIR`
//! escape hatch); `VIRTIO_ACCEL_AMDXDNA` forces the probe on (`1`, failing loudly) or off (`0`).
//! Hosts without HRX build and unit-test the portable surface plus the placeholder.
//!
//! The scaffold contains no `unsafe` at all, so the crate root forbids it outright. When the HRX
//! FFI and native `Accelerator` land, that ticket relaxes this to
//! `cfg_attr(not(va_amdxdna), forbid(unsafe_code))` and registers the audited exception in
//! `ci/check-release-policy.py`, exactly as the OpenVINO and Hexagon backends do.

#![forbid(unsafe_code)]

mod lower;

pub use lower::{AMDXDNA_TOSA_INTEGER_TARGET, AMDXDNA_TOSA_TARGET};

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
}

impl core::fmt::Display for InitError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for InitError {}

// Scaffold placeholder. It is compiled unconditionally today because no native module exists yet;
// when the native `Accelerator` implementation lands it becomes `cfg(not(va_amdxdna))` and the real
// type is exported under `cfg(va_amdxdna)`, exactly as the OpenVINO and Core ML crates are shaped.

/// Placeholder that keeps workspace consumers portable until the native backend lands.
#[derive(Clone, Copy, Debug, Default)]
pub struct AmdXdnaAccelerator;

impl AmdXdnaAccelerator {
    /// Report that no HRX runtime and native backend are available in this build.
    pub fn new() -> Result<Self, InitError> {
        Err(InitError::RuntimeUnavailable)
    }
}
