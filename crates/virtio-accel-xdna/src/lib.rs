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
//! loudly) or off (`0`). Hosts without HRX build the portable admission surface (`lower`), the
//! artifact codec, the offline compiler driver ([`compile_artifact`], unix), and a placeholder —
//! and compile no `unsafe` at all.
//!
//! **Scope today:** the full `Accelerator` lifecycle — the HRX device/stream owner, `hrx_buffer`
//! primitives (persistent mapping, range flush/invalidate, release), and the serialized dispatch
//! worker bridging `hrx_stream_dispatch`/`synchronize` to a latched nonblocking `poll_event`
//! (execution-model spec, issue #85). `load_program` accepts the crate-local precompiled format
//! ([`artifact`]) directly, and a TOSA artifact by admitting it and compiling it with the bounded
//! aiecc helper subprocess (issue #84). The compilable TOSA subsets today are BF16 IDENTITY,
//! BF16 → FP32 MATMUL, BF16 MAX_POOL2D, explicit FP8 → BF16 storage conversion, and exact INT8
//! IDENTITY plus zero-point-aware INT8 → INT32 MATMUL.
//! Admission (`lower`) unit-tests on every host.

#![cfg_attr(not(va_xdna), forbid(unsafe_code))]

pub mod artifact;
mod lower;

pub use artifact::{PrecompiledArtifact, XDNA_PRECOMPILED_FORMAT};
pub use lower::{
    AdmitError, CompilerSpec, Fp8Format, XDNA_TOSA_CAPABILITY, XDNA_TOSA_FP8_CAPABILITY,
    XDNA_TOSA_FP8_TARGET, XDNA_TOSA_INTEGER_CAPABILITY, XDNA_TOSA_INTEGER_TARGET, XDNA_TOSA_TARGET,
    admit,
};

use virtio_accel_tosa::{CapabilityDescriptor, TosaCapabilityProvider};

#[cfg(va_xdna)]
const TOSA_CAPABILITIES: &[CapabilityDescriptor] = &[
    XDNA_TOSA_CAPABILITY,
    XDNA_TOSA_FP8_CAPABILITY,
    XDNA_TOSA_INTEGER_CAPABILITY,
];
#[cfg(not(va_xdna))]
const TOSA_CAPABILITIES: &[CapabilityDescriptor] = &[];

/// The HRX runtime publishes no finite upper bound for a loaded program's device residency.
///
/// Requiring the maximal charge keeps the provider promise truthful: a process cannot retain
/// `u64::MAX` bytes for one program. Device integrations set their aggregate program-residency
/// policy accordingly when admitting an XDNA program. Mirrors the OpenVINO backend, whose runtime
/// has the same property.
pub const REQUIRED_RESIDENT_BYTES: u64 = u64::MAX;

/// `BackendError::External` domain tag for this backend's failures ("XDNA" in ASCII), covering
/// both HRX runtime errors and compiler-helper failures.
pub const XDNA_ERROR_DOMAIN: u32 = 0x5844_4e41;

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

// The compiler-helper driver is pure safe subprocess code: it needs the pinned toolchain at run
// time, never HRX, so it compiles on every unix host — the offline catalog-population host is
// exactly a build machine *without* libhrx. (Unix-only: it drives the helper in its own process
// group, a unix notion.)
#[cfg(unix)]
mod compiler;
#[cfg(va_xdna)]
mod ffi;
#[cfg(va_xdna)]
mod native;
#[cfg(va_xdna)]
pub use native::{
    XdnaAccelerator, XdnaBuffer, XdnaContext, XdnaEvent, XdnaProgram, XdnaQueue, XdnaResourceCounts,
};
#[cfg(all(va_xdna, feature = "test-control"))]
#[doc(hidden)]
pub use native::{XdnaTestConfig, XdnaTestFault};

/// Admit a TOSA artifact and compile it to a precompiled-artifact container ([`artifact`]) with
/// the pinned toolchain, without touching the device.
///
/// This is the offline / catalog-population path (compiler-helper contract, issue #84): a build
/// host produces artifacts that a device-less serving host later loads through the precompiled
/// format. Requires the toolchain (`VIRTIO_ACCEL_AMDXDNA_TOOLCHAIN`) at run time and no HRX at
/// all — it is available on any unix build, with or without the `va_xdna` native modules.
#[cfg(unix)]
pub fn compile_artifact(
    tosa: &[u8],
    target: virtio_accel_tosa::Target,
) -> Result<Vec<u8>, virtio_accel_core::BackendError> {
    let spec = lower::admit(tosa, target)?;
    compiler::Compiler::from_env()?.compile(spec)
}

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

impl TosaCapabilityProvider for XdnaAccelerator {
    fn tosa_capabilities(&self) -> &'static [CapabilityDescriptor] {
        TOSA_CAPABILITIES
    }
}
