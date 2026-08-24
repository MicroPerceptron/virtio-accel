//! Detect a pinned HRX (`libhrx`) install and gate the native XDNA backend on it.
//!
//! HRX exposes a plain C ABI, so unlike the Hexagon backend there is no C++ bridge and no
//! `cc`/CMake step: `ffi.rs` will declare the ABI directly. The probe checks that the resolved
//! prefix carries the two headers and the shared library, plus a content check that the amdxdna
//! header names `hrx_amdxdna_executable_create` — the one function whose absence marks an older,
//! incompatible libhrx generation (see `docs/research/hrx-runtime-contract.md`). When all checks
//! pass the `va_xdna` cfg enables the native modules; otherwise the crate compiles the portable
//! admission surface plus an unsupported-runtime placeholder.
//!
//! Discovery is by explicit configuration only. `VIRTIO_ACCEL_HRX_DIR` (highest priority) or
//! `HRX_DIR` (the variable the pinned toolchain `env.sh` and the fork's own tooling export) name
//! the install prefix; `VIRTIO_ACCEL_HRX_LIB_DIR` links a bare lib directory that ships no headers.
//! No standard locations are scanned: silently discovering an unpinned libhrx would defeat the pin.

#![forbid(unsafe_code)]

use std::path::{Path, PathBuf};

const FORCE_ENV: &str = "VIRTIO_ACCEL_XDNA";
const DIR_ENV: &str = "VIRTIO_ACCEL_HRX_DIR";
const LIB_ENV: &str = "VIRTIO_ACCEL_HRX_LIB_DIR";

fn main() {
    // Declared on every build, including placeholder builds: the workspace compiles with
    // `-D warnings` and `unexpected_cfgs` is deny-by-default there.
    println!("cargo::rustc-check-cfg=cfg(va_xdna)");
    for variable in [FORCE_ENV, DIR_ENV, LIB_ENV, "HRX_DIR"] {
        println!("cargo:rerun-if-env-changed={variable}");
    }

    let forced = match std::env::var(FORCE_ENV) {
        Ok(value) if value == "0" => return,
        Ok(value) if value == "1" => true,
        Ok(other) => panic!("{FORCE_ENV} must be \"0\" or \"1\", not {other:?}"),
        Err(_) => false,
    };

    // Escape hatch: a bare lib directory with no accompanying headers (e.g. a relocated library).
    if let Ok(dir) = std::env::var(LIB_ENV) {
        println!("cargo:rustc-link-search=native={dir}");
        println!("cargo:rustc-link-lib=dylib=hrx");
        println!("cargo::rustc-cfg=va_xdna");
        return;
    }

    let prefix = std::env::var_os(DIR_ENV)
        .or_else(|| std::env::var_os("HRX_DIR"))
        .map(PathBuf::from);
    let Some(prefix) = prefix else {
        assert!(!forced, "{FORCE_ENV}=1 requires {DIR_ENV} or HRX_DIR");
        return;
    };

    let runtime_header = prefix.join("include/hrx/hrx_runtime.h");
    let amdxdna_header = prefix.join("include/hrx/hrx_amdxdna.h");
    let lib_dir = prefix.join("lib");
    let library = lib_dir.join("libhrx.so");

    let complete = runtime_header.is_file()
        && amdxdna_header.is_file()
        && library.is_file()
        && header_names_amdxdna_api(&amdxdna_header);
    if !complete {
        assert!(
            !forced,
            "{FORCE_ENV}=1 but {prefix:?} is not a complete amdxdna-native HRX prefix \
             (needs include/hrx/hrx_runtime.h, include/hrx/hrx_amdxdna.h declaring \
             hrx_amdxdna_executable_create, and lib/libhrx.so)"
        );
        return;
    }

    println!("cargo:rerun-if-changed={}", runtime_header.display());
    println!("cargo:rerun-if-changed={}", amdxdna_header.display());
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=dylib=hrx");
    println!("cargo::rustc-cfg=va_xdna");
}

/// Reject an older libhrx generation whose amdxdna header predates the native executable API.
///
/// Turns a confusing link/runtime failure (missing `hrx_amdxdna_executable_create`) into a clear
/// build-time one.
fn header_names_amdxdna_api(header: &Path) -> bool {
    std::fs::read_to_string(header)
        .map(|contents| contents.contains("hrx_amdxdna_executable_create"))
        .unwrap_or(false)
}
