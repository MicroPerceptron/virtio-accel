//! Build-time gate for the optional Vulkan native path.
//!
//! Unlike the SDK-probing backends (`va_openvino`, `va_hexagon`, `va_xdna`), there is nothing to
//! detect on disk: `ash` under its `loaded` feature loads the platform's Vulkan loader
//! dynamically at run time (ADR 0002 in `docs/adr/`). The `va_vulkan` cfg therefore enumerates the
//! host target operating systems on which a Vulkan loader could exist, and runtime absence of the
//! loader surfaces as `InitError::RuntimeUnavailable`. The three-state `VIRTIO_ACCEL_VULKAN`
//! control and loud force-on failure semantics survive regardless.

#![forbid(unsafe_code)]

/// Host target operating systems on which a Vulkan loader may be dynamically loaded.
const SUPPORTED_TARGETS: &[&str] = &["android", "linux", "macos", "windows"];

fn main() {
    println!("cargo::rustc-check-cfg=cfg(va_vulkan)");
    println!("cargo:rerun-if-env-changed=VIRTIO_ACCEL_VULKAN");

    let forced = match std::env::var("VIRTIO_ACCEL_VULKAN") {
        Ok(value) if value == "0" => return,
        Ok(value) if value == "1" => true,
        Ok(other) => panic!("VIRTIO_ACCEL_VULKAN must be \"0\" or \"1\", not {other:?}"),
        Err(_) => false,
    };

    let os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if SUPPORTED_TARGETS.contains(&os.as_str()) {
        println!("cargo::rustc-cfg=va_vulkan");
    } else {
        assert!(
            !forced,
            "VIRTIO_ACCEL_VULKAN=1 but target OS {os:?} is not a Vulkan loader host target"
        );
    }
}
