//! Detect an OpenVINO C runtime and gate the native backend on its presence.
//!
//! Unlike the Core ML crate, the boundary is not a target operating system: OpenVINO is an
//! installable runtime, so continuous-integration hosts for the portable workspace do not have
//! it. When `libopenvino_c` is found the `va_openvino` cfg enables the native modules; otherwise
//! the crate compiles the portable lowering module plus an unsupported-runtime placeholder.

#![forbid(unsafe_code)]

fn main() {
    // The cfg must be declared on every build, including placeholder builds, because the
    // workspace compiles with `-D warnings` and `unexpected_cfgs` is deny-by-default there.
    println!("cargo::rustc-check-cfg=cfg(va_openvino)");
    println!("cargo:rerun-if-env-changed=VIRTIO_ACCEL_OPENVINO");
    println!("cargo:rerun-if-env-changed=VIRTIO_ACCEL_OPENVINO_LIB_DIR");

    let forced = match std::env::var("VIRTIO_ACCEL_OPENVINO") {
        Ok(value) if value == "0" => return,
        Ok(value) if value == "1" => true,
        Ok(other) => panic!("VIRTIO_ACCEL_OPENVINO must be \"0\" or \"1\", not {other:?}"),
        Err(_) => false,
    };

    // Escape hatch for archive or wheel installs that ship no pkg-config metadata.
    if let Ok(dir) = std::env::var("VIRTIO_ACCEL_OPENVINO_LIB_DIR") {
        println!("cargo:rustc-link-search=native={dir}");
        println!("cargo:rustc-link-lib=dylib=openvino_c");
        println!("cargo::rustc-cfg=va_openvino");
        return;
    }

    // `cargo_metadata(false)` is load-bearing: the module's default link metadata names every
    // OpenVINO frontend library, while this backend needs exactly the C API (its dynamic
    // dependencies pull in the core runtime).
    match pkg_config::Config::new()
        .cargo_metadata(false)
        .probe("openvino")
    {
        Ok(library) => {
            for path in &library.link_paths {
                println!("cargo:rustc-link-search=native={}", path.display());
            }
            println!("cargo:rustc-link-lib=dylib=openvino_c");
            println!("cargo::rustc-cfg=va_openvino");
        }
        Err(error) => {
            assert!(
                !forced,
                "VIRTIO_ACCEL_OPENVINO=1 but no OpenVINO runtime was found: {error}"
            );
        }
    }
}
