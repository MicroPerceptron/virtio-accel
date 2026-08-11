#![forbid(unsafe_code)]

use std::env;

fn main() {
    println!("cargo:rerun-if-changed=native/coreml_bridge.h");
    println!("cargo:rerun-if-changed=native/coreml_bridge.m");

    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos") {
        return;
    }

    cc::Build::new()
        .file("native/coreml_bridge.m")
        .flag("-fobjc-arc")
        .flag("-fblocks")
        .flag("-fmodules")
        .flag("-mmacosx-version-min=14.0")
        .warnings(true)
        .compile("virtio_accel_coreml_bridge");

    println!("cargo:rustc-link-lib=framework=CoreML");
    println!("cargo:rustc-link-lib=framework=Foundation");
}
