#![forbid(unsafe_code)]

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=native/coreml_bridge.h");
    println!("cargo:rerun-if-changed=native/coreml_bridge.m");

    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos") {
        return;
    }

    let output = PathBuf::from(env::var_os("OUT_DIR").expect("Cargo must provide OUT_DIR"));
    let module_cache = output.join("clang-module-cache");
    cc::Build::new()
        .file("native/coreml_bridge.m")
        .flag("-fobjc-arc")
        .flag("-fblocks")
        .flag("-fmodules")
        .flag(format!("-fmodules-cache-path={}", module_cache.display()))
        .flag("-mmacosx-version-min=14.0")
        .warnings(true)
        .compile("virtio_accel_coreml_bridge");

    // Apple's linker requires Mach-O members to begin at an 8-byte archive offset. The generic
    // archiver selected by `cc` only guarantees the traditional 2-byte alignment, so whether a
    // bridge links would otherwise depend on the preceding member sizes. Re-archive with Apple's
    // `libtool`, which records the required member alignment deterministically.
    let archive = output.join("libvirtio_accel_coreml_bridge.a");
    let aligned_archive = output.join("libvirtio_accel_coreml_bridge.aligned.a");
    let object = std::fs::read_dir(&output)
        .expect("failed to inspect native build output")
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .find(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.ends_with("-coreml_bridge.o"))
        })
        .expect("cc did not produce the Core ML bridge object");
    let status = Command::new("xcrun")
        .args(["libtool", "-static", "-o"])
        .arg(&aligned_archive)
        .arg(object)
        .status()
        .expect("failed to launch xcrun libtool");
    assert!(
        status.success(),
        "xcrun libtool failed to align bridge archive"
    );
    std::fs::rename(aligned_archive, archive).expect("failed to install aligned bridge archive");

    println!("cargo:rustc-link-lib=framework=CoreML");
    println!("cargo:rustc-link-lib=framework=Foundation");
}
