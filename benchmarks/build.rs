//! Consume the backends' actual build decisions; never repeat their SDK probes here.
#![forbid(unsafe_code)]

fn main() {
    for key in ["TARGET", "PROFILE", "OPT_LEVEL"] {
        println!(
            "cargo:rustc-env=VA_BENCH_BUILD_{key}={}",
            std::env::var(key).unwrap()
        );
    }
    for backend in ["coreml", "vulkan", "openvino", "hexagon", "xdna"] {
        let cfg = format!("va_bench_{backend}");
        println!("cargo::rustc-check-cfg=cfg({cfg})");
        let metadata = format!("DEP_VIRTIO_ACCEL_{}_NATIVE", backend.to_ascii_uppercase());
        println!("cargo:rerun-if-env-changed={metadata}");
        if std::env::var(metadata).as_deref() == Ok("1") {
            println!("cargo::rustc-cfg={cfg}");
        }
    }
}
