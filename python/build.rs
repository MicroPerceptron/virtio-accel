//! Consume the backends' actual build decisions; never repeat their SDK probes here.
#![forbid(unsafe_code)]

fn main() {
    for backend in ["coreml", "vulkan", "openvino", "hexagon", "xdna"] {
        let cfg = format!("va_python_{backend}");
        println!("cargo::rustc-check-cfg=cfg({cfg})");
        let metadata = format!("DEP_VIRTIO_ACCEL_{}_NATIVE", backend.to_ascii_uppercase());
        println!("cargo:rerun-if-env-changed={metadata}");
        if std::env::var(metadata).as_deref() == Ok("1") {
            println!("cargo::rustc-cfg={cfg}");
        }
    }
}
