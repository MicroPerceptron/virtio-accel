//! Detect a complete QAIRT/QNN SDK and gate the native Hexagon backend on it.
//!
//! Driver packages and inference-only bundles can contain HTP support libraries without the
//! public QNN C development surface. Native compilation is enabled only when the SDK root has the
//! interface header and the Windows ARM64 HTP import library. All other builds retain the portable
//! TOSA admission/lowering code and an unsupported-runtime placeholder.

#![forbid(unsafe_code)]

use std::path::{Path, PathBuf};

const FORCE_ENV: &str = "VIRTIO_ACCEL_HEXAGON";
const ROOT_ENV: &str = "VIRTIO_ACCEL_QNN_SDK_ROOT";
const LIB_ENV: &str = "VIRTIO_ACCEL_QNN_LIB_DIR";
const DIRECT_FORCE_ENV: &str = "VIRTIO_ACCEL_HEXAGON_DIRECT";
const HEXAGON_SDK_ENV: &str = "HEXAGON_SDK_ROOT";

fn main() {
    println!("cargo::rustc-check-cfg=cfg(va_hexagon)");
    println!("cargo::rustc-check-cfg=cfg(va_hexagon_direct)");
    for variable in [
        FORCE_ENV,
        ROOT_ENV,
        LIB_ENV,
        "QNN_SDK_ROOT",
        DIRECT_FORCE_ENV,
        HEXAGON_SDK_ENV,
    ] {
        println!("cargo:rerun-if-env-changed={variable}");
    }

    let forced = match std::env::var(FORCE_ENV) {
        Ok(value) if value == "0" => return,
        Ok(value) if value == "1" => true,
        Ok(other) => panic!("{FORCE_ENV} must be \"0\" or \"1\", not {other:?}"),
        Err(_) => false,
    };

    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let target_arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    if target_os != "windows" || target_arch != "aarch64" {
        assert!(
            !forced,
            "{FORCE_ENV}=1 requires the initial windows/aarch64 Snapdragon target, found {target_os}/{target_arch}"
        );
        return;
    }

    configure_direct_htp();

    let root = std::env::var_os(ROOT_ENV)
        .or_else(|| std::env::var_os("QNN_SDK_ROOT"))
        .map(PathBuf::from);
    let Some(root) = root else {
        assert!(!forced, "{FORCE_ENV}=1 requires {ROOT_ENV} or QNN_SDK_ROOT");
        return;
    };

    let required_headers = [
        "QnnInterface.h",
        "QnnOpDef.h",
        "HTP/QnnHtpCommon.h",
        "HTP/QnnHtpGraph.h",
    ];
    let include_dir = [root.join("include/QNN"), root.join("include")]
        .into_iter()
        .find(|directory| {
            required_headers
                .iter()
                .all(|header| directory.join(header).is_file())
        });
    let Some(include_dir) = include_dir else {
        assert!(
            !forced,
            "{FORCE_ENV}=1 but {root:?} contains no complete public QNN/HTP header set"
        );
        return;
    };

    let lib_dir = std::env::var_os(LIB_ENV)
        .map(PathBuf::from)
        .unwrap_or_else(|| root.join("lib/aarch64-windows-msvc"));
    if !has_import_library(&lib_dir, "QnnHtp") {
        assert!(
            !forced,
            "{FORCE_ENV}=1 but {lib_dir:?} contains no QnnHtp import library"
        );
        return;
    }

    for header in required_headers {
        println!(
            "cargo:rerun-if-changed={}",
            include_dir.join(header).display()
        );
    }
    println!("cargo:rerun-if-changed=native/qnn_bridge.cpp");
    println!("cargo:rerun-if-changed=native/qnn_bridge.h");
    cc::Build::new()
        .cpp(true)
        .file("native/qnn_bridge.cpp")
        .include(include_dir)
        .flag_if_supported("/std:c++17")
        .flag_if_supported("/EHsc")
        .flag_if_supported("-std=c++17")
        .warnings(true)
        .compile("virtio_accel_qnn_bridge");
    println!(
        "cargo:rustc-env=VIRTIO_ACCEL_QNN_SDK_ROOT={}",
        root.display()
    );
    println!("cargo::rustc-cfg=va_hexagon");
}

fn configure_direct_htp() {
    let forced = match std::env::var(DIRECT_FORCE_ENV) {
        Ok(value) if value == "0" => return,
        Ok(value) if value == "1" => true,
        Ok(other) => panic!("{DIRECT_FORCE_ENV} must be \"0\" or \"1\", not {other:?}"),
        Err(_) => false,
    };
    let Some(root) = std::env::var_os(HEXAGON_SDK_ENV).map(PathBuf::from) else {
        assert!(!forced, "{DIRECT_FORCE_ENV}=1 requires {HEXAGON_SDK_ENV}");
        return;
    };
    let headers = [
        root.join("incs/remote.h"),
        root.join("incs/rpcmem.h"),
        root.join("incs/domain.h"),
    ];
    let generated = [
        Path::new("native/direct_htp/generated/va_htp.h"),
        Path::new("native/direct_htp/generated/va_htp_stub.c"),
    ];
    if !headers.iter().all(|path| path.is_file()) || !generated.iter().all(|path| path.is_file()) {
        assert!(
            !forced,
            "{DIRECT_FORCE_ENV}=1 but the Hexagon SDK or generated FastRPC stub is incomplete"
        );
        return;
    }
    for path in &headers {
        println!("cargo:rerun-if-changed={}", path.display());
    }
    for path in &generated {
        println!("cargo:rerun-if-changed={}", path.display());
    }
    println!("cargo:rerun-if-changed=native/direct_htp/host_bridge.cpp");
    println!("cargo:rerun-if-changed=native/direct_htp/host_bridge.h");
    cc::Build::new()
        .cpp(true)
        .file("native/direct_htp/host_bridge.cpp")
        .file("native/direct_htp/generated/va_htp_stub.c")
        .include("native/direct_htp")
        .include("native/direct_htp/generated")
        .include(root.join("incs"))
        .include(root.join("incs/stddef"))
        .define("WINNT", None)
        .define("STATIC_LIB", None)
        .flag_if_supported("/std:c++17")
        .flag_if_supported("/EHsc")
        .flag_if_supported("-std=c++17")
        .warnings(true)
        .compile("virtio_accel_direct_htp_bridge");
    println!("cargo:rustc-link-lib=Advapi32");
    println!("cargo::rustc-cfg=va_hexagon_direct");
}

fn has_import_library(directory: &Path, stem: &str) -> bool {
    [format!("{stem}.lib"), format!("lib{stem}.a")]
        .iter()
        .any(|name| directory.join(name).is_file())
}
