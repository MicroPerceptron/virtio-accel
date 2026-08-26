//! The bounded aiecc compiler-helper subprocess (compiler-helper contract, issue #84).
//!
//! `load_program` calls [`Compiler::compile`] with a validated [`CompilerSpec`] (integers and
//! closed enums only — no guest bytes cross the boundary). The result is the precompiled artifact
//! container (`src/artifact.rs`) HRX consumes, content-addressed in a cache so a repeated program
//! shape never recompiles.
//!
//! The helper (`compiler/xdna_compile.py`, embedded here) runs under the pinned toolchain's venv
//! interpreter with a **cleared environment** plus only the toolchain prefix and a private workdir;
//! it self-configures the aiecc toolchain from the prefix. The compiler is never a Cargo dependency
//! and never runs in-process.

use std::fs;
use std::hash::{Hash, Hasher};
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use virtio_accel_core::BackendError;

use crate::XDNA_ERROR_DOMAIN;
use crate::artifact;
use crate::lower::{
    CompilerSpec, FP8_CAST_LINE_SIZE, Fp8Format, IDENTITY_LINE_SIZE, INT8_IDENTITY_MAX_LINE_SIZE,
    INT8_MATMUL_MAX_TOTAL_BYTES, MATMUL_MAX_DIM, MATMUL_TILE_K, MATMUL_TILE_M, MATMUL_TILE_N,
    MAX_POOL_MAX_KERNEL, MAX_POOL_MAX_STRIDE, MAX_POOL_MAX_TOTAL_ELEMENTS,
};

/// The embedded compiler helper; written into each private workdir before invocation.
const HELPER_SOURCE: &str = include_str!("../compiler/xdna_compile.py");

/// Default wall-clock bound for one compile (reference-machine compiles run ~1 min).
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(600);

/// Error-code offsets within [`XDNA_ERROR_DOMAIN`] for compiler failures.
mod code {
    pub const TOOLCHAIN_MISSING: i64 = 100;
    pub const SPAWN: i64 = 101;
    pub const TIMEOUT: i64 = 102;
    pub const HELPER_FAILED: i64 = 103;
    pub const BAD_OUTPUT: i64 = 104;
    pub const IO: i64 = 105;
}

fn external(code: i64) -> BackendError {
    BackendError::External {
        domain: XDNA_ERROR_DOMAIN,
        code,
    }
}

/// The `spec.json` the helper reads: integers and closed enums only.
fn spec_json(spec: CompilerSpec) -> String {
    // Common trailer: every spec targets npu2 in the unfolded-DDR ABI.
    let device = "\"device\":\"npu2\",\"fold_ddr_addr_offset\":false";
    match spec {
        CompilerSpec::Identity { elements } => {
            format!(
                "{{\"op\":\"IDENTITY\",\"dtype\":\"bf16\",\"elements\":{elements},\
                 \"line_size\":{IDENTITY_LINE_SIZE},{device}}}"
            )
        }
        CompilerSpec::Int8Identity {
            elements,
            line_size,
        } => format!(
            "{{\"op\":\"IDENTITY\",\"dtype\":\"i8\",\"elements\":{elements},\
             \"line_size\":{line_size},\"max_line_size\":{INT8_IDENTITY_MAX_LINE_SIZE},{device}}}"
        ),
        CompilerSpec::Fp8ToBf16 { format, elements } => {
            let input = match format {
                Fp8Format::E4M3 => "fp8e4m3",
                Fp8Format::E5M2 => "fp8e5m2",
            };
            format!(
                "{{\"op\":\"CAST\",\"in_dtype\":\"{input}\",\"out_dtype\":\"bf16\",\
                 \"elements\":{elements},\"line_size\":{FP8_CAST_LINE_SIZE},{device}}}"
            )
        }
        CompilerSpec::Matmul { m, k, n } => format!(
            "{{\"op\":\"MATMUL\",\"in_dtype\":\"bf16\",\"out_dtype\":\"f32\",\
             \"m\":{m},\"k\":{k},\"n\":{n},\"tile_m\":{MATMUL_TILE_M},\
             \"tile_k\":{MATMUL_TILE_K},\"tile_n\":{MATMUL_TILE_N},\
             \"max_dim\":{MATMUL_MAX_DIM},{device}}}"
        ),
        CompilerSpec::Int8Matmul {
            m,
            k,
            n,
            left_zero_point,
            right_zero_point,
        } => format!(
            "{{\"op\":\"MATMUL\",\"in_dtype\":\"i8\",\"out_dtype\":\"i32\",\
             \"m\":{m},\"k\":{k},\"n\":{n},\"left_zero_point\":{left_zero_point},\
             \"right_zero_point\":{right_zero_point},\"max_dim\":{MATMUL_MAX_DIM},\
             \"max_total_bytes\":{INT8_MATMUL_MAX_TOTAL_BYTES},{device}}}"
        ),
        CompilerSpec::MaxPool2d {
            input_h,
            input_w,
            channels,
            output_h,
            output_w,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
        } => format!(
            "{{\"op\":\"MAX_POOL2D\",\"dtype\":\"bf16\",\"layout\":\"NHWC\",\
             \"batch\":1,\"input_h\":{input_h},\"input_w\":{input_w},\
             \"channels\":{channels},\"output_h\":{output_h},\"output_w\":{output_w},\
             \"kernel_h\":{kernel_h},\"kernel_w\":{kernel_w},\
             \"stride_h\":{stride_h},\"stride_w\":{stride_w},\
             \"pad\":[0,0,0,0],\"nan_mode\":\"PROPAGATE\",\
             \"max_kernel\":{MAX_POOL_MAX_KERNEL},\"max_stride\":{MAX_POOL_MAX_STRIDE},\
             \"max_total_elements\":{MAX_POOL_MAX_TOTAL_ELEMENTS},{device}}}"
        ),
    }
}

/// The compiler-helper driver: a pinned toolchain prefix and a content-addressed artifact cache.
pub struct Compiler {
    toolchain: PathBuf,
    interpreter: PathBuf,
    cache_dir: PathBuf,
    timeout: Duration,
    /// Measured toolchain identity (the helper's `identity` mode output), computed once per driver
    /// and mixed into the cache key so an in-place toolchain update cannot serve stale artifacts.
    identity: std::sync::OnceLock<String>,
}

impl Compiler {
    /// Resolve the toolchain prefix (`VIRTIO_ACCEL_AMDXDNA_TOOLCHAIN`) and the cache directory
    /// (`VIRTIO_ACCEL_XDNA_CACHE`, else `$XDG_CACHE_HOME`/`$HOME`-derived).
    pub fn from_env() -> Result<Self, BackendError> {
        let toolchain = std::env::var_os("VIRTIO_ACCEL_AMDXDNA_TOOLCHAIN")
            .map(PathBuf::from)
            .ok_or_else(|| external(code::TOOLCHAIN_MISSING))?;
        let interpreter = toolchain.join("ironenv/bin/python3");
        if !interpreter.is_file() {
            return Err(external(code::TOOLCHAIN_MISSING));
        }
        let cache_dir = cache_root()?;
        Ok(Self {
            toolchain,
            interpreter,
            cache_dir,
            timeout: DEFAULT_TIMEOUT,
            identity: std::sync::OnceLock::new(),
        })
    }

    /// The installed toolchain's measured identity: the embedded helper's `identity` mode reports
    /// the key package versions (`mlir_aie`, `llvm-aie`) as JSON. Computed lazily, once per driver.
    /// If the probe fails the marker still keys the cache on the failure itself, so a
    /// half-installed toolchain never shares entries with a working one.
    fn toolchain_identity(&self) -> &str {
        self.identity.get_or_init(|| {
            let probe = || -> Option<String> {
                fs::create_dir_all(&self.cache_dir).ok()?;
                let helper = self.cache_dir.join(format!(
                    ".identity-{}-{}.py",
                    std::process::id(),
                    unique_suffix()
                ));
                fs::write(&helper, HELPER_SOURCE).ok()?;
                let output = Command::new(&self.interpreter)
                    .arg(&helper)
                    .arg("identity")
                    .env_clear()
                    .env("PATH", "/usr/bin:/bin")
                    .output();
                let _ = fs::remove_file(&helper);
                let output = output.ok()?;
                if !output.status.success() {
                    return None;
                }
                String::from_utf8(output.stdout).ok()
            };
            probe().unwrap_or_else(|| "identity-unavailable".to_owned())
        })
    }

    /// Compile `spec` to a precompiled-artifact container, using the cache when warm.
    pub fn compile(&self, spec: CompilerSpec) -> Result<Vec<u8>, BackendError> {
        let key = self.cache_key(spec);
        let cached = self.cache_dir.join(format!("{key}.xdnp"));
        if let Ok(bytes) = fs::read(&cached) {
            if artifact::PrecompiledArtifact::parse(&bytes).is_ok() {
                return Ok(bytes);
            }
        }
        let bytes = self.run_helper(spec)?;
        // Validate before caching, then write atomically (temp + rename) so a concurrent reader
        // never observes a partial file.
        artifact::PrecompiledArtifact::parse(&bytes).map_err(|_| external(code::BAD_OUTPUT))?;
        fs::create_dir_all(&self.cache_dir).map_err(|_| external(code::IO))?;
        // The staging name carries the pid: two processes compiling the same key must not share a
        // staging path (each process's counter starts at zero).
        let staging = self.cache_dir.join(format!(
            "{key}.{}-{}.tmp",
            std::process::id(),
            unique_suffix()
        ));
        if fs::write(&staging, &bytes).is_ok() {
            let _ = fs::rename(&staging, &cached);
        }
        Ok(bytes)
    }

    /// Content-address the artifact: spec ‖ toolchain prefix ‖ measured toolchain identity ‖
    /// embedded helper source. Two graphs with the same spec compile to the same artifact, so the
    /// spec is a complete key; the prefix, the measured package versions (so an in-place toolchain
    /// update invalidates), and the helper source invalidate on any compiler change.
    fn cache_key(&self, spec: CompilerSpec) -> String {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        2u32.hash(&mut hasher); // key schema version
        spec.hash(&mut hasher);
        self.toolchain.hash(&mut hasher);
        self.toolchain_identity().hash(&mut hasher);
        HELPER_SOURCE.hash(&mut hasher);
        format!("{:016x}", hasher.finish())
    }

    fn run_helper(&self, spec: CompilerSpec) -> Result<Vec<u8>, BackendError> {
        let workdir =
            self.cache_dir
                .join(format!(".wk-{}-{}", std::process::id(), unique_suffix()));
        let result = self.run_helper_in(spec, &workdir);
        let _ = fs::remove_dir_all(&workdir);
        result
    }

    fn run_helper_in(&self, spec: CompilerSpec, workdir: &Path) -> Result<Vec<u8>, BackendError> {
        // The kernel-object cache (NPU_CACHE_HOME) persists across compiles: the compute kernel is
        // byte-identical for every admitted shape of one template, so a per-run cache would re-run
        // the full Peano kernel build inside every cold compile. The spec is integers and closed
        // enums only, so no guest bytes ever enter this cache; a toolchain change moves the whole
        // artifact-cache key, and IRON keys its own entries by content underneath.
        let kernel_cache = self.cache_dir.join("npu-cache");
        fs::create_dir_all(&kernel_cache).map_err(|_| external(code::IO))?;
        fs::create_dir_all(workdir).map_err(|_| external(code::IO))?;
        fs::write(workdir.join("spec.json"), spec_json(spec)).map_err(|_| external(code::IO))?;
        let helper = workdir.join("xdna_compile.py");
        fs::write(&helper, HELPER_SOURCE).map_err(|_| external(code::IO))?;

        // Cleared environment plus only the pins the helper self-configures from.
        let mut child = Command::new(&self.interpreter)
            .arg(&helper)
            .arg("compile")
            .arg(workdir)
            .env_clear()
            .env("VIRTIO_ACCEL_AMDXDNA_TOOLCHAIN", &self.toolchain)
            .env("PATH", "/usr/bin:/bin")
            .env("HOME", workdir)
            .env("TMPDIR", workdir)
            .env("NPU_CACHE_HOME", &kernel_cache)
            .process_group(0)
            .spawn()
            .map_err(|_| external(code::SPAWN))?;

        let deadline = Instant::now() + self.timeout;
        let mut poll_interval = Duration::from_millis(20);
        loop {
            match child.try_wait() {
                Ok(Some(status)) => {
                    if !status.success() {
                        return Err(external(code::HELPER_FAILED));
                    }
                    break;
                }
                Ok(None) => {
                    if Instant::now() >= deadline {
                        // Bound the whole aiecc tree, not just the interpreter: the helper runs in
                        // its own process group (`process_group(0)` above, so pgid == child pid)
                        // precisely so a wedged grandchild (Peano, xclbinutil) dies with it. std
                        // exposes no killpg, so signal the group with kill(1); the un-reaped child
                        // keeps the pid (and thus the pgid) reserved until `wait` below.
                        let _ = Command::new("kill")
                            .args(["-s", "KILL", "--", &format!("-{}", child.id())])
                            .status();
                        let _ = child.kill();
                        let _ = child.wait();
                        return Err(external(code::TIMEOUT));
                    }
                    let remaining = deadline.saturating_duration_since(Instant::now());
                    std::thread::sleep(poll_interval.min(remaining));
                    poll_interval = (poll_interval + poll_interval).min(Duration::from_secs(1));
                }
                Err(_) => return Err(external(code::HELPER_FAILED)),
            }
        }

        // The helper reports its per-slot binding plan in result.json; read the artifacts and
        // package them.
        let xclbin =
            fs::read(workdir.join("final.xclbin")).map_err(|_| external(code::BAD_OUTPUT))?;
        let insts = fs::read(workdir.join("insts.bin")).map_err(|_| external(code::BAD_OUTPUT))?;
        let (input_bytes, output_bytes) = read_binding_plan(&workdir.join("result.json"))?;
        Ok(artifact::encode(
            "MLIR_AIE",
            &input_bytes,
            &output_bytes,
            &xclbin,
            &insts,
        ))
    }
}

/// Parse the binding plan (`ok`, `input_bytes`, `output_bytes`) from the helper's `result.json` —
/// a tiny file whose exact shape the embedded helper controls.
fn read_binding_plan(path: &Path) -> Result<(Vec<u64>, Vec<u64>), BackendError> {
    let text = fs::read_to_string(path).map_err(|_| external(code::BAD_OUTPUT))?;
    let field = |name: &str| -> Option<&str> {
        let needle = format!("\"{name}\"");
        let start = text.find(&needle)? + needle.len();
        Some(text[start..].trim_start().strip_prefix(':')?.trim_start())
    };
    // The exit status already gates on success; re-check the helper's own verdict as well.
    if !field("ok").is_some_and(|rest| rest.starts_with("true")) {
        return Err(external(code::BAD_OUTPUT));
    }
    let array = |name: &str| -> Option<Vec<u64>> {
        let rest = field(name)?.strip_prefix('[')?;
        let body = &rest[..rest.find(']')?];
        body.split(',')
            .map(|item| item.trim().parse::<u64>().ok().filter(|size| *size > 0))
            .collect()
    };
    match (array("input_bytes"), array("output_bytes")) {
        (Some(inputs), Some(outputs)) if !inputs.is_empty() && !outputs.is_empty() => {
            Ok((inputs, outputs))
        }
        _ => Err(external(code::BAD_OUTPUT)),
    }
}

fn cache_root() -> Result<PathBuf, BackendError> {
    if let Some(dir) = std::env::var_os("VIRTIO_ACCEL_XDNA_CACHE") {
        return Ok(PathBuf::from(dir));
    }
    if let Some(dir) = std::env::var_os("XDG_CACHE_HOME") {
        return Ok(PathBuf::from(dir).join("virtio-accel-xdna"));
    }
    let home = std::env::var_os("HOME").ok_or_else(|| external(code::IO))?;
    Ok(PathBuf::from(home).join(".cache/virtio-accel-xdna"))
}

fn unique_suffix() -> u64 {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    COUNTER.fetch_add(1, Ordering::Relaxed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_spec_carries_the_authoritative_line_size() {
        let json = spec_json(CompilerSpec::Identity { elements: 4096 });
        assert!(json.contains("\"line_size\":1024"));
    }

    #[test]
    fn int8_identity_spec_carries_the_admitted_line_size() {
        let json = spec_json(CompilerSpec::Int8Identity {
            elements: 8,
            line_size: 8,
        });
        assert!(json.contains("\"dtype\":\"i8\""));
        assert!(json.contains("\"line_size\":8"));
        assert!(json.contains("\"max_line_size\":1024"));
    }

    #[test]
    fn matmul_spec_carries_the_authoritative_tiling_envelope() {
        let json = spec_json(CompilerSpec::Matmul {
            m: 64,
            k: 128,
            n: 96,
        });
        for field in [
            "\"tile_m\":32",
            "\"tile_k\":64",
            "\"tile_n\":32",
            "\"max_dim\":512",
        ] {
            assert!(json.contains(field), "missing {field} in {json}");
        }
    }

    #[test]
    fn int8_matmul_spec_preserves_zero_points_and_memory_bound() {
        let json = spec_json(CompilerSpec::Int8Matmul {
            m: 2,
            k: 3,
            n: 2,
            left_zero_point: -2,
            right_zero_point: 3,
        });
        for field in [
            "\"in_dtype\":\"i8\"",
            "\"out_dtype\":\"i32\"",
            "\"left_zero_point\":-2",
            "\"right_zero_point\":3",
            "\"max_total_bytes\":16384",
        ] {
            assert!(json.contains(field), "missing {field} in {json}");
        }
    }

    #[test]
    fn fp8_cast_spec_names_the_encoding_and_authoritative_line_size() {
        for (format, dtype) in [(Fp8Format::E4M3, "fp8e4m3"), (Fp8Format::E5M2, "fp8e5m2")] {
            let json = spec_json(CompilerSpec::Fp8ToBf16 {
                format,
                elements: 4096,
            });
            assert!(json.contains(&format!("\"in_dtype\":\"{dtype}\"")));
            assert!(json.contains("\"out_dtype\":\"bf16\""));
            assert!(json.contains("\"line_size\":1024"));
        }
    }

    #[test]
    fn max_pool_spec_carries_shape_attributes_and_memory_envelope() {
        let json = spec_json(CompilerSpec::MaxPool2d {
            input_h: 4,
            input_w: 4,
            channels: 2,
            output_h: 2,
            output_w: 2,
            kernel_h: 2,
            kernel_w: 2,
            stride_h: 2,
            stride_w: 2,
        });
        for field in [
            "\"layout\":\"NHWC\"",
            "\"input_h\":4",
            "\"output_h\":2",
            "\"kernel_h\":2",
            "\"stride_h\":2",
            "\"nan_mode\":\"PROPAGATE\"",
            "\"max_total_elements\":8192",
        ] {
            assert!(json.contains(field), "missing {field} in {json}");
        }
    }
}
