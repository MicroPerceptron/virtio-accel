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

use crate::artifact;
use crate::lower::CompilerSpec;
use crate::native::XDNA_ERROR_DOMAIN;

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
            format!("{{\"op\":\"IDENTITY\",\"dtype\":\"bf16\",\"elements\":{elements},{device}}}")
        }
        CompilerSpec::Matmul { m, k, n } => format!(
            "{{\"op\":\"MATMUL\",\"in_dtype\":\"bf16\",\"out_dtype\":\"f32\",\
             \"m\":{m},\"k\":{k},\"n\":{n},{device}}}"
        ),
    }
}

/// The compiler-helper driver: a pinned toolchain prefix and a content-addressed artifact cache.
pub struct Compiler {
    toolchain: PathBuf,
    interpreter: PathBuf,
    cache_dir: PathBuf,
    timeout: Duration,
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
        let staging = self
            .cache_dir
            .join(format!("{key}.{}.tmp", unique_suffix()));
        if fs::write(&staging, &bytes).is_ok() {
            let _ = fs::rename(&staging, &cached);
        }
        Ok(bytes)
    }

    /// Content-address the artifact: spec ‖ toolchain prefix ‖ embedded helper source. Two graphs
    /// with the same op/dtype/element count compile to the same artifact, so the spec is a complete
    /// key; the prefix and helper source invalidate on a toolchain or helper change.
    fn cache_key(&self, spec: CompilerSpec) -> String {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        1u32.hash(&mut hasher); // key schema version
        spec.hash(&mut hasher);
        self.toolchain.hash(&mut hasher);
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
        let cache = workdir.join("cache");
        fs::create_dir_all(&cache).map_err(|_| external(code::IO))?;
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
            .env("NPU_CACHE_HOME", &cache)
            .process_group(0)
            .spawn()
            .map_err(|_| external(code::SPAWN))?;

        let deadline = Instant::now() + self.timeout;
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
                        let _ = child.kill();
                        let _ = child.wait();
                        return Err(external(code::TIMEOUT));
                    }
                    std::thread::sleep(Duration::from_millis(20));
                }
                Err(_) => return Err(external(code::HELPER_FAILED)),
            }
        }

        // The helper reports its I/O plan in result.json; read the artifacts and package them.
        let xclbin =
            fs::read(workdir.join("final.xclbin")).map_err(|_| external(code::BAD_OUTPUT))?;
        let insts = fs::read(workdir.join("insts.bin")).map_err(|_| external(code::BAD_OUTPUT))?;
        let (inputs, outputs) = read_io_counts(&workdir.join("result.json"))?;
        Ok(artifact::encode(
            "MLIR_AIE", inputs, outputs, &xclbin, &insts,
        ))
    }
}

/// Parse `inputs`/`outputs` from the helper's `result.json` (a tiny, helper-authored file).
fn read_io_counts(path: &Path) -> Result<(u32, u32), BackendError> {
    let text = fs::read_to_string(path).map_err(|_| external(code::BAD_OUTPUT))?;
    let field = |name: &str| -> Option<u32> {
        let needle = format!("\"{name}\"");
        let start = text.find(&needle)? + needle.len();
        let rest = text[start..].trim_start().strip_prefix(':')?.trim_start();
        let digits: String = rest.chars().take_while(char::is_ascii_digit).collect();
        digits.parse().ok()
    };
    match (field("inputs"), field("outputs")) {
        (Some(inputs), Some(outputs)) if inputs > 0 => Ok((inputs, outputs)),
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
