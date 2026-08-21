# virtio-accel-amdxdna: compiler-helper subprocess contract

Design record for [issue #84](https://github.com/MicroPerceptron/virtio-accel/issues/84)
(wayfinder map [#78](https://github.com/MicroPerceptron/virtio-accel/issues/78)). The
TOSA-admission ticket (#89) implements this contract inside the `compiler.rs` module reserved
by the #83 crate design. Grounding: the pinned toolchain proven on hardware (#81,
`~/toolchains/amdxdna-hrx-v2026.08`), the HRX artifact/fold-ABI facts (#79), the tier
decision (ADR-0001), and #75's decided constraints (compile during `load_program`, never
during `submit`; the compiler as a bounded subprocess, never a Cargo dependency).

**Alignment rule.** `virtio-accel-openvino` is the template because the architect wrote it —
its choices encode intent. This contract diverges from OpenVINO in exactly one place, forced
by the ecosystem: Intel ships its compiler *inside* `libopenvino_c` (one in-process C call),
while AMD's compiler is the aiecc/IRON Python + MLIR toolchain, which has no C API and must
not be linked in. Everything surrounding that divergence (admission flow, error surfaces,
env-var conventions, caching honesty) stays OpenVINO-shaped.

## 1. Helper form

An **in-repo Python program** (`compiler/helper/`, versioned with the crate, its version
string part of the cache key), executed as a subprocess via the pinned toolchain's venv
interpreter. It holds one small IRON design template per advertised op family — IDENTITY,
MATMUL bf16→fp32 (± fused CAST-to-bf16), MAX_POOL2D (bf16), integer MATMUL — instantiates
the template from the specialization, and drives `compile()` with explicit
`--xclbin-path`/`--insts-path` outputs (the #81-proven pattern; IRON's own JIT cache is
bypassed by explicit outputs and additionally quarantined, §3).

Rejected alternative: Rust-side MLIR (aie dialect) emission + bare `aiecc` CLI. It launches
the same Python (aiecc *is* Python) while transferring maintenance of MLIR generation and
tiling correctness from AMD's maintained IRON layer to us.

Two invocation modes:

- `compile`: read `spec.json` from the private workdir, write artifacts + `result.json`.
- `--identity`: print the installed toolchain identity as JSON (fork commit, `mlir_aie`,
  `llvm-aie`, HRX release tag, `hrx-xclbinutil` commit, helper version) — *measured* from the
  venv, not assumed, feeding the cache key (§6).

## 2. Boundary inputs and outputs

**Input: the specialization only — no guest bytes cross the boundary.** `lower.rs` admits the
TOSA graph first (target equality, ADR-0001 op×dtype surface), then derives `spec.json`:
op family, dtypes, static shapes, fusion flags, `fold_ddr_addr_offset: false` (the HRX ABI —
an XRT-folded `insts.bin` is not interchangeable), device (`npu2`), and the TOSA-bytes hash
(logging/cache correlation only). Every field is a validated integer or a value from a closed
enum; the injection surface is eliminated by construction — the same principle as OpenVINO's
"TOSA-declared names never reach the XML." The helper contains **no TOSA parser**; admission
logic exists in exactly one language.

(This deliberately narrows the ticket sketch's "validated TOSA bytes plus complete static
specialization": the bytes participate in the cache key as a hash, computed Rust-side, and
never enter the Python process.)

**Output:** `final.xclbin`, `insts.bin` (unfolded ABI), and `result.json`
(`{schema, ok, stage, message, artifacts, identity}`) in the workdir. The backend validates
sizes and the xclbin magic before HRX ever sees the artifacts (HRX validates again at
`hrx_amdxdna_executable_create`).

## 3. Environment pinning

Spawned with a **cleared environment**, then exactly:

| Variable | Value |
|---|---|
| `PEANO_INSTALL_DIR` | `$TOOLCHAIN/ironenv/lib/python3.12/site-packages/llvm-aie` (absolute) |
| `AIE_XCLBINUTIL` | `$TOOLCHAIN/amd-npu-compiler/bld-xclbinutil/tools/hrx-xclbinutil` (absolute; aiecc fails loudly if set-but-missing — desired) |
| `PATH` | `$TOOLCHAIN/ironenv/bin` + minimal system bin (for tools aiecc spawns) |
| `HOME`, `TMPDIR` | inside the private workdir |
| `NPU_CACHE_HOME` | inside the private workdir (quarantines IRON's JIT cache; no cross-invocation aliasing) |

Interpreter: `$TOOLCHAIN/ironenv/bin/python3` by absolute path. `$TOOLCHAIN` reaches the
backend as constructor configuration with env-var default `VIRTIO_ACCEL_AMDXDNA_TOOLCHAIN`
(runtime concern — the #83 build probe never touches the compiler). `NPU_RUNTIME` is not set:
the helper compiles, never dispatches. #81 verified the interpreter launches under `env -i`;
the first implementation test of this contract is a full clean-environment compile.

## 4. Bounds and lifecycle

- Private per-invocation workdir (0700), deleted on success; **kept on failure only behind a
  debug flag**, else deleted.
- Wall-clock timeout, default **600 s**, configurable (reference-machine kernel compiles ran
  ~1 min in #81).
- The helper runs in its **own process group**; timeout or backend drop kills the group
  (aiecc spawns children).
- stdout/stderr captured, capped at 256 KiB each (tail kept). Artifact size caps (default
  64 MiB each) enforced before acceptance.
- **Compiles are serialized** — at most one helper subprocess per accelerator instance.
  `load_program` is the slow path by contract (#75); memory stays bounded.

## 5. Failure taxonomy and diagnostics

`result.json` `stage ∈ {spec-rejected, template, compile, package, io}` + exit code.
Mapping:

| Condition | Backend surface |
|---|---|
| Toolchain root missing/incomplete at first use | load failure analogous to `InitError::RuntimeUnavailable` (OpenVINO pattern) |
| Helper failure on an admitted graph (any stage) | `BackendError::External { domain: AMDXDNA, code: stage }` — admission already passed, so this is a backend/toolchain fault, never blamed on the guest program |
| Timeout / killed | `External`, dedicated code |
| Missing/oversized/malformed artifacts | `External`, dedicated code |

Guest-visible errors carry **no host paths and no tool output** (threat-model hygiene);
captured stderr tails go to host-side logging only.

## 6. Cache — and the offline/no-toolchain mode

Content-addressed cache, on by default. Key = SHA-256 over: contract schema version ‖ hash of
the validated TOSA bytes ‖ `TargetIdentity` ‖ the full specialization ‖ device identity
(`1022:17f0` rev 0x20, `npu2`) ‖ the **measured** toolchain identity (§1 `--identity`) ‖
helper/lowering version ‖ the aiecc flag set including the fold-ABI bit (#79's lesson: HRX
and XRT artifacts must never alias). A prefix swap changes the measured identity and
invalidates every entry automatically.

Store: configurable directory, default `$XDG_CACHE_HOME/virtio-accel-amdxdna`. Entries hold
the artifact pair + `result.json`; validated on read (sizes, xclbin magic); size-capped
eviction (default 2 GiB, oldest-first). `load_program` consults the cache before spawning
anything.

**No-toolchain (catalog) mode — first-class.** Configured with a cache directory (possibly
read-only) and *no* toolchain root, the backend serves cache hits and cleanly rejects misses
(`BackendError::Unsupported`), with zero Python, zero toolchain, zero compiler on the serving
host. A prepopulated cache **is** an artifact catalog: compile the program zoo offline on a
build machine, ship the directory. This is the deployment shape for hosts that cannot or must
not carry the toolchain — including the kore future (§7).

## 7. Forward outlook: kore as host

kore — the capability-based OS intended as the eventual **host** (no Linux dependency) — will
never run CPython or the MLIR stack. This contract is built so it never has to:

- Dispatch (`submit`) is pure Rust + libhrx; Python exists only at compile time, and §6's
  catalog mode moves compile time **off the serving host entirely**.
- The out-of-process split is the *more* portable architecture than Intel's in-process
  compiler: a kore port replaces the thin runtime layer (Linux `amdxdna` driver + `libhrx`,
  isolated in `ffi.rs` per #83), never the compiler path.
- Non-TOSA workloads (e.g. voxel-rendering kernels) are out of scope for every backend in
  this repo — the protocol is TOSA-scoped, Intel included, so nothing is lost relative to the
  template. The honest future door is the artifact header's `format` field: a
  "precompiled XDNA artifact" format would let a sophisticated host scheduler (kore's
  manifold) submit offline-compiled raw kernels through `load_program`, bypassing TOSA and
  the compiler. That is a protocol change requiring classification and conformance rules —
  recorded here as a pointer, deliberately outside map #78.

## 8. Deferred

- #85: execution model, dispatch worker, event/timeout semantics (consumes the artifacts this
  contract produces).
- #89: implementation of this contract + the clean-environment compile proof and cache tests.
