# HRX runtime contract, toolchain pinning, and device support

Research notes for [issue #79](https://github.com/MicroPerceptron/virtio-accel/issues/79),
feeding the AMD XDNA NPU backend design ([issue #75](https://github.com/MicroPerceptron/virtio-accel/issues/75),
crate `virtio-accel-amdxdna`).

**Primary sources.** Everything below was read directly from:

- `MicroPerceptron/amd-npu-compiler` (fork of `Xilinx/mlir-aie`), default branch `main`,
  HEAD commit **`c95544269f0c074d6d3e213ee43cc34dc4100801`** (2026-08-14). All fork file
  paths and quotes are at this commit unless noted.
- The pinned HRX binary release **`jtuyls/hrx` tag `flm-hrx-amdxdna-v2026.07.30`**
  (published 2026-07-30). The Linux asset was downloaded and its SHA-256 verified against
  the fork's pin (`661ed94051cc6ad04f53739b2df7a791aecb658bc435bd5a6ff3c46716696345`);
  the C ABI below is quoted from the headers inside that asset.
- Linux kernel and linux-firmware references for the `amdxdna` driver (cited inline in §6).

Confidence levels are stated explicitly; §8 lists what could **not** be confirmed.

---

## 1. What HRX is, and when it is selected

Source: `programming_guide/hrx_runtime.md` (fork).

- **HRX** is an opt-in host runtime for IRON that dispatches designs on the AMD XDNA NPU
  through **`libhrx`** — an **IREE-based runtime with an `amdxdna` HAL** — consuming the
  same `aiecc` artifacts (`final.xclbin` + `insts.bin`) as the default XRT path.
  It requires **no XRT userspace at runtime**; combined with the bundled XRT-free
  `hrx-xclbinutil`, the whole build-and-run flow works on a machine with no XRT install.
- **Selection**: one variable, `NPU_RUNTIME`, drives both flows
  (`NPU_RUNTIME=hrx python design.py` for IRON/Python; `NPU_RUNTIME=hrx make run` for C++).
  Semantics (from `programming_guide/iron_configuration.md`, §"Host-runtime backend
  selection"): `auto` (default) = XRT if `pyxrt` imports, else CPU — **`auto` never
  selects HRX**; `xrt` = force XRT; `hrx` = force HRX (hard error at import if
  `libhrx.so` cannot be located). An invalid value is a hard error.
- **Package layout** (`python/utils/hostruntime/hrxruntime/`): `discovery.py`
  (path-only probe, no `dlopen`; backs `aie.utils.has_hrx`), `_bindings.py` (ctypes C ABI
  layer), `context.py` (`HRXContext` process-wide singleton), `tensor.py` (`HRXTensor`,
  persistent host-mapped buffer), `hostruntime.py` (`HRXHostRuntime` /
  `CachedHRXRuntime`, LRU executable cache, default size 32 via `HRX_EXE_CACHE_SIZE`).
- **Env vars** (documented in `programming_guide/iron_configuration.md`): discovery hints
  `HRX_LIBHRX` (full path) > `LIBHRX_DIR` > `HRX_DIR` (prefix), then standard locations
  (sibling `hrx` checkout, `$HOME/hrx`, `/opt/hrx`, `/usr/local/hrx`, loader path);
  behavior: `IRON_HRX_DEVICE` (force `npu1`/`npu2` instead of sysfs detection),
  `HRX_EXE_CACHE_SIZE` (default 32), `IRON_HRX_TIMEOUT` (watchdog seconds bounding the
  wait in `hrx_stream_synchronize`; 0/unset disables; on expiry the sync **cannot be
  cancelled** — the error is diagnosable but the device work keeps running).

## 2. Artifact contract: `final.xclbin` + `insts.bin`

Sources: `python/utils/hostruntime/hrxruntime/README.md` (the self-contained HRX runbook),
`runtime_lib/test_lib/hrx_test_wrapper.h`, `python/utils/hostruntime/hrxruntime/hostruntime.py`,
`include/hrx/hrx_amdxdna.h` (pinned release).

- **`final.xclbin`**: the packaged AMD xclbin container produced by `aiecc`, holding the
  PDI plus metadata sections. The bundled `hrx-xclbinutil` packages exactly the sections
  an NPU xclbin needs: `MEM_TOPOLOGY`, `AIE_PARTITION`, `EMBEDDED_METADATA`, `IP_LAYOUT`,
  `CONNECTIVITY`, `GROUP_TOPOLOGY`, `GROUP_CONNECTIVITY` (README §3b). A lit regression
  test, `test/aiecc/hrx_xclbin_sections.mlir`, asserts the first three are present.
- **`insts.bin`**: the raw **XAie transaction (TXN) stream** as little-endian `uint32`
  words — handed to libhrx **verbatim**. Length must be a multiple of 4 bytes
  (`context.py::create_executable` rejects otherwise). For an ELF input
  (`aiecc --aie-generate-elf`), the `.ctrltext` section *is* the TXN verbatim and is
  extracted host-side by `control_code_from_elf` (Python: `_bindings.py`; C++:
  `hrx_test_wrapper.h`) before being passed as the transaction.
- **Patch-table derivation**: HRX's amdxdna COMMAND_CHAIN path host-patches each I/O
  buffer's device address into the control code using a patch table of
  `(offset, arg_idx, addend)` triples. As of the `amdxdna-hal-native-rel` API, **libhrx
  derives this patch table itself inside `hrx_amdxdna_executable_create`, by scanning the
  XAie transaction's `BLOCKWRITE`/`DDR_PATCH` ops** (README §7; restated in
  `hrx_test_wrapper.h` header comment). The host does no patch-table extraction and no
  XADX serialization; `aiebu-asm`/XRT are not required. `hrx_amdxdna.h` phrases it as:
  "HRX derives backend-private relocation metadata from |transaction|".
  Failure symptom documented in README §7: without a patch table the NPU writes to
  address 0 → **all-zero output** with no error.
- **DDR-fold ABI (load-bearing for artifact generation)**: the `insts.bin` consumed by
  HRX must be compiled **unfolded** — `aiecc --fold-ddr-addr-offset=false` — so each DDR
  patch's `arg_plus` carries only the raw intra-buffer offset; **libhrx adds the AIE DDR
  aperture offset (0x80000000) itself, exactly once per argument**, independent of the
  firmware's first-5-args translation cutoff (module docstring,
  `python/utils/hostruntime/hrxruntime/hostruntime.py`). This is what lets designs with
  more than five host buffers dispatch correctly. The switch is resolved from the tensor
  backend's class attribute `HRXTensor.FOLDS_DDR_ADDR_OFFSET = False`
  (`python/utils/hostruntime/hrxruntime/tensor.py`, line 41) via
  `aie.utils.npu_runtime_folds_ddr_addr_offset()`, and is part of the JIT cache key
  (`python/utils/compile/jit/compilabledesign.py::_resolve_fold_ddr_addr_offset`), so
  XRT-folded and HRX-unfolded artifacts never alias in the cache.
  **Consequence for `virtio-accel-amdxdna`: an `insts.bin` compiled for the XRT path is
  not interchangeable with one compiled for HRX; the artifact must be built with the HRX
  fold ABI.**
- **Executable description passed to `hrx_amdxdna_executable_create`** (verbatim structs
  in §3): `create_params` (v0) carries N xclbins (`hrx_const_byte_span_t`) + N entry
  points; each entry point names a kernel (e.g. `"MLIR_AIE"`), selects
  `context_mode` (`CREATE` selects a PDI from an xclbin; `REUSE` dispatches against a
  context established by another entry point), `xclbin_ordinal`, `pdi_ordinal`, and a
  list of runs; each run carries the `transaction` span plus an optional
  `data_payload` ("reconfiguration data"). Every record starts with
  `(record_length, abi_version)` so libhrx can stride/validate; **ABI version 0 is the
  only version defined** in the pinned release. A sibling entry point,
  `hrx_amdxdna_xadx_serialize`, serializes the same description to the XADX container
  for tooling/caching; runtime users are told to use `hrx_amdxdna_executable_create`,
  which keeps the serialized form internal.

## 3. The HRX C ABI surface

Authoritative source: headers inside the pinned release asset
(`include/hrx/hrx_runtime.h`, 1130 lines; `include/hrx/hrx_amdxdna.h`, 114 lines;
`lib/libhrx.so.0.1.0`; `HRX_VERSION_MAJOR/MINOR/PATCH = 0/1/0`). The fork's ctypes layer
is `python/utils/hostruntime/hrxruntime/_bindings.py`; the C++ reference is
`runtime_lib/test_lib/hrx_test_wrapper.h`. Enum/flag values match IREE HAL counterparts
("Verified by _Static_assert in implementation", `hrx_runtime.h`).

### 3.1 Status convention

```c
typedef struct hrx_status_s* hrx_status_t;      // opaque; NULL == OK
HRX_API hrx_status_code_t hrx_status_code(hrx_status_t status);
HRX_API hrx_status_t hrx_status_to_string(hrx_status_t, char** out_message, size_t* out_length);
HRX_API void hrx_status_free_message(char* message);
HRX_API void hrx_status_ignore(hrx_status_t status);
```

Codes mirror `iree_status_code_t` (`HRX_STATUS_OK=0` … `HRX_STATUS_DATA_LOSS=15`,
including `DEADLINE_EXCEEDED=4`, `OUT_OF_MEMORY=8`, `INTERNAL=13`, `UNAVAILABLE=14`).
**Ownership**: a non-NULL status is owned by the caller and must be consumed —
both the fork's Python `_check()` and C++ `hrx_check()` call `hrx_status_to_string`,
free the message with `hrx_status_free_message`, then `hrx_status_ignore` **both** the
returned string-status and the original status. Failing to do this leaks.

### 3.2 Device/context lifecycle

```c
HRX_API hrx_status_t hrx_gpu_initialize(uint32_t flags);
HRX_API hrx_status_t hrx_gpu_shutdown(void);
HRX_API hrx_status_t hrx_gpu_device_count(int* count);
HRX_API hrx_status_t hrx_gpu_device_get(int index, hrx_device_t* device);
HRX_API void hrx_device_retain(hrx_device_t);   // refcounted
HRX_API void hrx_device_release(hrx_device_t);
HRX_API hrx_status_t hrx_stream_create(hrx_device_t device, uint32_t flags, hrx_stream_t* stream);
HRX_API void hrx_stream_retain(hrx_stream_t);
HRX_API void hrx_stream_release(hrx_stream_t);
```

The NPU appears under the **"gpu" accelerator namespace** (the amdxdna HAL); the exact
init sequence both the Python and C++ layers use is
`hrx_gpu_initialize(0)` → `hrx_gpu_device_get(0, &device)` → `hrx_stream_create(device, 0, &stream)`
(`context.py::HRXContext.__init__`; `hrx_test_wrapper.h::Context`).
**Teardown**: neither the fork's Python singleton nor the C++ `Context` ever calls
`hrx_gpu_shutdown` — the context lives for the process. `hrx_device_synchronize` is
explicitly deprecated ("devices do not own a single implicit timeline").

### 3.3 Buffers

```c
// Stream-ordered allocation (convenience over allocator).
HRX_API hrx_status_t hrx_buffer_allocate(hrx_stream_t stream, size_t size,
                                         hrx_memory_type_t mem_type,
                                         hrx_buffer_usage_t usage,
                                         hrx_buffer_t* buffer);
HRX_API void hrx_buffer_retain(hrx_buffer_t);
HRX_API void hrx_buffer_release(hrx_buffer_t);
HRX_API hrx_status_t hrx_buffer_map_with_mode(hrx_buffer_t buffer,
                                              hrx_mapping_mode_t mapping_mode,   // u32
                                              hrx_map_flags_t flags,             // u16
                                              size_t offset, size_t size,
                                              void** mapped_ptr);
HRX_API hrx_status_t hrx_buffer_unmap(hrx_buffer_t buffer);
HRX_API hrx_status_t hrx_buffer_flush_range(hrx_buffer_t buffer, size_t offset, size_t size);
HRX_API hrx_status_t hrx_buffer_invalidate_range(hrx_buffer_t buffer, size_t offset, size_t size);
```

- The canonical allocation both runtimes use for program-visible I/O:
  `mem_type = HRX_MEMORY_TYPE_HOST_LOCAL (0x46) | HRX_MEMORY_TYPE_DEVICE_VISIBLE (0x10)`,
  `usage = HRX_BUFFER_USAGE_DEFAULT (0x0C03) | HRX_BUFFER_USAGE_MAPPING_PERSISTENT (0x02000000)`,
  then `hrx_buffer_map_with_mode(buf, HRX_MAPPING_MODE_PERSISTENT (0x2),
  HRX_MAP_READ|HRX_MAP_WRITE (0x3), 0, size, &ptr)`
  (`context.py::allocate_persistent`; `hrx_test_wrapper.h::Buffer`).
- **Mapping rules** (header): only one mapping may be active per buffer at a time;
  persistent mappings remain valid until `hrx_buffer_unmap` or buffer release, and
  require `HRX_BUFFER_USAGE_MAPPING_PERSISTENT`.
- **Coherence is explicit**: `flush_range` pushes host writes device-ward,
  `invalidate_range` makes device writes visible to host reads; both are documented as
  "cheap (no copy); the buffer must be currently mapped". The C++ wrapper flushes every
  input after initialization and invalidates the output after every dispatch+sync.
- **Zero-size**: HRX rejects 0-size allocations (`hrx_test_wrapper.h` allocates
  `max(nbytes,1)`).
- **Ownership**: buffers are refcounted; `hrx_buffer_release` drops the reference. The
  fork releases exactly once per allocation and never unmaps persistent mappings first.
- Also available (unused by the fork's dispatch path, useful for `virtio-accel`
  transfers): `hrx_synchronous_h2d/d2h`, `hrx_stream_copy_h2d/d2h`,
  `hrx_stream_fill_buffer/copy_buffer/update_buffer`, `hrx_allocator_import_buffer`
  (imports an external host pointer), `hrx_buffer_get_device_ptr`, `hrx_buffer_get_size`.

### 3.4 Executables

```c
HRX_API hrx_status_t hrx_amdxdna_executable_create(
    hrx_device_t device, const hrx_amdxdna_executable_create_params_t* params,
    hrx_executable_t* executable);
HRX_API hrx_status_t hrx_executable_lookup_export_by_name(
    hrx_executable_t executable, const char* name, uint32_t* export_ordinal);
HRX_API void hrx_executable_retain(hrx_executable_t);
HRX_API void hrx_executable_release(hrx_executable_t);
```

with (from `hrx_amdxdna.h`, verbatim):

```c
typedef struct hrx_amdxdna_executable_run_t {
  uint32_t record_length;                 // = sizeof(run)
  uint32_t abi_version;                   // HRX_AMDXDNA_EXECUTABLE_RUN_ABI_VERSION_0
  hrx_const_byte_span_t transaction;      // XAie TXN (insts.bin verbatim)
  hrx_const_byte_span_t data_payload;     // optional reconfiguration data
} hrx_amdxdna_executable_run_t;

typedef struct hrx_amdxdna_executable_entry_point_t {
  uint32_t record_length;
  uint32_t abi_version;
  hrx_string_view_t name;                 // kernel/export name, e.g. "MLIR_AIE"
  hrx_amdxdna_context_mode_t context_mode; // CREATE=0 selects a PDI; REUSE=1
  uint32_t xclbin_ordinal;
  uint32_t pdi_ordinal;
  uint32_t source_line;
  hrx_string_view_t source_file;
  const hrx_amdxdna_executable_run_t* runs;
  size_t run_count;
} hrx_amdxdna_executable_entry_point_t;

typedef struct hrx_amdxdna_executable_create_params_t {
  uint32_t record_length;
  uint32_t abi_version;
  uint32_t flags;
  uint32_t reserved;
  const hrx_const_byte_span_t* xclbins;
  size_t xclbin_count;
  const hrx_amdxdna_executable_entry_point_t* entry_points;
  size_t entry_point_count;
} hrx_amdxdna_executable_create_params_t;
```

**Ownership**: "All input storage is borrowed for the duration of the call and may be
released after it returns" (`hrx_amdxdna.h`). The returned executable is refcounted
(`retain`/`release`); the fork's `HRXKernelHandle` takes an extra retain so an LRU cache
eviction cannot drop the last reference under a live handle
(`hostruntime.py::HRXKernelHandle`).
`hrx_executable_export_info` exposes per-export metadata (`binding_count`,
`constant_byte_length`, …) — useful for validating a slot plan at program admission.

### 3.5 Dispatch and synchronization

```c
HRX_API hrx_status_t hrx_stream_dispatch(
    hrx_stream_t stream, hrx_executable_t executable, uint32_t export_ordinal,
    const hrx_dispatch_config_t* config, const void* constants,
    size_t constants_size, const hrx_buffer_ref_t* bindings,
    size_t binding_count, uint32_t flags);
HRX_API hrx_status_t hrx_stream_synchronize(hrx_stream_t stream);   // flush + block
HRX_API hrx_status_t hrx_stream_query(hrx_stream_t stream, bool* complete);
HRX_API hrx_status_t hrx_stream_flush(hrx_stream_t stream);
HRX_API hrx_status_t hrx_stream_wait(hrx_stream_t stream);          // wait w/o flushing pending work

typedef struct hrx_dispatch_config_t {
  uint32_t workgroup_count[3];  // amdxdna path: {1,1,1}
  uint32_t workgroup_size[3];   // {1,1,1}
  uint32_t subgroup_size;       // 0
} hrx_dispatch_config_t;

typedef struct hrx_buffer_ref_t {
  hrx_buffer_t buffer;
  size_t offset;
  size_t length;
} hrx_buffer_ref_t;
```

- `hrx_stream_dispatch` **records** into the stream's pending command buffer;
  `hrx_stream_synchronize` submits and blocks until completion. Recording several
  dispatches then synchronizing once submits the whole batch as one execution — the
  amdxdna HAL lowers a multi-dispatch command buffer into **one `ERT_CMD_CHAIN`**, with
  an execution + memory barrier between dispatches, so producer→consumer chains observe
  earlier device writes (README §5c/§6b; `context.py::dispatch_chain`).
- Binding rules: `bindings[i]` maps to DDR-patch argument index `i` — **binding order is
  the argument order** (IRON convention: data buffers in declared order, output last).
  `hrx_stream_dispatch` copies the binding refs into the recorded command (a stack-local
  refs array is safe; `hrx_test_wrapper.h::dispatch_chain` comment). Constants are unused
  on this path (`NULL, 0`).
- **Nonblocking completion probes exist in the ABI** (`hrx_stream_query`,
  `hrx_stream_get_semaphore` + `hrx_semaphore_query/wait(timeout_ns)`, fences, events)
  but the fork's runtime uses only dispatch + blocking synchronize; none of the probes
  are exercised by the fork's code, so treat them as **unvalidated** on amdxdna until
  proven on hardware. (Also note `hrx_runtime.h` line 687: a `// TODO(hrx): Stubs —
  declared for streaming rebase, not yet implemented.` comment sits immediately above
  the dispatch-config/queue-dispatch block; `hrx_stream_dispatch` itself demonstrably
  works — it is the fork's whole dispatch path — but nearby declarations such as
  `hrx_queue_dispatch`, `hrx_queue_host_call`, and `hrx_stream_execution_barrier` may be
  stubs. Verify any of them before use.)
- **Cancellation does not exist**: the only completion primitive used is blocking
  `hrx_stream_synchronize`; the fork's `IRON_HRX_TIMEOUT` watchdog explicitly documents
  that the underlying sync "cannot be cancelled" and a wedged dispatch requires driver
  reload (`sudo rmmod amdxdna && sudo modprobe amdxdna`, README §8). This confirms
  issue #75's guidance: do not advertise `EVENT_CANCELLATION`.

### 3.6 Threading rules

From `programming_guide/hrx_runtime.md` §"Concurrency and multi-tenancy" and
`context.py` docstrings (the release headers themselves contain **no** thread-safety
statements — the fork's documentation is the only primary source):

- **Processes are fully isolated** (each builds its own context/buffers; the amdxdna
  kernel driver isolates per-process hardware contexts and memory). The only shared
  resource is the finite system-wide pool of amdxdna hardware contexts — exhaustion is a
  capacity failure, not a data-safety issue.
- **One context per process by design**; the fork's `HRXContext` is a thread-safe lazy
  singleton (double-checked locking) and `libhrx` binding is guarded the same way.
- **The dispatch stream is not safe for concurrent dispatch**: recording
  `dispatch`/`dispatch_chain`/`synchronize` from several threads interleaves into one
  pending command buffer. "Callers must serialize dispatch on a single context."
  This matches issue #75's serialized-worker design.
- Device-event sink callbacks (`hrx_runtime_set_device_event_sink`) "may run on
  driver-owned service, callback, or completion threads" and must not call back into the
  device or submit/wait/destroy from the callback (header comment). The sink must be set
  before any accelerator is initialized.

## 4. Pinned versions and the Fedora 44 install route

### 4.1 The pins (all verified to exist)

**Key structural fact (verified twice via the GitHub compare API):** the fork is
currently **byte-identical to upstream `Xilinx/mlir-aie` `main`** (`ahead_by: 0,
behind_by: 0, status: identical`). The whole HRX path — `hrxruntime`,
`utils/hrx-release.env`, `hrx-xclbinutil` wiring — is upstreamed, and is already
contained in upstream's latest stable release **v1.4.1** (tag commit
`601fc859532f2539bebb33ac89139584c76ae8a2`, published 2026-08-11; its
`hrx-release.env`, `peano-requirements.txt`, and `clone-llvm.sh` are byte-identical to
`main`). This means upstream's released `mlir_aie` wheels can serve as the pinned
toolchain until the fork actually diverges.

| Component | Pin | Source of pin |
|---|---|---|
| Fork (`amd-npu-compiler`) | branch `main`, commit `c95544269f0c074d6d3e213ee43cc34dc4100801` (≡ upstream main; tagged anchor: upstream `v1.4.1` = `601fc859…`) | HEAD at research time; the fork has no `v*` tags (only the rolling `mlir-distro` tag) |
| HRX (`libhrx`) | repo `jtuyls/hrx`, tag `flm-hrx-amdxdna-v2026.07.30`, asset `hrx-amdxdna-2026.07.30-amdxdna-hal-native-rel-eb0b39f-linux-x86_64.tar.zst`, SHA-256 `661ed94051cc6ad04f53739b2df7a791aecb658bc435bd5a6ff3c46716696345` | `utils/hrx-release.env` (quoted verbatim below) |
| HRX ABI | `libhrx.so.0.1.0` (`HRX_VERSION 0.1.0`); amdxdna executable records ABI version 0 (the only defined version); API generation `amdxdna-hal-native-rel`, HRX build `eb0b39f` | release asset contents + `hrx_amdxdna.h` |
| Peano (`llvm-aie`) | pip `llvm-aie==21.0.0.2026080301+c9c5ecb7` from `https://github.com/Xilinx/llvm-aie/releases/expanded_assets/nightly` | `utils/peano-requirements.txt` |
| `hrx-xclbinutil` | git submodule `third_party/hrx-xclbinutil` → `jtuyls/hrx-xclbinutil` @ `3940dd23bda7a941df9f2d10015fd66c9410b7af` | `.gitmodules` + submodule gitlink at `c955442` |
| MLIR distro (build dep) | fork release tag `mlir-distro`, wheels `mlir[-no-rtti]-24.0.0.2026081506+dcf8c75e` (manylinux_2_27/2_28 x86_64), `mlir_native_tools-24.0.0.2026081506+dcf8c75e` | fork releases API. Note: `utils/clone-llvm.sh` at `c955442` pins `LLVM_PROJECT_COMMIT=56bcc1871734e6c375a254dec0ec74eb18d04a2e` / `DATETIME=2026080106`; the published `mlir-distro` wheels (2026-08-15) are one LLVM bump ahead of the checked-in pin — pin the *wheel filenames* above, not the script output |
| `mlir_aie` (toolchain wheel) | upstream `mlir_aie==1.4.1`, `cp311`–`cp314`, `manylinux_2_35_x86_64` (e.g. `mlir_aie-1.4.1-cp312-cp312-manylinux_2_35_x86_64.whl`); **contains the `hrxruntime` package** | upstream v1.4.1 release assets (verified); not on PyPI — install via `-f https://github.com/Xilinx/mlir-aie/releases/expanded_assets/v1.4.1` |
| Python | 3.11–3.14 per the wheel tags and `docs/getting-started.md`; **`utils/env_install.sh` (the scripted route) hard-requires exactly 3.12** | wheel filenames; `utils/env_install.sh` |
| eudsl | `eudsl-python-extras==0.1.0.20260801.905+68a0d7a` (find-links `https://llvm.github.io/eudsl`) | `python/requirements.txt` |
| numpy | `>=2.5.1,<3.0` on Python ≥3.12 | `python/requirements.txt` |
| CMake | ≥ 3.30 for the C++ HRX host tests (`pip install "cmake>=3.30"` acceptable); ≥ 3.20 for standalone `hrx-xclbinutil` | HRX runbook §1/§3b |

`utils/hrx-release.env` verbatim (the single source of truth the fetch helper reads;
any of `HRX_RELEASE_{REPO,TAG,ASSET,SHA256}` may be overridden from the environment):

```bash
HRX_RELEASE_REPO="jtuyls/hrx"
HRX_RELEASE_TAG="flm-hrx-amdxdna-v2026.07.30"
HRX_RELEASE_ASSET="hrx-amdxdna-2026.07.30-amdxdna-hal-native-rel-eb0b39f-linux-x86_64.tar.zst"
HRX_RELEASE_SHA256="661ed94051cc6ad04f53739b2df7a791aecb658bc435bd5a6ff3c46716696345"
```

The Linux asset is a **relocatable install prefix**: `include/hrx/{hrx_runtime.h,
hrx_amdxdna.h, hrx_runtime_cxx.h}` + `lib/libhrx.so -> libhrx.so.0 -> libhrx.so.0.1.0`
(1.6 MB) + `lib/cmake/hrx/` (a proper `find_package(hrx CONFIG)` package exporting
`hrx::hrx`) + `LICENSES/`. ~636 KB compressed. **libhrx needs no source build.**

### 4.2 Install route on Fedora 44 (self-contained prefix)

What is prebuilt vs. built from source:

- **Prebuilt**: `libhrx` (release asset above); Peano (`llvm-aie` pip wheel); the
  `mlir_aie` toolchain itself — because fork == upstream, upstream's
  `mlir_aie==1.4.1` wheels apply and **contain the `hrxruntime` package** (verified
  present at `refs/tags/v1.4.1`). The runbook's "two-trees caveat" (§3) is about
  mixing a wheel with edits to a *different* source tree, not about wheel viability.
- **Must be built from source regardless of route**: `hrx-xclbinutil`. It is not in
  any release asset, and the wheel workflow (`.github/workflows/buildRyzenWheels.yml`)
  never sets `-DAIE_BUILD_HRXXCLBINUTIL=ON`, so wheels don't contain it either. Build
  it standalone from the submodule (C++17 + CMake ≥ 3.20) and point `AIE_XCLBINUTIL`
  at it.
- **Source build of `mlir_aie`** is needed only once the fork carries local patches:
  `utils/env_install.sh --dev` + `utils/build-mlir-aie-from-wheels.sh` (links against
  the `mlir`/`mlir-distro` wheels, manylinux_2_27/2_28 — fine on Fedora), with
  `EXTRA_CMAKE_ARGS="-DAIE_BUILD_HRXXCLBINUTIL=ON"` folding the xclbinutil build in.
  Note: wheels built from the fork get version `0.0.0.devN` because the fork has no
  `v*` tags (`utils/mlir_aie_wheels/_version_helper.py` uses git-describe).

Recommended wheel-based sequence (composed from `docs/getting-started.md`,
the HRX runbook §§2–4, and `docs/buildHostLinNonUbuntu.md`; **not yet executed on
Fedora 44** — see §8):

```bash
# 0. Host deps (Fedora): python3.12 (matches the scripted route's requirement),
#    zstd (fetch-hrx-release extraction), C++17 toolchain, ninja, git.
sudo dnf install python3.12 zstd gcc-c++ ninja-build git

git clone https://github.com/MicroPerceptron/amd-npu-compiler && cd amd-npu-compiler
git checkout c95544269f0c074d6d3e213ee43cc34dc4100801     # == upstream main == v1.4.1+3d
git submodule update --init third_party/hrx-xclbinutil

python3.12 -m venv ironenv && source ironenv/bin/activate
pip install "cmake>=3.30" ninja

# 1. Toolchain wheels (pin exact versions; mlir_aie wheels are on GitHub, not PyPI)
pip install mlir_aie==1.4.1 -f https://github.com/Xilinx/mlir-aie/releases/expanded_assets/v1.4.1
pip install -r utils/peano-requirements.txt        # llvm-aie==21.0.0.2026080301+c9c5ecb7
pip install -r python/requirements.txt

# 2. Provision libhrx (downloads + sha256-verifies + extracts the pinned release
#    into third_party/.hrx-release; env.sh sets HRX_DIR / LD_LIBRARY_PATH / CMAKE_PREFIX_PATH)
source "$(utils/fetch-hrx-release.sh)"

# 3. XRT-free xclbinutil (standalone build from the pinned submodule)
cmake -G Ninja -B bld-xclbinutil -S third_party/hrx-xclbinutil -DCMAKE_BUILD_TYPE=Release
cmake --build bld-xclbinutil --target hrx-xclbinutil
export AIE_XCLBINUTIL="$PWD/bld-xclbinutil/tools/hrx-xclbinutil"

# 4. Verify (path-only probe; no dlopen, no device touch)
python3 -c "import aie.utils as u; print('has_hrx:', u.has_hrx)"     # -> True
NPU_RUNTIME=hrx python3 -c "import aie.utils as u; print(u.DEFAULT_TENSOR_CLASS.__name__)"  # -> HRXTensor
```

Fedora / non-Ubuntu caveats (from `docs/buildHostLinNonUbuntu.md`, which is explicitly
community-contributed/experimental — its worked example is Void Linux; **Fedora 44
specifically is untested and undocumented**):

- With an in-tree `amdxdna` (Linux ≥ 6.14) do **not** install the DKMS/out-of-tree
  module on top — "a newer out-of-tree module can require newer firmware and break a
  working in-tree setup" (citing `amd/xdna-driver` issues #1074, #1219).
- Device permissions: udev rule
  `SUBSYSTEM=="accel", KERNEL=="accel*", GROUP="render", MODE="0660"` plus memlock
  limits.
- A generic `llvm-objcopy` must be on `PATH` — GNU objcopy rejects the AIE ELF machine
  type (`EM_AIE`, 0x108).
- `utils/env_setup.sh` errors if `xrt-smi` is absent (it probes the NPU with XRT's
  tool). For a pure-HRX box, export `NPU2=1` and set
  `MLIR_AIE_INSTALL_DIR`/`PEANO_INSTALL_DIR`/`PATH`/`PYTHONPATH`/`LD_LIBRARY_PATH`
  manually (the script's first half); the HRX runtime itself never needs `xrt-smi`.
- `aiecc` picks its `xclbinutil` from `AIE_XCLBINUTIL` (env) or `--xclbinutil-path`
  (flag), failing **loudly** if set-but-missing, so an HRX build never silently uses
  XRT's tool; `programming_examples/makefile-common` sets it to `aiecc`'s sibling
  automatically under `NPU_RUNTIME=hrx`. Bare `PATH` fallback is discouraged.
- The rolling `mlir-distro` tag churns (a `pruneReleaseAssets.yml` workflow deletes old
  assets) — pin **wheel version strings**, never the tag.
- For a Rust backend the Python environment is a **build/artifact-generation**
  dependency only; runtime needs are exactly `libhrx.so` + the two headers + the
  generated `final.xclbin`/`insts.bin`.

## 5. IRON JIT artifact generation and the vector-add example

### 5.1 JIT pipeline

Sources: `python/utils/compile/jit/compilabledesign.py`, `python/utils/compile/__init__.py`,
`python/iron/compile/__init__.py` (re-export shim), `programming_examples/makefile-common`.

- `@iron.jit` wraps a design generator into a `CompilableDesign`; `compile()` produces
  the `xclbin` + `insts.bin` pair by invoking `aiecc` (via
  `aie.utils.compile.compile_mlir_module`), auto-building any `aie.iron.kernels`
  ExternalFunctions with Peano.
- **Cache**: file-system cache rooted at `NPU_CACHE_HOME` (env; default
  `~/.npu/cache`, `python/utils/compile/__init__.py` line 19). The 24-hex cache key
  composes `recipe_hash` (generator identity + compile kwargs + aiecc/compile flags)
  and `artifact_hash` (source/object mtimes + tool mtimes + target device), and
  **includes the DDR-fold ABI bit**, so HRX and XRT artifacts never collide. Concurrent
  compiles are serialized by a file lock (timeout 1800 s).
- **Makefile integration** (`programming_examples/makefile-common`): the `jit_xclbin`
  / `jit_xclbin_elf` templates drive a design's `compile()` with
  `--xclbin-path=… --insts-path=…` (optionally `--elf-path=…`), writing artifacts
  straight into the example's `build/` dir; `build_host_exe` cmake-wraps the host
  binary, adding `-DUSE_HRX=ON` when `NPU_RUNTIME=hrx`. Peano flags for XDNA2:
  `--target=aie2p-none-unknown-elf` (`PEANOWRAP2P_FLAGS`).
- Device targeting: per-example `devicename ?= $(if $(filter 1,$(NPU2)),npu2,npu)`;
  designs receive `-d npu|npu2`.

### 5.2 Vector-add end to end

Source dir: `programming_examples/basic/vector_scalar_add/` (files: `Makefile`,
`vector_scalar_add.py`, `test.cpp`, `test_runlist.cpp`, `test_runlist_hrx.cpp`,
`CMakeLists.txt`, lit runners including `run_strix_makefile_runlist.lit`).

- **Design** (`vector_scalar_add.py`): `@iron.jit`-decorated generator
  (`inp: In, out: Out`, `CompileTime` params `problem_size=1024`,
  `aie_tile_width=32`) delegating to `aie.iron.algorithms.transform(lambda x: x + 1, …)`.
  Four invocation modes: standalone run; compile-only (`--xclbin-path/--insts-path`,
  used by the Makefile); **AOT export** (`--aot-dir` → named
  `vector_scalar_add.{xclbin,insts.bin,pdi,insts.elf}`, cache bypassed); and
  **bring-your-own** (`--from-xclbin/--from-insts` runs any prebuilt pair via
  `aie.utils.NPUKernel`, with `--dev` guarding artifact/hardware family mismatch —
  "a mismatched xclbin typically hangs or times out").
  The AOT + BYO modes are exactly the shape `virtio-accel-amdxdna` needs for its first
  precompiled artifact: generate once with `--aot-dir` under `NPU_RUNTIME=hrx`
  (unfolded ABI), ship the pair, load natively.
- **Build**: `make all` → `build/final.xclbin` + `build/insts.bin` (+ `build/insts.elf`).
- **Native HRX host** (`make run_runlist_hrx`, target self-selects `-DUSE_HRX=ON`):
  `test_runlist_hrx.cpp` + `hrx_test_wrapper.h` execute, in order:
  1. `Context::get()` → `hrx_gpu_initialize(0)`, `hrx_gpu_device_get(0,…)`,
     `hrx_stream_create(dev,0,…)` (once per process);
  2. `load_kernel(xclbin, insts, "MLIR_AIE")` → read files; if `insts` is an ELF,
     reduce to `.ctrltext`; fill the v0 run/entry-point/create-params records;
     `hrx_amdxdna_executable_create` → `hrx_executable_lookup_export_by_name`;
  3. per buffer: `hrx_buffer_allocate(stream, n, HOST_LOCAL|DEVICE_VISIBLE,
     DEFAULT|MAPPING_PERSISTENT)` → `hrx_buffer_map_with_mode(PERSISTENT, READ|WRITE)`
     → initialize through the mapping → `hrx_buffer_flush_range(0, n)`;
  4. `hrx_stream_dispatch(stream, exe, ordinal, {1,1,1}/{1,1,1}/0, NULL, 0,
     refs=[in…, out], n, HRX_DISPATCH_FLAG_NONE)` — once per chain link —
     then a single `hrx_stream_synchronize(stream)`;
  5. `hrx_buffer_invalidate_range(out, 0, n)` → verify through the mapping →
     `hrx_buffer_release` each buffer (destructors), executable release via handle.
  The single-dispatch variant (`test.cpp` under `NPU_RUNTIME=hrx make run`) is the same
  sequence with one dispatch; `xrt_test_wrapper.h` pulls in `hrx_test_wrapper.h` when
  `TEST_UTILS_USE_HRX` is defined, so example sources are unchanged.
- **Python self-running JIT example**: `programming_examples/basic/vector_vector_add`
  (`NPU_RUNTIME=hrx python3 vector_vector_add.py` → `PASS!`). Designs the runbook lists
  as known-passing on HRX: `vector_scalar_mul`, `vector_vector_add`,
  `vector_scalar_add`, `vector_reduce_add`, `passthrough_dmas`, `vector_reduce_max`.
  A hardware-dependent pytest for chained dispatch: `test/python/npu-hrx/test_chain_hrx.py`.
- **HRX feature limits** (hard errors, not warnings): trace capture and control packets
  are **not supported** on HRX (`hrx_test_wrapper.h::reject_unsupported_features`);
  trace/ctrl-pkt designs must use XRT.

## 6. Device support: 1022:17f0 rev 20, driver and firmware

### 6.1 What the fork supports (confirmed from fork source)

- The HRX runbook's prerequisites (§1): "**XDNA2 NPU** (Strix / `npu4` / aie2p),
  `/dev/accel/accel0` present; `amdxdna` driver loaded." The whole HRX path is
  XDNA2-oriented; Phoenix/XDNA1 (`npu1`) exists in the IRON device model but the HRX
  runbook targets XDNA2.
- Userspace PCI-ID recognition (`python/utils/hostruntime/hrxruntime/hostruntime.py`,
  lines 45–47):

  ```python
  _AMD_PCI_VENDOR = "0x1022"
  _PHOENIX_PCI_IDS = {"0x1502"}  # Phoenix -> npu1
  _STRIX_PCI_IDS = {"0x17f0", "0x17f1", "0x1640", "0x1641"}  # Strix/Krackan -> npu2
  ```

  Detection order: `IRON_HRX_DEVICE` env override → sysfs probe of
  `/sys/bus/pci/drivers/amdxdna/*/device` (falling back to scanning all AMD PCI
  devices) → default `npu2`. **`0x17f0` is explicitly recognized; the PCI revision is
  not consulted by the fork's userspace at all** — revision-specific handling lives in
  the kernel driver + firmware selection.
- `utils/env_setup.sh` groups `NPU Strix|NPU Strix Halo|NPU Krackan|RyzenAI-npu[456]`
  as NPU2, with the comment `npu4 => Strix, npu5 => Strix Halo, npu6 => Krackan`.
- Compiler-side XDNA2 support is the `aie2p` target (`PEANOWRAP2P_FLAGS
  --target=aie2p-none-unknown-elf`, `CHESSCCWRAP2P_FLAGS`), and lit runners exist for
  Strix (`run_strix*.lit` in the vector-add example).

Also confirmed from the fork's docs (`docs/Devices.md`): `aie.device(npu2)` "is present
in Ryzen AI: Strix, Strix Halo and Krackan Point SOCs. 8 Columns and 6 Rows" (partition
variants `npu2_1col`…`npu2_7col`), and the wheel build enables
`AIE_VITIS_COMPONENTS='AIE2;AIE2P'`.

**Verdict for device `1022:17f0` rev 0x20**: supported. The revision maps to
**NPU6 / Krackan Point** in the kernel driver (see §6.2) — an XDNA2/NPU4-class part
inside the fork's declared `npu2` envelope, which is the exact device class the HRX
runtime was built for. The revision byte selects the silicon variant at the
kernel/firmware level only; HRX userspace never reads it.

### 6.2 Kernel driver and firmware constraints

Confirmed from the fork:

- HRX opens the amdxdna accel device (`/dev/accel/accel0`) through its amdxdna HAL —
  no XRT userspace involved (runbook; asset name `amdxdna-hal-native`; chained
  dispatch lowered by the HAL "into one `ERT_CMD_CHAIN`"). It drives
  `/dev/accel/accelN` via amdxdna ioctls directly. The recovery procedure for a wedged
  dispatch is a driver reload (`rmmod amdxdna && modprobe amdxdna`, runbook §8).
- The amdxdna hardware-context pool is finite and system-wide; exhaustion surfaces as
  context/stream creation failure (`hrx_runtime.md`).

From Linux kernel / linux-firmware primary sources (researched by a delegated agent
against `torvalds/linux`, `amd/xdna-driver`, and linux-firmware `WHENCE`):

- **Minimum kernel: v6.14** — `drivers/accel/amdxdna` merged there, and v6.14's
  `amdxdna_pci_drv.c` PCI table already includes `{0x17f0, rev 0x20, npu6}`. Full
  table: `1502/0x00 → npu1` (Phoenix), `17f0/0x00 → npu2`, `17f0/0x10 → npu4` (Strix),
  `17f0/0x11 → npu5` (Strix Halo), `17f0/0x20 → npu6` (Krackan Point; vbnv
  `"RyzenAI-npu6"` in-tree, `"NPU Krackan"` in `amd/xdna-driver`'s `npu6_regs.c`).
  npu6 reuses the npu4 device-info tables (`NPU4_COMMON_DEV_INFO`). The fork's Ubuntu
  docs recommend ≥ 6.17 for the packaged route; the reference machine's
  `7.1.8-200.fc44` is far past both.
- **Firmware: rev 0x20 shares the Strix directory** — `npu6_regs.c` sets `fw_path`
  under `amdnpu/17f0_10/`; there is **no** `17f0_20` firmware directory.
  linux-firmware ships `amdnpu/17f0_10/npu.sbin → npu.sbin.1.0.0.63` and
  `amdnpu/17f0_10/npu_7.sbin → npu.sbin.1.1.2.64`; per the fork's own docs, "On a 7.0
  kernel the in-tree driver loads `npu_7.sbin` (firmware protocol 7)" — so the
  reference machine (kernel 7.1.8) uses `amdnpu/17f0_10/npu_7.sbin`. Firmware is
  PSP-verified/signed (`Documentation/accel/amdxdna/amdnpu.rst`), which also names XRT
  as the *expected* userspace but imposes no userspace requirement.
- **No explicit driver↔firmware↔runtime version constraints are documented anywhere**
  (kernel docs, fork, HRX release). The only stated coupling is the runbook §8 note
  that `libhrx.so` must be of the `amdxdna-hal-native-rel` API generation (an older
  libhrx lacks `hrx_amdxdna_executable_create`). One documented hazard: never layer
  the out-of-tree DKMS module over a working in-tree setup — a newer out-of-tree
  module can require newer firmware and break it (`docs/buildHostLinNonUbuntu.md`,
  citing `amd/xdna-driver` issues #1074 and #1219).
- Remaining machine-local check for the hardware ticket: confirm the firmware version
  the driver actually loaded (`dmesg | grep -i xdna`; `ls /lib/firmware/amdnpu/`).

## 7. Implications for `virtio-accel-amdxdna` (summary)

1. Pin exactly: fork `c955442…` (≡ upstream v1.4.1 anchor `601fc859…`),
   `mlir_aie==1.4.1` wheels, HRX `flm-hrx-amdxdna-v2026.07.30` (sha-verified),
   Peano `llvm-aie==21.0.0.2026080301+c9c5ecb7`, `hrx-xclbinutil` submodule
   `3940dd23…`, Python 3.12, kernel ≥ 6.14 with in-tree `amdxdna`, firmware
   `amdnpu/17f0_10/npu.sbin` ≥ 1.0.0.63 (`npu_7.sbin` 1.1.2.64 on protocol-7
   kernels). Treat the fork commit + HRX tag as one compatibility unit (the fold-ABI
   and `hrx_amdxdna_executable_create` contract tie them together).
2. The Rust FFI boundary is small and stable: status helpers, `hrx_gpu_initialize/
   device_get`, `hrx_stream_create`, `hrx_buffer_allocate/map_with_mode/flush_range/
   invalidate_range/release`, `hrx_amdxdna_executable_create`,
   `hrx_executable_lookup_export_by_name/retain/release`, `hrx_stream_dispatch`,
   `hrx_stream_synchronize`. All confirmed present in `libhrx.so.0.1.0`.
3. Artifacts for HRX must be compiled with the unfolded DDR ABI; never reuse
   XRT-flow `insts.bin`.
4. Serialize all dispatch on one process-wide context; bridge blocking
   `hrx_stream_synchronize` to `poll_event` via a dedicated worker; no cancellation.
5. First artifact: generate with the vector-add design's `--aot-dir` mode under
   `NPU_RUNTIME=hrx`, load via the §5.2 native sequence.

## 8. Unknowns and follow-ups

1. **Fedora 44 execution is unvalidated.** The §4.2 sequence is composed from the
   fork's scripts/docs but has not been run on Fedora; the fork's supported docs are
   Ubuntu-oriented and its non-Ubuntu guide is community-contributed (Void Linux
   example). `utils/env_setup.sh`'s `xrt-smi` dependency needs the documented
   workaround on an XRT-free box.
2. **Krackan (rev 0x20) is not specifically exercised by the fork's CI.** The fork's
   hardware runners are `aie2p-8col`/`npu2` (almost certainly Strix, rev 0x10).
   Rev 0x20 sits inside the declared `npu2` envelope and the kernel driver supports it
   since v6.14, but no fork test result on Krackan silicon is recorded in the repo.
   First-hardware ticket should run `vector_scalar_add` before any backend code, and
   confirm the loaded firmware (`dmesg | grep -i xdna`, expecting
   `amdnpu/17f0_10/npu_7.sbin` on a protocol-7 kernel).
3. **Whether libhrx's amdxdna HAL has device-generation-specific paths** (npu4 vs npu5
   vs npu6) is not observable from the release binary and the HRX source was not
   audited.
4. **`hrx_stream_query` and semaphore/fence-based nonblocking completion are
   ABI-present but unexercised** by the fork; if the backend ever wants nonblocking
   polling below the worker thread, it must validate these on hardware first (note the
   "Stubs — declared for streaming rebase" comment near the dispatch block in
   `hrx_runtime.h`).
5. **libhrx source was not audited.** `jtuyls/hrx` release assets (all four releases
   are marked *prerelease*; the pinned one is the newest) and the fork's consumers
   were examined; the patch-table derivation is documented behavior, quoted from the
   fork's docs and header comments, not from libhrx source. The pinned Linux asset
   downloaded successfully with authenticated `gh`; anonymous downloadability was not
   separately proven.
6. **`data_payload` ("reconfiguration data") semantics** in
   `hrx_amdxdna_executable_run_t` are undocumented beyond the header comment; the fork
   always passes NULL/0.
7. **Divergence risk.** The fork is identical to upstream *today*; once it diverges,
   the upstream-v1.4.1 wheel route stops being valid for fork-local changes and the
   §4.2 source-build alternative becomes mandatory (fork-built wheels will version as
   `0.0.0.devN` until the fork adds `v*` tags).

## 9. Research method note

Fork files were fetched at the pinned commit via the GitHub contents API and read in
full (`hrx_runtime.md`, the HRX runbook, `_bindings.py`, `context.py`,
`hostruntime.py`, `tensor.py`, `discovery.py` (listed), `hrx_test_wrapper.h`,
`makefile-common`, `vector_scalar_add/{Makefile,vector_scalar_add.py}`,
`compilabledesign.py`, `utils/{hrx-release.env,fetch-hrx-release.sh,env_install.sh,
env_setup.sh,peano-requirements.txt,clone-llvm.sh}`, `.gitmodules`). The pinned HRX
Linux asset was downloaded, checksum-verified, and its headers read verbatim. Kernel
driver/firmware facts and the upstream-parity check were researched by a delegated
agent against `torvalds/linux`, `amd/xdna-driver`, linux-firmware, and the GitHub
compare API, with the parity check and v1.4.1 assets re-verified directly. One
caution for future researchers: **GitHub code search does not index forks** — a code
search for `17f0` in the fork returns zero hits even though
`python/utils/hostruntime/hrxruntime/hostruntime.py` line 47 contains it; read files
directly instead of trusting fork code search.
