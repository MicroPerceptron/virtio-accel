# virtio-accel-amdxdna: crate layout, HRX FFI boundary, and build-time probe

Design record for [issue #83](https://github.com/MicroPerceptron/virtio-accel/issues/83)
(wayfinder map [#78](https://github.com/MicroPerceptron/virtio-accel/issues/78), feeding
[#75](https://github.com/MicroPerceptron/virtio-accel/issues/75)). The scaffold ticket (#86)
implements this design; the FFI ticket (#87) fills `ffi.rs`/SAFETY.md; the compiler-helper
contract is ticket #84 and the execution-model/event-bridge design is ticket #85 — both are
*reserved for*, not decided, here. Grounding facts: the HRX ABI research
(`docs/research/hrx-runtime-contract.md`), the tier decision (ADR-0001), and the
`virtio-accel-openvino` / `virtio-accel-hexagon` precedents.

## 1. Module layout

OpenVINO's four modules plus one:

| Path | Compiles | Contents |
|---|---|---|
| `src/lib.rs` | always | `va_amdxdna` cfg gate; placeholder `AmdXdnaAccelerator` + `InitError::RuntimeUnavailable` on non-native builds (`forbid(unsafe_code)` there); `REQUIRED_RESIDENT_BYTES`; re-exports |
| `src/lower.rs` | always | Portable admission: the two `Target` consts (ADR-0001), target equality check, per-op/dtype surface sweep (BF16 tier incl. FP32-as-accumulator positions and IDENTITY-only FP16/FP32 graphs; integer tier), and emission of the compiler-helper input (exact form: #84). Unit-tests the whole admission surface on every host |
| `src/ffi.rs` | `va_amdxdna` | Hand-written HRX C ABI declarations only (§2); no calls, no ownership logic |
| `src/native.rs` | `va_amdxdna` | The `Accelerator` impl: init, buffers, load/dispatch through `ffi`; hosts the dispatch worker whose design lands with #85 |
| `src/compiler.rs` | `va_amdxdna` | Safe code only: the bounded aiecc **subprocess** boundary (spawn, pinned environment, timeout, output validation). Reserved module; contract is #84. Never a Cargo dependency (#75) |
| `SAFETY.md` | — | Outline in §4; audit text written by #87/#89 |
| `build.rs` | — | §3. `forbid(unsafe_code)` |
| `README.md`, `examples/tosa_amdxdna.rs`, `tests/` | — | Template-conformant; README claims only what tests demonstrate |

No cargo feature gates the native path — the build probe sets the cfg, exactly as in both
precedent backends.

## 2. HRX FFI surface (`ffi.rs`)

Only functions the fork's runtime demonstrably exercises on hardware, plus one introspection
call. All from `libhrx.so.0.1.0` (`flm-hrx-amdxdna-v2026.07.30`, API generation
`amdxdna-hal-native-rel`):

- **Status** (5): `hrx_status_code`, `hrx_status_to_string`, `hrx_status_free_message`,
  `hrx_status_ignore` — plus the opaque `hrx_status_t` (NULL == OK) convention.
- **Device/stream**: `hrx_gpu_initialize`, `hrx_gpu_device_count`, `hrx_gpu_device_get`,
  `hrx_device_retain/release`, `hrx_stream_create`, `hrx_stream_retain/release`.
- **Buffers**: `hrx_buffer_allocate`, `hrx_buffer_map_with_mode`, `hrx_buffer_flush_range`,
  `hrx_buffer_invalidate_range`, `hrx_buffer_retain/release`. Host I/O goes through
  persistent mappings + flush/invalidate — **no** `hrx_synchronous_h2d/d2h` or copy-engine
  declarations in v1.
- **Executables**: `hrx_amdxdna_executable_create`, `hrx_executable_lookup_export_by_name`,
  `hrx_executable_export_info` (admission-time slot-plan validation),
  `hrx_executable_retain/release`.
- **Dispatch**: `hrx_stream_dispatch`, `hrx_stream_synchronize`.
- **Excluded deliberately**: `hrx_stream_query`, semaphores/fences/events, queue-dispatch —
  ABI-present but unexercised by the fork and adjacent to a "Stubs — declared for streaming
  rebase" header comment. Added only if #85's hardware validation adopts one.

Types: opaque handles via the OpenVINO `opaque_handle!` pattern; `hrx_const_byte_span_t`,
`hrx_string_view_t`, `hrx_buffer_ref_t`, `hrx_dispatch_config_t`, and the three v0
`hrx_amdxdna_executable_*` record structs transcribed verbatim from `hrx_amdxdna.h`, with
`size_of`/`offset_of` static assertions pinning layout; enum/flag values as typed consts
(`HRX_MEMORY_TYPE_*`, `HRX_BUFFER_USAGE_*`, `HRX_MAPPING_MODE_*`, `HRX_MAP_*`,
`HRX_AMDXDNA_EXECUTABLE_*_ABI_VERSION_0`).

Linking: `cargo:rustc-link-lib=dylib=hrx`, **no rpath** — `libhrx.so` resolves via
`LD_LIBRARY_PATH`, which the pinned prefix's `env.sh` sets (`~/toolchains/amdxdna-hrx-v2026.08`).

## 3. Build-time probe (`build.rs`)

Pure Rust, no `cc`/CMake step: HRX is a plain C ABI, so unlike Hexagon there is no native
bridge to compile. The release's `lib/cmake/hrx` package is deliberately ignored.

| Variable | Meaning |
|---|---|
| `VIRTIO_ACCEL_AMDXDNA` | `1` = force native, fail loudly if the probe fails; `0` = force placeholder; unset = auto |
| `VIRTIO_ACCEL_HRX_DIR` | Explicit HRX install prefix (highest priority) |
| `HRX_DIR` | Fallback — the variable the pinned prefix `env.sh` and the fork's own tooling export (Hexagon's honor-the-vendor-variable precedent) |
| `VIRTIO_ACCEL_HRX_LIB_DIR` | Escape hatch: bare lib directory, skips completeness checks (OpenVINO precedent) |

Probe = file-set completeness at the resolved prefix: `include/hrx/hrx_runtime.h`,
`include/hrx/hrx_amdxdna.h`, `lib/libhrx.so`, **plus** a content check that
`hrx_amdxdna.h` contains `hrx_amdxdna_executable_create` — rejecting the documented
older-generation libhrx that lacks the amdxdna-native API before it becomes a link/runtime
mystery. **No standard-location scanning** (`~/hrx`, `/opt/hrx`, …): silently discovering an
unpinned libhrx defeats the pin. On success: link metadata + `cargo::rustc-cfg=va_amdxdna`;
`rustc-check-cfg` declared on every build. Device presence (`/dev/accel/accel0`) is a runtime
concern (`hrx_gpu_device_count` at init), never a build gate. The **compiler helper is not
probed by build.rs** — it is runtime configuration (#84).

Placeholder builds keep `lower.rs` + its admission tests compiling and running on every
workspace host (including the guest-mode CI gate), with `AmdXdnaAccelerator::new() ->
Err(InitError::RuntimeUnavailable)`.

## 4. SAFETY.md outline

Modeled on the Hexagon audit; sections 4 and 6 are filled by #85/#87/#89:

1. **Boundary statement** — safe admission (`lower.rs`) and safe subprocess (`compiler.rs`);
   unsafe confined to `ffi.rs` declarations and `native.rs` call sites.
2. **ABI and runtime lifetime** — refcounted handles with one native owner each; construction
   stores each handle before the next fallible step and releases the initialized prefix on
   error; every non-NULL `hrx_status_t` consumed exactly once (`to_string` →
   `free_message` → `ignore` both statuses — leak otherwise); process-lifetime context,
   `hrx_gpu_shutdown` never called (fork precedent); `hrx_device_synchronize` deprecated,
   never declared.
3. **Buffers and mappings** — one persistent mapping per buffer (`MAPPING_PERSISTENT` usage,
   `PERSISTENT` mode), valid until buffer release; mapped access bounded by the allocation;
   `flush_range` after every host write, `invalidate_range` before every host read; zero-size
   allocations rejected before FFI; all `hrx_amdxdna_executable_create` input storage
   (spans/records) borrowed only for the call.
4. **Concurrency and the dispatch worker** — all `dispatch`/`synchronize` serialized on one
   owner (the stream is not safe for concurrent dispatch); worker design + safety argument:
   #85. Device-event sink unused. No cancellation exists; wedged sync recovery is a driver
   reload, documented rather than papered over.
5. **Teardown ordering** — events terminal → executables released → buffers released
   (persistent mappings not separately unmapped, matching the fork) → stream → device.
6. **Audited unsafe operations** — per-block `SAFETY:` comments; native test inventory.

## 5. Explicitly deferred

- #84: compiler-helper subprocess contract (command line, environment pinning to the
  toolchain prefix, bounds, error mapping, artifact caching).
- #85: execution model, the blocking-sync worker, event states, timeout semantics.
- #86/#87: scaffold + FFI implementation and the SAFETY.md audit text.
