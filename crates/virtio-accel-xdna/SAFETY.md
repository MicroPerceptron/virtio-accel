# Unsafe-code audit

This crate is a host-native exception to the portable workspace's `forbid(unsafe_code)` rule, on
the same terms as `virtio-accel-openvino`. Unsafe Rust is confined to the `va_xdna` build
configuration — `src/ffi.rs` (HRX C ABI declarations only) and `src/native.rs` (the calls) — and
the crate root carries `cfg_attr(not(va_xdna), forbid(unsafe_code))`. The compiler-helper driver
(`src/compiler.rs`) is safe code — it spawns the aiecc helper as a subprocess (bounding the
helper's whole process group on timeout via `kill(1)`, not FFI) and therefore compiles on every
unix host, HRX or not, under the placeholder build's `forbid(unsafe_code)`.
Builds without a detected HRX runtime compile no `unsafe` at all: only the portable admission
surface (`src/lower.rs`), the portable precompiled-artifact codec (`src/artifact.rs`), and a
placeholder.

Scope: the device/stream owner, `hrx_buffer` primitives, the amdxdna executable lifecycle (for both
the precompiled and compiled paths), and the serialized dispatch worker.

## ABI and status ownership

`src/ffi.rs` declares the HRX C ABI transcribed from the pinned release headers
(`include/hrx/hrx_runtime.h`, `include/hrx/hrx_amdxdna.h`, `libhrx.so.0.1.0`, version 0.1.0):
opaque refcounted handles, the `uint32`/`uint16` bitmask constants, the `#[repr(C)]` span/config
and v0 executable records, and the `extern "C"` functions this crate calls. No struct layout is
invented beyond the header's own definitions; handles are opaque pointers.

Every fallible HRX call returns an `hrx_status_t` — an owned, opaque pointer that is `NULL` on
success and must be consumed exactly once on failure. The single `check` helper in `native.rs`
enforces this: on a non-NULL status it reads the code, renders the message with
`hrx_status_to_string` (freeing that message with `hrx_status_free_message`), then `hrx_status_ignore`s
both the string-status and the original status. No status escapes unconsumed, so none leaks.

## Handle lifetime and teardown

Each HRX handle has exactly one Rust owner with a `Drop`:

- The process-wide device is initialized once through a `OnceLock` and held in `SharedDevice`; per
  the fork's model it is never shut down. `SharedDevice` is `unsafe impl Send + Sync` because it is
  used only as a read-only factory for per-instance streams and executables, and the amdxdna HAL
  exposes the device as a process-wide singleton.
- Each `XdnaAccelerator` owns one stream (in `Stream`), one dispatch worker, and one watchdog.
  Ordinary teardown has no pending event: `Drop` stops and joins both threads before the lane (and
  thus `Stream::drop` → `hrx_stream_release`) runs. If an accepted event is still pending — the
  tier-2 wedge case or a caller discarding an active instance — `Drop` stops the watchdog but
  detaches the dispatch worker. Its `Arc<Lane>` retains the stream, and its queued `Job` retains the
  executable and every native buffer allocation, so no HRX handle can be released under a blocked
  C call. This is an intentional quarantine: a truly wedged worker may retain those resources until
  process exit, but dropping the poisoned backend itself never blocks.
- Each `XdnaProgram` is an `Arc<ProgramInner>` owning one executable reference, released once in
  `ProgramInner::drop`. A submission clones the `Arc` into the queued job, so an in-flight dispatch
  keeps the executable alive even if the caller unloads the program first.
- Each `XdnaBuffer` holds an `Arc<BufferInner>`, which owns one `hrx_buffer_t`, its persistent mapping,
  and its in-flight gate. `BufferInner::drop` releases that native reference exactly once.
  A queued job clones the same `Arc`, making the allocation and gate address-stable even if the
  caller moves or discards the Rust handle. Allocation releases the buffer on any post-allocation
  error path (a failed map, a rejected `BufferInfo`), and export lookup releases the executable on
  failure, so no handle leaks on error.

## Buffers and mappings

`allocate_buffer` requests `HOST_LOCAL | DEVICE_VISIBLE` memory with `DEFAULT | MAPPING_PERSISTENT`
usage, then establishes one persistent `READ | WRITE` mapping over the whole buffer. The reported
alignment is the largest power of two dividing the mapping address, which `BufferInfo::new` checks
against the request. Host access is bounded: `checked_range` validates `[offset, offset+len)`
against the mapped length before any `slice::from_raw_parts[_mut]`, and every raw slice is confined
to that range. `write_buffer` copies into the mapping then `hrx_buffer_flush_range`s the range
device-ward; `read_buffer` `hrx_buffer_invalidate_range`s the range then copies out. Both enforce
the buffer's `TRANSFER_DESTINATION`/`TRANSFER_SOURCE` usage.

## Concurrency and the dispatch worker

The HRX stream is not safe for concurrent use, so all stream access is serialized by the `Lane`
stream mutex: `allocate_buffer` locks it briefly, and the worker holds it across
`hrx_stream_dispatch` + `hrx_stream_synchronize`. `Stream` is `unsafe impl Send` (moved to the
worker, only ever dereferenced under that mutex).

`submit` validates the bindings in one pass (a nonempty, within-limit list; a queue, program, and
every buffer from one context; unique slots via a 256-bit occupancy mask; per-slot access; ranges;
an exact byte-length match against the program's per-slot plan, since the compiled TXN stream DMAs
fixed tensor extents regardless of the bound length; and no alias involving a write slot — one
buffer in several read slots is admitted, as OpenVINO admits it), then — the acceptance boundary —
claims a preallocated ring entry and event slot, arms each bound buffer's `in_flight` gate, and
enqueues a `Job`. The cross-context rejection is load-bearing for the release/dispatch race: a job
only ever references buffers whose owning context also owns the queue, so a valid submission cannot
outlive its buffers through a foreign context. A full ring returns `Busy`; submit never blocks.
`Job` is `unsafe impl Send`: its raw pointers are HRX buffer/executable handles — heap objects owned
by the HRX runtime, whose addresses are independent of where the caller's Rust handle values live —
and its `Arc<ProgramInner>`/`Arc<BufferInner>` owners keep every one live. The handles are
dereferenced only on the worker while the stream mutex is held. The in-flight gates live inside
those `BufferInner` Arcs, not caller structs: the contract requires the caller keep handles *alive*
until the event is terminal, not address-stable, so a caller may legally move an `XdnaBuffer`
mid-flight; the shared allocation keeps both handle and gate valid regardless. The worker, per job,
dispatches, synchronizes,
`invalidate_range`s each output, clears the `in_flight` gates, and latches the event's terminal
state exactly once. While a buffer's gate is set, `write_buffer`/`read_buffer`/`free_buffer` reject
with `Busy`, so no host access or release races the device.

Finite timeouts are rejected before admission (no cancellation exists at any layer). A synchronize
error latches the event `Failed` (a normal terminal state — the kernel TDR has quiesced the device)
and then poisons the instance, which refuses further work with `DeviceLost`. `poll_event` reads the
latched atomic state without touching HRX. The worker arms a 120-second watchdog immediately before
dispatch, longer than the kernel's 60-second NPU TDR. If HRX still has not returned, the watchdog
poisons the lane but deliberately leaves the accepted event pending and every gate armed: there is
no trustworthy completion boundary. `poll_event` then reports `DeviceLost`; event release remains
retryably rejected as `Busy`; discarding the backend enters the quarantine described above.
`EVENT_CANCELLATION` is not advertised.

The per-instance resource tracker counts accepted contexts, native buffer allocations, loaded
executables, queues, and events. Counters increment only after successful admission and decrement
only at the matching successful release/native final `Drop`; rejected operations do not perturb
them. The shared conformance hook samples these counts before and after every ordinary case. HRX's
release functions return `void`, so v1 has no runtime path with genuinely unknown ownership and
does not manufacture `Indeterminate` results. The feature-gated `test-control` constructor injects
one failure before touching HRX and is absent from ordinary builds; it shortens the same watchdog
state machine for deterministic on-metal tier-1/tier-2 tests.

## Audited unsafe operations

Every `unsafe` block in `src/native.rs` carries a local `SAFETY:` comment naming the invariant it
relies on (handle validity, out-pointer locality, status consumption, mapped-range bounds,
borrowed-for-the-call executable inputs, and single-drop ownership). The on-hardware integration
tests (`tests/hardware.rs`, `va_xdna` only) exercise device info, the buffer round trip,
out-of-bounds and permission rejection, the advertised-limit aggregation, the full precompiled
passthrough lifecycle (load → allocate → write+flush → submit → worker dispatch/synchronize →
poll → invalidate+read → release → teardown), and a compiled BF16 → FP32 MATMUL that runs
bit-exact against an integer oracle, rejected ownership paths, exact lifecycle accounting, and both
device-loss tiers — all against a live NPU. `tests/conformance.rs` runs the shared
`virtio-accel-conformance` semantic suite on the device, including resource accounting and the
direct-binding copy-path diagnostics case. Unsupported hosts compile no native module.
