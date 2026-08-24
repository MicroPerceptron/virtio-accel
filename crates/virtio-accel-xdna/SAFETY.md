# Unsafe-code audit

This crate is a host-native exception to the portable workspace's `forbid(unsafe_code)` rule, on
the same terms as `virtio-accel-openvino`. Unsafe Rust is confined to the `va_xdna` build
configuration — `src/ffi.rs` (HRX C ABI declarations only) and `src/native.rs` (the calls) — and
the crate root carries `cfg_attr(not(va_xdna), forbid(unsafe_code))`. Builds without a detected HRX
runtime compile no `unsafe` at all: only the portable admission surface (`src/lower.rs`) and a
placeholder.

Scope implemented so far: the device/stream owner and `hrx_buffer` primitives. Program loading and
dispatch report `Unsupported`; their FFI and audit land with the execution path (at which point the
uninhabited `XdnaProgram`/`XdnaEvent` become real types and this file gains their sections).

## ABI and status ownership

`src/ffi.rs` declares the HRX C ABI transcribed from the pinned release headers
(`include/hrx/hrx_runtime.h`, `libhrx.so.0.1.0`, version 0.1.0): opaque refcounted handles, the
`uint32`/`uint16` bitmask constants, and the `extern "C"` functions this ticket calls. No struct
layout is invented; handles are opaque pointers.

Every fallible HRX call returns an `hrx_status_t` — an owned, opaque pointer that is `NULL` on
success and must be consumed exactly once on failure. The single `check` helper in `native.rs`
enforces this: on a non-NULL status it reads the code, renders the message with
`hrx_status_to_string` (freeing that message with `hrx_status_free_message`), then `hrx_status_ignore`s
both the string-status and the original status. No status escapes unconsumed, so none leaks.

## Handle lifetime and teardown

Each HRX handle has exactly one Rust owner with a `Drop`:

- The process-wide device is initialized once through a `OnceLock` (`hrx_gpu_initialize` →
  `hrx_gpu_device_count` → `hrx_gpu_device_get`) and held in `SharedDevice`. Per the fork's model
  (one device per process) it is never shut down; `hrx_gpu_shutdown` is never called. `SharedDevice`
  is `unsafe impl Send + Sync` because it is used only as a read-only factory for per-instance
  streams and the amdxdna HAL exposes the device as a process-wide singleton.
- Each `XdnaAccelerator` owns one `hrx_stream_t`, released exactly once in `Drop`. The instance is
  `!Send`/`!Sync` (a `PhantomData<*mut u8>`): one stream is not safe for concurrent dispatch, so the
  instance stays single-threaded until the execution path introduces the serialized worker.
- Each `XdnaBuffer` owns one `hrx_buffer_t` and its persistent mapping, released exactly once in
  `Drop` (the mapping is released together with the buffer; the fork never unmaps first). Buffers
  are `!Send`/`!Sync`. Allocation releases the buffer on any post-allocation error path (a failed
  map or a rejected `BufferInfo`), so no handle leaks on error.

## Buffers and mappings

`allocate_buffer` requests `HOST_LOCAL | DEVICE_VISIBLE` memory with
`DEFAULT | MAPPING_PERSISTENT` usage, then establishes one persistent `READ | WRITE` mapping over
the whole buffer (`hrx_buffer_map_with_mode`). The reported alignment is the largest power of two
dividing the mapping address, which `BufferInfo::new` checks against the requested alignment. Sizes
are `usize`-checked; `BufferDesc` guarantees a nonzero size (HRX rejects zero).

Host access to the mapping is bounded: `checked_range` validates `[offset, offset+len)` against the
mapped length before any `slice::from_raw_parts[_mut]`, and every raw slice is confined to that
validated range. `write_buffer` copies into the mapping then `hrx_buffer_flush_range`s the range
device-ward; `read_buffer` `hrx_buffer_invalidate_range`s the range then copies out — the explicit
cache management the persistent mapping requires. An `in_flight` gate (always zero until the
dispatch path sets it) rejects host access and release while the device may be touching the buffer;
`free_buffer` returns the live handle via `ReleaseFailure::Rejected` when the gate is set.

## Audited unsafe operations

Every `unsafe` block in `src/native.rs` carries a local `SAFETY:` comment naming the invariant it
relies on (handle validity, out-pointer locality, status consumption, mapped-range bounds, and
single-drop ownership). The on-hardware integration tests (`tests/hardware.rs`, `va_xdna` only)
exercise device info, the allocate / map / write+flush / read+invalidate / release round trip,
out-of-bounds rejection, and context/queue lifecycle against a live NPU. Unsupported hosts compile
no native module.
