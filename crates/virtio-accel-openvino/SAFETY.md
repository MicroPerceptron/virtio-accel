# Unsafe-code audit

This crate is a host-native exception to the portable workspace's `forbid(unsafe_code)` rule.
Unsafe Rust is confined to the `va_openvino` build configuration — `src/ffi.rs` (declarations
only) and `src/native.rs` — and has three responsibilities:

1. declaring and calling the OpenVINO C API (`libopenvino_c`, mirrored from `openvino/c/*.h`);
2. owning page-aligned allocations made through `std::alloc`; and
3. exposing those allocations to the runtime only while the `Accelerator` contract supplies the
   required exclusive or terminal-event access.

Builds without a detected runtime compile no `unsafe` at all: the crate root carries
`cfg_attr(not(va_openvino), forbid(unsafe_code))`.

Every OpenVINO handle has exactly one Rust owner with a `Drop` implementation, and every call
site checks `ov_status_e` before trusting out-pointers. One exception is deliberate: the process
holds a single shared `ov_core_t` for its lifetime, because re-creating plugin engines is not
crash-safe — a second `zeInitDrivers` through a re-initialized Intel NPU plugin segfaults inside
the Level Zero loader on hosts without a vendor driver (observed with ze_loader 1.28, OpenVINO
2026.3). `ov::Core` is documented thread-safe, which is the basis for the audited `Send`/`Sync`
implementations on that shared handle. The single C-variadic call site,
`ov_core_compile_model`, passes `property_args_size` as the count of variadic arguments (one
key/value pair contributes two), pinned by a unit test against the installed runtime.

`AlignedAllocation` owns exactly one `std::alloc::Layout`. Its pointer is non-null, is
deallocated with the same layout, and is kept alive by `Arc` clones held in event backing
guards. Buffers are deliberately neither `Send` nor `Sync`. Their atomic in-flight state admits
multiple native readers or one native writer, rejects host transfers while either mode is
active, and rejects every conflicting submission. Multiple bindings from one event to the same
allocation are collapsed to the strongest required access before acquiring a guard.

Event completion is poll-latch: no foreign callback ever owns Rust memory. Each submission's
infer request, bound tensors, and backing guards are owned by the Rust event handle. Polling
probes the request without blocking; on the first terminal observation the backing guards are
dropped strictly before the latched state becomes observable, so a caller that sees a terminal
state can immediately transfer buffer bytes. Dropping an unpolled live event first cancels and
joins the request, then frees the request and tensors, and only then releases the guards —
runtime threads can never touch freed Rust memory. The backend never creates public slices or
raw-pointer accessors.

OpenVINO receives tensors created from host pointers over Rust-owned memory. Submission
validates range length and scalar alignment before exposing a bound pointer, and completion
verifies the output tensor's data pointer before reporting success. A runtime that substitutes
its own output allocation is an execution failure, never a silent copy.
