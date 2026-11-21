# Unsafe-code audit

This crate is a host-native exception to the portable workspace's `forbid(unsafe_code)` rule.
Unsafe Rust is confined to the `va_openvino` build configuration — `src/ffi.rs` (typed declarations and runtime symbol loading) and `src/native.rs` — and has three responsibilities:

1. declaring and calling the OpenVINO C API (`libopenvino_c`, mirrored from `openvino/c/*.h`);
2. owning page-aligned allocations made through `std::alloc`; and
3. exposing those allocations to the runtime only while the `Accelerator` contract supplies the
   required exclusive or terminal-event access.

Builds without the native configuration compile no `unsafe` at all: the crate root carries
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
Quantized FP32 and exact FP8 activation graphs pass four arguments: execution mode `ACCURACY`
and inference precision `f32`. Other graphs pass a count of two; the runtime ignores the extra
variadic pair. All property keys and values remain live throughout the synchronous compile.
A startup probe with the same precision properties controls FP32 quantized admission.

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
runtime threads can never touch freed Rust memory.

## Shared immutable inputs and borrowed output access

`SharedInput::initialize` allocates through the same checked, zero-initialized
`AlignedAllocation`. Its exclusive callback receives only the logical byte range
before any Arc, buffer or native tensor can observe it. The callback cannot return
a borrow of this storage. After successful initialization, `SharedInput` exposes
only immutable bytes and clones of its Arc; no mutable reference or raw-pointer
import API is added. `import_shared_input` fixes usage to PROGRAM_INPUT and
TRANSFER_SOURCE. Both host writes and native Write/ReadWrite bindings are therefore
rejected by existing permission checks. Imports in separate contexts share the
same allocation and atomic event gate. Buffer and event Arcs keep it alive after
all caller input handles are dropped. This also justifies moving frozen input
handles across threads under the existing allocation Send/Sync implementations.

`OpenVinoBuffer::with_bytes` checks host-read permission and holds a shared
`EventBacking` guard across its closure-scoped immutable slice. A native writer
prevents entry; reentrant or concurrent native writers and host writes are rejected
while the slice exists. Shared native readers are compatible. The callback cannot
return a reference into the allocation, and unwinding releases the guard. Buffers
remain !Send/!Sync; no mutable buffer view or public raw-pointer accessor is added.
These host-only APIs do not advertise protocol external-memory import/export.

The queue may retain one empty high-water vector allocation for backing metadata and one for
bound-tensor metadata. Recycling clears and drops every element before the allocation enters the
queue spare, so a spare owns no OpenVINO handle, `Arc`, backing guard, or buffer pointer. Events
refer to the queue scratch weakly and therefore cannot extend queue lifetime. The separate
pointer-slot scratch uses an RAII guard that clears its length on every return path, including a
rejected submission.

A per-program cache retains at most eight successfully completed infer requests.
An event owns an `Rc<ProgramResident>` retaining the compiled model and constant
storage, even if the public program handle is unloaded before completion. The
resident drops cached requests before the compiled model, weights tensor, and
weights bytes. No global submission lock or cross-program request cache is used.

The C API cannot detach all bound host pointers, so each cached `BoundTensor`
retains its allocation Arc as well as its native tensor. Idle requests retain no
access guards and cannot run independently. Every submission validates bindings
and acquires fresh guards before taking a cached request. Exact pointer matches
reuse the tensor wrappers; other bindings retain all old wrappers until every
replacement succeeds. A partial rebind failure frees the request before either
old or new backing storage is released. Cache entries cannot confuse reused host
addresses because their allocation Arcs keep those addresses allocated.

Only events latched `Complete` enter the cache. Failed, cancelled, rejected, or
unobserved events use the existing quiesce/free path. `CachedRequest::drop` frees
its native handle before Rust drops bound tensors and allocation Arcs. Moving an
entry into an event transfers the unique request owner; no second handle owner
or reference cycle is introduced. Keeping a program alive can therefore retain
up to eight idle sets of host allocations after public buffer handles are freed.
Unloading the last program/event owner releases them. Counters expose request
creation/reuse for tests and benchmark assertions.


OpenVINO receives tensors created from host pointers over Rust-owned memory. Submission
validates range length and scalar alignment before exposing a bound pointer, and completion
verifies the output tensor's data pointer before reporting success. A runtime that substitutes
its own output allocation is an execution failure, never a silent copy.

## Optional runtime loading

The opt-in `runtime-loading` feature compiles the existing native boundary without
linking a vendor library into the process at startup. `ffi.rs` has one declaration
list for linked symbols and typed function pointers. `libloading` resolves the full
required ABI before any native initialization; a missing library or symbol returns
`RuntimeUnavailable`. The caller still supplies a compatible vendor implementation
of the audited C ABI. No function is called through an unresolved table.

A process-wide `OnceLock` retains the successful table and its `Library` for the
process lifetime. Native handles, asynchronous work and quarantined resources can
therefore never outlive their function pointers. Concurrent successful loads retain
one table; unused library references are dropped without replacing the live runtime.
Failed library loads are not cached, so configuration can be corrected before retry.
The existing one-time native initialization and all buffer/event ownership rules are
unchanged. Loader work occurs during initialization, not on the submission path.

The OpenVINO execution-mode property is an exported pointer variable. Its symbol
address is read as `*const *const c_char`; the non-null, NUL-terminated immutable key
is copied into an owned `CString` in the table before use. The C-variadic compilation
function keeps exactly the same signature and argument-count convention.

Compilation passes one to three key/value pairs as a contiguous variadic prefix:
ACCURACY, CPU/GPU `DYNAMIC_QUANTIZATION_GROUP_SIZE=0`, and optional FP32 precision.
The argument count is twice the number of pairs, and every key/value is a static C
string. NPU does not receive the unsupported dynamic-quantization property. Native i4
is internal constant storage only: no i4 type can reach host buffer binding/alignment.

On Windows an explicit runtime path is made absolute and loaded with
`LOAD_WITH_ALTERED_SEARCH_PATH`, keeping dependent DLL lookup relative to that
library without mutating the process-wide DLL search configuration.
