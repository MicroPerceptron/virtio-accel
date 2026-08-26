# Qualcomm native-boundary safety argument

The native backend is compiled only when `build.rs` finds the public QAIRT/QNN headers and Windows
ARM64 HTP import library. Safe TOSA parsing, semantic admission, shape arithmetic, slot planning,
and resource-policy checks remain in `lower.rs`. Rust FFI calls are confined to `ffi.rs` and
`native.rs`; the C++ implementation is confined to `native/qnn_bridge.cpp` and uses only the
public QNN C interface.

## ABI and runtime lifetime

The bridge loads the SDK's `QnnHtp.dll`, resolves `QnnInterface_getProviders`, selects only the HTP
backend with the expected core API major and a compatible minor, and verifies every required
function pointer before use. `RuntimeHandle` owns the loaded DLL, backend, and device. Graphs retain
that handle through `Rc`, and events retain their graph, so the provider table and DLL necessarily
outlive every native call.

Each QNN backend, device, context/graph, and event has one native owner. Construction stores each
handle before the next fallible step and releases the initialized prefix on error. Explicit graph
or event release changes the stored pointer only after QNN reports success. `ReleaseFailure`
returns still-live Rust resources on `Busy`; indeterminate native failures do not fabricate
successful ownership transfer.

## Descriptors and buffers

Rust passes bounded tensor descriptors containing an explicit element tag, rank/dimensions, and
optional scale-offset quantization metadata. The synchronous graph-creation call validates every
tag, dimension pointer, rank, tensor role, I/O index, referenced value, node arity, parameter slice,
static-constant pointer, and exact element-count-derived byte length before QNN sees it.
`VaQnnGraph` then owns tensor names, dimensions, constant payloads, generated index/axis tensors,
node parameters, and operation configs until the graph context is destroyed. Its tensor vector is
capacity-checked and reserved before raw pointers into those records are retained. Pool-window,
reverse-index, reduction-decomposition, and tensor-count arithmetic is checked before graph
construction.

Every `HexagonBuffer` owns one zero-initialized `AlignedAllocation` with at least 4096-byte
alignment.
Submission validates the context, slot, access mode, exact byte length, and checked range before
passing the allocation's range pointer directly as a QNN raw client buffer. It creates only QNN
descriptor vectors, never a tensor-content bounce buffer. Events retain `Rc` references and
shared/exclusive access guards for all allocations until native execution is observed terminal;
the event retains its graph until release. Host transfers reject an allocation while it is in
flight.

## Asynchronous execution

QAIRT 2.49's HTP asynchronous graph entry point returns `QNN_COMMON_ERROR_NOT_SUPPORTED` on the
validated Windows target. The bridge therefore admits at most one request and moves blocking
`graphExecute` to one owned `std::thread`. `submit` returns after thread creation; `poll_event`
reads atomic state only and never calls QNN. The worker writes the QNN result, clears the runtime's
in-flight gate, and then publishes a terminal event with release ordering. Event destruction joins
the terminal worker before releasing its descriptors, graph, or allocation guards.

Finite deadlines return `DeadlineExpired` before admission. Cancellation is not advertised. Consequently no
timeout or cancellation path can release memory while HTP may still access it.

## Audited unsafe operations

Every Rust unsafe block has a local `SAFETY:` comment covering its pointer validity, allocation
layout, ownership, and call duration. The native integration tests exercise every advertised
FP16/BOOL/INT32 operator and exact INT8/INT32 numerical execution, the reusable lifecycle suite,
direct-binding diagnostics, stable terminal polling, pre-admission timeout rejection, live-event
graph retention, and ordered teardown on the pinned HTP runtime. Unsupported hosts compile without
either native module.

## Direct HTP boundary

The optional direct provider is confined to `direct.rs`, `host_bridge.cpp`,
the generated FastRPC stub, and the V73 skel. `Runtime` owns the driver module,
FastRPC handle, mapped arena, and allocation table. Rust buffers retain the
runtime through `Rc`; programs, queues, and buffers carry a context identifier,
and submission validates context, ordinal slots, access modes, exact ranges,
and checked pointer offsets before entering native code.

The bridge allocates each client-visible buffer with Qualcomm `rpcmem` and
passes that exact address in the synchronous FastRPC buffer argument. The skel
does not retain input, output, parameter, or binding pointers after a request.
Program parameter blobs are length-checked in Rust and checked again by the
skel before casting to fixed-layout parameter structures. Matmul dimensions
and byte counts are checked on the Rust side before binding; the skel rejects
zero dimensions and short buffers.

The fused Kerr-frame artifact additionally checks `width * height == lanes`,
requires one sample per pixel, validates finite camera/tetrad/FOV values and a
finite physically ordered trace configuration, binds exactly four input bytes and
`32 + 4 * lanes` output bytes, and rechecks those relationships before the skel
casts parameters or writes output. Each worker owns disjoint pixel ranges and
two disjoint slots in its private VTCM slice. A user-DMA descriptor is waited
before its slot can be reused and again before the worker returns, so neither
the stack descriptor nor VTCM source can expire while DMA is active. The
synchronous FastRPC call keeps the exact `rpcmem` destination alive until all
workers and DMA transfers complete.

`DirectHexagonBuffer::mapped_bytes` is an explicitly unsafe provider-local
escape hatch for zero-copy reads. Its safe Axiom wrapper borrows the complete
Kerr session mutably for the lifetime of the returned frame view, preventing
another submission or buffer release while the slice is live. The view is
created only after synchronous FastRPC completion and validates the 32-byte
header, remaining byte count, and `u32` alignment before exposing pixels.

Direct events are terminal because FastRPC execution completes before
`submit` returns. Finite deadlines are rejected before admission and neither
asynchronous completion nor cancellation is advertised. Drop closes the remote
handle and frees the mapped arena only after all Rust owners are gone. The
ignored hardware tests exercise construction, exact transfers, all direct
probe operations, repeated polling, explicit destruction, and fused Axiom
workloads through the signed V73 module.
