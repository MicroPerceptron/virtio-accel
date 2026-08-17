# Qualcomm QNN native-boundary safety argument

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

`VaQnnGraph` owns tensor names, dimensions, constants, node parameters, and operation configs until
the graph context is destroyed. Its tensor vector is capacity-checked and reserved before raw
pointers into those records are retained. Pool-window index and tensor-count arithmetic is checked
before graph construction.

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

Finite deadlines are rejected before admission. Cancellation is not advertised. Consequently no
timeout or cancellation path can release memory while HTP may still access it.

## Audited unsafe operations

Every Rust unsafe block has a local `SAFETY:` comment covering its pointer validity, allocation
layout, ownership, and call duration. The native integration tests exercise numerical execution,
stable terminal polling, pre-admission timeout rejection, live-event graph retention, and ordered
teardown on the pinned HTP runtime. Unsupported hosts compile without either native module.
