# Qualcomm QNN native-boundary safety plan

The currently compiled SDK-free crate forbids unsafe code. This document defines the review gates
that must be satisfied before enabling the native QNN modules; it is not evidence that the native
backend is complete.

## Boundary

Only `ffi.rs` and `native.rs` may eventually contain unsafe code. TOSA parsing, semantic analysis,
operator admission, graph planning, binding-slot construction, size arithmetic, and test fixtures
remain safe Rust in `lower.rs`. No Qualcomm type may appear in `virtio-accel-core`,
`virtio-accel-tosa`, the facade, or another portable crate.

The bridge must use the public QNN C API obtained from the selected QAIRT SDK. Internal driver APIs,
the Windows `QnnHtp*Drv` libraries, sample-only interfaces, and layout guesses reconstructed without
the matching public headers are outside the boundary.

## Required native invariants

### Interface and ABI

- Load `QnnInterface_getProviders` from the selected HTP backend and copy only a provider whose
  core API major/minor range is validated against the headers used at build time.
- Treat the provider table and every function pointer as live only while the backend DLL remains
  loaded. The DLL owner must outlive backend, device, context, graph, signal, and tensor use.
- Reject missing functions and incompatible versions before creating native state. Never call a
  function pointer merely because a newer table is large enough to contain its slot.

### Handle ownership and teardown

Every QNN backend, device, context, graph, signal, and registered-memory handle has one Rust owner.
Drop order is the reverse of successful construction. Partial construction stores each new owner
immediately so an error releases exactly the initialized prefix. Explicit release transfers or
returns ownership according to `ReleaseFailure`; no handle is released twice.

An unrecoverable HTP result poisons the backend. The process does not try to reconstruct unknown
native ownership by reusing a context, graph, signal, or buffer registration after device loss.

### Tensor descriptors and direct buffers

- A program owns all QNN tensor names, dimensions, scalar/quantization descriptors, operation
  configs, and parameter storage until the documented QNN copy point. Anything documented as
  retained remains program-owned through graph destruction.
- A buffer owns one aligned, zero-initialized allocation. The exact pointer and bound range passed
  by the caller become the QNN client buffer or registered-memory view used at dispatch.
- Submission may create small descriptor metadata but never an input/output allocation or content
  copy. The program and every bound allocation remain strongly owned until the event is terminal
  and released.
- Explicit read/write operations perform any required QNN/HTP synchronization around the existing
  allocation. They are the only provider content-copy boundaries.

### Worker and event synchronization

The initial backend has one bounded serialized lane per device. `submit` reserves access guards and
enqueues owned work without waiting for QNN execution. The worker may call blocking QNN execution;
`poll_event` reads only an atomic/locked Rust latch and never enters QNN.

The worker releases output synchronization and buffer guards before publishing a terminal state.
The event retains its program, descriptor storage, allocations, and native completion state until
release. No callback receives a pointer to stack storage. If a QNN callback is used, its context is
an `Arc` raw pointer with an audited exactly-once foreign/Rust reference transfer.

### Cancellation and timeouts

Cancellation is advertised only if the selected QNN signal API proves bounded, race-safe
cancellation. Otherwise a finite timeout may reject before admission, but accepted native work
retains its event and allocation ownership until the worker observes completion. A timeout never
causes early descriptor or buffer destruction.

## Audit evidence required before native enablement

- Build the native boundary against the exact pinned QAIRT headers and Windows ARM64 libraries.
- Run malformed-input, lifecycle, concurrency, timeout, device-loss, and exact-once teardown tests.
- Pass the reusable semantic suite and every advertised FP16 numerical fixture on HTP.
- Demonstrate exact pointer/range binding and zero provider-staged submissions through conformance
  diagnostics.
- Record runtime logs proving HTP placement and no CPU/GPU partition.
- Add local `SAFETY:` comments to every unsafe block naming the invariant above that makes it valid.
