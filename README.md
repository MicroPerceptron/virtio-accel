# virtio-accel

[![CI](https://github.com/MicroPerceptron/virtio-accel/actions/workflows/ci.yml/badge.svg)](https://github.com/MicroPerceptron/virtio-accel/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/virtio-accel.svg)](https://crates.io/crates/virtio-accel)
[![docs.rs](https://docs.rs/virtio-accel/badge.svg)](https://docs.rs/virtio-accel)
[![GitHub last commit](https://img.shields.io/github/last-commit/MicroPerceptron/virtio-accel.svg)](https://github.com/MicroPerceptron/virtio-accel/commits/main)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)
[![MSRV](https://img.shields.io/badge/rustc-1.85+-blue.svg)](#development)
[![no_std](https://img.shields.io/badge/no__std-supported-brightgreen.svg)](#portability)

An experimental virtual-accelerator protocol plus production-oriented Rust implementations.

`virtio-accel` defines a protocol and ships executable `no_std` guest, device, transport, queue, and
TOSA layers for exposing an accelerator to a guest: contexts, buffers, programs, execution queues,
submissions, and events. The workspace also contains two real host backends: a macOS Core ML
backend that lowers device-neutral TOSA into Core ML and submits ANE-capable predictions with
direct buffer bindings, and an Intel OpenVINO backend that lowers the same TOSA artifacts to
in-memory OpenVINO IR and executes them on NPU, GPU, or CPU inference devices with direct
host-pointer tensors. The first target is NPU execution, while the object model deliberately
leaves room for GPUs, DSPs, and other program-driven accelerators.

This is no longer a specification-only repository: the frozen protocol review input is developed
alongside runnable guest/device machinery, conformance infrastructure, TOSA ingestion and analysis,
and host backends. Host integrations remain isolated in adapter crates and never become dependencies
of the portable facade. The project claims no Virtio device ID (_yet_).

This project is pre-standardization and experimental. Protocol 1.0 is frozen as a versioned review
input for independent implementation — it is stable enough to build against and to disagree with in
writing, not an approved Virtio specification.

## Backend support

“Supported” below means that the backend admits the declared program and dtype and exercises it
end-to-end; support in the TOSA parser or shared numerical corpus alone does not imply hardware
execution. “Not implemented” describes this repository, not necessarily the underlying hardware.

| Backend                                     | Status                 | Program admission                     | FP32            | FP16            | FP8 E4M3/E5M2   | INT8            | Packed INT4     | Program-visible buffers     |
| ------------------------------------------- | ---------------------- | ------------------------------------- | --------------- | --------------- | --------------- | --------------- | --------------- | --------------------------- |
| Apple Core ML / ANE (`virtio-accel-coreml`) | Implemented; macOS 14+ | Static TOSA 1.0 FP; INT8 tier on macOS 26+ | Supported       | Supported       | Not implemented | Identity + MATMUL | Not implemented | Direct host/shared bindings |
| Intel OpenVINO (`virtio-accel-openvino`)    | Implemented; OpenVINO 2026.x | Static TOSA 1.0 FP + INT8 tier | Supported       | Supported       | Not implemented | Identity + MATMUL | Not implemented | Direct host/shared bindings |
| AMD XDNA                                    | Planned                | Not implemented                       | Not implemented | Not implemented | Not implemented | Not implemented | Not implemented | Not implemented             |
| Qualcomm Hexagon                            | Planned                | Not implemented                       | Not implemented | Not implemented | Not implemented | Not implemented | Not implemented | Not implemented             |

The Core ML row describes model-boundary support; restricted INT32 outputs are also available.
Core ML chooses ANE or CPU placement per operation. Direct INT8 boundaries require macOS 26+, and
the exact integer tier currently covers identity and zero-point-aware MATMUL only. Its INT4
facilities are compressed-weight storage rather than TOSA INT4 tensor execution, and this backend
does not silently dequantize unsupported FP8, INT8, or INT4 graphs. See the
[`virtio-accel-coreml` support boundary](crates/virtio-accel-coreml/README.md#low-precision-boundary).

The OpenVINO row also describes model-boundary support with restricted INT32 outputs. Its integer
tier uses direct INT8 boundaries and explicit INT32 zero-point legalization for MATMUL. The backend compiles per enumerated device (NPU, then GPU, then CPU by default)
with OpenVINO's accuracy-preserving execution mode, and accepts completion only when the runtime
executed into the caller's own output allocation. NPU and GPU devices enumerate when the Intel
Level Zero NPU driver or GPU compute runtime is installed; the CPU plugin is exercised in CI. Like
the Core ML backend it rejects unsupported FP8, unsupported INT8 operators, and packed INT4 graphs while loading instead
of silently dequantizing. See the
[`virtio-accel-openvino` support boundary](crates/virtio-accel-openvino/README.md#low-precision-boundary).

Independently of backend execution, `virtio-accel-tosa` validates the TOSA 1.0 profiles and
extensions for all five dtype columns, and `virtio-accel-conformance` ships shared fixtures and
oracles for them. The byte-oriented `virtio-accel-mock` backend remains test infrastructure rather
than a typed hardware implementation.

## Workspace

| Crate                      | Tier           | Role                                                                                                         |
| -------------------------- | -------------- | ------------------------------------------------------------------------------------------------------------ |
| `virtio-accel`             | `core + alloc` | Facade re-exporting the portable layers                                                                      |
| `virtio-accel-proto`       | `core`         | Pointer-free, little-endian protocol 1.0 wire structures                                                     |
| `virtio-accel-transport`   | `core`         | Dependency-free descriptor-chain, queue, reset, and notification ports                                       |
| `virtio-accel-core`        | `core`         | Backend lifecycle, memory, program, queue, and event contracts                                               |
| `virtio-accel-tosa`        | `core + alloc` | Bounded zero-copy TOSA 1.0 validation, lowering analysis, specialization, and packed low-precision utilities |
| `virtio-accel-coreml`      | macOS `std`    | TOSA-to-Core ML lowering, direct buffers, and asynchronous ANE-capable prediction                            |
| `virtio-accel-openvino`    | Linux `std` (probed) | TOSA-to-OpenVINO IR lowering, direct host-pointer tensors, and asynchronous NPU/GPU/CPU inference      |
| `virtio-accel-split-queue` | `core + alloc` | Bounded in-memory split-ring reference model                                                                 |
| `virtio-accel-guest`       | `core + alloc` | Typed reference client with bounded request tracking                                                         |
| `virtio-accel-device`      | `core + alloc` | Device-owned state, including bounded generational IDs                                                       |
| `virtio-accel-mock`        | `std`          | In-memory backend with deterministic test-only artifacts and scripted faults                                 |
| `virtio-accel-conformance` | `std`          | Transport-free semantic suite and shared FP32/FP16/FP8/INT8/INT4 numerical TOSA corpus                       |
| `virtio-accel-cleanroom`   | `core`         | Independent conformance codec, written without the shared protocol types                                     |

Dependencies point downward only:

```text
virtio-accel-split-queue ---> virtio-accel-transport
                                      ^
                                      |
virtio-accel-device ----------+-------+------> virtio-accel-core
          |
          +-----> virtio-accel-proto

virtio-accel-guest -----------> virtio-accel-transport
          |
          +--------------------> virtio-accel-proto

virtio-accel-conformance --------------------> virtio-accel-core
virtio-accel-tosa ---------------------------> virtio-accel-core
virtio-accel-coreml ----------+--------------> virtio-accel-core
                              |
                              +--------------> virtio-accel-tosa
virtio-accel-openvino --------+--------------> virtio-accel-core
                              |
                              +--------------> virtio-accel-tosa
other provider adapters --------------------> virtio-accel-core
```

The transport crate exposes reset-scoped chain identities, flattened direction/length metadata, and
owned publication/completion tokens. Neither it nor the device-state layer leaks guest addresses,
ring pointers, or concrete descriptor types into the command engine or provider backend.

## Install

```toml
[dependencies]
virtio-accel = "0.2"
```

The facade is `no_std`. Add the reference backend as a dev-dependency to run the example below:

```toml
[dev-dependencies]
virtio-accel-mock = "0.2"
```

On an ANE-capable Mac, add `virtio-accel-coreml = "0.1"` separately for the host-native backend.
On a Linux host with an OpenVINO 2026.x runtime, add `virtio-accel-openvino = "0.1"` instead. Both
adapters accept the production TOSA 1.0 program format; validation, analysis, and native model
generation all happen inside the adapter. Neither is re-exported by the portable facade.

Add `virtio-accel-tosa = "0.1"` separately to validate TOSA 1.0 artifacts, inspect safe borrowed
graph and typed-attribute views, enforce complete stable-op semantics for a declared target, and
construct the device-neutral TOSA artifact envelope. `Model::analyze_for` also produces bounded
dense IDs, topological order, liveness, runtime obligations, and specialization keys for Core ML,
OpenVINO, or another provider. It is intentionally not re-exported by the facade.

## Production TOSA-to-Core ML example

On macOS 14+ with an accessible Apple Neural Engine, the backend-local example sends a TOSA 1.0
`IDENTITY` graph through the real lowering, compilation, direct-binding, asynchronous prediction,
and teardown path:

```sh
cargo run -p virtio-accel-coreml --example tosa_coreml
```

```text
TOSA -> Core ML -> ANE-capable result: 3.25
```

On a Linux host with an OpenVINO 2026.x runtime, the equivalent backend-local example executes the
same graph on the preferred available Intel inference device (NPU, then GPU, then CPU):

```sh
cargo run -p virtio-accel-openvino --example tosa_openvino
```

```text
TOSA -> OpenVINO -> CPU result: 3.25
```

The portable facade, device engine, transport, and guest layers see only the TOSA artifact format,
target identity, and opaque bytes. Core ML protobufs, temporary compilation assets, Foundation, and
the Objective-C bridge remain owned by `virtio-accel-coreml`.

## Portable lifecycle example

A full submission against the in-memory reference backend — allocate a buffer, load an artifact,
bind it to a slot, submit, and observe the event:

```rust
use virtio_accel::core::{
    Accelerator, AccessMode, ArtifactRef, BindingRef, BufferDesc, BufferRange, BufferUsage,
    ContextDesc, EventState, MemoryDomain, QueueDesc, SubmitFailure, Timeout,
};
use virtio_accel_mock::{MockAccelerator, reference};

let backend = MockAccelerator::default();
let context = backend.create_context(ContextDesc::default())?;

// An 8-byte shared buffer the program may read and write.
let desc = BufferDesc::new(
    8,
    8,
    MemoryDomain::Shared,
    BufferUsage::TRANSFER_SOURCE
        | BufferUsage::TRANSFER_DESTINATION
        | BufferUsage::PROGRAM_INPUT
        | BufferUsage::PROGRAM_OUTPUT
        | BufferUsage::MUTABLE_STATE,
)?;
let (mut buffer, _) = backend.allocate_buffer(&context, desc)?.into_parts();
backend.write_buffer(&mut buffer, 0, &[0x00, 0x11, 0x7f, 0x80, 0xa5, 0xff, 0x3c, 0xc3])?;

// A deterministic test-only artifact: XOR every byte bound to slot 7 with 0x5a.
let artifact = reference::ReferenceArtifact::xor(7, 0x5a);
let program = backend.load_program(
    &context,
    ArtifactRef {
        format: reference::ARTIFACT_FORMAT,
        target: reference::TARGET_IDENTITY,
        payload: artifact.as_bytes(),
        resident_bytes: reference::RESIDENT_BYTES,
    },
)?;
let queue = backend.create_queue(&context, QueueDesc::default())?;

let bindings = [BindingRef {
    slot: 7,
    buffer: &buffer,
    range: BufferRange::new(0, 8)?,
    access: AccessMode::ReadWrite,
}];

// Submission is asynchronous at the ownership boundary, so it always yields an event.
let event = backend
    .submit(&queue, &program, &bindings, Timeout::Infinite)
    .map_err(|failure| match failure {
        SubmitFailure::Rejected(error) | SubmitFailure::Indeterminate { error, .. } => error,
    })?;
assert_eq!(backend.poll_event(&event)?, EventState::Pending);

// The mock backend runs under harness control, so the caller drives completion.
backend.complete(&event)?;
assert_eq!(backend.poll_event(&event)?, EventState::Complete);

let mut output = [0_u8; 8];
backend.read_buffer(&buffer, 0, &mut output)?;
assert_eq!(output, [0x5a, 0x4b, 0x25, 0xda, 0xff, 0xa5, 0x66, 0x99]);
```

Every object is released explicitly, and a release can itself fail; see
[`examples/reference_execution.rs`](examples/reference_execution.rs) for the teardown path.

```sh
cargo run --example reference_execution
```

## Protocol 1.0

The protocol defines fixed headers and payloads for device discovery, contexts, buffers, programs,
execution queues, submissions, and events. Two properties shape most of the API:

- **Unknown values stay raw.** Unrecognized opcodes, statuses, and event states remain integers
  until validated, so decoding untrusted bytes never constructs an invalid Rust enum.
- **Failure still returns an event.** A successful submit returns an event; an _indeterminate_
  failure must also return one, because the operation's resources are still owned by the device.
  Guest-visible object IDs are opaque, kind-tagged, generational, and never reused after generation
  exhaustion.

The primary `zerocopy` ABI and the manual clean-room codec both decode and re-encode every canonical
frame. Their bridge test exchanges bytes only, providing an independent implementation check without
making the conformance codec a production dependency.

Non-Rust device and driver implementations can include
[`include/virtio_accel.h`](include/virtio_accel.h). The header is a packed C projection of the wire
contract, not a host backend plugin ABI. CI compiles it as C11 and C++11 and derives constant,
size, alignment, and offset assertions from the frozen layout manifest.

## Writing a backend

Implement the `Accelerator` contract from `virtio-accel-core`, then run the standard semantic suite
against it. The suite is transport-free: no wire format, virtqueue, OS, or vendor dependency.

```sh
cargo run --example backend_conformance
```

```text
memory.shared: Passed
buffer.transfer-permissions: Passed
submission.context-isolation: Passed
event.cancellation-races: Passed
accounting.resource-lifecycle: Passed
...
```

The [backend implementer guide](docs/backend-implementer-guide.md) walks through the hooks, the
optional resource-accounting and progress adapters, and the fault-injection harness.

## Documentation

| Document                                                          | Covers                                                                                       |
| ----------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| [specification.md](docs/specification.md)                         | Normative terminology, object model, compatibility rules, mandatory baseline                 |
| [wire-abi.md](docs/wire-abi.md)                                   | Exact byte layouts and the coordinated change procedure                                      |
| [virtio_accel.h](include/virtio_accel.h)                          | Checked C and C++ projection of the protocol 1.0 wire contract                               |
| [virtqueue.md](docs/virtqueue.md)                                 | Command-chain rules                                                                          |
| [architecture.md](docs/architecture.md)                           | Implementation invariants                                                                    |
| [threat-model.md](docs/threat-model.md)                           | Trust boundaries and finite resource policy                                                  |
| [portability.md](docs/portability.md)                             | Enforced target matrix and crate tiers                                                       |
| [performance.md](docs/performance.md)                             | v1 performance and copy budgets                                                              |
| [public-api.md](docs/public-api.md)                               | Public rustdoc policy                                                                        |
| [release-policy.md](docs/release-policy.md)                       | Release governance and evolution rules                                                       |
| [backend-implementer-guide.md](docs/backend-implementer-guide.md) | Running the semantic suite against a new backend                                             |
| [releases/v1.0.md](docs/releases/v1.0.md)                         | Protocol 1.0 release note                                                                    |
| [conformance/v1.0](conformance/v1.0/README.md)                    | Golden artifacts, canonical frames, and the [freeze audit](conformance/v1.0/freeze-audit.md) |
| [CONTRIBUTING.md](CONTRIBUTING.md)                                | Development gates, protocol change classification, and scope boundaries                      |
| [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)                          | Expected conduct in project spaces                                                           |
| [SECURITY.md](SECURITY.md)                                        | Reporting a vulnerability                                                                    |

## Portability

Project-authored portable and reference code forbids or denies unsafe code. The audited Core ML
adapter keeps its unsafe FFI isolated to macOS; the TOSA crate confines official generated
FlatBuffers accessors to a private module behind bounded verification. CI enforces each portability
tier, including compile-only checks of the adapter's unsupported-platform surface.

| Tier           | Allowed runtime surface                                                      |
| -------------- | ---------------------------------------------------------------------------- |
| `core`         | `core` only; no allocation                                                   |
| `core + alloc` | `core + alloc`; no OS, filesystem, sockets, threads, or host synchronization |
| `std`          | Portable `std`; no host-OS or vendor-specific API                            |
| macOS `std`    | Host-native Core ML/Foundation adapter; never a portable default dependency  |

Concrete VMM, kernel, OS, and vendor adapters do not change the portable v1 protocol and must not
become default dependencies of a portable crate. Cargo features must be additive: disabling default
features may remove convenience behavior, but must never select a different protocol interpretation.

## Development

Minimum supported Rust version is 1.85 (edition 2024), checked in CI.

```sh
cargo fmt --all -- --check
python3 ci/check-release-policy.py
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-targets --all-features
cargo run --example backend_conformance
cargo run --example reference_execution
cargo run -p virtio-accel-coreml --example tosa_coreml # macOS 14+ with ANE
cargo run -p virtio-accel-openvino --example tosa_openvino # Linux with OpenVINO 2026.x
python3 ci/publish-dry-run.py
```

Target checks need the corresponding standard libraries:

```sh
rustup target add aarch64-unknown-none riscv64gc-unknown-none-elf wasm32-unknown-unknown
```

## Status

Included in protocol 1.0:

- one command virtqueue at index zero
- device discovery and exact protocol compatibility checks
- contexts, buffers, opaque programs, execution queues, submissions, and events
- bounded explicit buffer transfers
- event polling, optional cancellation, release, reset, and backend-discard recovery
- direct-binding requirements for program-visible buffers
- checked finite limits for untrusted byte counts, descriptor counts, object counts, and retained
  backend storage
- an independent clean-room codec and a transport-free semantic conformance suite

Reserved and unadvertised — an implementation that advertises one of these is not 1.0 conformant
until a future version assigns its negotiation, ownership, synchronization, and conformance rules:

- multi-queue and event queues
- external memory import/export
- timeline fences
- secure contexts
- packed virtqueues
- protocol-level negotiation for additional VMM, kernel, OS, and vendor integrations
- a standardized graph IR, compiler, or executable format

Protocol 1.0 numeric opcodes, statuses, and payload layouts are frozen for the portable v1.0
baseline by the [final freeze audit](conformance/v1.0/freeze-audit.md). Future changes must follow
the coordinated change procedure in [wire-abi.md](docs/wire-abi.md) and the
[release and evolution policy](docs/release-policy.md); incompatible changes require a new protocol
major version.

## Contributing

Contributions are welcome, including disagreement with frozen decisions — a reasoned objection is
worth more than a workaround built on top of one. See [CONTRIBUTING.md](CONTRIBUTING.md) for the
local gates, the scope boundaries, and how wire changes are classified before code is merged.

- Questions and backend porting help →
  [Discussions → Q&A](https://github.com/MicroPerceptron/virtio-accel/discussions/categories/q-a)
- Early design ideas →
  [Discussions → Ideas](https://github.com/MicroPerceptron/virtio-accel/discussions/categories/ideas)
- Suspected vulnerabilities → **not** a public issue; follow [SECURITY.md](SECURITY.md)

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option.

Contributions are dual-licensed on the same terms, with no separate CLA.
