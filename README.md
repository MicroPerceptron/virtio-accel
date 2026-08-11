# virtio-accel

[![CI](https://github.com/MicroPerceptron/virtio-accel/actions/workflows/ci.yml/badge.svg)](https://github.com/MicroPerceptron/virtio-accel/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/virtio-accel.svg)](https://crates.io/crates/virtio-accel)
[![docs.rs](https://docs.rs/virtio-accel/badge.svg)](https://docs.rs/virtio-accel)
[![GitHub last commit](https://img.shields.io/github/last-commit/MicroPerceptron/virtio-accel.svg)](https://github.com/MicroPerceptron/virtio-accel/commits/main)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)
[![MSRV](https://img.shields.io/badge/rustc-1.85+-blue.svg)](#development)
[![no_std](https://img.shields.io/badge/no__std-supported-brightgreen.svg)](#portability)

Portable Rust foundations for a transport-neutral virtual accelerator device.

`virtio-accel` defines a protocol and a set of `no_std` Rust layers for exposing an accelerator to a
guest: contexts, buffers, opaque programs, execution queues, submissions, and events. The first
target is NPU execution, while the object model deliberately leaves room for GPUs, DSPs, and other
program-driven accelerators.

The repository is mostly portable protocol and lifecycle code. Host integrations are isolated in
adapter crates and never become dependencies of the portable facade; the first such adapter is the
macOS Core ML backend. The project claims no Virtio device ID.

This project is pre-standardization and experimental. Protocol 1.0 is frozen as a versioned review
input for independent implementation — it is stable enough to build against and to disagree with in
writing, not an approved Virtio specification.

## Workspace

| Crate | Tier | Role |
| --- | --- | --- |
| `virtio-accel` | `core + alloc` | Facade re-exporting the portable layers |
| `virtio-accel-proto` | `core` | Pointer-free, little-endian protocol 1.0 wire structures |
| `virtio-accel-transport` | `core` | Dependency-free descriptor-chain, queue, reset, and notification ports |
| `virtio-accel-core` | `core` | Backend lifecycle, memory, program, queue, and event contracts |
| `virtio-accel-coreml` | macOS `std` | Core ML backend with direct buffers and asynchronous ANE-capable prediction |
| `virtio-accel-split-queue` | `core + alloc` | Bounded in-memory split-ring reference model |
| `virtio-accel-guest` | `core + alloc` | Typed reference client with bounded request tracking |
| `virtio-accel-device` | `core + alloc` | Device-owned state, including bounded generational IDs |
| `virtio-accel-mock` | `std` | In-memory backend with deterministic test-only artifacts and scripted faults |
| `virtio-accel-conformance` | `std` | Transport-free semantic backend suite |
| `virtio-accel-cleanroom` | `core` | Independent conformance codec, written without the shared protocol types |

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
virtio-accel-coreml -------------------------> virtio-accel-core
other provider adapters --------------------> virtio-accel-core
```

The transport crate exposes reset-scoped chain identities, flattened direction/length metadata, and
owned publication/completion tokens. Neither it nor the device-state layer leaks guest addresses,
ring pointers, or concrete descriptor types into the command engine or provider backend.

## Install

```toml
[dependencies]
virtio-accel = "0.1"
```

The facade is `no_std`. Add the reference backend as a dev-dependency to run the example below:

```toml
[dev-dependencies]
virtio-accel-mock = "0.1"
```

On an ANE-capable Mac, add `virtio-accel-coreml = "0.1"` separately for the host-native Core ML
backend. It is intentionally not re-exported by the portable facade.

## Example

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
- **Failure still returns an event.** A successful submit returns an event; an *indeterminate*
  failure must also return one, because the operation's resources are still owned by the device.
  Guest-visible object IDs are opaque, kind-tagged, generational, and never reused after generation
  exhaustion.

The primary `zerocopy` ABI and the manual clean-room codec both decode and re-encode every canonical
frame. Their bridge test exchanges bytes only, providing an independent implementation check without
making the conformance codec a production dependency.

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

| Document | Covers |
| --- | --- |
| [specification.md](docs/specification.md) | Normative terminology, object model, compatibility rules, mandatory baseline |
| [wire-abi.md](docs/wire-abi.md) | Exact byte layouts and the coordinated change procedure |
| [virtqueue.md](docs/virtqueue.md) | Command-chain rules |
| [architecture.md](docs/architecture.md) | Implementation invariants |
| [threat-model.md](docs/threat-model.md) | Trust boundaries and finite resource policy |
| [portability.md](docs/portability.md) | Enforced target matrix and crate tiers |
| [performance.md](docs/performance.md) | v1 performance and copy budgets |
| [public-api.md](docs/public-api.md) | Public rustdoc policy |
| [release-policy.md](docs/release-policy.md) | Release governance and evolution rules |
| [backend-implementer-guide.md](docs/backend-implementer-guide.md) | Running the semantic suite against a new backend |
| [releases/v1.0.md](docs/releases/v1.0.md) | Protocol 1.0 release note |
| [conformance/v1.0](conformance/v1.0/README.md) | Golden artifacts, canonical frames, and the [freeze audit](conformance/v1.0/freeze-audit.md) |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Development gates, protocol change classification, and scope boundaries |
| [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) | Expected conduct in project spaces |
| [SECURITY.md](SECURITY.md) | Reporting a vulnerability |

## Portability

Every portable and reference crate is `#![forbid(unsafe_code)]`. The audited Core ML adapter keeps
its unsafe FFI isolated to macOS. CI enforces each portability tier, including compile-only checks
of the adapter's unsupported-platform surface.

| Tier | Allowed runtime surface |
| --- | --- |
| `core` | `core` only; no allocation |
| `core + alloc` | `core + alloc`; no OS, filesystem, sockets, threads, or host synchronization |
| `std` | Portable `std`; no host-OS or vendor-specific API |
| macOS `std` | Host-native Core ML/Foundation adapter; never a portable default dependency |

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
- concrete VMM, kernel, OS, and vendor SDK adapters
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
