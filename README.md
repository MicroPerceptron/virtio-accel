# virtio-accel

`virtio-accel` is an experimental, native-Rust foundation for a transport-neutral virtual
accelerator device. The first target is NPU execution, while the core model deliberately leaves
room for GPUs, DSPs, and other program-driven accelerators.

The repository currently concentrates on the portable majority of the system. It contains no
Linux ioctls, macOS frameworks, Windows APIs, guest physical addresses, vendor command formats, or
claimed virtio device ID.

## Workspace

- `virtio-accel-proto`: `no_std`, pointer-free, little-endian protocol 1.0 wire structures.
- `virtio-accel-transport`: dependency-free `no_std` descriptor-chain, queue, reset, and
  notification ports.
- `virtio-accel-split-queue`: `no_std + alloc` bounded in-memory split-ring reference model.
- `virtio-accel-guest`: `no_std + alloc` typed reference client with bounded request tracking.
- `virtio-accel-core`: `no_std` backend lifecycle, memory, program, queue, and event contracts.
- `virtio-accel-device`: `no_std + alloc` device-owned state, including bounded generational IDs.
- `virtio-accel-mock`: cross-platform in-memory backend with deterministic test-only artifacts,
  harness-controlled execution, and scripted ownership-boundary faults.
- `virtio-accel-cleanroom`: dependency-free `no_std` conformance codec implemented without shared
  protocol types.
- `virtio-accel`: small `no_std` facade re-exporting the portable layers.

The crate dependency direction is:

```text
virtio-accel-split-queue ---> virtio-accel-transport
                                      ^
                                      |
virtio-accel-device ----------+-------+------> virtio-accel-core
          |                                          |
          +-----> virtio-accel-proto                 v
                                             provider adapters

virtio-accel-guest -----------> virtio-accel-transport
          |
          +--------------------> virtio-accel-proto
```

Arrows point from a crate to its dependency. The transport crate exposes reset-scoped chain
identities, flattened direction/length metadata, and owned publication/completion tokens. Neither
it nor the device-state layer leaks guest addresses, ring pointers, or concrete descriptor types
into the command engine or provider backend.

## Protocol 1.0 candidate surface

The portable protocol 1.0 candidate defines fixed headers and payloads for device discovery,
contexts, buffers, programs, execution queues, submissions, and events. Unknown opcodes, statuses,
and event states remain raw integers until validated, so decoding untrusted bytes never creates an
invalid Rust enum.

Core execution is asynchronous at the ownership boundary: a successful submit returns an event,
while an indeterminate failure must also return an event that retains the operation's resources.
Guest-visible object IDs are opaque, kind-tagged, generational, and never reused after generation
exhaustion.

The [portable protocol foundations](docs/specification.md) define the normative terminology, object
model, compatibility rules, and mandatory baseline. The exact byte layouts live in
[docs/wire-abi.md](docs/wire-abi.md), the command-chain rules in
[docs/virtqueue.md](docs/virtqueue.md), and independent golden artifacts under
[conformance/v1.0](conformance/v1.0/README.md). See
[docs/architecture.md](docs/architecture.md) for implementation invariants and
[docs/portability.md](docs/portability.md) for the enforced target matrix.

The primary `zerocopy` ABI and the manual clean-room codec both decode and re-encode every canonical
frame. Their bridge test exchanges bytes only, providing an independent implementation check
without making the conformance codec a production dependency.

## Development

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-targets --all-features
```

The project remains pre-standardization work and claims no Virtio device ID. Protocol 1.0 numeric
opcodes, statuses, and payload layouts are versioned review inputs for independent implementation.
They remain pre-release candidates until the final freeze audit in
[issue #33](https://github.com/MicroPerceptron/virtio-accel/issues/33). Candidate changes must follow
the coordinated change procedure in [docs/wire-abi.md](docs/wire-abi.md); after the freeze,
incompatible changes require a new protocol major version.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT)
at your option.
