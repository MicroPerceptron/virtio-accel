# virtio-accel

`virtio-accel` is an experimental, native-Rust foundation for a transport-neutral virtual
accelerator device. The first target is NPU execution, while the core model deliberately leaves
room for GPUs, DSPs, and other program-driven accelerators.

The repository currently concentrates on the portable majority of the system. It contains no
Linux ioctls, macOS frameworks, Windows APIs, guest physical addresses, vendor command formats, or
claimed virtio device ID.

## Workspace

- `virtio-accel-proto`: `no_std`, pointer-free, little-endian protocol 1.0 wire structures.
- `virtio-accel-core`: `no_std` backend lifecycle, memory, program, queue, and event contracts.
- `virtio-accel-device`: `no_std + alloc` device-owned state, including bounded generational IDs.
- `virtio-accel-mock`: cross-platform in-memory backend that exercises the complete lifecycle.
- `virtio-accel`: small `no_std` facade re-exporting the portable layers.

The crate dependency direction is:

```text
transport adapters (future: rust-vmm, bare metal, tests)
                         |
                         v
              virtio-accel-device
                 /             \
                v               v
    virtio-accel-proto   virtio-accel-core
                                  |
                                  v
                    provider adapters (future)
```

Neither the wire protocol nor the device-state layer is allowed to leak transport descriptors or
guest addresses into a provider backend.

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
