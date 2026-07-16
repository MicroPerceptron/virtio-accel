# virtio-accel

`virtio-accel` is an experimental, native-Rust foundation for a transport-neutral virtual
accelerator device. The first target is NPU execution, while the core model deliberately leaves
room for GPUs, DSPs, and other program-driven accelerators.

The repository currently concentrates on the portable majority of the system. It contains no
Linux ioctls, macOS frameworks, Windows APIs, guest physical addresses, vendor command formats, or
claimed virtio device ID.

## Workspace

- `virtio-accel-proto`: `no_std`, pointer-free, little-endian draft wire structures.
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

## Current draft surface

The protocol defines fixed headers and payloads for device discovery, contexts, buffers, programs,
queues, submissions, and events. Unknown opcodes remain plain integers until validated, so decoding
untrusted bytes never creates an invalid Rust enum.

Core execution is asynchronous at the ownership boundary: a successful submit returns an event,
while an indeterminate failure must also return an event that retains the operation's resources.
Guest-visible object IDs are opaque, kind-tagged, generational, and never reused after generation
exhaustion.

The [portable protocol foundations](docs/specification.md) define the normative v1 terminology,
object model, compatibility rules, and mandatory baseline. See
[docs/architecture.md](docs/architecture.md) for the implementation invariants and
[docs/portability.md](docs/portability.md) for the enforced target matrix.

## Development

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-targets --all-features
```

The protocol is pre-standardization work. Numeric opcodes, feature bits, and payload layouts may
change until the device model is validated and a virtio specification proposal begins.
