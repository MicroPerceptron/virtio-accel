# virtio-accel-shared-queue

Both halves of the protocol 1.0 split virtqueue, laid out in memory a driver and a device
share. It implements the transport ports over the real layout (Virtio 1.2 §2.7) and reaches the
memory through one trait, `SharedMemory`, which the platform implements: a window a driver lent,
guest-physical memory, an IOMMU domain.

- `DriverRing` publishes chains and takes them back used. It never reads the descriptor table
  back, and checks every used entry before it believes it.
- `DeviceRing` takes chains from the available ring and uses them. It snapshots each descriptor
  once, bounds every index and address, and hands a malformed chain up to be completed empty.

No `unsafe`, no allocation, no lock: storage is inline and sized by const parameters, and the raw
memory access lives in the trait's implementation. The in-memory `virtio-accel-split-queue` model
is its oracle; this is the queue two address spaces can actually share.

Part of the [`virtio-accel`](https://github.com/MicroPerceptron/virtio-accel) workspace: an experimental native-Rust protocol and implementation stack for a
transport-neutral virtual accelerator device. Portable crates contain no host-OS or vendor APIs;
host integrations live in separate adapter crates and never become their dependencies. The
project claims no Virtio device ID.

**Portability tier:** `core-only` — `core` only; no `alloc`, no operating system, no host synchronization.

## Documentation

- [Command-chain rules](https://github.com/MicroPerceptron/virtio-accel/blob/main/docs/virtqueue.md)
- [Release and evolution policy](https://github.com/MicroPerceptron/virtio-accel/blob/main/docs/release-policy.md)
- [Portability and CI matrix](https://github.com/MicroPerceptron/virtio-accel/blob/main/docs/portability.md)
- [Security policy](https://github.com/MicroPerceptron/virtio-accel/blob/main/SECURITY.md)

The protocol 1.0 numeric opcodes, statuses, and payload layouts are versioned review inputs for
independent implementation, frozen for the portable v1.0 baseline by the
[final freeze audit](https://github.com/MicroPerceptron/virtio-accel/blob/main/conformance/v1.0/freeze-audit.md). This remains pre-standardization
work and claims no Virtio device ID.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option.
