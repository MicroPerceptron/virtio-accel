# virtio-accel-transport

Dependency-free descriptor-chain, virtqueue, reset, and notification port contracts. It
exposes reset-scoped chain identities, flattened direction/length metadata, and owned
publication/completion tokens without leaking guest addresses or concrete descriptor types.

Part of the [`virtio-accel`](https://github.com/MicroPerceptron/virtio-accel) workspace: an experimental, native-Rust foundation for a
transport-neutral virtual accelerator device. The workspace contains no Linux ioctls, macOS
frameworks, Windows APIs, guest physical addresses, vendor command formats, or claimed virtio
device ID.

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
