# virtio-accel-cleanroom

An independent manual codec for the protocol 1.0 byte contract, implemented without any
shared protocol types. It exists to cross-check the primary `zerocopy` ABI: the bridge test
exchanges bytes only, so the conformance codec never becomes a production dependency.

Part of the [`virtio-accel`](https://github.com/MicroPerceptron/virtio-accel) workspace: an experimental, native-Rust foundation for a
transport-neutral virtual accelerator device. The workspace contains no Linux ioctls, macOS
frameworks, Windows APIs, guest physical addresses, vendor command formats, or claimed virtio
device ID.

**Portability tier:** `core-only` — `core` only; no `alloc`, no operating system, no host synchronization.

## Documentation

- [Conformance artifacts](https://github.com/MicroPerceptron/virtio-accel/blob/main/conformance/v1.0/README.md)
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
