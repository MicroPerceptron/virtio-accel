# virtio-accel-device

Transport-neutral device-side state and validation, including bounded generational object
IDs. A future rust-vmm adapter translates descriptor chains into this layer; the command
engine and provider backend never see guest addresses or virtqueue descriptors.

Part of the [`virtio-accel`](https://github.com/MicroPerceptron/virtio-accel) workspace: an experimental native-Rust protocol and implementation stack for a
transport-neutral virtual accelerator device. Portable crates contain no host-OS or vendor APIs;
host integrations live in separate adapter crates and never become their dependencies. The
project claims no Virtio device ID.

**Portability tier:** `alloc-portable` — `core + alloc`; no operating system, filesystem, sockets, or threads.

## Documentation

- [Implementation invariants](https://github.com/MicroPerceptron/virtio-accel/blob/main/docs/architecture.md)
- [Threat model](https://github.com/MicroPerceptron/virtio-accel/blob/main/docs/threat-model.md)
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
