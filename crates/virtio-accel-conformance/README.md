# virtio-accel-conformance

A reusable, transport-free semantic conformance suite for virtio-accel backends, with
provider target, progress, and optional resource-accounting adapters. Third-party backend
authors run this suite to check their implementation against the normative semantics.

Part of the [`virtio-accel`](https://github.com/MicroPerceptron/virtio-accel) workspace: an experimental, native-Rust foundation for a
transport-neutral virtual accelerator device. The workspace contains no Linux ioctls, macOS
frameworks, Windows APIs, guest physical addresses, vendor command formats, or claimed virtio
device ID.

**Portability tier:** `std-reference` — Portable `std`; no host-OS or vendor-specific API.

## Documentation

- [Backend implementer guide](https://github.com/MicroPerceptron/virtio-accel/blob/main/docs/backend-implementer-guide.md)
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
