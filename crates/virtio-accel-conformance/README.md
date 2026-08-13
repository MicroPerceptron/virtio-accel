# virtio-accel-conformance

A reusable, transport-free semantic conformance suite for virtio-accel backends, with
provider target, progress, and optional resource-accounting adapters. Third-party backend
authors run this suite to check their implementation against the normative semantics.

The `numerics` module also publishes stable TOSA 1.0 acceptance artifacts with FP32, FP16, both
TOSA FP8 encodings, INT8, and packed INT4 tensors. The floating-point corpus covers non-finite,
subnormal, and signed-zero identity behavior; FP16/FP32 also cover non-square batched matrix
multiplication and multi-channel NHWC max pooling. Binary16 values use exact `u16` bits and FP8 and
integer values use their exact packed bytes, preserving the workspace's stable Rust 1.85 baseline
without nightly numeric primitives. Hardware backends consume the same graph bytes and oracles,
making numerical and layout comparisons device-neutral rather than provider-specific. A fixture's
presence defines the shared contract, not backend support: each provider must explicitly accept or
reject its TOSA profile, extension, and dtype during program loading.

Part of the [`virtio-accel`](https://github.com/MicroPerceptron/virtio-accel) workspace: an experimental native-Rust protocol and implementation stack for a
transport-neutral virtual accelerator device. Portable crates contain no host-OS or vendor APIs;
host integrations live in separate adapter crates and never become their dependencies. The
project claims no Virtio device ID.

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
