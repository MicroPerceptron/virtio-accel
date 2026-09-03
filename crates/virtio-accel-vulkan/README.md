# virtio-accel-vulkan

A vendor-neutral Vulkan compute host backend for `virtio-accel`, executing device-neutral TOSA
1.0 programs on any conformant Vulkan 1.x implementation (RADV, ANV, NVIDIA, Mali, or a software
ICD) without leaking Vulkan types into the portable crates.

**Portability tier:** `host-native` — `ash` loads the platform's Vulkan loader dynamically at run
time on the enumerated host targets (Linux, Android, Windows, macOS); a compile-only placeholder
elsewhere. Unlike the SDK-probing backends, there is nothing to detect at build time (ADR 0002 in
`docs/adr/`).

**This crate is currently the scaffold.** It ships the always-compiled admission constants and a
placeholder; the `ash` FFI, the native `Accelerator` implementation, and the advertised numerical
tiers land in subsequent tickets of the
[Vulkan wayfinder map](https://github.com/MicroPerceptron/virtio-accel/issues/154). The design
decisions ratified with this scaffold live in `docs/adr/` (ADRs 0001–0004).

## Build-time gate

`VIRTIO_ACCEL_VULKAN=1` makes an unsupported target a loud build failure, `=0` forces the
placeholder, and unset is auto. The supported set is enumerated in `build.rs`; runtime presence of
a Vulkan loader is discovered when the native backend initializes, never at build time.

## Planned numerical tier

Per ADR 0004, the backend plans an FP32 base tier (`VULKAN_TOSA_TARGET`) and a provisional INT8
tier (`VULKAN_TOSA_INTEGER_TARGET`); FP16 is deferred until per-device float-controls evidence
exists, and FP8 is rejected at admission. Graph admission and execution wire onto these targets in
the tickets following the scaffold.

## Running

```sh
cargo run -p virtio-accel-vulkan --example tosa_vulkan
cargo test -p virtio-accel-vulkan
```

The example reports the scaffold state and exits successfully; the tests exercise the advertised
targets on every host.

Part of the [`virtio-accel`](https://github.com/MicroPerceptron/virtio-accel) workspace: an
experimental native-Rust protocol and implementation stack for a transport-neutral virtual
accelerator device. Portable crates contain no host-OS or vendor APIs; host integrations live in
separate adapter crates and never become their dependencies. The project claims no Virtio device
ID.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option.
