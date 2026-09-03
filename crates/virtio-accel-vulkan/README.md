# virtio-accel-vulkan

A vendor-neutral Vulkan compute host backend for `virtio-accel`, executing device-neutral TOSA
1.0 programs on any conformant Vulkan 1.3 implementation (RADV, ANV, NVIDIA, Mali, or a software
ICD such as lavapipe) without leaking Vulkan types into the portable crates.

**Portability tier:** `host-native` — the pinned `ash` crate loads the platform's Vulkan loader
dynamically at run time on the enumerated host targets (Linux, Android, Windows, macOS); a
compile-only placeholder elsewhere. Unlike the SDK-probing backends, there is nothing to detect at
build time (ADR 0002 in `docs/adr/`).

## What executes today

- **FP32 IDENTITY** (`VULKAN_TOSA_TARGET`, `VULKAN_TOSA_CAPABILITY`): the single-operator TOSA
  graph, admitted hardware-free in `lower` and executed by a checked-in SPIR-V word-copy kernel
  specialized with the element count at `load_program`. Guest bytes never reach the driver's
  shader compiler (ADR 0003).
- **Memory domains** (ADR 0005): `Host` and `Shared` are persistently mapped host-coherent
  allocations; `Device` is device-local memory reached only through bounded staging inside
  `write_buffer`/`read_buffer`. `Shared` and `Device` are advertised only when the device exposes
  a matching memory type. Every buffer is a dedicated allocation bound directly as a storage
  buffer; alignment is measured, never assumed.
- **Execution** (ADR 0006): a bounded per-context ring of (command buffer, fence, descriptor set)
  triples; `vkQueueSubmit2` success is the admission boundary; `poll_event` is one
  `vkGetFenceStatus` read with no worker thread; finite timeouts are rejected before admission;
  `VK_ERROR_DEVICE_LOST` poisons the instance.
- **Diagnostics:** `direct_binding_admissions`, `explicit_transfer_bytes`, and `live_resources`
  feed the conformance suite's copy-path and accounting hooks.

The provisional integer target (`VULKAN_TOSA_INTEGER_TARGET`) is declared but not advertised; FP16
is undeclared until per-device float-controls evidence closes wayfinder ticket 5 (ADR 0004).

## Build-time gate

`VIRTIO_ACCEL_VULKAN=1` makes an unsupported target a loud build failure, `=0` forces the
placeholder, and unset is auto. The supported set is enumerated in `build.rs`; runtime presence of
a Vulkan 1.3 loader and a compute-capable device is discovered when the backend initializes and
reported as `InitError`.

## Running

```sh
cargo run -p virtio-accel-vulkan --example tosa_vulkan
cargo test -p virtio-accel-vulkan
```

The example executes the FP32 identity artifact on the preferred device (discrete, integrated,
virtual, then CPU) and exits successfully, or reports that no device is available. The native tests
run against every enumerated device and skip without one; `VIRTIO_ACCEL_VULKAN_REQUIRE_DEVICE=1`
turns absence into a failure, and `VK_DRIVER_FILES` pins the ICD (the CI lane pins lavapipe).

Part of the [`virtio-accel`](https://github.com/MicroPerceptron/virtio-accel) workspace: an
experimental native-Rust protocol and implementation stack for a transport-neutral virtual
accelerator device. Portable crates contain no host-OS or vendor APIs; host integrations live in
separate adapter crates and never become their dependencies. The project claims no Virtio device
ID.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option.
