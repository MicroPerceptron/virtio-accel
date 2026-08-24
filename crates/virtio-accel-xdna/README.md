# virtio-accel-xdna

An AMD XDNA (Ryzen AI NPU) host backend for `virtio-accel`, targeting XDNA2/Strix-class parts
through the HRX runtime (`libhrx`) with no XRT userspace dependency. It will execute device-neutral
TOSA 1.0 programs on the NPU, compiling admitted graphs with the pinned aiecc toolchain as a
bounded subprocess (never a Cargo dependency, never in-process Python).

**Portability tier:** `host-native` — the amdxdna-native HRX runtime (`libhrx`) when detected at
build time; a compile-only unsupported-runtime placeholder elsewhere.

**This crate is under construction.** In a `va_xdna` build it runs the full `Accelerator`
lifecycle for a *precompiled* artifact — device/stream owner, `hrx_buffer` primitives (persistent
mapping, range flush/invalidate, release), and a serialized dispatch worker bridging
`hrx_stream_dispatch`/`synchronize` to a latched nonblocking `poll_event` — validated on-device by
the `tests/hardware.rs` suite (an end-to-end DMA passthrough on the NPU). `load_program` accepts
the crate-local precompiled artifact format; TOSA compilation via the aiecc compiler helper, and
the executing numerical tiers, land in subsequent tickets of the
[AMD XDNA wayfinder map](https://github.com/MicroPerceptron/virtio-accel/issues/78). Hosts without
HRX build the portable admission surface plus a placeholder and compile no `unsafe`. The design
decisions live on their ticket branches: crate layout (#83), FFI/buffers (#87), execution model
(#85), compiler helper (#84), and the advertised numerical tier (#82).

## Build-time probe

The boundary is a build-environment probe rather than a target operating system. HRX exposes a
plain C ABI, so there is no C++ bridge and no `cc`/CMake step. The build script resolves an HRX
install prefix from `VIRTIO_ACCEL_HRX_DIR` (highest priority) or `HRX_DIR` (the variable the pinned
toolchain `env.sh` and the fork's tooling export), and enables the native modules only when that
prefix carries `include/hrx/hrx_runtime.h`, `include/hrx/hrx_amdxdna.h` (declaring
`hrx_amdxdna_executable_create` — its absence marks an older, incompatible libhrx generation), and
`lib/libhrx.so`.

`VIRTIO_ACCEL_XDNA=1` makes a missing or incomplete runtime a loud build failure, `=0` forces the
placeholder, and `VIRTIO_ACCEL_HRX_LIB_DIR` links a bare lib directory that ships no headers. No
standard locations are scanned: silently discovering an unpinned libhrx would defeat the version
pin. Builds without the runtime still compile and unit-test the portable admission surface.

## Advertised numerical tier

Per the numerical-tier decision (#82), the backend will advertise a BF16 floating-point tier (TOSA
`EXT-BF16`) and a separate integer tier, and reject FP32/FP16 at admission rather than silently
executing them as BF16. The two `Target` constants (`XDNA_TOSA_TARGET`, `XDNA_TOSA_INTEGER_TARGET`)
are already declared in `src/lower.rs`; graph admission and execution wire onto them in later
tickets.

## Running

```sh
cargo run -p virtio-accel-xdna --example tosa_xdna
cargo test -p virtio-accel-xdna
```

Without HRX the example reports the placeholder state; in a `va_xdna` build it initializes the
device and stream. The portable tests exercise the advertised targets on every host; the
`va_xdna` `tests/hardware.rs` suite exercises the buffer primitives against a live NPU.

Part of the [`virtio-accel`](https://github.com/MicroPerceptron/virtio-accel) workspace: an
experimental native-Rust protocol and implementation stack for a transport-neutral virtual
accelerator device. Portable crates contain no host-OS or vendor APIs; host integrations live in
separate adapter crates and never become their dependencies. The project claims no Virtio device
ID.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option.
