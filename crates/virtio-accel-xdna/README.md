# virtio-accel-xdna

An AMD XDNA (Ryzen AI NPU) host backend for `virtio-accel`, targeting XDNA2/Strix-class parts
through the HRX runtime (`libhrx`) with no XRT userspace dependency. It will execute device-neutral
TOSA 1.0 programs on the NPU, compiling admitted graphs with the pinned aiecc toolchain as a
bounded subprocess (never a Cargo dependency, never in-process Python).

**Portability tier:** `host-native` — the amdxdna-native HRX runtime (`libhrx`) when detected at
build time; a compile-only unsupported-runtime placeholder elsewhere.

**This crate is under construction.** In a `va_xdna` build it runs the full `Accelerator`
lifecycle — device/stream owner, `hrx_buffer` primitives (persistent mapping, range
flush/invalidate, release), and a serialized dispatch worker bridging
`hrx_stream_dispatch`/`synchronize` to a latched nonblocking `poll_event`. `load_program` accepts
the crate-local precompiled artifact format directly, and a TOSA artifact by admitting it and
compiling it with the bounded aiecc helper subprocess (`compiler/xdna_compile.py`, run under the
pinned toolchain venv in a cleared environment, content-addressed in a cache). The compilable TOSA
subset today is BF16 IDENTITY (a DMA copy), BF16 → FP32 MATMUL (the spec-mandated FP32-accumulator
shape, batch 1, at multiples of the tested compute tile), and BF16 NHWC MAX_POOL2D. MAX_POOL2D is
deliberately bounded to batch 1, zero padding, propagating NaNs, kernel and stride dimensions no
larger than 8, and at most 8,192 input-plus-output elements so both tensors fit in the worker's
local-memory budget. All of this is exercised by `tests/hardware.rs` (a DMA passthrough, compiled
TOSA IDENTITY, bit-exact non-square MATMUL, and the shared MAX_POOL2D oracle) and by
`tests/conformance.rs` (the shared semantic suite, including the direct-binding copy-path
diagnostics, on the device). Hosts without HRX build the portable admission surface plus a
placeholder, compile no `unsafe`, and still unit-test admission and the artifact codec. The
remaining executing numerical tiers land in subsequent tickets of the
[AMD XDNA wayfinder map](https://github.com/MicroPerceptron/virtio-accel/issues/78); the design
decisions live on their ticket branches: crate layout (#83), FFI/buffers (#87), execution model
(#85), compiler helper (#84), and the advertised numerical tier (#82).

The compiler is never a Cargo dependency and never runs in-process. `compile_artifact` exposes the
admit-then-compile path without a device (the offline / catalog-population use), so a build host can
produce precompiled artifacts that a device-less serving host later loads. It is available on any
unix build — with or without HRX — since the offline build host is exactly a machine without
libhrx; only the pinned toolchain (`VIRTIO_ACCEL_AMDXDNA_TOOLCHAIN`) is needed at run time.

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

Per the numerical-tier decision (#82), the crate defines a BF16 floating-point target (TOSA
`EXT-BF16`) and a separate future integer target, and rejects FP32/FP16 compute at admission rather
than silently executing it as BF16. Native builds expose the implemented BF16 IDENTITY, MATMUL,
and MAX_POOL2D surface through `TosaCapabilityProvider`; placeholder builds expose an empty
capability list, and the integer target is not advertised through the provider until its execution
tier lands. MAX_POOL2D advertises the same propagating-NaN and zero-padding semantic constraints as
the OpenVINO reference backend, while admission applies the narrower XDNA2 shape and local-memory
envelope described above.

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
