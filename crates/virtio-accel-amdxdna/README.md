# virtio-accel-amdxdna

An AMD XDNA (Ryzen AI NPU) host backend for `virtio-accel`, targeting XDNA2/Strix-class parts
through the HRX runtime (`libhrx`) with no XRT userspace dependency. It will execute device-neutral
TOSA 1.0 programs on the NPU, compiling admitted graphs with the pinned aiecc toolchain as a
bounded subprocess (never a Cargo dependency, never in-process Python).

**Portability tier:** `host-native` — the amdxdna-native HRX runtime (`libhrx`) when detected at
build time; a compile-only unsupported-runtime placeholder elsewhere.

**This crate is currently the scaffold.** It ships the portable admission surface and a
placeholder; the HRX FFI, the native `Accelerator` implementation, the compiler helper, and the
executing numerical tiers land in subsequent tickets of the
[AMD XDNA wayfinder map](https://github.com/MicroPerceptron/virtio-accel/issues/78). The design is
recorded in [`docs/research/amdxdna-crate-design.md`](../../docs/research/amdxdna-crate-design.md),
[`docs/research/amdxdna-compiler-helper-contract.md`](../../docs/research/amdxdna-compiler-helper-contract.md),
and [`docs/research/amdxdna-execution-model.md`](../../docs/research/amdxdna-execution-model.md);
the advertised numerical tier is
[`docs/adr/0001-amdxdna-first-numerical-tier.md`](../../docs/adr/0001-amdxdna-first-numerical-tier.md).

## Build-time probe

The boundary is a build-environment probe rather than a target operating system. HRX exposes a
plain C ABI, so there is no C++ bridge and no `cc`/CMake step. The build script resolves an HRX
install prefix from `VIRTIO_ACCEL_HRX_DIR` (highest priority) or `HRX_DIR` (the variable the pinned
toolchain `env.sh` and the fork's tooling export), and enables the native modules only when that
prefix carries `include/hrx/hrx_runtime.h`, `include/hrx/hrx_amdxdna.h` (declaring
`hrx_amdxdna_executable_create` — its absence marks an older, incompatible libhrx generation), and
`lib/libhrx.so`.

`VIRTIO_ACCEL_AMDXDNA=1` makes a missing or incomplete runtime a loud build failure, `=0` forces
the placeholder, and `VIRTIO_ACCEL_HRX_LIB_DIR` links a bare lib directory that ships no headers.
No standard locations are scanned: silently discovering an unpinned libhrx would defeat the version
pin. Builds without the runtime still compile and unit-test the portable admission surface.

## Advertised numerical tier

Per [ADR-0001](../../docs/adr/0001-amdxdna-first-numerical-tier.md), the backend will advertise a
BF16 floating-point tier (TOSA `EXT-BF16`) and a separate integer tier, and reject FP32/FP16 at
admission rather than silently executing them as BF16. The two `Target` constants
(`AMDXDNA_TOSA_TARGET`, `AMDXDNA_TOSA_INTEGER_TARGET`) are already declared in `src/lower.rs`;
graph admission and execution wire onto them in later tickets.

## Running

```sh
cargo run -p virtio-accel-amdxdna --example tosa_amdxdna
cargo test -p virtio-accel-amdxdna
```

The example reports the scaffold state and exits successfully; the tests exercise the placeholder
and the advertised targets on every host.

Part of the [`virtio-accel`](https://github.com/MicroPerceptron/virtio-accel) workspace: an
experimental native-Rust protocol and implementation stack for a transport-neutral virtual
accelerator device. Portable crates contain no host-OS or vendor APIs; host integrations live in
separate adapter crates and never become their dependencies. The project claims no Virtio device
ID.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option.
