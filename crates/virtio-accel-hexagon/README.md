# virtio-accel-hexagon

An in-progress Qualcomm Hexagon NPU host backend for `virtio-accel`. The portable portion already
validates device-neutral TOSA 1.0 artifacts and creates an owned, fixed-slot graph plan for the
initial FP16 identity, non-square batched MATMUL, and NHWC MAX_POOL2D tier.

**Current status:** portable lowering and build/runtime detection are implemented. Native QNN HTP
graph construction and execution remain unavailable until the full public QAIRT/QNN C development
package is supplied. The smaller public QAIRT AppBuilder/Genie bundle does not include the QNN C
headers or Windows ARM64 `QnnHtp` application import library and is not treated as sufficient.

**Portability tier:** `host-native` when completed; an explicit compile-only unavailable-runtime
placeholder on SDK-free hosts. This crate is not a dependency of the facade or any portable crate.

## Initial support boundary

The first target is Windows 11 ARM64 on Snapdragon X Series using QAIRT/QNN's HTP backend. The
portable admission layer currently accepts:

- TOSA 1.0, floating-point profile, level 8K, no extensions;
- static, positive-shape FP16 tensor boundaries;
- one region and one basic block with no runtime obligations;
- identity;
- non-square batched MATMUL with serialized FP16 zero points equal to positive or negative zero;
- NHWC MAX_POOL2D with propagating NaNs, a two-dimensional positive kernel/stride, and zero padding.

FP32 is rejected until HTP hardware evidence proves that the selected runtime preserves FP32
semantics without relaxed or forced FP16 computation. INT8, INT4, FP8, dynamic shapes, additional
operators, profiles, extensions, layouts, and attributes are also rejected during program
admission. The backend must never substitute CPU or GPU execution.

## SDK detection

The build script recognizes the following variables:

- `VIRTIO_ACCEL_HEXAGON=0` forces the portable placeholder;
- `VIRTIO_ACCEL_HEXAGON=1` requires a complete supported SDK and fails loudly otherwise;
- `VIRTIO_ACCEL_QNN_SDK_ROOT` selects the QAIRT/QNN SDK root;
- `QNN_SDK_ROOT` is accepted as the conventional fallback root;
- `VIRTIO_ACCEL_QNN_LIB_DIR` overrides the Windows ARM64 QNN import-library directory.

Autodetection requires Windows ARM64, a public `QnnInterface.h`, and a `QnnHtp` import library. A
driver-only or inference-only installation does not enable native modules.

## Running

```sh
cargo test -p virtio-accel-hexagon
cargo run -p virtio-accel-hexagon --example tosa_hexagon
```

Without the full QNN development runtime, the example reports why native execution is unavailable
and exits successfully.

## Hardware baseline discovered during bring-up

The initial development host is a Snapdragon X126100 Windows ARM64 system. Windows enumerates
`Snapdragon(R) X - X126100 - Qualcomm(R) Hexagon(TM) NPU` in the `ComputeAccelerator` class with
Qualcomm driver `30.0.222.0` dated 2026-04-01. The driver package includes the V73 HTP stub/skel and
prepare components, but not the public application-facing QNN C development surface.

No runtime/device row in the repository support matrix should change from `Planned` until the
native path passes semantic conformance and the advertised numerical corpus on that NPU.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option.
