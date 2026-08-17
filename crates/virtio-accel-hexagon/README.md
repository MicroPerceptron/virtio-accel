# virtio-accel-hexagon

Qualcomm Hexagon NPU host backend for `virtio-accel`. On Windows ARM64, a complete QAIRT/QNN SDK
enables direct QNN HTP graph execution. Other builds retain the portable TOSA admission tests and
an explicit `RuntimeUnavailable` placeholder. This crate is not a dependency of the facade or any
portable crate.

## Validated baseline

The initial hardware baseline is:

- Snapdragon X126100, Hexagon HTP v73;
- Windows 11 ARM64, Qualcomm NPU driver `30.0.222.0`;
- QAIRT `2.49.0.260730`;
- QNN core API `2.38.0`, HTP backend API `5.49.0`.

The backend accepts static, positive-shape FP16 TOSA 1.0 graphs containing identity, non-square
batched MATMUL, and zero-padded NHWC MAX_POOL2D. MAX_POOL2D is expressed as QNN Gather plus
ElementWiseMaximum nodes because this QAIRT Windows HTP build rejects its documented native
PoolMax2d tensor parameters at graph finalization. All resulting nodes still execute through the
HTP backend; there is no CPU or GPU fallback.

FP32, quantized types, dynamic shapes, additional operators, profiles, extensions, layouts, and
attributes are rejected during program admission. The initial execution lane permits one native
submission at a time. Finite submission timeouts and cancellation are not advertised because the
validated QNN HTP interface does not provide a working bounded asynchronous execution primitive.

## Install and configure QAIRT

Download the complete Qualcomm AI Runtime Community SDK, not an AppBuilder/Genie-only bundle. The
SDK root must contain both:

```text
include\QNN\QnnInterface.h
lib\aarch64-windows-msvc\QnnHtp.lib
```

For the validated archive, extraction to `C:\Qualcomm\AIStack\QAIRT\2.49.0.260730` gives the
following PowerShell setup:

```powershell
$sdk = 'C:\Qualcomm\AIStack\QAIRT\2.49.0.260730'
$env:VIRTIO_ACCEL_HEXAGON = '1'
$env:VIRTIO_ACCEL_QNN_SDK_ROOT = $sdk
$env:ADSP_LIBRARY_PATH = "$sdk\lib\hexagon-v73\unsigned"
```

`ADSP_LIBRARY_PATH` lets the Windows HTP stub locate the matching v73 DSP libraries. Do not commit
or redistribute Qualcomm SDK files from this repository.

The build probe also supports:

- `VIRTIO_ACCEL_HEXAGON=0` to force the portable placeholder;
- `QNN_SDK_ROOT` as a conventional SDK-root fallback;
- `VIRTIO_ACCEL_QNN_LIB_DIR` to override the Windows ARM64 import-library directory.

Forced-on mode fails immediately when the target is not Windows ARM64 or the SDK is incomplete.

## Manual hardware test

From the repository root in the configured PowerShell session:

```powershell
cargo check -p virtio-accel-hexagon
cargo test -p virtio-accel-hexagon --all-targets -- --test-threads=1
cargo run -p virtio-accel-hexagon --example tosa_hexagon
```

The integration tests initialize the pinned HTP provider, validate device identity, execute the
shared FP16 identity, batched non-square MATMUL, and NHWC MAX_POOL2D corpora, compare every result
with its numerical oracle, reject duplicate slots, wrong access, short bindings, and finite
timeouts, check stable terminal polling/output visibility, and verify that a live event keeps its
graph busy. The example prints the actual provider/build/API versions and ends with:

```text
TOSA FP16 identity -> QNN HTP v73: passed
```

To verify SDK-free portability in a fresh shell:

```powershell
$env:VIRTIO_ACCEL_HEXAGON = '0'
cargo test -p virtio-accel-hexagon --all-targets
```

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option.
