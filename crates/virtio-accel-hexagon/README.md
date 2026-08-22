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

The backend accepts two explicit TOSA 1.0 targets:

- the floating-point target supports 41 of the 42 operators shared by Core ML and OpenVINO, with
  FP16 tensors, BOOL conditions/results, and required INT32 indexing results; and
- the integer target supports bit-exact INT8 identity and nonzero-zero-point INT8 MATMUL with an
  INT32 accumulator/output.

Native instances expose these as distinct conservative descriptors through
`TosaCapabilityProvider`. SDK-free placeholders return no descriptors, so discovery never turns
the portable planner into a false execution claim.

The complete operator-to-QNN mapping, restrictions, and evidence are in the
[Hexagon operator matrix](../../docs/hexagon-operator-matrix.md). `ERF` is the sole shared-surface
exception because QAIRT 2.49's public operation definitions expose no ERF node.

INT8 tensors remain direct one-byte client storage. The QNN scale-offset encoding represents each
TOSA zero point without dequantizing through floating point. MAX_POOL2D is expressed as QNN Gather plus
ElementWiseMaximum nodes because this QAIRT Windows HTP build rejects its documented native
PoolMax2d tensor parameters at graph finalization. `REVERSE` uses a descending-index Gather, and
`REDUCE_PRODUCT` uses Gather plus ElementWiseMultiply because HTP v73 rejects the public ReduceProd
node. All resulting nodes still execute through the HTP backend; there is no CPU or GPU fallback.

FP32 is deliberately rejected. QAIRT accepts FLOAT_32 tensors and an FP32 graph-precision option,
but a v73 precision probe using inputs that differ only below FP16 precision returned the FP16-rounded
MATMUL result. Generic QNN FLOAT_8 also does not expose an unambiguous client-visible E4M3/E5M2
selection on this stack, so both TOSA FP8 targets remain rejected. Dynamic shapes, `ERF`, additional
operators, profiles, extensions, layouts, and unsupported attributes are rejected during program
admission. The execution lane permits one native submission at a time. Finite submission deadlines
expire before admission and cancellation is not advertised because this QNN HTP interface does not
provide a working bounded asynchronous execution primitive.

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
cargo test --release -p virtio-accel-hexagon --test hexagon `
  measures_warm_submission_and_completion_latency -- --ignored --nocapture --test-threads=1
cargo run -p virtio-accel-hexagon --example tosa_hexagon
cargo run -p virtio-accel-hexagon --example mock_classifier
```

The integration tests initialize the pinned HTP provider, validate device identity, execute all 41
advertised operators across constants/data movement, unary/activation, binary, BOOL/comparison,
selection, indexing/reduction, MATMUL, and MAX_POOL2D families, plus exact INT8 identity and
zero-point-aware MATMUL, and compare every result with its numerical oracle. They also run the
reusable backend semantic suite, including segmented transfers and
artifacts, allocation metadata, context isolation, binding validation, terminal stability,
pre-admission deadlines, and direct-binding diagnostics with zero hidden staging. The examples
print the actual provider/build/API versions and end with results such as:

```text
TOSA FP16 identity -> QNN HTP v73: passed
mock FP16 linear classifier -> QNN HTP v73: passed; output=[4400, bc00, 3c00, be00]
```

To verify SDK-free portability in a fresh shell:

```powershell
$env:VIRTIO_ACCEL_HEXAGON = '0'
cargo test -p virtio-accel-hexagon --all-targets
```

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option.
