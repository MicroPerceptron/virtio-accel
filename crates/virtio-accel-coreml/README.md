# virtio-accel-coreml

A host-native [`virtio_accel_core::Accelerator`](https://docs.rs/virtio-accel-core)
implementation for Apple's Core ML runtime and Neural Engine.

The backend requires macOS 14 or newer and refuses construction when Core ML does not report an
accessible `MLNeuralEngineComputeDevice`. Models are loaded with
`MLComputeUnitsCPUAndNeuralEngine`, which gives Core ML access to the ANE without permitting GPU
placement. Apple still decides placement per operation; a model containing unsupported ANE
operations can fall back to the CPU.

**Portability tier:** `host-native` — the real implementation is macOS-only. Other targets compile
a placeholder constructor that returns `InitError::UnsupportedPlatform`, keeping workspace and
cross-target dependency checks intact.

## Production TOSA artifacts

`CoreMlAccelerator::new_tosa()` accepts raw TOSA 1.0 FlatBuffers using
`virtio_accel_tosa::ARTIFACT_FORMAT`. Floating-point programs declare `COREML_TOSA_TARGET`; exact
INT8 programs declare `COREML_TOSA_INTEGER_TARGET`. Program loading verifies the bounded
FlatBuffer, runs the complete TOSA semantic and lowering analysis, emits a Core ML NeuralNetwork
or ML Program model in memory, and asks Core ML to compile and load it. A unique temporary `.mlmodel` source exists
only inside the native bridge for that synchronous compile and is removed before `load_program`
returns. No Core ML path, protobuf, feature name, or crate dependency crosses into the facade,
device, guest, transport, or queue crates.

The floating-point lowering tier accepts one static region and basic block with static `FP16`/`FP32`
boundary tensors (`INT32` outputs are also accepted for operators such as `ARGMAX`). It covers
identity and constants; floating-point unary, binary, comparison, logical, selection, clamp, and
reduction layers; batched matrix multiplication; unpadded NHWC max pooling through explicit NCHW
layout transposes; and concat, reshape, reverse, and transpose. `supports_tosa_operator` exposes the
operator set; unsupported attribute combinations such as nonzero pooling padding are rejected while
loading. Unsupported control flow, dynamic shapes, profiles, extensions, dtypes, and operators are
likewise rejected before admission. The separate integer-profile tier currently admits exact INT8
identity and INT8 batched matrix multiplication with INT32 output. MATMUL widens both operands to
INT32, explicitly subtracts the two TOSA zero points, and accumulates in INT32; it never converts
through floating point.

### Low-precision boundary

`supports_tosa_dtype` exposes model-boundary encoding capability independently of operator
coverage. FP16, FP32, INT8, and restricted INT32 outputs are encoded. INT8 requires the ML Program
path and macOS 26 or newer; older runtimes reject `COREML_TOSA_INTEGER_TARGET` before compilation.
The native suite executes identity across all signed-byte edge values and a non-square MATMUL with
nonzero zero points against the shared exact Rust oracle.

Packed INT4, FP8E4M3, and FP8E5M2 remain unsupported. Core ML INT4 facilities are compressed-weight
storage rather than TOSA INT4 tensor semantics, and Core ML exposes no FP8 tensor boundary. Those
artifacts are rejected without silent dequantization. Additional INT8 operators likewise require
explicit integer legalization and exact shared-oracle coverage before admission.

Bindings use a device-neutral deterministic rule: block inputs occupy slots `0..N` in declared
order and block outputs occupy `N..N+M`. Lowering assigns private Core ML feature and blob names;
portable callers never construct `CoreMlArtifact` or know those names.

Run the real TOSA-to-Core ML path on an ANE-capable Mac:

```sh
cargo run -p virtio-accel-coreml --example tosa_coreml
```

Core ML does not publish a finite model-residency ceiling, so TOSA artifacts must declare
`REQUIRED_RESIDENT_BYTES` (`u64::MAX`). This deliberately forces the device's aggregate
resident-program policy to opt into one Core ML model instead of pretending an unverifiable smaller
charge is exact.

## Host-owned Core ML compatibility path

`CoreMlAccelerator::new(model_root)` retains the original provider-specific path artifact for hosts
that already own `.mlmodel`, `.mlpackage`, or `.mlmodelc` assets. `CoreMlArtifact` paths remain
confined beneath that canonical root and are never the portable production format. Absolute paths,
parent traversal, symlink escape, unmapped features, optional features, non-`MLMultiArray` features,
and incompatible aliased layouts are rejected. Source assets compile synchronously; compiled
`.mlmodelc` directories load directly. Fixed-shape models receive Core ML's infrequent-reshape hint
on macOS 14.4+ and fast-prediction specialization on macOS 15+.

## Direct buffers and events

The backend advertises host and shared memory. Both are page-aligned provider allocations. Program
bindings wrap the exact bound range in `MLMultiArray`, and outputs use `MLPredictionOptions` output
backings. Completion verifies the returned output's data pointer, element type, shape, and strides;
a different Objective-C wrapper over the same exact storage remains valid, while a provider-side
result allocation fails with `BackendError::Incompatible`. Binding offsets must be aligned for the
model's scalar type.

Prediction uses Core ML's asynchronous completion API. Events retain every Rust allocation until
the native callback reaches a terminal state. Separate predictions may reuse a read-only input
allocation concurrently; any output or read-write binding retains exclusive native access. Host
transfers return `BackendError::Busy` while either access mode is active. Event cancellation is not
advertised because Core ML does not expose cancellation for an admitted prediction.

Program loading builds the sorted slot/access plan once. A queue retains reusable native-binding
scratch, so warm submission accepts arbitrary binding order without the former quadratic duplicate
scan or native-binding allocation. Only the event-owned, deduplicated backing guards are allocated
per admitted prediction; the native bridge performs a linear validation/wrapping pass and never
copies tensor contents. Cumulative direct-binding admissions and explicit-transfer bytes are
available through `direct_binding_admissions()` and `explicit_transfer_bytes()`.

The FFI and allocation invariants are documented in [SAFETY.md](SAFETY.md). Run the native
end-to-end and conformance tests on an ANE-capable Mac with:

```sh
cargo test -p virtio-accel-coreml
```

The native suite consumes the same numerical TOSA corpus exported by
`virtio-accel-conformance`. The FP16 and FP32 tiers check
non-finite values, subnormals and signed zero, non-square batched matrix multiplication, and
multi-channel NHWC max-pooling layout through Core ML. On macOS 26+, the INT8 tier additionally
checks bit-exact identity and INT32-accumulating MATMUL. The suite also checks overlapping asynchronous
predictions and repeated compile/unload source cleanup. Future host backends inherit these exact
artifacts and oracles instead of substituting provider-specific graphs.

For local warm-path latency evidence, run the ignored release-mode measurement:

```sh
cargo test --release -p virtio-accel-coreml \
  measures_warm_submission_and_completion_latency -- --ignored --nocapture
```

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option.
