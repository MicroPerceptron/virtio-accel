# virtio-accel-coreml

A host-native [`virtio_accel_core::Accelerator`](https://docs.rs/virtio-accel-core)
implementation for Apple's Core ML runtime, GPU and Neural Engine.

The backend requires macOS 14 or newer. `CoreMlAccelerator::new_tosa()` preserves the
`CpuAndNeuralEngine` default. `new_tosa_with_compute_units(CoreMlComputeUnits::CpuAndGpu)`
allows CPU/GPU placement, and `CoreMlComputeUnits::All` allows CPU/GPU/ANE placement.
Discovery requires the selected accelerator (both GPU and ANE for `All`). Device identity
reports NPU, GPU, or OTHER respectively; these are policy endpoints, not separate chips.
Core ML decides per-operation placement, with no promise of exclusive accelerator use or
simultaneous GPU/ANE activity. The legacy host-path constructor remains CPU/ANE only.

The floating-point NeuralNetwork format does not enforce intermediate precision on ANE.
To preserve FP32 graph arithmetic, programs containing FP32 tensors use CPUOnly under
the CPU/ANE policy, and CPUAndGPU under `All`. FP16 programs retain the full selected
policy. GPU low-precision accumulation is disabled. This selection happens before model
loading within the permitted compute units; it never retries a failed program elsewhere.
Exact integer ML Programs retain their explicit types and selected policy.

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

A whole-program `IDENTITY` is the deliberate exception: it is an asynchronous exact backing-to-
backing copy, not Core ML arithmetic. This preserves TOSA's byte-level identity semantics for NaNs,
signed zero, and subnormals on Core ML runtimes that canonicalize those values.

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

`TosaCapabilityProvider` exposes these as distinct floating and integer descriptors. The native
instance includes the integer, FP8, and mixed quantized descriptors only when the host runtime reports macOS 26+ INT8 ML
Program support; non-macOS placeholders expose no descriptors.

### Low-precision boundary

`supports_tosa_dtype` exposes model-boundary encoding capability independently of operator
coverage. FP16, FP32, INT8, and restricted INT32 outputs are encoded. INT8 requires the ML Program
path and macOS 26 or newer; older runtimes reject `COREML_TOSA_INTEGER_TARGET` before compilation.
The native suite executes identity across all signed-byte edge values and a non-square MATMUL with
nonzero zero points against the shared exact Rust oracle.

`COREML_TOSA_FP8_TARGET` adds E4M3/E5M2 weights and activations on macOS 26+.
Buffers retain one byte per element through an INT8 bit carrier; typed ML Program operations
perform decoding and direct FP32 round-to-nearest-even encoding. CAST, FP8 MATMUL (FP16
result), constants, and packed identity/reshape/transpose/concat/slice are admitted. ADD/MUL
operate on explicitly widened FP16/FP32 values. Cache append/window graphs preserve every
storage bit, including signed zero and NaN encodings. These are tensor primitives, not a
paged-attention or KV-cache allocator.

`COREML_TOSA_QUANTIZED_TARGET` admits the explicit NVFP4 projection graphs from
`virtio_accel_tosa_build::quantized`: two E2M1 values per byte, one E4M3 scale per 16 weights,
and a separate tensor scale. FP32 and explicit FP16 activation/result variants are available.
Only byte-derived lookup indices with a static bounds proof are accepted. Packed INT4 tensor
semantics and arbitrary dynamic gather indices remain unsupported.

These paths keep packed boundary storage and decode in the submitted graph. Native compilers
may materialize wider intermediate weights; this is not a native FP4 arithmetic or compressed
resident-weight claim. Typed ML Programs retain the selected compute policy, but Core ML may
place operations on CPU within that policy. See [ADR 0014](../../docs/adr/0014-packed-quantized-inference.md).

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
