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

## Model artifacts

`CoreMlAccelerator::new(model_root)` establishes a host-controlled root. A `CoreMlArtifact` names a
`.mlmodel` file, `.mlpackage` directory, or `.mlmodelc` directory beneath that root and maps every
model input/output feature to a virtio-accel slot. Absolute paths, parent traversal, symlink escape,
unmapped features, optional features, non-`MLMultiArray` features, and incompatible aliased layouts
are rejected.

```rust,no_run
# #[cfg(target_os = "macos")]
# fn example() -> Result<(), Box<dyn std::error::Error>> {
use virtio_accel_coreml::CoreMlArtifact;

let artifact = CoreMlArtifact::new("models/TwicePlusOne.mlmodel")?
    .map_input(7, "x")?
    .map_output(7, "y")?
    .encode()?;

// Input x and output y share slot 7, so submissions bind one ReadWrite range.
assert!(!artifact.is_empty());
# Ok(())
# }
```

Artifact ABI v1 supports nonoptional `MLMultiArray` features with `Float16`, `Float32`, `Float64`,
or `Int32` elements. Each feature uses the model constraint's declared default shape; alternate
flexible shapes are not selected by the artifact. Image, sequence, dictionary, scalar, optional,
and Core ML state features are rejected at model load rather than failing after admission. The
model's default function is used for multi-function assets.

Source `.mlmodel` files and `.mlpackage` directories are compiled synchronously during
`load_program`; compiled `.mlmodelc` directories load directly and are recommended for predictable
startup latency. Fixed-shape models receive Core ML's infrequent-reshape hint on macOS 14.4+ and the
fast-prediction specialization strategy on macOS 15+. Core ML does not publish a finite
model-residency ceiling, so artifacts must declare `REQUIRED_RESIDENT_BYTES` (`u64::MAX`). This
deliberately forces the device's aggregate resident-program policy to opt into one Core ML model
instead of pretending an unverifiable smaller charge is exact.

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

For local warm-path latency evidence, run the ignored release-mode measurement:

```sh
cargo test --release -p virtio-accel-coreml \
  measures_warm_submission_and_completion_latency -- --ignored --nocapture
```

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option.
