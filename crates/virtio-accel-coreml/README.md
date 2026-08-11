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
`.mlmodel` file or `.mlmodelc` directory beneath that root and maps every model input/output feature
to a virtio-accel slot. Absolute paths, parent traversal, symlink escape, unmapped features, optional
features, non-`MLMultiArray` features, and incompatible aliased layouts are rejected.

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

Source `.mlmodel` files are compiled synchronously during `load_program`; compiled `.mlmodelc`
directories load directly. Core ML does not publish a finite model-residency ceiling, so artifacts
must declare `REQUIRED_RESIDENT_BYTES` (`u64::MAX`). This deliberately forces the device's aggregate
resident-program policy to opt into one Core ML model instead of pretending an unverifiable smaller
charge is exact.

## Direct buffers and events

The backend advertises host and shared memory. Both are page-aligned provider allocations. Program
bindings wrap the exact bound range in `MLMultiArray`, and outputs use `MLPredictionOptions` output
backings. Completion fails with `BackendError::Incompatible` if Core ML returns a different output
object, so a hidden provider-side result copy is never reported as direct execution.

Prediction uses Core ML's asynchronous completion API. Events retain every Rust allocation until
the native callback reaches a terminal state. Event cancellation is not advertised because Core ML
does not expose cancellation for an admitted prediction.

The FFI and allocation invariants are documented in [SAFETY.md](SAFETY.md). Run the native
end-to-end and conformance tests on an ANE-capable Mac with:

```sh
cargo test -p virtio-accel-coreml
```

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option.
