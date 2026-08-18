# virtio-accel-vaccel

An adapter boundary crate for integrating external execution backends (including vAccel-like
providers) into `virtio-accel` via the portable [`virtio_accel_core::Accelerator`] contract.

The adapter is intentionally small and host-only (`std`): it owns the boundary, error and
lifetime semantics, and a best-effort submission-copy visibility signal, while leaving provider
APIs and external dependencies outside the portable layer.

## Setup

Add `virtio-accel-vaccel` beside `virtio-accel` in your adapter crate and keep your native provider
backend behind a separate concrete type:

```toml
[dependencies]
virtio-accel = { version = "0.3", path = "...", default-features = false }
virtio-accel-core = { version = "0.3", path = "...", default-features = false }
virtio-accel-vaccel = { version = "0.3", path = "..."}
```

Use this crate to wrap any backend that already implements
[`virtio_accel_core::Accelerator`]. The wrapper forwards all lifecycle and execution operations
without adding transport-specific assumptions.

## What this crate provides

- `VAccelAdapter` wraps a concrete backend that already implements
  [`virtio_accel_core::Accelerator`].
- `Accelerator` calls pass through to the wrapped backend.
- Adapter-level counters for:
  - `direct_binding_admissions()`
  - `explicit_transfer_bytes()`

The counters let conformance hooks report copy-path visibility even before native counters are
added to an upstream integration.

For full native visibility, providers should expose their own counters in their own adapters and add
custom hooks that read those values directly.

## Quickstart

Use this crate as the stable boundary around your concrete vAccel backend type:

    let report = virtio_accel_conformance::run(
        || VAccelAdapter::new(MyBackend::new(/* ... */)),
        &my_target,
        &my_hooks,
    );

## Representative validation path

The crate currently ships a representative conformance path against the in-repo mock backend to keep the
adapter seam itself continuously testable:

```sh
cargo run -p virtio-accel-vaccel --example backend_conformance
```

The example uses `virtio_accel_mock` internally and is intentionally independent from host/VMM/runtime
dependencies.

## Profiles

### Mock-contract profile

Use this when you are testing adapter wiring and conformance in CI without native provider runtime.

- Add `virtio-accel-mock` to your workspace.
- Run:

```sh
cargo run -p virtio-accel-vaccel --example backend_conformance
```

### Native adapter profile

Use this profile to connect a concrete provider type behind the adapter. Keep provider SDK/FFI
integration in your parent crate; only the wrapped type crosses into `VAccelAdapter`.

- Keep the concrete backend in a separate module or crate.
- Add adapter-specific counters in that concrete type when available.
- Optionally expose a richer [`SubmissionPathDiagnostics`](../virtio-accel-conformance/src/lib.rs)-style view
  in your local harness.

### Production host profile

Use host backends (OpenVINO, Core ML, Hexagon) as you already would, but keep the adapter contract
separation: portable crates do not import native runtimes, and runtime selection remains at the
application layer.

## Integration plan for native vAccel backends

1. Build a concrete backend type that maps native vAccel handles, lifecycle, and errors into the
   [`virtio_accel_core::Accelerator`] trait surface.
2. Wrap that backend with `VAccelAdapter::new(...)`.
3. Reuse the existing conformance hooks by sourcing submission-path diagnostics either from
   provider-native counters (preferred) or the adapter fallbacks.
4. Document any remaining exceptions where submission staging is unavoidable for that native backend.

## Limitations

- This crate intentionally does **not** introduce host/VMM/runtime dependencies itself.
- Adapter-level `direct_binding_admissions()` is a fallback admission counter and may be superseded by
  native provider counters for stronger evidence.
- `staged_direct_bindings` and `staged_direct_bytes` are not reported from this crate directly; provide
  them from your concrete provider adapter when they are measurable.
- The crate intentionally does not ship native runtime configuration or SDK discovery.

## Performance diagnostics

The adapter exports two signals:

- `direct_binding_admissions()` (from the wrapped backend, when supported).
- `explicit_transfer_bytes()` (bytes observed on `read_buffer`/`write_buffer`).

`submission.copy-path-diagnostics` in conformance expects direct binding evidence to increase on submission
admission. If your provider cannot yet report staging counters, report them as zero and treat that
visibility as a migration target.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option.
