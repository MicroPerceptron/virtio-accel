# virtio-accel-openvino

An Intel OpenVINO host backend for `virtio-accel`, executing device-neutral TOSA 1.0 programs on
the NPU, GPU, or CPU inference devices of one host. `OpenVinoAccelerator::new` selects the
preferred available device (NPU, then GPU, then CPU) and `with_device` selects one explicitly, by
enumerated name (`"GPU.1"`) or class prefix (`"NPU"`). Models are compiled with OpenVINO's
`ACCURACY` execution-mode hint so plugins may not silently execute a declared-FP32 graph at
reduced precision.

**Portability tier:** `host-native` — OpenVINO C runtime (`libopenvino_c`, 2026.x) when detected
at build time; a compile-only unsupported-runtime placeholder elsewhere.

The boundary is a build-environment probe rather than a target operating system: the build script
asks pkg-config for `openvino` and enables the native modules only on success.
`VIRTIO_ACCEL_OPENVINO=1` makes a missing runtime a loud build failure, `=0` forces the
placeholder, and `VIRTIO_ACCEL_OPENVINO_LIB_DIR` links archive installs that ship no pkg-config
metadata. Builds without the runtime still compile and unit-test the portable TOSA-to-IR encoder.

## Production TOSA artifacts

The production path accepts `virtio-accel-tosa` artifacts (`ArtifactFormat` `"TOSA"`) declaring
the floating-point profile of TOSA 1.0 at level 8K with no extensions, and requires the maximal
`resident_bytes` charge (`REQUIRED_RESIDENT_BYTES`): OpenVINO publishes no finite compiled-model
residency bound, so the provider promise stays truthful.

Loading validates and analyzes the FlatBuffer with `virtio-accel-tosa`, lowers the graph to an
in-memory OpenVINO IR (version 11) document plus weights blob inside this crate, reads it with
`ov_core_read_model_from_memory_buffer`, and compiles it for this instance's device. Program
admission accepts exactly one static region and basic block with no runtime obligations; static,
positive shapes; FP16 and FP32 tensor boundaries plus restricted INT32 outputs for operators such
as `ARGMAX`; and the operator set reported by `supports_tosa_operator` (the same static
floating-point subset as the Core ML backend). Everything else is rejected while loading, with
`Unsupported`, `Incompatible`, or `InvalidArgument` mapped exactly as the backend implementer
guide specifies.

Binding slots are device-neutral: inputs occupy slots `0..N` and outputs occupy `N..N+M`, both in
declared order.

### Low-precision boundary

`supports_tosa_dtype` reports FP16, FP32, and INT32 independently of operator coverage. INT8,
packed INT4, FP8E4M3, and FP8E5M2 artifacts are rejected before native compilation: executing
quantized TOSA tensor semantics on OpenVINO requires a quantization-aware lowering tier with
explicit calibration and rescale handling, not a silent dequantization to floating point. This
backend never dequantizes unsupported low-precision graphs.

## Direct buffers and events

Program buffers are page-aligned, zero-initialized provider allocations. Submissions wrap the
bound range of each buffer with `ov_tensor_create_from_host_ptr` — no bounce allocations — and
completion is accepted only when the runtime reports the caller's own allocation as its output
tensor storage; a provider-side output reallocation fails the event as `Incompatible` instead of
being staged invisibly.

Completion is poll-latched: each submission owns one OpenVINO infer request, `poll_event` probes
it without blocking, and the first observed terminal state is latched for stable re-observation.
Buffer in-flight guards admit concurrent read-only bindings or one writer, reject conflicting
submissions and host transfers with `Busy`, and are released before a terminal state is
published. Finite submission timeouts are enforced best-effort at poll time through
`ov_infer_request_cancel`, surfacing as `DeadlineExpired`. Event cancellation is not advertised.

## Devices

One OpenVINO core is created per process and shared by every backend instance, because plugin
re-initialization is not crash-safe on driverless hosts (see `SAFETY.md`). Device classes map to
`AcceleratorClass::NPU`, `GPU`, and `OTHER` (CPU); the NPU and GPU devices enumerate only when
the corresponding Intel Level Zero vendor driver is installed. A plugin that rejects a model —
for example FP32 on an FP16-centric NPU compiler — surfaces the failure from `load_program`;
nothing is silently downconverted or re-placed by this crate.

## Running

```sh
cargo run -p virtio-accel-openvino --example tosa_openvino
cargo test -p virtio-accel-openvino
```

Without an OpenVINO runtime the example prints a notice and exits successfully; without an
inference device the tests skip execution paths.

Part of the [`virtio-accel`](https://github.com/MicroPerceptron/virtio-accel) workspace: an experimental native-Rust protocol and implementation stack for a
transport-neutral virtual accelerator device. Portable crates contain no host-OS or vendor APIs;
host integrations live in separate adapter crates and never become their dependencies. The
project claims no Virtio device ID.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option.
