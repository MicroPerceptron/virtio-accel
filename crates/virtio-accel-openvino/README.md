# virtio-accel-openvino

An Intel OpenVINO host backend for `virtio-accel`, executing device-neutral TOSA 1.0 programs on
the NPU, GPU, or CPU inference devices of one host. `OpenVinoAccelerator::new` selects the
preferred available device (NPU, then GPU, then CPU) and `with_device` selects one explicitly, by
enumerated name (`"GPU.1"`) or class prefix (`"NPU"`). Models are compiled with OpenVINO's
`ACCURACY` execution-mode hint. Quantized FP32 and exact FP8 activation graphs additionally
require the explicit `f32` inference-precision hint; a device that rejects it cannot admit those
graphs. The accuracy hint alone is not an FP32 precision guarantee on every plugin.

**Portability tier:** `host-native` — OpenVINO C runtime (`libopenvino_c`, 2026.x) when detected
at build time; a compile-only unsupported-runtime placeholder elsewhere.

The boundary is a build-environment probe rather than a target operating system: the build script
asks pkg-config for `openvino` and enables the native modules only on success.
`VIRTIO_ACCEL_OPENVINO=1` makes a missing runtime a loud build failure, `=0` forces the
placeholder, and `VIRTIO_ACCEL_OPENVINO_LIB_DIR` links archive installs that ship no pkg-config
metadata. Builds without the runtime still compile and unit-test the portable TOSA-to-IR encoder.

## Production TOSA artifacts

The production path accepts `virtio-accel-tosa` artifacts (`ArtifactFormat` `"TOSA"`) declaring
the floating-point or integer profile of TOSA 1.0 at level 8K with no extensions, and requires the maximal
`resident_bytes` charge (`REQUIRED_RESIDENT_BYTES`): OpenVINO publishes no finite compiled-model
residency bound, so the provider promise stays truthful.

Loading validates and analyzes the FlatBuffer with `virtio-accel-tosa`, lowers the graph to an
in-memory OpenVINO IR (version 11) document plus weights blob inside this crate, reads it with
`ov_core_read_model_from_memory_buffer`, and compiles it for this instance's device. Program
admission accepts exactly one static region and basic block with no runtime obligations; static,
positive shapes; FP16, FP32, or INT8 tensor boundaries plus restricted INT32 outputs; and the
operator set reported by `supports_tosa_operator`. The integer target currently admits exact INT8
identity and INT8 MATMUL with INT32 output. Everything else is rejected while loading, with
`Unsupported`, `Incompatible`, or `InvalidArgument` mapped exactly as the backend implementer
guide specifies.

`TosaCapabilityProvider` publishes separate conservative floating and integer descriptors after a
native runtime/device has been constructed. Compile-only placeholders return an empty descriptor
slice instead of advertising work they cannot execute.

Binding slots are device-neutral: inputs occupy slots `0..N` and outputs occupy `N..N+M`, both in
declared order.

### Low-precision boundary

`supports_tosa_dtype` reports encoding capability independently of target and operator coverage.
The integer profile retains exact INT8 identity and INT32 MATMUL with explicit zero points.
The FP8 descriptor is advertised only after a native device compilation probe succeeds.
E4M3/E5M2 weight constants, activations and packed cache movement are supported.

FP8 graphs containing FP32 CAST or SLICE use INT8 bit carriers and explicit decode/encode operations,
avoiding native Convert's observed FP16 double rounding and the GPU's FP8 Slice restriction.
This path admits constants, CAST, identity, MATMUL, widened ADD/MUL, reshape, transpose, concat,
and static slice. FP32 rounding is direct nearest-even; movement preserves all storage bits.
Other existing FP8 graphs retain the native lowering described in ADR 0009.

`OPENVINO_TOSA_QUANTIZED_TARGET` accepts the standard TOSA graphs emitted by
`virtio_accel_tosa_build::quantized`, including NVFP4 E2M1 packed weights, E4M3 block scales
(group size 16), and an explicit tensor scale. FP32 support is probed with the required native
precision policy. Devices that reject it advertise `OPENVINO_TOSA_QUANTIZED_FP16_CAPABILITY`
instead: use `nvfp4_matmul_fp16` and explicit FP16 activations/output. No FP32 graph is silently
relabelled FP16. Lookup indices must have the static byte-widening bounds proof.

CPU and GPU pass the FP32 projection and exact FP8 activation/cache tests on OpenVINO 2026.4.1.
CPU, GPU and Lunar Lake NPU pass the explicit FP16 NVFP4 projection. That NPU still withholds
the native FP8 target. Packed boundary bytes do not imply native FP4 arithmetic or compressed
resident intermediates; decoding occurs inside the device graph. See
[ADR 0014](../../docs/adr/0014-packed-quantized-inference.md). INT4 tensor semantics, RESCALE,
and other unsupported integer operations remain rejected.

### Constant INT4 and INT8 weight-only projections

`quantized::{int4_matmul_fp16,int8_matmul_fp16}` author symmetric grouped W4A16/W8A16
projections with immutable weights/scales and FP16 activation/output bindings. The graph
spells out integer-to-float conversion and multiplication. At load time, an INT8 constant
whose values all fit [-8,7] and whose only consumers are FP16/FP32 casts can use a packed
native i4 constant. Integer boundaries and non-float consumers prevent this rewrite.
TOSA's INT4 dtype and exact INT8-to-INT32 MATMUL semantics are unchanged.

The backend folds a batch-one RHS transpose into native MatMul and exposes the grouped
weight reshape as `[N,K]`, enabling OpenVINO's compressed FullyConnected matcher. CPU/GPU
compilation disables implicit dynamic activation quantization; ACCURACY alone permits it.
There is no opt-in to lossy W4A8 under a W4A16 label. NPU lacks that runtime property.

On Lunar Lake / OpenVINO 2026.4.1, production-lowered IR reports GPU `weights_precision=i4`
and `runtimePrecision=f16`, with no DynamicQuantize node. CPU and NPU pass numerical
execution, but this does not establish compressed residency or a speedup on those devices;
the inspected CPU graph still expands the weights during execution. Portable artifacts
retain one byte per weight; the native IR uses half a byte, plus scales/alignment. Changing
weights requires program reload. See the [benchmark diagnostic](../../benchmarks/README.md)
and [ADR 0015](../../docs/adr/0015-compressed-integer-weights.md).

### Ordered comparison legalization

`GREATER_EQUAL` lowers to `Greater OR Equal` with an explicit `IsNaN` guard. On OpenVINO 2026.4.1, Lunar Lake NPU's
native `GreaterEqual` declares BOOL but produces FP16 storage, failing a caller-bound
one-byte output. The equivalent ordered comparisons preserve NaN, infinity and signed-zero
semantics and return actual Boolean bytes. The explicit guard also handles CPU ISA paths
that otherwise evaluate unordered comparisons as true. CPU/GPU/NPU regression tests cover these edges
and the full Boolean corpus; no CPU fallback or hidden output conversion is used.

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

Warm submissions reuse the queue's pointer-slot storage and one cleared high-water allocation for
event backing metadata and bound-tensor metadata. The spare vectors contain capacity only: event
completion drops all backing guards before recycling their vector, and event destruction frees the
native request and tensor handles before recycling tensor metadata. Infer requests are never
pooled because OpenVINO retains the tensor objects bound to a request and exposes no C reset API
that can prove those guest-memory references detached.

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

## Runtime loading for distribution

The opt-in Cargo feature `runtime-loading` compiles the adapter on Linux, macOS, and
Windows without a build-time SDK or a link dependency on OpenVINO. Python wheels use
this mode. Set `VIRTIO_ACCEL_OPENVINO_LIBRARY` to a C library path, or
`VIRTIO_ACCEL_OPENVINO_LIB_DIR` to its directory; otherwise the OS loader is used. The
Python wrapper can also locate the library in the optional `openvino` Python package.
On Windows, an explicit DLL path also supplies the dependency search directory.
All required symbols must resolve before discovery, and the loaded library is retained
for the process lifetime. Missing libraries return `RuntimeUnavailable`. Restart the
process to switch runtimes. `VIRTIO_ACCEL_OPENVINO=0` still disables compilation;
`=1` with runtime loading selects the adapter without requiring a build-time runtime.
Use `VIRTIO_ACCEL_OPENVINO_REQUIRE_HARDWARE=1` to make execution tests mandatory.
The default Rust build continues to use the linked mode described above.
