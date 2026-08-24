# API reference

The public Rust API is documented with rustdoc and built for the whole workspace
with all features enabled. The generated documentation is served under
[`/api/`](api/index.html).

## Crates

| Crate | Tier | Role |
| --- | --- | --- |
| [`virtio-accel`](api/virtio_accel/index.html) | `core + alloc` | Facade re-exporting the portable layers |
| [`virtio-accel-proto`](api/virtio_accel_proto/index.html) | `core` | Pointer-free, little-endian protocol 1.0 wire structures |
| [`virtio-accel-transport`](api/virtio_accel_transport/index.html) | `core` | Descriptor-chain, queue, reset, and notification ports |
| [`virtio-accel-core`](api/virtio_accel_core/index.html) | `core` | Backend lifecycle, memory, program, queue, and event contracts |
| [`virtio-accel-tosa`](api/virtio_accel_tosa/index.html) | `core + alloc` | Bounded zero-copy TOSA 1.0 validation and lowering analysis |
| [`virtio-accel-tosa-build`](api/virtio_accel_tosa_build/index.html) | `core + alloc` | Borrowed and owned static TOSA 1.0 authoring |
| [`virtio-accel-vaccel`](api/virtio_accel_vaccel/index.html) | `core` | Adapter seam for native provider contracts |
| [`virtio-accel-coreml`](api/virtio_accel_coreml/index.html) | macOS `std` | TOSA-to-Core ML lowering and ANE-capable prediction |
| [`virtio-accel-openvino`](api/virtio_accel_openvino/index.html) | Linux `std` | TOSA-to-OpenVINO IR lowering and NPU/GPU/CPU inference |
| [`virtio-accel-hexagon`](api/virtio_accel_hexagon/index.html) | Windows ARM64 `std` | Strict FP16/INT8 TOSA-to-QNN lowering |
| [`virtio-accel-split-queue`](api/virtio_accel_split_queue/index.html) | `core + alloc` | Bounded in-memory split-ring reference model |
| [`virtio-accel-guest`](api/virtio_accel_guest/index.html) | `core + alloc` | Typed reference client |
| [`virtio-accel-device`](api/virtio_accel_device/index.html) | `core + alloc` | Device-owned state with generational IDs |
| [`virtio-accel-mock`](api/virtio_accel_mock/index.html) | `std` | In-memory backend with deterministic test-only artifacts |
| [`virtio-accel-conformance`](api/virtio_accel_conformance/index.html) | `std` | Transport-free semantic suite and numerical corpus |
| [`virtio-accel-cleanroom`](api/virtio_accel_cleanroom/index.html) | `core` | Independent conformance codec |

The rustdoc index at [`/api/`](api/index.html) lists every crate and its
feature-gated items. See the [public API policy](docs/public-api.md) for how the
documentation layers are split and what is guaranteed stable.
