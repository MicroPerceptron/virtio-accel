# virtio-accel-tosa-build

`virtio-accel-tosa-build` is the safe, portable authoring companion to
`virtio-accel-tosa`. It constructs deterministic TOSA 1.0 FlatBuffers for
static, single-block graphs without exposing generated FlatBuffers bindings,
table slots, union tags, or raw enum discriminants to callers.

The crate is `no_std + alloc`; it needs no `flatc`, filesystem, host runtime,
or provider SDK. Borrowed `Graph` definitions serve statically declared graphs,
while `OwnedGraph` supports compiler frontends that discover graph metadata
incrementally. Both build paths validate every completed artifact with the
production `virtio-accel-tosa` parser and semantic validator before returning
bytes, so successful output is immediately usable as an `ArtifactRef`.

```rust
use virtio_accel_tosa::{
    DType, ExtensionSet, Level, ProfileSet, Target, Version,
};
use virtio_accel_tosa_build::{Graph, Operator, OperatorKind, Shape, Tensor};

let tensors = [
    Tensor::new("input", &[1], DType::FP32),
    Tensor::new("output", &[1], DType::FP32),
];
let operators = [Operator::new(
    OperatorKind::Identity,
    &["input"],
    &["output"],
)];
let target = Target::new(
    Version::TOSA_1_0,
    ProfileSet::FLOATING_POINT,
    Level::Level8K,
    ExtensionSet::NONE,
);
let bytes = Graph::new(
    "main",
    &tensors,
    &operators,
    &["input"],
    &["output"],
)
.build(target)?;
# Ok::<(), virtio_accel_tosa_build::BuildError>(())
```

Frontends can build the same artifact without maintaining parallel owned and borrowed adapter
structures:

```rust
use virtio_accel_tosa::{
    DType, ExtensionSet, Level, ProfileSet, Target, Version,
};
use virtio_accel_tosa_build::{OwnedGraph, OwnedOperator, OwnedTensor, OperatorKind};

let mut graph = OwnedGraph::new("main");
graph
    .push_tensor(OwnedTensor::new("input", vec![1], DType::FP32))
    .push_tensor(OwnedTensor::new("output", vec![1], DType::FP32))
    .push_operator(OwnedOperator::new(
        OperatorKind::Identity,
        vec!["input".into()],
        vec!["output".into()],
    ))
    .push_input("input")
    .push_output("output");
let target = Target::new(
    Version::TOSA_1_0,
    ProfileSet::FLOATING_POINT,
    Level::Level8K,
    ExtensionSet::NONE,
);
let bytes = graph.build(target)?;
# Ok::<(), virtio_accel_tosa_build::BuildError>(())
```

Compile-time shape operands use the separate typed `Shape` namespace and a `ConstShape` producer:

```rust
# use virtio_accel_tosa::{DType, ExtensionSet, Level, ProfileSet, Target, Version};
# use virtio_accel_tosa_build::{Graph, Operator, OperatorKind, Shape, Tensor};
# let target = Target::new(Version::TOSA_1_0, ProfileSet::FLOATING_POINT, Level::Level8K, ExtensionSet::NONE);
let tensors = [
    Tensor::new("input", &[1, 4], DType::FP32),
    Tensor::new("output", &[2, 2], DType::FP32),
];
let shapes = [Shape::new("target", &[2, 2])];
let operators = [
    Operator::new(OperatorKind::ConstShape, &[], &["target"]),
    Operator::new(OperatorKind::Reshape, &["input", "target"], &["output"]),
];
let bytes = Graph::new("main", &tensors, &operators, &["input"], &["output"])
    .with_shapes(&shapes)
    .build(target)?;
# Ok::<(), virtio_accel_tosa_build::BuildError>(())
```

Inline tensor bytes are accepted only for a nonempty, exactly sized `Tensor::constant` produced by
`OperatorKind::Const`; this prevents authoring data that ingestion would classify as nonconstant.

The surface deliberately covers the static operator set exercised by current compiler frontends.
The borrowed and owned graph definitions feed one serializer and validator; the owned surface adds
frontend-friendly metadata storage and incremental assembly, while constants can be either moved in
or borrowed without copying. Attribute-bearing operators use typed fields, so an unsupported
operator cannot be smuggled in as a raw number. Multi-block
control flow, dynamic shapes, variables, external tensor data, and custom
operators remain outside this authoring boundary.

## Packed NVFP4 projection

```rust
use virtio_accel_tosa_build::quantized::{nvfp4_matmul, nvfp4_matmul_fp16, TARGET};
let fp32_graph = nvfp4_matmul(1, 256, 512, 0.125)?;
let fp16_graph = nvfp4_matmul_fp16(1, 256, 512, 0x3000)?; // binary16 1/8
# Ok::<(), virtio_accel_tosa_build::quantized::QuantizedBuildError>(())
```

Both artifacts use TOSA 1.0 FLOATING_POINT | INTEGER, Level8K, no extensions (`TARGET`).
Select a Core ML or OpenVINO mixed quantized capability before loading.

| Slot | Shape | Storage |
|---|---|---|
| 0 | `[1, M, K]` | FP32 or explicit FP16 activations |
| 1 | `[1, N*K/2]` | INT8 bit carrier for E2M1 weights; low nibble first |
| 2 | `[1, N*K/16]` | INT8 bit carrier for E4M3 block scales |
| 3 | `[1, M, N]` | FP32 or explicit FP16 output |

K must be a positive multiple of 16. Each weight row has its own K/16 scales. The equation is
`output = activations @ (E2M1 * block_scale).T * tensor_scale`; the separate positive finite
tensor scale is applied after the matrix product. FP16 explicitly rounds the product before
its final scale. Runtime weights/scales may be updated without rebuilding the artifact.

The graph uses bounded lookup tables, reshape, transpose, multiply and matmul. It neither assigns
a new wire dtype nor mistakes NVFP4 for INT4 or MXFP4. Packed buffers stay packed at the backend
boundary, while native compilers may allocate wider intermediate tensors. This initial path
provides correctness and interoperability, not a native FP4 throughput guarantee.

## License

Licensed under Apache-2.0 or MIT, at your option.
