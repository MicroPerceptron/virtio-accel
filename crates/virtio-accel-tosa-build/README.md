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

## License

Licensed under Apache-2.0 or MIT, at your option.
