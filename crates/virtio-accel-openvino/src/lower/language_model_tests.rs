use super::*;
use virtio_accel_tosa_build::{
    OperatorKind as K, OwnedGraph, OwnedOperator, OwnedShape, OwnedTensor,
};

pub(crate) fn lookup(low: i32, high: i32) -> Vec<u8> {
    let mut g = OwnedGraph::new("bounded_lookup");
    for (name, shape, dtype) in [
        ("x", vec![1, 2], DType::FP32),
        ("cast", vec![1, 2], DType::INT32),
        ("lo", vec![1, 2], DType::INT32),
        ("index", vec![1, 2], DType::INT32),
        ("y", vec![1, 2, 2], DType::FP32),
    ] {
        g.tensors.push(OwnedTensor::new(name, shape, dtype));
    }
    for (name, shape, dtype, bytes) in [
        (
            "lower",
            vec![1, 1],
            DType::INT32,
            low.to_le_bytes().to_vec(),
        ),
        (
            "upper",
            vec![1, 1],
            DType::INT32,
            high.to_le_bytes().to_vec(),
        ),
        (
            "weights",
            vec![1, 4, 2],
            DType::FP32,
            (0..8).flat_map(|i| (i as f32).to_le_bytes()).collect(),
        ),
    ] {
        g.tensors
            .push(OwnedTensor::constant(name, shape, dtype, bytes));
        g.operators
            .push(OwnedOperator::new(K::Const, vec![], vec![name.into()]));
    }
    for (op, inputs, out) in [
        (K::Cast, vec!["x"], "cast"),
        (
            K::Maximum {
                nan_mode: NanPropagationMode::PROPAGATE,
            },
            vec!["cast", "lower"],
            "lo",
        ),
        (
            K::Minimum {
                nan_mode: NanPropagationMode::PROPAGATE,
            },
            vec!["lo", "upper"],
            "index",
        ),
        (K::Gather, vec!["weights", "index"], "y"),
    ] {
        g.operators.push(OwnedOperator::new(
            op,
            inputs.into_iter().map(Into::into).collect(),
            vec![out.into()],
        ));
    }
    g.inputs.push("x".into());
    g.outputs.push("y".into());
    g.build(OPENVINO_TOSA_QUANTIZED_TARGET).unwrap()
}

#[test]
fn lookup_bounds_are_proven_from_integer_operations_not_fixture_values() {
    for (lo, hi, accepted) in [
        (0, 3, true),
        (1, 2, true),
        (6, 3, true), // the outer minimum still bounds every result
        (-1, 3, false),
        (-6, -1, false),
        (0, 4, false),
        (i32::MIN, i32::MAX, false),
    ] {
        let bytes = lookup(lo, hi);
        let model = parse(&bytes).unwrap();
        let analysis = model.analyze_for(OPENVINO_TOSA_QUANTIZED_TARGET).unwrap();
        assert_eq!(analysis.conditions().len(), 1);
        assert_eq!(
            analysis.proves_integer_index_bounds(&analysis.conditions()[0]),
            accepted
        );
        let result = lower_tosa(&bytes, OPENVINO_TOSA_QUANTIZED_TARGET);
        assert_eq!(result.is_ok(), accepted, "{lo}..={hi}: {:?}", result.err());
    }
}

/// Optional production-lowering inspection of the pinned eight-layer fixture.
/// No checkpoint is embedded in the crate or fetched by the test suite.
#[test]
fn inspect_pretrained_language_model_normalization() {
    let Ok(path) = std::env::var("VIRTIO_ACCEL_OPENVINO_TEST_LANGUAGE_MODEL") else {
        return;
    };
    let bytes = std::fs::read(path).unwrap();
    for gpu in [false, true] {
        let lowered =
            lower_tosa_with_weights(&bytes, OPENVINO_TOSA_QUANTIZED_TARGET, !gpu, gpu).unwrap();
        let count = std::str::from_utf8(&lowered.xml)
            .unwrap()
            .matches("type=\"MVN\"")
            .count();
        assert_eq!(
            count, 17,
            "eight pre-attention norms, eight pre-MLP norms and the final norm"
        );
        eprintln!("TinyStories gpu={gpu}: fused_normalizations={count}");
        if let Ok(dir) = std::env::var("VIRTIO_ACCEL_OPENVINO_TEST_IR_DIR") {
            let dir = std::path::Path::new(&dir);
            std::fs::create_dir_all(dir).unwrap();
            let stem = if gpu {
                "language-model-GPU"
            } else {
                "language-model-CPU"
            };
            std::fs::write(dir.join(format!("{stem}.xml")), lowered.xml).unwrap();
            std::fs::write(dir.join(format!("{stem}.bin")), lowered.weights).unwrap();
        }
    }
}

#[test]
fn fp32_slice_uses_static_shape_parameters() {
    let mut g = OwnedGraph::new("slice");
    g.tensors
        .push(OwnedTensor::new("x", vec![1, 4], DType::FP32));
    g.tensors
        .push(OwnedTensor::new("y", vec![1, 2], DType::FP32));
    for (name, values) in [("start", vec![0, 1]), ("size", vec![1, 2])] {
        g.shapes.push(OwnedShape::new(name, values));
        g.operators
            .push(OwnedOperator::new(K::ConstShape, vec![], vec![name.into()]));
    }
    g.operators.push(OwnedOperator::new(
        K::Slice,
        vec!["x".into(), "start".into(), "size".into()],
        vec!["y".into()],
    ));
    g.inputs.push("x".into());
    g.outputs.push("y".into());
    let lowered = lower_tosa(
        &g.build(OPENVINO_TOSA_TARGET).unwrap(),
        OPENVINO_TOSA_TARGET,
    )
    .unwrap();
    assert!(
        std::str::from_utf8(&lowered.xml)
            .unwrap()
            .contains("type=\"Slice\"")
    );
}

pub(crate) fn normalization_graph(scale: f32, eps: f32, wrong_square: bool) -> Vec<u8> {
    let mut g = OwnedGraph::new("normalization");
    for (name, shape) in [
        ("x", vec![1, 2, 4]),
        ("sum", vec![1, 2, 1]),
        ("mean", vec![1, 2, 1]),
        ("centered", vec![1, 2, 4]),
        ("square", vec![1, 2, 4]),
        ("variance_sum", vec![1, 2, 1]),
        ("variance", vec![1, 2, 1]),
        ("eps_added", vec![1, 2, 1]),
        ("inverse", vec![1, 2, 1]),
        ("y", vec![1, 2, 4]),
    ] {
        g.tensors.push(OwnedTensor::new(name, shape, DType::FP32));
    }
    for (name, value) in [("scale", scale), ("epsilon", eps)] {
        g.tensors.push(OwnedTensor::constant(
            name,
            vec![1, 1, 1],
            DType::FP32,
            value.to_le_bytes().to_vec(),
        ));
        g.operators
            .push(OwnedOperator::new(K::Const, vec![], vec![name.into()]));
    }
    g.tensors.push(OwnedTensor::constant(
        "shift",
        vec![1],
        DType::INT8,
        vec![0],
    ));
    g.operators
        .push(OwnedOperator::new(K::Const, vec![], vec!["shift".into()]));
    for (op, inputs, out) in [
        (K::ReduceSum { axis: 2 }, vec!["x"], "sum"),
        (K::Mul, vec!["sum", "scale", "shift"], "mean"),
        (K::Sub, vec!["x", "mean"], "centered"),
        (
            K::Mul,
            vec![
                "centered",
                if wrong_square { "x" } else { "centered" },
                "shift",
            ],
            "square",
        ),
        (K::ReduceSum { axis: 2 }, vec!["square"], "variance_sum"),
        (K::Mul, vec!["variance_sum", "scale", "shift"], "variance"),
        (K::Add, vec!["variance", "epsilon"], "eps_added"),
        (K::Rsqrt, vec!["eps_added"], "inverse"),
        (K::Mul, vec!["centered", "inverse", "shift"], "y"),
    ] {
        g.operators.push(OwnedOperator::new(
            op,
            inputs.into_iter().map(Into::into).collect(),
            vec![out.into()],
        ));
    }
    g.inputs.push("x".into());
    g.outputs.push("y".into());
    g.build(OPENVINO_TOSA_TARGET).unwrap()
}

#[test]
fn normalization_fusion_requires_matching_mean_variance_and_positive_epsilon() {
    for (scale, eps, wrong_square, fused) in [
        (0.25, 1e-5, false, true),
        (0.5, 1e-5, false, false),
        (0.25, 0., false, false),
        (0.25, 1e-5, true, false),
    ] {
        let model = lower_tosa(
            &normalization_graph(scale, eps, wrong_square),
            OPENVINO_TOSA_TARGET,
        )
        .unwrap();
        let xml = std::str::from_utf8(&model.xml).unwrap();
        assert_eq!(xml.contains("type=\"MVN\""), fused);
        if fused {
            assert!(xml.contains("eps_mode=\"inside_sqrt\""));
        }
    }
}
