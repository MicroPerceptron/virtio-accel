use super::*;
use virtio_accel_tosa::{DType, NanPropagationMode};
use virtio_accel_tosa_build::{Graph, Operator, OperatorKind as K, Shape, Tensor};

fn gpu_backend() -> Option<CoreMlAccelerator> {
    match CoreMlAccelerator::new_tosa_with_compute_units(CoreMlComputeUnits::CpuAndGpu) {
        Ok(backend) => Some(backend),
        Err(InitError::GpuUnavailable)
            if std::env::var_os("VIRTIO_ACCEL_COREML_REQUIRE_QUANTIZED").is_none() =>
        {
            None
        }
        Err(error) => panic!("{error}"),
    }
}

#[test]
fn typed_repeated_concat_preserves_recurrent_layout() {
    let Some(backend) = gpu_backend() else {
        return;
    };
    let target = virtio_accel_coreml::COREML_TOSA_QUANTIZED_TARGET;
    let tensors = [
        Tensor::new("x", &[24, 64, 1], DType::FP32),
        Tensor::new("mid", &[24, 64, 64], DType::FP32),
        Tensor::new("y", &[24, 64, 128], DType::FP32),
    ];
    let repeats = ["x"; 64];
    let ops = [
        Operator::new(K::Concat { axis: 2 }, &repeats, &["mid"]),
        Operator::new(K::Concat { axis: 2 }, &["mid", "mid"], &["y"]),
    ];
    let graph = Graph::new("repeat", &tensors, &ops, &["x"], &["y"])
        .build(target)
        .unwrap();
    let input = float_bytes((0..1536).map(|i| i as f32 / 1536.));
    let out = run_tosa_bytes_on(&backend, &graph, target, &[&input], 24 * 64 * 128 * 4).unwrap();
    for (index, bytes) in out.chunks_exact(4).enumerate() {
        let actual = f32::from_le_bytes(bytes.try_into().unwrap());
        assert_eq!(actual, (index / 128) as f32 / 1536., "element {index}");
    }
}

#[test]
fn typed_tiled_decay_preserves_zero_recurrent_state() {
    let Some(backend) = gpu_backend() else {
        return;
    };
    let target = virtio_accel_coreml::COREML_TOSA_QUANTIZED_TARGET;
    let a = float_bytes((0..24).map(|h| -0.1 * (h + 1) as f32));
    let tensors = [
        Tensor::constant("a", &[24, 1, 1], DType::FP32, &a),
        Tensor::constant("shift", &[1], DType::INT8, &[0]),
        Tensor::new("dt", &[24, 64, 1], DType::FP32),
        Tensor::new("state", &[24, 64, 128], DType::FP32),
        Tensor::new("a1", &[24, 64, 1], DType::FP32),
        Tensor::new("a2", &[24, 64, 64], DType::FP32),
        Tensor::new("a3", &[24, 64, 128], DType::FP32),
        Tensor::new("disc", &[24, 64, 128], DType::FP32),
        Tensor::new("decay", &[24, 64, 128], DType::FP32),
        Tensor::new("old", &[24, 64, 128], DType::FP32),
        Tensor::new("y", &[24, 64, 128], DType::FP32),
    ];
    let repeat_a = ["a"; 64];
    let repeat_a1 = ["a1"; 64];
    let ops = [
        Operator::new(K::Const, &[], &["a"]),
        Operator::new(K::Const, &[], &["shift"]),
        Operator::new(K::Concat { axis: 1 }, &repeat_a, &["a1"]),
        Operator::new(K::Concat { axis: 2 }, &repeat_a1, &["a2"]),
        Operator::new(K::Concat { axis: 2 }, &["a2", "a2"], &["a3"]),
        Operator::new(K::Mul, &["a3", "dt", "shift"], &["disc"]),
        Operator::new(K::Exp, &["disc"], &["decay"]),
        Operator::new(K::Mul, &["state", "decay", "shift"], &["old"]),
        Operator::new(K::Add, &["old", "decay"], &["y"]),
    ];
    let graph = Graph::new("tiled_decay", &tensors, &ops, &["dt", "state"], &["y"])
        .build(target)
        .unwrap();
    let dt = float_bytes((0..1536).map(|i| 0.01 + i as f32 / 1536.));
    let state = vec![0; 24 * 64 * 128 * 4];
    let out = run_tosa_bytes_on(&backend, &graph, target, &[&dt, &state], state.len()).unwrap();
    for (i, bytes) in out.chunks_exact(4).enumerate() {
        let actual = f32::from_le_bytes(bytes.try_into().unwrap());
        let expected =
            (-0.1 * ((i / (64 * 128)) + 1) as f32 * (0.01 + (i / 128) as f32 / 1536.)).exp();
        assert!(
            actual.is_finite() && (actual - expected).abs() < 1e-5,
            "element {i}: {actual} != {expected}"
        );
    }
}

#[test]
fn typed_transformer_arithmetic_preserves_numerics() {
    let target = virtio_accel_coreml::COREML_TOSA_QUANTIZED_TARGET;
    for units in [
        CoreMlComputeUnits::CpuAndGpu,
        CoreMlComputeUnits::CpuAndNeuralEngine,
    ] {
        let backend = match CoreMlAccelerator::new_tosa_with_compute_units(units) {
            Ok(b) => b,
            Err(InitError::GpuUnavailable | InitError::NeuralEngineUnavailable)
                if std::env::var_os("VIRTIO_ACCEL_COREML_REQUIRE_QUANTIZED").is_none() =>
            {
                continue;
            }
            Err(e) => panic!("{e}"),
        };
        for (op, values, expected) in [
            (K::Rsqrt, vec![0.25, 1., 4., 16.], vec![2., 1., 0.5, 0.25]),
            (
                K::Reciprocal,
                vec![0.25, 1., 4., 16.],
                vec![4., 1., 0.25, 0.0625],
            ),
            (
                K::Exp,
                vec![0., 1., -1., 2.],
                vec![1., std::f32::consts::E, (-1_f32).exp(), 2_f32.exp()],
            ),
            (K::Floor, vec![-1.5, -0.1, 0.1, 1.5], vec![-2., -1., 0., 1.]),
            (K::Ceil, vec![-1.5, -0.1, 0.1, 1.5], vec![-1., 0., 1., 2.]),
            (
                K::Sigmoid,
                vec![-20., 0., 1., 20.],
                vec![0., 0.5, 1. / (1. + (-1_f32).exp()), 1.],
            ),
        ] {
            eprintln!("typed native {units:?} {op:?}");
            let tensors = [
                Tensor::new("x", &[1, 4], DType::FP32),
                Tensor::new("y", &[1, 4], DType::FP32),
            ];
            let ops = [Operator::new(op, &["x"], &["y"])];
            let graph = Graph::new("typed", &tensors, &ops, &["x"], &["y"])
                .build(target)
                .unwrap();
            let input = float_bytes(values);
            let out = run_tosa_bytes_on(&backend, &graph, target, &[&input], 16).unwrap();
            for (b, e) in out.chunks_exact(4).zip(expected) {
                let v = f32::from_le_bytes(b.try_into().unwrap());
                assert!((v - e).abs() < 1e-5, "{op:?} {v} != {e}");
            }
        }
        // Zero epsilon must preserve log(0)=-inf and log(negative)=NaN.
        let tensors = [
            Tensor::new("x", &[1, 4], DType::FP32),
            Tensor::new("y", &[1, 4], DType::FP32),
        ];
        let ops = [Operator::new(K::Log, &["x"], &["y"])];
        let graph = Graph::new("log_boundary", &tensors, &ops, &["x"], &["y"])
            .build(target)
            .unwrap();
        let input = float_bytes([0., -1., 1., std::f32::consts::E]);
        let out = run_tosa_bytes_on(&backend, &graph, target, &[&input], 16).unwrap();
        let values: Vec<_> = out
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        assert_eq!(values[0], f32::NEG_INFINITY);
        assert!(values[1].is_nan());
        assert_eq!(values[2], 0.);
        assert!((values[3] - 1.).abs() < 1e-5);
        let tensors = [
            Tensor::new("x", &[1, 4], DType::FP32),
            Tensor::constant("zero", &[1], DType::FP32, &[0, 0, 0, 0]),
            Tensor::new("y", &[1, 4], DType::FP32),
        ];
        let ops = [
            Operator::new(K::Const, &[], &["zero"]),
            Operator::new(K::Negate, &["x", "zero", "zero"], &["y"]),
        ];
        let graph = Graph::new("negate", &tensors, &ops, &["x"], &["y"])
            .build(target)
            .unwrap();
        let input = float_bytes([-2., -0.5, 0.5, 2.]);
        let out = run_tosa_bytes_on(&backend, &graph, target, &[&input], 16).unwrap();
        assert_eq!(out, float_bytes([2., 0.5, -0.5, -2.]));
        let infinity = float_bytes([f32::INFINITY]);
        let tensors = [
            Tensor::new("x", &[1, 4], DType::FP32),
            Tensor::constant("inf", &[1, 1], DType::FP32, &infinity),
            Tensor::new("y", &[1, 4], DType::FP32),
        ];
        let ops = [
            Operator::new(K::Const, &[], &["inf"]),
            Operator::new(
                K::Minimum {
                    nan_mode: NanPropagationMode::PROPAGATE,
                },
                &["x", "inf"],
                &["y"],
            ),
        ];
        let graph = Graph::new("infinity_min", &tensors, &ops, &["x"], &["y"])
            .build(target)
            .unwrap();
        let input = float_bytes([0., 1., -1., 10.]);
        let out = run_tosa_bytes_on(&backend, &graph, target, &[&input], 16).unwrap();
        assert_eq!(out, input);
        for op in [
            K::ReduceSum { axis: 1 },
            K::ReduceMax {
                axis: 1,
                nan_mode: NanPropagationMode::PROPAGATE,
            },
        ] {
            eprintln!("typed native {units:?} {op:?}");
            let tensors = [
                Tensor::new("x", &[2, 4], DType::FP32),
                Tensor::new("y", &[2, 1], DType::FP32),
            ];
            let ops = [Operator::new(op, &["x"], &["y"])];
            let graph = Graph::new("reduce", &tensors, &ops, &["x"], &["y"])
                .build(target)
                .unwrap();
            let input = float_bytes([1., 2., 3., 4., 1., f32::NAN, 3., 4.]);
            let out = run_tosa_bytes_on(&backend, &graph, target, &[&input], 8).unwrap();
            let values: Vec<_> = out
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect();
            assert_eq!(
                values[0],
                if matches!(op, K::ReduceSum { .. }) {
                    10.
                } else {
                    4.
                }
            );
            assert!(values[1].is_nan(), "{op:?} must propagate NaN: {values:?}");
        }
        for op in [
            K::Maximum {
                nan_mode: NanPropagationMode::PROPAGATE,
            },
            K::Minimum {
                nan_mode: NanPropagationMode::PROPAGATE,
            },
        ] {
            let tensors = [
                Tensor::new("a", &[1, 4], DType::FP32),
                Tensor::new("b", &[1, 4], DType::FP32),
                Tensor::new("y", &[1, 4], DType::FP32),
            ];
            let ops = [Operator::new(op, &["a", "b"], &["y"])];
            let graph = Graph::new("minmax", &tensors, &ops, &["a", "b"], &["y"])
                .build(target)
                .unwrap();
            let a = float_bytes([1., f32::NAN, 2., f32::NAN]);
            let b = float_bytes([2., 1., f32::NAN, f32::NAN]);
            let out = run_tosa_bytes_on(&backend, &graph, target, &[&a, &b], 16).unwrap();
            let values: Vec<_> = out
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect();
            assert_eq!(
                values[0],
                if matches!(op, K::Maximum { .. }) {
                    2.
                } else {
                    1.
                }
            );
            assert!(values[1..].iter().all(|v| v.is_nan()), "{op:?}: {values:?}");
        }
        let tensors = [
            Tensor::new("x", &[2, 4], DType::FP32),
            Tensor::new("y", &[2, 2], DType::FP32),
        ];
        let shapes = [Shape::new("start", &[0, 1]), Shape::new("size", &[2, 2])];
        let ops = [
            Operator::new(K::ConstShape, &[], &["start"]),
            Operator::new(K::ConstShape, &[], &["size"]),
            Operator::new(K::Slice, &["x", "start", "size"], &["y"]),
        ];
        let graph = Graph::new("slice", &tensors, &ops, &["x"], &["y"])
            .with_shapes(&shapes)
            .build(target)
            .unwrap();
        let input = float_bytes((0..8).map(|i| i as f32));
        let out = run_tosa_bytes_on(&backend, &graph, target, &[&input], 16).unwrap();
        assert_eq!(out, float_bytes([1., 2., 5., 6.]));
    }
}
