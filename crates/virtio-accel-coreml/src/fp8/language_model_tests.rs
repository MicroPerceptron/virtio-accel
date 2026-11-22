use super::*;
use virtio_accel_tosa::NanPropagationMode;
use virtio_accel_tosa_build::{OperatorKind as K, OwnedGraph, OwnedOperator, OwnedTensor};

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
    g.build(COREML_TOSA_QUANTIZED_TARGET).unwrap()
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
        let analysis = model.analyze_for(COREML_TOSA_QUANTIZED_TARGET).unwrap();
        assert_eq!(analysis.conditions().len(), 1);
        assert_eq!(
            analysis.proves_integer_index_bounds(&analysis.conditions()[0]),
            accepted
        );
        let result = lower(&bytes, COREML_TOSA_QUANTIZED_TARGET);
        assert_eq!(result.is_ok(), accepted, "{lo}..={hi}: {:?}", result.err());
    }
}

#[test]
fn export_language_model_operations_for_native_checks() {
    let Some(dir) = std::env::var_os("VIRTIO_ACCEL_COREML_TEST_LANGUAGE_MODEL_DIR") else {
        return;
    };
    let dir = std::path::PathBuf::from(dir);
    std::fs::create_dir_all(&dir).unwrap();
    let bytes = lookup(0, 3);
    let model = lower(&bytes, COREML_TOSA_QUANTIZED_TARGET).unwrap();
    std::fs::write(dir.join("lookup.mlmodel"), model.bytes).unwrap();
}
