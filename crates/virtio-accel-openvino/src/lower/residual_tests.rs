use super::*;
use virtio_accel_tosa_build::{Graph, Operator, OperatorKind, Tensor};

pub(crate) fn residual_graph(rows: i32, columns: i32, swapped: bool) -> Vec<u8> {
    let weights: Vec<u8> = (0..3 * columns)
        .flat_map(|i| ((i % 7 - 3) as f32 / 8.0).to_le_bytes())
        .collect();
    let input_shape = [1, rows, 3];
    let output_shape = [1, rows, columns];
    let weight_shape = [1, 3, columns];
    let scale = 0.5f32.to_le_bytes();
    let tensors = [
        Tensor::new("input", &input_shape, DType::FP32),
        Tensor::new("residual", &output_shape, DType::FP32),
        Tensor::constant("weights", &weight_shape, DType::FP32, &weights),
        Tensor::constant("zero", &[1], DType::FP32, &[0; 4]),
        Tensor::constant("scale", &[1, 1, 1], DType::FP32, &scale),
        Tensor::constant("shift", &[1], DType::INT8, &[0]),
        Tensor::new("product", &output_shape, DType::FP32),
        Tensor::new("sum", &output_shape, DType::FP32),
        Tensor::new("output", &output_shape, DType::FP32),
    ];
    let add = if swapped {
        ["residual", "product"]
    } else {
        ["product", "residual"]
    };
    let ops = [
        Operator::new(OperatorKind::Const, &[], &["weights"]),
        Operator::new(OperatorKind::Const, &[], &["zero"]),
        Operator::new(OperatorKind::Const, &[], &["scale"]),
        Operator::new(OperatorKind::Const, &[], &["shift"]),
        Operator::new(
            OperatorKind::MatMul,
            &["input", "weights", "zero", "zero"],
            &["product"],
        ),
        Operator::new(OperatorKind::Add, &add, &["sum"]),
        Operator::new(OperatorKind::Mul, &["sum", "scale", "shift"], &["output"]),
    ];
    Graph::new(
        "external_residual",
        &tensors,
        &ops,
        &["input", "residual"],
        &["output"],
    )
    .build(OPENVINO_TOSA_TARGET)
    .unwrap()
}

#[test]
fn gpu_external_residual_avoids_destination_sum_fusion_in_both_operand_orders() {
    for swapped in [false, true] {
        let bytes = residual_graph(2, 4, swapped);
        let generic = lower_tosa(&bytes, OPENVINO_TOSA_TARGET).unwrap();
        let gpu = lower_tosa_with_weights(&bytes, OPENVINO_TOSA_TARGET, false, true).unwrap();
        assert!(
            std::str::from_utf8(&generic.xml)
                .unwrap()
                .contains("type=\"Add\"")
        );
        let xml = std::str::from_utf8(&gpu.xml).unwrap();
        assert!(!xml.contains("type=\"Add\""));
        assert!(xml.contains("type=\"Negative\""));
        assert!(xml.contains("type=\"Subtract\""));
        assert_eq!(gpu.weights, generic.weights);
        assert_eq!(gpu.features.len(), 3);
    }
}
