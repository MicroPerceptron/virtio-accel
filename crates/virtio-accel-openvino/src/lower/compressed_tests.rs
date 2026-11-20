use super::*;
use virtio_accel_tosa_build::{Graph, Operator, OperatorKind, Tensor, quantized};

#[test]
fn constant_projection_preserves_compressed_ir() {
    for (name, weights, precision, bytes) in [
        ("int4", vec![0x78, 0xf0], "i4", 2),
        ("int8", vec![0x80, 0xff, 0x07, 0x7f], "i8", 4),
    ] {
        let graph = if name == "int4" {
            quantized::int4_matmul_fp16(1, 1, 4, 2, &weights, &[0x3000, 0x3400])
        } else {
            quantized::int8_matmul_fp16(
                1,
                1,
                4,
                2,
                &weights.iter().map(|v| *v as i8).collect::<Vec<_>>(),
                &[0x3000, 0x3400],
            )
        }
        .unwrap();
        let lowered = lower_tosa(&graph, OPENVINO_TOSA_QUANTIZED_TARGET).unwrap();
        let xml = std::str::from_utf8(&lowered.xml).unwrap();
        assert!(xml.contains(&format!(
            "element_type=\"{precision}\" shape=\"1,2,2\" offset=\"0\" size=\"{bytes}\""
        )));
        assert_eq!(&lowered.weights[..bytes], &weights);
        assert!(xml.contains("destination_type=\"f16\""));
        assert!(xml.contains("type=\"Multiply\""));
        assert!(xml.contains("transpose_b=\"true\""));
        assert!(xml.contains("_rhs_shape"));
        assert_eq!(lowered.features.len(), 2);
        assert!(lowered.features.iter().all(|f| f.element == OvElement::F16));
    }
}

fn cast_constant(values: &[u8], expose_integer: bool, integer_consumer: bool) -> LoweredModel {
    let shape = [1, values.len() as i32];
    let mut tensors = vec![
        Tensor::new("input", &shape, DType::FP16),
        Tensor::constant("q", &shape, DType::INT8, values),
        Tensor::new("decoded", &shape, DType::FP16),
        Tensor::new("out", &shape, DType::FP16),
    ];
    let mut ops = vec![
        Operator::new(OperatorKind::Const, &[], &["q"]),
        Operator::new(OperatorKind::Cast, &["q"], &["decoded"]),
        Operator::new(OperatorKind::Add, &["input", "decoded"], &["out"]),
    ];
    if integer_consumer {
        tensors.push(Tensor::new("wide", &shape, DType::INT32));
        ops.push(Operator::new(OperatorKind::Cast, &["q"], &["wide"]));
    }
    let outputs = if expose_integer {
        vec!["out", "q"]
    } else if integer_consumer {
        vec!["out", "wide"]
    } else {
        vec!["out"]
    };
    let graph = Graph::new("cast", &tensors, &ops, &["input"], &outputs)
        .build(quantized::TARGET)
        .unwrap();
    lower_tosa(&graph, quantized::TARGET).unwrap()
}

#[test]
fn packing_requires_exact_range_and_only_float_consumers() {
    for (values, exposed, integer, pack) in [
        (&[248, 0, 7][..], false, false, true),
        (&[247, 0, 7][..], false, false, false),
        (&[248, 0, 8][..], false, false, false),
        (&[248, 0, 7][..], true, false, false),
        (&[248, 0, 7][..], false, true, false),
    ] {
        let lowered = cast_constant(values, exposed, integer);
        assert_eq!(
            std::str::from_utf8(&lowered.xml)
                .unwrap()
                .contains("element_type=\"i4\""),
            pack
        );
        if pack {
            // Odd element count: unused high nibble is deterministic zero.
            assert_eq!(&lowered.weights[..2], &[0x08, 0x07]);
        }
    }
}

/// Optional artifact export for the companion native compiler inspection script.
/// Uses production lowering, so kernel evidence cannot accidentally test a different graph.
#[test]
fn export_compressed_projection_for_native_inspection() {
    let Ok(dir) = std::env::var("VIRTIO_ACCEL_OPENVINO_TEST_IR_DIR") else {
        return;
    };
    let dir = std::path::Path::new(&dir);
    std::fs::create_dir_all(dir).unwrap();
    let (n, k) = (128, 256);
    let packed = (0..n * k / 2).map(|i| i as u8).collect::<Vec<_>>();
    let graph =
        quantized::int4_matmul_fp16(1, n, k, 128, &packed, &vec![0x3000; n * k / 128]).unwrap();
    let lowered = lower_tosa(&graph, quantized::TARGET).unwrap();
    std::fs::write(dir.join("int4.xml"), lowered.xml).unwrap();
    std::fs::write(dir.join("int4.bin"), lowered.weights).unwrap();
    let lowered = lower_tosa_with_weights(&graph, quantized::TARGET, true).unwrap();
    std::fs::write(dir.join("int4-expanded.xml"), lowered.xml).unwrap();
    std::fs::write(dir.join("int4-expanded.bin"), lowered.weights).unwrap();
}

#[test]
fn expanded_constants_round_fractional_scales_at_load_time() {
    let graph = quantized::int4_matmul_fp16(1, 1, 4, 2, &[0x78, 0xf1], &[0x3555, 0x399a]).unwrap();
    let lowered = lower_tosa_with_weights(&graph, quantized::TARGET, true).unwrap();
    let xml = std::str::from_utf8(&lowered.xml).unwrap();
    let pos = xml.find("_rounded_weights\"").unwrap();
    let data = &xml[pos..];
    let offset = data
        .split("offset=\"")
        .nth(1)
        .unwrap()
        .split('"')
        .next()
        .unwrap()
        .parse::<usize>()
        .unwrap();
    // struct.pack('<e', q * struct.unpack('<e', scale)), independently computed.
    let expected = [0xc155u16, 0x40aa, 0x399a, 0xb99a]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect::<Vec<_>>();
    assert_eq!(&lowered.weights[offset..offset + 8], expected);
}
