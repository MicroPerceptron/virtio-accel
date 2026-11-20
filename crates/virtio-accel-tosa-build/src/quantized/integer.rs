//! Constant, symmetric integer weights with explicit FP16 dequantization.
use super::*;

/// Build an FP16 projection with immutable, symmetric signed four-bit weights.
///
/// `packed_weights` is row-major `[N,K]`, low nibble first, two's-complement `[-8,7]`.
/// N*K must be even; the packed byte slice must have exactly N*K/2 bytes.
/// `scale_bits` contains positive finite binary16 scales `[N,K/group_size]`.
/// Each weight is converted to FP16 then multiplied by its group's scale before
/// MATMUL. Slots are 0: FP16 activations `[1,M,K]`, 1: FP16 output `[1,M,N]`.
///
/// TOSA 1.0 has no INT4-to-float CAST. The portable artifact therefore serializes
/// these exact values as INT8 constants, using one byte per weight. OpenVINO and Core ML can
/// losslessly repack them to native compressed constants at load time. This is not a TOSA
/// INT4 tensor or NVFP4; no automatic quantization of floating weights occurs.
/// Changing weights or scales requires rebuilding and reloading the program.
pub fn int4_matmul_fp16(
    m: usize,
    n: usize,
    k: usize,
    group_size: usize,
    packed_weights: &[u8],
    scale_bits: &[u16],
) -> Result<Vec<u8>, QuantizedBuildError> {
    let elements = validate(m, n, k, group_size, scale_bits)?;
    if elements % 2 != 0 || packed_weights.len() != elements / 2 {
        return Err(QuantizedBuildError::InvalidShape);
    }
    let weights = packed_weights
        .iter()
        .flat_map(|byte| [byte & 15, byte >> 4].map(|v| ((v as i8) << 4) >> 4))
        .collect::<Vec<_>>();
    build(m, n, k, group_size, &weights, scale_bits)
}

/// As [`int4_matmul_fp16`], with signed eight-bit constant weights `[N,K]`.
/// Values are represented exactly; the backend may losslessly use narrower
/// storage when all values fit. This is W8A16, not integer-profile INT8 MATMUL.
pub fn int8_matmul_fp16(
    m: usize,
    n: usize,
    k: usize,
    group_size: usize,
    weights: &[i8],
    scale_bits: &[u16],
) -> Result<Vec<u8>, QuantizedBuildError> {
    let elements = validate(m, n, k, group_size, scale_bits)?;
    if weights.len() != elements {
        return Err(QuantizedBuildError::InvalidShape);
    }
    build(m, n, k, group_size, weights, scale_bits)
}

fn validate(
    m: usize,
    n: usize,
    k: usize,
    group: usize,
    scales: &[u16],
) -> Result<usize, QuantizedBuildError> {
    if m == 0
        || n == 0
        || k == 0
        || group == 0
        || k % group != 0
        || [m, n, k, group].iter().any(|v| i32::try_from(*v).is_err())
    {
        return Err(QuantizedBuildError::InvalidShape);
    }
    let elements = n.checked_mul(k).ok_or(QuantizedBuildError::InvalidShape)?;
    if scales.len() != elements / group {
        return Err(QuantizedBuildError::InvalidShape);
    }
    if scales
        .iter()
        .any(|b| *b == 0 || b & 0x8000 != 0 || b & 0x7c00 == 0x7c00)
    {
        return Err(QuantizedBuildError::InvalidScale);
    }
    Ok(elements)
}

fn build(
    m: usize,
    n: usize,
    k: usize,
    group: usize,
    weights: &[i8],
    scales: &[u16],
) -> Result<Vec<u8>, QuantizedBuildError> {
    let (m, n, k, group) = (m as i32, n as i32, k as i32, group as i32);
    let d = DType::FP16;
    let mut g = OwnedGraph::new("integer_weight_projection");
    tensor(&mut g, "activations", &[1, m, k], d);
    g.push_input("activations");
    constant(
        &mut g,
        "weights",
        &[n, k / group, group],
        DType::INT8,
        weights.iter().map(|v| *v as u8).collect(),
    );
    tensor(&mut g, "decoded", &[n, k / group, group], d);
    op(&mut g, OperatorKind::Cast, &["weights"], "decoded");
    constant(
        &mut g,
        "scales",
        &[n, k / group, 1],
        d,
        scales.iter().flat_map(|v| v.to_le_bytes()).collect(),
    );
    constant(&mut g, "shift", &[1], DType::INT8, vec![0]);
    tensor(&mut g, "scaled", &[n, k / group, group], d);
    op(
        &mut g,
        OperatorKind::Mul,
        &["decoded", "scales", "shift"],
        "scaled",
    );
    reshape(&mut g, "scaled", "matrix", &[1, n, k], d);
    tensor(&mut g, "transposed", &[1, k, n], d);
    op(
        &mut g,
        OperatorKind::Transpose {
            perms: [0, 2, 1, 0, 0, 0],
            rank: 3,
        },
        &["matrix"],
        "transposed",
    );
    constant(&mut g, "zero", &[1], d, vec![0, 0]);
    tensor(&mut g, "output", &[1, m, n], d);
    op(
        &mut g,
        OperatorKind::MatMul,
        &["activations", "transposed", "zero", "zero"],
        "output",
    );
    g.push_output("output");
    Ok(g.build(TARGET)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn signed_nibbles_and_groups_are_explicit_portable_operations() {
        let packed = [0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe];
        let bytes = int4_matmul_fp16(2, 2, 8, 4, &packed, &[0x3800; 4]).unwrap();
        let model = virtio_accel_tosa::parse(&bytes).unwrap();
        let a = model.analyze_for(TARGET).unwrap();
        assert!(a.conditions().is_empty());
        let weights = a.values().iter().find(|v| v.name() == "weights").unwrap();
        assert_eq!(
            a.serialized_constant(weights.id()).unwrap(),
            &[
                0, 1, 2, 3, 4, 5, 6, 7, 248, 249, 250, 251, 252, 253, 254, 255
            ]
        );
        assert_eq!(a.block_inputs(a.blocks()[0].id()).len(), 1);
        let w8 = int8_matmul_fp16(1, 1, 4, 4, &[-128, -8, 7, 127], &[0x3c00]).unwrap();
        virtio_accel_tosa::parse(&w8)
            .unwrap()
            .analyze_for(TARGET)
            .unwrap();
    }
    #[test]
    fn malformed_weight_geometry_and_scales_are_rejected() {
        for (m, n, k, g) in [
            (0, 1, 2, 2),
            (1, 0, 2, 2),
            (1, 1, 0, 2),
            (1, 1, 2, 0),
            (1, 1, 2, 3),
            (1, usize::MAX, 2, 2),
        ] {
            assert!(int4_matmul_fp16(m, n, k, g, &[0], &[0x3c00]).is_err());
        }
        assert!(int4_matmul_fp16(1, 1, 2, 2, &[], &[0x3c00]).is_err());
        assert!(int8_matmul_fp16(1, 1, 2, 2, &[1], &[0x3c00]).is_err());
        assert!(int4_matmul_fp16(1, 1, 2, 2, &[0], &[]).is_err());
        for scale in [0, 0x8000, 0xbc00, 0x7c00, 0xfc00, 0x7e00] {
            assert!(int4_matmul_fp16(1, 1, 2, 2, &[0], &[scale]).is_err());
        }
    }
}
