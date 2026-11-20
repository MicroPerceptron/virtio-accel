//! Explicit constant four-bit lookup indices and binary16 palettes.
use super::*;

/// Build a projection with immutable four-bit palette indices and FP16 activations.
///
/// Indices are row-major `[N,K]`, low nibble first. Each contiguous `group_size`
/// weights has its own 16-entry binary16 palette: `[N,K/group_size,16]`.
/// Entries may be arbitrary finite FP16 values (including signed zero); no
/// clustering, quantization or scaling is performed. Slots are FP16 activations
/// `[1,M,K]` and FP16 output `[1,M,N]`.
///
/// Portable TOSA stores unpacked INT32 indices and uses GATHER. Core ML can
/// repack these constants into `constexpr_lut_to_dense` at load time. This is
/// scalar palettization, not linear INT4 or NVFP4. Constants require reload to update.
pub fn palette4_matmul_fp16(
    m: usize,
    n: usize,
    k: usize,
    group_size: usize,
    packed_indices: &[u8],
    palette_bits: &[u16],
) -> Result<Vec<u8>, QuantizedBuildError> {
    if [m, n, k, group_size]
        .iter()
        .any(|v| *v == 0 || i32::try_from(*v).is_err())
        || k % group_size != 0
    {
        return Err(QuantizedBuildError::InvalidShape);
    }
    let count = n.checked_mul(k).ok_or(QuantizedBuildError::InvalidShape)?;
    let groups = count / group_size;
    if count % 2 != 0
        || packed_indices.len() != count / 2
        || i32::try_from(groups).is_err()
        || groups.checked_mul(16) != Some(palette_bits.len())
    {
        return Err(QuantizedBuildError::InvalidShape);
    }
    if palette_bits.iter().any(|v| v & 0x7c00 == 0x7c00) {
        return Err(QuantizedBuildError::InvalidScale);
    }
    let (m, n, k, group) = (m as i32, n as i32, k as i32, group_size as i32);
    let d = DType::FP16;
    let mut g = OwnedGraph::new("palette_weight_projection");
    tensor(&mut g, "activations", &[1, m, k], d);
    g.push_input("activations");
    constant(
        &mut g,
        "palette",
        &[groups as i32, 16, 1],
        d,
        palette_bits.iter().flat_map(|v| v.to_le_bytes()).collect(),
    );
    constant(
        &mut g,
        "indices",
        &[groups as i32, group],
        DType::INT32,
        packed_indices
            .iter()
            .flat_map(|v| [i32::from(v & 15), i32::from(v >> 4)])
            .flat_map(i32::to_le_bytes)
            .collect(),
    );
    tensor(&mut g, "decoded", &[groups as i32, group, 1], d);
    op(
        &mut g,
        OperatorKind::Gather,
        &["palette", "indices"],
        "decoded",
    );
    reshape(&mut g, "decoded", "matrix", &[1, n, k], d);
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
    fn palette_indices_and_geometry_are_explicit() {
        let data = palette4_matmul_fp16(2, 2, 16, 8, &[0x98; 16], &[0x3555; 64]).unwrap();
        let model = virtio_accel_tosa::parse(&data).unwrap();
        let a = model.analyze_for(TARGET).unwrap();
        let indices = a.values().iter().find(|v| v.name() == "indices").unwrap();
        assert_eq!(
            &a.serialized_constant(indices.id()).unwrap()[..8],
            &[8, 0, 0, 0, 9, 0, 0, 0]
        );
        assert!(
            a.conditions()
                .iter()
                .all(|c| a.proves_constant_index_bounds(c))
        );
        assert!(palette4_matmul_fp16(1, 1, 16, 0, &[0; 8], &[0; 16]).is_err());
        assert!(palette4_matmul_fp16(1, 1, 16, 3, &[0; 8], &[0; 16]).is_err());
        assert!(palette4_matmul_fp16(1, 1, 16, 16, &[0; 7], &[0; 16]).is_err());
        assert!(palette4_matmul_fp16(1, 1, 16, 16, &[0; 8], &[0; 15]).is_err());
        assert!(palette4_matmul_fp16(1, 1, 16, 16, &[0; 8], &[0x7c00; 16]).is_err());
    }
    #[test]
    fn constant_index_proof_rejects_negative_high_and_runtime_indices() {
        for (index, runtime, expected) in [
            (0, false, true),
            (15, false, true),
            (-1, false, false),
            (16, false, false),
            (0, true, false),
        ] {
            let mut g = OwnedGraph::new("bounds");
            tensor(&mut g, "table", &[1, 16, 1], DType::FP16);
            g.push_input("table");
            if runtime {
                tensor(&mut g, "indices", &[1, 1], DType::INT32);
                g.push_input("indices");
            } else {
                constant(
                    &mut g,
                    "indices",
                    &[1, 1],
                    DType::INT32,
                    i32::to_le_bytes(index).to_vec(),
                );
            }
            tensor(&mut g, "out", &[1, 1, 1], DType::FP16);
            op(&mut g, OperatorKind::Gather, &["table", "indices"], "out");
            g.push_output("out");
            let bytes = g.build(TARGET).unwrap();
            let model = virtio_accel_tosa::parse(&bytes).unwrap();
            let a = model.analyze_for(TARGET).unwrap();
            assert_eq!(a.conditions().len(), 1);
            assert_eq!(a.proves_constant_index_bounds(&a.conditions()[0]), expected);
        }
    }
}
