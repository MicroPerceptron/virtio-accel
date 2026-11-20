//! Explicit packed storage graphs. NVFP4 is expressed using released TOSA operations,
//! not mislabeled as INT4 or assigned a new TOSA dtype. The host receives packed bytes;
//! decoding and scaling occur in the submitted graph. Native compilers may materialize
//! wider intermediate weights: this API makes no compressed-residency or throughput promise.
use crate::{BuildError, OperatorKind, OwnedGraph, OwnedOperator, OwnedShape, OwnedTensor};
use alloc::{vec, vec::Vec};
use virtio_accel_tosa::{DType, ExtensionSet, Level, ProfileSet, Target, Version};

pub const TARGET: Target = Target::new(
    Version::TOSA_1_0,
    ProfileSet::FLOATING_POINT.union(ProfileSet::INTEGER),
    Level::Level8K,
    ExtensionSet::NONE,
);

#[derive(Debug)]
pub enum QuantizedBuildError {
    InvalidShape,
    InvalidScale,
    Graph(BuildError),
}
impl From<BuildError> for QuantizedBuildError {
    fn from(e: BuildError) -> Self {
        Self::Graph(e)
    }
}
impl core::fmt::Display for QuantizedBuildError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{self:?}")
    }
}
impl core::error::Error for QuantizedBuildError {}

fn tensor(g: &mut OwnedGraph<'_>, name: &str, shape: &[i32], dtype: DType) {
    g.push_tensor(OwnedTensor::new(name, shape.to_vec(), dtype));
}
fn op(g: &mut OwnedGraph<'_>, kind: OperatorKind, ins: &[&str], out: &str) {
    g.push_operator(OwnedOperator::new(
        kind,
        ins.iter().map(|s| (*s).into()).collect(),
        vec![out.into()],
    ));
}
fn constant(g: &mut OwnedGraph<'_>, name: &str, shape: &[i32], dtype: DType, bytes: Vec<u8>) {
    g.push_tensor(OwnedTensor::constant(name, shape.to_vec(), dtype, bytes));
    op(g, OperatorKind::Const, &[], name);
}
fn reshape(g: &mut OwnedGraph<'_>, input: &str, out: &str, shape: &[i32], dtype: DType) {
    let sn = alloc::format!("{out}_shape");
    g.push_shape(OwnedShape::new(
        &sn,
        shape.iter().map(|x| i64::from(*x)).collect(),
    ));
    op(g, OperatorKind::ConstShape, &[], &sn);
    tensor(g, out, shape, dtype);
    op(g, OperatorKind::Reshape, &[input, &sn], out);
}
fn lookup(
    g: &mut OwnedGraph<'_>,
    input: &str,
    prefix: &str,
    count: i32,
    table: &[f32],
    width: i32,
    dtype: DType,
) {
    let cast = alloc::format!("{prefix}_signed");
    let idx = alloc::format!("{prefix}_index");
    let lut = alloc::format!("{prefix}_lut");
    let result = alloc::format!("{prefix}_decoded");
    tensor(g, &cast, &[1, count], DType::INT32);
    op(g, OperatorKind::Cast, &[input], &cast);
    tensor(g, &idx, &[1, count], DType::INT32);
    op(g, OperatorKind::Add, &[&cast, "offset"], &idx);
    constant(
        g,
        &lut,
        &[1, 256, width],
        dtype,
        if dtype == DType::FP32 {
            table.iter().flat_map(|x| x.to_le_bytes()).collect()
        } else {
            table
                .iter()
                .flat_map(|x| exact_f16(*x).to_le_bytes())
                .collect()
        },
    );
    tensor(g, &result, &[1, count, width], dtype);
    op(g, OperatorKind::Gather, &[&lut, &idx], &result);
}
/// Build `output = activations @ decoded_weights.T * tensor_scale`.
///
/// Slots: 0 FP32 activations [1, M, K]; 1 packed E2M1 bytes [1, N*K/2];
/// 2 E4M3 block scales [1, N*K/16]; 3 FP32 output [1, M, N].
/// Byte buffers use INT8 solely as a bit-preserving storage carrier. Low nibble is
/// the first K element; each row has K/16 scales. Weights and scales remain runtime
/// inputs, so changing a block does not require rebuilding the program.
/// Scaling order matches Kore's projection: multiply E2M1 by its block scale,
/// accumulate the FP32 matrix product, then apply the separate FP32 tensor scale.
pub fn nvfp4_matmul(
    m: usize,
    n: usize,
    k: usize,
    tensor_scale: f32,
) -> Result<Vec<u8>, QuantizedBuildError> {
    if !tensor_scale.is_finite() || tensor_scale <= 0.0 {
        return Err(QuantizedBuildError::InvalidScale);
    }
    build_nvfp4(m, n, k, DType::FP32, tensor_scale.to_le_bytes().to_vec())
}
/// FP16 activation, accumulation-result, output, and tensor-scale variant. The
/// scale is supplied as IEEE binary16 bits so no implicit host rounding is involved.
/// As in TOSA FP16 MATMUL, accumulation has the specified wider accumulator and
/// the result rounds to FP16 before the final FP16 scaling operation.
pub fn nvfp4_matmul_fp16(
    m: usize,
    n: usize,
    k: usize,
    tensor_scale_bits: u16,
) -> Result<Vec<u8>, QuantizedBuildError> {
    if tensor_scale_bits == 0
        || tensor_scale_bits & 0x8000 != 0
        || tensor_scale_bits & 0x7c00 == 0x7c00
    {
        return Err(QuantizedBuildError::InvalidScale);
    }
    build_nvfp4(
        m,
        n,
        k,
        DType::FP16,
        tensor_scale_bits.to_le_bytes().to_vec(),
    )
}
// Only used for the E2M1 and E4M3 decode tables, all of whose finite values are
// exactly representable as normal binary16 values. No general float quantizer.
fn exact_f16(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    if b & 0x7fff_ffff == 0 {
        return sign;
    }
    if x.is_nan() {
        return sign | 0x7e00;
    }
    sign | ((((b >> 23) & 255) - 112) as u16) << 10 | ((b >> 13) & 1023) as u16
}
fn build_nvfp4(
    m: usize,
    n: usize,
    k: usize,
    dtype: DType,
    scale: Vec<u8>,
) -> Result<Vec<u8>, QuantizedBuildError> {
    if m == 0 || n == 0 || k == 0 || k % 16 != 0 {
        return Err(QuantizedBuildError::InvalidShape);
    }
    let elements = n.checked_mul(k).ok_or(QuantizedBuildError::InvalidShape)?;
    let dim = |x| i32::try_from(x).map_err(|_| QuantizedBuildError::InvalidShape);
    let (m, n, k, packed, groups) = (
        dim(m)?,
        dim(n)?,
        dim(k)?,
        dim(elements / 2)?,
        dim(elements / 16)?,
    );
    let mut g = OwnedGraph::new("nvfp4_projection");
    tensor(&mut g, "activations", &[1, m, k], dtype);
    tensor(&mut g, "weights", &[1, packed], DType::INT8);
    tensor(&mut g, "scales", &[1, groups], DType::INT8);
    for input in ["activations", "weights", "scales"] {
        g.push_input(input);
    }
    constant(
        &mut g,
        "offset",
        &[1, 1],
        DType::INT32,
        128i32.to_le_bytes().to_vec(),
    );
    let e2m1 = [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ];
    let low = (0..256)
        .map(|i| e2m1[((i as u8).wrapping_add(128) & 15) as usize])
        .collect::<Vec<_>>();
    let high = (0..256)
        .map(|i| e2m1[((i as u8).wrapping_add(128) >> 4) as usize])
        .collect::<Vec<_>>();
    let scales = (0..256)
        .map(|i| virtio_accel_tosa::fp8e4m3_to_f32((i as u8).wrapping_add(128)))
        .collect::<Vec<_>>();
    // Separate scalar lookups also avoid an OpenVINO 2026.4 CPU Gather issue with
    // two-column tables. The native regression tests exercise all 256 packed bytes.
    lookup(&mut g, "weights", "lo", packed, &low, 1, dtype);
    lookup(&mut g, "weights", "hi", packed, &high, 1, dtype);
    tensor(&mut g, "w_decoded", &[1, packed, 2], dtype);
    op(
        &mut g,
        OperatorKind::Concat { axis: 2 },
        &["lo_decoded", "hi_decoded"],
        "w_decoded",
    );
    lookup(&mut g, "scales", "s", groups, &scales, 1, dtype);
    reshape(&mut g, "w_decoded", "blocks", &[n, k / 16, 16], dtype);
    reshape(&mut g, "s_decoded", "block_scales", &[n, k / 16, 1], dtype);
    constant(&mut g, "shift", &[1], DType::INT8, vec![0]);
    tensor(&mut g, "scaled_blocks", &[n, k / 16, 16], dtype);
    op(
        &mut g,
        OperatorKind::Mul,
        &["blocks", "block_scales", "shift"],
        "scaled_blocks",
    );
    reshape(&mut g, "scaled_blocks", "matrix", &[1, n, k], dtype);
    tensor(&mut g, "transposed", &[1, k, n], dtype);
    op(
        &mut g,
        OperatorKind::Transpose {
            perms: [0, 2, 1, 0, 0, 0],
            rank: 3,
        },
        &["matrix"],
        "transposed",
    );
    constant(
        &mut g,
        "zero",
        &[1],
        dtype,
        vec![0; if dtype == DType::FP32 { 4 } else { 2 }],
    );
    tensor(&mut g, "product", &[1, m, n], dtype);
    op(
        &mut g,
        OperatorKind::MatMul,
        &["activations", "transposed", "zero", "zero"],
        "product",
    );
    constant(&mut g, "tensor_scale", &[1, 1, 1], dtype, scale);
    tensor(&mut g, "output", &[1, m, n], dtype);
    op(
        &mut g,
        OperatorKind::Mul,
        &["product", "tensor_scale", "shift"],
        "output",
    );
    g.push_output("output");
    Ok(g.build(TARGET)?)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn graph_is_released_tosa_with_proven_lookup_bounds() {
        let bytes = nvfp4_matmul(2, 3, 32, 0.25).unwrap();
        let model = virtio_accel_tosa::parse(&bytes).unwrap();
        let a = model.analyze_for(TARGET).unwrap();
        assert_eq!(a.conditions().len(), 3);
        assert!(
            a.conditions()
                .iter()
                .all(|c| a.proves_byte_lookup_bounds(c))
        );
    }
    #[test]
    fn invalid_geometry_and_scale_are_rejected() {
        for (m, n, k) in [(0, 1, 16), (1, 0, 16), (1, 1, 15), (1, usize::MAX, 16)] {
            assert!(nvfp4_matmul(m, n, k, 1.0).is_err());
        }
        for s in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            assert!(nvfp4_matmul(1, 1, 16, s).is_err());
        }
        for bits in [0, 0x8000, 0xbc00, 0x7c00, 0x7e00] {
            assert!(nvfp4_matmul_fp16(1, 1, 16, bits).is_err());
        }
        let bytes = nvfp4_matmul_fp16(2, 3, 32, 0x3400).unwrap();
        let model = virtio_accel_tosa::parse(&bytes).unwrap();
        let a = model.analyze_for(TARGET).unwrap();
        assert!(
            a.conditions()
                .iter()
                .all(|c| a.proves_byte_lookup_bounds(c))
        );
    }

    #[test]
    fn lookup_proof_rejects_wrong_offsets_unbounded_sources_and_small_tables() {
        for (offset, source_type, table_size, expected) in [
            (128, DType::INT8, 256, true),
            (127, DType::INT8, 256, false),
            (129, DType::INT8, 256, false),
            (128, DType::INT32, 256, false),
            (128, DType::INT8, 255, false),
        ] {
            let mut g = OwnedGraph::new("lookup_proof");
            tensor(&mut g, "source", &[1, 8], source_type);
            g.push_input("source");
            tensor(&mut g, "wide", &[1, 8], DType::INT32);
            op(
                &mut g,
                if source_type == DType::INT8 {
                    OperatorKind::Cast
                } else {
                    OperatorKind::Identity
                },
                &["source"],
                "wide",
            );
            constant(
                &mut g,
                "offset",
                &[1, 1],
                DType::INT32,
                i32::to_le_bytes(offset).to_vec(),
            );
            tensor(&mut g, "indices", &[1, 8], DType::INT32);
            op(&mut g, OperatorKind::Add, &["offset", "wide"], "indices");
            tensor(&mut g, "table", &[1, table_size, 1], DType::FP32);
            g.push_input("table");
            tensor(&mut g, "result", &[1, 8, 1], DType::FP32);
            op(
                &mut g,
                OperatorKind::Gather,
                &["table", "indices"],
                "result",
            );
            g.push_output("result");
            let bytes = g.build(TARGET).unwrap();
            let model = virtio_accel_tosa::parse(&bytes).unwrap();
            let a = model.analyze_for(TARGET).unwrap();
            assert_eq!(a.conditions().len(), 1);
            assert_eq!(
                a.proves_byte_lookup_bounds(&a.conditions()[0]),
                expected,
                "offset={offset}, source={source_type:?}, table_size={table_size}"
            );
        }
    }
}
