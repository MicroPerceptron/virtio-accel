use half::{bf16, f16};
use virtio_accel_tosa::{DType, ExtensionSet, Level, ProfileSet, Target, Version};
use virtio_accel_tosa_build::{Graph, Operator, OperatorKind, Shape, Tensor, quantized};

#[derive(Clone, Copy, Debug)]
pub enum Case {
    MatmulF16Decode,
    MatmulF16Prefill,
    MatmulBf16Decode,
    MatmulF32Decode,
    Nvfp4F16Decode,
    Nvfp4F16Prefill,
    Nvfp4F32Decode,
    Fp8E4M3Encode,
    Fp8E5M2Encode,
    Fp8E4M3Decode,
    Fp8E5M2Decode,
    Fp8CacheWindow,
    Int4F16ProjectionBatch,
    F16ProjectionBatch,
    Int4F16ProjectionDecode,
    Int4F16ProjectionPrefill,
    Palette4ChannelDecode,
    Palette4ChannelPrefill,
    F16ChannelDecode,
    F16ChannelPrefill,
    Palette4F16ProjectionDecode,
    Palette4F16ProjectionPrefill,
    Int8F16ProjectionDecode,
    Int8F16ProjectionPrefill,
    F16ProjectionDecode,
    F16ProjectionPrefill,
}
pub const CASES: &[Case] = &[
    Case::MatmulF16Decode,
    Case::MatmulF16Prefill,
    Case::MatmulBf16Decode,
    Case::MatmulF32Decode,
    Case::Nvfp4F16Decode,
    Case::Nvfp4F16Prefill,
    Case::Nvfp4F32Decode,
    Case::Fp8E4M3Encode,
    Case::Fp8E5M2Encode,
    Case::Fp8E4M3Decode,
    Case::Fp8E5M2Decode,
    Case::Fp8CacheWindow,
    Case::Int4F16ProjectionBatch,
    Case::F16ProjectionBatch,
    Case::Int4F16ProjectionDecode,
    Case::Int4F16ProjectionPrefill,
    Case::Palette4ChannelDecode,
    Case::Palette4ChannelPrefill,
    Case::F16ChannelDecode,
    Case::F16ChannelPrefill,
    Case::Palette4F16ProjectionDecode,
    Case::Palette4F16ProjectionPrefill,
    Case::Int8F16ProjectionDecode,
    Case::Int8F16ProjectionPrefill,
    Case::F16ProjectionDecode,
    Case::F16ProjectionPrefill,
];

pub struct Workload {
    pub name: String,
    pub artifact: Vec<u8>,
    pub target: Target,
    pub inputs: Vec<Vec<u8>>,
    pub expected: Vec<u8>,
    pub output_dtype: DType,
    pub shape: Vec<usize>,
    pub logical_flops: u64,
}
impl Workload {
    pub fn validate(&self, actual: &[u8]) -> Result<(), String> {
        if actual != self.expected {
            let first = actual.iter().zip(&self.expected).position(|(a, b)| a != b);
            return Err(format!(
                "{}: output mismatch at byte {first:?}; lengths {} vs {}",
                self.name,
                actual.len(),
                self.expected.len()
            ));
        }
        Ok(())
    }
    pub fn input_bytes(&self) -> usize {
        self.inputs.iter().map(Vec::len).sum()
    }
}
fn target(ext: ExtensionSet) -> Target {
    Target::new(
        Version::TOSA_1_0,
        ProfileSet::FLOATING_POINT,
        Level::Level8K,
        ext,
    )
}
pub fn fp8_target() -> Target {
    target(ExtensionSet::FP8E4M3.union(ExtensionSet::FP8E5M2))
}
fn encode(values: &[f32], d: DType) -> Vec<u8> {
    match d {
        DType::FP16 => values
            .iter()
            .flat_map(|v| f16::from_f32(*v).to_bits().to_le_bytes())
            .collect(),
        DType::BF16 => values
            .iter()
            .flat_map(|v| bf16::from_f32(*v).to_bits().to_le_bytes())
            .collect(),
        DType::FP32 => values.iter().flat_map(|v| v.to_le_bytes()).collect(),
        _ => unreachable!(),
    }
}
impl Case {
    pub fn build(self) -> Workload {
        let mut w = match self {
            Self::MatmulF16Decode => matmul(1, 128, 256, DType::FP16),
            Self::MatmulF16Prefill => matmul(16, 128, 256, DType::FP16),
            Self::MatmulF32Decode => matmul(1, 128, 256, DType::FP32),
            Self::MatmulBf16Decode => matmul(1, 128, 256, DType::BF16),
            Self::Nvfp4F16Decode => nvfp4(1, 128, 256, DType::FP16),
            Self::Nvfp4F16Prefill => nvfp4(16, 128, 256, DType::FP16),
            Self::Nvfp4F32Decode => nvfp4(1, 128, 256, DType::FP32),
            Self::Fp8E4M3Encode => cast(DType::FP8E4M3, true),
            Self::Fp8E5M2Encode => cast(DType::FP8E5M2, true),
            Self::Fp8E4M3Decode => cast(DType::FP8E4M3, false),
            Self::Fp8E5M2Decode => cast(DType::FP8E5M2, false),
            Self::Fp8CacheWindow => cache(),
            Self::Int4F16ProjectionBatch => projection(128, 1024, 2048, ProjectionWeights::Int4),
            Self::F16ProjectionBatch => projection(128, 1024, 2048, ProjectionWeights::F16),
            Self::Int4F16ProjectionDecode => projection(1, 1024, 2048, ProjectionWeights::Int4),
            Self::Int4F16ProjectionPrefill => projection(16, 1024, 2048, ProjectionWeights::Int4),
            Self::Palette4ChannelDecode => {
                projection(1, 1024, 2048, ProjectionWeights::Palette4Channel)
            }
            Self::Palette4ChannelPrefill => {
                projection(16, 1024, 2048, ProjectionWeights::Palette4Channel)
            }
            Self::F16ChannelDecode => projection(1, 1024, 2048, ProjectionWeights::F16Channel),
            Self::F16ChannelPrefill => projection(16, 1024, 2048, ProjectionWeights::F16Channel),
            Self::Palette4F16ProjectionDecode => {
                projection(1, 1024, 2048, ProjectionWeights::Palette4)
            }
            Self::Palette4F16ProjectionPrefill => {
                projection(16, 1024, 2048, ProjectionWeights::Palette4)
            }
            Self::Int8F16ProjectionDecode => projection(1, 1024, 2048, ProjectionWeights::Int8),
            Self::Int8F16ProjectionPrefill => projection(16, 1024, 2048, ProjectionWeights::Int8),
            Self::F16ProjectionDecode => projection(1, 1024, 2048, ProjectionWeights::F16),
            Self::F16ProjectionPrefill => projection(16, 1024, 2048, ProjectionWeights::F16),
        };
        w.name = format!("{self:?}");
        w
    }
}
#[derive(Clone, Copy, PartialEq)]
enum ProjectionWeights {
    Palette4Channel,
    F16Channel,
    Int4,
    Int8,
    Palette4,
    F16,
}

fn projection(m: usize, n: usize, k: usize, format: ProjectionWeights) -> Workload {
    let channel = matches!(
        format,
        ProjectionWeights::Palette4Channel | ProjectionWeights::F16Channel
    );
    let group = if channel { k } else { 128 };
    // Identical dequantized weights for all three representations. W8 uses q*16
    // and scale/16, ensuring it cannot accidentally take the narrower i4 path.
    let q = (0..n * k)
        .map(|i| {
            let hash = (i as u32).wrapping_mul(1664525).wrapping_add(1013904223);
            ((hash >> 24) & 15) as i8 - 8
        })
        .collect::<Vec<_>>();
    let scales = (0..n * k / group)
        .map(|i| {
            if (if channel { i / 16 } else { i }) % 3 == 0 {
                0.125
            } else {
                0.0625
            }
        })
        .collect::<Vec<f32>>();
    // 128 nonzero binary-fraction activations per row keep every possible
    // FP16 partial sum representable, independently of reduction order.
    // The full dense weight matrix is still read by the native kernels.
    let a = (0..m * k)
        .map(|i| if i % 16 == (i / k) % 16 { 0.125 } else { 0.0 })
        .collect::<Vec<_>>();
    let weights = (0..n * k)
        .map(|i| f32::from(q[i]) * scales[i / group])
        .collect::<Vec<_>>();
    let expected = (0..m * n)
        .map(|i| {
            (0..k)
                .map(|j| f64::from(a[(i / n) * k + j]) * f64::from(weights[(i % n) * k + j]))
                .sum::<f64>() as f32
        })
        .collect::<Vec<_>>();
    let d = DType::FP16;
    let artifact = match format {
        ProjectionWeights::Int4 => {
            let packed = q
                .chunks_exact(2)
                .map(|p| (p[0] as u8 & 15) | (p[1] as u8 & 15) << 4)
                .collect::<Vec<_>>();
            let scale_bits = scales
                .iter()
                .map(|v| f16::from_f32(*v).to_bits())
                .collect::<Vec<_>>();
            quantized::int4_matmul_fp16(m, n, k, group, &packed, &scale_bits).unwrap()
        }
        ProjectionWeights::Int8 => {
            let weights = q.iter().map(|v| *v * 16).collect::<Vec<_>>();
            let scale_bits = scales
                .iter()
                .map(|v| f16::from_f32(*v / 16.0).to_bits())
                .collect::<Vec<_>>();
            quantized::int8_matmul_fp16(m, n, k, group, &weights, &scale_bits).unwrap()
        }
        ProjectionWeights::Palette4 | ProjectionWeights::Palette4Channel => {
            let packed = q
                .chunks_exact(2)
                .map(|p| (p[0] as u8 & 15) | ((p[1] as u8 & 15) << 4))
                .collect::<Vec<_>>();
            let palette = scales
                .iter()
                .flat_map(|scale| {
                    (0..16u8).map(move |code| {
                        let signed = ((code as i8) << 4) >> 4;
                        f16::from_f32(f32::from(signed) * scale).to_bits()
                    })
                })
                .collect::<Vec<_>>();
            quantized::palette4_matmul_fp16(m, n, k, group, &packed, &palette).unwrap()
        }
        ProjectionWeights::F16 | ProjectionWeights::F16Channel => {
            let a_shape = [1, m as i32, k as i32];
            let b_shape = [1, k as i32, n as i32];
            let out_shape = [1, m as i32, n as i32];
            let transposed = (0..k * n)
                .map(|i| weights[(i % n) * k + i / n])
                .collect::<Vec<_>>();
            let data = encode(&transposed, d);
            let zero = [0, 0];
            let tensors = [
                Tensor::new("a", &a_shape, d),
                Tensor::constant("b", &b_shape, d, &data),
                Tensor::constant("zero", &[1], d, &zero),
                Tensor::new("out", &out_shape, d),
            ];
            let ops = [
                Operator::new(OperatorKind::Const, &[], &["b"]),
                Operator::new(OperatorKind::Const, &[], &["zero"]),
                Operator::new(OperatorKind::MatMul, &["a", "b", "zero", "zero"], &["out"]),
            ];
            Graph::new("projection", &tensors, &ops, &["a"], &["out"])
                .build(quantized::TARGET)
                .unwrap()
        }
    };
    Workload {
        name: String::new(),
        artifact,
        target: quantized::TARGET,
        inputs: vec![encode(&a, d)],
        expected: encode(&expected, d),
        output_dtype: d,
        shape: vec![m, n, k],
        logical_flops: (2 * m * n * k) as u64,
    }
}
fn matmul(m: usize, n: usize, k: usize, d: DType) -> Workload {
    // Positive binary fractions make every result nonzero and exactly representable;
    // With K=256 these products and every positive partial sum fit FP16 exactly,
    // independent of accumulation order. Zero-filled output cannot pass validation.
    let a = (0..m * k)
        .map(|i| (i % 4 + 1) as f32 * 0.125)
        .collect::<Vec<_>>();
    let b = (0..k * n)
        .map(|i| (i % 3 + 1) as f32 * 0.0625)
        .collect::<Vec<_>>();
    let output = if d == DType::BF16 { DType::FP32 } else { d };
    let expected = (0..m * n)
        .map(|i| {
            let (row, col) = (i / n, i % n);
            (0..k)
                .map(|j| f64::from(a[row * k + j]) * f64::from(b[j * n + col]))
                .sum::<f64>() as f32
        })
        .collect::<Vec<_>>();
    let zero = encode(&[0.0], d);
    let a_shape = [1, m as i32, k as i32];
    let b_shape = [1, k as i32, n as i32];
    let out_shape = [1, m as i32, n as i32];
    let tensors = [
        Tensor::new("a", &a_shape, d),
        Tensor::new("b", &b_shape, d),
        Tensor::constant("zero", &[1], d, &zero),
        Tensor::new("out", &out_shape, output),
    ];
    let ops = [
        Operator::new(OperatorKind::Const, &[], &["zero"]),
        Operator::new(OperatorKind::MatMul, &["a", "b", "zero", "zero"], &["out"]),
    ];
    let target = target(if d == DType::BF16 {
        ExtensionSet::BF16
    } else {
        ExtensionSet::NONE
    });
    Workload {
        name: String::new(),
        artifact: Graph::new("matmul", &tensors, &ops, &["a", "b"], &["out"])
            .build(target)
            .unwrap(),
        target,
        inputs: vec![encode(&a, d), encode(&b, d)],
        expected: encode(&expected, output),
        output_dtype: output,
        shape: vec![m, n, k],
        logical_flops: (2 * m * n * k) as u64,
    }
}
fn nvfp4(m: usize, n: usize, k: usize, d: DType) -> Workload {
    let a = (0..m * k)
        .map(|i| (i % 4 + 1) as f32 * 0.125)
        .collect::<Vec<_>>();
    let weights = (0..n * k / 2)
        .map(|i| [0x21, 0x43, 0x65, 0x10][i % 4])
        .collect::<Vec<_>>();
    let scales = (0..n * k / 16)
        .map(|i| if i % 3 == 0 { 0x30 } else { 0x38 })
        .collect::<Vec<_>>();
    let lut = [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ];
    let expected = (0..m * n)
        .map(|i| {
            let (row, col) = (i / n, i % n);
            let sum = (0..k)
                .map(|j| {
                    let byte = weights[col * k / 2 + j / 2];
                    let nibble = if j % 2 == 0 { byte & 15 } else { byte >> 4 };
                    f64::from(a[row * k + j])
                        * lut[nibble as usize]
                        * f64::from(virtio_accel_tosa::fp8e4m3_to_f32(
                            scales[col * k / 16 + j / 16],
                        ))
                })
                .sum::<f64>() as f32;
            (if d == DType::FP16 {
                f16::from_f32(sum).to_f32()
            } else {
                sum
            }) * 0.125
        })
        .collect::<Vec<_>>();
    let artifact = if d == DType::FP16 {
        quantized::nvfp4_matmul_fp16(m, n, k, 0x3000)
    } else {
        quantized::nvfp4_matmul(m, n, k, 0.125)
    }
    .unwrap();
    Workload {
        name: String::new(),
        artifact,
        target: quantized::TARGET,
        inputs: vec![encode(&a, d), weights, scales],
        expected: encode(&expected, d),
        output_dtype: d,
        shape: vec![m, n, k],
        logical_flops: (2 * m * n * k) as u64,
    }
}
fn cast(d: DType, encode_fp8: bool) -> Workload {
    let bytes = (0..4096).map(|i| (i % 120) as u8).collect::<Vec<_>>();
    let values = bytes
        .iter()
        .map(|b| {
            if d == DType::FP8E4M3 {
                virtio_accel_tosa::fp8e4m3_to_f32(*b)
            } else {
                virtio_accel_tosa::fp8e5m2_to_f32(*b)
            }
        })
        .collect::<Vec<_>>();
    let (input, output) = if encode_fp8 {
        (DType::FP32, d)
    } else {
        (d, DType::FP32)
    };
    let tensors = [
        Tensor::new("in", &[1, 4096], input),
        Tensor::new("out", &[1, 4096], output),
    ];
    let ops = [Operator::new(OperatorKind::Cast, &["in"], &["out"])];
    let floats = encode(&values, DType::FP32);
    let (data, expected) = if encode_fp8 {
        (floats, bytes)
    } else {
        (bytes, floats)
    };
    Workload {
        name: String::new(),
        artifact: Graph::new("cast", &tensors, &ops, &["in"], &["out"])
            .build(fp8_target())
            .unwrap(),
        target: fp8_target(),
        inputs: vec![data],
        expected,
        output_dtype: output,
        shape: vec![4096],
        logical_flops: 0,
    }
}
fn cache() -> Workload {
    let d = DType::FP8E4M3;
    let tensors = [
        Tensor::new("cache", &[1, 16384], d),
        Tensor::new("append", &[1, 128], d),
        Tensor::new("joined", &[1, 16512], d),
        Tensor::new("out", &[1, 16384], d),
    ];
    let shapes = [
        Shape::new("start", &[0, 128]),
        Shape::new("size", &[1, 16384]),
    ];
    let ops = [
        Operator::new(OperatorKind::ConstShape, &[], &["start"]),
        Operator::new(OperatorKind::ConstShape, &[], &["size"]),
        Operator::new(
            OperatorKind::Concat { axis: 1 },
            &["cache", "append"],
            &["joined"],
        ),
        Operator::new(OperatorKind::Slice, &["joined", "start", "size"], &["out"]),
    ];
    let data = (0..16512).map(|i| (i % 256) as u8).collect::<Vec<_>>();
    Workload {
        name: String::new(),
        artifact: Graph::new("cache", &tensors, &ops, &["cache", "append"], &["out"])
            .with_shapes(&shapes)
            .build(fp8_target())
            .unwrap(),
        target: fp8_target(),
        inputs: vec![data[..16384].to_vec(), data[16384..].to_vec()],
        expected: data[128..].to_vec(),
        output_dtype: d,
        shape: vec![16384, 128],
        logical_flops: 0,
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn compressed_projection_controls_have_identical_values() {
        for m in [1, 16] {
            let palette = projection(m, 32, 256, ProjectionWeights::Palette4Channel);
            let dense = projection(m, 32, 256, ProjectionWeights::F16Channel);
            assert_eq!(palette.inputs, dense.inputs);
            assert_eq!(palette.expected, dense.expected);
            let control = projection(m, 4, 256, ProjectionWeights::F16);
            for format in [
                ProjectionWeights::Int4,
                ProjectionWeights::Int8,
                ProjectionWeights::Palette4,
            ] {
                let packed = projection(m, 4, 256, format);
                assert_eq!(packed.inputs, control.inputs);
                assert_eq!(packed.expected, control.expected);
            }
        }
    }
    #[test]
    fn dense_fp16_reference_is_independent_of_accumulation_order() {
        for case in [Case::MatmulF16Decode, Case::MatmulF16Prefill] {
            let w = case.build();
            let decode = |bytes: &[u8]| {
                bytes
                    .chunks_exact(2)
                    .map(|v| f16::from_bits(u16::from_le_bytes([v[0], v[1]])).to_f32())
                    .collect::<Vec<_>>()
            };
            let a = decode(&w.inputs[0]);
            let b = decode(&w.inputs[1]);
            let [m, n, k] = w.shape[..] else {
                unreachable!()
            };
            for row in 0..m {
                for col in 0..n {
                    let units: Vec<_> = (0..k)
                        .map(|j| a[row * k + j] * b[j * n + col] * 128.0)
                        .collect();
                    assert!(units.iter().all(|v| *v > 0.0 && v.fract() == 0.0));
                    // Any partial sum is an integer <= 2048 scaled by 2^-7:
                    // representable in FP16 for every reduction order.
                    assert!(units.iter().sum::<f32>() <= 2048.0);
                }
            }
        }
    }
    #[test]
    fn workloads_are_valid_and_reject_corrupt_output() {
        for case in CASES {
            let w = case.build();
            virtio_accel_tosa::parse(&w.artifact)
                .unwrap()
                .analyze_for(w.target)
                .unwrap();
            assert!(w.validate(&w.expected).is_ok());
            assert!(w.validate(&vec![0; w.expected.len()]).is_err());
            let mut wrong = w.expected.clone();
            wrong[0] ^= 1;
            assert!(w.validate(&wrong).is_err());
            assert!(w.validate(&w.expected[..w.expected.len() - 1]).is_err());
        }
    }
}
