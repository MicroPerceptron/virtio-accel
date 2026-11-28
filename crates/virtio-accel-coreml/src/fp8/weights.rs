//! Lossless constant-weight rewrites. Runtime tensors never enter this path.
use super::*;
use virtio_accel_tosa::{TosaAnalysis, ValueId};

pub(super) const I4: u64 = 25;
pub(super) const U4: u64 = 35;

pub(super) struct Weights {
    n: i32,
    k: i32,
    group: i32,
    data: Storage,
}
enum Storage {
    Dense(Vec<u8>),
    Linear { values: Vec<u8>, scales: Vec<u8> },
    Palette { indices: Vec<u8>, lut: Vec<u8> },
}

fn producer<'a>(a: &'a TosaAnalysis<'_>, v: ValueId, op: Op) -> Option<&'a [ValueId]> {
    let id = a.value(v).producer()?;
    (a.operator(id).op() == op).then(|| a.operator_inputs(id))
}

pub(super) fn plans(a: &TosaAnalysis<'_>) -> Result<Vec<Option<Weights>>, LoweringError> {
    let mut plans = (0..a.values().len()).map(|_| None).collect::<Vec<_>>();
    for id in a.execution_order(a.blocks()[0].id()) {
        if a.operator(*id).op() != Op::MATMUL {
            continue;
        }
        let ins = a.operator_inputs(*id);
        let out = a.operator_outputs(*id)[0];
        if [ins[0], ins[1], out]
            .iter()
            .any(|v| tensor(a, *v).is_ok_and(|t| t.dtype() != DType::FP16))
        {
            continue;
        }
        if let Some(bytes) = a.serialized_constant(ins[1]) {
            let shape = static_shape(tensor(a, ins[1])?)?;
            if shape.len() == 3 && shape[0] == 1 {
                let (k, n) = (shape[1], shape[2]);
                let mut transposed = Vec::with_capacity(bytes.len());
                for row in 0..n as usize {
                    for col in 0..k as usize {
                        let pos = (col * n as usize + row) * 2;
                        transposed.extend_from_slice(&bytes[pos..pos + 2]);
                    }
                }
                plans[out.get() as usize] = Some(Weights {
                    n,
                    k,
                    group: k,
                    data: Storage::Dense(transposed),
                });
                continue;
            }
        }
        let Some(tr) = a.value(ins[1]).producer() else {
            continue;
        };
        if a.operator(tr).op() != Op::TRANSPOSE {
            continue;
        }
        let OpAttributes::Transpose { perms } = a.operator(tr).source().attributes() else {
            continue;
        };
        if !perms.iter().eq([0, 2, 1]) {
            continue;
        }
        let matrix = a.operator_inputs(tr)[0];
        let shape = static_shape(tensor(a, matrix)?)?;
        if shape.len() != 3 || shape[0] != 1 {
            continue;
        }
        let (n, k) = (shape[1], shape[2]);
        let Some(rs) = producer(a, matrix, Op::RESHAPE) else {
            continue;
        };
        let v = rs[0];
        if let Some(mul) = producer(a, v, Op::MUL) {
            if a.serialized_constant(mul[2])
                .is_none_or(|b| b.iter().any(|x| *x != 0))
            {
                continue;
            }
            for (decoded, scale) in [(mul[0], mul[1]), (mul[1], mul[0])] {
                let Some(cast) = producer(a, decoded, Op::CAST) else {
                    continue;
                };
                let src = cast[0];
                if tensor(a, src)?.dtype() != DType::INT8
                    || tensor(a, decoded)?.dtype() != DType::FP16
                    || tensor(a, scale)?.dtype() != DType::FP16
                {
                    continue;
                }
                let (Some(values), Some(scales)) =
                    (a.serialized_constant(src), a.serialized_constant(scale))
                else {
                    continue;
                };
                let ws = static_shape(tensor(a, src)?)?;
                if ws.len() != 3
                    || ws[0] != n
                    || ws[1].checked_mul(ws[2]) != Some(k)
                    || static_shape(tensor(a, scale)?)? != [n, ws[1], 1]
                {
                    continue;
                }
                // A finite scale of either sign, zero included, gives the same dot
                // products as the multiply it replaces; only the sign of a zero
                // weight can differ, which a sum does not observe. A NaN or an
                // infinity would turn zero weights into NaN, so those stay in the graph.
                if scales.chunks_exact(2).any(|b| {
                    let v = u16::from_le_bytes([b[0], b[1]]);
                    v & 0x7c00 == 0x7c00
                }) {
                    continue;
                }
                plans[out.get() as usize] = Some(Weights {
                    n,
                    k,
                    group: ws[2],
                    data: Storage::Linear {
                        values: values.to_vec(),
                        scales: scales.to_vec(),
                    },
                });
                break;
            }
        } else if let Some(gather) = producer(a, v, Op::GATHER) {
            let (Some(lut), Some(indices)) = (
                a.serialized_constant(gather[0]),
                a.serialized_constant(gather[1]),
            ) else {
                continue;
            };
            if tensor(a, gather[0])?.dtype() != DType::FP16
                || tensor(a, gather[1])?.dtype() != DType::INT32
            {
                continue;
            }
            let ls = static_shape(tensor(a, gather[0])?)?;
            let ix = static_shape(tensor(a, gather[1])?)?;
            if ls.len() != 3
                || ix.len() != 2
                || ls != [ix[0], 16, 1]
                || k % ix[1] != 0
                || n.checked_mul(k / ix[1]) != Some(ix[0])
            {
                continue;
            }
            let codes = indices
                .chunks_exact(4)
                .map(|b| i32::from_le_bytes(b.try_into().unwrap()))
                .collect::<Vec<_>>();
            if codes.iter().any(|v| !(0..16).contains(v)) {
                continue;
            }
            plans[out.get() as usize] = Some(Weights {
                n,
                k,
                group: ix[1],
                data: Storage::Palette {
                    indices: pack(codes.iter().map(|v| *v as u8)),
                    lut: lut.to_vec(),
                },
            });
        }
    }
    Ok(plans)
}

/// Keep shared producers and observable outputs. Only dead ancestors of a replaced edge disappear.
pub(super) fn live_values(a: &TosaAnalysis<'_>, plans: &[Option<Weights>]) -> Vec<bool> {
    let mut live = vec![false; a.values().len()];
    let mut stack = a.block_outputs(a.blocks()[0].id()).to_vec();
    while let Some(v) = stack.pop() {
        if core::mem::replace(&mut live[v.get() as usize], true) {
            continue;
        }
        if let Some(id) = a.value(v).producer() {
            let ins = a.operator_inputs(id);
            if plans[v.get() as usize].is_some() {
                stack.push(ins[0]);
                stack.extend_from_slice(&ins[2..]);
            } else {
                stack.extend_from_slice(ins);
            }
        }
    }
    live
}

fn pack(values: impl Iterator<Item = u8>) -> Vec<u8> {
    let mut data = Vec::new();
    for (i, v) in values.enumerate() {
        if i % 2 == 0 {
            data.push(v & 15);
        } else {
            *data.last_mut().unwrap() |= (v & 15) << 4;
        }
    }
    data
}

impl Weights {
    pub(super) fn emit(&self, b: &mut Builder, prefer_palette: bool) -> String {
        let shape = [self.n, self.k];
        let groups = self.k / self.group;
        match &self.data {
            Storage::Dense(values) => b.constant(F16, &shape, values),
            Storage::Palette { indices, lut } => self.palette(b, indices, lut),
            Storage::Linear { values, scales } => {
                let narrow = values.iter().all(|v| (-8..8).contains(&(*v as i8)));
                let packed = narrow.then(|| pack(values.iter().copied()));
                if prefer_palette && narrow {
                    let mut lut = Vec::with_capacity(scales.len() * 16);
                    for scale in scales.chunks_exact(2) {
                        let scale =
                            half::f16::from_bits(u16::from_le_bytes([scale[0], scale[1]])).to_f32();
                        for code in 0..16u8 {
                            let signed = ((code as i8) << 4) >> 4;
                            // I8*F16 fits exactly in F32; round once to the declared F16 weight.
                            let value = half::f16::from_f32(f32::from(signed) * scale);
                            lut.extend_from_slice(&value.to_bits().to_le_bytes());
                        }
                    }
                    self.palette(b, packed.as_ref().unwrap(), &lut)
                } else {
                    let data = b.constant(
                        if narrow { I4 } else { I8 },
                        &shape,
                        packed.as_deref().unwrap_or(values),
                    );
                    let scale = b.constant(F16, &[self.n, groups], scales);
                    b.op(
                        "constexpr_blockwise_shift_scale",
                        &[("data", &data), ("scale", &scale)],
                        F16,
                        &shape,
                    )
                }
            }
        }
    }
    fn palette(&self, b: &mut Builder, indices: &[u8], lut: &[u8]) -> String {
        let indices = b.constant(U4, &[self.n, self.k], indices);
        let (rows, cols, tables) =
            compact_palettes(self.n as usize, (self.k / self.group) as usize, lut);
        let lut = b.constant(F16, &[rows as i32, cols as i32, 16, 1], &tables);
        b.op(
            "constexpr_lut_to_dense",
            &[("indices", &indices), ("lut", &lut)],
            F16,
            &[self.n, self.k],
        )
    }
}

/// Deduplicate identical adjacent LUTs on each axis without changing index mapping.
/// A channel-only LUT grid enables ANE placement where input-axis blocks may not.
fn compact_palettes(rows: usize, cols: usize, lut: &[u8]) -> (usize, usize, Vec<u8>) {
    fn gcd(mut a: usize, mut b: usize) -> usize {
        while b != 0 {
            (a, b) = (b, a % b);
        }
        a
    }
    let table = |r: usize, c: usize| &lut[(r * cols + c) * 32..(r * cols + c + 1) * 32];
    let (mut rh, mut cw) = (rows, cols);
    for r in 0..rows {
        for c in 0..cols {
            if r > 0 && table(r, c) != table(r - 1, c) {
                rh = gcd(rh, r);
            }
            if c > 0 && table(r, c) != table(r, c - 1) {
                cw = gcd(cw, c);
            }
        }
    }
    let mut data = Vec::with_capacity((rows / rh) * (cols / cw) * 32);
    for r in (0..rows).step_by(rh) {
        for c in (0..cols).step_by(cw) {
            data.extend_from_slice(table(r, c));
        }
    }
    (rows / rh, cols / cw, data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use virtio_accel_tosa_build::quantized;
    #[test]
    fn native_compression_encodings() {
        let graph = quantized::int4_matmul_fp16(1, 8, 32, 16, &[0x98; 128], &[0x3555; 16]).unwrap();
        for (palette, kind) in [
            (false, "constexpr_blockwise_shift_scale"),
            (true, "constexpr_lut_to_dense"),
        ] {
            let model = lower_with_weights(&graph, quantized::TARGET, palette).unwrap();
            assert!(
                model
                    .bytes
                    .windows(kind.len())
                    .any(|b| b == kind.as_bytes())
            );
        }
        let graph =
            quantized::palette4_matmul_fp16(1, 8, 32, 16, &[0x98; 128], &[0x3555; 256]).unwrap();
        let model = parse(&graph).unwrap();
        let a = model.analyze_for(quantized::TARGET).unwrap();
        assert!(plans(&a).unwrap().iter().any(Option::is_some));
        lower_with_weights(&graph, quantized::TARGET, false).unwrap();
    }
    #[test]
    fn export_compressed_weights_for_native_inspection() {
        let Some(dir) = std::env::var_os("VIRTIO_ACCEL_COREML_TEST_MODEL_DIR") else {
            return;
        };
        let dir = std::path::PathBuf::from(dir);
        std::fs::create_dir_all(&dir).unwrap();
        let (n, k, group) = (1024, 2048, 128);
        let q = (0..n * k)
            .map(|i| {
                let hash = (i as u32).wrapping_mul(1664525).wrapping_add(1013904223);
                ((hash >> 24) & 15) as i8 - 8
            })
            .collect::<Vec<_>>();
        let packed = pack(q.iter().map(|v| *v as u8));
        let scales = (0..n * k / group)
            .map(|i| if i % 3 == 0 { 0x3000 } else { 0x2c00 })
            .collect::<Vec<_>>();
        let channel_palette = (0..n)
            .flat_map(|i| {
                let scale = if (i / 16) % 3 == 0 { 0.125 } else { 0.0625 };
                (0..16u8).map(move |c| {
                    half::f16::from_f32(f32::from(((c as i8) << 4) >> 4) * scale).to_bits()
                })
            })
            .collect::<Vec<_>>();
        for m in [1, 16, 128] {
            let w4 = quantized::int4_matmul_fp16(m, n, k, group, &packed, &scales).unwrap();
            let channel =
                quantized::palette4_matmul_fp16(m, n, k, k, &packed, &channel_palette).unwrap();
            for (name, bytes, palette) in [
                ("int4", &w4, false),
                ("palette4-block", &w4, true),
                ("palette4-channel", &channel, false),
            ] {
                let model = lower_with_weights(bytes, quantized::TARGET, palette).unwrap();
                std::fs::write(dir.join(format!("{name}-m{m}.mlmodel")), model.bytes).unwrap();
            }
            for channel in [false, true] {
                use virtio_accel_tosa_build::{Graph, Operator, OperatorKind, Tensor};
                let data = (0..k * n)
                    .flat_map(|i| {
                        let pos = (i % n) * k + i / n;
                        let scale = if channel {
                            if ((pos / k) / 16) % 3 == 0 {
                                0.125
                            } else {
                                0.0625
                            }
                        } else {
                            half::f16::from_bits(scales[pos / group]).to_f32()
                        };
                        half::f16::from_f32(f32::from(q[pos]) * scale)
                            .to_bits()
                            .to_le_bytes()
                    })
                    .collect::<Vec<_>>();
                let ashape = [1, m as i32, k as i32];
                let bshape = [1, k as i32, n as i32];
                let oshape = [1, m as i32, n as i32];
                let f = DType::FP16;
                let tensors = [
                    Tensor::new("a", &ashape, f),
                    Tensor::constant("b", &bshape, f, &data),
                    Tensor::constant("zero", &[1], f, &[0, 0]),
                    Tensor::new("out", &oshape, f),
                ];
                let ops = [
                    Operator::new(OperatorKind::Const, &[], &["b"]),
                    Operator::new(OperatorKind::Const, &[], &["zero"]),
                    Operator::new(OperatorKind::MatMul, &["a", "b", "zero", "zero"], &["out"]),
                ];
                let bytes = Graph::new("dense", &tensors, &ops, &["a"], &["out"])
                    .build(quantized::TARGET)
                    .unwrap();
                let model = lower_with_weights(&bytes, quantized::TARGET, false).unwrap();
                let name = if channel { "fp16-channel" } else { "fp16" };
                std::fs::write(dir.join(format!("{name}-m{m}.mlmodel")), model.bytes).unwrap();
            }
        }
    }
    #[test]
    fn palette_compaction_preserves_each_block_and_signed_zero() {
        for (rows, cols, rh, cw) in [(8, 6, 2, 3), (8, 4, 8, 4), (3, 5, 1, 1)] {
            let tables = (0..rows)
                .flat_map(|r| (0..cols).flat_map(move |c| [((r / rh) * 10 + c / cw) as u8; 32]))
                .collect::<Vec<_>>();
            let (nr, nc, packed) = compact_palettes(rows, cols, &tables);
            assert_eq!((nr, nc), (rows / rh, cols / cw));
            for r in 0..rows {
                for c in 0..cols {
                    let dst = ((r / rh) * nc + c / cw) * 32;
                    assert_eq!(
                        &tables[(r * cols + c) * 32..(r * cols + c + 1) * 32],
                        &packed[dst..dst + 32]
                    );
                }
            }
        }
        let mut tables = vec![0; 64];
        tables[33] = 0x80;
        assert_eq!(compact_palettes(2, 1, &tables), (2, 1, tables));
        assert_eq!(pack([0, 15, 8].into_iter()), vec![0xf0, 0x08]);
    }

    fn pattern(runtime_weights: bool, runtime_scales: bool, shared_output: bool) -> Vec<u8> {
        use virtio_accel_tosa_build::{Graph, Operator, OperatorKind as K, Shape, Tensor};
        let f = DType::FP16;
        let mut tensors = vec![
            Tensor::new("a", &[1, 1, 4], f),
            Tensor::constant("w", &[2, 1, 4], DType::INT8, &[0, 1, 2, 3, 4, 5, 6, 7]),
            Tensor::new("cast", &[2, 1, 4], f),
            Tensor::constant("scale", &[2, 1, 1], f, &[0, 0x38, 0, 0x3c]),
            Tensor::constant("shift", &[1], DType::INT8, &[0]),
            Tensor::new("scaled", &[2, 1, 4], f),
            Tensor::new("matrix", &[1, 2, 4], f),
            Tensor::new("tr", &[1, 4, 2], f),
            Tensor::constant("zero", &[1], f, &[0, 0]),
            Tensor::new("out", &[1, 1, 2], f),
        ];
        let mut inputs = vec!["a"];
        let mut ops = vec![];
        if runtime_weights {
            tensors[1] = Tensor::new("w", &[2, 1, 4], DType::INT8);
            inputs.push("w");
        } else {
            ops.push(Operator::new(K::Const, &[], &["w"]));
        }
        if runtime_scales {
            tensors[3] = Tensor::new("scale", &[2, 1, 1], f);
            inputs.push("scale");
        } else {
            ops.push(Operator::new(K::Const, &[], &["scale"]));
        }
        ops.extend([
            Operator::new(K::Cast, &["w"], &["cast"]),
            Operator::new(K::Const, &[], &["shift"]),
            Operator::new(K::Mul, &["cast", "scale", "shift"], &["scaled"]),
            Operator::new(K::ConstShape, &[], &["shape"]),
            Operator::new(K::Reshape, &["scaled", "shape"], &["matrix"]),
            Operator::new(
                K::Transpose {
                    perms: [0, 2, 1, 0, 0, 0],
                    rank: 3,
                },
                &["matrix"],
                &["tr"],
            ),
            Operator::new(K::Const, &[], &["zero"]),
            Operator::new(K::MatMul, &["a", "tr", "zero", "zero"], &["out"]),
        ]);
        let outputs = if shared_output {
            vec!["out", "scaled"]
        } else {
            vec!["out"]
        };
        Graph::new("pattern", &tensors, &ops, &inputs, &outputs)
            .with_shapes(&[Shape::new("shape", &[1, 2, 4])])
            .build(quantized::TARGET)
            .unwrap()
    }
    #[test]
    fn rewrites_only_constants_and_preserves_shared_outputs() {
        for (rw, rs, shared) in [
            (false, false, false),
            (true, false, false),
            (false, true, false),
            (false, false, true),
        ] {
            let bytes = pattern(rw, rs, shared);
            let model = parse(&bytes).unwrap();
            let a = model.analyze_for(quantized::TARGET).unwrap();
            let plans = plans(&a).unwrap();
            assert_eq!(plans.iter().any(Option::is_some), !rw && !rs);
            let live = live_values(&a, &plans);
            let scaled = a
                .values()
                .iter()
                .find(|v| v.name() == "scaled")
                .unwrap()
                .id();
            assert_eq!(live[scaled.get() as usize], rw || rs || shared);
            lower_with_weights(&bytes, quantized::TARGET, false).unwrap();
        }
    }
}
