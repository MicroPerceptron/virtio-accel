//! FP8 storage lowered to typed ML Programs. FP8 buffers remain one byte per element;
//! the native INT8 boundary transports their bits, never their signed numeric value.
//! Decode/encode are graph operations, not host-side tensor conversions.
#![cfg_attr(not(target_os = "macos"), allow(dead_code))]
use crate::lower::{
    LoweredExecution, LoweredFeature, LoweredFeatureRole, LoweredModel, LoweringError,
    encode_array_feature, static_shape, tensor,
};
use crate::mlprogram::*;
mod weights;
use virtio_accel_tosa::{
    AnalyzedValueKind, CapabilityDescriptor, DType, DTypeCapability, DTypeConstraints,
    ExtensionSet, GraphCapabilities, Level, Op, OpAttributes, OperatorCapability,
    OperatorConstraints, ProfileSet, RuntimeConditionSupport, Target, ValueRoles, Version, parse,
};

pub const COREML_TOSA_FP8_TARGET: Target = Target::new(
    Version::TOSA_1_0,
    ProfileSet::FLOATING_POINT,
    Level::Level8K,
    ExtensionSet::FP8E4M3.union(ExtensionSet::FP8E5M2),
);
pub const COREML_TOSA_FP8_CAPABILITY: CapabilityDescriptor = CapabilityDescriptor {
    target: COREML_TOSA_FP8_TARGET,
    dtypes: &[
        DTypeCapability::new(DType::FP8E4M3, ValueRoles::ALL),
        DTypeCapability::new(DType::FP8E5M2, ValueRoles::ALL),
        DTypeCapability::new(DType::FP16, ValueRoles::ALL),
        DTypeCapability::new(DType::FP32, ValueRoles::ALL),
        DTypeCapability::constrained(
            DType::INT8,
            ValueRoles::CONSTANT,
            DTypeConstraints::PARAMETER_ONLY,
        ),
    ],
    operators: &[
        OperatorCapability::new(Op::CONST),
        OperatorCapability::new(Op::CONST_SHAPE),
        OperatorCapability::new(Op::IDENTITY),
        OperatorCapability::new(Op::CAST),
        OperatorCapability::new(Op::ADD),
        OperatorCapability::constrained(Op::MUL, OperatorConstraints::ZERO_SHIFT),
        OperatorCapability::constrained(Op::MATMUL, OperatorConstraints::ZERO_ZERO_POINTS),
        OperatorCapability::new(Op::RESHAPE),
        OperatorCapability::new(Op::TRANSPOSE),
        OperatorCapability::new(Op::CONCAT),
        OperatorCapability::constrained(Op::SLICE, OperatorConstraints::CONSTANT_PARAMETERS),
    ],
    graph: GraphCapabilities {
        max_regions: 1,
        max_blocks: 1,
        dynamic_shapes: false,
        runtime_conditions: RuntimeConditionSupport::None,
    },
};
/// Mixed integer/float target for explicitly packed-storage graphs (including NVFP4).
pub const COREML_TOSA_QUANTIZED_TARGET: Target = Target::new(
    Version::TOSA_1_0,
    ProfileSet::FLOATING_POINT.union(ProfileSet::INTEGER),
    Level::Level8K,
    ExtensionSet::NONE,
);
pub const COREML_TOSA_QUANTIZED_CAPABILITY: CapabilityDescriptor = CapabilityDescriptor {
    target: COREML_TOSA_QUANTIZED_TARGET,
    dtypes: &[
        DTypeCapability::new(DType::INT8, ValueRoles::ALL),
        DTypeCapability::new(DType::INT32, ValueRoles::ALL),
        DTypeCapability::new(DType::FP32, ValueRoles::ALL),
        DTypeCapability::new(DType::FP16, ValueRoles::ALL),
    ],
    operators: &[
        OperatorCapability::new(Op::CONST),
        OperatorCapability::new(Op::CONST_SHAPE),
        OperatorCapability::new(Op::CAST),
        OperatorCapability::new(Op::ADD),
        OperatorCapability::constrained(Op::MUL, OperatorConstraints::ZERO_SHIFT),
        OperatorCapability::new(Op::GATHER),
        OperatorCapability::new(Op::CONCAT),
        OperatorCapability::constrained(Op::SLICE, OperatorConstraints::CONSTANT_PARAMETERS),
        OperatorCapability::new(Op::RESHAPE),
        OperatorCapability::new(Op::TRANSPOSE),
        OperatorCapability::constrained(Op::MATMUL, OperatorConstraints::ZERO_ZERO_POINTS),
        // FP16/FP32 only: what a normalised, gated or attention block needs around
        // its compressed projections, so that it stays one program.
        OperatorCapability::new(Op::SUB),
        OperatorCapability::new(Op::EXP),
        OperatorCapability::new(Op::RECIPROCAL),
        OperatorCapability::new(Op::RSQRT),
        OperatorCapability::new(Op::SIGMOID),
        OperatorCapability::new(Op::REDUCE_MAX),
        OperatorCapability::new(Op::REDUCE_SUM),
    ],
    graph: GraphCapabilities {
        max_regions: 1,
        max_blocks: 1,
        dynamic_shapes: false,
        runtime_conditions: RuntimeConditionSupport::ProvenConstantOrByteLookup,
    },
};
const F16: u64 = 10;
const F32: u64 = 11;
const I8: u64 = 21;
const I32: u64 = 23;
const BOOL: u64 = 1;
fn is_fp8(d: DType) -> bool {
    matches!(d, DType::FP8E4M3 | DType::FP8E5M2)
}
fn mil(d: DType) -> Result<u64, LoweringError> {
    Ok(match d {
        DType::INT8 | DType::FP8E4M3 | DType::FP8E5M2 => I8,
        DType::FP16 => F16,
        DType::FP32 => F32,
        DType::INT32 => I32,
        _ => return Err(LoweringError::UnsupportedType(d)),
    })
}
fn boundary(d: DType) -> DType {
    if is_fp8(d) { DType::INT8 } else { d }
}

#[derive(Default)]
struct Builder {
    ops: Vec<Vec<u8>>,
    next: usize,
}
impl Builder {
    fn name(&mut self) -> String {
        let n = format!("q{}", self.next);
        self.next += 1;
        n
    }
    fn op(&mut self, kind: &str, inputs: &[(&str, &str)], dtype: u64, shape: &[i32]) -> String {
        let n = self.name();
        self.ops
            .push(unary_operation(kind, inputs, &n, dtype, shape));
        n
    }
    fn string(&mut self, s: &str) -> String {
        let n = self.name();
        self.ops.push(const_string(&n, s));
        n
    }
    fn int(&mut self, x: i32) -> String {
        let n = self.name();
        self.ops.push(const_int32(&n, x));
        n
    }
    fn boolean(&mut self, x: bool) -> String {
        let n = self.name();
        self.ops.push(const_bool(&n, x));
        n
    }
    fn constant(&mut self, dtype: u64, shape: &[i32], bytes: &[u8]) -> String {
        let n = self.name();
        let mut repeated = Vec::new();
        let tensor_field = match dtype {
            F32 => {
                field_bytes(&mut repeated, 1, bytes);
                1
            }
            I32 => {
                let mut packed = Vec::new();
                for b in bytes.chunks_exact(4) {
                    varint(
                        &mut packed,
                        i32::from_le_bytes(b.try_into().unwrap()) as i64 as u64,
                    );
                }
                field_bytes(&mut repeated, 1, &packed);
                2
            }
            _ => {
                field_bytes(&mut repeated, 1, bytes);
                7
            }
        };
        let mut t = Vec::new();
        field_message(&mut t, tensor_field, &repeated);
        let mut imm = Vec::new();
        field_message(&mut imm, 1, &t);
        let mut value = Vec::new();
        field_message(&mut value, 2, &value_type(dtype, shape));
        field_message(&mut value, 3, &imm);
        let mut op = Vec::new();
        field_string(&mut op, 1, "const");
        field_message(&mut op, 3, &named_value_type(&n, dtype, shape));
        field_message(&mut op, 5, &attribute_entry("val", &value));
        field_message(&mut op, 5, &attribute_entry("name", &string_value(&n)));
        self.ops.push(op);
        n
    }
    fn float(&mut self, x: f32) -> String {
        self.constant(F32, &[], &x.to_le_bytes())
    }
    fn cast(&mut self, x: &str, dtype: u64, shape: &[i32]) -> String {
        let d = self.string(match dtype {
            I8 => "int8",
            I32 => "int32",
            F16 => "fp16",
            _ => "fp32",
        });
        self.op("cast", &[("x", x), ("dtype", &d)], dtype, shape)
    }
    fn binary(&mut self, k: &str, x: &str, y: &str, d: u64, s: &[i32]) -> String {
        self.op(k, &[("x", x), ("y", y)], d, s)
    }
    fn select(&mut self, c: &str, a: &str, b: &str, d: u64, s: &[i32]) -> String {
        self.op("select", &[("cond", c), ("a", a), ("b", b)], d, s)
    }
    fn lookup(&mut self, values: &[f32], index: &str, shape: &[i32]) -> String {
        let data = values
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect::<Vec<_>>();
        let table = self.constant(F32, &[values.len() as i32], &data);
        let axis = self.int(0);
        let validate = self.boolean(true);
        let batch = self.int(0);
        self.op(
            "gather",
            &[
                ("x", &table),
                ("indices", index),
                ("axis", &axis),
                ("batch_dims", &batch),
                ("validate_indices", &validate),
            ],
            F32,
            shape,
        )
    }
    fn decode(&mut self, x: &str, dtype: DType, s: &[i32]) -> String {
        let x = self.cast(x, I32, s);
        let offset = self.int(128);
        let idx = self.binary("add", &x, &offset, I32, s);
        // Signed-byte order lets an exact +128 produce a bounded [0,255] index.
        let values = (0..256)
            .map(|i| decode((i as u8).wrapping_add(128), dtype))
            .collect::<Vec<_>>();
        self.lookup(&values, &idx, s)
    }
    fn encode(&mut self, x: &str, dtype: DType, s: &[i32]) -> String {
        let x = self.cast(x, F32, s);
        let mag = self.op("abs", &[("x", &x)], F32, s);
        let zero = self.int(0);
        let mut code = zero.clone();
        let max_code = if dtype == DType::FP8E4M3 { 127 } else { 124 };
        let mut mid = vec![0.0f32; 128];
        let mut even = vec![0.0f32; 128];
        for i in 1..128 {
            mid[i] = if i >= max_code {
                if dtype == DType::FP8E4M3 {
                    464.0
                } else if i == 124 {
                    61440.0
                } else {
                    f32::INFINITY
                }
            } else {
                (decode((i - 1) as u8, dtype) + decode(i as u8, dtype)) * 0.5
            };
            even[i] = if i % 2 == 0 { 1.0 } else { 0.0 };
        }
        let fzero = self.float(0.0);
        // Seven bounded comparisons implement round-to-nearest, ties-to-even directly from
        // binary32; there is no FP16 intermediate and therefore no double rounding.
        for step in [64, 32, 16, 8, 4, 2, 1] {
            let trial_shape = if step == 64 { &[][..] } else { s };
            let step = self.int(step);
            let trial = self.binary("add", &code, &step, I32, trial_shape);
            let threshold = self.lookup(&mid, &trial, trial_shape);
            let gt = self.binary("greater", &mag, &threshold, BOOL, s);
            let eq = self.binary("equal", &mag, &threshold, BOOL, s);
            let parity = self.lookup(&even, &trial, trial_shape);
            let is_even = self.binary("greater", &parity, &fzero, BOOL, trial_shape);
            let tie = self.binary("logical_and", &eq, &is_even, BOOL, s);
            let up = self.binary("logical_or", &gt, &tie, BOOL, s);
            code = self.select(&up, &trial, &code, I32, s);
        }
        let inf = self.float(f32::INFINITY);
        let is_inf = self.binary("equal", &mag, &inf, BOOL, s);
        let overflow = self.int(max_code as i32);
        code = self.select(&is_inf, &overflow, &code, I32, s);
        let finite_or_inf = self.binary("equal", &x, &x, BOOL, s);
        let nan = self.int(if dtype == DType::FP8E4M3 { 127 } else { 126 });
        code = self.select(&finite_or_inf, &code, &nan, I32, s);
        let negative = self.binary("less", &x, &fzero, BOOL, s);
        let one = self.float(1.0);
        let inv = self.binary("real_div", &one, &x, F32, s);
        let neg_zero = self.binary("less", &inv, &fzero, BOOL, s);
        let sign = self.binary("logical_or", &negative, &neg_zero, BOOL, s);
        let signbit = self.int(128);
        let signed = self.binary("sub", &code, &signbit, I32, s);
        let bits = self.select(&sign, &signed, &code, I32, s);
        self.cast(&bits, I8, s)
    }
}
fn decode(x: u8, d: DType) -> f32 {
    if d == DType::FP8E4M3 {
        virtio_accel_tosa::fp8e4m3_to_f32(x)
    } else {
        virtio_accel_tosa::fp8e5m2_to_f32(x)
    }
}

/// The float operators admit one float type for every operand and the result.
fn float_only(
    a: &virtio_accel_tosa::TosaAnalysis<'_>,
    inputs: &[virtio_accel_tosa::ValueId],
    result: DType,
) -> Result<(), LoweringError> {
    if !matches!(result, DType::FP16 | DType::FP32) {
        return Err(LoweringError::UnsupportedType(result));
    }
    for value in inputs {
        let input = tensor(a, *value)?.dtype();
        if input != result {
            return Err(LoweringError::UnsupportedType(input));
        }
    }
    Ok(())
}

pub(crate) fn lower(bytes: &[u8], target: Target) -> Result<LoweredModel, LoweringError> {
    lower_with_weights(bytes, target, false)
}

pub(crate) fn lower_with_weights(
    bytes: &[u8],
    target: Target,
    prefer_palette: bool,
) -> Result<LoweredModel, LoweringError> {
    let model = parse(bytes).map_err(LoweringError::Parse)?;
    let a = model.analyze_for(target).map_err(LoweringError::Analysis)?;
    let capability = if target == COREML_TOSA_FP8_TARGET {
        &COREML_TOSA_FP8_CAPABILITY
    } else if target == COREML_TOSA_QUANTIZED_TARGET {
        &COREML_TOSA_QUANTIZED_CAPABILITY
    } else {
        return Err(LoweringError::UnsupportedGraph);
    };
    if a.blocks().len() != 1
        || a.regions().len() != 1
        || a.conditions().iter().any(|c| {
            target != COREML_TOSA_QUANTIZED_TARGET
                || !(a.proves_byte_lookup_bounds(c) || a.proves_constant_index_bounds(c))
        })
    {
        return Err(LoweringError::UnsupportedGraph);
    }
    let block = a.blocks()[0].id();
    let ins = a.block_inputs(block);
    let outs = a.block_outputs(block);
    if ins.is_empty() || outs.is_empty() || ins.iter().any(|v| outs.contains(v)) {
        return Err(LoweringError::UnsupportedGraph);
    }
    for v in a.values() {
        if let AnalyzedValueKind::Tensor(t) = v.kind() {
            mil(t.dtype())?;
            if (target == COREML_TOSA_FP8_TARGET
                && t.dtype() == DType::INT8
                && !(a.serialized_constant(v.id()).is_some()
                    && crate::lower::constant_is_parameter_only(&a, v.id())))
                || !capability.dtypes.iter().any(|d| d.dtype == t.dtype())
            {
                return Err(LoweringError::UnsupportedType(t.dtype()));
            }
        }
    }
    let plans = weights::plans(&a)?;
    let live = weights::live_values(&a, &plans);
    let mut b = Builder::default();
    let mut names = vec![String::new(); a.values().len()];
    let mut description = Vec::new();
    let mut features = Vec::new();
    let mut function = Vec::new();
    for (i, v) in ins.iter().enumerate() {
        let t = tensor(&a, *v)?;
        let s = static_shape(t)?;
        let n = format!("input_{i}");
        encode_array_feature(&mut description, 1, &n, boundary(t.dtype()), &s)?;
        field_message(&mut function, 1, &named_value_type(&n, mil(t.dtype())?, &s));
        names[v.get() as usize] = n.clone();
        features.push(LoweredFeature {
            slot: i as u32,
            role: LoweredFeatureRole::Input,
            name: n,
        });
    }
    for id in a.execution_order(block) {
        let op = a.operator(*id);
        let inputs = a.operator_inputs(*id);
        let outputs = a.operator_outputs(*id);
        if !capability.supports_operator(op.op()) {
            return Err(LoweringError::UnsupportedOperator(op.op()));
        }
        if op.op() == Op::CONST_SHAPE {
            continue;
        }
        let out = outputs[0];
        if !live[out.get() as usize] {
            continue;
        }
        let t = tensor(&a, out)?;
        let s = static_shape(t)?;
        let d = mil(t.dtype())?;
        let args = inputs
            .iter()
            .map(|v| names[v.get() as usize].clone())
            .collect::<Vec<_>>();
        let n = match op.op() {
            Op::CONST => b.constant(
                d,
                &s,
                a.serialized_constant(out)
                    .ok_or(LoweringError::InvalidConstant)?,
            ),
            Op::CAST => {
                let input_dtype = tensor(&a, inputs[0])?.dtype();
                let float = |d| matches!(d, DType::FP16 | DType::FP32);
                if target == COREML_TOSA_QUANTIZED_TARGET
                    && !((input_dtype == DType::INT8
                        && (t.dtype() == DType::INT32 || float(t.dtype())))
                        || (float(input_dtype) && float(t.dtype())))
                {
                    return Err(LoweringError::UnsupportedType(input_dtype));
                }
                if input_dtype == t.dtype() {
                    args[0].clone()
                } else {
                    let x = if is_fp8(input_dtype) {
                        b.decode(&args[0], input_dtype, &s)
                    } else {
                        args[0].clone()
                    };
                    if is_fp8(t.dtype()) {
                        b.encode(&x, t.dtype(), &s)
                    } else {
                        b.cast(&x, d, &s)
                    }
                }
            }
            Op::IDENTITY => args[0].clone(),
            Op::ADD | Op::MUL => {
                if !matches!(t.dtype(), DType::FP16 | DType::FP32)
                    && !(op.op() == Op::ADD && t.dtype() == DType::INT32)
                {
                    return Err(LoweringError::UnsupportedType(t.dtype()));
                }
                if op.op() == Op::MUL
                    && a.serialized_constant(inputs[2])
                        .is_none_or(|s| s.iter().any(|x| *x != 0))
                {
                    return Err(LoweringError::UnsupportedGraph);
                }
                b.binary(
                    if op.op() == Op::ADD { "add" } else { "mul" },
                    &args[0],
                    &args[1],
                    d,
                    &s,
                )
            }
            Op::SUB | Op::EXP | Op::RECIPROCAL | Op::RSQRT | Op::SIGMOID => {
                float_only(&a, inputs, t.dtype())?;
                match op.op() {
                    Op::SUB => b.binary("sub", &args[0], &args[1], d, &s),
                    Op::EXP => b.op("exp", &[("x", &args[0])], d, &s),
                    Op::SIGMOID => b.op("sigmoid", &[("x", &args[0])], d, &s),
                    // MIL's own inverse and rsqrt add an epsilon; these do not.
                    kind => {
                        let divisor = if kind == Op::RSQRT {
                            b.op("sqrt", &[("x", &args[0])], d, &s)
                        } else {
                            args[0].clone()
                        };
                        let one = if t.dtype() == DType::FP16 {
                            b.constant(F16, &[], &0x3c00u16.to_le_bytes())
                        } else {
                            b.float(1.0)
                        };
                        b.binary("real_div", &one, &divisor, d, &s)
                    }
                }
            }
            Op::REDUCE_MAX | Op::REDUCE_SUM => {
                float_only(&a, inputs, t.dtype())?;
                let axis = match op.source().attributes() {
                    OpAttributes::ReduceMax { axis, nan_mode } => {
                        crate::lower::require_propagating_nan(nan_mode)?;
                        axis
                    }
                    OpAttributes::ReduceSum { axis } => axis,
                    _ => return Err(LoweringError::UnsupportedGraph),
                };
                let axes = b.constant(I32, &[1], &axis.to_le_bytes());
                let keep = b.boolean(true);
                b.op(
                    if op.op() == Op::REDUCE_MAX {
                        "reduce_max"
                    } else {
                        "reduce_sum"
                    },
                    &[("x", &args[0]), ("axes", &axes), ("keep_dims", &keep)],
                    d,
                    &s,
                )
            }
            Op::GATHER => {
                if !matches!(t.dtype(), DType::FP32 | DType::FP16) {
                    return Err(LoweringError::UnsupportedType(t.dtype()));
                }
                let axis = b.int(1);
                let batch = b.int(1);
                let validate = b.boolean(true);
                b.op(
                    "gather",
                    &[
                        ("x", &args[0]),
                        ("indices", &args[1]),
                        ("axis", &axis),
                        ("batch_dims", &batch),
                        ("validate_indices", &validate),
                    ],
                    d,
                    &s,
                )
            }
            Op::MATMUL => {
                if !matches!(t.dtype(), DType::FP16 | DType::FP32) {
                    return Err(LoweringError::UnsupportedType(t.dtype()));
                }
                for v in &inputs[2..] {
                    let bytes = a
                        .serialized_constant(*v)
                        .ok_or(LoweringError::UnsupportedGraph)?;
                    if bytes.iter().any(|x| *x != 0) {
                        return Err(LoweringError::UnsupportedGraph);
                    }
                }
                if let Some(plan) = &plans[out.get() as usize] {
                    let w = plan.emit(&mut b, prefer_palette);
                    let result = b.op("linear", &[("x", &args[0]), ("weight", &w)], F16, &s);
                    names[out.get() as usize] = result;
                    continue;
                }
                let mut xy = Vec::new();
                for v in &inputs[..2] {
                    let td = tensor(&a, *v)?;
                    let sh = static_shape(td)?;
                    let name = &names[v.get() as usize];
                    let x = if is_fp8(td.dtype()) {
                        b.decode(name, td.dtype(), &sh)
                    } else {
                        b.cast(name, F32, &sh)
                    };
                    xy.push(x)
                }
                let no = b.boolean(false);
                let result = b.op(
                    "matmul",
                    &[
                        ("x", &xy[0]),
                        ("y", &xy[1]),
                        ("transpose_x", &no),
                        ("transpose_y", &no),
                    ],
                    F32,
                    &s,
                );
                b.cast(&result, d, &s)
            }
            Op::SLICE => {
                let input = tensor(&a, inputs[0])?;
                let input_s = static_shape(input)?;
                let x = if d == I8 {
                    b.cast(&args[0], I32, &input_s)
                } else {
                    args[0].clone()
                };
                let start = a
                    .serialized_constant(inputs[1])
                    .ok_or(LoweringError::UnsupportedGraph)?;
                a.serialized_constant(inputs[2])
                    .ok_or(LoweringError::UnsupportedGraph)?;
                let begin = start
                    .chunks_exact(8)
                    .map(|c| {
                        i32::try_from(i64::from_le_bytes(c.try_into().unwrap()))
                            .map_err(|_| LoweringError::UnsupportedGraph)
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let begin = b.constant(
                    I32,
                    &[begin.len() as i32],
                    &begin
                        .iter()
                        .flat_map(|i| i.to_le_bytes())
                        .collect::<Vec<_>>(),
                );
                let size = b.constant(
                    I32,
                    &[s.len() as i32],
                    &s.iter().flat_map(|i| i.to_le_bytes()).collect::<Vec<_>>(),
                );
                let result = b.op(
                    "slice_by_size",
                    &[("x", &x), ("begin", &begin), ("size", &size)],
                    if d == I8 { I32 } else { d },
                    &s,
                );
                if d == I8 {
                    b.cast(&result, I8, &s)
                } else {
                    result
                }
            }
            Op::RESHAPE | Op::TRANSPOSE => {
                let input = tensor(&a, inputs[0])?;
                let input_s = static_shape(input)?;
                let x = if d == I8 {
                    b.cast(&args[0], I32, &input_s)
                } else {
                    args[0].clone()
                };
                let (kind, key, data) = if op.op() == Op::RESHAPE {
                    ("reshape", "shape", s.clone())
                } else {
                    let OpAttributes::Transpose { perms } = op.source().attributes() else {
                        return Err(LoweringError::UnsupportedGraph);
                    };
                    ("transpose", "perm", perms.iter().collect::<Vec<_>>())
                };
                let param = b.constant(
                    I32,
                    &[data.len() as i32],
                    &data
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect::<Vec<_>>(),
                );
                let y = b.op(
                    kind,
                    &[("x", &x), (key, &param)],
                    if d == I8 { I32 } else { d },
                    &s,
                );
                if d == I8 { b.cast(&y, I8, &s) } else { y }
            }
            Op::CONCAT => {
                let OpAttributes::Concat { axis } = op.source().attributes() else {
                    return Err(LoweringError::UnsupportedGraph);
                };
                let axis = b.int(axis);
                let inter = b.boolean(false);
                let n = b.name();
                let mut op = unary_operation(
                    "concat",
                    &[("axis", &axis), ("interleave", &inter)],
                    &n,
                    if d == I8 { I32 } else { d },
                    &s,
                );
                let mut argument = Vec::new();
                for v in inputs {
                    let x = &names[v.get() as usize];
                    let x = if d == I8 {
                        b.cast(x, I32, &static_shape(tensor(&a, *v)?)?)
                    } else {
                        x.clone()
                    };
                    let mut binding = Vec::new();
                    field_string(&mut binding, 1, &x);
                    field_message(&mut argument, 1, &binding);
                }
                field_message(&mut op, 2, &map_entry("values", &argument));
                b.ops.push(op);
                if d == I8 { b.cast(&n, I8, &s) } else { n }
            }
            _ => return Err(LoweringError::UnsupportedOperator(op.op())),
        };
        names[out.get() as usize] = n;
    }
    let mut block_body = Vec::new();
    for (i, v) in outs.iter().enumerate() {
        let t = tensor(&a, *v)?;
        let s = static_shape(t)?;
        let n = format!("output_{i}");
        let input = &names[v.get() as usize];
        // Identity over int8 is not supported by MIL; the exact cast pair preserves all bits.
        let x = if mil(t.dtype())? == I8 {
            b.cast(input, I32, &s)
        } else {
            input.clone()
        };
        if mil(t.dtype())? == I8 {
            let dtype = b.string("int8");
            b.ops.push(unary_operation(
                "cast",
                &[("x", &x), ("dtype", &dtype)],
                &n,
                I8,
                &s,
            ));
        } else {
            b.ops.push(unary_operation(
                "identity",
                &[("x", &x)],
                &n,
                mil(t.dtype())?,
                &s,
            ));
        }
        encode_array_feature(&mut description, 10, &n, boundary(t.dtype()), &s)?;
        field_string(&mut block_body, 2, &n);
        features.push(LoweredFeature {
            slot: (ins.len() + i) as u32,
            role: LoweredFeatureRole::Output,
            name: n,
        });
    }
    for op in b.ops {
        field_message(&mut block_body, 3, &op)
    }
    field_string(&mut function, 2, "CoreML9");
    field_message(&mut function, 3, &map_entry("CoreML9", &block_body));
    let mut program = Vec::new();
    field_varint(&mut program, 1, 1);
    field_message(&mut program, 2, &map_entry("main", &function));
    let mut encoded = Vec::new();
    field_varint(&mut encoded, 1, 10);
    field_message(&mut encoded, 2, &description);
    field_message(&mut encoded, 502, &program);
    Ok(LoweredModel {
        bytes: encoded,
        features,
        execution: LoweredExecution::CoreMl,
        requires_fp32: false,
    })
}
