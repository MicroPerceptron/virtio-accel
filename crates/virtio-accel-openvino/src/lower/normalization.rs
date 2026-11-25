//! Recognize the centered, population-variance TOSA normalization expression.
//! Keep source operations intact for other users; unreachable lowered nodes are
//! removed by the native compiler. No tensor names or model-specific sizes matter.
use super::*;

fn inputs_of<'a>(a: &'a TosaAnalysis<'_>, value: ValueId, op: Op) -> Option<&'a [ValueId]> {
    let p = a.value(value).producer()?;
    (a.operator(p).op() == op).then(|| a.operator_inputs(p))
}
fn scalar(a: &TosaAnalysis<'_>, v: ValueId) -> Option<f32> {
    if tensor(a, v).ok()?.dtype() != DType::FP32 {
        return None;
    }
    let b = a.serialized_constant(v)?;
    (b.len() == 4).then(|| f32::from_le_bytes(b.try_into().unwrap()))
}
fn mean(a: &TosaAnalysis<'_>, value: ValueId, source: ValueId, axis: i32, scale: f32) -> bool {
    let Some(mul) = inputs_of(a, value, Op::MUL) else {
        return false;
    };
    if a.serialized_constant(mul[2]) != Some(&[0]) {
        return false;
    }
    for (reduced, factor) in [(mul[0], mul[1]), (mul[1], mul[0])] {
        if scalar(a, factor) != Some(scale) {
            continue;
        }
        let Some(p) = a.value(reduced).producer() else {
            continue;
        };
        if a.operator(p).op() == Op::REDUCE_SUM
            && a.operator_inputs(p) == [source]
            && matches!(a.operator(p).source().attributes(), OpAttributes::ReduceSum { axis: x } if x == axis)
        {
            return true;
        }
    }
    false
}

pub(super) fn emit(
    builder: &mut IrBuilder,
    a: &TosaAnalysis<'_>,
    inputs: &[ValueId],
    output: ValueId,
    stem: &str,
) -> Result<bool, LoweringError> {
    if tensor(a, output)?.dtype() != DType::FP32 {
        return Ok(false);
    }
    for (centered, inverse) in [(inputs[0], inputs[1]), (inputs[1], inputs[0])] {
        let (Some(sub), Some(rsqrt)) = (
            inputs_of(a, centered, Op::SUB),
            inputs_of(a, inverse, Op::RSQRT),
        ) else {
            continue;
        };
        let x = sub[0];
        if tensor(a, x)?.dtype() != DType::FP32 {
            continue;
        }
        let shape = static_dims(tensor(a, x)?)?;
        if shape.is_empty() || static_dims(tensor(a, output)?)? != shape {
            continue;
        }
        let axis = (shape.len() - 1) as i32;
        let scale = 1.0 / *shape.last().unwrap() as f32;
        if !mean(a, sub[1], x, axis, scale) {
            continue;
        }
        let Some(add) = inputs_of(a, rsqrt[0], Op::ADD) else {
            continue;
        };
        for (variance, epsilon) in [(add[0], add[1]), (add[1], add[0])] {
            let Some(eps) = scalar(a, epsilon).filter(|x| x.is_finite() && *x > 0.0) else {
                continue;
            };
            let Some(var_mul) = inputs_of(a, variance, Op::MUL) else {
                continue;
            };
            for reduced in &var_mul[..2] {
                let Some(reduction) = inputs_of(a, *reduced, Op::REDUCE_SUM) else {
                    continue;
                };
                let square = reduction[0];
                let Some(sq) = inputs_of(a, square, Op::MUL) else {
                    continue;
                };
                if sq[..2] != [centered, centered]
                    || a.serialized_constant(sq[2]) != Some(&[0])
                    || !mean(a, variance, square, axis, scale)
                {
                    continue;
                }
                let axes = builder.emit_i64_const(
                    &format!("{stem}_mvn_axes"),
                    &[i64::from(axis)],
                    false,
                )?;
                let port = builder.emit_layer(
                    "MVN",
                    "opset6",
                    stem,
                    &format!("normalize_variance=\"true\" eps=\"{eps}\" eps_mode=\"inside_sqrt\""),
                    &[(builder.source(x)?, &shape), (axes, &[1])],
                    &[(OvElement::F32, &shape)],
                )[0];
                builder.set_source(output, port);
                return Ok(true);
            }
        }
    }
    Ok(false)
}
