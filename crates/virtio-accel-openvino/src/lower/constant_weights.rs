//! Exact load-time evaluation for devices without a proven W4A16 kernel.
use super::*;

/// Fold only the explicit constant-weight dequantization pattern. CPU otherwise
/// expands i4 during each inference, and the NPU fuses arbitrary scales without
/// preserving the intermediate FP16 rounding. GPU keeps the compressed pattern.
pub(super) fn fold(
    builder: &mut IrBuilder,
    analysis: &TosaAnalysis<'_>,
    inputs: &[ValueId],
    output: ValueId,
    stem: &str,
) -> Result<bool, LoweringError> {
    if tensor(analysis, output)?.dtype() != DType::FP16 {
        return Ok(false);
    }
    for (decoded, scale) in [(inputs[0], inputs[1]), (inputs[1], inputs[0])] {
        let Some(cast) = analysis.value(decoded).producer() else {
            continue;
        };
        if analysis.operator(cast).op() != Op::CAST {
            continue;
        }
        let source = analysis.operator_inputs(cast)[0];
        if tensor(analysis, source)?.dtype() != DType::INT8
            || tensor(analysis, decoded)?.dtype() != DType::FP16
            || tensor(analysis, scale)?.dtype() != DType::FP16
        {
            continue;
        }
        let (Some(weights), Some(scales)) = (
            analysis.serialized_constant(source),
            analysis.serialized_constant(scale),
        ) else {
            continue;
        };
        let shape = static_dims(tensor(analysis, decoded)?)?;
        let scale_shape = static_dims(tensor(analysis, scale)?)?;
        if shape.len() != 3 || scale_shape != [shape[0], shape[1], 1] {
            continue;
        }
        let group = usize::try_from(shape[2]).map_err(|_| LoweringError::ResourceLimit)?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(
                weights
                    .len()
                    .checked_mul(2)
                    .ok_or(LoweringError::ResourceLimit)?,
            )
            .map_err(|_| LoweringError::ResourceLimit)?;
        for (i, weight) in weights.iter().enumerate() {
            let offset = (i / group) * 2;
            let scale =
                half::f16::from_bits(u16::from_le_bytes([scales[offset], scales[offset + 1]]));
            // I8 -> F16 is exact. Its product with a binary16 scale needs at most
            // 18 significand bits: f32 evaluates it exactly, then half rounds once
            // to the declared FP16 intermediate (including subnormals/overflow).
            let value = half::f16::from_f32(f32::from(*weight as i8) * scale.to_f32());
            bytes.extend_from_slice(&value.to_bits().to_le_bytes());
        }
        let port = builder.emit_const(
            &format!("{stem}_rounded_weights"),
            OvElement::F16,
            &shape,
            &bytes,
        )?;
        builder.set_source(output, port);
        return Ok(true);
    }
    Ok(false)
}
