use core::fmt;

use crate::{
    AnalyzedValueKind, DType, LevelLimits, Op, OpAttributes, OperatorId, RuntimeCondition,
    TosaAnalysis, ValueId,
};

/// One host-readable dynamic CTC value supplied for specialization.
#[derive(Clone, Copy, Debug)]
pub struct RuntimeValue<'a> {
    pub value: ValueId,
    pub bytes: &'a [u8],
}

/// Failure while resolving or checking dynamic CTC data.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RuntimeErrorKind {
    ValuesOutOfOrder,
    MissingValue,
    InvalidEncoding,
    RequiredConditionFailed,
}

/// Located runtime-specialization failure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RuntimeError {
    pub operator: Option<OperatorId>,
    pub value: Option<ValueId>,
    pub kind: RuntimeErrorKind,
}

impl fmt::Display for RuntimeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

/// Result of checking every dynamic CTC value needed by one specialization.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RuntimeValidation {
    /// At least one advisory TOSA `REQUIRE` condition failed.
    pub unpredictable: bool,
}

/// Validate dynamic CTC encodings and every associated mandatory `ERROR_IF` condition.
///
/// `values` must be strictly sorted by [`ValueId`]. Only dynamic CTC values are inspected; normal
/// tensor inputs stay on the provider's direct execution path. Per-element advisory conditions are
/// represented in [`TosaAnalysis::conditions`] but deliberately are not scanned here.
pub fn validate_runtime_values(
    analysis: &TosaAnalysis<'_>,
    values: &[RuntimeValue<'_>],
) -> Result<RuntimeValidation, RuntimeError> {
    if values.windows(2).any(|pair| pair[0].value >= pair[1].value) {
        return Err(RuntimeError {
            operator: None,
            value: None,
            kind: RuntimeErrorKind::ValuesOutOfOrder,
        });
    }

    let mut validation = RuntimeValidation::default();
    for operator in analysis.operators() {
        let dynamic = analysis
            .operator_conditions(operator.id())
            .iter()
            .any(|condition| matches!(condition, RuntimeCondition::DynamicCompileTimeInput { .. }));
        if !dynamic {
            continue;
        }
        for condition in analysis.operator_conditions(operator.id()) {
            if let RuntimeCondition::DynamicCompileTimeInput { value, .. } = *condition {
                let bytes = lookup(values, value).ok_or(RuntimeError {
                    operator: Some(operator.id()),
                    value: Some(value),
                    kind: RuntimeErrorKind::MissingValue,
                })?;
                if !valid_encoding(analysis, value, bytes) {
                    return Err(RuntimeError {
                        operator: Some(operator.id()),
                        value: Some(value),
                        kind: RuntimeErrorKind::InvalidEncoding,
                    });
                }
            }
        }
        match check_operator(analysis, values, operator.id())? {
            ConditionResult::Valid => {}
            ConditionResult::Unpredictable => validation.unpredictable = true,
        }
    }
    Ok(validation)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ConditionResult {
    Valid,
    Unpredictable,
}

fn lookup<'a>(values: &'a [RuntimeValue<'_>], value: ValueId) -> Option<&'a [u8]> {
    values
        .binary_search_by_key(&value, |candidate| candidate.value)
        .ok()
        .map(|index| values[index].bytes)
}

fn data_for<'a>(
    analysis: &'a TosaAnalysis<'_>,
    values: &'a [RuntimeValue<'_>],
    value: ValueId,
) -> Option<&'a [u8]> {
    analysis
        .serialized_constant(value)
        .or_else(|| lookup(values, value))
}

fn valid_encoding(analysis: &TosaAnalysis<'_>, value: ValueId, bytes: &[u8]) -> bool {
    let analyzed = analysis.value(value);
    let (dtype, elements) = match analyzed.kind() {
        AnalyzedValueKind::Tensor(tensor) => {
            let Some(_) = tensor.rank() else {
                return false;
            };
            let Some(elements) = tensor.dimensions().try_fold(1_usize, |count, dimension| {
                count.checked_mul(usize::try_from(dimension).ok()?)
            }) else {
                return false;
            };
            (tensor.dtype(), elements)
        }
        AnalyzedValueKind::Shape(shape) => (DType::SHAPE, shape.rank() as usize),
    };
    let expected = match dtype {
        DType::INT4 => elements.div_ceil(2),
        DType::BOOL | DType::INT8 | DType::FP8E4M3 | DType::FP8E5M2 => elements,
        DType::INT16 | DType::FP16 | DType::BF16 => elements.saturating_mul(2),
        DType::INT32 | DType::FP32 => elements.saturating_mul(4),
        DType::INT48 | DType::SHAPE => elements.saturating_mul(8),
        _ => return false,
    };
    if bytes.len() != expected {
        return false;
    }
    match dtype {
        DType::BOOL => bytes.iter().all(|value| *value <= 1),
        DType::INT4 => (0..elements).all(|index| integer_at(dtype, bytes, index) != Some(-8)),
        DType::INT48 => (0..elements).all(|index| {
            integer_at(dtype, bytes, index)
                .is_some_and(|value| (-(1_i64 << 47)..(1_i64 << 47)).contains(&value))
        }),
        DType::SHAPE => {
            let magnitude = 1_i128 << analysis.target().level.limits().max_log2_size;
            (0..elements).all(|index| {
                shape_at(bytes, index).is_some_and(|value| {
                    i128::from(value) >= -magnitude && i128::from(value) < magnitude
                })
            })
        }
        _ => true,
    }
}

fn check_operator(
    analysis: &TosaAnalysis<'_>,
    runtime: &[RuntimeValue<'_>],
    operator: OperatorId,
) -> Result<ConditionResult, RuntimeError> {
    let plan = analysis.operator(operator);
    let inputs = analysis.operator_inputs(operator);
    let fail = |value| RuntimeError {
        operator: Some(operator),
        value: Some(value),
        kind: RuntimeErrorKind::RequiredConditionFailed,
    };
    let data = |index: usize| {
        data_for(analysis, runtime, inputs[index]).ok_or(RuntimeError {
            operator: Some(operator),
            value: Some(inputs[index]),
            kind: RuntimeErrorKind::MissingValue,
        })
    };

    match plan.op() {
        Op::AVG_POOL2D => {
            for index in [1, 2] {
                if !zero_point_valid(analysis, inputs[index], data(index)?, false) {
                    return Err(fail(inputs[index]));
                }
            }
            Ok(ConditionResult::Valid)
        }
        Op::CONV2D | Op::CONV3D | Op::DEPTHWISE_CONV2D | Op::TRANSPOSE_CONV2D => {
            for index in [3, 4] {
                if !zero_point_valid(analysis, inputs[index], data(index)?, false) {
                    return Err(fail(inputs[index]));
                }
            }
            Ok(ConditionResult::Valid)
        }
        Op::MATMUL => {
            for index in [2, 3] {
                if !zero_point_valid(analysis, inputs[index], data(index)?, false) {
                    return Err(fail(inputs[index]));
                }
            }
            Ok(ConditionResult::Valid)
        }
        Op::MUL => {
            let shift = integer_value(analysis, inputs[2], data(2)?, 0);
            let dtype = tensor_dtype(analysis, inputs[0]);
            if shift.is_some_and(|shift| {
                (0..=63).contains(&shift) && (dtype == DType::INT32 || shift == 0)
            }) {
                Ok(ConditionResult::Valid)
            } else {
                Ok(ConditionResult::Unpredictable)
            }
        }
        Op::TABLE => Ok(ConditionResult::Valid),
        Op::NEGATE => {
            for index in [1, 2] {
                if !zero_point_valid(analysis, inputs[index], data(index)?, false) {
                    return Err(fail(inputs[index]));
                }
            }
            Ok(ConditionResult::Valid)
        }
        Op::PAD => {
            if !zero_point_valid(analysis, inputs[2], data(2)?, false) {
                return Err(fail(inputs[2]));
            }
            if !pad_valid(analysis, operator, inputs, data(1)?) {
                return Err(fail(inputs[1]));
            }
            Ok(ConditionResult::Valid)
        }
        Op::RESHAPE => {
            if reshape_valid(analysis, operator, inputs, data(1)?) {
                Ok(ConditionResult::Valid)
            } else {
                Err(fail(inputs[1]))
            }
        }
        Op::SLICE => {
            if slice_valid(analysis, operator, inputs, data(1)?, data(2)?) {
                Ok(ConditionResult::Valid)
            } else {
                Err(fail(inputs[1]))
            }
        }
        Op::TILE => {
            if tile_valid(analysis, operator, inputs, data(1)?) {
                Ok(ConditionResult::Valid)
            } else {
                Err(fail(inputs[1]))
            }
        }
        Op::RESIZE => {
            if resize_valid(
                analysis,
                operator,
                inputs,
                data(1)?,
                data(2)?,
                data(3)?,
                analysis.target().level.limits(),
            ) {
                Ok(ConditionResult::Valid)
            } else {
                Err(fail(inputs[1]))
            }
        }
        Op::RESCALE => {
            let OpAttributes::Rescale {
                input_unsigned,
                output_unsigned,
                ..
            } = plan.source().attributes()
            else {
                unreachable!()
            };
            if !zero_point_valid(analysis, inputs[3], data(3)?, input_unsigned) {
                return Err(fail(inputs[3]));
            }
            if !zero_point_valid(analysis, inputs[4], data(4)?, output_unsigned) {
                return Err(fail(inputs[4]));
            }
            let multipliers = data(1)?;
            let shifts = data(2)?;
            let count = tensor_elements(analysis, inputs[1]).unwrap_or(0);
            if (0..count).all(|index| {
                integer_value(analysis, inputs[1], multipliers, index)
                    .is_some_and(|value| value >= 0)
                    && integer_value(analysis, inputs[2], shifts, index)
                        .is_some_and(|value| (2..=62).contains(&value))
            }) {
                Ok(ConditionResult::Valid)
            } else {
                Ok(ConditionResult::Unpredictable)
            }
        }
        _ => Ok(ConditionResult::Valid),
    }
}

fn zero_point_valid(
    analysis: &TosaAnalysis<'_>,
    value: ValueId,
    bytes: &[u8],
    unsigned: bool,
) -> bool {
    let dtype = tensor_dtype(analysis, value);
    dtype == DType::INT8
        || (dtype == DType::INT16
            && unsigned
            && bytes
                .get(..2)
                .and_then(|bytes| <[u8; 2]>::try_from(bytes).ok())
                .is_some_and(|bytes| matches!(u16::from_le_bytes(bytes), 0 | 32_768)))
        || value_is_zero(dtype, bytes, 0)
}

fn value_is_zero(dtype: DType, bytes: &[u8], index: usize) -> bool {
    match dtype {
        DType::INT4 | DType::INT8 | DType::INT16 | DType::INT32 | DType::INT48 => {
            integer_at(dtype, bytes, index) == Some(0)
        }
        DType::FP8E4M3 | DType::FP8E5M2 => bytes.get(index).is_some_and(|bits| bits & 0x7f == 0),
        DType::FP16 | DType::BF16 => bytes
            .get(index * 2..index * 2 + 2)
            .and_then(|bytes| <[u8; 2]>::try_from(bytes).ok())
            .is_some_and(|bytes| u16::from_le_bytes(bytes) & 0x7fff == 0),
        DType::FP32 => bytes
            .get(index * 4..index * 4 + 4)
            .and_then(|bytes| <[u8; 4]>::try_from(bytes).ok())
            .is_some_and(|bytes| u32::from_le_bytes(bytes) & 0x7fff_ffff == 0),
        _ => false,
    }
}

fn integer_value(
    analysis: &TosaAnalysis<'_>,
    value: ValueId,
    bytes: &[u8],
    index: usize,
) -> Option<i64> {
    integer_at(tensor_dtype(analysis, value), bytes, index)
}

fn integer_at(dtype: DType, bytes: &[u8], index: usize) -> Option<i64> {
    let width = match dtype {
        DType::INT4 | DType::INT8 => 1,
        DType::INT16 => 2,
        DType::INT32 => 4,
        DType::INT48 => 8,
        _ => return None,
    };
    let start = index.checked_mul(width)?;
    match dtype {
        DType::INT4 => {
            let byte = *bytes.get(index / 2)?;
            let nibble = if index % 2 == 0 {
                byte & 0x0f
            } else {
                byte >> 4
            };
            Some(i64::from((nibble as i8) << 4 >> 4))
        }
        DType::INT8 => Some(i64::from(*bytes.get(start)? as i8)),
        DType::INT16 => Some(i64::from(i16::from_le_bytes(
            bytes.get(start..start + 2)?.try_into().ok()?,
        ))),
        DType::INT32 => Some(i64::from(i32::from_le_bytes(
            bytes.get(start..start + 4)?.try_into().ok()?,
        ))),
        DType::INT48 => Some(i64::from_le_bytes(
            bytes.get(start..start + 8)?.try_into().ok()?,
        )),
        _ => None,
    }
}

fn shape_at(bytes: &[u8], index: usize) -> Option<i64> {
    let start = index.checked_mul(8)?;
    Some(i64::from_le_bytes(
        bytes.get(start..start + 8)?.try_into().ok()?,
    ))
}

fn tensor_dtype(analysis: &TosaAnalysis<'_>, value: ValueId) -> DType {
    match analysis.value(value).kind() {
        AnalyzedValueKind::Tensor(tensor) => tensor.dtype(),
        AnalyzedValueKind::Shape(_) => DType::SHAPE,
    }
}

fn tensor_elements(analysis: &TosaAnalysis<'_>, value: ValueId) -> Option<usize> {
    match analysis.value(value).kind() {
        AnalyzedValueKind::Tensor(tensor) => {
            let _ = tensor.rank()?;
            tensor.dimensions().try_fold(1_usize, |count, dimension| {
                count.checked_mul(usize::try_from(dimension).ok()?)
            })
        }
        AnalyzedValueKind::Shape(shape) => Some(shape.rank() as usize),
    }
}

fn pad_valid(
    analysis: &TosaAnalysis<'_>,
    operator: OperatorId,
    inputs: &[ValueId],
    padding: &[u8],
) -> bool {
    let output = analysis.operator_outputs(operator)[0];
    let Some(rank) = tensor_rank(analysis, inputs[0]) else {
        return false;
    };
    tensor_rank(analysis, output) == Some(rank)
        && (0..rank).all(|index| {
            let (Some(input), Some(output)) = (
                tensor_dimension(analysis, inputs[0], index),
                tensor_dimension(analysis, output, index),
            ) else {
                return false;
            };
            let before = shape_at(padding, index * 2);
            let after = shape_at(padding, index * 2 + 1);
            before.is_some_and(|before| before >= 0)
                && after.is_some_and(|after| after >= 0)
                && before.zip(after).is_some_and(|(before, after)| {
                    i128::from(input) + i128::from(before) + i128::from(after) == i128::from(output)
                })
        })
}

fn reshape_valid(
    analysis: &TosaAnalysis<'_>,
    operator: OperatorId,
    inputs: &[ValueId],
    shape: &[u8],
) -> bool {
    let output = analysis.operator_outputs(operator)[0];
    let dimensions_match = tensor_rank(analysis, output).is_some_and(|rank| {
        (0..rank).all(|index| {
            tensor_dimension(analysis, output, index)
                .is_some_and(|dimension| shape_at(shape, index) == Some(i64::from(dimension)))
        })
    });
    dimensions_match && tensor_elements(analysis, inputs[0]) == tensor_elements(analysis, output)
}

fn slice_valid(
    analysis: &TosaAnalysis<'_>,
    operator: OperatorId,
    inputs: &[ValueId],
    starts: &[u8],
    sizes: &[u8],
) -> bool {
    let output = analysis.operator_outputs(operator)[0];
    let Some(rank) = tensor_rank(analysis, inputs[0]) else {
        return false;
    };
    tensor_rank(analysis, output) == Some(rank)
        && (0..rank).all(|index| {
            let (Some(input), Some(output)) = (
                tensor_dimension(analysis, inputs[0], index),
                tensor_dimension(analysis, output, index),
            ) else {
                return false;
            };
            let start = shape_at(starts, index);
            let size = shape_at(sizes, index);
            start.is_some_and(|start| start >= 0)
                && size.is_some_and(|size| size > 0)
                && start.zip(size).is_some_and(|(start, size)| {
                    i128::from(start) + i128::from(size) <= i128::from(input)
                        && size == i64::from(output)
                })
        })
}

fn tile_valid(
    analysis: &TosaAnalysis<'_>,
    operator: OperatorId,
    inputs: &[ValueId],
    multiples: &[u8],
) -> bool {
    let output = analysis.operator_outputs(operator)[0];
    let Some(rank) = tensor_rank(analysis, inputs[0]) else {
        return false;
    };
    tensor_rank(analysis, output) == Some(rank)
        && (0..rank).all(|index| {
            let (Some(input), Some(output)) = (
                tensor_dimension(analysis, inputs[0], index),
                tensor_dimension(analysis, output, index),
            ) else {
                return false;
            };
            shape_at(multiples, index).is_some_and(|multiple| {
                multiple >= 1 && i128::from(input) * i128::from(multiple) == i128::from(output)
            })
        })
}

fn resize_valid(
    analysis: &TosaAnalysis<'_>,
    operator: OperatorId,
    inputs: &[ValueId],
    scale: &[u8],
    offset: &[u8],
    border: &[u8],
    limits: LevelLimits,
) -> bool {
    let Some(input) = dimensions4(analysis, inputs[0]) else {
        return false;
    };
    let Some(output) = dimensions4(analysis, analysis.operator_outputs(operator)[0]) else {
        return false;
    };
    let (Some(yn_), Some(yd), Some(xn), Some(xd), Some(oy), Some(ox), Some(by), Some(bx)) = (
        shape_at(scale, 0).map(i128::from),
        shape_at(scale, 1).map(i128::from),
        shape_at(scale, 2).map(i128::from),
        shape_at(scale, 3).map(i128::from),
        shape_at(offset, 0).map(i128::from),
        shape_at(offset, 1).map(i128::from),
        shape_at(border, 0).map(i128::from),
        shape_at(border, 1).map(i128::from),
    ) else {
        return false;
    };
    if yn_ <= 0
        || yd <= 0
        || xn <= 0
        || xd <= 0
        || yn_ > 2_048
        || xn > 2_048
        || yn_ > i128::from(limits.max_scale) * yd
        || xn > i128::from(limits.max_scale) * xd
        || yd >= 16 * yn_
        || xd >= 16 * xn
        || !(-yn_..16 * yn_).contains(&oy)
        || !(-xn..16 * xn).contains(&ox)
        || !(-16 * yn_..yn_).contains(&by)
        || !(-16 * xn..xn).contains(&bx)
        || [input[1], input[2], output[1], output[2]]
            .into_iter()
            .any(|value| value >= 16_384)
    {
        return false;
    }
    let height_numerator = (i128::from(input[1]) - 1) * yn_ - oy + by;
    let width_numerator = (i128::from(input[2]) - 1) * xn - ox + bx;
    height_numerator % yd == 0
        && width_numerator % xd == 0
        && output
            == [
                input[0],
                i64::try_from(height_numerator / yd + 1).unwrap_or(i64::MIN),
                i64::try_from(width_numerator / xd + 1).unwrap_or(i64::MIN),
                input[3],
            ]
}

fn dimensions4(analysis: &TosaAnalysis<'_>, value: ValueId) -> Option<[i64; 4]> {
    (tensor_rank(analysis, value) == Some(4)).then(|| {
        [
            i64::from(tensor_dimension(analysis, value, 0).unwrap()),
            i64::from(tensor_dimension(analysis, value, 1).unwrap()),
            i64::from(tensor_dimension(analysis, value, 2).unwrap()),
            i64::from(tensor_dimension(analysis, value, 3).unwrap()),
        ]
    })
}

fn tensor_rank(analysis: &TosaAnalysis<'_>, value: ValueId) -> Option<usize> {
    match analysis.value(value).kind() {
        AnalyzedValueKind::Tensor(tensor) => tensor.rank(),
        AnalyzedValueKind::Shape(_) => None,
    }
}

fn tensor_dimension(analysis: &TosaAnalysis<'_>, value: ValueId, index: usize) -> Option<i32> {
    match analysis.value(value).kind() {
        AnalyzedValueKind::Tensor(tensor) => tensor.dimension(index),
        AnalyzedValueKind::Shape(_) => None,
    }
}
