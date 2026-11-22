//! Typed floating operations used by transformer graphs. All work is preparation-time.
use super::*;
use virtio_accel_tosa::{NanPropagationMode, OperatorId, TosaAnalysis};

pub(super) fn emit(
    b: &mut Builder,
    a: &TosaAnalysis<'_>,
    id: OperatorId,
    args: &[String],
    dtype: u64,
    shape: &[i32],
) -> Result<String, LoweringError> {
    let op = a.operator(id);
    let integer_minmax = dtype == I32 && matches!(op.op(), Op::MAXIMUM | Op::MINIMUM);
    if !matches!(dtype, F16 | F32) && !integer_minmax && !(dtype == BOOL && op.op() == Op::GREATER)
    {
        return Err(LoweringError::UnsupportedGraph);
    }
    Ok(match op.op() {
        Op::SUB => b.binary("sub", &args[0], &args[1], dtype, shape),
        Op::EXP => b.op("exp", &[("x", &args[0])], dtype, shape),
        Op::FLOOR | Op::CEIL => b.op(
            if op.op() == Op::FLOOR {
                "floor"
            } else {
                "ceil"
            },
            &[("x", &args[0])],
            dtype,
            shape,
        ),
        Op::GREATER => b.binary("greater", &args[0], &args[1], dtype, shape),
        Op::RSQRT | Op::RECIPROCAL => {
            // MIL's default epsilon changes TOSA arithmetic. Emit an explicit zero.
            let zero = b.constant(
                dtype,
                &[],
                if dtype == F16 { &[0, 0] } else { &[0, 0, 0, 0] },
            );
            b.op(
                if op.op() == Op::RSQRT {
                    "rsqrt"
                } else {
                    "inverse"
                },
                &[("x", &args[0]), ("epsilon", &zero)],
                dtype,
                shape,
            )
        }
        Op::MAXIMUM | Op::MINIMUM => {
            let mode = match op.source().attributes() {
                OpAttributes::Maximum { nan_mode } | OpAttributes::Minimum { nan_mode } => nan_mode,
                _ => return Err(LoweringError::UnsupportedGraph),
            };
            if mode != NanPropagationMode::PROPAGATE {
                return Err(LoweringError::UnsupportedGraph);
            }
            b.binary(
                if op.op() == Op::MAXIMUM {
                    "maximum"
                } else {
                    "minimum"
                },
                &args[0],
                &args[1],
                dtype,
                shape,
            )
        }
        Op::REDUCE_SUM | Op::REDUCE_MAX => {
            let axis = match op.source().attributes() {
                OpAttributes::ReduceSum { axis } => axis,
                OpAttributes::ReduceMax {
                    axis,
                    nan_mode: NanPropagationMode::PROPAGATE,
                } => axis,
                _ => return Err(LoweringError::UnsupportedGraph),
            };
            let axes = b.constant(I32, &[1], &axis.to_le_bytes());
            let keep = b.boolean(true);
            b.op(
                if op.op() == Op::REDUCE_SUM {
                    "reduce_sum"
                } else {
                    "reduce_max"
                },
                &[("x", &args[0]), ("axes", &axes), ("keep_dims", &keep)],
                dtype,
                shape,
            )
        }
        Op::SELECT => b.select(&args[0], &args[1], &args[2], dtype, shape),
        _ => return Err(LoweringError::UnsupportedOperator(op.op())),
    })
}
