//! Exact integer primitives shared by TOSA conformance and provider legalization.
//!
//! These functions implement the load-bearing arithmetic from TOSA 1.0.1 rather than relying on
//! Rust's build-profile-dependent overflow behavior. A provider may use wider native operations,
//! but its visible result must match these helpers for every predictable input.

/// A TOSA integer arithmetic precondition was not satisfied.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IntegerError {
    /// Operands that form one dot product have different lengths.
    LengthMismatch,
    /// TOSA requires a non-negative fixed-point multiplier.
    NegativeMultiplier,
    /// TOSA scaling shifts are restricted to the inclusive range `2..=62`.
    ShiftOutOfRange,
    /// The input violates an operator `REQUIRE` range.
    InputOutOfRange,
    /// Exact intermediate arithmetic does not fit its TOSA accumulator type.
    AccumulatorOverflow,
}

/// Compute an exact signed INT8 dot product with INT32 accumulation.
///
/// `left` and `right` contain the tensor's two's-complement storage bytes. Zero points are
/// subtracted before multiplication, and `bias` initializes the accumulator. Wrapping is never
/// used: a result outside INT32 is reported because TOSA classifies such an input as unpredictable
/// rather than defining modular arithmetic.
pub fn dot_i8_i32(
    left: &[u8],
    right: &[u8],
    left_zero_point: i8,
    right_zero_point: i8,
    bias: i32,
) -> Result<i32, IntegerError> {
    if left.len() != right.len() {
        return Err(IntegerError::LengthMismatch);
    }
    let left_zero_point = i64::from(left_zero_point);
    let right_zero_point = i64::from(right_zero_point);
    let mut accumulator = i64::from(bias);
    for (&left, &right) in left.iter().zip(right) {
        let left = i64::from(left as i8) - left_zero_point;
        let right = i64::from(right as i8) - right_zero_point;
        accumulator = accumulator
            .checked_add(left * right)
            .ok_or(IntegerError::AccumulatorOverflow)?;
    }
    i32::try_from(accumulator).map_err(|_| IntegerError::AccumulatorOverflow)
}

/// Apply TOSA's 32-bit fixed-point scaling helper exactly.
///
/// This is `apply_scale_32` from TOSA 1.0.1. The arithmetic right shift is explicit and every
/// pseudocode `REQUIRE` is checked before calculating the result.
pub fn apply_scale_32(
    value: i32,
    multiplier: i32,
    shift: i8,
    double_round: bool,
) -> Result<i32, IntegerError> {
    if multiplier < 0 {
        return Err(IntegerError::NegativeMultiplier);
    }
    let shift = checked_shift(shift)?;
    let value64 = i64::from(value);
    let bound = 1_i64 << (shift - 1);
    if !(-bound..bound).contains(&value64) {
        return Err(IntegerError::InputOutOfRange);
    }

    let mut round = 1_i64 << (shift - 1);
    if double_round && shift > 31 {
        if value >= 0 {
            round += 1_i64 << 30;
        } else {
            round -= 1_i64 << 30;
        }
    }
    let result = value64
        .checked_mul(i64::from(multiplier))
        .and_then(|value| value.checked_add(round))
        .ok_or(IntegerError::AccumulatorOverflow)?
        >> shift;
    i32::try_from(result).map_err(|_| IntegerError::AccumulatorOverflow)
}

/// Apply TOSA's 16-bit fixed-point scaling helper exactly.
///
/// `value` is represented in an `i64`, but must fit TOSA's signed 48-bit accumulator domain.
pub fn apply_scale_16(value: i64, multiplier: i16, shift: i8) -> Result<i32, IntegerError> {
    if multiplier < 0 {
        return Err(IntegerError::NegativeMultiplier);
    }
    if !(-(1_i64 << 47)..(1_i64 << 47)).contains(&value) {
        return Err(IntegerError::InputOutOfRange);
    }
    let shift = checked_shift(shift)?;
    let round = 1_i64 << (shift - 1);
    let result = value
        .checked_mul(i64::from(multiplier))
        .and_then(|value| value.checked_add(round))
        .ok_or(IntegerError::AccumulatorOverflow)?
        >> shift;
    i32::try_from(result).map_err(|_| IntegerError::AccumulatorOverflow)
}

/// Rescale one INT32 accumulator into a signed INT8 tensor value.
///
/// The input zero point for an INT32 TOSA tensor is necessarily zero. `output_zero_point` is added
/// after scaling and the final value is saturated to the signed INT8 storage range.
pub fn rescale_i32_to_i8(
    value: i32,
    multiplier: i32,
    shift: i8,
    output_zero_point: i8,
    double_round: bool,
) -> Result<i8, IntegerError> {
    let scaled = apply_scale_32(value, multiplier, shift, double_round)?;
    let shifted = i64::from(scaled) + i64::from(output_zero_point);
    Ok(shifted.clamp(i64::from(i8::MIN), i64::from(i8::MAX)) as i8)
}

const fn checked_shift(shift: i8) -> Result<u32, IntegerError> {
    if shift < 2 || shift > 62 {
        Err(IntegerError::ShiftOutOfRange)
    } else {
        Ok(shift as u32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_product_interprets_storage_as_signed_and_applies_zero_points() {
        let left = [0x80, 0xff, 0x00, 0x7f];
        let right = [0x7f, 0x01, 0xff, 0x80];
        // Sum of each `(left - -2) * (right - 1)` term plus the bias.
        let expected = -32_504;
        assert_eq!(dot_i8_i32(&left, &right, -2, 1, 17), Ok(expected));
    }

    #[test]
    fn dot_product_rejects_shape_and_accumulator_violations() {
        assert_eq!(
            dot_i8_i32(&[1], &[], 0, 0, 0),
            Err(IntegerError::LengthMismatch)
        );
        let positive = alloc::vec![0x7f; 140_000];
        assert_eq!(
            dot_i8_i32(&positive, &positive, -128, -128, i32::MAX),
            Err(IntegerError::AccumulatorOverflow)
        );
    }

    #[test]
    fn scale_32_matches_tosa_unity_and_signed_rounding() {
        let unity = 1_i32 << 30;
        for value in [-127, -1, 0, 1, 127] {
            assert_eq!(apply_scale_32(value, unity, 30, false), Ok(value));
        }
        assert_eq!(apply_scale_32(3, 2, 3, false), Ok(1));
        assert_eq!(apply_scale_32(-3, 2, 3, false), Ok(-1));
    }

    #[test]
    fn scale_32_checks_every_pseudocode_precondition() {
        assert_eq!(
            apply_scale_32(0, -1, 30, false),
            Err(IntegerError::NegativeMultiplier)
        );
        assert_eq!(
            apply_scale_32(0, 1, 1, false),
            Err(IntegerError::ShiftOutOfRange)
        );
        assert_eq!(
            apply_scale_32(2, 1, 2, false),
            Err(IntegerError::InputOutOfRange)
        );
    }

    #[test]
    fn scale_16_matches_tosa_unity_and_checks_int48() {
        let unity = 1_i16 << 14;
        for value in [-32_768, -1, 0, 1, 32_767] {
            assert_eq!(apply_scale_16(value, unity, 14), Ok(value as i32));
        }
        assert_eq!(
            apply_scale_16(1_i64 << 47, unity, 14),
            Err(IntegerError::InputOutOfRange)
        );
    }

    #[test]
    fn rescale_saturates_only_after_output_zero_point() {
        let unity = 1_i32 << 30;
        assert_eq!(rescale_i32_to_i8(100, unity, 30, 20, false), Ok(120));
        assert_eq!(rescale_i32_to_i8(120, unity, 30, 20, false), Ok(127));
        assert_eq!(rescale_i32_to_i8(-120, unity, 30, -20, false), Ok(-128));
    }
}
