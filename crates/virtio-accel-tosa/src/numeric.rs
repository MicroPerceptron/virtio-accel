//! Stable-Rust helpers for TOSA low-precision tensor encodings.
//!
//! These functions describe the device-neutral wire representation. They do not imply that a
//! particular accelerator can execute a graph containing the corresponding [`DType`].

use crate::DType;

/// Return the packed byte count for an INT4, INT8, FP8E4M3, or FP8E5M2 tensor.
///
/// INT4 elements are packed low nibble first. `None` means that `dtype` is not one of these
/// low-precision types. The packed byte count itself cannot overflow `usize`.
pub const fn low_precision_storage_bytes(dtype: DType, elements: usize) -> Option<usize> {
    match dtype {
        DType::INT4 => Some(elements / 2 + elements % 2),
        DType::INT8 | DType::FP8E4M3 | DType::FP8E5M2 => Some(elements),
        _ => None,
    }
}

/// Decode one signed two's-complement INT4 value packed low nibble first.
///
/// This returns all mechanically representable values, including `-8`. TOSA 1.0 tensor constants
/// use the narrower `-7..=7` range; semantic validation rejects `-8` where the specification does.
pub fn unpack_int4(bytes: &[u8], index: usize) -> Option<i8> {
    let byte = *bytes.get(index / 2)?;
    let nibble = if index % 2 == 0 {
        byte & 0x0f
    } else {
        byte >> 4
    };
    Some((nibble as i8) << 4 >> 4)
}

/// Pack two TOSA INT4 values, with `low` in the low nibble and `high` in the high nibble.
///
/// TOSA 1.0 defines INT4 values over `-7..=7`, so `-8` and out-of-range inputs are rejected.
pub const fn pack_int4(low: i8, high: i8) -> Option<u8> {
    if low < -7 || low > 7 || high < -7 || high > 7 {
        return None;
    }
    Some(((high as u8 & 0x0f) << 4) | (low as u8 & 0x0f))
}

/// Convert one TOSA/OCP FP8 E4M3 bit pattern to an exactly representable `f32` value.
///
/// E4M3 has no infinities. Exponent `0b1111` remains finite except when the fraction is `0b111`,
/// which represents NaN.
pub fn fp8e4m3_to_f32(bits: u8) -> f32 {
    let sign = u32::from(bits & 0x80) << 24;
    let exponent = u32::from((bits >> 3) & 0x0f);
    let fraction = u32::from(bits & 0x07);
    if exponent == 0 {
        return signed_fp8_subnormal(sign, fraction, 9);
    }
    if exponent == 0x0f && fraction == 0x07 {
        return f32::from_bits(sign | 0x7fc0_0000);
    }
    f32::from_bits(sign | ((exponent + 120) << 23) | (fraction << 20))
}

/// Convert one TOSA/OCP FP8 E5M2 bit pattern to an exactly representable `f32` value.
pub fn fp8e5m2_to_f32(bits: u8) -> f32 {
    let sign = u32::from(bits & 0x80) << 24;
    let exponent = u32::from((bits >> 2) & 0x1f);
    let fraction = u32::from(bits & 0x03);
    if exponent == 0 {
        return signed_fp8_subnormal(sign, fraction, 16);
    }
    if exponent == 0x1f {
        return f32::from_bits(sign | 0x7f80_0000 | (fraction << 21));
    }
    f32::from_bits(sign | ((exponent + 112) << 23) | (fraction << 21))
}

fn signed_fp8_subnormal(sign: u32, fraction: u32, scale: i32) -> f32 {
    if fraction == 0 {
        return f32::from_bits(sign);
    }
    let power_of_two = f32::from_bits(((127 - scale) as u32) << 23);
    let value = (fraction as f32) * power_of_two;
    f32::from_bits(value.to_bits() | sign)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn low_precision_sizes_are_checked_and_int4_is_nibble_packed() {
        assert_eq!(low_precision_storage_bytes(DType::INT4, 0), Some(0));
        assert_eq!(low_precision_storage_bytes(DType::INT4, 7), Some(4));
        assert_eq!(low_precision_storage_bytes(DType::INT4, 8), Some(4));
        assert_eq!(
            low_precision_storage_bytes(DType::INT4, usize::MAX),
            Some(usize::MAX / 2 + 1)
        );
        assert_eq!(low_precision_storage_bytes(DType::INT8, 8), Some(8));
        assert_eq!(low_precision_storage_bytes(DType::FP8E4M3, 8), Some(8));
        assert_eq!(low_precision_storage_bytes(DType::FP8E5M2, 8), Some(8));
        assert_eq!(low_precision_storage_bytes(DType::FP16, 8), None);

        assert_eq!(pack_int4(-7, 7), Some(0x79));
        assert_eq!(pack_int4(-8, 0), None);
        assert_eq!(unpack_int4(&[0x79], 0), Some(-7));
        assert_eq!(unpack_int4(&[0x79], 1), Some(7));
        assert_eq!(unpack_int4(&[0x08], 0), Some(-8));
        assert_eq!(unpack_int4(&[], 0), None);
    }

    #[test]
    fn fp8e4m3_decodes_signed_zero_subnormal_finite_max_and_nan() {
        assert_eq!(fp8e4m3_to_f32(0x00).to_bits(), 0.0_f32.to_bits());
        assert_eq!(fp8e4m3_to_f32(0x80).to_bits(), (-0.0_f32).to_bits());
        assert_eq!(fp8e4m3_to_f32(0x01), 2.0_f32.powi(-9));
        assert_eq!(fp8e4m3_to_f32(0x38), 1.0);
        assert_eq!(fp8e4m3_to_f32(0x7e), 448.0);
        assert!(fp8e4m3_to_f32(0x7f).is_nan());
        assert!(fp8e4m3_to_f32(0xff).is_nan());
    }

    #[test]
    fn fp8e5m2_decodes_signed_zero_subnormal_finite_max_infinity_and_nan() {
        assert_eq!(fp8e5m2_to_f32(0x00).to_bits(), 0.0_f32.to_bits());
        assert_eq!(fp8e5m2_to_f32(0x80).to_bits(), (-0.0_f32).to_bits());
        assert_eq!(fp8e5m2_to_f32(0x01), 2.0_f32.powi(-16));
        assert_eq!(fp8e5m2_to_f32(0x3c), 1.0);
        assert_eq!(fp8e5m2_to_f32(0x7b), 57_344.0);
        assert_eq!(fp8e5m2_to_f32(0x7c), f32::INFINITY);
        assert_eq!(fp8e5m2_to_f32(0xfc), f32::NEG_INFINITY);
        assert!(fp8e5m2_to_f32(0x7d).is_nan());
    }
}
