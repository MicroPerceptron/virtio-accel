//! Bit-level reference model of XDNA2 `bfp16ebs8`, as characterized on silicon (issue #146).
//!
//! Two deliberately independent formulations live here:
//!
//! - [`encode_block`] models the hardware converter (`to_v64bfp16ebs8`) exactly as probes
//!   P0–P3 observed it: shared exponent from the max member's IEEE FP32 exponent field,
//!   subnormal inputs flushed to zero, Inf/NaN passed through structurally at `e = 255`,
//!   per-`crrnd`-mode rounding, and post-rounding renormalization when a mantissa overflows
//!   +127.
//! - [`mxint8_quantize_block`] implements OCP MX v1.0 MXINT8 quantization (block-32, E8M0
//!   scale, round-to-nearest-even, saturating) from the spec, not from the hardware.
//!
//! The unit tests pin both against the raw silicon outputs recorded under
//! `research/bfp16ebs8/results/`, so the model is verifiable without an NPU. The one
//! deliberate divergence between the two formulations is documented at
//! [`CONVERTER_OCP_OVERFLOW_DIVERGENCE`].

/// Decode one element: `value = m · 2^(e − 127 − 6)`.
pub fn decode(mantissa: i8, exponent: u8) -> f64 {
    f64::from(mantissa) * (f64::from(exponent) - 133.0).exp2()
}

/// The silicon-observed rounding functions, by `crrnd` mode value (P1: all ten bit-exact).
pub fn round_mode(mode: u32, x: f64) -> f64 {
    let half = x.abs().fract() == 0.5;
    match mode {
        0 => x.floor(),
        1 => x.ceil(),
        2 => x.trunc(),
        3 => x.abs().ceil().copysign(x),
        8 => {
            if half {
                x.floor()
            } else {
                x.round()
            }
        }
        9 => {
            if half {
                x.ceil()
            } else {
                x.round()
            }
        }
        10 => {
            if half {
                x.trunc()
            } else {
                x.round()
            }
        }
        11 => x.round(),
        12 => {
            if half {
                let down = x.floor();
                if (down as i64) % 2 == 0 { down } else { x.ceil() }
            } else {
                x.round()
            }
        }
        13 => {
            if half {
                let down = x.floor();
                if (down as i64) % 2 != 0 { down } else { x.ceil() }
            } else {
                x.round()
            }
        }
        _ => panic!("unknown crrnd mode {mode}"),
    }
}

/// The hardware converter's default mode at kernel entry (P0/P1: `rnd_floor`).
pub const HARDWARE_DEFAULT_MODE: u32 = 0;

/// Round-to-nearest-even, the mode OCP MX v1.0 requires (`rnd_conv_even`).
pub const CONV_EVEN: u32 = 12;

/// Where the hardware converter and the OCP quantizer deliberately part ways.
///
/// At an exact mantissa of ±127.5 under a mode that rounds the magnitude up, the hardware
/// converter renormalizes: it bumps the shared exponent and re-quantizes the whole block
/// (P1 `sat` case). The OCP MX v1.0 quantization procedure instead saturates the element at
/// the int8 maximum without re-selecting the scale. Data quantized by the hardware converter
/// at that boundary therefore differs from OCP-quantized data by one representation (both
/// decode to valid values; the OCP form loses the boundary value's low bit). A tier claiming
/// MXINT8 semantics must quantize with the OCP procedure (host/guest side), not with the
/// hardware converter, whenever inputs can sit on that boundary.
pub const CONVERTER_OCP_OVERFLOW_DIVERGENCE: &str = "see doc comment";

/// One IEEE FP32 exponent field (0 = zero/subnormal, 255 = Inf/NaN).
fn exponent_field(value: f32) -> u8 {
    ((value.to_bits() >> 23) & 0xff) as u8
}

/// Model of the hardware converter for one 8-element block under `mode`.
///
/// Matches probes P0–P3 on every recorded case. Behavior for blocks whose exponent would be
/// bumped past 254 is outside the probed envelope and deliberately panics.
pub fn encode_block(values: &[f32; 8], mode: u32) -> ([i8; 8], u8) {
    // Shared exponent: max member IEEE exponent field; subnormals flush (field 0 = zero).
    let mut e = values.iter().map(|v| exponent_field(*v)).max().unwrap();
    if e == 0 {
        return ([0i8; 8], 0);
    }

    loop {
        let mut out = [0i8; 8];
        let mut bumped = false;
        for (lane, value) in values.iter().enumerate() {
            if value.is_infinite() {
                out[lane] = if *value > 0.0 { 64 } else { -64 };
                continue;
            }
            if value.is_nan() {
                out[lane] = if value.is_sign_negative() { -96 } else { 96 };
                continue;
            }
            if exponent_field(*value) == 0 {
                out[lane] = 0; // flush-to-zero, including -0.0
                continue;
            }
            let exact = f64::from(*value) * (133.0 - f64::from(e)).exp2();
            let rounded = round_mode(mode, exact);
            if rounded > 127.0 {
                bumped = true;
                break;
            }
            assert!(rounded >= -128.0, "below int8: {rounded}");
            out[lane] = rounded as i8;
        }
        if !bumped {
            return (out, e);
        }
        assert!(e < 254, "renormalization past e=254 is outside the probed envelope");
        e += 1;
    }
}

/// Model of the converter over a 64-element vector (8 independent blocks).
pub fn encode_v64(values: &[f32; 64], mode: u32) -> ([i8; 64], [u8; 8]) {
    let mut mantissa = [0i8; 64];
    let mut exponent = [0u8; 8];
    for block in 0..8 {
        let mut chunk = [0f32; 8];
        chunk.copy_from_slice(&values[block * 8..block * 8 + 8]);
        let (m, e) = encode_block(&chunk, mode);
        mantissa[block * 8..block * 8 + 8].copy_from_slice(&m);
        exponent[block] = e;
    }
    (mantissa, exponent)
}

/// OCP MX v1.0 MXINT8 quantization of one block: E8M0 shared scale, int8 elements with six
/// fraction bits, round-to-nearest-even, saturating (no scale re-selection on overflow).
/// Implemented from the spec as the independent oracle; `BLOCK` is 32 for standard MX.
pub fn mxint8_quantize_block<const BLOCK: usize>(values: &[f32; BLOCK]) -> ([i8; BLOCK], u8) {
    let max_abs = values.iter().fold(0f32, |a, v| a.max(v.abs()));
    assert!(max_abs.is_finite(), "MXINT8 has no Inf/NaN representation");
    if max_abs == 0.0 {
        return ([0i8; BLOCK], 127); // scale 2^0; all elements zero
    }
    let shared = max_abs.log2().floor() as i32;
    let e = u8::try_from(127 + shared).expect("scale within E8M0 range");
    let mut out = [0i8; BLOCK];
    for (lane, value) in values.iter().enumerate() {
        let exact = f64::from(*value) * (133.0 - f64::from(e)).exp2();
        let rounded = round_mode(CONV_EVEN, exact);
        out[lane] = rounded.clamp(-128.0, 127.0) as i8;
    }
    (out, e)
}

/// Exact dot product of two encoded operand rows (any equal length), in f64.
pub fn dot_reference(a_m: &[i8], a_e: &[u8], b_m: &[i8], b_e: &[u8], block: usize) -> f64 {
    assert_eq!(a_m.len(), b_m.len());
    let mut sum = 0f64;
    for lane in 0..a_m.len() {
        let ea = a_e[lane / block];
        let eb = b_e[lane / block];
        sum += f64::from(a_m[lane])
            * f64::from(b_m[lane])
            * (f64::from(ea) + f64::from(eb) - 266.0).exp2();
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    /// P0 case A (results/p0-2026-08-27.txt): per-block powers of two.
    #[test]
    fn silicon_p0_case_a() {
        let mut values = [0f32; 64];
        for block in 0..8 {
            for lane in 0..8 {
                values[block * 8 + lane] = (2f32).powi(block as i32 - 3);
            }
        }
        let (m, e) = encode_v64(&values, HARDWARE_DEFAULT_MODE);
        assert_eq!(e, [124, 125, 126, 127, 128, 129, 130, 131]);
        assert!(m.iter().all(|&x| x == 64));
    }

    /// P0 case B: sign and range, max member -2.0 selects e = 128.
    #[test]
    fn silicon_p0_case_b() {
        let mut values = [0f32; 64];
        values[..8].copy_from_slice(&[1.0, -1.0, 0.5, -0.5, 1.5, -1.5, 127.0 / 64.0, -2.0]);
        let (m, e) = encode_v64(&values, HARDWARE_DEFAULT_MODE);
        assert_eq!(e[0], 128);
        assert_eq!(&m[..8], &[32, -32, 16, -16, 48, -48, 63, -64]);
        assert_eq!(&e[1..], &[0; 7]);
    }

    /// P0 case C: mixed magnitudes at e = 127; floor rounding of small members.
    #[test]
    fn silicon_p0_case_c() {
        let mut values = [0f32; 64];
        values[..8].copy_from_slice(&[
            1.0,
            1.0 / 64.0,
            1.0 / 128.0,
            1.5 / 64.0,
            -1.0 / 64.0,
            0.0,
            1.0 + 1.0 / 64.0,
            -1.0 - 1.0 / 64.0,
        ]);
        let (m, e) = encode_v64(&values, HARDWARE_DEFAULT_MODE);
        assert_eq!(e[0], 127);
        assert_eq!(&m[..8], &[64, 1, 0, 1, -1, 0, 65, -65]);
    }

    /// P1 sat case: exact +-127.5; floor keeps e=127 and emits -128, conv_even bumps to 128.
    #[test]
    fn silicon_p1_saturation_boundary() {
        let mut values = [0f32; 64];
        values[..8].copy_from_slice(&[
            127.5 / 64.0,
            -127.5 / 64.0,
            1.0,
            -1.0,
            127.0 / 64.0,
            -127.0 / 64.0,
            0.5,
            -0.5,
        ]);
        let (m, e) = encode_v64(&values, HARDWARE_DEFAULT_MODE);
        assert_eq!(e[0], 127, "floor never overflows +127");
        assert_eq!(&m[..2], &[127, -128], "floor emits -128 for -127.5");
        let (m, e) = encode_v64(&values, CONV_EVEN);
        assert_eq!(e[0], 128, "conv_even rounds +127.5 to 128 and renormalizes");
        assert_eq!(&m[..2], &[64, -64]);
    }

    /// P2 N2/N3/N4: negative-only block, subnormal flush, top of the exponent range.
    #[test]
    fn silicon_p2_normalization() {
        let mut n2 = [0f32; 64];
        n2[..8].copy_from_slice(&[-1.0, -0.5, -0.25, -0.75, -1.25, -1.75, -1.984375, -0.125]);
        let (m, e) = encode_v64(&n2, HARDWARE_DEFAULT_MODE);
        assert_eq!(e[0], 127);
        assert_eq!(&m[..8], &[-64, -32, -16, -48, -80, -112, -127, -8]);

        let mut n3 = [0f32; 64];
        n3[0] = f32::from_bits(0x0000_0001);
        n3[1] = f32::from_bits(0x007f_ffff);
        let (m, e) = encode_v64(&n3, HARDWARE_DEFAULT_MODE);
        assert_eq!((m[0], m[1], e[0]), (0, 0, 0), "subnormal inputs flush to zero");

        let mut n4 = [0f32; 64];
        n4[..6].copy_from_slice(&[
            f32::MAX,
            f32::MAX / 2.0,
            1.0,
            -1.0,
            (2f32).powi(120),
            -(2f32).powi(120),
        ]);
        let (m, e) = encode_v64(&n4, HARDWARE_DEFAULT_MODE);
        assert_eq!(e[0], 254);
        assert_eq!(&m[..6], &[127, 63, 0, -1, 0, -1]);
    }

    /// P3 X1: Inf/NaN structural encodings and their effect on block neighbors.
    #[test]
    fn silicon_p3_exceptional() {
        let mut values = [0f32; 64];
        values[..8].copy_from_slice(&[
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            -f32::NAN,
            1.0,
            -1.0,
            0.0,
            -0.0,
        ]);
        let (m, e) = encode_v64(&values, HARDWARE_DEFAULT_MODE);
        assert_eq!(e[0], 255);
        assert_eq!(&m[..8], &[64, -64, 96, -96, 0, -1, 0, 0]);
    }

    /// The H6 statement at model level: an MXINT8 block-32 quantization decomposed into four
    /// equal-exponent block-8 groups decodes to identical values, and the dot references
    /// agree exactly.
    #[test]
    fn mxint8_decomposition_is_value_preserving() {
        let mut values = [0f32; 32];
        let mut state = 0x5eed_cafeu32;
        for v in values.iter_mut() {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            *v = ((state >> 16) as i16 as f32) / 16384.0;
        }
        let (m32, e32) = mxint8_quantize_block::<32>(&values);
        // Decompose: four block-8 groups sharing the SAME exponent byte.
        let e8 = [e32; 4];
        let m8 = m32;
        for lane in 0..32 {
            assert_eq!(
                decode(m8[lane], e8[lane / 8]),
                decode(m32[lane], e32),
                "decomposition must not change any element value"
            );
        }
        let d32 = dot_reference(&m32, &[e32], &m32, &[e32], 32);
        let d8 = dot_reference(&m8, &e8, &m8, &e8, 8);
        assert_eq!(d32, d8);
    }

    /// The documented converter-vs-OCP divergence at the +-127.5 boundary.
    #[test]
    fn converter_and_ocp_diverge_only_at_the_overflow_boundary() {
        let boundary = 127.5 / 64.0;
        let mut block8 = [0f32; 8];
        block8[0] = boundary;
        let (m_hw, e_hw) = encode_block(&block8, CONV_EVEN);
        let mut block32 = [0f32; 32];
        block32[0] = boundary;
        let (m_ocp, e_ocp) = mxint8_quantize_block::<32>(&block32);
        // Hardware renormalizes to exactly 2.0; OCP saturates at 127/64.
        assert_eq!((m_hw[0], e_hw), (64, 128));
        assert_eq!((m_ocp[0], e_ocp), (127, 127));
        assert_eq!(decode(m_hw[0], e_hw), 2.0);
        assert_eq!(decode(m_ocp[0], e_ocp), 127.0 / 64.0);
    }
}
