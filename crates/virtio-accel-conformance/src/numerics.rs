//! Device-neutral numerical acceptance cases for hardware backends.
//!
//! Each case couples a stable TOSA artifact with exact input shapes and a numerical oracle. Host
//! backends consume the same bytes and values, so a provider cannot quietly substitute a
//! backend-specific graph while claiming cross-device equivalence.

/// One immutable IEEE-754 binary16 tensor in a numerical acceptance case.
///
/// Elements are stored as their exact bit patterns so this stable-Rust crate does not require the
/// nightly-only primitive `f16` feature.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Float16Tensor {
    /// Static row-major shape.
    pub shape: &'static [usize],
    /// Row-major IEEE-754 binary16 element bits.
    pub bits: &'static [u16],
}

/// A stable TOSA graph and binary16 numerical oracle shared by host backends.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TosaFloat16Case {
    /// Diagnostic case name.
    pub name: &'static str,
    /// TOSA 1.0 FlatBuffer payload.
    pub artifact: &'static [u8],
    /// Block inputs in declared slot order.
    pub inputs: &'static [Float16Tensor],
    /// Block outputs in declared slot order.
    pub outputs: &'static [Float16Tensor],
}

impl TosaFloat16Case {
    /// Compare one backend output with this case's selected output oracle.
    ///
    /// Every non-NaN value must match bit-for-bit, including signed zero, infinities, and
    /// subnormals. NaN payloads may be canonicalized by the accelerator.
    pub fn output_matches(&self, output: usize, actual: &[u16]) -> bool {
        let Some(expected) = self.outputs.get(output).map(|tensor| tensor.bits) else {
            return false;
        };
        expected.len() == actual.len()
            && expected.iter().zip(actual).all(|(expected, actual)| {
                if is_binary16_nan(*expected) {
                    is_binary16_nan(*actual)
                } else {
                    expected == actual
                }
            })
    }
}

const fn is_binary16_nan(bits: u16) -> bool {
    bits & 0x7c00 == 0x7c00 && bits & 0x03ff != 0
}

/// One immutable FP32 tensor in a numerical acceptance case.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Float32Tensor {
    /// Static row-major shape.
    pub shape: &'static [usize],
    /// Row-major tensor elements.
    pub values: &'static [f32],
}

/// A stable TOSA graph and FP32 numerical oracle shared by host backends.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TosaFloat32Case {
    /// Diagnostic case name.
    pub name: &'static str,
    /// TOSA 1.0 FlatBuffer payload.
    pub artifact: &'static [u8],
    /// Block inputs in declared slot order.
    pub inputs: &'static [Float32Tensor],
    /// Block outputs in declared slot order.
    pub outputs: &'static [Float32Tensor],
    /// Maximum absolute error accepted for finite values.
    pub absolute_tolerance: f32,
    /// Maximum relative error accepted for finite values.
    pub relative_tolerance: f32,
}

impl TosaFloat32Case {
    /// Compare one backend output with this case's selected output oracle.
    pub fn output_matches(&self, output: usize, actual: &[f32]) -> bool {
        let Some(expected) = self.outputs.get(output).map(|tensor| tensor.values) else {
            return false;
        };
        expected.len() == actual.len()
            && expected.iter().zip(actual).all(|(expected, actual)| {
                if expected.is_nan() {
                    actual.is_nan()
                } else if expected.is_infinite() || *expected == 0.0 {
                    expected.to_bits() == actual.to_bits()
                } else {
                    let difference = (expected - actual).abs();
                    difference <= self.absolute_tolerance
                        || difference <= self.relative_tolerance * expected.abs()
                }
            })
    }
}

const MATMUL_LHS: &[f32] = &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
const MATMUL_RHS: &[f32] = &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
const MATMUL_OUTPUT: &[f32] = &[58.0, 64.0, 139.0, 154.0];
const MATMUL_INPUTS: &[Float32Tensor] = &[
    Float32Tensor {
        shape: &[1, 2, 3],
        values: MATMUL_LHS,
    },
    Float32Tensor {
        shape: &[1, 3, 2],
        values: MATMUL_RHS,
    },
];
const MATMUL_OUTPUTS: &[Float32Tensor] = &[Float32Tensor {
    shape: &[1, 2, 2],
    values: MATMUL_OUTPUT,
}];

/// FP32 batched matrix multiplication with non-square operands.
pub const MATMUL_FP32: TosaFloat32Case = TosaFloat32Case {
    name: "matmul-fp32",
    artifact: include_bytes!("data/matmul-fp32-v1.0.0.tosa"),
    inputs: MATMUL_INPUTS,
    outputs: MATMUL_OUTPUTS,
    absolute_tolerance: 1.0e-5,
    relative_tolerance: 1.0e-5,
};

const MAX_POOL2D_INPUT: &[f32] = &[
    1.0, 101.0, 2.0, 102.0, 3.0, 103.0, 4.0, 104.0, 5.0, 105.0, 6.0, 106.0, 7.0, 107.0, 8.0, 108.0,
    9.0, 109.0, 10.0, 110.0, 11.0, 111.0, 12.0, 112.0, 13.0, 113.0, 14.0, 114.0, 15.0, 115.0, 16.0,
    116.0,
];
const MAX_POOL2D_OUTPUT: &[f32] = &[6.0, 106.0, 8.0, 108.0, 14.0, 114.0, 16.0, 116.0];
const MAX_POOL2D_INPUTS: &[Float32Tensor] = &[Float32Tensor {
    shape: &[1, 4, 4, 2],
    values: MAX_POOL2D_INPUT,
}];
const MAX_POOL2D_OUTPUTS: &[Float32Tensor] = &[Float32Tensor {
    shape: &[1, 2, 2, 2],
    values: MAX_POOL2D_OUTPUT,
}];

/// FP32 two-channel NHWC max pooling with a 2x2 kernel and stride two.
pub const MAX_POOL2D_FP32: TosaFloat32Case = TosaFloat32Case {
    name: "max-pool2d-fp32",
    artifact: include_bytes!("data/max-pool2d-fp32-v1.0.0.tosa"),
    inputs: MAX_POOL2D_INPUTS,
    outputs: MAX_POOL2D_OUTPUTS,
    absolute_tolerance: 0.0,
    relative_tolerance: 0.0,
};

const IDENTITY_EDGE_VALUES: &[f32] = &[
    f32::NAN,
    f32::NEG_INFINITY,
    -0.0,
    0.0,
    f32::from_bits(1),
    f32::MIN_POSITIVE,
    1.0,
    f32::INFINITY,
];
const IDENTITY_EDGE_INPUTS: &[Float32Tensor] = &[Float32Tensor {
    shape: &[8],
    values: IDENTITY_EDGE_VALUES,
}];
const IDENTITY_EDGE_OUTPUTS: &[Float32Tensor] = IDENTITY_EDGE_INPUTS;

/// FP32 identity over NaN, infinities, signed zeros, a subnormal, and ordinary finite values.
pub const IDENTITY_EDGES_FP32: TosaFloat32Case = TosaFloat32Case {
    name: "identity-edges-fp32",
    artifact: include_bytes!("data/identity-edges-fp32-v1.0.0.tosa"),
    inputs: IDENTITY_EDGE_INPUTS,
    outputs: IDENTITY_EDGE_OUTPUTS,
    absolute_tolerance: 0.0,
    relative_tolerance: 0.0,
};

const MATMUL_LHS_FP16_BITS: &[u16] = &[0x3c00, 0x4000, 0x4200, 0x4400, 0x4500, 0x4600];
const MATMUL_RHS_FP16_BITS: &[u16] = &[0x4700, 0x4800, 0x4880, 0x4900, 0x4980, 0x4a00];
const MATMUL_OUTPUT_FP16_BITS: &[u16] = &[0x5340, 0x5400, 0x5858, 0x58d0];
const MATMUL_INPUTS_FP16: &[Float16Tensor] = &[
    Float16Tensor {
        shape: &[1, 2, 3],
        bits: MATMUL_LHS_FP16_BITS,
    },
    Float16Tensor {
        shape: &[1, 3, 2],
        bits: MATMUL_RHS_FP16_BITS,
    },
];
const MATMUL_OUTPUTS_FP16: &[Float16Tensor] = &[Float16Tensor {
    shape: &[1, 2, 2],
    bits: MATMUL_OUTPUT_FP16_BITS,
}];

/// Binary16 batched matrix multiplication with non-square operands.
pub const MATMUL_FP16: TosaFloat16Case = TosaFloat16Case {
    name: "matmul-fp16",
    artifact: include_bytes!("data/matmul-fp16-v1.0.0.tosa"),
    inputs: MATMUL_INPUTS_FP16,
    outputs: MATMUL_OUTPUTS_FP16,
};

const MAX_POOL2D_INPUT_FP16_BITS: &[u16] = &[
    0x3c00, 0x5650, 0x4000, 0x5660, 0x4200, 0x5670, 0x4400, 0x5680, 0x4500, 0x5690, 0x4600, 0x56a0,
    0x4700, 0x56b0, 0x4800, 0x56c0, 0x4880, 0x56d0, 0x4900, 0x56e0, 0x4980, 0x56f0, 0x4a00, 0x5700,
    0x4a80, 0x5710, 0x4b00, 0x5720, 0x4b80, 0x5730, 0x4c00, 0x5740,
];
const MAX_POOL2D_OUTPUT_FP16_BITS: &[u16] = &[
    0x4600, 0x56a0, 0x4800, 0x56c0, 0x4b00, 0x5720, 0x4c00, 0x5740,
];
const MAX_POOL2D_INPUTS_FP16: &[Float16Tensor] = &[Float16Tensor {
    shape: &[1, 4, 4, 2],
    bits: MAX_POOL2D_INPUT_FP16_BITS,
}];
const MAX_POOL2D_OUTPUTS_FP16: &[Float16Tensor] = &[Float16Tensor {
    shape: &[1, 2, 2, 2],
    bits: MAX_POOL2D_OUTPUT_FP16_BITS,
}];

/// Binary16 two-channel NHWC max pooling with a 2x2 kernel and stride two.
pub const MAX_POOL2D_FP16: TosaFloat16Case = TosaFloat16Case {
    name: "max-pool2d-fp16",
    artifact: include_bytes!("data/max-pool2d-fp16-v1.0.0.tosa"),
    inputs: MAX_POOL2D_INPUTS_FP16,
    outputs: MAX_POOL2D_OUTPUTS_FP16,
};

const IDENTITY_EDGE_FP16_BITS: &[u16] = &[
    0x7e00, 0xfc00, 0x8000, 0x0000, 0x0001, 0x0400, 0x3c00, 0x7c00,
];
const IDENTITY_EDGE_INPUTS_FP16: &[Float16Tensor] = &[Float16Tensor {
    shape: &[8],
    bits: IDENTITY_EDGE_FP16_BITS,
}];
const IDENTITY_EDGE_OUTPUTS_FP16: &[Float16Tensor] = IDENTITY_EDGE_INPUTS_FP16;

/// Binary16 identity over NaN, infinities, signed zeros, a subnormal, and finite values.
pub const IDENTITY_EDGES_FP16: TosaFloat16Case = TosaFloat16Case {
    name: "identity-edges-fp16",
    artifact: include_bytes!("data/identity-edges-fp16-v1.0.0.tosa"),
    inputs: IDENTITY_EDGE_INPUTS_FP16,
    outputs: IDENTITY_EDGE_OUTPUTS_FP16,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matmul_oracle_checks_shape_values_and_signed_zero() {
        assert!(MATMUL_FP32.output_matches(0, MATMUL_OUTPUT));
        assert!(!MATMUL_FP32.output_matches(0, &[58.0, 64.0]));
        assert!(!MATMUL_FP32.output_matches(1, MATMUL_OUTPUT));

        let zero = TosaFloat32Case {
            outputs: &[Float32Tensor {
                shape: &[1],
                values: &[-0.0],
            }],
            ..MATMUL_FP32
        };
        assert!(zero.output_matches(0, &[-0.0]));
        assert!(!zero.output_matches(0, &[0.0]));
    }

    #[test]
    fn max_pool_oracle_preserves_nhwc_order() {
        assert!(MAX_POOL2D_FP32.output_matches(0, MAX_POOL2D_OUTPUT));
        assert!(
            !MAX_POOL2D_FP32.output_matches(0, &[6.0, 8.0, 14.0, 16.0, 106.0, 108.0, 114.0, 116.0])
        );
    }

    #[test]
    fn identity_edge_oracle_handles_nonfinite_and_signed_zero_values() {
        assert!(IDENTITY_EDGES_FP32.output_matches(0, IDENTITY_EDGE_VALUES));
        let mut wrong_zero = IDENTITY_EDGE_VALUES.to_vec();
        wrong_zero[2] = 0.0;
        assert!(!IDENTITY_EDGES_FP32.output_matches(0, &wrong_zero));
    }

    #[test]
    fn fp16_oracle_is_exact_except_for_nan_payloads() {
        assert!(MATMUL_FP16.output_matches(0, MATMUL_OUTPUT_FP16_BITS));
        assert!(!MATMUL_FP16.output_matches(0, &[0x5340, 0x5400]));
        assert!(!MATMUL_FP16.output_matches(1, MATMUL_OUTPUT_FP16_BITS));

        let mut canonicalized_nan = IDENTITY_EDGE_FP16_BITS.to_vec();
        canonicalized_nan[0] = 0x7fff;
        assert!(IDENTITY_EDGES_FP16.output_matches(0, &canonicalized_nan));
        canonicalized_nan[2] = 0x0000;
        assert!(!IDENTITY_EDGES_FP16.output_matches(0, &canonicalized_nan));
    }

    #[test]
    fn fp16_max_pool_oracle_preserves_nhwc_order() {
        assert!(MAX_POOL2D_FP16.output_matches(0, MAX_POOL2D_OUTPUT_FP16_BITS));
        assert!(!MAX_POOL2D_FP16.output_matches(
            0,
            &[
                0x4600, 0x4800, 0x4b00, 0x4c00, 0x56a0, 0x56c0, 0x5720, 0x5740
            ]
        ));
    }
}
