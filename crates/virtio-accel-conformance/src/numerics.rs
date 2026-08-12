//! Device-neutral numerical acceptance cases for hardware backends.
//!
//! Each case couples a stable TOSA artifact with exact input shapes and a numerical oracle. Host
//! backends consume the same bytes and values, so a provider cannot quietly substitute a
//! backend-specific graph while claiming cross-device equivalence.

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
}
