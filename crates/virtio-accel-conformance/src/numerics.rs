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

/// One immutable bfloat16 tensor represented by exact storage bits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Bfloat16Tensor {
    /// Static row-major shape.
    pub shape: &'static [usize],
    /// Row-major IEEE-754 bfloat16 element bits.
    pub bits: &'static [u16],
}

/// A stable TOSA EXT-BF16 graph and bit-exact bfloat16 numerical oracle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TosaBfloat16Case {
    /// Diagnostic case name.
    pub name: &'static str,
    /// TOSA 1.0 FlatBuffer payload.
    pub artifact: &'static [u8],
    /// Block inputs in declared slot order.
    pub inputs: &'static [Bfloat16Tensor],
    /// Block outputs in declared slot order.
    pub outputs: &'static [Bfloat16Tensor],
}

impl TosaBfloat16Case {
    /// Compare one backend output bit-for-bit, allowing only NaN payload canonicalization.
    pub fn output_matches(&self, output: usize, actual: &[u16]) -> bool {
        let Some(expected) = self.outputs.get(output).map(|tensor| tensor.bits) else {
            return false;
        };
        expected.len() == actual.len()
            && expected.iter().zip(actual).all(|(expected, actual)| {
                if is_bfloat16_nan(*expected) {
                    is_bfloat16_nan(*actual)
                } else {
                    expected == actual
                }
            })
    }
}

/// Raw tensor storage used by mixed-type Hexagon operator-parity cases.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RawTensor {
    /// IEEE-754 binary16 elements represented by their exact bits.
    Fp16(&'static [u16]),
    /// TOSA BOOL storage, one zero-or-one byte per element.
    Bool(&'static [u8]),
    /// Signed INT32 elements.
    Int32(&'static [i32]),
}

impl RawTensor {
    /// Encode the tensor in the client-visible little-endian storage layout.
    pub fn bytes(self) -> Vec<u8> {
        match self {
            Self::Fp16(values) => values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect(),
            Self::Bool(values) => values.to_vec(),
            Self::Int32(values) => values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect(),
        }
    }

    /// Exact client-visible storage size.
    pub fn byte_len(self) -> usize {
        match self {
            Self::Fp16(values) => values.len() * 2,
            Self::Bool(values) => values.len(),
            Self::Int32(values) => values.len() * 4,
        }
    }
}

/// One mixed-type TOSA operator case with a numerical output oracle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TosaRawCase {
    /// Diagnostic case name.
    pub name: &'static str,
    /// TOSA 1.0 FlatBuffer payload.
    pub artifact: &'static [u8],
    /// Block inputs in declared slot order.
    pub inputs: &'static [RawTensor],
    /// Single block output.
    pub output: RawTensor,
    /// Maximum accepted binary16 ULP distance; ignored for exact BOOL/INT32 outputs.
    pub fp16_max_ulps: u16,
}

impl TosaRawCase {
    /// Compare raw client-visible bytes with the typed numerical oracle.
    pub fn output_matches(self, actual: &[u8]) -> bool {
        if actual.len() != self.output.byte_len() {
            return false;
        }
        match self.output {
            RawTensor::Bool(expected) => actual == expected,
            RawTensor::Int32(expected) => actual
                .chunks_exact(4)
                .map(|bytes| i32::from_le_bytes(bytes.try_into().expect("four-byte chunk")))
                .eq(expected.iter().copied()),
            RawTensor::Fp16(expected) => {
                actual
                    .chunks_exact(2)
                    .zip(expected)
                    .all(|(bytes, expected)| {
                        let actual = u16::from_le_bytes([bytes[0], bytes[1]]);
                        if is_binary16_nan(*expected) {
                            is_binary16_nan(actual)
                        } else if self.fp16_max_ulps == 0
                            || (*expected & 0x8000) != (actual & 0x8000)
                            || (*expected & 0x7fff) == 0
                        {
                            actual == *expected
                        } else {
                            actual.abs_diff(*expected) <= self.fp16_max_ulps
                        }
                    })
            }
        }
    }
}

const fn is_binary16_nan(bits: u16) -> bool {
    bits & 0x7c00 == 0x7c00 && bits & 0x03ff != 0
}

const fn is_bfloat16_nan(bits: u16) -> bool {
    bits & 0x7f80 == 0x7f80 && bits & 0x007f != 0
}

/// Packed scalar encoding used by a low-precision TOSA acceptance case.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PackedDType {
    /// Signed two's-complement values packed low nibble first.
    Int4,
    /// Signed two's-complement bytes.
    Int8,
    /// TOSA FP8 E4M3 bytes.
    Fp8E4M3,
    /// TOSA FP8 E5M2 bytes.
    Fp8E5M2,
}

impl PackedDType {
    /// Number of bytes needed for `elements` densely packed values.
    pub const fn storage_bytes(self, elements: usize) -> Option<usize> {
        match self {
            Self::Int4 => Some(elements / 2 + elements % 2),
            Self::Int8 | Self::Fp8E4M3 | Self::Fp8E5M2 => Some(elements),
        }
    }
}

/// One immutable packed low-precision tensor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PackedTensor {
    /// Static row-major logical shape.
    pub shape: &'static [usize],
    /// Densely packed elements in TOSA byte order.
    pub bytes: &'static [u8],
}

/// A stable explicit FP8 → BF16 CAST with a bit-exact output oracle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TosaFp8ToBfloat16Case {
    /// Diagnostic case name.
    pub name: &'static str,
    /// Source FP8 storage encoding.
    pub input_dtype: PackedDType,
    /// TOSA 1.0 FlatBuffer payload.
    pub artifact: &'static [u8],
    /// Single graph-visible FP8 input.
    pub input: PackedTensor,
    /// Single graph-visible BF16 output.
    pub output: Bfloat16Tensor,
}

impl TosaFp8ToBfloat16Case {
    /// Compare BF16 output bits exactly, allowing only NaN payload canonicalization.
    pub fn output_matches(self, actual: &[u16]) -> bool {
        self.output.bits.len() == actual.len()
            && self
                .output
                .bits
                .iter()
                .zip(actual)
                .all(|(expected, actual)| {
                    if is_bfloat16_nan(*expected) {
                        is_bfloat16_nan(*actual)
                    } else {
                        expected == actual
                    }
                })
    }
}

/// One immutable INT32 tensor produced by an integer-profile acceptance case.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Int32Tensor {
    /// Static row-major shape.
    pub shape: &'static [usize],
    /// Exact row-major tensor elements.
    pub values: &'static [i32],
}

/// A stable TOSA INT8 matrix multiplication with exact INT32 accumulation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TosaInt8MatmulCase {
    /// Diagnostic case name.
    pub name: &'static str,
    /// TOSA 1.0 FlatBuffer payload.
    pub artifact: &'static [u8],
    /// Signed INT8 block inputs in declared slot order.
    pub inputs: &'static [PackedTensor],
    /// Compile-time zero points for the left and right operands.
    pub zero_points: [i8; 2],
    /// Exact INT32 block outputs in declared slot order.
    pub outputs: &'static [Int32Tensor],
}

impl TosaInt8MatmulCase {
    /// Compare one backend output with the exact INT32 oracle.
    pub fn output_matches(&self, output: usize, actual: &[i32]) -> bool {
        self.outputs
            .get(output)
            .is_some_and(|expected| expected.values == actual)
    }
}

/// A stable TOSA graph and packed low-precision oracle shared by host backends.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TosaPackedCase {
    /// Diagnostic case name.
    pub name: &'static str,
    /// Tensor scalar encoding.
    pub dtype: PackedDType,
    /// TOSA 1.0 FlatBuffer payload.
    pub artifact: &'static [u8],
    /// Block inputs in declared slot order.
    pub inputs: &'static [PackedTensor],
    /// Block outputs in declared slot order.
    pub outputs: &'static [PackedTensor],
}

impl TosaPackedCase {
    /// Compare one backend output with this case's selected storage oracle.
    ///
    /// Integer values must match bit-for-bit. FP8 values do too, except that NaN sign and payload
    /// may be canonicalized by an accelerator.
    pub fn output_matches(&self, output: usize, actual: &[u8]) -> bool {
        let Some(expected) = self.outputs.get(output).map(|tensor| tensor.bytes) else {
            return false;
        };
        expected.len() == actual.len()
            && expected.iter().zip(actual).all(|(expected, actual)| {
                if packed_is_nan(self.dtype, *expected) {
                    packed_is_nan(self.dtype, *actual)
                } else {
                    expected == actual
                }
            })
    }
}

const fn packed_is_nan(dtype: PackedDType, bits: u8) -> bool {
    match dtype {
        PackedDType::Fp8E4M3 => bits & 0x7f == 0x7f,
        PackedDType::Fp8E5M2 => bits & 0x7c == 0x7c && bits & 0x03 != 0,
        PackedDType::Int4 | PackedDType::Int8 => false,
    }
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

const MAX_POOL2D_INPUT_BF16_BITS: &[u16] = &[
    0x3f80, 0x42ca, 0x4000, 0x42cc, 0x4040, 0x42ce, 0x4080, 0x42d0, 0x40a0, 0x42d2, 0x40c0, 0x42d4,
    0x40e0, 0x42d6, 0x4100, 0x42d8, 0x4110, 0x42da, 0x4120, 0x42dc, 0x4130, 0x42de, 0x4140, 0x42e0,
    0x4150, 0x42e2, 0x4160, 0x42e4, 0x4170, 0x42e6, 0x4180, 0x42e8,
];
const MAX_POOL2D_OUTPUT_BF16_BITS: &[u16] = &[
    0x40c0, 0x42d4, 0x4100, 0x42d8, 0x4160, 0x42e4, 0x4180, 0x42e8,
];
const MAX_POOL2D_INPUTS_BF16: &[Bfloat16Tensor] = &[Bfloat16Tensor {
    shape: &[1, 4, 4, 2],
    bits: MAX_POOL2D_INPUT_BF16_BITS,
}];
const MAX_POOL2D_OUTPUTS_BF16: &[Bfloat16Tensor] = &[Bfloat16Tensor {
    shape: &[1, 2, 2, 2],
    bits: MAX_POOL2D_OUTPUT_BF16_BITS,
}];

/// BF16 two-channel NHWC max pooling with a 2x2 kernel, stride two, zero padding, and an exact
/// integer-valued oracle.
pub const MAX_POOL2D_BF16: TosaBfloat16Case = TosaBfloat16Case {
    name: "max-pool2d-bf16",
    artifact: include_bytes!("data/max-pool2d-bf16-v1.0.0.tosa"),
    inputs: MAX_POOL2D_INPUTS_BF16,
    outputs: MAX_POOL2D_OUTPUTS_BF16,
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

const BINARY_LEFT_FP16_BITS: &[u16] = &[0x4000, 0x4400];
const BINARY_RIGHT_FP16_BITS: &[u16] = &[0x3c00, 0x4000, 0x4200];
const BINARY_INPUTS_FP16: &[Float16Tensor] = &[
    Float16Tensor {
        shape: &[2, 1],
        bits: BINARY_LEFT_FP16_BITS,
    },
    Float16Tensor {
        shape: &[1, 3],
        bits: BINARY_RIGHT_FP16_BITS,
    },
];
const ADD_OUTPUT_FP16_BITS: &[u16] = &[0x4200, 0x4400, 0x4500, 0x4500, 0x4600, 0x4700];
const SUB_OUTPUT_FP16_BITS: &[u16] = &[0x3c00, 0x0000, 0xbc00, 0x4200, 0x4000, 0x3c00];
const MUL_OUTPUT_FP16_BITS: &[u16] = &[0x4000, 0x4400, 0x4600, 0x4400, 0x4800, 0x4a00];
const POW_OUTPUT_FP16_BITS: &[u16] = &[0x4000, 0x4400, 0x4800, 0x4400, 0x4c00, 0x5400];
const MAXIMUM_OUTPUT_FP16_BITS: &[u16] = &[0x4000, 0x4000, 0x4200, 0x4400, 0x4400, 0x4400];
const MINIMUM_OUTPUT_FP16_BITS: &[u16] = &[0x3c00, 0x4000, 0x4000, 0x3c00, 0x4000, 0x4200];

const fn binary_output(bits: &'static [u16]) -> [Float16Tensor; 1] {
    [Float16Tensor {
        shape: &[2, 3],
        bits,
    }]
}

/// Binary16 broadcast addition over `[2, 1]` and `[1, 3]` inputs.
pub const ADD_FP16: TosaFloat16Case = TosaFloat16Case {
    name: "add-fp16",
    artifact: include_bytes!("data/add-fp16-v1.0.0.tosa"),
    inputs: BINARY_INPUTS_FP16,
    outputs: &binary_output(ADD_OUTPUT_FP16_BITS),
};

/// Binary16 broadcast subtraction over `[2, 1]` and `[1, 3]` inputs.
pub const SUB_FP16: TosaFloat16Case = TosaFloat16Case {
    name: "sub-fp16",
    artifact: include_bytes!("data/sub-fp16-v1.0.0.tosa"),
    inputs: BINARY_INPUTS_FP16,
    outputs: &binary_output(SUB_OUTPUT_FP16_BITS),
};

/// Binary16 broadcast multiplication with a compile-time zero shift.
pub const MUL_FP16: TosaFloat16Case = TosaFloat16Case {
    name: "mul-fp16",
    artifact: include_bytes!("data/mul-fp16-v1.0.0.tosa"),
    inputs: BINARY_INPUTS_FP16,
    outputs: &binary_output(MUL_OUTPUT_FP16_BITS),
};

/// Binary16 broadcast power over `[2, 1]` and `[1, 3]` inputs.
pub const POW_FP16: TosaFloat16Case = TosaFloat16Case {
    name: "pow-fp16",
    artifact: include_bytes!("data/pow-fp16-v1.0.0.tosa"),
    inputs: BINARY_INPUTS_FP16,
    outputs: &binary_output(POW_OUTPUT_FP16_BITS),
};

/// Binary16 broadcast maximum over `[2, 1]` and `[1, 3]` inputs.
pub const MAXIMUM_FP16: TosaFloat16Case = TosaFloat16Case {
    name: "maximum-fp16",
    artifact: include_bytes!("data/maximum-fp16-v1.0.0.tosa"),
    inputs: BINARY_INPUTS_FP16,
    outputs: &binary_output(MAXIMUM_OUTPUT_FP16_BITS),
};

/// Binary16 broadcast minimum over `[2, 1]` and `[1, 3]` inputs.
pub const MINIMUM_FP16: TosaFloat16Case = TosaFloat16Case {
    name: "minimum-fp16",
    artifact: include_bytes!("data/minimum-fp16-v1.0.0.tosa"),
    inputs: BINARY_INPUTS_FP16,
    outputs: &binary_output(MINIMUM_OUTPUT_FP16_BITS),
};

const UNARY_INPUTS_RAW: &[RawTensor] = &[RawTensor::Fp16(&[0x3800, 0x3c00, 0x4000, 0x4400])];

const fn unary_raw_case(
    name: &'static str,
    artifact: &'static [u8],
    output: &'static [u16],
    fp16_max_ulps: u16,
) -> TosaRawCase {
    TosaRawCase {
        name,
        artifact,
        inputs: UNARY_INPUTS_RAW,
        output: RawTensor::Fp16(output),
        fp16_max_ulps,
    }
}

/// FP16 unary and activation cases supported by the QNN HTP operator package.
pub const HEXAGON_UNARY_FP16_CASES: &[TosaRawCase] = &[
    unary_raw_case(
        "abs-fp16",
        include_bytes!("data/abs-fp16-v1.0.0.tosa"),
        &[0x3800, 0x3c00, 0x4000, 0x4400],
        0,
    ),
    unary_raw_case(
        "ceil-fp16",
        include_bytes!("data/ceil-fp16-v1.0.0.tosa"),
        &[0x3c00, 0x3c00, 0x4000, 0x4400],
        0,
    ),
    unary_raw_case(
        "cos-fp16",
        include_bytes!("data/cos-fp16-v1.0.0.tosa"),
        &[0x3b05, 0x3853, 0xb6a9, 0xb93b],
        8,
    ),
    unary_raw_case(
        "exp-fp16",
        include_bytes!("data/exp-fp16-v1.0.0.tosa"),
        &[0x3e98, 0x4170, 0x4764, 0x52d3],
        4,
    ),
    unary_raw_case(
        "floor-fp16",
        include_bytes!("data/floor-fp16-v1.0.0.tosa"),
        &[0x0000, 0x3c00, 0x4000, 0x4400],
        0,
    ),
    unary_raw_case(
        "log-fp16",
        include_bytes!("data/log-fp16-v1.0.0.tosa"),
        &[0xb98c, 0x0000, 0x398c, 0x3d8c],
        4,
    ),
    unary_raw_case(
        "negate-fp16",
        include_bytes!("data/negate-fp16-v1.0.0.tosa"),
        &[0xb800, 0xbc00, 0xc000, 0xc400],
        0,
    ),
    unary_raw_case(
        "reciprocal-fp16",
        include_bytes!("data/reciprocal-fp16-v1.0.0.tosa"),
        &[0x4000, 0x3c00, 0x3800, 0x3400],
        2,
    ),
    unary_raw_case(
        "rsqrt-fp16",
        include_bytes!("data/rsqrt-fp16-v1.0.0.tosa"),
        &[0x3da8, 0x3c00, 0x39a8, 0x3800],
        4,
    ),
    unary_raw_case(
        "sin-fp16",
        include_bytes!("data/sin-fp16-v1.0.0.tosa"),
        &[0x37ac, 0x3abb, 0x3b46, 0xba0e],
        8,
    ),
    unary_raw_case(
        "sigmoid-fp16",
        include_bytes!("data/sigmoid-fp16-v1.0.0.tosa"),
        &[0x38fb, 0x39d9, 0x3b0c, 0x3bdb],
        4,
    ),
    unary_raw_case(
        "tanh-fp16",
        include_bytes!("data/tanh-fp16-v1.0.0.tosa"),
        &[0x3765, 0x3a18, 0x3bb6, 0x3bff],
        4,
    ),
    unary_raw_case(
        "clamp-fp16",
        include_bytes!("data/clamp-fp16-v1.0.0.tosa"),
        &[0x3800, 0x3c00, 0x3c00, 0x3c00],
        0,
    ),
];

const COMPARISON_INPUTS_RAW: &[RawTensor] = &[
    RawTensor::Fp16(&[0x3c00, 0x4000, 0x4200, 0x4400]),
    RawTensor::Fp16(&[0x3c00, 0x4200, 0x4000, 0x4400]),
];
const LOGICAL_INPUTS_RAW: &[RawTensor] = &[
    RawTensor::Bool(&[0, 0, 1, 1]),
    RawTensor::Bool(&[0, 1, 0, 1]),
];
const LOGICAL_NOT_INPUT_RAW: &[RawTensor] = &[RawTensor::Bool(&[0, 0, 1, 1])];
const SELECT_INPUTS_RAW: &[RawTensor] = &[
    RawTensor::Bool(&[0, 1, 0, 1]),
    RawTensor::Fp16(&[0x3c00, 0x4000, 0x4200, 0x4400]),
    RawTensor::Fp16(&[0x4500, 0x4600, 0x4700, 0x4800]),
];

/// Mixed BOOL/FP16 comparison, logical, and selection cases.
pub const HEXAGON_LOGICAL_CASES: &[TosaRawCase] = &[
    TosaRawCase {
        name: "equal-fp16",
        artifact: include_bytes!("data/equal-fp16-v1.0.0.tosa"),
        inputs: COMPARISON_INPUTS_RAW,
        output: RawTensor::Bool(&[1, 0, 0, 1]),
        fp16_max_ulps: 0,
    },
    TosaRawCase {
        name: "greater-fp16",
        artifact: include_bytes!("data/greater-fp16-v1.0.0.tosa"),
        inputs: COMPARISON_INPUTS_RAW,
        output: RawTensor::Bool(&[0, 0, 1, 0]),
        fp16_max_ulps: 0,
    },
    TosaRawCase {
        name: "greater-equal-fp16",
        artifact: include_bytes!("data/greater-equal-fp16-v1.0.0.tosa"),
        inputs: COMPARISON_INPUTS_RAW,
        output: RawTensor::Bool(&[1, 0, 1, 1]),
        fp16_max_ulps: 0,
    },
    TosaRawCase {
        name: "logical-and",
        artifact: include_bytes!("data/logical-and-fp16-v1.0.0.tosa"),
        inputs: LOGICAL_INPUTS_RAW,
        output: RawTensor::Bool(&[0, 0, 0, 1]),
        fp16_max_ulps: 0,
    },
    TosaRawCase {
        name: "logical-or",
        artifact: include_bytes!("data/logical-or-fp16-v1.0.0.tosa"),
        inputs: LOGICAL_INPUTS_RAW,
        output: RawTensor::Bool(&[0, 1, 1, 1]),
        fp16_max_ulps: 0,
    },
    TosaRawCase {
        name: "logical-xor",
        artifact: include_bytes!("data/logical-xor-fp16-v1.0.0.tosa"),
        inputs: LOGICAL_INPUTS_RAW,
        output: RawTensor::Bool(&[0, 1, 1, 0]),
        fp16_max_ulps: 0,
    },
    TosaRawCase {
        name: "logical-not",
        artifact: include_bytes!("data/logical-not-fp16-v1.0.0.tosa"),
        inputs: LOGICAL_NOT_INPUT_RAW,
        output: RawTensor::Bool(&[1, 1, 0, 0]),
        fp16_max_ulps: 0,
    },
    TosaRawCase {
        name: "select-fp16",
        artifact: include_bytes!("data/select-fp16-v1.0.0.tosa"),
        inputs: SELECT_INPUTS_RAW,
        output: RawTensor::Fp16(&[0x4500, 0x4000, 0x4700, 0x4400]),
        fp16_max_ulps: 0,
    },
];

const REDUCTION_INPUTS_RAW: &[RawTensor] = &[RawTensor::Fp16(&[
    0x3c00, 0x4200, 0x4000, 0xbc00, 0x4400, 0x4000,
])];

/// FP16 reductions and INT32 argmax over a two-row input.
pub const HEXAGON_REDUCTION_CASES: &[TosaRawCase] = &[
    TosaRawCase {
        name: "argmax-fp16",
        artifact: include_bytes!("data/argmax-fp16-v1.0.0.tosa"),
        inputs: REDUCTION_INPUTS_RAW,
        output: RawTensor::Int32(&[1, 1]),
        fp16_max_ulps: 0,
    },
    TosaRawCase {
        name: "reduce-max-fp16",
        artifact: include_bytes!("data/reduce-max-fp16-v1.0.0.tosa"),
        inputs: REDUCTION_INPUTS_RAW,
        output: RawTensor::Fp16(&[0x4200, 0x4400]),
        fp16_max_ulps: 0,
    },
    TosaRawCase {
        name: "reduce-min-fp16",
        artifact: include_bytes!("data/reduce-min-fp16-v1.0.0.tosa"),
        inputs: REDUCTION_INPUTS_RAW,
        output: RawTensor::Fp16(&[0x3c00, 0xbc00]),
        fp16_max_ulps: 0,
    },
    TosaRawCase {
        name: "reduce-product-fp16",
        artifact: include_bytes!("data/reduce-product-fp16-v1.0.0.tosa"),
        inputs: REDUCTION_INPUTS_RAW,
        output: RawTensor::Fp16(&[0x4600, 0xc800]),
        fp16_max_ulps: 1,
    },
    TosaRawCase {
        name: "reduce-sum-fp16",
        artifact: include_bytes!("data/reduce-sum-fp16-v1.0.0.tosa"),
        inputs: REDUCTION_INPUTS_RAW,
        output: RawTensor::Fp16(&[0x4600, 0x4500]),
        fp16_max_ulps: 1,
    },
];

/// Static constants and FP16 data-movement cases.
pub const HEXAGON_MOVEMENT_CASES: &[TosaRawCase] = &[
    TosaRawCase {
        name: "const-add-fp16",
        artifact: include_bytes!("data/const-fp16-v1.0.0.tosa"),
        inputs: &[RawTensor::Fp16(&[0x4900, 0x4d00, 0x4f80, 0x5100])],
        output: RawTensor::Fp16(&[0x4980, 0x4d80, 0x5020, 0x5180]),
        fp16_max_ulps: 0,
    },
    TosaRawCase {
        name: "reshape-const-shape-fp16",
        artifact: include_bytes!("data/reshape-fp16-v1.0.0.tosa"),
        inputs: &[RawTensor::Fp16(&[0x3c00, 0x4000, 0x4200, 0x4400])],
        output: RawTensor::Fp16(&[0x3c00, 0x4000, 0x4200, 0x4400]),
        fp16_max_ulps: 0,
    },
    TosaRawCase {
        name: "transpose-fp16",
        artifact: include_bytes!("data/transpose-fp16-v1.0.0.tosa"),
        inputs: &[RawTensor::Fp16(&[
            0x3c00, 0x4000, 0x4200, 0x4400, 0x4500, 0x4600,
        ])],
        output: RawTensor::Fp16(&[0x3c00, 0x4400, 0x4000, 0x4500, 0x4200, 0x4600]),
        fp16_max_ulps: 0,
    },
    TosaRawCase {
        name: "reverse-fp16",
        artifact: include_bytes!("data/reverse-fp16-v1.0.0.tosa"),
        inputs: &[RawTensor::Fp16(&[
            0x3c00, 0x4000, 0x4200, 0x4400, 0x4500, 0x4600,
        ])],
        output: RawTensor::Fp16(&[0x4200, 0x4000, 0x3c00, 0x4600, 0x4500, 0x4400]),
        fp16_max_ulps: 0,
    },
    TosaRawCase {
        name: "concat-fp16",
        artifact: include_bytes!("data/concat-fp16-v1.0.0.tosa"),
        inputs: &[
            RawTensor::Fp16(&[0x3c00, 0x4000]),
            RawTensor::Fp16(&[0x4200, 0x4400]),
        ],
        output: RawTensor::Fp16(&[0x3c00, 0x4200, 0x4000, 0x4400]),
        fp16_max_ulps: 0,
    },
];

const MOCK_CLASSIFIER_FEATURES_FP16_BITS: &[u16] = &[
    0x3c00, 0x4000, 0x4200, // [1.0, 2.0, 3.0]
    0xbc00, 0x3800, 0x4000, // [-1.0, 0.5, 2.0]
];
const MOCK_CLASSIFIER_WEIGHTS_FP16_BITS: &[u16] = &[
    0x3c00, 0x0000, // feature 0 -> [1.0, 0.0]
    0x0000, 0x3c00, // feature 1 -> [0.0, 1.0]
    0x3c00, 0xbc00, // feature 2 -> [1.0, -1.0]
];
const MOCK_CLASSIFIER_LOGITS_FP16_BITS: &[u16] = &[
    0x4400, 0xbc00, // [4.0, -1.0]
    0x3c00, 0xbe00, // [1.0, -1.5]
];
const MOCK_CLASSIFIER_INPUTS_FP16: &[Float16Tensor] = &[
    Float16Tensor {
        shape: &[1, 2, 3],
        bits: MOCK_CLASSIFIER_FEATURES_FP16_BITS,
    },
    Float16Tensor {
        shape: &[1, 3, 2],
        bits: MOCK_CLASSIFIER_WEIGHTS_FP16_BITS,
    },
];
const MOCK_CLASSIFIER_OUTPUTS_FP16: &[Float16Tensor] = &[Float16Tensor {
    shape: &[1, 2, 2],
    bits: MOCK_CLASSIFIER_LOGITS_FP16_BITS,
}];

/// Two-sample, three-feature FP16 linear classifier with a direct-bound 3x2 weight matrix.
pub const MOCK_LINEAR_CLASSIFIER_FP16: TosaFloat16Case = TosaFloat16Case {
    name: "mock-linear-classifier-fp16",
    artifact: MATMUL_FP16.artifact,
    inputs: MOCK_CLASSIFIER_INPUTS_FP16,
    outputs: MOCK_CLASSIFIER_OUTPUTS_FP16,
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

const IDENTITY_INT8_BYTES: &[u8] = &[0x80, 0x81, 0xff, 0x00, 0x01, 0x7e, 0x7f, 0x2a];
const IDENTITY_INT8_TENSORS: &[PackedTensor] = &[PackedTensor {
    shape: &[8],
    bytes: IDENTITY_INT8_BYTES,
}];

/// INT8 identity spanning negative, zero, and positive values.
pub const IDENTITY_INT8: TosaPackedCase = TosaPackedCase {
    name: "identity-int8",
    dtype: PackedDType::Int8,
    artifact: include_bytes!("data/identity-int8-v1.0.0.tosa"),
    inputs: IDENTITY_INT8_TENSORS,
    outputs: IDENTITY_INT8_TENSORS,
};

const MATMUL_INT8_LHS: &[u8] = &[0x80, 0xff, 0x7f, 0x05, 0xfa, 0x07];
const MATMUL_INT8_RHS: &[u8] = &[0x08, 0xf7, 0x0a, 0x0b, 0x0c, 0xf3];
const MATMUL_INT8_INPUTS: &[PackedTensor] = &[
    PackedTensor {
        shape: &[1, 2, 3],
        bytes: MATMUL_INT8_LHS,
    },
    PackedTensor {
        shape: &[1, 3, 2],
        bytes: MATMUL_INT8_RHS,
    },
];
const MATMUL_INT8_OUTPUT: &[i32] = &[538, -544, 88, -260];
const MATMUL_INT8_OUTPUTS: &[Int32Tensor] = &[Int32Tensor {
    shape: &[1, 2, 2],
    values: MATMUL_INT8_OUTPUT,
}];

/// Non-square INT8 batched matrix multiplication with nonzero zero points and INT32 accumulation.
pub const MATMUL_INT8: TosaInt8MatmulCase = TosaInt8MatmulCase {
    name: "matmul-int8",
    artifact: include_bytes!("data/matmul-int8-v1.0.0.tosa"),
    inputs: MATMUL_INT8_INPUTS,
    zero_points: [-2, 3],
    outputs: MATMUL_INT8_OUTPUTS,
};

// Logical values [-7, -3, -1, 0, 1, 3, 6, 7], packed low nibble first.
const IDENTITY_INT4_BYTES: &[u8] = &[0xd9, 0x0f, 0x31, 0x76];
const IDENTITY_INT4_TENSORS: &[PackedTensor] = &[PackedTensor {
    shape: &[8],
    bytes: IDENTITY_INT4_BYTES,
}];

/// Packed INT4 identity spanning the TOSA-defined finite range.
pub const IDENTITY_INT4: TosaPackedCase = TosaPackedCase {
    name: "identity-int4",
    dtype: PackedDType::Int4,
    artifact: include_bytes!("data/identity-int4-v1.0.0.tosa"),
    inputs: IDENTITY_INT4_TENSORS,
    outputs: IDENTITY_INT4_TENSORS,
};

const IDENTITY_FP8E4M3_BYTES: &[u8] = &[0x00, 0x80, 0x01, 0x81, 0x38, 0xb8, 0x7e, 0x7f];
const IDENTITY_FP8E4M3_TENSORS: &[PackedTensor] = &[PackedTensor {
    shape: &[8],
    bytes: IDENTITY_FP8E4M3_BYTES,
}];

/// FP8 E4M3 identity over signed zeros, subnormals, ordinary values, finite maximum, and NaN.
pub const IDENTITY_FP8E4M3: TosaPackedCase = TosaPackedCase {
    name: "identity-fp8e4m3",
    dtype: PackedDType::Fp8E4M3,
    artifact: include_bytes!("data/identity-fp8e4m3-v1.0.0.tosa"),
    inputs: IDENTITY_FP8E4M3_TENSORS,
    outputs: IDENTITY_FP8E4M3_TENSORS,
};

const IDENTITY_FP8E5M2_BYTES: &[u8] = &[0x00, 0x80, 0x01, 0x81, 0x3c, 0x7b, 0x7c, 0x7d];
const IDENTITY_FP8E5M2_TENSORS: &[PackedTensor] = &[PackedTensor {
    shape: &[8],
    bytes: IDENTITY_FP8E5M2_BYTES,
}];

/// FP8 E5M2 identity over signed zeros, subnormals, one, finite maximum, infinity, and NaN.
pub const IDENTITY_FP8E5M2: TosaPackedCase = TosaPackedCase {
    name: "identity-fp8e5m2",
    dtype: PackedDType::Fp8E5M2,
    artifact: include_bytes!("data/identity-fp8e5m2-v1.0.0.tosa"),
    inputs: IDENTITY_FP8E5M2_TENSORS,
    outputs: IDENTITY_FP8E5M2_TENSORS,
};

const fn all_fp8_encodings() -> [u8; 1024] {
    let mut values = [0u8; 1024];
    let mut index = 0;
    while index < values.len() {
        values[index] = index as u8;
        index += 1;
    }
    values
}

const fn fp8e4m3_bf16_oracle() -> [u16; 1024] {
    let mut values = [0u16; 1024];
    let mut index = 0;
    while index < values.len() {
        let bits = index as u8;
        let sign = ((bits & 0x80) as u16) << 8;
        let exponent = ((bits >> 3) & 0x0f) as u16;
        let fraction = (bits & 0x07) as u16;
        values[index] = if exponent == 0 {
            let subnormal = [
                0x0000, 0x3b00, 0x3b80, 0x3bc0, 0x3c00, 0x3c20, 0x3c40, 0x3c60,
            ];
            sign | subnormal[fraction as usize]
        } else if exponent == 0x0f && fraction == 0x07 {
            sign | 0x7fc0
        } else {
            sign | ((exponent + 120) << 7) | (fraction << 4)
        };
        index += 1;
    }
    values
}

const fn fp8e5m2_bf16_oracle() -> [u16; 1024] {
    let mut values = [0u16; 1024];
    let mut index = 0;
    while index < values.len() {
        let bits = index as u8;
        let sign = ((bits & 0x80) as u16) << 8;
        let exponent = ((bits >> 2) & 0x1f) as u16;
        let fraction = (bits & 0x03) as u16;
        values[index] = if exponent == 0 {
            let subnormal = [0x0000, 0x3780, 0x3800, 0x3840];
            sign | subnormal[fraction as usize]
        } else if exponent == 0x1f {
            sign | if fraction == 0 { 0x7f80 } else { 0x7fc0 }
        } else {
            sign | ((exponent + 112) << 7) | (fraction << 5)
        };
        index += 1;
    }
    values
}

const ALL_FP8_ENCODINGS: [u8; 1024] = all_fp8_encodings();
const CAST_FP8E4M3_OUTPUT: [u16; 1024] = fp8e4m3_bf16_oracle();

/// Explicit FP8 E4M3 → BF16 CAST spanning signed zeros, subnormals, ordinary values, finite
/// maximum, and NaN. All 256 byte encodings repeat four times across one XDNA conversion tile.
pub const CAST_FP8E4M3_TO_BF16: TosaFp8ToBfloat16Case = TosaFp8ToBfloat16Case {
    name: "cast-fp8e4m3-to-bf16",
    input_dtype: PackedDType::Fp8E4M3,
    artifact: include_bytes!("data/cast-fp8e4m3-to-bf16-v1.0.0.tosa"),
    input: PackedTensor {
        shape: &[1024],
        bytes: &ALL_FP8_ENCODINGS,
    },
    output: Bfloat16Tensor {
        shape: &[1024],
        bits: &CAST_FP8E4M3_OUTPUT,
    },
};

const CAST_FP8E5M2_OUTPUT: [u16; 1024] = fp8e5m2_bf16_oracle();

/// Explicit FP8 E5M2 → BF16 CAST spanning signed zeros, subnormals, one, finite maximum,
/// infinity, and NaN. All 256 byte encodings repeat four times across one XDNA conversion tile.
pub const CAST_FP8E5M2_TO_BF16: TosaFp8ToBfloat16Case = TosaFp8ToBfloat16Case {
    name: "cast-fp8e5m2-to-bf16",
    input_dtype: PackedDType::Fp8E5M2,
    artifact: include_bytes!("data/cast-fp8e5m2-to-bf16-v1.0.0.tosa"),
    input: PackedTensor {
        shape: &[1024],
        bytes: &ALL_FP8_ENCODINGS,
    },
    output: Bfloat16Tensor {
        shape: &[1024],
        bits: &CAST_FP8E5M2_OUTPUT,
    },
};

#[cfg(test)]
mod tests {
    use super::*;
    use virtio_accel_tosa::{
        DType, ExtensionSet, Level, ProfileSet, Target, Version, low_precision_storage_bytes, parse,
    };

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
    fn bf16_max_pool_oracle_is_exact_and_nhwc_ordered() {
        assert!(MAX_POOL2D_BF16.output_matches(0, MAX_POOL2D_OUTPUT_BF16_BITS));
        assert!(!MAX_POOL2D_BF16.output_matches(
            0,
            &[
                0x40c0, 0x4100, 0x4160, 0x4180, 0x42d4, 0x42d8, 0x42e4, 0x42e8
            ]
        ));
        parse(MAX_POOL2D_BF16.artifact)
            .unwrap()
            .validate_for(Target::new(
                Version::TOSA_1_0,
                ProfileSet::FLOATING_POINT,
                Level::Level8K,
                ExtensionSet::BF16,
            ))
            .unwrap();
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
        assert!(MOCK_LINEAR_CLASSIFIER_FP16.output_matches(0, MOCK_CLASSIFIER_LOGITS_FP16_BITS));
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

    #[test]
    fn packed_oracles_preserve_storage_and_int4_layout() {
        for case in [
            IDENTITY_INT4,
            IDENTITY_INT8,
            IDENTITY_FP8E4M3,
            IDENTITY_FP8E5M2,
        ] {
            let tensor = case.inputs[0];
            let elements = tensor.shape.iter().product();
            assert_eq!(case.dtype.storage_bytes(elements), Some(tensor.bytes.len()));
            assert!(case.output_matches(0, tensor.bytes));
            assert!(!case.output_matches(1, tensor.bytes));
        }
        assert_eq!(IDENTITY_INT4.inputs[0].bytes, &[0xd9, 0x0f, 0x31, 0x76]);

        let mut canonicalized_e4m3_nan = IDENTITY_FP8E4M3.outputs[0].bytes.to_vec();
        canonicalized_e4m3_nan[7] = 0xff;
        assert!(IDENTITY_FP8E4M3.output_matches(0, &canonicalized_e4m3_nan));
        let mut canonicalized_e5m2_nan = IDENTITY_FP8E5M2.outputs[0].bytes.to_vec();
        canonicalized_e5m2_nan[7] = 0x7f;
        assert!(IDENTITY_FP8E5M2.output_matches(0, &canonicalized_e5m2_nan));
    }

    #[test]
    fn fp8_to_bf16_cast_oracles_are_derived_from_the_shared_exact_decoders() {
        use virtio_accel_tosa::{
            fp8e4m3_to_bf16_bits, fp8e4m3_to_f32, fp8e5m2_to_bf16_bits, fp8e5m2_to_f32,
        };

        for case in [CAST_FP8E4M3_TO_BF16, CAST_FP8E5M2_TO_BF16] {
            assert_eq!(case.input.bytes.len(), case.output.bits.len());
            for (input, expected) in case.input.bytes.iter().zip(case.output.bits) {
                let value = match case.input_dtype {
                    PackedDType::Fp8E4M3 => fp8e4m3_to_f32(*input),
                    PackedDType::Fp8E5M2 => fp8e5m2_to_f32(*input),
                    _ => unreachable!("CAST case must use FP8"),
                };
                if value.is_nan() {
                    assert!(is_bfloat16_nan(*expected));
                } else {
                    assert_eq!(*expected, (value.to_bits() >> 16) as u16);
                }
                let shared_bits = match case.input_dtype {
                    PackedDType::Fp8E4M3 => fp8e4m3_to_bf16_bits(*input),
                    PackedDType::Fp8E5M2 => fp8e5m2_to_bf16_bits(*input),
                    _ => unreachable!("CAST case must use FP8"),
                };
                assert_eq!(*expected, shared_bits);
            }
            assert!(case.output_matches(case.output.bits));
            assert!(!case.output_matches(&case.output.bits[..8]));
        }
    }

    #[test]
    fn int8_matmul_oracle_is_derived_from_the_shared_exact_dot_product() {
        use virtio_accel_tosa::dot_i8_i32;

        let lhs = MATMUL_INT8.inputs[0].bytes;
        let rhs = MATMUL_INT8.inputs[1].bytes;
        let mut actual = Vec::new();
        for row in 0..2 {
            for column in 0..2 {
                let left = &lhs[row * 3..row * 3 + 3];
                let right = [rhs[column], rhs[2 + column], rhs[4 + column]];
                actual.push(
                    dot_i8_i32(
                        left,
                        &right,
                        MATMUL_INT8.zero_points[0],
                        MATMUL_INT8.zero_points[1],
                        0,
                    )
                    .unwrap(),
                );
            }
        }
        assert!(MATMUL_INT8.output_matches(0, &actual));
        assert!(!MATMUL_INT8.output_matches(0, &[538, -544]));
        assert!(!MATMUL_INT8.output_matches(1, &actual));
    }

    #[test]
    fn packed_artifacts_are_valid_for_their_declared_tosa_profiles_and_extensions() {
        let integer = Target::new(
            Version::TOSA_1_0,
            ProfileSet::INTEGER,
            Level::Level8K,
            ExtensionSet::NONE,
        );
        let floating = |extension| {
            Target::new(
                Version::TOSA_1_0,
                ProfileSet::FLOATING_POINT,
                Level::Level8K,
                extension,
            )
        };
        for (case, target, dtype) in [
            (IDENTITY_INT8, integer, DType::INT8),
            (
                IDENTITY_INT4,
                Target::new(
                    Version::TOSA_1_0,
                    ProfileSet::INTEGER,
                    Level::Level8K,
                    ExtensionSet::INT4,
                ),
                DType::INT4,
            ),
            (
                IDENTITY_FP8E4M3,
                floating(ExtensionSet::FP8E4M3),
                DType::FP8E4M3,
            ),
            (
                IDENTITY_FP8E5M2,
                floating(ExtensionSet::FP8E5M2),
                DType::FP8E5M2,
            ),
        ] {
            parse(case.artifact).unwrap().validate_for(target).unwrap();
            let elements = case.inputs[0].shape.iter().product();
            assert_eq!(
                low_precision_storage_bytes(dtype, elements),
                Some(case.inputs[0].bytes.len())
            );
        }
        parse(MATMUL_INT8.artifact)
            .unwrap()
            .validate_for(integer)
            .unwrap();

        let fp8_storage = Target::new(
            Version::TOSA_1_0,
            ProfileSet::FLOATING_POINT,
            Level::Level8K,
            ExtensionSet::BF16
                .union(ExtensionSet::FP8E4M3)
                .union(ExtensionSet::FP8E5M2),
        );
        for case in [CAST_FP8E4M3_TO_BF16, CAST_FP8E5M2_TO_BF16] {
            parse(case.artifact)
                .unwrap()
                .validate_for(fp8_storage)
                .unwrap();
        }
    }
}
