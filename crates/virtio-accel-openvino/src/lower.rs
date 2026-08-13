//! TOSA 1.0 to OpenVINO IR lowering.
//!
//! This module intentionally owns the OpenVINO IR (version 11) XML and weights encoding.
//! Portable crates expose only the verified TOSA model and provider-neutral analysis; no
//! OpenVINO type, header, or dependency crosses the backend boundary, so this encoder compiles
//! and unit-tests on every platform.

// Builds without a detected OpenVINO runtime type-check and unit-test this backend-local
// encoder, but only the native runtime modules call it from `load_program`.
#![cfg_attr(not(va_openvino), allow(dead_code))]

use std::fmt;

use virtio_accel_tosa::{
    AnalysisError, DType, Error as ParseError, ExtensionSet, Level, Op, ProfileSet, Target, Version,
};

/// TOSA target currently lowered by the OpenVINO backend.
pub const OPENVINO_TOSA_TARGET: Target = Target::new(
    Version::TOSA_1_0,
    ProfileSet::FLOATING_POINT,
    Level::Level8K,
    ExtensionSet::NONE,
);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LoweringError {
    Parse(ParseError),
    Analysis(AnalysisError),
    UnsupportedGraph,
    UnsupportedType(DType),
    UnsupportedOperator(Op),
    InvalidConstant,
    ResourceLimit,
}

impl fmt::Display for LoweringError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for LoweringError {}

/// Whether the initial OpenVINO lowering tier can lower `op` for supported types and attributes.
pub const fn supports_tosa_operator(op: Op) -> bool {
    matches!(
        op,
        Op::ARGMAX
            | Op::MATMUL
            | Op::MAX_POOL2D
            | Op::CLAMP
            | Op::ERF
            | Op::SIGMOID
            | Op::TANH
            | Op::ADD
            | Op::LOGICAL_AND
            | Op::LOGICAL_OR
            | Op::LOGICAL_XOR
            | Op::MAXIMUM
            | Op::MINIMUM
            | Op::MUL
            | Op::POW
            | Op::SUB
            | Op::ABS
            | Op::CEIL
            | Op::COS
            | Op::EXP
            | Op::FLOOR
            | Op::LOG
            | Op::LOGICAL_NOT
            | Op::NEGATE
            | Op::RECIPROCAL
            | Op::RSQRT
            | Op::SIN
            | Op::SELECT
            | Op::EQUAL
            | Op::GREATER
            | Op::GREATER_EQUAL
            | Op::REDUCE_MAX
            | Op::REDUCE_MIN
            | Op::REDUCE_PRODUCT
            | Op::REDUCE_SUM
            | Op::CONCAT
            | Op::RESHAPE
            | Op::REVERSE
            | Op::TRANSPOSE
            | Op::CONST
            | Op::CONST_SHAPE
            | Op::IDENTITY
    )
}

/// Whether this lowering can expose `dtype` at an OpenVINO model boundary.
///
/// Operator-specific validation still applies. In particular, INT32 is limited to outputs from
/// operators such as `ARGMAX`. Quantized TOSA tensor types require a future quantization-aware
/// lowering tier with explicit calibration semantics; the current IR encoder rejects them during
/// program admission instead of silently dequantizing.
pub const fn supports_tosa_dtype(dtype: DType) -> bool {
    matches!(dtype, DType::FP16 | DType::FP32 | DType::INT32)
}
