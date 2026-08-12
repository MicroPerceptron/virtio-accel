use core::fmt;

use crate::generated::tosa as wire;

/// Stable TOSA graph version.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Version {
    pub major: u16,
    pub minor: u16,
    pub patch: u16,
}

impl Version {
    /// Version accepted by the default validator.
    pub const TOSA_1_0: Self = Self {
        major: 1,
        minor: 0,
        patch: 0,
    };

    pub const fn new(major: u16, minor: u16, patch: u16) -> Self {
        Self {
            major,
            minor,
            patch,
        }
    }
}

/// A TOSA tensor data-type discriminant.
///
/// Raw values remain representable so policy and diagnostic utilities can discuss newer schemas.
/// Successfully parsed models contain only values accepted by [`DType::is_tosa_1_0`].
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct DType(u32);

impl DType {
    pub const BOOL: Self = Self(1);
    pub const INT4: Self = Self(2);
    pub const INT8: Self = Self(3);
    pub const INT16: Self = Self(4);
    pub const INT32: Self = Self(5);
    pub const INT48: Self = Self(6);
    pub const FP32: Self = Self(7);
    pub const FP16: Self = Self(8);
    pub const BF16: Self = Self(9);
    pub const SHAPE: Self = Self(10);
    pub const FP8E4M3: Self = Self(11);
    pub const FP8E5M2: Self = Self(12);

    pub const fn new(raw: u32) -> Self {
        Self(raw)
    }

    pub const fn get(self) -> u32 {
        self.0
    }

    pub const fn is_tosa_1_0(self) -> bool {
        self.0 >= Self::BOOL.0 && self.0 <= Self::FP8E5M2.0
    }

    pub fn name(self) -> Option<&'static str> {
        wire::DType(self.0).variant_name()
    }
}

impl fmt::Debug for DType {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.name() {
            Some(name) => formatter.write_str(name),
            None => formatter.debug_tuple("DType").field(&self.0).finish(),
        }
    }
}

/// A TOSA operator discriminant.
///
/// Constants cover the stable TOSA 1.0 operator set. Raw values above [`Op::CONST_SHAPE`] belong
/// to schema additions newer than TOSA 1.0 and are rejected by [`crate::parse`].
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct Op(u32);

macro_rules! op_constants {
    ($($name:ident = $value:literal),+ $(,)?) => {
        $(pub const $name: Self = Self($value);)+

        /// Every operator in the stable TOSA 1.0 serialization set, in wire order.
        pub const ALL: &'static [Self] = &[$(Self::$name),+];
    };
}

/// Accepted serialized operand counts for an operator.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Arity {
    pub min_inputs: usize,
    pub max_inputs: Option<usize>,
    pub min_outputs: usize,
    pub max_outputs: Option<usize>,
    pub matching_input_output_counts: bool,
}

impl Arity {
    pub const fn exact(inputs: usize, outputs: usize) -> Self {
        Self {
            min_inputs: inputs,
            max_inputs: Some(inputs),
            min_outputs: outputs,
            max_outputs: Some(outputs),
            matching_input_output_counts: false,
        }
    }

    pub const fn accepts(self, inputs: usize, outputs: usize) -> bool {
        inputs >= self.min_inputs
            && match self.max_inputs {
                Some(maximum) => inputs <= maximum,
                None => true,
            }
            && outputs >= self.min_outputs
            && match self.max_outputs {
                Some(maximum) => outputs <= maximum,
                None => true,
            }
            && (!self.matching_input_output_counts || inputs == outputs)
    }
}

impl Op {
    op_constants! {
        ARGMAX = 1,
        AVG_POOL2D = 2,
        CONV2D = 3,
        CONV3D = 4,
        DEPTHWISE_CONV2D = 5,
        FFT2D = 6,
        MATMUL = 7,
        MAX_POOL2D = 8,
        RFFT2D = 9,
        TRANSPOSE_CONV2D = 10,
        CLAMP = 11,
        ERF = 12,
        SIGMOID = 13,
        TANH = 14,
        ADD = 15,
        ARITHMETIC_RIGHT_SHIFT = 16,
        BITWISE_AND = 17,
        BITWISE_OR = 18,
        BITWISE_XOR = 19,
        INTDIV = 20,
        LOGICAL_AND = 21,
        LOGICAL_LEFT_SHIFT = 22,
        LOGICAL_RIGHT_SHIFT = 23,
        LOGICAL_OR = 24,
        LOGICAL_XOR = 25,
        MAXIMUM = 26,
        MINIMUM = 27,
        MUL = 28,
        POW = 29,
        SUB = 30,
        TABLE = 31,
        ABS = 32,
        BITWISE_NOT = 33,
        CEIL = 34,
        CLZ = 35,
        COS = 36,
        EXP = 37,
        FLOOR = 38,
        LOG = 39,
        LOGICAL_NOT = 40,
        NEGATE = 41,
        RECIPROCAL = 42,
        RSQRT = 43,
        SIN = 44,
        SELECT = 45,
        EQUAL = 46,
        GREATER = 47,
        GREATER_EQUAL = 48,
        REDUCE_ALL = 49,
        REDUCE_ANY = 50,
        REDUCE_MAX = 51,
        REDUCE_MIN = 52,
        REDUCE_PRODUCT = 53,
        REDUCE_SUM = 54,
        CONCAT = 55,
        PAD = 56,
        RESHAPE = 57,
        REVERSE = 58,
        SLICE = 59,
        TILE = 60,
        TRANSPOSE = 61,
        GATHER = 62,
        SCATTER = 63,
        RESIZE = 64,
        CAST = 65,
        RESCALE = 66,
        CONST = 67,
        IDENTITY = 68,
        CUSTOM = 69,
        COND_IF = 70,
        WHILE_LOOP = 71,
        VARIABLE = 72,
        VARIABLE_WRITE = 73,
        VARIABLE_READ = 74,
        CONST_SHAPE = 75,
    }

    pub const fn new(raw: u32) -> Self {
        Self(raw)
    }

    pub const fn get(self) -> u32 {
        self.0
    }

    pub const fn is_tosa_1_0(self) -> bool {
        self.0 >= Self::ARGMAX.0 && self.0 <= Self::CONST_SHAPE.0
    }

    pub fn name(self) -> Option<&'static str> {
        wire::Op(self.0).variant_name()
    }

    /// Stable TOSA 1.0 serialized operand-count contract.
    pub const fn arity(self) -> Option<Arity> {
        let exact = match self.0 {
            1 => (1, 1),
            2 => (3, 1),
            3..=5 => (5, 1),
            6 => (2, 2),
            7 => (4, 1),
            8 => (1, 1),
            9 => (1, 2),
            10 => (5, 1),
            11..=14 => (1, 1),
            15..=27 => (2, 1),
            28 => (3, 1),
            29..=31 => (2, 1),
            32..=40 => (1, 1),
            41 => (3, 1),
            42..=44 => (1, 1),
            45 => (3, 1),
            46..=48 => (2, 1),
            49..=54 => (1, 1),
            56 => (3, 1),
            57 => (2, 1),
            58 => (1, 1),
            59 => (3, 1),
            60 => (2, 1),
            61 => (1, 1),
            62 => (2, 1),
            63 => (3, 1),
            64 => (4, 1),
            65 => (1, 1),
            66 => (5, 1),
            67 => (0, 1),
            68 => (1, 1),
            72 => (0, 0),
            73 => (1, 0),
            74 => (0, 1),
            75 => (0, 1),
            55 | 69..=71 => {
                return Some(match self.0 {
                    55 => Arity {
                        min_inputs: 1,
                        max_inputs: None,
                        min_outputs: 1,
                        max_outputs: Some(1),
                        matching_input_output_counts: false,
                    },
                    69 => Arity {
                        min_inputs: 0,
                        max_inputs: None,
                        min_outputs: 0,
                        max_outputs: None,
                        matching_input_output_counts: false,
                    },
                    70 => Arity {
                        min_inputs: 1,
                        max_inputs: None,
                        min_outputs: 0,
                        max_outputs: None,
                        matching_input_output_counts: false,
                    },
                    71 => Arity {
                        min_inputs: 0,
                        max_inputs: None,
                        min_outputs: 0,
                        max_outputs: None,
                        matching_input_output_counts: true,
                    },
                    _ => unreachable!(),
                });
            }
            _ => return None,
        };
        Some(Arity::exact(exact.0, exact.1))
    }
}

impl fmt::Debug for Op {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.name() {
            Some(name) => formatter.write_str(name),
            None => formatter.debug_tuple("Op").field(&self.0).finish(),
        }
    }
}

/// Discriminant of an operator's serialized attribute table.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct AttributeKind(u8);

impl AttributeKind {
    pub const fn new(raw: u8) -> Self {
        Self(raw)
    }

    pub const fn get(self) -> u8 {
        self.0
    }

    pub fn name(self) -> Option<&'static str> {
        wire::Attribute(self.0).variant_name()
    }
}

impl fmt::Debug for AttributeKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.name() {
            Some(name) => formatter.write_str(name),
            None => formatter
                .debug_tuple("AttributeKind")
                .field(&self.0)
                .finish(),
        }
    }
}
