use core::fmt;

use flatbuffers::Vector;

use crate::generated::tosa as wire;
use crate::{DType, Op};

macro_rules! numeric_kind {
    ($name:ident, $wire:ident, $($constant:ident),+ $(,)?) => {
        #[derive(Clone, Copy, PartialEq, Eq, Hash)]
        #[repr(transparent)]
        pub struct $name(u32);

        impl $name {
            $(pub const $constant: Self = Self(wire::$wire::$constant.0);)+

            pub const fn new(raw: u32) -> Self {
                Self(raw)
            }

            pub const fn get(self) -> u32 {
                self.0
            }

            pub fn name(self) -> Option<&'static str> {
                wire::$wire(self.0).variant_name()
            }
        }

        impl fmt::Debug for $name {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                match self.name() {
                    Some(name) => formatter.write_str(name),
                    None => formatter.debug_tuple(stringify!($name)).field(&self.0).finish(),
                }
            }
        }
    };
}

numeric_kind!(
    NanPropagationMode,
    NanPropagationMode,
    UNKNOWN,
    PROPAGATE,
    IGNORE,
);
numeric_kind!(ResizeMode, ResizeMode, UNKNOWN, NEAREST, BILINEAR);
numeric_kind!(
    RoundingMode,
    RoundingMode,
    UNKNOWN,
    SINGLE_ROUND,
    INEXACT_ROUND,
    DOUBLE_ROUND,
);

/// Borrowed FlatBuffers vector of little-endian `i32` attribute values.
#[derive(Clone, Copy)]
pub struct I32List<'a>(Option<Vector<'a, i32>>);

impl<'a> I32List<'a> {
    pub fn len(self) -> usize {
        match self.0 {
            Some(values) => values.len(),
            None => 0,
        }
    }

    pub fn is_empty(self) -> bool {
        self.len() == 0
    }

    pub fn get(self, index: usize) -> Option<i32> {
        self.0
            .and_then(|values| (index < values.len()).then(|| values.get(index)))
    }

    pub const fn iter(self) -> I32Values<'a> {
        I32Values {
            vector: self.0,
            index: 0,
        }
    }
}

impl fmt::Debug for I32List<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_list().entries(self.iter()).finish()
    }
}

/// Exact-size iterator returned by [`I32List::iter`].
#[derive(Clone)]
pub struct I32Values<'a> {
    vector: Option<Vector<'a, i32>>,
    index: usize,
}

impl Iterator for I32Values<'_> {
    type Item = i32;

    fn next(&mut self) -> Option<Self::Item> {
        let values = self.vector?;
        if self.index >= values.len() {
            return None;
        }
        let value = values.get(self.index);
        self.index += 1;
        Some(value)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self
            .vector
            .map_or(0, |values| values.len().saturating_sub(self.index));
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for I32Values<'_> {}

/// Safe, exhaustive view of the attribute payload used by a stable TOSA 1.0 operator.
///
/// Operators whose schema table has no fields return [`Self::Empty`]; [`Self::Empty::op`] still
/// identifies the exact table. Vector fields remain borrowed and are decoded without allocation.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub enum OpAttributes<'a> {
    Empty {
        op: Op,
    },
    ArgMax {
        axis: i32,
        nan_mode: NanPropagationMode,
    },
    AvgPool2d {
        kernel: I32List<'a>,
        stride: I32List<'a>,
        pad: I32List<'a>,
        acc_type: DType,
    },
    Conv2d {
        pad: I32List<'a>,
        stride: I32List<'a>,
        dilation: I32List<'a>,
        local_bound: bool,
        acc_type: DType,
    },
    Conv3d {
        pad: I32List<'a>,
        stride: I32List<'a>,
        dilation: I32List<'a>,
        local_bound: bool,
        acc_type: DType,
    },
    DepthwiseConv2d {
        pad: I32List<'a>,
        stride: I32List<'a>,
        dilation: I32List<'a>,
        local_bound: bool,
        acc_type: DType,
    },
    Fft2d {
        inverse: bool,
        local_bound: bool,
    },
    MaxPool2d {
        kernel: I32List<'a>,
        stride: I32List<'a>,
        pad: I32List<'a>,
        nan_mode: NanPropagationMode,
    },
    Rfft2d {
        local_bound: bool,
    },
    TransposeConv2d {
        out_pad: I32List<'a>,
        stride: I32List<'a>,
        local_bound: bool,
        acc_type: DType,
    },
    Clamp {
        min_val: &'a [u8],
        max_val: &'a [u8],
        nan_mode: NanPropagationMode,
    },
    ArithmeticRightShift {
        round: bool,
    },
    Maximum {
        nan_mode: NanPropagationMode,
    },
    Minimum {
        nan_mode: NanPropagationMode,
    },
    ReduceAll {
        axis: i32,
    },
    ReduceAny {
        axis: i32,
    },
    ReduceMax {
        axis: i32,
        nan_mode: NanPropagationMode,
    },
    ReduceMin {
        axis: i32,
        nan_mode: NanPropagationMode,
    },
    ReduceProduct {
        axis: i32,
    },
    ReduceSum {
        axis: i32,
    },
    Concat {
        axis: i32,
    },
    Reverse {
        axis: i32,
    },
    Transpose {
        perms: I32List<'a>,
    },
    Resize {
        mode: ResizeMode,
    },
    Rescale {
        scale32: bool,
        rounding_mode: RoundingMode,
        per_channel: bool,
        input_unsigned: bool,
        output_unsigned: bool,
    },
    Custom {
        operator_name: Option<&'a str>,
        domain_name: Option<&'a str>,
        implementation_attrs: &'a [u8],
    },
    CondIf {
        then_graph: Option<&'a str>,
        else_graph: Option<&'a str>,
    },
    WhileLoop {
        cond_graph: Option<&'a str>,
        body_graph: Option<&'a str>,
    },
}

impl<'a> OpAttributes<'a> {
    pub(crate) fn from_wire(operator: wire::TosaOperator<'a>) -> Self {
        let op = Op::new(operator.op().0);
        match op.get() {
            1 => {
                let value = operator
                    .attribute_as_arg_max_attribute()
                    .expect("validated ARGMAX attribute");
                Self::ArgMax {
                    axis: value.axis(),
                    nan_mode: NanPropagationMode::new(value.nan_mode().0),
                }
            }
            2 => {
                let value = operator
                    .attribute_as_avg_pool_2d_attribute()
                    .expect("validated AVG_POOL2D attribute");
                Self::AvgPool2d {
                    kernel: I32List(value.kernel()),
                    stride: I32List(value.stride()),
                    pad: I32List(value.pad()),
                    acc_type: DType::new(value.acc_type().0),
                }
            }
            3 => {
                let value = operator
                    .attribute_as_conv_2d_attribute()
                    .expect("validated CONV2D attribute");
                Self::Conv2d {
                    pad: I32List(value.pad()),
                    stride: I32List(value.stride()),
                    dilation: I32List(value.dilation()),
                    local_bound: value.local_bound(),
                    acc_type: DType::new(value.acc_type().0),
                }
            }
            4 => {
                let value = operator
                    .attribute_as_conv_3d_attribute()
                    .expect("validated CONV3D attribute");
                Self::Conv3d {
                    pad: I32List(value.pad()),
                    stride: I32List(value.stride()),
                    dilation: I32List(value.dilation()),
                    local_bound: value.local_bound(),
                    acc_type: DType::new(value.acc_type().0),
                }
            }
            5 => {
                let value = operator
                    .attribute_as_depthwise_conv_2d_attribute()
                    .expect("validated DEPTHWISE_CONV2D attribute");
                Self::DepthwiseConv2d {
                    pad: I32List(value.pad()),
                    stride: I32List(value.stride()),
                    dilation: I32List(value.dilation()),
                    local_bound: value.local_bound(),
                    acc_type: DType::new(value.acc_type().0),
                }
            }
            6 => {
                let value = operator
                    .attribute_as_fft2d_attribute()
                    .expect("validated FFT2D attribute");
                Self::Fft2d {
                    inverse: value.inverse(),
                    local_bound: value.local_bound(),
                }
            }
            8 => {
                let value = operator
                    .attribute_as_max_pool_2d_attribute()
                    .expect("validated MAX_POOL2D attribute");
                Self::MaxPool2d {
                    kernel: I32List(value.kernel()),
                    stride: I32List(value.stride()),
                    pad: I32List(value.pad()),
                    nan_mode: NanPropagationMode::new(value.nan_mode().0),
                }
            }
            9 => {
                let value = operator
                    .attribute_as_rfft2d_attribute()
                    .expect("validated RFFT2D attribute");
                Self::Rfft2d {
                    local_bound: value.local_bound(),
                }
            }
            10 => {
                let value = operator
                    .attribute_as_transpose_conv_2d_attribute()
                    .expect("validated TRANSPOSE_CONV2D attribute");
                Self::TransposeConv2d {
                    out_pad: I32List(value.out_pad()),
                    stride: I32List(value.stride()),
                    local_bound: value.local_bound(),
                    acc_type: DType::new(value.acc_type().0),
                }
            }
            11 => {
                let value = operator
                    .attribute_as_clamp_attribute()
                    .expect("validated CLAMP attribute");
                Self::Clamp {
                    min_val: value.min_val().map_or(&[], |bytes| bytes.bytes()),
                    max_val: value.max_val().map_or(&[], |bytes| bytes.bytes()),
                    nan_mode: NanPropagationMode::new(value.nan_mode().0),
                }
            }
            16 => {
                let value = operator
                    .attribute_as_arithmetic_right_shift_attribute()
                    .expect("validated ARITHMETIC_RIGHT_SHIFT attribute");
                Self::ArithmeticRightShift {
                    round: value.round(),
                }
            }
            26 => {
                let value = operator
                    .attribute_as_maximum_attribute()
                    .expect("validated MAXIMUM attribute");
                Self::Maximum {
                    nan_mode: NanPropagationMode::new(value.nan_mode().0),
                }
            }
            27 => {
                let value = operator
                    .attribute_as_minimum_attribute()
                    .expect("validated MINIMUM attribute");
                Self::Minimum {
                    nan_mode: NanPropagationMode::new(value.nan_mode().0),
                }
            }
            49 => {
                let value = operator
                    .attribute_as_reduce_all_attribute()
                    .expect("validated REDUCE_ALL attribute");
                Self::ReduceAll { axis: value.axis() }
            }
            50 => {
                let value = operator
                    .attribute_as_reduce_any_attribute()
                    .expect("validated REDUCE_ANY attribute");
                Self::ReduceAny { axis: value.axis() }
            }
            51 => {
                let value = operator
                    .attribute_as_reduce_max_attribute()
                    .expect("validated REDUCE_MAX attribute");
                Self::ReduceMax {
                    axis: value.axis(),
                    nan_mode: NanPropagationMode::new(value.nan_mode().0),
                }
            }
            52 => {
                let value = operator
                    .attribute_as_reduce_min_attribute()
                    .expect("validated REDUCE_MIN attribute");
                Self::ReduceMin {
                    axis: value.axis(),
                    nan_mode: NanPropagationMode::new(value.nan_mode().0),
                }
            }
            53 => {
                let value = operator
                    .attribute_as_reduce_product_attribute()
                    .expect("validated REDUCE_PRODUCT attribute");
                Self::ReduceProduct { axis: value.axis() }
            }
            54 => {
                let value = operator
                    .attribute_as_reduce_sum_attribute()
                    .expect("validated REDUCE_SUM attribute");
                Self::ReduceSum { axis: value.axis() }
            }
            55 => {
                let value = operator
                    .attribute_as_concat_attribute()
                    .expect("validated CONCAT attribute");
                Self::Concat { axis: value.axis() }
            }
            58 => {
                let value = operator
                    .attribute_as_reverse_attribute()
                    .expect("validated REVERSE attribute");
                Self::Reverse { axis: value.axis() }
            }
            61 => {
                let value = operator
                    .attribute_as_transpose_attribute()
                    .expect("validated TRANSPOSE attribute");
                Self::Transpose {
                    perms: I32List(value.perms()),
                }
            }
            64 => {
                let value = operator
                    .attribute_as_resize_attribute()
                    .expect("validated RESIZE attribute");
                Self::Resize {
                    mode: ResizeMode::new(value.mode().0),
                }
            }
            66 => {
                let value = operator
                    .attribute_as_rescale_attribute()
                    .expect("validated RESCALE attribute");
                Self::Rescale {
                    scale32: value.scale32(),
                    rounding_mode: RoundingMode::new(value.rounding_mode().0),
                    per_channel: value.per_channel(),
                    input_unsigned: value.input_unsigned(),
                    output_unsigned: value.output_unsigned(),
                }
            }
            69 => {
                let value = operator
                    .attribute_as_custom_attribute()
                    .expect("validated CUSTOM attribute");
                Self::Custom {
                    operator_name: value.operator_name(),
                    domain_name: value.domain_name(),
                    implementation_attrs: value
                        .implementation_attrs()
                        .map_or(&[], |bytes| bytes.bytes()),
                }
            }
            70 => {
                let value = operator
                    .attribute_as_cond_if_attribute()
                    .expect("validated COND_IF attribute");
                Self::CondIf {
                    then_graph: value.then_graph(),
                    else_graph: value.else_graph(),
                }
            }
            71 => {
                let value = operator
                    .attribute_as_while_loop_attribute()
                    .expect("validated WHILE_LOOP attribute");
                Self::WhileLoop {
                    cond_graph: value.cond_graph(),
                    body_graph: value.body_graph(),
                }
            }
            _ => Self::Empty { op },
        }
    }
}
