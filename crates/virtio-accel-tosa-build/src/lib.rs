//! Safe authoring of static, single-block TOSA 1.0 artifacts.
//!
//! The raw FlatBuffers layout is private to this crate. Borrowed [`Graph`] definitions suit
//! statically declared artifacts, while [`OwnedGraph`] supports frontends that discover names,
//! shapes, constants, and operators incrementally. Both surfaces round-trip every artifact through
//! `virtio-accel-tosa`'s bounded parser and complete target validator before returning owned bytes.

#![no_std]
#![forbid(unsafe_code)]

extern crate alloc;

use alloc::borrow::Cow;
use alloc::collections::BTreeSet;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::fmt;

use flatbuffers::{FlatBufferBuilder, TableFinishedWIPOffset, WIPOffset};
use virtio_accel_tosa::{DType, NanPropagationMode, Op, SemanticError, Target};

type Table = WIPOffset<TableFinishedWIPOffset>;

// These are the six stable container-table vtable positions used by the
// single-block surface. Keeping them here prevents consumers from duplicating
// schema layout knowledge; the completed artifact is always verified by the
// ingestion crate before it crosses this API boundary.
mod slot {
    pub const TENSOR_NAME: u16 = 4;
    pub const TENSOR_SHAPE: u16 = 6;
    pub const TENSOR_TYPE: u16 = 8;
    pub const TENSOR_DATA: u16 = 10;

    pub const SHAPE_NAME: u16 = 4;
    pub const SHAPE_RANK: u16 = 6;
    pub const SHAPE_DATA: u16 = 8;

    pub const OPERATOR_OP: u16 = 4;
    pub const OPERATOR_ATTRIBUTE_TYPE: u16 = 6;
    pub const OPERATOR_ATTRIBUTE: u16 = 8;
    pub const OPERATOR_INPUTS: u16 = 10;
    pub const OPERATOR_OUTPUTS: u16 = 12;

    pub const BLOCK_NAME: u16 = 4;
    pub const BLOCK_OPERATORS: u16 = 6;
    pub const BLOCK_TENSORS: u16 = 8;
    pub const BLOCK_INPUTS: u16 = 10;
    pub const BLOCK_OUTPUTS: u16 = 12;
    pub const BLOCK_SHAPES: u16 = 14;

    pub const REGION_NAME: u16 = 4;
    pub const REGION_BLOCKS: u16 = 6;

    pub const GRAPH_VERSION: u16 = 4;
    pub const GRAPH_REGIONS: u16 = 6;

    pub const VERSION_MAJOR: u16 = 4;
    pub const VERSION_MINOR: u16 = 6;
    pub const VERSION_PATCH: u16 = 8;
    pub const VERSION_DRAFT: u16 = 10;

    pub const NAN_MODE: u16 = 4;
    pub const MAX_POOL_KERNEL: u16 = 4;
    pub const MAX_POOL_STRIDE: u16 = 6;
    pub const MAX_POOL_PAD: u16 = 8;
    pub const MAX_POOL_NAN_MODE: u16 = 10;
}

/// A serialized compile-time shape value.
///
/// Shape values occupy the TOSA shape namespace rather than the tensor namespace. They must be
/// produced by [`OperatorKind::ConstShape`] before an operator such as `RESHAPE` consumes them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shape<'a> {
    /// Unique block-local shape name.
    pub name: &'a str,
    /// Signed 64-bit TOSA shape components. A scalar target shape uses an empty slice.
    pub values: &'a [i64],
}

impl<'a> Shape<'a> {
    pub const fn new(name: &'a str, values: &'a [i64]) -> Self {
        Self { name, values }
    }
}

/// A statically shaped tensor definition.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tensor<'a> {
    /// Unique block-local tensor name.
    pub name: &'a str,
    /// Concrete dimensions. Scalars use an empty slice.
    pub shape: &'a [i32],
    /// Stable TOSA 1.0 element type.
    pub dtype: DType,
    /// Inline constant bytes, when this tensor is produced by `CONST`.
    pub data: Option<&'a [u8]>,
}

impl<'a> Tensor<'a> {
    /// Define a non-constant tensor.
    pub const fn new(name: &'a str, shape: &'a [i32], dtype: DType) -> Self {
        Self {
            name,
            shape,
            dtype,
            data: None,
        }
    }

    /// Define a tensor with inline constant storage.
    pub const fn constant(name: &'a str, shape: &'a [i32], dtype: DType, data: &'a [u8]) -> Self {
        Self {
            name,
            shape,
            dtype,
            data: Some(data),
        }
    }
}

/// Typed operator kinds supported by the initial authoring surface.
///
/// Variants with no fields still name their exact TOSA attribute table. This
/// prevents a caller from pairing an opcode with the wrong union member.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum OperatorKind {
    MatMul,
    MaxPool2d {
        kernel: [i32; 2],
        stride: [i32; 2],
        pad: [i32; 4],
        nan_mode: NanPropagationMode,
    },
    Sigmoid,
    Tanh,
    Add,
    LogicalAnd,
    LogicalOr,
    LogicalXor,
    Maximum {
        nan_mode: NanPropagationMode,
    },
    Minimum {
        nan_mode: NanPropagationMode,
    },
    Mul,
    Pow,
    Sub,
    Abs,
    Cos,
    Exp,
    Log,
    LogicalNot,
    Negate,
    Reciprocal,
    Rsqrt,
    Sin,
    Select,
    Equal,
    Greater,
    GreaterEqual,
    Reshape,
    Cast,
    Const,
    Identity,
    ConstShape,
}

impl OperatorKind {
    /// Stable TOSA opcode selected by this typed variant.
    pub const fn op(self) -> Op {
        match self {
            Self::MatMul => Op::MATMUL,
            Self::MaxPool2d { .. } => Op::MAX_POOL2D,
            Self::Sigmoid => Op::SIGMOID,
            Self::Tanh => Op::TANH,
            Self::Add => Op::ADD,
            Self::LogicalAnd => Op::LOGICAL_AND,
            Self::LogicalOr => Op::LOGICAL_OR,
            Self::LogicalXor => Op::LOGICAL_XOR,
            Self::Maximum { .. } => Op::MAXIMUM,
            Self::Minimum { .. } => Op::MINIMUM,
            Self::Mul => Op::MUL,
            Self::Pow => Op::POW,
            Self::Sub => Op::SUB,
            Self::Abs => Op::ABS,
            Self::Cos => Op::COS,
            Self::Exp => Op::EXP,
            Self::Log => Op::LOG,
            Self::LogicalNot => Op::LOGICAL_NOT,
            Self::Negate => Op::NEGATE,
            Self::Reciprocal => Op::RECIPROCAL,
            Self::Rsqrt => Op::RSQRT,
            Self::Sin => Op::SIN,
            Self::Select => Op::SELECT,
            Self::Equal => Op::EQUAL,
            Self::Greater => Op::GREATER,
            Self::GreaterEqual => Op::GREATER_EQUAL,
            Self::Reshape => Op::RESHAPE,
            Self::Cast => Op::CAST,
            Self::Const => Op::CONST,
            Self::Identity => Op::IDENTITY,
            Self::ConstShape => Op::CONST_SHAPE,
        }
    }
}

/// One operator and its block-local tensor references.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Operator<'a> {
    pub kind: OperatorKind,
    pub inputs: &'a [&'a str],
    pub outputs: &'a [&'a str],
}

impl<'a> Operator<'a> {
    pub const fn new(kind: OperatorKind, inputs: &'a [&'a str], outputs: &'a [&'a str]) -> Self {
        Self {
            kind,
            inputs,
            outputs,
        }
    }
}

/// An owned compile-time shape for incrementally assembled graphs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OwnedShape {
    pub name: String,
    pub values: Vec<i64>,
}

impl OwnedShape {
    pub fn new(name: impl Into<String>, values: Vec<i64>) -> Self {
        Self {
            name: name.into(),
            values,
        }
    }
}

impl From<Shape<'_>> for OwnedShape {
    fn from(shape: Shape<'_>) -> Self {
        Self::new(shape.name, shape.values.to_vec())
    }
}

/// An owned tensor definition for incrementally assembled graphs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OwnedTensor<'a> {
    pub name: String,
    pub shape: Vec<i32>,
    pub dtype: DType,
    /// Inline constant bytes. Existing frontend storage may be borrowed to avoid a second copy;
    /// newly transformed data can be moved into this field.
    pub data: Option<Cow<'a, [u8]>>,
}

impl<'a> OwnedTensor<'a> {
    /// Define a non-constant owned tensor.
    pub fn new(name: impl Into<String>, shape: Vec<i32>, dtype: DType) -> Self {
        Self {
            name: name.into(),
            shape,
            dtype,
            data: None,
        }
    }

    /// Define an owned tensor with inline constant storage.
    pub fn constant(
        name: impl Into<String>,
        shape: Vec<i32>,
        dtype: DType,
        data: impl Into<Cow<'a, [u8]>>,
    ) -> Self {
        Self {
            name: name.into(),
            shape,
            dtype,
            data: Some(data.into()),
        }
    }
}

impl<'a> From<Tensor<'a>> for OwnedTensor<'a> {
    fn from(tensor: Tensor<'a>) -> Self {
        match tensor.data {
            Some(data) => Self::constant(tensor.name, tensor.shape.to_vec(), tensor.dtype, data),
            None => Self::new(tensor.name, tensor.shape.to_vec(), tensor.dtype),
        }
    }
}

/// An owned typed operator for incrementally assembled graphs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OwnedOperator {
    pub kind: OperatorKind,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
}

impl OwnedOperator {
    /// Define an operator from owned operand names.
    pub const fn new(kind: OperatorKind, inputs: Vec<String>, outputs: Vec<String>) -> Self {
        Self {
            kind,
            inputs,
            outputs,
        }
    }
}

impl From<Operator<'_>> for OwnedOperator {
    fn from(operator: Operator<'_>) -> Self {
        Self::new(
            operator.kind,
            operator
                .inputs
                .iter()
                .map(|name| (*name).to_string())
                .collect(),
            operator
                .outputs
                .iter()
                .map(|name| (*name).to_string())
                .collect(),
        )
    }
}

/// An owned, incrementally assembled static TOSA graph.
///
/// This surface is intended for compiler frontends whose names, shapes, constants, and operator
/// lists are discovered at runtime. It owns names and metadata while permitting explicit borrowing
/// of existing constant storage. It creates borrowed [`Graph`] views only during [`Self::build`],
/// so callers do not need parallel adapter structures. The same structural and target validation
/// as [`Graph::build`] remains authoritative.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OwnedGraph<'a> {
    pub name: String,
    pub tensors: Vec<OwnedTensor<'a>>,
    pub shapes: Vec<OwnedShape>,
    pub operators: Vec<OwnedOperator>,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
}

impl<'a> OwnedGraph<'a> {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            tensors: Vec::new(),
            shapes: Vec::new(),
            operators: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
        }
    }

    pub fn push_tensor(&mut self, tensor: OwnedTensor<'a>) -> &mut Self {
        self.tensors.push(tensor);
        self
    }

    pub fn push_shape(&mut self, shape: OwnedShape) -> &mut Self {
        self.shapes.push(shape);
        self
    }

    pub fn push_operator(&mut self, operator: OwnedOperator) -> &mut Self {
        self.operators.push(operator);
        self
    }

    pub fn push_input(&mut self, name: impl Into<String>) -> &mut Self {
        self.inputs.push(name.into());
        self
    }

    pub fn push_output(&mut self, name: impl Into<String>) -> &mut Self {
        self.outputs.push(name.into());
        self
    }

    /// Serialize and semantically validate this graph for `target`.
    ///
    /// The graph remains reusable after a build. All metadata views allocated here are transient;
    /// the returned artifact bytes are the only allocation retained by the caller.
    pub fn build(&self, target: Target) -> Result<Vec<u8>, BuildError> {
        let tensors: Vec<_> = self
            .tensors
            .iter()
            .map(|tensor| match &tensor.data {
                Some(data) => {
                    Tensor::constant(&tensor.name, &tensor.shape, tensor.dtype, data.as_ref())
                }
                None => Tensor::new(&tensor.name, &tensor.shape, tensor.dtype),
            })
            .collect();
        let shapes: Vec<_> = self
            .shapes
            .iter()
            .map(|shape| Shape::new(&shape.name, &shape.values))
            .collect();
        let operator_inputs: Vec<Vec<&str>> = self
            .operators
            .iter()
            .map(|operator| operator.inputs.iter().map(String::as_str).collect())
            .collect();
        let operator_outputs: Vec<Vec<&str>> = self
            .operators
            .iter()
            .map(|operator| operator.outputs.iter().map(String::as_str).collect())
            .collect();
        let operators: Vec<_> = self
            .operators
            .iter()
            .enumerate()
            .map(|(index, operator)| {
                Operator::new(
                    operator.kind,
                    &operator_inputs[index],
                    &operator_outputs[index],
                )
            })
            .collect();
        let inputs: Vec<_> = self.inputs.iter().map(String::as_str).collect();
        let outputs: Vec<_> = self.outputs.iter().map(String::as_str).collect();

        Graph::new(&self.name, &tensors, &operators, &inputs, &outputs)
            .with_shapes(&shapes)
            .build(target)
    }
}

impl<'a> From<Graph<'a>> for OwnedGraph<'a> {
    fn from(graph: Graph<'a>) -> Self {
        Self {
            name: graph.name.to_string(),
            tensors: graph.tensors.iter().copied().map(Into::into).collect(),
            shapes: graph.shapes.iter().copied().map(Into::into).collect(),
            operators: graph.operators.iter().copied().map(Into::into).collect(),
            inputs: graph
                .inputs
                .iter()
                .map(|name| (*name).to_string())
                .collect(),
            outputs: graph
                .outputs
                .iter()
                .map(|name| (*name).to_string())
                .collect(),
        }
    }
}

/// A static TOSA graph containing one region and one basic block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Graph<'a> {
    pub name: &'a str,
    pub tensors: &'a [Tensor<'a>],
    pub shapes: &'a [Shape<'a>],
    pub operators: &'a [Operator<'a>],
    pub inputs: &'a [&'a str],
    pub outputs: &'a [&'a str],
}

impl<'a> Graph<'a> {
    pub const fn new(
        name: &'a str,
        tensors: &'a [Tensor<'a>],
        operators: &'a [Operator<'a>],
        inputs: &'a [&'a str],
        outputs: &'a [&'a str],
    ) -> Self {
        Self {
            name,
            tensors,
            shapes: &[],
            operators,
            inputs,
            outputs,
        }
    }

    /// Add compile-time shape values to this graph.
    pub const fn with_shapes(mut self, shapes: &'a [Shape<'a>]) -> Self {
        self.shapes = shapes;
        self
    }

    /// Serialize and semantically validate this graph for `target`.
    ///
    /// The returned vector is the only allocation retained by the caller.
    /// Construction is linear in graph metadata and constant bytes. Validation
    /// runs once on this cold authoring path; providers still perform their own
    /// authoritative admission during `load_program`.
    pub fn build(self, target: Target) -> Result<Vec<u8>, BuildError> {
        self.check_static_surface()?;
        let bytes = self.serialize();
        let model = virtio_accel_tosa::parse(&bytes).map_err(BuildError::Parse)?;
        model.validate_for(target).map_err(BuildError::Semantic)?;
        Ok(bytes)
    }

    fn check_static_surface(self) -> Result<(), BuildError> {
        if self.name.is_empty() {
            return Err(BuildError::EmptyGraphName);
        }
        for tensor in self.tensors {
            if tensor.shape.iter().any(|dimension| *dimension < 0) {
                return Err(BuildError::DynamicShape);
            }
            if !tensor.dtype.is_tosa_1_0() {
                return Err(BuildError::UnsupportedDType(tensor.dtype));
            }
        }

        for shape in self.shapes {
            if u32::try_from(shape.values.len()).is_err() {
                return Err(BuildError::ShapeRankOverflow);
            }
        }

        let mut constant_tensors = BTreeSet::new();
        let mut constant_shapes = BTreeSet::new();
        for operator in self.operators {
            let outputs = match operator.kind {
                OperatorKind::Const => &mut constant_tensors,
                OperatorKind::ConstShape => &mut constant_shapes,
                _ => continue,
            };
            outputs.extend(operator.outputs.iter().copied());
        }

        for tensor in self.tensors {
            match tensor.data {
                Some([]) => return Err(BuildError::EmptyConstantData),
                Some(_) if !constant_tensors.contains(tensor.name) => {
                    return Err(BuildError::TensorDataWithoutConst);
                }
                Some(_) => {}
                None if constant_tensors.contains(tensor.name) => {
                    return Err(BuildError::ConstWithoutTensorData);
                }
                None => {}
            }
        }
        if self
            .shapes
            .iter()
            .any(|shape| !constant_shapes.contains(shape.name))
        {
            return Err(BuildError::ShapeWithoutConstShape);
        }
        Ok(())
    }

    fn serialize(self) -> Vec<u8> {
        let mut builder = FlatBufferBuilder::with_capacity(4096);

        let tensors = {
            let tables: Vec<_> = self
                .tensors
                .iter()
                .map(|tensor| tensor_table(&mut builder, *tensor))
                .collect();
            builder.create_vector(&tables)
        };
        let shapes = {
            let tables: Vec<_> = self
                .shapes
                .iter()
                .map(|shape| shape_table(&mut builder, *shape))
                .collect();
            builder.create_vector(&tables)
        };
        let operators = {
            let tables: Vec<_> = self
                .operators
                .iter()
                .map(|operator| operator_table(&mut builder, *operator))
                .collect();
            builder.create_vector(&tables)
        };
        let block = {
            let name = builder.create_string(self.name);
            let inputs = string_vector(&mut builder, self.inputs);
            let outputs = string_vector(&mut builder, self.outputs);
            let table = builder.start_table();
            builder.push_slot_always(slot::BLOCK_NAME, name);
            builder.push_slot_always(slot::BLOCK_OPERATORS, operators);
            builder.push_slot_always(slot::BLOCK_TENSORS, tensors);
            builder.push_slot_always(slot::BLOCK_INPUTS, inputs);
            builder.push_slot_always(slot::BLOCK_OUTPUTS, outputs);
            builder.push_slot_always(slot::BLOCK_SHAPES, shapes);
            builder.end_table(table)
        };
        let region = {
            let name = builder.create_string(self.name);
            let blocks = builder.create_vector(&[block]);
            let table = builder.start_table();
            builder.push_slot_always(slot::REGION_NAME, name);
            builder.push_slot_always(slot::REGION_BLOCKS, blocks);
            builder.end_table(table)
        };
        let graph = {
            let version = {
                let table = builder.start_table();
                builder.push_slot::<i32>(slot::VERSION_MAJOR, 1, -1);
                builder.push_slot::<i32>(slot::VERSION_MINOR, 0, -1);
                builder.push_slot::<i32>(slot::VERSION_PATCH, 0, -1);
                builder.push_slot::<bool>(slot::VERSION_DRAFT, false, true);
                builder.end_table(table)
            };
            let regions = builder.create_vector(&[region]);
            let table = builder.start_table();
            builder.push_slot_always(slot::GRAPH_VERSION, version);
            builder.push_slot_always(slot::GRAPH_REGIONS, regions);
            builder.end_table(table)
        };

        builder.finish(graph, Some("TOSA"));
        builder.finished_data().to_vec()
    }
}

fn shape_table(builder: &mut FlatBufferBuilder<'_>, shape: Shape<'_>) -> Table {
    let name = builder.create_string(shape.name);
    let bytes: Vec<_> = shape
        .values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    let data = builder.create_vector(&bytes);
    let table = builder.start_table();
    builder.push_slot_always(slot::SHAPE_NAME, name);
    builder.push_slot::<u32>(
        slot::SHAPE_RANK,
        u32::try_from(shape.values.len()).expect("shape rank checked before serialization"),
        0,
    );
    builder.push_slot_always(slot::SHAPE_DATA, data);
    builder.end_table(table)
}

fn tensor_table(builder: &mut FlatBufferBuilder<'_>, tensor: Tensor<'_>) -> Table {
    let name = builder.create_string(tensor.name);
    let shape = builder.create_vector(tensor.shape);
    let data = tensor.data.map(|bytes| builder.create_vector(bytes));
    let table = builder.start_table();
    builder.push_slot_always(slot::TENSOR_NAME, name);
    builder.push_slot_always(slot::TENSOR_SHAPE, shape);
    builder.push_slot::<u32>(slot::TENSOR_TYPE, tensor.dtype.get(), 0);
    if let Some(data) = data {
        builder.push_slot_always(slot::TENSOR_DATA, data);
    }
    builder.end_table(table)
}

fn operator_table(builder: &mut FlatBufferBuilder<'_>, operator: Operator<'_>) -> Table {
    let inputs = string_vector(builder, operator.inputs);
    let outputs = string_vector(builder, operator.outputs);
    let max_pool = match operator.kind {
        OperatorKind::MaxPool2d {
            kernel,
            stride,
            pad,
            nan_mode,
        } => Some((
            builder.create_vector(&kernel),
            builder.create_vector(&stride),
            builder.create_vector(&pad),
            nan_mode,
        )),
        _ => None,
    };
    let attribute = {
        let table = builder.start_table();
        match operator.kind {
            OperatorKind::Maximum { nan_mode } | OperatorKind::Minimum { nan_mode } => {
                builder.push_slot::<u32>(slot::NAN_MODE, nan_mode.get(), 0);
            }
            OperatorKind::MaxPool2d { .. } => {
                let (kernel, stride, pad, nan_mode) =
                    max_pool.expect("MAX_POOL2D vectors were constructed");
                builder.push_slot_always(slot::MAX_POOL_KERNEL, kernel);
                builder.push_slot_always(slot::MAX_POOL_STRIDE, stride);
                builder.push_slot_always(slot::MAX_POOL_PAD, pad);
                builder.push_slot::<u32>(slot::MAX_POOL_NAN_MODE, nan_mode.get(), 0);
            }
            _ => {}
        }
        builder.end_table(table)
    };
    let op = operator.kind.op();
    let table = builder.start_table();
    builder.push_slot::<u32>(slot::OPERATOR_OP, op.get(), 0);
    // In the pinned TOSA 1.0 schema, stable Attribute union members are in
    // the same order as their corresponding stable Op values.
    builder.push_slot::<u8>(slot::OPERATOR_ATTRIBUTE_TYPE, op.get() as u8, 0);
    builder.push_slot_always(slot::OPERATOR_ATTRIBUTE, attribute);
    builder.push_slot_always(slot::OPERATOR_INPUTS, inputs);
    builder.push_slot_always(slot::OPERATOR_OUTPUTS, outputs);
    builder.end_table(table)
}

fn string_vector<'a>(
    builder: &mut FlatBufferBuilder<'a>,
    values: &[&str],
) -> WIPOffset<flatbuffers::Vector<'a, flatbuffers::ForwardsUOffset<&'a str>>> {
    let offsets: Vec<_> = values
        .iter()
        .map(|value| builder.create_string(value))
        .collect();
    builder.create_vector(&offsets)
}

/// Failure to construct a static graph accepted by the shared TOSA target.
#[derive(Debug)]
#[non_exhaustive]
pub enum BuildError {
    EmptyGraphName,
    DynamicShape,
    ShapeRankOverflow,
    EmptyConstantData,
    TensorDataWithoutConst,
    ConstWithoutTensorData,
    ShapeWithoutConstShape,
    UnsupportedDType(DType),
    Parse(virtio_accel_tosa::Error),
    Semantic(SemanticError),
}

impl fmt::Display for BuildError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyGraphName => formatter.write_str("graph name must not be empty"),
            Self::DynamicShape => {
                formatter.write_str("the single-block authoring surface requires static shapes")
            }
            Self::ShapeRankOverflow => formatter.write_str("shape rank does not fit TOSA u32"),
            Self::EmptyConstantData => {
                formatter.write_str("tensor constants require nonempty inline data")
            }
            Self::TensorDataWithoutConst => {
                formatter.write_str("inline tensor data requires a CONST producer")
            }
            Self::ConstWithoutTensorData => {
                formatter.write_str("CONST outputs require nonempty inline tensor data")
            }
            Self::ShapeWithoutConstShape => {
                formatter.write_str("shape values require a CONST_SHAPE producer")
            }
            Self::UnsupportedDType(dtype) => {
                write!(formatter, "unsupported TOSA 1.0 dtype {dtype:?}")
            }
            Self::Parse(error) => write!(
                formatter,
                "constructed artifact failed structural validation: {error}"
            ),
            Self::Semantic(error) => write!(
                formatter,
                "constructed artifact failed semantic validation: {error}"
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use virtio_accel_tosa::{ExtensionSet, Level, ProfileSet, Version};

    const FLOAT_TARGET: Target = Target::new(
        Version::TOSA_1_0,
        ProfileSet::FLOATING_POINT,
        Level::Level8K,
        ExtensionSet::NONE,
    );

    #[test]
    fn identity_round_trips_through_production_validation() {
        let tensors = [
            Tensor::new("input", &[1, 4], DType::FP32),
            Tensor::new("output", &[1, 4], DType::FP32),
        ];
        let operators = [Operator::new(
            OperatorKind::Identity,
            &["input"],
            &["output"],
        )];
        let graph = Graph::new("main", &tensors, &operators, &["input"], &["output"]);

        let first = graph.build(FLOAT_TARGET).unwrap();
        let second = graph.build(FLOAT_TARGET).unwrap();
        assert_eq!(first, second);

        let model = virtio_accel_tosa::parse(&first).unwrap();
        model.validate_for(FLOAT_TARGET).unwrap();
        let block = model.regions().next().unwrap().blocks().next().unwrap();
        assert_eq!(block.operators().next().unwrap().op(), Op::IDENTITY);
    }

    #[test]
    fn constants_and_typed_nan_attributes_round_trip() {
        let one = 1.0_f32.to_le_bytes();
        let tensors = [
            Tensor::new("lhs", &[1], DType::FP32),
            Tensor::constant("rhs", &[1], DType::FP32, &one),
            Tensor::new("output", &[1], DType::FP32),
        ];
        let operators = [
            Operator::new(OperatorKind::Const, &[], &["rhs"]),
            Operator::new(
                OperatorKind::Maximum {
                    nan_mode: NanPropagationMode::PROPAGATE,
                },
                &["lhs", "rhs"],
                &["output"],
            ),
        ];
        let bytes = Graph::new("main", &tensors, &operators, &["lhs"], &["output"])
            .build(FLOAT_TARGET)
            .unwrap();

        let model = virtio_accel_tosa::parse(&bytes).unwrap();
        let block = model.regions().next().unwrap().blocks().next().unwrap();
        assert_eq!(block.operators().len(), 2);
    }

    #[test]
    fn incrementally_owned_graph_matches_borrowed_artifact_bytes() {
        let one = 1.0_f32.to_le_bytes();
        let tensors = [
            Tensor::new("lhs", &[1], DType::FP32),
            Tensor::constant("rhs", &[1], DType::FP32, &one),
            Tensor::new("output", &[1], DType::FP32),
        ];
        let operators = [
            Operator::new(OperatorKind::Const, &[], &["rhs"]),
            Operator::new(OperatorKind::Add, &["lhs", "rhs"], &["output"]),
        ];
        let expected = Graph::new("main", &tensors, &operators, &["lhs"], &["output"])
            .build(FLOAT_TARGET)
            .unwrap();

        let mut graph = OwnedGraph::new(String::from("main"));
        graph
            .push_tensor(OwnedTensor::new(String::from("lhs"), vec![1], DType::FP32))
            .push_tensor(OwnedTensor::constant(
                String::from("rhs"),
                vec![1],
                DType::FP32,
                one.to_vec(),
            ))
            .push_tensor(OwnedTensor::new("output", vec![1], DType::FP32))
            .push_operator(OwnedOperator::new(
                OperatorKind::Const,
                vec![],
                vec![String::from("rhs")],
            ))
            .push_operator(OwnedOperator::new(
                OperatorKind::Add,
                vec![String::from("lhs"), String::from("rhs")],
                vec![String::from("output")],
            ))
            .push_input(String::from("lhs"))
            .push_output(String::from("output"));

        let first = graph.build(FLOAT_TARGET).unwrap();
        let second = graph.build(FLOAT_TARGET).unwrap();
        assert_eq!(first, expected);
        assert_eq!(second, expected);
    }

    #[test]
    fn borrowed_graph_converts_to_an_incremental_graph() {
        let tensors = [
            Tensor::new("input", &[1, 4], DType::FP32),
            Tensor::new("output", &[2, 2], DType::FP32),
        ];
        let shapes = [Shape::new("target", &[2, 2])];
        let operators = [
            Operator::new(OperatorKind::ConstShape, &[], &["target"]),
            Operator::new(OperatorKind::Reshape, &["input", "target"], &["output"]),
        ];
        let borrowed =
            Graph::new("main", &tensors, &operators, &["input"], &["output"]).with_shapes(&shapes);
        let expected = borrowed.build(FLOAT_TARGET).unwrap();

        let owned = OwnedGraph::from(borrowed);
        assert_eq!(owned.build(FLOAT_TARGET).unwrap(), expected);
        assert_eq!(owned.shapes, vec![OwnedShape::new("target", vec![2, 2])]);
    }

    #[test]
    fn owned_tensor_can_reuse_existing_constant_storage() {
        let data = 1.0_f32.to_le_bytes();
        let tensor = OwnedTensor::constant("value", vec![1], DType::FP32, data.as_slice());

        let Some(Cow::Borrowed(stored)) = tensor.data else {
            panic!("borrowed constant data was copied");
        };
        assert!(core::ptr::eq(stored, data.as_slice()));
    }

    #[test]
    fn owned_graph_uses_the_borrowed_validation_contract() {
        let mut graph = OwnedGraph::new("main");
        graph
            .push_tensor(OwnedTensor::constant(
                "value",
                vec![1],
                DType::FP32,
                1.0_f32.to_le_bytes().to_vec(),
            ))
            .push_input("value")
            .push_output("value");

        assert!(matches!(
            graph.build(FLOAT_TARGET),
            Err(BuildError::TensorDataWithoutConst)
        ));
    }

    #[test]
    fn reshape_uses_a_typed_const_shape_operand() {
        let tensors = [
            Tensor::new("input", &[1, 4], DType::FP32),
            Tensor::new("output", &[2, 2], DType::FP32),
        ];
        let shapes = [Shape::new("target", &[2, 2])];
        let operators = [
            Operator::new(OperatorKind::ConstShape, &[], &["target"]),
            Operator::new(OperatorKind::Reshape, &["input", "target"], &["output"]),
        ];
        let bytes = Graph::new("main", &tensors, &operators, &["input"], &["output"])
            .with_shapes(&shapes)
            .build(FLOAT_TARGET)
            .unwrap();

        let model = virtio_accel_tosa::parse(&bytes).unwrap();
        let block = model.regions().next().unwrap().blocks().next().unwrap();
        assert_eq!(block.shapes().len(), 1);
        assert_eq!(block.shapes().next().unwrap().values().unwrap().len(), 2);
        assert_eq!(block.operators().len(), 2);
    }

    #[test]
    fn inline_tensor_data_requires_a_nonempty_const_producer() {
        let one = 1.0_f32.to_le_bytes();
        let data_without_const = [Tensor::constant("value", &[1], DType::FP32, &one)];
        assert!(matches!(
            Graph::new("main", &data_without_const, &[], &["value"], &["value"])
                .build(FLOAT_TARGET),
            Err(BuildError::TensorDataWithoutConst)
        ));

        let missing_data = [Tensor::new("value", &[1], DType::FP32)];
        let constant = [Operator::new(OperatorKind::Const, &[], &["value"])];
        assert!(matches!(
            Graph::new("main", &missing_data, &constant, &[], &["value"]).build(FLOAT_TARGET),
            Err(BuildError::ConstWithoutTensorData)
        ));

        let empty_data = [Tensor::constant("value", &[0], DType::FP32, &[])];
        assert!(matches!(
            Graph::new("main", &empty_data, &constant, &[], &["value"]).build(FLOAT_TARGET),
            Err(BuildError::EmptyConstantData)
        ));
    }

    #[test]
    fn every_operator_kind_serializes_its_pinned_opcode_and_union_tag() {
        let cases = [
            (OperatorKind::MatMul, Op::MATMUL, 7),
            (
                OperatorKind::MaxPool2d {
                    kernel: [2, 2],
                    stride: [2, 2],
                    pad: [0; 4],
                    nan_mode: NanPropagationMode::PROPAGATE,
                },
                Op::MAX_POOL2D,
                8,
            ),
            (OperatorKind::Sigmoid, Op::SIGMOID, 13),
            (OperatorKind::Tanh, Op::TANH, 14),
            (OperatorKind::Add, Op::ADD, 15),
            (OperatorKind::LogicalAnd, Op::LOGICAL_AND, 21),
            (OperatorKind::LogicalOr, Op::LOGICAL_OR, 24),
            (OperatorKind::LogicalXor, Op::LOGICAL_XOR, 25),
            (
                OperatorKind::Maximum {
                    nan_mode: NanPropagationMode::PROPAGATE,
                },
                Op::MAXIMUM,
                26,
            ),
            (
                OperatorKind::Minimum {
                    nan_mode: NanPropagationMode::PROPAGATE,
                },
                Op::MINIMUM,
                27,
            ),
            (OperatorKind::Mul, Op::MUL, 28),
            (OperatorKind::Pow, Op::POW, 29),
            (OperatorKind::Sub, Op::SUB, 30),
            (OperatorKind::Abs, Op::ABS, 32),
            (OperatorKind::Cos, Op::COS, 36),
            (OperatorKind::Exp, Op::EXP, 37),
            (OperatorKind::Log, Op::LOG, 39),
            (OperatorKind::LogicalNot, Op::LOGICAL_NOT, 40),
            (OperatorKind::Negate, Op::NEGATE, 41),
            (OperatorKind::Reciprocal, Op::RECIPROCAL, 42),
            (OperatorKind::Rsqrt, Op::RSQRT, 43),
            (OperatorKind::Sin, Op::SIN, 44),
            (OperatorKind::Select, Op::SELECT, 45),
            (OperatorKind::Equal, Op::EQUAL, 46),
            (OperatorKind::Greater, Op::GREATER, 47),
            (OperatorKind::GreaterEqual, Op::GREATER_EQUAL, 48),
            (OperatorKind::Reshape, Op::RESHAPE, 57),
            (OperatorKind::Cast, Op::CAST, 65),
            (OperatorKind::Const, Op::CONST, 67),
            (OperatorKind::Identity, Op::IDENTITY, 68),
            (OperatorKind::ConstShape, Op::CONST_SHAPE, 75),
        ];

        for (kind, expected_op, expected_attribute) in cases {
            let tensors = [
                Tensor::new("input", &[1], DType::FP32),
                Tensor::new("output", &[1], DType::FP32),
            ];
            let shapes = [Shape::new("target", &[1])];
            let regular_inputs = ["input"];
            let reshape_inputs = ["input", "target"];
            let tensor_output = ["output"];
            let shape_output = ["target"];
            let (inputs, outputs): (&[&str], &[&str]) = match kind {
                OperatorKind::Const => (&[], &tensor_output),
                OperatorKind::ConstShape => (&[], &shape_output),
                OperatorKind::Reshape => (&reshape_inputs, &tensor_output),
                _ => (&regular_inputs, &tensor_output),
            };
            let operator = [Operator::new(kind, inputs, outputs)];
            let graph = Graph::new("main", &tensors, &operator, &[], &[]).with_shapes(&shapes);
            let bytes = graph.serialize();
            let model = virtio_accel_tosa::parse(&bytes).unwrap();
            let parsed = model
                .regions()
                .next()
                .unwrap()
                .blocks()
                .next()
                .unwrap()
                .operators()
                .next()
                .unwrap();
            assert_eq!(parsed.op(), expected_op);
            assert_eq!(parsed.attribute_kind().get(), expected_attribute);
        }
    }

    #[test]
    fn rejects_dynamic_shapes_before_serialization() {
        let tensors = [Tensor::new("value", &[-1], DType::FP32)];
        let graph = Graph::new("main", &tensors, &[], &["value"], &["value"]);
        assert!(matches!(
            graph.build(FLOAT_TARGET),
            Err(BuildError::DynamicShape)
        ));
    }
}
