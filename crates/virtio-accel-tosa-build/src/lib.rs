//! Safe authoring of static, single-block TOSA 1.0 artifacts.
//!
//! The raw FlatBuffers layout is private to this crate. [`Graph::build`]
//! round-trips every artifact through `virtio-accel-tosa`'s bounded parser and
//! complete target validator before returning owned bytes.

#![no_std]
#![forbid(unsafe_code)]

extern crate alloc;

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

    pub const REGION_NAME: u16 = 4;
    pub const REGION_BLOCKS: u16 = 6;

    pub const GRAPH_VERSION: u16 = 4;
    pub const GRAPH_REGIONS: u16 = 6;

    pub const VERSION_MAJOR: u16 = 4;
    pub const VERSION_MINOR: u16 = 6;
    pub const VERSION_PATCH: u16 = 8;
    pub const VERSION_DRAFT: u16 = 10;

    pub const NAN_MODE: u16 = 4;
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
    Sigmoid,
    Tanh,
    Add,
    LogicalAnd,
    LogicalOr,
    LogicalXor,
    Maximum { nan_mode: NanPropagationMode },
    Minimum { nan_mode: NanPropagationMode },
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
}

impl OperatorKind {
    /// Stable TOSA opcode selected by this typed variant.
    pub const fn op(self) -> Op {
        match self {
            Self::MatMul => Op::MATMUL,
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

/// A static TOSA graph containing one region and one basic block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Graph<'a> {
    pub name: &'a str,
    pub tensors: &'a [Tensor<'a>],
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
            operators,
            inputs,
            outputs,
        }
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
    let attribute = {
        let table = builder.start_table();
        match operator.kind {
            OperatorKind::Maximum { nan_mode } | OperatorKind::Minimum { nan_mode } => {
                builder.push_slot::<u32>(slot::NAN_MODE, nan_mode.get(), 0);
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
    fn rejects_dynamic_shapes_before_serialization() {
        let tensors = [Tensor::new("value", &[-1], DType::FP32)];
        let graph = Graph::new("main", &tensors, &[], &["value"], &["value"]);
        assert!(matches!(
            graph.build(FLOAT_TARGET),
            Err(BuildError::DynamicShape)
        ));
    }
}
