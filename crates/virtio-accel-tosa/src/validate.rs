use alloc::vec::Vec;
use core::fmt;

use crate::generated::tosa as wire;
use crate::view::SliceSource;
use crate::{DType, Model, Op, Version};

/// Resource ceilings applied before and during graph traversal.
///
/// Every field is public so a provider can derive stricter limits from its admission policy. The
/// defaults are finite and intentionally much smaller than the FlatBuffers runtime defaults.
/// `max_rank` bounds tensor ranks; serialized `shape_t` value lists may contain up to twice that
/// many entries because `PAD` carries a before/after pair for each dimension.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Limits {
    pub max_model_bytes: usize,
    pub max_apparent_bytes: usize,
    pub max_flatbuffer_depth: usize,
    pub max_flatbuffer_tables: usize,
    pub max_regions: usize,
    pub max_blocks: usize,
    pub max_tensors: usize,
    pub max_shapes: usize,
    pub max_operators: usize,
    pub max_edges: usize,
    pub max_name_bytes: usize,
    pub max_rank: usize,
    pub max_constant_bytes: usize,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            max_model_bytes: 256 * 1024 * 1024,
            max_apparent_bytes: 512 * 1024 * 1024,
            max_flatbuffer_depth: 64,
            max_flatbuffer_tables: 1_000_000,
            max_regions: 64,
            max_blocks: 4_096,
            max_tensors: 262_144,
            max_shapes: 65_536,
            max_operators: 1_000_000,
            max_edges: 8_000_000,
            max_name_bytes: 1_024,
            max_rank: 32,
            max_constant_bytes: 256 * 1024 * 1024,
        }
    }
}

/// Counts collected during the bounded validation pass.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Stats {
    pub regions: usize,
    pub blocks: usize,
    pub tensors: usize,
    pub shapes: usize,
    pub operators: usize,
    pub edges: usize,
    pub constant_bytes: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Resource {
    ModelBytes,
    Regions,
    Blocks,
    Tensors,
    Shapes,
    Operators,
    Edges,
    NameBytes,
    Rank,
    ConstantBytes,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NameKind {
    Region,
    Block,
    Tensor,
    Shape,
    Symbol,
    Reference,
}

/// Failure returned for malformed, unsupported, or over-budget input.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    EmptyInput,
    MissingIdentifier,
    InvalidFlatbuffer,
    UnsupportedVersion {
        major: i32,
        minor: i32,
        patch: i32,
        draft: bool,
    },
    LimitExceeded {
        resource: Resource,
        limit: usize,
    },
    AllocationFailed(Resource),
    MissingName(NameKind),
    EmptyName(NameKind),
    DuplicateName(NameKind),
    UnknownDataType(u32),
    UnsupportedDataType(u32),
    RankedTensorWithoutShape,
    UnrankedTensorWithDimensions,
    InvalidDimension(i32),
    InvalidShapeData,
    ExternalDataRange,
    UnknownOperator(u32),
    UnsupportedOperator(u32),
    MissingAttribute(Op),
    AttributeMismatch {
        op: Op,
        attribute: u8,
    },
    UnknownSymbol,
    MultipleProducers,
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

/// Parse with finite production defaults.
pub fn parse(bytes: &[u8]) -> Result<Model<'_>, Error> {
    parse_with_limits(bytes, Limits::default())
}

/// Verify and parse a stable TOSA 1.0 graph under caller-selected resource ceilings.
pub fn parse_with_limits(bytes: &[u8], limits: Limits) -> Result<Model<'_>, Error> {
    if bytes.is_empty() {
        return Err(Error::EmptyInput);
    }
    check_limit(bytes.len(), limits.max_model_bytes, Resource::ModelBytes)?;
    // The FlatBuffers identifier helper asserts its eight-byte minimum instead of returning false.
    if bytes.len() < 8 || !wire::tosa_graph_buffer_has_identifier(bytes) {
        return Err(Error::MissingIdentifier);
    }

    let verifier = flatbuffers::VerifierOptions {
        max_depth: limits.max_flatbuffer_depth,
        max_tables: limits.max_flatbuffer_tables,
        max_apparent_size: limits.max_apparent_bytes,
        ignore_missing_null_terminator: false,
    };
    let graph = wire::root_as_tosa_graph_with_opts(&verifier, bytes)
        .map_err(|_| Error::InvalidFlatbuffer)?;
    let raw_version = graph.version();
    let (major, minor, patch, draft) = (
        raw_version._major(),
        raw_version._minor(),
        raw_version._patch(),
        raw_version._draft(),
    );
    if (major, minor, patch, draft) != (1, 0, 0, false) {
        return Err(Error::UnsupportedVersion {
            major,
            minor,
            patch,
            draft,
        });
    }

    let mut validator = StructuralValidator {
        bytes,
        limits,
        stats: Stats::default(),
    };
    validator.validate_graph(graph)?;

    Ok(Model {
        graph,
        bytes,
        version: Version::TOSA_1_0,
        stats: validator.stats,
        source: SliceSource(bytes),
    })
}

struct StructuralValidator<'a> {
    bytes: &'a [u8],
    limits: Limits,
    stats: Stats,
}

impl StructuralValidator<'_> {
    fn validate_graph(&mut self, graph: wire::TosaGraph<'_>) -> Result<(), Error> {
        let regions = graph.regions();
        let region_count = regions.map_or(0, |items| items.len());
        self.add(Resource::Regions, region_count)?;

        let mut region_names = Vec::new();
        reserve(&mut region_names, region_count, Resource::Regions)?;
        if let Some(regions) = regions {
            for region in regions {
                let name = self.name(region.name(), NameKind::Region)?;
                region_names.push(name);
                self.validate_region(region)?;
            }
        }
        reject_duplicates(&mut region_names, NameKind::Region)?;
        Ok(())
    }

    fn validate_region(&mut self, region: wire::TosaRegion<'_>) -> Result<(), Error> {
        let blocks = region.blocks();
        self.add(Resource::Blocks, blocks.map_or(0, |items| items.len()))?;

        let block_count = blocks.map_or(0, |items| items.len());
        let mut block_names = Vec::new();
        reserve(&mut block_names, block_count, Resource::Blocks)?;
        if let Some(blocks) = blocks {
            for block in blocks {
                let name = self.name(block.name(), NameKind::Block)?;
                block_names.push(name);
                self.validate_block(block)?;
            }
        }
        reject_duplicates(&mut block_names, NameKind::Block)?;
        Ok(())
    }

    fn validate_block(&mut self, block: wire::TosaBasicBlock<'_>) -> Result<(), Error> {
        let tensors = block.tensors();
        let shapes = block.shapes();
        let operators = block.operators();
        let tensor_count = tensors.map_or(0, |items| items.len());
        let shape_count = shapes.map_or(0, |items| items.len());
        self.add(Resource::Tensors, tensor_count)?;
        self.add(Resource::Shapes, shape_count)?;
        self.add(
            Resource::Operators,
            operators.map_or(0, |items| items.len()),
        )?;

        let symbol_count = tensor_count
            .checked_add(shape_count)
            .ok_or(Error::LimitExceeded {
                resource: Resource::Tensors,
                limit: self.limits.max_tensors,
            })?;
        let mut tensor_names = Vec::new();
        reserve(&mut tensor_names, tensor_count, Resource::Tensors)?;
        let mut variable_names = Vec::new();
        reserve(&mut variable_names, tensor_count, Resource::Tensors)?;
        if let Some(tensors) = tensors {
            for tensor in tensors {
                let name = self.name(tensor.name(), NameKind::Tensor)?;
                tensor_names.push(name);
                if tensor.variable() {
                    variable_names.push(name);
                }
                self.validate_tensor(tensor)?;
            }
        }
        reject_duplicates(&mut tensor_names, NameKind::Tensor)?;
        variable_names.sort_unstable();

        let mut shape_names = Vec::new();
        reserve(&mut shape_names, shape_count, Resource::Shapes)?;
        if let Some(shapes) = shapes {
            for shape in shapes {
                let name = self.name(shape.name(), NameKind::Shape)?;
                shape_names.push(name);
                self.validate_shape(shape)?;
            }
        }
        reject_duplicates(&mut shape_names, NameKind::Shape)?;

        let mut symbols = Vec::new();
        reserve(&mut symbols, symbol_count, Resource::Tensors)?;
        symbols.extend_from_slice(&tensor_names);
        symbols.extend_from_slice(&shape_names);
        reject_duplicates(&mut symbols, NameKind::Symbol)?;

        self.validate_references(block.inputs(), &symbols)?;
        self.validate_references(block.outputs(), &symbols)?;

        let producer_count = operators.map_or(0, |items| {
            items
                .iter()
                .try_fold(0_usize, |count, operator| {
                    count.checked_add(operator.outputs().map_or(0, |outputs| outputs.len()))
                })
                .unwrap_or(usize::MAX)
        });
        check_limit(producer_count, self.limits.max_edges, Resource::Edges)?;
        let mut produced = Vec::new();
        reserve(&mut produced, producer_count, Resource::Edges)?;
        if let Some(operators) = operators {
            for operator in operators {
                self.validate_operator(operator, &symbols, &variable_names, &mut produced)?;
            }
        }
        produced.sort_unstable();
        if produced.windows(2).any(|names| names[0] == names[1]) {
            return Err(Error::MultipleProducers);
        }
        Ok(())
    }

    fn validate_tensor(&mut self, tensor: wire::TosaTensor<'_>) -> Result<(), Error> {
        let dtype = DType::new(tensor.type_().0);
        if dtype.get() == 0 || dtype.name().is_none() {
            return Err(Error::UnknownDataType(dtype.get()));
        }
        if !dtype.is_tosa_1_0() {
            return Err(Error::UnsupportedDataType(dtype.get()));
        }

        match (tensor.is_unranked(), tensor.shape()) {
            (true, Some(shape)) if !shape.is_empty() => {
                return Err(Error::UnrankedTensorWithDimensions);
            }
            (false, None) => return Err(Error::RankedTensorWithoutShape),
            (false, Some(shape)) => {
                check_limit(shape.len(), self.limits.max_rank, Resource::Rank)?;
                for dimension in shape {
                    if dimension < 1 {
                        return Err(Error::InvalidDimension(dimension));
                    }
                }
            }
            (true, Some(_) | None) => {}
        }

        let embedded = tensor.data().map_or(0, |data| data.len());
        let offset = tensor.offset();
        let external = usize::try_from(tensor.size()).map_err(|_| Error::ExternalDataRange)?;
        if external != 0 {
            if embedded != 0 || offset <= 1 {
                return Err(Error::ExternalDataRange);
            }
            let start = usize::try_from(offset).map_err(|_| Error::ExternalDataRange)?;
            let end = start
                .checked_add(external)
                .filter(|end| *end <= self.bytes.len())
                .ok_or(Error::ExternalDataRange)?;
            let _ = end;
        } else if offset != 0 {
            return Err(Error::ExternalDataRange);
        }
        let constant_bytes = embedded.checked_add(external).ok_or(Error::LimitExceeded {
            resource: Resource::ConstantBytes,
            limit: self.limits.max_constant_bytes,
        })?;
        self.add(Resource::ConstantBytes, constant_bytes)
    }

    fn validate_shape(&mut self, shape: wire::TosaShape<'_>) -> Result<(), Error> {
        let rank = usize::try_from(shape.rank()).map_err(|_| Error::LimitExceeded {
            resource: Resource::Rank,
            limit: self.limits.max_rank,
        })?;
        let max_shape_values = self.limits.max_rank.saturating_mul(2);
        check_limit(rank, max_shape_values, Resource::Rank)?;
        let data_len = shape.data().map_or(0, |data| data.len());
        if data_len != 0 {
            let required = rank
                .checked_mul(core::mem::size_of::<i64>())
                .ok_or(Error::InvalidShapeData)?;
            if data_len != required {
                return Err(Error::InvalidShapeData);
            }
        }
        self.add(Resource::ConstantBytes, data_len)
    }

    fn validate_operator<'a>(
        &mut self,
        operator: wire::TosaOperator<'a>,
        symbols: &[&'a str],
        variable_names: &[&'a str],
        produced: &mut Vec<&'a str>,
    ) -> Result<(), Error> {
        let op = Op::new(operator.op().0);
        if op.get() == 0 || op.name().is_none() {
            return Err(Error::UnknownOperator(op.get()));
        }
        if !op.is_tosa_1_0() {
            return Err(Error::UnsupportedOperator(op.get()));
        }
        let attribute = operator.attribute_type().0;
        if u32::from(attribute) != op.get() {
            return Err(Error::AttributeMismatch { op, attribute });
        }
        if operator.attribute().is_none() {
            return Err(Error::MissingAttribute(op));
        }

        self.validate_references(operator.inputs(), symbols)?;
        self.validate_references(operator.outputs(), symbols)?;
        if let Some(outputs) = operator.outputs() {
            for output in outputs {
                if variable_names.binary_search(&output).is_err() {
                    produced.push(output);
                }
            }
        }
        Ok(())
    }

    fn validate_references<'a>(
        &mut self,
        references: Option<flatbuffers::Vector<'a, flatbuffers::ForwardsUOffset<&'a str>>>,
        symbols: &[&'a str],
    ) -> Result<(), Error> {
        self.add(Resource::Edges, references.map_or(0, |items| items.len()))?;
        if let Some(references) = references {
            for reference in references {
                self.name(Some(reference), NameKind::Reference)?;
                if symbols.binary_search(&reference).is_err() {
                    return Err(Error::UnknownSymbol);
                }
            }
        }
        Ok(())
    }

    fn name<'a>(&self, name: Option<&'a str>, kind: NameKind) -> Result<&'a str, Error> {
        let name = name.ok_or(Error::MissingName(kind))?;
        if name.is_empty() {
            return Err(Error::EmptyName(kind));
        }
        check_limit(name.len(), self.limits.max_name_bytes, Resource::NameBytes)?;
        Ok(name)
    }

    fn add(&mut self, resource: Resource, amount: usize) -> Result<(), Error> {
        let (value, limit) = match resource {
            Resource::Regions => (&mut self.stats.regions, self.limits.max_regions),
            Resource::Blocks => (&mut self.stats.blocks, self.limits.max_blocks),
            Resource::Tensors => (&mut self.stats.tensors, self.limits.max_tensors),
            Resource::Shapes => (&mut self.stats.shapes, self.limits.max_shapes),
            Resource::Operators => (&mut self.stats.operators, self.limits.max_operators),
            Resource::Edges => (&mut self.stats.edges, self.limits.max_edges),
            Resource::ConstantBytes => (
                &mut self.stats.constant_bytes,
                self.limits.max_constant_bytes,
            ),
            Resource::ModelBytes | Resource::NameBytes | Resource::Rank => unreachable!(),
        };
        *value = value
            .checked_add(amount)
            .ok_or(Error::LimitExceeded { resource, limit })?;
        check_limit(*value, limit, resource)
    }
}

fn check_limit(actual: usize, limit: usize, resource: Resource) -> Result<(), Error> {
    if actual > limit {
        Err(Error::LimitExceeded { resource, limit })
    } else {
        Ok(())
    }
}

fn reserve<T>(values: &mut Vec<T>, additional: usize, resource: Resource) -> Result<(), Error> {
    values
        .try_reserve_exact(additional)
        .map_err(|_| Error::AllocationFailed(resource))
}

fn reject_duplicates(values: &mut [&str], kind: NameKind) -> Result<(), Error> {
    values.sort_unstable();
    if values.windows(2).any(|names| names[0] == names[1]) {
        Err(Error::DuplicateName(kind))
    } else {
        Ok(())
    }
}
