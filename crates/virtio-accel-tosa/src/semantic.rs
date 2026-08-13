use alloc::vec::Vec;
use core::fmt;

use crate::{
    BasicBlock, DType, ExtensionSet, I32List, LevelLimits, Model, ModelValidator,
    NanPropagationMode, Op, OpAttributes, Operator, ProfileSet, ResizeMode, RoundingMode, Shape,
    Target, TargetError, Tensor,
};

/// Operand side used in semantic diagnostics.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OperandRole {
    Input,
    Output,
}

/// Operator-level failure after the serialization envelope has already been validated.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SemanticErrorKind {
    GraphIoMustBeTensor,
    InvalidArity {
        op: Op,
        inputs: usize,
        outputs: usize,
    },
    TensorListLimit {
        op: Op,
        actual: usize,
        limit: usize,
    },
    ExpectedTensor {
        op: Op,
        role: OperandRole,
        index: usize,
    },
    ExpectedShape {
        op: Op,
        role: OperandRole,
        index: usize,
    },
    InvalidRank {
        op: Op,
        role: OperandRole,
        index: usize,
        rank: Option<usize>,
        minimum: usize,
        maximum: usize,
    },
    UnsupportedTypeProfile(Op),
    InvalidAttribute(Op),
    InvalidShape(Op),
    ConstantRequired {
        op: Op,
        input: usize,
    },
    InvalidConstantData {
        op: Op,
        operand: usize,
    },
    InvalidTensorData,
    TensorSizeLimit,
    ShapeValueLimit,
    InvalidVariable,
    DisconnectedSymbol,
    GraphInputProduced,
    DataflowCycle,
    UnknownControlFlowRegion(Op),
    ControlFlowSignature(Op),
    ControlFlowCycle,
    ControlFlowNestingLimit {
        actual: usize,
        limit: usize,
    },
}

/// Located semantic validation failure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SemanticError {
    InvalidTarget(TargetError),
    AllocationFailed,
    Graph {
        region: usize,
        block: usize,
        operator: Option<usize>,
        kind: SemanticErrorKind,
    },
}

impl fmt::Display for SemanticError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

/// Complete stable-TOSA semantic pass for a declared device-neutral target.
#[derive(Clone, Copy, Debug)]
pub struct SemanticValidator {
    target: Target,
}

impl SemanticValidator {
    pub const fn new(target: Target) -> Self {
        Self { target }
    }

    pub const fn target(&self) -> Target {
        self.target
    }
}

impl ModelValidator for SemanticValidator {
    type Error = SemanticError;

    fn validate(&mut self, model: &Model<'_>) -> Result<(), Self::Error> {
        validate_semantics(model, self.target)
    }
}

/// Validate stable TOSA operator, profile, extension, level, shape, and numerical constraints.
pub fn validate_semantics(model: &Model<'_>, target: Target) -> Result<(), SemanticError> {
    let target = target.validate().map_err(SemanticError::InvalidTarget)?;
    if target.version != model.version() {
        return Err(SemanticError::InvalidTarget(TargetError::VersionMismatch {
            target: target.version,
            model: model.version(),
        }));
    }

    let mut region_names = Vec::new();
    region_names
        .try_reserve_exact(model.regions().len())
        .map_err(|_| SemanticError::AllocationFailed)?;
    for region in model.regions() {
        region_names.push(region.name());
    }
    region_names.sort_unstable();
    validate_control_flow_nesting(model, &region_names, target.level.limits().max_nesting)?;

    for (region_index, region) in model.regions().enumerate() {
        for (block_index, block) in region.blocks().enumerate() {
            validate_block(
                model,
                block,
                target,
                &region_names,
                region_index,
                block_index,
            )?;
        }
    }
    Ok(())
}

fn validate_control_flow_nesting(
    model: &Model<'_>,
    region_names: &[&str],
    limit: usize,
) -> Result<(), SemanticError> {
    let mut states = Vec::new();
    let mut depths = Vec::new();
    states
        .try_reserve_exact(region_names.len())
        .map_err(|_| SemanticError::AllocationFailed)?;
    depths
        .try_reserve_exact(region_names.len())
        .map_err(|_| SemanticError::AllocationFailed)?;
    for _ in region_names {
        states.push(0_u8);
        depths.push(0_usize);
    }
    for index in 0..region_names.len() {
        let depth = control_flow_depth(
            model,
            region_names,
            &mut states,
            &mut depths,
            index,
            0,
            limit,
        )
        .map_err(|kind| SemanticError::Graph {
            region: index,
            block: 0,
            operator: None,
            kind,
        })?;
        if depth > limit {
            return Err(SemanticError::Graph {
                region: index,
                block: 0,
                operator: None,
                kind: SemanticErrorKind::ControlFlowNestingLimit {
                    actual: depth,
                    limit,
                },
            });
        }
    }
    Ok(())
}

fn control_flow_depth(
    model: &Model<'_>,
    region_names: &[&str],
    states: &mut [u8],
    depths: &mut [usize],
    index: usize,
    path_depth: usize,
    limit: usize,
) -> Result<usize, SemanticErrorKind> {
    if path_depth > limit {
        return Err(SemanticErrorKind::ControlFlowNestingLimit {
            actual: path_depth,
            limit,
        });
    }
    match states[index] {
        1 => return Err(SemanticErrorKind::ControlFlowCycle),
        2 => return Ok(depths[index]),
        _ => {}
    }
    states[index] = 1;
    let mut maximum = 0_usize;
    if let Some(region) = model
        .regions()
        .find(|region| region.name() == region_names[index])
    {
        for block in region.blocks() {
            for operator in block.operators() {
                let attributes = operator.attributes();
                let children: &[Option<&str>] = match attributes {
                    OpAttributes::CondIf {
                        then_graph,
                        else_graph,
                    } => &[then_graph, else_graph],
                    OpAttributes::WhileLoop {
                        cond_graph,
                        body_graph,
                    } => &[cond_graph, body_graph],
                    _ => &[],
                };
                for child in children.iter().flatten() {
                    let Ok(child_index) = region_names.binary_search(child) else {
                        continue;
                    };
                    let child_depth = control_flow_depth(
                        model,
                        region_names,
                        states,
                        depths,
                        child_index,
                        path_depth + 1,
                        limit,
                    )?;
                    maximum = maximum.max(child_depth.saturating_add(1));
                }
            }
        }
    }
    states[index] = 2;
    depths[index] = maximum;
    Ok(maximum)
}

#[derive(Clone, Copy)]
enum Symbol<'a> {
    Tensor(Tensor<'a>),
    Shape(Shape<'a>),
}

fn validate_block<'a>(
    model: &Model<'a>,
    block: BasicBlock<'a>,
    target: Target,
    region_names: &[&str],
    region_index: usize,
    block_index: usize,
) -> Result<(), SemanticError> {
    let symbol_count = block.tensors().len() + block.shapes().len();
    let mut symbols = Vec::new();
    symbols
        .try_reserve_exact(symbol_count)
        .map_err(|_| SemanticError::AllocationFailed)?;
    for tensor in block.tensors() {
        validate_tensor_encoding(model, tensor).map_err(|kind| SemanticError::Graph {
            region: region_index,
            block: block_index,
            operator: None,
            kind,
        })?;
        if !tensor_within_level(tensor, target.level.limits()) {
            return Err(SemanticError::Graph {
                region: region_index,
                block: block_index,
                operator: None,
                kind: SemanticErrorKind::TensorSizeLimit,
            });
        }
        if (tensor.is_variable()
            && (!target.extensions.contains(ExtensionSet::VARIABLE)
                || tensor.variable_name().is_none_or(str::is_empty)))
            || (!tensor.is_variable()
                && tensor.variable_name().is_some_and(|name| !name.is_empty()))
        {
            return Err(SemanticError::Graph {
                region: region_index,
                block: block_index,
                operator: None,
                kind: SemanticErrorKind::InvalidVariable,
            });
        }
        symbols.push((tensor.name(), Symbol::Tensor(tensor)));
    }
    for shape in block.shapes() {
        if !shape_within_level(shape, target.level.limits()) {
            return Err(SemanticError::Graph {
                region: region_index,
                block: block_index,
                operator: None,
                kind: SemanticErrorKind::ShapeValueLimit,
            });
        }
        symbols.push((shape.name(), Symbol::Shape(shape)));
    }
    symbols.sort_unstable_by_key(|(name, _)| *name);

    let constant_count = block
        .operators()
        .filter(|operator| matches!(operator.op(), Op::CONST | Op::CONST_SHAPE))
        .try_fold(0_usize, |count, operator| {
            count.checked_add(operator.outputs().len())
        })
        .ok_or(SemanticError::AllocationFailed)?;
    let mut constants = Vec::new();
    constants
        .try_reserve_exact(constant_count)
        .map_err(|_| SemanticError::AllocationFailed)?;
    for operator in block.operators() {
        if matches!(operator.op(), Op::CONST | Op::CONST_SHAPE) {
            constants.extend(operator.outputs());
        }
    }
    constants.sort_unstable();

    for name in block.inputs().chain(block.outputs()) {
        if !matches!(resolve(&symbols, name), Symbol::Tensor(_)) {
            return Err(SemanticError::Graph {
                region: region_index,
                block: block_index,
                operator: None,
                kind: SemanticErrorKind::GraphIoMustBeTensor,
            });
        }
    }
    validate_dataflow(block, &symbols, region_index, block_index)?;

    let mut inputs = Vec::new();
    let mut outputs = Vec::new();
    for (operator_index, operator) in block.operators().enumerate() {
        inputs.clear();
        outputs.clear();
        inputs
            .try_reserve(operator.inputs().len())
            .map_err(|_| SemanticError::AllocationFailed)?;
        outputs
            .try_reserve(operator.outputs().len())
            .map_err(|_| SemanticError::AllocationFailed)?;
        for name in operator.inputs() {
            inputs.push(resolve(&symbols, name));
        }
        for name in operator.outputs() {
            outputs.push(resolve(&symbols, name));
        }

        validate_operator(
            model,
            operator,
            &inputs,
            &outputs,
            target,
            region_names,
            &constants,
        )
        .map_err(|kind| SemanticError::Graph {
            region: region_index,
            block: block_index,
            operator: Some(operator_index),
            kind,
        })?;
    }
    Ok(())
}

fn resolve<'a>(symbols: &[(&'a str, Symbol<'a>)], name: &str) -> Symbol<'a> {
    let index = symbols
        .binary_search_by_key(&name, |(candidate, _)| *candidate)
        .expect("structurally validated symbol reference");
    symbols[index].1
}

fn validate_dataflow<'a>(
    block: BasicBlock<'a>,
    symbols: &[(&'a str, Symbol<'a>)],
    region_index: usize,
    block_index: usize,
) -> Result<(), SemanticError> {
    let allocation = |_| SemanticError::AllocationFailed;
    let located = |kind| SemanticError::Graph {
        region: region_index,
        block: block_index,
        operator: None,
        kind,
    };
    let operator_count = block.operators().len();

    let mut sources = Vec::new();
    let source_capacity = block
        .inputs()
        .len()
        .checked_add(block.tensors().len())
        .ok_or(SemanticError::AllocationFailed)?;
    sources
        .try_reserve_exact(source_capacity)
        .map_err(allocation)?;
    sources.extend(block.inputs());
    sources.extend(
        block
            .tensors()
            .filter(Tensor::is_variable)
            .map(|tensor| tensor.name()),
    );
    sources.sort_unstable();

    let mut producers = Vec::new();
    producers
        .try_reserve_exact(symbols.len())
        .map_err(allocation)?;
    let mut edge_capacity = 0_usize;
    for (operator_index, operator) in block.operators().enumerate() {
        edge_capacity = edge_capacity
            .checked_add(operator.inputs().len())
            .ok_or(SemanticError::AllocationFailed)?;
        for name in operator.outputs() {
            if !matches!(resolve(symbols, name), Symbol::Tensor(value) if value.is_variable()) {
                if sources.binary_search(&name).is_ok() {
                    return Err(located(SemanticErrorKind::GraphInputProduced));
                }
                producers.push((name, operator_index));
            }
        }
    }
    producers.sort_unstable_by_key(|(name, _)| *name);

    let mut indegrees = Vec::new();
    indegrees
        .try_reserve_exact(operator_count)
        .map_err(allocation)?;
    indegrees.resize(operator_count, 0_usize);
    let mut edges = Vec::new();
    edges.try_reserve_exact(edge_capacity).map_err(allocation)?;
    for (consumer, operator) in block.operators().enumerate() {
        for name in operator.inputs() {
            if sources.binary_search(&name).is_ok() {
                continue;
            }
            let producer = producers
                .binary_search_by_key(&name, |(candidate, _)| *candidate)
                .ok()
                .map(|index| producers[index].1)
                .ok_or_else(|| located(SemanticErrorKind::DisconnectedSymbol))?;
            edges.push((producer, consumer));
            indegrees[consumer] = indegrees[consumer]
                .checked_add(1)
                .ok_or(SemanticError::AllocationFailed)?;
        }
    }
    for name in block.outputs() {
        if sources.binary_search(&name).is_err()
            && producers
                .binary_search_by_key(&name, |(candidate, _)| *candidate)
                .is_err()
        {
            return Err(located(SemanticErrorKind::DisconnectedSymbol));
        }
    }

    edges.sort_unstable();
    let mut ready = Vec::new();
    ready
        .try_reserve_exact(operator_count)
        .map_err(allocation)?;
    ready.extend(
        indegrees
            .iter()
            .enumerate()
            .filter_map(|(index, indegree)| (*indegree == 0).then_some(index)),
    );
    let mut cursor = 0_usize;
    while cursor < ready.len() {
        let producer = ready[cursor];
        cursor += 1;
        let start = edges.partition_point(|(candidate, _)| *candidate < producer);
        let end = edges.partition_point(|(candidate, _)| *candidate <= producer);
        for &(_, consumer) in &edges[start..end] {
            indegrees[consumer] -= 1;
            if indegrees[consumer] == 0 {
                ready.push(consumer);
            }
        }
    }
    if cursor == operator_count {
        Ok(())
    } else {
        Err(located(SemanticErrorKind::DataflowCycle))
    }
}

fn validate_operator(
    model: &Model<'_>,
    operator: Operator<'_>,
    inputs: &[Symbol<'_>],
    outputs: &[Symbol<'_>],
    target: Target,
    region_names: &[&str],
    constants: &[&str],
) -> Result<(), SemanticErrorKind> {
    let op = operator.op();
    let arity = op.arity().expect("stable operator has an arity");
    if !arity.accepts(inputs.len(), outputs.len()) {
        return Err(SemanticErrorKind::InvalidArity {
            op,
            inputs: inputs.len(),
            outputs: outputs.len(),
        });
    }
    validate_tensor_list_limit(op, inputs.len(), outputs.len(), target.level.limits())?;
    validate_operand_kinds(op, inputs, outputs)?;
    validate_ranks(op, inputs, outputs, target.level.limits())?;
    validate_type_profile(op, operator.attributes(), inputs, outputs, target)?;
    validate_attributes(op, operator.attributes(), inputs, outputs, target)?;
    validate_ctc_inputs(
        model,
        op,
        operator,
        operator.attributes(),
        inputs,
        constants,
        target,
    )?;
    validate_shapes(
        model,
        op,
        operator.attributes(),
        inputs,
        outputs,
        target,
        region_names,
    )
}

fn validate_tensor_list_limit(
    op: Op,
    inputs: usize,
    outputs: usize,
    limits: LevelLimits,
) -> Result<(), SemanticErrorKind> {
    let actual = match op.get() {
        55 => inputs,
        69 => inputs.max(outputs),
        70 => inputs.saturating_sub(1).max(outputs),
        71 => inputs.max(outputs),
        _ => return Ok(()),
    };
    if actual > limits.max_tensor_list_size {
        Err(SemanticErrorKind::TensorListLimit {
            op,
            actual,
            limit: limits.max_tensor_list_size,
        })
    } else {
        Ok(())
    }
}

fn validate_operand_kinds(
    op: Op,
    inputs: &[Symbol<'_>],
    outputs: &[Symbol<'_>],
) -> Result<(), SemanticErrorKind> {
    let shape_inputs: &[usize] = match op.get() {
        56 | 57 | 60 => &[1],
        59 => &[1, 2],
        64 => &[1, 2, 3],
        _ => &[],
    };
    for (index, input) in inputs.iter().enumerate() {
        let wants_shape = shape_inputs.contains(&index);
        match (wants_shape, input) {
            (true, Symbol::Shape(_)) | (false, Symbol::Tensor(_)) => {}
            (true, _) => {
                return Err(SemanticErrorKind::ExpectedShape {
                    op,
                    role: OperandRole::Input,
                    index,
                });
            }
            (false, _) => {
                return Err(SemanticErrorKind::ExpectedTensor {
                    op,
                    role: OperandRole::Input,
                    index,
                });
            }
        }
    }
    for (index, output) in outputs.iter().enumerate() {
        let wants_shape = op == Op::CONST_SHAPE;
        match (wants_shape, output) {
            (true, Symbol::Shape(_)) | (false, Symbol::Tensor(_)) => {}
            (true, _) => {
                return Err(SemanticErrorKind::ExpectedShape {
                    op,
                    role: OperandRole::Output,
                    index,
                });
            }
            (false, _) => {
                return Err(SemanticErrorKind::ExpectedTensor {
                    op,
                    role: OperandRole::Output,
                    index,
                });
            }
        }
    }
    Ok(())
}

fn tensor(symbol: Symbol<'_>) -> Tensor<'_> {
    match symbol {
        Symbol::Tensor(tensor) => tensor,
        Symbol::Shape(_) => unreachable!("operand kind validated"),
    }
}

fn shape(symbol: Symbol<'_>) -> Shape<'_> {
    match symbol {
        Symbol::Shape(shape) => shape,
        Symbol::Tensor(_) => unreachable!("operand kind validated"),
    }
}

fn validate_tensor_encoding(
    model: &Model<'_>,
    tensor: Tensor<'_>,
) -> Result<(), SemanticErrorKind> {
    if tensor.rank().is_none() {
        return Ok(());
    }
    let mut elements = 1_usize;
    for dimension in tensor.dimensions() {
        elements = elements
            .checked_mul(dimension as usize)
            .ok_or(SemanticErrorKind::InvalidTensorData)?;
    }
    let data = tensor_data(model, tensor);
    if !data.is_empty() {
        let required = if tensor.dtype() == DType::INT4 {
            elements
                .checked_add(1)
                .ok_or(SemanticErrorKind::InvalidTensorData)?
                / 2
        } else {
            elements
                .checked_mul(
                    dtype_width(tensor.dtype()).ok_or(SemanticErrorKind::InvalidTensorData)?,
                )
                .ok_or(SemanticErrorKind::InvalidTensorData)?
        };
        if data.len() != required {
            return Err(SemanticErrorKind::InvalidTensorData);
        }
        let valid_values = match tensor.dtype() {
            DType::BOOL => data.iter().all(|value| *value <= 1),
            DType::INT4 => (0..elements).all(|index| crate::unpack_int4(data, index) != Some(-8)),
            DType::INT48 => data.chunks_exact(8).all(|bytes| {
                let value = i64::from_le_bytes(bytes.try_into().expect("chunk size is exact"));
                (-(1_i64 << 47)..(1_i64 << 47)).contains(&value)
            }),
            _ => true,
        };
        if !valid_values {
            return Err(SemanticErrorKind::InvalidTensorData);
        }
    }
    Ok(())
}

fn tensor_data<'a>(model: &Model<'a>, tensor: Tensor<'a>) -> &'a [u8] {
    if let Some((offset, size)) = tensor.external_data_range() {
        let start = usize::try_from(offset).expect("validated external offset");
        let len = usize::try_from(size).expect("validated external size");
        &model.as_bytes()[start..start + len]
    } else {
        tensor.data()
    }
}

fn dtype_width(dtype: DType) -> Option<usize> {
    match dtype.get() {
        1..=3 | 11..=12 => Some(1),
        4 | 8 | 9 => Some(2),
        5 | 7 => Some(4),
        6 | 10 => Some(8),
        _ => None,
    }
}

fn dtype_element_bytes(dtype: DType) -> Option<usize> {
    match dtype {
        DType::INT48 => Some(6),
        _ => dtype_width(dtype),
    }
}

fn tensor_within_level(tensor: Tensor<'_>, limits: LevelLimits) -> bool {
    let Some(_) = tensor.rank() else {
        return true;
    };
    let Some(elements) = tensor.dimensions().try_fold(1_u128, |count, dimension| {
        count.checked_mul(dimension as u128)
    }) else {
        return false;
    };
    let Some(bytes) =
        dtype_element_bytes(tensor.dtype()).and_then(|width| elements.checked_mul(width as u128))
    else {
        return false;
    };
    let maximum = (1_u128 << limits.max_log2_size) - 1;
    bytes <= maximum
}

fn shape_within_level(shape: Shape<'_>, limits: LevelLimits) -> bool {
    let Some(mut values) = shape.values() else {
        return true;
    };
    let magnitude = 1_i128 << limits.max_log2_size;
    values.all(|value| i128::from(value) >= -magnitude && i128::from(value) < magnitude)
}

// Implemented below in rule-focused sections so every stable opcode is mechanically covered.
fn validate_ranks(
    op: Op,
    inputs: &[Symbol<'_>],
    outputs: &[Symbol<'_>],
    limits: LevelLimits,
) -> Result<(), SemanticErrorKind> {
    for (role, operands) in [(OperandRole::Input, inputs), (OperandRole::Output, outputs)] {
        for (index, operand) in operands.iter().copied().enumerate() {
            let Symbol::Tensor(value) = operand else {
                continue;
            };
            let rank = value.rank();
            if rank.is_none_or(|rank| rank > limits.max_rank) {
                return Err(SemanticErrorKind::InvalidRank {
                    op,
                    role,
                    index,
                    rank,
                    minimum: 0,
                    maximum: limits.max_rank,
                });
            }
        }
    }

    let (input_ranks, output_ranks): (&[Option<usize>], &[Option<usize>]) = match op.get() {
        1 => (&[None], &[None]),
        2 => (&[Some(4), Some(1), Some(1)], &[Some(4)]),
        3 | 5 | 10 => (&[Some(4), Some(4), Some(1), Some(1), Some(1)], &[Some(4)]),
        4 => (&[Some(5), Some(5), Some(1), Some(1), Some(1)], &[Some(5)]),
        6 => (&[Some(3), Some(3)], &[Some(3), Some(3)]),
        7 => (&[Some(3), Some(3), Some(1), Some(1)], &[Some(3)]),
        8 => (&[Some(4)], &[Some(4)]),
        9 => (&[Some(3)], &[Some(3), Some(3)]),
        28 => (&[None, None, Some(1)], &[None]),
        31 => (&[None, Some(1)], &[None]),
        41 => (&[None, Some(1), Some(1)], &[None]),
        49..=54 => (&[None], &[None]),
        55 => (&[], &[None]),
        56 => (&[None, None, Some(1)], &[None]),
        57 => (&[None, None], &[None]),
        58 => (&[None], &[None]),
        59 => (&[None, None, None], &[None]),
        60 => (&[None, None], &[None]),
        61 => (&[None], &[None]),
        62 => (&[Some(3), Some(2)], &[Some(3)]),
        63 => (&[Some(3), Some(2), Some(3)], &[Some(3)]),
        64 => (&[Some(4), None, None, None], &[Some(4)]),
        66 => (&[None, Some(1), Some(1), Some(1), Some(1)], &[None]),
        _ => (&[], &[]),
    };
    for (index, expected) in input_ranks.iter().copied().enumerate() {
        if let Some(expected) = expected {
            require_rank(
                op,
                OperandRole::Input,
                index,
                inputs[index],
                expected,
                expected,
            )?;
        }
    }
    for (index, expected) in output_ranks.iter().copied().enumerate() {
        if let Some(expected) = expected {
            require_rank(
                op,
                OperandRole::Output,
                index,
                outputs[index],
                expected,
                expected,
            )?;
        }
    }

    let minimum_rank = match op.get() {
        1 | 49..=56 | 58..=61 => 1,
        _ => 0,
    };
    if minimum_rank != 0 {
        for (role, operands) in [(OperandRole::Input, inputs), (OperandRole::Output, outputs)] {
            for (index, operand) in operands.iter().copied().enumerate() {
                if matches!(operand, Symbol::Tensor(_)) {
                    require_rank(op, role, index, operand, minimum_rank, limits.max_rank)?;
                }
            }
        }
    }
    Ok(())
}

fn validate_type_profile(
    op: Op,
    attributes: OpAttributes<'_>,
    inputs: &[Symbol<'_>],
    outputs: &[Symbol<'_>],
    target: Target,
) -> Result<(), SemanticErrorKind> {
    let i = |index| tensor(inputs[index]).dtype();
    let o = |index| tensor(outputs[index]).dtype();
    let same = || tensor_types_equal(inputs, outputs);
    let ok = match op.get() {
        1 => o(0) == DType::INT32 && supports_argmax(i(0), target),
        2 => {
            let acc = match attributes {
                OpAttributes::AvgPool2d { acc_type, .. } => acc_type,
                _ => unreachable!(),
            };
            i(0) == i(1) && i(0) == i(2) && i(0) == o(0) && supports_pool(i(0), acc, target)
        }
        3..=5 | 10 => {
            let acc = conv_acc_type(attributes);
            i(3) == i(0)
                && i(4) == i(1)
                && i(2) == o(0)
                && supports_conv(i(0), i(1), o(0), acc, target)
        }
        6 => same() && i(0) == DType::FP32 && has_ext(target, ExtensionSet::FFT),
        7 => i(0) == i(1) && i(2) == i(0) && i(3) == i(1) && supports_matmul(i(0), o(0), target),
        8 => same() && supports_pool_value(i(0), target),
        9 => same() && i(0) == DType::FP32 && has_ext(target, ExtensionSet::FFT),
        11 => same() && supports_clamp(i(0), target),
        12..=14 | 34 | 36..=39 | 42..=44 => same() && supports_float(i(0), target),
        15 | 30 => same() && supports_add_sub(i(0), target),
        16..=19 | 33 => same() && supports_integer_bits(i(0), target),
        20 => same() && i(0) == DType::INT32 && has_any_profile(target),
        21 | 24 | 25 | 40 => same() && i(0) == DType::BOOL && has_any_profile(target),
        22 | 23 => same() && supports_base_integer(i(0), target, true),
        26 | 27 | 32 => same() && supports_ordered(i(0), target),
        28 => i(0) == i(1) && i(2) == DType::INT8 && supports_mul(i(0), o(0), target),
        29 => same() && supports_float(i(0), target),
        31 => {
            (i(0) == DType::INT8
                && i(1) == DType::INT8
                && o(0) == DType::INT8
                && has_profile(target, ProfileSet::INTEGER))
                || (i(0) == DType::INT16
                    && i(1) == DType::INT16
                    && o(0) == DType::INT32
                    && has_ext(target, ExtensionSet::INT16))
        }
        35 => same() && i(0) == DType::INT32 && has_profile(target, ProfileSet::INTEGER),
        41 => i(0) == i(1) && i(0) == i(2) && i(0) == o(0) && supports_negate(i(0), target),
        45 => i(0) == DType::BOOL && i(1) == i(2) && i(1) == o(0) && supports_select(i(1), target),
        46..=48 => i(0) == i(1) && o(0) == DType::BOOL && supports_ordered(i(0), target),
        49 | 50 => same() && i(0) == DType::BOOL && has_any_profile(target),
        51 | 52 => same() && supports_reduce_minmax(i(0), target),
        53 => same() && supports_float(i(0), target),
        54 => same() && supports_reduce_sum(i(0), target),
        55 => same() && supports_concat(i(0), target),
        56..=61 => same() && supports_data_movement(i(0), target),
        62 => i(1) == DType::INT32 && i(0) == o(0) && supports_gather(i(0), target),
        63 => i(1) == DType::INT32 && i(0) == i(2) && i(0) == o(0) && supports_gather(i(0), target),
        64 => supports_resize(i(0), o(0), target),
        65 => supports_cast(i(0), o(0), target),
        66 => {
            let scale32 = match attributes {
                OpAttributes::Rescale { scale32, .. } => scale32,
                _ => unreachable!(),
            };
            i(1) == if scale32 { DType::INT32 } else { DType::INT16 }
                && i(2) == DType::INT8
                && i(3) == i(0)
                && i(4) == o(0)
                && supports_rescale(i(0), o(0), target)
        }
        67 => supports_constant(o(0), target),
        68 => same() && supports_constant(i(0), target),
        69 => inputs
            .iter()
            .chain(outputs)
            .all(|operand| supports_constant(tensor(*operand).dtype(), target)),
        70 => {
            i(0) == DType::BOOL
                && has_ext(target, ExtensionSet::CONTROL_FLOW)
                && inputs[1..]
                    .iter()
                    .chain(outputs)
                    .all(|operand| supports_constant(tensor(*operand).dtype(), target))
        }
        71 => {
            has_ext(target, ExtensionSet::CONTROL_FLOW)
                && inputs
                    .iter()
                    .chain(outputs)
                    .all(|operand| supports_constant(tensor(*operand).dtype(), target))
        }
        72 => has_ext(target, ExtensionSet::VARIABLE),
        73 => has_ext(target, ExtensionSet::VARIABLE) && supports_variable(i(0), target),
        74 => has_ext(target, ExtensionSet::VARIABLE) && supports_variable(o(0), target),
        75 => has_any_profile(target),
        _ => false,
    };
    if ok {
        Ok(())
    } else {
        Err(SemanticErrorKind::UnsupportedTypeProfile(op))
    }
}

fn require_rank(
    op: Op,
    role: OperandRole,
    index: usize,
    operand: Symbol<'_>,
    minimum: usize,
    maximum: usize,
) -> Result<(), SemanticErrorKind> {
    let rank = match operand {
        Symbol::Tensor(value) => value.rank(),
        Symbol::Shape(value) => Some(value.rank() as usize),
    };
    if rank.is_some_and(|rank| rank >= minimum && rank <= maximum) {
        Ok(())
    } else {
        Err(SemanticErrorKind::InvalidRank {
            op,
            role,
            index,
            rank,
            minimum,
            maximum,
        })
    }
}

fn tensor_types_equal(inputs: &[Symbol<'_>], outputs: &[Symbol<'_>]) -> bool {
    let mut operands = inputs
        .iter()
        .chain(outputs)
        .filter_map(|operand| match operand {
            Symbol::Tensor(value) => Some(value.dtype()),
            Symbol::Shape(_) => None,
        });
    let Some(first) = operands.next() else {
        return true;
    };
    operands.all(|dtype| dtype == first)
}

fn has_profile(target: Target, profile: ProfileSet) -> bool {
    target.profiles.intersects(profile)
}

fn has_any_profile(target: Target) -> bool {
    target.profiles.intersects(ProfileSet::ALL)
}

fn has_ext(target: Target, extension: ExtensionSet) -> bool {
    target.extensions.contains(extension)
}

fn supports_argmax(dtype: DType, target: Target) -> bool {
    match dtype {
        DType::INT8 => has_profile(target, ProfileSet::INTEGER),
        DType::INT16 => has_ext(target, ExtensionSet::INT16),
        DType::FP8E4M3 => has_ext(target, ExtensionSet::FP8E4M3),
        DType::FP8E5M2 => has_ext(target, ExtensionSet::FP8E5M2),
        DType::FP16 | DType::FP32 => has_profile(target, ProfileSet::FLOATING_POINT),
        DType::BF16 => has_ext(target, ExtensionSet::BF16),
        _ => false,
    }
}

fn supports_pool(dtype: DType, acc: DType, target: Target) -> bool {
    match (dtype, acc) {
        (DType::INT8, DType::INT32) => has_profile(target, ProfileSet::INTEGER),
        (DType::INT16, DType::INT32) => has_ext(target, ExtensionSet::INT16),
        (DType::FP8E4M3, DType::FP16) => has_ext(target, ExtensionSet::FP8E4M3),
        (DType::FP8E5M2, DType::FP16) => has_ext(target, ExtensionSet::FP8E5M2),
        (DType::FP16, DType::FP16 | DType::FP32) => has_profile(target, ProfileSet::FLOATING_POINT),
        (DType::BF16, DType::FP32) => has_ext(target, ExtensionSet::BF16),
        (DType::FP32, DType::FP32) => has_profile(target, ProfileSet::FLOATING_POINT),
        _ => false,
    }
}

fn supports_pool_value(dtype: DType, target: Target) -> bool {
    match dtype {
        DType::INT8 => has_profile(target, ProfileSet::INTEGER),
        DType::INT16 => has_ext(target, ExtensionSet::INT16),
        DType::FP8E4M3 => has_ext(target, ExtensionSet::FP8E4M3),
        DType::FP8E5M2 => has_ext(target, ExtensionSet::FP8E5M2),
        DType::FP16 | DType::FP32 => has_profile(target, ProfileSet::FLOATING_POINT),
        DType::BF16 => has_ext(target, ExtensionSet::BF16),
        _ => false,
    }
}

fn conv_acc_type(attributes: OpAttributes<'_>) -> DType {
    match attributes {
        OpAttributes::Conv2d { acc_type, .. }
        | OpAttributes::Conv3d { acc_type, .. }
        | OpAttributes::DepthwiseConv2d { acc_type, .. }
        | OpAttributes::TransposeConv2d { acc_type, .. } => acc_type,
        _ => unreachable!(),
    }
}

fn supports_conv(input: DType, weight: DType, output: DType, acc: DType, target: Target) -> bool {
    match (input, weight, output, acc) {
        (DType::INT8, DType::INT8, DType::INT32, DType::INT32) => {
            has_profile(target, ProfileSet::INTEGER)
        }
        (DType::INT8, DType::INT4, DType::INT32, DType::INT32) => {
            has_ext(target, ExtensionSet::INT4)
        }
        (DType::INT16, DType::INT8, DType::INT48, DType::INT48) => {
            has_ext(target, ExtensionSet::INT16)
        }
        (DType::FP8E4M3, DType::FP8E4M3, DType::FP16, DType::FP16) => {
            has_ext(target, ExtensionSet::FP8E4M3)
        }
        (DType::FP8E5M2, DType::FP8E5M2, DType::FP16, DType::FP16) => {
            has_ext(target, ExtensionSet::FP8E5M2)
        }
        (DType::FP16, DType::FP16, DType::FP16, DType::FP16 | DType::FP32) => {
            has_profile(target, ProfileSet::FLOATING_POINT)
        }
        (DType::BF16, DType::BF16, DType::BF16, DType::FP32) => has_ext(target, ExtensionSet::BF16),
        (DType::FP32, DType::FP32, DType::FP32, DType::FP32) => {
            has_profile(target, ProfileSet::FLOATING_POINT)
        }
        _ => false,
    }
}

fn supports_matmul(input: DType, output: DType, target: Target) -> bool {
    match (input, output) {
        (DType::INT8, DType::INT32) => has_profile(target, ProfileSet::INTEGER),
        (DType::INT16, DType::INT48) => has_ext(target, ExtensionSet::INT16),
        (DType::FP8E4M3, DType::FP16) => has_ext(target, ExtensionSet::FP8E4M3),
        (DType::FP8E5M2, DType::FP16) => has_ext(target, ExtensionSet::FP8E5M2),
        (DType::FP16, DType::FP16 | DType::FP32) => has_profile(target, ProfileSet::FLOATING_POINT),
        (DType::BF16, DType::FP32) => has_ext(target, ExtensionSet::BF16),
        (DType::FP32, DType::FP32) => has_profile(target, ProfileSet::FLOATING_POINT),
        _ => false,
    }
}

fn supports_float(dtype: DType, target: Target) -> bool {
    match dtype {
        DType::FP16 | DType::FP32 => has_profile(target, ProfileSet::FLOATING_POINT),
        DType::BF16 => has_ext(target, ExtensionSet::BF16),
        _ => false,
    }
}

fn supports_clamp(dtype: DType, target: Target) -> bool {
    match dtype {
        DType::INT8 => has_profile(target, ProfileSet::INTEGER),
        DType::INT16 => has_ext(target, ExtensionSet::INT16),
        _ => supports_float(dtype, target),
    }
}

fn supports_add_sub(dtype: DType, target: Target) -> bool {
    (dtype == DType::INT32 && has_any_profile(target)) || supports_float(dtype, target)
}

fn supports_integer_bits(dtype: DType, target: Target) -> bool {
    matches!(dtype, DType::INT8 | DType::INT16 | DType::INT32)
        && has_profile(target, ProfileSet::INTEGER)
}

fn supports_base_integer(dtype: DType, target: Target, either_profile: bool) -> bool {
    matches!(dtype, DType::INT8 | DType::INT16 | DType::INT32)
        && if either_profile {
            has_any_profile(target)
        } else {
            has_profile(target, ProfileSet::INTEGER)
        }
}

fn supports_ordered(dtype: DType, target: Target) -> bool {
    (dtype == DType::INT32 && has_profile(target, ProfileSet::INTEGER))
        || supports_float(dtype, target)
}

fn supports_mul(input: DType, output: DType, target: Target) -> bool {
    match (input, output) {
        (DType::INT8 | DType::INT16, DType::INT32) => has_profile(target, ProfileSet::INTEGER),
        (DType::INT32, DType::INT32) => has_any_profile(target),
        (DType::FP16, DType::FP16) | (DType::FP32, DType::FP32) => {
            has_profile(target, ProfileSet::FLOATING_POINT)
        }
        (DType::BF16, DType::BF16) => has_ext(target, ExtensionSet::BF16),
        _ => false,
    }
}

fn supports_negate(dtype: DType, target: Target) -> bool {
    matches!(dtype, DType::INT8 | DType::INT16 | DType::INT32)
        && has_profile(target, ProfileSet::INTEGER)
        || supports_float(dtype, target)
}

fn supports_select(dtype: DType, target: Target) -> bool {
    match dtype {
        DType::BOOL => has_any_profile(target),
        DType::INT8 | DType::INT16 | DType::INT32 => has_profile(target, ProfileSet::INTEGER),
        _ => supports_float(dtype, target),
    }
}

fn supports_reduce_minmax(dtype: DType, target: Target) -> bool {
    matches!(dtype, DType::INT8 | DType::INT16 | DType::INT32)
        && has_profile(target, ProfileSet::INTEGER)
        || supports_float(dtype, target)
}

fn supports_reduce_sum(dtype: DType, target: Target) -> bool {
    dtype == DType::INT32 && has_profile(target, ProfileSet::INTEGER)
        || supports_float(dtype, target)
}

fn supports_concat(dtype: DType, target: Target) -> bool {
    match dtype {
        DType::BOOL => has_any_profile(target),
        DType::INT8 | DType::INT32 => has_profile(target, ProfileSet::INTEGER),
        DType::INT16 => has_ext(target, ExtensionSet::INT16),
        DType::FP8E4M3 => has_ext(target, ExtensionSet::FP8E4M3),
        DType::FP8E5M2 => has_ext(target, ExtensionSet::FP8E5M2),
        _ => supports_float(dtype, target),
    }
}

fn supports_data_movement(dtype: DType, target: Target) -> bool {
    match dtype {
        DType::BOOL => has_any_profile(target),
        DType::INT8 | DType::INT16 | DType::INT32 => has_profile(target, ProfileSet::INTEGER),
        DType::FP8E4M3 => has_ext(target, ExtensionSet::FP8E4M3),
        DType::FP8E5M2 => has_ext(target, ExtensionSet::FP8E5M2),
        _ => supports_float(dtype, target),
    }
}

fn supports_gather(dtype: DType, target: Target) -> bool {
    match dtype {
        DType::INT8 | DType::INT16 | DType::INT32 => has_profile(target, ProfileSet::INTEGER),
        DType::FP8E4M3 => has_ext(target, ExtensionSet::FP8E4M3),
        DType::FP8E5M2 => has_ext(target, ExtensionSet::FP8E5M2),
        _ => supports_float(dtype, target),
    }
}

fn supports_resize(input: DType, output: DType, target: Target) -> bool {
    match (input, output) {
        (DType::INT8, DType::INT8 | DType::INT32) => has_profile(target, ProfileSet::INTEGER),
        (DType::INT16, DType::INT16 | DType::INT48) => has_ext(target, ExtensionSet::INT16),
        (DType::FP16, DType::FP16) | (DType::FP32, DType::FP32) => {
            has_profile(target, ProfileSet::FLOATING_POINT)
        }
        (DType::BF16, DType::BF16) => has_ext(target, ExtensionSet::BF16),
        _ => false,
    }
}

fn supports_cast(input: DType, output: DType, target: Target) -> bool {
    if input == output {
        return false;
    }
    let integer = |dtype| {
        matches!(
            dtype,
            DType::BOOL | DType::INT8 | DType::INT16 | DType::INT32
        )
    };
    if integer(input) && integer(output) {
        return has_profile(target, ProfileSet::INTEGER);
    }
    match (input, output) {
        (DType::INT8 | DType::INT16 | DType::INT32, DType::FP16 | DType::FP32)
        | (DType::FP16 | DType::FP32, DType::INT8 | DType::INT16 | DType::INT32)
        | (DType::FP16, DType::FP32)
        | (DType::FP32, DType::FP16) => has_profile(target, ProfileSet::FLOATING_POINT),
        (DType::INT8 | DType::INT16 | DType::INT32, DType::BF16)
        | (DType::BF16, DType::INT8 | DType::INT16 | DType::INT32)
        | (DType::BF16, DType::FP8E4M3 | DType::FP8E5M2 | DType::FP32)
        | (DType::FP32, DType::BF16) => has_ext(target, ExtensionSet::BF16),
        (DType::FP8E4M3, DType::FP16 | DType::BF16 | DType::FP32)
        | (DType::FP16 | DType::FP32, DType::FP8E4M3) => has_ext(target, ExtensionSet::FP8E4M3),
        (DType::FP8E5M2, DType::FP16 | DType::BF16 | DType::FP32)
        | (DType::FP16 | DType::FP32, DType::FP8E5M2) => has_ext(target, ExtensionSet::FP8E5M2),
        _ => false,
    }
}

fn supports_rescale(input: DType, output: DType, target: Target) -> bool {
    let output_ok = matches!(output, DType::INT8 | DType::INT16 | DType::INT32);
    if !output_ok {
        return false;
    }
    match input {
        DType::INT8 | DType::INT16 | DType::INT32 => has_profile(target, ProfileSet::INTEGER),
        DType::INT48 => has_ext(target, ExtensionSet::INT16),
        _ => false,
    }
}

fn supports_constant(dtype: DType, target: Target) -> bool {
    match dtype {
        DType::BOOL | DType::INT8 | DType::INT16 | DType::INT32 => has_any_profile(target),
        DType::INT4 => has_ext(target, ExtensionSet::INT4),
        DType::INT48 => has_ext(target, ExtensionSet::INT16),
        DType::FP8E4M3 => has_ext(target, ExtensionSet::FP8E4M3),
        DType::FP8E5M2 => has_ext(target, ExtensionSet::FP8E5M2),
        DType::FP16 | DType::FP32 => has_profile(target, ProfileSet::FLOATING_POINT),
        DType::BF16 => has_ext(target, ExtensionSet::BF16),
        _ => false,
    }
}

fn supports_variable(dtype: DType, target: Target) -> bool {
    match dtype {
        DType::INT8 => has_profile(target, ProfileSet::INTEGER),
        DType::FP16 | DType::FP32 => has_profile(target, ProfileSet::FLOATING_POINT),
        _ => false,
    }
}

fn validate_attributes(
    op: Op,
    attributes: OpAttributes<'_>,
    inputs: &[Symbol<'_>],
    _outputs: &[Symbol<'_>],
    target: Target,
) -> Result<(), SemanticErrorKind> {
    let limits = target.level.limits();
    let input_rank = |index| tensor(inputs[index]).rank().expect("rank validated");
    let valid = match attributes {
        OpAttributes::Empty { op: attribute_op } => attribute_op == op,
        OpAttributes::ArgMax { axis, nan_mode } => {
            valid_axis(axis, input_rank(0)) && valid_nan_mode(nan_mode)
        }
        OpAttributes::AvgPool2d {
            kernel,
            stride,
            pad,
            ..
        }
        | OpAttributes::MaxPool2d {
            kernel,
            stride,
            pad,
            ..
        } => {
            list_positive_bounded(kernel, 2, limits.max_kernel)
                && list_positive_bounded(stride, 2, limits.max_stride)
                && list_nonnegative_bounded(pad, 4, limits.max_kernel)
                && pool_padding_valid(kernel, pad)
                && match attributes {
                    OpAttributes::MaxPool2d { nan_mode, .. } => valid_nan_mode(nan_mode),
                    _ => true,
                }
        }
        OpAttributes::Conv2d {
            pad,
            stride,
            dilation,
            ..
        }
        | OpAttributes::DepthwiseConv2d {
            pad,
            stride,
            dilation,
            ..
        } => {
            list_nonnegative_bounded(pad, 4, limits.max_kernel)
                && list_positive_bounded(stride, 2, limits.max_stride)
                && list_positive_bounded(dilation, 2, limits.max_stride)
        }
        OpAttributes::Conv3d {
            pad,
            stride,
            dilation,
            ..
        } => {
            list_nonnegative_bounded(pad, 6, limits.max_kernel)
                && list_positive_bounded(stride, 3, limits.max_stride)
                && list_positive_bounded(dilation, 3, limits.max_stride)
        }
        OpAttributes::Fft2d { .. } | OpAttributes::Rfft2d { .. } => true,
        OpAttributes::TransposeConv2d {
            out_pad, stride, ..
        } => {
            list_exact(out_pad, 4)
                && out_pad.iter().all(|value| value <= limits.max_kernel)
                && list_positive_bounded(stride, 2, limits.max_stride)
        }
        OpAttributes::Clamp {
            min_val,
            max_val,
            nan_mode,
        } => {
            valid_nan_mode(nan_mode)
                && scalar_bytes_valid(tensor(inputs[0]).dtype(), min_val, max_val)
        }
        OpAttributes::ArithmeticRightShift { .. } => true,
        OpAttributes::Maximum { nan_mode } | OpAttributes::Minimum { nan_mode } => {
            valid_nan_mode(nan_mode)
        }
        OpAttributes::ReduceAll { axis }
        | OpAttributes::ReduceAny { axis }
        | OpAttributes::ReduceProduct { axis }
        | OpAttributes::ReduceSum { axis }
        | OpAttributes::Concat { axis }
        | OpAttributes::Reverse { axis } => valid_axis(axis, input_rank(0)),
        OpAttributes::ReduceMax { axis, nan_mode } | OpAttributes::ReduceMin { axis, nan_mode } => {
            valid_axis(axis, input_rank(0)) && valid_nan_mode(nan_mode)
        }
        OpAttributes::Transpose { perms } => valid_permutation(perms, input_rank(0)),
        OpAttributes::Resize { mode } => valid_resize_mode(mode),
        OpAttributes::Rescale {
            scale32,
            rounding_mode,
            per_channel,
            input_unsigned,
            output_unsigned,
        } => {
            let input = tensor(inputs[0]).dtype();
            let output = match _outputs[0] {
                Symbol::Tensor(value) => value.dtype(),
                Symbol::Shape(_) => unreachable!(),
            };
            valid_rounding_mode(rounding_mode)
                && (rounding_mode != RoundingMode::DOUBLE_ROUND
                    || target.extensions.contains(ExtensionSet::DOUBLE_ROUND))
                && (rounding_mode != RoundingMode::INEXACT_ROUND
                    || target.extensions.contains(ExtensionSet::INEXACT_ROUND))
                && !(scale32 && input == DType::INT48)
                && !(!scale32 && rounding_mode == RoundingMode::DOUBLE_ROUND)
                && !(input_unsigned && output_unsigned)
                && !(output == DType::INT32 && input_unsigned)
                && !(matches!(input, DType::INT32 | DType::INT48) && input_unsigned)
                && !(matches!(input, DType::INT32 | DType::INT48) && output_unsigned)
                && !(output == DType::INT32 && output_unsigned)
                && (!per_channel || input_rank(0) >= 1)
        }
        OpAttributes::Custom {
            operator_name,
            domain_name,
            ..
        } => {
            operator_name.is_some_and(|name| !name.is_empty())
                && domain_name.is_some_and(|name| !name.is_empty())
        }
        OpAttributes::CondIf {
            then_graph,
            else_graph,
        } => {
            then_graph.is_some_and(|name| !name.is_empty())
                && else_graph.is_some_and(|name| !name.is_empty())
        }
        OpAttributes::WhileLoop {
            cond_graph,
            body_graph,
        } => {
            cond_graph.is_some_and(|name| !name.is_empty())
                && body_graph.is_some_and(|name| !name.is_empty())
        }
    };
    if valid {
        Ok(())
    } else {
        Err(SemanticErrorKind::InvalidAttribute(op))
    }
}

fn validate_ctc_inputs(
    model: &Model<'_>,
    op: Op,
    operator: Operator<'_>,
    attributes: OpAttributes<'_>,
    inputs: &[Symbol<'_>],
    constants: &[&str],
    target: Target,
) -> Result<(), SemanticErrorKind> {
    let required: &[usize] = match op.get() {
        2 => &[1, 2],
        3..=5 | 10 => &[3, 4],
        7 => &[2, 3],
        28 => &[2],
        31 => &[1],
        41 => &[1, 2],
        56 => &[1, 2],
        57 => &[1],
        59 => &[1, 2],
        60 => &[1],
        64 => &[1, 2, 3],
        66 => &[1, 2, 3, 4],
        _ => &[],
    };
    let dynamic = target.extensions.contains(ExtensionSet::DYNAMIC);
    for &index in required {
        let connected_to_constant = operator
            .inputs()
            .nth(index)
            .is_some_and(|name| constants.binary_search(&name).is_ok());
        let present = match inputs[index] {
            Symbol::Tensor(value) => !tensor_data(model, value).is_empty(),
            Symbol::Shape(value) => value.values().is_some(),
        };
        if (!connected_to_constant || !present) && !dynamic {
            return Err(SemanticErrorKind::ConstantRequired { op, input: index });
        }
    }

    let check_zero_point = |index: usize, unsigned: bool| -> bool {
        let value = tensor(inputs[index]);
        if tensor_data(model, value).is_empty() {
            return dynamic;
        }
        value.dtype() == DType::INT8
            || (value.dtype() == DType::INT16
                && unsigned
                && tensor_data(model, value)
                    .get(..2)
                    .and_then(|bytes| <[u8; 2]>::try_from(bytes).ok())
                    .is_some_and(|bytes| matches!(u16::from_le_bytes(bytes), 0 | 32_768)))
            || constant_is_zero(model, value, 0)
    };
    let zero_points: &[usize] = match op.get() {
        2 => &[1, 2],
        3..=5 | 10 => &[3, 4],
        7 => &[2, 3],
        41 => &[1, 2],
        56 => &[2],
        _ => &[],
    };
    if zero_points
        .iter()
        .any(|index| !check_zero_point(*index, false))
    {
        return Err(SemanticErrorKind::InvalidConstantData {
            op,
            operand: *zero_points.first().unwrap_or(&0),
        });
    }

    if op == Op::MUL {
        let Some(shift) = constant_integer(model, tensor(inputs[2]), 0) else {
            return if dynamic {
                Ok(())
            } else {
                Err(SemanticErrorKind::ConstantRequired { op, input: 2 })
            };
        };
        if !(0..=63).contains(&shift) || (tensor(inputs[0]).dtype() != DType::INT32 && shift != 0) {
            return Err(SemanticErrorKind::InvalidConstantData { op, operand: 2 });
        }
    }

    if op == Op::RESCALE {
        let OpAttributes::Rescale {
            per_channel,
            input_unsigned,
            output_unsigned,
            ..
        } = attributes
        else {
            unreachable!()
        };
        if !check_zero_point(3, input_unsigned) || !check_zero_point(4, output_unsigned) {
            return Err(SemanticErrorKind::InvalidConstantData { op, operand: 3 });
        }
        let input = tensor(inputs[0]);
        let channels = if per_channel {
            dimension(input, input.rank().expect("rank validated") - 1) as usize
        } else {
            1
        };
        for index in [1, 2] {
            let value = tensor(inputs[index]);
            if dimension(value, 0) as usize != channels {
                return Err(SemanticErrorKind::InvalidConstantData { op, operand: index });
            }
        }
        if !tensor_data(model, tensor(inputs[1])).is_empty() {
            for index in 0..channels {
                if constant_integer(model, tensor(inputs[1]), index).is_none_or(|value| value < 0) {
                    return Err(SemanticErrorKind::InvalidConstantData { op, operand: 1 });
                }
            }
        }
        if !tensor_data(model, tensor(inputs[2])).is_empty() {
            for index in 0..channels {
                if constant_integer(model, tensor(inputs[2]), index)
                    .is_none_or(|value| !(2..=62).contains(&value))
                {
                    return Err(SemanticErrorKind::InvalidConstantData { op, operand: 2 });
                }
            }
        }
    }
    Ok(())
}

fn valid_axis(axis: i32, rank: usize) -> bool {
    axis >= 0 && (axis as usize) < rank
}

fn list_positive_bounded(values: I32List<'_>, length: usize, maximum: i32) -> bool {
    list_exact(values, length) && values.iter().all(|value| value >= 1 && value <= maximum)
}

fn list_nonnegative_bounded(values: I32List<'_>, length: usize, maximum: i32) -> bool {
    list_exact(values, length) && values.iter().all(|value| value >= 0 && value <= maximum)
}

fn pool_padding_valid(kernel: I32List<'_>, pad: I32List<'_>) -> bool {
    pad.get(0)
        .is_some_and(|value| value < kernel.get(0).unwrap_or(0))
        && pad
            .get(1)
            .is_some_and(|value| value < kernel.get(0).unwrap_or(0))
        && pad
            .get(2)
            .is_some_and(|value| value < kernel.get(1).unwrap_or(0))
        && pad
            .get(3)
            .is_some_and(|value| value < kernel.get(1).unwrap_or(0))
}

fn valid_permutation(perms: I32List<'_>, rank: usize) -> bool {
    if perms.len() != rank {
        return false;
    }
    for index in 0..rank {
        let Some(value) = perms.get(index) else {
            return false;
        };
        if !valid_axis(value, rank) || (0..index).any(|prior| perms.get(prior) == Some(value)) {
            return false;
        }
    }
    true
}

fn scalar_bytes_valid(dtype: DType, minimum: &[u8], maximum: &[u8]) -> bool {
    let Some(width) = dtype_width(dtype) else {
        return false;
    };
    if minimum.len() != width || maximum.len() != width {
        return false;
    }
    match dtype {
        DType::INT8 => (minimum[0] as i8) <= (maximum[0] as i8),
        DType::INT16 => {
            i16::from_le_bytes([minimum[0], minimum[1]])
                <= i16::from_le_bytes([maximum[0], maximum[1]])
        }
        DType::FP16 => {
            let min = f16_to_f32(u16::from_le_bytes([minimum[0], minimum[1]]));
            let max = f16_to_f32(u16::from_le_bytes([maximum[0], maximum[1]]));
            !min.is_nan() && !max.is_nan() && min <= max
        }
        DType::BF16 => {
            let min = f32::from_bits(u32::from(u16::from_le_bytes([minimum[0], minimum[1]])) << 16);
            let max = f32::from_bits(u32::from(u16::from_le_bytes([maximum[0], maximum[1]])) << 16);
            !min.is_nan() && !max.is_nan() && min <= max
        }
        DType::FP32 => {
            let min = f32::from_le_bytes(minimum.try_into().expect("length validated"));
            let max = f32::from_le_bytes(maximum.try_into().expect("length validated"));
            !min.is_nan() && !max.is_nan() && min <= max
        }
        _ => false,
    }
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exponent = (bits >> 10) & 0x1f;
    let fraction = u32::from(bits & 0x03ff);
    let converted = match exponent {
        0 if fraction == 0 => sign,
        0 => {
            let shift = fraction.leading_zeros() - 21;
            let normalized = fraction << shift;
            sign | ((127 - 15 - shift + 1) << 23) | ((normalized & 0x03ff) << 13)
        }
        0x1f => sign | 0x7f80_0000 | (fraction << 13),
        _ => sign | ((u32::from(exponent) + 127 - 15) << 23) | (fraction << 13),
    };
    f32::from_bits(converted)
}

fn constant_integer(model: &Model<'_>, value: Tensor<'_>, index: usize) -> Option<i64> {
    let data = tensor_data(model, value);
    let start = index.checked_mul(dtype_width(value.dtype())?)?;
    match value.dtype() {
        DType::INT4 => Some(i64::from(crate::unpack_int4(data, index)?)),
        DType::INT8 => Some(i64::from(*data.get(start)? as i8)),
        DType::INT16 => Some(i64::from(i16::from_le_bytes(
            data.get(start..start + 2)?.try_into().ok()?,
        ))),
        DType::INT32 => Some(i64::from(i32::from_le_bytes(
            data.get(start..start + 4)?.try_into().ok()?,
        ))),
        DType::INT48 => Some(i64::from_le_bytes(
            data.get(start..start + 8)?.try_into().ok()?,
        )),
        _ => None,
    }
}

fn constant_is_zero(model: &Model<'_>, value: Tensor<'_>, index: usize) -> bool {
    let data = tensor_data(model, value);
    let Some(width) = dtype_width(value.dtype()) else {
        return false;
    };
    let Some(start) = index.checked_mul(width) else {
        return false;
    };
    match value.dtype() {
        DType::INT4 | DType::INT8 | DType::INT16 | DType::INT32 | DType::INT48 => {
            constant_integer(model, value, index) == Some(0)
        }
        DType::FP8E4M3 | DType::FP8E5M2 => data.get(start).is_some_and(|bits| bits & 0x7f == 0),
        DType::FP16 | DType::BF16 => data
            .get(start..start + 2)
            .and_then(|bytes| <[u8; 2]>::try_from(bytes).ok())
            .is_some_and(|bytes| u16::from_le_bytes(bytes) & 0x7fff == 0),
        DType::FP32 => data
            .get(start..start + 4)
            .and_then(|bytes| <[u8; 4]>::try_from(bytes).ok())
            .is_some_and(|bytes| u32::from_le_bytes(bytes) & 0x7fff_ffff == 0),
        _ => false,
    }
}

fn validate_shapes(
    model: &Model<'_>,
    op: Op,
    attributes: OpAttributes<'_>,
    inputs: &[Symbol<'_>],
    outputs: &[Symbol<'_>],
    target: Target,
    region_names: &[&str],
) -> Result<(), SemanticErrorKind> {
    let i = |index| tensor(inputs[index]);
    let o = |index| tensor(outputs[index]);
    let dynamic = target.extensions.contains(ExtensionSet::DYNAMIC);
    let valid = match op.get() {
        1 => {
            let OpAttributes::ArgMax { axis, .. } = attributes else {
                unreachable!()
            };
            shape_without_axis(i(0), o(0), axis as usize)
        }
        2 => {
            let OpAttributes::AvgPool2d {
                kernel,
                stride,
                pad,
                ..
            } = attributes
            else {
                unreachable!()
            };
            pool_shape(i(0), o(0), kernel, stride, pad) && scalar_shape(i(1)) && scalar_shape(i(2))
        }
        3 | 5 => {
            let (pad, stride, dilation) = match attributes {
                OpAttributes::Conv2d {
                    pad,
                    stride,
                    dilation,
                    ..
                }
                | OpAttributes::DepthwiseConv2d {
                    pad,
                    stride,
                    dilation,
                    ..
                } => (pad, stride, dilation),
                _ => unreachable!(),
            };
            conv2d_shape(
                Conv2dGeometry {
                    op,
                    input: i(0),
                    weight: i(1),
                    bias: i(2),
                    output: o(0),
                    pad,
                    stride,
                    dilation,
                },
                target,
            ) && scalar_shape(i(3))
                && scalar_shape(i(4))
        }
        4 => {
            let OpAttributes::Conv3d {
                pad,
                stride,
                dilation,
                ..
            } = attributes
            else {
                unreachable!()
            };
            conv3d_shape(
                Conv3dGeometry {
                    input: i(0),
                    weight: i(1),
                    bias: i(2),
                    output: o(0),
                    pad,
                    stride,
                    dilation,
                },
                target,
            ) && scalar_shape(i(3))
                && scalar_shape(i(4))
        }
        6 => {
            same_shape(i(0), i(1))
                && same_shape(i(0), o(0))
                && same_shape(i(0), o(1))
                && fft_shape(i(0), target.level.limits())
        }
        7 => {
            dimension(i(0), 0) == dimension(i(1), 0)
                && dimension(i(0), 2) == dimension(i(1), 1)
                && dimensions_equal(
                    o(0),
                    &[dimension(i(0), 0), dimension(i(0), 1), dimension(i(1), 2)],
                )
                && scalar_shape(i(2))
                && scalar_shape(i(3))
        }
        8 => {
            let OpAttributes::MaxPool2d {
                kernel,
                stride,
                pad,
                ..
            } = attributes
            else {
                unreachable!()
            };
            pool_shape(i(0), o(0), kernel, stride, pad)
        }
        9 => {
            same_prefix(i(0), o(0), 2)
                && same_shape(o(0), o(1))
                && dimension(o(0), 2) == dimension(i(0), 2) / 2 + 1
                && fft_shape(i(0), target.level.limits())
        }
        10 => {
            let OpAttributes::TransposeConv2d {
                out_pad, stride, ..
            } = attributes
            else {
                unreachable!()
            };
            transpose_conv2d_shape(i(0), i(1), i(2), o(0), out_pad, stride, target)
                && scalar_shape(i(3))
                && scalar_shape(i(4))
        }
        11..=14 | 32..=44 | 58 | 65 | 66 | 68 => same_shape(i(0), o(0)),
        15..=27 | 29 | 30 => broadcast_shape(&[i(0), i(1)], o(0)),
        28 => broadcast_shape(&[i(0), i(1)], o(0)) && scalar_shape(i(2)),
        31 => {
            same_shape(i(0), o(0))
                && dimension(i(1), 0)
                    == if i(0).dtype() == DType::INT8 {
                        256
                    } else {
                        513
                    }
        }
        45 => broadcast_shape(&[i(0), i(1), i(2)], o(0)),
        46..=48 => broadcast_shape(&[i(0), i(1)], o(0)),
        49..=54 => {
            let axis = reduction_axis(attributes);
            reduced_shape(i(0), o(0), axis as usize)
        }
        55 => {
            let OpAttributes::Concat { axis } = attributes else {
                unreachable!()
            };
            concat_shape(inputs, o(0), axis as usize)
        }
        56 => pad_shape(i(0), shape(inputs[1]), i(2), o(0), dynamic),
        57 => reshape_shape(i(0), shape(inputs[1]), o(0), dynamic),
        59 => slice_shape(i(0), shape(inputs[1]), shape(inputs[2]), o(0), dynamic),
        60 => tile_shape(i(0), shape(inputs[1]), o(0), dynamic),
        61 => {
            let OpAttributes::Transpose { perms } = attributes else {
                unreachable!()
            };
            transpose_shape(i(0), o(0), perms)
        }
        62 => {
            dimensions_equal(
                o(0),
                &[dimension(i(0), 0), dimension(i(1), 1), dimension(i(0), 2)],
            ) && dimension(i(0), 0) == dimension(i(1), 0)
        }
        63 => {
            same_shape(i(0), o(0))
                && dimension(i(0), 0) == dimension(i(1), 0)
                && dimensions_equal(
                    i(2),
                    &[dimension(i(1), 0), dimension(i(1), 1), dimension(i(0), 2)],
                )
        }
        64 => resize_shape(
            i(0),
            shape(inputs[1]),
            shape(inputs[2]),
            shape(inputs[3]),
            o(0),
            target.level.limits(),
            dynamic,
        ),
        67 => !tensor_data(model, o(0)).is_empty(),
        69 => true,
        70 => {
            let OpAttributes::CondIf {
                then_graph,
                else_graph,
            } = attributes
            else {
                unreachable!()
            };
            tensor_elements(i(0)) == Some(1)
                && control_region_exists(region_names, then_graph)
                && control_region_exists(region_names, else_graph)
                && region_matches(model, then_graph.unwrap(), &inputs[1..], outputs)
                && region_matches(model, else_graph.unwrap(), &inputs[1..], outputs)
        }
        71 => {
            let OpAttributes::WhileLoop {
                cond_graph,
                body_graph,
            } = attributes
            else {
                unreachable!()
            };
            control_region_exists(region_names, cond_graph)
                && control_region_exists(region_names, body_graph)
                && tensor_lists_same(inputs, outputs)
                && condition_region_matches(model, cond_graph.unwrap(), inputs)
                && region_matches(model, body_graph.unwrap(), inputs, outputs)
        }
        72 => true,
        73 => i(0).is_variable() && i(0).variable_name().is_some(),
        74 => o(0).is_variable() && o(0).variable_name().is_some(),
        75 => shape(outputs[0]).values().is_some(),
        _ => false,
    };
    if valid {
        Ok(())
    } else if matches!(op, Op::COND_IF | Op::WHILE_LOOP)
        && match attributes {
            OpAttributes::CondIf {
                then_graph,
                else_graph,
            } => {
                !control_region_exists(region_names, then_graph)
                    || !control_region_exists(region_names, else_graph)
            }
            OpAttributes::WhileLoop {
                cond_graph,
                body_graph,
            } => {
                !control_region_exists(region_names, cond_graph)
                    || !control_region_exists(region_names, body_graph)
            }
            _ => false,
        }
    {
        Err(SemanticErrorKind::UnknownControlFlowRegion(op))
    } else if matches!(op, Op::COND_IF | Op::WHILE_LOOP) {
        Err(SemanticErrorKind::ControlFlowSignature(op))
    } else {
        Err(SemanticErrorKind::InvalidShape(op))
    }
}

fn dimension(tensor: Tensor<'_>, index: usize) -> i32 {
    tensor
        .dimensions()
        .nth(index)
        .expect("rank and dimension validated")
}

fn dimensions_equal(tensor: Tensor<'_>, expected: &[i32]) -> bool {
    tensor.rank() == Some(expected.len()) && tensor.dimensions().eq(expected.iter().copied())
}

fn same_shape(left: Tensor<'_>, right: Tensor<'_>) -> bool {
    left.rank() == right.rank() && left.dimensions().eq(right.dimensions())
}

fn same_prefix(left: Tensor<'_>, right: Tensor<'_>, length: usize) -> bool {
    (0..length).all(|index| dimension(left, index) == dimension(right, index))
}

fn scalar_shape(value: Tensor<'_>) -> bool {
    dimensions_equal(value, &[1])
}

fn shape_without_axis(input: Tensor<'_>, output: Tensor<'_>, axis: usize) -> bool {
    output.rank() == input.rank().map(|rank| rank - 1)
        && input
            .dimensions()
            .enumerate()
            .filter_map(|(index, value)| (index != axis).then_some(value))
            .eq(output.dimensions())
}

fn checked_window_output(
    input: i32,
    before: i32,
    after: i32,
    kernel: i64,
    stride: i32,
) -> Option<i32> {
    let numerator = i64::from(input) + i64::from(before) + i64::from(after) - kernel;
    (numerator >= 0 && numerator % i64::from(stride) == 0)
        .then(|| numerator / i64::from(stride) + 1)
        .and_then(|value| i32::try_from(value).ok())
}

fn pool_shape(
    input: Tensor<'_>,
    output: Tensor<'_>,
    kernel: I32List<'_>,
    stride: I32List<'_>,
    pad: I32List<'_>,
) -> bool {
    let Some(height) = checked_window_output(
        dimension(input, 1),
        pad.get(0).unwrap(),
        pad.get(1).unwrap(),
        i64::from(kernel.get(0).unwrap()),
        stride.get(0).unwrap(),
    ) else {
        return false;
    };
    let Some(width) = checked_window_output(
        dimension(input, 2),
        pad.get(2).unwrap(),
        pad.get(3).unwrap(),
        i64::from(kernel.get(1).unwrap()),
        stride.get(1).unwrap(),
    ) else {
        return false;
    };
    dimensions_equal(
        output,
        &[dimension(input, 0), height, width, dimension(input, 3)],
    )
}

fn effective_kernel(size: i32, dilation: i32) -> Option<i64> {
    i64::from(size)
        .checked_sub(1)?
        .checked_mul(i64::from(dilation))?
        .checked_add(1)
}

fn kernel_within_level(weight: Tensor<'_>, indexes: &[usize], limits: LevelLimits) -> bool {
    indexes.iter().all(|index| {
        let value = dimension(weight, *index);
        value <= limits.max_kernel
    })
}

fn dilated_kernel_within_level(
    weight: Tensor<'_>,
    indexes: &[usize],
    dilation: I32List<'_>,
    limits: LevelLimits,
) -> bool {
    indexes.iter().enumerate().all(|(dilation_index, index)| {
        i64::from(dimension(weight, *index)) * i64::from(dilation.get(dilation_index).unwrap())
            <= i64::from(limits.max_kernel)
    })
}

struct Conv2dGeometry<'a> {
    op: Op,
    input: Tensor<'a>,
    weight: Tensor<'a>,
    bias: Tensor<'a>,
    output: Tensor<'a>,
    pad: I32List<'a>,
    stride: I32List<'a>,
    dilation: I32List<'a>,
}

fn conv2d_shape(geometry: Conv2dGeometry<'_>, target: Target) -> bool {
    let Conv2dGeometry {
        op,
        input,
        weight,
        bias,
        output,
        pad,
        stride,
        dilation,
    } = geometry;
    let limits = target.level.limits();
    if !dilated_kernel_within_level(weight, &[0, 1], dilation, limits) && op == Op::DEPTHWISE_CONV2D
        || !dilated_kernel_within_level(weight, &[1, 2], dilation, limits) && op == Op::CONV2D
    {
        return false;
    }
    let (kernel_h, kernel_w, channels, output_channels) = if op == Op::DEPTHWISE_CONV2D {
        let channels = dimension(weight, 2);
        let Some(output_channels) = channels.checked_mul(dimension(weight, 3)) else {
            return false;
        };
        (
            dimension(weight, 0),
            dimension(weight, 1),
            channels,
            output_channels,
        )
    } else {
        (
            dimension(weight, 1),
            dimension(weight, 2),
            dimension(weight, 3),
            dimension(weight, 0),
        )
    };
    let Some(height) = effective_kernel(kernel_h, dilation.get(0).unwrap()).and_then(|kernel| {
        checked_window_output(
            dimension(input, 1),
            pad.get(0).unwrap(),
            pad.get(1).unwrap(),
            kernel,
            stride.get(0).unwrap(),
        )
    }) else {
        return false;
    };
    let Some(width) = effective_kernel(kernel_w, dilation.get(1).unwrap()).and_then(|kernel| {
        checked_window_output(
            dimension(input, 2),
            pad.get(2).unwrap(),
            pad.get(3).unwrap(),
            kernel,
            stride.get(1).unwrap(),
        )
    }) else {
        return false;
    };
    dimension(input, 3) == channels
        && (dimension(bias, 0) == 1 || dimension(bias, 0) == output_channels)
        && dimensions_equal(
            output,
            &[dimension(input, 0), height, width, output_channels],
        )
}

struct Conv3dGeometry<'a> {
    input: Tensor<'a>,
    weight: Tensor<'a>,
    bias: Tensor<'a>,
    output: Tensor<'a>,
    pad: I32List<'a>,
    stride: I32List<'a>,
    dilation: I32List<'a>,
}

fn conv3d_shape(geometry: Conv3dGeometry<'_>, target: Target) -> bool {
    let Conv3dGeometry {
        input,
        weight,
        bias,
        output,
        pad,
        stride,
        dilation,
    } = geometry;
    if !dilated_kernel_within_level(weight, &[1, 2, 3], dilation, target.level.limits()) {
        return false;
    }
    let mut spatial = [0_i32; 3];
    for (index, spatial_dimension) in spatial.iter_mut().enumerate() {
        let Some(value) =
            effective_kernel(dimension(weight, index + 1), dilation.get(index).unwrap()).and_then(
                |kernel| {
                    checked_window_output(
                        dimension(input, index + 1),
                        pad.get(index * 2).unwrap(),
                        pad.get(index * 2 + 1).unwrap(),
                        kernel,
                        stride.get(index).unwrap(),
                    )
                },
            )
        else {
            return false;
        };
        *spatial_dimension = value;
    }
    let channels = dimension(weight, 4);
    let output_channels = dimension(weight, 0);
    dimension(input, 4) == channels
        && (dimension(bias, 0) == 1 || dimension(bias, 0) == output_channels)
        && dimensions_equal(
            output,
            &[
                dimension(input, 0),
                spatial[0],
                spatial[1],
                spatial[2],
                output_channels,
            ],
        )
}

fn fft_shape(input: Tensor<'_>, limits: LevelLimits) -> bool {
    [dimension(input, 1), dimension(input, 2)]
        .into_iter()
        .all(|value| {
            let value = value as u32;
            value <= limits.max_kernel as u32
                && value.is_power_of_two()
                && value.ilog2() <= limits.max_log2_size
        })
}

fn transpose_conv2d_shape(
    input: Tensor<'_>,
    weight: Tensor<'_>,
    bias: Tensor<'_>,
    output: Tensor<'_>,
    out_pad: I32List<'_>,
    stride: I32List<'_>,
    target: Target,
) -> bool {
    if !kernel_within_level(weight, &[1, 2], target.level.limits()) {
        return false;
    }
    let kernel_h = dimension(weight, 1);
    let kernel_w = dimension(weight, 2);
    if out_pad.get(0).unwrap() <= -kernel_h
        || out_pad.get(1).unwrap() <= -kernel_h
        || out_pad.get(2).unwrap() <= -kernel_w
        || out_pad.get(3).unwrap() <= -kernel_w
    {
        return false;
    }
    let calculate = |size: i32, stride: i32, before: i32, after: i32, kernel: i32| {
        i64::from(size - 1)
            .checked_mul(i64::from(stride))?
            .checked_add(i64::from(before))?
            .checked_add(i64::from(after))?
            .checked_add(i64::from(kernel))
            .and_then(|value| i32::try_from(value).ok())
    };
    let Some(height) = calculate(
        dimension(input, 1),
        stride.get(0).unwrap(),
        out_pad.get(0).unwrap(),
        out_pad.get(1).unwrap(),
        kernel_h,
    ) else {
        return false;
    };
    let Some(width) = calculate(
        dimension(input, 2),
        stride.get(1).unwrap(),
        out_pad.get(2).unwrap(),
        out_pad.get(3).unwrap(),
        kernel_w,
    ) else {
        return false;
    };
    let output_channels = dimension(weight, 0);
    dimension(input, 3) == dimension(weight, 3)
        && (dimension(bias, 0) == 1 || dimension(bias, 0) == output_channels)
        && dimensions_equal(
            output,
            &[dimension(input, 0), height, width, output_channels],
        )
}

fn broadcast_shape(inputs: &[Tensor<'_>], output: Tensor<'_>) -> bool {
    let output_rank = output.rank().expect("rank validated");
    if inputs
        .iter()
        .any(|input| input.rank().expect("rank validated") > output_rank)
    {
        return false;
    }
    for output_axis in 0..output_rank {
        let output_dimension = dimension(output, output_axis);
        let mut expected = 1;
        for input in inputs {
            let input_rank = input.rank().unwrap();
            if output_axis + input_rank >= output_rank {
                let input_axis = output_axis + input_rank - output_rank;
                let value = dimension(*input, input_axis);
                if value != 1 && expected != 1 && value != expected {
                    return false;
                }
                expected = expected.max(value);
            }
        }
        if output_dimension != expected {
            return false;
        }
    }
    true
}

fn reduction_axis(attributes: OpAttributes<'_>) -> i32 {
    match attributes {
        OpAttributes::ReduceAll { axis }
        | OpAttributes::ReduceAny { axis }
        | OpAttributes::ReduceMax { axis, .. }
        | OpAttributes::ReduceMin { axis, .. }
        | OpAttributes::ReduceProduct { axis }
        | OpAttributes::ReduceSum { axis } => axis,
        _ => unreachable!(),
    }
}

fn reduced_shape(input: Tensor<'_>, output: Tensor<'_>, axis: usize) -> bool {
    input.rank() == output.rank()
        && input
            .dimensions()
            .enumerate()
            .all(|(index, value)| dimension(output, index) == if index == axis { 1 } else { value })
}

fn concat_shape(inputs: &[Symbol<'_>], output: Tensor<'_>, axis: usize) -> bool {
    let rank = output.rank().unwrap();
    let mut axis_size = 0_i64;
    for operand in inputs {
        let value = tensor(*operand);
        if value.rank() != Some(rank) {
            return false;
        }
        for index in 0..rank {
            if index == axis {
                axis_size += i64::from(dimension(value, index));
            } else if dimension(value, index) != dimension(output, index) {
                return false;
            }
        }
    }
    i64::from(dimension(output, axis)) == axis_size
}

fn shape_value(value: Shape<'_>, index: usize) -> Option<i64> {
    value.values()?.nth(index)
}

fn pad_shape(
    input: Tensor<'_>,
    padding: Shape<'_>,
    pad: Tensor<'_>,
    output: Tensor<'_>,
    dynamic: bool,
) -> bool {
    let rank = input.rank().unwrap();
    if padding.rank() as usize != rank * 2 || !scalar_shape(pad) || output.rank() != Some(rank) {
        return false;
    }
    if padding.values().is_none() {
        return dynamic;
    }
    (0..rank).all(|index| {
        let before = shape_value(padding, index * 2);
        let after = shape_value(padding, index * 2 + 1);
        matches!((before, after), (Some(before), Some(after)) if before >= 0
            && after >= 0
            && i64::from(dimension(output, index))
                == i64::from(dimension(input, index)) + before + after)
    })
}

fn tensor_elements(value: Tensor<'_>) -> Option<u64> {
    value.dimensions().try_fold(1_u64, |count, dimension| {
        count.checked_mul(dimension as u64)
    })
}

fn reshape_shape(
    input: Tensor<'_>,
    new_shape: Shape<'_>,
    output: Tensor<'_>,
    dynamic: bool,
) -> bool {
    new_shape.rank() as usize == output.rank().unwrap()
        && tensor_elements(input) == tensor_elements(output)
        && if new_shape.values().is_some() {
            (0..output.rank().unwrap()).all(|index| {
                shape_value(new_shape, index) == Some(i64::from(dimension(output, index)))
            })
        } else {
            dynamic
        }
}

fn slice_shape(
    input: Tensor<'_>,
    start: Shape<'_>,
    size: Shape<'_>,
    output: Tensor<'_>,
    dynamic: bool,
) -> bool {
    let rank = input.rank().unwrap();
    if start.rank() as usize != rank || size.rank() as usize != rank || output.rank() != Some(rank)
    {
        return false;
    }
    if start.values().is_none() || size.values().is_none() {
        return dynamic;
    }
    (0..rank).all(|index| {
        let (Some(start), Some(size)) = (shape_value(start, index), shape_value(size, index))
        else {
            return false;
        };
        start >= 0
            && size > 0
            && start + size <= i64::from(dimension(input, index))
            && size == i64::from(dimension(output, index))
    })
}

fn tile_shape(input: Tensor<'_>, multiples: Shape<'_>, output: Tensor<'_>, dynamic: bool) -> bool {
    let rank = input.rank().unwrap();
    if multiples.rank() as usize != rank || output.rank() != Some(rank) {
        return false;
    }
    if multiples.values().is_none() {
        return dynamic;
    }
    (0..rank).all(|index| {
        shape_value(multiples, index).is_some_and(|multiple| {
            multiple >= 1
                && i64::from(dimension(input, index)) * multiple
                    == i64::from(dimension(output, index))
        })
    })
}

fn transpose_shape(input: Tensor<'_>, output: Tensor<'_>, perms: I32List<'_>) -> bool {
    input.rank() == output.rank()
        && (0..output.rank().unwrap()).all(|index| {
            dimension(input, perms.get(index).unwrap() as usize) == dimension(output, index)
        })
}

fn resize_shape(
    input: Tensor<'_>,
    scale: Shape<'_>,
    offset: Shape<'_>,
    border: Shape<'_>,
    output: Tensor<'_>,
    limits: LevelLimits,
    dynamic: bool,
) -> bool {
    if scale.rank() != 4 || offset.rank() != 2 || border.rank() != 2 {
        return false;
    }
    if scale.values().is_none() || offset.values().is_none() || border.values().is_none() {
        return dynamic
            && dimension(input, 0) == dimension(output, 0)
            && dimension(input, 3) == dimension(output, 3)
            && [
                dimension(input, 1),
                dimension(input, 2),
                dimension(output, 1),
                dimension(output, 2),
            ]
            .into_iter()
            .all(|value| value < 16_384);
    }
    let (Some(yn_), Some(yd), Some(xn), Some(xd), Some(oy), Some(ox), Some(by), Some(bx)) = (
        shape_value(scale, 0),
        shape_value(scale, 1),
        shape_value(scale, 2),
        shape_value(scale, 3),
        shape_value(offset, 0),
        shape_value(offset, 1),
        shape_value(border, 0),
        shape_value(border, 1),
    ) else {
        return false;
    };
    if yn_ <= 0
        || yd <= 0
        || xn <= 0
        || xd <= 0
        || yn_ > 2_048
        || xn > 2_048
        || i128::from(yn_) > i128::from(limits.max_scale) * i128::from(yd)
        || i128::from(xn) > i128::from(limits.max_scale) * i128::from(xd)
        || yd >= 16 * yn_
        || xd >= 16 * xn
        || !(-yn_..16 * yn_).contains(&oy)
        || !(-xn..16 * xn).contains(&ox)
        || !(-16 * yn_..yn_).contains(&by)
        || !(-16 * xn..xn).contains(&bx)
        || [
            dimension(input, 1),
            dimension(input, 2),
            dimension(output, 1),
            dimension(output, 2),
        ]
        .into_iter()
        .any(|value| value >= 16_384)
    {
        return false;
    }
    let height_numerator = (i64::from(dimension(input, 1)) - 1) * yn_ - oy + by;
    let width_numerator = (i64::from(dimension(input, 2)) - 1) * xn - ox + bx;
    height_numerator % yd == 0
        && width_numerator % xd == 0
        && dimensions_equal(
            output,
            &[
                dimension(input, 0),
                i32::try_from(height_numerator / yd + 1).unwrap_or(0),
                i32::try_from(width_numerator / xd + 1).unwrap_or(0),
                dimension(input, 3),
            ],
        )
}

fn control_region_exists(region_names: &[&str], name: Option<&str>) -> bool {
    name.is_some_and(|name| region_names.binary_search(&name).is_ok())
}

fn region_block<'a>(model: &Model<'a>, name: &str) -> Option<BasicBlock<'a>> {
    let region = model.regions().find(|region| region.name() == name)?;
    let mut blocks = region.blocks();
    let first = blocks.next()?;
    if first.name() == name {
        Some(first)
    } else {
        blocks.find(|block| block.name() == name).or(Some(first))
    }
}

fn block_tensor<'a>(block: BasicBlock<'a>, name: &str) -> Option<Tensor<'a>> {
    block.tensors().find(|value| value.name() == name)
}

fn signature_matches(block: BasicBlock<'_>, operands: &[Symbol<'_>], inputs: bool) -> bool {
    let names = if inputs {
        block.inputs()
    } else {
        block.outputs()
    };
    if names.len() != operands.len() {
        return false;
    }
    names.zip(operands).all(|(name, operand)| {
        block_tensor(block, name).is_some_and(|value| {
            let expected = tensor(*operand);
            value.dtype() == expected.dtype() && same_shape(value, expected)
        })
    })
}

fn region_matches(
    model: &Model<'_>,
    name: &str,
    inputs: &[Symbol<'_>],
    outputs: &[Symbol<'_>],
) -> bool {
    region_block(model, name).is_some_and(|block| {
        signature_matches(block, inputs, true) && signature_matches(block, outputs, false)
    })
}

fn condition_region_matches(model: &Model<'_>, name: &str, inputs: &[Symbol<'_>]) -> bool {
    let Some(block) = region_block(model, name) else {
        return false;
    };
    if !signature_matches(block, inputs, true) || block.outputs().len() != 1 {
        return false;
    }
    let Some(condition) = block
        .outputs()
        .next()
        .and_then(|name| block_tensor(block, name))
    else {
        return false;
    };
    condition.dtype() == DType::BOOL && tensor_elements(condition) == Some(1)
}

fn tensor_lists_same(inputs: &[Symbol<'_>], outputs: &[Symbol<'_>]) -> bool {
    inputs.len() == outputs.len()
        && inputs.iter().zip(outputs).all(|(input, output)| {
            let input = tensor(*input);
            let output = tensor(*output);
            input.dtype() == output.dtype() && same_shape(input, output)
        })
}

#[allow(dead_code)]
fn valid_nan_mode(mode: NanPropagationMode) -> bool {
    mode == NanPropagationMode::PROPAGATE || mode == NanPropagationMode::IGNORE
}

#[allow(dead_code)]
fn valid_resize_mode(mode: ResizeMode) -> bool {
    mode == ResizeMode::NEAREST || mode == ResizeMode::BILINEAR
}

#[allow(dead_code)]
fn valid_rounding_mode(mode: RoundingMode) -> bool {
    mode == RoundingMode::SINGLE_ROUND
        || mode == RoundingMode::INEXACT_ROUND
        || mode == RoundingMode::DOUBLE_ROUND
}

#[allow(dead_code)]
fn list_exact(values: I32List<'_>, length: usize) -> bool {
    values.len() == length
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Level, Version};

    fn complete_target() -> Target {
        Target::new(
            Version::TOSA_1_0,
            ProfileSet::ALL,
            Level::Unbounded,
            ExtensionSet::ALL,
        )
    }

    #[test]
    fn stable_opcode_and_arity_tables_are_exhaustive() {
        assert_eq!(Op::ALL.len(), 75);
        for (index, op) in Op::ALL.iter().copied().enumerate() {
            assert_eq!(op.get() as usize, index + 1);
            assert!(op.arity().is_some(), "missing arity for {op:?}");
        }
    }

    #[test]
    fn cast_matrix_matches_the_stable_specification_rows() {
        let target = complete_target();
        let dtypes = [
            DType::BOOL,
            DType::INT4,
            DType::INT8,
            DType::INT16,
            DType::INT32,
            DType::INT48,
            DType::FP32,
            DType::FP16,
            DType::BF16,
            DType::SHAPE,
            DType::FP8E4M3,
            DType::FP8E5M2,
        ];
        let rows = dtypes
            .into_iter()
            .flat_map(|input| dtypes.into_iter().map(move |output| (input, output)))
            .filter(|(input, output)| supports_cast(*input, *output, target))
            .count();
        assert_eq!(rows, 46);
    }

    #[test]
    fn half_conversion_preserves_zero_order_and_nan() {
        assert_eq!(f16_to_f32(0), 0.0);
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xbc00), -1.0);
        assert!(f16_to_f32(0x7e00).is_nan());
        assert!(f16_to_f32(0x0001) > 0.0);
    }
}
