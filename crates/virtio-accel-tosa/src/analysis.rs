use alloc::vec::Vec;
use core::fmt;

use crate::{
    BasicBlock, DType, ExtensionSet, Model, Op, Operator, SemanticError, Shape, Target, Tensor,
    validate_semantics,
};

macro_rules! index_type {
    ($name:ident) => {
        #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
        #[repr(transparent)]
        pub struct $name(u32);

        impl $name {
            pub const fn get(self) -> u32 {
                self.0
            }

            fn from_usize(value: usize) -> Result<Self, AnalysisError> {
                match u32::try_from(value) {
                    Ok(value) => Ok(Self(value)),
                    Err(_) => Err(AnalysisError::TooManyObjects),
                }
            }

            const fn index(self) -> usize {
                self.0 as usize
            }
        }
    };
}

index_type!(RegionId);
index_type!(BlockId);
index_type!(ValueId);
index_type!(OperatorId);

#[cfg(test)]
impl ValueId {
    pub(crate) const fn from_raw(raw: u32) -> Self {
        Self(raw)
    }
}

/// Compact half-open span into one of [`TosaAnalysis`]'s indexed slices.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AnalysisSpan {
    start: u32,
    len: u32,
}

impl AnalysisSpan {
    pub const fn start(self) -> u32 {
        self.start
    }

    pub const fn len(self) -> usize {
        self.len as usize
    }

    pub const fn is_empty(self) -> bool {
        self.len == 0
    }

    fn new(start: usize, len: usize) -> Result<Self, AnalysisError> {
        Ok(Self {
            start: u32::try_from(start).map_err(|_| AnalysisError::TooManyObjects)?,
            len: u32::try_from(len).map_err(|_| AnalysisError::TooManyObjects)?,
        })
    }

    fn range(self) -> core::ops::Range<usize> {
        let start = self.start as usize;
        start..start + self.len as usize
    }
}

/// Limits for optional analysis work. They do not relax semantic or parser limits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AnalysisOptions {
    /// Largest aggregate output that may be proposed for constant folding.
    pub max_folded_constant_bytes: u64,
}

impl Default for AnalysisOptions {
    fn default() -> Self {
        Self {
            max_folded_constant_bytes: 1024 * 1024,
        }
    }
}

/// Failure while producing a compact lowering plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AnalysisError {
    Semantic(SemanticError),
    AllocationFailed,
    TooManyObjects,
}

impl fmt::Display for AnalysisError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl From<SemanticError> for AnalysisError {
    fn from(value: SemanticError) -> Self {
        Self::Semantic(value)
    }
}

/// Zero-copy source value retained by an analyzed plan.
#[derive(Clone, Copy, Debug)]
pub enum AnalyzedValueKind<'a> {
    Tensor(Tensor<'a>),
    Shape(Shape<'a>),
}

/// How an analyzed value becomes constant for lowering.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConstantState {
    NonConstant,
    /// Bytes are already serialized in the artifact by `CONST` or `CONST_SHAPE`.
    Serialized,
    /// A pure, bounded constant subgraph can be evaluated once by the provider.
    Foldable,
}

/// One globally indexed value in a model. Names and serialized payloads remain borrowed.
#[derive(Clone, Copy, Debug)]
pub struct AnalyzedValue<'a> {
    id: ValueId,
    block: BlockId,
    name: &'a str,
    kind: AnalyzedValueKind<'a>,
    producer: Option<OperatorId>,
    constant: ConstantState,
    byte_size: Option<u64>,
    consumers: u32,
    first_use: Option<u32>,
    last_use: Option<u32>,
}

impl<'a> AnalyzedValue<'a> {
    pub const fn id(&self) -> ValueId {
        self.id
    }

    pub const fn block(&self) -> BlockId {
        self.block
    }

    pub const fn name(&self) -> &'a str {
        self.name
    }

    pub const fn kind(&self) -> AnalyzedValueKind<'a> {
        self.kind
    }

    pub const fn producer(&self) -> Option<OperatorId> {
        self.producer
    }

    pub const fn constant(&self) -> ConstantState {
        self.constant
    }

    pub const fn byte_size(&self) -> Option<u64> {
        self.byte_size
    }

    pub const fn consumers(&self) -> u32 {
        self.consumers
    }

    /// First operator execution position that reads this value within its block.
    pub const fn first_use(&self) -> Option<u32> {
        self.first_use
    }

    /// Last operator execution position that reads this value within its block.
    /// Block outputs remain live through the position immediately after the final operator.
    pub const fn last_use(&self) -> Option<u32> {
        self.last_use
    }
}

/// Conditions that a provider may need to lower or inspect at execution time.
///
/// `REQUIRE` failures make a TOSA graph unpredictable and need not be detected. Dynamic CTC
/// inputs can affect mandatory `ERROR_IF` checks and therefore are classified separately.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RuntimeCondition<'a> {
    DynamicCompileTimeInput {
        operator: OperatorId,
        input_index: u16,
        value: ValueId,
        required_error_check: bool,
    },
    ShiftInRange {
        operator: OperatorId,
        value: ValueId,
        maximum: u8,
    },
    NonZero {
        operator: OperatorId,
        value: ValueId,
    },
    Int32MultiplyInRange {
        operator: OperatorId,
        left: ValueId,
        right: ValueId,
        shift: ValueId,
    },
    PowDomain {
        operator: OperatorId,
        base: ValueId,
        exponent: ValueId,
    },
    IndicesInRange {
        operator: OperatorId,
        indices: ValueId,
        upper_bound: u64,
    },
    ScatterIndicesUnique {
        operator: OperatorId,
        indices: ValueId,
    },
    VariableState {
        operator: OperatorId,
        value: ValueId,
    },
    Custom {
        operator: OperatorId,
        domain: &'a str,
        name: &'a str,
    },
}

impl RuntimeCondition<'_> {
    /// Whether a failing condition must be surfaced as a TOSA error for predictable inputs.
    pub const fn error_detection_required(self) -> bool {
        matches!(
            self,
            Self::DynamicCompileTimeInput {
                required_error_check: true,
                ..
            }
        )
    }
}

/// Numerically safe and provider-conditional lowering opportunities.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(transparent)]
pub struct OptimizationHints(u16);

impl OptimizationHints {
    pub const NONE: Self = Self(0);
    /// The operator and its pure dependency chain do not contribute to block outputs.
    pub const DEAD: Self = Self(1 << 0);
    /// Output storage may alias input zero when the provider supports internal views.
    pub const ALIAS_INPUT_ZERO: Self = Self(1 << 1);
    /// All inputs are constant and bounded by [`AnalysisOptions::max_folded_constant_bytes`].
    pub const FOLD_CONSTANT: Self = Self(1 << 2);
    /// A preceding single-use reshape can be composed without changing tensor values.
    pub const COMPOSE_RESHAPE: Self = Self(1 << 3);
    /// A preceding single-use transpose can be composed by combining permutations.
    pub const COMPOSE_TRANSPOSE: Self = Self(1 << 4);
    /// Provider-specific policy and lowering are required.
    pub const CUSTOM: Self = Self(1 << 5);
    /// Provider control-flow lowering is required.
    pub const CONTROL_FLOW: Self = Self(1 << 6);

    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    const fn with(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
}

/// One operator indexed independently of serialized names.
#[derive(Clone, Copy, Debug)]
pub struct AnalyzedOperator<'a> {
    id: OperatorId,
    block: BlockId,
    source_index: u32,
    execution_index: u32,
    op: Op,
    source: Operator<'a>,
    inputs: AnalysisSpan,
    outputs: AnalysisSpan,
    conditions: AnalysisSpan,
    hints: OptimizationHints,
}

impl<'a> AnalyzedOperator<'a> {
    pub const fn id(&self) -> OperatorId {
        self.id
    }

    pub const fn block(&self) -> BlockId {
        self.block
    }

    pub const fn source_index(&self) -> u32 {
        self.source_index
    }

    pub const fn execution_index(&self) -> u32 {
        self.execution_index
    }

    pub const fn op(&self) -> Op {
        self.op
    }

    /// Original verified operator view for zero-copy attribute and location access.
    pub const fn source(&self) -> Operator<'a> {
        self.source
    }

    pub const fn hints(&self) -> OptimizationHints {
        self.hints
    }
}

/// One basic block and its precomputed execution order.
#[derive(Clone, Copy, Debug)]
pub struct AnalyzedBlock<'a> {
    id: BlockId,
    region: RegionId,
    name: &'a str,
    values: AnalysisSpan,
    operators: AnalysisSpan,
    inputs: AnalysisSpan,
    outputs: AnalysisSpan,
    execution_order: AnalysisSpan,
}

impl<'a> AnalyzedBlock<'a> {
    pub const fn id(&self) -> BlockId {
        self.id
    }

    pub const fn region(&self) -> RegionId {
        self.region
    }

    pub const fn name(&self) -> &'a str {
        self.name
    }
}

/// One region and its contiguous block span.
#[derive(Clone, Copy, Debug)]
pub struct AnalyzedRegion<'a> {
    id: RegionId,
    name: &'a str,
    blocks: AnalysisSpan,
}

impl<'a> AnalyzedRegion<'a> {
    pub const fn id(&self) -> RegionId {
        self.id
    }

    pub const fn name(&self) -> &'a str {
        self.name
    }
}

/// Compact provider-neutral overlay used by lowering and specialization code.
///
/// The plan owns only bounded indexes and bookkeeping. Names, tensor metadata, shape metadata, and
/// constant payloads remain borrowed from the verified model.
#[derive(Debug)]
pub struct TosaAnalysis<'a> {
    target: Target,
    model_bytes: &'a [u8],
    regions: Vec<AnalyzedRegion<'a>>,
    blocks: Vec<AnalyzedBlock<'a>>,
    values: Vec<AnalyzedValue<'a>>,
    operators: Vec<AnalyzedOperator<'a>>,
    operands: Vec<ValueId>,
    block_io: Vec<ValueId>,
    execution_order: Vec<OperatorId>,
    conditions: Vec<RuntimeCondition<'a>>,
}

impl<'a> TosaAnalysis<'a> {
    pub fn build(model: &Model<'a>, target: Target) -> Result<Self, AnalysisError> {
        Self::build_with_options(model, target, AnalysisOptions::default())
    }

    pub fn build_with_options(
        model: &Model<'a>,
        target: Target,
        options: AnalysisOptions,
    ) -> Result<Self, AnalysisError> {
        validate_semantics(model, target)?;
        Builder::new(model, target, options).build()
    }

    pub const fn target(&self) -> Target {
        self.target
    }

    /// Serialized bytes for a `CONST`/`CONST_SHAPE` value, including external constant storage.
    pub fn serialized_constant(&self, id: ValueId) -> Option<&'a [u8]> {
        let value = self.value(id);
        if value.constant != ConstantState::Serialized {
            return None;
        }
        match value.kind {
            AnalyzedValueKind::Tensor(tensor) => {
                if !tensor.data().is_empty() {
                    return Some(tensor.data());
                }
                let (offset, size) = tensor.external_data_range()?;
                let start = usize::try_from(offset).ok()?;
                let size = usize::try_from(size).ok()?;
                self.model_bytes.get(start..start.checked_add(size)?)
            }
            AnalyzedValueKind::Shape(shape) => Some(shape.data()),
        }
    }

    pub fn regions(&self) -> &[AnalyzedRegion<'a>] {
        &self.regions
    }

    pub fn blocks(&self) -> &[AnalyzedBlock<'a>] {
        &self.blocks
    }

    pub fn values(&self) -> &[AnalyzedValue<'a>] {
        &self.values
    }

    pub fn operators(&self) -> &[AnalyzedOperator<'a>] {
        &self.operators
    }

    pub fn conditions(&self) -> &[RuntimeCondition<'a>] {
        &self.conditions
    }

    pub fn value(&self, id: ValueId) -> &AnalyzedValue<'a> {
        &self.values[id.index()]
    }

    pub fn operator(&self, id: OperatorId) -> &AnalyzedOperator<'a> {
        &self.operators[id.index()]
    }

    pub fn operator_inputs(&self, operator: OperatorId) -> &[ValueId] {
        &self.operands[self.operator(operator).inputs.range()]
    }

    pub fn operator_outputs(&self, operator: OperatorId) -> &[ValueId] {
        &self.operands[self.operator(operator).outputs.range()]
    }

    pub fn operator_conditions(&self, operator: OperatorId) -> &[RuntimeCondition<'a>] {
        &self.conditions[self.operator(operator).conditions.range()]
    }

    pub fn block_values(&self, block: BlockId) -> &[AnalyzedValue<'a>] {
        &self.values[self.blocks[block.index()].values.range()]
    }

    pub fn block_operators(&self, block: BlockId) -> &[AnalyzedOperator<'a>] {
        &self.operators[self.blocks[block.index()].operators.range()]
    }

    pub fn block_inputs(&self, block: BlockId) -> &[ValueId] {
        &self.block_io[self.blocks[block.index()].inputs.range()]
    }

    pub fn block_outputs(&self, block: BlockId) -> &[ValueId] {
        &self.block_io[self.blocks[block.index()].outputs.range()]
    }

    pub fn execution_order(&self, block: BlockId) -> &[OperatorId] {
        &self.execution_order[self.blocks[block.index()].execution_order.range()]
    }

    pub fn region_blocks(&self, region: RegionId) -> &[AnalyzedBlock<'a>] {
        &self.blocks[self.regions[region.index()].blocks.range()]
    }
}

struct Builder<'model, 'a> {
    model: &'model Model<'a>,
    target: Target,
    options: AnalysisOptions,
    analysis: TosaAnalysis<'a>,
}

impl<'model, 'a> Builder<'model, 'a> {
    fn new(model: &'model Model<'a>, target: Target, options: AnalysisOptions) -> Self {
        Self {
            model,
            target,
            options,
            analysis: TosaAnalysis {
                target,
                model_bytes: model.as_bytes(),
                regions: Vec::new(),
                blocks: Vec::new(),
                values: Vec::new(),
                operators: Vec::new(),
                operands: Vec::new(),
                block_io: Vec::new(),
                execution_order: Vec::new(),
                conditions: Vec::new(),
            },
        }
    }

    fn build(mut self) -> Result<TosaAnalysis<'a>, AnalysisError> {
        reserve(&mut self.analysis.regions, self.model.regions().len())?;
        reserve(&mut self.analysis.blocks, self.model.stats().blocks)?;
        reserve(
            &mut self.analysis.values,
            self.model.stats().tensors + self.model.stats().shapes,
        )?;
        reserve(&mut self.analysis.operators, self.model.stats().operators)?;
        reserve(&mut self.analysis.operands, self.model.stats().edges)?;
        reserve(&mut self.analysis.block_io, self.model.stats().edges)?;
        reserve(
            &mut self.analysis.execution_order,
            self.model.stats().operators,
        )?;

        for region in self.model.regions() {
            let region_id = RegionId::from_usize(self.analysis.regions.len())?;
            let block_start = self.analysis.blocks.len();
            for block in region.blocks() {
                self.build_block(region_id, block)?;
            }
            self.analysis.regions.push(AnalyzedRegion {
                id: region_id,
                name: region.name(),
                blocks: AnalysisSpan::new(block_start, self.analysis.blocks.len() - block_start)?,
            });
        }
        Ok(self.analysis)
    }

    fn build_block(
        &mut self,
        region_id: RegionId,
        block: BasicBlock<'a>,
    ) -> Result<(), AnalysisError> {
        let block_id = BlockId::from_usize(self.analysis.blocks.len())?;
        let value_start = self.analysis.values.len();
        let operator_start = self.analysis.operators.len();
        let mut symbols = Vec::new();
        reserve(&mut symbols, block.tensors().len() + block.shapes().len())?;

        for tensor in block.tensors() {
            let id = ValueId::from_usize(self.analysis.values.len())?;
            let byte_size = tensor_byte_size(tensor);
            self.analysis.values.push(AnalyzedValue {
                id,
                block: block_id,
                name: tensor.name(),
                kind: AnalyzedValueKind::Tensor(tensor),
                producer: None,
                constant: ConstantState::NonConstant,
                byte_size,
                consumers: 0,
                first_use: None,
                last_use: None,
            });
            symbols.push((tensor.name(), id));
        }
        for shape in block.shapes() {
            let id = ValueId::from_usize(self.analysis.values.len())?;
            self.analysis.values.push(AnalyzedValue {
                id,
                block: block_id,
                name: shape.name(),
                kind: AnalyzedValueKind::Shape(shape),
                producer: None,
                constant: ConstantState::NonConstant,
                byte_size: Some(u64::from(shape.rank()) * 8),
                consumers: 0,
                first_use: None,
                last_use: None,
            });
            symbols.push((shape.name(), id));
        }
        symbols.sort_unstable_by_key(|(name, _)| *name);

        let input_start = self.analysis.block_io.len();
        for name in block.inputs() {
            self.analysis.block_io.push(resolve(&symbols, name));
        }
        let input_span =
            AnalysisSpan::new(input_start, self.analysis.block_io.len() - input_start)?;
        let output_start = self.analysis.block_io.len();
        for name in block.outputs() {
            self.analysis.block_io.push(resolve(&symbols, name));
        }
        let output_span =
            AnalysisSpan::new(output_start, self.analysis.block_io.len() - output_start)?;

        let local_operator_count = block.operators().len();
        let mut local_inputs = Vec::new();
        let mut local_outputs = Vec::new();
        reserve(&mut local_inputs, local_operator_count)?;
        reserve(&mut local_outputs, local_operator_count)?;
        for (source_index, operator) in block.operators().enumerate() {
            let id = OperatorId::from_usize(self.analysis.operators.len())?;
            let inputs_start = self.analysis.operands.len();
            for name in operator.inputs() {
                self.analysis.operands.push(resolve(&symbols, name));
            }
            let inputs =
                AnalysisSpan::new(inputs_start, self.analysis.operands.len() - inputs_start)?;
            let outputs_start = self.analysis.operands.len();
            for name in operator.outputs() {
                let value = resolve(&symbols, name);
                self.analysis.operands.push(value);
                if !matches!(self.analysis.values[value.index()].kind, AnalyzedValueKind::Tensor(tensor) if tensor.is_variable())
                {
                    self.analysis.values[value.index()].producer = Some(id);
                }
            }
            let outputs =
                AnalysisSpan::new(outputs_start, self.analysis.operands.len() - outputs_start)?;
            if matches!(operator.op(), Op::CONST | Op::CONST_SHAPE) {
                for value in &self.analysis.operands[outputs.range()] {
                    self.analysis.values[value.index()].constant = ConstantState::Serialized;
                }
            }
            self.analysis.operators.push(AnalyzedOperator {
                id,
                block: block_id,
                source_index: u32::try_from(source_index)
                    .map_err(|_| AnalysisError::TooManyObjects)?,
                execution_index: 0,
                op: operator.op(),
                source: operator,
                inputs,
                outputs,
                conditions: AnalysisSpan::default(),
                hints: OptimizationHints::NONE,
            });
            local_inputs.push(inputs);
            local_outputs.push(outputs);
        }

        for (local_index, operator) in block.operators().enumerate() {
            let operator_index = operator_start + local_index;
            let analyzed = self.analysis.operators[operator_index];
            let condition_start = self.analysis.conditions.len();
            add_conditions(
                ConditionContext {
                    target: self.target,
                    operator: analyzed.id,
                    op: analyzed.op,
                    attributes: operator.attributes(),
                    inputs: &self.analysis.operands[analyzed.inputs.range()],
                    outputs: &self.analysis.operands[analyzed.outputs.range()],
                    values: &self.analysis.values,
                },
                &mut self.analysis.conditions,
            )?;
            self.analysis.operators[operator_index].conditions = AnalysisSpan::new(
                condition_start,
                self.analysis.conditions.len() - condition_start,
            )?;
        }

        let order = topological_order(
            operator_start,
            &local_inputs,
            &self.analysis.values,
            &self.analysis.operands,
            local_operator_count,
        )?;
        let execution_start = self.analysis.execution_order.len();
        for (position, local_index) in order.iter().copied().enumerate() {
            let operator_index = operator_start + local_index;
            let operator_id = self.analysis.operators[operator_index].id;
            self.analysis.operators[operator_index].execution_index =
                u32::try_from(position).map_err(|_| AnalysisError::TooManyObjects)?;
            self.analysis.execution_order.push(operator_id);
            let input_span = self.analysis.operators[operator_index].inputs;
            for value in &self.analysis.operands[input_span.range()] {
                let value = &mut self.analysis.values[value.index()];
                value.consumers = value
                    .consumers
                    .checked_add(1)
                    .ok_or(AnalysisError::TooManyObjects)?;
                let position =
                    u32::try_from(position).map_err(|_| AnalysisError::TooManyObjects)?;
                value.first_use.get_or_insert(position);
                value.last_use = Some(position);
            }
            self.propagate_constants(operator_index)?;
        }
        let block_end =
            u32::try_from(local_operator_count).map_err(|_| AnalysisError::TooManyObjects)?;
        for value in &self.analysis.block_io[output_span.range()] {
            self.analysis.values[value.index()].last_use = Some(block_end);
        }
        let mut output_values = Vec::new();
        reserve(&mut output_values, output_span.len())?;
        output_values.extend_from_slice(&self.analysis.block_io[output_span.range()]);
        self.mark_live_and_hints(operator_start, &order, &output_values)?;

        self.analysis.blocks.push(AnalyzedBlock {
            id: block_id,
            region: region_id,
            name: block.name(),
            values: AnalysisSpan::new(value_start, self.analysis.values.len() - value_start)?,
            operators: AnalysisSpan::new(
                operator_start,
                self.analysis.operators.len() - operator_start,
            )?,
            inputs: input_span,
            outputs: output_span,
            execution_order: AnalysisSpan::new(
                execution_start,
                self.analysis.execution_order.len() - execution_start,
            )?,
        });
        Ok(())
    }

    fn propagate_constants(&mut self, operator_index: usize) -> Result<(), AnalysisError> {
        let operator = self.analysis.operators[operator_index];
        if !foldable_operator(operator.op)
            || !self.analysis.operands[operator.inputs.range()]
                .iter()
                .all(|value| {
                    self.analysis.values[value.index()].constant != ConstantState::NonConstant
                })
        {
            return Ok(());
        }
        let total_bytes = self.analysis.operands[operator.outputs.range()]
            .iter()
            .try_fold(0_u64, |total, value| {
                total.checked_add(self.analysis.values[value.index()].byte_size?)
            });
        if total_bytes.is_some_and(|bytes| bytes <= self.options.max_folded_constant_bytes) {
            self.analysis.operators[operator_index].hints = self.analysis.operators[operator_index]
                .hints
                .with(OptimizationHints::FOLD_CONSTANT);
            for value in &self.analysis.operands[operator.outputs.range()] {
                self.analysis.values[value.index()].constant = ConstantState::Foldable;
            }
        }
        Ok(())
    }

    fn mark_live_and_hints(
        &mut self,
        operator_start: usize,
        order: &[usize],
        block_outputs: &[ValueId],
    ) -> Result<(), AnalysisError> {
        let mut live = Vec::new();
        reserve(&mut live, order.len())?;
        live.resize(order.len(), false);
        for value in block_outputs {
            if let Some(producer) = self.analysis.values[value.index()].producer {
                live[producer.index() - operator_start] = true;
            }
        }
        for &local_index in order {
            let op = self.analysis.operators[operator_start + local_index].op;
            if has_side_effects(op) {
                live[local_index] = true;
            }
        }
        for &local_index in order.iter().rev() {
            let operator_index = operator_start + local_index;
            if live[local_index] {
                let inputs = self.analysis.operators[operator_index].inputs;
                for value in &self.analysis.operands[inputs.range()] {
                    if let Some(producer) = self.analysis.values[value.index()].producer {
                        live[producer.index() - operator_start] = true;
                    }
                }
            } else {
                self.analysis.operators[operator_index].hints = self.analysis.operators
                    [operator_index]
                    .hints
                    .with(OptimizationHints::DEAD);
            }
        }

        for &local_index in order {
            let operator_index = operator_start + local_index;
            let operator = self.analysis.operators[operator_index];
            let mut hints = operator.hints;
            if matches!(operator.op, Op::IDENTITY | Op::RESHAPE) {
                hints = hints.with(OptimizationHints::ALIAS_INPUT_ZERO);
            }
            if operator.op == Op::CUSTOM {
                hints = hints.with(OptimizationHints::CUSTOM);
            }
            if matches!(operator.op, Op::COND_IF | Op::WHILE_LOOP) {
                hints = hints.with(OptimizationHints::CONTROL_FLOW);
            }
            let inputs = &self.analysis.operands[operator.inputs.range()];
            if let Some(input) = inputs.first() {
                if self.analysis.values[input.index()].consumers == 1 {
                    if let Some(producer) = self.analysis.values[input.index()].producer {
                        let producer_op = self.analysis.operators[producer.index()].op;
                        if operator.op == Op::RESHAPE && producer_op == Op::RESHAPE {
                            hints = hints.with(OptimizationHints::COMPOSE_RESHAPE);
                        }
                        if operator.op == Op::TRANSPOSE && producer_op == Op::TRANSPOSE {
                            hints = hints.with(OptimizationHints::COMPOSE_TRANSPOSE);
                        }
                    }
                }
            }
            self.analysis.operators[operator_index].hints = hints;
        }
        Ok(())
    }
}

fn reserve<T>(values: &mut Vec<T>, additional: usize) -> Result<(), AnalysisError> {
    values
        .try_reserve_exact(additional)
        .map_err(|_| AnalysisError::AllocationFailed)
}

fn resolve(symbols: &[(&str, ValueId)], name: &str) -> ValueId {
    symbols[symbols
        .binary_search_by_key(&name, |(candidate, _)| *candidate)
        .expect("semantically validated symbol")]
    .1
}

fn topological_order(
    operator_start: usize,
    inputs: &[AnalysisSpan],
    values: &[AnalyzedValue<'_>],
    operands: &[ValueId],
    operator_count: usize,
) -> Result<Vec<usize>, AnalysisError> {
    let mut indegrees = Vec::new();
    reserve(&mut indegrees, operator_count)?;
    indegrees.resize(operator_count, 0_usize);
    let mut edges = Vec::new();
    let edge_count = inputs
        .iter()
        .try_fold(0_usize, |count, span| count.checked_add(span.len()))
        .ok_or(AnalysisError::TooManyObjects)?;
    reserve(&mut edges, edge_count)?;
    for (consumer, span) in inputs.iter().copied().enumerate() {
        for value in &operands[span.range()] {
            if let Some(producer) = values[value.index()].producer {
                let producer = producer
                    .index()
                    .checked_sub(operator_start)
                    .ok_or(AnalysisError::TooManyObjects)?;
                edges.push((producer, consumer));
                indegrees[consumer] = indegrees[consumer]
                    .checked_add(1)
                    .ok_or(AnalysisError::TooManyObjects)?;
            }
        }
    }
    edges.sort_unstable();
    let mut order = Vec::new();
    reserve(&mut order, operator_count)?;
    order.extend(
        indegrees
            .iter()
            .enumerate()
            .filter_map(|(index, indegree)| (*indegree == 0).then_some(index)),
    );
    let mut cursor = 0;
    while cursor < order.len() {
        let producer = order[cursor];
        cursor += 1;
        let start = edges.partition_point(|(candidate, _)| *candidate < producer);
        let end = edges.partition_point(|(candidate, _)| *candidate <= producer);
        for &(_, consumer) in &edges[start..end] {
            indegrees[consumer] -= 1;
            if indegrees[consumer] == 0 {
                order.push(consumer);
            }
        }
    }
    if order.len() == operator_count {
        Ok(order)
    } else {
        // The semantic pass has already rejected cycles.
        Err(AnalysisError::TooManyObjects)
    }
}

fn tensor_byte_size(tensor: Tensor<'_>) -> Option<u64> {
    let _ = tensor.rank()?;
    let elements = tensor.dimensions().try_fold(1_u64, |count, dimension| {
        count.checked_mul(u64::try_from(dimension).ok()?)
    })?;
    let bytes = match tensor.dtype() {
        DType::INT4 => return elements.checked_add(1).map(|value| value / 2),
        DType::BOOL | DType::INT8 | DType::FP8E4M3 | DType::FP8E5M2 => 1,
        DType::INT16 | DType::FP16 | DType::BF16 => 2,
        DType::INT32 | DType::FP32 => 4,
        DType::INT48 => 6,
        _ => return None,
    };
    elements.checked_mul(bytes)
}

fn foldable_operator(op: Op) -> bool {
    matches!(
        op,
        Op::IDENTITY
            | Op::RESHAPE
            | Op::TRANSPOSE
            | Op::REVERSE
            | Op::SLICE
            | Op::TILE
            | Op::CONCAT
    )
}

fn has_side_effects(op: Op) -> bool {
    matches!(
        op,
        Op::CUSTOM
            | Op::COND_IF
            | Op::WHILE_LOOP
            | Op::VARIABLE
            | Op::VARIABLE_READ
            | Op::VARIABLE_WRITE
    )
}

struct ConditionContext<'a, 'plan> {
    target: Target,
    operator: OperatorId,
    op: Op,
    attributes: crate::OpAttributes<'a>,
    inputs: &'plan [ValueId],
    outputs: &'plan [ValueId],
    values: &'plan [AnalyzedValue<'a>],
}

fn add_conditions<'a>(
    context: ConditionContext<'a, '_>,
    conditions: &mut Vec<RuntimeCondition<'a>>,
) -> Result<(), AnalysisError> {
    let ConditionContext {
        target,
        operator,
        op,
        attributes,
        inputs,
        outputs,
        values,
    } = context;
    let dynamic = target.extensions.contains(ExtensionSet::DYNAMIC);
    if dynamic {
        for &input_index in ctc_inputs(op) {
            let value = inputs[input_index];
            if values[value.index()].constant != ConstantState::Serialized {
                conditions
                    .try_reserve(1)
                    .map_err(|_| AnalysisError::AllocationFailed)?;
                conditions.push(RuntimeCondition::DynamicCompileTimeInput {
                    operator,
                    input_index: u16::try_from(input_index)
                        .map_err(|_| AnalysisError::TooManyObjects)?,
                    value,
                    required_error_check: dynamic_ctc_requires_error(op, input_index),
                });
            }
        }
    }
    let mut push = |condition| -> Result<(), AnalysisError> {
        conditions
            .try_reserve(1)
            .map_err(|_| AnalysisError::AllocationFailed)?;
        conditions.push(condition);
        Ok(())
    };
    match op {
        Op::ARITHMETIC_RIGHT_SHIFT | Op::LOGICAL_LEFT_SHIFT | Op::LOGICAL_RIGHT_SHIFT => {
            let maximum = match values[inputs[0].index()].kind {
                AnalyzedValueKind::Tensor(tensor) if tensor.dtype() == DType::INT32 => 31,
                AnalyzedValueKind::Tensor(tensor) if tensor.dtype() == DType::INT16 => 15,
                _ => 7,
            };
            push(RuntimeCondition::ShiftInRange {
                operator,
                value: inputs[1],
                maximum,
            })?;
        }
        Op::INTDIV => push(RuntimeCondition::NonZero {
            operator,
            value: inputs[1],
        })?,
        Op::MUL
            if matches!(
                values[inputs[0].index()].kind,
                AnalyzedValueKind::Tensor(tensor) if tensor.dtype() == DType::INT32
            ) =>
        {
            push(RuntimeCondition::Int32MultiplyInRange {
                operator,
                left: inputs[0],
                right: inputs[1],
                shift: inputs[2],
            })?;
        }
        Op::POW => push(RuntimeCondition::PowDomain {
            operator,
            base: inputs[0],
            exponent: inputs[1],
        })?,
        Op::GATHER | Op::SCATTER => {
            let data = match values[inputs[0].index()].kind {
                AnalyzedValueKind::Tensor(tensor) => tensor,
                AnalyzedValueKind::Shape(_) => unreachable!(),
            };
            let upper_bound = data.dimensions().nth(1).unwrap_or(0) as u64;
            push(RuntimeCondition::IndicesInRange {
                operator,
                indices: inputs[1],
                upper_bound,
            })?;
            if op == Op::SCATTER {
                push(RuntimeCondition::ScatterIndicesUnique {
                    operator,
                    indices: inputs[1],
                })?;
            }
        }
        Op::VARIABLE_WRITE => push(RuntimeCondition::VariableState {
            operator,
            value: inputs[0],
        })?,
        Op::VARIABLE_READ => {
            if let Some(value) = outputs.first() {
                push(RuntimeCondition::VariableState {
                    operator,
                    value: *value,
                })?;
            }
        }
        Op::CUSTOM => {
            let crate::OpAttributes::Custom {
                operator_name,
                domain_name,
                ..
            } = attributes
            else {
                unreachable!()
            };
            push(RuntimeCondition::Custom {
                operator,
                domain: domain_name.unwrap_or(""),
                name: operator_name.unwrap_or(""),
            })?;
        }
        _ => {}
    }
    Ok(())
}

fn ctc_inputs(op: Op) -> &'static [usize] {
    match op.get() {
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
    }
}

const fn dynamic_ctc_requires_error(op: Op, input_index: usize) -> bool {
    match op.get() {
        2..=5 | 7 | 10 | 41 | 56..=57 | 59..=60 | 64 => true,
        66 => input_index >= 3,
        28 | 31 => false,
        _ => false,
    }
}
