//! The crate-authored SPIR-V compute kernels (ADR 0003, ADR 0007).
//!
//! Every module the backend hands to a driver is assembled here from fixed templates and
//! specialized at `load_program` through specialization constants alone. Guest bytes never reach
//! the driver's shader compiler: a TOSA artifact selects a [`KernelKey`] and supplies validated
//! shape parameters as a specialization payload, nothing else (`docs/threat-model.md`,
//! transient-compile budget). The only template parameters that change a module's instructions
//! are device properties fixed when the backend opens a device — the workgroup size, the MATMUL
//! tile, the length of the storage-buffer descriptor array — plus the operator selection itself.
//!
//! ## Operand addressing
//!
//! Each kernel reads and writes tensors through one descriptor: set 0, binding 0, an array of
//! `{ uint words[]; }` storage-buffer blocks. An [`Operand`] names an array element and a base
//! offset in 32-bit words, both specialization constants, so one module per kernel serves every
//! binding layout, the program-owned arena, and `CONCAT` with any input count. Word-storage
//! tensors (`FP32`, `INT32`) are addressed one element per word; byte-storage tensors (`BOOL`)
//! are read with word loads and written with `OpAtomicAnd`/`OpAtomicOr`, so a kernel never
//! modifies bytes outside the elements it owns even at a tensor's unaligned tail.
//!
//! ## Numerics policy
//!
//! Every floating-point arithmetic result carries `NoContraction`, so no driver may fuse a
//! multiply and an add: the same TOSA graph yields the same bits on every conformant device.
//! `SIN`, `COS`, `TANH`, and `ERF` are evaluated by crate-authored range reductions and
//! polynomials rather than the driver's built-ins, whose precision Vulkan specifies loosely
//! (`sin`/`cos`: absolute error 2⁻¹¹) or not at all (`tanh`). `EXP`, `LOG`, `POW`, and
//! `RSQRT` use the `GLSL.std.450` built-ins, whose relative-error bounds Vulkan does specify.
//! NaN-mode attributes (`PROPAGATE`/`IGNORE`) follow the TOSA 1.0 pseudocode literally, with
//! explicit `OpIsNan` selects instead of the driver's undefined NaN handling for `FMax`/`FMin`.

use std::collections::HashMap;

/// SPIR-V 1.3: the version every Vulkan 1.1+ implementation must consume, and the first with the
/// `StorageBuffer` storage class in core.
pub const SPIRV_VERSION_1_3: u32 = 0x0001_0300;
/// The SPIR-V magic number.
pub const SPIRV_MAGIC: u32 = 0x0723_0203;

/// The leading fraction bits of 2/π, most significant first, preceded by one zero word.
///
/// The zero word biases the Payne–Hanek bit window so its offset is never negative for any
/// argument the reduction handles; eleven words of 2/π (352 bits) cover every binary32 exponent
/// with the 128-bit window the reduction extracts (`docs/adr/0007-fp32-operator-tier.md`).
const TWO_OVER_PI_BITS: &[u32] = &[
    0x0000_0000,
    0xa2f9_836e,
    0x4e44_1529,
    0xfc27_57d1,
    0xf534_ddc0,
    0xdb62_9599,
    0x3c43_9041,
    0xfe51_63ab,
    0xdebb_c561,
    0xb724_6e3a,
    0x424d_d2e0,
    0x0649_2eea,
];

/// Below this magnitude the three-part π/4 subtraction is exact enough; at or above it the
/// Payne–Hanek reduction takes over.
const SINCOS_FAST_RANGE: f32 = 8192.0;

/// TOSA level 8K rank bound the strided kernels are sized for.
pub const MAX_RANK: usize = 6;
/// Elementwise kernels take at most three tensor inputs (`SELECT`).
pub const MAX_ELEMENTWISE_INPUTS: usize = 3;

// SPIR-V opcodes (Unified specification, section 3.52).
const OP_EXT_INST_IMPORT: u16 = 11;
const OP_EXT_INST: u16 = 12;
const OP_MEMORY_MODEL: u16 = 14;
const OP_ENTRY_POINT: u16 = 15;
const OP_EXECUTION_MODE: u16 = 16;
const OP_CAPABILITY: u16 = 17;
const OP_TYPE_VOID: u16 = 19;
const OP_TYPE_BOOL: u16 = 20;
const OP_TYPE_INT: u16 = 21;
const OP_TYPE_FLOAT: u16 = 22;
const OP_TYPE_VECTOR: u16 = 23;
const OP_TYPE_ARRAY: u16 = 28;
const OP_TYPE_RUNTIME_ARRAY: u16 = 29;
const OP_TYPE_STRUCT: u16 = 30;
const OP_TYPE_POINTER: u16 = 32;
const OP_TYPE_FUNCTION: u16 = 33;
const OP_CONSTANT_FALSE: u16 = 42;
const OP_CONSTANT: u16 = 43;
const OP_CONSTANT_COMPOSITE: u16 = 44;
const OP_SPEC_CONSTANT: u16 = 50;
const OP_FUNCTION: u16 = 54;
const OP_FUNCTION_END: u16 = 56;
const OP_VARIABLE: u16 = 59;
const OP_LOAD: u16 = 61;
const OP_STORE: u16 = 62;
const OP_ACCESS_CHAIN: u16 = 65;
const OP_DECORATE: u16 = 71;
const OP_MEMBER_DECORATE: u16 = 72;
const OP_CONVERT_F_TO_U: u16 = 109;
const OP_CONVERT_U_TO_F: u16 = 112;
const OP_BITCAST: u16 = 124;
const OP_F_NEGATE: u16 = 127;
const OP_I_ADD: u16 = 128;
const OP_F_ADD: u16 = 129;
const OP_I_SUB: u16 = 130;
const OP_F_SUB: u16 = 131;
const OP_I_MUL: u16 = 132;
const OP_F_MUL: u16 = 133;
const OP_U_DIV: u16 = 134;
const OP_F_DIV: u16 = 136;
const OP_U_MOD: u16 = 137;
const OP_IS_NAN: u16 = 156;
const OP_LOGICAL_NOT_EQUAL: u16 = 165;
const OP_LOGICAL_OR: u16 = 166;
const OP_LOGICAL_AND: u16 = 167;
const OP_LOGICAL_NOT: u16 = 168;
const OP_SELECT: u16 = 169;
const OP_I_EQUAL: u16 = 170;
const OP_I_NOT_EQUAL: u16 = 171;
const OP_U_GREATER_THAN_EQUAL: u16 = 174;
const OP_U_LESS_THAN: u16 = 176;
const OP_F_ORD_EQUAL: u16 = 180;
const OP_F_ORD_LESS_THAN: u16 = 184;
const OP_F_ORD_GREATER_THAN: u16 = 186;
const OP_F_ORD_GREATER_THAN_EQUAL: u16 = 190;
const OP_SHIFT_RIGHT_LOGICAL: u16 = 194;
const OP_SHIFT_LEFT_LOGICAL: u16 = 196;
const OP_BITWISE_OR: u16 = 197;
const OP_BITWISE_AND: u16 = 199;
const OP_NOT: u16 = 200;
const OP_CONTROL_BARRIER: u16 = 224;
const OP_ATOMIC_AND: u16 = 240;
const OP_ATOMIC_OR: u16 = 241;
const OP_LOOP_MERGE: u16 = 246;
const OP_SELECTION_MERGE: u16 = 247;
const OP_LABEL: u16 = 248;
const OP_BRANCH: u16 = 249;
const OP_BRANCH_CONDITIONAL: u16 = 250;
const OP_RETURN: u16 = 253;

// Enumerants (section 3).
const CAPABILITY_SHADER: u32 = 1;
const ADDRESSING_MODEL_LOGICAL: u32 = 0;
const MEMORY_MODEL_GLSL450: u32 = 1;
const EXECUTION_MODEL_GL_COMPUTE: u32 = 5;
const EXECUTION_MODE_LOCAL_SIZE: u32 = 17;
const STORAGE_CLASS_INPUT: u32 = 1;
const STORAGE_CLASS_WORKGROUP: u32 = 4;
const STORAGE_CLASS_PRIVATE: u32 = 6;
const STORAGE_CLASS_FUNCTION: u32 = 7;
const STORAGE_CLASS_STORAGE_BUFFER: u32 = 12;
const DECORATION_SPEC_ID: u32 = 1;
const DECORATION_BLOCK: u32 = 2;
const DECORATION_ARRAY_STRIDE: u32 = 6;
const DECORATION_BUILT_IN: u32 = 11;
const DECORATION_BINDING: u32 = 33;
const DECORATION_DESCRIPTOR_SET: u32 = 34;
const DECORATION_OFFSET: u32 = 35;
const DECORATION_NO_CONTRACTION: u32 = 42;
const BUILT_IN_NUM_WORKGROUPS: u32 = 24;
const BUILT_IN_WORKGROUP_ID: u32 = 26;
const BUILT_IN_LOCAL_INVOCATION_ID: u32 = 27;
const BUILT_IN_GLOBAL_INVOCATION_ID: u32 = 28;
const FUNCTION_CONTROL_NONE: u32 = 0;
const SELECTION_CONTROL_NONE: u32 = 0;
const LOOP_CONTROL_NONE: u32 = 0;
const SCOPE_DEVICE: u32 = 1;
const SCOPE_WORKGROUP: u32 = 2;
const MEMORY_SEMANTICS_RELAXED: u32 = 0;
const MEMORY_SEMANTICS_ACQUIRE_RELEASE_WORKGROUP: u32 = 0x8 | 0x100;

// GLSL.std.450 extended instructions.
const GLSL_FABS: u32 = 4;
const GLSL_FLOOR: u32 = 8;
const GLSL_CEIL: u32 = 9;
const GLSL_POW: u32 = 26;
const GLSL_EXP: u32 = 27;
const GLSL_LOG: u32 = 28;
const GLSL_INVERSE_SQRT: u32 = 32;
const GLSL_UMIN: u32 = 38;
const GLSL_FIND_U_MSB: u32 = 75;

/// A SPIR-V result id.
pub type Id = u32;

/// How a tensor's scalars are laid out in the storage words a kernel addresses.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Storage {
    /// One 32-bit word per element (`FP32`, `INT32`).
    Word,
    /// One byte per element, four to a word (`BOOL`).
    Byte,
}

/// TOSA NaN-propagation attribute value a kernel is specialized for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NanMode {
    /// A NaN operand yields NaN (`apply_max_s`/`apply_min_s` propagate).
    Propagate,
    /// A NaN operand is ignored in favor of the other operand.
    Ignore,
}

/// Elementwise operator lanes: the scalar function one invocation applies per output element.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ElementwiseOp {
    Abs,
    Ceil,
    Cos,
    Erf,
    Exp,
    Floor,
    Log,
    Negate,
    Reciprocal,
    Rsqrt,
    Sin,
    Sigmoid,
    Tanh,
    /// `apply_min(apply_max(x, lo), hi)`; the bounds arrive as two trailing specialization
    /// constants holding FP32 bit patterns.
    Clamp(NanMode),
    Add,
    Sub,
    Mul,
    Pow,
    Maximum(NanMode),
    Minimum(NanMode),
    Equal,
    Greater,
    GreaterEqual,
    LogicalAnd,
    LogicalOr,
    LogicalXor,
    LogicalNot,
    Select,
    /// Byte-exact copy of a `BOOL` tensor (`IDENTITY`, `RESHAPE` on byte storage).
    CopyBytes,
}

impl ElementwiseOp {
    /// Storage of each tensor input, in operand order.
    pub const fn inputs(self) -> &'static [Storage] {
        match self {
            Self::Abs
            | Self::Ceil
            | Self::Cos
            | Self::Erf
            | Self::Exp
            | Self::Floor
            | Self::Log
            | Self::Negate
            | Self::Reciprocal
            | Self::Rsqrt
            | Self::Sin
            | Self::Sigmoid
            | Self::Tanh
            | Self::Clamp(_) => &[Storage::Word],
            Self::Add
            | Self::Sub
            | Self::Mul
            | Self::Pow
            | Self::Maximum(_)
            | Self::Minimum(_)
            | Self::Equal
            | Self::Greater
            | Self::GreaterEqual => &[Storage::Word, Storage::Word],
            Self::LogicalAnd | Self::LogicalOr | Self::LogicalXor => {
                &[Storage::Byte, Storage::Byte]
            }
            Self::LogicalNot | Self::CopyBytes => &[Storage::Byte],
            Self::Select => &[Storage::Byte, Storage::Word, Storage::Word],
        }
    }

    /// Storage of the output tensor.
    pub const fn output(self) -> Storage {
        match self {
            Self::Equal
            | Self::Greater
            | Self::GreaterEqual
            | Self::LogicalAnd
            | Self::LogicalOr
            | Self::LogicalXor
            | Self::LogicalNot
            | Self::CopyBytes => Storage::Byte,
            _ => Storage::Word,
        }
    }

    /// Trailing operator-specific specialization constants (after the operand and shape block).
    pub const fn extra_spec_constants(self) -> u32 {
        match self {
            Self::Clamp(_) => 2,
            _ => 0,
        }
    }
}

/// Reduction operator lanes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ReduceOp {
    Sum,
    Product,
    Max(NanMode),
    Min(NanMode),
    /// `INT32` index of the maximum along the axis (lowest index on ties).
    ArgMax(NanMode),
}

/// One assembled kernel variant. Everything that changes instructions is in the key; everything
/// that changes only numbers is a specialization constant.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum KernelKey {
    /// Elementwise lanes over `count` output elements; `broadcast` selects the strided
    /// multi-index addressing, otherwise every operand shares the output's linear index.
    Elementwise {
        op: ElementwiseOp,
        broadcast: bool,
        workgroup: u32,
        buffers: u32,
    },
    /// Axis reduction: one invocation per output element, sequential ascending-axis fold.
    Reduce {
        op: ReduceOp,
        workgroup: u32,
        buffers: u32,
    },
    /// FP32 batched matrix multiplication over `tile × tile` workgroup-shared tiles.
    Matmul { tile: u32, buffers: u32 },
    /// NHWC FP32 max pooling with padding excluded from the window.
    MaxPool {
        nan_mode: NanMode,
        workgroup: u32,
        buffers: u32,
    },
    /// Strided copy over a rank-`MAX_RANK` iteration space (`TRANSPOSE`, `REVERSE`, `CONCAT`
    /// segments); `contiguous` collapses to a linear copy.
    Move {
        storage: Storage,
        contiguous: bool,
        workgroup: u32,
        buffers: u32,
    },
}

impl KernelKey {
    /// Assemble the module for this variant.
    pub fn assemble(self) -> Vec<u32> {
        match self {
            Self::Elementwise {
                op,
                broadcast,
                workgroup,
                buffers,
            } => assemble_elementwise(op, broadcast, workgroup, buffers),
            Self::Reduce {
                op,
                workgroup,
                buffers,
            } => assemble_reduce(op, workgroup, buffers),
            Self::Matmul { tile, buffers } => assemble_matmul(tile, buffers),
            Self::MaxPool {
                nan_mode,
                workgroup,
                buffers,
            } => assemble_max_pool(nan_mode, workgroup, buffers),
            Self::Move {
                storage,
                contiguous,
                workgroup,
                buffers,
            } => assemble_move(storage, contiguous, workgroup, buffers),
        }
    }

    /// Number of specialization constants the module declares, ids `0..count`.
    pub const fn spec_constant_count(self) -> u32 {
        match self {
            Self::Elementwise { op, broadcast, .. } => {
                let inputs = op.inputs().len() as u32;
                let base = 1 + 2 * inputs + 2;
                let shape = if broadcast {
                    MAX_RANK as u32 * (1 + inputs)
                } else {
                    0
                };
                base + shape + op.extra_spec_constants()
            }
            Self::Reduce { .. } => 2 + 2 + 3,
            Self::Matmul { .. } => 3 * 2 + 4,
            Self::MaxPool { .. } => 2 + 2 + 12,
            Self::Move { contiguous, .. } => {
                if contiguous {
                    2 + 2 + 1
                } else {
                    2 + 2 + 1 + MAX_RANK as u32 * 3 + 2
                }
            }
        }
    }

    /// `OpExecutionMode LocalSize` of the module.
    pub const fn local_size(self) -> [u32; 3] {
        match self {
            Self::Elementwise { workgroup, .. }
            | Self::Reduce { workgroup, .. }
            | Self::MaxPool { workgroup, .. }
            | Self::Move { workgroup, .. } => [workgroup, 1, 1],
            Self::Matmul { tile, .. } => [tile, tile, 1],
        }
    }
}

/// Where a kernel operand lives: a descriptor-array element and a base offset in 32-bit words.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Operand {
    pub buffer: u32,
    pub base: u32,
}

impl Operand {
    fn push(self, words: &mut Vec<u32>) {
        words.push(self.buffer);
        words.push(self.base);
    }
}

/// Specialization payload of an elementwise dispatch. The declaration order in the elementwise
/// kernel is: `count`, each input operand, the output operand, then (broadcast
/// only) the output dims and each input's element strides, then the clamp bounds.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ElementwiseSpec<'a> {
    pub count: u32,
    pub inputs: &'a [Operand],
    pub output: Operand,
    /// Output shape padded with leading ones to `MAX_RANK`.
    pub dims: [u32; MAX_RANK],
    /// Per input, element strides over `dims` (0 along a broadcast dimension).
    pub strides: &'a [[u32; MAX_RANK]],
    /// `CLAMP` bounds as FP32 bit patterns.
    pub clamp: Option<[u32; 2]>,
}

impl ElementwiseSpec<'_> {
    pub fn words(&self, broadcast: bool) -> Vec<u32> {
        let mut words = vec![self.count];
        for input in self.inputs {
            input.push(&mut words);
        }
        self.output.push(&mut words);
        if broadcast {
            words.extend_from_slice(&self.dims);
            for strides in self.strides {
                words.extend_from_slice(strides);
            }
        }
        if let Some(bounds) = self.clamp {
            words.extend_from_slice(&bounds);
        }
        words
    }
}

/// Specialization payload of a reduction: input, output, `outer`, `axis`, `inner`.
pub fn reduce_spec(input: Operand, output: Operand, outer: u32, axis: u32, inner: u32) -> Vec<u32> {
    let mut words = Vec::with_capacity(7);
    input.push(&mut words);
    output.push(&mut words);
    words.extend_from_slice(&[outer, axis, inner]);
    words
}

/// Specialization payload of a MATMUL: `lhs`, `rhs`, output, then `m`, `n`, `k`, `batch`.
pub fn matmul_spec(
    lhs: Operand,
    rhs: Operand,
    output: Operand,
    m: u32,
    n: u32,
    k: u32,
    batch: u32,
) -> Vec<u32> {
    let mut words = Vec::with_capacity(10);
    lhs.push(&mut words);
    rhs.push(&mut words);
    output.push(&mut words);
    words.extend_from_slice(&[m, n, k, batch]);
    words
}

/// NHWC pooling geometry: batch, input height/width, channels, output height/width, kernel,
/// stride, and the top/left pads (the bottom/right pads only shape the output).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PoolGeometry {
    pub batch: u32,
    pub height: u32,
    pub width: u32,
    pub channels: u32,
    pub out_height: u32,
    pub out_width: u32,
    pub kernel: [u32; 2],
    pub stride: [u32; 2],
    pub pad_top: u32,
    pub pad_left: u32,
}

/// Specialization payload of a MAX_POOL2D dispatch.
pub fn max_pool_spec(input: Operand, output: Operand, geometry: PoolGeometry) -> Vec<u32> {
    let mut words = Vec::with_capacity(16);
    input.push(&mut words);
    output.push(&mut words);
    words.extend_from_slice(&[
        geometry.batch,
        geometry.height,
        geometry.width,
        geometry.channels,
        geometry.out_height,
        geometry.out_width,
        geometry.kernel[0],
        geometry.kernel[1],
        geometry.stride[0],
        geometry.stride[1],
        geometry.pad_top,
        geometry.pad_left,
    ]);
    words
}

/// Iteration space of a strided copy: `dims` (leading ones to `MAX_RANK`), element strides and
/// element offsets on each side. Strides are wrapping `u32` so a reversed axis is `-inner`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MoveGeometry {
    pub count: u32,
    pub dims: [u32; MAX_RANK],
    pub in_strides: [u32; MAX_RANK],
    pub in_offset: u32,
    pub out_strides: [u32; MAX_RANK],
    pub out_offset: u32,
}

/// Specialization payload of a copy dispatch.
pub fn move_spec(
    input: Operand,
    output: Operand,
    geometry: MoveGeometry,
    contiguous: bool,
) -> Vec<u32> {
    let mut words = Vec::with_capacity(25);
    input.push(&mut words);
    output.push(&mut words);
    words.push(geometry.count);
    if !contiguous {
        words.extend_from_slice(&geometry.dims);
        words.extend_from_slice(&geometry.in_strides);
        words.push(geometry.in_offset);
        words.extend_from_slice(&geometry.out_strides);
        words.push(geometry.out_offset);
    }
    words
}

/// Number of workgroups covering `count` items at `workgroup` invocations each, capped at
/// `limit`; the kernels loop with a grid stride so the cap only trades parallelism, never
/// coverage.
pub const fn linear_workgroups(count: u32, workgroup: u32, limit: u32) -> u32 {
    let needed = count.div_ceil(workgroup);
    if needed == 0 {
        1
    } else if needed > limit {
        limit
    } else {
        needed
    }
}

/// Workgroup counts of a tiled MATMUL over `m` rows, `n` columns, and `batch` batches.
pub const fn matmul_workgroups(m: u32, n: u32, batch: u32, tile: u32) -> [u32; 3] {
    [n.div_ceil(tile), m.div_ceil(tile), batch]
}

// ---------------------------------------------------------------------------------------------
// Module builder
// ---------------------------------------------------------------------------------------------

#[derive(Clone, PartialEq, Eq, Hash)]
enum TypeKey {
    Void,
    Bool,
    U32,
    F32,
    Vector(Id, u32),
    Pointer(u32, Id),
    RuntimeArray(Id),
    Array(Id, Id),
    Function(Id),
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum ConstKey {
    U32(u32),
    F32(u32),
    False,
}

/// Section-ordered SPIR-V assembler. Word counts and the id bound are derived; every instruction
/// is emitted opcode-first exactly as the specification tabulates it.
struct Builder {
    next_id: Id,
    capabilities: Vec<u32>,
    imports: Vec<u32>,
    memory_model: Vec<u32>,
    entry_point: Vec<u32>,
    execution_modes: Vec<u32>,
    annotations: Vec<u32>,
    declarations: Vec<u32>,
    functions: Vec<u32>,
    types: HashMap<TypeKey, Id>,
    constants: HashMap<ConstKey, Id>,
    glsl: Id,
    spec_next: u32,
    /// Where the entry block's `OpVariable Function` declarations are spliced in.
    local_variable_cursor: usize,
    interface: Vec<Id>,
    main: Id,
}

fn instruction(target: &mut Vec<u32>, opcode: u16, operands: &[u32]) {
    let word_count = u32::try_from(operands.len() + 1).expect("instruction fits");
    target.push((word_count << 16) | u32::from(opcode));
    target.extend_from_slice(operands);
}

/// Encode a literal string operand: UTF-8 bytes, NUL terminated, zero-padded to whole words.
fn literal_string(text: &str) -> Vec<u32> {
    let mut bytes = text.as_bytes().to_vec();
    bytes.push(0);
    while bytes.len() % 4 != 0 {
        bytes.push(0);
    }
    bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

impl Builder {
    fn new() -> Self {
        let mut builder = Self {
            next_id: 1,
            capabilities: Vec::new(),
            imports: Vec::new(),
            memory_model: Vec::new(),
            entry_point: Vec::new(),
            execution_modes: Vec::new(),
            annotations: Vec::new(),
            declarations: Vec::new(),
            functions: Vec::new(),
            types: HashMap::new(),
            constants: HashMap::new(),
            glsl: 0,
            spec_next: 0,
            local_variable_cursor: 0,
            interface: Vec::new(),
            main: 0,
        };
        instruction(
            &mut builder.capabilities,
            OP_CAPABILITY,
            &[CAPABILITY_SHADER],
        );
        builder.glsl = builder.id();
        let mut import = vec![builder.glsl];
        import.extend(literal_string("GLSL.std.450"));
        instruction(&mut builder.imports, OP_EXT_INST_IMPORT, &import);
        instruction(
            &mut builder.memory_model,
            OP_MEMORY_MODEL,
            &[ADDRESSING_MODEL_LOGICAL, MEMORY_MODEL_GLSL450],
        );
        builder.main = builder.id();
        builder
    }

    fn id(&mut self) -> Id {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    fn finish(mut self, local_size: [u32; 3]) -> Vec<u32> {
        let mut entry = vec![EXECUTION_MODEL_GL_COMPUTE, self.main];
        entry.extend(literal_string("main"));
        entry.extend_from_slice(&self.interface);
        instruction(&mut self.entry_point, OP_ENTRY_POINT, &entry);
        instruction(
            &mut self.execution_modes,
            OP_EXECUTION_MODE,
            &[
                self.main,
                EXECUTION_MODE_LOCAL_SIZE,
                local_size[0],
                local_size[1],
                local_size[2],
            ],
        );
        let mut words = vec![SPIRV_MAGIC, SPIRV_VERSION_1_3, 0, self.next_id, 0];
        for section in [
            &self.capabilities,
            &self.imports,
            &self.memory_model,
            &self.entry_point,
            &self.execution_modes,
            &self.annotations,
            &self.declarations,
            &self.functions,
        ] {
            words.extend_from_slice(section);
        }
        words
    }

    // -- types and constants ------------------------------------------------------------------

    fn ty(&mut self, key: TypeKey) -> Id {
        if let Some(id) = self.types.get(&key) {
            return *id;
        }
        let id = self.id();
        match &key {
            TypeKey::Void => instruction(&mut self.declarations, OP_TYPE_VOID, &[id]),
            TypeKey::Bool => instruction(&mut self.declarations, OP_TYPE_BOOL, &[id]),
            TypeKey::U32 => instruction(&mut self.declarations, OP_TYPE_INT, &[id, 32, 0]),
            TypeKey::F32 => instruction(&mut self.declarations, OP_TYPE_FLOAT, &[id, 32]),
            TypeKey::Vector(element, count) => {
                instruction(
                    &mut self.declarations,
                    OP_TYPE_VECTOR,
                    &[id, *element, *count],
                );
            }
            TypeKey::Pointer(class, pointee) => {
                instruction(
                    &mut self.declarations,
                    OP_TYPE_POINTER,
                    &[id, *class, *pointee],
                );
            }
            TypeKey::RuntimeArray(element) => {
                instruction(
                    &mut self.declarations,
                    OP_TYPE_RUNTIME_ARRAY,
                    &[id, *element],
                );
            }
            TypeKey::Array(element, length) => {
                instruction(
                    &mut self.declarations,
                    OP_TYPE_ARRAY,
                    &[id, *element, *length],
                );
            }
            TypeKey::Function(ret) => {
                instruction(&mut self.declarations, OP_TYPE_FUNCTION, &[id, *ret]);
            }
        }
        self.types.insert(key, id);
        id
    }

    fn void(&mut self) -> Id {
        self.ty(TypeKey::Void)
    }
    fn bool_ty(&mut self) -> Id {
        self.ty(TypeKey::Bool)
    }
    fn u32_ty(&mut self) -> Id {
        self.ty(TypeKey::U32)
    }
    fn f32_ty(&mut self) -> Id {
        self.ty(TypeKey::F32)
    }
    fn uvec3(&mut self) -> Id {
        let u32_ty = self.u32_ty();
        self.ty(TypeKey::Vector(u32_ty, 3))
    }
    fn pointer(&mut self, class: u32, pointee: Id) -> Id {
        self.ty(TypeKey::Pointer(class, pointee))
    }

    fn constant(&mut self, key: ConstKey) -> Id {
        if let Some(id) = self.constants.get(&key) {
            return *id;
        }
        let id = self.id();
        match key {
            ConstKey::U32(value) => {
                let ty = self.u32_ty();
                instruction(&mut self.declarations, OP_CONSTANT, &[ty, id, value]);
            }
            ConstKey::F32(bits) => {
                let ty = self.f32_ty();
                instruction(&mut self.declarations, OP_CONSTANT, &[ty, id, bits]);
            }
            ConstKey::False => {
                let ty = self.bool_ty();
                instruction(&mut self.declarations, OP_CONSTANT_FALSE, &[ty, id]);
            }
        }
        self.constants.insert(key, id);
        id
    }

    fn c_u32(&mut self, value: u32) -> Id {
        self.constant(ConstKey::U32(value))
    }
    fn c_f32(&mut self, value: f32) -> Id {
        self.constant(ConstKey::F32(value.to_bits()))
    }
    fn c_false(&mut self) -> Id {
        self.constant(ConstKey::False)
    }

    /// Declare the next `u32` specialization constant (ids are assigned in declaration order).
    fn spec_u32(&mut self, default: u32) -> Id {
        let ty = self.u32_ty();
        let id = self.id();
        instruction(&mut self.declarations, OP_SPEC_CONSTANT, &[ty, id, default]);
        instruction(
            &mut self.annotations,
            OP_DECORATE,
            &[id, DECORATION_SPEC_ID, self.spec_next],
        );
        self.spec_next += 1;
        id
    }

    fn spec_operand(&mut self) -> (Id, Id) {
        let buffer = self.spec_u32(0);
        let base = self.spec_u32(0);
        (buffer, base)
    }

    fn spec_dims(&mut self) -> [Id; MAX_RANK] {
        let mut ids = [0; MAX_RANK];
        for id in &mut ids {
            *id = self.spec_u32(1);
        }
        ids
    }

    fn spec_strides(&mut self) -> [Id; MAX_RANK] {
        let mut ids = [0; MAX_RANK];
        for id in &mut ids {
            *id = self.spec_u32(0);
        }
        ids
    }

    // -- global variables ---------------------------------------------------------------------

    fn builtin_uvec3(&mut self, builtin: u32) -> Id {
        let uvec3 = self.uvec3();
        let pointer = self.pointer(STORAGE_CLASS_INPUT, uvec3);
        let id = self.id();
        instruction(
            &mut self.declarations,
            OP_VARIABLE,
            &[pointer, id, STORAGE_CLASS_INPUT],
        );
        instruction(
            &mut self.annotations,
            OP_DECORATE,
            &[id, DECORATION_BUILT_IN, builtin],
        );
        self.interface.push(id);
        id
    }

    /// The descriptor: set 0, binding 0, `buffers` storage-buffer blocks of `{ uint words[]; }`.
    fn buffer_array(&mut self, buffers: u32) -> Id {
        let u32_ty = self.u32_ty();
        let words = self.ty(TypeKey::RuntimeArray(u32_ty));
        instruction(
            &mut self.annotations,
            OP_DECORATE,
            &[words, DECORATION_ARRAY_STRIDE, 4],
        );
        let block = self.id();
        instruction(&mut self.declarations, OP_TYPE_STRUCT, &[block, words]);
        instruction(
            &mut self.annotations,
            OP_DECORATE,
            &[block, DECORATION_BLOCK],
        );
        instruction(
            &mut self.annotations,
            OP_MEMBER_DECORATE,
            &[block, 0, DECORATION_OFFSET, 0],
        );
        let length = self.c_u32(buffers);
        let array = self.ty(TypeKey::Array(block, length));
        let pointer = self.pointer(STORAGE_CLASS_STORAGE_BUFFER, array);
        let variable = self.id();
        instruction(
            &mut self.declarations,
            OP_VARIABLE,
            &[pointer, variable, STORAGE_CLASS_STORAGE_BUFFER],
        );
        instruction(
            &mut self.annotations,
            OP_DECORATE,
            &[variable, DECORATION_DESCRIPTOR_SET, 0],
        );
        instruction(
            &mut self.annotations,
            OP_DECORATE,
            &[variable, DECORATION_BINDING, 0],
        );
        variable
    }

    /// A `Private` array of `u32` initialized with `values`; indexable at run time.
    fn private_u32_array(&mut self, values: &[u32]) -> Id {
        let u32_ty = self.u32_ty();
        let length = self.c_u32(values.len() as u32);
        let array = self.ty(TypeKey::Array(u32_ty, length));
        let elements: Vec<Id> = values.iter().map(|value| self.c_u32(*value)).collect();
        let composite = self.id();
        let mut operands = vec![array, composite];
        operands.extend_from_slice(&elements);
        instruction(&mut self.declarations, OP_CONSTANT_COMPOSITE, &operands);
        let pointer = self.pointer(STORAGE_CLASS_PRIVATE, array);
        let variable = self.id();
        instruction(
            &mut self.declarations,
            OP_VARIABLE,
            &[pointer, variable, STORAGE_CLASS_PRIVATE, composite],
        );
        variable
    }

    /// A workgroup-shared `float[length]` array.
    fn shared_f32_array(&mut self, length: u32) -> Id {
        let f32_ty = self.f32_ty();
        let length = self.c_u32(length);
        let array = self.ty(TypeKey::Array(f32_ty, length));
        let pointer = self.pointer(STORAGE_CLASS_WORKGROUP, array);
        let variable = self.id();
        instruction(
            &mut self.declarations,
            OP_VARIABLE,
            &[pointer, variable, STORAGE_CLASS_WORKGROUP],
        );
        variable
    }

    // -- function body ------------------------------------------------------------------------

    fn begin_main(&mut self) {
        let void = self.void();
        let fn_type = self.ty(TypeKey::Function(void));
        instruction(
            &mut self.functions,
            OP_FUNCTION,
            &[void, self.main, FUNCTION_CONTROL_NONE, fn_type],
        );
        let entry = self.id();
        instruction(&mut self.functions, OP_LABEL, &[entry]);
        self.local_variable_cursor = self.functions.len();
    }

    fn end_main(&mut self) {
        instruction(&mut self.functions, OP_RETURN, &[]);
        instruction(&mut self.functions, OP_FUNCTION_END, &[]);
    }

    /// A function-scope variable of `ty`, declared at the top of the entry block.
    fn local(&mut self, ty: Id) -> Id {
        let pointer = self.pointer(STORAGE_CLASS_FUNCTION, ty);
        let id = self.id();
        let mut declaration = Vec::with_capacity(4);
        instruction(
            &mut declaration,
            OP_VARIABLE,
            &[pointer, id, STORAGE_CLASS_FUNCTION],
        );
        let cursor = self.local_variable_cursor;
        self.functions
            .splice(cursor..cursor, declaration.iter().copied());
        self.local_variable_cursor += declaration.len();
        id
    }

    fn emit(&mut self, opcode: u16, operands: &[u32]) {
        instruction(&mut self.functions, opcode, operands);
    }

    fn value(&mut self, opcode: u16, ty: Id, operands: &[u32]) -> Id {
        let id = self.id();
        let mut all = Vec::with_capacity(operands.len() + 2);
        all.push(ty);
        all.push(id);
        all.extend_from_slice(operands);
        instruction(&mut self.functions, opcode, &all);
        id
    }

    fn float_value(&mut self, opcode: u16, operands: &[u32]) -> Id {
        let f32_ty = self.f32_ty();
        let id = self.value(opcode, f32_ty, operands);
        instruction(
            &mut self.annotations,
            OP_DECORATE,
            &[id, DECORATION_NO_CONTRACTION],
        );
        id
    }

    fn load(&mut self, ty: Id, pointer: Id) -> Id {
        self.value(OP_LOAD, ty, &[pointer])
    }
    fn store(&mut self, pointer: Id, value: Id) {
        self.emit(OP_STORE, &[pointer, value]);
    }
    fn access_chain(&mut self, pointer_ty: Id, base: Id, indices: &[Id]) -> Id {
        let mut operands = vec![base];
        operands.extend_from_slice(indices);
        self.value(OP_ACCESS_CHAIN, pointer_ty, &operands)
    }

    /// Component `index` of a builtin `uvec3`.
    fn builtin_component(&mut self, variable: Id, index: u32) -> Id {
        let u32_ty = self.u32_ty();
        let pointer = self.pointer(STORAGE_CLASS_INPUT, u32_ty);
        let component = self.c_u32(index);
        let chain = self.access_chain(pointer, variable, &[component]);
        self.load(u32_ty, chain)
    }

    fn iadd(&mut self, a: Id, b: Id) -> Id {
        let ty = self.u32_ty();
        self.value(OP_I_ADD, ty, &[a, b])
    }
    fn isub(&mut self, a: Id, b: Id) -> Id {
        let ty = self.u32_ty();
        self.value(OP_I_SUB, ty, &[a, b])
    }
    fn imul(&mut self, a: Id, b: Id) -> Id {
        let ty = self.u32_ty();
        self.value(OP_I_MUL, ty, &[a, b])
    }
    fn udiv(&mut self, a: Id, b: Id) -> Id {
        let ty = self.u32_ty();
        self.value(OP_U_DIV, ty, &[a, b])
    }
    fn umod(&mut self, a: Id, b: Id) -> Id {
        let ty = self.u32_ty();
        self.value(OP_U_MOD, ty, &[a, b])
    }
    fn umin(&mut self, a: Id, b: Id) -> Id {
        let ty = self.u32_ty();
        let glsl = self.glsl;
        self.value(OP_EXT_INST, ty, &[glsl, GLSL_UMIN, a, b])
    }
    fn shl(&mut self, a: Id, shift: Id) -> Id {
        let ty = self.u32_ty();
        self.value(OP_SHIFT_LEFT_LOGICAL, ty, &[a, shift])
    }
    fn shr(&mut self, a: Id, shift: Id) -> Id {
        let ty = self.u32_ty();
        self.value(OP_SHIFT_RIGHT_LOGICAL, ty, &[a, shift])
    }
    fn bor(&mut self, a: Id, b: Id) -> Id {
        let ty = self.u32_ty();
        self.value(OP_BITWISE_OR, ty, &[a, b])
    }
    fn band(&mut self, a: Id, b: Id) -> Id {
        let ty = self.u32_ty();
        self.value(OP_BITWISE_AND, ty, &[a, b])
    }
    fn bnot(&mut self, a: Id) -> Id {
        let ty = self.u32_ty();
        self.value(OP_NOT, ty, &[a])
    }
    fn ult(&mut self, a: Id, b: Id) -> Id {
        let ty = self.bool_ty();
        self.value(OP_U_LESS_THAN, ty, &[a, b])
    }
    fn uge(&mut self, a: Id, b: Id) -> Id {
        let ty = self.bool_ty();
        self.value(OP_U_GREATER_THAN_EQUAL, ty, &[a, b])
    }
    fn ieq(&mut self, a: Id, b: Id) -> Id {
        let ty = self.bool_ty();
        self.value(OP_I_EQUAL, ty, &[a, b])
    }
    fn ine(&mut self, a: Id, b: Id) -> Id {
        let ty = self.bool_ty();
        self.value(OP_I_NOT_EQUAL, ty, &[a, b])
    }
    fn land(&mut self, a: Id, b: Id) -> Id {
        let ty = self.bool_ty();
        self.value(OP_LOGICAL_AND, ty, &[a, b])
    }
    fn lor(&mut self, a: Id, b: Id) -> Id {
        let ty = self.bool_ty();
        self.value(OP_LOGICAL_OR, ty, &[a, b])
    }
    fn lxor(&mut self, a: Id, b: Id) -> Id {
        let ty = self.bool_ty();
        self.value(OP_LOGICAL_NOT_EQUAL, ty, &[a, b])
    }
    fn lnot(&mut self, a: Id) -> Id {
        let ty = self.bool_ty();
        self.value(OP_LOGICAL_NOT, ty, &[a])
    }
    fn select(&mut self, ty: Id, condition: Id, then: Id, otherwise: Id) -> Id {
        self.value(OP_SELECT, ty, &[condition, then, otherwise])
    }
    fn select_u32(&mut self, condition: Id, then: Id, otherwise: Id) -> Id {
        let ty = self.u32_ty();
        self.select(ty, condition, then, otherwise)
    }
    fn select_f32(&mut self, condition: Id, then: Id, otherwise: Id) -> Id {
        let ty = self.f32_ty();
        self.select(ty, condition, then, otherwise)
    }
    fn bitcast_f32(&mut self, word: Id) -> Id {
        let ty = self.f32_ty();
        self.value(OP_BITCAST, ty, &[word])
    }
    fn bitcast_u32(&mut self, float: Id) -> Id {
        let ty = self.u32_ty();
        self.value(OP_BITCAST, ty, &[float])
    }
    fn u_to_f(&mut self, word: Id) -> Id {
        let ty = self.f32_ty();
        self.value(OP_CONVERT_U_TO_F, ty, &[word])
    }
    fn f_to_u(&mut self, float: Id) -> Id {
        let ty = self.u32_ty();
        self.value(OP_CONVERT_F_TO_U, ty, &[float])
    }

    fn fadd(&mut self, a: Id, b: Id) -> Id {
        self.float_value(OP_F_ADD, &[a, b])
    }
    fn fsub(&mut self, a: Id, b: Id) -> Id {
        self.float_value(OP_F_SUB, &[a, b])
    }
    fn fmul(&mut self, a: Id, b: Id) -> Id {
        self.float_value(OP_F_MUL, &[a, b])
    }
    fn fdiv(&mut self, a: Id, b: Id) -> Id {
        self.float_value(OP_F_DIV, &[a, b])
    }
    fn fneg(&mut self, a: Id) -> Id {
        let ty = self.f32_ty();
        self.value(OP_F_NEGATE, ty, &[a])
    }
    fn fma_free(&mut self, a: Id, b: Id, c: Id) -> Id {
        // `a * b + c` as two separately rounded operations (never contracted).
        let product = self.fmul(a, b);
        self.fadd(product, c)
    }
    fn ext_f32(&mut self, op: u32, args: &[Id]) -> Id {
        let ty = self.f32_ty();
        let glsl = self.glsl;
        let mut operands = vec![glsl, op];
        operands.extend_from_slice(args);
        self.value(OP_EXT_INST, ty, &operands)
    }
    fn fabs(&mut self, a: Id) -> Id {
        self.ext_f32(GLSL_FABS, &[a])
    }
    fn is_nan(&mut self, a: Id) -> Id {
        let ty = self.bool_ty();
        self.value(OP_IS_NAN, ty, &[a])
    }
    fn foeq(&mut self, a: Id, b: Id) -> Id {
        let ty = self.bool_ty();
        self.value(OP_F_ORD_EQUAL, ty, &[a, b])
    }
    fn folt(&mut self, a: Id, b: Id) -> Id {
        let ty = self.bool_ty();
        self.value(OP_F_ORD_LESS_THAN, ty, &[a, b])
    }
    fn fogt(&mut self, a: Id, b: Id) -> Id {
        let ty = self.bool_ty();
        self.value(OP_F_ORD_GREATER_THAN, ty, &[a, b])
    }
    fn foge(&mut self, a: Id, b: Id) -> Id {
        let ty = self.bool_ty();
        self.value(OP_F_ORD_GREATER_THAN_EQUAL, ty, &[a, b])
    }

    /// Element `index` of a `Private` `u32` array.
    fn private_element(&mut self, array: Id, index: Id) -> Id {
        let u32_ty = self.u32_ty();
        let pointer = self.pointer(STORAGE_CLASS_PRIVATE, u32_ty);
        let chain = self.access_chain(pointer, array, &[index]);
        self.load(u32_ty, chain)
    }

    /// Index of the most significant set bit of `value` (`-1` for zero, unused here).
    fn find_msb(&mut self, value: Id) -> Id {
        let ty = self.u32_ty();
        let glsl = self.glsl;
        self.value(OP_EXT_INST, ty, &[glsl, GLSL_FIND_U_MSB, value])
    }

    /// `a * b` as a `(high, low)` pair of 32-bit words, built from four 16-bit products so no
    /// 64-bit integer capability is required.
    fn mul_wide(&mut self, a: Id, b: Id) -> (Id, Id) {
        let sixteen = self.c_u32(16);
        let mask = self.c_u32(0xffff);
        let a_lo = self.band(a, mask);
        let a_hi = self.shr(a, sixteen);
        let b_lo = self.band(b, mask);
        let b_hi = self.shr(b, sixteen);
        let ll = self.imul(a_lo, b_lo);
        let lh = self.imul(a_lo, b_hi);
        let hl = self.imul(a_hi, b_lo);
        let hh = self.imul(a_hi, b_hi);
        let ll_hi = self.shr(ll, sixteen);
        let lh_lo = self.band(lh, mask);
        let hl_lo = self.band(hl, mask);
        let middle = self.iadd(ll_hi, lh_lo);
        let middle = self.iadd(middle, hl_lo);
        let ll_lo = self.band(ll, mask);
        let middle_lo = self.band(middle, mask);
        let middle_shifted = self.shl(middle_lo, sixteen);
        let low = self.bor(ll_lo, middle_shifted);
        let lh_hi = self.shr(lh, sixteen);
        let hl_hi = self.shr(hl, sixteen);
        let middle_hi = self.shr(middle, sixteen);
        let high = self.iadd(hh, lh_hi);
        let high = self.iadd(high, hl_hi);
        let high = self.iadd(high, middle_hi);
        (high, low)
    }

    /// `a + b` as `(sum, carry)`.
    fn add_carry(&mut self, a: Id, b: Id) -> (Id, Id) {
        let sum = self.iadd(a, b);
        let carried = self.ult(sum, a);
        let one = self.c_u32(1);
        let zero = self.c_u32(0);
        let carry = self.select_u32(carried, one, zero);
        (sum, carry)
    }

    /// `2^exponent` as an FP32 bit pattern, for `exponent` inside the normal range.
    fn pow2(&mut self, exponent: Id) -> Id {
        let bias = self.c_u32(127);
        let biased = self.iadd(exponent, bias);
        let twenty_three = self.c_u32(23);
        let bits = self.shl(biased, twenty_three);
        self.bitcast_f32(bits)
    }

    fn label(&mut self, id: Id) {
        self.emit(OP_LABEL, &[id]);
    }
    fn branch(&mut self, target: Id) {
        self.emit(OP_BRANCH, &[target]);
    }
    fn branch_conditional(&mut self, condition: Id, then: Id, otherwise: Id) {
        self.emit(OP_BRANCH_CONDITIONAL, &[condition, then, otherwise]);
    }
    fn workgroup_barrier(&mut self) {
        let scope = self.c_u32(SCOPE_WORKGROUP);
        let semantics = self.c_u32(MEMORY_SEMANTICS_ACQUIRE_RELEASE_WORKGROUP);
        self.emit(OP_CONTROL_BARRIER, &[scope, scope, semantics]);
    }

    // -- structured loops ---------------------------------------------------------------------

    /// Open `for (; *counter < limit; )`: emits the header and the body label. The returned
    /// [`LoopScope`] must be closed with [`Self::end_loop`], which adds `step` to the counter.
    fn begin_loop(&mut self, counter: Id, limit: Id) -> (LoopScope, Id) {
        let scope = LoopScope {
            header: self.id(),
            body: self.id(),
            cont: self.id(),
            merge: self.id(),
        };
        self.branch(scope.header);
        self.label(scope.header);
        let u32_ty = self.u32_ty();
        let current = self.load(u32_ty, counter);
        let in_range = self.ult(current, limit);
        self.emit(OP_LOOP_MERGE, &[scope.merge, scope.cont, LOOP_CONTROL_NONE]);
        self.branch_conditional(in_range, scope.body, scope.merge);
        self.label(scope.body);
        (scope, current)
    }

    fn end_loop(&mut self, scope: LoopScope, counter: Id, step: Id) {
        self.branch(scope.cont);
        self.label(scope.cont);
        let u32_ty = self.u32_ty();
        let current = self.load(u32_ty, counter);
        let next = self.iadd(current, step);
        self.store(counter, next);
        self.branch(scope.header);
        self.label(scope.merge);
    }

    /// `if condition { then(); }` with no value flowing out.
    fn if_then(&mut self, condition: Id, then: impl FnOnce(&mut Self)) {
        let then_label = self.id();
        let merge = self.id();
        self.emit(OP_SELECTION_MERGE, &[merge, SELECTION_CONTROL_NONE]);
        self.branch_conditional(condition, then_label, merge);
        self.label(then_label);
        then(self);
        self.branch(merge);
        self.label(merge);
    }

    // -- tensor access -------------------------------------------------------------------------

    /// Pointer to word `word_index` of the operand `(buffer, base)` inside `buffers`.
    fn word_pointer(&mut self, buffers: Id, operand: (Id, Id), word_index: Id) -> Id {
        let u32_ty = self.u32_ty();
        let pointer = self.pointer(STORAGE_CLASS_STORAGE_BUFFER, u32_ty);
        let zero = self.c_u32(0);
        let index = self.iadd(operand.1, word_index);
        self.access_chain(pointer, buffers, &[operand.0, zero, index])
    }

    fn load_word(&mut self, buffers: Id, operand: (Id, Id), word_index: Id) -> Id {
        let pointer = self.word_pointer(buffers, operand, word_index);
        let u32_ty = self.u32_ty();
        self.load(u32_ty, pointer)
    }

    fn load_f32(&mut self, buffers: Id, operand: (Id, Id), element: Id) -> Id {
        let word = self.load_word(buffers, operand, element);
        self.bitcast_f32(word)
    }

    fn store_word(&mut self, buffers: Id, operand: (Id, Id), word_index: Id, value: Id) {
        let pointer = self.word_pointer(buffers, operand, word_index);
        self.store(pointer, value);
    }

    fn store_f32(&mut self, buffers: Id, operand: (Id, Id), element: Id, value: Id) {
        let word = self.bitcast_u32(value);
        self.store_word(buffers, operand, element, word);
    }

    /// Byte `element` of a byte-storage operand as a boolean (any nonzero byte is true).
    fn load_bool(&mut self, buffers: Id, operand: (Id, Id), element: Id) -> Id {
        let two = self.c_u32(2);
        let three = self.c_u32(3);
        let mask = self.c_u32(0xff);
        let zero = self.c_u32(0);
        let word_index = self.shr(element, two);
        let word = self.load_word(buffers, operand, word_index);
        let lane = self.band(element, three);
        let eight = self.c_u32(8);
        let shift = self.imul(lane, eight);
        let shifted = self.shr(word, shift);
        let byte = self.band(shifted, mask);
        self.ine(byte, zero)
    }

    /// Write byte `element` of a byte-storage operand as canonical `0`/`1` without touching the
    /// other bytes of its word: clear with `OpAtomicAnd`, then set with `OpAtomicOr`.
    fn store_bool(&mut self, buffers: Id, operand: (Id, Id), element: Id, value: Id) {
        let two = self.c_u32(2);
        let three = self.c_u32(3);
        let mask = self.c_u32(0xff);
        let one = self.c_u32(1);
        let zero = self.c_u32(0);
        let word_index = self.shr(element, two);
        let pointer = self.word_pointer(buffers, operand, word_index);
        let lane = self.band(element, three);
        let eight = self.c_u32(8);
        let shift = self.imul(lane, eight);
        let clear = self.shl(mask, shift);
        let clear = self.bnot(clear);
        let byte = self.select_u32(value, one, zero);
        let set = self.shl(byte, shift);
        let scope = self.c_u32(SCOPE_DEVICE);
        let semantics = self.c_u32(MEMORY_SEMANTICS_RELAXED);
        let u32_ty = self.u32_ty();
        self.value(OP_ATOMIC_AND, u32_ty, &[pointer, scope, semantics, clear]);
        self.value(OP_ATOMIC_OR, u32_ty, &[pointer, scope, semantics, set]);
    }

    /// Decompose linear index `i` over `dims` (last dimension fastest) and accumulate per-operand
    /// element indices from `strides`.
    fn strided_indices(
        &mut self,
        i: Id,
        dims: &[Id; MAX_RANK],
        strides: &[[Id; MAX_RANK]],
    ) -> Vec<Id> {
        let zero = self.c_u32(0);
        let mut indices = vec![zero; strides.len()];
        let mut remainder = i;
        for d in (0..MAX_RANK).rev() {
            let coordinate = self.umod(remainder, dims[d]);
            remainder = self.udiv(remainder, dims[d]);
            for (operand, stride) in strides.iter().enumerate() {
                let term = self.imul(coordinate, stride[d]);
                indices[operand] = self.iadd(indices[operand], term);
            }
        }
        indices
    }

    /// The grid-stride prologue: `counter = gid.x`, returning `(counter, stride)` where stride is
    /// `NumWorkgroups.x * workgroup`.
    fn grid_stride(&mut self, workgroup: u32) -> (Id, Id) {
        let gid = self.builtin_uvec3(BUILT_IN_GLOBAL_INVOCATION_ID);
        let num_workgroups = self.builtin_uvec3(BUILT_IN_NUM_WORKGROUPS);
        self.begin_main();
        let u32_ty = self.u32_ty();
        let counter = self.local(u32_ty);
        let start = self.builtin_component(gid, 0);
        self.store(counter, start);
        let groups = self.builtin_component(num_workgroups, 0);
        let size = self.c_u32(workgroup);
        let stride = self.imul(groups, size);
        (counter, stride)
    }

    // -- scalar lanes ---------------------------------------------------------------------------

    /// TOSA `apply_max_s(a, b)`: `a >= b ? a : b` with the NaN mode applied first.
    fn apply_max(&mut self, a: Id, b: Id, nan_mode: NanMode) -> Id {
        let ordered = self.foge(a, b);
        let picked = self.select_f32(ordered, a, b);
        self.apply_nan_mode(a, b, picked, nan_mode)
    }

    /// TOSA `apply_min_s(a, b)`: `a < b ? a : b` with the NaN mode applied first.
    fn apply_min(&mut self, a: Id, b: Id, nan_mode: NanMode) -> Id {
        let ordered = self.folt(a, b);
        let picked = self.select_f32(ordered, a, b);
        self.apply_nan_mode(a, b, picked, nan_mode)
    }

    fn apply_nan_mode(&mut self, a: Id, b: Id, picked: Id, nan_mode: NanMode) -> Id {
        let a_nan = self.is_nan(a);
        let b_nan = self.is_nan(b);
        match nan_mode {
            NanMode::Propagate => {
                let nan = self.c_f32(f32::NAN);
                let any_nan = self.lor(a_nan, b_nan);
                self.select_f32(any_nan, nan, picked)
            }
            NanMode::Ignore => {
                let without_b = self.select_f32(b_nan, a, picked);
                self.select_f32(a_nan, b, without_b)
            }
        }
    }

    /// Horner evaluation of `Σ coefficients[i] * x^i`, highest degree first in `coefficients`.
    fn horner(&mut self, x: Id, coefficients: &[f32]) -> Id {
        let mut acc = self.c_f32(coefficients[0]);
        for coefficient in &coefficients[1..] {
            let c = self.c_f32(*coefficient);
            acc = self.fma_free(acc, x, c);
        }
        acc
    }

    /// Cephes octant reduction: the three-part π/4 subtraction, exact while the octant count is
    /// exactly representable. Returns `(octant & 7, z)` with the octant already bumped to even,
    /// so `z = |x| - octant·π/4` lies in `[-π/4, π/4]`.
    fn cephes_reduce(&mut self, magnitude: Id) -> (Id, Id) {
        let four_over_pi = self.c_f32(1.273_239_5);
        let scaled = self.fmul(magnitude, four_over_pi);
        let octant = self.f_to_u(scaled);
        let one = self.c_u32(1);
        let zero = self.c_u32(0);
        let odd = self.band(octant, one);
        let is_odd = self.ine(odd, zero);
        let bumped = self.iadd(octant, one);
        let octant = self.select_u32(is_odd, bumped, octant);
        let y = self.u_to_f(octant);
        // Extended-precision modular arithmetic: |x| - y·(DP1 + DP2 + DP3).
        let dp1 = self.c_f32(0.785_156_25);
        let dp2 = self.c_f32(2.418_756_5e-4);
        let dp3 = self.c_f32(3.774_895e-8);
        let t1 = self.fmul(y, dp1);
        let r = self.fsub(magnitude, t1);
        let t2 = self.fmul(y, dp2);
        let r = self.fsub(r, t2);
        let t3 = self.fmul(y, dp3);
        let z = self.fsub(r, t3);
        let seven = self.c_u32(7);
        let octant = self.band(octant, seven);
        (octant, z)
    }

    /// Payne–Hanek octant reduction: exact for every finite magnitude, at the cost of a
    /// 128-bit window of 2/π and one 24×128-bit integer multiply.
    ///
    /// `|x| = m·2^(e-149)` with `m` the 24-bit significand and `e` the biased exponent, so
    /// `|x|·4/π = m·2^(e-148)·(2/π)`. Every 2/π bit whose product weight is an integer multiple
    /// of eight leaves the octant unchanged, so only the 128 bits starting at bit `e-120` of the
    /// biased stream contribute: their product with `m` carries the low three integer bits and
    /// 64 fraction bits of `|x|·4/π`. The fraction is renormalized before it becomes a float, so
    /// arguments that fall close to a multiple of π/4 keep their relative accuracy.
    ///
    /// Returns `(octant & 7, z)` on the same contract as [`Self::cephes_reduce`].
    fn payne_hanek_reduce(&mut self, magnitude: Id) -> (Id, Id) {
        let table = self.private_u32_array(TWO_OVER_PI_BITS);
        let bits = self.bitcast_u32(magnitude);
        let twenty_three = self.c_u32(23);
        let exponent_mask = self.c_u32(0xff);
        let exponent = self.shr(bits, twenty_three);
        let exponent = self.band(exponent, exponent_mask);
        let significand_mask = self.c_u32(0x007f_ffff);
        let implicit = self.c_u32(0x0080_0000);
        let significand = self.band(bits, significand_mask);
        let m = self.bor(significand, implicit);

        // Window offset into the biased bit stream; the leading zero word keeps it positive for
        // every magnitude this path serves (|x| >= 8192, so the exponent is at least 140).
        let bias = self.c_u32(120);
        let offset = self.isub(exponent, bias);
        let five = self.c_u32(5);
        let thirty_one = self.c_u32(31);
        let base = self.shr(offset, five);
        let shift = self.band(offset, thirty_one);
        let thirty_two = self.c_u32(32);
        let complement = self.isub(thirty_two, shift);
        let complement = self.band(complement, thirty_one);
        let zero = self.c_u32(0);
        let aligned = self.ieq(shift, zero);
        let mut window = [0; 4];
        for (index, slot) in window.iter_mut().enumerate() {
            let step = self.c_u32(index as u32);
            let first = self.iadd(base, step);
            let one = self.c_u32(1);
            let second = self.iadd(first, one);
            let high = self.private_element(table, first);
            let low = self.private_element(table, second);
            let high = self.shl(high, shift);
            let low = self.shr(low, complement);
            let low = self.select_u32(aligned, zero, low);
            *slot = self.bor(high, low);
        }

        // Schoolbook m × window, most significant window word first.
        let mut product = [0; 5];
        let mut carry = zero;
        for (index, word) in window.iter().rev().enumerate() {
            let (high, low) = self.mul_wide(m, *word);
            let (sum, overflow) = self.add_carry(low, carry);
            product[index] = sum;
            carry = self.iadd(high, overflow);
        }
        product[4] = carry;

        // Bit 125 of the product is the binary point: three integer bits above it, the fraction
        // below.
        let twenty_nine = self.c_u32(29);
        let three = self.c_u32(3);
        let seven = self.c_u32(7);
        let integer_low = self.shr(product[3], twenty_nine);
        let integer_high = self.shl(product[4], three);
        let integer = self.bor(integer_low, integer_high);
        let integer = self.band(integer, seven);
        let fraction_high = self.shl(product[3], three);
        let carry_in = self.shr(product[2], twenty_nine);
        let fraction_high = self.bor(fraction_high, carry_in);
        let fraction_low = self.shl(product[2], three);
        let carry_in = self.shr(product[1], twenty_nine);
        let fraction_low = self.bor(fraction_low, carry_in);

        // Bump an odd octant to the next even one; the fraction becomes negative.
        let one = self.c_u32(1);
        let odd = self.band(integer, one);
        let is_odd = self.ine(odd, zero);
        let bumped = self.iadd(integer, one);
        let octant = self.select_u32(is_odd, bumped, integer);
        let octant = self.band(octant, seven);
        // Two's complement of the 64-bit fraction when it is subtracted from one.
        let negated_low = self.bnot(fraction_low);
        let (negated_low, overflow) = self.add_carry(negated_low, one);
        let negated_high = self.bnot(fraction_high);
        let negated_high = self.iadd(negated_high, overflow);
        let magnitude_high = self.select_u32(is_odd, negated_high, fraction_high);
        let magnitude_low = self.select_u32(is_odd, negated_low, fraction_low);

        // Renormalize: the leading set bit sets the exponent, the next 24 bits the significand.
        let nonzero_high = self.ine(magnitude_high, zero);
        let msb_high = self.find_msb(magnitude_high);
        let msb_low = self.find_msb(magnitude_low);
        let shift_high = self.isub(thirty_one, msb_high);
        let shift_low = self.isub(thirty_one, msb_low);
        let shift_low_total = self.iadd(shift_low, thirty_two);
        let leading = self.select_u32(nonzero_high, shift_high, shift_low_total);
        // `magnitude_high << leading` with the bits shifted in from `magnitude_low`, or the low
        // word alone once the high word is empty.
        let complement = self.isub(thirty_two, leading);
        let complement_masked = self.band(complement, thirty_one);
        let aligned = self.ieq(leading, zero);
        let top_high = self.shl(magnitude_high, leading);
        let carried = self.shr(magnitude_low, complement_masked);
        let carried = self.select_u32(aligned, zero, carried);
        let top_high = self.bor(top_high, carried);
        let top_low = self.shl(magnitude_low, shift_low);
        let top = self.select_u32(nonzero_high, top_high, top_low);
        let eight = self.c_u32(8);
        let significand = self.shr(top, eight);
        let value = self.u_to_f(significand);
        // The renormalized fraction is `significand · 2^-(24 + leading)`.
        let twenty_four = self.c_u32(24);
        let exponent = self.iadd(twenty_four, leading);
        let exponent = self.isub(zero, exponent);
        let scale = self.pow2(exponent);
        let fraction = self.fmul(value, scale);
        // A fraction of exactly zero renormalizes to nothing; the octant is already correct.
        let empty_low = self.ieq(magnitude_low, zero);
        let empty_high = self.ieq(magnitude_high, zero);
        let empty = self.land(empty_low, empty_high);
        let zero_f = self.c_f32(0.0);
        let fraction = self.select_f32(empty, zero_f, fraction);
        let negated = self.fneg(fraction);
        let fraction = self.select_f32(is_odd, negated, fraction);
        // z = fraction · π/4, split so the product keeps more than binary32 precision.
        let pi_over_four_high = self.c_f32(0.785_156_25);
        let pi_over_four_low = self.c_f32(2.419_134e-4);
        let high = self.fmul(fraction, pi_over_four_high);
        let low = self.fmul(fraction, pi_over_four_low);
        let z = self.fadd(high, low);
        (octant, z)
    }

    /// `sin` or `cos` from crate-authored range reduction and minimax polynomials: the Cephes
    /// three-part subtraction below [`SINCOS_FAST_RANGE`], Payne–Hanek above it, and NaN for a
    /// non-finite argument. The driver's own `sin`/`cos` are never used: Vulkan bounds them only
    /// to an absolute 2⁻¹¹ inside an unspecified range, which is not a numerical contract a tier
    /// can advertise.
    fn sincos(&mut self, x: Id, cosine: bool) -> Id {
        let magnitude = self.fabs(x);
        let (fast_octant, fast_z) = self.cephes_reduce(magnitude);
        let (exact_octant, exact_z) = self.payne_hanek_reduce(magnitude);
        let threshold = self.c_f32(SINCOS_FAST_RANGE);
        let fast = self.folt(magnitude, threshold);
        let octant = self.select_u32(fast, fast_octant, exact_octant);
        let z = self.select_f32(fast, fast_z, exact_z);

        let three = self.c_u32(3);
        let reflect = self.ult(three, octant);
        let four = self.c_u32(4);
        let reduced = self.isub(octant, four);
        let octant = self.select_u32(reflect, reduced, octant);
        let zz = self.fmul(z, z);
        // cos polynomial: 1 - zz/2 + zz² · (c0 zz² + c1 zz + c2)
        let cos_poly = self.horner(zz, &[2.443_315_7e-5, -1.388_731_6e-3, 4.166_664_6e-2]);
        let zz2 = self.fmul(zz, zz);
        let cos_tail = self.fmul(cos_poly, zz2);
        let half = self.c_f32(0.5);
        let half_zz = self.fmul(half, zz);
        let cos_value = self.fsub(cos_tail, half_zz);
        let one_f = self.c_f32(1.0);
        let cos_value = self.fadd(cos_value, one_f);
        // sin polynomial: z + z · zz · (s0 zz² + s1 zz + s2)
        let sin_poly = self.horner(zz, &[-1.951_529_6e-4, 8.332_161e-3, -1.666_665_5e-1]);
        let sin_tail = self.fmul(sin_poly, zz);
        let sin_tail = self.fmul(sin_tail, z);
        let sin_value = self.fadd(sin_tail, z);
        let one = self.c_u32(1);
        let two = self.c_u32(2);
        let octant_is_1 = self.ieq(octant, one);
        let octant_is_2 = self.ieq(octant, two);
        let middle = self.lor(octant_is_1, octant_is_2);
        // sin uses the cos polynomial in octants 1..2, cos uses the sin polynomial there.
        let value = if cosine {
            self.select_f32(middle, sin_value, cos_value)
        } else {
            self.select_f32(middle, cos_value, sin_value)
        };
        let mut negate = reflect;
        if cosine {
            let upper_half = self.ult(one, octant);
            negate = self.lxor(negate, upper_half);
        } else {
            // The sign bit, not an ordered comparison: `sin(-0)` is `-0`.
            let negative_input = self.signbit(x);
            negate = self.lxor(negate, negative_input);
        }
        let negated = self.fneg(value);
        let value = self.select_f32(negate, negated, value);
        // `sin` and `cos` of a non-finite argument are NaN, and no reduction defines one.
        let infinity = self.c_f32(f32::INFINITY);
        let finite = self.folt(magnitude, infinity);
        let nan = self.c_f32(f32::NAN);
        self.select_f32(finite, value, nan)
    }

    /// Cephes `tanhf`: odd polynomial below 0.625, `1 - 2 / (exp(2|x|) + 1)` above.
    fn tanh(&mut self, x: Id) -> Id {
        let magnitude = self.fabs(x);
        let square = self.fmul(x, x);
        let poly = self.horner(
            square,
            &[
                -5.704_988_7e-3,
                2.063_909e-2,
                -5.373_971_6e-2,
                1.333_144_2e-1,
                -3.333_328e-1,
            ],
        );
        let small = self.fmul(poly, square);
        let small = self.fma_free(small, x, x);
        let two = self.c_f32(2.0);
        let doubled = self.fmul(magnitude, two);
        let exp = self.ext_f32(GLSL_EXP, &[doubled]);
        let one = self.c_f32(1.0);
        let denominator = self.fadd(exp, one);
        let ratio = self.fdiv(two, denominator);
        let large = self.fsub(one, ratio);
        let negative = self.signbit(x);
        let negated = self.fneg(large);
        let large = self.select_f32(negative, negated, large);
        let threshold = self.c_f32(0.625);
        let use_large = self.foge(magnitude, threshold);
        let value = self.select_f32(use_large, large, small);
        // `tanh(±0) = ±0`: the polynomial's `0 + (-0)` would round the sign away.
        let zero = self.c_f32(0.0);
        let is_zero = self.foeq(x, zero);
        self.select_f32(is_zero, x, value)
    }

    /// `erf`: the alternating Maclaurin series through `x^21` below `|x| = 1` (relative accuracy
    /// near zero), and `1 - erfc` with the Chebyshev-fitted `erfc` rational form above it
    /// (fractional error below 1.2e-7 everywhere).
    fn erf(&mut self, x: Id) -> Id {
        let magnitude = self.fabs(x);
        let square = self.fmul(x, x);
        let series = self.horner(
            square,
            &[
                1.0 / 76_204_800.0,
                -1.0 / 6_894_720.0,
                1.0 / 685_440.0,
                -1.0 / 75_600.0,
                1.0 / 9_360.0,
                -1.0 / 1_320.0,
                1.0 / 216.0,
                -1.0 / 42.0,
                1.0 / 10.0,
                -1.0 / 3.0,
                1.0,
            ],
        );
        let two_over_sqrt_pi = self.c_f32(core::f32::consts::FRAC_2_SQRT_PI);
        let series = self.fmul(series, two_over_sqrt_pi);
        let series = self.fmul(series, x);
        let half = self.c_f32(0.5);
        let one = self.c_f32(1.0);
        let half_magnitude = self.fmul(magnitude, half);
        let denominator = self.fadd(one, half_magnitude);
        let t = self.fdiv(one, denominator);
        let poly = self.horner(
            t,
            &[
                0.170_872_77,
                -0.822_152_23,
                1.488_515_9,
                -1.135_204,
                0.278_868_07,
                -0.186_288_06,
                0.096_784_18,
                0.374_091_96,
                1.000_023_7,
                -1.265_512_2,
            ],
        );
        let exponent = self.fsub(poly, square);
        let exp = self.ext_f32(GLSL_EXP, &[exponent]);
        let erfc = self.fmul(t, exp);
        let tail = self.fsub(one, erfc);
        let zero = self.c_f32(0.0);
        let negative = self.folt(x, zero);
        let negated = self.fneg(tail);
        let tail = self.select_f32(negative, negated, tail);
        let use_series = self.folt(magnitude, one);
        let value = self.select_f32(use_series, series, tail);
        // `erf(±0) = ±0` exactly, whatever the series rounds to.
        let is_zero = self.foeq(x, zero);
        self.select_f32(is_zero, x, value)
    }

    /// IEEE-style `pow` on top of the built-in: negative bases with integral exponents keep the
    /// parity sign, negative bases with fractional exponents are NaN, `pow(x, 0) = 1`, and
    /// `pow(1, y) = 1`.
    fn pow(&mut self, x: Id, y: Id) -> Id {
        let magnitude = self.fabs(x);
        let raw = self.ext_f32(GLSL_POW, &[magnitude, y]);
        let zero = self.c_f32(0.0);
        let one = self.c_f32(1.0);
        let half = self.c_f32(0.5);
        let floor_y = self.ext_f32(GLSL_FLOOR, &[y]);
        let y_integral = self.foeq(floor_y, y);
        let half_y = self.fmul(y, half);
        let floor_half = self.ext_f32(GLSL_FLOOR, &[half_y]);
        let y_even = self.foeq(floor_half, half_y);
        let y_odd = self.lnot(y_even);
        let y_odd = self.land(y_integral, y_odd);
        let negated = self.fneg(raw);
        let signed = self.select_f32(y_odd, negated, raw);
        let nan = self.c_f32(f32::NAN);
        let negative_base = self.select_f32(y_integral, signed, nan);
        let x_negative = self.folt(x, zero);
        // A zero base: the built-in is undefined for y <= 0, so spell the limits out.
        let x_zero = self.foeq(x, zero);
        let y_positive = self.fogt(y, zero);
        let inf = self.c_f32(f32::INFINITY);
        let zero_base = self.select_f32(y_positive, zero, inf);
        let zero_base_neg = self.fneg(zero_base);
        let x_sign = self.bitcast_f32_sign(x);
        let x_sign_negative = self.folt(x_sign, zero);
        let zero_base_signed = self.land(x_sign_negative, y_odd);
        let zero_base = self.select_f32(zero_base_signed, zero_base_neg, zero_base);
        let value = self.select_f32(x_negative, negative_base, raw);
        let value = self.select_f32(x_zero, zero_base, value);
        let y_zero = self.foeq(y, zero);
        let value = self.select_f32(y_zero, one, value);
        let x_one = self.foeq(x, one);
        self.select_f32(x_one, one, value)
    }

    /// Whether `x` carries the IEEE sign bit; true for `-0.0`, unlike an ordered `x < 0`.
    fn signbit(&mut self, x: Id) -> Id {
        let bits = self.bitcast_u32(x);
        let sign_bit = self.c_u32(0x8000_0000);
        let masked = self.band(bits, sign_bit);
        let zero = self.c_u32(0);
        self.ine(masked, zero)
    }

    /// `-1.0` when `x` carries the sign bit (including `-0.0`), else `+1.0`.
    fn bitcast_f32_sign(&mut self, x: Id) -> Id {
        let bits = self.bitcast_u32(x);
        let sign_bit = self.c_u32(0x8000_0000);
        let masked = self.band(bits, sign_bit);
        let zero = self.c_u32(0);
        let negative = self.ine(masked, zero);
        let minus_one = self.c_f32(-1.0);
        let plus_one = self.c_f32(1.0);
        self.select_f32(negative, minus_one, plus_one)
    }

    /// The scalar lane of `op` over already-loaded inputs (`f32` ids for word inputs, `bool` ids
    /// for byte inputs), yielding an `f32` or `bool` id per [`ElementwiseOp::output`].
    fn elementwise_lane(
        &mut self,
        op: ElementwiseOp,
        inputs: &[Id],
        clamp: Option<(Id, Id)>,
    ) -> Id {
        let x = inputs[0];
        match op {
            ElementwiseOp::Abs => self.fabs(x),
            ElementwiseOp::Ceil => self.ext_f32(GLSL_CEIL, &[x]),
            ElementwiseOp::Floor => self.ext_f32(GLSL_FLOOR, &[x]),
            ElementwiseOp::Cos => self.sincos(x, true),
            ElementwiseOp::Sin => self.sincos(x, false),
            ElementwiseOp::Erf => self.erf(x),
            ElementwiseOp::Exp => self.ext_f32(GLSL_EXP, &[x]),
            ElementwiseOp::Log => self.ext_f32(GLSL_LOG, &[x]),
            ElementwiseOp::Negate => self.fneg(x),
            ElementwiseOp::Reciprocal => {
                let one = self.c_f32(1.0);
                self.fdiv(one, x)
            }
            ElementwiseOp::Rsqrt => self.ext_f32(GLSL_INVERSE_SQRT, &[x]),
            ElementwiseOp::Sigmoid => {
                let negated = self.fneg(x);
                let exp = self.ext_f32(GLSL_EXP, &[negated]);
                let one = self.c_f32(1.0);
                let denominator = self.fadd(one, exp);
                self.fdiv(one, denominator)
            }
            ElementwiseOp::Tanh => self.tanh(x),
            ElementwiseOp::Clamp(nan_mode) => {
                let (lo_bits, hi_bits) = clamp.expect("clamp bounds");
                let lo = self.bitcast_f32(lo_bits);
                let hi = self.bitcast_f32(hi_bits);
                let floored = self.apply_max(x, lo, nan_mode);
                self.apply_min(floored, hi, nan_mode)
            }
            ElementwiseOp::Add => self.fadd(x, inputs[1]),
            ElementwiseOp::Sub => self.fsub(x, inputs[1]),
            ElementwiseOp::Mul => self.fmul(x, inputs[1]),
            ElementwiseOp::Pow => self.pow(x, inputs[1]),
            ElementwiseOp::Maximum(nan_mode) => self.apply_max(x, inputs[1], nan_mode),
            ElementwiseOp::Minimum(nan_mode) => self.apply_min(x, inputs[1], nan_mode),
            ElementwiseOp::Equal => self.foeq(x, inputs[1]),
            ElementwiseOp::Greater => self.fogt(x, inputs[1]),
            ElementwiseOp::GreaterEqual => self.foge(x, inputs[1]),
            ElementwiseOp::LogicalAnd => self.land(x, inputs[1]),
            ElementwiseOp::LogicalOr => self.lor(x, inputs[1]),
            ElementwiseOp::LogicalXor => self.lxor(x, inputs[1]),
            ElementwiseOp::LogicalNot => self.lnot(x),
            ElementwiseOp::Select => self.select_f32(x, inputs[1], inputs[2]),
            ElementwiseOp::CopyBytes => x,
        }
    }
}

#[derive(Clone, Copy)]
struct LoopScope {
    header: Id,
    body: Id,
    cont: Id,
    merge: Id,
}

// ---------------------------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------------------------

/// Elementwise kernel: grid-stride over `count` output elements.
///
/// Specialization order: `count`; per input `(buffer, base)`; output `(buffer, base)`; when
/// `broadcast`, output `dims[MAX_RANK]` then per input `strides[MAX_RANK]`; then the operator's
/// trailing constants (`CLAMP`: `lo`, `hi` bit patterns).
fn assemble_elementwise(
    op: ElementwiseOp,
    broadcast: bool,
    workgroup: u32,
    buffers: u32,
) -> Vec<u32> {
    let mut b = Builder::new();
    let array = b.buffer_array(buffers);
    let count = b.spec_u32(1);
    let input_storage = op.inputs();
    let inputs: Vec<(Id, Id)> = input_storage.iter().map(|_| b.spec_operand()).collect();
    let output = b.spec_operand();
    let shape = broadcast.then(|| {
        let dims = b.spec_dims();
        let strides: Vec<[Id; MAX_RANK]> = input_storage.iter().map(|_| b.spec_strides()).collect();
        (dims, strides)
    });
    let clamp = matches!(op, ElementwiseOp::Clamp(_)).then(|| {
        let lo = b.spec_u32(0);
        let hi = b.spec_u32(0);
        (lo, hi)
    });

    let (counter, stride) = b.grid_stride(workgroup);
    let (scope, i) = b.begin_loop(counter, count);
    let indices = match &shape {
        Some((dims, strides)) => b.strided_indices(i, dims, strides),
        None => vec![i; inputs.len()],
    };
    let mut values = Vec::with_capacity(inputs.len());
    for (k, operand) in inputs.iter().enumerate() {
        let value = match input_storage[k] {
            Storage::Word => b.load_f32(array, *operand, indices[k]),
            Storage::Byte => b.load_bool(array, *operand, indices[k]),
        };
        values.push(value);
    }
    let result = b.elementwise_lane(op, &values, clamp);
    match op.output() {
        Storage::Word => b.store_f32(array, output, i, result),
        Storage::Byte => b.store_bool(array, output, i, result),
    }
    b.end_loop(scope, counter, stride);
    b.end_main();
    b.finish([workgroup, 1, 1])
}

/// Reduction kernel: one invocation per `(outer, inner)` output element folds the axis in
/// ascending order.
///
/// Specialization order: input `(buffer, base)`, output `(buffer, base)`, `outer`, `axis`,
/// `inner`.
fn assemble_reduce(op: ReduceOp, workgroup: u32, buffers: u32) -> Vec<u32> {
    let mut b = Builder::new();
    let array = b.buffer_array(buffers);
    let input = b.spec_operand();
    let output = b.spec_operand();
    let outer = b.spec_u32(1);
    let axis = b.spec_u32(1);
    let inner = b.spec_u32(1);

    let (counter, stride) = b.grid_stride(workgroup);
    let u32_ty = b.u32_ty();
    let f32_ty = b.f32_ty();
    let acc_var = b.local(f32_ty);
    let index_var = b.local(u32_ty);
    let done_var = {
        let bool_ty = b.bool_ty();
        b.local(bool_ty)
    };
    let a_var = b.local(u32_ty);
    let count = b.imul(outer, inner);
    let (scope, o) = b.begin_loop(counter, count);
    let outer_index = b.udiv(o, inner);
    let inner_index = b.umod(o, inner);
    let row = b.imul(outer_index, axis);
    let row = b.imul(row, inner);
    let base = b.iadd(row, inner_index);
    let init = match op {
        ReduceOp::Sum => b.c_f32(0.0),
        ReduceOp::Product => b.c_f32(1.0),
        ReduceOp::Max(_) | ReduceOp::ArgMax(_) => b.c_f32(f32::NEG_INFINITY),
        ReduceOp::Min(_) => b.c_f32(f32::INFINITY),
    };
    b.store(acc_var, init);
    let zero = b.c_u32(0);
    b.store(index_var, zero);
    let false_id = b.c_false();
    b.store(done_var, false_id);
    b.store(a_var, zero);
    let one = b.c_u32(1);
    let (inner_scope, a) = b.begin_loop(a_var, axis);
    let offset = b.imul(a, inner);
    let element = b.iadd(base, offset);
    let value = b.load_f32(array, input, element);
    let acc = b.load(f32_ty, acc_var);
    match op {
        ReduceOp::Sum => {
            let next = b.fadd(acc, value);
            b.store(acc_var, next);
        }
        ReduceOp::Product => {
            let next = b.fmul(acc, value);
            b.store(acc_var, next);
        }
        ReduceOp::Max(nan_mode) => {
            let next = b.apply_max(acc, value, nan_mode);
            b.store(acc_var, next);
        }
        ReduceOp::Min(nan_mode) => {
            let next = b.apply_min(acc, value, nan_mode);
            b.store(acc_var, next);
        }
        ReduceOp::ArgMax(nan_mode) => {
            let bool_ty = b.bool_ty();
            let done = b.load(bool_ty, done_var);
            let index = b.load(u32_ty, index_var);
            let greater = b.fogt(value, acc);
            let value_nan = b.is_nan(value);
            let take = match nan_mode {
                NanMode::Propagate => {
                    // The first NaN wins and freezes the result.
                    let candidate = b.lor(greater, value_nan);
                    let not_done = b.lnot(done);
                    let take = b.land(candidate, not_done);
                    let next_done = b.lor(done, value_nan);
                    b.store(done_var, next_done);
                    take
                }
                NanMode::Ignore => {
                    let ordered = b.lnot(value_nan);
                    b.land(greater, ordered)
                }
            };
            let next_acc = b.select_f32(take, value, acc);
            let next_index = b.select_u32(take, a, index);
            b.store(acc_var, next_acc);
            b.store(index_var, next_index);
        }
    }
    b.end_loop(inner_scope, a_var, one);
    match op {
        ReduceOp::ArgMax(_) => {
            let index = b.load(u32_ty, index_var);
            b.store_word(array, output, o, index);
        }
        _ => {
            let acc = b.load(f32_ty, acc_var);
            b.store_f32(array, output, o, acc);
        }
    }
    b.end_loop(scope, counter, stride);
    b.end_main();
    b.finish([workgroup, 1, 1])
}

/// FP32 batched MATMUL over `tile × tile` workgroup tiles of both operands.
///
/// `out[b, m, n] = Σ_k lhs[b, m, k] · rhs[b, k, n]`, accumulated in ascending `k` with separately
/// rounded multiply and add, so the result is bit-identical to the untiled loop. Out-of-range
/// tile loads read element zero and contribute nothing: the inner loop bound is `min(tile,
/// k - k0)`, never a padded zero product, so signed zeros survive.
///
/// Specialization order: `lhs`, `rhs`, output `(buffer, base)`; `m`, `n`, `k`, `batch`.
fn assemble_matmul(tile: u32, buffers: u32) -> Vec<u32> {
    let mut b = Builder::new();
    let array = b.buffer_array(buffers);
    let lhs = b.spec_operand();
    let rhs = b.spec_operand();
    let output = b.spec_operand();
    let m = b.spec_u32(1);
    let n = b.spec_u32(1);
    let k = b.spec_u32(1);
    let _batch = b.spec_u32(1);
    let lhs_tile = b.shared_f32_array(tile * tile);
    let rhs_tile = b.shared_f32_array(tile * tile);
    let local_id = b.builtin_uvec3(BUILT_IN_LOCAL_INVOCATION_ID);
    let group_id = b.builtin_uvec3(BUILT_IN_WORKGROUP_ID);

    b.begin_main();
    let u32_ty = b.u32_ty();
    let f32_ty = b.f32_ty();
    let acc_var = b.local(f32_ty);
    let t_var = b.local(u32_ty);
    let kk_var = b.local(u32_ty);
    let tx = b.builtin_component(local_id, 0);
    let ty = b.builtin_component(local_id, 1);
    let gx = b.builtin_component(group_id, 0);
    let gy = b.builtin_component(group_id, 1);
    let z = b.builtin_component(group_id, 2);
    let tile_c = b.c_u32(tile);
    let row = b.imul(gy, tile_c);
    let row = b.iadd(row, ty);
    let col = b.imul(gx, tile_c);
    let col = b.iadd(col, tx);
    let row_ok = b.ult(row, m);
    let col_ok = b.ult(col, n);
    let zero = b.c_u32(0);
    let one = b.c_u32(1);
    let zero_f = b.c_f32(0.0);
    b.store(acc_var, zero_f);
    b.store(t_var, zero);
    // lhs batch base: z * m * k; rhs batch base: z * k * n.
    let lhs_batch = b.imul(z, m);
    let lhs_batch = b.imul(lhs_batch, k);
    let rhs_batch = b.imul(z, k);
    let rhs_batch = b.imul(rhs_batch, n);
    let lhs_row = b.imul(row, k);
    let lhs_row = b.iadd(lhs_batch, lhs_row);
    let rhs_col = b.iadd(rhs_batch, col);
    let local_index = b.imul(ty, tile_c);
    let local_index = b.iadd(local_index, tx);
    let workgroup_ptr = {
        let f32_ty = b.f32_ty();
        b.pointer(STORAGE_CLASS_WORKGROUP, f32_ty)
    };
    let tiles = {
        let k_plus = b.iadd(k, tile_c);
        let k_plus = b.isub(k_plus, one);
        b.udiv(k_plus, tile_c)
    };
    let (outer, t) = b.begin_loop(t_var, tiles);
    let k0 = b.imul(t, tile_c);
    // Load lhs[row, k0 + tx] and rhs[k0 + ty, col], zero when out of range.
    let ka = b.iadd(k0, tx);
    let ka_ok = b.ult(ka, k);
    let lhs_ok = b.land(row_ok, ka_ok);
    let lhs_index = b.iadd(lhs_row, ka);
    let lhs_index = b.select_u32(lhs_ok, lhs_index, zero);
    let lhs_value = b.load_f32(array, lhs, lhs_index);
    let lhs_value = b.select_f32(lhs_ok, lhs_value, zero_f);
    let kb = b.iadd(k0, ty);
    let kb_ok = b.ult(kb, k);
    let rhs_ok = b.land(kb_ok, col_ok);
    let rhs_index = b.imul(kb, n);
    let rhs_index = b.iadd(rhs_index, rhs_col);
    let rhs_index = b.select_u32(rhs_ok, rhs_index, zero);
    let rhs_value = b.load_f32(array, rhs, rhs_index);
    let rhs_value = b.select_f32(rhs_ok, rhs_value, zero_f);
    let lhs_slot = b.access_chain(workgroup_ptr, lhs_tile, &[local_index]);
    b.store(lhs_slot, lhs_value);
    let rhs_slot = b.access_chain(workgroup_ptr, rhs_tile, &[local_index]);
    b.store(rhs_slot, rhs_value);
    b.workgroup_barrier();
    let remaining = b.isub(k, k0);
    let k_max = b.umin(remaining, tile_c);
    b.store(kk_var, zero);
    let (inner, kk) = b.begin_loop(kk_var, k_max);
    let a_index = b.imul(ty, tile_c);
    let a_index = b.iadd(a_index, kk);
    let a_ptr = b.access_chain(workgroup_ptr, lhs_tile, &[a_index]);
    let a = b.load(f32_ty, a_ptr);
    let b_index = b.imul(kk, tile_c);
    let b_index = b.iadd(b_index, tx);
    let b_ptr = b.access_chain(workgroup_ptr, rhs_tile, &[b_index]);
    let bv = b.load(f32_ty, b_ptr);
    let acc = b.load(f32_ty, acc_var);
    let next = b.fma_free(a, bv, acc);
    b.store(acc_var, next);
    b.end_loop(inner, kk_var, one);
    b.workgroup_barrier();
    b.end_loop(outer, t_var, one);
    let in_range = b.land(row_ok, col_ok);
    b.if_then(in_range, |b| {
        let out_batch = b.imul(z, m);
        let out_batch = b.imul(out_batch, n);
        let out_row = b.imul(row, n);
        let out_index = b.iadd(out_batch, out_row);
        let out_index = b.iadd(out_index, col);
        let acc = b.load(f32_ty, acc_var);
        b.store_f32(array, output, out_index, acc);
    });
    b.end_main();
    b.finish([tile, tile, 1])
}

/// NHWC FP32 MAX_POOL2D: one invocation per output element folds its window with
/// `apply_max_s` from `-inf`; padded positions are skipped, never substituted.
///
/// Specialization order: input, output `(buffer, base)`; `batch`, `height`, `width`, `channels`,
/// `out_height`, `out_width`, `kernel_h`, `kernel_w`, `stride_h`, `stride_w`, `pad_top`,
/// `pad_left`.
fn assemble_max_pool(nan_mode: NanMode, workgroup: u32, buffers: u32) -> Vec<u32> {
    let mut b = Builder::new();
    let array = b.buffer_array(buffers);
    let input = b.spec_operand();
    let output = b.spec_operand();
    let batch = b.spec_u32(1);
    let height = b.spec_u32(1);
    let width = b.spec_u32(1);
    let channels = b.spec_u32(1);
    let out_height = b.spec_u32(1);
    let out_width = b.spec_u32(1);
    let kernel_h = b.spec_u32(1);
    let kernel_w = b.spec_u32(1);
    let stride_h = b.spec_u32(1);
    let stride_w = b.spec_u32(1);
    let pad_top = b.spec_u32(0);
    let pad_left = b.spec_u32(0);

    let (counter, stride) = b.grid_stride(workgroup);
    let u32_ty = b.u32_ty();
    let f32_ty = b.f32_ty();
    let acc_var = b.local(f32_ty);
    let kh_var = b.local(u32_ty);
    let kw_var = b.local(u32_ty);
    let count = b.imul(batch, out_height);
    let count = b.imul(count, out_width);
    let count = b.imul(count, channels);
    let zero = b.c_u32(0);
    let one = b.c_u32(1);
    let (scope, o) = b.begin_loop(counter, count);
    let c = b.umod(o, channels);
    let t = b.udiv(o, channels);
    let ow = b.umod(t, out_width);
    let t = b.udiv(t, out_width);
    let oh = b.umod(t, out_height);
    let nb = b.udiv(t, out_height);
    let neg_inf = b.c_f32(f32::NEG_INFINITY);
    b.store(acc_var, neg_inf);
    b.store(kh_var, zero);
    let row_origin = b.imul(oh, stride_h);
    let col_origin = b.imul(ow, stride_w);
    let (rows, kh) = b.begin_loop(kh_var, kernel_h);
    let padded_row = b.iadd(row_origin, kh);
    let row_in_low = b.uge(padded_row, pad_top);
    let ih = b.isub(padded_row, pad_top);
    let row_in_high = b.ult(ih, height);
    let row_ok = b.land(row_in_low, row_in_high);
    b.store(kw_var, zero);
    let (cols, kw) = b.begin_loop(kw_var, kernel_w);
    let padded_col = b.iadd(col_origin, kw);
    let col_in_low = b.uge(padded_col, pad_left);
    let iw = b.isub(padded_col, pad_left);
    let col_in_high = b.ult(iw, width);
    let col_ok = b.land(col_in_low, col_in_high);
    let ok = b.land(row_ok, col_ok);
    // ((n * H + ih) * W + iw) * C + c, clamped to element zero when the tap is padding.
    let index = b.imul(nb, height);
    let index = b.iadd(index, ih);
    let index = b.imul(index, width);
    let index = b.iadd(index, iw);
    let index = b.imul(index, channels);
    let index = b.iadd(index, c);
    let index = b.select_u32(ok, index, zero);
    let value = b.load_f32(array, input, index);
    let acc = b.load(f32_ty, acc_var);
    let folded = b.apply_max(acc, value, nan_mode);
    let next = b.select_f32(ok, folded, acc);
    b.store(acc_var, next);
    b.end_loop(cols, kw_var, one);
    b.end_loop(rows, kh_var, one);
    let acc = b.load(f32_ty, acc_var);
    b.store_f32(array, output, o, acc);
    b.end_loop(scope, counter, stride);
    b.end_main();
    b.finish([workgroup, 1, 1])
}

/// Strided copy: `out[out_offset + Σ c_d · out_stride_d] = in[in_offset + Σ c_d · in_stride_d]`
/// over the iteration space `dims`; `contiguous` degenerates to `out[i] = in[i]`.
///
/// Specialization order: input, output `(buffer, base)`; `count`; then (strided only)
/// `dims[MAX_RANK]`, `in_strides[MAX_RANK]`, `in_offset`, `out_strides[MAX_RANK]`, `out_offset`.
fn assemble_move(storage: Storage, contiguous: bool, workgroup: u32, buffers: u32) -> Vec<u32> {
    let mut b = Builder::new();
    let array = b.buffer_array(buffers);
    let input = b.spec_operand();
    let output = b.spec_operand();
    let count = b.spec_u32(1);
    let geometry = (!contiguous).then(|| {
        let dims = b.spec_dims();
        let in_strides = b.spec_strides();
        let in_offset = b.spec_u32(0);
        let out_strides = b.spec_strides();
        let out_offset = b.spec_u32(0);
        (dims, in_strides, in_offset, out_strides, out_offset)
    });

    let (counter, stride) = b.grid_stride(workgroup);
    let (scope, i) = b.begin_loop(counter, count);
    let (source, destination) = match &geometry {
        Some((dims, in_strides, in_offset, out_strides, out_offset)) => {
            let indices = b.strided_indices(i, dims, &[*in_strides, *out_strides]);
            let source = b.iadd(indices[0], *in_offset);
            let destination = b.iadd(indices[1], *out_offset);
            (source, destination)
        }
        None => (i, i),
    };
    match storage {
        Storage::Word => {
            let word = b.load_word(array, input, source);
            b.store_word(array, output, destination, word);
        }
        Storage::Byte => {
            let value = b.load_bool(array, input, source);
            b.store_bool(array, output, destination, value);
        }
    }
    b.end_loop(scope, counter, stride);
    b.end_main();
    b.finish([workgroup, 1, 1])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every kernel variant the lowering can select, at one representative tuning.
    fn all_keys() -> Vec<KernelKey> {
        let mut keys = Vec::new();
        let ops = [
            ElementwiseOp::Abs,
            ElementwiseOp::Ceil,
            ElementwiseOp::Cos,
            ElementwiseOp::Erf,
            ElementwiseOp::Exp,
            ElementwiseOp::Floor,
            ElementwiseOp::Log,
            ElementwiseOp::Negate,
            ElementwiseOp::Reciprocal,
            ElementwiseOp::Rsqrt,
            ElementwiseOp::Sin,
            ElementwiseOp::Sigmoid,
            ElementwiseOp::Tanh,
            ElementwiseOp::Clamp(NanMode::Propagate),
            ElementwiseOp::Clamp(NanMode::Ignore),
            ElementwiseOp::Add,
            ElementwiseOp::Sub,
            ElementwiseOp::Mul,
            ElementwiseOp::Pow,
            ElementwiseOp::Maximum(NanMode::Propagate),
            ElementwiseOp::Minimum(NanMode::Ignore),
            ElementwiseOp::Equal,
            ElementwiseOp::Greater,
            ElementwiseOp::GreaterEqual,
            ElementwiseOp::LogicalAnd,
            ElementwiseOp::LogicalOr,
            ElementwiseOp::LogicalXor,
            ElementwiseOp::LogicalNot,
            ElementwiseOp::Select,
            ElementwiseOp::CopyBytes,
        ];
        for op in ops {
            for broadcast in [false, true] {
                keys.push(KernelKey::Elementwise {
                    op,
                    broadcast,
                    workgroup: 64,
                    buffers: 17,
                });
            }
        }
        for op in [
            ReduceOp::Sum,
            ReduceOp::Product,
            ReduceOp::Max(NanMode::Propagate),
            ReduceOp::Min(NanMode::Ignore),
            ReduceOp::ArgMax(NanMode::Propagate),
            ReduceOp::ArgMax(NanMode::Ignore),
        ] {
            keys.push(KernelKey::Reduce {
                op,
                workgroup: 64,
                buffers: 17,
            });
        }
        keys.push(KernelKey::Matmul {
            tile: 16,
            buffers: 17,
        });
        keys.push(KernelKey::Matmul {
            tile: 8,
            buffers: 5,
        });
        for nan_mode in [NanMode::Propagate, NanMode::Ignore] {
            keys.push(KernelKey::MaxPool {
                nan_mode,
                workgroup: 64,
                buffers: 17,
            });
        }
        for storage in [Storage::Word, Storage::Byte] {
            for contiguous in [false, true] {
                keys.push(KernelKey::Move {
                    storage,
                    contiguous,
                    workgroup: 64,
                    buffers: 17,
                });
            }
        }
        keys
    }

    /// Walk a module instruction by instruction: word counts tile the body exactly, every id is
    /// below the bound, the section order is legal, and exactly one function exists.
    fn check_well_formed(key: KernelKey, words: &[u32]) {
        assert_eq!(words[0], SPIRV_MAGIC, "{key:?}");
        assert_eq!(words[1], SPIRV_VERSION_1_3, "{key:?}");
        let bound = words[3];
        let mut cursor = 5;
        let mut opcodes = Vec::new();
        let mut spec_ids = Vec::new();
        let mut defined = std::collections::HashSet::new();
        while cursor < words.len() {
            let word_count = (words[cursor] >> 16) as usize;
            assert!(
                word_count >= 1,
                "{key:?}: zero-length instruction at {cursor}"
            );
            assert!(
                cursor + word_count <= words.len(),
                "{key:?}: instruction overruns"
            );
            let opcode = (words[cursor] & 0xffff) as u16;
            opcodes.push(opcode);
            if opcode == OP_DECORATE && words[cursor + 2] == DECORATION_SPEC_ID {
                spec_ids.push(words[cursor + 3]);
            }
            // Result-id-bearing instructions used here: check the bound and uniqueness.
            let result_id = match opcode {
                OP_TYPE_VOID
                | OP_TYPE_BOOL
                | OP_TYPE_INT
                | OP_TYPE_FLOAT
                | OP_TYPE_VECTOR
                | OP_TYPE_ARRAY
                | OP_TYPE_RUNTIME_ARRAY
                | OP_TYPE_STRUCT
                | OP_TYPE_POINTER
                | OP_TYPE_FUNCTION
                | OP_LABEL
                | OP_EXT_INST_IMPORT => Some(words[cursor + 1]),
                OP_CONSTANT | OP_SPEC_CONSTANT | OP_CONSTANT_FALSE | OP_VARIABLE | OP_LOAD
                | OP_ACCESS_CHAIN | OP_FUNCTION | OP_EXT_INST | OP_SELECT | OP_BITCAST => {
                    Some(words[cursor + 2])
                }
                _ => None,
            };
            if let Some(id) = result_id {
                assert!(id < bound, "{key:?}: id {id} exceeds bound {bound}");
                assert!(defined.insert(id), "{key:?}: id {id} defined twice");
            }
            cursor += word_count;
        }
        assert_eq!(cursor, words.len(), "{key:?}");
        assert_eq!(opcodes[0], OP_CAPABILITY, "{key:?}");
        assert_eq!(opcodes[1], OP_EXT_INST_IMPORT, "{key:?}");
        assert_eq!(opcodes[2], OP_MEMORY_MODEL, "{key:?}");
        assert_eq!(opcodes[3], OP_ENTRY_POINT, "{key:?}");
        assert_eq!(opcodes[4], OP_EXECUTION_MODE, "{key:?}");
        assert_eq!(*opcodes.last().unwrap(), OP_FUNCTION_END, "{key:?}");
        assert_eq!(
            opcodes.iter().filter(|op| **op == OP_FUNCTION).count(),
            1,
            "{key:?}"
        );
        // Annotations precede every type declaration.
        let last_decorate = opcodes
            .iter()
            .rposition(|op| *op == OP_DECORATE || *op == OP_MEMBER_DECORATE)
            .unwrap();
        let first_type = opcodes.iter().position(|op| *op == OP_TYPE_VOID).unwrap();
        assert!(
            last_decorate < first_type,
            "{key:?}: annotation after types"
        );
        // Specialization ids are exactly 0..count.
        spec_ids.sort_unstable();
        let expected: Vec<u32> = (0..key.spec_constant_count()).collect();
        assert_eq!(spec_ids, expected, "{key:?}: specialization ids");
        // Every function-local variable sits at the top of the entry block.
        let function_at = opcodes.iter().position(|op| *op == OP_FUNCTION).unwrap();
        let body = &opcodes[function_at + 2..];
        let first_non_variable = body.iter().position(|op| *op != OP_VARIABLE).unwrap();
        assert!(
            !body[first_non_variable..].contains(&OP_VARIABLE),
            "{key:?}: OpVariable after the entry block prologue"
        );
    }

    #[test]
    fn every_kernel_variant_is_well_formed() {
        for key in all_keys() {
            let words = key.assemble();
            check_well_formed(key, &words);
        }
    }

    #[test]
    fn assembly_is_deterministic() {
        for key in all_keys() {
            assert_eq!(key.assemble(), key.assemble(), "{key:?}");
        }
    }

    #[test]
    fn spec_encoders_match_declared_counts() {
        let operand = Operand { buffer: 1, base: 2 };
        let strides = [[1; MAX_RANK]; 3];
        for key in all_keys() {
            let words = match key {
                KernelKey::Elementwise { op, broadcast, .. } => ElementwiseSpec {
                    count: 8,
                    inputs: &vec![operand; op.inputs().len()],
                    output: operand,
                    dims: [1; MAX_RANK],
                    strides: &strides[..op.inputs().len()],
                    clamp: matches!(op, ElementwiseOp::Clamp(_)).then_some([0, 0x3f80_0000]),
                }
                .words(broadcast),
                KernelKey::Reduce { .. } => reduce_spec(operand, operand, 2, 3, 1),
                KernelKey::Matmul { .. } => matmul_spec(operand, operand, operand, 1, 2, 3, 1),
                KernelKey::MaxPool { .. } => max_pool_spec(
                    operand,
                    operand,
                    PoolGeometry {
                        batch: 1,
                        height: 4,
                        width: 4,
                        channels: 2,
                        out_height: 2,
                        out_width: 2,
                        kernel: [2, 2],
                        stride: [2, 2],
                        pad_top: 0,
                        pad_left: 0,
                    },
                ),
                KernelKey::Move { contiguous, .. } => move_spec(
                    operand,
                    operand,
                    MoveGeometry {
                        count: 6,
                        dims: [1; MAX_RANK],
                        in_strides: [1; MAX_RANK],
                        in_offset: 0,
                        out_strides: [1; MAX_RANK],
                        out_offset: 0,
                    },
                    contiguous,
                ),
            };
            assert_eq!(
                words.len() as u32,
                key.spec_constant_count(),
                "{key:?}: encoder length"
            );
        }
    }

    #[test]
    fn literal_strings_are_nul_terminated_and_word_padded() {
        assert_eq!(literal_string("main"), vec![0x6e69_616d, 0]);
        assert_eq!(literal_string("abc"), vec![0x0063_6261]);
    }

    #[test]
    fn linear_workgroups_cover_and_cap() {
        assert_eq!(linear_workgroups(1, 64, 65_535), 1);
        assert_eq!(linear_workgroups(64, 64, 65_535), 1);
        assert_eq!(linear_workgroups(65, 64, 65_535), 2);
        assert_eq!(linear_workgroups(u32::MAX, 64, 65_535), 65_535);
        assert_eq!(linear_workgroups(0, 64, 65_535), 1);
    }

    #[test]
    fn matmul_workgroups_cover_all_dimensions() {
        assert_eq!(matmul_workgroups(1, 1, 1, 16), [1, 1, 1]);
        assert_eq!(matmul_workgroups(16, 16, 1, 16), [1, 1, 1]);
        assert_eq!(matmul_workgroups(17, 16, 1, 16), [1, 2, 1]);
        assert_eq!(matmul_workgroups(16, 17, 1, 16), [2, 1, 1]);
        assert_eq!(matmul_workgroups(8, 8, 3, 8), [1, 1, 3]);
    }

    #[test]
    fn elementwise_storage_tables_are_consistent() {
        for key in all_keys() {
            if let KernelKey::Elementwise { op, .. } = key {
                assert!(!op.inputs().is_empty());
                assert!(op.inputs().len() <= MAX_ELEMENTWISE_INPUTS);
            }
        }
    }
}
