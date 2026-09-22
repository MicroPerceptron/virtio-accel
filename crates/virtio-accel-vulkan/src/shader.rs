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
//! tensors (`FP32`, `INT32`) are addressed one element per word; half-storage tensors (`FP16`)
//! are packed two elements per word and byte-storage tensors (`BOOL`) four per word, and both
//! are read with word loads and written with `OpAtomicAnd`/`OpAtomicOr`, so a kernel never
//! modifies bytes outside the elements it owns even at a tensor's unaligned tail.
//!
//! ## Binary16 kernels (ADR 0008)
//!
//! FP16 kernels are separate [`KernelKey`] variants (`float: Storage::Half`) built from the same
//! 32-bit-only instruction set as the FP32 kernels — no 16-bit types, no device features, no
//! driver float-controls to trust. Packed binary16 tensors are unpacked with integer ops and
//! widened to binary32 exactly (`Builder::widen_f16`); every float lane evaluates in binary32
//! — which TOSA 1.0 §1.10.3 explicitly permits ("fp16_t operations \[may\] be implemented using
//! the fp32_t datatype") — and results narrow back through crate-owned round-to-nearest-even
//! integer code (`Builder::narrow_f16`) that produces subnormals on every device. For
//! `ADD`/`SUB`/`MUL` that is the correctly rounded binary16 result (their exact results fit the
//! binary32 significand); the comparison and selection lanes are exact; `RECIPROCAL` stays
//! within TOSA's tolerance; the transcendental lanes keep ADR 0007's crate-owned binary32
//! numerics; MATMUL and reduction folds accumulate in binary32, the accumulator width TOSA
//! assigns FP16. `NEGATE`/`ABS` are integer sign masks on the packed lane, and data movement
//! copies the 16-bit lanes as integers, so NaN payloads and subnormals move bit-exactly. Because
//! the conversions are integer code, no compiler can demote the binary32 arithmetic back to
//! f16, and binary16 results are bit-identical on every device.
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
const OP_EXTENSION: u16 = 10;
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
const OP_F_CONVERT: u16 = 115;
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
const OP_BITWISE_XOR: u16 = 198;
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
const OP_GROUP_NON_UNIFORM_F_ADD: u16 = 350;
const OP_TYPE_COOPERATIVE_MATRIX_KHR: u16 = 4456;
const OP_COOPERATIVE_MATRIX_LOAD_KHR: u16 = 4457;
const OP_COOPERATIVE_MATRIX_STORE_KHR: u16 = 4458;
const OP_COOPERATIVE_MATRIX_MUL_ADD_KHR: u16 = 4459;

// Enumerants (section 3).
const CAPABILITY_SHADER: u32 = 1;
const CAPABILITY_FLOAT16: u32 = 9;
const CAPABILITY_GROUP_NON_UNIFORM_ARITHMETIC: u32 = 63;
const CAPABILITY_VULKAN_MEMORY_MODEL: u32 = 5345;
const CAPABILITY_COOPERATIVE_MATRIX_KHR: u32 = 6022;
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
const GLSL_ROUND_EVEN: u32 = 2;
const GLSL_FABS: u32 = 4;
const GLSL_FLOOR: u32 = 8;
const GLSL_CEIL: u32 = 9;
const GLSL_POW: u32 = 26;
const GLSL_EXP: u32 = 27;
const GLSL_LOG: u32 = 28;
const GLSL_INVERSE_SQRT: u32 = 32;
const GLSL_UMIN: u32 = 38;
const GLSL_FMA: u32 = 50;
const GLSL_FIND_U_MSB: u32 = 75;

/// A SPIR-V result id.
pub type Id = u32;

/// Which of the two TOSA FP8 encodings a byte-storage float tensor carries. They differ in
/// exponent width, bias, and specials: E4M3 has no infinity (`0x7f`/`0xff` are its only NaNs,
/// finite max 448) while E5M2 is IEEE-shaped (infinity at `0x7c`, finite max 57344).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Fp8Format {
    E4M3,
    E5M2,
}

/// How a tensor's scalars are laid out in the storage words a kernel addresses.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Storage {
    /// One 32-bit word per element (`FP32`, `INT32`).
    Word,
    /// One byte per element, four to a word (`BOOL`).
    Byte,
    /// One byte per element, four to a word, holding an FP8 encoding. Geometrically identical
    /// to [`Self::Byte`]; it is a separate variant because the byte is a raw float pattern, not
    /// a canonical `0`/`1`, so it is loaded through an exact widening and moved as raw bits
    /// (ADR 0009).
    Quarter(Fp8Format),
    /// Two bytes per element, two to a word (`FP16`); the 16-bit lanes are unpacked on load and
    /// repacked with `OpAtomicAnd`/`OpAtomicOr` on store, so a kernel never modifies the
    /// neighbouring element of its word.
    Half,
}

impl Storage {
    /// Elements packed into one 32-bit storage word.
    pub const fn lanes(self) -> u32 {
        match self {
            Self::Word => 1,
            Self::Half => 2,
            Self::Byte | Self::Quarter(_) => 4,
        }
    }
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
    /// Storage of each tensor input, in operand order. `Word` marks a floating-point lane: the
    /// kernel resolves it to its own float storage (`Word` for FP32, `Half` for FP16); `Byte`
    /// lanes are always `BOOL`.
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

    /// Storage of the output tensor, with the same float-lane convention as [`Self::inputs`].
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
    /// F32 activations times row-major packed E2M1 weights and E4M3 block scales.
    Nvfp4Matmul {
        buffers: u32,
        cooperative: bool,
        subgroup: bool,
    },
    /// Elementwise lanes over `count` output elements; `broadcast` selects the strided
    /// multi-index addressing, otherwise every operand shares the output's linear index. `float`
    /// is the storage of the operator's floating-point tensors (`Word` for FP32, `Half` for
    /// FP16); `BOOL` lanes are byte storage in either variant. Binary16 lanes evaluate in
    /// binary32 and narrow once, except the integer `NEGATE`/`ABS` sign lanes (ADR 0008).
    Elementwise {
        op: ElementwiseOp,
        float: Storage,
        broadcast: bool,
        workgroup: u32,
        buffers: u32,
    },
    /// Axis reduction: one invocation per output element, sequential ascending-axis fold. The
    /// input is read at `float` storage; sums and products fold in binary32 (the TOSA
    /// accumulator width), and the output is stored back at `float` storage.
    Reduce {
        op: ReduceOp,
        float: Storage,
        workgroup: u32,
        buffers: u32,
    },
    /// Batched, register-tiled matrix multiplication: a `tile × tile` workgroup computes a
    /// [`matmul_block`]-sided output square from workgroup-shared binary32 slabs. Both
    /// operands are read at `input` storage and accumulate in binary32 — the accumulator width
    /// TOSA assigns FP16 and FP8 MATMUL — and the result is stored at `output` storage. The two
    /// differ only for the FP8 tier, where TOSA defines MATMUL as `(FP8, FP8) -> FP16`.
    Matmul {
        input: Storage,
        output: Storage,
        tile: u32,
        buffers: u32,
    },
    /// Split-`k` streaming MATMUL for `m ≤` [`STREAM_ROWS`] rows:
    /// a 1-D workgroup of [`STREAM_WORKGROUP`] invocations over [`STREAM_COLUMNS`] columns of
    /// `rhs` (read at `rhs` storage, a word per invocation), the lhs read as binary32 words —
    /// lowering widens a narrower lhs beforehand — and one fixed-order reduction of the `k`
    /// slices at the end. Same accumulator width as [`Self::Matmul`].
    MatmulStream {
        rhs: Storage,
        output: Storage,
        buffers: u32,
    },
    /// NHWC max pooling with padding excluded from the window, at `float` storage.
    MaxPool {
        nan_mode: NanMode,
        float: Storage,
        workgroup: u32,
        buffers: u32,
    },
    /// Elementwise float conversion: read at `input` storage, write at `output`. One dispatch
    /// per `CAST` between float dtypes, including both FP8 directions (ADR 0009).
    Cast {
        input: Storage,
        output: Storage,
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
            Self::Nvfp4Matmul {
                buffers,
                cooperative: false,
                subgroup: false,
            } => assemble_nvfp4_matmul(buffers),
            Self::Nvfp4Matmul {
                buffers,
                cooperative: true,
                ..
            } => assemble_nvfp4_matmul_cooperative(buffers),
            Self::Nvfp4Matmul {
                buffers,
                cooperative: false,
                subgroup: true,
            } => assemble_nvfp4_matmul_subgroup(buffers),
            Self::Elementwise {
                op,
                float,
                broadcast,
                workgroup,
                buffers,
            } => assemble_elementwise(op, float, broadcast, workgroup, buffers),
            Self::Reduce {
                op,
                float,
                workgroup,
                buffers,
            } => assemble_reduce(op, float, workgroup, buffers),
            Self::Matmul {
                input,
                output,
                tile,
                buffers,
            } => assemble_matmul(input, output, MatmulGeometry::wide(tile), buffers),
            Self::MatmulStream {
                rhs,
                output,
                buffers,
            } => assemble_matmul_stream(rhs, output, buffers),
            Self::MaxPool {
                nan_mode,
                float,
                workgroup,
                buffers,
            } => assemble_max_pool(nan_mode, float, workgroup, buffers),
            Self::Move {
                storage,
                contiguous,
                workgroup,
                buffers,
            } => assemble_move(storage, contiguous, workgroup, buffers),
            Self::Cast {
                input,
                output,
                workgroup,
                buffers,
            } => assemble_cast(input, output, workgroup, buffers),
        }
    }

    /// Every kernel variant this backend can assemble, at one representative tuning. The
    /// SPIR-V validation sweep and the specialization-count test both walk this list, so a new
    /// variant is validated the moment it is added here.
    pub fn every_variant() -> Vec<KernelKey> {
        let mut keys = Vec::new();
        keys.push(KernelKey::Nvfp4Matmul {
            buffers: 17,
            cooperative: false,
            subgroup: false,
        });
        keys.push(KernelKey::Nvfp4Matmul {
            buffers: 17,
            cooperative: true,
            subgroup: false,
        });
        keys.push(KernelKey::Nvfp4Matmul {
            buffers: 17,
            cooperative: false,
            subgroup: true,
        });
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
            for float in [Storage::Word, Storage::Half] {
                for broadcast in [false, true] {
                    keys.push(KernelKey::Elementwise {
                        op,
                        float,
                        broadcast,
                        workgroup: 64,
                        buffers: 17,
                    });
                }
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
            for float in [Storage::Word, Storage::Half] {
                keys.push(KernelKey::Reduce {
                    op,
                    float,
                    workgroup: 64,
                    buffers: 17,
                });
            }
        }
        for float in [Storage::Word, Storage::Half] {
            keys.push(KernelKey::Matmul {
                input: float,
                output: float,
                tile: 16,
                buffers: 17,
            });
            keys.push(KernelKey::Matmul {
                input: float,
                output: float,
                tile: 8,
                buffers: 5,
            });
        }
        for float in [Storage::Word, Storage::Half] {
            keys.push(KernelKey::MatmulStream {
                rhs: float,
                output: float,
                buffers: 17,
            });
        }
        // The lhs widening lowering inserts ahead of a skinny FP16 MATMUL.
        keys.push(KernelKey::Cast {
            input: Storage::Half,
            output: Storage::Word,
            workgroup: 64,
            buffers: 17,
        });
        keys.push(KernelKey::Cast {
            input: Storage::Word,
            output: Storage::Half,
            workgroup: 64,
            buffers: 17,
        });
        // The FP8 tier: TOSA's `(FP8, FP8) -> FP16` MATMUL, and exact FP8 data movement.
        for format in [Fp8Format::E4M3, Fp8Format::E5M2] {
            keys.push(KernelKey::MatmulStream {
                rhs: Storage::Quarter(format),
                output: Storage::Half,
                buffers: 17,
            });
            keys.push(KernelKey::Matmul {
                input: Storage::Quarter(format),
                output: Storage::Half,
                tile: 16,
                buffers: 17,
            });
            keys.push(KernelKey::Matmul {
                input: Storage::Quarter(format),
                output: Storage::Half,
                tile: 8,
                buffers: 5,
            });
        }
        for nan_mode in [NanMode::Propagate, NanMode::Ignore] {
            for float in [Storage::Word, Storage::Half] {
                keys.push(KernelKey::MaxPool {
                    nan_mode,
                    float,
                    workgroup: 64,
                    buffers: 17,
                });
            }
        }
        // The FP8 tier's pooling and ARGMAX: pooling selects an existing encoding, ARGMAX
        // compares widened values and emits an INT32 index.
        for format in [Fp8Format::E4M3, Fp8Format::E5M2] {
            for nan_mode in [NanMode::Propagate, NanMode::Ignore] {
                keys.push(KernelKey::MaxPool {
                    nan_mode,
                    float: Storage::Quarter(format),
                    workgroup: 64,
                    buffers: 17,
                });
                keys.push(KernelKey::Reduce {
                    op: ReduceOp::ArgMax(nan_mode),
                    float: Storage::Quarter(format),
                    workgroup: 64,
                    buffers: 17,
                });
            }
        }
        // Every CAST pair the FP8 tier lowers, both directions.
        for format in [Fp8Format::E4M3, Fp8Format::E5M2] {
            for wide in [Storage::Word, Storage::Half] {
                keys.push(KernelKey::Cast {
                    input: Storage::Quarter(format),
                    output: wide,
                    workgroup: 64,
                    buffers: 17,
                });
                keys.push(KernelKey::Cast {
                    input: wide,
                    output: Storage::Quarter(format),
                    workgroup: 64,
                    buffers: 17,
                });
            }
        }
        for storage in [
            Storage::Word,
            Storage::Byte,
            Storage::Half,
            Storage::Quarter(Fp8Format::E4M3),
            Storage::Quarter(Fp8Format::E5M2),
        ] {
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
            Self::Matmul { .. } | Self::MatmulStream { .. } => 3 * 2 + 4,
            // Activation, six packed bindings, six scale bindings, tensor scale and output,
            // followed by m/n/k/epilogue/weight mode.
            Self::Nvfp4Matmul { .. } => 15 * 2 + 5,
            Self::MaxPool { .. } => 2 + 2 + 12,
            Self::Move { contiguous, .. } => {
                if contiguous {
                    2 + 2 + 1
                } else {
                    2 + 2 + 1 + MAX_RANK as u32 * 3 + 2
                }
            }
            Self::Cast { .. } => 2 + 2 + 1,
        }
    }

    /// `OpExecutionMode LocalSize` of the module.
    pub const fn local_size(self) -> [u32; 3] {
        match self {
            Self::Elementwise { workgroup, .. }
            | Self::Reduce { workgroup, .. }
            | Self::MaxPool { workgroup, .. }
            | Self::Move { workgroup, .. }
            | Self::Cast { workgroup, .. } => [workgroup, 1, 1],
            Self::Matmul { tile, .. } => MatmulGeometry::wide(tile).local_size(),
            Self::MatmulStream { .. } => [STREAM_WORKGROUP, 1, 1],
            Self::Nvfp4Matmul {
                cooperative: false,
                subgroup: false,
                ..
            } => [STREAM_WORKGROUP, 1, 1],
            Self::Nvfp4Matmul {
                cooperative: true, ..
            } => [32, 1, 1],
            Self::Nvfp4Matmul {
                cooperative: false,
                subgroup: true,
                ..
            } => [32, 1, 1],
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

/// Specialization payload for native row-major NVFP4: activation, packed
/// weights, block scales, tensor scale, output, then `m`, `n`, `k`, activation.
pub fn nvfp4_matmul_spec(
    activation: Operand,
    packed: &[Operand],
    block_scales: &[Operand],
    tensor_scale: Operand,
    output: Operand,
    m: u32,
    n: u32,
    k: u32,
    epilogue: u32,
    weight_mode: u32,
) -> Vec<u32> {
    assert!(!packed.is_empty() && packed.len() <= 6 && packed.len() == block_scales.len());
    let mut words = Vec::with_capacity(35);
    activation.push(&mut words);
    for index in 0..6 {
        packed[index.min(packed.len() - 1)].push(&mut words);
    }
    for index in 0..6 {
        block_scales[index.min(block_scales.len() - 1)].push(&mut words);
    }
    tensor_scale.push(&mut words);
    output.push(&mut words);
    words.extend_from_slice(&[m, n, k, epilogue, weight_mode]);
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

/// Outputs per invocation per side of the register-tiled MATMUL: each invocation of a
/// `tile × tile` workgroup accumulates a `MATMUL_MICRO × MATMUL_MICRO` block, so the workgroup
/// covers a [`matmul_block`]-sided square of the result.
pub const MATMUL_MICRO: u32 = 4;

/// Rows the streaming MATMUL kernel carries per invocation, and the row count at or below which
/// lowering selects it over the register-tiled kernel.
pub const STREAM_ROWS: u32 = 8;

/// Output columns one streaming MATMUL workgroup covers.
pub const STREAM_COLUMNS: u32 = 16;

/// Invocations of a streaming MATMUL workgroup: `STREAM_COLUMNS / lanes` weight words times
/// `STREAM_WORKGROUP / that` slices of `k`.
pub const STREAM_WORKGROUP: u32 = 64;

/// Bytes of workgroup-shared memory the streaming kernel declares for its final reduction:
/// every invocation's `STREAM_ROWS × lanes` partial sums, at the widest lane count.
pub const fn stream_matmul_shared_bytes() -> u32 {
    STREAM_WORKGROUP * STREAM_ROWS * 4 * 4
}

/// Workgroup counts of a streaming MATMUL over `n` columns and `batch` batches.
pub const fn stream_matmul_workgroups(n: u32, batch: u32) -> [u32; 3] {
    [n.div_ceil(STREAM_COLUMNS), 1, batch]
}

/// Side of the output square one MATMUL workgroup of `tile × tile` invocations computes.
pub const fn matmul_block(tile: u32) -> u32 {
    tile * MATMUL_MICRO
}

/// Bytes of workgroup-shared memory the MATMUL kernels at `tile` declare, whichever kernel
/// needs more.
pub const fn matmul_shared_bytes(tile: u32) -> u32 {
    let wide = MatmulGeometry::wide(tile).shared_bytes();
    let stream = stream_matmul_shared_bytes();
    if wide > stream { wide } else { stream }
}

/// Workgroup counts of a tiled MATMUL over `m` rows, `n` columns, and `batch` batches.
pub const fn matmul_workgroups(m: u32, n: u32, batch: u32, tile: u32) -> [u32; 3] {
    let block = matmul_block(tile);
    [n.div_ceil(block), m.div_ceil(block), batch]
}

/// Shape of one MATMUL workgroup: `tile_x × tile_y` invocations, each accumulating a
/// `micro_m × micro_n` register block, over shared-memory slabs `depth` deep in `k`. Every
/// dimension is a power of two and the two slabs stage evenly over the invocations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MatmulGeometry {
    pub tile_x: u32,
    pub tile_y: u32,
    pub micro_m: u32,
    pub micro_n: u32,
    pub depth: u32,
}

impl MatmulGeometry {
    /// The square register-tiled geometry: `tile × tile` invocations, [`MATMUL_MICRO`]² each,
    /// `tile` deep — a 64 × 64 block at the preferred tile.
    pub const fn wide(tile: u32) -> Self {
        Self {
            tile_x: tile,
            tile_y: tile,
            micro_m: MATMUL_MICRO,
            micro_n: MATMUL_MICRO,
            depth: tile,
        }
    }

    pub const fn invocations(self) -> u32 {
        self.tile_x * self.tile_y
    }

    pub const fn block_m(self) -> u32 {
        self.tile_y * self.micro_m
    }

    pub const fn block_n(self) -> u32 {
        self.tile_x * self.micro_n
    }

    pub const fn local_size(self) -> [u32; 3] {
        [self.tile_x, self.tile_y, 1]
    }

    /// Two binary32 slabs: lhs `block_m × depth` (stored transposed) and rhs `depth × block_n`.
    pub const fn shared_bytes(self) -> u32 {
        (self.block_m() + self.block_n()) * self.depth * 4
    }
}

// ---------------------------------------------------------------------------------------------
// Binary16 host conversions
// ---------------------------------------------------------------------------------------------

/// The exact binary32 value of a binary16 bit pattern (host side of the kernels' unpack: every
/// binary16 value, subnormals included, is exactly representable in binary32; NaN payloads are
/// preserved).
/// Narrow binary32 to one FP8 encoding with round-to-nearest, ties-to-even.
///
/// **Overflow policy.** TOSA 1.0 through 1.2 leave float-to-FP8 overflow undefined, so this is
/// the crate's policy rather than the spec's. A magnitude too large for E4M3 — which has no
/// infinity — becomes NaN, not the finite maximum. The reason is that the alternatives are not
/// symmetric: a consumer who wants saturation can `CLAMP` in a wider dtype before the `CAST`
/// and get it exactly, while a consumer handed a saturated 448 cannot tell it from a value that
/// was always 448. NaN preserves the choice; saturation destroys it. It also matches what this
/// crate already does one format up, where `narrow_f16` signals unrepresentability as infinity
/// rather than clamping to 65504.
///
/// E5M2 needs no policy: it is IEEE-shaped, so overflow becomes infinity like any binary float.
/// NaN in becomes a canonical quiet NaN out, sign preserved, for both encodings.
pub fn f32_to_fp8_bits(format: Fp8Format, value: f32) -> u8 {
    let (mantissa_bits, bias, nan_out, max_finite): (u32, u32, u8, u32) = match format {
        // E4M3's `0x7f` is its only NaN, so the finite encodings stop at `0x7e` (448).
        Fp8Format::E4M3 => (3, 7, 0x7f, 0x7e),
        // E5M2's `0x7c` is infinity; finite encodings stop at `0x7b` (57344).
        Fp8Format::E5M2 => (2, 15, 0x7e, 0x7b),
    };
    let bits = value.to_bits();
    let sign = ((bits >> 24) as u8) & 0x80;
    let magnitude = bits & 0x7fff_ffff;
    if magnitude > 0x7f80_0000 {
        return sign | nan_out;
    }
    let shift = 23 - mantissa_bits;
    let normal_floor = (128 - bias) << 23;
    let body = if magnitude >= normal_floor {
        // Rebias into the target's exponent range, then round the significand with the
        // add-half-ulp-plus-guard trick; a significand carry increments the exponent by itself.
        let adjusted = magnitude - ((127 - bias) << 23);
        let guard = (adjusted >> shift) & 1;
        let half_ulp = (1 << (shift - 1)) - 1;
        (adjusted + half_ulp + guard) >> shift
    } else {
        // Subnormal or zero: scale into integer range — exact for every representable
        // subnormal — and round to the nearest integer, ties to even. A rounding carry reaches
        // the smallest normal encoding on its own.
        let scale = f32::from_bits((127 + bias - 1 + mantissa_bits) << 23);
        (f32::from_bits(magnitude) * scale).round_ties_even() as u32
    };
    // Rounding has already happened, so landing above the last finite encoding *is* the
    // overflow test; infinity in reaches it too.
    if body > max_finite {
        let overflow = match format {
            Fp8Format::E4M3 => nan_out,
            Fp8Format::E5M2 => 0x7c,
        };
        return sign | overflow;
    }
    sign | body as u8
}

pub fn f16_to_f32(bits: u16) -> f32 {
    let bits = u32::from(bits);
    let sign = (bits & 0x8000) << 16;
    let exponent = (bits >> 10) & 0x1f;
    let mantissa = bits & 0x3ff;
    let converted = if exponent == 0 {
        // Subnormal or zero: `mantissa · 2^-24`, exact for every 10-bit mantissa.
        (mantissa as f32) * (1.0 / 16_777_216.0)
    } else if exponent == 31 {
        f32::from_bits(0x7f80_0000 | (mantissa << 13))
    } else {
        f32::from_bits(((exponent + 112) << 23) | (mantissa << 13))
    };
    f32::from_bits(converted.to_bits() | sign)
}

/// The binary16 bit pattern nearest to `value`, round-to-nearest-even. NaN is canonicalized to
/// the quiet `0x7e00` payload with its sign;
/// magnitudes at or above 65520 round to infinity, and subnormals are produced, never flushed).
pub fn f32_to_f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let magnitude = bits & 0x7fff_ffff;
    if magnitude > 0x7f80_0000 {
        return sign | 0x7e00;
    }
    if magnitude >= 0x477f_f000 {
        // At or above 65520 — the midpoint between 65504 and the overflow binade — round to
        // infinity (65504 itself is `0x477f_e000`, finite).
        return sign | 0x7c00;
    }
    if magnitude >= 0x3880_0000 {
        // Normal binary16: rebias the exponent, then round the significand with the
        // add-half-ulp-plus-guard trick; a mantissa carry increments the exponent on its own.
        let adjusted = magnitude - 0x3800_0000;
        let rounded = adjusted + 0x0000_0fff + ((adjusted >> 13) & 1);
        return sign | (rounded >> 13) as u16;
    }
    // Subnormal binary16 or zero: scale into integer range (exact while the value is at or
    // above 2^-25; anything smaller rounds to zero either way) and round to the nearest integer.
    let scaled = f32::from_bits(magnitude) * 16_777_216.0;
    let truncated = scaled as u16;
    let fraction = scaled - f32::from(truncated);
    let rounded = if fraction > 0.5 || (fraction == 0.5 && truncated & 1 != 0) {
        truncated + 1
    } else {
        truncated
    };
    sign | rounded
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
    F16,
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
    extensions: Vec<u32>,
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
            extensions: Vec::new(),
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
            &self.extensions,
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
            TypeKey::F16 => instruction(&mut self.declarations, OP_TYPE_FLOAT, &[id, 16]),
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
    fn f16_ty(&mut self) -> Id {
        self.ty(TypeKey::F16)
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

    /// A workgroup-shared binary16 array used as a cooperative-matrix staging tile.
    fn shared_f16_array(&mut self, length: u32) -> Id {
        let f16_ty = self.f16_ty();
        let length = self.c_u32(length);
        let array = self.ty(TypeKey::Array(f16_ty, length));
        let pointer = self.pointer(STORAGE_CLASS_WORKGROUP, array);
        let variable = self.id();
        instruction(
            &mut self.declarations,
            OP_VARIABLE,
            &[pointer, variable, STORAGE_CLASS_WORKGROUP],
        );
        variable
    }

    fn enable_cooperative_matrix(&mut self) {
        instruction(&mut self.capabilities, OP_CAPABILITY, &[CAPABILITY_FLOAT16]);
        instruction(
            &mut self.capabilities,
            OP_CAPABILITY,
            &[CAPABILITY_COOPERATIVE_MATRIX_KHR],
        );
        instruction(
            &mut self.capabilities,
            OP_CAPABILITY,
            &[CAPABILITY_VULKAN_MEMORY_MODEL],
        );
        let memory_extension = literal_string("SPV_KHR_vulkan_memory_model");
        instruction(&mut self.extensions, OP_EXTENSION, &memory_extension);
        let extension = literal_string("SPV_KHR_cooperative_matrix");
        instruction(&mut self.extensions, OP_EXTENSION, &extension);
        self.memory_model.clear();
        instruction(
            &mut self.memory_model,
            OP_MEMORY_MODEL,
            &[ADDRESSING_MODEL_LOGICAL, 3], // VulkanKHR
        );
    }

    fn enable_subgroup_arithmetic(&mut self) {
        instruction(
            &mut self.capabilities,
            OP_CAPABILITY,
            &[CAPABILITY_GROUP_NON_UNIFORM_ARITHMETIC],
        );
    }

    fn cooperative_matrix_ty(&mut self, component: Id, rows: u32, columns: u32, usage: u32) -> Id {
        let ty = self.id();
        let scope = self.c_u32(3); // Scope Subgroup
        let rows = self.c_u32(rows);
        let columns = self.c_u32(columns);
        let usage = self.c_u32(usage);
        instruction(
            &mut self.declarations,
            OP_TYPE_COOPERATIVE_MATRIX_KHR,
            &[ty, component, scope, rows, columns, usage],
        );
        ty
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
        self.float_typed(f32_ty, opcode, operands)
    }

    /// A `NoContraction`-decorated float result of `ty`, so no driver may fuse a multiply and an
    /// add: the same TOSA graph yields the same bits on every conformant device.
    fn float_typed(&mut self, ty: Id, opcode: u16, operands: &[u32]) -> Id {
        let id = self.value(opcode, ty, operands);
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
    fn bxor(&mut self, a: Id, b: Id) -> Id {
        let ty = self.u32_ty();
        self.value(OP_BITWISE_XOR, ty, &[a, b])
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
    fn f32_to_f16(&mut self, float: Id) -> Id {
        let ty = self.f16_ty();
        self.value(OP_F_CONVERT, ty, &[float])
    }

    fn cooperative_load(&mut self, ty: Id, pointer: Id, stride: Id) -> Id {
        let row_major = self.c_u32(0);
        self.value(
            OP_COOPERATIVE_MATRIX_LOAD_KHR,
            ty,
            &[pointer, row_major, stride],
        )
    }

    fn cooperative_store(&mut self, pointer: Id, value: Id, stride: Id) {
        let row_major = self.c_u32(0);
        self.emit(
            OP_COOPERATIVE_MATRIX_STORE_KHR,
            &[pointer, value, row_major, stride],
        );
    }

    fn cooperative_mul_add(&mut self, ty: Id, a: Id, b: Id, c: Id) -> Id {
        self.value(OP_COOPERATIVE_MATRIX_MUL_ADD_KHR, ty, &[a, b, c])
    }

    fn subgroup_sum_f32(&mut self, value: Id) -> Id {
        let ty = self.f32_ty();
        let subgroup_scope = self.c_u32(3);
        self.value(
            OP_GROUP_NON_UNIFORM_F_ADD,
            ty,
            &[subgroup_scope, 0, value], // GroupOperation Reduce
        )
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
    /// `a · b + c` as two separately rounded operations, never contracted: the polynomial
    /// lanes were tuned against binary64 references with exactly this rounding.
    fn fma_free(&mut self, a: Id, b: Id, c: Id) -> Id {
        let product = self.fmul(a, b);
        self.fadd(product, c)
    }
    /// `a · b + c` as one fused multiply-add: the GLSL `Fma` instruction decorated
    /// `NoContraction`, which Vulkan defines as a single correctly rounded operation rather than
    /// a multiply the driver may or may not fuse with the add (ADR 0011).
    fn fma(&mut self, a: Id, b: Id, c: Id) -> Id {
        let ty = self.f32_ty();
        let glsl = self.glsl;
        let id = self.value(OP_EXT_INST, ty, &[glsl, GLSL_FMA, a, b, c]);
        instruction(
            &mut self.annotations,
            OP_DECORATE,
            &[id, DECORATION_NO_CONTRACTION],
        );
        id
    }

    // -- crate-owned binary16 conversions (ADR 0008) --------------------------------------------

    /// Widen packed FP8 bits (a `u32` in `0..=0xff`) to binary32, exactly. The kernel twin of
    /// [`virtio_accel_tosa::fp8e4m3_to_f32`] / [`virtio_accel_tosa::fp8e5m2_to_f32`]; every FP8
    /// value is representable in binary32, so all 256 patterns of each format widen exactly on
    /// every device, and no `OpFConvert` appears that a driver could demote.
    ///
    /// The exponent-offset construction (shared with [`Self::widen_f16`]): place the magnitude
    /// bits at the top of a binary32 significand under a fixed exponent `K` chosen so that a
    /// *normal* encoding reads off directly as `1.m · 2^(e - bias)`. A subnormal encoding then
    /// reads as `1.m · 2^(1 - bias)` and the value wanted is `0.m · 2^(1 - bias)`, which is
    /// `(x - 2^(1 - bias)) · 2` — two exact binary32 operations, since `x` and the constant
    /// share a binade. Nothing here produces a binary32 denormal (the smallest result is the
    /// format's smallest subnormal, `2^-9` or `2^-16`), so a device that flushes denormals
    /// computes the same bits. Specials are one compare and one select. About half the
    /// instructions of the integer-only expansion it replaced, which matters because the
    /// MATMUL kernels widen every staged element.
    fn widen_fp8(&mut self, format: Fp8Format, bits: Id) -> Id {
        // Mantissa width, the fixed exponent `K = 127 - bias`, and the top exponent field.
        let (mantissa_bits, k, top_exponent) = match format {
            Fp8Format::E4M3 => (3_u32, 120_u32, 15_u32),
            Fp8Format::E5M2 => (2, 112, 31),
        };
        let magnitude_mask = self.c_u32(0x7f);
        let shift = self.c_u32(23 - mantissa_bits);
        let k_bits = self.c_u32(k << 23);
        let mantissa_shift = self.c_u32(mantissa_bits);
        let zero = self.c_u32(0);
        let implicit_one = self.c_f32(f32::from_bits(k << 23));
        let sign_mask = self.c_u32(0x80);
        let twenty_four = self.c_u32(24);
        let magnitude = self.band(bits, magnitude_mask);
        let placed = self.shl(magnitude, shift);
        // An add, not an or: the encoding's exponent field and `K` overlap in the binary32
        // exponent bits, and the construction is `e + K`.
        let x_bits = self.iadd(placed, k_bits);
        let x = self.bitcast_f32(x_bits);
        let exponent = self.shr(magnitude, mantissa_shift);
        let is_subnormal = self.ieq(exponent, zero);
        let difference = self.fsub(x, implicit_one);
        let subnormal = self.fadd(difference, difference);
        let value = self.select_f32(is_subnormal, subnormal, x);
        let value = self.bitcast_u32(value);
        // Specials differ: E4M3's top exponent stays finite except for the all-ones fraction,
        // which is its only NaN; E5M2's top exponent is infinity or NaN as usual, payload kept.
        let value = match format {
            Fp8Format::E4M3 => {
                let nan_pattern = self.c_u32(0x7f);
                let quiet_nan = self.c_u32(0x7fc0_0000);
                let is_nan = self.ieq(magnitude, nan_pattern);
                self.select_u32(is_nan, quiet_nan, value)
            }
            Fp8Format::E5M2 => {
                let top = self.c_u32(top_exponent);
                let inf_bits = self.c_u32(0x7f80_0000);
                let fraction_mask = self.c_u32((1 << mantissa_bits) - 1);
                let is_infnan = self.ieq(exponent, top);
                let fraction = self.band(magnitude, fraction_mask);
                let payload = self.shl(fraction, shift);
                let infnan = self.bor(inf_bits, payload);
                self.select_u32(is_infnan, infnan, value)
            }
        };
        let sign = self.band(bits, sign_mask);
        let sign = self.shl(sign, twenty_four);
        let value = self.bor(value, sign);
        self.bitcast_f32(value)
    }

    /// Widen packed binary16 bits (a `u32` in `0..=0xffff`) to binary32, exactly. This is the
    /// kernel twin of the host [`f16_to_f32`], by the exponent-offset construction described at
    /// [`Self::widen_fp8`] (`K = 112`, subnormals as `(x - 2^-15) · 2`): every value —
    /// subnormals included — is exact on every device, no binary32 denormal is ever formed, and
    /// there is no `OpFConvert` pattern a driver can demote back to f16 (ADR 0008).
    fn widen_f16(&mut self, bits: Id) -> Id {
        let magnitude_mask = self.c_u32(0x7fff);
        let thirteen = self.c_u32(13);
        let ten = self.c_u32(10);
        let sixteen = self.c_u32(16);
        let k_bits = self.c_u32(112 << 23);
        let implicit_one = self.c_f32(f32::from_bits(112 << 23));
        let zero = self.c_u32(0);
        let thirty_one = self.c_u32(31);
        let inf_bits = self.c_u32(0x7f80_0000);
        let mantissa_mask = self.c_u32(0x3ff);
        let sign_mask = self.c_u32(0x8000);
        let magnitude = self.band(bits, magnitude_mask);
        let placed = self.shl(magnitude, thirteen);
        // An add, not an or: the encoding's exponent field and `K` overlap in the binary32
        // exponent bits, and the construction is `e + K`.
        let x_bits = self.iadd(placed, k_bits);
        let x = self.bitcast_f32(x_bits);
        let exponent = self.shr(magnitude, ten);
        let is_subnormal = self.ieq(exponent, zero);
        let difference = self.fsub(x, implicit_one);
        let subnormal = self.fadd(difference, difference);
        let value = self.select_f32(is_subnormal, subnormal, x);
        let value = self.bitcast_u32(value);
        // Infinity or NaN: exponent all ones, payload preserved.
        let is_infnan = self.ieq(exponent, thirty_one);
        let mantissa = self.band(magnitude, mantissa_mask);
        let payload = self.shl(mantissa, thirteen);
        let infnan = self.bor(inf_bits, payload);
        let value = self.select_u32(is_infnan, infnan, value);
        let sign = self.band(bits, sign_mask);
        let sign = self.shl(sign, sixteen);
        let value = self.bor(value, sign);
        self.bitcast_f32(value)
    }

    /// Narrow a binary32 value to packed binary16 bits (a `u32` in `0..=0xffff`),
    /// round-to-nearest-even, NaN canonicalized to the quiet `0x7e00` payload with its sign.
    /// This is the kernel twin of the host [`f32_to_f16_bits`]: integer and binary32 operations
    /// only, so subnormals are produced — never flushed — on every device (ADR 0008).
    /// Narrow binary32 to FP8, the kernel twin of [`f32_to_fp8_bits`] and bound by its tests.
    /// Integer rounding throughout, with one `RoundEven` for the subnormal range, so the result
    /// is identical on every device. The overflow policy is that function's: NaN for E4M3,
    /// infinity for E5M2 (ADR 0009).
    fn narrow_fp8(&mut self, format: Fp8Format, value: Id) -> Id {
        let (mantissa_bits, bias, nan_out, max_finite, overflow_out) = match format {
            Fp8Format::E4M3 => (3_u32, 7_u32, 0x7f_u32, 0x7e_u32, 0x7f_u32),
            Fp8Format::E5M2 => (2, 15, 0x7e, 0x7b, 0x7c),
        };
        let shift = 23 - mantissa_bits;
        let twenty_four = self.c_u32(24);
        let one = self.c_u32(1);
        let sign_mask = self.c_u32(0x80);
        let magnitude_mask = self.c_u32(0x7fff_ffff);
        let inf_bits = self.c_u32(0x7f80_0000);
        let normal_floor = self.c_u32((128 - bias) << 23);
        let rebias = self.c_u32((127 - bias) << 23);
        let shift_c = self.c_u32(shift);
        let half_ulp = self.c_u32((1 << (shift - 1)) - 1);
        let scale = self.c_f32(f32::from_bits((127 + bias - 1 + mantissa_bits) << 23));
        let max_finite_c = self.c_u32(max_finite);
        let nan_c = self.c_u32(nan_out);
        let overflow_c = self.c_u32(overflow_out);
        let bits = self.bitcast_u32(value);
        let sign = self.shr(bits, twenty_four);
        let sign = self.band(sign, sign_mask);
        let magnitude = self.band(bits, magnitude_mask);
        let is_nan = self.ult(inf_bits, magnitude);
        let is_normal = self.uge(magnitude, normal_floor);
        // Normal: rebias, then round the significand with the add-half-ulp-plus-guard trick; a
        // carry increments the exponent on its own. The subnormal branch discards the wrap.
        let adjusted = self.isub(magnitude, rebias);
        let guard = self.shr(adjusted, shift_c);
        let guard = self.band(guard, one);
        let rounding = self.iadd(half_ulp, guard);
        let rounded = self.iadd(adjusted, rounding);
        let normal_out = self.shr(rounded, shift_c);
        // Subnormal or zero: scale into integer range, exact for every representable
        // subnormal, and round to the nearest integer. A carry reaches the smallest normal.
        let safe_magnitude = self.select_u32(is_normal, normal_floor, magnitude);
        let scaled = self.bitcast_f32(safe_magnitude);
        let scaled = self.fmul(scaled, scale);
        let scaled = self.ext_f32(GLSL_ROUND_EVEN, &[scaled]);
        let subnormal_out = self.f_to_u(scaled);
        let body = self.select_u32(is_normal, normal_out, subnormal_out);
        // Rounding has happened, so landing past the last finite encoding is the overflow test.
        let overflowed = self.ult(max_finite_c, body);
        let body = self.select_u32(overflowed, overflow_c, body);
        let body = self.select_u32(is_nan, nan_c, body);
        self.bor(sign, body)
    }

    fn narrow_f16(&mut self, value: Id) -> Id {
        let sixteen = self.c_u32(16);
        let thirteen = self.c_u32(13);
        let one = self.c_u32(1);
        let sign_mask = self.c_u32(0x8000);
        let magnitude_mask = self.c_u32(0x7fff_ffff);
        let inf_bits = self.c_u32(0x7f80_0000);
        let overflow_bits = self.c_u32(0x477f_f000);
        let normal_floor = self.c_u32(0x3880_0000);
        let rebias = self.c_u32(0x3800_0000);
        let half_ulp = self.c_u32(0x0000_0fff);
        let inf_out = self.c_u32(0x7c00);
        let nan_out = self.c_u32(0x7e00);
        let scale = self.c_f32(16_777_216.0);
        let bits = self.bitcast_u32(value);
        let sign = self.shr(bits, sixteen);
        let sign = self.band(sign, sign_mask);
        let magnitude = self.band(bits, magnitude_mask);
        let is_nan = self.ult(inf_bits, magnitude);
        // At or above 65520 — the midpoint between 65504 and the overflow binade — round to
        // infinity (65504 itself is `0x477f_e000`, finite).
        let overflow = self.uge(magnitude, overflow_bits);
        // Normal binary16: rebias the exponent, then round the significand with the
        // add-half-ulp-plus-guard trick; a mantissa carry increments the exponent on its own.
        let is_normal = self.uge(magnitude, normal_floor);
        let adjusted = self.isub(magnitude, rebias);
        let guard = self.shr(adjusted, thirteen);
        let guard = self.band(guard, one);
        let rounding = self.iadd(half_ulp, guard);
        let rounded = self.iadd(adjusted, rounding);
        let normal_out = self.shr(rounded, thirteen);
        // Subnormal binary16 or zero: scale into integer range (exact while the value is at or
        // above 2^-25; anything smaller rounds to zero either way) and round to the nearest
        // integer. The result is at most 1024 — the smallest normal encoding, correctly reached
        // by the rounding carry.
        let safe_magnitude = self.select_u32(is_normal, normal_floor, magnitude);
        let scaled = self.bitcast_f32(safe_magnitude);
        let scaled = self.fmul(scaled, scale);
        let scaled = self.ext_f32(GLSL_ROUND_EVEN, &[scaled]);
        let subnormal_out = self.f_to_u(scaled);
        let body = self.select_u32(is_normal, normal_out, subnormal_out);
        let body = self.select_u32(overflow, inf_out, body);
        let body = self.select_u32(is_nan, nan_out, body);
        self.bor(sign, body)
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

    /// The raw 8 bits of byte-storage `element` as a `u32` (`0..=0xff`): word load, shift the
    /// lane down, mask. No interpretation — an FP8 bit pattern passes through untouched.
    fn load_byte_bits(&mut self, buffers: Id, operand: (Id, Id), element: Id) -> Id {
        let two = self.c_u32(2);
        let three = self.c_u32(3);
        let mask = self.c_u32(0xff);
        let word_index = self.shr(element, two);
        let word = self.load_word(buffers, operand, word_index);
        let lane = self.band(element, three);
        let eight = self.c_u32(8);
        let shift = self.imul(lane, eight);
        let shifted = self.shr(word, shift);
        self.band(shifted, mask)
    }

    /// Byte `element` of a byte-storage operand as a boolean (any nonzero byte is true).
    fn load_bool(&mut self, buffers: Id, operand: (Id, Id), element: Id) -> Id {
        let zero = self.c_u32(0);
        let byte = self.load_byte_bits(buffers, operand, element);
        self.ine(byte, zero)
    }

    /// Write the low 8 bits of `byte` into byte `element` without touching the other bytes of
    /// its word: clear with `OpAtomicAnd`, then set with `OpAtomicOr`. The neighbour-safe
    /// sequence `BOOL` and FP8 lanes share.
    fn store_byte_bits(&mut self, buffers: Id, operand: (Id, Id), element: Id, byte: Id) {
        let two = self.c_u32(2);
        let three = self.c_u32(3);
        let mask = self.c_u32(0xff);
        let word_index = self.shr(element, two);
        let pointer = self.word_pointer(buffers, operand, word_index);
        let lane = self.band(element, three);
        let eight = self.c_u32(8);
        let shift = self.imul(lane, eight);
        let clear = self.shl(mask, shift);
        let clear = self.bnot(clear);
        let masked = self.band(byte, mask);
        let set = self.shl(masked, shift);
        let scope = self.c_u32(SCOPE_DEVICE);
        let semantics = self.c_u32(MEMORY_SEMANTICS_RELAXED);
        let u32_ty = self.u32_ty();
        self.value(OP_ATOMIC_AND, u32_ty, &[pointer, scope, semantics, clear]);
        self.value(OP_ATOMIC_OR, u32_ty, &[pointer, scope, semantics, set]);
    }

    /// Write byte `element` of a byte-storage operand as canonical `0`/`1`.
    fn store_bool(&mut self, buffers: Id, operand: (Id, Id), element: Id, value: Id) {
        let one = self.c_u32(1);
        let zero = self.c_u32(0);
        let byte = self.select_u32(value, one, zero);
        self.store_byte_bits(buffers, operand, element, byte);
    }

    /// The raw 16 bits of half-storage `element` as a `u32` (`0..=0xffff`): word load, shift
    /// the lane down, mask. No float conversion — bit patterns (NaN payloads, subnormals) pass
    /// through untouched.
    fn load_half_bits(&mut self, buffers: Id, operand: (Id, Id), element: Id) -> Id {
        let one = self.c_u32(1);
        let sixteen = self.c_u32(16);
        let mask = self.c_u32(0xffff);
        let word_index = self.shr(element, one);
        let word = self.load_word(buffers, operand, word_index);
        let lane = self.band(element, one);
        let shift = self.imul(lane, sixteen);
        let shifted = self.shr(word, shift);
        self.band(shifted, mask)
    }

    /// Write the low 16 bits of `bits` to half-storage `element` without touching the other
    /// element of its word: clear with `OpAtomicAnd`, then set with `OpAtomicOr` (the same
    /// neighbour-safe pattern as [`Self::store_bool`]).
    fn store_half_bits(&mut self, buffers: Id, operand: (Id, Id), element: Id, bits: Id) {
        let one = self.c_u32(1);
        let sixteen = self.c_u32(16);
        let mask = self.c_u32(0xffff);
        let word_index = self.shr(element, one);
        let pointer = self.word_pointer(buffers, operand, word_index);
        let lane = self.band(element, one);
        let shift = self.imul(lane, sixteen);
        let clear = self.shl(mask, shift);
        let clear = self.bnot(clear);
        let bits = self.band(bits, mask);
        let set = self.shl(bits, shift);
        let scope = self.c_u32(SCOPE_DEVICE);
        let semantics = self.c_u32(MEMORY_SEMANTICS_RELAXED);
        let u32_ty = self.u32_ty();
        self.value(OP_ATOMIC_AND, u32_ty, &[pointer, scope, semantics, clear]);
        self.value(OP_ATOMIC_OR, u32_ty, &[pointer, scope, semantics, set]);
    }

    /// Store one binary32 float at `storage`, narrowing where the storage is narrower.
    fn store_float(
        &mut self,
        storage: Storage,
        buffers: Id,
        operand: (Id, Id),
        element: Id,
        value: Id,
    ) {
        match storage {
            Storage::Word => self.store_f32(buffers, operand, element, value),
            Storage::Half => {
                let bits = self.narrow_f16(value);
                self.store_half_bits(buffers, operand, element, bits);
            }
            Storage::Quarter(format) => {
                let bits = self.narrow_fp8(format, value);
                self.store_byte_bits(buffers, operand, element, bits);
            }
            Storage::Byte => unreachable!("BOOL is not a float storage"),
        }
    }

    /// Load one float element as binary32: word storage bitcasts, half storage is unpacked and
    /// widened exactly by [`Self::widen_f16`].
    fn load_float(&mut self, storage: Storage, buffers: Id, operand: (Id, Id), element: Id) -> Id {
        match storage {
            Storage::Word => self.load_f32(buffers, operand, element),
            Storage::Quarter(format) => {
                let bits = self.load_byte_bits(buffers, operand, element);
                self.widen_fp8(format, bits)
            }
            Storage::Half => {
                let bits = self.load_half_bits(buffers, operand, element);
                self.widen_f16(bits)
            }
            Storage::Byte => unreachable!("byte storage is not a float lane"),
        }
    }

    /// The raw bits of the `lanes` consecutive elements `base..base + lanes` of a `storage`
    /// operand, one `u32` per element (`0..=0xff` at byte storage, `0..=0xffff` at half storage,
    /// the whole word otherwise), loading every storage word the group touches exactly once.
    ///
    /// `base` must be a multiple of `lanes`. When the source packs no more elements per word
    /// than the group holds, the group starts on a word boundary and each lane's word and shift
    /// are compile-time constants; when it packs more (FP8 bytes feeding an FP16 pair), the group
    /// is a sub-span of one word at a run-time byte offset, and still one load.
    fn load_lane_group(
        &mut self,
        storage: Storage,
        buffers: Id,
        operand: (Id, Id),
        base: Id,
        lanes: u32,
    ) -> Vec<Id> {
        let source_lanes = storage.lanes();
        let lane_bits = 32 / source_lanes;
        let first = match source_lanes.trailing_zeros() {
            0 => base,
            shift => {
                let shift = self.c_u32(shift);
                self.shr(base, shift)
            }
        };
        let words: Vec<Id> = (0..lanes.div_ceil(source_lanes))
            .map(|word| {
                let word = self.c_u32(word);
                let index = self.iadd(first, word);
                self.load_word(buffers, operand, index)
            })
            .collect();
        if lane_bits == 32 {
            return words;
        }
        let mask = self.c_u32((1 << lane_bits) - 1);
        let offset = (source_lanes > lanes).then(|| {
            let modulus = self.c_u32(source_lanes - 1);
            self.band(base, modulus)
        });
        (0..lanes)
            .map(|lane| {
                let word = words[(lane / source_lanes) as usize];
                let shift = match offset {
                    None => self.c_u32((lane % source_lanes) * lane_bits),
                    Some(offset) => {
                        let lane = self.c_u32(lane % source_lanes);
                        let position = self.iadd(offset, lane);
                        let bits = self.c_u32(lane_bits);
                        self.imul(position, bits)
                    }
                };
                let shifted = self.shr(word, shift);
                self.band(shifted, mask)
            })
            .collect()
    }

    /// Pack lane values — each already within `32 / values.len()` bits — into one storage word,
    /// lane 0 lowest.
    fn pack_lanes(&mut self, values: &[Id]) -> Id {
        let lane_bits = 32 / values.len() as u32;
        let mut word = values[0];
        for (lane, value) in values.iter().enumerate().skip(1) {
            let shift = self.c_u32(lane as u32 * lane_bits);
            let shifted = self.shl(*value, shift);
            word = self.bor(word, shifted);
        }
        word
    }

    /// Store raw bits into one element of `storage` without touching its word neighbours.
    fn store_lane_bits(
        &mut self,
        storage: Storage,
        buffers: Id,
        operand: (Id, Id),
        element: Id,
        bits: Id,
    ) {
        match storage {
            Storage::Word => self.store_word(buffers, operand, element, bits),
            Storage::Half => self.store_half_bits(buffers, operand, element, bits),
            Storage::Byte | Storage::Quarter(_) => {
                self.store_byte_bits(buffers, operand, element, bits)
            }
        }
    }

    /// Widen one element's raw storage bits to binary32.
    fn widen_bits(&mut self, storage: Storage, bits: Id) -> Id {
        match storage {
            Storage::Word => self.bitcast_f32(bits),
            Storage::Half => self.widen_f16(bits),
            Storage::Quarter(format) => self.widen_fp8(format, bits),
            Storage::Byte => unreachable!("byte storage is not a float lane"),
        }
    }

    /// Narrow binary32 to the raw storage bits of `storage`, crate-owned rounding throughout.
    fn narrow_bits(&mut self, storage: Storage, value: Id) -> Id {
        match storage {
            Storage::Word => self.bitcast_u32(value),
            Storage::Half => self.narrow_f16(value),
            Storage::Quarter(format) => self.narrow_fp8(format, value),
            Storage::Byte => unreachable!("BOOL is not a float storage"),
        }
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

    /// The scalar lane of `op` over already-loaded inputs (`f32` ids for float inputs — binary16
    /// tensors are widened on load by [`Self::widen_f16`] and narrowed on store by
    /// [`Self::narrow_f16`], so every float lane is binary32 — and `bool` ids for byte inputs),
    /// yielding an `f32` or `bool` id per [`ElementwiseOp::output`].
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
                // The bounds arrive as binary32 bit patterns (binary16 bounds are widened
                // host-side, exactly).
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

/// Elementwise kernel: grid-stride over `count` output elements, with `float` the storage of
/// the operator's floating-point tensors (`Word` for FP32, `Half` for FP16).
///
/// Specialization order: `count`; per input `(buffer, base)`; output `(buffer, base)`; when
/// `broadcast`, output `dims[MAX_RANK]` then per input `strides[MAX_RANK]`; then the operator's
/// trailing constants (`CLAMP`: `lo`, `hi` as binary32 bit patterns — binary16 bounds are
/// widened host-side).
fn assemble_elementwise(
    op: ElementwiseOp,
    float: Storage,
    broadcast: bool,
    workgroup: u32,
    buffers: u32,
) -> Vec<u32> {
    let mut b = Builder::new();
    let array = b.buffer_array(buffers);
    let count = b.spec_u32(1);
    let input_storage = op.inputs();
    // Lane tables mark floating-point operands `Word`; resolve those to this kernel's float
    // storage, keeping `BOOL` lanes on byte storage.
    let lane = |storage: Storage| match storage {
        Storage::Word => float,
        other => other,
    };
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
    // IEEE negate and abs are sign-bit operations, not arithmetic: on binary16 they are integer
    // masks on the packed lane, exact for every pattern — subnormals and NaN payloads included
    // (ADR 0008). Every other binary16 lane widens exactly, evaluates at binary32, and narrows
    // once, all in crate-owned integer/binary32 code no driver can demote.
    let bits_lane =
        float == Storage::Half && matches!(op, ElementwiseOp::Negate | ElementwiseOp::Abs);
    let mut values = Vec::with_capacity(inputs.len());
    for (k, operand) in inputs.iter().enumerate() {
        let value = match lane(input_storage[k]) {
            Storage::Word => b.load_f32(array, *operand, indices[k]),
            Storage::Half if bits_lane => b.load_half_bits(array, *operand, indices[k]),
            Storage::Half => {
                let bits = b.load_half_bits(array, *operand, indices[k]);
                b.widen_f16(bits)
            }
            Storage::Byte => b.load_bool(array, *operand, indices[k]),
            // TOSA admits no FP8 elementwise operator, so the tier never selects this lane.
            Storage::Quarter(_) => unreachable!("elementwise FP8 operand"),
        };
        values.push(value);
    }
    let result = if bits_lane {
        let bits = values[0];
        match op {
            ElementwiseOp::Negate => {
                let sign = b.c_u32(0x8000);
                b.bxor(bits, sign)
            }
            ElementwiseOp::Abs => {
                let magnitude = b.c_u32(0x7fff);
                b.band(bits, magnitude)
            }
            _ => unreachable!("bits lane is negate/abs only"),
        }
    } else {
        b.elementwise_lane(op, &values, clamp)
    };
    match lane(op.output()) {
        Storage::Word => b.store_f32(array, output, i, result),
        Storage::Half if bits_lane => b.store_half_bits(array, output, i, result),
        Storage::Half => {
            let bits = b.narrow_f16(result);
            b.store_half_bits(array, output, i, bits);
        }
        Storage::Byte => b.store_bool(array, output, i, result),
        Storage::Quarter(_) => unreachable!("elementwise FP8 result"),
    }
    b.end_loop(scope, counter, stride);
    b.end_main();
    b.finish([workgroup, 1, 1])
}

/// Reduction kernel: one invocation per `(outer, inner)` output element folds the axis in
/// ascending order. Inputs are read at `float` storage and widened to binary32 for the fold —
/// the accumulator width TOSA assigns FP16 sums and products, and exact for the max/min/argmax
/// selections — then narrowed back to `float` storage on store.
///
/// Specialization order: input `(buffer, base)`, output `(buffer, base)`, `outer`, `axis`,
/// `inner`.
fn assemble_reduce(op: ReduceOp, float: Storage, workgroup: u32, buffers: u32) -> Vec<u32> {
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
    let one = b.c_u32(1);
    b.store(index_var, zero);
    let false_id = b.c_false();
    b.store(done_var, false_id);
    b.store(a_var, zero);
    let (inner_scope, a) = b.begin_loop(a_var, axis);
    let offset = b.imul(a, inner);
    let element = b.iadd(base, offset);
    let value = b.load_float(float, array, input, element);
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
            b.store_float(float, array, output, o, acc);
        }
    }
    b.end_loop(scope, counter, stride);
    b.end_main();
    b.finish([workgroup, 1, 1])
}

/// One operand slab of the MATMUL kernel: `rows × cols` elements whose element `(r, c)` lives at
/// `base + (row0 + r) · stride + (col0 + c)` and is in range when `row0 + r < row_limit` and
/// `col0 + c < col_limit`; staged to `tile` at slot `c · rows + r` when `transposed`, else
/// `r · cols + c`.
struct Slab {
    operand: (Id, Id),
    rows: u32,
    cols: u32,
    row0: Id,
    col0: Id,
    stride: Id,
    base: Id,
    row_limit: Id,
    col_limit: Id,
    transposed: bool,
    tile: Id,
}

/// Stage a slab cooperatively: the `invocations` of the workgroup take slab elements (or, for
/// sub-word storage, slab *words*) `q · invocations + lid` in turn, so consecutive invocations
/// read consecutive addresses. Out-of-range elements read element zero and stage `0.0`.
///
/// Sub-word storage is staged a storage word per invocation — four FP8 or two FP16 elements
/// from one `OpLoad` — whenever `stride` is a multiple of the lanes per word, which makes every
/// slab row word-aligned (`col0` and `base` are multiples of the row length or of `stride`).
/// The check is on a specialization constant, so the driver folds it and only one path
/// survives pipeline creation. This is what makes FP8 cheaper than FP32 rather than merely
/// smaller: on Intel Xe3 the per-element path issued one load instruction per element whatever
/// the width, and a GEMV over 16 MiB of FP8 weights ran no faster than one over 64 MiB of FP32,
/// because the memory pipeline was bound by load instructions rather than bytes.
fn stage_slab(b: &mut Builder, input: Storage, array: Id, invocations: u32, lid: Id, slab: &Slab) {
    let f32_ty = b.f32_ty();
    let workgroup_ptr = b.pointer(STORAGE_CLASS_WORKGROUP, f32_ty);
    let zero = b.c_u32(0);
    let zero_f = b.c_f32(0.0);
    let rows_c = b.c_u32(slab.rows);
    let cols_c = b.c_u32(slab.cols);
    let elements = slab.rows * slab.cols;
    let slot = |b: &mut Builder, r: Id, c: Id| {
        if slab.transposed {
            let slot = b.imul(c, rows_c);
            b.iadd(slot, r)
        } else {
            let slot = b.imul(r, cols_c);
            b.iadd(slot, c)
        }
    };
    // Element `(r, c)`'s global index and validity.
    let locate = |b: &mut Builder, r: Id, c: Id| {
        let global_row = b.iadd(slab.row0, r);
        let global_col = b.iadd(slab.col0, c);
        let row_ok = b.ult(global_row, slab.row_limit);
        let col_ok = b.ult(global_col, slab.col_limit);
        let ok = b.land(row_ok, col_ok);
        let element = b.imul(global_row, slab.stride);
        let element = b.iadd(element, slab.base);
        let element = b.iadd(element, global_col);
        let element = b.select_u32(ok, element, zero);
        (element, ok)
    };
    // `q · invocations + lid` over `count` items, guarded only when the count does not divide.
    let each = |b: &mut Builder, count: u32, body: &dyn Fn(&mut Builder, Id)| {
        for q in 0..count.div_ceil(invocations) {
            let offset = b.c_u32(q * invocations);
            let index = b.iadd(offset, lid);
            if count % invocations == 0 {
                body(b, index);
            } else {
                let limit = b.c_u32(count);
                let in_range = b.ult(index, limit);
                b.if_then(in_range, |b| body(b, index));
            }
        }
    };
    let scalar = |b: &mut Builder| {
        each(b, elements, &|b, index| {
            let r = b.udiv(index, cols_c);
            let c = b.umod(index, cols_c);
            let (element, ok) = locate(b, r, c);
            let loaded = b.load_float(input, array, slab.operand, element);
            let value = b.select_f32(ok, loaded, zero_f);
            let slot = slot(b, r, c);
            let pointer = b.access_chain(workgroup_ptr, slab.tile, &[slot]);
            b.store(pointer, value);
        });
    };
    let lanes = input.lanes();
    if lanes == 1 || slab.cols % lanes != 0 {
        scalar(b);
        return;
    }
    let lanes_c = b.c_u32(lanes);
    let words_per_row = slab.cols / lanes;
    let words_per_row_c = b.c_u32(words_per_row);
    let remainder = b.umod(slab.stride, lanes_c);
    let aligned = b.ieq(remainder, zero);
    b.if_then(aligned, |b| {
        each(b, elements / lanes, &|b, word| {
            let r = b.udiv(word, words_per_row_c);
            let word_in_row = b.umod(word, words_per_row_c);
            let c = b.imul(word_in_row, lanes_c);
            // `stride`, `base` and `col0` are all multiples of `lanes` here and `col_limit`
            // need not be, but a word is either wholly inside the tensor or wholly past it
            // only when `col_limit` is a multiple of `lanes`; check the last lane too.
            let (element, ok) = locate(b, r, c);
            let last = b.c_u32(lanes - 1);
            let last_col = b.iadd(c, last);
            let (_, last_ok) = locate(b, r, last_col);
            let ok = b.land(ok, last_ok);
            let element = b.select_u32(ok, element, zero);
            let bits = b.load_lane_group(input, array, slab.operand, element, lanes);
            for (lane, bits) in bits.into_iter().enumerate() {
                let widened = b.widen_bits(input, bits);
                let value = b.select_f32(ok, widened, zero_f);
                let lane_c = b.c_u32(lane as u32);
                let col = b.iadd(c, lane_c);
                let slot = slot(b, r, col);
                let pointer = b.access_chain(workgroup_ptr, slab.tile, &[slot]);
                b.store(pointer, value);
            }
        });
    });
    let unaligned = b.lnot(aligned);
    b.if_then(unaligned, scalar);
}

/// Batched, register-tiled MATMUL reading `input` storage (FP32 words, packed FP16, or packed
/// FP8) and accumulating in binary32 — the accumulator width TOSA assigns FP16 and FP8 MATMUL,
/// and the FP32 tier's own width.
///
/// `out[b, m, n] = Σ_k lhs[b, m, k] · rhs[b, k, n]`, accumulated in ascending `k` with separately
/// rounded multiply and add, so every element is bit-identical to the untiled sequential loop.
///
/// Geometry ([`MatmulGeometry`]): a `tile_x × tile_y` workgroup computes a `block_m × block_n`
/// output block, each invocation a `micro_m × micro_n` register block whose rows are
/// `i · tile_y + ty` and columns `j · tile_x + tx` — interleaved rather than contiguous, so the
/// invocations of a row of the workgroup read consecutive columns and write consecutive
/// outputs. Per `depth`-deep step of `k` the workgroup stages a slab of each operand in shared
/// memory cooperatively (consecutive invocations loading consecutive addresses, the lhs slab
/// stored transposed so its inner-loop reads are conflict-free), then every invocation issues
/// `micro_m + micro_n` shared loads for `micro_m · micro_n` multiply-adds. The
/// one-output-per-invocation kernel this replaces issued two shared loads per multiply-add and
/// staged one element per invocation; on Intel Xe3 it ran a 1024³ FP8 MATMUL at 570 GFLOP/s,
/// below its own FP32 rate, because the per-element FP8 widening was paid once per multiply-add
/// rather than once per `MATMUL_MICRO` of them. (A 256-entry shared-memory widening table,
/// built per workgroup, was measured against the inline integer expansion on Intel Xe3 and made
/// no difference within noise; the expansion stays, having no table to build or barrier to wait
/// on.)
///
/// Out-of-range slab loads read element zero and contribute nothing: the inner loop bound is
/// `min(depth, k - k0)`, never a padded zero product, so signed zeros survive. Out-of-range
/// outputs are computed and discarded.
///
/// Specialization order: `lhs`, `rhs`, output `(buffer, base)`; `m`, `n`, `k`, `batch`.
fn assemble_matmul(
    input: Storage,
    output_storage: Storage,
    geometry: MatmulGeometry,
    buffers: u32,
) -> Vec<u32> {
    let MatmulGeometry {
        tile_x,
        tile_y,
        micro_m,
        micro_n,
        depth,
    } = geometry;
    let invocations = geometry.invocations();
    let block_m = geometry.block_m();
    let block_n = geometry.block_n();
    debug_assert_eq!((block_m * depth) % invocations, 0, "lhs slab stages evenly");
    debug_assert_eq!((block_n * depth) % invocations, 0, "rhs slab stages evenly");
    let mut b = Builder::new();
    let array = b.buffer_array(buffers);
    let lhs = b.spec_operand();
    let rhs = b.spec_operand();
    let output = b.spec_operand();
    let m = b.spec_u32(1);
    let n = b.spec_u32(1);
    let k = b.spec_u32(1);
    let _batch = b.spec_u32(1);
    // lhs slab transposed: `[kk][row]`, `depth × block_m`; rhs slab as is: `[kk][col]`,
    // `depth × block_n`.
    let lhs_tile = b.shared_f32_array(block_m * depth);
    let rhs_tile = b.shared_f32_array(block_n * depth);
    let local_id = b.builtin_uvec3(BUILT_IN_LOCAL_INVOCATION_ID);
    let group_id = b.builtin_uvec3(BUILT_IN_WORKGROUP_ID);

    b.begin_main();
    let u32_ty = b.u32_ty();
    let f32_ty = b.f32_ty();
    let accumulators: Vec<Id> = (0..micro_m * micro_n).map(|_| b.local(f32_ty)).collect();
    let t_var = b.local(u32_ty);
    let kk_var = b.local(u32_ty);
    let tx = b.builtin_component(local_id, 0);
    let ty = b.builtin_component(local_id, 1);
    let gx = b.builtin_component(group_id, 0);
    let gy = b.builtin_component(group_id, 1);
    let z = b.builtin_component(group_id, 2);
    let tile_x_c = b.c_u32(tile_x);
    let depth_c = b.c_u32(depth);
    let block_m_c = b.c_u32(block_m);
    let block_n_c = b.c_u32(block_n);
    let zero = b.c_u32(0);
    let one = b.c_u32(1);
    let zero_f = b.c_f32(0.0);
    for accumulator in &accumulators {
        b.store(*accumulator, zero_f);
    }
    b.store(t_var, zero);
    let row0 = b.imul(gy, block_m_c);
    let col0 = b.imul(gx, block_n_c);
    let lid = b.imul(ty, tile_x_c);
    let lid = b.iadd(lid, tx);
    // lhs batch base: z * m * k; rhs batch base: z * k * n.
    let lhs_batch = b.imul(z, m);
    let lhs_batch = b.imul(lhs_batch, k);
    let rhs_batch = b.imul(z, k);
    let rhs_batch = b.imul(rhs_batch, n);
    let workgroup_ptr = b.pointer(STORAGE_CLASS_WORKGROUP, f32_ty);
    let steps = {
        let k_plus = b.iadd(k, depth_c);
        let k_plus = b.isub(k_plus, one);
        b.udiv(k_plus, depth_c)
    };
    let (outer, t) = b.begin_loop(t_var, steps);
    let k0 = b.imul(t, depth_c);
    // Stage both slabs. lhs: `block_m` rows of `depth` consecutive `k`, stored transposed;
    // rhs: `depth` rows of `block_n` consecutive `n`.
    stage_slab(
        &mut b,
        input,
        array,
        invocations,
        lid,
        &Slab {
            operand: lhs,
            rows: block_m,
            cols: depth,
            row0,
            col0: k0,
            stride: k,
            base: lhs_batch,
            row_limit: m,
            col_limit: k,
            transposed: true,
            tile: lhs_tile,
        },
    );
    stage_slab(
        &mut b,
        input,
        array,
        invocations,
        lid,
        &Slab {
            operand: rhs,
            rows: depth,
            cols: block_n,
            row0: k0,
            col0,
            stride: n,
            base: rhs_batch,
            row_limit: k,
            col_limit: n,
            transposed: false,
            tile: rhs_tile,
        },
    );
    b.workgroup_barrier();
    let remaining = b.isub(k, k0);
    let k_max = b.umin(remaining, depth_c);
    // One `kk` of the inner product for every register-block element.
    let step = |b: &mut Builder, kk: Id| {
        let lhs_row = b.imul(kk, block_m_c);
        let rhs_row = b.imul(kk, block_n_c);
        let a: Vec<Id> = (0..micro_m)
            .map(|i| {
                let offset = b.c_u32(i * tile_y);
                let local_row = b.iadd(offset, ty);
                let slot = b.iadd(lhs_row, local_row);
                let pointer = b.access_chain(workgroup_ptr, lhs_tile, &[slot]);
                b.load(f32_ty, pointer)
            })
            .collect();
        let bv: Vec<Id> = (0..micro_n)
            .map(|j| {
                let offset = b.c_u32(j * tile_x);
                let local_col = b.iadd(offset, tx);
                let slot = b.iadd(rhs_row, local_col);
                let pointer = b.access_chain(workgroup_ptr, rhs_tile, &[slot]);
                b.load(f32_ty, pointer)
            })
            .collect();
        for i in 0..micro_m {
            for j in 0..micro_n {
                let accumulator = accumulators[(i * micro_n + j) as usize];
                let acc = b.load(f32_ty, accumulator);
                let next = b.fma(a[i as usize], bv[j as usize], acc);
                b.store(accumulator, next);
            }
        }
    };
    b.store(kk_var, zero);
    let (inner, kk) = b.begin_loop(kk_var, k_max);
    step(&mut b, kk);
    b.end_loop(inner, kk_var, one);
    b.workgroup_barrier();
    b.end_loop(outer, t_var, one);
    let out_batch = b.imul(z, m);
    let out_batch = b.imul(out_batch, n);
    for i in 0..micro_m {
        let offset = b.c_u32(i * tile_y);
        let row = b.iadd(row0, offset);
        let row = b.iadd(row, ty);
        let row_ok = b.ult(row, m);
        let out_row = b.imul(row, n);
        let out_row = b.iadd(out_batch, out_row);
        for j in 0..micro_n {
            let offset = b.c_u32(j * tile_x);
            let col = b.iadd(col0, offset);
            let col = b.iadd(col, tx);
            let col_ok = b.ult(col, n);
            let in_range = b.land(row_ok, col_ok);
            let accumulator = accumulators[(i * micro_n + j) as usize];
            b.if_then(in_range, |b| {
                let out_index = b.iadd(out_row, col);
                let acc = b.load(f32_ty, accumulator);
                match output_storage {
                    Storage::Half => {
                        let bits = b.narrow_f16(acc);
                        b.store_half_bits(array, output, out_index, bits);
                    }
                    // TOSA defines no MATMUL whose result is FP8 or BOOL; the tier never
                    // selects one.
                    Storage::Byte | Storage::Quarter(_) => unreachable!("MATMUL result storage"),
                    Storage::Word => b.store_f32(array, output, out_index, acc),
                }
            });
        }
    }
    b.end_main();
    b.finish(geometry.local_size())
}

/// Split-`k` streaming MATMUL for `m ≤` [`STREAM_ROWS`]: the decode shape, a few activation rows
/// against a wide weight matrix, where the kernel's only job is to stream `rhs` once at the
/// memory system's rate.
///
/// A 1-D workgroup of [`STREAM_WORKGROUP`] invocations covers [`STREAM_COLUMNS`] columns.
/// Invocation `lid` owns weight word `lid % words` — `lanes` adjacent columns, four FP8, two
/// FP16 or one FP32 — and `k` slice `lid / words` of `splits = STREAM_WORKGROUP / words` equal
/// slices, and carries all [`STREAM_ROWS`] rows of those columns in registers. Its loop is one
/// `rhs` word load, one uniform binary32 lhs load per row, and `rows · lanes` fused
/// multiply-adds — no shared memory, no barrier, and no widening of the lhs, which lowering
/// pre-widens to binary32 (an `m × k` intermediate, negligible next to `rhs`) when the tier's
/// lhs is narrower. Rows at or past `m` are compiled out: the row test is on a specialization
/// constant. At the end every invocation writes its partials to shared memory and each
/// invocation reduces two outputs over the `splits` slices in ascending order, then stores.
///
/// Numerics (ADR 0011): binary32 accumulation with fused multiply-add, `k` split into `splits`
/// contiguous ascending slices summed in fixed order. Not bit-identical to the sequential sum,
/// but deterministic: the same device always produces the same bits, and every device whose
/// `Fma` is correctly rounded produces the same bits as every other.
///
/// The `rhs` word path needs every row of `rhs` word-aligned, i.e. `n` a multiple of `lanes`;
/// the check is on the specialization constant `n`, so only one path survives pipeline
/// creation, and an unaligned `n` falls back to per-element loads.
///
/// Specialization order: `lhs`, `rhs`, output `(buffer, base)`; `m`, `n`, `k`, `batch` — the
/// same payload as [`assemble_matmul`].
fn assemble_matmul_stream(rhs_storage: Storage, output_storage: Storage, buffers: u32) -> Vec<u32> {
    let lanes = rhs_storage.lanes();
    let words = STREAM_COLUMNS / lanes;
    let splits = STREAM_WORKGROUP / words;
    let rows = STREAM_ROWS;
    let mut b = Builder::new();
    let array = b.buffer_array(buffers);
    let lhs = b.spec_operand();
    let rhs = b.spec_operand();
    let output = b.spec_operand();
    let m = b.spec_u32(1);
    let n = b.spec_u32(1);
    let k = b.spec_u32(1);
    let _batch = b.spec_u32(1);
    let partials = b.shared_f32_array(STREAM_WORKGROUP * rows * lanes);
    let local_id = b.builtin_uvec3(BUILT_IN_LOCAL_INVOCATION_ID);
    let group_id = b.builtin_uvec3(BUILT_IN_WORKGROUP_ID);

    b.begin_main();
    let u32_ty = b.u32_ty();
    let f32_ty = b.f32_ty();
    let accumulators: Vec<Id> = (0..rows * lanes).map(|_| b.local(f32_ty)).collect();
    let kk_var = b.local(u32_ty);
    let lid = b.builtin_component(local_id, 0);
    let gx = b.builtin_component(group_id, 0);
    let z = b.builtin_component(group_id, 2);
    let zero = b.c_u32(0);
    let one = b.c_u32(1);
    let zero_f = b.c_f32(0.0);
    let words_c = b.c_u32(words);
    let lanes_c = b.c_u32(lanes);
    let splits_c = b.c_u32(splits);
    let columns_c = b.c_u32(STREAM_COLUMNS);
    for accumulator in &accumulators {
        b.store(*accumulator, zero_f);
    }
    let w = b.umod(lid, words_c);
    let slice = b.udiv(lid, words_c);
    let col0 = b.imul(gx, columns_c);
    let word_col = b.imul(w, lanes_c);
    let col_base = b.iadd(col0, word_col);
    // This invocation's `k` slice: `[slice · chunk, min(k, (slice + 1) · chunk))`.
    let splits_less_one = b.c_u32(splits - 1);
    let k_plus = b.iadd(k, splits_less_one);
    let chunk = b.udiv(k_plus, splits_c);
    let k_begin = b.imul(slice, chunk);
    let k_end = b.iadd(k_begin, chunk);
    let k_end = b.umin(k_end, k);
    let lhs_batch = b.imul(z, m);
    let lhs_batch = b.imul(lhs_batch, k);
    let rhs_batch = b.imul(z, k);
    let rhs_batch = b.imul(rhs_batch, n);
    let rhs_col = b.iadd(rhs_batch, col_base);
    let row_ok: Vec<Id> = (0..rows)
        .map(|i| {
            let row = b.c_u32(i);
            b.ult(row, m)
        })
        .collect();
    let lhs_rows: Vec<Id> = (0..rows)
        .map(|i| {
            let row = b.c_u32(i);
            let offset = b.imul(row, k);
            b.iadd(lhs_batch, offset)
        })
        .collect();
    // Element index of `rhs[kk, col_base]`, clamped to zero when the word lies past `n`.
    let rhs_word_element = |b: &mut Builder, kk: Id| {
        let col_ok = b.ult(col_base, n);
        let element = b.imul(kk, n);
        let element = b.iadd(element, rhs_col);
        b.select_u32(col_ok, element, zero)
    };
    // The inner product over the slice, `load_rhs` yielding the `lanes` weights at `kk`.
    let accumulate = |b: &mut Builder, load_rhs: &dyn Fn(&mut Builder, Id) -> Vec<Id>| {
        b.store(kk_var, k_begin);
        let (scope, kk) = b.begin_loop(kk_var, k_end);
        let weights = load_rhs(b, kk);
        for i in 0..rows as usize {
            b.if_then(row_ok[i], |b| {
                let element = b.iadd(lhs_rows[i], kk);
                let a = b.load_f32(array, lhs, element);
                for (l, weight) in weights.iter().enumerate() {
                    let accumulator = accumulators[i * lanes as usize + l];
                    let acc = b.load(f32_ty, accumulator);
                    let next = b.fma(a, *weight, acc);
                    b.store(accumulator, next);
                }
            });
        }
        b.end_loop(scope, kk_var, one);
    };
    if lanes == 1 {
        accumulate(&mut b, &|b, kk| {
            let element = rhs_word_element(b, kk);
            vec![b.load_f32(array, rhs, element)]
        });
    } else {
        let remainder = b.umod(n, lanes_c);
        let aligned = b.ieq(remainder, zero);
        b.if_then(aligned, |b| {
            accumulate(b, &|b, kk| {
                let element = rhs_word_element(b, kk);
                let bits = b.load_lane_group(rhs_storage, array, rhs, element, lanes);
                bits.into_iter()
                    .map(|bits| b.widen_bits(rhs_storage, bits))
                    .collect()
            });
        });
        let unaligned = b.lnot(aligned);
        b.if_then(unaligned, |b| {
            accumulate(b, &|b, kk| {
                (0..lanes)
                    .map(|l| {
                        let l = b.c_u32(l);
                        let col = b.iadd(col_base, l);
                        let col_ok = b.ult(col, n);
                        let element = b.imul(kk, n);
                        let element = b.iadd(element, rhs_batch);
                        let element = b.iadd(element, col);
                        let element = b.select_u32(col_ok, element, zero);
                        b.load_float(rhs_storage, array, rhs, element)
                    })
                    .collect()
            });
        });
    }
    // Partials to shared memory: invocation `lid`'s block of `rows · lanes` values.
    let workgroup_ptr = b.pointer(STORAGE_CLASS_WORKGROUP, f32_ty);
    let block = b.c_u32(rows * lanes);
    let base = b.imul(lid, block);
    for (index, accumulator) in accumulators.iter().enumerate() {
        let offset = b.c_u32(index as u32);
        let slot = b.iadd(base, offset);
        let value = b.load(f32_ty, *accumulator);
        let pointer = b.access_chain(workgroup_ptr, partials, &[slot]);
        b.store(pointer, value);
    }
    b.workgroup_barrier();
    // Reduce: output `o = lid · per_invocation + j` is row `o / STREAM_COLUMNS`, column
    // `o % STREAM_COLUMNS`, summed over the slices in ascending order from slice zero.
    let per_invocation = rows * STREAM_COLUMNS / STREAM_WORKGROUP;
    let per_invocation_c = b.c_u32(per_invocation);
    let first = b.imul(lid, per_invocation_c);
    let out_batch = b.imul(z, m);
    let out_batch = b.imul(out_batch, n);
    for j in 0..per_invocation {
        let j = b.c_u32(j);
        let o = b.iadd(first, j);
        let row = b.udiv(o, columns_c);
        let col = b.umod(o, columns_c);
        let word = b.udiv(col, lanes_c);
        let lane = b.umod(col, lanes_c);
        let within = b.imul(row, lanes_c);
        let within = b.iadd(within, lane);
        let mut sum = None;
        for split in 0..splits {
            let offset = b.c_u32(split * words);
            let owner = b.iadd(offset, word);
            let slot = b.imul(owner, block);
            let slot = b.iadd(slot, within);
            let pointer = b.access_chain(workgroup_ptr, partials, &[slot]);
            let partial = b.load(f32_ty, pointer);
            sum = Some(match sum {
                None => partial,
                Some(sum) => b.fadd(sum, partial),
            });
        }
        let sum = sum.expect("at least one slice");
        let global_col = b.iadd(col0, col);
        let row_in = b.ult(row, m);
        let col_in = b.ult(global_col, n);
        let in_range = b.land(row_in, col_in);
        b.if_then(in_range, |b| {
            let out = b.imul(row, n);
            let out = b.iadd(out, out_batch);
            let out = b.iadd(out, global_col);
            b.store_float(output_storage, array, output, out, sum);
        });
    }
    b.end_main();
    b.finish([STREAM_WORKGROUP, 1, 1])
}

/// Cooperative-matrix NVFP4 projection for devices that advertise subgroup FP16 8x16x16 with
/// FP32 accumulation. One subgroup owns an 8-token by 16-output tile. Packed FP4 weights are
/// decoded once into a binary16 workgroup tile, then the driver lowers
/// `OpCooperativeMatrixMulAddKHR` to the architecture's matrix engine (XMX on Intel Xe2/Xe3).
fn assemble_nvfp4_matmul_cooperative(buffers: u32) -> Vec<u32> {
    let mut b = Builder::new();
    b.enable_cooperative_matrix();
    let array = b.buffer_array(buffers);
    let activation = b.spec_operand();
    let packed = core::array::from_fn::<_, 6, _>(|_| b.spec_operand());
    let block_scales = core::array::from_fn::<_, 6, _>(|_| b.spec_operand());
    let tensor_scale = b.spec_operand();
    let output = b.spec_operand();
    let m = b.spec_u32(1);
    let n = b.spec_u32(1);
    let k = b.spec_u32(16);
    let epilogue = b.spec_u32(0);
    let weight_mode = b.spec_u32(0);

    let a_tile = b.shared_f16_array(8 * 16);
    let b_tile = b.shared_f16_array(16 * 16);
    let c_tile = b.shared_f32_array(8 * 16);
    let local_id = b.builtin_uvec3(BUILT_IN_LOCAL_INVOCATION_ID);
    let group_id = b.builtin_uvec3(BUILT_IN_WORKGROUP_ID);

    b.begin_main();
    let u32_ty = b.u32_ty();
    let f32_ty = b.f32_ty();
    let f16_ty = b.f16_ty();
    let matrix_a_ty = b.cooperative_matrix_ty(f16_ty, 8, 16, 0);
    let matrix_b_ty = b.cooperative_matrix_ty(f16_ty, 16, 16, 1);
    let matrix_c_ty = b.cooperative_matrix_ty(f32_ty, 8, 16, 2);
    let accumulator = b.local(matrix_c_ty);
    let block_var = b.local(u32_ty);
    let fp4 = b.private_u32_array(&[
        0.0f32.to_bits(),
        0.5f32.to_bits(),
        1.0f32.to_bits(),
        1.5f32.to_bits(),
        2.0f32.to_bits(),
        3.0f32.to_bits(),
        4.0f32.to_bits(),
        6.0f32.to_bits(),
        (-0.0f32).to_bits(),
        (-0.5f32).to_bits(),
        (-1.0f32).to_bits(),
        (-1.5f32).to_bits(),
        (-2.0f32).to_bits(),
        (-3.0f32).to_bits(),
        (-4.0f32).to_bits(),
        (-6.0f32).to_bits(),
    ]);
    let lid = b.builtin_component(local_id, 0);
    let output_tile = b.builtin_component(group_id, 0);
    let token_tile = b.builtin_component(group_id, 1);
    let zero = b.c_u32(0);
    let one = b.c_u32(1);
    let two = b.c_u32(2);
    let four = b.c_u32(4);
    let eight = b.c_u32(8);
    let sixteen = b.c_u32(16);
    let zero_f = b.c_f32(0.0);
    let zero_h = b.f32_to_f16(zero_f);
    let blocks = b.udiv(k, sixteen);
    let output_base = b.imul(output_tile, sixteen);
    let token_base = b.imul(token_tile, eight);
    // Cooperative lowering is selected only for shared weights. Keep the specialization operand
    // live so an artifact compiled with the wrong mode cannot silently use this shader.
    let shared_mode = b.ieq(weight_mode, zero);
    let row_bytes = b.udiv(k, two);

    let f32_workgroup_ptr = b.pointer(STORAGE_CLASS_WORKGROUP, f32_ty);
    let f16_workgroup_ptr = b.pointer(STORAGE_CLASS_WORKGROUP, f16_ty);
    // Initialize the accumulator tile once. Each lane owns four consecutive-stride elements.
    for wave in 0..4 {
        let offset = b.c_u32(wave * 32);
        let element = b.iadd(lid, offset);
        let pointer = b.access_chain(f32_workgroup_ptr, c_tile, &[element]);
        b.store(pointer, zero_f);
    }
    b.workgroup_barrier();
    let c_pointer = b.access_chain(f32_workgroup_ptr, c_tile, &[zero]);
    let initial = b.cooperative_load(matrix_c_ty, c_pointer, sixteen);
    b.store(accumulator, initial);

    b.store(block_var, zero);
    let (scope, block) = b.begin_loop(block_var, blocks);
    let activation_block = b.imul(block, sixteen);
    // A is a real 8x16 token tile. The final partial tile is zero padded, allowing every native
    // cooperative-matrix row to perform useful prompt work whenever eight tokens remain.
    for wave in 0..4 {
        let offset = b.c_u32(wave * 32);
        let element = b.iadd(lid, offset);
        let tile_row = b.udiv(element, sixteen);
        let tile_column = b.umod(element, sixteen);
        let token = b.iadd(token_base, tile_row);
        let token_in_range = b.ult(token, m);
        let activation_row = b.imul(token, k);
        let source = b.iadd(activation_row, activation_block);
        let source = b.iadd(source, tile_column);
        let source = b.select_u32(token_in_range, source, zero);
        let value = b.load_f32(array, activation, source);
        let value = b.select_f32(token_in_range, value, zero_f);
        let value = b.f32_to_f16(value);
        let pointer = b.access_chain(f16_workgroup_ptr, a_tile, &[element]);
        b.store(pointer, value);
    }

    // B is KxN in row-major cooperative-matrix order, while the checkpoint is N rows of packed
    // K. Decode and transpose one 16x16 tile cooperatively.
    for wave in 0..8 {
        let offset = b.c_u32(wave * 32);
        let element = b.iadd(lid, offset);
        let k_lane = b.udiv(element, sixteen);
        let column = b.umod(element, sixteen);
        let output_row = b.iadd(output_base, column);
        let in_range = b.ult(output_row, n);
        let pointer = b.access_chain(f16_workgroup_ptr, b_tile, &[element]);
        b.store(pointer, zero_h);
        b.if_then(in_range, |b| {
            let weight_row = output_row;
            let packed_row = b.imul(weight_row, row_bytes);
            let packed_block = b.imul(block, eight);
            let packed_base = b.iadd(packed_row, packed_block);
            let packed_lane = b.udiv(k_lane, two);
            let packed_index = b.iadd(packed_base, packed_lane);
            let codes = b.load_byte_bits(array, packed[0], packed_index);
            let nibble_mask = b.c_u32(15);
            let low = b.band(codes, nibble_mask);
            let high = b.shr(codes, four);
            let parity = b.band(k_lane, one);
            let odd = b.ine(parity, zero);
            let code = b.select_u32(odd, high, low);
            let private_u32 = b.pointer(STORAGE_CLASS_PRIVATE, u32_ty);
            let fp4_pointer = b.access_chain(private_u32, fp4, &[code]);
            let weight_bits = b.load(u32_ty, fp4_pointer);
            let weight = b.bitcast_f32(weight_bits);
            let scale_row = b.imul(weight_row, blocks);
            let scale_index = b.iadd(scale_row, block);
            let scale_bits = b.load_byte_bits(array, block_scales[0], scale_index);
            let scale = b.widen_fp8(Fp8Format::E4M3, scale_bits);
            let weight = b.fmul(weight, scale);
            let weight = b.f32_to_f16(weight);
            b.store(pointer, weight);
        });
    }
    b.workgroup_barrier();
    let a_pointer = b.access_chain(f16_workgroup_ptr, a_tile, &[zero]);
    let b_pointer = b.access_chain(f16_workgroup_ptr, b_tile, &[zero]);
    let matrix_a = b.cooperative_load(matrix_a_ty, a_pointer, sixteen);
    let matrix_b = b.cooperative_load(matrix_b_ty, b_pointer, sixteen);
    let matrix_c = b.load(matrix_c_ty, accumulator);
    let matrix_c = b.cooperative_mul_add(matrix_c_ty, matrix_a, matrix_b, matrix_c);
    b.store(accumulator, matrix_c);
    b.workgroup_barrier();
    b.end_loop(scope, block_var, one);

    let matrix_c = b.load(matrix_c_ty, accumulator);
    b.cooperative_store(c_pointer, matrix_c, sixteen);
    b.workgroup_barrier();
    for wave in 0..4 {
        let offset = b.c_u32(wave * 32);
        let element = b.iadd(lid, offset);
        let tile_row = b.udiv(element, sixteen);
        let tile_column = b.umod(element, sixteen);
        let token = b.iadd(token_base, tile_row);
        let output_row = b.iadd(output_base, tile_column);
        let token_in_range = b.ult(token, m);
        let row_in_tensor = b.ult(output_row, n);
        let write = b.land(token_in_range, row_in_tensor);
        let write = b.land(write, shared_mode);
        b.if_then(write, |b| {
            let pointer = b.access_chain(f32_workgroup_ptr, c_tile, &[element]);
            let sum = b.load(f32_ty, pointer);
            let scale = b.load_f32(array, tensor_scale, zero);
            let sum = b.fmul(sum, scale);
            let negated = b.fneg(sum);
            let exp = b.ext_f32(GLSL_EXP, &[negated]);
            let one_f = b.c_f32(1.0);
            let denominator = b.fadd(one_f, exp);
            let sigmoid = b.fdiv(one_f, denominator);
            let silu = b.fmul(sum, sigmoid);
            let is_silu = b.ieq(epilogue, one);
            let is_sigmoid = b.ieq(epilogue, two);
            let three = b.c_u32(3);
            let zero_f = b.c_f32(0.0);
            let squared_input = sum;
            let positive = b.fogt(squared_input, zero_f);
            let rectified = b.select_f32(positive, squared_input, zero_f);
            let squared = b.fmul(rectified, rectified);
            let is_squared = b.ieq(epilogue, three);
            let sum = b.select_f32(is_silu, silu, sum);
            let sum = b.select_f32(is_sigmoid, sigmoid, sum);
            let sum = b.select_f32(is_squared, squared, sum);
            let destination = b.imul(token, n);
            let destination = b.iadd(destination, output_row);
            b.store_f32(array, output, destination, sum);
        });
    }
    b.end_main();
    b.finish([32, 1, 1])
}

/// Subgroup-reduced scalar NVFP4 projection. A fixed 32-lane subgroup folds eight output rows,
/// reusing every activation load across them and cutting dispatch count by four. Devices without
/// subgroup arithmetic retain the portable 64-lane shared-memory reduction below.
fn assemble_nvfp4_matmul_subgroup(buffers: u32) -> Vec<u32> {
    let mut b = Builder::new();
    b.enable_subgroup_arithmetic();
    let array = b.buffer_array(buffers);
    let activation = b.spec_operand();
    let packed = core::array::from_fn::<_, 6, _>(|_| b.spec_operand());
    let block_scales = core::array::from_fn::<_, 6, _>(|_| b.spec_operand());
    let tensor_scale = b.spec_operand();
    let output = b.spec_operand();
    let _m = b.spec_u32(1);
    let n = b.spec_u32(1);
    let k = b.spec_u32(16);
    let epilogue = b.spec_u32(0);
    let weight_mode = b.spec_u32(0);
    let local_id = b.builtin_uvec3(BUILT_IN_LOCAL_INVOCATION_ID);
    let group_id = b.builtin_uvec3(BUILT_IN_WORKGROUP_ID);

    b.begin_main();
    let u32_ty = b.u32_ty();
    let f32_ty = b.f32_ty();
    let block_var = b.local(u32_ty);
    let accumulators = core::array::from_fn::<_, 8, _>(|_| b.local(f32_ty));
    let fp4 = b.private_u32_array(&[
        0.0f32.to_bits(),
        0.5f32.to_bits(),
        1.0f32.to_bits(),
        1.5f32.to_bits(),
        2.0f32.to_bits(),
        3.0f32.to_bits(),
        4.0f32.to_bits(),
        6.0f32.to_bits(),
        (-0.0f32).to_bits(),
        (-0.5f32).to_bits(),
        (-1.0f32).to_bits(),
        (-1.5f32).to_bits(),
        (-2.0f32).to_bits(),
        (-3.0f32).to_bits(),
        (-4.0f32).to_bits(),
        (-6.0f32).to_bits(),
    ]);
    let lid = b.builtin_component(local_id, 0);
    let output_group = b.builtin_component(group_id, 0);
    let token = b.builtin_component(group_id, 1);
    let zero = b.c_u32(0);
    let zero_f = b.c_f32(0.0);
    let four = b.c_u32(4);
    let eight = b.c_u32(8);
    let sixteen = b.c_u32(16);
    let thirty_two = b.c_u32(32);
    let nibble_mask = b.c_u32(15);
    let blocks = b.udiv(k, sixteen);
    let output_base = b.imul(output_group, eight);
    let mut packed_operand = packed[0];
    let mut scale_operand = block_scales[0];
    for index in 1..6 {
        let index_id = b.c_u32(index as u32);
        let selected = b.ieq(token, index_id);
        packed_operand.0 = b.select_u32(selected, packed[index].0, packed_operand.0);
        packed_operand.1 = b.select_u32(selected, packed[index].1, packed_operand.1);
        scale_operand.0 = b.select_u32(selected, block_scales[index].0, scale_operand.0);
        scale_operand.1 = b.select_u32(selected, block_scales[index].1, scale_operand.1);
    }
    let contiguous_mode = b.c_u32(1);
    let contiguous = b.ieq(weight_mode, contiguous_mode);
    let weight_batch = b.select_u32(contiguous, token, zero);
    let batch_rows = b.imul(weight_batch, n);
    let two = b.c_u32(2);
    let row_bytes = b.udiv(k, two);
    let activation_row = b.imul(token, k);
    let output_rows = core::array::from_fn::<_, 8, _>(|index| {
        let offset = b.c_u32(index as u32);
        b.iadd(output_base, offset)
    });
    let row_valid = output_rows.map(|row| b.ult(row, n));
    let weight_rows = core::array::from_fn::<_, 8, _>(|index| {
        let safe_row = b.select_u32(row_valid[index], output_rows[index], zero);
        b.iadd(batch_rows, safe_row)
    });
    let row_blocks = weight_rows.map(|row| b.imul(row, blocks));
    let packed_rows = weight_rows.map(|row| b.imul(row, row_bytes));
    for accumulator in accumulators {
        b.store(accumulator, zero_f);
    }
    b.store(block_var, lid);
    let (scope, block) = b.begin_loop(block_var, blocks);
    let scales = core::array::from_fn::<_, 8, _>(|index| {
        let scale_element = b.iadd(row_blocks[index], block);
        let scale_bits = b.load_byte_bits(array, scale_operand, scale_element);
        b.widen_fp8(Fp8Format::E4M3, scale_bits)
    });
    let packed_block = b.imul(block, eight);
    let packed_bases = packed_rows.map(|row| b.iadd(row, packed_block));
    let activation_block = b.imul(block, sixteen);
    let activation_base = b.iadd(activation_row, activation_block);
    for byte in 0..8 {
        let byte_offset = b.c_u32(byte);
        let codes = core::array::from_fn::<_, 8, _>(|index| {
            let byte_index = b.iadd(packed_bases[index], byte_offset);
            b.load_byte_bits(array, packed_operand, byte_index)
        });
        for lane in 0..2 {
            let inner = b.c_u32(byte * 2 + lane as u32);
            let element = b.iadd(activation_base, inner);
            let value = b.load_f32(array, activation, element);
            for index in 0..8 {
                let code = if lane == 0 {
                    b.band(codes[index], nibble_mask)
                } else {
                    b.shr(codes[index], four)
                };
                let pointer_ty = b.pointer(STORAGE_CLASS_PRIVATE, u32_ty);
                let pointer = b.access_chain(pointer_ty, fp4, &[code]);
                let weight_bits = b.load(u32_ty, pointer);
                let weight = b.bitcast_f32(weight_bits);
                let weight = b.fmul(weight, scales[index]);
                let acc = b.load(f32_ty, accumulators[index]);
                let next = b.fma(value, weight, acc);
                b.store(accumulators[index], next);
            }
        }
    }
    b.end_loop(scope, block_var, thirty_two);

    let sums = accumulators.map(|accumulator| {
        let partial = b.load(f32_ty, accumulator);
        b.subgroup_sum_f32(partial)
    });

    let first = b.ieq(lid, zero);
    b.if_then(first, |b| {
        let shared_mode = b.c_u32(0);
        let batched = b.ine(weight_mode, shared_mode);
        let tensor_scale_index = b.select_u32(batched, token, zero);
        let tensor_scale = b.load_f32(array, tensor_scale, tensor_scale_index);
        for index in 0..8 {
            b.if_then(row_valid[index], |b| {
                let sum = b.fmul(sums[index], tensor_scale);
                let negated = b.fneg(sum);
                let exp = b.ext_f32(GLSL_EXP, &[negated]);
                let one_f = b.c_f32(1.0);
                let denominator = b.fadd(one_f, exp);
                let sigmoid = b.fdiv(one_f, denominator);
                let silu = b.fmul(sum, sigmoid);
                let silu_mode = b.c_u32(1);
                let sigmoid_mode = b.c_u32(2);
                let squared_mode = b.c_u32(3);
                let is_silu = b.ieq(epilogue, silu_mode);
                let is_sigmoid = b.ieq(epilogue, sigmoid_mode);
                let zero_f = b.c_f32(0.0);
                let squared_input = sum;
                let positive = b.fogt(squared_input, zero_f);
                let rectified = b.select_f32(positive, squared_input, zero_f);
                let squared = b.fmul(rectified, rectified);
                let is_squared = b.ieq(epilogue, squared_mode);
                let sum = b.select_f32(is_silu, silu, sum);
                let sum = b.select_f32(is_sigmoid, sigmoid, sum);
                let sum = b.select_f32(is_squared, squared, sum);
                let element = b.imul(token, n);
                let element = b.iadd(element, output_rows[index]);
                b.store_f32(array, output, element, sum);
            });
        }
    });
    b.end_main();
    b.finish([32, 1, 1])
}

/// Scalar baseline for native NVFP4 projections. One workgroup owns one
/// `[token, output-row]` dot product. Its 64 invocations divide whole
/// 16-weight scale blocks, so every packed byte and scale is read exactly once
/// and only 64 partial sums cross workgroup memory. The same artifact can later
/// select a cooperative-matrix/XMX kernel without changing its storage ABI.
fn assemble_nvfp4_matmul(buffers: u32) -> Vec<u32> {
    let mut b = Builder::new();
    let array = b.buffer_array(buffers);
    let activation = b.spec_operand();
    let packed = core::array::from_fn::<_, 6, _>(|_| b.spec_operand());
    let block_scales = core::array::from_fn::<_, 6, _>(|_| b.spec_operand());
    let tensor_scale = b.spec_operand();
    let output = b.spec_operand();
    let _m = b.spec_u32(1);
    let n = b.spec_u32(1);
    let k = b.spec_u32(16);
    let epilogue = b.spec_u32(0);
    let weight_mode = b.spec_u32(0);
    let partials = b.shared_f32_array(STREAM_WORKGROUP);
    let local_id = b.builtin_uvec3(BUILT_IN_LOCAL_INVOCATION_ID);
    let group_id = b.builtin_uvec3(BUILT_IN_WORKGROUP_ID);

    b.begin_main();
    let u32_ty = b.u32_ty();
    let f32_ty = b.f32_ty();
    let block_var = b.local(u32_ty);
    let accumulator = b.local(f32_ty);
    let fp4 = b.private_u32_array(&[
        0.0f32.to_bits(),
        0.5f32.to_bits(),
        1.0f32.to_bits(),
        1.5f32.to_bits(),
        2.0f32.to_bits(),
        3.0f32.to_bits(),
        4.0f32.to_bits(),
        6.0f32.to_bits(),
        (-0.0f32).to_bits(),
        (-0.5f32).to_bits(),
        (-1.0f32).to_bits(),
        (-1.5f32).to_bits(),
        (-2.0f32).to_bits(),
        (-3.0f32).to_bits(),
        (-4.0f32).to_bits(),
        (-6.0f32).to_bits(),
    ]);
    let lid = b.builtin_component(local_id, 0);
    let out_row = b.builtin_component(group_id, 0);
    let token = b.builtin_component(group_id, 1);
    let zero = b.c_u32(0);
    let zero_f = b.c_f32(0.0);
    let one = b.c_u32(1);
    let four = b.c_u32(4);
    let eight = b.c_u32(8);
    let sixteen = b.c_u32(16);
    let sixty_four = b.c_u32(STREAM_WORKGROUP);
    let nibble_mask = b.c_u32(15);
    let blocks = b.udiv(k, sixteen);
    let mut packed_operand = packed[0];
    let mut scale_operand = block_scales[0];
    for index in 1..6 {
        let index_id = b.c_u32(index as u32);
        let selected = b.ieq(token, index_id);
        packed_operand.0 = b.select_u32(selected, packed[index].0, packed_operand.0);
        packed_operand.1 = b.select_u32(selected, packed[index].1, packed_operand.1);
        scale_operand.0 = b.select_u32(selected, block_scales[index].0, scale_operand.0);
        scale_operand.1 = b.select_u32(selected, block_scales[index].1, scale_operand.1);
    }
    let contiguous_mode = b.c_u32(1);
    let contiguous = b.ieq(weight_mode, contiguous_mode);
    let weight_batch = b.select_u32(contiguous, token, zero);
    let batch_rows = b.imul(weight_batch, n);
    let weight_row = b.iadd(batch_rows, out_row);
    let row_blocks = b.imul(weight_row, blocks);
    let two = b.c_u32(2);
    let row_bytes = b.udiv(k, two);
    let packed_row = b.imul(weight_row, row_bytes);
    let activation_row = b.imul(token, k);
    b.store(accumulator, zero_f);
    b.store(block_var, lid);
    let (scope, block) = b.begin_loop(block_var, blocks);
    let scale_element = b.iadd(row_blocks, block);
    let scale_bits = b.load_byte_bits(array, scale_operand, scale_element);
    let scale = b.widen_fp8(Fp8Format::E4M3, scale_bits);
    let packed_block = b.imul(block, eight);
    let packed_base = b.iadd(packed_row, packed_block);
    let activation_block = b.imul(block, sixteen);
    let activation_base = b.iadd(activation_row, activation_block);
    for byte in 0..8 {
        let byte_offset = b.c_u32(byte);
        let byte_index = b.iadd(packed_base, byte_offset);
        let codes = b.load_byte_bits(array, packed_operand, byte_index);
        let low = b.band(codes, nibble_mask);
        let high = b.shr(codes, four);
        for (lane, code) in [low, high].into_iter().enumerate() {
            let pointer_ty = b.pointer(STORAGE_CLASS_PRIVATE, u32_ty);
            let pointer = b.access_chain(pointer_ty, fp4, &[code]);
            let weight_bits = b.load(u32_ty, pointer);
            let weight = b.bitcast_f32(weight_bits);
            let weight = b.fmul(weight, scale);
            let inner = b.c_u32(byte * 2 + lane as u32);
            let element = b.iadd(activation_base, inner);
            let value = b.load_f32(array, activation, element);
            let acc = b.load(f32_ty, accumulator);
            let next = b.fma(value, weight, acc);
            b.store(accumulator, next);
        }
    }
    b.end_loop(scope, block_var, sixty_four);

    let workgroup_ptr = b.pointer(STORAGE_CLASS_WORKGROUP, f32_ty);
    let partial_ptr = b.access_chain(workgroup_ptr, partials, &[lid]);
    let partial = b.load(f32_ty, accumulator);
    b.store(partial_ptr, partial);
    b.workgroup_barrier();

    let first = b.ieq(lid, zero);
    b.if_then(first, |b| {
        let sum_var = b.local(f32_ty);
        let index_var = b.local(u32_ty);
        b.store(sum_var, zero_f);
        b.store(index_var, zero);
        let (sum_scope, index) = b.begin_loop(index_var, sixty_four);
        let pointer = b.access_chain(workgroup_ptr, partials, &[index]);
        let value = b.load(f32_ty, pointer);
        let sum = b.load(f32_ty, sum_var);
        let next = b.fadd(sum, value);
        b.store(sum_var, next);
        b.end_loop(sum_scope, index_var, one);
        let sum = b.load(f32_ty, sum_var);
        let shared_mode = b.c_u32(0);
        let batched = b.ine(weight_mode, shared_mode);
        let tensor_scale_index = b.select_u32(batched, token, zero);
        let tensor_scale = b.load_f32(array, tensor_scale, tensor_scale_index);
        let sum = b.fmul(sum, tensor_scale);
        let negated = b.fneg(sum);
        let exp = b.ext_f32(GLSL_EXP, &[negated]);
        let one_f = b.c_f32(1.0);
        let denominator = b.fadd(one_f, exp);
        let sigmoid = b.fdiv(one_f, denominator);
        let silu = b.fmul(sum, sigmoid);
        let silu_mode = b.c_u32(1);
        let sigmoid_mode = b.c_u32(2);
        let squared_mode = b.c_u32(3);
        let is_silu = b.ieq(epilogue, silu_mode);
        let is_sigmoid = b.ieq(epilogue, sigmoid_mode);
        let zero_f = b.c_f32(0.0);
        let squared_input = sum;
        let positive = b.fogt(squared_input, zero_f);
        let rectified = b.select_f32(positive, squared_input, zero_f);
        let squared = b.fmul(rectified, rectified);
        let is_squared = b.ieq(epilogue, squared_mode);
        let sum = b.select_f32(is_silu, silu, sum);
        let sum = b.select_f32(is_sigmoid, sigmoid, sum);
        let sum = b.select_f32(is_squared, squared, sum);
        let element = b.imul(token, n);
        let element = b.iadd(element, out_row);
        b.store_f32(array, output, element, sum);
    });
    b.end_main();
    b.finish([STREAM_WORKGROUP, 1, 1])
}

/// NHWC MAX_POOL2D at `float` storage: one invocation per output element folds its window with
/// `apply_max_s` from `-inf`; padded positions are skipped, never substituted. FP16 inputs are
/// widened for the fold — an exact selection, so the narrowed store is exact.
///
/// Specialization order: input, output `(buffer, base)`; `batch`, `height`, `width`, `channels`,
/// `out_height`, `out_width`, `kernel_h`, `kernel_w`, `stride_h`, `stride_w`, `pad_top`,
/// `pad_left`.
fn assemble_max_pool(nan_mode: NanMode, float: Storage, workgroup: u32, buffers: u32) -> Vec<u32> {
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
    let value = b.load_float(float, array, input, index);
    let acc = b.load(f32_ty, acc_var);
    let folded = b.apply_max(acc, value, nan_mode);
    let next = b.select_f32(ok, folded, acc);
    b.store(acc_var, next);
    b.end_loop(cols, kw_var, one);
    b.end_loop(rows, kh_var, one);
    let acc = b.load(f32_ty, acc_var);
    b.store_float(float, array, output, o, acc);
    b.end_loop(scope, counter, stride);
    b.end_main();
    b.finish([workgroup, 1, 1])
}

/// Elementwise float conversion over `count` elements: load at `input` storage as binary32,
/// store at `output` storage. Every narrowing is crate-owned integer code, so a `CAST` produces
/// the same bits on every device. Sub-word outputs are written a whole word per invocation
/// ([`assemble_contiguous_lanes`]).
///
/// Specialization order: input, output `(buffer, base)`; `count`.
fn assemble_cast(
    input: Storage,
    output_storage: Storage,
    workgroup: u32,
    buffers: u32,
) -> Vec<u32> {
    assemble_contiguous_lanes(input, output_storage, workgroup, buffers, |b, bits| {
        let value = b.widen_bits(input, bits);
        b.narrow_bits(output_storage, value)
    })
}

/// A contiguous lane kernel: `out[i] = convert(in[i])` over `count` elements, where `convert`
/// maps one element's raw source bits to its raw destination bits.
///
/// Word-storage outputs run one element per invocation. Sub-word outputs (`Half`, `Byte`, FP8
/// `Quarter`) run one *destination word* per invocation: the invocation loads the source lanes
/// of that word's elements (each source word once), converts them, packs them, and stores the
/// word with a plain `OpStore`. That is the difference between a copy and a read-modify-write:
/// the per-element path clears and sets every lane through two device-scope atomics on a word
/// three other invocations are also atomically updating, which on Intel Xe3 ran FP8 `IDENTITY`
/// at 6 GB/s against 109 GB/s for FP32. Only a tensor's final partial word — whose remaining
/// bytes are not the tensor's to write — still goes through the neighbour-safe atomic sequence,
/// lane by lane, so the documented guarantee holds unchanged.
///
/// Specialization order: input, output `(buffer, base)`; `count` (elements).
fn assemble_contiguous_lanes(
    input: Storage,
    output: Storage,
    workgroup: u32,
    buffers: u32,
    convert: impl Fn(&mut Builder, Id) -> Id,
) -> Vec<u32> {
    let mut b = Builder::new();
    let array = b.buffer_array(buffers);
    let source = b.spec_operand();
    let destination = b.spec_operand();
    let count = b.spec_u32(1);
    let (counter, stride) = b.grid_stride(workgroup);
    let lanes = output.lanes();
    if lanes == 1 {
        let (scope, i) = b.begin_loop(counter, count);
        let bits = b.load_lane_group(input, array, source, i, 1)[0];
        let word = convert(&mut b, bits);
        b.store_word(array, destination, i, word);
        b.end_loop(scope, counter, stride);
    } else {
        let lanes_c = b.c_u32(lanes);
        let lanes_less_one = b.c_u32(lanes - 1);
        let words = b.iadd(count, lanes_less_one);
        let words = b.udiv(words, lanes_c);
        let (scope, w) = b.begin_loop(counter, words);
        let base = b.imul(w, lanes_c);
        let sources = b.load_lane_group(input, array, source, base, lanes);
        let bits: Vec<Id> = sources
            .iter()
            .map(|source_bits| convert(&mut b, *source_bits))
            .collect();
        let packed = b.pack_lanes(&bits);
        let end = b.iadd(base, lanes_c);
        let full = b.uge(count, end);
        b.if_then(full, |b| b.store_word(array, destination, w, packed));
        let partial = b.lnot(full);
        b.if_then(partial, |b| {
            for (lane, bits) in bits.iter().enumerate() {
                let lane = b.c_u32(lane as u32);
                let element = b.iadd(base, lane);
                let in_range = b.ult(element, count);
                b.if_then(in_range, |b| {
                    b.store_lane_bits(output, array, destination, element, *bits);
                });
            }
        });
        b.end_loop(scope, counter, stride);
    }
    b.end_main();
    b.finish([workgroup, 1, 1])
}

/// Strided copy: `out[out_offset + Σ c_d · out_stride_d] = in[in_offset + Σ c_d · in_stride_d]`
/// over the iteration space `dims`; `contiguous` degenerates to `out[i] = in[i]`, which runs
/// through [`assemble_contiguous_lanes`] — a whole-word copy for sub-word storage, with `BOOL`
/// canonicalized to `0`/`1` lane by lane and FP8/FP16 bit patterns moved untouched.
///
/// Specialization order: input, output `(buffer, base)`; `count`; then (strided only)
/// `dims[MAX_RANK]`, `in_strides[MAX_RANK]`, `in_offset`, `out_strides[MAX_RANK]`, `out_offset`.
fn assemble_move(storage: Storage, contiguous: bool, workgroup: u32, buffers: u32) -> Vec<u32> {
    if contiguous {
        return assemble_contiguous_lanes(storage, storage, workgroup, buffers, |b, bits| {
            match storage {
                // Canonical `0`/`1`: any nonzero byte is true.
                Storage::Byte => {
                    let zero = b.c_u32(0);
                    let one = b.c_u32(1);
                    let set = b.ine(bits, zero);
                    b.select_u32(set, one, zero)
                }
                // Raw lanes: an FP8 or binary16 pattern — NaN payload, subnormal, signed
                // zero — moves exactly. The `Byte` arm above must not be reused for these.
                Storage::Word | Storage::Half | Storage::Quarter(_) => bits,
            }
        });
    }
    let mut b = Builder::new();
    let array = b.buffer_array(buffers);
    let input = b.spec_operand();
    let output = b.spec_operand();
    let count = b.spec_u32(1);
    let dims = b.spec_dims();
    let in_strides = b.spec_strides();
    let in_offset = b.spec_u32(0);
    let out_strides = b.spec_strides();
    let out_offset = b.spec_u32(0);

    let (counter, stride) = b.grid_stride(workgroup);
    let (scope, i) = b.begin_loop(counter, count);
    let indices = b.strided_indices(i, &dims, &[in_strides, out_strides]);
    let source = b.iadd(indices[0], in_offset);
    let destination = b.iadd(indices[1], out_offset);
    match storage {
        Storage::Word => {
            let word = b.load_word(array, input, source);
            b.store_word(array, output, destination, word);
        }
        Storage::Byte => {
            let value = b.load_bool(array, input, source);
            b.store_bool(array, output, destination, value);
        }
        // A raw 8-bit lane copy: FP8 bit patterns — NaNs, subnormals, signed zeros — move
        // exactly. The `Byte` arm above must not be reused: it canonicalizes to `0`/`1`.
        Storage::Quarter(_) => {
            let bits = b.load_byte_bits(array, input, source);
            b.store_byte_bits(array, output, destination, bits);
        }
        // A raw 16-bit lane copy: no float conversion, so every binary16 bit pattern — NaN
        // payloads and subnormals included — moves exactly.
        Storage::Half => {
            let bits = b.load_half_bits(array, input, source);
            b.store_half_bits(array, output, destination, bits);
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
        KernelKey::every_variant()
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
        let import = opcodes
            .iter()
            .position(|op| *op == OP_EXT_INST_IMPORT)
            .unwrap();
        let memory = opcodes
            .iter()
            .position(|op| *op == OP_MEMORY_MODEL)
            .unwrap();
        let entry = opcodes.iter().position(|op| *op == OP_ENTRY_POINT).unwrap();
        let mode = opcodes
            .iter()
            .position(|op| *op == OP_EXECUTION_MODE)
            .unwrap();
        assert!(import < memory && memory < entry && entry < mode, "{key:?}");
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
                KernelKey::Matmul { .. } | KernelKey::MatmulStream { .. } => {
                    matmul_spec(operand, operand, operand, 1, 2, 3, 1)
                }
                KernelKey::Nvfp4Matmul { .. } => nvfp4_matmul_spec(
                    operand,
                    &[operand],
                    &[operand],
                    operand,
                    operand,
                    1,
                    2,
                    16,
                    0,
                    0,
                ),
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
                // A CAST's specialization payload is a contiguous move's: two operands and a
                // count.
                KernelKey::Cast { .. } => move_spec(
                    operand,
                    operand,
                    MoveGeometry {
                        count: 8,
                        dims: [1; MAX_RANK],
                        in_strides: [1; MAX_RANK],
                        in_offset: 0,
                        out_strides: [1; MAX_RANK],
                        out_offset: 0,
                    },
                    true,
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
        assert_eq!(matmul_block(16), 64);
        assert_eq!(matmul_block(8), 32);
        assert_eq!(MatmulGeometry::wide(16).shared_bytes(), 8192);
        assert_eq!(stream_matmul_shared_bytes(), 8192);
        assert_eq!(matmul_shared_bytes(8), 8192);
        for tile in [8, 16] {
            let geometry = MatmulGeometry::wide(tile);
            assert_eq!(geometry.invocations(), tile * tile, "{geometry:?}");
            assert_eq!(
                (geometry.block_m() * geometry.depth) % geometry.invocations(),
                0,
                "{geometry:?}"
            );
            assert_eq!(
                (geometry.block_n() * geometry.depth) % geometry.invocations(),
                0,
                "{geometry:?}"
            );
        }
        // Every storage's word count divides the workgroup, and the outputs divide evenly.
        for lanes in [1, 2, 4] {
            let words = STREAM_COLUMNS / lanes;
            assert_eq!(STREAM_WORKGROUP % words, 0);
            assert_eq!((STREAM_ROWS * STREAM_COLUMNS) % STREAM_WORKGROUP, 0);
        }
        assert_eq!(stream_matmul_workgroups(4096, 2), [256, 1, 2]);
        assert_eq!(stream_matmul_workgroups(17, 1), [2, 1, 1]);
        assert_eq!(matmul_workgroups(1, 1, 1, 16), [1, 1, 1]);
        assert_eq!(matmul_workgroups(64, 64, 1, 16), [1, 1, 1]);
        assert_eq!(matmul_workgroups(65, 64, 1, 16), [1, 2, 1]);
        assert_eq!(matmul_workgroups(64, 65, 1, 16), [2, 1, 1]);
        assert_eq!(matmul_workgroups(32, 32, 3, 8), [1, 1, 3]);
        assert_eq!(matmul_workgroups(33, 8, 3, 8), [1, 2, 3]);
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

    /// Independent binary32 → binary16 reference: decompose the (exact) binary64 value into its
    /// integer significand and round to the binary16 significand with round-to-nearest-even —
    /// deliberately a different formulation than [`f32_to_f16_bits`].
    fn reference_f32_to_f16(value: f32) -> u16 {
        let value = f64::from(value);
        if value.is_nan() {
            return if value.is_sign_negative() {
                0xfe00
            } else {
                0x7e00
            };
        }
        let sign: u16 = if value.is_sign_negative() { 0x8000 } else { 0 };
        let magnitude = value.abs();
        if magnitude >= 65520.0 {
            return sign | 0x7c00;
        }
        if magnitude == 0.0 {
            return sign;
        }
        let bits = magnitude.to_bits();
        let biased = ((bits >> 52) & 0x7ff) as i32;
        // `f32` inputs are never binary64-subnormal, so the implicit leading bit is present.
        let mantissa = (bits & ((1_u64 << 52) - 1)) | (1_u64 << 52);
        let exponent = biased - 1023;
        // value = mantissa · 2^(exponent - 52)
        if exponent >= -14 {
            // Binary16 normal: keep the top 11 significand bits, round the rest.
            let kept = mantissa >> 42;
            let dropped = mantissa & ((1_u64 << 42) - 1);
            let half = 1_u64 << 41;
            let rounded = if dropped > half || (dropped == half && kept & 1 == 1) {
                kept + 1
            } else {
                kept
            };
            let (exponent, mantissa) = if rounded == 2048 {
                (exponent + 1, 1024)
            } else {
                (exponent, rounded)
            };
            let biased = exponent + 15;
            if biased >= 31 {
                return sign | 0x7c00;
            }
            return sign | ((biased as u16) << 10) | (mantissa - 1024) as u16;
        }
        // Binary16 subnormal (step 2^-24): scale into the integer domain and round.
        let drop = (28 - exponent) as u32;
        if drop >= 64 {
            return sign;
        }
        let kept = mantissa >> drop;
        let dropped = mantissa & ((1_u64 << drop) - 1);
        let half = 1_u64 << (drop - 1);
        let rounded = if dropped > half || (dropped == half && kept & 1 == 1) {
            kept + 1
        } else {
            kept
        };
        // At most 1024: the smallest normal encoding, correctly reached by the rounding carry.
        sign | rounded as u16
    }

    #[test]
    fn binary16_widening_is_exact_for_every_pattern() {
        for pattern in 0_u32..=0xffff {
            let bits = pattern as u16;
            let value = f16_to_f32(bits);
            let exponent = (bits >> 10) & 0x1f;
            let mantissa = bits & 0x3ff;
            match (exponent, mantissa) {
                (0, 0) => assert_eq!(value.to_bits(), (u32::from(bits) & 0x8000) << 16),
                (31, 0) => assert!(value.is_infinite(), "{bits:#06x}"),
                (31, _) => assert!(value.is_nan(), "{bits:#06x}"),
                _ => {
                    // Round trip: every binary16 value is exactly binary32-representable.
                    assert_eq!(f32_to_f16_bits(value), bits, "{bits:#06x}");
                }
            }
        }
    }

    #[test]
    fn binary16_narrowing_matches_the_reference_everywhere() {
        // Every binary16 pattern (round trip), the rounding boundaries between them, and a
        // pseudo-random sweep across the whole binary32 range.
        let mut cases: Vec<f32> = (0_u32..=0xffff)
            .map(|pattern| f16_to_f32(pattern as u16))
            .collect();
        for pattern in 0_u32..=0xfffe {
            let lo = f16_to_f32(pattern as u16);
            let hi = f16_to_f32((pattern + 1) as u16);
            if lo.is_finite() && hi.is_finite() {
                cases.push(f32::from_bits(
                    lo.to_bits() + hi.to_bits().abs_diff(lo.to_bits()) / 2,
                ));
            }
        }
        let mut state = 0x243f_6a88_85a3_08d3_u64;
        for _ in 0..1_000_000 {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            cases.push(f32::from_bits((state >> 32) as u32));
        }
        for case in cases {
            assert_eq!(
                f32_to_f16_bits(case),
                reference_f32_to_f16(case),
                "{case:e} ({:#010x})",
                case.to_bits()
            );
        }
    }
}

#[cfg(test)]
mod fp8_narrowing_tests {
    use super::*;
    use virtio_accel_tosa::{fp8e4m3_to_f32, fp8e5m2_to_f32};

    /// An independent oracle: scan every non-negative encoding for the nearest magnitude, ties
    /// to even, then apply the sign separately as IEEE does — so a value that rounds to zero
    /// keeps its sign. No bit arithmetic in common with [`f32_to_fp8_bits`].
    fn nearest_by_search(format: Fp8Format, value: f32) -> u8 {
        let sign = if value.is_sign_negative() { 0x80 } else { 0x00 };
        let magnitude = value.abs();
        let mut best: Option<(f32, u8)> = None;
        for bits in 0..=0x7f_u8 {
            let candidate = fp8_decode(format, bits);
            if !candidate.is_finite() {
                continue;
            }
            let distance = (candidate - magnitude).abs();
            best = match best {
                None => Some((distance, bits)),
                Some((best_distance, _)) if distance < best_distance => Some((distance, bits)),
                Some((best_distance, best_bits))
                    if distance == best_distance && bits & 1 == 0 && best_bits & 1 == 1 =>
                {
                    Some((distance, bits))
                }
                other => other,
            };
        }
        sign | best.expect("a finite encoding exists").1
    }

    #[test]
    fn narrowing_round_trips_every_encoding_exactly() {
        for format in [Fp8Format::E4M3, Fp8Format::E5M2] {
            for bits in 0..=u8::MAX {
                let value = match format {
                    Fp8Format::E4M3 => fp8e4m3_to_f32(bits),
                    Fp8Format::E5M2 => fp8e5m2_to_f32(bits),
                };
                if !value.is_finite() {
                    continue;
                }
                assert_eq!(
                    f32_to_fp8_bits(format, value),
                    bits,
                    "{format:?}: {value} did not return to {bits:#04x}"
                );
            }
        }
    }

    #[test]
    fn narrowing_matches_an_independent_nearest_search() {
        for format in [Fp8Format::E4M3, Fp8Format::E5M2] {
            let max_finite = match format {
                Fp8Format::E4M3 => 448.0_f32,
                Fp8Format::E5M2 => 57344.0_f32,
            };
            let mut state = 0x1234_5678_u32;
            for index in 0..200_000 {
                // A mix of exact midpoints, subnormal-range values, and pseudo-random draws.
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let value = if index % 3 == 0 {
                    let a = fp8_decode(format, (index % 256) as u8);
                    let b = fp8_decode(format, ((index + 1) % 256) as u8);
                    (a + b) * 0.5
                } else {
                    let scaled = (state >> 8) as f32 / (1_u32 << 24) as f32;
                    (scaled * 2.0 - 1.0) * max_finite * 1.2
                };
                if !value.is_finite() {
                    continue;
                }
                let actual = f32_to_fp8_bits(format, value);
                if value.abs() > max_finite {
                    continue; // overflow is this crate's policy, not a nearest-value question
                }
                assert_eq!(
                    actual,
                    nearest_by_search(format, value),
                    "{format:?}: {value}"
                );
            }
        }
    }

    fn fp8_decode(format: Fp8Format, bits: u8) -> f32 {
        match format {
            Fp8Format::E4M3 => fp8e4m3_to_f32(bits),
            Fp8Format::E5M2 => fp8e5m2_to_f32(bits),
        }
    }

    #[test]
    fn overflow_policy_is_nan_for_e4m3_and_infinity_for_e5m2() {
        // 464 is the midpoint above 448 and ties to even, so it stays finite; anything beyond
        // it is unrepresentable.
        assert_eq!(f32_to_fp8_bits(Fp8Format::E4M3, 464.0), 0x7e);
        assert_eq!(f32_to_fp8_bits(Fp8Format::E4M3, 464.001), 0x7f);
        assert_eq!(f32_to_fp8_bits(Fp8Format::E4M3, -464.001), 0xff);
        assert_eq!(f32_to_fp8_bits(Fp8Format::E4M3, f32::INFINITY), 0x7f);
        assert_eq!(f32_to_fp8_bits(Fp8Format::E4M3, f32::NAN), 0x7f);
        assert_eq!(f32_to_fp8_bits(Fp8Format::E5M2, 1.0e9), 0x7c);
        assert_eq!(f32_to_fp8_bits(Fp8Format::E5M2, f32::NEG_INFINITY), 0xfc);
        assert_eq!(f32_to_fp8_bits(Fp8Format::E5M2, f32::NAN), 0x7e);
        // Signed zero survives.
        assert_eq!(f32_to_fp8_bits(Fp8Format::E4M3, -0.0), 0x80);
        assert_eq!(f32_to_fp8_bits(Fp8Format::E5M2, 0.0), 0x00);
    }
}
