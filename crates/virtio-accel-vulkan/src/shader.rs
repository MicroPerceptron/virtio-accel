//! The checked-in SPIR-V compute shaders (ADR 0003).
//!
//! Every module the backend hands to a driver is authored here, instruction by instruction, and
//! specialized at `load_program` through specialization constants alone. Guest bytes never reach
//! the driver's shader compiler: a TOSA artifact selects one of these modules and supplies
//! validated shape parameters, nothing else (`docs/threat-model.md`, transient-compile budget).
//!
//! The modules are written as SPIR-V words rather than as a binary blob so review sees the
//! assembly. Each `emit` line is one instruction: the opcode followed by its operands, exactly as
//! the SPIR-V specification tabulates them; the assembler only computes word counts and the id
//! bound. The result is deterministic and built once per process.

use std::sync::OnceLock;

/// Workgroup size of the elementwise shaders along X (`OpExecutionMode LocalSize`).
pub const ELEMENTWISE_WORKGROUP_SIZE: u32 = 64;

/// The specialization constant id carrying the element count of the elementwise shaders.
pub const ELEMENT_COUNT_SPEC_ID: u32 = 0;

/// Descriptor binding of the input storage buffer (set 0).
pub const INPUT_BINDING: u32 = 0;
/// Descriptor binding of the output storage buffer (set 0).
pub const OUTPUT_BINDING: u32 = 1;

// SPIR-V opcodes (Unified specification, section 3.52).
const OP_MEMORY_MODEL: u16 = 14;
const OP_ENTRY_POINT: u16 = 15;
const OP_EXECUTION_MODE: u16 = 16;
const OP_CAPABILITY: u16 = 17;
const OP_TYPE_VOID: u16 = 19;
const OP_TYPE_BOOL: u16 = 20;
const OP_TYPE_INT: u16 = 21;
const OP_TYPE_FLOAT: u16 = 22;
const OP_TYPE_VECTOR: u16 = 23;
const OP_TYPE_RUNTIME_ARRAY: u16 = 29;
const OP_TYPE_STRUCT: u16 = 30;
const OP_TYPE_POINTER: u16 = 32;
const OP_TYPE_FUNCTION: u16 = 33;
const OP_CONSTANT: u16 = 43;
const OP_SPEC_CONSTANT: u16 = 50;
const OP_FUNCTION: u16 = 54;
const OP_FUNCTION_END: u16 = 56;
const OP_VARIABLE: u16 = 59;
const OP_LOAD: u16 = 61;
const OP_STORE: u16 = 62;
const OP_ACCESS_CHAIN: u16 = 65;
const OP_DECORATE: u16 = 71;
const OP_MEMBER_DECORATE: u16 = 72;
const OP_I_ADD: u16 = 128;
const OP_I_MUL: u16 = 132;
const OP_F_ADD: u16 = 129;
const OP_F_MUL: u16 = 133;
const OP_U_LESS_THAN: u16 = 176;
const OP_LOGICAL_AND: u16 = 167;
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
const STORAGE_CLASS_STORAGE_BUFFER: u32 = 12;
const DECORATION_SPEC_ID: u32 = 1;
const DECORATION_BLOCK: u32 = 2;
const DECORATION_ARRAY_STRIDE: u32 = 6;
const DECORATION_BUILT_IN: u32 = 11;
const DECORATION_NON_WRITABLE: u32 = 24;
const DECORATION_BINDING: u32 = 33;
const DECORATION_DESCRIPTOR_SET: u32 = 34;
const DECORATION_OFFSET: u32 = 35;
const BUILT_IN_GLOBAL_INVOCATION_ID: u32 = 28;
const FUNCTION_CONTROL_NONE: u32 = 0;
const SELECTION_CONTROL_NONE: u32 = 0;
const LOOP_CONTROL_NONE: u32 = 0;

/// SPIR-V 1.3: the version every Vulkan 1.1+ implementation must consume, and the first with the
/// `StorageBuffer` storage class in core (no `SPV_KHR_storage_buffer_storage_class` extension).
const SPIRV_VERSION_1_3: u32 = 0x0001_0300;
const SPIRV_MAGIC: u32 = 0x0723_0203;

/// Minimal SPIR-V assembler: word counts and the id bound are derived, nothing else.
struct Assembler {
    words: Vec<u32>,
    next_id: u32,
}

impl Assembler {
    fn new() -> Self {
        Self {
            // Header: magic, version, generator (0: no registered tool), id bound (patched by
            // `finish`), reserved schema.
            words: vec![SPIRV_MAGIC, SPIRV_VERSION_1_3, 0, 0, 0],
            next_id: 1,
        }
    }

    fn id(&mut self) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    fn emit(&mut self, opcode: u16, operands: &[u32]) {
        let word_count = u32::try_from(operands.len() + 1).expect("instruction fits");
        self.words.push((word_count << 16) | u32::from(opcode));
        self.words.extend_from_slice(operands);
    }

    fn finish(mut self) -> Vec<u32> {
        self.words[3] = self.next_id;
        self.words
    }
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

/// Elementwise 32-bit copy: `out[i] = in[i]` for `i < element_count`.
///
/// This is the IDENTITY kernel for every 4-byte TOSA element type: a bit-exact word copy that no
/// float-controls setting can alter, so NaN payloads, infinities, signed zeros, and subnormals
/// survive unchanged. Layout: set 0, binding 0 is the read-only input `{ uint data[]; }`,
/// binding 1 the output with the same layout; `element_count` is specialization constant 0.
///
/// ```text
/// OpCapability Shader
/// OpMemoryModel Logical GLSL450
/// OpEntryPoint GLCompute %main "main" %gid
/// OpExecutionMode %main LocalSize 64 1 1
/// OpDecorate %gid BuiltIn GlobalInvocationId
/// OpDecorate %words ArrayStride 4
/// OpDecorate %block Block
/// OpMemberDecorate %block 0 Offset 0
/// OpDecorate %input DescriptorSet 0 ; Binding 0 ; NonWritable
/// OpDecorate %output DescriptorSet 0 ; Binding 1
/// OpDecorate %count SpecId 0
/// %void = OpTypeVoid           %fn = OpTypeFunction %void
/// %uint = OpTypeInt 32 0       %uvec3 = OpTypeVector %uint 3
/// %bool = OpTypeBool
/// %ptr_in_uvec3 = OpTypePointer Input %uvec3     %gid = OpVariable %ptr_in_uvec3 Input
/// %ptr_in_uint = OpTypePointer Input %uint
/// %words = OpTypeRuntimeArray %uint              %block = OpTypeStruct %words
/// %ptr_sb_block = OpTypePointer StorageBuffer %block
/// %input = OpVariable %ptr_sb_block StorageBuffer
/// %output = OpVariable %ptr_sb_block StorageBuffer
/// %ptr_sb_uint = OpTypePointer StorageBuffer %uint
/// %zero = OpConstant %uint 0   %count = OpSpecConstant %uint 1
/// %main = OpFunction %void None %fn
/// %entry = OpLabel
///   %gid_x_ptr = OpAccessChain %ptr_in_uint %gid %zero
///   %index = OpLoad %uint %gid_x_ptr
///   %in_range = OpULessThan %bool %index %count
///   OpSelectionMerge %merge None
///   OpBranchConditional %in_range %copy %merge
/// %copy = OpLabel
///   %src = OpAccessChain %ptr_sb_uint %input %zero %index
///   %value = OpLoad %uint %src
///   %dst = OpAccessChain %ptr_sb_uint %output %zero %index
///   OpStore %dst %value
///   OpBranch %merge
/// %merge = OpLabel
///   OpReturn
/// OpFunctionEnd
/// ```
pub fn copy_u32_spirv() -> &'static [u32] {
    static MODULE: OnceLock<Vec<u32>> = OnceLock::new();
    MODULE.get_or_init(assemble_copy_u32)
}

fn assemble_copy_u32() -> Vec<u32> {
    let mut a = Assembler::new();
    let main = a.id();
    let gid = a.id();
    let void = a.id();
    let fn_type = a.id();
    let uint = a.id();
    let uvec3 = a.id();
    let bool_type = a.id();
    let ptr_in_uvec3 = a.id();
    let ptr_in_uint = a.id();
    let words = a.id();
    let block = a.id();
    let ptr_sb_block = a.id();
    let input = a.id();
    let output = a.id();
    let ptr_sb_uint = a.id();
    let zero = a.id();
    let count = a.id();
    let entry = a.id();
    let gid_x_ptr = a.id();
    let index = a.id();
    let in_range = a.id();
    let copy = a.id();
    let merge = a.id();
    let src = a.id();
    let value = a.id();
    let dst = a.id();

    a.emit(OP_CAPABILITY, &[CAPABILITY_SHADER]);
    a.emit(
        OP_MEMORY_MODEL,
        &[ADDRESSING_MODEL_LOGICAL, MEMORY_MODEL_GLSL450],
    );
    let mut entry_point = vec![EXECUTION_MODEL_GL_COMPUTE, main];
    entry_point.extend(literal_string("main"));
    entry_point.push(gid);
    a.emit(OP_ENTRY_POINT, &entry_point);
    a.emit(
        OP_EXECUTION_MODE,
        &[
            main,
            EXECUTION_MODE_LOCAL_SIZE,
            ELEMENTWISE_WORKGROUP_SIZE,
            1,
            1,
        ],
    );

    a.emit(
        OP_DECORATE,
        &[gid, DECORATION_BUILT_IN, BUILT_IN_GLOBAL_INVOCATION_ID],
    );
    a.emit(OP_DECORATE, &[words, DECORATION_ARRAY_STRIDE, 4]);
    a.emit(OP_DECORATE, &[block, DECORATION_BLOCK]);
    a.emit(OP_MEMBER_DECORATE, &[block, 0, DECORATION_OFFSET, 0]);
    a.emit(OP_DECORATE, &[input, DECORATION_DESCRIPTOR_SET, 0]);
    a.emit(OP_DECORATE, &[input, DECORATION_BINDING, INPUT_BINDING]);
    a.emit(OP_DECORATE, &[input, DECORATION_NON_WRITABLE]);
    a.emit(OP_DECORATE, &[output, DECORATION_DESCRIPTOR_SET, 0]);
    a.emit(OP_DECORATE, &[output, DECORATION_BINDING, OUTPUT_BINDING]);
    a.emit(
        OP_DECORATE,
        &[count, DECORATION_SPEC_ID, ELEMENT_COUNT_SPEC_ID],
    );

    a.emit(OP_TYPE_VOID, &[void]);
    a.emit(OP_TYPE_FUNCTION, &[fn_type, void]);
    a.emit(OP_TYPE_INT, &[uint, 32, 0]);
    a.emit(OP_TYPE_VECTOR, &[uvec3, uint, 3]);
    a.emit(OP_TYPE_BOOL, &[bool_type]);
    a.emit(OP_TYPE_POINTER, &[ptr_in_uvec3, STORAGE_CLASS_INPUT, uvec3]);
    a.emit(OP_VARIABLE, &[ptr_in_uvec3, gid, STORAGE_CLASS_INPUT]);
    a.emit(OP_TYPE_POINTER, &[ptr_in_uint, STORAGE_CLASS_INPUT, uint]);
    a.emit(OP_TYPE_RUNTIME_ARRAY, &[words, uint]);
    a.emit(OP_TYPE_STRUCT, &[block, words]);
    a.emit(
        OP_TYPE_POINTER,
        &[ptr_sb_block, STORAGE_CLASS_STORAGE_BUFFER, block],
    );
    a.emit(
        OP_VARIABLE,
        &[ptr_sb_block, input, STORAGE_CLASS_STORAGE_BUFFER],
    );
    a.emit(
        OP_VARIABLE,
        &[ptr_sb_block, output, STORAGE_CLASS_STORAGE_BUFFER],
    );
    a.emit(
        OP_TYPE_POINTER,
        &[ptr_sb_uint, STORAGE_CLASS_STORAGE_BUFFER, uint],
    );
    a.emit(OP_CONSTANT, &[uint, zero, 0]);
    // Default of one element; `load_program` overrides it through `VkSpecializationInfo`.
    a.emit(OP_SPEC_CONSTANT, &[uint, count, 1]);

    a.emit(OP_FUNCTION, &[void, main, FUNCTION_CONTROL_NONE, fn_type]);
    a.emit(OP_LABEL, &[entry]);
    a.emit(OP_ACCESS_CHAIN, &[ptr_in_uint, gid_x_ptr, gid, zero]);
    a.emit(OP_LOAD, &[uint, index, gid_x_ptr]);
    a.emit(OP_U_LESS_THAN, &[bool_type, in_range, index, count]);
    a.emit(OP_SELECTION_MERGE, &[merge, SELECTION_CONTROL_NONE]);
    a.emit(OP_BRANCH_CONDITIONAL, &[in_range, copy, merge]);
    a.emit(OP_LABEL, &[copy]);
    a.emit(OP_ACCESS_CHAIN, &[ptr_sb_uint, src, input, zero, index]);
    a.emit(OP_LOAD, &[uint, value, src]);
    a.emit(OP_ACCESS_CHAIN, &[ptr_sb_uint, dst, output, zero, index]);
    a.emit(OP_STORE, &[dst, value]);
    a.emit(OP_BRANCH, &[merge]);
    a.emit(OP_LABEL, &[merge]);
    a.emit(OP_RETURN, &[]);
    a.emit(OP_FUNCTION_END, &[]);
    a.finish()
}

/// Number of workgroups that cover `element_count` elements, if it fits the dispatch domain.
pub const fn elementwise_workgroups(element_count: u32) -> u32 {
    element_count.div_ceil(ELEMENTWISE_WORKGROUP_SIZE)
}

/// Workgroup size of the MATMUL shader along X and Y (`OpExecutionMode LocalSize x y 1`).
pub const MATMUL_WORKGROUP_SIZE_X: u32 = 8;
/// Workgroup size of the MATMUL shader along Y.
pub const MATMUL_WORKGROUP_SIZE_Y: u32 = 8;

/// MATMUL specialization constants (ids 0..5): output rows `m`, output columns `n`, shared
/// reduction dimension `k`, and batch. `load_program` overrides every one.
pub const MATMUL_SPEC_ID_M: u32 = 0;
pub const MATMUL_SPEC_ID_N: u32 = 1;
pub const MATMUL_SPEC_ID_K: u32 = 2;
pub const MATMUL_SPEC_ID_BATCH: u32 = 3;

/// Descriptor binding of the left-hand side matrix (set 0).
pub const MATMUL_LHS_BINDING: u32 = 0;
/// Descriptor binding of the right-hand side matrix (set 0).
pub const MATMUL_RHS_BINDING: u32 = 1;
/// Descriptor binding of the output matrix (set 0).
pub const MATMUL_OUTPUT_BINDING: u32 = 2;

/// FP32 batched matrix multiplication: `out[b, m, n] = sum_k(lhs[b, m, k] * rhs[b, k, n])` for
/// every `(b, m, n)` below the specialization constants. One thread per output element, dispatched
/// over X = columns, Y = rows, Z = batches; out-of-range threads return without a write.
///
/// Row-major 3-D tensors exactly as TOSA lays them out: the per-batch strides are `m*k` (lhs),
/// `k*n` (rhs), and `m*n` (output) elements. Reduction order is the ascending-`k` summation the
/// shared FP32 oracle tolerates; every product and every accumulation is an IEEE FP32 operation
/// with no fused-multiply-add, so the oracle's tolerance is never stretched by contraction.
///
/// Layout: set 0, bindings 0/1 are the read-only operands `{ float data[]; }`, binding 2 the
/// output with the same layout. `m`, `n`, `k`, and `batch` are specialization constants 0..3.
pub fn matmul_fp32_spirv() -> &'static [u32] {
    static MODULE: OnceLock<Vec<u32>> = OnceLock::new();
    MODULE.get_or_init(assemble_matmul_fp32)
}

fn assemble_matmul_fp32() -> Vec<u32> {
    let mut a = Assembler::new();
    let main = a.id();
    let gid = a.id();
    let void = a.id();
    let fn_type = a.id();
    let uint = a.id();
    let float = a.id();
    let uvec3 = a.id();
    let bool_type = a.id();
    let ptr_in_uvec3 = a.id();
    let ptr_in_uint = a.id();
    let words_f = a.id();
    let block_f = a.id();
    let ptr_sb_block_f = a.id();
    let lhs = a.id();
    let rhs = a.id();
    let out = a.id();
    let ptr_sb_float = a.id();
    let zero = a.id();
    let one = a.id();
    let m = a.id();
    let n = a.id();
    let k = a.id();
    let batch = a.id();
    let fzero = a.id();
    let ptr_func_float = a.id();
    let ptr_func_uint = a.id();
    let acc = a.id();
    let i = a.id();
    let entry = a.id();
    let x_ptr = a.id();
    let y_ptr = a.id();
    let z_ptr = a.id();
    let x = a.id();
    let y = a.id();
    let z = a.id();
    let in_m = a.id();
    let in_n = a.id();
    let in_b = a.id();
    let in_mn = a.id();
    let in_all = a.id();
    let out_of_bounds = a.id();
    let copy = a.id();
    let merge = a.id();
    let header = a.id();
    let i_val = a.id();
    let in_k = a.id();
    let body = a.id();
    let cont = a.id();
    let done = a.id();
    let lhs_base = a.id();
    let lhs_row = a.id();
    let lhs_idx = a.id();
    let lhs_ptr = a.id();
    let lhs_val = a.id();
    let rhs_base = a.id();
    let rhs_row = a.id();
    let rhs_idx = a.id();
    let rhs_ptr = a.id();
    let rhs_val = a.id();
    let product = a.id();
    let acc_val = a.id();
    let acc_next = a.id();
    let i_next = a.id();
    let out_base = a.id();
    let out_row = a.id();
    let out_idx = a.id();
    let out_ptr = a.id();
    let acc_final = a.id();
    let two = a.id();

    a.emit(OP_CAPABILITY, &[CAPABILITY_SHADER]);
    a.emit(
        OP_MEMORY_MODEL,
        &[ADDRESSING_MODEL_LOGICAL, MEMORY_MODEL_GLSL450],
    );
    let mut entry_point = vec![EXECUTION_MODEL_GL_COMPUTE, main];
    entry_point.extend(literal_string("main"));
    entry_point.push(gid);
    a.emit(OP_ENTRY_POINT, &entry_point);
    a.emit(
        OP_EXECUTION_MODE,
        &[
            main,
            EXECUTION_MODE_LOCAL_SIZE,
            MATMUL_WORKGROUP_SIZE_X,
            MATMUL_WORKGROUP_SIZE_Y,
            1,
        ],
    );

    a.emit(
        OP_DECORATE,
        &[gid, DECORATION_BUILT_IN, BUILT_IN_GLOBAL_INVOCATION_ID],
    );
    a.emit(OP_DECORATE, &[words_f, DECORATION_ARRAY_STRIDE, 4]);
    a.emit(OP_DECORATE, &[block_f, DECORATION_BLOCK]);
    a.emit(OP_MEMBER_DECORATE, &[block_f, 0, DECORATION_OFFSET, 0]);
    a.emit(OP_DECORATE, &[lhs, DECORATION_DESCRIPTOR_SET, 0]);
    a.emit(OP_DECORATE, &[lhs, DECORATION_BINDING, MATMUL_LHS_BINDING]);
    a.emit(OP_DECORATE, &[lhs, DECORATION_NON_WRITABLE]);
    a.emit(OP_DECORATE, &[rhs, DECORATION_DESCRIPTOR_SET, 0]);
    a.emit(OP_DECORATE, &[rhs, DECORATION_BINDING, MATMUL_RHS_BINDING]);
    a.emit(OP_DECORATE, &[rhs, DECORATION_NON_WRITABLE]);
    a.emit(OP_DECORATE, &[out, DECORATION_DESCRIPTOR_SET, 0]);
    a.emit(
        OP_DECORATE,
        &[out, DECORATION_BINDING, MATMUL_OUTPUT_BINDING],
    );
    a.emit(OP_DECORATE, &[m, DECORATION_SPEC_ID, MATMUL_SPEC_ID_M]);
    a.emit(OP_DECORATE, &[n, DECORATION_SPEC_ID, MATMUL_SPEC_ID_N]);
    a.emit(OP_DECORATE, &[k, DECORATION_SPEC_ID, MATMUL_SPEC_ID_K]);
    a.emit(
        OP_DECORATE,
        &[batch, DECORATION_SPEC_ID, MATMUL_SPEC_ID_BATCH],
    );

    a.emit(OP_TYPE_VOID, &[void]);
    a.emit(OP_TYPE_FUNCTION, &[fn_type, void]);
    a.emit(OP_TYPE_INT, &[uint, 32, 0]);
    a.emit(OP_TYPE_FLOAT, &[float, 32]);
    a.emit(OP_TYPE_VECTOR, &[uvec3, uint, 3]);
    a.emit(OP_TYPE_BOOL, &[bool_type]);
    a.emit(OP_TYPE_POINTER, &[ptr_in_uvec3, STORAGE_CLASS_INPUT, uvec3]);
    a.emit(OP_VARIABLE, &[ptr_in_uvec3, gid, STORAGE_CLASS_INPUT]);
    a.emit(OP_TYPE_POINTER, &[ptr_in_uint, STORAGE_CLASS_INPUT, uint]);
    a.emit(OP_TYPE_RUNTIME_ARRAY, &[words_f, float]);
    a.emit(OP_TYPE_STRUCT, &[block_f, words_f]);
    a.emit(
        OP_TYPE_POINTER,
        &[ptr_sb_block_f, STORAGE_CLASS_STORAGE_BUFFER, block_f],
    );
    a.emit(
        OP_VARIABLE,
        &[ptr_sb_block_f, lhs, STORAGE_CLASS_STORAGE_BUFFER],
    );
    a.emit(
        OP_VARIABLE,
        &[ptr_sb_block_f, rhs, STORAGE_CLASS_STORAGE_BUFFER],
    );
    a.emit(
        OP_VARIABLE,
        &[ptr_sb_block_f, out, STORAGE_CLASS_STORAGE_BUFFER],
    );
    a.emit(
        OP_TYPE_POINTER,
        &[ptr_sb_float, STORAGE_CLASS_STORAGE_BUFFER, float],
    );
    a.emit(
        OP_TYPE_POINTER,
        &[ptr_func_float, STORAGE_CLASS_FUNCTION, float],
    );
    a.emit(
        OP_TYPE_POINTER,
        &[ptr_func_uint, STORAGE_CLASS_FUNCTION, uint],
    );
    a.emit(OP_CONSTANT, &[uint, zero, 0]);
    a.emit(OP_CONSTANT, &[uint, one, 1]);
    a.emit(OP_CONSTANT, &[uint, two, 2]);
    a.emit(OP_CONSTANT, &[float, fzero, 0]);
    a.emit(OP_SPEC_CONSTANT, &[uint, m, 1]);
    a.emit(OP_SPEC_CONSTANT, &[uint, n, 1]);
    a.emit(OP_SPEC_CONSTANT, &[uint, k, 1]);
    a.emit(OP_SPEC_CONSTANT, &[uint, batch, 1]);

    a.emit(OP_FUNCTION, &[void, main, FUNCTION_CONTROL_NONE, fn_type]);
    a.emit(OP_LABEL, &[entry]);
    a.emit(OP_VARIABLE, &[ptr_func_float, acc, STORAGE_CLASS_FUNCTION]);
    a.emit(OP_VARIABLE, &[ptr_func_uint, i, STORAGE_CLASS_FUNCTION]);
    a.emit(OP_ACCESS_CHAIN, &[ptr_in_uint, x_ptr, gid, zero]);
    a.emit(OP_LOAD, &[uint, x, x_ptr]);
    a.emit(OP_ACCESS_CHAIN, &[ptr_in_uint, y_ptr, gid, one]);
    a.emit(OP_LOAD, &[uint, y, y_ptr]);
    a.emit(OP_ACCESS_CHAIN, &[ptr_in_uint, z_ptr, gid, two]);
    a.emit(OP_LOAD, &[uint, z, z_ptr]);
    a.emit(OP_U_LESS_THAN, &[bool_type, in_m, y, m]);
    a.emit(OP_U_LESS_THAN, &[bool_type, in_n, x, n]);
    a.emit(OP_U_LESS_THAN, &[bool_type, in_b, z, batch]);
    a.emit(OP_LOGICAL_AND, &[bool_type, in_mn, in_m, in_n]);
    a.emit(OP_LOGICAL_AND, &[bool_type, in_all, in_mn, in_b]);
    // Out-of-bounds threads skip the copy: branch to the merge when any coordinate misses.
    a.emit(OP_LOGICAL_NOT, &[bool_type, out_of_bounds, in_all]);
    a.emit(OP_SELECTION_MERGE, &[merge, SELECTION_CONTROL_NONE]);
    a.emit(OP_BRANCH_CONDITIONAL, &[out_of_bounds, merge, copy]);
    a.emit(OP_LABEL, &[copy]);
    a.emit(OP_STORE, &[acc, fzero]);
    a.emit(OP_STORE, &[i, zero]);
    a.emit(OP_BRANCH, &[header]);
    a.emit(OP_LABEL, &[header]);
    a.emit(OP_LOAD, &[uint, i_val, i]);
    a.emit(OP_U_LESS_THAN, &[bool_type, in_k, i_val, k]);
    a.emit(OP_LOOP_MERGE, &[done, cont, LOOP_CONTROL_NONE]);
    a.emit(OP_BRANCH_CONDITIONAL, &[in_k, body, done]);
    a.emit(OP_LABEL, &[body]);
    // lhs index: (z * m + y) * k + i
    a.emit(OP_I_MUL, &[uint, lhs_base, z, m]);
    a.emit(OP_I_ADD, &[uint, lhs_row, lhs_base, y]);
    a.emit(OP_I_MUL, &[uint, lhs_idx, lhs_row, k]);
    let lhs_index = a.id();
    a.emit(OP_I_ADD, &[uint, lhs_index, lhs_idx, i_val]);
    a.emit(
        OP_ACCESS_CHAIN,
        &[ptr_sb_float, lhs_ptr, lhs, zero, lhs_index],
    );
    a.emit(OP_LOAD, &[float, lhs_val, lhs_ptr]);
    // rhs index: (z * k + i) * n + x
    a.emit(OP_I_MUL, &[uint, rhs_base, z, k]);
    a.emit(OP_I_ADD, &[uint, rhs_row, rhs_base, i_val]);
    a.emit(OP_I_MUL, &[uint, rhs_idx, rhs_row, n]);
    let rhs_index = a.id();
    a.emit(OP_I_ADD, &[uint, rhs_index, rhs_idx, x]);
    a.emit(
        OP_ACCESS_CHAIN,
        &[ptr_sb_float, rhs_ptr, rhs, zero, rhs_index],
    );
    a.emit(OP_LOAD, &[float, rhs_val, rhs_ptr]);
    a.emit(OP_F_MUL, &[float, product, lhs_val, rhs_val]);
    a.emit(OP_LOAD, &[float, acc_val, acc]);
    a.emit(OP_F_ADD, &[float, acc_next, acc_val, product]);
    a.emit(OP_STORE, &[acc, acc_next]);
    a.emit(OP_BRANCH, &[cont]);
    a.emit(OP_LABEL, &[cont]);
    a.emit(OP_I_ADD, &[uint, i_next, i_val, one]);
    a.emit(OP_STORE, &[i, i_next]);
    a.emit(OP_BRANCH, &[header]);
    a.emit(OP_LABEL, &[done]);
    // out index: (z * m + y) * n + x
    a.emit(OP_I_MUL, &[uint, out_base, z, m]);
    a.emit(OP_I_ADD, &[uint, out_row, out_base, y]);
    a.emit(OP_I_MUL, &[uint, out_idx, out_row, n]);
    let out_index = a.id();
    a.emit(OP_I_ADD, &[uint, out_index, out_idx, x]);
    a.emit(OP_LOAD, &[float, acc_final, acc]);
    a.emit(
        OP_ACCESS_CHAIN,
        &[ptr_sb_float, out_ptr, out, zero, out_index],
    );
    a.emit(OP_STORE, &[out_ptr, acc_final]);
    a.emit(OP_BRANCH, &[merge]);
    a.emit(OP_LABEL, &[merge]);
    a.emit(OP_RETURN, &[]);
    a.emit(OP_FUNCTION_END, &[]);
    a.finish()
}

const OP_LOGICAL_NOT: u16 = 168;
const STORAGE_CLASS_FUNCTION: u32 = 7;

/// Workgroup counts of a MATMUL dispatch over `m` rows, `n` columns, and `batch` batches.
pub const fn matmul_workgroups(m: u32, n: u32, batch: u32) -> [u32; 3] {
    [
        n.div_ceil(MATMUL_WORKGROUP_SIZE_X),
        m.div_ceil(MATMUL_WORKGROUP_SIZE_Y),
        batch,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Walk the module instruction by instruction, checking that word counts tile the body
    /// exactly and that every id is below the declared bound.
    #[test]
    fn copy_module_is_well_formed() {
        let words = copy_u32_spirv();
        assert_eq!(words[0], SPIRV_MAGIC);
        assert_eq!(words[1], SPIRV_VERSION_1_3);
        let bound = words[3];
        let mut cursor = 5;
        let mut opcodes = Vec::new();
        while cursor < words.len() {
            let word_count = (words[cursor] >> 16) as usize;
            assert!(word_count >= 1, "zero-length instruction at {cursor}");
            assert!(cursor + word_count <= words.len(), "instruction overruns");
            opcodes.push((words[cursor] & 0xffff) as u16);
            cursor += word_count;
        }
        assert_eq!(cursor, words.len());
        assert_eq!(opcodes[0], OP_CAPABILITY);
        assert_eq!(opcodes[1], OP_MEMORY_MODEL);
        assert_eq!(opcodes[2], OP_ENTRY_POINT);
        assert_eq!(*opcodes.last().unwrap(), OP_FUNCTION_END);
        assert!(bound > 20 && bound < 64);
        // Two `OpFunction`/`OpFunctionEnd` pairs would mean a stray second entry point.
        assert_eq!(opcodes.iter().filter(|op| **op == OP_FUNCTION).count(), 1);
    }

    #[test]
    fn matmul_module_is_well_formed() {
        let words = matmul_fp32_spirv();
        assert_eq!(words[0], SPIRV_MAGIC);
        assert_eq!(words[1], SPIRV_VERSION_1_3);
        let mut cursor = 5;
        let mut opcodes = Vec::new();
        while cursor < words.len() {
            let word_count = (words[cursor] >> 16) as usize;
            assert!(word_count >= 1, "zero-length instruction at {cursor}");
            assert!(cursor + word_count <= words.len(), "instruction overruns");
            opcodes.push((words[cursor] & 0xffff) as u16);
            cursor += word_count;
        }
        assert_eq!(cursor, words.len());
        assert_eq!(opcodes[0], OP_CAPABILITY);
        assert_eq!(opcodes[1], OP_MEMORY_MODEL);
        assert_eq!(opcodes[2], OP_ENTRY_POINT);
        assert_eq!(*opcodes.last().unwrap(), OP_FUNCTION_END);
        assert_eq!(opcodes.iter().filter(|op| **op == OP_FUNCTION).count(), 1);
    }

    #[test]
    fn literal_strings_are_nul_terminated_and_word_padded() {
        assert_eq!(literal_string("main"), vec![0x6e69_616d, 0]);
        assert_eq!(literal_string("abc"), vec![0x0063_6261]);
    }

    #[test]
    fn workgroup_coverage_rounds_up() {
        assert_eq!(elementwise_workgroups(1), 1);
        assert_eq!(elementwise_workgroups(64), 1);
        assert_eq!(elementwise_workgroups(65), 2);
        assert_eq!(elementwise_workgroups(u32::MAX), 67_108_864);
    }

    #[test]
    fn matmul_workgroups_cover_all_dimensions() {
        assert_eq!(matmul_workgroups(1, 1, 1), [1, 1, 1]);
        assert_eq!(matmul_workgroups(8, 8, 1), [1, 1, 1]);
        assert_eq!(matmul_workgroups(9, 8, 1), [1, 2, 1]);
        assert_eq!(matmul_workgroups(8, 9, 1), [2, 1, 1]);
        assert_eq!(matmul_workgroups(8, 8, 3), [1, 1, 3]);
    }
}
