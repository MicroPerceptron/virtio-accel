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
const OP_U_LESS_THAN: u16 = 176;
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
}
