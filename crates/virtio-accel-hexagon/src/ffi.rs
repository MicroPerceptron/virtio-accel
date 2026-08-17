//! Narrow C ABI implemented by `native/qnn_bridge.cpp` against the detected QAIRT headers.

use core::ffi::{c_char, c_void};

pub(crate) const SUCCESS: u64 = 0;
pub(crate) const ERROR_INTERNAL: u64 = u64::MAX;
pub(crate) const ERROR_INCOMPATIBLE: u64 = u64::MAX - 1;
pub(crate) const ERROR_BUSY: u64 = u64::MAX - 2;
pub(crate) const ERROR_INVALID_ARGUMENT: u64 = u64::MAX - 3;
pub(crate) const ERROR_OUT_OF_MEMORY: u64 = u64::MAX - 4;

pub(crate) const TENSOR_NATIVE: u32 = 0;
pub(crate) const TENSOR_INPUT: u32 = 1;
pub(crate) const TENSOR_OUTPUT: u32 = 2;
pub(crate) const TENSOR_STATIC: u32 = 3;
pub(crate) const NO_IO_INDEX: u32 = u32::MAX;

pub(crate) const ELEMENT_BOOL: u32 = 0;
pub(crate) const ELEMENT_F16: u32 = 1;
pub(crate) const ELEMENT_F32: u32 = 2;
pub(crate) const ELEMENT_I8: u32 = 3;
pub(crate) const ELEMENT_I32: u32 = 4;

pub(crate) const PRECISION_DEFAULT: u32 = 0;
pub(crate) const PRECISION_F16: u32 = 1;
pub(crate) const PRECISION_F32: u32 = 2;

pub(crate) const NODE_RESHAPE: u32 = 1;
pub(crate) const NODE_MATMUL: u32 = 2;
pub(crate) const NODE_MAX_POOL_2D: u32 = 3;
pub(crate) const NODE_ADD: u32 = 4;
pub(crate) const NODE_SUBTRACT: u32 = 5;
pub(crate) const NODE_MAXIMUM: u32 = 6;
pub(crate) const NODE_MINIMUM: u32 = 7;
pub(crate) const NODE_MULTIPLY: u32 = 8;
pub(crate) const NODE_TRANSPOSE: u32 = 9;
pub(crate) const NODE_REVERSE: u32 = 10;
pub(crate) const NODE_CONCAT: u32 = 11;
pub(crate) const NODE_POWER: u32 = 12;
pub(crate) const NODE_ABS: u32 = 13;
pub(crate) const NODE_CEIL: u32 = 14;
pub(crate) const NODE_COS: u32 = 15;
pub(crate) const NODE_EXP: u32 = 16;
pub(crate) const NODE_FLOOR: u32 = 17;
pub(crate) const NODE_LOG: u32 = 18;
pub(crate) const NODE_NEGATE: u32 = 19;
pub(crate) const NODE_RECIPROCAL: u32 = 20;
pub(crate) const NODE_RSQRT: u32 = 21;
pub(crate) const NODE_SIN: u32 = 22;
pub(crate) const NODE_SIGMOID: u32 = 23;
pub(crate) const NODE_TANH: u32 = 24;
pub(crate) const NODE_CLAMP: u32 = 25;
pub(crate) const NODE_EQUAL: u32 = 26;
pub(crate) const NODE_GREATER: u32 = 27;
pub(crate) const NODE_GREATER_EQUAL: u32 = 28;
pub(crate) const NODE_SELECT: u32 = 29;
pub(crate) const NODE_LOGICAL_AND: u32 = 30;
pub(crate) const NODE_LOGICAL_OR: u32 = 31;
pub(crate) const NODE_LOGICAL_XOR: u32 = 32;
pub(crate) const NODE_LOGICAL_NOT: u32 = 33;
pub(crate) const NODE_ARGMAX: u32 = 34;
pub(crate) const NODE_REDUCE_MAX: u32 = 35;
pub(crate) const NODE_REDUCE_MIN: u32 = 36;
pub(crate) const NODE_REDUCE_PRODUCT: u32 = 37;
pub(crate) const NODE_REDUCE_SUM: u32 = 38;

pub(crate) const EVENT_PENDING: u32 = 0;
pub(crate) const EVENT_COMPLETE: u32 = 1;
pub(crate) const EVENT_FAILED: u32 = 2;

#[repr(C)]
pub(crate) struct Runtime {
    _private: [u8; 0],
}

#[repr(C)]
pub(crate) struct Graph {
    _private: [u8; 0],
}

#[repr(C)]
pub(crate) struct Event {
    _private: [u8; 0],
}

#[derive(Clone, Copy)]
#[repr(C)]
pub(crate) struct RuntimeInfo {
    pub backend_id: u32,
    pub core_major: u32,
    pub core_minor: u32,
    pub core_patch: u32,
    pub backend_major: u32,
    pub backend_minor: u32,
    pub backend_patch: u32,
    pub provider_name: [c_char; 128],
    pub build_id: [c_char; 256],
}

impl Default for RuntimeInfo {
    fn default() -> Self {
        Self {
            backend_id: 0,
            core_major: 0,
            core_minor: 0,
            core_patch: 0,
            backend_major: 0,
            backend_minor: 0,
            backend_patch: 0,
            provider_name: [0; 128],
            build_id: [0; 256],
        }
    }
}

#[derive(Clone, Copy)]
#[repr(C)]
pub(crate) struct TensorDesc {
    pub value: u32,
    pub role: u32,
    pub io_index: u32,
    pub element: u32,
    pub quantized: u32,
    pub rank: u32,
    pub dimensions: *const u32,
    pub constant_data: *const u8,
    pub constant_size: u64,
    pub scale: f32,
    pub offset: i32,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub(crate) struct NodeDesc {
    pub kind: u32,
    pub inputs: *const u32,
    pub input_count: u32,
    pub outputs: *const u32,
    pub output_count: u32,
    pub parameters: *const i32,
    pub parameter_count: u32,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub(crate) struct Binding {
    pub data: *mut c_void,
    pub size: u64,
}

unsafe extern "C" {
    pub(crate) fn va_qnn_runtime_create(
        library_path: *const c_char,
        runtime: *mut *mut Runtime,
        info: *mut RuntimeInfo,
        message: *mut c_char,
        message_size: usize,
    ) -> u64;
    pub(crate) fn va_qnn_runtime_free(runtime: *mut Runtime) -> u64;

    pub(crate) fn va_qnn_graph_create(
        runtime: *mut Runtime,
        tensors: *const TensorDesc,
        tensor_count: u32,
        nodes: *const NodeDesc,
        node_count: u32,
        precision: u32,
        graph: *mut *mut Graph,
        message: *mut c_char,
        message_size: usize,
    ) -> u64;
    pub(crate) fn va_qnn_graph_free(graph: *mut Graph) -> u64;

    pub(crate) fn va_qnn_graph_execute_async(
        graph: *mut Graph,
        inputs: *const Binding,
        input_count: u32,
        outputs: *const Binding,
        output_count: u32,
        event: *mut *mut Event,
        message: *mut c_char,
        message_size: usize,
    ) -> u64;
    pub(crate) fn va_qnn_event_poll(event: *const Event, qnn_error: *mut u64) -> u32;
    pub(crate) fn va_qnn_event_free(event: *mut Event) -> u64;
}
