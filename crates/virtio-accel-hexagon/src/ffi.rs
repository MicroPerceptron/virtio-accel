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

pub(crate) const NODE_RESHAPE: u32 = 1;
pub(crate) const NODE_MATMUL: u32 = 2;
pub(crate) const NODE_MAX_POOL_2D: u32 = 3;

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
    pub dimensions: *const u32,
    pub rank: u32,
}

#[derive(Clone, Copy, Default)]
#[repr(C)]
pub(crate) struct NodeDesc {
    pub kind: u32,
    pub input0: u32,
    pub input1: u32,
    pub output: u32,
    pub kernel: [u32; 2],
    pub stride: [u32; 2],
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
