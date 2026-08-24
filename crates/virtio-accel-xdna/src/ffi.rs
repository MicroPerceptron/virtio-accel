//! Hand-written declarations for the HRX C ABI (`libhrx`).
//!
//! This module is the crate's only foreign ABI boundary. Every declaration is transcribed from the
//! pinned release headers (`include/hrx/hrx_runtime.h`, `libhrx.so.0.1.0`, version 0.1.0) and is
//! audited in `SAFETY.md`. It contains type, constant, and `extern` declarations only; ownership
//! rules and every call live in `native.rs`.
//!
//! Scope is the status, device/stream, buffer, executable, and dispatch subset the native
//! execution path calls; declarations are added as the backend grows.

#![allow(non_camel_case_types)]

use core::ffi::{c_char, c_int, c_void};
use core::marker::{PhantomData, PhantomPinned};

/// `hrx_status_code_t` (`hrx_runtime.h`); mirrors IREE's status codes. `0` is success.
pub(crate) type hrx_status_code_t = c_int;

/// Memory-type bitmask (`typedef uint32_t hrx_memory_type_t`).
pub(crate) type hrx_memory_type_t = u32;
pub(crate) const HRX_MEMORY_TYPE_HOST_LOCAL: hrx_memory_type_t = 0x0000_0046;
pub(crate) const HRX_MEMORY_TYPE_DEVICE_VISIBLE: hrx_memory_type_t = 0x0000_0010;

/// Buffer-usage bitmask (`typedef uint32_t hrx_buffer_usage_t`).
pub(crate) type hrx_buffer_usage_t = u32;
pub(crate) const HRX_BUFFER_USAGE_DEFAULT: hrx_buffer_usage_t = 0x0000_0C03;
pub(crate) const HRX_BUFFER_USAGE_MAPPING_PERSISTENT: hrx_buffer_usage_t = 0x0200_0000;

/// Mapping-mode value (`typedef uint32_t hrx_mapping_mode_t`).
pub(crate) type hrx_mapping_mode_t = u32;
pub(crate) const HRX_MAPPING_MODE_PERSISTENT: hrx_mapping_mode_t = 0x0000_0002;

/// Map-access flags (`typedef uint16_t hrx_map_flags_t`), from `HRX_MEMORY_ACCESS_*`.
pub(crate) type hrx_map_flags_t = u16;
pub(crate) const HRX_MAP_READ: hrx_map_flags_t = 0x01;
pub(crate) const HRX_MAP_WRITE: hrx_map_flags_t = 0x02;

macro_rules! opaque_handle {
    ($(#[$doc:meta] $name:ident),* $(,)?) => {
        $(
            #[$doc]
            #[repr(C)]
            pub(crate) struct $name {
                _unconstructable: [u8; 0],
                _not_send_sync: PhantomData<(*mut u8, PhantomPinned)>,
            }
        )*
    };
}

opaque_handle! {
    /// Opaque status object (`hrx_status_s`); a non-NULL `*mut` is an error the caller owns.
    hrx_status_s,
    /// Opaque device handle target (`hrx_device_s`).
    hrx_device_s,
    /// Opaque stream handle target (`hrx_stream_s`).
    hrx_stream_s,
    /// Opaque buffer handle target (`hrx_buffer_s`).
    hrx_buffer_s,
    /// Opaque executable handle target (`hrx_executable_s`).
    hrx_executable_s,
}

/// `hrx_status_t` — opaque owned pointer; `NULL` means OK (`hrx_runtime.h`).
pub(crate) type hrx_status_t = *mut hrx_status_s;
/// `hrx_device_t` — refcounted device handle.
pub(crate) type hrx_device_t = *mut hrx_device_s;
/// `hrx_stream_t` — refcounted stream handle.
pub(crate) type hrx_stream_t = *mut hrx_stream_s;
/// `hrx_buffer_t` — refcounted buffer handle.
pub(crate) type hrx_buffer_t = *mut hrx_buffer_s;
/// `hrx_executable_t` — refcounted executable handle.
pub(crate) type hrx_executable_t = *mut hrx_executable_s;

/// Borrowed byte span (`hrx_const_byte_span_t`).
#[repr(C)]
pub(crate) struct hrx_const_byte_span_t {
    pub data: *const c_void,
    pub data_length: usize,
}

/// Borrowed string view (`hrx_string_view_t`); not necessarily NUL-terminated.
#[repr(C)]
pub(crate) struct hrx_string_view_t {
    pub data: *const c_char,
    pub size: usize,
}

/// Dispatch grid configuration (`hrx_dispatch_config_t`); the amdxdna path uses {1,1,1}/{1,1,1}/0.
#[repr(C)]
pub(crate) struct hrx_dispatch_config_t {
    pub workgroup_count: [u32; 3],
    pub workgroup_size: [u32; 3],
    pub subgroup_size: u32,
}

/// One bound buffer range for dispatch (`hrx_buffer_ref_t`).
#[repr(C)]
pub(crate) struct hrx_buffer_ref_t {
    pub buffer: hrx_buffer_t,
    pub offset: usize,
    pub length: usize,
}

pub(crate) const HRX_AMDXDNA_CONTEXT_MODE_CREATE: u32 = 0;
pub(crate) const HRX_AMDXDNA_EXECUTABLE_RUN_ABI_VERSION_0: u32 = 0;
pub(crate) const HRX_AMDXDNA_EXECUTABLE_ENTRY_POINT_ABI_VERSION_0: u32 = 0;
pub(crate) const HRX_AMDXDNA_EXECUTABLE_CREATE_PARAMS_ABI_VERSION_0: u32 = 0;

/// One control-code run (`hrx_amdxdna_executable_run_t`); `transaction` is the `insts.bin` TXN.
#[repr(C)]
pub(crate) struct hrx_amdxdna_executable_run_t {
    pub record_length: u32,
    pub abi_version: u32,
    pub transaction: hrx_const_byte_span_t,
    pub data_payload: hrx_const_byte_span_t,
}

/// One dispatchable entry point (`hrx_amdxdna_executable_entry_point_t`).
#[repr(C)]
pub(crate) struct hrx_amdxdna_executable_entry_point_t {
    pub record_length: u32,
    pub abi_version: u32,
    pub name: hrx_string_view_t,
    pub context_mode: u32,
    pub xclbin_ordinal: u32,
    pub pdi_ordinal: u32,
    pub source_line: u32,
    pub source_file: hrx_string_view_t,
    pub runs: *const hrx_amdxdna_executable_run_t,
    pub run_count: usize,
}

/// Executable creation parameters (`hrx_amdxdna_executable_create_params_t`).
#[repr(C)]
pub(crate) struct hrx_amdxdna_executable_create_params_t {
    pub record_length: u32,
    pub abi_version: u32,
    pub flags: u32,
    pub reserved: u32,
    pub xclbins: *const hrx_const_byte_span_t,
    pub xclbin_count: usize,
    pub entry_points: *const hrx_amdxdna_executable_entry_point_t,
    pub entry_point_count: usize,
}

unsafe extern "C" {
    // Status.
    pub(crate) fn hrx_status_code(status: hrx_status_t) -> hrx_status_code_t;
    pub(crate) fn hrx_status_to_string(
        status: hrx_status_t,
        out_message: *mut *mut c_char,
        out_length: *mut usize,
    ) -> hrx_status_t;
    pub(crate) fn hrx_status_free_message(message: *mut c_char);
    pub(crate) fn hrx_status_ignore(status: hrx_status_t);

    // Device lifecycle. `hrx_gpu_initialize` is process-wide; the NPU appears under the "gpu"
    // accelerator namespace (the amdxdna HAL). No shutdown is called: the process owns it.
    pub(crate) fn hrx_gpu_initialize(flags: u32) -> hrx_status_t;
    pub(crate) fn hrx_gpu_device_count(count: *mut c_int) -> hrx_status_t;
    pub(crate) fn hrx_gpu_device_get(index: c_int, device: *mut hrx_device_t) -> hrx_status_t;

    // Stream lifecycle.
    pub(crate) fn hrx_stream_create(
        device: hrx_device_t,
        flags: u32,
        stream: *mut hrx_stream_t,
    ) -> hrx_status_t;
    pub(crate) fn hrx_stream_release(stream: hrx_stream_t);

    // Buffers: stream-ordered allocation, persistent host mapping, explicit cache management.
    pub(crate) fn hrx_buffer_allocate(
        stream: hrx_stream_t,
        size: usize,
        mem_type: hrx_memory_type_t,
        usage: hrx_buffer_usage_t,
        buffer: *mut hrx_buffer_t,
    ) -> hrx_status_t;
    pub(crate) fn hrx_buffer_release(buffer: hrx_buffer_t);
    pub(crate) fn hrx_buffer_map_with_mode(
        buffer: hrx_buffer_t,
        mapping_mode: hrx_mapping_mode_t,
        flags: hrx_map_flags_t,
        offset: usize,
        size: usize,
        mapped_ptr: *mut *mut c_void,
    ) -> hrx_status_t;
    pub(crate) fn hrx_buffer_flush_range(
        buffer: hrx_buffer_t,
        offset: usize,
        size: usize,
    ) -> hrx_status_t;
    pub(crate) fn hrx_buffer_invalidate_range(
        buffer: hrx_buffer_t,
        offset: usize,
        size: usize,
    ) -> hrx_status_t;

    // Executables (amdxdna): create/load from xclbin(s) + TXN runs, look up an export, refcount.
    pub(crate) fn hrx_amdxdna_executable_create(
        device: hrx_device_t,
        params: *const hrx_amdxdna_executable_create_params_t,
        executable: *mut hrx_executable_t,
    ) -> hrx_status_t;
    pub(crate) fn hrx_executable_lookup_export_by_name(
        executable: hrx_executable_t,
        name: *const c_char,
        export_ordinal: *mut u32,
    ) -> hrx_status_t;
    pub(crate) fn hrx_executable_release(executable: hrx_executable_t);

    // Dispatch: record into the stream's command buffer, then flush + block on synchronize.
    pub(crate) fn hrx_stream_dispatch(
        stream: hrx_stream_t,
        executable: hrx_executable_t,
        export_ordinal: u32,
        config: *const hrx_dispatch_config_t,
        constants: *const c_void,
        constants_size: usize,
        bindings: *const hrx_buffer_ref_t,
        binding_count: usize,
        flags: u32,
    ) -> hrx_status_t;
    pub(crate) fn hrx_stream_synchronize(stream: hrx_stream_t) -> hrx_status_t;
}
