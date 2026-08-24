//! Hand-written declarations for the HRX C ABI (`libhrx`).
//!
//! This module is the crate's only foreign ABI boundary. Every declaration is transcribed from the
//! pinned release headers (`include/hrx/hrx_runtime.h`, `libhrx.so.0.1.0`, version 0.1.0) and is
//! audited in `SAFETY.md`. It contains type, constant, and `extern` declarations only; ownership
//! rules and every call live in `native.rs`.
//!
//! Scope is the buffer/device/stream subset this ticket implements; the executable and dispatch
//! entry points are declared with the native execution path in a later ticket.

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
}

/// `hrx_status_t` — opaque owned pointer; `NULL` means OK (`hrx_runtime.h`).
pub(crate) type hrx_status_t = *mut hrx_status_s;
/// `hrx_device_t` — refcounted device handle.
pub(crate) type hrx_device_t = *mut hrx_device_s;
/// `hrx_stream_t` — refcounted stream handle.
pub(crate) type hrx_stream_t = *mut hrx_stream_s;
/// `hrx_buffer_t` — refcounted buffer handle.
pub(crate) type hrx_buffer_t = *mut hrx_buffer_s;

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
}
