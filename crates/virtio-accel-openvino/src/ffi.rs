//! Hand-written declarations for the OpenVINO C API (`libopenvino_c`).
//!
//! This module is the crate's only foreign ABI boundary. Every declaration mirrors the stable C
//! headers shipped with OpenVINO (`openvino/c/*.h`) and is audited in `SAFETY.md`. It contains
//! type and constant definitions only; ownership rules and every call live in `native.rs`.

#![allow(non_camel_case_types)]

use core::ffi::{c_char, c_uint, c_void};
use core::marker::{PhantomData, PhantomPinned};

/// Status codes returned by every fallible OpenVINO C API function.
///
/// Mirrors `ov_status_e` in `openvino/c/ov_common.h`; negative values map C++ exceptions.
pub(crate) type ov_status_e = core::ffi::c_int;

pub(crate) const OV_STATUS_OK: ov_status_e = 0;
pub(crate) const OV_STATUS_NOT_IMPLEMENTED: ov_status_e = -2;
pub(crate) const OV_STATUS_PARAMETER_MISMATCH: ov_status_e = -4;
pub(crate) const OV_STATUS_NOT_FOUND: ov_status_e = -5;
pub(crate) const OV_STATUS_OUT_OF_BOUNDS: ov_status_e = -6;
pub(crate) const OV_STATUS_REQUEST_BUSY: ov_status_e = -8;
pub(crate) const OV_STATUS_RESULT_NOT_READY: ov_status_e = -9;
pub(crate) const OV_STATUS_NOT_ALLOCATED: ov_status_e = -10;
pub(crate) const OV_STATUS_INFER_CANCELLED: ov_status_e = -13;
pub(crate) const OV_STATUS_INVALID_C_PARAM: ov_status_e = -14;
pub(crate) const OV_STATUS_NOT_IMPLEMENT_C_METHOD: ov_status_e = -16;

/// Element type codes, aligned with `ov_element_type_e` in `openvino/c/ov_common.h`.
pub(crate) type ov_element_type_e = c_uint;

pub(crate) const ELEMENT_BOOLEAN: ov_element_type_e = 1;
pub(crate) const ELEMENT_F16: ov_element_type_e = 3;
pub(crate) const ELEMENT_F32: ov_element_type_e = 4;
pub(crate) const ELEMENT_I8: ov_element_type_e = 7;
pub(crate) const ELEMENT_I32: ov_element_type_e = 9;
pub(crate) const ELEMENT_I64: ov_element_type_e = 10;
pub(crate) const ELEMENT_U8: ov_element_type_e = 16;

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
    /// Opaque OpenVINO core handle (`ov_core_t`).
    ov_core_t,
    /// Opaque source model handle (`ov_model_t`).
    ov_model_t,
    /// Opaque compiled model handle (`ov_compiled_model_t`).
    ov_compiled_model_t,
    /// Opaque inference request handle (`ov_infer_request_t`).
    ov_infer_request_t,
    /// Opaque tensor handle (`ov_tensor_t`).
    ov_tensor_t,
}

/// Tensor shape (`ov_shape_t` in `openvino/c/ov_shape.h`); passed by value where the C API does.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct ov_shape_t {
    pub rank: i64,
    pub dims: *mut i64,
}

/// Device enumeration result (`ov_available_devices_t` in `openvino/c/ov_core.h`).
#[repr(C)]
pub(crate) struct ov_available_devices_t {
    pub devices: *mut *mut c_char,
    pub size: usize,
}

unsafe extern "C" {
    // Core lifecycle and discovery.
    pub(crate) fn ov_core_create(core: *mut *mut ov_core_t) -> ov_status_e;
    pub(crate) fn ov_core_free(core: *mut ov_core_t);
    pub(crate) fn ov_core_get_available_devices(
        core: *const ov_core_t,
        devices: *mut ov_available_devices_t,
    ) -> ov_status_e;
    pub(crate) fn ov_available_devices_free(devices: *mut ov_available_devices_t);

    // Model reading and compilation.
    pub(crate) fn ov_core_read_model_from_memory_buffer(
        core: *const ov_core_t,
        model_str: *const c_char,
        str_len: usize,
        weights: *const ov_tensor_t,
        model: *mut *mut ov_model_t,
    ) -> ov_status_e;
    /// C-variadic: `property_args_size` counts the variadic arguments, two per key/value pair.
    pub(crate) fn ov_core_compile_model(
        core: *const ov_core_t,
        model: *const ov_model_t,
        device_name: *const c_char,
        property_args_size: usize,
        compiled_model: *mut *mut ov_compiled_model_t,
        ...
    ) -> ov_status_e;
    pub(crate) fn ov_model_free(model: *mut ov_model_t);

    // Compiled models.
    pub(crate) fn ov_compiled_model_create_infer_request(
        compiled_model: *const ov_compiled_model_t,
        infer_request: *mut *mut ov_infer_request_t,
    ) -> ov_status_e;
    pub(crate) fn ov_compiled_model_inputs_size(
        compiled_model: *const ov_compiled_model_t,
        size: *mut usize,
    ) -> ov_status_e;
    pub(crate) fn ov_compiled_model_outputs_size(
        compiled_model: *const ov_compiled_model_t,
        size: *mut usize,
    ) -> ov_status_e;
    pub(crate) fn ov_compiled_model_free(compiled_model: *mut ov_compiled_model_t);

    // Shapes and tensors.
    pub(crate) fn ov_shape_create(
        rank: i64,
        dims: *const i64,
        shape: *mut ov_shape_t,
    ) -> ov_status_e;
    pub(crate) fn ov_shape_free(shape: *mut ov_shape_t) -> ov_status_e;
    pub(crate) fn ov_tensor_create_from_host_ptr(
        ty: ov_element_type_e,
        shape: ov_shape_t,
        host_ptr: *mut c_void,
        tensor: *mut *mut ov_tensor_t,
    ) -> ov_status_e;
    pub(crate) fn ov_tensor_data(tensor: *const ov_tensor_t, data: *mut *mut c_void)
    -> ov_status_e;
    pub(crate) fn ov_tensor_free(tensor: *mut ov_tensor_t);

    // Inference requests.
    pub(crate) fn ov_infer_request_set_input_tensor_by_index(
        infer_request: *mut ov_infer_request_t,
        idx: usize,
        tensor: *const ov_tensor_t,
    ) -> ov_status_e;
    pub(crate) fn ov_infer_request_set_output_tensor_by_index(
        infer_request: *mut ov_infer_request_t,
        idx: usize,
        tensor: *const ov_tensor_t,
    ) -> ov_status_e;
    pub(crate) fn ov_infer_request_get_output_tensor_by_index(
        infer_request: *const ov_infer_request_t,
        idx: usize,
        tensor: *mut *mut ov_tensor_t,
    ) -> ov_status_e;
    pub(crate) fn ov_infer_request_start_async(
        infer_request: *mut ov_infer_request_t,
    ) -> ov_status_e;
    pub(crate) fn ov_infer_request_wait(infer_request: *mut ov_infer_request_t) -> ov_status_e;
    pub(crate) fn ov_infer_request_wait_for(
        infer_request: *mut ov_infer_request_t,
        timeout_ms: i64,
    ) -> ov_status_e;
    pub(crate) fn ov_infer_request_cancel(infer_request: *mut ov_infer_request_t) -> ov_status_e;
    pub(crate) fn ov_infer_request_free(infer_request: *mut ov_infer_request_t);

    /// Exported property-key symbol for the execution-mode hint (`openvino/c/ov_property.h`).
    pub(crate) static ov_property_key_hint_execution_mode: *const c_char;
}
