//! Hand-written declarations for the OpenVINO C API (`libopenvino_c`).
//!
//! This module is the crate's only foreign ABI boundary. Every declaration mirrors the stable C
//! headers shipped with OpenVINO (`openvino/c/*.h`) and is audited in `SAFETY.md`. It contains
//! type and constant definitions only; ownership rules and every call live in `native.rs`.

#![allow(non_camel_case_types)]

use core::ffi::c_char;
use core::marker::{PhantomData, PhantomPinned};

/// Status codes returned by every fallible OpenVINO C API function.
///
/// Mirrors `ov_status_e` in `openvino/c/ov_common.h`; negative values map C++ exceptions.
pub(crate) type ov_status_e = core::ffi::c_int;

pub(crate) const OV_STATUS_OK: ov_status_e = 0;

/// Opaque OpenVINO core handle (`ov_core_t`).
#[repr(C)]
pub(crate) struct ov_core_t {
    _unconstructable: [u8; 0],
    _not_send_sync: PhantomData<(*mut u8, PhantomPinned)>,
}

/// Device enumeration result (`ov_available_devices_t` in `openvino/c/ov_core.h`).
#[repr(C)]
pub(crate) struct ov_available_devices_t {
    pub devices: *mut *mut c_char,
    pub size: usize,
}

unsafe extern "C" {
    pub(crate) fn ov_core_create(core: *mut *mut ov_core_t) -> ov_status_e;
    pub(crate) fn ov_core_free(core: *mut ov_core_t);
    pub(crate) fn ov_core_get_available_devices(
        core: *const ov_core_t,
        devices: *mut ov_available_devices_t,
    ) -> ov_status_e;
    pub(crate) fn ov_available_devices_free(devices: *mut ov_available_devices_t);
}
