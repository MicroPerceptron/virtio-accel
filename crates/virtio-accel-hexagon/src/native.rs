//! Native QNN activation guard.
//!
//! QAIRT detection alone is not a support claim. Until the audited bridge is complete, even a host
//! with the full SDK receives an explicit unavailable result rather than a CPU/GPU fallback or a
//! partially initialized HTP backend.

#![forbid(unsafe_code)]

use crate::{InitError, ffi};

#[derive(Clone, Copy, Debug, Default)]
pub struct HexagonAccelerator;

impl HexagonAccelerator {
    pub fn new() -> Result<Self, InitError> {
        debug_assert!(!ffi::NATIVE_BRIDGE_IMPLEMENTED);
        Err(InitError::RuntimeUnavailable)
    }

    pub fn available_devices() -> Result<Vec<String>, InitError> {
        Err(InitError::RuntimeUnavailable)
    }
}
