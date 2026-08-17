//! Reserved native QNN C ABI boundary.
//!
//! This module is compiled only after the build probe finds the public QNN development surface.
//! The initial repository integration intentionally contains no guessed ABI declarations: the
//! complete matching QAIRT headers are required before this boundary can be implemented safely.

#![forbid(unsafe_code)]

pub(crate) const NATIVE_BRIDGE_IMPLEMENTED: bool = false;
