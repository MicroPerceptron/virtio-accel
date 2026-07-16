//! Portable foundations for an experimental virtio accelerator device.
//!
//! The facade intentionally contains no transport, operating-system, or vendor implementation.
//! Those adapters depend on the portable layers re-exported here.

#![no_std]
#![forbid(unsafe_code)]

pub use virtio_accel_core as core;
pub use virtio_accel_device as device;
pub use virtio_accel_proto as proto;
pub use virtio_accel_transport as transport;
