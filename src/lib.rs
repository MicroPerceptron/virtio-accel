//! Portable facade for the executable virtio-accel guest/device stack.
//!
//! The wider workspace ships concrete queue, guest, device, TOSA, conformance, mock, and Core ML
//! implementations. This facade intentionally re-exports only the portable runtime layers;
//! host-native backends depend inward on them instead of becoming cross-platform dependencies.

#![no_std]
#![forbid(unsafe_code)]

pub use virtio_accel_core as core;
pub use virtio_accel_device as device;
pub use virtio_accel_guest as guest;
pub use virtio_accel_proto as proto;
pub use virtio_accel_split_queue as split_queue;
pub use virtio_accel_transport as transport;
