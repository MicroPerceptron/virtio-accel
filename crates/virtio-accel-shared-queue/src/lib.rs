//! The protocol 1.0 split virtqueue, laid out in memory a driver and a device share.
//!
//! `virtio-accel-transport` says what a queue must do and names no memory. The reference model,
//! `virtio-accel-split-queue`, does it with both halves in one value that owns every byte, which
//! is what makes it an oracle and what keeps it from being a queue two parties can use. This crate
//! is the same contracts over the real layout (Virtio 1.2 §2.7): a descriptor table, an available
//! ring the driver writes and a used ring the device writes, at device addresses, reached through
//! [`SharedMemory`] — the one thing a platform supplies.
//!
//! - [`DriverRing`] is the driver's half ([`DriverQueue`](virtio_accel_transport::DriverQueue)).
//! - [`DeviceRing`] is the device's half ([`DeviceQueue`](virtio_accel_transport::DeviceQueue)).
//!
//! Each trusts nothing the other side wrote, allocates nothing, and contains no `unsafe`: the
//! raw access is behind the trait, in the implementation that knows how its side maps the memory.
//! Storage is inline and sized by const parameters. The profile is the protocol's — direct
//! descriptors, no indirect tables, no event indices, the two suppression flags.

#![no_std]
#![forbid(unsafe_code)]

mod device;
mod driver;
mod layout;
mod memory;

#[cfg(test)]
extern crate std;
#[cfg(test)]
mod tests;

pub use device::{DeviceRing, DeviceRingChain, DeviceRingError, RingSink, RingSource};
pub use driver::{
    ChainShapeError, DriverChain, DriverRing, DriverRingError, ReclaimedChains, Region,
};
pub use layout::{
    RingAddresses, VIRTQ_AVAIL_F_NO_INTERRUPT, VIRTQ_DESC_F_INDIRECT, VIRTQ_DESC_F_NEXT,
    VIRTQ_DESC_F_WRITE, VIRTQ_USED_F_NO_NOTIFY,
};
pub use memory::{Access, MemoryFault, SharedMemory};
