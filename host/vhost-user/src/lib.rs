//! Linux vhost-user binding. One connection owns one accelerator on its calling thread.
//! No guest pointers reach the portable command processor or accelerator backend.
#![forbid(unsafe_code)]

mod memory;
mod server;

pub use server::{Host, serve};

use virtio_accel_proto::{Le16, Le32, WireConfig};

/// This implementation's bounds, not additional protocol limits.
pub const QUEUE_SIZE: u16 = 1024;
pub const MAX_DESCRIPTORS: usize = 256;
pub const CONFIG: WireConfig = WireConfig {
    protocol_major: Le16::new(1),
    protocol_minor: Le16::new(0),
    command_queue_count: Le16::new(1),
    max_chain_descriptors: Le16::new(MAX_DESCRIPTORS as u16),
    max_request_bytes: Le32::new(1 << 20),
    max_response_bytes: Le32::new(1 << 20),
};

/// Base Virtio and vhost-user features. Device-specific features remain zero.
pub const FEATURES: u64 = (1 << 32) | (1 << 30);
