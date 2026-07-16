//! Portable descriptor-chain, queue, and notification contracts.
//!
//! This crate owns no guest addresses, mappings, ring layout, threads, or global runtime. Concrete
//! transports retain those details and expose only validated chain metadata plus owned queue tokens.

#![no_std]
#![forbid(unsafe_code)]

mod bytes;
mod queue;
mod regions;

pub use bytes::{ByteAccessError, ReadableBytes, WritableBytes};
pub use queue::{
    ChainError, ChainId, ChainIo, ChainIoResult, DeviceChain, DeviceQueue, DriverQueue,
    MAX_SPLIT_QUEUE_SIZE, MalformedChain, NotificationHint, NotificationRecheck, PublishError,
    PublishErrorKind, PublishedChain, QueueConfigError, QueueControl, QueueEpoch, QueueError,
    QueuePort, QueueSize, QueueState, ReclaimedChain, UsedChain, UsedLength,
};
pub use regions::{
    ChainLayout, ChainLayoutError, ChainRegion, RegionDirection, validate_chain_layout,
};
