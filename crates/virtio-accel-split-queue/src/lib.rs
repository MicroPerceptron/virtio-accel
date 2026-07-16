//! Deterministic in-memory model of the protocol 1.0 split-virtqueue profile.
//!
//! The model owns descriptor, available-ring, and used-ring state without guest addresses or a
//! concrete transport library. It is deliberately single-owner and contains no lock or executor.

#![no_std]
#![forbid(unsafe_code)]

extern crate alloc;

mod chain;
mod queue;

pub use chain::{
    ChainBuildError, Descriptor, DriverChain, SplitDeviceChain, SplitSink, SplitSource,
    VIRTQ_DESC_F_INDIRECT, VIRTQ_DESC_F_NEXT, VIRTQ_DESC_F_WRITE,
};
pub use queue::{ReclaimedChains, RingCounters, SplitQueue, SplitQueueError, SplitQueueInitError};
