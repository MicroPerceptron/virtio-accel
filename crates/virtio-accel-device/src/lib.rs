//! Transport-neutral device-side state and validation.
//!
//! A future rust-vmm adapter will translate descriptor chains into this layer. Provider backends
//! must never receive guest addresses or virtqueue descriptors.

#![no_std]
#![forbid(unsafe_code)]

extern crate alloc;

mod decoder;
mod engine;
mod frame;
mod object_table;
mod regions;
mod response;
mod state;

pub use decoder::{
    DecodedBinding, DecodedRequest, DecodedRequestBody, DecoderLimits, DecoderLimitsError,
    FrameDecodeError, FrameDecoder, UnrecoverableDecodeError,
};
pub use engine::{
    AcceleratorState, CommandOutcome, CommandProcessError, CommandProcessor,
    CommandProcessorInitError, DeviceHealth, ResetDisposition, ResetError, ResetReport,
};
pub use frame::{FramePreflight, FramePreflightError, UnusableFrame, preflight_command_frame};
pub use object_table::{ObjectId, ObjectKind, ObjectNamespace, ObjectTable, ObjectTableError};
pub use regions::{
    ChainLayout, ChainLayoutError, ChainRegion, ReadableRegion, RegionDirection,
    SegmentedRegionError, SegmentedSink, SegmentedSource, WritableRegion, validate_chain_layout,
};
pub use response::{ResponsePayload, ResponseWriteError, ResponseWriter};
pub use state::{
    BufferCreateOutcome, BufferRecord, ChildCounts, ContextRecord, CreateError, DeviceState,
    DeviceStateConfigError, DeviceStateError, EventRecord, ProgramRecord, QueueRecord,
    ReleaseState, ResourceCounts, ResourcePolicy, RestoreError, RetainedBytes, SubmissionResources,
};
use virtio_accel_core::BackendError;
use virtio_accel_proto::StatusCode;

pub const fn status_from_backend_error(error: BackendError) -> StatusCode {
    match error {
        BackendError::Unsupported => StatusCode::UNSUPPORTED,
        BackendError::Incompatible => StatusCode::INCOMPATIBLE,
        BackendError::InvalidArgument => StatusCode::INVALID_ARGUMENT,
        BackendError::OutOfBounds => StatusCode::OUT_OF_BOUNDS,
        BackendError::Busy => StatusCode::BUSY,
        BackendError::OutOfMemory => StatusCode::OUT_OF_MEMORY,
        BackendError::ResourceLimit => StatusCode::RESOURCE_LIMIT,
        BackendError::DeadlineExpired => StatusCode::DEADLINE_EXPIRED,
        BackendError::DeviceLost => StatusCode::DEVICE_LOST,
        BackendError::PermissionDenied => StatusCode::PERMISSION_DENIED,
        BackendError::External { .. } => StatusCode::INTERNAL_ERROR,
    }
}

pub const fn status_from_object_table_error(error: ObjectTableError) -> StatusCode {
    match error {
        ObjectTableError::InvalidId => StatusCode::INVALID_ARGUMENT,
        ObjectTableError::WrongKind | ObjectTableError::StaleId => StatusCode::STALE_OBJECT,
        ObjectTableError::Full => StatusCode::RESOURCE_LIMIT,
        ObjectTableError::AllocationFailed => StatusCode::OUT_OF_MEMORY,
    }
}

pub const fn status_from_device_state_error(error: DeviceStateError) -> StatusCode {
    match error {
        DeviceStateError::InvalidArgument | DeviceStateError::InvalidObject => {
            StatusCode::INVALID_ARGUMENT
        }
        DeviceStateError::StaleObject | DeviceStateError::ContextMismatch => {
            StatusCode::STALE_OBJECT
        }
        DeviceStateError::Busy | DeviceStateError::Releasing => StatusCode::BUSY,
        DeviceStateError::ResourceLimit | DeviceStateError::ReferenceCountOverflow => {
            StatusCode::RESOURCE_LIMIT
        }
        DeviceStateError::OutOfMemory => StatusCode::OUT_OF_MEMORY,
        DeviceStateError::InvalidTransition => StatusCode::INTERNAL_ERROR,
    }
}
