//! Transport-neutral device-side state and validation.
//!
//! A future rust-vmm adapter will translate descriptor chains into this layer. Provider backends
//! must never receive guest addresses or virtqueue descriptors.

#![no_std]
#![forbid(unsafe_code)]

extern crate alloc;

mod object_table;

pub use object_table::{ObjectId, ObjectKind, ObjectTable, ObjectTableError};
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
