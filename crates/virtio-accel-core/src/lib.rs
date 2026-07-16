//! Transport-independent accelerator semantics.
//!
//! This crate does not name virtqueues, guest memory, host operating systems, or vendor APIs.
//! Transport adapters validate untrusted input and translate it into these typed contracts.

#![no_std]
#![forbid(unsafe_code)]

use bitflags::bitflags;
use core::fmt;
use core::num::{NonZeroU32, NonZeroU64};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackendError {
    Unsupported,
    Incompatible,
    InvalidArgument,
    OutOfBounds,
    Busy,
    OutOfMemory,
    ResourceLimit,
    DeadlineExpired,
    DeviceLost,
    PermissionDenied,
    /// Stable provider-owned error namespace. Transport adapters must not reinterpret it.
    External {
        domain: u32,
        code: i64,
    },
}

impl fmt::Display for BackendError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

/// Extensible accelerator class. Unknown values remain representable across newer implementations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(transparent)]
pub struct AcceleratorClass(u16);

impl AcceleratorClass {
    pub const OTHER: Self = Self(0);
    pub const NPU: Self = Self(1);
    pub const GPU: Self = Self(2);
    pub const DSP: Self = Self(3);

    pub const fn new(value: u16) -> Self {
        Self(value)
    }

    pub const fn get(self) -> u16 {
        self.0
    }
}

bitflags! {
    /// Semantic capabilities exposed by a backend, independent of virtio feature negotiation.
    ///
    /// Capabilities describe which accelerator operations the backend can perform. They do not
    /// change the wire layout. A transport feature bit is required separately whenever enabling a
    /// capability would change descriptor framing or any other device/driver protocol behavior.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Capabilities: u64 {
        const HOST_VISIBLE_MEMORY = 1 << 0;
        const DEVICE_LOCAL_MEMORY = 1 << 1;
        const EVENT_CANCELLATION = 1 << 2;
        const EXTERNAL_MEMORY = 1 << 3;
        const SECURE_CONTEXTS = 1 << 4;
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeviceIdentity {
    pub uuid: [u8; 16],
    pub class: AcceleratorClass,
    pub vendor_id: u32,
    pub device_id: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeviceLimits {
    pub max_contexts: u32,
    pub max_queues_per_context: u32,
    pub max_buffers_per_context: u32,
    pub max_bindings_per_submission: u32,
    pub max_buffer_bytes: u64,
    pub max_artifact_bytes: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeviceInfo {
    pub identity: DeviceIdentity,
    pub capabilities: Capabilities,
    pub limits: DeviceLimits,
}

bitflags! {
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct ContextFlags: u32 {
        const SECURE = 1 << 0;
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ContextDesc {
    pub flags: ContextFlags,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum MemoryDomain {
    Host = 1,
    Device = 2,
    Shared = 3,
}

impl TryFrom<u8> for MemoryDomain {
    type Error = BackendError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(Self::Host),
            2 => Ok(Self::Device),
            3 => Ok(Self::Shared),
            _ => Err(BackendError::InvalidArgument),
        }
    }
}

bitflags! {
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct BufferUsage: u32 {
        const TRANSFER_SOURCE = 1 << 0;
        const TRANSFER_DESTINATION = 1 << 1;
        const PROGRAM_INPUT = 1 << 2;
        const PROGRAM_OUTPUT = 1 << 3;
        const MUTABLE_STATE = 1 << 4;
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BufferDesc {
    bytes: NonZeroU64,
    alignment: NonZeroU64,
    pub domain: MemoryDomain,
    pub usage: BufferUsage,
}

impl BufferDesc {
    pub fn new(
        bytes: u64,
        alignment: u64,
        domain: MemoryDomain,
        usage: BufferUsage,
    ) -> Result<Self, BackendError> {
        let bytes = NonZeroU64::new(bytes).ok_or(BackendError::InvalidArgument)?;
        let alignment = NonZeroU64::new(alignment).ok_or(BackendError::InvalidArgument)?;
        if !alignment.get().is_power_of_two() {
            return Err(BackendError::InvalidArgument);
        }
        Ok(Self {
            bytes,
            alignment,
            domain,
            usage,
        })
    }

    pub const fn bytes(self) -> u64 {
        self.bytes.get()
    }

    pub const fn alignment(self) -> u64 {
        self.alignment.get()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BufferRange {
    pub offset: u64,
    bytes: NonZeroU64,
}

impl BufferRange {
    pub fn new(offset: u64, bytes: u64) -> Result<Self, BackendError> {
        let bytes = NonZeroU64::new(bytes).ok_or(BackendError::InvalidArgument)?;
        offset
            .checked_add(bytes.get())
            .ok_or(BackendError::OutOfBounds)?;
        Ok(Self { offset, bytes })
    }

    pub const fn bytes(self) -> u64 {
        self.bytes.get()
    }

    pub const fn end(self) -> u64 {
        self.offset + self.bytes.get()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum AccessMode {
    Read = 1,
    Write = 2,
    ReadWrite = 3,
}

impl TryFrom<u8> for AccessMode {
    type Error = BackendError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(Self::Read),
            2 => Ok(Self::Write),
            3 => Ok(Self::ReadWrite),
            _ => Err(BackendError::InvalidArgument),
        }
    }
}

/// Opaque, provider-owned executable format identifier.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(transparent)]
pub struct ArtifactFormat(NonZeroU32);

impl ArtifactFormat {
    pub const fn new(value: u32) -> Option<Self> {
        match NonZeroU32::new(value) {
            Some(value) => Some(Self(value)),
            None => None,
        }
    }

    pub const fn get(self) -> u32 {
        self.0.get()
    }
}

/// Opaque target words. Their schema belongs to the artifact format, not this crate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(transparent)]
pub struct TargetIdentity(pub [u32; 12]);

#[derive(Clone, Copy, Debug)]
pub struct ArtifactRef<'a> {
    pub format: ArtifactFormat,
    pub target: TargetIdentity,
    pub payload: &'a [u8],
    pub resident_bytes: u64,
}

bitflags! {
    /// Flags for an accelerator execution queue.
    ///
    /// This queue is a backend object used to submit programs. It is not a virtqueue; the v1
    /// protocol uses the term *command virtqueue* for the transport queue carrying requests.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct QueueFlags: u32 {
        const IN_ORDER = 1 << 0;
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct QueueDesc {
    pub flags: QueueFlags,
}

/// A relative timeout measured from backend admission. Zero on the wire means infinite.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Timeout {
    Infinite,
    AfterNs(NonZeroU64),
}

impl Timeout {
    pub const fn from_wire_ns(value: u64) -> Self {
        match NonZeroU64::new(value) {
            Some(value) => Self::AfterNs(value),
            None => Self::Infinite,
        }
    }

    pub const fn to_wire_ns(self) -> u64 {
        match self {
            Self::Infinite => 0,
            Self::AfterNs(value) => value.get(),
        }
    }
}

/// One validated binding. The referenced buffer must remain alive until its event is reclaimed.
#[derive(Debug)]
pub struct BindingRef<'a, B> {
    pub slot: u32,
    pub buffer: &'a B,
    pub range: BufferRange,
    pub access: AccessMode,
}

pub fn validate_bindings<B>(
    bindings: &[BindingRef<'_, B>],
    max_bindings: u32,
) -> Result<(), BackendError> {
    if bindings.is_empty() || bindings.len() > max_bindings as usize {
        return Err(BackendError::ResourceLimit);
    }
    for (index, binding) in bindings.iter().enumerate() {
        if bindings[..index]
            .iter()
            .any(|prior| prior.slot == binding.slot)
        {
            return Err(BackendError::InvalidArgument);
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EventState {
    Pending,
    Complete,
    Failed(BackendError),
    Cancelled,
}

/// Submission failure that makes the provider acceptance boundary explicit.
#[derive(Debug)]
pub enum SubmitFailure<E> {
    /// The backend guarantees execution was not accepted and no resources were retained.
    Rejected(BackendError),
    /// Acceptance is uncertain; the event owns the resources until it reaches a terminal state.
    Indeterminate { error: BackendError, event: E },
}

/// Failure to release a provider handle.
#[derive(Debug)]
pub enum ReleaseFailure<R> {
    /// The backend rejected the release and returns the still-live resource for retry.
    Rejected { error: BackendError, resource: R },
    /// The resource state is unknown. The adapter must invalidate its ID and request device reset.
    Indeterminate { error: BackendError },
}

impl<R> ReleaseFailure<R> {
    pub const fn error(&self) -> BackendError {
        match self {
            Self::Rejected { error, .. } | Self::Indeterminate { error } => *error,
        }
    }
}

/// Native accelerator lifecycle. Handles remain provider-owned and statically dispatched.
///
/// Destructive methods consume handles. A transport adapter must reject parent destruction while
/// child objects or in-flight events still exist; it must not use `Drop` timing as protocol state.
///
/// [`Self::Queue`] is an accelerator execution queue. It must not be confused with the command
/// virtqueue used by a transport adapter to deliver protocol requests.
pub trait Accelerator {
    type Context;
    type Buffer;
    type Program;
    type Queue;
    type Event;

    fn device_info(&self) -> Result<DeviceInfo, BackendError>;
    fn create_context(&self, desc: ContextDesc) -> Result<Self::Context, BackendError>;
    fn destroy_context(&self, context: Self::Context) -> Result<(), ReleaseFailure<Self::Context>>;

    fn allocate_buffer(
        &self,
        context: &Self::Context,
        desc: BufferDesc,
    ) -> Result<Self::Buffer, BackendError>;
    fn write_buffer(
        &self,
        buffer: &Self::Buffer,
        offset: u64,
        data: &[u8],
    ) -> Result<(), BackendError>;
    fn read_buffer(
        &self,
        buffer: &Self::Buffer,
        offset: u64,
        data: &mut [u8],
    ) -> Result<(), BackendError>;
    fn free_buffer(&self, buffer: Self::Buffer) -> Result<(), ReleaseFailure<Self::Buffer>>;

    fn load_program(
        &self,
        context: &Self::Context,
        artifact: ArtifactRef<'_>,
    ) -> Result<Self::Program, BackendError>;
    fn unload_program(&self, program: Self::Program) -> Result<(), ReleaseFailure<Self::Program>>;

    fn create_queue(
        &self,
        context: &Self::Context,
        desc: QueueDesc,
    ) -> Result<Self::Queue, BackendError>;
    fn destroy_queue(&self, queue: Self::Queue) -> Result<(), ReleaseFailure<Self::Queue>>;

    fn submit(
        &self,
        queue: &Self::Queue,
        program: &Self::Program,
        bindings: &[BindingRef<'_, Self::Buffer>],
        timeout: Timeout,
    ) -> Result<Self::Event, SubmitFailure<Self::Event>>;
    fn poll_event(&self, event: &Self::Event) -> Result<EventState, BackendError>;
    fn cancel_event(&self, _event: &Self::Event) -> Result<(), BackendError> {
        Err(BackendError::Unsupported)
    }
    fn destroy_event(&self, event: Self::Event) -> Result<(), ReleaseFailure<Self::Event>>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buffer_descriptors_reject_invalid_alignment() {
        assert!(BufferDesc::new(1, 0, MemoryDomain::Host, BufferUsage::empty()).is_err());
        assert!(BufferDesc::new(1, 3, MemoryDomain::Host, BufferUsage::empty()).is_err());
        assert_eq!(
            BufferDesc::new(64, 16, MemoryDomain::Shared, BufferUsage::PROGRAM_INPUT)
                .unwrap()
                .alignment(),
            16
        );
    }

    #[test]
    fn bindings_are_nonempty_bounded_and_unique() {
        let buffer = ();
        let range = BufferRange::new(0, 16).unwrap();
        let binding = BindingRef {
            slot: 3,
            buffer: &buffer,
            range,
            access: AccessMode::Read,
        };
        assert!(validate_bindings(&[binding], 1).is_ok());

        let duplicate = [
            BindingRef {
                slot: 3,
                buffer: &buffer,
                range,
                access: AccessMode::Read,
            },
            BindingRef {
                slot: 3,
                buffer: &buffer,
                range,
                access: AccessMode::Write,
            },
        ];
        assert_eq!(
            validate_bindings(&duplicate, 2),
            Err(BackendError::InvalidArgument)
        );
        assert_eq!(
            validate_bindings::<()>(&[], 1),
            Err(BackendError::ResourceLimit)
        );
    }

    #[test]
    fn wire_timeouts_are_relative_and_zero_is_infinite() {
        assert_eq!(Timeout::from_wire_ns(0), Timeout::Infinite);
        assert_eq!(Timeout::from_wire_ns(42).to_wire_ns(), 42);
    }
}
