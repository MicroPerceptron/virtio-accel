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
        /// Supports [`MemoryDomain::Host`] allocations.
        const HOST_VISIBLE_MEMORY = 1 << 0;
        /// Supports [`MemoryDomain::Device`] allocations.
        const DEVICE_LOCAL_MEMORY = 1 << 1;
        const EVENT_CANCELLATION = 1 << 2;
        /// Reserved for post-v1 external-allocation import/export semantics.
        const EXTERNAL_MEMORY = 1 << 3;
        /// Reserved until secure-context isolation requirements are specified.
        const SECURE_CONTEXTS = 1 << 4;
        /// Supports provider-owned [`MemoryDomain::Shared`] allocations.
        const SHARED_MEMORY = 1 << 5;
    }
}

impl Capabilities {
    /// Whether the backend can allocate the requested provider-owned memory domain.
    pub const fn supports_memory_domain(self, domain: MemoryDomain) -> bool {
        match domain {
            MemoryDomain::Host => self.contains(Self::HOST_VISIBLE_MEMORY),
            MemoryDomain::Device => self.contains(Self::DEVICE_LOCAL_MEMORY),
            MemoryDomain::Shared => self.contains(Self::SHARED_MEMORY),
        }
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
    pub max_buffers_per_context: u32,
    pub max_programs_per_context: u32,
    pub max_queues_per_context: u32,
    pub max_events_per_context: u32,
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
    /// Provider memory optimized for host transfers.
    ///
    /// If the usage includes program access, the returned allocation is still directly bindable;
    /// this value never permits per-submission staging.
    Host = 1,
    /// Provider memory optimized for accelerator access.
    ///
    /// Explicit read/write transfers may stage through provider-owned temporary memory.
    Device = 2,
    /// One provider-owned allocation that is host visible and directly accelerator bindable.
    ///
    /// This does not imply cross-process export, guest-memory import, cache coherence, or any
    /// platform external-memory handle.
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
        /// The buffer may be the source of an explicit [`Accelerator::read_buffer`] transfer.
        const TRANSFER_SOURCE = 1 << 0;
        /// The buffer may be the destination of an explicit [`Accelerator::write_buffer`] transfer.
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
        if !alignment.get().is_power_of_two()
            || usage.is_empty()
            || !BufferUsage::all().contains(usage)
        {
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

    /// Whether this allocation can appear in a program binding.
    pub const fn is_program_visible(self) -> bool {
        self.usage.intersects(
            BufferUsage::PROGRAM_INPUT
                .union(BufferUsage::PROGRAM_OUTPUT)
                .union(BufferUsage::MUTABLE_STATE),
        )
    }
}

bitflags! {
    /// Properties of the actual provider allocation returned for a [`BufferDesc`].
    ///
    /// These properties describe the backing allocation, not an aspirational fast path. A backend
    /// must reject allocation rather than advertise a property that it can satisfy only by
    /// allocating and copying a full-size bounce buffer during submission.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct BufferProperties: u32 {
        /// The provider can access the allocation through a host mapping.
        const HOST_VISIBLE = 1 << 0;
        /// The allocation uses the provider's accelerator-local placement class.
        const DEVICE_LOCAL = 1 << 1;
        /// Compatible program submissions bind this exact allocation without copying the bound
        /// byte range into or out of a different allocation.
        const DIRECT_BINDING = 1 << 2;
    }
}

/// Verified properties of one provider allocation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BufferInfo {
    desc: BufferDesc,
    allocation_bytes: NonZeroU64,
    alignment: NonZeroU64,
    properties: BufferProperties,
}

impl BufferInfo {
    /// Validate that actual allocation properties honestly satisfy the requested descriptor.
    pub fn new(
        desc: BufferDesc,
        allocation_bytes: u64,
        alignment: u64,
        properties: BufferProperties,
    ) -> Result<Self, BackendError> {
        let allocation_bytes =
            NonZeroU64::new(allocation_bytes).ok_or(BackendError::InvalidArgument)?;
        let alignment = NonZeroU64::new(alignment).ok_or(BackendError::InvalidArgument)?;
        if !BufferProperties::all().contains(properties) {
            return Err(BackendError::InvalidArgument);
        }
        if allocation_bytes.get() < desc.bytes()
            || !alignment.get().is_power_of_two()
            || alignment.get() < desc.alignment()
        {
            return Err(BackendError::Incompatible);
        }

        let required = match desc.domain {
            MemoryDomain::Host => BufferProperties::HOST_VISIBLE,
            MemoryDomain::Device => BufferProperties::DEVICE_LOCAL,
            MemoryDomain::Shared => {
                BufferProperties::HOST_VISIBLE.union(BufferProperties::DIRECT_BINDING)
            }
        };
        if !properties.contains(required)
            || (desc.is_program_visible() && !properties.contains(BufferProperties::DIRECT_BINDING))
        {
            return Err(BackendError::Incompatible);
        }

        Ok(Self {
            desc,
            allocation_bytes,
            alignment,
            properties,
        })
    }

    pub const fn desc(self) -> BufferDesc {
        self.desc
    }

    /// Physical/provider backing bytes retained for this logical buffer.
    pub const fn allocation_bytes(self) -> u64 {
        self.allocation_bytes.get()
    }

    /// Alignment guaranteed by the actual provider allocation.
    pub const fn alignment(self) -> u64 {
        self.alignment.get()
    }

    pub const fn properties(self) -> BufferProperties {
        self.properties
    }
}

impl DeviceInfo {
    /// Validate allocation size and memory-domain support before backend invocation.
    pub fn validate_buffer_desc(self, desc: BufferDesc) -> Result<(), BackendError> {
        if desc.bytes() > self.limits.max_buffer_bytes {
            return Err(BackendError::ResourceLimit);
        }
        if !self.capabilities.supports_memory_domain(desc.domain) {
            return Err(BackendError::Unsupported);
        }
        Ok(())
    }

    /// Validate that a backend allocation describes the request it was asked to satisfy.
    pub fn validate_buffer_info(
        self,
        requested: BufferDesc,
        actual: BufferInfo,
    ) -> Result<(), BackendError> {
        self.validate_buffer_desc(requested)?;
        if actual.desc() != requested {
            return Err(BackendError::Incompatible);
        }
        Ok(())
    }
}

/// A newly allocated native buffer handle and its verified backing properties.
///
/// Device implementations should retain `info` in their object record and pass only `buffer` to
/// backend hot paths.
#[derive(Debug)]
pub struct AllocatedBuffer<B> {
    buffer: B,
    info: BufferInfo,
}

impl<B> AllocatedBuffer<B> {
    pub const fn new(buffer: B, info: BufferInfo) -> Self {
        Self { buffer, info }
    }

    pub const fn buffer(&self) -> &B {
        &self.buffer
    }

    pub fn buffer_mut(&mut self) -> &mut B {
        &mut self.buffer
    }

    pub const fn info(&self) -> BufferInfo {
        self.info
    }

    pub fn into_parts(self) -> (B, BufferInfo) {
        (self.buffer, self.info)
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

/// A bounded byte source that may be physically segmented.
///
/// Transport adapters can implement this trait over validated descriptor-backed regions so
/// providers can read directly into final program or buffer storage without first coalescing the
/// complete payload. Every range fully contained in `0..len()` must be readable for the duration of
/// the backend call. The optional contiguous view preserves the single-slice fast path.
pub trait ByteSource: fmt::Debug {
    /// Stable logical length of this source.
    fn len(&self) -> u64;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Fill `target` from the exact logical range beginning at `offset`.
    fn read_at(&self, offset: u64, target: &mut [u8]) -> Result<(), BackendError>;

    /// Borrow the complete logical source when it is one contiguous region.
    ///
    /// A returned slice has length [`Self::len`] and contains the same bytes as `read_at`.
    fn as_contiguous(&self) -> Option<&[u8]> {
        None
    }
}

impl ByteSource for [u8] {
    fn len(&self) -> u64 {
        self.len() as u64
    }

    fn read_at(&self, offset: u64, target: &mut [u8]) -> Result<(), BackendError> {
        let start = usize::try_from(offset).map_err(|_| BackendError::OutOfBounds)?;
        let end = start
            .checked_add(target.len())
            .filter(|end| *end <= self.len())
            .ok_or(BackendError::OutOfBounds)?;
        target.copy_from_slice(&self[start..end]);
        Ok(())
    }

    fn as_contiguous(&self) -> Option<&[u8]> {
        Some(self)
    }
}

impl<const N: usize> ByteSource for [u8; N] {
    fn len(&self) -> u64 {
        N as u64
    }

    fn read_at(&self, offset: u64, target: &mut [u8]) -> Result<(), BackendError> {
        self.as_slice().read_at(offset, target)
    }

    fn as_contiguous(&self) -> Option<&[u8]> {
        Some(self)
    }
}

/// A bounded byte destination that may be physically segmented.
///
/// Providers can write buffer contents directly into validated response regions. The optional
/// contiguous view avoids callback overhead when the destination is already one slice. Every range
/// fully contained in `0..len()` must be writable for the duration of the backend call.
pub trait ByteSink: fmt::Debug {
    /// Stable logical length of this destination.
    fn len(&self) -> u64;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Write `source` to the exact logical range beginning at `offset`.
    fn write_at(&mut self, offset: u64, source: &[u8]) -> Result<(), BackendError>;

    /// Mutably borrow the complete logical destination when it is one contiguous region.
    ///
    /// A returned slice has length [`Self::len`] and represents the same bytes as `write_at`.
    fn as_contiguous_mut(&mut self) -> Option<&mut [u8]> {
        None
    }
}

impl ByteSink for [u8] {
    fn len(&self) -> u64 {
        self.len() as u64
    }

    fn write_at(&mut self, offset: u64, source: &[u8]) -> Result<(), BackendError> {
        let start = usize::try_from(offset).map_err(|_| BackendError::OutOfBounds)?;
        let end = start
            .checked_add(source.len())
            .filter(|end| *end <= self.len())
            .ok_or(BackendError::OutOfBounds)?;
        self[start..end].copy_from_slice(source);
        Ok(())
    }

    fn as_contiguous_mut(&mut self) -> Option<&mut [u8]> {
        Some(self)
    }
}

impl<const N: usize> ByteSink for [u8; N] {
    fn len(&self) -> u64 {
        N as u64
    }

    fn write_at(&mut self, offset: u64, source: &[u8]) -> Result<(), BackendError> {
        self.as_mut_slice().write_at(offset, source)
    }

    fn as_contiguous_mut(&mut self) -> Option<&mut [u8]> {
        Some(self)
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

/// Borrowed program artifact envelope.
///
/// Payload bytes may be segmented; providers should stream them into final resident storage or use
/// [`ByteSource::as_contiguous`] when a borrowed slice is available.
#[derive(Clone, Copy, Debug)]
pub struct ArtifactRef<'a> {
    pub format: ArtifactFormat,
    pub target: TargetIdentity,
    pub payload: &'a dyn ByteSource,
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
///
/// Program-visible buffers carry [`BufferProperties::DIRECT_BINDING`]. A backend must reject an
/// incompatible buffer/program combination instead of copying the range into a hidden bounce
/// allocation.
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
/// The only baseline operations that explicitly transfer buffer contents are [`Self::write_buffer`]
/// and [`Self::read_buffer`]. Allocation, submission, polling, and release must not hide full-range
/// staging copies. In particular, `submit` binds the provider allocation directly or rejects it as
/// [`BackendError::Incompatible`].
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
    ) -> Result<AllocatedBuffer<Self::Buffer>, BackendError>;
    /// Perform one explicit host-to-buffer transfer.
    ///
    /// The command engine calls this only for buffers with [`BufferUsage::TRANSFER_DESTINATION`].
    /// The provider must not retain `data`. A segmented source should be read directly into final
    /// backing rather than coalesced. Host-visible allocations should copy directly into their
    /// final backing; device-local allocations may use bounded temporary staging during this
    /// explicit transfer.
    fn write_buffer(
        &self,
        buffer: &mut Self::Buffer,
        offset: u64,
        data: &dyn ByteSource,
    ) -> Result<(), BackendError>;
    /// Perform one explicit buffer-to-host transfer.
    ///
    /// The command engine calls this only for buffers with [`BufferUsage::TRANSFER_SOURCE`]. The
    /// provider must not retain `data` and should write directly across segmented destinations.
    /// Returning `Ok(())` guarantees that every byte in `data` was initialized; the command engine
    /// may publish the complete destination without first clearing it.
    fn read_buffer(
        &self,
        buffer: &Self::Buffer,
        offset: u64,
        data: &mut dyn ByteSink,
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
        assert!(BufferDesc::new(1, 3, MemoryDomain::Host, BufferUsage::TRANSFER_SOURCE).is_err());
        assert!(BufferDesc::new(1, 1, MemoryDomain::Host, BufferUsage::empty()).is_err());
        assert_eq!(
            BufferDesc::new(64, 16, MemoryDomain::Shared, BufferUsage::PROGRAM_INPUT)
                .unwrap()
                .alignment(),
            16
        );
    }

    #[test]
    fn allocation_properties_reject_hidden_submission_staging() {
        let host_input =
            BufferDesc::new(64, 16, MemoryDomain::Host, BufferUsage::PROGRAM_INPUT).unwrap();
        assert_eq!(
            BufferInfo::new(host_input, 64, 16, BufferProperties::HOST_VISIBLE),
            Err(BackendError::Incompatible)
        );
        assert!(
            BufferInfo::new(
                host_input,
                64,
                16,
                BufferProperties::HOST_VISIBLE | BufferProperties::DIRECT_BINDING
            )
            .is_ok()
        );

        let shared =
            BufferDesc::new(64, 16, MemoryDomain::Shared, BufferUsage::TRANSFER_SOURCE).unwrap();
        assert_eq!(
            BufferInfo::new(shared, 64, 16, BufferProperties::HOST_VISIBLE),
            Err(BackendError::Incompatible)
        );
        assert_eq!(
            BufferInfo::new(
                shared,
                63,
                16,
                BufferProperties::HOST_VISIBLE | BufferProperties::DIRECT_BINDING
            ),
            Err(BackendError::Incompatible)
        );
        assert_eq!(
            BufferInfo::new(
                shared,
                64,
                8,
                BufferProperties::HOST_VISIBLE | BufferProperties::DIRECT_BINDING
            ),
            Err(BackendError::Incompatible)
        );
    }

    #[test]
    fn capabilities_report_memory_domains_independently() {
        let capabilities = Capabilities::HOST_VISIBLE_MEMORY | Capabilities::SHARED_MEMORY;
        assert!(capabilities.supports_memory_domain(MemoryDomain::Host));
        assert!(capabilities.supports_memory_domain(MemoryDomain::Shared));
        assert!(!capabilities.supports_memory_domain(MemoryDomain::Device));
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
