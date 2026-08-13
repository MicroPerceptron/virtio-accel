//! Transport-independent accelerator semantics.
//!
//! This crate does not name virtqueues, guest memory, host operating systems, or vendor APIs.
//! Transport adapters validate untrusted input and translate it into these typed contracts.

#![no_std]
#![forbid(unsafe_code)]

use bitflags::bitflags;
use core::fmt;
use core::num::{NonZeroU32, NonZeroU64};
use virtio_accel_transport::{ByteAccessError, ReadableBytes, WritableBytes};

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
        /// [`Accelerator::cancel_event`] is implemented for pending events.
        const EVENT_CANCELLATION = 1 << 2;
        /// Reserved for post-v1 external-allocation import/export semantics.
        const EXTERNAL_MEMORY = 1 << 3;
        /// Reserved until secure-context isolation requirements are specified.
        const SECURE_CONTEXTS = 1 << 4;
        /// Supports provider-owned [`MemoryDomain::Shared`] allocations.
        const SHARED_MEMORY = 1 << 5;

        /// Capabilities that make at least one provider-owned memory domain usable.
        const MEMORY_DOMAINS = Self::HOST_VISIBLE_MEMORY.bits()
            | Self::DEVICE_LOCAL_MEMORY.bits()
            | Self::SHARED_MEMORY.bits();
        /// Assigned bits whose semantics remain reserved by this version of the contract.
        const RESERVED = Self::EXTERNAL_MEMORY.bits() | Self::SECURE_CONTEXTS.bits();
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

/// Invalid provider metadata discovered before any resource operation is invoked.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceInfoError {
    /// A capability whose semantics are still reserved was advertised.
    ReservedCapabilities,
    /// No provider-owned memory domain can be allocated.
    MissingMemoryDomain,
    /// A mandatory resource or byte limit is zero.
    ZeroLimit,
}

bitflags! {
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct ContextFlags: u32 {
        /// Reserved until secure-context isolation and transport semantics are specified.
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

    /// Whether this declaration permits one program binding access mode.
    pub const fn allows_access(self, access: AccessMode) -> bool {
        match access {
            AccessMode::Read => self
                .usage
                .intersects(BufferUsage::PROGRAM_INPUT.union(BufferUsage::MUTABLE_STATE)),
            AccessMode::Write => self
                .usage
                .intersects(BufferUsage::PROGRAM_OUTPUT.union(BufferUsage::MUTABLE_STATE)),
            AccessMode::ReadWrite => self.usage.contains(BufferUsage::MUTABLE_STATE),
        }
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
    /// Validate immutable provider metadata once, before constructing live object state.
    ///
    /// Unknown capability bits remain representable for forward-compatible diagnostics. Assigned
    /// reserved bits are rejected because this version cannot enforce their ownership and
    /// synchronization rules.
    pub const fn validate(self) -> Result<(), DeviceInfoError> {
        if self.capabilities.intersects(Capabilities::RESERVED) {
            return Err(DeviceInfoError::ReservedCapabilities);
        }
        if !self.capabilities.intersects(Capabilities::MEMORY_DOMAINS) {
            return Err(DeviceInfoError::MissingMemoryDomain);
        }
        if self.limits.max_contexts == 0
            || self.limits.max_buffers_per_context == 0
            || self.limits.max_programs_per_context == 0
            || self.limits.max_queues_per_context == 0
            || self.limits.max_events_per_context == 0
            || self.limits.max_bindings_per_submission == 0
            || self.limits.max_buffer_bytes == 0
            || self.limits.max_artifact_bytes == 0
        {
            return Err(DeviceInfoError::ZeroLimit);
        }
        Ok(())
    }

    /// Validate context intent before backend invocation.
    ///
    /// This contract currently reserves every nonempty context flag set.
    pub fn validate_context_desc(self, desc: ContextDesc) -> Result<(), BackendError> {
        if desc.flags.is_empty() {
            Ok(())
        } else {
            Err(BackendError::Unsupported)
        }
    }

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

    /// Validate execution-queue intent before backend invocation.
    ///
    /// This contract currently reserves every nonempty execution-queue flag set.
    pub fn validate_queue_desc(self, desc: QueueDesc) -> Result<(), BackendError> {
        if desc.flags.is_empty() {
            Ok(())
        } else {
            Err(BackendError::Unsupported)
        }
    }

    /// Validate event-cancellation support before backend invocation.
    pub fn validate_event_cancellation(self) -> Result<(), BackendError> {
        if self.capabilities.contains(Capabilities::EVENT_CANCELLATION) {
            Ok(())
        } else {
            Err(BackendError::Unsupported)
        }
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

/// Zero-copy core byte-source adapter over a transport-owned readable port.
#[derive(Debug)]
pub struct TransportByteSource<'a, T: ?Sized>(&'a T);

impl<'a, T: ?Sized> TransportByteSource<'a, T> {
    /// Borrow a transport-readable port without copying or coalescing its bytes.
    pub const fn new(source: &'a T) -> Self {
        Self(source)
    }

    /// Recover the wrapped transport port.
    pub const fn into_inner(self) -> &'a T {
        self.0
    }
}

impl<T: ReadableBytes + ?Sized> ByteSource for TransportByteSource<'_, T> {
    fn len(&self) -> u64 {
        ReadableBytes::len(self.0)
    }

    fn read_at(&self, offset: u64, target: &mut [u8]) -> Result<(), BackendError> {
        ReadableBytes::read_at(self.0, offset, target).map_err(backend_error_from_byte_access)
    }

    fn as_contiguous(&self) -> Option<&[u8]> {
        ReadableBytes::as_contiguous(self.0)
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
        ByteSource::read_at(self.as_slice(), offset, target)
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

/// Zero-copy core byte-sink adapter over a transport-owned writable port.
#[derive(Debug)]
pub struct TransportByteSink<'a, T: ?Sized>(&'a mut T);

impl<'a, T: ?Sized> TransportByteSink<'a, T> {
    /// Borrow a transport-writable port without copying or coalescing its bytes.
    pub const fn new(sink: &'a mut T) -> Self {
        Self(sink)
    }

    /// Recover the wrapped transport port.
    pub fn into_inner(self) -> &'a mut T {
        self.0
    }
}

impl<T: WritableBytes + ?Sized> ByteSink for TransportByteSink<'_, T> {
    fn len(&self) -> u64 {
        WritableBytes::len(self.0)
    }

    fn write_at(&mut self, offset: u64, source: &[u8]) -> Result<(), BackendError> {
        WritableBytes::write_at(self.0, offset, source).map_err(backend_error_from_byte_access)
    }

    fn as_contiguous_mut(&mut self) -> Option<&mut [u8]> {
        WritableBytes::as_contiguous_mut(self.0)
    }
}

const fn backend_error_from_byte_access(error: ByteAccessError) -> BackendError {
    match error {
        ByteAccessError::OutOfBounds => BackendError::OutOfBounds,
        ByteAccessError::Busy | ByteAccessError::Reset => BackendError::Busy,
        ByteAccessError::Access => BackendError::DeviceLost,
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
        ByteSink::write_at(self.as_mut_slice(), offset, source)
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
/// [`ByteSource::as_contiguous`] when a borrowed slice is available. `resident_bytes` is the
/// caller-authorized upper bound for all provider storage retained by the returned program; a
/// provider must reject the artifact if it cannot stay within that charge.
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
        /// Reserved until ordering behavior and capability negotiation are specified.
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

/// One borrowed program binding. The referenced buffer must remain alive until its event is
/// reclaimed.
///
/// Program-visible buffers carry [`BufferProperties::DIRECT_BINDING`]. A backend must reject an
/// incompatible buffer/program combination instead of copying the range into a hidden bounce
/// allocation. Binding order is not semantic; command engines may present the slice in slot order.
///
/// Before [`Accelerator::submit`], hosts must reject an [`AccessMode`] incompatible with the
/// buffer's declared [`BufferUsage`] (see [`Self::validate_for_submit`]).
#[derive(Debug)]
pub struct BindingRef<'a, B> {
    pub slot: u32,
    pub buffer: &'a B,
    pub range: BufferRange,
    pub access: AccessMode,
}

impl<'a, B> BindingRef<'a, B> {
    /// Slot/count checks plus [`BufferDesc::allows_access`] for each binding.
    ///
    /// Hosts must enforce both checks before [`Accelerator::submit`]. This combined
    /// helper is suitable when descriptors are already available as a slice. A host
    /// resolving descriptors individually may call [`validate_bindings`] once and
    /// [`BufferDesc::allows_access`] as each buffer is resolved, avoiding a descriptor
    /// mirror allocation. `descs[i]` must be the descriptor for the buffer behind
    /// `bindings[i].buffer` (equal length alone is not enough). A usage mismatch
    /// returns [`BackendError::PermissionDenied`].
    pub fn validate_for_submit(
        bindings: &[Self],
        descs: &[BufferDesc],
        max_bindings: u32,
    ) -> Result<(), BackendError> {
        validate_bindings(bindings, max_bindings)?;
        if bindings.len() != descs.len() {
            return Err(BackendError::InvalidArgument);
        }
        for (binding, desc) in bindings.iter().zip(descs.iter()) {
            if !desc.allows_access(binding.access) {
                return Err(BackendError::PermissionDenied);
            }
        }
        Ok(())
    }
}

/// Slot/count uniqueness helper for program bindings.
///
/// Structural only: nonempty, bounded by `max_bindings`, and unique slots. This is
/// **incomplete** for pre-admission checks -- it does not enforce access/usage
/// compatibility required before backend admission. Prefer
/// [`BindingRef::validate_for_submit`] before [`Accelerator::submit`]. Strictly
/// slot-ordered input takes a linear, allocation-free path; arbitrary order remains
/// supported by the allocation-free fallback.
pub fn validate_bindings<B>(
    bindings: &[BindingRef<'_, B>],
    max_bindings: u32,
) -> Result<(), BackendError> {
    if bindings.is_empty() || bindings.len() > max_bindings as usize {
        return Err(BackendError::ResourceLimit);
    }

    // The wire decoder canonicalizes bindings into slot order. Recognizing that
    // invariant here avoids a second quadratic uniqueness pass on every host
    // submission while preserving the public API's order-independent semantics.
    if bindings.windows(2).all(|pair| pair[0].slot < pair[1].slot) {
        return Ok(());
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

/// Native accelerator lifecycle over provider-owned handle types.
///
/// The reference command engine is generic over this trait, so its calls are statically dispatched
/// and native handles need no boxing. The trait imposes no `Send` or `Sync` bounds: a provider may
/// preserve thread-affine handles, while a provider that opts into those auto traits must make the
/// corresponding shared calls safe. Callers must not overlap a mutable borrow, a consumed handle,
/// or destruction with another use of the same resource.
///
/// Borrowed arguments are valid only for the duration of a call and must not be retained as Rust
/// references. Destructive methods consume handles. A caller must reject parent destruction while
/// child objects or in-flight events still exist; it must not use `Drop` timing as lifecycle state.
///
/// The only operations that explicitly transfer buffer contents are [`Self::write_buffer`] and
/// [`Self::read_buffer`]. Allocation, submission, polling, and release must not hide full-range
/// staging copies. In particular, `submit` binds the exact provider allocation directly or rejects
/// it as [`BackendError::Incompatible`].
///
/// Dynamic loading, a stable binary interface, and erased cross-boundary handle ownership are not
/// defined here. An integration that needs dynamic dispatch must fix one concrete handle family in
/// an adapter without weakening this trait's borrowing, acceptance, or release contracts.
pub trait Accelerator {
    /// Owned context handle. It may be a native value and need not be boxed, cloneable, or thread
    /// safe.
    type Context;
    /// Owned handle for the exact allocation described by its accompanying [`BufferInfo`].
    type Buffer;
    /// Owned resident-program handle with no borrow of its source artifact.
    type Program;
    /// Owned accelerator execution-queue handle.
    type Queue;
    /// Owned submission and completion token with no borrow of the submitted binding slice.
    type Event;

    /// Return immutable identity, capability, and limit metadata.
    ///
    /// - **Ownership/lifetime:** no ownership changes; a successful value must remain stable for
    ///   the lifetime of this backend instance.
    /// - **Progress/concurrency:** discovery may perform bounded synchronous provider work but must
    ///   not wait for resource progress. Concurrent calls are permitted only when the concrete
    ///   backend is `Sync`.
    /// - **Failure/retry:** an error creates no resource and may be retried; callers validate and
    ///   cache the first successful result before invoking resource methods.
    /// - **Allocation/copies:** the call must not allocate resource backing or copy bulk content.
    fn device_info(&self) -> Result<DeviceInfo, BackendError>;

    /// Create one context from prevalidated intent.
    ///
    /// - **Ownership/lifetime:** `desc` is consumed by value and not retained by reference; success
    ///   returns one owned context. All current nonempty context flags are unsupported.
    /// - **Progress/concurrency:** provider setup may synchronously block, but must not wait for
    ///   unrelated resource progress. Independent creation may overlap only when concrete types
    ///   permit it.
    /// - **Failure/retry:** `Err` guarantees that no context resource was retained and the request
    ///   may be retried.
    /// - **Allocation/copies:** context bookkeeping may be allocated; no buffer content is copied.
    fn create_context(&self, desc: ContextDesc) -> Result<Self::Context, BackendError>;

    /// Destroy an empty context.
    ///
    /// - **Ownership/lifetime:** the handle is consumed and must have no live child resources.
    /// - **Progress/concurrency:** release may synchronously block, but must not wait for children
    ///   or in-flight work; no use of this context may overlap the call.
    /// - **Failure/retry:** [`ReleaseFailure::Rejected`] returns the live handle for retry;
    ///   [`ReleaseFailure::Indeterminate`] invalidates it and forbids retry.
    /// - **Allocation/copies:** the call releases provider bookkeeping and copies no content.
    fn destroy_context(&self, context: Self::Context) -> Result<(), ReleaseFailure<Self::Context>>;

    /// Allocate one exact provider-owned buffer backing.
    ///
    /// - **Ownership/lifetime:** `context` is borrowed only for this call. Success returns an owned
    ///   handle plus metadata for the actual backing; neither may borrow `context`.
    /// - **Progress/concurrency:** allocation may synchronously block. Independent contexts may be
    ///   used concurrently only when the concrete backend and handles permit it.
    /// - **Failure/retry:** `Err` guarantees that no buffer backing was retained and may be retried.
    /// - **Allocation/copies:** this is the buffer-allocation boundary. Program-visible requests
    ///   allocate directly bindable backing here or fail; they must not reserve a submission-time
    ///   bounce allocation or copy buffer content.
    fn allocate_buffer(
        &self,
        context: &Self::Context,
        desc: BufferDesc,
    ) -> Result<AllocatedBuffer<Self::Buffer>, BackendError>;

    /// Perform one explicit host-to-buffer transfer.
    ///
    /// - **Ownership/lifetime:** `buffer` is exclusively borrowed and `data` is borrowed only for
    ///   this call. The provider must not retain either reference.
    /// - **Progress/concurrency:** the call may synchronously block until the explicit transfer is
    ///   complete. The exclusive buffer borrow prevents overlapping access without forcing
    ///   interior synchronization; unrelated buffers may progress when concrete types permit it.
    /// - **Failure/retry:** on `Err`, the requested range may be partially modified but the handle
    ///   remains live. A later successful full-range write replaces it; device loss is not
    ///   retryable on the same backend instance.
    /// - **Allocation/copies:** this is an explicit content-copy boundary. Segmented input should
    ///   flow into final backing without frame-sized coalescing. Device-local backing may use
    ///   bounded temporary staging during this call.
    fn write_buffer(
        &self,
        buffer: &mut Self::Buffer,
        offset: u64,
        data: &dyn ByteSource,
    ) -> Result<(), BackendError>;
    /// Perform one explicit buffer-to-host transfer.
    ///
    /// - **Ownership/lifetime:** `buffer` is shared-borrowed and `data` is exclusively borrowed only
    ///   for this call. The provider must not retain either reference.
    /// - **Progress/concurrency:** the call may synchronously block until the explicit transfer is
    ///   complete. Shared reads may overlap only when the concrete buffer is `Sync` and the
    ///   provider supports that access.
    /// - **Failure/retry:** `Err` leaves the destination potentially partially initialized; the
    ///   caller must not publish it. The buffer is unchanged and a complete read may be retried
    ///   unless the backend is lost. `Ok(())` guarantees every destination byte was initialized.
    /// - **Allocation/copies:** this is an explicit content-copy boundary. The provider should
    ///   write directly across segmented destinations; device-local backing may use bounded
    ///   temporary staging during this call.
    fn read_buffer(
        &self,
        buffer: &Self::Buffer,
        offset: u64,
        data: &mut dyn ByteSink,
    ) -> Result<(), BackendError>;

    /// Release an unreferenced buffer and its exact backing allocation.
    ///
    /// - **Ownership/lifetime:** the handle is consumed and must not be bound to an in-flight event.
    /// - **Progress/concurrency:** release may synchronously block but must not wait for references
    ///   to disappear; no access to this buffer may overlap the call.
    /// - **Failure/retry:** rejected release returns the live handle for retry; indeterminate
    ///   release invalidates it and requires recovery.
    /// - **Allocation/copies:** backing is deallocated without copying its contents or allocating a
    ///   replacement.
    fn free_buffer(&self, buffer: Self::Buffer) -> Result<(), ReleaseFailure<Self::Buffer>>;

    /// Create a resident program from an opaque, possibly segmented artifact.
    ///
    /// - **Ownership/lifetime:** `context`, `artifact.payload`, and the envelope are borrowed only
    ///   for this call. Success returns an owned program with no source borrow.
    /// - **Progress/concurrency:** program creation may synchronously block. Independent lifecycle
    ///   work may overlap only when the concrete backend and context permit it.
    /// - **Failure/retry:** `Err` guarantees that no program resource was retained and may be
    ///   retried with a still-live context and artifact.
    /// - **Allocation/copies:** resident program storage may be allocated but all storage retained
    ///   by the returned handle must fit `artifact.resident_bytes`. Segmented bytes should stream
    ///   into final resident storage rather than require one artifact-sized coalescing copy.
    fn load_program(
        &self,
        context: &Self::Context,
        artifact: ArtifactRef<'_>,
    ) -> Result<Self::Program, BackendError>;

    /// Release an unreferenced resident program.
    ///
    /// - **Ownership/lifetime:** the program is consumed and must not be referenced by an event.
    /// - **Progress/concurrency:** release may synchronously block but must not wait for in-flight
    ///   references; no use of this program may overlap the call.
    /// - **Failure/retry:** rejected release returns the live handle for retry; indeterminate
    ///   release invalidates it and requires recovery.
    /// - **Allocation/copies:** resident storage is released without copying buffer contents or
    ///   allocating replacement state.
    fn unload_program(&self, program: Self::Program) -> Result<(), ReleaseFailure<Self::Program>>;

    /// Create one accelerator execution queue.
    ///
    /// - **Ownership/lifetime:** `context` is borrowed only for this call and success returns an
    ///   owned queue. All current nonempty queue flags are unsupported.
    /// - **Progress/concurrency:** queue setup may synchronously block. Independent creation may
    ///   overlap only when concrete types permit it.
    /// - **Failure/retry:** `Err` guarantees that no queue resource was retained and may be retried.
    /// - **Allocation/copies:** queue bookkeeping may be allocated; no program or buffer content is
    ///   copied.
    fn create_queue(
        &self,
        context: &Self::Context,
        desc: QueueDesc,
    ) -> Result<Self::Queue, BackendError>;

    /// Release an unreferenced execution queue.
    ///
    /// - **Ownership/lifetime:** the queue is consumed and must not be referenced by an event.
    /// - **Progress/concurrency:** release may synchronously block but must not wait for submitted
    ///   work; no use of this queue may overlap the call.
    /// - **Failure/retry:** rejected release returns the live handle for retry; indeterminate
    ///   release invalidates it and requires recovery.
    /// - **Allocation/copies:** queue state is released without copying buffer content or allocating
    ///   replacement state.
    fn destroy_queue(&self, queue: Self::Queue) -> Result<(), ReleaseFailure<Self::Queue>>;

    /// Attempt to admit one program execution and return its event.
    ///
    /// Hosts must reject an [`AccessMode`] incompatible with each buffer's [`BufferUsage`] before
    /// calling this method (see [`BufferDesc::allows_access`] and
    /// [`BindingRef::validate_for_submit`]). Providers may repeat the check as defense in depth, but
    /// host-side rejection is required by Wire ABI section 4.4.
    ///
    /// - **Ownership/lifetime:** queue, program, buffers, and the binding slice are borrowed only
    ///   during admission and must not be retained as Rust references. The caller keeps every
    ///   referenced handle alive until the returned event is terminal and destroyed.
    /// - **Progress/concurrency:** synchronous work is limited to validation and admission; the call
    ///   must not wait for execution to finish. Concurrent submission requires concrete `Sync`
    ///   handles and provider support; the trait requires no lock or atomic operation by itself.
    /// - **Failure/retry:** [`SubmitFailure::Rejected`] guarantees no acceptance and permits retry.
    ///   Success or [`SubmitFailure::Indeterminate`] transfers invocation ownership to the event and
    ///   must not be retried as though rejected.
    /// - **Allocation/copies:** the borrowed slice requires no per-binding box or owned mirror.
    ///   Providers may use amortized event storage, but must directly bind each exact allocation and
    ///   reject incompatibility instead of allocating or copying through hidden bounce buffers.
    fn submit(
        &self,
        queue: &Self::Queue,
        program: &Self::Program,
        bindings: &[BindingRef<'_, Self::Buffer>],
        timeout: Timeout,
    ) -> Result<Self::Event, SubmitFailure<Self::Event>>;

    /// Observe event state without blocking or driving an executor.
    ///
    /// - **Ownership/lifetime:** the event is borrowed only for this call and remains live.
    /// - **Progress/concurrency:** polling is bounded, nonblocking, and safe to race with provider
    ///   completion when the concrete event is `Sync`.
    /// - **Failure/retry:** errors do not make an event terminal; polling may be retried unless the
    ///   backend is lost. Once observed, a terminal state is stable across every later success.
    /// - **Allocation/copies:** polling allocates no per-call state and copies no bulk content.
    fn poll_event(&self, event: &Self::Event) -> Result<EventState, BackendError>;

    /// Attempt to make a pending event terminal as [`EventState::Cancelled`].
    ///
    /// - **Ownership/lifetime:** the event is borrowed only for this call and remains live.
    /// - **Progress/concurrency:** cancellation is bounded and nonblocking. It may race with
    ///   completion; the provider chooses exactly one terminal result without requiring a lock in
    ///   the handle contract.
    /// - **Failure/retry:** `Ok(())` means cancellation won. [`BackendError::Busy`] means completion
    ///   won and the caller should poll. The default `Unsupported` implementation is conformant only
    ///   when [`Capabilities::EVENT_CANCELLATION`] is absent.
    /// - **Allocation/copies:** cancellation allocates no per-call state and copies no bulk content.
    fn cancel_event(&self, _event: &Self::Event) -> Result<(), BackendError> {
        Err(BackendError::Unsupported)
    }

    /// Release one terminal event and its provider invocation state.
    ///
    /// - **Ownership/lifetime:** the event is consumed. Every referenced queue, program, and buffer
    ///   must remain live until this release succeeds or becomes indeterminate.
    /// - **Progress/concurrency:** release may synchronously block but must not wait for a pending
    ///   event to finish; no poll or cancellation may overlap this call.
    /// - **Failure/retry:** rejected release returns the live event for retry; indeterminate release
    ///   invalidates it and requires recovery.
    /// - **Allocation/copies:** invocation state is released without copying buffer content or
    ///   allocating replacement state.
    fn destroy_event(&self, event: Self::Event) -> Result<(), ReleaseFailure<Self::Event>>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug)]
    struct TransportBytes([u8; 4]);

    impl ReadableBytes for TransportBytes {
        fn len(&self) -> u64 {
            self.0.as_slice().len() as u64
        }

        fn read_at(&self, offset: u64, target: &mut [u8]) -> Result<(), ByteAccessError> {
            let start = usize::try_from(offset).map_err(|_| ByteAccessError::OutOfBounds)?;
            let end = start
                .checked_add(target.len())
                .filter(|end| *end <= self.0.as_slice().len())
                .ok_or(ByteAccessError::OutOfBounds)?;
            target.copy_from_slice(&self.0[start..end]);
            Ok(())
        }
    }

    impl WritableBytes for TransportBytes {
        fn len(&self) -> u64 {
            self.0.as_slice().len() as u64
        }

        fn write_at(&mut self, offset: u64, source: &[u8]) -> Result<(), ByteAccessError> {
            let start = usize::try_from(offset).map_err(|_| ByteAccessError::OutOfBounds)?;
            let end = start
                .checked_add(source.len())
                .filter(|end| *end <= self.0.as_slice().len())
                .ok_or(ByteAccessError::OutOfBounds)?;
            self.0[start..end].copy_from_slice(source);
            Ok(())
        }
    }

    fn valid_device_info() -> DeviceInfo {
        DeviceInfo {
            identity: DeviceIdentity {
                uuid: [0; 16],
                class: AcceleratorClass::OTHER,
                vendor_id: 0,
                device_id: 0,
            },
            capabilities: Capabilities::HOST_VISIBLE_MEMORY,
            limits: DeviceLimits {
                max_contexts: 1,
                max_buffers_per_context: 1,
                max_programs_per_context: 1,
                max_queues_per_context: 1,
                max_events_per_context: 1,
                max_bindings_per_submission: 1,
                max_buffer_bytes: 1,
                max_artifact_bytes: 1,
            },
        }
    }

    #[test]
    fn transport_byte_adapters_preserve_segment_ports_without_copying() {
        let mut bytes = TransportBytes(*b"abcd");
        let source = TransportByteSource::new(&bytes);
        let mut read = [0; 2];
        ByteSource::read_at(&source, 1, &mut read).unwrap();
        assert_eq!(&read, b"bc");

        let mut sink = TransportByteSink::new(&mut bytes);
        ByteSink::write_at(&mut sink, 2, b"xy").unwrap();
        assert_eq!(&bytes.0, b"abxy");
    }

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
    fn buffer_usage_defines_submission_access_compatibility() {
        let input =
            BufferDesc::new(64, 16, MemoryDomain::Host, BufferUsage::PROGRAM_INPUT).unwrap();
        assert!(input.allows_access(AccessMode::Read));
        assert!(!input.allows_access(AccessMode::Write));
        assert!(!input.allows_access(AccessMode::ReadWrite));

        let output =
            BufferDesc::new(64, 16, MemoryDomain::Host, BufferUsage::PROGRAM_OUTPUT).unwrap();
        assert!(!output.allows_access(AccessMode::Read));
        assert!(output.allows_access(AccessMode::Write));
        assert!(!output.allows_access(AccessMode::ReadWrite));

        let mutable =
            BufferDesc::new(64, 16, MemoryDomain::Host, BufferUsage::MUTABLE_STATE).unwrap();
        assert!(mutable.allows_access(AccessMode::Read));
        assert!(mutable.allows_access(AccessMode::Write));
        assert!(mutable.allows_access(AccessMode::ReadWrite));
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
    fn device_information_rejects_unusable_provider_contracts() {
        let valid = valid_device_info();
        assert_eq!(valid.validate(), Ok(()));

        let mut reserved = valid;
        reserved.capabilities |= Capabilities::EXTERNAL_MEMORY;
        assert_eq!(
            reserved.validate(),
            Err(DeviceInfoError::ReservedCapabilities)
        );

        let mut no_memory = valid;
        no_memory.capabilities = Capabilities::EVENT_CANCELLATION;
        assert_eq!(
            no_memory.validate(),
            Err(DeviceInfoError::MissingMemoryDomain)
        );

        let mut zero_limit = valid;
        zero_limit.limits.max_bindings_per_submission = 0;
        assert_eq!(zero_limit.validate(), Err(DeviceInfoError::ZeroLimit));

        let mut unknown = valid;
        unknown.capabilities |= Capabilities::from_bits_retain(1 << 63);
        assert_eq!(unknown.validate(), Ok(()));
    }

    #[test]
    fn reserved_operations_are_rejected_before_provider_invocation() {
        let mut info = valid_device_info();
        assert_eq!(info.validate_context_desc(ContextDesc::default()), Ok(()));
        assert_eq!(info.validate_queue_desc(QueueDesc::default()), Ok(()));
        assert_eq!(
            info.validate_context_desc(ContextDesc {
                flags: ContextFlags::SECURE,
            }),
            Err(BackendError::Unsupported)
        );
        assert_eq!(
            info.validate_queue_desc(QueueDesc {
                flags: QueueFlags::IN_ORDER,
            }),
            Err(BackendError::Unsupported)
        );
        assert_eq!(
            info.validate_event_cancellation(),
            Err(BackendError::Unsupported)
        );

        info.capabilities |= Capabilities::EVENT_CANCELLATION;
        assert_eq!(info.validate_event_cancellation(), Ok(()));
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

        let arbitrary_order = [
            BindingRef {
                slot: 7,
                buffer: &buffer,
                range,
                access: AccessMode::Read,
            },
            BindingRef {
                slot: 2,
                buffer: &buffer,
                range,
                access: AccessMode::Write,
            },
        ];
        assert!(validate_bindings(&arbitrary_order, 2).is_ok());

        let canonical_order = [
            BindingRef {
                slot: 2,
                buffer: &buffer,
                range,
                access: AccessMode::Read,
            },
            BindingRef {
                slot: 7,
                buffer: &buffer,
                range,
                access: AccessMode::Write,
            },
        ];
        assert!(validate_bindings(&canonical_order, 2).is_ok());
    }

    #[test]
    fn binding_access_rejects_usage_mismatch_with_unique_slots() {
        let buffer = ();
        let range = BufferRange::new(0, 16).unwrap();
        let bindings = [BindingRef {
            slot: 0,
            buffer: &buffer,
            range,
            access: AccessMode::Write,
        }];
        let input =
            BufferDesc::new(64, 16, MemoryDomain::Host, BufferUsage::PROGRAM_INPUT).unwrap();
        assert!(!input.allows_access(AccessMode::Write));
        // Slot-only checks still pass; the usage gate lives on validate_for_submit.
        assert!(validate_bindings(&bindings, 1).is_ok());
        assert_eq!(
            BindingRef::validate_for_submit(&bindings, &[input], 1),
            Err(BackendError::PermissionDenied)
        );

        let read_bindings = [BindingRef {
            slot: 0,
            buffer: &buffer,
            range,
            access: AccessMode::Read,
        }];
        assert!(BindingRef::validate_for_submit(&read_bindings, &[input], 1).is_ok());
        assert_eq!(
            BindingRef::validate_for_submit(&read_bindings, &[], 1),
            Err(BackendError::InvalidArgument)
        );
    }

    #[test]
    fn wire_timeouts_are_relative_and_zero_is_infinite() {
        assert_eq!(Timeout::from_wire_ns(0), Timeout::Infinite);
        assert_eq!(Timeout::from_wire_ns(42).to_wire_ns(), 42);
    }
}
