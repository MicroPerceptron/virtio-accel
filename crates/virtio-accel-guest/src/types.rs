use bitflags::bitflags;
use core::num::{NonZeroU32, NonZeroU64};

use virtio_accel_proto::{HARD_MAX_BINDINGS, StatusCode, WireDeviceInfo};
use virtio_accel_transport::QueueEpoch;

const CAPABILITY_HOST_VISIBLE_MEMORY: u64 = 1 << 0;
const CAPABILITY_DEVICE_LOCAL_MEMORY: u64 = 1 << 1;
const CAPABILITY_EVENT_CANCELLATION: u64 = 1 << 2;
const CAPABILITY_RESERVED_EXTERNAL_MEMORY: u64 = 1 << 3;
const CAPABILITY_RESERVED_SECURE_CONTEXTS: u64 = 1 << 4;
const CAPABILITY_SHARED_MEMORY: u64 = 1 << 5;
const RESERVED_CAPABILITIES: u64 =
    CAPABILITY_RESERVED_EXTERNAL_MEMORY | CAPABILITY_RESERVED_SECURE_CONTEXTS;

bitflags! {
    /// Protocol 1.0 buffer usage bits.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct BufferUsage: u32 {
        const TRANSFER_SOURCE = 1 << 0;
        const TRANSFER_DESTINATION = 1 << 1;
        const PROGRAM_INPUT = 1 << 2;
        const PROGRAM_OUTPUT = 1 << 3;
        const MUTABLE_STATE = 1 << 4;
    }
}

/// Requested provider-owned memory placement.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum MemoryDomain {
    /// Host-preferred memory.
    Host = 1,
    /// Accelerator-local memory.
    Device = 2,
    /// Provider-owned host-visible, directly bindable memory.
    Shared = 3,
}

/// Program binding access mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum AccessMode {
    /// Program reads the buffer.
    Read = 1,
    /// Program writes the buffer.
    Write = 2,
    /// Program reads and writes the buffer.
    ReadWrite = 3,
}

/// Locally invalid typed request value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ValueError {
    /// A required scalar is zero.
    Zero,
    /// A byte range overflows.
    Overflow,
    /// Alignment is not a nonzero power of two.
    Alignment,
    /// Buffer usage is empty or contains an unknown bit.
    BufferUsage,
}

/// Guest-side ownership meaning of a well-formed non-success response.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FailureDisposition {
    /// The device rejected the operation before consuming semantic ownership.
    Retryable,
    /// A referenced object is stale and must not be retried as live.
    Invalidated,
    /// Device loss leaves semantic ownership uncertain and requires reset.
    Indeterminate,
    /// An unknown status has no protocol 1.0 ownership meaning.
    Unknown,
}

/// Requested buffer properties retained by the typed guest handle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BufferDesc {
    /// Logical buffer bytes.
    pub(crate) bytes: u64,
    /// Required power-of-two alignment.
    pub(crate) alignment: u64,
    /// Requested memory placement.
    pub(crate) memory_domain: MemoryDomain,
    /// Declared operation usage.
    pub(crate) usage: BufferUsage,
}

impl BufferDesc {
    /// Construct a locally valid buffer request.
    pub fn new(
        bytes: u64,
        alignment: u64,
        memory_domain: MemoryDomain,
        usage: BufferUsage,
    ) -> Result<Self, ValueError> {
        if bytes == 0 {
            return Err(ValueError::Zero);
        }
        if !alignment.is_power_of_two() {
            return Err(ValueError::Alignment);
        }
        if usage.is_empty() || BufferUsage::from_bits(usage.bits()).is_none() {
            return Err(ValueError::BufferUsage);
        }
        Ok(Self {
            bytes,
            alignment,
            memory_domain,
            usage,
        })
    }

    /// Logical buffer bytes.
    pub const fn bytes(self) -> u64 {
        self.bytes
    }

    /// Required power-of-two alignment.
    pub const fn alignment(self) -> u64 {
        self.alignment
    }

    /// Requested memory placement.
    pub const fn memory_domain(self) -> MemoryDomain {
        self.memory_domain
    }

    /// Declared operation usage.
    pub const fn usage(self) -> BufferUsage {
        self.usage
    }
}

/// Nonempty checked buffer byte range.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BufferRange {
    /// Starting byte offset.
    pub(crate) offset: u64,
    /// Number of bytes.
    pub(crate) bytes: u64,
}

impl BufferRange {
    /// Construct a nonempty, non-overflowing range.
    pub fn new(offset: u64, bytes: u64) -> Result<Self, ValueError> {
        if bytes == 0 {
            return Err(ValueError::Zero);
        }
        offset.checked_add(bytes).ok_or(ValueError::Overflow)?;
        Ok(Self { offset, bytes })
    }

    /// Starting byte offset.
    pub const fn offset(self) -> u64 {
        self.offset
    }

    /// Number of bytes.
    pub const fn bytes(self) -> u64 {
        self.bytes
    }

    pub(crate) fn fits(self, limit: u64) -> bool {
        self.offset
            .checked_add(self.bytes)
            .is_some_and(|end| end <= limit)
    }
}

/// Opaque program envelope retained until program creation completes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProgramDesc {
    /// Provider-owned nonzero artifact format.
    pub format: NonZeroU32,
    /// Opaque provider-owned target words.
    pub target: [u32; 12],
    /// Declared nonzero resident-memory charge.
    pub resident_bytes: NonZeroU64,
}

impl ProgramDesc {
    /// Construct a program descriptor.
    pub const fn new(format: NonZeroU32, target: [u32; 12], resident_bytes: NonZeroU64) -> Self {
        Self {
            format,
            target,
            resident_bytes,
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Handle {
    id: NonZeroU64,
    epoch: QueueEpoch,
}

impl Handle {
    pub(crate) const fn new(id: NonZeroU64, epoch: QueueEpoch) -> Self {
        Self { id, epoch }
    }

    pub(crate) const fn id(&self) -> NonZeroU64 {
        self.id
    }

    pub(crate) const fn epoch(&self) -> QueueEpoch {
        self.epoch
    }
}

macro_rules! simple_handle {
    ($name:ident, $doc:literal) => {
        #[doc = $doc]
        #[derive(Debug, PartialEq, Eq)]
        pub struct $name {
            pub(crate) handle: Handle,
            pub(crate) context: Option<NonZeroU64>,
        }

        impl $name {
            /// Raw opaque wire object ID.
            pub const fn raw(&self) -> u64 {
                self.handle.id().get()
            }

            /// Reset epoch in which this handle was created.
            pub const fn epoch(&self) -> QueueEpoch {
                self.handle.epoch()
            }
        }
    };
}

simple_handle!(Context, "Guest-owned context handle.");
simple_handle!(Program, "Guest-owned resident program handle.");
simple_handle!(
    ExecutionQueue,
    "Guest-owned accelerator execution-queue handle."
);
simple_handle!(Event, "Guest-owned submission event handle.");

/// Guest-owned buffer handle and retained bounds needed for local validation.
#[derive(Debug, PartialEq, Eq)]
pub struct Buffer {
    pub(crate) handle: Handle,
    pub(crate) context: NonZeroU64,
    pub(crate) desc: BufferDesc,
}

impl Buffer {
    /// Raw opaque wire object ID.
    pub const fn raw(&self) -> u64 {
        self.handle.id().get()
    }

    /// Reset epoch in which this handle was created.
    pub const fn epoch(&self) -> QueueEpoch {
        self.handle.epoch()
    }

    /// Buffer properties accepted by the device.
    pub const fn desc(&self) -> BufferDesc {
        self.desc
    }
}

/// One borrowed binding encoded directly into a submission chain.
#[derive(Clone, Copy, Debug)]
pub struct Binding<'a> {
    /// Buffer retained by the event on successful or indeterminate admission.
    pub buffer: &'a Buffer,
    /// Bound nonempty range.
    pub range: BufferRange,
    /// Program-defined slot number.
    pub slot: u32,
    /// Program access mode.
    pub access: AccessMode,
}

/// Validated device discovery result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeviceInfo {
    /// Stable device UUID.
    pub uuid: [u8; 16],
    /// Extensible raw accelerator class.
    pub class: u16,
    /// Provider vendor identifier.
    pub vendor_id: u32,
    /// Provider device identifier.
    pub device_id: u32,
    /// Raw semantic capabilities, including unknown diagnostic bits.
    pub capabilities: u64,
    /// Device-wide live context limit.
    pub max_contexts: u32,
    /// Per-context live buffer limit.
    pub max_buffers_per_context: u32,
    /// Per-context live program limit.
    pub max_programs_per_context: u32,
    /// Per-context live execution-queue limit.
    pub max_queues_per_context: u32,
    /// Per-context live event limit.
    pub max_events_per_context: u32,
    /// Maximum bindings accepted by one submission.
    pub max_bindings_per_submission: u32,
    /// Maximum logical buffer allocation.
    pub max_buffer_bytes: u64,
    /// Maximum program artifact tail.
    pub max_artifact_bytes: u64,
}

impl DeviceInfo {
    pub(crate) fn from_wire(
        wire: WireDeviceInfo,
        max_request_bytes: u32,
    ) -> Result<Self, DeviceInfoError> {
        if wire.reserved.get() != 0 {
            return Err(DeviceInfoError::Reserved);
        }
        let capabilities = wire.capabilities.get();
        if capabilities & RESERVED_CAPABILITIES != 0 {
            return Err(DeviceInfoError::ReservedCapabilities);
        }
        let limits = [
            wire.max_contexts.get(),
            wire.max_buffers_per_context.get(),
            wire.max_programs_per_context.get(),
            wire.max_queues_per_context.get(),
            wire.max_events_per_context.get(),
        ];
        if limits.contains(&0) {
            return Err(DeviceInfoError::ZeroLimit);
        }
        let max_bindings = wire.max_bindings_per_submission.get();
        if !(1..=HARD_MAX_BINDINGS).contains(&max_bindings) {
            return Err(DeviceInfoError::BindingLimit);
        }
        if wire.max_buffer_bytes.get() == 0 || wire.max_artifact_bytes.get() == 0 {
            return Err(DeviceInfoError::ZeroLimit);
        }
        let artifact_frame = 16_u64
            .checked_add(80)
            .and_then(|bytes| bytes.checked_add(wire.max_artifact_bytes.get()))
            .ok_or(DeviceInfoError::ArtifactLimit)?;
        if artifact_frame > u64::from(max_request_bytes) {
            return Err(DeviceInfoError::ArtifactLimit);
        }
        Ok(Self {
            uuid: wire.uuid,
            class: wire.class.get(),
            vendor_id: wire.vendor_id.get(),
            device_id: wire.device_id.get(),
            capabilities,
            max_contexts: wire.max_contexts.get(),
            max_buffers_per_context: wire.max_buffers_per_context.get(),
            max_programs_per_context: wire.max_programs_per_context.get(),
            max_queues_per_context: wire.max_queues_per_context.get(),
            max_events_per_context: wire.max_events_per_context.get(),
            max_bindings_per_submission: max_bindings,
            max_buffer_bytes: wire.max_buffer_bytes.get(),
            max_artifact_bytes: wire.max_artifact_bytes.get(),
        })
    }

    /// Whether event cancellation is semantically available.
    pub const fn supports_event_cancellation(self) -> bool {
        self.capabilities & CAPABILITY_EVENT_CANCELLATION != 0
    }

    pub(crate) const fn supports_domain(self, domain: MemoryDomain) -> bool {
        let bit = match domain {
            MemoryDomain::Host => CAPABILITY_HOST_VISIBLE_MEMORY,
            MemoryDomain::Device => CAPABILITY_DEVICE_LOCAL_MEMORY,
            MemoryDomain::Shared => CAPABILITY_SHARED_MEMORY,
        };
        self.capabilities & bit != 0
    }
}

/// Invalid device-information payload.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceInfoError {
    /// Reserved structure bytes are nonzero.
    Reserved,
    /// A reserved semantic capability was advertised.
    ReservedCapabilities,
    /// A mandatory advertised limit is zero.
    ZeroLimit,
    /// Binding limit is outside protocol bounds.
    BindingLimit,
    /// Artifact limit does not fit the configured request frame.
    ArtifactLimit,
}

/// Validated accelerator execution state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EventState {
    /// Execution remains pending.
    Pending,
    /// Execution completed successfully.
    Complete,
    /// Execution failed with an opaque protocol status.
    Failed(StatusCode),
    /// Execution was cancelled.
    Cancelled,
}

/// Successful or ownership-indeterminate submission admission.
#[derive(Debug, PartialEq, Eq)]
pub enum SubmissionOutcome {
    /// Backend admission was accepted.
    Accepted(Event),
    /// Admission could not be determined; the event retains all operation resources.
    Indeterminate {
        /// Mapped non-success admission status.
        status: StatusCode,
        /// Event that owns the possibly accepted operation.
        event: Event,
    },
}

/// Metadata for a successful `READ_BUFFER` response retained in its returned chain.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReadBufferOutput {
    /// Payload bytes beginning at writable offset 16.
    pub bytes: u64,
}
