use alloc::vec::Vec;
use core::mem::{self, size_of};

use virtio_accel_proto::{
    AllocateBufferRequest, CreateContextRequest, CreateQueueRequest, Le32, Le64,
    LoadProgramRequest, ObjectPayload, RequestFlags, RequestHeader, ResponseHeader, StatusCode,
    SubmitRequest, TransferBufferRequest, WireBinding, WireDeviceInfo, WireEventState, read_exact,
};
use virtio_accel_transport::{
    ByteAccessError, ChainId, DriverChainBuffer, DriverQueue, NotificationRecheck,
    PublishErrorKind, QueueControl, QueueEpoch, QueueError, QueueSize, QueueState, ReadableBytes,
    UsedChain, UsedLength,
};
use zerocopy::{Immutable, IntoBytes};

use crate::config::{GuestConfig, GuestConfigError};
use crate::operation::{
    AllocateBuffer, CancelEvent, CreateContext, CreateExecutionQueue, DestroyContext, DestroyEvent,
    DestroyExecutionQueue, FreeBuffer, GetDeviceInfo, LoadProgram, Operation, OperationResult,
    PollEvent, ReadBuffer, ResponseError, Submit, UnloadProgram, WriteBuffer,
};
use crate::types::{
    AccessMode, Binding, Buffer, BufferDesc, BufferRange, BufferUsage, Context, DeviceInfo, Event,
    ExecutionQueue, FailureDisposition, Program, ProgramDesc,
};

const REQUEST_HEADER_BYTES: u64 = size_of::<RequestHeader>() as u64;
const RESPONSE_HEADER_BYTES: u64 = size_of::<ResponseHeader>() as u64;
const COPY_SCRATCH_BYTES: usize = 256;

/// Whether the client may publish new work.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClientHealth {
    /// Queue tracking and response framing remain trustworthy.
    Running,
    /// An unexpected or malformed completion requires a queue reset.
    NeedsReset,
}

/// Failure to construct a bounded guest client.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClientInitError {
    /// Configuration or queue state is incompatible with the client.
    Config(GuestConfigError),
    /// Bounded in-flight tracking storage could not be allocated.
    AllocationFailed,
}

/// Failure while restoring queue configuration after reset.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QueueSetupError<E> {
    /// Requested queue size contradicts the validated guest configuration.
    Config(GuestConfigError),
    /// Concrete queue configuration failed.
    Queue(QueueError<E>),
}

/// Failure before request ownership transfers to the queue.
#[derive(Debug, PartialEq, Eq)]
pub enum StartErrorKind<QE, CE> {
    /// A prior malformed completion requires reset.
    NeedsReset,
    /// Device information must be discovered first.
    DiscoveryRequired,
    /// Every bounded tracking slot is occupied.
    InflightLimit,
    /// No request identifier could be selected without aliasing live work.
    RequestIdExhausted,
    /// A typed handle belongs to a prior queue epoch.
    StaleHandle,
    /// Typed arguments contradict retained object bounds or usage.
    InvalidArgument,
    /// A request exceeds an advertised semantic device limit.
    DeviceLimit,
    /// The device-readable chain length is not the exact request frame length.
    RequestCapacity {
        /// Required exact frame length.
        required: u64,
        /// Supplied readable bytes.
        available: u64,
    },
    /// The device-writable chain cannot hold the largest valid response.
    ResponseCapacity {
        /// Required response capacity.
        required: u64,
        /// Supplied writable bytes.
        available: u64,
    },
    /// The caller-provided request source could not be read.
    SourceAccess(ByteAccessError),
    /// The chain request region could not be written.
    ChainAccess(CE),
    /// Queue publication failed before ownership transfer.
    Publish(PublishErrorKind<QE>),
}

/// Failed request start with both caller-owned inputs returned.
#[derive(Debug)]
pub struct StartError<C, O, QE, CE> {
    chain: C,
    operation: O,
    kind: StartErrorKind<QE, CE>,
}

impl<C, O, QE, CE> StartError<C, O, QE, CE> {
    /// Borrow the failure classification.
    pub const fn kind(&self) -> &StartErrorKind<QE, CE> {
        &self.kind
    }

    /// Recover the unpublished chain, typed operation, and failure classification.
    pub fn into_parts(self) -> (C, O, StartErrorKind<QE, CE>) {
        (self.chain, self.operation, self.kind)
    }
}

/// Result of starting one typed operation on queue `Q`.
pub type StartResult<Q, O> = Result<
    Pending<O>,
    StartError<
        <Q as DriverQueue>::Chain,
        O,
        <Q as DriverQueue>::Error,
        <<Q as DriverQueue>::Chain as DriverChainBuffer>::Error,
    >,
>;

/// Typed operation token retained until its matching completion is observed.
#[derive(Debug)]
pub struct Pending<O> {
    slot: u16,
    request_id: u64,
    epoch: QueueEpoch,
    operation: O,
}

impl<O> Pending<O> {
    /// Nonzero wire request identifier.
    pub const fn request_id(&self) -> u64 {
        self.request_id
    }

    /// Queue epoch in which this request was published.
    pub const fn epoch(&self) -> QueueEpoch {
        self.epoch
    }

    /// Borrow operation metadata retained for response validation or recovery.
    pub const fn operation(&self) -> &O {
        &self.operation
    }
}

/// Operation metadata recovered only after its queue epoch is stale.
#[derive(Debug)]
pub struct StaleOperation<O> {
    request_id: u64,
    epoch: QueueEpoch,
    operation: O,
}

impl<O> StaleOperation<O> {
    /// Request identifier from the invalidated queue epoch.
    pub const fn request_id(&self) -> u64 {
        self.request_id
    }

    /// Invalidated queue epoch.
    pub const fn epoch(&self) -> QueueEpoch {
        self.epoch
    }

    /// Borrow the invalidated operation metadata.
    pub const fn operation(&self) -> &O {
        &self.operation
    }

    /// Recover operation metadata for cleanup or inspection.
    ///
    /// Typed handles from this operation remain stale and cannot be republished in the new epoch.
    pub fn into_operation(self) -> O {
        self.operation
    }
}

/// Terminal result for one matched request completion.
#[derive(Debug)]
pub enum Completion<O: Operation, C, CE> {
    /// Protocol success with the reclaimed caller chain.
    Success {
        /// Typed response value. Bulk read bytes remain in `chain`.
        output: O::Output,
        /// Reclaimed caller-owned chain.
        chain: C,
    },
    /// Well-formed protocol error; operation metadata is returned for retry decisions.
    DeviceError {
        /// Raw status, preserving unknown provider values.
        status: StatusCode,
        /// Whether typed operation ownership is retryable, invalid, or uncertain.
        disposition: FailureDisposition,
        /// Original typed operation.
        operation: O,
        /// Reclaimed caller-owned chain.
        chain: C,
    },
    /// Malformed or inaccessible response; reset is required.
    InvalidResponse {
        /// Validation failure.
        error: ResponseError<CE>,
        /// Original typed operation.
        operation: O,
        /// Reclaimed caller-owned chain.
        chain: C,
    },
}

/// Result of synchronously polling one typed request.
pub enum RequestPoll<O: Operation, C, QE, CE> {
    /// No matching completion has arrived yet.
    Pending(Pending<O>),
    /// Matching request reached a terminal result.
    Ready(Completion<O, C, CE>),
    /// Used-ring access failed; the pending token remains valid for retry.
    QueueError {
        /// Original pending token.
        pending: Pending<O>,
        /// Queue failure.
        error: QueueError<QE>,
    },
    /// Queue reset invalidated this operation.
    Stale(StaleOperation<O>),
    /// Queue health requires reset before more completion processing.
    NeedsReset(Pending<O>),
}

/// Result of consuming at most one used-ring entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PumpResult {
    /// No completion was available.
    Idle,
    /// One tracked completion was retained for its pending token.
    Completion {
        /// Request identifier associated with the completed chain.
        request_id: u64,
    },
    /// A used chain had no live matching publication.
    UnexpectedCompletion,
    /// Client already requires reset.
    NeedsReset,
}

enum Slot<C> {
    Vacant,
    InFlight { request_id: u64, chain_id: ChainId },
    Completed { request_id: u64, used: UsedChain<C> },
    Recovered(UsedChain<C>),
}

enum FrameWriteError<E> {
    Chain(E),
    Source(ByteAccessError),
}

/// Bounded, single-owner reference driver over a portable command queue.
///
/// The type performs no internal locking. Callers that share one client across execution contexts
/// choose their own synchronization policy; ordinary single-owner polling needs none.
pub struct GuestClient<Q: DriverQueue>
where
    Q::Chain: DriverChainBuffer,
{
    queue: Q,
    config: GuestConfig,
    device_info: Option<DeviceInfo>,
    slots: Vec<Slot<Q::Chain>>,
    next_request_id: u64,
    health: ClientHealth,
    unexpected: Option<UsedChain<Q::Chain>>,
}

impl<Q> GuestClient<Q>
where
    Q: DriverQueue,
    Q::Chain: DriverChainBuffer,
{
    /// Construct a client after transport feature and queue setup.
    pub fn new(queue: Q, config: GuestConfig) -> Result<Self, ClientInitError> {
        config
            .validate_queue(queue.state())
            .map_err(ClientInitError::Config)?;
        let count = usize::from(config.max_inflight());
        let mut slots = Vec::new();
        slots
            .try_reserve_exact(count)
            .map_err(|_| ClientInitError::AllocationFailed)?;
        for _ in 0..count {
            slots.push(Slot::Vacant);
        }
        Ok(Self {
            queue,
            config,
            device_info: None,
            slots,
            next_request_id: 1,
            health: ClientHealth::Running,
            unexpected: None,
        })
    }

    /// Current queue lifecycle snapshot.
    pub fn queue_state(&self) -> QueueState {
        self.queue.state()
    }

    /// Validated guest configuration.
    pub const fn config(&self) -> GuestConfig {
        self.config
    }

    /// Latest discovery result in the current epoch.
    pub const fn device_info(&self) -> Option<DeviceInfo> {
        self.device_info
    }

    /// Current response-tracking health.
    pub const fn health(&self) -> ClientHealth {
        self.health
    }

    /// Disable used notifications before draining completions.
    pub fn disable_used_notifications(&mut self) -> Result<(), QueueError<Q::Error>> {
        self.queue.disable_used_notifications()
    }

    /// Atomically enable used notifications and check for missed completions.
    pub fn enable_used_notifications(
        &mut self,
    ) -> Result<NotificationRecheck, QueueError<Q::Error>> {
        self.queue.enable_used_notifications()
    }

    /// Consume and classify at most one used-ring entry.
    pub fn pump(&mut self) -> Result<PumpResult, QueueError<Q::Error>> {
        if self.health == ClientHealth::NeedsReset {
            return Ok(PumpResult::NeedsReset);
        }
        let Some(used) = self.queue.pop_used()? else {
            return Ok(PumpResult::Idle);
        };
        let id = used.id();
        let Some(index) = self
            .slots
            .iter()
            .position(|slot| matches!(slot, Slot::InFlight { chain_id, .. } if *chain_id == id))
        else {
            self.unexpected = Some(used);
            self.health = ClientHealth::NeedsReset;
            return Ok(PumpResult::UnexpectedCompletion);
        };
        let Slot::InFlight { request_id, .. } = mem::replace(&mut self.slots[index], Slot::Vacant)
        else {
            unreachable!("matched slot must be in flight")
        };
        self.slots[index] = Slot::Completed { request_id, used };
        Ok(PumpResult::Completion { request_id })
    }

    /// Poll one typed operation, consuming at most one queue completion.
    pub fn poll<O: Operation>(
        &mut self,
        pending: Pending<O>,
    ) -> RequestPoll<O, Q::Chain, Q::Error, <Q::Chain as DriverChainBuffer>::Error> {
        if pending.epoch != self.queue.state().epoch() {
            return RequestPoll::Stale(stale_operation(pending));
        }
        if self.health == ClientHealth::NeedsReset {
            return RequestPoll::NeedsReset(pending);
        }
        if !self.pending_matches(&pending) {
            return RequestPoll::Stale(stale_operation(pending));
        }
        if matches!(self.slots[usize::from(pending.slot)], Slot::InFlight { .. }) {
            match self.pump() {
                Err(error) => return RequestPoll::QueueError { pending, error },
                Ok(PumpResult::UnexpectedCompletion | PumpResult::NeedsReset) => {
                    return RequestPoll::NeedsReset(pending);
                }
                Ok(PumpResult::Idle | PumpResult::Completion { .. }) => {}
            }
        }
        let index = usize::from(pending.slot);
        if !matches!(self.slots[index], Slot::Completed { .. }) {
            return RequestPoll::Pending(pending);
        }
        let Slot::Completed { request_id, used } =
            mem::replace(&mut self.slots[index], Slot::Vacant)
        else {
            unreachable!("checked slot must contain a completion")
        };
        if request_id != pending.request_id {
            self.slots[index] = Slot::Completed { request_id, used };
            return RequestPoll::Stale(stale_operation(pending));
        }
        RequestPoll::Ready(self.decode_completion(pending, used))
    }

    /// Reset the queue and invalidate all pending operations and typed handles.
    ///
    /// Published chains are returned by the queue's reclaimed collection. Chains already popped
    /// from the used ring are retained for [`Self::pop_recovered_completion`].
    pub fn reset(&mut self, next_epoch: QueueEpoch) -> Result<Q::Reclaimed, QueueError<Q::Error>> {
        let reclaimed = self.queue.reset(next_epoch)?;
        for slot in &mut self.slots {
            let old = mem::replace(slot, Slot::Vacant);
            *slot = match old {
                Slot::Completed { used, .. } | Slot::Recovered(used) => Slot::Recovered(used),
                Slot::Vacant | Slot::InFlight { .. } => Slot::Vacant,
            };
        }
        self.device_info = None;
        self.next_request_id = 1;
        self.health = ClientHealth::Running;
        Ok(reclaimed)
    }

    /// Configure and ready the command queue after transport reset.
    pub fn reconfigure_queue(
        &mut self,
        size: QueueSize,
    ) -> Result<(), QueueSetupError<<Q as DriverQueue>::Error>>
    where
        Q: QueueControl<Error = <Q as DriverQueue>::Error>,
    {
        self.config
            .validate_size(size)
            .map_err(QueueSetupError::Config)?;
        self.queue.configure(size).map_err(QueueSetupError::Queue)?;
        self.queue.set_ready(true).map_err(QueueSetupError::Queue)
    }

    /// Recover one completion that had been popped before reset invalidated its pending token.
    pub fn pop_recovered_completion(&mut self) -> Option<UsedChain<Q::Chain>> {
        let index = self
            .slots
            .iter()
            .position(|slot| matches!(slot, Slot::Recovered(_)))?;
        let Slot::Recovered(used) = mem::replace(&mut self.slots[index], Slot::Vacant) else {
            unreachable!("matched slot must contain a recovered completion")
        };
        Some(used)
    }

    /// Recover the unexpected completion that forced reset.
    pub fn take_unexpected_completion(&mut self) -> Option<UsedChain<Q::Chain>> {
        self.unexpected.take()
    }

    /// Start protocol discovery.
    pub fn get_device_info(&mut self, chain: Q::Chain) -> StartResult<Q, GetDeviceInfo> {
        self.start_frame(
            chain,
            GetDeviceInfo,
            0,
            RESPONSE_HEADER_BYTES + size_of::<WireDeviceInfo>() as u64,
            |_| Ok(()),
        )
    }

    /// Start context creation.
    pub fn create_context(&mut self, chain: Q::Chain) -> StartResult<Q, CreateContext> {
        let request = CreateContextRequest {
            flags: Le32::new(0),
            reserved: Le32::new(0),
        };
        self.start_wire(
            chain,
            CreateContext,
            &request,
            RESPONSE_HEADER_BYTES + size_of::<ObjectPayload>() as u64,
        )
    }

    /// Start context destruction, consuming the handle until completion or failure recovery.
    pub fn destroy_context(
        &mut self,
        chain: Q::Chain,
        context: Context,
    ) -> StartResult<Q, DestroyContext> {
        let stale = context.epoch() != self.queue.state().epoch();
        let object = ObjectPayload {
            object_id: Le64::new(context.raw()),
        };
        let operation = DestroyContext { context };
        if stale {
            return self.start_failure(chain, operation, StartErrorKind::StaleHandle);
        }
        self.start_wire(chain, operation, &object, RESPONSE_HEADER_BYTES)
    }

    /// Start a bounded buffer allocation.
    pub fn allocate_buffer(
        &mut self,
        chain: Q::Chain,
        context: &Context,
        desc: BufferDesc,
    ) -> StartResult<Q, AllocateBuffer> {
        let operation = AllocateBuffer {
            context: context.handle.id(),
            desc,
        };
        if context.epoch() != self.queue.state().epoch() {
            return self.start_failure(chain, operation, StartErrorKind::StaleHandle);
        }
        let Some(info) = self.device_info else {
            return self.start_failure(chain, operation, StartErrorKind::DiscoveryRequired);
        };
        if desc.bytes > info.max_buffer_bytes || !info.supports_domain(desc.memory_domain) {
            return self.start_failure(chain, operation, StartErrorKind::DeviceLimit);
        }
        let request = AllocateBufferRequest {
            context_id: Le64::new(context.raw()),
            bytes: Le64::new(desc.bytes),
            alignment: Le64::new(desc.alignment),
            memory_domain: desc.memory_domain as u8,
            reserved0: [0; 7],
            usage: Le32::new(desc.usage.bits()),
            reserved1: Le32::new(0),
        };
        self.start_wire(
            chain,
            operation,
            &request,
            RESPONSE_HEADER_BYTES + size_of::<ObjectPayload>() as u64,
        )
    }

    /// Start buffer release, consuming the handle until completion or failure recovery.
    pub fn free_buffer(&mut self, chain: Q::Chain, buffer: Buffer) -> StartResult<Q, FreeBuffer> {
        let stale = buffer.epoch() != self.queue.state().epoch();
        let object = ObjectPayload {
            object_id: Le64::new(buffer.raw()),
        };
        let operation = FreeBuffer { buffer };
        if stale {
            return self.start_failure(chain, operation, StartErrorKind::StaleHandle);
        }
        self.start_wire(chain, operation, &object, RESPONSE_HEADER_BYTES)
    }

    /// Start a copy from caller source bytes into a provider-owned buffer.
    pub fn write_buffer<R: ReadableBytes + ?Sized>(
        &mut self,
        chain: Q::Chain,
        buffer: &Buffer,
        offset: u64,
        source: &R,
    ) -> StartResult<Q, WriteBuffer> {
        let operation = WriteBuffer;
        if buffer.epoch() != self.queue.state().epoch() {
            return self.start_failure(chain, operation, StartErrorKind::StaleHandle);
        }
        let Ok(range) = BufferRange::new(offset, source.len()) else {
            return self.start_failure(chain, operation, StartErrorKind::InvalidArgument);
        };
        if !range.fits(buffer.desc.bytes)
            || !buffer
                .desc
                .usage
                .contains(BufferUsage::TRANSFER_DESTINATION)
        {
            return self.start_failure(chain, operation, StartErrorKind::InvalidArgument);
        }
        let prefix = TransferBufferRequest {
            buffer_id: Le64::new(buffer.raw()),
            offset: Le64::new(offset),
            bytes: Le64::new(source.len()),
        };
        let Some(payload) = (size_of::<TransferBufferRequest>() as u64).checked_add(source.len())
        else {
            return self.start_failure(chain, operation, StartErrorKind::DeviceLimit);
        };
        self.start_frame(chain, operation, payload, RESPONSE_HEADER_BYTES, |chain| {
            write_wire(chain, REQUEST_HEADER_BYTES, &prefix).map_err(FrameWriteError::Chain)?;
            copy_source(
                chain,
                REQUEST_HEADER_BYTES + size_of::<TransferBufferRequest>() as u64,
                source,
            )
        })
    }

    /// Start a zero-copy buffer write from an already prepared request-chain tail.
    ///
    /// Bytes beginning after the transfer prefix are left untouched and become the write payload.
    pub fn write_buffer_prepared(
        &mut self,
        chain: Q::Chain,
        buffer: &Buffer,
        range: BufferRange,
    ) -> StartResult<Q, WriteBuffer> {
        let operation = WriteBuffer;
        if buffer.epoch() != self.queue.state().epoch() {
            return self.start_failure(chain, operation, StartErrorKind::StaleHandle);
        }
        if !range.fits(buffer.desc.bytes)
            || !buffer
                .desc
                .usage
                .contains(BufferUsage::TRANSFER_DESTINATION)
        {
            return self.start_failure(chain, operation, StartErrorKind::InvalidArgument);
        }
        let prefix = TransferBufferRequest {
            buffer_id: Le64::new(buffer.raw()),
            offset: Le64::new(range.offset),
            bytes: Le64::new(range.bytes),
        };
        let Some(payload) = (size_of::<TransferBufferRequest>() as u64).checked_add(range.bytes)
        else {
            return self.start_failure(chain, operation, StartErrorKind::DeviceLimit);
        };
        self.start_frame(chain, operation, payload, RESPONSE_HEADER_BYTES, |chain| {
            write_wire(chain, REQUEST_HEADER_BYTES, &prefix).map_err(FrameWriteError::Chain)
        })
    }

    /// Start a provider-buffer read whose bytes remain in the reclaimed chain.
    pub fn read_buffer(
        &mut self,
        chain: Q::Chain,
        buffer: &Buffer,
        range: BufferRange,
    ) -> StartResult<Q, ReadBuffer> {
        let operation = ReadBuffer { bytes: range.bytes };
        if buffer.epoch() != self.queue.state().epoch() {
            return self.start_failure(chain, operation, StartErrorKind::StaleHandle);
        }
        if !range.fits(buffer.desc.bytes)
            || !buffer.desc.usage.contains(BufferUsage::TRANSFER_SOURCE)
        {
            return self.start_failure(chain, operation, StartErrorKind::InvalidArgument);
        }
        let request = TransferBufferRequest {
            buffer_id: Le64::new(buffer.raw()),
            offset: Le64::new(range.offset),
            bytes: Le64::new(range.bytes),
        };
        let Some(response_bytes) = RESPONSE_HEADER_BYTES.checked_add(range.bytes) else {
            return self.start_failure(chain, operation, StartErrorKind::DeviceLimit);
        };
        self.start_wire(chain, operation, &request, response_bytes)
    }

    /// Start loading an opaque program artifact directly from caller bytes.
    pub fn load_program<R: ReadableBytes + ?Sized>(
        &mut self,
        chain: Q::Chain,
        context: &Context,
        desc: ProgramDesc,
        artifact: &R,
    ) -> StartResult<Q, LoadProgram> {
        let operation = LoadProgram {
            context: context.handle.id(),
            desc,
        };
        if context.epoch() != self.queue.state().epoch() {
            return self.start_failure(chain, operation, StartErrorKind::StaleHandle);
        }
        let Some(info) = self.device_info else {
            return self.start_failure(chain, operation, StartErrorKind::DiscoveryRequired);
        };
        if artifact.is_empty() {
            return self.start_failure(chain, operation, StartErrorKind::InvalidArgument);
        }
        if artifact.len() > info.max_artifact_bytes {
            return self.start_failure(chain, operation, StartErrorKind::DeviceLimit);
        }
        let request = LoadProgramRequest {
            context_id: Le64::new(context.raw()),
            format: Le32::new(desc.format.get()),
            flags: Le32::new(0),
            target: core::array::from_fn(|index| Le32::new(desc.target[index])),
            payload_bytes: Le64::new(artifact.len()),
            resident_bytes: Le64::new(desc.resident_bytes.get()),
        };
        let payload = size_of::<LoadProgramRequest>() as u64 + artifact.len();
        self.start_frame(
            chain,
            operation,
            payload,
            RESPONSE_HEADER_BYTES + size_of::<ObjectPayload>() as u64,
            |chain| {
                write_wire(chain, REQUEST_HEADER_BYTES, &request)
                    .map_err(FrameWriteError::Chain)?;
                copy_source(
                    chain,
                    REQUEST_HEADER_BYTES + size_of::<LoadProgramRequest>() as u64,
                    artifact,
                )
            },
        )
    }

    /// Start a zero-copy program load from an already prepared request-chain tail.
    ///
    /// Artifact bytes beginning after the fixed load prefix are left untouched.
    pub fn load_program_prepared(
        &mut self,
        chain: Q::Chain,
        context: &Context,
        desc: ProgramDesc,
        artifact_bytes: u64,
    ) -> StartResult<Q, LoadProgram> {
        let operation = LoadProgram {
            context: context.handle.id(),
            desc,
        };
        if context.epoch() != self.queue.state().epoch() {
            return self.start_failure(chain, operation, StartErrorKind::StaleHandle);
        }
        let Some(info) = self.device_info else {
            return self.start_failure(chain, operation, StartErrorKind::DiscoveryRequired);
        };
        if artifact_bytes == 0 {
            return self.start_failure(chain, operation, StartErrorKind::InvalidArgument);
        }
        if artifact_bytes > info.max_artifact_bytes {
            return self.start_failure(chain, operation, StartErrorKind::DeviceLimit);
        }
        let request = LoadProgramRequest {
            context_id: Le64::new(context.raw()),
            format: Le32::new(desc.format.get()),
            flags: Le32::new(0),
            target: core::array::from_fn(|index| Le32::new(desc.target[index])),
            payload_bytes: Le64::new(artifact_bytes),
            resident_bytes: Le64::new(desc.resident_bytes.get()),
        };
        self.start_frame(
            chain,
            operation,
            size_of::<LoadProgramRequest>() as u64 + artifact_bytes,
            RESPONSE_HEADER_BYTES + size_of::<ObjectPayload>() as u64,
            |chain| {
                write_wire(chain, REQUEST_HEADER_BYTES, &request).map_err(FrameWriteError::Chain)
            },
        )
    }

    /// Start program release, consuming the handle until completion or failure recovery.
    pub fn unload_program(
        &mut self,
        chain: Q::Chain,
        program: Program,
    ) -> StartResult<Q, UnloadProgram> {
        let stale = program.epoch() != self.queue.state().epoch();
        let object = ObjectPayload {
            object_id: Le64::new(program.raw()),
        };
        let operation = UnloadProgram { program };
        if stale {
            return self.start_failure(chain, operation, StartErrorKind::StaleHandle);
        }
        self.start_wire(chain, operation, &object, RESPONSE_HEADER_BYTES)
    }

    /// Start creation of an accelerator execution queue.
    pub fn create_execution_queue(
        &mut self,
        chain: Q::Chain,
        context: &Context,
    ) -> StartResult<Q, CreateExecutionQueue> {
        let operation = CreateExecutionQueue {
            context: context.handle.id(),
        };
        if context.epoch() != self.queue.state().epoch() {
            return self.start_failure(chain, operation, StartErrorKind::StaleHandle);
        }
        let request = CreateQueueRequest {
            context_id: Le64::new(context.raw()),
            flags: Le32::new(0),
            reserved: Le32::new(0),
        };
        self.start_wire(
            chain,
            operation,
            &request,
            RESPONSE_HEADER_BYTES + size_of::<ObjectPayload>() as u64,
        )
    }

    /// Start execution-queue release, consuming the handle until completion or failure recovery.
    pub fn destroy_execution_queue(
        &mut self,
        chain: Q::Chain,
        queue: ExecutionQueue,
    ) -> StartResult<Q, DestroyExecutionQueue> {
        let stale = queue.epoch() != self.queue.state().epoch();
        let object = ObjectPayload {
            object_id: Le64::new(queue.raw()),
        };
        let operation = DestroyExecutionQueue { queue };
        if stale {
            return self.start_failure(chain, operation, StartErrorKind::StaleHandle);
        }
        self.start_wire(chain, operation, &object, RESPONSE_HEADER_BYTES)
    }

    /// Start submission with bindings encoded directly into the caller chain.
    pub fn submit(
        &mut self,
        chain: Q::Chain,
        queue: &ExecutionQueue,
        program: &Program,
        bindings: &[Binding<'_>],
        timeout_ns: u64,
    ) -> StartResult<Q, Submit> {
        let context = queue
            .context
            .expect("execution queue always retains context");
        let operation = Submit { context };
        let epoch = self.queue.state().epoch();
        if queue.epoch() != epoch || program.epoch() != epoch {
            return self.start_failure(chain, operation, StartErrorKind::StaleHandle);
        }
        if program.context != Some(context) || bindings.is_empty() {
            return self.start_failure(chain, operation, StartErrorKind::InvalidArgument);
        }
        let Some(info) = self.device_info else {
            return self.start_failure(chain, operation, StartErrorKind::DiscoveryRequired);
        };
        let Ok(binding_count) = u32::try_from(bindings.len()) else {
            return self.start_failure(chain, operation, StartErrorKind::DeviceLimit);
        };
        if binding_count > info.max_bindings_per_submission {
            return self.start_failure(chain, operation, StartErrorKind::DeviceLimit);
        }
        // Canonical slot order proves uniqueness in one linear pass. Keep the
        // prefix-scan fallback below so arbitrary binding order and its existing
        // validation precedence remain unchanged without allocating scratch space.
        let bindings_are_canonical = bindings.windows(2).all(|pair| pair[0].slot < pair[1].slot);
        for (index, binding) in bindings.iter().enumerate() {
            if binding.buffer.epoch() != epoch {
                return self.start_failure(chain, operation, StartErrorKind::StaleHandle);
            }
            if binding.buffer.context != context
                || !binding.range.fits(binding.buffer.desc.bytes)
                || !usage_allows(binding.buffer.desc.usage, binding.access)
                || (!bindings_are_canonical
                    && bindings[..index]
                        .iter()
                        .any(|prior| prior.slot == binding.slot))
            {
                return self.start_failure(chain, operation, StartErrorKind::InvalidArgument);
            }
        }
        let request = SubmitRequest {
            queue_id: Le64::new(queue.raw()),
            program_id: Le64::new(program.raw()),
            binding_count: Le32::new(binding_count),
            flags: Le32::new(0),
            timeout_ns: Le64::new(timeout_ns),
        };
        let payload = size_of::<SubmitRequest>() as u64
            + size_of::<WireBinding>() as u64 * u64::from(binding_count);
        self.start_frame(
            chain,
            operation,
            payload,
            RESPONSE_HEADER_BYTES + size_of::<virtio_accel_proto::SubmitResponse>() as u64,
            |chain| {
                write_wire(chain, REQUEST_HEADER_BYTES, &request)
                    .map_err(FrameWriteError::Chain)?;
                let mut offset = REQUEST_HEADER_BYTES + size_of::<SubmitRequest>() as u64;
                for binding in bindings {
                    let wire = WireBinding {
                        buffer_id: Le64::new(binding.buffer.raw()),
                        offset: Le64::new(binding.range.offset),
                        bytes: Le64::new(binding.range.bytes),
                        slot: Le32::new(binding.slot),
                        access: binding.access as u8,
                        reserved: [0; 3],
                    };
                    write_wire(chain, offset, &wire).map_err(FrameWriteError::Chain)?;
                    offset += size_of::<WireBinding>() as u64;
                }
                Ok(())
            },
        )
    }

    /// Start a nonblocking event-state poll.
    pub fn poll_event(&mut self, chain: Q::Chain, event: &Event) -> StartResult<Q, PollEvent> {
        if event.epoch() != self.queue.state().epoch() {
            return self.start_failure(chain, PollEvent, StartErrorKind::StaleHandle);
        }
        let request = ObjectPayload {
            object_id: Le64::new(event.raw()),
        };
        self.start_wire(
            chain,
            PollEvent,
            &request,
            RESPONSE_HEADER_BYTES + size_of::<WireEventState>() as u64,
        )
    }

    /// Start event cancellation.
    pub fn cancel_event(&mut self, chain: Q::Chain, event: &Event) -> StartResult<Q, CancelEvent> {
        if event.epoch() != self.queue.state().epoch() {
            return self.start_failure(chain, CancelEvent, StartErrorKind::StaleHandle);
        }
        let request = ObjectPayload {
            object_id: Le64::new(event.raw()),
        };
        self.start_wire(chain, CancelEvent, &request, RESPONSE_HEADER_BYTES)
    }

    /// Start event release, consuming the handle until completion or failure recovery.
    pub fn destroy_event(&mut self, chain: Q::Chain, event: Event) -> StartResult<Q, DestroyEvent> {
        let stale = event.epoch() != self.queue.state().epoch();
        let object = ObjectPayload {
            object_id: Le64::new(event.raw()),
        };
        let operation = DestroyEvent { event };
        if stale {
            return self.start_failure(chain, operation, StartErrorKind::StaleHandle);
        }
        self.start_wire(chain, operation, &object, RESPONSE_HEADER_BYTES)
    }

    fn start_wire<O: Operation, T: IntoBytes + Immutable>(
        &mut self,
        chain: Q::Chain,
        operation: O,
        value: &T,
        response_bytes: u64,
    ) -> StartResult<Q, O> {
        self.start_frame(
            chain,
            operation,
            size_of::<T>() as u64,
            response_bytes,
            |chain| write_wire(chain, REQUEST_HEADER_BYTES, value).map_err(FrameWriteError::Chain),
        )
    }

    fn start_frame<O: Operation>(
        &mut self,
        mut chain: Q::Chain,
        operation: O,
        payload_bytes: u64,
        response_bytes: u64,
        write_payload: impl FnOnce(
            &mut Q::Chain,
        ) -> Result<
            (),
            FrameWriteError<<Q::Chain as DriverChainBuffer>::Error>,
        >,
    ) -> StartResult<Q, O> {
        if self.health == ClientHealth::NeedsReset {
            return self.start_failure(chain, operation, StartErrorKind::NeedsReset);
        }
        if operation.requires_discovery() && self.device_info.is_none() {
            return self.start_failure(chain, operation, StartErrorKind::DiscoveryRequired);
        }
        let Some(slot) = self
            .slots
            .iter()
            .position(|slot| matches!(slot, Slot::Vacant))
        else {
            return self.start_failure(chain, operation, StartErrorKind::InflightLimit);
        };
        let readable_bytes = chain.device_readable_len();
        let writable_bytes = chain.device_writable_len();
        let Some(frame_bytes) = REQUEST_HEADER_BYTES.checked_add(payload_bytes) else {
            return self.start_failure(
                chain,
                operation,
                StartErrorKind::RequestCapacity {
                    required: u64::MAX,
                    available: readable_bytes,
                },
            );
        };
        let max_request = u64::from(self.config.wire().max_request_bytes.get());
        let max_response = u64::from(self.config.wire().max_response_bytes.get());
        if frame_bytes > max_request || frame_bytes != readable_bytes {
            return self.start_failure(
                chain,
                operation,
                StartErrorKind::RequestCapacity {
                    required: frame_bytes,
                    available: readable_bytes,
                },
            );
        }
        if response_bytes > max_response || writable_bytes < response_bytes {
            return self.start_failure(
                chain,
                operation,
                StartErrorKind::ResponseCapacity {
                    required: response_bytes,
                    available: writable_bytes,
                },
            );
        }
        let Ok(payload_bytes) = u32::try_from(payload_bytes) else {
            return self.start_failure(
                chain,
                operation,
                StartErrorKind::RequestCapacity {
                    required: frame_bytes,
                    available: readable_bytes,
                },
            );
        };
        let Some(request_id) = self.allocate_request_id() else {
            return self.start_failure(chain, operation, StartErrorKind::RequestIdExhausted);
        };
        let header = RequestHeader::new(
            operation.opcode(),
            RequestFlags::empty(),
            payload_bytes,
            request_id,
        );
        if let Err(error) = write_wire(&mut chain, 0, &header) {
            return self.start_failure(chain, operation, StartErrorKind::ChainAccess(error));
        }
        if let Err(error) = write_payload(&mut chain) {
            let kind = match error {
                FrameWriteError::Chain(error) => StartErrorKind::ChainAccess(error),
                FrameWriteError::Source(error) => StartErrorKind::SourceAccess(error),
            };
            return self.start_failure(chain, operation, kind);
        }
        let published = match self.queue.publish(chain) {
            Ok(published) => published,
            Err(error) => {
                let (chain, kind) = error.into_parts();
                return self.start_failure(chain, operation, StartErrorKind::Publish(kind));
            }
        };
        let epoch = published.id().epoch();
        self.slots[slot] = Slot::InFlight {
            request_id,
            chain_id: published.id(),
        };
        Ok(Pending {
            slot: slot as u16,
            request_id,
            epoch,
            operation,
        })
    }

    fn start_failure<O>(
        &self,
        chain: Q::Chain,
        operation: O,
        kind: StartErrorKind<Q::Error, <Q::Chain as DriverChainBuffer>::Error>,
    ) -> StartResult<Q, O> {
        Err(StartError {
            chain,
            operation,
            kind,
        })
    }

    fn allocate_request_id(&mut self) -> Option<u64> {
        for _ in 0..=self.slots.len() {
            let candidate = self.next_request_id.max(1);
            self.next_request_id = candidate.wrapping_add(1).max(1);
            if !self.slots.iter().any(|slot| {
                matches!(slot, Slot::InFlight { request_id, .. } | Slot::Completed { request_id, .. } if *request_id == candidate)
            }) {
                return Some(candidate);
            }
        }
        None
    }

    fn pending_matches<O>(&self, pending: &Pending<O>) -> bool {
        self.slots
            .get(usize::from(pending.slot))
            .is_some_and(|slot| {
                matches!(slot, Slot::InFlight { request_id, .. } | Slot::Completed { request_id, .. } if *request_id == pending.request_id)
            })
    }

    fn decode_completion<O: Operation>(
        &mut self,
        pending: Pending<O>,
        used: UsedChain<Q::Chain>,
    ) -> Completion<O, Q::Chain, <Q::Chain as DriverChainBuffer>::Error> {
        let (_, used_length, chain) = used.into_parts();
        let result = self.validate_response(&pending, &chain, used_length);
        match result {
            Ok(OperationResult::Success(output)) => {
                if O::output_requires_reset(&output) {
                    self.health = ClientHealth::NeedsReset;
                }
                if let Some(info) = O::discovered_info(&output) {
                    self.device_info = Some(info);
                }
                Completion::Success { output, chain }
            }
            Ok(OperationResult::DeviceError(status)) => {
                let disposition = O::failure_disposition(status);
                if disposition == FailureDisposition::Indeterminate {
                    self.health = ClientHealth::NeedsReset;
                }
                Completion::DeviceError {
                    status,
                    disposition,
                    operation: pending.operation,
                    chain,
                }
            }
            Err(error) => {
                self.health = ClientHealth::NeedsReset;
                Completion::InvalidResponse {
                    error,
                    operation: pending.operation,
                    chain,
                }
            }
        }
    }

    fn validate_response<O: Operation>(
        &self,
        pending: &Pending<O>,
        chain: &Q::Chain,
        used: UsedLength,
    ) -> Result<OperationResult<O::Output>, ResponseError<<Q::Chain as DriverChainBuffer>::Error>>
    {
        if u64::from(used.get()) < RESPONSE_HEADER_BYTES {
            return Err(ResponseError::UsedLength { used: used.get() });
        }
        if used.get() > self.config.wire().max_response_bytes.get() {
            return Err(ResponseError::ResponseLimit);
        }
        let mut bytes = [0_u8; size_of::<ResponseHeader>()];
        chain
            .read_device_writable(0, &mut bytes)
            .map_err(ResponseError::HeaderAccess)?;
        let header =
            read_exact::<ResponseHeader>(&bytes).map_err(|_| ResponseError::PayloadEncoding)?;
        let actual_id = header.request_id.get();
        if actual_id != pending.request_id {
            return Err(ResponseError::RequestId {
                expected: pending.request_id,
                actual: actual_id,
            });
        }
        if header.flags.get() != 0 {
            return Err(ResponseError::Flags(header.flags.get()));
        }
        let expected_used = RESPONSE_HEADER_BYTES
            .checked_add(u64::from(header.payload_bytes.get()))
            .ok_or(ResponseError::UsedLength { used: used.get() })?;
        if expected_used != u64::from(used.get()) {
            return Err(ResponseError::UsedLength { used: used.get() });
        }
        pending.operation.decode(
            chain,
            StatusCode(header.status.get()),
            header.payload_bytes.get(),
            pending.epoch,
            self.config,
        )
    }

    #[cfg(test)]
    fn set_next_request_id(&mut self, value: u64) {
        self.next_request_id = value;
    }
}

fn write_wire<C: DriverChainBuffer, T: IntoBytes + Immutable>(
    chain: &mut C,
    offset: u64,
    value: &T,
) -> Result<(), C::Error> {
    chain.write_device_readable(offset, value.as_bytes())
}

fn copy_source<C: DriverChainBuffer, R: ReadableBytes + ?Sized>(
    chain: &mut C,
    destination_offset: u64,
    source: &R,
) -> Result<(), FrameWriteError<C::Error>> {
    if let Some(bytes) = source.as_contiguous() {
        return chain
            .write_device_readable(destination_offset, bytes)
            .map_err(FrameWriteError::Chain);
    }
    let mut scratch = [0_u8; COPY_SCRATCH_BYTES];
    let mut copied = 0_u64;
    while copied < source.len() {
        let count = usize::try_from(core::cmp::min(
            source.len() - copied,
            COPY_SCRATCH_BYTES as u64,
        ))
        .expect("bounded copy chunk fits usize");
        source
            .read_at(copied, &mut scratch[..count])
            .map_err(FrameWriteError::Source)?;
        chain
            .write_device_readable(destination_offset + copied, &scratch[..count])
            .map_err(FrameWriteError::Chain)?;
        copied += count as u64;
    }
    Ok(())
}

fn usage_allows(usage: BufferUsage, access: AccessMode) -> bool {
    match access {
        AccessMode::Read => {
            usage.intersects(BufferUsage::PROGRAM_INPUT | BufferUsage::MUTABLE_STATE)
        }
        AccessMode::Write => {
            usage.intersects(BufferUsage::PROGRAM_OUTPUT | BufferUsage::MUTABLE_STATE)
        }
        AccessMode::ReadWrite => usage.contains(BufferUsage::MUTABLE_STATE),
    }
}

fn stale_operation<O>(pending: Pending<O>) -> StaleOperation<O> {
    StaleOperation {
        request_id: pending.request_id,
        epoch: pending.epoch,
        operation: pending.operation,
    }
}

#[cfg(test)]
mod tests {
    extern crate std;

    use core::num::{NonZeroU32, NonZeroU64};
    use std::vec;
    use std::vec::Vec;

    use virtio_accel_proto::{Le16, PROTOCOL_MAJOR, PROTOCOL_MINOR, SubmitResponse, WireConfig};
    use virtio_accel_split_queue::{Descriptor, DriverChain, SplitDeviceChain, SplitQueue};
    use virtio_accel_transport::{
        DeviceChain, DeviceQueue, DriverChainBuffer, QueueControl, QueueSize, WritableBytes,
    };

    use super::*;
    use crate::types::{DeviceInfoError, MemoryDomain, SubmissionOutcome};

    type TestClient = GuestClient<SplitQueue>;

    fn test_client(size: u16, max_inflight: u16, max_chain_descriptors: u16) -> TestClient {
        let size = QueueSize::new(size).unwrap();
        let mut queue = SplitQueue::new(size, max_chain_descriptors).unwrap();
        QueueControl::configure(&mut queue, size).unwrap();
        QueueControl::set_ready(&mut queue, true).unwrap();
        let config = GuestConfig::new(
            WireConfig {
                protocol_major: Le16::new(PROTOCOL_MAJOR),
                protocol_minor: Le16::new(PROTOCOL_MINOR),
                command_queue_count: Le16::new(1),
                max_chain_descriptors: Le16::new(max_chain_descriptors),
                max_request_bytes: Le32::new(4_096),
                max_response_bytes: Le32::new(4_096),
            },
            0,
            max_inflight,
        )
        .unwrap();
        GuestClient::new(queue, config).unwrap()
    }

    fn chain(request_bytes: usize, response_bytes: usize) -> DriverChain {
        DriverChain::direct(vec![
            Descriptor::readable(vec![0; request_bytes]),
            Descriptor::writable(vec![0; response_bytes]),
        ])
        .unwrap()
    }

    fn prepared_chain(prefix_bytes: usize, payload: &[u8], response_bytes: usize) -> DriverChain {
        DriverChain::direct(vec![
            Descriptor::readable(vec![0; prefix_bytes]),
            Descriptor::readable(payload.to_vec()),
            Descriptor::writable(vec![0; response_bytes]),
        ])
        .unwrap()
    }

    fn device_info() -> WireDeviceInfo {
        WireDeviceInfo {
            uuid: [0x5a; 16],
            class: Le16::new(1),
            reserved: Le16::new(0),
            vendor_id: Le32::new(0x1234),
            device_id: Le32::new(0x5678),
            capabilities: Le64::new((1 << 0) | (1 << 1) | (1 << 2) | (1 << 5)),
            max_contexts: Le32::new(8),
            max_buffers_per_context: Le32::new(8),
            max_programs_per_context: Le32::new(8),
            max_queues_per_context: Le32::new(8),
            max_events_per_context: Le32::new(8),
            max_bindings_per_submission: Le32::new(8),
            max_buffer_bytes: Le64::new(4_096),
            max_artifact_bytes: Le64::new(4_000),
        }
    }

    fn pop_device_chain(client: &mut TestClient) -> SplitDeviceChain {
        DeviceQueue::pop_available(&mut client.queue)
            .unwrap()
            .expect("published request")
    }

    fn complete_chain(
        client: &mut TestClient,
        mut chain: SplitDeviceChain,
        status: StatusCode,
        payload: &[u8],
        response_id: Option<u64>,
    ) -> Vec<u8> {
        let request = {
            let (_, request, response) = chain.io().unwrap().into_parts();
            let mut request_bytes = vec![0; request.len() as usize];
            request.read_at(0, &mut request_bytes).unwrap();
            let header =
                read_exact::<RequestHeader>(&request_bytes[..size_of::<RequestHeader>()]).unwrap();
            let response_header = ResponseHeader::new(
                status,
                payload.len() as u32,
                response_id.unwrap_or_else(|| header.request_id.get()),
            );
            response.write_at(0, response_header.as_bytes()).unwrap();
            response.write_at(RESPONSE_HEADER_BYTES, payload).unwrap();
            request_bytes
        };
        DeviceQueue::complete(
            &mut client.queue,
            chain,
            UsedLength::new((RESPONSE_HEADER_BYTES as usize + payload.len()) as u32),
        )
        .unwrap();
        request
    }

    fn complete_next(client: &mut TestClient, status: StatusCode, payload: &[u8]) -> Vec<u8> {
        let chain = pop_device_chain(client);
        complete_chain(client, chain, status, payload, None)
    }

    fn success<O: Operation>(
        client: &mut TestClient,
        pending: Pending<O>,
    ) -> (O::Output, DriverChain) {
        match client.poll(pending) {
            RequestPoll::Ready(Completion::Success { output, chain }) => (output, chain),
            RequestPoll::Pending(_) => panic!("request remained pending"),
            RequestPoll::Ready(Completion::DeviceError { .. }) => panic!("unexpected device error"),
            RequestPoll::Ready(Completion::InvalidResponse { .. }) => {
                panic!("unexpected invalid response")
            }
            RequestPoll::QueueError { .. } => panic!("unexpected queue error"),
            RequestPoll::Stale(_) => panic!("unexpected stale request"),
            RequestPoll::NeedsReset(_) => panic!("unexpected reset requirement"),
        }
    }

    fn discover(client: &mut TestClient) {
        let pending = client
            .get_device_info(chain(
                size_of::<RequestHeader>(),
                size_of::<ResponseHeader>() + size_of::<WireDeviceInfo>(),
            ))
            .unwrap();
        let info = device_info();
        complete_next(client, StatusCode::OK, info.as_bytes());
        let (decoded, _) = success(client, pending);
        assert_eq!(decoded.uuid, [0x5a; 16]);
    }

    fn object_payload(id: u64) -> ObjectPayload {
        ObjectPayload {
            object_id: Le64::new(id),
        }
    }

    #[test]
    fn out_of_order_completions_match_request_ids() {
        let mut client = test_client(16, 8, 4);
        let first = client
            .get_device_info(chain(16, 16 + size_of::<WireDeviceInfo>()))
            .unwrap();
        let second = client
            .get_device_info(chain(16, 16 + size_of::<WireDeviceInfo>()))
            .unwrap();
        let first_chain = pop_device_chain(&mut client);
        let second_chain = pop_device_chain(&mut client);
        let info = device_info();
        complete_chain(
            &mut client,
            second_chain,
            StatusCode::OK,
            info.as_bytes(),
            None,
        );

        let first = match client.poll(first) {
            RequestPoll::Pending(pending) => pending,
            _ => panic!("first request completed out of order"),
        };
        success(&mut client, second);
        complete_chain(
            &mut client,
            first_chain,
            StatusCode::OK,
            info.as_bytes(),
            None,
        );
        success(&mut client, first);
        assert_eq!(client.health(), ClientHealth::Running);
    }

    #[test]
    fn request_id_wrap_skips_live_ids() {
        let mut client = test_client(16, 8, 4);
        client.set_next_request_id(u64::MAX);
        let first = client.get_device_info(chain(16, 92)).unwrap();
        let second = client.get_device_info(chain(16, 92)).unwrap();
        client.set_next_request_id(u64::MAX);
        let third = client.get_device_info(chain(16, 92)).unwrap();
        assert_eq!(first.request_id(), u64::MAX);
        assert_eq!(second.request_id(), 1);
        assert_eq!(third.request_id(), 2);

        let next = client.queue_state().epoch().checked_next().unwrap();
        assert_eq!(client.reset(next).unwrap().count(), 3);
    }

    #[test]
    fn malformed_and_unknown_responses_never_create_values() {
        let mut client = test_client(16, 8, 4);
        let malformed = client.get_device_info(chain(16, 92)).unwrap();
        let device_chain = pop_device_chain(&mut client);
        let info = device_info();
        complete_chain(
            &mut client,
            device_chain,
            StatusCode::OK,
            info.as_bytes(),
            Some(malformed.request_id() + 1),
        );
        match client.poll(malformed) {
            RequestPoll::Ready(Completion::InvalidResponse {
                error: ResponseError::RequestId { .. },
                ..
            }) => {}
            _ => panic!("wrong request ID was accepted"),
        }
        assert_eq!(client.health(), ClientHealth::NeedsReset);
        let error = client.get_device_info(chain(16, 92)).unwrap_err();
        assert!(matches!(error.kind(), StartErrorKind::NeedsReset));

        let next = client.queue_state().epoch().checked_next().unwrap();
        client.reset(next).unwrap();
        client
            .reconfigure_queue(QueueSize::new(16).unwrap())
            .unwrap();
        let unknown = client.get_device_info(chain(16, 92)).unwrap();
        complete_next(&mut client, StatusCode(0x1234), &[]);
        match client.poll(unknown) {
            RequestPoll::Ready(Completion::DeviceError {
                status,
                disposition,
                ..
            }) => {
                assert_eq!(status, StatusCode(0x1234));
                assert_eq!(disposition, FailureDisposition::Unknown);
            }
            _ => panic!("unknown status was not preserved"),
        }
        assert!(client.device_info().is_none());
    }

    #[test]
    fn discovery_rejects_a_device_without_a_baseline_memory_domain() {
        let mut client = test_client(16, 8, 4);
        let pending = client.get_device_info(chain(16, 92)).unwrap();
        let mut info = device_info();
        info.capabilities = Le64::new(1 << 2);
        complete_next(&mut client, StatusCode::OK, info.as_bytes());

        assert!(matches!(
            client.poll(pending),
            RequestPoll::Ready(Completion::InvalidResponse {
                error: ResponseError::DeviceInfo(DeviceInfoError::MissingMemoryDomain),
                ..
            })
        ));
        assert!(client.device_info().is_none());
    }

    #[test]
    fn reset_returns_queue_and_operation_ownership() {
        let mut client = test_client(16, 8, 4);
        discover(&mut client);
        let create = client
            .create_context(chain(16 + size_of::<CreateContextRequest>(), 24))
            .unwrap();
        complete_next(&mut client, StatusCode::OK, object_payload(7).as_bytes());
        let (context, _) = success(&mut client, create);

        let rejected = client.destroy_context(chain(24, 16), context).unwrap();
        complete_next(&mut client, StatusCode::BUSY, &[]);
        let context = match client.poll(rejected) {
            RequestPoll::Ready(Completion::DeviceError {
                status: StatusCode::BUSY,
                disposition: FailureDisposition::Retryable,
                operation,
                ..
            }) => operation.into_context(),
            _ => panic!("rejected release was not returned as retryable"),
        };

        let destroy = client.destroy_context(chain(24, 16), context).unwrap();
        let next = client.queue_state().epoch().checked_next().unwrap();
        assert_eq!(client.reset(next).unwrap().count(), 1);
        let stale = match client.poll(destroy) {
            RequestPoll::Stale(pending) => pending,
            _ => panic!("reset did not stale pending release"),
        };
        assert_eq!(stale.into_operation().into_context().raw(), 7);
    }

    #[test]
    fn reset_recovers_a_completion_already_popped_from_the_used_ring() {
        let mut client = test_client(16, 8, 4);
        let pending = client.get_device_info(chain(16, 92)).unwrap();
        let info = device_info();
        complete_next(&mut client, StatusCode::OK, info.as_bytes());
        assert_eq!(
            client.pump().unwrap(),
            PumpResult::Completion {
                request_id: pending.request_id()
            }
        );

        let next = client.queue_state().epoch().checked_next().unwrap();
        assert_eq!(client.reset(next).unwrap().count(), 0);
        assert!(matches!(client.poll(pending), RequestPoll::Stale(_)));
        let recovered = client.pop_recovered_completion().unwrap();
        assert_eq!(recovered.used().get(), 92);
        assert!(client.pop_recovered_completion().is_none());
    }

    #[test]
    fn device_loss_makes_release_ownership_indeterminate() {
        let mut client = test_client(16, 8, 4);
        discover(&mut client);
        let create = client.create_context(chain(24, 24)).unwrap();
        complete_next(&mut client, StatusCode::OK, object_payload(9).as_bytes());
        let (context, _) = success(&mut client, create);

        let destroy = client.destroy_context(chain(24, 16), context).unwrap();
        complete_next(&mut client, StatusCode::DEVICE_LOST, &[]);
        match client.poll(destroy) {
            RequestPoll::Ready(Completion::DeviceError {
                status: StatusCode::DEVICE_LOST,
                disposition: FailureDisposition::Indeterminate,
                operation,
                ..
            }) => assert_eq!(operation.into_context().raw(), 9),
            _ => panic!("device loss did not preserve indeterminate ownership"),
        }
        assert_eq!(client.health(), ClientHealth::NeedsReset);
    }

    #[test]
    fn prepublication_backpressure_returns_chain_and_operation() {
        let mut client = test_client(4, 4, 2);
        let _first = client.get_device_info(chain(16, 92)).unwrap();
        let _second = client.get_device_info(chain(16, 92)).unwrap();
        let error = client.get_device_info(chain(16, 92)).unwrap_err();
        let (returned, _operation, kind) = error.into_parts();
        assert_eq!(returned.device_readable_len(), 16);
        assert!(matches!(
            kind,
            StartErrorKind::Publish(PublishErrorKind::InsufficientDescriptors)
        ));
    }

    #[test]
    fn complete_typed_lifecycle_uses_caller_owned_chain_storage() {
        let mut client = test_client(16, 8, 4);
        discover(&mut client);

        let create_context = client.create_context(chain(24, 24)).unwrap();
        complete_next(&mut client, StatusCode::OK, object_payload(1).as_bytes());
        let (context, _) = success(&mut client, create_context);

        let desc = BufferDesc::new(
            64,
            16,
            MemoryDomain::Host,
            BufferUsage::TRANSFER_SOURCE
                | BufferUsage::TRANSFER_DESTINATION
                | BufferUsage::PROGRAM_INPUT
                | BufferUsage::PROGRAM_OUTPUT,
        )
        .unwrap();
        let allocate = client
            .allocate_buffer(
                chain(16 + size_of::<AllocateBufferRequest>(), 24),
                &context,
                desc,
            )
            .unwrap();
        complete_next(&mut client, StatusCode::OK, object_payload(2).as_bytes());
        let (buffer, _) = success(&mut client, allocate);

        let copied = b"copy";
        let write = client
            .write_buffer(
                chain(16 + size_of::<TransferBufferRequest>() + copied.len(), 16),
                &buffer,
                0,
                &copied[..],
            )
            .unwrap();
        let request = complete_next(&mut client, StatusCode::OK, &[]);
        assert_eq!(&request[40..], copied);
        success(&mut client, write);

        let prepared = b"direct";
        let write = client
            .write_buffer_prepared(
                prepared_chain(40, prepared, 16),
                &buffer,
                BufferRange::new(4, prepared.len() as u64).unwrap(),
            )
            .unwrap();
        let request = complete_next(&mut client, StatusCode::OK, &[]);
        assert_eq!(&request[40..], prepared);
        success(&mut client, write);

        let read = client
            .read_buffer(chain(40, 20), &buffer, BufferRange::new(0, 4).unwrap())
            .unwrap();
        complete_next(&mut client, StatusCode::OK, b"data");
        let (read_output, read_chain) = success(&mut client, read);
        assert_eq!(read_output.bytes, 4);
        let mut read_bytes = [0; 4];
        read_chain
            .read_device_writable(RESPONSE_HEADER_BYTES, &mut read_bytes)
            .unwrap();
        assert_eq!(&read_bytes, b"data");

        let program_desc = ProgramDesc::new(
            NonZeroU32::new(1).unwrap(),
            [0; 12],
            NonZeroU64::new(32).unwrap(),
        );
        let artifact = b"program";
        let load = client
            .load_program_prepared(
                prepared_chain(16 + size_of::<LoadProgramRequest>(), artifact, 24),
                &context,
                program_desc,
                artifact.len() as u64,
            )
            .unwrap();
        let request = complete_next(&mut client, StatusCode::OK, object_payload(3).as_bytes());
        assert_eq!(&request[96..], artifact);
        let (program, _) = success(&mut client, load);

        let create_queue = client
            .create_execution_queue(chain(16 + size_of::<CreateQueueRequest>(), 24), &context)
            .unwrap();
        complete_next(&mut client, StatusCode::OK, object_payload(4).as_bytes());
        let (queue, _) = success(&mut client, create_queue);

        let bindings = [Binding {
            buffer: &buffer,
            range: BufferRange::new(0, 16).unwrap(),
            slot: 0,
            access: AccessMode::Read,
        }];
        let submit = client
            .submit(
                chain(
                    16 + size_of::<SubmitRequest>() + size_of::<WireBinding>(),
                    16 + size_of::<SubmitResponse>(),
                ),
                &queue,
                &program,
                &bindings,
                1_000_000,
            )
            .unwrap();
        let submit_response = SubmitResponse {
            event_id: Le64::new(5),
        };
        complete_next(&mut client, StatusCode::OK, submit_response.as_bytes());
        let (outcome, _) = success(&mut client, submit);
        let event = match outcome {
            SubmissionOutcome::Accepted(event) => event,
            SubmissionOutcome::Indeterminate { .. } => panic!("unexpected indeterminate submit"),
        };

        let poll = client.poll_event(chain(24, 24), &event).unwrap();
        let state = WireEventState {
            state: Le16::new(1),
            error: Le16::new(0),
            reserved: Le32::new(0),
        };
        complete_next(&mut client, StatusCode::OK, state.as_bytes());
        assert_eq!(success(&mut client, poll).0, crate::EventState::Complete);

        let cancel = client.cancel_event(chain(24, 16), &event).unwrap();
        complete_next(&mut client, StatusCode::OK, &[]);
        success(&mut client, cancel);

        let destroy_event = client.destroy_event(chain(24, 16), event).unwrap();
        complete_next(&mut client, StatusCode::OK, &[]);
        success(&mut client, destroy_event);

        let destroy_queue = client
            .destroy_execution_queue(chain(24, 16), queue)
            .unwrap();
        complete_next(&mut client, StatusCode::OK, &[]);
        success(&mut client, destroy_queue);

        let unload = client.unload_program(chain(24, 16), program).unwrap();
        complete_next(&mut client, StatusCode::OK, &[]);
        success(&mut client, unload);

        let free = client.free_buffer(chain(24, 16), buffer).unwrap();
        complete_next(&mut client, StatusCode::OK, &[]);
        success(&mut client, free);

        let destroy_context = client.destroy_context(chain(24, 16), context).unwrap();
        complete_next(&mut client, StatusCode::OK, &[]);
        success(&mut client, destroy_context);
    }

    #[test]
    fn mutable_state_allows_every_program_access_mode() {
        assert!(usage_allows(BufferUsage::MUTABLE_STATE, AccessMode::Read));
        assert!(usage_allows(BufferUsage::MUTABLE_STATE, AccessMode::Write));
        assert!(usage_allows(
            BufferUsage::MUTABLE_STATE,
            AccessMode::ReadWrite
        ));
        assert!(!usage_allows(
            BufferUsage::PROGRAM_INPUT | BufferUsage::PROGRAM_OUTPUT,
            AccessMode::ReadWrite
        ));
    }
}
