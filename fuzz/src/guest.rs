use core::cell::RefCell;
use core::mem::size_of;
use core::num::{NonZeroU32, NonZeroU64};
use std::collections::{BTreeSet, VecDeque};
use std::rc::Rc;

use virtio_accel_guest::{
    AccessMode, AllocateBuffer, Binding, Buffer, BufferDesc, BufferRange, BufferUsage, CancelEvent,
    ClientHealth, Completion, Context, CreateContext, CreateExecutionQueue, DestroyContext,
    DestroyEvent, DestroyExecutionQueue, Event, ExecutionQueue, FreeBuffer, GetDeviceInfo,
    GuestClient, GuestConfig, LoadProgram, MemoryDomain, Operation, Pending, PollEvent, Program,
    ProgramDesc, PumpResult, ReadBuffer, RequestPoll, StartResult, SubmissionOutcome, Submit,
    UnloadProgram, WriteBuffer,
};
use virtio_accel_proto::{
    AllocateBufferRequest, BASELINE_COMMAND_QUEUES, CreateContextRequest, CreateQueueRequest,
    KnownEventState, KnownOpcode, Le16, Le32, Le64, LoadProgramRequest, ObjectPayload,
    PROTOCOL_MAJOR, PROTOCOL_MINOR, RequestHeader, ResponseHeader, StatusCode, SubmitRequest,
    SubmitResponse, TransferBufferRequest, WireBinding, WireConfig, WireDeviceInfo, WireEventState,
    read_exact,
};
use virtio_accel_transport::{
    ByteAccessError, ChainId, DriverChainBuffer, DriverQueue, NotificationHint,
    NotificationRecheck, PublishError, PublishErrorKind, PublishedChain, QueueControl, QueueEpoch,
    QueueError, QueuePort, QueueSize, QueueState, ReclaimedChain, UsedChain, UsedLength,
};
use zerocopy::IntoBytes;

use crate::Input;

const MAX_SEQUENCE_BYTES: usize = 1_024;
const ACTION_BYTES: usize = 8;
const ACTION_COUNT: u8 = 17;

const QUEUE_SIZE: u16 = 16;
const MAX_INFLIGHT: u16 = 8;
const MAX_CHAIN_DESCRIPTORS: u16 = 4;
const MAX_REQUEST_BYTES: u32 = 512;
const MAX_RESPONSE_BYTES: u32 = 512;
const RING_CAPACITY: usize = 6;

const MAX_BUFFER_BYTES: u64 = 256;
const MAX_ARTIFACT_BYTES: u64 = 256;
const BUFFER_BYTES: u64 = 64;
const ARTIFACT_BYTES: usize = 8;
const READ_BYTES: u64 = 8;
const WRITE_BYTES: usize = 8;

const REQUEST_HEADER_BYTES: usize = size_of::<RequestHeader>();
const RESPONSE_HEADER_BYTES: usize = size_of::<ResponseHeader>();

/// Status values a non-conforming device may publish, including an unassigned one.
const STATUS_CHOICES: [StatusCode; 8] = [
    StatusCode::OK,
    StatusCode::INVALID_ARGUMENT,
    StatusCode::BUSY,
    StatusCode::RESOURCE_LIMIT,
    StatusCode::DEVICE_LOST,
    StatusCode::STALE_OBJECT,
    StatusCode::INTERNAL_ERROR,
    StatusCode(0x4242),
];

/// Caller-owned request/response storage with no transport validation of its own.
///
/// The harness owns both sides so a hostile device can publish arbitrary response bytes without
/// a split-ring model silently normalizing them first.
#[derive(Debug)]
pub(crate) struct FuzzChain {
    tag: u64,
    readable: Vec<u8>,
    writable: Vec<u8>,
}

impl FuzzChain {
    fn new(tag: u64, readable_bytes: usize, writable_bytes: usize) -> Self {
        Self {
            tag,
            readable: vec![0; readable_bytes],
            writable: vec![0; writable_bytes],
        }
    }
}

impl DriverChainBuffer for FuzzChain {
    type Error = ByteAccessError;

    fn device_readable_len(&self) -> u64 {
        self.readable.len() as u64
    }

    fn device_writable_len(&self) -> u64 {
        self.writable.len() as u64
    }

    fn write_device_readable(&mut self, offset: u64, source: &[u8]) -> Result<(), Self::Error> {
        let range = span(self.readable.len(), offset, source.len())?;
        self.readable[range].copy_from_slice(source);
        Ok(())
    }

    fn read_device_writable(&self, offset: u64, target: &mut [u8]) -> Result<(), Self::Error> {
        let range = span(self.writable.len(), offset, target.len())?;
        target.copy_from_slice(&self.writable[range]);
        Ok(())
    }
}

fn span(len: usize, offset: u64, bytes: usize) -> Result<core::ops::Range<usize>, ByteAccessError> {
    let start = usize::try_from(offset).map_err(|_| ByteAccessError::OutOfBounds)?;
    let end = start
        .checked_add(bytes)
        .ok_or(ByteAccessError::OutOfBounds)?;
    if end > len {
        return Err(ByteAccessError::OutOfBounds);
    }
    Ok(start..end)
}

/// Queue storage shared between the client-owned driver port and the harness device port.
struct Ring {
    state: QueueState,
    published: VecDeque<(ChainId, FuzzChain)>,
    used: VecDeque<UsedChain<FuzzChain>>,
    next_token: u64,
    notifications: bool,
    inject_pop_error: bool,
}

impl Ring {
    fn new() -> Self {
        let max_size = QueueSize::new(QUEUE_SIZE).expect("queue size is a valid power of two");
        Self {
            state: QueueState::unconfigured(max_size, QueueEpoch::INITIAL),
            published: VecDeque::new(),
            used: VecDeque::new(),
            next_token: 1,
            notifications: true,
            inject_pop_error: false,
        }
    }
}

/// Driver-side queue port backed by harness-controlled storage.
struct FuzzQueue {
    ring: Rc<RefCell<Ring>>,
}

impl QueuePort for FuzzQueue {
    fn state(&self) -> QueueState {
        self.ring.borrow().state
    }
}

impl DriverQueue for FuzzQueue {
    type Chain = FuzzChain;
    type Reclaimed = Vec<ReclaimedChain<FuzzChain>>;
    type Error = ByteAccessError;

    fn publish(
        &mut self,
        chain: Self::Chain,
    ) -> Result<PublishedChain, PublishError<Self::Chain, Self::Error>> {
        let mut ring = self.ring.borrow_mut();
        if !ring.state.ready() {
            return Err(PublishError::new(chain, PublishErrorKind::NotReady));
        }
        if ring.published.len() >= RING_CAPACITY {
            return Err(PublishError::new(chain, PublishErrorKind::QueueFull));
        }
        let id = ChainId::new(ring.state.epoch(), ring.next_token);
        ring.next_token = ring.next_token.wrapping_add(1);
        let notification = if ring.notifications {
            NotificationHint::Notify
        } else {
            NotificationHint::Suppressed
        };
        ring.published.push_back((id, chain));
        Ok(PublishedChain::new(id, notification))
    }

    fn pop_used(&mut self) -> Result<Option<UsedChain<Self::Chain>>, QueueError<Self::Error>> {
        let mut ring = self.ring.borrow_mut();
        if ring.inject_pop_error {
            ring.inject_pop_error = false;
            return Err(QueueError::Transport(ByteAccessError::OutOfBounds));
        }
        Ok(ring.used.pop_front())
    }

    fn disable_used_notifications(&mut self) -> Result<(), QueueError<Self::Error>> {
        self.ring.borrow_mut().notifications = false;
        Ok(())
    }

    fn enable_used_notifications(
        &mut self,
    ) -> Result<NotificationRecheck, QueueError<Self::Error>> {
        let mut ring = self.ring.borrow_mut();
        ring.notifications = true;
        Ok(if ring.used.is_empty() {
            NotificationRecheck::Idle
        } else {
            NotificationRecheck::WorkPending
        })
    }

    fn reset(
        &mut self,
        next_epoch: QueueEpoch,
    ) -> Result<Self::Reclaimed, QueueError<Self::Error>> {
        let mut ring = self.ring.borrow_mut();
        let current = ring.state.epoch();
        if next_epoch.get() <= current.get() {
            return Err(QueueError::ResetRace {
                operation: next_epoch,
                current,
            });
        }
        let mut reclaimed = Vec::new();
        for (id, chain) in ring.published.drain(..) {
            reclaimed.push(ReclaimedChain::new(id, chain));
        }
        for used in ring.used.drain(..) {
            let (id, _, chain) = used.into_parts();
            reclaimed.push(ReclaimedChain::new(id, chain));
        }
        ring.state = QueueState::unconfigured(ring.state.max_size(), next_epoch);
        ring.notifications = true;
        Ok(reclaimed)
    }
}

impl QueueControl for FuzzQueue {
    type Error = ByteAccessError;

    fn configure(&mut self, size: QueueSize) -> Result<(), QueueError<Self::Error>> {
        let mut ring = self.ring.borrow_mut();
        let state = QueueState::new(ring.state.max_size(), Some(size), false, ring.state.epoch())
            .map_err(QueueError::InvalidConfiguration)?;
        ring.state = state;
        Ok(())
    }

    fn set_ready(&mut self, ready: bool) -> Result<(), QueueError<Self::Error>> {
        let mut ring = self.ring.borrow_mut();
        let state = QueueState::new(
            ring.state.max_size(),
            ring.state.size(),
            ready,
            ring.state.epoch(),
        )
        .map_err(QueueError::InvalidConfiguration)?;
        ring.state = state;
        Ok(())
    }
}

type Client = GuestClient<FuzzQueue>;

/// One in-flight typed request, retained so every response decoder stays reachable.
enum AnyPending {
    GetDeviceInfo(Pending<GetDeviceInfo>),
    CreateContext(Pending<CreateContext>),
    DestroyContext(Pending<DestroyContext>),
    AllocateBuffer(Pending<AllocateBuffer>),
    FreeBuffer(Pending<FreeBuffer>),
    WriteBuffer(Pending<WriteBuffer>),
    ReadBuffer(Pending<ReadBuffer>),
    LoadProgram(Pending<LoadProgram>),
    UnloadProgram(Pending<UnloadProgram>),
    CreateExecutionQueue(Pending<CreateExecutionQueue>),
    DestroyExecutionQueue(Pending<DestroyExecutionQueue>),
    Submit(Pending<Submit>),
    PollEvent(Pending<PollEvent>),
    CancelEvent(Pending<CancelEvent>),
    DestroyEvent(Pending<DestroyEvent>),
}

/// Coverage counters that keep the harness unit tests from passing vacuously.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Progress {
    handles: u32,
    invalid_responses: u32,
    device_errors: u32,
    resets: u32,
}

/// Split a leading length-prefixed response-byte pool from the trailing action stream.
///
/// Seeds can therefore carry real canonical response frames while the fuzzer still mutates the
/// command sequence independently.
fn split_pool(data: &[u8]) -> (&[u8], &[u8]) {
    let Some((header, rest)) = data.split_at_checked(2) else {
        return (&[], data);
    };
    let requested = usize::from(u16::from_le_bytes([header[0], header[1]]));
    rest.split_at(requested.min(rest.len()))
}

struct Harness {
    client: Client,
    progress: Progress,
    pool: Vec<u8>,
    ring: Rc<RefCell<Ring>>,
    pendings: Vec<AnyPending>,
    popped: Vec<(ChainId, FuzzChain)>,
    outstanding: BTreeSet<u64>,
    next_tag: u64,
    next_object_id: u64,
    contexts: Vec<Context>,
    buffers: Vec<Buffer>,
    programs: Vec<Program>,
    queues: Vec<ExecutionQueue>,
    events: Vec<Event>,
}

pub fn fuzz_guest_client(data: &[u8]) {
    run(data);
}

fn run(data: &[u8]) -> Progress {
    let data = &data[..data.len().min(MAX_SEQUENCE_BYTES)];
    let (pool, data) = split_pool(data);
    let mut harness = Harness::new();
    harness.pool = pool.to_vec();

    for action_bytes in data.chunks(ACTION_BYTES) {
        let mut input = Input::new(action_bytes);
        let action = input.byte() % ACTION_COUNT;
        let selector = input.byte();
        let argument = input.u16();
        let entropy = u64::from(u32::from_le_bytes([
            input.byte(),
            input.byte(),
            input.byte(),
            input.byte(),
        ]));
        harness.step(action, selector, argument, entropy);
        harness.assert_invariants();
    }

    harness.drain()
}

impl Harness {
    fn new() -> Self {
        let ring = Rc::new(RefCell::new(Ring::new()));
        let mut queue = FuzzQueue {
            ring: Rc::clone(&ring),
        };
        let size = QueueSize::new(QUEUE_SIZE).expect("queue size is a valid power of two");
        QueueControl::configure(&mut queue, size).expect("initial configuration succeeds");
        QueueControl::set_ready(&mut queue, true).expect("initial readiness succeeds");
        let client = GuestClient::new(queue, config()).expect("client construction succeeds");
        Self {
            client,
            progress: Progress::default(),
            pool: Vec::new(),
            ring,
            pendings: Vec::new(),
            popped: Vec::new(),
            outstanding: BTreeSet::new(),
            next_tag: 1,
            next_object_id: 1,
            contexts: Vec::new(),
            buffers: Vec::new(),
            programs: Vec::new(),
            queues: Vec::new(),
            events: Vec::new(),
        }
    }

    fn step(&mut self, action: u8, selector: u8, argument: u16, entropy: u64) {
        match action {
            0 => self.start_get_device_info(),
            1 => self.start_create_context(),
            2 => self.start_allocate_buffer(selector, argument),
            3 => self.start_load_program(selector),
            4 => self.start_create_execution_queue(selector),
            5 => self.start_submit(selector, argument),
            6 => self.start_event_command(selector, true),
            7 => self.start_event_command(selector, false),
            8 => self.start_read_buffer(selector),
            9 => self.start_write_buffer(selector),
            10 => self.start_release(selector),
            11 => self.device_pop(),
            12 => self.device_complete(selector, argument, entropy),
            13 => self.pump(),
            14 => self.poll_one(selector),
            15 => self.reset(),
            16 => self.notifications(selector),
            _ => unreachable!("action is reduced modulo ACTION_COUNT"),
        }
    }

    fn chain(&mut self, readable_bytes: usize, writable_bytes: usize) -> FuzzChain {
        let tag = self.next_tag;
        self.next_tag = self.next_tag.wrapping_add(1);
        self.outstanding.insert(tag);
        FuzzChain::new(tag, readable_bytes, writable_bytes)
    }

    fn recover(&mut self, chain: FuzzChain) {
        assert!(
            self.outstanding.remove(&chain.tag),
            "chain {} was returned twice or was never issued",
            chain.tag
        );
    }

    fn record<O>(&mut self, result: StartResult<FuzzQueue, O>, wrap: fn(Pending<O>) -> AnyPending) {
        match result {
            Ok(pending) => {
                // Only `start_frame` can admit work, and it rejects before publishing whenever
                // health is degraded. Typed argument validation runs earlier and may reject an
                // unhealthy client's request with its own diagnostic instead, so the meaningful
                // invariant is that admission never happens, not which error is reported.
                assert_eq!(
                    self.client.health(),
                    ClientHealth::Running,
                    "an unhealthy client admitted new work"
                );
                self.pendings.push(wrap(pending));
            }
            Err(error) => {
                let (chain, _operation, _kind) = error.into_parts();
                self.recover(chain);
            }
        }
    }

    fn start_get_device_info(&mut self) {
        let chain = self.chain(
            REQUEST_HEADER_BYTES,
            RESPONSE_HEADER_BYTES + size_of::<WireDeviceInfo>(),
        );
        let result = self.client.get_device_info(chain);
        self.record(result, AnyPending::GetDeviceInfo);
    }

    fn start_create_context(&mut self) {
        let chain = self.chain(
            REQUEST_HEADER_BYTES + size_of::<CreateContextRequest>(),
            object_response_bytes(),
        );
        let result = self.client.create_context(chain);
        self.record(result, AnyPending::CreateContext);
    }

    fn start_allocate_buffer(&mut self, selector: u8, argument: u16) {
        if self.contexts.is_empty() {
            return;
        }
        let index = usize::from(selector) % self.contexts.len();
        let bytes = u64::from(argument % BUFFER_BYTES as u16) + 1;
        let Ok(desc) = BufferDesc::new(
            bytes,
            1 << (selector % 4),
            domain(selector),
            BufferUsage::all(),
        ) else {
            return;
        };
        let chain = self.chain(
            REQUEST_HEADER_BYTES + size_of::<AllocateBufferRequest>(),
            object_response_bytes(),
        );
        let context = &self.contexts[index];
        let result = self.client.allocate_buffer(chain, context, desc);
        self.record(result, AnyPending::AllocateBuffer);
    }

    fn start_load_program(&mut self, selector: u8) {
        if self.contexts.is_empty() {
            return;
        }
        let index = usize::from(selector) % self.contexts.len();
        let desc = ProgramDesc::new(
            NonZeroU32::new(1).expect("one is nonzero"),
            [0; 12],
            NonZeroU64::new(32).expect("thirty-two is nonzero"),
        );
        let chain = self.chain(
            REQUEST_HEADER_BYTES + size_of::<LoadProgramRequest>() + ARTIFACT_BYTES,
            object_response_bytes(),
        );
        let artifact = [selector; ARTIFACT_BYTES];
        let context = &self.contexts[index];
        let result = self
            .client
            .load_program(chain, context, desc, &artifact[..]);
        self.record(result, AnyPending::LoadProgram);
    }

    fn start_create_execution_queue(&mut self, selector: u8) {
        if self.contexts.is_empty() {
            return;
        }
        let index = usize::from(selector) % self.contexts.len();
        let chain = self.chain(
            REQUEST_HEADER_BYTES + size_of::<CreateQueueRequest>(),
            object_response_bytes(),
        );
        let context = &self.contexts[index];
        let result = self.client.create_execution_queue(chain, context);
        self.record(result, AnyPending::CreateExecutionQueue);
    }

    fn start_submit(&mut self, selector: u8, argument: u16) {
        if self.queues.is_empty() || self.programs.is_empty() || self.buffers.is_empty() {
            return;
        }
        let queue_index = usize::from(selector) % self.queues.len();
        let program_index = usize::from(selector.rotate_left(2)) % self.programs.len();
        let buffer_index = usize::from(selector.rotate_left(4)) % self.buffers.len();
        let bytes = self.buffers[buffer_index].desc().bytes().min(BUFFER_BYTES);
        let Ok(range) = BufferRange::new(0, bytes) else {
            return;
        };
        let chain = self.chain(
            REQUEST_HEADER_BYTES + size_of::<SubmitRequest>() + size_of::<WireBinding>(),
            RESPONSE_HEADER_BYTES + size_of::<SubmitResponse>(),
        );
        let bindings = [Binding {
            buffer: &self.buffers[buffer_index],
            range,
            slot: 0,
            access: access(selector),
        }];
        let result = self.client.submit(
            chain,
            &self.queues[queue_index],
            &self.programs[program_index],
            &bindings,
            u64::from(argument),
        );
        self.record(result, AnyPending::Submit);
    }

    fn start_event_command(&mut self, selector: u8, poll: bool) {
        if self.events.is_empty() {
            return;
        }
        let index = usize::from(selector) % self.events.len();
        if poll {
            let chain = self.chain(
                object_request_bytes(),
                RESPONSE_HEADER_BYTES + size_of::<WireEventState>(),
            );
            let event = &self.events[index];
            let result = self.client.poll_event(chain, event);
            self.record(result, AnyPending::PollEvent);
        } else {
            let chain = self.chain(object_request_bytes(), RESPONSE_HEADER_BYTES);
            let event = &self.events[index];
            let result = self.client.cancel_event(chain, event);
            self.record(result, AnyPending::CancelEvent);
        }
    }

    fn start_read_buffer(&mut self, selector: u8) {
        if self.buffers.is_empty() {
            return;
        }
        let index = usize::from(selector) % self.buffers.len();
        let buffer = &self.buffers[index];
        let bytes = READ_BYTES.min(buffer.desc().bytes());
        let Ok(range) = BufferRange::new(0, bytes) else {
            return;
        };
        let chain = self.chain(
            REQUEST_HEADER_BYTES + size_of::<TransferBufferRequest>(),
            RESPONSE_HEADER_BYTES + bytes as usize,
        );
        let buffer = &self.buffers[index];
        let result = self.client.read_buffer(chain, buffer, range);
        self.record(result, AnyPending::ReadBuffer);
    }

    fn start_write_buffer(&mut self, selector: u8) {
        if self.buffers.is_empty() {
            return;
        }
        let index = usize::from(selector) % self.buffers.len();
        let payload = [selector; WRITE_BYTES];
        let chain = self.chain(
            REQUEST_HEADER_BYTES + size_of::<TransferBufferRequest>() + WRITE_BYTES,
            RESPONSE_HEADER_BYTES,
        );
        let buffer = &self.buffers[index];
        let result = self.client.write_buffer(chain, buffer, 0, &payload[..]);
        self.record(result, AnyPending::WriteBuffer);
    }

    fn start_release(&mut self, selector: u8) {
        let chain = self.chain(object_request_bytes(), RESPONSE_HEADER_BYTES);
        match selector % 5 {
            0 if !self.events.is_empty() => {
                let event = self
                    .events
                    .swap_remove(usize::from(selector) % self.events.len());
                let result = self.client.destroy_event(chain, event);
                self.record(result, AnyPending::DestroyEvent);
            }
            1 if !self.queues.is_empty() => {
                let queue = self
                    .queues
                    .swap_remove(usize::from(selector) % self.queues.len());
                let result = self.client.destroy_execution_queue(chain, queue);
                self.record(result, AnyPending::DestroyExecutionQueue);
            }
            2 if !self.programs.is_empty() => {
                let program = self
                    .programs
                    .swap_remove(usize::from(selector) % self.programs.len());
                let result = self.client.unload_program(chain, program);
                self.record(result, AnyPending::UnloadProgram);
            }
            3 if !self.buffers.is_empty() => {
                let buffer = self
                    .buffers
                    .swap_remove(usize::from(selector) % self.buffers.len());
                let result = self.client.free_buffer(chain, buffer);
                self.record(result, AnyPending::FreeBuffer);
            }
            4 if !self.contexts.is_empty() => {
                let context = self
                    .contexts
                    .swap_remove(usize::from(selector) % self.contexts.len());
                let result = self.client.destroy_context(chain, context);
                self.record(result, AnyPending::DestroyContext);
            }
            _ => self.recover(chain),
        }
    }

    fn device_pop(&mut self) {
        let entry = self.ring.borrow_mut().published.pop_front();
        if let Some(entry) = entry {
            self.popped.push(entry);
        }
    }

    /// Publish one completion as an arbitrary, possibly non-conforming device.
    fn device_complete(&mut self, selector: u8, argument: u16, entropy: u64) {
        if self.popped.is_empty() {
            return;
        }
        let index = usize::from(selector) % self.popped.len();
        let (id, mut chain) = self.popped.swap_remove(index);

        let mut header_bytes = [0_u8; REQUEST_HEADER_BYTES];
        let request_id = if chain.readable.len() >= REQUEST_HEADER_BYTES {
            header_bytes.copy_from_slice(&chain.readable[..REQUEST_HEADER_BYTES]);
            read_exact::<RequestHeader>(&header_bytes)
                .map(|header| header.request_id.get())
                .unwrap_or(0)
        } else {
            0
        };
        let opcode = read_exact::<RequestHeader>(&header_bytes)
            .map(|header| header.opcode.get())
            .unwrap_or(0);

        let well_formed = selector & 0x80 == 0;
        let mut payload = if well_formed {
            self.canonical_payload(opcode)
        } else {
            self.pooled_payload(entropy, argument)
        };
        if well_formed {
            let max_payload = chain.writable.len().saturating_sub(RESPONSE_HEADER_BYTES);
            payload.truncate(max_payload);
        }

        let (status, flags, declared_bytes, response_id) = if well_formed {
            (StatusCode::OK, 0, payload.len() as u32, request_id)
        } else {
            (
                STATUS_CHOICES[usize::from(selector >> 4) % STATUS_CHOICES.len()],
                if selector & 0x08 == 0 { 0 } else { argument },
                u32::from(argument),
                if selector & 0x10 == 0 {
                    request_id
                } else {
                    request_id.wrapping_add(entropy | 1)
                },
            )
        };

        let mut header = ResponseHeader::new(status, declared_bytes, response_id);
        header.flags = Le16::new(flags);
        write_clamped(&mut chain.writable, 0, header.as_bytes());
        write_clamped(&mut chain.writable, RESPONSE_HEADER_BYTES, &payload);

        let used = if well_formed {
            (RESPONSE_HEADER_BYTES + payload.len()) as u32
        } else {
            u32::from(argument) % (chain.writable.len() as u32 + 8)
        };

        self.ring
            .borrow_mut()
            .used
            .push_back(UsedChain::new(id, UsedLength::new(used), chain));
    }

    /// Build the response payload a conforming device would return for `opcode`.
    fn canonical_payload(&mut self, opcode: u16) -> Vec<u8> {
        let Ok(opcode) = KnownOpcode::try_from(opcode) else {
            return Vec::new();
        };
        match opcode {
            KnownOpcode::GetDeviceInfo => Vec::from(device_info().as_bytes()),
            KnownOpcode::CreateContext
            | KnownOpcode::AllocateBuffer
            | KnownOpcode::LoadProgram
            | KnownOpcode::CreateQueue => {
                let payload = ObjectPayload {
                    object_id: Le64::new(self.allocate_object_id()),
                };
                Vec::from(payload.as_bytes())
            }
            KnownOpcode::Submit => {
                let payload = SubmitResponse {
                    event_id: Le64::new(self.allocate_object_id()),
                };
                Vec::from(payload.as_bytes())
            }
            KnownOpcode::PollEvent => {
                let payload = WireEventState {
                    state: Le16::new(KnownEventState::Complete as u16),
                    error: Le16::new(0),
                    reserved: Le32::new(0),
                };
                Vec::from(payload.as_bytes())
            }
            KnownOpcode::ReadBuffer => vec![0xa5; READ_BYTES as usize],
            KnownOpcode::DestroyContext
            | KnownOpcode::FreeBuffer
            | KnownOpcode::WriteBuffer
            | KnownOpcode::UnloadProgram
            | KnownOpcode::DestroyQueue
            | KnownOpcode::CancelEvent
            | KnownOpcode::DestroyEvent => Vec::new(),
        }
    }

    /// Take response bytes straight from the input pool so seeds can carry canonical frames.
    fn pooled_payload(&self, offset: u64, length: u16) -> Vec<u8> {
        if self.pool.is_empty() {
            return Vec::new();
        }
        let start = usize::try_from(offset % self.pool.len() as u64).unwrap_or(0);
        let available = self.pool.len() - start;
        let bytes = usize::from(length) % (available + 1);
        self.pool[start..start + bytes].to_vec()
    }

    fn note_handle(&mut self) {
        self.progress.handles = self.progress.handles.saturating_add(1);
    }

    fn allocate_object_id(&mut self) -> u64 {
        let id = self.next_object_id;
        self.next_object_id = self.next_object_id.wrapping_add(1).max(1);
        id
    }

    fn pump(&mut self) {
        if let Ok(PumpResult::UnexpectedCompletion) = self.client.pump() {
            assert_eq!(
                self.client.health(),
                ClientHealth::NeedsReset,
                "an unexpected completion left the client healthy"
            );
        }
    }

    fn poll_one(&mut self, selector: u8) {
        if self.pendings.is_empty() {
            return;
        }
        let index = usize::from(selector) % self.pendings.len();
        let epoch = self.client.queue_state().epoch();
        match self.pendings.swap_remove(index) {
            AnyPending::GetDeviceInfo(pending) => {
                assert_stale_epoch(&pending, epoch);
                match self.client.poll(pending) {
                    RequestPoll::Ready(completion) => {
                        self.finish(completion);
                    }
                    other => self.retain(other, AnyPending::GetDeviceInfo),
                }
            }
            AnyPending::CreateContext(pending) => match self.client.poll(pending) {
                RequestPoll::Ready(completion) => {
                    if let Some(context) = self.finish(completion) {
                        self.contexts.push(context);
                        self.note_handle();
                    }
                }
                other => self.retain(other, AnyPending::CreateContext),
            },
            AnyPending::AllocateBuffer(pending) => match self.client.poll(pending) {
                RequestPoll::Ready(completion) => {
                    if let Some(buffer) = self.finish(completion) {
                        self.buffers.push(buffer);
                        self.note_handle();
                    }
                }
                other => self.retain(other, AnyPending::AllocateBuffer),
            },
            AnyPending::LoadProgram(pending) => match self.client.poll(pending) {
                RequestPoll::Ready(completion) => {
                    if let Some(program) = self.finish(completion) {
                        self.programs.push(program);
                        self.note_handle();
                    }
                }
                other => self.retain(other, AnyPending::LoadProgram),
            },
            AnyPending::CreateExecutionQueue(pending) => match self.client.poll(pending) {
                RequestPoll::Ready(completion) => {
                    if let Some(queue) = self.finish(completion) {
                        self.queues.push(queue);
                        self.note_handle();
                    }
                }
                other => self.retain(other, AnyPending::CreateExecutionQueue),
            },
            AnyPending::Submit(pending) => match self.client.poll(pending) {
                RequestPoll::Ready(completion) => {
                    if let Some(outcome) = self.finish(completion) {
                        let event = match outcome {
                            SubmissionOutcome::Accepted(event) => event,
                            SubmissionOutcome::Indeterminate { event, .. } => event,
                        };
                        self.events.push(event);
                        self.note_handle();
                    }
                }
                other => self.retain(other, AnyPending::Submit),
            },
            AnyPending::PollEvent(pending) => match self.client.poll(pending) {
                RequestPoll::Ready(completion) => {
                    self.finish(completion);
                }
                other => self.retain(other, AnyPending::PollEvent),
            },
            AnyPending::DestroyContext(pending) => match self.client.poll(pending) {
                RequestPoll::Ready(completion) => {
                    self.finish(completion);
                }
                other => self.retain(other, AnyPending::DestroyContext),
            },
            AnyPending::FreeBuffer(pending) => match self.client.poll(pending) {
                RequestPoll::Ready(completion) => {
                    self.finish(completion);
                }
                other => self.retain(other, AnyPending::FreeBuffer),
            },
            AnyPending::WriteBuffer(pending) => match self.client.poll(pending) {
                RequestPoll::Ready(completion) => {
                    self.finish(completion);
                }
                other => self.retain(other, AnyPending::WriteBuffer),
            },
            AnyPending::ReadBuffer(pending) => match self.client.poll(pending) {
                RequestPoll::Ready(completion) => {
                    self.finish(completion);
                }
                other => self.retain(other, AnyPending::ReadBuffer),
            },
            AnyPending::UnloadProgram(pending) => match self.client.poll(pending) {
                RequestPoll::Ready(completion) => {
                    self.finish(completion);
                }
                other => self.retain(other, AnyPending::UnloadProgram),
            },
            AnyPending::DestroyExecutionQueue(pending) => match self.client.poll(pending) {
                RequestPoll::Ready(completion) => {
                    self.finish(completion);
                }
                other => self.retain(other, AnyPending::DestroyExecutionQueue),
            },
            AnyPending::CancelEvent(pending) => match self.client.poll(pending) {
                RequestPoll::Ready(completion) => {
                    self.finish(completion);
                }
                other => self.retain(other, AnyPending::CancelEvent),
            },
            AnyPending::DestroyEvent(pending) => match self.client.poll(pending) {
                RequestPoll::Ready(completion) => {
                    self.finish(completion);
                }
                other => self.retain(other, AnyPending::DestroyEvent),
            },
        }
    }

    /// Reclaim the caller chain from a terminal completion and expose the typed output.
    fn finish<O: Operation>(
        &mut self,
        completion: Completion<O, FuzzChain, ByteAccessError>,
    ) -> Option<O::Output> {
        match completion {
            Completion::Success { output, chain } => {
                self.recover(chain);
                Some(output)
            }
            Completion::DeviceError { chain, .. } => {
                self.recover(chain);
                self.progress.device_errors = self.progress.device_errors.saturating_add(1);
                None
            }
            Completion::InvalidResponse { chain, .. } => {
                self.recover(chain);
                self.progress.invalid_responses = self.progress.invalid_responses.saturating_add(1);
                assert_eq!(
                    self.client.health(),
                    ClientHealth::NeedsReset,
                    "a malformed response left the client healthy"
                );
                None
            }
        }
    }

    fn retain<O: Operation>(
        &mut self,
        poll: RequestPoll<O, FuzzChain, ByteAccessError, ByteAccessError>,
        wrap: fn(Pending<O>) -> AnyPending,
    ) {
        match poll {
            RequestPoll::Pending(pending) | RequestPoll::NeedsReset(pending) => {
                self.pendings.push(wrap(pending));
            }
            RequestPoll::QueueError { pending, .. } => self.pendings.push(wrap(pending)),
            RequestPoll::Stale(_) => {}
            RequestPoll::Ready(_) => unreachable!("terminal completions are handled by the caller"),
        }
    }

    fn reset(&mut self) {
        let Some(next) = self.client.queue_state().epoch().checked_next() else {
            return;
        };
        let reclaimed = match self.client.reset(next) {
            Ok(reclaimed) => reclaimed,
            Err(_) => return,
        };
        for chain in reclaimed {
            let (_, chain) = chain.into_parts();
            self.recover(chain);
        }
        for (_, chain) in core::mem::take(&mut self.popped) {
            self.recover(chain);
        }
        while let Some(used) = self.client.pop_recovered_completion() {
            let (_, _, chain) = used.into_parts();
            self.recover(chain);
        }
        if let Some(used) = self.client.take_unexpected_completion() {
            let (_, _, chain) = used.into_parts();
            self.recover(chain);
        }

        assert_eq!(
            self.client.health(),
            ClientHealth::Running,
            "reset did not restore client health"
        );
        assert!(
            self.client.device_info().is_none(),
            "reset retained discovery from an invalidated epoch"
        );
        assert_eq!(self.client.queue_state().epoch(), next);
        self.progress.resets = self.progress.resets.saturating_add(1);

        self.contexts.clear();
        self.buffers.clear();
        self.programs.clear();
        self.queues.clear();
        self.events.clear();

        let size = QueueSize::new(QUEUE_SIZE).expect("queue size is a valid power of two");
        self.client
            .reconfigure_queue(size)
            .expect("reconfiguration after reset succeeds");

        // Every pending token belongs to the invalidated epoch and must never resolve again.
        for pending in core::mem::take(&mut self.pendings) {
            assert_stale(&mut self.client, pending);
        }
    }

    fn notifications(&mut self, selector: u8) {
        // Bit 1 injects a one-shot transport error on the next pop_used().
        self.ring.borrow_mut().inject_pop_error = selector & 2 != 0;
        if selector & 1 == 0 {
            let _ = self.client.disable_used_notifications();
        } else {
            let _ = self.client.enable_used_notifications();
        }
    }

    fn assert_invariants(&self) {
        assert!(
            self.pendings.len() <= usize::from(MAX_INFLIGHT),
            "client admitted {} requests beyond the {MAX_INFLIGHT} in-flight bound",
            self.pendings.len()
        );
        let epoch = self.client.queue_state().epoch();
        assert!(
            self.ring
                .borrow()
                .published
                .iter()
                .all(|(id, _)| id.epoch() == epoch),
            "a published chain outlived its queue epoch"
        );
    }

    /// Return every caller chain and prove none was retained by the client.
    fn drain(mut self) -> Progress {
        self.reset();
        for (_, chain) in core::mem::take(&mut self.popped) {
            self.recover(chain);
        }
        let leftover = core::mem::take(&mut self.ring.borrow_mut().published);
        for (_, chain) in leftover {
            self.recover(chain);
        }
        let leftover = core::mem::take(&mut self.ring.borrow_mut().used);
        for used in leftover {
            let (_, _, chain) = used.into_parts();
            self.recover(chain);
        }
        assert!(
            self.outstanding.is_empty(),
            "client retained {} caller chains after full teardown",
            self.outstanding.len()
        );
        self.progress
    }
}

fn assert_stale(client: &mut Client, pending: AnyPending) {
    macro_rules! stale {
        ($pending:expr) => {
            assert!(
                matches!(client.poll($pending), RequestPoll::Stale(_)),
                "an invalidated epoch resolved a pending request"
            )
        };
    }
    match pending {
        AnyPending::GetDeviceInfo(pending) => stale!(pending),
        AnyPending::CreateContext(pending) => stale!(pending),
        AnyPending::DestroyContext(pending) => stale!(pending),
        AnyPending::AllocateBuffer(pending) => stale!(pending),
        AnyPending::FreeBuffer(pending) => stale!(pending),
        AnyPending::WriteBuffer(pending) => stale!(pending),
        AnyPending::ReadBuffer(pending) => stale!(pending),
        AnyPending::LoadProgram(pending) => stale!(pending),
        AnyPending::UnloadProgram(pending) => stale!(pending),
        AnyPending::CreateExecutionQueue(pending) => stale!(pending),
        AnyPending::DestroyExecutionQueue(pending) => stale!(pending),
        AnyPending::Submit(pending) => stale!(pending),
        AnyPending::PollEvent(pending) => stale!(pending),
        AnyPending::CancelEvent(pending) => stale!(pending),
        AnyPending::DestroyEvent(pending) => stale!(pending),
    }
}

fn assert_stale_epoch<O>(pending: &Pending<O>, epoch: QueueEpoch) {
    assert!(
        pending.epoch().get() <= epoch.get(),
        "a pending token claims a future queue epoch"
    );
}

fn write_clamped(target: &mut [u8], offset: usize, source: &[u8]) {
    let Some(window) = target.get_mut(offset..) else {
        return;
    };
    let count = window.len().min(source.len());
    window[..count].copy_from_slice(&source[..count]);
}

fn object_request_bytes() -> usize {
    REQUEST_HEADER_BYTES + size_of::<ObjectPayload>()
}

fn object_response_bytes() -> usize {
    RESPONSE_HEADER_BYTES + size_of::<ObjectPayload>()
}

fn domain(selector: u8) -> MemoryDomain {
    match selector % 3 {
        0 => MemoryDomain::Host,
        1 => MemoryDomain::Device,
        _ => MemoryDomain::Shared,
    }
}

fn access(selector: u8) -> AccessMode {
    match selector % 3 {
        0 => AccessMode::Read,
        1 => AccessMode::Write,
        _ => AccessMode::ReadWrite,
    }
}

fn config() -> GuestConfig {
    let wire = WireConfig {
        protocol_major: Le16::new(PROTOCOL_MAJOR),
        protocol_minor: Le16::new(PROTOCOL_MINOR),
        command_queue_count: Le16::new(BASELINE_COMMAND_QUEUES),
        max_chain_descriptors: Le16::new(MAX_CHAIN_DESCRIPTORS),
        max_request_bytes: Le32::new(MAX_REQUEST_BYTES),
        max_response_bytes: Le32::new(MAX_RESPONSE_BYTES),
    };
    GuestConfig::new(wire, 0, MAX_INFLIGHT).expect("harness configuration is valid")
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
        max_buffer_bytes: Le64::new(MAX_BUFFER_BYTES),
        max_artifact_bytes: Le64::new(MAX_ARTIFACT_BYTES),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build an input with an explicit response-byte pool followed by zero-argument actions.
    fn input(pool: &[u8], actions: &[u8]) -> Vec<u8> {
        let mut bytes = Vec::from((pool.len() as u16).to_le_bytes());
        bytes.extend_from_slice(pool);
        for action in actions {
            bytes.extend_from_slice(&[*action, 0, 0, 0, 0, 0, 0, 0]);
        }
        bytes
    }

    #[test]
    fn well_formed_lifecycle_builds_the_whole_object_graph() {
        // Discover, then create a context, buffer, program, queue, and event against a conforming
        // device. Each object needs its own start/pop/complete/poll quartet.
        let sequence = input(
            &[],
            &[
                0, 11, 12, 14, 1, 11, 12, 14, 2, 11, 12, 14, 3, 11, 12, 14, 4, 11, 12, 14, 5, 11,
                12, 14, 6, 11, 12, 14,
            ],
        );
        let progress = run(&sequence);
        assert_eq!(
            progress.handles, 5,
            "conforming lifecycle did not build the full object graph: {progress:?}"
        );
        assert_eq!(progress.invalid_responses, 0);
    }

    #[test]
    fn hostile_completions_are_rejected_without_resolving_stale_epochs() {
        let mut bytes = Vec::from(4_u16.to_le_bytes());
        bytes.extend_from_slice(&[0xde, 0xad, 0xbe, 0xef]);
        bytes.extend_from_slice(&[0, 0, 0, 0, 0, 0, 0, 0]);
        bytes.extend_from_slice(&[11, 0, 0, 0, 0, 0, 0, 0]);
        // Selector bit 0x80 selects a non-conforming response; bit 0x10 corrupts the request ID.
        bytes.extend_from_slice(&[12, 0x90, 0x60, 0x00, 0x33, 0x44, 0x55, 0x66]);
        bytes.extend_from_slice(&[14, 0, 0, 0, 0, 0, 0, 0]);
        bytes.extend_from_slice(&[15, 0, 0, 0, 0, 0, 0, 0]);
        let progress = run(&bytes);
        assert_eq!(
            progress.invalid_responses, 1,
            "a corrupted response was not classified as invalid: {progress:?}"
        );
        assert_eq!(progress.resets, 2, "final teardown always resets once more");
    }

    #[test]
    fn empty_and_truncated_inputs_are_accepted() {
        fuzz_guest_client(&[]);
        fuzz_guest_client(&[13]);
        fuzz_guest_client(&[12, 0x80, 0xff]);
    }
}
