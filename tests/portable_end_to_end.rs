use std::cell::{Cell, RefCell};
use std::collections::BTreeSet;
use std::mem::size_of;
use std::num::{NonZeroU32, NonZeroU64};
use std::rc::Rc;

use virtio_accel::core::{
    Accelerator, AllocatedBuffer, ArtifactRef, BackendError, BindingRef as BackendBindingRef,
    BufferDesc as BackendBufferDesc, ByteSource, ContextDesc, DeviceInfo as BackendDeviceInfo,
    EventState as BackendEventState, QueueDesc, ReleaseFailure, SubmitFailure, Timeout,
    TransportByteSink, TransportByteSource,
};
use virtio_accel::device::{
    CommandOutcome, CommandProcessor, ObjectNamespace, ResetDisposition, ResetReport,
};
use virtio_accel::guest::{
    AccessMode, Binding, Buffer, BufferDesc, BufferRange, BufferUsage, ClientHealth, Completion,
    Context, Event, EventState, ExecutionQueue, FailureDisposition, GuestClient, MemoryDomain,
    Operation, Pending, Program, ProgramDesc, RequestPoll, ResponseError, StartErrorKind,
    SubmissionOutcome,
};
use virtio_accel::proto::{
    AllocateBufferRequest, CreateContextRequest, CreateQueueRequest, KnownOpcode, Le16, Le32,
    LoadProgramRequest, ObjectPayload, PROTOCOL_MAJOR, PROTOCOL_MINOR, RequestHeader,
    ResponseHeader, StatusCode, SubmitRequest, TransferBufferRequest, WireBinding, WireConfig,
    WireDeviceInfo, WireEventState, read_exact,
};
use virtio_accel::split_queue::{
    Descriptor, DriverChain, ReclaimedChains, SplitDeviceChain, SplitQueue, SplitQueueError,
};
use virtio_accel::transport::{
    DeviceChain, DeviceQueue, DriverChainBuffer, DriverQueue, NotificationHint,
    NotificationRecheck, PublishError, PublishedChain, QueueControl, QueueEpoch, QueueError,
    QueuePort, QueueSize, QueueState, RegionDirection, UsedChain, UsedLength,
};
use virtio_accel_mock::{
    MockAccelerator, MockBuffer, MockContext, MockEvent, MockProgram, MockQueue,
};

const RESPONSE_HEADER_BYTES: u64 = size_of::<ResponseHeader>() as u64;
const STANDARD_QUEUE_SIZE: u16 = 32;
const STANDARD_MAX_INFLIGHT: u16 = 16;
const STANDARD_MAX_DESCRIPTORS: u16 = 8;

const SUBMIT_AUTO_COMPLETE: u8 = 1;
const SUBMIT_DEADLINE: u8 = 2;
const POLL_DEVICE_LOST: u8 = 1;

const SERVICE_IMMEDIATE: u8 = 0;
const SERVICE_DEFERRED: u8 = 1;
const SERVICE_REVERSE_PAIR: u8 = 2;

#[derive(Default)]
struct BackendControl {
    submit_mode: Cell<u8>,
    poll_mode: Cell<u8>,
}

impl BackendControl {
    fn auto_complete(&self) {
        self.submit_mode.set(SUBMIT_AUTO_COMPLETE);
    }

    fn reject_deadlines(&self) {
        self.submit_mode.set(SUBMIT_DEADLINE);
    }

    fn lose_device_on_poll(&self) {
        self.poll_mode.set(POLL_DEVICE_LOST);
    }
}

struct ScenarioBackend {
    inner: MockAccelerator,
    control: Rc<BackendControl>,
}

impl ScenarioBackend {
    fn new(control: Rc<BackendControl>) -> Self {
        Self {
            inner: MockAccelerator::default(),
            control,
        }
    }
}

impl Accelerator for ScenarioBackend {
    type Context = MockContext;
    type Buffer = MockBuffer;
    type Program = MockProgram;
    type Queue = MockQueue;
    type Event = MockEvent;

    fn device_info(&self) -> Result<BackendDeviceInfo, BackendError> {
        let mut info = self.inner.device_info()?;
        info.limits.max_buffer_bytes = 4_096;
        info.limits.max_artifact_bytes = 4_000;
        Ok(info)
    }

    fn create_context(&self, desc: ContextDesc) -> Result<Self::Context, BackendError> {
        self.inner.create_context(desc)
    }

    fn destroy_context(&self, context: Self::Context) -> Result<(), ReleaseFailure<Self::Context>> {
        self.inner.destroy_context(context)
    }

    fn allocate_buffer(
        &self,
        context: &Self::Context,
        desc: BackendBufferDesc,
    ) -> Result<AllocatedBuffer<Self::Buffer>, BackendError> {
        self.inner.allocate_buffer(context, desc)
    }

    fn write_buffer(
        &self,
        buffer: &mut Self::Buffer,
        offset: u64,
        data: &dyn ByteSource,
    ) -> Result<(), BackendError> {
        self.inner.write_buffer(buffer, offset, data)
    }

    fn read_buffer(
        &self,
        buffer: &Self::Buffer,
        offset: u64,
        data: &mut dyn virtio_accel::core::ByteSink,
    ) -> Result<(), BackendError> {
        self.inner.read_buffer(buffer, offset, data)
    }

    fn free_buffer(&self, buffer: Self::Buffer) -> Result<(), ReleaseFailure<Self::Buffer>> {
        self.inner.free_buffer(buffer)
    }

    fn load_program(
        &self,
        context: &Self::Context,
        artifact: ArtifactRef<'_>,
    ) -> Result<Self::Program, BackendError> {
        self.inner.load_program(context, artifact)
    }

    fn unload_program(&self, program: Self::Program) -> Result<(), ReleaseFailure<Self::Program>> {
        self.inner.unload_program(program)
    }

    fn create_queue(
        &self,
        context: &Self::Context,
        desc: QueueDesc,
    ) -> Result<Self::Queue, BackendError> {
        self.inner.create_queue(context, desc)
    }

    fn destroy_queue(&self, queue: Self::Queue) -> Result<(), ReleaseFailure<Self::Queue>> {
        self.inner.destroy_queue(queue)
    }

    fn submit(
        &self,
        queue: &Self::Queue,
        program: &Self::Program,
        bindings: &[BackendBindingRef<'_, Self::Buffer>],
        timeout: Timeout,
    ) -> Result<Self::Event, SubmitFailure<Self::Event>> {
        let mode = self.control.submit_mode.get();
        if mode == SUBMIT_DEADLINE && matches!(timeout, Timeout::AfterNs(_)) {
            return Err(SubmitFailure::Rejected(BackendError::DeadlineExpired));
        }
        let event = self.inner.submit(queue, program, bindings, timeout)?;
        if mode == SUBMIT_AUTO_COMPLETE {
            self.inner
                .complete(&event)
                .expect("fresh mock event completes exactly once");
        }
        Ok(event)
    }

    fn poll_event(&self, event: &Self::Event) -> Result<BackendEventState, BackendError> {
        if self.control.poll_mode.get() == POLL_DEVICE_LOST {
            return Err(BackendError::DeviceLost);
        }
        self.inner.poll_event(event)
    }

    fn cancel_event(&self, event: &Self::Event) -> Result<(), BackendError> {
        self.inner.cancel_event(event)
    }

    fn destroy_event(&self, event: Self::Event) -> Result<(), ReleaseFailure<Self::Event>> {
        self.inner.destroy_event(event)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct TraceEntry {
    request_id: u64,
    opcode: u16,
    status: u16,
    used: u32,
    readable_segments: u16,
    writable_segments: u16,
}

struct HarnessControl {
    service_mode: Cell<u8>,
    short_next_response: Cell<bool>,
    service_on_notification_disable: Cell<bool>,
    suppress_available_notifications: Cell<bool>,
    trace: RefCell<Vec<TraceEntry>>,
    publication_hints: RefCell<Vec<NotificationHint>>,
    completion_hints: RefCell<Vec<NotificationHint>>,
    last_reset: Cell<Option<ResetReport>>,
}

impl Default for HarnessControl {
    fn default() -> Self {
        Self {
            service_mode: Cell::new(SERVICE_IMMEDIATE),
            short_next_response: Cell::new(false),
            service_on_notification_disable: Cell::new(false),
            suppress_available_notifications: Cell::new(false),
            trace: RefCell::new(Vec::new()),
            publication_hints: RefCell::new(Vec::new()),
            completion_hints: RefCell::new(Vec::new()),
            last_reset: Cell::new(None),
        }
    }
}

impl HarnessControl {
    fn defer(&self) {
        self.service_mode.set(SERVICE_DEFERRED);
    }

    fn service_immediately(&self) {
        self.service_mode.set(SERVICE_IMMEDIATE);
    }

    fn reverse_next_pair(&self) {
        self.service_mode.set(SERVICE_REVERSE_PAIR);
    }

    fn shorten_next_response(&self) {
        self.short_next_response.set(true);
    }

    fn trace(&self) -> Vec<TraceEntry> {
        self.trace.borrow().clone()
    }
}

struct EndToEndQueue {
    queue: SplitQueue,
    processor: CommandProcessor<ScenarioBackend>,
    held: Vec<SplitDeviceChain>,
    control: Rc<HarnessControl>,
    available_notifications_suppressed: bool,
}

impl EndToEndQueue {
    fn process_chain(&mut self, mut chain: SplitDeviceChain) {
        let (entry, used) = {
            let (regions, request, response) = chain.io().unwrap().into_parts();
            let source = TransportByteSource::new(request);
            let mut sink = TransportByteSink::new(response);
            let mut header_bytes = [0_u8; size_of::<RequestHeader>()];
            ByteSource::read_at(&source, 0, &mut header_bytes).unwrap();
            let header = read_exact::<RequestHeader>(&header_bytes).unwrap();
            let readable_segments = regions
                .iter()
                .filter(|region| region.direction == RegionDirection::DeviceReadable)
                .count();
            let writable_segments = regions
                .iter()
                .filter(|region| region.direction == RegionDirection::DeviceWritable)
                .count();
            let outcome = self
                .processor
                .process(regions, &source, &mut sink)
                .expect("guest-generated command frame is processable");
            let CommandOutcome::Response { status, used, .. } = outcome else {
                panic!("guest generated an unusable command frame")
            };
            (
                TraceEntry {
                    request_id: header.request_id.get(),
                    opcode: header.opcode.get(),
                    status: status.0,
                    used,
                    readable_segments: u16::try_from(readable_segments).unwrap(),
                    writable_segments: u16::try_from(writable_segments).unwrap(),
                },
                used,
            )
        };

        let mut entry = entry;
        let used = if self.control.short_next_response.replace(false) {
            used.checked_sub(1).expect("all responses contain a header")
        } else {
            used
        };
        entry.used = used;
        self.control.trace.borrow_mut().push(entry);
        let hint = DeviceQueue::complete(&mut self.queue, chain, UsedLength::new(used)).unwrap();
        self.control.completion_hints.borrow_mut().push(hint);
    }

    fn service_one(&mut self) -> bool {
        let Some(chain) = DeviceQueue::pop_available(&mut self.queue).unwrap() else {
            return false;
        };
        self.process_chain(chain);
        true
    }

    fn service_for_poll(&mut self) {
        match self.control.service_mode.get() {
            SERVICE_IMMEDIATE => {
                self.service_one();
            }
            SERVICE_DEFERRED => {}
            SERVICE_REVERSE_PAIR => {
                while self.held.len() < 2 {
                    let Some(chain) = DeviceQueue::pop_available(&mut self.queue).unwrap() else {
                        break;
                    };
                    self.held.push(chain);
                }
                if self.held.len() == 2 {
                    while let Some(chain) = self.held.pop() {
                        self.process_chain(chain);
                    }
                    self.control.service_immediately();
                }
            }
            _ => panic!("unknown service mode"),
        }
    }

    fn set_available_notification_mode(&mut self) {
        let suppress = self.control.suppress_available_notifications.get();
        if suppress == self.available_notifications_suppressed {
            return;
        }
        if suppress {
            DeviceQueue::disable_available_notifications(&mut self.queue).unwrap();
        } else {
            DeviceQueue::enable_available_notifications(&mut self.queue).unwrap();
        }
        self.available_notifications_suppressed = suppress;
    }
}

impl QueuePort for EndToEndQueue {
    fn state(&self) -> QueueState {
        self.queue.state()
    }
}

impl QueueControl for EndToEndQueue {
    type Error = SplitQueueError;

    fn configure(&mut self, size: QueueSize) -> Result<(), QueueError<Self::Error>> {
        QueueControl::configure(&mut self.queue, size)
    }

    fn set_ready(&mut self, ready: bool) -> Result<(), QueueError<Self::Error>> {
        QueueControl::set_ready(&mut self.queue, ready)
    }
}

impl DriverQueue for EndToEndQueue {
    type Chain = DriverChain;
    type Reclaimed = ReclaimedChains;
    type Error = SplitQueueError;

    fn publish(
        &mut self,
        chain: Self::Chain,
    ) -> Result<PublishedChain, PublishError<Self::Chain, Self::Error>> {
        self.set_available_notification_mode();
        let published = DriverQueue::publish(&mut self.queue, chain)?;
        self.control
            .publication_hints
            .borrow_mut()
            .push(published.notification());
        Ok(published)
    }

    fn pop_used(&mut self) -> Result<Option<UsedChain<Self::Chain>>, QueueError<Self::Error>> {
        self.service_for_poll();
        DriverQueue::pop_used(&mut self.queue)
    }

    fn disable_used_notifications(&mut self) -> Result<(), QueueError<Self::Error>> {
        DriverQueue::disable_used_notifications(&mut self.queue)?;
        if self.control.service_on_notification_disable.replace(false) {
            self.service_one();
        }
        Ok(())
    }

    fn enable_used_notifications(
        &mut self,
    ) -> Result<NotificationRecheck, QueueError<Self::Error>> {
        DriverQueue::enable_used_notifications(&mut self.queue)
    }

    fn reset(
        &mut self,
        next_epoch: QueueEpoch,
    ) -> Result<Self::Reclaimed, QueueError<Self::Error>> {
        let reclaimed = DriverQueue::reset(&mut self.queue, next_epoch)?;
        self.held.clear();
        let namespace = u16::try_from(next_epoch.get())
            .ok()
            .and_then(ObjectNamespace::new)
            .expect("end-to-end test epochs fit object namespaces");
        let report = self
            .processor
            .reset(namespace)
            .expect("new queue epoch provides a fresh object namespace");
        self.control.last_reset.set(Some(report));
        self.control.service_immediately();
        self.available_notifications_suppressed = false;
        Ok(reclaimed)
    }
}

type TestClient = GuestClient<EndToEndQueue>;

fn wire_config(max_descriptors: u16) -> WireConfig {
    WireConfig {
        protocol_major: Le16::new(PROTOCOL_MAJOR),
        protocol_minor: Le16::new(PROTOCOL_MINOR),
        command_queue_count: Le16::new(1),
        max_chain_descriptors: Le16::new(max_descriptors),
        max_request_bytes: Le32::new(4_096),
        max_response_bytes: Le32::new(4_096),
    }
}

fn stack(
    queue_size: u16,
    max_inflight: u16,
    max_descriptors: u16,
) -> (TestClient, Rc<HarnessControl>, Rc<BackendControl>) {
    let size = QueueSize::new(queue_size).unwrap();
    let mut queue = SplitQueue::new(size, max_descriptors).unwrap();
    QueueControl::configure(&mut queue, size).unwrap();
    QueueControl::set_ready(&mut queue, true).unwrap();

    let config = wire_config(max_descriptors);
    let backend_control = Rc::new(BackendControl::default());
    let backend = ScenarioBackend::new(Rc::clone(&backend_control));
    let processor =
        CommandProcessor::new(backend, &config, ObjectNamespace::new(1).unwrap()).unwrap();
    let harness_control = Rc::new(HarnessControl::default());
    let queue = EndToEndQueue {
        queue,
        processor,
        held: Vec::new(),
        control: Rc::clone(&harness_control),
        available_notifications_suppressed: false,
    };
    let guest_config = virtio_accel::guest::GuestConfig::new(config, 0, max_inflight).unwrap();
    (
        GuestClient::new(queue, guest_config).unwrap(),
        harness_control,
        backend_control,
    )
}

fn standard_stack() -> (TestClient, Rc<HarnessControl>, Rc<BackendControl>) {
    stack(
        STANDARD_QUEUE_SIZE,
        STANDARD_MAX_INFLIGHT,
        STANDARD_MAX_DESCRIPTORS,
    )
}

fn chain(request_bytes: usize, response_bytes: usize) -> DriverChain {
    DriverChain::direct(vec![
        Descriptor::readable(vec![0; request_bytes]),
        Descriptor::writable(vec![0; response_bytes]),
    ])
    .unwrap()
}

fn segmented_chain(request_bytes: usize, response_bytes: usize) -> DriverChain {
    assert!(request_bytes > 3 && response_bytes > 5);
    DriverChain::direct(vec![
        Descriptor::readable(vec![0; 3]),
        Descriptor::readable(vec![0; request_bytes - 3]),
        Descriptor::writable(vec![0; 5]),
        Descriptor::writable(vec![0; response_bytes - 5]),
    ])
    .unwrap()
}

fn prepared_chain(prefix_bytes: usize, payload: &[u8], response_bytes: usize) -> DriverChain {
    assert!(prefix_bytes > 7 && response_bytes > 5);
    DriverChain::direct(vec![
        Descriptor::readable(vec![0; 7]),
        Descriptor::readable(vec![0; prefix_bytes - 7]),
        Descriptor::readable(payload.to_vec()),
        Descriptor::writable(vec![0; 5]),
        Descriptor::writable(vec![0; response_bytes - 5]),
    ])
    .unwrap()
}

fn success<O: Operation>(client: &mut TestClient, pending: Pending<O>) -> (O::Output, DriverChain) {
    match client.poll(pending) {
        RequestPoll::Ready(Completion::Success { output, chain }) => (output, chain),
        RequestPoll::Pending(_) => panic!("request remained pending"),
        RequestPoll::Ready(Completion::DeviceError { status, .. }) => {
            panic!("unexpected device status {status:?}")
        }
        RequestPoll::Ready(Completion::InvalidResponse { error, .. }) => {
            panic!("unexpected invalid response {error:?}")
        }
        RequestPoll::QueueError { error, .. } => panic!("unexpected queue error {error:?}"),
        RequestPoll::Stale(_) => panic!("unexpected stale request"),
        RequestPoll::NeedsReset(_) => panic!("unexpected reset requirement"),
    }
}

fn device_error<O: Operation>(
    client: &mut TestClient,
    pending: Pending<O>,
    expected: StatusCode,
) -> O {
    match client.poll(pending) {
        RequestPoll::Ready(Completion::DeviceError {
            status,
            disposition,
            operation,
            ..
        }) => {
            assert_eq!(status, expected);
            assert_eq!(disposition, FailureDisposition::Retryable);
            operation
        }
        _ => panic!("request did not produce the expected device error"),
    }
}

fn discover(client: &mut TestClient) {
    let pending = client
        .get_device_info(segmented_chain(
            size_of::<RequestHeader>(),
            size_of::<ResponseHeader>() + size_of::<WireDeviceInfo>(),
        ))
        .unwrap();
    let (info, _) = success(client, pending);
    assert_eq!(info.uuid, *b"virtio-accelmock");
}

struct Resources {
    context: Context,
    buffer: Buffer,
    program: Program,
    queue: ExecutionQueue,
}

fn create_resources(client: &mut TestClient) -> Resources {
    discover(client);
    let pending = client
        .create_context(chain(
            size_of::<RequestHeader>() + size_of::<CreateContextRequest>(),
            size_of::<ResponseHeader>() + size_of::<ObjectPayload>(),
        ))
        .unwrap();
    let (context, _) = success(client, pending);

    let desc = BufferDesc::new(
        64,
        16,
        MemoryDomain::Shared,
        BufferUsage::TRANSFER_SOURCE
            | BufferUsage::TRANSFER_DESTINATION
            | BufferUsage::PROGRAM_INPUT
            | BufferUsage::PROGRAM_OUTPUT
            | BufferUsage::MUTABLE_STATE,
    )
    .unwrap();
    let pending = client
        .allocate_buffer(
            chain(
                size_of::<RequestHeader>() + size_of::<AllocateBufferRequest>(),
                size_of::<ResponseHeader>() + size_of::<ObjectPayload>(),
            ),
            &context,
            desc,
        )
        .unwrap();
    let (buffer, _) = success(client, pending);

    let program_desc = ProgramDesc::new(
        NonZeroU32::new(1).unwrap(),
        core::array::from_fn(|index| index as u32),
        NonZeroU64::new(64).unwrap(),
    );
    let artifact = [0xa1, 0xb2, 0xc3, 0xd4];
    let pending = client
        .load_program(
            chain(
                size_of::<RequestHeader>()
                    + size_of::<LoadProgramRequest>()
                    + artifact.as_slice().len(),
                size_of::<ResponseHeader>() + size_of::<ObjectPayload>(),
            ),
            &context,
            program_desc,
            artifact.as_slice(),
        )
        .unwrap();
    let (program, _) = success(client, pending);

    let pending = client
        .create_execution_queue(
            chain(
                size_of::<RequestHeader>() + size_of::<CreateQueueRequest>(),
                size_of::<ResponseHeader>() + size_of::<ObjectPayload>(),
            ),
            &context,
        )
        .unwrap();
    let (queue, _) = success(client, pending);
    Resources {
        context,
        buffer,
        program,
        queue,
    }
}

fn submit(
    client: &mut TestClient,
    resources: &Resources,
    timeout_ns: u64,
) -> Pending<virtio_accel::guest::Submit> {
    let binding = [Binding {
        buffer: &resources.buffer,
        range: BufferRange::new(0, 16).unwrap(),
        slot: 0,
        access: AccessMode::ReadWrite,
    }];
    client
        .submit(
            chain(
                size_of::<RequestHeader>() + size_of::<SubmitRequest>() + size_of::<WireBinding>(),
                size_of::<ResponseHeader>() + size_of::<virtio_accel::proto::SubmitResponse>(),
            ),
            &resources.queue,
            &resources.program,
            &binding,
            timeout_ns,
        )
        .unwrap()
}

fn accepted_event(client: &mut TestClient, pending: Pending<virtio_accel::guest::Submit>) -> Event {
    let (outcome, _) = success(client, pending);
    let SubmissionOutcome::Accepted(event) = outcome else {
        panic!("mock submission unexpectedly became indeterminate")
    };
    event
}

fn operation_name(opcode: KnownOpcode) -> &'static str {
    match opcode {
        KnownOpcode::GetDeviceInfo => "GET_DEVICE_INFO",
        KnownOpcode::CreateContext => "CREATE_CONTEXT",
        KnownOpcode::DestroyContext => "DESTROY_CONTEXT",
        KnownOpcode::AllocateBuffer => "ALLOCATE_BUFFER",
        KnownOpcode::FreeBuffer => "FREE_BUFFER",
        KnownOpcode::WriteBuffer => "WRITE_BUFFER",
        KnownOpcode::ReadBuffer => "READ_BUFFER",
        KnownOpcode::LoadProgram => "LOAD_PROGRAM",
        KnownOpcode::UnloadProgram => "UNLOAD_PROGRAM",
        KnownOpcode::CreateQueue => "CREATE_QUEUE",
        KnownOpcode::DestroyQueue => "DESTROY_QUEUE",
        KnownOpcode::Submit => "SUBMIT",
        KnownOpcode::PollEvent => "POLL_EVENT",
        KnownOpcode::CancelEvent => "CANCEL_EVENT",
        KnownOpcode::DestroyEvent => "DESTROY_EVENT",
    }
}

fn expected_trace(name: &str) -> Vec<(u64, u16, u16, u32)> {
    let corpus: serde_json::Value =
        serde_json::from_str(include_str!("../conformance/v1.0/scenarios.json")).unwrap();
    let scenario = corpus["scenarios"]
        .as_array()
        .unwrap()
        .iter()
        .find(|scenario| scenario["name"] == name)
        .unwrap_or_else(|| panic!("missing scenario {name}"));
    scenario["trace"]
        .as_array()
        .unwrap()
        .iter()
        .map(|entry| {
            let opcode = u16::try_from(entry["opcode"].as_u64().unwrap()).unwrap();
            let known = KnownOpcode::try_from(opcode).expect("scenario opcode is assigned");
            assert_eq!(entry["operation"], operation_name(known));
            (
                entry["request_id"].as_u64().unwrap(),
                opcode,
                u16::try_from(entry["status"].as_u64().unwrap()).unwrap(),
                u32::try_from(entry["used"].as_u64().unwrap()).unwrap(),
            )
        })
        .collect()
}

fn assert_trace(name: &str, control: &HarnessControl) {
    let actual: Vec<_> = control
        .trace()
        .iter()
        .map(|entry| (entry.request_id, entry.opcode, entry.status, entry.used))
        .collect();
    assert_eq!(actual, expected_trace(name));
}

#[test]
fn complete_lifecycle_crosses_every_portable_boundary() {
    let (mut client, control, backend) = standard_stack();
    discover(&mut client);

    let pending = client
        .create_context(segmented_chain(
            size_of::<RequestHeader>() + size_of::<CreateContextRequest>(),
            size_of::<ResponseHeader>() + size_of::<ObjectPayload>(),
        ))
        .unwrap();
    let (context, _) = success(&mut client, pending);

    let desc = BufferDesc::new(
        64,
        16,
        MemoryDomain::Shared,
        BufferUsage::TRANSFER_SOURCE
            | BufferUsage::TRANSFER_DESTINATION
            | BufferUsage::PROGRAM_INPUT
            | BufferUsage::PROGRAM_OUTPUT
            | BufferUsage::MUTABLE_STATE,
    )
    .unwrap();
    let pending = client
        .allocate_buffer(
            segmented_chain(
                size_of::<RequestHeader>() + size_of::<AllocateBufferRequest>(),
                size_of::<ResponseHeader>() + size_of::<ObjectPayload>(),
            ),
            &context,
            desc,
        )
        .unwrap();
    let (buffer, _) = success(&mut client, pending);

    let payload = *b"portable";
    let range = BufferRange::new(8, payload.len()).unwrap();
    let pending = client
        .write_buffer_prepared(
            prepared_chain(
                size_of::<RequestHeader>() + size_of::<TransferBufferRequest>(),
                &payload,
                size_of::<ResponseHeader>(),
            ),
            &buffer,
            range,
        )
        .unwrap();
    success(&mut client, pending);

    let pending = client
        .read_buffer(
            segmented_chain(
                size_of::<RequestHeader>() + size_of::<TransferBufferRequest>(),
                size_of::<ResponseHeader>() + payload.as_slice().len(),
            ),
            &buffer,
            range,
        )
        .unwrap();
    let (read, read_chain) = success(&mut client, pending);
    assert_eq!(read.bytes, payload.len());
    let mut round_trip = [0_u8; 8];
    read_chain
        .read_device_writable(RESPONSE_HEADER_BYTES, &mut round_trip)
        .unwrap();
    assert_eq!(round_trip, payload);

    let program_desc = ProgramDesc::new(
        NonZeroU32::new(1).unwrap(),
        [0; 12],
        NonZeroU64::new(64).unwrap(),
    );
    let artifact = [0xa1, 0xb2, 0xc3, 0xd4];
    let pending = client
        .load_program_prepared(
            prepared_chain(
                size_of::<RequestHeader>() + size_of::<LoadProgramRequest>(),
                &artifact,
                size_of::<ResponseHeader>() + size_of::<ObjectPayload>(),
            ),
            &context,
            program_desc,
            artifact.len(),
        )
        .unwrap();
    let (program, _) = success(&mut client, pending);

    let pending = client
        .create_execution_queue(
            segmented_chain(
                size_of::<RequestHeader>() + size_of::<CreateQueueRequest>(),
                size_of::<ResponseHeader>() + size_of::<ObjectPayload>(),
            ),
            &context,
        )
        .unwrap();
    let (queue, _) = success(&mut client, pending);
    let mut resources = Resources {
        context,
        buffer,
        program,
        queue,
    };

    let pending = submit(&mut client, &resources, 0);
    let event = accepted_event(&mut client, pending);
    let pending = client
        .free_buffer(
            segmented_chain(
                size_of::<RequestHeader>() + size_of::<ObjectPayload>(),
                size_of::<ResponseHeader>(),
            ),
            resources.buffer,
        )
        .unwrap();
    resources.buffer = device_error(&mut client, pending, StatusCode::BUSY).into_buffer();

    let pending = client
        .poll_event(
            segmented_chain(
                size_of::<RequestHeader>() + size_of::<ObjectPayload>(),
                size_of::<ResponseHeader>() + size_of::<WireEventState>(),
            ),
            &event,
        )
        .unwrap();
    let (state, _) = success(&mut client, pending);
    assert_eq!(state, EventState::Pending);

    let pending = client
        .cancel_event(
            segmented_chain(
                size_of::<RequestHeader>() + size_of::<ObjectPayload>(),
                size_of::<ResponseHeader>(),
            ),
            &event,
        )
        .unwrap();
    success(&mut client, pending);
    let pending = client
        .poll_event(
            segmented_chain(
                size_of::<RequestHeader>() + size_of::<ObjectPayload>(),
                size_of::<ResponseHeader>() + size_of::<WireEventState>(),
            ),
            &event,
        )
        .unwrap();
    let (state, _) = success(&mut client, pending);
    assert_eq!(state, EventState::Cancelled);
    let pending = client
        .destroy_event(
            segmented_chain(
                size_of::<RequestHeader>() + size_of::<ObjectPayload>(),
                size_of::<ResponseHeader>(),
            ),
            event,
        )
        .unwrap();
    success(&mut client, pending);

    backend.auto_complete();
    let pending = submit(&mut client, &resources, 0);
    let event = accepted_event(&mut client, pending);
    let pending = client
        .poll_event(
            chain(
                size_of::<RequestHeader>() + size_of::<ObjectPayload>(),
                size_of::<ResponseHeader>() + size_of::<WireEventState>(),
            ),
            &event,
        )
        .unwrap();
    let (state, _) = success(&mut client, pending);
    assert_eq!(state, EventState::Complete);
    let pending = client
        .destroy_event(
            chain(
                size_of::<RequestHeader>() + size_of::<ObjectPayload>(),
                size_of::<ResponseHeader>(),
            ),
            event,
        )
        .unwrap();
    success(&mut client, pending);

    let pending = client
        .destroy_execution_queue(
            chain(
                size_of::<RequestHeader>() + size_of::<ObjectPayload>(),
                size_of::<ResponseHeader>(),
            ),
            resources.queue,
        )
        .unwrap();
    success(&mut client, pending);
    let pending = client
        .unload_program(
            chain(
                size_of::<RequestHeader>() + size_of::<ObjectPayload>(),
                size_of::<ResponseHeader>(),
            ),
            resources.program,
        )
        .unwrap();
    success(&mut client, pending);
    let pending = client
        .free_buffer(
            chain(
                size_of::<RequestHeader>() + size_of::<ObjectPayload>(),
                size_of::<ResponseHeader>(),
            ),
            resources.buffer,
        )
        .unwrap();
    success(&mut client, pending);
    let pending = client
        .destroy_context(
            chain(
                size_of::<RequestHeader>() + size_of::<ObjectPayload>(),
                size_of::<ResponseHeader>(),
            ),
            resources.context,
        )
        .unwrap();
    success(&mut client, pending);

    assert_trace("complete_lifecycle", &control);
    let opcodes: BTreeSet<_> = control.trace().iter().map(|entry| entry.opcode).collect();
    assert_eq!(opcodes.len(), 15, "the lifecycle must cross every opcode");
    assert!(
        control
            .trace()
            .iter()
            .any(|entry| { entry.readable_segments > 1 && entry.writable_segments > 1 })
    );
}

#[test]
fn multiple_inflight_requests_complete_out_of_order() {
    let (mut client, control, _) = standard_stack();
    control.reverse_next_pair();
    let first = client
        .get_device_info(chain(size_of::<RequestHeader>(), 92))
        .unwrap();
    let second = client
        .get_device_info(chain(size_of::<RequestHeader>(), 92))
        .unwrap();

    let first = match client.poll(first) {
        RequestPoll::Pending(pending) => pending,
        _ => panic!("first request completed before the reversed peer"),
    };
    success(&mut client, second);
    success(&mut client, first);
    assert_trace("out_of_order", &control);
}

#[test]
fn queue_pressure_returns_caller_ownership_before_draining() {
    let (mut client, control, _) = stack(4, 4, 4);
    control.defer();
    let first = client
        .get_device_info(chain(size_of::<RequestHeader>(), 92))
        .unwrap();
    let second = client
        .get_device_info(chain(size_of::<RequestHeader>(), 92))
        .unwrap();
    let error = client
        .get_device_info(chain(size_of::<RequestHeader>(), 92))
        .unwrap_err();
    let (returned, _, kind) = error.into_parts();
    assert_eq!(
        returned.device_readable_len(),
        size_of::<RequestHeader>() as u64
    );
    assert!(matches!(
        kind,
        StartErrorKind::Publish(virtio_accel::transport::PublishErrorKind::InsufficientDescriptors)
    ));

    control.service_immediately();
    success(&mut client, first);
    success(&mut client, second);
    assert_trace("backpressure", &control);
}

#[test]
fn notification_suppression_preserves_the_missed_work_recheck() {
    let (mut client, control, _) = standard_stack();
    control.suppress_available_notifications.set(true);
    control.service_on_notification_disable.set(true);
    let pending = client
        .get_device_info(segmented_chain(size_of::<RequestHeader>(), 92))
        .unwrap();
    client.disable_used_notifications().unwrap();
    assert_eq!(
        client.enable_used_notifications().unwrap(),
        NotificationRecheck::WorkPending
    );
    success(&mut client, pending);

    assert_eq!(
        *control.publication_hints.borrow(),
        [NotificationHint::Suppressed]
    );
    assert_eq!(
        *control.completion_hints.borrow(),
        [NotificationHint::Suppressed]
    );
    assert_trace("notification_suppression", &control);
}

#[test]
fn short_response_is_detected_before_typed_output_exists() {
    let (mut client, control, _) = standard_stack();
    control.shorten_next_response();
    let pending = client
        .get_device_info(chain(size_of::<RequestHeader>(), 92))
        .unwrap();
    match client.poll(pending) {
        RequestPoll::Ready(Completion::InvalidResponse {
            error: ResponseError::UsedLength { used: 91 },
            ..
        }) => {}
        _ => panic!("short completion was not rejected"),
    }
    assert_eq!(client.health(), ClientHealth::NeedsReset);
    assert_trace("short_response", &control);
}

#[test]
fn finite_timeout_rejection_retains_all_referenced_resources() {
    let (mut client, control, backend) = standard_stack();
    let resources = create_resources(&mut client);
    backend.reject_deadlines();
    let pending = submit(&mut client, &resources, 1_000_000);
    device_error(&mut client, pending, StatusCode::DEADLINE_EXPIRED);

    let pending = client
        .destroy_execution_queue(chain(24, 16), resources.queue)
        .unwrap();
    success(&mut client, pending);
    let pending = client
        .unload_program(chain(24, 16), resources.program)
        .unwrap();
    success(&mut client, pending);
    let pending = client.free_buffer(chain(24, 16), resources.buffer).unwrap();
    success(&mut client, pending);
    let pending = client
        .destroy_context(chain(24, 16), resources.context)
        .unwrap();
    success(&mut client, pending);
    assert_trace("timeout", &control);
}

#[test]
fn reset_reclaims_transport_work_and_reinitializes_both_peers() {
    let (mut client, control, _) = standard_stack();
    let resources = create_resources(&mut client);
    let pending = submit(&mut client, &resources, 0);
    let event = accepted_event(&mut client, pending);
    control.defer();
    let pending = client.poll_event(chain(24, 24), &event).unwrap();

    let next_epoch = client.queue_state().epoch().checked_next().unwrap();
    assert_eq!(client.reset(next_epoch).unwrap().count(), 1);
    assert!(matches!(client.poll(pending), RequestPoll::Stale(_)));
    let report = control.last_reset.get().unwrap();
    assert_eq!(report.disposition, ResetDisposition::BackendReusable);
    assert_eq!(report.released.contexts, 1);
    assert_eq!(report.released.buffers, 1);
    assert_eq!(report.released.programs, 1);
    assert_eq!(report.released.queues, 1);
    assert_eq!(report.released.events, 1);

    client
        .reconfigure_queue(QueueSize::new(STANDARD_QUEUE_SIZE).unwrap())
        .unwrap();
    discover(&mut client);
    assert_eq!(client.health(), ClientHealth::Running);
    assert_trace("reset_inflight", &control);
}

#[test]
fn device_loss_crosses_backend_engine_and_guest_recovery_boundaries() {
    let (mut client, control, backend) = standard_stack();
    let resources = create_resources(&mut client);
    let pending = submit(&mut client, &resources, 0);
    let event = accepted_event(&mut client, pending);
    backend.lose_device_on_poll();
    let pending = client.poll_event(chain(24, 24), &event).unwrap();
    match client.poll(pending) {
        RequestPoll::Ready(Completion::DeviceError {
            status: StatusCode::DEVICE_LOST,
            disposition: FailureDisposition::Indeterminate,
            ..
        }) => {}
        _ => panic!("device loss did not propagate as indeterminate ownership"),
    }
    assert_eq!(client.health(), ClientHealth::NeedsReset);
    assert_trace("device_loss", &control);
}
