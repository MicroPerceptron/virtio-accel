use std::cell::Cell;
use std::cell::RefCell;
use std::rc::Rc;
use std::vec::Vec;

use virtio_accel_core::{
    Accelerator, AccessMode, AllocatedBuffer, ArtifactRef, BackendError, BindingRef, BufferDesc,
    BufferInfo, BufferUsage, ByteSink, ByteSource, Capabilities, ContextDesc, DeviceInfo,
    DeviceInfoError, EventState, MemoryDomain, QueueDesc, ReleaseFailure, SubmitFailure, Timeout,
};
use virtio_accel_device::{
    ChainRegion, CommandOutcome, CommandProcessError, CommandProcessor, CommandProcessorInitError,
    DecoderLimitsError, DeviceHealth, DeviceStateError, ObjectId, ObjectNamespace,
    ResetDisposition, ResetError, ResourceCounts, ResourcePolicy, ResponseWriteError,
    RetainedBytes, SegmentedSink, SegmentedSource, UnusableFrame,
};
use virtio_accel_mock::fault::{
    FaultAccelerator, FaultAction, FaultPoint, FaultScript, FaultStep, ResourceState,
};
use virtio_accel_mock::{
    MockAccelerator, MockBuffer, MockContext, MockEvent, MockProgram, MockQueue, reference,
};
use virtio_accel_proto::{
    AllocateBufferRequest, BASELINE_COMMAND_QUEUES, CreateContextRequest, CreateQueueRequest,
    HARD_MAX_REQUEST_BYTES, HARD_MAX_RESPONSE_BYTES, KnownEventState, Le16, Le32, Le64,
    LoadProgramRequest, ObjectPayload, PROTOCOL_MAJOR, PROTOCOL_MINOR, RequestFlags, RequestHeader,
    ResponseHeader, StatusCode, SubmitRequest, SubmitResponse, TransferBufferRequest, WireBinding,
    WireConfig, WireDeviceInfo, WireEventState, read_exact,
};
use zerocopy::IntoBytes;

const REQUEST_ID: u64 = 0x0102_0304_0506_0708;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ReleaseMode {
    Pass,
    Reject(BackendError),
    Indeterminate,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SubmitMode {
    Accept,
    Reject,
    Indeterminate,
}

struct RecordingBackend {
    inner: MockAccelerator,
    info_override: Cell<Option<DeviceInfo>>,
    device_info_calls: Rc<Cell<u32>>,
    create_context_error: Cell<Option<BackendError>>,
    create_context_calls: Cell<u32>,
    context_release_mode: Cell<ReleaseMode>,
    free_mode: Cell<ReleaseMode>,
    free_calls: Cell<u32>,
    allocate_buffer_calls: Cell<u32>,
    allocation_bytes_override: Cell<Option<u64>>,
    program_release_mode: Cell<ReleaseMode>,
    queue_release_mode: Cell<ReleaseMode>,
    release_log: RefCell<Vec<&'static str>>,
    write_calls: Cell<u32>,
    read_calls: Cell<u32>,
    segmented_write: Cell<bool>,
    segmented_read: Cell<bool>,
    segmented_artifact: Cell<bool>,
    load_program_calls: Cell<u32>,
    submit_mode: Cell<SubmitMode>,
    submit_calls: Cell<u32>,
    last_timeout: Cell<Option<Timeout>>,
    poll_error: Cell<Option<BackendError>>,
    event_state_override: Cell<Option<EventState>>,
    poll_calls: Cell<u32>,
    cancel_calls: Cell<u32>,
    event_release_mode: Cell<ReleaseMode>,
    destroy_event_calls: Cell<u32>,
}

impl Default for RecordingBackend {
    fn default() -> Self {
        Self {
            inner: MockAccelerator::default(),
            info_override: Cell::new(None),
            device_info_calls: Rc::new(Cell::new(0)),
            create_context_error: Cell::new(None),
            create_context_calls: Cell::new(0),
            context_release_mode: Cell::new(ReleaseMode::Pass),
            free_mode: Cell::new(ReleaseMode::Pass),
            free_calls: Cell::new(0),
            allocate_buffer_calls: Cell::new(0),
            allocation_bytes_override: Cell::new(None),
            program_release_mode: Cell::new(ReleaseMode::Pass),
            queue_release_mode: Cell::new(ReleaseMode::Pass),
            release_log: RefCell::new(Vec::new()),
            write_calls: Cell::new(0),
            read_calls: Cell::new(0),
            segmented_write: Cell::new(false),
            segmented_read: Cell::new(false),
            segmented_artifact: Cell::new(false),
            load_program_calls: Cell::new(0),
            submit_mode: Cell::new(SubmitMode::Accept),
            submit_calls: Cell::new(0),
            last_timeout: Cell::new(None),
            poll_error: Cell::new(None),
            event_state_override: Cell::new(None),
            poll_calls: Cell::new(0),
            cancel_calls: Cell::new(0),
            event_release_mode: Cell::new(ReleaseMode::Pass),
            destroy_event_calls: Cell::new(0),
        }
    }
}

impl Accelerator for RecordingBackend {
    type Context = MockContext;
    type Buffer = MockBuffer;
    type Program = MockProgram;
    type Queue = MockQueue;
    type Event = MockEvent;

    fn device_info(&self) -> Result<DeviceInfo, BackendError> {
        self.device_info_calls.set(self.device_info_calls.get() + 1);
        match self.info_override.get() {
            Some(info) => Ok(info),
            None => self.inner.device_info(),
        }
    }

    fn create_context(&self, desc: ContextDesc) -> Result<Self::Context, BackendError> {
        self.create_context_calls
            .set(self.create_context_calls.get() + 1);
        if let Some(error) = self.create_context_error.take() {
            return Err(error);
        }
        self.inner.create_context(desc)
    }

    fn destroy_context(&self, context: Self::Context) -> Result<(), ReleaseFailure<Self::Context>> {
        self.release_log.borrow_mut().push("context");
        match self.context_release_mode.get() {
            ReleaseMode::Pass => self.inner.destroy_context(context),
            ReleaseMode::Reject(error) => Err(ReleaseFailure::Rejected {
                error,
                resource: context,
            }),
            ReleaseMode::Indeterminate => Err(ReleaseFailure::Indeterminate {
                error: BackendError::DeviceLost,
            }),
        }
    }

    fn allocate_buffer(
        &self,
        context: &Self::Context,
        desc: BufferDesc,
    ) -> Result<AllocatedBuffer<Self::Buffer>, BackendError> {
        self.allocate_buffer_calls
            .set(self.allocate_buffer_calls.get() + 1);
        let allocated = self.inner.allocate_buffer(context, desc)?;
        let Some(allocation_bytes) = self.allocation_bytes_override.get() else {
            return Ok(allocated);
        };
        let (buffer, info) = allocated.into_parts();
        let info = BufferInfo::new(
            info.desc(),
            allocation_bytes,
            info.alignment(),
            info.properties(),
        )?;
        Ok(AllocatedBuffer::new(buffer, info))
    }

    fn write_buffer(
        &self,
        buffer: &mut Self::Buffer,
        offset: u64,
        data: &dyn ByteSource,
    ) -> Result<(), BackendError> {
        self.write_calls.set(self.write_calls.get() + 1);
        self.segmented_write.set(data.as_contiguous().is_none());
        self.inner.write_buffer(buffer, offset, data)
    }

    fn read_buffer(
        &self,
        buffer: &Self::Buffer,
        offset: u64,
        data: &mut dyn ByteSink,
    ) -> Result<(), BackendError> {
        self.read_calls.set(self.read_calls.get() + 1);
        self.segmented_read.set(data.as_contiguous_mut().is_none());
        self.inner.read_buffer(buffer, offset, data)
    }

    fn free_buffer(&self, buffer: Self::Buffer) -> Result<(), ReleaseFailure<Self::Buffer>> {
        self.free_calls.set(self.free_calls.get() + 1);
        self.release_log.borrow_mut().push("buffer");
        match self.free_mode.get() {
            ReleaseMode::Pass => self.inner.free_buffer(buffer),
            ReleaseMode::Reject(error) => Err(ReleaseFailure::Rejected {
                error,
                resource: buffer,
            }),
            ReleaseMode::Indeterminate => Err(ReleaseFailure::Indeterminate {
                error: BackendError::DeviceLost,
            }),
        }
    }

    fn load_program(
        &self,
        context: &Self::Context,
        artifact: ArtifactRef<'_>,
    ) -> Result<Self::Program, BackendError> {
        self.load_program_calls
            .set(self.load_program_calls.get() + 1);
        self.segmented_artifact
            .set(artifact.payload.as_contiguous().is_none());
        self.inner.load_program(context, artifact)
    }

    fn unload_program(&self, program: Self::Program) -> Result<(), ReleaseFailure<Self::Program>> {
        self.release_log.borrow_mut().push("program");
        match self.program_release_mode.get() {
            ReleaseMode::Pass => self.inner.unload_program(program),
            ReleaseMode::Reject(error) => Err(ReleaseFailure::Rejected {
                error,
                resource: program,
            }),
            ReleaseMode::Indeterminate => Err(ReleaseFailure::Indeterminate {
                error: BackendError::DeviceLost,
            }),
        }
    }

    fn create_queue(
        &self,
        context: &Self::Context,
        desc: QueueDesc,
    ) -> Result<Self::Queue, BackendError> {
        self.inner.create_queue(context, desc)
    }

    fn destroy_queue(&self, queue: Self::Queue) -> Result<(), ReleaseFailure<Self::Queue>> {
        self.release_log.borrow_mut().push("queue");
        match self.queue_release_mode.get() {
            ReleaseMode::Pass => self.inner.destroy_queue(queue),
            ReleaseMode::Reject(error) => Err(ReleaseFailure::Rejected {
                error,
                resource: queue,
            }),
            ReleaseMode::Indeterminate => Err(ReleaseFailure::Indeterminate {
                error: BackendError::DeviceLost,
            }),
        }
    }

    fn submit(
        &self,
        queue: &Self::Queue,
        program: &Self::Program,
        bindings: &[BindingRef<'_, Self::Buffer>],
        timeout: Timeout,
    ) -> Result<Self::Event, SubmitFailure<Self::Event>> {
        self.submit_calls.set(self.submit_calls.get() + 1);
        self.last_timeout.set(Some(timeout));
        match self.submit_mode.get() {
            SubmitMode::Accept => self.inner.submit(queue, program, bindings, timeout),
            SubmitMode::Reject => Err(SubmitFailure::Rejected(BackendError::Busy)),
            SubmitMode::Indeterminate => {
                match self.inner.submit(queue, program, bindings, timeout) {
                    Ok(event) => Err(SubmitFailure::Indeterminate {
                        error: BackendError::DeadlineExpired,
                        event,
                    }),
                    Err(error) => Err(error),
                }
            }
        }
    }

    fn poll_event(&self, event: &Self::Event) -> Result<EventState, BackendError> {
        self.poll_calls.set(self.poll_calls.get() + 1);
        if let Some(error) = self.poll_error.take() {
            return Err(error);
        }
        if let Some(state) = self.event_state_override.get() {
            return Ok(state);
        }
        self.inner.poll_event(event)
    }

    fn cancel_event(&self, event: &Self::Event) -> Result<(), BackendError> {
        self.cancel_calls.set(self.cancel_calls.get() + 1);
        self.release_log.borrow_mut().push("cancel_event");
        self.inner.cancel_event(event)
    }

    fn destroy_event(&self, event: Self::Event) -> Result<(), ReleaseFailure<Self::Event>> {
        self.destroy_event_calls
            .set(self.destroy_event_calls.get() + 1);
        self.release_log.borrow_mut().push("event");
        match self.event_release_mode.get() {
            ReleaseMode::Pass => match self.event_state_override.get() {
                Some(EventState::Complete | EventState::Failed(_) | EventState::Cancelled) => {
                    Ok(())
                }
                Some(EventState::Pending) | None => self.inner.destroy_event(event),
            },
            ReleaseMode::Reject(error) => Err(ReleaseFailure::Rejected {
                error,
                resource: event,
            }),
            ReleaseMode::Indeterminate => Err(ReleaseFailure::Indeterminate {
                error: BackendError::DeviceLost,
            }),
        }
    }
}

#[derive(Debug)]
struct FailingSink {
    len: u64,
}

impl ByteSink for FailingSink {
    fn len(&self) -> u64 {
        self.len
    }

    fn write_at(&mut self, _offset: u64, _source: &[u8]) -> Result<(), BackendError> {
        Err(BackendError::DeviceLost)
    }
}

fn config() -> WireConfig {
    WireConfig {
        protocol_major: Le16::new(PROTOCOL_MAJOR),
        protocol_minor: Le16::new(PROTOCOL_MINOR),
        command_queue_count: Le16::new(BASELINE_COMMAND_QUEUES),
        max_chain_descriptors: Le16::new(16),
        max_request_bytes: Le32::new(HARD_MAX_REQUEST_BYTES),
        max_response_bytes: Le32::new(HARD_MAX_RESPONSE_BYTES),
    }
}

fn resource_policy() -> ResourcePolicy {
    ResourcePolicy::new(1 << 30, 1 << 30).unwrap()
}

fn processor() -> CommandProcessor<RecordingBackend> {
    CommandProcessor::new(
        RecordingBackend::default(),
        &config(),
        ObjectNamespace::new(1).unwrap(),
        resource_policy(),
    )
    .unwrap()
}

fn request(opcode: virtio_accel_proto::KnownOpcode, payload: &[u8]) -> Vec<u8> {
    let header = RequestHeader::new(
        opcode,
        RequestFlags::empty(),
        payload.len() as u32,
        REQUEST_ID,
    );
    let mut bytes = Vec::from(header.as_bytes());
    bytes.extend_from_slice(payload);
    bytes
}

fn object_request(opcode: virtio_accel_proto::KnownOpcode, id: ObjectId) -> Vec<u8> {
    request(
        opcode,
        ObjectPayload {
            object_id: Le64::new(id.get()),
        }
        .as_bytes(),
    )
}

fn run<A: Accelerator>(
    processor: &mut CommandProcessor<A>,
    request: &[u8],
    response_capacity: usize,
) -> (CommandOutcome, Vec<u8>) {
    let request_split = request.len() / 2;
    let request_segments = [&request[..request_split], &request[request_split..]];
    let source = SegmentedSource::new(&request_segments).unwrap();

    let mut response = vec![0xaa_u8; response_capacity];
    let response_split = response_capacity / 2;
    let (left, right) = response.split_at_mut(response_split);
    let mut response_segments: [&mut [u8]; 2] = [left, right];
    let mut sink = SegmentedSink::new(&mut response_segments).unwrap();
    let regions = [
        ChainRegion::readable(request_split as u64),
        ChainRegion::readable((request.len() - request_split) as u64),
        ChainRegion::writable(response_split as u64),
        ChainRegion::writable((response_capacity - response_split) as u64),
    ];

    let outcome = processor.process(&regions, &source, &mut sink).unwrap();
    (outcome, response)
}

fn status(outcome: CommandOutcome) -> StatusCode {
    let CommandOutcome::Response { status, .. } = outcome else {
        panic!("command did not produce a response");
    };
    status
}

fn object_id(response: &[u8]) -> ObjectId {
    let payload = read_exact::<ObjectPayload>(&response[16..24]).unwrap();
    ObjectId::from_raw(payload.object_id.get()).unwrap()
}

fn create_context<A: Accelerator>(processor: &mut CommandProcessor<A>) -> ObjectId {
    let frame = request(
        virtio_accel_proto::KnownOpcode::CreateContext,
        CreateContextRequest {
            flags: Le32::new(0),
            reserved: Le32::new(0),
        }
        .as_bytes(),
    );
    let (outcome, response) = run(processor, &frame, 24);
    assert_eq!(status(outcome), StatusCode::OK);
    object_id(&response)
}

fn allocate_buffer<A: Accelerator>(
    processor: &mut CommandProcessor<A>,
    context_id: ObjectId,
) -> ObjectId {
    allocate_buffer_with_usage(
        processor,
        context_id,
        BufferUsage::TRANSFER_SOURCE | BufferUsage::TRANSFER_DESTINATION,
    )
}

fn allocate_buffer_with_usage<A: Accelerator>(
    processor: &mut CommandProcessor<A>,
    context_id: ObjectId,
    usage: BufferUsage,
) -> ObjectId {
    let frame = request(
        virtio_accel_proto::KnownOpcode::AllocateBuffer,
        AllocateBufferRequest {
            context_id: Le64::new(context_id.get()),
            bytes: Le64::new(8),
            alignment: Le64::new(1),
            memory_domain: MemoryDomain::Host as u8,
            reserved0: [0; 7],
            usage: Le32::new(usage.bits()),
            reserved1: Le32::new(0),
        }
        .as_bytes(),
    );
    let (outcome, response) = run(processor, &frame, 24);
    assert_eq!(status(outcome), StatusCode::OK);
    object_id(&response)
}

fn load_program<A: Accelerator>(
    processor: &mut CommandProcessor<A>,
    context_id: ObjectId,
) -> ObjectId {
    let artifact = reference::ReferenceArtifact::barrier(0);
    let load = LoadProgramRequest {
        context_id: Le64::new(context_id.get()),
        format: Le32::new(reference::ARTIFACT_FORMAT.get()),
        flags: Le32::new(0),
        target: reference::TARGET_IDENTITY.0.map(Le32::new),
        payload_bytes: Le64::new(reference::ARTIFACT_BYTES as u64),
        resident_bytes: Le64::new(reference::RESIDENT_BYTES),
    };
    let mut payload = Vec::from(load.as_bytes());
    payload.extend_from_slice(artifact.as_bytes());
    let (outcome, response) = run(
        processor,
        &request(virtio_accel_proto::KnownOpcode::LoadProgram, &payload),
        24,
    );
    assert_eq!(status(outcome), StatusCode::OK);
    object_id(&response)
}

fn create_queue<A: Accelerator>(
    processor: &mut CommandProcessor<A>,
    context_id: ObjectId,
) -> ObjectId {
    let frame = request(
        virtio_accel_proto::KnownOpcode::CreateQueue,
        CreateQueueRequest {
            context_id: Le64::new(context_id.get()),
            flags: Le32::new(0),
            reserved: Le32::new(0),
        }
        .as_bytes(),
    );
    let (outcome, response) = run(processor, &frame, 24);
    assert_eq!(status(outcome), StatusCode::OK);
    object_id(&response)
}

fn submission_objects<A: Accelerator>(
    processor: &mut CommandProcessor<A>,
    usage: BufferUsage,
) -> (ObjectId, ObjectId, ObjectId, ObjectId) {
    let context = create_context(processor);
    let buffer = allocate_buffer_with_usage(processor, context, usage);
    let program = load_program(processor, context);
    let queue = create_queue(processor, context);
    (context, buffer, program, queue)
}

fn submit_request(
    queue_id: ObjectId,
    program_id: ObjectId,
    buffer_id: ObjectId,
    offset: u64,
    bytes: u64,
    access: AccessMode,
) -> Vec<u8> {
    submit_request_with_timeout(queue_id, program_id, buffer_id, offset, bytes, access, 0)
}

fn submit_request_with_timeout(
    queue_id: ObjectId,
    program_id: ObjectId,
    buffer_id: ObjectId,
    offset: u64,
    bytes: u64,
    access: AccessMode,
    timeout_ns: u64,
) -> Vec<u8> {
    let submit = SubmitRequest {
        queue_id: Le64::new(queue_id.get()),
        program_id: Le64::new(program_id.get()),
        binding_count: Le32::new(1),
        flags: Le32::new(0),
        timeout_ns: Le64::new(timeout_ns),
    };
    let binding = WireBinding {
        buffer_id: Le64::new(buffer_id.get()),
        offset: Le64::new(offset),
        bytes: Le64::new(bytes),
        slot: Le32::new(0),
        access: access as u8,
        reserved: [0; 3],
    };
    let mut payload = Vec::from(submit.as_bytes());
    payload.extend_from_slice(binding.as_bytes());
    request(virtio_accel_proto::KnownOpcode::Submit, &payload)
}

fn event_id(response: &[u8]) -> ObjectId {
    let payload = read_exact::<SubmitResponse>(&response[16..24]).unwrap();
    ObjectId::from_raw(payload.event_id.get()).unwrap()
}

fn event_state(response: &[u8]) -> WireEventState {
    read_exact::<WireEventState>(&response[16..24]).unwrap()
}

#[test]
fn invalid_config_is_rejected_before_backend_discovery() {
    let backend = RecordingBackend::default();
    let device_info_calls = Rc::clone(&backend.device_info_calls);
    let mut invalid = config();
    invalid.protocol_major = Le16::new(PROTOCOL_MAJOR + 1);

    assert!(matches!(
        CommandProcessor::new(
            backend,
            &invalid,
            ObjectNamespace::new(1).unwrap(),
            resource_policy(),
        ),
        Err(CommandProcessorInitError::Decoder(
            DecoderLimitsError::Config(virtio_accel_proto::ConfigError::Version)
        ))
    ));
    assert_eq!(device_info_calls.get(), 0);
}

#[test]
fn deterministic_fault_script_preserves_command_engine_ownership() {
    let discovery_script = FaultScript::new([FaultStep::new(
        FaultPoint::DeviceInfo,
        1,
        FaultAction::ErrorBefore(BackendError::Busy),
    )])
    .unwrap();
    let backend = FaultAccelerator::new(MockAccelerator::default(), discovery_script);
    let discovery_control = backend.control();
    assert_eq!(
        CommandProcessor::new(
            backend,
            &config(),
            ObjectNamespace::new(1).unwrap(),
            resource_policy(),
        )
        .unwrap_err(),
        CommandProcessorInitError::Backend(BackendError::Busy)
    );
    assert!(discovery_control.snapshot().is_clean());

    let script = FaultScript::new([
        FaultStep::new(
            FaultPoint::CreateContext,
            1,
            FaultAction::ErrorBefore(BackendError::OutOfMemory),
        ),
        FaultStep::new(
            FaultPoint::AllocateBuffer,
            1,
            FaultAction::ErrorAfter(BackendError::OutOfMemory),
        ),
        FaultStep::new(
            FaultPoint::WriteBuffer,
            1,
            FaultAction::ErrorAfter(BackendError::Busy),
        ),
        FaultStep::new(
            FaultPoint::ReadBuffer,
            1,
            FaultAction::ErrorAfter(BackendError::Busy),
        ),
        FaultStep::new(
            FaultPoint::LoadProgram,
            1,
            FaultAction::ErrorAfter(BackendError::OutOfMemory),
        ),
        FaultStep::new(
            FaultPoint::CreateQueue,
            1,
            FaultAction::ErrorAfter(BackendError::ResourceLimit),
        ),
        FaultStep::new(
            FaultPoint::Submit,
            1,
            FaultAction::Rejected(BackendError::Busy),
        ),
        FaultStep::new(
            FaultPoint::PollEvent,
            1,
            FaultAction::ErrorBefore(BackendError::Busy),
        ),
        FaultStep::new(
            FaultPoint::CancelEvent,
            1,
            FaultAction::ErrorBefore(BackendError::Busy),
        ),
        FaultStep::new(
            FaultPoint::DestroyEvent,
            1,
            FaultAction::Rejected(BackendError::Busy),
        ),
        FaultStep::new(
            FaultPoint::DestroyQueue,
            1,
            FaultAction::Rejected(BackendError::Busy),
        ),
        FaultStep::new(
            FaultPoint::UnloadProgram,
            1,
            FaultAction::Rejected(BackendError::Busy),
        ),
        FaultStep::new(
            FaultPoint::FreeBuffer,
            1,
            FaultAction::Rejected(BackendError::Busy),
        ),
        FaultStep::new(
            FaultPoint::DestroyContext,
            1,
            FaultAction::Indeterminate(BackendError::DeviceLost),
        ),
    ])
    .unwrap();
    let backend = FaultAccelerator::new(MockAccelerator::default(), script);
    let control = backend.control();
    let mut processor = CommandProcessor::new(
        backend,
        &config(),
        ObjectNamespace::new(1).unwrap(),
        resource_policy(),
    )
    .unwrap();

    let create = request(
        virtio_accel_proto::KnownOpcode::CreateContext,
        CreateContextRequest {
            flags: Le32::new(0),
            reserved: Le32::new(0),
        }
        .as_bytes(),
    );
    assert_eq!(
        status(run(&mut processor, &create, 24).0),
        StatusCode::OUT_OF_MEMORY
    );
    assert_eq!(
        processor.state().resource_counts(),
        ResourceCounts::default()
    );
    let context = create_context(&mut processor);

    let usage = BufferUsage::TRANSFER_SOURCE
        | BufferUsage::TRANSFER_DESTINATION
        | BufferUsage::PROGRAM_INPUT;
    let allocate = request(
        virtio_accel_proto::KnownOpcode::AllocateBuffer,
        AllocateBufferRequest {
            context_id: Le64::new(context.get()),
            bytes: Le64::new(8),
            alignment: Le64::new(1),
            memory_domain: MemoryDomain::Host as u8,
            reserved0: [0; 7],
            usage: Le32::new(usage.bits()),
            reserved1: Le32::new(0),
        }
        .as_bytes(),
    );
    assert_eq!(
        status(run(&mut processor, &allocate, 24).0),
        StatusCode::OUT_OF_MEMORY
    );
    assert_eq!(
        processor.state().resource_counts(),
        ResourceCounts {
            contexts: 1,
            ..ResourceCounts::default()
        }
    );
    let buffer = allocate_buffer_with_usage(&mut processor, context, usage);

    let transfer = TransferBufferRequest {
        buffer_id: Le64::new(buffer.get()),
        offset: Le64::new(0),
        bytes: Le64::new(8),
    };
    let mut write_payload = Vec::from(transfer.as_bytes());
    write_payload.extend_from_slice(b"scripted");
    let write = request(virtio_accel_proto::KnownOpcode::WriteBuffer, &write_payload);
    assert_eq!(status(run(&mut processor, &write, 16).0), StatusCode::BUSY);
    assert_eq!(status(run(&mut processor, &write, 16).0), StatusCode::OK);

    let read = request(
        virtio_accel_proto::KnownOpcode::ReadBuffer,
        transfer.as_bytes(),
    );
    let (outcome, _) = run(&mut processor, &read, 24);
    assert!(matches!(
        outcome,
        CommandOutcome::Response {
            status: StatusCode::BUSY,
            used: 16,
            ..
        }
    ));
    let (outcome, response) = run(&mut processor, &read, 24);
    assert_eq!(status(outcome), StatusCode::OK);
    assert_eq!(&response[16..24], b"scripted");

    let artifact = reference::ReferenceArtifact::barrier(0);
    let load = LoadProgramRequest {
        context_id: Le64::new(context.get()),
        format: Le32::new(reference::ARTIFACT_FORMAT.get()),
        flags: Le32::new(0),
        target: reference::TARGET_IDENTITY.0.map(Le32::new),
        payload_bytes: Le64::new(reference::ARTIFACT_BYTES as u64),
        resident_bytes: Le64::new(reference::RESIDENT_BYTES),
    };
    let mut load_payload = Vec::from(load.as_bytes());
    load_payload.extend_from_slice(artifact.as_bytes());
    let load = request(virtio_accel_proto::KnownOpcode::LoadProgram, &load_payload);
    assert_eq!(
        status(run(&mut processor, &load, 24).0),
        StatusCode::OUT_OF_MEMORY
    );
    assert_eq!(processor.state().resource_counts().programs, 0);
    let program = load_program(&mut processor, context);

    let create_queue_frame = request(
        virtio_accel_proto::KnownOpcode::CreateQueue,
        CreateQueueRequest {
            context_id: Le64::new(context.get()),
            flags: Le32::new(0),
            reserved: Le32::new(0),
        }
        .as_bytes(),
    );
    assert_eq!(
        status(run(&mut processor, &create_queue_frame, 24).0),
        StatusCode::RESOURCE_LIMIT
    );
    assert_eq!(processor.state().resource_counts().queues, 0);
    let queue = create_queue(&mut processor, context);

    let submit = submit_request(queue, program, buffer, 0, 8, AccessMode::Read);
    assert_eq!(status(run(&mut processor, &submit, 24).0), StatusCode::BUSY);
    assert_eq!(processor.state().event_count(), 0);
    assert_eq!(
        processor.state().buffer_record(buffer).unwrap().in_flight(),
        0
    );
    let (outcome, response) = run(&mut processor, &submit, 24);
    assert_eq!(status(outcome), StatusCode::OK);
    let event = event_id(&response);

    let poll = object_request(virtio_accel_proto::KnownOpcode::PollEvent, event);
    assert_eq!(status(run(&mut processor, &poll, 24).0), StatusCode::BUSY);
    assert_eq!(processor.state().event_count(), 1);
    let cancel = object_request(virtio_accel_proto::KnownOpcode::CancelEvent, event);
    assert_eq!(status(run(&mut processor, &cancel, 16).0), StatusCode::BUSY);
    assert_eq!(status(run(&mut processor, &cancel, 16).0), StatusCode::OK);

    let destroy_event = object_request(virtio_accel_proto::KnownOpcode::DestroyEvent, event);
    assert_eq!(
        status(run(&mut processor, &destroy_event, 16).0),
        StatusCode::BUSY
    );
    assert_eq!(processor.state().event_count(), 1);
    assert_eq!(
        status(run(&mut processor, &destroy_event, 16).0),
        StatusCode::OK
    );
    assert_eq!(processor.state().event_count(), 0);

    for (opcode, id) in [
        (virtio_accel_proto::KnownOpcode::DestroyQueue, queue),
        (virtio_accel_proto::KnownOpcode::UnloadProgram, program),
        (virtio_accel_proto::KnownOpcode::FreeBuffer, buffer),
    ] {
        let frame = object_request(opcode, id);
        assert_eq!(status(run(&mut processor, &frame, 16).0), StatusCode::BUSY);
        assert_eq!(status(run(&mut processor, &frame, 16).0), StatusCode::OK);
    }
    assert_eq!(
        processor.state().resource_counts(),
        ResourceCounts {
            contexts: 1,
            ..ResourceCounts::default()
        }
    );

    let destroy_context = object_request(virtio_accel_proto::KnownOpcode::DestroyContext, context);
    assert_eq!(
        status(run(&mut processor, &destroy_context, 16).0),
        StatusCode::DEVICE_LOST
    );
    assert_eq!(
        processor.state().resource_counts(),
        ResourceCounts::default()
    );
    assert_eq!(processor.health(), DeviceHealth::BackendDiscardRequired);

    let snapshot = control.snapshot();
    assert!(snapshot.pending_faults.is_empty());
    assert!(snapshot.violations.is_empty());
    assert_eq!(
        snapshot.resources_in(ResourceState::Indeterminate).contexts,
        1
    );
    control.discard_all();
    assert!(control.snapshot().is_clean());
}

#[test]
fn invalid_provider_metadata_is_rejected_before_object_state_exists() {
    for (capabilities, expected) in [
        (
            Capabilities::HOST_VISIBLE_MEMORY | Capabilities::EXTERNAL_MEMORY,
            DeviceInfoError::ReservedCapabilities,
        ),
        (
            Capabilities::EVENT_CANCELLATION,
            DeviceInfoError::MissingMemoryDomain,
        ),
    ] {
        let backend = RecordingBackend::default();
        let mut info = backend.device_info().unwrap();
        backend.device_info_calls.set(0);
        info.capabilities = capabilities;
        backend.info_override.set(Some(info));
        let device_info_calls = Rc::clone(&backend.device_info_calls);

        assert!(matches!(
            CommandProcessor::new(
                backend,
                &config(),
                ObjectNamespace::new(1).unwrap(),
                resource_policy(),
            ),
            Err(CommandProcessorInitError::DeviceInfo(error)) if error == expected
        ));
        assert_eq!(device_info_calls.get(), 1);
    }

    let backend = RecordingBackend::default();
    let mut info = backend.device_info().unwrap();
    backend.device_info_calls.set(0);
    info.limits.max_events_per_context = 0;
    backend.info_override.set(Some(info));
    assert_eq!(
        CommandProcessor::new(
            backend,
            &config(),
            ObjectNamespace::new(1).unwrap(),
            resource_policy(),
        )
        .unwrap_err(),
        CommandProcessorInitError::DeviceInfo(DeviceInfoError::ZeroLimit)
    );
}

#[test]
fn baseline_lifecycle_preserves_segmented_zero_copy_ports() {
    let mut processor = processor();

    let get_info = request(virtio_accel_proto::KnownOpcode::GetDeviceInfo, &[]);
    let (outcome, response) = run(
        &mut processor,
        &get_info,
        16 + core::mem::size_of::<WireDeviceInfo>(),
    );
    assert_eq!(status(outcome), StatusCode::OK);
    let info = read_exact::<WireDeviceInfo>(&response[16..]).unwrap();
    assert_eq!(info.max_contexts.get(), 64);

    let context = create_context(&mut processor);
    let buffer = allocate_buffer(&mut processor, context);

    let transfer = TransferBufferRequest {
        buffer_id: Le64::new(buffer.get()),
        offset: Le64::new(0),
        bytes: Le64::new(8),
    };
    let mut write_payload = Vec::from(transfer.as_bytes());
    write_payload.extend_from_slice(b"abcdefgh");
    let write = request(virtio_accel_proto::KnownOpcode::WriteBuffer, &write_payload);
    assert_eq!(status(run(&mut processor, &write, 16).0), StatusCode::OK);

    let read = request(
        virtio_accel_proto::KnownOpcode::ReadBuffer,
        transfer.as_bytes(),
    );
    let (outcome, response) = run(&mut processor, &read, 24);
    assert_eq!(status(outcome), StatusCode::OK);
    assert_eq!(&response[16..24], b"abcdefgh");

    let artifact = reference::ReferenceArtifact::barrier(0);
    let load = LoadProgramRequest {
        context_id: Le64::new(context.get()),
        format: Le32::new(reference::ARTIFACT_FORMAT.get()),
        flags: Le32::new(0),
        target: reference::TARGET_IDENTITY.0.map(Le32::new),
        payload_bytes: Le64::new(reference::ARTIFACT_BYTES as u64),
        resident_bytes: Le64::new(reference::RESIDENT_BYTES),
    };
    let mut load_payload = Vec::from(load.as_bytes());
    load_payload.extend_from_slice(artifact.as_bytes());
    let load = request(virtio_accel_proto::KnownOpcode::LoadProgram, &load_payload);
    let (outcome, response) = run(&mut processor, &load, 24);
    assert_eq!(status(outcome), StatusCode::OK);
    let program = object_id(&response);

    let create_queue = request(
        virtio_accel_proto::KnownOpcode::CreateQueue,
        CreateQueueRequest {
            context_id: Le64::new(context.get()),
            flags: Le32::new(0),
            reserved: Le32::new(0),
        }
        .as_bytes(),
    );
    let (outcome, response) = run(&mut processor, &create_queue, 24);
    assert_eq!(status(outcome), StatusCode::OK);
    let queue = object_id(&response);

    assert_eq!(
        status(
            run(
                &mut processor,
                &object_request(virtio_accel_proto::KnownOpcode::DestroyContext, context),
                16,
            )
            .0,
        ),
        StatusCode::BUSY
    );

    for (opcode, id) in [
        (virtio_accel_proto::KnownOpcode::DestroyQueue, queue),
        (virtio_accel_proto::KnownOpcode::UnloadProgram, program),
        (virtio_accel_proto::KnownOpcode::FreeBuffer, buffer),
        (virtio_accel_proto::KnownOpcode::DestroyContext, context),
    ] {
        assert_eq!(
            status(run(&mut processor, &object_request(opcode, id), 16).0),
            StatusCode::OK
        );
    }

    assert!(processor.accelerator().segmented_write.get());
    assert!(processor.accelerator().segmented_read.get());
    assert!(processor.accelerator().segmented_artifact.get());
    assert_eq!(processor.state().context_count(), 0);
}

#[test]
fn malformed_reference_artifacts_do_not_create_resident_programs() {
    let mut processor = processor();
    let context = create_context(&mut processor);
    let mut artifact = *reference::ReferenceArtifact::barrier(0).as_bytes();
    artifact[17] = 1;
    let load = LoadProgramRequest {
        context_id: Le64::new(context.get()),
        format: Le32::new(reference::ARTIFACT_FORMAT.get()),
        flags: Le32::new(0),
        target: reference::TARGET_IDENTITY.0.map(Le32::new),
        payload_bytes: Le64::new(reference::ARTIFACT_BYTES as u64),
        resident_bytes: Le64::new(reference::RESIDENT_BYTES),
    };
    let mut payload = Vec::from(load.as_bytes());
    payload.extend_from_slice(&artifact);

    assert_eq!(
        status(
            run(
                &mut processor,
                &request(virtio_accel_proto::KnownOpcode::LoadProgram, &payload),
                24,
            )
            .0,
        ),
        StatusCode::INVALID_ARGUMENT
    );
    assert_eq!(
        processor.state().resource_counts(),
        ResourceCounts {
            contexts: 1,
            ..ResourceCounts::default()
        }
    );

    load_program(&mut processor, context);
    assert_eq!(
        processor.state().resource_counts(),
        ResourceCounts {
            contexts: 1,
            programs: 1,
            ..ResourceCounts::default()
        }
    );
}

#[test]
fn semantic_validation_prevents_backend_calls() {
    let backend = RecordingBackend::default();
    let mut info = backend.device_info().unwrap();
    backend.device_info_calls.set(0);
    info.limits.max_contexts = 1;
    backend.info_override.set(Some(info));
    let mut processor = CommandProcessor::new(
        backend,
        &config(),
        ObjectNamespace::new(1).unwrap(),
        resource_policy(),
    )
    .unwrap();

    let context = create_context(&mut processor);
    let create = request(
        virtio_accel_proto::KnownOpcode::CreateContext,
        CreateContextRequest {
            flags: Le32::new(0),
            reserved: Le32::new(0),
        }
        .as_bytes(),
    );
    assert_eq!(
        status(run(&mut processor, &create, 24).0),
        StatusCode::RESOURCE_LIMIT
    );
    assert_eq!(processor.accelerator().create_context_calls.get(), 1);

    let buffer = allocate_buffer_with_usage(&mut processor, context, BufferUsage::TRANSFER_SOURCE);
    let transfer = TransferBufferRequest {
        buffer_id: Le64::new(buffer.get()),
        offset: Le64::new(0),
        bytes: Le64::new(8),
    };
    let mut write_payload = Vec::from(transfer.as_bytes());
    write_payload.extend_from_slice(b"abcdefgh");
    let write = request(virtio_accel_proto::KnownOpcode::WriteBuffer, &write_payload);
    assert_eq!(
        status(run(&mut processor, &write, 16).0),
        StatusCode::PERMISSION_DENIED
    );
    assert_eq!(processor.accelerator().write_calls.get(), 0);

    let out_of_bounds = TransferBufferRequest {
        buffer_id: Le64::new(buffer.get()),
        offset: Le64::new(1),
        bytes: Le64::new(8),
    };
    let read = request(
        virtio_accel_proto::KnownOpcode::ReadBuffer,
        out_of_bounds.as_bytes(),
    );
    assert_eq!(
        status(run(&mut processor, &read, 24).0),
        StatusCode::OUT_OF_BOUNDS
    );
    assert_eq!(processor.accelerator().read_calls.get(), 0);

    let wrong_kind = object_request(virtio_accel_proto::KnownOpcode::FreeBuffer, context);
    assert_eq!(
        status(run(&mut processor, &wrong_kind, 16).0),
        StatusCode::STALE_OBJECT
    );
    assert_eq!(processor.accelerator().free_calls.get(), 0);

    let read = request(
        virtio_accel_proto::KnownOpcode::ReadBuffer,
        transfer.as_bytes(),
    );
    let (outcome, response) = run(&mut processor, &read, 16);
    assert!(matches!(
        outcome,
        CommandOutcome::Unusable(UnusableFrame::InsufficientResponse { .. })
    ));
    assert_eq!(response, [0xaa; 16]);
    assert_eq!(processor.accelerator().read_calls.get(), 0);

    assert_eq!(
        status(
            run(
                &mut processor,
                &object_request(virtio_accel_proto::KnownOpcode::FreeBuffer, buffer),
                16,
            )
            .0,
        ),
        StatusCode::OK
    );
    assert_eq!(processor.accelerator().free_calls.get(), 1);
    assert_eq!(
        status(run(&mut processor, &read, 24).0),
        StatusCode::STALE_OBJECT
    );
    assert_eq!(processor.accelerator().read_calls.get(), 0);
}

#[test]
fn unsupported_memory_domain_is_rejected_before_backend_allocation() {
    let backend = RecordingBackend::default();
    let mut info = backend.device_info().unwrap();
    info.capabilities.remove(Capabilities::DEVICE_LOCAL_MEMORY);
    backend.info_override.set(Some(info));
    let mut processor = CommandProcessor::new(
        backend,
        &config(),
        ObjectNamespace::new(1).unwrap(),
        resource_policy(),
    )
    .unwrap();
    let context = create_context(&mut processor);
    let allocate = request(
        virtio_accel_proto::KnownOpcode::AllocateBuffer,
        AllocateBufferRequest {
            context_id: Le64::new(context.get()),
            bytes: Le64::new(8),
            alignment: Le64::new(1),
            memory_domain: MemoryDomain::Device as u8,
            reserved0: [0; 7],
            usage: Le32::new(BufferUsage::TRANSFER_SOURCE.bits()),
            reserved1: Le32::new(0),
        }
        .as_bytes(),
    );

    assert_eq!(
        status(run(&mut processor, &allocate, 24).0),
        StatusCode::UNSUPPORTED
    );
    assert_eq!(processor.accelerator().allocate_buffer_calls.get(), 0);
    assert!(processor.retained_bytes().is_empty());
}

#[test]
fn aggregate_retained_byte_policy_cleans_up_before_exposing_an_id() {
    let backend = RecordingBackend::default();
    backend.allocation_bytes_override.set(Some(16));
    let mut processor = CommandProcessor::new(
        backend,
        &config(),
        ObjectNamespace::new(1).unwrap(),
        ResourcePolicy::new(8, 1 << 20).unwrap(),
    )
    .unwrap();
    let context = create_context(&mut processor);
    let allocate = request(
        virtio_accel_proto::KnownOpcode::AllocateBuffer,
        AllocateBufferRequest {
            context_id: Le64::new(context.get()),
            bytes: Le64::new(8),
            alignment: Le64::new(1),
            memory_domain: MemoryDomain::Host as u8,
            reserved0: [0; 7],
            usage: Le32::new(BufferUsage::TRANSFER_SOURCE.bits()),
            reserved1: Le32::new(0),
        }
        .as_bytes(),
    );

    assert_eq!(
        status(run(&mut processor, &allocate, 24).0),
        StatusCode::RESOURCE_LIMIT
    );
    assert_eq!(processor.accelerator().free_calls.get(), 1);
    assert_eq!(
        processor.state().resource_counts(),
        ResourceCounts {
            contexts: 1,
            ..ResourceCounts::default()
        }
    );
    assert!(processor.retained_bytes().is_empty());
    assert_eq!(processor.health(), DeviceHealth::Running);
}

#[test]
fn rejected_cleanup_of_an_unpublished_allocation_requires_backend_discard() {
    let backend = RecordingBackend::default();
    backend.allocation_bytes_override.set(Some(16));
    backend
        .free_mode
        .set(ReleaseMode::Reject(BackendError::Busy));
    let mut processor = CommandProcessor::new(
        backend,
        &config(),
        ObjectNamespace::new(1).unwrap(),
        ResourcePolicy::new(8, 1 << 20).unwrap(),
    )
    .unwrap();
    let context = create_context(&mut processor);
    let allocate = request(
        virtio_accel_proto::KnownOpcode::AllocateBuffer,
        AllocateBufferRequest {
            context_id: Le64::new(context.get()),
            bytes: Le64::new(8),
            alignment: Le64::new(1),
            memory_domain: MemoryDomain::Host as u8,
            reserved0: [0; 7],
            usage: Le32::new(BufferUsage::TRANSFER_SOURCE.bits()),
            reserved1: Le32::new(0),
        }
        .as_bytes(),
    );

    assert_eq!(
        status(run(&mut processor, &allocate, 24).0),
        StatusCode::DEVICE_LOST
    );
    assert_eq!(processor.health(), DeviceHealth::BackendDiscardRequired);
    assert_eq!(
        processor.retained_bytes(),
        RetainedBytes {
            buffer_backing: 16,
            program_resident: 0,
        }
    );
    let report = processor.reset(ObjectNamespace::new(2).unwrap()).unwrap();
    assert_eq!(report.disposition, ResetDisposition::BackendDiscardRequired);
    assert_eq!(report.quarantined.contexts, 1);
    assert_eq!(report.quarantined.buffers, 1);
    assert_eq!(report.quarantined_bytes.buffer_backing, 16);
    assert!(report.released_bytes.is_empty());
}

#[test]
fn indeterminate_cleanup_of_an_unpublished_allocation_quarantines_actual_backing() {
    let backend = RecordingBackend::default();
    backend.allocation_bytes_override.set(Some(16));
    backend.free_mode.set(ReleaseMode::Indeterminate);
    let mut processor = CommandProcessor::new(
        backend,
        &config(),
        ObjectNamespace::new(1).unwrap(),
        ResourcePolicy::new(8, 1 << 20).unwrap(),
    )
    .unwrap();
    let context = create_context(&mut processor);
    let allocate = request(
        virtio_accel_proto::KnownOpcode::AllocateBuffer,
        AllocateBufferRequest {
            context_id: Le64::new(context.get()),
            bytes: Le64::new(8),
            alignment: Le64::new(1),
            memory_domain: MemoryDomain::Host as u8,
            reserved0: [0; 7],
            usage: Le32::new(BufferUsage::TRANSFER_SOURCE.bits()),
            reserved1: Le32::new(0),
        }
        .as_bytes(),
    );

    assert_eq!(
        status(run(&mut processor, &allocate, 24).0),
        StatusCode::DEVICE_LOST
    );
    assert_eq!(processor.health(), DeviceHealth::BackendDiscardRequired);
    assert_eq!(
        processor.state().resource_counts(),
        ResourceCounts {
            contexts: 1,
            ..ResourceCounts::default()
        }
    );
    assert_eq!(
        processor.retained_bytes(),
        RetainedBytes {
            buffer_backing: 16,
            program_resident: 0,
        }
    );
    let report = processor.reset(ObjectNamespace::new(2).unwrap()).unwrap();
    assert_eq!(report.disposition, ResetDisposition::BackendDiscardRequired);
    assert_eq!(report.quarantined.contexts, 1);
    assert_eq!(report.quarantined.buffers, 1);
    assert_eq!(report.quarantined_bytes.buffer_backing, 16);
    assert!(report.released_bytes.is_empty());
}

#[test]
fn program_resident_policy_rejects_before_provider_invocation() {
    let backend = RecordingBackend::default();
    let mut processor = CommandProcessor::new(
        backend,
        &config(),
        ObjectNamespace::new(1).unwrap(),
        ResourcePolicy::new(1 << 20, reference::RESIDENT_BYTES - 1).unwrap(),
    )
    .unwrap();
    let context = create_context(&mut processor);
    let artifact = reference::ReferenceArtifact::barrier(0);
    let load = LoadProgramRequest {
        context_id: Le64::new(context.get()),
        format: Le32::new(reference::ARTIFACT_FORMAT.get()),
        flags: Le32::new(0),
        target: reference::TARGET_IDENTITY.0.map(Le32::new),
        payload_bytes: Le64::new(reference::ARTIFACT_BYTES as u64),
        resident_bytes: Le64::new(reference::RESIDENT_BYTES),
    };
    let mut payload = Vec::from(load.as_bytes());
    payload.extend_from_slice(artifact.as_bytes());

    assert_eq!(
        status(
            run(
                &mut processor,
                &request(virtio_accel_proto::KnownOpcode::LoadProgram, &payload),
                24,
            )
            .0
        ),
        StatusCode::RESOURCE_LIMIT
    );
    assert_eq!(processor.accelerator().load_program_calls.get(), 0);
    assert!(processor.retained_bytes().is_empty());
}

#[test]
fn short_responses_and_backend_rejection_never_publish_state() {
    let mut processor = processor();
    let create = request(
        virtio_accel_proto::KnownOpcode::CreateContext,
        CreateContextRequest {
            flags: Le32::new(0),
            reserved: Le32::new(0),
        }
        .as_bytes(),
    );

    let (outcome, response) = run(&mut processor, &create, 16);
    assert!(matches!(
        outcome,
        CommandOutcome::Unusable(UnusableFrame::InsufficientResponse { .. })
    ));
    assert_eq!(response, [0xaa; 16]);
    assert_eq!(processor.accelerator().create_context_calls.get(), 0);
    assert_eq!(processor.state().context_count(), 0);

    processor
        .accelerator()
        .create_context_error
        .set(Some(BackendError::OutOfMemory));
    let (outcome, response) = run(&mut processor, &create, 24);
    assert_eq!(status(outcome), StatusCode::OUT_OF_MEMORY);
    assert_eq!(
        read_exact::<ResponseHeader>(&response[..16])
            .unwrap()
            .payload_bytes
            .get(),
        0
    );
    assert_eq!(processor.state().context_count(), 0);
}

#[test]
fn rejected_release_restores_the_id_and_indeterminate_release_requires_reset() {
    let mut processor = processor();
    let context = create_context(&mut processor);
    let buffer = allocate_buffer(&mut processor, context);
    let free = object_request(virtio_accel_proto::KnownOpcode::FreeBuffer, buffer);

    processor
        .accelerator()
        .free_mode
        .set(ReleaseMode::Reject(BackendError::Busy));
    assert_eq!(status(run(&mut processor, &free, 16).0), StatusCode::BUSY);
    assert!(processor.state().buffer_record(buffer).is_ok());

    processor.accelerator().free_mode.set(ReleaseMode::Pass);
    assert_eq!(status(run(&mut processor, &free, 16).0), StatusCode::OK);
    assert_eq!(
        processor.state().buffer_record(buffer).unwrap_err(),
        DeviceStateError::StaleObject
    );

    let second = allocate_buffer(&mut processor, context);
    processor
        .accelerator()
        .free_mode
        .set(ReleaseMode::Indeterminate);
    assert_eq!(
        status(
            run(
                &mut processor,
                &object_request(virtio_accel_proto::KnownOpcode::FreeBuffer, second),
                16,
            )
            .0,
        ),
        StatusCode::DEVICE_LOST
    );
    assert_eq!(processor.health(), DeviceHealth::BackendDiscardRequired);
    assert_eq!(
        processor.state().buffer_record(second).unwrap_err(),
        DeviceStateError::StaleObject
    );

    let get_info = request(virtio_accel_proto::KnownOpcode::GetDeviceInfo, &[]);
    assert_eq!(
        status(
            run(
                &mut processor,
                &get_info,
                16 + core::mem::size_of::<WireDeviceInfo>(),
            )
            .0,
        ),
        StatusCode::DEVICE_LOST
    );
    let report = processor.reset(ObjectNamespace::new(2).unwrap()).unwrap();
    assert_eq!(report.disposition, ResetDisposition::BackendDiscardRequired);
    assert_eq!(
        report.quarantined,
        ResourceCounts {
            contexts: 1,
            buffers: 1,
            ..ResourceCounts::default()
        }
    );
}

#[test]
fn response_failure_after_creation_requires_reset() {
    let mut processor = processor();
    let create = request(
        virtio_accel_proto::KnownOpcode::CreateContext,
        CreateContextRequest {
            flags: Le32::new(0),
            reserved: Le32::new(0),
        }
        .as_bytes(),
    );
    let request_segments = [create.as_slice()];
    let source = SegmentedSource::new(&request_segments).unwrap();
    let regions = [
        ChainRegion::readable(create.len() as u64),
        ChainRegion::writable(24),
    ];
    let mut sink = FailingSink { len: 24 };

    assert_eq!(
        processor.process(&regions, &source, &mut sink),
        Err(CommandProcessError::ResponseWrite(
            ResponseWriteError::SinkAccess
        ))
    );
    assert_eq!(processor.state().context_count(), 1);
    assert_eq!(processor.health(), DeviceHealth::NeedsReset);

    let create_calls = processor.accelerator().create_context_calls.get();
    assert_eq!(
        status(run(&mut processor, &create, 24).0),
        StatusCode::DEVICE_LOST
    );
    assert_eq!(
        processor.accelerator().create_context_calls.get(),
        create_calls
    );
    assert_eq!(processor.state().context_count(), 1);

    let report = processor.reset(ObjectNamespace::new(2).unwrap()).unwrap();
    assert_eq!(report.disposition, ResetDisposition::BackendReusable);
    assert_eq!(
        report.released,
        ResourceCounts {
            contexts: 1,
            ..ResourceCounts::default()
        }
    );
    assert!(report.quarantined.is_empty());
    assert!(report.released_bytes.is_empty());
    assert!(report.quarantined_bytes.is_empty());
    assert_eq!(processor.health(), DeviceHealth::Running);
    assert!(processor.state().is_empty());
}

#[test]
fn accepted_events_retain_resources_and_resolve_cancel_completion_races() {
    let mut processor = processor();
    let (_context, buffer, program, queue) =
        submission_objects(&mut processor, BufferUsage::PROGRAM_INPUT);
    let submit =
        submit_request_with_timeout(queue, program, buffer, 0, 8, AccessMode::Read, 1_000_000);

    let (outcome, response) = run(&mut processor, &submit, 24);
    assert_eq!(status(outcome), StatusCode::OK);
    assert_eq!(
        processor.accelerator().last_timeout.get(),
        Some(Timeout::from_wire_ns(1_000_000))
    );
    let cancelled_event = event_id(&response);
    assert_eq!(processor.state().event_count(), 1);
    assert_eq!(
        processor.state().buffer_record(buffer).unwrap().in_flight(),
        1
    );
    assert_eq!(
        processor
            .state()
            .program_record(program)
            .unwrap()
            .in_flight(),
        1
    );
    assert_eq!(
        processor.state().queue_record(queue).unwrap().in_flight(),
        1
    );

    let free = object_request(virtio_accel_proto::KnownOpcode::FreeBuffer, buffer);
    assert_eq!(status(run(&mut processor, &free, 16).0), StatusCode::BUSY);

    let poll = object_request(virtio_accel_proto::KnownOpcode::PollEvent, cancelled_event);
    let (outcome, response) = run(&mut processor, &poll, 24);
    assert_eq!(status(outcome), StatusCode::OK);
    assert_eq!(
        event_state(&response).known_state().unwrap(),
        KnownEventState::Pending
    );

    let destroy = object_request(
        virtio_accel_proto::KnownOpcode::DestroyEvent,
        cancelled_event,
    );
    assert_eq!(
        status(run(&mut processor, &destroy, 16).0),
        StatusCode::BUSY
    );
    assert_eq!(processor.accelerator().destroy_event_calls.get(), 0);

    let cancel = object_request(
        virtio_accel_proto::KnownOpcode::CancelEvent,
        cancelled_event,
    );
    assert_eq!(status(run(&mut processor, &cancel, 16).0), StatusCode::OK);
    let (outcome, response) = run(&mut processor, &poll, 24);
    assert_eq!(status(outcome), StatusCode::OK);
    assert_eq!(
        event_state(&response).known_state().unwrap(),
        KnownEventState::Cancelled
    );
    assert_eq!(status(run(&mut processor, &destroy, 16).0), StatusCode::OK);
    assert_eq!(processor.state().event_count(), 0);
    assert_eq!(
        processor.state().buffer_record(buffer).unwrap().in_flight(),
        0
    );
    assert_eq!(
        processor
            .state()
            .program_record(program)
            .unwrap()
            .in_flight(),
        0
    );
    assert_eq!(
        processor.state().queue_record(queue).unwrap().in_flight(),
        0
    );

    let (outcome, response) = run(&mut processor, &submit, 24);
    assert_eq!(status(outcome), StatusCode::OK);
    let pending_event = event_id(&response);
    let (outcome, response) = run(&mut processor, &submit, 24);
    assert_eq!(status(outcome), StatusCode::OK);
    let completed_event = event_id(&response);
    assert_ne!(pending_event, completed_event);
    assert_eq!(
        processor.state().buffer_record(buffer).unwrap().in_flight(),
        2
    );
    let event = processor
        .state()
        .event_record(completed_event)
        .unwrap()
        .resource()
        .unwrap();
    processor.accelerator().inner.complete(event).unwrap();

    let poll = object_request(virtio_accel_proto::KnownOpcode::PollEvent, pending_event);
    let (outcome, response) = run(&mut processor, &poll, 24);
    assert_eq!(status(outcome), StatusCode::OK);
    assert_eq!(
        event_state(&response).known_state().unwrap(),
        KnownEventState::Pending
    );

    let cancel = object_request(
        virtio_accel_proto::KnownOpcode::CancelEvent,
        completed_event,
    );
    assert_eq!(status(run(&mut processor, &cancel, 16).0), StatusCode::BUSY);
    let poll = object_request(virtio_accel_proto::KnownOpcode::PollEvent, completed_event);
    let (outcome, response) = run(&mut processor, &poll, 24);
    assert_eq!(status(outcome), StatusCode::OK);
    assert_eq!(
        event_state(&response).known_state().unwrap(),
        KnownEventState::Complete
    );
    let destroy = object_request(
        virtio_accel_proto::KnownOpcode::DestroyEvent,
        completed_event,
    );
    assert_eq!(status(run(&mut processor, &destroy, 16).0), StatusCode::OK);
    assert_eq!(
        processor.state().buffer_record(buffer).unwrap().in_flight(),
        1
    );

    let cancel = object_request(virtio_accel_proto::KnownOpcode::CancelEvent, pending_event);
    assert_eq!(status(run(&mut processor, &cancel, 16).0), StatusCode::OK);
    let destroy = object_request(virtio_accel_proto::KnownOpcode::DestroyEvent, pending_event);
    assert_eq!(status(run(&mut processor, &destroy, 16).0), StatusCode::OK);
    assert_eq!(
        processor.state().buffer_record(buffer).unwrap().in_flight(),
        0
    );
}

#[test]
fn rejected_and_indeterminate_submissions_preserve_the_admission_boundary() {
    let mut processor = processor();
    let (_context, buffer, program, queue) =
        submission_objects(&mut processor, BufferUsage::PROGRAM_INPUT);
    let submit = submit_request(queue, program, buffer, 0, 8, AccessMode::Read);

    processor.accelerator().submit_mode.set(SubmitMode::Reject);
    let (outcome, response) = run(&mut processor, &submit, 24);
    assert_eq!(status(outcome), StatusCode::BUSY);
    assert_eq!(
        read_exact::<ResponseHeader>(&response[..16])
            .unwrap()
            .payload_bytes
            .get(),
        0
    );
    assert_eq!(processor.state().event_count(), 0);
    assert_eq!(
        processor.state().buffer_record(buffer).unwrap().in_flight(),
        0
    );
    assert_eq!(
        processor
            .state()
            .program_record(program)
            .unwrap()
            .in_flight(),
        0
    );
    assert_eq!(
        processor.state().queue_record(queue).unwrap().in_flight(),
        0
    );

    processor
        .accelerator()
        .submit_mode
        .set(SubmitMode::Indeterminate);
    let (outcome, response) = run(&mut processor, &submit, 24);
    assert_eq!(status(outcome), StatusCode::DEADLINE_EXPIRED);
    let event_id = event_id(&response);
    assert_eq!(processor.health(), DeviceHealth::Running);
    assert_eq!(processor.state().event_count(), 1);
    assert_eq!(
        processor.state().buffer_record(buffer).unwrap().in_flight(),
        1
    );
    assert_eq!(
        processor
            .state()
            .program_record(program)
            .unwrap()
            .in_flight(),
        1
    );
    assert_eq!(
        processor.state().queue_record(queue).unwrap().in_flight(),
        1
    );

    let event = processor
        .state()
        .event_record(event_id)
        .unwrap()
        .resource()
        .unwrap();
    processor.accelerator().inner.complete(event).unwrap();
    let destroy = object_request(virtio_accel_proto::KnownOpcode::DestroyEvent, event_id);
    assert_eq!(status(run(&mut processor, &destroy, 16).0), StatusCode::OK);
    assert_eq!(
        processor.state().buffer_record(buffer).unwrap().in_flight(),
        0
    );
    assert_eq!(
        processor
            .state()
            .program_record(program)
            .unwrap()
            .in_flight(),
        0
    );
    assert_eq!(
        processor.state().queue_record(queue).unwrap().in_flight(),
        0
    );
}

#[test]
fn submission_validation_rejects_before_backend_admission() {
    let mut processor = processor();
    let (_context, buffer, program, queue) =
        submission_objects(&mut processor, BufferUsage::PROGRAM_OUTPUT);

    let wrong_access = submit_request(queue, program, buffer, 0, 8, AccessMode::Read);
    assert_eq!(
        status(run(&mut processor, &wrong_access, 24).0),
        StatusCode::PERMISSION_DENIED
    );

    let out_of_bounds = submit_request(queue, program, buffer, 1, 8, AccessMode::Write);
    assert_eq!(
        status(run(&mut processor, &out_of_bounds, 24).0),
        StatusCode::OUT_OF_BOUNDS
    );

    let other_context = create_context(&mut processor);
    let other_buffer =
        allocate_buffer_with_usage(&mut processor, other_context, BufferUsage::PROGRAM_INPUT);
    let wrong_context = submit_request(queue, program, other_buffer, 0, 8, AccessMode::Read);
    assert_eq!(
        status(run(&mut processor, &wrong_context, 24).0),
        StatusCode::STALE_OBJECT
    );
    assert_eq!(processor.accelerator().submit_calls.get(), 0);
    assert_eq!(processor.state().event_count(), 0);
    assert_eq!(
        processor.state().buffer_record(buffer).unwrap().in_flight(),
        0
    );
    assert_eq!(
        processor
            .state()
            .program_record(program)
            .unwrap()
            .in_flight(),
        0
    );
    assert_eq!(
        processor.state().queue_record(queue).unwrap().in_flight(),
        0
    );
    assert_eq!(
        processor
            .state()
            .buffer_record(other_buffer)
            .unwrap()
            .in_flight(),
        0
    );
}

#[test]
fn event_release_rejection_retries_and_indeterminate_release_requires_reset() {
    let mut processor = processor();
    let (_context, buffer, program, queue) =
        submission_objects(&mut processor, BufferUsage::PROGRAM_INPUT);
    let submit = submit_request(queue, program, buffer, 0, 8, AccessMode::Read);

    let (_, response) = run(&mut processor, &submit, 24);
    let first_event = event_id(&response);
    let event = processor
        .state()
        .event_record(first_event)
        .unwrap()
        .resource()
        .unwrap();
    processor.accelerator().inner.complete(event).unwrap();
    let destroy = object_request(virtio_accel_proto::KnownOpcode::DestroyEvent, first_event);

    processor
        .accelerator()
        .event_release_mode
        .set(ReleaseMode::Reject(BackendError::Busy));
    assert_eq!(
        status(run(&mut processor, &destroy, 16).0),
        StatusCode::BUSY
    );
    assert!(processor.state().event_record(first_event).is_ok());
    assert_eq!(
        processor.state().buffer_record(buffer).unwrap().in_flight(),
        1
    );

    processor
        .accelerator()
        .event_release_mode
        .set(ReleaseMode::Pass);
    assert_eq!(status(run(&mut processor, &destroy, 16).0), StatusCode::OK);
    assert_eq!(
        processor.state().buffer_record(buffer).unwrap().in_flight(),
        0
    );

    let (_, response) = run(&mut processor, &submit, 24);
    let second_event = event_id(&response);
    let event = processor
        .state()
        .event_record(second_event)
        .unwrap()
        .resource()
        .unwrap();
    processor.accelerator().inner.complete(event).unwrap();
    processor
        .accelerator()
        .event_release_mode
        .set(ReleaseMode::Indeterminate);
    let destroy = object_request(virtio_accel_proto::KnownOpcode::DestroyEvent, second_event);
    assert_eq!(
        status(run(&mut processor, &destroy, 16).0),
        StatusCode::DEVICE_LOST
    );
    assert_eq!(processor.health(), DeviceHealth::BackendDiscardRequired);
    assert_eq!(
        processor.state().event_record(second_event).unwrap_err(),
        DeviceStateError::StaleObject
    );
    assert_eq!(
        processor.state().buffer_record(buffer).unwrap().in_flight(),
        0
    );
}

#[test]
fn event_faults_and_unreportable_admission_require_recovery() {
    let mut poll_processor = processor();
    let (_context, buffer, program, queue) =
        submission_objects(&mut poll_processor, BufferUsage::PROGRAM_INPUT);
    let submit = submit_request(queue, program, buffer, 0, 8, AccessMode::Read);
    let (_, response) = run(&mut poll_processor, &submit, 24);
    let event = event_id(&response);
    poll_processor
        .accelerator()
        .poll_error
        .set(Some(BackendError::DeviceLost));
    let poll = object_request(virtio_accel_proto::KnownOpcode::PollEvent, event);
    assert_eq!(
        status(run(&mut poll_processor, &poll, 24).0),
        StatusCode::DEVICE_LOST
    );
    assert_eq!(
        poll_processor.health(),
        DeviceHealth::BackendDiscardRequired
    );
    assert_eq!(poll_processor.state().event_count(), 1);

    let mut terminal_processor = processor();
    let (_context, buffer, program, queue) =
        submission_objects(&mut terminal_processor, BufferUsage::PROGRAM_INPUT);
    let submit = submit_request(queue, program, buffer, 0, 8, AccessMode::Read);
    let (_, response) = run(&mut terminal_processor, &submit, 24);
    let event = event_id(&response);
    terminal_processor
        .accelerator()
        .event_state_override
        .set(Some(EventState::Failed(BackendError::DeviceLost)));
    let poll = object_request(virtio_accel_proto::KnownOpcode::PollEvent, event);
    let (outcome, response) = run(&mut terminal_processor, &poll, 24);
    assert_eq!(status(outcome), StatusCode::OK);
    let state = event_state(&response);
    assert_eq!(state.known_state().unwrap(), KnownEventState::Failed);
    assert_eq!(state.error.get(), StatusCode::DEVICE_LOST.0);
    assert_eq!(
        terminal_processor.health(),
        DeviceHealth::BackendDiscardRequired
    );
    let poll_calls = terminal_processor.accelerator().poll_calls.get();
    let report = terminal_processor
        .reset(ObjectNamespace::new(2).unwrap())
        .unwrap();
    assert_eq!(report.disposition, ResetDisposition::BackendDiscardRequired);
    assert_eq!(
        report.quarantined,
        ResourceCounts {
            contexts: 1,
            buffers: 1,
            programs: 1,
            queues: 1,
            events: 1,
        }
    );
    assert_eq!(
        terminal_processor.accelerator().poll_calls.get(),
        poll_calls
    );
    assert!(
        terminal_processor
            .accelerator()
            .release_log
            .borrow()
            .is_empty()
    );

    let mut completion_processor = processor();
    let (_context, buffer, program, queue) =
        submission_objects(&mut completion_processor, BufferUsage::PROGRAM_INPUT);
    let submit = submit_request(queue, program, buffer, 0, 8, AccessMode::Read);
    let (_, response) = run(&mut completion_processor, &submit, 24);
    let event = event_id(&response);
    completion_processor
        .accelerator()
        .event_state_override
        .set(Some(EventState::Failed(BackendError::DeadlineExpired)));
    let poll = object_request(virtio_accel_proto::KnownOpcode::PollEvent, event);
    for _ in 0..2 {
        let (outcome, response) = run(&mut completion_processor, &poll, 24);
        assert_eq!(status(outcome), StatusCode::OK);
        let state = event_state(&response);
        assert_eq!(state.known_state().unwrap(), KnownEventState::Failed);
        assert_eq!(state.error.get(), StatusCode::DEADLINE_EXPIRED.0);
    }
    let destroy = object_request(virtio_accel_proto::KnownOpcode::DestroyEvent, event);
    assert_eq!(
        status(run(&mut completion_processor, &destroy, 16).0),
        StatusCode::OK
    );
    assert_eq!(
        completion_processor
            .state()
            .buffer_record(buffer)
            .unwrap()
            .in_flight(),
        0
    );

    let mut response_processor = processor();
    let (_context, buffer, program, queue) =
        submission_objects(&mut response_processor, BufferUsage::PROGRAM_INPUT);
    response_processor
        .accelerator()
        .submit_mode
        .set(SubmitMode::Indeterminate);
    let submit = submit_request(queue, program, buffer, 0, 8, AccessMode::Read);
    let request_segments = [submit.as_slice()];
    let source = SegmentedSource::new(&request_segments).unwrap();
    let regions = [
        ChainRegion::readable(submit.len() as u64),
        ChainRegion::writable(24),
    ];
    let mut sink = FailingSink { len: 24 };

    assert_eq!(
        response_processor.process(&regions, &source, &mut sink),
        Err(CommandProcessError::ResponseWrite(
            ResponseWriteError::SinkAccess
        ))
    );
    assert_eq!(response_processor.state().event_count(), 1);
    assert_eq!(
        response_processor
            .state()
            .buffer_record(buffer)
            .unwrap()
            .in_flight(),
        1
    );
    assert_eq!(response_processor.health(), DeviceHealth::NeedsReset);
}

#[test]
fn cancellation_is_not_forwarded_without_the_capability() {
    let backend = RecordingBackend::default();
    let mut info = backend.device_info().unwrap();
    backend.device_info_calls.set(0);
    info.capabilities.remove(Capabilities::EVENT_CANCELLATION);
    backend.info_override.set(Some(info));
    let mut processor = CommandProcessor::new(
        backend,
        &config(),
        ObjectNamespace::new(1).unwrap(),
        resource_policy(),
    )
    .unwrap();
    let (_context, buffer, program, queue) =
        submission_objects(&mut processor, BufferUsage::PROGRAM_INPUT);
    let submit = submit_request(queue, program, buffer, 0, 8, AccessMode::Read);
    let (_, response) = run(&mut processor, &submit, 24);
    let event = event_id(&response);
    let cancel = object_request(virtio_accel_proto::KnownOpcode::CancelEvent, event);

    assert_eq!(
        status(run(&mut processor, &cancel, 16).0),
        StatusCode::UNSUPPORTED
    );
    assert_eq!(processor.accelerator().cancel_calls.get(), 0);
    let poll = object_request(virtio_accel_proto::KnownOpcode::PollEvent, event);
    let (outcome, response) = run(&mut processor, &poll, 24);
    assert_eq!(status(outcome), StatusCode::OK);
    assert_eq!(
        event_state(&response).known_state().unwrap(),
        KnownEventState::Pending
    );
}

#[test]
fn idle_reset_renews_the_namespace_without_backend_work() {
    let mut processor = processor();

    let first = processor.reset(ObjectNamespace::new(2).unwrap()).unwrap();
    assert_eq!(first.disposition, ResetDisposition::BackendReusable);
    assert!(first.released.is_empty());
    assert!(first.quarantined.is_empty());
    assert!(processor.accelerator().release_log.borrow().is_empty());

    assert_eq!(
        processor.reset(ObjectNamespace::new(2).unwrap()),
        Err(ResetError::NamespaceReuse)
    );
    let second = processor.reset(ObjectNamespace::new(3).unwrap()).unwrap();
    assert_eq!(second.disposition, ResetDisposition::BackendReusable);
    assert!(second.released.is_empty());
    assert!(processor.accelerator().release_log.borrow().is_empty());
}

#[test]
fn discard_report_supersedes_a_prior_successful_reset() {
    let mut processor = processor();
    let first = processor.reset(ObjectNamespace::new(2).unwrap()).unwrap();
    assert_eq!(first.disposition, ResetDisposition::BackendReusable);

    let context = create_context(&mut processor);
    let buffer = allocate_buffer(&mut processor, context);
    processor
        .accelerator()
        .free_mode
        .set(ReleaseMode::Indeterminate);
    let free = object_request(virtio_accel_proto::KnownOpcode::FreeBuffer, buffer);
    assert_eq!(
        status(run(&mut processor, &free, 16).0),
        StatusCode::DEVICE_LOST
    );

    let report = processor.reset(ObjectNamespace::new(3).unwrap()).unwrap();
    assert_eq!(report.disposition, ResetDisposition::BackendDiscardRequired);
    assert_eq!(
        report.quarantined,
        ResourceCounts {
            contexts: 1,
            buffers: 1,
            ..ResourceCounts::default()
        }
    );
}

#[test]
fn reset_cancels_pending_events_then_tears_down_child_before_parent() {
    let mut processor = processor();
    let (context, buffer, program, queue) =
        submission_objects(&mut processor, BufferUsage::PROGRAM_INPUT);
    let submit = submit_request(queue, program, buffer, 0, 8, AccessMode::Read);
    let (_, response) = run(&mut processor, &submit, 24);
    let event = event_id(&response);

    let report = processor.reset(ObjectNamespace::new(2).unwrap()).unwrap();
    assert_eq!(report.disposition, ResetDisposition::BackendReusable);
    assert_eq!(
        report.released,
        ResourceCounts {
            contexts: 1,
            buffers: 1,
            programs: 1,
            queues: 1,
            events: 1,
        }
    );
    assert!(report.quarantined.is_empty());
    assert_eq!(
        report.released_bytes,
        RetainedBytes {
            buffer_backing: 8,
            program_resident: u128::from(reference::RESIDENT_BYTES),
        }
    );
    assert!(report.quarantined_bytes.is_empty());
    assert_eq!(
        processor.accelerator().release_log.borrow().as_slice(),
        [
            "cancel_event",
            "event",
            "queue",
            "program",
            "buffer",
            "context"
        ]
    );
    assert_eq!(processor.health(), DeviceHealth::Running);
    assert!(processor.state().is_empty());

    for (opcode, id) in [
        (virtio_accel_proto::KnownOpcode::DestroyContext, context),
        (virtio_accel_proto::KnownOpcode::FreeBuffer, buffer),
        (virtio_accel_proto::KnownOpcode::UnloadProgram, program),
        (virtio_accel_proto::KnownOpcode::DestroyQueue, queue),
        (virtio_accel_proto::KnownOpcode::DestroyEvent, event),
    ] {
        assert_eq!(
            status(run(&mut processor, &object_request(opcode, id), 16).0),
            StatusCode::STALE_OBJECT
        );
    }

    let new_context = create_context(&mut processor);
    assert_ne!(new_context, context);
}

#[test]
fn reset_reclaims_completed_and_cancelled_events_exactly_once() {
    let mut processor = processor();
    let (_context, buffer, program, queue) =
        submission_objects(&mut processor, BufferUsage::PROGRAM_INPUT);
    let submit = submit_request(queue, program, buffer, 0, 8, AccessMode::Read);
    let (_, response) = run(&mut processor, &submit, 24);
    let completed = event_id(&response);
    let (_, response) = run(&mut processor, &submit, 24);
    let cancelled = event_id(&response);

    let event = processor
        .state()
        .event_record(completed)
        .unwrap()
        .resource()
        .unwrap();
    processor.accelerator().inner.complete(event).unwrap();
    let cancel = object_request(virtio_accel_proto::KnownOpcode::CancelEvent, cancelled);
    assert_eq!(status(run(&mut processor, &cancel, 16).0), StatusCode::OK);
    processor.accelerator().release_log.borrow_mut().clear();

    let report = processor.reset(ObjectNamespace::new(2).unwrap()).unwrap();
    assert_eq!(report.disposition, ResetDisposition::BackendReusable);
    assert_eq!(report.released.events, 2);
    assert_eq!(report.released.total(), 6);
    assert!(report.quarantined.is_empty());
    assert_eq!(
        processor.accelerator().release_log.borrow().as_slice(),
        ["event", "event", "queue", "program", "buffer", "context"]
    );
}

#[test]
fn rejected_reset_release_is_quarantined_and_reset_is_idempotent() {
    let mut processor = processor();
    submission_objects(&mut processor, BufferUsage::PROGRAM_INPUT);
    processor
        .accelerator()
        .free_mode
        .set(ReleaseMode::Reject(BackendError::Busy));

    let first = processor.reset(ObjectNamespace::new(2).unwrap()).unwrap();
    assert_eq!(first.disposition, ResetDisposition::BackendDiscardRequired);
    assert_eq!(
        first.released,
        ResourceCounts {
            programs: 1,
            queues: 1,
            ..ResourceCounts::default()
        }
    );
    assert_eq!(
        first.quarantined,
        ResourceCounts {
            contexts: 1,
            buffers: 1,
            ..ResourceCounts::default()
        }
    );
    assert_eq!(
        first.released_bytes,
        RetainedBytes {
            buffer_backing: 0,
            program_resident: u128::from(reference::RESIDENT_BYTES),
        }
    );
    assert_eq!(
        first.quarantined_bytes,
        RetainedBytes {
            buffer_backing: 8,
            program_resident: 0,
        }
    );
    assert_eq!(
        processor.accelerator().release_log.borrow().as_slice(),
        ["queue", "program", "buffer"]
    );
    assert_eq!(processor.health(), DeviceHealth::BackendDiscardRequired);

    let calls = processor.accelerator().release_log.borrow().len();
    let second = processor.reset(ObjectNamespace::new(3).unwrap()).unwrap();
    assert_eq!(second, first);
    assert_eq!(processor.accelerator().release_log.borrow().len(), calls);
}

#[test]
fn reset_stops_backend_calls_after_device_lost_release_rejection() {
    let mut processor = processor();
    submission_objects(&mut processor, BufferUsage::PROGRAM_INPUT);
    create_context(&mut processor);
    processor
        .accelerator()
        .free_mode
        .set(ReleaseMode::Reject(BackendError::DeviceLost));

    let report = processor.reset(ObjectNamespace::new(2).unwrap()).unwrap();
    assert_eq!(report.disposition, ResetDisposition::BackendDiscardRequired);
    assert_eq!(
        report.released,
        ResourceCounts {
            programs: 1,
            queues: 1,
            ..ResourceCounts::default()
        }
    );
    assert_eq!(
        report.quarantined,
        ResourceCounts {
            contexts: 2,
            buffers: 1,
            ..ResourceCounts::default()
        }
    );
    assert_eq!(
        processor.accelerator().release_log.borrow().as_slice(),
        ["queue", "program", "buffer"]
    );
}

#[test]
fn indeterminate_event_release_quarantines_the_complete_object_graph() {
    let mut processor = processor();
    let (_context, buffer, program, queue) =
        submission_objects(&mut processor, BufferUsage::PROGRAM_INPUT);
    let submit = submit_request(queue, program, buffer, 0, 8, AccessMode::Read);
    let (_, response) = run(&mut processor, &submit, 24);
    let event_id = event_id(&response);
    let event = processor
        .state()
        .event_record(event_id)
        .unwrap()
        .resource()
        .unwrap();
    processor.accelerator().inner.complete(event).unwrap();
    processor
        .accelerator()
        .event_release_mode
        .set(ReleaseMode::Indeterminate);

    let first = processor.reset(ObjectNamespace::new(2).unwrap()).unwrap();
    assert_eq!(first.disposition, ResetDisposition::BackendDiscardRequired);
    assert!(first.released.is_empty());
    assert_eq!(
        first.quarantined,
        ResourceCounts {
            contexts: 1,
            buffers: 1,
            programs: 1,
            queues: 1,
            events: 1,
        }
    );
    assert!(first.released_bytes.is_empty());
    assert_eq!(
        first.quarantined_bytes,
        RetainedBytes {
            buffer_backing: 8,
            program_resident: u128::from(reference::RESIDENT_BYTES),
        }
    );
    assert_eq!(
        processor.accelerator().release_log.borrow().as_slice(),
        ["event"]
    );

    let second = processor.reset(ObjectNamespace::new(3).unwrap()).unwrap();
    assert_eq!(second, first);
    assert_eq!(
        processor.accelerator().release_log.borrow().as_slice(),
        ["event"]
    );
}

#[test]
fn pending_event_without_cancellation_requires_backend_discard() {
    let backend = RecordingBackend::default();
    let mut info = backend.device_info().unwrap();
    backend.device_info_calls.set(0);
    info.capabilities.remove(Capabilities::EVENT_CANCELLATION);
    backend.info_override.set(Some(info));
    let mut processor = CommandProcessor::new(
        backend,
        &config(),
        ObjectNamespace::new(1).unwrap(),
        resource_policy(),
    )
    .unwrap();
    let (_context, buffer, program, queue) =
        submission_objects(&mut processor, BufferUsage::PROGRAM_INPUT);
    let submit = submit_request(queue, program, buffer, 0, 8, AccessMode::Read);
    run(&mut processor, &submit, 24);

    let report = processor.reset(ObjectNamespace::new(2).unwrap()).unwrap();
    assert_eq!(report.disposition, ResetDisposition::BackendDiscardRequired);
    assert!(report.released.is_empty());
    assert_eq!(
        report.quarantined,
        ResourceCounts {
            contexts: 1,
            buffers: 1,
            programs: 1,
            queues: 1,
            events: 1,
        }
    );
    assert_eq!(processor.accelerator().cancel_calls.get(), 0);
    assert_eq!(processor.accelerator().destroy_event_calls.get(), 0);
    assert!(processor.accelerator().release_log.borrow().is_empty());
}
