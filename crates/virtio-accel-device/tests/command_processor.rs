use std::cell::Cell;
use std::rc::Rc;
use std::vec::Vec;

use virtio_accel_core::{
    Accelerator, AllocatedBuffer, ArtifactRef, BackendError, BindingRef, BufferDesc, BufferUsage,
    ByteSink, ByteSource, ContextDesc, DeviceInfo, EventState, MemoryDomain, QueueDesc,
    ReleaseFailure, SubmitFailure, Timeout,
};
use virtio_accel_device::{
    ChainRegion, CommandOutcome, CommandProcessError, CommandProcessor, CommandProcessorInitError,
    DecoderLimitsError, DeviceHealth, DeviceStateError, ObjectId, ObjectNamespace,
    ResponseWriteError, SegmentedSink, SegmentedSource, UnusableFrame,
};
use virtio_accel_mock::{
    MockAccelerator, MockBuffer, MockContext, MockEvent, MockProgram, MockQueue,
};
use virtio_accel_proto::{
    AllocateBufferRequest, BASELINE_COMMAND_QUEUES, CreateContextRequest, CreateQueueRequest,
    HARD_MAX_REQUEST_BYTES, HARD_MAX_RESPONSE_BYTES, Le16, Le32, Le64, LoadProgramRequest,
    ObjectPayload, PROTOCOL_MAJOR, PROTOCOL_MINOR, RequestFlags, RequestHeader, ResponseHeader,
    StatusCode, TransferBufferRequest, WireConfig, WireDeviceInfo, read_exact,
};
use zerocopy::IntoBytes;

const REQUEST_ID: u64 = 0x0102_0304_0506_0708;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ReleaseMode {
    Pass,
    Reject,
    Indeterminate,
}

struct RecordingBackend {
    inner: MockAccelerator,
    info_override: Cell<Option<DeviceInfo>>,
    device_info_calls: Rc<Cell<u32>>,
    create_context_error: Cell<Option<BackendError>>,
    create_context_calls: Cell<u32>,
    free_mode: Cell<ReleaseMode>,
    free_calls: Cell<u32>,
    write_calls: Cell<u32>,
    read_calls: Cell<u32>,
    segmented_write: Cell<bool>,
    segmented_read: Cell<bool>,
    segmented_artifact: Cell<bool>,
}

impl Default for RecordingBackend {
    fn default() -> Self {
        Self {
            inner: MockAccelerator::default(),
            info_override: Cell::new(None),
            device_info_calls: Rc::new(Cell::new(0)),
            create_context_error: Cell::new(None),
            create_context_calls: Cell::new(0),
            free_mode: Cell::new(ReleaseMode::Pass),
            free_calls: Cell::new(0),
            write_calls: Cell::new(0),
            read_calls: Cell::new(0),
            segmented_write: Cell::new(false),
            segmented_read: Cell::new(false),
            segmented_artifact: Cell::new(false),
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
        self.inner.destroy_context(context)
    }

    fn allocate_buffer(
        &self,
        context: &Self::Context,
        desc: BufferDesc,
    ) -> Result<AllocatedBuffer<Self::Buffer>, BackendError> {
        self.inner.allocate_buffer(context, desc)
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
        match self.free_mode.get() {
            ReleaseMode::Pass => self.inner.free_buffer(buffer),
            ReleaseMode::Reject => Err(ReleaseFailure::Rejected {
                error: BackendError::Busy,
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
        self.segmented_artifact
            .set(artifact.payload.as_contiguous().is_none());
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
        bindings: &[BindingRef<'_, Self::Buffer>],
        timeout: Timeout,
    ) -> Result<Self::Event, SubmitFailure<Self::Event>> {
        self.inner.submit(queue, program, bindings, timeout)
    }

    fn poll_event(&self, event: &Self::Event) -> Result<EventState, BackendError> {
        self.inner.poll_event(event)
    }

    fn cancel_event(&self, event: &Self::Event) -> Result<(), BackendError> {
        self.inner.cancel_event(event)
    }

    fn destroy_event(&self, event: Self::Event) -> Result<(), ReleaseFailure<Self::Event>> {
        self.inner.destroy_event(event)
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

fn processor() -> CommandProcessor<RecordingBackend> {
    CommandProcessor::new(
        RecordingBackend::default(),
        &config(),
        ObjectNamespace::new(1).unwrap(),
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

fn run(
    processor: &mut CommandProcessor<RecordingBackend>,
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

fn create_context(processor: &mut CommandProcessor<RecordingBackend>) -> ObjectId {
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

fn allocate_buffer(
    processor: &mut CommandProcessor<RecordingBackend>,
    context_id: ObjectId,
) -> ObjectId {
    allocate_buffer_with_usage(
        processor,
        context_id,
        BufferUsage::TRANSFER_SOURCE | BufferUsage::TRANSFER_DESTINATION,
    )
}

fn allocate_buffer_with_usage(
    processor: &mut CommandProcessor<RecordingBackend>,
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

#[test]
fn invalid_config_is_rejected_before_backend_discovery() {
    let backend = RecordingBackend::default();
    let device_info_calls = Rc::clone(&backend.device_info_calls);
    let mut invalid = config();
    invalid.protocol_major = Le16::new(PROTOCOL_MAJOR + 1);

    assert!(matches!(
        CommandProcessor::new(backend, &invalid, ObjectNamespace::new(1).unwrap(),),
        Err(CommandProcessorInitError::Decoder(
            DecoderLimitsError::Config(virtio_accel_proto::ConfigError::Version)
        ))
    ));
    assert_eq!(device_info_calls.get(), 0);
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

    let artifact = b"program";
    let load = LoadProgramRequest {
        context_id: Le64::new(context.get()),
        format: Le32::new(1),
        flags: Le32::new(0),
        target: [Le32::new(0); 12],
        payload_bytes: Le64::new(artifact.as_slice().len() as u64),
        resident_bytes: Le64::new(artifact.as_slice().len() as u64),
    };
    let mut load_payload = Vec::from(load.as_bytes());
    load_payload.extend_from_slice(artifact);
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
fn semantic_validation_prevents_backend_calls() {
    let backend = RecordingBackend::default();
    let mut info = backend.device_info().unwrap();
    backend.device_info_calls.set(0);
    info.limits.max_contexts = 1;
    backend.info_override.set(Some(info));
    let mut processor =
        CommandProcessor::new(backend, &config(), ObjectNamespace::new(1).unwrap()).unwrap();

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

    processor.accelerator().free_mode.set(ReleaseMode::Reject);
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
    assert_eq!(processor.health(), DeviceHealth::NeedsReset);
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
}
