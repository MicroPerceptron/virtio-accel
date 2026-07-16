//! Transport-neutral baseline command processing.
//!
//! `CommandProcessor` owns one backend and one device-state graph. Processing requires exclusive
//! access, so the portable layer contains no locks, atomics, reference-counted handles, or hidden
//! executor. Transport integrations may schedule distinct processors or backend work concurrently,
//! but state admission and response commitment remain explicit serialization boundaries.

use virtio_accel_core::{
    Accelerator, ArtifactRef, BackendError, BufferRange, BufferUsage, ByteSink, ByteSource,
    DeviceInfo, ReleaseFailure,
};
use virtio_accel_proto::{Le16, Le32, Le64, ObjectPayload, StatusCode, WireConfig, WireDeviceInfo};
use zerocopy::IntoBytes;

use crate::{
    BufferRecord, ChainRegion, CreateError, DecodedRequest, DecodedRequestBody, DecoderLimits,
    DecoderLimitsError, DeviceState, DeviceStateConfigError, DeviceStateError, FrameDecoder,
    FramePreflight, FramePreflightError, ObjectId, ObjectNamespace, ResponseWriteError,
    ResponseWriter, UnusableFrame, preflight_command_frame, status_from_backend_error,
    status_from_device_state_error,
};

pub type AcceleratorState<A> = DeviceState<
    <A as Accelerator>::Context,
    <A as Accelerator>::Buffer,
    <A as Accelerator>::Program,
    <A as Accelerator>::Queue,
    <A as Accelerator>::Event,
>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceHealth {
    Running,
    NeedsReset,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CommandProcessorInitError {
    Backend(BackendError),
    Decoder(DecoderLimitsError),
    State(DeviceStateConfigError),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CommandProcessError {
    Preflight(FramePreflightError),
    ResponseWrite(ResponseWriteError),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CommandOutcome {
    Response {
        request_id: u64,
        status: StatusCode,
        used: u32,
    },
    Unusable(UnusableFrame),
}

/// One synchronization-free command engine for a backend instance.
///
/// The backend's immutable device information is fetched once during construction and then used
/// for decoding, validation, and discovery responses. This prevents limits from changing beneath
/// live object state and removes a backend call from the discovery path.
pub struct CommandProcessor<A: Accelerator> {
    accelerator: A,
    info: DeviceInfo,
    decoder: FrameDecoder,
    state: AcceleratorState<A>,
    health: DeviceHealth,
}

impl<A: Accelerator> core::fmt::Debug for CommandProcessor<A> {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("CommandProcessor")
            .field("info", &self.info)
            .field("decoder", &self.decoder)
            .field("health", &self.health)
            .finish_non_exhaustive()
    }
}

impl<A: Accelerator> CommandProcessor<A> {
    pub fn new(
        accelerator: A,
        config: &WireConfig,
        namespace: ObjectNamespace,
    ) -> Result<Self, CommandProcessorInitError> {
        config.validate().map_err(|error| {
            CommandProcessorInitError::Decoder(DecoderLimitsError::Config(error))
        })?;
        let info = accelerator
            .device_info()
            .map_err(CommandProcessorInitError::Backend)?;
        let limits =
            DecoderLimits::new(config, info).map_err(CommandProcessorInitError::Decoder)?;
        let state =
            DeviceState::new(namespace, info.limits).map_err(CommandProcessorInitError::State)?;
        Ok(Self {
            accelerator,
            info,
            decoder: FrameDecoder::new(limits),
            state,
            health: DeviceHealth::Running,
        })
    }

    pub const fn device_info(&self) -> DeviceInfo {
        self.info
    }

    pub const fn health(&self) -> DeviceHealth {
        self.health
    }

    pub const fn decoder(&self) -> &FrameDecoder {
        &self.decoder
    }

    pub const fn accelerator(&self) -> &A {
        &self.accelerator
    }

    pub const fn state(&self) -> &AcceleratorState<A> {
        &self.state
    }

    /// Process one complete flattened command chain.
    ///
    /// A successful preflight guarantees response capacity for the command's largest success
    /// payload before semantic state or the backend is touched. Once recovery is required, later
    /// valid frames receive `DEVICE_LOST` without reaching semantic dispatch.
    pub fn process(
        &mut self,
        regions: &[ChainRegion],
        request: &dyn ByteSource,
        response: &mut dyn ByteSink,
    ) -> Result<CommandOutcome, CommandProcessError> {
        match preflight_command_frame(&self.decoder, regions, request, response)
            .map_err(CommandProcessError::Preflight)?
        {
            FramePreflight::Rejected {
                request_id,
                status,
                used,
            } => Ok(CommandOutcome::Response {
                request_id,
                status,
                used,
            }),
            FramePreflight::Unusable(error) => Ok(CommandOutcome::Unusable(error)),
            FramePreflight::Ready(request) => {
                if self.health == DeviceHealth::NeedsReset {
                    return self.respond_empty(
                        response,
                        StatusCode::DEVICE_LOST,
                        request.request_id(),
                        false,
                    );
                }
                self.dispatch(request, response)
            }
        }
    }

    fn dispatch(
        &mut self,
        request: DecodedRequest<'_>,
        response: &mut dyn ByteSink,
    ) -> Result<CommandOutcome, CommandProcessError> {
        let request_id = request.request_id();
        match request.into_body() {
            DecodedRequestBody::GetDeviceInfo => {
                let payload = wire_device_info(self.info);
                self.respond_bytes(
                    response,
                    StatusCode::OK,
                    request_id,
                    payload.as_bytes(),
                    false,
                )
            }
            DecodedRequestBody::CreateContext(desc) => {
                let accelerator = &self.accelerator;
                match self
                    .state
                    .create_context_with(|| accelerator.create_context(desc))
                {
                    Ok(id) => self.respond_object(response, request_id, id, true),
                    Err(error) => self.respond_create_error(response, request_id, error),
                }
            }
            DecodedRequestBody::DestroyContext { context_id } => {
                self.destroy_context(response, request_id, context_id)
            }
            DecodedRequestBody::AllocateBuffer { context_id, desc } => {
                let accelerator = &self.accelerator;
                let info = self.info;
                match self
                    .state
                    .create_buffer_with(context_id, desc, |context, requested| {
                        let allocated = accelerator.allocate_buffer(context, requested)?;
                        info.validate_buffer_info(requested, allocated.info())?;
                        Ok(allocated.into_parts())
                    }) {
                    Ok(id) => self.respond_object(response, request_id, id, true),
                    Err(error) => self.respond_create_error(response, request_id, error),
                }
            }
            DecodedRequestBody::FreeBuffer { buffer_id } => {
                self.free_buffer(response, request_id, buffer_id)
            }
            DecodedRequestBody::WriteBuffer {
                buffer_id,
                range,
                data,
            } => self.write_buffer(response, request_id, buffer_id, range, &data),
            DecodedRequestBody::ReadBuffer { buffer_id, range } => {
                self.read_buffer(response, request_id, buffer_id, range)
            }
            DecodedRequestBody::LoadProgram {
                context_id,
                format,
                target,
                payload,
                resident_bytes,
            } => {
                let accelerator = &self.accelerator;
                let artifact_bytes = payload.len();
                match self.state.create_program_with(
                    context_id,
                    artifact_bytes,
                    resident_bytes,
                    |context| {
                        accelerator.load_program(
                            context,
                            ArtifactRef {
                                format,
                                target,
                                payload: &payload,
                                resident_bytes,
                            },
                        )
                    },
                ) {
                    Ok(id) => self.respond_object(response, request_id, id, true),
                    Err(error) => self.respond_create_error(response, request_id, error),
                }
            }
            DecodedRequestBody::UnloadProgram { program_id } => {
                self.unload_program(response, request_id, program_id)
            }
            DecodedRequestBody::CreateQueue { context_id, desc } => {
                let accelerator = &self.accelerator;
                match self.state.create_queue_with(context_id, |context| {
                    accelerator.create_queue(context, desc)
                }) {
                    Ok(id) => self.respond_object(response, request_id, id, true),
                    Err(error) => self.respond_create_error(response, request_id, error),
                }
            }
            DecodedRequestBody::DestroyQueue { queue_id } => {
                self.destroy_queue(response, request_id, queue_id)
            }
            DecodedRequestBody::Submit { .. }
            | DecodedRequestBody::PollEvent { .. }
            | DecodedRequestBody::CancelEvent { .. }
            | DecodedRequestBody::DestroyEvent { .. } => {
                self.respond_empty(response, StatusCode::UNSUPPORTED, request_id, false)
            }
        }
    }

    fn write_buffer(
        &mut self,
        response: &mut dyn ByteSink,
        request_id: u64,
        buffer_id: ObjectId,
        range: BufferRange,
        data: &dyn ByteSource,
    ) -> Result<CommandOutcome, CommandProcessError> {
        let result = {
            let record = match self.state.buffer_record_mut(buffer_id) {
                Ok(record) => record,
                Err(error) => {
                    return self.respond_state_error(response, request_id, error);
                }
            };
            if let Err(status) = validate_transfer(record, range, BufferUsage::TRANSFER_DESTINATION)
            {
                return self.respond_empty(response, status, request_id, false);
            }
            let buffer = match record.resource_mut() {
                Ok(buffer) => buffer,
                Err(error) => {
                    return self.respond_state_error(response, request_id, error);
                }
            };
            self.accelerator.write_buffer(buffer, range.offset, data)
        };

        match result {
            Ok(()) => self.respond_empty(response, StatusCode::OK, request_id, true),
            Err(error) => self.respond_backend_error(response, request_id, error, true),
        }
    }

    fn read_buffer(
        &mut self,
        response: &mut dyn ByteSink,
        request_id: u64,
        buffer_id: ObjectId,
        range: BufferRange,
    ) -> Result<CommandOutcome, CommandProcessError> {
        let record = match self.state.buffer_record(buffer_id) {
            Ok(record) => record,
            Err(error) => {
                return self.respond_state_error(response, request_id, error);
            }
        };
        if let Err(status) = validate_transfer(record, range, BufferUsage::TRANSFER_SOURCE) {
            return self.respond_empty(response, status, request_id, false);
        }
        let buffer = match record.resource() {
            Ok(buffer) => buffer,
            Err(error) => {
                return self.respond_state_error(response, request_id, error);
            }
        };

        let mut writer = ResponseWriter::new(response, self.decoder.limits().max_response_bytes());
        let error = {
            let mut payload = writer
                .payload(range.bytes())
                .map_err(CommandProcessError::ResponseWrite)?;
            match self
                .accelerator
                .read_buffer(buffer, range.offset, &mut payload)
            {
                Ok(()) => {
                    let result = payload.commit(StatusCode::OK, request_id);
                    return self.complete_response(result, StatusCode::OK, request_id, false);
                }
                Err(error) => error,
            }
        };
        let status = self.backend_status(error);
        let result = writer.write_empty(status, request_id);
        self.complete_response(result, status, request_id, false)
    }

    fn destroy_context(
        &mut self,
        response: &mut dyn ByteSink,
        request_id: u64,
        id: ObjectId,
    ) -> Result<CommandOutcome, CommandProcessError> {
        let resource = match self.state.begin_context_release(id) {
            Ok(resource) => resource,
            Err(error) => return self.respond_state_error(response, request_id, error),
        };
        match self.accelerator.destroy_context(resource) {
            Ok(()) => match self.state.commit_context_release(id) {
                Ok(()) => self.respond_empty(response, StatusCode::OK, request_id, true),
                Err(_) => self.recovery_response(response, request_id),
            },
            Err(ReleaseFailure::Rejected { error, resource }) => {
                match self.state.restore_context_release(id, resource) {
                    Ok(()) => self.respond_backend_error(response, request_id, error, false),
                    Err(_) => self.recovery_response(response, request_id),
                }
            }
            Err(ReleaseFailure::Indeterminate { .. }) => {
                let _ = self.state.commit_context_release(id);
                self.recovery_response(response, request_id)
            }
        }
    }

    fn free_buffer(
        &mut self,
        response: &mut dyn ByteSink,
        request_id: u64,
        id: ObjectId,
    ) -> Result<CommandOutcome, CommandProcessError> {
        let resource = match self.state.begin_buffer_release(id) {
            Ok(resource) => resource,
            Err(error) => return self.respond_state_error(response, request_id, error),
        };
        match self.accelerator.free_buffer(resource) {
            Ok(()) => match self.state.commit_buffer_release(id) {
                Ok(()) => self.respond_empty(response, StatusCode::OK, request_id, true),
                Err(_) => self.recovery_response(response, request_id),
            },
            Err(ReleaseFailure::Rejected { error, resource }) => {
                match self.state.restore_buffer_release(id, resource) {
                    Ok(()) => self.respond_backend_error(response, request_id, error, false),
                    Err(_) => self.recovery_response(response, request_id),
                }
            }
            Err(ReleaseFailure::Indeterminate { .. }) => {
                let _ = self.state.commit_buffer_release(id);
                self.recovery_response(response, request_id)
            }
        }
    }

    fn unload_program(
        &mut self,
        response: &mut dyn ByteSink,
        request_id: u64,
        id: ObjectId,
    ) -> Result<CommandOutcome, CommandProcessError> {
        let resource = match self.state.begin_program_release(id) {
            Ok(resource) => resource,
            Err(error) => return self.respond_state_error(response, request_id, error),
        };
        match self.accelerator.unload_program(resource) {
            Ok(()) => match self.state.commit_program_release(id) {
                Ok(()) => self.respond_empty(response, StatusCode::OK, request_id, true),
                Err(_) => self.recovery_response(response, request_id),
            },
            Err(ReleaseFailure::Rejected { error, resource }) => {
                match self.state.restore_program_release(id, resource) {
                    Ok(()) => self.respond_backend_error(response, request_id, error, false),
                    Err(_) => self.recovery_response(response, request_id),
                }
            }
            Err(ReleaseFailure::Indeterminate { .. }) => {
                let _ = self.state.commit_program_release(id);
                self.recovery_response(response, request_id)
            }
        }
    }

    fn destroy_queue(
        &mut self,
        response: &mut dyn ByteSink,
        request_id: u64,
        id: ObjectId,
    ) -> Result<CommandOutcome, CommandProcessError> {
        let resource = match self.state.begin_queue_release(id) {
            Ok(resource) => resource,
            Err(error) => return self.respond_state_error(response, request_id, error),
        };
        match self.accelerator.destroy_queue(resource) {
            Ok(()) => match self.state.commit_queue_release(id) {
                Ok(()) => self.respond_empty(response, StatusCode::OK, request_id, true),
                Err(_) => self.recovery_response(response, request_id),
            },
            Err(ReleaseFailure::Rejected { error, resource }) => {
                match self.state.restore_queue_release(id, resource) {
                    Ok(()) => self.respond_backend_error(response, request_id, error, false),
                    Err(_) => self.recovery_response(response, request_id),
                }
            }
            Err(ReleaseFailure::Indeterminate { .. }) => {
                let _ = self.state.commit_queue_release(id);
                self.recovery_response(response, request_id)
            }
        }
    }

    fn respond_create_error(
        &mut self,
        response: &mut dyn ByteSink,
        request_id: u64,
        error: CreateError<BackendError>,
    ) -> Result<CommandOutcome, CommandProcessError> {
        match error {
            CreateError::State(error) => self.respond_state_error(response, request_id, error),
            CreateError::Provider(error) => {
                self.respond_backend_error(response, request_id, error, false)
            }
        }
    }

    fn respond_state_error(
        &mut self,
        response: &mut dyn ByteSink,
        request_id: u64,
        error: DeviceStateError,
    ) -> Result<CommandOutcome, CommandProcessError> {
        self.respond_empty(
            response,
            status_from_device_state_error(error),
            request_id,
            false,
        )
    }

    fn respond_backend_error(
        &mut self,
        response: &mut dyn ByteSink,
        request_id: u64,
        error: BackendError,
        mutated: bool,
    ) -> Result<CommandOutcome, CommandProcessError> {
        let status = self.backend_status(error);
        self.respond_empty(response, status, request_id, mutated)
    }

    fn backend_status(&mut self, error: BackendError) -> StatusCode {
        if error == BackendError::DeviceLost {
            self.health = DeviceHealth::NeedsReset;
        }
        status_from_backend_error(error)
    }

    fn recovery_response(
        &mut self,
        response: &mut dyn ByteSink,
        request_id: u64,
    ) -> Result<CommandOutcome, CommandProcessError> {
        self.health = DeviceHealth::NeedsReset;
        self.respond_empty(response, StatusCode::DEVICE_LOST, request_id, true)
    }

    fn respond_object(
        &mut self,
        response: &mut dyn ByteSink,
        request_id: u64,
        id: ObjectId,
        mutated: bool,
    ) -> Result<CommandOutcome, CommandProcessError> {
        let payload = ObjectPayload {
            object_id: Le64::new(id.get()),
        };
        self.respond_bytes(
            response,
            StatusCode::OK,
            request_id,
            payload.as_bytes(),
            mutated,
        )
    }

    fn respond_bytes(
        &mut self,
        response: &mut dyn ByteSink,
        status: StatusCode,
        request_id: u64,
        payload: &[u8],
        mutated: bool,
    ) -> Result<CommandOutcome, CommandProcessError> {
        let result = (|| {
            let mut writer =
                ResponseWriter::new(response, self.decoder.limits().max_response_bytes());
            let payload_bytes =
                u64::try_from(payload.len()).map_err(|_| ResponseWriteError::FrameTooLarge)?;
            let mut destination = writer.payload(payload_bytes)?;
            destination
                .write_at(0, payload)
                .map_err(|_| ResponseWriteError::SinkAccess)?;
            destination.commit(status, request_id)
        })();
        self.complete_response(result, status, request_id, mutated)
    }

    fn respond_empty(
        &mut self,
        response: &mut dyn ByteSink,
        status: StatusCode,
        request_id: u64,
        mutated: bool,
    ) -> Result<CommandOutcome, CommandProcessError> {
        let result = ResponseWriter::new(response, self.decoder.limits().max_response_bytes())
            .write_empty(status, request_id);
        self.complete_response(result, status, request_id, mutated)
    }

    fn complete_response(
        &mut self,
        result: Result<u32, ResponseWriteError>,
        status: StatusCode,
        request_id: u64,
        mutated: bool,
    ) -> Result<CommandOutcome, CommandProcessError> {
        match result {
            Ok(used) => Ok(CommandOutcome::Response {
                request_id,
                status,
                used,
            }),
            Err(error) => {
                if mutated {
                    self.health = DeviceHealth::NeedsReset;
                }
                Err(CommandProcessError::ResponseWrite(error))
            }
        }
    }
}

fn validate_transfer<B>(
    record: &BufferRecord<B>,
    range: BufferRange,
    required_usage: BufferUsage,
) -> Result<(), StatusCode> {
    if record.in_flight() != 0 {
        return Err(StatusCode::BUSY);
    }
    let desc = record.info().desc();
    if range.end() > desc.bytes() {
        return Err(StatusCode::OUT_OF_BOUNDS);
    }
    if !desc.usage.contains(required_usage) {
        return Err(StatusCode::PERMISSION_DENIED);
    }
    Ok(())
}

fn wire_device_info(info: DeviceInfo) -> WireDeviceInfo {
    WireDeviceInfo {
        uuid: info.identity.uuid,
        class: Le16::new(info.identity.class.get()),
        reserved: Le16::new(0),
        vendor_id: Le32::new(info.identity.vendor_id),
        device_id: Le32::new(info.identity.device_id),
        capabilities: Le64::new(info.capabilities.bits()),
        max_contexts: Le32::new(info.limits.max_contexts),
        max_buffers_per_context: Le32::new(info.limits.max_buffers_per_context),
        max_programs_per_context: Le32::new(info.limits.max_programs_per_context),
        max_queues_per_context: Le32::new(info.limits.max_queues_per_context),
        max_events_per_context: Le32::new(info.limits.max_events_per_context),
        max_bindings_per_submission: Le32::new(info.limits.max_bindings_per_submission),
        max_buffer_bytes: Le64::new(info.limits.max_buffer_bytes),
        max_artifact_bytes: Le64::new(info.limits.max_artifact_bytes),
    }
}
