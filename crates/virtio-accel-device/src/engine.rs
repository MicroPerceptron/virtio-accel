//! Transport-neutral baseline command processing.
//!
//! `CommandProcessor` owns one backend and one device-state graph. Processing requires exclusive
//! access, so the portable layer contains no locks, atomics, reference-counted handles, or hidden
//! executor. Transport integrations may schedule distinct processors or backend work concurrently,
//! but state admission and response commitment remain explicit serialization boundaries.

use alloc::vec::Vec;

use virtio_accel_core::{
    Accelerator, ArtifactRef, BackendError, BindingRef, BufferRange, BufferUsage, ByteSink,
    ByteSource, Capabilities, DeviceInfo, DeviceInfoError, EventState, ReleaseFailure,
    SubmitFailure, Timeout,
};
use virtio_accel_proto::{
    KnownEventState, Le16, Le32, Le64, ObjectPayload, StatusCode, SubmitResponse, WireConfig,
    WireDeviceInfo, WireEventState,
};
use zerocopy::IntoBytes;

use crate::{
    BufferRecord, ChainRegion, CreateError, DecodedBinding, DecodedRequest, DecodedRequestBody,
    DecoderLimits, DecoderLimitsError, DeviceState, DeviceStateConfigError, DeviceStateError,
    FrameDecoder, FramePreflight, FramePreflightError, ObjectId, ObjectNamespace, ResourceCounts,
    ResponseWriteError, ResponseWriter, UnusableFrame, preflight_command_frame,
    status_from_backend_error, status_from_device_state_error,
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
    BackendDiscardRequired,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResetDisposition {
    BackendReusable,
    BackendDiscardRequired,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResetReport {
    pub disposition: ResetDisposition,
    pub released: ResourceCounts,
    pub quarantined: ResourceCounts,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResetError {
    NamespaceReuse,
    State(DeviceStateConfigError),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CommandProcessorInitError {
    Backend(BackendError),
    DeviceInfo(DeviceInfoError),
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SubmitCreateError {
    State(DeviceStateError),
    OutOfMemory,
    Rejected(BackendError),
    Validation(StatusCode),
}

/// One synchronization-free command engine for a backend instance.
///
/// The backend's immutable device information is fetched once during construction and then used
/// for decoding, validation, and discovery responses. This prevents limits from changing beneath
/// live object state and removes a backend call from the discovery path.
pub struct CommandProcessor<A: Accelerator> {
    state: AcceleratorState<A>,
    accelerator: A,
    info: DeviceInfo,
    decoder: FrameDecoder,
    health: DeviceHealth,
    quarantined: ResourceCounts,
    last_reset: Option<ResetReport>,
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
        info.validate()
            .map_err(CommandProcessorInitError::DeviceInfo)?;
        let limits =
            DecoderLimits::new(config, info).map_err(CommandProcessorInitError::Decoder)?;
        let state =
            DeviceState::new(namespace, info.limits).map_err(CommandProcessorInitError::State)?;
        Ok(Self {
            state,
            accelerator,
            info,
            decoder: FrameDecoder::new(limits),
            health: DeviceHealth::Running,
            quarantined: ResourceCounts::default(),
            last_reset: None,
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

    /// Perform one bounded, child-before-parent reset pass.
    ///
    /// The transport must stop fetching command chains and stop publishing completions before
    /// calling this method. A reusable result renews every object table with `namespace`; every
    /// prior ID is then stale. A discard-required result is sticky and makes no backend calls on
    /// later reset attempts. The caller must discard the complete processor/backend instance.
    pub fn reset(&mut self, namespace: ObjectNamespace) -> Result<ResetReport, ResetError> {
        if self.health == DeviceHealth::BackendDiscardRequired {
            let report = match self.last_reset {
                Some(report) => report,
                None => {
                    let report = self.discard_report(ResourceCounts::default());
                    self.last_reset = Some(report);
                    report
                }
            };
            return Ok(report);
        }
        if namespace == self.state.namespace() {
            return Err(ResetError::NamespaceReuse);
        }

        self.health = DeviceHealth::NeedsReset;
        self.last_reset = None;
        let mut released = ResourceCounts::default();
        let mut progress = ResetProgress::default();

        self.reset_events(&mut released, &mut progress);
        if progress.backend_callable {
            self.reset_queues(&mut released, &mut progress);
        }
        if progress.backend_callable {
            self.reset_programs(&mut released, &mut progress);
        }
        if progress.backend_callable {
            self.reset_buffers(&mut released, &mut progress);
        }
        if progress.backend_callable {
            self.reset_contexts(&mut released, &mut progress);
        }

        let remaining = self.state.resource_counts();
        let quarantined = self.quarantined.saturating_add(remaining);
        if progress.backend_reusable && self.state.is_empty() && quarantined.is_empty() {
            self.state =
                DeviceState::new(namespace, self.info.limits).map_err(ResetError::State)?;
            self.health = DeviceHealth::Running;
            let report = ResetReport {
                disposition: ResetDisposition::BackendReusable,
                released,
                quarantined,
            };
            self.last_reset = None;
            return Ok(report);
        }

        self.health = DeviceHealth::BackendDiscardRequired;
        let report = ResetReport {
            disposition: ResetDisposition::BackendDiscardRequired,
            released,
            quarantined,
        };
        self.last_reset = Some(report);
        Ok(report)
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
                if self.health != DeviceHealth::Running {
                    return self.respond_empty(
                        response,
                        StatusCode::DEVICE_LOST,
                        request.request_id(),
                        false,
                    );
                }
                self.last_reset = None;
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
                if let Err(error) = self.info.validate_context_desc(desc) {
                    return self.respond_backend_error(response, request_id, error, false);
                }
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
                if let Err(error) = self.info.validate_queue_desc(desc) {
                    return self.respond_backend_error(response, request_id, error, false);
                }
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
            DecodedRequestBody::Submit {
                queue_id,
                program_id,
                bindings,
                timeout,
            } => self.submit(
                response, request_id, queue_id, program_id, bindings, timeout,
            ),
            DecodedRequestBody::PollEvent { event_id } => {
                self.poll_event(response, request_id, event_id)
            }
            DecodedRequestBody::CancelEvent { event_id } => {
                self.cancel_event(response, request_id, event_id)
            }
            DecodedRequestBody::DestroyEvent { event_id } => {
                self.destroy_event(response, request_id, event_id)
            }
        }
    }

    fn submit(
        &mut self,
        response: &mut dyn ByteSink,
        request_id: u64,
        queue_id: ObjectId,
        program_id: ObjectId,
        bindings: Vec<DecodedBinding>,
        timeout: Timeout,
    ) -> Result<CommandOutcome, CommandProcessError> {
        let mut buffer_ids = Vec::new();
        if buffer_ids.try_reserve_exact(bindings.len()).is_err() {
            return self.respond_empty(response, StatusCode::OUT_OF_MEMORY, request_id, false);
        }
        buffer_ids.extend(bindings.iter().map(|binding| binding.buffer_id));

        let accelerator = &self.accelerator;
        let mut admission_status = StatusCode::OK;
        let mut admission_requires_discard = false;
        let result = self
            .state
            .create_event_with(queue_id, program_id, buffer_ids, |resources| {
                let mut native_bindings = Vec::new();
                native_bindings
                    .try_reserve_exact(bindings.len())
                    .map_err(|_| SubmitCreateError::OutOfMemory)?;
                for binding in &bindings {
                    let (buffer, info) = resources
                        .buffer_with_info_by_id(binding.buffer_id)
                        .map_err(SubmitCreateError::State)?;
                    let desc = info.desc();
                    if binding.range.end() > desc.bytes() {
                        return Err(SubmitCreateError::Validation(StatusCode::OUT_OF_BOUNDS));
                    }
                    if !desc.allows_access(binding.access) {
                        return Err(SubmitCreateError::Validation(StatusCode::PERMISSION_DENIED));
                    }
                    native_bindings.push(BindingRef {
                        slot: binding.slot,
                        buffer,
                        range: binding.range,
                        access: binding.access,
                    });
                }
                match accelerator.submit(
                    resources.queue(),
                    resources.program(),
                    &native_bindings,
                    timeout,
                ) {
                    Ok(event) => Ok(event),
                    Err(SubmitFailure::Rejected(error)) => Err(SubmitCreateError::Rejected(error)),
                    Err(SubmitFailure::Indeterminate { error, event }) => {
                        admission_status = status_from_backend_error(error);
                        admission_requires_discard = error == BackendError::DeviceLost;
                        Ok(event)
                    }
                }
            });

        match result {
            Ok(event_id) => {
                if admission_requires_discard {
                    self.require_backend_discard();
                }
                self.respond_event_id(response, admission_status, request_id, event_id, true)
            }
            Err(CreateError::State(error)) => self.respond_state_error(response, request_id, error),
            Err(CreateError::Provider(SubmitCreateError::Rejected(error))) => {
                self.respond_backend_error(response, request_id, error, false)
            }
            Err(CreateError::Provider(SubmitCreateError::OutOfMemory)) => {
                self.respond_empty(response, StatusCode::OUT_OF_MEMORY, request_id, false)
            }
            Err(CreateError::Provider(SubmitCreateError::Validation(status))) => {
                self.respond_empty(response, status, request_id, false)
            }
            Err(CreateError::Provider(SubmitCreateError::State(_))) => {
                self.discard_response(response, request_id)
            }
        }
    }

    fn reset_events(&mut self, released: &mut ResourceCounts, progress: &mut ResetProgress) {
        let mut cursor = 0;
        while let Some((next, id)) = self.state.next_event_id(cursor) {
            cursor = next;
            if !progress.backend_callable {
                break;
            }
            self.reset_event(id, released, progress);
        }
    }

    fn reset_event(
        &mut self,
        id: ObjectId,
        released: &mut ResourceCounts,
        progress: &mut ResetProgress,
    ) {
        let mut event_state = {
            let event = match self
                .state
                .event_record(id)
                .and_then(|record| record.resource())
            {
                Ok(event) => event,
                Err(_) => {
                    progress.backend_reusable = false;
                    return;
                }
            };
            match self.accelerator.poll_event(event) {
                Ok(state) => state,
                Err(error) => {
                    progress.backend_reusable = false;
                    if error == BackendError::DeviceLost {
                        progress.backend_callable = false;
                    }
                    return;
                }
            }
        };

        if event_state == EventState::Pending {
            if !self
                .info
                .capabilities
                .contains(Capabilities::EVENT_CANCELLATION)
            {
                progress.backend_reusable = false;
                return;
            }
            let cancel_result = {
                let event = match self
                    .state
                    .event_record(id)
                    .and_then(|record| record.resource())
                {
                    Ok(event) => event,
                    Err(_) => {
                        progress.backend_reusable = false;
                        return;
                    }
                };
                self.accelerator.cancel_event(event)
            };
            match cancel_result {
                Ok(()) => event_state = EventState::Cancelled,
                Err(BackendError::Busy) => {
                    let event = match self
                        .state
                        .event_record(id)
                        .and_then(|record| record.resource())
                    {
                        Ok(event) => event,
                        Err(_) => {
                            progress.backend_reusable = false;
                            return;
                        }
                    };
                    match self.accelerator.poll_event(event) {
                        Ok(state) => event_state = state,
                        Err(error) => {
                            progress.backend_reusable = false;
                            if error == BackendError::DeviceLost {
                                progress.backend_callable = false;
                            }
                            return;
                        }
                    }
                }
                Err(error) => {
                    progress.backend_reusable = false;
                    if error == BackendError::DeviceLost {
                        progress.backend_callable = false;
                    }
                    return;
                }
            }
        }

        match event_state {
            EventState::Pending => {
                progress.backend_reusable = false;
                return;
            }
            EventState::Failed(BackendError::DeviceLost) => {
                progress.backend_reusable = false;
                progress.backend_callable = false;
                return;
            }
            EventState::Complete | EventState::Failed(_) | EventState::Cancelled => {}
        }

        let event = match self.state.begin_event_release(id) {
            Ok(event) => event,
            Err(_) => {
                progress.backend_reusable = false;
                return;
            }
        };
        match self.accelerator.destroy_event(event) {
            Ok(()) => match self.state.commit_event_release(id) {
                Ok(()) => released.events += 1,
                Err(_) => {
                    progress.backend_reusable = false;
                    progress.backend_callable = false;
                }
            },
            Err(ReleaseFailure::Rejected { error, resource }) => {
                progress.backend_reusable = false;
                if error == BackendError::DeviceLost {
                    progress.backend_callable = false;
                }
                if self.state.restore_event_release(id, resource).is_err() {
                    progress.backend_callable = false;
                }
            }
            Err(ReleaseFailure::Indeterminate { .. }) => {
                progress.backend_reusable = false;
                progress.backend_callable = false;
            }
        }
    }

    fn reset_queues(&mut self, released: &mut ResourceCounts, progress: &mut ResetProgress) {
        let mut cursor = 0;
        while let Some((next, id)) = self.state.next_queue_id(cursor) {
            cursor = next;
            let queue = match self.state.begin_queue_release(id) {
                Ok(queue) => queue,
                Err(_) => {
                    progress.backend_reusable = false;
                    continue;
                }
            };
            match self.accelerator.destroy_queue(queue) {
                Ok(()) => match self.state.commit_queue_release(id) {
                    Ok(()) => released.queues += 1,
                    Err(_) => {
                        progress.backend_reusable = false;
                        progress.backend_callable = false;
                        break;
                    }
                },
                Err(ReleaseFailure::Rejected { error, resource }) => {
                    progress.backend_reusable = false;
                    if error == BackendError::DeviceLost {
                        progress.backend_callable = false;
                    }
                    if self.state.restore_queue_release(id, resource).is_err() {
                        progress.backend_callable = false;
                        break;
                    }
                }
                Err(ReleaseFailure::Indeterminate { .. }) => {
                    progress.backend_reusable = false;
                    progress.backend_callable = false;
                    break;
                }
            }
        }
    }

    fn reset_programs(&mut self, released: &mut ResourceCounts, progress: &mut ResetProgress) {
        let mut cursor = 0;
        while let Some((next, id)) = self.state.next_program_id(cursor) {
            cursor = next;
            let program = match self.state.begin_program_release(id) {
                Ok(program) => program,
                Err(_) => {
                    progress.backend_reusable = false;
                    continue;
                }
            };
            match self.accelerator.unload_program(program) {
                Ok(()) => match self.state.commit_program_release(id) {
                    Ok(()) => released.programs += 1,
                    Err(_) => {
                        progress.backend_reusable = false;
                        progress.backend_callable = false;
                        break;
                    }
                },
                Err(ReleaseFailure::Rejected { error, resource }) => {
                    progress.backend_reusable = false;
                    if error == BackendError::DeviceLost {
                        progress.backend_callable = false;
                    }
                    if self.state.restore_program_release(id, resource).is_err() {
                        progress.backend_callable = false;
                        break;
                    }
                }
                Err(ReleaseFailure::Indeterminate { .. }) => {
                    progress.backend_reusable = false;
                    progress.backend_callable = false;
                    break;
                }
            }
        }
    }

    fn reset_buffers(&mut self, released: &mut ResourceCounts, progress: &mut ResetProgress) {
        let mut cursor = 0;
        while let Some((next, id)) = self.state.next_buffer_id(cursor) {
            cursor = next;
            let buffer = match self.state.begin_buffer_release(id) {
                Ok(buffer) => buffer,
                Err(_) => {
                    progress.backend_reusable = false;
                    continue;
                }
            };
            match self.accelerator.free_buffer(buffer) {
                Ok(()) => match self.state.commit_buffer_release(id) {
                    Ok(()) => released.buffers += 1,
                    Err(_) => {
                        progress.backend_reusable = false;
                        progress.backend_callable = false;
                        break;
                    }
                },
                Err(ReleaseFailure::Rejected { error, resource }) => {
                    progress.backend_reusable = false;
                    if error == BackendError::DeviceLost {
                        progress.backend_callable = false;
                    }
                    if self.state.restore_buffer_release(id, resource).is_err() {
                        progress.backend_callable = false;
                        break;
                    }
                }
                Err(ReleaseFailure::Indeterminate { .. }) => {
                    progress.backend_reusable = false;
                    progress.backend_callable = false;
                    break;
                }
            }
        }
    }

    fn reset_contexts(&mut self, released: &mut ResourceCounts, progress: &mut ResetProgress) {
        let mut cursor = 0;
        while let Some((next, id)) = self.state.next_context_id(cursor) {
            cursor = next;
            let context = match self.state.begin_context_release(id) {
                Ok(context) => context,
                Err(_) => {
                    progress.backend_reusable = false;
                    continue;
                }
            };
            match self.accelerator.destroy_context(context) {
                Ok(()) => match self.state.commit_context_release(id) {
                    Ok(()) => released.contexts += 1,
                    Err(_) => {
                        progress.backend_reusable = false;
                        progress.backend_callable = false;
                        break;
                    }
                },
                Err(ReleaseFailure::Rejected { error, resource }) => {
                    progress.backend_reusable = false;
                    if error == BackendError::DeviceLost {
                        progress.backend_callable = false;
                    }
                    if self.state.restore_context_release(id, resource).is_err() {
                        progress.backend_callable = false;
                        break;
                    }
                }
                Err(ReleaseFailure::Indeterminate { .. }) => {
                    progress.backend_reusable = false;
                    progress.backend_callable = false;
                    break;
                }
            }
        }
    }

    fn poll_event(
        &mut self,
        response: &mut dyn ByteSink,
        request_id: u64,
        event_id: ObjectId,
    ) -> Result<CommandOutcome, CommandProcessError> {
        let event = match self.state.event_record(event_id) {
            Ok(record) => match record.resource() {
                Ok(event) => event,
                Err(error) => return self.respond_state_error(response, request_id, error),
            },
            Err(error) => return self.respond_state_error(response, request_id, error),
        };
        match self.accelerator.poll_event(event) {
            Ok(state) => {
                if state == EventState::Failed(BackendError::DeviceLost) {
                    self.require_backend_discard();
                }
                let payload = wire_event_state(state);
                self.respond_bytes(
                    response,
                    StatusCode::OK,
                    request_id,
                    payload.as_bytes(),
                    false,
                )
            }
            Err(error) => self.respond_backend_error(response, request_id, error, false),
        }
    }

    fn cancel_event(
        &mut self,
        response: &mut dyn ByteSink,
        request_id: u64,
        event_id: ObjectId,
    ) -> Result<CommandOutcome, CommandProcessError> {
        if self.info.validate_event_cancellation().is_err() {
            return self.respond_empty(response, StatusCode::UNSUPPORTED, request_id, false);
        }
        let event = match self.state.event_record(event_id) {
            Ok(record) => match record.resource() {
                Ok(event) => event,
                Err(error) => return self.respond_state_error(response, request_id, error),
            },
            Err(error) => return self.respond_state_error(response, request_id, error),
        };
        match self.accelerator.cancel_event(event) {
            Ok(()) => self.respond_empty(response, StatusCode::OK, request_id, true),
            Err(error) => self.respond_backend_error(response, request_id, error, false),
        }
    }

    fn destroy_event(
        &mut self,
        response: &mut dyn ByteSink,
        request_id: u64,
        event_id: ObjectId,
    ) -> Result<CommandOutcome, CommandProcessError> {
        let state = {
            let event = match self.state.event_record(event_id) {
                Ok(record) => match record.resource() {
                    Ok(event) => event,
                    Err(error) => return self.respond_state_error(response, request_id, error),
                },
                Err(error) => return self.respond_state_error(response, request_id, error),
            };
            self.accelerator.poll_event(event)
        };
        match state {
            Ok(EventState::Pending) => {
                return self.respond_empty(response, StatusCode::BUSY, request_id, false);
            }
            Ok(EventState::Failed(BackendError::DeviceLost)) => {
                self.require_backend_discard();
                return self.respond_empty(response, StatusCode::DEVICE_LOST, request_id, false);
            }
            Ok(EventState::Complete | EventState::Failed(_) | EventState::Cancelled) => {}
            Err(error) => {
                return self.respond_backend_error(response, request_id, error, false);
            }
        }

        let event = match self.state.begin_event_release(event_id) {
            Ok(event) => event,
            Err(error) => return self.respond_state_error(response, request_id, error),
        };
        match self.accelerator.destroy_event(event) {
            Ok(()) => match self.state.commit_event_release(event_id) {
                Ok(()) => self.respond_empty(response, StatusCode::OK, request_id, true),
                Err(_) => self.discard_response(response, request_id),
            },
            Err(ReleaseFailure::Rejected { error, resource }) => {
                match self.state.restore_event_release(event_id, resource) {
                    Ok(()) => self.respond_backend_error(response, request_id, error, false),
                    Err(_) => self.discard_response(response, request_id),
                }
            }
            Err(ReleaseFailure::Indeterminate { .. }) => {
                if self.state.commit_event_release(event_id).is_ok() {
                    self.quarantined.events += 1;
                }
                self.discard_response(response, request_id)
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
                Err(_) => self.discard_response(response, request_id),
            },
            Err(ReleaseFailure::Rejected { error, resource }) => {
                match self.state.restore_context_release(id, resource) {
                    Ok(()) => self.respond_backend_error(response, request_id, error, false),
                    Err(_) => self.discard_response(response, request_id),
                }
            }
            Err(ReleaseFailure::Indeterminate { .. }) => {
                if self.state.commit_context_release(id).is_ok() {
                    self.quarantined.contexts += 1;
                }
                self.discard_response(response, request_id)
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
                Err(_) => self.discard_response(response, request_id),
            },
            Err(ReleaseFailure::Rejected { error, resource }) => {
                match self.state.restore_buffer_release(id, resource) {
                    Ok(()) => self.respond_backend_error(response, request_id, error, false),
                    Err(_) => self.discard_response(response, request_id),
                }
            }
            Err(ReleaseFailure::Indeterminate { .. }) => {
                if self.state.commit_buffer_release(id).is_ok() {
                    self.quarantined.buffers += 1;
                }
                self.discard_response(response, request_id)
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
                Err(_) => self.discard_response(response, request_id),
            },
            Err(ReleaseFailure::Rejected { error, resource }) => {
                match self.state.restore_program_release(id, resource) {
                    Ok(()) => self.respond_backend_error(response, request_id, error, false),
                    Err(_) => self.discard_response(response, request_id),
                }
            }
            Err(ReleaseFailure::Indeterminate { .. }) => {
                if self.state.commit_program_release(id).is_ok() {
                    self.quarantined.programs += 1;
                }
                self.discard_response(response, request_id)
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
                Err(_) => self.discard_response(response, request_id),
            },
            Err(ReleaseFailure::Rejected { error, resource }) => {
                match self.state.restore_queue_release(id, resource) {
                    Ok(()) => self.respond_backend_error(response, request_id, error, false),
                    Err(_) => self.discard_response(response, request_id),
                }
            }
            Err(ReleaseFailure::Indeterminate { .. }) => {
                if self.state.commit_queue_release(id).is_ok() {
                    self.quarantined.queues += 1;
                }
                self.discard_response(response, request_id)
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
            self.require_backend_discard();
        }
        status_from_backend_error(error)
    }

    fn require_backend_discard(&mut self) {
        self.health = DeviceHealth::BackendDiscardRequired;
        self.last_reset = None;
    }

    fn discard_response(
        &mut self,
        response: &mut dyn ByteSink,
        request_id: u64,
    ) -> Result<CommandOutcome, CommandProcessError> {
        self.require_backend_discard();
        self.respond_empty(response, StatusCode::DEVICE_LOST, request_id, true)
    }

    fn discard_report(&self, released: ResourceCounts) -> ResetReport {
        ResetReport {
            disposition: ResetDisposition::BackendDiscardRequired,
            released,
            quarantined: self
                .quarantined
                .saturating_add(self.state.resource_counts()),
        }
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

    fn respond_event_id(
        &mut self,
        response: &mut dyn ByteSink,
        status: StatusCode,
        request_id: u64,
        id: ObjectId,
        mutated: bool,
    ) -> Result<CommandOutcome, CommandProcessError> {
        let payload = SubmitResponse {
            event_id: Le64::new(id.get()),
        };
        self.respond_bytes(response, status, request_id, payload.as_bytes(), mutated)
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
                if mutated && self.health == DeviceHealth::Running {
                    self.health = DeviceHealth::NeedsReset;
                    self.last_reset = None;
                }
                Err(CommandProcessError::ResponseWrite(error))
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct ResetProgress {
    backend_reusable: bool,
    backend_callable: bool,
}

impl Default for ResetProgress {
    fn default() -> Self {
        Self {
            backend_reusable: true,
            backend_callable: true,
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

fn wire_event_state(state: EventState) -> WireEventState {
    let (state, error) = match state {
        EventState::Pending => (KnownEventState::Pending, StatusCode::OK),
        EventState::Complete => (KnownEventState::Complete, StatusCode::OK),
        EventState::Failed(error) => (KnownEventState::Failed, status_from_backend_error(error)),
        EventState::Cancelled => (KnownEventState::Cancelled, StatusCode::OK),
    };
    WireEventState {
        state: Le16::new(state as u16),
        error: Le16::new(error.0),
        reserved: Le32::new(0),
    }
}
