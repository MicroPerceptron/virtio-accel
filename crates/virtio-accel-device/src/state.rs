//! Context-scoped device ownership, quotas, references, and release transitions.
//!
//! `DeviceState` contains no locks or interior mutability. Every transition requires exclusive
//! access, giving a future concurrent command engine one outer synchronization boundary and no
//! internal lock-ordering graph. Creation methods validate quotas and reserve table capacity before
//! invoking a provider closure. Release methods move resources through an explicit `Releasing`
//! state so rejected provider releases can restore ownership without reviving a stale ID.

use alloc::vec::Vec;

use virtio_accel_core::{BufferDesc, BufferInfo, DeviceLimits};
use virtio_accel_proto::HARD_MAX_BINDINGS;

use crate::{ObjectId, ObjectKind, ObjectNamespace, ObjectTable, ObjectTableError};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceStateConfigError {
    ZeroLimit,
    BindingLimit,
    CountOverflow,
    ReferenceCountOverflow,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceStateError {
    InvalidArgument,
    InvalidObject,
    StaleObject,
    ContextMismatch,
    Busy,
    ResourceLimit,
    OutOfMemory,
    Releasing,
    InvalidTransition,
    ReferenceCountOverflow,
}

#[derive(Debug)]
pub enum CreateError<E> {
    State(DeviceStateError),
    Provider(E),
}

impl<E> From<DeviceStateError> for CreateError<E> {
    fn from(error: DeviceStateError) -> Self {
        Self::State(error)
    }
}

#[derive(Debug)]
pub struct RestoreError<R> {
    pub error: DeviceStateError,
    pub resource: R,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReleaseState {
    Live,
    Releasing,
}

#[derive(Debug)]
struct ResourceSlot<R> {
    resource: Option<R>,
    release: ReleaseState,
}

impl<R> ResourceSlot<R> {
    fn new(resource: R) -> Self {
        Self {
            resource: Some(resource),
            release: ReleaseState::Live,
        }
    }

    fn get(&self) -> Result<&R, DeviceStateError> {
        self.resource.as_ref().ok_or(DeviceStateError::Releasing)
    }

    fn get_mut(&mut self) -> Result<&mut R, DeviceStateError> {
        self.resource.as_mut().ok_or(DeviceStateError::Releasing)
    }

    fn begin_release(&mut self) -> Result<R, DeviceStateError> {
        if self.release != ReleaseState::Live {
            return Err(DeviceStateError::Releasing);
        }
        let resource = self.resource.take().ok_or(DeviceStateError::Releasing)?;
        self.release = ReleaseState::Releasing;
        Ok(resource)
    }

    fn restore(&mut self, resource: R) -> Result<(), RestoreError<R>> {
        if self.release != ReleaseState::Releasing || self.resource.is_some() {
            return Err(RestoreError {
                error: DeviceStateError::InvalidTransition,
                resource,
            });
        }
        self.resource = Some(resource);
        self.release = ReleaseState::Live;
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ChildCounts {
    pub buffers: u32,
    pub programs: u32,
    pub queues: u32,
    pub events: u32,
}

impl ChildCounts {
    pub const fn is_empty(self) -> bool {
        self.buffers == 0 && self.programs == 0 && self.queues == 0 && self.events == 0
    }
}

#[derive(Debug)]
pub struct ContextRecord<C> {
    resource: ResourceSlot<C>,
    children: ChildCounts,
}

impl<C> ContextRecord<C> {
    pub fn resource(&self) -> Result<&C, DeviceStateError> {
        self.resource.get()
    }

    pub const fn release_state(&self) -> ReleaseState {
        self.resource.release
    }

    pub const fn children(&self) -> ChildCounts {
        self.children
    }
}

#[derive(Debug)]
pub struct BufferRecord<B> {
    resource: ResourceSlot<B>,
    context_id: ObjectId,
    info: BufferInfo,
    in_flight: u32,
}

impl<B> BufferRecord<B> {
    pub fn resource(&self) -> Result<&B, DeviceStateError> {
        self.resource.get()
    }

    /// Borrow the live provider buffer for an explicit mutating operation.
    ///
    /// The command engine remains responsible for validating the operation and any in-flight
    /// access policy before invoking the provider.
    pub fn resource_mut(&mut self) -> Result<&mut B, DeviceStateError> {
        self.resource.get_mut()
    }

    pub const fn context_id(&self) -> ObjectId {
        self.context_id
    }

    pub const fn info(&self) -> BufferInfo {
        self.info
    }

    pub const fn in_flight(&self) -> u32 {
        self.in_flight
    }

    pub const fn release_state(&self) -> ReleaseState {
        self.resource.release
    }
}

#[derive(Debug)]
pub struct ProgramRecord<P> {
    resource: ResourceSlot<P>,
    context_id: ObjectId,
    resident_bytes: u64,
    in_flight: u32,
}

impl<P> ProgramRecord<P> {
    pub fn resource(&self) -> Result<&P, DeviceStateError> {
        self.resource.get()
    }

    pub const fn context_id(&self) -> ObjectId {
        self.context_id
    }

    pub const fn resident_bytes(&self) -> u64 {
        self.resident_bytes
    }

    pub const fn in_flight(&self) -> u32 {
        self.in_flight
    }

    pub const fn release_state(&self) -> ReleaseState {
        self.resource.release
    }
}

#[derive(Debug)]
pub struct QueueRecord<Q> {
    resource: ResourceSlot<Q>,
    context_id: ObjectId,
    in_flight: u32,
}

impl<Q> QueueRecord<Q> {
    pub fn resource(&self) -> Result<&Q, DeviceStateError> {
        self.resource.get()
    }

    pub const fn context_id(&self) -> ObjectId {
        self.context_id
    }

    pub const fn in_flight(&self) -> u32 {
        self.in_flight
    }

    pub const fn release_state(&self) -> ReleaseState {
        self.resource.release
    }
}

#[derive(Debug)]
pub struct EventRecord<E> {
    resource: ResourceSlot<E>,
    context_id: ObjectId,
    queue_id: ObjectId,
    program_id: ObjectId,
    buffer_ids: Vec<ObjectId>,
}

impl<E> EventRecord<E> {
    pub fn resource(&self) -> Result<&E, DeviceStateError> {
        self.resource.get()
    }

    pub const fn context_id(&self) -> ObjectId {
        self.context_id
    }

    pub const fn queue_id(&self) -> ObjectId {
        self.queue_id
    }

    pub const fn program_id(&self) -> ObjectId {
        self.program_id
    }

    pub fn buffer_ids(&self) -> &[ObjectId] {
        &self.buffer_ids
    }

    pub const fn release_state(&self) -> ReleaseState {
        self.resource.release
    }
}

/// Validated provider resources for one event-producing submission.
pub struct SubmissionResources<'a, B, P, Q> {
    context_id: ObjectId,
    queue: &'a Q,
    program: &'a P,
    buffers: &'a ObjectTable<BufferRecord<B>>,
    buffer_ids: &'a [ObjectId],
}

impl<B, P, Q> core::fmt::Debug for SubmissionResources<'_, B, P, Q> {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("SubmissionResources")
            .field("context_id", &self.context_id)
            .field("buffer_ids", &self.buffer_ids)
            .finish_non_exhaustive()
    }
}

impl<'a, B, P, Q> SubmissionResources<'a, B, P, Q> {
    pub const fn context_id(&self) -> ObjectId {
        self.context_id
    }

    pub const fn queue(&self) -> &'a Q {
        self.queue
    }

    pub const fn program(&self) -> &'a P {
        self.program
    }

    pub fn buffer_ids(&self) -> &[ObjectId] {
        self.buffer_ids
    }

    pub fn buffer(&self, index: usize) -> Result<&'a B, DeviceStateError> {
        let id = *self
            .buffer_ids
            .get(index)
            .ok_or(DeviceStateError::InvalidArgument)?;
        self.buffers.get(id).map_err(map_table_error)?.resource()
    }
}

/// Complete typed object graph for one device instance.
pub struct DeviceState<C, B, P, Q, E> {
    limits: DeviceLimits,
    contexts: ObjectTable<ContextRecord<C>>,
    buffers: ObjectTable<BufferRecord<B>>,
    programs: ObjectTable<ProgramRecord<P>>,
    queues: ObjectTable<QueueRecord<Q>>,
    events: ObjectTable<EventRecord<E>>,
}

impl<C, B, P, Q, E> DeviceState<C, B, P, Q, E> {
    pub fn new(
        namespace: ObjectNamespace,
        limits: DeviceLimits,
    ) -> Result<Self, DeviceStateConfigError> {
        if limits.max_contexts == 0
            || limits.max_buffers_per_context == 0
            || limits.max_programs_per_context == 0
            || limits.max_queues_per_context == 0
            || limits.max_events_per_context == 0
            || limits.max_buffer_bytes == 0
            || limits.max_artifact_bytes == 0
        {
            return Err(DeviceStateConfigError::ZeroLimit);
        }
        if !(1..=HARD_MAX_BINDINGS).contains(&limits.max_bindings_per_submission) {
            return Err(DeviceStateConfigError::BindingLimit);
        }
        limits
            .max_events_per_context
            .checked_mul(limits.max_bindings_per_submission)
            .ok_or(DeviceStateConfigError::ReferenceCountOverflow)?;

        let buffers = aggregate_slots(limits.max_contexts, limits.max_buffers_per_context)?;
        let programs = aggregate_slots(limits.max_contexts, limits.max_programs_per_context)?;
        let queues = aggregate_slots(limits.max_contexts, limits.max_queues_per_context)?;
        let events = aggregate_slots(limits.max_contexts, limits.max_events_per_context)?;

        Ok(Self {
            limits,
            contexts: ObjectTable::with_namespace(
                ObjectKind::Context,
                limits.max_contexts,
                namespace,
            ),
            buffers: ObjectTable::with_namespace(ObjectKind::Buffer, buffers, namespace),
            programs: ObjectTable::with_namespace(ObjectKind::Program, programs, namespace),
            queues: ObjectTable::with_namespace(ObjectKind::Queue, queues, namespace),
            events: ObjectTable::with_namespace(ObjectKind::Event, events, namespace),
        })
    }

    pub const fn limits(&self) -> DeviceLimits {
        self.limits
    }

    pub const fn context_count(&self) -> u32 {
        self.contexts.len()
    }

    pub const fn buffer_count(&self) -> u32 {
        self.buffers.len()
    }

    pub const fn program_count(&self) -> u32 {
        self.programs.len()
    }

    pub const fn queue_count(&self) -> u32 {
        self.queues.len()
    }

    pub const fn event_count(&self) -> u32 {
        self.events.len()
    }

    pub fn context_record(&self, id: ObjectId) -> Result<&ContextRecord<C>, DeviceStateError> {
        self.contexts.get(id).map_err(map_table_error)
    }

    pub fn buffer_record(&self, id: ObjectId) -> Result<&BufferRecord<B>, DeviceStateError> {
        self.buffers.get(id).map_err(map_table_error)
    }

    pub fn buffer_record_mut(
        &mut self,
        id: ObjectId,
    ) -> Result<&mut BufferRecord<B>, DeviceStateError> {
        self.buffers.get_mut(id).map_err(map_table_error)
    }

    pub fn program_record(&self, id: ObjectId) -> Result<&ProgramRecord<P>, DeviceStateError> {
        self.programs.get(id).map_err(map_table_error)
    }

    pub fn queue_record(&self, id: ObjectId) -> Result<&QueueRecord<Q>, DeviceStateError> {
        self.queues.get(id).map_err(map_table_error)
    }

    pub fn event_record(&self, id: ObjectId) -> Result<&EventRecord<E>, DeviceStateError> {
        self.events.get(id).map_err(map_table_error)
    }

    pub fn create_context_with<ProviderError>(
        &mut self,
        create: impl FnOnce() -> Result<C, ProviderError>,
    ) -> Result<ObjectId, CreateError<ProviderError>> {
        if self.contexts.len() >= self.limits.max_contexts {
            return Err(CreateError::State(DeviceStateError::ResourceLimit));
        }
        self.contexts
            .try_reserve_insert()
            .map_err(|error| CreateError::State(map_table_error(error)))?;
        let resource = create().map_err(CreateError::Provider)?;
        Ok(self.contexts.insert_prepared(ContextRecord {
            resource: ResourceSlot::new(resource),
            children: ChildCounts::default(),
        }))
    }

    pub fn create_buffer_with<ProviderError>(
        &mut self,
        context_id: ObjectId,
        desc: BufferDesc,
        create: impl FnOnce(&C, BufferDesc) -> Result<(B, BufferInfo), ProviderError>,
    ) -> Result<ObjectId, CreateError<ProviderError>> {
        if desc.bytes() > self.limits.max_buffer_bytes {
            return Err(CreateError::State(DeviceStateError::ResourceLimit));
        }
        self.check_child_admission(
            context_id,
            ChildKind::Buffer,
            self.limits.max_buffers_per_context,
        )
        .map_err(CreateError::State)?;
        self.buffers
            .try_reserve_insert()
            .map_err(|error| CreateError::State(map_table_error(error)))?;

        let context = self
            .contexts
            .get_mut(context_id)
            .map_err(|error| CreateError::State(map_table_error(error)))?;
        let (resource, info) = create(context.resource()?, desc).map_err(CreateError::Provider)?;
        let id = self.buffers.insert_prepared(BufferRecord {
            resource: ResourceSlot::new(resource),
            context_id,
            info,
            in_flight: 0,
        });
        context.children.buffers += 1;
        Ok(id)
    }

    pub fn create_program_with<ProviderError>(
        &mut self,
        context_id: ObjectId,
        artifact_bytes: u64,
        resident_bytes: u64,
        create: impl FnOnce(&C) -> Result<P, ProviderError>,
    ) -> Result<ObjectId, CreateError<ProviderError>> {
        if artifact_bytes == 0 || resident_bytes == 0 {
            return Err(CreateError::State(DeviceStateError::InvalidArgument));
        }
        if artifact_bytes > self.limits.max_artifact_bytes {
            return Err(CreateError::State(DeviceStateError::ResourceLimit));
        }
        self.check_child_admission(
            context_id,
            ChildKind::Program,
            self.limits.max_programs_per_context,
        )
        .map_err(CreateError::State)?;
        self.programs
            .try_reserve_insert()
            .map_err(|error| CreateError::State(map_table_error(error)))?;

        let context = self
            .contexts
            .get_mut(context_id)
            .map_err(|error| CreateError::State(map_table_error(error)))?;
        let resource = create(context.resource()?).map_err(CreateError::Provider)?;
        let id = self.programs.insert_prepared(ProgramRecord {
            resource: ResourceSlot::new(resource),
            context_id,
            resident_bytes,
            in_flight: 0,
        });
        context.children.programs += 1;
        Ok(id)
    }

    pub fn create_queue_with<ProviderError>(
        &mut self,
        context_id: ObjectId,
        create: impl FnOnce(&C) -> Result<Q, ProviderError>,
    ) -> Result<ObjectId, CreateError<ProviderError>> {
        self.check_child_admission(
            context_id,
            ChildKind::Queue,
            self.limits.max_queues_per_context,
        )
        .map_err(CreateError::State)?;
        self.queues
            .try_reserve_insert()
            .map_err(|error| CreateError::State(map_table_error(error)))?;

        let context = self
            .contexts
            .get_mut(context_id)
            .map_err(|error| CreateError::State(map_table_error(error)))?;
        let resource = create(context.resource()?).map_err(CreateError::Provider)?;
        let id = self.queues.insert_prepared(QueueRecord {
            resource: ResourceSlot::new(resource),
            context_id,
            in_flight: 0,
        });
        context.children.queues += 1;
        Ok(id)
    }

    pub fn create_event_with<ProviderError>(
        &mut self,
        queue_id: ObjectId,
        program_id: ObjectId,
        buffer_ids: &[ObjectId],
        create: impl FnOnce(SubmissionResources<'_, B, P, Q>) -> Result<E, ProviderError>,
    ) -> Result<ObjectId, CreateError<ProviderError>> {
        if buffer_ids.is_empty() {
            return Err(CreateError::State(DeviceStateError::InvalidArgument));
        }
        if buffer_ids.len() > self.limits.max_bindings_per_submission as usize {
            return Err(CreateError::State(DeviceStateError::ResourceLimit));
        }

        let context_id = self
            .validate_submission(queue_id, program_id, buffer_ids)
            .map_err(CreateError::State)?;
        let context = self
            .contexts
            .get(context_id)
            .map_err(|error| CreateError::State(map_table_error(error)))?;
        if context.children.events >= self.limits.max_events_per_context {
            return Err(CreateError::State(DeviceStateError::ResourceLimit));
        }
        self.events
            .try_reserve_insert()
            .map_err(|error| CreateError::State(map_table_error(error)))?;

        let mut retained_buffers = Vec::new();
        retained_buffers
            .try_reserve_exact(buffer_ids.len())
            .map_err(|_| CreateError::State(DeviceStateError::OutOfMemory))?;
        retained_buffers.extend_from_slice(buffer_ids);
        retained_buffers.sort_unstable_by_key(|id| id.get());
        self.check_reference_increments(queue_id, program_id, &retained_buffers)
            .map_err(CreateError::State)?;
        self.increment_event_references(context_id, queue_id, program_id, buffer_ids)
            .map_err(CreateError::State)?;

        let event_result = {
            let queue = self
                .queues
                .get(queue_id)
                .map_err(|error| CreateError::State(map_table_error(error)))?
                .resource()?;
            let program = self
                .programs
                .get(program_id)
                .map_err(|error| CreateError::State(map_table_error(error)))?
                .resource()?;
            create(SubmissionResources {
                context_id,
                queue,
                program,
                buffers: &self.buffers,
                buffer_ids,
            })
        };
        let event = match event_result {
            Ok(event) => event,
            Err(error) => {
                self.decrement_event_references(context_id, queue_id, program_id, buffer_ids)
                    .map_err(CreateError::State)?;
                return Err(CreateError::Provider(error));
            }
        };

        let id = self.events.insert_prepared(EventRecord {
            resource: ResourceSlot::new(event),
            context_id,
            queue_id,
            program_id,
            buffer_ids: retained_buffers,
        });
        Ok(id)
    }

    pub fn begin_context_release(&mut self, id: ObjectId) -> Result<C, DeviceStateError> {
        let record = self.contexts.get_mut(id).map_err(map_table_error)?;
        if !record.children.is_empty() {
            return Err(DeviceStateError::Busy);
        }
        record.resource.begin_release()
    }

    pub fn restore_context_release(
        &mut self,
        id: ObjectId,
        resource: C,
    ) -> Result<(), RestoreError<C>> {
        restore_resource(&mut self.contexts, id, resource)
    }

    pub fn commit_context_release(&mut self, id: ObjectId) -> Result<(), DeviceStateError> {
        ensure_releasing(self.contexts.get(id).map_err(map_table_error)?)?;
        self.contexts.remove(id).map_err(map_table_error)?;
        Ok(())
    }

    pub fn begin_buffer_release(&mut self, id: ObjectId) -> Result<B, DeviceStateError> {
        let record = self.buffers.get_mut(id).map_err(map_table_error)?;
        if record.in_flight != 0 {
            return Err(DeviceStateError::Busy);
        }
        record.resource.begin_release()
    }

    pub fn restore_buffer_release(
        &mut self,
        id: ObjectId,
        resource: B,
    ) -> Result<(), RestoreError<B>> {
        restore_resource(&mut self.buffers, id, resource)
    }

    pub fn commit_buffer_release(&mut self, id: ObjectId) -> Result<(), DeviceStateError> {
        let context_id = {
            let record = self.buffers.get(id).map_err(map_table_error)?;
            ensure_releasing(record)?;
            record.context_id
        };
        self.buffers.remove(id).map_err(map_table_error)?;
        self.contexts
            .get_mut(context_id)
            .map_err(map_table_error)?
            .children
            .buffers -= 1;
        Ok(())
    }

    pub fn begin_program_release(&mut self, id: ObjectId) -> Result<P, DeviceStateError> {
        let record = self.programs.get_mut(id).map_err(map_table_error)?;
        if record.in_flight != 0 {
            return Err(DeviceStateError::Busy);
        }
        record.resource.begin_release()
    }

    pub fn restore_program_release(
        &mut self,
        id: ObjectId,
        resource: P,
    ) -> Result<(), RestoreError<P>> {
        restore_resource(&mut self.programs, id, resource)
    }

    pub fn commit_program_release(&mut self, id: ObjectId) -> Result<(), DeviceStateError> {
        let context_id = {
            let record = self.programs.get(id).map_err(map_table_error)?;
            ensure_releasing(record)?;
            record.context_id
        };
        self.programs.remove(id).map_err(map_table_error)?;
        self.contexts
            .get_mut(context_id)
            .map_err(map_table_error)?
            .children
            .programs -= 1;
        Ok(())
    }

    pub fn begin_queue_release(&mut self, id: ObjectId) -> Result<Q, DeviceStateError> {
        let record = self.queues.get_mut(id).map_err(map_table_error)?;
        if record.in_flight != 0 {
            return Err(DeviceStateError::Busy);
        }
        record.resource.begin_release()
    }

    pub fn restore_queue_release(
        &mut self,
        id: ObjectId,
        resource: Q,
    ) -> Result<(), RestoreError<Q>> {
        restore_resource(&mut self.queues, id, resource)
    }

    pub fn commit_queue_release(&mut self, id: ObjectId) -> Result<(), DeviceStateError> {
        let context_id = {
            let record = self.queues.get(id).map_err(map_table_error)?;
            ensure_releasing(record)?;
            record.context_id
        };
        self.queues.remove(id).map_err(map_table_error)?;
        self.contexts
            .get_mut(context_id)
            .map_err(map_table_error)?
            .children
            .queues -= 1;
        Ok(())
    }

    pub fn begin_event_release(&mut self, id: ObjectId) -> Result<E, DeviceStateError> {
        self.events
            .get_mut(id)
            .map_err(map_table_error)?
            .resource
            .begin_release()
    }

    pub fn restore_event_release(
        &mut self,
        id: ObjectId,
        resource: E,
    ) -> Result<(), RestoreError<E>> {
        restore_resource(&mut self.events, id, resource)
    }

    pub fn commit_event_release(&mut self, id: ObjectId) -> Result<(), DeviceStateError> {
        {
            let record = self.events.get(id).map_err(map_table_error)?;
            ensure_releasing(record)?;
            self.validate_submission(record.queue_id, record.program_id, &record.buffer_ids)?;
        }
        let record = self.events.remove(id).map_err(map_table_error)?;
        self.decrement_event_references(
            record.context_id,
            record.queue_id,
            record.program_id,
            &record.buffer_ids,
        )
    }

    fn check_child_admission(
        &self,
        context_id: ObjectId,
        child: ChildKind,
        limit: u32,
    ) -> Result<(), DeviceStateError> {
        let context = self.contexts.get(context_id).map_err(map_table_error)?;
        context.resource()?;
        let count = match child {
            ChildKind::Buffer => context.children.buffers,
            ChildKind::Program => context.children.programs,
            ChildKind::Queue => context.children.queues,
        };
        if count >= limit {
            return Err(DeviceStateError::ResourceLimit);
        }
        Ok(())
    }

    fn validate_submission(
        &self,
        queue_id: ObjectId,
        program_id: ObjectId,
        buffer_ids: &[ObjectId],
    ) -> Result<ObjectId, DeviceStateError> {
        let queue = self.queues.get(queue_id).map_err(map_table_error)?;
        queue.resource()?;
        let program = self.programs.get(program_id).map_err(map_table_error)?;
        program.resource()?;
        if program.context_id != queue.context_id {
            return Err(DeviceStateError::ContextMismatch);
        }
        for buffer_id in buffer_ids {
            let buffer = self.buffers.get(*buffer_id).map_err(map_table_error)?;
            buffer.resource()?;
            if buffer.context_id != queue.context_id {
                return Err(DeviceStateError::ContextMismatch);
            }
        }
        Ok(queue.context_id)
    }

    fn check_reference_increments(
        &self,
        queue_id: ObjectId,
        program_id: ObjectId,
        sorted_buffer_ids: &[ObjectId],
    ) -> Result<(), DeviceStateError> {
        self.queues
            .get(queue_id)
            .map_err(map_table_error)?
            .in_flight
            .checked_add(1)
            .ok_or(DeviceStateError::ReferenceCountOverflow)?;
        self.programs
            .get(program_id)
            .map_err(map_table_error)?
            .in_flight
            .checked_add(1)
            .ok_or(DeviceStateError::ReferenceCountOverflow)?;

        let mut index = 0;
        while index < sorted_buffer_ids.len() {
            let id = sorted_buffer_ids[index];
            let mut end = index + 1;
            while end < sorted_buffer_ids.len() && sorted_buffer_ids[end] == id {
                end += 1;
            }
            let count =
                u32::try_from(end - index).map_err(|_| DeviceStateError::ReferenceCountOverflow)?;
            self.buffers
                .get(id)
                .map_err(map_table_error)?
                .in_flight
                .checked_add(count)
                .ok_or(DeviceStateError::ReferenceCountOverflow)?;
            index = end;
        }
        Ok(())
    }

    fn increment_event_references(
        &mut self,
        context_id: ObjectId,
        queue_id: ObjectId,
        program_id: ObjectId,
        buffer_ids: &[ObjectId],
    ) -> Result<(), DeviceStateError> {
        self.queues
            .get_mut(queue_id)
            .map_err(map_table_error)?
            .in_flight += 1;
        self.programs
            .get_mut(program_id)
            .map_err(map_table_error)?
            .in_flight += 1;
        for buffer_id in buffer_ids {
            self.buffers
                .get_mut(*buffer_id)
                .map_err(map_table_error)?
                .in_flight += 1;
        }
        self.contexts
            .get_mut(context_id)
            .map_err(map_table_error)?
            .children
            .events += 1;
        Ok(())
    }

    fn decrement_event_references(
        &mut self,
        context_id: ObjectId,
        queue_id: ObjectId,
        program_id: ObjectId,
        buffer_ids: &[ObjectId],
    ) -> Result<(), DeviceStateError> {
        self.queues
            .get_mut(queue_id)
            .map_err(map_table_error)?
            .in_flight -= 1;
        self.programs
            .get_mut(program_id)
            .map_err(map_table_error)?
            .in_flight -= 1;
        for buffer_id in buffer_ids {
            self.buffers
                .get_mut(*buffer_id)
                .map_err(map_table_error)?
                .in_flight -= 1;
        }
        self.contexts
            .get_mut(context_id)
            .map_err(map_table_error)?
            .children
            .events -= 1;
        Ok(())
    }
}

#[derive(Clone, Copy)]
enum ChildKind {
    Buffer,
    Program,
    Queue,
}

trait ReleasableRecord {
    fn release_state(&self) -> ReleaseState;
}

impl<C> ReleasableRecord for ContextRecord<C> {
    fn release_state(&self) -> ReleaseState {
        self.release_state()
    }
}

impl<B> ReleasableRecord for BufferRecord<B> {
    fn release_state(&self) -> ReleaseState {
        self.release_state()
    }
}

impl<P> ReleasableRecord for ProgramRecord<P> {
    fn release_state(&self) -> ReleaseState {
        self.release_state()
    }
}

impl<Q> ReleasableRecord for QueueRecord<Q> {
    fn release_state(&self) -> ReleaseState {
        self.release_state()
    }
}

impl<E> ReleasableRecord for EventRecord<E> {
    fn release_state(&self) -> ReleaseState {
        self.release_state()
    }
}

fn ensure_releasing(record: &impl ReleasableRecord) -> Result<(), DeviceStateError> {
    if record.release_state() != ReleaseState::Releasing {
        return Err(DeviceStateError::InvalidTransition);
    }
    Ok(())
}

trait ResourceRecord<R> {
    fn resource_mut(&mut self) -> &mut ResourceSlot<R>;
}

impl<C> ResourceRecord<C> for ContextRecord<C> {
    fn resource_mut(&mut self) -> &mut ResourceSlot<C> {
        &mut self.resource
    }
}

impl<B> ResourceRecord<B> for BufferRecord<B> {
    fn resource_mut(&mut self) -> &mut ResourceSlot<B> {
        &mut self.resource
    }
}

impl<P> ResourceRecord<P> for ProgramRecord<P> {
    fn resource_mut(&mut self) -> &mut ResourceSlot<P> {
        &mut self.resource
    }
}

impl<Q> ResourceRecord<Q> for QueueRecord<Q> {
    fn resource_mut(&mut self) -> &mut ResourceSlot<Q> {
        &mut self.resource
    }
}

impl<E> ResourceRecord<E> for EventRecord<E> {
    fn resource_mut(&mut self) -> &mut ResourceSlot<E> {
        &mut self.resource
    }
}

fn restore_resource<R, Record: ResourceRecord<R>>(
    table: &mut ObjectTable<Record>,
    id: ObjectId,
    resource: R,
) -> Result<(), RestoreError<R>> {
    let record = match table.get_mut(id) {
        Ok(record) => record,
        Err(error) => {
            return Err(RestoreError {
                error: map_table_error(error),
                resource,
            });
        }
    };
    record.resource_mut().restore(resource)
}

fn aggregate_slots(contexts: u32, per_context: u32) -> Result<u32, DeviceStateConfigError> {
    contexts
        .checked_mul(per_context)
        .ok_or(DeviceStateConfigError::CountOverflow)
}

fn map_table_error(error: ObjectTableError) -> DeviceStateError {
    match error {
        ObjectTableError::InvalidId => DeviceStateError::InvalidObject,
        ObjectTableError::WrongKind | ObjectTableError::StaleId => DeviceStateError::StaleObject,
        ObjectTableError::Full => DeviceStateError::ResourceLimit,
        ObjectTableError::AllocationFailed => DeviceStateError::OutOfMemory,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::cell::Cell;
    use virtio_accel_core::{BufferDesc, BufferProperties, BufferUsage, MemoryDomain};

    type TestState = DeviceState<u32, u32, u32, u32, u32>;

    fn limits(max_contexts: u32) -> DeviceLimits {
        DeviceLimits {
            max_contexts,
            max_buffers_per_context: 1,
            max_programs_per_context: 1,
            max_queues_per_context: 1,
            max_events_per_context: 1,
            max_bindings_per_submission: 4,
            max_buffer_bytes: 1 << 20,
            max_artifact_bytes: 1 << 20,
        }
    }

    fn state(namespace: u16, max_contexts: u32) -> TestState {
        DeviceState::new(
            ObjectNamespace::new(namespace).unwrap(),
            limits(max_contexts),
        )
        .unwrap()
    }

    fn buffer_desc() -> BufferDesc {
        BufferDesc::new(
            4096,
            64,
            MemoryDomain::Shared,
            BufferUsage::TRANSFER_SOURCE
                | BufferUsage::TRANSFER_DESTINATION
                | BufferUsage::PROGRAM_INPUT,
        )
        .unwrap()
    }

    fn buffer_info(desc: BufferDesc) -> BufferInfo {
        BufferInfo::new(
            desc,
            4096,
            64,
            BufferProperties::HOST_VISIBLE | BufferProperties::DIRECT_BINDING,
        )
        .unwrap()
    }

    fn create_context(state: &mut TestState, resource: u32) -> ObjectId {
        state
            .create_context_with(|| Ok::<_, &'static str>(resource))
            .unwrap()
    }

    #[test]
    fn complete_lifecycle_tracks_children_references_and_release_rollback() {
        let mut state = state(1, 1);
        let context = create_context(&mut state, 10);
        let buffer = state
            .create_buffer_with(context, buffer_desc(), |context, desc| {
                assert_eq!(*context, 10);
                Ok::<_, &'static str>((20, buffer_info(desc)))
            })
            .unwrap();
        *state
            .buffer_record_mut(buffer)
            .unwrap()
            .resource_mut()
            .unwrap() = 21;
        let program = state
            .create_program_with(context, 4096, 8192, |context| {
                assert_eq!(*context, 10);
                Ok::<_, &'static str>(30)
            })
            .unwrap();
        let queue = state
            .create_queue_with(context, |context| {
                assert_eq!(*context, 10);
                Ok::<_, &'static str>(40)
            })
            .unwrap();
        let event = state
            .create_event_with(queue, program, &[buffer, buffer], |resources| {
                assert_eq!(resources.context_id(), context);
                assert_eq!(*resources.queue(), 40);
                assert_eq!(*resources.program(), 30);
                assert_eq!(*resources.buffer(0).unwrap(), 21);
                assert_eq!(*resources.buffer(1).unwrap(), 21);
                Ok::<_, &'static str>(50)
            })
            .unwrap();

        assert_eq!(
            state.context_record(context).unwrap().children(),
            ChildCounts {
                buffers: 1,
                programs: 1,
                queues: 1,
                events: 1,
            }
        );
        assert_eq!(state.buffer_record(buffer).unwrap().in_flight(), 2);
        assert_eq!(state.program_record(program).unwrap().in_flight(), 1);
        assert_eq!(state.queue_record(queue).unwrap().in_flight(), 1);
        assert_eq!(
            state.begin_context_release(context),
            Err(DeviceStateError::Busy)
        );
        assert_eq!(
            state.begin_buffer_release(buffer),
            Err(DeviceStateError::Busy)
        );
        assert_eq!(
            state.begin_program_release(program),
            Err(DeviceStateError::Busy)
        );
        assert_eq!(
            state.begin_queue_release(queue),
            Err(DeviceStateError::Busy)
        );

        let event_resource = state.begin_event_release(event).unwrap();
        assert_eq!(event_resource, 50);
        assert_eq!(
            state.event_record(event).unwrap().release_state(),
            ReleaseState::Releasing
        );
        state.restore_event_release(event, event_resource).unwrap();
        assert_eq!(*state.event_record(event).unwrap().resource().unwrap(), 50);
        let event_resource = state.begin_event_release(event).unwrap();
        assert_eq!(event_resource, 50);
        state.commit_event_release(event).unwrap();
        assert!(matches!(
            state.event_record(event),
            Err(DeviceStateError::StaleObject)
        ));
        assert_eq!(state.buffer_record(buffer).unwrap().in_flight(), 0);
        assert_eq!(state.program_record(program).unwrap().in_flight(), 0);
        assert_eq!(state.queue_record(queue).unwrap().in_flight(), 0);

        let buffer_resource = state.begin_buffer_release(buffer).unwrap();
        state
            .restore_buffer_release(buffer, buffer_resource)
            .unwrap();
        let buffer_resource = state.begin_buffer_release(buffer).unwrap();
        assert_eq!(buffer_resource, 21);
        state.commit_buffer_release(buffer).unwrap();

        let program_resource = state.begin_program_release(program).unwrap();
        state
            .restore_program_release(program, program_resource)
            .unwrap();
        let program_resource = state.begin_program_release(program).unwrap();
        assert_eq!(program_resource, 30);
        state.commit_program_release(program).unwrap();

        let queue_resource = state.begin_queue_release(queue).unwrap();
        state.restore_queue_release(queue, queue_resource).unwrap();
        let queue_resource = state.begin_queue_release(queue).unwrap();
        assert_eq!(queue_resource, 40);
        state.commit_queue_release(queue).unwrap();

        assert!(state.context_record(context).unwrap().children().is_empty());
        let context_resource = state.begin_context_release(context).unwrap();
        state
            .restore_context_release(context, context_resource)
            .unwrap();
        let context_resource = state.begin_context_release(context).unwrap();
        assert_eq!(context_resource, 10);
        state.commit_context_release(context).unwrap();
        assert!(matches!(
            state.context_record(context),
            Err(DeviceStateError::StaleObject)
        ));
        assert_eq!(state.context_count(), 0);
        assert_eq!(state.buffer_count(), 0);
        assert_eq!(state.program_count(), 0);
        assert_eq!(state.queue_count(), 0);
        assert_eq!(state.event_count(), 0);
    }

    #[test]
    fn quota_exhaustion_never_invokes_provider_callbacks() {
        let mut state = state(1, 1);
        let calls = Cell::new(0_u32);
        let context = state
            .create_context_with(|| {
                calls.set(calls.get() + 1);
                Ok::<_, &'static str>(10)
            })
            .unwrap();
        assert!(matches!(
            state.create_context_with(|| {
                calls.set(calls.get() + 1);
                Ok::<_, &'static str>(11)
            }),
            Err(CreateError::State(DeviceStateError::ResourceLimit))
        ));

        let buffer = state
            .create_buffer_with(context, buffer_desc(), |_, desc| {
                calls.set(calls.get() + 1);
                Ok::<_, &'static str>((20, buffer_info(desc)))
            })
            .unwrap();
        assert!(matches!(
            state.create_buffer_with(context, buffer_desc(), |_, desc| {
                calls.set(calls.get() + 1);
                Ok::<_, &'static str>((21, buffer_info(desc)))
            }),
            Err(CreateError::State(DeviceStateError::ResourceLimit))
        ));

        let program = state
            .create_program_with(context, 1, 1, |_| {
                calls.set(calls.get() + 1);
                Ok::<_, &'static str>(30)
            })
            .unwrap();
        assert!(matches!(
            state.create_program_with(context, 1, 1, |_| {
                calls.set(calls.get() + 1);
                Ok::<_, &'static str>(31)
            }),
            Err(CreateError::State(DeviceStateError::ResourceLimit))
        ));

        let queue = state
            .create_queue_with(context, |_| {
                calls.set(calls.get() + 1);
                Ok::<_, &'static str>(40)
            })
            .unwrap();
        assert!(matches!(
            state.create_queue_with(context, |_| {
                calls.set(calls.get() + 1);
                Ok::<_, &'static str>(41)
            }),
            Err(CreateError::State(DeviceStateError::ResourceLimit))
        ));

        state
            .create_event_with(queue, program, &[buffer], |_| {
                calls.set(calls.get() + 1);
                Ok::<_, &'static str>(50)
            })
            .unwrap();
        assert!(matches!(
            state.create_event_with(queue, program, &[buffer], |_| {
                calls.set(calls.get() + 1);
                Ok::<_, &'static str>(51)
            }),
            Err(CreateError::State(DeviceStateError::ResourceLimit))
        ));
        assert_eq!(calls.get(), 5);
        assert_eq!(state.context_count(), 1);
        assert_eq!(state.buffer_count(), 1);
        assert_eq!(state.program_count(), 1);
        assert_eq!(state.queue_count(), 1);
        assert_eq!(state.event_count(), 1);
    }

    #[test]
    fn provider_rejection_rolls_back_every_creation_path() {
        let mut state = state(1, 1);
        assert!(matches!(
            state.create_context_with(|| Err::<u32, _>("context")),
            Err(CreateError::Provider("context"))
        ));
        assert_eq!(state.context_count(), 0);

        let context = create_context(&mut state, 10);
        assert!(matches!(
            state.create_buffer_with(context, buffer_desc(), |_, _| {
                Err::<(u32, BufferInfo), _>("buffer")
            }),
            Err(CreateError::Provider("buffer"))
        ));
        assert!(matches!(
            state.create_program_with(context, 1, 1, |_| Err::<u32, _>("program")),
            Err(CreateError::Provider("program"))
        ));
        assert!(matches!(
            state.create_queue_with(context, |_| Err::<u32, _>("queue")),
            Err(CreateError::Provider("queue"))
        ));
        assert_eq!(
            state.context_record(context).unwrap().children(),
            ChildCounts::default()
        );

        let buffer = state
            .create_buffer_with(context, buffer_desc(), |_, desc| {
                Ok::<_, &'static str>((20, buffer_info(desc)))
            })
            .unwrap();
        let program = state
            .create_program_with(context, 1, 1, |_| Ok::<_, &'static str>(30))
            .unwrap();
        let queue = state
            .create_queue_with(context, |_| Ok::<_, &'static str>(40))
            .unwrap();
        assert!(matches!(
            state.create_event_with(queue, program, &[buffer], |_| { Err::<u32, _>("event") }),
            Err(CreateError::Provider("event"))
        ));
        assert_eq!(state.event_count(), 0);
        assert_eq!(state.buffer_record(buffer).unwrap().in_flight(), 0);
        assert_eq!(state.program_record(program).unwrap().in_flight(), 0);
        assert_eq!(state.queue_record(queue).unwrap().in_flight(), 0);
        assert_eq!(state.context_record(context).unwrap().children().events, 0);
    }

    #[test]
    fn wrong_kind_cross_context_and_cross_device_ids_fail_before_provider_use() {
        let mut first = state(1, 2);
        let first_context = create_context(&mut first, 10);
        let second_context = create_context(&mut first, 11);
        let buffer = first
            .create_buffer_with(first_context, buffer_desc(), |_, desc| {
                Ok::<_, &'static str>((20, buffer_info(desc)))
            })
            .unwrap();
        let program = first
            .create_program_with(second_context, 1, 1, |_| Ok::<_, &'static str>(30))
            .unwrap();
        let queue = first
            .create_queue_with(first_context, |_| Ok::<_, &'static str>(40))
            .unwrap();
        let called = Cell::new(false);
        assert!(matches!(
            first.create_event_with(queue, program, &[buffer], |_| {
                called.set(true);
                Ok::<_, &'static str>(50)
            }),
            Err(CreateError::State(DeviceStateError::ContextMismatch))
        ));
        assert!(!called.get());
        assert!(matches!(
            first.buffer_record(queue),
            Err(DeviceStateError::StaleObject)
        ));

        let second = state(2, 2);
        assert!(matches!(
            second.context_record(first_context),
            Err(DeviceStateError::StaleObject)
        ));
    }

    #[test]
    fn invalid_limits_are_rejected_before_tables_exist() {
        let namespace = ObjectNamespace::new(1).unwrap();
        let mut invalid = limits(1);
        invalid.max_bindings_per_submission = 0;
        assert!(matches!(
            TestState::new(namespace, invalid),
            Err(DeviceStateConfigError::BindingLimit)
        ));

        let mut invalid = limits(u32::MAX);
        invalid.max_buffers_per_context = 2;
        assert!(matches!(
            TestState::new(namespace, invalid),
            Err(DeviceStateConfigError::CountOverflow)
        ));

        let mut invalid = limits(1);
        invalid.max_events_per_context = u32::MAX;
        invalid.max_bindings_per_submission = HARD_MAX_BINDINGS;
        assert!(matches!(
            TestState::new(namespace, invalid),
            Err(DeviceStateConfigError::ReferenceCountOverflow)
        ));
    }

    #[test]
    fn byte_limits_and_resident_charges_have_distinct_semantics() {
        let mut state = state(1, 1);
        let context = create_context(&mut state, 10);
        let calls = Cell::new(0_u32);
        let oversized_buffer = BufferDesc::new(
            state.limits().max_buffer_bytes + 1,
            1,
            MemoryDomain::Host,
            BufferUsage::TRANSFER_SOURCE,
        )
        .unwrap();

        assert!(matches!(
            state.create_buffer_with(context, oversized_buffer, |_, desc| {
                calls.set(calls.get() + 1);
                Ok::<_, &'static str>((20, buffer_info(desc)))
            }),
            Err(CreateError::State(DeviceStateError::ResourceLimit))
        ));
        assert!(matches!(
            state.create_program_with(context, state.limits().max_artifact_bytes + 1, 1, |_| {
                calls.set(calls.get() + 1);
                Ok::<_, &'static str>(30)
            }),
            Err(CreateError::State(DeviceStateError::ResourceLimit))
        ));
        let resident_bytes = state.limits().max_artifact_bytes + 1;
        let program = state
            .create_program_with(context, 1, resident_bytes, |_| {
                calls.set(calls.get() + 1);
                Ok::<_, &'static str>(30)
            })
            .unwrap();
        assert_eq!(calls.get(), 1);
        assert_eq!(state.buffer_count(), 0);
        assert_eq!(state.program_count(), 1);
        assert_eq!(
            state.program_record(program).unwrap().resident_bytes(),
            resident_bytes
        );
        assert_eq!(
            state.context_record(context).unwrap().children(),
            ChildCounts {
                programs: 1,
                ..ChildCounts::default()
            }
        );
    }
}
