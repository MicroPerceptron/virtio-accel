//! Deterministic fault injection and ownership auditing for accelerator backends.
//!
//! [`FaultAccelerator`] wraps any [`Accelerator`] with an explicit script keyed by method and
//! one-based call occurrence. The wrapper records calls, release attempts, and provider ownership
//! without requiring host synchronization. It is test infrastructure, not part of the normative
//! accelerator ABI.

use std::cell::RefCell;
use std::rc::Rc;
use std::vec::Vec;
use virtio_accel_core::{
    Accelerator, AllocatedBuffer, ArtifactRef, BackendError, BindingRef, BufferDesc, ByteSink,
    ByteSource, ContextDesc, DeviceInfo, EventState, QueueDesc, ReleaseFailure, SubmitFailure,
    Timeout,
};

const FAULT_POINT_COUNT: usize = 15;

/// One fallible method in the [`Accelerator`] contract.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FaultPoint {
    DeviceInfo,
    CreateContext,
    DestroyContext,
    AllocateBuffer,
    WriteBuffer,
    ReadBuffer,
    FreeBuffer,
    LoadProgram,
    UnloadProgram,
    CreateQueue,
    DestroyQueue,
    Submit,
    PollEvent,
    CancelEvent,
    DestroyEvent,
}

impl FaultPoint {
    const fn index(self) -> usize {
        match self {
            Self::DeviceInfo => 0,
            Self::CreateContext => 1,
            Self::DestroyContext => 2,
            Self::AllocateBuffer => 3,
            Self::WriteBuffer => 4,
            Self::ReadBuffer => 5,
            Self::FreeBuffer => 6,
            Self::LoadProgram => 7,
            Self::UnloadProgram => 8,
            Self::CreateQueue => 9,
            Self::DestroyQueue => 10,
            Self::Submit => 11,
            Self::PollEvent => 12,
            Self::CancelEvent => 13,
            Self::DestroyEvent => 14,
        }
    }

    const fn allows(self, action: FaultAction) -> bool {
        match self {
            Self::DestroyContext
            | Self::FreeBuffer
            | Self::UnloadProgram
            | Self::DestroyQueue
            | Self::DestroyEvent
            | Self::Submit => matches!(
                action,
                FaultAction::Rejected(_) | FaultAction::Indeterminate(_)
            ),
            Self::PollEvent => match action {
                FaultAction::ErrorBefore(_) | FaultAction::ErrorAfter(_) => true,
                FaultAction::Completion(state) => !matches!(state, EventState::Pending),
                FaultAction::Rejected(_) | FaultAction::Indeterminate(_) => false,
            },
            _ => matches!(
                action,
                FaultAction::ErrorBefore(_) | FaultAction::ErrorAfter(_)
            ),
        }
    }
}

/// Fault behavior applied to one scripted method occurrence.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FaultAction {
    /// Return an error without invoking the wrapped method.
    ErrorBefore(BackendError),
    /// Invoke the wrapped method, then replace a successful result with an error.
    ///
    /// Resource creation methods release the newly created resource before returning this error.
    ErrorAfter(BackendError),
    /// Reject a release or submission before ownership changes.
    Rejected(BackendError),
    /// Make release or submission ownership indeterminate after the acceptance boundary.
    Indeterminate(BackendError),
    /// Publish and retain one synthetic terminal completion state.
    Completion(EventState),
}

/// One fault keyed by method and one-based call occurrence.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FaultStep {
    pub point: FaultPoint,
    pub occurrence: u32,
    pub action: FaultAction,
}

impl FaultStep {
    pub const fn new(point: FaultPoint, occurrence: u32, action: FaultAction) -> Self {
        Self {
            point,
            occurrence,
            action,
        }
    }
}

/// Invalid deterministic script.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FaultScriptError {
    ZeroOccurrence {
        point: FaultPoint,
    },
    DuplicateStep {
        point: FaultPoint,
        occurrence: u32,
    },
    IncompatibleAction {
        point: FaultPoint,
        action: FaultAction,
    },
}

/// Validated explicit fault schedule.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FaultScript {
    steps: Vec<FaultStep>,
}

impl FaultScript {
    pub fn new(steps: impl IntoIterator<Item = FaultStep>) -> Result<Self, FaultScriptError> {
        let mut validated = Vec::new();
        for step in steps {
            if step.occurrence == 0 {
                return Err(FaultScriptError::ZeroOccurrence { point: step.point });
            }
            if !step.point.allows(step.action) {
                return Err(FaultScriptError::IncompatibleAction {
                    point: step.point,
                    action: step.action,
                });
            }
            if validated.iter().any(|prior: &FaultStep| {
                prior.point == step.point && prior.occurrence == step.occurrence
            }) {
                return Err(FaultScriptError::DuplicateStep {
                    point: step.point,
                    occurrence: step.occurrence,
                });
            }
            validated.push(step);
        }
        Ok(Self { steps: validated })
    }
}

/// Stable harness-owned identity for one provider resource.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct ResourceId(u64);

impl ResourceId {
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Provider resource category tracked by the harness.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResourceKind {
    Context,
    Buffer,
    Program,
    Queue,
    Event,
}

/// Last known provider ownership state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResourceState {
    Live,
    Released,
    Indeterminate,
    Discarded,
}

/// One recorded resource and its last known state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResourceRecord {
    pub id: ResourceId,
    pub kind: ResourceKind,
    pub state: ResourceState,
}

/// One backend method call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CallRecord {
    pub sequence: u64,
    pub point: FaultPoint,
    pub occurrence: u32,
    pub injected: Option<FaultAction>,
}

/// Result of one provider release attempt.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReleaseOutcome {
    Success,
    Rejected(BackendError),
    Indeterminate(BackendError),
}

/// One release attempt, including internal rollback after an injected post-create fault.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReleaseRecord {
    pub resource_id: ResourceId,
    pub kind: ResourceKind,
    pub attempt: u32,
    pub outcome: ReleaseOutcome,
    pub rollback: bool,
}

/// Ownership-contract violation detected by the harness.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OwnershipViolation {
    UnknownResource {
        id: ResourceId,
        expected: ResourceKind,
    },
    KindMismatch {
        id: ResourceId,
        expected: ResourceKind,
        actual: ResourceKind,
    },
    ResourceNotLive {
        id: ResourceId,
        kind: ResourceKind,
        state: ResourceState,
    },
    PrematureRelease {
        id: ResourceId,
        kind: ResourceKind,
        retained_by: ResourceId,
    },
    RollbackFailed {
        id: ResourceId,
        kind: ResourceKind,
        error: BackendError,
    },
}

/// Resource counts for one ownership state.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ResourceTally {
    pub contexts: u32,
    pub buffers: u32,
    pub programs: u32,
    pub queues: u32,
    pub events: u32,
}

impl ResourceTally {
    fn increment(&mut self, kind: ResourceKind) {
        let count = match kind {
            ResourceKind::Context => &mut self.contexts,
            ResourceKind::Buffer => &mut self.buffers,
            ResourceKind::Program => &mut self.programs,
            ResourceKind::Queue => &mut self.queues,
            ResourceKind::Event => &mut self.events,
        };
        *count = count.saturating_add(1);
    }
}

/// Point-in-time copy of fault and ownership state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FaultSnapshot {
    pub calls: Vec<CallRecord>,
    pub releases: Vec<ReleaseRecord>,
    pub resources: Vec<ResourceRecord>,
    pub pending_faults: Vec<FaultStep>,
    pub violations: Vec<OwnershipViolation>,
}

impl FaultSnapshot {
    pub fn resources_in(&self, state: ResourceState) -> ResourceTally {
        let mut tally = ResourceTally::default();
        for resource in self.resources.iter().filter(|item| item.state == state) {
            tally.increment(resource.kind);
        }
        tally
    }

    pub fn is_clean(&self) -> bool {
        self.pending_faults.is_empty()
            && self.violations.is_empty()
            && self.resources_in(ResourceState::Live) == ResourceTally::default()
            && self.resources_in(ResourceState::Indeterminate) == ResourceTally::default()
    }
}

#[derive(Debug)]
struct FaultState {
    pending_faults: Vec<FaultStep>,
    occurrences: [u32; FAULT_POINT_COUNT],
    next_sequence: u64,
    next_resource_id: u64,
    calls: Vec<CallRecord>,
    releases: Vec<ReleaseRecord>,
    resources: Vec<ResourceRecord>,
    parents: Vec<(ResourceId, ResourceId)>,
    event_dependencies: Vec<(ResourceId, Vec<ResourceId>)>,
    event_completions: Vec<(ResourceId, EventState)>,
    violations: Vec<OwnershipViolation>,
}

impl FaultState {
    fn new(script: FaultScript) -> Self {
        Self {
            pending_faults: script.steps,
            occurrences: [0; FAULT_POINT_COUNT],
            next_sequence: 1,
            next_resource_id: 1,
            calls: Vec::new(),
            releases: Vec::new(),
            resources: Vec::new(),
            parents: Vec::new(),
            event_dependencies: Vec::new(),
            event_completions: Vec::new(),
            violations: Vec::new(),
        }
    }
}

/// Cloneable control and audit handle for a [`FaultAccelerator`].
#[derive(Clone, Debug)]
pub struct FaultControl {
    state: Rc<RefCell<FaultState>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ReleaseValidation {
    Ready,
    Invalid,
    Premature,
}

impl FaultControl {
    fn new(script: FaultScript) -> Self {
        Self {
            state: Rc::new(RefCell::new(FaultState::new(script))),
        }
    }

    fn begin_call(&self, point: FaultPoint) -> Option<FaultAction> {
        let mut state = self.state.borrow_mut();
        let occurrence = state.occurrences[point.index()]
            .checked_add(1)
            .expect("fault call occurrence overflow");
        state.occurrences[point.index()] = occurrence;
        let injected = state
            .pending_faults
            .iter()
            .position(|step| step.point == point && step.occurrence == occurrence)
            .map(|index| state.pending_faults.remove(index).action);
        let sequence = state.next_sequence;
        state.next_sequence = state
            .next_sequence
            .checked_add(1)
            .expect("fault call sequence overflow");
        state.calls.push(CallRecord {
            sequence,
            point,
            occurrence,
            injected,
        });
        injected
    }

    fn register(&self, kind: ResourceKind) -> ResourceId {
        let mut state = self.state.borrow_mut();
        let id = ResourceId(state.next_resource_id);
        state.next_resource_id = state
            .next_resource_id
            .checked_add(1)
            .expect("fault resource identity overflow");
        state.resources.push(ResourceRecord {
            id,
            kind,
            state: ResourceState::Live,
        });
        id
    }

    fn register_child(&self, kind: ResourceKind, parent: ResourceId) -> ResourceId {
        let id = self.register(kind);
        self.state.borrow_mut().parents.push((id, parent));
        id
    }

    fn register_event(&self, dependencies: Vec<ResourceId>) -> ResourceId {
        let id = self.register(ResourceKind::Event);
        self.state
            .borrow_mut()
            .event_dependencies
            .push((id, dependencies));
        id
    }

    fn set_completion(&self, id: ResourceId, completion: EventState) {
        let mut state = self.state.borrow_mut();
        if let Some((_, prior)) = state
            .event_completions
            .iter_mut()
            .find(|(event, _)| *event == id)
        {
            *prior = completion;
        } else {
            state.event_completions.push((id, completion));
        }
    }

    fn completion(&self, id: ResourceId) -> Option<EventState> {
        self.state
            .borrow()
            .event_completions
            .iter()
            .find_map(|(event, state)| (*event == id).then_some(*state))
    }

    fn validate(&self, id: ResourceId, expected: ResourceKind) -> bool {
        let mut state = self.state.borrow_mut();
        let Some(index) = state.resources.iter().position(|item| item.id == id) else {
            state
                .violations
                .push(OwnershipViolation::UnknownResource { id, expected });
            return false;
        };
        let resource = state.resources[index];
        if resource.kind != expected {
            state.violations.push(OwnershipViolation::KindMismatch {
                id,
                expected,
                actual: resource.kind,
            });
            return false;
        }
        if resource.state != ResourceState::Live {
            state.violations.push(OwnershipViolation::ResourceNotLive {
                id,
                kind: expected,
                state: resource.state,
            });
            return false;
        }
        true
    }

    fn validate_release(&self, id: ResourceId, expected: ResourceKind) -> ReleaseValidation {
        if !self.validate(id, expected) {
            return ReleaseValidation::Invalid;
        }

        let mut state = self.state.borrow_mut();
        let retained_by = match expected {
            ResourceKind::Context => state.parents.iter().find_map(|(child, parent)| {
                (*parent == id
                    && state.resources.iter().any(|resource| {
                        resource.id == *child
                            && matches!(
                                resource.state,
                                ResourceState::Live | ResourceState::Indeterminate
                            )
                    }))
                .then_some(*child)
            }),
            ResourceKind::Buffer | ResourceKind::Program | ResourceKind::Queue => state
                .event_dependencies
                .iter()
                .find_map(|(event, dependencies)| {
                    (dependencies.contains(&id)
                        && state.resources.iter().any(|resource| {
                            resource.id == *event
                                && matches!(
                                    resource.state,
                                    ResourceState::Live | ResourceState::Indeterminate
                                )
                        }))
                    .then_some(*event)
                }),
            ResourceKind::Event => None,
        };
        match retained_by {
            Some(retained_by) => {
                state.violations.push(OwnershipViolation::PrematureRelease {
                    id,
                    kind: expected,
                    retained_by,
                });
                ReleaseValidation::Premature
            }
            None => ReleaseValidation::Ready,
        }
    }

    fn finish_release(
        &self,
        id: ResourceId,
        kind: ResourceKind,
        outcome: ReleaseOutcome,
        rollback: bool,
    ) {
        let mut state = self.state.borrow_mut();
        let prior_attempts = state
            .releases
            .iter()
            .filter(|record| record.resource_id == id)
            .count();
        let attempt = u32::try_from(prior_attempts)
            .expect("fault release attempt overflow")
            .checked_add(1)
            .expect("fault release attempt overflow");
        state.releases.push(ReleaseRecord {
            resource_id: id,
            kind,
            attempt,
            outcome,
            rollback,
        });
        if let Some(resource) = state.resources.iter_mut().find(|item| item.id == id) {
            resource.state = match outcome {
                ReleaseOutcome::Success => ResourceState::Released,
                ReleaseOutcome::Rejected(_) => ResourceState::Live,
                ReleaseOutcome::Indeterminate(_) => ResourceState::Indeterminate,
            };
        }
    }

    fn finish_rollback(&self, id: ResourceId, kind: ResourceKind, outcome: ReleaseOutcome) {
        self.finish_release(id, kind, outcome, true);
        if let ReleaseOutcome::Rejected(error) | ReleaseOutcome::Indeterminate(error) = outcome {
            let mut state = self.state.borrow_mut();
            if let Some(resource) = state.resources.iter_mut().find(|item| item.id == id) {
                resource.state = ResourceState::Indeterminate;
            }
            state
                .violations
                .push(OwnershipViolation::RollbackFailed { id, kind, error });
        }
    }

    /// Mark all provider state discarded after the caller discards or replaces the backend instance.
    pub fn discard_all(&self) {
        for resource in &mut self.state.borrow_mut().resources {
            if matches!(
                resource.state,
                ResourceState::Live | ResourceState::Indeterminate
            ) {
                resource.state = ResourceState::Discarded;
            }
        }
    }

    pub fn snapshot(&self) -> FaultSnapshot {
        let state = self.state.borrow();
        FaultSnapshot {
            calls: state.calls.clone(),
            releases: state.releases.clone(),
            resources: state.resources.clone(),
            pending_faults: state.pending_faults.clone(),
            violations: state.violations.clone(),
        }
    }
}

#[derive(Clone, Debug)]
struct Tracked<T> {
    id: ResourceId,
    inner: T,
}

macro_rules! tracked_handle {
    ($name:ident) => {
        #[derive(Clone, Debug)]
        pub struct $name<T>(Tracked<T>);

        impl<T> $name<T> {
            pub const fn id(&self) -> ResourceId {
                self.0.id
            }

            pub const fn inner(&self) -> &T {
                &self.0.inner
            }

            pub fn inner_mut(&mut self) -> &mut T {
                &mut self.0.inner
            }
        }
    };
}

tracked_handle!(FaultContext);
tracked_handle!(FaultBuffer);
tracked_handle!(FaultProgram);
tracked_handle!(FaultQueue);
tracked_handle!(FaultEvent);

/// Accelerator wrapper driven by a validated deterministic fault script.
pub struct FaultAccelerator<A> {
    inner: A,
    control: FaultControl,
}

impl<A> FaultAccelerator<A> {
    pub fn new(inner: A, script: FaultScript) -> Self {
        Self {
            inner,
            control: FaultControl::new(script),
        }
    }

    pub const fn inner(&self) -> &A {
        &self.inner
    }

    pub fn control(&self) -> FaultControl {
        self.control.clone()
    }
}

const fn before_error(action: Option<FaultAction>) -> Option<BackendError> {
    match action {
        Some(FaultAction::ErrorBefore(error)) => Some(error),
        _ => None,
    }
}

const fn after_error(action: Option<FaultAction>) -> Option<BackendError> {
    match action {
        Some(FaultAction::ErrorAfter(error)) => Some(error),
        _ => None,
    }
}

impl<A: Accelerator> FaultAccelerator<A> {
    fn rollback_resource<R>(
        &self,
        id: ResourceId,
        kind: ResourceKind,
        resource: R,
        release: impl FnOnce(&A, R) -> Result<(), ReleaseFailure<R>>,
    ) {
        let outcome = match release(&self.inner, resource) {
            Ok(()) => ReleaseOutcome::Success,
            Err(ReleaseFailure::Rejected { error, .. }) => ReleaseOutcome::Rejected(error),
            Err(ReleaseFailure::Indeterminate { error }) => ReleaseOutcome::Indeterminate(error),
        };
        self.control.finish_rollback(id, kind, outcome);
    }

    fn release_resource<R, W>(
        &self,
        point: FaultPoint,
        kind: ResourceKind,
        tracked: Tracked<R>,
        wrap: fn(Tracked<R>) -> W,
        release: impl FnOnce(&A, R) -> Result<(), ReleaseFailure<R>>,
    ) -> Result<(), ReleaseFailure<W>> {
        let action = self.control.begin_call(point);
        let id = tracked.id;
        match self.control.validate_release(id, kind) {
            ReleaseValidation::Ready => {}
            ReleaseValidation::Invalid => {
                return Err(ReleaseFailure::Indeterminate {
                    error: BackendError::DeviceLost,
                });
            }
            ReleaseValidation::Premature => {
                self.control.finish_release(
                    id,
                    kind,
                    ReleaseOutcome::Indeterminate(BackendError::DeviceLost),
                    false,
                );
                return Err(ReleaseFailure::Indeterminate {
                    error: BackendError::DeviceLost,
                });
            }
        }
        match action {
            Some(FaultAction::Rejected(error)) => {
                self.control
                    .finish_release(id, kind, ReleaseOutcome::Rejected(error), false);
                Err(ReleaseFailure::Rejected {
                    error,
                    resource: wrap(tracked),
                })
            }
            Some(FaultAction::Indeterminate(error)) => {
                self.control
                    .finish_release(id, kind, ReleaseOutcome::Indeterminate(error), false);
                Err(ReleaseFailure::Indeterminate { error })
            }
            None => {
                let Tracked { id, inner } = tracked;
                match release(&self.inner, inner) {
                    Ok(()) => {
                        self.control
                            .finish_release(id, kind, ReleaseOutcome::Success, false);
                        Ok(())
                    }
                    Err(ReleaseFailure::Rejected { error, resource }) => {
                        self.control.finish_release(
                            id,
                            kind,
                            ReleaseOutcome::Rejected(error),
                            false,
                        );
                        Err(ReleaseFailure::Rejected {
                            error,
                            resource: wrap(Tracked {
                                id,
                                inner: resource,
                            }),
                        })
                    }
                    Err(ReleaseFailure::Indeterminate { error }) => {
                        self.control.finish_release(
                            id,
                            kind,
                            ReleaseOutcome::Indeterminate(error),
                            false,
                        );
                        Err(ReleaseFailure::Indeterminate { error })
                    }
                }
            }
            Some(
                FaultAction::ErrorBefore(_)
                | FaultAction::ErrorAfter(_)
                | FaultAction::Completion(_),
            ) => {
                unreachable!("validated release scripts contain only ownership outcomes")
            }
        }
    }

    fn validate_resource<T>(
        &self,
        resource: &Tracked<T>,
        kind: ResourceKind,
    ) -> Result<(), BackendError> {
        if self.control.validate(resource.id, kind) {
            Ok(())
        } else {
            Err(BackendError::DeviceLost)
        }
    }
}

impl<A: Accelerator> Accelerator for FaultAccelerator<A> {
    type Context = FaultContext<A::Context>;
    type Buffer = FaultBuffer<A::Buffer>;
    type Program = FaultProgram<A::Program>;
    type Queue = FaultQueue<A::Queue>;
    type Event = FaultEvent<A::Event>;

    fn device_info(&self) -> Result<DeviceInfo, BackendError> {
        let action = self.control.begin_call(FaultPoint::DeviceInfo);
        if let Some(error) = before_error(action) {
            return Err(error);
        }
        let result = self.inner.device_info();
        match (result, after_error(action)) {
            (Ok(_), Some(error)) => Err(error),
            (result, _) => result,
        }
    }

    fn create_context(&self, desc: ContextDesc) -> Result<Self::Context, BackendError> {
        let action = self.control.begin_call(FaultPoint::CreateContext);
        if let Some(error) = before_error(action) {
            return Err(error);
        }
        let context = self.inner.create_context(desc)?;
        let id = self.control.register(ResourceKind::Context);
        if let Some(error) = after_error(action) {
            self.rollback_resource(id, ResourceKind::Context, context, A::destroy_context);
            return Err(error);
        }
        Ok(FaultContext(Tracked { id, inner: context }))
    }

    fn destroy_context(&self, context: Self::Context) -> Result<(), ReleaseFailure<Self::Context>> {
        self.release_resource(
            FaultPoint::DestroyContext,
            ResourceKind::Context,
            context.0,
            FaultContext,
            A::destroy_context,
        )
    }

    fn allocate_buffer(
        &self,
        context: &Self::Context,
        desc: BufferDesc,
    ) -> Result<AllocatedBuffer<Self::Buffer>, BackendError> {
        let action = self.control.begin_call(FaultPoint::AllocateBuffer);
        self.validate_resource(&context.0, ResourceKind::Context)?;
        if let Some(error) = before_error(action) {
            return Err(error);
        }
        let allocation = self.inner.allocate_buffer(context.inner(), desc)?;
        let (buffer, info) = allocation.into_parts();
        let id = self
            .control
            .register_child(ResourceKind::Buffer, context.id());
        if let Some(error) = after_error(action) {
            self.rollback_resource(id, ResourceKind::Buffer, buffer, A::free_buffer);
            return Err(error);
        }
        Ok(AllocatedBuffer::new(
            FaultBuffer(Tracked { id, inner: buffer }),
            info,
        ))
    }

    fn write_buffer(
        &self,
        buffer: &mut Self::Buffer,
        offset: u64,
        data: &dyn ByteSource,
    ) -> Result<(), BackendError> {
        let action = self.control.begin_call(FaultPoint::WriteBuffer);
        self.validate_resource(&buffer.0, ResourceKind::Buffer)?;
        if let Some(error) = before_error(action) {
            return Err(error);
        }
        let result = self.inner.write_buffer(buffer.inner_mut(), offset, data);
        match (result, after_error(action)) {
            (Ok(()), Some(error)) => Err(error),
            (result, _) => result,
        }
    }

    fn read_buffer(
        &self,
        buffer: &Self::Buffer,
        offset: u64,
        data: &mut dyn ByteSink,
    ) -> Result<(), BackendError> {
        let action = self.control.begin_call(FaultPoint::ReadBuffer);
        self.validate_resource(&buffer.0, ResourceKind::Buffer)?;
        if let Some(error) = before_error(action) {
            return Err(error);
        }
        let result = self.inner.read_buffer(buffer.inner(), offset, data);
        match (result, after_error(action)) {
            (Ok(()), Some(error)) => Err(error),
            (result, _) => result,
        }
    }

    fn free_buffer(&self, buffer: Self::Buffer) -> Result<(), ReleaseFailure<Self::Buffer>> {
        self.release_resource(
            FaultPoint::FreeBuffer,
            ResourceKind::Buffer,
            buffer.0,
            FaultBuffer,
            A::free_buffer,
        )
    }

    fn load_program(
        &self,
        context: &Self::Context,
        artifact: ArtifactRef<'_>,
    ) -> Result<Self::Program, BackendError> {
        let action = self.control.begin_call(FaultPoint::LoadProgram);
        self.validate_resource(&context.0, ResourceKind::Context)?;
        if let Some(error) = before_error(action) {
            return Err(error);
        }
        let program = self.inner.load_program(context.inner(), artifact)?;
        let id = self
            .control
            .register_child(ResourceKind::Program, context.id());
        if let Some(error) = after_error(action) {
            self.rollback_resource(id, ResourceKind::Program, program, A::unload_program);
            return Err(error);
        }
        Ok(FaultProgram(Tracked { id, inner: program }))
    }

    fn unload_program(&self, program: Self::Program) -> Result<(), ReleaseFailure<Self::Program>> {
        self.release_resource(
            FaultPoint::UnloadProgram,
            ResourceKind::Program,
            program.0,
            FaultProgram,
            A::unload_program,
        )
    }

    fn create_queue(
        &self,
        context: &Self::Context,
        desc: QueueDesc,
    ) -> Result<Self::Queue, BackendError> {
        let action = self.control.begin_call(FaultPoint::CreateQueue);
        self.validate_resource(&context.0, ResourceKind::Context)?;
        if let Some(error) = before_error(action) {
            return Err(error);
        }
        let queue = self.inner.create_queue(context.inner(), desc)?;
        let id = self
            .control
            .register_child(ResourceKind::Queue, context.id());
        if let Some(error) = after_error(action) {
            self.rollback_resource(id, ResourceKind::Queue, queue, A::destroy_queue);
            return Err(error);
        }
        Ok(FaultQueue(Tracked { id, inner: queue }))
    }

    fn destroy_queue(&self, queue: Self::Queue) -> Result<(), ReleaseFailure<Self::Queue>> {
        self.release_resource(
            FaultPoint::DestroyQueue,
            ResourceKind::Queue,
            queue.0,
            FaultQueue,
            A::destroy_queue,
        )
    }

    fn submit(
        &self,
        queue: &Self::Queue,
        program: &Self::Program,
        bindings: &[BindingRef<'_, Self::Buffer>],
        timeout: Timeout,
    ) -> Result<Self::Event, SubmitFailure<Self::Event>> {
        let action = self.control.begin_call(FaultPoint::Submit);
        self.validate_resource(&queue.0, ResourceKind::Queue)
            .map_err(SubmitFailure::Rejected)?;
        self.validate_resource(&program.0, ResourceKind::Program)
            .map_err(SubmitFailure::Rejected)?;

        let mut inner_bindings = Vec::with_capacity(bindings.len());
        for binding in bindings {
            self.validate_resource(&binding.buffer.0, ResourceKind::Buffer)
                .map_err(SubmitFailure::Rejected)?;
            inner_bindings.push(BindingRef {
                slot: binding.slot,
                buffer: binding.buffer.inner(),
                range: binding.range,
                access: binding.access,
            });
        }

        if let Some(FaultAction::Rejected(error)) = action {
            return Err(SubmitFailure::Rejected(error));
        }
        let scripted_indeterminate = match action {
            Some(FaultAction::Indeterminate(error)) => Some(error),
            _ => None,
        };
        match self
            .inner
            .submit(queue.inner(), program.inner(), &inner_bindings, timeout)
        {
            Ok(event) => {
                let mut dependencies = Vec::with_capacity(bindings.len().saturating_add(2));
                dependencies.push(queue.id());
                dependencies.push(program.id());
                dependencies.extend(bindings.iter().map(|binding| binding.buffer.id()));
                let id = self.control.register_event(dependencies);
                let event = FaultEvent(Tracked { id, inner: event });
                match scripted_indeterminate {
                    Some(error) => Err(SubmitFailure::Indeterminate { error, event }),
                    None => Ok(event),
                }
            }
            Err(SubmitFailure::Rejected(error)) => Err(SubmitFailure::Rejected(error)),
            Err(SubmitFailure::Indeterminate { error, event }) => {
                let mut dependencies = Vec::with_capacity(bindings.len().saturating_add(2));
                dependencies.push(queue.id());
                dependencies.push(program.id());
                dependencies.extend(bindings.iter().map(|binding| binding.buffer.id()));
                let id = self.control.register_event(dependencies);
                Err(SubmitFailure::Indeterminate {
                    error: scripted_indeterminate.unwrap_or(error),
                    event: FaultEvent(Tracked { id, inner: event }),
                })
            }
        }
    }

    fn poll_event(&self, event: &Self::Event) -> Result<EventState, BackendError> {
        let action = self.control.begin_call(FaultPoint::PollEvent);
        self.validate_resource(&event.0, ResourceKind::Event)?;
        if let Some(error) = before_error(action) {
            return Err(error);
        }
        if let Some(FaultAction::Completion(completion)) = action {
            self.control.set_completion(event.id(), completion);
        }
        let result = match self.control.completion(event.id()) {
            Some(completion) => Ok(completion),
            None => self.inner.poll_event(event.inner()),
        };
        match (result, after_error(action)) {
            (Ok(_), Some(error)) => Err(error),
            (result, _) => result,
        }
    }

    fn cancel_event(&self, event: &Self::Event) -> Result<(), BackendError> {
        let action = self.control.begin_call(FaultPoint::CancelEvent);
        self.validate_resource(&event.0, ResourceKind::Event)?;
        if let Some(error) = before_error(action) {
            return Err(error);
        }
        if self.control.completion(event.id()).is_some() {
            return Err(BackendError::Busy);
        }
        let result = self.inner.cancel_event(event.inner());
        match (result, after_error(action)) {
            (Ok(()), Some(error)) => Err(error),
            (result, _) => result,
        }
    }

    fn destroy_event(&self, event: Self::Event) -> Result<(), ReleaseFailure<Self::Event>> {
        let synthetic_completion = self.control.completion(event.id()).is_some();
        self.release_resource(
            FaultPoint::DestroyEvent,
            ResourceKind::Event,
            event.0,
            FaultEvent,
            move |inner, resource| {
                if synthetic_completion {
                    if let Err(error) = inner.cancel_event(&resource) {
                        if error != BackendError::Busy {
                            return Err(ReleaseFailure::Rejected { error, resource });
                        }
                    }
                }
                inner.destroy_event(resource)
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MockAccelerator, reference};
    use virtio_accel_core::{
        AccessMode, ArtifactRef, BindingRef, BufferRange, BufferUsage, ContextDesc, MemoryDomain,
    };

    fn buffer_desc() -> BufferDesc {
        BufferDesc::new(
            8,
            1,
            MemoryDomain::Shared,
            BufferUsage::TRANSFER_SOURCE
                | BufferUsage::TRANSFER_DESTINATION
                | BufferUsage::MUTABLE_STATE,
        )
        .unwrap()
    }

    fn load_barrier(
        backend: &FaultAccelerator<MockAccelerator>,
        context: &FaultContext<crate::MockContext>,
    ) -> FaultProgram<crate::MockProgram> {
        let artifact = reference::ReferenceArtifact::barrier(0);
        backend
            .load_program(
                context,
                ArtifactRef {
                    format: reference::ARTIFACT_FORMAT,
                    target: reference::TARGET_IDENTITY,
                    payload: artifact.as_bytes(),
                    resident_bytes: reference::RESIDENT_BYTES,
                },
            )
            .unwrap()
    }

    fn rejected_resource<R>(failure: ReleaseFailure<R>) -> R {
        match failure {
            ReleaseFailure::Rejected { resource, .. } => resource,
            ReleaseFailure::Indeterminate { error } => {
                panic!("expected rejected release, got {error:?}")
            }
        }
    }

    #[test]
    fn scripts_reject_invalid_or_ambiguous_steps() {
        assert_eq!(
            FaultScript::new([FaultStep::new(
                FaultPoint::Submit,
                0,
                FaultAction::Rejected(BackendError::Busy),
            )]),
            Err(FaultScriptError::ZeroOccurrence {
                point: FaultPoint::Submit,
            })
        );
        assert!(matches!(
            FaultScript::new([
                FaultStep::new(
                    FaultPoint::PollEvent,
                    1,
                    FaultAction::ErrorBefore(BackendError::Busy),
                ),
                FaultStep::new(
                    FaultPoint::PollEvent,
                    1,
                    FaultAction::ErrorAfter(BackendError::DeviceLost),
                ),
            ]),
            Err(FaultScriptError::DuplicateStep { .. })
        ));
        assert!(matches!(
            FaultScript::new([FaultStep::new(
                FaultPoint::FreeBuffer,
                1,
                FaultAction::ErrorBefore(BackendError::Busy),
            )]),
            Err(FaultScriptError::IncompatibleAction { .. })
        ));
        assert!(matches!(
            FaultScript::new([FaultStep::new(
                FaultPoint::PollEvent,
                1,
                FaultAction::Completion(EventState::Pending),
            )]),
            Err(FaultScriptError::IncompatibleAction { .. })
        ));
    }

    #[test]
    fn every_nonrelease_boundary_is_faultable_and_replayable() {
        let script = FaultScript::new([
            FaultStep::new(
                FaultPoint::DeviceInfo,
                1,
                FaultAction::ErrorBefore(BackendError::Busy),
            ),
            FaultStep::new(
                FaultPoint::CreateContext,
                1,
                FaultAction::ErrorBefore(BackendError::OutOfMemory),
            ),
            FaultStep::new(
                FaultPoint::CreateContext,
                2,
                FaultAction::ErrorAfter(BackendError::OutOfMemory),
            ),
            FaultStep::new(
                FaultPoint::AllocateBuffer,
                1,
                FaultAction::ErrorBefore(BackendError::OutOfMemory),
            ),
            FaultStep::new(
                FaultPoint::AllocateBuffer,
                2,
                FaultAction::ErrorAfter(BackendError::OutOfMemory),
            ),
            FaultStep::new(
                FaultPoint::WriteBuffer,
                1,
                FaultAction::ErrorBefore(BackendError::DeviceLost),
            ),
            FaultStep::new(
                FaultPoint::ReadBuffer,
                1,
                FaultAction::ErrorAfter(BackendError::DeviceLost),
            ),
            FaultStep::new(
                FaultPoint::LoadProgram,
                1,
                FaultAction::ErrorBefore(BackendError::OutOfMemory),
            ),
            FaultStep::new(
                FaultPoint::LoadProgram,
                2,
                FaultAction::ErrorAfter(BackendError::OutOfMemory),
            ),
            FaultStep::new(
                FaultPoint::CreateQueue,
                1,
                FaultAction::ErrorBefore(BackendError::ResourceLimit),
            ),
            FaultStep::new(
                FaultPoint::CreateQueue,
                2,
                FaultAction::ErrorAfter(BackendError::ResourceLimit),
            ),
            FaultStep::new(
                FaultPoint::Submit,
                1,
                FaultAction::Rejected(BackendError::Busy),
            ),
            FaultStep::new(
                FaultPoint::Submit,
                2,
                FaultAction::Indeterminate(BackendError::DeadlineExpired),
            ),
            FaultStep::new(
                FaultPoint::PollEvent,
                1,
                FaultAction::ErrorBefore(BackendError::Busy),
            ),
            FaultStep::new(
                FaultPoint::PollEvent,
                3,
                FaultAction::Completion(EventState::Failed(BackendError::DeadlineExpired)),
            ),
            FaultStep::new(
                FaultPoint::CancelEvent,
                1,
                FaultAction::ErrorBefore(BackendError::Busy),
            ),
        ])
        .unwrap();
        let backend = FaultAccelerator::new(MockAccelerator::default(), script);
        let control = backend.control();

        assert_eq!(backend.device_info(), Err(BackendError::Busy));
        backend.device_info().unwrap();
        assert_eq!(
            backend.create_context(ContextDesc::default()).unwrap_err(),
            BackendError::OutOfMemory
        );
        assert_eq!(
            backend.create_context(ContextDesc::default()).unwrap_err(),
            BackendError::OutOfMemory
        );
        let context = backend.create_context(ContextDesc::default()).unwrap();

        assert_eq!(
            backend
                .allocate_buffer(&context, buffer_desc())
                .unwrap_err(),
            BackendError::OutOfMemory
        );
        assert_eq!(
            backend
                .allocate_buffer(&context, buffer_desc())
                .unwrap_err(),
            BackendError::OutOfMemory
        );
        let allocation = backend.allocate_buffer(&context, buffer_desc()).unwrap();
        let (mut buffer, _) = allocation.into_parts();
        assert_eq!(
            backend.write_buffer(&mut buffer, 0, b"faulted!"),
            Err(BackendError::DeviceLost)
        );
        backend.write_buffer(&mut buffer, 0, b"faulted!").unwrap();
        let mut output = [0; 8];
        assert_eq!(
            backend.read_buffer(&buffer, 0, &mut output),
            Err(BackendError::DeviceLost)
        );
        output.fill(0);
        backend.read_buffer(&buffer, 0, &mut output).unwrap();
        assert_eq!(&output, b"faulted!");

        let artifact = reference::ReferenceArtifact::barrier(0);
        let artifact = ArtifactRef {
            format: reference::ARTIFACT_FORMAT,
            target: reference::TARGET_IDENTITY,
            payload: artifact.as_bytes(),
            resident_bytes: reference::RESIDENT_BYTES,
        };
        assert_eq!(
            backend.load_program(&context, artifact).unwrap_err(),
            BackendError::OutOfMemory
        );
        assert_eq!(
            backend.load_program(&context, artifact).unwrap_err(),
            BackendError::OutOfMemory
        );
        let program = backend.load_program(&context, artifact).unwrap();
        assert_eq!(
            backend
                .create_queue(&context, QueueDesc::default())
                .unwrap_err(),
            BackendError::ResourceLimit
        );
        assert_eq!(
            backend
                .create_queue(&context, QueueDesc::default())
                .unwrap_err(),
            BackendError::ResourceLimit
        );
        let queue = backend
            .create_queue(&context, QueueDesc::default())
            .unwrap();
        let bindings = [BindingRef {
            slot: 0,
            buffer: &buffer,
            range: BufferRange::new(0, 8).unwrap(),
            access: AccessMode::ReadWrite,
        }];
        assert!(matches!(
            backend.submit(&queue, &program, &bindings, Timeout::Infinite),
            Err(SubmitFailure::Rejected(BackendError::Busy))
        ));
        let event = match backend.submit(&queue, &program, &bindings, Timeout::Infinite) {
            Err(SubmitFailure::Indeterminate { error, event }) => {
                assert_eq!(error, BackendError::DeadlineExpired);
                event
            }
            _ => panic!("second submit must cross the indeterminate boundary"),
        };
        assert_eq!(backend.poll_event(&event), Err(BackendError::Busy));
        assert_eq!(backend.poll_event(&event), Ok(EventState::Pending));
        assert_eq!(backend.cancel_event(&event), Err(BackendError::Busy));
        backend.cancel_event(&event).unwrap();

        backend.destroy_event(event).unwrap();
        let failed_event = backend
            .submit(&queue, &program, &bindings, Timeout::Infinite)
            .unwrap();
        let failed = EventState::Failed(BackendError::DeadlineExpired);
        assert_eq!(backend.poll_event(&failed_event), Ok(failed));
        assert_eq!(backend.poll_event(&failed_event), Ok(failed));
        assert_eq!(backend.cancel_event(&failed_event), Err(BackendError::Busy));
        backend.destroy_event(failed_event).unwrap();
        backend.destroy_queue(queue).unwrap();
        backend.unload_program(program).unwrap();
        backend.free_buffer(buffer).unwrap();
        backend.destroy_context(context).unwrap();

        let snapshot = control.snapshot();
        assert!(snapshot.pending_faults.is_empty());
        assert_eq!(
            snapshot
                .releases
                .iter()
                .filter(|record| record.rollback)
                .count(),
            4
        );
        assert!(snapshot.is_clean());
        for point in [
            FaultPoint::DeviceInfo,
            FaultPoint::CreateContext,
            FaultPoint::DestroyContext,
            FaultPoint::AllocateBuffer,
            FaultPoint::WriteBuffer,
            FaultPoint::ReadBuffer,
            FaultPoint::FreeBuffer,
            FaultPoint::LoadProgram,
            FaultPoint::UnloadProgram,
            FaultPoint::CreateQueue,
            FaultPoint::DestroyQueue,
            FaultPoint::Submit,
            FaultPoint::PollEvent,
            FaultPoint::CancelEvent,
            FaultPoint::DestroyEvent,
        ] {
            assert!(snapshot.calls.iter().any(|call| call.point == point));
        }
    }

    #[test]
    fn every_resource_release_supports_rejected_and_indeterminate_outcomes() {
        let context_script = FaultScript::new([
            FaultStep::new(
                FaultPoint::DestroyContext,
                1,
                FaultAction::Rejected(BackendError::Busy),
            ),
            FaultStep::new(
                FaultPoint::DestroyContext,
                3,
                FaultAction::Indeterminate(BackendError::DeviceLost),
            ),
        ])
        .unwrap();
        let backend = FaultAccelerator::new(MockAccelerator::default(), context_script);
        let control = backend.control();
        let first = backend.create_context(ContextDesc::default()).unwrap();
        let second = backend.create_context(ContextDesc::default()).unwrap();
        let first = rejected_resource(backend.destroy_context(first).unwrap_err());
        backend.destroy_context(first).unwrap();
        assert!(matches!(
            backend.destroy_context(second),
            Err(ReleaseFailure::Indeterminate {
                error: BackendError::DeviceLost
            })
        ));
        assert_eq!(
            control
                .snapshot()
                .resources_in(ResourceState::Indeterminate),
            ResourceTally {
                contexts: 1,
                ..ResourceTally::default()
            }
        );
        control.discard_all();
        assert!(control.snapshot().is_clean());

        let buffer_script = FaultScript::new([
            FaultStep::new(
                FaultPoint::FreeBuffer,
                1,
                FaultAction::Rejected(BackendError::Busy),
            ),
            FaultStep::new(
                FaultPoint::FreeBuffer,
                3,
                FaultAction::Indeterminate(BackendError::DeviceLost),
            ),
        ])
        .unwrap();
        let backend = FaultAccelerator::new(MockAccelerator::default(), buffer_script);
        let control = backend.control();
        let context = backend.create_context(ContextDesc::default()).unwrap();
        let first = backend
            .allocate_buffer(&context, buffer_desc())
            .unwrap()
            .into_parts()
            .0;
        let second = backend
            .allocate_buffer(&context, buffer_desc())
            .unwrap()
            .into_parts()
            .0;
        let first = rejected_resource(backend.free_buffer(first).unwrap_err());
        backend.free_buffer(first).unwrap();
        assert!(matches!(
            backend.free_buffer(second),
            Err(ReleaseFailure::Indeterminate { .. })
        ));
        assert_eq!(
            control
                .snapshot()
                .resources_in(ResourceState::Indeterminate)
                .buffers,
            1
        );
        control.discard_all();
        assert!(control.snapshot().is_clean());

        let program_script = FaultScript::new([
            FaultStep::new(
                FaultPoint::UnloadProgram,
                1,
                FaultAction::Rejected(BackendError::Busy),
            ),
            FaultStep::new(
                FaultPoint::UnloadProgram,
                3,
                FaultAction::Indeterminate(BackendError::DeviceLost),
            ),
        ])
        .unwrap();
        let backend = FaultAccelerator::new(MockAccelerator::default(), program_script);
        let control = backend.control();
        let context = backend.create_context(ContextDesc::default()).unwrap();
        let first = load_barrier(&backend, &context);
        let second = load_barrier(&backend, &context);
        let first = rejected_resource(backend.unload_program(first).unwrap_err());
        backend.unload_program(first).unwrap();
        assert!(matches!(
            backend.unload_program(second),
            Err(ReleaseFailure::Indeterminate { .. })
        ));
        assert_eq!(
            control
                .snapshot()
                .resources_in(ResourceState::Indeterminate)
                .programs,
            1
        );
        control.discard_all();
        assert!(control.snapshot().is_clean());

        let queue_script = FaultScript::new([
            FaultStep::new(
                FaultPoint::DestroyQueue,
                1,
                FaultAction::Rejected(BackendError::Busy),
            ),
            FaultStep::new(
                FaultPoint::DestroyQueue,
                3,
                FaultAction::Indeterminate(BackendError::DeviceLost),
            ),
        ])
        .unwrap();
        let backend = FaultAccelerator::new(MockAccelerator::default(), queue_script);
        let control = backend.control();
        let context = backend.create_context(ContextDesc::default()).unwrap();
        let first = backend
            .create_queue(&context, QueueDesc::default())
            .unwrap();
        let second = backend
            .create_queue(&context, QueueDesc::default())
            .unwrap();
        let first = rejected_resource(backend.destroy_queue(first).unwrap_err());
        backend.destroy_queue(first).unwrap();
        assert!(matches!(
            backend.destroy_queue(second),
            Err(ReleaseFailure::Indeterminate { .. })
        ));
        assert_eq!(
            control
                .snapshot()
                .resources_in(ResourceState::Indeterminate)
                .queues,
            1
        );
        control.discard_all();
        assert!(control.snapshot().is_clean());

        let event_script = FaultScript::new([
            FaultStep::new(
                FaultPoint::DestroyEvent,
                1,
                FaultAction::Rejected(BackendError::Busy),
            ),
            FaultStep::new(
                FaultPoint::DestroyEvent,
                3,
                FaultAction::Indeterminate(BackendError::DeviceLost),
            ),
        ])
        .unwrap();
        let backend = FaultAccelerator::new(MockAccelerator::default(), event_script);
        let control = backend.control();
        let context = backend.create_context(ContextDesc::default()).unwrap();
        let buffer = backend
            .allocate_buffer(&context, buffer_desc())
            .unwrap()
            .into_parts()
            .0;
        let program = load_barrier(&backend, &context);
        let queue = backend
            .create_queue(&context, QueueDesc::default())
            .unwrap();
        let bindings = [BindingRef {
            slot: 0,
            buffer: &buffer,
            range: BufferRange::new(0, 8).unwrap(),
            access: AccessMode::ReadWrite,
        }];
        let first = backend
            .submit(&queue, &program, &bindings, Timeout::Infinite)
            .unwrap();
        let second = backend
            .submit(&queue, &program, &bindings, Timeout::Infinite)
            .unwrap();
        backend.cancel_event(&first).unwrap();
        backend.cancel_event(&second).unwrap();
        let first = rejected_resource(backend.destroy_event(first).unwrap_err());
        backend.destroy_event(first).unwrap();
        assert!(matches!(
            backend.destroy_event(second),
            Err(ReleaseFailure::Indeterminate { .. })
        ));
        assert_eq!(
            control
                .snapshot()
                .resources_in(ResourceState::Indeterminate)
                .events,
            1
        );
        control.discard_all();
        assert!(control.snapshot().is_clean());
    }

    #[test]
    fn ownership_audit_detects_double_release_and_leaks() {
        let backend = FaultAccelerator::new(MockAccelerator::default(), FaultScript::default());
        let control = backend.control();
        let context = backend.create_context(ContextDesc::default()).unwrap();
        let duplicate = context.clone();
        backend.destroy_context(context).unwrap();
        assert!(matches!(
            backend.destroy_context(duplicate),
            Err(ReleaseFailure::Indeterminate {
                error: BackendError::DeviceLost
            })
        ));
        assert!(matches!(
            control.snapshot().violations.as_slice(),
            [OwnershipViolation::ResourceNotLive {
                kind: ResourceKind::Context,
                state: ResourceState::Released,
                ..
            }]
        ));

        let premature_backend =
            FaultAccelerator::new(MockAccelerator::default(), FaultScript::default());
        let premature_control = premature_backend.control();
        let context = premature_backend
            .create_context(ContextDesc::default())
            .unwrap();
        let duplicate = context.clone();
        let _buffer = premature_backend
            .allocate_buffer(&context, buffer_desc())
            .unwrap()
            .into_parts()
            .0;
        assert!(matches!(
            premature_backend.destroy_context(duplicate),
            Err(ReleaseFailure::Indeterminate {
                error: BackendError::DeviceLost
            })
        ));
        assert!(matches!(
            premature_control.snapshot().violations.as_slice(),
            [OwnershipViolation::PrematureRelease {
                kind: ResourceKind::Context,
                ..
            }]
        ));

        let leak_backend =
            FaultAccelerator::new(MockAccelerator::default(), FaultScript::default());
        let leak_control = leak_backend.control();
        {
            let _leaked = leak_backend.create_context(ContextDesc::default()).unwrap();
        }
        assert_eq!(
            leak_control
                .snapshot()
                .resources_in(ResourceState::Live)
                .contexts,
            1
        );
        leak_control.discard_all();
        assert!(leak_control.snapshot().is_clean());
    }
}
