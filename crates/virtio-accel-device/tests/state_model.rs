use std::panic::{AssertUnwindSafe, catch_unwind};
use std::vec::Vec;

// Deterministic differential lifecycle coverage for the transport-neutral command engine.
//
// The model intentionally tracks only protocol-visible ownership, references, and recovery state.
// Backend-specific execution remains behind `FaultAccelerator<MockAccelerator>`, so each generated
// schedule compares the processor against a small executable reference graph while still auditing
// native ownership through the fault wrapper.

use virtio_accel_core::{
    AccessMode, BackendError, BufferUsage, MemoryDomain, TransportByteSink, TransportByteSource,
};
use virtio_accel_device::{
    ChainRegion, CommandOutcome, CommandProcessor, DeviceHealth, ObjectId, ObjectNamespace,
    ResetDisposition, ResourceCounts, ResourcePolicy, RetainedBytes,
};
use virtio_accel_mock::fault::{FaultAccelerator, FaultAction, FaultPoint, FaultScript, FaultStep};
use virtio_accel_mock::{MockAccelerator, reference};
use virtio_accel_proto::{
    AllocateBufferRequest, BASELINE_COMMAND_QUEUES, CreateContextRequest, CreateQueueRequest,
    KnownEventState, KnownOpcode, Le16, Le32, Le64, LoadProgramRequest, ObjectPayload,
    PROTOCOL_MAJOR, PROTOCOL_MINOR, RequestFlags, RequestHeader, ResponseHeader, StatusCode,
    SubmitRequest, SubmitResponse, TransferBufferRequest, WireBinding, WireConfig, WireEventState,
    read_exact,
};
use zerocopy::IntoBytes;

const RESPONSE_BYTES: usize = 128;
const BUFFER_BYTES: u64 = 16;
const BUFFER_POLICY_BYTES: u64 = 256;
const PROGRAM_POLICY_BYTES: u64 = 256;
const DEFAULT_STEPS: usize = 144;
const DEEP_STEPS: usize = 768;

type Backend = FaultAccelerator<MockAccelerator>;
type Processor = CommandProcessor<Backend>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ActionKind {
    CreateContext,
    AllocateBuffer,
    LoadProgram,
    CreateQueue,
    Submit,
    CrossContextSubmit,
    PollEvent,
    CancelEvent,
    CompleteEvent,
    FailEvent,
    DestroyEvent,
    WriteBuffer,
    ReadBuffer,
    FreeBuffer,
    UnloadProgram,
    DestroyQueue,
    DestroyContext,
    StaleProbe,
    Reset,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Action {
    kind: ActionKind,
    selector: u8,
    aux: u8,
}

#[derive(Clone, Debug)]
struct Scenario {
    name: &'static str,
    seed: u64,
    script: FaultScript,
    actions: Vec<Action>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ContextModel {
    id: ObjectId,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct BufferModel {
    id: ObjectId,
    context: ObjectId,
    bytes: u64,
    in_flight: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ProgramModel {
    id: ObjectId,
    context: ObjectId,
    resident_bytes: u64,
    in_flight: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct QueueModel {
    id: ObjectId,
    context: ObjectId,
    in_flight: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EventModelState {
    Pending,
    Complete,
    FailedDeviceLost,
    Cancelled,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct EventModel {
    id: ObjectId,
    context: ObjectId,
    queue: ObjectId,
    program: ObjectId,
    buffers: Vec<ObjectId>,
    state: EventModelState,
}

#[derive(Clone, Debug, Default)]
struct Model {
    contexts: Vec<ContextModel>,
    buffers: Vec<BufferModel>,
    programs: Vec<ProgramModel>,
    queues: Vec<QueueModel>,
    events: Vec<EventModel>,
    stale: Vec<ObjectId>,
    namespace: u16,
    discard_required: bool,
    reset_report: Option<ModelResetReport>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ModelResetReport {
    disposition: ResetDisposition,
    released: ResourceCounts,
    quarantined: ResourceCounts,
    released_bytes: RetainedBytes,
    quarantined_bytes: RetainedBytes,
}

impl Default for ModelResetReport {
    fn default() -> Self {
        Self {
            disposition: ResetDisposition::BackendReusable,
            released: ResourceCounts::default(),
            quarantined: ResourceCounts::default(),
            released_bytes: RetainedBytes::default(),
            quarantined_bytes: RetainedBytes::default(),
        }
    }
}

#[derive(Debug)]
struct Harness {
    processor: Processor,
    model: Model,
    request_id: u64,
    trace: Vec<Action>,
}

#[derive(Clone, Debug)]
struct Response {
    status: StatusCode,
    bytes: Vec<u8>,
}

#[derive(Clone, Copy, Debug)]
enum Pick {
    Live(ObjectId),
    Stale(ObjectId),
    Zero,
}

#[test]
fn generated_object_graphs_match_the_reference_model() {
    let seeds = [
        0x0000_0000_c001_d00d,
        0x0000_0000_51a7_eeee,
        0x0000_0000_5eed_f00d,
        0x0000_0000_8ad0_b1d5,
        0x0000_0000_d15c_a11d,
        0x0000_0000_fade_beef,
        0x1234_5678_9abc_def0,
        0xfedc_ba98_7654_3210,
    ];
    for seed in seeds {
        assert_scenario(Scenario {
            name: "generated_object_graph",
            seed,
            script: FaultScript::default(),
            actions: generated_actions(seed, DEFAULT_STEPS),
        });
    }
}

#[test]
fn explicit_race_replays_match_the_reference_model() {
    for scenario in explicit_scenarios() {
        assert_scenario(scenario);
    }
}

#[test]
#[ignore = "manual deeper exploration: VIRTIO_ACCEL_STATE_MODEL_SEED controls the first seed"]
fn deep_generated_object_graphs_match_the_reference_model() {
    let first_seed = std::env::var("VIRTIO_ACCEL_STATE_MODEL_SEED")
        .ok()
        .and_then(|value| u64::from_str_radix(value.trim_start_matches("0x"), 16).ok())
        .unwrap_or(0x9e37_79b9_7f4a_7c15);
    let mut rng = Rng::new(first_seed);
    for index in 0..64 {
        let seed = rng.next_u64();
        assert_scenario(Scenario {
            name: "deep_generated_object_graph",
            seed: seed ^ index,
            script: FaultScript::default(),
            actions: generated_actions(seed, DEEP_STEPS),
        });
    }
}

fn explicit_scenarios() -> Vec<Scenario> {
    vec![
        Scenario {
            name: "completion_before_cancel",
            seed: 0,
            script: FaultScript::default(),
            actions: vec![
                a(ActionKind::CreateContext),
                a(ActionKind::AllocateBuffer),
                a(ActionKind::LoadProgram),
                a(ActionKind::CreateQueue),
                a(ActionKind::Submit),
                a(ActionKind::CompleteEvent),
                a(ActionKind::CancelEvent),
                a(ActionKind::PollEvent),
                a(ActionKind::DestroyEvent),
                a(ActionKind::DestroyQueue),
                a(ActionKind::UnloadProgram),
                a(ActionKind::FreeBuffer),
                a(ActionKind::DestroyContext),
                a(ActionKind::Reset),
            ],
        },
        Scenario {
            name: "cancel_before_completion",
            seed: 0,
            script: FaultScript::default(),
            actions: vec![
                a(ActionKind::CreateContext),
                a(ActionKind::AllocateBuffer),
                a(ActionKind::LoadProgram),
                a(ActionKind::CreateQueue),
                a(ActionKind::Submit),
                a(ActionKind::CancelEvent),
                a(ActionKind::CompleteEvent),
                a(ActionKind::PollEvent),
                a(ActionKind::DestroyEvent),
                a(ActionKind::DestroyQueue),
                a(ActionKind::UnloadProgram),
                a(ActionKind::FreeBuffer),
                a(ActionKind::DestroyContext),
                a(ActionKind::Reset),
            ],
        },
        Scenario {
            name: "reset_while_pending",
            seed: 0,
            script: FaultScript::default(),
            actions: vec![
                a(ActionKind::CreateContext),
                a(ActionKind::AllocateBuffer),
                a(ActionKind::LoadProgram),
                a(ActionKind::CreateQueue),
                a(ActionKind::Submit),
                a(ActionKind::Reset),
                a(ActionKind::StaleProbe),
            ],
        },
        Scenario {
            name: "device_loss_before_reset",
            seed: 0,
            script: FaultScript::default(),
            actions: vec![
                a(ActionKind::CreateContext),
                a(ActionKind::AllocateBuffer),
                a(ActionKind::LoadProgram),
                a(ActionKind::CreateQueue),
                a(ActionKind::Submit),
                a(ActionKind::FailEvent),
                a(ActionKind::PollEvent),
                a(ActionKind::Reset),
            ],
        },
        Scenario {
            name: "rejected_submission_keeps_graph_unchanged",
            seed: 0,
            script: FaultScript::new([FaultStep::new(
                FaultPoint::Submit,
                1,
                FaultAction::Rejected(BackendError::Busy),
            )])
            .unwrap(),
            actions: vec![
                a(ActionKind::CreateContext),
                a(ActionKind::AllocateBuffer),
                a(ActionKind::LoadProgram),
                a(ActionKind::CreateQueue),
                a(ActionKind::Submit),
                a(ActionKind::Reset),
            ],
        },
        Scenario {
            name: "indeterminate_submission_retains_event",
            seed: 0,
            script: FaultScript::new([FaultStep::new(
                FaultPoint::Submit,
                1,
                FaultAction::Indeterminate(BackendError::DeadlineExpired),
            )])
            .unwrap(),
            actions: vec![
                a(ActionKind::CreateContext),
                a(ActionKind::AllocateBuffer),
                a(ActionKind::LoadProgram),
                a(ActionKind::CreateQueue),
                a(ActionKind::Submit),
                a(ActionKind::CancelEvent),
                a(ActionKind::DestroyEvent),
                a(ActionKind::Reset),
            ],
        },
    ]
}

fn a(kind: ActionKind) -> Action {
    Action {
        kind,
        selector: 0,
        aux: 0,
    }
}

fn generated_actions(seed: u64, steps: usize) -> Vec<Action> {
    let mut rng = Rng::new(seed);
    let mut actions = vec![
        a(ActionKind::CreateContext),
        a(ActionKind::AllocateBuffer),
        a(ActionKind::LoadProgram),
        a(ActionKind::CreateQueue),
    ];
    while actions.len() < steps {
        let kind = match rng.byte() % 19 {
            0 => ActionKind::CreateContext,
            1 => ActionKind::AllocateBuffer,
            2 => ActionKind::LoadProgram,
            3 => ActionKind::CreateQueue,
            4 => ActionKind::Submit,
            5 => ActionKind::CrossContextSubmit,
            6 => ActionKind::PollEvent,
            7 => ActionKind::CancelEvent,
            8 => ActionKind::CompleteEvent,
            9 => ActionKind::FailEvent,
            10 => ActionKind::DestroyEvent,
            11 => ActionKind::WriteBuffer,
            12 => ActionKind::ReadBuffer,
            13 => ActionKind::FreeBuffer,
            14 => ActionKind::UnloadProgram,
            15 => ActionKind::DestroyQueue,
            16 => ActionKind::DestroyContext,
            17 => ActionKind::StaleProbe,
            _ => ActionKind::Reset,
        };
        actions.push(Action {
            kind,
            selector: rng.byte(),
            aux: rng.byte(),
        });
    }
    actions
}

fn assert_scenario(scenario: Scenario) {
    if let Err(payload) = catch_unwind(AssertUnwindSafe(|| run_scenario(&scenario))) {
        let minimized = minimize(&scenario);
        let reason = panic_payload(payload);
        panic!(
            "state-model scenario failed: {}\nseed: 0x{:016x}\nreason: {}\nminimized replay:\n{}",
            scenario.name,
            scenario.seed,
            reason,
            format_actions(&minimized)
        );
    }
}

fn minimize(scenario: &Scenario) -> Vec<Action> {
    let mut candidate = scenario.actions.clone();
    let mut chunk = candidate.len().next_power_of_two() / 2;
    while chunk > 0 {
        let mut index = 0;
        while index < candidate.len() {
            let end = index.saturating_add(chunk).min(candidate.len());
            let mut trial = candidate.clone();
            trial.drain(index..end);
            if !trial.is_empty()
                && scenario_fails(&Scenario {
                    actions: trial.clone(),
                    ..scenario.clone()
                })
            {
                candidate = trial;
            } else {
                index += chunk;
            }
        }
        chunk /= 2;
    }
    candidate
}

fn scenario_fails(scenario: &Scenario) -> bool {
    catch_unwind(AssertUnwindSafe(|| run_scenario(scenario))).is_err()
}

fn panic_payload(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else if let Some(message) = payload.downcast_ref::<&'static str>() {
        (*message).to_owned()
    } else {
        "non-string panic payload".to_owned()
    }
}

fn format_actions(actions: &[Action]) -> String {
    let mut out = String::new();
    for (index, action) in actions.iter().enumerate() {
        out.push_str(&format!(
            "{index:03}: {:?}(selector={}, aux={})\n",
            action.kind, action.selector, action.aux
        ));
    }
    out
}

fn run_scenario(scenario: &Scenario) {
    let mut harness = Harness::new(scenario.script.clone());
    for action in &scenario.actions {
        harness.apply(*action);
    }
    harness.reset();
    harness.assert_matches_model();
    let control = harness.processor.accelerator().control();
    let snapshot = control.snapshot();
    if harness.model.discard_required {
        control.discard_all();
        assert!(control.snapshot().is_clean(), "{:?}", harness.trace);
    } else {
        assert!(snapshot.is_clean(), "{:?}", harness.trace);
    }
}

impl Harness {
    fn new(script: FaultScript) -> Self {
        let processor = CommandProcessor::new(
            FaultAccelerator::new(MockAccelerator::default(), script),
            &config(),
            ObjectNamespace::new(1).unwrap(),
            ResourcePolicy::new(BUFFER_POLICY_BYTES, PROGRAM_POLICY_BYTES).unwrap(),
        )
        .unwrap();
        Self {
            processor,
            model: Model {
                namespace: 1,
                ..Model::default()
            },
            request_id: 1,
            trace: Vec::new(),
        }
    }

    fn apply(&mut self, action: Action) {
        self.trace.push(action);
        if self.model.discard_required && action.kind != ActionKind::Reset {
            self.device_lost_probe();
            self.assert_matches_model();
            return;
        }
        if matches!(
            action.kind,
            ActionKind::CreateContext
                | ActionKind::AllocateBuffer
                | ActionKind::LoadProgram
                | ActionKind::CreateQueue
                | ActionKind::Submit
                | ActionKind::CrossContextSubmit
                | ActionKind::PollEvent
                | ActionKind::CancelEvent
                | ActionKind::DestroyEvent
                | ActionKind::WriteBuffer
                | ActionKind::ReadBuffer
                | ActionKind::FreeBuffer
                | ActionKind::UnloadProgram
                | ActionKind::DestroyQueue
                | ActionKind::DestroyContext
                | ActionKind::StaleProbe
        ) {
            self.model.reset_report = None;
        }

        match action.kind {
            ActionKind::CreateContext => self.create_context(),
            ActionKind::AllocateBuffer => self.allocate_buffer(action.selector),
            ActionKind::LoadProgram => self.load_program(action.selector),
            ActionKind::CreateQueue => self.create_queue(action.selector),
            ActionKind::Submit => self.submit(action.selector, false),
            ActionKind::CrossContextSubmit => self.submit(action.selector, true),
            ActionKind::PollEvent => self.poll_event(action.selector),
            ActionKind::CancelEvent => self.cancel_event(action.selector),
            ActionKind::CompleteEvent => self.complete_event(action.selector),
            ActionKind::FailEvent => self.fail_event(action.selector),
            ActionKind::DestroyEvent => self.destroy_event(action.selector),
            ActionKind::WriteBuffer => self.transfer(action.selector, action.aux, true),
            ActionKind::ReadBuffer => self.transfer(action.selector, action.aux, false),
            ActionKind::FreeBuffer => self.object_command(
                KnownOpcode::FreeBuffer,
                self.pick_buffer(action.selector),
                Self::remove_buffer_if_ok,
            ),
            ActionKind::UnloadProgram => self.object_command(
                KnownOpcode::UnloadProgram,
                self.pick_program(action.selector),
                Self::remove_program_if_ok,
            ),
            ActionKind::DestroyQueue => self.object_command(
                KnownOpcode::DestroyQueue,
                self.pick_queue(action.selector),
                Self::remove_queue_if_ok,
            ),
            ActionKind::DestroyContext => self.object_command(
                KnownOpcode::DestroyContext,
                self.pick_context(action.selector),
                Self::remove_context_if_ok,
            ),
            ActionKind::StaleProbe => self.stale_probe(action.selector),
            ActionKind::Reset => self.reset(),
        }
        self.assert_matches_model();
    }

    fn create_context(&mut self) {
        let payload = CreateContextRequest {
            flags: Le32::new(0),
            reserved: Le32::new(0),
        };
        let response = self.run(
            KnownOpcode::CreateContext,
            payload.as_bytes(),
            RESPONSE_BYTES,
        );
        if response.status == StatusCode::OK {
            self.model.contexts.push(ContextModel {
                id: response.object_id(),
            });
        }
        assert!(matches!(
            response.status,
            StatusCode::OK | StatusCode::RESOURCE_LIMIT | StatusCode::OUT_OF_MEMORY
        ));
    }

    fn allocate_buffer(&mut self, selector: u8) {
        let context = self.pick_context(selector);
        let payload = AllocateBufferRequest {
            context_id: Le64::new(context.raw()),
            bytes: Le64::new(BUFFER_BYTES),
            alignment: Le64::new(8),
            memory_domain: MemoryDomain::Host as u8,
            reserved0: [0; 7],
            usage: Le32::new(
                (BufferUsage::TRANSFER_SOURCE
                    | BufferUsage::TRANSFER_DESTINATION
                    | BufferUsage::PROGRAM_INPUT)
                    .bits(),
            ),
            reserved1: Le32::new(0),
        };
        let response = self.run(
            KnownOpcode::AllocateBuffer,
            payload.as_bytes(),
            RESPONSE_BYTES,
        );
        if response.status == StatusCode::OK {
            let context = context.expect_live();
            self.model.buffers.push(BufferModel {
                id: response.object_id(),
                context,
                bytes: BUFFER_BYTES,
                in_flight: 0,
            });
        }
        assert!(matches!(
            response.status,
            StatusCode::OK
                | StatusCode::STALE_OBJECT
                | StatusCode::INVALID_ARGUMENT
                | StatusCode::RESOURCE_LIMIT
                | StatusCode::OUT_OF_MEMORY
        ));
    }

    fn load_program(&mut self, selector: u8) {
        let context = self.pick_context(selector);
        let artifact = reference::ReferenceArtifact::barrier(0);
        let payload = LoadProgramRequest {
            context_id: Le64::new(context.raw()),
            format: Le32::new(reference::ARTIFACT_FORMAT.get()),
            flags: Le32::new(0),
            target: reference::TARGET_IDENTITY.0.map(Le32::new),
            payload_bytes: Le64::new(reference::ARTIFACT_BYTES as u64),
            resident_bytes: Le64::new(reference::RESIDENT_BYTES),
        };
        let mut body = Vec::from(payload.as_bytes());
        body.extend_from_slice(artifact.as_bytes());
        let response = self.run(KnownOpcode::LoadProgram, &body, RESPONSE_BYTES);
        if response.status == StatusCode::OK {
            let context = context.expect_live();
            self.model.programs.push(ProgramModel {
                id: response.object_id(),
                context,
                resident_bytes: reference::RESIDENT_BYTES,
                in_flight: 0,
            });
        }
        assert!(matches!(
            response.status,
            StatusCode::OK
                | StatusCode::STALE_OBJECT
                | StatusCode::INVALID_ARGUMENT
                | StatusCode::RESOURCE_LIMIT
                | StatusCode::OUT_OF_MEMORY
        ));
    }

    fn create_queue(&mut self, selector: u8) {
        let context = self.pick_context(selector);
        let payload = CreateQueueRequest {
            context_id: Le64::new(context.raw()),
            flags: Le32::new(0),
            reserved: Le32::new(0),
        };
        let response = self.run(KnownOpcode::CreateQueue, payload.as_bytes(), RESPONSE_BYTES);
        if response.status == StatusCode::OK {
            let context = context.expect_live();
            self.model.queues.push(QueueModel {
                id: response.object_id(),
                context,
                in_flight: 0,
            });
        }
        assert!(matches!(
            response.status,
            StatusCode::OK
                | StatusCode::STALE_OBJECT
                | StatusCode::INVALID_ARGUMENT
                | StatusCode::RESOURCE_LIMIT
                | StatusCode::OUT_OF_MEMORY
        ));
    }

    fn submit(&mut self, selector: u8, force_cross_context: bool) {
        let Some(queue) = self.live_queue(selector) else {
            self.invalid_submit(selector);
            return;
        };
        let Some(program) = self.live_program_for(queue.context, selector, force_cross_context)
        else {
            self.invalid_submit(selector);
            return;
        };
        let Some(buffer) = self.live_buffer_for(queue.context, selector, force_cross_context)
        else {
            self.invalid_submit(selector);
            return;
        };

        let response = self.submit_frame(queue.id, program.id, buffer.id, BUFFER_BYTES, 0);
        if response.has_event_id() {
            let event_id = response.event_id();
            self.model.events.push(EventModel {
                id: event_id,
                context: queue.context,
                queue: queue.id,
                program: program.id,
                buffers: vec![buffer.id],
                state: EventModelState::Pending,
            });
            self.queue_mut(queue.id).in_flight += 1;
            self.program_mut(program.id).in_flight += 1;
            self.buffer_mut(buffer.id).in_flight += 1;
            assert!(matches!(
                response.status,
                StatusCode::OK | StatusCode::DEADLINE_EXPIRED | StatusCode::DEVICE_LOST
            ));
            if response.status == StatusCode::DEVICE_LOST {
                self.model.discard_required = true;
            }
        } else if force_cross_context
            || program.context != queue.context
            || buffer.context != queue.context
        {
            assert_eq!(response.status, StatusCode::STALE_OBJECT);
        } else {
            assert_eq!(response.status, StatusCode::BUSY);
        }
    }

    fn invalid_submit(&mut self, selector: u8) {
        let invalid = self.pick_stale(selector).map_or(0, ObjectId::get);
        let queue = invalid;
        let program = self.pick_program(selector.rotate_left(1)).raw();
        let buffer = self.pick_buffer(selector.rotate_left(2)).raw();
        let response = self.submit_frame_raw(queue, program, buffer, BUFFER_BYTES, 0);
        assert!(matches!(
            response.status,
            StatusCode::STALE_OBJECT
                | StatusCode::INVALID_ARGUMENT
                | StatusCode::OUT_OF_BOUNDS
                | StatusCode::PERMISSION_DENIED
                | StatusCode::BUSY
        ));
    }

    fn poll_event(&mut self, selector: u8) {
        let event = self.pick_event(selector);
        let response = self.run(
            KnownOpcode::PollEvent,
            ObjectPayload {
                object_id: Le64::new(event.raw()),
            }
            .as_bytes(),
            RESPONSE_BYTES,
        );
        match event {
            Pick::Live(id) => {
                assert_eq!(response.status, StatusCode::OK);
                let state = response.event_state();
                let model = self.event(id);
                assert_eq!(state, model.state.known_state());
                if model.state == EventModelState::FailedDeviceLost {
                    self.model.discard_required = true;
                }
            }
            Pick::Stale(_) => assert_eq!(response.status, StatusCode::STALE_OBJECT),
            Pick::Zero => assert_eq!(response.status, StatusCode::INVALID_ARGUMENT),
        }
    }

    fn cancel_event(&mut self, selector: u8) {
        let event = self.pick_event(selector);
        let response = self.run(
            KnownOpcode::CancelEvent,
            ObjectPayload {
                object_id: Le64::new(event.raw()),
            }
            .as_bytes(),
            RESPONSE_BYTES,
        );
        match event {
            Pick::Live(id) => {
                let model = self.event_mut(id);
                match model.state {
                    EventModelState::Pending => {
                        assert_eq!(response.status, StatusCode::OK);
                        model.state = EventModelState::Cancelled;
                    }
                    EventModelState::Complete
                    | EventModelState::FailedDeviceLost
                    | EventModelState::Cancelled => {
                        assert_eq!(response.status, StatusCode::BUSY);
                    }
                }
            }
            Pick::Stale(_) => assert_eq!(response.status, StatusCode::STALE_OBJECT),
            Pick::Zero => assert_eq!(response.status, StatusCode::INVALID_ARGUMENT),
        }
    }

    fn complete_event(&mut self, selector: u8) {
        let Some(id) = self.live_event_id(selector) else {
            return;
        };
        let event = self
            .processor
            .state()
            .event_record(id)
            .unwrap()
            .resource()
            .unwrap()
            .inner();
        let result = self.processor.accelerator().inner().complete(event);
        let model = self.event_mut(id);
        match model.state {
            EventModelState::Pending => {
                result.unwrap();
                model.state = EventModelState::Complete;
            }
            EventModelState::Complete
            | EventModelState::FailedDeviceLost
            | EventModelState::Cancelled => {
                assert_eq!(result, Err(BackendError::Busy));
            }
        }
    }

    fn fail_event(&mut self, selector: u8) {
        let Some(id) = self.live_event_id(selector) else {
            return;
        };
        let event = self
            .processor
            .state()
            .event_record(id)
            .unwrap()
            .resource()
            .unwrap()
            .inner();
        let result = self.processor.accelerator().inner().fail_device_lost(event);
        let model = self.event_mut(id);
        match model.state {
            EventModelState::Pending => {
                result.unwrap();
                model.state = EventModelState::FailedDeviceLost;
            }
            EventModelState::Complete
            | EventModelState::FailedDeviceLost
            | EventModelState::Cancelled => {
                assert_eq!(result, Err(BackendError::Busy));
            }
        }
    }

    fn destroy_event(&mut self, selector: u8) {
        let event = self.pick_event(selector);
        let response = self.run(
            KnownOpcode::DestroyEvent,
            ObjectPayload {
                object_id: Le64::new(event.raw()),
            }
            .as_bytes(),
            RESPONSE_BYTES,
        );
        match event {
            Pick::Live(id) => match self.event(id).state {
                EventModelState::Pending => assert_eq!(response.status, StatusCode::BUSY),
                EventModelState::Complete | EventModelState::Cancelled => {
                    assert_eq!(response.status, StatusCode::OK);
                    self.remove_event(id);
                }
                EventModelState::FailedDeviceLost => {
                    assert_eq!(response.status, StatusCode::DEVICE_LOST);
                    self.model.discard_required = true;
                }
            },
            Pick::Stale(_) => assert_eq!(response.status, StatusCode::STALE_OBJECT),
            Pick::Zero => assert_eq!(response.status, StatusCode::INVALID_ARGUMENT),
        }
    }

    fn transfer(&mut self, selector: u8, aux: u8, write: bool) {
        let buffer = self.pick_buffer(selector);
        let bytes = u64::from(aux % BUFFER_BYTES as u8) + 1;
        let transfer = TransferBufferRequest {
            buffer_id: Le64::new(buffer.raw()),
            offset: Le64::new(u64::from(selector % 4)),
            bytes: Le64::new(bytes),
        };
        let mut payload = Vec::from(transfer.as_bytes());
        let opcode = if write {
            payload.extend((0..bytes).map(|index| selector.wrapping_add(index as u8)));
            KnownOpcode::WriteBuffer
        } else {
            KnownOpcode::ReadBuffer
        };
        let response = self.run(opcode, &payload, RESPONSE_BYTES);
        match buffer {
            Pick::Live(_) => assert!(matches!(
                response.status,
                StatusCode::OK
                    | StatusCode::BUSY
                    | StatusCode::OUT_OF_BOUNDS
                    | StatusCode::PERMISSION_DENIED
            )),
            Pick::Stale(_) => assert_eq!(response.status, StatusCode::STALE_OBJECT),
            Pick::Zero => assert_eq!(response.status, StatusCode::INVALID_ARGUMENT),
        }
    }

    fn object_command(&mut self, opcode: KnownOpcode, pick: Pick, on_ok: fn(&mut Self, ObjectId)) {
        let response = self.run(
            opcode,
            ObjectPayload {
                object_id: Le64::new(pick.raw()),
            }
            .as_bytes(),
            RESPONSE_BYTES,
        );
        match pick {
            Pick::Live(id) => match response.status {
                StatusCode::OK => on_ok(self, id),
                StatusCode::BUSY => {}
                status => panic!("unexpected {opcode:?} status for live id {id:?}: {status:?}"),
            },
            Pick::Stale(_) => assert_eq!(response.status, StatusCode::STALE_OBJECT),
            Pick::Zero => assert_eq!(response.status, StatusCode::INVALID_ARGUMENT),
        }
    }

    fn stale_probe(&mut self, selector: u8) {
        let Some(id) = self.pick_stale(selector) else {
            return;
        };
        for opcode in [
            KnownOpcode::DestroyContext,
            KnownOpcode::FreeBuffer,
            KnownOpcode::UnloadProgram,
            KnownOpcode::DestroyQueue,
            KnownOpcode::DestroyEvent,
        ] {
            let response = self.run(
                opcode,
                ObjectPayload {
                    object_id: Le64::new(id.get()),
                }
                .as_bytes(),
                RESPONSE_BYTES,
            );
            assert_eq!(response.status, StatusCode::STALE_OBJECT);
        }
    }

    fn device_lost_probe(&mut self) {
        let response = self.run(KnownOpcode::GetDeviceInfo, &[], RESPONSE_BYTES);
        assert_eq!(response.status, StatusCode::DEVICE_LOST, "{:?}", self.trace);
    }

    fn reset(&mut self) {
        self.model.namespace = self.model.namespace.saturating_add(1).max(2);
        let expected = self.model.expected_reset_report();
        let report = self
            .processor
            .reset(ObjectNamespace::new(self.model.namespace).unwrap())
            .unwrap();
        assert_eq!(report.disposition, expected.disposition, "{:?}", self.trace);
        assert_eq!(report.released, expected.released, "{:?}", self.trace);
        assert_eq!(report.quarantined, expected.quarantined, "{:?}", self.trace);
        assert_eq!(
            report.released_bytes, expected.released_bytes,
            "{:?}",
            self.trace
        );
        assert_eq!(
            report.quarantined_bytes, expected.quarantined_bytes,
            "{:?}",
            self.trace
        );
        self.model.apply_reset(expected);
    }

    fn submit_frame(
        &mut self,
        queue: ObjectId,
        program: ObjectId,
        buffer: ObjectId,
        bytes: u64,
        offset: u64,
    ) -> Response {
        self.submit_frame_raw(queue.get(), program.get(), buffer.get(), bytes, offset)
    }

    fn submit_frame_raw(
        &mut self,
        queue_id: u64,
        program_id: u64,
        buffer_id: u64,
        bytes: u64,
        offset: u64,
    ) -> Response {
        let submit = SubmitRequest {
            queue_id: Le64::new(queue_id),
            program_id: Le64::new(program_id),
            binding_count: Le32::new(1),
            flags: Le32::new(0),
            timeout_ns: Le64::new(0),
        };
        let binding = WireBinding {
            buffer_id: Le64::new(buffer_id),
            offset: Le64::new(offset),
            bytes: Le64::new(bytes),
            slot: Le32::new(0),
            access: AccessMode::Read as u8,
            reserved: [0; 3],
        };
        let mut payload = Vec::from(submit.as_bytes());
        payload.extend_from_slice(binding.as_bytes());
        self.run(KnownOpcode::Submit, &payload, RESPONSE_BYTES)
    }

    fn run(&mut self, opcode: KnownOpcode, payload: &[u8], response_bytes: usize) -> Response {
        let header = RequestHeader::new(
            opcode,
            RequestFlags::empty(),
            payload.len() as u32,
            self.request_id,
        );
        self.request_id = self.request_id.saturating_add(1).max(1);
        let mut frame = Vec::from(header.as_bytes());
        frame.extend_from_slice(payload);
        let regions = [
            ChainRegion::readable(frame.len() as u64),
            ChainRegion::writable(response_bytes as u64),
        ];
        let mut response = vec![0xa5; response_bytes];
        let source = TransportByteSource::new(frame.as_slice());
        let mut sink = TransportByteSink::new(response.as_mut_slice());
        let outcome = self
            .processor
            .process(&regions, &source, &mut sink)
            .unwrap();
        let CommandOutcome::Response { status, used, .. } = outcome else {
            panic!("generated command was unusable: {:?}", self.trace);
        };
        let used = used as usize;
        assert!((16..=response_bytes).contains(&used), "{:?}", self.trace);
        assert!(
            response[used..].iter().all(|byte| *byte == 0xa5),
            "{:?}",
            self.trace
        );
        let header = read_exact::<ResponseHeader>(&response[..16]).unwrap();
        assert_eq!(StatusCode(header.status.get()), status);
        assert_eq!(header.payload_bytes.get() as usize + 16, used);
        response.truncate(used);
        Response {
            status,
            bytes: response,
        }
    }

    fn assert_matches_model(&self) {
        if self.model.discard_required {
            assert_eq!(
                self.processor.health(),
                DeviceHealth::BackendDiscardRequired
            );
            return;
        }
        assert_eq!(
            self.processor.health(),
            DeviceHealth::Running,
            "{:?}",
            self.trace
        );
        assert_eq!(
            self.processor.state().resource_counts(),
            self.model.resource_counts(),
            "{:?}",
            self.trace
        );
        assert_eq!(
            self.processor.retained_bytes(),
            self.model.retained_bytes(),
            "{:?}",
            self.trace
        );

        for context in &self.model.contexts {
            let record = self.processor.state().context_record(context.id).unwrap();
            let children = record.children();
            assert_eq!(
                children.buffers,
                self.model
                    .buffers
                    .iter()
                    .filter(|buffer| buffer.context == context.id)
                    .count() as u32
            );
            assert_eq!(
                children.programs,
                self.model
                    .programs
                    .iter()
                    .filter(|program| program.context == context.id)
                    .count() as u32
            );
            assert_eq!(
                children.queues,
                self.model
                    .queues
                    .iter()
                    .filter(|queue| queue.context == context.id)
                    .count() as u32
            );
            assert_eq!(
                children.events,
                self.model
                    .events
                    .iter()
                    .filter(|event| event.context == context.id)
                    .count() as u32
            );
        }
        for buffer in &self.model.buffers {
            let record = self.processor.state().buffer_record(buffer.id).unwrap();
            assert_eq!(record.context_id(), buffer.context);
            assert_eq!(record.info().allocation_bytes(), buffer.bytes);
            assert_eq!(record.in_flight(), buffer.in_flight);
        }
        for program in &self.model.programs {
            let record = self.processor.state().program_record(program.id).unwrap();
            assert_eq!(record.context_id(), program.context);
            assert_eq!(record.resident_bytes(), program.resident_bytes);
            assert_eq!(record.in_flight(), program.in_flight);
        }
        for queue in &self.model.queues {
            let record = self.processor.state().queue_record(queue.id).unwrap();
            assert_eq!(record.context_id(), queue.context);
            assert_eq!(record.in_flight(), queue.in_flight);
        }
        for event in &self.model.events {
            let record = self.processor.state().event_record(event.id).unwrap();
            assert_eq!(record.context_id(), event.context);
            assert_eq!(record.queue_id(), event.queue);
            assert_eq!(record.program_id(), event.program);
            assert_eq!(record.buffer_ids(), event.buffers.as_slice());
        }
        for stale in &self.model.stale {
            assert!(self.processor.state().context_record(*stale).is_err());
            assert!(self.processor.state().buffer_record(*stale).is_err());
            assert!(self.processor.state().program_record(*stale).is_err());
            assert!(self.processor.state().queue_record(*stale).is_err());
            assert!(self.processor.state().event_record(*stale).is_err());
        }
    }

    fn pick_context(&self, selector: u8) -> Pick {
        pick_id(
            self.model.contexts.iter().map(|item| item.id),
            &self.model.stale,
            selector,
        )
    }

    fn pick_buffer(&self, selector: u8) -> Pick {
        pick_id(
            self.model.buffers.iter().map(|item| item.id),
            &self.model.stale,
            selector,
        )
    }

    fn pick_program(&self, selector: u8) -> Pick {
        pick_id(
            self.model.programs.iter().map(|item| item.id),
            &self.model.stale,
            selector,
        )
    }

    fn pick_queue(&self, selector: u8) -> Pick {
        pick_id(
            self.model.queues.iter().map(|item| item.id),
            &self.model.stale,
            selector,
        )
    }

    fn pick_event(&self, selector: u8) -> Pick {
        pick_id(
            self.model.events.iter().map(|item| item.id),
            &self.model.stale,
            selector,
        )
    }

    fn pick_stale(&self, selector: u8) -> Option<ObjectId> {
        self.model
            .stale
            .get(usize::from(selector) % self.model.stale.len().max(1))
            .copied()
    }

    fn live_queue(&self, selector: u8) -> Option<QueueModel> {
        pick_live(&self.model.queues, selector)
    }

    fn live_program_for(
        &self,
        context: ObjectId,
        selector: u8,
        force_cross_context: bool,
    ) -> Option<ProgramModel> {
        pick_matching_or_cross_context(
            &self.model.programs,
            context,
            selector,
            force_cross_context,
            |item| item.context,
        )
    }

    fn live_buffer_for(
        &self,
        context: ObjectId,
        selector: u8,
        force_cross_context: bool,
    ) -> Option<BufferModel> {
        pick_matching_or_cross_context(
            &self.model.buffers,
            context,
            selector,
            force_cross_context,
            |item| item.context,
        )
    }

    fn live_event_id(&self, selector: u8) -> Option<ObjectId> {
        pick_live(&self.model.events, selector).map(|event| event.id)
    }

    fn event(&self, id: ObjectId) -> &EventModel {
        self.model
            .events
            .iter()
            .find(|event| event.id == id)
            .unwrap()
    }

    fn event_mut(&mut self, id: ObjectId) -> &mut EventModel {
        self.model
            .events
            .iter_mut()
            .find(|event| event.id == id)
            .unwrap()
    }

    fn queue_mut(&mut self, id: ObjectId) -> &mut QueueModel {
        self.model
            .queues
            .iter_mut()
            .find(|queue| queue.id == id)
            .unwrap()
    }

    fn program_mut(&mut self, id: ObjectId) -> &mut ProgramModel {
        self.model
            .programs
            .iter_mut()
            .find(|program| program.id == id)
            .unwrap()
    }

    fn buffer_mut(&mut self, id: ObjectId) -> &mut BufferModel {
        self.model
            .buffers
            .iter_mut()
            .find(|buffer| buffer.id == id)
            .unwrap()
    }

    fn remove_event(&mut self, id: ObjectId) {
        let index = self
            .model
            .events
            .iter()
            .position(|event| event.id == id)
            .unwrap();
        let event = self.model.events.swap_remove(index);
        self.queue_mut(event.queue).in_flight -= 1;
        self.program_mut(event.program).in_flight -= 1;
        for buffer in event.buffers {
            self.buffer_mut(buffer).in_flight -= 1;
        }
        self.model.stale.push(event.id);
    }

    fn remove_buffer_if_ok(&mut self, id: ObjectId) {
        if let Some(index) = self.model.buffers.iter().position(|buffer| buffer.id == id) {
            let buffer = self.model.buffers.swap_remove(index);
            assert_eq!(buffer.in_flight, 0);
            self.model.stale.push(buffer.id);
        }
    }

    fn remove_program_if_ok(&mut self, id: ObjectId) {
        if let Some(index) = self
            .model
            .programs
            .iter()
            .position(|program| program.id == id)
        {
            let program = self.model.programs.swap_remove(index);
            assert_eq!(program.in_flight, 0);
            self.model.stale.push(program.id);
        }
    }

    fn remove_queue_if_ok(&mut self, id: ObjectId) {
        if let Some(index) = self.model.queues.iter().position(|queue| queue.id == id) {
            let queue = self.model.queues.swap_remove(index);
            assert_eq!(queue.in_flight, 0);
            self.model.stale.push(queue.id);
        }
    }

    fn remove_context_if_ok(&mut self, id: ObjectId) {
        if let Some(index) = self
            .model
            .contexts
            .iter()
            .position(|context| context.id == id)
        {
            assert!(!self.model.buffers.iter().any(|buffer| buffer.context == id));
            assert!(
                !self
                    .model
                    .programs
                    .iter()
                    .any(|program| program.context == id)
            );
            assert!(!self.model.queues.iter().any(|queue| queue.context == id));
            assert!(!self.model.events.iter().any(|event| event.context == id));
            let context = self.model.contexts.swap_remove(index);
            self.model.stale.push(context.id);
        }
    }
}

impl Model {
    fn resource_counts(&self) -> ResourceCounts {
        ResourceCounts {
            contexts: self.contexts.len() as u64,
            buffers: self.buffers.len() as u64,
            programs: self.programs.len() as u64,
            queues: self.queues.len() as u64,
            events: self.events.len() as u64,
        }
    }

    fn retained_bytes(&self) -> RetainedBytes {
        RetainedBytes {
            buffer_backing: self
                .buffers
                .iter()
                .map(|buffer| u128::from(buffer.bytes))
                .sum(),
            program_resident: self
                .programs
                .iter()
                .map(|program| u128::from(program.resident_bytes))
                .sum(),
        }
    }

    fn expected_reset_report(&self) -> ModelResetReport {
        if self.discard_required {
            return self.reset_report.unwrap_or_else(|| ModelResetReport {
                disposition: ResetDisposition::BackendDiscardRequired,
                quarantined: self.resource_counts(),
                quarantined_bytes: self.retained_bytes(),
                ..ModelResetReport::default()
            });
        }

        let mut events = self.events.iter().collect::<Vec<_>>();
        events.sort_unstable_by_key(|event| event.id.get() as u32);
        if let Some(failed_index) = events
            .iter()
            .position(|event| event.state == EventModelState::FailedDeviceLost)
        {
            let released = ResourceCounts {
                events: failed_index as u64,
                ..ResourceCounts::default()
            };
            let mut quarantined = self.resource_counts();
            quarantined.events -= released.events;
            return ModelResetReport {
                disposition: ResetDisposition::BackendDiscardRequired,
                released,
                quarantined,
                released_bytes: RetainedBytes::default(),
                quarantined_bytes: self.retained_bytes(),
            };
        }

        let released = self.resource_counts();
        let released_bytes = self.retained_bytes();
        ModelResetReport {
            disposition: ResetDisposition::BackendReusable,
            released,
            quarantined: ResourceCounts::default(),
            released_bytes,
            quarantined_bytes: RetainedBytes::default(),
        }
    }

    fn apply_reset(&mut self, report: ModelResetReport) {
        self.reset_report = Some(report);
        self.stale.extend(self.contexts.iter().map(|item| item.id));
        self.stale.extend(self.buffers.iter().map(|item| item.id));
        self.stale.extend(self.programs.iter().map(|item| item.id));
        self.stale.extend(self.queues.iter().map(|item| item.id));
        self.stale.extend(self.events.iter().map(|item| item.id));
        self.contexts.clear();
        self.buffers.clear();
        self.programs.clear();
        self.queues.clear();
        self.events.clear();
        self.discard_required = report.disposition == ResetDisposition::BackendDiscardRequired;
    }
}

impl EventModelState {
    fn known_state(self) -> KnownEventState {
        match self {
            Self::Pending => KnownEventState::Pending,
            Self::Complete => KnownEventState::Complete,
            Self::FailedDeviceLost => KnownEventState::Failed,
            Self::Cancelled => KnownEventState::Cancelled,
        }
    }
}

impl Response {
    fn object_id(&self) -> ObjectId {
        let payload = read_exact::<ObjectPayload>(&self.bytes[16..24]).unwrap();
        ObjectId::from_raw(payload.object_id.get()).unwrap()
    }

    fn has_event_id(&self) -> bool {
        self.bytes.len() >= 24
            && matches!(
                self.status,
                StatusCode::OK | StatusCode::DEADLINE_EXPIRED | StatusCode::DEVICE_LOST
            )
    }

    fn event_id(&self) -> ObjectId {
        let payload = read_exact::<SubmitResponse>(&self.bytes[16..24]).unwrap();
        ObjectId::from_raw(payload.event_id.get()).unwrap()
    }

    fn event_state(&self) -> KnownEventState {
        let payload = read_exact::<WireEventState>(&self.bytes[16..24]).unwrap();
        payload.known_state().unwrap()
    }
}

impl Pick {
    fn raw(self) -> u64 {
        match self {
            Self::Live(id) | Self::Stale(id) => id.get(),
            Self::Zero => 0,
        }
    }

    fn expect_live(self) -> ObjectId {
        match self {
            Self::Live(id) => id,
            Self::Stale(_) | Self::Zero => panic!("expected live pick"),
        }
    }
}

fn pick_id(live: impl Iterator<Item = ObjectId>, stale: &[ObjectId], selector: u8) -> Pick {
    let live = live.collect::<Vec<_>>();
    if selector & 0x80 == 0 && !live.is_empty() {
        Pick::Live(live[usize::from(selector) % live.len()])
    } else if !stale.is_empty() {
        Pick::Stale(stale[usize::from(selector) % stale.len()])
    } else {
        Pick::Zero
    }
}

fn pick_live<T: Clone>(items: &[T], selector: u8) -> Option<T> {
    items
        .get(usize::from(selector) % items.len().max(1))
        .cloned()
}

fn pick_matching_or_cross_context<T: Copy>(
    items: &[T],
    context: ObjectId,
    selector: u8,
    force_cross_context: bool,
    context_of: impl Fn(T) -> ObjectId,
) -> Option<T> {
    if force_cross_context {
        items
            .iter()
            .copied()
            .find(|item| context_of(*item) != context)
    } else {
        let matching = items
            .iter()
            .copied()
            .filter(|item| context_of(*item) == context)
            .collect::<Vec<_>>();
        matching
            .get(usize::from(selector) % matching.len().max(1))
            .copied()
    }
}

fn config() -> WireConfig {
    WireConfig {
        protocol_major: Le16::new(PROTOCOL_MAJOR),
        protocol_minor: Le16::new(PROTOCOL_MINOR),
        command_queue_count: Le16::new(BASELINE_COMMAND_QUEUES),
        max_chain_descriptors: Le16::new(8),
        max_request_bytes: Le32::new(512),
        max_response_bytes: Le32::new(RESPONSE_BYTES as u32),
    }
}

#[derive(Clone, Copy, Debug)]
struct Rng {
    state: u64,
}

impl Rng {
    const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    fn byte(&mut self) -> u8 {
        self.next_u64() as u8
    }
}
