//! Integration and acceptance suite for the OpenVINO backend.
//!
//! Runs against every enumerated inference device. Without an OpenVINO runtime this file does
//! not compile (the crate builds its placeholder); with a runtime but no device, tests skip.

#![cfg(va_openvino)]

use std::time::{Duration, Instant};

use virtio_accel_conformance::numerics::{
    IDENTITY_EDGES_FP16, IDENTITY_EDGES_FP32, IDENTITY_FP8E4M3, IDENTITY_FP8E5M2, IDENTITY_INT4,
    IDENTITY_INT8, MATMUL_FP16, MATMUL_FP32, MAX_POOL2D_FP16, MAX_POOL2D_FP32,
};
use virtio_accel_conformance::{
    BindingFixture, CaseStatus, ConformanceHooks, ProgramFixture, SubmissionPathDiagnostics,
    TargetDescription, run,
};
use virtio_accel_core::{
    Accelerator, AccessMode, ArtifactRef, BackendError, BindingRef, BufferDesc, BufferRange,
    BufferUsage, ByteSink, ByteSource, ContextDesc, EventState, MemoryDomain, QueueDesc,
    SubmitFailure, Timeout,
};
use virtio_accel_openvino::{
    InitError, OPENVINO_TOSA_TARGET, OpenVinoAccelerator, OpenVinoEvent, REQUIRED_RESIDENT_BYTES,
};
use virtio_accel_tosa::parse;

const IDENTITY_FP32_LOCAL: &[u8] = include_bytes!("data/identity-fp32-v1.0.0.tosa");

const BUFFER_ALIGNMENT: u64 = 4096;

#[derive(Debug)]
struct SliceSource<'a>(&'a [u8]);

impl ByteSource for SliceSource<'_> {
    fn len(&self) -> u64 {
        self.0.len() as u64
    }

    fn read_at(&self, offset: u64, target: &mut [u8]) -> Result<(), BackendError> {
        ByteSource::read_at(self.0, offset, target)
    }

    fn as_contiguous(&self) -> Option<&[u8]> {
        Some(self.0)
    }
}

#[derive(Debug)]
struct VecSink(Vec<u8>);

impl ByteSink for VecSink {
    fn len(&self) -> u64 {
        self.0.len() as u64
    }

    fn write_at(&mut self, offset: u64, source: &[u8]) -> Result<(), BackendError> {
        ByteSink::write_at(self.0.as_mut_slice(), offset, source)
    }

    fn as_contiguous_mut(&mut self) -> Option<&mut [u8]> {
        Some(&mut self.0)
    }
}

/// A default-device backend, or `None` when the host enumerates no inference device.
fn backend() -> Option<OpenVinoAccelerator> {
    match OpenVinoAccelerator::new() {
        Ok(backend) => Some(backend),
        Err(InitError::DeviceUnavailable) => None,
        Err(error) => panic!("backend initialization failed: {error}"),
    }
}

fn devices() -> Vec<String> {
    backend()
        .map(|backend| backend.runtime_devices().unwrap())
        .unwrap_or_default()
}

fn wait_for_terminal(backend: &OpenVinoAccelerator, event: &OpenVinoEvent) -> EventState {
    let deadline = Instant::now() + Duration::from_secs(15);
    loop {
        match backend.poll_event(event).unwrap() {
            EventState::Pending => {
                assert!(Instant::now() < deadline, "inference never completed");
                std::thread::sleep(Duration::from_millis(1));
            }
            terminal => return terminal,
        }
    }
}

/// Load a device-neutral TOSA artifact; distinguishes a plugin compile rejection (report and
/// skip on non-CPU devices) from every other failure.
enum LoadOutcome<A: Accelerator> {
    Loaded(A::Program),
    Rejected(BackendError),
}

fn load_tosa(
    backend: &OpenVinoAccelerator,
    context: &<OpenVinoAccelerator as Accelerator>::Context,
    artifact: &[u8],
) -> LoadOutcome<OpenVinoAccelerator> {
    let model = parse(artifact).unwrap();
    let artifact = model
        .artifact_ref(OPENVINO_TOSA_TARGET, REQUIRED_RESIDENT_BYTES)
        .unwrap();
    match backend.load_program(context, artifact) {
        Ok(program) => LoadOutcome::Loaded(program),
        Err(error) => LoadOutcome::Rejected(error),
    }
}

/// Full lifecycle: allocate, write inputs, execute, and read the single output back.
///
/// Returns `None` when the device's plugin rejected the model at load time; that is reported by
/// the caller and tolerated only off-CPU (no silent downconversion, no silent skip on CPU).
fn run_tosa_bytes(
    device: &str,
    artifact: &[u8],
    inputs: &[&[u8]],
    output_bytes: usize,
) -> Option<Vec<u8>> {
    let backend = OpenVinoAccelerator::with_device(device).unwrap();
    let context = backend.create_context(ContextDesc::default()).unwrap();
    let program = match load_tosa(&backend, &context, artifact) {
        LoadOutcome::Loaded(program) => program,
        LoadOutcome::Rejected(error) => {
            assert!(
                !device.starts_with("CPU"),
                "the CPU plugin must admit every corpus artifact: {error:?}"
            );
            eprintln!("skipping {device}: plugin rejected the model at load time: {error:?}");
            backend.destroy_context(context).unwrap();
            return None;
        }
    };

    let mut input_buffers = Vec::with_capacity(inputs.len());
    for bytes in inputs {
        let desc = BufferDesc::new(
            bytes.len() as u64,
            BUFFER_ALIGNMENT,
            MemoryDomain::Shared,
            BufferUsage::TRANSFER_DESTINATION | BufferUsage::PROGRAM_INPUT,
        )
        .unwrap();
        let (mut buffer, _) = backend
            .allocate_buffer(&context, desc)
            .unwrap()
            .into_parts();
        backend
            .write_buffer(&mut buffer, 0, &SliceSource(bytes))
            .unwrap();
        input_buffers.push(buffer);
    }
    let output_desc = BufferDesc::new(
        output_bytes as u64,
        BUFFER_ALIGNMENT,
        MemoryDomain::Shared,
        BufferUsage::TRANSFER_SOURCE | BufferUsage::PROGRAM_OUTPUT,
    )
    .unwrap();
    let (output, _) = backend
        .allocate_buffer(&context, output_desc)
        .unwrap()
        .into_parts();
    let queue = backend
        .create_queue(&context, QueueDesc::default())
        .unwrap();

    let mut bindings = input_buffers
        .iter()
        .enumerate()
        .map(|(slot, buffer)| BindingRef {
            slot: slot as u32,
            buffer,
            range: BufferRange::new(0, inputs[slot].len() as u64).unwrap(),
            access: AccessMode::Read,
        })
        .collect::<Vec<_>>();
    bindings.push(BindingRef {
        slot: inputs.len() as u32,
        buffer: &output,
        range: BufferRange::new(0, output_bytes as u64).unwrap(),
        access: AccessMode::Write,
    });
    let event = backend
        .submit(&queue, &program, &bindings, Timeout::Infinite)
        .unwrap_or_else(|failure| match failure {
            SubmitFailure::Rejected(error) => panic!("{device}: submission rejected: {error:?}"),
            SubmitFailure::Indeterminate { error, .. } => {
                panic!("{device}: submission indeterminate: {error:?}")
            }
        });
    assert_eq!(
        wait_for_terminal(&backend, &event),
        EventState::Complete,
        "{device}"
    );
    backend.destroy_event(event).unwrap();

    let mut bytes = VecSink(vec![0; output_bytes]);
    backend.read_buffer(&output, 0, &mut bytes).unwrap();
    drop(bindings);
    backend.destroy_queue(queue).unwrap();
    backend.unload_program(program).unwrap();
    backend.free_buffer(output).unwrap();
    for buffer in input_buffers {
        backend.free_buffer(buffer).unwrap();
    }
    backend.destroy_context(context).unwrap();
    Some(bytes.0)
}

fn float_bytes(values: impl IntoIterator<Item = f32>) -> Vec<u8> {
    values
        .into_iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect()
}

fn run_tosa_fp32(
    device: &str,
    artifact: &[u8],
    inputs: &[&[f32]],
    outputs: usize,
) -> Option<Vec<f32>> {
    let encoded = inputs
        .iter()
        .map(|values| float_bytes(values.iter().copied()))
        .collect::<Vec<_>>();
    let input_bytes = encoded.iter().map(Vec::as_slice).collect::<Vec<_>>();
    run_tosa_bytes(
        device,
        artifact,
        &input_bytes,
        outputs.checked_mul(4).unwrap(),
    )
    .map(|bytes| {
        bytes
            .chunks_exact(4)
            .map(|bytes| f32::from_ne_bytes(bytes.try_into().unwrap()))
            .collect()
    })
}

fn run_tosa_fp16(
    device: &str,
    artifact: &[u8],
    inputs: &[&[u16]],
    outputs: usize,
) -> Option<Vec<u16>> {
    let encoded = inputs
        .iter()
        .map(|values| {
            values
                .iter()
                .flat_map(|bits| bits.to_ne_bytes())
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let input_bytes = encoded.iter().map(Vec::as_slice).collect::<Vec<_>>();
    run_tosa_bytes(
        device,
        artifact,
        &input_bytes,
        outputs.checked_mul(2).unwrap(),
    )
    .map(|bytes| {
        bytes
            .chunks_exact(2)
            .map(|bytes| u16::from_ne_bytes(bytes.try_into().unwrap()))
            .collect()
    })
}

#[test]
fn reports_stable_valid_metadata_for_every_device() {
    for device in devices() {
        let backend = OpenVinoAccelerator::with_device(&device).unwrap();
        let info = backend.device_info().unwrap();
        info.validate().unwrap();
        assert_eq!(info, backend.device_info().unwrap(), "{device}");
        assert_eq!(backend.device_name(), device);
    }
}

#[test]
fn lowers_and_executes_device_neutral_tosa() {
    let Some(backend) = backend() else { return };
    eprintln!("executing on {}", backend.device_name());
    let payload = float_bytes([42.5]);
    let output = run_tosa_bytes(
        backend.device_name(),
        IDENTITY_FP32_LOCAL,
        &[&payload],
        payload.len(),
    );
    assert_eq!(output, Some(payload));
}

#[test]
fn executes_the_shared_fp32_corpus_on_every_available_device() {
    for device in devices() {
        for case in [&MATMUL_FP32, &MAX_POOL2D_FP32, &IDENTITY_EDGES_FP32] {
            let inputs = case
                .inputs
                .iter()
                .map(|tensor| tensor.values)
                .collect::<Vec<_>>();
            let Some(actual) = run_tosa_fp32(
                &device,
                case.artifact,
                &inputs,
                case.outputs[0].values.len(),
            ) else {
                continue;
            };
            assert!(
                case.output_matches(0, &actual),
                "{device}: {} produced {actual:?}",
                case.name
            );
        }
    }
}

#[test]
fn executes_the_shared_fp16_corpus_on_every_available_device() {
    for device in devices() {
        for case in [&MATMUL_FP16, &MAX_POOL2D_FP16, &IDENTITY_EDGES_FP16] {
            let inputs = case
                .inputs
                .iter()
                .map(|tensor| tensor.bits)
                .collect::<Vec<_>>();
            let Some(actual) =
                run_tosa_fp16(&device, case.artifact, &inputs, case.outputs[0].bits.len())
            else {
                continue;
            };
            assert!(
                case.output_matches(0, &actual),
                "{device}: {} produced {actual:?}",
                case.name
            );
        }
    }
}

#[test]
fn rejects_packed_low_precision_artifacts_while_loading() {
    let Some(backend) = backend() else { return };
    let context = backend.create_context(ContextDesc::default()).unwrap();
    for case in [
        &IDENTITY_INT8,
        &IDENTITY_INT4,
        &IDENTITY_FP8E4M3,
        &IDENTITY_FP8E5M2,
    ] {
        // The declared floating-point target does not admit these artifacts; loading must fail
        // before any native compilation, and never silently dequantize.
        let artifact = ArtifactRef {
            format: virtio_accel_tosa::ARTIFACT_FORMAT,
            target: OPENVINO_TOSA_TARGET.to_identity(),
            payload: &SliceSource(case.artifact),
            resident_bytes: REQUIRED_RESIDENT_BYTES,
        };
        let error = backend.load_program(&context, artifact).unwrap_err();
        assert!(
            matches!(
                error,
                BackendError::Unsupported | BackendError::InvalidArgument
            ),
            "{}: {error:?}",
            case.name
        );
    }
    backend.destroy_context(context).unwrap();
}

#[test]
fn permits_overlapping_read_only_inputs_across_async_inferences() {
    const IN_FLIGHT: usize = 16;
    let Some(backend) = backend() else { return };
    let context = backend.create_context(ContextDesc::default()).unwrap();
    let program = match load_tosa(&backend, &context, IDENTITY_FP32_LOCAL) {
        LoadOutcome::Loaded(program) => program,
        LoadOutcome::Rejected(error) => panic!("identity load failed: {error:?}"),
    };
    let queue = backend
        .create_queue(&context, QueueDesc::default())
        .unwrap();

    let payload = float_bytes([7.75]);
    let input_desc = BufferDesc::new(
        payload.len() as u64,
        BUFFER_ALIGNMENT,
        MemoryDomain::Shared,
        BufferUsage::TRANSFER_DESTINATION | BufferUsage::PROGRAM_INPUT,
    )
    .unwrap();
    let (mut shared_input, _) = backend
        .allocate_buffer(&context, input_desc)
        .unwrap()
        .into_parts();
    backend
        .write_buffer(&mut shared_input, 0, &SliceSource(&payload))
        .unwrap();

    let output_desc = BufferDesc::new(
        payload.len() as u64,
        BUFFER_ALIGNMENT,
        MemoryDomain::Shared,
        BufferUsage::TRANSFER_SOURCE | BufferUsage::PROGRAM_OUTPUT,
    )
    .unwrap();
    let mut outputs = Vec::with_capacity(IN_FLIGHT);
    for _ in 0..IN_FLIGHT {
        outputs.push(
            backend
                .allocate_buffer(&context, output_desc)
                .unwrap()
                .into_parts()
                .0,
        );
    }

    let mut events = Vec::with_capacity(IN_FLIGHT);
    for output in &outputs {
        let bindings = [
            BindingRef {
                slot: 0,
                buffer: &shared_input,
                range: BufferRange::new(0, payload.len() as u64).unwrap(),
                access: AccessMode::Read,
            },
            BindingRef {
                slot: 1,
                buffer: output,
                range: BufferRange::new(0, payload.len() as u64).unwrap(),
                access: AccessMode::Write,
            },
        ];
        events.push(
            backend
                .submit(&queue, &program, &bindings, Timeout::Infinite)
                .unwrap_or_else(|failure| match failure {
                    SubmitFailure::Rejected(error) => {
                        panic!("overlapping read-only submission rejected: {error:?}")
                    }
                    SubmitFailure::Indeterminate { error, .. } => {
                        panic!("submission indeterminate: {error:?}")
                    }
                }),
        );
    }
    for event in &events {
        assert_eq!(wait_for_terminal(&backend, event), EventState::Complete);
    }
    assert_eq!(backend.direct_binding_admissions(), (IN_FLIGHT * 2) as u64);
    for event in events {
        backend.destroy_event(event).unwrap();
    }
    for output in outputs {
        let mut bytes = VecSink(vec![0; payload.len()]);
        backend.read_buffer(&output, 0, &mut bytes).unwrap();
        assert_eq!(bytes.0, payload);
        backend.free_buffer(output).unwrap();
    }
    backend.free_buffer(shared_input).unwrap();
    backend.destroy_queue(queue).unwrap();
    backend.unload_program(program).unwrap();
    backend.destroy_context(context).unwrap();
}

#[test]
fn repeated_load_unload_is_stable() {
    let Some(backend) = backend() else { return };
    let context = backend.create_context(ContextDesc::default()).unwrap();
    for _ in 0..8 {
        match load_tosa(&backend, &context, MATMUL_FP32.artifact) {
            LoadOutcome::Loaded(program) => backend.unload_program(program).unwrap(),
            LoadOutcome::Rejected(error) => panic!("matmul load failed: {error:?}"),
        }
    }
    backend.destroy_context(context).unwrap();
}

#[test]
fn finite_deadlines_reach_a_stable_terminal_state() {
    let Some(backend) = backend() else { return };
    let context = backend.create_context(ContextDesc::default()).unwrap();
    let program = match load_tosa(&backend, &context, IDENTITY_FP32_LOCAL) {
        LoadOutcome::Loaded(program) => program,
        LoadOutcome::Rejected(error) => panic!("identity load failed: {error:?}"),
    };
    let queue = backend
        .create_queue(&context, QueueDesc::default())
        .unwrap();
    let desc = BufferDesc::new(
        4,
        BUFFER_ALIGNMENT,
        MemoryDomain::Shared,
        BufferUsage::TRANSFER_SOURCE
            | BufferUsage::TRANSFER_DESTINATION
            | BufferUsage::PROGRAM_INPUT
            | BufferUsage::PROGRAM_OUTPUT,
    )
    .unwrap();
    let (input, _) = backend
        .allocate_buffer(&context, desc)
        .unwrap()
        .into_parts();
    let (output, _) = backend
        .allocate_buffer(&context, desc)
        .unwrap()
        .into_parts();
    let bindings = [
        BindingRef {
            slot: 0,
            buffer: &input,
            range: BufferRange::new(0, 4).unwrap(),
            access: AccessMode::Read,
        },
        BindingRef {
            slot: 1,
            buffer: &output,
            range: BufferRange::new(0, 4).unwrap(),
            access: AccessMode::Write,
        },
    ];
    let timeout = Timeout::AfterNs(core::num::NonZeroU64::new(1).unwrap());
    match backend.submit(&queue, &program, &bindings, timeout) {
        Ok(event) => {
            // Either the work beat the deadline or the deadline cancelled it; both must be
            // stable terminal states.
            let terminal = wait_for_terminal(&backend, &event);
            assert!(
                matches!(
                    terminal,
                    EventState::Complete | EventState::Failed(BackendError::DeadlineExpired)
                ),
                "unexpected terminal state {terminal:?}"
            );
            assert_eq!(backend.poll_event(&event).unwrap(), terminal);
            assert_eq!(backend.poll_event(&event).unwrap(), terminal);
            backend.destroy_event(event).unwrap();
        }
        Err(SubmitFailure::Rejected(BackendError::DeadlineExpired)) => {}
        Err(SubmitFailure::Rejected(error)) => panic!("finite timeout rejected: {error:?}"),
        Err(SubmitFailure::Indeterminate { error, .. }) => panic!("indeterminate: {error:?}"),
    }
    backend.destroy_queue(queue).unwrap();
    backend.unload_program(program).unwrap();
    backend.free_buffer(input).unwrap();
    backend.free_buffer(output).unwrap();
    backend.destroy_context(context).unwrap();
}

struct Hooks;

impl ConformanceHooks<OpenVinoAccelerator> for Hooks {
    fn complete_event(
        &self,
        backend: &OpenVinoAccelerator,
        event: &OpenVinoEvent,
    ) -> Result<(), BackendError> {
        let deadline = Instant::now() + Duration::from_secs(15);
        loop {
            match backend.poll_event(event)? {
                EventState::Pending => {
                    if Instant::now() >= deadline {
                        return Err(BackendError::DeadlineExpired);
                    }
                    std::thread::sleep(Duration::from_millis(1));
                }
                EventState::Complete => return Ok(()),
                EventState::Failed(error) => return Err(error),
                EventState::Cancelled => return Err(BackendError::Busy),
            }
        }
    }

    fn submission_path_diagnostics(
        &self,
        backend: &OpenVinoAccelerator,
    ) -> Option<SubmissionPathDiagnostics> {
        Some(SubmissionPathDiagnostics {
            direct_bindings: backend.direct_binding_admissions(),
            explicit_transfer_bytes: backend.explicit_transfer_bytes(),
            ..SubmissionPathDiagnostics::default()
        })
    }
}

fn conformance_target() -> TargetDescription {
    let input = float_bytes([13.5]);
    let program = ProgramFixture::new(
        virtio_accel_tosa::ARTIFACT_FORMAT,
        OPENVINO_TOSA_TARGET.to_identity(),
        IDENTITY_FP32_LOCAL,
        REQUIRED_RESIDENT_BYTES,
    )
    .unwrap();
    TargetDescription::with_bindings(
        program,
        vec![
            BindingFixture::read_only(0, MemoryDomain::Shared, BUFFER_ALIGNMENT, input.clone())
                .unwrap(),
            BindingFixture::new(
                1,
                AccessMode::Write,
                MemoryDomain::Shared,
                BUFFER_ALIGNMENT,
                vec![0; input.len()],
                input,
            )
            .unwrap(),
        ],
    )
    .unwrap()
}

#[test]
fn openvino_backend_passes_the_standard_semantic_suite() {
    if backend().is_none() {
        return;
    }
    let target = conformance_target();
    // `event.pending-release-terminal-stability` requires observing a pending event before the
    // completion hook runs; a sufficiently fast device can occasionally complete first. Retry
    // that specific precondition race a bounded number of times and fail on anything else.
    for attempt in 1..=5 {
        let report = run(|| OpenVinoAccelerator::new().unwrap(), &target, &Hooks);
        let racy_precondition = report.cases().iter().any(|case| {
            matches!(
                &case.status,
                CaseStatus::Failed(message)
                    if case.id == "event.pending-release-terminal-stability"
                        && message.contains("did not expose a controllable pending event")
            )
        });
        let other_failure = report.failures().any(|case| {
            !(case.id == "event.pending-release-terminal-stability"
                && matches!(&case.status, CaseStatus::Failed(message)
                    if message.contains("did not expose a controllable pending event")))
        });
        if other_failure || !racy_precondition {
            report.assert_conformant();
            return;
        }
        eprintln!(
            "attempt {attempt}: the device completed before the pending observation; retrying"
        );
    }
    panic!("the pending-event precondition raced on every attempt");
}

#[test]
#[ignore = "manual native performance evidence"]
fn measures_warm_submission_and_completion_latency() {
    let Some(backend) = backend() else { return };
    let context = backend.create_context(ContextDesc::default()).unwrap();
    let program = match load_tosa(&backend, &context, IDENTITY_FP32_LOCAL) {
        LoadOutcome::Loaded(program) => program,
        LoadOutcome::Rejected(error) => panic!("identity load failed: {error:?}"),
    };
    let queue = backend
        .create_queue(&context, QueueDesc::default())
        .unwrap();
    let desc = BufferDesc::new(
        4,
        BUFFER_ALIGNMENT,
        MemoryDomain::Shared,
        BufferUsage::TRANSFER_SOURCE
            | BufferUsage::TRANSFER_DESTINATION
            | BufferUsage::PROGRAM_INPUT
            | BufferUsage::PROGRAM_OUTPUT,
    )
    .unwrap();
    let (input, _) = backend
        .allocate_buffer(&context, desc)
        .unwrap()
        .into_parts();
    let (output, _) = backend
        .allocate_buffer(&context, desc)
        .unwrap()
        .into_parts();

    let submit_once = || {
        let bindings = [
            BindingRef {
                slot: 0,
                buffer: &input,
                range: BufferRange::new(0, 4).unwrap(),
                access: AccessMode::Read,
            },
            BindingRef {
                slot: 1,
                buffer: &output,
                range: BufferRange::new(0, 4).unwrap(),
                access: AccessMode::Write,
            },
        ];
        let started = Instant::now();
        let event = backend
            .submit(&queue, &program, &bindings, Timeout::Infinite)
            .unwrap_or_else(|_| panic!("warm submission rejected"));
        let deadline = started + Duration::from_secs(15);
        loop {
            match backend.poll_event(&event).unwrap() {
                EventState::Pending => {
                    assert!(Instant::now() < deadline, "inference never completed");
                    std::thread::yield_now();
                }
                EventState::Complete => break,
                state => panic!("unexpected terminal state {state:?}"),
            }
        }
        let elapsed = started.elapsed();
        backend.destroy_event(event).unwrap();
        elapsed
    };

    for _ in 0..20 {
        submit_once();
    }
    let mut samples = (0..200).map(|_| submit_once()).collect::<Vec<_>>();
    samples.sort_unstable();
    eprintln!(
        "device {}: warm submit-to-complete p50 {:?} p95 {:?} p99 {:?}",
        backend.device_name(),
        samples[samples.len() / 2],
        samples[samples.len() * 95 / 100],
        samples[samples.len() * 99 / 100],
    );

    backend.destroy_queue(queue).unwrap();
    backend.unload_program(program).unwrap();
    backend.free_buffer(input).unwrap();
    backend.free_buffer(output).unwrap();
    backend.destroy_context(context).unwrap();
}
