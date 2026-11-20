//! Integration and acceptance suite for the OpenVINO backend.
//!
//! Runs against every enumerated inference device. Without an OpenVINO runtime this file does
//! not compile (the crate builds its placeholder); with a runtime but no device, tests skip.

#![cfg(va_openvino)]

use std::time::{Duration, Instant};

use virtio_accel_conformance::numerics::{
    HEXAGON_LOGICAL_CASES, IDENTITY_EDGES_FP16, IDENTITY_EDGES_FP32, IDENTITY_FP8E4M3,
    IDENTITY_FP8E5M2, IDENTITY_INT4, IDENTITY_INT8, MATMUL_FP16, MATMUL_FP32, MATMUL_INT8,
    MAX_POOL2D_FP16, MAX_POOL2D_FP32, MUL_FP16,
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
    InitError, OPENVINO_TOSA_INTEGER_TARGET, OPENVINO_TOSA_TARGET, OpenVinoAccelerator,
    OpenVinoEvent, REQUIRED_RESIDENT_BYTES,
};
use virtio_accel_tosa::{Target, parse};

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
        Err(InitError::DeviceUnavailable | InitError::RuntimeUnavailable)
            if std::env::var("VIRTIO_ACCEL_OPENVINO_REQUIRE_HARDWARE").as_deref() != Ok("1")
                && std::env::var("VIRTIO_ACCEL_OPENVINO").as_deref() != Ok("1") =>
        {
            None
        }
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
    load_tosa_for_target(backend, context, artifact, OPENVINO_TOSA_TARGET)
}

fn load_tosa_for_target(
    backend: &OpenVinoAccelerator,
    context: &<OpenVinoAccelerator as Accelerator>::Context,
    artifact: &[u8],
    target: Target,
) -> LoadOutcome<OpenVinoAccelerator> {
    let model = parse(artifact).unwrap();
    let artifact = model.artifact_ref(target, REQUIRED_RESIDENT_BYTES).unwrap();
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
    run_tosa_bytes_for_target(device, artifact, OPENVINO_TOSA_TARGET, inputs, output_bytes)
}

fn run_tosa_bytes_for_target(
    device: &str,
    artifact: &[u8],
    target: Target,
    inputs: &[&[u8]],
    output_bytes: usize,
) -> Option<Vec<u8>> {
    let backend = OpenVinoAccelerator::with_device(device).unwrap();
    let context = backend.create_context(ContextDesc::default()).unwrap();
    let program = match load_tosa_for_target(&backend, &context, artifact, target) {
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
fn executes_boolean_boundaries_on_every_available_device() {
    for device in devices() {
        for case in HEXAGON_LOGICAL_CASES {
            eprintln!("executing {} on {device}", case.name);
            let encoded = case
                .inputs
                .iter()
                .map(|tensor| tensor.bytes())
                .collect::<Vec<_>>();
            let inputs = encoded.iter().map(Vec::as_slice).collect::<Vec<_>>();
            let Some(actual) =
                run_tosa_bytes(&device, case.artifact, &inputs, case.output.byte_len())
            else {
                continue;
            };
            assert!(
                case.output_matches(&actual),
                "{device}: {} produced {actual:?}",
                case.name
            );
        }
    }
}

#[test]
fn executes_float_mul_with_parameter_only_int8_shift_on_every_available_device() {
    for device in devices() {
        let inputs = MUL_FP16
            .inputs
            .iter()
            .map(|tensor| tensor.bits)
            .collect::<Vec<_>>();
        let Some(actual) = run_tosa_fp16(
            &device,
            MUL_FP16.artifact,
            &inputs,
            MUL_FP16.outputs[0].bits.len(),
        ) else {
            continue;
        };
        assert!(
            MUL_FP16.output_matches(0, &actual),
            "{device}: {} produced {actual:?}",
            MUL_FP16.name
        );
    }
}

#[test]
fn executes_the_shared_int8_corpus_on_every_available_device() {
    for device in devices() {
        let identity_input = IDENTITY_INT8.inputs[0].bytes;
        if let Some(actual) = run_tosa_bytes_for_target(
            &device,
            IDENTITY_INT8.artifact,
            OPENVINO_TOSA_INTEGER_TARGET,
            &[identity_input],
            IDENTITY_INT8.outputs[0].bytes.len(),
        ) {
            assert!(
                IDENTITY_INT8.output_matches(0, &actual),
                "{device}: {} produced {actual:?}",
                IDENTITY_INT8.name
            );
        }

        let inputs = MATMUL_INT8
            .inputs
            .iter()
            .map(|tensor| tensor.bytes)
            .collect::<Vec<_>>();
        let Some(actual) = run_tosa_bytes_for_target(
            &device,
            MATMUL_INT8.artifact,
            OPENVINO_TOSA_INTEGER_TARGET,
            &inputs,
            MATMUL_INT8.outputs[0].values.len().checked_mul(4).unwrap(),
        ) else {
            continue;
        };
        let actual = actual
            .chunks_exact(4)
            .map(|bytes| i32::from_ne_bytes(bytes.try_into().unwrap()))
            .collect::<Vec<_>>();
        assert!(
            MATMUL_INT8.output_matches(0, &actual),
            "{device}: {} produced {actual:?}",
            MATMUL_INT8.name
        );
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
    for device in devices() {
        measure_warm_latency_on(&device);
    }
}

fn measure_warm_latency_on(device: &str) {
    let backend = OpenVinoAccelerator::with_device(device).unwrap();
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
        let admission = started.elapsed();
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
        let completion = started.elapsed();
        backend.destroy_event(event).unwrap();
        (admission, completion)
    };

    for _ in 0..20 {
        submit_once();
    }
    let (mut admission, mut completion): (Vec<_>, Vec<_>) = (0..200).map(|_| submit_once()).unzip();
    admission.sort_unstable();
    completion.sort_unstable();
    eprintln!(
        "device {}: warm admission p50 {:?} p95 {:?} p99 {:?}; submit-to-complete p50 {:?} p95 {:?} p99 {:?}",
        backend.device_name(),
        admission[admission.len() / 2],
        admission[admission.len() * 95 / 100],
        admission[admission.len() * 99 / 100],
        completion[completion.len() / 2],
        completion[completion.len() * 95 / 100],
        completion[completion.len() * 99 / 100],
    );

    backend.destroy_queue(queue).unwrap();
    backend.unload_program(program).unwrap();
    backend.free_buffer(input).unwrap();
    backend.free_buffer(output).unwrap();
    backend.destroy_context(context).unwrap();
}

#[test]
fn nvfp4_projection_decodes_packed_weights_and_block_scales() {
    use virtio_accel_openvino::OPENVINO_TOSA_QUANTIZED_TARGET;
    let (m, n, k) = (2, 16, 32);
    let graph = virtio_accel_tosa_build::quantized::nvfp4_matmul(m, n, k, 0.125).unwrap();
    let weights = (0..=255u8).collect::<Vec<_>>();
    let scale_codes = [0u8, 1, 8, 0x38, 0x40, 0x7e];
    let scales = (0..n * k / 16)
        .map(|i| scale_codes[i % scale_codes.as_slice().len()])
        .collect::<Vec<_>>();
    let x = (0..m * k)
        .map(|i| (i as i32 % 9 - 4) as f32 * 0.25)
        .collect::<Vec<_>>();
    let bytes = x.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>();
    let lut = [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ];
    let mut expected = vec![0f32; m * n];
    for row in 0..m {
        for col in 0..n {
            for j in 0..k {
                let index = col * k + j;
                let nibble = (weights[index / 2] >> ((index % 2) * 4)) & 15;
                let scale = virtio_accel_tosa::fp8e4m3_to_f32(scales[index / 16]);
                expected[row * n + col] += x[row * k + j] * (lut[nibble as usize] * scale);
            }
            expected[row * n + col] *= 0.125;
        }
    }
    for device in devices() {
        use virtio_accel_tosa::{DType, TosaCapabilityProvider, ValueRoles};
        let backend = OpenVinoAccelerator::with_device(&device).unwrap();
        if !backend.tosa_capabilities().iter().any(|c| {
            c.target == OPENVINO_TOSA_QUANTIZED_TARGET
                && c.supports_dtype(DType::FP32, ValueRoles::INPUT)
        }) {
            eprintln!(
                "{device}: explicit FP32 inference policy unavailable; testing the FP16 contract separately"
            );
            continue;
        }
        eprintln!("NVFP4 projection on {device}");
        let out = run_tosa_bytes_for_target(
            &device,
            &graph,
            OPENVINO_TOSA_QUANTIZED_TARGET,
            &[&bytes, &weights, &scales],
            m * n * 4,
        )
        .expect("NVFP4 projection must execute on the selected device");
        for (i, b) in out.chunks_exact(4).enumerate() {
            let got = f32::from_le_bytes(b.try_into().unwrap());
            assert!(
                (got - expected[i]).abs() <= 0.0001,
                "{device} {i}: {got} != {}",
                expected[i]
            );
        }
    }
}

#[test]
fn nvfp4_fp16_projection_obeys_explicit_rounding() {
    let graph = virtio_accel_tosa_build::quantized::nvfp4_matmul_fp16(2, 16, 32, 0x3000).unwrap();
    let weights = (0..=255u8).collect::<Vec<_>>();
    let codes = [0u8, 1, 8, 0x38, 0x40, 0x7e];
    let scales = (0..32).map(|i| codes[i % 6]).collect::<Vec<_>>();
    let xb = [
        0xbc00u16, 0xba00, 0xb800, 0xb400, 0, 0x3400, 0x3800, 0x3a00, 0x3c00,
    ];
    let x = (0..64)
        .flat_map(|i| xb[i % 9].to_le_bytes())
        .collect::<Vec<_>>();
    // Independent binary64 dot products, rounded to binary16 before and after scale=1/8.
    // Generated with Python's struct.pack('<e'), including every E2M1 pattern and
    // zero/subnormal/normal/max-finite E4M3 block scales across row boundaries.
    let expected = [
        0xe00u16, 0x3104, 0x52fe, 0xa00, 0x2bb0, 0xb880, 0x8800, 0xb254, 0x5543, 0xf00, 0x340e,
        0x57e9, 0x1100, 0x361e, 0x5a2a, 0x1480, 0x9660, 0xba22, 0xdd24, 0x95a0, 0xb962, 0xdc44,
        0x9460, 0xb6c3, 0xdd94, 0x96a0, 0xbae2, 0xde3c, 0x9760, 0xbbe2, 0xdf54, 0x98b0,
    ];
    for device in devices() {
        eprintln!("NVFP4 FP16 projection on {device}");
        let out = run_tosa_bytes_for_target(
            &device,
            &graph,
            virtio_accel_openvino::OPENVINO_TOSA_QUANTIZED_TARGET,
            &[&x, &weights, &scales],
            64,
        )
        .expect("explicit FP16 projection must execute");
        for (i, b) in out.chunks_exact(2).enumerate() {
            let bits = u16::from_le_bytes(b.try_into().unwrap());
            assert_eq!(bits, expected[i], "FP16 output {i}");
        }
    }
}

fn fp8_cast_graph(
    input: virtio_accel_tosa::DType,
    output: virtio_accel_tosa::DType,
    count: usize,
) -> Vec<u8> {
    use virtio_accel_tosa_build::{Graph, Operator, OperatorKind, Tensor};
    let shape = [1, count as i32];
    let tensors = [
        Tensor::new("x", &shape, input),
        Tensor::new("y", &shape, output),
    ];
    Graph::new(
        "main",
        &tensors,
        &[Operator::new(OperatorKind::Cast, &["x"], &["y"])],
        &["x"],
        &["y"],
    )
    .build(virtio_accel_openvino::OPENVINO_TOSA_FP8_TARGET)
    .unwrap()
}

#[test]
fn fp8_packed_casts_preserve_values_and_round_ties() {
    use virtio_accel_openvino::OPENVINO_TOSA_FP8_TARGET as target;
    use virtio_accel_tosa::{DType, TosaCapabilityProvider, fp8e4m3_to_f32, fp8e5m2_to_f32};
    for device in devices() {
        let backend = OpenVinoAccelerator::with_device(&device).unwrap();
        if !backend
            .tosa_capabilities()
            .iter()
            .any(|c| c.target == target)
        {
            continue;
        }
        for (dtype, decode) in [
            (DType::FP8E4M3, fp8e4m3_to_f32 as fn(u8) -> f32),
            (DType::FP8E5M2, fp8e5m2_to_f32 as fn(u8) -> f32),
        ] {
            let input = (0..=255u8).collect::<Vec<_>>();
            let graph = fp8_cast_graph(dtype, DType::FP32, 256);
            let bytes =
                run_tosa_bytes_for_target(&device, &graph, target, &[&input], 1024).unwrap();
            for (i, b) in bytes.chunks_exact(4).enumerate() {
                let got = f32::from_le_bytes(b.try_into().unwrap());
                let want = decode(i as u8);
                if want.is_nan() {
                    assert!(got.is_nan(), "{device} {dtype:?} {i}")
                } else {
                    assert_eq!(got.to_bits(), want.to_bits(), "{device} {dtype:?} {i}");
                }
            }
            let mut values = Vec::new();
            let mut expected = Vec::new();
            for i in 0..=255u8 {
                let f = decode(i);
                if !f.is_nan() {
                    values.push(f);
                    expected.push(i);
                }
            }
            let max = if dtype == DType::FP8E4M3 {
                126u8
            } else {
                123u8
            };
            for i in 0..max {
                let midpoint = (decode(i) + decode(i + 1)) * 0.5;
                for (f, e) in [
                    (f32::from_bits(midpoint.to_bits() - 1), i),
                    (midpoint, if i % 2 == 0 { i } else { i + 1 }),
                    (f32::from_bits(midpoint.to_bits() + 1), i + 1),
                ] {
                    values.push(f);
                    expected.push(e);
                    values.push(-f);
                    expected.push(e | 128);
                }
            }
            let edges: &[(f32, u8)] = if dtype == DType::FP8E4M3 {
                &[
                    (464.0, 126),
                    (f32::from_bits(464.0f32.to_bits() + 1), 127),
                    (-f32::from_bits(464.0f32.to_bits() + 1), 255),
                    (f32::INFINITY, 127),
                    (f32::NEG_INFINITY, 255),
                    (f32::NAN, 127),
                ]
            } else {
                &[
                    (61440.0, 124),
                    (f32::from_bits(61440.0f32.to_bits() - 1), 123),
                    (f32::MAX, 124),
                    (f32::INFINITY, 124),
                    (f32::NEG_INFINITY, 252),
                    (f32::NAN, 126),
                ]
            };
            for &(value, code) in edges {
                values.push(value);
                expected.push(code);
            }
            let input = values
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>();
            let graph = fp8_cast_graph(DType::FP32, dtype, values.len());
            let got = run_tosa_bytes_for_target(&device, &graph, target, &[&input], values.len())
                .unwrap();
            for (i, (got, want)) in got.iter().zip(&expected).enumerate() {
                assert_eq!(got, want, "{device} {dtype:?} value {:?} at {i}", values[i]);
            }
        }
    }
}

#[test]
fn fp8_cache_append_and_window_preserve_every_storage_bit() {
    use virtio_accel_tosa::{DType, TosaCapabilityProvider};
    use virtio_accel_tosa_build::{Graph, Operator, OperatorKind, Shape, Tensor};
    for dtype in [DType::FP8E4M3, DType::FP8E5M2] {
        let tensors = [
            Tensor::new("cache", &[1, 128], dtype),
            Tensor::new("append", &[1, 128], dtype),
            Tensor::new("joined", &[1, 256], dtype),
            Tensor::new("window", &[1, 128], dtype),
        ];
        let shapes = [Shape::new("start", &[0, 64]), Shape::new("size", &[1, 128])];
        let ops = [
            Operator::new(OperatorKind::ConstShape, &[], &["start"]),
            Operator::new(OperatorKind::ConstShape, &[], &["size"]),
            Operator::new(
                OperatorKind::Concat { axis: 1 },
                &["cache", "append"],
                &["joined"],
            ),
            Operator::new(
                OperatorKind::Slice,
                &["joined", "start", "size"],
                &["window"],
            ),
        ];
        let target = virtio_accel_openvino::OPENVINO_TOSA_FP8_TARGET;
        let graph = Graph::new(
            "kv_window",
            &tensors,
            &ops,
            &["cache", "append"],
            &["window"],
        )
        .with_shapes(&shapes)
        .build(target)
        .unwrap();
        let bytes = (0..=255u8).collect::<Vec<_>>();
        for device in devices() {
            let backend = OpenVinoAccelerator::with_device(&device).unwrap();
            if !backend
                .tosa_capabilities()
                .iter()
                .any(|c| c.target == target)
            {
                continue;
            }
            let out = run_tosa_bytes_for_target(
                &device,
                &graph,
                target,
                &[&bytes[..128], &bytes[128..]],
                128,
            )
            .unwrap();
            assert_eq!(out, &bytes[64..192]);
        }
    }
}

fn fp8_constant_projection_graph(dtype: virtio_accel_tosa::DType) -> Vec<u8> {
    use virtio_accel_tosa::DType;
    use virtio_accel_tosa_build::{Graph, Operator, OperatorKind, Tensor};
    let weights = if dtype == DType::FP8E4M3 {
        [0x38, 0x40, 0x44, 0x48, 0xb8, 0xc0, 0x30, 0x38]
    } else {
        [0x3c, 0x40, 0x42, 0x44, 0xbc, 0xc0, 0x38, 0x3c]
    };
    let tensors = [
        Tensor::new("input", &[1, 1, 4], DType::FP32),
        Tensor::new("activation", &[1, 1, 4], dtype),
        Tensor::constant("weights", &[1, 4, 2], dtype, &weights),
        Tensor::constant("zero", &[1], dtype, &[0]),
        Tensor::new("out", &[1, 1, 2], DType::FP16),
    ];
    let ops = [
        Operator::new(OperatorKind::Const, &[], &["weights"]),
        Operator::new(OperatorKind::Const, &[], &["zero"]),
        Operator::new(OperatorKind::Cast, &["input"], &["activation"]),
        Operator::new(
            OperatorKind::MatMul,
            &["activation", "weights", "zero", "zero"],
            &["out"],
        ),
    ];
    Graph::new("projection", &tensors, &ops, &["input"], &["out"])
        .build(virtio_accel_openvino::OPENVINO_TOSA_FP8_TARGET)
        .unwrap()
}

#[test]
fn fp8_activation_and_constant_weights_execute_together() {
    use virtio_accel_openvino::OPENVINO_TOSA_FP8_TARGET;
    use virtio_accel_tosa::{DType, TosaCapabilityProvider};
    let input = [1.0f32, 2.0, -1.0, 0.5]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect::<Vec<_>>();
    for device in devices() {
        let backend = OpenVinoAccelerator::with_device(&device).unwrap();
        if !backend
            .tosa_capabilities()
            .iter()
            .any(|c| c.target == OPENVINO_TOSA_FP8_TARGET)
        {
            continue;
        }
        for dtype in [DType::FP8E4M3, DType::FP8E5M2] {
            let graph = fp8_constant_projection_graph(dtype);
            let out =
                run_tosa_bytes_for_target(&device, &graph, OPENVINO_TOSA_FP8_TARGET, &[&input], 4)
                    .unwrap();
            assert_eq!(out, [0x20, 0x48, 0x40, 0x4a], "{device} {dtype:?}");
        }
    }
}
