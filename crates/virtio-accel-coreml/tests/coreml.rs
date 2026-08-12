#![cfg(target_os = "macos")]

use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};
use virtio_accel_conformance::{
    BindingFixture, ConformanceHooks, ProgramFixture, SubmissionPathDiagnostics, TargetDescription,
    run,
};
use virtio_accel_core::{
    Accelerator, AccessMode, ArtifactRef, BackendError, BindingRef, BufferDesc, BufferRange,
    BufferUsage, ByteSource, ContextDesc, EventState, MemoryDomain, QueueDesc, Timeout,
};
use virtio_accel_coreml::{
    ARTIFACT_FORMAT, COREML_TOSA_TARGET, CoreMlAccelerator, CoreMlArtifact, CoreMlEvent, InitError,
    REQUIRED_RESIDENT_BYTES, TARGET_IDENTITY,
};
use virtio_accel_tosa::parse;

// Core ML protobuf for one Float32[8] neural network: y = 2*x + 1.
const MODEL: &[u8] = &[
    0x08, 0x01, 0x12, 0x20, 0x0a, 0x0e, 0x0a, 0x01, 0x78, 0x1a, 0x09, 0x2a, 0x07, 0x0a, 0x01, 0x08,
    0x10, 0xa0, 0x80, 0x04, 0x52, 0x0e, 0x0a, 0x01, 0x79, 0x1a, 0x09, 0x2a, 0x07, 0x0a, 0x01, 0x08,
    0x10, 0xa0, 0x80, 0x04, 0xa2, 0x1f, 0x27, 0x0a, 0x25, 0x0a, 0x0e, 0x74, 0x77, 0x69, 0x63, 0x65,
    0x5f, 0x70, 0x6c, 0x75, 0x73, 0x5f, 0x6f, 0x6e, 0x65, 0x12, 0x01, 0x78, 0x1a, 0x01, 0x79, 0x92,
    0x08, 0x0c, 0x2a, 0x0a, 0x0d, 0x00, 0x00, 0x00, 0x40, 0x15, 0x00, 0x00, 0x80, 0x3f,
];

const TOSA_IDENTITY: &[u8] = include_bytes!("data/identity-fp32-v1.0.0.tosa");

static NEXT_FIXTURE: AtomicU64 = AtomicU64::new(1);
#[derive(Debug)]
struct SliceSource<'a>(&'a [u8]);

impl ByteSource for SliceSource<'_> {
    fn len(&self) -> u64 {
        self.0.len() as u64
    }

    fn read_at(&self, offset: u64, target: &mut [u8]) -> Result<(), BackendError> {
        let start = usize::try_from(offset).map_err(|_| BackendError::OutOfBounds)?;
        let end = start
            .checked_add(target.len())
            .filter(|end| *end <= self.0.len())
            .ok_or(BackendError::OutOfBounds)?;
        target.copy_from_slice(&self.0[start..end]);
        Ok(())
    }

    fn as_contiguous(&self) -> Option<&[u8]> {
        Some(self.0)
    }
}

struct Fixture {
    root: PathBuf,
    artifact: Vec<u8>,
}

impl Fixture {
    fn new() -> Self {
        let unique = NEXT_FIXTURE.fetch_add(1, Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!(
            "virtio-accel-coreml-{}-{unique}",
            std::process::id()
        ));
        fs::create_dir(&root).unwrap();
        fs::write(root.join("TwicePlusOne.mlmodel"), MODEL).unwrap();
        let artifact = CoreMlArtifact::new("TwicePlusOne.mlmodel")
            .unwrap()
            .map_input(7, "x")
            .unwrap()
            .map_output(7, "y")
            .unwrap()
            .encode()
            .unwrap();
        Self { root, artifact }
    }

    fn backend(&self) -> Result<CoreMlAccelerator, InitError> {
        CoreMlAccelerator::new(&self.root)
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.root).unwrap();
    }
}

fn float_bytes(values: impl IntoIterator<Item = f32>) -> Vec<u8> {
    values
        .into_iter()
        .flat_map(f32::to_ne_bytes)
        .collect::<Vec<_>>()
}

fn split_artifact() -> Vec<u8> {
    CoreMlArtifact::new("TwicePlusOne.mlmodel")
        .unwrap()
        .map_input(7, "x")
        .unwrap()
        .map_output(8, "y")
        .unwrap()
        .encode()
        .unwrap()
}

fn wait_for_terminal(
    backend: &CoreMlAccelerator,
    event: &CoreMlEvent,
) -> Result<EventState, BackendError> {
    let deadline = Instant::now() + Duration::from_secs(15);
    loop {
        match backend.poll_event(event)? {
            EventState::Pending if Instant::now() < deadline => std::thread::yield_now(),
            EventState::Pending => return Err(BackendError::DeadlineExpired),
            terminal => return Ok(terminal),
        }
    }
}

#[test]
fn lowers_and_executes_device_neutral_tosa() {
    let backend = match CoreMlAccelerator::new_tosa() {
        Ok(backend) => backend,
        Err(InitError::NeuralEngineUnavailable) => return,
        Err(error) => panic!("Core ML backend initialization failed: {error}"),
    };
    let context = backend.create_context(ContextDesc::default()).unwrap();
    let desc = |usage| BufferDesc::new(4, 16 * 1024, MemoryDomain::Shared, usage).unwrap();
    let (mut input, _) = backend
        .allocate_buffer(
            &context,
            desc(BufferUsage::TRANSFER_DESTINATION | BufferUsage::PROGRAM_INPUT),
        )
        .unwrap()
        .into_parts();
    let (output, _) = backend
        .allocate_buffer(
            &context,
            desc(BufferUsage::TRANSFER_SOURCE | BufferUsage::PROGRAM_OUTPUT),
        )
        .unwrap()
        .into_parts();
    let expected = 3.25_f32.to_ne_bytes();
    backend
        .write_buffer(&mut input, 0, &SliceSource(&expected))
        .unwrap();

    let model = parse(TOSA_IDENTITY).unwrap();
    let program = backend
        .load_program(
            &context,
            model
                .artifact_ref(COREML_TOSA_TARGET, REQUIRED_RESIDENT_BYTES)
                .unwrap(),
        )
        .unwrap();
    let queue = backend
        .create_queue(&context, QueueDesc::default())
        .unwrap();
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
    let event = backend
        .submit(&queue, &program, &bindings, Timeout::Infinite)
        .unwrap();
    assert_eq!(
        wait_for_terminal(&backend, &event).unwrap(),
        EventState::Complete
    );
    backend.destroy_event(event).unwrap();

    let mut actual = [0; 4];
    backend.read_buffer(&output, 0, &mut actual).unwrap();
    assert_eq!(actual, expected);
    assert_eq!(backend.direct_binding_admissions(), 2);

    backend.destroy_queue(queue).unwrap();
    backend.unload_program(program).unwrap();
    backend.free_buffer(output).unwrap();
    backend.free_buffer(input).unwrap();
    backend.destroy_context(context).unwrap();
}

#[test]
fn executes_a_coreml_model_with_exact_shared_backing() {
    let fixture = Fixture::new();
    let backend = match fixture.backend() {
        Ok(backend) => backend,
        Err(InitError::NeuralEngineUnavailable) => return,
        Err(error) => panic!("Core ML backend initialization failed: {error}"),
    };
    let context = backend.create_context(ContextDesc::default()).unwrap();
    let initial = float_bytes((0..8).map(|value| value as f32));
    let expected = float_bytes((0..8).map(|value| value as f32 * 2.0 + 1.0));
    let desc = BufferDesc::new(
        initial.len() as u64,
        16 * 1024,
        MemoryDomain::Shared,
        BufferUsage::TRANSFER_SOURCE
            | BufferUsage::TRANSFER_DESTINATION
            | BufferUsage::MUTABLE_STATE,
    )
    .unwrap();
    let (mut buffer, info) = backend
        .allocate_buffer(&context, desc)
        .unwrap()
        .into_parts();
    assert!(
        info.properties()
            .contains(virtio_accel_core::BufferProperties::DIRECT_BINDING)
    );
    backend
        .write_buffer(&mut buffer, 0, &SliceSource(&initial))
        .unwrap();

    let artifact_source = SliceSource(&fixture.artifact);
    let program = backend
        .load_program(
            &context,
            ArtifactRef {
                format: ARTIFACT_FORMAT,
                target: TARGET_IDENTITY,
                payload: &artifact_source,
                resident_bytes: REQUIRED_RESIDENT_BYTES,
            },
        )
        .unwrap();
    let queue = backend
        .create_queue(&context, QueueDesc::default())
        .unwrap();
    let event = backend
        .submit(
            &queue,
            &program,
            &[BindingRef {
                slot: 7,
                buffer: &buffer,
                range: BufferRange::new(0, initial.len() as u64).unwrap(),
                access: AccessMode::ReadWrite,
            }],
            Timeout::Infinite,
        )
        .unwrap();
    assert_eq!(
        wait_for_terminal(&backend, &event).unwrap(),
        EventState::Complete
    );
    assert_eq!(backend.direct_binding_admissions(), 1);

    let mut output = [0; 32];
    backend.read_buffer(&buffer, 0, &mut output).unwrap();
    assert_eq!(output.as_slice(), expected);

    backend.destroy_event(event).unwrap();
    backend.destroy_queue(queue).unwrap();
    backend.unload_program(program).unwrap();
    backend.free_buffer(buffer).unwrap();
    backend.destroy_context(context).unwrap();
}

#[test]
fn accepts_unsorted_distinct_input_and_output_bindings() {
    let fixture = Fixture::new();
    let backend = match fixture.backend() {
        Ok(backend) => backend,
        Err(InitError::NeuralEngineUnavailable) => return,
        Err(error) => panic!("Core ML backend initialization failed: {error}"),
    };
    let context = backend.create_context(ContextDesc::default()).unwrap();
    let input_bytes = float_bytes((0..8).map(|value| value as f32));
    let expected = float_bytes((0..8).map(|value| value as f32 * 2.0 + 1.0));
    let input_desc = BufferDesc::new(
        input_bytes.len() as u64,
        16 * 1024,
        MemoryDomain::Shared,
        BufferUsage::TRANSFER_DESTINATION | BufferUsage::PROGRAM_INPUT,
    )
    .unwrap();
    let output_desc = BufferDesc::new(
        expected.len() as u64,
        16 * 1024,
        MemoryDomain::Shared,
        BufferUsage::TRANSFER_SOURCE | BufferUsage::PROGRAM_OUTPUT,
    )
    .unwrap();
    let (mut input, _) = backend
        .allocate_buffer(&context, input_desc)
        .unwrap()
        .into_parts();
    let (output, _) = backend
        .allocate_buffer(&context, output_desc)
        .unwrap()
        .into_parts();
    backend
        .write_buffer(&mut input, 0, &SliceSource(&input_bytes))
        .unwrap();

    let artifact = split_artifact();
    let artifact_source = SliceSource(&artifact);
    let program = backend
        .load_program(
            &context,
            ArtifactRef {
                format: ARTIFACT_FORMAT,
                target: TARGET_IDENTITY,
                payload: &artifact_source,
                resident_bytes: REQUIRED_RESIDENT_BYTES,
            },
        )
        .unwrap();
    let queue = backend
        .create_queue(&context, QueueDesc::default())
        .unwrap();
    let bindings = [
        BindingRef {
            slot: 8,
            buffer: &output,
            range: BufferRange::new(0, expected.len() as u64).unwrap(),
            access: AccessMode::Write,
        },
        BindingRef {
            slot: 7,
            buffer: &input,
            range: BufferRange::new(0, input_bytes.len() as u64).unwrap(),
            access: AccessMode::Read,
        },
    ];
    let event = backend
        .submit(&queue, &program, &bindings, Timeout::Infinite)
        .unwrap();
    assert_eq!(
        wait_for_terminal(&backend, &event).unwrap(),
        EventState::Complete
    );
    backend.destroy_event(event).unwrap();

    let mut actual = [0; 32];
    backend.read_buffer(&output, 0, &mut actual).unwrap();
    assert_eq!(actual.as_slice(), expected);
    assert_eq!(backend.direct_binding_admissions(), 2);
    assert_eq!(backend.explicit_transfer_bytes(), 64);

    backend.destroy_queue(queue).unwrap();
    backend.unload_program(program).unwrap();
    backend.free_buffer(output).unwrap();
    backend.free_buffer(input).unwrap();
    backend.destroy_context(context).unwrap();
}

#[test]
fn rejects_misaligned_tensor_bindings_before_admission() {
    let fixture = Fixture::new();
    let backend = match fixture.backend() {
        Ok(backend) => backend,
        Err(InitError::NeuralEngineUnavailable) => return,
        Err(error) => panic!("Core ML backend initialization failed: {error}"),
    };
    let context = backend.create_context(ContextDesc::default()).unwrap();
    let input_bytes = float_bytes((0..8).map(|value| value as f32));
    let input_desc = BufferDesc::new(
        input_bytes.len() as u64 + 1,
        16 * 1024,
        MemoryDomain::Shared,
        BufferUsage::TRANSFER_DESTINATION | BufferUsage::PROGRAM_INPUT,
    )
    .unwrap();
    let output_desc = BufferDesc::new(
        input_bytes.len() as u64,
        16 * 1024,
        MemoryDomain::Shared,
        BufferUsage::PROGRAM_OUTPUT,
    )
    .unwrap();
    let (mut input, _) = backend
        .allocate_buffer(&context, input_desc)
        .unwrap()
        .into_parts();
    let (output, _) = backend
        .allocate_buffer(&context, output_desc)
        .unwrap()
        .into_parts();
    backend
        .write_buffer(&mut input, 1, &SliceSource(&input_bytes))
        .unwrap();
    let artifact = split_artifact();
    let artifact_source = SliceSource(&artifact);
    let program = backend
        .load_program(
            &context,
            ArtifactRef {
                format: ARTIFACT_FORMAT,
                target: TARGET_IDENTITY,
                payload: &artifact_source,
                resident_bytes: REQUIRED_RESIDENT_BYTES,
            },
        )
        .unwrap();
    let queue = backend
        .create_queue(&context, QueueDesc::default())
        .unwrap();
    let bindings = [
        BindingRef {
            slot: 7,
            buffer: &input,
            range: BufferRange::new(1, input_bytes.len() as u64).unwrap(),
            access: AccessMode::Read,
        },
        BindingRef {
            slot: 8,
            buffer: &output,
            range: BufferRange::new(0, input_bytes.len() as u64).unwrap(),
            access: AccessMode::Write,
        },
    ];
    match backend.submit(&queue, &program, &bindings, Timeout::Infinite) {
        Err(virtio_accel_core::SubmitFailure::Rejected(BackendError::Incompatible)) => {}
        result => panic!("misaligned binding was not rejected: {result:?}"),
    }

    backend.destroy_queue(queue).unwrap();
    backend.unload_program(program).unwrap();
    backend.free_buffer(output).unwrap();
    backend.free_buffer(input).unwrap();
    backend.destroy_context(context).unwrap();
}

struct Hooks;

impl ConformanceHooks<CoreMlAccelerator> for Hooks {
    fn complete_event(
        &self,
        backend: &CoreMlAccelerator,
        event: &CoreMlEvent,
    ) -> Result<(), BackendError> {
        match wait_for_terminal(backend, event)? {
            EventState::Complete => Ok(()),
            EventState::Failed(error) => Err(error),
            EventState::Cancelled => Err(BackendError::DeviceLost),
            EventState::Pending => Err(BackendError::DeadlineExpired),
        }
    }

    fn submission_path_diagnostics(
        &self,
        backend: &CoreMlAccelerator,
    ) -> Option<SubmissionPathDiagnostics> {
        Some(SubmissionPathDiagnostics {
            direct_bindings: backend.direct_binding_admissions(),
            explicit_transfer_bytes: backend.explicit_transfer_bytes(),
            ..SubmissionPathDiagnostics::default()
        })
    }
}

fn target(artifact: &[u8]) -> TargetDescription {
    TargetDescription::new(
        ProgramFixture::new(
            ARTIFACT_FORMAT,
            TARGET_IDENTITY,
            artifact,
            REQUIRED_RESIDENT_BYTES,
        )
        .unwrap(),
        BindingFixture::new(
            7,
            AccessMode::ReadWrite,
            MemoryDomain::Shared,
            16 * 1024,
            float_bytes((0..8).map(|value| value as f32)),
            float_bytes((0..8).map(|value| value as f32 * 2.0 + 1.0)),
        )
        .unwrap(),
    )
}

#[test]
fn coreml_backend_passes_the_standard_semantic_suite() {
    let fixture = Fixture::new();
    if matches!(fixture.backend(), Err(InitError::NeuralEngineUnavailable)) {
        return;
    }
    let report = run(
        || fixture.backend().unwrap(),
        &target(&fixture.artifact),
        &Hooks,
    );
    report.assert_conformant();
}

#[test]
fn model_paths_cannot_escape_the_host_root() {
    let fixture = Fixture::new();
    let backend = match fixture.backend() {
        Ok(backend) => backend,
        Err(InitError::NeuralEngineUnavailable) => return,
        Err(error) => panic!("Core ML backend initialization failed: {error}"),
    };
    let context = backend.create_context(ContextDesc::default()).unwrap();
    let escaped = CoreMlArtifact::new("../TwicePlusOne.mlmodel")
        .unwrap()
        .map_input(7, "x")
        .unwrap()
        .map_output(7, "y")
        .unwrap()
        .encode()
        .unwrap();
    let escaped_source = SliceSource(&escaped);
    assert!(matches!(
        backend.load_program(
            &context,
            ArtifactRef {
                format: ARTIFACT_FORMAT,
                target: TARGET_IDENTITY,
                payload: &escaped_source,
                resident_bytes: REQUIRED_RESIDENT_BYTES,
            },
        ),
        Err(BackendError::PermissionDenied)
    ));
    backend.destroy_context(context).unwrap();
}

#[test]
fn nonmaximal_resident_charge_is_rejected_before_native_loading() {
    let fixture = Fixture::new();
    let backend = match fixture.backend() {
        Ok(backend) => backend,
        Err(InitError::NeuralEngineUnavailable) => return,
        Err(error) => panic!("Core ML backend initialization failed: {error}"),
    };
    let context = backend.create_context(ContextDesc::default()).unwrap();
    let artifact_source = SliceSource(&fixture.artifact);
    assert!(matches!(
        backend.load_program(
            &context,
            ArtifactRef {
                format: ARTIFACT_FORMAT,
                target: TARGET_IDENTITY,
                payload: &artifact_source,
                resident_bytes: u64::MAX - 1,
            },
        ),
        Err(BackendError::ResourceLimit)
    ));
    backend.destroy_context(context).unwrap();
}

#[test]
#[ignore = "manual native performance evidence"]
fn measures_warm_submission_and_completion_latency() {
    const WARMUPS: usize = 20;
    const ITERATIONS: usize = 200;

    let fixture = Fixture::new();
    let backend = match fixture.backend() {
        Ok(backend) => backend,
        Err(InitError::NeuralEngineUnavailable) => return,
        Err(error) => panic!("Core ML backend initialization failed: {error}"),
    };
    let context = backend.create_context(ContextDesc::default()).unwrap();
    let input_bytes = float_bytes((0..8).map(|value| value as f32));
    let expected = float_bytes((0..8).map(|value| value as f32 * 2.0 + 1.0));
    let input_desc = BufferDesc::new(
        input_bytes.len() as u64,
        16 * 1024,
        MemoryDomain::Shared,
        BufferUsage::TRANSFER_DESTINATION | BufferUsage::PROGRAM_INPUT,
    )
    .unwrap();
    let output_desc = BufferDesc::new(
        expected.len() as u64,
        16 * 1024,
        MemoryDomain::Shared,
        BufferUsage::TRANSFER_SOURCE | BufferUsage::PROGRAM_OUTPUT,
    )
    .unwrap();
    let (mut input, _) = backend
        .allocate_buffer(&context, input_desc)
        .unwrap()
        .into_parts();
    let (output, _) = backend
        .allocate_buffer(&context, output_desc)
        .unwrap()
        .into_parts();
    backend
        .write_buffer(&mut input, 0, &SliceSource(&input_bytes))
        .unwrap();

    let artifact = split_artifact();
    let artifact_source = SliceSource(&artifact);
    let program = backend
        .load_program(
            &context,
            ArtifactRef {
                format: ARTIFACT_FORMAT,
                target: TARGET_IDENTITY,
                payload: &artifact_source,
                resident_bytes: REQUIRED_RESIDENT_BYTES,
            },
        )
        .unwrap();
    let queue = backend
        .create_queue(&context, QueueDesc::default())
        .unwrap();
    let bindings = [
        BindingRef {
            slot: 7,
            buffer: &input,
            range: BufferRange::new(0, input_bytes.len() as u64).unwrap(),
            access: AccessMode::Read,
        },
        BindingRef {
            slot: 8,
            buffer: &output,
            range: BufferRange::new(0, expected.len() as u64).unwrap(),
            access: AccessMode::Write,
        },
    ];

    let mut admission = Vec::with_capacity(ITERATIONS);
    let mut completion = Vec::with_capacity(ITERATIONS);
    for iteration in 0..WARMUPS + ITERATIONS {
        let started = Instant::now();
        let event = backend
            .submit(&queue, &program, &bindings, Timeout::Infinite)
            .unwrap();
        let admitted = Instant::now();
        assert_eq!(
            wait_for_terminal(&backend, &event).unwrap(),
            EventState::Complete
        );
        let completed = Instant::now();
        backend.destroy_event(event).unwrap();
        if iteration >= WARMUPS {
            admission.push(admitted.duration_since(started));
            completion.push(completed.duration_since(started));
        }
    }
    admission.sort_unstable();
    completion.sort_unstable();
    let percentile = |samples: &[Duration], numerator: usize, denominator: usize| {
        samples[(samples.len() - 1) * numerator / denominator]
    };
    eprintln!(
        "Core ML warm path ({ITERATIONS} iterations): admission median={:?} p95={:?}; completion median={:?} p95={:?}",
        percentile(&admission, 1, 2),
        percentile(&admission, 95, 100),
        percentile(&completion, 1, 2),
        percentile(&completion, 95, 100),
    );

    let mut actual = [0; 32];
    backend.read_buffer(&output, 0, &mut actual).unwrap();
    assert_eq!(actual.as_slice(), expected);
    backend.destroy_queue(queue).unwrap();
    backend.unload_program(program).unwrap();
    backend.free_buffer(output).unwrap();
    backend.free_buffer(input).unwrap();
    backend.destroy_context(context).unwrap();
}
