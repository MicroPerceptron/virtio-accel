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
    ARTIFACT_FORMAT, CoreMlAccelerator, CoreMlArtifact, CoreMlEvent, InitError,
    REQUIRED_RESIDENT_BYTES, TARGET_IDENTITY,
};

// Core ML protobuf for one Float32[8] neural network: y = 2*x + 1.
const MODEL: &[u8] = &[
    0x08, 0x01, 0x12, 0x20, 0x0a, 0x0e, 0x0a, 0x01, 0x78, 0x1a, 0x09, 0x2a, 0x07, 0x0a, 0x01, 0x08,
    0x10, 0xa0, 0x80, 0x04, 0x52, 0x0e, 0x0a, 0x01, 0x79, 0x1a, 0x09, 0x2a, 0x07, 0x0a, 0x01, 0x08,
    0x10, 0xa0, 0x80, 0x04, 0xa2, 0x1f, 0x27, 0x0a, 0x25, 0x0a, 0x0e, 0x74, 0x77, 0x69, 0x63, 0x65,
    0x5f, 0x70, 0x6c, 0x75, 0x73, 0x5f, 0x6f, 0x6e, 0x65, 0x12, 0x01, 0x78, 0x1a, 0x01, 0x79, 0x92,
    0x08, 0x0c, 0x2a, 0x0a, 0x0d, 0x00, 0x00, 0x00, 0x40, 0x15, 0x00, 0x00, 0x80, 0x3f,
];

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
