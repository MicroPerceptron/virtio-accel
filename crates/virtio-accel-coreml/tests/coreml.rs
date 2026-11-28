#![cfg(target_os = "macos")]

use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};
use virtio_accel_conformance::numerics::{
    IDENTITY_EDGES_FP16, IDENTITY_EDGES_FP32, IDENTITY_INT8, MATMUL_FP16, MATMUL_FP32, MATMUL_INT8,
    MAX_POOL2D_FP16, MAX_POOL2D_FP32,
};
use virtio_accel_conformance::{
    BindingFixture, ConformanceHooks, ProgramFixture, SubmissionPathDiagnostics, TargetDescription,
    run,
};
use virtio_accel_core::{
    Accelerator, AccessMode, ArtifactRef, BackendError, BindingRef, BufferDesc, BufferRange,
    BufferUsage, ByteSink, ByteSource, ContextDesc, EventState, MemoryDomain, QueueDesc, Timeout,
};
use virtio_accel_coreml::{
    ARTIFACT_FORMAT, COREML_TOSA_INTEGER_TARGET, COREML_TOSA_TARGET, CoreMlAccelerator,
    CoreMlArtifact, CoreMlComputeUnits, CoreMlEvent, InitError, REQUIRED_RESIDENT_BYTES,
    TARGET_IDENTITY,
};
use virtio_accel_tosa::{Target, parse};

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

#[derive(Debug)]
struct VecSink(Vec<u8>);

impl ByteSink for VecSink {
    fn len(&self) -> u64 {
        self.0.len() as u64
    }

    fn write_at(&mut self, offset: u64, source: &[u8]) -> Result<(), BackendError> {
        let start = usize::try_from(offset).map_err(|_| BackendError::OutOfBounds)?;
        let end = start
            .checked_add(source.len())
            .filter(|end| *end <= self.0.len())
            .ok_or(BackendError::OutOfBounds)?;
        self.0[start..end].copy_from_slice(source);
        Ok(())
    }

    fn as_contiguous_mut(&mut self) -> Option<&mut [u8]> {
        Some(self.0.as_mut_slice())
    }
}

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

fn run_tosa_bytes(
    artifact: &[u8],
    target: Target,
    inputs: &[&[u8]],
    output_bytes: usize,
) -> Option<Vec<u8>> {
    let backend = match CoreMlAccelerator::new_tosa() {
        Ok(backend) => backend,
        Err(InitError::NeuralEngineUnavailable) => return None,
        Err(error) => panic!("Core ML backend initialization failed: {error}"),
    };
    run_tosa_bytes_on(&backend, artifact, target, inputs, output_bytes)
}

fn run_tosa_bytes_on(
    backend: &CoreMlAccelerator,
    artifact: &[u8],
    target: Target,
    inputs: &[&[u8]],
    output_bytes: usize,
) -> Option<Vec<u8>> {
    let context = backend.create_context(ContextDesc::default()).unwrap();
    let mut input_buffers = Vec::with_capacity(inputs.len());
    for bytes in inputs {
        let desc = BufferDesc::new(
            bytes.len() as u64,
            16 * 1024,
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
        16 * 1024,
        MemoryDomain::Shared,
        BufferUsage::TRANSFER_SOURCE | BufferUsage::PROGRAM_OUTPUT,
    )
    .unwrap();
    let (output, _) = backend
        .allocate_buffer(&context, output_desc)
        .unwrap()
        .into_parts();

    let model = parse(artifact).unwrap();
    let program = match backend.load_program(
        &context,
        model.artifact_ref(target, REQUIRED_RESIDENT_BYTES).unwrap(),
    ) {
        Ok(program) => program,
        Err(BackendError::Unsupported) if target == COREML_TOSA_INTEGER_TARGET => {
            backend.free_buffer(output).unwrap();
            for buffer in input_buffers {
                backend.free_buffer(buffer).unwrap();
            }
            backend.destroy_context(context).unwrap();
            return None;
        }
        Err(error) => panic!("Core ML TOSA load failed: {error:?}"),
    };
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
        .unwrap();
    assert_eq!(
        wait_for_terminal(backend, &event).unwrap(),
        EventState::Complete
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

fn run_tosa_fp32(artifact: &[u8], inputs: &[&[f32]], output_elements: usize) -> Option<Vec<f32>> {
    let encoded = inputs
        .iter()
        .map(|values| float_bytes(values.iter().copied()))
        .collect::<Vec<_>>();
    let input_bytes = encoded.iter().map(Vec::as_slice).collect::<Vec<_>>();
    run_tosa_bytes(
        artifact,
        COREML_TOSA_TARGET,
        &input_bytes,
        output_elements.checked_mul(4).unwrap(),
    )
    .map(|bytes| {
        bytes
            .chunks_exact(4)
            .map(|bytes| f32::from_ne_bytes(bytes.try_into().unwrap()))
            .collect()
    })
}

fn run_tosa_fp16(artifact: &[u8], inputs: &[&[u16]], output_elements: usize) -> Option<Vec<u16>> {
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
        artifact,
        COREML_TOSA_TARGET,
        &input_bytes,
        output_elements.checked_mul(2).unwrap(),
    )
    .map(|bytes| {
        bytes
            .chunks_exact(2)
            .map(|bytes| u16::from_ne_bytes(bytes.try_into().unwrap()))
            .collect()
    })
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
fn gpu_policy_reports_gpu_identity_and_executes_tosa() {
    check_compute_policy(
        CoreMlComputeUnits::CpuAndGpu,
        virtio_accel_core::AcceleratorClass::GPU,
        "VIRTIO_ACCEL_COREML_REQUIRE_GPU",
    );
}

#[test]
fn all_policy_reports_other_identity_and_executes_tosa() {
    check_compute_policy(
        CoreMlComputeUnits::All,
        virtio_accel_core::AcceleratorClass::OTHER,
        "VIRTIO_ACCEL_COREML_REQUIRE_ALL",
    );
}

fn check_compute_policy(
    units: CoreMlComputeUnits,
    class: virtio_accel_core::AcceleratorClass,
    require_env: &str,
) {
    let backend = match CoreMlAccelerator::new_tosa_with_compute_units(units) {
        Ok(backend) => backend,
        Err(InitError::GpuUnavailable | InitError::NeuralEngineUnavailable)
            if std::env::var_os(require_env).is_none() =>
        {
            return;
        }
        Err(error) => panic!("Core ML {units:?} initialization failed: {error}"),
    };
    assert_eq!(backend.device_info().unwrap().identity.class, class);
    let inputs = MATMUL_FP32
        .inputs
        .iter()
        .map(|tensor| float_bytes(tensor.values.iter().copied()))
        .collect::<Vec<_>>();
    let bytes = run_tosa_bytes_on(
        &backend,
        MATMUL_FP32.artifact,
        COREML_TOSA_TARGET,
        &inputs.iter().map(Vec::as_slice).collect::<Vec<_>>(),
        MATMUL_FP32.outputs[0].values.len() * 4,
    )
    .unwrap();
    let actual = bytes
        .chunks_exact(4)
        .map(|b| f32::from_ne_bytes(b.try_into().unwrap()))
        .collect::<Vec<_>>();
    assert!(MATMUL_FP32.output_matches(0, &actual));
    let inputs = MATMUL_FP16
        .inputs
        .iter()
        .map(|tensor| {
            tensor
                .bits
                .iter()
                .flat_map(|b| b.to_ne_bytes())
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let bytes = run_tosa_bytes_on(
        &backend,
        MATMUL_FP16.artifact,
        COREML_TOSA_TARGET,
        &inputs.iter().map(Vec::as_slice).collect::<Vec<_>>(),
        MATMUL_FP16.outputs[0].bits.len() * 2,
    )
    .unwrap();
    let actual = bytes
        .chunks_exact(2)
        .map(|b| u16::from_ne_bytes(b.try_into().unwrap()))
        .collect::<Vec<_>>();
    assert!(MATMUL_FP16.output_matches(0, &actual));
}

#[test]
fn lowers_and_executes_compute_heavy_tosa_matmul() {
    let inputs = MATMUL_FP32
        .inputs
        .iter()
        .map(|tensor| tensor.values)
        .collect::<Vec<_>>();
    let Some(actual) = run_tosa_fp32(
        MATMUL_FP32.artifact,
        &inputs,
        MATMUL_FP32.outputs[0].values.len(),
    ) else {
        return;
    };

    assert!(MATMUL_FP32.output_matches(0, &actual));
}

#[test]
fn lowers_and_executes_compute_heavy_tosa_matmul_fp16() {
    let inputs = MATMUL_FP16
        .inputs
        .iter()
        .map(|tensor| tensor.bits)
        .collect::<Vec<_>>();
    let Some(actual) = run_tosa_fp16(
        MATMUL_FP16.artifact,
        &inputs,
        MATMUL_FP16.outputs[0].bits.len(),
    ) else {
        return;
    };

    assert!(MATMUL_FP16.output_matches(0, &actual));
}

#[test]
fn lowers_and_executes_exact_int8_identity() {
    let input = IDENTITY_INT8.inputs[0].bytes;
    let Some(actual) = run_tosa_bytes(
        IDENTITY_INT8.artifact,
        COREML_TOSA_INTEGER_TARGET,
        &[input],
        IDENTITY_INT8.outputs[0].bytes.len(),
    ) else {
        // INT8 multi-array model boundaries are a macOS 26 runtime capability.
        return;
    };
    assert!(IDENTITY_INT8.output_matches(0, &actual));
}

#[test]
fn lowers_and_executes_exact_int8_matmul() {
    let inputs = MATMUL_INT8
        .inputs
        .iter()
        .map(|tensor| tensor.bytes)
        .collect::<Vec<_>>();
    let Some(actual) = run_tosa_bytes(
        MATMUL_INT8.artifact,
        COREML_TOSA_INTEGER_TARGET,
        &inputs,
        MATMUL_INT8.outputs[0].values.len().checked_mul(4).unwrap(),
    ) else {
        return;
    };
    let actual = actual
        .chunks_exact(4)
        .map(|bytes| i32::from_ne_bytes(bytes.try_into().unwrap()))
        .collect::<Vec<_>>();
    assert!(MATMUL_INT8.output_matches(0, &actual));
}

#[test]
fn lowers_and_executes_nhwc_tosa_max_pool2d() {
    let inputs = MAX_POOL2D_FP32
        .inputs
        .iter()
        .map(|tensor| tensor.values)
        .collect::<Vec<_>>();
    let Some(actual) = run_tosa_fp32(
        MAX_POOL2D_FP32.artifact,
        &inputs,
        MAX_POOL2D_FP32.outputs[0].values.len(),
    ) else {
        return;
    };

    assert!(MAX_POOL2D_FP32.output_matches(0, &actual));
}

#[test]
fn lowers_and_executes_nhwc_tosa_max_pool2d_fp16() {
    let inputs = MAX_POOL2D_FP16
        .inputs
        .iter()
        .map(|tensor| tensor.bits)
        .collect::<Vec<_>>();
    let Some(actual) = run_tosa_fp16(
        MAX_POOL2D_FP16.artifact,
        &inputs,
        MAX_POOL2D_FP16.outputs[0].bits.len(),
    ) else {
        return;
    };

    assert!(MAX_POOL2D_FP16.output_matches(0, &actual));
}

#[test]
fn preserves_tosa_fp32_nonfinite_subnormal_and_signed_zero_edges() {
    let inputs = IDENTITY_EDGES_FP32
        .inputs
        .iter()
        .map(|tensor| tensor.values)
        .collect::<Vec<_>>();
    let Some(actual) = run_tosa_fp32(
        IDENTITY_EDGES_FP32.artifact,
        &inputs,
        IDENTITY_EDGES_FP32.outputs[0].values.len(),
    ) else {
        return;
    };

    assert!(IDENTITY_EDGES_FP32.output_matches(0, &actual));
}

#[test]
fn preserves_tosa_fp16_nonfinite_subnormal_and_signed_zero_edges() {
    let inputs = IDENTITY_EDGES_FP16
        .inputs
        .iter()
        .map(|tensor| tensor.bits)
        .collect::<Vec<_>>();
    let Some(actual) = run_tosa_fp16(
        IDENTITY_EDGES_FP16.artifact,
        &inputs,
        IDENTITY_EDGES_FP16.outputs[0].bits.len(),
    ) else {
        return;
    };

    assert!(IDENTITY_EDGES_FP16.output_matches(0, &actual));
}

#[test]
fn permits_overlapping_read_only_inputs_across_async_tosa_predictions() {
    const PREDICTIONS: usize = 16;

    let backend = match CoreMlAccelerator::new_tosa() {
        Ok(backend) => backend,
        Err(InitError::NeuralEngineUnavailable) => return,
        Err(error) => panic!("Core ML backend initialization failed: {error}"),
    };
    let context = backend.create_context(ContextDesc::default()).unwrap();
    let input_desc = BufferDesc::new(
        4,
        16 * 1024,
        MemoryDomain::Shared,
        BufferUsage::TRANSFER_DESTINATION | BufferUsage::PROGRAM_INPUT,
    )
    .unwrap();
    let output_desc = BufferDesc::new(
        4,
        16 * 1024,
        MemoryDomain::Shared,
        BufferUsage::TRANSFER_SOURCE | BufferUsage::PROGRAM_OUTPUT,
    )
    .unwrap();
    let (mut input, _) = backend
        .allocate_buffer(&context, input_desc)
        .unwrap()
        .into_parts();
    let expected = (-0.0_f32).to_ne_bytes();
    backend
        .write_buffer(&mut input, 0, &SliceSource(&expected))
        .unwrap();
    let mut outputs = Vec::with_capacity(PREDICTIONS);
    for _ in 0..PREDICTIONS {
        outputs.push(
            backend
                .allocate_buffer(&context, output_desc)
                .unwrap()
                .into_parts()
                .0,
        );
    }

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
    let mut events = Vec::with_capacity(PREDICTIONS);
    for output in &outputs {
        events.push(
            backend
                .submit(
                    &queue,
                    &program,
                    &[
                        BindingRef {
                            slot: 0,
                            buffer: &input,
                            range: BufferRange::new(0, 4).unwrap(),
                            access: AccessMode::Read,
                        },
                        BindingRef {
                            slot: 1,
                            buffer: output,
                            range: BufferRange::new(0, 4).unwrap(),
                            access: AccessMode::Write,
                        },
                    ],
                    Timeout::Infinite,
                )
                .unwrap(),
        );
    }
    assert_eq!(backend.direct_binding_admissions(), PREDICTIONS as u64 * 2);

    for event in events {
        assert_eq!(
            wait_for_terminal(&backend, &event).unwrap(),
            EventState::Complete
        );
        backend.destroy_event(event).unwrap();
    }
    for output in outputs {
        let mut actual = [0; 4];
        backend.read_buffer(&output, 0, &mut actual).unwrap();
        assert_eq!(actual, expected);
        backend.free_buffer(output).unwrap();
    }

    backend.destroy_queue(queue).unwrap();
    backend.unload_program(program).unwrap();
    backend.free_buffer(input).unwrap();
    backend.destroy_context(context).unwrap();
}

#[test]
fn repeated_tosa_compilation_leaves_no_source_directories() {
    const LOADS: usize = 8;

    let backend = match CoreMlAccelerator::new_tosa() {
        Ok(backend) => backend,
        Err(InitError::NeuralEngineUnavailable) => return,
        Err(error) => panic!("Core ML backend initialization failed: {error}"),
    };
    let context = backend.create_context(ContextDesc::default()).unwrap();
    let source_directories = || {
        fs::read_dir(std::env::temp_dir())
            .unwrap()
            .filter_map(Result::ok)
            .filter(|entry| {
                let name = entry.file_name();
                let name = name.to_string_lossy();
                let suffix = name.strip_prefix("virtio-accel-coreml-");
                suffix.is_some_and(|suffix| {
                    suffix.len() == 36
                        && suffix.bytes().enumerate().all(|(index, byte)| match index {
                            8 | 13 | 18 | 23 => byte == b'-',
                            _ => byte.is_ascii_hexdigit(),
                        })
                })
            })
            .map(|entry| entry.file_name())
            .collect::<std::collections::BTreeSet<_>>()
    };
    let before = source_directories();
    let model = parse(MATMUL_FP32.artifact).unwrap();
    for _ in 0..LOADS {
        let program = backend
            .load_program(
                &context,
                model
                    .artifact_ref(COREML_TOSA_TARGET, REQUIRED_RESIDENT_BYTES)
                    .unwrap(),
            )
            .unwrap();
        backend.unload_program(program).unwrap();
    }
    let deadline = Instant::now() + Duration::from_secs(2);
    loop {
        let after = source_directories();
        if after.difference(&before).next().is_none() {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "TOSA compilation left source directories behind: {:?}",
            after.difference(&before).collect::<Vec<_>>()
        );
        std::thread::sleep(Duration::from_millis(10));
    }
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
    .build(virtio_accel_coreml::COREML_TOSA_FP8_TARGET)
    .unwrap()
}

#[test]
fn fp8_packed_casts_preserve_values_and_round_ties() {
    use virtio_accel_coreml::COREML_TOSA_FP8_TARGET as target;
    use virtio_accel_tosa::{DType, TosaCapabilityProvider, fp8e4m3_to_f32, fp8e5m2_to_f32};
    for units in [
        CoreMlComputeUnits::CpuAndNeuralEngine,
        CoreMlComputeUnits::CpuAndGpu,
        CoreMlComputeUnits::All,
    ] {
        let backend = match CoreMlAccelerator::new_tosa_with_compute_units(units) {
            Ok(b) => b,
            Err(InitError::GpuUnavailable | InitError::NeuralEngineUnavailable)
                if std::env::var_os("VIRTIO_ACCEL_COREML_REQUIRE_QUANTIZED").is_none() =>
            {
                continue;
            }
            Err(e) => panic!("{units:?}: {e}"),
        };
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
            let bytes = run_tosa_bytes_on(&backend, &graph, target, &[&input], 1024).unwrap();
            for (i, b) in bytes.chunks_exact(4).enumerate() {
                let got = f32::from_le_bytes(b.try_into().unwrap());
                let want = decode(i as u8);
                if want.is_nan() {
                    assert!(got.is_nan(), "{units:?} {dtype:?} {i}")
                } else {
                    assert_eq!(got.to_bits(), want.to_bits(), "{units:?} {dtype:?} {i}");
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
            let got = run_tosa_bytes_on(&backend, &graph, target, &[&input], values.len()).unwrap();
            for (i, (got, want)) in got.iter().zip(&expected).enumerate() {
                assert_eq!(
                    got, want,
                    "{units:?} {dtype:?} value {:?} at {i}",
                    values[i]
                );
            }
        }
    }
}

#[test]
fn nvfp4_projection_decodes_packed_weights_and_block_scales() {
    use virtio_accel_coreml::COREML_TOSA_QUANTIZED_TARGET;
    use virtio_accel_tosa::TosaCapabilityProvider;
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
    for units in [
        CoreMlComputeUnits::CpuAndNeuralEngine,
        CoreMlComputeUnits::CpuAndGpu,
        CoreMlComputeUnits::All,
    ] {
        let backend = match CoreMlAccelerator::new_tosa_with_compute_units(units) {
            Ok(b) => b,
            Err(InitError::GpuUnavailable | InitError::NeuralEngineUnavailable)
                if std::env::var_os("VIRTIO_ACCEL_COREML_REQUIRE_QUANTIZED").is_none() =>
            {
                continue;
            }
            Err(e) => panic!("{units:?}: {e}"),
        };
        if !backend
            .tosa_capabilities()
            .iter()
            .any(|c| c.target == COREML_TOSA_QUANTIZED_TARGET)
        {
            continue;
        }
        let out = run_tosa_bytes_on(
            &backend,
            &graph,
            COREML_TOSA_QUANTIZED_TARGET,
            &[&bytes, &weights, &scales],
            m * n * 4,
        )
        .unwrap();
        for (i, b) in out.chunks_exact(4).enumerate() {
            let got = f32::from_le_bytes(b.try_into().unwrap());
            assert!(
                (got - expected[i]).abs() <= 0.0001,
                "{units:?} {i}: {got} != {}",
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
    use virtio_accel_tosa::TosaCapabilityProvider;
    for units in [
        CoreMlComputeUnits::CpuAndNeuralEngine,
        CoreMlComputeUnits::CpuAndGpu,
        CoreMlComputeUnits::All,
    ] {
        let backend = match CoreMlAccelerator::new_tosa_with_compute_units(units) {
            Ok(b) => b,
            Err(InitError::GpuUnavailable | InitError::NeuralEngineUnavailable)
                if std::env::var_os("VIRTIO_ACCEL_COREML_REQUIRE_QUANTIZED").is_none() =>
            {
                continue;
            }
            Err(e) => panic!("{units:?}: {e}"),
        };
        let target = virtio_accel_coreml::COREML_TOSA_QUANTIZED_TARGET;
        if !backend
            .tosa_capabilities()
            .iter()
            .any(|c| c.target == target)
        {
            continue;
        }
        let out =
            run_tosa_bytes_on(&backend, &graph, target, &[&x, &weights, &scales], 64).unwrap();
        for (i, b) in out.chunks_exact(2).enumerate() {
            let bits = u16::from_le_bytes(b.try_into().unwrap());
            assert_eq!(bits, expected[i], "FP16 output {i}");
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
        let target = virtio_accel_coreml::COREML_TOSA_FP8_TARGET;
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
        for units in [
            CoreMlComputeUnits::CpuAndNeuralEngine,
            CoreMlComputeUnits::CpuAndGpu,
            CoreMlComputeUnits::All,
        ] {
            let backend = match CoreMlAccelerator::new_tosa_with_compute_units(units) {
                Ok(b) => b,
                Err(InitError::GpuUnavailable | InitError::NeuralEngineUnavailable)
                    if std::env::var_os("VIRTIO_ACCEL_COREML_REQUIRE_QUANTIZED").is_none() =>
                {
                    continue;
                }
                Err(e) => panic!("{units:?}: {e}"),
            };
            if !backend
                .tosa_capabilities()
                .iter()
                .any(|c| c.target == target)
            {
                continue;
            }
            let out = run_tosa_bytes_on(
                &backend,
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
        .build(virtio_accel_coreml::COREML_TOSA_FP8_TARGET)
        .unwrap()
}

#[test]
fn fp8_activation_and_constant_weights_execute_together() {
    use virtio_accel_coreml::COREML_TOSA_FP8_TARGET;
    use virtio_accel_tosa::{DType, TosaCapabilityProvider};
    let input = [1.0f32, 2.0, -1.0, 0.5]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect::<Vec<_>>();
    for units in [
        CoreMlComputeUnits::CpuAndNeuralEngine,
        CoreMlComputeUnits::CpuAndGpu,
        CoreMlComputeUnits::All,
    ] {
        let backend = match CoreMlAccelerator::new_tosa_with_compute_units(units) {
            Ok(b) => b,
            Err(InitError::GpuUnavailable | InitError::NeuralEngineUnavailable)
                if std::env::var_os("VIRTIO_ACCEL_COREML_REQUIRE_QUANTIZED").is_none() =>
            {
                continue;
            }
            Err(e) => panic!("{units:?}: {e}"),
        };
        if !backend
            .tosa_capabilities()
            .iter()
            .any(|c| c.target == COREML_TOSA_FP8_TARGET)
        {
            continue;
        }
        for dtype in [DType::FP8E4M3, DType::FP8E5M2] {
            let graph = fp8_constant_projection_graph(dtype);
            let out =
                run_tosa_bytes_on(&backend, &graph, COREML_TOSA_FP8_TARGET, &[&input], 4).unwrap();
            assert_eq!(out, [0x20, 0x48, 0x40, 0x4a], "{units:?} {dtype:?}");
        }
    }
}

#[test]
fn compressed_integer_weights_preserve_fp16_rounding() {
    use half::f16;
    use virtio_accel_tosa::TosaCapabilityProvider;
    use virtio_accel_tosa_build::quantized;
    // Basis vectors isolate weight decoding from reduction-order differences.
    // The first shape's constants stay inline in the ML Program; the second's are
    // large enough to be stored in its weight file.
    let scales = [0x3555u16, 0x399a, 0x2e66, 0x0400];
    for ((m, n, k, group), bits) in [(32, 8, 32, 16), (256, 64, 256, 32)]
        .into_iter()
        .flat_map(|shape| [(shape, 4), (shape, 8)])
    {
        let weights = (0..n * k)
            .map(|i| ((i % 16) as i8 - 8) * if bits == 8 { 16 } else { 1 })
            .collect::<Vec<_>>();
        let scales = (0..n * k / group)
            .map(|i| scales[i % scales.len()])
            .collect::<Vec<_>>();
        let graph = if bits == 4 {
            let packed = weights
                .chunks_exact(2)
                .map(|v| (v[0] as u8 & 15) | ((v[1] as u8 & 15) << 4))
                .collect::<Vec<_>>();
            quantized::int4_matmul_fp16(m, n, k, group, &packed, &scales).unwrap()
        } else {
            quantized::int8_matmul_fp16(m, n, k, group, &weights, &scales).unwrap()
        };
        let x = (0..m * k)
            .flat_map(|i| (if i / k == i % k { 0x3c00u16 } else { 0 }).to_le_bytes())
            .collect::<Vec<_>>();
        let expected = (0..m * n)
            .map(|i| {
                let offset = (i % n) * k + i / n;
                f16::from_f32(
                    f32::from(weights[offset]) * f16::from_bits(scales[offset / group]).to_f32(),
                )
                .to_bits()
            })
            .collect::<Vec<_>>();
        for units in [
            CoreMlComputeUnits::CpuAndGpu,
            CoreMlComputeUnits::CpuAndNeuralEngine,
            CoreMlComputeUnits::All,
        ] {
            let backend = match CoreMlAccelerator::new_tosa_with_compute_units(units) {
                Ok(b) => b,
                Err(InitError::GpuUnavailable | InitError::NeuralEngineUnavailable)
                    if std::env::var_os("VIRTIO_ACCEL_COREML_REQUIRE_QUANTIZED").is_none() =>
                {
                    continue;
                }
                Err(e) => panic!("{units:?}: {e}"),
            };
            if !backend
                .tosa_capabilities()
                .iter()
                .any(|c| c.target == quantized::TARGET)
            {
                assert!(
                    std::env::var_os("VIRTIO_ACCEL_COREML_REQUIRE_QUANTIZED").is_none(),
                    "quantized target required"
                );
                continue;
            }
            let out =
                run_tosa_bytes_on(&backend, &graph, quantized::TARGET, &[&x], m * n * 2).unwrap();
            // TOSA 1.0 FP16-output MATMUL permits FP16 accumulation: it does not
            // promise correctly rounded FP64 dots. Exercise dense fractional
            // activations with a local error bound, while keeping the independent
            // basis-vector weight decoding check below bit-exact.
            let codes = [0x3555u16, 0xb99a, 0x2e66, 0x3c01, 0x33ab, 0xb601, 0x3999];
            let activation = (0..m * k)
                .map(|i| f16::from_bits(codes[(i * 13 + i / k) % codes.len()]))
                .collect::<Vec<_>>();
            let bytes = activation
                .iter()
                .flat_map(|v| v.to_bits().to_le_bytes())
                .collect::<Vec<_>>();
            let result =
                run_tosa_bytes_on(&backend, &graph, quantized::TARGET, &[&bytes], m * n * 2)
                    .unwrap();
            for (i, b) in result.chunks_exact(2).enumerate() {
                let terms = (0..k)
                    .map(|j| {
                        let offset = (i % n) * k + j;
                        let w = f16::from_f32(
                            f32::from(weights[offset])
                                * f16::from_bits(scales[offset / group]).to_f32(),
                        );
                        f64::from(activation[(i / n) * k + j].to_f32()) * f64::from(w.to_f32())
                    })
                    .collect::<Vec<_>>();
                let reference = terms.iter().sum::<f64>();
                // Two normalized rounding units is stricter than the TOSA dot
                // absolute bound (2*K). Use sum(abs(products)), not abs(sum), so
                // cancellation does not turn a valid FP16 reduction into a failure.
                // https://www.mlplatform.org/tosa/tosa_spec_1_0_0.html#_dot_product_accuracy_requirements
                let bound =
                    (terms.iter().map(|x| x.abs()).sum::<f64>() / 2048.0).max(1.0 / 16384.0);
                let actual =
                    f64::from(f16::from_bits(u16::from_le_bytes(b.try_into().unwrap())).to_f32());
                assert!(
                    (actual - reference).abs() <= 2.0 * bound,
                    "fractional W{bits} {units:?} [{i}]: {actual} vs {reference}, bound {}",
                    2.0 * bound
                );
            }
            for (i, b) in out.chunks_exact(2).enumerate() {
                let got = u16::from_le_bytes(b.try_into().unwrap());
                // Dot products can canonicalize negative zero; every nonzero weight is exact.
                assert!(
                    got == expected[i] || (got & 0x7fff == 0 && expected[i] & 0x7fff == 0),
                    "W{bits} {units:?} [{i}]: {got:04x} != {:04x}",
                    expected[i]
                );
            }
        }
    }
}

#[test]
fn compressed_palette_weights_preserve_arbitrary_entries() {
    use virtio_accel_tosa::TosaCapabilityProvider;
    use virtio_accel_tosa_build::quantized;
    let (m, n, k, group) = (32, 8, 32, 16);
    let entries = [
        0x0000u16, 0x8000, 0x0001, 0x03ff, 0x0400, 0x3555, 0xb99a, 0x3c00, 0xc123, 0x4555, 0xbc01,
        0x2e66, 0x1001, 0x8701, 0x3333, 0x3888,
    ];
    let lut = (0..n * k / group * 16)
        .map(|i| entries[(i + i / 16) % 16])
        .collect::<Vec<_>>();
    let packed = (0..n * k / 2)
        .map(|i| ((2 * i) % 16) as u8 | (((2 * i + 1) % 16) as u8) << 4)
        .collect::<Vec<_>>();
    let graph = quantized::palette4_matmul_fp16(m, n, k, group, &packed, &lut).unwrap();
    use virtio_accel_tosa_build::{Graph, Operator, OperatorKind, Tensor};
    let dense_bytes = (0..k * n)
        .flat_map(|i| {
            let pos = (i % n) * k + i / n;
            lut[(pos / group) * 16 + pos % 16].to_le_bytes()
        })
        .collect::<Vec<_>>();
    let a_shape = [1, m as i32, k as i32];
    let b_shape = [1, k as i32, n as i32];
    let out_shape = [1, m as i32, n as i32];
    let f = virtio_accel_tosa::DType::FP16;
    let tensors = [
        Tensor::new("a", &a_shape, f),
        Tensor::constant("b", &b_shape, f, &dense_bytes),
        Tensor::constant("z", &[1], f, &[0, 0]),
        Tensor::new("out", &out_shape, f),
    ];
    let ops = [
        Operator::new(OperatorKind::Const, &[], &["b"]),
        Operator::new(OperatorKind::Const, &[], &["z"]),
        Operator::new(OperatorKind::MatMul, &["a", "b", "z", "z"], &["out"]),
    ];
    let dense = Graph::new("dense", &tensors, &ops, &["a"], &["out"])
        .build(quantized::TARGET)
        .unwrap();
    let x = (0..m * k)
        .flat_map(|i| (if i / k == i % k { 0x3c00u16 } else { 0 }).to_le_bytes())
        .collect::<Vec<_>>();
    for units in [
        CoreMlComputeUnits::CpuAndGpu,
        CoreMlComputeUnits::CpuAndNeuralEngine,
        CoreMlComputeUnits::All,
    ] {
        let backend = match CoreMlAccelerator::new_tosa_with_compute_units(units) {
            Ok(b) => b,
            Err(InitError::GpuUnavailable | InitError::NeuralEngineUnavailable)
                if std::env::var_os("VIRTIO_ACCEL_COREML_REQUIRE_QUANTIZED").is_none() =>
            {
                continue;
            }
            Err(e) => panic!("{units:?}: {e}"),
        };
        if !backend
            .tosa_capabilities()
            .iter()
            .any(|c| c.target == quantized::TARGET)
        {
            assert!(
                std::env::var_os("VIRTIO_ACCEL_COREML_REQUIRE_QUANTIZED").is_none(),
                "quantized target required"
            );
            continue;
        }
        let out = run_tosa_bytes_on(&backend, &graph, quantized::TARGET, &[&x], m * n * 2).unwrap();
        let dense_out =
            run_tosa_bytes_on(&backend, &dense, quantized::TARGET, &[&x], m * n * 2).unwrap();
        assert_eq!(
            out, dense_out,
            "compressed and dense linear must agree for {units:?}"
        );
        for (i, b) in out.chunks_exact(2).enumerate() {
            let pos = (i % n) * k + i / n;
            let expected = lut[(pos / group) * 16 + pos % 16];
            let got = u16::from_le_bytes(b.try_into().unwrap());
            assert!(
                got == expected || (got & 0x7fff == 0 && expected & 0x7fff == 0),
                "palette {units:?} [{i}]: {got:04x} != {expected:04x}"
            );
        }
    }
}
