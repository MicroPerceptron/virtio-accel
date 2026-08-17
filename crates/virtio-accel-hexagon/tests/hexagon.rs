#![cfg(va_hexagon)]

use std::time::{Duration, Instant};
use virtio_accel_conformance::numerics::{
    ADD_FP16, HEXAGON_LOGICAL_CASES, HEXAGON_MOVEMENT_CASES, HEXAGON_REDUCTION_CASES,
    HEXAGON_UNARY_FP16_CASES, IDENTITY_EDGES_FP16, IDENTITY_INT8, MATMUL_FP16, MATMUL_INT8,
    MAX_POOL2D_FP16, MAXIMUM_FP16, MINIMUM_FP16, MOCK_LINEAR_CLASSIFIER_FP16, MUL_FP16, POW_FP16,
    SUB_FP16, TosaFloat16Case, TosaInt8MatmulCase, TosaPackedCase, TosaRawCase,
};
use virtio_accel_conformance::{
    BindingFixture, ConformanceHooks, ProgramFixture, SubmissionPathDiagnostics, TargetDescription,
    run,
};
use virtio_accel_core::{
    Accelerator, AccessMode, BackendError, BindingRef, BufferDesc, BufferRange, BufferUsage,
    ByteSink, ByteSource, ContextDesc, EventState, MemoryDomain, QueueDesc, ReleaseFailure,
    SubmitFailure, Timeout,
};
use virtio_accel_hexagon::{
    HEXAGON_TOSA_INTEGER_TARGET, HEXAGON_TOSA_TARGET, HexagonAccelerator, REQUIRED_RESIDENT_BYTES,
};
use virtio_accel_tosa::{Target, parse};

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
struct SliceSink<'a>(&'a mut [u8]);

impl ByteSink for SliceSink<'_> {
    fn len(&self) -> u64 {
        self.0.len() as u64
    }

    fn write_at(&mut self, offset: u64, source: &[u8]) -> Result<(), BackendError> {
        ByteSink::write_at(self.0, offset, source)
    }

    fn as_contiguous_mut(&mut self) -> Option<&mut [u8]> {
        Some(self.0)
    }
}

fn fp16_bytes(values: &[u16]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn run_case(case: TosaFloat16Case) {
    let input_bytes = case
        .inputs
        .iter()
        .map(|input| fp16_bytes(input.bits))
        .collect::<Vec<_>>();
    let output_len = case.outputs[0].bits.len() * 2;
    run_raw_case(
        case.name,
        case.artifact,
        HEXAGON_TOSA_TARGET,
        input_bytes,
        output_len,
        |output_bytes| {
            let actual = output_bytes
                .chunks_exact(2)
                .map(|bytes| u16::from_le_bytes([bytes[0], bytes[1]]))
                .collect::<Vec<_>>();
            case.output_matches(0, &actual)
        },
    );
}

fn run_int8_identity(case: TosaPackedCase) {
    let inputs = case
        .inputs
        .iter()
        .map(|input| input.bytes.to_vec())
        .collect::<Vec<_>>();
    run_raw_case(
        case.name,
        case.artifact,
        HEXAGON_TOSA_INTEGER_TARGET,
        inputs,
        case.outputs[0].bytes.len(),
        |actual| case.output_matches(0, actual),
    );
}

fn run_int8_matmul(case: TosaInt8MatmulCase) {
    let inputs = case
        .inputs
        .iter()
        .map(|input| input.bytes.to_vec())
        .collect::<Vec<_>>();
    run_raw_case(
        case.name,
        case.artifact,
        HEXAGON_TOSA_INTEGER_TARGET,
        inputs,
        case.outputs[0].values.len() * 4,
        |output_bytes| {
            let actual = output_bytes
                .chunks_exact(4)
                .map(|bytes| i32::from_le_bytes(bytes.try_into().expect("four-byte chunk")))
                .collect::<Vec<_>>();
            case.output_matches(0, &actual)
        },
    );
}

fn run_raw_oracle_case(case: TosaRawCase) {
    run_raw_case(
        case.name,
        case.artifact,
        HEXAGON_TOSA_TARGET,
        case.inputs.iter().map(|input| input.bytes()).collect(),
        case.output.byte_len(),
        |actual| case.output_matches(actual),
    );
}

fn run_raw_case(
    name: &str,
    artifact: &[u8],
    target: Target,
    input_bytes: Vec<Vec<u8>>,
    output_len: usize,
    output_matches: impl FnOnce(&[u8]) -> bool,
) {
    let backend = HexagonAccelerator::new().expect("initialize QNN HTP");
    let context = backend
        .create_context(ContextDesc::default())
        .expect("create context");
    let model = parse(artifact).expect("parse corpus TOSA");
    let program = backend
        .load_program(
            &context,
            model
                .artifact_ref(target, REQUIRED_RESIDENT_BYTES)
                .expect("artifact envelope"),
        )
        .expect("finalize corpus graph on HTP");

    let mut inputs = Vec::with_capacity(input_bytes.len());
    for bytes in &input_bytes {
        let (mut buffer, _) = backend
            .allocate_buffer(
                &context,
                BufferDesc::new(
                    bytes.len() as u64,
                    4096,
                    MemoryDomain::Shared,
                    BufferUsage::TRANSFER_DESTINATION | BufferUsage::PROGRAM_INPUT,
                )
                .expect("input descriptor"),
            )
            .expect("allocate input")
            .into_parts();
        backend
            .write_buffer(&mut buffer, 0, &SliceSource(bytes))
            .expect("write input");
        inputs.push(buffer);
    }
    let (output, _) = backend
        .allocate_buffer(
            &context,
            BufferDesc::new(
                output_len as u64,
                4096,
                MemoryDomain::Shared,
                BufferUsage::TRANSFER_SOURCE | BufferUsage::PROGRAM_OUTPUT,
            )
            .expect("output descriptor"),
        )
        .expect("allocate output")
        .into_parts();
    let queue = backend
        .create_queue(&context, QueueDesc::default())
        .expect("create queue");
    let mut bindings = inputs
        .iter()
        .enumerate()
        .map(|(slot, buffer)| BindingRef {
            slot: slot as u32,
            buffer,
            range: BufferRange::new(0, input_bytes[slot].len() as u64).expect("input range"),
            access: AccessMode::Read,
        })
        .collect::<Vec<_>>();
    bindings.push(BindingRef {
        slot: inputs.len() as u32,
        buffer: &output,
        range: BufferRange::new(0, output_len as u64).expect("output range"),
        access: AccessMode::Write,
    });

    let output_binding = bindings.len() - 1;
    bindings[output_binding].access = AccessMode::Read;
    assert!(matches!(
        backend.submit(&queue, &program, &bindings, Timeout::Infinite),
        Err(SubmitFailure::Rejected(BackendError::Incompatible))
    ));
    bindings[output_binding].access = AccessMode::Write;
    bindings[output_binding].range =
        BufferRange::new(0, output_len as u64 - 2).expect("short output range");
    assert!(matches!(
        backend.submit(&queue, &program, &bindings, Timeout::Infinite),
        Err(SubmitFailure::Rejected(BackendError::Incompatible))
    ));
    bindings[output_binding].range = BufferRange::new(0, output_len as u64).expect("output range");
    bindings[output_binding].slot = 0;
    assert!(matches!(
        backend.submit(&queue, &program, &bindings, Timeout::Infinite),
        Err(SubmitFailure::Rejected(BackendError::InvalidArgument))
    ));
    bindings[output_binding].slot = inputs.len() as u32;
    assert!(matches!(
        backend.submit(&queue, &program, &bindings, Timeout::from_wire_ns(1)),
        Err(SubmitFailure::Rejected(BackendError::DeadlineExpired))
    ));
    let event = backend
        .submit(&queue, &program, &bindings, Timeout::Infinite)
        .unwrap_or_else(|failure| match failure {
            SubmitFailure::Rejected(error) => panic!("{name} rejected: {error:?}"),
            SubmitFailure::Indeterminate { error, .. } => {
                panic!("{name} indeterminate: {error:?}")
            }
        });
    let deadline = Instant::now() + Duration::from_secs(15);
    loop {
        match backend.poll_event(&event).expect("poll event") {
            EventState::Pending if Instant::now() < deadline => std::thread::yield_now(),
            EventState::Pending => panic!("{name} timed out"),
            EventState::Complete => break,
            terminal => panic!("{name} failed: {terminal:?}"),
        }
    }
    assert_eq!(
        backend
            .poll_event(&event)
            .expect("poll stable terminal event"),
        EventState::Complete
    );
    let mut output_bytes = vec![0; output_len];
    backend
        .read_buffer(&output, 0, &mut SliceSink(&mut output_bytes))
        .expect("read output while the terminal event remains live");
    assert!(
        output_matches(&output_bytes),
        "{name} oracle mismatch: {output_bytes:02x?}"
    );

    let program = match backend.unload_program(program) {
        Err(ReleaseFailure::Rejected {
            error: BackendError::Busy,
            resource,
        }) => resource,
        result => panic!("program release with a live event returned {result:?}"),
    };
    backend
        .destroy_event(event)
        .expect("destroy terminal event");
    backend.destroy_queue(queue).expect("destroy queue");
    backend
        .unload_program(program)
        .expect("unload released program");
    backend.free_buffer(output).expect("free output");
    for input in inputs {
        backend.free_buffer(input).expect("free input");
    }
    backend.destroy_context(context).expect("destroy context");
}

#[test]
fn reports_the_pinned_qnn_htp_runtime() {
    let backend = HexagonAccelerator::new().expect("initialize QNN HTP");
    assert_eq!(backend.runtime_info().core_version, [2, 38, 0]);
    assert_eq!(backend.runtime_info().backend_version, [5, 49, 0]);
    assert!(backend.runtime_info().provider_name.contains("HTP"));
    backend
        .device_info()
        .expect("device info")
        .validate()
        .unwrap();
}

#[test]
fn executes_fp16_identity_on_htp() {
    run_case(IDENTITY_EDGES_FP16);
}

#[test]
fn executes_batched_non_square_fp16_matmul_on_htp() {
    run_case(MATMUL_FP16);
}

#[test]
fn executes_mock_linear_classifier_on_htp() {
    run_case(MOCK_LINEAR_CLASSIFIER_FP16);
}

#[test]
fn executes_nhwc_fp16_max_pool_on_htp() {
    run_case(MAX_POOL2D_FP16);
}

#[test]
fn executes_broadcast_fp16_binary_family_on_htp() {
    for case in [
        ADD_FP16,
        SUB_FP16,
        MUL_FP16,
        POW_FP16,
        MAXIMUM_FP16,
        MINIMUM_FP16,
    ] {
        run_case(case);
    }
}

#[test]
fn executes_fp16_unary_and_activation_family_on_htp() {
    for case in HEXAGON_UNARY_FP16_CASES {
        run_raw_oracle_case(*case);
    }
}

#[test]
fn executes_comparison_selection_and_logical_family_on_htp() {
    for case in HEXAGON_LOGICAL_CASES {
        run_raw_oracle_case(*case);
    }
}

#[test]
fn executes_reduction_and_argmax_family_on_htp() {
    for case in HEXAGON_REDUCTION_CASES {
        run_raw_oracle_case(*case);
    }
}

#[test]
fn executes_constant_and_data_movement_family_on_htp() {
    for case in HEXAGON_MOVEMENT_CASES {
        run_raw_oracle_case(*case);
    }
}

#[test]
fn executes_int8_identity_on_htp() {
    run_int8_identity(IDENTITY_INT8);
}

#[test]
fn executes_zero_point_aware_int8_matmul_on_htp() {
    run_int8_matmul(MATMUL_INT8);
}

#[test]
#[ignore = "manual native performance evidence"]
fn measures_warm_submission_and_completion_latency() {
    measure_warm_latency(
        "fp16-identity",
        IDENTITY_EDGES_FP16.artifact,
        HEXAGON_TOSA_TARGET,
        fp16_bytes(IDENTITY_EDGES_FP16.inputs[0].bits),
        IDENTITY_EDGES_FP16.outputs[0].bits.len() * 2,
    );
    measure_warm_latency(
        "int8-identity",
        IDENTITY_INT8.artifact,
        HEXAGON_TOSA_INTEGER_TARGET,
        IDENTITY_INT8.inputs[0].bytes.to_vec(),
        IDENTITY_INT8.outputs[0].bytes.len(),
    );
}

fn measure_warm_latency(
    name: &str,
    artifact: &[u8],
    target: Target,
    input_bytes: Vec<u8>,
    output_len: usize,
) {
    const WARMUPS: usize = 20;
    const SAMPLES: usize = 200;

    let backend = HexagonAccelerator::new().expect("initialize QNN HTP");
    let context = backend
        .create_context(ContextDesc::default())
        .expect("create context");
    let model = parse(artifact).expect("parse benchmark TOSA");
    let program = backend
        .load_program(
            &context,
            model
                .artifact_ref(target, REQUIRED_RESIDENT_BYTES)
                .expect("artifact envelope"),
        )
        .expect("finalize benchmark graph on HTP");
    let queue = backend
        .create_queue(&context, QueueDesc::default())
        .expect("create queue");
    let (mut input, _) = backend
        .allocate_buffer(
            &context,
            BufferDesc::new(
                input_bytes.len() as u64,
                4096,
                MemoryDomain::Shared,
                BufferUsage::TRANSFER_DESTINATION | BufferUsage::PROGRAM_INPUT,
            )
            .expect("input descriptor"),
        )
        .expect("allocate input")
        .into_parts();
    backend
        .write_buffer(&mut input, 0, &SliceSource(&input_bytes))
        .expect("write input");
    let (output, _) = backend
        .allocate_buffer(
            &context,
            BufferDesc::new(
                output_len as u64,
                4096,
                MemoryDomain::Shared,
                BufferUsage::PROGRAM_OUTPUT,
            )
            .expect("output descriptor"),
        )
        .expect("allocate output")
        .into_parts();

    let direct_before = backend.direct_binding_admissions();
    let transfers_before = backend.explicit_transfer_bytes();
    let submit_once = || {
        let bindings = [
            BindingRef {
                slot: 0,
                buffer: &input,
                range: BufferRange::new(0, input_bytes.len() as u64).expect("input range"),
                access: AccessMode::Read,
            },
            BindingRef {
                slot: 1,
                buffer: &output,
                range: BufferRange::new(0, output_len as u64).expect("output range"),
                access: AccessMode::Write,
            },
        ];
        let started = Instant::now();
        let event = backend
            .submit(&queue, &program, &bindings, Timeout::Infinite)
            .unwrap_or_else(|_| panic!("{name} warm submission rejected"));
        let admission = started.elapsed();
        let deadline = started + Duration::from_secs(15);
        loop {
            match backend.poll_event(&event).expect("poll benchmark event") {
                EventState::Pending => {
                    assert!(Instant::now() < deadline, "{name} inference timed out");
                    std::thread::yield_now();
                }
                EventState::Complete => break,
                state => panic!("{name} reached unexpected terminal state {state:?}"),
            }
        }
        let completion = started.elapsed();
        backend
            .destroy_event(event)
            .expect("destroy benchmark event");
        (admission, completion)
    };

    for _ in 0..WARMUPS {
        submit_once();
    }
    let (mut admission, mut completion): (Vec<_>, Vec<_>) =
        (0..SAMPLES).map(|_| submit_once()).unzip();
    admission.sort_unstable();
    completion.sort_unstable();
    let runtime = backend.runtime_info();
    eprintln!(
        "{name}: provider={} build={} core={:?} backend={:?}; warmups={WARMUPS} samples={SAMPLES}; admission p50={:?} p95={:?}; submit-to-complete p50={:?} p95={:?}; direct_bindings={} explicit_transfer_bytes={}",
        runtime.provider_name,
        runtime.build_id,
        runtime.core_version,
        runtime.backend_version,
        admission[SAMPLES / 2],
        admission[SAMPLES * 95 / 100],
        completion[SAMPLES / 2],
        completion[SAMPLES * 95 / 100],
        backend.direct_binding_admissions() - direct_before,
        backend.explicit_transfer_bytes() - transfers_before,
    );

    backend.destroy_queue(queue).expect("destroy queue");
    backend.unload_program(program).expect("unload program");
    backend.free_buffer(input).expect("free input");
    backend.free_buffer(output).expect("free output");
    backend.destroy_context(context).expect("destroy context");
}

struct Hooks;

impl ConformanceHooks<HexagonAccelerator> for Hooks {
    fn complete_event(
        &self,
        backend: &HexagonAccelerator,
        event: &virtio_accel_hexagon::HexagonEvent,
    ) -> Result<(), BackendError> {
        let deadline = Instant::now() + Duration::from_secs(15);
        loop {
            match backend.poll_event(event)? {
                EventState::Pending if Instant::now() < deadline => std::thread::yield_now(),
                EventState::Pending => return Err(BackendError::DeadlineExpired),
                EventState::Complete => return Ok(()),
                EventState::Failed(error) => return Err(error),
                EventState::Cancelled => return Err(BackendError::DeviceLost),
            }
        }
    }

    fn submission_path_diagnostics(
        &self,
        backend: &HexagonAccelerator,
    ) -> Option<SubmissionPathDiagnostics> {
        Some(SubmissionPathDiagnostics {
            direct_bindings: backend.direct_binding_admissions(),
            explicit_transfer_bytes: backend.explicit_transfer_bytes(),
            ..SubmissionPathDiagnostics::default()
        })
    }
}

fn fp16_identity_target() -> TargetDescription {
    let bytes = fp16_bytes(IDENTITY_EDGES_FP16.inputs[0].bits);
    TargetDescription::with_bindings(
        ProgramFixture::new(
            virtio_accel_tosa::ARTIFACT_FORMAT,
            HEXAGON_TOSA_TARGET.to_identity(),
            IDENTITY_EDGES_FP16.artifact,
            REQUIRED_RESIDENT_BYTES,
        )
        .unwrap(),
        vec![
            BindingFixture::read_only(0, MemoryDomain::Shared, 4096, bytes.clone()).unwrap(),
            BindingFixture::new(
                1,
                AccessMode::Write,
                MemoryDomain::Shared,
                4096,
                vec![0; bytes.len()],
                bytes,
            )
            .unwrap(),
        ],
    )
    .unwrap()
}

#[test]
fn passes_the_standard_backend_semantic_suite() {
    let target = fp16_identity_target();
    let report = run(|| HexagonAccelerator::new().unwrap(), &target, &Hooks);
    report.assert_conformant();
}
