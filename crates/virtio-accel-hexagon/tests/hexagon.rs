#![cfg(va_hexagon)]

use std::time::{Duration, Instant};
use virtio_accel_conformance::numerics::{
    IDENTITY_EDGES_FP16, MATMUL_FP16, MAX_POOL2D_FP16, MOCK_LINEAR_CLASSIFIER_FP16, TosaFloat16Case,
};
use virtio_accel_core::{
    Accelerator, AccessMode, BackendError, BindingRef, BufferDesc, BufferRange, BufferUsage,
    ByteSink, ByteSource, ContextDesc, EventState, MemoryDomain, QueueDesc, ReleaseFailure,
    SubmitFailure, Timeout,
};
use virtio_accel_hexagon::{HEXAGON_TOSA_TARGET, HexagonAccelerator, REQUIRED_RESIDENT_BYTES};
use virtio_accel_tosa::parse;

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
    let backend = HexagonAccelerator::new().expect("initialize QNN HTP");
    let context = backend
        .create_context(ContextDesc::default())
        .expect("create context");
    let model = parse(case.artifact).expect("parse corpus TOSA");
    let program = backend
        .load_program(
            &context,
            model
                .artifact_ref(HEXAGON_TOSA_TARGET, REQUIRED_RESIDENT_BYTES)
                .expect("artifact envelope"),
        )
        .expect("finalize corpus graph on HTP");

    let input_bytes = case
        .inputs
        .iter()
        .map(|input| fp16_bytes(input.bits))
        .collect::<Vec<_>>();
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
    let output_len = case.outputs[0].bits.len() * 2;
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
        Err(SubmitFailure::Rejected(BackendError::PermissionDenied))
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
        Err(SubmitFailure::Rejected(BackendError::Unsupported))
    ));
    let event = backend
        .submit(&queue, &program, &bindings, Timeout::Infinite)
        .unwrap_or_else(|failure| match failure {
            SubmitFailure::Rejected(error) => panic!("{} rejected: {error:?}", case.name),
            SubmitFailure::Indeterminate { error, .. } => {
                panic!("{} indeterminate: {error:?}", case.name)
            }
        });
    let deadline = Instant::now() + Duration::from_secs(15);
    loop {
        match backend.poll_event(&event).expect("poll event") {
            EventState::Pending if Instant::now() < deadline => std::thread::yield_now(),
            EventState::Pending => panic!("{} timed out", case.name),
            EventState::Complete => break,
            terminal => panic!("{} failed: {terminal:?}", case.name),
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
    let actual = output_bytes
        .chunks_exact(2)
        .map(|bytes| u16::from_le_bytes([bytes[0], bytes[1]]))
        .collect::<Vec<_>>();
    assert!(
        case.output_matches(0, &actual),
        "{} oracle mismatch",
        case.name
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
