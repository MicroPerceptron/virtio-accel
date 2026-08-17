use std::time::{Duration, Instant};
use virtio_accel_core::{
    Accelerator, AccessMode, BackendError, BindingRef, BufferDesc, BufferRange, BufferUsage,
    ByteSink, ByteSource, ContextDesc, EventState, MemoryDomain, QueueDesc, SubmitFailure, Timeout,
};
use virtio_accel_hexagon::{
    HEXAGON_TOSA_TARGET, HexagonAccelerator, REQUIRED_RESIDENT_BYTES, TESTED_QAIRT_RELEASE,
};
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

fn fp16_matches(expected: &[u16], actual: &[u16]) -> bool {
    expected.len() == actual.len()
        && expected.iter().zip(actual).all(|(expected, actual)| {
            let expected_is_nan = expected & 0x7c00 == 0x7c00 && expected & 0x03ff != 0;
            let actual_is_nan = actual & 0x7c00 == 0x7c00 && actual & 0x03ff != 0;
            expected == actual || expected_is_nan && actual_is_nan
        })
}

pub fn run_fp16_model(name: &str, artifact: &[u8], input_bits: &[&[u16]], expected: &[u16]) {
    let backend = HexagonAccelerator::new().expect("initialize the QNN HTP backend");
    println!(
        "QAIRT {TESTED_QAIRT_RELEASE}: provider={} build={} core={:?} backend={:?}",
        backend.runtime_info().provider_name,
        backend.runtime_info().build_id,
        backend.runtime_info().core_version,
        backend.runtime_info().backend_version,
    );
    let context = backend
        .create_context(ContextDesc::default())
        .expect("create context");
    let model = parse(artifact).expect("parse model TOSA");
    let program = backend
        .load_program(
            &context,
            model
                .artifact_ref(HEXAGON_TOSA_TARGET, REQUIRED_RESIDENT_BYTES)
                .expect("build artifact envelope"),
        )
        .expect("lower and finalize model graph on HTP");

    let input_bytes = input_bits
        .iter()
        .map(|values| fp16_bytes(values))
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
            .expect("write model input");
        inputs.push(buffer);
    }

    let output_len = expected.len() * 2;
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

    let event = backend
        .submit(&queue, &program, &bindings, Timeout::Infinite)
        .unwrap_or_else(|failure| match failure {
            SubmitFailure::Rejected(error) => panic!("HTP submission rejected: {error:?}"),
            SubmitFailure::Indeterminate { error, .. } => {
                panic!("HTP submission indeterminate: {error:?}")
            }
        });
    let deadline = Instant::now() + Duration::from_secs(15);
    loop {
        match backend.poll_event(&event).expect("poll HTP event") {
            EventState::Pending if Instant::now() < deadline => std::thread::yield_now(),
            EventState::Pending => panic!("HTP execution timed out"),
            EventState::Complete => break,
            terminal => panic!("HTP execution failed: {terminal:?}"),
        }
    }

    let mut actual_bytes = vec![0; output_len];
    backend
        .read_buffer(&output, 0, &mut SliceSink(&mut actual_bytes))
        .expect("read HTP output");
    let actual = actual_bytes
        .chunks_exact(2)
        .map(|bytes| u16::from_le_bytes([bytes[0], bytes[1]]))
        .collect::<Vec<_>>();
    assert!(
        fp16_matches(expected, &actual),
        "{name} result differs from the expected FP16 output: {actual:04x?}"
    );

    backend.destroy_event(event).expect("destroy event");
    backend.destroy_queue(queue).expect("destroy queue");
    backend.unload_program(program).expect("unload program");
    backend.free_buffer(output).expect("free output");
    for input in inputs {
        backend.free_buffer(input).expect("free input");
    }
    backend.destroy_context(context).expect("destroy context");
    println!("{name} -> QNN HTP v73: passed; output={actual:04x?}");
}
