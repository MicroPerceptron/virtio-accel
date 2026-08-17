//! Execute the shared FP16 identity corpus through QNN on Qualcomm Hexagon HTP.

#[cfg(va_hexagon)]
mod enabled {
    use std::time::{Duration, Instant};
    use virtio_accel_conformance::numerics::IDENTITY_EDGES_FP16;
    use virtio_accel_core::{
        Accelerator, AccessMode, BackendError, BindingRef, BufferDesc, BufferRange, BufferUsage,
        ByteSink, ByteSource, ContextDesc, EventState, MemoryDomain, QueueDesc, SubmitFailure,
        Timeout,
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

    pub fn run() {
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
        let model = parse(IDENTITY_EDGES_FP16.artifact).expect("parse identity TOSA");
        let program = backend
            .load_program(
                &context,
                model
                    .artifact_ref(HEXAGON_TOSA_TARGET, REQUIRED_RESIDENT_BYTES)
                    .expect("build artifact envelope"),
            )
            .expect("lower and finalize identity graph on HTP");
        let input_bytes = IDENTITY_EDGES_FP16.inputs[0]
            .bits
            .iter()
            .flat_map(|bits| bits.to_le_bytes())
            .collect::<Vec<_>>();
        let output_bytes = IDENTITY_EDGES_FP16.outputs[0].bits.len() * 2;
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
        let (output, _) = backend
            .allocate_buffer(
                &context,
                BufferDesc::new(
                    output_bytes as u64,
                    4096,
                    MemoryDomain::Shared,
                    BufferUsage::TRANSFER_SOURCE | BufferUsage::PROGRAM_OUTPUT,
                )
                .expect("output descriptor"),
            )
            .expect("allocate output")
            .into_parts();
        backend
            .write_buffer(&mut input, 0, &SliceSource(&input_bytes))
            .expect("write identity input");
        let queue = backend
            .create_queue(&context, QueueDesc::default())
            .expect("create queue");
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
                range: BufferRange::new(0, output_bytes as u64).expect("output range"),
                access: AccessMode::Write,
            },
        ];
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
        backend.destroy_event(event).expect("destroy event");

        let mut actual_bytes = vec![0; output_bytes];
        backend
            .read_buffer(&output, 0, &mut SliceSink(&mut actual_bytes))
            .expect("read HTP output");
        let actual = actual_bytes
            .chunks_exact(2)
            .map(|bytes| u16::from_le_bytes([bytes[0], bytes[1]]))
            .collect::<Vec<_>>();
        assert!(
            IDENTITY_EDGES_FP16.output_matches(0, &actual),
            "HTP identity result differs from the shared FP16 oracle"
        );

        backend.destroy_queue(queue).expect("destroy queue");
        backend.unload_program(program).expect("unload program");
        backend.free_buffer(output).expect("free output");
        backend.free_buffer(input).expect("free input");
        backend.destroy_context(context).expect("destroy context");
        println!("TOSA FP16 identity -> QNN HTP v73: passed");
    }
}

#[cfg(va_hexagon)]
fn main() {
    enabled::run();
}

#[cfg(not(va_hexagon))]
fn main() {
    println!(
        "Qualcomm Hexagon HTP unavailable: build on Windows ARM64 with a complete QAIRT/QNN SDK"
    );
}
