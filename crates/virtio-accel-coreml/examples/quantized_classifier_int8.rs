//! Quantized INT8 linear classifier executed on the Apple Neural Engine.
//!
//! The graph is the shared exact INT8 MATMUL tier: two samples of three quantized features,
//! times a direct-bound 3x2 weight matrix, producing exact INT32 logits. The host chooses the
//! prediction with a plain integer argmax; nothing in the graph dequantizes through floating
//! point. Requires macOS 26+ (the exact integer tier) and an Apple Neural Engine.

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!(
        "the quantized INT8 classifier example requires macOS 26+ and an Apple Neural Engine"
    );
}

#[cfg(target_os = "macos")]
const SAMPLES: usize = 2;

#[cfg(target_os = "macos")]
fn main() -> Result<(), ExampleError> {
    use std::time::{Duration, Instant};
    use virtio_accel_conformance::numerics::QUANTIZED_CLASSIFIER_INT8;
    use virtio_accel_core::{
        Accelerator, AccessMode, BackendError, BindingRef, BufferDesc, BufferRange, BufferUsage,
        ContextDesc, EventState, MemoryDomain, QueueDesc, ReleaseFailure, SubmitFailure, Timeout,
    };
    use virtio_accel_coreml::{
        COREML_TOSA_INTEGER_TARGET, CoreMlAccelerator, InitError, REQUIRED_RESIDENT_BYTES,
    };
    use virtio_accel_tosa::parse;

    fn release<T>(result: Result<(), ReleaseFailure<T>>) -> Result<(), BackendError> {
        match result {
            Ok(()) => Ok(()),
            Err(
                ReleaseFailure::Rejected { error, .. } | ReleaseFailure::Indeterminate { error },
            ) => Err(error),
        }
    }

    let case = QUANTIZED_CLASSIFIER_INT8;
    let model = parse(case.artifact)?;
    let backend = match CoreMlAccelerator::new_tosa() {
        Ok(backend) => backend,
        Err(InitError::NeuralEngineUnavailable) => {
            eprintln!("Core ML reports no accessible Apple Neural Engine; example skipped");
            return Ok(());
        }
        Err(error) => return Err(error.into()),
    };
    let context = backend.create_context(ContextDesc::default())?;

    let mut buffers = Vec::new();
    for (slot, tensor) in case.inputs.iter().enumerate() {
        let desc = BufferDesc::new(
            tensor.bytes.len() as u64,
            16,
            MemoryDomain::Shared,
            BufferUsage::TRANSFER_DESTINATION | BufferUsage::PROGRAM_INPUT,
        )?;
        let (mut buffer, _) = backend.allocate_buffer(&context, desc)?.into_parts();
        let bytes: [u8; 6] = tensor
            .bytes
            .try_into()
            .expect("fixture tensor lanes are six packed bytes");
        backend.write_buffer(&mut buffer, 0, &bytes)?;
        buffers.push((slot, tensor, buffer));
    }
    let (output, _) = backend
        .allocate_buffer(
            &context,
            BufferDesc::new(
                (case.outputs[0].values.len() * 4) as u64,
                16,
                MemoryDomain::Shared,
                BufferUsage::TRANSFER_SOURCE | BufferUsage::PROGRAM_OUTPUT,
            )?,
        )?
        .into_parts();

    let program = backend.load_program(
        &context,
        model.artifact_ref(COREML_TOSA_INTEGER_TARGET, REQUIRED_RESIDENT_BYTES)?,
    )?;
    let queue = backend.create_queue(&context, QueueDesc::default())?;
    let mut bindings: Vec<BindingRef<_>> = Vec::new();
    for (slot, tensor, buffer) in &buffers {
        bindings.push(BindingRef {
            slot: *slot as u32,
            buffer,
            range: BufferRange::new(0, tensor.bytes.len() as u64)?,
            access: AccessMode::Read,
        });
    }
    bindings.push(BindingRef {
        slot: case.inputs.len() as u32,
        buffer: &output,
        range: BufferRange::new(0, (case.outputs[0].values.len() * 4) as u64)?,
        access: AccessMode::Write,
    });
    let event = backend
        .submit(&queue, &program, &bindings, Timeout::Infinite)
        .map_err(|failure| match failure {
            SubmitFailure::Rejected(error) | SubmitFailure::Indeterminate { error, .. } => error,
        })?;
    let deadline = Instant::now() + Duration::from_secs(15);
    loop {
        match backend.poll_event(&event)? {
            EventState::Pending if Instant::now() < deadline => std::thread::yield_now(),
            EventState::Pending => return Err(BackendError::DeadlineExpired.into()),
            EventState::Complete => break,
            EventState::Failed(error) => return Err(error.into()),
            EventState::Cancelled => return Err(BackendError::DeviceLost.into()),
        }
    }

    let classes = case.inputs[1].shape[2];
    let mut raw = [0u8; 16];
    assert_eq!(
        raw.len(),
        SAMPLES * classes * 4,
        "logit lane count must equal samples times the class dimension"
    );
    backend.read_buffer(&output, 0, &mut raw)?;
    let logits: Vec<i32> = raw
        .chunks_exact(4)
        .map(|bytes| i32::from_ne_bytes(bytes.try_into().expect("packed i32 lane")))
        .collect();
    for sample in 0..SAMPLES {
        let row = &logits[sample * classes..(sample + 1) * classes];
        let winner = row
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.cmp(b))
            .expect("nonempty logits")
            .0;
        println!("sample {sample}: logits {row:?} -> class {winner}");
    }

    release(backend.destroy_event(event))?;
    release(backend.destroy_queue(queue))?;
    release(backend.unload_program(program))?;
    release(backend.free_buffer(output))?;
    for (_, _, buffer) in buffers {
        release(backend.free_buffer(buffer))?;
    }
    release(backend.destroy_context(context))?;
    Ok(())
}

#[cfg(target_os = "macos")]
#[derive(Debug)]
struct ExampleError(String);

#[cfg(target_os = "macos")]
impl std::fmt::Display for ExampleError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

#[cfg(target_os = "macos")]
impl std::error::Error for ExampleError {}

#[cfg(target_os = "macos")]
impl From<virtio_accel_core::BackendError> for ExampleError {
    fn from(error: virtio_accel_core::BackendError) -> Self {
        Self(format!("backend error: {error:?}"))
    }
}

#[cfg(target_os = "macos")]
impl From<virtio_accel_tosa::Error> for ExampleError {
    fn from(error: virtio_accel_tosa::Error) -> Self {
        Self(format!("TOSA parse error: {error:?}"))
    }
}

#[cfg(target_os = "macos")]
impl From<virtio_accel_tosa::ArtifactError> for ExampleError {
    fn from(error: virtio_accel_tosa::ArtifactError) -> Self {
        Self(format!("TOSA artifact error: {error:?}"))
    }
}

#[cfg(target_os = "macos")]
impl From<virtio_accel_coreml::InitError> for ExampleError {
    fn from(error: virtio_accel_coreml::InitError) -> Self {
        Self(format!("Core ML initialization error: {error:?}"))
    }
}
