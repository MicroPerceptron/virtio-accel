#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("the TOSA-to-Core ML example requires macOS 14+ and an Apple Neural Engine");
}

#[cfg(target_os = "macos")]
fn main() -> Result<(), ExampleError> {
    use std::time::{Duration, Instant};
    use virtio_accel_core::{
        Accelerator, AccessMode, BackendError, BindingRef, BufferDesc, BufferRange, BufferUsage,
        ContextDesc, EventState, MemoryDomain, QueueDesc, ReleaseFailure, SubmitFailure, Timeout,
    };
    use virtio_accel_coreml::{
        COREML_TOSA_TARGET, CoreMlAccelerator, InitError, REQUIRED_RESIDENT_BYTES,
    };
    use virtio_accel_tosa::parse;

    const MODEL: &[u8] = include_bytes!("../tests/data/identity-fp32-v1.0.0.tosa");

    fn release<T>(result: Result<(), ReleaseFailure<T>>) -> Result<(), BackendError> {
        match result {
            Ok(()) => Ok(()),
            Err(
                ReleaseFailure::Rejected { error, .. } | ReleaseFailure::Indeterminate { error },
            ) => Err(error),
        }
    }

    let model = parse(MODEL)?;
    let backend = match CoreMlAccelerator::new_tosa() {
        Ok(backend) => backend,
        Err(InitError::NeuralEngineUnavailable) => {
            eprintln!("Core ML reports no accessible Apple Neural Engine; example skipped");
            return Ok(());
        }
        Err(error) => return Err(error.into()),
    };
    let context = backend.create_context(ContextDesc::default())?;
    let (mut input, _) = backend
        .allocate_buffer(
            &context,
            BufferDesc::new(
                4,
                16 * 1024,
                MemoryDomain::Shared,
                BufferUsage::TRANSFER_DESTINATION | BufferUsage::PROGRAM_INPUT,
            )?,
        )?
        .into_parts();
    let (output, _) = backend
        .allocate_buffer(
            &context,
            BufferDesc::new(
                4,
                16 * 1024,
                MemoryDomain::Shared,
                BufferUsage::TRANSFER_SOURCE | BufferUsage::PROGRAM_OUTPUT,
            )?,
        )?
        .into_parts();
    let input_value = 3.25_f32.to_ne_bytes();
    backend.write_buffer(&mut input, 0, &input_value)?;

    let program = backend.load_program(
        &context,
        model.artifact_ref(COREML_TOSA_TARGET, REQUIRED_RESIDENT_BYTES)?,
    )?;
    let queue = backend.create_queue(&context, QueueDesc::default())?;
    let event = backend
        .submit(
            &queue,
            &program,
            &[
                BindingRef {
                    slot: 0,
                    buffer: &input,
                    range: BufferRange::new(0, 4)?,
                    access: AccessMode::Read,
                },
                BindingRef {
                    slot: 1,
                    buffer: &output,
                    range: BufferRange::new(0, 4)?,
                    access: AccessMode::Write,
                },
            ],
            Timeout::Infinite,
        )
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

    let mut output_value = [0; 4];
    backend.read_buffer(&output, 0, &mut output_value)?;
    println!(
        "TOSA -> Core ML -> ANE-capable result: {}",
        f32::from_ne_bytes(output_value)
    );

    release(backend.destroy_event(event))?;
    release(backend.destroy_queue(queue))?;
    release(backend.unload_program(program))?;
    release(backend.free_buffer(output))?;
    release(backend.free_buffer(input))?;
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
