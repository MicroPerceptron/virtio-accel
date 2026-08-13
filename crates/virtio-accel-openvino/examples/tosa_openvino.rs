#[cfg(not(va_openvino))]
fn main() {
    eprintln!("the TOSA-to-OpenVINO example requires an OpenVINO 2026.x runtime (libopenvino_c)");
}

#[cfg(va_openvino)]
fn main() -> Result<(), ExampleError> {
    use std::time::{Duration, Instant};
    use virtio_accel_core::{
        Accelerator, AccessMode, BackendError, BindingRef, BufferDesc, BufferRange, BufferUsage,
        ContextDesc, EventState, MemoryDomain, QueueDesc, ReleaseFailure, SubmitFailure, Timeout,
    };
    use virtio_accel_openvino::{
        InitError, OPENVINO_TOSA_TARGET, OpenVinoAccelerator, REQUIRED_RESIDENT_BYTES,
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
    let backend = match OpenVinoAccelerator::new() {
        Ok(backend) => backend,
        Err(InitError::DeviceUnavailable) => {
            eprintln!("OpenVINO enumerates no inference device on this host; example skipped");
            return Ok(());
        }
        Err(error) => return Err(error.into()),
    };
    eprintln!("executing on the {} device", backend.device_name());
    let context = backend.create_context(ContextDesc::default())?;
    let (mut input, _) = backend
        .allocate_buffer(
            &context,
            BufferDesc::new(
                4,
                4096,
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
                4096,
                MemoryDomain::Shared,
                BufferUsage::TRANSFER_SOURCE | BufferUsage::PROGRAM_OUTPUT,
            )?,
        )?
        .into_parts();
    let input_value = 3.25_f32.to_ne_bytes();
    backend.write_buffer(&mut input, 0, &input_value)?;

    let program = backend.load_program(
        &context,
        model.artifact_ref(OPENVINO_TOSA_TARGET, REQUIRED_RESIDENT_BYTES)?,
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
        "TOSA -> OpenVINO -> {} result: {}",
        backend.device_name(),
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

#[cfg(va_openvino)]
#[derive(Debug)]
struct ExampleError(String);

#[cfg(va_openvino)]
impl std::fmt::Display for ExampleError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

#[cfg(va_openvino)]
impl std::error::Error for ExampleError {}

#[cfg(va_openvino)]
impl From<virtio_accel_core::BackendError> for ExampleError {
    fn from(error: virtio_accel_core::BackendError) -> Self {
        Self(format!("backend error: {error:?}"))
    }
}

#[cfg(va_openvino)]
impl From<virtio_accel_tosa::Error> for ExampleError {
    fn from(error: virtio_accel_tosa::Error) -> Self {
        Self(format!("TOSA parse error: {error:?}"))
    }
}

#[cfg(va_openvino)]
impl From<virtio_accel_tosa::ArtifactError> for ExampleError {
    fn from(error: virtio_accel_tosa::ArtifactError) -> Self {
        Self(format!("TOSA artifact error: {error:?}"))
    }
}

#[cfg(va_openvino)]
impl From<virtio_accel_openvino::InitError> for ExampleError {
    fn from(error: virtio_accel_openvino::InitError) -> Self {
        Self(format!("OpenVINO initialization error: {error:?}"))
    }
}
