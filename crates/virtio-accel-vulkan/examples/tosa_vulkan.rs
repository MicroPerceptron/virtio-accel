//! Load the device-neutral FP32 IDENTITY artifact, execute it on the preferred Vulkan device, and
//! print the round-tripped value. Skips (exit 0) when no Vulkan loader or device is present.

fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(va_vulkan)]
    example::run()?;
    #[cfg(not(va_vulkan))]
    eprintln!("virtio-accel-vulkan was built as the placeholder; the example has nothing to run");
    Ok(())
}

#[cfg(va_vulkan)]
mod example {
    use std::time::{Duration, Instant};

    use virtio_accel_core::{
        Accelerator, AccessMode, BackendError, BindingRef, BufferDesc, BufferRange, BufferUsage,
        ContextDesc, EventState, MemoryDomain, QueueDesc, ReleaseFailure, SubmitFailure, Timeout,
    };
    use virtio_accel_tosa::parse;
    use virtio_accel_vulkan::{
        InitError, REQUIRED_RESIDENT_BYTES, VULKAN_TOSA_TARGET, VulkanAccelerator,
    };

    const MODEL: &[u8] = include_bytes!("../tests/data/identity-fp32-v1.0.0.tosa");

    fn release<T>(result: Result<(), ReleaseFailure<T>>) -> Result<(), BackendError> {
        result.map_err(|failure| failure.error())
    }

    pub fn run() -> Result<(), ExampleError> {
        let model = parse(MODEL)?;
        let backend = match VulkanAccelerator::new() {
            Ok(backend) => backend,
            Err(InitError::RuntimeUnavailable | InitError::DeviceUnavailable) => {
                eprintln!(
                    "no Vulkan 1.3 compute device is available on this host; example skipped"
                );
                return Ok(());
            }
            Err(error) => return Err(error.into()),
        };
        eprintln!("executing on {}", backend.device_name());
        let info = backend.device_info()?;
        // Prefer the zero-copy domain when the device exposes one; every device offers `Host`.
        let domain = if info
            .capabilities
            .contains(virtio_accel_core::Capabilities::SHARED_MEMORY)
        {
            MemoryDomain::Shared
        } else {
            MemoryDomain::Host
        };

        let context = backend.create_context(ContextDesc::default())?;
        let (mut input, _) = backend
            .allocate_buffer(
                &context,
                BufferDesc::new(
                    4,
                    64,
                    domain,
                    BufferUsage::TRANSFER_DESTINATION | BufferUsage::PROGRAM_INPUT,
                )?,
            )?
            .into_parts();
        let (output, _) = backend
            .allocate_buffer(
                &context,
                BufferDesc::new(
                    4,
                    64,
                    domain,
                    BufferUsage::TRANSFER_SOURCE | BufferUsage::PROGRAM_OUTPUT,
                )?,
            )?
            .into_parts();
        backend.write_buffer(&mut input, 0, &3.25_f32.to_ne_bytes())?;

        let program = backend.load_program(
            &context,
            model.artifact_ref(VULKAN_TOSA_TARGET, REQUIRED_RESIDENT_BYTES)?,
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
                SubmitFailure::Rejected(error) | SubmitFailure::Indeterminate { error, .. } => {
                    error
                }
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
            "TOSA -> Vulkan -> {} result: {}",
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

    #[derive(Debug)]
    pub struct ExampleError(String);

    impl std::fmt::Display for ExampleError {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            formatter.write_str(&self.0)
        }
    }

    impl std::error::Error for ExampleError {}

    impl From<BackendError> for ExampleError {
        fn from(error: BackendError) -> Self {
            Self(format!("backend error: {error:?}"))
        }
    }

    impl From<virtio_accel_tosa::Error> for ExampleError {
        fn from(error: virtio_accel_tosa::Error) -> Self {
            Self(format!("TOSA parse error: {error:?}"))
        }
    }

    impl From<virtio_accel_tosa::ArtifactError> for ExampleError {
        fn from(error: virtio_accel_tosa::ArtifactError) -> Self {
            Self(format!("TOSA artifact error: {error:?}"))
        }
    }

    impl From<InitError> for ExampleError {
        fn from(error: InitError) -> Self {
            Self(format!("Vulkan initialization error: {error}"))
        }
    }
}
