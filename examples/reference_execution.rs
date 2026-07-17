use virtio_accel::core::{
    Accelerator, AccessMode, ArtifactRef, BackendError, BindingRef, BufferDesc, BufferRange,
    BufferUsage, ContextDesc, EventState, MemoryDomain, QueueDesc, ReleaseFailure, Timeout,
};
use virtio_accel_mock::{MockAccelerator, reference};

fn release_ok<T>(result: Result<(), ReleaseFailure<T>>) -> Result<(), BackendError> {
    match result {
        Ok(()) => Ok(()),
        Err(ReleaseFailure::Rejected { error, .. } | ReleaseFailure::Indeterminate { error }) => {
            Err(error)
        }
    }
}

fn main() -> Result<(), BackendError> {
    let backend = MockAccelerator::default();
    let context = backend.create_context(ContextDesc::default())?;

    let desc = BufferDesc::new(
        8,
        8,
        MemoryDomain::Shared,
        BufferUsage::TRANSFER_SOURCE
            | BufferUsage::TRANSFER_DESTINATION
            | BufferUsage::PROGRAM_INPUT
            | BufferUsage::PROGRAM_OUTPUT
            | BufferUsage::MUTABLE_STATE,
    )?;
    let (mut buffer, _) = backend.allocate_buffer(&context, desc)?.into_parts();
    backend.write_buffer(
        &mut buffer,
        0,
        &[0x00, 0x11, 0x7f, 0x80, 0xa5, 0xff, 0x3c, 0xc3],
    )?;

    let artifact = reference::ReferenceArtifact::xor(7, 0x5a);
    let program = backend.load_program(
        &context,
        ArtifactRef {
            format: reference::ARTIFACT_FORMAT,
            target: reference::TARGET_IDENTITY,
            payload: artifact.as_bytes(),
            resident_bytes: reference::RESIDENT_BYTES,
        },
    )?;
    let queue = backend.create_queue(&context, QueueDesc::default())?;

    let binding = [BindingRef {
        slot: 7,
        buffer: &buffer,
        range: BufferRange::new(0, 8)?,
        access: AccessMode::ReadWrite,
    }];
    let event = backend
        .submit(&queue, &program, &binding, Timeout::Infinite)
        .map_err(|failure| match failure {
            virtio_accel::core::SubmitFailure::Rejected(error)
            | virtio_accel::core::SubmitFailure::Indeterminate { error, .. } => error,
        })?;

    assert_eq!(backend.poll_event(&event)?, EventState::Pending);
    backend.complete(&event)?;
    assert_eq!(backend.poll_event(&event)?, EventState::Complete);

    let mut output = [0_u8; 8];
    backend.read_buffer(&buffer, 0, &mut output)?;
    assert_eq!(output, [0x5a, 0x4b, 0x25, 0xda, 0xff, 0xa5, 0x66, 0x99]);
    println!("completed output: {output:02x?}");

    release_ok(backend.destroy_event(event))?;
    release_ok(backend.destroy_queue(queue))?;
    release_ok(backend.unload_program(program))?;
    release_ok(backend.free_buffer(buffer))?;
    release_ok(backend.destroy_context(context))?;
    Ok(())
}
