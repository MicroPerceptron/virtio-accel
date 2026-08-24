//! On-hardware tests for the HRX buffer primitives.
//!
//! These compile and run only in a `va_xdna` build (a detected HRX prefix) and require an
//! accessible NPU. They cover the allocate / map / write+flush / read+invalidate / release cycle
//! plus context and queue lifecycle. Program loading and dispatch are covered by the execution
//! ticket.
#![cfg(va_xdna)]

use virtio_accel_core::{
    Accelerator, BackendError, BufferDesc, BufferUsage, ByteSink, ByteSource, ContextDesc,
    MemoryDomain, QueueDesc,
};
use virtio_accel_xdna::{InitError, XdnaAccelerator};

/// Construct a backend, or skip the test when no NPU is accessible on this host.
fn backend() -> Option<XdnaAccelerator> {
    match XdnaAccelerator::new() {
        Ok(backend) => Some(backend),
        Err(InitError::DeviceUnavailable) => {
            eprintln!("no XDNA NPU device accessible; skipping hardware test");
            None
        }
        Err(error) => panic!("unexpected initialization failure: {error}"),
    }
}

#[derive(Debug)]
struct Slice<'a>(&'a [u8]);

impl ByteSource for Slice<'_> {
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

#[derive(Debug)]
struct SliceMut<'a>(&'a mut [u8]);

impl ByteSink for SliceMut<'_> {
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
        Some(self.0)
    }
}

fn shared_desc(bytes: u64) -> BufferDesc {
    BufferDesc::new(
        bytes,
        4096,
        MemoryDomain::Shared,
        BufferUsage::TRANSFER_DESTINATION
            | BufferUsage::TRANSFER_SOURCE
            | BufferUsage::PROGRAM_INPUT,
    )
    .expect("valid buffer descriptor")
}

#[test]
fn device_info_reports_the_npu() {
    let Some(backend) = backend() else { return };
    let info = backend.device_info().expect("device info");
    assert_eq!(info.identity.vendor_id, 0x1022);
    assert_eq!(info.identity.device_id, 0x17f0);
}

#[test]
fn buffer_write_flush_invalidate_read_roundtrips() {
    let Some(backend) = backend() else { return };
    let context = backend
        .create_context(ContextDesc::default())
        .expect("context");
    let (mut buffer, info) = backend
        .allocate_buffer(&context, shared_desc(256))
        .expect("allocate")
        .into_parts();
    assert!(info.allocation_bytes() >= 256);

    let payload: Vec<u8> = (0..256u32).map(|i| (i * 7) as u8).collect();
    backend
        .write_buffer(&mut buffer, 0, &Slice(&payload))
        .expect("write + flush");

    let mut readback = vec![0u8; 256];
    backend
        .read_buffer(&buffer, 0, &mut SliceMut(&mut readback))
        .expect("invalidate + read");
    assert_eq!(readback, payload, "device-visible mapping must round-trip");

    // A sub-range write is observable at its offset and nowhere else.
    backend
        .write_buffer(&mut buffer, 64, &Slice(&[0xAB; 16]))
        .expect("sub-range write");
    let mut window = vec![0u8; 16];
    backend
        .read_buffer(&buffer, 64, &mut SliceMut(&mut window))
        .expect("sub-range read");
    assert_eq!(window, [0xAB; 16]);

    backend.free_buffer(buffer).expect("free");
    backend.destroy_context(context).expect("destroy context");
}

#[test]
fn out_of_bounds_transfers_are_rejected() {
    let Some(backend) = backend() else { return };
    let context = backend
        .create_context(ContextDesc::default())
        .expect("context");
    let (mut buffer, _) = backend
        .allocate_buffer(&context, shared_desc(64))
        .expect("allocate")
        .into_parts();

    let too_big = vec![0u8; 128];
    assert!(matches!(
        backend.write_buffer(&mut buffer, 0, &Slice(&too_big)),
        Err(BackendError::OutOfBounds)
    ));
    assert!(matches!(
        backend.write_buffer(&mut buffer, 60, &Slice(&[0u8; 8])),
        Err(BackendError::OutOfBounds)
    ));

    backend.free_buffer(buffer).expect("free");
    backend.destroy_context(context).expect("destroy context");
}

#[test]
fn context_and_queue_lifecycle() {
    let Some(backend) = backend() else { return };
    let context = backend
        .create_context(ContextDesc::default())
        .expect("context");
    let queue = backend
        .create_queue(&context, QueueDesc::default())
        .expect("queue");
    backend.destroy_queue(queue).expect("destroy queue");
    backend.destroy_context(context).expect("destroy context");
}
