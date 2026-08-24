//! On-hardware tests for the HRX buffer primitives.
//!
//! These compile and run only in a `va_xdna` build (a detected HRX prefix) and require an
//! accessible NPU. They cover the allocate / map / write+flush / read+invalidate / release cycle
//! plus context and queue lifecycle. Program loading and dispatch are covered by the execution
//! ticket.
#![cfg(va_xdna)]

use std::time::{Duration, Instant};

use virtio_accel_core::{
    Accelerator, AccessMode, ArtifactRef, BackendError, BindingRef, BufferDesc, BufferRange,
    BufferUsage, ByteSink, ByteSource, ContextDesc, EventState, MemoryDomain, QueueDesc,
    TargetIdentity, Timeout,
};
use virtio_accel_xdna::{InitError, XDNA_PRECOMPILED_FORMAT, XdnaAccelerator};

/// Precompiled DMA passthrough for npu2, built with the pinned toolchain from
/// `programming_examples/basic/passthrough_dmas` (n=4096 int32; entry `MLIR_AIE`) and packaged with
/// `virtio_accel_xdna::artifact::encode`. See `tests/data/README.md`. The design declares three
/// runtime buffers — `a_in`, an unused second input `_b_unused`, and `c_out` — so it binds two
/// inputs and one output; the DMA copies the first input to the output.
const PASSTHROUGH: &[u8] = include_bytes!("data/passthrough-dmas-npu2.xdnp");
const PASSTHROUGH_BYTES: usize = 4096 * 4;

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
fn read_requires_transfer_source_permission() {
    let Some(backend) = backend() else { return };
    let context = backend
        .create_context(ContextDesc::default())
        .expect("context");
    // A write-only buffer (no TRANSFER_SOURCE) must reject readback with PermissionDenied.
    let desc = BufferDesc::new(
        64,
        4096,
        MemoryDomain::Shared,
        BufferUsage::TRANSFER_DESTINATION | BufferUsage::PROGRAM_INPUT,
    )
    .expect("valid descriptor");
    let (buffer, _) = backend
        .allocate_buffer(&context, desc)
        .expect("allocate")
        .into_parts();
    let mut out = vec![0u8; 8];
    assert!(matches!(
        backend.read_buffer(&buffer, 0, &mut SliceMut(&mut out)),
        Err(BackendError::PermissionDenied)
    ));
    backend.free_buffer(buffer).expect("free");
    backend.destroy_context(context).expect("destroy context");
}

#[test]
fn advertised_limits_are_aggregation_safe() {
    let Some(backend) = backend() else { return };
    // The device-state layer checked-multiplies `max_contexts` by each per-context limit and
    // `max_events_per_context` by `max_bindings_per_submission`; the advertised limits must not
    // overflow those u32 products, or the backend is unusable through the command processor.
    let limits = backend.device_info().expect("device info").limits;
    for per_context in [
        limits.max_buffers_per_context,
        limits.max_programs_per_context,
        limits.max_queues_per_context,
        limits.max_events_per_context,
    ] {
        assert!(limits.max_contexts.checked_mul(per_context).is_some());
    }
    assert!(
        limits
            .max_events_per_context
            .checked_mul(limits.max_bindings_per_submission)
            .is_some()
    );
}

#[test]
fn precompiled_passthrough_runs_the_full_lifecycle() {
    let Some(backend) = backend() else { return };
    let context = backend
        .create_context(ContextDesc::default())
        .expect("context");
    let queue = backend
        .create_queue(&context, QueueDesc::default())
        .expect("queue");

    // Load the precompiled DMA passthrough (the precompiled format ignores the target words).
    let program = backend
        .load_program(
            &context,
            ArtifactRef {
                format: XDNA_PRECOMPILED_FORMAT,
                target: TargetIdentity([0; 12]),
                payload: &Slice(PASSTHROUGH),
                resident_bytes: u64::MAX,
            },
        )
        .expect("load precompiled passthrough");

    let input_desc = BufferDesc::new(
        PASSTHROUGH_BYTES as u64,
        4096,
        MemoryDomain::Shared,
        BufferUsage::TRANSFER_DESTINATION | BufferUsage::PROGRAM_INPUT,
    )
    .unwrap();
    let (mut input, _) = backend
        .allocate_buffer(&context, input_desc)
        .expect("input buffer")
        .into_parts();
    // The design's second input is unused by the DMA copy but still occupies a binding slot.
    let (unused, _) = backend
        .allocate_buffer(&context, input_desc)
        .expect("unused input buffer")
        .into_parts();
    let (output, _) = backend
        .allocate_buffer(
            &context,
            BufferDesc::new(
                PASSTHROUGH_BYTES as u64,
                4096,
                MemoryDomain::Shared,
                BufferUsage::TRANSFER_SOURCE | BufferUsage::PROGRAM_OUTPUT,
            )
            .unwrap(),
        )
        .expect("output buffer")
        .into_parts();

    // Deterministic input pattern.
    let payload: Vec<u8> = (0..PASSTHROUGH_BYTES).map(|i| (i * 31 + 7) as u8).collect();
    backend
        .write_buffer(&mut input, 0, &Slice(&payload))
        .expect("write input");

    let range = BufferRange::new(0, PASSTHROUGH_BYTES as u64).unwrap();
    let bindings = [
        BindingRef {
            slot: 0,
            buffer: &input,
            range,
            access: AccessMode::Read,
        },
        BindingRef {
            slot: 1,
            buffer: &unused,
            range,
            access: AccessMode::Read,
        },
        BindingRef {
            slot: 2,
            buffer: &output,
            range,
            access: AccessMode::Write,
        },
    ];

    let event = match backend.submit(&queue, &program, &bindings, Timeout::Infinite) {
        Ok(event) => event,
        Err(failure) => panic!("submit rejected: {failure:?}"),
    };

    // Poll to a terminal state (nonblocking poll; the worker bridges the blocking synchronize).
    let deadline = Instant::now() + Duration::from_secs(10);
    let state = loop {
        match backend.poll_event(&event).expect("poll") {
            EventState::Pending => {
                assert!(
                    Instant::now() < deadline,
                    "dispatch did not complete in 10s"
                );
                std::thread::yield_now();
            }
            terminal => break terminal,
        }
    };
    assert!(
        matches!(state, EventState::Complete),
        "expected Complete, got {state:?}"
    );

    // The passthrough copies input to output verbatim.
    let mut result = vec![0u8; PASSTHROUGH_BYTES];
    backend
        .read_buffer(&output, 0, &mut SliceMut(&mut result))
        .expect("read output");
    assert_eq!(result, payload, "DMA passthrough must copy input to output");

    backend.destroy_event(event).expect("destroy event");
    backend.free_buffer(input).expect("free input");
    backend.free_buffer(unused).expect("free unused");
    backend.free_buffer(output).expect("free output");
    backend.unload_program(program).expect("unload");
    backend.destroy_queue(queue).expect("destroy queue");
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
