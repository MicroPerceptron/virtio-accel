//! On-hardware tests for the HRX buffer primitives.
//!
//! These compile and run only in a `va_xdna` build (a detected HRX prefix) and require an
//! accessible NPU. They cover the allocate / map / write+flush / read+invalidate / release cycle
//! plus context and queue lifecycle. Program loading and dispatch are covered by the execution
//! ticket.
#![cfg(va_xdna)]

use std::time::{Duration, Instant};

mod common;
use common::{
    bf16_identity_tosa, bf16_matmul_tosa, fp8e4m3_matmul_tosa, fp8e4m3_to_bf16_tosa,
    int8_matmul_tosa, poll_to_terminal,
};

use virtio_accel_conformance::numerics::{
    CAST_FP8E4M3_TO_BF16, CAST_FP8E5M2_TO_BF16, IDENTITY_INT8, MATMUL_INT8, MAX_POOL2D_BF16,
    RESCALE_INT32_TO_INT8, TosaFp8ToBfloat16Case,
};
use virtio_accel_core::{
    Accelerator, AccessMode, ArtifactFormat, ArtifactRef, BackendError, BindingRef, BufferDesc,
    BufferRange, BufferUsage, ByteSink, ByteSource, ContextDesc, EventState, MemoryDomain,
    QueueDesc, ReleaseFailure, SubmitFailure, TargetIdentity, Timeout,
};
use virtio_accel_tosa::{
    ARTIFACT_FORMAT, DType, TosaCapabilityProvider, dot_i8_i32, fp8e4m3_to_bf16_bits,
};
use virtio_accel_tosa_build::{OperatorKind, OwnedGraph, OwnedOperator, OwnedTensor};
use virtio_accel_xdna::{
    XDNA_PRECOMPILED_FORMAT, XDNA_TOSA_FP8_TARGET, XDNA_TOSA_INTEGER_TARGET, XDNA_TOSA_TARGET,
    XdnaAccelerator, XdnaBuffer, XdnaContext, XdnaProgram, XdnaQueue, XdnaResourceCounts,
    compile_artifact,
};
#[cfg(feature = "test-control")]
use virtio_accel_xdna::{XdnaTestConfig, XdnaTestFault};

const REQUIRE_HARDWARE_ENV: &str = "VIRTIO_ACCEL_XDNA_REQUIRE_HARDWARE";

/// Whether this run is the documented manual hardware gate, where skipping would be a false pass.
fn hardware_required() -> bool {
    match std::env::var(REQUIRE_HARDWARE_ENV) {
        Ok(value) if value == "1" => true,
        Ok(value) if value == "0" => false,
        Ok(value) => panic!("{REQUIRE_HARDWARE_ENV} must be \"0\" or \"1\", not {value:?}"),
        Err(std::env::VarError::NotPresent) => false,
        Err(error) => panic!("invalid {REQUIRE_HARDWARE_ENV}: {error}"),
    }
}

/// Whether the pinned compiler toolchain is configured (the compiler tests need it, not a device).
fn toolchain_present() -> bool {
    std::env::var_os("VIRTIO_ACCEL_AMDXDNA_TOOLCHAIN").is_some()
}

/// Encode an exactly-representable value as a little-endian BF16 (truncate the FP32 low half; exact
/// for small integers, whose low mantissa bits are zero).
fn bf16_le(value: f32) -> [u8; 2] {
    ((value.to_bits() >> 16) as u16).to_le_bytes()
}

/// Decode a little-endian FP32 element.
fn f32_le(bytes: &[u8]) -> f32 {
    f32::from_le_bytes(bytes.try_into().expect("4 bytes"))
}

/// Precompiled DMA passthrough for npu2, built with the pinned toolchain from
/// `programming_examples/basic/passthrough_dmas` (n=4096 int32; entry `MLIR_AIE`) and packaged with
/// `virtio_accel_xdna::artifact::encode`. See `tests/data/README.md`. The design declares three
/// runtime buffers — `a_in`, an unused second input `_b_unused`, and `c_out` — so it binds two
/// inputs and one output; the DMA copies the first input to the output.
const PASSTHROUGH: &[u8] = include_bytes!("data/passthrough-dmas-npu2.xdnp");
const PASSTHROUGH_BYTES: usize = 4096 * 4;

struct PassthroughResources {
    context: XdnaContext,
    queue: XdnaQueue,
    program: XdnaProgram,
    input: XdnaBuffer,
    unused: XdnaBuffer,
    output: XdnaBuffer,
}

impl PassthroughResources {
    fn create(backend: &XdnaAccelerator) -> Self {
        let context = backend
            .create_context(ContextDesc::default())
            .expect("context");
        let queue = backend
            .create_queue(&context, QueueDesc::default())
            .expect("queue");
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
            .expect("load passthrough");
        let input_desc = BufferDesc::new(
            PASSTHROUGH_BYTES as u64,
            4096,
            MemoryDomain::Shared,
            BufferUsage::TRANSFER_DESTINATION | BufferUsage::PROGRAM_INPUT,
        )
        .expect("input descriptor");
        let (input, _) = backend
            .allocate_buffer(&context, input_desc)
            .expect("input")
            .into_parts();
        let (unused, _) = backend
            .allocate_buffer(&context, input_desc)
            .expect("unused input")
            .into_parts();
        let output_desc = BufferDesc::new(
            PASSTHROUGH_BYTES as u64,
            4096,
            MemoryDomain::Shared,
            BufferUsage::TRANSFER_SOURCE | BufferUsage::PROGRAM_OUTPUT,
        )
        .expect("output descriptor");
        let (output, _) = backend
            .allocate_buffer(&context, output_desc)
            .expect("output")
            .into_parts();
        Self {
            context,
            queue,
            program,
            input,
            unused,
            output,
        }
    }

    fn bindings(&self) -> [BindingRef<'_, XdnaBuffer>; 3] {
        let range = BufferRange::new(0, PASSTHROUGH_BYTES as u64).expect("binding range");
        [
            BindingRef {
                slot: 0,
                buffer: &self.input,
                range,
                access: AccessMode::Read,
            },
            BindingRef {
                slot: 1,
                buffer: &self.unused,
                range,
                access: AccessMode::Read,
            },
            BindingRef {
                slot: 2,
                buffer: &self.output,
                range,
                access: AccessMode::Write,
            },
        ]
    }

    fn release(self, backend: &XdnaAccelerator) {
        backend.free_buffer(self.input).expect("free input");
        backend.free_buffer(self.unused).expect("free unused");
        backend.free_buffer(self.output).expect("free output");
        backend.unload_program(self.program).expect("unload");
        backend.destroy_queue(self.queue).expect("destroy queue");
        backend
            .destroy_context(self.context)
            .expect("destroy context");
    }
}

/// Construct a backend, or skip when the complete native runtime is unavailable on this host.
fn backend() -> Option<XdnaAccelerator> {
    match XdnaAccelerator::new() {
        Ok(backend) => Some(backend),
        Err(error) => {
            assert!(
                !hardware_required(),
                "{REQUIRE_HARDWARE_ENV}=1 but the XDNA runtime is unusable: {error}"
            );
            eprintln!("XDNA runtime unavailable ({error}); skipping hardware test");
            None
        }
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
    assert_eq!(backend.tosa_capabilities().len(), 3);
    assert_eq!(backend.tosa_capabilities()[0].target, XDNA_TOSA_TARGET);
    assert_eq!(backend.tosa_capabilities()[1].target, XDNA_TOSA_FP8_TARGET);
    assert_eq!(
        backend.tosa_capabilities()[2].target,
        XDNA_TOSA_INTEGER_TARGET
    );
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
fn malformed_and_unsupported_artifacts_leave_no_native_resources() {
    let Some(backend) = backend() else { return };
    let context = backend
        .create_context(ContextDesc::default())
        .expect("context");
    assert_eq!(
        backend.resource_counts(),
        XdnaResourceCounts {
            contexts: 1,
            ..XdnaResourceCounts::default()
        }
    );

    let malformed = b"not-an-xdnp-container";
    assert!(matches!(
        backend.load_program(
            &context,
            ArtifactRef {
                format: XDNA_PRECOMPILED_FORMAT,
                target: TargetIdentity([0; 12]),
                payload: &Slice(malformed),
                resident_bytes: u64::MAX,
            }
        ),
        Err(BackendError::InvalidArgument)
    ));
    let unknown = ArtifactFormat::new(0x554e_4b4e).expect("nonzero format tag");
    assert!(matches!(
        backend.load_program(
            &context,
            ArtifactRef {
                format: unknown,
                target: TargetIdentity([0; 12]),
                payload: &Slice(PASSTHROUGH),
                resident_bytes: u64::MAX,
            }
        ),
        Err(BackendError::Unsupported)
    ));
    assert_eq!(
        backend.resource_counts(),
        XdnaResourceCounts {
            contexts: 1,
            ..XdnaResourceCounts::default()
        },
        "rejected artifacts must not retain an executable"
    );
    backend.destroy_context(context).expect("destroy context");
    assert_eq!(backend.resource_counts(), XdnaResourceCounts::default());
}

#[test]
fn concurrent_submit_is_rejected_without_disturbing_the_accepted_job() {
    let Some(backend) = backend() else { return };
    let resources = PassthroughResources::create(&backend);
    let bindings = resources.bindings();
    let event = backend
        .submit(
            &resources.queue,
            &resources.program,
            &bindings,
            Timeout::Infinite,
        )
        .expect("first submit");
    assert!(matches!(
        backend.submit(
            &resources.queue,
            &resources.program,
            &bindings,
            Timeout::Infinite,
        ),
        Err(SubmitFailure::Rejected(BackendError::Busy))
    ));
    assert_eq!(backend.resource_counts().events, 1);
    let state = poll_to_terminal(&backend, &event, Duration::from_secs(10))
        .expect("accepted job did not complete");
    assert!(matches!(state, EventState::Complete), "got {state:?}");
    backend.destroy_event(event).expect("destroy event");
    resources.release(&backend);
    assert_eq!(backend.resource_counts(), XdnaResourceCounts::default());
}

#[test]
fn pending_releases_return_the_same_live_resources_for_retry() {
    let Some(backend) = backend() else { return };
    let PassthroughResources {
        context,
        queue,
        program,
        input,
        unused,
        output,
    } = PassthroughResources::create(&backend);
    let range = BufferRange::new(0, PASSTHROUGH_BYTES as u64).expect("binding range");
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
    let event = backend
        .submit(&queue, &program, &bindings, Timeout::Infinite)
        .expect("submit");
    let input = match backend.free_buffer(input) {
        Err(ReleaseFailure::Rejected {
            error: BackendError::Busy,
            resource,
        }) => resource,
        other => panic!("in-flight buffer release was not retryable: {other:?}"),
    };
    let event = match backend.destroy_event(event) {
        Err(ReleaseFailure::Rejected {
            error: BackendError::Busy,
            resource,
        }) => resource,
        other => panic!("pending event release was not retryable: {other:?}"),
    };
    assert_eq!(backend.resource_counts().events, 1);
    let state =
        poll_to_terminal(&backend, &event, Duration::from_secs(10)).expect("job did not complete");
    assert!(matches!(state, EventState::Complete), "got {state:?}");
    backend.destroy_event(event).expect("retry event release");
    backend.free_buffer(input).expect("retry input release");
    backend.free_buffer(unused).expect("free unused");
    backend.free_buffer(output).expect("free output");
    backend.unload_program(program).expect("unload");
    backend.destroy_queue(queue).expect("destroy queue");
    backend.destroy_context(context).expect("destroy context");
    assert_eq!(backend.resource_counts(), XdnaResourceCounts::default());
}

#[cfg(feature = "test-control")]
#[test]
fn tier1_device_loss_latches_failure_and_releases_every_resource_once() {
    let backend = XdnaAccelerator::new_for_testing(XdnaTestConfig {
        watchdog_timeout: Duration::from_secs(2),
        fault: XdnaTestFault::Tier1,
    })
    .expect("fault-controlled backend");
    let resources = PassthroughResources::create(&backend);
    let event = backend
        .submit(
            &resources.queue,
            &resources.program,
            &resources.bindings(),
            Timeout::Infinite,
        )
        .expect("accepted tier-1 submission");
    let state = poll_to_terminal(&backend, &event, Duration::from_secs(2))
        .expect("tier-1 event did not become terminal");
    assert_eq!(state, EventState::Failed(BackendError::DeviceLost));
    assert_eq!(backend.poll_event(&event), Ok(state));
    assert!(matches!(
        backend.submit(
            &resources.queue,
            &resources.program,
            &resources.bindings(),
            Timeout::Infinite,
        ),
        Err(SubmitFailure::Rejected(BackendError::DeviceLost))
    ));
    backend.destroy_event(event).expect("destroy failed event");
    resources.release(&backend);
    assert_eq!(backend.resource_counts(), XdnaResourceCounts::default());
}

#[cfg(feature = "test-control")]
#[test]
fn tier2_watchdog_reports_device_loss_and_discard_never_blocks() {
    let backend = XdnaAccelerator::new_for_testing(XdnaTestConfig {
        watchdog_timeout: Duration::from_millis(50),
        fault: XdnaTestFault::Tier2 {
            stall: Duration::from_secs(1),
        },
    })
    .expect("fault-controlled backend");
    let resources = PassthroughResources::create(&backend);
    let event = backend
        .submit(
            &resources.queue,
            &resources.program,
            &resources.bindings(),
            Timeout::Infinite,
        )
        .expect("accepted tier-2 submission");

    let deadline = std::time::Instant::now() + Duration::from_secs(2);
    loop {
        match backend.poll_event(&event) {
            Err(BackendError::DeviceLost) => break,
            Ok(EventState::Pending) if std::time::Instant::now() < deadline => {
                std::thread::yield_now();
            }
            other => panic!("tier-2 watchdog produced {other:?}"),
        }
    }
    let event = match backend.destroy_event(event) {
        Err(ReleaseFailure::Rejected {
            error: BackendError::Busy,
            resource,
        }) => resource,
        other => panic!("wedged event release produced {other:?}"),
    };
    assert!(matches!(
        backend.submit(
            &resources.queue,
            &resources.program,
            &resources.bindings(),
            Timeout::Infinite,
        ),
        Err(SubmitFailure::Rejected(BackendError::DeviceLost))
    ));
    assert_eq!(
        backend.resource_counts(),
        XdnaResourceCounts {
            contexts: 1,
            buffers: 3,
            programs: 1,
            queues: 1,
            events: 1,
        }
    );

    // No HRX call established a trustworthy terminal boundary, so these handles are deliberately
    // discarded with the poisoned instance. The queued job's Arcs quarantine the native buffer
    // and executable handles; dropping the backend detaches rather than joining the wedged worker.
    drop(event);
    drop(resources);
    let started = std::time::Instant::now();
    drop(backend);
    assert!(started.elapsed() < Duration::from_millis(250));
}

/// An event that is dropped rather than destroyed must still release its ring slot. The ring depth
/// is one, so stranding the slot would fail every later submission with `Busy` for the life of the
/// instance — and leave `resource_counts` reporting an event that no longer exists.
#[test]
fn dropping_a_completed_event_reclaims_its_ring_slot() {
    let Some(backend) = backend() else { return };
    let context = backend
        .create_context(ContextDesc::default())
        .expect("context");
    let queue = backend
        .create_queue(&context, QueueDesc::default())
        .expect("queue");
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
    let (input, _) = backend
        .allocate_buffer(&context, input_desc)
        .expect("input buffer")
        .into_parts();
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

    // Two rounds through the single ring slot, releasing the first event by dropping it.
    for round in 0..2 {
        let event = match backend.submit(&queue, &program, &bindings, Timeout::Infinite) {
            Ok(event) => event,
            Err(failure) => panic!("submit rejected in round {round}: {failure:?}"),
        };
        let state = poll_to_terminal(&backend, &event, Duration::from_secs(10))
            .expect("dispatch did not complete in 10s");
        assert!(
            matches!(state, EventState::Complete),
            "round {round}: expected Complete, got {state:?}"
        );
        assert_eq!(backend.resource_counts().events, 1, "round {round}");
        drop(event);
        // The slot and the resource charge are both released by the drop.
        assert_eq!(
            backend.resource_counts().events,
            0,
            "round {round}: dropping a terminal event must release its charge"
        );
    }

    backend.free_buffer(input).expect("free input");
    backend.free_buffer(unused).expect("free unused");
    backend.free_buffer(output).expect("free output");
    backend.unload_program(program).expect("unload");
    backend.destroy_queue(queue).expect("destroy queue");
    backend.destroy_context(context).expect("destroy context");
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
    let state = poll_to_terminal(&backend, &event, Duration::from_secs(10))
        .expect("dispatch did not complete in 10s");
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
fn tosa_bf16_identity_compiles_to_a_wellformed_artifact() {
    // Hardware-free: needs the compiler toolchain but never initializes a device.
    if !toolchain_present() {
        assert!(
            !hardware_required(),
            "{REQUIRE_HARDWARE_ENV}=1 but VIRTIO_ACCEL_AMDXDNA_TOOLCHAIN is not configured"
        );
        eprintln!("no XDNA toolchain configured; skipping compiler test");
        return;
    }
    let tosa = bf16_identity_tosa(4096);
    let container = compile_artifact(&tosa, XDNA_TOSA_TARGET).expect("compile TOSA identity");
    let parsed =
        virtio_accel_xdna::PrecompiledArtifact::parse(&container).expect("valid container");
    // The packaged xclbin is an AMD xclbin container; its instruction stream is TXN words.
    assert!(
        parsed.xclbin.starts_with(b"xclbin2"),
        "expected xclbin2 magic"
    );
    assert!(!parsed.insts.is_empty() && parsed.insts.len() % 4 == 0);
    assert_eq!((parsed.inputs, parsed.outputs), (1, 1));
    assert_eq!(parsed.entry, "MLIR_AIE");

    // Non-subset graphs are rejected before any compile runs.
    let fp32 = {
        let shape = vec![1, 1, 4096];
        let mut g = OwnedGraph::new("main");
        g.push_tensor(OwnedTensor::new("x", shape.clone(), DType::FP32));
        g.push_tensor(OwnedTensor::new("y", shape, DType::FP32));
        g.push_operator(OwnedOperator::new(
            OperatorKind::Identity,
            vec!["x".into()],
            vec!["y".into()],
        ));
        g.push_input("x");
        g.push_output("y");
        g.build(XDNA_TOSA_TARGET).expect("build fp32 identity")
    };
    assert!(matches!(
        compile_artifact(&fp32, XDNA_TOSA_TARGET),
        Err(BackendError::Unsupported)
    ));
}

#[test]
fn tosa_bf16_identity_runs_on_the_npu() {
    let Some(backend) = backend() else { return };
    if !toolchain_present() {
        assert!(
            !hardware_required(),
            "{REQUIRE_HARDWARE_ENV}=1 but VIRTIO_ACCEL_AMDXDNA_TOOLCHAIN is not configured"
        );
        eprintln!("no XDNA toolchain configured; skipping TOSA execution test");
        return;
    }
    const ELEMENTS: usize = 4096;
    const BYTES: usize = ELEMENTS * 2; // bf16

    let context = backend
        .create_context(ContextDesc::default())
        .expect("context");
    let queue = backend
        .create_queue(&context, QueueDesc::default())
        .expect("queue");
    let tosa = bf16_identity_tosa(ELEMENTS);
    let program = backend
        .load_program(
            &context,
            ArtifactRef {
                format: ARTIFACT_FORMAT,
                target: XDNA_TOSA_TARGET.to_identity(),
                payload: &Slice(&tosa),
                resident_bytes: u64::MAX,
            },
        )
        .expect("load + compile TOSA identity");

    let (mut input, _) = backend
        .allocate_buffer(
            &context,
            BufferDesc::new(
                BYTES as u64,
                4096,
                MemoryDomain::Shared,
                BufferUsage::TRANSFER_DESTINATION | BufferUsage::PROGRAM_INPUT,
            )
            .unwrap(),
        )
        .expect("input")
        .into_parts();
    let (output, _) = backend
        .allocate_buffer(
            &context,
            BufferDesc::new(
                BYTES as u64,
                4096,
                MemoryDomain::Shared,
                BufferUsage::TRANSFER_SOURCE | BufferUsage::PROGRAM_OUTPUT,
            )
            .unwrap(),
        )
        .expect("output")
        .into_parts();

    let payload: Vec<u8> = (0..BYTES).map(|i| (i * 13 + 5) as u8).collect();
    backend
        .write_buffer(&mut input, 0, &Slice(&payload))
        .expect("write input");
    let range = BufferRange::new(0, BYTES as u64).unwrap();
    let bindings = [
        BindingRef {
            slot: 0,
            buffer: &input,
            range,
            access: AccessMode::Read,
        },
        BindingRef {
            slot: 1,
            buffer: &output,
            range,
            access: AccessMode::Write,
        },
    ];
    let event = backend
        .submit(&queue, &program, &bindings, Timeout::Infinite)
        .expect("submit");
    let state = poll_to_terminal(&backend, &event, Duration::from_secs(10))
        .expect("identity did not complete");
    assert!(matches!(state, EventState::Complete), "got {state:?}");

    let mut result = vec![0u8; BYTES];
    backend
        .read_buffer(&output, 0, &mut SliceMut(&mut result))
        .expect("read output");
    assert_eq!(result, payload, "TOSA IDENTITY must copy input to output");

    backend.destroy_event(event).expect("destroy event");
    backend.free_buffer(input).expect("free input");
    backend.free_buffer(output).expect("free output");
    backend.unload_program(program).expect("unload");
    backend.destroy_queue(queue).expect("destroy queue");
    backend.destroy_context(context).expect("destroy context");
}

#[test]
fn tosa_integer_corpus_compiles_to_wellformed_artifacts() {
    if !toolchain_present() {
        assert!(
            !hardware_required(),
            "{REQUIRE_HARDWARE_ENV}=1 but VIRTIO_ACCEL_AMDXDNA_TOOLCHAIN is not configured"
        );
        eprintln!("no XDNA toolchain configured; skipping compiler test");
        return;
    }
    for (case_name, artifact, expected_slots) in [
        (IDENTITY_INT8.name, IDENTITY_INT8.artifact, &[8, 8][..]),
        (MATMUL_INT8.name, MATMUL_INT8.artifact, &[8, 8, 16][..]),
        (
            RESCALE_INT32_TO_INT8.name,
            RESCALE_INT32_TO_INT8.artifact,
            &[64, 16][..],
        ),
    ] {
        let container = compile_artifact(artifact, XDNA_TOSA_INTEGER_TARGET)
            .unwrap_or_else(|error| panic!("compile {case_name}: {error:?}"));
        let parsed = virtio_accel_xdna::PrecompiledArtifact::parse(&container)
            .unwrap_or_else(|error| panic!("parse {case_name}: {error:?}"));
        assert!(parsed.xclbin.starts_with(b"xclbin2"));
        assert!(!parsed.insts.is_empty() && parsed.insts.len() % 4 == 0);
        assert_eq!(parsed.slot_bytes, expected_slots);
        assert_eq!(parsed.entry, "MLIR_AIE");
    }
}

#[test]
fn tosa_int8_identity_matches_the_shared_exact_oracle_on_the_npu() {
    let Some(backend) = backend() else { return };
    if !toolchain_present() {
        assert!(
            !hardware_required(),
            "{REQUIRE_HARDWARE_ENV}=1 but VIRTIO_ACCEL_AMDXDNA_TOOLCHAIN is not configured"
        );
        eprintln!("no XDNA toolchain configured; skipping TOSA execution test");
        return;
    }
    let context = backend
        .create_context(ContextDesc::default())
        .expect("context");
    let queue = backend
        .create_queue(&context, QueueDesc::default())
        .expect("queue");
    let program = backend
        .load_program(
            &context,
            ArtifactRef {
                format: ARTIFACT_FORMAT,
                target: XDNA_TOSA_INTEGER_TARGET.to_identity(),
                payload: &Slice(IDENTITY_INT8.artifact),
                resident_bytes: u64::MAX,
            },
        )
        .expect("load INT8 identity");
    let input_bytes = IDENTITY_INT8.inputs[0].bytes;
    let output_len = IDENTITY_INT8.outputs[0].bytes.len();
    let (mut input, _) = backend
        .allocate_buffer(
            &context,
            BufferDesc::new(
                input_bytes.len() as u64,
                4096,
                MemoryDomain::Shared,
                BufferUsage::TRANSFER_DESTINATION | BufferUsage::PROGRAM_INPUT,
            )
            .unwrap(),
        )
        .expect("input buffer")
        .into_parts();
    let (output, _) = backend
        .allocate_buffer(
            &context,
            BufferDesc::new(
                output_len as u64,
                4096,
                MemoryDomain::Shared,
                BufferUsage::TRANSFER_SOURCE | BufferUsage::PROGRAM_OUTPUT,
            )
            .unwrap(),
        )
        .expect("output buffer")
        .into_parts();
    backend
        .write_buffer(&mut input, 0, &Slice(input_bytes))
        .expect("write INT8 input");

    let direct_before = backend.direct_binding_admissions();
    let transfer_before = backend.explicit_transfer_bytes();
    let event = backend
        .submit(
            &queue,
            &program,
            &[
                BindingRef {
                    slot: 0,
                    buffer: &input,
                    range: BufferRange::new(0, input_bytes.len() as u64).unwrap(),
                    access: AccessMode::Read,
                },
                BindingRef {
                    slot: 1,
                    buffer: &output,
                    range: BufferRange::new(0, output_len as u64).unwrap(),
                    access: AccessMode::Write,
                },
            ],
            Timeout::Infinite,
        )
        .expect("submit INT8 identity");
    let state = poll_to_terminal(&backend, &event, Duration::from_secs(30))
        .expect("INT8 identity did not complete");
    assert!(matches!(state, EventState::Complete), "got {state:?}");
    assert_eq!(backend.direct_binding_admissions() - direct_before, 2);
    assert_eq!(backend.explicit_transfer_bytes() - transfer_before, 0);

    let mut result = vec![0u8; output_len];
    backend
        .read_buffer(&output, 0, &mut SliceMut(&mut result))
        .expect("read INT8 output");
    assert!(IDENTITY_INT8.output_matches(0, &result));

    backend.destroy_event(event).expect("destroy event");
    backend.free_buffer(input).expect("free input");
    backend.free_buffer(output).expect("free output");
    backend.unload_program(program).expect("unload program");
    backend.destroy_queue(queue).expect("destroy queue");
    backend.destroy_context(context).expect("destroy context");
}

#[test]
fn tosa_int8_matmul_matches_the_shared_exact_oracle_on_the_npu() {
    let Some(backend) = backend() else { return };
    if !toolchain_present() {
        assert!(
            !hardware_required(),
            "{REQUIRE_HARDWARE_ENV}=1 but VIRTIO_ACCEL_AMDXDNA_TOOLCHAIN is not configured"
        );
        eprintln!("no XDNA toolchain configured; skipping TOSA execution test");
        return;
    }
    let context = backend
        .create_context(ContextDesc::default())
        .expect("context");
    let queue = backend
        .create_queue(&context, QueueDesc::default())
        .expect("queue");
    let program = backend
        .load_program(
            &context,
            ArtifactRef {
                format: ARTIFACT_FORMAT,
                target: XDNA_TOSA_INTEGER_TARGET.to_identity(),
                payload: &Slice(MATMUL_INT8.artifact),
                resident_bytes: u64::MAX,
            },
        )
        .expect("load INT8 matmul");
    let input_desc = |bytes: usize| {
        BufferDesc::new(
            bytes as u64,
            4096,
            MemoryDomain::Shared,
            BufferUsage::TRANSFER_DESTINATION | BufferUsage::PROGRAM_INPUT,
        )
        .unwrap()
    };
    let lhs_len = (MATMUL_INT8.inputs[0].bytes.len() + 3) & !3;
    let rhs_len = (MATMUL_INT8.inputs[1].bytes.len() + 3) & !3;
    let (mut lhs, _) = backend
        .allocate_buffer(&context, input_desc(lhs_len))
        .expect("lhs buffer")
        .into_parts();
    let (mut rhs, _) = backend
        .allocate_buffer(&context, input_desc(rhs_len))
        .expect("rhs buffer")
        .into_parts();
    let output_len = MATMUL_INT8.outputs[0].values.len() * 4;
    let (output, _) = backend
        .allocate_buffer(
            &context,
            BufferDesc::new(
                output_len as u64,
                4096,
                MemoryDomain::Shared,
                BufferUsage::TRANSFER_SOURCE | BufferUsage::PROGRAM_OUTPUT,
            )
            .unwrap(),
        )
        .expect("output buffer")
        .into_parts();
    let mut lhs_bytes = vec![0u8; lhs_len];
    lhs_bytes[..MATMUL_INT8.inputs[0].bytes.len()].copy_from_slice(MATMUL_INT8.inputs[0].bytes);
    let mut rhs_bytes = vec![0u8; rhs_len];
    rhs_bytes[..MATMUL_INT8.inputs[1].bytes.len()].copy_from_slice(MATMUL_INT8.inputs[1].bytes);
    backend
        .write_buffer(&mut lhs, 0, &Slice(&lhs_bytes))
        .expect("write lhs");
    backend
        .write_buffer(&mut rhs, 0, &Slice(&rhs_bytes))
        .expect("write rhs");

    let direct_before = backend.direct_binding_admissions();
    let transfer_before = backend.explicit_transfer_bytes();
    let event = backend
        .submit(
            &queue,
            &program,
            &[
                BindingRef {
                    slot: 0,
                    buffer: &lhs,
                    range: BufferRange::new(0, lhs_len as u64).unwrap(),
                    access: AccessMode::Read,
                },
                BindingRef {
                    slot: 1,
                    buffer: &rhs,
                    range: BufferRange::new(0, rhs_len as u64).unwrap(),
                    access: AccessMode::Read,
                },
                BindingRef {
                    slot: 2,
                    buffer: &output,
                    range: BufferRange::new(0, output_len as u64).unwrap(),
                    access: AccessMode::Write,
                },
            ],
            Timeout::Infinite,
        )
        .expect("submit INT8 matmul");
    let state = poll_to_terminal(&backend, &event, Duration::from_secs(30))
        .expect("INT8 matmul did not complete");
    assert!(matches!(state, EventState::Complete), "got {state:?}");
    assert_eq!(backend.direct_binding_admissions() - direct_before, 3);
    assert_eq!(backend.explicit_transfer_bytes() - transfer_before, 0);

    let mut result = vec![0u8; output_len];
    backend
        .read_buffer(&output, 0, &mut SliceMut(&mut result))
        .expect("read INT32 output");
    let values: Vec<_> = result
        .chunks_exact(4)
        .map(|bytes| i32::from_le_bytes(bytes.try_into().expect("four-byte chunk")))
        .collect();
    assert!(MATMUL_INT8.output_matches(0, &values));

    backend.destroy_event(event).expect("destroy event");
    backend.free_buffer(lhs).expect("free lhs");
    backend.free_buffer(rhs).expect("free rhs");
    backend.free_buffer(output).expect("free output");
    backend.unload_program(program).expect("unload program");
    backend.destroy_queue(queue).expect("destroy queue");
    backend.destroy_context(context).expect("destroy context");
}

/// The DMA-tiled INT8 MATMUL path (shapes on the native 8x8x8 MMUL tiling) matches the shared
/// exact oracle. The corpus fixture (2x3x2) exercises only the scalar fallback, so without this
/// test the tier's fast path would be oracle-checked exclusively by an `#[ignore]`d benchmark.
/// Shapes cover the minimal tile, an asymmetric multi-tile, and both zero-point extremes; inputs
/// sweep the full INT8 range including -128.
#[test]
fn tosa_int8_matmul_tiled_path_matches_the_exact_oracle_on_the_npu() {
    let Some(backend) = backend() else { return };
    if !toolchain_present() {
        assert!(
            !hardware_required(),
            "{REQUIRE_HARDWARE_ENV}=1 but VIRTIO_ACCEL_AMDXDNA_TOOLCHAIN is not configured"
        );
        eprintln!("no XDNA toolchain configured; skipping TOSA execution test");
        return;
    }
    let context = backend
        .create_context(ContextDesc::default())
        .expect("context");
    let queue = backend
        .create_queue(&context, QueueDesc::default())
        .expect("queue");

    for (m, k, n, left_zero_point, right_zero_point) in [
        (16usize, 16usize, 16usize, -128i8, 127i8),
        (32, 24, 16, 127, -128),
        (64, 64, 32, 0, -1),
    ] {
        let tosa = int8_matmul_tosa(
            m as i32,
            k as i32,
            n as i32,
            left_zero_point,
            right_zero_point,
        );
        let program = backend
            .load_program(
                &context,
                ArtifactRef {
                    format: ARTIFACT_FORMAT,
                    target: XDNA_TOSA_INTEGER_TARGET.to_identity(),
                    payload: &Slice(&tosa),
                    resident_bytes: u64::MAX,
                },
            )
            .expect("load tiled INT8 matmul");

        let lhs_len = m * k;
        let rhs_len = k * n;
        let output_len = m * n * 4;
        let input_desc = |bytes: usize| {
            BufferDesc::new(
                bytes as u64,
                4096,
                MemoryDomain::Shared,
                BufferUsage::TRANSFER_DESTINATION | BufferUsage::PROGRAM_INPUT,
            )
            .unwrap()
        };
        let (mut lhs, _) = backend
            .allocate_buffer(&context, input_desc(lhs_len))
            .expect("lhs buffer")
            .into_parts();
        let (mut rhs, _) = backend
            .allocate_buffer(&context, input_desc(rhs_len))
            .expect("rhs buffer")
            .into_parts();
        let (output, _) = backend
            .allocate_buffer(
                &context,
                BufferDesc::new(
                    output_len as u64,
                    4096,
                    MemoryDomain::Shared,
                    BufferUsage::TRANSFER_SOURCE | BufferUsage::PROGRAM_OUTPUT,
                )
                .unwrap(),
            )
            .expect("output buffer")
            .into_parts();

        // Deterministic full-range INT8 patterns (both extremes appear in every operand).
        let lhs_bytes: Vec<u8> = (0..lhs_len).map(|i| (i * 37 + 11) as u8).collect();
        let rhs_bytes: Vec<u8> = (0..rhs_len).map(|i| (i * 53 + 197) as u8).collect();
        backend
            .write_buffer(&mut lhs, 0, &Slice(&lhs_bytes))
            .expect("write lhs");
        backend
            .write_buffer(&mut rhs, 0, &Slice(&rhs_bytes))
            .expect("write rhs");

        let event = backend
            .submit(
                &queue,
                &program,
                &[
                    BindingRef {
                        slot: 0,
                        buffer: &lhs,
                        range: BufferRange::new(0, lhs_len as u64).unwrap(),
                        access: AccessMode::Read,
                    },
                    BindingRef {
                        slot: 1,
                        buffer: &rhs,
                        range: BufferRange::new(0, rhs_len as u64).unwrap(),
                        access: AccessMode::Read,
                    },
                    BindingRef {
                        slot: 2,
                        buffer: &output,
                        range: BufferRange::new(0, output_len as u64).unwrap(),
                        access: AccessMode::Write,
                    },
                ],
                Timeout::Infinite,
            )
            .expect("submit tiled INT8 matmul");
        let state = poll_to_terminal(&backend, &event, Duration::from_secs(30))
            .expect("tiled INT8 matmul did not complete");
        assert!(matches!(state, EventState::Complete), "got {state:?}");

        let mut result = vec![0u8; output_len];
        backend
            .read_buffer(&output, 0, &mut SliceMut(&mut result))
            .expect("read INT32 output");
        for row in 0..m {
            for column in 0..n {
                let left_row = &lhs_bytes[row * k..(row + 1) * k];
                let right_column: Vec<u8> = (0..k).map(|i| rhs_bytes[i * n + column]).collect();
                let expected = dot_i8_i32(
                    left_row,
                    &right_column,
                    left_zero_point,
                    right_zero_point,
                    0,
                )
                .expect("exact oracle");
                let actual = i32::from_le_bytes(
                    result[(row * n + column) * 4..][..4]
                        .try_into()
                        .expect("four-byte element"),
                );
                assert_eq!(
                    actual, expected,
                    "{m}x{k}x{n} zp=[{left_zero_point},{right_zero_point}]: \
                     C[{row},{column}] oracle mismatch"
                );
            }
        }

        backend.destroy_event(event).expect("destroy event");
        backend.free_buffer(lhs).expect("free lhs");
        backend.free_buffer(rhs).expect("free rhs");
        backend.free_buffer(output).expect("free output");
        backend.unload_program(program).expect("unload program");
    }

    backend.destroy_queue(queue).expect("destroy queue");
    backend.destroy_context(context).expect("destroy context");
}

#[test]
fn tosa_int32_to_int8_rescale_matches_the_shared_exact_oracle_on_the_npu() {
    let Some(backend) = backend() else { return };
    if !toolchain_present() {
        assert!(
            !hardware_required(),
            "{REQUIRE_HARDWARE_ENV}=1 but VIRTIO_ACCEL_AMDXDNA_TOOLCHAIN is not configured"
        );
        eprintln!("no XDNA toolchain configured; skipping TOSA execution test");
        return;
    }
    let context = backend
        .create_context(ContextDesc::default())
        .expect("context");
    let queue = backend
        .create_queue(&context, QueueDesc::default())
        .expect("queue");
    let program = backend
        .load_program(
            &context,
            ArtifactRef {
                format: ARTIFACT_FORMAT,
                target: XDNA_TOSA_INTEGER_TARGET.to_identity(),
                payload: &Slice(RESCALE_INT32_TO_INT8.artifact),
                resident_bytes: u64::MAX,
            },
        )
        .expect("load INT32-to-INT8 rescale");
    let input_bytes: Vec<_> = RESCALE_INT32_TO_INT8
        .input
        .values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    let output_len = RESCALE_INT32_TO_INT8.output.bytes.len();
    let (mut input, _) = backend
        .allocate_buffer(
            &context,
            BufferDesc::new(
                input_bytes.len() as u64,
                4096,
                MemoryDomain::Shared,
                BufferUsage::TRANSFER_DESTINATION | BufferUsage::PROGRAM_INPUT,
            )
            .unwrap(),
        )
        .expect("input buffer")
        .into_parts();
    let (output, _) = backend
        .allocate_buffer(
            &context,
            BufferDesc::new(
                output_len as u64,
                4096,
                MemoryDomain::Shared,
                BufferUsage::TRANSFER_SOURCE | BufferUsage::PROGRAM_OUTPUT,
            )
            .unwrap(),
        )
        .expect("output buffer")
        .into_parts();
    backend
        .write_buffer(&mut input, 0, &Slice(&input_bytes))
        .expect("write INT32 input");

    let direct_before = backend.direct_binding_admissions();
    let transfer_before = backend.explicit_transfer_bytes();
    let event = backend
        .submit(
            &queue,
            &program,
            &[
                BindingRef {
                    slot: 0,
                    buffer: &input,
                    range: BufferRange::new(0, input_bytes.len() as u64).unwrap(),
                    access: AccessMode::Read,
                },
                BindingRef {
                    slot: 1,
                    buffer: &output,
                    range: BufferRange::new(0, output_len as u64).unwrap(),
                    access: AccessMode::Write,
                },
            ],
            Timeout::Infinite,
        )
        .expect("submit INT32-to-INT8 rescale");
    let state = poll_to_terminal(&backend, &event, Duration::from_secs(30))
        .expect("INT32-to-INT8 rescale did not complete");
    assert!(matches!(state, EventState::Complete), "got {state:?}");
    assert_eq!(backend.direct_binding_admissions() - direct_before, 2);
    assert_eq!(backend.explicit_transfer_bytes() - transfer_before, 0);

    let mut result = vec![0u8; output_len];
    backend
        .read_buffer(&output, 0, &mut SliceMut(&mut result))
        .expect("read INT8 output");
    assert!(RESCALE_INT32_TO_INT8.output_matches(&result));

    backend.destroy_event(event).expect("destroy event");
    backend.free_buffer(input).expect("free input");
    backend.free_buffer(output).expect("free output");
    backend.unload_program(program).expect("unload program");
    backend.destroy_queue(queue).expect("destroy queue");
    backend.destroy_context(context).expect("destroy context");
}

#[test]
#[ignore = "manual native performance evidence"]
fn measures_exact_int8_matmul_latency() {
    let Some(backend) = backend() else { return };
    if !toolchain_present() {
        assert!(
            !hardware_required(),
            "{REQUIRE_HARDWARE_ENV}=1 but VIRTIO_ACCEL_AMDXDNA_TOOLCHAIN is not configured"
        );
        eprintln!("no XDNA toolchain configured; skipping INT8 benchmark");
        return;
    }
    const M: usize = 64;
    const K: usize = 64;
    const N: usize = 32;
    const LEFT_ZERO_POINT: i8 = -2;
    const RIGHT_ZERO_POINT: i8 = 3;
    const WARMUPS: usize = 20;
    const SAMPLES: usize = 200;

    let artifact = int8_matmul_tosa(
        M as i32,
        K as i32,
        N as i32,
        LEFT_ZERO_POINT,
        RIGHT_ZERO_POINT,
    );
    let context = backend
        .create_context(ContextDesc::default())
        .expect("context");
    let queue = backend
        .create_queue(&context, QueueDesc::default())
        .expect("queue");
    let program = backend
        .load_program(
            &context,
            ArtifactRef {
                format: ARTIFACT_FORMAT,
                target: XDNA_TOSA_INTEGER_TARGET.to_identity(),
                payload: &Slice(&artifact),
                resident_bytes: u64::MAX,
            },
        )
        .expect("load INT8 benchmark program");
    let lhs_bytes: Vec<u8> = (0..M * K).map(|index| (index * 17 + 3) as u8).collect();
    let rhs_bytes: Vec<u8> = (0..K * N).map(|index| (index * 29 + 11) as u8).collect();
    let output_len = M * N * 4;
    let input_desc = |bytes: usize| {
        BufferDesc::new(
            bytes as u64,
            4096,
            MemoryDomain::Shared,
            BufferUsage::TRANSFER_DESTINATION | BufferUsage::PROGRAM_INPUT,
        )
        .expect("input descriptor")
    };
    let (mut lhs, _) = backend
        .allocate_buffer(&context, input_desc(lhs_bytes.len()))
        .expect("lhs buffer")
        .into_parts();
    let (mut rhs, _) = backend
        .allocate_buffer(&context, input_desc(rhs_bytes.len()))
        .expect("rhs buffer")
        .into_parts();
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
        .expect("output buffer")
        .into_parts();
    backend
        .write_buffer(&mut lhs, 0, &Slice(&lhs_bytes))
        .expect("initialize lhs");
    backend
        .write_buffer(&mut rhs, 0, &Slice(&rhs_bytes))
        .expect("initialize rhs");

    let direct_before = backend.direct_binding_admissions();
    let transfer_before = backend.explicit_transfer_bytes();
    let submit_once = || {
        let bindings = [
            BindingRef {
                slot: 0,
                buffer: &lhs,
                range: BufferRange::new(0, lhs_bytes.len() as u64).expect("lhs range"),
                access: AccessMode::Read,
            },
            BindingRef {
                slot: 1,
                buffer: &rhs,
                range: BufferRange::new(0, rhs_bytes.len() as u64).expect("rhs range"),
                access: AccessMode::Read,
            },
            BindingRef {
                slot: 2,
                buffer: &output,
                range: BufferRange::new(0, output_len as u64).expect("output range"),
                access: AccessMode::Write,
            },
        ];
        let started = Instant::now();
        let event = backend
            .submit(&queue, &program, &bindings, Timeout::Infinite)
            .expect("warm INT8 submission");
        let admission = started.elapsed();
        let state = poll_to_terminal(&backend, &event, Duration::from_secs(30))
            .expect("INT8 benchmark completion");
        assert!(matches!(state, EventState::Complete), "got {state:?}");
        let completion = started.elapsed();
        backend.destroy_event(event).expect("destroy event");
        (admission, completion)
    };
    for _ in 0..WARMUPS {
        submit_once();
    }
    let (mut admission, mut completion): (Vec<_>, Vec<_>) =
        (0..SAMPLES).map(|_| submit_once()).unzip();
    admission.sort_unstable();
    completion.sort_unstable();

    let submissions = (WARMUPS + SAMPLES) as u64;
    let direct_bindings = backend.direct_binding_admissions() - direct_before;
    let explicit_transfer_bytes = backend.explicit_transfer_bytes() - transfer_before;
    assert_eq!(direct_bindings, submissions * 3);
    assert_eq!(explicit_transfer_bytes, 0);

    let mut result = vec![0u8; output_len];
    backend
        .read_buffer(&output, 0, &mut SliceMut(&mut result))
        .expect("read benchmark output");
    for row in 0..M {
        for column in 0..N {
            let right: Vec<_> = (0..K).map(|inner| rhs_bytes[inner * N + column]).collect();
            let expected = dot_i8_i32(
                &lhs_bytes[row * K..(row + 1) * K],
                &right,
                LEFT_ZERO_POINT,
                RIGHT_ZERO_POINT,
                0,
            )
            .expect("exact oracle");
            let offset = (row * N + column) * 4;
            let actual = i32::from_le_bytes(
                result[offset..offset + 4]
                    .try_into()
                    .expect("four-byte output"),
            );
            assert_eq!(actual, expected, "C[{row},{column}] oracle mismatch");
        }
    }

    let p50 = completion[SAMPLES / 2];
    let operations = (2 * M * K * N) as f64;
    let gigaops = operations / p50.as_secs_f64() / 1_000_000_000.0;
    eprintln!(
        "XDNA exact INT8 MATMUL: shape=1x{M}x{K} · 1x{K}x{N}; worker=1; zero_points=[{LEFT_ZERO_POINT},{RIGHT_ZERO_POINT}]; warmups={WARMUPS}; samples={SAMPLES}; admission p50={:?} p95={:?}; submit-to-complete p50={:?} p95={:?}; effective={gigaops:.3} GOPS; direct_bindings={direct_bindings}; explicit_transfer_bytes={explicit_transfer_bytes}",
        admission[SAMPLES / 2],
        admission[SAMPLES * 95 / 100],
        p50,
        completion[SAMPLES * 95 / 100],
    );

    backend.free_buffer(lhs).expect("free lhs");
    backend.free_buffer(rhs).expect("free rhs");
    backend.free_buffer(output).expect("free output");
    backend.unload_program(program).expect("unload program");
    backend.destroy_queue(queue).expect("destroy queue");
    backend.destroy_context(context).expect("destroy context");
}

#[test]
fn tosa_fp8_casts_compile_to_wellformed_artifacts() {
    if !toolchain_present() {
        assert!(
            !hardware_required(),
            "{REQUIRE_HARDWARE_ENV}=1 but VIRTIO_ACCEL_AMDXDNA_TOOLCHAIN is not configured"
        );
        eprintln!("no XDNA toolchain configured; skipping compiler test");
        return;
    }
    for case in [CAST_FP8E4M3_TO_BF16, CAST_FP8E5M2_TO_BF16] {
        let container = compile_artifact(case.artifact, XDNA_TOSA_FP8_TARGET)
            .unwrap_or_else(|error| panic!("compile {}: {error:?}", case.name));
        let parsed = virtio_accel_xdna::PrecompiledArtifact::parse(&container)
            .unwrap_or_else(|error| panic!("parse {}: {error:?}", case.name));
        assert!(parsed.xclbin.starts_with(b"xclbin2"));
        assert!(!parsed.insts.is_empty() && parsed.insts.len() % 4 == 0);
        assert_eq!((parsed.inputs, parsed.outputs), (1, 1));
        assert_eq!(parsed.slot_bytes, [1024, 2048]);
        assert_eq!(parsed.entry, "MLIR_AIE");
    }
}

fn run_fp8_cast_case(backend: &XdnaAccelerator, case: TosaFp8ToBfloat16Case) {
    let context = backend
        .create_context(ContextDesc::default())
        .expect("context");
    let queue = backend
        .create_queue(&context, QueueDesc::default())
        .expect("queue");
    let program = backend
        .load_program(
            &context,
            ArtifactRef {
                format: ARTIFACT_FORMAT,
                target: XDNA_TOSA_FP8_TARGET.to_identity(),
                payload: &Slice(case.artifact),
                resident_bytes: u64::MAX,
            },
        )
        .unwrap_or_else(|error| panic!("load {}: {error:?}", case.name));

    let input_len = case.input.bytes.len();
    let output_len = case.output.bits.len() * 2;
    let (mut input, _) = backend
        .allocate_buffer(
            &context,
            BufferDesc::new(
                input_len as u64,
                4096,
                MemoryDomain::Shared,
                BufferUsage::TRANSFER_DESTINATION | BufferUsage::PROGRAM_INPUT,
            )
            .unwrap(),
        )
        .expect("input buffer")
        .into_parts();
    let (output, _) = backend
        .allocate_buffer(
            &context,
            BufferDesc::new(
                output_len as u64,
                4096,
                MemoryDomain::Shared,
                BufferUsage::TRANSFER_SOURCE | BufferUsage::PROGRAM_OUTPUT,
            )
            .unwrap(),
        )
        .expect("output buffer")
        .into_parts();
    backend
        .write_buffer(&mut input, 0, &Slice(case.input.bytes))
        .expect("write FP8 input");
    let event = backend
        .submit(
            &queue,
            &program,
            &[
                BindingRef {
                    slot: 0,
                    buffer: &input,
                    range: BufferRange::new(0, input_len as u64).unwrap(),
                    access: AccessMode::Read,
                },
                BindingRef {
                    slot: 1,
                    buffer: &output,
                    range: BufferRange::new(0, output_len as u64).unwrap(),
                    access: AccessMode::Write,
                },
            ],
            Timeout::Infinite,
        )
        .unwrap_or_else(|error| panic!("submit {}: {error:?}", case.name));
    let state = poll_to_terminal(backend, &event, Duration::from_secs(30))
        .unwrap_or_else(|error| panic!("poll {}: {error:?}", case.name));
    assert!(matches!(state, EventState::Complete), "got {state:?}");

    let mut result = vec![0u8; output_len];
    backend
        .read_buffer(&output, 0, &mut SliceMut(&mut result))
        .expect("read BF16 output");
    let result_bits: Vec<_> = result
        .chunks_exact(2)
        .map(|bytes| u16::from_le_bytes([bytes[0], bytes[1]]))
        .collect();
    assert!(
        case.output_matches(&result_bits),
        "{} oracle mismatch",
        case.name
    );

    backend.destroy_event(event).expect("destroy event");
    backend.free_buffer(input).expect("free input");
    backend.free_buffer(output).expect("free output");
    backend.unload_program(program).expect("unload");
    backend.destroy_queue(queue).expect("destroy queue");
    backend.destroy_context(context).expect("destroy context");
}

#[test]
fn tosa_fp8_casts_match_the_shared_bit_exact_oracles_on_the_npu() {
    let Some(backend) = backend() else { return };
    if !toolchain_present() {
        assert!(
            !hardware_required(),
            "{REQUIRE_HARDWARE_ENV}=1 but VIRTIO_ACCEL_AMDXDNA_TOOLCHAIN is not configured"
        );
        eprintln!("no XDNA toolchain configured; skipping TOSA execution test");
        return;
    }
    for case in [CAST_FP8E4M3_TO_BF16, CAST_FP8E5M2_TO_BF16] {
        run_fp8_cast_case(&backend, case);
    }
}

#[test]
#[ignore = "manual native performance evidence"]
fn measures_fp8_cast_scaling_on_one_aie_worker() {
    let Some(backend) = backend() else { return };
    if !toolchain_present() {
        assert!(
            !hardware_required(),
            "{REQUIRE_HARDWARE_ENV}=1 but VIRTIO_ACCEL_AMDXDNA_TOOLCHAIN is not configured"
        );
        eprintln!("no XDNA toolchain configured; skipping FP8 benchmark");
        return;
    }

    const WARMUPS: usize = 20;
    const SAMPLES: usize = 200;
    const ELEMENT_COUNTS: [usize; 4] = [1_024, 16_384, 262_144, 1_048_576];

    for elements in ELEMENT_COUNTS {
        measure_fp8_cast_size(&backend, elements, WARMUPS, SAMPLES);
    }
}

fn measure_fp8_cast_size(
    backend: &XdnaAccelerator,
    elements: usize,
    warmups: usize,
    samples: usize,
) {
    let artifact = fp8e4m3_to_bf16_tosa(elements);
    let context = backend
        .create_context(ContextDesc::default())
        .expect("context");
    let queue = backend
        .create_queue(&context, QueueDesc::default())
        .expect("queue");
    let program = backend
        .load_program(
            &context,
            ArtifactRef {
                format: ARTIFACT_FORMAT,
                target: XDNA_TOSA_FP8_TARGET.to_identity(),
                payload: &Slice(&artifact),
                resident_bytes: u64::MAX,
            },
        )
        .expect("load FP8 benchmark program");

    let input_len = elements;
    let output_len = elements * 2;
    let input_desc = BufferDesc::new(
        input_len as u64,
        4096,
        MemoryDomain::Shared,
        BufferUsage::TRANSFER_DESTINATION | BufferUsage::PROGRAM_INPUT,
    )
    .expect("input descriptor");
    let output_desc = BufferDesc::new(
        output_len as u64,
        4096,
        MemoryDomain::Shared,
        BufferUsage::TRANSFER_SOURCE | BufferUsage::PROGRAM_OUTPUT,
    )
    .expect("output descriptor");
    let (mut input, _) = backend
        .allocate_buffer(&context, input_desc)
        .expect("input buffer")
        .into_parts();
    let (output, _) = backend
        .allocate_buffer(&context, output_desc)
        .expect("output buffer")
        .into_parts();
    let input_bytes: Vec<u8> = (0..elements).map(|index| index as u8).collect();
    backend
        .write_buffer(&mut input, 0, &Slice(&input_bytes))
        .expect("initialize FP8 input");

    let direct_before = backend.direct_binding_admissions();
    let transfer_before = backend.explicit_transfer_bytes();
    let submit_once = || {
        let bindings = [
            BindingRef {
                slot: 0,
                buffer: &input,
                range: BufferRange::new(0, input_len as u64).expect("input range"),
                access: AccessMode::Read,
            },
            BindingRef {
                slot: 1,
                buffer: &output,
                range: BufferRange::new(0, output_len as u64).expect("output range"),
                access: AccessMode::Write,
            },
        ];
        let started = Instant::now();
        let event = backend
            .submit(&queue, &program, &bindings, Timeout::Infinite)
            .expect("warm FP8 submission");
        let admission = started.elapsed();
        let state = poll_to_terminal(backend, &event, Duration::from_secs(30))
            .expect("FP8 benchmark completion");
        assert!(matches!(state, EventState::Complete), "got {state:?}");
        let completion = started.elapsed();
        backend.destroy_event(event).expect("destroy event");
        (admission, completion)
    };

    for _ in 0..warmups {
        submit_once();
    }
    let (mut admission, mut completion): (Vec<_>, Vec<_>) =
        (0..samples).map(|_| submit_once()).unzip();
    admission.sort_unstable();
    completion.sort_unstable();

    let measured_submissions = (warmups + samples) as u64;
    let direct_bindings = backend.direct_binding_admissions() - direct_before;
    let explicit_transfer_bytes = backend.explicit_transfer_bytes() - transfer_before;
    assert_eq!(direct_bindings, measured_submissions * 2);
    assert_eq!(explicit_transfer_bytes, 0);

    let mut result = vec![0u8; output_len];
    backend
        .read_buffer(&output, 0, &mut SliceMut(&mut result))
        .expect("read FP8 benchmark output");
    for (index, bytes) in result.chunks_exact(2).enumerate() {
        assert_eq!(
            u16::from_le_bytes([bytes[0], bytes[1]]),
            fp8e4m3_to_bf16_bits(index as u8),
            "FP8 benchmark oracle mismatch at element {index}"
        );
    }

    let p50 = completion[completion.len() / 2];
    let transferred_bytes = (input_len + output_len) as f64;
    let gib_per_second = transferred_bytes / p50.as_secs_f64() / (1024.0 * 1024.0 * 1024.0);
    eprintln!(
        "XDNA FP8 E4M3 -> BF16: elements={elements}; worker=1; warmups={warmups}; samples={samples}; admission p50={:?} p95={:?}; submit-to-complete p50={:?} p95={:?}; effective IO={gib_per_second:.3} GiB/s; direct_bindings={direct_bindings}; explicit_transfer_bytes={explicit_transfer_bytes}",
        admission[admission.len() / 2],
        admission[admission.len() * 95 / 100],
        p50,
        completion[completion.len() * 95 / 100],
    );

    backend.free_buffer(input).expect("free input");
    backend.free_buffer(output).expect("free output");
    backend.unload_program(program).expect("unload program");
    backend.destroy_queue(queue).expect("destroy queue");
    backend.destroy_context(context).expect("destroy context");
}

#[test]
fn tosa_bf16_max_pool2d_compiles_to_a_wellformed_artifact() {
    if !toolchain_present() {
        assert!(
            !hardware_required(),
            "{REQUIRE_HARDWARE_ENV}=1 but VIRTIO_ACCEL_AMDXDNA_TOOLCHAIN is not configured"
        );
        eprintln!("no XDNA toolchain configured; skipping compiler test");
        return;
    }
    let container = compile_artifact(MAX_POOL2D_BF16.artifact, XDNA_TOSA_TARGET)
        .expect("compile TOSA max pool2d");
    let parsed =
        virtio_accel_xdna::PrecompiledArtifact::parse(&container).expect("valid container");
    assert!(parsed.xclbin.starts_with(b"xclbin2"));
    assert!(!parsed.insts.is_empty() && parsed.insts.len() % 4 == 0);
    assert_eq!((parsed.inputs, parsed.outputs), (1, 1));
    assert_eq!(parsed.slot_bytes, [4 * 4 * 2 * 2, 2 * 2 * 2 * 2]);
    assert_eq!(parsed.entry, "MLIR_AIE");
}

#[test]
fn tosa_bf16_max_pool2d_matches_the_shared_oracle_on_the_npu() {
    let Some(backend) = backend() else { return };
    if !toolchain_present() {
        assert!(
            !hardware_required(),
            "{REQUIRE_HARDWARE_ENV}=1 but VIRTIO_ACCEL_AMDXDNA_TOOLCHAIN is not configured"
        );
        eprintln!("no XDNA toolchain configured; skipping TOSA execution test");
        return;
    }

    let context = backend
        .create_context(ContextDesc::default())
        .expect("context");
    let queue = backend
        .create_queue(&context, QueueDesc::default())
        .expect("queue");
    let program = backend
        .load_program(
            &context,
            ArtifactRef {
                format: ARTIFACT_FORMAT,
                target: XDNA_TOSA_TARGET.to_identity(),
                payload: &Slice(MAX_POOL2D_BF16.artifact),
                resident_bytes: u64::MAX,
            },
        )
        .expect("load + compile TOSA max pool2d");

    let mut input_bits = MAX_POOL2D_BF16.inputs[0].bits.to_vec();
    let input_bytes = |bits: &[u16]| {
        bits.iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>()
    };
    let input_len = input_bits.len() * 2;
    let output_len = MAX_POOL2D_BF16.outputs[0].bits.len() * 2;
    let (mut input, _) = backend
        .allocate_buffer(
            &context,
            BufferDesc::new(
                input_len as u64,
                4096,
                MemoryDomain::Shared,
                BufferUsage::TRANSFER_DESTINATION | BufferUsage::PROGRAM_INPUT,
            )
            .unwrap(),
        )
        .expect("input buffer")
        .into_parts();
    let (output, _) = backend
        .allocate_buffer(
            &context,
            BufferDesc::new(
                output_len as u64,
                4096,
                MemoryDomain::Shared,
                BufferUsage::TRANSFER_SOURCE | BufferUsage::PROGRAM_OUTPUT,
            )
            .unwrap(),
        )
        .expect("output buffer")
        .into_parts();

    let bytes = input_bytes(&input_bits);
    backend
        .write_buffer(&mut input, 0, &Slice(&bytes))
        .expect("write corpus input");
    let event = backend
        .submit(
            &queue,
            &program,
            &[
                BindingRef {
                    slot: 0,
                    buffer: &input,
                    range: BufferRange::new(0, input_len as u64).unwrap(),
                    access: AccessMode::Read,
                },
                BindingRef {
                    slot: 1,
                    buffer: &output,
                    range: BufferRange::new(0, output_len as u64).unwrap(),
                    access: AccessMode::Write,
                },
            ],
            Timeout::Infinite,
        )
        .expect("submit max pool2d");
    let state = poll_to_terminal(&backend, &event, Duration::from_secs(30))
        .expect("max pool2d did not complete");
    assert!(matches!(state, EventState::Complete), "got {state:?}");
    let mut result = vec![0u8; output_len];
    backend
        .read_buffer(&output, 0, &mut SliceMut(&mut result))
        .expect("read corpus output");
    let result_bits: Vec<_> = result
        .chunks_exact(2)
        .map(|bytes| u16::from_le_bytes(bytes.try_into().expect("two-byte chunk")))
        .collect();
    assert!(MAX_POOL2D_BF16.output_matches(0, &result_bits));
    backend.destroy_event(event).expect("destroy event");

    // The advertised PROPAGATING_NAN constraint is executable behavior, not metadata only.
    input_bits[0] = 0x7fc1;
    let bytes = input_bytes(&input_bits);
    backend
        .write_buffer(&mut input, 0, &Slice(&bytes))
        .expect("write NaN input");
    let event = backend
        .submit(
            &queue,
            &program,
            &[
                BindingRef {
                    slot: 0,
                    buffer: &input,
                    range: BufferRange::new(0, input_len as u64).unwrap(),
                    access: AccessMode::Read,
                },
                BindingRef {
                    slot: 1,
                    buffer: &output,
                    range: BufferRange::new(0, output_len as u64).unwrap(),
                    access: AccessMode::Write,
                },
            ],
            Timeout::Infinite,
        )
        .expect("submit NaN max pool2d");
    let state = poll_to_terminal(&backend, &event, Duration::from_secs(30))
        .expect("NaN max pool2d did not complete");
    assert!(matches!(state, EventState::Complete), "got {state:?}");
    backend
        .read_buffer(&output, 0, &mut SliceMut(&mut result))
        .expect("read NaN output");
    let first = u16::from_le_bytes(result[0..2].try_into().expect("first bf16"));
    assert_eq!(first & 0x7f80, 0x7f80, "expected a BF16 NaN: {first:#06x}");
    assert_ne!(first & 0x007f, 0, "expected a BF16 NaN: {first:#06x}");
    backend.destroy_event(event).expect("destroy event");

    assert_eq!(backend.direct_binding_admissions(), 4);
    backend.free_buffer(input).expect("free input");
    backend.free_buffer(output).expect("free output");
    backend.unload_program(program).expect("unload");
    backend.destroy_queue(queue).expect("destroy queue");
    backend.destroy_context(context).expect("destroy context");
}

#[test]
fn tosa_bf16_matmul_compiles_to_a_wellformed_artifact() {
    // Hardware-free: needs the compiler toolchain but never initializes a device.
    if !toolchain_present() {
        assert!(
            !hardware_required(),
            "{REQUIRE_HARDWARE_ENV}=1 but VIRTIO_ACCEL_AMDXDNA_TOOLCHAIN is not configured"
        );
        eprintln!("no XDNA toolchain configured; skipping compiler test");
        return;
    }
    let tosa = bf16_matmul_tosa(32, 64, 32);
    let container = compile_artifact(&tosa, XDNA_TOSA_TARGET).expect("compile TOSA matmul");
    let parsed =
        virtio_accel_xdna::PrecompiledArtifact::parse(&container).expect("valid container");
    assert!(
        parsed.xclbin.starts_with(b"xclbin2"),
        "expected xclbin2 magic"
    );
    assert!(!parsed.insts.is_empty() && parsed.insts.len() % 4 == 0);
    // A/B are runtime inputs; C is the output. The zero-points are compile-time constants.
    assert_eq!((parsed.inputs, parsed.outputs), (2, 1));
    assert_eq!(parsed.entry, "MLIR_AIE");

    // A shape off the tested tiling is rejected before any compile runs.
    let untiled = bf16_matmul_tosa(48, 64, 32);
    assert!(matches!(
        compile_artifact(&untiled, XDNA_TOSA_TARGET),
        Err(BackendError::Unsupported)
    ));
}

/// The fused FP8 MATMUL must be bit-identical to running the two admitted tiers back to back.
///
/// This is the tier's whole correctness claim: fusing changes *where* the graph's explicit BF16
/// promotion happens (core-local scratch instead of a DDR round trip), never the arithmetic. Both
/// paths run on the NPU here and their FP32 results are compared byte for byte, so the comparison
/// cannot drift with a host-side oracle.
#[test]
fn fused_fp8_matmul_is_bit_identical_to_cast_then_matmul() {
    let Some(backend) = backend() else { return };
    if !toolchain_present() {
        assert!(
            !hardware_required(),
            "{REQUIRE_HARDWARE_ENV}=1 but VIRTIO_ACCEL_AMDXDNA_TOOLCHAIN is not configured"
        );
        eprintln!("no XDNA toolchain configured; skipping TOSA execution test");
        return;
    }
    // Multi-tile in every dimension, and both operand element counts are multiples of the CAST
    // tier's 1,024-element line so the unfused reference path is admissible too.
    const M: usize = 64;
    const K: usize = 128;
    const N: usize = 96;

    // Finite FP8E4M3 encodings only (0x7f/0xff are NaN), including the subnormals at 0x00..=0x07.
    let lhs_fp8: Vec<u8> = (0..M * K).map(|i| (i % 120) as u8).collect();
    let rhs_fp8: Vec<u8> = (0..K * N).map(|i| ((i * 7 + 3) % 120) as u8).collect();

    let context = backend
        .create_context(ContextDesc::default())
        .expect("context");
    let queue = backend
        .create_queue(&context, QueueDesc::default())
        .expect("queue");

    let load = |tosa: &[u8], target: virtio_accel_tosa::Target| {
        backend
            .load_program(
                &context,
                ArtifactRef {
                    format: ARTIFACT_FORMAT,
                    target: target.to_identity(),
                    payload: &Slice(tosa),
                    resident_bytes: u64::MAX,
                },
            )
            .expect("load + compile TOSA program")
    };
    let in_desc = |bytes: usize| {
        BufferDesc::new(
            bytes as u64,
            4096,
            MemoryDomain::Shared,
            BufferUsage::TRANSFER_DESTINATION | BufferUsage::PROGRAM_INPUT,
        )
        .unwrap()
    };
    let out_desc = |bytes: usize| {
        BufferDesc::new(
            bytes as u64,
            4096,
            MemoryDomain::Shared,
            BufferUsage::TRANSFER_SOURCE | BufferUsage::PROGRAM_OUTPUT,
        )
        .unwrap()
    };

    // One binary program run: write every input, submit, poll, read the output back.
    let run = |program: &_, inputs: &[&[u8]], output_len: usize| -> Vec<u8> {
        let mut in_buffers = Vec::new();
        for bytes in inputs {
            let (mut buffer, _) = backend
                .allocate_buffer(&context, in_desc(bytes.len()))
                .expect("input buffer")
                .into_parts();
            backend
                .write_buffer(&mut buffer, 0, &Slice(bytes))
                .expect("write input");
            in_buffers.push(buffer);
        }
        let (out_buffer, _) = backend
            .allocate_buffer(&context, out_desc(output_len))
            .expect("output buffer")
            .into_parts();

        let mut bindings: Vec<BindingRef<'_, _>> = in_buffers
            .iter()
            .enumerate()
            .map(|(slot, buffer)| BindingRef {
                slot: slot as u32,
                buffer,
                range: BufferRange::new(0, inputs[slot].len() as u64).unwrap(),
                access: AccessMode::Read,
            })
            .collect();
        bindings.push(BindingRef {
            slot: in_buffers.len() as u32,
            buffer: &out_buffer,
            range: BufferRange::new(0, output_len as u64).unwrap(),
            access: AccessMode::Write,
        });

        let event = backend
            .submit(&queue, program, &bindings, Timeout::Infinite)
            .expect("submit");
        let state = poll_to_terminal(&backend, &event, Duration::from_secs(30))
            .expect("program did not complete");
        assert!(matches!(state, EventState::Complete), "got {state:?}");

        let mut result = vec![0u8; output_len];
        backend
            .read_buffer(&out_buffer, 0, &mut SliceMut(&mut result))
            .expect("read output");
        backend.destroy_event(event).expect("destroy event");
        for buffer in in_buffers {
            backend.free_buffer(buffer).expect("free input");
        }
        backend.free_buffer(out_buffer).expect("free output");
        result
    };

    // Fused: FP8 in, FP32 out, no BF16 tensor anywhere.
    let fused_program = load(
        &fp8e4m3_matmul_tosa(M as i32, K as i32, N as i32),
        XDNA_TOSA_FP8_TARGET,
    );
    let fused = run(&fused_program, &[&lhs_fp8, &rhs_fp8], M * N * 4);

    // Reference: promote each operand with the standalone CAST tier, then multiply in BF16.
    let lhs_cast = load(&fp8e4m3_to_bf16_tosa(M * K), XDNA_TOSA_FP8_TARGET);
    let lhs_bf16 = run(&lhs_cast, &[&lhs_fp8], M * K * 2);
    let rhs_cast = load(&fp8e4m3_to_bf16_tosa(K * N), XDNA_TOSA_FP8_TARGET);
    let rhs_bf16 = run(&rhs_cast, &[&rhs_fp8], K * N * 2);
    let matmul = load(
        &bf16_matmul_tosa(M as i32, K as i32, N as i32),
        XDNA_TOSA_TARGET,
    );
    let unfused = run(&matmul, &[&lhs_bf16, &rhs_bf16], M * N * 4);

    assert_eq!(
        fused, unfused,
        "fusing the promotion must not change a single result bit"
    );

    for program in [fused_program, lhs_cast, rhs_cast, matmul] {
        backend.unload_program(program).expect("unload");
    }
    backend.destroy_queue(queue).expect("destroy queue");
    backend.destroy_context(context).expect("destroy context");
}

#[test]
fn tosa_bf16_matmul_runs_on_the_npu() {
    let Some(backend) = backend() else { return };
    if !toolchain_present() {
        assert!(
            !hardware_required(),
            "{REQUIRE_HARDWARE_ENV}=1 but VIRTIO_ACCEL_AMDXDNA_TOOLCHAIN is not configured"
        );
        eprintln!("no XDNA toolchain configured; skipping TOSA execution test");
        return;
    }
    // Non-square, multi-tile in every dimension (M/32=2, K/64=2, N/32=3).
    const M: usize = 64;
    const K: usize = 128;
    const N: usize = 96;

    let context = backend
        .create_context(ContextDesc::default())
        .expect("context");
    let queue = backend
        .create_queue(&context, QueueDesc::default())
        .expect("queue");
    let tosa = bf16_matmul_tosa(M as i32, K as i32, N as i32);
    let program = backend
        .load_program(
            &context,
            ArtifactRef {
                format: ARTIFACT_FORMAT,
                target: XDNA_TOSA_TARGET.to_identity(),
                payload: &Slice(&tosa),
                resident_bytes: u64::MAX,
            },
        )
        .expect("load + compile TOSA matmul");

    // Small-integer inputs: exact in BF16, and every FP32 partial sum is exact, so the result is
    // bit-exact regardless of the kernel's tiling/summation order (bit-exact by construction).
    let a: Vec<f32> = (0..M * K).map(|i| (i % 7) as f32).collect();
    let b: Vec<f32> = (0..K * N).map(|i| (i % 5) as f32).collect();
    let a_bytes: Vec<u8> = a.iter().flat_map(|&x| bf16_le(x)).collect();
    let b_bytes: Vec<u8> = b.iter().flat_map(|&x| bf16_le(x)).collect();

    let in_desc = |bytes: usize| {
        BufferDesc::new(
            bytes as u64,
            4096,
            MemoryDomain::Shared,
            BufferUsage::TRANSFER_DESTINATION | BufferUsage::PROGRAM_INPUT,
        )
        .unwrap()
    };
    let (mut lhs, _) = backend
        .allocate_buffer(&context, in_desc(a_bytes.len()))
        .expect("lhs buffer")
        .into_parts();
    let (mut rhs, _) = backend
        .allocate_buffer(&context, in_desc(b_bytes.len()))
        .expect("rhs buffer")
        .into_parts();
    let (output, _) = backend
        .allocate_buffer(
            &context,
            BufferDesc::new(
                (M * N * 4) as u64,
                4096,
                MemoryDomain::Shared,
                BufferUsage::TRANSFER_SOURCE | BufferUsage::PROGRAM_OUTPUT,
            )
            .unwrap(),
        )
        .expect("output buffer")
        .into_parts();

    backend
        .write_buffer(&mut lhs, 0, &Slice(&a_bytes))
        .expect("write lhs");
    backend
        .write_buffer(&mut rhs, 0, &Slice(&b_bytes))
        .expect("write rhs");

    let bindings = [
        BindingRef {
            slot: 0,
            buffer: &lhs,
            range: BufferRange::new(0, a_bytes.len() as u64).unwrap(),
            access: AccessMode::Read,
        },
        BindingRef {
            slot: 1,
            buffer: &rhs,
            range: BufferRange::new(0, b_bytes.len() as u64).unwrap(),
            access: AccessMode::Read,
        },
        BindingRef {
            slot: 2,
            buffer: &output,
            range: BufferRange::new(0, (M * N * 4) as u64).unwrap(),
            access: AccessMode::Write,
        },
    ];
    let event = backend
        .submit(&queue, &program, &bindings, Timeout::Infinite)
        .expect("submit");
    let state = poll_to_terminal(&backend, &event, Duration::from_secs(30))
        .expect("matmul did not complete");
    assert!(matches!(state, EventState::Complete), "got {state:?}");

    let mut result = vec![0u8; M * N * 4];
    backend
        .read_buffer(&output, 0, &mut SliceMut(&mut result))
        .expect("read output");

    // Exact integer oracle: C[i,j] = sum_k A[i,k] * B[k,j].
    for i in 0..M {
        for j in 0..N {
            let mut expected = 0.0f32;
            for kk in 0..K {
                expected += a[i * K + kk] * b[kk * N + j];
            }
            let got = f32_le(&result[(i * N + j) * 4..(i * N + j) * 4 + 4]);
            assert_eq!(
                got.to_bits(),
                expected.to_bits(),
                "matmul C[{i},{j}] mismatch: got {got}, want {expected}"
            );
        }
    }

    // The submission bound all three buffers directly, with no submission-time staging copy.
    assert_eq!(backend.direct_binding_admissions(), bindings.len() as u64);

    backend.destroy_event(event).expect("destroy event");
    backend.free_buffer(lhs).expect("free lhs");
    backend.free_buffer(rhs).expect("free rhs");
    backend.free_buffer(output).expect("free output");
    backend.unload_program(program).expect("unload");
    backend.destroy_queue(queue).expect("destroy queue");
    backend.destroy_context(context).expect("destroy context");
}

#[test]
fn tosa_matmul_with_a_shared_input_buffer_runs_on_the_npu() {
    // X·X: one caller buffer bound to both read slots. Read-read aliasing is admitted (the kernel
    // only loads from the buffer; the OpenVINO backend admits the same), and the result must still
    // be bit-exact.
    let Some(backend) = backend() else { return };
    if !toolchain_present() {
        assert!(
            !hardware_required(),
            "{REQUIRE_HARDWARE_ENV}=1 but VIRTIO_ACCEL_AMDXDNA_TOOLCHAIN is not configured"
        );
        eprintln!("no XDNA toolchain configured; skipping TOSA execution test");
        return;
    }
    const DIM: usize = 64;
    const IN_BYTES: usize = DIM * DIM * 2;
    const OUT_BYTES: usize = DIM * DIM * 4;

    let context = backend
        .create_context(ContextDesc::default())
        .expect("context");
    let queue = backend
        .create_queue(&context, QueueDesc::default())
        .expect("queue");
    let tosa = bf16_matmul_tosa(DIM as i32, DIM as i32, DIM as i32);
    let program = backend
        .load_program(
            &context,
            ArtifactRef {
                format: ARTIFACT_FORMAT,
                target: XDNA_TOSA_TARGET.to_identity(),
                payload: &Slice(&tosa),
                resident_bytes: u64::MAX,
            },
        )
        .expect("load + compile square matmul");

    let x: Vec<f32> = (0..DIM * DIM).map(|i| (i % 5) as f32).collect();
    let x_bytes: Vec<u8> = x.iter().flat_map(|&value| bf16_le(value)).collect();
    let (mut shared, _) = backend
        .allocate_buffer(
            &context,
            BufferDesc::new(
                IN_BYTES as u64,
                4096,
                MemoryDomain::Shared,
                BufferUsage::TRANSFER_DESTINATION | BufferUsage::PROGRAM_INPUT,
            )
            .unwrap(),
        )
        .expect("shared input buffer")
        .into_parts();
    let (output, _) = backend
        .allocate_buffer(
            &context,
            BufferDesc::new(
                OUT_BYTES as u64,
                4096,
                MemoryDomain::Shared,
                BufferUsage::TRANSFER_SOURCE | BufferUsage::PROGRAM_OUTPUT,
            )
            .unwrap(),
        )
        .expect("output buffer")
        .into_parts();
    backend
        .write_buffer(&mut shared, 0, &Slice(&x_bytes))
        .expect("write shared input");

    let in_range = BufferRange::new(0, IN_BYTES as u64).unwrap();
    let bindings = [
        BindingRef {
            slot: 0,
            buffer: &shared,
            range: in_range,
            access: AccessMode::Read,
        },
        BindingRef {
            slot: 1,
            buffer: &shared,
            range: in_range,
            access: AccessMode::Read,
        },
        BindingRef {
            slot: 2,
            buffer: &output,
            range: BufferRange::new(0, OUT_BYTES as u64).unwrap(),
            access: AccessMode::Write,
        },
    ];
    let event = backend
        .submit(&queue, &program, &bindings, Timeout::Infinite)
        .expect("submit with shared input");
    let state = poll_to_terminal(&backend, &event, Duration::from_secs(30))
        .expect("shared-input matmul stalled");
    assert!(matches!(state, EventState::Complete), "got {state:?}");

    let mut result = vec![0u8; OUT_BYTES];
    backend
        .read_buffer(&output, 0, &mut SliceMut(&mut result))
        .expect("read output");
    for i in 0..DIM {
        for j in 0..DIM {
            let mut expected = 0.0f32;
            for kk in 0..DIM {
                expected += x[i * DIM + kk] * x[kk * DIM + j];
            }
            let got = f32_le(&result[(i * DIM + j) * 4..(i * DIM + j) * 4 + 4]);
            assert_eq!(got.to_bits(), expected.to_bits(), "X·X C[{i},{j}] mismatch");
        }
    }

    backend.destroy_event(event).expect("destroy event");
    backend.free_buffer(shared).expect("free shared");
    backend.free_buffer(output).expect("free output");
    backend.unload_program(program).expect("unload");
    backend.destroy_queue(queue).expect("destroy queue");
    backend.destroy_context(context).expect("destroy context");
}

#[test]
fn submit_enforces_the_per_slot_binding_plan_and_load_enforces_residency() {
    let Some(backend) = backend() else { return };
    let context = backend
        .create_context(ContextDesc::default())
        .expect("context");
    let queue = backend
        .create_queue(&context, QueueDesc::default())
        .expect("queue");

    // A finite residency charge cannot be honored (HRX publishes no bound); reject at load.
    assert!(matches!(
        backend.load_program(
            &context,
            ArtifactRef {
                format: XDNA_PRECOMPILED_FORMAT,
                target: TargetIdentity([0; 12]),
                payload: &Slice(PASSTHROUGH),
                resident_bytes: 4096,
            },
        ),
        Err(BackendError::ResourceLimit)
    ));

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

    // Oversized buffers so a short range stays in bounds: the rejection must come from the
    // program's per-slot byte plan (Incompatible), not the buffer bounds (OutOfBounds).
    let desc = |usage| {
        BufferDesc::new(
            2 * PASSTHROUGH_BYTES as u64,
            4096,
            MemoryDomain::Shared,
            usage,
        )
        .unwrap()
    };
    let (input, _) = backend
        .allocate_buffer(
            &context,
            desc(BufferUsage::TRANSFER_DESTINATION | BufferUsage::PROGRAM_INPUT),
        )
        .expect("input")
        .into_parts();
    let (unused, _) = backend
        .allocate_buffer(
            &context,
            desc(BufferUsage::TRANSFER_DESTINATION | BufferUsage::PROGRAM_INPUT),
        )
        .expect("unused")
        .into_parts();
    let (output, _) = backend
        .allocate_buffer(
            &context,
            desc(BufferUsage::TRANSFER_SOURCE | BufferUsage::PROGRAM_OUTPUT),
        )
        .expect("output")
        .into_parts();

    let full = BufferRange::new(0, PASSTHROUGH_BYTES as u64).unwrap();
    let short = BufferRange::new(0, 64).unwrap();
    let binding = |slot, buffer, range, access| BindingRef {
        slot,
        buffer,
        range,
        access,
    };
    // A short input range: in bounds, but not the tensor size the TXN stream transfers.
    let wrong_length = [
        binding(0, &input, short, AccessMode::Read),
        binding(1, &unused, full, AccessMode::Read),
        binding(2, &output, full, AccessMode::Write),
    ];
    assert!(matches!(
        backend.submit(&queue, &program, &wrong_length, Timeout::Infinite),
        Err(SubmitFailure::Rejected(BackendError::Incompatible))
    ));

    // Exactly the tensor length and in bounds (the buffers are twice the tensor size), but starting
    // mid-word. An AIE DMA descriptor cannot express a sub-word start, so this is rejected before
    // the device sees it rather than handed to the driver as a malformed transfer.
    let misaligned = BufferRange::new(2, PASSTHROUGH_BYTES as u64).unwrap();
    let unaligned_start = [
        binding(0, &input, misaligned, AccessMode::Read),
        binding(1, &unused, full, AccessMode::Read),
        binding(2, &output, full, AccessMode::Write),
    ];
    assert!(matches!(
        backend.submit(&queue, &program, &unaligned_start, Timeout::Infinite),
        Err(SubmitFailure::Rejected(BackendError::Incompatible))
    ));

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
