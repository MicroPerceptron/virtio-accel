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
    SubmitFailure, TargetIdentity, Timeout,
};
use virtio_accel_tosa::{ARTIFACT_FORMAT, DType};
use virtio_accel_tosa_build::{OperatorKind, OwnedGraph, OwnedOperator, OwnedTensor};
use virtio_accel_xdna::{
    InitError, XDNA_PRECOMPILED_FORMAT, XDNA_TOSA_TARGET, XdnaAccelerator, compile_artifact,
};

/// Whether the pinned compiler toolchain is configured (the compiler tests need it, not a device).
fn toolchain_present() -> bool {
    std::env::var_os("VIRTIO_ACCEL_AMDXDNA_TOOLCHAIN").is_some()
}

/// Author a BF16 IDENTITY TOSA artifact of `elements` for the advertised target.
fn bf16_identity_tosa(elements: usize) -> Vec<u8> {
    let shape = vec![1, 1, elements as i32];
    let mut graph = OwnedGraph::new("main");
    graph.push_tensor(OwnedTensor::new("x", shape.clone(), DType::BF16));
    graph.push_tensor(OwnedTensor::new("y", shape, DType::BF16));
    graph.push_operator(OwnedOperator::new(
        OperatorKind::Identity,
        vec!["x".into()],
        vec!["y".into()],
    ));
    graph.push_input("x");
    graph.push_output("y");
    graph.build(XDNA_TOSA_TARGET).expect("build bf16 identity")
}

/// Author a batch-1 BF16 → FP32 MATMUL `C[1,M,N] = A[1,M,K] · B[1,K,N]` for the advertised target.
/// The two zero-points are constant-zero (TOSA requires floating-point zero-points to be zero).
fn bf16_matmul_tosa(m: i32, k: i32, n: i32) -> Vec<u8> {
    let mut graph = OwnedGraph::new("main");
    graph
        .push_tensor(OwnedTensor::new("lhs", vec![1, m, k], DType::BF16))
        .push_tensor(OwnedTensor::new("rhs", vec![1, k, n], DType::BF16))
        .push_tensor(OwnedTensor::constant(
            "lhs_zp",
            vec![1],
            DType::BF16,
            vec![0u8; 2],
        ))
        .push_tensor(OwnedTensor::constant(
            "rhs_zp",
            vec![1],
            DType::BF16,
            vec![0u8; 2],
        ))
        .push_tensor(OwnedTensor::new("output", vec![1, m, n], DType::FP32))
        .push_operator(OwnedOperator::new(
            OperatorKind::Const,
            vec![],
            vec!["lhs_zp".into()],
        ))
        .push_operator(OwnedOperator::new(
            OperatorKind::Const,
            vec![],
            vec!["rhs_zp".into()],
        ))
        .push_operator(OwnedOperator::new(
            OperatorKind::MatMul,
            vec!["lhs".into(), "rhs".into(), "lhs_zp".into(), "rhs_zp".into()],
            vec!["output".into()],
        ))
        .push_input("lhs")
        .push_input("rhs")
        .push_output("output");
    graph.build(XDNA_TOSA_TARGET).expect("build bf16 matmul")
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
fn tosa_bf16_identity_compiles_to_a_wellformed_artifact() {
    // Hardware-free: needs the compiler toolchain but never initializes a device.
    if !toolchain_present() {
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
    let deadline = Instant::now() + Duration::from_secs(10);
    let state = loop {
        match backend.poll_event(&event).expect("poll") {
            EventState::Pending => {
                assert!(Instant::now() < deadline, "identity did not complete");
                std::thread::yield_now();
            }
            terminal => break terminal,
        }
    };
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
fn tosa_bf16_matmul_compiles_to_a_wellformed_artifact() {
    // Hardware-free: needs the compiler toolchain but never initializes a device.
    if !toolchain_present() {
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

#[test]
fn tosa_bf16_matmul_runs_on_the_npu() {
    let Some(backend) = backend() else { return };
    if !toolchain_present() {
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
    let deadline = Instant::now() + Duration::from_secs(30);
    let state = loop {
        match backend.poll_event(&event).expect("poll") {
            EventState::Pending => {
                assert!(Instant::now() < deadline, "matmul did not complete");
                std::thread::yield_now();
            }
            terminal => break terminal,
        }
    };
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
    let deadline = Instant::now() + Duration::from_secs(30);
    let state = loop {
        match backend.poll_event(&event).expect("poll") {
            EventState::Pending => {
                assert!(Instant::now() < deadline, "shared-input matmul stalled");
                std::thread::yield_now();
            }
            terminal => break terminal,
        }
    };
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
