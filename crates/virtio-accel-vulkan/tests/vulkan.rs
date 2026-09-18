//! Native integration and acceptance suite for the Vulkan backend.
//!
//! Runs against every suitable device the loader enumerates (a real GPU and a software ICD such as
//! lavapipe both count). Without a loader or device the tests skip, unless
//! `VIRTIO_ACCEL_VULKAN_REQUIRE_DEVICE=1` turns absence into a failure (the CI lane sets it so a
//! silently missing ICD cannot pass as green). Placeholder builds do not compile this file.

#![cfg(va_vulkan)]

use std::time::{Duration, Instant};

use virtio_accel_conformance::numerics::{
    ADD_FP16, FP32_OPERATOR_CASE_GROUPS, Fp32TierTensor, HEXAGON_LOGICAL_CASES,
    HEXAGON_MOVEMENT_CASES, HEXAGON_REDUCTION_CASES, HEXAGON_UNARY_FP16_CASES, IDENTITY_EDGES_FP16,
    IDENTITY_EDGES_FP32, IDENTITY_INT8, LINEAR_TANH_FP32, MATMUL_FP16, MATMUL_FP32,
    MAX_POOL2D_FP16, MAXIMUM_FP16, MINIMUM_FP16, MOCK_LINEAR_CLASSIFIER_FP16, MUL_FP16, POW_FP16,
    SUB_FP16, TosaFloat16Case, TosaFp32OperatorCase, TosaRawCase,
};
use virtio_accel_conformance::{
    BindingFixture, ConformanceHooks, ProgramFixture, ResourceCounts, SubmissionPathDiagnostics,
    TargetDescription, run,
};
use virtio_accel_core::{
    Accelerator, AccessMode, ArtifactRef, BackendError, BindingRef, BufferDesc, BufferRange,
    BufferUsage, ByteSink, ByteSource, Capabilities, ContextDesc, EventState, MemoryDomain,
    QueueDesc, ReleaseFailure, SubmitFailure, Timeout,
};
use virtio_accel_tosa::{
    DType, NanPropagationMode, Target, TosaCapabilityProvider, ValueRoles, parse,
};
use virtio_accel_tosa_build::{OperatorKind, OwnedGraph, OwnedOperator, OwnedTensor};
use virtio_accel_vulkan::{
    InitError, REQUIRED_RESIDENT_BYTES, VULKAN_TOSA_FP8_TARGET, VULKAN_TOSA_INTEGER_TARGET,
    VULKAN_TOSA_TARGET, VulkanAccelerator, VulkanEvent,
};

const IDENTITY_FP32_LOCAL: &[u8] = include_bytes!("data/identity-fp32-v1.0.0.tosa");

/// Page alignment: what the conformance fixtures request and what every Mesa allocation honors.
const BUFFER_ALIGNMENT: u64 = 4096;

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

/// A source that hides its contiguity, forcing the segmented `read_at` path.
#[derive(Debug)]
struct SegmentedSource<'a>(&'a [u8]);

impl ByteSource for SegmentedSource<'_> {
    fn len(&self) -> u64 {
        self.0.len() as u64
    }

    fn read_at(&self, offset: u64, target: &mut [u8]) -> Result<(), BackendError> {
        ByteSource::read_at(self.0, offset, target)
    }
}

#[derive(Debug)]
struct VecSink(Vec<u8>);

impl ByteSink for VecSink {
    fn len(&self) -> u64 {
        self.0.len() as u64
    }

    fn write_at(&mut self, offset: u64, source: &[u8]) -> Result<(), BackendError> {
        ByteSink::write_at(self.0.as_mut_slice(), offset, source)
    }

    fn as_contiguous_mut(&mut self) -> Option<&mut [u8]> {
        Some(&mut self.0)
    }
}

/// A sink that hides its contiguity, forcing the segmented `write_at` path.
#[derive(Debug)]
struct SegmentedSink(Vec<u8>);

impl ByteSink for SegmentedSink {
    fn len(&self) -> u64 {
        self.0.len() as u64
    }

    fn write_at(&mut self, offset: u64, source: &[u8]) -> Result<(), BackendError> {
        ByteSink::write_at(self.0.as_mut_slice(), offset, source)
    }
}

fn device_required() -> bool {
    std::env::var("VIRTIO_ACCEL_VULKAN_REQUIRE_DEVICE").is_ok_and(|value| value == "1")
}

/// Every suitable device by enumerated name, or empty when the host has none.
fn devices() -> Vec<String> {
    match VulkanAccelerator::available_devices() {
        Ok(devices) if !devices.is_empty() => devices,
        Ok(_) | Err(InitError::RuntimeUnavailable | InitError::DeviceUnavailable) => {
            assert!(
                !device_required(),
                "VIRTIO_ACCEL_VULKAN_REQUIRE_DEVICE=1 but no Vulkan device was enumerated"
            );
            Vec::new()
        }
        Err(error) => panic!("device enumeration failed: {error}"),
    }
}

fn open(device: &str) -> VulkanAccelerator {
    VulkanAccelerator::with_device(device)
        .unwrap_or_else(|error| panic!("{device}: backend initialization failed: {error}"))
}

fn release<T>(result: Result<(), ReleaseFailure<T>>) {
    if let Err(failure) = result {
        panic!("release failed: {:?}", failure.error());
    }
}

fn wait_for_terminal(backend: &VulkanAccelerator, event: &VulkanEvent) -> EventState {
    let deadline = Instant::now() + Duration::from_secs(15);
    loop {
        match backend.poll_event(event).unwrap() {
            EventState::Pending => {
                assert!(Instant::now() < deadline, "submission never completed");
                std::thread::yield_now();
            }
            terminal => return terminal,
        }
    }
}

fn float_bytes(values: impl IntoIterator<Item = f32>) -> Vec<u8> {
    values
        .into_iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect()
}

fn floats(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|bytes| f32::from_ne_bytes(bytes.try_into().unwrap()))
        .collect()
}

fn load(
    backend: &VulkanAccelerator,
    context: &<VulkanAccelerator as Accelerator>::Context,
    artifact: &[u8],
    target: Target,
) -> Result<<VulkanAccelerator as Accelerator>::Program, BackendError> {
    let model = parse(artifact).unwrap();
    let artifact = model.artifact_ref(target, REQUIRED_RESIDENT_BYTES).unwrap();
    backend.load_program(context, artifact)
}

/// Full lifecycle in `domain`: allocate, write, execute the FP32 identity, read the output back.
fn run_identity(
    backend: &VulkanAccelerator,
    artifact: &[u8],
    input: &[u8],
    domain: MemoryDomain,
) -> Vec<u8> {
    let device = backend.device_name();
    let context = backend.create_context(ContextDesc::default()).unwrap();
    let program = load(backend, &context, artifact, VULKAN_TOSA_TARGET)
        .unwrap_or_else(|error| panic!("{device}: identity load failed: {error:?}"));
    let input_desc = BufferDesc::new(
        input.len() as u64,
        BUFFER_ALIGNMENT,
        domain,
        BufferUsage::TRANSFER_DESTINATION | BufferUsage::PROGRAM_INPUT,
    )
    .unwrap();
    let (mut input_buffer, info) = backend
        .allocate_buffer(&context, input_desc)
        .unwrap()
        .into_parts();
    backend
        .device_info()
        .unwrap()
        .validate_buffer_info(input_desc, info)
        .unwrap();
    backend
        .write_buffer(&mut input_buffer, 0, &SliceSource(input))
        .unwrap();
    let output_desc = BufferDesc::new(
        input.len() as u64,
        BUFFER_ALIGNMENT,
        domain,
        BufferUsage::TRANSFER_SOURCE | BufferUsage::PROGRAM_OUTPUT,
    )
    .unwrap();
    let (output, _) = backend
        .allocate_buffer(&context, output_desc)
        .unwrap()
        .into_parts();
    let queue = backend
        .create_queue(&context, QueueDesc::default())
        .unwrap();
    let bindings = [
        BindingRef {
            slot: 0,
            buffer: &input_buffer,
            range: BufferRange::new(0, input.len() as u64).unwrap(),
            access: AccessMode::Read,
        },
        BindingRef {
            slot: 1,
            buffer: &output,
            range: BufferRange::new(0, input.len() as u64).unwrap(),
            access: AccessMode::Write,
        },
    ];
    let event = backend
        .submit(&queue, &program, &bindings, Timeout::Infinite)
        .unwrap_or_else(|failure| match failure {
            SubmitFailure::Rejected(error) => panic!("{device}: submission rejected: {error:?}"),
            SubmitFailure::Indeterminate { error, .. } => {
                panic!("{device}: submission indeterminate: {error:?}")
            }
        });
    assert_eq!(
        wait_for_terminal(backend, &event),
        EventState::Complete,
        "{device}"
    );
    release(backend.destroy_event(event));

    let mut bytes = VecSink(vec![0; input.len()]);
    backend.read_buffer(&output, 0, &mut bytes).unwrap();
    release(backend.destroy_queue(queue));
    release(backend.unload_program(program));
    release(backend.free_buffer(output));
    release(backend.free_buffer(input_buffer));
    release(backend.destroy_context(context));
    assert_eq!(backend.live_resources(), Default::default(), "{device}");
    bytes.0
}

/// Full lifecycle for the shared FP32 MATMUL case: allocate two inputs plus an output in
/// `domain`, execute, and read the product back.
fn run_matmul(
    backend: &VulkanAccelerator,
    lhs: &[u8],
    rhs: &[u8],
    output_len: usize,
    domain: MemoryDomain,
) -> Vec<u8> {
    let device = backend.device_name();
    let context = backend.create_context(ContextDesc::default()).unwrap();
    let program = load(backend, &context, MATMUL_FP32.artifact, VULKAN_TOSA_TARGET)
        .unwrap_or_else(|error| panic!("{device}: matmul load failed: {error:?}"));
    let input_usage = BufferUsage::TRANSFER_DESTINATION | BufferUsage::PROGRAM_INPUT;
    let lhs_desc =
        BufferDesc::new(lhs.len() as u64, BUFFER_ALIGNMENT, domain, input_usage).unwrap();
    let (mut lhs_buffer, _) = backend
        .allocate_buffer(&context, lhs_desc)
        .unwrap()
        .into_parts();
    backend
        .write_buffer(&mut lhs_buffer, 0, &SliceSource(lhs))
        .unwrap();
    let rhs_desc =
        BufferDesc::new(rhs.len() as u64, BUFFER_ALIGNMENT, domain, input_usage).unwrap();
    let (mut rhs_buffer, _) = backend
        .allocate_buffer(&context, rhs_desc)
        .unwrap()
        .into_parts();
    backend
        .write_buffer(&mut rhs_buffer, 0, &SliceSource(rhs))
        .unwrap();
    let output_desc = BufferDesc::new(
        output_len as u64,
        BUFFER_ALIGNMENT,
        domain,
        BufferUsage::TRANSFER_SOURCE | BufferUsage::PROGRAM_OUTPUT,
    )
    .unwrap();
    let (output, _) = backend
        .allocate_buffer(&context, output_desc)
        .unwrap()
        .into_parts();
    let queue = backend
        .create_queue(&context, QueueDesc::default())
        .unwrap();
    let bindings = [
        BindingRef {
            slot: 0,
            buffer: &lhs_buffer,
            range: BufferRange::new(0, lhs.len() as u64).unwrap(),
            access: AccessMode::Read,
        },
        BindingRef {
            slot: 1,
            buffer: &rhs_buffer,
            range: BufferRange::new(0, rhs.len() as u64).unwrap(),
            access: AccessMode::Read,
        },
        BindingRef {
            slot: 2,
            buffer: &output,
            range: BufferRange::new(0, output_len as u64).unwrap(),
            access: AccessMode::Write,
        },
    ];
    let event = backend
        .submit(&queue, &program, &bindings, Timeout::Infinite)
        .unwrap_or_else(|failure| match failure {
            SubmitFailure::Rejected(error) => panic!("{device}: submission rejected: {error:?}"),
            SubmitFailure::Indeterminate { error, .. } => {
                panic!("{device}: submission indeterminate: {error:?}")
            }
        });
    assert_eq!(
        wait_for_terminal(backend, &event),
        EventState::Complete,
        "{device}"
    );
    release(backend.destroy_event(event));

    let mut bytes = VecSink(vec![0; output_len]);
    backend.read_buffer(&output, 0, &mut bytes).unwrap();
    release(backend.destroy_queue(queue));
    release(backend.unload_program(program));
    release(backend.free_buffer(output));
    release(backend.free_buffer(rhs_buffer));
    release(backend.free_buffer(lhs_buffer));
    release(backend.destroy_context(context));
    assert_eq!(backend.live_resources(), Default::default(), "{device}");
    bytes.0
}

fn advertised_domains(backend: &VulkanAccelerator) -> Vec<MemoryDomain> {
    let capabilities = backend.device_info().unwrap().capabilities;
    [
        MemoryDomain::Host,
        MemoryDomain::Device,
        MemoryDomain::Shared,
    ]
    .into_iter()
    .filter(|domain| capabilities.supports_memory_domain(*domain))
    .collect()
}

#[test]
fn executes_the_shared_fp32_matmul_in_every_advertised_domain() {
    let case = &MATMUL_FP32;
    let lhs = float_bytes(case.inputs[0].values.iter().copied());
    let rhs = float_bytes(case.inputs[1].values.iter().copied());
    let output_len = case.outputs[0].values.len() * 4;
    for device in devices() {
        let backend = open(&device);
        for domain in advertised_domains(&backend) {
            let bytes = run_matmul(&backend, &lhs, &rhs, output_len, domain);
            let actual = floats(&bytes);
            assert!(
                case.output_matches(0, &actual),
                "{device}: {domain:?}: {actual:?} does not match {:?}",
                case.outputs[0].values
            );
        }
    }
}

#[test]
fn reports_stable_valid_metadata_for_every_device() {
    for device in devices() {
        let backend = open(&device);
        let info = backend.device_info().unwrap();
        info.validate().unwrap();
        assert_eq!(info, backend.device_info().unwrap(), "{device}");
        assert_eq!(backend.device_name(), device);
        assert!(
            info.capabilities
                .contains(Capabilities::HOST_VISIBLE_MEMORY),
            "{device}: every Vulkan device has a host-coherent type"
        );
        assert!(
            !info.capabilities.contains(Capabilities::EVENT_CANCELLATION),
            "{device}: Vulkan has no cancel primitive (ADR 0006)"
        );
        assert!(!backend.is_poisoned());
        eprintln!("{device}: {info:?}");
    }
}

#[test]
fn executes_the_fp32_identity_in_every_advertised_domain() {
    for device in devices() {
        let backend = open(&device);
        for domain in advertised_domains(&backend) {
            let payload = float_bytes([42.5]);
            let output = run_identity(&backend, IDENTITY_FP32_LOCAL, &payload, domain);
            assert_eq!(output, payload, "{device}: {domain:?}");
        }
    }
}

#[test]
fn preserves_fp32_edge_values_bit_exactly_on_every_device() {
    for device in devices() {
        let backend = open(&device);
        let case = &IDENTITY_EDGES_FP32;
        let input = float_bytes(case.inputs[0].values.iter().copied());
        for domain in advertised_domains(&backend) {
            let output = run_identity(&backend, case.artifact, &input, domain);
            let actual = floats(&output);
            assert!(
                case.output_matches(0, &actual),
                "{device}: {domain:?}: {} produced {actual:?}",
                case.name
            );
            // The oracle tolerates NaN payload changes; the copy kernel must not even do that.
            assert_eq!(output, input, "{device}: {domain:?}: bit-exact copy");
        }
    }
}

#[test]
fn copies_aligned_offset_bindings_exactly() {
    // Exercise a nonzero descriptor offset that satisfies every device's advertised storage
    // buffer alignment. The guarded ranges must remain untouched.
    for device in devices() {
        let backend = open(&device);
        let case = &IDENTITY_EDGES_FP32;
        let context = backend.create_context(ContextDesc::default()).unwrap();
        let program = load(&backend, &context, case.artifact, VULKAN_TOSA_TARGET).unwrap();
        let bytes = 8 * 4;
        let offset = BUFFER_ALIGNMENT;
        let buffer_bytes = offset + bytes + offset;
        // Allocate a larger buffer and bind the tensor in the middle: bytes outside the bound
        // range must stay untouched.
        let desc = BufferDesc::new(
            buffer_bytes,
            BUFFER_ALIGNMENT,
            MemoryDomain::Host,
            BufferUsage::TRANSFER_SOURCE
                | BufferUsage::TRANSFER_DESTINATION
                | BufferUsage::PROGRAM_INPUT
                | BufferUsage::PROGRAM_OUTPUT,
        )
        .unwrap();
        let (mut input, _) = backend
            .allocate_buffer(&context, desc)
            .unwrap()
            .into_parts();
        let (mut output, _) = backend
            .allocate_buffer(&context, desc)
            .unwrap()
            .into_parts();
        let payload = float_bytes(case.inputs[0].values.iter().copied());
        backend
            .write_buffer(&mut input, offset, &SliceSource(&payload))
            .unwrap();
        let sentinel = vec![0xa5; buffer_bytes as usize];
        backend
            .write_buffer(&mut output, 0, &SliceSource(&sentinel))
            .unwrap();
        let queue = backend
            .create_queue(&context, QueueDesc::default())
            .unwrap();
        let bindings = [
            BindingRef {
                slot: 0,
                buffer: &input,
                range: BufferRange::new(offset, bytes).unwrap(),
                access: AccessMode::Read,
            },
            BindingRef {
                slot: 1,
                buffer: &output,
                range: BufferRange::new(offset, bytes).unwrap(),
                access: AccessMode::Write,
            },
        ];
        let event = backend
            .submit(&queue, &program, &bindings, Timeout::Infinite)
            .unwrap_or_else(|_| panic!("{device}: offset submission rejected"));
        assert_eq!(wait_for_terminal(&backend, &event), EventState::Complete);
        release(backend.destroy_event(event));
        let mut result = VecSink(vec![0; buffer_bytes as usize]);
        backend.read_buffer(&output, 0, &mut result).unwrap();
        let (head, rest) = result.0.split_at(offset as usize);
        let (middle, tail) = rest.split_at(bytes as usize);
        assert_eq!(
            head,
            &sentinel[..offset as usize],
            "{device}: head clobbered"
        );
        assert_eq!(middle, payload.as_slice(), "{device}");
        assert_eq!(
            tail,
            &sentinel[(offset + bytes) as usize..],
            "{device}: tail clobbered"
        );
        release(backend.destroy_queue(queue));
        release(backend.unload_program(program));
        release(backend.free_buffer(output));
        release(backend.free_buffer(input));
        release(backend.destroy_context(context));
    }
}

#[test]
fn segmented_transfers_reach_device_local_memory_through_staging() {
    for device in devices() {
        let backend = open(&device);
        if !advertised_domains(&backend).contains(&MemoryDomain::Device) {
            eprintln!("{device}: no device-local memory type; staging path not exercised");
            continue;
        }
        let context = backend.create_context(ContextDesc::default()).unwrap();
        let desc = BufferDesc::new(
            64,
            BUFFER_ALIGNMENT,
            MemoryDomain::Device,
            BufferUsage::TRANSFER_SOURCE | BufferUsage::TRANSFER_DESTINATION,
        )
        .unwrap();
        let (mut buffer, info) = backend
            .allocate_buffer(&context, desc)
            .unwrap()
            .into_parts();
        assert!(
            info.properties()
                .contains(virtio_accel_core::BufferProperties::DEVICE_LOCAL)
        );
        let pattern = (0..64_u8).collect::<Vec<_>>();
        backend
            .write_buffer(&mut buffer, 0, &SegmentedSource(&pattern))
            .unwrap();
        let mut sink = SegmentedSink(vec![0; 64]);
        backend.read_buffer(&buffer, 0, &mut sink).unwrap();
        assert_eq!(sink.0, pattern, "{device}");
        // Partial, offset range through the same staging path.
        backend
            .write_buffer(&mut buffer, 8, &SliceSource(&[0xff; 4]))
            .unwrap();
        let mut sink = VecSink(vec![0; 16]);
        backend.read_buffer(&buffer, 0, &mut sink).unwrap();
        assert_eq!(&sink.0[..8], &pattern[..8], "{device}");
        assert_eq!(&sink.0[8..12], &[0xff; 4], "{device}");
        assert_eq!(&sink.0[12..], &pattern[12..16], "{device}");
        assert_eq!(backend.explicit_transfer_bytes(), 64 + 64 + 4 + 16);
        release(backend.free_buffer(buffer));
        release(backend.destroy_context(context));
    }
}

#[test]
fn rejects_out_of_tier_artifacts_before_any_pipeline_exists() {
    for device in devices() {
        let backend = open(&device);
        let context = backend.create_context(ContextDesc::default()).unwrap();
        // CAST is outside the FP32 tier: rejected as unsupported before any pipeline exists.
        let mut cast = OwnedGraph::new("main");
        cast.push_tensor(OwnedTensor::new("x", vec![4], DType::FP32))
            .push_tensor(OwnedTensor::new("y", vec![4], DType::INT32))
            .push_operator(OwnedOperator::new(
                OperatorKind::Cast,
                vec!["x".into()],
                vec!["y".into()],
            ))
            .push_input("x")
            .push_output("y");
        let cast = cast
            .build(VULKAN_TOSA_TARGET)
            .expect("valid FP32 CAST graph");
        assert!(matches!(
            load(&backend, &context, &cast, VULKAN_TOSA_TARGET),
            Err(BackendError::Unsupported)
        ));
        assert!(matches!(
            load(
                &backend,
                &context,
                IDENTITY_INT8.artifact,
                VULKAN_TOSA_INTEGER_TARGET
            ),
            Err(BackendError::Incompatible)
        ));
        // INT8 bytes under the FP32 target: never relabeled.
        assert!(matches!(
            load(
                &backend,
                &context,
                IDENTITY_INT8.artifact,
                VULKAN_TOSA_TARGET
            ),
            Err(BackendError::Unsupported | BackendError::InvalidArgument)
        ));
        // Wrong format and wrong residency promise are rejected before parsing.
        let model = parse(IDENTITY_FP32_LOCAL).unwrap();
        let mut artifact = model
            .artifact_ref(VULKAN_TOSA_TARGET, REQUIRED_RESIDENT_BYTES)
            .unwrap();
        artifact.resident_bytes = 1 << 20;
        assert_eq!(
            backend.load_program(&context, artifact).unwrap_err(),
            BackendError::ResourceLimit
        );
        let garbage = ArtifactRef {
            format: virtio_accel_tosa::ARTIFACT_FORMAT,
            target: VULKAN_TOSA_TARGET.to_identity(),
            payload: &SliceSource(b"not a flatbuffer"),
            resident_bytes: REQUIRED_RESIDENT_BYTES,
        };
        assert_eq!(
            backend.load_program(&context, garbage).unwrap_err(),
            BackendError::InvalidArgument
        );
        assert_eq!(backend.live_resources().programs, 0);
        release(backend.destroy_context(context));
    }
}

#[test]
fn rejects_misaligned_and_mis_sized_bindings_as_incompatible() {
    for device in devices() {
        let backend = open(&device);
        let context = backend.create_context(ContextDesc::default()).unwrap();
        let program = load(&backend, &context, IDENTITY_FP32_LOCAL, VULKAN_TOSA_TARGET).unwrap();
        let desc = BufferDesc::new(
            64,
            BUFFER_ALIGNMENT,
            MemoryDomain::Host,
            BufferUsage::PROGRAM_INPUT | BufferUsage::PROGRAM_OUTPUT,
        )
        .unwrap();
        let (input, _) = backend
            .allocate_buffer(&context, desc)
            .unwrap()
            .into_parts();
        let (output, _) = backend
            .allocate_buffer(&context, desc)
            .unwrap()
            .into_parts();
        let queue = backend
            .create_queue(&context, QueueDesc::default())
            .unwrap();
        let attempt = |input_range: BufferRange, output_range: BufferRange| {
            let bindings = [
                BindingRef {
                    slot: 0,
                    buffer: &input,
                    range: input_range,
                    access: AccessMode::Read,
                },
                BindingRef {
                    slot: 1,
                    buffer: &output,
                    range: output_range,
                    access: AccessMode::Write,
                },
            ];
            match backend.submit(&queue, &program, &bindings, Timeout::Infinite) {
                Err(SubmitFailure::Rejected(error)) => error,
                Ok(event) => {
                    wait_for_terminal(&backend, &event);
                    release(backend.destroy_event(event));
                    panic!("{device}: submission unexpectedly accepted")
                }
                Err(SubmitFailure::Indeterminate { error, .. }) => {
                    panic!("{device}: indeterminate: {error:?}")
                }
            }
        };
        let exact = BufferRange::new(0, 4).unwrap();
        assert_eq!(
            attempt(BufferRange::new(0, 8).unwrap(), exact),
            BackendError::Incompatible,
            "{device}: oversized range"
        );
        assert_eq!(
            attempt(BufferRange::new(1, 4).unwrap(), exact),
            BackendError::Incompatible,
            "{device}: scalar-misaligned offset"
        );
        // In-place identity: one allocation aliased across the input and output slots.
        let aliased = [
            BindingRef {
                slot: 0,
                buffer: &input,
                range: exact,
                access: AccessMode::Read,
            },
            BindingRef {
                slot: 1,
                buffer: &input,
                range: exact,
                access: AccessMode::Write,
            },
        ];
        assert!(matches!(
            backend.submit(&queue, &program, &aliased, Timeout::Infinite),
            Err(SubmitFailure::Rejected(BackendError::Incompatible))
        ));
        // Only one binding for a two-slot program.
        let single = [BindingRef {
            slot: 0,
            buffer: &input,
            range: exact,
            access: AccessMode::Read,
        }];
        assert!(matches!(
            backend.submit(&queue, &program, &single, Timeout::Infinite),
            Err(SubmitFailure::Rejected(BackendError::Incompatible))
        ));
        // Finite deadlines are refused before admission (ADR 0006).
        let bindings = [
            BindingRef {
                slot: 0,
                buffer: &input,
                range: exact,
                access: AccessMode::Read,
            },
            BindingRef {
                slot: 1,
                buffer: &output,
                range: exact,
                access: AccessMode::Write,
            },
        ];
        assert!(matches!(
            backend.submit(
                &queue,
                &program,
                &bindings,
                Timeout::AfterNs(core::num::NonZeroU64::new(1).unwrap())
            ),
            Err(SubmitFailure::Rejected(BackendError::DeadlineExpired))
        ));
        assert_eq!(backend.direct_binding_admissions(), 0);
        assert_eq!(backend.live_resources().events, 0);
        release(backend.destroy_queue(queue));
        release(backend.unload_program(program));
        release(backend.free_buffer(output));
        release(backend.free_buffer(input));
        release(backend.destroy_context(context));
    }
}

#[test]
fn permits_overlapping_read_only_inputs_across_in_flight_submissions() {
    const IN_FLIGHT: usize = 16;
    for device in devices() {
        let backend = open(&device);
        let context = backend.create_context(ContextDesc::default()).unwrap();
        let program = load(&backend, &context, IDENTITY_FP32_LOCAL, VULKAN_TOSA_TARGET).unwrap();
        let queue = backend
            .create_queue(&context, QueueDesc::default())
            .unwrap();
        let payload = float_bytes([7.75]);
        let input_desc = BufferDesc::new(
            4,
            BUFFER_ALIGNMENT,
            MemoryDomain::Host,
            BufferUsage::TRANSFER_DESTINATION | BufferUsage::PROGRAM_INPUT,
        )
        .unwrap();
        let (mut shared_input, _) = backend
            .allocate_buffer(&context, input_desc)
            .unwrap()
            .into_parts();
        backend
            .write_buffer(&mut shared_input, 0, &SliceSource(&payload))
            .unwrap();
        let output_desc = BufferDesc::new(
            4,
            BUFFER_ALIGNMENT,
            MemoryDomain::Host,
            BufferUsage::TRANSFER_SOURCE | BufferUsage::PROGRAM_OUTPUT,
        )
        .unwrap();
        let outputs = (0..IN_FLIGHT)
            .map(|_| {
                backend
                    .allocate_buffer(&context, output_desc)
                    .unwrap()
                    .into_parts()
                    .0
            })
            .collect::<Vec<_>>();
        let mut events = Vec::with_capacity(IN_FLIGHT);
        for output in &outputs {
            let bindings = [
                BindingRef {
                    slot: 0,
                    buffer: &shared_input,
                    range: BufferRange::new(0, 4).unwrap(),
                    access: AccessMode::Read,
                },
                BindingRef {
                    slot: 1,
                    buffer: output,
                    range: BufferRange::new(0, 4).unwrap(),
                    access: AccessMode::Write,
                },
            ];
            events.push(
                backend
                    .submit(&queue, &program, &bindings, Timeout::Infinite)
                    .unwrap_or_else(|_| panic!("{device}: overlapping read-only rejected")),
            );
        }
        // While the reads are in flight the shared input may not be transferred or freed, and
        // the program may not be unloaded. Both refusals are asserted before anything is polled:
        // a submission stays in flight only until its terminal state is *observed*, so polling
        // first lets a device quick enough to retire all `IN_FLIGHT` submissions drop the guards
        // and make either release legitimately succeed.
        assert_eq!(
            backend
                .write_buffer(&mut shared_input, 0, &SliceSource(&payload))
                .unwrap_err(),
            BackendError::Busy,
            "{device}"
        );
        let program = match backend.unload_program(program) {
            Err(ReleaseFailure::Rejected {
                error: BackendError::Busy,
                resource,
            }) => resource,
            Err(other) => panic!("{device}: unexpected unload result {:?}", other.error()),
            Ok(()) => panic!("{device}: unloaded a program with events in flight"),
        };
        for event in &events {
            assert_eq!(wait_for_terminal(&backend, event), EventState::Complete);
        }
        assert_eq!(backend.direct_binding_admissions(), (IN_FLIGHT * 2) as u64);
        assert_eq!(backend.live_resources().events, IN_FLIGHT as u64);
        for event in events {
            release(backend.destroy_event(event));
        }
        for output in outputs {
            let mut bytes = VecSink(vec![0; 4]);
            backend.read_buffer(&output, 0, &mut bytes).unwrap();
            assert_eq!(bytes.0, payload, "{device}");
            release(backend.free_buffer(output));
        }
        release(backend.free_buffer(shared_input));
        release(backend.destroy_queue(queue));
        release(backend.unload_program(program));
        release(backend.destroy_context(context));
        assert_eq!(backend.live_resources(), Default::default());
    }
}

#[test]
fn ring_exhaustion_is_a_resource_limit_not_a_hang() {
    for device in devices() {
        let backend = open(&device);
        let ring = backend.device_info().unwrap().limits.max_events_per_context as usize;
        let context = backend.create_context(ContextDesc::default()).unwrap();
        let program = load(&backend, &context, IDENTITY_FP32_LOCAL, VULKAN_TOSA_TARGET).unwrap();
        let queue = backend
            .create_queue(&context, QueueDesc::default())
            .unwrap();
        let desc = BufferDesc::new(
            4,
            BUFFER_ALIGNMENT,
            MemoryDomain::Host,
            BufferUsage::PROGRAM_INPUT | BufferUsage::PROGRAM_OUTPUT,
        )
        .unwrap();
        let (input, _) = backend
            .allocate_buffer(&context, desc)
            .unwrap()
            .into_parts();
        let outputs = (0..=ring)
            .map(|_| {
                backend
                    .allocate_buffer(&context, desc)
                    .unwrap()
                    .into_parts()
                    .0
            })
            .collect::<Vec<_>>();
        let mut events = Vec::new();
        let mut exhausted = false;
        for output in &outputs {
            let bindings = [
                BindingRef {
                    slot: 0,
                    buffer: &input,
                    range: BufferRange::new(0, 4).unwrap(),
                    access: AccessMode::Read,
                },
                BindingRef {
                    slot: 1,
                    buffer: output,
                    range: BufferRange::new(0, 4).unwrap(),
                    access: AccessMode::Write,
                },
            ];
            match backend.submit(&queue, &program, &bindings, Timeout::Infinite) {
                Ok(event) => events.push(event),
                Err(SubmitFailure::Rejected(BackendError::ResourceLimit)) => {
                    exhausted = true;
                    break;
                }
                Err(failure) => panic!("{device}: unexpected {:?}", failure_error(failure)),
            }
        }
        assert_eq!(events.len(), ring, "{device}: ring depth");
        assert!(
            exhausted,
            "{device}: the ring-plus-one submission must be refused"
        );
        for event in events {
            assert_eq!(wait_for_terminal(&backend, &event), EventState::Complete);
            release(backend.destroy_event(event));
        }
        for output in outputs {
            release(backend.free_buffer(output));
        }
        release(backend.free_buffer(input));
        release(backend.destroy_queue(queue));
        release(backend.unload_program(program));
        release(backend.destroy_context(context));
    }
}

fn failure_error<E>(failure: SubmitFailure<E>) -> BackendError {
    match failure {
        SubmitFailure::Rejected(error) | SubmitFailure::Indeterminate { error, .. } => error,
    }
}

#[test]
fn parents_refuse_release_while_children_live() {
    for device in devices() {
        let backend = open(&device);
        let context = backend.create_context(ContextDesc::default()).unwrap();
        let desc = BufferDesc::new(
            4,
            BUFFER_ALIGNMENT,
            MemoryDomain::Host,
            BufferUsage::TRANSFER_SOURCE,
        )
        .unwrap();
        let (buffer, _) = backend
            .allocate_buffer(&context, desc)
            .unwrap()
            .into_parts();
        let context = match backend.destroy_context(context) {
            Err(ReleaseFailure::Rejected {
                error: BackendError::Busy,
                resource,
            }) => resource,
            Err(other) => panic!("{device}: {:?}", other.error()),
            Ok(()) => panic!("{device}: destroyed a context with a live buffer"),
        };
        release(backend.free_buffer(buffer));
        release(backend.destroy_context(context));
        assert_eq!(backend.live_resources(), Default::default());
    }
}

#[test]
fn repeated_load_unload_is_stable() {
    for device in devices() {
        let backend = open(&device);
        let context = backend.create_context(ContextDesc::default()).unwrap();
        for _ in 0..16 {
            let program =
                load(&backend, &context, IDENTITY_FP32_LOCAL, VULKAN_TOSA_TARGET).unwrap();
            release(backend.unload_program(program));
        }
        release(backend.destroy_context(context));
    }
}

struct Hooks;

impl ConformanceHooks<VulkanAccelerator> for Hooks {
    fn complete_event(
        &self,
        backend: &VulkanAccelerator,
        event: &VulkanEvent,
    ) -> Result<(), BackendError> {
        let deadline = Instant::now() + Duration::from_secs(15);
        loop {
            match backend.poll_event(event)? {
                EventState::Pending => {
                    if Instant::now() >= deadline {
                        return Err(BackendError::DeadlineExpired);
                    }
                    std::thread::yield_now();
                }
                EventState::Complete => return Ok(()),
                EventState::Failed(error) => return Err(error),
                EventState::Cancelled => return Err(BackendError::Busy),
            }
        }
    }

    fn resource_counts(&self, backend: &VulkanAccelerator) -> Option<ResourceCounts> {
        let live = backend.live_resources();
        Some(ResourceCounts {
            contexts: live.contexts,
            buffers: live.buffers,
            programs: live.programs,
            queues: live.queues,
            events: live.events,
        })
    }

    fn submission_path_diagnostics(
        &self,
        backend: &VulkanAccelerator,
    ) -> Option<SubmissionPathDiagnostics> {
        Some(SubmissionPathDiagnostics {
            direct_bindings: backend.direct_binding_admissions(),
            explicit_transfer_bytes: backend.explicit_transfer_bytes(),
            ..SubmissionPathDiagnostics::default()
        })
    }
}

fn conformance_target(domain: MemoryDomain) -> TargetDescription {
    let input = float_bytes([13.5]);
    let program = ProgramFixture::new(
        virtio_accel_tosa::ARTIFACT_FORMAT,
        VULKAN_TOSA_TARGET.to_identity(),
        IDENTITY_FP32_LOCAL,
        REQUIRED_RESIDENT_BYTES,
    )
    .unwrap();
    TargetDescription::with_bindings(
        program,
        vec![
            BindingFixture::read_only(0, domain, BUFFER_ALIGNMENT, input.clone()).unwrap(),
            BindingFixture::new(
                1,
                AccessMode::Write,
                domain,
                BUFFER_ALIGNMENT,
                vec![0; input.len()],
                input,
            )
            .unwrap(),
        ],
    )
    .unwrap()
}

#[test]
fn vulkan_backend_passes_the_standard_semantic_suite_on_every_device() {
    for device in devices() {
        let backend = open(&device);
        for domain in advertised_domains(&backend) {
            let target = conformance_target(domain);
            // `event.pending-release-terminal-stability` needs to observe a pending event at
            // both edges: the poll before the release, and the release itself. A fast device
            // may finish the one-element copy before the first poll ("did not expose a
            // controllable pending event") or between the poll and the release ("pending event
            // release reported success" — releasing the then-Complete event is correct, so the
            // case simply observed nothing). Retry that precondition race a bounded number of
            // times and fail on anything else.
            let mut passed = false;
            for attempt in 1..=8 {
                let report = run(|| open(&device), &target, &Hooks);
                let racy = |case: &virtio_accel_conformance::CaseResult| {
                    case.id == "event.pending-release-terminal-stability"
                        && matches!(&case.status,
                            virtio_accel_conformance::CaseStatus::Failed(message)
                                if message.contains("did not expose a controllable pending event")
                                    || message.contains("pending event release reported success"))
                };
                let racy_precondition = report.cases().iter().any(racy);
                let other_failure = report.failures().any(|case| !racy(case));
                if other_failure || !racy_precondition {
                    assert!(report.passed(), "{device}: {domain:?}: {report}");
                    passed = true;
                    break;
                }
                eprintln!(
                    "{device}: {domain:?}: attempt {attempt} completed before the pending observation; retrying"
                );
            }
            assert!(
                passed,
                "{device}: {domain:?}: the pending-event precondition raced on every attempt"
            );
        }
    }
}

/// Allocate one buffer of `bytes` in `domain`.
fn allocate(
    backend: &VulkanAccelerator,
    context: &<VulkanAccelerator as Accelerator>::Context,
    bytes: u64,
    domain: MemoryDomain,
    usage: BufferUsage,
) -> <VulkanAccelerator as Accelerator>::Buffer {
    let desc = BufferDesc::new(bytes, BUFFER_ALIGNMENT, domain, usage).unwrap();
    backend
        .allocate_buffer(context, desc)
        .unwrap_or_else(|error| panic!("{}: allocate failed: {error:?}", backend.device_name()))
        .into_parts()
        .0
}

/// Submit `program` over `inputs` (slots `0..n`) and one output (slot `n`), wait, and read the
/// output back.
fn execute(
    backend: &VulkanAccelerator,
    context: &<VulkanAccelerator as Accelerator>::Context,
    program: &<VulkanAccelerator as Accelerator>::Program,
    inputs: &[Vec<u8>],
    output_len: usize,
    domain: MemoryDomain,
) -> Vec<u8> {
    let device = backend.device_name();
    let mut input_buffers = Vec::with_capacity(inputs.len());
    for bytes in inputs {
        let mut buffer = allocate(
            backend,
            context,
            bytes.len() as u64,
            domain,
            BufferUsage::TRANSFER_DESTINATION | BufferUsage::PROGRAM_INPUT,
        );
        backend
            .write_buffer(&mut buffer, 0, &SliceSource(bytes))
            .unwrap();
        input_buffers.push(buffer);
    }
    let output = allocate(
        backend,
        context,
        output_len as u64,
        domain,
        BufferUsage::TRANSFER_SOURCE | BufferUsage::PROGRAM_OUTPUT,
    );
    let queue = backend.create_queue(context, QueueDesc::default()).unwrap();
    let mut bindings: Vec<BindingRef<'_, _>> = input_buffers
        .iter()
        .zip(inputs)
        .enumerate()
        .map(|(slot, (buffer, bytes))| BindingRef {
            slot: slot as u32,
            buffer,
            range: BufferRange::new(0, bytes.len() as u64).unwrap(),
            access: AccessMode::Read,
        })
        .collect();
    bindings.push(BindingRef {
        slot: inputs.len() as u32,
        buffer: &output,
        range: BufferRange::new(0, output_len as u64).unwrap(),
        access: AccessMode::Write,
    });
    let event = backend
        .submit(&queue, program, &bindings, Timeout::Infinite)
        .unwrap_or_else(|failure| match failure {
            SubmitFailure::Rejected(error) => panic!("{device}: submission rejected: {error:?}"),
            SubmitFailure::Indeterminate { error, .. } => {
                panic!("{device}: submission indeterminate: {error:?}")
            }
        });
    assert_eq!(
        wait_for_terminal(backend, &event),
        EventState::Complete,
        "{device}"
    );
    release(backend.destroy_event(event));
    let mut bytes = VecSink(vec![0; output_len]);
    backend.read_buffer(&output, 0, &mut bytes).unwrap();
    release(backend.destroy_queue(queue));
    release(backend.free_buffer(output));
    for buffer in input_buffers {
        release(backend.free_buffer(buffer));
    }
    bytes.0
}

/// Load `artifact`, execute it once over `inputs`, and return the output bytes.
fn run_graph(
    backend: &VulkanAccelerator,
    artifact: &[u8],
    inputs: &[Vec<u8>],
    output_len: usize,
    domain: MemoryDomain,
) -> Vec<u8> {
    run_graph_for(
        backend,
        artifact,
        VULKAN_TOSA_TARGET,
        inputs,
        output_len,
        domain,
    )
}

/// [`run_graph`] against an explicit target, for artifacts outside the FP32/FP16 tier.
fn run_graph_for(
    backend: &VulkanAccelerator,
    artifact: &[u8],
    target: Target,
    inputs: &[Vec<u8>],
    output_len: usize,
    domain: MemoryDomain,
) -> Vec<u8> {
    let device = backend.device_name();
    let context = backend.create_context(ContextDesc::default()).unwrap();
    let program = load(backend, &context, artifact, target)
        .unwrap_or_else(|error| panic!("{device}: load failed: {error:?}"));
    let output = execute(backend, &context, &program, inputs, output_len, domain);
    release(backend.unload_program(program));
    release(backend.destroy_context(context));
    assert_eq!(backend.live_resources(), Default::default(), "{device}");
    output
}

fn run_operator_case(
    backend: &VulkanAccelerator,
    case: &TosaFp32OperatorCase,
    domain: MemoryDomain,
) {
    let inputs: Vec<Vec<u8>> = case.inputs.iter().map(|input| input.bytes()).collect();
    let actual = run_graph(
        backend,
        case.artifact,
        &inputs,
        case.output.byte_len(),
        domain,
    );
    assert!(
        case.output_matches(&actual),
        "{}: {} in {domain:?}: expected {:?}, got {:?}",
        backend.device_name(),
        case.name,
        case.output,
        describe(case.output, &actual)
    );
}

/// Render output bytes in the oracle's element type for failure messages.
fn describe(shape: Fp32TierTensor, bytes: &[u8]) -> String {
    match shape {
        Fp32TierTensor::Fp32(_) => format!("{:?}", floats_le(bytes)),
        Fp32TierTensor::Bool(_) => format!("{bytes:?}"),
        Fp32TierTensor::Int32(_) => format!(
            "{:?}",
            bytes
                .chunks_exact(4)
                .map(|chunk| i32::from_le_bytes(chunk.try_into().unwrap()))
                .collect::<Vec<_>>()
        ),
    }
}

fn floats_le(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

fn float_bytes_le(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

/// Every operator of the FP32 tier, every corpus case, every advertised memory domain.
#[test]
fn executes_every_fp32_operator_case_in_every_advertised_domain() {
    for device in devices() {
        let backend = open(&device);
        for domain in advertised_domains(&backend) {
            for case in FP32_OPERATOR_CASE_GROUPS
                .iter()
                .flat_map(|group| group.iter())
            {
                run_operator_case(&backend, case, domain);
            }
        }
    }
}

/// The three-operator graph is one submission: three dispatches over one program arena holding
/// the zero points, the bias, and both intermediates; RESHAPE-free, so no view aliasing here.
#[test]
fn multi_operator_graphs_run_as_one_submission_over_an_arena() {
    for device in devices() {
        let backend = open(&device);
        let context = backend.create_context(ContextDesc::default()).unwrap();
        let program = load(
            &backend,
            &context,
            LINEAR_TANH_FP32.artifact,
            VULKAN_TOSA_TARGET,
        )
        .unwrap();
        assert_eq!(program.dispatch_count(), 3, "{device}");
        assert!(program.arena_bytes() > 0, "{device}");
        let inputs: Vec<Vec<u8>> = LINEAR_TANH_FP32
            .inputs
            .iter()
            .map(|input| input.bytes())
            .collect();
        let events_before = backend.live_resources().events;
        let actual = execute(
            &backend,
            &context,
            &program,
            &inputs,
            LINEAR_TANH_FP32.output.byte_len(),
            MemoryDomain::Host,
        );
        assert_eq!(backend.live_resources().events, events_before, "{device}");
        assert!(
            LINEAR_TANH_FP32.output_matches(&actual),
            "{device}: {:?}",
            floats_le(&actual)
        );
        release(backend.unload_program(program));
        release(backend.destroy_context(context));
    }
}

/// A five-element BOOL output bound inside a larger buffer: the kernel's atomic byte writes must
/// leave every neighbouring byte, including the three trailing bytes of the last word, intact.
#[test]
fn byte_storage_outputs_leave_neighbouring_bytes_untouched() {
    let mut graph = OwnedGraph::new("main");
    graph
        .push_tensor(OwnedTensor::new("a", vec![5], DType::FP32))
        .push_tensor(OwnedTensor::new("b", vec![5], DType::FP32))
        .push_tensor(OwnedTensor::new("y", vec![5], DType::BOOL))
        .push_operator(OwnedOperator::new(
            OperatorKind::Greater,
            vec!["a".into(), "b".into()],
            vec!["y".into()],
        ))
        .push_input("a")
        .push_input("b")
        .push_output("y");
    let artifact = graph.build(VULKAN_TOSA_TARGET).unwrap();
    let a = float_bytes_le(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let b = float_bytes_le(&[5.0, 4.0, 3.0, 2.0, 1.0]);
    for device in devices() {
        let backend = open(&device);
        let context = backend.create_context(ContextDesc::default()).unwrap();
        let program = load(&backend, &context, &artifact, VULKAN_TOSA_TARGET).unwrap();
        // Two inputs and one output need three bindings; every device this backend opens
        // advertises at least that many.
        let max_bindings = backend
            .device_info()
            .unwrap()
            .limits
            .max_bindings_per_submission;
        assert!(
            max_bindings >= 3,
            "{device}: {max_bindings} bindings per submission"
        );
        let offset = 64_u64;
        let total = 192_u64;
        let mut lhs = allocate(
            &backend,
            &context,
            a.len() as u64,
            MemoryDomain::Host,
            BufferUsage::TRANSFER_DESTINATION | BufferUsage::PROGRAM_INPUT,
        );
        let mut rhs = allocate(
            &backend,
            &context,
            b.len() as u64,
            MemoryDomain::Host,
            BufferUsage::TRANSFER_DESTINATION | BufferUsage::PROGRAM_INPUT,
        );
        backend.write_buffer(&mut lhs, 0, &SliceSource(&a)).unwrap();
        backend.write_buffer(&mut rhs, 0, &SliceSource(&b)).unwrap();
        let mut output = allocate(
            &backend,
            &context,
            total,
            MemoryDomain::Host,
            BufferUsage::TRANSFER_SOURCE
                | BufferUsage::TRANSFER_DESTINATION
                | BufferUsage::PROGRAM_OUTPUT,
        );
        let sentinel = vec![0xaa_u8; total as usize];
        backend
            .write_buffer(&mut output, 0, &SliceSource(&sentinel))
            .unwrap();
        let queue = backend
            .create_queue(&context, QueueDesc::default())
            .unwrap();
        let bindings = [
            BindingRef {
                slot: 0,
                buffer: &lhs,
                range: BufferRange::new(0, a.len() as u64).unwrap(),
                access: AccessMode::Read,
            },
            BindingRef {
                slot: 1,
                buffer: &rhs,
                range: BufferRange::new(0, b.len() as u64).unwrap(),
                access: AccessMode::Read,
            },
            BindingRef {
                slot: 2,
                buffer: &output,
                range: BufferRange::new(offset, 5).unwrap(),
                access: AccessMode::Write,
            },
        ];
        let event = backend
            .submit(&queue, &program, &bindings, Timeout::Infinite)
            .unwrap_or_else(|_| panic!("{device}: submission rejected"));
        assert_eq!(wait_for_terminal(&backend, &event), EventState::Complete);
        release(backend.destroy_event(event));
        let mut bytes = VecSink(vec![0; total as usize]);
        backend.read_buffer(&output, 0, &mut bytes).unwrap();
        let bytes = bytes.0;
        let (start, end) = (offset as usize, offset as usize + 5);
        assert_eq!(
            &bytes[start..end],
            &[0, 0, 0, 1, 1],
            "{device}: predicate bytes"
        );
        assert!(
            bytes[..start].iter().all(|byte| *byte == 0xaa),
            "{device}: bytes before the tensor were modified"
        );
        assert!(
            bytes[end..].iter().all(|byte| *byte == 0xaa),
            "{device}: bytes after the tensor were modified: {:?}",
            &bytes[end..end + 8]
        );
        release(backend.destroy_queue(queue));
        release(backend.unload_program(program));
        release(backend.free_buffer(output));
        release(backend.free_buffer(rhs));
        release(backend.free_buffer(lhs));
        release(backend.destroy_context(context));
    }
}

/// BOOL inputs are read as "any nonzero byte is true" and written back canonically as 0/1.
#[test]
fn byte_storage_inputs_treat_any_nonzero_byte_as_true() {
    let mut graph = OwnedGraph::new("main");
    graph
        .push_tensor(OwnedTensor::new("x", vec![5], DType::BOOL))
        .push_tensor(OwnedTensor::new("y", vec![5], DType::BOOL))
        .push_operator(OwnedOperator::new(
            OperatorKind::LogicalNot,
            vec!["x".into()],
            vec!["y".into()],
        ))
        .push_input("x")
        .push_output("y");
    let artifact = graph.build(VULKAN_TOSA_TARGET).unwrap();
    for device in devices() {
        let backend = open(&device);
        let actual = run_graph(
            &backend,
            &artifact,
            &[vec![0xff, 0, 2, 0, 1]],
            5,
            MemoryDomain::Host,
        );
        assert_eq!(actual, vec![0, 1, 0, 1, 0], "{device}");
    }
}

/// Distance in binary32 ulps between two finite values of the same sign.
fn ulp_distance(a: f32, b: f32) -> u64 {
    fn ordered(value: f32) -> i64 {
        let bits = value.to_bits() as i32;
        i64::from(if bits < 0 { i32::MIN - bits } else { bits })
    }
    ordered(a).abs_diff(ordered(b))
}

/// One unary FP32 graph of `elements` elements built with the shared graph builder.
fn unary_artifact(kind: OperatorKind, elements: usize) -> Vec<u8> {
    let mut graph = OwnedGraph::new("main");
    graph
        .push_tensor(OwnedTensor::new("x", vec![elements as i32], DType::FP32))
        .push_tensor(OwnedTensor::new("y", vec![elements as i32], DType::FP32))
        .push_operator(OwnedOperator::new(kind, vec!["x".into()], vec!["y".into()]))
        .push_input("x")
        .push_output("y");
    graph.build(VULKAN_TOSA_TARGET).unwrap()
}

/// The crate-authored SIN, COS, TANH, and ERF evaluations against binary64 references: within
/// a few ulps across the polynomial range and exact for the non-finite edges. This is the
/// evidence behind the FP32 tier's numerics policy (ADR 0007), independent of the driver's own
/// transcendental precision.
#[test]
fn software_transcendentals_track_binary64_references() {
    const ELEMENTS: usize = 4096;
    let mut inputs = Vec::with_capacity(ELEMENTS);
    // Dense sweep of [-40, 40], a coarse sweep to the 8192 range-reduction boundary, tiny
    // values, and the non-finite edges.
    for index in 0..3000 {
        inputs.push(-40.0 + 80.0 * index as f32 / 2999.0);
    }
    for index in 0..1000 {
        inputs.push(-8000.0 + 16_000.0 * index as f32 / 999.0);
    }
    inputs.extend_from_slice(&[
        0.0,
        -0.0,
        1.0e-30,
        -1.0e-30,
        1.0e-7,
        f32::MIN_POSITIVE,
        1.0e6,
        -1.0e6,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        f32::MAX,
    ]);
    inputs.resize(ELEMENTS, 0.5);
    struct Reference {
        kind: OperatorKind,
        name: &'static str,
        f: fn(f64) -> f64,
        max_ulps: u64,
    }
    let references = [
        Reference {
            kind: OperatorKind::Sin,
            name: "sin",
            f: f64::sin,
            max_ulps: 2,
        },
        Reference {
            kind: OperatorKind::Cos,
            name: "cos",
            f: f64::cos,
            max_ulps: 2,
        },
        Reference {
            kind: OperatorKind::Tanh,
            name: "tanh",
            f: f64::tanh,
            max_ulps: 2,
        },
        Reference {
            kind: OperatorKind::Erf,
            name: "erf",
            f: |x| libm_erf(x),
            max_ulps: 4,
        },
    ];
    for device in devices() {
        let backend = open(&device);
        for reference in &references {
            let artifact = unary_artifact(reference.kind, ELEMENTS);
            let actual = floats_le(&run_graph(
                &backend,
                &artifact,
                &[float_bytes_le(&inputs)],
                ELEMENTS * 4,
                MemoryDomain::Host,
            ));
            let mut worst = 0_u64;
            for (x, got) in inputs.iter().zip(&actual) {
                let expected = (reference.f)(f64::from(*x)) as f32;
                if expected.is_nan() {
                    assert!(
                        got.is_nan(),
                        "{device}: {}({x}) = {got}, expected NaN",
                        reference.name
                    );
                    continue;
                }
                if expected.is_infinite() || expected == 0.0 {
                    assert_eq!(
                        got.to_bits(),
                        expected.to_bits(),
                        "{device}: {}({x}) = {got}, expected {expected}",
                        reference.name
                    );
                    continue;
                }
                let ulps = ulp_distance(*got, expected);
                worst = worst.max(ulps);
                assert!(
                    ulps <= reference.max_ulps,
                    "{device}: {}({x}) = {got}, expected {expected} ({ulps} ulps)",
                    reference.name
                );
            }
            eprintln!(
                "{device}: {} worst-case error {worst} ulp over {ELEMENTS} samples",
                reference.name
            );
        }
    }
}

/// `erf` in binary64 (the Rust standard library has no `erf`): the Abramowitz–Stegun 7.1.26
/// form is far too coarse, so use the classic Numerical Recipes Chebyshev `erfc` with
/// fractional error below 1.2e-7 — well inside binary32 for an oracle.
fn libm_erf(x: f64) -> f64 {
    // `erf(±0) = ±0`; the series below would round the sign away.
    if x == 0.0 {
        return x;
    }
    // Use the series for small |x| so the oracle has full relative accuracy near zero.
    if x.abs() < 1.0 {
        let z = x * x;
        let mut term = x;
        let mut sum = x;
        for n in 1..40 {
            term *= -z / n as f64;
            sum += term / (2 * n + 1) as f64;
        }
        return sum * 2.0 / std::f64::consts::PI.sqrt();
    }
    let t = 1.0 / (1.0 + 0.5 * x.abs());
    let poly = -1.265_512_23
        + t * (1.000_023_68
            + t * (0.374_091_96
                + t * (0.096_784_18
                    + t * (-0.186_288_06
                        + t * (0.278_868_07
                            + t * (-1.135_203_98
                                + t * (1.488_515_87 + t * (-0.822_152_23 + t * 0.170_872_77))))))));
    let erfc = t * (-x * x + poly).exp();
    if x < 0.0 { erfc - 1.0 } else { 1.0 - erfc }
}

/// A MATMUL graph with constant zero points over `[batch, m, k] × [batch, k, n]`.
fn matmul_artifact(batch: i32, m: i32, k: i32, n: i32) -> Vec<u8> {
    let zero = 0_f32.to_le_bytes().to_vec();
    let mut graph = OwnedGraph::new("main");
    graph
        .push_tensor(OwnedTensor::new("a", vec![batch, m, k], DType::FP32))
        .push_tensor(OwnedTensor::new("b", vec![batch, k, n], DType::FP32))
        .push_tensor(OwnedTensor::constant(
            "a_zp",
            vec![1],
            DType::FP32,
            zero.clone(),
        ))
        .push_tensor(OwnedTensor::constant("b_zp", vec![1], DType::FP32, zero))
        .push_tensor(OwnedTensor::new("y", vec![batch, m, n], DType::FP32))
        .push_operator(OwnedOperator::new(
            OperatorKind::Const,
            vec![],
            vec!["a_zp".into()],
        ))
        .push_operator(OwnedOperator::new(
            OperatorKind::Const,
            vec![],
            vec!["b_zp".into()],
        ))
        .push_operator(OwnedOperator::new(
            OperatorKind::MatMul,
            vec!["a".into(), "b".into(), "a_zp".into(), "b_zp".into()],
            vec!["y".into()],
        ))
        .push_input("a")
        .push_input("b")
        .push_output("y");
    graph.build(VULKAN_TOSA_TARGET).unwrap()
}

/// `tanh(x · w1) · w2` with both weight matrices as `CONST` tensors, every column of each
/// filled with one value, so every output column of a row must be equal. The second weight
/// matrix is first read by the third dispatch, after the first MATMUL's output has died: a
/// packer that hands that dead region to the constant places load-time bytes where a run-time
/// dispatch writes, and the columns diverge.
fn chained_matmul_constants_artifact(m: i32, k: i32, hidden: i32, n: i32) -> Vec<u8> {
    let zero = 0_f32.to_le_bytes().to_vec();
    let fill = |count: i32, value: f32| -> Vec<u8> {
        (0..count).flat_map(|_| value.to_le_bytes()).collect()
    };
    let mut graph = OwnedGraph::new("main");
    graph
        .push_tensor(OwnedTensor::new("x", vec![1, m, k], DType::FP32))
        .push_tensor(OwnedTensor::constant(
            "w1",
            vec![1, k, hidden],
            DType::FP32,
            fill(k * hidden, 0.03125),
        ))
        .push_tensor(OwnedTensor::constant(
            "w2",
            vec![1, hidden, n],
            DType::FP32,
            fill(hidden * n, 0.0625),
        ))
        .push_tensor(OwnedTensor::constant("zp", vec![1], DType::FP32, zero))
        .push_tensor(OwnedTensor::new("h", vec![1, m, hidden], DType::FP32))
        .push_tensor(OwnedTensor::new("a", vec![1, m, hidden], DType::FP32))
        .push_tensor(OwnedTensor::new("y", vec![1, m, n], DType::FP32));
    for name in ["w1", "w2", "zp"] {
        graph.push_operator(OwnedOperator::new(
            OperatorKind::Const,
            vec![],
            vec![name.into()],
        ));
    }
    graph
        .push_operator(OwnedOperator::new(
            OperatorKind::MatMul,
            vec!["x".into(), "w1".into(), "zp".into(), "zp".into()],
            vec!["h".into()],
        ))
        .push_operator(OwnedOperator::new(
            OperatorKind::Tanh,
            vec!["h".into()],
            vec!["a".into()],
        ))
        .push_operator(OwnedOperator::new(
            OperatorKind::MatMul,
            vec!["a".into(), "w2".into(), "zp".into(), "zp".into()],
            vec!["y".into()],
        ))
        .push_input("x")
        .push_output("y");
    graph.build(VULKAN_TOSA_TARGET).unwrap()
}

/// Regression: a `CONST` first used after an intermediate's arena region was freed must not be
/// packed into that region, because constants are uploaded at load and the intermediate is
/// written by a dispatch that runs afterwards.
#[test]
fn constants_first_used_late_survive_earlier_dispatches() {
    let (m, k, hidden, n) = (64, 8, 16, 3);
    let artifact = chained_matmul_constants_artifact(m, k, hidden, n);
    let x: Vec<f32> = (0..m * k).map(|i| i as f32 / 512.0 - 0.5).collect();
    let input: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
    for device in devices() {
        let backend = open(&device);
        let output = run_graph(
            &backend,
            &artifact,
            std::slice::from_ref(&input),
            (m * n) as usize * 4,
            MemoryDomain::Host,
        );
        let y: Vec<f32> = output
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        for row in 0..m as usize {
            let hidden_value = (x[row * k as usize..(row + 1) * k as usize]
                .iter()
                .sum::<f32>()
                * 0.03125)
                .tanh();
            let expected = hidden_value * 0.0625 * hidden as f32;
            for column in 0..n as usize {
                let got = y[row * n as usize + column];
                assert!(
                    (got - expected).abs() <= 1e-5,
                    "{device}: row {row} column {column}: {got} != {expected}"
                );
            }
        }
    }
}

/// Deterministic pseudo-random values in `[-2, 2)`.
fn pseudo_random(count: usize, seed: u32) -> Vec<f32> {
    let mut state = seed;
    (0..count)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 8) as f32 / (1_u32 << 24) as f32 * 4.0 - 2.0
        })
        .collect()
}

/// The tiled MATMUL kernel is bit-identical to a sequential ascending-k loop with separately
/// rounded multiplies and adds, at sizes that are not tile multiples and across batches.
#[test]
fn tiled_matmul_is_bit_identical_to_the_sequential_reference() {
    for device in devices() {
        let backend = open(&device);
        for (batch, m, k, n) in [
            (1, 1, 1, 1),
            (2, 33, 45, 17),
            (1, 16, 16, 16),
            (3, 7, 100, 5),
        ] {
            let a = pseudo_random((batch * m * k) as usize, 7);
            let b = pseudo_random((batch * k * n) as usize, 11);
            let mut expected = vec![0_f32; (batch * m * n) as usize];
            for z in 0..batch as usize {
                for i in 0..m as usize {
                    for j in 0..n as usize {
                        let mut acc = 0_f32;
                        for kk in 0..k as usize {
                            let product = a[(z * m as usize + i) * k as usize + kk]
                                * b[(z * k as usize + kk) * n as usize + j];
                            acc += product;
                        }
                        expected[(z * m as usize + i) * n as usize + j] = acc;
                    }
                }
            }
            let actual = floats_le(&run_graph(
                &backend,
                &matmul_artifact(batch, m, k, n),
                &[float_bytes_le(&a), float_bytes_le(&b)],
                expected.len() * 4,
                MemoryDomain::Host,
            ));
            for (index, (got, want)) in actual.iter().zip(&expected).enumerate() {
                assert_eq!(
                    got.to_bits(),
                    want.to_bits(),
                    "{device}: [{batch},{m},{k}]x[{batch},{k},{n}] element {index}: {got} vs {want}"
                );
            }
        }
    }
}

/// Rank-4 broadcasting through the strided elementwise path, with a `[2, 1, 3, 1]` operand
/// against `[1, 4, 1, 5]`, and a rank-4 reduction over the middle axis.
#[test]
fn broadcast_elementwise_uses_the_strided_index_path() {
    let mut graph = OwnedGraph::new("main");
    graph
        .push_tensor(OwnedTensor::new("a", vec![2, 1, 3, 1], DType::FP32))
        .push_tensor(OwnedTensor::new("b", vec![1, 4, 1, 5], DType::FP32))
        .push_tensor(OwnedTensor::new("y", vec![2, 4, 3, 5], DType::FP32))
        .push_operator(OwnedOperator::new(
            OperatorKind::Sub,
            vec!["a".into(), "b".into()],
            vec!["y".into()],
        ))
        .push_input("a")
        .push_input("b")
        .push_output("y");
    let artifact = graph.build(VULKAN_TOSA_TARGET).unwrap();
    let a: Vec<f32> = (0..6).map(|i| i as f32 * 100.0).collect();
    let b: Vec<f32> = (0..20).map(|i| i as f32).collect();
    let mut expected = Vec::with_capacity(120);
    for i0 in 0..2 {
        for i1 in 0..4 {
            for i2 in 0..3 {
                for i3 in 0..5 {
                    expected.push(a[i0 * 3 + i2] - b[i1 * 5 + i3]);
                }
            }
        }
    }
    for device in devices() {
        let backend = open(&device);
        let actual = floats_le(&run_graph(
            &backend,
            &artifact,
            &[float_bytes_le(&a), float_bytes_le(&b)],
            expected.len() * 4,
            MemoryDomain::Host,
        ));
        assert_eq!(actual, expected, "{device}");
    }
}

#[test]
#[ignore = "manual native performance evidence"]
fn measures_warm_submission_and_completion_latency() {
    for device in devices() {
        measure_warm_latency_on(&device);
    }
}

fn measure_warm_latency_on(device: &str) {
    let backend = open(device);
    let context = backend.create_context(ContextDesc::default()).unwrap();
    let program = load(&backend, &context, IDENTITY_FP32_LOCAL, VULKAN_TOSA_TARGET).unwrap();
    let queue = backend
        .create_queue(&context, QueueDesc::default())
        .unwrap();
    let desc = BufferDesc::new(
        4,
        BUFFER_ALIGNMENT,
        MemoryDomain::Host,
        BufferUsage::TRANSFER_SOURCE
            | BufferUsage::TRANSFER_DESTINATION
            | BufferUsage::PROGRAM_INPUT
            | BufferUsage::PROGRAM_OUTPUT,
    )
    .unwrap();
    let (input, _) = backend
        .allocate_buffer(&context, desc)
        .unwrap()
        .into_parts();
    let (output, _) = backend
        .allocate_buffer(&context, desc)
        .unwrap()
        .into_parts();

    let submit_once = || {
        let bindings = [
            BindingRef {
                slot: 0,
                buffer: &input,
                range: BufferRange::new(0, 4).unwrap(),
                access: AccessMode::Read,
            },
            BindingRef {
                slot: 1,
                buffer: &output,
                range: BufferRange::new(0, 4).unwrap(),
                access: AccessMode::Write,
            },
        ];
        let started = Instant::now();
        let event = backend
            .submit(&queue, &program, &bindings, Timeout::Infinite)
            .unwrap_or_else(|_| panic!("warm submission rejected"));
        let admission = started.elapsed();
        let deadline = started + Duration::from_secs(15);
        loop {
            match backend.poll_event(&event).unwrap() {
                EventState::Pending => {
                    assert!(Instant::now() < deadline, "submission never completed");
                    std::thread::yield_now();
                }
                EventState::Complete => break,
                state => panic!("unexpected terminal state {state:?}"),
            }
        }
        let completion = started.elapsed();
        release(backend.destroy_event(event));
        (admission, completion)
    };

    for _ in 0..20 {
        submit_once();
    }
    let (mut admission, mut completion): (Vec<_>, Vec<_>) = (0..200).map(|_| submit_once()).unzip();
    admission.sort_unstable();
    completion.sort_unstable();
    eprintln!(
        "device {}: warm admission p50 {:?} p95 {:?} p99 {:?}; submit-to-complete p50 {:?} p95 {:?} p99 {:?}",
        backend.device_name(),
        admission[admission.len() / 2],
        admission[admission.len() * 95 / 100],
        admission[admission.len() * 99 / 100],
        completion[completion.len() / 2],
        completion[completion.len() * 95 / 100],
        completion[completion.len() * 99 / 100],
    );

    release(backend.destroy_queue(queue));
    release(backend.unload_program(program));
    release(backend.free_buffer(input));
    release(backend.free_buffer(output));
    release(backend.destroy_context(context));
}

// ---------------------------------------------------------------------------------------------
// FP16 tier (ADR 0008)
// ---------------------------------------------------------------------------------------------

/// Whether this instance advertises the FP16 tier. With crate-owned conversions the tier needs
/// no device feature (ADR 0008), so this is every native instance; the check stays so the tests
/// assert the advertisement rather than assume it.
fn advertises_fp16(backend: &VulkanAccelerator) -> bool {
    backend
        .tosa_capabilities()
        .iter()
        .any(|capability| capability.supports_dtype(DType::FP16, ValueRoles::INPUT))
}

fn fp16_bytes(values: &[u16]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn fp16s_le(bytes: &[u8]) -> Vec<u16> {
    bytes
        .chunks_exact(2)
        .map(|chunk| u16::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

/// The bit-exact binary16 corpus: every non-NaN value must match bit-for-bit, signed zeros and
/// subnormals included; NaN payloads may canonicalize.
const FP16_BIT_EXACT_CASES: &[&TosaFloat16Case] = &[
    &MATMUL_FP16,
    &MOCK_LINEAR_CLASSIFIER_FP16,
    &ADD_FP16,
    &SUB_FP16,
    &MUL_FP16,
    &POW_FP16,
    &MAXIMUM_FP16,
    &MINIMUM_FP16,
    &MAX_POOL2D_FP16,
    &IDENTITY_EDGES_FP16,
];

#[test]
fn executes_every_fp16_bit_exact_case_in_every_advertised_domain() {
    for device in devices() {
        let backend = open(&device);
        assert!(
            advertises_fp16(&backend),
            "{device}: FP16 tier not advertised"
        );
        for domain in advertised_domains(&backend) {
            for case in FP16_BIT_EXACT_CASES {
                let inputs: Vec<Vec<u8>> = case
                    .inputs
                    .iter()
                    .map(|input| fp16_bytes(input.bits))
                    .collect();
                let actual = run_graph(
                    &backend,
                    case.artifact,
                    &inputs,
                    case.outputs[0].bits.len() * 2,
                    domain,
                );
                let actual = fp16s_le(&actual);
                assert!(
                    case.output_matches(0, &actual),
                    "{device}: {} in {domain:?}: {actual:x?}",
                    case.name
                );
            }
        }
    }
}

/// The ulp-tolerated binary16 corpus: unary and activation, comparison/logical/selection,
/// reduction, and data-movement cases.
const FP16_RAW_CASE_GROUPS: &[&[TosaRawCase]] = &[
    HEXAGON_UNARY_FP16_CASES,
    HEXAGON_LOGICAL_CASES,
    HEXAGON_REDUCTION_CASES,
    HEXAGON_MOVEMENT_CASES,
];

#[test]
fn executes_every_fp16_raw_oracle_case_in_every_advertised_domain() {
    for device in devices() {
        let backend = open(&device);
        assert!(
            advertises_fp16(&backend),
            "{device}: FP16 tier not advertised"
        );
        for domain in advertised_domains(&backend) {
            for case in FP16_RAW_CASE_GROUPS
                .iter()
                .flat_map(|group| group.iter().copied())
            {
                let inputs: Vec<Vec<u8>> = case.inputs.iter().map(|input| input.bytes()).collect();
                let actual = run_graph(
                    &backend,
                    case.artifact,
                    &inputs,
                    case.output.byte_len(),
                    domain,
                );
                assert!(
                    case.output_matches(&actual),
                    "{device}: {} in {domain:?}: {actual:x?}",
                    case.name
                );
            }
        }
    }
}

/// Every one of the 65536 binary16 bit patterns through NEGATE: the sign bit flips, the rest of
/// the pattern — NaN payloads, subnormals, signed zeros — passes through untouched. The lane is
/// an integer sign operation (ADR 0008), so this is exact on every driver.
#[test]
fn fp16_negate_round_trips_every_binary16_bit_pattern() {
    const ELEMENTS: i32 = 65536;
    let mut graph = OwnedGraph::new("main");
    graph
        .push_tensor(OwnedTensor::new("x", vec![ELEMENTS], DType::FP16))
        .push_tensor(OwnedTensor::constant(
            "input_zp",
            vec![1],
            DType::FP16,
            vec![0, 0],
        ))
        .push_tensor(OwnedTensor::constant(
            "output_zp",
            vec![1],
            DType::FP16,
            vec![0, 0],
        ))
        .push_tensor(OwnedTensor::new("y", vec![ELEMENTS], DType::FP16))
        .push_operator(OwnedOperator::new(
            OperatorKind::Const,
            vec![],
            vec!["input_zp".into()],
        ))
        .push_operator(OwnedOperator::new(
            OperatorKind::Const,
            vec![],
            vec!["output_zp".into()],
        ))
        .push_operator(OwnedOperator::new(
            OperatorKind::Negate,
            vec!["x".into(), "input_zp".into(), "output_zp".into()],
            vec!["y".into()],
        ))
        .push_input("x")
        .push_output("y");
    let artifact = graph.build(VULKAN_TOSA_TARGET).unwrap();
    let patterns: Vec<u16> = (0..=u16::MAX).collect();
    let input = fp16_bytes(&patterns);
    for device in devices() {
        let backend = open(&device);
        assert!(
            advertises_fp16(&backend),
            "{device}: FP16 tier not advertised"
        );
        for domain in advertised_domains(&backend) {
            let actual = fp16s_le(&run_graph(
                &backend,
                &artifact,
                std::slice::from_ref(&input),
                input.len(),
                domain,
            ));
            for (index, (expected, actual)) in patterns
                .iter()
                .map(|pattern| pattern ^ 0x8000)
                .zip(&actual)
                .enumerate()
            {
                assert_eq!(
                    expected, *actual,
                    "{device}: {domain:?}: pattern {index:#06x}"
                );
            }
        }
    }
}

/// `expected` and `actual` binary16 patterns within `max_ulps` of each other, NaN-tolerant,
/// sign- and zero-exact (the shared raw-case comparison rule).
fn fp16_within_ulps(expected: u16, actual: u16, max_ulps: u16) -> bool {
    if expected & 0x7c00 == 0x7c00 && expected & 0x03ff != 0 {
        return actual & 0x7c00 == 0x7c00 && actual & 0x03ff != 0;
    }
    if max_ulps == 0 || (expected ^ actual) & 0x8000 != 0 || expected & 0x7fff == 0 {
        return expected == actual;
    }
    expected.abs_diff(actual) <= max_ulps
}

/// Binary16 subnormal arithmetic: the tier's crate-owned widen/narrow conversions produce
/// subnormals on every device, so these lanes must produce the exact IEEE results, subnormal
/// operands included — the held-to-contract check on every stack (ADR 0008).
#[test]
fn fp16_subnormal_arithmetic_is_exact_where_the_tier_is_advertised() {
    struct Probe {
        name: &'static str,
        kind: OperatorKind,
        /// Optional second operand as an in-graph `CONST` (binary16 bits).
        constant: Option<u16>,
        inputs: &'static [u16],
        /// Expected binary16 patterns, or 0/1 bytes for a BOOL output.
        expected: &'static [u16],
        output_bool: bool,
    }
    const SUB_MIN: u16 = 0x0001; // 2^-24, the smallest subnormal
    let probes = &[
        Probe {
            // (-2^-24) + (-2^-24) = -2^-23; the other rows exercise exact cancellation to +0
            // and subnormal sign/zero propagation through the constant operand.
            name: "add-subnormal",
            kind: OperatorKind::Add,
            constant: Some(0x8001),
            inputs: &[0x8001, 0x0001, 0x8000, 0x0000],
            expected: &[0x8002, 0x0000, 0x8001, 0x8001],
            output_bool: false,
        },
        Probe {
            // 2^-23 - 2^-24 = 2^-24; -2^-23 - 2^-24 = -3·2^-24; exact cancellation to +0.
            name: "sub-subnormal",
            kind: OperatorKind::Sub,
            constant: Some(SUB_MIN),
            inputs: &[0x0002, 0x8002, 0x0001],
            expected: &[0x0001, 0x8003, 0x0000],
            output_bool: false,
        },
        Probe {
            name: "mul-by-one",
            kind: OperatorKind::Mul,
            constant: Some(0x3c00),
            inputs: &[0x0001, 0x8001, 0x03ff],
            expected: &[0x0001, 0x8001, 0x03ff],
            output_bool: false,
        },
        Probe {
            name: "mul-producing-subnormal",
            kind: OperatorKind::Mul,
            constant: Some(0x3800),
            inputs: &[0x0002, 0x8002],
            expected: &[0x0001, 0x8001],
            output_bool: false,
        },
        Probe {
            name: "greater-than-zero",
            kind: OperatorKind::Greater,
            constant: Some(0x0000),
            inputs: &[0x0001, 0x8001, 0x0000],
            expected: &[1, 0, 0],
            output_bool: true,
        },
        Probe {
            name: "maximum-with-zero",
            kind: OperatorKind::Maximum {
                nan_mode: NanPropagationMode::PROPAGATE,
            },
            constant: Some(0x0000),
            inputs: &[0x0001, 0x8001],
            expected: &[0x0001, 0x0000],
            output_bool: false,
        },
        Probe {
            name: "minimum-with-zero",
            kind: OperatorKind::Minimum {
                nan_mode: NanPropagationMode::PROPAGATE,
            },
            constant: Some(0x0000),
            inputs: &[0x0001, 0x8001],
            expected: &[0x0000, 0x8001],
            output_bool: false,
        },
        // CEIL/FLOOR have no `OperatorKind` in the shared builder; their native binary16 GLSL
        // lanes are covered by the corpus fixtures and the narrowing sweep instead.
        Probe {
            name: "reciprocal-of-subnormal",
            kind: OperatorKind::Reciprocal,
            constant: None,
            inputs: &[0x0400],
            expected: &[0x7400],
            output_bool: false,
        },
        Probe {
            name: "abs-subnormal",
            kind: OperatorKind::Abs,
            constant: None,
            inputs: &[0x8001, 0x0001],
            expected: &[0x0001, 0x0001],
            output_bool: false,
        },
    ];
    for probe in probes {
        let elements = probe.inputs.len() as i32;
        let mut graph = OwnedGraph::new("main");
        graph.push_tensor(OwnedTensor::new("x", vec![elements], DType::FP16));
        let mut operands = vec!["x".into()];
        if let Some(constant) = probe.constant {
            graph
                .push_tensor(OwnedTensor::constant(
                    "c",
                    vec![1],
                    DType::FP16,
                    fp16_bytes(&[constant]),
                ))
                .push_operator(OwnedOperator::new(
                    OperatorKind::Const,
                    vec![],
                    vec!["c".into()],
                ));
            operands.push("c".into());
        }
        if probe.kind == OperatorKind::Mul {
            // TOSA MUL's third operand is the INT8 shift; the tier admits zero shift only.
            graph
                .push_tensor(OwnedTensor::constant(
                    "shift",
                    vec![1],
                    DType::INT8,
                    vec![0],
                ))
                .push_operator(OwnedOperator::new(
                    OperatorKind::Const,
                    vec![],
                    vec!["shift".into()],
                ));
            operands.push("shift".into());
        }
        graph
            .push_tensor(OwnedTensor::new(
                "y",
                vec![elements],
                if probe.output_bool {
                    DType::BOOL
                } else {
                    DType::FP16
                },
            ))
            .push_operator(OwnedOperator::new(probe.kind, operands, vec!["y".into()]))
            .push_input("x")
            .push_output("y");
        let artifact = graph.build(VULKAN_TOSA_TARGET).unwrap();
        let input = fp16_bytes(probe.inputs);
        let output_len = if probe.output_bool {
            probe.inputs.len()
        } else {
            probe.inputs.len() * 2
        };
        for device in devices() {
            let backend = open(&device);
            assert!(
                advertises_fp16(&backend),
                "{device}: FP16 tier not advertised"
            );
            let actual = run_graph(
                &backend,
                &artifact,
                std::slice::from_ref(&input),
                output_len,
                MemoryDomain::Host,
            );
            if probe.output_bool {
                let expected: Vec<u8> = probe.expected.iter().map(|bit| *bit as u8).collect();
                assert_eq!(actual, expected, "{device}: {}", probe.name);
            } else {
                assert_eq!(
                    fp16s_le(&actual),
                    probe.expected,
                    "{device}: {}",
                    probe.name
                );
            }
        }
    }
}

/// The higher-precision lanes against binary64 references: `SIN`, `COS`, `TANH`, `ERF`, and the
/// `EXP`/`LOG`/`RSQRT`/`SIGMOID` built-ins are evaluated in binary32 and rounded once, so the
/// binary16 result must land within one ulp of the correctly rounded exact value over the whole
/// finite binary16 domain. (`LOG`/`RSQRT` skip negative inputs, whose behaviour Vulkan leaves
/// undefined, and their fixtures elsewhere cover the positive domain.)
#[test]
fn fp16_higher_precision_lanes_track_binary64_references() {
    /// One transcendental case: corpus name, the unary operator, the binary64 oracle, and
    /// whether the domain is restricted to positive inputs.
    struct TranscendentalCase {
        name: &'static str,
        kind: OperatorKind,
        oracle: fn(f64) -> f64,
        positive_only: bool,
    }
    let cases = [
        TranscendentalCase {
            name: "sin",
            kind: OperatorKind::Sin,
            oracle: f64::sin,
            positive_only: false,
        },
        TranscendentalCase {
            name: "cos",
            kind: OperatorKind::Cos,
            oracle: f64::cos,
            positive_only: false,
        },
        TranscendentalCase {
            name: "tanh",
            kind: OperatorKind::Tanh,
            oracle: f64::tanh,
            positive_only: false,
        },
        TranscendentalCase {
            name: "erf",
            kind: OperatorKind::Erf,
            oracle: libm_erf,
            positive_only: false,
        },
        TranscendentalCase {
            name: "exp",
            kind: OperatorKind::Exp,
            oracle: f64::exp,
            positive_only: false,
        },
        TranscendentalCase {
            name: "log",
            kind: OperatorKind::Log,
            oracle: f64::ln,
            positive_only: true,
        },
        TranscendentalCase {
            name: "rsqrt",
            kind: OperatorKind::Rsqrt,
            oracle: |x| 1.0 / x.sqrt(),
            positive_only: true,
        },
        TranscendentalCase {
            name: "sigmoid",
            kind: OperatorKind::Sigmoid,
            oracle: |x| 1.0 / (1.0 + (-x).exp()),
            positive_only: false,
        },
    ];
    // Every finite binary16 pattern as f64-exact input.
    let finite: Vec<u16> = (0..=u16::MAX)
        .filter(|pattern| pattern & 0x7c00 != 0x7c00)
        .collect();
    for TranscendentalCase {
        name,
        kind,
        oracle,
        positive_only,
    } in cases
    {
        let inputs: Vec<u16> = finite
            .iter()
            .copied()
            .filter(|pattern| !positive_only || pattern & 0x8000 == 0)
            .collect();
        let mut graph = OwnedGraph::new("main");
        graph
            .push_tensor(OwnedTensor::new(
                "x",
                vec![inputs.len() as i32],
                DType::FP16,
            ))
            .push_tensor(OwnedTensor::new(
                "y",
                vec![inputs.len() as i32],
                DType::FP16,
            ))
            .push_operator(OwnedOperator::new(kind, vec!["x".into()], vec!["y".into()]))
            .push_input("x")
            .push_output("y");
        let artifact = graph.build(VULKAN_TOSA_TARGET).unwrap();
        let input = fp16_bytes(&inputs);
        let expected: Vec<u16> = inputs
            .iter()
            .map(|pattern| {
                let exact = oracle(f64::from(virtio_accel_vulkan::shader::f16_to_f32(*pattern)));
                virtio_accel_vulkan::shader::f32_to_f16_bits(exact as f32)
            })
            .collect();
        for device in devices() {
            let backend = open(&device);
            assert!(
                advertises_fp16(&backend),
                "{device}: FP16 tier not advertised"
            );
            let actual = fp16s_le(&run_graph(
                &backend,
                &artifact,
                std::slice::from_ref(&input),
                input.len(),
                MemoryDomain::Host,
            ));
            for (index, (expected, actual)) in expected.iter().zip(&actual).enumerate() {
                assert!(
                    fp16_within_ulps(*expected, *actual, 1),
                    "{device}: {name}({:#06x}): expected within 1 ulp of {expected:#06x}, got {actual:#06x}",
                    inputs[index]
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// The FP8 tier (ADR 0009)
// ---------------------------------------------------------------------------------------------

fn advertises_fp8(backend: &VulkanAccelerator) -> bool {
    backend
        .tosa_capabilities()
        .iter()
        .any(|capability| capability.supports_dtype(DType::FP8E4M3, ValueRoles::INPUT))
}

/// Every FP8 bit pattern of `dtype`, in order.
fn all_fp8_patterns() -> Vec<u8> {
    (0..=u8::MAX).collect()
}

/// An `IDENTITY` over `elements` FP8 scalars: the tier's data-movement path.
fn fp8_identity_artifact(dtype: DType, elements: i32) -> Vec<u8> {
    let shape = vec![1, 1, elements];
    let mut graph = OwnedGraph::new("main");
    graph
        .push_tensor(OwnedTensor::new("x", shape.clone(), dtype))
        .push_tensor(OwnedTensor::new("y", shape, dtype))
        .push_operator(OwnedOperator::new(
            OperatorKind::Identity,
            vec!["x".into()],
            vec!["y".into()],
        ))
        .push_input("x")
        .push_output("y");
    graph.build(VULKAN_TOSA_FP8_TARGET).unwrap()
}

/// `(FP8, FP8) -> FP16` MATMUL over `[1, m, k] x [1, k, n]`, with the zero points TOSA requires.
fn fp8_matmul_artifact(dtype: DType, m: i32, k: i32, n: i32) -> Vec<u8> {
    let mut graph = OwnedGraph::new("main");
    graph
        .push_tensor(OwnedTensor::new("a", vec![1, m, k], dtype))
        .push_tensor(OwnedTensor::new("b", vec![1, k, n], dtype))
        .push_tensor(OwnedTensor::constant("a_zp", vec![1], dtype, vec![0]))
        .push_tensor(OwnedTensor::constant("b_zp", vec![1], dtype, vec![0]))
        .push_tensor(OwnedTensor::new("y", vec![1, m, n], DType::FP16))
        .push_operator(OwnedOperator::new(
            OperatorKind::Const,
            vec![],
            vec!["a_zp".into()],
        ))
        .push_operator(OwnedOperator::new(
            OperatorKind::Const,
            vec![],
            vec!["b_zp".into()],
        ))
        .push_operator(OwnedOperator::new(
            OperatorKind::MatMul,
            vec!["a".into(), "b".into(), "a_zp".into(), "b_zp".into()],
            vec!["y".into()],
        ))
        .push_input("a")
        .push_input("b")
        .push_output("y");
    graph.build(VULKAN_TOSA_FP8_TARGET).unwrap()
}

/// Exhaustive: all 256 patterns of each FP8 encoding move through `IDENTITY` bit-for-bit. The
/// analogue of the FP16 tier's 65536-pattern `NEGATE` round trip, and the reason data movement
/// is a raw byte copy rather than a widen/narrow pair — NaNs, subnormals and signed zeros all
/// survive.
#[test]
fn every_fp8_pattern_moves_bit_exactly_on_every_device() {
    let patterns = all_fp8_patterns();
    for device in devices() {
        let backend = open(&device);
        assert!(
            advertises_fp8(&backend),
            "{device}: FP8 tier not advertised"
        );
        for dtype in [DType::FP8E4M3, DType::FP8E5M2] {
            let artifact = fp8_identity_artifact(dtype, patterns.len() as i32);
            for domain in advertised_domains(&backend) {
                let actual = run_graph_for(
                    &backend,
                    &artifact,
                    VULKAN_TOSA_FP8_TARGET,
                    std::slice::from_ref(&patterns),
                    patterns.len(),
                    domain,
                );
                assert_eq!(
                    actual, patterns,
                    "{device}: {dtype:?} identity in {domain:?} changed a bit pattern"
                );
            }
        }
    }
}

/// `(FP8, FP8) -> FP16` MATMUL against a host reference that widens with the crate's own
/// decoders and accumulates in binary32 — the accumulator width TOSA assigns FP8 MATMUL — then
/// narrows once. Bit-exact: the kernel performs the same operations in the same order.
#[test]
fn fp8_matmul_matches_the_widened_reference_on_every_device() {
    let (m, k, n) = (5_i32, 7_i32, 3_i32);
    for device in devices() {
        let backend = open(&device);
        assert!(
            advertises_fp8(&backend),
            "{device}: FP8 tier not advertised"
        );
        for dtype in [DType::FP8E4M3, DType::FP8E5M2] {
            // Finite patterns only: this checks arithmetic, and the exhaustive identity test
            // above already covers NaN and infinity transport.
            let finite = |seed: usize, count: usize| -> Vec<u8> {
                (0..count)
                    .map(|index| {
                        let bits = ((index * 37 + seed * 11) % 120) as u8;
                        // Keep both operands well inside each format's finite range.
                        if index % 3 == 0 { bits | 0x80 } else { bits }
                    })
                    .collect()
            };
            let a = finite(1, (m * k) as usize);
            let b = finite(2, (k * n) as usize);
            let widen = |bits: u8| -> f32 {
                match dtype {
                    DType::FP8E4M3 => virtio_accel_tosa::fp8e4m3_to_f32(bits),
                    _ => virtio_accel_tosa::fp8e5m2_to_f32(bits),
                }
            };
            let mut expected = Vec::new();
            for row in 0..m as usize {
                for column in 0..n as usize {
                    let mut accumulator = 0.0_f32;
                    for inner in 0..k as usize {
                        accumulator += widen(a[row * k as usize + inner])
                            * widen(b[inner * n as usize + column]);
                    }
                    expected.push(virtio_accel_vulkan::shader::f32_to_f16_bits(accumulator));
                }
            }
            let artifact = fp8_matmul_artifact(dtype, m, k, n);
            for domain in advertised_domains(&backend) {
                let actual = run_graph_for(
                    &backend,
                    &artifact,
                    VULKAN_TOSA_FP8_TARGET,
                    &[a.clone(), b.clone()],
                    (m * n) as usize * 2,
                    domain,
                );
                assert_eq!(
                    fp16s_le(&actual),
                    expected,
                    "{device}: {dtype:?} MATMUL in {domain:?}"
                );
            }
        }
    }
}

/// The FP8 tier is a distinct target, and the extension bits are load-bearing: an FP8 artifact
/// is refused under the FP32/FP16 target, because TOSA gates FP8 legality on the extension that
/// target does not carry. The converse is deliberately *not* asserted — the FP8 target's
/// envelope is a superset, so an FP32 graph remains legal under it.
#[test]
fn fp8_artifacts_are_refused_under_the_float_target() {
    for device in devices() {
        let backend = open(&device);
        let context = backend.create_context(ContextDesc::default()).unwrap();
        for dtype in [DType::FP8E4M3, DType::FP8E5M2] {
            let fp8 = fp8_identity_artifact(dtype, 4);
            assert!(
                load(&backend, &context, &fp8, VULKAN_TOSA_TARGET).is_err(),
                "{device}: a {dtype:?} artifact loaded under the float target"
            );
        }
        release(backend.destroy_context(context));
    }
}

/// An FP8 weight matrix as an in-graph `CONST`: the dominant shape, and the reason the tier
/// does not need `CAST` to be useful. The host narrows once (as `axnn` does) and the constant
/// crosses as packed bytes.
#[test]
fn fp8_matmul_admits_a_constant_weight_matrix() {
    let (m, k, n) = (4_i32, 6_i32, 2_i32);
    let weights: Vec<u8> = (0..(k * n) as usize)
        .map(|index| ((index * 9 + 56) % 120) as u8)
        .collect();
    for device in devices() {
        let backend = open(&device);
        for dtype in [DType::FP8E4M3, DType::FP8E5M2] {
            let mut graph = OwnedGraph::new("main");
            graph
                .push_tensor(OwnedTensor::new("a", vec![1, m, k], dtype))
                .push_tensor(OwnedTensor::constant(
                    "w",
                    vec![1, k, n],
                    dtype,
                    weights.clone(),
                ))
                .push_tensor(OwnedTensor::constant("zp", vec![1], dtype, vec![0]))
                .push_tensor(OwnedTensor::new("y", vec![1, m, n], DType::FP16));
            for name in ["w", "zp"] {
                graph.push_operator(OwnedOperator::new(
                    OperatorKind::Const,
                    vec![],
                    vec![name.into()],
                ));
            }
            graph
                .push_operator(OwnedOperator::new(
                    OperatorKind::MatMul,
                    vec!["a".into(), "w".into(), "zp".into(), "zp".into()],
                    vec!["y".into()],
                ))
                .push_input("a")
                .push_output("y");
            let artifact = graph.build(VULKAN_TOSA_FP8_TARGET).unwrap();
            let a: Vec<u8> = (0..(m * k) as usize)
                .map(|index| ((index * 13 + 40) % 120) as u8)
                .collect();
            let widen = |bits: u8| match dtype {
                DType::FP8E4M3 => virtio_accel_tosa::fp8e4m3_to_f32(bits),
                _ => virtio_accel_tosa::fp8e5m2_to_f32(bits),
            };
            let mut expected = Vec::new();
            for row in 0..m as usize {
                for column in 0..n as usize {
                    let mut accumulator = 0.0_f32;
                    for inner in 0..k as usize {
                        accumulator += widen(a[row * k as usize + inner])
                            * widen(weights[inner * n as usize + column]);
                    }
                    expected.push(virtio_accel_vulkan::shader::f32_to_f16_bits(accumulator));
                }
            }
            for domain in advertised_domains(&backend) {
                let actual = run_graph_for(
                    &backend,
                    &artifact,
                    VULKAN_TOSA_FP8_TARGET,
                    std::slice::from_ref(&a),
                    (m * n) as usize * 2,
                    domain,
                );
                assert_eq!(
                    fp16s_le(&actual),
                    expected,
                    "{device}: {dtype:?} constant-weight MATMUL in {domain:?}"
                );
            }
        }
    }
}

/// A `CAST` between two float dtypes, shape `[1, 1, elements]`.
fn fp8_cast_artifact(from: DType, to: DType, elements: i32) -> Vec<u8> {
    let shape = vec![1, 1, elements];
    let mut graph = OwnedGraph::new("main");
    graph
        .push_tensor(OwnedTensor::new("x", shape.clone(), from))
        .push_tensor(OwnedTensor::new("y", shape, to))
        .push_operator(OwnedOperator::new(
            OperatorKind::Cast,
            vec!["x".into()],
            vec!["y".into()],
        ))
        .push_input("x")
        .push_output("y");
    graph.build(VULKAN_TOSA_FP8_TARGET).unwrap()
}

/// Exhaustive both ways: every FP8 encoding widens to the value the TOSA crate's own decoder
/// gives, and every finite value returns to its own encoding. The narrowing direction is the
/// device twin of `f32_to_fp8_bits`, whose policy and rounding the unit tests pin.
#[test]
fn fp8_cast_round_trips_every_encoding_on_every_device() {
    let patterns: Vec<u8> = (0..=u8::MAX).collect();
    for device in devices() {
        let backend = open(&device);
        for dtype in [DType::FP8E4M3, DType::FP8E5M2] {
            let decode = |bits: u8| match dtype {
                DType::FP8E4M3 => virtio_accel_tosa::fp8e4m3_to_f32(bits),
                _ => virtio_accel_tosa::fp8e5m2_to_f32(bits),
            };
            let widen = fp8_cast_artifact(dtype, DType::FP32, patterns.len() as i32);
            let narrow = fp8_cast_artifact(DType::FP32, dtype, patterns.len() as i32);
            for domain in advertised_domains(&backend) {
                let widened = run_graph_for(
                    &backend,
                    &widen,
                    VULKAN_TOSA_FP8_TARGET,
                    std::slice::from_ref(&patterns),
                    patterns.len() * 4,
                    domain,
                );
                let widened: Vec<f32> = widened
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                    .collect();
                for (bits, actual) in patterns.iter().zip(&widened) {
                    let expected = decode(*bits);
                    assert!(
                        (expected.is_nan() && actual.is_nan())
                            || expected.to_bits() == actual.to_bits(),
                        "{device}: {dtype:?} widen {bits:#04x} in {domain:?}: {actual}"
                    );
                }
                // Narrow the finite values back; NaN and infinity are policy, covered by the
                // unit tests and the case below.
                let bytes: Vec<u8> = widened
                    .iter()
                    .flat_map(|value| if value.is_finite() { *value } else { 0.0 }.to_le_bytes())
                    .collect();
                let narrowed = run_graph_for(
                    &backend,
                    &narrow,
                    VULKAN_TOSA_FP8_TARGET,
                    std::slice::from_ref(&bytes),
                    patterns.len(),
                    domain,
                );
                for (index, bits) in patterns.iter().enumerate() {
                    let expected = if decode(*bits).is_finite() { *bits } else { 0 };
                    assert_eq!(
                        narrowed[index], expected,
                        "{device}: {dtype:?} narrow {bits:#04x} in {domain:?}"
                    );
                }
            }
        }
    }
}

/// The device honours the overflow policy: E4M3 takes an out-of-range magnitude to NaN rather
/// than saturating to 448, and E5M2 takes it to infinity.
#[test]
fn fp8_cast_overflow_policy_holds_on_every_device() {
    let values: [f32; 6] = [464.0, 464.001, -464.001, 1.0e9, -1.0e9, 100_000.0];
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    for device in devices() {
        let backend = open(&device);
        for dtype in [DType::FP8E4M3, DType::FP8E5M2] {
            let format = match dtype {
                DType::FP8E4M3 => virtio_accel_vulkan::shader::Fp8Format::E4M3,
                _ => virtio_accel_vulkan::shader::Fp8Format::E5M2,
            };
            let expected: Vec<u8> = values
                .iter()
                .map(|value| virtio_accel_vulkan::shader::f32_to_fp8_bits(format, *value))
                .collect();
            let artifact = fp8_cast_artifact(DType::FP32, dtype, values.len() as i32);
            for domain in advertised_domains(&backend) {
                let actual = run_graph_for(
                    &backend,
                    &artifact,
                    VULKAN_TOSA_FP8_TARGET,
                    std::slice::from_ref(&bytes),
                    values.len(),
                    domain,
                );
                assert_eq!(
                    actual, expected,
                    "{device}: {dtype:?} overflow in {domain:?}"
                );
            }
        }
    }
}

/// Two FP8 matmuls chained through a `CAST`, which is the shape `CAST` exists for: layer one
/// produces FP16, the cast re-narrows it to FP8, and layer two consumes that — with no host
/// round trip between the layers.
#[test]
fn chained_fp8_matmuls_need_no_host_round_trip() {
    let (m, k, h, n) = (3_i32, 4_i32, 5_i32, 2_i32);
    for device in devices() {
        let backend = open(&device);
        for dtype in [DType::FP8E4M3, DType::FP8E5M2] {
            let format = match dtype {
                DType::FP8E4M3 => virtio_accel_vulkan::shader::Fp8Format::E4M3,
                _ => virtio_accel_vulkan::shader::Fp8Format::E5M2,
            };
            let decode = |bits: u8| match dtype {
                DType::FP8E4M3 => virtio_accel_tosa::fp8e4m3_to_f32(bits),
                _ => virtio_accel_tosa::fp8e5m2_to_f32(bits),
            };
            let pick = |seed: usize, count: usize| -> Vec<u8> {
                (0..count)
                    .map(|index| ((index * 7 + seed * 5) % 40 + 48) as u8)
                    .collect()
            };
            let a = pick(1, (m * k) as usize);
            let w1 = pick(2, (k * h) as usize);
            let w2 = pick(3, (h * n) as usize);

            let mut graph = OwnedGraph::new("main");
            graph
                .push_tensor(OwnedTensor::new("a", vec![1, m, k], dtype))
                .push_tensor(OwnedTensor::constant(
                    "w1",
                    vec![1, k, h],
                    dtype,
                    w1.clone(),
                ))
                .push_tensor(OwnedTensor::constant(
                    "w2",
                    vec![1, h, n],
                    dtype,
                    w2.clone(),
                ))
                .push_tensor(OwnedTensor::constant("zp", vec![1], dtype, vec![0]))
                .push_tensor(OwnedTensor::new("h16", vec![1, m, h], DType::FP16))
                .push_tensor(OwnedTensor::new("h8", vec![1, m, h], dtype))
                .push_tensor(OwnedTensor::new("y", vec![1, m, n], DType::FP16));
            for name in ["w1", "w2", "zp"] {
                graph.push_operator(OwnedOperator::new(
                    OperatorKind::Const,
                    vec![],
                    vec![name.into()],
                ));
            }
            graph
                .push_operator(OwnedOperator::new(
                    OperatorKind::MatMul,
                    vec!["a".into(), "w1".into(), "zp".into(), "zp".into()],
                    vec!["h16".into()],
                ))
                .push_operator(OwnedOperator::new(
                    OperatorKind::Cast,
                    vec!["h16".into()],
                    vec!["h8".into()],
                ))
                .push_operator(OwnedOperator::new(
                    OperatorKind::MatMul,
                    vec!["h8".into(), "w2".into(), "zp".into(), "zp".into()],
                    vec!["y".into()],
                ))
                .push_input("a")
                .push_output("y");
            let artifact = graph.build(VULKAN_TOSA_FP8_TARGET).unwrap();

            // Host reference: widen, accumulate in binary32, narrow to FP16, re-narrow to FP8,
            // widen again, second matmul, narrow to FP16.
            let mut hidden = Vec::new();
            for row in 0..m as usize {
                for column in 0..h as usize {
                    let mut accumulator = 0.0_f32;
                    for inner in 0..k as usize {
                        accumulator += decode(a[row * k as usize + inner])
                            * decode(w1[inner * h as usize + column]);
                    }
                    let as_f16 = virtio_accel_vulkan::shader::f16_to_f32(
                        virtio_accel_vulkan::shader::f32_to_f16_bits(accumulator),
                    );
                    hidden.push(decode(virtio_accel_vulkan::shader::f32_to_fp8_bits(
                        format, as_f16,
                    )));
                }
            }
            let mut expected = Vec::new();
            for row in 0..m as usize {
                for column in 0..n as usize {
                    let mut accumulator = 0.0_f32;
                    for inner in 0..h as usize {
                        accumulator += hidden[row * h as usize + inner]
                            * decode(w2[inner * n as usize + column]);
                    }
                    expected.push(virtio_accel_vulkan::shader::f32_to_f16_bits(accumulator));
                }
            }
            for domain in advertised_domains(&backend) {
                let actual = run_graph_for(
                    &backend,
                    &artifact,
                    VULKAN_TOSA_FP8_TARGET,
                    std::slice::from_ref(&a),
                    (m * n) as usize * 2,
                    domain,
                );
                assert_eq!(
                    fp16s_le(&actual),
                    expected,
                    "{device}: chained {dtype:?} matmuls in {domain:?}"
                );
            }
        }
    }
}

/// `MAX_POOL2D` over FP8: a 1x1x4x4 NHWC plane pooled 2x2, and `ARGMAX` over the last axis of
/// an FP8 tensor. Pooling selects an existing encoding and ARGMAX emits an INT32 index, so
/// neither introduces a rounding decision — the results are exact by construction.
#[test]
fn fp8_pooling_and_argmax_execute_on_every_device() {
    for device in devices() {
        let backend = open(&device);
        for dtype in [DType::FP8E4M3, DType::FP8E5M2] {
            let decode = |bits: u8| match dtype {
                DType::FP8E4M3 => virtio_accel_tosa::fp8e4m3_to_f32(bits),
                _ => virtio_accel_tosa::fp8e5m2_to_f32(bits),
            };
            // Finite, distinct, both signs.
            let values: Vec<u8> = (0..16).map(|i| ((i * 5 + 48) % 112) as u8).collect();

            let mut pool = OwnedGraph::new("main");
            pool.push_tensor(OwnedTensor::new("x", vec![1, 4, 4, 1], dtype))
                .push_tensor(OwnedTensor::new("y", vec![1, 2, 2, 1], dtype))
                .push_operator(OwnedOperator::new(
                    OperatorKind::MaxPool2d {
                        kernel: [2, 2],
                        stride: [2, 2],
                        pad: [0; 4],
                        nan_mode: NanPropagationMode::PROPAGATE,
                    },
                    vec!["x".into()],
                    vec!["y".into()],
                ))
                .push_input("x")
                .push_output("y");
            let pool = pool.build(VULKAN_TOSA_FP8_TARGET).unwrap();

            let mut expected_pool = Vec::new();
            for row in 0..2usize {
                for column in 0..2usize {
                    let window = [
                        values[row * 8 + column * 2],
                        values[row * 8 + column * 2 + 1],
                        values[row * 8 + 4 + column * 2],
                        values[row * 8 + 4 + column * 2 + 1],
                    ];
                    let best = window
                        .iter()
                        .copied()
                        .max_by(|a, b| decode(*a).total_cmp(&decode(*b)))
                        .unwrap();
                    expected_pool.push(best);
                }
            }

            let mut argmax = OwnedGraph::new("main");
            argmax
                .push_tensor(OwnedTensor::new("x", vec![1, 4, 4], dtype))
                .push_tensor(OwnedTensor::new("y", vec![1, 4], DType::INT32))
                .push_operator(OwnedOperator::new(
                    OperatorKind::ArgMax {
                        axis: 2,
                        nan_mode: NanPropagationMode::PROPAGATE,
                    },
                    vec!["x".into()],
                    vec!["y".into()],
                ))
                .push_input("x")
                .push_output("y");
            let argmax = argmax.build(VULKAN_TOSA_FP8_TARGET).unwrap();

            let expected_argmax: Vec<i32> = (0..4usize)
                .map(|row| {
                    let lane = &values[row * 4..row * 4 + 4];
                    let mut best = 0usize;
                    for index in 1..4 {
                        if decode(lane[index]) > decode(lane[best]) {
                            best = index;
                        }
                    }
                    best as i32
                })
                .collect();

            for domain in advertised_domains(&backend) {
                let actual = run_graph_for(
                    &backend,
                    &pool,
                    VULKAN_TOSA_FP8_TARGET,
                    std::slice::from_ref(&values),
                    4,
                    domain,
                );
                assert_eq!(
                    actual, expected_pool,
                    "{device}: {dtype:?} MAX_POOL2D in {domain:?}"
                );
                let actual = run_graph_for(
                    &backend,
                    &argmax,
                    VULKAN_TOSA_FP8_TARGET,
                    std::slice::from_ref(&values),
                    4 * 4,
                    domain,
                );
                let actual: Vec<i32> = actual
                    .chunks_exact(4)
                    .map(|chunk| i32::from_le_bytes(chunk.try_into().unwrap()))
                    .collect();
                assert_eq!(
                    actual, expected_argmax,
                    "{device}: {dtype:?} ARGMAX in {domain:?}"
                );
            }
        }
    }
}

/// The FP8 MATMUL at aftershock's colour-stage scale and value range, which the small-shape
/// tests above never reach: `(m, k, n) = (6912, 128, 3)`, falloff in `(0, 1]` and colours around
/// `0.01..0.08`.
///
/// Aftershock observed a light-field mismatch of 0.047 against a peak of 1.0 on this shape --
/// bit-identical on ANV and on lavapipe, which rules out a driver -- while every existing FP8
/// test passed on four stacks. The existing tests use `(5, 7, 3)`, so either the shape or the
/// distribution is what they miss. This test exists to say which.
///
/// Unlike the small-shape test this compares in binary32 with a tolerance rather than asserting
/// bit equality. With `k = 128` a tiled kernel may sum in a different order than the sequential
/// reference, and f32 reassociation alone would break exact equality without anything being
/// wrong. The tolerance is generous against the observed 4.7%-of-peak error: anything near that
/// magnitude fails loudly, while legitimate reassociation lands orders of magnitude below it.
#[test]
fn fp8_matmul_holds_at_aftershock_scale() {
    let (m, k, n) = (6912_usize, 128_usize, 3_usize);
    for device in devices() {
        let backend = open(&device);
        assert!(
            advertises_fp8(&backend),
            "{device}: FP8 tier not advertised"
        );
        for dtype in [DType::FP8E4M3, DType::FP8E5M2] {
            let format = match dtype {
                DType::FP8E4M3 => virtio_accel_vulkan::shader::Fp8Format::E4M3,
                _ => virtio_accel_vulkan::shader::Fp8Format::E5M2,
            };
            let narrow = |value: f32| virtio_accel_vulkan::shader::f32_to_fp8_bits(format, value);
            let widen = |bits: u8| -> f32 {
                match dtype {
                    DType::FP8E4M3 => virtio_accel_tosa::fp8e4m3_to_f32(bits),
                    _ => virtio_accel_tosa::fp8e5m2_to_f32(bits),
                }
            };

            // Falloff: exp of a non-positive quantity, so (0, 1] with most mass small -- the
            // distribution that matters, since E4M3 subnormals begin below 2^-6.
            let a: Vec<u8> = (0..m * k)
                .map(|index| {
                    let t = (index % 997) as f32 / 997.0;
                    narrow((-6.0 * t).exp())
                })
                .collect();
            // Colours: the same order of magnitude aftershock uses.
            let b: Vec<u8> = (0..k * n)
                .map(|index| narrow(0.01 + (index % 7) as f32 * 0.01))
                .collect();

            let mut expected = Vec::with_capacity(m * n);
            for row in 0..m {
                for column in 0..n {
                    let mut accumulator = 0.0_f32;
                    for inner in 0..k {
                        accumulator += widen(a[row * k + inner]) * widen(b[inner * n + column]);
                    }
                    expected.push(accumulator);
                }
            }
            let peak = expected
                .iter()
                .fold(0.0_f32, |worst, &v| worst.max(v.abs()));
            assert!(peak > 0.0, "reference is empty");

            let artifact = fp8_matmul_artifact(dtype, m as i32, k as i32, n as i32);
            for domain in advertised_domains(&backend) {
                let actual = run_graph_for(
                    &backend,
                    &artifact,
                    VULKAN_TOSA_FP8_TARGET,
                    &[a.clone(), b.clone()],
                    m * n * 2,
                    domain,
                );
                let got = fp16s_le(&actual);
                assert_eq!(
                    got.len(),
                    expected.len(),
                    "{device}: {dtype:?} output length"
                );
                let mut worst = 0.0_f32;
                let mut worst_at = 0;
                for (index, (&bits, &reference)) in got.iter().zip(&expected).enumerate() {
                    let error = (virtio_accel_vulkan::shader::f16_to_f32(bits) - reference).abs();
                    if error > worst {
                        worst = error;
                        worst_at = index;
                    }
                }
                eprintln!(
                    "{device}: {dtype:?} {domain:?} worst abs {worst:.6} at {worst_at} \
                     (peak {peak:.6}, {:.3}% of peak)",
                    worst / peak * 100.0
                );
                assert!(
                    worst < peak * 0.01,
                    "{device}: {dtype:?} in {domain:?} is off by {worst} at index {worst_at}, \
                     {:.2}% of peak {peak} -- far past f32 reassociation",
                    worst / peak * 100.0
                );
            }
        }
    }
}
