//! Native integration and acceptance suite for the Vulkan backend.
//!
//! Runs against every suitable device the loader enumerates (a real GPU and a software ICD such as
//! lavapipe both count). Without a loader or device the tests skip, unless
//! `VIRTIO_ACCEL_VULKAN_REQUIRE_DEVICE=1` turns absence into a failure (the CI lane sets it so a
//! silently missing ICD cannot pass as green). Placeholder builds do not compile this file.

#![cfg(va_vulkan)]

use std::time::{Duration, Instant};

use virtio_accel_conformance::numerics::{
    IDENTITY_EDGES_FP32, IDENTITY_INT8, MATMUL_FP32, MAX_POOL2D_FP32,
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
use virtio_accel_tosa::{Target, parse};
use virtio_accel_vulkan::{
    InitError, REQUIRED_RESIDENT_BYTES, VULKAN_TOSA_INTEGER_TARGET, VULKAN_TOSA_TARGET,
    VulkanAccelerator, VulkanEvent,
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
        // MAX_POOL2D is not yet admitted: the FP32 base tier grows one operator at a time.
        assert!(matches!(
            load(
                &backend,
                &context,
                MAX_POOL2D_FP32.artifact,
                VULKAN_TOSA_TARGET
            ),
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
            // `event.pending-release-terminal-stability` needs to observe a pending event before
            // the completion hook; a fast device may finish a one-element copy first. Retry that
            // specific precondition race a bounded number of times and fail on anything else.
            let mut passed = false;
            for attempt in 1..=8 {
                let report = run(|| open(&device), &target, &Hooks);
                let racy = |case: &virtio_accel_conformance::CaseResult| {
                    case.id == "event.pending-release-terminal-stability"
                        && matches!(&case.status,
                            virtio_accel_conformance::CaseStatus::Failed(message)
                                if message.contains("did not expose a controllable pending event"))
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
