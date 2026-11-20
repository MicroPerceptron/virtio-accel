use crate::cases::{Case, Workload};
use divan::{
    Bencher,
    counter::{BytesCount, ItemsCount},
};
use std::{
    fs::OpenOptions,
    io::Write,
    time::{Duration, Instant},
};
use virtio_accel_core::*;
use virtio_accel_tosa::{TosaCapabilityProvider, parse};

pub enum Phase {
    LoadUnload,
    Upload,
    Download,
    Warm,
    RoundTrip,
}
impl Phase {
    fn boundary(&self) -> &'static str {
        match self {
            Self::LoadUnload => {
                "host wall clock: load_program plus unload_program; caches retained"
            }
            Self::Upload => "host wall clock: write_buffer for all inputs including weights",
            Self::Download => "host wall clock: read_buffer into preallocated host output",
            Self::Warm => {
                "host wall clock: bindings, submit, terminal polling and event release; resident buffers"
            }
            Self::RoundTrip => {
                "host wall clock: upload all inputs, execute to terminal, release event, download output"
            }
        }
    }
    fn name(&self) -> &'static str {
        match self {
            Self::LoadUnload => "load_unload",
            Self::Upload => "upload",
            Self::Download => "download",
            Self::Warm => "warm_execution",
            Self::RoundTrip => "round_trip",
        }
    }
}
#[derive(Debug)]
struct Source<'a>(&'a [u8]);
impl ByteSource for Source<'_> {
    fn len(&self) -> u64 {
        self.0.len() as u64
    }
    fn read_at(&self, o: u64, t: &mut [u8]) -> Result<(), BackendError> {
        self.0.read_at(o, t)
    }
    fn as_contiguous(&self) -> Option<&[u8]> {
        Some(self.0)
    }
}
#[derive(Debug)]
struct Sink<'a>(&'a mut [u8]);
impl ByteSink for Sink<'_> {
    fn len(&self) -> u64 {
        self.0.len() as u64
    }
    fn write_at(&mut self, o: u64, s: &[u8]) -> Result<(), BackendError> {
        self.0.write_at(o, s)
    }
    fn as_contiguous_mut(&mut self) -> Option<&mut [u8]> {
        Some(self.0)
    }
}
fn release<T>(r: Result<(), ReleaseFailure<T>>) {
    r.unwrap_or_else(|e| panic!("release failed: {:?}", e.error()));
}
fn number(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .map(|s| s.parse().unwrap_or_else(|_| panic!("invalid {name}")))
        .unwrap_or(default)
}
struct Fixture<A: Accelerator> {
    backend: A,
    context: A::Context,
    program: A::Program,
    queue: A::Queue,
    inputs: Vec<A::Buffer>,
    output: A::Buffer,
    readback: Vec<u8>,
    workload: Workload,
    wait_us: u64,
    timeout: Duration,
}
impl<A: Accelerator> Fixture<A> {
    fn upload(&mut self) {
        for (buffer, data) in self.inputs.iter_mut().zip(&self.workload.inputs) {
            self.backend.write_buffer(buffer, 0, &Source(data)).unwrap();
        }
    }
    fn download(&mut self) {
        self.backend
            .read_buffer(&self.output, 0, &mut Sink(&mut self.readback))
            .unwrap();
    }
    fn execute(&self) {
        assert!(self.inputs.len() <= 3);
        // Fixed scratch avoids allocating a bindings Vec in every timed submission.
        let bindings: [BindingRef<'_, A::Buffer>; 4] = std::array::from_fn(|i| {
            if i < self.inputs.len() {
                BindingRef {
                    slot: i as u32,
                    buffer: &self.inputs[i],
                    range: BufferRange::new(0, self.workload.inputs[i].len() as u64).unwrap(),
                    access: AccessMode::Read,
                }
            } else {
                BindingRef {
                    slot: self.inputs.len() as u32,
                    buffer: &self.output,
                    range: BufferRange::new(0, self.workload.expected.len() as u64).unwrap(),
                    access: AccessMode::Write,
                }
            }
        });
        let event = match self.backend.submit(
            &self.queue,
            &self.program,
            &bindings[..self.inputs.len() + 1],
            Timeout::Infinite,
        ) {
            Ok(e) => e,
            Err(SubmitFailure::Rejected(e)) => panic!("submit rejected: {e:?}"),
            // Drop must preserve backend quarantine rules; do not manually release live buffers.
            Err(SubmitFailure::Indeterminate { error, .. }) => {
                panic!("indeterminate submit: {error:?}")
            }
        };
        let deadline = Instant::now() + self.timeout;
        loop {
            match self.backend.poll_event(&event).unwrap() {
                EventState::Pending => {}
                EventState::Complete => break,
                other => panic!("execution did not complete: {other:?}"),
            }
            assert!(
                Instant::now() < deadline,
                "benchmark timed out; no timing result is valid"
            );
            if self.wait_us == 0 {
                std::hint::spin_loop()
            } else {
                std::thread::sleep(Duration::from_micros(self.wait_us))
            }
        }
        release(self.backend.destroy_event(event));
    }
    fn validate(&mut self) {
        self.download();
        self.workload.validate(&self.readback).unwrap();
    }
    fn close(self) {
        release(self.backend.destroy_queue(self.queue));
        release(self.backend.unload_program(self.program));
        release(self.backend.free_buffer(self.output));
        for b in self.inputs {
            release(self.backend.free_buffer(b));
        }
        release(self.backend.destroy_context(self.context));
    }
}
fn metadata(value: serde_json::Value) {
    if let Some(path) = std::env::var_os("VA_BENCH_METADATA") {
        let mut f = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .unwrap();
        writeln!(f, "{value}").unwrap();
    } else {
        eprintln!("benchmark metadata: {value}");
    }
}
pub fn measure<A: Accelerator + TosaCapabilityProvider>(
    b: Bencher,
    case: Case,
    phase: Phase,
    backend: A,
) {
    let workload = case.build();
    assert!(
        backend
            .tosa_capabilities()
            .iter()
            .any(|c| c.target == workload.target),
        "requested target is not advertised; choose a supported case explicitly"
    );
    let context = backend.create_context(ContextDesc::default()).unwrap();
    let model = parse(&workload.artifact).unwrap();
    let artifact = model.artifact_ref(workload.target, u64::MAX).unwrap();
    let start = Instant::now();
    let program = backend
        .load_program(&context, artifact)
        .expect("requested case must compile; no fallback or skipped measurement");
    let initial_load = start.elapsed();
    let queue = backend
        .create_queue(&context, QueueDesc::default())
        .unwrap();
    let allocate = |len: usize, input: bool| {
        let usage = if input {
            BufferUsage::TRANSFER_DESTINATION | BufferUsage::PROGRAM_INPUT
        } else {
            BufferUsage::TRANSFER_SOURCE | BufferUsage::PROGRAM_OUTPUT
        };
        backend
            .allocate_buffer(
                &context,
                BufferDesc::new(len as u64, 16384, MemoryDomain::Shared, usage).unwrap(),
            )
            .unwrap()
            .into_parts()
            .0
    };
    let inputs = workload
        .inputs
        .iter()
        .map(|v| allocate(v.len(), true))
        .collect();
    let output = allocate(workload.expected.len(), false);
    let readback = vec![0; workload.expected.len()];
    let mut fingerprint = blake3::Hasher::new();
    fingerprint.update(&workload.artifact);
    for input in &workload.inputs {
        fingerprint.update(input);
    }
    let warmups = number("VA_BENCH_WARMUPS", 3);
    let wait_us = number("VA_BENCH_WAIT_US", 0);
    let timeout = Duration::from_secs(number("VA_BENCH_TIMEOUT_SECS", 30));
    assert!(!timeout.is_zero());
    metadata(
        serde_json::json!({"schema":1,"build_target":env!("VA_BENCH_BUILD_TARGET"),
        "build_profile":env!("VA_BENCH_BUILD_PROFILE"),"optimization":env!("VA_BENCH_BUILD_OPT_LEVEL"),
        "case":workload.name,"phase":phase.name(),
        "backend":std::env::var("VA_BENCH_BACKEND").ok(),"requested_device":std::env::var("VA_BENCH_DEVICE").ok(),
        "device_info":format!("{:?}",backend.device_info().unwrap()),"target":format!("{:?}",workload.target),
        "shape":workload.shape,"output_dtype":format!("{:?}",workload.output_dtype),
        "input_bytes":workload.input_bytes(),"output_bytes":workload.expected.len(),
        "logical_flops":workload.logical_flops,"fingerprint":fingerprint.finalize().to_hex().to_string(),
        "warmups":warmups,"poll_wait_us":wait_us,"timeout_seconds":timeout.as_secs(),
        "initial_load_ns":initial_load.as_nanos().to_string(),"boundary":phase.boundary(),
        "placement":"selected backend only; Core ML policy may include CPU; no engine-level timing"}),
    );
    let mut f = Fixture {
        backend,
        context,
        program,
        queue,
        inputs,
        output,
        readback,
        workload,
        wait_us,
        timeout,
    };
    f.upload();
    f.execute();
    f.validate();
    for _ in 0..warmups {
        f.execute();
    }
    match phase {
        Phase::LoadUnload => {
            let model = parse(&f.workload.artifact).unwrap();
            let artifact = model.artifact_ref(f.workload.target, u64::MAX).unwrap();
            b.counter(ItemsCount::new(1u64)).bench_local(|| {
                let p = f.backend.load_program(&f.context, artifact).unwrap();
                release(f.backend.unload_program(p));
            });
        }
        Phase::Upload => {
            let bytes = f.workload.input_bytes();
            b.counter(BytesCount::new(bytes)).bench_local(|| f.upload());
        }
        Phase::Download => {
            let bytes = f.readback.len();
            b.counter(BytesCount::new(bytes))
                .bench_local(|| f.download());
        }
        Phase::Warm => b.counter(ItemsCount::new(1u64)).bench_local(|| f.execute()),
        Phase::RoundTrip => b.counter(ItemsCount::new(1u64)).bench_local(|| {
            f.upload();
            f.execute();
            f.download();
        }),
    }
    f.validate();
    f.close();
}

/// Explicit selection; constructing an unavailable provider or choosing an unsupported case fails.
pub fn selected(b: Bencher, case: Case, phase: Phase) {
    let provider = std::env::var("VA_BENCH_BACKEND").expect("set VA_BENCH_BACKEND (or use run.py)");
    #[allow(unused_variables)]
    let device = std::env::var("VA_BENCH_DEVICE").unwrap_or_default();
    match provider.as_str() {
        #[cfg(va_bench_openvino)]
        "openvino" => {
            assert!(
                !device.is_empty(),
                "select CPU, GPU, NPU or an exact device"
            );
            measure(
                b,
                case,
                phase,
                virtio_accel_openvino::OpenVinoAccelerator::with_device(&device).unwrap(),
            )
        }
        #[cfg(va_bench_coreml)]
        "coreml" => {
            use virtio_accel_coreml::{CoreMlAccelerator, CoreMlComputeUnits};
            let policy = match device.as_str() {
                "cpu-ane" => CoreMlComputeUnits::CpuAndNeuralEngine,
                "cpu-gpu" => CoreMlComputeUnits::CpuAndGpu,
                "all" => CoreMlComputeUnits::All,
                _ => panic!("select cpu-ane, cpu-gpu or all"),
            };
            measure(
                b,
                case,
                phase,
                CoreMlAccelerator::new_tosa_with_compute_units(policy).unwrap(),
            )
        }
        #[cfg(va_bench_vulkan)]
        "vulkan" => {
            assert!(!device.is_empty(), "select an explicit Vulkan device");
            measure(
                b,
                case,
                phase,
                virtio_accel_vulkan::VulkanAccelerator::with_device(&device).unwrap(),
            )
        }
        #[cfg(va_bench_xdna)]
        "xdna" => measure(
            b,
            case,
            phase,
            virtio_accel_xdna::XdnaAccelerator::new().unwrap(),
        ),
        #[cfg(va_bench_hexagon)]
        "hexagon" => measure(
            b,
            case,
            phase,
            virtio_accel_hexagon::HexagonAccelerator::new().unwrap(),
        ),
        _ => {
            let _ = (b, case, phase);
            panic!("unknown or uncompiled backend {provider}; enable its Cargo feature")
        }
    }
}
