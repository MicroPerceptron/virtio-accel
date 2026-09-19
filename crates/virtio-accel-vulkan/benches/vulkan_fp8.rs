//! Throughput benchmarks for the Vulkan backend's FP8 tier and the FP16/FP32 paths it is
//! measured against.
//!
//! Every case is a whole TOSA graph submitted through the public `Accelerator` surface, timed
//! from `submit` to a terminal `poll_event`, so the number is the one a guest sees: host
//! recording, queue submission, kernel execution, and fence completion. Cases are grouped by
//! what they stress:
//!
//! - **floor**: a four-element identity, the per-submission cost every other number contains.
//! - **move**: `IDENTITY` at FP8, FP16 and FP32 storage — the raw copy path, whose byte and half
//!   lanes are stored through the neighbour-safe atomic sequence.
//! - **cast**: every FP8 direction of `CAST` plus the FP16/FP32 pair, the dequantization path.
//! - **matmul**: square `MATMUL` at FP8 (native: widened inline in the kernel), FP16 and FP32,
//!   and the two-dispatch dequantize-then-multiply graphs (`CAST` FP8 → FP16/FP32 followed by
//!   the wider `MATMUL`) an FP8-weight graph takes when it keeps FP16 activations.
//! - **gemv**: the decode shape — one or eight rows against a 4096 × 4096 weight matrix — at
//!   the same three storages plus both dequantize paths. This is the shape FP8 weights exist
//!   for, and it is bandwidth-bound: the number that matters is bytes of weight per second.
//!
//! Runs against every enumerated device (`VIRTIO_ACCEL_VULKAN_BENCH_DEVICE=<substring>` pins
//! one), in the `Device` memory domain when the device advertises it and `Host` otherwise.
//! `VIRTIO_ACCEL_VULKAN_BENCH_ITERS` (default 10) sets the timed submissions per case after two
//! warm-ups; `VIRTIO_ACCEL_VULKAN_BENCH_QUICK=1` runs the small sizes only, for a software ICD.
//! Output is a Markdown table per device. Without a Vulkan loader or device the bench exits 0.
//!
//! This is a plain `harness = false` binary rather than a Criterion bench: the workspace bans
//! duplicate dependency versions and gates licenses, Criterion's tree would breach both, and a
//! GPU submission is timed by its fence, not by a sampling loop.

fn main() {
    #[cfg(va_vulkan)]
    bench::run();
    #[cfg(not(va_vulkan))]
    eprintln!("virtio-accel-vulkan was built as the placeholder; nothing to benchmark");
}

#[cfg(va_vulkan)]
mod bench {
    use std::time::{Duration, Instant};

    use virtio_accel_core::{
        Accelerator, AccessMode, BackendError, BindingRef, BufferDesc, BufferRange, BufferUsage,
        ByteSink, ByteSource, ContextDesc, EventState, MemoryDomain, QueueDesc, SubmitFailure,
        Timeout,
    };
    use virtio_accel_tosa::{DType, parse};
    use virtio_accel_tosa_build::{OperatorKind, OwnedGraph, OwnedOperator, OwnedTensor};
    use virtio_accel_vulkan::{
        InitError, REQUIRED_RESIDENT_BYTES, VULKAN_TOSA_FP8_TARGET, VULKAN_TOSA_TARGET,
        VulkanAccelerator,
    };

    const BUFFER_ALIGNMENT: u64 = 4096;
    const WARMUPS: usize = 2;

    /// A contiguous host source: the backend's `write_buffer` takes a trait object.
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

    #[derive(Debug)]
    struct VecSink<'a>(&'a mut Vec<u8>);

    impl ByteSink for VecSink<'_> {
        fn len(&self) -> u64 {
            self.0.len() as u64
        }

        fn write_at(&mut self, offset: u64, source: &[u8]) -> Result<(), BackendError> {
            ByteSink::write_at(self.0.as_mut_slice(), offset, source)
        }

        fn as_contiguous_mut(&mut self) -> Option<&mut [u8]> {
            Some(self.0)
        }
    }

    /// What a case moves or computes, for the derived throughput column.
    #[derive(Clone, Copy)]
    enum Metric {
        /// Bytes read plus bytes written: reported as GB/s.
        Bytes(u64),
        /// Multiply-adds times two: reported as GFLOP/s, with the weight bytes it streamed.
        Flops { flops: u64, weight_bytes: u64 },
    }

    struct Case {
        group: &'static str,
        name: String,
        artifact: Vec<u8>,
        target: virtio_accel_tosa::Target,
        inputs: Vec<Vec<u8>>,
        output_len: usize,
        metric: Metric,
    }

    struct Sample {
        median: Duration,
        min: Duration,
        dispatches: usize,
    }

    fn env_flag(name: &str) -> bool {
        std::env::var(name).is_ok_and(|value| value == "1")
    }

    pub fn run() {
        let devices = match VulkanAccelerator::available_devices() {
            Ok(devices) if !devices.is_empty() => devices,
            Ok(_) | Err(InitError::RuntimeUnavailable | InitError::DeviceUnavailable) => {
                eprintln!("no Vulkan 1.3 compute device is available on this host; skipped");
                return;
            }
            Err(error) => panic!("device enumeration failed: {error}"),
        };
        let filter = std::env::var("VIRTIO_ACCEL_VULKAN_BENCH_DEVICE").ok();
        let iterations: usize = std::env::var("VIRTIO_ACCEL_VULKAN_BENCH_ITERS")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(10);
        let quick = env_flag("VIRTIO_ACCEL_VULKAN_BENCH_QUICK");
        let cases = cases(quick);
        for device in devices {
            if filter
                .as_ref()
                .is_some_and(|filter| !device.contains(filter.as_str()))
            {
                continue;
            }
            let backend = VulkanAccelerator::with_device(&device)
                .unwrap_or_else(|error| panic!("{device}: initialization failed: {error}"));
            let info = backend.device_info().unwrap();
            let domain = if info
                .capabilities
                .supports_memory_domain(MemoryDomain::Device)
            {
                MemoryDomain::Device
            } else {
                MemoryDomain::Host
            };
            println!("\n### {device} ({domain:?} domain, {iterations} timed submissions)\n");
            println!("| group | case | dispatches | median | min | throughput |");
            println!("|---|---|---:|---:|---:|---:|");
            for case in &cases {
                let sample = time_case(&backend, case, domain, iterations);
                println!(
                    "| {} | {} | {} | {} | {} | {} |",
                    case.group,
                    case.name,
                    sample.dispatches,
                    format_duration(sample.median),
                    format_duration(sample.min),
                    format_throughput(case.metric, sample.median),
                );
            }
        }
    }

    fn format_duration(duration: Duration) -> String {
        let micros = duration.as_secs_f64() * 1e6;
        if micros >= 1000.0 {
            format!("{:.3} ms", micros / 1000.0)
        } else {
            format!("{micros:.1} µs")
        }
    }

    fn format_throughput(metric: Metric, duration: Duration) -> String {
        let seconds = duration.as_secs_f64();
        match metric {
            Metric::Bytes(bytes) => format!("{:.2} GB/s", bytes as f64 / seconds / 1e9),
            Metric::Flops {
                flops,
                weight_bytes,
            } => format!(
                "{:.1} GFLOP/s, {:.2} GB/s weights",
                flops as f64 / seconds / 1e9,
                weight_bytes as f64 / seconds / 1e9
            ),
        }
    }

    // -- cases -----------------------------------------------------------------------------------

    fn cases(quick: bool) -> Vec<Case> {
        let mut cases = Vec::new();
        cases.push(identity_case("floor", DType::FP32, 4));

        let elements: u32 = if quick { 1 << 20 } else { 1 << 24 };
        for dtype in [DType::FP8E4M3, DType::FP8E5M2, DType::FP16, DType::FP32] {
            cases.push(identity_case("move", dtype, elements));
        }
        for (from, to) in [
            (DType::FP8E4M3, DType::FP16),
            (DType::FP8E4M3, DType::FP32),
            (DType::FP8E5M2, DType::FP16),
            (DType::FP16, DType::FP8E4M3),
            (DType::FP32, DType::FP8E4M3),
            (DType::FP16, DType::FP8E5M2),
            (DType::FP16, DType::FP32),
            (DType::FP32, DType::FP16),
        ] {
            cases.push(cast_case(from, to, elements));
        }

        let sides: &[u32] = if quick { &[256] } else { &[256, 512, 1024] };
        for &side in sides {
            for dtype in [DType::FP8E4M3, DType::FP16, DType::FP32] {
                cases.push(matmul_case("matmul", dtype, 1, side, side, side));
            }
            cases.push(matmul_case("matmul", DType::FP8E5M2, 1, side, side, side));
            cases.push(dequant_matmul_case(
                "matmul",
                DType::FP16,
                1,
                side,
                side,
                side,
            ));
            cases.push(dequant_matmul_case(
                "matmul",
                DType::FP32,
                1,
                side,
                side,
                side,
            ));
        }

        let (k, n) = if quick { (1024, 1024) } else { (4096, 4096) };
        for m in [1, 8] {
            for dtype in [DType::FP8E4M3, DType::FP16, DType::FP32] {
                cases.push(matmul_case("gemv", dtype, 1, m, k, n));
            }
            cases.push(dequant_matmul_case("gemv", DType::FP16, 1, m, k, n));
            cases.push(dequant_matmul_case("gemv", DType::FP32, 1, m, k, n));
        }
        cases
    }

    fn dtype_name(dtype: DType) -> &'static str {
        match dtype {
            DType::FP8E4M3 => "fp8e4m3",
            DType::FP8E5M2 => "fp8e5m2",
            DType::FP16 => "fp16",
            DType::FP32 => "fp32",
            _ => unreachable!("not a float dtype the bench uses"),
        }
    }

    fn element_bytes(dtype: DType) -> u64 {
        match dtype {
            DType::FP8E4M3 | DType::FP8E5M2 => 1,
            DType::FP16 => 2,
            DType::FP32 => 4,
            _ => unreachable!("not a float dtype the bench uses"),
        }
    }

    fn is_fp8(dtype: DType) -> bool {
        matches!(dtype, DType::FP8E4M3 | DType::FP8E5M2)
    }

    /// The target an artifact over `dtypes` needs: the FP8 target whenever any FP8 tensor
    /// appears, the base float target otherwise.
    fn target_for(dtypes: &[DType]) -> virtio_accel_tosa::Target {
        if dtypes.iter().any(|dtype| is_fp8(*dtype)) {
            VULKAN_TOSA_FP8_TARGET
        } else {
            VULKAN_TOSA_TARGET
        }
    }

    /// Finite, well-inside-range pseudo-random tensor bytes for `dtype`: FP8 patterns stay below
    /// each encoding's top exponent, binary16 and binary32 values sit in `[-1, 1)`.
    fn finite_bytes(dtype: DType, elements: u64, seed: u32) -> Vec<u8> {
        let mut state = seed.wrapping_mul(0x9e37_79b9) | 1;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        };
        let count = usize::try_from(elements).unwrap();
        match dtype {
            DType::FP8E4M3 | DType::FP8E5M2 => (0..count)
                .map(|_| {
                    let word = next();
                    // Exponent field below its maximum: never NaN or infinity in either format.
                    let bits = (word & 0x7f) as u8;
                    let bits = if bits >= 0x78 { bits - 0x40 } else { bits };
                    bits | ((word >> 8) & 0x80) as u8
                })
                .collect(),
            DType::FP16 => (0..count)
                .flat_map(|_| {
                    let unit = (next() >> 8) as f32 / (1u32 << 24) as f32;
                    virtio_accel_vulkan::shader::f32_to_f16_bits(unit * 2.0 - 1.0).to_le_bytes()
                })
                .collect(),
            DType::FP32 => (0..count)
                .flat_map(|_| {
                    let unit = (next() >> 8) as f32 / (1u32 << 24) as f32;
                    (unit * 2.0 - 1.0).to_le_bytes()
                })
                .collect(),
            _ => unreachable!("not a float dtype the bench uses"),
        }
    }

    fn identity_case(group: &'static str, dtype: DType, elements: u32) -> Case {
        let shape = vec![1, 1, elements as i32];
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
        let target = target_for(&[dtype]);
        let bytes = u64::from(elements) * element_bytes(dtype);
        Case {
            group,
            name: format!("identity {} × {}", dtype_name(dtype), elements),
            artifact: graph.build(target).unwrap(),
            target,
            inputs: vec![finite_bytes(dtype, u64::from(elements), 1)],
            output_len: bytes as usize,
            metric: Metric::Bytes(2 * bytes),
        }
    }

    fn cast_case(from: DType, to: DType, elements: u32) -> Case {
        let shape = vec![1, 1, elements as i32];
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
        // `CAST` is admitted only under the FP8 target (its envelope is a superset of the base
        // float target's), whatever the pair of dtypes.
        let target = VULKAN_TOSA_FP8_TARGET;
        let in_bytes = u64::from(elements) * element_bytes(from);
        let out_bytes = u64::from(elements) * element_bytes(to);
        Case {
            group: "cast",
            name: format!("{} → {} × {}", dtype_name(from), dtype_name(to), elements),
            artifact: graph.build(target).unwrap(),
            target,
            inputs: vec![finite_bytes(from, u64::from(elements), 2)],
            output_len: out_bytes as usize,
            metric: Metric::Bytes(in_bytes + out_bytes),
        }
    }

    fn push_zero_points(graph: &mut OwnedGraph<'_>, dtype: DType) {
        let zero = vec![0_u8; element_bytes(dtype) as usize];
        graph
            .push_tensor(OwnedTensor::constant("a_zp", vec![1], dtype, zero.clone()))
            .push_tensor(OwnedTensor::constant("b_zp", vec![1], dtype, zero))
            .push_operator(OwnedOperator::new(
                OperatorKind::Const,
                vec![],
                vec!["a_zp".into()],
            ))
            .push_operator(OwnedOperator::new(
                OperatorKind::Const,
                vec![],
                vec!["b_zp".into()],
            ));
    }

    /// `[batch, m, k] × [batch, k, n]` at `dtype`; FP8 operands produce FP16 as TOSA defines.
    fn matmul_case(group: &'static str, dtype: DType, batch: u32, m: u32, k: u32, n: u32) -> Case {
        let result = if is_fp8(dtype) { DType::FP16 } else { dtype };
        let mut graph = OwnedGraph::new("main");
        graph
            .push_tensor(OwnedTensor::new(
                "a",
                vec![batch as i32, m as i32, k as i32],
                dtype,
            ))
            .push_tensor(OwnedTensor::new(
                "b",
                vec![batch as i32, k as i32, n as i32],
                dtype,
            ))
            .push_tensor(OwnedTensor::new(
                "y",
                vec![batch as i32, m as i32, n as i32],
                result,
            ));
        push_zero_points(&mut graph, dtype);
        graph
            .push_operator(OwnedOperator::new(
                OperatorKind::MatMul,
                vec!["a".into(), "b".into(), "a_zp".into(), "b_zp".into()],
                vec!["y".into()],
            ))
            .push_input("a")
            .push_input("b")
            .push_output("y");
        let target = target_for(&[dtype]);
        let (batch, m, k, n) = (u64::from(batch), u64::from(m), u64::from(k), u64::from(n));
        Case {
            group,
            name: format!("matmul {} {m}×{k}×{n}", dtype_name(dtype)),
            artifact: graph.build(target).unwrap(),
            target,
            inputs: vec![
                finite_bytes(dtype, batch * m * k, 3),
                finite_bytes(dtype, batch * k * n, 4),
            ],
            output_len: (batch * m * n * element_bytes(result)) as usize,
            metric: Metric::Flops {
                flops: 2 * batch * m * k * n,
                weight_bytes: batch * k * n * element_bytes(dtype),
            },
        }
    }

    /// The dequantize path: `wide` activations `[batch, m, k]` against FP8 E4M3 weights
    /// `[batch, k, n]` cast to `wide` in the graph, then a `wide` MATMUL. Two dispatches with a
    /// barrier and an arena intermediate the size of the widened weights.
    fn dequant_matmul_case(
        group: &'static str,
        wide: DType,
        batch: u32,
        m: u32,
        k: u32,
        n: u32,
    ) -> Case {
        let weight = DType::FP8E4M3;
        let mut graph = OwnedGraph::new("main");
        graph
            .push_tensor(OwnedTensor::new(
                "a",
                vec![batch as i32, m as i32, k as i32],
                wide,
            ))
            .push_tensor(OwnedTensor::new(
                "w8",
                vec![batch as i32, k as i32, n as i32],
                weight,
            ))
            .push_tensor(OwnedTensor::new(
                "w",
                vec![batch as i32, k as i32, n as i32],
                wide,
            ))
            .push_tensor(OwnedTensor::new(
                "y",
                vec![batch as i32, m as i32, n as i32],
                wide,
            ));
        push_zero_points(&mut graph, wide);
        graph
            .push_operator(OwnedOperator::new(
                OperatorKind::Cast,
                vec!["w8".into()],
                vec!["w".into()],
            ))
            .push_operator(OwnedOperator::new(
                OperatorKind::MatMul,
                vec!["a".into(), "w".into(), "a_zp".into(), "b_zp".into()],
                vec!["y".into()],
            ))
            .push_input("a")
            .push_input("w8")
            .push_output("y");
        let target = VULKAN_TOSA_FP8_TARGET;
        let (batch, m, k, n) = (u64::from(batch), u64::from(m), u64::from(k), u64::from(n));
        Case {
            group,
            name: format!(
                "cast fp8e4m3 → {} + matmul {} {m}×{k}×{n}",
                dtype_name(wide),
                dtype_name(wide)
            ),
            artifact: graph.build(target).unwrap(),
            target,
            inputs: vec![
                finite_bytes(wide, batch * m * k, 5),
                finite_bytes(weight, batch * k * n, 6),
            ],
            output_len: (batch * m * n * element_bytes(wide)) as usize,
            metric: Metric::Flops {
                flops: 2 * batch * m * k * n,
                weight_bytes: batch * k * n,
            },
        }
    }

    // -- execution -------------------------------------------------------------------------------

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

    fn time_case(
        backend: &VulkanAccelerator,
        case: &Case,
        domain: MemoryDomain,
        iterations: usize,
    ) -> Sample {
        let device = backend.device_name();
        let context = backend.create_context(ContextDesc::default()).unwrap();
        let model = parse(&case.artifact).unwrap();
        let artifact = model
            .artifact_ref(case.target, REQUIRED_RESIDENT_BYTES)
            .unwrap();
        let program = backend
            .load_program(&context, artifact)
            .unwrap_or_else(|error| panic!("{device}: {} failed to load: {error:?}", case.name));
        let mut inputs = Vec::with_capacity(case.inputs.len());
        for bytes in &case.inputs {
            let mut buffer = allocate(
                backend,
                &context,
                bytes.len() as u64,
                domain,
                BufferUsage::TRANSFER_DESTINATION | BufferUsage::PROGRAM_INPUT,
            );
            backend
                .write_buffer(&mut buffer, 0, &SliceSource(bytes))
                .unwrap();
            inputs.push(buffer);
        }
        let output = allocate(
            backend,
            &context,
            case.output_len as u64,
            domain,
            BufferUsage::TRANSFER_SOURCE | BufferUsage::PROGRAM_OUTPUT,
        );
        let queue = backend
            .create_queue(&context, QueueDesc::default())
            .unwrap();
        let mut bindings: Vec<BindingRef<'_, _>> = inputs
            .iter()
            .zip(&case.inputs)
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
            range: BufferRange::new(0, case.output_len as u64).unwrap(),
            access: AccessMode::Write,
        });

        let mut durations = Vec::with_capacity(iterations);
        for round in 0..WARMUPS + iterations {
            let start = Instant::now();
            let event = backend
                .submit(&queue, &program, &bindings, Timeout::Infinite)
                .unwrap_or_else(|failure| match failure {
                    SubmitFailure::Rejected(error) => {
                        panic!("{device}: {} rejected: {error:?}", case.name)
                    }
                    SubmitFailure::Indeterminate { error, .. } => {
                        panic!("{device}: {} indeterminate: {error:?}", case.name)
                    }
                });
            loop {
                match backend.poll_event(&event).unwrap() {
                    EventState::Pending => std::hint::spin_loop(),
                    EventState::Complete => break,
                    terminal => panic!("{device}: {} ended {terminal:?}", case.name),
                }
            }
            let elapsed = start.elapsed();
            backend.destroy_event(event).unwrap_or_else(|failure| {
                panic!("{device}: event release failed: {:?}", failure.error())
            });
            if round >= WARMUPS {
                durations.push(elapsed);
            }
        }
        durations.sort();
        let sample = Sample {
            median: durations[durations.len() / 2],
            min: durations[0],
            dispatches: program.dispatch_count(),
        };

        // A guest reads the result back; do it once so a kernel that never wrote is at least
        // noticed as all-zero output on a device that zero-fills (not asserted: correctness is
        // the test suite's job).
        let mut sink = vec![0_u8; case.output_len];
        backend
            .read_buffer(&output, 0, &mut VecSink(&mut sink))
            .unwrap();

        backend.destroy_queue(queue).map_err(|f| f.error()).unwrap();
        backend
            .unload_program(program)
            .map_err(|f| f.error())
            .unwrap();
        backend.free_buffer(output).map_err(|f| f.error()).unwrap();
        for buffer in inputs {
            backend.free_buffer(buffer).map_err(|f| f.error()).unwrap();
        }
        backend
            .destroy_context(context)
            .map_err(|f| f.error())
            .unwrap();
        sample
    }
}
