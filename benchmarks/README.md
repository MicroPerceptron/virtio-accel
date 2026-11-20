# Backend benchmarks

This unpublished, independent Rust workspace uses [Divan](https://docs.rs/divan/0.1.21/divan/)
for sampling, timer calibration, batching and throughput counters, following Axiom's CPU-kernel
benchmark approach. Dependencies stay outside the portable library workspace and published crates.
It replaces the Core ML/OpenVINO ignored integration-test timing loops; correctness tests remain ordinary tests.

## Run

From the repository root, with Rust 1.85+ and Python 3.9+:

```sh
python3 benchmarks/run.py --backend portable
python3 benchmarks/run.py --backend coreml --device cpu-gpu --case Nvfp4F16Decode
python3 benchmarks/run.py --backend coreml --device cpu-ane --case Fp8CacheWindow
python3 benchmarks/run.py --backend coreml --device all --case Nvfp4F32Decode
python3 benchmarks/run.py --backend openvino --device NPU --case Nvfp4F16Decode
python3 benchmarks/run.py --backend openvino --device GPU --case Fp8E4M3Encode
python3 benchmarks/run.py --backend vulkan --device "exact enumerated device name" --case MatmulF32Decode
python3 benchmarks/run.py --backend xdna --case MatmulBf16Decode
python3 benchmarks/run.py --backend hexagon --case MatmulF16Decode
```

The same wrapper works in PowerShell (`python` in place of `python3`). Install the selected
backend's runtime/SDK first: [Core ML](../crates/virtio-accel-coreml/README.md),
[OpenVINO](../crates/virtio-accel-openvino/README.md),
[Vulkan](../crates/virtio-accel-vulkan/README.md),
[XDNA](../crates/virtio-accel-xdna/README.md), or
[Hexagon](../crates/virtio-accel-hexagon/README.md).
The selected provider must actually be compiled and available. Unsupported devices, targets,
models, failed events, wrong output and empty hardware filters fail the run. Nothing silently
substitutes a different provider or records a skipped workload as a result.

Each command creates a new `benchmarks/results/<UTC>-<backend>/` directory. Use `--output PATH`
for a new explicit directory and `--runtime-version VERSION` to record the vendor runtime version
(the label is operator supplied, not discovered automatically). Existing directories are rejected.

```sh
# Exercise every timing boundary once; this validates the harness, not performance.
python3 benchmarks/run.py --backend openvino --device CPU --case Nvfp4F16Decode --phase all -- --test
# Let Divan batch and sample a selected case; all arguments after -- go to Divan.
python3 benchmarks/run.py --backend coreml --device all --case Nvfp4F16Prefill -- --sample-count 100
```

For native Divan filtering/listing, invoke Cargo directly:

```sh
cargo bench --locked --manifest-path benchmarks/Cargo.toml --bench hardware --features coreml -- --list
```

Set `VA_BENCH_BACKEND`, `VA_BENCH_DEVICE` and optionally `VA_BENCH_METADATA` before executing
hardware cases directly. Do not run the entire case list blindly: support varies by backend.
The wrapper defaults to FP16 matmul, FP32 for Vulkan, or BF16 for XDNA.

## Workloads

| Case | Shape and contract |
|---|---|
| `MatmulF16Decode`, `MatmulF32Decode`, `MatmulBf16Decode` | M=1, N=128, K=256; BF16 inputs accumulate to FP32 |
| `MatmulF16Prefill` | M=16, N=128, K=256 |
| `Nvfp4F16Decode`, `Nvfp4F32Decode` | M=1, N=128, K=256; packed E2M1 weights, E4M3 block scales, tensor scale |
| `Nvfp4F16Prefill` | M=16, N=128, K=256 |
| `Fp8E4M3Encode`, `Fp8E5M2Encode` | 4096 FP32 elements converted to FP8 |
| `Fp8E4M3Decode`, `Fp8E5M2Decode` | 4096 FP8 elements converted to FP32 |
| `Fp8CacheWindow` | Append 128 bytes to a 16384-byte FP8 cache, then slice a 16384-byte window |

These are small deterministic fixtures, not whole-model token-rate or attention benchmarks.
FP16 NVFP4 is the initial Intel NPU path; FP32 boundary cases and FP8 conversions target
OpenVINO CPU/GPU and Core ML. Vulkan, XDNA and Hexagon retain their own advertised capability
limits; enabling their benchmark feature does not add quantized support to those providers.
Packed host input does not imply native FP4 arithmetic or compressed intermediate storage.

Every hardware fixture executes and compares exact output bytes against its reference before
measurement and after measurement. Binary-fraction inputs make the projection reference stable;
FP8 cache movement checks all byte patterns. Zero-filled output cannot pass. Validation, fixture
construction, context/queue/buffer creation and initial compilation are outside the samples.

## Timing boundaries

| `--phase` | Included in each Divan iteration |
|---|---|
| `load_unload` | Load and unload one program in an existing context; runtime/filesystem/compiler caches retained |
| `upload` | Write all inputs, **including weights**, to existing backend buffers |
| `download` | Read the output into an existing host allocation |
| `warm_execution` (default) | Prepare bindings, submit, poll to terminal completion and release the event; buffers/program resident |
| `round_trip` | Upload all inputs, warm execution, download output |
| `all` | Each of the five phases separately |

The initial load duration is also recorded once in metadata; it is **not** a cold-compilation
statistical result. `load_unload` retains the fixture's initial resident program, so it measures
additional program lifecycle cost in that context. Transfer counters count API payload bytes,
not measured PCIe/DRAM traffic; shared-memory providers may perform a host copy. Very short
transfers need Divan's automatic batching, not forced one-iteration samples.

These are serial host wall-clock API measurements, including backend/runtime overhead. Core ML
policies allow CPU participation and do not prove GPU/ANE placement. No device timestamp,
engine utilization, native low-bit instruction throughput, concurrency scaling or VM transport
cost is inferred from these results. The portable target measures graph authoring and
parse/analyze only.

`VA_BENCH_WARMUPS` defaults to 3 untimed executions after the initial correctness execution.
`VA_BENCH_WAIT_US=0` spins while waiting; a positive value sleeps that many microseconds between
polls. The wait strategy changes latency and CPU usage and must match across comparisons.
`VA_BENCH_TIMEOUT_SECS` defaults to 30 for asynchronous completion polling; it cannot interrupt
a vendor call that blocks internally. A timeout invalidates the run.

## Reproducibility and CI

Keep all three result files together:

- `report.txt`: native Divan statistics and build/runtime diagnostics. Divan owns the estimator;
  the wrapper does not parse table text into invented JSON samples or compute percentiles.
- `run.json`: exact invocation, exit status, UTC times, commit/dirty state and tracked-diff hash,
  lockfile hash, Rust/Cargo, OS/architecture, selected runtime environment and operator version.
- `cases.jsonl`: hardware identity, compiled target/profile/optimization, workload/input fingerprint,
  shape/dtype/bytes/logical work, actual buffer properties/alignment, phase boundary, warmups and polling settings.

Only exit code zero is a successful run. Metadata alone is not evidence that execution succeeded.
Use a clean committed checkout for publishable comparisons; the tracked-diff hash does not capture
untracked source files. Record power mode, temperature, competing workloads and driver versions
alongside results. Compare matching fixtures, compiler/runtime versions, placement policy and
polling settings on the same hardware. Repeat runs and report variation; do not rank hardware
from a single smoke run. Divan 0.1.21 has no supported structured sample exporter, so the preserved
native report is the statistical record.

CI formats/lints/checks the package on Linux, macOS and Windows, checks Rust 1.85, runs reference
and capture tests, and executes portable and pinned OpenVINO CPU smoke cases. It uploads result
bundles for inspection. Shared hosted runners are correctness gates, **not performance gates**.
Dedicated hardware measurements remain an explicit operator action.
