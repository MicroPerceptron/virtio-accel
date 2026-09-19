# Performance and memory budgets

The portable v1 performance posture is explicit before the API freezes. The checked-in budget
artifact is [`performance-budgets.json`](../conformance/v1.0/performance-budgets.json), and the
baseline metadata is
[`performance-baseline.json`](../conformance/v1.0/performance-baseline.json).

The default CI budget is deterministic. It checks complexity classes, allocation boundaries,
copy-path counters, and representative hot-path byte reads. Wall-clock timings are useful release
evidence, but they are not stable enough for ordinary pull-request gating across hosted runners.

| Operation area | Expected cost | Allocation profile | Copy boundary |
|---|---:|---|---|
| Config and scalar request decode | `O(1)` | none | fixed scalar bytes only |
| Non-`SUBMIT` request decode | `O(1)` plus descriptor validation | none | transfer and artifact tails stay borrowed |
| `SUBMIT` decode | `O(b log b)` | one bounded metadata vector after binding-count validation | binding metadata only |
| Segmented byte-port access | portable worst case `O(s + n)`; indexed split queue `O(log s + k + n)` | none per access | exact caller-requested range |
| Object lookup | `O(1)` | none | none |
| Command dispatch | request-specific | bounded object-table reservation before mutation | response publication, except explicit transfers |
| Submission admission | `O(b log b)` plus lookups; canonical binding revalidation is `O(b)` | bounded event dependency and binding metadata | no hidden buffer staging |
| Polling | `O(1)` | none | event-state response only |
| Reset | object graph walk | releases existing state; no new guest-count allocation | none |
| TOSA parse + target semantics | `O(f + g log s + c)` | bounded borrowed-name/symbol/control-flow metadata after FlatBuffer verification | no graph, string, or constant-data copy |
| TOSA lowering analysis | `O(g log g)` | compact dense IDs, spans, topological/liveness metadata, and runtime obligations | borrowed graph and constant payloads remain in place |
| TOSA dynamic specialization | `O(d)` plus exact-key cache lookup | caller-bounded key and LRU entries | dynamic CTC bytes only; ordinary tensor inputs are not scanned |

`b` is a validated binding count, `s` is segment count (or the largest TOSA symbol table in the
TOSA row), `k` is the number of descriptor segments touched by one logical byte access, `n` is
explicitly requested bytes, `f` is verified FlatBuffer structure, `g` is the number of graph objects
and edges, and `c` is compile-time-constant data inspected by the semantic pass.

## Copy accounting

The baseline content-copy boundaries are `Accelerator::write_buffer` and
`Accelerator::read_buffer`. They report explicit transfer bytes separately from provider staging.
Submission binds the exact provider allocation. If a provider stages a direct-binding buffer through
a hidden bounce allocation during submission, the conformance diagnostics case fails.

The `ConformanceHooks::submission_path_diagnostics` hook reports cumulative direct, shared/imported,
staged-direct, staged-byte, and explicit-transfer counters. Providers that cannot report these
counters skip the diagnostics case, but release evidence should include them for any hardware
adapter claiming v1 performance conformance.

## Budget exceptions

The portable decoder keeps one bounded `DecodedBinding` vector for `SUBMIT` duplicate-slot
validation. The command engine also owns bounded event dependency and binding metadata while
admitting a submission. These allocations are deliberately after guest count validation and contain
metadata only, never program-buffer contents.

The decoder's slot sort is also the canonical handoff to core admission. Core and guest validation
recognize strictly increasing slot order in `O(b)` without allocation. Their public APIs continue to
accept arbitrary binding order through an allocation-free fallback, so this optimization does not
make ordering semantic.

Split-queue chain construction records bounded logical descriptor spans alongside the flattened
regions. Each later byte access binary-searches the first touched span instead of rescanning from
descriptor zero. This metadata is allocated only while the driver owns and constructs the bounded
chain; queue publication, command decoding, completion, and reset remain allocation-free.

Device admission validates each resolved buffer descriptor in place instead of retaining a parallel
descriptor vector. The provider-facing binding vector and event-owned buffer dependency vector remain
necessary, but descriptor validation adds no per-submission allocation.

The current v1 budget treats those metadata allocations as acceptable. It does not permit an
allocation sized by an unvalidated guest count and does not permit full-range program-buffer copies
outside explicit transfer calls.

## Local checks

```sh
python3 ci/check-performance-budgets.py --check
cargo test --test performance_budgets --all-features
cargo test -p virtio-accel-conformance --all-features
```

## TOSA artifact evidence

`virtio-accel-tosa` first runs the official FlatBuffers verifier with explicit depth, table-count,
apparent-size, and input-byte limits. Its one structural pass stores borrowed `&str` keys in bounded,
fallibly reserved vectors, sorts them once per scope, and uses binary search for reference lookup.
It never creates an owned graph and never copies names, tensor bytes, shape data, or appended
constant buffers. Returned views read the already-verified buffer in place.

`Model::validate_for` then walks those borrowed views without constructing an owned IR. It keeps
fallibly reserved symbol and control-flow bookkeeping, validates bounded compile-time constants in
place, and performs rank-bounded shape arithmetic. No tensor or shape payload is copied.

`Model::analyze_for` amortizes provider lowering work at program load: every name lookup becomes a
dense ID/span access, topological order and liveness are retained, dead/layout/constant-folding
opportunities are marked conservatively, and runtime `ERROR_IF` work is separated from advisory
per-element `REQUIRE` conditions. Dynamic CTC validation is allocation-free after the caller has
assembled its sorted borrowed value list. Specialization keys are caller-bounded and collision-safe;
the portable LRU uses exact words after its fingerprint and caps retained compiled variants.

The parser's default graph counts and byte ceilings are finite and callers can lower every one via
`Limits`. Tests compare the input and returned buffer pointers, exercise caller-selected ceilings,
parse a `flatc`-encoded upstream stable graph, and traverse all public views. The `tosa_parse` fuzz
target mutates that upstream seed, cross-checks traversal counts and constant bytes against the
validation statistics, materializes every safe attribute view, and runs both a fully enabled and a
minimal Level 8K semantic target to exercise rejection paths.

## Core ML provider evidence

`virtio-accel-coreml` builds a sorted slot/access plan at model load. Warm submission reuses the
queue's native-binding array, resolves arbitrary binding order against that plan, and performs one
`O(b log b)` retained-allocation deduplication before admission. The event keeps that one backing
vector directly, avoiding the previous second vector allocation/conversion. Submission copies no
tensor bytes. Read-only allocations may be shared by overlapping predictions; any output or
read-write use remains exclusive.

The crate includes an ignored release-mode measurement for fixed provider overhead:

```sh
cargo test --release -p virtio-accel-coreml \
  measures_warm_submission_and_completion_latency -- --ignored --nocapture
```

On an Apple M4 running macOS 26.5.2, five runs of 200 measured iterations after 20 warmups reported
per-run median admission between 5.00 and 5.46 microseconds and median completion between 95.42 and
103.29 microseconds for the embedded `Float32[8]` model. The pre-pass measurement was 5.25
microseconds admission and 98.46 microseconds completion. The optimization therefore removes
submission allocation/scan work without claiming a timing improvement below the noise floor of this
micro-model. This is evidence for host and Core ML fixed overhead, not representative ANE
throughput, and remains non-gating wall-clock data.

## OpenVINO provider evidence

`virtio-accel-openvino` builds the sorted slot/access/shape plan once at program load. Warm
submission reuses the queue's pointer-slot storage plus one empty high-water vector allocation for
backing guards and one for tensor/check metadata. Each spare is cleared before the queue can retain
it; therefore reuse removes Rust metadata allocations without retaining buffer pointers, backing
guards, tensor handles, or native requests. Concurrent events remain supported: when the one spare
is occupied, another event allocates independently, and completion retains at most the larger
returned allocation.

The native infer request remains event-owned and is created for every submission. Pooling it would
be an unsafe optimization because OpenVINO copies bound tensor objects into the request and its C
API has no reset operation that detaches all input and output tensors. Submission still copies no
tensor bytes.

The crate includes an ignored release-mode measurement that reports admission separately from
submit-to-complete latency:

```sh
cargo test --release -p virtio-accel-openvino \
  measures_warm_submission_and_completion_latency -- --ignored --nocapture
```

Wall-clock results must be recorded on a pinned OpenVINO runtime and identified device before a
timing claim is made; the deterministic regression tests instead pin capacity reuse, pointer
scrubbing, guard release at terminal observation, and tensor-metadata release after request
destruction.

## AMD XDNA provider evidence

`virtio-accel-xdna` compiles each admitted TOSA shape once at program load and stores the resulting
precompiled artifact in a content-addressed cache. Warm FP8 submission binds the caller's FP8 input
and BF16 output allocations directly. The conversion streams fixed 1,024-element tiles through one
AIE2P worker; no host conversion, submission-time bounce copy, or tensor-sized Rust allocation is
part of the warm path.

The crate includes an ignored release-mode scaling measurement matching the OpenVINO structure:

```sh
source ~/toolchains/amdxdna-hrx-v2026.08/env.sh
export VIRTIO_ACCEL_AMDXDNA_TOOLCHAIN=~/toolchains/amdxdna-hrx-v2026.08
cargo test --release -p virtio-accel-xdna --test hardware \
  measures_fp8_cast_scaling_on_one_aie_worker -- --ignored --nocapture --test-threads=1
```

On August 25, 2026, a `1022:17f0` XDNA2 NPU with the v2026.08 HRX/aiecc toolchain produced the
following E4M3-to-BF16 results. Each shape was loaded once, warmed up for 20 submissions, and then
measured for 200 sequential submissions. Effective I/O counts one FP8 input byte plus two BF16
output bytes per element.

| Elements | Admission median / p95 | Submit-to-complete median / p95 | Effective I/O | Diagnostics |
|---:|---:|---:|---:|---|
| 1,024 | 0.692 / 1.513 µs | 0.086 / 0.113 ms | 0.033 GiB/s | 440 direct bindings; 0 explicit bytes |
| 16,384 | 0.631 / 1.513 µs | 0.486 / 0.512 ms | 0.094 GiB/s | 440 direct bindings; 0 explicit bytes |
| 262,144 | 1.072 / 5.080 µs | 6.743 / 6.790 ms | 0.109 GiB/s | 440 direct bindings; 0 explicit bytes |
| 1,048,576 | 2.585 / 8.526 µs | 26.656 / 26.816 ms | 0.110 GiB/s | 440 direct bindings; 0 explicit bytes |

The benchmark validates every output against the exact FP8 oracle after timing. The near-constant
large-shape rate documents the current single-worker envelope without claiming it is the final
throughput configuration. Multi-worker striping is an optional optimization; deterministic CI
continues to gate exact numerics, direct binding, and zero submission-time transfer bytes instead
of wall-clock latency.

The exact INT8 MATMUL benchmark uses the same 20 warmups and 200 measured submissions. Shapes on
the native 8x8x8 INT8 MMUL grid (M % 16, K % 8 with K >= 16, N % 16) stream through DMA-side
micro-tile layout transforms into the fork's vectorized i8/i32 `mm` kernel on raw INT8 values,
followed by an exact zero-point correction pass (`C = R - zb*rowsum(A) - za*colsum(B) + K*za*zb`,
every term provably inside INT32); no core cycle widens or repacks an operand. Off-grid shapes
retain the scalar exact kernel. Run it with:

```sh
source ~/toolchains/amdxdna-hrx-v2026.08/env.sh
export VIRTIO_ACCEL_AMDXDNA_TOOLCHAIN=~/toolchains/amdxdna-hrx-v2026.08
cargo test --release -p virtio-accel-xdna --test hardware \
  measures_exact_int8_matmul_latency -- --ignored --nocapture --test-threads=1
```

On August 27, 2026, the same `1022:17f0` XDNA2 NPU and v2026.08 toolchain measured the 64x64x32
specialization at 0.851 microseconds admission median, 72.089 microseconds submit-to-complete
median, and 99.427 microseconds p95, or 3.636 effective GOPS — 4.7x the August 26 widening-kernel
baseline (334.065 microseconds, 0.785 GOPS), with every output still matching the shared exact
oracle. All 660 bindings across 220 submissions were direct and submission reported zero
explicit-transfer bytes. Disassembly of the retired kernel attributed ~99% of its time to scalar
zero-point widening and lane-by-lane packing around a single `vmac`; the correction-term
formulation removed that work entirely, and the submit-to-complete median now sits at the measured
per-submission overhead floor (the 1,024-element FP8 case measures 86 microseconds), so the
remaining latency is submission-path cost, not kernel cost. Issue #151 tracks the next steps
(submission overlap, worker striping) without weakening exactness or direct binding.

The pipelined-throughput benchmark keeps four submissions in flight over four rotating buffer
sets (`measures_pipelined_int8_matmul_throughput`, same shape and oracle). On August 27, 2026 it
measured 73.3-74.9 microseconds amortized per inference across three 400-completion runs --
statistically identical to the sequential submit-to-complete median (69.3-74.6 microseconds
across three runs of the latency benchmark on the same worker). A batched-flush variant (all
in-flight dispatches submitted under one `hrx_stream_flush`) measured 70.1 microseconds, also
identical. The conclusion this evidence supports: the per-submission floor is per-command
driver/firmware round-trip cost inside one hardware context, and neither deeper host-side
pipelining nor flush batching moves it. Raising effective throughput therefore requires more work
per dispatch (larger admitted envelopes, worker striping -- issue #151 steps 5-6) or parallel
hardware contexts (issue #121), not further submission-path restructuring. The ring depth of four
still pays for itself in semantics: submissions overlap with host-side polling and readback, and
completion waits no longer serialize against `allocate_buffer`.

## Vulkan provider evidence status

`virtio-accel-vulkan`'s warm path is a descriptor update plus `vkQueueSubmit` on a preallocated
bounded ring of (command buffer, fence, descriptor set) slots — no worker thread, no
submission-time staging; completion is a nonblocking `vkGetFenceStatus` poll (ADR 0006). Programs
compile once at `load_program` (checked-in SPIR-V + specialization constants) and are charged
against `ArtifactRef::resident_bytes`.

Manual hardware commands (per the #75 precedent — no self-hosted runner on a public repo):

```sh
# Real GPU (any Vulkan 1.3 compute device; pin the ICD explicitly):
VIRTIO_ACCEL_VULKAN_REQUIRE_DEVICE=1 \
  cargo test -p virtio-accel-vulkan --test vulkan -- --nocapture

# Software ICD rehearsal (the CI lane's shape):
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json \
VIRTIO_ACCEL_VULKAN_REQUIRE_DEVICE=1 \
  cargo test -p virtio-accel-vulkan --test vulkan -- --nocapture
```

Verified driver stacks for the FP32 operator tier (ADR 0007): Intel Arc 140V (Lunar Lake, Mesa 26.0.8 ANV, Vulkan 1.4.335)
(full suite, 2026-09-06, alongside the same host's llvmpipe LLVM 21.1.8), Mesa lavapipe in the
`vulkan-lavapipe-test` CI lane, and Apple M4 via MoltenVK 1.4.2 (full suite, 2026-09-17; local
validation only, not a CI lane). On ANV and llvmpipe alike the crate-authored
transcendentals measured 1 ulp (sin, cos, tanh) and 2 ulp (erf) worst case against binary64 over
4096 samples spanning ±8000, ±1e6, `f32::MAX`, and the non-finite edges — the same numbers, which
is what the `NoContraction` and software-reduction policy exists to guarantee. Copy-path diagnostics across all three: every submission is a direct binding;
`explicit_transfer_bytes` stays zero for `Host` and `Shared` domains, and `Device` staging is
confined to `write_buffer`/`read_buffer` as the memory-domain contract requires.

The FP16 tier (ADR 0008) shares the FP32 submission path exactly — the same dispatch geometry,
arena, and ring; only the element storage is packed two per word — so no separate timing claims
are made. Its corpus has executed end-to-end on Apple M4 via MoltenVK, Intel Arc LNL (Mesa ANV),
and AMD Radeon 860M (RADV) (2026-09-17), and the lavapipe CI lane exercises the tier on every
change.

The FP32 operator tier (ADR 0007) adds the structural optimizations a real graph needs before any
timing is worth publishing: a whole graph is one command buffer with barriers only between
dependent dispatches; constants and intermediates live in one device-local arena per program with
lifetime-packed regions, `RESHAPE`/`IDENTITY` views instead of copies, and dead operators elided;
`MATMUL` is a register-tiled shared-memory kernel (a 64 × 64 block per workgroup, ADR 0010) or,
for eight rows or fewer, a barrier-free split-k streaming kernel (ADR 0011), both with fused
multiply-add and a stated error bound rather than bit-identity to the sequential sum; pipelines are
created against a per-instance `VkPipelineCache`; every 1-D kernel is a grid-stride loop so
dispatch counts stay inside `maxComputeWorkGroupCount` at any tensor size. Known costs, recorded so
they are measured rather than assumed: predicate (`BOOL`) elementwise outputs, strided sub-word
moves, and `MATMUL`'s FP16 result are written with two atomics per element, and `SIN`/`COS`
evaluate both range reductions and select.

Warm-latency numbers in the XDNA structure (load once, warm 20, measure 200) are not yet
published; the throughput benchmark below reports the submission floor it measures alongside
each case, and the broadened tier still owes a MoltenVK run (the same commands above).

### Vulkan FP8 tier throughput (ADR 0010)

`cargo bench -p virtio-accel-vulkan` times whole TOSA graphs submit-to-fence through the public
`Accelerator` surface, per enumerated device, after a warm-up of at least 300 ms per case so a
frequency-scaling GPU is at clock. On 2026-09-18, Intel Arc (Panther Lake, Mesa 26.0.8 ANV,
Vulkan 1.4.335), `Device` memory domain, median of 30 timed submissions, 16 Mi elements for the
elementwise cases; "before" is the kernels as shipped by ADR 0009 under the same harness:

| Case | Before | After |
|---|---:|---:|
| `IDENTITY` FP8 | 4.60 ms, 7.3 GB/s | 0.37 ms, 92 GB/s |
| `IDENTITY` FP16 | 1.70 ms, 39 GB/s | 0.65 ms, 103 GB/s |
| `IDENTITY` FP32 | 1.22 ms, 110 GB/s | 1.23 ms, 109 GB/s |
| `CAST` FP8 → FP16 | 1.67 ms, 30 GB/s | 0.55 ms, 91 GB/s |
| `CAST` FP16 → FP8 | 3.27 ms, 15 GB/s | 0.52 ms, 97 GB/s |
| `CAST` FP32 → FP8 | 3.27 ms, 26 GB/s | 0.80 ms, 104 GB/s |
| `CAST` FP32 → FP16 | 1.88 ms, 54 GB/s | 0.95 ms, 106 GB/s |
| `MATMUL` FP8 → FP16, 1024³ | 3.76 ms, 572 GFLOP/s | 1.43 ms, 1503 GFLOP/s |
| `MATMUL` FP16, 1024³ | 2.86 ms, 751 GFLOP/s | 1.14 ms, 1882 GFLOP/s |
| `MATMUL` FP32, 1024³ | 2.74 ms, 785 GFLOP/s | 1.11 ms, 1928 GFLOP/s |
| `CAST` FP8 → FP16 + `MATMUL` FP16, 1024³ | 3.24 ms | 1.28 ms |
| GEMV FP8 → FP16, 1 × 4096 × 4096 | 0.98 ms, 17 GB/s of weights | 0.44 ms, 39 GB/s of weights |
| GEMV FP16, 1 × 4096 × 4096 | 1.06 ms, 32 GB/s of weights | 0.53 ms, 63 GB/s of weights |
| GEMV FP32, 1 × 4096 × 4096 | 1.06 ms, 63 GB/s of weights | 0.72 ms, 93 GB/s of weights |
| `CAST` FP8 → FP16 + GEMV FP16, 1 × 4096 × 4096 | 2.73 ms | 0.95 ms |

GB/s counts bytes read plus written; the GEMV weight figures count the weight matrix alone. The
submission floor (a four-element identity) measured 100–170 µs across runs and is included in
every number. What the table says about the FP8 tier: data movement now runs at the device's copy
rate, so the quarter-width storage delivers its bandwidth, and GEMV now orders the right way
(FP8 0.44 ms, FP16 0.53, FP32 0.72). Net of the floor the FP8 GEMV streams weights at roughly half
the rate the FP32 kernel shows the memory system delivers; the remainder is per-step fixed cost,
recorded in ADR 0010 as the next objective. The 1024³ cases vary about ±15% run to run on this
device even at 30 samples.

## Qualcomm Hexagon evidence status

`virtio-accel-hexagon` includes an ignored release-mode measurement for fixed submission overhead:

```powershell
cargo test --release -p virtio-accel-hexagon --test hexagon `
  measures_warm_submission_and_completion_latency -- --ignored --nocapture --test-threads=1
```

On August 17, 2026, a Snapdragon X126100 Hexagon HTP v73 with NPU driver `30.0.222.0`, Windows
Balanced power mode, QAIRT `2.49.0.260730`, provider build `v2.49.0.260730134355`, QNN core API
`2.38.0`, and HTP backend API `5.49.0` produced the following single-run results. Each graph was
loaded once, warmed up for 20 submissions, and then measured for 200 sequential submissions.

| Graph | Dtype | Admission median / p95 | Submit-to-complete median / p95 | Diagnostics |
|---|---|---:|---:|---|
| identity, 8 elements | FP16 | 27.6 / 61.5 µs | 2.8098 / 3.0682 ms | 440 direct bindings; 0 explicit submission bytes |
| identity, 8 elements | INT8 | 23.1 / 58.9 µs | 2.8465 / 3.0333 ms | 440 direct bindings; 0 explicit submission bytes |

The counts cover two exact caller-owned bindings for all 20 warmups and 200 samples. The input was
initialized before counters were sampled; no read or write occurred during measured submission.
These are fixed-overhead micro-model results, not throughput claims or representative large-model
latency. Ordinary CI gates correctness and copy diagnostics rather than wall-clock values.
