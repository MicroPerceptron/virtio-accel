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
