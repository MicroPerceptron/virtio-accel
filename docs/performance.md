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
| Segmented byte-port access | `O(s + n)` | none | exact caller-requested range |
| Object lookup | `O(1)` | none | none |
| Command dispatch | request-specific | bounded object-table reservation before mutation | response publication, except explicit transfers |
| Submission admission | `O(b log b)` plus lookups | bounded event dependency and binding metadata | no hidden buffer staging |
| Polling | `O(1)` | none | event-state response only |
| Reset | object graph walk | releases existing state; no new guest-count allocation | none |
| TOSA parse + target semantics | `O(f + g log s + c)` | bounded borrowed-name/symbol/control-flow metadata after FlatBuffer verification | no graph, string, or constant-data copy |
| TOSA lowering analysis | `O(g log g)` | compact dense IDs, spans, topological/liveness metadata, and runtime obligations | borrowed graph and constant payloads remain in place |
| TOSA dynamic specialization | `O(d)` plus exact-key cache lookup | caller-bounded key and LRU entries | dynamic CTC bytes only; ordinary tensor inputs are not scanned |

`b` is a validated binding count, `s` is segment count (or the largest TOSA symbol table in the
TOSA row), `n` is explicitly requested bytes, `f` is verified FlatBuffer structure, and `g` is the
number of graph objects and edges, and `c` is compile-time-constant data inspected by the semantic
pass.

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
