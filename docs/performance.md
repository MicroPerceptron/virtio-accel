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

`b` is a validated binding count, `s` is segment count, and `n` is explicitly requested bytes.

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
