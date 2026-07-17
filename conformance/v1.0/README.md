# Protocol 1.0 conformance artifacts

This directory contains implementation-independent inputs for the portable protocol 1.0 candidate.

- [`layout.json`](layout.json) records protocol constants and every structure size, byte alignment,
  and field offset.
- [`vectors.json`](vectors.json) contains canonical hexadecimal bytes for the device configuration,
  all 15 request opcodes, every success and command-specific response shape, all event states, and
  reviewed malformed/unknown boundary cases.
- [`scenarios.json`](scenarios.json) contains replayable end-to-end traces with exact request bytes,
  published response bytes, used lengths, flattened descriptor-region layouts, and portable replay
  controls for lifecycle, scheduling, pressure, notification, malformed completion, recovery,
  timeout, reset, and device-loss behavior.
- [`coverage.md`](coverage.md) maps the normative document areas to executable evidence or an
  explicitly tracked implementation issue.
- [`performance-budgets.json`](performance-budgets.json) and
  [`performance-baseline.json`](performance-baseline.json) record the v1 hot-path complexity,
  allocation, copy-boundary, diagnostic, and baseline evidence.
- [`requirements.json`](requirements.json) catalogs every normative keyword occurrence with a
  content-derived ID, source line, executable evidence, tracked issues, and rationale.
- [`../rust-clean-room`](../rust-clean-room) contains a dependency-free `no_std` Rust codec that
  implements the byte contract manually without importing `virtio-accel-proto` or its wire types.
- [`../../crates/virtio-accel-conformance`](../../crates/virtio-accel-conformance) runs the semantic
  backend contract without importing wire, virtqueue, OS, or vendor types; its provider adapter is
  documented in the [backend implementer guide](../../docs/backend-implementer-guide.md).

The files are deliberately plain JSON with hexadecimal byte strings and explicit control metadata
so implementations do not need Rust tooling or test-internal harness types to consume them.

Ordinary tests parse these checked-in files as inputs. They do not regenerate them. An intentional
candidate revision must update the normative specification, Rust layout assertions, manifest, and
vectors in one reviewed change. After the final freeze audit, incompatible changes require a new
versioned directory.

The primary ABI and clean-room codec independently decode and encode every canonical frame. The
primary crate's bridge test runs both implementations over the same bytes and compares their raw
headers and exact output. A separate semantic interoperability test constructs every distinct
request and response layout in each implementation, crosses only bytes, and checks every decoded
field in the other implementation. The portable end-to-end suite also records the full guest/device
trace and compares it with the checked-in scenario corpus, including negative differential checks
that reject intentionally corrupted status, used length, descriptor-region, request-byte,
response-byte, and completion-order observations. CI also enforces that the clean-room codec remains
dependency-free.

`ci/check-normative-requirements.py --check` reconstructs the requirement ledger from the normative
Markdown. CI fails if a requirement is added, removed, moved, or reworded without regenerating and
reviewing its exact evidence/rationale entry.

`ci/check-performance-budgets.py --check` validates the performance budget and baseline manifests.
The root `performance_budgets` test also proves that bulk artifact tails are not read during decode
and that an unvalidated `SUBMIT` binding count is rejected before the binding tail is touched.
