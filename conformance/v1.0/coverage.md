# Protocol 1.0 candidate conformance coverage

This matrix summarizes where each normative area is executable today and where implementation-level
behavior is deliberately tracked by a later issue. The exact per-keyword ledger is
[`requirements.json`](requirements.json), mechanically checked by
`ci/check-normative-requirements.py`. A tracked issue is the explicit rationale required by epic
#2; it is not an assertion that unimplemented behavior already conforms.

| Normative area | Current executable evidence | Tracked completion rationale |
|---|---|---|
| `specification.md` sections 2-3: scope and layer boundaries | Workspace dependency direction, `no_std` target checks, the clean-room dependency guard, the audited provider trait, and the full reference guest-to-backend path | Concrete platform adapters remain intentionally outside portable v1 |
| Section 4: contexts, buffers, programs, queues, submissions, events, and resource policy | `virtio-accel-core`, the reusable transport-free backend suite, construction-time metadata validation, object-table lifecycle and retained-byte tests, deterministic mock execution with verifiable buffer output, and full serialized split-queue scenarios | Provider-specific artifact fixtures remain implementation inputs |
| Section 5: mandatory baseline, reserved features, flags, and scalar namespaces | Primary ABI namespace tests plus clean-room request validation and edge vectors | Optional features remain unadvertised; their future policy is #32 |
| Section 6: versions, exact lengths, unknown values, and extension rules | Config/version edge vectors, both codecs, immutable manifests, and layout assertions | Candidate-to-stable freeze and clean-room review are audited again in #33 |
| Sections 7-9: ownership, time, progress, and error truth | Per-method provider contracts plus the standard backend suite, broken-backend controls, generational-ID, timeout, deterministic fault injection, ownership accounting, indeterminate-submit, end-to-end recovery, and response-atomicity tests | Quantitative hot-path evidence is #29 |
| Section 10: portability | Stable/MSRV, Linux/macOS/Windows, Wasm, AArch64, RISC-V, feature, dependency, and documentation CI | Concrete OS/VMM adapters remain intentionally outside portable v1 |
| `wire-abi.md` sections 1-6: config, headers, namespaces, all payloads, statuses, limits, and reserved fields | `layout.json`, `vectors.json`, primary `zerocopy` assertions, the manual dependency-free codec, live-object end-to-end traces with exact request and response bytes, and aggregate retained-byte policy tests | Every wire dimension and provider-retained bulk-storage class has an authoritative finite bound |
| `wire-abi.md` section 7: preflight and response atomicity | Error-shape vectors, ownership-aware core error types, explicit transfer-failure contracts, reusable backend admission cases, end-to-end short-completion rejection, and scripted pre/post-mutation faults | Further adversarial sequencing is #27 |
| `wire-abi.md` sections 8-9: versioned artifacts and change control | Checked-in inputs, drift tests, and coordinated-change documentation | Release freeze governance is finalized by #32 and #33 |
| `virtqueue.md` sections 1-12 | Stable executable cases and `scenarios.json` drive the full reference guest, split queue, device loop, command engine, and backend through segmented chains, out-of-order completion, notification suppression, backpressure, short responses, timeout, cancellation, reset, and device loss while comparing descriptor regions, used lengths, response bytes, and replay controls | Concrete VMM adapter integration remains intentionally outside portable v1 |

## Independent implementation evidence

The two Rust implementations share only the normative documents and checked-in byte artifacts:

1. `virtio-accel-proto` uses explicit packed Rust wire structures and `zerocopy`.
2. `virtio-accel-cleanroom` uses manual slice indexing and little-endian conversion, defines its own
   semantic types, has no normal/build dependencies, and is never a production dependency.

The clean-room suite:

- decodes and re-encodes both configurations, all 15 request opcodes, and all 20 canonical response
  shapes byte-for-byte;
- validates every binding rather than copying an opaque submission tail;
- independently classifies every reviewed adversarial vector;
- preserves unknown response statuses as opaque failures;
- rejects unknown event states with recovery required; and
- tests zero IDs, integer overflow, duplicate binding slots, response correlation, reserved values,
  and invalid event-state combinations.

The bridge test in `virtio-accel-proto` decodes each frame with both implementations, compares the
independently derived header semantics, and requires the manual encoder to reproduce the exact
canonical bytes. `crates/virtio-accel-proto/tests/semantic_interop.rs` additionally constructs every
distinct config, request, and response layout in each implementation and checks every semantic
field after the other implementation decodes the bytes. No Rust wire structure crosses either
boundary.

The clean-room binding uniqueness check is intentionally allocation-free and quadratic. This crate
is portable conformance evidence, not the production command-engine hot path; the command processor
uses one bounded decoded-binding allocation and sorts it in place.
