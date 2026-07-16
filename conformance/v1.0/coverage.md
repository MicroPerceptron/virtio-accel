# Protocol 1.0 candidate conformance coverage

This matrix records where each normative area is executable today and where implementation-level
behavior is deliberately tracked by a later issue. A tracked issue is the explicit rationale
required by epic #2; it is not an assertion that unimplemented behavior already conforms.

| Normative area | Current executable evidence | Tracked completion rationale |
|---|---|---|
| `specification.md` sections 2-3: scope and layer boundaries | Workspace dependency direction, `no_std` target checks, and the clean-room dependency guard | Concrete transport/provider boundary audits continue in #17, #20, and #21 |
| Section 4: contexts, buffers, programs, queues, submissions, and events | `virtio-accel-core`, object-table, and mock lifecycle tests | Full command-engine ownership behavior is #20; remaining backend semantics are #21 |
| Section 5: mandatory baseline, reserved features, flags, and scalar namespaces | Primary ABI namespace tests plus clean-room request validation and edge vectors | Optional features remain unadvertised; their future policy is #32 |
| Section 6: versions, exact lengths, unknown values, and extension rules | Config/version edge vectors, both codecs, immutable manifests, and layout assertions | Candidate-to-stable freeze and clean-room review are audited again in #33 |
| Sections 7-9: ownership, time, progress, and error truth | Generational-ID, timeout, mock lifecycle, unknown-status, and indeterminate-submit tests | End-to-end recovery and response atomicity are #20; backend policy details are #21 |
| Section 10: portability | Stable/MSRV, Linux/macOS/Windows, Wasm, AArch64, RISC-V, feature, dependency, and documentation CI | Concrete OS/VMM adapters remain intentionally outside portable v1 |
| `wire-abi.md` sections 1-6: config, headers, namespaces, all payloads, statuses, limits, and reserved fields | `layout.json`, `vectors.json`, primary `zerocopy` assertions, and the manual dependency-free codec | Device-state limits that require live objects are exercised by #20 and threat-model issue #25 |
| `wire-abi.md` section 7: preflight and response atomicity | Error-shape vectors and ownership-aware core error types | Writable-region preflight and post-mutation failure are VQ-012/VQ-020 in #18 and #20 |
| `wire-abi.md` sections 8-9: versioned artifacts and change control | Checked-in inputs, drift tests, and coordinated-change documentation | Release freeze governance is finalized by #32 and #33 |
| `virtqueue.md` sections 1-12 | Stable executable case identifiers `VQ-001` through `VQ-020` define the required assertions | Region ports are #17, split-ring model tests are #18, guest compatibility is #19, and full-path behavior is #20 |

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
canonical bytes. No Rust wire structure crosses that boundary.

The clean-room binding uniqueness check is intentionally allocation-free and quadratic. This crate
is portable conformance evidence, not the production command-engine hot path; #20 may use a bounded
set or table consistent with its resource accounting.
