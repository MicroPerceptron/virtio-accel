# Protocol 1.0 conformance artifacts

This directory contains implementation-independent inputs for the portable protocol 1.0 candidate.

- [`layout.json`](layout.json) records protocol constants and every structure size, byte alignment,
  and field offset.
- [`vectors.json`](vectors.json) contains canonical hexadecimal bytes for the device configuration,
  all 15 request opcodes, every success and command-specific response shape, all event states, and
  reviewed malformed/unknown boundary cases.
- [`scenarios.json`](scenarios.json) contains deterministic end-to-end command, status, and used-
  length traces for lifecycle, scheduling, pressure, notification, recovery, timeout, and
  device-loss behavior.
- [`coverage.md`](coverage.md) maps the normative document areas to executable evidence or an
  explicitly tracked implementation issue.
- [`requirements.json`](requirements.json) catalogs every normative keyword occurrence with a
  content-derived ID, source line, executable evidence, tracked issues, and rationale.
- [`../rust-clean-room`](../rust-clean-room) contains a dependency-free `no_std` Rust codec that
  implements the byte contract manually without importing `virtio-accel-proto` or its wire types.

The files are deliberately plain JSON with hexadecimal byte strings so implementations do not need
Rust tooling to consume them.

Ordinary tests parse these checked-in files as inputs. They do not regenerate them. An intentional
candidate revision must update the normative specification, Rust layout assertions, manifest, and
vectors in one reviewed change. After the final freeze audit, incompatible changes require a new
versioned directory.

The primary ABI and clean-room codec independently decode and encode every canonical frame. The
primary crate's bridge test runs both implementations over the same bytes and compares their raw
headers and exact output. A separate semantic interoperability test constructs every distinct
request and response layout in each implementation, crosses only bytes, and checks every decoded
field in the other implementation. CI also enforces that the clean-room codec remains
dependency-free.

`ci/check-normative-requirements.py --check` reconstructs the requirement ledger from the normative
Markdown. CI fails if a requirement is added, removed, moved, or reworded without regenerating and
reviewing its exact evidence/rationale entry.
