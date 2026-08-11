# Security policy

## Reporting a vulnerability

Report suspected vulnerabilities privately to **info@microperceptron.com**. Please do not open a
public GitHub issue for a suspected vulnerability, and please do not disclose details publicly
before we have agreed on a timeline.

A useful report includes:

- the affected crate and version, or commit hash;
- which trust boundary the issue crosses — see [docs/threat-model.md](docs/threat-model.md);
- the bytes, call sequence, or descriptor-chain shape that triggers it, ideally as a failing test or
  a fuzz input; and
- the impact you believe it has.

Reports in any form are welcome. A plain description is worth more than no report at all.

## What to expect

This project is pre-standardization work maintained on a best-effort basis. These are the targets we
hold ourselves to, not a contractual service level:

| Stage | Target |
|---|---|
| Acknowledgement of your report | 3 business days |
| Initial assessment, with a severity and a plan | 10 business days |
| Fix released, or a public advisory explaining why not | 90 days from acknowledgement |

We will tell you if a stage is going to slip. If you have not heard from us within 10 business days,
please follow up — a missed mail is far more likely than a decision to ignore you.

We will credit you in the advisory and release notes unless you ask us not to. There is no bug
bounty.

## Supported versions

| Version | Supported |
|---|---|
| Cargo `0.1.x` | Yes — the current line, and the only one receiving fixes |
| Protocol 1.0 wire ABI | Yes — frozen by the [freeze audit](conformance/v1.0/freeze-audit.md) |

Only the most recent `0.1.x` release receives security fixes. There are no earlier published
releases, and no long-term-support branch. Support for a given line ends when its successor is
published, unless a release note says otherwise.

A fix that changes accepted or emitted protocol bytes is a wire-compatibility event, not only a
Cargo release. It is classified under [docs/release-policy.md](docs/release-policy.md) like any
other protocol change, and may require a new conformance directory even when it is a security fix.

## Scope

In scope, in the crates published from this repository:

- memory-safety or panic reachability from untrusted input — protocol bytes, descriptor-chain
  shapes, or configuration values;
- unsound handling of unknown opcodes, statuses, flags, event states, or reserved bits;
- a device-side path that leaks guest addresses, virtqueue descriptors, or another context's data
  across the boundaries described in [docs/threat-model.md](docs/threat-model.md);
- unbounded resource growth reachable from a bounded guest request; and
- any breach of the `#![forbid(unsafe_code)]` invariant recorded in
  [docs/release-policy.md](docs/release-policy.md); and
- memory-safety, path-containment, direct-backing, or event-publication defects in the audited
  `virtio-accel-coreml` unsafe boundary.

Out of scope:

- `virtio-accel-mock` and `virtio-accel-conformance` used as production components. They are
  test-only reference code with deterministic, non-secret artifacts and scripted faults.
- Reserved-but-unassigned protocol values. A constant that records a reserved number is not
  permission for a device to advertise it; a device that does is out of specification.
- The absence of a claimed Virtio device ID. That is deliberate and documented.
- Vulnerabilities in a downstream VMM, kernel, operating system, or vendor adapter that merely
  depends on these crates. Please report those to the relevant project.

The in-repository Core ML adapter is in scope; it is not considered a downstream vendor adapter.
Its unsafe invariants and ownership scheme are recorded in
[`crates/virtio-accel-coreml/SAFETY.md`](crates/virtio-accel-coreml/SAFETY.md).

## Advisories

Published advisories will appear in this repository's GitHub Security Advisories and, where a
crates.io release is affected, via the RustSec advisory database. Dependency advisories are checked
in CI by cargo-deny; see [docs/portability.md](docs/portability.md).
