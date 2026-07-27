# Protocol 1.0 freeze audit

Status: complete for the portable v1.0 baseline.

This audit freezes the protocol 1.0 assigned constants, exact wire layouts, canonical golden bytes,
scenario traces, normative requirement ledger, and performance budgets in this directory. It also
records the Rust API, portability, dependency, unsafe-code, security, and performance review used
to decide that the remaining optional work is explicitly deferred rather than silently missing.

## Audit result

| Area | Result | Evidence |
|---|---|---|
| Normative requirements | Pass | `requirements.json`; `ci/check-normative-requirements.py --check`; `coverage.md` |
| Wire layouts and constants | Frozen for v1.0 | `layout.json`; `vectors.json`; `wire-abi.md`; primary `zerocopy` layout assertions |
| Golden request/response bytes | Frozen for v1.0 | `vectors.json`; `tests/cleanroom_vectors.rs`; `tests/wire_abi_conformance.rs`; `crates/virtio-accel-proto/tests/semantic_interop.rs` |
| Virtqueue behavior | Pass | `virtqueue.md`; `crates/virtio-accel-split-queue`; `tests/portable_end_to_end.rs`; `scenarios.json` |
| State machines and ownership | Pass | `docs/architecture.md`; `crates/virtio-accel-device/src/state.rs`; `crates/virtio-accel-device/tests/command_processor.rs`; `tests/portable_end_to_end.rs` |
| Reset and recovery | Pass | `docs/threat-model.md`; command-processor reset tests; reset and device-loss scenarios |
| Error mapping | Pass | `wire-abi.md` section 6; clean-room malformed vectors; command-processor failure tests |
| Public Rust APIs | Pass | `docs/public-api.md`; rustdoc with warnings denied; backend implementer guide; runnable examples |
| Semver and evolution policy | Pass | `docs/release-policy.md`; `docs/releases/v1.0.md`; post-freeze wire procedure in `wire-abi.md` |
| Portability | Pass | `docs/portability.md`; `.github/workflows/ci.yml`; `ci/check-portable-dependencies.sh` |
| Unsafe code | Pass | every crate root and `fuzz/src/lib.rs` forbids unsafe code; `ci/check-release-policy.py` |
| Dependencies and licenses | Pass | `deny.toml`; `ci/check-release-policy.py`; dependency-policy CI job |
| Security and finite resource bounds | Pass | `docs/threat-model.md`; resource-policy tests; state-model tests; fuzz smoke |
| Performance and copy boundaries | Pass | `docs/performance.md`; `performance-budgets.json`; `performance-baseline.json`; `ci/check-performance-budgets.py --check`; performance budget tests |
| Independent review implementation | Pass | dependency-free `conformance/rust-clean-room` codec; byte-only bridge/interoperability tests |

## Frozen protocol artifacts

The following files are versioned protocol 1.0 inputs:

- `layout.json`;
- `vectors.json`;
- `scenarios.json`;
- `requirements.json`;
- `performance-budgets.json`;
- `performance-baseline.json`;
- `coverage.md`; and
- this audit result.

Post-freeze edits to assigned values, field offsets, sizes, canonical bytes, scenario observations,
or ownership semantics are not ordinary patches. They must be classified by
`docs/release-policy.md` as an erratum, compatible protocol-minor extension, or protocol-major
change before merge.

## Checklist

| Check | Result |
|---|---|
| All blocking implementation and verification issues for the portable v1 baseline are closed by this audit PR. | Pass |
| Every normative requirement has evidence or a documented non-runtime classification. | Pass |
| Unknown-field, exact-length, unknown-opcode, unknown-status, and feature-negotiation rules match the normative specification. | Pass |
| Protocol constants and golden vectors are frozen for v1.0. | Pass |
| Deferred optional features are unadvertised and documented as out of scope. | Pass |
| Public Rust APIs preserve ownership, blocking, allocation, copy-boundary, and recovery contracts. | Pass |
| Platform adapters do not leak into portable default dependencies. | Pass |
| Unsafe code is forbidden across current crates and fuzz support code. | Pass |
| Dependency, license, publish metadata, and MSRV policy are documented and checked. | Pass |
| At least one independent implementation exercise validates spec clarity. | Pass |

## Deferred optional features

These features are outside the frozen protocol 1.0 baseline and must remain unadvertised:

- multi-queue transport behavior;
- event queues and unsolicited completion publication;
- external-memory import/export, DMA-BUF, Windows shared handles, and other platform handles;
- timeline fences and cache-coherency protocols;
- secure contexts;
- packed virtqueues;
- concrete KVM, vhost-user, VFIO, QEMU, Windows, macOS, Linux kernel, or vendor SDK adapters; and
- standardized graph IR, compiler, or executable formats.

The reserved constants and bits keep namespace space only. They are invalid until a future
conformance directory assigns semantics.

## Required release commands

The local release review should run the host-independent command set from `docs/portability.md`,
including:

```sh
python3 ci/check-release-policy.py
python3 ci/check-normative-requirements.py --check
python3 ci/check-performance-budgets.py --check
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --all-features --no-deps
cargo test --workspace --all-targets --all-features
cargo test --workspace --doc --all-features
cargo run --example backend_conformance
cargo run --example reference_execution
```

Target, feature-powerset, dependency-policy, and fuzz-smoke checks remain part of CI and release
evidence even when they require installed targets or tools that are not present on every developer
machine.
