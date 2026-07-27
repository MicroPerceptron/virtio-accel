# Portability and continuous integration

The portable v1 promise is enforced by `.github/workflows/ci.yml`. This file is the source of truth
for required checks; the table below explains why each job exists.

The minimum supported Rust version (MSRV) is 1.85.0. Rust 1.85 is the first release supporting the
Rust 2024 edition selected by the workspace. Every package inherits the same `rust-version`.

## CI matrix

| Job | Toolchain and targets | Contract enforced |
|---|---|---|
| `style-and-api` | Current stable on Ubuntu | Formatting, complete normative-requirement ledger, release-policy invariants, Clippy with warnings denied, and warning-free public docs |
| `native-test` | Current stable on Ubuntu, macOS, and Windows | All workspace unit, integration, target, feature, and documentation tests, runnable examples, and release-profile checking |
| `msrv` | Rust 1.85.0 on Ubuntu | Every workspace target and test continues to compile at the declared MSRV |
| `portable-target` | Stable `aarch64-unknown-none`, `riscv64gc-unknown-none-elf`, and `wasm32-unknown-unknown` | `cleanroom`, `proto`, `transport`, and `core` remain `no_std`; guest, split-queue, device, and facade layers require at most `alloc`; Wasm also checks the std reference crates |
| `feature-sets-and-dependencies` | Stable on Ubuntu | Every Cargo feature combination plus dependency and `std`/`alloc` leakage guards for the portable codecs, queue ports, and core |
| `dependency-policy` | Cargo-deny with Rust 1.85.0 | Advisories, yanked crates, duplicate versions, wildcard requirements, licenses, and dependency sources |
| `fuzz-smoke` | Pinned nightly on Ubuntu | Shared harness tests plus bounded smoke iterations over generated seed corpora and committed minimized regressions for every fuzz target |

The native jobs intentionally use GitHub-hosted `*-latest` images so runner security and supported
host versions advance without changing the project’s semantic target list. Release evidence records
the concrete runner versions used for the release.

## Crate portability tiers

| Tier | Crates | Allowed runtime surface |
|---|---|---|
| `core-only` | `virtio-accel-cleanroom`, `virtio-accel-proto`, `virtio-accel-transport`, `virtio-accel-core` | `core`; the clean-room codec and transport ports have no normal/build dependencies, while proc-macros used by other crates may execute with `std` on the build host |
| `alloc-portable` | `virtio-accel-guest`, `virtio-accel-split-queue`, `virtio-accel-device`, `virtio-accel` | `core + alloc`; no OS, filesystem, sockets, threads, or host synchronization |
| `std-reference` | `virtio-accel-mock`, `virtio-accel-conformance` | Portable `std`; no host-OS or vendor-specific API |

Concrete VMM, kernel, OS, and vendor adapters are outside the portable-v1 milestone and must not
become default dependencies of a portable crate.

## Cargo feature policy

CI runs `cargo hack check --feature-powerset --no-dev-deps` across the workspace. Features must be
additive: disabling default features may remove convenience behavior but must not select a different
protocol interpretation.

The portable dependency guard inspects normal and build target features for
`virtio-accel-cleanroom`, `virtio-accel-proto`, `virtio-accel-transport`, and
`virtio-accel-core`, `virtio-accel-guest`, and `virtio-accel-split-queue`; test-only development dependencies are
intentionally outside the target runtime graph. It additionally proves that the clean-room codec
and transport ports have no normal or build dependencies at all. A dependency's host-side derive
macro may use `std`, but the target graph for these crates must not enable a dependency feature
named `std` or `alloc`.

## Dependency policy

`deny.toml` permits only crates.io dependencies and the workspace’s path dependencies. It rejects:

- known advisories and yanked releases;
- unmaintained direct workspace dependencies;
- unsound dependencies;
- multiple resolved versions of the same crate;
- wildcard version requirements; and
- licenses outside Apache-2.0, BSD-2-Clause, MIT, and Unicode-3.0.

GitHub Actions are pinned to immutable commit SHAs with their human-readable release versions kept
in comments. Standalone CI tools are pinned too: `cargo-hack` 0.6.45, `cargo-fuzz` 0.13.2 on
`nightly-2026-07-13`, and `cargo-deny` 0.20.2 (bundled by the pinned cargo-deny action).

## Local verification

The host-independent checks can be reproduced with:

```sh
cargo fmt --all -- --check
python3 ci/check-normative-requirements.py --check
python3 ci/check-release-policy.py
cargo clippy --workspace --all-targets --all-features -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --all-features --no-deps
cargo test --workspace --all-targets --all-features
cargo test -p virtio-accel-device --test state_model
cargo test --workspace --doc --all-features
cargo run --example backend_conformance
cargo run --example reference_execution
cargo +1.85.0 check --workspace --all-targets --all-features
cargo hack check --workspace --feature-powerset --no-dev-deps
bash ci/check-portable-dependencies.sh
cargo deny --all-features check
```

Deeper deterministic state-model exploration can be run manually with:

```sh
VIRTIO_ACCEL_STATE_MODEL_SEED=9e3779b97f4a7c15 cargo test -p virtio-accel-device --test state_model deep_generated_object_graphs_match_the_reference_model -- --ignored
```

Fuzz smoke coverage can be reproduced with:

```sh
cargo test --manifest-path fuzz/Cargo.toml --lib --no-default-features
python3 ci/seed-fuzz-corpus.py
cargo fuzz run protocol_decode fuzz/corpus/protocol_decode fuzz/regressions/protocol_decode -- -runs=256 -max_total_time=20 -timeout=5 -rss_limit_mb=2048 -max_len=65536
cargo fuzz run descriptor_end_to_end fuzz/corpus/descriptor_end_to_end fuzz/regressions/descriptor_end_to_end -- -runs=256 -max_total_time=20 -timeout=5 -rss_limit_mb=2048 -max_len=65536
cargo fuzz run stateful_commands fuzz/corpus/stateful_commands fuzz/regressions/stateful_commands -- -runs=256 -max_total_time=20 -timeout=5 -rss_limit_mb=2048 -max_len=65536
cargo fuzz run guest_client fuzz/corpus/guest_client fuzz/regressions/guest_client -- -runs=256 -max_total_time=20 -timeout=5 -rss_limit_mb=2048 -max_len=65536
```

Target checks require the corresponding Rust standard libraries:

```sh
rustup target add aarch64-unknown-none riscv64gc-unknown-none-elf wasm32-unknown-unknown
cargo check \
  -p virtio-accel-cleanroom \
  -p virtio-accel-proto \
  -p virtio-accel-transport \
  -p virtio-accel-core \
  -p virtio-accel-device \
  -p virtio-accel-guest \
  -p virtio-accel-split-queue \
  -p virtio-accel \
  --target aarch64-unknown-none \
  --no-default-features
```
