# Portability and continuous integration

The portable v1 promise is enforced by `.github/workflows/ci.yml`. This file is the source of truth
for required checks; the table below explains why each job exists.

The minimum supported Rust version (MSRV) is 1.85.0. Rust 1.85 is the first release supporting the
Rust 2024 edition selected by the workspace. Every package inherits the same `rust-version`.

## CI matrix

| Job | Toolchain and targets | Contract enforced |
|---|---|---|
| `style-and-api` | Current stable on Ubuntu | Formatting, Clippy with warnings denied, and warning-free public docs |
| `native-test` | Current stable on Ubuntu, macOS, and Windows | All workspace unit, integration, target, feature, and documentation tests plus release-profile checking |
| `msrv` | Rust 1.85.0 on Ubuntu | Every workspace target and test continues to compile at the declared MSRV |
| `portable-target` | Stable `aarch64-unknown-none`, `riscv64gc-unknown-none-elf`, and `wasm32-unknown-unknown` | `proto` and `core` remain `no_std`; device/facade layers require at most `alloc`; Wasm also checks the std reference crates |
| `feature-sets-and-dependencies` | Stable on Ubuntu | Every Cargo feature combination plus a guard against `std`/`alloc` feature leakage into `proto` and `core` |
| `dependency-policy` | Cargo-deny with Rust 1.85.0 | Advisories, yanked crates, duplicate versions, wildcard requirements, licenses, and dependency sources |
| `fuzz-smoke` | Nightly on Ubuntu when `fuzz/Cargo.toml` exists | Every fuzz target gets bounded smoke iterations; issue #26 activates the job by adding the fuzz workspace |

The native jobs intentionally use GitHub-hosted `*-latest` images so runner security and supported
host versions advance without changing the project’s semantic target list. Release evidence records
the concrete runner versions used for the release.

## Crate portability tiers

| Tier | Crates | Allowed runtime surface |
|---|---|---|
| `core-only` | `virtio-accel-proto`, `virtio-accel-core` | `core`; proc-macros may execute with `std` on the build host, but target dependencies may not enable `std` or `alloc` |
| `alloc-portable` | `virtio-accel-device`, `virtio-accel` | `core + alloc`; no OS, filesystem, sockets, threads, or host synchronization |
| `std-reference` | `virtio-accel-mock` | Portable `std`; no host-OS or vendor-specific API |

Future reference guest and queue crates belong to `alloc-portable`. Concrete VMM, kernel, OS, and
vendor adapters are outside the portable-v1 milestone and must not become default dependencies of a
portable crate.

## Cargo feature policy

CI runs `cargo hack check --feature-powerset --no-dev-deps` across the workspace. Features must be
additive: disabling default features may remove convenience behavior but must not select a different
protocol interpretation.

The portable dependency guard inspects resolved target features for `virtio-accel-proto` and
`virtio-accel-core`. A dependency’s host-side derive macro may use `std`, but the target graph for
these crates must not enable a dependency feature named `std` or `alloc`.

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
cargo clippy --workspace --all-targets --all-features -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --all-features --no-deps
cargo test --workspace --all-targets --all-features
cargo test --workspace --doc --all-features
cargo +1.85.0 check --workspace --all-targets --all-features
cargo hack check --workspace --feature-powerset --no-dev-deps
bash ci/check-portable-dependencies.sh
cargo deny --all-features check
```

Target checks require the corresponding Rust standard libraries:

```sh
rustup target add aarch64-unknown-none riscv64gc-unknown-none-elf wasm32-unknown-unknown
cargo check \
  -p virtio-accel-proto \
  -p virtio-accel-core \
  -p virtio-accel-device \
  -p virtio-accel \
  --target aarch64-unknown-none \
  --no-default-features
```
