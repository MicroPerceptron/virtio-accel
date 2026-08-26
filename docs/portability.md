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
| `openvino-host-test` | Current stable on Ubuntu with a pinned Intel OpenVINO runtime | The real (probed) OpenVINO backend: lint, unit, integration, semantic-conformance, and example runs against the CPU plugin |
| `msrv` | Rust 1.85.0 on Ubuntu | Every workspace target and test continues to compile at the declared MSRV |
| `portable-target` | Stable `aarch64-unknown-none`, `riscv64gc-unknown-none-elf`, and `wasm32-unknown-unknown` | `cleanroom`, `proto`, `transport`, and `core` remain `no_std`; guest, split-queue, device, and facade layers require at most `alloc`; Wasm also checks the std reference crates |
| `feature-sets-and-dependencies` | Stable on Ubuntu | Every Cargo feature combination plus dependency and `std`/`alloc` leakage guards for the portable codecs, queue ports, and core |
| `dependency-policy` | Cargo-deny with Rust 1.85.0 | Advisories, yanked crates, duplicate versions, wildcard requirements, licenses, and dependency sources |
| `fuzz-smoke` | Pinned nightly on Ubuntu | Shared harness tests plus bounded smoke iterations over generated seed corpora and committed minimized regressions for every fuzz target |
| `publish-dry-run` | Stable on Ubuntu | Every published crate packages in documented order into an isolated local registry, and each one builds, tests, and documents from its own extracted tarball rather than from the workspace |

The native jobs intentionally use GitHub-hosted `*-latest` images so runner security and supported
host versions advance without changing the project’s semantic target list. Release evidence records
the concrete runner versions used for the release.

## Crate portability tiers

Every crate below is published to crates.io, so its tier is a promise to downstream users rather
than an internal convention: a crate may not move to a less-portable tier without the release-note
entry and portability review required by [release-policy.md](release-policy.md). The `std-reference`
tier is the ceiling for the reference and conformance crates, not a licence for host-OS or
vendor-specific APIs — third parties depend on `virtio-accel-conformance` and `virtio-accel-mock`
directly, and they must keep working on any platform the portable crates support.

| Tier | Crates | Allowed runtime surface |
|---|---|---|
| `core-only` | `virtio-accel-cleanroom`, `virtio-accel-proto`, `virtio-accel-transport`, `virtio-accel-core` | `core`; the clean-room codec and transport ports have no normal/build dependencies, while proc-macros used by other crates may execute with `std` on the build host |
| `alloc-portable` | `virtio-accel-guest`, `virtio-accel-split-queue`, `virtio-accel-device`, `virtio-accel-tosa`, `virtio-accel-tosa-build`, `virtio-accel` | `core + alloc`; no OS, filesystem, sockets, threads, or host synchronization |
| `std-reference` | `virtio-accel-mock`, `virtio-accel-conformance` | Portable `std`; no host-OS or vendor-specific API |
| `host-native` | `virtio-accel-coreml`, `virtio-accel-openvino`, `virtio-accel-hexagon`, `virtio-accel-xdna` | Core ML/Foundation on macOS 14+, the OpenVINO C runtime (`libopenvino_c` 2026.x), the complete QAIRT/QNN C SDK on Windows ARM64, or the amdxdna-native HRX runtime (`libhrx`) when detected at build time; a compile-only unsupported-platform or unsupported-runtime placeholder elsewhere |

No host-native crate is a dependency of the facade or any portable layer. The Core ML crate's
Objective-C bridge and TOSA-to-Core ML model compilation are built only when the Cargo target is
macOS. The Linux, Windows, and Wasm workspace jobs compile the placeholder API and backend-local
lowering utilities, while the macOS native job executes its real model and semantic-conformance
tests. An accessible Apple Neural Engine is required to construct the real backend; macOS runners
without one skip execution after checking that availability through Core ML.

The OpenVINO crate's boundary is a build environment rather than a target operating system: its
build script probes pkg-config for `openvino` and compiles the native FFI modules only on success.
`VIRTIO_ACCEL_OPENVINO=1` turns a missing runtime into a loud build failure, `=0` forces the
placeholder, and `VIRTIO_ACCEL_OPENVINO_LIB_DIR` links installations that ship no pkg-config
metadata. The default `native-test` runners have no OpenVINO and compile the placeholder plus the
portable TOSA-to-IR encoder; the dedicated `openvino-host-test` job installs a pinned runtime and
exercises the real backend against the CPU plugin. NPU and GPU devices additionally require the
Intel Level Zero NPU driver or the Intel OpenCL/Level Zero GPU runtime on the host; hosts without
an inference device skip execution after enumeration.

The Hexagon crate exercises its SDK-free placeholder and strict FP16/BOOL/INT32 plus INT8 TOSA graph
planner in portable CI. Its parity test compares the real Core ML/OpenVINO 42-operator surface and
allowlists only `ERF`; portable fixtures lower every other shared operator without QAIRT. Its build
script distinguishes a complete public QAIRT/QNN development
installation from driver-only and AppBuilder/Genie bundles by requiring `QnnInterface.h` and the
Windows ARM64 `QnnHtp` import library. `VIRTIO_ACCEL_HEXAGON=0` forces the placeholder;
`VIRTIO_ACCEL_HEXAGON=1` makes missing requirements a build failure;
`VIRTIO_ACCEL_QNN_SDK_ROOT`/`QNN_SDK_ROOT` and `VIRTIO_ACCEL_QNN_LIB_DIR` select the SDK. On the
pinned Windows ARM64 hardware tier, backend-local tests execute numerical fixtures for all 41
advertised operators plus INT8 identity and zero-point-aware INT8 MATMUL through QNN HTP, followed
by the reusable semantic suite. A public hardware CI lane remains unavailable, so the README
publishes the exact manual replacement commands.

The AMD XDNA crate compiles its portable admission surface (`lower`, including the TOSA IDENTITY,
MATMUL, MAX_POOL2D, and explicit FP8-to-BF16 CAST admissions and its advertised `Target`
constants), the portable
precompiled-artifact codec, and a placeholder on every host; portable CI unit-tests admission. HRX
exposes a plain C ABI, so its
build script has no `cc`/CMake step; it enables the native modules (HRX FFI, the `Accelerator`
implementation with its serialized dispatch worker, and the compiler-helper subprocess) only when
an amdxdna-native HRX prefix (`VIRTIO_ACCEL_HRX_DIR`/`HRX_DIR`) carries both HRX headers — the
amdxdna header must declare `hrx_amdxdna_executable_create`, whose absence marks an older,
incompatible libhrx generation — and `lib/libhrx.so`. `VIRTIO_ACCEL_XDNA=0` forces the placeholder,
`VIRTIO_ACCEL_XDNA=1` makes a missing or incomplete runtime a build failure, and
`VIRTIO_ACCEL_HRX_LIB_DIR` links a bare lib directory. No standard locations are scanned, keeping
the toolchain pin authoritative. On the reference machine with the pinned toolchain
(`VIRTIO_ACCEL_AMDXDNA_TOOLCHAIN`) and an NPU, backend-local tests run a DMA passthrough, a
compiled TOSA BF16 IDENTITY, a bit-exact BF16 → FP32 MATMUL, BF16 NHWC MAX_POOL2D against the
shared bit-exact oracle, both FP8E4M3/E5M2 → BF16 CASTs against shared bit-exact oracles, and the
shared semantic conformance suite end to end; compilation invokes
the aiecc helper as a bounded subprocess (never a Cargo dependency). The conformance run includes
provider-resource accounting and direct-binding diagnostics; a feature-gated on-metal fault suite
proves definite device loss, the 120-second tier-2 watchdog state machine with a shortened test
deadline, rejected pending releases, stable terminal polling, and nonblocking poisoned-instance
discard. MAX_POOL2D mirrors OpenVINO's
propagating-NaN and zero-padding semantics, but XDNA admission is deliberately narrower: batch 1,
kernel and stride dimensions at most 8, and at most 8,192 input-plus-output elements so the full
tensors fit in the AIE2P worker's local memory. A public hardware CI lane remains unavailable.

Concrete VMM, kernel, OS, and vendor adapters are outside the portable-v1 milestone and must not
become default dependencies of a portable crate.

## Cargo feature policy

CI runs `cargo hack check --feature-powerset --no-dev-deps` across the workspace. Features must be
additive: disabling default features may remove convenience behavior but must not select a different
protocol interpretation.

The portable dependency guard inspects normal and build target features for
`virtio-accel-cleanroom`, `virtio-accel-proto`, `virtio-accel-transport`, and
`virtio-accel-core`, `virtio-accel-guest`, `virtio-accel-split-queue`, and
`virtio-accel-tosa`; test-only development dependencies are
intentionally outside the target runtime graph. It additionally proves that the clean-room codec
and transport ports have no normal or build dependencies at all. A dependency's host-side derive
macro may use `std`, but the target graph for these crates must not enable a dependency feature
named `std` or `alloc`.

The official FlatBuffers runtime uses a pure-Rust build script with `rustc_version` and `semver` to
detect the compiler. Those host-only dependencies use `std`; the guard checks the TOSA crate's
normal target graph separately and still forbids a `std` or `alloc` dependency feature there.

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
cargo run -p virtio-accel-coreml --example tosa_coreml
cargo run -p virtio-accel-openvino --example tosa_openvino
cargo run -p virtio-accel-hexagon --example tosa_hexagon
cargo run -p virtio-accel-hexagon --example mock_classifier
cargo +1.85.0 check --workspace --all-targets --all-features
cargo hack check --workspace --feature-powerset --no-dev-deps
bash ci/check-portable-dependencies.sh
cargo deny --all-features check
```

The ordered publication dry run needs network access the first time, to vendor third-party
dependencies into its local registry:

```sh
python3 ci/publish-dry-run.py
```

Add `--keep` to inspect the registry and the extracted per-crate sources afterwards.

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
cargo fuzz run tosa_parse fuzz/corpus/tosa_parse fuzz/regressions/tosa_parse -- -runs=256 -max_total_time=20 -timeout=5 -rss_limit_mb=2048 -max_len=65536
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
  -p virtio-accel-tosa \
  -p virtio-accel \
  --target aarch64-unknown-none \
  --no-default-features
```
