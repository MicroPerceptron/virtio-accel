# Public API documentation

The public Rust API is split into three documentation layers:

1. Human-facing contracts in `virtio-accel-core`, `virtio-accel-guest`,
   `virtio-accel-device`, `virtio-accel-transport`, `virtio-accel-tosa`,
   `virtio-accel-tosa-build`, and `virtio-accel-conformance`.
2. Pointer-free wire mirrors in `virtio-accel-proto` and the checked C projection in
   [`include/virtio_accel.h`](../include/virtio_accel.h).
3. Independent conformance artifacts and the clean-room codec under `conformance/`.

The first layer documents ownership, blocking behavior, allocation, copy boundaries, and recovery
semantics in rustdoc. The normative protocol text remains in
[`specification.md`](specification.md), [`wire-abi.md`](wire-abi.md), and
[`virtqueue.md`](virtqueue.md); rustdoc links back to those documents rather than restating every
wire rule in multiple places.

## Raw wire and clean-room exceptions

`virtio-accel-proto` deliberately exposes `repr(C)` structures whose public fields match the
normative wire names exactly. Field-level rustdoc would mostly duplicate
[`wire-abi.md`](wire-abi.md) and increase drift risk. The authoritative field semantics are the
normative document, [`layout.json`](../conformance/v1.0/layout.json), and
[`vectors.json`](../conformance/v1.0/vectors.json).

`include/virtio_accel.h` exposes the same constants and packed layouts to C11 and C++11 consumers.
It deliberately contains no functions, provider handles, allocator hooks, or callbacks: it is a
wire header, not a stable backend plugin ABI. `ci/check-c-header.py` compiles manifest-derived
assertions so adding or changing a recorded namespace or layout cannot silently drift the header.

`virtio-accel-cleanroom` is also intentionally raw. It is a dependency-free independent decoder for
golden-vector validation, not the ergonomic production API. Its public names mirror protocol
terms so another implementer can compare behavior without importing the primary wire crate.

These are documented exceptions, not permission for platform adapters or future public crates to
skip API docs. New ergonomic APIs should document ownership, lifetime, error, blocking, allocation,
copy, and portability behavior at the item where consumers call it.

`virtio-accel-tosa` is an ergonomic exception around private raw bindings: its public API exposes
only verified borrowed views, typed stable-op attributes, and raw forward-compatible enum numbers.
`Model::validate_for` applies the complete stable TOSA 1.0 target semantic pass.
`Model::analyze_for` additionally returns the compact dense-ID execution/liveness/constant/runtime
plan intended for provider lowering, and retains verified `Operator`/`Tensor`/`Shape` views rather
than copying an owned graph. Dynamic providers can validate host-readable CTC values and use the
bounded exact-key specialization cache. Provider-specific capability utilities extend the layer
through `ModelValidator`; generated FlatBuffers tables and unchecked roots remain private.

`virtio-accel-tosa-build` is the authoring companion, not a second ingestion path. It exposes only
owned output bytes and typed static graph definitions, keeps raw table slots and union tags private,
and invokes `virtio-accel-tosa` structural and target validation before returning success.

## Runnable entry points

Six examples are part of the default CI workflow:

- `cargo run --example backend_conformance`
- `cargo run --example reference_execution`
- `cargo run -p virtio-accel-coreml --example tosa_coreml`
- `cargo run -p virtio-accel-openvino --example tosa_openvino`
- `cargo run -p virtio-accel-hexagon --example tosa_hexagon`
- `cargo run -p virtio-accel-hexagon --example mock_classifier`

`backend_conformance` shows how a backend author wires a provider to the reusable conformance
suite. `reference_execution` runs a complete context/buffer/program/queue/submit/poll/read/release
lifecycle against the portable mock backend. `tosa_coreml` proves the production path from a
device-neutral TOSA artifact through backend-local Core ML lowering and direct-bound asynchronous
execution; non-macOS hosts compile the placeholder, and macOS hosts without an ANE skip execution.
`tosa_openvino` proves the same production path through backend-local OpenVINO IR lowering on the
preferred available Intel inference device; hosts without an OpenVINO runtime compile the
placeholder, and hosts without an inference device skip execution.
`tosa_hexagon` executes the shared FP16 identity graph through QNN HTP when the complete QAIRT SDK
is selected on Windows ARM64. The backend advertises 41 of the 42 Core ML/OpenVINO TOSA operators;
the exact restrictions are in the [Hexagon operator matrix](hexagon-operator-matrix.md). SDK-free
hosts retain the compile-only `RuntimeUnavailable` surface without falling back to CPU or GPU.
`mock_classifier` uses the same native lifecycle to compute two sets of class logits from three FP16
features and a direct-bound 3x2 weight matrix.

## Baseline, reserved, and post-v1 work

Baseline v1 is the mandatory command, queue, object, reset, error, and conformance behavior in the
normative documents. Reserved feature bits, opcodes, flags, and fields are not optional features;
they are invalid until a later policy assigns semantics. Platform integrations such as KVM,
vhost-user, VFIO, Windows, macOS, or vendor SDK adapters do not change protocol 1.0 and must not
leak into portable default dependencies. `virtio-accel-coreml` and `virtio-accel-openvino` are the
first concrete host backends. `virtio-accel-hexagon` is a separately packaged, pinned experimental
QNN HTP adapter. Each depends inward on `virtio-accel-core` and `virtio-accel-tosa`, while the
facade and portable runtime crates depend on none of them.

The compatibility and release classification rules are in
[`release-policy.md`](release-policy.md). The protocol 1.0 frozen surface is summarized in
[`releases/v1.0.md`](releases/v1.0.md) and audited in
[`../conformance/v1.0/freeze-audit.md`](../conformance/v1.0/freeze-audit.md).
