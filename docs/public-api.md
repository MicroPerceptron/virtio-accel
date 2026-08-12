# Public API documentation

The public Rust API is split into three documentation layers:

1. Human-facing contracts in `virtio-accel-core`, `virtio-accel-guest`,
   `virtio-accel-device`, `virtio-accel-transport`, `virtio-accel-tosa`, and
   `virtio-accel-conformance`.
2. Pointer-free wire mirrors in `virtio-accel-proto`.
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

## Runnable entry points

Two examples are part of the default CI workflow:

- `cargo run --example backend_conformance`
- `cargo run --example reference_execution`

`backend_conformance` shows how a backend author wires a provider to the reusable conformance
suite. `reference_execution` runs a complete context/buffer/program/queue/submit/poll/read/release
lifecycle against the portable mock backend.

## Baseline, reserved, and post-v1 work

Baseline v1 is the mandatory command, queue, object, reset, error, and conformance behavior in the
normative documents. Reserved feature bits, opcodes, flags, and fields are not optional features;
they are invalid until a later policy assigns semantics. Platform integrations such as KVM,
vhost-user, VFIO, Windows, macOS, or vendor SDK adapters do not change protocol 1.0 and must not
leak into portable default dependencies. `virtio-accel-coreml` is the first concrete example: it
depends inward on `virtio-accel-core`, while the facade and portable crates do not depend on it.

The compatibility and release classification rules are in
[`release-policy.md`](release-policy.md). The protocol 1.0 frozen surface is summarized in
[`releases/v1.0.md`](releases/v1.0.md) and audited in
[`../conformance/v1.0/freeze-audit.md`](../conformance/v1.0/freeze-audit.md).
