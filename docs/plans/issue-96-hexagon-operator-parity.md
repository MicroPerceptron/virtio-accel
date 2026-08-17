# Issue #96 plan: Qualcomm Hexagon TOSA operator parity

Status: implemented and validated; `ERF` remains the evidence-backed shared-surface exception

Issue: [#96 — Qualcomm Hexagon: reach shared TOSA operator parity](https://github.com/MicroPerceptron/virtio-accel/issues/96)

## Completed delivery

- [x] Extend the owned Rust/C ABI for static tensors, BOOL storage, variable parameters, and all
      operator families without exposing QNN types to portable code.
- [x] Add constants, reshape, transpose, reverse, and concat with checked static metadata.
- [x] Add FP16 unary, activation, comparison, selection, logical, reduction, indexing, and power
      lowering.
- [x] Decompose `MAX_POOL2D`, `REVERSE`, and `REDUCE_PRODUCT` where the pinned HTP rejects or lacks
      the direct public QNN operation.
- [x] Add a parity test against the actual Core ML and OpenVINO support functions. It asserts 42
      shared operators and allowlists only `ERF`.
- [x] Add portable lowering coverage and checked-in TOSA fixtures for every newly advertised family.
- [x] Run numerical fixtures for all 41 advertised operators on Snapdragon X126100 HTP v73.
- [x] Revalidate FP16 `MATMUL`, `MAX_POOL2D`, and the separate INT8 identity/MATMUL tier.
- [x] Add a release-mode benchmark with 20 warmups, 200 samples, median/p95 reporting, runtime
      identity, direct-binding counts, and explicit-transfer diagnostics.
- [x] Update the operator matrix, architecture, portability, public API, safety, README, and
      performance evidence.

## Evidence boundary

QAIRT `2.49.0.260730` public `QnnOpDef.h` has no ERF definition, so this branch does not advertise
`ERF`. This is preferable to an undocumented operation name or host fallback. FP32 and FP8 remain
blocked by the precision and encoding evidence recorded in the Issue #95 plan. No unsupported graph
falls back to QNN CPU or GPU.

The exact operator/QNN/test mapping is in
[hexagon-operator-matrix.md](../hexagon-operator-matrix.md). Hardware commands and pinned runtime
identity are in the [crate README](../../crates/virtio-accel-hexagon/README.md).
