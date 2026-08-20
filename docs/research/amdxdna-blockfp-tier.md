# Block-scaled (block-FP) tier for virtio-accel-amdxdna: forward spec

Companion to [ADR-0001](../adr/0001-amdxdna-first-numerical-tier.md) and
[issue #82](https://github.com/MicroPerceptron/virtio-accel/issues/82). This is a **forward
spec**: nothing here is advertised, implemented, or promised by the first tier. It records the
decided shape of the future block-scaled tier so the work starts from decisions, not from a
blank page. Implementation is tracked outside wayfinder map #78.

## Why a block-FP tier at all

XDNA2 has native block-FP16 hardware (`bfp16ebs8`: groups of 8 values share one 8-bit
exponent; AMD's implementation of Microsoft's MSFP lineage) at roughly integer-rate
throughput. The industry is standardizing the same idea as the OCP Microscaling (MX) formats,
and TOSA's development spec has grown first-class support: `block_shape_t`, the
`EXT-MX-COMMON` / `EXT-MX-INT8` / `EXT-MX-FP8*/FP6*/FP4*` extension family, and block-scaled
MATMUL/CAST/CONV — all currently `status="Experimental"`, `BLOCK_SHAPE_32` only.

Honest uses (per issue #82 discussion): explicitly quantized / BFP models; weight-only
constant packing; fused dequantization kernels; formats whose numerical contract already
permits approximation. **Not** a substitute for FP16 or BF16: block conversion is lossy and
each value's error depends on its block neighbors.

## Decided shape

1. **Two candidate tiers, separately labeled — never one label with two maths.**
   - **Primary: MXINT8 semantics** exactly per the OCP MX v1.0 spec (block-32, E8M0 shared
     scale, int8 elements), advertised under a **provisional vendor bit** in a reserved
     high-bit range of `ExtensionSet`, renamed to the standard `EXT-MX-INT8` bit when TOSA
     releases it and the protocol adopts a TOSA version that includes it. Migration is a
     bit-rename, not a reimplementation.
   - **Optional: native `bfp16ebs8` semantics** as its **own distinct provisional label**,
     if a workload ever needs AMD-native block-8 semantics that MXINT8 cannot express.
   Both may be implemented side by side. The guest chooses per program by requesting a
   target; host configuration may gate which tiers are advertised, and may never change what
   an advertised label means (ADR-0001 principle).
2. **Prerequisite verification ticket** (blocks everything): confirm the mapping hypothesis
   that block-8 hardware executes block-32 MXINT8 exactly — a 32-block with one E8M0 scale
   decomposes into four 8-blocks with equal exponents. Verify element encoding (two's
   complement int8 vs sign-magnitude mantissa), scale alignment, saturation, and NaN/Inf
   handling end to end on silicon. If it fails, the primary tier is re-decided.
3. **Oracle approach**: bit-exact against a **shared reference quantizer** (the `dot_i8_i32`
   pattern): the block quantization step is deterministic given the rounding mode (the
   kernels force round-to-nearest-even), so fixtures quantize with the reference and compare
   exact — no tolerance bands hiding data-dependent error.
4. **Gating protocol work** (why this cannot ship sooner than it will): the TOSA 1.0
   FlatBuffers schema cannot encode block-scaled tensors. Adopting a TOSA version that can is
   a protocol **compatible extension** under `docs/wire-abi.md` §9 — new minor-version
   conformance directory, negotiation, shared-crate (`virtio-accel-tosa`) parser/semantic/
   `DType` support — before any backend work is possible.
5. **CONTRIBUTING boundary**: no advertisement without the conformance story. The tier
   appears in no `Target`, README row, or admission path until its fixtures and oracles
   exist.

## Sequencing

Standalone issue chain, outside map #78: verify mapping (§2) → protocol/schema adoption (§4)
→ shared-crate support + reference quantizer + fixtures (§3) → backend tier + advertisement
(§1). The chain can start any time; its critical path is the TOSA standardization timeline,
which map #78 deliberately does not wait on.
