# 9. FP8 operator tier: a separate target, exact widening, no FP8 arithmetic

- Status: accepted (implemented; the exhaustive 256-pattern identity round trip for both
  encodings and the `(FP8, FP8) -> FP16` MATMUL corpus pass on Intel Arc B390 (Mesa ANV) and
  Mesa lavapipe, in every advertised memory domain)
- Extends: ADR 0003 (checked-in shaders), ADR 0007 (operator tier mechanics), ADR 0008 (whose
  crate-owned-conversion argument this reuses)
- Resolves: the FP8 half of the low-precision boundary — the tier's target, capability
  descriptor, operator subset, evaluation widths, and advertisement

## Context

FP8 is the tier every NPU host provider currently rejects, which is why a CPU fallback exists at
all. Both TOSA encodings are already modelled by `virtio-accel-tosa`, and the hardware situation
argues for doing this in Vulkan first: on the OpenVINO stack (2026.3 and 2026.4), no device
advertises FP8 among its optimization capabilities and the NPU compiler rejects FP8 MATMUL
operands outright (`IE.MatMul op operand #0 must be ranked tensor of 16-bit float or 32-bit
float or 64-bit float or 32-bit signed integer or quantized type`). FP8 there is a storage
format that is converted before it is multiplied. Vulkan needs no such gate: as with FP16, if
the conversions are crate-owned integer code, every device that hosts the FP32 tier hosts this
one with identical numerics.

Two facts shape the tier, and both differ from ADR 0008.

**FP8 is extension-gated, so the target identity must change.** FP16 lives in the base
floating-point profile, so the FP16 tier was a pure dtype narrowing under `VULKAN_TOSA_TARGET`.
TOSA gates FP8 legality on `ExtensionSet::FP8E4M3` / `FP8E5M2`, so the envelope itself differs
and the tier needs a target of its own — the split `virtio-accel-xdna` already makes.

**TOSA admits no FP8 arithmetic at all.** No elementwise lane takes FP8: not `ADD`, `SUB`,
`MUL`, `NEGATE`, `CLAMP`, `SELECT`, any comparison, any reduction, or any transcendental. What
TOSA does admit is `MATMUL` as `(FP8, FP8) -> FP16`, the convolutions, `MAX_POOL2D`, `ARGMAX`,
`CONCAT`, `GATHER`/`SCATTER`, the data-movement operators, `CAST`, `CONST` and `IDENTITY`. So
this is a storage-and-matmul tier, not a narrower FP16 tier, and copying ADR 0008's
"same 42 operators" envelope would advertise lanes admission must then reject.

## Decision

1. **A separate target and a subset envelope.** `VULKAN_TOSA_FP8_TARGET` carries the
   floating-point profile at level 8K with both FP8 extension bits.
   `VULKAN_TOSA_FP8_CAPABILITY` lists only what TOSA admits for FP8 *and* this crate executes:
   `MATMUL`, `CONCAT`, `RESHAPE`, `REVERSE`, `TRANSPOSE`, `CONST`, `CONST_SHAPE`, `IDENTITY`. A
   capability descriptor cannot express per-operator dtype legality, so the operator list is the
   honest intersection rather than the union. `CAST`, `MAX_POOL2D` and `ARGMAX` are admitted by
   TOSA for FP8 and deliberately absent: they need kernels this tier does not yet carry.

2. **Packed byte storage, and nothing writes FP8 except a raw copy.** `Storage::Quarter(format)`
   is geometrically identical to the `BOOL` byte lane and reuses its neighbour-safe
   `OpAtomicAnd`/`OpAtomicOr` store, but the byte is a raw float pattern rather than a canonical
   `0`/`1`, so data movement copies the lanes as integers. Every one of the 256 patterns of each
   encoding therefore moves bit-exactly — NaNs, infinities, subnormals and signed zeros
   included. It is a separate variant from `Byte` precisely so the canonicalizing path cannot be
   reused by accident.

3. **Crate-owned exact widening; no narrowing exists.** `Builder::widen_fp8` is the kernel twin
   of `virtio_accel_tosa::fp8e4m3_to_f32` / `fp8e5m2_to_f32`: integer expansion for normals and
   specials, one exact multiply for subnormals. Every FP8 value is representable in binary32, so
   the widening is exact on every device by construction and no `OpFConvert` appears that a
   driver could demote. The two encodings are separate variants because they differ in exponent
   width, bias and specials: E4M3 has no infinity and exactly one NaN per sign, while E5M2 is
   IEEE-shaped. Because MATMUL produces FP16 and data movement copies raw bytes, **no f32-to-FP8
   narrowing is needed anywhere in the tier**, so none was written.

4. **Mixed input and output storage in the MATMUL key.** `KernelKey::Matmul` carried one
   `float: Storage` for both operands and the result, which cannot express `(FP8, FP8) -> FP16`.
   It now carries `input` and `output` separately; the FP32 and FP16 tiers set them equal. The
   accumulator stays binary32, the width TOSA assigns both FP16 and FP8 MATMUL, and the result
   narrows once through ADR 0008's round-to-nearest-even code.

5. **Advertised on every device, no gate.** As in ADR 0008 §3, nothing here needs a device
   feature: the conversions are crate-owned integer and binary32 code. `tosa_capabilities()`
   returns the FP16 and FP8 descriptors on every device the backend opens.

## Evidence

- Exhaustive: all 256 patterns of each encoding survive `IDENTITY` bit-for-bit on Intel Arc B390
  (Mesa ANV) and lavapipe, in every advertised memory domain. This is the analogue of the FP16
  tier's 65536-pattern `NEGATE` round trip and the reason movement is a raw byte copy.
- `(FP8, FP8) -> FP16` MATMUL is bit-exact against a host reference that widens with the TOSA
  crate's own decoders, accumulates in binary32 and narrows once, on both devices and in every
  domain.
- The extension bits are load-bearing: an FP8 artifact is refused under the FP32/FP16 target.
  The converse is not asserted, because the FP8 target's envelope is a superset and an FP32
  graph remains legal under it.

## Consequences

- A consumer that places FP8 work now has a provider that accepts it on every Vulkan device,
  rather than falling back to a CPU provider. What it gets is exact widening and FP16 results,
  not FP8 arithmetic — which is what TOSA defines FP8 MATMUL to be.
- Graphs mixing FP8 matmuls with FP32 elementwise work partition across the two targets. That is
  the honest shape: TOSA has no FP8 elementwise operator to fuse them with.
- FP8 reaches a graph three ways the tier already admits: as a program input, as a `CONST`, and
  as the result of data movement over either. The dominant shape — a weight matrix narrowed once
  on the host, as `axnn` has done since its first FP8 support, then carried as packed bytes — is
  covered today and tested.
- `CAST` is therefore a refinement rather than the unlock: it is the only operator that *narrows*
  a wider float to FP8 *inside* a graph, which matters for a graph that computes in FP16 or FP32
  and re-narrows mid-graph without a host round trip. It needs the same mixed-storage kernel key
  this ADR introduces. `MAX_POOL2D`, `ARGMAX` and the gather family also output FP8, but only
  from FP8 that already exists.
