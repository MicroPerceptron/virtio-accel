# 9. FP8 operator tier: a separate target, exact widening, no FP8 arithmetic

- Status: accepted (implemented; the full device suite passes on Intel Arc B390 (Panther Lake,
  Mesa ANV), on a Lunar Lake host (Xe2, Mesa ANV), and on Mesa lavapipe, in every advertised
  memory domain, with identical results across all three)
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

3. **Crate-owned exact widening, and a narrowing whose overflow policy is this crate's.** `Builder::widen_fp8` is the kernel twin
   of `virtio_accel_tosa::fp8e4m3_to_f32` / `fp8e5m2_to_f32`: integer expansion for normals and
   specials, one exact multiply for subnormals. Every FP8 value is representable in binary32, so
   the widening is exact on every device by construction and no `OpFConvert` appears that a
   driver could demote. The two encodings are separate variants because they differ in exponent
   width, bias and specials: E4M3 has no infinity and exactly one NaN per sign, while E5M2 is
   IEEE-shaped. Because MATMUL produces FP16 and data movement copies raw bytes, **no f32-to-FP8
   narrowing is needed for MATMUL or data movement**. `CAST` does need it, and TOSA 1.0 through
   1.2 leave float-to-FP8 overflow undefined, so the policy is this crate's: a magnitude too
   large for E4M3 — which has no infinity — becomes NaN rather than saturating to 448. The
   alternatives are not symmetric. A consumer who wants saturation can `CLAMP` in a wider dtype
   before the `CAST` and get it exactly; a consumer handed a saturated 448 cannot distinguish it
   from a value that was always 448. NaN preserves the choice, saturation destroys it, and it
   keeps the crate consistent with `narrow_f16`, which signals unrepresentability as infinity
   rather than clamping to 65504. E5M2 needs no policy: it is IEEE-shaped and overflows to
   infinity.

4. **Mixed input and output storage in the MATMUL key.** `KernelKey::Matmul` carried one
   `float: Storage` for both operands and the result, which cannot express `(FP8, FP8) -> FP16`.
   It now carries `input` and `output` separately; the FP32 and FP16 tiers set them equal. The
   accumulator stays binary32, the width TOSA assigns both FP16 and FP8 MATMUL, and the result
   narrows once through ADR 0008's round-to-nearest-even code.

5. **Admission follows the target's own descriptor.** The operator check consulted the FP32
   tier's capability whatever the target was, so a tier could admit operators it does not
   advertise. It now consults the descriptor for the target being lowered, which makes what the
   FP8 target admits exactly what it promises.

6. **Advertised on every device, no gate.** As in ADR 0008 §3, nothing here needs a device
   feature: the conversions are crate-owned integer and binary32 code. `tosa_capabilities()`
   returns the FP16 and FP8 descriptors on every device the backend opens.

## Evidence

- All 170 kernel variants the backend can assemble, the FP8 ones included, pass
  `spirv-val --target-env vulkan1.3`, and the sweep is now a test rather than a manual step
  (`tests/targets.rs`). ADR 0007 and ADR 0008 both cited this validation, but nothing ran it;
  an invalid module surfaces from pipeline creation as an opaque `VK_ERROR_UNKNOWN` naming
  neither the instruction nor the reason, which cost a bisect during this tier's development.
  The device suite also runs clean under `VK_LAYER_KHRONOS_validation`.

- Two Intel silicon generations agree exactly. The tier's claim is that FP8 numerics cannot vary
  by device, because every widening and narrowing is crate-owned integer and binary32 code with
  no device feature involved; Panther Lake (Xe3 class) and Lunar Lake (Xe2) producing identical
  results, alongside a software ICD, is that claim measured rather than argued.
- Exhaustive: all 256 patterns of each encoding survive `IDENTITY` bit-for-bit on every device,
  in every advertised memory domain. This is the analogue of the FP16
  tier's 65536-pattern `NEGATE` round trip and the reason movement is a raw byte copy.
- `(FP8, FP8) -> FP16` MATMUL is bit-exact against a host reference that widens with the TOSA
  crate's own decoders, accumulates in binary32 and narrows once, on both devices and in every
  domain.
- `CAST` between any two float dtypes the tier stores, both FP8 directions included: every
  encoding widens to the value the TOSA crate's own decoder gives and every finite value returns
  to its own encoding, on both devices in every domain. The host narrowing is verified separately
  and exhaustively — all 256 encodings round-trip, 200k values agree with an independent
  nearest-encoding search sharing no bit arithmetic with it, and the boundaries are pinned (464
  ties to even and stays finite at 448; beyond it is NaN; a value rounding to zero keeps its
  sign).
- Two FP8 matmuls chained through a `CAST` match a host reference layer for layer, which is the
  shape `CAST` exists for: without it every layer after the first would round-trip to the host.
- The extension bits are load-bearing: an FP8 artifact is refused under the FP32/FP16 target.
  The converse is not asserted, because the FP8 target's envelope is a superset and an FP32
  graph remains legal under it.

## Consequences

- A consumer that places FP8 work now has a provider that accepts it on every Vulkan device,
  rather than falling back to a CPU provider. What it gets is exact widening and FP16 results,
  not FP8 arithmetic — which is what TOSA defines FP8 MATMUL to be.
- Graphs mixing FP8 matmuls with FP32 elementwise work partition across the two targets. That is
  the honest shape: TOSA has no FP8 elementwise operator to fuse them with.
- FP8 reaches a graph four ways: as a program input, as a `CONST`, as the result of data
  movement over either, and now as the result of a `CAST` from a wider float. The dominant
  shape — a weight matrix narrowed once on the host, as `axnn` has done since its first FP8
  support — was covered before `CAST`; what `CAST` adds is the chain.
- Authoring `ARGMAX` needed a variant `virtio-accel-tosa-build` did not carry, so this tier adds
  one. Its attribute union tag is its opcode, as the schema arranges for every operator, and the
  crate's existing opcode/tag test covers it.
- What TOSA admits for FP8 and this tier still does not execute is the convolution family and
  the gather family, neither of which this backend implements for any dtype.
