# 8. FP16 operator tier: binary32 evaluation behind a per-device float-controls gate

- Status: accepted (implemented; verified on Apple M4 via MoltenVK — with the denormal clause of
  the gate experimentally relaxed for measurement — and on Intel Arc LNL (Mesa ANV) and AMD
  Radeon 860M (RADV) against the shipped gate; the subnormal-arithmetic probe's re-runs on ANV
  and RADV are the final owed evidence)
- Extends: ADR 0003 (checked-in shaders), ADR 0007 (operator tier mechanics), ADR 0004 (whose
  FP16 deferral this resolves), ADR 0005 (the float-controls probe)
- Resolves: the FP16 half of wayfinder map #154 ticket 5 — the tier's targets, capability
  descriptors, operator subset, evaluation widths, and per-device gating

## Context

ADR 0004 deferred FP16: a device may offer it only when `shaderFloat16` and
`VK_KHR_shader_float_controls` probing prove the shared corpus's non-finite, subnormal, and
signed-zero edges. The first implementation of this tier executed arithmetic natively on
`Float16` values, reasoning that a tier should use the hardware's own binary16 arithmetic where
the device gates it in, and that computing everything in binary32 would be a hidden fallback.

Three production stacks then measured otherwise, against both reported properties and the
`SPV_KHR_float_controls` execution modes:

- **Intel Arc LNL (Mesa ANV)** — reports `shaderDenormPreserveFloat16 = true`, tier advertised:
  its f16 ALU flushed a subnormal `OpFNegate` result to `-0` (sign-preserved, TOSA-tolerable).
- **AMD Radeon 860M (RADV)** — reports every required property, tier advertised, execution modes
  in the module: `0x8001 + 0x8001` returned `+0` — a *sign-losing*, non-uniform flush. TOSA 1.0
  §1.9 requires subnormals to be "supported or flushed to **sign-preserved** zero", and §1.10.3
  requires the choice to be uniform: RADV's f16 adder is non-compliant for negative subnormal
  sums.
- **Apple M4 (MoltenVK)** — reports no denormal preservation (never gated): the same sign-losing
  flush, and its compiler additionally demotes trivially demotable widen–add–narrow chains back
  to f16, so no shader-level construction delivers compliant subnormal arithmetic there.

TOSA 1.0 §1.10.3 settles the design question the native path raised: "These requirements allow
fp16_t operations to be implemented using the fp32_t datatype." Evaluating in binary32 is not a
fallback or a relabeling; it is a compliance mechanism the specification names explicitly — and
after the measurements above, the only one that works on every stack.

## Decision

1. **Packed binary16 storage, integer bit handling where the operation is bits.** Tensors stay
   packed two elements per word (the guest-visible TOSA layout); loads extract the 16-bit lane
   and bitcast it to `f16`, stores bitcast back and repack with the same
   `OpAtomicAnd`/`OpAtomicOr` neighbour-safe pattern `BOOL` bytes already use, so no 16-bit
   storage feature is required. Data-movement kernels (`Move` with `Storage::Half`) copy the
   lanes as integers: `IDENTITY`/`RESHAPE`/`TRANSPOSE`/`REVERSE`/`CONCAT` preserve every bit
   pattern — NaN payloads and subnormals included — with no float capability at all. `NEGATE`
   and `ABS` are integer sign operations on the packed lane: IEEE negate/abs are sign-bit
   operations, exact for all 65536 patterns on any driver (this decision was forced by ANV's
   measured `OpFNegate` flush).
2. **Every other float lane evaluates in binary32 and rounds once.** `ADD`/`SUB`/`MUL` are
   bit-identical to correctly rounded binary16 on every device: a binary16 product needs at most
   22 significand bits and a sum at most 2·11+2 = 24, so the binary32 evaluation is exact before
   the single round-to-nearest-even narrowing — no double rounding. `RECIPROCAL` double-rounds
   in principle and stays within TOSA's tolerance. Comparisons, `MAXIMUM`/`MINIMUM`/`CLAMP`,
   `SELECT`, `CEIL`/`FLOOR` are exact selections and evaluations of exactly widened values.
   `SIN`/`COS`/`TANH`/`ERF` and the `EXP`/`LOG`/`RSQRT`/`POW`/`SIGMOID` built-ins keep ADR
   0007's crate-owned binary32 numerics — the only precision at which range reduction is
   meaningful. MATMUL and reduction sums/products accumulate in binary32, the accumulator width
   TOSA assigns FP16. Widening is exact (`OpFConvert`, and binary16 subnormals become binary32
   normals); narrowing is round-to-nearest-even and produces subnormals — the full-support
   choice TOSA's "supported or flushed" rule offers, delivered uniformly.
3. **The shaders still ask for the contract explicitly.** Every binary16 module that converts to
   or from `f16` emits the `SPV_KHR_float_controls` execution modes (`DenormPreserve`,
   `SignedZeroInfNanPreserve`, `RoundingModeRTE` at width 16). Their Vulkan VUIDs name exactly
   the device properties the gate probes, and they govern the conversions the tier now depends
   on.
4. **Per-device advertisement, never a fallback.** The backend advertises the FP16 tier only
   where the probe finds all of: `shaderFloat16` and `shaderInt16` (the conversions and
   packed-storage bitcasts), `shaderDenormPreserveFloat16`,
   `shaderSignedZeroInfNanPreserveFloat16`, and `shaderRoundingModeRTEFloat16`. The two features
   are enabled only on such devices. `VulkanAccelerator::tosa_capabilities()` returns the FP16
   descriptor (`VULKAN_TOSA_FP16_CAPABILITY` — the same 42 operators and graph envelope,
   binary16 added to every dtype role, under the same floating-point target identity, the
   Hexagon dtype-narrowing pattern) on gated devices and the FP32-only descriptor everywhere
   else. Admission rejects FP16 tensors on ungated instances with `UnsupportedType(FP16)` —
   rejected, never silently widened. lavapipe and MoltenVK both report
   `shaderDenormPreserveFloat16 = false`, so the CI lane and Apple hosts do not advertise the
   tier; the corpus runs there skip explicitly.

## Evidence

- Host-side: the binary16 widen/narrow conversions are verified exhaustively — all 65536
  patterns round-trip, and the narrowing matches an independent round-to-nearest-even reference
  implementation at every pattern, every midpoint, and a million pseudo-random binary32 values.
  Every kernel variant assembles and passes `spirv-val --target-env vulkan1.3`.
- Intel Arc LNL (Mesa ANV, 2026-09-17, shipped gate): tier advertised; the bit-exact corpus,
  the ulp-tolerated groups, and the binary64 sweep pass. The exhaustive `NEGATE` round trip
  caught the f16 ALU's subnormal flush and forced the integer sign lanes.
- AMD Radeon 860M (RADV, 2026-09-17, shipped gate): tier advertised; the bit-exact corpus, the
  ulp-tolerated groups, the binary64 sweep, and the fully strict 65536-pattern `NEGATE` round
  trip pass. The subnormal-arithmetic probe caught the sign-losing f16-adder flush and forced
  the binary32 evaluation policy; its re-run against the final kernels is owed.
- Apple M4 (MoltenVK 1.4.2, 2026-09-17, denormal clause experimentally relaxed — a measurement,
  not shipped): the full corpus passes. Its sign-losing subnormal flush survives even the
  widen–narrow path because the compiler demotes the chain back to f16 — the demonstration that
  the gate, not shader construction, is what keeps non-compliant stacks unadvertised.
- The `fp16_subnormal_arithmetic_is_exact_where_the_tier_is_advertised` probe (add/sub/mul/
  compare/max/min/reciprocal/abs over subnormal operands, exact IEEE results) is the
  held-to-contract check on every gated stack; ANV and RADV re-runs against the final kernels
  are the owed evidence.

## Consequences

- The FP32 tier is untouched: same kernels, same keys, same admission; the `float` field is
  `Storage::Word` everywhere it was implicitly binary32 before.
- Mixed-dtype graphs are admitted naturally: each dispatch carries its own float storage, and
  the lowering requires every float lane of one operator to share one dtype, as TOSA requires.
- `CLAMP` bounds serialize in the tensor dtype (two bytes for FP16) and are widened host-side to
  the binary32 bit patterns the kernel applies; `MATMUL` and `NEGATE` zero-point constants are
  checked at the tensor dtype, FP16 included.
- Binary16 results are bit-identical across devices for every lane: the correctly rounded
  result of exactly widened inputs has no driver degree of freedom left. That is the same
  numerics policy ADR 0007 committed to, now measured to be unavailable from f16 ALUs.
- Reported float-controls properties are a floor, not a proof: two of three stacks flushed
  subnormals despite reporting preservation, one of them non-compliantly. Anything the tier
  promises bit-exactly must either avoid the ALU (sign lanes, data movement) or be pinned by an
  execution test on each stack — the subnormal-arithmetic probe exists for exactly this.
- The lavapipe CI lane cannot prove the tier (`shaderDenormPreserveFloat16 = false`), so FP16
  execution evidence lives on the real-GPU lanes; the FP16 tests report an explicit skip
  elsewhere. INT8 gating remains open under ticket 5, unchanged by this ADR.
