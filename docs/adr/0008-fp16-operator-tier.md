# 8. FP16 operator tier: native binary16 kernels behind a per-device float-controls gate

- Status: accepted (implemented; verified on Apple M4 via MoltenVK — with the denormal clause of
  the gate experimentally relaxed for measurement, see below — and owed a run on the ANV and RADV
  reference stacks before the README evidence row flips)
- Extends: ADR 0003 (checked-in shaders), ADR 0007 (operator tier mechanics), ADR 0004 (whose
  FP16 deferral this resolves), ADR 0005 (the float-controls probe)
- Resolves: the FP16 half of wayfinder map #154 ticket 5 — the tier's targets, capability
  descriptors, operator subset, and per-device gating

## Context

ADR 0004 deferred FP16: a device may offer it only when `shaderFloat16` and
`VK_KHR_shader_float_controls` probing prove the shared corpus's non-finite, subnormal, and
signed-zero edges. Two design questions were left open: how the kernels evaluate binary16
(arithmetic natively in binary16, or unpacked to binary32 at the storage boundary), and which
Vulkan features the gate must require. ADR 0005 recorded the probe evidence: ANV preserves
binary16 denormals; llvmpipe (and with it the lavapipe CI lane) reports
`shaderDenormPreserveFloat16 = false`, so the FP16 tier cannot be proven there.

A third option was considered and rejected: compute every FP16 operator in binary32 and round at
the storage boundary on *all* devices, unconditionally. It would have made the tier available
everywhere, but silently: a device with native binary16 arithmetic would never use it, the
driver's float-controls variance would be replaced by an invisible mechanism the tier does not
declare, and division-class operators would double-round through binary32 instead of producing
the correctly rounded binary16 result. That is a fallback dressed as a tier, and this project
rejects fallbacks.

## Decision

1. **Native binary16 kernels.** FP16 graphs lower to dedicated `KernelKey` variants carrying
   `float: Storage::Half`. Elementwise arithmetic (`ADD`, `SUB`, `MUL`, `RECIPROCAL`,
   `ABS`/`CEIL`/`FLOOR`/`NEGATE`), comparisons, `MAXIMUM`/`MINIMUM`/`CLAMP`, and `SELECT` execute
   on `Float16` values directly, `NoContraction`-decorated like their binary32 counterparts.
   Tensors stay packed two elements per word (the guest-visible TOSA layout); loads extract the
   16-bit lane and bitcast it to `f16`, stores bitcast back and repack with the same
   `OpAtomicAnd`/`OpAtomicOr` neighbour-safe pattern `BOOL` bytes already use, so no 16-bit
   storage feature is required. The `Float16` and `Int16` capabilities are emitted only by the
   binary16 modules; data-movement kernels (`Move` with `Storage::Half`) copy the 16-bit lanes as
   integers, so `IDENTITY`/`RESHAPE`/`TRANSPOSE`/`REVERSE`/`CONCAT` preserve every bit pattern —
   NaN payloads and subnormals included — and pull in no capability at all.
2. **Binary32 only where TOSA calls for it, documented per operator.** MATMUL accumulates in
   binary32 because TOSA assigns FP16 MATMUL a binary32 accumulator; reduction sums and products
   fold in binary32 for the same reason (max/min/argmax folds are exact selections either way).
   The transcendental lanes (`SIN`, `COS`, `TANH`, `ERF`) and the `EXP`/`LOG`/`RSQRT`/`POW`/
   `SIGMOID` built-ins evaluate their argument in binary32 and round once: no meaningful range
   reduction exists at binary16 precision, and this is the higher-precision evaluation TOSA
   permits — the same reasoning ADR 0007 used to own these functions rather than inherit the
   driver's. The round trip through the exact widening and round-to-nearest-even narrowing keeps
   these lanes bit-identical across devices.
3. **Per-device advertisement, never a fallback.** The backend advertises the FP16 tier only
   where the probe finds all of: `shaderFloat16` and `shaderInt16` (the packed-storage bitcasts),
   `shaderDenormPreserveFloat16`, `shaderSignedZeroInfNanPreserveFloat16`, and
   `shaderRoundingModeRTEFloat16`. The two features are enabled only on such devices.
   `VulkanAccelerator::tosa_capabilities()` returns the FP16 descriptor
   (`VULKAN_TOSA_FP16_CAPABILITY` — the same 42 operators and graph envelope, binary16 added to
   every dtype role, under the same floating-point target identity, the Hexagon dtype-narrowing
   pattern) on gated devices and the FP32-only descriptor everywhere else. Admission rejects FP16
   tensors on ungated instances with `UnsupportedType(FP16)` — rejected, never silently widened.
   lavapipe and MoltenVK both report `shaderDenormPreserveFloat16 = false`, so the CI lane and
   Apple hosts do not advertise the tier; the corpus runs there skip explicitly.

## Evidence

- Host-side: the binary16 widen/narrow conversions are verified exhaustively — all 65536
  patterns round-trip, and the narrowing matches an independent round-to-nearest-even reference
  implementation at every pattern, every midpoint, and a million pseudo-random binary32 values.
  Every kernel variant assembles and passes `spirv-val --target-env vulkan1.3`.
- Device-side (Apple M4, MoltenVK 1.4.2, 2026-09-17): with the denormal clause of the gate
  experimentally relaxed — a measurement, not shipped — the full FP16 corpus passed: the ten
  bit-exact cases (`MATMUL`, `ADD`, `SUB`, `MUL`, `POW`, `MAXIMUM`, `MINIMUM`, `MAX_POOL2D`,
  `IDENTITY_EDGES`, the mock linear classifier), the ulp-tolerated unary/comparison/logical/
  reduction/movement groups, an exhaustive 65536-pattern `NEGATE` round trip, and the eight
  higher-precision lanes within 1 ulp of the correctly rounded binary64 references over the
  entire finite binary16 domain, subnormal-producing outputs included. Two MoltenVK behaviours
  were measured: binary16 NaN payloads are canonicalized by the Metal compiler even through
  `OpFNegate` (permitted by the TOSA pseudocode and the shared corpus, and consistent with the
  gate staying shut — `shaderSignedZeroInfNanPreserveFloat16` overstates the guarantee), and
  binary16 subnormals are produced and preserved in practice, which the reported
  `shaderDenormPreserveFloat16 = false` understates.
- The shipped gate remains the strict one above: advertisement follows what the driver
  *reports*, and the per-device corpus runs are the evidence on each stack. ANV and RADV runs on
  the reference hardware are owed before the README evidence row claims them.

## Consequences

- The FP32 tier is untouched: same kernels, same keys, same admission; the `float` field is
  `Storage::Word` everywhere it was implicitly binary32 before.
- Mixed-dtype graphs are admitted naturally: each dispatch carries its own float storage, and
  the lowering requires every float lane of one operator to share one dtype, as TOSA requires.
- `CLAMP` bounds arrive in the tensor dtype's own bit pattern (two bytes for FP16) and are
  validated host-side on the widened binary32 values; `MATMUL` and `NEGATE` zero-point constants
  are checked at the tensor dtype, FP16 included.
- The exhaustive `NEGATE` round trip asserts the shared corpus's NaN rule (any NaN payload
  satisfies a NaN expectation) rather than payload preservation, because MoltenVK measured
  canonicalization and TOSA leaves payloads implementation-defined. A future driver stack that
  preserves payloads still passes.
- The lavapipe CI lane cannot prove the tier (`shaderDenormPreserveFloat16 = false`), so FP16
  execution evidence lives on the real-GPU lanes; the FP16 tests report an explicit skip
  elsewhere. INT8 gating remains open under ticket 5, unchanged by this ADR.
