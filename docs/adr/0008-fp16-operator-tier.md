# 8. FP16 operator tier: crate-owned binary16 conversions, binary32 evaluation

- Status: accepted (implemented; the full FP16 corpus, the exhaustive 65536-pattern `NEGATE`
  round trip, the binary64 transcendental sweep, and the subnormal-arithmetic probe pass on
  Apple M4 via MoltenVK, Intel Arc LNL (Mesa ANV), and AMD Radeon 860M (RADV); the lavapipe CI
  lane covers the tier from the same run)
- Extends: ADR 0003 (checked-in shaders), ADR 0007 (operator tier mechanics), ADR 0004 (whose
  FP16 deferral this resolves), ADR 0005 (the float-controls probe)
- Resolves: the FP16 half of wayfinder map #154 ticket 5 — the tier's targets, capability
  descriptors, operator subset, evaluation widths, and advertisement

## Context

ADR 0004 deferred FP16 behind per-device `shaderFloat16` and float-controls evidence. The first
implementation of this tier executed arithmetic natively on `Float16` values behind exactly that
gate. Two genuine driver behaviours were measured:

- **Intel Arc LNL (Mesa ANV)**: its f16 ALU flushed a subnormal `OpFNegate` result to `-0` — a
  sign-preserved flush TOSA 1.0 §1.9 explicitly tolerates, but short of the full subnormal
  support the tier wants to deliver.
- **Apple M4 (MoltenVK)**: the Metal compiler canonicalized binary16 NaN payloads through
  `OpFNegate` — also TOSA-tolerated (payloads are implementation-defined), and it reports no
  denormal preservation.

A third claim was made during development and is **retracted**: that RADV's and Apple's f16
adders flushed *negative* subnormal sums to `+0` (sign-losing, non-compliant), and that all
three drivers demoted `OpFConvert` widen–narrow chains back to f16. Both came from an arithmetic
error in the development probe's expected values (`-2^-24 + 2^-24 = +0` was asserted as
`-2^-23`); the hardware was computing the correct answer on every stack. The retraction is
recorded here because the intermediate revision of this ADR and the pull request's history
asserted both.

What the corrected evidence says: native f16 arithmetic was never shown broken for
`ADD`/`SUB`/`MUL` on any stack, but the two genuine quirks above prove the ALU class is not
uniformly trustworthy, TOSA 1.0 §1.10.3 explicitly permits fp16 operations to be implemented in
fp32, and a crate-owned conversion path removes the entire question — along with the device
gate, the float-controls probe, and every per-driver code path — while making the lavapipe CI
lane a continuous FP16 lane.

## Decision

1. **Packed binary16 storage, integer bit handling where the operation is bits.** Tensors stay
   packed two elements per word (the guest-visible TOSA layout); loads extract the 16-bit lane,
   stores repack with the same `OpAtomicAnd`/`OpAtomicOr` neighbour-safe pattern `BOOL` bytes
   already use. Data-movement kernels (`Move` with `Storage::Half`) copy the lanes as integers:
   `IDENTITY`/`RESHAPE`/`TRANSPOSE`/`REVERSE`/`CONCAT` preserve every bit pattern — NaN payloads
   and subnormals included. `NEGATE` and `ABS` are integer sign masks on the packed lane: IEEE
   negate/abs are sign-bit operations, exact for all 65536 patterns on any driver (the two
   genuine ALU quirks above make the float forms strictly worse).
2. **Crate-owned conversions; every other float lane evaluates in binary32.** The widening is an
   integer expansion to binary32, exact for every pattern (subnormals become binary32 normals);
   the narrowing is crate-owned round-to-nearest-even integer code that produces subnormals on
   every device. `ADD`/`SUB`/`MUL` are bit-identical to correctly rounded binary16: a binary16
   product needs at most 22 significand bits and a sum at most 2·11+2 = 24, so the binary32
   evaluation is exact before the single narrowing — no double rounding. `RECIPROCAL`
   double-rounds in principle and stays within TOSA's tolerance. Comparisons,
   `MAXIMUM`/`MINIMUM`/`CLAMP`, `SELECT`, `CEIL`/`FLOOR` are exact over exactly widened values.
   `SIN`/`COS`/`TANH`/`ERF` and the `EXP`/`LOG`/`RSQRT`/`POW`/`SIGMOID` built-ins keep ADR
   0007's crate-owned binary32 numerics. MATMUL and reduction sums/products accumulate in
   binary32, the accumulator width TOSA assigns FP16. Because the conversions are integer code,
   no compiler can demote the binary32 arithmetic back to f16, and no 16-bit type, capability,
   or device feature appears anywhere in the kernels.
3. **Advertised on every device, no gate.** The tier needs nothing from the driver beyond what
   the FP32 tier already relies on, so `VulkanAccelerator::tosa_capabilities()` always returns
   the FP16 descriptor (`VULKAN_TOSA_FP16_CAPABILITY` — the same 42 operators and graph
   envelope, binary16 added to every dtype role, under the same floating-point target identity,
   the Hexagon dtype-narrowing pattern). There is no float-controls probe and no feature
   enablement; the numerics are identical on every device by construction, and the lavapipe CI
   lane exercises the tier continuously.

## Evidence

- Host-side: the binary16 widen/narrow conversions are verified exhaustively — all 65536
  patterns round-trip, and the narrowing matches an independent round-to-nearest-even reference
  implementation at every pattern, every midpoint, and a million pseudo-random binary32 values.
  Every kernel variant assembles and passes `spirv-val --target-env vulkan1.3`.
- Apple M4 (MoltenVK 1.4.2, 2026-09-17): the full backend suite passes — the ten bit-exact
  corpus cases, the ulp-tolerated unary/comparison/logical/reduction/movement groups, the fully
  strict 65536-pattern `NEGATE` round trip, the eight higher-precision lanes within 1 ulp of the
  binary64 references over the whole finite binary16 domain, and the subnormal-arithmetic probe
  (add/sub/mul/compare/max/min/reciprocal/abs over subnormal operands, exact IEEE results).
- Intel Arc LNL (Mesa ANV) and AMD Radeon 860M (RADV), 2026-09-17: the bit-exact corpus, the
  ulp-tolerated groups, the binary64 sweep, and the `NEGATE` round trip pass against the final
  kernels; the corrected probe's re-run on both stacks is the owed confirmation. (Earlier runs
  of these stacks against intermediate kernels pass identically; their "failures" at the time
  were the retracted probe bug, not hardware behaviour.)
- The lavapipe CI lane runs the same suite on every change; llvmpipe executes the tier like any
  other device.

## Consequences

- The FP32 tier is untouched: same kernels, same keys, same admission; the `float` field is
  `Storage::Word` everywhere it was implicitly binary32 before.
- Mixed-dtype graphs are admitted naturally: each dispatch carries its own float storage, and
  the lowering requires every float lane of one operator to share one dtype, as TOSA requires.
- `CLAMP` bounds serialize in the tensor dtype (two bytes for FP16) and are widened host-side to
  the binary32 bit patterns the kernel applies; `MATMUL` and `NEGATE` zero-point constants are
  checked at the tensor dtype, FP16 included.
- Binary16 results are bit-identical across devices for every lane: with crate-owned conversions
  and binary32 evaluation there is no driver degree of freedom left. That is ADR 0007's numerics
  policy carried to its conclusion.
- ADR 0004/0005's per-device float-controls gate is dissolved, not failed: it was designed for a
  native-arithmetic implementation, and the implementation that shipped needs nothing it probed
  for. INT8 gating remains open under ticket 5, unchanged by this ADR.
