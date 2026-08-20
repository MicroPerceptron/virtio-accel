# AMD XDNA first advertised numerical tier: BF16 + integer, FP32/FP16 rejected loudly

Status: accepted ([issue #82](https://github.com/MicroPerceptron/virtio-accel/issues/82), 2026-08-20)

XDNA2 silicon has no FP32 and no FP16 compute path at any layer (kernels, toolchain dtype
universe, or vector datapath — [#80](https://github.com/MicroPerceptron/virtio-accel/issues/80)).
`virtio-accel-amdxdna` therefore advertises what the hardware honestly executes — a BF16 tier
(TOSA `EXT-BF16`) and an integer tier — and rejects FP32/FP16 at program admission instead of
silently substituting BF16 math under an FP32/FP16 label. FP16 rejection reflects that **no
honest path exists today**, not that one is fundamentally impossible: a future FP16 tier is
admissible if and only if an exact emulation (multi-component BF16, FP32-scalar paths,
operator-specific lowerings) is built and validated operator by operator against the corpus's
bit-exact FP16 oracles, including per-op accumulator semantics (FP32-accumulate-then-round-once
differs from FP16-accumulate-per-step).

## The tiers

Two `Target` consts, mirroring the OpenVINO two-target pattern:

- **BF16 target** — `TOSA_1_0, FLOATING_POINT, Level8K, EXT-BF16`. Operator surface:
  IDENTITY (bf16), MATMUL (bf16 → fp32, the spec-mandated shape — the FP32 output is the
  accumulator), CAST (fp32 ↔ bf16), MAX_POOL2D (bf16). FP32 tensors may *exist* (as MATMUL
  outputs or CAST operands); any op *computing* on FP32 is rejected. A MATMUL immediately
  followed by CAST-to-bf16 lowers to the fused native bf16→bf16 kernel. **CAST is in the
  surface because MATMUL's output is constrained to FP32 by TOSA — without CAST, a guest
  could never obtain BF16-valued results; implementers must carry this rationale as a code
  comment at the CAST admission site.**
- **Integer target** — `TOSA_1_0, INTEGER, Level8K`. Operator surface: IDENTITY + MATMUL
  (i8, with TOSA zero-point semantics matching the shared `dot_i8_i32` contract exactly).
  MAX_POOL2D is **deliberately deferred** for integers (no shared fixture exists; matches the
  OpenVINO/Core ML integer-tier precedent) — leave a code comment marking the deferral as
  deliberate, not an oversight.
- **Passthrough breadth**: the BF16 target additionally admits IDENTITY-*only* graphs in
  FP16/FP32 — byte movement is honest in any dtype, and the existing corpus IDENTITY fixtures
  become conformance evidence. Integer dtypes stay confined to the integer target.
- **FP8 storage tier (in-map, sequenced after the mandatory tiers)**: FP8E4M3/E5M2 tensors
  stay graph-visible; an explicit `CAST(fp8 → bf16)` lowers to a small on-NPU conversion
  kernel (every finite FP8 value is exactly representable in BF16); subsequent compute uses
  native BF16 kernels; FP8 *constants* may be expanded at `load_program` (compile time — the
  no-submission-time-bounce-buffer rule is untouched).

## Numerics contract

- **Rounding**: the FP32→BF16 output conversion is fixed to round-to-nearest-even,
  unconditionally set by the kernels we compile (the core's ambient default is floor, which
  would bias every output downward; TOSA mandates RNE). Not configurable.
- **Oracles**: BF16 fixtures are **bit-exact by construction** — fixture values chosen so
  every FP32 partial sum is exactly representable, making the result identical under any
  summation order — plus one tolerance-banded case with realistic values. This keeps the
  corpus's exact-first discipline (eleven of twelve existing oracles are exact; FP32 MATMUL
  is the lone documented tolerance case).
- **README**: the support table gains a sixth BF16 dtype column (existing backends:
  "Not implemented").

## The principle: guest-chosen tiers, never host knobs

Any choice that changes numerical results is made by the **guest, per program, from the
advertised menu** — never by host-side configuration that changes what an unchanged label
means. Host configuration may gate which tiers are *offered*; it may never redefine one.
Precision adaptation in the lossy direction (e.g. FP16 models on this hardware) happens
client-side as an explicit conversion to BF16 before submission — the same consumer-side
pattern `axiom/axnn-vaccel` implements for FP8 (`PromoteFp8ToF16`: opt-in, lossless
direction, label re-declared).

## Rejected alternatives

- **Advertising the shared FP32/FP16 corpus** — requires either emulated-FP32 (silent
  24→8-mantissa-bit input truncation) or FP16-relabeled-as-BF16 (10→7 bits); both violate the
  never-reduce-declared-precision constraint, and the corpus's bit-exact FP16 oracles fail
  such a backend immediately.
- **A host config knob toggling bfp16 (block-FP) execution under the BF16 label** — same
  program, same label, different answers depending on host state invisible to the guest.
- **bfp16 (block-FP16) in the first tier** — a different math (8 values share one exponent;
  data-dependent error), no released TOSA vocabulary (the `EXT-MX-*` block-scaled family is
  development-spec, Experimental, block-32 only), and the TOSA 1.0 serialization schema cannot
  encode block-scaled tensors at all. Committed as follow-on work outside map #78 — see
  `docs/research/amdxdna-blockfp-tier.md`.
- **"BF8" terminology** — elementwise TOSA FP8 converts value-by-value and exactly; block
  formats are neighbor-dependent and lossy. Two different animals; never conflate.
