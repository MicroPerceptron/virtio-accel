# AMD XDNA2 `bfp16ebs8` characterization (issue #146)

Status: **complete** — toolchain surface frozen 2026-08-27; probes P0–P5 run on silicon
2026-08-27/28 (results in §6); bit-level reference model delivered and cross-validated against
silicon (§4). Verdict: **exact MX mapping, with three stated conditions** (§6a). This meets the
issue #146 exit criteria: the format is described by a cited note and a host-verifiable
reference model, on-metal tests distinguish exact matches from mismatches, and the
exact-MX-mapping outcome is chosen explicitly. No public `Target`, extension bit, capability
row, or protocol value was added.
Part of #110. Companion to
[tosa-int8-block-fp-status.md](tosa-int8-block-fp-status.md) and the forward spec
`docs/research/amdxdna-blockfp-tier.md` (branch `grilling/first-numerical-tier`).

This note advertises nothing. Per the ticket boundary, no public TOSA `Target`, extension bit,
`TosaCapabilityProvider` row, or stable protocol value is added by this work.

## Pinned identities

Every fact below is cited against the pinned toolchain; a different revision re-opens the fact.

| Component | Identity |
|---|---|
| Toolchain prefix | `~/toolchains/amdxdna-hrx-v2026.08` |
| `mlir_aie` wheel | `1.4.1` (cp312, manylinux_2_35_x86_64) |
| Peano | `llvm-aie==21.0.0.2026080301+c9c5ecb7` |
| Fork `amd-npu-compiler` | commit `c95544269f0c074d6d3e213ee43cc34dc4100801` |
| Reference NPU | PCI `1022:17f0` rev `0x20` (XDNA2, `__AIE_ARCH__ == 21`) |

Header paths below are relative to
`ironenv/lib/python3.12/site-packages/` inside the prefix; `aie_api` is
`mlir_aie/include/aie_api`, `aie2p` is `llvm-aie/lib/clang/21/include/aie2p`.

## 1. Frozen from the pinned toolchain

These are compiler-surface facts, not silicon behavior. They constrain what a kernel can express;
the probes in §3 decide what the hardware actually computes.

**Format parameters** (`aie_api/aie_doc.hpp:230-237`): `bfp16ebs8` is a block of **8** values,
**8** mantissa bits each, **no sub-tile shifts**, one shared **8-bit** exponent, **9 bytes per
block**. `bfp16ebs16` is the block-16 analogue (17 bytes). A 64-element vector is 8 blocks,
72 bytes; its load/store FIFO width is 576 bits (`aie2p/aie2p_ldst.h:287,559`).

**Register-level split** (`aie2p/aie2p_srs.h:1308-1313`): `v64bfp16ebs8` is
`{ mantissa: v64char, exponent: v8char }` — mantissas are *signed bytes* at the C level. The
in-memory byte order of the 72-byte unit (exponent plane vs mantissa plane, per-block vs grouped)
is **not** derivable from the headers; probe P0 pins it.

**XDNA2 supports no MX-native type** (`aie_api/aie_doc.hpp:247-252`): the supported block types on
XDNA2 are `bfp16ebs8` and `bfp16ebs16` at 32/64/128/256 elements. The `mx4`/`mx6`/`mx9` types
(block-16, E8M0 scale, with sub-tile shifts — the OCP-shaped family) exist only on **AIE-MLv2**.
Consequence: on this part, OCP MX block-32 semantics can only be reached through the
block-8 decomposition hypothesis (#110 stage 1). There is no native alternative to characterize
against.

**Native MMUL surface** (`aie_api/detail/aie2p/mmul_bfp16_bfp16.hpp`):

| Shape (M×K×N) | Types | Accumulator | Intrinsic |
|---|---|---|---|
| 8×8×8 | `bfp16ebs8 × bfp16ebs8` | `accfloat` (FP32) | `mul_8x8_8x8T` / `mac_8x8_8x8T_conf` |
| 8×8×16 | `bfp16ebs8 × bfp16ebs8` | `accfloat` ×2 | `mul_4x8_8x16T` ×2 (shuffle high half) |
| 8×8×8 | `bfloat16 × bfp16ebs8` | `accfloat` | lhs converted in-core via `to_v64bfp16ebs8` |

Accumulation for `bfp16ebs8 × bfp16ebs8` is **`accfloat` (FP32)** on XDNA2
(`aie_api/aie_doc.hpp:1027-1035`). The `T` suffix on every intrinsic indicates a transposed-B
layout; the exact operand layout contract is confirmed by probe P4.

**Conversion in** (`aie2p/aie2p_srs.h:1308,1331`): `to_v64bfp16ebs8(v64accfloat)` converts an FP32
accumulator to one 64-element block vector. Rounding is governed by the core's `crrnd` control
register — `to_v64bfp16ebs8_conf(a, rnd)` brackets the conversion with `set_rnd(rnd)`. The mode
vocabulary (`aie2p/aie2p_defines.h:27-47`): `rnd_floor(0)`, `rnd_ceil(1)`, `rnd_sym_floor(2)`,
`rnd_sym_ceil(3)`, `rnd_neg_inf(8)`, `rnd_pos_inf(9)`, `rnd_sym_zero(10)`, `rnd_sym_inf(11)`,
`rnd_conv_even(12)`, `rnd_conv_odd(13)`. The forward spec's assumption that kernels force
round-to-nearest-even corresponds to `rnd_conv_even = 12`; the *default* mode at kernel entry is
a runtime fact for probe P1.

**No direct conversion out** (`aie_api/detail/aie2p/accum.hpp:745-760`): there is no
bfp16ebs8→float intrinsic. The AIE API converts out by **MMUL against an identity matrix**, and
only on `__AIE_ARCH__ == 21` (with `bfp16ebs16` unsupported, per the `CRVO-9745` FIXME). Any
"storage conversion" tier for this format is therefore matrix-unit math, not an SRS-style cast —
a structural difference from the FP8→BF16 tier.

**Sparse variants exist** (`v64bfp16ebs8_sparse`, `aie2p/aie2p_aie_api_compat.h:59-62`) — noted
and out of scope for #146.

## 2. Hypotheses extracted from the toolchain (to confirm or refute on silicon)

The identity matrix used by the API's own convert-out path
(`aie_api/detail/aie2p/accum.hpp:753-760`) encodes diagonal **1.0** as mantissa byte `0x40` with
exponent byte `127 + shift`. That single construction implies most of the numerical contract:

- **H1 — element semantics**: value = `m · 2^(e − 127 − 6)` with `m` a two's-complement int8
  mantissa (so `0x40` = 64 = `1.0·2^6` mantissa units at bias-127) and `e` the shared unsigned
  8-bit exponent. Equivalently: int8 mantissa with 6 fractional bits, range (−2, +2), E8-style
  bias-127 scale. **If H1 holds, the element semantics coincide with OCP MXINT8's** (int8 element,
  2 integer + 6 fractional bits, shared E8M0 scale) — the mapping hypothesis reduces to block-size
  decomposition alone.
- **H2 — memory layout**: undetermined between exponent-plane-then-mantissa-plane, per-block
  interleaving, or another order. The register struct keeps them separate; DMA order is unknown.
- **H3 — rounding**: conversion honors the `crrnd` register; default mode at kernel entry
  presumed `rnd_conv_even` but unverified.
- **H4 — normalization**: the shared exponent of a converted block is presumed
  `max(exponent of members)` aligned so the largest member uses the full mantissa range;
  members with smaller exponents lose low bits by the H3 rounding. Unverified.
- **H5 — exceptional values**: E8M0-style exponents have no sign, and the OCP MX spec reserves
  `0xFF` as NaN; whether `to_v64bfp16ebs8` saturates, produces `0xFF`, or wraps on FP32
  overflow/Inf/NaN input is unknown. Signed zero and FP32 subnormal handling likewise.
- **H6 — the #110 stage-1 mapping**: one OCP MXINT8 block-32 (one E8M0 scale, 32 int8 elements)
  decomposed into four `bfp16ebs8` blocks with *equal* exponent bytes produces bit-identical
  dot-product results in FP32 accumulation. Given H1 this is plausible by construction; it must
  still be proven under exponent disagreement between sub-blocks (where the four-block form is
  *more* precise than block-32, and the mapping direction matters).

## 3. Probe plan

Each probe is a backend-local precompiled artifact (`XDNA_PRECOMPILED_FORMAT` via
`virtio_accel_xdna::artifact::encode`), run through the released `Accelerator` lifecycle with
direct binding on the reference NPU — the same harness discipline as `tests/hardware.rs`, kept
entirely out of the TOSA admission path. Probe kernels are compiled by a standalone IRON driver
under `research/` (deliberately separate from `compiler/xdna_compile.py`, which the serving path
owns).

- **P0 — encoding & layout dump.** Kernel converts a chosen `v64accfloat` pattern with
  `to_v64bfp16ebs8` and DMA-writes the raw 72-byte unit to the output buffer. Host decodes under
  each candidate layout; distinct FP32 inputs with known exponents/mantissas separate the planes.
  Freezes H1 + H2.
- **P1 — rounding.** Tie-case FP32 inputs (`…0.5` mantissa-unit boundaries, odd/even neighbors)
  converted under the default mode and under explicit `set_rnd` values; compared to the reference
  model per mode. Freezes H3 and the default.
- **P2 — normalization & intra-block loss.** Blocks mixing magnitudes (equal, off-by-one
  exponent, extreme spread, all-zero block) to observe shared-exponent selection and low-bit
  loss. Freezes H4.
- **P3 — exceptional values.** ±0, FP32 subnormals, ±Inf, NaN, and overflow-magnitude inputs
  through conversion; exponent byte `0xFF` and mantissa patterns recorded. Freezes H5.
- **P4 — MMUL contract.** Host-crafted raw blocks (bypassing conversion entirely) through
  `mul_8x8_8x8T`; FP32 accumulator out. Establishes operand layout (the `T` transpose), the
  multiply semantics `(m_a·2^{e_a−133})·(m_b·2^{e_b−133})` summed in FP32, and accumulation
  order effects if any. Includes the mixed `bfloat16 × bfp16ebs8` shape.
- **P5 — MX mapping verdict.** Pinned OCP MX v1.0 MXINT8 reference quantizer (host, Rust)
  produces block-32 vectors; decomposition to four equal-exponent block-8 groups runs on the
  matrix unit; results compared bit-exactly against the reference dot product in FP32, including
  constructed cases where the four sub-block exponents *disagree* (the mapping's directionality
  evidence). Decides H6 — and with it, #110 stage 1: exact MX mapping, transform with documented
  loss, or distinct vendor-only contract.

Every probe records the toolchain identity table above plus driver/firmware versions at run time,
and preserves its artifact and raw output vectors under `research/` for repeatability.

## 4. Reference model (delivered)

`research/bfp16ebs8/runner/src/model.rs` holds the bit-level model, host-verifiable without an
NPU (`cargo test` in the runner project replays the recorded silicon planes as fixtures):

- `encode_block`/`encode_v64`: the hardware converter exactly as P0–P3 observed it — max-member
  IEEE exponent-field selection, subnormal flush, structural Inf/NaN at `e = 255`, all ten
  `crrnd` rounding functions, post-rounding renormalization;
- `mxint8_quantize_block`: OCP MX v1.0 MXINT8 quantization implemented from the spec (RNE,
  saturating) as the deliberately independent second formulation;
- `decode` and `dot_reference` for exact value-level comparisons.

The probe runner cross-validates live: every P0/P2/P3 case asserts the silicon planes are
bit-identical to `encode_v64`, including a 64-element pseudorandom case
(`results/p0-2026-08-28.txt`, `p2-…`, `p3-…`: eleven cases, all `model check: PASS`).

One divergence between the two formulations is real and documented in the model
(`CONVERTER_OCP_OVERFLOW_DIVERGENCE`): at an exact mantissa of ±127.5, an up-rounding hardware
conversion renormalizes to the next exponent while the OCP quantization procedure saturates the
element without re-selecting the scale. Both are self-consistent; a tier claiming MXINT8
semantics must therefore quantize with the OCP procedure rather than the hardware converter
whenever inputs can sit on that boundary.

## 5. Feed into #145 (draft tracking)

The `mx4/mx6/mx9` discovery is itself #145 evidence: the AIE-API's MX-shaped family is block-16
with 1-bit sub-tile shifts — not the TOSA 1.1-draft's block-32 — and targets AIE-MLv2, not XDNA2.
Neither AMD surface matches the draft's `BLOCK_SHAPE_32` vocabulary natively. Any released TOSA
MX contract will be executed on this part through a decomposition, which strengthens the case for
keeping the vendor tier and the (future) MX tier separately labeled, as ADR-0001 requires.

## 6. P0 results (silicon, 2026-08-27)

Probe sources: `research/bfp16ebs8/` (kernel `kernel_p0.cc`, IRON driver `probe_compile.py`,
runner `runner/`); raw output `research/bfp16ebs8/results/p0-2026-08-27.txt`; the compiled
artifact is preserved under `research/bfp16ebs8/artifacts/`. Run on the reference `1022:17f0`
rev `0x20` NPU with HRX `hrx-amdxdna-2026.07.30-amdxdna-hal-native` through the released
`Accelerator` lifecycle (direct binding, `XDNA_PRECOMPILED_FORMAT`).

**H1 — element semantics: CONFIRMED.** For every non-flagged case,
`value = m · 2^(e − 127 − 6)` with `m` a two's-complement signed int8 mantissa and `e` the
unsigned shared exponent byte. Negative values produce negative mantissa bytes (two's
complement, not sign-magnitude): `−1.0 → m = −32` at `e = 128`, `−0.5 → m = −16`. Powers of
two encode as `m = 64` with `e = 127 + log2(v)` when they are the block maximum
(case A: `0.125..16 → e = 124..131`, all `m = 64`). **This is OCP MXINT8's element contract**
(int8, 2 integer + 6 fraction bits, E8-biased scale), differing only in block size.

**H2 — layout: PINNED.** The native in-memory form of `v64bfp16ebs8` is the 64-byte mantissa
plane followed by the 8-byte exponent plane (element order within blocks, block order across
the vector; confirmed by comparing an explicit plane dump against the struct stored as-is).
An all-zero block encodes as `e = 0, m = 0`.

**H3 — rounding: REFUTED as assumed.** `get_rnd()` at kernel entry is **0 = `rnd_floor`**
(round toward −∞), not round-to-nearest-even. Observed: `127/64` at forced `e = 128` has exact
mantissa 63.5 and encodes as `m = 63` (floor), `1/128` at `e = 127` (exact 0.5) encodes as
`m = 0`, and `1.5/64` (exact 1.5) as `m = 1`. Consequence for #110/#148: an MX-exact kernel
must explicitly `set_rnd(rnd_conv_even)` (12) around every conversion — the forward spec's
"kernels force round-to-nearest-even" is a requirement on our kernels, not a hardware default.
Probe P1 sweeps the modes and the negative-tie cases to pin each mode's exact function.

**H4 — normalization: characterized, sharper than hypothesized.** The shared exponent is the
maximum member's *IEEE FP32 exponent* — not the smallest exponent that fits the int8 range.
Case B (max member `−2.0`) chose `e = 128` with `m = −64`, although `e = 127` with `m = −128`
is representable and would have kept a full extra bit for every other member (it would have
made `127/64` exact). The conversion therefore normalizes the largest-magnitude member into
`|m| ∈ [64, 127]` (exactly 64 for powers of two); `m = −128` is never a *normalization target*,
though rounding a near-max negative member can still produce it (P1 `sat`: `−127.5` under floor
encodes as `m = −128` at the unbumped exponent). Mixed-magnitude
members are quantized at that exponent and lose low bits under the active rounding mode
(case C: at `e = 127`, `1/128 → m = 0`, `1.5/64 → m = 1`, while `1 ± 1/64 → m = ±65` exactly).
This matches OCP MX v1.0's scale rule (`X = 2^(floor(log2(max|v|)) − emax_elem)` with
`emax_elem = 0` for INT8's 2.0 ceiling — same outcome as "max member's IEEE exponent" for all
P0 cases); P2 adds the boundary cases where the two formulations could differ.

**P1 — rounding modes: all ten exact.** Every `crrnd` mode (`floor, ceil, sym_floor, sym_ceil,
neg_inf, pos_inf, sym_zero, sym_inf, conv_even, conv_odd`) matched the reference rounding
function on all 64 lanes, including negative ties and large-mantissa ties
(`results/p1-2026-08-27.txt`). `set_rnd`/`to_v64bfp16ebs8_conf` work from kernel context and
restore the previous mode. **Round-to-nearest-even is therefore available and exact**, which is
what the MX tier needs — it just is not the default.

**P1b — overflow renormalizes, never saturates or wraps.** With exact mantissa `±127.5` at
`e = 127`, every mode that rounds the magnitude up to 128 (ceil, sym_ceil, pos_inf, sym_inf,
conv_even) bumps the shared exponent to 128 and re-quantizes the whole block; every mode that
rounds down (floor, sym_floor, neg_inf, sym_zero, conv_odd — 127.5 ties to odd 127) keeps
`e = 127`. The observed mantissas match the reference at the *post-bump* exponent in every mode.

**P2 — normalization: `e` is the max member's IEEE FP32 exponent *field*.** Verified through
`e = 254` (`f32::MAX → m = 127`). An all-subnormal-FP32 block flushes to zero (`e = 0, m = 0`) —
input FTZ. A negative-only block is symmetric (`−1.984375 → m = −127` exact at `e = 127`).
Members far below the block maximum quantize at the shared scale under the active mode — under
the default floor mode a tiny *negative* member becomes `m = −1` rather than 0, one more reason
the MX tier must run under `conv_even`. (`results/p2-2026-08-27.txt`)

**P3 — exceptional values are structural, not special-cased.** `+Inf → e = 255, m = +64`
(the implicit-one pattern), `−Inf → m = −64`, `NaN → e = 255, m = ±96` (quiet-bit pattern),
`±0 → m = 0`. Other members of an Inf/NaN block simply quantize at the `e = 255` scale
(`2^122`): positives floor to 0, negatives to −1 under the default mode — the max-exponent rule
extended to field 255, with per-block isolation confirmed. Consequence for the MX mapping: OCP
E8M0 reserves 255 as NaN and MXINT8 has no Inf, so `e = 255` blocks are outside MXINT8's domain;
the mapping direction MX → `bfp16ebs8` never produces them, and any `bfp16ebs8 → MX` path must
reject or canonicalize them. (`results/p3-2026-08-27.txt`)

**Emerging model.** All P0–P3 observations are consistent with one description: the conversion
takes each member's IEEE FP32 representation, selects the block exponent as the maximum member
exponent field (0 treated as flush-to-zero, 255 passed through), forms each member's signed
mantissa `sign · 1.fraction · 2^6` shifted right by `e − e_member`, and rounds per the `crrnd`
mode with post-rounding renormalization. This is the reference model P4/P5 will encode.

**P4 — MMUL contract: bit-exact against the H1 semantic model.** The operand layout is the
transposed-B form the intrinsic names imply: A lane `i*8+k`, B lane `j*8+k`, C lane `i*8+j`,
with block `n`'s exponent byte applying to row `n` of its operand (established empirically with
single-entry operands, `results/p4-2026-08-27.txt`). With host-crafted raw planes bypassing the
converter entirely, `mul_8x8_8x8T`/`mac_8x8_8x8T` chains over K = 32 reproduce the reference
`sum(m_a · m_b · 2^(e_a + e_b − 266))` bit-exactly in FP32 — including per-chunk exponent
differences, per-*block* exponent disagreement inside one chunk, and mantissa `−128`, which the
converter never emits (P0/P2) but the matrix unit honors as exactly −2.0.

**P5 — the #110 stage-1 hypothesis: CONFIRMED.** A block-32 MXINT8 dot product (one E8M0 scale,
32 int8 elements per operand row) decomposed into four `bfp16ebs8` blocks with equal exponent
bytes is bit-exact on the matrix unit against the pinned reference (`results/p5-2026-08-27.txt`).
Scale-representation equivalence holds: the same numeric operand expressed as `(m, e)` and as
`(m/2, e+1)` produces bit-identical accumulator lanes, so results depend on values, not on the
chosen block normalization.

### 6a. Verdict for issue #146

**Exact MX mapping**, with the conditions the probes surfaced — each a requirement on the future
tier (#148 / #110 stages 2+), not a hardware defect:

1. **Quantization must run under `rnd_conv_even`.** The hardware default is `rnd_floor` (P1);
   OCP MX v1.0 specifies round-to-nearest-even. The mode is exact when set (P1), so this is one
   `set_rnd` per conversion kernel.
2. **`e = 255` is outside MXINT8's domain.** The hardware represents Inf (`m = ±64`) and NaN
   (`m = ±96`) at exponent 255 (P3); OCP E8M0 reserves 255 as NaN and MXINT8 has no Inf. The
   MX → `bfp16ebs8` direction never produces such blocks; any path accepting raw `bfp16ebs8`
   must reject or canonicalize them before claiming MX semantics.
3. **Exactness is proven for integer-exact accumulations.** The P4/P5 dot products keep every
   partial sum exactly representable in FP32 (products ≤ 2^14, ≤ 32 terms), so accumulation
   order cannot affect them — the same bounded-envelope discipline the exact INT8 tier uses.
   A tier admitting larger K must either re-prove order-independence for its envelope or bound
   K accordingly. FP32 accumulation is confirmed (`accfloat`, P4); the accumulation *order* for
   general values is deliberately not part of this verdict.

Additional tier-relevant facts: conversion out of `bfp16ebs8` is matrix-unit math (identity
MMUL), not a cast (§1); the converter's normalization matches OCP MX v1.0's scale rule on every
probed case (P2); and subnormal FP32 inputs flush to zero on conversion (P2), which an MX
quantizer reference must model.

**Bonus finding.** The conversion path `vector<float,64> → accum<accfloat,64> →
to_v64bfp16ebs8` compiles and runs on the first attempt through the standalone probe pipeline,
which validates the P4/P5 plan of crafting raw planes host-side: the layout is now known well
enough to construct arbitrary mantissa/exponent combinations without the converter.

## 7. Out of scope for #146

Sparse block vectors; `bfp16ebs16`; performance of any kind; TOSA schema or protocol work
(#110 stages 2+); any advertisement or capability row (#148 prototypes only after this ticket's
verdict).
