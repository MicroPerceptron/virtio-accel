# Issue #148 — AMD `bfp16ebs8` vendor-tier prototype: design

Non-normative plan. Builds on the #146 verdict
([characterization note](../research/amdxdna-bfp16ebs8-characterization.md), §6a) and stays
inside the #148 boundaries: backend-local, no TOSA artifact format, no
`TosaCapabilityProvider` row, guest-visible immutable label, offline compilation, direct
binding, rejection before native resource creation.

## What the prototype exposes

One operation, the one the probes proved: **block-scaled MATMUL with MXINT8 semantics on the
`bfp16ebs8` decomposition** — `C[M,N] (FP32) = A[M,K] · B[N,K]ᵀ`, where A and B are streams of
32-element MXINT8 groups (one E8M0 scale byte + 32 int8 mantissas per group), executed as four
equal-exponent block-8 groups per the proven mapping. Conversion tiers and the native
per-8-block form wait until this lifecycle is proven end to end; the artifact format leaves
room for both.

## Numerical contract (guest-visible, immutable)

- Element semantics: `value = m · 2^(e − 127 − 6)`, two's-complement int8 `m`, E8M0 scale `e`
  (bias 127). Exactly the #146-verified contract.
- Scale bytes `e = 255` are rejected at load (MXINT8 has no Inf/NaN; hardware would compute
  structural Inf/NaN semantics the label does not promise).
- Accumulation: FP32, in the **documented order** — within each 32-group, the four block-8
  MACs chain in ascending k; groups accumulate in ascending k. The conformance oracle is an
  FP32 fold in that exact order (`model.rs::dot_reference` generalized to fold order), so the
  contract is bit-exact by construction rather than tolerance-bounded, without overclaiming
  order-independence.
- Quantization is the guest's job. The tier consumes already-quantized planes; it never
  invokes the hardware converter on guest data (converter-vs-OCP boundary divergence,
  characterization §4).

## Envelope (initial, provable)

`M = 8`, `N = 8`, `K ∈ {32, 64, …, 512}` (multiples of 32). One AIE2P worker, the proven
`mul_8x8_8x8T`/`mac_8x8_8x8T` chain. Local-memory footprint at K=512: A 8×512 mantissas +
scales = 4,224 B, B likewise, C 256 B — comfortably inside the one-core envelope. Larger M/N
tiling is #148 follow-up, not the first slice; the point of the prototype is the lifecycle,
oracle, and label discipline, not throughput (that lesson is #149/#151's).

## Experimental artifact container

New backend-local format constant `XDNA_BFP_EXPERIMENT_FORMAT` (distinct nonzero u32,
`0x5842_4650` "XBFP"), version 1, parsed by a new `bfp_experiment` module in
`virtio-accel-xdna`:

```
magic "XBFP" | version u32 | flavor u32 (1 = MXINT8-on-block8 MATMUL) |
m u32 | k u32 | n u32 |
xclbin_len u64 | insts_len u64 | xclbin | insts
```

Slot plan derived, not self-declared: slot 0 = A planes (`m·k` mantissas + `m·k/32` scale
bytes, layout documented), slot 1 = B planes, slot 2 = C (`m·n·4` bytes). Parser rejects
unknown magic/version/flavor, non-envelope shapes, and length mismatches before any HRX call —
mirroring `artifact.rs`'s checked framing. The target identity carried in `ArtifactRef` must be
the experiment's own constant identity; anything else is `Incompatible`.

Scale-byte validation (`e != 255` for every group) happens at load, scanning the… **not
possible at load** — planes arrive at submit time through buffers. Correction: the *kernel*
cannot cheaply reject, so the contract documents that `e = 255` input is guest error with
unspecified-but-safe results (the hardware computes finite/Inf FP32; no memory unsafety), and
the conformance suite pins the rejection at the reference-oracle level instead. This is the
honest option that keeps zero submission-time scanning; revisit if #110 standardization lands
a stricter obligation.

## Crate integration

- `lower.rs` is untouched: this is not TOSA admission. `load_program` gains one new format arm
  dispatching to `bfp_experiment::parse` (same shape as the `XDNA_PRECOMPILED_FORMAT` arm).
- Kernels compile offline through a new emitter **in the probe pipeline first**
  (`research/bfp16ebs8/`), promoted into `compiler/xdna_compile.py` only after #162 (the
  concurrent throughput work that owns that file) merges — avoiding a two-agent merge race on
  the serving helper.
- Oracle: `model.rs` is copied into `crates/virtio-accel-xdna/tests/bfp_model.rs` with a
  provenance header pointing at the research module (the #148 requirement is "reuse the
  characterization model exactly"; a path dev-dependency on a non-workspace research project
  would leak into the published crate's manifest).
- Docs: README gains an "AMD `bfp16ebs8` vendor experiment (block-8)" section using exactly
  the CONTEXT.md-required terminology and explaining why it is not a TOSA capability.

## Acceptance mapping (from #148)

| #148 criterion | This design |
|---|---|
| Artifact encodes every scale/value plane parameter | `XBFP` header + derived slot plan; no hidden defaults |
| Exact or tolerance-bounded oracles incl. exponent disagreement | bit-exact fold-order oracle from the #146 model; disagreement cases from P4/P5 reused |
| Full lifecycle on the NPU with direct binding | hardware test mirroring `precompiled_passthrough_runs_the_full_lifecycle` |
| Unsupported/mismatched artifacts rejected before native creation | parser + shape/envelope checks ahead of `hrx_*` calls |
| Documentation labels and disclaims | README section, CONTEXT.md vocabulary |
| No wire change, no MX-compat claim | backend-local constant, experiment-only docs |

## Sequencing

1. **Done (2026-08-28).** Kernel emitter + flavor-1 kernel at the envelope ceiling (K = 512)
   produced in `research/bfp16ebs8/` (`kernel_xbfp.cc`) and executed on the NPU by the probe
   runner: all 64 output lanes bit-exact against the fold-order oracle
   (`model.rs::dot_fold_f32`), on inputs where 51 of 64 lanes *distinguish* that oracle from a
   single-rounded f64 sum — the accumulation-order contract is silicon-proven, not assumed
   (`research/bfp16ebs8/results/xbfp-2026-08-28.txt`).
2. Crate integration (`bfp_experiment` module, `load_program` arm, tests) once #162 merges.
3. On-metal suite + README, then close #148.
