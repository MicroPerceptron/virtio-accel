# 10. Vulkan kernel throughput: whole-word lane stores, register-tiled MATMUL, a skinny geometry, and a benchmark

- Status: accepted (implemented in two rounds; measured on Intel Arc (Panther Lake, Mesa 26.0.8
  ANV, Vulkan 1.4.335) on 2026-09-18, with the full device suite passing on ANV and llvmpipe,
  every kernel variant passing `spirv-val`, and the suite clean under the Khronos validation
  layer)
- Extends: ADR 0007 (kernel mechanics and the MATMUL bit-identity property), ADR 0008 and
  ADR 0009 (whose packed FP16 and FP8 lanes this makes fast)
- Resolves: the first performance pass over the FP8 tier — the paths ADR 0009 made correct
  without measuring

## Context

ADR 0009 shipped the FP8 tier with a correctness argument and no throughput evidence. The tier's
value is bandwidth: FP8 weights are a quarter of FP32, so an FP8 layer should stream in a quarter
of the time. Nothing in the crate measured whether it did. A benchmark was written first, and it
found three inefficiencies, none specific to FP8 but all paid most heavily there.

**Sub-word stores were read-modify-write atomics.** Every `BOOL`, FP16 and FP8 lane was written
through `OpAtomicAnd` then `OpAtomicOr` on its containing word — the neighbour-safe sequence
ADR 0007 chose so a predicate output never touches an adjacent byte — with one invocation per
element, so four invocations contended for every FP8 word. FP8 `IDENTITY` over 16 Mi elements
ran at 7.3 GB/s where FP32 ran at 110; FP16 → FP8 `CAST` at 15 GB/s.

**MATMUL computed one output per invocation.** Two shared-memory loads per multiply-add, and an
operand staged one element per invocation per `k` step, so the ~25-instruction FP8 widening was
paid once per multiply-add. A 1024³ FP8 MATMUL ran at 572 GFLOP/s, *below* the FP32 kernel's
785, on a device whose FP32 rate itself was well under what the arithmetic allows.

**The decode shape had no kernel.** A few activation rows against a 4096 × 4096 weight matrix —
the shape FP8 weights exist for — idled 15 of 16 rows of every workgroup and took 0.98 ms for
16 MiB of FP8 weights: 17 GB/s, the same wall time as FP32 over four times the bytes.

Two constraints from earlier ADRs hold throughout. Every MATMUL element must stay bit-identical
to the sequential ascending-`k` sum with separately rounded multiply and add (ADR 0007), which
rules out split-`k` reductions and fused multiply-add. And a sub-word store must never modify a
byte outside the tensor (ADR 0007), which rules out a plain read-modify-write of a tensor's final
partial word.

## Decision

1. **A benchmark in the crate, run before and after every kernel change.**
   `benches/vulkan_fp8.rs` (`cargo bench -p virtio-accel-vulkan`) times whole TOSA graphs
   submit-to-fence through the public `Accelerator` surface: the submission floor, `IDENTITY` at
   every float storage, every `CAST` direction, square MATMUL at FP8/FP16/FP32 beside the
   dequantize-then-multiply graphs, and the one- and eight-row GEMV against a 4096 × 4096 weight
   matrix. It is a `harness = false` binary rather than Criterion: the workspace denies duplicate
   dependency versions and gates licenses, and a GPU submission is timed by its fence. It warms
   each case for at least 300 ms because an integrated Intel GPU ramps its clock over hundreds of
   milliseconds — without that floor the first heavy case after a light one reported bimodal
   samples with a median 1.7× its minimum, and an early run attributed a 2× gap to the E4M3
   encoding that was entirely clock ramp.

2. **Contiguous sub-word kernels own a destination word per invocation.** `CAST` and contiguous
   `IDENTITY`/`RESHAPE` copies load the source lanes of one destination word (each source word
   once, including the FP8 pair inside one word that FP8 → FP16 needs), convert, pack, and store
   with a plain `OpStore`. Only a tensor's final partial word still goes through the atomic
   sequence, lane by lane, so the neighbour guarantee is unchanged. Lowering dispatches words
   rather than elements. Strided moves (`TRANSPOSE`, `REVERSE`, `CONCAT` segments) keep the
   per-element path; they were not measured to be on any hot path yet.

3. **MATMUL is register-tiled.** Each invocation of a `tile × tile` workgroup accumulates a
   4 × 4 register block, so the workgroup covers a 64 × 64 output square (32 × 32 at the 8-wide
   fallback tile). Per 16-deep `k` step the workgroup stages a 64 × 16 slab of each operand
   cooperatively — four elements per invocation per operand, consecutive invocations at
   consecutive addresses, the lhs slab stored transposed so its inner-loop reads are
   conflict-free — then issues eight shared loads per sixteen multiply-adds. Rows and columns
   are interleaved (`i · tile + ty`, `j · tile + tx`) so a row of invocations reads and writes
   consecutive addresses. Accumulation order is unchanged, so bit identity holds by
   construction and the existing test, widened to shapes that cross the block with a partial
   `k` step, confirms it.

4. **A flat geometry for skinny shapes.** The kernel is driven by a `MatmulGeometry`
   (invocation grid, register micro-block, slab depth), and lowering selects a second geometry
   for `m ≤ 8`: the same 256 invocations laid out 64 wide by 4 tall, two rows each, one column
   each, 32 deep — an 8 × 64 block where every invocation stages eight weight elements per step.
   A 4096-wide layer gets 64 workgroups of 256 invocations all streaming the weight slab.

5. **Sub-word slabs are staged a storage word per invocation.** After the geometry work, FP8,
   FP16 and FP32 GEMV still took the same wall time, and a diagnostic run with the inner loop
   removed showed why: the FP8 kernel spent 0.41 ms above the floor on staging alone while FP32
   hit memory speed with the identical instruction structure. The memory pipeline was bound by
   *load instructions*, not bytes, and every FP8 lane issued one load per element to use one
   byte of the word it fetched. Staging is now a shared routine over a slab description that,
   for sub-word storage, loads one word per invocation — four FP8 or two FP16 elements from one
   `OpLoad` — whenever the operand's row stride is a multiple of the lanes per word. The check
   is on a specialization constant, so the driver folds it and only one path survives pipeline
   creation; the per-element path remains for unaligned strides and for FP32. This is the step
   that made FP8 *faster* than the wider storages rather than merely smaller.

6. **Widening by exponent offset.** `widen_fp8` and `widen_f16` place the encoding's magnitude
   bits under a fixed binary32 exponent `K = 127 − bias`, so a normal reads off directly and a
   subnormal is `(x − 2^(1 − bias)) · 2`, two exact operations in one binade; specials are one
   compare and one select. Half the instructions of the integer-only expansion, still no
   `OpFConvert`, and no binary32 denormal is ever formed. It measured no change on Xe3 (the
   staging ALU was never the bound) and is kept for being smaller. The exhaustive 256-pattern
   `CAST` test caught the one bug in it — the exponent field and `K` overlap, so the placement
   is an add, not an or — at pattern `0x40`.

7. **The skinny inner loop is unrolled; the wide one is not.** A full 32-deep step of the
   skinny geometry runs unrolled (0.47 → 0.42 ms on the decode GEMV); the same treatment of the
   wide geometry's sixteen multiply-adds per iteration slowed 1024³ by a quarter from register
   pressure, so unrolling is a `MatmulGeometry` property.

8. **Rejected alternatives, recorded because they were measured.** A one-column-per-invocation
   1-D GEMV kernel streams FP32 weights at 84 GB/s but leaves only 4096 lanes to widen FP8 and
   FP16 operands, and those ran three times *slower* than the square block; the widening must
   be spread over the whole workgroup, which the flat geometry does. A 256-entry shared-memory
   FP8 widening table, built per workgroup, was within noise of the inline expansion (1495 vs
   1409 GFLOP/s at 1024³, GEMV unchanged) and was dropped: nothing to build, no barrier to wait
   on. A 64-deep skinny slab was slower than 32 (0.84 vs 0.73 ms). Register prefetch of the
   next step's slab (software pipelining) left GEMV unchanged and slowed the square MATMUL by
   a tenth from the extra live registers; the kernel was not latency-bound in that way.

## Evidence

Intel Arc (Panther Lake), `Device` domain, median of 30 timed submissions after warm-up, 16 Mi
elements for the elementwise cases. "Before" is the kernels as of ADR 0009 under the same
harness on the same day.

| Case | Before | After |
|---|---:|---:|
| `IDENTITY` FP8 | 4.60 ms, 7.3 GB/s | 0.37 ms, 92 GB/s |
| `IDENTITY` FP16 | 1.70 ms, 39 GB/s | 0.65 ms, 103 GB/s |
| `IDENTITY` FP32 (unchanged path) | 1.22 ms, 110 GB/s | 1.23 ms, 109 GB/s |
| `CAST` FP8 → FP16 | 1.67 ms, 30 GB/s | 0.55 ms, 91 GB/s |
| `CAST` FP16 → FP8 | 3.27 ms, 15 GB/s | 0.52 ms, 97 GB/s |
| `CAST` FP32 → FP8 | 3.27 ms, 26 GB/s | 0.80 ms, 104 GB/s |
| `CAST` FP32 → FP16 | 1.88 ms, 54 GB/s | 0.95 ms, 106 GB/s |
| `MATMUL` FP8 512³ | 0.66 ms, 408 GFLOP/s | 0.43 ms, 625 GFLOP/s |
| `MATMUL` FP8 1024³ | 3.76 ms, 572 GFLOP/s | 1.43 ms, 1503 GFLOP/s |
| `MATMUL` FP16 1024³ | 2.86 ms, 751 GFLOP/s | 1.14 ms, 1882 GFLOP/s |
| `MATMUL` FP32 1024³ | 2.74 ms, 785 GFLOP/s | 1.11 ms, 1928 GFLOP/s |
| `CAST` FP8 → FP16 + `MATMUL` FP16 1024³ | 3.24 ms | 1.28 ms |
| GEMV FP8 1 × 4096 × 4096 | 0.98 ms, 17 GB/s weights | 0.44 ms, 39 GB/s weights |
| GEMV FP16 1 × 4096 × 4096 | 1.06 ms, 32 GB/s weights | 0.53 ms, 63 GB/s weights |
| GEMV FP32 1 × 4096 × 4096 | 1.06 ms, 63 GB/s weights | 0.72 ms, 93 GB/s weights |
| GEMV FP8 8 × 4096 × 4096 | 1.01 ms | 0.46 ms |
| `CAST` FP8 → FP16 + GEMV FP16 1 × 4096 × 4096 | 2.73 ms | 0.95 ms |

The GB/s figures count bytes read plus written; GEMV "weights" figures count the weight matrix
alone. Every number includes the submission floor, measured at 100–170 µs across runs, which
is why the GEMV weight rates understate the kernels: net of a 170 µs floor the FP8 GEMV
streams its 16 MiB at about 63 GB/s and the FP32 one its 64 MiB at about 120 GB/s. The 1024³
cases vary by about ±15% run to run on this device even at 30 samples; the elementwise and
GEMV cases are stable to a few percent.

The exhaustive 256-pattern FP8 identity, the FP8 `CAST` round trips, the `BOOL` neighbour-safety
tests, the MATMUL bit-identity test over nine shapes (including 65 × 70 × 130, 1 × 300 × 257
and 8 × 129 × 65), and the FP8 MATMUL reference test over three shapes pass on ANV and llvmpipe
in every advertised memory domain; all 176 kernel variants pass `spirv-val --target-env
vulkan1.3`; the device suite runs clean under `VK_LAYER_KHRONOS_validation`.

## Consequences

- The FP8 tier's bandwidth claim now holds for data movement: FP8 copies and casts run at the
  device's copy rate, which is what a quarter-width storage format is for.
- For GEMV the order is now the right one — FP8 0.44 ms, FP16 0.53, FP32 0.72 — but FP8 is
  not yet at the memory bound: net of the floor it streams weights at about half the rate the
  FP32 kernel demonstrates. What remains is per-step fixed cost (two barriers and a shared
  round trip per 32 `k`) that bytes do not amortize; the constraint that closes the obvious
  door (no split-`k`) is ADR 0007's, and reopening it is a numerics decision, not a
  performance one. ADR 0011 takes that decision: it replaces the flat geometry of §4 with a
  split-`k` streaming kernel and fuses the multiply-add in both kernels.
- The square MATMUL runs at about 1.5–1.9 TFLOP/s against an arithmetic ceiling near half the
  device's fused-multiply-add peak (every multiply-add is two separately rounded instructions
  by ADR 0007; ADR 0011 fuses them). Larger micro-blocks trade occupancy for shared-load
  pressure and were not explored.
- Strided moves over sub-word storage, the FP16 output store of MATMUL, and `BOOL` elementwise
  outputs still take the per-element atomic path. None is on a measured hot path.
- Every kernel change from here has a number to beat, and the number is taken on metal.
