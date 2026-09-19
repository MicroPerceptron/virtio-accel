# 11. MATMUL numerics: fused multiply-add, and a split-k streaming kernel for skinny shapes

- Status: accepted (implemented; measured on Intel Arc (Panther Lake, Mesa 26.0.8 ANV, Vulkan
  1.4.335) on 2026-09-18, with the device suite passing on ANV and llvmpipe, every kernel
  variant passing `spirv-val`, and the suite clean under the Khronos validation layer)
- Extends: ADR 0010 (whose remaining GEMV gap this closes), ADR 0007 (whose MATMUL numerics
  property this narrows)
- Supersedes, for `MATMUL` only: ADR 0007's "bit-identical to the sequential ascending-`k` sum"

## Context

ADR 0007 made every `MATMUL` element bit-identical to a sequential ascending-`k` loop with
separately rounded multiply and add, and ADR 0008/0009 leaned on it: FP16 and FP8 results were
identical on every device by construction. ADR 0010 kept that property through a first
performance pass and ended with the FP8 decode GEMV at 0.44 ms for 16 MiB of weights — twice the
memory floor — with the remaining cost being per-step fixed overhead (two barriers and a shared
round trip per 32 `k`) that the property's two consequences forbade removing:

- **No fused multiply-add.** Two separately rounded instructions per product, so the
  arithmetic ceiling of the register-tiled kernel is half the device's FMA rate.
- **No split-`k`.** One invocation must walk each output's `k` in order, so a workgroup over a
  few rows has no way to put more of the device to work on the weight stream than its barrier
  cadence allows.

The property was a *choice* of numerics, not a TOSA requirement: TOSA specifies binary32
accumulation for these MATMULs and bounds the result; it does not fix the association or forbid
fusion, and the shared conformance corpus already compares `MATMUL_FP32` at a 1e-5 tolerance.
This ADR trades the property for throughput, deliberately and by a bounded amount, having been
told that a reasonable loss of bit-exactness is acceptable if it buys throughput.

## Decision

1. **Fused multiply-add in every MATMUL kernel.** The inner products use the GLSL `Fma`
   instruction decorated `NoContraction` — one rounding per product instead of two. On the
   register-tiled kernel this is the difference between sixteen and thirty-two arithmetic
   instructions per `k` step per invocation.

2. **A split-`k` streaming kernel for `m ≤ 8`.** A 1-D workgroup of 64 invocations covers 16
   output columns. Each invocation owns one weight *word* — four FP8, two FP16 or one FP32
   columns — and one of `64 / words` equal contiguous slices of `k`, and carries all eight rows
   of its columns in registers. Its loop is one `rhs` word load, one uniform binary32 lhs load
   per live row, and `rows × lanes` fused multiply-adds: no shared memory, no barrier. At the
   end every invocation writes its partials to shared memory and the workgroup reduces each
   output over the slices in ascending order, then narrows and stores. Rows at or past `m` are
   compiled out (the row test is on a specialization constant), so the one-row shape does one
   multiply-add per weight element.

3. **The lhs is pre-widened by lowering.** The streaming kernel reads the lhs as binary32 words.
   When the tier's lhs is FP16 or FP8, lowering inserts a `CAST` dispatch into an arena
   intermediate of `batch · m · k` words that lives only for that operator; the widening that
   every one of 64 invocations would otherwise repeat per `k` happens once, and the kernel's
   hot loop has no conversion in it at all. The intermediate is small next to the weights it
   pays for (128 KiB against 16 MiB on the decode shape).

4. **What is promised now.** Every MATMUL result is within `(k + 64) · 2⁻²³ · Σ|aᵢ·bᵢ|` of the
   exact sum — fused multiply-add over `k` products plus at most 64 partial sums in fixed order
   — and is *deterministic per device*: the same graph over the same bytes gives the same bits
   on every submission and in every memory domain. Results are no longer bit-identical to the
   sequential sum, and no longer identical across devices, for one measured reason: drivers
   differ on whether `Fma` under `NoContraction` is fused. On the 65 × 70 × 33 probe, ANV's
   output matches a host fused (`mul_add`) sequential reference in 2145 of 2145 elements and a
   separately rounded one in 585; lavapipe matches the separately rounded reference in 2145 of
   2145 and the fused one in 585. Both are internally consistent; they are not the same.

5. **The flat 8 × 64 geometry of ADR 0010 is retired**, replaced by the streaming kernel; the
   register-tiled geometry stays for `m > 8`.

## Evidence

Intel Arc (Panther Lake), `Device` domain, median of 20 timed submissions after warm-up.
"ADR 0010" is the branch state before this decision.

| Case | ADR 0010 | Separately rounded, split-`k` | Fused, split-`k` |
|---|---:|---:|---:|
| GEMV FP8 → FP16, 1 × 4096 × 4096 | 0.44 ms | 0.33 ms | 0.35 ms |
| GEMV FP8 → FP16, 8 × 4096 × 4096 | 0.46 ms | 0.51 ms | 0.41 ms |
| GEMV FP16, 1 × 4096 × 4096 | 0.53 ms | 0.43 ms | 0.44 ms |
| GEMV FP32, 1 × 4096 × 4096 | 0.72 ms | 0.69 ms | 0.68 ms |
| `MATMUL` FP8 → FP16, 1024³ | 1.43 ms, 1503 GFLOP/s | 1.39 ms | 1.17 ms, 1842 GFLOP/s |
| `MATMUL` FP16, 1024³ | 1.14 ms, 1882 GFLOP/s | 1.12 ms | 0.79 ms, 2712 GFLOP/s |
| `MATMUL` FP32, 1024³ | 1.11 ms, 1928 GFLOP/s | 1.13 ms | 0.97 ms, 2209 GFLOP/s |

Split-`k` alone takes the one-row FP8 GEMV from 0.44 to 0.33 ms; fusion alone takes the square
kernels 15–30% further and is what makes the eight-row GEMV faster rather than slower (its
thirty-two products per `k` were arithmetic-bound when separately rounded). Every number includes
the submission floor (100–170 µs); net of it the one-row FP8 GEMV streams its 16 MiB at about
90 GB/s, against the ~120 GB/s the FP32 kernel demonstrates for the memory system.

The MATMUL test corpus moved from bit-identity to the bound above (ten shapes across both
kernels, including `k` that does not split evenly and `n` that is not a multiple of the columns
per workgroup), the FP8 MATMUL reference tests to one binary16 ulp, and a determinism test runs
each shape three times across every memory domain. The shared corpus cases were already exact
under any association (small integers, or products whose sums fit a binary32 significand) and
pass unchanged.

## Consequences

- The FP8 tier's bandwidth claim now holds for GEMV as well as for data movement: the one-row
  decode layer is within about 25% of the memory floor, and FP8 < FP16 < FP32 in wall time.
- `MATMUL` results are deterministic per device, not identical across devices. The FP16 and FP8
  tiers' "identical on every device" statements (ADR 0008, ADR 0009) still hold for every other
  operator — the conversions are unchanged — and hold for `MATMUL` between devices whose
  drivers make the same fusion choice. A consumer who needs bit-identical results across a
  lavapipe CI lane and a real GPU has lost that for `MATMUL`; the trade was made knowingly.
- The error bound is looser than TOSA's own for these shapes only in association: fused
  multiply-add rounds *less* than the two-instruction form, so the typical error fell even as
  bit-identity was given up.
- The pre-widened lhs is the first operator-private arena intermediate lowering creates on its
  own initiative. It follows the existing lifetime and hazard machinery (the matmul's read of it
  is a barrier-ordered RAW like any other), so nothing new is trusted.
- What remains between the one-row FP8 GEMV and the memory floor is the submission floor itself
  and the reduction epilogue; neither is FP8-specific.
