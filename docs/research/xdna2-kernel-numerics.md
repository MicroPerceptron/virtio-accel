# Research: which numerics the fork's XDNA2 kernel paths honestly implement

- **Ticket:** [MicroPerceptron/virtio-accel#80](https://github.com/MicroPerceptron/virtio-accel/issues/80) (wayfinder child of #78, feeding the XDNA backend effort in #75)
- **Date:** 2026-08-17
- **Tree inspected:** [MicroPerceptron/amd-npu-compiler](https://github.com/MicroPerceptron/amd-npu-compiler) (fork of Xilinx/mlir-aie) at commit `c95544269f0c074d6d3e213ee43cc34dc4100801`
- **Fork divergence:** commit `c9554426` resolves in upstream `Xilinx/mlir-aie` as well (verified via `gh api repos/Xilinx/mlir-aie/commits/c9554426…`), i.e. the fork's `main` HEAD is an upstream commit. Everything below therefore describes upstream mlir-aie semantics too; the fork adds no divergent numerics code at this commit.

Scope: the XDNA2 target — `npu2` device (AIE2P / "AIE-ML v2" cores; Strix, Strix Halo, Krackan Point SoCs per `docs/Devices.md` in the fork). File paths below are relative to the fork root at the commit above.

---

## 1. Passthrough / DMA identity designs (no compute)

| Design | Data moved | Compute | npu2 support |
|---|---|---|---|
| `programming_examples/basic/passthrough_dmas/` | `np.int32` vectors (`passthrough_dmas.py:39-40`) | none — pure shim-DMA loopback | yes (devicename=npu2 make flow) |
| `programming_examples/basic/memcpy/` | `np.int32` (`memcpy.py:39`) | none / optional passthrough kernel | yes |
| `programming_examples/basic/passthrough_kernel/` | `np.uint8` lines (`passthrough_kernel.py:37`) | copy kernel `aie_kernels/generic/passThrough.cc` (vectorized byte copy) | yes |
| `programming_examples/basic/passthrough_pykernel/` | `np.uint8` | Python-defined copy kernel | yes |
| `programming_examples/ml/block_datatypes/vector_passthrough/` | bfp16 block data | none (movement of block-FP payloads) | npu2-only example family |

**Verdict.** DMA identity paths move bytes; element type is a host-side reinterpretation. The examples happen to use i32/ui8, but nothing in the DMA/BD path inspects element semantics. An IDENTITY/passthrough tier is honest for **any** dtype, including FP32/FP16 payloads, because no arithmetic touches the values.

## 2. Matrix-multiplication designs

### 2.1 What the designs accept

The dtype universe of the IRON Python layer is closed: `python/iron/dtype.py:13-17` maps exactly `{bf16, i8, i16, f32, i32}` (bf16 = `ml_dtypes.bfloat16`, not `np.float16`). **`f16` is not expressible anywhere in the stack** — not in the dtype map, not in the kernel factories, not in any example. The matmul kernel factory `python/iron/kernels/linalg.py` registers exactly seven combos: `i8_i8`, `i8_i16`, `i8_i32`, `i16_i16`, `i16_i32`, `bf16_bf16`, `bf16_f32` (`linalg.py:15-18, 50-55`) — mirroring the kernel table in section 2.2.

Both `single_core` and `whole_array` expose exactly:

- `--dtype_in` ∈ {`bf16`, `i8`, `i16`} — `single_core/single_core.py:207`, `whole_array/whole_array.py:458`
- `--dtype_out` ∈ {`bf16`, `i8`, `i16`, `f32`, `i32`} — `single_core.py:209-213`, `whole_array.py:460-463`
- assertions: input and output must both be integral or both float; output width ≥ input width (`single_core.py:72-77`, `whole_array.py:82-87`)

**There is no FP32 or FP16 input path.** `f32` appears only as an *output* type: the FP32 accumulator of a bf16×bf16 product stored to memory unconverted. This matches upstream Xilinx/mlir-aie (same `choices` list verified in upstream `single_core.py` via `gh api`), so it is an ecosystem-wide boundary, not a fork restriction.

`matrix_vector/` is hard-coded i16→i32 (`matrix_vector.py:57-58`) and marked *work in progress* in `matrix_multiplication/README.md`. `cascade/` (K-reduction over the cascade streams) is *"Currently scalar-only"* per the same README.

### 2.2 Kernel combos actually instantiated for AIE2P

From `aie_kernels/aie2p/mm.cc` (combos at lines 437–487; helpers at 260–412). The kernel is a 2×2-expanded `aie::mmul<r,s,t,T_in,T_in,accauto>` (line 85).

| Path | In (A=B) | Accumulator | Out | MMUL tile r×s×t | Native or emulated | Evidence |
|---|---|---|---|---|---|---|
| `matmul_vectorized_…_i8_i8` | int8 | integer acc (`accauto`) | int8 | 8×8×8 | native | `mm.cc:368-381,437` |
| `…_i8_i16` | int8 | integer acc | int16 | 8×8×8 | native | `mm.cc:384-397,441` |
| `…_i8_i32` | int8 | integer acc | int32 | 8×8×8 | native | `mm.cc:401-414,445` |
| `…_i16_i16` | int16 | integer acc | int16 | 4×4×8 | native | `mm.cc:260-272,449` |
| `…_i16_i32` | int16 | integer acc | int32 | 4×4×8 | native | `mm.cc:277-289,453` |
| `…_bf16_bf16` | bf16 | **FP32** (`accfloat` via `accauto`) | bf16 | 4×8×8 | native bf16 MAC | `mm.cc:295-307,461,486` |
| `…_bf16_f32` | bf16 | **FP32** | f32 | 4×8×8 | native bf16 MAC, FP32 store | `mm.cc:333-345,469,487` |
| `…_bf16_bf16` (bfp16 emulation) | bf16→bfp16ebs8 | FP32 | bf16 | 8×8×8 | **emulated** (block-FP) | `mm.cc:458-459`, gate `AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16` |
| `…_bf16_f32` (bfp16 emulation) | bf16→bfp16ebs8 | FP32 | f32 | 8×8×8 | **emulated** (block-FP) | `mm.cc:466-467` |

There is **no f32 and no f16 combo anywhere in `aie_kernels/aie2p/mm.cc`** (nor in `aie_kernels/aie2/mm.cc` beyond the same families).

Block-FP-native kernels also exist for npu2: `aie_kernels/aie2p/mm_bfp.cc` (bfp16ebs8 in/out, `accfloat` accumulation, e.g. `zero_vectorized_v64bfp16ebs8` at `mm_bfp.cc:12-24`) and `mm_bfp_mixed.cc` (mixed A/B dtypes with an A-side conversion). The `programming_examples/ml/block_datatypes/` family (bfp_conversion, gemm_asymmetric_tile_buffering) exercises these at npu2-only, e.g. GEMM configs `bf16/bfp16ebs8/bf16/bf16` at 4096×4096×2048 (`gemm_asymmetric_tile_buffering/README.md:79-81`).

### 2.3 Shapes/tilings actually tested on npu2

Lit tests are gated `REQUIRES: ryzen_ai_npu2, peano` + `REQUIRES: makefile_examples` (feature enabled on any POSIX host with `make`, `python/aie_lit_utils/lit_config_helpers.py:108-112`). CI does invoke them: the programming-examples tree is registered as the `check-reference-designs` lit suite (`programming_examples/CMakeLists.txt:131-136`), and `.github/workflows/buildAndTestRyzenAI.yml:219-224` runs it on npu2 runners (`expected_npu: npu2`, line 63) in two slices — `refs` with `LIT_FILTER_OUT=$GEMM` and `atb` with `LIT_FILTER=$GEMM`, where `GEMM=gemm_asymmetric_tile_buffering` (line 207). One inventory pass reported "matmul examples have no CI coverage"; that is contradicted by the workflow wiring above and is recorded here as resolved in favor of the wiring:

| Test | Combo | M×K×N | m×k×n tile | Extra |
|---|---|---|---|---|
| `single_core/tests/run_strix_makefile_default.lit` | i16→i32 (defaults) | 512³ (Makefile defaults) | 32³ defaults | + trace |
| `single_core/tests/run_strix_makefile_i8.lit` | i8→i8 | 512³ | 64×128×64 | + trace |
| `single_core/tests/run_strix_makefile_bf16.lit` | bf16→bf16, emulation=0 | 512³ | 32³ | + trace |
| `single_core/tests/run_strix_makefile_bf16_emulated.lit` | bf16, emulation=1 | 512³ | 32³ | |
| `whole_array/tests/run_strix_makefile_default.lit` | i16→i32 | 512³ | defaults | 4 cols |
| `whole_array/tests/run_strix_makefile_{1,2,8}_col.lit` | i16→i32 | 512³ | defaults | 1/2/8 columns |
| `whole_array/tests/run_strix_makefile_4_col_i8.lit` | i8→i8 | 512³ | 64×128×64 | 4 cols |
| `whole_array/tests/run_strix_makefile_bf16*.lit` | bf16→bf16 (native + emulated, c-col-maj) | 512³ | 32³ | |
| `whole_array/tests/run_strix_makefile_i16_i32_c_col_maj.lit` | i16→i32 | 512³ | defaults | col-maj C |

### 2.4 Rounding / conversion behavior at boundaries

- Output store uses `C.to_vector<T_out>()` **without** an SRS shift for float paths (`mm.cc:198-207`; the shift-form is present but commented out at `mm.cc:198`). bf16 output therefore takes the FP32 accumulator → bf16 conversion of the AIE API under the core's rounding-mode register.
- The bfp16-emulation path documents the rounding hazard explicitly (`mm.cc:89-102`): each bf16→bfp16 tile conversion follows the core rounding mode, whose **default is `rounding_mode::floor`** (bias accumulates over the K reduction); the kernel swaps to **`conv_even` (round-to-nearest-even)** for the duration and restores the caller's mode (`mm.cc:98-101`, `226-228`).
- **The native bf16 path inherits the ambient core rounding mode** for the FP32→bf16 output conversion; only the bfp16-emulation path forces `conv_even`. Since the documented default is `floor` (`mm.cc:91-92`), un-managed bf16 output conversion rounds toward negative infinity — a numerics caveat the backend must own (set the rounding mode explicitly or document it).
- Integer paths are verified **bit-exact** against the host reference (`get_abs_tol`/`get_rel_tol` return `0.0` for int8/int16/int32, `common.h:237-278`); float paths use **abs tol 0.5, rel tol 0.05** for both `bfloat16_t` and `float` outputs (`common.h:246-253, 271-278`). The README's "Note on Numerical Tolerances" spells out that bf16 designs accumulate in FP32 ("when multiplying `bfloat16` numbers, the AI Engine accumulates results in higher-precision `float32`"), that truncation to a lower-precision output happens once per k-tile writeback, and that the host reference is computed in float32 (int64 for integer paths, `single_core.py:224-229`).

## 3. Pooling / reduction kernels (NHWC MAX_POOL2D)

- **No max-pooling kernel exists.** `grep -rn "max_pool|maxpool|MaxPool|pooling"` over `aie_kernels/`, `programming_examples/`, `include/`, `lib/` hits only `programming_examples/ml/mobilenet/` — and that is a fused **global average pool** inside the int8 MobileNet pipeline (`network_spec.py:40`), not MAX_POOL2D.
- The IRON reduction factory layer confirms the boundary: `python/iron/kernels/reduce.py` offers `reduce_add` and `reduce_min` for **int32 only** (raises `ValueError` otherwise, `reduce.py:24-31`), and `reduce_max` for **int32 or bfloat16** (`reduce.py:78-89`); `python/iron/kernels/vision.py` is color-conversion kernels only, no pooling.
- Nearest primitive: `programming_examples/basic/vector_reduce_max/` — a 1-D whole-tensor max reduction, single-core/single-column/multi-column designs, dtypes **int32 and bfloat16** only (`single_core_designs/vector_reduce_max.py:9-10,40`), backed by `aie_kernels/aie2/reduce_max.cc` (`reduce_max_vector`, `reduce_max_vector_bfloat16`, vectorized `aie::max` + shuffle-reduce at lines 17-51). Strix lit tests exist (`single_core_designs/run_strix.lit`, `run_strix_makefile.lit`).
- Verdict: NHWC MAX_POOL2D would be **new kernel work** (windowed max over H×W with strides, not a global reduce), though the `aie::max` vector primitive and the reduce_max scaffolding show the ingredients exist for i32/bf16 (and integer types generally). The conv pipeline's row-streaming templates (`aie_kernels/aie2/conv2dk*.cc`) are the natural scaffold for a windowed kernel.

## 4. Hardware ground truth: AIE-ML v2 (AIE2P) vector datapath

| Element type | Vector MAC support on XDNA2 / AIE-ML v2 | Source |
|---|---|---|
| int4 / int8 / int16 | native (highest throughput) | [AM020 AIE-ML Architecture Manual](https://docs.amd.com/r/en-US/am020-versal-aie-ml); AMD IEEE Micro paper "[AMD XDNA NPU in Ryzen AI Processors](https://dl.acm.org/doi/10.1109/MM.2024.3423692)" (XDNA/XDNA2 natively support int8, int16, bf16) |
| int32 | supported at reduced rate (no 32-bit native vector manipulation tier of original AIE) | AM020 |
| bfloat16 | **native** MAC, accumulation in FP32 (`accfloat`) | AM020; [AIE API mmul docs](https://xilinx.github.io/aie_api/group__group__mmul.html); `aie_kernels/aie2p/mm.cc` |
| bfp16ebs8 (Block FP16, shared 8-bit exponent per 8 mantissas) | **native on XDNA2 only** — AMD: "first NPU supporting advanced **Block FP16**" | [AMD Computex 2024 press release](https://www.amd.com/en/newsroom/press-releases/2024-6-2-amd-extends-ai-and-high-performance-leadership-in-.html); `mm_bfp.cc`, `block_datatypes/` examples |
| **FP32** | **not native in the vector MAC datapath** — "float multiplications are **emulated** on AIE-ML/XDNA 1 and XDNA 2 using native bfloat16 multiplications" | [AIE API mmul docs](https://xilinx.github.io/aie_api/group__group__mmul.html) (exact wording); AM020 (AIE-ML "lacks support for … IEEE floating point" in the native vector tier) |
| **FP16** | **no `aie::mmul` support on any AIE architecture** — the AIE API mmul tables contain no fp16/half rows for AIE, AIE-ML, or XDNA2 | [AIE API mmul docs](https://xilinx.github.io/aie_api/group__group__mmul.html) |

Notes:
- The scalar processor can execute FP32 arithmetic, but that is the scalar path (orders of magnitude below the vector MAC array) — not something a matmul/conv tier can honestly be advertised on.
- The AIE-ML v2 architecture manual is [AM027](https://docs.amd.com/r/en-US/am027-versal-aie-ml-v2); its deep sections sit behind a JS portal and were not fully extracted here (see gaps), but the AIE API tables and the AIE2P kernel sources in-tree are dispositive for what the compiler stack can emit.
- Device model corroboration in-tree: `npu2` is the full Strix-class part (`NPU2TargetModel`, `include/aie/Dialect/AIE/IR/AIETargetModel.h:898,935`), and the target model's supported block types are exactly `{v8bfp16ebs8, v16bfp16ebs16}` (`lib/Dialect/AIE/IR/AIETargetModel.cpp:1618`) — block FP is a first-class npu2 type; IEEE FP16 appears nowhere in the device model either.
- Two in-tree inconsistencies noticed in passing (not numerics-tier blockers, but worth upstream issues): (a) the aie2 (npu1) `matmul_vectorized_4x8x8_i8_i16` / `_i8_i32` wrappers drop the `is_c_row_maj` template argument, so `C_COL_MAJ` is silently ignored for those combos (`aie_kernels/aie2/mm.cc:880-909`; the aie2p kernel passes it correctly); (b) `getComputeTileLoadStoreBusWidth()` returns 256 for the AIE2 family with no npu2 override (`AIETargetModel.h:655`) while aie2p kernels issue 512-bit vector stores (`aie_kernels/aie2p/mm_bfp_mixed.cc:12-20`).

## 5. Implication for the numerical-tier decision (issue #82)

**The shared TOSA FP32/FP16 corpus cannot be honestly advertised on this hardware.**

- **FP16:** no vector datapath, no kernels, no design flags — nothing in the tree or the hardware accepts f16. Any "FP16 support" would be a silent cast to bf16 (10→7 mantissa bits): exactly the relabeling issue #75 forbids.
- **FP32:** the only FP32 the hardware does natively is *accumulation* of bf16 products. An "FP32 matmul" would truncate inputs from 24→8 mantissa bits (bf16) before the multiply — silent precision reduction. The AIE API's own wording ("emulated … using native bfloat16 multiplications") makes this unambiguous.
- **What is honest, today, with tested kernels on npu2:**
  1. **Integer tier** — i8→{i8,i16,i32} and i16→{i16,i32} matmul, bit-exact verification, npu2 lit-tested. Aligns with TOSA PRO-INT.
  2. **BF16 tier** — bf16×bf16 with FP32 accumulation, output bf16 or f32, npu2 lit-tested (native and bfp16-emulated variants). This is precisely the [TOSA BF16 extension (EXT-BF16)](https://www.mlplatform.org/tosa/tosa_spec.html) profile shape, and would need **new BF16 fixtures** rather than the shared FP32/FP16 corpus.
  3. **Passthrough/IDENTITY** — dtype-agnostic at the byte level; honest for any element type.
- Two additional honesty constraints for the backend:
  - The **bfp16 emulation flag** changes numerics (block-shared exponent quantization, conv_even conversion rounding). It must remain an explicit, surfaced mode — never silently enabled under a "bf16" label.
  - `dtype_out=f32` on a bf16 matmul is an *accumulator format*, not FP32 compute; it must not be advertised as FP32 GEMM.
- MAX_POOL2D (any dtype) is **not implementable from existing kernels** without new windowed-max kernel work; the closest tested primitive is a global 1-D reduce-max (i32/bf16).

## Confidence and gaps

- **High confidence (read directly from the tree at `c9554426`):** design dtype boundaries (argparse + asserts), the closed IRON dtype map (`python/iron/dtype.py`), aie2p mm.cc combo list and MMUL shapes, npu2 lit-test matrix and its CI wiring via `check-reference-designs`, tolerance values in `common.h`, rounding-mode handling (floor default, conv_even under bfp16 emulation), absence of pooling kernels, absence of any f16/f32-input kernel, the AIE-API "float emulated via bfloat16 on XDNA1/2" statement, fork HEAD == upstream commit.
- **Medium confidence / unverified detail:**
  - Exact integer accumulator widths behind `accauto` (acc32 vs acc64 per combo) were not read out of the `third_party/aie_api` headers (uninitialized submodule in the shallow clone); the AIE API docs are the reference.
  - AM027's internal datatype/throughput tables were not extracted (JS documentation portal); the FP32/FP16 conclusions rest on the AIE API tables, AM020, AMD's XDNA press/paper material, and the in-tree kernel + device-model evidence, which all agree.
  - "Tested" means lit tests wired into `check-reference-designs` on `ryzen_ai_npu2`-gated CI runners; no board runs were performed for this research, and per-run CI logs were not audited (an earlier inventory pass claimed the matmul examples never run in CI; the workflow wiring contradicts it, but only a green npu2 CI log would settle it beyond doubt).
  - `mm_activation_epilogue.cc`, `conv2dk14.cc`, and the wider ml/ example set were not inventoried in depth (out of scope for the matmul/pooling/passthrough tier question).
- **Process note:** two delegated inventory passes completed but their sessions could not be resumed to retrieve full reports; every relayed claim used here was independently re-verified against the tree before inclusion, except where explicitly marked otherwise.
