# ADR 0016: Core ML constant integer and palette weights

Status: proposed

## Context

The runtime FP8/NVFP4 paths decode storage in ordinary graph operations. Those paths
remain useful for mutable tensors but do not expose Core ML's compressed constant
operations. Apple recommends blockwise INT4 for Mac GPU and palettization for ANE;
these recommendations do not guarantee placement for every shape or palette geometry.

## Decision

No wire bytes, assigned dtype numbers, target identities or boundary layouts change.
On the existing macOS 26+ mixed integer/float tier, recognize constant weight
`CAST(INT8 -> FP16) * FP16 group_scale`, reshape and RHS transpose before FP16 MATMUL.
Emit a typed FP16 `linear` with two-dimensional compressed constants. Dense FP16
constant RHS tensors in the same mixed tier also lower to `linear`, with an exact
load-time transpose; matched controls therefore share the native operation and model format. GPU/All policies
use `constexpr_blockwise_shift_scale`, packing to signed int4 only when every value
fits [-8,7], otherwise keeping int8. CPU/ANE uses a 16-entry FP16 palette for int4
values; each entry is rounded exactly as the original FP16 multiplication requires.
No activation quantization or float-weight quantizer is introduced. Exact integer-profile
INT8/INT32 arithmetic and mutable NVFP4/FP8 tensors keep their existing paths.

Portable authoring additionally exposes `palette4_matmul_fp16`: immutable nibble indices
and explicit finite binary16 palettes per contiguous input group. This represents arbitrary
16-level values, not necessarily evenly spaced INT4 or the prescribed NVFP4 encoding.
TOSA serializes INT32 indices and GATHER, not a new packed tensor type. The Core ML
adapter proves all serialized indices are in range and emits `constexpr_lut_to_dense`.
The additive analysis helper and capability variant distinguish this proof from the
existing bounded runtime byte-lookup idiom; arbitrary dynamic indices remain rejected.
Other backends do not gain admission merely because the authoring helper exists.

Identical adjacent tables are compacted independently along both LUT axes, comparing
raw bytes (including signed zero). This changes storage only. Per-channel or shared
palettes can expose ANE placement; input-axis blocks may leave the projection on CPU.
Dead decoding ancestors are removed only after tracing all observable outputs: shared
consumers retain their producers. All transformations occur during load, outside submission.
`half` stays in the host adapter and is not added to the portable authoring crate.

## Validation and limits

Unit tests cover constant-only matching, shared outputs, packed signed nibbles, palette
compaction, malformed authoring inputs and rejection of negative/high/runtime indices.
FP16-output MATMUL retains TOSA's FP16 accumulation accuracy contract; it does not
promise bit-identical FP32/FP64 reductions. Dense fractional activation tests use a local
absolute-error bound stricter than TOSA's 2*K bound. Native basis-vector tests isolate
decoded weights from reduction order and check arbitrary
fractional scales, signed values, FP16 subnormals and all palette entries on GPU, ANE and
All policies. Existing FP8, NVFP4, exact integer and asynchronous ownership tests remain gates.
Matched Divan cases compare identical weights/activations against FP16 at M=1/16/128,
N=1024, K=2048. Channel palettes share a table across 16 output channels; input-block
palettes are separate cases so their geometry cannot be hidden in a timing comparison.

The optional model exporter plus `benchmarks/inspect_coreml.m` records Apple's anticipated
compute plan using production precision/optimization settings. It does not measure actual
engine utilization, native arithmetic precision or resident memory. On the development M4,
channel palettes permit ANE prefill placement; small linear workloads can prefer CPU even
under CPU/GPU policy. Emitting a constexpr operation alone proves neither compressed
residency nor accelerator execution. Benchmark claims must retain these limits.

W8A8 requires an explicitly calibrated activation contract and is separate work. No ANE-only
or GPU-only execution promise, native NVFP4 compression, or implicit format conversion is made.

## References

- [Apple compression overview](https://apple.github.io/coremltools/docs-guides/source/opt-overview.html)
- [Apple palettization performance](https://apple.github.io/coremltools/docs-guides/source/opt-palettization-perf.html)
- [Apple MIL compression definitions](https://github.com/apple/coremltools/blob/main/coremltools/converters/mil/mil/ops/defs/iOS18/compression.py)
