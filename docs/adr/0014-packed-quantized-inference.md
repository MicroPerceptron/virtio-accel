# ADR 0014: Packed FP8 activations and NVFP4 projections

Status: proposed

## Context

Kore needs both compressed weights and FP8 activation/KV storage. ADR 0009's native OpenVINO
FP8 tier did not cover Core ML or an explicit NVFP4 scale representation. FP4 is not a TOSA
1.0 tensor dtype, and NVFP4 cannot be represented faithfully by labelling its bytes INT4.

NVFP4 uses E2M1 values, E4M3 block scales for groups of 16, and a separate tensor scale
([NVIDIA format description](https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/nvfp4/nvfp4.html)).
MXFP4's scale format and group geometry are different. Core ML's compressed-weight operations
do not supply a general FP8 model boundary; this implementation uses explicitly typed graphs
and existing INT8 boundary storage.

## Decision

No protocol change, new dtype number, custom operator or artifact format is introduced. Native
IR construction remains confined to the host adapters. The portable authoring crate gains
GATHER/SLICE authoring and `quantized::{nvfp4_matmul,nvfp4_matmul_fp16}`.

### NVFP4

The projection builders use TOSA 1.0 FLOATING_POINT | INTEGER, Level8K, no extensions:

- Activations `[1,M,K]` and output `[1,M,N]` are explicitly FP32 or FP16.
- Packed weights `[1,N*K/2]` and scales `[1,N*K/16]` use INT8 as a raw byte carrier.
- Low nibble comes first along K; each row has independent groups of 16.
- The equation is `output = activations @ (E2M1 * block_scale).T * tensor_scale`.
  The tensor scale is positive and finite and applied after MATMUL. The FP16 variant accepts
  its scale as binary16 bits and rounds MATMUL's result before the final FP16 multiply.

Lookup indices are exactly `CAST(INT8 -> INT32) + 128`. `TosaAnalysis` proves their range
is `[0,255]` from the producer graph and serialized offset, without trusting names or host
assertions. The mixed target accepts only these proven gather conditions; arbitrary indices
and all other unresolved runtime conditions are rejected. Scalar low/high-nibble lookups are
separate: a two-column Gather produced incorrect +/-2 entries on OpenVINO 2026.4.1 CPU.

These builders accept runtime weights and scales. Updating buffers does not require program
recompilation. They do not impose a checkpoint format or perform automatic host quantization.

### FP8 activations and cache tensors

Core ML's new FP8 target lowers E4M3/E5M2 to typed ML Programs with byte-sized INT8 boundary
carriers on macOS 26+. Decode is a 256-entry lookup. Encode performs direct binary32
nearest-even rounding, including subnormals, signed zero, overflow and NaN classification.
It does not round through binary16. E4M3 overflow produces NaN; E5M2 overflow produces
infinity. This is not a saturating quantizer. Model-specific scales remain explicit graph
operations around CAST.

FP8 MATMUL decodes operands, accumulates in the wider native type, and produces the TOSA
FP16 result. Constants and identity/reshape/transpose/concat/static-slice preserve packed
storage. Movement preserves all byte patterns, including NaN payloads. These primitives
support cache append and sliding windows; cache allocation, paged attention and attention
fusion remain responsibilities of the inference frontend.

OpenVINO retains ADR 0009's native lowering for existing graphs. Graphs containing FP32 CAST or
SLICE use byte carriers for FP8 throughout: native Convert exhibited FP16 double rounding,
and the GPU rejected native FP8 Slice. This exact path admits CONST/CONST_SHAPE, CAST,
IDENTITY, MATMUL, widened ADD/MUL, RESHAPE, TRANSPOSE, CONCAT and static SLICE. Other combined
graphs are rejected instead of returning an incorrectly narrowed result. The native FP8
compilation probe still controls target advertisement; Lunar Lake NPU withholds that target.

### Precision and placement

OpenVINO's ACCURACY hint alone allowed a declared-FP32 projection to round as FP16 on Lunar
Lake NPU (197.1875 became 197.25). Quantized FP32 and exact FP8 arithmetic now additionally
require `INFERENCE_PRECISION_HINT=f32`. A startup compile probe determines whether the device
accepts that policy. Devices that reject it advertise the FP16 mixed-target descriptor and
reject FP32 quantized programs. The caller must select the explicit FP16 variant.

Core ML's typed ML Programs retain the selected CpuAndNeuralEngine, CpuAndGpu or All policy.
Every policy allows CPU participation; numerical success does not prove GPU/ANE placement
or simultaneous use. The older NeuralNetwork FP32 policy restrictions remain unchanged.

## Validation

Native tests cover all 256 FP8 decode patterns, finite round trips, every finite rounding
midpoint and its adjacent binary32 values, overflow/NaN edges, constant FP8 weights with
FP8 activations, and cache concat/slice preserving every byte. NVFP4 projection fixtures
exercise every packed E2M1 byte and zero/subnormal/normal/max-finite E4M3 block scales across
row boundaries. FP16 outputs are checked against independent binary64 dot products rounded
with Python's binary16 packing before and after scaling. Portable negative tests reject
wrong lookup offsets, unrestricted INT32 sources and undersized lookup tables.

On 2026-11-19, Apple M4/macOS 27.0.1 passes these paths under all three Core ML policies.
Gigatron/OpenVINO 2026.4.1 passes FP8 activation/cache and FP32 NVFP4 on CPU/GPU, and explicit
FP16 NVFP4 on CPU/GPU/Lunar Lake NPU. The NPU's native FP8 and FP32-policy refusals remain
visible. These are device execution results, not QEMU or guest-driver validation.

## Costs and limits

Boundary storage remains one byte per FP8 element and 4.5 bits per NVFP4 weight including
its block scale, excluding the separate tensor scale. Native graphs may materialize FP16,
FP32 and INT32 intermediates, including expanded weights on each execution. No native FP4
arithmetic, compressed resident-weight footprint, large-model throughput or memory-capacity
claim is made. Persistent predecoded weights, backend-specific compressed constants and fused
quantized attention need separate measured work; they must preserve these numerical contracts.
