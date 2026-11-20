# ADR 0015: Constant integer weights for OpenVINO

Status: proposed

## Context

Runtime NVFP4 graphs reduce boundary bytes but do not select OpenVINO's native compressed
integer-weight kernels. Weight-only INT4 is a distinct format, useful when weights are
immutable and decode is limited by weight traffic. TOSA 1.0 permits neither INT4-to-float
CAST nor an INT4/FP16 MATMUL, so relabelling such a graph as TOSA INT4 would be invalid.

## Decision

No protocol, dtype number, profile identity or artifact format changes. Portable authoring
adds symmetric grouped INT4/INT8 constant-weight projections with FP16 activations and
outputs. INT4 accepts packed two's-complement [-8,7]; serialization uses exact INT8 values.
This includes -8 legally as INT8, without changing TOSA INT4's separate [-7,7] contract.
Scales are explicit positive finite binary16 values. There is no host float quantizer,
implicit zero point, asymmetric mode or NVFP4 conversion. Updating constants requires reload.

OpenVINO repacks an INT8 constant only if every value fits [-8,7], it is not a block output,
and all consumers are FP16/FP32 CASTs. This changes storage only: all scalar values remain
exact. Odd counts zero the unused high nibble. Other INT8 constants remain byte-sized.
Integer-profile MATMUL retains its exact INT32 arithmetic and zero points.

A shared RHS transpose `[0,2,1]` with batch dimension one folds into MatMul's transpose flag.
An equivalent two-dimensional weight reshape exposes the compressed-weight pattern to the
plugin. Only that MatMul's edge changes; other consumers retain their original values.
Native compilers remain responsible for placement and kernel selection.

OpenVINO CPU/GPU compilation explicitly disables dynamic activation quantization. Probing
showed that ACCURACY alone inserts DynamicQuantize and changes FP16 activations to INT8.
The property is not sent to NPU, which does not support it. Existing explicit FP32 policy
and admission probes remain in force. No inference-time allocations or ownership rules change.

## Evidence and limits

Unit tests check signed nibbles, range/consumer/boundary exclusions, odd counts, scale and
shape validation, and the native constant bytes. Native tests execute W4A16 and W8A16 with
matching independent golden outputs. The benchmark suite has matched constant FP16/W8/W4
controls at `[M, N, K] = [1 or 16, 1024, 2048]`.

The compiler diagnostic reads production-generated IR, performs 12 random-input comparisons
to binary64 reference products rounded to binary16, and retains the execution graph.
Lunar Lake GPU with OpenVINO 2026.4.1 reports i4 weights with f16 arithmetic and no
DynamicQuantize. CPU and NPU numerical success does not prove the same kernel selection:
the CPU report still contains runtime weight conversion, and the NPU report does not expose
useful weight precision. No whole-model resident-memory or tokens/second claim follows.

The portable artifact still costs eight bits per weight. The native IR uses four bits plus
group scales and alignment, but physical resident memory is plugin-dependent. Runtime
weights, asymmetric INT4, native compressed NVFP4 and exact optimized integer-profile
MATMUL are separate work; none is silently substituted by this path.
