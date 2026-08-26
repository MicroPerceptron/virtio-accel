# virtio-accel

A portable, pre-standardization protocol through which a guest submits TOSA programs to host
accelerator backends.

## Language

**Advertised tier**:
The set of dtypes and operators a backend publicly promises to execute, named by a TOSA `Target`
and proven by conformance fixtures. Everything else is rejected at admission.
_Avoid_: supported formats, capabilities (ambiguous)

**Guest-chosen tier**:
A numerical contract selected by the guest per program from the advertised tiers. Host
configuration may decide whether a tier is offered, but cannot change what its label means.
_Avoid_: precision knob, host override

**Relabeling**:
Executing one numerical contract while reporting the result under another contract's label.
Relabeling is forbidden; unsupported contracts are rejected.
_Avoid_: transparent downcast, automatic conversion

**Promotion**:
Lossless widening into a larger elementwise format, such as FP8 to BF16, represented explicitly in
the graph rather than hidden inside a backend.
_Avoid_: upcast (when it implies an invisible conversion)

**Exact INT8/INT32 tier**:
A conservative TOSA Integer-profile subset whose INT8 inputs, zero points, and INT32 results follow
the released integer semantics exactly.
_Avoid_: TOSA Integer-profile compliant (unless the complete profile is implemented)

**Block-scaled format**:
A numerical format in which a group of values shares a scale. Conversion is lossy and
neighbor-dependent, unlike elementwise FP8 promotion.
_Avoid_: BF8, block FP16

**TOSA draft MX experiment**:
An experiment following a pinned, unreleased TOSA MX block-scaled draft. It is not a stable TOSA
target or compatibility promise.
_Avoid_: TOSA MX support, EXT-MX support

**AMD `bfp16ebs8` vendor experiment**:
AMD's native block-8 shared-exponent format, treated as a distinct numerical contract from TOSA MX,
FP16, and BF16.
_Avoid_: BFP16, TOSA block FP16, FP16 tier
