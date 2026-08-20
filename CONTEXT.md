# virtio-accel

A portable, pre-standardization protocol (with executable Rust implementations) through which
a guest submits TOSA programs to host accelerator backends.

## Language

**Advertised tier**:
The set of dtypes and operators a backend publicly promises to execute, named by a TOSA
`Target` (profiles + extensions) and proven by conformance fixtures. What is not advertised is
rejected at admission, never approximated.
_Avoid_: supported formats, capabilities (ambiguous)

**Guest-chosen tier**:
The principle that any choice changing numerical results is made by the guest, per program,
from the advertised menu. Host configuration may gate which tiers are offered; it may never
change what an advertised label means.
_Avoid_: precision knob, host override

**Relabeling**:
Executing math in one precision while reporting results under another precision's label
(e.g. BF16 execution labeled FP16). Forbidden in every form; the loud alternative is
admission rejection.
_Avoid_: transparent downcast, automatic conversion

**Promotion**:
Lossless widening of values into a larger format (e.g. FP8 → BF16), performed explicitly —
as an in-graph CAST or a consumer-side conversion — never silently inside a backend.
_Avoid_: upcast (when meant implicitly)

**Block-scaled format**:
A format in which a group of values shares one scale factor (OCP MX, AMD `bfp16ebs8`).
Conversion into it is lossy and neighbor-dependent. Distinct from elementwise formats such as
TOSA FP8, which convert value-by-value and exactly.
_Avoid_: BF8, "Block FP16 ≈ FP16"
