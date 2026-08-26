# TOSA INT8 and block-scaled status for AMD XDNA

Status checked: **2026-08-25**

Implementation update (2026-08-26): issue #144 implements the first exact XDNA integer tier:
INT8 IDENTITY and batch-1, zero-point-aware INT8 × INT8 → INT32 MATMUL. It uses the shared
`IDENTITY_INT8` and `MATMUL_INT8` corpus oracles on the physical NPU and does not claim complete
`PRO-INT` support. The semantic surface mirrors OpenVINO. XDNA's documented hardware divergences
are a bounded one-core memory envelope and explicit four-byte input-slot padding required by AIE
DMA; neither path introduces host arithmetic or hidden staging. Tile-compatible MATMUL widens the
zero-point-adjusted values exactly to INT16 before using AIE2P's native 4x4x8 matrix unit, while
small incomplete tiles use an exact scalar device kernel.

Implementation update (2026-08-26): issue #147 adds exact signed INT32 → INT8 `RESCALE` for the
released per-tensor `scale32`/`SINGLE_ROUND` form. This intentionally expands beyond OpenVINO's
current `CONST`/`IDENTITY`/`MATMUL` operator surface while preserving its strict-admission and
no-fallback precedent. The AIE kernel performs the multiply, signed rounding, output-zero-point
addition, saturation, and DMA-tail clearing on the NPU. The shared oracle covers negative values,
ties, nonzero output zero point, and saturation; the complete lifecycle passed on XDNA2 hardware.

## Executive conclusion

INT8 inference and INT8 `MATMUL` are stable TOSA work. The latest official release is TOSA
1.0.2, tagged on 2026-07-17, and its non-draft machine-readable specification marks the Integer
profile (`PRO-INT`) **Complete**. It defines INT8 × INT8 `MATMUL` with INT32 output and exact
integer results. [Official release tags](https://github.com/arm/tosa-specification/tags),
[TOSA 1.0.2 machine-readable specification](https://github.com/arm/tosa-specification/blob/v1.0.2/tosa.xml#L3-L16)

Block-scaled formats are not released TOSA vocabulary. They appear in the official repository's
TOSA 1.1 development source, whose version is explicitly `draft="true"`, under experimental
`EXT-MX-*` extensions. The draft uses **MX**, **microscaling**, and **block-scaled** terminology;
it does not define a standard format named “block FP16” or “BFP16.”
[TOSA 1.1 draft version and extension declarations](https://github.com/arm/tosa-specification/blob/main/tosa.xml#L3-L86)

The safest project split is therefore:

1. implement a released-semantics **XDNA exact INT8/INT32 operator tier** now, beginning with
   `IDENTITY` and zero-point-aware `MATMUL` and without claiming full `PRO-INT` compliance;
2. add `RESCALE` and further integer inference operators in separate evidence-backed tickets; and
3. keep **AMD `bfp16ebs8` block-8** and **TOSA 1.1-draft MX block-32** as separately named,
   separately tracked experiments until the schema, protocol negotiation, reference semantics,
   and conformance story exist.

## Source and status baseline

The official publication index and Git tags identify 1.0.2 as the newest released specification
as of the date above. The tagged XML says `1.0.2`, `draft="false"`; the repository's `main` XML
says `1.1.0`, `draft="true"`. [Official TOSA publication index](https://www.mlplatform.org/tosa/tosa_spec.html),
[released 1.0.2 XML](https://raw.githubusercontent.com/arm/tosa-specification/v1.0.2/tosa.xml),
[development XML](https://raw.githubusercontent.com/arm/tosa-specification/main/tosa.xml)

TOSA status labels apply to profiles and extensions independently of whether the containing
document was released. In the specification, **Complete** means operators are specified,
conformance tests exist, and backward compatibility applies; **Experimental** means operators may
change and backward compatibility is not guaranteed. Thus a released document can still contain
an experimental extension, as TOSA 1.0.2 does for `EXT-BF16` and the elementwise FP8 extensions.
[TOSA status definitions](https://github.com/arm/tosa-specification/blob/v1.0.2/chapters/introduction.adoc#L169-L180),
[TOSA 1.0.2 extension statuses](https://github.com/arm/tosa-specification/blob/v1.0.2/tosa.xml#L8-L45)

This distinction matters here:

- `PRO-INT`: released in 1.0.2 and **Complete**;
- `EXT-BF16`: present in released 1.0.2 but **Experimental**;
- `EXT-MX-*`: absent from released 1.0.2, present only in the 1.1 development source, and
  **Experimental** there.

## 1. Released INT8 inference semantics

### INT8 `MATMUL`

TOSA 1.0.2 `MATMUL` takes rank-3 tensors `A[N,H,C]` and `B[N,C,W]`, two rank-1 zero-point
tensors, and produces `output[N,H,W]`. The released type table assigns INT8 × INT8 → INT32 to
`PRO-INT`; it was added in TOSA 1.0. The zero points may be nonzero for INT8 and are compile-time
constants unless `EXT-DYNAMIC` is used.
[Released `MATMUL` arguments and INT8 type row](https://github.com/arm/tosa-specification/blob/v1.0.2/tosa.xml#L550-L594)

Integer-profile operations must match exactly. That makes an INT8 XDNA implementation's visible
contract bit-exact INT32 accumulation, including subtraction of both graph-declared zero points;
silently converting through floating point or relying on vendor-default quantization would not be
equivalent. [Integer profile compliance](https://github.com/arm/tosa-specification/blob/v1.0.2/chapters/introduction.adoc#L365-L375)

This repository already has the right reusable proof machinery:

- [`dot_i8_i32`](../../crates/virtio-accel-tosa/src/integer.rs) implements exact signed INT8 dot
  products, explicit zero-point subtraction, checked INT32 accumulation, and no wrapping;
- [`MATMUL_INT8`](../../crates/virtio-accel-conformance/src/numerics.rs) is a stable non-square
  corpus case with nonzero zero points and an exact INT32 oracle; and
- [OpenVINO's integer lowering](../../crates/virtio-accel-openvino/src/lower.rs) widens both INT8
  operands, subtracts the two TOSA zero points explicitly, and then performs INT32 `MatMul`. That
  is the structural precedent XDNA should mirror unless an XDNA kernel is proven to implement the
  same equation directly.

### Quantized inference beyond `MATMUL`

TOSA does not attach an implicit floating scale to every integer tensor. Zero points are explicit
operator inputs, and a change of quantization scale is represented by an explicit `RESCALE`
operator. Convolution-family operators produce an INT32 accumulator result, after which `RESCALE`
maps into the desired lower-precision output domain. [Released quantization and integer-convolution rules](https://github.com/arm/tosa-specification/blob/v1.0.2/chapters/introduction.adoc#L518-L545)

For an inference-capable XDNA path, the useful sequence is therefore:

1. exact INT8 storage and data movement;
2. exact INT8 × INT8 → INT32 `MATMUL`;
3. explicit `RESCALE` from INT32 to INT8, with the selected TOSA rounding semantics, zero point,
   and saturation; and
4. later, operator-by-operator expansion into INT8 convolution, pooling, activation,
   elementwise, and shape/data-movement operations from the released Integer profile.

The repository already provides [`rescale_i32_to_i8`](../../crates/virtio-accel-tosa/src/integer.rs),
so `RESCALE` should reuse that shared arithmetic contract rather than introducing a backend-local
oracle.

### What may be advertised

TOSA says an implementation claiming a profile must implement all operator/type combinations in
that profile. [Profile requirements](https://github.com/arm/tosa-specification/blob/v1.0.2/chapters/introduction.adoc#L130-L143),
[Integer profile compliance](https://github.com/arm/tosa-specification/blob/v1.0.2/chapters/introduction.adoc#L365-L375)

Consequently, implementing only `IDENTITY` and `MATMUL` does **not** justify saying “XDNA is a
TOSA Integer-profile-compliant implementation.” The safe project wording is:

> XDNA exact INT8/INT32 tier implementing a conservative TOSA 1.0 Integer-profile operator
> subset: `IDENTITY` and zero-point-aware `MATMUL`.

That wording matches the repository's existing OpenVINO precedent: it uses a TOSA 1.0 integer
target but publishes a separate conservative capability descriptor limited to `CONST`,
`IDENTITY`, and `MATMUL`. The capability descriptor is the claim boundary; the target identifies
the numerical rules under which admitted graphs are analyzed.
[OpenVINO integer target and conservative capability](../../crates/virtio-accel-openvino/src/lower.rs)

The XDNA crate defines
[`XDNA_TOSA_INTEGER_TARGET`](../../crates/virtio-accel-xdna/src/lower.rs) and now advertises a
similarly narrow `XDNA_TOSA_INTEGER_CAPABILITY` because its exact corpus and on-metal lifecycle
pass. It does not add the whole Integer profile's operators merely because the shared TOSA
validator recognizes them.

## 2. Block floating-point and block-scaled status

### What released TOSA contains

TOSA 1.0.2 contains ordinary elementwise `FP16`, experimental `BF16`, and experimental
elementwise `FP8E4M3`/`FP8E5M2`. Its extension list contains no `EXT-MX-*`, its released number
formats contain no block-scaled composite type, and its serialization therefore has no released
way to express an MX tensor. [Released 1.0.2 extension list](https://github.com/arm/tosa-specification/blob/v1.0.2/tosa.xml#L8-L45),
[released number formats](https://github.com/arm/tosa-specification/blob/v1.0.2/chapters/introduction.adoc#L182-L310)

The local parser deliberately enforces that released boundary: [`Version::TOSA_1_0` and
`DType::is_tosa_1_0`](../../crates/virtio-accel-tosa/src/types.rs) stop at the ordinary FP8 types,
and [`ExtensionSet`](../../crates/virtio-accel-tosa/src/artifact.rs) contains the TOSA 1.0
extension set but no MX bits.

### What the official 1.1 draft contains

The official development XML declares these experimental extensions:

- `EXT-MX-COMMON`;
- `EXT-MX-FP4E2M1`, `EXT-MX-FP6E2M3`, `EXT-MX-FP6E3M2`;
- `EXT-MX-FP8E4M3`, `EXT-MX-FP8E5M2`;
- `EXT-MX-INT8`; and
- `EXT-MXFP-CONV` for block-scaled convolution.

All are marked **Experimental**, and the containing specification is still the 1.1 draft.
[Draft MX extension declarations](https://github.com/arm/tosa-specification/blob/main/tosa.xml#L47-L86)

The draft's generic vocabulary is `block_scale_t<block_shape, scale_t, value_t>`. Its currently
listed set uses `BLOCK_SHAPE_32`, an `fp8ue8m0_t` scale, and FP4, FP6, FP8, or `mxint8_t` values.
It does not list FP16 as the block value format and therefore does not standardize a type named
“block FP16.” [Draft block-scaled type and format set](https://github.com/arm/tosa-specification/blob/main/chapters/introduction.adoc#L340-L375),
[draft `BLOCK_SHAPE_32`](https://github.com/arm/tosa-specification/blob/main/tosa.xml#L5031-L5041)

The semantics are also still moving. The current draft has generic `CAST` rows to and from MX,
dedicated `CAST_FROM_BLOCK_SCALED` and `CAST_TO_BLOCK_SCALED` operators that carry separate value
and scale tensors, and a dedicated `CONV2D_BLOCK_SCALED`. The ordinary `MATMUL` table in the same
snapshot has no block-scaled type row. [Draft MX `CAST` rows and dedicated conversion operator](https://github.com/arm/tosa-specification/blob/main/tosa.xml#L3952-L4055),
[draft block-scaled convolution](https://github.com/arm/tosa-specification/blob/main/tosa.xml#L489-L570),
[draft ordinary `MATMUL` type table](https://github.com/arm/tosa-specification/blob/main/tosa.xml#L855-L938)

Therefore it would be inaccurate, as of this date, to describe an AMD block-format `MATMUL` as
implementing a released TOSA MX `MATMUL`. The official draft provides useful direction for types,
conversion, layout, and convolution, but not a finalized compatibility promise for this project's
wire or a released block-scaled matrix-multiplication contract.

### AMD `bfp16ebs8` is a separate project concept

The project's numerical-tier decision records AMD `bfp16ebs8` as a native XDNA block-8 format:
eight values share an exponent. It also explicitly rejects treating that math as FP16 or BF16 and
keeps it outside the first numerical tier. [Issue #82 resolution](https://github.com/MicroPerceptron/virtio-accel/issues/82#issuecomment-5363112248),
[project block-scaled forward specification](https://github.com/MicroPerceptron/virtio-accel/blob/grilling/first-numerical-tier/docs/research/amdxdna-blockfp-tier.md)

That differs from the TOSA draft MX set's block-32, E8M0-scaled formats. A proposed mapping from
one MX block into four AMD block-8 groups remains a hardware hypothesis, not a semantic identity.
It must be proven for element encoding, scale alignment, rounding, saturation, exceptional values,
and exact output bits before any compatibility claim. [Existing project tracking issue #110](https://github.com/MicroPerceptron/virtio-accel/issues/110)

## 3. Required project terminology

Use these terms:

- **“XDNA exact INT8/INT32 tier”** or **“TOSA 1.0 INT8 operator subset”** for the initial
  `IDENTITY`/`MATMUL` work.
- **“TOSA Integer-profile compliant”** only after the entire released profile and official
  conformance obligations are met.
- **“TOSA 1.1-draft MX block-scaled experiment”** when discussing the current `EXT-MX-*`
  design. Include the draft version or a pinned source revision in technical work.
- **“AMD XDNA `bfp16ebs8` vendor experiment (block-8)”** for the native AMD format.
- **“MXINT8 block-scaled tier”** only for the exact OCP/TOSA-draft-style block-32 E8M0 +
  `mxint8_t` contract, after that contract is pinned and implemented.

Avoid these terms:

- **“TOSA block FP16”**: no released or draft TOSA type has that name.
- **“BFP16”** without the AMD format name: it is ambiguous and can be mistaken for BF16 or
  ordinary IEEE FP16.
- **“FP16 tier”** for `bfp16ebs8`: block sharing changes the numerical result and does not
  implement FP16 semantics.
- **“EXT-MX-INT8 support”** in a public target today: the extension is draft-only and this
  repository's stable TOSA 1.0 target/schema cannot encode it.
- **“provisional TOSA extension bit”** without protocol negotiation: allocating a reserved target
  bit is a protocol compatible extension, not a backend-private implementation detail.

If pre-standard hardware exploration is useful, use backend-local test artifacts and call it an
AMD vendor experiment. Do not expose it through `ARTIFACT_FORMAT` as TOSA, a public
`TosaCapabilityProvider`, or a TOSA 1.0 `Target`.

## 4. Recommended ticket boundaries

### INT8 chain: implement now

1. **XDNA exact INT8 `IDENTITY` and zero-point-aware `MATMUL`.** Add the conservative integer
   capability, strict admission, native INT8 × INT8 → INT32 compilation, exact slot plans, the
   existing shared `IDENTITY_INT8`/`MATMUL_INT8` oracles, direct-binding diagnostics, and on-metal
   lifecycle proof. Mirror OpenVINO's target/capability split and explicit zero-point semantics;
   diverge only where the AIE kernel API implements the same arithmetic directly.
2. **XDNA TOSA `RESCALE` INT32 → INT8 (implemented by #147).** Scale, shift, rounding mode, output
   zero point, and saturation are the public numerical contract. The implementation reuses the
   shared integer oracle and includes edge, invalid-parameter, and on-metal cases. Optional
   rounding extensions remain excluded until they have separate evidence.
3. **Integer operator expansion, one numerical family per ticket.** Suggested order: `CONV2D`,
   `DEPTHWISE_CONV2D`, integer pooling, then elementwise/activation/data movement. Each ticket
   adds only the capability rows demonstrated by shared fixtures and hardware runs.
4. **Full `PRO-INT` compliance umbrella, optional and last.** Track the gap against every released
   Integer-profile operator/type row and official conformance case. Only closing this umbrella
   permits the unqualified compliance claim.

### Block-scaled chain: research and protocol first

1. **Characterize AMD `bfp16ebs8` on silicon.** Freeze its exact block-8 encoding and behavior;
   test the proposed block-32 MXINT8 mapping. This ticket advertises nothing and may use only
   backend-local/precompiled test artifacts.
2. **Track and pin the official TOSA MX draft.** Record a source revision and diff it as the 1.1
   draft changes. Do not copy draft discriminants into the stable schema merely to unblock the
   backend.
3. **Adopt a released block-scaled-capable schema and negotiate it.** Once TOSA releases suitable
   vocabulary, classify the change under [`docs/wire-abi.md` section 9](../wire-abi.md#9-candidate-and-post-freeze-change-procedure):
   add explicit feature/new-opcode negotiation, preserve all 1.0 frames, and create the required
   minor-version conformance directory. If the released semantics still lack block `MATMUL`, keep
   that operation vendor-specific rather than filling the gap under a TOSA label.
4. **Add shared parser, type, reference, and oracle support.** Implement the released composite
   type/layout, deterministic reference quantization and conversion, serialization vectors, and
   conformance fixtures before any backend capability advertisement.
5. **Implement the XDNA standard MX tier.** Advertise only the exact released extension set and
   operator rows proven on metal.
6. **Optionally implement AMD `bfp16ebs8` as a distinct vendor tier.** Give it separate negotiation,
   target identity, fixtures, and documentation. Never make a host knob swap it underneath an MX,
   FP16, or BF16 label.

This split preserves the project's guest-chosen-tier rule: the graph/target label fixes the math,
while host configuration may only decide whether that distinct tier is offered.
