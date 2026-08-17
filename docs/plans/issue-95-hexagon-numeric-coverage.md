# Issue #95 plan: broaden Qualcomm Hexagon numeric and operator coverage

Status: in progress; exact INT8/INT32 tier implemented and hardware-validated

Issue: [#95 — Qualcomm - Broaden numeric types coverage](https://github.com/MicroPerceptron/virtio-accel/issues/95)

Planning branch: `codex/issue-95-hexagon-numeric-coverage`

Last reviewed: 2026-08-17

## Implementation checkpoint (2026-08-17)

The first implementation slice is complete on the pinned Snapdragon X126100 / QAIRT
`2.49.0.260730` stack:

- added and exported a distinct TOSA integer-profile target;
- generalized tensor planning, binding byte lengths, and the Rust/C ABI to carry FP16, FP32, INT8,
  INT32, and QNN scale-offset metadata without submission-time tensor copies;
- executed shared INT8 identity and nonzero-zero-point INT8 MATMUL fixtures on HTP with bit-exact
  INT8 storage and INT32 results;
- ran an FP32 precision-distinguishing MATMUL probe. QNN accepted FLOAT_32 tensors and the HTP FP32
  graph configuration but returned the FP16-rounded result, so production admission still rejects
  FP32;
- inspected and probed the pinned public FLOAT8 surface. It exposes no unambiguous ordinary tensor
  selector for E4M3 versus E5M2, so neither TOSA FP8 target is advertised;
- added direct-binding and explicit-transfer counters and passed the reusable backend semantic suite
  on HTP with zero hidden staging; and
- retained all existing FP16 hardware oracles.

The issue remains open for the shared operator-surface expansion and controlled performance sweep.
Those cells will not be advertised until each has a TOSA oracle and HTP evidence.

## Outcome

Bring `virtio-accel-hexagon` to the same evidence standard and advertised TOSA operator surface as
the Core ML and OpenVINO host backends, while adding only numeric tiers that execute with proven
TOSA semantics on the pinned QNN HTP runtime. The completed backend will:

- retain the existing FP16 tier;
- add FP32 only if HTP preserves the shared edge-value and operator oracles without silently using
  FP16 math;
- add the TOSA integer profile with exact INT8 storage, explicit zero-point behavior, and required
  INT32 results;
- add FP8E4M3 and FP8E5M2 only when the installed QAIRT release and Snapdragon device can expose
  each format unambiguously and pass hardware oracles;
- cover the 42 operators currently advertised by both Core ML and OpenVINO for valid floating-point
  combinations, while retaining the narrower reference-backend integer matrix where applicable;
- pass reusable semantic, numerical, lifecycle, and direct-binding conformance; and
- publish reproducible latency, throughput, staging, and resource evidence rather than asserting
  performance parity across unlike hardware.

An unsupported precision or operator remains rejected during program loading. No tier may fall back
to QNN CPU/GPU, relabel lower-precision computation, or widen `supports_tosa_*` before its portable
and hardware gates pass.

## Definition of parity

Core ML and OpenVINO currently advertise the same 42-operator surface:

- graph/data movement: `CONST`, `CONST_SHAPE`, `IDENTITY`, `RESHAPE`, `TRANSPOSE`, `REVERSE`, and
  `CONCAT`;
- binary arithmetic: `ADD`, `SUB`, `MUL`, `POW`, `MAXIMUM`, and `MINIMUM`;
- unary/activation: `ABS`, `CEIL`, `COS`, `ERF`, `EXP`, `FLOOR`, `LOG`, `NEGATE`, `RECIPROCAL`,
  `RSQRT`, `SIN`, `SIGMOID`, `TANH`, and `CLAMP`;
- comparison/selection/logical: `EQUAL`, `GREATER`, `GREATER_EQUAL`, `SELECT`, `LOGICAL_AND`,
  `LOGICAL_OR`, `LOGICAL_XOR`, and `LOGICAL_NOT`;
- reductions: `ARGMAX`, `REDUCE_MAX`, `REDUCE_MIN`, `REDUCE_PRODUCT`, and `REDUCE_SUM`; and
- compute: `MATMUL` and `MAX_POOL2D`.

Operator parity means Hexagon accepts every valid FP16 and proven FP32 case in that shared surface,
including the same attribute, axis, rank, constant, and output-type restrictions. It does not mean
that every operator accepts every dtype. Integer parity initially means the exact reference tier:
INT8 identity and zero-point-aware INT8 `MATMUL` with INT32 accumulation/output. FP8 is an optional
additional tier and does not weaken the required FP16, FP32, or INT8 work.

Performance parity means equivalent engineering evidence: direct provider bindings with zero hidden
submission staging, bounded warm-path metadata, separate admission and completion measurements, and
an identified hardware/runtime baseline. Absolute latency across an Apple NPU, Intel/ARM CPU, and
Qualcomm NPU is not a meaningful universal gate. On the Snapdragon host, compare HTP with OpenVINO
CPU over the same model/shape sweep and record the crossover or any physical limitation.

## Current baseline and design constraints

PR #94 provides FP16 `IDENTITY`, `MATMUL`, and zero-padded `MAX_POOL2D` on Windows ARM64 with QAIRT
`2.49.0.260730`, QNN HTP provider build `v2.49.0.260730134355`, and the v73 device tier. This branch
has replaced its hard-coded FP16 tensor representation with typed tensor/quantization descriptors;
the fixed node descriptor still cannot represent the parity operator set or variable parameters.

The SDK headers expose `QNN_DATATYPE_FLOAT_32`, `QNN_DATATYPE_INT_8`,
`QNN_DATATYPE_INT_32`, and a generic `QNN_DATATYPE_FLOAT_8`. Header presence is not device support.
In particular, the public HTP graph precision option distinguishes FP16 and FP32 math, while the
base tensor datatype does not by itself prove whether generic FLOAT8 is E4M3 or E5M2 on this target.
Those are hardware-spike questions, not assumptions.

TOSA FP16 and FP32 share the same floating-point target identity. Therefore FP32 cannot be modeled
as a fake extension or a second identical `Target`; admitting it broadens `HEXAGON_TOSA_TARGET` and
must be documented as such. INT8 uses a distinct integer-profile target. FP8E4M3 and FP8E5M2 use
separate, truthful floating-point extension targets.

## Phase 0: build the capability and parity matrix

- [ ] Land or rebase onto PR #94 and record its final merge commit before changing behavior.
- [ ] Add one checked-in matrix mapping each of the 42 TOSA operators and each relevant dtype to the
      required QNN op, parameter tensors, output dtype, semantic restrictions, portable tests, and
      hardware tests. Keep unsupported cells explicit.
- [ ] Keep a standalone native capability probe for proposed numeric tiers. The implementation spike
      created/finalized FP32 and INT8/INT32 graphs through `QnnHtp` and recorded the selected
      provider/API versions; FP8 was blocked at the public encoding gate before graph creation.
- [x] Run FP32 identity edges containing NaNs, infinities, signed zero, subnormals, and values that
      distinguish FP16 from FP32. Reject FP32 if HTP canonicalization exceeds the TOSA oracle or if
      profiling/configuration cannot rule out silent FP16 computation.
- [x] Run INT8 identity plus nonzero-zero-point `MATMUL` with INT32 output. Decide whether QNN can
      express TOSA integer semantics directly or requires explicit widen/subtract/matmul nodes.
- [x] Determine how QAIRT selects E4M3 versus E5M2 for `QNN_DATATYPE_FLOAT_8`, whether v73 supports
      client-visible FP8 tensors, and whether both encodings survive identity bit-class tests. If
      the public API cannot select an encoding unambiguously, mark that FP8 target unavailable.
- [ ] Probe every proposed QNN operator with representative ranks, axes, broadcasting, constants,
      and edge attributes before adding it to the public support function.

Exit gate: the matrix identifies a legal QNN representation and reproducible evidence command for
every proposed cell. FP32, INT8, and each FP8 format have an explicit supported or blocked result.

## Phase 1: generalize the owned graph plan and native ABI

- [ ] Extend the new FP16/FP32/INT8/INT32 tensor descriptor with BOOL, constant bytes, and any future
      explicit FP8 encoding. Checked scalar sizes, exact boundary byte lengths, QNN datatypes, and
      scale-offset quantization metadata are implemented for the current numeric tiers.
- [x] Preserve separate target admission: floating/no-extension and integer/no-extension; reject
      unavailable FP8E4M3 and FP8E5M2 targets before graph construction.
- [ ] Replace the fixed `NodeDesc { input0, input1, output, kernel, stride }` ABI with bounded,
      owned input/output/parameter slices that can represent variable arity, axes, permutations,
      scalar attributes, and generated parameter tensors without borrowed-lifetime ambiguity.
- [ ] Make the C++ bridge copy all descriptors and parameter storage before returning. Validate
      dtype, role, I/O index, tensor references, arity, rank, parameter lengths, and all size
      conversions before calling QNN.
- [x] Set each QNN tensor's real datatype and encoding instead of hard-coding FP16. Retain exact
      model I/O ordering and use static/native tensor types correctly for constants and generated
      parameters.
- [x] Generalize buffer range validation from two bytes per element to the planned tensor byte
      length. Preserve alignment, direct binding, conflict guards, and event-owned lifetimes for
      every scalar size.
- [ ] Add portable ABI/layout assertions, malformed-descriptor tests, overflow tests, and synthetic
      mixed-type graph-plan tests. Update `SAFETY.md` for the generalized descriptor ownership.

Exit gate: the refactor preserves every existing FP16 hardware oracle, supports typed graphs
without special cases in submission, and leaves all unproven tiers rejected.

## Phase 2: add the FP32 tier

- [x] Keep FP32 rejected after Phase 0 proved that the pinned HTP stack silently rounds
      precision-distinguishing inputs to FP16 during MATMUL.
- [ ] Configure HTP graph precision explicitly when required and fail graph loading if the runtime
      rejects the precision request. Never retry through a relaxed FP16 path.
- [ ] Run the shared FP32 identity-edge, non-square batched `MATMUL`, and NHWC `MAX_POOL2D` cases on
      HTP before advertising FP32.
- [ ] Add FP32 constant, broadcast, comparison, reduction, transcendental, and boundary-value cases
      as their operator families land. Use TOSA tolerances per operation; preserve exact signed-zero
      and nonfinite requirements where the shared oracle requires them.
- [ ] Add negative tests proving FP32 rejection on a runtime/device that cannot meet the precision
      gate and proving FP8/INT8 targets cannot enter the FP32 path.

Exit gate: all shared FP32 corpus cases and every advertised FP32 operator pass on the pinned HTP
stack, with evidence that the graph did not execute as FP16.

## Phase 3: add exact INT8 and INT32 semantics

- [x] Add `HEXAGON_TOSA_INTEGER_TARGET` for TOSA 1.0 integer profile with no extensions and export it
      alongside the floating target.
- [x] Admit INT8 model inputs/outputs and the INT32 outputs/internal tensors required by TOSA
      accumulation. Compute exact checked byte lengths and preserve raw two's-complement storage.
- [x] Lower INT8 identity without dequantizing through floating point.
- [x] Lower INT8 `MATMUL` with explicit input zero points and exact INT32 accumulation/output. If
      QNN quantization metadata changes numeric interpretation, prefer explicit integer arithmetic
      nodes or keep the case unsupported.
- [x] Validate zero-point scalar constants, rank/shape constraints, conflicting tensor encodings,
      overflow rules, output dtype,
      and QNN parameter lifetimes in portable lowering before graph creation.
- [x] Run `IDENTITY_INT8` and `MATMUL_INT8` from the shared conformance corpus bit-exactly on HTP,
      including nonzero zero points and negative inputs.
- [x] Add target-crossing and short-binding rejection tests. Wrong-output-dtype and invalid-constant
      cases are enforced by TOSA analysis and scalar decoding; accumulator edges remain a corpus
      expansion item.

Exit gate: the Hexagon integer target matches the existing Core ML/OpenVINO integer numerical tier
without float conversion or backend fallback.

## Phase 4: reach the shared 42-operator surface

Implement operators in reviewable families. For every family, add portable lowering tests,
hardware numerical tests for each advertised floating dtype, negative attribute/rank tests, and
resource-limit tests before updating `supports_tosa_operator`.

- [ ] Data movement and constants: `CONST`, `CONST_SHAPE`, `RESHAPE`, `TRANSPOSE`, `REVERSE`, and
      `CONCAT`, preserving exact axis/permutation/shape behavior and constant storage.
- [ ] Binary arithmetic and broadcasting: `ADD`, `SUB`, `MUL`, `POW`, `MAXIMUM`, and `MINIMUM`.
- [ ] Comparisons, selection, and logical operations: `EQUAL`, `GREATER`, `GREATER_EQUAL`, `SELECT`,
      `LOGICAL_AND`, `LOGICAL_OR`, `LOGICAL_XOR`, and `LOGICAL_NOT`, including BOOL tensors that are
      internal or model-visible only where the TOSA/backend contract permits them.
- [ ] Unary and activation operations: `ABS`, `CEIL`, `COS`, `ERF`, `EXP`, `FLOOR`, `LOG`,
      `NEGATE`, `RECIPROCAL`, `RSQRT`, `SIN`, `SIGMOID`, `TANH`, and `CLAMP`.
- [ ] Reductions and indexing: `ARGMAX`, `REDUCE_MAX`, `REDUCE_MIN`, `REDUCE_PRODUCT`, and
      `REDUCE_SUM`, with owned axis tensors, `keep_dims`, output dtype, and empty/invalid-axis
      behavior validated before QNN.
- [ ] Revalidate `MATMUL` and `MAX_POOL2D` across the new dtypes and the broader constant/parameter
      machinery. Preserve the existing Gather/maximum MAX_POOL2D fallback only when it remains
      HTP-native and semantically equivalent.
- [ ] Add a parity test that compares the Hexagon support table with the shared Core ML/OpenVINO
      operator set, plus an allowlist of intentionally blocked cells linked to hardware evidence.

Exit gate: no operator-level gap remains for the required floating-point surface, and every
dtype/operator restriction is represented in the matrix and enforced before native work.

## Phase 5: add FP8 conditionally

- [ ] Define separate `HEXAGON_TOSA_FP8E4M3_TARGET` and `HEXAGON_TOSA_FP8E5M2_TARGET` constants only
      for formats proven in Phase 0.
- [ ] Preserve exact one-byte client storage and carry an explicit format through the Rust plan and
      native boundary. Do not infer E4M3/E5M2 from a generic QNN FLOAT8 datatype.
- [ ] Start with the shared FP8 identity fixtures and check signed zero, subnormals, finite maxima,
      infinities where defined, and NaN classes according to each format.
- [ ] Add FP8 operator cells incrementally only where TOSA allows them and HTP passes numerical
      evidence. An unavailable format remains a documented rejection, not a skipped passing test.
- [ ] Report runtime/device capability separately for E4M3 and E5M2 so a machine supporting only
      one format never advertises both.

Exit gate: each advertised FP8 target has an unambiguous public QNN encoding and passes every
corresponding hardware oracle. Otherwise FP8 remains explicitly unavailable without blocking the
required FP32/INT8 completion.

## Phase 6: conformance, performance, and release evidence

- [x] Adapt the reusable backend conformance harness to Hexagon and run mandatory lifecycle,
      ownership, timeout, stable polling, wrong-binding, overlap/conflict, release, and resource
      cases for representative models from every enabled numeric target.
- [x] Add Hexagon counters/hooks for direct bindings, shared/imported bindings, staged bindings and
      bytes, explicit transfer bytes, live resources, and retained-allocation high-water marks.
      Require `staged_direct_bindings == 0` and `staged_direct_bytes == 0`.
- [ ] Add an ignored release-mode benchmark matching the Core ML/OpenVINO measurement structure:
      graph load/finalize, warm admission, submit-to-complete, steady-state throughput, and retained
      resources after warmup.
- [ ] Run fixed warmups and enough samples to report median and p95 for FP16, FP32, and INT8 over a
      shape/model-size sweep. Record SoC, Windows build, NPU driver, QAIRT/provider build, power
      mode, graph, dtype, sample count, and whether profiling was enabled.
- [ ] On the same Snapdragon machine, compare HTP against OpenVINO CPU for identical artifacts and
      bindings. Document overhead-dominated small graphs, the throughput crossover, unsupported
      cells, and physical limits rather than hiding them in an aggregate score.
- [ ] Stress repeated program load/unload and execution for each target. Verify contexts, graphs,
      events, allocations, generated tensors, worker jobs, and registrations return to baseline.
- [ ] Update the Hexagon README, architecture, portability, public API, performance evidence,
      support matrix, examples, and issue plan with the final supported/blocked matrix and exact
      reproduction commands.
- [ ] Add a controlled Windows ARM64 QAIRT hardware lane or document the manual release gate if SDK
      licensing prevents hosted CI. Keep SDK-free builds and examples green on every existing CI
      platform.

Exit gate: all required semantic and numerical cases pass, every advertised operator/dtype cell has
hardware evidence, direct-binding diagnostics show no hidden staging, performance results are
reproducible, and documentation matches the runtime surface exactly.

## Test and verification commands

SDK-free checks, required throughout development:

```sh
cargo fmt --all -- --check
cargo test -p virtio-accel-hexagon
cargo clippy -p virtio-accel-hexagon --all-targets -- -D warnings
cargo test --workspace
cargo clippy --workspace --all-targets --all-features -- -D warnings
```

Pinned Windows ARM64 QAIRT checks:

```powershell
$env:VIRTIO_ACCEL_HEXAGON = "1"
$env:VIRTIO_ACCEL_QNN_SDK_ROOT = "C:\path\to\qairt\2.49.0.260730"
cargo test -p virtio-accel-hexagon --test hexagon -- --nocapture
cargo test --release -p virtio-accel-hexagon `
  measures_warm_submission_and_completion_latency -- --ignored --nocapture
```

Run the capability probe and each numerical target separately so an unsupported optional FP8 tier
cannot mask a required FP32 or INT8 failure. Archive the command output with the hardware/runtime
identity in `docs/performance.md` or the final pull request evidence.

## Expected implementation files

- `crates/virtio-accel-hexagon/src/lower.rs`: target-specific admission, typed tensor plans,
  operator lowering, constants, attributes, and portable tests.
- `crates/virtio-accel-hexagon/src/ffi.rs` and `native/qnn_bridge.{h,cpp}`: generalized owned graph
  ABI, real QNN datatypes/encodings, operator construction, capability/profiling evidence.
- `crates/virtio-accel-hexagon/src/native.rs`: typed byte/range validation, diagnostics, target
  dispatch, and measurement hooks without submission-time tensor copies.
- `crates/virtio-accel-hexagon/tests/hexagon.rs`: parameterized hardware runner and negative,
  lifecycle, stress, and performance cases.
- `crates/virtio-accel-conformance/src/numerics.rs` and TOSA fixture-generation tests: shared
  operator/dtype artifacts and provider-neutral oracles needed to prove parity.
- `docs/performance.md`, backend/support documentation, and examples: truthful final matrix and
  reproducible evidence.

## Main risks and stop conditions

- HTP may accept FP32 tensors while executing relaxed FP16 math. Any edge-oracle failure or
  inability to control/verify precision blocks FP32 advertisement.
- QAIRT's generic FLOAT8 type may not provide a public, client-visible E4M3/E5M2 selector on v73.
  Ambiguity blocks that FP8 format; it must not be guessed from hardware marketing.
- QNN quantization metadata can alter integer interpretation. If exact TOSA zero-point and INT32
  accumulation semantics cannot be expressed, INT8 `MATMUL` remains blocked until an explicit
  integer graph is available.
- Some QNN ops may differ on NaN propagation, broadcasting, axes, rounding, or output dtype. A
  native op name match is insufficient; failed semantics keep the TOSA cell unsupported or require
  an HTP-native decomposition.
- Forty-two operators substantially enlarge the native descriptor surface. Descriptor ownership,
  checked bounds, and `SAFETY.md` updates land before operator expansion, not afterward.
- Performance work must not replace correctness gates. A faster graph that changes TOSA semantics,
  stages direct buffers, or silently falls back is a failed implementation.
