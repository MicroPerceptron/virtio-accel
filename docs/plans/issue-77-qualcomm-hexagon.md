# Issue #77 plan: Qualcomm Hexagon NPU backend

Status: proposed implementation plan

Issue: [#77 — Build the Qualcomm Hexagon NPU backend](https://github.com/MicroPerceptron/virtio-accel/issues/77)

Planning branch: `codex/issue-77-qualcomm-hexagon-plan`

Last reviewed: 2026-08-17

## Outcome

Add a separately packaged `virtio-accel-hexagon` host-native crate that implements
`virtio_accel_core::Accelerator` over the Qualcomm Hexagon NPU. The first supported path will
execute a deliberately small, static TOSA 1.0 subset on a Snapdragon NPU, keep Qualcomm types and
unsafe code inside the adapter crate, and compile as an unavailable-runtime placeholder everywhere
else.

Completion means that the backend passes the mandatory semantic conformance suite, passes every
numerical corpus case it advertises, binds caller-owned provider allocations without a
submission-time provider bounce buffer, and has reproducible hardware evidence on the selected
Snapdragon device and runtime.

## Initial technical decisions

1. **Runtime:** use the public Qualcomm AI Runtime SDK (QAIRT), specifically the QNN C API with the
   HTP backend (`QnnHtp`), rather than SNPE, ONNX Runtime, or a framework delegate. QNN is the
   lowest public graph/runtime boundary intended to target the Hexagon NPU directly.
2. **First host/device tier:** target Windows 11 on ARM64 and Snapdragon X Series first. Add Android,
   Qualcomm Linux, and other Snapdragon/Dragonwing families only as later runtime adapters after
   the first tier is conformant. Record the exact SoC, Windows build, NPU driver, QAIRT release,
   QNN API version, and HTP library set used by the hardware lane.
3. **Artifact path:** accept TOSA bytes at the portable boundary, validate and analyze them with
   `virtio-accel-tosa`, then construct and finalize a QNN graph inside `load_program`. Do not add an
   ONNX/TFLite intermediate or invoke compilation from `submit`.
4. **Execution target:** load only the HTP backend and report only `AcceleratorClass::NPU`. Do not
   load QNN CPU or GPU backends as fallback paths.
5. **Precision:** advertise only cases demonstrated to preserve TOSA semantics on the chosen HTP.
   FP16 is the likely first floating-point tier. Reject FP32 during admission unless the selected
   SDK/device can prove FP32 computation without relaxed or forced FP16 math. Never relabel FP16 as
   FP32.
6. **Scheduling:** start with one bounded, serialized worker lane per detected device. `submit`
   validates and enqueues; the worker performs native dispatch/wait; `poll_event` only reads a
   latched state and never blocks or drives QNN.
7. **Buffers:** allocate stable, aligned, provider-owned host memory once, bind its exact range to
   QNN tensors, and retain it through terminal event release. Explicit reads and writes are the only
   provider copy boundaries. Runtime-internal transfers are recorded as a platform property, not
   disguised as provider zero-copy.
8. **Version policy:** pin one QAIRT/driver combination for hardware evidence. At initialization,
   validate the QNN provider/API version and required capabilities. A new QAIRT minor or driver
   becomes supported only after conformance and numerical regression runs; an untested ABI or
   missing symbol produces `RuntimeUnavailable`/`Unsupported`, not best-effort execution.

These choices follow Qualcomm's public description of QNN as the low-level, accelerator-specific
API within QAIRT and its documented HTP path for Snapdragon NPUs:

- [Qualcomm AI Engine Direct SDK](https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk)
- [Qualcomm AI Engine Direct/QNN documentation](https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-10/index_QNN.html?product=1601111740009302)
- [QAIRT and QNN terminology](https://dev.aihub.qualcomm.com/docs/)
- [Windows on Snapdragon AI](https://www.qualcomm.com/developer/windows-on-snapdragon/windows-on-snapdragon-ai)

## Phase 0: prove the hard constraints

Do this spike before building the full `Accelerator` implementation. Check the spike into the
backend's tests/examples or record its reproducible source and results; do not leave it as an
unreviewable local experiment.

- [ ] Install one redistributable QAIRT release on a Snapdragon X test system and inventory the
      headers, import libraries/DLLs, HTP support libraries, device driver, API version, and license
      constraints. Do not check vendor binaries or headers into this repository unless their license
      explicitly permits it.
- [ ] Use only QNN's HTP backend to create/finalize/execute a minimal identity graph. Confirm from
      runtime logs/profiling that no node is placed on CPU or GPU.
- [ ] Bind inputs and outputs directly from stable provider allocations. Record pointer/range and
      memory-registration behavior before and after execution. Prove that the adapter creates no
      temporary input/output tensor storage during submission.
- [ ] Exercise repeated and overlapping submissions to determine QNN graph/context thread-safety,
      whether execution must be serialized, and when QNN stops retaining tensor descriptors,
      client-buffer pointers, signals, and graph/context handles.
- [ ] Determine whether the selected API offers an honest cancellation primitive. If cancellation
      cannot be bounded and race-safe, leave `EVENT_CANCELLATION` unadvertised.
- [ ] Run the FP16 and FP32 identity fixtures and inspect the chosen QAIRT release notes/configuration
      for relaxed or forced FP16 computation. Advertise FP32 only if edge-value tests prove its
      semantics. Recent QAIRT notes about HTP floating-point behavior make this a release gate, not
      an assumption.
- [ ] Verify native support for the exact initial MATMUL and NHWC MAX_POOL2D shapes/attributes. A
      failing case narrows the advertised target; it is not silently skipped or rerouted.
- [ ] Measure the largest supported tensor/rank/alignment and any graph/context/queue limits that can
      be reported honestly through `DeviceLimits`.

Exit gate: a single HTP-only identity runs from a caller-owned provider allocation with stable
teardown, the precision claim is known, and the SDK redistribution/build strategy is acceptable.
If exact binding cannot be implemented with the public QNN API on this platform, stop and document
the blocked acceptance criterion before expanding the backend.

## Phase 1: scaffold a portable host-native crate

- [ ] Add `crates/virtio-accel-hexagon` with `Cargo.toml`, `README.md`, `SAFETY.md`, dual license
      files, `build.rs`, `src/lib.rs`, `src/ffi.rs`, `src/native.rs`, `src/lower.rs`, tests, and a
      `tosa_hexagon` example.
- [ ] Add the crate to workspace members and centralize `virtio-accel-core`,
      `virtio-accel-tosa`, and test-only `virtio-accel-conformance` dependencies like the existing
      host backends.
- [ ] Make `build.rs` probe the QAIRT/QNN include and library roots and verify the required QNN API
      headers/libraries. Support explicit controls such as `VIRTIO_ACCEL_HEXAGON=0|1`,
      `VIRTIO_ACCEL_QNN_SDK_ROOT`, and platform-specific library search overrides.
- [ ] Emit a private `va_hexagon` cfg only when all required build-time pieces exist. Forced-on mode
      must fail loudly; autodetect mode must compile the placeholder when dependencies are absent.
- [ ] Keep portable lowering/admission unit tests available without QAIRT. Under
      `not(va_hexagon)`, forbid unsafe code and expose constructors that consistently return
      `InitError::RuntimeUnavailable`.
- [ ] Confirm `cargo check`, `cargo test`, rustdoc, and clippy pass on x86_64 Linux, x86_64 Windows,
      macOS, and other normal CI hosts with no Qualcomm installation.

## Phase 2: confine and audit the QNN boundary

- [ ] Hand-bind only the QNN C ABI symbols required for backend/provider discovery, logging,
      device/platform information, backend/context/graph/tensor lifecycle, execution, signals or
      notification, memory registration if used, error reporting, and teardown.
- [ ] Load the versioned QNN interface through its provider/interface discovery entry point and
      validate major/minor compatibility before calling function pointers. Avoid binding internal
      or sample-only APIs.
- [ ] Wrap every native handle in one Rust owner with a documented destruction order. Model shared
      process/device state explicitly with `Arc`/`OnceLock` only where QNN requires it; do not infer
      that QNN globals are safely re-creatable.
- [ ] Map QNN failures into stable `BackendError`, `SubmitFailure`, and `ReleaseFailure` outcomes.
      Preserve the distinction between rejected work and work whose native acceptance is
      indeterminate.
- [ ] Write `SAFETY.md` alongside the FFI implementation. Cover ABI/version validation, function
      pointer lifetime, handle ownership, callback/worker synchronization, tensor descriptor
      lifetime, client-buffer lifetime and alignment, teardown order, thread-safety, and device-loss
      poisoning. Give every unsafe block a local `SAFETY:` justification tied to those invariants.

Exit gate: repeated initialization, discovery, and teardown return diagnostics to baseline under
unit/hardware stress, and unsupported hosts still compile without enabling unsafe modules.

## Phase 3: implement strict TOSA admission and QNN lowering

- [ ] Define one `TargetIdentity` for the first proven HTP tier. Use a separate target identity for
      any future precision/operator tier; capability expansion must not change an existing target's
      meaning.
- [ ] Reuse `virtio-accel-tosa` validation/analysis and require one static region/basic block,
      positive static shapes, no runtime obligations, supported profile/extension declarations,
      and an exact input/output slot order.
- [ ] Implement an explicit lowering table for only the proven cases: identity first, then
      non-square batched MATMUL, then NHWC MAX_POOL2D. Validate dtype, rank, layout, axes,
      kernel/stride/padding, accumulator behavior, and output shape before any native graph calls.
- [ ] Keep constants, QNN parameter objects, tensor names/descriptors, dimensions, and op configs
      owned by the program builder until QNN's documented copy/retention point.
- [ ] Construct and finalize the QNN graph in `load_program`; retain the finalized graph/context and
      a sorted immutable binding plan in the program object. No graph building, shape inference,
      compilation, or heap growth proportional to model structure may occur in `submit`.
- [ ] Reject unsupported FP32, INT8, INT4, FP8, dynamic shapes, profiles, extensions, layouts, ops,
      or attributes during admission with the repository's documented error classification.
- [ ] Add hardware-free tests for malformed artifacts, unsupported combinations, binding plans,
      QNN descriptor construction, overflow/limit checks, and deterministic lowering.

Exit gate: every accepted artifact has one deterministic slot/shape/access plan and a finalized HTP
graph; everything outside that target is rejected before execution.

## Phase 4: implement the `Accelerator` object model

Use the following ownership mapping:

| `Accelerator` object | Hexagon/QNN representation |
| --- | --- |
| Backend | One validated QNN HTP provider/device plus a bounded serialized execution lane |
| Context | Rust ownership/accounting scope backed by the minimum QNN context state required |
| Buffer | One aligned provider allocation, registration/mapping metadata, and atomic shared/exclusive in-flight guard |
| Program | Validated TOSA metadata, immutable slot plan, runtime identity, QNN context/graph, and retained native descriptors |
| Queue | Bounded admission channel and reusable pointer/descriptor scratch owned by the serialized lane |
| Event | Latched state plus owned program/buffer guards and native completion/error state until release |

- [ ] Discover the HTP device and report stable Qualcomm vendor/product identity,
      `AcceleratorClass::NPU`, truthful memory domains/alignment/limits, and only demonstrated
      capabilities.
- [ ] Implement contexts as ownership/quota scopes; avoid duplicating process-wide provider/device
      state or scarce QNN resources per guest context without evidence that it is required.
- [ ] Implement zero-initialized, fallibly allocated, aligned buffers with checked size/alignment and
      exact retained byte ranges. If QNN memory registration is required, register once at buffer
      creation and deregister exactly once after all event references are gone.
- [ ] Implement explicit chunked `write_buffer`/`read_buffer`, including any documented cache
      synchronization. Reject host transfers while an exclusive native writer is in flight.
- [ ] Allow overlapping read-only bindings when QNN permits them; enforce one writer or read-write
      user with `Busy` on conflicts. Validate ownership, access mode, range, alignment, slot,
      dtype/shape byte size, and duplicate/conflicting bindings before native acceptance.
- [ ] Bound contexts, buffers, programs, queues, queued submissions, events, and retained bytes.
      Make all size/count arithmetic checked and surface exhaustion without partial mutation.

Exit gate: the full object lifecycle works with the identity program and no resource, registration,
or guard leak remains after success, error, explicit release, or backend discard.

## Phase 5: asynchronous execution and failure semantics

- [ ] Have `submit` complete validation and conflict reservation before enqueueing. Once native
      acceptance may have occurred, retain an event and report indeterminate ownership when the
      contract requires it.
- [ ] Use a fixed-capacity channel/ring and one worker lane initially. The worker binds the exact
      buffer ranges, dispatches the finalized graph, waits outside `submit`/`poll_event`, performs
      required output synchronization, releases in-flight guards, and publishes one terminal state.
- [ ] Make `poll_event` a wait-free/nonblocking read of a latched pending/success/error state. Stable
      terminal polling must never call back into QNN.
- [ ] Enforce finite timeouts honestly. Reject before admission when possible; after native
      acceptance, keep ownership until terminal completion. Map a proven bounded cancellation API
      to `DeadlineExpired`; otherwise document that the timeout cannot revoke native work and do not
      advertise cancellation.
- [ ] Poison the backend/device after unrecoverable HTP loss or a completion result that makes
      resource ownership unknowable. New work fails until the complete backend is discarded.
- [ ] Order terminal publication after all output visibility work and binding-guard release. Order
      event teardown so QNN can no longer access descriptors or allocations before Rust releases
      them.

Exit gate: submissions are nonblocking, terminal states are stable, timeout/indeterminate paths
preserve ownership, and fault injection cannot produce use-after-free or double release.

## Phase 6: conformance and numerical evidence

- [ ] Add backend-local integration tests modeled on `virtio-accel-openvino/tests/openvino.rs`, with
      native execution gated by `va_hexagon` and runtime/device availability.
- [ ] Run every mandatory `virtio-accel-conformance` case. Implement hooks for resource counts,
      submission-path diagnostics, completion, and fault/device-loss behavior. Every conditional
      skip must contain an explicit capability reason.
- [ ] Prove exact provider binding by recording allocation identity/range at bind and dispatch time,
      with `provider_staged_submissions == 0`. Test input/output offsets and incompatible ranges, not
      only whole-buffer bindings.
- [ ] Cover malformed and unsupported TOSA, duplicate/missing bindings, wrong access/shape/size,
      read/read overlap, read/write conflict, host-transfer conflicts, bounded admission, finite
      timeouts, stable terminal polling, device loss, rejected versus indeterminate submissions,
      and all realizable release failures.
- [ ] Run all advertised shared numerical fixtures on hardware and compare to their oracle,
      including edge-value precision cases. An advertised dtype/operator pair must have no silent
      skips.
- [ ] Stress repeated load/unload, submit/complete/release, context destruction, and backend discard.
      Assert QNN handles, registrations, events, allocations, and worker jobs return to baseline.
- [ ] Add ignored release-mode measurements for admission latency, submit-to-complete latency,
      retained resource high-water marks, and provider staging counts. Record hardware/runtime
      identity with the results.

Exit gate: mandatory semantic conformance passes; every advertised numerical case passes on the
selected Snapdragon NPU; resource and direct-binding diagnostics return to baseline.

## Phase 7: documentation, CI, and release integration

- [ ] Add `tosa_hexagon` as a backend-local end-to-end example. Without QAIRT it must print a clear
      unavailability message and exit successfully; on supported hardware it must execute on HTP and
      verify its result.
- [ ] Document SDK acquisition, licensing, environment variables, supported Snapdragon devices,
      Windows/driver/QAIRT versions, HTP support-library deployment, runtime logging needed to prove
      placement, build/test commands, target identity, precise operator/dtype boundary, and known
      limitations in the crate README.
- [ ] Update root `README.md`, `CONTRIBUTING.md`, `docs/architecture.md`, `docs/portability.md`,
      `docs/performance.md`, `docs/public-api.md`, `docs/backend-implementer-guide.md`, the pull
      request template, and CI examples. Change the Qualcomm support-table row from `Planned` only
      for cases backed by hardware tests.
- [ ] Update the release-policy crate count/order, `ci/check-release-policy.py` package and unsafe
      exception allowlists, `ci/publication.py`, workspace metadata, lockfile, packaging tests, and
      ordered local-registry dry run.
- [ ] Add portable CI for format, clippy, unit tests, docs, placeholder builds, forced-off builds,
      and package verification on hosts without QAIRT.
- [ ] Add a hardware workflow only when a controlled Snapdragon runner exists. Pin its OS image,
      NPU driver, QAIRT release, and SDK checksum; avoid storing licensed SDK payloads in the repo or
      ordinary public CI artifacts.
- [ ] Run the workspace release checks and document whether adding this adapter is a Cargo minor
      with no protocol change. Do not modify the virtio wire ABI or portable `Accelerator` contract
      solely for QNN.

Exit gate: portable CI is green without Qualcomm dependencies, the pinned hardware lane is green or
its exact manual replacement commands are published, packaging succeeds, and support claims match
the evidence.

## Expected file impact

New paths:

- `crates/virtio-accel-hexagon/**`
- `docs/plans/issue-77-qualcomm-hexagon.md`

Existing paths expected to change during implementation:

- `Cargo.toml` and `Cargo.lock`
- `README.md` and `CONTRIBUTING.md`
- `.github/workflows/ci.yml` and `.github/PULL_REQUEST_TEMPLATE.md`
- `ci/check-release-policy.py` and `ci/publication.py`
- `docs/architecture.md`, `docs/backend-implementer-guide.md`, `docs/performance.md`,
  `docs/portability.md`, `docs/public-api.md`, and `docs/release-policy.md`

Portable protocol, guest, device, and transport crates should not need Qualcomm-specific changes.
Any discovered need to change them is a design-review stop rather than routine scope.

## Recommended implementation slices

1. **Feasibility evidence:** HTP-only identity, exact buffer binding, precision result, runtime/license
   inventory.
2. **Portable scaffold:** crate, build probe, placeholder, lowering data structures and unit tests.
3. **Native lifecycle:** audited FFI, discovery, contexts, allocations, transfers, graph load/release.
4. **Execution semantics:** bounded worker, submission/event lifetime, conflicts, timeout/device-loss
   behavior.
5. **Coverage:** MATMUL/MAX_POOL2D lowerings, semantic conformance, numerical corpus, stress and
   diagnostics.
6. **Productization:** example, docs, CI/hardware evidence, performance notes, package/release policy.

Each slice should leave unsupported capabilities unadvertised and keep workspace builds green on a
host with no QAIRT installation.

## Definition of done

- [ ] A supported Snapdragon host enumerates an honest Qualcomm Hexagon NPU device and runs the
      documented TOSA example through QNN HTP.
- [ ] The complete workspace builds, lints, tests, documents, and packages without Qualcomm
      dependencies.
- [ ] Every mandatory semantic conformance case passes or has a contract-authorized, explicit
      capability skip.
- [ ] Every advertised TOSA operator/dtype case passes the shared numerical oracle on hardware.
- [ ] Provider diagnostics prove direct caller-allocation binding with no submission-time provider
      staging.
- [ ] Timeout, stable polling, binding conflicts, device loss, rejected/indeterminate ownership,
      and exact-once teardown are tested.
- [ ] Unsafe QNN interaction is confined and audited in `SAFETY.md`.
- [ ] Documentation and the root support matrix claim no more than the pinned evidence proves.
- [ ] No Qualcomm API or type leaks into a portable crate, and protocol 1.0 bytes/semantics remain
      unchanged.

## Open questions to close during Phase 0

- Which exact Snapdragon X SoC and test machine will be the baseline hardware runner?
- Which QAIRT release and Windows NPU driver combination is redistributable and supportable for the
  repository's public CI/release process?
- Does that QNN HTP combination preserve FP32 semantics, or must the first target advertise FP16
  only?
- Which public QNN allocation/registration path gives stable caller-owned buffers on Windows ARM64,
  and what alignment/cache rules apply?
- At what calls does QNN copy versus retain graph configs, tensor descriptors, dimensions, names,
  client buffers, and completion objects?
- Can QNN cancel accepted HTP work with bounded race-safe semantics, or should the backend expose
  timeout observation without cancellation?
- Which MATMUL and MAX_POOL2D attributes/shapes compile on the baseline HTP without CPU/GPU
  partitioning?
- Is one process-wide backend/device/context required by the selected QNN release, or can contexts be
  safely isolated without duplicating scarce hardware state?
