# Accelerator backend implementer guide

This guide is for providers implementing `virtio_accel_core::Accelerator`. The trait is the native,
transport-independent boundary: backend crates do not decode wire frames, inspect virtqueues, map
guest object IDs, or depend on a host operating-system adapter.

The reusable `virtio-accel-conformance` crate exercises that boundary directly. It needs three
provider-owned inputs:

1. A factory that returns a fresh backend instance for each case.
2. One executable `TargetDescription` containing an artifact and its bindings — a single
   observable binding, or one fixture per slot for programs with disjoint input and output slots.
3. A `ConformanceHooks` implementation that advances a pending event to successful completion.

The same flow is available as a runnable crate-level example:
`cargo run --example backend_conformance`.

Both crates are published, so a provider outside this repository depends on them directly. The
conformance suite is test-only and belongs in `[dev-dependencies]`, which keeps it out of the
shipped dependency graph:

```toml
[dependencies]
virtio-accel-core = "0.1"

[dev-dependencies]
virtio-accel-conformance = "0.1"
```

`virtio-accel-mock` is also published, if a reference backend is useful to compare against while
bringing a provider up. Like the conformance suite it is test-only reference code, with
deterministic non-secret artifacts and scripted faults; see [../SECURITY.md](../SECURITY.md) for
what that does and does not cover.

The completion hook is test control, not a new production requirement. A backend with an external
scheduler can signal that scheduler; a deterministic backend can execute the retained invocation
directly. The target operation must remain pending until the hook runs, transform the fixture's
initial bytes into its expected bytes, and use the declared slot and access mode. Its observable
binding is `Write` or `ReadWrite`; a read-only binding cannot carry the required output evidence.

A program whose artifact declares disjoint input and output slots — the shape every lowered TOSA
graph produces — is described with `TargetDescription::with_bindings`, pairing
`BindingFixture::read_only` inputs with at least one observable writable fixture. The executing
cases bind every fixture in declared order and, after completion, verify every fixture's expected
bytes, so clobbered inputs fail the suite alongside wrong outputs.

```rust,ignore
use virtio_accel_conformance::{
    BindingFixture, ConformanceHooks, ProgramFixture, TargetDescription, run,
};

struct Hooks;

impl ConformanceHooks<MyBackend> for Hooks {
    fn complete_event(
        &self,
        backend: &MyBackend,
        event: &MyEvent,
    ) -> Result<(), virtio_accel_core::BackendError> {
        backend.test_control().complete(event)
    }
}

#[test]
fn backend_conformance() {
    let target = TargetDescription::new(
        ProgramFixture::new(FORMAT, TARGET, ARTIFACT, RESIDENT_BYTES).unwrap(),
        BindingFixture::new(
            SLOT,
            ACCESS,
            DOMAIN,
            ALIGNMENT,
            INITIAL_BYTES,
            EXPECTED_BYTES,
        )
        .unwrap(),
    );
    run(MyBackend::new, &target, &Hooks).assert_conformant();
}
```

The factory must not return a process-global singleton carrying state from another case. Fresh
instances are how the suite isolates indeterminate ownership and mirror the device recovery rule:
once reuse cannot be proved safe, discard the complete backend instance.

## Standard cases

Case IDs are stable diagnostic names. Mandatory cases never skip. Capability cases skip only when
the corresponding `DeviceInfo.capabilities` bit is absent, and the report records that reason.
Accounting is optional and is reported separately rather than disguised as a passing mandatory
case.

| Case ID | Requirement | Contract exercised |
|---|---|---|
| `metadata.stable-valid` | Mandatory | `device_info` succeeds, validates, and remains stable for one backend instance |
| `intent.reserved-flags` | Mandatory | Reserved context and execution-queue flags return `Unsupported` without resources |
| `memory.host` | `HOST_VISIBLE_MEMORY` | Honest host allocation metadata and successful release |
| `memory.device` | `DEVICE_LOCAL_MEMORY` | Honest device-local allocation metadata and successful release |
| `memory.shared` | `SHARED_MEMORY` | Host-visible, directly bindable shared allocation metadata |
| `buffer.segmented-transfer-bounds` | Mandatory | Segmented explicit transfers, exact bytes, and out-of-bounds rejection |
| `buffer.transfer-permissions` | Mandatory | Transfer-source and transfer-destination usage are enforced independently |
| `program.segmented-artifact-bounds` | Mandatory | Segmented artifact input, target loading, advertised artifact limit, and release |
| `submission.binding-validation` | Mandatory | Nonempty/unique slots, ranges, program access, valid execution, and acceptance truth |
| `submission.context-isolation` | Mandatory | A buffer from another context is rejected before admission |
| `event.pending-release-terminal-stability` | Mandatory | Pending release returns the live event, completion is stable, output is visible, and retry releases |
| `timeout.finite-admission` | Mandatory | Finite timeout results preserve rejected, accepted, and indeterminate ownership shapes |
| `event.cancellation-races` | `EVENT_CANCELLATION` | Cancellation-first and completion-first select one stable terminal result |
| `accounting.resource-lifecycle` | Optional hook | Fresh and post-case live/indeterminate provider totals are zero |

The target's memory domain must be advertised even though the three general memory-domain cases are
capability-conditional. This prevents an implementer from selecting an unusable target fixture and
then mistaking the resulting skips for execution evidence.

## Accounting hook

Implement `ConformanceHooks::resource_counts` when provider diagnostics can report retained native
resources. Return the sum of resources known live and resources whose release or admission is
indeterminate. Do not count an unknown resource as freed merely because its guest-visible ID was
invalidated.

The runner samples accounting before and after every case. A nonzero post-case count is attached to
that case's failure, so a semantic error that drops a Rust handle without crossing the provider
release boundary is also reported as a leak. The reference backend runs under
`virtio_accel_mock::fault::FaultAccelerator` to exercise this path.

## Trait obligations

### Discovery and capabilities

`device_info` is immutable for the backend instance. Every resource and byte limit is nonzero, at
least one provider-owned memory domain is available, and reserved capabilities are not advertised.
Advertise `EVENT_CANCELLATION` only when pending-event cancellation is implemented and never returns
`Unsupported`.

Capability bits are promises, not hints. An advertised memory domain must allocate the requested
backing or return a request-specific resource error; an unadvertised optional capability cannot be
used to skip mandatory context, transfer, program, queue, submission, polling, or release behavior.

TOSA providers may additionally implement `virtio_accel_tosa::TosaCapabilityProvider`. This is a
host-side artifact-planning interface, not a protocol or `DeviceInfo` capability. Return one
`CapabilityDescriptor` per exact target tier and keep role-specific dtypes, operator constraints,
shape limits, and runtime-condition policy conservative. An unavailable runtime returns an empty
slice. A positive descriptor query permits a load attempt only; `load_program` remains responsible
for concrete shape relationships, resources, native compilation, and device-state failures.

### Handles and ownership

Handles own provider state and borrow no call argument. Context children and event dependencies are
caller-managed, but accepted events must retain all provider invocation state until terminal release.
Do not use `Drop` timing as protocol state.

Creation errors retain no resource. Submission rejection proves that no work was accepted and no
event exists. If acceptance cannot be proved false, return `SubmitFailure::Indeterminate` with the
event. A rejected release returns the same live handle for retry; an indeterminate release consumes
the handle and requires complete backend recovery.

The suite retries one rejected release. It never retries an indeterminate operation on the same
backend instance.

### Buffers and copies

`BufferInfo` describes the backing that was actually allocated. Its descriptor exactly matches the
request, allocation size and alignment meet the request, and properties are truthful. Every
program-visible buffer reports `DIRECT_BINDING`; inability to bind the exact allocation is an
allocation or submission error, not permission for a hidden bounce buffer.

`write_buffer` and `read_buffer` are the only baseline bulk-copy boundaries. They accept segmented
ports and may use bounded staging for device-local transfers. Allocation, submission, polling, and
release do not copy complete bound ranges. The semantic suite validates direct-binding metadata,
observable output, and optional copy-path diagnostics; the v1 budgets in
[`performance-budgets.json`](../conformance/v1.0/performance-budgets.json) define the regression
thresholds.

### Programs, bindings, and admission

Artifacts are opaque to transports but not to the provider. Read segmented bytes without requiring
one artifact-sized coalescing allocation, validate the format/target envelope, and retain no borrow
of the source. `ArtifactRef::resident_bytes` is the caller-authorized upper bound for all storage
retained by the returned program handle, including compiled code and provider metadata attributable
to that program. Reject the load when that bound cannot be honored; do not treat it as an estimate.

Submission validates nonempty bounded bindings, unique slots, nonempty in-range regions, declared
buffer usage (access must be compatible with the buffer’s usage bits), program-specific
slot/access compatibility, and one context across queue, program, and buffers. Reject usage
mismatches before provider admission. Validation and admission are bounded. The borrowed binding
slice is not an owned per-binding mirror and must not survive the call as Rust references.

### Events, cancellation, and time

Polling is bounded and nonblocking. Once a terminal state is observed, every later successful poll
returns exactly that state. A pending event cannot be destroyed. Cancellation and completion race
to one terminal result: cancellation success means later polls are `Cancelled`; if completion won,
cancellation returns `Busy` and preserves the completed or failed result.

Timeouts are relative to backend admission. Never compare a guest duration with an absolute host
timestamp. A timeout before acceptance is rejected as `DeadlineExpired`; uncertainty after the
acceptance boundary is indeterminate and carries an event. The standard finite-timeout case accepts
any of the three truthful ownership shapes because provider scheduling policy is implementation
specific.

### Reset and device loss

`Accelerator` deliberately has no reset method. The command engine owns object IDs and teardown;
the provider owns native resource truth. Successful child-before-parent release permits backend
reuse. Device loss, indeterminate release, unresolved pending work, or accounting contradiction
requires discarding the entire instance and constructing a fresh one through the factory.

Use `virtio_accel_mock::fault::FaultAccelerator` or an equivalent provider-local injector to test
rejected and indeterminate outcomes at each native API boundary. The standard suite remains usable
without a vendor fault API, while the command-engine and full-stack fault tests prove the portable
recovery policy.

For a complete portable lifecycle without the conformance harness, run
`cargo run --example reference_execution`. It demonstrates the context, buffer, program, queue,
submit, poll, transfer, and teardown sequence against the mock backend. On macOS, run
`cargo run -p virtio-accel-coreml --example tosa_coreml` for the production artifact path from a
device-neutral TOSA graph through Core ML execution and the same backend lifecycle.

`cargo run -p virtio-accel-hexagon --example tosa_hexagon` exercises the QNN HTP lifecycle and
direct bindings on a configured Windows ARM64 QAIRT host. Without that build environment, it retains
the explicit unavailable-runtime surface.

## Common traps

- Advertising a capability because a slow fallback exists, while the fallback violates direct
  binding or synchronization semantics.
- Returning `Rejected` after a native queue accepted work or a timeout raced with admission.
- Dropping an event on failed destruction instead of returning it in `ReleaseFailure::Rejected`.
- Letting a second poll regress from complete, failed, or cancelled to pending.
- Accepting duplicate slots or matching bindings by array position rather than slot number.
- Retaining `ArtifactRef`, `BindingRef`, or byte-port references after their call returns.
- Adding a lock, atomic, allocation, or owned binding mirror to every submission when native handle
  ownership already provides the required exclusivity.
- Treating a Rust handle drop, ID invalidation, or reset request as proof that provider state was
  released.
