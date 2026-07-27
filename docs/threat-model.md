# Protocol 1.0 threat model and resource policy

Status: security model for the portable protocol 1.0 candidate. This document explains the threat
assumptions and maps them to normative requirements, implementation owners, and executable
evidence. The requirements themselves live in [specification.md](specification.md),
[wire-abi.md](wire-abi.md), and [virtqueue.md](virtqueue.md).

## Security goals

The portable device treats every guest-controlled byte, count, object ID, descriptor shape,
timeout, command ordering, and reset timing as hostile. Its goals are to:

- preserve memory safety without trusting wire values or guest mappings;
- keep guest-visible objects isolated by device, reset epoch, kind, and context;
- bound the CPU work and allocation attributable to one command;
- bound live object counts and provider-retained bulk storage;
- publish no uninitialized or ownership-ambiguous response;
- recover deterministically when provider ownership is known; and
- stop admitting work and require backend discard when ownership cannot be proven.

Availability of the host process or physical accelerator against a malicious host backend is not a
portable-layer guarantee. The device still preserves guest-visible ownership truth when a trusted
backend reports a hang, device loss, or indeterminate result.

## Trust boundaries

```text
untrusted guest
    |
    | descriptors, bytes, counts, IDs, timing
    v
transport adapter  -- validates/maps guest memory and queue topology
    |
    | address-free readable/writable ports and owned chain tokens
    v
command engine     -- validates frames, ownership, quotas, and resource policy
    |
    | semantic values and provider-owned handles only
    v
accelerator backend -- trusted implementation, failure-prone provider/device
    |
    v
trusted host process and platform integration
```

The guest is adversarial. The transport adapter, command engine, backend implementation, and host
process are trusted code, but mappings, allocations, provider calls, and hardware may fail. Safe
Rust prevents memory unsafety in the portable crates; it does not make a dishonest backend
conformant.

The transport adapter owns guest-address validation, access permissions, pinning, and base Virtio
ring synchronization. The command engine never receives an address. The backend receives neither
wire structures nor guest memory and must not retain borrowed byte ports after a call returns.

## In-scope threats and bounds

| Attacker-controlled dimension or failure | Bound and enforcement owner | Failure behavior | Executable evidence |
|---|---|---|---|
| Descriptor topology, direction, loops, and segmentation | The transport validates a flattened chain in one pass. `WireConfig::max_chain_descriptors` is 2 through `HARD_MAX_CHAIN_DESCRIPTORS` (256); the reference split queue also bounds traversal by its configured table. | Invalid topology completes without backend invocation; unusable chains write no bytes. | `chain_layout_rejects_direction_and_length_errors`, `vq_004_vq_005_vq_006_malformed_topology_is_classified_boundedly`, and `unusable_frames_write_nothing_and_never_become_dispatchable` |
| Request and response byte totals | Configuration limits are capped by `HARD_MAX_REQUEST_BYTES` and `HARD_MAX_RESPONSE_BYTES` (16 MiB each). Checked addition enforces exact frame lengths and response capacity before mutation. | Oversized or inconsistent frames return `RESOURCE_LIMIT` or `INVALID_ARGUMENT`; short success capacity writes nothing. | `edge_vectors_cover_limits_reserved_bits_and_exact_lengths`, `short_success_capacity_writes_nothing`, and `short_responses_and_backend_rejection_never_publish_state` |
| Variable binding arrays | `DeviceLimits::max_bindings_per_submission` is nonzero and at most `HARD_MAX_BINDINGS` (4096). Decode performs one fallible bounded reservation and sorts in place. Event metadata retains at most the same bounded ID count. | Limit excess returns `RESOURCE_LIMIT`; allocation failure returns `OUT_OF_MEMORY`; no backend call occurs. | `duplicate_submission_slots_are_rejected_before_dispatch`, `bindings_are_nonempty_bounded_and_unique`, and `submission_validation_rejects_before_backend_admission` |
| Object IDs and cross-context references | IDs are nonzero, namespace-, kind-, slot-, and generation-tagged. Slots retire before generation wrap. Every combined operation checks one context before provider invocation. | Zero or malformed IDs are invalid; stale, wrong-kind, cross-context, and reset-invalidated IDs are rejected without handle access. | `stale_ids_never_resolve_after_slot_reuse`, `exhausted_generations_retire_the_slot_before_an_id_can_revive`, and `wrong_kind_cross_context_and_cross_device_ids_fail_before_provider_use` |
| Live contexts, buffers, programs, execution queues, and events | Nonzero `DeviceLimits` cap contexts and each per-context child class. Aggregate table capacities use checked multiplication. Event/reference maxima are checked before state exists. | Exhaustion returns `RESOURCE_LIMIT` before provider invocation or table growth. Fallible table reservation maps to `OUT_OF_MEMORY`. | `limits_are_enforced_before_growth`, `invalid_limits_are_rejected_before_tables_exist`, and `quota_exhaustion_never_invokes_provider_callbacks` |
| Logical buffer size and transfer ranges | `max_buffer_bytes` caps each logical allocation. Nonzero ranges use checked end arithmetic and must fit the logical buffer. Usage bits gate read, write, and program access. | Oversize returns `RESOURCE_LIMIT`; arithmetic/range failure returns `INVALID_ARGUMENT` or `OUT_OF_BOUNDS`; permission mismatch returns `PERMISSION_DENIED`. | `transfer_range_overflow_is_a_protocol_error`, `semantic_validation_prevents_backend_calls`, and `mutable_state_allows_every_program_access_mode` |
| Aggregate provider buffer backing | A mandatory device-private `ResourcePolicy::max_buffer_backing_bytes` limits the sum of actual `BufferInfo::allocation_bytes`, not logical guest bytes. The state prechecks the logical lower bound before allocation and reconciles the actual charge before publishing an ID. | An over-budget allocation is released before its ID is exposed. Rejected or indeterminate cleanup makes the backend discard-required and retains or quarantines the exact charge. | `aggregate_policy_uses_actual_backing_and_charges_until_release_commits`, `aggregate_retained_byte_policy_cleans_up_before_exposing_an_id`, `rejected_cleanup_of_an_unpublished_allocation_requires_backend_discard`, and `indeterminate_cleanup_of_an_unpublished_allocation_quarantines_actual_backing` |
| Artifact input and resident program storage | `max_artifact_bytes` bounds the borrowed input payload. `ResourcePolicy::max_program_resident_bytes` bounds the aggregate declared resident charge before provider invocation. A conforming backend retains no more storage than `ArtifactRef::resident_bytes`. | Excess returns `RESOURCE_LIMIT` before loading. Invalid or unsupported artifacts create no program resource. | `byte_limits_and_resident_charges_have_distinct_semantics`, `program_policy_rejects_before_provider_invocation`, and `malformed_reference_artifacts_do_not_create_resident_programs` |
| Command-queue occupancy and guest request tracking | Virtio `QueueSize` bounds descriptor/ring ownership. `GuestConfig::max_inflight` is nonzero and no greater than queue capacity. The split reference model preallocates ring state at configuration. | Prepublication pressure returns caller ownership as retryable backpressure; it is not fabricated as a protocol response. | `publication_backpressure_returns_the_chain`, `prepublication_backpressure_returns_chain_and_operation`, and `queue_pressure_returns_caller_ownership_before_draining` |
| Submission and in-flight references | Per-context events and per-submission bindings bound retained invocation metadata. Queue, program, and buffer reference increments are checked before admission and released exactly once with the event. | Referenced objects return `BUSY`; rejected admission rolls back references; indeterminate admission retains an event ownership token. | `complete_lifecycle_tracks_children_references_and_release_rollback`, `rejected_and_indeterminate_submissions_preserve_the_admission_boundary`, and `finite_timeout_rejection_retains_all_referenced_resources` |
| Cancellation, completion, release, and reset races | One mutable command processor serializes semantic transitions. Provider events own native completion races. Reset makes one bounded child-before-parent pass and never waits for progress. | Rejected release restores the same ID; indeterminate release invalidates it and requires backend discard. A pending uncancellable event prevents backend reuse. | `accepted_events_retain_resources_and_resolve_cancel_completion_races`, `rejected_reset_release_is_quarantined_and_reset_is_idempotent`, and `reset_reclaims_completed_and_cancelled_events_exactly_once` |
| Response truncation and information disclosure | Command-specific capacity is preflighted before mutation. Payload bytes are initialized before the header commits their length; used length includes only initialized bytes. | Unusable capacity writes nothing. Post-mutation write failure sets recovery instead of claiming rejection. | `payload_guard_commits_the_preflighted_length`, `direct_payload_region_does_not_touch_excess_capacity`, and `response_failure_after_creation_requires_reset` |
| Backend device loss or ownership uncertainty | The acceptance and release result types distinguish rejected from indeterminate outcomes. Quarantined counts and retained-byte charges survive until the complete backend instance is discarded. | New semantic work stops. Reset is sticky, makes no later provider calls, and reports represented plus quarantined resources. | `event_faults_and_unreportable_admission_require_recovery`, `indeterminate_event_release_quarantines_the_complete_object_graph`, and `device_loss_crosses_backend_engine_and_guest_recovery_boundaries` |

All named tests are located in the corresponding crate's `src` unit tests,
[`crates/virtio-accel-device/tests/command_processor.rs`](../crates/virtio-accel-device/tests/command_processor.rs),
[`crates/virtio-accel-device/tests/state_model.rs`](../crates/virtio-accel-device/tests/state_model.rs),
or [`tests/portable_end_to_end.rs`](../tests/portable_end_to_end.rs). Coverage-guided malformed
input checks live under [`fuzz/`](../fuzz/): protocol decode is compared with the clean-room codec,
descriptor segmentation is driven through the split-queue model, and bounded stateful sequences
check resource accounting after every action. A fourth target drives the reference guest client
against a non-conforming device. That direction is reference-implementation robustness rather than
a portable-layer security boundary, because a malicious backend or host process is excluded below;
it exists so the driver-side obligations in this protocol — opaque unknown statuses, recovery on
unknown event states, bounded in-flight tracking, and epoch-scoped handle staleness — hold against
inputs no example test enumerates. The deterministic state-model suite runs a bounded
seed set in normal CI, asserts stale-ID retirement, context isolation, reference release, quota, and
retained-byte invariants after every action, and prints minimized replayable schedules for failures.
An ignored deep test provides manual or nightly-style exploration with `VIRTIO_ACCEL_STATE_MODEL_SEED`.

## Resource-accounting rules

`DeviceLimits` are backend capabilities and per-object/count bounds visible to the driver.
`WireConfig` bounds command framing and descriptor presentation. `ResourcePolicy` is host policy:
it is supplied when constructing a `CommandProcessor`, is never guest-controlled, and is not a
promise that an allocation will succeed.

The policy has two nonzero aggregate limits:

- `max_buffer_backing_bytes`: actual provider allocation bytes across all live buffer handles; and
- `max_program_resident_bytes`: declared resident charges across all live program handles.

`RetainedBytes` uses `u128` totals so the sum of a `u32` object count and `u64` per-object charges is
representable. A charge begins before the corresponding object ID is returned. It remains while a
handle is live or in the `Releasing` state and ends only when provider release succeeds. Rejected
release restores the handle without dropping its charge. Indeterminate release transfers the
charge to quarantine and requires the entire backend instance to be discarded.

Buffer padding is provider knowledge, so the command engine first checks the logical byte lower
bound, then reconciles the returned `BufferInfo::allocation_bytes`. If the actual charge crosses
the aggregate budget, the engine attempts release before exposing the new ID. Failure to prove that
cleanup succeeded cannot be represented as an ordinary `RESOURCE_LIMIT`; it is device loss.

Program resident storage is declared before provider invocation. `ArtifactRef::resident_bytes` is
the maximum storage the provider may retain for the returned program, including compiled code and
provider metadata attributable to that program. A backend that cannot honor the charge rejects the
load. Temporary provider memory used only during the synchronous call is governed by the backend
and host allocator, not counted as retained program storage. The portable layer cannot observe that
internal allocation, so a backend or isolated provider worker must enforce its own transient compile
memory and work budget when hostile artifacts can expand beyond their bounded input.

Object-table capacity and per-event binding vectors are bounded by count limits rather than byte
estimates because their Rust layout is implementation-specific. Quantitative retained metadata,
allocation, and copy-path budgets are tracked in [performance.md](performance.md); changing their
representation does not change the security bounds above.

## Progress and denial of service

Every portable parsing, lookup, queue, polling, cancellation, and reset operation is bounded by a
validated count and contains no retry loop waiting for guest or provider progress. Repeated hostile
commands can consume an unbounded amount of CPU over unbounded time; rate limiting belongs to the
transport or host integration because the portable layer has no clock, scheduler, tenant identity,
or worker policy. Rate limiting must not change ownership or fabricate successful responses.
Malformed guest values and configured resource exhaustion are ordinary protocol errors: they do not
panic, create an unbounded allocation, or expose uninitialized response storage.

Provider lifecycle and explicit transfer calls are synchronous and may block according to the
`Accelerator` contract. The portable layer cannot safely preempt arbitrary provider code or reclaim
native handles from a hung call. Integrations that require availability against provider hangs must
use an appropriate process or worker with CPU and memory limits, a watchdog, or a hardware-reset
boundary and discard the complete backend instance after loss. Turning provider calls into futures
would not itself enforce bounded polling or safe cancellation.

Timeouts are relative admission constraints, not host watchdogs. A zero timeout is infinite. After
admission, an uncertain timeout retains an event rather than permitting a duplicate submission.

## Exclusions

Protocol 1.0 does not claim protection against:

- a malicious backend or compromised host process;
- physical attacks, accelerator side channels, power analysis, or denial of service by hardware;
- confidentiality between mutually untrusted workloads inside one provider context unless the
  provider supplies that isolation;
- guest-memory pinning limits, IOMMU policy, or cache-coherency bugs in a future platform adapter;
- artifact-language safety inside an opaque vendor compiler or executable; or
- platform external-memory lifetime and synchronization, which remain an unadvertised feature.

These exclusions do not permit a portable adapter to weaken object ownership, response atomicity,
or resource accounting. A future feature that crosses one of these boundaries needs its own threat
model, negotiation, conformance evidence, and release audit.

## Review obligations

A change that adds a guest-controlled count, retained allocation, new descriptor topology, new
queue, new asynchronous ownership state, or new platform handle must update this document and map
the dimension to one authoritative limit. It must also add executable evidence and refresh the
normative requirement ledger. Silent defaults, duplicate conflicting caps, and best-effort cleanup
across an indeterminate ownership boundary are not conformant.
