# virtio-accel portable protocol foundations

Status: portable protocol 1.0 candidate. Implementation conformance and the final freeze audit
remain in progress.

This document defines the portable semantic foundation for `virtio-accel`. It does not assign a
Virtio device ID and is not an OASIS Virtio specification. The transport model is intended to remain
compatible with the general facilities defined by the
[Virtio specification](https://docs.oasis-open.org/virtio/virtio/v1.3/virtio-v1.3.html).

Sections 1 through 11, [wire-abi.md](wire-abi.md), and [virtqueue.md](virtqueue.md) are normative for
the portable protocol 1.0 candidate. Appendix A maps the current Rust surface to the terms in this
document and is non-normative implementation guidance.

## 1. Normative language

The key words **MUST**, **MUST NOT**, **REQUIRED**, **SHOULD**, **SHOULD NOT**, **MAY**, and
**OPTIONAL** are to be interpreted as described by RFC 2119 and RFC 8174 when, and only when, they
appear in bold capitals.

Explanatory paragraphs beginning with “Rationale” are non-normative.

## 2. Scope

The portable v1 contract defines:

- device discovery and protocol compatibility;
- contexts and context-scoped resource ownership;
- buffers and bounded byte transfers;
- opaque program artifacts;
- accelerator execution queues;
- submissions, bindings, relative timeouts, and asynchronous events;
- cancellation when supported;
- explicit destruction, reset, and failure recovery; and
- a single baseline command virtqueue carrying request and response frames.

The portable v1 contract deliberately does not define:

- a standardized accelerator graph, compiler IR, or executable format;
- an OASIS device ID or standards submission;
- PCI, MMIO, vhost-user, QEMU, kernel, DRM, VFIO, or hypervisor integration;
- operating-system or vendor accelerator APIs;
- DMA-BUF, platform shared handles, zero-copy external memory, cache-coherency protocols, or
  timeline fences;
- packed virtqueues; or
- process, thread, executor, or async-runtime policy.

Adding an integration listed above **MUST NOT** change the portable object lifecycle by convention.
If it changes driver/device-visible behavior, it requires a negotiated feature or a new protocol
version.

## 3. Roles and boundaries

### 3.1 Driver

The **driver** is the request initiator. It discovers the device, validates protocol compatibility,
negotiates Virtio features, constructs request frames, publishes descriptor chains, validates
responses, and owns guest-side request tracking.

The driver **MUST** treat all device-written bytes as untrusted.

### 3.2 Device

The **device** owns the guest-visible protocol state. It validates descriptor direction and frame
bytes, maps opaque object IDs to live resources, enforces limits and context isolation, invokes an
accelerator backend, writes responses, and performs reset.

The device **MUST** treat all driver-written bytes, lengths, counts, object IDs, flags, and timing as
untrusted.

### 3.3 Transport adapter

A **transport adapter** maps a concrete Virtio implementation to portable readable/writable regions,
queue notification, used-length reporting, and reset operations.

A transport adapter **MUST NOT** expose guest addresses, descriptor objects, host mappings, or
transport-specific synchronization types to the accelerator backend.

### 3.4 Command engine

The **command engine** is the transport-neutral device state machine. It decodes one validated
request, performs object and quota checks, invokes the backend, updates ownership state, and produces
one response.

The command engine **MUST NOT** depend on an operating system, VMM, kernel, guest-memory crate, vendor
API, or compiler API.

### 3.5 Accelerator backend

The **accelerator backend** implements device-local execution semantics using provider-owned native
handles. It is represented by `virtio_accel_core::Accelerator`.

A backend **MUST NOT** receive wire structures, object IDs, guest addresses, or virtqueue
descriptors. It receives validated semantic values and provider-owned handles.

## 4. Terminology and object model

### 4.1 Device instance

A **device instance** is one initialized protocol endpoint and its backend. It owns:

- one device identity and set of semantic capabilities;
- advertised limits;
- one object-ID namespace;
- zero or more contexts; and
- one reset epoch.

Object IDs are meaningful only within the device instance and reset epoch that created them.

### 4.2 Command virtqueue

The **command virtqueue** is the Virtio transport queue that carries request and response frames.
Queue index zero is the only command virtqueue in the mandatory baseline.

The command virtqueue is not an accelerator execution queue.

### 4.3 Context

A **context** is the isolation and ownership parent for buffers, programs, accelerator execution
queues, and the events created by its submissions.

A device **MUST** reject an operation that combines objects from different contexts before invoking
the backend. A context **MUST NOT** be destroyed while any child object or in-flight reference
remains live.

### 4.4 Buffer

A **buffer** is a bounded backend allocation with:

- a nonzero byte length;
- a nonzero power-of-two alignment;
- a memory domain; and
- declared usage flags.

The buffer’s guest-visible object ID is not an address. Every transfer and binding range **MUST** fit
within the buffer without integer overflow.

Memory domains are strict allocation requirements:

- `Host` requests provider memory optimized for host transfers;
- `Device` requests the provider's accelerator-local placement class; and
- `Shared` requests one provider-owned allocation that is both host visible and directly bindable
  by the accelerator.

`Shared` does not mean guest-memory import, cross-process export, a platform shared handle, or
implicit cache coherence. Those operations require the reserved external-memory feature and have no
protocol 1.0 semantics.

An allocation result **MUST** report the requested descriptor, actual retained allocation bytes,
actual guaranteed alignment, and honest backing properties. Actual bytes **MUST NOT** be smaller
than the logical buffer and actual alignment **MUST NOT** be weaker than requested. Device resource
accounting uses actual retained bytes rather than assuming logical bytes include provider padding.
A provider **MUST NOT** return an allocation whose reported placement can be achieved only by
copying through another full-size allocation during submission.

A buffer whose usage includes program input, program output, or mutable state **MUST** report direct
binding. A compatible submission **MUST** bind the exact provider allocation without copying the
bound range into or out of another allocation. If the provider cannot satisfy that invariant, it
**MUST** reject allocation. If a particular program is incompatible with an otherwise valid buffer,
submission **MUST** be rejected as `INCOMPATIBLE`; the provider **MUST NOT** silently stage it.

`WRITE_BUFFER` and `READ_BUFFER` are the only baseline operations that explicitly transfer buffer
contents. A write requires `TRANSFER_DESTINATION`; a read requires `TRANSFER_SOURCE`. Device-local
memory may use bounded provider staging during these explicit transfers, but no transfer may retain
the request or response byte region after the backend call returns.

The semantic transfer API accepts bounded byte sources and sinks that may be segmented. This avoids
requiring the command engine to allocate a contiguous copy of a valid transfer payload or response
before calling the backend. A contiguous region remains available to providers as an optional fast
path.

### 4.5 Program

A **program** is a resident backend object created from an opaque artifact format, opaque target
identity, payload, and declared resident-byte requirement.

The transport **MUST NOT** interpret vendor artifact contents. Artifact-format adapters own their
validation beyond the portable envelope fields.

The semantic artifact payload is a bounded byte source rather than a required contiguous slice.
Providers may inspect an available contiguous view or read segmented payload bytes directly into
final resident storage.

### 4.6 Accelerator execution queue

An **accelerator execution queue** is a context-scoped backend object used to submit programs for
execution. It is created by the protocol `CreateQueue` command and represented by
`Accelerator::Queue`.

It is unrelated to a Virtio queue index. Documentation and APIs **SHOULD** use “execution queue”
whenever omitting “accelerator” would make the distinction unclear.

### 4.7 Binding

A **binding** associates one program slot with a nonempty buffer range and an intended access mode
for one submission.

Binding slot numbers **MUST** be unique within a submission. A binding does not transfer ownership
of its buffer, but the device **MUST** retain the buffer until the resulting event is safely
reclaimed.

Read access requires a buffer declared for program input or mutable state. Write access requires a
buffer declared for program output or mutable state. Read-write access requires mutable state.
Binding-array order has no semantic meaning; the slot number identifies the program argument.

### 4.8 Submission

A **submission** is an attempt to admit one program, execution queue, binding list, and relative
timeout to the backend.

Admission has an explicit acceptance boundary:

- **rejected** means the backend guarantees that execution was not accepted and no operation
  resources were retained;
- **accepted** returns an event; and
- **indeterminate** means acceptance cannot be established and therefore also returns an event that
  owns the retained resources.

The command engine **MUST NOT** convert an indeterminate submission into an ordinary error.

### 4.9 Event

An **event** is the ownership and completion token for one accepted or indeterminate submission. Its
state is pending, complete, failed, or cancelled.

An event **MUST** retain its execution queue, program, buffers, and per-invocation backend state until
it is terminal and successfully destroyed. A pending event **MUST NOT** be destroyed.

Terminal event states are stable. Cancellation and completion select exactly one terminal result:
if cancellation wins, the cancellation command succeeds and polling reports cancelled; if
completion wins, cancellation returns `BUSY` and polling reports the completed or failed result.

### 4.10 Request and response

A **request** is one driver-readable command frame identified by a request ID. A **response** is the
corresponding device-written frame with the same request ID.

Request IDs correlate command completion only. They are not object IDs or execution event IDs.

### 4.11 Reset

A **reset** stops admission, disposes or quarantines in-flight state according to its known ownership,
invalidates every guest-visible object ID, advances the device epoch, resets command queues, and
returns the device to its initial negotiation state.

The transport **MUST** stop fetching command chains and publishing completions before portable
object teardown begins. Teardown **MUST** be bounded and child-before-parent: events precede their
execution queues, programs, and buffers, and all context children precede the context.

The device **MAY** reuse a backend instance only when every known resource is released
successfully. An unresolved pending event, a rejected reset-time release, an indeterminate release,
backend device loss, or an accounting contradiction requires discarding the complete backend
instance. Once discard is required, repeated reset attempts **MUST NOT** invoke that backend again.

Successful reinitialization **MUST** use a fresh nonzero object namespace. No object ID created
before reset may resolve after reset.

## 5. Baseline capabilities and feature policy

### 5.1 Mandatory baseline

The portable v1 baseline **MUST** provide:

- command virtqueue zero;
- fixed-width little-endian request and response headers;
- device information;
- context creation and destruction;
- buffer allocation, destruction, and bounded read/write transfer;
- opaque program loading and destruction;
- accelerator execution queue creation and destruction;
- submission with bounded bindings and relative timeout;
- event polling and destruction;
- stale-object and cross-context rejection; and
- whole-device reset.

The command opcode for cancellation is part of the baseline namespace. If the backend does not
advertise semantic event-cancellation capability, the device **MUST** return `UNSUPPORTED` without
changing the event.

### 5.2 Transport feature bits

Transport feature bits alter driver/device protocol behavior. The mandatory baseline has no
device-specific transport features: `BASELINE_FEATURES` is empty.

The currently reserved bits `MULTI_QUEUE`, `EVENT_QUEUE`, `EXTERNAL_MEMORY`,
`TIMELINE_FENCES`, and `SECURE_CONTEXTS` reserve numeric positions only. A baseline device **MUST NOT**
advertise them and a baseline driver **MUST NOT** accept them. Protocol 1.0 assigns no semantics to
these positions.

Unknown device-specific feature bits **MUST NOT** be accepted by a driver. A device **MUST NOT**
offer a feature it cannot honor if accepted.

### 5.3 Semantic backend capabilities

Backend `Capabilities` report whether a semantic operation or resource class is supported. They do
not independently change wire framing.

Protocol 1.0 assigns these semantic capability bits:

| Bit | Name | Meaning |
|---:|---|---|
| 0 | `HOST_VISIBLE_MEMORY` | `MemoryDomain::Host` allocation is supported |
| 1 | `DEVICE_LOCAL_MEMORY` | `MemoryDomain::Device` allocation is supported |
| 2 | `EVENT_CANCELLATION` | Pending events may support `CANCEL_EVENT` |
| 5 | `SHARED_MEMORY` | Provider-owned `MemoryDomain::Shared` allocation is supported |

Semantic bits 3 (`EXTERNAL_MEMORY`) and 4 (`SECURE_CONTEXTS`) are reserved until their transport,
ownership, synchronization, and isolation rules are specified. A protocol 1.0 device **MUST NOT**
advertise either bit.

The device **MUST** reject an allocation for an unsupported memory domain before invoking the
backend. Advertising a memory-domain capability commits the backend to the corresponding allocation
properties and direct-binding rules from section 4.4; capability reporting is not permission to
substitute a staged implementation.

If enabling a backend capability would require different descriptor direction, additional queues,
new synchronization, or changed lifetime rules, the device **MUST** also negotiate an appropriate
transport feature.

The exact secure-context and execution-queue-flag semantics remain tracked by issue #21. Protocol
1.0 accepts only the wire values explicitly listed in [wire-abi.md](wire-abi.md); the reference
implementation **MUST NOT** translate an underspecified capability into additional wire behavior.

### 5.4 Request and object flags

All request-header flags are zero in protocol 1.0. The former draft `NO_WAIT` position, bit zero, is
reserved and **MUST** be zero.

Unknown request, context, execution-queue, submission, buffer-usage, or binding-access flag bits
**MUST** be rejected before backend invocation. Reserved fields **MUST** be zero.

## 6. Versioning and compatibility

### 6.1 Candidate version

This specification assigns version `1.0` to the candidate exercised by the conformance artifacts.
Candidate implementations **MUST** expose major `1` and minor `0` in device-specific configuration
and **MUST NOT** claim candidate protocol 1.0 conformance unless they satisfy the normative wire,
queue, lifecycle, and compatibility requirements.

Protocol 1.0 does not become a stable compatibility promise until the final audit tracked by issue
#33. Before that audit, an intentional candidate revision **MUST** follow the coordinated procedure
in [wire-abi.md](wire-abi.md) and **MUST NOT** be described as a frozen or stable release.

### 6.2 Stable major versions

For protocol major one and later:

- a driver **MUST** reject a device with an unsupported major version;
- a device minor version is backward compatible with all earlier minor versions of the same major;
- a driver **MUST NOT** use behavior newer than the device minor version it observed; and
- feature negotiation remains required even when both endpoints know a feature’s numeric bit.

A major version changes when compatibility cannot be preserved through a new opcode, a new
negotiated feature, or a previously reserved value.

### 6.3 Extending existing frames

The config space does not communicate the driver’s supported minor version back to the device.
Therefore a minor version alone **MUST NOT** append fields to an existing response that an older
driver would receive.

Within a stable major version, an existing request or response payload length is immutable unless a
negotiated feature explicitly selects a different layout. Extensions **SHOULD** use a new opcode or
feature-gated payload.

Without such a feature, a receiver **MUST** require the exact payload length for the opcode and
**MUST** reject trailing bytes. It **MUST NOT** silently reinterpret or ignore an unknown tail.

### 6.4 Unknown numeric values

- An unknown opcode produces `UNSUPPORTED` and no semantic state change.
- An unknown request or object flag bit produces `UNSUPPORTED` and no backend call.
- An unknown object ID or zero object ID produces the specified invalid/stale-object failure.
- An unknown response status is an opaque failure. A driver **MUST NOT** treat it as success or
  construct an invalid Rust enum value.
- Unknown accelerator classes and provider-owned artifact formats remain representable as opaque
  numeric values, but using them may produce `UNSUPPORTED` or `INCOMPATIBLE`.

## 7. Ownership and destruction

The device owns the mapping from opaque object IDs to backend handles. The mapping **MUST** reject:

- zero IDs;
- stale generations;
- wrong resource kinds;
- IDs from another context; and
- IDs invalidated by reset.

Object-ID encoding is implementation-private. Drivers **MUST NOT** infer slot, kind, generation, or
address information from an ID.

Destructive backend calls consume handles and have an explicit release boundary:

- a rejected release returns the still-live handle and the device **MUST** restore it for retry;
- a successful release removes the object permanently; and
- an indeterminate release invalidates the guest ID and requires recovery. The device **MUST NOT**
  reuse or free the handle based on an assumption.

`Drop` may provide defensive cleanup inside an implementation but **MUST NOT** be used as
guest-visible protocol state.

## 8. Time and progress

Wire timeouts are relative nanosecond durations measured from backend admission. Zero means
infinite. Absolute guest timestamps **MUST NOT** be compared with a host clock because their
monotonic epochs are unrelated.

Polling an event **MUST** be nonblocking and bounded. The portable contract creates no background
thread and selects no async executor.

A timeout before admission is rejected. A timeout or communication failure after admission that
cannot prove rejection is indeterminate and **MUST** retain an event.

## 9. Errors

The portable status namespace distinguishes unsupported behavior, incompatibility, invalid
arguments, out-of-bounds access, busy resources, host allocation failure, configured resource-limit
exhaustion, deadline expiration, device loss, permission failure, stale objects, and internal
errors.

A device **MUST** map errors deterministically and **MUST NOT** expose uninitialized response bytes.
Provider-specific `External` domains and codes are not part of the baseline wire ABI; absent a
future negotiated diagnostic extension, they map to `INTERNAL_ERROR`.

An error response **MUST NOT** imply that a backend operation was rejected when acceptance was
indeterminate.

## 10. Portability requirements

Portable crates **MUST NOT** select an operating system, VMM, kernel, vendor API, or global runtime.

- `virtio-accel-proto`, `virtio-accel-transport`, and `virtio-accel-core` require neither `std` nor
  `alloc`.
- `virtio-accel-device` and the future reference guest/queue layers may require `alloc` but not
  `std`.
- reference host tooling and mock backends may require `std`.

Platform adapters depend inward on the portable layers. A portable layer **MUST NOT** conditionally
expose different semantics by host OS.

## 11. Tracked completion work

This document, [wire-abi.md](wire-abi.md), and [virtqueue.md](virtqueue.md) define the current
driver/device protocol candidate. The following implementation, provider, and verification details
remain explicitly tracked rather than silently decided here:

- issue #21 completes the backend capability, memory-domain, execution-queue, blocking,
  concurrency, and release semantics before the wire contract freezes;
- issue #25 turns the trust assumptions into enforceable resource and threat-model limits; and
- issue #32 defines post-1.0 semver and wire-evolution policy; and
- issue #33 performs the final protocol and API audit, including independent clean-room review,
  before freezing protocol 1.0.

No implementation **MAY** advertise a reserved optional feature merely because a Rust constant
records its numeric position.

## Appendix A: Rust surface mapping

This appendix is non-normative. It ensures every current public concept has an explicit place in the
model or is identified as an implementation helper.

| Rust concept | Protocol term or classification |
|---|---|
| `BackendError` | Backend failure taxonomy translated by the command engine into portable status |
| `AcceleratorClass` | Extensible device class in device identity |
| `Capabilities` | Semantic backend capabilities; not Virtio feature bits |
| `DeviceIdentity` | Device instance identity |
| `DeviceLimits` | Context, buffer, program, execution-queue, event, binding, and byte limits enforced before backend invocation |
| `DeviceInfo` | Device identity, semantic capabilities, and limits |
| `ContextFlags`, `ContextDesc` | Context creation intent; protocol 1.0 accepts only empty context flags |
| `MemoryDomain` | Strict provider-owned buffer placement requirement |
| `BufferUsage`, `BufferDesc`, `BufferRange` | Buffer allocation intent and checked byte range |
| `BufferProperties`, `BufferInfo`, `AllocatedBuffer` | Verified backing properties kept out of the backend hot path |
| `AccessMode` | Binding read/write intent |
| `ByteSource`, `ByteSink` | Bounded contiguous-or-segmented bulk byte ports |
| `ArtifactFormat`, `TargetIdentity`, `ArtifactRef` | Opaque program artifact description over a byte source |
| `QueueFlags`, `QueueDesc` | Accelerator execution-queue creation intent, not a Virtio queue |
| `Timeout` | Relative admission timeout; zero on wire means infinite |
| `BindingRef`, `validate_bindings` | Validated semantic binding and implementation helper |
| `EventState` | Event completion state |
| `SubmitFailure` | Rejected versus indeterminate submission acceptance boundary |
| `ReleaseFailure` | Rejected versus indeterminate resource-release boundary |
| `Accelerator` | Accelerator backend contract |
| `DeviceHealth` | Running, known-state reset required, or backend-discard-required processor state |
| `ResourceCounts`, `ResetDisposition`, `ResetReport`, `ResetError` | Reset accounting, reuse decision, and namespace validation |
| `Le16`, `Le32`, `Le64` | Wire implementation aliases; not semantic API |
| `PROTOCOL_MAJOR`, `PROTOCOL_MINOR` | Candidate protocol version 1.0 |
| `COMMAND_QUEUE` | Baseline command virtqueue index |
| `BASELINE_COMMAND_QUEUES`, `HARD_MAX_*`, `MIN_MAX_*` | Candidate queue, frame, binding, and configuration bounds |
| `KNOWN_*_BITS`, `RESERVED_REQUEST_FLAG_NO_WAIT` | Assigned and reserved-zero flag namespaces |
| `FeatureBits` | Device-specific Virtio transport features |
| `BASELINE_FEATURES` | Mandatory feature set, currently empty |
| `RESERVED_FEATURES` | Reserved namespace that must not be advertised |
| `RequestFlags` | Per-request wire flags; empty in protocol 1.0 |
| `KnownOpcode`, `UnknownOpcode` | Validated command namespace and opaque unknown opcode |
| `StatusCode` | Extensible response status namespace |
| `WireConfig`, `ConfigError` | Device-specific configuration and executable validity rules |
| `RequestHeader`, `ResponseHeader` | Request and response frame headers |
| `WireDeviceInfo` | Device-information response payload |
| `CreateContextRequest` | Context-create request payload |
| `ObjectPayload` | One opaque object ID in a request or response |
| `AllocateBufferRequest` | Buffer-allocation request payload |
| `TransferBufferRequest` | Bounded buffer-transfer request prefix |
| `LoadProgramRequest` | Program-load request prefix |
| `CreateQueueRequest` | Accelerator execution-queue-create payload |
| `SubmitRequest`, `WireBinding` | Submission prefix and binding array entry |
| `SubmitResponse` | Event ownership payload for accepted or indeterminate submission |
| `WireEventState`, `KnownEventState`, `UnknownEventState` | Event-poll response payload and extensible raw state handling |
| `DecodeError`, `read_exact`, `checked_array_bytes` | Protocol decoding helpers; implementation-only |
| `QueueSize`, `QueueState`, `QueueEpoch`, `QueueControl` | Validated Virtio queue configuration and reset lifecycle |
| `ChainId`, `ChainRegion`, `ChainLayout`, `ChainIo`, `DeviceChain` | Reset-scoped descriptor-chain identity, flattened metadata, and mapped byte-port presentation |
| `DriverQueue`, `PublishedChain`, `UsedChain`, `ReclaimedChain`, `PublishError` | Ownership-preserving driver publication, completion, reset reclamation, and retryable backpressure |
| `DeviceQueue`, `UsedLength`, `NotificationHint`, `NotificationRecheck` | Consuming device completion, exact used length, and lost-wakeup-safe notification decisions |
| `SegmentedSource`, `SegmentedSink` | Device test/reference byte ports over segmented storage |
| `FrameDecoder`, `DecodedRequest`, `DecodedRequestBody`, `FramePreflight` | Complete untrusted request validation before semantic dispatch |
| `ResponseWriter`, `ResponsePayload` | Bounded response framing and exact direct payload destination |
| `ObjectNamespace`, `ObjectKind`, `ObjectId` | Device-private namespaced typed object identity |
| `ObjectTable`, `ObjectTableError` | Device implementation state; encoding is not wire ABI |
| `DeviceState`, typed record types, `ChildCounts` | Context-scoped ownership graph, quotas, and in-flight reference accounting |
| `ReleaseState`, `CreateError`, `RestoreError` | Device-state admission and release rollback helpers |
| `status_from_backend_error`, `status_from_object_table_error`, `status_from_device_state_error` | Device implementation mapping helpers |
| `MockAccelerator` and mock handle types | Reference/test implementation, not normative protocol |
