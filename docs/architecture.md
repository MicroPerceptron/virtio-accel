# Architecture

## Scope

The portable layers define the semantics that must agree across every guest transport, virtual
machine monitor, host operating system, and hardware provider. Platform integrations are adapters;
they do not own accelerator semantics.

Protocol 1.0 does not assign a Virtio device ID. It also does not standardize vendor executable
formats: artifact format identifiers and target words remain opaque to the transport.

The normative terminology, object model, and compatibility rules live in
[specification.md](specification.md), with exact layouts in [wire-abi.md](wire-abi.md) and command
queue rules in [virtqueue.md](virtqueue.md). This document explains the implementation boundaries
that preserve those rules.

## Load-bearing invariants

### Wire safety

Every wire structure is fixed-width, little-endian, pointer-free, and valid at byte alignment.
`zerocopy` derives reject layouts containing implicit padding. Raw numeric opcodes and status values
are validated before conversion to semantic types, preserving forward compatibility without invalid
Rust enum discriminants.

Request and response buffers are untrusted. The command-frame preflight validates descriptor
direction, total byte counts, reserved-zero fields, array multiplication, configured limits, and
command-specific response capacity before a decoded request can reach semantic dispatch.

### Object identity and ownership

Guest-visible handles are opaque `u64` values. The device table combines a slot number with a
device-instance namespace, resource-kind tag, and generation. Removing an object increments its
generation; a slot is permanently retired before generation overflow. Therefore stale, wrong-kind,
cross-device, and pre-reset handles never alias a live object when each reset epoch receives a fresh
namespace.

`DeviceState` composes typed context, buffer, program, execution-queue, and event tables. Context
records retain live-child counts, while event records retain queue, program, and buffer references
until event destruction. Destroying a context with children or a referenced buffer, program, or
queue returns `BUSY`.

The state model has no internal locks or interior mutability. Every transition requires exclusive
access, so a future concurrent command engine has one outer synchronization boundary and no nested
resource-lock order. Creation checks quotas and reserves fallible table capacity before invoking a
provider. Release moves a handle to an explicit `Releasing` state and either commits removal or
restores the same live ID after a rejected provider release.

`CommandProcessor` preserves that ownership model rather than placing an atomic or lock in each
record. One mutable processor owns one backend and one object graph. A transport may move that owner
between workers or serialize admission at its queue boundary; provider-native asynchronous work
continues behind borrowed handles and event objects. Atomics belong in provider completion tokens
or the concrete Virtio status publication path where state is genuinely shared, not in portable
object lookup or reference counting.

Provider releases have an explicit failure boundary too. A rejected release returns the still-live
handle for retry. An indeterminate release invalidates the guest ID and requires device recovery;
the adapter must never guess that the resource is either safe to reuse or safe to free.

### Reset and quarantine

The transport stops fetching chains and publishing completions before handing exclusive ownership
to `CommandProcessor::reset`. The processor then makes one bounded pass: events first, followed by
execution queues, programs, buffers, and contexts. Pending events are cancelled only when the
backend advertises cancellation; no reset path spins, waits, or creates a background executor.

A completely drained graph receives a fresh `ObjectNamespace` and may continue with the same
backend. Any unresolved pending event, rejected reset release, indeterminate release, device loss,
or accounting contradiction produces `BackendDiscardRequired`. That result reports both resources
released during the pass and resources still represented or previously orphaned in quarantine.
The result is sticky: later reset calls make no provider calls, and the complete processor/backend
instance must be discarded rather than reattached to newly initialized queues.

This keeps synchronization at the existing owner boundary. Reset needs no per-record atomics or
locks; provider completion tokens remain responsible for the cancellation/completion race.

### Submission acceptance

A rejected submission guarantees that the backend accepted no execution and retained no resources.
When acceptance cannot be determined, `SubmitFailure::Indeterminate` carries an event. That event is
the ownership token for all referenced resources until it becomes terminal and is destroyed.

This distinction must survive every provider and transport adapter. Collapsing the two cases into a
single error would permit use-after-free during device-reset and timeout races.

### Time

Wire timeouts are relative nanosecond durations measured from device admission. Guest and host
monotonic clocks do not share an epoch, so absolute guest timestamps are never compared with a host
clock. A zero timeout means infinite.

### Memory

The baseline contract uses device-owned buffers plus bounded read/write transfers. External memory,
shared mappings, and fences are optional features because their ownership, cache visibility, and
synchronization rules differ by transport and host OS. Adding them requires a separate invariant
and threat-model pass.

Provider-owned shared memory is distinct from external memory. `MemoryDomain::Shared` requests one
allocation that the provider can access through a host mapping and bind directly for accelerator
execution. It does not expose a guest address or platform handle and does not imply cross-process
sharing or implicit cache coherence.

The allocation result reports verified backing properties, actual retained bytes, and actual
alignment separately from the provider-native handle. Logical buffer bytes may be smaller than a
page-, section-, or device-aligned backing allocation. The command engine retains those facts in its
buffer record for compatibility checks and resource accounting, while submission passes only
borrowed native handles. This lets the device reject a dishonest or degraded allocation before it
becomes guest-visible without adding metadata lookup or boxing to the execution hot path.

Bulk byte payloads cross the semantic boundary through `ByteSource` and `ByteSink`. Both abstractions
support checked random access over segmented storage and an optional contiguous view. A command
engine can therefore expose validated descriptor-backed regions directly: a provider streams them
into final storage, while an already contiguous payload remains one borrowed slice.

## Queue model

Command virtqueue zero is the baseline bidirectional transport queue. One descriptor chain contains
device-readable request bytes and device-writable response bytes. Completion may be out of
submission order, keyed by the request ID.

`virtio-accel-transport` defines the queue boundary without choosing a ring implementation or guest
memory library. Driver publication transfers ownership of a complete chain until used-ring
consumption or reset returns it. Device pop returns a non-`Copy` chain consumed by completion. Queue
identities include a monotonic reset epoch, so stale completion is rejected before guest bytes, a
used element, or a notification can be published.

Publication and completion are release boundaries for request and response bytes; the corresponding
pop operations are acquire boundaries. Notification enablement includes the required atomic recheck,
represented explicitly as `Idle` or `WorkPending`. Concrete adapters may use atomics and atomic
pointers for shared indices and ownership transfer, but the portable traits require no lock, thread,
executor, or global runtime.

Queue configuration may reserve storage bounded by the validated queue size. Every steady-state
operation is nonblocking and heap-allocation-free: publish, pop, complete, notification suppression,
notification recheck, and reset. Reset may move already-owned storage into a reclamation result but
does not allocate or wait for a peer.

`virtio-accel-guest` owns one portable driver queue without internal synchronization. It
preallocates a caller-selected number of tracking slots, writes fixed prefixes directly into
caller-owned chains, and retains bulk read responses in reclaimed chain storage. Prepared transfer
and artifact tails are published without another payload copy. Non-`Copy` typed handles carry the
queue reset epoch; release operations consume them and report whether a failure is retryable,
invalidated, indeterminate, or an opaque unknown status.

`virtio-accel-split-queue` is the deterministic executable implementation of that boundary. It
preallocates descriptor ownership, chain records, available entries, and used entries at
configuration. Its split-ring counters use wrapping `u16` arithmetic, direct chains retain their
scatter/gather buffers, and profile-invalid flags or indirect descriptors are classified before
byte access. Driver and device operations take `&mut SplitQueue`, so ordinary ring state needs no
atomics, atomic pointers, locks, or compare-and-swap loops. Non-atomic `Rc` ownership keeps a reset
reclamation token and a consumed device token tied to the same buffers; one `AtomicU64` reset epoch
is the only synchronization primitive, because it must invalidate byte ports already issued to a
device token before driver ownership is reclaimed.

The baseline `SUBMIT` command returns an event object; `POLL_EVENT` provides portable progress without
requiring unsolicited device writes. Optional multi-queue and event-queue features are reserved for
later validation. Split and packed virtqueue mechanics belong to transport adapters, not the command
engine.

An accelerator execution queue is a separate context-scoped backend object. It never denotes a
Virtio queue index.

## Performance posture

The semantic hot path uses associated handle types and borrowed binding slices, avoiding trait-object
dispatch and per-binding boxing. Wire decoding will operate directly over validated descriptor-backed
regions. Object lookup is constant time and bounded by advertised limits. The queue ports add no
allocation or copy to the steady-state path; mapping implementations can present borrowed segmented
byte ports directly to the command processor.

`Accelerator` deliberately places no `Send` or `Sync` bound on the backend or its associated handle
types. The reference command engine specializes over one concrete backend and owns it behind one
mutable admission boundary. A provider can therefore preserve thread-affine native handles without
boxing, atomics, or locks. Providers that opt into cross-thread auto traits own the synchronization
needed by their actual shared state; the portable object graph does not speculate by adding it to
every handle.

The source-level trait may be erased only after an adapter fixes all associated handle types. Stable
binary plugin loading, cross-module allocation ownership, and an erased handle ABI are deliberately
outside v1. A future plugin adapter can add those policies without changing static providers or
weakening the submit and release contracts.

Backend metadata is fetched and validated once before object tables are constructed. Assigned
reserved capabilities, a missing usable memory domain, and zero advertised limits fail construction.
Unknown capability bits remain available for diagnostics but do not select operations. The command
engine then uses the cached capabilities and limits to reject unsupported work before provider
invocation.

`WRITE_BUFFER` and `READ_BUFFER` are the baseline's explicit content-copy boundaries. Device-local
memory may require bounded staging during those operations. Allocation, submission, polling, and
release do not receive permission to copy a bound buffer merely because a native import or binding
path is inconvenient.

Every buffer declared for program input, output, or mutable state reports
`BufferProperties::DIRECT_BINDING`. This means a compatible submission binds that exact allocation
without copying the bound range to or from a different allocation. A backend that cannot honor the
requested placement and direct-binding contract rejects allocation; a program-specific alignment or
format mismatch rejects submission as `INCOMPATIBLE`. Neither path may silently degrade to a bounce
buffer.

The mutable side of an explicit write receives `&mut Buffer`, allowing implementations to use
ordinary provider handles and mappings rather than forcing interior mutability or a lock into every
buffer. Submission remains borrowed and allocation-free in the semantic API.

Program artifacts use the same byte-source abstraction. Program loading is a slow lifecycle path,
so an object-safe source is an acceptable dispatch cost; forcing a frame-sized allocation and copy
for every segmented artifact is not. Providers can parse a contiguous borrowed artifact in place or
read segmented bytes directly into final resident storage.

Zero-copy guest-memory imports are deliberately deferred rather than pretending that DMA-BUF,
Windows shared handles, and other mechanisms have identical lifetime or coherency semantics. When
external memory is specified, fallback staging will require explicit negotiation and copy
accounting; it will not weaken the provider-owned direct-binding rule.

### Backend fast-path checklist

A provider implementation should make the native buffer handle own or reference everything needed
to reuse the allocation efficiently: the final backing object, device address or import, host
mapping when present, alignment facts, and synchronization state. Native mapping or import setup
belongs at allocation or another amortized lifecycle boundary, not in every submission.

The intended steady-state submission path is a bounded walk over the borrowed binding slice,
validation of program-specific compatibility, native handle/address binding, and queue admission. It
does not allocate per binding, assemble a second binding array with owned payloads, or copy tensor
contents. Small command and metadata writes are not buffer staging and remain provider-specific.

The command engine uses three bounded metadata allocations for `SUBMIT`: decoded bindings, retained
buffer IDs, and borrowed native binding references. Duplicate-slot detection sorts the decoded
allocation in place; event state takes ownership of the retained-ID allocation. No allocation owns
buffer contents, boxes individual bindings, or survives event reclamation except the retained ID
list required for exactly-once reference release.

Issue #29 owns quantitative evidence: explicit-transfer bytes, staged bytes and allocations,
submission allocations, retained memory, and host preparation versus device execution time. A
backend should be diagnosable when it misses the intended path rather than requiring a profiler to
discover an undocumented copy.

## Deterministic reference execution

`virtio-accel-mock::reference` defines a test-only artifact envelope for executable backend tests.
Its fixed 24-byte payload carries an artifact version, a binding-ABI version, an operation, binding
slots, one byte operand, and reserved-zero bytes. The mock additionally requires its provider-owned
format ID, target identity, and exact resident charge before accepting a program. These values and
payload bytes are implementation fixtures, not additions to the normative accelerator ABI;
production command and transport layers continue to pass artifact formats, targets, and payloads
through opaquely.

The reference operations are a lifecycle barrier, equal-length copy, fill, and in-place XOR. Each
operation validates its exact slot and access contract before event admission. Buffers use shared
atomic-byte backing so an accepted event retains only fixed operation metadata, ranges, and atomic
reference-counted backing pointers. Submission does not lock, stage buffer contents, or allocate an
owned binding mirror. Explicit segmented transfers use a fixed-size stack window rather than
coalescing the complete transfer.

Events remain pending until the harness calls `complete`. A single compare-exchange chooses among
execution, cancellation, and injected device loss; after execution starts, cancellation and device
loss report `Busy`. Completion publishes buffer mutations before the terminal event state, while
the harness controls latency and completion order by deciding when and in which order to complete
accepted events.

## Next implementation boundary

The portable command engine depends on `virtio-accel-proto`, `virtio-accel-core`, and the
transport-neutral region metadata re-exported by `virtio-accel-device`. Its baseline processor:

1. Decodes one bounded request from abstract readable/writable byte regions.
2. Maintains typed object records and context dependency counts.
3. Translates wire types into validated semantic values.
4. Passes transfer and artifact regions directly to backend byte ports.
5. Produces a response without knowing about rust-vmm or a host operating system.

Submission/event retention, deterministic reset, the bounded split-virtqueue model, the no-std
reference guest, and deterministic reference execution now complete both portable queue endpoints
and exercise verifiable buffer output. The next backend boundary is issue #23's deterministic fault
injection at every ownership boundary. A thin rust-vmm adapter supplying `virtio-device`,
`virtio-queue`, and `vm-memory` integration remains a later platform layer, as do Linux vhost-user
and an in-kernel guest driver.
