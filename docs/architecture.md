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
and cross-device handles never alias a live object during the lifetime of a device instance.

`DeviceState` composes typed context, buffer, program, execution-queue, and event tables. Context
records retain live-child counts, while event records retain queue, program, and buffer references
until event destruction. Destroying a context with children or a referenced buffer, program, or
queue returns `BUSY`.

The state model has no internal locks or interior mutability. Every transition requires exclusive
access, so a future concurrent command engine has one outer synchronization boundary and no nested
resource-lock order. Creation checks quotas and reserves fallible table capacity before invoking a
provider. Release moves a handle to an explicit `Releasing` state and either commits removal or
restores the same live ID after a rejected provider release.

Provider releases have an explicit failure boundary too. A rejected release returns the still-live
handle for retry. An indeterminate release invalidates the guest ID and requires device recovery;
the adapter must never guess that the resource is either safe to reuse or safe to free.

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

The baseline `SUBMIT` command returns an event object; `POLL_EVENT` provides portable progress without
requiring unsolicited device writes. Optional multi-queue and event-queue features are reserved for
later validation. Split and packed virtqueue mechanics belong to transport adapters, not the command
engine.

An accelerator execution queue is a separate context-scoped backend object. It never denotes a
Virtio queue index.

## Performance posture

The semantic hot path uses associated handle types and borrowed binding slices, avoiding trait-object
dispatch and per-binding boxing. Wire decoding will operate directly over validated descriptor-backed
regions. Object lookup is constant time and bounded by advertised limits.

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

Issue #29 owns quantitative evidence: explicit-transfer bytes, staged bytes and allocations,
submission allocations, retained memory, and host preparation versus device execution time. A
backend should be diagnosable when it misses the intended path rather than requiring a profiler to
discover an undocumented copy.

## Next implementation boundary

The next portable milestone is a command engine that depends only on `virtio-accel-proto` and
`virtio-accel-core`. It will:

1. Decode one bounded request from abstract readable/writable byte regions.
2. Maintain typed object records and context dependency counts.
3. Translate wire types into validated semantic values.
4. Retain buffer, program, and queue ownership through event completion.
5. Produce a response without knowing about rust-vmm or a host operating system.

After that engine passes adversarial parser and lifecycle tests, a thin rust-vmm adapter can supply
`virtio-device`, `virtio-queue`, and `vm-memory` integration. Linux vhost-user and an in-kernel guest
driver remain later platform layers.
