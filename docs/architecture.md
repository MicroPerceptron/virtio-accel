# Architecture

## Scope

The portable layers define the semantics that must agree across every guest transport, virtual
machine monitor, host operating system, and hardware provider. Platform integrations are adapters;
they do not own accelerator semantics.

The draft does not yet assign a virtio device ID. It also does not standardize vendor executable
formats: artifact format identifiers and target words remain opaque to the transport.

The normative terminology, object model, and compatibility rules live in
[specification.md](specification.md). This document explains the implementation boundaries that
preserve those rules.

## Load-bearing invariants

### Wire safety

Every wire structure is fixed-width, little-endian, pointer-free, and valid at byte alignment.
`zerocopy` derives reject layouts containing implicit padding. Raw numeric opcodes and status values
are validated before conversion to semantic types, preserving forward compatibility without invalid
Rust enum discriminants.

Request and response buffers are untrusted. A future command engine must validate descriptor
direction, total byte counts, reserved-zero fields, array multiplication, configured limits, and
object ownership before invoking a backend.

### Object identity and ownership

Guest-visible handles are opaque `u64` values. The current device table combines a slot number with
a generation that contains a resource-kind tag. Removing an object increments its generation; a
slot is permanently retired before generation overflow. Therefore stale or wrong-kind handles never
alias a live object during the lifetime of a device instance.

The command engine must additionally track the parent context and live-child counts for buffers,
programs, queues, and events. Destroying a context with children, a queue with in-flight work, or an
event before terminal completion returns `BUSY`.

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

The initial buffer transfer path permits copies because it is the compatibility baseline. Zero-copy
imports are deliberately deferred rather than pretending that DMA-BUF, Windows shared handles, and
other mechanisms have identical lifetime or coherency semantics.

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
