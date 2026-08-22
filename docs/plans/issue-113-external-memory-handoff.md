# Issue #113 design: negotiated external-memory handoff

Status: design complete; protocol values and implementation intentionally unassigned

Issue: [#113 — Design a negotiated external-memory handoff extension for heterogeneous schedules](https://github.com/MicroPerceptron/virtio-accel/issues/113)

Last reviewed: 2026-08-22

## Decision

External-memory handoff is a **compatible protocol extension** under
[`wire-abi.md` section 9](../wire-abi.md#9-candidate-and-post-freeze-change-procedure). A future
minor protocol may assign the reserved external-memory feature position and new opcodes only when
the complete normative, implementation, and conformance change lands. Every Protocol 1.0 frame,
payload length, status, object rule, and fallback remain unchanged.

This design does not assign a feature bit, opcode, status, structure layout, or protocol minor. A
1.0 implementation that advertises external memory is still nonconformant. The design fixes the
ownership and synchronization contract that a later wire proposal must encode.

The first extension uses **completion-gated handoff**, not portable timeline fences. A producer
event must be terminal-successful before export becomes visible to an importer. Import completion
is the acquire point. A consumer's ordinary terminal event plus imported-buffer release is the
release point. Platform adapters perform any required handle transfer, cache maintenance, and DMA
quiescence behind those points. A later timeline-fence extension may optimize the wait without
weakening these transitions.

## Goals and non-goals

The extension must let a scheduler move one byte range from a producing provider to a compatible
consumer without a host byte round trip while preserving:

- exact range, usage, access, and device-compatibility validation;
- exclusive ownership of mutable bytes and stable read-only sharing;
- bounded resource accounting and revocation;
- reset-epoch isolation and stale-object rejection;
- explicit completion and cache-visibility transitions; and
- an always-correct protocol 1.0 `READ_BUFFER` plus `WRITE_BUFFER` fallback.

It does not standardize DMA-BUF, IOSurface, Mach ports, NT handles, D3D resources, vendor handles,
IOMMU APIs, or a platform fence format. It does not reinterpret `MemoryDomain::Shared`, permit
hidden submission staging, promise that arbitrary provider pairs interoperate, or make external
memory mandatory.

## Layering and representation

The portable extension manipulates three opaque object classes:

1. An **export lease** names a source-buffer byte range, access rights, compatibility properties,
   and a completed producer visibility transition.
2. A **transfer slot** is a reset-scoped rendezvous identifier allocated by a transport/platform
   adapter. It carries no host-handle bits. The adapter associates the slot with its private handle,
   authorization record, and optional native synchronization object out of band.
3. An **imported buffer** is a provider buffer view backed by an accepted export lease. Once
   published, it follows the ordinary buffer binding, event retention, and release rules.

Only opaque object or slot IDs and portable descriptors cross command frames. Actual platform
handles cross an adapter-owned channel whose representation is outside the protocol and portable
crates. A VMM may translate a slot into a host mapping; an in-process scheduler may use a broker;
an adapter that cannot transfer a handle for the requested pair rejects the operation.

The portable crates may define traits over opaque owned adapter values. They must not add OS,
vendor-SDK, filesystem, socket, thread, or VMM dependencies, and they must not serialize an adapter
value into a protocol frame.

## Required operations

The later normative proposal needs operations equivalent to the following. Names are descriptive,
not assigned opcodes.

- **Query external-memory compatibility** returns bounded, pair-specific properties: supported
  access modes, minimum offset/length alignment, maximum range, visibility contract, and whether a
  range can be isolated without exposing adjacent bytes. A general feature bit alone never proves
  that two providers or allocations are compatible.
- **Create export lease** takes a source buffer, checked range, rights, and terminal-successful
  producer event. The event must belong to the same context, retain that buffer, and declare write
  access to the exported range; an unrelated successful event is not a completion proof. The
  operation creates the lease only after validation and the release-to-import visibility
  transition succeeds.
- **Authorize transfer slot** takes a lease, one empty adapter slot, an importing security domain
  and device identity, and rights no broader than the lease. It populates one single-use slot. A
  read-only lease needs a distinct authorization and slot for each importer; an exclusive lease may
  have only one outstanding or consumed slot.
- **Import lease** takes a populated slot, requested descriptor and rights, and expected source
  compatibility identity. It atomically consumes the single-use authorization, performs the
  acquire visibility transition, creates the provider import, reconciles actual retained bytes,
  and only then publishes an imported-buffer ID.
- **Revoke export lease** first invalidates every unconsumed slot and prevents new authorizations.
  It succeeds only after every import, event, native mapping, and consumed slot reference is gone;
  otherwise it returns a retryable busy result. Reset uses the quarantine rules below rather than
  waiting.
- **Release imported buffer/export lease** uses the existing rejected-versus-indeterminate release
  distinction. Indeterminate release requires recovery and retains/quarantines all backing that
  may still be reachable.

The implementation must not append fields to a 1.0 request or response. New operations use exact
new payloads gated by successful feature negotiation.

## Ownership state machine

| State | Permitted action | Bytes may be accessed by |
|---|---|---|
| `Local` | producer submission or export preparation | exporting provider only |
| `Exporting` | validate event, rights, range, isolation, and adapter slot | nobody newly; prior producer event is already terminal |
| `LeasedReadOnly` | create read-only imports | importers read; exporter must not mutate |
| `LeasedExclusive` | create exactly one read-write import | exclusive importer only |
| `Revoking` | release imports, slots, mappings, and retained fences | existing holders only; no new import |
| `Local` after successful revoke | producer may reuse the range | exporting provider only |
| `Quarantined` | destroy/discard owning backend and broker state | nobody; backing is not reusable |

An export lease is a child of its source buffer and context. The source buffer cannot be released
or mutated incompatibly while a lease exists. A read-only lease may have multiple imports, but its
source range remains immutable until all imports and events are released. An exclusive lease has at
most one live importer and transfers read/write authority for the range; the exporter cannot bind,
read, write, re-export, or release that range until successful revocation.

Imported buffers are children of their importing context and retain the broker lease, native
import, and any event that references them. Releasing a guest-visible source-buffer ID never proves
that backing is free: the export lease retains it. Object IDs remain device-, kind-, generation-,
context-, and reset-epoch-scoped and never serve as cross-device capabilities.

Overlapping leases are legal only when every overlapping lease is read-only. Any overlap involving
a writer is rejected before adapter or provider invocation. Checked interval arithmetic is
mandatory; zero length and wraparound are invalid.

## Completion and visibility

The first extension has four ordered transition points:

1. The producer event reaches terminal success. Failed, cancelled, pending, unknown, or
   indeterminate producer events cannot authorize export.
2. Export completion means producer writes are complete and the adapter's release/cache transition
   for the exact range has completed. Only then may the transfer slot become populated.
3. Import completion means the consumer mapping and acquire/cache transition have completed. Only
   then may the imported-buffer ID be published or bound.
4. The consumer event reaches a terminal state and the imported buffer is successfully released.
   Only after every consumer is released may revocation return ownership to the exporter.

Successful consumer completion makes bytes written through an exclusive import valid for the next
owner. Failed or cancelled exclusive work may leave those bytes unspecified even after safe DMA
quiescence; revocation must report that content outcome and the exporter must overwrite the range
before reading or re-exporting it. Read-only consumer failure cannot invalidate otherwise stable
source contents. Device loss follows the stricter quarantine rules below.

These are semantic happens-before edges even on a coherent platform. A provider or adapter may
implement them as no-ops only when it can prove hardware coherence for the exact devices and range.
The portable contract never asks a scheduler to guess when to flush or invalidate caches.

This extension does not accept a platform fence payload. If the separately reserved timeline-fence
feature is later negotiated, an adapter may carry a native acquire/release fence in its private
slot and the wire extension may add explicitly gated fence operations. Failure, cancellation,
device loss, and reset must still resolve to the same ownership states above.

## Range, rights, and compatibility validation

Every export/import validates before native work:

- `offset + length` is checked, nonzero, and within the source allocation's logical bytes;
- offset and length meet both providers' and adapter's reported alignment;
- export rights are a subset of the source buffer usage and import rights are a subset of export
  rights;
- program input requires read, program output and mutable state require exclusive write, and
  transfer usage follows the existing source/destination direction rules;
- requested importing alignment, placement, and direct-binding properties are honestly met;
- format, tiling, plane, row-pitch, device-address-width, and heap compatibility are represented by
  a bounded opaque compatibility identity chosen by the adapter, not inferred from byte length;
- suballocation export is rejected unless the adapter proves that page/granule rounding cannot
  expose bytes outside the authorized range; and
- the importing IOMMU or device mapping grants no wider range or access than the lease.

An imported program-visible buffer retains `DIRECT_BINDING`: a compatible submission binds the
imported backing itself. A provider must reject an incompatible import or submission rather than
silently copy through a bounce buffer.

## Failure, device loss, revocation, and reset

Failures before semantic mutation return ordinary errors and leave the transfer slot empty. Once a
slot, native export, import, or mapping may have been created, a response-write failure or unknown
provider result is indeterminate: the affected processor enters recovery, publishes no usable
object ID, and quarantines the broker lease and backing.

Exporter device loss immediately prevents new imports and poisons every lease from that backend.
Existing importing devices must stop new submissions using those imports. Pending consumer events
resolve as failed/device-lost or force their importing backend to be discarded; they never imply
that the exporter may reuse memory.

Importer device loss makes DMA quiescence uncertain. The broker retains the exporter backing until
the platform adapter proves mappings are detached and DMA is stopped, or until the complete
exporting/importing backend set is discarded. An exclusively imported writable range is also
content-invalid after importer loss. Time passing alone is not proof of release.

Whole-device reset remains bounded and nonblocking:

1. stop admission and transport completion;
2. invalidate local IDs and transfer slots for the old epoch;
3. cancel/release events before imported buffers, export leases, ordinary buffers, and contexts;
4. revoke broker authorization for new imports; and
5. quarantine unresolved external state and discard every backend instance that may still access
   it.

Reset does not synchronously wait for another device. An export lease may outlive the exporting
device's guest-visible IDs inside the broker solely to keep live imports safe. It is never reused or
reattached to the reset device. Repeated reset after discard makes no provider calls, matching the
1.0 recovery invariant.

## Authorization, isolation, and accounting

Transfer slots and broker leases are unforgeable capabilities scoped to one security domain,
source allocation generation, exact range, rights, compatible target identity, and reset epochs.
The broker authenticates both endpoints. Guessing an object or slot ID must not reveal whether a
different tenant's allocation exists.

The adapter validates platform-handle provenance and type independently of guest fields. It must
use least-privilege mappings, close duplicate handles on every rejection path, scrub or prevent
observation of padding, and reject a sharing granule that would expose adjacent tenant data.

Host policy supplies nonzero bounds for live leases, slots, imports, pinned bytes, mapped bytes,
and quarantined bytes per device and security domain. Charges begin before native creation and end
only after release is proven. The same backing may avoid double-charging physical residency, but
every endpoint's pinned/mapped resource cost remains charged; accounting must never undercount
because two objects alias.

## Fallback and performance diagnostics

External-memory negotiation never removes the baseline copy path. A scheduler selects one of three
truthful routes:

- `direct-handoff`: one exclusive consumer binds the producer's exported backing with no payload
  copy and returns write ownership explicitly;
- `imported-sharing`: one or more read-only consumers bind the producer's stable exported backing,
  also with no payload copy; or
- `explicit-copy`: the protocol 1.0 read/write path, potentially with provider-owned bounded
  staging inside those explicit transfers.

Import failure must not silently choose another route. The scheduler receives a stable route and
reason category such as feature-unavailable, provider-pair-incompatible, range/alignment,
authorization, resource-pressure, visibility-transition, device-loss, or indeterminate. It may
then choose `explicit-copy` deliberately.

Diagnostics record bytes, range count, setup latency, visibility-transition latency, reuse count,
and copy bytes. They must distinguish a zero-copy handoff from an explicitly selected copy and must
not expose host handles, addresses, tenant identifiers, or timing from another security domain.
Counters are observability only and cannot affect ownership or success semantics.

## Conformance requirements for the implementing minor

The new conformance directory must include independent codec and semantic scenarios for at least:

- negotiation absent, rejected, and accepted while every 1.0 vector remains byte-identical;
- invalid, stale, wrong-kind, cross-context, cross-device, and pre-reset object/slot IDs;
- zero, overflowing, out-of-bounds, misaligned, and isolation-granule-crossing ranges;
- usage escalation, read-only mutation, overlapping writers, and a second exclusive importer;
- pending/failed/cancelled producer events and successful producer-to-consumer visibility;
- import rejection before mutation versus indeterminate native import after mutation;
- consumer completion followed by successful release/revoke and producer reuse;
- source release while leased, import release while event-retained, and retryable busy revocation;
- exporter loss, importer loss, reset during export/import, and reset with pending consumer work;
- stale completion after reset without guest-byte publication;
- quarantine accounting and exactly-once native handle closure on every teardown permutation;
- direct-handoff/imported-sharing diagnostics reporting zero copy bytes; and
- explicit-copy fallback reporting the exact copied byte count without hidden staging.

State-model generation must interleave export, import, submit, poll, cancel, release, revoke,
device loss, response truncation, and reset. Fault injection must cover every adapter mutation
boundary. Platform acceptance needs at least one real producer/consumer pair per adapter; a broker
mock or QEMU-only path proves protocol behavior, not physical DMA isolation or cache visibility.

## Implementation gates and required artifacts

The implementation PR must land as one coordinated future-minor change and update:

- `specification.md`, `wire-abi.md`, `virtqueue.md`, and the threat model;
- Rust feature/opcode/status constants, exact structures, codecs, layout assertions, guest/device
  operations, broker ports, and public API documentation;
- a new minor-version `layout.json`, canonical vectors, scenarios, requirements, and freeze audit;
- clean-room codec and compatibility coverage proving every 1.0 frame is preserved;
- semantic conformance, state-model, fuzz, performance-budget, and fault-injection coverage; and
- transport/platform adapter documentation and physical visibility/isolation evidence.

Before that patch, a platform spike may test whether a proposed pair can export, import, isolate,
flush, invalidate, and quiesce safely. A spike must remain unadvertised and cannot add protocol
constants or reinterpret `Shared`.

## Deferred choices

The implementing proposal still has to assign the protocol minor, exact opcodes and layouts,
portable reason codes, limits, and adapter trait shapes. Those choices depend on at least two real
platform spikes and an independent codec review. Timeline fences, cross-security-domain sharing,
mutable multi-consumer aliases, and automatic copy fallback remain separate extensions; none is a
prerequisite for the completion-gated contract above.
