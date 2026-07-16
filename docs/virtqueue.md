# virtio-accel protocol 1.0 command-virtqueue contract

This document defines how protocol 1.0 request and response frames inhabit Virtio descriptor
chains. It is normative together with [specification.md](specification.md) and
[wire-abi.md](wire-abi.md).

The portable reference path targets a split virtqueue. It relies only on ordered readable and
writable byte regions plus queue publication/completion operations, so future transport adapters do
not need to expose guest addresses or transport-specific descriptor types to the command engine.

## 1. Relationship to the base Virtio specification

The [Virtio 1.3 specification](https://docs.oasis-open.org/virtio/virtio/v1.3/virtio-v1.3.html)
continues to supply the base rules, including descriptor-loop prohibition, negotiated indirect
descriptors, queue publication barriers, notification suppression, and transport reset. This
document adds device-specific framing requirements.

Protocol 1.0 defines command virtqueue index zero. It defines no device-specific multi-queue feature.
The reference device **MUST NOT** offer `VIRTIO_F_RING_PACKED`, `VIRTIO_F_IN_ORDER`,
`VIRTIO_F_RING_RESET`, `VIRTIO_F_INDIRECT_DESC`, or `VIRTIO_F_EVENT_IDX`. It uses direct split-ring
descriptors and the basic available/used notification-suppression flags. A future adapter may
support one of these base Virtio features only when it preserves the device-specific rules here and
adds corresponding conformance evidence. Whole-device reset remains mandatory.

## 2. Flattened chain model

For validation, a transport adapter presents one available descriptor head as:

- an ordered list of device-readable byte regions;
- followed by an ordered list of device-writable byte regions; and
- the original queue head token used for completion.

A future adapter may flatten a valid indirect descriptor table when `VIRTIO_F_INDIRECT_DESC` was
negotiated. The configured `max_chain_descriptors` then applies after flattening and includes both
readable and writable descriptors. The protocol 1.0 reference profile rejects indirect
descriptors because it does not negotiate that feature.

The command engine sees region lengths and byte access only. It **MUST NOT** receive guest physical
addresses, descriptor indices, indirect-table addresses, ring pointers, or notification objects.

## 3. Descriptor-chain validity

A protocol 1.0 command chain is valid only when all of the following hold:

1. The chain is acyclic and structurally valid under the base Virtio specification.
2. Its flattened descriptor count is 2 through configured `max_chain_descriptors`.
3. Every descriptor length is nonzero.
4. At least one device-readable descriptor precedes at least one device-writable descriptor.
5. No device-readable descriptor follows a device-writable descriptor.
6. The readable total is representable, no greater than configured `max_request_bytes`, and exactly
   one complete request frame.
7. The writable total is representable and large enough for the command’s maximum possible
   response shape.

Cross-region reads and writes concatenate regions without inserting padding. Header or payload
fields may cross descriptor boundaries at any byte.

The transport adapter **MUST** validate the descriptor topology and map every readable and required
writable byte before the command engine performs semantic validation or invokes a backend.

## 4. One command per chain

Exactly one request frame occupies the complete readable portion of a chain. The first readable byte
is request-header byte zero. The readable total **MUST** equal:

```text
16 + request_header.payload_bytes
```

No bytes before the header, between fixed and variable payload portions, or after the declared
payload are permitted.

The response begins at writable byte zero. Writable capacity beyond the actual response remains
untouched and is not included in used length.

## 5. Validation order and failure atomicity

The device validates a chain in this order:

1. descriptor topology, direction, count, nonzero lengths, addressability, and total-length
   arithmetic;
2. availability of a complete 16-byte request header and 16-byte writable response header;
3. configured request limit and exact `payload_bytes` equality;
4. opcode and request-header flags;
5. command-specific fixed length, variable counts, reserved fields, scalar namespaces, object
   relationships, and required writable capacity;
6. quotas and semantic preconditions; and
7. backend invocation.

No failure in steps 1 through 5 may mutate device semantic state or invoke the backend.

If descriptor topology is invalid, the request header is truncated, or writable capacity is less
than 16 bytes, the device returns the descriptor head used with used length zero and writes no
bytes.

If the request header is valid enough to recover `request_id`, and writable capacity is at least 16,
later validation failures produce a complete error response header. The used length is 16.

Required success capacity is checked before semantic mutation. A short success buffer therefore
produces used length zero, no response bytes, and no semantic state change; it is not converted to a
smaller protocol error because the driver failed the response-buffer contract.

## 6. Response writing and used length

The device **MUST** commit a response atomically from the protocol’s point of view:

- every byte counted as used is initialized;
- the response header’s `payload_bytes` equals the bytes written after it;
- no byte beyond `16 + payload_bytes` is written; and
- the split-ring used element length equals `16 + payload_bytes`.

Used length counts device-written bytes only. It never includes readable request bytes.

Response bytes may span writable descriptors. A device **MUST** write them as if to one concatenated
byte sequence.

An unexpected mapping or write failure after the preflight succeeds indicates broken transport or
memory state. If semantic mutation has not occurred, the chain may complete with used length zero.
If mutation or uncertain backend acceptance has occurred, the device **MUST** set
`DEVICE_NEEDS_RESET`, quarantine retained resources, and avoid a response that falsely claims
rejection.

## 7. Command and execution ordering

The baseline command queue permits out-of-order command completion. The device **MUST** consume
available heads in available-ring order, but may dispatch their semantic work concurrently and
return them used in a different order as work completes. Drivers therefore track both descriptor
heads and nonzero request IDs.

Request-ID uniqueness lasts until the corresponding chain is returned used. A driver **MUST NOT**
infer ordering from numeric request IDs.

Command completion and accelerator execution completion are distinct:

- `SUBMIT` command completion reports admission and returns an event ID;
- the accelerator operation may remain pending after the command chain is used; and
- `POLL_EVENT` observes execution completion independently of command ordering.

Implementations may serialize command execution, but they **MUST NOT** promise in-order completion
as protocol behavior.

## 8. Backpressure and queue fullness

Queue capacity is transport state, not a protocol error response.

- A driver with no free descriptor head or insufficient writable buffers **MUST** retain the command
  locally and report backpressure to its caller.
- It **MUST NOT** publish a partial command or reuse descriptors still owned by the device.
- A device may defer consumption or completion without busy-looping.
- Resource limits discovered after a valid chain is consumed produce `RESOURCE_LIMIT`; descriptor
  scarcity before publication does not.

The reference guest API should represent pre-publication backpressure separately from protocol
statuses so callers can retry without constructing a new semantic command.

## 9. Notifications

Available and used notifications follow the base Virtio split-ring suppression rules. Suppressing a
notification changes neither command visibility nor completion semantics.

- The driver publishes descriptors and the available-ring entry before deciding whether to notify.
- The device publishes the used element and used index before deciding whether to notify.
- Both sides must recheck queue state when required by the base specification to avoid lost wakeups.

The device-specific protocol defines no polling interval, thread, executor, or interrupt affinity.

## 10. Malformed chains

The following are isolated command-chain failures and do not by themselves require device reset:

- descriptor loops or an invalid indirect table caught before byte access;
- too many, zero-length, missing-readable, missing-writable, or interleaved-direction descriptors;
- truncated headers or payloads;
- oversized frames;
- exact-length or reserved-zero violations;
- unknown opcodes, flags, or scalar values; and
- insufficient writable capacity detected before semantic mutation.

The device completes a structurally recoverable malformed chain according to section 5 and
continues with later available chains.

Repeated malformed input may be rate-limited by the transport or device integration, but rate
limiting **MUST NOT** change object ownership or fabricate successful responses.

## 11. Device reset and split-ring disposition

Protocol 1.0 uses whole-device reset. The reference device does not negotiate independent
`VIRTIO_F_RING_RESET`.

When reset begins, the device:

1. stops fetching new available chains;
2. stops publishing used entries and notifications;
3. prevents new backend admission;
4. resolves, cancels, or quarantines already-started operations according to known ownership;
5. invalidates all object IDs and request tracking from the old reset epoch; and
6. resets command-queue available/used state as required by the base transport.

After steps 1 and 2 quiesce queue access, the transport gives exclusive ownership to the portable
command processor for one bounded teardown pass. A reusable result permits queue and object-table
reinitialization with a fresh namespace. A discard-required result forbids further backend calls;
the transport discards that processor/backend instance and creates a new one before exposing new
queues.

Descriptor chains that were available or in progress when reset began are not completed after reset.
Once the driver has observed reset completion, it may reclaim all queue memory and descriptors under
the base Virtio reset rule. It **MUST NOT** expect response bytes or used entries for those chains.

After reinitialization, request IDs may be reused, all object IDs from the prior epoch are stale, and
the queue starts from its base initial state. No late backend completion from the old epoch may write
guest memory or a new used ring.

## 12. `DEVICE_NEEDS_RESET`

Ordinary hostile input does not set `DEVICE_NEEDS_RESET`. The device sets it when continued protocol
operation cannot preserve ownership or response truth, including:

- indeterminate provider release that invalidates a guest object but leaves backend ownership
  unknown;
- inability to report or retain an indeterminate accepted submission;
- response-write failure after semantic mutation;
- internal state corruption or accounting contradiction; or
- backend device loss that prevents bounded recovery.

After observing `DEVICE_NEEDS_RESET`, the driver **SHOULD** stop submitting commands and perform
whole-device reset. The device may finish responses whose ownership and output remain provably safe,
but it **MUST NOT** accept new semantic work.

## 13. Protocol conformance cases

The portable conformance suite uses these stable case identifiers:

| ID | Required assertion |
|---|---|
| `VQ-001` | One readable and one writable descriptor carries a valid command |
| `VQ-002` | Header and every fixed payload decode across every possible byte split |
| `VQ-003` | Multiple readable and writable segments concatenate without padding |
| `VQ-004` | Missing readable or writable region completes with used length zero |
| `VQ-005` | Readable descriptor after a writable descriptor is rejected before backend invocation |
| `VQ-006` | Zero-length, looping, excessive, or invalid-indirect chains are rejected |
| `VQ-007` | Truncated request header writes nothing and reports used length zero |
| `VQ-008` | Valid header plus malformed payload writes exactly a 16-byte error response |
| `VQ-009` | Unknown opcode returns `UNSUPPORTED` without semantic mutation |
| `VQ-010` | Nonzero request, command, or reserved flags are rejected |
| `VQ-011` | Trailing readable bytes are rejected |
| `VQ-012` | Short command-specific response capacity is detected before mutation and writes nothing |
| `VQ-013` | Success and error used lengths equal exact initialized response bytes |
| `VQ-014` | Distinct requests can complete out of publication order and retain correct request IDs |
| `VQ-015` | `SUBMIT` command completion is independent of event completion |
| `VQ-016` | Queue-full pre-publication behavior is retryable backpressure, not a protocol status |
| `VQ-017` | Notification suppression does not lose available or used work |
| `VQ-018` | Reset produces no late used entries or guest writes from the old epoch |
| `VQ-019` | Reset invalidates every old object ID and permits request-ID reuse |
| `VQ-020` | Post-mutation output failure or indeterminate release sets `DEVICE_NEEDS_RESET` |

The byte vectors under [`conformance/v1.0`](../conformance/v1.0/) cover protocol frames. Queue-model
implementations consume the case identifiers above so the same behavioral scenarios can be reused
by the in-memory split ring and future platform transports. The dependency-free
`virtio-accel-transport` crate provides the executable region, ownership, reset-epoch, backpressure,
and notification port contracts. `virtio-accel-split-queue` exercises the ring-level portions of
`VQ-001`, `VQ-003` through `VQ-006`, `VQ-013`, `VQ-014`, and `VQ-016` through `VQ-018`. Guest
compatibility behavior remains tracked by #19 and full-path semantic assertions by #20.

## Appendix A: portable Rust queue-port mapping

This appendix is non-normative. `DriverQueue::publish` consumes a complete driver chain on success
and returns it unchanged through `PublishError` on pre-publication failure. `pop_used` or `reset`
returns every successfully published chain, preventing safe callers from reusing descriptors while
the device owns them.

`DeviceQueue::pop_available` returns a non-`Copy` `DeviceChain`. Completion consumes that value and
must compare its `ChainId` epoch with current `QueueState` before publishing bytes or ring state.
`ChainIo` contains only flattened `ChainRegion` values and generic readable/writable byte ports; it
cannot carry guest addresses or a concrete descriptor type.

All steady-state queue-port methods are specified as nonblocking and allocation-free in their Rust
documentation. Publication/completion establish release ordering, their peer-side pops establish
acquire ordering, and notification enablement returns `WorkPending` when its mandatory recheck finds
work that raced with suppression.

## Appendix B: in-memory split-ring model

This appendix is non-normative. `SplitQueue` preallocates one descriptor-ownership table, chain
record table, available ring, and used ring from a validated power-of-two `QueueSize`. Publication,
available consumption, out-of-order completion, used consumption, and notification rechecks move
owned values or update fixed slots; none allocates or coalesces payload bytes.

`DriverChain::direct` constructs the baseline direct profile. `DriverChain::raw` retains malformed
topology for deterministic device tests, while `SplitQueue::inject_available` bypasses only normal
driver-side profile validation. Traversal remains bounded by the supplied descriptor table and
never allocates from a descriptor byte length, `next` value, or other guest-controlled scalar.
Unknown flags and `VIRTQ_DESC_F_INDIRECT` are rejected because protocol 1.0 negotiates neither.

The four public `RingCounters` use the split ring's wrapping `u16` index arithmetic. Test hooks can
place an empty queue at a chosen index and choose the next descriptor allocation slot, making ring
wrap, descriptor wrap, exhaustion, and notification races reproducible without timing or threads.
Reset publishes a new atomic epoch before removing ring storage; every old `ChainIo` byte access and
completion checks that epoch before touching a payload or used entry.
