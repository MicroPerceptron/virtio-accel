# virtio-accel protocol 1.0 wire ABI

This document defines the protocol 1.0 candidate byte contract used by the command virtqueue. It is
normative together with [specification.md](specification.md) and [virtqueue.md](virtqueue.md).
Structure names refer to the Rust implementation for convenience; implementations in other
languages depend only on the byte layouts and rules below.

All multibyte integers are unsigned little-endian values unless a field explicitly says otherwise.
Every structure has byte alignment one and contains no implicit padding. Offsets and sizes are
listed in the checked-in [layout manifest](../conformance/v1.0/layout.json).

## 1. Global limits

| Constant | Value | Requirement |
|---|---:|---|
| Protocol version | 1.0 | Configuration **MUST** report major 1 and minor 0 |
| Baseline command queues | 1 | Only command virtqueue index 0 exists |
| Hard maximum chain descriptors | 256 | The advertised maximum **MUST** be 2 through 256 |
| Hard maximum request frame | 16 MiB | Includes the 16-byte request header |
| Hard maximum response frame | 16 MiB | Includes the 16-byte response header |
| Hard maximum submission bindings | 4096 | The semantic advertised limit may be smaller |

The device-specific configuration limits are additional negotiated bounds. A device **MUST NOT**
advertise a limit above a hard maximum. A driver **MUST** reject invalid configuration rather than
allocating from it.

Every count-to-byte conversion **MUST** use checked multiplication and addition. A value that
overflows the implementation address space or the fixed-width containing field is invalid.

## 2. Device-specific configuration

`WireConfig` is 16 bytes:

| Offset | Bytes | Field | Protocol 1.0 rule |
|---:|---:|---|---|
| 0 | 2 | `protocol_major` | **MUST** be 1 |
| 2 | 2 | `protocol_minor` | A conforming 1.0 device reports 0; a 1.0 driver **MUST** accept a higher minor and use only 1.0 behavior |
| 4 | 2 | `command_queue_count` | **MUST** be 1 |
| 6 | 2 | `max_chain_descriptors` | **MUST** be 2 through 256 and no greater than the configured queue size |
| 8 | 4 | `max_request_bytes` | **MUST** be 97 through 16 MiB |
| 12 | 4 | `max_response_bytes` | **MUST** be 92 through 16 MiB |

The minimum request limit admits a one-byte program artifact. The minimum response limit admits the
complete device-information response.

The 16 bytes above are the protocol 1.0 configuration prefix. A future minor version may append
configuration only when a 1.0 driver can safely ignore it and the extension does not alter baseline
behavior without feature negotiation. A 1.0 driver reads and validates the prefix and does not
require the entire transport-specific configuration region to be exactly 16 bytes.

The baseline device-specific feature set is empty. Feature-bit positions 0 through 4 are reserved
for multi-queue, event-queue, external-memory, timeline-fence, and secure-context proposals. A
protocol 1.0 device **MUST NOT** advertise them and a protocol 1.0 driver **MUST NOT** accept them.

## 3. Request and response frames

### 3.1 Request header

Every request begins with the 16-byte `RequestHeader`:

| Offset | Bytes | Field | Rule |
|---:|---:|---|---|
| 0 | 2 | `opcode` | Raw opcode from section 5 |
| 2 | 2 | `flags` | **MUST** be zero |
| 4 | 4 | `payload_bytes` | Exact number of readable bytes after this header |
| 8 | 8 | `request_id` | Nonzero and unique among requests outstanding on this device instance |

Bit zero of `flags`, formerly drafted as `NO_WAIT`, is reserved and has no 1.0 semantics.

The readable byte count **MUST** equal `16 + payload_bytes` exactly. There is no tolerated trailing
extension area. An unknown opcode remains a raw integer long enough to produce `UNSUPPORTED`; it
**MUST NOT** be materialized as an invalid language enum.

A request ID may be reused only after the corresponding descriptor chain has been returned used or
after the driver has completed a device reset. Request IDs correlate command completion; they do not
identify accelerator events.

### 3.2 Response header

Every written response begins with the 16-byte `ResponseHeader`:

| Offset | Bytes | Field | Rule |
|---:|---:|---|---|
| 0 | 2 | `status` | Raw status from section 6 |
| 2 | 2 | `flags` | **MUST** be zero |
| 4 | 4 | `payload_bytes` | Exact number of bytes written after this header |
| 8 | 8 | `request_id` | Exact request ID from the corresponding valid request header |

The driver **MUST** validate that the response request ID matches the request associated with the
used descriptor head. An unknown status is an opaque failure, never success.

Except for an indeterminate `SUBMIT`, every non-`OK` response has an empty payload. An indeterminate
`SUBMIT` has the original mapped failure status and an eight-byte `SubmitResponse`; possession of
that event ID prevents premature resource release.

## 4. Common value namespaces

### 4.1 Object IDs

An object ID is an opaque nonzero `u64`. Zero is invalid. Its encoding is device-private and no bit
has driver-visible meaning.

### 4.2 Memory domains

| Value | Meaning |
|---:|---|
| 1 | Host-preferred memory |
| 2 | Device-preferred memory |
| 3 | Shared/coherent memory class |

All other values produce `INVALID_ARGUMENT`. These values express placement intent only and do not
create a host mapping or external-memory handle.

### 4.3 Buffer usage bits

| Bit | Value | Meaning |
|---:|---:|---|
| 0 | `0x00000001` | Transfer source |
| 1 | `0x00000002` | Transfer destination |
| 2 | `0x00000004` | Program input |
| 3 | `0x00000008` | Program output |
| 4 | `0x00000010` | Mutable program state |

At least one usage bit **MUST** be set. Unknown bits produce `UNSUPPORTED`.

### 4.4 Binding access

| Value | Meaning |
|---:|---|
| 1 | Read |
| 2 | Write |
| 3 | Read and write |

All other values produce `INVALID_ARGUMENT`.

### 4.5 Event states

| Value | State | `error` field |
|---:|---|---|
| 0 | Pending | `OK` |
| 1 | Complete | `OK` |
| 2 | Failed | Non-`OK` status explaining execution failure |
| 3 | Cancelled | `OK` |

Unknown event states make the response invalid to a 1.0 driver. A driver **MUST** retain the event
and request recovery rather than guessing that the event is terminal.

## 5. Opcodes and payloads

The request payload length is exact. A fixed prefix followed by variable bytes has no alignment
padding between the prefix and tail.

| Opcode | Value | Request payload | `OK` response payload | Maximum required writable capacity |
|---|---:|---|---|---:|
| `GET_DEVICE_INFO` | `0x0001` | Empty | `WireDeviceInfo` | 92 |
| `CREATE_CONTEXT` | `0x0100` | `CreateContextRequest` | `ObjectPayload` context ID | 24 |
| `DESTROY_CONTEXT` | `0x0101` | `ObjectPayload` context ID | Empty | 16 |
| `ALLOCATE_BUFFER` | `0x0200` | `AllocateBufferRequest` | `ObjectPayload` buffer ID | 24 |
| `FREE_BUFFER` | `0x0201` | `ObjectPayload` buffer ID | Empty | 16 |
| `WRITE_BUFFER` | `0x0202` | `TransferBufferRequest` + data | Empty | 16 |
| `READ_BUFFER` | `0x0203` | `TransferBufferRequest` | Exactly `bytes` data | `16 + bytes` |
| `LOAD_PROGRAM` | `0x0300` | `LoadProgramRequest` + artifact | `ObjectPayload` program ID | 24 |
| `UNLOAD_PROGRAM` | `0x0301` | `ObjectPayload` program ID | Empty | 16 |
| `CREATE_QUEUE` | `0x0400` | `CreateQueueRequest` | `ObjectPayload` execution-queue ID | 24 |
| `DESTROY_QUEUE` | `0x0401` | `ObjectPayload` execution-queue ID | Empty | 16 |
| `SUBMIT` | `0x0500` | `SubmitRequest` + `WireBinding[]` | `SubmitResponse` event ID | 24 |
| `POLL_EVENT` | `0x0501` | `ObjectPayload` event ID | `WireEventState` | 24 |
| `CANCEL_EVENT` | `0x0502` | `ObjectPayload` event ID | Empty | 16 |
| `DESTROY_EVENT` | `0x0503` | `ObjectPayload` event ID | Empty | 16 |

The maximum required writable capacity is validated before any semantic mutation or backend call.
Protocol 1.0 has no object-list payload: every destruction or event operation names exactly one
object ID. The binding array is the only variable-count structured array in a baseline request.

### 5.1 `WireDeviceInfo`

`WireDeviceInfo` is 76 bytes:

| Field | Rule |
|---|---|
| `uuid[16]` | Stable identity for this accelerator device |
| `class` | Extensible raw class; 0 other, 1 NPU, 2 GPU, 3 DSP |
| `reserved` | Zero |
| `vendor_id`, `device_id` | Provider identity; zero means unspecified |
| `capabilities` | Assigned semantic capability bits only |
| `max_contexts` | Nonzero device-wide live-context limit |
| `max_buffers_per_context` | Nonzero live-buffer limit |
| `max_programs_per_context` | Nonzero live-program limit |
| `max_queues_per_context` | Nonzero live execution-queue limit |
| `max_events_per_context` | Nonzero live-event/in-flight-submission limit |
| `max_bindings_per_submission` | 1 through 4096 |
| `max_buffer_bytes` | Nonzero maximum allocation size |
| `max_artifact_bytes` | Nonzero maximum artifact tail size, additionally bounded by the request-frame limit |

Capability bits are semantic reports, not Virtio feature bits. A capability **MUST NOT** alter wire
framing without a separately negotiated feature. Unknown capability bits are ignored for operation
selection and preserved by diagnostic tooling.

Assigned protocol 1.0 semantic capability bits are:

| Bit | Name |
|---:|---|
| 0 | `HOST_VISIBLE_MEMORY` |
| 1 | `DEVICE_LOCAL_MEMORY` |
| 2 | `EVENT_CANCELLATION` |
| 5 | `SHARED_MEMORY` |

Bits 3 (`EXTERNAL_MEMORY`) and 4 (`SECURE_CONTEXTS`) are reserved and **MUST NOT** be advertised by a
protocol 1.0 device.

### 5.2 Context

`CreateContextRequest` is eight bytes: `flags: u32` followed by `reserved: u32`. Both fields **MUST**
be zero in protocol 1.0.

Context destruction uses an eight-byte `ObjectPayload`.

### 5.3 Buffers

`AllocateBufferRequest` is 40 bytes:

| Field | Rule |
|---|---|
| `context_id` | Live context |
| `bytes` | Nonzero and no greater than `max_buffer_bytes` |
| `alignment` | Nonzero power of two |
| `memory_domain` | Assigned value from section 4.2 |
| `reserved0[7]` | All zero |
| `usage` | Nonempty subset of assigned usage bits |
| `reserved1` | Zero |

The device **MUST** reject a memory domain whose corresponding semantic capability is absent before
backend invocation. `Host`, `Device`, and provider-owned `Shared` allocations use capability bits 0,
1, and 5 respectively. Successful allocation commits the backend to the placement and direct-binding
rules in [specification.md](specification.md); it may not silently substitute a staged submission
path.

`TransferBufferRequest` is 24 bytes containing `buffer_id`, `offset`, and `bytes`. `bytes` **MUST**
be nonzero. `offset + bytes` **MUST NOT** overflow and **MUST** fit in the buffer.

For `WRITE_BUFFER`, the request payload length **MUST** be `24 + bytes`, and bytes following the
prefix are copied to the buffer. For `READ_BUFFER`, the request payload is exactly 24 bytes and the
success response payload contains exactly `bytes` bytes. Transfers must also fit the configured
request or response frame maximum.

`WRITE_BUFFER` requires buffer usage `TRANSFER_DESTINATION`; `READ_BUFFER` requires
`TRANSFER_SOURCE`. These commands are explicit copy boundaries. Their existence does not permit
allocation or submission to copy program bindings through hidden bounce buffers.

### 5.4 Programs

`LoadProgramRequest` is an 80-byte prefix:

| Field | Rule |
|---|---|
| `context_id` | Live context |
| `format` | Nonzero provider-owned format ID |
| `flags` | Zero |
| `target[12]` | Opaque format-owned target words |
| `payload_bytes` | Nonzero, equals the exact artifact tail length |
| `resident_bytes` | Nonzero declared resident-memory charge |

`80 + payload_bytes` **MUST** fit the request payload and configured frame limit.

### 5.5 Execution queues

`CreateQueueRequest` is 16 bytes: `context_id: u64`, `flags: u32`, and `reserved: u32`. Both flag and
reserved words **MUST** be zero in protocol 1.0.

### 5.6 Submission and events

`SubmitRequest` is a 32-byte prefix containing `queue_id`, `program_id`, `binding_count`, `flags`,
and `timeout_ns`.

- `binding_count` **MUST** be 1 through both advertised `max_bindings_per_submission` and 4096.
- `flags` **MUST** be zero.
- `timeout_ns` is relative to backend admission; zero means infinite.
- The payload length **MUST** be `32 + binding_count * 32`.

Each 32-byte `WireBinding` contains `buffer_id`, `offset`, `bytes`, `slot`, `access`, and three
reserved-zero bytes. Buffer ranges are nonempty and checked for overflow. Slots are unique within
the submission. Every object belongs to the same context.

`SubmitResponse` is an eight-byte event ID. It is returned with `OK` after accepted admission and
with the mapped non-`OK` status when admission is indeterminate. A rejected submission has an empty
error payload.

`WireEventState` is eight bytes: `state: u16`, `error: u16`, and `reserved: u32`. Reserved bytes are
zero and the state/error combinations are exactly those in section 4.5.

## 6. Status namespace

| Value | Name | Meaning |
|---:|---|---|
| 0 | `OK` | Command completed successfully |
| 1 | `UNSUPPORTED` | Opcode, flag, feature, capability, or operation is not supported |
| 2 | `INCOMPATIBLE` | Known artifact, target, object, or capability combination is incompatible |
| 3 | `INVALID_ARGUMENT` | Malformed value, reserved field, length mismatch, zero required value, or duplicate slot |
| 4 | `OUT_OF_BOUNDS` | Checked byte range does not fit its object |
| 5 | `BUSY` | Object is live, referenced, pending, or otherwise retryable |
| 6 | `OUT_OF_MEMORY` | Host/provider allocation failed |
| 7 | `RESOURCE_LIMIT` | Configured count or byte limit would be exceeded |
| 8 | `DEADLINE_EXPIRED` | Operation expired according to the relative timeout contract |
| 9 | `DEVICE_LOST` | Backend/device state cannot continue normally |
| 10 | `PERMISSION_DENIED` | Isolation or provider policy denied the operation |
| 11 | `STALE_OBJECT` | Nonzero object ID is stale, wrong-kind, reset-invalidated, or not valid in this context |
| 65535 | `INTERNAL_ERROR` | Unclassified implementation/provider failure |

Unknown status values are opaque non-success failures. Provider-specific error domains do not cross
the 1.0 wire boundary; absent a future diagnostic feature, they map to `INTERNAL_ERROR`.

Malformed input is classified before backend invocation:

- unknown opcode or nonzero/unknown flags: `UNSUPPORTED`;
- fixed-length mismatch, trailing bytes, reserved nonzero, invalid scalar, zero required value, or
  arithmetic overflow: `INVALID_ARGUMENT`;
- frame, binding, object, or configured quota exceeded: `RESOURCE_LIMIT`;
- valid object ID with wrong generation, kind, reset epoch, or context: `STALE_OBJECT`; and
- valid byte range outside the selected object: `OUT_OF_BOUNDS`.

## 7. Response atomicity

Before backend invocation, the device **MUST** validate the complete readable frame and enough
writable capacity for every possible response shape of that command. It **MUST** initialize every
response byte it reports used.

Ordinary protocol errors have no semantic state change. If an unexpected transport write failure
occurs after semantic mutation, or a release becomes indeterminate, the device **MUST** enter
recovery and expose the Virtio `DEVICE_NEEDS_RESET` condition. It **MUST NOT** report an ordinary
rejected response that would let the driver free resources whose ownership is uncertain.

## 8. Versioned compatibility artifacts

Protocol constants, layouts, and canonical bytes are checked in under
[`conformance/v1.0`](../conformance/v1.0/). They are review inputs, not test-generated output.
Before the final freeze audit, changing any assigned value, field, size, or vector requires one
coordinated candidate revision under section 9. After the freeze, such a change requires a
protocol-major change unless it uses a previously reserved value under the compatibility rules in
[specification.md](specification.md).

## 9. Candidate and post-freeze change procedure

A proposed wire change **MUST** be classified before code is merged:

1. Before the final freeze audit, a candidate revision may change assigned bytes only when the same
   reviewed change updates the normative documents, Rust layout assertions, manifest, vectors, and
   compatibility tests and records the rationale for independent reviewers.
2. After the freeze, an erratum that changes no accepted or emitted bytes may clarify the 1.0
   documents and tests.
3. A compatible extension uses a previously reserved number plus explicit feature or new-opcode
   negotiation, preserves every 1.0 frame, and receives a new minor-version conformance directory.
4. After the freeze, any changed assigned number, existing payload length, field meaning, required
   response, or ownership interpretation requires a new protocol major version and a new
   conformance directory.

The same reviewed change **MUST** update the normative documents, Rust constants/layout assertions,
machine layout manifest, canonical vectors, and compatibility tests. Protocol version directories
are never regenerated opportunistically from current Rust types.
