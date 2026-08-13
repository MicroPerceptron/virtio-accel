/* SPDX-License-Identifier: MIT OR Apache-2.0 */
#ifndef VIRTIO_ACCEL_H
#define VIRTIO_ACCEL_H

/*
 * Language-neutral wire definitions for virtio-accel protocol 1.0.
 *
 * The normative contract is docs/wire-abi.md plus docs/virtqueue.md.  This
 * header is a checked C projection of conformance/v1.0/layout.json; it is not
 * a host backend plugin ABI.  Every multibyte field is stored little-endian
 * and every structure is byte-aligned.  Consumers must use unaligned-safe
 * little-endian loads and stores rather than directly dereferencing packed
 * multibyte members.
 */

#include <stddef.h>
#include <stdint.h>

#define VIRTIO_ACCEL_PROTOCOL_MAJOR UINT16_C(1)
#define VIRTIO_ACCEL_PROTOCOL_MINOR UINT16_C(0)

#define VIRTIO_ACCEL_COMMAND_QUEUE UINT16_C(0)
#define VIRTIO_ACCEL_BASELINE_COMMAND_QUEUES UINT16_C(1)
#define VIRTIO_ACCEL_HARD_MAX_CHAIN_DESCRIPTORS UINT16_C(256)
#define VIRTIO_ACCEL_MIN_MAX_REQUEST_BYTES UINT32_C(97)
#define VIRTIO_ACCEL_MIN_MAX_RESPONSE_BYTES UINT32_C(92)
#define VIRTIO_ACCEL_HARD_MAX_REQUEST_BYTES UINT32_C(16777216)
#define VIRTIO_ACCEL_HARD_MAX_RESPONSE_BYTES UINT32_C(16777216)
#define VIRTIO_ACCEL_HARD_MAX_BINDINGS UINT32_C(4096)

/* Device-specific feature bit positions.  Protocol 1.0 advertises none. */
#define VIRTIO_ACCEL_F_MULTI_QUEUE 0
#define VIRTIO_ACCEL_F_EVENT_QUEUE 1
#define VIRTIO_ACCEL_F_EXTERNAL_MEMORY 2
#define VIRTIO_ACCEL_F_TIMELINE_FENCES 3
#define VIRTIO_ACCEL_F_SECURE_CONTEXTS 4
#define VIRTIO_ACCEL_BASELINE_FEATURES UINT64_C(0)
#define VIRTIO_ACCEL_RESERVED_FEATURES UINT64_C(0x1f)

/* Semantic capability masks returned by GET_DEVICE_INFO. */
#define VIRTIO_ACCEL_CAP_HOST_VISIBLE_MEMORY (UINT64_C(1) << 0)
#define VIRTIO_ACCEL_CAP_DEVICE_LOCAL_MEMORY (UINT64_C(1) << 1)
#define VIRTIO_ACCEL_CAP_EVENT_CANCELLATION (UINT64_C(1) << 2)
#define VIRTIO_ACCEL_CAP_EXTERNAL_MEMORY (UINT64_C(1) << 3) /* reserved */
#define VIRTIO_ACCEL_CAP_SECURE_CONTEXTS (UINT64_C(1) << 4) /* reserved */
#define VIRTIO_ACCEL_CAP_SHARED_MEMORY (UINT64_C(1) << 5)

/* Accelerator classes.  Unknown uint16_t values remain representable. */
#define VIRTIO_ACCEL_CLASS_OTHER UINT16_C(0)
#define VIRTIO_ACCEL_CLASS_NPU UINT16_C(1)
#define VIRTIO_ACCEL_CLASS_GPU UINT16_C(2)
#define VIRTIO_ACCEL_CLASS_DSP UINT16_C(3)

/* Buffer placement intent. */
#define VIRTIO_ACCEL_MEMORY_HOST UINT8_C(1)
#define VIRTIO_ACCEL_MEMORY_DEVICE UINT8_C(2)
#define VIRTIO_ACCEL_MEMORY_SHARED UINT8_C(3)

/* Buffer usage masks. */
#define VIRTIO_ACCEL_BUFFER_TRANSFER_SOURCE (UINT32_C(1) << 0)
#define VIRTIO_ACCEL_BUFFER_TRANSFER_DESTINATION (UINT32_C(1) << 1)
#define VIRTIO_ACCEL_BUFFER_PROGRAM_INPUT (UINT32_C(1) << 2)
#define VIRTIO_ACCEL_BUFFER_PROGRAM_OUTPUT (UINT32_C(1) << 3)
#define VIRTIO_ACCEL_BUFFER_MUTABLE_STATE (UINT32_C(1) << 4)
#define VIRTIO_ACCEL_KNOWN_BUFFER_USAGE_BITS UINT32_C(0x1f)

/* Program binding access values. */
#define VIRTIO_ACCEL_ACCESS_READ UINT8_C(1)
#define VIRTIO_ACCEL_ACCESS_WRITE UINT8_C(2)
#define VIRTIO_ACCEL_ACCESS_READ_WRITE UINT8_C(3)

/* Request opcodes. */
#define VIRTIO_ACCEL_OP_GET_DEVICE_INFO UINT16_C(0x0001)
#define VIRTIO_ACCEL_OP_CREATE_CONTEXT UINT16_C(0x0100)
#define VIRTIO_ACCEL_OP_DESTROY_CONTEXT UINT16_C(0x0101)
#define VIRTIO_ACCEL_OP_ALLOCATE_BUFFER UINT16_C(0x0200)
#define VIRTIO_ACCEL_OP_FREE_BUFFER UINT16_C(0x0201)
#define VIRTIO_ACCEL_OP_WRITE_BUFFER UINT16_C(0x0202)
#define VIRTIO_ACCEL_OP_READ_BUFFER UINT16_C(0x0203)
#define VIRTIO_ACCEL_OP_LOAD_PROGRAM UINT16_C(0x0300)
#define VIRTIO_ACCEL_OP_UNLOAD_PROGRAM UINT16_C(0x0301)
#define VIRTIO_ACCEL_OP_CREATE_QUEUE UINT16_C(0x0400)
#define VIRTIO_ACCEL_OP_DESTROY_QUEUE UINT16_C(0x0401)
#define VIRTIO_ACCEL_OP_SUBMIT UINT16_C(0x0500)
#define VIRTIO_ACCEL_OP_POLL_EVENT UINT16_C(0x0501)
#define VIRTIO_ACCEL_OP_CANCEL_EVENT UINT16_C(0x0502)
#define VIRTIO_ACCEL_OP_DESTROY_EVENT UINT16_C(0x0503)

/* Response status values.  Unknown values are opaque non-success failures. */
#define VIRTIO_ACCEL_STATUS_OK UINT16_C(0)
#define VIRTIO_ACCEL_STATUS_UNSUPPORTED UINT16_C(1)
#define VIRTIO_ACCEL_STATUS_INCOMPATIBLE UINT16_C(2)
#define VIRTIO_ACCEL_STATUS_INVALID_ARGUMENT UINT16_C(3)
#define VIRTIO_ACCEL_STATUS_OUT_OF_BOUNDS UINT16_C(4)
#define VIRTIO_ACCEL_STATUS_BUSY UINT16_C(5)
#define VIRTIO_ACCEL_STATUS_OUT_OF_MEMORY UINT16_C(6)
#define VIRTIO_ACCEL_STATUS_RESOURCE_LIMIT UINT16_C(7)
#define VIRTIO_ACCEL_STATUS_DEADLINE_EXPIRED UINT16_C(8)
#define VIRTIO_ACCEL_STATUS_DEVICE_LOST UINT16_C(9)
#define VIRTIO_ACCEL_STATUS_PERMISSION_DENIED UINT16_C(10)
#define VIRTIO_ACCEL_STATUS_STALE_OBJECT UINT16_C(11)
#define VIRTIO_ACCEL_STATUS_INTERNAL_ERROR UINT16_C(0xffff)

/* Event state values returned by POLL_EVENT. */
#define VIRTIO_ACCEL_EVENT_PENDING UINT16_C(0)
#define VIRTIO_ACCEL_EVENT_COMPLETE UINT16_C(1)
#define VIRTIO_ACCEL_EVENT_FAILED UINT16_C(2)
#define VIRTIO_ACCEL_EVENT_CANCELLED UINT16_C(3)

/* Protocol 1.0 accepts no request or object-specific flag bits. */
#define VIRTIO_ACCEL_KNOWN_REQUEST_FLAGS UINT16_C(0)
#define VIRTIO_ACCEL_KNOWN_CONTEXT_FLAGS UINT32_C(0)
#define VIRTIO_ACCEL_KNOWN_PROGRAM_FLAGS UINT32_C(0)
#define VIRTIO_ACCEL_KNOWN_QUEUE_FLAGS UINT32_C(0)
#define VIRTIO_ACCEL_KNOWN_SUBMIT_FLAGS UINT32_C(0)

/*
 * C has no standard packed-structure spelling.  The compilers supported by
 * this project all honor pack(push, 1); the assertions below fail compilation
 * instead of silently exposing a differently aligned ABI.
 */
#pragma pack(push, 1)

struct virtio_accel_config {
    uint16_t protocol_major;
    uint16_t protocol_minor;
    uint16_t command_queue_count;
    uint16_t max_chain_descriptors;
    uint32_t max_request_bytes;
    uint32_t max_response_bytes;
};

struct virtio_accel_request_header {
    uint16_t opcode;
    uint16_t flags;
    uint32_t payload_bytes;
    uint64_t request_id;
};

struct virtio_accel_response_header {
    uint16_t status;
    uint16_t flags;
    uint32_t payload_bytes;
    uint64_t request_id;
};

struct virtio_accel_device_info {
    uint8_t uuid[16];
    uint16_t accelerator_class;
    uint16_t reserved;
    uint32_t vendor_id;
    uint32_t device_id;
    uint64_t capabilities;
    uint32_t max_contexts;
    uint32_t max_buffers_per_context;
    uint32_t max_programs_per_context;
    uint32_t max_queues_per_context;
    uint32_t max_events_per_context;
    uint32_t max_bindings_per_submission;
    uint64_t max_buffer_bytes;
    uint64_t max_artifact_bytes;
};

struct virtio_accel_create_context_request {
    uint32_t flags;
    uint32_t reserved;
};

struct virtio_accel_object_payload {
    uint64_t object_id;
};

struct virtio_accel_allocate_buffer_request {
    uint64_t context_id;
    uint64_t bytes;
    uint64_t alignment;
    uint8_t memory_domain;
    uint8_t reserved0[7];
    uint32_t usage;
    uint32_t reserved1;
};

struct virtio_accel_transfer_buffer_request {
    uint64_t buffer_id;
    uint64_t offset;
    uint64_t bytes;
};

struct virtio_accel_load_program_request {
    uint64_t context_id;
    uint32_t format;
    uint32_t flags;
    uint32_t target[12];
    uint64_t payload_bytes;
    uint64_t resident_bytes;
};

struct virtio_accel_create_queue_request {
    uint64_t context_id;
    uint32_t flags;
    uint32_t reserved;
};

struct virtio_accel_submit_request {
    uint64_t queue_id;
    uint64_t program_id;
    uint32_t binding_count;
    uint32_t flags;
    uint64_t timeout_ns;
};

struct virtio_accel_binding {
    uint64_t buffer_id;
    uint64_t offset;
    uint64_t bytes;
    uint32_t slot;
    uint8_t access;
    uint8_t reserved[3];
};

struct virtio_accel_submit_response {
    uint64_t event_id;
};

struct virtio_accel_event_state {
    uint16_t state;
    uint16_t error;
    uint32_t reserved;
};

#pragma pack(pop)

#if defined(__cplusplus)
#define VIRTIO_ACCEL_STATIC_ASSERT(condition, message) static_assert(condition, message)
#define VIRTIO_ACCEL_ALIGNOF(type) alignof(type)
#else
#define VIRTIO_ACCEL_STATIC_ASSERT(condition, message) _Static_assert(condition, message)
#define VIRTIO_ACCEL_ALIGNOF(type) _Alignof(type)
#endif

#define VIRTIO_ACCEL_ASSERT_LAYOUT(type, bytes)                                      \
    VIRTIO_ACCEL_STATIC_ASSERT(sizeof(struct type) == (bytes), #type " size");       \
    VIRTIO_ACCEL_STATIC_ASSERT(VIRTIO_ACCEL_ALIGNOF(struct type) == 1, #type " align")
#define VIRTIO_ACCEL_ASSERT_OFFSET(type, field, offset)                              \
    VIRTIO_ACCEL_STATIC_ASSERT(offsetof(struct type, field) == (offset),             \
                               #type "." #field " offset")

VIRTIO_ACCEL_ASSERT_LAYOUT(virtio_accel_config, 16);
VIRTIO_ACCEL_ASSERT_OFFSET(virtio_accel_config, protocol_major, 0);
VIRTIO_ACCEL_ASSERT_OFFSET(virtio_accel_config, protocol_minor, 2);
VIRTIO_ACCEL_ASSERT_OFFSET(virtio_accel_config, command_queue_count, 4);
VIRTIO_ACCEL_ASSERT_OFFSET(virtio_accel_config, max_chain_descriptors, 6);
VIRTIO_ACCEL_ASSERT_OFFSET(virtio_accel_config, max_request_bytes, 8);
VIRTIO_ACCEL_ASSERT_OFFSET(virtio_accel_config, max_response_bytes, 12);

VIRTIO_ACCEL_ASSERT_LAYOUT(virtio_accel_request_header, 16);
VIRTIO_ACCEL_ASSERT_OFFSET(virtio_accel_request_header, opcode, 0);
VIRTIO_ACCEL_ASSERT_OFFSET(virtio_accel_request_header, flags, 2);
VIRTIO_ACCEL_ASSERT_OFFSET(virtio_accel_request_header, payload_bytes, 4);
VIRTIO_ACCEL_ASSERT_OFFSET(virtio_accel_request_header, request_id, 8);

VIRTIO_ACCEL_ASSERT_LAYOUT(virtio_accel_response_header, 16);
VIRTIO_ACCEL_ASSERT_OFFSET(virtio_accel_response_header, status, 0);
VIRTIO_ACCEL_ASSERT_OFFSET(virtio_accel_response_header, flags, 2);
VIRTIO_ACCEL_ASSERT_OFFSET(virtio_accel_response_header, payload_bytes, 4);
VIRTIO_ACCEL_ASSERT_OFFSET(virtio_accel_response_header, request_id, 8);

VIRTIO_ACCEL_ASSERT_LAYOUT(virtio_accel_device_info, 76);
VIRTIO_ACCEL_ASSERT_OFFSET(virtio_accel_device_info, uuid, 0);
VIRTIO_ACCEL_ASSERT_OFFSET(virtio_accel_device_info, accelerator_class, 16);
VIRTIO_ACCEL_ASSERT_OFFSET(virtio_accel_device_info, reserved, 18);
VIRTIO_ACCEL_ASSERT_OFFSET(virtio_accel_device_info, vendor_id, 20);
VIRTIO_ACCEL_ASSERT_OFFSET(virtio_accel_device_info, device_id, 24);
VIRTIO_ACCEL_ASSERT_OFFSET(virtio_accel_device_info, capabilities, 28);
VIRTIO_ACCEL_ASSERT_OFFSET(virtio_accel_device_info, max_contexts, 36);
VIRTIO_ACCEL_ASSERT_OFFSET(virtio_accel_device_info, max_buffers_per_context, 40);
VIRTIO_ACCEL_ASSERT_OFFSET(virtio_accel_device_info, max_programs_per_context, 44);
VIRTIO_ACCEL_ASSERT_OFFSET(virtio_accel_device_info, max_queues_per_context, 48);
VIRTIO_ACCEL_ASSERT_OFFSET(virtio_accel_device_info, max_events_per_context, 52);
VIRTIO_ACCEL_ASSERT_OFFSET(virtio_accel_device_info, max_bindings_per_submission, 56);
VIRTIO_ACCEL_ASSERT_OFFSET(virtio_accel_device_info, max_buffer_bytes, 60);
VIRTIO_ACCEL_ASSERT_OFFSET(virtio_accel_device_info, max_artifact_bytes, 68);

VIRTIO_ACCEL_ASSERT_LAYOUT(virtio_accel_create_context_request, 8);
VIRTIO_ACCEL_ASSERT_OFFSET(virtio_accel_create_context_request, flags, 0);
VIRTIO_ACCEL_ASSERT_OFFSET(virtio_accel_create_context_request, reserved, 4);

VIRTIO_ACCEL_ASSERT_LAYOUT(virtio_accel_object_payload, 8);
VIRTIO_ACCEL_ASSERT_OFFSET(virtio_accel_object_payload, object_id, 0);

VIRTIO_ACCEL_ASSERT_LAYOUT(virtio_accel_allocate_buffer_request, 40);
VIRTIO_ACCEL_ASSERT_OFFSET(virtio_accel_allocate_buffer_request, context_id, 0);
VIRTIO_ACCEL_ASSERT_OFFSET(virtio_accel_allocate_buffer_request, bytes, 8);
VIRTIO_ACCEL_ASSERT_OFFSET(virtio_accel_allocate_buffer_request, alignment, 16);
VIRTIO_ACCEL_ASSERT_OFFSET(virtio_accel_allocate_buffer_request, memory_domain, 24);
VIRTIO_ACCEL_ASSERT_OFFSET(virtio_accel_allocate_buffer_request, reserved0, 25);
VIRTIO_ACCEL_ASSERT_OFFSET(virtio_accel_allocate_buffer_request, usage, 32);
VIRTIO_ACCEL_ASSERT_OFFSET(virtio_accel_allocate_buffer_request, reserved1, 36);

VIRTIO_ACCEL_ASSERT_LAYOUT(virtio_accel_transfer_buffer_request, 24);
VIRTIO_ACCEL_ASSERT_OFFSET(virtio_accel_transfer_buffer_request, buffer_id, 0);
VIRTIO_ACCEL_ASSERT_OFFSET(virtio_accel_transfer_buffer_request, offset, 8);
VIRTIO_ACCEL_ASSERT_OFFSET(virtio_accel_transfer_buffer_request, bytes, 16);

VIRTIO_ACCEL_ASSERT_LAYOUT(virtio_accel_load_program_request, 80);
VIRTIO_ACCEL_ASSERT_OFFSET(virtio_accel_load_program_request, context_id, 0);
VIRTIO_ACCEL_ASSERT_OFFSET(virtio_accel_load_program_request, format, 8);
VIRTIO_ACCEL_ASSERT_OFFSET(virtio_accel_load_program_request, flags, 12);
VIRTIO_ACCEL_ASSERT_OFFSET(virtio_accel_load_program_request, target, 16);
VIRTIO_ACCEL_ASSERT_OFFSET(virtio_accel_load_program_request, payload_bytes, 64);
VIRTIO_ACCEL_ASSERT_OFFSET(virtio_accel_load_program_request, resident_bytes, 72);

VIRTIO_ACCEL_ASSERT_LAYOUT(virtio_accel_create_queue_request, 16);
VIRTIO_ACCEL_ASSERT_OFFSET(virtio_accel_create_queue_request, context_id, 0);
VIRTIO_ACCEL_ASSERT_OFFSET(virtio_accel_create_queue_request, flags, 8);
VIRTIO_ACCEL_ASSERT_OFFSET(virtio_accel_create_queue_request, reserved, 12);

VIRTIO_ACCEL_ASSERT_LAYOUT(virtio_accel_submit_request, 32);
VIRTIO_ACCEL_ASSERT_OFFSET(virtio_accel_submit_request, queue_id, 0);
VIRTIO_ACCEL_ASSERT_OFFSET(virtio_accel_submit_request, program_id, 8);
VIRTIO_ACCEL_ASSERT_OFFSET(virtio_accel_submit_request, binding_count, 16);
VIRTIO_ACCEL_ASSERT_OFFSET(virtio_accel_submit_request, flags, 20);
VIRTIO_ACCEL_ASSERT_OFFSET(virtio_accel_submit_request, timeout_ns, 24);

VIRTIO_ACCEL_ASSERT_LAYOUT(virtio_accel_binding, 32);
VIRTIO_ACCEL_ASSERT_OFFSET(virtio_accel_binding, buffer_id, 0);
VIRTIO_ACCEL_ASSERT_OFFSET(virtio_accel_binding, offset, 8);
VIRTIO_ACCEL_ASSERT_OFFSET(virtio_accel_binding, bytes, 16);
VIRTIO_ACCEL_ASSERT_OFFSET(virtio_accel_binding, slot, 24);
VIRTIO_ACCEL_ASSERT_OFFSET(virtio_accel_binding, access, 28);
VIRTIO_ACCEL_ASSERT_OFFSET(virtio_accel_binding, reserved, 29);

VIRTIO_ACCEL_ASSERT_LAYOUT(virtio_accel_submit_response, 8);
VIRTIO_ACCEL_ASSERT_OFFSET(virtio_accel_submit_response, event_id, 0);

VIRTIO_ACCEL_ASSERT_LAYOUT(virtio_accel_event_state, 8);
VIRTIO_ACCEL_ASSERT_OFFSET(virtio_accel_event_state, state, 0);
VIRTIO_ACCEL_ASSERT_OFFSET(virtio_accel_event_state, error, 2);
VIRTIO_ACCEL_ASSERT_OFFSET(virtio_accel_event_state, reserved, 4);

#undef VIRTIO_ACCEL_ASSERT_OFFSET
#undef VIRTIO_ACCEL_ASSERT_LAYOUT
#undef VIRTIO_ACCEL_ALIGNOF
#undef VIRTIO_ACCEL_STATIC_ASSERT

#endif /* VIRTIO_ACCEL_H */
