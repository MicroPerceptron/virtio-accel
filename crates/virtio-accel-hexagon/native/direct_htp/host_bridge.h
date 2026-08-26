// SPDX-License-Identifier: MIT OR Apache-2.0
#ifndef VIRTIO_ACCEL_DIRECT_HTP_BRIDGE_H
#define VIRTIO_ACCEL_DIRECT_HTP_BRIDGE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct VaHtpRuntime VaHtpRuntime;

typedef struct VaHtpRuntimeInfo {
    uint32_t arch;
    uint32_t hvx_units;
    uint32_t vtcm_bytes;
    uint32_t arena_bytes;
} VaHtpRuntimeInfo;

typedef struct VaHtpBinding {
    void *address;
    uint32_t bytes;
    uint32_t slot;
    uint32_t access;
} VaHtpBinding;

enum {
    VA_HTP_SUCCESS = 0,
    VA_HTP_ERROR_UNAVAILABLE = 1,
    VA_HTP_ERROR_INCOMPATIBLE = 2,
    VA_HTP_ERROR_INVALID_ARGUMENT = 3,
    VA_HTP_ERROR_OUT_OF_MEMORY = 4,
    VA_HTP_ERROR_DEVICE_LOST = 5,
    VA_HTP_ERROR_BUSY = 6,
};

uint64_t va_htp_runtime_create(
    const char *module_directory,
    uint32_t arena_bytes,
    VaHtpRuntime **runtime,
    VaHtpRuntimeInfo *info,
    char *message,
    size_t message_bytes);
uint64_t va_htp_runtime_free(VaHtpRuntime *runtime);

uint64_t va_htp_buffer_alloc(
    VaHtpRuntime *runtime,
    uint32_t bytes,
    uint32_t alignment,
    uint32_t *offset,
    void **address);
uint64_t va_htp_buffer_free(VaHtpRuntime *runtime, void *address, uint32_t bytes);

uint64_t va_htp_execute_direct(
    VaHtpRuntime *runtime,
    uint32_t opcode,
    uint32_t lanes,
    const void *parameters,
    uint32_t parameter_bytes,
    const VaHtpBinding *bindings,
    uint32_t binding_count,
    uint64_t *elapsed_cycles);

#ifdef __cplusplus
}
#endif
#endif
