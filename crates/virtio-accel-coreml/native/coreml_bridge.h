#ifndef VIRTIO_ACCEL_COREML_BRIDGE_H
#define VIRTIO_ACCEL_COREML_BRIDGE_H

#include <stddef.h>
#include <stdint.h>

enum va_coreml_error_kind {
    VA_COREML_OK = 0,
    VA_COREML_UNSUPPORTED = 1,
    VA_COREML_INCOMPATIBLE = 2,
    VA_COREML_INVALID_ARGUMENT = 3,
    VA_COREML_OUT_OF_BOUNDS = 4,
    VA_COREML_OUT_OF_MEMORY = 5,
    VA_COREML_RESOURCE_LIMIT = 6,
    VA_COREML_DEVICE_LOST = 7,
    VA_COREML_EXTERNAL = 8,
};

enum va_coreml_feature_role {
    VA_COREML_INPUT = 1,
    VA_COREML_OUTPUT = 2,
};

enum va_coreml_access_mode {
    VA_COREML_READ = 1,
    VA_COREML_WRITE = 2,
    VA_COREML_READ_WRITE = 3,
};

enum va_coreml_event_status {
    VA_COREML_EVENT_PENDING = 0,
    VA_COREML_EVENT_COMPLETE = 1,
    VA_COREML_EVENT_FAILED = 2,
};

struct va_coreml_error {
    uint32_t kind;
    uint32_t domain;
    int64_t code;
};

struct va_coreml_feature_mapping {
    uint32_t slot;
    uint8_t role;
    const uint8_t *name;
    size_t name_len;
};

struct va_coreml_binding {
    uint32_t slot;
    uint8_t access;
    void *data;
    uint64_t bytes;
};

typedef void (*va_coreml_release_context_fn)(void *context);

int va_coreml_has_neural_engine(void);

void *va_coreml_model_load(const uint8_t *path,
                           size_t path_len,
                           const struct va_coreml_feature_mapping *mappings,
                           size_t mapping_count,
                           struct va_coreml_error *error);

void va_coreml_model_release(void *model);

void *va_coreml_submit(void *model,
                       const struct va_coreml_binding *bindings,
                       size_t binding_count,
                       void *context,
                       va_coreml_release_context_fn release_context,
                       struct va_coreml_error *error);

uint32_t va_coreml_event_poll(void *event, struct va_coreml_error *error);

void va_coreml_event_release(void *event);

#endif
