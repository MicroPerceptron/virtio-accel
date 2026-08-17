#ifndef VIRTIO_ACCEL_QNN_BRIDGE_H
#define VIRTIO_ACCEL_QNN_BRIDGE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct VaQnnRuntime VaQnnRuntime;
typedef struct VaQnnGraph VaQnnGraph;
typedef struct VaQnnEvent VaQnnEvent;

enum VaQnnNodeKind {
  VA_QNN_NODE_RESHAPE = 1,
  VA_QNN_NODE_MATMUL = 2,
  VA_QNN_NODE_MAX_POOL_2D = 3,
  VA_QNN_NODE_ADD = 4,
  VA_QNN_NODE_SUBTRACT = 5,
  VA_QNN_NODE_MAXIMUM = 6,
  VA_QNN_NODE_MINIMUM = 7,
  VA_QNN_NODE_MULTIPLY = 8,
  VA_QNN_NODE_TRANSPOSE = 9,
  VA_QNN_NODE_REVERSE = 10,
  VA_QNN_NODE_CONCAT = 11,
  VA_QNN_NODE_POWER = 12,
  VA_QNN_NODE_ABS = 13,
  VA_QNN_NODE_CEIL = 14,
  VA_QNN_NODE_COS = 15,
  VA_QNN_NODE_EXP = 16,
  VA_QNN_NODE_FLOOR = 17,
  VA_QNN_NODE_LOG = 18,
  VA_QNN_NODE_NEGATE = 19,
  VA_QNN_NODE_RECIPROCAL = 20,
  VA_QNN_NODE_RSQRT = 21,
  VA_QNN_NODE_SIN = 22,
  VA_QNN_NODE_SIGMOID = 23,
  VA_QNN_NODE_TANH = 24,
  VA_QNN_NODE_CLAMP = 25,
  VA_QNN_NODE_EQUAL = 26,
  VA_QNN_NODE_GREATER = 27,
  VA_QNN_NODE_GREATER_EQUAL = 28,
  VA_QNN_NODE_SELECT = 29,
  VA_QNN_NODE_LOGICAL_AND = 30,
  VA_QNN_NODE_LOGICAL_OR = 31,
  VA_QNN_NODE_LOGICAL_XOR = 32,
  VA_QNN_NODE_LOGICAL_NOT = 33,
  VA_QNN_NODE_ARGMAX = 34,
  VA_QNN_NODE_REDUCE_MAX = 35,
  VA_QNN_NODE_REDUCE_MIN = 36,
  VA_QNN_NODE_REDUCE_PRODUCT = 37,
  VA_QNN_NODE_REDUCE_SUM = 38,
};

enum VaQnnTensorRole {
  VA_QNN_TENSOR_NATIVE = 0,
  VA_QNN_TENSOR_INPUT = 1,
  VA_QNN_TENSOR_OUTPUT = 2,
  VA_QNN_TENSOR_STATIC = 3,
};

enum VaQnnElement {
  VA_QNN_ELEMENT_BOOL = 0,
  VA_QNN_ELEMENT_F16 = 1,
  VA_QNN_ELEMENT_F32 = 2,
  VA_QNN_ELEMENT_I8 = 3,
  VA_QNN_ELEMENT_I32 = 4,
};

enum VaQnnPrecision {
  VA_QNN_PRECISION_DEFAULT = 0,
  VA_QNN_PRECISION_F16 = 1,
  VA_QNN_PRECISION_F32 = 2,
};

enum VaQnnEventState {
  VA_QNN_EVENT_PENDING = 0,
  VA_QNN_EVENT_COMPLETE = 1,
  VA_QNN_EVENT_FAILED = 2,
};

#define VA_QNN_ERROR_INTERNAL UINT64_MAX
#define VA_QNN_ERROR_INCOMPATIBLE (UINT64_MAX - UINT64_C(1))
#define VA_QNN_ERROR_BUSY (UINT64_MAX - UINT64_C(2))
#define VA_QNN_ERROR_INVALID_ARGUMENT (UINT64_MAX - UINT64_C(3))
#define VA_QNN_ERROR_OUT_OF_MEMORY (UINT64_MAX - UINT64_C(4))

typedef struct VaQnnRuntimeInfo {
  uint32_t backend_id;
  uint32_t core_major;
  uint32_t core_minor;
  uint32_t core_patch;
  uint32_t backend_major;
  uint32_t backend_minor;
  uint32_t backend_patch;
  char provider_name[128];
  char build_id[256];
} VaQnnRuntimeInfo;

typedef struct VaQnnTensorDesc {
  uint32_t value;
  uint32_t role;
  uint32_t io_index;
  uint32_t element;
  uint32_t quantized;
  uint32_t rank;
  const uint32_t *dimensions;
  const uint8_t *constant_data;
  uint64_t constant_size;
  float scale;
  int32_t offset;
} VaQnnTensorDesc;

typedef struct VaQnnNodeDesc {
  uint32_t kind;
  const uint32_t *inputs;
  uint32_t input_count;
  const uint32_t *outputs;
  uint32_t output_count;
  const int32_t *parameters;
  uint32_t parameter_count;
} VaQnnNodeDesc;

typedef struct VaQnnBinding {
  void *data;
  uint64_t size;
} VaQnnBinding;

uint64_t va_qnn_runtime_create(const char *library_path, VaQnnRuntime **runtime,
                               VaQnnRuntimeInfo *info, char *message,
                               size_t message_size);
uint64_t va_qnn_runtime_free(VaQnnRuntime *runtime);

uint64_t va_qnn_graph_create(VaQnnRuntime *runtime,
                             const VaQnnTensorDesc *tensors,
                             uint32_t tensor_count, const VaQnnNodeDesc *nodes,
                             uint32_t node_count, uint32_t precision,
                             VaQnnGraph **graph,
                             char *message, size_t message_size);
uint64_t va_qnn_graph_free(VaQnnGraph *graph);

uint64_t va_qnn_graph_execute_async(VaQnnGraph *graph,
                                    const VaQnnBinding *inputs,
                                    uint32_t input_count,
                                    const VaQnnBinding *outputs,
                                    uint32_t output_count, VaQnnEvent **event,
                                    char *message, size_t message_size);
uint32_t va_qnn_event_poll(const VaQnnEvent *event, uint64_t *qnn_error);
uint64_t va_qnn_event_free(VaQnnEvent *event);

#ifdef __cplusplus
}
#endif

#endif
