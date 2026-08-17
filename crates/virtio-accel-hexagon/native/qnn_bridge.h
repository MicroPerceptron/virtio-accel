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
};

enum VaQnnTensorRole {
  VA_QNN_TENSOR_NATIVE = 0,
  VA_QNN_TENSOR_INPUT = 1,
  VA_QNN_TENSOR_OUTPUT = 2,
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
  const uint32_t *dimensions;
  uint32_t rank;
} VaQnnTensorDesc;

typedef struct VaQnnNodeDesc {
  uint32_t kind;
  uint32_t input0;
  uint32_t input1;
  uint32_t output;
  uint32_t kernel[2];
  uint32_t stride[2];
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
                             uint32_t node_count, VaQnnGraph **graph,
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
