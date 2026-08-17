#define _CRT_SECURE_NO_WARNINGS
#define NOMINMAX

#include "qnn_bridge.h"

#include <HTP/QnnHtpCommon.h>
#include <HTP/QnnHtpGraph.h>
#include <QnnInterface.h>
#include <QnnOpDef.h>

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <atomic>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

void set_message(char *output, size_t size, const char *message) {
  if (output == nullptr || size == 0) {
    return;
  }
  const char *source = message == nullptr ? "" : message;
  std::strncpy(output, source, size - 1);
  output[size - 1] = '\0';
}

template <size_t N> void copy_text(char (&output)[N], const char *source) {
  const char *text = source == nullptr ? "" : source;
  std::strncpy(output, text, N - 1);
  output[N - 1] = '\0';
}

struct TensorStorage {
  uint32_t value = 0;
  std::string name;
  std::vector<uint32_t> dimensions;
  std::vector<uint32_t> constant_values;
  std::vector<uint8_t> constant_data;
  Qnn_Tensor_t tensor = []() {
    Qnn_Tensor_t value{};
    value.version = QNN_TENSOR_VERSION_2;
    value.v2 = QNN_TENSOR_V2_INIT;
    return value;
  }();
};

struct NodeStorage {
  std::string name;
  std::string type;
  std::vector<Qnn_Tensor_t> inputs;
  std::vector<Qnn_Tensor_t> outputs;
  std::vector<Qnn_Param_t> params;
  Qnn_OpConfig_t config = QNN_OPCONFIG_INIT;

  void finish() {
    config.v1.name = name.c_str();
    config.v1.packageName = QNN_OP_PACKAGE_NAME_QTI_AISW;
    config.v1.typeName = type.c_str();
    config.v1.numOfParams = static_cast<uint32_t>(params.size());
    config.v1.params = params.empty() ? nullptr : params.data();
    config.v1.numOfInputs = static_cast<uint32_t>(inputs.size());
    config.v1.inputTensors = inputs.data();
    config.v1.numOfOutputs = static_cast<uint32_t>(outputs.size());
    config.v1.outputTensors = outputs.data();
  }
};

} // namespace

struct VaQnnRuntime {
  HMODULE library = nullptr;
  QNN_INTERFACE_VER_TYPE api = QNN_INTERFACE_VER_TYPE_INIT;
  Qnn_BackendHandle_t backend = nullptr;
  Qnn_DeviceHandle_t device = nullptr;
  std::atomic<uint32_t> in_flight{0};
};

struct VaQnnGraph {
  VaQnnRuntime *runtime = nullptr;
  Qnn_ContextHandle_t context = nullptr;
  Qnn_GraphHandle_t graph = nullptr;
  std::vector<TensorStorage> tensors;
  std::unordered_map<uint32_t, size_t> by_value;
  std::vector<size_t> inputs;
  std::vector<size_t> outputs;
  std::vector<std::unique_ptr<NodeStorage>> nodes;
};

struct VaQnnEvent {
  VaQnnRuntime *runtime = nullptr;
  std::atomic<uint32_t> state{VA_QNN_EVENT_PENDING};
  std::atomic<uint64_t> error{QNN_SUCCESS};
  std::vector<Qnn_Tensor_t> inputs;
  std::vector<Qnn_Tensor_t> outputs;
  std::thread worker;
};

namespace {

const QnnInterface_t *select_provider(const QnnInterface_t **providers,
                                      uint32_t count) {
  for (uint32_t index = 0; index < count; ++index) {
    const QnnInterface_t *provider = providers[index];
    if (provider != nullptr && provider->backendId == QNN_BACKEND_ID_HTP &&
        provider->apiVersion.coreApiVersion.major == QNN_API_VERSION_MAJOR &&
        provider->apiVersion.coreApiVersion.minor >= QNN_API_VERSION_MINOR) {
      return provider;
    }
  }
  return nullptr;
}

bool required_api_present(const QNN_INTERFACE_VER_TYPE &api) {
  return api.backendCreate != nullptr && api.backendGetApiVersion != nullptr &&
         api.backendGetBuildId != nullptr && api.backendFree != nullptr &&
         api.deviceCreate != nullptr && api.deviceFree != nullptr &&
         api.contextCreate != nullptr && api.contextFree != nullptr &&
         api.graphCreate != nullptr && api.graphAddNode != nullptr &&
         api.graphFinalize != nullptr && api.graphExecute != nullptr &&
         api.tensorCreateGraphTensor != nullptr;
}

Qnn_TensorType_t tensor_type(uint32_t role) {
  switch (role) {
  case VA_QNN_TENSOR_INPUT:
    return QNN_TENSOR_TYPE_APP_WRITE;
  case VA_QNN_TENSOR_OUTPUT:
    return QNN_TENSOR_TYPE_APP_READ;
  case VA_QNN_TENSOR_STATIC:
    return QNN_TENSOR_TYPE_STATIC;
  default:
    return QNN_TENSOR_TYPE_NATIVE;
  }
}

Qnn_DataType_t tensor_data_type(const VaQnnTensorDesc &description) {
  switch (description.element) {
  case VA_QNN_ELEMENT_BOOL:
    return QNN_DATATYPE_BOOL_8;
  case VA_QNN_ELEMENT_F16:
    return QNN_DATATYPE_FLOAT_16;
  case VA_QNN_ELEMENT_F32:
    return QNN_DATATYPE_FLOAT_32;
  case VA_QNN_ELEMENT_I8:
    return description.quantized != 0 ? QNN_DATATYPE_SFIXED_POINT_8
                                      : QNN_DATATYPE_INT_8;
  case VA_QNN_ELEMENT_I32:
    return description.quantized != 0 ? QNN_DATATYPE_SFIXED_POINT_32
                                      : QNN_DATATYPE_INT_32;
  default:
    return QNN_DATATYPE_UNDEFINED;
  }
}

size_t tensor_storage_bytes(const VaQnnTensorDesc &description) {
  size_t bytes = 0;
  switch (description.element) {
  case VA_QNN_ELEMENT_BOOL:
  case VA_QNN_ELEMENT_I8:
    bytes = 1;
    break;
  case VA_QNN_ELEMENT_F16:
    bytes = 2;
    break;
  case VA_QNN_ELEMENT_F32:
  case VA_QNN_ELEMENT_I32:
    bytes = 4;
    break;
  default:
    return 0;
  }
  if (description.rank != 0 && description.dimensions == nullptr) {
    return 0;
  }
  for (uint32_t index = 0; index < description.rank; ++index) {
    const size_t dimension = description.dimensions[index];
    if (dimension == 0 ||
        bytes > std::numeric_limits<size_t>::max() / dimension) {
      return 0;
    }
    bytes *= dimension;
  }
  return bytes;
}

uint64_t append_scalar_bool(NodeStorage &node, const char *name,
                            uint8_t value) {
  Qnn_Param_t parameter = QNN_PARAM_INIT;
  parameter.paramType = QNN_PARAMTYPE_SCALAR;
  parameter.name = name;
  parameter.scalarParam.dataType = QNN_DATATYPE_BOOL_8;
  parameter.scalarParam.bool8Value = value;
  node.params.push_back(parameter);
  return QNN_SUCCESS;
}

void append_scalar_i32(NodeStorage &node, const char *name, int32_t value) {
  Qnn_Param_t parameter = QNN_PARAM_INIT;
  parameter.paramType = QNN_PARAMTYPE_SCALAR;
  parameter.name = name;
  parameter.scalarParam.dataType = QNN_DATATYPE_INT_32;
  parameter.scalarParam.int32Value = value;
  node.params.push_back(parameter);
}

void append_scalar_f32(NodeStorage &node, const char *name, int32_t bits) {
  Qnn_Param_t parameter = QNN_PARAM_INIT;
  parameter.paramType = QNN_PARAMTYPE_SCALAR;
  parameter.name = name;
  parameter.scalarParam.dataType = QNN_DATATYPE_FLOAT_32;
  static_assert(sizeof(bits) == sizeof(parameter.scalarParam.floatValue));
  std::memcpy(&parameter.scalarParam.floatValue, &bits, sizeof(bits));
  node.params.push_back(parameter);
}

void append_tensor_param(NodeStorage &node, const char *name,
                         const TensorStorage &tensor) {
  Qnn_Param_t parameter = QNN_PARAM_INIT;
  parameter.paramType = QNN_PARAMTYPE_TENSOR;
  parameter.name = name;
  parameter.tensorParam = tensor.tensor;
  node.params.push_back(parameter);
}

uint64_t commit_node(VaQnnGraph &graph, std::unique_ptr<NodeStorage> node,
                     char *message, size_t message_size) {
  node->finish();
  const uint64_t status =
      graph.runtime->api.graphAddNode(graph.graph, node->config);
  if (status != QNN_SUCCESS) {
    set_message(message, message_size, "QNN rejected a graph node");
    return status;
  }
  graph.nodes.push_back(std::move(node));
  return QNN_SUCCESS;
}

uint64_t create_internal_tensor(VaQnnGraph &graph, std::string name,
                                std::vector<uint32_t> dimensions,
                                Qnn_TensorType_t type, Qnn_DataType_t data_type,
                                std::vector<uint32_t> constant_values,
                                TensorStorage **output, char *message,
                                size_t message_size) {
  TensorStorage storage;
  storage.name = std::move(name);
  storage.dimensions = std::move(dimensions);
  storage.constant_values = std::move(constant_values);
  storage.tensor.v2.name = storage.name.c_str();
  storage.tensor.v2.type = type;
  storage.tensor.v2.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
  storage.tensor.v2.dataType = data_type;
  storage.tensor.v2.rank = static_cast<uint32_t>(storage.dimensions.size());
  storage.tensor.v2.dimensions = storage.dimensions.data();
  storage.tensor.v2.memType = QNN_TENSORMEMTYPE_RAW;
  if (type == QNN_TENSOR_TYPE_STATIC) {
    storage.tensor.v2.clientBuf.data = storage.constant_values.data();
    storage.tensor.v2.clientBuf.dataSize = static_cast<uint32_t>(
        storage.constant_values.size() * sizeof(uint32_t));
  }
  graph.tensors.push_back(std::move(storage));
  TensorStorage &retained = graph.tensors.back();
  retained.tensor.v2.name = retained.name.c_str();
  retained.tensor.v2.dimensions = retained.dimensions.data();
  if (type == QNN_TENSOR_TYPE_STATIC) {
    retained.tensor.v2.clientBuf.data = retained.constant_values.data();
  }
  const uint64_t status =
      graph.runtime->api.tensorCreateGraphTensor(graph.graph, &retained.tensor);
  if (status != QNN_SUCCESS) {
    set_message(message, message_size, "QNN rejected an internal tensor");
    return status;
  }
  *output = &retained;
  return QNN_SUCCESS;
}

uint64_t add_gather(VaQnnGraph &graph, const std::string &name,
                    TensorStorage &data, TensorStorage &indices,
                    TensorStorage &output, int32_t axis, char *message,
                    size_t message_size) {
  auto node = std::make_unique<NodeStorage>();
  node->name = name;
  node->type = QNN_OP_GATHER;
  node->inputs = {data.tensor, indices.tensor};
  node->outputs = {output.tensor};
  node->params.reserve(1);
  append_scalar_i32(*node, QNN_OP_GATHER_PARAM_AXIS, axis);
  return commit_node(graph, std::move(node), message, message_size);
}

uint64_t add_binary_maximum(VaQnnGraph &graph, const std::string &name,
                            TensorStorage &left, TensorStorage &right,
                            TensorStorage &output, char *message,
                            size_t message_size) {
  auto node = std::make_unique<NodeStorage>();
  node->name = name;
  node->type = QNN_OP_ELEMENT_WISE_MAXIMUM;
  node->inputs = {left.tensor, right.tensor};
  node->outputs = {output.tensor};
  return commit_node(graph, std::move(node), message, message_size);
}

uint64_t add_reverse(VaQnnGraph &graph, const VaQnnNodeDesc &description,
                     uint32_t index, TensorStorage &input,
                     TensorStorage &output, char *message,
                     size_t message_size) {
  if (description.parameter_count != 1 || description.parameters == nullptr ||
      description.parameters[0] < 0 ||
      static_cast<size_t>(description.parameters[0]) >=
          input.dimensions.size() ||
      input.dimensions != output.dimensions) {
    set_message(message, message_size, "invalid reverse axis or shape");
    return VA_QNN_ERROR_INVALID_ARGUMENT;
  }
  const int32_t axis = description.parameters[0];
  const uint32_t extent = input.dimensions[static_cast<size_t>(axis)];
  std::vector<uint32_t> indices;
  indices.reserve(extent);
  for (uint32_t value = extent; value != 0; --value) {
    indices.push_back(value - 1);
  }
  TensorStorage *index_tensor = nullptr;
  const std::string prefix =
      "virtio_accel_reverse_" + std::to_string(index) + "_";
  uint64_t status = create_internal_tensor(
      graph, prefix + "indices", {extent}, QNN_TENSOR_TYPE_STATIC,
      QNN_DATATYPE_UINT_32, std::move(indices), &index_tensor, message,
      message_size);
  if (status != QNN_SUCCESS)
    return status;
  return add_gather(graph, prefix + "gather", input, *index_tensor, output,
                    axis, message, message_size);
}

uint64_t add_reduce_product(VaQnnGraph &graph,
                            const VaQnnNodeDesc &description, uint32_t index,
                            TensorStorage &input, TensorStorage &output,
                            char *message, size_t message_size) {
  if (description.parameter_count != 1 || description.parameters == nullptr ||
      description.parameters[0] < 0 ||
      static_cast<size_t>(description.parameters[0]) >=
          input.dimensions.size() ||
      input.dimensions.size() != output.dimensions.size()) {
    set_message(message, message_size, "invalid reduce-product axis or rank");
    return VA_QNN_ERROR_INVALID_ARGUMENT;
  }
  const int32_t axis = description.parameters[0];
  const uint32_t extent = input.dimensions[static_cast<size_t>(axis)];
  std::vector<uint32_t> reduced_dimensions = input.dimensions;
  reduced_dimensions[static_cast<size_t>(axis)] = 1;
  if (extent == 0 || reduced_dimensions != output.dimensions) {
    set_message(message, message_size, "invalid reduce-product output shape");
    return VA_QNN_ERROR_INVALID_ARGUMENT;
  }
  const std::string prefix =
      "virtio_accel_reduce_product_" + std::to_string(index) + "_";
  std::vector<TensorStorage *> values;
  values.reserve(extent);
  for (uint32_t position = 0; position < extent; ++position) {
    TensorStorage *indices = nullptr;
    uint64_t status = create_internal_tensor(
        graph, prefix + "index_" + std::to_string(position), {1},
        QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_UINT_32, {position}, &indices,
        message, message_size);
    if (status != QNN_SUCCESS)
      return status;
    TensorStorage *value = &output;
    if (extent != 1) {
      status = create_internal_tensor(
          graph, prefix + "value_" + std::to_string(position),
          reduced_dimensions, QNN_TENSOR_TYPE_NATIVE, input.tensor.v2.dataType,
          {}, &value, message, message_size);
      if (status != QNN_SUCCESS)
        return status;
    }
    status = add_gather(graph, prefix + "gather_" + std::to_string(position),
                        input, *indices, *value, axis, message, message_size);
    if (status != QNN_SUCCESS)
      return status;
    values.push_back(value);
  }
  if (extent == 1)
    return QNN_SUCCESS;
  TensorStorage *accumulator = values[0];
  for (uint32_t position = 1; position < extent; ++position) {
    TensorStorage *product = &output;
    if (position + 1 != extent) {
      uint64_t status = create_internal_tensor(
          graph, prefix + "product_" + std::to_string(position),
          reduced_dimensions, QNN_TENSOR_TYPE_NATIVE, input.tensor.v2.dataType,
          {}, &product, message, message_size);
      if (status != QNN_SUCCESS)
        return status;
    }
    auto node = std::make_unique<NodeStorage>();
    node->name = prefix + "multiply_" + std::to_string(position);
    node->type = QNN_OP_ELEMENT_WISE_MULTIPLY;
    node->inputs = {accumulator->tensor, values[position]->tensor};
    node->outputs = {product->tensor};
    const uint64_t status =
        commit_node(graph, std::move(node), message, message_size);
    if (status != QNN_SUCCESS)
      return status;
    accumulator = product;
  }
  return QNN_SUCCESS;
}

uint64_t add_max_pool(VaQnnGraph &graph, const VaQnnNodeDesc &description,
                      uint32_t index, TensorStorage &input,
                      TensorStorage &output, char *message,
                      size_t message_size) {
  // QNN's Windows HTP backend rejects the documented PoolMax2d tensor
  // parameters during graph finalization in QAIRT 2.49. Lowering the same
  // zero-padding semantics to Gather and ElementWiseMaximum keeps every
  // operation on HTP and avoids a host fallback.
  if (description.parameter_count != 4 || description.parameters == nullptr ||
      description.parameters[0] <= 0 || description.parameters[1] <= 0 ||
      description.parameters[2] <= 0 || description.parameters[3] <= 0 ||
      input.dimensions.size() != 4 || output.dimensions.size() != 4) {
    set_message(message, message_size, "invalid max-pool shape or attributes");
    return VA_QNN_ERROR_INVALID_ARGUMENT;
  }
  const uint32_t kernel[] = {
      static_cast<uint32_t>(description.parameters[0]),
      static_cast<uint32_t>(description.parameters[1])};
  const uint32_t stride[] = {
      static_cast<uint32_t>(description.parameters[2]),
      static_cast<uint32_t>(description.parameters[3])};
  const uint32_t input_height = input.dimensions[1];
  const uint32_t input_width = input.dimensions[2];
  const uint32_t output_height = output.dimensions[1];
  const uint32_t output_width = output.dimensions[2];
  if (input.dimensions[0] != output.dimensions[0] ||
      input.dimensions[3] != output.dimensions[3] ||
      kernel[0] > input_height || kernel[1] > input_width ||
      (input_height - kernel[0]) / stride[0] + 1 !=
          output_height ||
      (input_width - kernel[1]) / stride[1] + 1 !=
          output_width) {
    set_message(message, message_size,
                "max-pool output shape does not match its attributes");
    return VA_QNN_ERROR_INVALID_ARGUMENT;
  }

  const std::string prefix = "virtio_accel_pool_" + std::to_string(index) + "_";
  std::vector<TensorStorage *> row_indices;
  std::vector<TensorStorage *> column_indices;
  std::vector<TensorStorage *> row_views;
  std::vector<TensorStorage *> windows;
  row_indices.reserve(kernel[0]);
  column_indices.reserve(kernel[1]);
  row_views.reserve(kernel[0]);
  windows.reserve(static_cast<size_t>(kernel[0]) * kernel[1]);

  for (uint32_t kernel_row = 0; kernel_row < kernel[0]; ++kernel_row) {
    std::vector<uint32_t> values;
    values.reserve(output_height);
    for (uint32_t row = 0; row < output_height; ++row) {
      const uint64_t position =
          kernel_row + static_cast<uint64_t>(row) * stride[0];
      if (position >= input_height) {
        set_message(message, message_size,
                    "max-pool row index exceeds the input");
        return VA_QNN_ERROR_INVALID_ARGUMENT;
      }
      values.push_back(static_cast<uint32_t>(position));
    }
    TensorStorage *tensor = nullptr;
    uint64_t status = create_internal_tensor(
        graph, prefix + "rows_" + std::to_string(kernel_row), {output_height},
        QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_UINT_32, std::move(values),
        &tensor, message, message_size);
    if (status != QNN_SUCCESS)
      return status;
    row_indices.push_back(tensor);
  }
  for (uint32_t kernel_column = 0; kernel_column < kernel[1];
       ++kernel_column) {
    std::vector<uint32_t> values;
    values.reserve(output_width);
    for (uint32_t column = 0; column < output_width; ++column) {
      const uint64_t position =
          kernel_column + static_cast<uint64_t>(column) * stride[1];
      if (position >= input_width) {
        set_message(message, message_size,
                    "max-pool column index exceeds the input");
        return VA_QNN_ERROR_INVALID_ARGUMENT;
      }
      values.push_back(static_cast<uint32_t>(position));
    }
    TensorStorage *tensor = nullptr;
    uint64_t status = create_internal_tensor(
        graph, prefix + "columns_" + std::to_string(kernel_column),
        {output_width}, QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_UINT_32,
        std::move(values), &tensor, message, message_size);
    if (status != QNN_SUCCESS)
      return status;
    column_indices.push_back(tensor);
  }

  for (uint32_t kernel_row = 0; kernel_row < kernel[0]; ++kernel_row) {
    TensorStorage *row_view = nullptr;
    uint64_t status = create_internal_tensor(
        graph, prefix + "row_view_" + std::to_string(kernel_row),
        {input.dimensions[0], output_height, input_width, input.dimensions[3]},
        QNN_TENSOR_TYPE_NATIVE, input.tensor.v2.dataType, {}, &row_view,
        message, message_size);
    if (status != QNN_SUCCESS)
      return status;
    row_views.push_back(row_view);
    status = add_gather(
        graph, prefix + "gather_rows_" + std::to_string(kernel_row), input,
        *row_indices[kernel_row], *row_view, 1, message, message_size);
    if (status != QNN_SUCCESS)
      return status;
  }

  for (uint32_t kernel_row = 0; kernel_row < kernel[0]; ++kernel_row) {
    for (uint32_t kernel_column = 0; kernel_column < kernel[1];
         ++kernel_column) {
      TensorStorage *window = nullptr;
      const std::string suffix =
          std::to_string(kernel_row) + "_" + std::to_string(kernel_column);
      uint64_t status = create_internal_tensor(
          graph, prefix + "window_" + suffix, output.dimensions,
          QNN_TENSOR_TYPE_NATIVE, input.tensor.v2.dataType, {}, &window, message,
          message_size);
      if (status != QNN_SUCCESS)
        return status;
      windows.push_back(window);
      status = add_gather(
          graph, prefix + "gather_columns_" + suffix, *row_views[kernel_row],
          *column_indices[kernel_column], *window, 2, message, message_size);
      if (status != QNN_SUCCESS)
        return status;
    }
  }

  if (windows.size() == 1) {
    auto node = std::make_unique<NodeStorage>();
    node->name = prefix + "identity";
    node->type = QNN_OP_RESHAPE;
    node->inputs = {windows[0]->tensor};
    node->outputs = {output.tensor};
    return commit_node(graph, std::move(node), message, message_size);
  }
  TensorStorage *accumulator = windows[0];
  for (size_t window_index = 1; window_index < windows.size(); ++window_index) {
    TensorStorage *maximum = &output;
    if (window_index + 1 != windows.size()) {
      uint64_t status = create_internal_tensor(
          graph, prefix + "maximum_" + std::to_string(window_index),
          output.dimensions, QNN_TENSOR_TYPE_NATIVE, input.tensor.v2.dataType, {},
          &maximum, message, message_size);
      if (status != QNN_SUCCESS)
        return status;
    }
    const uint64_t status = add_binary_maximum(
        graph, prefix + "max_" + std::to_string(window_index), *accumulator,
        *windows[window_index], *maximum, message, message_size);
    if (status != QNN_SUCCESS)
      return status;
    accumulator = maximum;
  }
  return QNN_SUCCESS;
}

uint64_t add_node(VaQnnGraph &graph, const VaQnnNodeDesc &description,
                  uint32_t index, char *message, size_t message_size) {
  auto find_tensor = [&](uint32_t value) -> TensorStorage * {
    const auto found = graph.by_value.find(value);
    return found == graph.by_value.end() ? nullptr
                                         : &graph.tensors[found->second];
  };

  auto node = std::make_unique<NodeStorage>();
  node->name = "virtio_accel_node_" + std::to_string(index);
  node->inputs.reserve(description.input_count);
  node->outputs.reserve(description.output_count);
  for (uint32_t input_index = 0; input_index < description.input_count;
       ++input_index) {
    TensorStorage *tensor = find_tensor(description.inputs[input_index]);
    if (tensor == nullptr) {
      set_message(message, message_size, "node references an unknown input");
      return VA_QNN_ERROR_INVALID_ARGUMENT;
    }
    node->inputs.push_back(tensor->tensor);
  }
  for (uint32_t output_index = 0; output_index < description.output_count;
       ++output_index) {
    TensorStorage *tensor = find_tensor(description.outputs[output_index]);
    if (tensor == nullptr) {
      set_message(message, message_size, "node references an unknown output");
      return VA_QNN_ERROR_INVALID_ARGUMENT;
    }
    node->outputs.push_back(tensor->tensor);
  }

  auto require_arity = [&](uint32_t inputs, uint32_t outputs,
                           uint32_t parameters) -> bool {
    return description.input_count == inputs &&
           description.output_count == outputs &&
           description.parameter_count == parameters;
  };

  if (description.kind == VA_QNN_NODE_MAX_POOL_2D) {
    if (!require_arity(1, 1, 4)) {
      set_message(message, message_size, "invalid max-pool descriptor arity");
      return VA_QNN_ERROR_INVALID_ARGUMENT;
    }
    TensorStorage *input = find_tensor(description.inputs[0]);
    TensorStorage *output = find_tensor(description.outputs[0]);
    return add_max_pool(graph, description, index, *input, *output, message,
                        message_size);
  }

  if (description.kind == VA_QNN_NODE_REVERSE) {
    if (!require_arity(1, 1, 1)) {
      set_message(message, message_size, "invalid reverse descriptor arity");
      return VA_QNN_ERROR_INVALID_ARGUMENT;
    }
    TensorStorage *input = find_tensor(description.inputs[0]);
    TensorStorage *output = find_tensor(description.outputs[0]);
    return add_reverse(graph, description, index, *input, *output, message,
                       message_size);
  }

  if (description.kind == VA_QNN_NODE_REDUCE_PRODUCT) {
    if (!require_arity(1, 1, 1)) {
      set_message(message, message_size,
                  "invalid reduce-product descriptor arity");
      return VA_QNN_ERROR_INVALID_ARGUMENT;
    }
    TensorStorage *input = find_tensor(description.inputs[0]);
    TensorStorage *output = find_tensor(description.outputs[0]);
    return add_reduce_product(graph, description, index, *input, *output,
                              message, message_size);
  }

  auto append_u32_vector = [&](const char *name) -> uint64_t {
    if (description.parameter_count == 0 || description.parameters == nullptr)
      return VA_QNN_ERROR_INVALID_ARGUMENT;
    std::vector<uint32_t> values;
    values.reserve(description.parameter_count);
    for (uint32_t parameter_index = 0;
         parameter_index < description.parameter_count; ++parameter_index) {
      if (description.parameters[parameter_index] < 0)
        return VA_QNN_ERROR_INVALID_ARGUMENT;
      values.push_back(
          static_cast<uint32_t>(description.parameters[parameter_index]));
    }
    TensorStorage *parameter = nullptr;
    const uint64_t status = create_internal_tensor(
        graph,
        "virtio_accel_param_" + std::to_string(index) + "_" + name,
        {description.parameter_count}, QNN_TENSOR_TYPE_STATIC,
        QNN_DATATYPE_UINT_32, std::move(values), &parameter, message,
        message_size);
    if (status == QNN_SUCCESS)
      append_tensor_param(*node, name, *parameter);
    return status;
  };

  switch (description.kind) {
  case VA_QNN_NODE_RESHAPE:
    if (!require_arity(1, 1, 0))
      break;
    node->type = QNN_OP_RESHAPE;
    return commit_node(graph, std::move(node), message, message_size);
  case VA_QNN_NODE_MATMUL:
    if (!require_arity(2, 1, 0))
      break;
    node->type = QNN_OP_MAT_MUL;
    node->params.reserve(2);
    append_scalar_bool(*node, QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN0, 0);
    append_scalar_bool(*node, QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN1, 0);
    return commit_node(graph, std::move(node), message, message_size);
  case VA_QNN_NODE_ADD:
    if (!require_arity(2, 1, 0))
      break;
    node->type = QNN_OP_ELEMENT_WISE_ADD;
    return commit_node(graph, std::move(node), message, message_size);
  case VA_QNN_NODE_SUBTRACT:
    if (!require_arity(2, 1, 0))
      break;
    node->type = QNN_OP_ELEMENT_WISE_SUBTRACT;
    return commit_node(graph, std::move(node), message, message_size);
  case VA_QNN_NODE_MAXIMUM:
    if (!require_arity(2, 1, 0))
      break;
    node->type = QNN_OP_ELEMENT_WISE_MAXIMUM;
    return commit_node(graph, std::move(node), message, message_size);
  case VA_QNN_NODE_MINIMUM:
    if (!require_arity(2, 1, 0))
      break;
    node->type = QNN_OP_ELEMENT_WISE_MINIMUM;
    return commit_node(graph, std::move(node), message, message_size);
  case VA_QNN_NODE_MULTIPLY:
    if (!require_arity(2, 1, 0))
      break;
    node->type = QNN_OP_ELEMENT_WISE_MULTIPLY;
    return commit_node(graph, std::move(node), message, message_size);
  case VA_QNN_NODE_TRANSPOSE:
    if (description.input_count != 1 || description.output_count != 1 ||
        description.parameter_count == 0)
      break;
    node->type = QNN_OP_TRANSPOSE;
    if (const uint64_t status =
            append_u32_vector(QNN_OP_TRANSPOSE_PARAM_PERM);
        status != QNN_SUCCESS)
      return status;
    return commit_node(graph, std::move(node), message, message_size);
  case VA_QNN_NODE_CONCAT:
    if (description.input_count == 0 || description.output_count != 1 ||
        description.parameter_count != 1)
      break;
    node->type = QNN_OP_CONCAT;
    append_scalar_i32(*node, QNN_OP_CONCAT_PARAM_AXIS,
                      description.parameters[0]);
    return commit_node(graph, std::move(node), message, message_size);
  case VA_QNN_NODE_POWER:
    if (!require_arity(2, 1, 0))
      break;
    node->type = QNN_OP_ELEMENT_WISE_POWER;
    return commit_node(graph, std::move(node), message, message_size);
  case VA_QNN_NODE_ABS:
  case VA_QNN_NODE_CEIL:
  case VA_QNN_NODE_COS:
  case VA_QNN_NODE_EXP:
  case VA_QNN_NODE_FLOOR:
  case VA_QNN_NODE_LOG:
  case VA_QNN_NODE_NEGATE:
  case VA_QNN_NODE_RSQRT:
  case VA_QNN_NODE_SIN:
  case VA_QNN_NODE_LOGICAL_NOT:
    if (!require_arity(1, 1, 0))
      break;
    switch (description.kind) {
    case VA_QNN_NODE_ABS:
      node->type = QNN_OP_ELEMENT_WISE_ABS;
      break;
    case VA_QNN_NODE_CEIL:
      node->type = QNN_OP_ELEMENT_WISE_CEIL;
      break;
    case VA_QNN_NODE_COS:
      node->type = QNN_OP_ELEMENT_WISE_COS;
      break;
    case VA_QNN_NODE_EXP:
      node->type = QNN_OP_ELEMENT_WISE_EXP;
      break;
    case VA_QNN_NODE_FLOOR:
      node->type = QNN_OP_ELEMENT_WISE_FLOOR;
      break;
    case VA_QNN_NODE_LOG:
      node->type = QNN_OP_ELEMENT_WISE_LOG;
      break;
    case VA_QNN_NODE_NEGATE:
      node->type = QNN_OP_ELEMENT_WISE_NEG;
      break;
    case VA_QNN_NODE_RSQRT:
      node->type = QNN_OP_ELEMENT_WISE_RSQRT;
      break;
    case VA_QNN_NODE_SIN:
      node->type = QNN_OP_ELEMENT_WISE_SIN;
      break;
    default:
      node->type = QNN_OP_ELEMENT_WISE_NOT;
      break;
    }
    return commit_node(graph, std::move(node), message, message_size);
  case VA_QNN_NODE_RECIPROCAL:
    if (!require_arity(1, 1, 0))
      break;
    node->type = QNN_OP_ELEMENT_WISE_UNARY;
    append_scalar_i32(*node, QNN_OP_ELEMENT_WISE_UNARY_PARAM_OPERATION,
                      QNN_OP_ELEMENT_WISE_UNARY_OPERATION_RECIPROCAL);
    return commit_node(graph, std::move(node), message, message_size);
  case VA_QNN_NODE_SIGMOID:
  case VA_QNN_NODE_TANH:
    if (!require_arity(1, 1, 0))
      break;
    node->type = description.kind == VA_QNN_NODE_SIGMOID ? QNN_OP_SIGMOID
                                                         : QNN_OP_TANH;
    return commit_node(graph, std::move(node), message, message_size);
  case VA_QNN_NODE_CLAMP:
    if (!require_arity(1, 1, 2))
      break;
    node->type = QNN_OP_RELU_MIN_MAX;
    append_scalar_f32(*node, QNN_OP_RELU_MIN_MAX_PARAM_MIN_VALUE,
                      description.parameters[0]);
    append_scalar_f32(*node, QNN_OP_RELU_MIN_MAX_PARAM_MAX_VALUE,
                      description.parameters[1]);
    return commit_node(graph, std::move(node), message, message_size);
  case VA_QNN_NODE_EQUAL:
  case VA_QNN_NODE_GREATER:
  case VA_QNN_NODE_GREATER_EQUAL:
  case VA_QNN_NODE_LOGICAL_AND:
  case VA_QNN_NODE_LOGICAL_OR:
  case VA_QNN_NODE_LOGICAL_XOR:
    if (!require_arity(2, 1, 0))
      break;
    switch (description.kind) {
    case VA_QNN_NODE_EQUAL:
      node->type = QNN_OP_ELEMENT_WISE_EQUAL;
      break;
    case VA_QNN_NODE_GREATER:
      node->type = QNN_OP_ELEMENT_WISE_GREATER;
      break;
    case VA_QNN_NODE_GREATER_EQUAL:
      node->type = QNN_OP_ELEMENT_WISE_GREATER_EQUAL;
      break;
    case VA_QNN_NODE_LOGICAL_AND:
      node->type = QNN_OP_ELEMENT_WISE_AND;
      break;
    case VA_QNN_NODE_LOGICAL_OR:
      node->type = QNN_OP_ELEMENT_WISE_OR;
      break;
    default:
      node->type = QNN_OP_ELEMENT_WISE_XOR;
      break;
    }
    return commit_node(graph, std::move(node), message, message_size);
  case VA_QNN_NODE_SELECT:
    if (!require_arity(3, 1, 0))
      break;
    node->type = QNN_OP_ELEMENT_WISE_SELECT;
    return commit_node(graph, std::move(node), message, message_size);
  case VA_QNN_NODE_ARGMAX:
    if (!require_arity(1, 1, 1))
      break;
    node->type = QNN_OP_ARGMAX;
    append_scalar_i32(*node, QNN_OP_ARGMAX_PARAM_AXIS,
                      description.parameters[0]);
    append_scalar_bool(*node, QNN_OP_ARGMAX_PARAM_KEEP_DIMS, 0);
    return commit_node(graph, std::move(node), message, message_size);
  case VA_QNN_NODE_REDUCE_MAX:
  case VA_QNN_NODE_REDUCE_MIN:
  case VA_QNN_NODE_REDUCE_SUM: {
    if (!require_arity(1, 1, 1))
      break;
    const char *axes_name = nullptr;
    const char *keep_dims_name = nullptr;
    switch (description.kind) {
    case VA_QNN_NODE_REDUCE_MAX:
      node->type = QNN_OP_REDUCE_MAX;
      axes_name = QNN_OP_REDUCE_MAX_PARAM_AXES;
      keep_dims_name = QNN_OP_REDUCE_MAX_PARAM_KEEP_DIMS;
      break;
    case VA_QNN_NODE_REDUCE_MIN:
      node->type = QNN_OP_REDUCE_MIN;
      axes_name = QNN_OP_REDUCE_MIN_PARAM_AXES;
      keep_dims_name = QNN_OP_REDUCE_MIN_PARAM_KEEP_DIMS;
      break;
    default:
      node->type = QNN_OP_REDUCE_SUM;
      axes_name = QNN_OP_REDUCE_SUM_PARAM_AXES;
      keep_dims_name = QNN_OP_REDUCE_SUM_PARAM_KEEP_DIMS;
      break;
    }
    if (const uint64_t status = append_u32_vector(axes_name);
        status != QNN_SUCCESS)
      return status;
    append_scalar_bool(*node, keep_dims_name, 1);
    return commit_node(graph, std::move(node), message, message_size);
  }
  default:
    set_message(message, message_size, "unknown QNN node kind");
    return VA_QNN_ERROR_INVALID_ARGUMENT;
  }
  set_message(message, message_size, "invalid QNN node descriptor arity");
  return VA_QNN_ERROR_INVALID_ARGUMENT;
}

} // namespace

extern "C" uint64_t va_qnn_runtime_create(const char *library_path,
                                          VaQnnRuntime **output,
                                          VaQnnRuntimeInfo *info, char *message,
                                          size_t message_size) {
  if (output == nullptr || info == nullptr) {
    return VA_QNN_ERROR_INVALID_ARGUMENT;
  }
  *output = nullptr;
  *info = {};
  try {
    auto runtime = std::make_unique<VaQnnRuntime>();
    runtime->library = LoadLibraryExA(
        library_path == nullptr ? "QnnHtp.dll" : library_path, nullptr,
        library_path == nullptr ? 0 : LOAD_WITH_ALTERED_SEARCH_PATH);
    if (runtime->library == nullptr) {
      set_message(message, message_size, "QnnHtp.dll could not be loaded");
      return VA_QNN_ERROR_INCOMPATIBLE;
    }
    using GetProviders = Qnn_ErrorHandle_t (*)(
        const QnnInterface_t ***providerList, uint32_t *numProviders);
    auto get_providers = reinterpret_cast<GetProviders>(
        GetProcAddress(runtime->library, "QnnInterface_getProviders"));
    if (get_providers == nullptr) {
      FreeLibrary(runtime->library);
      runtime->library = nullptr;
      set_message(message, message_size,
                  "QnnHtp.dll exports no QnnInterface_getProviders");
      return VA_QNN_ERROR_INCOMPATIBLE;
    }
    const QnnInterface_t **providers = nullptr;
    uint32_t provider_count = 0;
    uint64_t status = get_providers(&providers, &provider_count);
    if (status != QNN_SUCCESS) {
      FreeLibrary(runtime->library);
      runtime->library = nullptr;
      set_message(message, message_size, "QnnInterface_getProviders failed");
      return status;
    }
    const QnnInterface_t *provider = select_provider(providers, provider_count);
    if (provider == nullptr) {
      FreeLibrary(runtime->library);
      runtime->library = nullptr;
      set_message(message, message_size,
                  "no compatible QNN HTP interface provider");
      return VA_QNN_ERROR_INCOMPATIBLE;
    }

    runtime->api = provider->QNN_INTERFACE_VER_NAME;
    if (!required_api_present(runtime->api)) {
      FreeLibrary(runtime->library);
      runtime->library = nullptr;
      set_message(message, message_size,
                  "QNN HTP provider lacks a required API function");
      return VA_QNN_ERROR_INCOMPATIBLE;
    }
    status = runtime->api.backendCreate(nullptr, nullptr, &runtime->backend);
    if (status != QNN_SUCCESS) {
      FreeLibrary(runtime->library);
      runtime->library = nullptr;
      set_message(message, message_size, "QNN HTP backend creation failed");
      return status;
    }
    status = runtime->api.deviceCreate(nullptr, nullptr, &runtime->device);
    if (status != QNN_SUCCESS) {
      runtime->api.backendFree(runtime->backend);
      runtime->backend = nullptr;
      FreeLibrary(runtime->library);
      runtime->library = nullptr;
      set_message(message, message_size, "QNN HTP device creation failed");
      return status;
    }

    Qnn_ApiVersion_t version = QNN_API_VERSION_INIT;
    status = runtime->api.backendGetApiVersion(&version);
    if (status != QNN_SUCCESS) {
      runtime->api.deviceFree(runtime->device);
      runtime->api.backendFree(runtime->backend);
      FreeLibrary(runtime->library);
      runtime->library = nullptr;
      set_message(message, message_size, "QNN HTP API version query failed");
      return status;
    }
    const char *build_id = nullptr;
    status = runtime->api.backendGetBuildId(&build_id);
    if (status != QNN_SUCCESS) {
      runtime->api.deviceFree(runtime->device);
      runtime->api.backendFree(runtime->backend);
      FreeLibrary(runtime->library);
      runtime->library = nullptr;
      set_message(message, message_size,
                  "QNN HTP build identifier query failed");
      return status;
    }

    info->backend_id = provider->backendId;
    info->core_major = version.coreApiVersion.major;
    info->core_minor = version.coreApiVersion.minor;
    info->core_patch = version.coreApiVersion.patch;
    info->backend_major = version.backendApiVersion.major;
    info->backend_minor = version.backendApiVersion.minor;
    info->backend_patch = version.backendApiVersion.patch;
    copy_text(info->provider_name, provider->providerName);
    copy_text(info->build_id, build_id);
    *output = runtime.release();
    return QNN_SUCCESS;
  } catch (const std::bad_alloc &) {
    set_message(message, message_size, "native QNN bridge allocation failed");
    return VA_QNN_ERROR_OUT_OF_MEMORY;
  } catch (...) {
    set_message(message, message_size, "unexpected native QNN bridge failure");
    return VA_QNN_ERROR_INTERNAL;
  }
}

extern "C" uint64_t va_qnn_runtime_free(VaQnnRuntime *runtime) {
  if (runtime == nullptr) {
    return VA_QNN_ERROR_INVALID_ARGUMENT;
  }
  if (runtime->in_flight.load(std::memory_order_acquire) != 0) {
    return VA_QNN_ERROR_BUSY;
  }
  uint64_t first_error = QNN_SUCCESS;
  if (runtime->device != nullptr) {
    first_error = runtime->api.deviceFree(runtime->device);
  }
  if (runtime->backend != nullptr) {
    const uint64_t status = runtime->api.backendFree(runtime->backend);
    if (first_error == QNN_SUCCESS) {
      first_error = status;
    }
  }
  if (runtime->library != nullptr) {
    FreeLibrary(runtime->library);
    runtime->library = nullptr;
  }
  delete runtime;
  return first_error;
}

extern "C" uint64_t
va_qnn_graph_create(VaQnnRuntime *runtime,
                    const VaQnnTensorDesc *tensor_descriptions,
                    uint32_t tensor_count,
                    const VaQnnNodeDesc *node_descriptions, uint32_t node_count,
                    uint32_t precision, VaQnnGraph **output, char *message,
                    size_t message_size) {
  if (runtime == nullptr || tensor_descriptions == nullptr ||
      tensor_count == 0 || node_descriptions == nullptr || node_count == 0 ||
      output == nullptr) {
    return VA_QNN_ERROR_INVALID_ARGUMENT;
  }
  *output = nullptr;
  try {
    auto graph = std::make_unique<VaQnnGraph>();
    graph->runtime = runtime;
    uint64_t status = runtime->api.contextCreate(
        runtime->backend, runtime->device, nullptr, &graph->context);
    if (status != QNN_SUCCESS) {
      set_message(message, message_size, "QNN context creation failed");
      return status;
    }
    QnnHtpGraph_CustomConfig_t htp_config = QNN_HTP_GRAPH_CUSTOM_CONFIG_INIT;
    QnnGraph_Config_t graph_config = QNN_GRAPH_CONFIG_INIT;
    const QnnGraph_Config_t *graph_configs[] = {&graph_config, nullptr};
    const QnnGraph_Config_t **selected_configs = nullptr;
    if (precision == VA_QNN_PRECISION_F16 ||
        precision == VA_QNN_PRECISION_F32) {
      htp_config.option = QNN_HTP_GRAPH_CONFIG_OPTION_PRECISION;
      htp_config.precision = precision == VA_QNN_PRECISION_F32
                                 ? QNN_PRECISION_FLOAT32
                                 : QNN_PRECISION_FLOAT16;
      graph_config.option = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
      graph_config.customConfig = &htp_config;
      selected_configs = graph_configs;
    } else if (precision != VA_QNN_PRECISION_DEFAULT) {
      runtime->api.contextFree(graph->context, nullptr);
      graph->context = nullptr;
      set_message(message, message_size, "invalid QNN graph precision");
      return VA_QNN_ERROR_INVALID_ARGUMENT;
    }
    status = runtime->api.graphCreate(graph->context, "virtio_accel_graph",
                                      selected_configs, &graph->graph);
    if (status != QNN_SUCCESS) {
      runtime->api.contextFree(graph->context, nullptr);
      graph->context = nullptr;
      set_message(message, message_size, "QNN graph creation failed");
      return status;
    }

    size_t input_count = 0;
    size_t output_count = 0;
    for (uint32_t index = 0; index < tensor_count; ++index) {
      const VaQnnTensorDesc &description = tensor_descriptions[index];
      switch (description.role) {
      case VA_QNN_TENSOR_NATIVE:
        if (description.io_index != UINT32_MAX) {
          runtime->api.contextFree(graph->context, nullptr);
          graph->context = nullptr;
          set_message(message, message_size,
                      "native QNN tensor has a model I/O index");
          return VA_QNN_ERROR_INVALID_ARGUMENT;
        }
        break;
      case VA_QNN_TENSOR_INPUT:
        ++input_count;
        break;
      case VA_QNN_TENSOR_OUTPUT:
        ++output_count;
        break;
      case VA_QNN_TENSOR_STATIC:
        if (description.io_index != UINT32_MAX) {
          runtime->api.contextFree(graph->context, nullptr);
          graph->context = nullptr;
          set_message(message, message_size,
                      "static QNN tensor has a model I/O index");
          return VA_QNN_ERROR_INVALID_ARGUMENT;
        }
        break;
      default:
        runtime->api.contextFree(graph->context, nullptr);
        graph->context = nullptr;
        set_message(message, message_size, "invalid QNN tensor role");
        return VA_QNN_ERROR_INVALID_ARGUMENT;
      }
    }
    const size_t unbound = std::numeric_limits<size_t>::max();
    graph->inputs.assign(input_count, unbound);
    graph->outputs.assign(output_count, unbound);

    size_t retained_tensor_count = tensor_count;
    for (uint32_t index = 0; index < node_count; ++index) {
      const VaQnnNodeDesc &description = node_descriptions[index];
      if (description.input_count == 0 || description.input_count > 256 ||
          description.output_count == 0 || description.output_count > 256 ||
          description.parameter_count > 256 || description.inputs == nullptr ||
          description.outputs == nullptr ||
          (description.parameter_count != 0 &&
           description.parameters == nullptr)) {
        runtime->api.contextFree(graph->context, nullptr);
        graph->context = nullptr;
        set_message(message, message_size, "invalid QNN node descriptor slices");
        return VA_QNN_ERROR_INVALID_ARGUMENT;
      }
      if (description.kind == VA_QNN_NODE_TRANSPOSE ||
          description.kind == VA_QNN_NODE_REVERSE ||
          description.kind == VA_QNN_NODE_REDUCE_MAX ||
          description.kind == VA_QNN_NODE_REDUCE_MIN ||
          description.kind == VA_QNN_NODE_REDUCE_SUM) {
        if (retained_tensor_count == std::numeric_limits<size_t>::max()) {
          runtime->api.contextFree(graph->context, nullptr);
          graph->context = nullptr;
          return VA_QNN_ERROR_OUT_OF_MEMORY;
        }
        ++retained_tensor_count;
      }
      if (description.kind == VA_QNN_NODE_REDUCE_PRODUCT) {
        const VaQnnTensorDesc *input = nullptr;
        for (uint32_t tensor_index = 0; tensor_index < tensor_count;
             ++tensor_index) {
          if (tensor_descriptions[tensor_index].value ==
              description.inputs[0]) {
            input = &tensor_descriptions[tensor_index];
            break;
          }
        }
        if (input == nullptr || description.parameter_count != 1 ||
            description.parameters[0] < 0 ||
            static_cast<uint32_t>(description.parameters[0]) >= input->rank ||
            input->dimensions == nullptr) {
          runtime->api.contextFree(graph->context, nullptr);
          graph->context = nullptr;
          set_message(message, message_size,
                      "invalid reduce-product reservation metadata");
          return VA_QNN_ERROR_INVALID_ARGUMENT;
        }
        const size_t extent = input->dimensions[description.parameters[0]];
        if (extent == 0 ||
            extent > std::numeric_limits<size_t>::max() / 3) {
          runtime->api.contextFree(graph->context, nullptr);
          graph->context = nullptr;
          return VA_QNN_ERROR_OUT_OF_MEMORY;
        }
        const size_t extra = extent * 3 - 2;
        if (retained_tensor_count >
            std::numeric_limits<size_t>::max() - extra) {
          runtime->api.contextFree(graph->context, nullptr);
          graph->context = nullptr;
          return VA_QNN_ERROR_OUT_OF_MEMORY;
        }
        retained_tensor_count += extra;
      }
      if (description.kind != VA_QNN_NODE_MAX_POOL_2D)
        continue;
      if (description.parameter_count != 4 ||
          description.parameters[0] <= 0 || description.parameters[1] <= 0) {
        runtime->api.contextFree(graph->context, nullptr);
        graph->context = nullptr;
        set_message(message, message_size, "invalid max-pool parameters");
        return VA_QNN_ERROR_INVALID_ARGUMENT;
      }
      const size_t rows = static_cast<size_t>(description.parameters[0]);
      const size_t columns = static_cast<size_t>(description.parameters[1]);
      if (rows > 256 || columns > 256 ||
          rows > std::numeric_limits<size_t>::max() / columns) {
        runtime->api.contextFree(graph->context, nullptr);
        graph->context = nullptr;
        set_message(message, message_size,
                    "max-pool kernel exceeds the native resource limit");
        return VA_QNN_ERROR_INVALID_ARGUMENT;
      }
      const size_t windows = rows * columns;
      const size_t maximums = windows > 1 ? windows - 2 : 0;
      const size_t extra = rows + columns + rows + windows + maximums;
      if (retained_tensor_count > std::numeric_limits<size_t>::max() - extra) {
        runtime->api.contextFree(graph->context, nullptr);
        graph->context = nullptr;
        return VA_QNN_ERROR_OUT_OF_MEMORY;
      }
      retained_tensor_count += extra;
    }
    graph->tensors.reserve(retained_tensor_count);
    graph->by_value.reserve(tensor_count);
    for (uint32_t index = 0; index < tensor_count; ++index) {
      const VaQnnTensorDesc &description = tensor_descriptions[index];
      const Qnn_DataType_t data_type = tensor_data_type(description);
      const bool is_static = description.role == VA_QNN_TENSOR_STATIC;
      const size_t expected_bytes = tensor_storage_bytes(description);
      if ((description.rank != 0 && description.dimensions == nullptr) ||
          data_type == QNN_DATATYPE_UNDEFINED ||
          expected_bytes == 0 ||
          description.quantized > 1 ||
          (description.quantized != 0 && !(description.scale > 0.0f)) ||
          (is_static &&
           (description.constant_data == nullptr ||
            description.constant_size == 0 ||
            description.constant_size > std::numeric_limits<uint32_t>::max() ||
            description.constant_size != expected_bytes)) ||
          (!is_static &&
           (description.constant_data != nullptr ||
            description.constant_size != 0)) ||
          graph->by_value.find(description.value) != graph->by_value.end()) {
        runtime->api.contextFree(graph->context, nullptr);
        set_message(message, message_size,
                    "invalid or duplicate QNN tensor description");
        return VA_QNN_ERROR_INVALID_ARGUMENT;
      }
      TensorStorage storage;
      storage.value = description.value;
      storage.name = "virtio_accel_tensor_" + std::to_string(description.value);
      if (description.rank != 0) {
        storage.dimensions.assign(description.dimensions,
                                  description.dimensions + description.rank);
      }
      if (is_static) {
        storage.constant_data.assign(
            description.constant_data,
            description.constant_data + description.constant_size);
      }
      storage.tensor.v2.name = storage.name.c_str();
      storage.tensor.v2.type = tensor_type(description.role);
      storage.tensor.v2.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
      storage.tensor.v2.dataType = data_type;
      if (description.quantized != 0) {
        storage.tensor.v2.quantizeParams.encodingDefinition =
            QNN_DEFINITION_DEFINED;
        storage.tensor.v2.quantizeParams.quantizationEncoding =
            QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
        storage.tensor.v2.quantizeParams.scaleOffsetEncoding.scale =
            description.scale;
        storage.tensor.v2.quantizeParams.scaleOffsetEncoding.offset =
            description.offset;
      }
      storage.tensor.v2.rank = description.rank;
      storage.tensor.v2.dimensions =
          storage.dimensions.empty() ? nullptr : storage.dimensions.data();
      storage.tensor.v2.memType = QNN_TENSORMEMTYPE_RAW;
      if (is_static) {
        storage.tensor.v2.clientBuf.data = storage.constant_data.data();
        storage.tensor.v2.clientBuf.dataSize =
            static_cast<uint32_t>(storage.constant_data.size());
      }
      graph->tensors.push_back(std::move(storage));
      TensorStorage &retained = graph->tensors.back();
      retained.tensor.v2.name = retained.name.c_str();
      retained.tensor.v2.dimensions =
          retained.dimensions.empty() ? nullptr : retained.dimensions.data();
      if (is_static) {
        retained.tensor.v2.clientBuf.data = retained.constant_data.data();
      }
      status =
          runtime->api.tensorCreateGraphTensor(graph->graph, &retained.tensor);
      if (status != QNN_SUCCESS) {
        runtime->api.contextFree(graph->context, nullptr);
        set_message(message, message_size, "QNN graph tensor creation failed");
        return status;
      }
      graph->by_value.emplace(description.value, index);
      if (description.role == VA_QNN_TENSOR_INPUT) {
        if (description.io_index >= graph->inputs.size() ||
            graph->inputs[description.io_index] != unbound) {
          runtime->api.contextFree(graph->context, nullptr);
          graph->context = nullptr;
          set_message(message, message_size,
                      "duplicate or invalid QNN input index");
          return VA_QNN_ERROR_INVALID_ARGUMENT;
        }
        graph->inputs[description.io_index] = index;
      } else if (description.role == VA_QNN_TENSOR_OUTPUT) {
        if (description.io_index >= graph->outputs.size() ||
            graph->outputs[description.io_index] != unbound) {
          runtime->api.contextFree(graph->context, nullptr);
          graph->context = nullptr;
          set_message(message, message_size,
                      "duplicate or invalid QNN output index");
          return VA_QNN_ERROR_INVALID_ARGUMENT;
        }
        graph->outputs[description.io_index] = index;
      }
    }

    graph->nodes.reserve(node_count);
    for (uint32_t index = 0; index < node_count; ++index) {
      status = add_node(*graph, node_descriptions[index], index, message,
                        message_size);
      if (status != QNN_SUCCESS) {
        runtime->api.contextFree(graph->context, nullptr);
        graph->context = nullptr;
        return status;
      }
    }
    status = runtime->api.graphFinalize(graph->graph, nullptr, nullptr);
    if (status != QNN_SUCCESS) {
      runtime->api.contextFree(graph->context, nullptr);
      graph->context = nullptr;
      set_message(message, message_size, "QNN HTP graph finalization failed");
      return status;
    }
    *output = graph.release();
    return QNN_SUCCESS;
  } catch (const std::bad_alloc &) {
    set_message(message, message_size, "native QNN graph allocation failed");
    return VA_QNN_ERROR_OUT_OF_MEMORY;
  } catch (...) {
    set_message(message, message_size, "unexpected native QNN graph failure");
    return VA_QNN_ERROR_INTERNAL;
  }
}

extern "C" uint64_t va_qnn_graph_free(VaQnnGraph *graph) {
  if (graph == nullptr) {
    return VA_QNN_ERROR_INVALID_ARGUMENT;
  }
  if (graph->runtime->in_flight.load(std::memory_order_acquire) != 0) {
    return VA_QNN_ERROR_BUSY;
  }
  const uint64_t status =
      graph->runtime->api.contextFree(graph->context, nullptr);
  delete graph;
  return status;
}

extern "C" uint64_t va_qnn_graph_execute_async(
    VaQnnGraph *graph, const VaQnnBinding *input_bindings, uint32_t input_count,
    const VaQnnBinding *output_bindings, uint32_t output_count,
    VaQnnEvent **output, char *message, size_t message_size) {
  if (graph == nullptr || output == nullptr ||
      input_count != graph->inputs.size() ||
      output_count != graph->outputs.size() ||
      (input_count != 0 && input_bindings == nullptr) ||
      (output_count != 0 && output_bindings == nullptr)) {
    return VA_QNN_ERROR_INVALID_ARGUMENT;
  }
  *output = nullptr;
  uint32_t expected = 0;
  if (!graph->runtime->in_flight.compare_exchange_strong(
          expected, 1, std::memory_order_acq_rel, std::memory_order_acquire)) {
    return VA_QNN_ERROR_BUSY;
  }
  try {
    auto event = std::make_unique<VaQnnEvent>();
    event->runtime = graph->runtime;
    event->inputs.reserve(input_count);
    event->outputs.reserve(output_count);
    for (uint32_t index = 0; index < input_count; ++index) {
      if (input_bindings[index].data == nullptr ||
          input_bindings[index].size > std::numeric_limits<uint32_t>::max()) {
        graph->runtime->in_flight.store(0, std::memory_order_release);
        return VA_QNN_ERROR_INVALID_ARGUMENT;
      }
      Qnn_Tensor_t tensor = graph->tensors[graph->inputs[index]].tensor;
      tensor.v2.clientBuf.data = input_bindings[index].data;
      tensor.v2.clientBuf.dataSize =
          static_cast<uint32_t>(input_bindings[index].size);
      event->inputs.push_back(tensor);
    }
    for (uint32_t index = 0; index < output_count; ++index) {
      if (output_bindings[index].data == nullptr ||
          output_bindings[index].size > std::numeric_limits<uint32_t>::max()) {
        graph->runtime->in_flight.store(0, std::memory_order_release);
        return VA_QNN_ERROR_INVALID_ARGUMENT;
      }
      Qnn_Tensor_t tensor = graph->tensors[graph->outputs[index]].tensor;
      tensor.v2.clientBuf.data = output_bindings[index].data;
      tensor.v2.clientBuf.dataSize =
          static_cast<uint32_t>(output_bindings[index].size);
      event->outputs.push_back(tensor);
    }

    VaQnnEvent *accepted = event.get();
    accepted->worker = std::thread([graph, accepted]() {
      const uint64_t status = graph->runtime->api.graphExecute(
          graph->graph, accepted->inputs.data(),
          static_cast<uint32_t>(accepted->inputs.size()),
          accepted->outputs.data(),
          static_cast<uint32_t>(accepted->outputs.size()), nullptr, nullptr);
      accepted->error.store(status, std::memory_order_relaxed);
      accepted->runtime->in_flight.store(0, std::memory_order_release);
      accepted->state.store(status == QNN_SUCCESS ? VA_QNN_EVENT_COMPLETE
                                                  : VA_QNN_EVENT_FAILED,
                            std::memory_order_release);
    });
    *output = event.release();
    return QNN_SUCCESS;
  } catch (const std::bad_alloc &) {
    graph->runtime->in_flight.store(0, std::memory_order_release);
    return VA_QNN_ERROR_OUT_OF_MEMORY;
  } catch (...) {
    graph->runtime->in_flight.store(0, std::memory_order_release);
    return VA_QNN_ERROR_INTERNAL;
  }
}

extern "C" uint32_t va_qnn_event_poll(const VaQnnEvent *event,
                                      uint64_t *qnn_error) {
  if (event == nullptr) {
    if (qnn_error != nullptr) {
      *qnn_error = VA_QNN_ERROR_INVALID_ARGUMENT;
    }
    return VA_QNN_EVENT_FAILED;
  }
  const uint32_t state = event->state.load(std::memory_order_acquire);
  if (qnn_error != nullptr) {
    *qnn_error = event->error.load(std::memory_order_relaxed);
  }
  return state;
}

extern "C" uint64_t va_qnn_event_free(VaQnnEvent *event) {
  if (event == nullptr) {
    return VA_QNN_ERROR_INVALID_ARGUMENT;
  }
  if (event->state.load(std::memory_order_acquire) == VA_QNN_EVENT_PENDING) {
    return VA_QNN_ERROR_BUSY;
  }
  if (event->worker.joinable()) {
    event->worker.join();
  }
  delete event;
  return QNN_SUCCESS;
}
