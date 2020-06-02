// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/initializer.h"
#include "core/optimizer/matmul_float_fusion.h"
#include "core/graph/graph_utils.h"
#include <deque>

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::common;
namespace onnxruntime {

Status MatMulFloatFusion::ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const {
  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();

  for (auto node_index : node_topology_list) {
    auto* node_ptr = graph.GetNode(node_index);
    if (nullptr == node_ptr)
      continue;  // node was removed

    auto& node = *node_ptr;

    ORT_RETURN_IF_ERROR(Recurse(node, modified, graph_level, logger));

    if (!graph_utils::IsSupportedOptypeVersionAndDomain(node, "MatMulInteger", {10}) ||
        !graph_utils::IsSupportedProvider(node, GetCompatibleExecutionProviders()) ||
        node.GetOutputEdgesCount() != 1) {
      continue;
    }

    if (!graph.GetNodeOutputsInGraphOutputs(node).empty()) {
      continue;
    }

    auto cast_node_itr = node.OutputNodesBegin();
    if (cast_node_itr == node.OutputNodesEnd()) {
      continue;
    }

    // bugbug: check to type
    Node& cast_node = *graph.GetNode(cast_node_itr->Index());
    if (!graph_utils::IsSupportedOptypeVersionAndDomain(cast_node, "Cast", {9}) ||
        cast_node.GetExecutionProviderType() != node.GetExecutionProviderType() ||
        cast_node.GetOutputEdgesCount() != 1) {
      continue;
    }

    if (!graph.GetNodeOutputsInGraphOutputs(cast_node).empty()) {
      continue;
    }

    auto mul_node_itr = cast_node.OutputNodesBegin();
    if (mul_node_itr == cast_node.OutputNodesEnd()) {
      continue;
    }

    Node& mul_node = *graph.GetNode(mul_node_itr->Index());
    if (!graph_utils::IsSupportedOptypeVersionAndDomain(mul_node, "Mul", {7}) ||
        mul_node.GetExecutionProviderType() != node.GetExecutionProviderType()) {
      continue;
    }

    Node* mul_other_input = nullptr;
    for (auto it = mul_node.InputNodesBegin(); it != mul_node.InputNodesEnd(); ++it) {
      const auto& input_node = *it;
      if (&input_node == &cast_node) {
        continue;
      } else if (mul_other_input == nullptr) {
        mul_other_input = graph.GetNode(it->Index());
      } else {
        mul_other_input = nullptr;
        break;
      }
    }

    if (mul_other_input == nullptr) {
      continue;
    }

printf("!!! can fuse\n");

    graph_utils::RemoveNodeOutputEdges(graph, *mul_other_input);

    auto fused_node_inputs = node.MutableInputDefs();
    fused_node_inputs.push_back(mul_other_input->MutableOutputDefs()[0]);

    Node& fused_node = graph.AddNode(graph.GenerateNodeName("MegaFusion"),
                                     "MegaFusion",
                                     "mega fusion",
                                     fused_node_inputs,
                                     mul_node.MutableOutputDefs(),
                                     nullptr,
                                     kMSDomain);

    // Assign provider to this new node. Provider should be same as the provider for old node.
    fused_node.SetExecutionProviderType(node.GetExecutionProviderType());

    graph_utils::FinalizeNodeFusion(graph, {node, cast_node, mul_node}, fused_node);

    modified = true;
  }

  return Status::OK();
}
}  // namespace onnxruntime
