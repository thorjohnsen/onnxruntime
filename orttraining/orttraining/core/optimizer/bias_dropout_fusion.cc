// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/initializer.h"
#include "orttraining/core/optimizer/bias_dropout_fusion.h"
#include "core/graph/graph_utils.h"
#include <deque>

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::common;
namespace onnxruntime {

Status BiasDropoutFusion::ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const {
  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();

  for (auto node_index : node_topology_list) {
    auto* node_ptr = graph.GetNode(node_index);
    if (nullptr == node_ptr)
      continue;  // node was removed

    auto& node = *node_ptr;

    ORT_RETURN_IF_ERROR(Recurse(node, modified, graph_level, logger));

    if (!graph_utils::IsSupportedOptypeVersionAndDomain(node, "Add", {7}) ||
        !graph_utils::IsSupportedProvider(node, GetCompatibleExecutionProviders()) ||
        node.GetOutputEdgesCount() != 1) {
      continue;
    }

    std::vector<NodeArg*> dropout_input;
    const TensorShapeProto* input1_shape = node.MutableInputDefs()[0]->Shape();
    const TensorShapeProto* input2_shape = node.MutableInputDefs()[1]->Shape();

    if (input1_shape == nullptr ||
        input2_shape == nullptr ||
        input1_shape->dim_size() < 1 ||
        input2_shape->dim_size() < 1) {
      continue;
    }

    int last_dim_shape1 = input1_shape->dim_size() - 1;
    int last_dim_shape2 = input2_shape->dim_size() - 1;
    if (!utils::HasDimValue(input1_shape->dim(last_dim_shape1)) ||
        !utils::HasDimValue(input2_shape->dim(last_dim_shape2)) ||
        input1_shape->dim(last_dim_shape1).dim_value() != input2_shape->dim(last_dim_shape2).dim_value()) {
      continue;
    }

    if (input1_shape->dim_size() == 1) {     
      dropout_input.push_back(node.MutableInputDefs()[1]);  // droput input
      dropout_input.push_back(node.MutableInputDefs()[0]);  // bias
    } else if (input2_shape->dim_size() == 1) {  
      dropout_input.push_back(node.MutableInputDefs()[0]);  // dropout input
      dropout_input.push_back(node.MutableInputDefs()[1]);  // bias
    } else {
      continue;
    }

    auto next_node_itr = node.OutputNodesBegin();
    if (next_node_itr == node.OutputNodesEnd()) {
      continue;
    }

    const Node& next_node = (*next_node_itr);
    if (!(graph_utils::IsSupportedOptypeVersionAndDomain(next_node, "Dropout", {12}, kMSDomain) ||
          graph_utils::IsSupportedOptypeVersionAndDomain(next_node, "TrainableDropout", {9}, kOnnxDomain)) ||
        next_node.GetExecutionProviderType() != node.GetExecutionProviderType()) {
      continue;
    }
   
    if (!graph.GetNodeOutputsInGraphOutputs(node).empty()) {
      continue;
    }

    Node& add_node = node;
    Node& dropout_node = const_cast<Node&>(next_node);
    bool is_onnx_dropout = next_node.OpType().compare("Dropout") == 0;

    if (dropout_node.InputDefs().size() > 1) {
      dropout_input.push_back(dropout_node.MutableInputDefs()[1]);  // ratio
    }
    if (is_onnx_dropout && dropout_node.InputDefs().size() > 2) {
      dropout_input.push_back(dropout_node.MutableInputDefs()[2]);  // training_mode
    }

    const std::string op_type = "BiasDropout";
    Node& dropout_add_fusion_node = graph.AddNode(graph.GenerateNodeName(op_type),
                                                  op_type,
                                                  "fused Add and Dropout",
                                                  dropout_input,
                                                  {},
                                                  {},
                                                  kMSDomain);

    // Get attribute "seed" from "Dropout" node if available. 
    NodeAttributes dropout_attrs = dropout_node.GetAttributes();
    NodeAttributes::const_iterator seed = dropout_attrs.find("seed");
    if (seed != dropout_attrs.end()) {
      dropout_add_fusion_node.AddAttribute("seed", seed->second);
    }

    // Assign provider to this new node. Provider should be same as the provider for old node.
    dropout_add_fusion_node.SetExecutionProviderType(dropout_node.GetExecutionProviderType());

    // move output definitions and edges from dropout_node to dropout_add_fusion_node
    // delete add_node and dropout_node.
    graph_utils::FinalizeNodeFusion(graph, {add_node, dropout_node}, dropout_add_fusion_node);
   
    modified = true;
  }

  return Status::OK();
}
}  // namespace onnxruntime
