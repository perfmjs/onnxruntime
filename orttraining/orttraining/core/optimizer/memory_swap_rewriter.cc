// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/logging/logging.h"
#include "core/graph/op.h"
#include "core/optimizer/rewrite_rule.h"
#include "orttraining/core/optimizer/memory_swap_rewriter.h"
#include "core/graph/graph_utils.h"

namespace onnxruntime {

bool MemorySwapRewriter::AddSwapInOut(Graph& graph, Node& curr_node) const {
  ORT_UNUSED_PARAMETER(graph);
  ORT_UNUSED_PARAMETER(curr_node);
#if 0
  ONNX_NAMESPACE::TypeProto bool_tensor;
  bool_tensor.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_BOOL);

  if (curr_node.GetOutputEdgesCount() < 1) {
    return false;
  }

  std::vector<GraphEdgeHelper> output_edges = GetNodeOutputEdges(curr_node);

  auto curr_node_output_defs = curr_node.OutputDefs();
  auto curr_node_output_def_name = curr_node_output_defs[0]->Name();
  auto* curr_node_output_arg = graph.GetNodeArg(curr_node_output_def_name);

  std::string encode_node_name = graph.GenerateNodeName(MEMORY_SWAP_OUT_NODE_NAME_BASE);
  for (int i = 0; i < curr_node_output_arg->Shape()->dim_size(); i++) {
    bool_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(curr_node_output_arg->Shape()->dim(i).dim_value());
  }
  auto& encode_output_def_compressed_arg = graph.GetOrCreateNodeArg(encode_node_name, &bool_tensor);
  auto& encode_output_def_uncompressed_arg = graph.GetOrCreateNodeArg(encode_node_name + "_identity", curr_node_output_arg->TypeAsProto());
  auto& encode = graph.AddNode(encode_node_name, compression_type + "Encoder", "Encode", {curr_node_output_arg}, {&encode_output_def_uncompressed_arg, &encode_output_def_compressed_arg}, {}, kMSDomain);

  std::string decode_arg_name = graph.GenerateNodeName(MEMORY_SWAP_IN_NODE_NAME_BASE);
  auto& decode_output_def_uncompressed_arg = graph.GetOrCreateNodeArg(decode_arg_name, curr_node_output_arg->TypeAsProto());
  auto& decode_output_def_dummy_arg = graph.GetOrCreateNodeArg(decode_arg_name + "_late_dec", curr_node_output_arg->TypeAsProto());
  auto& decode = graph.AddNode(decode_arg_name, compression_type + "Decoder", "Decode", {&decode_output_def_dummy_arg, &encode_output_def_compressed_arg}, {&decode_output_def_uncompressed_arg}, {}, kMSDomain);

  bool early_encoding = false;
  bool late_decoding = false;
  for (auto& output_edge : output_edges) {
    Node* node_dst = graph.GetNode(output_edge.dst_node);
    if (node_dst->Description() == "Backward pass" && (node_dst->OpType() == "ReluGrad")) {
      graph.AddEdge(output_edge.src_node, encode.Index(), output_edge.src_arg_index, 0);
      graph.AddEdge(encode.Index(), decode.Index(), 1, 1);
      graph.AddEdge(decode.Index(), output_edge.dst_node, 0, output_edge.dst_arg_index);
      std::vector<GraphEdgeHelper> input_edges_dst = GetNodeInputEdges(*node_dst);
      size_t i = 0;
      while (!late_decoding && i < input_edges_dst.size()) {
        if (graph.GetNode(input_edges_dst[i].src_node)->OpType() != curr_node.OpType()) {
          graph.AddEdge(input_edges_dst[i].src_node, decode.Index(), input_edges_dst[i].src_arg_index, 0);
          late_decoding = true;
        }
        i++;
      }
    } else if (!early_encoding) {
      graph.AddEdge(encode.Index(), output_edge.dst_node, 0, output_edge.dst_arg_index);
      early_encoding = true;
    }
  }
#endif
  return true;
}

std::vector<std::string> MemorySwapRewriter::TargetOpTypes() const noexcept {
  return {"MatMul"};  // only rewrite MatMul for now
}

Status MemorySwapRewriter::Apply(Graph& graph, Node& node, RewriteRuleEffect& rule_effect, const logging::Logger& /*logger*/) const {
  if (AddSwapInOut(graph, node)) {
    rule_effect = RewriteRuleEffect::kModifiedRestOfGraph;
  }
  return Status::OK();
}

bool MemorySwapRewriter::SatisfyCondition(const Graph& graph, const Node& node, const logging::Logger& /*logger*/) const {
  return false;
}

}  // namespace onnxruntime
