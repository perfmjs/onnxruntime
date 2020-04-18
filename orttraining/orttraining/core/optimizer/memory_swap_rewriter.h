// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/rewrite_rule.h"

namespace onnxruntime {

/**
@Class MemorySwapRewriter

Rewrite rule for memory swap.
*/
class MemorySwapRewriter : public RewriteRule {
 public:
  MemorySwapRewriter(int min_topo_distance) noexcept
      : RewriteRule("MemorySwap"),
        min_topo_distance_(min_topo_distance) {
  }

  std::vector<std::string> TargetOpTypes() const noexcept override;

 private:
  bool SatisfyCondition(const Graph& graph, const Node& node, const logging::Logger& logger) const override;

  Status Apply(Graph& graph, Node& node, RewriteRuleEffect& rule_effect, const logging::Logger& logger) const override;
  bool AddSwap(Graph& graph, Node& curr_node) const;

  int min_topo_distance_;

  static const Graph* last_graph_;
  static std::unordered_map<NodeIndex, int> topo_indices_;
};

}  // namespace onnxruntime