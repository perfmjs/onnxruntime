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
  MemorySwapRewriter() noexcept : RewriteRule("MemorySwap") {}

  static constexpr const char* MEMORY_SWAP_OUT_NODE_NAME_BASE = "memswap_out_";
  static constexpr const char* MEMORY_SWAP_IN_NODE_NAME_BASE = "memswap_int_";

  std::vector<std::string> TargetOpTypes() const noexcept override;

 private:
  bool SatisfyCondition(const Graph& graph, const Node& node, const logging::Logger& logger) const override;

  Status Apply(Graph& graph, Node& node, RewriteRuleEffect& rule_effect, const logging::Logger& logger) const override;
  bool AddSwapInOut(Graph& graph, Node& curr_node) const;
};

}