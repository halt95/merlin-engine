// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
// torch.ops._C registrations for the vendored persistent top-k, in our own
// STABLE_TORCH_LIBRARY_FRAGMENT(_C, ...) so upstream torch_bindings.cpp needs
// no Merlin lines (the same pattern marlin_moe_wna16/ops.cu uses for _moe_C).
//
// The merlin_persistent_topk* names are what Python calls (qsa_indexer.py).
// merlin_persistent_topk binds merlin::topk::persistent_topk (the vendored
// topk.cu); upstream's _C::persistent_topk stays bound, in upstream
// torch_bindings.cpp, to upstream's own csrc/topk.cu (separate entry points,
// see csrc/merlin/merlin.cmake). The three persistent_topk_exact_counters /
// _slots / toggle_state registrations are also kept under their unprefixed
// names, as legacy aliases for out-of-tree callers.
#include "ops.h"
#include "merlin/topk/merlin_topk_ops.h"

#include <torch/csrc/stable/library.h>

STABLE_TORCH_LIBRARY_FRAGMENT(_C, ops) {
  // Same schema as upstream's persistent_topk.
  ops.def(
      "merlin_persistent_topk(Tensor logits, Tensor lengths, Tensor! output, "
      "Tensor workspace, int k, int max_seq_len) -> ()");
  // Read (and optionally reset) the medium-path exact-restart counters.
  ops.def(
      "merlin_persistent_topk_exact_counters(Tensor! out, int reset, int sync) "
      "-> ()");
  ops.def("merlin_persistent_topk_exact_counters_slots() -> int");
  // The switches as this process cached them (topk.cu for the layout).
  ops.def("merlin_persistent_topk_toggle_state() -> int");

  // Legacy aliases for out-of-tree callers.
  // Read (and optionally reset) the medium-path exact-restart counters.
  ops.def(
      "persistent_topk_exact_counters(Tensor! out, int reset, int sync) -> ()");
  ops.def("persistent_topk_exact_counters_slots() -> int");
  // The switches as this process cached them (topk.cu for the layout).
  ops.def("persistent_topk_toggle_state() -> int");
}

STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, ops) {
  ops.impl("merlin_persistent_topk",
           TORCH_BOX(&merlin::topk::persistent_topk));
  ops.impl("merlin_persistent_topk_exact_counters",
           TORCH_BOX(&persistent_topk_exact_counters));
  ops.impl("persistent_topk_exact_counters",
           TORCH_BOX(&persistent_topk_exact_counters));
}

STABLE_TORCH_LIBRARY_IMPL(_C, CompositeExplicitAutograd, ops) {
  // Tensor-free: the counter layout the BUILD carries. Registered here, not
  // in the CUDA block -- with no tensor argument there is no dispatch key, so
  // a CUDA-only registration can never be called (raises
  // NotImplementedError "no fallback function is registered").
  ops.impl("merlin_persistent_topk_exact_counters_slots",
           TORCH_BOX(&persistent_topk_exact_counters_slots));
  // Tensor-free for the same reason as the slot count above.
  ops.impl("merlin_persistent_topk_toggle_state",
           TORCH_BOX(&persistent_topk_toggle_state));
  ops.impl("persistent_topk_exact_counters_slots",
           TORCH_BOX(&persistent_topk_exact_counters_slots));
  ops.impl("persistent_topk_toggle_state",
           TORCH_BOX(&persistent_topk_toggle_state));
}
