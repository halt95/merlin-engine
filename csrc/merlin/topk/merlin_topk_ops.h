// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
// Host entry points of the vendored top-k (topk.cu in this directory).
// Registrations: bindings.cpp in this directory.
#pragma once

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>

#include <cstdint>

namespace merlin::topk {
// Merlin's persistent top-k (topk.cu here), bound to
// _C::merlin_persistent_topk. Same signature as upstream's global
// persistent_topk() in csrc/ops.h, which upstream csrc/topk.cu defines and
// _C::persistent_topk binds; the namespace keeps the two apart.
void persistent_topk(const torch::stable::Tensor& logits,
                     const torch::stable::Tensor& lengths,
                     torch::stable::Tensor& output,
                     torch::stable::Tensor& workspace, int64_t k,
                     int64_t max_seq_len);
}  // namespace merlin::topk

// Medium-path exact-restart counters; see topk.cu for the slot layout.
void persistent_topk_exact_counters(torch::stable::Tensor& out, int64_t reset,
                                    int64_t sync);

// The slot count of THIS build's counter array, so the Python
// reader never has to guess it from an environment variable.
int64_t persistent_topk_exact_counters_slots();

// The persistent_topk switches as this
// process cached them, packed; see topk.cu for the layout.
int64_t persistent_topk_toggle_state();
