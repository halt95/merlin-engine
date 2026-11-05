// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
// Launch interface of the sampled top-k ported from upstream
// (merlin_sampled_topk.cuh).
//
// The sampled kernels are compiled in their own translation unit
// (sampled_topk.cu): sharing a TU with the vendored persistent/FilteredTopK
// kernels (topk.cu) changes the SASS nvcc generates for the FilteredTopK
// instantiations. topk.cu therefore holds only those vendored kernels; this
// header is the only thing it sees of the sampled port, and it declares no
// device code.
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace merlin::sampled_topk {

// Upstream's three gates, unchanged (num_rows > 64, max_seq_len >= kMinSampledLength<K>, opt-in smem >= 144 KiB).
// Instantiated for K = 512, 1024, 2048 in sampled_topk.cu.
template <int K>
bool gate_open(int64_t num_rows, int64_t max_seq_len, int max_smem_per_block);

// cudaFuncSetAttribute(max dynamic smem) + the launch. Returns the
// cudaFuncSetAttribute status (the caller checks it); the launch itself is unchecked.
// `max_seq_len` is already min(max_seq_len, logits.size(1)).
template <int K>
cudaError_t launch(const float* logits, const int32_t* lengths, int32_t* output,
                   int64_t stride, int max_seq_len, int64_t num_rows,
                   cudaStream_t stream);

}  // namespace merlin::sampled_topk
