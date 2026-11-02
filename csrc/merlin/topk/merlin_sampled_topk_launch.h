// merlin-hook: csrc-topk-vendored
// Launch interface of the ported upstream 0.32 sampled top-k (merlin_sampled_topk.cuh).
//
// The sampled kernels are compiled in their OWN translation unit (sampled_topk.cu): when they shared a TU with
// the vendored persistent/FilteredTopK kernels (topk.cu), nvcc generated different SASS for six FilteredTopK
// instantiations (V=1,2) than for the kernels alone. Kernel source was unchanged; the cause was the sampled kernels'
// presence in the TU (verified: removing them restores 22/22 identical functions). Split so that
// topk.cu holds only those vendored kernels. This header is the only thing topk.cu sees of the
// sampled port; it declares no device code.
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace merlin::sampled_topk {

// Upstream's three gates, unchanged (num_rows > 64, max_seq_len >= kMinSampledLength<K>, opt-in smem >= 144 KiB).
// Instantiated for K = 512, 1024, 2048 in sampled_topk.cu.
template <int K>
bool gate_open(int64_t num_rows, int64_t max_seq_len, int max_smem_per_block);

// cudaFuncSetAttribute(max dynamic smem) + the launch. Returns the
// cudaFuncSetAttribute status (the caller checks it); the launch itself is unchecked, as before.
// `max_seq_len` is already min(max_seq_len, logits.size(1)).
template <int K>
cudaError_t launch(const float* logits, const int32_t* lengths, int32_t* output,
                   int64_t stride, int max_seq_len, int64_t num_rows,
                   cudaStream_t stream);

}  // namespace merlin::sampled_topk
