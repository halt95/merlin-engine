// merlin-hook: csrc-topk-vendored
// Upstream v0.32.0rc0 sampled top-k, ported into the Merlin op, in its own translation unit
// (see merlin_sampled_topk_launch.h for why). launch_persistent_topk() in topk.cu launches it.

#include "merlin/topk/merlin_sampled_topk_launch.h"

#include <cuda_runtime.h>

#include <cstddef>

#include "merlin_sampled_topk.cuh"
// Included for the static_assert below only: no FilteredTopK kernel is instantiated in this TU.
#include "persistent_topk.cuh"

namespace merlin::sampled_topk {

template <int K>
bool gate_open(int64_t num_rows, int64_t max_seq_len, int max_smem_per_block) {
  // Allow static fallback storage in addition to the 128 KiB dynamic buffer.
  return num_rows > 64 && max_seq_len >= kMinSampledLength<K> &&
         max_smem_per_block >= 144 * 1024;
}

template <int K>
cudaError_t launch(const float* logits, const int32_t* lengths, int32_t* output,
                   int64_t stride, int max_seq_len, int64_t num_rows,
                   cudaStream_t stream) {
  static_assert(filtered_topk::FILTERED_TOPK_SMEM_DYNAMIC ==
                    merlin::filtered_topk::FILTERED_TOPK_SMEM_DYNAMIC,
                "sampled_topk dynamic smem must match the FilteredTopK "
                "buffer upstream sizes it from");
  static_assert(sizeof(Storage) <= filtered_topk::FILTERED_TOPK_SMEM_DYNAMIC,
                "sampled_topk Storage must fit its dynamic smem");
  auto kernel = sampled_topk_kernel<K>;
  constexpr size_t smem_size = filtered_topk::FILTERED_TOPK_SMEM_DYNAMIC;
  cudaError_t status = cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
  if (status != cudaSuccess) return status;
  kernel<<<num_rows, kThreads, smem_size, stream>>>(logits, lengths, output,
                                                    stride, max_seq_len);
  return cudaSuccess;
}

template bool gate_open<512>(int64_t, int64_t, int);
template bool gate_open<1024>(int64_t, int64_t, int);
template bool gate_open<2048>(int64_t, int64_t, int);
template cudaError_t launch<512>(const float*, const int32_t*, int32_t*, int64_t,
                                 int, int64_t, cudaStream_t);
template cudaError_t launch<1024>(const float*, const int32_t*, int32_t*,
                                  int64_t, int, int64_t, cudaStream_t);
template cudaError_t launch<2048>(const float*, const int32_t*, int32_t*,
                                  int64_t, int, int64_t, cudaStream_t);

}  // namespace merlin::sampled_topk
