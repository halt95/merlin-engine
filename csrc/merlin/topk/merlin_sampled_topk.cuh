// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Modified by the Merlin project; see NOTICE.
// Sampled top-k for the vendored persistent top-k launcher (topk.cu in this
// directory), adapted from upstream vLLM's sampled top-k so that the Merlin op
// merlin_persistent_topk keeps upstream's dispatch order:
//   sampled (> 64 rows, long rows, >= 144 KiB smem) -> FilteredTopK -> persist.
//
// The kernel and the per-row FilteredTopK it uses (the nested
// merlin::sampled_topk::filtered_topk namespace) are upstream text under our
// own namespace, so they cannot collide with upstream's csrc/topk.cu, which is
// compiled again and owns _C::persistent_topk (see csrc/merlin/merlin.cmake).
//
// Reachability: the launcher gates this kernel exactly as upstream does
// (num_rows > 64, max_seq_len >= kMinSampledLength<K>, opt-in smem per block
// >= 144 KiB). sm_86 / sm_89 report 99 KiB, so it never launches there; it is
// still compiled for every arch in the target's CUDA_ARCHS.
#pragma once

#include <cub/cub.cuh>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

#include "topk_histogram_4096.cuh"

namespace merlin {
namespace sampled_topk {

namespace filtered_topk {

namespace hist4096 = topk_histogram_4096;

#define FLASHINFER_INLINE inline __attribute__((always_inline)) __device__

template <typename T, size_t N>
struct vec_t {
  T data[N];

  FLASHINFER_INLINE T& operator[](size_t i) { return data[i]; }
  FLASHINFER_INLINE const T& operator[](size_t i) const { return data[i]; }

  FLASHINFER_INLINE void cast_load(const T* ptr) {
#pragma unroll
    for (size_t i = 0; i < N; ++i) {
      data[i] = ptr[i];
    }
  }
};
#undef FLASHINFER_INLINE

// FilteredTopK traits for different data types
template <typename DType>
struct FilteredTopKTraits;

// Specialization for float (32-bit): coarse histogram uses FP16 high 8 bits, 4
// refinement rounds
template <>
struct FilteredTopKTraits<float> {
  using OrderedType = uint32_t;
  static constexpr int NUM_REFINE_ROUNDS = 4;
  static constexpr int FIRST_REFINE_SHIFT = 24;

  __device__ __forceinline__ static uint8_t ToCoarseKey(float x) {
    // Convert to FP16 representation and extract high 8 bits
    __half h = __float2half_rn(x);
    uint16_t bits = __half_as_ushort(h);
    uint16_t key = (bits & 0x8000) ? static_cast<uint16_t>(~bits)
                                   : static_cast<uint16_t>(bits | 0x8000);
    return static_cast<uint8_t>(key >> 8);
  }

  __device__ __forceinline__ static OrderedType ToOrdered(float x) {
    uint32_t bits = __float_as_uint(x);
    return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
  }
};

constexpr uint32_t FILTERED_TOPK_BLOCK_THREADS = 1024;
constexpr uint32_t FILTERED_TOPK_SMEM_INPUT_SIZE =
    16 * 1024;  // 16K indices per buffer
constexpr size_t FILTERED_TOPK_SMEM_DYNAMIC =
    sizeof(int) * 2 * FILTERED_TOPK_SMEM_INPUT_SIZE;  // 128KB

template <uint32_t MAX_K>
struct FilteredTopKStorage {
  alignas(128) int histogram[2][256 + 128];
  alignas(128) int counter;
  alignas(128) int threshold_bin;
  alignas(128) int num_input[2];
  alignas(128) int indices[MAX_K];
  int last_remain;
};

// With CheckOverflow, return false before using a truncated stash so the caller
// can retry with exact full-row selection.
template <typename DType, typename IdType, int VEC_SIZE, uint32_t MAX_K,
          bool UsePredicatedShortLoads, bool CheckOverflow = false>
__device__ __forceinline__ bool filtered_topk_row(
    const DType* score, IdType* dst, int length, uint32_t top_k,
    FilteredTopKStorage<MAX_K>& storage) {
  constexpr uint32_t BLOCK_SIZE = FILTERED_TOPK_BLOCK_THREADS;
  constexpr int RADIX = 256;
  constexpr int SMEM_INPUT_SIZE = FILTERED_TOPK_SMEM_INPUT_SIZE;
  const int tx = threadIdx.x;

  // Trivial case: length <= top_k
  if (length <= static_cast<int>(top_k)) {
    for (int i = tx; i < static_cast<int>(top_k); i += BLOCK_SIZE) {
      dst[i] = (i < length) ? static_cast<IdType>(i) : static_cast<IdType>(-1);
    }
    return true;
  }

  // Short path
  if (length <= 32768) {
    extern __shared__ uint8_t _smem_reg[];
    if constexpr (UsePredicatedShortLoads) {
      hist4096::histogram_4096_topk_predicated<MAX_K, 12, 8>(score, dst, length,
                                                             _smem_reg);
    } else {
      hist4096::histogram_4096_topk<MAX_K, 12, 8>(score, dst, length,
                                                  _smem_reg);
    }
    return true;
  }

  auto& s_histogram_buf = storage.histogram;
  auto& s_counter = storage.counter;
  auto& s_threshold_bin_id = storage.threshold_bin;
  auto& s_num_input = storage.num_input;
  auto& s_indices = storage.indices;
  auto& s_histogram = s_histogram_buf[0];

  // Dynamic shared memory for input double buffer
  extern __shared__ int s_input_idx[][SMEM_INPUT_SIZE];

  using Traits = FilteredTopKTraits<DType>;
  int topk = top_k;

  // Stage 1: 8-bit coarse histogram with vectorized loads
  if (tx < RADIX + 1) s_histogram[tx] = 0;
  __syncthreads();

  vec_t<DType, VEC_SIZE> score_vec;

  const int aligned_length = (length / VEC_SIZE) * VEC_SIZE;
#pragma unroll 2
  for (int base = tx * VEC_SIZE; base < aligned_length;
       base += BLOCK_SIZE * VEC_SIZE) {
    score_vec.cast_load(&score[base]);
#pragma unroll
    for (int j = 0; j < VEC_SIZE; ++j) {
      const auto bin = Traits::ToCoarseKey(score_vec[j]);
      atomicAdd(&s_histogram[bin], 1);
    }
  }
  // Handle tail
  for (int i = aligned_length + tx; i < length; i += BLOCK_SIZE) {
    const auto bin = Traits::ToCoarseKey(score[i]);
    atomicAdd(&s_histogram[bin], 1);
  }
  __syncthreads();

  // Suffix sum
  const auto run_cumsum = [&]() {
#pragma unroll 8
    for (int i = 0; i < 8; ++i) {
      if (tx < RADIX) {
        const auto j = 1 << i;
        const auto k = i & 1;
        auto value = s_histogram_buf[k][tx];
        if (tx < RADIX - j) {
          value += s_histogram_buf[k][tx + j];
        }
        s_histogram_buf[k ^ 1][tx] = value;
      }
      __syncthreads();
    }
  };

  run_cumsum();
  if (tx < RADIX && s_histogram[tx] > topk && s_histogram[tx + 1] <= topk) {
    s_threshold_bin_id = tx;
    s_num_input[0] = 0;
    s_counter = 0;
  }
  __syncthreads();

  const auto threshold_bin = s_threshold_bin_id;
  topk -= s_histogram[threshold_bin + 1];

  constexpr int NUM_ROUNDS = Traits::NUM_REFINE_ROUNDS;
  constexpr int FIRST_SHIFT = Traits::FIRST_REFINE_SHIFT;

  if (topk == 0) {
    // Collect indices where bin > threshold
#pragma unroll 2
    for (int base = tx * VEC_SIZE; base < aligned_length;
         base += BLOCK_SIZE * VEC_SIZE) {
      score_vec.cast_load(&score[base]);
#pragma unroll
      for (int j = 0; j < VEC_SIZE; ++j) {
        const auto bin = static_cast<int>(Traits::ToCoarseKey(score_vec[j]));
        if (bin > threshold_bin) {
          const auto pos = atomicAdd(&s_counter, 1);
          s_indices[pos] = base + j;
        }
      }
    }
    // Handle tail
    for (int i = aligned_length + tx; i < length; i += BLOCK_SIZE) {
      const auto bin = static_cast<int>(Traits::ToCoarseKey(score[i]));
      if (bin > threshold_bin) {
        const auto pos = atomicAdd(&s_counter, 1);
        s_indices[pos] = i;
      }
    }
    __syncthreads();
  } else {
    __syncthreads();
    if (tx < RADIX + 1) s_histogram[tx] = 0;
    __syncthreads();

    // Filter + histogram for refinement
    auto filter_and_add_to_histogram = [&](auto raw_input, int index) {
      const auto bin = static_cast<int>(Traits::ToCoarseKey(raw_input));
      if (bin > threshold_bin) {
        const auto pos = atomicAdd(&s_counter, 1);
        s_indices[pos] = index;
      } else if (bin == threshold_bin) {
        const auto pos = atomicAdd(&s_num_input[0], 1);
        if (__builtin_expect(pos < SMEM_INPUT_SIZE, 1)) {
          s_input_idx[0][pos] = index;
          const auto ordered = Traits::ToOrdered(raw_input);
          const auto sub_bin = (ordered >> FIRST_SHIFT) & 0xFF;
          atomicAdd(&s_histogram[sub_bin], 1);
        }
      }
    };
#pragma unroll 2
    for (int base = tx * VEC_SIZE; base < aligned_length;
         base += BLOCK_SIZE * VEC_SIZE) {
      score_vec.cast_load(&score[base]);
#pragma unroll
      for (int j = 0; j < VEC_SIZE; ++j) {
        filter_and_add_to_histogram(score_vec[j], base + j);
      }
    }
    // Handle tail
    for (int i = aligned_length + tx; i < length; i += BLOCK_SIZE) {
      filter_and_add_to_histogram(score[i], i);
    }
    __syncthreads();

    // Stage 2: refine with 8bit radix passes
#pragma unroll
    for (int round = 0; round < NUM_ROUNDS; ++round) {
      auto& s_last_remain = storage.last_remain;
      const auto r_idx = round % 2;

      const auto _raw_num_input = s_num_input[r_idx];
      if constexpr (CheckOverflow) {
        if (_raw_num_input > SMEM_INPUT_SIZE) return false;
      }
      const auto num_input =
          (_raw_num_input < SMEM_INPUT_SIZE) ? _raw_num_input : SMEM_INPUT_SIZE;

      run_cumsum();
      if (tx < RADIX && s_histogram[tx] > topk && s_histogram[tx + 1] <= topk) {
        s_threshold_bin_id = tx;
        s_num_input[r_idx ^ 1] = 0;
        s_last_remain = topk - s_histogram[tx + 1];
      }
      __syncthreads();

      const auto threshold = s_threshold_bin_id;
      topk -= s_histogram[threshold + 1];

      const int offset = FIRST_SHIFT - round * 8;
      const bool is_last_round = (round == NUM_ROUNDS - 1);

      if (topk == 0) {
        for (int i = tx; i < num_input; i += BLOCK_SIZE) {
          const auto idx = s_input_idx[r_idx][i];
          const auto bin = (Traits::ToOrdered(score[idx]) >> offset) & 0xFF;
          if (static_cast<int>(bin) > threshold) {
            const auto pos = atomicAdd(&s_counter, 1);
            s_indices[pos] = idx;
          }
        }
        __syncthreads();
        break;
      } else {
        __syncthreads();
        if (tx < RADIX + 1) s_histogram[tx] = 0;
        __syncthreads();
        for (int i = tx; i < num_input; i += BLOCK_SIZE) {
          const auto idx = s_input_idx[r_idx][i];
          const auto raw_input = score[idx];
          const auto bin = (Traits::ToOrdered(raw_input) >> offset) & 0xFF;
          if (static_cast<int>(bin) > threshold) {
            const auto pos = atomicAdd(&s_counter, 1);
            s_indices[pos] = idx;
          } else if (static_cast<int>(bin) == threshold) {
            if (is_last_round) {
              const auto pos = atomicAdd(&s_last_remain, -1);
              if (pos > 0) {
                s_indices[top_k - pos] = idx;
              }
            } else {
              const auto pos = atomicAdd(&s_num_input[r_idx ^ 1], 1);
              if (__builtin_expect(pos < SMEM_INPUT_SIZE, 1)) {
                s_input_idx[r_idx ^ 1][pos] = idx;
                const auto bin32 = Traits::ToOrdered(raw_input);
                const auto sub_bin = (bin32 >> (offset - 8)) & 0xFF;
                atomicAdd(&s_histogram[sub_bin], 1);
              }
            }
          }
        }
        __syncthreads();
      }
    }
  }

  // Output phase - mode-specific
#pragma unroll 2
  for (int base = tx; base < static_cast<int>(top_k); base += BLOCK_SIZE) {
    const int idx = s_indices[base];
    dst[base] = static_cast<IdType>(idx);
  }
  return true;
}

}  // namespace filtered_topk

constexpr int kThreads = 1024;
// Conservative crossover bounds from the B300 batch/length sweep.
template <int K>
constexpr int kMinSampledLength = K == 512 ? 98304 : 65536;
// Target half the buffer to leave room for sampling error.
constexpr int kSample = 4096;
constexpr int kCapacity = 8192;

struct Storage {
  uint2 candidates[kCapacity];
  int histogram[2048];
  typename cub::BlockScan<int, kThreads>::TempStorage scan;
  int count;
  int remaining;
  int bin;
  uint32_t prefix;
};

__device__ __forceinline__ uint32_t ordered(float value) {
  const uint32_t bits = __float_as_uint(value);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

__device__ __forceinline__ uint32_t coarse(uint32_t key) {
  const uint32_t bits = (key & 0x80000000u) ? (key ^ 0x80000000u) : ~key;
  return topk_histogram_4096::extract_coarse_bin_N<11>(__uint_as_float(bits));
}

// Select the bin containing the rank and update the rank within that bin.
template <int Bins>
__device__ void select_bin(Storage& s) {
  constexpr int kItems = (Bins + kThreads - 1) / kThreads;
  int sums[kItems];
#pragma unroll
  for (int j = 0; j < kItems; ++j) {
    const int bin = Bins - 1 - (threadIdx.x * kItems + j);
    sums[j] = bin >= 0 ? s.histogram[bin] : 0;
  }
  cub::BlockScan<int, kThreads>(s.scan).InclusiveSum(sums, sums);
  const int remaining = s.remaining;
  __syncthreads();
#pragma unroll
  for (int j = 0; j < kItems; ++j) {
    const int bin = Bins - 1 - (threadIdx.x * kItems + j);
    if (bin >= 0 && sums[j] >= remaining &&
        sums[j] - s.histogram[bin] < remaining) {
      s.bin = bin;
      s.remaining = remaining - (sums[j] - s.histogram[bin]);
    }
  }
  __syncthreads();
}

// Find a coarse bin, then refine its full FP32 keys one byte at a time.
// Read shared candidates normally, or the full row for exact fallback.
template <bool Buffered>
__device__ uint32_t threshold(Storage& s, const float* row, int count,
                              int rank) {
  const int tid = threadIdx.x;
  if (tid == 0) {
    s.remaining = rank;
    s.prefix = 0;
  }
  for (int i = tid; i < 2048; i += kThreads) s.histogram[i] = 0;
  __syncthreads();
  for (int i = tid; i < count; i += kThreads) {
    const uint32_t key = Buffered ? s.candidates[i].y : ordered(row[i]);
    atomicAdd(&s.histogram[coarse(key)], 1);
  }
  __syncthreads();
  select_bin<2048>(s);
  const int coarse_bin = s.bin;
  uint32_t mask = 0;
  for (int shift = 24; shift >= 0; shift -= 8) {
    if (tid < 256) s.histogram[tid] = 0;
    __syncthreads();
    const uint32_t prefix = s.prefix;
    for (int i = tid; i < count; i += kThreads) {
      const uint32_t key = Buffered ? s.candidates[i].y : ordered(row[i]);
      if (coarse(key) == coarse_bin && (key & mask) == prefix) {
        atomicAdd(&s.histogram[(key >> shift) & 255], 1);
      }
    }
    __syncthreads();
    select_bin<256>(s);
    if (tid == 0) s.prefix = prefix | (static_cast<uint32_t>(s.bin) << shift);
    mask |= 255u << shift;
    __syncthreads();
  }
  return s.prefix;
}

__device__ __forceinline__ int reserve(bool hit, int* counter) {
  const uint32_t mask = __ballot_sync(0xffffffffu, hit);
  if (mask == 0) return 0;
  const int lane = threadIdx.x & 31;
  int base = 0;
  if (lane == 0) base = atomicAdd(counter, __popc(mask));
  base = __shfl_sync(0xffffffffu, base, 0);
  return base + __popc(mask & ((1u << lane) - 1));
}

template <bool Buffered>
__device__ void emit(Storage& s, const float* row, int32_t* dst, int count,
                     uint32_t cutoff) {
  const int tid = threadIdx.x;
  if (tid == 0) s.count = 0;
  __syncthreads();
  for (int base = 0; base < count; base += kThreads) {
    const int i = base + tid;
    const uint2 pair = i < count ? (Buffered ? s.candidates[i]
                                             : make_uint2(i, ordered(row[i])))
                                 : make_uint2(0, 0);
    bool keep = i < count && pair.y > cutoff;
    if (i < count && pair.y == cutoff) {
      keep = atomicSub(&s.remaining, 1) > 0;
    }
    const int offset = reserve(keep, &s.count);
    if (keep) dst[offset] = pair.x;
  }
}

template <int K>
__global__ void __launch_bounds__(kThreads)
    sampled_topk_kernel(const float* __restrict__ input,
                        const int32_t* __restrict__ lengths,
                        int32_t* __restrict__ output, int64_t stride,
                        int max_length) {
  extern __shared__ __align__(16) unsigned char smem[];
  auto& s = *reinterpret_cast<Storage*>(smem);
  const int tid = threadIdx.x;
  const int length = max(0, min(lengths[blockIdx.x], max_length));
  const float* row = input + blockIdx.x * stride;
  int32_t* dst = output + blockIdx.x * K;
  // Graph allocations can hold rows much shorter than the host length bound.
  if (length < kMinSampledLength<K>) {
    __shared__ filtered_topk::FilteredTopKStorage<K> fallback_storage;
    bool selected;
    if ((stride & 3) == 0 && (reinterpret_cast<uintptr_t>(row) & 15) == 0) {
      selected =
          filtered_topk::filtered_topk_row<float, int32_t, 4, K, false, true>(
              row, dst, length, K, fallback_storage);
    } else {
      selected =
          filtered_topk::filtered_topk_row<float, int32_t, 1, K, true, true>(
              row, dst, length, K, fallback_storage);
    }
    if (!selected) {
      const uint32_t exact_cutoff = threshold<false>(s, row, length, K);
      emit<false>(s, row, dst, length, exact_cutoff);
    }
    return;
  }
  if (tid == 0) s.remaining = max(1, (kCapacity / 2) * kSample / length);
  for (int i = tid; i < 2048; i += kThreads) s.histogram[i] = 0;
  __syncthreads();
  // Sample contiguous warps spread across the row to keep loads coalesced.
  for (int i = tid; i < kSample; i += kThreads) {
    const int index = (i / 32) * (length / (kSample / 32)) + i % 32;
    atomicAdd(&s.histogram[coarse(ordered(row[index]))], 1);
  }
  __syncthreads();
  select_bin<2048>(s);
  const uint16_t half_key = s.bin << 5;
  const uint16_t half_bits =
      (half_key & 0x8000u) ? (half_key ^ 0x8000u) : ~half_key;
  const uint32_t cutoff = ordered(__half2float(__ushort_as_half(half_bits)));
  if (tid == 0) s.count = 0;
  __syncthreads();
  constexpr int kItems = 16;
  // Compact survivors with one reservation per warp per tile. Keep counting
  // beyond capacity so overflow triggers fallback instead of using a subset.
  for (int base = 0; base < length; base += kItems * kThreads) {
    uint32_t keys[kItems];
    uint32_t hit_mask = 0;
    if ((reinterpret_cast<uintptr_t>(row) & 15) == 0 &&
        base + kItems * kThreads <= length) {
#pragma unroll
      for (int j = 0; j < kItems / 4; ++j) {
        const float4 values =
            reinterpret_cast<const float4*>(row + base)[tid + j * kThreads];
        keys[4 * j] = ordered(values.x);
        keys[4 * j + 1] = ordered(values.y);
        keys[4 * j + 2] = ordered(values.z);
        keys[4 * j + 3] = ordered(values.w);
      }
    } else {
#pragma unroll
      for (int j = 0; j < kItems; ++j) {
        const int i = base + tid * 4 + (j / 4) * kThreads * 4 + j % 4;
        keys[j] = i < length ? ordered(row[i]) : 0;
      }
    }
#pragma unroll
    for (int j = 0; j < kItems; ++j) {
      const int i = base + tid * 4 + (j / 4) * kThreads * 4 + j % 4;
      if (i < length && keys[j] >= cutoff) hit_mask |= 1u << j;
    }
    const int lane = tid & 31;
    const int hits = __popc(hit_mask);
    int inclusive = hits;
#pragma unroll
    for (int delta = 1; delta < 32; delta *= 2) {
      const int previous = __shfl_up_sync(0xffffffffu, inclusive, delta);
      if (lane >= delta) inclusive += previous;
    }
    const int total = __shfl_sync(0xffffffffu, inclusive, 31);
    int warp_base = 0;
    if (lane == 31 && total != 0) warp_base = atomicAdd(&s.count, total);
    warp_base = __shfl_sync(0xffffffffu, warp_base, 31);
    int offset = warp_base + inclusive - hits;
#pragma unroll
    for (int j = 0; j < kItems; ++j) {
      if (hit_mask & (1u << j)) {
        if (offset < kCapacity) {
          s.candidates[offset] = make_uint2(
              base + tid * 4 + (j / 4) * kThreads * 4 + j % 4, keys[j]);
        }
        ++offset;
      }
    }
  }
  __syncthreads();
  const int count = s.count;
  if (count >= K && count <= kCapacity) {
    const uint32_t exact_cutoff = threshold<true>(s, row, count, K);
    emit<true>(s, row, dst, count, exact_cutoff);
  } else {
    const uint32_t exact_cutoff = threshold<false>(s, row, length, K);
    emit<false>(s, row, dst, length, exact_cutoff);
  }
}

}  // namespace sampled_topk
}  // namespace merlin
