// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Modified by the Merlin project; see NOTICE.
/*
 * Persistent TopK Scheduler for DSA Indexer
 */

#ifndef PERSISTENT_TOPK_CUH_
#define PERSISTENT_TOPK_CUH_

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cstdint>

#include "topk_histogram_4096.cuh"

namespace merlin {
namespace persistent {

// ============================================================================
// Constants
// ============================================================================

constexpr int kThreadsPerBlock = 1024;
constexpr int RADIX = 256;

// Medium path: all shared state in dynamic smem (no static __shared__,
// which would inflate the kernel's smem footprint and kill occupancy
// for the decode/trivial paths).
constexpr size_t kMediumHistBytes = 2 * (RADIX + 128) * sizeof(int);  // 3072
// 6 scalars: the last is the per-row "overflow already counted" flag of the
// medium-path exact restart. kMediumHeaderSize rounds up to 128 B.
constexpr size_t kMediumScalarsBytes = 6 * sizeof(int);               // 24
constexpr size_t kMediumHeaderSize =
    (kMediumHistBytes + kMediumScalarsBytes + 127) & ~size_t(127);  // 3200
constexpr int MAX_BUFFERED_ITEMS = 4096;
constexpr size_t kSmemMedium =
    kMediumHeaderSize + 2 * MAX_BUFFERED_ITEMS * sizeof(int);  // 35968
constexpr uint32_t RADIX_THRESHOLD = 32768;

// Decode path constants
constexpr int kDecodeBins = 2048;
constexpr uint32_t HIST2048_THRESHOLD = 8192;

// Large path: fixed shared memory for histograms + scalars
constexpr size_t kFixedSmemLarge =
    ((RADIX + RADIX + 5) * sizeof(uint32_t) + 15) & ~size_t(15);

// ============================================================================
// Common helpers
// ============================================================================

__device__ __forceinline__ auto convert_to_uint32_v2(float x) -> uint32_t {
  uint32_t bits = __float_as_uint(x);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

// The definition of the effective tie direction, called by both the host (to
// report it) and the device (to obey it), so the reported policy and the
// executed policy cannot drift.
//
// With `det_ties` off the buffered paths keep their original racy atomicAdd
// rule, which has no direction; the medium-path exact restart is deterministic
// either way, so it keeps the most-recent-wins direction.
__host__ __device__ __forceinline__ bool effective_tie_recent(bool det_ties,
                                                              bool tie_recent) {
  return det_ties ? tie_recent : true;
}

__device__ __forceinline__ auto convert_to_uint8(float x) -> uint8_t {
  __half h = __float2half_rn(x);
  uint16_t bits = __half_as_ushort(h);
  uint16_t key = (bits & 0x8000) ? static_cast<uint16_t>(~bits)
                                 : static_cast<uint16_t>(bits | 0x8000);
  return static_cast<uint8_t>(key >> 8);
}

// ============================================================================
// Vectorized load helpers
// ============================================================================

// Unconditional float4 load with cache hint (.cg = cache at global level only).
__device__ __forceinline__ void load_float4(const float* ptr, float& v0,
                                            float& v1, float& v2, float& v3) {
  uint32_t r0, r1, r2, r3;
  asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "l"(ptr));
  v0 = __uint_as_float(r0);
  v1 = __uint_as_float(r1);
  v2 = __uint_as_float(r2);
  v3 = __uint_as_float(r3);
}

// Per-element predicated scalar loads with -inf default.
__device__ __forceinline__ void load_float4_predicated(const float* ptr,
                                                       int base, int seq_len,
                                                       float& v0, float& v1,
                                                       float& v2, float& v3) {
  uint32_t r0, r1, r2, r3;
  int p0 = (base < seq_len);
  int p1 = (base + 1 < seq_len);
  int p2 = (base + 2 < seq_len);
  int p3 = (base + 3 < seq_len);
  asm volatile(
      "{\n"
      "  .reg .pred pr0, pr1, pr2, pr3;\n"
      "  setp.ne.u32 pr0, %4, 0;\n"
      "  setp.ne.u32 pr1, %5, 0;\n"
      "  setp.ne.u32 pr2, %6, 0;\n"
      "  setp.ne.u32 pr3, %7, 0;\n"
      "  mov.u32 %0, 0xFF800000;\n"
      "  mov.u32 %1, 0xFF800000;\n"
      "  mov.u32 %2, 0xFF800000;\n"
      "  mov.u32 %3, 0xFF800000;\n"
      "  @pr0 ld.global.cg.u32 %0, [%8];\n"
      "  @pr1 ld.global.cg.u32 %1, [%8+4];\n"
      "  @pr2 ld.global.cg.u32 %2, [%8+8];\n"
      "  @pr3 ld.global.cg.u32 %3, [%8+12];\n"
      "}\n"
      : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
      : "r"(p0), "r"(p1), "r"(p2), "r"(p3), "l"(ptr));
  v0 = __uint_as_float(r0);
  v1 = __uint_as_float(r1);
  v2 = __uint_as_float(r2);
  v3 = __uint_as_float(r3);
}

// ============================================================================
// Large path: inter-CTA coordination state (one per group)
// ============================================================================

struct RadixRowState {
  uint32_t histogram[3][256];  // Triple-buffered histograms
  uint32_t remaining_k;
  uint32_t prefix;
  int arrival_counter;
  int output_counter;
};

// ============================================================================
// Kernel parameters
// ============================================================================

struct PersistentTopKParams {
  const float* __restrict__ input;      // [num_rows, stride]
  int32_t* __restrict__ output;         // [num_rows, top_k]
  const int32_t* __restrict__ lengths;  // [num_rows]
  RadixRowState* row_states;            // large path: per-group state
  uint32_t num_rows;
  uint32_t stride;
  uint32_t top_k;           // actual k value for output stride
  uint32_t chunk_size;      // large path: elements per CTA
  uint32_t ctas_per_group;  // 1=medium, >1=large
  uint32_t max_seq_len;     // max seq_len across all rows (for early CTA exit)
  // The policy fields below are launch constants: topk.cu passes the same
  // values on every launch, which a cudagraph bakes in at capture.
  //
  // Medium-path exact top-k restart.
  // `exact_counters` is a process-lifetime device buffer of
  // kNumTopKExactCounters 64-bit counters, or nullptr; the kernel only ever
  // atomicAdds into it, never reads back.
  bool exact_medium;
  unsigned long long* exact_counters;
  // Deterministic tie-break at the k-th cutoff.
  bool det_ties;
  // Ranks ties by DESCENDING element index, so the surviving blocks are the
  // most recent rather than the oldest. Both rules are deterministic; they
  // differ in which tied blocks survive, and measurably in draft acceptance at
  // depth. Only read when det_ties is set.
  bool tie_recent;
  // On an overflowing DECODE row, take the same exact restart the medium path
  // takes instead of ranking a truncated candidate set. It costs four full-row
  // rescans on the overflowing rows only; counter [10] counts them.
  bool exact_decode;
  // Large-path deterministic emission,
  // ANDed per launch with the capacity gate 4 * ctas_per_group <= RADIX (the
  // per-CTA counts and the two fault-verdict regions are staged in one
  // 256-slot RadixRowState histogram row). With det ties on, a launch that
  // fails that gate or is not co-resident never reaches this kernel: it runs
  // exact_rows_topk_kernel (launch_persistent_topk in topk.cu).
  bool det_large;
  // Large-path fault injector.  -1 disables it; any other value names the
  // `cta_in_group` whose CTA-LOCAL fault predicate is forced, and suppresses
  // __trap().  The host always passes -1; the MERLIN_TOPK_DIAG_FORCE_FAULT
  // build forces the same predicate with the trap kept.
  int fault_inject_cta;
  // Which fault the injector forces in that CTA: false = the PRE-EMISSION
  // predicate `my_keep > local_eq` (X3, emission skipped on that row); true =
  // the EMISSION-BOUND slot check in the placement loop (X4, the only way to
  // exercise `suffix_sum[6]`).  The host always passes false.
  bool fault_inject_emit;
  // Sorted emission: every row leaves the kernel in canonical order --
  // ascending index, padding last as -1 -- so the Python sort can be dropped.
  // Paths that already emit that order by construction (trivial, medium-path
  // restart, deterministic large path) are untouched; the paths that place by
  // atomic arrival (decode-buffered, medium-buffered, the baseline large tail)
  // re-emit their own selected set through sorted_emit_row(). Selection is never
  // re-derived, so the selected set cannot change. A launch constant that a
  // cudagraph bakes in.
  bool sorted_emit;
};

// ============================================================================
// Device-global exact-path counters, declared here above every user.
//
// The decode template increments `kCtrDecodeOverflow` through `counters`, a
// plain `unsigned long long*` parameter, so that enumerator is a non-dependent
// name: two-phase lookup resolves it when the template is defined, not when it
// is instantiated, so the enum must be declared before the template.
//
// The safeguard is the declaration order, not a literal: every increment still
// names its enumerator, so a slot renumbering stays a one-line edit here.
// ============================================================================

// Device-global counters. Plain device atomics on a process-lifetime buffer:
// no host sync and no host-visible allocation on the launch path, so this is
// safe to record inside a cudagraph capture and on every replay. Read from
// Python with `torch.ops._C.persistent_topk_exact_counters`.
//
// The counters are `unsigned long long`, not `unsigned int`, and the reader
// tensor is int64: a 32-bit counter read through a signed tensor goes negative
// and then wraps. atomicAdd(unsigned long long*) is native on sm_86, so the
// width costs nothing on the increment path.
//
// Slot map:
//
//   [0] medium rows      [1] medium overflow   [2] medium exact fallback
//   [3] medium exact fault
//   [4]-[7]  large path (rows_large, ties_cut, ties_dropped, faults_large)
//   [8] exact medium restart on   [9] deterministic large path on
//   [10] decode overflow
//
// Slots 8-9 are written by the HOST after the device copy; the device never
// touches them, and they are slots inside the array, not two elements appended
// after it. A reader buffer of any other length is rejected outright by the
// length check in the reader op, which is the fail-loud direction.
enum TopKExactCounter : int {
  kCtrMediumRows = 0,           // denominator: medium-path rows entered
  kCtrMediumOverflow = 1,       // rows where a 4,096-slot guard would drop
  kCtrMediumExactFallback = 2,  // rows that took the exact restart
  kCtrMediumExactFault = 3,     // exact restart could not complete
  // Large path (multi-CTA radix_topk, seq_len > RADIX_THRESHOLD) of THIS
  // kernel. Slots 4-7 are only touched by its deterministic emission. A launch
  // that topk.cu sends to exact_rows_topk_kernel instead (cooperative group not
  // co-resident, or 4 * ctas_per_group > RADIX) touches no counter, so
  // `[9] == 1 && [4] == 0` while large rows ran means those rows took the exact
  // one-CTA-per-row kernel (still deterministic), not a degraded tail; slot [9]
  // reports the compiled policy value (det_large_enabled() in topk.cu), not the
  // per-launch path.
  kCtrLargeDetRows = 4,       // large rows emitted deterministically
  kCtrLargeTiesCut = 5,       // rows where the pivot-equal group was cut
  kCtrLargeTiesDropped = 6,   // pivot-equal elements discarded by those cuts
  kCtrLargeFault = 7,         // large-path invariant violation
  kSlotSwitchMedium = 8,        // host-written flag, NOT a device counter
  kSlotSwitchLarge = 9,         // host-written flag, NOT a device counter
  kCtrDecodeOverflow = 10,      // decode rows whose buffered-candidate count
                                // exceeded DBUF before deterministic ranking
  kNumTopKExactCounters = 11,   // total slots: device buffer AND reader output
};

// ============================================================================
// In-kernel ascending emission of a row's OWN selected set. Declared here,
// above every caller (decode/medium dispatch, the baseline large tail, and the
// standalone fallback kernel), for the two-phase-lookup reason given above the
// counter enum.
//
// CONTRACT.  On return, row[0, TopK) is byte-identical to what
// `_qsa_canonicalise` (vllm/models/qwen4_exp/nvidia/ops/qsa_indexer.py) makes
// of the row it was handed:
//
//     tmp = where(row < 0, INT32_MAX, row); tmp = sort(tmp)
//     row = where(tmp == INT32_MAX, -1, tmp)
//
// i.e. the non-negative entries ascending, then -1 for every negative entry
// (and, to the byte, for any entry equal to INT32_MAX -- the Python spelling
// maps those to -1 too).  It is a function of the MULTISET of values in the
// row and nothing else, so it cannot change which blocks were selected: the
// selection is whatever the path already wrote, including a truncated or
// racy one.
//
// The placement follows upstream vLLM #55122 (det_select_row): a per-index
// membership flag and a BLOCKED-ownership BlockScan give each selected index
// its ascending position with no atomics. Membership comes from the path's own emission
// instead: a 32,768-bit shared bitmap, one word per thread.  Thread tx owns
// indices [32 tx, 32 tx + 32); its exclusive prefix of popcounts is the
// ascending position of its first set bit.
//
// FAST PATH.  Every non-negative entry is < 32,768 (= RADIX_THRESHOLD, so every
// single-CTA path qualifies) and no entry repeats.  Cost: one smem word
// cleared and at most TopK/1024 global loads per thread, TopK shared
// atomicOr's with no two on the same bit, one 1,024-wide BlockScan, TopK
// global stores, four block barriers.  No global atomics.
//
// FALLBACK.  An out-of-range value or a duplicate (bitmap atomicOr returns the
// bit already set) cannot be expressed as a bitmap, so the row is ranked
// instead: rank(i) = #{j : key_j < key_i} + #{j < i : key_j == key_i}, a stable
// counting sort that equals torch.sort on ANY int32 multiset.  O(TopK^2 / 1024)
// shared reads per thread; reached only by the baseline large tail (indices
// past 32,768) and by rows whose unfilled slots hold stale values (buffered
// overflow without the exact repair).
//
// SHARED MEMORY.  Aliases the dynamic window from byte 0.  The caller's shared
// state is dead on entry -- the leading barrier makes sure no thread is still
// reading it -- and the trailing barrier keeps the next row's loader off this
// row's bitmap.  Size is static_asserted against kSmemMedium, the floor every
// persistent_topk launch allocates.
// ============================================================================
constexpr uint32_t kSortedEmitBitmapWords = static_cast<uint32_t>(kThreadsPerBlock);
constexpr uint32_t kSortedEmitBitmapBits = 32u * kSortedEmitBitmapWords;
static_assert(kSortedEmitBitmapWords == 1024u,
              "one bitmap word per thread: the scan below is one item per thread");
static_assert(RADIX_THRESHOLD <= kSortedEmitBitmapBits,
              "every single-CTA path's indices must fit the fast-path bitmap");

template <int TopK>
struct SortedEmitSmem {
  typename cub::BlockScan<uint32_t, kThreadsPerBlock>::TempStorage scan;
  uint32_t bitmap[kSortedEmitBitmapWords];
  int32_t vals[TopK];
  uint32_t fallback;
};

template <int TopK>
__host__ __device__ constexpr size_t sorted_emit_smem_bytes() {
  return (sizeof(SortedEmitSmem<TopK>) + 15) & ~size_t(15);
}

template <int TopK>
__device__ __noinline__ void sorted_emit_row(int32_t* __restrict__ row) {
  static_assert(sorted_emit_smem_bytes<TopK>() <= kSmemMedium,
                "sorted emission must fit the smallest persistent_topk window");
  static_assert(alignof(SortedEmitSmem<TopK>) <= 16,
                "SortedEmitSmem over-aligned for the dynamic window");
  using BlockScan = cub::BlockScan<uint32_t, kThreadsPerBlock>;
  extern __shared__ __align__(16) char sorted_emit_smem[];
  auto* s = reinterpret_cast<SortedEmitSmem<TopK>*>(sorted_emit_smem);
  const uint32_t tx = threadIdx.x;

  // E0: every emission store of the path is issued and no thread still reads
  // the path's shared state (block-scope ordering of the global stores too).
  __syncthreads();
  s->bitmap[tx] = 0u;
  if (tx == 0) s->fallback = 0u;
  __syncthreads();

  // E1: membership.  Reads of `row` all happen before E2; writes all after.
  for (uint32_t i = tx; i < static_cast<uint32_t>(TopK); i += kThreadsPerBlock) {
    const int32_t v = row[i];
    s->vals[i] = v;
    if (v >= 0) {
      const uint32_t u = static_cast<uint32_t>(v);
      if (u < kSortedEmitBitmapBits) {
        const uint32_t bit = 1u << (u & 31u);
        if ((atomicOr(&s->bitmap[u >> 5], bit) & bit) != 0u) {
          atomicOr(&s->fallback, 1u);  // duplicate value
        }
      } else {
        atomicOr(&s->fallback, 1u);  // beyond the bitmap
      }
    }
  }
  __syncthreads();  // E2

  if (s->fallback == 0u) {  // block-uniform: written only before E2
    uint32_t word = s->bitmap[tx];
    uint32_t pos = 0u;
    uint32_t total = 0u;
    BlockScan(s->scan).ExclusiveSum(static_cast<uint32_t>(__popc(word)), pos,
                                    total);
    while (word != 0u) {
      const uint32_t b = static_cast<uint32_t>(__ffs(word) - 1);
      row[pos++] = static_cast<int32_t>((tx << 5) | b);
      word &= word - 1u;
    }
    for (uint32_t i = total + tx; i < static_cast<uint32_t>(TopK);
         i += kThreadsPerBlock) {
      row[i] = -1;
    }
  } else {
    constexpr int32_t kBig = 0x7FFFFFFF;
    for (uint32_t i = tx; i < static_cast<uint32_t>(TopK);
         i += kThreadsPerBlock) {
      const int32_t vi = s->vals[i];
      const int32_t ki = vi < 0 ? kBig : vi;
      uint32_t rank = 0u;
      for (uint32_t j = 0; j < static_cast<uint32_t>(TopK); ++j) {
        const int32_t vj = s->vals[j];
        const int32_t kj = vj < 0 ? kBig : vj;
        rank += (kj < ki || (kj == ki && j < i)) ? 1u : 0u;
      }
      row[rank] = (ki == kBig) ? -1 : ki;
    }
  }
  // E3: the next row (or the next path in this row) reuses the window.
  __syncthreads();
}

// ============================================================================
// Decode path: 2048-bin histogram for short sequences (seq_len <= 8192)
// Uses 11-bit half-precision bins for fine granularity.
// One histogram pass typically suffices since 8192/2048 = 4 elements/bin avg.
// ============================================================================

// 11-bit bin from half-precision representation (ascending: high values -> high
// bins)
__device__ __forceinline__ uint32_t decode_bin(float x) {
  __half hx = __float2half(x);
  uint16_t bits = __half_as_ushort(hx);
  uint16_t key = (bits & 0x8000) ? static_cast<uint16_t>(~bits)
                                 : static_cast<uint16_t>(bits | 0x8000);
  return key >> 5;
}

// Returns TRUE when this row overflowed its candidate buffer AND the caller
// asked for the exact repair, in which case nothing has been emitted that the
// restart will not overwrite and the caller must run it.
// The restart itself lives with the medium path further down and is called
// from the dispatch site, so this function needs no forward declaration of it
// -- some host compilers reject one for a `__device__` template.
template <int TopK>
__device__ __noinline__ bool histogram_2048_topk(
    const float* __restrict__ logits, int32_t* __restrict__ output_indices,
    int32_t seq_len, bool det_ties, bool tie_recent,
    unsigned long long* __restrict__ counters, bool exact_decode) {
  extern __shared__ int decode_smem[];
  const int tx = threadIdx.x;
  const int lane = tx & 31;

  // ---- Layout constants ----
  constexpr int SBASE = 8192 - 8;           // 8184
  constexpr int RHIST = RADIX + 128;        // 384
  constexpr int BOFF = 2 * RHIST;           // 768
  constexpr int DBUF = (SBASE - BOFF) / 2;  // 3708
  constexpr int MAX_ITEMS_PER_THREAD =
      (HIST2048_THRESHOLD + kThreadsPerBlock - 1) / kThreadsPerBlock;

  // sOVF uses a scalar slot that the `tx < 8` loop below already zeroes, so
  // the decode shared-memory budget is unchanged.
  enum : int {
    sTHR = 0,
    sOUT = 1,
    sREF = 2,
    sFIN = 3,
    sBUF0 = 4,
    sBUF1 = 5,
    sOVF = 6
  };

  // ---- Initialize scalars (prevents stale data from prior rows) ----
  if (tx < 8) {
    decode_smem[SBASE + tx] = 0;
  }

  // ---- Phase 1: Build 2048-bin histogram with float4 vectorized loads ----
  int* histo = decode_smem;
  uint16_t reg_bins[MAX_ITEMS_PER_THREAD];
  int nitems = 0;

  for (int i = tx; i < kDecodeBins; i += kThreadsPerBlock) {
    histo[i] = 0;
  }
  __syncthreads();

  const int n_vec = (seq_len + 3) >> 2;
  const bool row_aligned = ((reinterpret_cast<uintptr_t>(logits) & 15) == 0);

  for (int i = tx; i < n_vec; i += kThreadsPerBlock) {
    const int base = i << 2;
    float v0, v1, v2, v3;

    if (row_aligned && base + 3 < seq_len) {
      load_float4(logits + base, v0, v1, v2, v3);
    } else {
      load_float4_predicated(logits + base, base, seq_len, v0, v1, v2, v3);
    }

    const uint16_t b0 = static_cast<uint16_t>(decode_bin(v0));
    const uint16_t b1 = static_cast<uint16_t>(decode_bin(v1));
    const uint16_t b2 = static_cast<uint16_t>(decode_bin(v2));
    const uint16_t b3 = static_cast<uint16_t>(decode_bin(v3));
    reg_bins[nitems++] = b0;
    reg_bins[nitems++] = b1;
    reg_bins[nitems++] = b2;
    reg_bins[nitems++] = b3;
    atomicAdd(&histo[b0], 1);
    atomicAdd(&histo[b1], 1);
    atomicAdd(&histo[b2], 1);
    atomicAdd(&histo[b3], 1);
  }
  __syncthreads();

  // ---- CUB suffix sum ----
  using BlockScanT = cub::BlockScan<int, kThreadsPerBlock>;
  const int h0 = histo[2 * tx];
  const int pair_sum = h0 + histo[2 * tx + 1];

  auto& scan_storage = *reinterpret_cast<typename BlockScanT::TempStorage*>(
      decode_smem + kDecodeBins);

  int pair_prefix, total;
  BlockScanT(scan_storage).ExclusiveSum(pair_sum, pair_prefix, total);

  // Find threshold bin purely from registers
  const int pair_suffix = total - pair_prefix;

  if (pair_suffix >= TopK && (pair_suffix - h0) < TopK) {
    decode_smem[SBASE + sTHR] = 2 * tx;
  }
  {
    const int right_suf = pair_suffix - h0;
    const int next_suf = pair_suffix - pair_sum;
    if (right_suf >= TopK && next_suf < TopK) {
      decode_smem[SBASE + sTHR] = 2 * tx + 1;
    }
  }
  __syncthreads();

  const int threshold = decode_smem[SBASE + sTHR];

  // ---- Phase 2: Collection with warp-aggregated atomicAdds ----
  int* bufs[2] = {decode_smem + BOFF, decode_smem + BOFF + DBUF};
  const int sOUT_abs = SBASE + sOUT;
  const int sBUF0_abs = SBASE + sBUF0;

  {
    const uint32_t uthr = static_cast<uint32_t>(threshold);
    int item = 0;
    const int n_vec_iters = (n_vec + kThreadsPerBlock - 1) / kThreadsPerBlock;

    for (int iter = 0; iter < n_vec_iters; iter++) {
      const int i = tx + iter * kThreadsPerBlock;
      const bool vec_valid = (i < n_vec);
      const int base_idx = i << 2;

#pragma unroll 4
      for (int sub = 0; sub < 4; sub++) {
        const int elem_idx = base_idx + sub;
        uint32_t bin = 0;
        if (vec_valid) bin = reg_bins[item++];
        const bool is_above = vec_valid && (bin > uthr);
        const bool is_equal = vec_valid && (bin == uthr);

        const uint32_t above_mask = __ballot_sync(0xffffffff, is_above);
        if (above_mask) {
          const int above_count = __popc(above_mask);
          const int above_rank = __popc(above_mask & ((1u << lane) - 1));
          int above_base;
          if (lane == 0) {
            above_base = atomicAdd(&decode_smem[sOUT_abs], above_count);
          }
          above_base = __shfl_sync(0xffffffff, above_base, 0);
          if (is_above) {
            output_indices[above_base + above_rank] = elem_idx;
          }
        }

        const uint32_t equal_mask = __ballot_sync(0xffffffff, is_equal);
        if (equal_mask) {
          const int equal_count = __popc(equal_mask);
          const int equal_rank = __popc(equal_mask & ((1u << lane) - 1));
          int equal_base;
          if (lane == 0) {
            equal_base = atomicAdd(&decode_smem[sBUF0_abs], equal_count);
          }
          equal_base = __shfl_sync(0xffffffff, equal_base, 0);
          if (is_equal && __builtin_expect(equal_base + equal_rank < DBUF, 1)) {
            bufs[0][equal_base + equal_rank] = elem_idx;
          }
        }
      }
    }
  }
  __syncthreads();

  int remaining_k = TopK - decode_smem[SBASE + sOUT];
  if (remaining_k <= 0) return false;

  // If all buffered elements fit, output them all (common for short seqs)
  const int raw_buf0 = decode_smem[SBASE + sBUF0];
  if (raw_buf0 <= remaining_k) {
    const int nb = (raw_buf0 < DBUF) ? raw_buf0 : DBUF;
    const int base = decode_smem[SBASE + sOUT];
    for (int i = tx; i < nb; i += kThreadsPerBlock) {
      output_indices[base + i] = bufs[0][i];
    }
    __syncthreads();
    return false;
  }

  // ---- Phase 3: Deferred refinement (rare path) ----
  int* refine[2] = {decode_smem, decode_smem + RHIST};
  const int num_buf0 = (raw_buf0 < DBUF) ? raw_buf0 : DBUF;

  for (int i = tx; i < RHIST; i += kThreadsPerBlock) {
    refine[0][i] = 0;
  }
  __syncthreads();

  for (int i = tx; i < num_buf0; i += kThreadsPerBlock) {
    const uint32_t fp32 = convert_to_uint32_v2(logits[bufs[0][i]]);
    atomicAdd(&refine[0][(fp32 >> 24) & 0xFF], 1);
  }
  __syncthreads();

  auto compute_suffix_sum = [&]() {
#pragma unroll 8
    for (int i = 0; i < 8; ++i) {
      if (tx < RADIX) {
        const int stride = 1 << i;
        const int s = i & 1;
        const int d = s ^ 1;
        int value = refine[s][tx];
        if (tx < RADIX - stride) value += refine[s][tx + stride];
        refine[d][tx] = value;
      }
      __syncthreads();
    }
  };

#pragma unroll 4
  for (int pass = 0; pass < 4; ++pass) {
    const int src = pass & 1;
    const int dst = src ^ 1;

    const int raw_buf = decode_smem[SBASE + sBUF0 + src];

    // ---- Decode truncation, detected ----
    //
    // `DBUF` is 3,708 and the Phase-2 append guard drops everything past it, but
    // the atomicAdd that hands out slots sits outside that guard, so `raw_buf`
    // keeps counting and a post-hoc test here is exact, as for the medium path.
    // Dropped candidates are invisible to refinement and to the deterministic
    // tie ranking further down: ranking orders the survivors, it cannot recall
    // a winner that was never buffered. The failing class: an 8,192-element row
    // inside one coarse bin with one element larger at FP32 but in the same
    // half-precision bin; a legal append order drops that winner, so ranking
    // the truncated set would be silently wrong. Counter [10] counts every
    // overflowing row once. With exact_decode on (the compiled policy,
    // exact_decode_enabled() in topk.cu) the row is handed back and
    // medium_exact_restart re-selects it from a full-row rescan with
    // ascending-index emission, so the row is exact and deterministic and a
    // nonzero counter [10] is the repair rate, not an error. A restart that
    // cannot complete counts in [3] (shared with the medium path), fills the
    // row with -1 and traps unless MERLIN_TOPK_EXACT_TRAP is 0. Only with
    // exact_decode off is an overflowing row ranked from the truncated set.
    if (__builtin_expect(raw_buf > DBUF, 0)) {
      if (tx == 0 && decode_smem[SBASE + sOVF] == 0) {
        decode_smem[SBASE + sOVF] = 1;  // once per row, not once per pass
        if (counters != nullptr) atomicAdd(&counters[kCtrDecodeOverflow], 1ull);
      }
      if (exact_decode) {
        // Hand the row back for the same repair the medium path takes:
        // throw the buffered state away and re-select from scratch with a
        // full-row radix rescan.  The restart writes all TopK slots (or
        // fills the row with -1 and faults), so Phase 2's `> threshold`
        // emissions are overwritten rather than double-counted.  `raw_buf`
        // is block-uniform and `exact_decode` is a launch constant, so this
        // return is taken by the whole block -- which is what makes it safe
        // to leave a function that still owns the shared-memory window.
        __syncthreads();
        return true;
      }
      __syncthreads();
    }

    const int num_buffered = (raw_buf < DBUF) ? raw_buf : DBUF;

    compute_suffix_sum();

    if (tx < RADIX && refine[0][tx] > remaining_k &&
        refine[0][tx + 1] <= remaining_k) {
      decode_smem[SBASE + sREF] = tx;
      decode_smem[SBASE + sBUF0 + dst] = 0;
      decode_smem[SBASE + sFIN] = remaining_k - refine[0][tx + 1];
    }
    __syncthreads();

    const int ref_thr = decode_smem[SBASE + sREF];
    // Read before any thread decrements sFIN, so every thread agrees on
    // how many of the tied candidates survive the cutoff.
    const int fin_total = decode_smem[SBASE + sFIN];
    remaining_k -= refine[0][ref_thr + 1];
    const int bit_offset = 24 - pass * 8;

    if (remaining_k == 0) {
      for (int i = tx; i < num_buffered; i += kThreadsPerBlock) {
        const int idx = bufs[src][i];
        const uint32_t fp32 = convert_to_uint32_v2(logits[idx]);
        if (((fp32 >> bit_offset) & 0xFF) > static_cast<uint32_t>(ref_thr)) {
          const int pos = atomicAdd(&decode_smem[SBASE + sOUT], 1);
          output_indices[pos] = idx;
        }
      }
      __syncthreads();
      break;
    }

    __syncthreads();
    if (tx < RADIX + 1) refine[0][tx] = 0;
    __syncthreads();

    for (int i = tx; i < num_buffered; i += kThreadsPerBlock) {
      const int idx = bufs[src][i];
      const float logit_val = logits[idx];
      const uint32_t fp32 = convert_to_uint32_v2(logit_val);
      const int bin = (fp32 >> bit_offset) & 0xFF;

      if (bin > ref_thr) {
        const int pos = atomicAdd(&decode_smem[SBASE + sOUT], 1);
        output_indices[pos] = idx;
      } else if (bin == ref_thr) {
        if (pass == 3) {
          if (det_ties) {
            // Every candidate that reaches here has an identical 32-bit value,
            // and the racy branch below hands out slots by atomicAdd -- first
            // warp to arrive wins, so which tied block survives the cutoff is
            // a race. Rank by element index instead and place at a position
            // derived from that rank, so the surviving set is arrival-order
            // independent. No atomics on this path.
            //
            // O(t^2) over the exactly-tied set t. Exact 32-bit ties are rare,
            // so t is small in practice, but that is not a bound.
            int rank = 0;
            for (int j = 0; j < num_buffered; ++j) {
              const int jdx = bufs[src][j];
              const bool ahead = tie_recent ? (jdx > idx) : (jdx < idx);
              if (!ahead) continue;
              const uint32_t jf = convert_to_uint32_v2(logits[jdx]);
              if (((jf >> bit_offset) & 0xFF) == static_cast<uint32_t>(ref_thr)) {
                ++rank;
              }
            }
            if (rank < fin_total) {
              output_indices[TopK - fin_total + rank] = idx;
            }
          } else {
            const int slot = atomicAdd(&decode_smem[SBASE + sFIN], -1);
            if (slot > 0) output_indices[TopK - slot] = idx;
          }
        } else {
          const int bp = atomicAdd(&decode_smem[SBASE + sBUF0 + dst], 1);
          if (__builtin_expect(bp < DBUF, 1)) {
            bufs[dst][bp] = idx;
            const int nbo = bit_offset - 8;
            atomicAdd(&refine[0][(fp32 >> nbo) & 0xFF], 1);
          }
        }
      }
    }
    __syncthreads();
  }
  return false;
}

// ============================================================================
// Medium-path exact top-k restart
//
// The medium path buffers the coarse threshold bin's candidates into a fixed
// 4,096-slot shared buffer and increments the NEXT-LEVEL histogram inside the
// capacity guard, so a candidate dropped on overflow is invisible to
// refinement in both the buffer and the histogram. The returned set is then
// non-exact on overflowing rows, and which candidates survive is decided by
// atomicAdd arrival order.
//
// The repair is the mechanism of vLLM PR #52149 with a different emission
// rule: detect the overflow, then throw away everything the buffered path did
// and re-select from scratch (`prefix = 0`, `remaining = TopK`) with a
// full-row radix rescan. Because the restart starts from an empty prefix and
// an undebited `remaining`, the coarse pass's winners are re-counted rather
// than double-counted.
//
// Where this diverges from #52149: its rescan emits by raw `atomicAdd` arrival
// order in both the `> pivot` and `== pivot` phases, which is set-exact but
// order- and tie-membership-nondeterministic. Sparse attention sums the
// selected keys in output order, so that would re-introduce the
// nondeterminism the deterministic tie policy exists to remove. This restart
// emits in **ascending index order** via a block scan (no atomics anywhere on
// the placement path).
//
// Tie direction is one policy, decided centrally: `effective_tie_recent()`
// (above) resolves it as `det_ties ? tie_recent : true`, and the dispatch site
// passes the result down, so the medium-path restart and the buffered
// decode/medium paths agree on it.
//
// The deterministic large path takes the same value.
// `radix_emit_deterministic` receives `effective_tie_recent(det_ties,
// tie_recent)` from the dispatch site and mirrors both halves of its cut -- the
// cross-CTA walk and the within-chunk window -- so all five paths
// (decode-buffered, decode-restart, medium-buffered, medium-restart, large)
// obey one decision and the host's policy line prints it for all five.
// ============================================================================

// Abort the kernel when the exact fallback cannot complete, instead of
// returning a truncated row.  Compile-time so it costs nothing when off; the
// counters below record the fault either way, so a build with the trap off is
// still never silently wrong.
#ifndef MERLIN_TOPK_EXACT_TRAP
  #define MERLIN_TOPK_EXACT_TRAP 1
#endif

// ============================================================================
// DIAGNOSTIC BUILDS ONLY -- never set in a release or serving build.
//
// The compile-time defines below build the diagnostic variants of the
// large-path fault cases. Every one defaults to 0 and every use sits behind
// `#if`, so the default build's device code does not include them. A build
// carrying any of them prints a
// `[merlin] persistent_topk DIAGNOSTIC BUILD` line on its first launch
// (topk.cu), which a startup-warning scan must reject.
//
//   MERLIN_TOPK_DIAG_DELAY=<cycles>
//       In `radix_topk`'s round loop, every NON-LEADER warp (warp != 0) of
//       every NON-LEADER CTA (cta_in_group != 0) spins <cycles> clock64 ticks
//       before merging its shared bins into the group's global histogram.
//       Warp 0 holds thread 0, the arrival publisher, so it is never delayed:
//       the delay opens exactly the window the histogram-publication barrier
//       closes, a publisher that announces arrival before its own CTA's other
//       warps have produced. CTA 0 is never delayed, so it reads the histogram
//       as soon as the arrival count completes.
//   MERLIN_TOPK_DIAG_PREREPAIR=1
//       Compile out the histogram-publication barrier.  This is the
//       negative control for the delayed-publication stress: with a delay it
//       is expected to return wrong or faulted rows; the normal build with
//       the same delay must not.  Only meaningful with MERLIN_TOPK_DIAG_DELAY.
//   MERLIN_TOPK_DIAG_FORCE_FAULT=1  (+ MERLIN_TOPK_DIAG_FORCE_FAULT_CTA, 1)
//       Force the CTA-local X3 predicate in that CTA on every deterministic
//       large row, as `fault_inject_cta` does -- but WITHOUT
//       suppressing `__trap()`.  The trap-ON fault arm: the process's CUDA
//       context dies.  Run only in a disposable subprocess.
// ============================================================================
#ifndef MERLIN_TOPK_DIAG_DELAY
  #define MERLIN_TOPK_DIAG_DELAY 0
#endif
#ifndef MERLIN_TOPK_DIAG_PREREPAIR
  #define MERLIN_TOPK_DIAG_PREREPAIR 0
#endif
#ifndef MERLIN_TOPK_DIAG_FORCE_FAULT
  #define MERLIN_TOPK_DIAG_FORCE_FAULT 0
#endif
#ifndef MERLIN_TOPK_DIAG_FORCE_FAULT_CTA
  #define MERLIN_TOPK_DIAG_FORCE_FAULT_CTA 1
#endif
#define MERLIN_TOPK_DIAG_ANY                               \
  ((MERLIN_TOPK_DIAG_DELAY) != 0 || (MERLIN_TOPK_DIAG_PREREPAIR) != 0 || \
   (MERLIN_TOPK_DIAG_FORCE_FAULT) != 0)
#if MERLIN_TOPK_DIAG_FORCE_FAULT && !MERLIN_TOPK_EXACT_TRAP
  #error "MERLIN_TOPK_DIAG_FORCE_FAULT is the trap-ON diagnostic build; it needs MERLIN_TOPK_EXACT_TRAP=1"
#endif
#if MERLIN_TOPK_DIAG_FORCE_FAULT && (MERLIN_TOPK_DIAG_DELAY || MERLIN_TOPK_DIAG_PREREPAIR)
  #error "build the trap-ON and the delayed-publication diagnostics separately"
#endif

#if MERLIN_TOPK_DIAG_DELAY
// Busy-wait on the SM clock.  `volatile` sink so the loop is not elided.
__device__ __noinline__ void merlin_topk_diag_spin(long long cycles) {
  const long long start = clock64();
  volatile long long now = start;
  while (now - start < cycles) {
    now = clock64();
  }
}
#endif

// Exact, bounded-memory re-selection for a single medium-path row.
//
// `scores` is already offset to the row.  Every radix round rescans the whole
// row, so correctness does not depend on how many values share a coarse bin.
// Entered only after an overflow has been detected.
//
// `tie_recent` is the EFFECTIVE tie direction for this launch, resolved once
// on the host (see the header comment above): true keeps the highest element
// indices among the pivot-equal group, false keeps the lowest.  It is a launch
// constant and block-uniform, so every walk and the emission predicate below
// agree on it without a barrier.
template <int TopK>
__device__ __noinline__ void medium_exact_restart(
    const float* __restrict__ scores, int32_t* __restrict__ output,
    uint32_t length, unsigned long long* __restrict__ counters,
    bool tie_recent) {
  static_assert(kThreadsPerBlock >= RADIX,
                "the single-thread bin walk assumes RADIX <= blockDim.x");

  using BlockScan = cub::BlockScan<uint32_t, kThreadsPerBlock>;

  struct ExactSmem {
    uint32_t histogram[RADIX];
    typename BlockScan::TempStorage scan;
    uint32_t prefix;      // ordered-uint radix prefix, grows 8 bits per round
    uint32_t remaining;   // how many of the pivot-equal group must be kept
    uint32_t tie_total;   // how many elements have key == pivot in total
    uint32_t tie_cut;     // pivot-equal cut: lowest kept index under
                          // most-recent-wins, highest kept under oldest-wins
    uint32_t fault;       // set by the walks; block-wide, read after a barrier
  };

  // The medium path's dynamic smem window is kSmemMedium (35,968 B) and the
  // buffered state in it is dead the moment we restart, so we overlay it.
  // Declared with an explicit alignment rather than inheriting the `char`
  // array's alignment of 1 -- cub::BlockScan::TempStorage needs more.
  extern __shared__ __align__(16) char medium_exact_smem[];
  static_assert(alignof(ExactSmem) <= 16, "ExactSmem over-aligned for smem");
  static_assert(sizeof(ExactSmem) <= kSmemMedium,
                "exact restart does not fit the medium smem window");

  auto* smem = reinterpret_cast<ExactSmem*>(medium_exact_smem);
  const uint32_t tx = threadIdx.x;

  if (tx == 0) {
    smem->prefix = 0;
    smem->remaining = static_cast<uint32_t>(TopK);
    smem->tie_total = 0;
    // The cut is a bound on the kept index range, so its identity element is
    // direction-dependent.  When tie_total == equal_count the walks below are
    // skipped entirely and this value is what the emission predicate reads:
    // `idx >= 0` keeps the whole group, and so does `idx <= 0xFFFFFFFF`.
    smem->tie_cut = tie_recent ? 0u : 0xFFFFFFFFu;
    // The index-space tie walk below is two 8-bit rounds, so it resolves
    // indices below 65,536.  The medium path serves seq_len <= 32,768, but
    // fail loudly rather than silently mis-cut if that ever changes.
    smem->fault = (length > 65536u || length < static_cast<uint32_t>(TopK))
                      ? 1u
                      : 0u;
  }
  __syncthreads();

  // ---- four 8-bit radix rounds over the full row -------------------------
#pragma unroll
  for (uint32_t round = 0; round < 4; ++round) {
    // An entry fault (bad length) skips the four full-row rescans.
    // Block-uniform -- the flag is only ever written before a barrier that
    // precedes this read.
    if (smem->fault != 0u) break;
    for (uint32_t bin = tx; bin < RADIX; bin += kThreadsPerBlock) {
      smem->histogram[bin] = 0;
    }
    __syncthreads();

    // Table, not `~0u << (32 - round * 8)`: the round-0 form of that
    // expression is a 32-bit shift of a 32-bit type, which is undefined even
    // though the ternary never selects it at runtime, and the loop is fully
    // unrolled so the compiler does constant-fold the dead branch.
    constexpr uint32_t kPrefixMask[4] = {0u, 0xFF000000u, 0xFFFF0000u,
                                         0xFFFFFF00u};
    const uint32_t shift = 24u - round * 8u;
    const uint32_t prefix_mask = kPrefixMask[round];
    const uint32_t prefix = smem->prefix;

    for (uint32_t idx = tx; idx < length; idx += kThreadsPerBlock) {
      const uint32_t ordered = convert_to_uint32_v2(scores[idx]);
      if ((ordered & prefix_mask) == prefix) {
        atomicAdd(&smem->histogram[(ordered >> shift) & 0xFFu], 1u);
      }
    }
    __syncthreads();

    // Single-threaded descending walk.  #52149's wide-radix variant does this
    // cooperatively and reads `remaining` in every thread before one thread
    // writes it back with no barrier in between (a lost update on a
    // non-atomic shared `|=`); one thread owning both the read and the write
    // removes that class of bug rather than adding the missing barrier.
    if (tx == 0) {
      const uint32_t remaining = smem->remaining;
      uint32_t count_above = 0;
      bool found = false;
      for (int bin = RADIX - 1; bin >= 0; --bin) {
        const uint32_t count = smem->histogram[bin];
        if (count_above + count >= remaining) {
          smem->remaining = remaining - count_above;
          smem->prefix |= static_cast<uint32_t>(bin) << shift;
          if (round == 3) smem->tie_total = count;
          found = true;
          break;
        }
        count_above += count;
      }
      // No crossing bin means the entry invariant (length >= TopK) is broken.
      if (!found) atomicOr(&smem->fault, 1u);
    }
    __syncthreads();
  }

  const uint32_t pivot = smem->prefix;
  const uint32_t equal_count = smem->remaining;
  const uint32_t tie_total = smem->tie_total;

  // `> pivot` holds exactly TopK - equal_count elements, so the selection is
  // TopK elements exactly.  equal_count <= tie_total is an invariant of the
  // walk (count >= remaining - count_above at the crossing bin).
  if (tx == 0 && (equal_count == 0u || equal_count > tie_total)) {
    atomicOr(&smem->fault, 1u);
  }
  __syncthreads();

  // Note: a fault raised before emission short-circuits ALL
  // selection work -- no tie walk, no scan, no candidate index ever written
  // -- so the row goes straight to the -1 fill below.  The condition is
  // block-uniform (shared, read after the barrier above).
  if (smem->fault == 0u) {
    // ---- tie cut at the k-th value --------------------------------------
    //
    // Keep the `equal_count` HIGHEST element indices among the pivot-equal
    // group when `tie_recent`, the `equal_count` LOWEST when not.  Finding the
    // cut index turns the tie rule into a per-element predicate, which is what
    // lets the emission below be atomic-free.
    //
    // The two index walks below are the only code that has
    // to know the direction; they differ from each other only in which end of
    // the index-bin array they start from, so `count_above` means "population
    // on the far side of the walk" in both directions and the crossing test is
    // unchanged.
    //
    // Exact 32-bit ties at the cutoff are rare, so the common case is
    // tie_total == equal_count and the whole group is kept with no extra pass.
    if (tie_total > equal_count) {
      uint32_t need = equal_count;
      uint32_t high_byte = 0;

      // Round A: bucket the pivot-equal indices by their high byte.
      for (uint32_t bin = tx; bin < RADIX; bin += kThreadsPerBlock) {
        smem->histogram[bin] = 0;
      }
      __syncthreads();
      for (uint32_t idx = tx; idx < length; idx += kThreadsPerBlock) {
        if (convert_to_uint32_v2(scores[idx]) == pivot) {
          atomicAdd(&smem->histogram[(idx >> 8) & 0xFFu], 1u);
        }
      }
      __syncthreads();
      if (tx == 0) {
        const int first_bin = tie_recent ? RADIX - 1 : 0;
        const int bin_step = tie_recent ? -1 : 1;
        uint32_t count_above = 0;
        bool found = false;
        for (int n = 0; n < RADIX; ++n) {
          const int bin = first_bin + n * bin_step;
          const uint32_t count = smem->histogram[bin];
          if (count_above + count >= need) {
            smem->tie_cut = static_cast<uint32_t>(bin);  // the high byte
            smem->remaining = need - count_above;        // carried into round B
            found = true;
            break;
          }
          count_above += count;
        }
        if (!found) atomicOr(&smem->fault, 1u);
      }
      __syncthreads();
      high_byte = smem->tie_cut;
      need = smem->remaining;

      // Round B: within that bucket, bucket by the low byte.  Each (high, low)
      // pair names one index, so every count here is 0 or 1 and the walk lands
      // exactly on the need-th largest index.
      for (uint32_t bin = tx; bin < RADIX; bin += kThreadsPerBlock) {
        smem->histogram[bin] = 0;
      }
      __syncthreads();
      for (uint32_t idx = tx; idx < length; idx += kThreadsPerBlock) {
        if (((idx >> 8) & 0xFFu) == high_byte &&
            convert_to_uint32_v2(scores[idx]) == pivot) {
          atomicAdd(&smem->histogram[idx & 0xFFu], 1u);
        }
      }
      __syncthreads();
      if (tx == 0) {
        const int first_bin = tie_recent ? RADIX - 1 : 0;
        const int bin_step = tie_recent ? -1 : 1;
        uint32_t count_above = 0;
        bool found = false;
        for (int n = 0; n < RADIX; ++n) {
          const int bin = first_bin + n * bin_step;
          const uint32_t count = smem->histogram[bin];
          if (count_above + count >= need) {
            smem->tie_cut = (high_byte << 8) | static_cast<uint32_t>(bin);
            found = true;
            break;
          }
          count_above += count;
        }
        if (!found) atomicOr(&smem->fault, 1u);
      }
      __syncthreads();
    }
  }

  // Re-test after the tie walks. The guard above was entered before them, so a
  // fault raised at their bin-walk failure sites is caught here, before
  // emission. No barrier is needed between the two tests: the tie walks write
  // the flag from tx == 0 and each is followed by a barrier inside the block
  // above.
  if (smem->fault == 0u) {
    const uint32_t tie_cut = smem->tie_cut;

    // ---- emission: ascending index order, no atomics ---------------------
    //
    // selected(i) = ordered(i) > pivot ||
    //               (ordered(i) == pivot && (tie_recent ? i >= tie_cut
    //                                                   : i <= tie_cut))
    //
    // is a pure function of the element, so the output slot is a prefix sum
    // over the row rather than an arrival rank.  Tiles run in ascending base
    // and within a tile thread `tx` owns `base + tx`, so output[] comes out
    // strictly ascending in index and bit-reproducible run to run.
    uint32_t base_rank = 0;
    const uint32_t tiles =
        (length + kThreadsPerBlock - 1u) / kThreadsPerBlock * kThreadsPerBlock;
    for (uint32_t base = 0; base < tiles; base += kThreadsPerBlock) {
      const uint32_t idx = base + tx;
      uint32_t flag = 0;
      if (idx < length) {
        const uint32_t ordered = convert_to_uint32_v2(scores[idx]);
        const bool keep_tie = tie_recent ? (idx >= tie_cut) : (idx <= tie_cut);
        flag = (ordered > pivot || (ordered == pivot && keep_tie)) ? 1u : 0u;
      }
      uint32_t rank = 0;
      uint32_t total = 0;
      BlockScan(smem->scan).ExclusiveSum(flag, rank, total);
      if (flag != 0u) {
        const uint32_t pos = base_rank + rank;
        // Bounded on both emission phases -- #52149 bounds
        // `collect_pivot_matches` but not its fused `> pivot` write, so an
        // invariant break there becomes an out-of-bounds store rather than a
        // detected fault.
        if (pos < static_cast<uint32_t>(TopK)) {
          output[pos] = static_cast<int32_t>(idx);
        } else {
          // This is the one fault site with concurrent
          // writers.  Equal stored values do not make a data race a
          // protocol, so make it an atomic.  The tx==0 sites below and
          // above are single-writer and separated by barriers, but they use
          // the same primitive so the flag has one access discipline.
          atomicOr(&smem->fault, 1u);
        }
      }
      base_rank += total;
      __syncthreads();  // BlockScan TempStorage is reused next tile
    }

    if (tx == 0 && base_rank != static_cast<uint32_t>(TopK)) {
      atomicOr(&smem->fault, 1u);
    }
  }
  __syncthreads();

  // Consume the decision into a register and barrier again
  // BEFORE anyone can leave.  The persistent kernel reuses this same dynamic
  // shared-memory window for the next row -- the large-row loader writes into
  // it before its first barrier -- so a fast warp returning here while a slow
  // warp still had to read the shared flag would let the next row's writes
  // decide this row's fault branch.  "Read after a barrier" is uniform only
  // until the storage is reused; this closes that window.  After the second
  // barrier every thread already holds its copy, so no later write can
  // change any thread's decision, and the branch below is register-uniform.
  const uint32_t row_fault = smem->fault;
  __syncthreads();

  // ---- fail loud ---------------------------------------------------------
  if (row_fault != 0u) {
    if (tx == 0 && counters != nullptr) {
      atomicAdd(&counters[kCtrMediumExactFault], 1ull);
    }
    // -1 is the kernel's own padding value and the consumer rejects it
    // (`valid = ... & (token >= 0)`), so a faulted row attends nothing and is
    // visible downstream instead of quietly attending the wrong blocks.
    for (uint32_t i = tx; i < static_cast<uint32_t>(TopK);
         i += kThreadsPerBlock) {
      output[i] = -1;
    }
    __threadfence();
    __syncthreads();
#if MERLIN_TOPK_EXACT_TRAP
    __trap();
#endif
  }
}

// ============================================================================
// Medium path: coarse FP16 histogram + 4-pass FP32 radix refinement
// For sequences 8K < seq_len <= 64K.
// ============================================================================

// Adapted from:
// https://github.com/sgl-project/sglang/blob/v0.5.8/sgl-kernel/csrc/elementwise/topk.cu#L87
// by: DarkSharpness
// which at the same time is an optimized topk kernel copied from tilelang
// kernel
//
// Returns TRUE when the row was handed to medium_exact_restart(), whose
// emission is already ascending (sorted emission skips those rows);
// FALSE when the buffered path placed the row by atomic arrival.  The return
// value is block-uniform (every return below is taken by the whole block).
template <int TopK>
__device__ __noinline__ bool histogram_256_topk(
    const float* __restrict__ logits, int* __restrict__ output_indices,
    int logits_offset, int seq_len, bool exact_medium,
    unsigned long long* __restrict__ counters, bool det_ties,
    bool tie_recent) {
  // No default arguments on purpose: a new call site must state whether it
  // wants the exact path, rather than silently inheriting a default.
  // All shared state lives in dynamic shared memory to avoid static
  // __shared__ storage.
  extern __shared__ char medium_smem[];

  int (*shared_histogram)[RADIX + 128] =
      reinterpret_cast<int (*)[RADIX + 128]>(medium_smem);
  int* medium_scalars = reinterpret_cast<int*>(medium_smem + kMediumHistBytes);
  int& shared_output_count = medium_scalars[0];
  int& shared_threshold_bin = medium_scalars[1];
  int* shared_buffered_count = &medium_scalars[2];
  int& shared_final_k = medium_scalars[4];
  // One extra scalar so an overflowing row is counted once, not once per
  // refinement pass.  kMediumScalarsBytes covers it (see the constant).
  int& shared_overflow_counted = medium_scalars[5];
  int (*buffered_indices)[MAX_BUFFERED_ITEMS] =
      reinterpret_cast<int (*)[MAX_BUFFERED_ITEMS]>(medium_smem +
                                                    kMediumHeaderSize);

  const int thread_id = threadIdx.x;
  int remaining_k = TopK;

  if (thread_id == 0) {
    shared_overflow_counted = 0;
    if (counters != nullptr) atomicAdd(&counters[kCtrMediumRows], 1ull);
  }

  if (thread_id < RADIX + 1) {
    shared_histogram[0][thread_id] = 0;
  }
  __syncthreads();

  for (int idx = thread_id; idx < seq_len; idx += kThreadsPerBlock) {
    const auto bin = convert_to_uint8(logits[idx + logits_offset]);
    atomicAdd(&shared_histogram[0][bin], 1);
  }
  __syncthreads();

  auto compute_cumulative_sum = [&]() {
#pragma unroll 8
    for (int i = 0; i < 8; ++i) {
      if (__builtin_expect(thread_id < RADIX, 1)) {
        const int stride = 1 << i;
        const int src_buffer = i & 1;
        const int dst_buffer = src_buffer ^ 1;
        int value = shared_histogram[src_buffer][thread_id];
        if (thread_id < RADIX - stride) {
          value += shared_histogram[src_buffer][thread_id + stride];
        }
        shared_histogram[dst_buffer][thread_id] = value;
      }
      __syncthreads();
    }
  };

  compute_cumulative_sum();

  if (thread_id < RADIX && shared_histogram[0][thread_id] > remaining_k &&
      shared_histogram[0][thread_id + 1] <= remaining_k) {
    shared_threshold_bin = thread_id;
    // Medium-path restart / guard site M1.  shared_histogram[0] is in suffix-sum
    // form here -- it is what the predicate just above tests -- so
    // hist[t] - hist[t+1] is the EXACT population of the threshold bin, which
    // is exactly what the collection loop below would buffer.  The prediction
    // is precise, not conservative, so no row pays the exact path unnecessarily
    // and no overflowing row escapes it.
    const int threshold_bin_count =
        shared_histogram[0][thread_id] - shared_histogram[0][thread_id + 1];
    const bool will_overflow = threshold_bin_count > MAX_BUFFERED_ITEMS;
    // Sentinel: the refinement loop's entry check reads this and cannot tell
    // a predicted overflow from a measured one, so both take the same path.
    shared_buffered_count[0] =
        (will_overflow && exact_medium) ? MAX_BUFFERED_ITEMS + 1 : 0;
    shared_output_count = 0;
    // Flag only -- the counter is incremented past the early return below.
    // When hist[t+1] == TopK exactly the coarse pass already resolves the row
    // and the threshold bin is never collected, so a large bin population is
    // NOT an overflow and must not be counted as one.
    shared_overflow_counted = will_overflow ? 1 : 0;
  }
  __syncthreads();

  const int threshold_bin = shared_threshold_bin;
  remaining_k -= shared_histogram[0][threshold_bin + 1];

  if (remaining_k == 0) {
    for (int idx = thread_id; idx < seq_len; idx += kThreadsPerBlock) {
      const int bin = convert_to_uint8(logits[idx + logits_offset]);
      if (bin > threshold_bin) {
        const int output_pos = atomicAdd(&shared_output_count, 1);
        output_indices[output_pos] = idx;
      }
    }
    __syncthreads();
    return false;
  }

  // Past the early return the threshold bin WILL be collected, so the M1
  // prediction is a real overflow event.
  if (thread_id == 0 && shared_overflow_counted == 1 && counters != nullptr) {
    atomicAdd(&counters[kCtrMediumOverflow], 1ull);
  }

  __syncthreads();
  if (thread_id < RADIX + 1) {
    shared_histogram[0][thread_id] = 0;
  }
  __syncthreads();

  // On a predicted overflow the collection pass is skipped whole --
  // including its `bin > threshold_bin` emissions.  That matters: the exact
  // restart rewrites the entire row from slot 0, so anything emitted here
  // would be double-counted.  Skipping is also why the detector cannot ship
  // without the restart -- on its own it would emit nothing, which is a
  // fail-loud half, never a silently-truncated one.
  //
  // The entry predicate must be IMMUTABLE, so it does not read
  // `shared_buffered_count[0]`, which the loop body increments: with
  // exact_medium = false (the control arm, where the count starts at 0 and a
  // real overflow pushes it past capacity mid-loop) a late warp could evaluate
  // the predicate after early warps had driven the count over 4096 and skip
  // ALL of its elements, including above-threshold winners, a timing-dependent
  // truncation.
  // `shared_overflow_counted` is the M1 prediction, written once before the
  // barrier above and never touched here; combined with the kernel parameter
  // it is block-uniform and constant for the whole loop.
  const bool skip_collection = exact_medium && (shared_overflow_counted == 1);
  if (__builtin_expect(!skip_collection, 1)) {
    for (int idx = thread_id; idx < seq_len; idx += kThreadsPerBlock) {
      const float logit_value = logits[idx + logits_offset];
      const int bin = convert_to_uint8(logit_value);
      if (bin > threshold_bin) {
        const int output_pos = atomicAdd(&shared_output_count, 1);
        output_indices[output_pos] = idx;
      } else if (bin == threshold_bin) {
        const int buffer_pos = atomicAdd(&shared_buffered_count[0], 1);
        if (__builtin_expect(buffer_pos < MAX_BUFFERED_ITEMS, 1)) {
          buffered_indices[0][buffer_pos] = idx;
          const uint32_t fp32_bits = convert_to_uint32_v2(logit_value);
          const int next_bin = (fp32_bits >> 24) & 0xFF;
          atomicAdd(&shared_histogram[0][next_bin], 1);
        }
      }
    }
  }
  __syncthreads();

#pragma unroll 4
  for (int pass = 0; pass < 4; ++pass) {
    const int src_buffer = pass % 2;
    const int dst_buffer = src_buffer ^ 1;
    const int raw_buffered = shared_buffered_count[src_buffer];
    // Medium-path restart / guard site M2 and the M1 sentinel land here.  The
    // append's atomicAdd sits OUTSIDE the capacity guard, so raw_buffered
    // keeps counting past capacity and a post-hoc test is exact.  pass == 3
    // emits through shared_final_k without buffering, so every buffer filled
    // at passes 0-2 is read back here: there is no last-pass hole.
    if (__builtin_expect(raw_buffered > MAX_BUFFERED_ITEMS, 0)) {
      if (thread_id == 0 && shared_overflow_counted == 0) {
        shared_overflow_counted = 1;
        if (counters != nullptr) atomicAdd(&counters[kCtrMediumOverflow], 1ull);
      }
      if (exact_medium) {
        if (thread_id == 0 && counters != nullptr) {
          atomicAdd(&counters[kCtrMediumExactFallback], 1ull);
        }
        __syncthreads();  // no thread may clobber medium_smem before this
        // The restart takes the same effective direction as the buffered branch
        // below, so crossing the 4,096-candidate capacity does not flip the tie
        // policy.  `det_ties == false` means the buffered branch is the racy
        // atomicAdd rule with no direction at all, and the restart then keeps
        // the medium path's most-recent-wins.
        medium_exact_restart<TopK>(logits + logits_offset, output_indices,
                                   static_cast<uint32_t>(seq_len), counters,
                                   effective_tie_recent(det_ties, tie_recent));
        return true;
      }
      __syncthreads();
    }
    const int num_buffered =
        (raw_buffered < MAX_BUFFERED_ITEMS) ? raw_buffered : MAX_BUFFERED_ITEMS;

    compute_cumulative_sum();

    if (thread_id < RADIX && shared_histogram[0][thread_id] > remaining_k &&
        shared_histogram[0][thread_id + 1] <= remaining_k) {
      shared_threshold_bin = thread_id;
      shared_buffered_count[dst_buffer] = 0;
      shared_final_k = remaining_k - shared_histogram[0][thread_id + 1];
    }
    __syncthreads();

    const int threshold_bin = shared_threshold_bin;
    remaining_k -= shared_histogram[0][threshold_bin + 1];
    const int bit_offset = 24 - pass * 8;

    if (remaining_k == 0) {
      for (int i = thread_id; i < num_buffered; i += kThreadsPerBlock) {
        const int idx = buffered_indices[src_buffer][i];
        const uint32_t fp32_bits =
            convert_to_uint32_v2(logits[idx + logits_offset]);
        const int bin = (fp32_bits >> bit_offset) & 0xFF;
        if (bin > threshold_bin) {
          const int output_pos = atomicAdd(&shared_output_count, 1);
          output_indices[output_pos] = idx;
        }
      }
      __syncthreads();
      break;
    }

    __syncthreads();
    if (thread_id < RADIX + 1) {
      shared_histogram[0][thread_id] = 0;
    }
    __syncthreads();

    // Read before any thread decrements shared_final_k; see the decode
    // path's fin_total for the reason.
    const int fin_total_m = shared_final_k;
    for (int i = thread_id; i < num_buffered; i += kThreadsPerBlock) {
      const int idx = buffered_indices[src_buffer][i];
      const float logit_value = logits[idx + logits_offset];
      const uint32_t fp32_bits = convert_to_uint32_v2(logit_value);
      const int bin = (fp32_bits >> bit_offset) & 0xFF;
      if (bin > threshold_bin) {
        const int output_pos = atomicAdd(&shared_output_count, 1);
        output_indices[output_pos] = idx;
      } else if (bin == threshold_bin) {
        if (pass == 3) {
          if (det_ties) {
            // Same rule as the decode path: every candidate here is bit-equal,
            // so rank by element index rather than by warp arrival order.
            //
            // NOTE the medium-path restart interaction: when exact_medium is on
            // AND this row overflowed the candidate buffer,
            // medium_exact_restart() has already returned and this branch is
            // never reached.  That path is handed this same `tie_recent`, so the
            // medium path has one tie direction either side of the capacity
            // boundary.
            int rank = 0;
            for (int j = 0; j < num_buffered; ++j) {
              const int jdx = buffered_indices[src_buffer][j];
              const bool ahead = tie_recent ? (jdx > idx) : (jdx < idx);
              if (!ahead) continue;
              const uint32_t jf =
                  convert_to_uint32_v2(logits[jdx + logits_offset]);
              if (static_cast<int>((jf >> bit_offset) & 0xFF) == threshold_bin) {
                ++rank;
              }
            }
            if (rank < fin_total_m) {
              output_indices[TopK - fin_total_m + rank] = idx;
            }
          } else {
            const int slot = atomicAdd(&shared_final_k, -1);
            if (slot > 0) {
              output_indices[TopK - slot] = idx;
            }
          }
        } else {
          const int buffer_pos =
              atomicAdd(&shared_buffered_count[dst_buffer], 1);
          if (__builtin_expect(buffer_pos < MAX_BUFFERED_ITEMS, 1)) {
            buffered_indices[dst_buffer][buffer_pos] = idx;
            const int next_bit_offset = bit_offset - 8;
            const int next_bin = (fp32_bits >> next_bit_offset) & 0xFF;
            atomicAdd(&shared_histogram[0][next_bin], 1);
          }
        }
      }
    }
    __syncthreads();
  }
  return false;
}

// ============================================================================
// Inter-CTA sync primitives
// ============================================================================

// `asm volatile` alone is NOT a compiler memory barrier: it stops the statement
// being deleted or duplicated, but says nothing about the surrounding loads and
// stores. Every helper below therefore declares a `"memory"` clobber, and the
// fence is fused into the SAME asm statement as the atomic it orders so that no
// optimizer pass can separate them.
//
// Inline PTX rather than `cuda::atomic_ref` / `__threadfence()`:
// `__threadfence()` compiles to the sequentially-consistent `membar.gl`, which
// is stronger than the `fence.acq_rel.gpu` this protocol needs, on the hottest
// barrier in the kernel; `atomic_ref` would pull libcu++ into a header compiled
// for every arch, and `red` (a reduction with no return value) has no
// `atomic_ref` spelling.
__device__ __forceinline__ int ld_acquire(int* ptr) {
  int state = 0;
#if (__CUDA_ARCH__ >= 700)
  asm volatile("ld.global.acquire.gpu.b32 %0, [%1];\n"
               : "=r"(state)
               : "l"(ptr)
               : "memory");
#else
  asm volatile("ld.cg.global.b32 %0, [%1];\n"
               : "=r"(state)
               : "l"(ptr)
               : "memory");
#endif
  return state;
}

__device__ __forceinline__ void red_release(int* ptr, int val) {
#if (__CUDA_ARCH__ >= 700)
  // One statement: the fence and the arrival add cannot be separated, and the
  // clobber forbids moving any access across either of them.
  asm volatile(
      "fence.acq_rel.gpu;\n"
      "red.relaxed.gpu.global.add.s32 [%0], %1;\n"
      :
      : "l"(ptr), "r"(val)
      : "memory");
#else
  __threadfence();
  atomicAdd(ptr, val);
#endif
}

__device__ __forceinline__ void st_release(int* ptr, int val) {
#if (__CUDA_ARCH__ >= 700)
  asm volatile(
      "fence.acq_rel.gpu;\n"
      "st.release.gpu.global.b32 [%0], %1;\n"
      :
      : "l"(ptr), "r"(val)
      : "memory");
#else
  __threadfence();
  atomicExch(ptr, val);
#endif
}

__device__ __forceinline__ void wait_ge(int* ptr, int target_val,
                                        int thread_idx) {
  if (thread_idx == 0) {
#pragma unroll 1
    while (ld_acquire(ptr) < target_val) {
    }
  }
  __syncthreads();
}

// ============================================================================
// Large path: multi-CTA radix select for sequences > 64K
//
// Each row is processed by a group of CTAs. Each CTA loads its chunk into
// shared memory as ordered uint32, then participates in 4 rounds of
// coordinated radix select via global-memory histograms and barriers.
// ============================================================================

// ============================================================================
// Deterministic placement and tie membership for the large path
//
// radix_topk's pivot is already order-independent; placement across and
// inside CTAs and the cross-CTA tie cut are not. Here every CTA counts its own
// `> pivot` and `== pivot` elements from shared memory, publishes them into the
// histogram slot round 3 has finished with, and derives its base by an
// exclusive prefix over cta_in_group, so all CTAs agree. The tie cut keeps the
// highest indices under most-recent-wins (the lowest under oldest-wins), and
// emission is a prefix sum over ascending index: the row is strictly
// ascending and bit-reproducible. Faults are fail-closed for the whole group
// (every CTA ORs every verdict before emitting). Cost: four inter-CTA
// barriers per row and no second global pass.
// ============================================================================
template <int TopK>
__device__ __noinline__ void radix_emit_deterministic(
    int32_t* __restrict__ row_output, uint32_t ordered_pivot,
    uint32_t equal_needed, uint32_t my_chunk_start, uint32_t actual_chunk_size,
    uint32_t* local_histogram, uint32_t* suffix_sum, uint32_t* shared_ordered,
    RadixRowState* state, uint32_t cta_in_group, uint32_t ctas_per_group,
    int& barrier_phase, uint32_t radix_iter, uint32_t tx,
    unsigned long long* counters, int fault_inject_cta,
    bool fault_inject_emit, bool tie_recent) {
  constexpr uint32_t kWarpsPerBlock = kThreadsPerBlock / 32u;
  // Staging for the published per-CTA counts.  The host gates the switch on
  // 4 * ctas_per_group <= RADIX, so these regions never collide.
  constexpr uint32_t kEqStage = RADIX / 2u;
  // Two more regions of the same published histogram row
  // carry the per-CTA fault verdicts -- the pre-emission one at [2C, 3C) and
  // the emission-bound one at [3C, 4C).  They are SEPARATE regions on purpose:
  // reusing [2C, 3C) for the second round would let a fast CTA overwrite a word
  // a slow CTA had not yet read.  Each CTA writes only its own word, so nothing
  // has to pre-zero them.  Host gate: 4 * ctas_per_group <= RADIX.
  const uint32_t kFaultStage = 2u * ctas_per_group;
  const uint32_t kEmitStage = 3u * ctas_per_group;

  // suffix_sum scratch map (shared, RADIX words):
  //   [0] gt count  [1] eq count  [2] gt_base  [3] eq_keep_base  [4] cut_rank
  //   [5] group fault verdict  [6] this CTA's emission-bound violation
  //   [7] group emission-bound verdict  [8] ties dropped on this row
  //   [9] my_keep (the pivot-equal window width; with [4] it spells the cut
  //       in BOTH tie directions -- see the tie_recent note above)
  // ---- per-CTA counts, out of the resident chunk -------------------------
  if (tx < 2u) suffix_sum[tx] = 0u;
  if (tx == 6u) suffix_sum[6] = 0u;
  __syncthreads();

  uint32_t my_gt = 0u;
  uint32_t my_eq = 0u;
  for (uint32_t i = tx; i < actual_chunk_size; i += kThreadsPerBlock) {
    const uint32_t ord = shared_ordered[i];
    if (ord > ordered_pivot) {
      my_gt++;
    } else if (ord == ordered_pivot) {
      my_eq++;
    }
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    my_gt += __shfl_down_sync(0xffffffffu, my_gt, off);
    my_eq += __shfl_down_sync(0xffffffffu, my_eq, off);
  }
  if ((tx & 31u) == 0u) {
    // Integer counts: the atomic's arrival order cannot change the sum, so
    // this is a reduction, not a placement.
    if (my_gt > 0u) atomicAdd(&suffix_sum[0], my_gt);
    if (my_eq > 0u) atomicAdd(&suffix_sum[1], my_eq);
  }
  __syncthreads();
  const uint32_t local_gt = suffix_sum[0];
  const uint32_t local_eq = suffix_sum[1];

  // ---- barrier X1: the histogram slot is free ----------------------------
  // Round 3's `current_hist` is state->histogram[(radix_iter*4+3) % 3], i.e.
  // histogram[radix_iter % 3] (4r+3 == r mod 3).  Every CTA copies it into
  // shared memory AFTER round 3's own barrier and there is no inter-CTA
  // barrier between that copy and here, so a fast CTA could otherwise
  // overwrite the slot while a slow CTA is still reading it.  This barrier is
  // the one the slot-reuse argument requires.
  if (tx == 0) {
    red_release(&state->arrival_counter, 1);
  }
  wait_ge(&state->arrival_counter,
          (barrier_phase + 1) * static_cast<int>(ctas_per_group), tx);
  barrier_phase++;
  __syncthreads();

  uint32_t* cta_counts = state->histogram[radix_iter % 3u];
  if (tx == 0) {
    cta_counts[cta_in_group] = local_gt;
    cta_counts[ctas_per_group + cta_in_group] = local_eq;
  }

  // ---- barrier X2: the counts are published ------------------------------
  // red_release issues `fence.acq_rel.gpu` before the arrival add and wait_ge
  // reads with `ld.global.acquire.gpu`, so the two plain stores above are
  // visible at device scope to every CTA that leaves this barrier.  This is
  // the same release/acquire pairing the kernel relies on for CTA-0's
  // plain-store zeroing of `next_hist`.  The argument holds only if the
  // compiler also emits the stores before the fence; the `"memory"` clobber on
  // red_release is what makes that true.
  if (tx == 0) {
    red_release(&state->arrival_counter, 1);
  }
  wait_ge(&state->arrival_counter,
          (barrier_phase + 1) * static_cast<int>(ctas_per_group), tx);
  barrier_phase++;
  __syncthreads();

  // ---- bases and the tie cut ---------------------------------------------
  for (uint32_t c = tx; c < ctas_per_group; c += kThreadsPerBlock) {
    local_histogram[c] = cta_counts[c];
    local_histogram[kEqStage + c] = cta_counts[ctas_per_group + c];
  }
  __syncthreads();

  if (tx == 0) {
    uint32_t gt_base = 0u;
    uint32_t gt_total = 0u;
    for (uint32_t c = 0; c < ctas_per_group; ++c) {
      const uint32_t g = local_histogram[c];
      if (c < cta_in_group) gt_base += g;
      gt_total += g;
    }

    // Tie-determinism + large-path integration: the walk direction IS the tie
    // policy.  Most-recent-wins walks CTAs from the top down, handing each one
    // as many of the `equal_needed` slots as the CTAs above it left unclaimed;
    // oldest-wins is the same walk mirrored, from the bottom up.  Chunks are
    // handed out in ascending index order (`my_chunk_start = cta_in_group *
    // chunk_size`), so "the highest `equal_needed` pivot-equal indices" is
    // exactly "the top CTAs first" and the lowest is exactly the reverse.
    // `tie_recent` is the EFFECTIVE direction, resolved once on the host by
    // `effective_tie_recent(det_ties, tie_recent)` -- the same value the
    // medium restart takes and the same value the host's policy line prints,
    // so the three paths cannot drift.  It is a launch constant and this walk
    // is single-threaded, so the branch costs nothing that matters.  keep[] is
    // written over eq[] in place (eq[c] is read before keep[c] is stored).
    uint32_t eq_seen = 0u;
    if (tie_recent) {
      for (int c = static_cast<int>(ctas_per_group) - 1; c >= 0; --c) {
        const uint32_t e = local_histogram[kEqStage + c];
        const uint32_t want =
            (eq_seen >= equal_needed) ? 0u : (equal_needed - eq_seen);
        local_histogram[kEqStage + c] = (want < e) ? want : e;
        eq_seen += e;
      }
    } else {
      for (uint32_t c = 0; c < ctas_per_group; ++c) {
        const uint32_t e = local_histogram[kEqStage + c];
        const uint32_t want =
            (eq_seen >= equal_needed) ? 0u : (equal_needed - eq_seen);
        local_histogram[kEqStage + c] = (want < e) ? want : e;
        eq_seen += e;
      }
    }
    const uint32_t eq_total = eq_seen;

    uint32_t eq_keep_base = 0u;
    for (uint32_t c = 0; c < cta_in_group; ++c) {
      eq_keep_base += local_histogram[kEqStage + c];
    }
    const uint32_t my_keep = local_histogram[kEqStage + cta_in_group];

    // Invariants.  The radix walk guarantees #{key > pivot} == TopK -
    // equal_needed and equal_needed <= #{key == pivot}, and the two together
    // make the emission a bijection onto [0, TopK).
    //
    // The first two ARE group-uniform -- every CTA derives them from the same
    // published integers in the same fixed order -- but the third,
    // `my_keep > local_eq`, compares a value that came back through global
    // memory against this CTA's own shared-memory count, and a local
    // inconsistency is exactly what it exists to detect.  It can therefore be
    // true in CTA 1 and false in CTA 0, so the verdict is published and ORed
    // across the group below (X3) before ANY CTA emits.  The fourth check bounds
    // this CTA's highest emission slot ahead of time, so the per-element
    // `slot < TopK` test in the loop is a backstop rather than the only guard.
    uint32_t fault = 0u;
    if (gt_total + equal_needed != static_cast<uint32_t>(TopK)) fault = 1u;
    if (equal_needed > eq_total) fault = 1u;
    uint32_t my_keep_checked = my_keep;
    // Test-only: force the CTA-LOCAL predicate in one named CTA, so a fault
    // that originates outside CTA 0 can be observed.  Off (-1) in production;
    // the host resolves it once per process, so it is a launch constant.
    if (fault_inject_cta >= 0 && !fault_inject_emit &&
        static_cast<int>(cta_in_group) == fault_inject_cta) {
      my_keep_checked = local_eq + 1u;
    }
#if MERLIN_TOPK_DIAG_FORCE_FAULT
    // DIAGNOSTIC BUILD ONLY: the same CTA-local predicate, forced at compile
    // time, so the trap below is NOT suppressed (fault_inject_cta stays -1).
    if (cta_in_group == static_cast<uint32_t>(MERLIN_TOPK_DIAG_FORCE_FAULT_CTA)) {
      my_keep_checked = local_eq + 1u;
    }
#endif
    if (my_keep_checked > local_eq) fault = 1u;
    if (gt_base + eq_keep_base + local_gt + my_keep >
        static_cast<uint32_t>(TopK)) {
      fault = 1u;
    }

    suffix_sum[2] = gt_base;
    suffix_sum[3] = eq_keep_base;
    // The kept pivot-equal elements of this chunk are the `my_keep` whose
    // ascending rank within the chunk lies in [cut_rank, cut_rank + my_keep).
    // Most-recent-wins puts that window at the TOP of the chunk
    // (cut_rank = local_eq - my_keep, so the upper bound is vacuous and the
    // emission predicate is byte-for-byte the rule the large path shipped); oldest-wins
    // puts it at the BOTTOM (cut_rank = 0, and the upper bound is what binds).
    suffix_sum[4] =
        tie_recent ? ((my_keep <= local_eq) ? (local_eq - my_keep) : 0u) : 0u;
    suffix_sum[9] = my_keep;
    suffix_sum[8] = (eq_total > equal_needed) ? (eq_total - equal_needed) : 0u;

    // Publish this CTA's verdict.  Own word, so no pre-zeroing is needed.
    cta_counts[kFaultStage + cta_in_group] = fault;
  }

  // ---- barrier X3: the fault verdicts are published ----------------------
  // Unconditional, so the per-CTA barrier count stays group-uniform.
  if (tx == 0) {
    red_release(&state->arrival_counter, 1);
  }
  wait_ge(&state->arrival_counter,
          (barrier_phase + 1) * static_cast<int>(ctas_per_group), tx);
  barrier_phase++;
  __syncthreads();

  // Tie measurement staged in thread 0's register; recorded only after BOTH
  // verdicts (X3 here, X4 after emission) have cleared the row, so a row that
  // an emission-bound fault later fails closed does not count tie cuts.
  uint32_t staged_ties_dropped = 0u;
  if (tx == 0) {
    uint32_t row_fault = 0u;
    for (uint32_t c = 0; c < ctas_per_group; ++c) {
      row_fault |= cta_counts[kFaultStage + c];
    }
    suffix_sum[5] = row_fault;
    staged_ties_dropped = suffix_sum[8];

    if (counters != nullptr && cta_in_group == 0u) {
      atomicAdd(&counters[kCtrLargeDetRows], 1ull);
    }
  }
  __syncthreads();

  const uint32_t gt_base = suffix_sum[2];
  const uint32_t eq_keep_base = suffix_sum[3];
  const uint32_t cut_rank = suffix_sum[4];
  const uint32_t keep_width = suffix_sum[9];
  const uint32_t row_fault = suffix_sum[5];

  // TEST ONLY: force the emission-bound violation in one
  // named CTA.  Block-uniform (both operands are launch constants and
  // `cta_in_group` is block-uniform), so it changes no control flow the whole
  // block does not take together.  It deliberately fires for EVERY emitting
  // thread of that CTA rather than one: the failure it models is several
  // threads reaching the same write to `suffix_sum[6]` at once.
  const bool force_emit_fault =
      fault_inject_emit && fault_inject_cta >= 0 &&
      static_cast<int>(cta_in_group) == fault_inject_cta;

  // ---- emission: rank by index, no atomics -------------------------------
  if (row_fault == 0u) {
    // Scratch for the tile scan.  The staged per-CTA counts above are dead by
    // now (thread 0 consumed them before the barrier), so local_histogram is
    // free again; 33 slots each at 0 and at 64 stay inside RADIX.
    uint32_t* warp_gt = local_histogram;
    uint32_t* warp_eq = local_histogram + 64;
    const uint32_t lane = tx & 31u;
    const uint32_t warp = tx >> 5;
    uint32_t gt_run = 0u;
    uint32_t eq_run = 0u;

    // Block-uniform trip count: every thread runs every tile, so the ballots
    // and the barriers inside are reached by the whole block.
    for (uint32_t base = 0u; base < actual_chunk_size;
         base += kThreadsPerBlock) {
      const uint32_t i = base + tx;
      const bool in_range = (i < actual_chunk_size);
      const uint32_t ord = in_range ? shared_ordered[i] : 0u;
      const bool f_gt = in_range && (ord > ordered_pivot);
      const bool f_eq = in_range && (ord == ordered_pivot);

      const uint32_t bgt = __ballot_sync(0xffffffffu, f_gt);
      const uint32_t beq = __ballot_sync(0xffffffffu, f_eq);
      const uint32_t lane_mask = (1u << lane) - 1u;
      const uint32_t gt_in_warp = static_cast<uint32_t>(__popc(bgt & lane_mask));
      const uint32_t eq_in_warp = static_cast<uint32_t>(__popc(beq & lane_mask));
      if (lane == 0u) {
        warp_gt[warp] = static_cast<uint32_t>(__popc(bgt));
        warp_eq[warp] = static_cast<uint32_t>(__popc(beq));
      }
      __syncthreads();

      if (tx < 32u) {
        const uint32_t cg = (tx < kWarpsPerBlock) ? warp_gt[tx] : 0u;
        const uint32_t ce = (tx < kWarpsPerBlock) ? warp_eq[tx] : 0u;
        uint32_t sg = cg;
        uint32_t se = ce;
#pragma unroll
        for (int off = 1; off < 32; off <<= 1) {
          const uint32_t ng = __shfl_up_sync(0xffffffffu, sg, off);
          const uint32_t ne = __shfl_up_sync(0xffffffffu, se, off);
          if (tx >= static_cast<uint32_t>(off)) {
            sg += ng;
            se += ne;
          }
        }
        warp_gt[tx] = sg - cg;  // inclusive -> exclusive
        warp_eq[tx] = se - ce;
        if (tx == 31u) {
          warp_gt[32] = sg;  // tile totals (lanes >= kWarpsPerBlock add 0)
          warp_eq[32] = se;
        }
      }
      __syncthreads();

      const uint32_t gt_rank = gt_run + warp_gt[warp] + gt_in_warp;
      const uint32_t eq_rank = eq_run + warp_eq[warp] + eq_in_warp;
      const uint32_t tile_gt = warp_gt[32];
      const uint32_t tile_eq = warp_eq[32];

      if (f_gt || (f_eq && eq_rank >= cut_rank &&
                   eq_rank - cut_rank < keep_width)) {
        // Selected elements before this one, in ascending index order: all
        // `> pivot` ones, plus the pivot-equal ones already inside the cut
        // window.  The clamp to `keep_width` is what makes the expression work
        // in BOTH directions: under most-recent-wins the window ends at the
        // last pivot-equal element of the chunk, so `eq_rank - cut_rank` can
        // never reach `keep_width` and the clamp is dead; under oldest-wins
        // the window ends early and every pivot-equal element after it must
        // stop contributing, or a later `> pivot` element is placed past the
        // end of the row.
        uint32_t kept_eq_before =
            (eq_rank > cut_rank) ? (eq_rank - cut_rank) : 0u;
        if (kept_eq_before > keep_width) kept_eq_before = keep_width;
        const uint32_t slot = gt_base + eq_keep_base + gt_rank + kept_eq_before;
        if (slot < static_cast<uint32_t>(TopK) && !force_emit_fault) {
          row_output[slot] = static_cast<int32_t>(my_chunk_start + i);
        } else {
          // An out-of-range slot must not bump a counter and
          // carry on: with trapping ON that would neither trap nor produce the
          // promised all-`-1` row.  It is recorded as a row failure and
          // goes through the SAME policy as the pre-emission faults below: no
          // store here, and the row is filled with -1 after the group agrees.
          //
          // ATOMIC, not a plain store.  Every thread whose slot is out of range
          // reaches this line, and an invariant failure is precisely the
          // situation in which several of them do at once, so a plain store
          // would be a concurrent non-atomic write.  It is a shared-memory word
          // and the flag is monotonic, so `atomicOr` suffices; the healthy path
          // never executes it.  The medium restart uses atomicOr for the same
          // job.
          atomicOr(&suffix_sum[6], 1u);
        }
      }

      __syncthreads();  // before warp_gt / warp_eq are rewritten
      gt_run += tile_gt;
      eq_run += tile_eq;
    }
    // No thread may start the next row's chunk load -- which writes
    // shared_ordered -- until every warp has finished reading it here.
    __syncthreads();
  }

  // ---- barrier X4: the emission verdicts are published -------------------
  // Unconditional (both arms of the branch above reach it), so the per-CTA
  // barrier count stays group-uniform.  On the healthy path this costs one
  // inter-CTA barrier per large row and nothing else: every word written here
  // is this CTA's own.
  if (tx == 0) {
    cta_counts[kEmitStage + cta_in_group] = suffix_sum[6];
    red_release(&state->arrival_counter, 1);
  }
  wait_ge(&state->arrival_counter,
          (barrier_phase + 1) * static_cast<int>(ctas_per_group), tx);
  barrier_phase++;
  __syncthreads();

  if (tx == 0) {
    uint32_t emit_fault = 0u;
    for (uint32_t c = 0; c < ctas_per_group; ++c) {
      emit_fault |= cta_counts[kEmitStage + c];
    }
    suffix_sum[7] = emit_fault;
    // Only meaningful on a healthy row: on a faulted one the tie arithmetic
    // is what is under suspicion, so do not record it as a measurement.
    if (counters != nullptr && cta_in_group == 0u && row_fault == 0u &&
        emit_fault == 0u && staged_ties_dropped > 0u) {
      atomicAdd(&counters[kCtrLargeTiesCut], 1ull);
      atomicAdd(&counters[kCtrLargeTiesDropped],
                static_cast<unsigned long long>(staged_ties_dropped));
    }
  }
  __syncthreads();

  // ---- fail closed, for the whole group ----------------------------------
  // Either verdict is now group-uniform, so every CTA agrees on whether this
  // row failed, and CTA 0 can overwrite the row alone -- including the case
  // where some other CTA had already emitted before an out-of-range slot was
  // discovered.  -1 is the kernel's own padding value and the consumer rejects
  // it, so a faulted row attends nothing instead of quietly attending the
  // wrong blocks.  The counter is incremented exactly ONCE per failed row.
  if (row_fault != 0u || suffix_sum[7] != 0u) {
    if (cta_in_group == 0u) {
      for (uint32_t i = tx; i < static_cast<uint32_t>(TopK);
           i += kThreadsPerBlock) {
        row_output[i] = -1;
      }
      if (tx == 0 && counters != nullptr) {
        atomicAdd(&counters[kCtrLargeFault], 1ull);
      }
    }
    __threadfence();
    __syncthreads();
#if MERLIN_TOPK_EXACT_TRAP
    // Fault injection is a test mode and deliberately suppresses the trap:
    // the point of injecting is to OBSERVE the trap-off policy (all-`-1` row,
    // counter += 1), which a context-killing __trap() would make unobservable
    // without a second build.
    if (fault_inject_cta < 0) {
      __trap();
    }
#endif
  }
}

// ============================================================================
// Multi-CTA cooperative RadixTopK for a single large row.
// Adapted from https://github.com/flashinfer-ai/flashinfer/pull/2215
// ============================================================================

template <int TopK, uint32_t VEC_SIZE>
__device__ void radix_topk(const float* __restrict__ row_input,
                           int32_t* __restrict__ row_output, uint32_t seq_len,
                           uint32_t my_chunk_start, uint32_t chunk_size,
                           uint32_t* local_histogram, uint32_t* suffix_sum,
                           uint32_t* shared_scalars, uint32_t* shared_ordered,
                           RadixRowState* state, uint32_t cta_in_group,
                           uint32_t ctas_per_group, int& barrier_phase,
                           uint32_t radix_iter, uint32_t tx, bool det_large,
                           unsigned long long* counters, int fault_inject_cta,
                           bool fault_inject_emit, bool tie_recent,
                           bool sorted_emit) {
  const uint32_t my_chunk_end = (my_chunk_start + chunk_size < seq_len)
                                    ? my_chunk_start + chunk_size
                                    : seq_len;
  const uint32_t actual_chunk_size =
      (my_chunk_start < seq_len) ? (my_chunk_end - my_chunk_start) : 0;

  // -- Stage 1: Load chunk to shared memory as ordered uint32 --
  {
    const uint32_t aligned_size = (actual_chunk_size / VEC_SIZE) * VEC_SIZE;

    for (uint32_t i = tx * VEC_SIZE; i < aligned_size;
         i += kThreadsPerBlock * VEC_SIZE) {
      const float* src = row_input + my_chunk_start + i;
      if constexpr (VEC_SIZE == 4) {
        float4 v = *reinterpret_cast<const float4*>(src);
        shared_ordered[i] = convert_to_uint32_v2(v.x);
        shared_ordered[i + 1] = convert_to_uint32_v2(v.y);
        shared_ordered[i + 2] = convert_to_uint32_v2(v.z);
        shared_ordered[i + 3] = convert_to_uint32_v2(v.w);
      } else if constexpr (VEC_SIZE == 2) {
        float2 v = *reinterpret_cast<const float2*>(src);
        shared_ordered[i] = convert_to_uint32_v2(v.x);
        shared_ordered[i + 1] = convert_to_uint32_v2(v.y);
      } else {
        shared_ordered[i] = convert_to_uint32_v2(*src);
      }
    }
    for (uint32_t i = aligned_size + tx; i < actual_chunk_size;
         i += kThreadsPerBlock) {
      shared_ordered[i] = convert_to_uint32_v2(row_input[my_chunk_start + i]);
    }
  }
  __syncthreads();

  // -- Init radix select state --
  if (tx == 0) {
    shared_scalars[0] = 0;     // prefix
    shared_scalars[1] = TopK;  // remaining_k
  }
  __syncthreads();

  // -- Initial barrier --
  if (tx == 0) {
    red_release(&state->arrival_counter, 1);
  }
  wait_ge(&state->arrival_counter,
          (barrier_phase + 1) * static_cast<int>(ctas_per_group), tx);
  barrier_phase++;
  __syncthreads();

  if (cta_in_group == 0 && tx == 0) {
    st_release(&state->output_counter, 0);
  }

  // -- Stage 2: 4 rounds of radix select --
  for (uint32_t round = 0; round < 4; round++) {
    const uint32_t global_round = radix_iter * 4 + round;
    const uint32_t shift = 24 - round * 8;
    const uint32_t prefix = shared_scalars[0];
    const uint32_t remaining_k = shared_scalars[1];

    uint32_t* current_hist = state->histogram[global_round % 3];
    uint32_t* next_hist = state->histogram[(global_round + 1) % 3];

    for (uint32_t i = tx; i < RADIX; i += kThreadsPerBlock) {
      local_histogram[i] = 0;
    }
    __syncthreads();

    for (uint32_t i = tx; i < actual_chunk_size; i += kThreadsPerBlock) {
      uint32_t ordered = shared_ordered[i];
      uint32_t mask = (round == 0) ? 0u : (~0u << (32 - round * 8));
      if ((ordered & mask) == prefix) {
        uint32_t bucket = (ordered >> shift) & 0xFF;
        atomicAdd(&local_histogram[bucket], 1);
      }
    }
    __syncthreads();

#if MERLIN_TOPK_DIAG_DELAY
    // DIAGNOSTIC BUILD ONLY (see the define block): hold back the producers
    // that are NOT the arrival publisher, in every CTA that is not the leader.
    if (cta_in_group != 0u && (tx >> 5) != 0u) {
      merlin_topk_diag_spin(static_cast<long long>(MERLIN_TOPK_DIAG_DELAY));
    }
#endif

    for (uint32_t i = tx; i < RADIX; i += kThreadsPerBlock) {
      if (local_histogram[i] > 0) {
        atomicAdd(&current_hist[i], local_histogram[i]);
      }
    }

    if (cta_in_group == 0) {
      for (uint32_t i = tx; i < RADIX; i += kThreadsPerBlock) {
        next_hist[i] = 0;
      }
    }

    // The arrival publication below is made by thread 0
    // ALONE, and `red_release`'s `fence.acq_rel.gpu` orders only the issuing
    // thread's own prior accesses.  Without this barrier thread 0 could signal
    // arrival while other warps of this CTA had not yet merged their bins into
    // `current_hist`, or (in CTA 0) had not finished zeroing `next_hist`.  A
    // peer CTA that observes the completed arrival count then leaves its own
    // `wait_ge` -- whose `__syncthreads()` is AFTER arrival, so it orders
    // nothing here -- and reads a half-written histogram, which changes the
    // pivot.  The barrier must be between the producing stores and the leader's
    // publication; program order in one thread is not cross-thread ordering.
#if !MERLIN_TOPK_DIAG_PREREPAIR
    __syncthreads();
#endif

    if (tx == 0) {
      red_release(&state->arrival_counter, 1);
    }
    wait_ge(&state->arrival_counter,
            (barrier_phase + 1) * static_cast<int>(ctas_per_group), tx);
    barrier_phase++;
    __syncthreads();

    for (uint32_t i = tx; i < RADIX; i += kThreadsPerBlock) {
      suffix_sum[i] = current_hist[i];
    }
    __syncthreads();

    for (uint32_t stride = 1; stride < RADIX; stride *= 2) {
      uint32_t val = 0;
      if (tx < RADIX) {
        val = suffix_sum[tx];
        if (tx + stride < RADIX) val += suffix_sum[tx + stride];
      }
      __syncthreads();
      if (tx < RADIX) suffix_sum[tx] = val;
      __syncthreads();
    }

    if (tx == 0) {
      shared_scalars[2] = 0;
      shared_scalars[3] = remaining_k;
    }
    __syncthreads();

    if (tx < RADIX) {
      uint32_t count_ge = suffix_sum[tx];
      uint32_t count_gt = (tx + 1 < RADIX) ? suffix_sum[tx + 1] : 0;
      if (count_ge >= remaining_k && count_gt < remaining_k) {
        shared_scalars[2] = tx;
        shared_scalars[3] = remaining_k - count_gt;
      }
    }
    __syncthreads();

    if (tx == 0) {
      shared_scalars[0] = prefix | (shared_scalars[2] << shift);
      shared_scalars[1] = shared_scalars[3];
    }
    __syncthreads();
  }  // end 4 radix rounds

  // -- Count local > pivot elements --
  const uint32_t ordered_pivot = shared_scalars[0];

  // Deterministic placement + tie membership.  Everything below this branch is
  // the baseline tail, which runs only when det_large is off.
  if (det_large) {
    radix_emit_deterministic<TopK>(
        row_output, ordered_pivot, shared_scalars[1], my_chunk_start,
        actual_chunk_size, local_histogram, suffix_sum, shared_ordered, state,
        cta_in_group, ctas_per_group, barrier_phase, radix_iter, tx, counters,
        fault_inject_cta, fault_inject_emit, tie_recent);
    return;
  }

  if (tx == 0) suffix_sum[0] = 0;
  __syncthreads();

  uint32_t my_gt_count = 0;
  for (uint32_t i = tx; i < actual_chunk_size; i += kThreadsPerBlock) {
    if (shared_ordered[i] > ordered_pivot) my_gt_count++;
  }
  for (int offset = 16; offset > 0; offset /= 2) {
    my_gt_count += __shfl_down_sync(0xffffffff, my_gt_count, offset);
  }
  if (tx % 32 == 0 && my_gt_count > 0) {
    atomicAdd(&suffix_sum[0], my_gt_count);
  }
  __syncthreads();
  const uint32_t local_gt_count = suffix_sum[0];

  // -- Stage 3: Collect top-k indices --
  if (tx == 0) {
    local_histogram[0] = 0;
    if (local_gt_count > 0) {
      local_histogram[1] =
          atomicAdd(&state->output_counter, static_cast<int>(local_gt_count));
    }
  }
  __syncthreads();

  for (uint32_t i = tx; i < actual_chunk_size; i += kThreadsPerBlock) {
    if (shared_ordered[i] > ordered_pivot) {
      uint32_t local_pos = atomicAdd(&local_histogram[0], 1);
      int pos = static_cast<int>(local_histogram[1]) + local_pos;
      row_output[pos] = static_cast<int32_t>(my_chunk_start + i);
    }
  }

  if (tx == 0) {
    red_release(&state->arrival_counter, 1);
  }
  wait_ge(&state->arrival_counter,
          (barrier_phase + 1) * static_cast<int>(ctas_per_group), tx);
  barrier_phase++;
  __syncthreads();

  for (uint32_t i = tx; i < actual_chunk_size; i += kThreadsPerBlock) {
    if (shared_ordered[i] == ordered_pivot) {
      int pos = atomicAdd(&state->output_counter, 1);
      if (pos < TopK) {
        row_output[pos] = static_cast<int32_t>(my_chunk_start + i);
      }
    }
  }
  // The tie loop above reads shared_ordered, and the next row this CTA
  // processes -- a large row's Stage-1 chunk load, or a decode/medium row
  // reusing the same dynamic shared memory -- writes it before its first
  // barrier. Without a closing barrier a fast warp can overwrite entries a
  // slow warp is still comparing against the pivot, changing tie MEMBERSHIP,
  // not only order. Every thread of the block reaches this point (the loop
  // bound is block-uniform and there is no early return in the baseline tail),
  // so the barrier is uniform.
  __syncthreads();

  // Sorted emission on the baseline tail (det_large off, or the host's
  // staging-capacity gate): the row was written by EVERY CTA of the group, by
  // arrival order, so CTA 0 may only re-emit it after all of them have
  // finished.  One more inter-CTA barrier -- the same red_release / wait_ge
  // pair as every other barrier here, taken by every CTA of the group because
  // `sorted_emit` is a launch constant, so barrier_phase stays group-uniform.
  // The closing __syncthreads() just above orders each CTA's stores before its
  // thread 0's release fence.  Indices here run past 32,768, so
  // sorted_emit_row() takes its counting-sort fallback.
  if (sorted_emit) {
    if (tx == 0) {
      red_release(&state->arrival_counter, 1);
    }
    wait_ge(&state->arrival_counter,
            (barrier_phase + 1) * static_cast<int>(ctas_per_group), tx);
    barrier_phase++;
    __syncthreads();
    if (cta_in_group == 0) {
      sorted_emit_row<TopK>(row_output);
    }
  }
}

// ============================================================================
// Persistent kernel -- BS<=32, decode/medium/large paths with RadixTopK
// ============================================================================

template <int TopK = 2048, uint32_t VEC_SIZE = 1>
__global__ void __launch_bounds__(kThreadsPerBlock, 2)
    persistent_topk_kernel(PersistentTopKParams params) {
  const uint32_t tx = threadIdx.x;
  extern __shared__ uint8_t smem_raw[];

  // ========================================================================
  // Group mode: multi-CTA groups with static round-robin row assignment.
  // Non-large rows: CTA-0 handles trivial/decode/medium.
  // Large rows: all CTAs in the group cooperate via RadixTopK.
  // ========================================================================
  const uint32_t ctas_per_group = params.ctas_per_group;
  const uint32_t group_id = blockIdx.x / ctas_per_group;
  const uint32_t cta_in_group = blockIdx.x % ctas_per_group;
  const uint32_t num_groups = gridDim.x / ctas_per_group;
  const uint32_t chunk_size = params.chunk_size;

  if (blockIdx.x >= num_groups * ctas_per_group) return;

  // Early exit: non-CTA-0 threads are never needed if no large rows exist
  if (cta_in_group != 0 && params.max_seq_len <= RADIX_THRESHOLD) return;

  uint32_t* local_histogram = reinterpret_cast<uint32_t*>(smem_raw);
  uint32_t* suffix_sum = local_histogram + RADIX;
  uint32_t* shared_scalars = suffix_sum + RADIX;
  uint32_t* shared_ordered =
      reinterpret_cast<uint32_t*>(smem_raw + kFixedSmemLarge);

  // RadixRowState for multi-CTA cooperative radix.
  // Zero-initialization is done host-side via cudaMemsetAsync in topk.cu
  // before launch: that gives a stream-ordered happens-before edge for all
  // CTAs, which an in-kernel init by CTA 0 alone would not provide.
  RadixRowState* state = &params.row_states[group_id];

  int barrier_phase = 0;
  uint32_t radix_iter = 0;
  const uint32_t total_iters = (params.num_rows + num_groups - 1) / num_groups;

  for (uint32_t iter = 0; iter < total_iters; iter++) {
    // Static round-robin: all CTAs in the group implicitly agree on the row
    uint32_t row_idx = group_id + iter * num_groups;
    if (row_idx >= params.num_rows) break;

    // Clamp the row length before any decision is made on it.
    //
    // `lengths` is int32 and is consumed here as uint32, so a negative value
    // (e.g. a padded decode slot whose per-token context length underflowed)
    // would reinterpret as ~4e9 and sail past every threshold below. Any
    // value beyond the row width would also read into the next row.
    //
    // Clamping to max_seq_len additionally keeps this per-row decision
    // consistent with the `cta_in_group != 0` early exit above, which is
    // taken from the host-side scalar: when max_seq_len <= RADIX_THRESHOLD
    // the non-leader CTAs return immediately, so a leader that reached the
    // cooperative radix path would wait on the inter-CTA barrier for peers
    // that no longer exist and spin until the kernel is killed.
    const int32_t raw_len = params.lengths[row_idx];
    const uint32_t row_bound =
        params.stride < params.max_seq_len ? params.stride : params.max_seq_len;
    const uint32_t non_negative_len =
        raw_len > 0 ? static_cast<uint32_t>(raw_len) : 0u;
    const uint32_t seq_len =
        non_negative_len < row_bound ? non_negative_len : row_bound;
    int32_t* row_output = params.output + row_idx * params.top_k;
    const float* row_input = params.input + row_idx * params.stride;

    if (seq_len <= RADIX_THRESHOLD) {
      if (cta_in_group == 0) {
        if (seq_len <= static_cast<uint32_t>(TopK)) {
          // Trivial case: seq_len <= TopK
          for (uint32_t i = tx; i < static_cast<uint32_t>(TopK);
               i += kThreadsPerBlock) {
            row_output[i] = (i < seq_len) ? static_cast<int32_t>(i) : -1;
          }
        } else if (seq_len <= static_cast<uint32_t>(HIST2048_THRESHOLD)) {
          // An overflowing decode row comes back here for the same exact
          // restart the medium path takes.  The return is block-uniform and
          // nothing has been emitted that the restart does not overwrite.  With
          // exact_decode off this is always false.
          const bool decode_needs_exact = histogram_2048_topk<TopK>(
              row_input, row_output, seq_len, params.det_ties,
              params.tie_recent, params.exact_counters, params.exact_decode);
          if (decode_needs_exact) {
            // Already ascending (restart emission): no sorted re-emission.
            medium_exact_restart<TopK>(
                row_input, row_output, seq_len, params.exact_counters,
                effective_tie_recent(params.det_ties, params.tie_recent));
          } else if (params.sorted_emit) {
            // Decode-buffered placement is by atomic arrival.
            sorted_emit_row<TopK>(row_output);
          }
        } else {
          const bool restarted = histogram_256_topk<TopK>(
              row_input, row_output, 0, seq_len, params.exact_medium,
              params.exact_counters, params.det_ties, params.tie_recent);
          if (!restarted && params.sorted_emit) {
            // Medium-buffered placement is by atomic arrival.
            sorted_emit_row<TopK>(row_output);
          }
        }
        // The trivial branch above writes [0, seq_len) then -1 in index
        // order, which is already the canonical order.
      }
      continue;
    }

    const uint32_t my_chunk_start = cta_in_group * chunk_size;
    radix_topk<TopK, VEC_SIZE>(
        row_input, row_output, seq_len, my_chunk_start, chunk_size,
        local_histogram, suffix_sum, shared_scalars, shared_ordered, state,
        cta_in_group, ctas_per_group, barrier_phase, radix_iter, tx,
        params.det_large, params.exact_counters, params.fault_inject_cta,
        params.fault_inject_emit,
        effective_tie_recent(params.det_ties, params.tie_recent),
        params.sorted_emit);
    radix_iter++;
  }
}

// Sorted emission for the atomic host-side fallbacks that bypass
// persistent_topk_kernel when det ties are off (sampled_topk and FilteredTopK,
// which need >= 144 / 128 KiB opt-in smem, and top_k_per_row_decode, for a
// cooperative launch that does not fit).  The op's contract is canonical order
// on EVERY dispatch, so those rows are re-emitted here.  One block per row,
// launched on the same stream right after the fallback, so it is
// stream-ordered after the fallback's stores.
template <int TopK>
__global__ void __launch_bounds__(kThreadsPerBlock)
    sorted_emit_rows_kernel(int32_t* __restrict__ output, uint32_t num_rows) {
  if (blockIdx.x >= num_rows) return;
  sorted_emit_row<TopK>(output + static_cast<size_t>(blockIdx.x) * TopK);
}

// Under det ties, the launch persistent_topk_kernel cannot run deterministically
// (its cooperative group is not co-resident, or 4 * ctas_per_group overflows the
// det_large staging row) takes this instead of an atomic fallback.  One CTA per
// row, no inter-CTA state: four 8-bit radix rounds over the ordered key find the
// k-th key, four over the element index find the tie cut (medium_exact_restart's
// walk, widened to 32-bit indices), then the row is emitted ascending by prefix
// sum.  Same row clamp, trivial branch, key order and tie rule as the persistent
// kernel; the selection is exact, so no arrival order or geometry enters it.
template <int TopK>
__global__ void __launch_bounds__(kThreadsPerBlock)
    exact_rows_topk_kernel(const float* __restrict__ input,
                           const int32_t* __restrict__ lengths,
                           int32_t* __restrict__ output, uint32_t stride,
                           uint32_t row_bound, bool tie_recent) {
  using BlockScan = cub::BlockScan<uint32_t, kThreadsPerBlock>;
  __shared__ uint32_t histogram[RADIX];
  __shared__ typename BlockScan::TempStorage scan;
  __shared__ uint32_t prefix_s;  // radix prefix, grows 8 bits per round
  __shared__ uint32_t need_s;    // elements still to keep inside the prefix

  const uint32_t tx = threadIdx.x;
  const float* scores = input + static_cast<size_t>(blockIdx.x) * stride;
  int32_t* out = output + static_cast<size_t>(blockIdx.x) * TopK;
  const int32_t raw_len = lengths[blockIdx.x];
  const uint32_t non_negative_len = raw_len > 0 ? static_cast<uint32_t>(raw_len) : 0u;
  const uint32_t len = non_negative_len < row_bound ? non_negative_len : row_bound;
  if (len <= static_cast<uint32_t>(TopK)) {
    for (uint32_t i = tx; i < static_cast<uint32_t>(TopK); i += kThreadsPerBlock) {
      out[i] = (i < len) ? static_cast<int32_t>(i) : -1;
    }
    return;
  }

  const auto key = [&](uint32_t i) { return convert_to_uint32_v2(scores[i]); };
  // Histogram byte `shift` of value(i) over the elements with match(i); thread 0
  // walks the bins from the top (or the bottom) to the one holding the
  // need_s-th element and appends that byte to prefix_s.
  const auto radix_round = [&](auto match, auto value, uint32_t shift, bool from_top) {
    for (uint32_t b = tx; b < RADIX; b += kThreadsPerBlock) histogram[b] = 0;
    __syncthreads();
    for (uint32_t i = tx; i < len; i += kThreadsPerBlock) {
      if (match(i)) atomicAdd(&histogram[(value(i) >> shift) & 0xFFu], 1u);
    }
    __syncthreads();
    if (tx == 0) {
      uint32_t beyond = 0;
      for (int n = 0; n < RADIX; ++n) {
        const int b = from_top ? RADIX - 1 - n : n;
        if (beyond + histogram[b] >= need_s) {
          need_s -= beyond;
          prefix_s |= static_cast<uint32_t>(b) << shift;
          break;
        }
        beyond += histogram[b];
      }
    }
    __syncthreads();
  };
  const auto high_mask = [](uint32_t shift) {
    return static_cast<uint32_t>(~0ull << (shift + 8u));
  };

  if (tx == 0) {
    prefix_s = 0;
    need_s = static_cast<uint32_t>(TopK);
  }
  __syncthreads();
  for (uint32_t round = 0; round < 4; ++round) {
    const uint32_t shift = 24u - 8u * round;
    const uint32_t prefix = prefix_s, mask = high_mask(shift);
    radix_round([&](uint32_t i) { return (key(i) & mask) == prefix; }, key,
                shift, true);
  }
  const uint32_t pivot = prefix_s;  // need_s = pivot-equal elements to keep
  __syncthreads();
  if (tx == 0) prefix_s = 0;
  __syncthreads();
  for (uint32_t round = 0; round < 4; ++round) {
    const uint32_t shift = 24u - 8u * round;
    const uint32_t prefix = prefix_s, mask = high_mask(shift);
    radix_round(
        [&](uint32_t i) { return (i & mask) == prefix && key(i) == pivot; },
        [](uint32_t i) { return i; }, shift, tie_recent);
  }
  const uint32_t tie_cut = prefix_s;

  uint32_t emitted = 0;
  for (uint32_t base = 0; base < len; base += kThreadsPerBlock) {
    const uint32_t i = base + tx;
    uint32_t keep = 0;
    if (i < len) {
      const uint32_t k = key(i);
      keep = (k > pivot ||
              (k == pivot && (tie_recent ? i >= tie_cut : i <= tie_cut)))
                 ? 1u
                 : 0u;
    }
    uint32_t rank = 0, total = 0;
    BlockScan(scan).ExclusiveSum(keep, rank, total);
    if (keep != 0u && emitted + rank < static_cast<uint32_t>(TopK)) {
      out[emitted + rank] = static_cast<int32_t>(i);
    }
    emitted += total;
    __syncthreads();  // scan storage is reused next tile
  }
  // Exactly TopK by construction; anything else is a kernel bug, so fail loud.
  if (emitted != static_cast<uint32_t>(TopK)) __trap();
}

}  // namespace persistent

// ============================================================================
// Optimized FilteredTopK -- single CTA per row for bs > 32.
// ============================================================================
namespace filtered_topk {

namespace hist4096 = topk_histogram_4096;

// ============================================================================
// FilteredTopK -- single CTA per row for bs > 32
// Adapted from https://github.com/flashinfer-ai/flashinfer/pull/2215
// ============================================================================

#define FLASHINFER_CUDA_CALL(func, ...) \
  {                                     \
    cudaError_t e = (func);             \
    if (e != cudaSuccess) {             \
      return e;                         \
    }                                   \
  }

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

// Filtered Top-K kernel for ragged sequences.
template <typename DType, typename IdType, int VEC_SIZE, uint32_t MAX_K = 2048,
          bool UsePredicatedShortLoads = false>
__global__ void __launch_bounds__(FILTERED_TOPK_BLOCK_THREADS)
    FilteredTopKUnifiedKernel(const DType* __restrict__ input,
                              IdType* __restrict__ output,
                              const IdType* __restrict__ lengths,
                              uint32_t num_rows, uint32_t top_k,
                              uint32_t max_len) {
  constexpr uint32_t BLOCK_SIZE = FILTERED_TOPK_BLOCK_THREADS;
  constexpr int RADIX = 256;
  constexpr int SMEM_INPUT_SIZE = FILTERED_TOPK_SMEM_INPUT_SIZE;

  const uint32_t bid = blockIdx.x;
  const int tx = threadIdx.x;

  if (bid >= num_rows) return;

  const int length =
      (lengths != nullptr) ? lengths[bid] : static_cast<int>(max_len);
  const DType* score = input + bid * max_len;
  IdType* dst = output + bid * top_k;

  // Trivial case: length <= top_k
  if (length <= static_cast<int>(top_k)) {
    for (int i = tx; i < static_cast<int>(top_k); i += BLOCK_SIZE) {
      dst[i] = (i < length) ? static_cast<IdType>(i) : static_cast<IdType>(-1);
    }
    return;
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
    return;
  }

  // Static shared memory
  alignas(128) __shared__ int s_histogram_buf[2][RADIX + 128];
  alignas(128) __shared__ int s_counter;
  alignas(128) __shared__ int s_threshold_bin_id;
  alignas(128) __shared__ int s_num_input[2];
  alignas(128) __shared__ int s_indices[MAX_K];

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
      __shared__ int s_last_remain;
      const auto r_idx = round % 2;

      const auto _raw_num_input = s_num_input[r_idx];
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
}

// Helper to compute GCD for VEC_SIZE selection
constexpr uint32_t gcd(uint32_t a, uint32_t b) {
  while (b != 0) {
    uint32_t t = b;
    b = a % b;
    a = t;
  }
  return a;
}

// Compute optimal VEC_SIZE (1, 2, 4 or 8) from max_len and dtype.
template <typename DType>
constexpr int ComputeFilteredTopKVecSize(uint32_t max_len) {
  constexpr int MAX_VEC = 16 / sizeof(DType);  // 4 for float32, 8 for fp16/bf16
  // Use GCD to find largest power-of-2 divisor
  const uint32_t g = gcd(max_len, static_cast<uint32_t>(MAX_VEC));
  return static_cast<int>(g);
}

template <typename DType, typename IdType, uint32_t MAX_K = 2048>
cudaError_t FilteredTopKRaggedTransform(const DType* input,
                                        IdType* output_indices,
                                        const IdType* lengths,
                                        uint32_t num_rows, uint32_t top_k_val,
                                        uint32_t max_len,
                                        cudaStream_t stream = 0) {
  constexpr size_t smem_size = FILTERED_TOPK_SMEM_DYNAMIC;
  constexpr int MAX_VEC = 16 / sizeof(DType);

  dim3 grid(num_rows);
  dim3 block(FILTERED_TOPK_BLOCK_THREADS);
  void* args[] = {&input,    &output_indices, &lengths,
                  &num_rows, &top_k_val,      &max_len};

  const int vec_size = ComputeFilteredTopKVecSize<DType>(max_len);

#define DISPATCH_VEC_SIZE(VS)                                                 \
  if (vec_size == VS) {                                                       \
    auto kernel =                                                             \
        FilteredTopKUnifiedKernel<DType, IdType, VS, MAX_K, (VS != MAX_VEC)>; \
    FLASHINFER_CUDA_CALL(cudaFuncSetAttribute(                                \
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));     \
    FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, grid, block, args,   \
                                          smem_size, stream));                \
    return cudaSuccess;                                                       \
  }

  DISPATCH_VEC_SIZE(1)
  DISPATCH_VEC_SIZE(2)
  DISPATCH_VEC_SIZE(4)
  if constexpr (MAX_VEC >= 8) {
    DISPATCH_VEC_SIZE(8)
  }
#undef DISPATCH_VEC_SIZE

  return cudaSuccess;
}

}  // namespace filtered_topk

template <typename DType, typename IdType, uint32_t MAX_K = 2048>
cudaError_t FilteredTopKRaggedTransform(const DType* input,
                                        IdType* output_indices,
                                        const IdType* lengths,
                                        uint32_t num_rows, uint32_t top_k_val,
                                        uint32_t max_len,
                                        cudaStream_t stream = 0) {
  return filtered_topk::FilteredTopKRaggedTransform<DType, IdType, MAX_K>(
      input, output_indices, lengths, num_rows, top_k_val, max_len, stream);
}

}  // namespace merlin

#endif  // PERSISTENT_TOPK_CUH_
