// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Modified by the Merlin project; see NOTICE.
// Persistent TopK kernel for DeepSeek V3 sparse attention indexer.
// See persistent_topk.cuh for kernel implementation.

#include <cuda_runtime.h>
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

#include "ops.h"
#include "torch_utils.h"
#include "merlin/topk/merlin_topk_ops.h"

#ifndef USE_ROCM
  #include "persistent_topk.cuh"
  // The sampled branch (ported from upstream) is its own translation unit
  // (sampled_topk.cu): only its launch interface is visible here.
  #include "merlin/topk/merlin_sampled_topk_launch.h"
#endif

namespace {

#ifndef USE_ROCM

// ---------------------------------------------------------------------------
// Persistent TopK policy. The tie, exact-restart and sorted-emission policies
// are fixed at their shipped values: deterministic ties, most-recent-wins
// direction, exact medium restart, deterministic large path, in-kernel sorted
// emission. No environment variable is read; the kernel still takes each
// policy as a runtime parameter.
//
// This is Merlin's own policy rather than upstream's VLLM_BATCH_INVARIANT, which
// aborts engine startup on GDN models (upstream #42960) and so can never reach
// this kernel on Flash-Next.
// ---------------------------------------------------------------------------
// Read-only mirror of the cached policy values, for
// persistent_topk_toggle_state().
//
// Each getter below caches its value in a function-local static filled once on
// first use. The initialiser also stores the value into the matching atomic
// here (publish_cached), and nothing else writes these atomics, so a mirror
// holds either -1 (not initialised yet) or exactly the cached value. The query
// only loads them and never calls a getter. Host code only.
// ---------------------------------------------------------------------------
std::atomic<int> g_cached_tie_recent{-1};
std::atomic<int> g_cached_tie_det{-1};
std::atomic<int> g_cached_exact_medium{-1};
std::atomic<int> g_cached_det_large{-1};
std::atomic<int> g_cached_exact_decode{-1};
std::atomic<int> g_cached_sorted_emit{-1};

bool publish_cached(std::atomic<int>& mirror, bool value) {
  mirror.store(value ? 1 : 0, std::memory_order_release);
  return value;
}

// Tie direction: most-recent-wins, deterministic. It is also the direction
// medium_exact_restart() implements, so the medium path agrees with itself on
// overflowing rows.
bool tie_recent_enabled() {
  static const bool enabled = publish_cached(g_cached_tie_recent, true);
  return enabled;
}

// Deterministic tie-break at the k-th cutoff.
bool det_ties_enabled() {
  static const bool enabled = publish_cached(g_cached_tie_det, true);
  return enabled;
}

// Medium-path exact restart.
bool exact_medium_enabled() {
  static const bool enabled = publish_cached(g_cached_exact_medium, true);
  return enabled;
}

// Large-path deterministic emission. The kernel keeps the baseline
// `radix_topk` tail behind `if (det_large)`; with det ties on, a launch whose
// ctas_per_group overflows the staging row (det_large_fits in
// launch_persistent_topk) takes exact_rows_topk_kernel instead, so the tail
// only runs with det ties off.
bool det_large_enabled() {
  static const bool enabled = publish_cached(g_cached_det_large, true);
  return enabled;
}

// Decode-path exact restart. A decode row whose buffered candidates
// overflow DBUF takes the medium path's exact restart instead of ranking a
// truncated set, which could drop a true winner. Counter [10] still counts
// every overflowing decode row.
bool exact_decode_enabled() {
  static const bool enabled = publish_cached(g_cached_exact_decode, true);
  return enabled;
}

// ---------------------------------------------------------------------------
// Sorted emission: every persistent_topk row leaves the op in canonical order
// (ascending index, padding last as -1), so qsa_indexer.py can skip its own
// sort. See sorted_emit_row() in persistent_topk.cuh for the contract; the
// selected SET is never touched. qsa_indexer.py asks this cache, through
// toggle-state field 6, after its first launch, so it skips its sort only when
// the kernel that ran says it sorted.
// ---------------------------------------------------------------------------
bool sorted_emit_enabled() {
  static const bool enabled = publish_cached(g_cached_sorted_emit, true);
  return enabled;
}

// ---------------------------------------------------------------------------
// Report the effective tie policy once per process.
//
// One stderr line at the first launch names the policy the kernel applies on
// every path. It is a plain fprintf on the host with no CUDA call, so it is safe
// from a capturing thread, and the `static` initialiser makes it fire exactly
// once per process regardless of how many workers share the log.
// ---------------------------------------------------------------------------
void report_tie_policy_once() {
  static const bool reported = [] {
    const bool det = det_ties_enabled();
    const bool recent =
        merlin::persistent::effective_tie_recent(det, tie_recent_enabled());
    const char* dir = recent ? "most-recent-wins" : "oldest-wins";
    std::fprintf(
        stderr,
        "[merlin] persistent_topk tie policy: MERLIN_TIE_DET=%d "
        "MERLIN_TIE_RECENT=%d MERLIN_TOPK_EXACT_MEDIUM=%d "
        "MERLIN_TOPK_EXACT_DECODE=%d MERLIN_TOPK_DET_LARGE=%d "
        "MERLIN_TOPK_SORTED_EMIT=%d -> "
        "decode-buffered=%s decode-restart=%s medium-buffered=%s "
        "medium-restart=%s large=%s fallbacks=%s emission=%s\n",
        det ? 1 : 0, tie_recent_enabled() ? 1 : 0,
        exact_medium_enabled() ? 1 : 0, exact_decode_enabled() ? 1 : 0,
        det_large_enabled() ? 1 : 0, sorted_emit_enabled() ? 1 : 0,
        det ? dir : "atomic-arrival",
        exact_decode_enabled() ? dir : "n/a (detect-only; counter [10])",
        det ? dir : "atomic-arrival",
        exact_medium_enabled() ? dir : "n/a (exact restart disabled)",
        det_large_enabled() ? dir : "atomic-arrival (non-deterministic)",
        det ? "exact_rows_topk (atomic fallbacks closed)"
            : "sampled/FilteredTopK/top_k_per_row_decode (atomic-arrival)",
        sorted_emit_enabled()
            ? "ascending-in-kernel (Python QSA sort skipped)"
            : "arrival-order on buffered paths (Python QSA sort decides)");
    return true;
  }();
  (void)reported;
}

#if MERLIN_TOPK_DIAG_ANY
// DIAGNOSTIC BUILD ONLY (persistent_topk.cuh, "DIAGNOSTIC BUILDS ONLY"). Say
// so once per process, on stderr, before the first launch, so a diagnostic build
// is recognisable in the log; a startup-warning scan must reject any process
// that prints it.
void report_diag_build_once() {
  static const bool reported = [] {
    std::fprintf(stderr,
                 "[merlin] persistent_topk DIAGNOSTIC BUILD: "
                 "MERLIN_TOPK_DIAG_DELAY=%lld MERLIN_TOPK_DIAG_PREREPAIR=%d "
                 "MERLIN_TOPK_DIAG_FORCE_FAULT=%d "
                 "MERLIN_TOPK_DIAG_FORCE_FAULT_CTA=%d -- NOT FOR SERVING\n",
                 static_cast<long long>(MERLIN_TOPK_DIAG_DELAY),
                 static_cast<int>(MERLIN_TOPK_DIAG_PREREPAIR),
                 static_cast<int>(MERLIN_TOPK_DIAG_FORCE_FAULT),
                 static_cast<int>(MERLIN_TOPK_DIAG_FORCE_FAULT_CTA));
    return true;
  }();
  (void)reported;
}
#endif

// Process-lifetime device counters, one block per device.  The kernel only
// ever atomicAdds into them, so the launch path needs no host sync and the
// values accumulate across cudagraph replays.
//
// Allocation, however, is NOT capture-safe: cudaMalloc during stream capture
// is at best not captured and at worst invalidates the capture.  So:
//   - the allocation only ever happens on a call that is explicitly allowed to
//     allocate, which is a non-capturing launch or the reader op;
//   - a call made *during* capture never allocates and never latches a result,
//     so a capture that happens to be first does not permanently poison the
//     instrument;
//   - the launch path fails loudly if it is capturing with no counters, rather
//     than baking a graph that runs the patched kernel with no instrument at
//     all.  A silently uninstrumented graph is the failure this is here to
//     prevent, so the check is deliberately not a warning.
// Read (and, on the first call, allocate) with
// torch.ops._C.persistent_topk_exact_counters, which is the explicit
// pre-capture init path.
constexpr int kMaxCounterDevices = 16;

enum CounterState : int {
  kCountersUninit = 0,
  kCountersReady = 1,
  kCountersFailed = 2,
};

// Atomics, not plain globals: the fast path reads these without the mutex, and
// a second host thread may be publishing them from the slow path.  Release on
// publish, acquire on read, and the pointer is always published BEFORE the
// state that advertises it.
std::atomic<unsigned long long*> g_exact_counters[kMaxCounterDevices] = {};
std::atomic<int> g_exact_counters_state[kMaxCounterDevices] = {};
std::mutex g_exact_counters_mutex;

// Lookup only.  Never allocates, never blocks, safe from any context.
unsigned long long* get_exact_counters(int device_index) {
  if (device_index < 0 || device_index >= kMaxCounterDevices) return nullptr;
  if (g_exact_counters_state[device_index].load(std::memory_order_acquire) !=
      kCountersReady) {
    return nullptr;
  }
  return g_exact_counters[device_index].load(std::memory_order_acquire);
}

// Allocate, zero, and publish.  MUST NOT be called on a capturing stream, and
// should be called in a quiescent lifecycle phase: it allocates and blocks,
// and neither is safe while any thread in the process is capturing a graph.
// `stream_is_capturing` only proves the CALLING stream is idle of capture, so
// the intended entry point is the reader op called explicitly before capture;
// the non-capturing launch path calls this as a fallback.
unsigned long long* ensure_exact_counters(int device_index, cudaStream_t stream) {
  if (device_index < 0 || device_index >= kMaxCounterDevices) return nullptr;

  const int state =
      g_exact_counters_state[device_index].load(std::memory_order_acquire);
  if (state == kCountersReady) {
    return g_exact_counters[device_index].load(std::memory_order_acquire);
  }
  // A previous attempt failed for real (OOM); do not retry on every launch.
  // Only a call that was actually allowed to allocate can reach this state, so
  // a capture-time call never latches it.
  if (state == kCountersFailed) return nullptr;

  std::lock_guard<std::mutex> lock(g_exact_counters_mutex);
  // Re-check under the lock; another thread may have published meanwhile.
  const int locked_state =
      g_exact_counters_state[device_index].load(std::memory_order_relaxed);
  if (locked_state != kCountersUninit) {
    return (locked_state == kCountersReady)
               ? g_exact_counters[device_index].load(std::memory_order_acquire)
               : nullptr;
  }

  constexpr size_t bytes =
      merlin::persistent::kNumTopKExactCounters * sizeof(unsigned long long);
  unsigned long long* ptr = nullptr;
  if (cudaMalloc(&ptr, bytes) == cudaSuccess && ptr != nullptr) {
    // Host acquire/release orders the HOST state word; it says
    // nothing about the device.  cudaMemset on device memory is asynchronous
    // with respect to the host, so publishing Ready straight after it would
    // let a first launch -- on this or any other stream -- increment counters
    // that had not been zeroed yet, or read uninitialised bytes.  Zero on a
    // known stream and WAIT for it, so initialisation is complete before the
    // pointer becomes visible to any consumer.
    if (cudaMemsetAsync(ptr, 0, bytes, stream) != cudaSuccess ||
        cudaStreamSynchronize(stream) != cudaSuccess) {
      cudaFree(ptr);
      ptr = nullptr;
    }
  } else {
    ptr = nullptr;
  }
  // A failed allocation leaves the error state set; clear it so the launch's
  // own cudaGetLastError() gate does not report this as a kernel failure.
  if (ptr == nullptr) cudaGetLastError();

  g_exact_counters[device_index].store(ptr, std::memory_order_release);
  g_exact_counters_state[device_index].store(
      ptr != nullptr ? kCountersReady : kCountersFailed,
      std::memory_order_release);
  return ptr;
}

// True when the stream is mid-capture, in which case nothing on this path may
// allocate.  A query failure is treated as "capturing": the conservative
// direction is to refuse to allocate, not to risk invalidating a capture.
bool stream_is_capturing(cudaStream_t stream) {
  cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
  if (cudaStreamIsCapturing(stream, &status) != cudaSuccess) {
    cudaGetLastError();  // do not leak this into the launch's error gate
    return true;
  }
  return status != cudaStreamCaptureStatusNone;
}

template <int TopK>
void launch_persistent_topk(const torch::stable::Tensor& logits,
                            const torch::stable::Tensor& lengths,
                            torch::stable::Tensor& output,
                            torch::stable::Tensor& workspace,
                            int64_t max_seq_len) {
  namespace P = merlin::persistent;

  const torch::stable::accelerator::DeviceGuard device_guard(
      logits.get_device_index());
  const int64_t num_rows = logits.size(0);
  const int64_t stride = logits.stride(0);
  const cudaStream_t stream = get_current_cuda_stream();

  static int num_sms = 0;
  static int max_smem_per_block = 0;
  if (num_sms == 0) {
    const cudaDeviceProp* device_prop = get_device_prop();
    num_sms = device_prop->multiProcessorCount;
    max_smem_per_block = device_prop->sharedMemPerBlockOptin;
  }

  // Re-emit every row ascending, on the same stream, for the two fallbacks
  // that bypass persistent_topk_kernel. A plain launch with a fixed
  // configuration, so it is capture-safe.
  const auto sorted_emit_fallback_rows = [&]() {
    if (!sorted_emit_enabled()) return;
    report_tie_policy_once();
    P::sorted_emit_rows_kernel<TopK>
        <<<static_cast<unsigned>(num_rows), P::kThreadsPerBlock,
           P::sorted_emit_smem_bytes<TopK>(), stream>>>(
            output.mutable_data_ptr<int32_t>(),
            static_cast<uint32_t>(num_rows));
  };

  // Fallbacks that settle ties by atomic arrival order: the sampled branch
  // (sampled_topk) and FilteredTopK here, and top_k_per_row_decode /
  // FilteredTopK when the cooperative launch does not fit. Their rows are
  // re-emitted ascending afterwards, because qsa_indexer.py skips its own sort
  // when toggle-state field 6 says the kernel sorted. sm_86 / sm_89 report
  // 99 KiB of opt-in smem, so gate_open() (>= 144 KiB) and the FilteredTopK
  // test (>= 128 KiB) are already false there.
  // The fallbacks are closed whenever the deterministic tie policy is on (the
  // default). Those requests take persistent_topk_kernel, or
  // exact_rows_topk_kernel where it cannot run deterministically.
  const bool allow_atomic_fallbacks = !det_ties_enabled();
  if (allow_atomic_fallbacks &&
      merlin::sampled_topk::gate_open<TopK>(num_rows, max_seq_len,
                                            max_smem_per_block)) {
    namespace S = merlin::sampled_topk;
    cudaError_t status = S::launch<TopK>(
        logits.const_data_ptr<float>(), lengths.const_data_ptr<int32_t>(),
        output.mutable_data_ptr<int32_t>(), stride,
        static_cast<int>(std::min(max_seq_len, logits.size(1))), num_rows,
        stream);
    STD_TORCH_CHECK(status == cudaSuccess,
                    "sampled_topk smem failed: ", cudaGetErrorString(status));
    sorted_emit_fallback_rows();
  } else if (allow_atomic_fallbacks && num_rows > 32 &&
             max_smem_per_block >= 128 * 1024) {
    cudaError_t status =
        merlin::FilteredTopKRaggedTransform<float, int32_t, TopK>(
            logits.const_data_ptr<float>(), output.mutable_data_ptr<int32_t>(),
            lengths.const_data_ptr<int32_t>(), static_cast<uint32_t>(num_rows),
            static_cast<uint32_t>(TopK), static_cast<uint32_t>(stride), stream);
    STD_TORCH_CHECK(status == cudaSuccess,
                    "FilteredTopK failed: ", cudaGetErrorString(status));
    sorted_emit_fallback_rows();
  } else {
    STD_TORCH_CHECK(workspace.is_cuda(), "workspace must be CUDA tensor");
    STD_TORCH_CHECK(
        workspace.scalar_type() == torch::headeronly::ScalarType::Byte,
        "workspace must be uint8");

    int effective_max_smem;
    if (num_rows <= 4) {
      effective_max_smem =
          std::min(max_smem_per_block, static_cast<int>(P::kSmemMedium));
    } else if (num_rows <= 8) {
      constexpr int kSmemCapMedium = 48 * 1024;
      effective_max_smem = std::min(max_smem_per_block, kSmemCapMedium);
    } else {
      effective_max_smem = max_smem_per_block;
    }

    size_t available_for_ordered =
        static_cast<size_t>(effective_max_smem) - P::kFixedSmemLarge;
    uint32_t max_chunk_elements =
        static_cast<uint32_t>(available_for_ordered / sizeof(uint32_t));

    uint32_t vec_size = 1;
    if (stride % 4 == 0)
      vec_size = 4;
    else if (stride % 2 == 0)
      vec_size = 2;

    max_chunk_elements = (max_chunk_elements / vec_size) * vec_size;
    uint32_t min_chunk = vec_size * P::kThreadsPerBlock;
    if (max_chunk_elements < min_chunk) max_chunk_elements = min_chunk;

    uint32_t ctas_per_group =
        (static_cast<uint32_t>(stride) + max_chunk_elements - 1) /
        max_chunk_elements;
    uint32_t chunk_size =
        (static_cast<uint32_t>(stride) + ctas_per_group - 1) / ctas_per_group;
    chunk_size = ((chunk_size + vec_size - 1) / vec_size) * vec_size;
    if (chunk_size > max_chunk_elements) chunk_size = max_chunk_elements;

    size_t smem_size = P::kFixedSmemLarge + chunk_size * sizeof(uint32_t);
    if (smem_size < P::kSmemMedium) smem_size = P::kSmemMedium;

    // Query occupancy for the instantiation that will actually launch;
    // overestimating it deadlocks the cooperative barrier.
    int occupancy = 1;
    cudaError_t occ_err = cudaSuccess;
    if (vec_size == 4) {
      occ_err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &occupancy, P::persistent_topk_kernel<TopK, 4>, P::kThreadsPerBlock,
          smem_size);
    } else if (vec_size == 2) {
      occ_err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &occupancy, P::persistent_topk_kernel<TopK, 2>, P::kThreadsPerBlock,
          smem_size);
    } else {
      occ_err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &occupancy, P::persistent_topk_kernel<TopK, 1>, P::kThreadsPerBlock,
          smem_size);
    }
    STD_TORCH_CHECK(occ_err == cudaSuccess,
                    "persistent_topk occupancy query failed: ",
                    cudaGetErrorString(occ_err));
    if (occupancy < 1) occupancy = 1;

    // The cooperative spin-wait barrier only runs when at least one row hits
    // the radix path (seq_len > RADIX_THRESHOLD). Below that, non-CTA-0 CTAs
    // early-exit, so oversubscription can't deadlock and headroom is wasted.
    const bool needs_cooperative =
        static_cast<uint32_t>(max_seq_len) > P::RADIX_THRESHOLD;

    const uint32_t hw_resident_cap =
        static_cast<uint32_t>(num_sms) * static_cast<uint32_t>(occupancy);
    uint32_t max_resident_ctas = hw_resident_cap;
    if (needs_cooperative) {
      // Reserve one CTA per SM when occupancy allows; fall back to a single
      // CTA when occupancy == 1 (the most deadlock-prone case -- any straggler
      // kernel that takes the only slot on one SM hangs the barrier). Never
      // drop below one full group's worth.
      uint32_t headroom = (occupancy > 1) ? static_cast<uint32_t>(num_sms) : 1u;
      if (max_resident_ctas >= headroom + ctas_per_group) {
        max_resident_ctas -= headroom;
      }
    }
    uint32_t num_groups = std::min(max_resident_ctas / ctas_per_group,
                                   static_cast<uint32_t>(num_rows));
    if (num_groups == 0) num_groups = 1;
    uint32_t total_ctas = num_groups * ctas_per_group;

    // Under det ties, a launch whose cooperative group would not be co-resident,
    // or whose groups overflow the det_large staging row (4 * ctas_per_group
    // <= RADIX, see params.det_large), takes the exact one-CTA-per-row kernel.
    // Both conditions are checked at run time; ordinary long-context shapes
    // need only a few CTAs per group, so this is a safety net.
    const bool overflows_residency = total_ctas > hw_resident_cap;
    const bool det_large_fits =
        4u * ctas_per_group <= static_cast<uint32_t>(P::RADIX);
    if (!allow_atomic_fallbacks && needs_cooperative &&
        (overflows_residency || !det_large_fits)) {
      report_tie_policy_once();
      if (num_rows > 0) {
        P::exact_rows_topk_kernel<TopK>
            <<<static_cast<unsigned>(num_rows), P::kThreadsPerBlock, 0,
               stream>>>(
                logits.const_data_ptr<float>(),
                lengths.const_data_ptr<int32_t>(),
                output.mutable_data_ptr<int32_t>(),
                static_cast<uint32_t>(stride),
                std::min(static_cast<uint32_t>(stride),
                         static_cast<uint32_t>(max_seq_len)),
                P::effective_tie_recent(true, tie_recent_enabled()));
      }
      const cudaError_t ex_err = cudaGetLastError();
      STD_TORCH_CHECK(ex_err == cudaSuccess, "exact_rows_topk failed: ",
                      cudaGetErrorString(ex_err));
      return;
    }

    // If the cooperative launch wouldn't fit, use the generic decode kernel on
    // low-smem devices, else FilteredTopK where its 128 KiB requirement is met.
    if (needs_cooperative && overflows_residency) {
      if (max_smem_per_block < 128 * 1024) {
        const int64_t next_n = lengths.dim() == 2 ? lengths.size(1) : 1;
        top_k_per_row_decode(logits, next_n, lengths, output, num_rows,
                             logits.stride(0), logits.stride(1), TopK);
        sorted_emit_fallback_rows();
        const cudaError_t fb_err = cudaGetLastError();
        STD_TORCH_CHECK(fb_err == cudaSuccess, "sorted emission failed: ",
                        cudaGetErrorString(fb_err));
        return;
      }
      cudaError_t status =
          merlin::FilteredTopKRaggedTransform<float, int32_t, TopK>(
              logits.const_data_ptr<float>(),
              output.mutable_data_ptr<int32_t>(),
              lengths.const_data_ptr<int32_t>(),
              static_cast<uint32_t>(num_rows), static_cast<uint32_t>(TopK),
              static_cast<uint32_t>(stride), stream);
      STD_TORCH_CHECK(status == cudaSuccess, "FilteredTopK fallback failed: ",
                      cudaGetErrorString(status));
      sorted_emit_fallback_rows();
      const cudaError_t fb_err = cudaGetLastError();
      STD_TORCH_CHECK(fb_err == cudaSuccess, "sorted emission failed: ",
                      cudaGetErrorString(fb_err));
      return;
    }

    size_t state_bytes = num_groups * sizeof(P::RadixRowState);
    STD_TORCH_CHECK(workspace.size(0) >= static_cast<int64_t>(state_bytes),
                    "workspace too small, need ", state_bytes, " bytes");

    // Zero the per-group RadixRowState region before every launch.
    //
    // Issued unconditionally so the memset is captured as its own node in the
    // cudagraph, stream-ordered before the persistent_topk_kernel launch. A
    // capture-time host guard on needs_cooperative would leave the memset out of
    // graphs captured with a short max_seq_len, and the workspace state would
    // then accumulate across replays.
    //
    // The reset is required whichever path the kernel takes:
    //   1. arrival_counter accumulates within a launch and is never reset, so
    //      a stale value would satisfy the next call's first wait_ge at once
    //      and break the barrier.
    //   2. An in-kernel init by CTA 0 has no happens-before edge to the other
    //      CTAs' first red_release. cudaMemsetAsync is stream-ordered: the zero
    //      is globally visible before any CTA runs.
    {
      cudaError_t mz_err = cudaMemsetAsync(
          workspace.mutable_data_ptr<uint8_t>(), 0, state_bytes, stream);
      STD_TORCH_CHECK(mz_err == cudaSuccess,
                      "row_states memset failed: ", cudaGetErrorString(mz_err));
    }

    P::PersistentTopKParams params;
    params.input = logits.const_data_ptr<float>();
    params.output = output.mutable_data_ptr<int32_t>();
    params.lengths = lengths.const_data_ptr<int32_t>();
    params.num_rows = static_cast<uint32_t>(num_rows);
    params.stride = static_cast<uint32_t>(stride);
    params.top_k = static_cast<uint32_t>(TopK);
    params.chunk_size = chunk_size;
    params.row_states = reinterpret_cast<P::RadixRowState*>(
        workspace.mutable_data_ptr<uint8_t>());
    params.ctas_per_group = ctas_per_group;
    params.max_seq_len = static_cast<uint32_t>(max_seq_len);
    params.exact_medium = exact_medium_enabled();
    // The per-CTA selected / pivot-equal counts are staged in one 256-slot
    // RadixRowState histogram row (the slot round 3 has just finished reading),
    // 4 * ctas_per_group uint32s. That is why the workspace contract is
    // unchanged, and it is also the only limit: above 64 CTAs per group the
    // deterministic path cannot be expressed there. With det ties on such a
    // launch takes exact_rows_topk_kernel above; with them off it degrades to
    // the baseline tail and says so once.
    params.det_large = det_large_enabled() && det_large_fits;
    // The served kernel never injects a fault and keeps __trap() armed; the
    // MERLIN_TOPK_DIAG_FORCE_FAULT diagnostic build exercises the fault
    // predicates.
    params.fault_inject_cta = -1;
    params.fault_inject_emit = false;
    if (det_large_enabled() && !det_large_fits) {
      static std::atomic<bool> det_large_warned{false};
      if (!det_large_warned.exchange(true)) {
        std::fprintf(stderr,
                     "[vllm] MERLIN_TOPK_DET_LARGE is on but ctas_per_group=%u "
                     "needs %u of %d staging slots; the large path falls back "
                     "to nondeterministic placement at this shape (counter [4] "
                     "stays 0).\n",
                     ctas_per_group, 4u * ctas_per_group, P::RADIX);
      }
    }
    params.det_ties = det_ties_enabled();
    params.tie_recent = tie_recent_enabled();
    params.exact_decode = exact_decode_enabled();
    params.sorted_emit = sorted_emit_enabled();
    report_tie_policy_once();
#if MERLIN_TOPK_DIAG_ANY
    report_diag_build_once();
#endif
    // Allocate only off the capture path; in the ordinary engine lifecycle the
    // eager warmup lands here first, long before capture.  The explicit,
    // quiescent-phase entry point is the reader op.
    const int device_index = static_cast<int>(logits.get_device_index());
    params.exact_counters = stream_is_capturing(stream)
                                ? get_exact_counters(device_index)
                                : ensure_exact_counters(device_index, stream);
    // Unconditional, eager as well as capturing: an unavailable instrument must
    // raise rather than read as healthy zero counts, which is the failure the
    // counters exist to prevent.
    STD_TORCH_CHECK(
        params.exact_counters != nullptr,
        "persistent_topk: the medium-path exact-restart counters are unavailable on "
        "device ", device_index,
        " (not initialised before cudagraph capture, or the one-time "
        "allocation failed). Call "
        "torch.ops._C.persistent_topk_exact_counters(out, 0, 1) once per "
        "device in a quiescent phase before capture.");

  #define LAUNCH_PERSISTENT(TOPK_VAL, VS)                                     \
    do {                                                                      \
      auto kernel = &P::persistent_topk_kernel<TOPK_VAL, VS>;                 \
      cudaError_t err = cudaFuncSetAttribute(                                 \
          kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);    \
      STD_TORCH_CHECK(err == cudaSuccess,                                     \
                      "Failed to set smem: ", cudaGetErrorString(err));       \
      kernel<<<total_ctas, P::kThreadsPerBlock, smem_size, stream>>>(params); \
    } while (0)

    if (vec_size == 4) {
      LAUNCH_PERSISTENT(TopK, 4);
    } else if (vec_size == 2) {
      LAUNCH_PERSISTENT(TopK, 2);
    } else {
      LAUNCH_PERSISTENT(TopK, 1);
    }
  #undef LAUNCH_PERSISTENT
  }

  cudaError_t err = cudaGetLastError();
  STD_TORCH_CHECK(err == cudaSuccess,
                  "persistent_topk failed: ", cudaGetErrorString(err));
}
#endif

}  // anonymous namespace

// Merlin's entry point, bound to _C::merlin_persistent_topk (bindings.cpp).
// In namespace merlin::topk because upstream's csrc/topk.cu is compiled
// again and defines the global persistent_topk() behind _C::persistent_topk.
namespace merlin::topk {

void persistent_topk(const torch::stable::Tensor& logits,
                     const torch::stable::Tensor& lengths,
                     torch::stable::Tensor& output,
                     torch::stable::Tensor& workspace, int64_t k,
                     int64_t max_seq_len) {
#ifndef USE_ROCM
  STD_TORCH_CHECK(logits.is_cuda(), "logits must be CUDA tensor");
  STD_TORCH_CHECK(lengths.is_cuda(), "lengths must be CUDA tensor");
  STD_TORCH_CHECK(output.is_cuda(), "output must be CUDA tensor");
  STD_TORCH_CHECK(logits.scalar_type() == torch::headeronly::ScalarType::Float,
                  "Only float32 supported");
  STD_TORCH_CHECK(lengths.scalar_type() == torch::headeronly::ScalarType::Int,
                  "lengths must be int32");
  STD_TORCH_CHECK(output.scalar_type() == torch::headeronly::ScalarType::Int,
                  "output must be int32");
  STD_TORCH_CHECK(logits.dim() == 2, "logits must be 2D");
  STD_TORCH_CHECK(lengths.dim() == 1 || lengths.dim() == 2,
                  "lengths must be 1D or 2D");
  STD_TORCH_CHECK(lengths.is_contiguous(), "lengths must be contiguous");
  STD_TORCH_CHECK(output.dim() == 2, "output must be 2D");

  const int64_t num_rows = logits.size(0);

  STD_TORCH_CHECK(lengths.numel() == num_rows, "lengths size mismatch");
  STD_TORCH_CHECK(output.size(0) == num_rows && output.size(1) == k,
                  "output size mismatch");
  STD_TORCH_CHECK(
      k == 512 || k == 1024 || k == 2048,
      "persistent_topk supports k=512, k=1024, or k=2048, got k=", k);

  const torch::stable::accelerator::DeviceGuard device_guard(
      logits.get_device_index());

  if (k == 512) {
    launch_persistent_topk<512>(logits, lengths, output, workspace,
                                max_seq_len);
  } else if (k == 1024) {
    launch_persistent_topk<1024>(logits, lengths, output, workspace,
                                 max_seq_len);
  } else {
    launch_persistent_topk<2048>(logits, lengths, output, workspace,
                                 max_seq_len);
  }
#else
  STD_TORCH_CHECK(false, "persistent_topk is not supported on ROCm");
#endif
}

}  // namespace merlin::topk

// Read (and, on first call, allocate) the exact-restart counters; also the
// pre-capture init path, so call it once per device before any cudagraph
// capture, never during one. `out` is an int64 CUDA tensor of exactly
// persistent_topk_exact_counters_slots() elements (64-bit counters cannot
// wrap); any other dtype or length is rejected. `sync != 0` synchronizes before
// and after the copy/reset; `reset` requires `sync`.
//
// Slots:
//   [0] medium-path rows entered           (denominator)
//   [1] rows where a 4,096-slot guard would drop a candidate
//   [2] rows that took the exact restart
//   [3] exact restarts that could not complete (must stay 0)
//   [4] large-path rows emitted deterministically
//   [5] large-path rows whose pivot-equal group was cut at the k-th value
//   [6] large-path pivot-equal elements discarded by those cuts
//   [7] large-path invariant violations (must stay 0)
//   [8] 1 if the exact medium restart is on, else 0   (host-written)
//   [9] 1 if the deterministic large path is on, else 0 (host-written)
//   [10] decode rows whose buffered candidates exceeded DBUF (3,708)
void persistent_topk_exact_counters(torch::stable::Tensor& out, int64_t reset,
                                    int64_t sync) {
#ifndef USE_ROCM
  STD_TORCH_CHECK(out.is_cuda(), "out must be a CUDA tensor");
  STD_TORCH_CHECK(
      out.scalar_type() == torch::headeronly::ScalarType::Long,
      "out must be int64: the exact-restart counters are 64-bit as of the "
      "large-path counter protocol change, and an int32 buffer would be filled "
      "from half of each counter. Allocate "
      "torch.zeros(persistent_topk_exact_counters_slots(), dtype=torch.int64)."
      " (int32 was protocol version 1 and is deliberately rejected.)");
  // Element count alone is not enough. A CUDA scalar expanded to the slot count
  // (stride 0) satisfies numel but owns one element's storage, and the raw int64
  // writes below would run past it; an ordinary strided view would receive the
  // values in the wrong slots. Require a contiguous 1-D buffer before taking the
  // pointer.
  STD_TORCH_CHECK(out.dim() == 1, "out must be 1-D");
  STD_TORCH_CHECK(out.is_contiguous(),
                  "out must be contiguous (expanded or strided views would be "
                  "written out of bounds or into the wrong elements)");
  // Exactly the build's slot count, not "at least": a caller compiled against
  // another layout must get an error rather than a plausible answer. Ask
  // persistent_topk_exact_counters_slots() and size the buffer to what it says.
  STD_TORCH_CHECK(out.numel() == merlin::persistent::kNumTopKExactCounters,
                  "out must hold exactly ",
                  merlin::persistent::kNumTopKExactCounters,
                  " elements (this build's persistent_topk_exact_counters_"
                  "slots(); the switch flags are slots inside the array, not "
                  "appended after it), got ", out.numel());
  STD_TORCH_CHECK(
      reset == 0 || sync != 0,
      "persistent_topk_exact_counters: reset requires sync != 0, otherwise the "
      "zeroing races in-flight kernel atomics and silently drops events");

  const torch::stable::accelerator::DeviceGuard device_guard(
      out.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream();
  STD_TORCH_CHECK(!stream_is_capturing(stream),
                  "persistent_topk_exact_counters must not be called during "
                  "cudagraph capture");
  constexpr size_t bytes =
      merlin::persistent::kNumTopKExactCounters * sizeof(unsigned long long);

  int64_t* dst = out.mutable_data_ptr<int64_t>();
  unsigned long long* src =
      ensure_exact_counters(static_cast<int>(out.get_device_index()), stream);

  // An unavailable buffer must raise: it must not read as healthy zero counts,
  // which would be indistinguishable from no medium work and no faults.
  STD_TORCH_CHECK(src != nullptr,
                  "persistent_topk_exact_counters: the counter buffer could "
                  "not be allocated on this device; the instrument is "
                  "unavailable and its absence must not read as zero counts");

  if (sync != 0) {
    // Everything that could still be adding to the block must have retired
    // before the snapshot (and before a reset) for the numbers to mean
    // anything across streams.
    STD_TORCH_CHECK(cudaDeviceSynchronize() == cudaSuccess,
                    "exact counter synchronize failed");
  }

  STD_TORCH_CHECK(
      cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, stream) ==
          cudaSuccess,
      "exact counter read failed");
  if (reset != 0) {
    STD_TORCH_CHECK(cudaMemsetAsync(src, 0, bytes, stream) == cudaSuccess,
                    "exact counter reset failed");
  }

  // Static storage: the source of an async H2D copy must outlive the copy.
  // Written after the device copy, which has already zeroed both slots: they
  // are slots [8] and [9] inside the array and the device never touches them.
  // The large flag reports the compiled policy value (det_large_enabled()), not
  // the per-launch effective value: a launch whose ctas_per_group overflowed the staging row still
  // reports 1 here and leaves counter [4] at 0. The two flags are adjacent, so
  // one copy writes both.
  static_assert(merlin::persistent::kSlotSwitchLarge ==
                    merlin::persistent::kSlotSwitchMedium + 1,
                "the two host-written flags must stay adjacent for this copy");
  static const int64_t enabled[2] = {exact_medium_enabled() ? 1 : 0,
                                     det_large_enabled() ? 1 : 0};
  STD_TORCH_CHECK(
      cudaMemcpyAsync(dst + merlin::persistent::kSlotSwitchMedium, enabled,
                      2 * sizeof(int64_t), cudaMemcpyHostToDevice,
                      stream) == cudaSuccess,
      "exact counter flag write failed");

  if (sync != 0) {
    // The leading synchronize leaves the copy and the reset merely enqueued;
    // complete the handoff before returning so the caller cannot race the
    // outstanding reset from another stream.
    STD_TORCH_CHECK(cudaDeviceSynchronize() == cudaSuccess,
                    "exact counter completion synchronize failed");
  }
#else
  STD_TORCH_CHECK(false, "persistent_topk is not supported on ROCm");
#endif
}

// How many 64-bit slots this build's counter array carries.
//
// Readers must size their buffer from this value: a buffer of any other length
// is rejected by the length check above. No CUDA call, no allocation, safe from
// anywhere.
int64_t persistent_topk_exact_counters_slots() {
#ifndef USE_ROCM
  return static_cast<int64_t>(merlin::persistent::kNumTopKExactCounters);
#else
  return 0;
#endif
}

// The persistent_topk policy switches as this process cached them -- the values
// every launch of this process uses -- read from the mirrors above. Tensor-free,
// no CUDA call, no allocation, never initialises a cache (see publish_cached),
// so it is safe from anywhere and asking changes nothing.
//
// Packed into one int64 so the op needs no tensor and no dispatch key.
// 2 bits per field, field i at bits [2i, 2i+1]: 0 = off, 1 = on,
// 2 = the static has not been initialised in this process yet.
//   field 0  deterministic ties                  (det_ties_enabled)
//   field 1  tie direction as configured         (tie_recent_enabled)
//   field 2  effective direction, 1 = most-recent (effective_tie_recent)
//   field 3  exact medium restart                (exact_medium_enabled)
//   field 4  exact decode restart                (exact_decode_enabled)
//   field 5  deterministic large path            (det_large_enabled)
//   field 6  sorted emission                     (sorted_emit_enabled)
// bits [48, 55] = field count (7), bits [56, 62] = layout version (2). A
// reader must check both and refuse any other layout rather than guess. Field 6
// is what qsa_indexer.py reads to decide whether its Python sort may be
// skipped.
int64_t persistent_topk_toggle_state() {
#ifndef USE_ROCM
  const auto field = [](int cached) -> int64_t {
    return cached < 0 ? 2 : (cached != 0 ? 1 : 0);
  };
  const int det = g_cached_tie_det.load(std::memory_order_acquire);
  const int recent = g_cached_tie_recent.load(std::memory_order_acquire);
  // The kernel's own rule, from the kernel's own header, on the cached values.
  const int effective =
      (det < 0 || recent < 0)
          ? -1
          : (merlin::persistent::effective_tie_recent(det != 0, recent != 0)
                 ? 1
                 : 0);
  int64_t state = 0;
  state |= field(det) << 0;
  state |= field(recent) << 2;
  state |= field(effective) << 4;
  state |= field(g_cached_exact_medium.load(std::memory_order_acquire)) << 6;
  state |= field(g_cached_exact_decode.load(std::memory_order_acquire)) << 8;
  state |= field(g_cached_det_large.load(std::memory_order_acquire)) << 10;
  state |= field(g_cached_sorted_emit.load(std::memory_order_acquire)) << 12;
  state |= int64_t{7} << 48;
  state |= int64_t{2} << 56;
  return state;
#else
  return 0;  // layout version 0: no persistent_topk on ROCm
#endif
}
