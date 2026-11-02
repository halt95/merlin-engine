# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Sync-free FlashInfer planning for MTP draft decode (VLLM_DRAFT_BOUNDED_PLAN).

Pure numpy/torch, no vllm imports, so the CPU unit tests can load this file alone.

Problem. Draft step s attends over ``true_s = min(T - rej + s, L)`` KV tokens per request,
where T is the verify step's target length and ``rej`` the number of drafts the verify step
rejected. Both live only on the GPU, so the FlashInfer build calls ``seq_lens.cpu()`` (a
blocking device->host copy behind the whole verify forward) to plan the decode.

What the plan actually depends on. With CUDA graphs on (fa2, sm_86), FlashInfer 0.6.18's
planner reads only per-request PAGE counts (kv_indptr), never kv lengths
(include/flashinfer/attention/scheduler.cuh PrefillSplitQOKVIndptr L545-683, csrc/batch_prefill.cu
L63-70: kv_len_arr is not passed to PrefillPlan). Everything it writes to the int workspace
(request/qo/kv tile indices, o_indptr, merge_indptr, block_valid_mask, kv_chunk_size) is a
function of two things only:
  * c      = the KV chunk size in pages (PrefillBinarySearchKVChunkSize, scheduler.cuh L101-130),
  * n_i    = ceil(max(pages_i, 1) / c), the chunk count of each request (L651-652).
The kernel reads the true length from the GPU page table at replay
(page.cuh L820-825 get_length; prefill.cuh L3602-3616).

So the plan built from an upper bound is BIT-IDENTICAL to the plan built from the true
lengths whenever c and every n_i agree. The CPU knows a per-row interval [lb, ub] that
contains the true length (anchored to the verify step's exact, already-synced
target lengths); this module decides on the CPU whether every length in that
interval yields the same (c, n_i). If yes, plan with ub and write the true page table on
the GPU (no sync). If no, the caller syncs exactly as before.

Why checking the two endpoints suffices. f(mid) = sum_i q_i * ceil(max(p_i,1)/mid) is
non-decreasing in every p_i and non-increasing in mid; the search returns
min{mid in [cmin, H] : f(mid) <= M} (or H, or cmin when cmin >= H) with H = max(1, max p_i),
also non-decreasing in every p_i. Hence c(p) is monotone in p, and lb <= p <= ub pointwise gives
c(lb) <= c(p) <= c(ub). If c(lb) == c(ub) == c, then n_i(p) = ceil(max(p_i,1)/c) is monotone in
p_i, so n_i(lb_i) == n_i(ub_i) pins n_i(p_i). Pages are a monotone function of length, so the
interval in tokens maps to an interval in pages.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


def cdiv_np(a: np.ndarray, b: int) -> np.ndarray:
    return (a + (b - 1)) // b


def num_pages(seq_lens: np.ndarray, page_size: int) -> np.ndarray:
    """Pages per request, as FlashInferMetadataBuilder.build derives them."""
    return cdiv_np(seq_lens.astype(np.int64), page_size)


def fa2_binary_search_chunk_pages(
    pages: np.ndarray,
    q_tiles_per_req: int,
    max_batch_size_if_split: int,
    min_kv_chunk_pages: int,
) -> int:
    """Replica of PrefillBinarySearchKVChunkSize (FlashInfer 0.6.18 scheduler.cuh L101-130).

    window_left < 0 (no sliding window), so effective_kv_len == kv_len (L628-634).
    q_tiles_per_req = ceil(packed_qo_len / cta_tile_q), identical for every row of a
    uniform q_len=1 decode batch.
    """
    max_kv_len = max(1, int(pages.max())) if pages.size else 1
    low = int(min_kv_chunk_pages)
    high = max_kv_len
    p = np.maximum(pages.astype(np.int64), 1)
    while low < high:
        mid = (low + high) // 2
        new_batch_size = int(q_tiles_per_req * cdiv_np(p, mid).sum())
        if new_batch_size > max_batch_size_if_split:
            low = mid + 1
        else:
            high = mid
    return low


@dataclass(frozen=True)
class Fa2DecodeSchedule:
    """The kv-dependent part of an fa2 cuda-graph plan (everything else is fixed per wrapper)."""

    chunk_pages: int  # c, kv_chunk_size in pages (the plan stores c * page_size)
    chunks: tuple[int, ...]  # n_i per request


def fa2_decode_schedule(
    pages: np.ndarray,
    q_tiles_per_req: int,
    max_batch_size_if_split: int,
    min_kv_chunk_pages: int,
) -> Fa2DecodeSchedule:
    c = fa2_binary_search_chunk_pages(
        pages, q_tiles_per_req, max_batch_size_if_split, min_kv_chunk_pages
    )
    chunks = cdiv_np(np.maximum(pages.astype(np.int64), 1), c)
    return Fa2DecodeSchedule(c, tuple(int(x) for x in chunks))


def draft_seq_len_bounds(
    target_seq_lens_exact: np.ndarray,
    num_reqs: int,
    num_reqs_padded: int,
    step: int,
    max_rejected: int,
    max_model_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    """[lb, ub] for the draft-step-``step`` seq_lens the GPU holds (int64 arrays).

    ``target_seq_lens_exact`` are the verify step's EXACT target lengths T, i.e. the CPU copy
    the target FlashInfer build synced this step (never the scheduler's upper bound, which
    can run any number of tokens ahead: the warm-up's synthetic schedule never rolls back
    rejections). prepare_decode_inputs writes min(T - rej + 1, L) and every later draft step
    adds 1 (clamped), so true = min(T - rej + step, L) with 0 <= rej <= max_rejected (the
    rejection sampler never rejects more than the drafts it verified, <= K). Holds by
    construction. Padded rows of a FULL-graph batch: prepare_decode_inputs writes 0 and every
    replayed draft step's update kernel adds 1 (it runs over the padded batch), so step s
    sees s - 1; an eager batch leaves 0. Both lie in [0, step - 1].
    """
    ub = np.zeros(num_reqs_padded, dtype=np.int64)
    lb = np.zeros(num_reqs_padded, dtype=np.int64)
    t = target_seq_lens_exact[:num_reqs].astype(np.int64)
    ub[:num_reqs] = np.minimum(t + step, max_model_len)
    lb[:num_reqs] = np.minimum(np.maximum(t - max_rejected + step, 0), max_model_len)
    if num_reqs_padded > num_reqs:
        ub[num_reqs:] = min(max(step - 1, 0), max_model_len)
    return lb, ub


@dataclass(frozen=True)
class BoundedPlanDecision:
    exact: bool  # True: the ub plan is the plan of every length in [lb, ub]
    ambiguous_rows: int  # rows whose page count is not pinned by the bounds
    ub_schedule: Fa2DecodeSchedule
    lb_schedule: Fa2DecodeSchedule | None


def decide_bounded_plan(
    lb: np.ndarray,
    ub: np.ndarray,
    page_size: int,
    q_tiles_per_req: int,
    max_batch_size_if_split: int,
    min_kv_chunk_pages: int,
) -> BoundedPlanDecision:
    """Whether planning with ``ub`` gives the same fa2 schedule as any true length in [lb, ub]."""
    p_ub = num_pages(ub, page_size)
    p_lb = num_pages(lb, page_size)
    s_ub = fa2_decode_schedule(
        p_ub, q_tiles_per_req, max_batch_size_if_split, min_kv_chunk_pages
    )
    # Rows with 0 or 1 pages contribute max(p, 1) = 1 either way.
    differ = np.maximum(p_ub, 1) != np.maximum(p_lb, 1)
    n_amb = int(differ.sum())
    if n_amb == 0:
        return BoundedPlanDecision(True, 0, s_ub, None)
    s_lb = fa2_decode_schedule(
        p_lb, q_tiles_per_req, max_batch_size_if_split, min_kv_chunk_pages
    )
    return BoundedPlanDecision(s_lb == s_ub, n_amb, s_ub, s_lb)


def kv_metadata_reference(
    seq_lens: torch.Tensor, block_table: torch.Tensor, page_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """paged_kv_indptr, packed paged_kv_indices, paged_kv_last_page_len exactly as
    FlashInferMetadataBuilder._compute_flashinfer_kv_metadata writes them (int32, CPU).

    Also the specification of the GPU fix-up kernel (_draft_bounded_kv_fixup_kernel).
    """
    s = seq_lens.to(torch.int64).cpu()
    nb = (s + page_size - 1) // page_size
    indptr = torch.zeros(s.numel() + 1, dtype=torch.int64)
    indptr[1:] = torch.cumsum(nb, 0)
    bt = block_table.cpu()
    if s.numel() and int(nb.max()) > bt.shape[1]:
        # a truncated row would silently describe pages that do not exist
        raise ValueError(
            f"block table has {bt.shape[1]} columns, a row needs {int(nb.max())} pages"
        )
    indices = torch.cat(
        [bt[i, : int(nb[i])].to(torch.int64) for i in range(s.numel())]
        or [torch.zeros(0, dtype=torch.int64)]
    )
    lpl = s % page_size
    lpl = torch.where((lpl == 0) & (s != 0), torch.full_like(lpl, page_size), lpl)
    return indptr.to(torch.int32), indices.to(torch.int32), lpl.to(torch.int32)


def fixup_kernel_emulation(
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    page_size: int,
    indptr_out: torch.Tensor,
    indices_out: torch.Tensor,
    lpl_out: torch.Tensor,
) -> None:
    """Program-by-program CPU emulation of _draft_bounded_kv_fixup_kernel (same arithmetic,
    same per-program prefix sum), so the kernel's algorithm is unit-tested off the GPU."""
    n = seq_lens.numel()
    s = seq_lens.to(torch.int32)
    for req in range(n):  # one Triton program per row
        start = 0
        for r in range(req):
            start += int((int(s[r]) + page_size - 1) // page_size)
        seq = int(s[req])
        nb = (seq + page_size - 1) // page_size
        if req == 0:
            indptr_out[0] = 0
        indptr_out[req + 1] = start + nb
        lpl = seq % page_size
        lpl_out[req] = page_size if (lpl == 0 and seq != 0) else lpl
        indices_out[start : start + nb] = block_table[req, :nb]


def plan_bytes_schedule(
    pinned_int_ws: torch.Tensor, plan_info, batch_size: int, page_size: int
) -> Fa2DecodeSchedule:
    """Read (c, n_i) back from a FlashInfer fa2 plan in the host int workspace.

    plan_info is PrefillPlanInfo::ToVector (scheduler.cuh L720-736): [8] o_indptr_offset,
    [9] kv_chunk_size_ptr_offset. q_len = 1, so o_indptr[i+1] - o_indptr[i] = n_i.
    """
    o_off = int(plan_info[8])
    c_off = int(plan_info[9])
    o = pinned_int_ws[o_off : o_off + 4 * (batch_size + 1)].view(torch.int32)
    c_tok = int(pinned_int_ws[c_off : c_off + 4].view(torch.int32)[0])
    chunks = (o[1:] - o[:-1]).tolist()
    if c_tok % page_size:
        return Fa2DecodeSchedule(-1, tuple(chunks))
    return Fa2DecodeSchedule(c_tok // page_size, tuple(int(x) for x in chunks))
