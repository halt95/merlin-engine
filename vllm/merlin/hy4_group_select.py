# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Fused glue around the IndexPool group top-k (``MERLIN_HY4_INDEXER_SELECT=1``).

Profiled on Hy4 Flash (sm_86): on each DSA layer the group selection of
``vllm/merlin/hy4_kvshard.py`` (``local_group_candidates`` +
``merge_packed_candidates``, or ``_sorted_select`` without DCP) runs about 40
small torch kernels around its two ``topk`` calls and one ``sort`` (0.40 ms per
step per PP stage). Here the elementwise glue becomes four Triton kernels and
the ``topk`` / ``sort`` calls stay torch's own:

    mask    scores[t, w] = -inf if w >= local_count(upper[t]) else logits[t, w]
    topk    (torch)
    pack    [T, budget, 2]: (score, global group id as fp32), -inf / -1 padded
    (all-gather under DCP, unchanged)
    merge   s[t, n*K + k] = -inf if gid < 0 else score; g = long(gid)
    topk    (torch)
    pick    chosen = g[t, pick] where value != -inf else INT32_MAX (int32)
    sort    (torch)
    finish  INT32_MAX -> -1, padded to budget

Bitwise: every step is a selection, an integer map, or an exact conversion
(fp32 -> fp32, int -> fp32 below 2**24, fp32 -> int64 of an integral value),
and the top-k / sort inputs are the same values in the same contiguous [T, W]
layout as the reference, so torch returns the same results. GPU test:
tests/merlin/test_hy4_group_select_gpu.py in the development tree (torch.equal
against the reference).
"""

from __future__ import annotations

import os

import torch

ENV = "MERLIN_HY4_INDEXER_SELECT"
SENTINEL = torch.iinfo(torch.int32).max


def select_fused_requested(environ=None) -> bool:
    raw = ((os.environ if environ is None else environ).get(ENV, "") or "").strip()
    if raw not in ("", "0", "1"):
        raise ValueError(f"{ENV} must be 0 or 1 (got {raw!r})")
    return raw == "1"


def _layout_args(layout):
    """(sharded, unit, dcp_size, rank) for the group-unit arithmetic."""
    return (bool(layout.sharded), int(layout.group_interleave), int(layout.dcp_size),
            int(layout.dcp_rank))


def local_candidates(logits_local, upper, layout, budget):
    """``local_group_candidates`` (packed [T, budget, 2] fp32), fused."""
    from vllm.merlin import hy4_group_select_kernels as K

    t, w = logits_local.shape
    upper = upper.contiguous()
    if logits_local.stride(1) != 1:
        logits_local = logits_local.contiguous()
    sharded, unit, size, rank = _layout_args(layout)
    scores = torch.empty((t, w), dtype=torch.float32, device=logits_local.device)
    if t and w:
        K.mask(logits_local, upper, scores, sharded, unit, size, rank)
    count = min(budget, w)
    values, ids = scores.topk(count, dim=-1)
    packed = torch.empty((t, budget, 2), dtype=torch.float32, device=logits_local.device)
    if t:
        K.pack(values, ids, packed, count, budget, sharded, unit, size, rank)
    return packed


def merge_candidates(gathered, budget):
    """``merge_packed_candidates`` ([N, T, K, 2] -> [T, budget] int32), fused."""
    from vllm.merlin import hy4_group_select_kernels as K

    n, t, k, _ = gathered.shape
    s = torch.empty((t, n * k), dtype=torch.float32, device=gathered.device)
    g = torch.empty((t, n * k), dtype=torch.int64, device=gathered.device)
    if t and n * k:
        K.merge(gathered, s, g)
    count = min(budget, n * k)
    values, pick = s.topk(count, dim=-1)
    chosen = torch.empty((t, count), dtype=torch.int32, device=gathered.device)
    if t and count:
        K.pick(values, pick, g, chosen)
    chosen = chosen.sort(dim=-1).values
    out = torch.empty((t, budget), dtype=torch.int32, device=gathered.device)
    if t:
        K.finish(chosen, out, count, budget)
    return out


def sorted_select(logits, upper, layout, budget):
    """``_sorted_select`` (no DCP): one rank's candidates, merged."""
    return merge_candidates(local_candidates(logits, upper, layout, budget)[None], budget)
