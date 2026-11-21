# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""DFly2 target top-k snapshots sized by the rows the drafter can read
.

The drafter reads one target row per request: the row of its last accepted
token, ``source_row = query_end - num_rejected - 1`` with ``num_rejected <= K``
(dfly_inputs.py). That row is always among the request's last ``K + 1`` rows,
so each target index layer keeps only those rows, ``max_num_seqs x (K + 1)`` in
all, instead of a copy of its whole ``max_num_batched_tokens``-row top-k buffer
(5 x 2048 x 2048 int32 = 80 MiB on the last PP stage for 48 readable rows).

Layout: request ``r``'s tail row ``t`` (``t = 0..K``, target row
``query_end_r - (K + 1) + t``) is snapshot row ``r * (K + 1) + t``; tail rows
before the request's first row hold -1 (never read: the source row is inside
the request). The drafter's source row maps to
``r * (K + 1) + (K + 1) - (query_end_r - source_row)``. The values copied are
the target's own top-k rows, so the draft indices are unchanged.
"""

from __future__ import annotations

import torch


def snapshot_rows(scheduler_config, width: int) -> int:
    """Snapshot rows: ``max_num_seqs x width`` (width = K + 1), at most the step budget."""
    tokens = scheduler_config.max_num_batched_tokens
    seqs = getattr(scheduler_config, "max_num_seqs", None)
    if not seqs:
        return tokens
    return min(tokens, int(seqs) * width)


def copy_tail_rows(
    snapshot: torch.Tensor, topk: torch.Tensor, query_start_loc: torch.Tensor, width: int
) -> None:
    """Copy each request's last ``width`` rows of ``topk`` into ``snapshot`` (-1 where a
    tail row falls before the request)."""
    cap = snapshot.shape[0] // width
    n = min(query_start_loc.shape[0] - 1, cap)
    if n <= 0:
        return
    qsl = query_start_loc[: n + 1].long()
    offsets = torch.arange(width, device=topk.device)
    rows = qsl[1:, None] - width + offsets[None, :]
    valid = rows >= qsl[:-1, None]
    picked = topk.index_select(0, rows.clamp_min(0).reshape(-1)).view(n, width, -1)
    picked = torch.where(valid[..., None], picked, torch.full_like(picked, -1))
    snapshot[: n * width].view(n, width, -1).copy_(picked)


def copy_from_forward_context(
    snapshot: torch.Tensor, topk: torch.Tensor, layer_name: str, width: int
) -> None:
    """``copy_tail_rows`` with the step's query starts from this layer's metadata."""
    from vllm.forward_context import get_forward_context

    metadata = get_forward_context().attn_metadata
    if metadata is None:  # dummy / profile runs: the drafter reads no snapshot
        return
    if not isinstance(metadata, dict):
        raise TypeError("DFly2 target snapshots require per-layer metadata")
    md = metadata.get(layer_name)
    qsl = getattr(md, "query_start_loc", None)
    if qsl is None:
        raise TypeError(f"DFly2 target layer {layer_name} has no query_start_loc")
    copy_tail_rows(snapshot, topk, qsl, width)


def source_snapshot_rows(
    source_rows: torch.Tensor,
    query_start_loc: torch.Tensor,
    padded_batch: int,
    width: int,
    cap_rows: int,
) -> torch.Tensor:
    """Snapshot row of each request's source row; padding requests get row 0."""
    out = torch.zeros(padded_batch, dtype=torch.int64, device=source_rows.device)
    n = min(padded_batch, query_start_loc.shape[0] - 1)
    if n > 0:
        ends = query_start_loc[1 : n + 1].long()
        req = torch.arange(n, device=source_rows.device)
        out[:n] = req * width + width - (ends - source_rows[:n].long())
    return out.clamp_(0, max(cap_rows - 1, 0))
