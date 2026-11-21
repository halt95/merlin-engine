# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
# Modified by the Merlin project; see NOTICE.
"""Causal window indexing over engine-managed sliding MLA pages.

``FlashWindowIndexer`` is #68296 ``hy_v4_flash/nvidia/swa.py:59-94``, its body
split into ``select`` so the CPU gate can call it without a forward context;
``window_indices`` is a copy of :50-56. Upstream's FlashMLA-based
``FlashSparseBackend``/``FlashSWABackend`` (:17-47) are replaced on this lane by
the ``Hy4TritonSparseBackend``/``Hy4TritonSWABackend``
(``vllm.merlin.hy4_backend``), whose builder already takes ``topk_tokens``
from the layer's ``topk_buffer_width``; ``backend.build_attention_parts`` picks
them.
"""

import os
from typing import Any

import torch
from torch import nn

from vllm.forward_context import get_forward_context
from vllm.utils.math_utils import round_up

from .indexpool_ops import window_indices

__all__ = ["FlashWindowIndexer", "window_indices"]


# The causal window rows depend only on the
# layer's attention metadata and the window, not on the layer: with
# MERLIN_HY4_SWA_WINDOW_REUSE=1 the first SWA layer of a forward computes them
# (about 30 small torch kernels) and keeps a copy on the forward context; every
# later SWA layer with the same metadata object and window copies that instead
# (one kernel). The rows are identical, so the attention is unchanged. Inside a
# CUDA graph capture the first layer's kernels are captured and recompute the
# copy from the live metadata on every replay. Unset/blank/0 = stock.
def swa_window_reuse_requested(environ=None) -> bool:
    raw = ((os.environ if environ is None else environ).get("MERLIN_HY4_SWA_WINDOW_REUSE", "") or "").strip()
    if raw in ("", "0"):
        return False
    if raw == "1":
        return True
    raise ValueError(f"MERLIN_HY4_SWA_WINDOW_REUSE must be 0 or 1 (got {raw!r})")


_WINDOW_REUSE = swa_window_reuse_requested()
_CACHE_ATTR = "_merlin_swa_window"


def _tensor_key(t):
    """Identity of a tensor's values within one forward (hy4-swa-reuse-groups)."""
    return (t.data_ptr(), tuple(t.shape), tuple(t.stride()), t.dtype, t.device)


class FlashWindowIndexer(nn.Module):
    def __init__(self, window: int, topk_indices_buffer, attention_prefix: str):
        super().__init__()
        self.window = window
        self.attention_prefix = attention_prefix
        # Logical-to-physical remapping uses 128-column tiles.
        width = round_up(window, 128)
        if window < 1 or topk_indices_buffer.shape[1] < width:
            raise ValueError("SWA index width must cover its causal window")
        rows = topk_indices_buffer.shape[0]
        # Reuse the shared scratch storage with a packed, window-sized stride.
        self.topk_indices_buffer = topk_indices_buffer.view(-1)[: rows * width].view(
            rows, width
        )

    def forward(self, hidden_states, q_c, positions, rope=None):
        output = self.topk_indices_buffer[: hidden_states.shape[0]]
        metadata = get_forward_context().attn_metadata
        if metadata is None:
            return output.fill_(-1)
        if not isinstance(metadata, dict):
            raise TypeError("Flash SWA requires per-layer metadata")
        meta: Any = metadata[self.attention_prefix]
        if not _WINDOW_REUSE:
            return self.select(meta, output)
        return self.select_reused(get_forward_context(), meta, output)


    def select_reused(self, context, meta, output):
        """``select`` once per forward context and window-row inputs; later layers
        copy the kept rows into their (shared) index buffer.

        the hybrid KV planner deals SWA
        layers out to its groups round-robin (``layers[i::num_groups]``), so
        neighbouring SWA layers carry different metadata objects and a one-slot
        cache keyed on the object missed on every layer (in a profile the ~27-kernel
        window build ran on all 15 SWA layers of each PP stage). The cache now
        holds one entry per input set, keyed on the storage of the tensors the
        rows are computed from (query_start_loc, seq_lens) plus
        num_actual_tokens, the window and the output shape: groups whose
        metadata share those tensors share one entry, other groups get their
        own. The rows are a pure function of those inputs, so the copy is
        bitwise the rows ``select`` would write. Each entry keeps its metadata
        object alive, so no key's storage can be reused within the forward."""
        kept = getattr(context, _CACHE_ATTR, None)
        if not isinstance(kept, dict):
            kept = {}
            setattr(context, _CACHE_ATTR, kept)
        key = (_tensor_key(meta.query_start_loc), _tensor_key(meta.seq_lens),
               int(meta.num_actual_tokens), self.window, tuple(output.shape))
        entry = kept.get(key)
        if entry is not None:
            return output.copy_(entry[1])
        self.select(meta, output)
        kept[key] = (meta, output.clone())
        return output

    def select(self, meta, output):
        """Fill ``output`` from one layer's attention metadata (split for tests)."""
        if meta.num_actual_tokens == 0 or meta.seq_lens.numel() == 0:
            return output.fill_(-1)
        rows = torch.arange(output.shape[0], device=output.device)
        starts = meta.query_start_loc
        requests = torch.searchsorted(starts[1:], rows, right=True)
        requests = requests.clamp_max(starts.numel() - 2).long()
        logical = meta.seq_lens[requests] - (starts[requests + 1] - starts[requests])
        logical = logical + rows - starts[requests]
        live = (rows < meta.num_actual_tokens) & (rows < starts[-1])
        output.copy_(
            window_indices(torch.where(live, logical, -1), self.window, output.shape[1])
        )
        return output
