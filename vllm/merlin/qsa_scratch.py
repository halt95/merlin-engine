# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""QSA indexer selection-scratch bounds and reservation mode (port of
vllm#60490).

The Flash-Next QSA indexer allocates its selection scratch on demand while
serving: the prefill logits backing, or the decode logits, plus the 1 MiB
top-k workspace, and the ``block_indices`` output. The memory profiler's dummy
run returned before the indexer, so none of it was in the profiled peak.

Upstream vllm#60490 touches ``max(VLLM_SPARSE_INDEXER_MAX_LOGITS_MB, decode
rows x columns x 4) + top-k`` in that branch. Here the bound is derived from
the same arithmetic the serving path runs (``ops/qsa_indexer.py``), so the two
cannot drift: the fixed prefill backing (or its granule rounding), the decode
logits at the runner-aligned block-table width, the top-k workspace, and
``block_indices``.

``VLLM_MERLIN_QSA_SCRATCH`` picks what the dummy run does with that bound:

- ``off`` (default): nothing.
- ``reserve``: a transient allocation of the whole bound, as upstream
  does. The profiler sees it, so the KV pool shrinks by the bound
  (minus whatever peak already covered it).
- ``workspace``: the logits part is taken from the shared MRV2
  ``WorkspaceManager`` (as upstream's sparse indexers and modular MoE do), so it
  shares one buffer with the other per-layer scratch users (MoE, GDN
  checkpoints) and is reserved before the workspace locks, so it cannot
  grow after capture. Serving then slices that buffer instead of allocating, only when
  it already fits and never inside CUDA-graph capture, so captured graphs are
  unchanged and the workspace never grows after the profile run because of
  QSA. The top-k workspace and ``block_indices`` (live across the decode and
  prefill selections) stay transient allocations and transient reservations.

Every mode runs the same kernels on tensors of the same shape, stride and
dtype; only the allocation source differs, so outputs are bit-identical.
The arithmetic is stdlib only (CPU-tested by path); the mode is parsed once,
by the fork env registry.
"""

from __future__ import annotations

FP32_BYTES = 4
INT32_BYTES = 4
# Must equal ops/qsa_indexer.py _TOPK_WORKSPACE_BYTES.
TOPK_WORKSPACE_BYTES = 1024 * 1024


def scratch_mode() -> str:
    """``VLLM_MERLIN_QSA_SCRATCH``: off (default), reserve or workspace; the
    registry getter validates it."""
    import vllm.envs as envs

    return envs.VLLM_MERLIN_QSA_SCRATCH


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


def _round_up(a: int, b: int) -> int:
    return _cdiv(a, b) * b


def prefill_backing_numel(
    num_queries: int, logits_width: int, granule_mb: int, max_logits_bytes: int
) -> int:
    """Same arithmetic as ops/qsa_indexer.py ``_prefill_backing_numel``."""
    need = num_queries * logits_width
    granule = granule_mb * 2**20 // FP32_BYTES
    rounded = _cdiv(need, granule) * granule
    return max(need, min(rounded, max_logits_bytes // FP32_BYTES))


def prefill_logits_width_bound(max_model_len: int, compress_ratio: int) -> int:
    """Upper bound of the serving ``logits_width`` (before the page-table cap):
    ``max(64, round_up(cdiv(max_seq_len, ratio), 64))`` at the longest context."""
    return max(64, _round_up(_cdiv(max_model_len, compress_ratio), 64))


def prefill_scratch_numel(
    *,
    max_num_batched_tokens: int,
    max_model_len: int,
    compress_ratio: int,
    backing_bytes: int,
    granule_mb: int,
    max_logits_bytes: int,
) -> int:
    """Largest fp32 backing ``qsa_select_paged_prefill`` requests.

    Granule off: the fixed ``backing_bytes`` (from ``_prefill_workspace_bytes``)
    or one row when a row is wider (the one-row exception). Granule on: the
    rounded need of the largest chunk, still under the logits cap unless one
    row exceeds it. Serving's need is monotone in rows and width, so the bound
    at (max_num_batched_tokens, widest row) covers every step.
    """
    width = prefill_logits_width_bound(max_model_len, compress_ratio)
    if granule_mb <= 0:
        return max(backing_bytes // FP32_BYTES, width)
    cap_numel = max_logits_bytes // FP32_BYTES
    need = max(min(max_num_batched_tokens * width, cap_numel), width)
    return prefill_backing_numel(1, need, granule_mb, max_logits_bytes)


def decode_scratch_numel(
    *,
    max_num_seqs: int,
    num_speculative_tokens: int,
    max_num_batched_tokens: int,
    decode_columns: int,
) -> int:
    """fp32 decode logits ``[decode rows, block-table width x slots per block]``
    at the most decode rows a step schedules."""
    rows = min(max_num_seqs * (num_speculative_tokens + 1), max_num_batched_tokens)
    return rows * decode_columns


def block_indices_bytes(
    *, max_num_batched_tokens: int, token_topk: int, compress_ratio: int
) -> int:
    """``block_indices`` [tokens, token_topk // ratio] int32."""
    return max_num_batched_tokens * (token_topk // compress_ratio) * INT32_BYTES
