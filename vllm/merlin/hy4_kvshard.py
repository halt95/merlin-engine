# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Hy4 Flash: the sparse-MLA latent KV and the IndexPool cache sharded over TP (DCP).

Hy4 Flash (``hy_v4_flash``) has 12 sparse-MLA layers whose 576-wide latent row
(656 B in ``fp8_ds_mla``) is one MQA "head": plain TP2 keeps a full copy on both
ranks. Decode context parallelism (DCP, this tree's ``decode_context_parallel_size``)
reuses the TP ranks and stores each token on one rank only. This module is the
Hy4-specific part of that, written in plain torch so the CPU gate can check it and
so it does not wait on the sm_86 kernel:

* ``Hy4ShardLayout`` -- which rank owns a token / an IndexPool group and where it
  sits in that rank's local cache. Same formula as the block-table slot-mapping
  kernel (``vllm/v1/worker/gpu/block_table.py``) and
  ``triton_filter_and_convert_dcp_index``: ``owner(t) = (t // I) % N``,
  ``local(t) = (t // (I*N)) * I + t % I`` (I = ``cp_kv_cache_interleave_size``).
  IndexPool groups are ``ratio`` consecutive tokens; with ``I % ratio == 0`` a group
  and all its tokens live on the same rank, so the pooled key sits next to the
  latent rows it selects.
* The learnable sink, single owner. Under DCP every rank computes *all* heads of its
  DCP group over its token shard and the shards are merged by log-sum-exp. A sink
  added on every rank would be counted N times. The sink therefore lives on DCP
  rank 0 only (the full group head set); the other ranks carry ``-inf`` (a no-op
  sink). Rank 0 always holds token 0, so the empty-shard mask in the combine never
  drops it. ``make_dcp_owner_attention_sink`` builds and loads that vector;
  ``fold_sink_into_partial`` applies it for a kernel without native sink support.
* ``sparse_mla_reference`` / ``lse_combine_reference`` -- torch reference attention
  and combine (natural-log LSE) the tests and the kernels compare against.
* IndexPool selection under DCP: each rank scores only its own groups, takes its
  local top-``budget`` with scores, the ``(score, global group id)`` candidates are
  all-gathered, and the global top-``budget`` is taken from the union. Exact: a group
  in the global top-k is in its owner's local top-k. Output format is upstream
  ``select_groups``' (ascending ids, ``-1`` padded), so ``expand_groups`` is unchanged.
* Cache-spec policy: sparse latent and IndexPool pool are DCP-sharded; the 30 SWA
  layers (512-token window, ~8.8 MB per sequence) and the IndexPool tail ring stay
  replicated.

The model port, the attention backend, pipeline parallelism and the hy4flash lane
all go through the functions below.
"""

from __future__ import annotations

import dataclasses
import enum
from collections.abc import Callable
from dataclasses import dataclass

import torch

NEG_INF = float("-inf")


def _select_fused_requested() -> bool:
    from vllm.merlin.hy4_group_select import select_fused_requested

    return select_fused_requested()


_SELECT_FUSED = _select_fused_requested()
SINK_OWNER_RANK = 0
"""DCP rank that carries the learnable sink. Must be 0: rank 0 owns token 0, so it is
never an empty shard for a real row (see ``mask_dcp_empty_shards_``)."""

POOL_PAGE_SIZE = 64
"""IndexPool kernel page, in groups (#68296 ``indexpool_cache.py:28``)."""

IntOrTensor = int | torch.Tensor


def _clamp(x: IntOrTensor, lo: int, hi: int) -> IntOrTensor:
    if isinstance(x, torch.Tensor):
        return x.clamp(lo, hi)
    return max(lo, min(hi, x))


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Hy4ShardLayout:
    """Token and IndexPool-group placement over one DCP group.

    ``dcp_size``/``dcp_rank``: the DCP group (== the TP group of a PP stage when
    DCP == TP). ``interleave``: ``cp_kv_cache_interleave_size`` in tokens.
    ``ratio``: ``compressed_indexer_compress_ratio`` (tokens per pooled key).
    """

    dcp_size: int = 1
    dcp_rank: int = 0
    interleave: int = 1
    ratio: int = 4

    def __post_init__(self) -> None:
        if self.dcp_size < 1:
            raise ValueError(f"dcp_size must be >= 1, got {self.dcp_size}")
        if not 0 <= self.dcp_rank < self.dcp_size:
            raise ValueError(f"dcp_rank {self.dcp_rank} not in [0, {self.dcp_size})")
        if self.interleave < 1 or self.ratio < 1:
            raise ValueError("interleave and ratio must be positive")
        if self.dcp_size > 1 and self.interleave % self.ratio:
            raise ValueError(
                "Hy4 DCP needs cp_kv_cache_interleave_size "
                f"({self.interleave}) to be a multiple of the IndexPool ratio "
                f"({self.ratio}) so a pooled group never straddles two ranks"
            )

    @property
    def sharded(self) -> bool:
        return self.dcp_size > 1

    @property
    def group_interleave(self) -> int:
        """Interleave in IndexPool groups (``I / ratio``)."""
        return max(1, self.interleave // self.ratio) if self.sharded else 1

    def for_rank(self, rank: int) -> Hy4ShardLayout:
        return dataclasses.replace(self, dcp_rank=rank)

    # generic unit arithmetic ------------------------------------------------
    def _owner(self, x: IntOrTensor, unit: int) -> IntOrTensor:
        if not self.sharded:
            return x * 0 if isinstance(x, torch.Tensor) else 0
        return (x // unit) % self.dcp_size

    def _to_local(self, x: IntOrTensor, unit: int) -> IntOrTensor:
        if not self.sharded:
            return x
        return (x // (unit * self.dcp_size)) * unit + x % unit

    def _to_global(self, local: IntOrTensor, unit: int, rank: int) -> IntOrTensor:
        if not self.sharded:
            return local
        return ((local // unit) * self.dcp_size + rank) * unit + local % unit

    def _count(self, n: IntOrTensor, unit: int, rank: int) -> IntOrTensor:
        """How many of ``[0, n)`` this rank owns."""
        if not self.sharded:
            return n
        span = unit * self.dcp_size
        return (n // span) * unit + _clamp(n % span - rank * unit, 0, unit)

    # tokens ------------------------------------------------------------------
    def token_owner(self, pos: IntOrTensor) -> IntOrTensor:
        return self._owner(pos, self.interleave)

    def token_to_local(self, pos: IntOrTensor) -> IntOrTensor:
        return self._to_local(pos, self.interleave)

    def local_to_token(
        self, local: IntOrTensor, rank: int | None = None
    ) -> IntOrTensor:
        r = self.dcp_rank if rank is None else rank
        return self._to_global(local, self.interleave, r)

    def num_local_tokens(self, n: IntOrTensor, rank: int | None = None) -> IntOrTensor:
        r = self.dcp_rank if rank is None else rank
        return self._count(n, self.interleave, r)

    def max_local_tokens(self, n: int) -> int:
        """Upper bound of ``num_local_tokens(n)`` over ranks (rank 0 is the max)."""
        return int(self.num_local_tokens(n, rank=0))

    # IndexPool groups ----------------------------------------------------------
    def group_owner(self, group: IntOrTensor) -> IntOrTensor:
        return self._owner(group, self.group_interleave)

    def group_to_local(self, group: IntOrTensor) -> IntOrTensor:
        return self._to_local(group, self.group_interleave)

    def local_to_group(
        self, local: IntOrTensor, rank: int | None = None
    ) -> IntOrTensor:
        r = self.dcp_rank if rank is None else rank
        return self._to_global(local, self.group_interleave, r)

    def num_local_groups(
        self, n_groups: IntOrTensor, rank: int | None = None
    ) -> IntOrTensor:
        """Local complete groups among global groups ``[0, n_groups)``."""
        r = self.dcp_rank if rank is None else rank
        return self._count(n_groups, self.group_interleave, r)

    def local_scoring_width(self, max_seq_len: int, page: int = POOL_PAGE_SIZE) -> int:
        """IndexPool logits width on this rank, in groups, page-aligned.

        The per-rank analogue of #68296 ``compressed_indexer.py`` ``width``.
        """
        local_groups = -(-self.max_local_tokens(max_seq_len) // self.ratio)
        return max(page, -(-local_groups // page) * page)

    # index filtering ----------------------------------------------------------
    def filter_global_to_local(self, token_indices: torch.Tensor) -> torch.Tensor:
        """Global per-request token ids (``-1`` padded) -> this rank's local token
        positions, ``-1`` where another rank owns the token. Positional (not
        compacted); the torch twin of ``triton_filter_and_convert_dcp_index`` before
        the block-table lookup."""
        valid = token_indices >= 0
        if self.sharded:
            valid = valid & (self.token_owner(token_indices) == self.dcp_rank)
        local = self.token_to_local(token_indices.clamp_min(0))
        return torch.where(valid, local, torch.full_like(token_indices, -1))


def validate_hy4_dcp(
    *,
    tp_size: int,
    dcp_size: int,
    pcp_size: int,
    interleave: int,
    block_size: int,
    ratio: int,
    num_heads: int,
    swa_num_heads: int,
) -> None:
    """Reject DCP settings this sharding cannot run. Called from the model's
    ``verify_and_update_config`` and the hy4flash lane."""
    if dcp_size == 1:
        return
    if pcp_size != 1:
        raise ValueError("Hy4 Flash DCP does not support prefill context parallel")
    if tp_size % dcp_size:
        raise ValueError(f"tp_size={tp_size} must be divisible by dcp_size={dcp_size}")
    if block_size % interleave:
        raise ValueError(
            f"block_size={block_size} must be divisible by "
            f"cp_kv_cache_interleave_size={interleave}"
        )
    if interleave % ratio:
        raise ValueError(
            f"cp_kv_cache_interleave_size={interleave} must be a multiple of "
            f"compressed_indexer_compress_ratio={ratio}"
        )
    for name, heads in (
        ("num_attention_heads", num_heads),
        ("swa_num_attention_heads", swa_num_heads),
    ):
        if heads % tp_size:
            raise ValueError(f"{name}={heads} must be divisible by tp_size={tp_size}")


# ---------------------------------------------------------------------------
# Cache-spec policy
# ---------------------------------------------------------------------------


class Hy4CacheKind(enum.Enum):
    SPARSE_LATENT = "sparse_latent"
    SWA_LATENT = "swa_latent"
    INDEXPOOL = "indexpool"
    INDEXPOOL_TAIL = "indexpool_tail"


DCP_SHARDED: dict[Hy4CacheKind, bool] = {
    Hy4CacheKind.SPARSE_LATENT: True,
    Hy4CacheKind.SWA_LATENT: False,
    Hy4CacheKind.INDEXPOOL: True,
    Hy4CacheKind.INDEXPOOL_TAIL: False,
}


class Hy4ReplicatedKVImplMixin:
    """Put first in the bases of the Hy4 SWA attention impl, e.g.
    ``class Hy4TritonSWAImpl(Hy4ReplicatedKVImplMixin, Hy4TritonSparseImpl)``.

    The SWA cache is replicated over DCP, so the layer runs as plain TP: no query
    all-gather, no combine, local heads, ordinary TP sink. ``AttentionImplBase.
    __new__`` set ``dcp_world_size`` from the global config; this resets it
    *before* the base ``__init__`` runs, so ``MLAAttention`` builds no
    ``MLADCPManager`` (``mla_attention.py:695``) and ``get_kv_cache_spec`` (hook
    ``hy4-kvshard``) emits ``dcp_sharded=False``. The SWA group's sparse-MLA
    metadata builder follows the spec (hook in ``sparse_mla_attention.py``)."""

    kv_dcp_replicated = True

    def __init__(self, *args, **kwargs):
        self.dcp_world_size, self.dcp_rank = 1, 0
        self.total_cp_world_size = getattr(self, "pcp_world_size", 1)
        self.total_cp_rank = getattr(self, "pcp_rank", 0)
        self.need_to_return_lse_for_decode = False
        super().__init__(*args, **kwargs)


def apply_dcp_policy(spec, kind: Hy4CacheKind):
    """Return ``spec`` with ``dcp_sharded`` set by Hy4's policy (specs are frozen)."""
    want = DCP_SHARDED[kind]
    if getattr(spec, "dcp_sharded", None) == want:
        return spec
    return dataclasses.replace(spec, dcp_sharded=want)


# ---------------------------------------------------------------------------
# Learnable sink: one owner per DCP group
# ---------------------------------------------------------------------------


def load_dcp_owner_attention_sink(
    param: torch.Tensor,
    loaded_weight: torch.Tensor,
    *,
    tp_rank: int,
    tp_size: int,
    dcp_size: int,
    dcp_rank: int,
) -> None:
    """Load the DCP-group head slice of a global ``[num_heads]`` sink.

    The DCP group of TP rank ``r`` is TP ranks ``[r - r % N, r - r % N + N)``
    (``parallel_state`` builds DCP groups from consecutive TP ranks when PCP == 1),
    and the query all-gather concatenates their heads in rank order, so the group's
    heads are the contiguous slice ``[g * N * h, (g + 1) * N * h)`` of the
    checkpoint vector (``h`` = heads per TP rank). Only ``SINK_OWNER_RANK`` keeps
    the values; the other ranks and any padded lanes hold ``-inf``.
    """
    heads = loaded_weight.numel()
    if heads % tp_size:
        raise ValueError(
            f"Attention sink heads ({heads}) not divisible by tp_size ({tp_size})"
        )
    if dcp_rank != tp_rank % dcp_size:
        raise ValueError(
            f"DCP rank {dcp_rank} does not match TP rank {tp_rank} % {dcp_size}; "
            "the sink slice assumes DCP groups of consecutive TP ranks"
        )
    local = heads // tp_size
    width = local * dcp_size
    start = (tp_rank // dcp_size) * width
    if param.numel() < width:
        raise ValueError(f"sink param has {param.numel()} lanes, needs {width}")
    with torch.no_grad():
        param.data.fill_(NEG_INF)
        if dcp_rank == SINK_OWNER_RANK:
            param.data[:width].copy_(loaded_weight.reshape(-1).narrow(0, start, width))


def make_dcp_owner_attention_sink(
    num_local_heads: int,
    *,
    tp_rank: int,
    tp_size: int,
    dcp_size: int,
    dcp_rank: int,
    dtype: torch.dtype = torch.float32,
) -> torch.nn.Parameter:
    """Sink parameter for a DCP-sharded sparse layer: ``[num_local_heads * dcp_size]``
    fp32, real on the owner rank, ``-inf`` elsewhere. Same checkpoint name as the
    TP sink (``learnable_sink_param``); only the loader differs."""
    sink = torch.nn.Parameter(
        torch.full((num_local_heads * dcp_size,), NEG_INF, dtype=dtype),
        requires_grad=False,
    )

    def _loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
        load_dcp_owner_attention_sink(
            param,
            loaded_weight,
            tp_rank=tp_rank,
            tp_size=tp_size,
            dcp_size=dcp_size,
            dcp_rank=dcp_rank,
        )

    sink.weight_loader = _loader  # type: ignore[attr-defined]
    _tag_sink(sink, dcp_size, dcp_rank == SINK_OWNER_RANK)
    return sink


def _tag_sink(sink: torch.nn.Parameter, dcp_size: int, owner: bool) -> None:
    """Record the sink's DCP layout on the parameter, for ``validate_dfly_kv_alias``."""
    sink.hy4_dcp_size = dcp_size  # type: ignore[attr-defined]
    sink.hy4_sink_owner = owner  # type: ignore[attr-defined]


def active_sink_dcp(
    dcp_sharded_kv: bool,
    active_dcp_size: int | None,
    group_size: int,
    group_rank: int,
) -> tuple[int, int]:
    """DCP ``(size, rank)`` a sink must be built for.

    ``active_dcp_size``: ``decode_context_parallel_size`` of the config being
    constructed (``None`` = no current config, as in unit tests). A model built at
    DCP 1 inside a DCP>1 process group (an independent replicated draft, the
    runner's ``model_runner.py:800-807`` arrangement) gets the plain TP sink, the
    same rule ``AttentionImplBase.__new__`` uses for the impl."""
    if not dcp_sharded_kv or group_size <= 1:
        return 1, 0
    if active_dcp_size is not None and active_dcp_size <= 1:
        return 1, 0
    return group_size, group_rank


def make_layer_sink(
    num_local_heads: int, *, dcp_sharded_kv: bool
) -> torch.nn.Parameter:
    """Runtime factory for the model port: the DCP owner sink for a sharded sparse
    layer, the ordinary TP sink (``make_attention_sink``) otherwise."""
    from vllm.config import get_current_vllm_config_or_none
    from vllm.distributed import (
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_world_size,
    )
    from vllm.distributed.parallel_state import get_dcp_world_size_and_rank
    from vllm.model_executor.model_loader.weight_utils import make_attention_sink

    config = get_current_vllm_config_or_none()
    active = (
        None if config is None else config.parallel_config.decode_context_parallel_size
    )
    group_size, group_rank = get_dcp_world_size_and_rank(True)
    dcp_size, dcp_rank = active_sink_dcp(dcp_sharded_kv, active, group_size, group_rank)
    if dcp_size == 1:
        sink = make_attention_sink(num_local_heads, torch.float32)
        _tag_sink(sink, 1, True)
        return sink
    return make_dcp_owner_attention_sink(
        num_local_heads,
        tp_rank=get_tensor_model_parallel_rank(),
        tp_size=get_tensor_model_parallel_world_size(),
        dcp_size=dcp_size,
        dcp_rank=dcp_rank,
    )


def fold_sink_into_partial(
    out: torch.Tensor, lse: torch.Tensor, sink: torch.Tensor | None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fold a sink into one shard's partial result: ``lse' = log(e^lse + e^s)``,
    ``out' = out * e^(lse - lse')``. For a kernel that cannot apply the sink; call it
    with the owner vector (``-inf`` lanes are no-ops). ``out [T, H, D]``,
    ``lse [T, H]`` natural log, ``sink [H]``. Rows with ``lse = -inf`` give
    ``out' = 0`` even if ``out`` holds garbage."""
    if sink is None:
        return out, lse
    lse32 = lse.float()
    s = sink.float().reshape(1, -1).expand_as(lse32)
    new = torch.logaddexp(lse32, s)
    finite = torch.isfinite(lse32) & torch.isfinite(new)
    scale = torch.where(finite, torch.exp(lse32 - torch.where(finite, new, 0.0)), 0.0)
    folded = torch.where(
        finite[..., None],
        out.float() * scale[..., None],
        torch.zeros_like(out, dtype=torch.float32),
    )
    return folded.to(out.dtype), new.to(lse.dtype)


def validate_dfly_kv_alias(source, dest) -> None:
    """Fail closed unless a DFly2 draft layer can read its target layer's (possibly
    DCP-sharded) KV through the same DCP path.

    ``source``/``dest``: the target and draft ``MLAAttention`` modules (anything
    with ``impl``, ``dcp_manager`` and ``get_kv_cache_spec``). Checked: equal DCP
    size and rank on the impls, the same cache sharding, both or neither running
    the query gather + combine, and sinks built for the same DCP layout with the
    same owner. A DCP-1 draft aliasing a DCP>1 target is rejected: a replicated
    draft cannot read a sharded target's history. There is no replicated
    fallback; it would have to copy the target KV and account for its memory."""
    s_impl, d_impl = source.impl, dest.impl
    s_dcp = getattr(s_impl, "dcp_world_size", 1)
    d_dcp = getattr(d_impl, "dcp_world_size", 1)
    if s_dcp != d_dcp:
        raise ValueError(
            f"DFly draft runs at DCP {d_dcp} but its target KV layer is DCP {s_dcp}; "
            "a replicated draft cannot alias a sharded target cache"
        )
    if getattr(s_impl, "dcp_rank", 0) != getattr(d_impl, "dcp_rank", 0):
        raise ValueError("DFly draft and target disagree on the DCP rank")
    if bool(getattr(s_impl, "kv_dcp_replicated", False)) != bool(
        getattr(d_impl, "kv_dcp_replicated", False)
    ):
        raise ValueError("DFly draft and target disagree on cache replication")
    if (getattr(source, "dcp_manager", None) is None) != (
        getattr(dest, "dcp_manager", None) is None
    ):
        raise ValueError("DFly draft and target must both run the DCP gather/combine")
    s_sink, d_sink = getattr(s_impl, "sinks", None), getattr(d_impl, "sinks", None)
    if (s_sink is None) != (d_sink is None):
        raise ValueError("DFly draft and target must both apply a learnable sink")
    if s_sink is not None:
        if tuple(s_sink.shape) != tuple(d_sink.shape):
            raise ValueError(
                f"DFly sink layouts differ: target {tuple(s_sink.shape)}, "
                f"draft {tuple(d_sink.shape)}"
            )
        for attr in ("hy4_dcp_size", "hy4_sink_owner"):
            if getattr(s_sink, attr, None) != getattr(d_sink, attr, None):
                raise ValueError(f"DFly draft and target sinks differ in {attr}")


# ---------------------------------------------------------------------------
# Torch reference attention and combine
# ---------------------------------------------------------------------------


def sparse_mla_reference(
    q: torch.Tensor,
    cache: torch.Tensor,
    indices: torch.Tensor,
    scale: float,
    v_dim: int,
    sink: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sparse MQA over latent rows, fp32. ``q [T, H, Dqk]`` (absorbed query,
    ``kv_lora_rank + rope``), ``cache [R, Dqk]`` dequantised rows, ``indices [T, K]``
    row ids into ``cache`` (``-1`` = unused). Value = first ``v_dim`` columns.
    Returns ``out [T, H, v_dim]`` and ``lse [T, H]`` (natural log, sink included).
    Empty row: ``out = 0``, ``lse = sink`` (or ``-inf``)."""
    T, H, _ = q.shape
    valid = indices >= 0
    rows = cache.float()[indices.clamp_min(0).long()]  # [T, K, D]
    scores = torch.einsum("thd,tkd->thk", q.float(), rows) * scale
    scores = scores.masked_fill(~valid[:, None, :], NEG_INF)
    if sink is not None:
        s = sink.float().reshape(1, H, 1).expand(T, H, 1)
        lse = torch.logsumexp(torch.cat((scores, s), dim=-1), dim=-1)
    else:
        lse = torch.logsumexp(scores, dim=-1)
    finite = torch.isfinite(lse)
    p = torch.exp(scores - torch.where(finite, lse, 0.0)[..., None])
    p = torch.where(finite[..., None], p, 0.0)
    out = torch.einsum("thk,tkd->thd", p, rows[..., :v_dim])
    return out, lse


def lse_combine_reference(
    outs: torch.Tensor, lses: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact merge of N shard partials. ``outs [N, T, H, D]``, ``lses [N, T, H]``
    natural log. Shards with ``-inf`` contribute nothing (their ``out`` may be
    garbage). Same maths as ``_lse_weighted_combine`` in ``v1/attention/ops/dcp.py``."""
    lses32 = lses.float()
    total = torch.logsumexp(lses32, dim=0)
    finite = torch.isfinite(lses32) & torch.isfinite(total)[None]
    safe_total = torch.where(torch.isfinite(total), total, 0.0)[None]
    w = torch.where(finite, torch.exp(lses32 - safe_total), 0.0)
    contrib = torch.where(finite[..., None], outs.float() * w[..., None], 0.0)
    return contrib.sum(0), total


# ---------------------------------------------------------------------------
# IndexPool selection under DCP
# ---------------------------------------------------------------------------


def select_groups_with_scores(
    logits: torch.Tensor, upper: torch.Tensor, budget: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-``budget`` columns below ``upper`` per row, with their scores.
    ``logits [T, W]``, ``upper [T]`` (columns ``>= upper`` are masked). Returns
    ``scores [T, budget]`` fp32 (``-inf`` padded) and ``ids [T, budget]`` int32
    (``-1`` padded), unsorted. Fixed shapes (graph-safe). The scored half of
    #68296 ``indexpool_ops.py`` ``select_groups``."""
    T, W = logits.shape
    columns = torch.arange(W, device=logits.device)
    scores = logits.float().masked_fill(columns[None] >= upper.reshape(-1, 1), NEG_INF)
    count = min(budget, W)
    values, ids = scores.topk(count, dim=-1)
    ids = torch.where(values != NEG_INF, ids, -1).int()
    if count < budget:
        values = torch.nn.functional.pad(values, (0, budget - count), value=NEG_INF)
        ids = torch.nn.functional.pad(ids, (0, budget - count), value=-1)
    return values, ids


def merge_group_candidates(
    scores: torch.Tensor, group_ids: torch.Tensor, budget: int
) -> torch.Tensor:
    """Global top-``budget`` from per-rank candidates. ``scores``/``group_ids``
    ``[N, T, K]`` (global ids, ``-1`` = none). Returns ``[T, budget]`` int32 in
    upstream ``select_groups`` format: ascending ids, ``-1`` padded at the end."""
    N, T, K = scores.shape
    s = scores.permute(1, 0, 2).reshape(T, N * K).float()
    g = group_ids.permute(1, 0, 2).reshape(T, N * K)
    s = s.masked_fill(g < 0, NEG_INF)
    count = min(budget, N * K)
    values, pick = s.topk(count, dim=-1)
    chosen = g.gather(1, pick)
    sentinel = torch.iinfo(torch.int32).max
    chosen = torch.where(values != NEG_INF, chosen, sentinel).int()
    chosen = chosen.sort(dim=-1).values
    chosen = chosen.masked_fill(chosen == sentinel, -1)
    if count < budget:
        chosen = torch.nn.functional.pad(chosen, (0, budget - count), value=-1)
    return chosen


AllGather = Callable[[torch.Tensor], torch.Tensor]
"""``x [T, K, C] -> [N, T, K, C]`` over the DCP group, rank order."""


def dcp_all_gather() -> AllGather:
    """Runtime all-gather over this tree's DCP group."""
    from vllm.distributed import get_dcp_group

    group = get_dcp_group()

    def _gather(x: torch.Tensor) -> torch.Tensor:
        return group.all_gather(x.unsqueeze(0).contiguous(), dim=0)

    return _gather


def sharded_select_groups(
    logits_local: torch.Tensor,
    upper: torch.Tensor,
    layout: Hy4ShardLayout,
    budget: int,
    all_gather: AllGather | None = None,
) -> torch.Tensor:
    """IndexPool group selection with the pool sharded over DCP.

    ``logits_local [T, W_local]``: scores of this rank's local groups (column = local
    group index, as the paged MQA logits kernel produces over the local pool
    table). ``upper [T]``: global complete-group count per token (#68296
    ``prepare_groups`` ``upper``). Returns global group ids in ``select_groups``
    format, identical on every rank, ready for ``expand_groups``.
    One collective of ``T * budget * 2`` fp32 per layer (8 KB per token at
    budget 511). Group ids travel as fp32: exact below 2**24 groups (64 M tokens).
    """
    # MERLIN_HY4_INDEXER_SELECT=1: the same
    # selection with the glue around topk/sort fused (bitwise; hy4_group_select)
    if _SELECT_FUSED and logits_local.is_cuda:
        from vllm.merlin import hy4_group_select as fused

        if not layout.sharded:
            return fused.sorted_select(logits_local, upper, layout, budget)
        packed = fused.local_candidates(logits_local, upper, layout, budget)
        gather = all_gather if all_gather is not None else dcp_all_gather()
        return fused.merge_candidates(gather(packed), budget)
    if not layout.sharded:
        return _sorted_select(logits_local, upper, budget)
    packed = local_group_candidates(logits_local, upper, layout, budget)
    gather = all_gather if all_gather is not None else dcp_all_gather()
    return merge_packed_candidates(gather(packed), budget)


def local_group_candidates(
    logits_local: torch.Tensor,
    upper: torch.Tensor,
    layout: Hy4ShardLayout,
    budget: int,
) -> torch.Tensor:
    """This rank's ``(score, global group id)`` candidates, packed ``[T, budget, 2]``
    fp32 for one all-gather."""
    local_upper = layout.num_local_groups(upper.long())
    scores, local_ids = select_groups_with_scores(logits_local, local_upper, budget)
    gids = torch.where(local_ids >= 0, layout.local_to_group(local_ids.long()), -1)
    return torch.stack((scores, gids.float()), dim=-1)


def merge_packed_candidates(gathered: torch.Tensor, budget: int) -> torch.Tensor:
    """``[N, T, K, 2]`` gathered candidates -> ``select_groups``-format ids."""
    return merge_group_candidates(gathered[..., 0], gathered[..., 1].long(), budget)


def _sorted_select(
    logits: torch.Tensor, upper: torch.Tensor, budget: int
) -> torch.Tensor:
    scores, ids = select_groups_with_scores(logits, upper, budget)
    return merge_group_candidates(scores[None], ids[None].long(), budget)


def shard_pool_writes(
    groups: torch.Tensor, active: torch.Tensor, layout: Hy4ShardLayout
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map completed global groups to this rank's local pool rows. Returns
    ``(local_groups, active_local)`` to pass to #68296 ``write_pool`` in place of
    ``(groups, active)``; the pool block table is already the rank-local one.
    Every rank computes the pooled key (inputs are replicated); only the owner
    writes it."""
    if not layout.sharded:
        return groups, active
    owned = layout.group_owner(groups) == layout.dcp_rank
    local = torch.where(owned, layout.group_to_local(groups), torch.zeros_like(groups))
    return local, active & owned


def local_group_slots(
    table: torch.Tensor,
    owners: torch.Tensor,
    local_groups: torch.Tensor,
    active: torch.Tensor,
    page: int = POOL_PAGE_SIZE,
) -> torch.Tensor:
    """Physical pool slots for ``write_index_pool(cache, slots, ...)``:
    ``table[owner, lg // page] * page + lg % page``, ``-1`` where not active.
    ``table`` is the rank-local pool table in ``page``-group kernel pages;
    ``(local_groups, active)`` come from ``shard_pool_writes``."""
    cols = (local_groups // page).clamp(0, table.shape[1] - 1)
    base = table[owners.long(), cols.long()].long() * page + local_groups.long() % page
    return torch.where(active, base, torch.full_like(base, -1))


def gather_local_columns(
    logits_full: torch.Tensor, layout: Hy4ShardLayout
) -> torch.Tensor:
    """Test helper: the columns of a full ``[T, G]`` group-logit matrix this rank
    would score, in local order (what the paged kernel yields over the local pool)."""
    G = logits_full.shape[1]
    n_local = int(layout.num_local_groups(G))
    local = torch.arange(n_local, device=logits_full.device)
    return logits_full[:, layout.local_to_group(local)]
