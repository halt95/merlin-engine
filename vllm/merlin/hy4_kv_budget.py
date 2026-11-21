# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Hy4 Flash per-GPU KV bytes under PPxTP with and without DCP sharding.

Arithmetic only (no allocation); the CPU gate checks the saving the 4-card fit needs
(with a replicated latent the worst card is short of the fit). The engine's own planner
(``FullAttentionSpec.max_memory_usage_bytes``, ``SlidingWindowSpec``) is the truth at
serve time; this mirrors it so a lane profile can be sized before a GPU run.

Per GPU of a stage, for ``tokens`` pooled tokens:

* sparse latent: ``n_sparse * local_tokens * latent_bytes`` (656 B ``fp8_ds_mla``);
* IndexPool pool: ``n_sparse * ceil(local_tokens / ratio) * pool_bytes`` (132 B:
  128 FP8 + 4 B scale, #68296 ``indexpool_cache.py``);
* IndexPool tail ring (replicated): ``n_sparse * seqs * ring * 131 * 4``;
* SWA latent (replicated): ``n_swa * seqs * (window + spec_tokens) * latent_bytes``.

``local_tokens`` is the rank-0 (largest) shard: ``ceil(tokens / (I*N)) * I`` bound.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from vllm.merlin.hy4_kvshard import Hy4ShardLayout

SPARSE = "sparse"
SWA = "swa"
GIB = 1024**3


@dataclass(frozen=True)
class StageKV:
    stage: int
    n_sparse: int
    n_swa: int
    sparse_latent: int
    indexpool: int
    indexpool_tail: int
    swa_latent: int

    @property
    def total(self) -> int:
        return (
            self.sparse_latent + self.indexpool + self.indexpool_tail + self.swa_latent
        )


def _tail_ring(ratio: int, spec_tokens: int) -> int:
    # #68296 indexpool_cache.py tail_ring_size
    need = -(-(ratio + spec_tokens) // ratio)
    p = 1
    while p < need:
        p *= 2
    return ratio * p


def kv_bytes_per_gpu(
    *,
    layer_kinds: Sequence[str],
    partition: Sequence[int],
    tokens: int,
    seqs: int,
    dcp_size: int,
    interleave: int = 64,
    ratio: int = 4,
    window: int = 512,
    spec_tokens: int = 0,
    latent_bytes: int = 656,
    pool_bytes: int = 132,
) -> list[StageKV]:
    """KV bytes on one GPU of each PP stage. ``layer_kinds`` per layer is
    ``"sparse"`` or ``"swa"``; ``partition`` = layers per stage (sums to the
    layer count). ``dcp_size`` 1 = replicated latent (plain TP)."""
    if sum(partition) != len(layer_kinds):
        raise ValueError("partition must cover every layer")
    layout = Hy4ShardLayout(
        dcp_size=dcp_size,
        dcp_rank=0,
        interleave=interleave if dcp_size > 1 else 1,
        ratio=ratio,
    )
    local_tokens = layout.max_local_tokens(tokens)
    ring = _tail_ring(ratio, spec_tokens)
    out: list[StageKV] = []
    start = 0
    for stage, n in enumerate(partition):
        kinds = layer_kinds[start : start + n]
        start += n
        n_sparse = sum(1 for k in kinds if k == SPARSE)
        n_swa = sum(1 for k in kinds if k == SWA)
        if n_sparse + n_swa != n:
            raise ValueError(f"unknown layer kind in stage {stage}: {set(kinds)}")
        out.append(
            StageKV(
                stage=stage,
                n_sparse=n_sparse,
                n_swa=n_swa,
                sparse_latent=n_sparse * local_tokens * latent_bytes,
                indexpool=n_sparse * (-(-local_tokens // ratio)) * pool_bytes,
                indexpool_tail=n_sparse * seqs * ring * 131 * 4,
                swa_latent=n_swa * seqs * (window + spec_tokens) * latent_bytes,
            )
        )
    return out


def hy4_flash_layer_kinds(num_layers: int = 42) -> list[str]:
    """Hy4 Flash default layout per the checkpoint config (inventory section 4):
    sparse on layers 0, 1, 5, 9, ..., 41; sliding elsewhere. Use the real
    ``config.layer_types`` when a config is at hand."""
    sparse = {0, 1} | set(range(5, num_layers, 4))
    return [SPARSE if i in sparse else SWA for i in range(num_layers)]


def layer_kinds_from_config(layer_types: Sequence[str]) -> list[str]:
    """Map ``config.layer_types`` (#68296 ``configs.py``) to sparse/swa."""
    kinds = []
    for t in layer_types:
        if t == "sliding_attention":
            kinds.append(SWA)
        elif t in (
            "deepseek_sparse_attention",
            "indexed_attention",
            "sparse_attention",
            "sparse",
        ):
            kinds.append(SPARSE)
        else:
            raise ValueError(f"unknown Hy4 Flash layer type {t!r}")
    return kinds
