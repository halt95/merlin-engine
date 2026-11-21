# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
# Modified by the Merlin project; see NOTICE.
"""Flash coarse-only gated IndexPool with bounded paged FP8 scoring, on sm_86.

Adapted from #68296 ``hy_v4_flash/nvidia/compressed_indexer.py:1-210``. The
module, its parameter names and dtypes (``key_value_projection``,
``key_gate_projection``, ``key_norm`` and ``intra_group_ape`` in FP32, :68-74)
and the step order are upstream's. Substitutions:

- DeepGEMM paged MQA scoring -> the Hy4 ``paged_mqa_logits`` (sm_80+, software
  E4M3 decode); no head padding to 32/64, no MX-scale conversion;
- native-FP8 ``fwht_quantize`` -> the Hy4 byte-returning one (same rounding);
- upstream ``write_pool`` -> the Hy4 ``write_index_pool`` on physical group slots.

Under DCP (``decode_context_parallel_size > 1``) the pool is sharded: only the
owner rank writes a group, each rank scores its local groups and the global
top-k comes from ``vllm.merlin.hy4_kvshard.sharded_select_groups``. Output
indices are global per-request token positions on every rank.
"""

import torch
from torch import nn

from vllm import envs
from vllm.config import CUDAGraphMode
from vllm.forward_context import get_forward_context
from vllm.merlin.hy4_kvshard import (
    Hy4ShardLayout,
    local_group_slots,
    shard_pool_writes,
    sharded_select_groups,
)
from vllm.model_executor.layers.linear import ReplicatedLinear

from .indexpool_cache import IndexPoolCache, IndexPoolMetadata
from .indexpool_ops import (
    INDEX_DIM,
    expand_groups,
    fwht_quantize,
    gated_pool,
    paged_mqa_logits,
    rotate_indexer,
    select_groups,
    write_index_pool,
)
from .indexpool_prepare import prepare_groups, write_tails


# MERLIN_HY4_INDEXER_FAST=1: the FWHT/FP8 chain as
# one Triton kernel (bitwise with the torch path) and IndexPool scoring that skips
# tiles past each row's bound (same scores), stable expansion and RoPE.
# Unset/blank/0 = stock. FAST2 independently enables experimental pooling.
def indexer_fast_requested(environ=None, *, name="MERLIN_HY4_INDEXER_FAST") -> bool:
    import os

    raw = ((os.environ if environ is None else environ).get(name, "") or "").strip()
    if raw in ("", "0"):
        return False
    if raw == "1":
        return True
    raise ValueError(f"{name} must be 0 or 1 (got {raw!r})")


_INDEXER_FAST = indexer_fast_requested()
# FAST2: the pooling reduction order needs
# separate bitwise qualification; default off.
_INDEXER_FAST2 = indexer_fast_requested(name="MERLIN_HY4_INDEXER_FAST2")


def _quantize(vectors):
    """fwht_quantize, through the fused kernel when asked for and on CUDA."""
    if _INDEXER_FAST and vectors.is_cuda:
        from vllm.merlin.hy4_kernels import fwht_quantize_fused

        return fwht_quantize_fused(vectors)
    return fwht_quantize(vectors)


def shard_layout_for(vllm_config, ratio: int) -> Hy4ShardLayout:
    """This rank's IndexPool placement over its DCP group (DCP 1: unsharded)."""
    parallel = vllm_config.parallel_config
    size = getattr(parallel, "decode_context_parallel_size", 1) or 1
    if size == 1:
        return Hy4ShardLayout(ratio=ratio)
    from vllm.distributed import get_dcp_group

    return Hy4ShardLayout(
        dcp_size=size,
        dcp_rank=get_dcp_group().rank_in_group,
        interleave=parallel.cp_kv_cache_interleave_size,
        ratio=ratio,
    )


class FlashCompressedIndexer(nn.Module):
    def __init__(
        self,
        vllm_config,
        config,
        topk_indices_buffer,
        prefix,
        *,
        layout: Hy4ShardLayout | None = None,
        all_gather=None,
        rope_parameters=None,
    ):
        super().__init__()
        self.ratio = config.compressed_indexer_compress_ratio
        self.group_budget = config.compressed_indexer_group_topk - 1
        self.num_heads = config.compressed_indexer_num_attention_heads
        if not 1 <= self.num_heads <= 64:
            raise ValueError("IndexPool requires 1..64 scoring heads")
        width = (self.group_budget + 1) * self.ratio
        if topk_indices_buffer.shape[1] != width:
            raise ValueError(
                f"IndexPool writes {width} indices per token; the top-k buffer "
                f"is {topk_indices_buffer.shape[1]} wide"
            )
        self.rope_dim = config.qk_rope_head_dim
        # Upstream deep-copies the config to set the layer's rope (attention.py
        # :38-39); the factory passes config.rope_for_layer(layer_idx) instead.
        self.rope_parameters = (
            config.rope_parameters if rope_parameters is None else rope_parameters
        )
        self.eps = config.rms_norm_eps
        self.topk_indices_buffer = topk_indices_buffer
        self.max_model_len = vllm_config.model_config.max_model_len
        self.logits_budget = envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024 // 4
        for name, input_size, output_size in (
            ("query_projection", config.q_lora_rank, self.num_heads * INDEX_DIM),
            ("head_weight_projection", config.hidden_size, self.num_heads),
            ("token_key_projection", config.hidden_size, INDEX_DIM),
        ):
            setattr(
                self,
                name,
                ReplicatedLinear(
                    input_size,
                    output_size,
                    bias=False,
                    quant_config=vllm_config.quant_config,
                    prefix=f"{prefix}.{name}",
                ),
            )
        self.token_key_norm = nn.RMSNorm(INDEX_DIM, eps=self.eps)
        self.key_value_projection = nn.Linear(
            INDEX_DIM, INDEX_DIM, bias=False, dtype=torch.float32
        )
        self.key_gate_projection = nn.Linear(
            INDEX_DIM, INDEX_DIM, bias=False, dtype=torch.float32
        )
        self.key_norm = nn.RMSNorm(INDEX_DIM, eps=self.eps, dtype=torch.float32)
        self.intra_group_ape = nn.Parameter(
            torch.empty(self.ratio, INDEX_DIM, dtype=torch.float32)
        )
        self.k_cache = IndexPoolCache(
            f"{prefix}.k_cache", vllm_config.cache_config, self.ratio
        )
        self.tail_cache = IndexPoolCache(
            f"{prefix}.tail_cache",
            vllm_config.cache_config,
            self.ratio,
            tail=True,
        )
        self.layout = (
            layout if layout is not None else shard_layout_for(vllm_config, self.ratio)
        )
        if self.layout.ratio != self.ratio:
            raise ValueError("IndexPool shard layout has a different ratio")
        # None: the runtime DCP all-gather (tests inject a fake one).
        self.all_gather = all_gather

    def forward(self, hidden_states, q_c, positions, rope=None):
        output = self.topk_indices_buffer[: hidden_states.shape[0]]
        output.fill_(-1)
        metadata = get_forward_context().attn_metadata
        if metadata is None:
            return output
        if not isinstance(metadata, dict):
            raise TypeError("IndexPool requires per-layer metadata")
        km = metadata[self.k_cache.prefix]
        tm = metadata[self.tail_cache.prefix]
        if not isinstance(km, IndexPoolMetadata) or not isinstance(
            tm, IndexPoolMetadata
        ):
            raise TypeError("IndexPool cache metadata has an incompatible type")
        full_graph = get_forward_context().cudagraph_runtime_mode == CUDAGraphMode.FULL
        return self.select(hidden_states, q_c, positions, km, tm, output, full_graph)

    def select(self, hidden_states, q_c, positions, km, tm, output, full_graph=False):
        """One IndexPool step on bound caches; ``output`` is pre-filled with -1."""
        tokens = km.num_actual_tokens
        if tokens == 0 or km.seq_lens.numel() == 0:
            return output
        keys = self.token_key_norm(
            self.token_key_projection(hidden_states)[0]
        ).bfloat16()
        cache, tail = self.k_cache.kv_cache, self.tail_cache.kv_cache
        (
            raw,
            group_positions,
            group_owner,
            groups,
            group_live,
            owner,
            logical,
            upper,
            live,
        ) = prepare_groups(
            keys,
            positions,
            tail,
            tm.block_table,
            km.query_start_loc,
            km.seq_lens,
            tokens,
            self.ratio,
        )
        if raw.shape[0]:
            values = self.key_value_projection(raw.float())
            gates = self.key_gate_projection(raw.float())
            pooled = gated_pool(
                values, gates, self.intra_group_ape, self.key_norm.weight, self.eps,
                experimental=_INDEXER_FAST2,
            ).bfloat16()
            quant, scales = _quantize(
                rotate_indexer(
                    pooled, group_positions, self.rope_dim, self.rope_parameters,
                    fast=_INDEXER_FAST,
                )
            )
            # DCP: every rank pools (inputs are replicated); the owner writes.
            local, active = shard_pool_writes(groups, group_live, self.layout)
            slots = local_group_slots(km.block_table, group_owner, local, active)
            write_index_pool(cache, slots, quant, scales)
        write_tails(
            keys, positions, tail, tm.block_table, km.query_start_loc, km.seq_lens
        )
        query = self.query_projection(q_c)[0][:tokens].reshape(
            tokens, self.num_heads, INDEX_DIM
        )
        query = rotate_indexer(
            query, positions[..., :tokens], self.rope_dim, self.rope_parameters,
            fast=_INDEXER_FAST,
        )
        quant_q, scales_q = _quantize(
            query.reshape(-1, INDEX_DIM).bfloat16()
        )
        quant_q = quant_q.reshape(tokens, self.num_heads, INDEX_DIM)
        weights = self.head_weight_projection(hidden_states)[0][:tokens].float()
        weights = weights * (self.num_heads**-0.5 * INDEX_DIM**-0.5)
        weights = weights * scales_q.reshape(tokens, self.num_heads)
        max_seq_len = self.max_model_len if full_graph else km.max_seq_len
        # Upstream width on DCP 1; the per-rank local width under DCP.
        width = self.layout.local_scoring_width(max_seq_len)
        if self.logits_budget < width:
            raise ValueError("IndexPool logits budget cannot hold a single scoring row")
        # Chunk bounds depend only on tokens and width, equal on every DCP rank,
        # so every rank issues the same all-gathers.
        chunk = max(1, self.logits_budget // width)
        for begin in range(0, tokens, chunk):
            end = min(tokens, begin + chunk)
            bounds = upper[begin:end]
            if self.layout.sharded:
                local_bounds = self.layout.num_local_groups(bounds.long())
            else:
                local_bounds = bounds
            logits = paged_mqa_logits(
                quant_q[begin:end],
                cache,
                weights[begin:end],
                owner[begin:end],
                local_bounds,
                km.block_table,
                width,
                early_exit=_INDEXER_FAST,
            )
            if self.layout.sharded:
                recalled = sharded_select_groups(
                    logits, bounds, self.layout, self.group_budget, self.all_gather
                )
            else:
                recalled = select_groups(logits, bounds, self.group_budget)
            expand_groups(  # writes into output
                recalled,
                logical[begin:end],
                self.ratio,
                active=live[begin:end],
                fast=_INDEXER_FAST,
                out=output[begin:end],
            )
        return output
