# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modified by the Merlin project; see NOTICE.
"""Adapted from #68296 Flash compressed_indexer/cache/prepare and swa."""

from dataclasses import dataclass, replace
from math import lcm

import torch
from torch import nn

from vllm import envs
from vllm.config import CUDAGraphMode, get_current_vllm_config
from vllm.distributed import get_dcp_group
from vllm.forward_context import get_forward_context
from vllm.merlin.hy4_kernels import paged_mqa_logits, write_index_pool
from vllm.merlin.hy4_kvshard import (
    Hy4ShardLayout,
    local_group_slots,
    shard_pool_writes,
    sharded_select_groups,
)
from vllm.merlin.hy4_reference import expand_groups, fwht_quantize, gated_pool
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.models.deepseek_v2 import DeepseekV32IndexerCache
from vllm.models.hy_v4.indexpool import (
    logical_window_indices,
    prepare_groups,
    rotate_indexer,
)
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backend import AttentionCGSupport, AttentionMetadataBuilder
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerBackend,
    KpoolTailBackend,
)
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    MLAAttentionSpec,
    SparseCacheRole,
)

POOL_PAGE_SIZE = 64


@dataclass
class IndexPoolMetadata:
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    num_actual_tokens: int
    max_seq_len: int


class IndexPoolMetadataBuilder(AttentionMetadataBuilder[IndexPoolMetadata]):
    _cudagraph_support = AttentionCGSupport.ALWAYS

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.max_model_len = vllm_config.model_config.max_model_len

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        cm = common_attn_metadata
        return IndexPoolMetadata(
            cm.query_start_loc[: cm.num_reqs + 1],
            cm.seq_lens[: cm.num_reqs],
            cm.block_table_tensor[: cm.num_reqs],
            cm.num_actual_tokens,
            cm.max_seq_len,
        )

    def build_for_cudagraph_capture(self, common_attn_metadata):
        return replace(
            self.build(0, common_attn_metadata),
            max_seq_len=self.max_model_len,
        )


class IndexPoolBackend(DeepseekV32IndexerBackend):
    @staticmethod
    def get_name():
        return "HYV4_FLASH_INDEXPOOL"

    @staticmethod
    def get_builder_cls():
        return IndexPoolMetadataBuilder

    @staticmethod
    def get_supported_kernel_block_sizes(kv_cache_spec=None):
        if kv_cache_spec is None:
            config = get_current_vllm_config().model_config.hf_text_config
            ratio = config.compressed_indexer_compress_ratio
        else:
            ratio = int(kv_cache_spec.tokens_per_state)
        return [ratio * POOL_PAGE_SIZE]

    @classmethod
    def get_supported_head_sizes(cls):
        return [132]

    @classmethod
    def supports_pcp(cls):
        return False

    @classmethod
    def supports_dcp(cls):
        return True

    @classmethod
    def supports_device_cpu_query_lens_mismatch(cls):
        return True


class IndexPoolTailBackend(KpoolTailBackend):
    @staticmethod
    def get_name():
        return "HYV4_FLASH_INDEXPOOL_TAIL"

    @staticmethod
    def get_builder_cls():
        return IndexPoolMetadataBuilder

    @classmethod
    def supports_pcp(cls):
        return False

    @classmethod
    def supports_dcp(cls):
        return True


def tail_ring_size(ratio, speculative_tokens):
    groups = (ratio + speculative_tokens + ratio - 1) // ratio
    return ratio * (1 << (groups - 1).bit_length())


class IndexPoolCache(DeepseekV32IndexerCache):
    def __init__(self, prefix, cache_config, ratio, *, tail=False):
        super().__init__(
            131 if tail else 132,
            torch.float32 if tail else torch.uint8,
            prefix,
            cache_config,
        )
        self.ratio = ratio
        self.tail = tail

    def get_attn_backend(self):
        return IndexPoolTailBackend if self.tail else IndexPoolBackend

    def get_kv_cache_spec(self, vllm_config):
        ring = tail_ring_size(self.ratio, vllm_config.num_speculative_tokens)
        if self.tail:
            return CircularBufferSpec(
                block_size=ring,
                num_kv_heads=1,
                head_size=131,
                head_size_v=0,
                dtype=torch.float32,
                dcp_sharded=False,
            )
        return MLAAttentionSpec(
            block_size=lcm(
                self.cache_config.block_size or 64,
                self.ratio * POOL_PAGE_SIZE,
                ring,
            ),
            num_kv_heads=1,
            max_tp_shards=1,
            head_size=132,
            dtype=torch.uint8,
            tokens_per_state=self.ratio,
            cache_role=SparseCacheRole.INDEXER,
            dcp_sharded=True,
        )

    def bind_kv_cache(self, kv_cache):
        super().bind_kv_cache(kv_cache)
        if not self.tail and self.kv_cache.shape[1] != POOL_PAGE_SIZE:
            raise ValueError("IndexPool requires 64-state kernel pages")


@triton.jit
def _write_tails(
    Keys,
    Positions,
    Tail,
    Table,
    Starts,
    Lengths,
    RING: tl.constexpr,
    KEY_ROW: tl.constexpr,
    KEY_COL: tl.constexpr,
    POS_PLANE: tl.constexpr,
    POS_TOKEN: tl.constexpr,
    THREE: tl.constexpr,
    TAIL_BLOCK: tl.constexpr,
    TAIL_ROW: tl.constexpr,
    TABLE_ROW: tl.constexpr,
):
    request, offset = tl.program_id(0), tl.program_id(1)
    start = tl.load(Starts + request)
    end = tl.load(Starts + request + 1)
    length = tl.load(Lengths + request)
    context = length - (end - start)
    token = length - RING + offset
    live = (end > start) & (context >= 0) & (token >= context) & (token >= 0)
    source = start + token - context
    block = tl.load(Table + request * TABLE_ROW, live, 0).to(tl.int64)
    live = live & (block >= 0)
    channel = tl.arange(0, 256)
    key = tl.load(
        Keys + source * KEY_ROW + channel * KEY_COL,
        live & (channel < 128),
        0,
    ).to(tl.float32)
    if THREE:
        axis = tl.maximum(channel - 128, 0)
        position = tl.load(
            Positions + axis * POS_PLANE + source * POS_TOKEN,
            live & (channel >= 128) & (channel < 131),
            0,
        ).to(tl.float32)
    else:
        position = tl.load(
            Positions + source * POS_TOKEN,
            live,
            0,
        ).to(tl.float32)
    tl.store(
        Tail + block * TAIL_BLOCK + (token % RING) * TAIL_ROW + channel,
        tl.where(channel < 128, key, position),
        live & (channel < 131),
    )


def write_tails(keys, positions, tail, table, starts, lengths):
    if lengths.numel():
        _write_tails[(lengths.numel(), tail.shape[1])](
            keys,
            positions,
            tail,
            table,
            starts,
            lengths,
            tail.shape[1],
            keys.stride(0),
            keys.stride(1),
            positions.stride(0) if positions.ndim == 2 else 0,
            positions.stride(-1),
            positions.ndim == 2,
            tail.stride(0),
            tail.stride(1),
            table.stride(0),
        )


class FlashCompressedIndexer(nn.Module):
    def __init__(
        self,
        vllm_config,
        config,
        topk_indices_buffer,
        prefix,
        *,
        rope_parameters,
        quant_config,
    ):
        super().__init__()
        self.ratio = config.compressed_indexer_compress_ratio
        self.group_budget = config.compressed_indexer_group_topk - 1
        self.num_heads = config.compressed_indexer_num_attention_heads
        if not 1 <= self.num_heads <= 64:
            raise ValueError("IndexPool requires 1..64 scoring heads")
        self.rope_dim = config.qk_rope_head_dim
        self.rope_parameters = rope_parameters
        self.eps = config.rms_norm_eps
        self.topk_indices_buffer = topk_indices_buffer
        self.max_model_len = vllm_config.model_config.max_model_len
        self.logits_bytes = envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024
        parallel = vllm_config.parallel_config
        size = parallel.decode_context_parallel_size
        rank = get_dcp_group().rank_in_group if size > 1 else 0
        self.layout = Hy4ShardLayout(
            size, rank, parallel.cp_kv_cache_interleave_size, self.ratio
        )
        for name, input_size, output_size in (
            ("query_projection", config.q_lora_rank, self.num_heads * 128),
            ("head_weight_projection", config.hidden_size, self.num_heads),
            ("token_key_projection", config.hidden_size, 128),
        ):
            setattr(
                self,
                name,
                ReplicatedLinear(
                    input_size,
                    output_size,
                    bias=False,
                    quant_config=quant_config,
                    prefix=f"{prefix}.{name}",
                ),
            )
        self.token_key_norm = nn.LayerNorm(128, eps=self.eps)
        self.key_value_projection = nn.Linear(128, 128, bias=False, dtype=torch.float32)
        self.key_gate_projection = nn.Linear(128, 128, bias=False, dtype=torch.float32)
        self.key_norm = nn.RMSNorm(128, eps=self.eps, dtype=torch.float32)
        self.intra_group_ape = nn.Parameter(
            torch.empty(self.ratio, 128, dtype=torch.float32)
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

    def forward(self, hidden_states, q_c, positions, rope=None):
        output = self.topk_indices_buffer[: hidden_states.shape[0]]
        output.fill_(-1)
        context = get_forward_context()
        metadata = context.attn_metadata
        if metadata is None:
            return output
        if not isinstance(metadata, dict):
            raise TypeError("IndexPool requires per-layer metadata")
        km = metadata[self.k_cache.prefix]
        tm = metadata[self.tail_cache.prefix]
        if not isinstance(km, IndexPoolMetadata) or not isinstance(
            tm, IndexPoolMetadata
        ):
            raise TypeError("Incompatible IndexPool metadata")
        tokens = km.num_actual_tokens
        cache, tail = self.k_cache.kv_cache, self.tail_cache.kv_cache
        if tokens == 0 or km.seq_lens.numel() == 0:
            return output
        if cache.numel() == 0 or tail.numel() == 0:
            return output

        keys = self.token_key_norm(
            self.token_key_projection(hidden_states)[0]
        ).bfloat16()
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
            pooled = gated_pool(
                self.key_value_projection(raw.float()),
                self.key_gate_projection(raw.float()),
                self.intra_group_ape,
                self.key_norm.weight,
                self.eps,
            ).bfloat16()
            quant, scales = fwht_quantize(
                rotate_indexer(
                    pooled,
                    group_positions,
                    self.rope_dim,
                    self.rope_parameters,
                )
            )
            local, active = shard_pool_writes(groups, group_live, self.layout)
            slots = local_group_slots(km.block_table, group_owner, local, active)
            write_index_pool(cache, slots, quant, scales)

        write_tails(
            keys,
            positions,
            tail,
            tm.block_table,
            km.query_start_loc,
            km.seq_lens,
        )
        query = self.query_projection(q_c)[0][:tokens].reshape(
            tokens, self.num_heads, 128
        )
        query = rotate_indexer(
            query,
            positions[..., :tokens],
            self.rope_dim,
            self.rope_parameters,
        )
        quant_q, scales_q = fwht_quantize(query.reshape(-1, 128).bfloat16())
        quant_q = quant_q.reshape(tokens, self.num_heads, 128)
        weights = self.head_weight_projection(hidden_states)[0][:tokens].float()
        weights = (
            weights
            * (self.num_heads**-0.5 * 128**-0.5)
            * scales_q.reshape(tokens, self.num_heads)
        )

        max_len = (
            self.max_model_len
            if context.cudagraph_runtime_mode == CUDAGraphMode.FULL
            else km.max_seq_len
        )
        width = self.layout.local_scoring_width(max_len)
        if self.logits_bytes < width * 4:
            raise ValueError("IndexPool logits budget cannot hold one row")
        chunk = max(1, self.logits_bytes // (width * 4))
        for begin in range(0, tokens, chunk):
            end = min(tokens, begin + chunk)
            bounds = upper[begin:end]
            logits = paged_mqa_logits(
                quant_q[begin:end],
                cache,
                weights[begin:end],
                owner[begin:end],
                self.layout.num_local_groups(bounds),
                km.block_table,
                width,
            )
            recalled = sharded_select_groups(
                logits, bounds, self.layout, self.group_budget
            )
            output[begin:end].copy_(
                expand_groups(
                    recalled,
                    logical[begin:end],
                    self.ratio,
                    active=live[begin:end],
                )
            )
        return output


class FlashWindowIndexer(nn.Module):
    def __init__(self, window, topk_indices_buffer, attention_prefix):
        super().__init__()
        self.window = window
        self.attention_prefix = attention_prefix
        width = ((window + 127) // 128) * 128
        rows = topk_indices_buffer.shape[0]
        if topk_indices_buffer.shape[1] < width:
            raise ValueError("SWA top-k storage is too narrow")
        self.topk_indices_buffer = topk_indices_buffer.view(-1)[: rows * width].view(
            rows, width
        )

    def forward(self, hidden_states, q_c, positions, rope=None):
        output = self.topk_indices_buffer[: hidden_states.shape[0]]
        metadata = get_forward_context().attn_metadata
        if metadata is None:
            return output.fill_(-1)
        if not isinstance(metadata, dict):
            raise TypeError("SWA requires per-layer metadata")
        meta = metadata[self.attention_prefix]
        output.copy_(
            logical_window_indices(
                meta.query_start_loc,
                meta.seq_lens,
                output.shape[0],
                meta.num_actual_tokens,
                self.window,
                output.shape[1],
            )
        )
        return output
