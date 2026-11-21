# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modified by the Merlin project; see NOTICE.
"""Adapted from #68296 nvidia/attention.py:1-231."""

from copy import deepcopy

import torch
from torch import nn

from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.distributed import get_tensor_model_parallel_world_size

from vllm.merlin.hy4_kvshard import make_layer_sink
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.models.hy_v4_flash.backend import build_attention_parts
from vllm.models.hy_v4_flash.configs import SLIDING_ATTENTION

_RUNTIME_QUANT = object()


class HYV4FlashAttention(nn.Module):
    def __init__(
        self,
        vllm_config,
        config,
        layer_idx,
        topk_indices_buffer,
        prefix,
        *,
        quant_config=_RUNTIME_QUANT,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config
        self.is_window_layer = config.layer_types[layer_idx] == SLIDING_ATTENTION
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.num_heads = (
            config.swa_num_attention_heads
            if self.is_window_layer
            else config.num_attention_heads
        )
        self.num_local_heads = self.num_heads // get_tensor_model_parallel_world_size()
        self.qk_nope_head_dim = (
            config.swa_qk_nope_head_dim
            if self.is_window_layer
            else config.qk_nope_head_dim
        )
        self.qk_rope_head_dim = (
            config.swa_qk_rope_head_dim
            if self.is_window_layer
            else config.qk_rope_head_dim
        )
        self.v_head_dim = (
            config.swa_v_head_dim if self.is_window_layer else config.v_head_dim
        )
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.scaling = self.qk_head_dim**-0.5
        if (
            not config.learnable_sink
            or not config.gated_mla
            or config.gating_type != "elementwise"
        ):
            raise ValueError("Flash requires sinks and elementwise gated MLA")
        if any(
            getattr(config, name, None) is not None
            for name in ("gate_lora_rank", "o_lora_rank")
        ):
            raise ValueError("Flash requires direct gate/output projections")
        if (
            getattr(config, "gate_groups", 1) != 1
            or getattr(config, "o_groups", 1) != 1
            or getattr(config, "full_nope", False)
            or not getattr(config, "qk_norm", True)
        ):
            raise ValueError("Flash requires normalized ungrouped RoPE MLA")
        quant = (
            vllm_config.quant_config if quant_config is _RUNTIME_QUANT else quant_config
        )
        self.q_a_proj = ReplicatedLinear(
            config.hidden_size,
            self.q_lora_rank,
            bias=False,
            quant_config=quant,
            prefix=f"{prefix}.q_a_proj",
        )
        self.kv_a_proj_with_mqa = ReplicatedLinear(
            config.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=False,
            quant_config=quant,
            prefix=f"{prefix}.kv_a_proj_with_mqa",
        )
        self.q_a_layernorm = RMSNorm(self.q_lora_rank, config.rms_norm_eps)
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, config.rms_norm_eps)
        self.q_b_proj = ColumnParallelLinear(
            self.q_lora_rank,
            self.num_heads * self.qk_head_dim,
            bias=False,
            quant_config=quant,
            prefix=f"{prefix}.q_b_proj",
        )
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=quant,
            prefix=f"{prefix}.kv_b_proj",
        )
        self.linear_gate = ColumnParallelLinear(
            config.hidden_size,
            self.num_heads * self.v_head_dim,
            bias=False,
            quant_config=quant,
            prefix=f"{prefix}.linear_gate",
        )
        self.o_proj = RowParallelLinear(
            self.num_heads * self.v_head_dim,
            config.hidden_size,
            bias=False,
            quant_config=quant,
            prefix=f"{prefix}.o_proj",
        )
        rope = (
            config.rope_for_layer(layer_idx)
            if hasattr(config, "rope_for_layer")
            else deepcopy(config.rope_parameters)
        )
        # Text positions are identical along all MRoPE axes.
        rope.pop("mrope_section", None)
        rope.pop("mrope_interleaved", None)
        self.rotary_emb = get_rope(
            self.qk_rope_head_dim,
            max_position=vllm_config.model_config.max_model_len,
            rope_parameters=rope,
            is_neox_style=False,
        )
        self.learnable_sink_param = make_layer_sink(
            self.num_local_heads, dcp_sharded_kv=not self.is_window_layer
        )
        # A DFly2 draft layer reads top-k indices the
        # speculator writes (#68296 hy_v4/nvidia/attention.py:317-336): the
        # factory then returns indexer=None. A target index layer copies its
        # top-k into topk_snapshot after indexing (:802-805).
        self.external_topk = bool(getattr(config, "dfly_external_topk", False))
        self.topk_indices_buffer = topk_indices_buffer
        self.topk_snapshot: torch.Tensor | None = None
        self.topk_snapshot_width = 0  # (K + 1)
        parts = build_attention_parts(
            vllm_config=vllm_config,
            config=config,
            layer_idx=layer_idx,
            topk_indices_buffer=topk_indices_buffer,
            prefix=prefix,
            kv_b_proj=self.kv_b_proj,
            sinks=self.learnable_sink_param,
            quant_config=quant,
        )
        if self.external_topk and parts.indexer is not None:
            raise ValueError("A DFly draft layer must not build its own indexer")
        self.indexer = parts.indexer
        self.mla_attn = parts.mla_attn
        self.post_proj_norm = (
            RMSNorm(config.hidden_size, config.rms_norm_eps)
            if config.sandwich_norm
            and (self.is_window_layer or not config.dsa_skip_attn_post_norm)
            else None
        )

    @property
    def layer_name(self):
        return self.mla_attn.layer_name

    def forward(self, positions, hidden_states, skip_post_norm=False):
        # skip_post_norm: the fused iHC post applies it
        if positions.ndim != 1:
            raise ValueError(
                "The Hy4 Flash model currently accepts text positions only"
            )
        q_c = self.q_a_layernorm(self.q_a_proj(hidden_states)[0])
        q = self.q_b_proj(q_c)[0].reshape(-1, self.num_local_heads, self.qk_head_dim)
        kv = self.kv_a_proj_with_mqa(hidden_states)[0]
        kv_c, k_pe = kv.split((self.kv_lora_rank, self.qk_rope_head_dim), -1)
        kv_c = self.kv_a_layernorm(kv_c)
        k_pe = k_pe.unsqueeze(1)
        q[..., self.qk_nope_head_dim :], k_pe = self.rotary_emb(
            positions, q[..., self.qk_nope_head_dim :], k_pe
        )
        out = torch.empty(
            (hidden_states.shape[0], self.num_local_heads * self.v_head_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        self._indexer_and_attn(hidden_states, q_c, positions, q, kv_c, k_pe, out)
        out = out * self.linear_gate(hidden_states)[0].sigmoid()
        out = self.o_proj(out)[0]
        if skip_post_norm:
            return out
        return self.post_proj_norm(out) if self.post_proj_norm is not None else out

    @eager_break_during_capture
    def _indexer_and_attn(self, hidden, q_c, positions, q, kv_c, k_pe, out):
        if self.indexer is not None:
            self.indexer(hidden, q_c, positions, self.rotary_emb)
            if self.topk_snapshot is not None:
                # Each request's last K + 1 rows
                from vllm.models.hy_v4.nvidia.dfly_snapshot import (
                    copy_from_forward_context,
                )

                copy_from_forward_context(
                    self.topk_snapshot,
                    self.topk_indices_buffer,
                    self.mla_attn.layer_name,
                    self.topk_snapshot_width,
                )
        impl = getattr(self.mla_attn, "impl", None)
        ready = getattr(impl, "record_logical_topk_ready", None)
        if ready is not None:
            ready()
        out.copy_(self.mla_attn(q, kv_c, k_pe, output_shape=out.shape))
