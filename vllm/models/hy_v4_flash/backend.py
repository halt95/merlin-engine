# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Ampere target/draft attention adapter for the in-tree HYV4 model.

``build_attention_parts`` keeps the port's keyword-only signature. It returns:

- ``indexer``: #68296's ``FlashWindowIndexer`` on SWA layers, or its
  ``FlashCompressedIndexer`` (IndexPool) on sparse layers, ported onto the Hy4
  sm_86 kernels and the DCP group selection (``nvidia/compressed_indexer.py``);
  ``None`` on DFly2 draft layers (``config.dfly_external_topk``), whose top-k
  the speculator writes from the target's snapshots;
- ``mla_attn``: an ``MLAAttention`` bound to the Hy4 Triton sparse (or SWA)
  backend with the learnable sink, as #68296 ``nvidia/attention.py:162-186`` binds the
  FlashMLA one.
"""

from dataclasses import dataclass

import torch
from torch import nn

from vllm.models.hy_v4_flash.configs import SLIDING_ATTENTION


@dataclass(frozen=True)
class AttentionParts:
    indexer: nn.Module | None
    mla_attn: nn.Module


def _is_window_layer(config, layer_idx: int) -> bool:
    return config.layer_types[layer_idx] == SLIDING_ATTENTION


def _mla_layer_name(prefix: str) -> str:
    # The module path: HYV4FlashAttention registers the layer as ``mla_attn``.
    return f"{prefix}.mla_attn"


def build_indexer(
    *,
    vllm_config,
    config,
    layer_idx: int,
    topk_indices_buffer: torch.Tensor,
    prefix: str,
) -> nn.Module:
    """The layer's top-k producer (#68296 attention.py:153-161).

    Sparse layers own the checkpoint's ``compressed_indexer.*`` tensors (renamed
    ``indexer.*`` by the mapper); SWA layers have no parameters.
    """
    if _is_window_layer(config, layer_idx):
        from vllm.models.hy_v4_flash.nvidia.swa import FlashWindowIndexer

        return FlashWindowIndexer(
            config.sliding_window, topk_indices_buffer, _mla_layer_name(prefix)
        )
    from vllm.models.hy_v4_flash.nvidia.compressed_indexer import (
        FlashCompressedIndexer,
    )

    return FlashCompressedIndexer(
        vllm_config,
        config,
        topk_indices_buffer,
        f"{prefix}.indexer",
        rope_parameters=config.rope_for_layer(layer_idx),
    )


def build_mla_attention(
    *,
    vllm_config,
    config,
    layer_idx: int,
    indexer: nn.Module | None,
    topk_indices_buffer: torch.Tensor,
    prefix: str,
    kv_b_proj: nn.Module,
    sinks: torch.Tensor,
    quant_config=None,
) -> nn.Module:
    from vllm.distributed import get_tensor_model_parallel_world_size
    from vllm.merlin.hy4_backend import Hy4TritonSparseBackend, Hy4TritonSWABackend
    from vllm.model_executor.layers.attention.mla_attention import MLAAttention

    window = _is_window_layer(config, layer_idx)
    heads = config.swa_num_attention_heads if window else config.num_attention_heads
    nope = config.swa_qk_nope_head_dim if window else config.qk_nope_head_dim
    rope = config.swa_qk_rope_head_dim if window else config.qk_rope_head_dim
    value = config.swa_v_head_dim if window else config.v_head_dim
    if config.kv_lora_rank != 512 or rope != 64:
        raise ValueError("The Hy4 Ampere backend requires latent512 + rotary64")
    # Sparse MQA (dense MLA prefill cannot apply sinks) is
    # set at config time (the HYV4FlashForCausalLM verify_and_update_config
    # hook), so it is in the compile-cache key; setting it here, after hashing,
    # was not.
    if not vllm_config.attention_config.sparse_mla_force_mqa:
        raise ValueError(
            "Hy4 Flash needs attention_config.sparse_mla_force_mqa=True from its "
            "config hook (vllm/merlin/hy4flash_pp.py); it was not applied"
        )
    return MLAAttention(
        num_heads=heads // get_tensor_model_parallel_world_size(),
        scale=(nope + rope) ** -0.5,
        qk_nope_head_dim=nope,
        qk_rope_head_dim=rope,
        v_head_dim=value,
        q_lora_rank=config.q_lora_rank,
        kv_lora_rank=config.kv_lora_rank,
        kv_b_proj=kv_b_proj,
        cache_config=vllm_config.cache_config,
        quant_config=quant_config,
        prefix=_mla_layer_name(prefix),
        attn_backend=Hy4TritonSWABackend if window else Hy4TritonSparseBackend,
        use_sparse=True,
        indexer=indexer,
        topk_indices_buffer=topk_indices_buffer,
        sliding_window=config.sliding_window if window else None,
        sinks=sinks,
    )


def build_attention_parts(
    *,
    vllm_config,
    config,
    layer_idx: int,
    topk_indices_buffer: torch.Tensor,
    prefix: str,
    kv_b_proj: nn.Module,
    sinks: torch.Tensor,
    quant_config=None,
) -> AttentionParts:

    if vllm_config.cache_config.cache_dtype not in ("auto", "bfloat16", "fp8_ds_mla"):
        raise ValueError("Hy4 requires BF16 or explicit fp8_ds_mla KV")
    if topk_indices_buffer is None:
        raise ValueError("Hy4 requires logical top-k storage")
    if getattr(config, "dfly_external_topk", False):
        # Draft layers read the speculator's indices.
        indexer = None
    else:
        indexer = build_indexer(
            vllm_config=vllm_config,
            config=config,
            layer_idx=layer_idx,
            topk_indices_buffer=topk_indices_buffer,
            prefix=prefix,
        )
        topk_indices_buffer = indexer.topk_indices_buffer
    mla_attn = build_mla_attention(
        vllm_config=vllm_config,
        config=config,
        layer_idx=layer_idx,
        indexer=indexer,
        topk_indices_buffer=topk_indices_buffer,
        prefix=prefix,
        kv_b_proj=kv_b_proj,
        sinks=sinks,
        quant_config=quant_config,
    )
    return AttentionParts(indexer=indexer, mla_attn=mla_attn)
