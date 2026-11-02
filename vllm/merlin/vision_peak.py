# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Opt-in lower-peak eager path for the Qwen3 vision tower.

``VLLM_MERLIN_VISION_PEAK=1`` runs the upstream modules in the upstream order
with the same inputs, GEMM shapes and kernels, so the output bits do not
change. Only tensor lifetimes change:

* the interpolated position embedding is released once it has been added;
* every block's norm output, contiguous QK copy, QKV result and rotated QK are
  released as soon as their last reader returns, not at function exit;
* with ``VLLM_MERLIN_VISION_ROTARY_INPLACE=1`` the rotary embedding rotates the
  Q and K parts of the QKV result in place (the same flash-attn Triton kernel on
  strided views, elementwise, so the same bits), and attention reads Q/K/V as
  views of it: no contiguous QK copy and no rotated QK buffer at all;
* the two residual adds of each block, the position add and the GELU in the
  block MLP and patch mergers write in place into tensors this path owns, so
  the block input is never kept next to a second residual stream;
* without DeepStack the merger output is returned as is instead of being
  copied by a single-input ``torch.cat``.

At ~32K patch rows on two-way tensor parallelism this removes the per-block
attention-projection overlap (norm, QKV, QK copy, rotated QK, projection and
its all-reduce output alive together) that sets the tower's peak.

Only eager calls that carry a grid take this path. Compiled vision blocks,
metadata-only encoder CUDA-graph calls, gradient mode and towers whose blocks
are not the upstream Qwen3 block/attention/MLP classes keep the plain forward.
Patch mergers of another class are called as modules.
"""

import logging
import os

import einops
import torch
from torch import nn

logger = logging.getLogger(__name__)

# Process-local count of tower calls that took this path.
low_peak_vision_calls = 0
# Blocks whose Q/K were rotated in place in the QKV buffer
# (VLLM_MERLIN_VISION_ROTARY_INPLACE).
inplace_rotary_blocks = 0


def _is_upstream(module: nn.Module, name: str, filename: str) -> bool:
    """True if ``module`` is the named upstream class running its own forward."""
    code = getattr(type(module).forward, "__code__", None)
    return (
        type(module).__name__ == name
        and "forward" not in vars(module)
        and code is not None
        and os.path.basename(code.co_filename) == filename
    )


def _eager(module: nn.Module) -> bool:
    return getattr(module, "do_not_compile", True)


def supported(tower: nn.Module) -> bool:
    """Every block (attention, MLP) is the upstream eager Qwen3 one."""
    return _eager(tower.patch_embed) and all(
        _eager(blk)
        and _is_upstream(blk, "Qwen3_VisionBlock", "qwen3_vl.py")
        and _is_upstream(blk.attn, "Qwen2_5_VisionAttention", "qwen2_5_vl.py")
        and _is_upstream(blk.mlp, "Qwen3_VisionMLP", "qwen3_vl.py")
        for blk in tower.blocks
    )


def _add_(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """``x + y``, written into ``x`` when that cannot change dtype or shape."""
    if y.dtype == x.dtype and y.shape == x.shape:
        return x.add_(y)
    return x + y


def _act_(act_fn: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Native GELU in place (same kernel and approximation); others as is."""
    if type(act_fn) is nn.GELU:
        return torch.ops.aten.gelu_(x, approximate=act_fn.approximate)
    return act_fn(x)


def _rotary_in_place(attn: nn.Module):
    """The in-place rotary kernel when it computes exactly what ``attn`` would.

    Only the CUDA dispatch of ApplyRotaryEmb (NeoX style, no fp32 upcast) calls the
    flash-attn Triton rotary; any other dispatch keeps the copying path. Only the
    flash-attn ViT backend reads strided Q/K without copying them, so other backends
    keep it too.
    """
    from vllm import envs

    op = attn.apply_rotary_emb
    if (
        not envs.VLLM_MERLIN_VISION_ROTARY_INPLACE
        or type(op).__name__ != "ApplyRotaryEmb"
        or not op.is_neox_style
        or op.enable_fp32_compute
        or getattr(op, "_forward_method", None) != op.forward_cuda
        or not getattr(getattr(attn, "attn", None), "is_flash_attn_backend", False)
    ):
        return None
    from vllm.vllm_flash_attn.layers.rotary import apply_rotary_emb

    return apply_rotary_emb


def block_forward(blk: nn.Module, x: torch.Tensor, metadata: dict) -> torch.Tensor:
    """Qwen3_VisionBlock + Qwen2_5_VisionAttention + Qwen3_VisionMLP forward.

    ``x`` must be owned by the caller's tower pass; it is updated in place.
    """
    global inplace_rotary_blocks
    attn = blk.attn
    heads = attn.num_attention_heads_per_partition
    head_dim = attn.hidden_size_per_attention_head

    h = blk.norm1(x)
    qkv, _ = attn.qkv(h)
    del h
    seq_len, batch_size, _ = qkv.shape
    qkv = einops.rearrange(
        qkv,
        "s b (three head head_dim) -> b s three head head_dim",
        three=3,
        head=heads,
    )
    cos = metadata["rotary_pos_emb_cos"]
    sin = metadata["rotary_pos_emb_sin"]
    rotary_ = _rotary_in_place(attn) if cos is not None and sin is not None else None
    if rotary_ is not None:
        inplace_rotary_blocks += 1
        if inplace_rotary_blocks == 1:
            logger.info("Merlin vision peak: Q/K rotary in place in the QKV buffer")
        # Q and K are rotated where the QKV GEMM wrote them; v views the same buffer.
        for part in (0, 1):
            rotary_(qkv[:, :, part], cos, sin, interleaved=False, inplace=True)
        q, k, v = qkv.unbind(dim=2)
    elif cos is not None and sin is not None:
        qk, v = qkv[:, :, :2], qkv[:, :, 2]
        qk = einops.rearrange(
            qk, "b s two head head_dim -> (two b) s head head_dim", two=2
        )
        qk = qk.contiguous()
        qk_rotated = attn.apply_rotary_emb(qk, cos, sin)
        del qk
        q, k = qk_rotated.view(2, batch_size, seq_len, heads, head_dim).unbind(dim=0)
        del qk_rotated
    else:
        q, k, v = qkv.unbind(dim=2)
    del qkv  # its views (v, and q/k when rotated in place) live until attention returns

    context = attn.attn(
        query=q,
        key=k,
        value=v,
        cu_seqlens=metadata["cu_seqlens"],
        max_seqlen=metadata["max_seqlen"],
        sequence_lengths=metadata.get("sequence_lengths"),
    )
    del q, k, v
    context = einops.rearrange(
        context, "b s h d -> s b (h d)", b=batch_size
    ).contiguous()
    out, _ = attn.proj(context)
    del context
    x = _add_(x, out)
    del out

    mlp = blk.mlp
    h = blk.norm2(x)
    hidden = mlp.linear_fc1(h)
    del h
    hidden = _act_(mlp.act_fn, hidden)
    out = mlp.linear_fc2(hidden)
    del hidden
    return _add_(x, out)


def merger_forward(merger: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Qwen3_VisionPatchMerger forward without the second expanded buffer."""
    if not _is_upstream(merger, "Qwen3_VisionPatchMerger", "qwen3_vl.py"):
        return merger(x)
    if merger.use_postshuffle_norm:
        h = merger.norm(x.view(-1, merger.hidden_size))
    else:
        h = merger.norm(x).view(-1, merger.hidden_size)
    hidden, _ = merger.linear_fc1(h)
    del h
    hidden = _act_(merger.act_fn, hidden)
    out, _ = merger.linear_fc2(hidden)
    return out


def maybe_low_peak_vision(
    tower: nn.Module,
    x: torch.Tensor,
    grid_thw: torch.Tensor | list[list[int]] | None,
    encoder_metadata: dict | None,
) -> torch.Tensor | None:
    """Return None to execute the untouched upstream forward."""
    from vllm import envs

    global low_peak_vision_calls
    if (
        not envs.VLLM_MERLIN_VISION_PEAK
        or grid_thw is None
        or encoder_metadata is not None
        or torch.is_grad_enabled()
        or torch.compiler.is_compiling()
        or not supported(tower)
    ):
        return None
    low_peak_vision_calls += 1
    if low_peak_vision_calls == 1:
        logger.info(
            "Merlin vision peak path active: patch_rows=%d blocks=%d",
            x.shape[0],
            len(tower.blocks),
        )

    hidden = tower.input_norm(x.to(device=tower.device, non_blocking=True), tower.dtype)
    hidden = tower.patch_embed(hidden)
    grid = grid_thw if isinstance(grid_thw, list) else grid_thw.tolist()
    metadata = tower.prepare_encoder_metadata(grid)
    hidden = _add_(hidden, metadata.pop("pos_embeds"))
    hidden = hidden.unsqueeze(1)

    features = []
    for layer_num, blk in enumerate(tower.blocks):
        hidden = block_forward(blk, hidden, metadata)
        if layer_num in tower.deepstack_visual_indexes:
            idx = tower.deepstack_visual_indexes.index(layer_num)
            features.append(merger_forward(tower.deepstack_merger_list[idx], hidden))
    out = merger_forward(tower.merger, hidden)
    del hidden
    if not features:
        return out.contiguous()
    return torch.cat([out, *features], dim=1)
