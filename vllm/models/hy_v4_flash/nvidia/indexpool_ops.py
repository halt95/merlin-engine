# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
# Modified by the Merlin project; see NOTICE.
"""Coarse IndexPool mathematics for the sm_86 lane.

``rotate_indexer`` is #68296 ``hy_v4_flash/nvidia/indexpool_ops.py:37-73``
verbatim. The quantisation, pooling, selection and expansion maths are the
software-FP8 versions (``vllm.merlin.hy4_reference``, adapted from the same
#68296 lines 15-107), so no native FP8 cast or DeepGEMM call is made.

Scoring and the pooled-cache write use the Hy4 Triton kernels on CUDA. The torch
versions here (``paged_mqa_logits_torch``, ``write_index_pool_torch``) run the
same contract on CPU, for the CPU gate only.
"""

import torch

# expand_groups / gated_pool wrap the references
from vllm.merlin.hy4_reference import (  # noqa: F401  (re-exported)
    e4m3_decode,
    expand_groups as _expand_groups_torch,
    fwht_quantize,
    gated_pool as _gated_pool_torch,
    select_groups,
    window_indices,
)

INDEX_DIM = 128
POOL_ROW_BYTES = INDEX_DIM + 4


def _apply_interleaved_rope(planes, section):
    from vllm.model_executor.layers.rotary_embedding.mrope import (
        apply_interleaved_rope,
    )

    return apply_interleaved_rope(planes, section)


# Fast / experimental: the Triton glue
# (vllm/merlin/hy4_indexer_glue.py); otherwise the torch reference.
def expand_groups(group_ids, positions, ratio, active=None, *, fast=False, out=None):
    if ratio < 1:
        raise ValueError("ratio must be positive")
    if fast and group_ids.is_cuda and (group_ids.shape[1] + 1) * ratio <= 8192:
        from vllm.merlin.hy4_indexer_glue import expand_groups_fused

        return expand_groups_fused(group_ids, positions, ratio, active, out=out)
    result = _expand_groups_torch(group_ids, positions, ratio, active)
    if out is not None:
        out.copy_(result)
        return out
    return result


def gated_pool(values, gates, ape, norm_weight, eps, *, experimental=False):
    if experimental and values.is_cuda and values.shape[1] == 4:
        from vllm.merlin.hy4_indexer_glue import gated_pool_fused

        return gated_pool_fused(values, gates, ape, norm_weight, eps)
    return _gated_pool_torch(values, gates, ape, norm_weight, eps)


def rotate_indexer(x, positions, rope_dim, parameters, *, fast=False):
    if fast and x.is_cuda:
        from vllm.merlin.hy4_indexer_glue import rotate_indexer_fused

        return rotate_indexer_fused(x, positions, rope_dim, parameters)
    if rope_dim == 0:
        return x
    if parameters.get("rope_type", "default") != "default":
        raise ValueError("Flash IndexPool currently requires default RoPE")
    theta = parameters.get("rope_theta", 10000.0)
    inverse = theta ** (
        -torch.arange(0, rope_dim, 2, device=x.device, dtype=torch.float32) / rope_dim
    )
    if positions.ndim == 1:
        frequency = positions.float()[:, None] * inverse
    else:
        section = parameters.get("mrope_section")
        if positions.shape[0] != 3 or section is None or sum(section) * 2 != rope_dim:
            raise ValueError("IndexPool requires valid three-axis MRoPE sections")
        planes = positions.float()[:, :, None] * inverse
        if parameters.get("mrope_interleaved", True):
            frequency = _apply_interleaved_rope(planes, section)
        else:
            parts, offset = [], 0
            for axis, width in enumerate(section):
                parts.append(planes[axis, :, offset : offset + width])
                offset += width
            frequency = torch.cat(parts, -1)
    nope, pe = x.split((x.shape[-1] - rope_dim, rope_dim), dim=-1)
    pe = (
        pe.reshape(*pe.shape[:-1], rope_dim // 2, 2).transpose(-2, -1).reshape(pe.shape)
    )
    frequency = torch.cat((frequency, frequency), -1)
    if pe.ndim == 3:
        frequency = frequency[:, None]
    left, right = pe.float().chunk(2, dim=-1)
    rotated = (
        pe.float() * frequency.cos() + torch.cat((-right, left), -1) * frequency.sin()
    )
    return torch.cat((nope, rotated.to(x.dtype)), dim=-1)


def _page_bytes(cache):
    # [pages, P, 132] with page-planar content: P*128 key bytes, then P*4 scales.
    pages, page_size, width = cache.shape
    if cache.dtype != torch.uint8 or width != POOL_ROW_BYTES:
        raise ValueError("IndexPool cache must be uint8 [pages,P,132]")
    return cache.reshape(pages, page_size * width), page_size


def write_index_pool_torch(cache, slots, quant, scales):
    """CPU twin of ``hy4_kernels.write_index_pool``: physical group slots, -1 skips."""
    if not quant.shape[0]:
        return
    flat, page_size = _page_bytes(cache)
    capacity = cache.shape[0] * page_size
    live = (slots >= 0) & (slots < capacity)
    rows = live.nonzero().flatten()
    if not rows.numel():
        return
    slot = slots[rows].long()
    page, offset = slot // page_size, slot % page_size
    lanes = torch.arange(INDEX_DIM, device=cache.device)
    key_cols = offset[:, None] * INDEX_DIM + lanes
    flat[page[:, None], key_cols] = quant[rows]
    scale_cols = (
        page_size * INDEX_DIM
        + offset[:, None] * 4
        + torch.arange(4, device=cache.device)
    )
    flat[page[:, None], scale_cols] = (
        scales[rows].float().contiguous().view(torch.uint8).reshape(-1, 4)
    )
    if flat.data_ptr() != cache.data_ptr():  # reshape copied a padded cache
        cache.copy_(flat.view(cache.shape))


def paged_mqa_logits_torch(q, cache, weights, owners, upper, table, max_groups):
    """CPU twin of ``hy4_kernels.paged_mqa_logits`` (same masking, FP32 result)."""
    t, h, _ = q.shape
    if not t or not max_groups:
        return torch.empty((t, max_groups), dtype=torch.float32, device=q.device)
    flat, page_size = _page_bytes(cache)
    groups = torch.arange(max_groups, device=q.device)
    column, offset = groups // page_size, groups % page_size
    owner = owners.long()
    valid = (
        (groups[None] < upper.long()[:, None])
        & (column[None] < table.shape[1])
        & ((owner >= 0) & (owner < table.shape[0]))[:, None]
    )
    page = table[owner.clamp(0, table.shape[0] - 1)][
        :, column.clamp(max=table.shape[1] - 1)
    ].long()
    valid &= (page >= 0) & (page < cache.shape[0])
    page = page.clamp(0, cache.shape[0] - 1)
    lanes = torch.arange(INDEX_DIM, device=q.device)
    keys = flat[page[..., None], (offset[None, :, None] * INDEX_DIM + lanes)]
    scale_cols = (
        page_size * INDEX_DIM
        + offset[None, :, None] * 4
        + torch.arange(4, device=q.device)
    )
    key_scale = flat[page[..., None], scale_cols].contiguous().view(torch.float32)
    dot = torch.einsum("thd,tgd->thg", e4m3_decode(q), e4m3_decode(keys))
    score = (dot.clamp_min(0) * weights.float()[..., None]).sum(1)
    score = score * key_scale.squeeze(-1)
    return torch.where(valid, score, -torch.inf)


def write_index_pool(cache, slots, quant, scales):
    if cache.is_cuda:
        from vllm.merlin.hy4_kernels import write_index_pool as kernel

        return kernel(cache, slots, quant, scales)
    return write_index_pool_torch(cache, slots, quant, scales)


def paged_mqa_logits(q, cache, weights, owners, upper, table, max_groups, early_exit=False):
    if q.is_cuda:
        from vllm.merlin.hy4_kernels import paged_mqa_logits as kernel

        # early_exit: same scores, skipped tiles
        return kernel(q, cache, weights, owners, upper, table, max_groups, early_exit=early_exit)
    return paged_mqa_logits_torch(q, cache, weights, owners, upper, table, max_groups)
