# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Hy4 numerical contracts; imports only Torch.

IndexPool math is adapted from vLLM #68296 indexpool_ops.py:15-36,
77-107 and swa.py:50-56. FP8 encoding is software-only.
"""

import torch
import torch.nn.functional as F


def e4m3_decode(codes: torch.Tensor) -> torch.Tensor:
    """Decode all E4M3FN byte patterns, including signed zero and NaNs."""
    if codes.dtype != torch.uint8:
        raise ValueError("E4M3 codes must be uint8")
    bits = codes.to(torch.int32)
    exponent = (bits >> 3) & 15
    mantissa = bits & 7
    normal = (((exponent + 120) << 23) | (mantissa << 20)).view(torch.float32)
    value = torch.where(exponent == 0, mantissa.float() / 512.0, normal)
    value = torch.where((bits & 127) == 127, float("nan"), value)
    return torch.where((bits & 128) != 0, -value, value)


def e4m3_encode(values: torch.Tensor) -> torch.Tensor:
    """E4M3FN saturating encode, round-to-nearest-even; no FP8 cast."""
    x = values.float().contiguous()
    sign = torch.signbit(x).to(torch.int32) << 7
    magnitude = x.abs().clamp(max=448.0).contiguous()
    bits = magnitude.view(torch.int32)
    rounded = ((bits + 0x7FFFF + ((bits >> 20) & 1)) >> 20) - 960
    subnormal = torch.round(magnitude * 512.0).to(torch.int32)
    code = torch.where(magnitude < 2.0**-6, subnormal, rounded)
    code = code.clamp(0, 126)
    code = torch.where(torch.isnan(x), 127, code)
    return (code | sign).to(torch.uint8)


def pack_mla_reference(kv: torch.Tensor) -> torch.Tensor:
    """Pack [N,576] into the engine's uint8 [N,656] fp8_ds_mla format."""
    if kv.ndim != 2 or kv.shape[1] != 576:
        raise ValueError("packed MLA requires [N,576]")
    latent = kv[:, :512].float().reshape(-1, 4, 128)
    scales = torch.exp2(
        torch.ceil(torch.log2((latent.abs().amax(-1) / 448.0).clamp_min(1e-4)))
    )
    quant = e4m3_encode(latent / scales[..., None]).flatten(1)
    scale_bytes = scales.contiguous().view(torch.uint8)
    rope_bytes = kv[:, 512:].to(torch.bfloat16).contiguous().view(torch.uint8)
    return torch.cat((quant, scale_bytes, rope_bytes), dim=-1)


def unpack_mla_reference(cache: torch.Tensor) -> torch.Tensor:
    if cache.dtype != torch.uint8 or cache.ndim != 2 or cache.shape[1] != 656:
        raise ValueError("packed MLA requires uint8 [N,656]")
    scales = cache[:, 512:528].contiguous().view(torch.float32)
    latent = e4m3_decode(cache[:, :512]) * scales.repeat_interleave(128, -1)
    rope = cache[:, 528:].contiguous().view(torch.bfloat16).float()
    return torch.cat((latent, rope), dim=-1)


def sparse_mla_reference(
    q,
    kv,
    indices,
    scale,
    *,
    latent_dim=512,
    sinks=None,
    sink_owner=True,
    lengths=None,
):
    """FP32 reference. Selection contains physical, unique, causal KV rows."""
    if kv.dtype == torch.uint8:
        kv = unpack_mla_reference(kv)
    t, h, width = q.shape
    k = indices.shape[1]
    valid = (indices >= 0) & (indices < kv.shape[0])
    if lengths is not None:
        valid &= torch.arange(k, device=q.device)[None] < lengths[:, None]
    if kv.shape[0]:
        selected = kv[indices.long().clamp(0, kv.shape[0] - 1)].float()
        selected = torch.where(valid[..., None], selected, 0.0)
    else:
        selected = torch.zeros((t, k, width), device=q.device)
    logits = torch.einsum("thd,tkd->thk", q.float(), selected) * scale
    logits = logits.masked_fill(~valid[:, None], -torch.inf)
    denominator_logits = logits
    if sinks is not None and sink_owner:
        sink_column = sinks.float()[None, :, None].expand(t, h, 1)
        denominator_logits = torch.cat((logits, sink_column), dim=-1)
    lse = torch.logsumexp(denominator_logits, dim=-1)
    safe_lse = torch.where(torch.isfinite(lse), lse, 0.0)
    probabilities = torch.exp(logits - safe_lse[..., None])
    output = torch.einsum("thk,tkd->thd", probabilities, selected[..., :latent_dim])
    return output, lse


def merge_attention_reference(outputs, lses):
    """Merge normalized partials whose natural-log LSE includes owned sinks."""
    states = torch.stack(outputs).float()
    logs = torch.stack(lses).float()
    total = torch.logsumexp(logs, dim=0)
    safe = torch.where(torch.isfinite(total), total, 0.0)
    weights = torch.exp(logs - safe[None])
    return (states * weights[..., None]).sum(0), total


def gated_pool(values, gates, ape, norm_weight, eps):
    probabilities = (gates.float() + ape.float()).softmax(1)
    pooled = (values.float() * probabilities).sum(1)
    return F.rms_norm(pooled, (128,), norm_weight.float(), eps)


def fwht_quantize(vectors):
    """Normalized FWHT and power-of-two FP8 quantization; returns raw bytes."""
    if vectors.ndim != 2 or vectors.shape[1] != 128:
        raise ValueError("IndexPool quantization requires [rows,128]")
    x = vectors.float()
    stride = 1
    while stride < 128:
        pairs = x.reshape(-1, 2, stride)
        left, right = pairs[:, 0], pairs[:, 1]
        x = torch.stack((left + right, left - right), dim=1).reshape(vectors.shape)
        stride *= 2
    x = x * 128**-0.5
    scale = torch.exp2(torch.ceil(torch.log2(x.abs().amax(-1).clamp_min(1e-4) / 448.0)))
    return e4m3_encode(x / scale[:, None]), scale


def select_groups(logits, upper, budget):
    if budget < 0:
        raise ValueError("group budget must be nonnegative")
    columns = torch.arange(logits.shape[1], device=logits.device)
    scores = logits.masked_fill(columns[None] >= upper[:, None], -torch.inf)
    count = min(budget, logits.shape[1])
    values, indices = scores.topk(count, dim=-1)
    sentinel = torch.iinfo(torch.int32).max
    indices = torch.where(values != -torch.inf, indices, sentinel).int()
    indices = indices.sort(dim=-1).values
    indices.masked_fill_(indices == sentinel, -1)
    return F.pad(indices, (0, budget - count), value=-1)


def expand_groups(group_ids, positions, ratio, active=None):
    if ratio < 1:
        raise ValueError("ratio must be positive")
    offsets = torch.arange(ratio, device=group_ids.device)
    upper = positions // ratio
    live = positions >= 0
    if active is not None:
        live &= active
    history = group_ids[..., None] * ratio + offsets
    valid = (group_ids >= 0) & (group_ids < upper[:, None]) & live[:, None]
    history = torch.where(valid[..., None], history, -1)
    tail = upper[:, None] * ratio + offsets
    tail = torch.where(live[:, None] & (tail <= positions[:, None]), tail, -1)
    candidates = torch.cat((history.flatten(1), tail), -1).int()
    valid = candidates >= 0
    destinations = torch.where(
        valid,
        valid.long().cumsum(-1) - 1,
        valid.sum(-1, keepdim=True) + (~valid).long().cumsum(-1) - 1,
    )
    return torch.empty_like(candidates).scatter(-1, destinations, candidates)


def window_indices(positions, window, width):
    if window < 1 or width < window:
        raise ValueError("SWA width must cover the causal window")
    columns = torch.arange(width, device=positions.device)
    indices = positions[:, None] - window + 1 + columns
    live = (positions[:, None] >= 0) & (columns < window) & (indices >= 0)
    return torch.where(live, indices, -1).int()


def paged_mqa_logits_reference(q, cache, weights, owners, upper, table, max_groups):
    """Small-case oracle for page-planar pooled keys; intentionally unoptimized."""
    result = torch.full(
        (q.shape[0], max_groups), -torch.inf, dtype=torch.float32, device=q.device
    )
    page_size = cache.shape[1]
    query = e4m3_decode(q)
    for row in range(q.shape[0]):
        owner = int(owners[row])
        if not 0 <= owner < table.shape[0]:
            continue
        for group in range(min(max(0, int(upper[row])), max_groups)):
            column, offset = divmod(group, page_size)
            if column >= table.shape[1]:
                continue
            page = int(table[owner, column])
            if not 0 <= page < cache.shape[0]:
                continue
            data = cache[page].reshape(-1)
            key = e4m3_decode(data[offset * 128 : (offset + 1) * 128])
            key_scale = (
                data[page_size * 128 + offset * 4 : page_size * 128 + (offset + 1) * 4]
                .contiguous()
                .view(torch.float32)[0]
            )
            dot = (query[row] * key[None]).sum(-1)
            result[row, group] = (
                dot.clamp_min(0) * weights[row].float()
            ).sum() * key_scale
    return result
