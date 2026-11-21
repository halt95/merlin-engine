# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Graph-safe IndexPool expansion, RoPE and experimental gated pooling.

Expansion preserves the reference's stable partition after its int32 cast.
RoPE keeps Torch's inverse frequencies and separate FP32 multiply/add steps.
Pooling is qualified separately: ATen reduction trees vary by Torch build.
"""

import torch

from vllm.merlin.hy4_kernels import _cuda_tensors, _integer_vector, _require
from vllm.triton_utils import tl, triton
from vllm.triton_utils import tldevice as libdevice


@triton.jit
def _expand(G, P, ACTIVE, OUT, G0, G1, P0, A0, O0,
            K: tl.constexpr, R: tl.constexpr, HAS_ACTIVE: tl.constexpr,
            BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    column = tl.arange(0, BLOCK)
    width: tl.constexpr = (K + 1) * R
    pos = tl.load(P + row * P0).to(tl.int64)
    live = pos >= 0
    if HAS_ACTIVE:
        live = live & tl.load(ACTIVE + row * A0)
    upper = pos // R
    group = tl.load(G + row * G0 + (column // R) * G1,
                    column < K * R, -1).to(tl.int64)
    history_live = live & (group >= 0) & (group < upper)
    history = tl.where(history_live, group * R + column % R, -1)
    tail = upper * R + column % R
    tail = tl.where(live & (tail <= pos), tail, -1)
    candidate = tl.where(column < K * R, history, tail).to(tl.int32)
    # The reference tests validity AFTER narrowing to int32.
    valid = (column < width) & (candidate >= 0)
    prefix = tl.cumsum(valid.to(tl.int32), 0)
    count = tl.sum(valid.to(tl.int32), 0)
    # Invalid destinations continue after the valid prefix, in input order.
    destination = tl.where(valid, prefix - 1, count + column - prefix)
    tl.store(OUT + row * O0 + destination, candidate, column < width)


def expand_groups_fused(group_ids, positions, ratio, active=None, *, out=None):
    _require(ratio >= 1, "ratio must be positive")
    _require(group_ids.ndim == 2 and group_ids.dtype in (torch.int32, torch.int64),
             "group_ids must be int32/int64 [T,K]")
    rows, budget = group_ids.shape
    _integer_vector(positions, rows, "positions")
    _cuda_tensors(group_ids, positions)
    if active is not None:
        _require(active.shape == positions.shape and active.dtype == torch.bool,
                 "active must be bool [T]")
        _cuda_tensors(group_ids, active)
    width = (budget + 1) * ratio
    _require(width <= 8192, "fused expansion supports at most 8192 columns")
    if out is None:
        out = torch.empty((rows, width), dtype=torch.int32, device=group_ids.device)
    _require(out.shape == (rows, width) and out.dtype == torch.int32
             and out.stride(1) == 1 and out.stride(0) >= width,
             "out must be nonoverlapping int32 [T,(K+1)*ratio]")
    _cuda_tensors(group_ids, out)
    if rows:
        _expand[(rows,)](
            group_ids, positions, active if active is not None else positions, out,
            *group_ids.stride(), positions.stride(0),
            active.stride(0) if active is not None else 0, out.stride(0),
            budget, ratio, active is not None, triton.next_power_of_2(width),
            num_warps=4,
        )
    return out


@triton.jit
def _rotate(X, P, INV, OUT, X0, X1, X2, P0, P1,
            H: tl.constexpr, R: tl.constexpr, MROPE: tl.constexpr,
            INTERLEAVED: tl.constexpr, ST: tl.constexpr,
            SH: tl.constexpr, SW: tl.constexpr):
    token = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1).to(tl.int64)
    d = tl.arange(0, 128)
    nope: tl.constexpr = 128 - R
    half: tl.constexpr = R // 2
    rotary = d >= nope
    r = tl.maximum(d - nope, 0)
    k = r % half
    second = r >= half
    axis = tl.full((128,), 0, tl.int32)
    if MROPE:
        if INTERLEAVED:
            axis = tl.where((k % 3 == 1) & (k < SH * 3), 1, axis)
            axis = tl.where((k % 3 == 2) & (k < SW * 3), 2, axis)
        else:
            axis = tl.where(k < ST, 0, tl.where(k < ST + SH, 1, 2))
    pos = tl.load(P + axis * P0 + token * P1).to(tl.float32)
    inverse = tl.load(INV + k)
    frequency = pos * inverse
    cosine = libdevice.cos(frequency)
    sine = libdevice.sin(frequency)
    base = X + token * X0 + head * X1
    original = tl.load(base + d * X2)
    a = tl.load(base + (nope + 2 * k + second.to(tl.int32)) * X2)
    b = tl.load(base + (nope + 2 * k + 1 - second.to(tl.int32)) * X2)
    partner = tl.where(second, b.to(tl.float32), -b.to(tl.float32))
    # enable_fp_fusion=False keeps both products rounded before the addition.
    rotated = a.to(tl.float32) * cosine + partner * sine
    result = tl.where(rotary, rotated.to(original.dtype), original)
    tl.store(OUT + (token * H + head) * 128 + d, result)


def rotate_indexer_fused(x, positions, rope_dim, parameters):
    if rope_dim == 0:
        return x
    _require(parameters.get("rope_type", "default") == "default",
             "Flash IndexPool currently requires default RoPE")
    _require(x.ndim in (2, 3) and x.shape[-1] == 128
             and x.dtype in (torch.float16, torch.bfloat16, torch.float32),
             "IndexPool RoPE requires FP16/BF16/FP32 [T,128] or [T,H,128]")
    _require(0 < rope_dim <= 128 and rope_dim % 2 == 0, "invalid rope_dim")
    _require(positions.dtype in (torch.int32, torch.int64),
             "positions must be int32/int64")
    _cuda_tensors(x, positions)
    rows = x.shape[0]
    mrope = positions.ndim != 1
    section = parameters.get("mrope_section")
    if mrope:
        _require(positions.ndim == 2 and positions.shape == (3, rows)
                 and section is not None and len(section) == 3
                 and all(s >= 0 for s in section) and sum(section) * 2 == rope_dim,
                 "IndexPool requires valid three-axis MRoPE sections")
        p0, p1 = positions.stride()
    else:
        _integer_vector(positions, rows, "positions")
        p0, p1 = 0, positions.stride(0)
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    if not x.numel():
        return out
    # Keep the exact Torch expression, including its FP32 division and pow.
    theta = parameters.get("rope_theta", 10000.0)
    inverse = theta ** (
        -torch.arange(0, rope_dim, 2, device=x.device, dtype=torch.float32) / rope_dim
    )
    heads = x.shape[1] if x.ndim == 3 else 1
    _rotate[(rows, heads)](
        x, positions, inverse, out, x.stride(0),
        x.stride(1) if x.ndim == 3 else 0, x.stride(-1), p0, p1,
        heads, rope_dim, mrope, parameters.get("mrope_interleaved", True),
        *(section if mrope else (0, 0, 0)),
        num_warps=4, enable_fp_fusion=False,
    )
    return out


@triton.jit
def _pool(V, G, APE, W, OUT, V0, V1, V2, G0, G1, G2, A0, A1, W0,
          EPS: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    d = tl.arange(0, 128)
    v = V + row * V0 + d * V2
    g = G + row * G0 + d * G2
    a = APE + d * A1
    l0 = tl.load(g).to(tl.float32) + tl.load(a).to(tl.float32)
    l1 = tl.load(g + G1).to(tl.float32) + tl.load(a + A0).to(tl.float32)
    l2 = tl.load(g + 2 * G1).to(tl.float32) + tl.load(a + 2 * A0).to(tl.float32)
    l3 = tl.load(g + 3 * G1).to(tl.float32) + tl.load(a + 3 * A0).to(tl.float32)
    maximum = tl.full((128,), -3.4028234663852886e38, tl.float32)
    maximum = tl.where(maximum < l0, l0, maximum)
    maximum = tl.where(maximum < l1, l1, maximum)
    maximum = tl.where(maximum < l2, l2, maximum)
    maximum = tl.where(maximum < l3, l3, maximum)
    e0 = libdevice.exp(l0 - maximum)
    e1 = libdevice.exp(l1 - maximum)
    e2 = libdevice.exp(l2 - maximum)
    e3 = libdevice.exp(l3 - maximum)
    denominator = ((e0 + e1) + e2) + e3
    p0 = tl.math.div_rn(e0, denominator)
    p1 = tl.math.div_rn(e1, denominator)
    p2 = tl.math.div_rn(e2, denominator)
    p3 = tl.math.div_rn(e3, denominator)
    z0 = tl.load(v).to(tl.float32) * p0
    z1 = tl.load(v + V1).to(tl.float32) * p1
    z2 = tl.load(v + 2 * V1).to(tl.float32) * p2
    z3 = tl.load(v + 3 * V1).to(tl.float32) * p3
    pooled = ((z0 + z1) + z2) + z3
    mean_square = tl.sum(pooled * pooled, 0) * (1.0 / 128.0)
    normalized = pooled * libdevice.rsqrt(mean_square + EPS)
    result = normalized * tl.load(W + d * W0).to(tl.float32)
    tl.store(OUT + row * 128 + d, result)


def gated_pool_fused(values, gates, ape, norm_weight, eps):
    """Experimental ratio-four pool; strict bitwise qualification is required."""
    _require(values.ndim == 3 and values.shape[1:] == (4, 128)
             and gates.shape == values.shape and ape.shape == (4, 128)
             and norm_weight.shape == (128,), "invalid gated-pool geometry")
    tensors = (values, gates, ape, norm_weight)
    _cuda_tensors(*tensors)
    _require(all(x.dtype in (torch.float16, torch.bfloat16, torch.float32)
                 for x in tensors), "gated pool requires FP16/BF16/FP32")
    out = torch.empty((values.shape[0], 128), dtype=torch.float32, device=values.device)
    if values.shape[0]:
        _pool[(values.shape[0],)](
            *tensors, out, *values.stride(), *gates.stride(), *ape.stride(),
            norm_weight.stride(0), eps, num_warps=4, enable_fp_fusion=False,
        )
    return out
