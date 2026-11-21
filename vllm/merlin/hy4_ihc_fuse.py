# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
# The stage1 / stage2 / post code is adapted from vllm/models/hy_v4/nvidia/triton_ihc.py,
# itself adapted from the SGLang HY V4 implementation (sgl-project/sglang#36805).
"""Fused iHC boundaries for the Hy4 Flash decoder layer (``MERLIN_HY4_FUSE_IHC=1``).

Stock, one iHC decoder layer runs 11 small kernels around its two sub-blocks
(profiled on layer 3)::

    attn in   _ihc_pre_stage1, _ihc_pre_stage2, rms_norm (input_layernorm)
    attn out  rms_norm (post_proj_norm), _ihc_post
    mlp in    _ihc_pre_stage1, _ihc_pre_stage2, rms_norm (post_attention_layernorm)
    mlp out   rms_norm (combine_output_norm / post_fc2_norm), _ihc_post

With the knob on the layer runs 6 (5 fewer)::

    attn in   _ihc_pre_stage1 (stock), K1 = stage2 + rms_norm
    attn out  K2S = rms_norm + _ihc_post + the mlp _ihc_pre_stage1
    mlp in    K1 = stage2 + rms_norm
    mlp out   K2 = rms_norm + _ihc_post

Bit-identical to the stock kernels by construction: every value
is computed with the same operations in the same order and rounded at the
same points.

* The stage1/stage2/post parts are the stock Triton code (triton_ihc.py), with
  the same reduction shapes, ``num_warps`` and ``enable_fp_fusion=False``. A
  stage1 split of 1024 elements is exactly the (channel, 1024-block) tile a
  K2S program writes, so its sums see the same values in the same layout.
* The rms_norm part replays csrc/layernorm_kernels.cu ``rms_norm_kernel`` as
  launched for a [T, H] BF16/FP16 row (VEC 8; min(H/8, 1024) threads when
  T < 256, else min(H/8, 256): a prefill chunk of 256+ tokens runs 256):
  - each thread's ``variance += x * x`` over its vectors t, t + threads, ...
    (8 contiguous elements each) is an FMA chain (nvcc contracts it) from 0.0,
    in that order;
  - cub's warp reduction (shuffle-down tree, offsets 1, 2, ..., 16) adds
    contiguous pairs, then pairs of pairs, inside each 32-lane warp;
  - thread 0 adds the warp totals in warp order;
  - ``rsqrtf(variance / H + eps)`` (IEEE division, approximate rsqrt), then
    ``(x * s) * w`` rounded once to the input dtype.
  Each step is written out explicitly (no tl.sum over more than one live
  value), so the result does not depend on Triton's own reduction layout.

The GPU test (tests/merlin/test_hy4_ihc_fuse_gpu.py, development tree) checks
every output bitwise against the stock kernels; the layer falls back to stock for
anything the fused kernels do not cover (another hc_mult, H not a multiple of 1024, an
HPC iHC op, batch-invariant mode, a non-CUDA tensor).
"""

from __future__ import annotations

import os

import torch

from vllm.triton_utils import tl, triton

ENV = "MERLIN_HY4_FUSE_IHC"
BLOCK = 1024  # the stock iHC _BLOCK_K and _BLOCK_D
VEC = 8  # rms_norm_kernel's vector width for 2-byte dtypes
_STATE = {"on": "unset"}


def fuse_requested(environ=None) -> bool:
    """1 = fused iHC boundaries; unset, blank or 0 = stock."""
    cache = environ is None
    if cache and _STATE["on"] != "unset":
        return _STATE["on"]
    raw = ((os.environ if environ is None else environ).get(ENV, "") or "").strip()
    if raw not in ("", "0", "1"):
        raise ValueError(f"{ENV} must be 0 or 1 (got {raw!r})")
    on = raw == "1"
    if cache:
        _STATE["on"] = on
    return on


def shapes_supported(hc_mult: int, hidden: int) -> bool:
    """The shapes the fused kernels replay exactly."""
    threads = hidden // VEC
    return (
        hc_mult == 4
        and hidden % BLOCK == 0
        and threads % 32 == 0
        and threads <= 1024
    )


# ------------------------------------------------------------------ device code
@triton.jit
def _warp_tree_inv(var, H: tl.constexpr, LANES: tl.constexpr, EPS: tl.constexpr,
                   THREADS: tl.constexpr):
    """From the per-thread sums (``var``, one per rms_norm thread, padding lanes
    0): cub's warp tree (contiguous pairs, five levels), the real warps' totals
    in warp order, then rsqrtf(total / H + EPS). Every add has one live operand
    pair, so Triton's reduction layout cannot reorder it."""
    WARPS: tl.constexpr = THREADS // 32
    WARPS_P2: tl.constexpr = LANES // 32
    w = tl.sum(tl.reshape(var, (WARPS_P2, 16, 2)), axis=2)
    w = tl.sum(tl.reshape(w, (WARPS_P2, 8, 2)), axis=2)
    w = tl.sum(tl.reshape(w, (WARPS_P2, 4, 2)), axis=2)
    w = tl.sum(tl.reshape(w, (WARPS_P2, 2, 2)), axis=2)
    w = tl.reshape(tl.sum(tl.reshape(w, (WARPS_P2, 1, 2)), axis=2), (WARPS_P2,))
    widx = tl.arange(0, WARPS_P2)
    total = tl.sum(tl.where(widx == 0, w, 0.0), axis=0)
    for k in tl.static_range(1, WARPS):
        total = total + tl.sum(tl.where(widx == k, w, 0.0), axis=0)
    return tl.rsqrt(tl.math.div_rn(total, H * 1.0) + EPS)


@triton.jit
def _row_inv(row_ptr, H: tl.constexpr, LANES: tl.constexpr, EPS: tl.constexpr,
             THREADS: tl.constexpr):
    """rms_norm_kernel's 1/rms of one stored row: thread t's
    variance = fma(x_j, x_j, variance) over its vectors v = t, t + THREADS, ...
    and their elements v * 8 + j, j = 0..7, in that order.

    THREADS is the stock launch's block size,
    min(H / 8, 1024 if num_tokens < 256 else 256) (csrc/layernorm_kernels.cu):
    a 256-token or larger call (a prefill chunk) runs 256 threads, two vectors
    each for the first 128, and so a different FMA chain and warp tree."""
    t = tl.arange(0, LANES)
    ROUNDS: tl.constexpr = (H // 8 + THREADS - 1) // THREADS
    var = tl.zeros((LANES,), dtype=tl.float32)
    for r in tl.static_range(ROUNDS):
        v = t + r * THREADS
        live = (t < THREADS) & (v < H // 8)
        for j in tl.static_range(8):
            x = tl.load(row_ptr + v * 8 + j, mask=live, other=0.0).to(tl.float32)
            var = tl.fma(x, x, var)
    return _warp_tree_inv(var, H, LANES, EPS, THREADS)


@triton.jit
def _stage2_row_inv(x_row, pre, H: tl.constexpr, LANES: tl.constexpr,
                    EPS: tl.constexpr, HC_MULT: tl.constexpr, HC_POW2: tl.constexpr,
                    DTYPE: tl.constexpr, THREADS: tl.constexpr):
    """The same 1/rms of the row stage2 would store: each element is the stock
    stage2 sum over the channels, rounded to DTYPE, then squared into the
    thread's FMA chain (vectors t, t + THREADS, ...; see _row_inv)."""
    t = tl.arange(0, LANES)
    ROUNDS: tl.constexpr = (H // 8 + THREADS - 1) // THREADS
    hc_idx = tl.arange(0, HC_POW2)
    var = tl.zeros((LANES,), dtype=tl.float32)
    for r in tl.static_range(ROUNDS):
        v = t + r * THREADS
        live = (t < THREADS) & (v < H // 8)
        for j in tl.static_range(8):
            acc = tl.zeros((LANES,), dtype=tl.float32)
            for c in tl.static_range(HC_MULT):
                channel = tl.load(x_row + c * H + v * 8 + j, mask=live, other=0.0)
                gate = tl.sum(tl.where(hc_idx == c, pre, 0.0), axis=0)
                acc += gate * channel.to(tl.float32)
            x = acc.to(DTYPE).to(tl.float32)
            var = tl.fma(x, x, var)
    return _warp_tree_inv(var, H, LANES, EPS, THREADS)


@triton.jit
def _stage2_gates(partial_row, scale_ptr, base_ptr, K_TOTAL: tl.constexpr,
                  HC_MULT: tl.constexpr, HC_POW2: tl.constexpr,
                  NUM_SPLITS: tl.constexpr, PARTIAL_STRIDE: tl.constexpr,
                  MAGNITUDE: tl.constexpr, NORM_EPS: tl.constexpr,
                  HC_EPS: tl.constexpr):
    """The stock _ihc_pre_stage2 gate math, verbatim."""
    hc_idx = tl.arange(0, HC_POW2)
    hc_mask = hc_idx < HC_MULT
    sum_squares = tl.zeros((), dtype=tl.float32)
    pre_mix = tl.zeros((HC_POW2,), dtype=tl.float32)
    post_mix = tl.zeros((HC_POW2,), dtype=tl.float32)
    for split_idx in tl.static_range(NUM_SPLITS):
        partial = partial_row + split_idx * PARTIAL_STRIDE
        sum_squares += tl.load(partial)
        pre_mix += tl.load(partial + 1 + hc_idx, mask=hc_mask, other=0.0)
        post_mix += tl.load(partial + 1 + HC_POW2 + hc_idx, mask=hc_mask, other=0.0)
    reciprocal_rms = tl.rsqrt(sum_squares / K_TOTAL + NORM_EPS)
    pre_scale = tl.load(scale_ptr)
    post_scale = tl.load(scale_ptr + 1)
    pre_base = tl.load(base_ptr + hc_idx, mask=hc_mask, other=0.0)
    post_base = tl.load(base_ptr + HC_MULT + hc_idx, mask=hc_mask, other=0.0)
    pre = tl.sigmoid(pre_mix * reciprocal_rms * pre_scale + pre_base) + HC_EPS
    post = (
        MAGNITUDE * tl.sigmoid(post_mix * reciprocal_rms * post_scale + post_base)
        + HC_EPS
    )
    return pre, post


@triton.jit
def _pre2_norm_kernel(x_ptr, partial_ptr, scale_ptr, base_ptr, norm_w_ptr, out_ptr,
                      post_ptr,
                      H: tl.constexpr, HC_MULT: tl.constexpr, HC_POW2: tl.constexpr,
                      NUM_SPLITS: tl.constexpr, PARTIAL_STRIDE: tl.constexpr,
                      MAGNITUDE: tl.constexpr, NORM_EPS: tl.constexpr,
                      HC_EPS: tl.constexpr, LN_EPS: tl.constexpr,
                      LANES: tl.constexpr, THREADS: tl.constexpr):
    """K1: stage2 (reduce the channels with the pre gates, round) + rms_norm."""
    token = tl.program_id(0).to(tl.int64)
    block = tl.program_id(1)
    K_TOTAL: tl.constexpr = HC_MULT * H
    hc_idx = tl.arange(0, HC_POW2)
    hc_mask = hc_idx < HC_MULT
    pre, post = _stage2_gates(partial_ptr + token * NUM_SPLITS * PARTIAL_STRIDE,
                              scale_ptr, base_ptr, K_TOTAL, HC_MULT, HC_POW2,
                              NUM_SPLITS, PARTIAL_STRIDE, MAGNITUDE, NORM_EPS, HC_EPS)
    if block == 0:
        tl.store(post_ptr + token * HC_MULT + hc_idx, post, mask=hc_mask)
    x_row = x_ptr + token * K_TOTAL
    inv = _stage2_row_inv(x_row, pre, H, LANES, LN_EPS, HC_MULT, HC_POW2,
                          out_ptr.dtype.element_ty, THREADS)
    # this program's 1024 elements: the same per-element math
    offs = block * 1024 + tl.arange(0, 1024)
    acc = tl.zeros((1024,), dtype=tl.float32)
    for c in tl.static_range(HC_MULT):
        channel = tl.load(x_row + c * H + offs).to(tl.float32)
        gate = tl.sum(tl.where(hc_idx == c, pre, 0.0), axis=0)
        acc += gate * channel
    y = acc.to(out_ptr.dtype.element_ty).to(tl.float32)
    w = tl.load(norm_w_ptr + offs).to(tl.float32)
    tl.store(out_ptr + token * H + offs, (y * inv * w).to(out_ptr.dtype.element_ty))


@triton.jit
def _norm_post_kernel(x_ptr, norm_w_ptr, residual_ptr, post_ptr, out_ptr,
                      weight_ptr, partial_ptr,
                      H: tl.constexpr, HC_MULT: tl.constexpr, HC_POW2: tl.constexpr,
                      HAS_NORM: tl.constexpr, LN_EPS: tl.constexpr,
                      STAGE1: tl.constexpr, NUM_SPLITS: tl.constexpr,
                      PARTIAL_STRIDE: tl.constexpr, LANES: tl.constexpr,
                      THREADS: tl.constexpr):
    """K2 / K2S, one program per (token, channel, 1024-block) tile: rms_norm
    (optional) + _ihc_post of the tile, and with STAGE1 the next pre's stage1
    partial of the same tile (one stage1 split)."""
    token = tl.program_id(0).to(tl.int64)
    tile = tl.program_id(1)
    K_TOTAL: tl.constexpr = HC_MULT * H
    BLOCKS: tl.constexpr = H // 1024
    c = tile // BLOCKS
    block = tile % BLOCKS
    hc_idx = tl.arange(0, HC_POW2)
    hc_mask = hc_idx < HC_MULT
    offs = block * 1024 + tl.arange(0, 1024)
    x = tl.load(x_ptr + token * H + offs).to(tl.float32)
    if HAS_NORM:
        inv = _row_inv(x_ptr + token * H, H, LANES, LN_EPS, THREADS)
        w = tl.load(norm_w_ptr + offs).to(tl.float32)
        x = (x * inv * w).to(out_ptr.dtype.element_ty).to(tl.float32)
    post = tl.load(post_ptr + token * HC_MULT + hc_idx, mask=hc_mask, other=0.0)
    out_row = out_ptr + token * K_TOTAL
    residual = tl.load(residual_ptr + token * K_TOTAL + c * H + offs).to(tl.float32)
    gate = tl.sum(tl.where(hc_idx == c, post, 0.0), axis=0)
    value = (gate * x + residual).to(out_ptr.dtype.element_ty)
    tl.store(out_row + c * H + offs, value)
    if STAGE1:
        # the stock _ihc_pre_stage1 of split (c, block), on the tile just
        # written, reloaded with stage1's own load (same layout, same sums)
        tl.debug_barrier()
        split = c * BLOCKS + block
        k_offsets = split * 1024 + tl.arange(0, 1024)
        k_mask = k_offsets < K_TOTAL
        xs = tl.load(out_row + k_offsets, mask=k_mask, other=0.0).to(tl.float32)
        sum_squares = tl.sum(xs * xs, axis=0)
        weight_offsets = hc_idx[:, None] * K_TOTAL + k_offsets[None, :]
        weight_mask = hc_mask[:, None] & k_mask[None, :]
        pre_mix = tl.sum(
            tl.load(weight_ptr + weight_offsets, mask=weight_mask, other=0.0)
            * xs[None, :],
            axis=1,
        )
        post_mix = tl.sum(
            tl.load(weight_ptr + HC_MULT * K_TOTAL + weight_offsets,
                    mask=weight_mask, other=0.0)
            * xs[None, :],
            axis=1,
        )
        partial = partial_ptr + (token * NUM_SPLITS + split) * PARTIAL_STRIDE
        tl.store(partial, sum_squares)
        tl.store(partial + 1 + hc_idx, pre_mix, mask=hc_mask)
        tl.store(partial + 1 + HC_POW2 + hc_idx, post_mix, mask=hc_mask)


# ------------------------------------------------------------------- host side
def rms_threads(tokens: int, hidden: int) -> int:
    """the stock rms_norm launch's block size for a
    call of ``tokens`` rows (csrc/layernorm_kernels.cu rms_norm, not batch
    invariant): min(hidden / VEC, 1024 if tokens < 256 else 256)."""
    return min(hidden // VEC, 1024 if tokens < 256 else 256)


def _partials(tokens: int, hidden: int, hc: int, device):
    splits = (hc * hidden) // BLOCK
    stride = 1 + 2 * triton.next_power_of_2(hc)
    buf = torch.empty((tokens, splits, stride), dtype=torch.float32, device=device)
    return buf, splits, stride


def stage1(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """The stock stage1 launch into a partial buffer (the layer's first pre)."""
    from vllm.models.hy_v4.nvidia.triton_ihc import _ihc_pre_stage1

    tokens, hc, hidden = x.shape
    partial, splits, stride = _partials(tokens, hidden, hc, x.device)
    if tokens:
        _ihc_pre_stage1[(tokens, splits)](
            x, weight, partial, K_TOTAL=hc * hidden, HC_MULT=hc,
            HC_POW2=triton.next_power_of_2(hc), NUM_SPLITS=splits, BLOCK_K=BLOCK,
            PARTIAL_STRIDE=stride, num_warps=8, enable_fp_fusion=False,
        )
    return partial


def pre2_norm(x: torch.Tensor, partial: torch.Tensor, pre_layer, norm):
    """K1: (rms_norm(stage2(x)), post gates), bit-identical to the stock pair."""
    tokens, hc, hidden = x.shape
    out = torch.empty((tokens, hidden), dtype=x.dtype, device=x.device)
    post = torch.empty((tokens, hc), dtype=torch.float32, device=x.device)
    if not tokens:
        return out, post
    hc_pow2 = triton.next_power_of_2(hc)
    blocks = hidden // BLOCK
    threads = rms_threads(tokens, hidden)
    _pre2_norm_kernel[(tokens, blocks)](
        x, partial, pre_layer.hc_scale, pre_layer.hc_base, norm.weight, out, post,
        H=hidden, HC_MULT=hc, HC_POW2=hc_pow2, NUM_SPLITS=partial.shape[1],
        PARTIAL_STRIDE=partial.shape[2], MAGNITUDE=pre_layer.magnitude,
        NORM_EPS=pre_layer.layernorm_epsilon, HC_EPS=pre_layer.hc_eps,
        LN_EPS=norm.variance_epsilon, LANES=triton.next_power_of_2(threads),
        THREADS=threads, num_warps=8, enable_fp_fusion=False,
    )
    return out, post


def norm_post(x: torch.Tensor, norm, residual: torch.Tensor, post: torch.Tensor,
              next_weight: torch.Tensor | None = None):
    """K2 (next_weight None): ihc_post(rms_norm(x)). K2S: also the next pre's
    stage1 partials. ``norm`` None = no norm (a layer without it)."""
    tokens, hc, hidden = residual.shape
    out = torch.empty_like(residual)
    stage1_on = next_weight is not None
    rows = tokens if stage1_on else 0
    partial, splits, stride = _partials(rows, hidden, hc, x.device)
    threads = rms_threads(tokens, hidden)
    if tokens:
        _norm_post_kernel[(tokens, hc * (hidden // BLOCK))](
            x, norm.weight if norm is not None else x, residual, post, out,
            next_weight if stage1_on else post, partial if stage1_on else post,
            H=hidden, HC_MULT=hc, HC_POW2=triton.next_power_of_2(hc),
            HAS_NORM=norm is not None,
            LN_EPS=norm.variance_epsilon if norm is not None else 0.0,
            STAGE1=stage1_on, NUM_SPLITS=splits, PARTIAL_STRIDE=stride,
            LANES=triton.next_power_of_2(threads), THREADS=threads,
            num_warps=8, enable_fp_fusion=False,
        )
    return (out, partial) if stage1_on else out


# ------------------------------------------------------------------ the layer
def _rms_ok(norm, x: torch.Tensor) -> bool:
    return (
        norm is None
        or (getattr(norm, "variance_size_override", None) is None
            and getattr(norm, "has_weight", True)
            and norm.weight.dtype == x.dtype
            and norm.weight.shape == (x.shape[-1],))
    )


def layer_supported(layer, x3: torch.Tensor) -> bool:
    """Every condition the fused layer needs; else the stock forward runs."""
    from vllm import envs as vllm_envs

    a, m = layer.hc_attn_layer, layer.hc_mlp_layer
    if not (getattr(layer, "enable_ihc", False) and a.enable_ihc and m.enable_ihc):
        return False
    if vllm_envs.VLLM_BATCH_INVARIANT or not x3.is_cuda or x3.dtype not in (
            torch.bfloat16, torch.float16):
        return False
    if any(p.hc_pre.hpc_op is not None or p.hc_post.hpc_op is not None for p in (a, m)):
        return False
    if not shapes_supported(x3.shape[1], x3.shape[2]):
        return False
    norms = (layer.input_layernorm, layer.post_attention_layernorm,
             attn_out_norm(layer), mlp_out_norm(layer))
    if norms[0] is None or norms[1] is None:
        return False
    return all(_rms_ok(n, x3) for n in norms)


def attn_out_norm(layer):
    return getattr(layer.self_attn, "post_proj_norm", None)


def mlp_out_norm(layer):
    mlp = layer.mlp
    norm = getattr(mlp, "combine_output_norm", None)
    return norm if norm is not None else getattr(mlp, "post_fc2_norm", None)


def forward_layer(layer, positions: torch.Tensor, hidden_states: torch.Tensor):
    """HYV4DecoderLayer._forward_ihc with the fused boundaries. The sub-blocks
    run with their own output norms off (the fused kernels apply them)."""
    a, m = layer.hc_attn_layer, layer.hc_mlp_layer
    x3 = a.prepare_input(hidden_states).contiguous()
    partial = stage1(x3, a.hc_pre.hc_fn.weight)
    xn, post_a = pre2_norm(x3, partial, a.hc_pre, layer.input_layernorm)
    attn = layer.self_attn(positions=positions, hidden_states=xn, skip_post_norm=True)
    res1, partial = norm_post(attn.contiguous(), attn_out_norm(layer), x3, post_a,
                              next_weight=m.hc_pre.hc_fn.weight)
    xn, post_m = pre2_norm(res1, partial, m.hc_pre, layer.post_attention_layernorm)
    out = layer.mlp(xn, skip_norm=True)
    res2 = norm_post(out.contiguous(), mlp_out_norm(layer), res1, post_m)
    return res2, None
