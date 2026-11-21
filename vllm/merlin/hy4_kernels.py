# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Hy4 SM80+ kernels: layouts and provenance are given per kernel below.

The tiled attention/MQA structure adapts vllm-project/vllm#68633. Sink semantics
follow sgl-project/sglang#51159 (the semantics only, as a reference),
with sink-inclusive natural-log LSE for shard composition.
All FP8 accesses use bytes and software conversion.
"""

import math
from typing import NamedTuple

import torch

from vllm.merlin.hy4_reference import expand_groups, select_groups
from vllm.triton_utils import tl, triton
from vllm.triton_utils import tldevice as libdevice

LOG2E = tl.constexpr(1.4426950408889634)  # constexpr: read inside @triton.jit
LN2 = tl.constexpr(0.6931471805599453)  # constexpr: read inside @triton.jit


@triton.jit
def _decode_e4m3(raw):
    bits = raw.to(tl.int32)
    exponent = (bits >> 3) & 15
    mantissa = bits & 7
    normal = (((exponent + 120) << 23) | (mantissa << 20)).to(tl.float32, bitcast=True)
    value = tl.where(exponent == 0, mantissa.to(tl.float32) / 512.0, normal)
    value = tl.where((bits & 127) == 127, float("nan"), value)
    return tl.where((bits & 128) != 0, -value, value)


@triton.jit
def _latent_chunk(
    rows,
    valid,
    c,
    G: tl.constexpr,
    C: tl.constexpr,
    PACKED: tl.constexpr,
    dtype: tl.constexpr,
):
    """Latent columns [G*C, G*C+C) of BLOCK_N KV rows as ``dtype``.

    Packed rows hold one FP32 scale per 128 latent columns and C == 128, so a
    chunk needs one scale per row: a [BLOCK_N] vector, never a broadcast
    [BLOCK_N, D] FP32 tile (that tile is what Triton staged in shared memory).
    """
    if PACKED:
        raw = tl.load(rows[:, None] + G * C + c[None], valid[:, None], 0)
        scale = tl.load((rows + 512).to(tl.pointer_type(tl.float32)) + G, valid, 0.0)
        return (_decode_e4m3(raw) * scale[:, None]).to(dtype)
    return tl.load(rows[:, None] + G * C + c[None], valid[:, None], 0.0).to(dtype)


@triton.jit(do_not_specialize=["Q0", "Q1", "K0", "I0", "N", "K", "H"])
def _sparse_partial(
    Q,
    KV,
    IDX,
    LENGTHS,
    PART,
    LOGS,
    Q0,
    Q1,
    K0,
    I0,
    N,
    K,
    H,
    D: tl.constexpr,
    R: tl.constexpr,
    SPLITS: tl.constexpr,
    SCALE: tl.constexpr,
    PACKED: tl.constexpr,
    HAS_LENGTHS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # The latent dimension is processed in NG chunks of C <= 128 columns (the
    # split-dim layout follows sgl-project/sglang#51159): every dot operand is at most
    # [BLOCK_N, 128], which keeps D=512 inside sm_86's 99 KiB per block.
    C: tl.constexpr = D if D < 128 else 128
    NG: tl.constexpr = D // C
    row = tl.program_id(0).to(tl.int64)
    heads = tl.program_id(1) * 16 + tl.arange(0, 16)
    split = tl.program_id(2)
    c = tl.arange(0, C)
    live = heads[:, None] < H
    q_row = Q + row * Q0 + heads[:, None] * Q1
    q0 = tl.load(q_row + c[None], live, 0.0)
    if NG >= 2:
        q1 = tl.load(q_row + C + c[None], live, 0.0)
    if NG >= 4:
        q2 = tl.load(q_row + 2 * C + c[None], live, 0.0)
        q3 = tl.load(q_row + 3 * C + c[None], live, 0.0)
    if R:
        r = tl.arange(0, R)
        qr = tl.load(q_row + D + r[None], live, 0.0)
    maximum = tl.full((16,), float("-inf"), tl.float32)
    denominator = tl.zeros((16,), tl.float32)
    acc0 = tl.zeros((16, C), tl.float32)
    if NG >= 2:
        acc1 = tl.zeros((16, C), tl.float32)
    if NG >= 4:
        acc2 = tl.zeros((16, C), tl.float32)
        acc3 = tl.zeros((16, C), tl.float32)
    span = tl.cdiv(K, SPLITS)
    begin = split * span
    end = tl.minimum(begin + span, K)
    usable = K
    if HAS_LENGTHS:
        usable = tl.minimum(tl.maximum(tl.load(LENGTHS + row), 0), K)
    for start in range(begin, end, BLOCK_N):
        column = start + tl.arange(0, BLOCK_N)
        index = tl.load(IDX + row * I0 + column, column < end, -1).to(tl.int64)
        valid = (column < end) & (column < usable) & (index >= 0) & (index < N)
        rows = KV + index * K0
        k0 = _latent_chunk(rows, valid, c, 0, C, PACKED, q0.dtype)
        score = tl.dot(q0, tl.trans(k0), out_dtype=tl.float32)
        if NG >= 2:
            k1 = _latent_chunk(rows, valid, c, 1, C, PACKED, q0.dtype)
            score = tl.dot(q1, tl.trans(k1), score, out_dtype=tl.float32)
        if NG >= 4:
            k2 = _latent_chunk(rows, valid, c, 2, C, PACKED, q0.dtype)
            score = tl.dot(q2, tl.trans(k2), score, out_dtype=tl.float32)
            k3 = _latent_chunk(rows, valid, c, 3, C, PACKED, q0.dtype)
            score = tl.dot(q3, tl.trans(k3), score, out_dtype=tl.float32)
        if R:
            if PACKED:
                rope_ptr = (rows + 528).to(tl.pointer_type(tl.bfloat16))
                kr = tl.load(rope_ptr[:, None] + r[None], valid[:, None], 0.0)
            else:
                kr = tl.load(rows[:, None] + D + r[None], valid[:, None], 0.0)
            score = tl.dot(qr, tl.trans(kr.to(qr.dtype)), score, out_dtype=tl.float32)
        score = tl.where(valid[None], score * SCALE * LOG2E, float("-inf"))
        next_max = tl.maximum(maximum, tl.max(score, 1))
        safe_max = tl.where(next_max == float("-inf"), 0.0, next_max)
        alpha = tl.exp2(maximum - safe_max)
        probability = tl.exp2(score - safe_max[:, None])
        weights = probability.to(q0.dtype)
        acc0 = tl.dot(weights, k0, acc0 * alpha[:, None], out_dtype=tl.float32)
        if NG >= 2:
            acc1 = tl.dot(weights, k1, acc1 * alpha[:, None], out_dtype=tl.float32)
        if NG >= 4:
            acc2 = tl.dot(weights, k2, acc2 * alpha[:, None], out_dtype=tl.float32)
            acc3 = tl.dot(weights, k3, acc3 * alpha[:, None], out_dtype=tl.float32)
        denominator = denominator * alpha + tl.sum(probability, 1)
        maximum = next_max
    safe_denominator = tl.where(denominator > 0, denominator, 1.0)
    logsum = tl.where(
        denominator > 0,
        maximum + tl.log2(safe_denominator),
        float("-inf"),
    )
    state = (row * H + heads.to(tl.int64)) * SPLITS + split
    out = PART + state[:, None] * D + c[None]
    tl.store(out, acc0 / safe_denominator[:, None], live)
    if NG >= 2:
        tl.store(out + C, acc1 / safe_denominator[:, None], live)
    if NG >= 4:
        tl.store(out + 2 * C, acc2 / safe_denominator[:, None], live)
        tl.store(out + 3 * C, acc3 / safe_denominator[:, None], live)
    tl.store(LOGS + state, logsum, heads < H)


@triton.jit(do_not_specialize=["H"])
def _sparse_finish(
    PART,
    LOGS,
    SINK,
    OUT,
    LSE,
    H,
    D: tl.constexpr,
    SPLITS: tl.constexpr,
    HAS_SINK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1)
    state = row * H + head
    splits = tl.arange(0, SPLITS)
    d = tl.arange(0, D)
    logs = tl.load(LOGS + state * SPLITS + splits)
    sink = float("-inf")
    if HAS_SINK:
        sink = tl.load(SINK + head) * LOG2E
    maximum = tl.maximum(tl.max(logs, 0), sink)
    shift = tl.where(maximum == float("-inf"), 0.0, maximum)
    weights = tl.exp2(logs - shift)
    sink_weight = tl.exp2(sink - shift)
    denominator = tl.sum(weights, 0) + sink_weight
    partial = tl.load(PART + (state * SPLITS + splits[:, None]) * D + d[None])
    numerator = tl.sum(partial * weights[:, None], 0)
    safe_denominator = tl.where(denominator > 0, denominator, 1.0)
    tl.store(OUT + state * D + d, numerator / safe_denominator)
    total = tl.where(
        denominator > 0,
        (shift + tl.log2(safe_denominator)) * LN2,
        float("-inf"),
    )
    tl.store(LSE + state, total)


@triton.jit(do_not_specialize=["C0", "Q0", "PAGE", "CAPACITY"])
def _write_pool(
    CACHE,
    SLOTS,
    QUANT,
    SCALES,
    C0,
    Q0,
    PAGE,
    CAPACITY,
):
    row = tl.program_id(0).to(tl.int64)
    slot = tl.load(SLOTS + row).to(tl.int64)
    if (slot >= 0) & (slot < CAPACITY):
        page = slot // PAGE
        offset = slot % PAGE
        d = tl.arange(0, 128)
        raw = tl.load(QUANT + row * Q0 + d)
        tl.store(CACHE + page * C0 + offset * 128 + d, raw)
        scale_ptr = (CACHE + page * C0 + PAGE * 128 + offset * 4).to(
            tl.pointer_type(tl.float32)
        )
        tl.store(scale_ptr, tl.load(SCALES + row))


@triton.jit
def _paged_mqa_tile(
    Q,
    CACHE,
    WEIGHTS,
    OWNERS,
    UPPER,
    TABLE,
    OUT,
    Q0,
    Q1,
    W0,
    C0,
    T0,
    H,
    BH: tl.constexpr,
    PAGE,
    PAGES,
    REQUESTS,
    COLUMNS,
    WIDTH,
):
    # The stock _paged_mqa body, shared by the
    # stock kernel and the tile-skipping one (same math, same bits)
    row = tl.program_id(0).to(tl.int64)
    group = tl.program_id(1) * 32 + tl.arange(0, 32)
    owner = tl.load(OWNERS + row).to(tl.int64)
    upper = tl.load(UPPER + row)
    column = group // PAGE
    valid = (
        (group < WIDTH)
        & (group < upper)
        & (column < COLUMNS)
        & (owner >= 0)
        & (owner < REQUESTS)
    )
    page = tl.load(TABLE + owner * T0 + column, valid, -1).to(tl.int64)
    valid = valid & (page >= 0) & (page < PAGES)
    offset = group % PAGE
    d = tl.arange(0, 128)
    heads = tl.arange(0, BH)
    query = _decode_e4m3(
        tl.load(
            Q + row * Q0 + heads[:, None] * Q1 + d[None],
            heads[:, None] < H,
            0,
        )
    ).to(tl.bfloat16)
    key = _decode_e4m3(
        tl.load(
            CACHE + page[:, None] * C0 + offset[:, None] * 128 + d[None],
            valid[:, None],
            0,
        )
    ).to(tl.bfloat16)
    scale_ptr = (CACHE + page * C0 + PAGE * 128 + offset * 4).to(
        tl.pointer_type(tl.float32)
    )
    scale = tl.load(scale_ptr, valid, 0.0)
    weights = tl.load(WEIGHTS + row * W0 + heads, heads < H, 0.0)
    dot = tl.dot(query, tl.trans(key), out_dtype=tl.float32)
    score = tl.sum(tl.maximum(dot, 0.0) * weights[:, None], 0) * scale
    score = tl.where(valid, score, float("-inf"))
    tl.store(OUT + row * WIDTH + group, score, group < WIDTH)


@triton.jit(
    do_not_specialize=[
        "Q0",
        "Q1",
        "W0",
        "C0",
        "T0",
        "H",
        "PAGE",
        "PAGES",
        "REQUESTS",
        "COLUMNS",
        "WIDTH",
    ]
)
def _paged_mqa(
    Q,
    CACHE,
    WEIGHTS,
    OWNERS,
    UPPER,
    TABLE,
    OUT,
    Q0,
    Q1,
    W0,
    C0,
    T0,
    H,
    BH: tl.constexpr,
    PAGE,
    PAGES,
    REQUESTS,
    COLUMNS,
    WIDTH,
):
    _paged_mqa_tile(Q, CACHE, WEIGHTS, OWNERS, UPPER, TABLE, OUT, Q0, Q1, W0, C0, T0, H, BH, PAGE, PAGES, REQUESTS, COLUMNS, WIDTH)


@triton.jit(
    do_not_specialize=[
        "Q0",
        "Q1",
        "W0",
        "C0",
        "T0",
        "H",
        "PAGE",
        "PAGES",
        "REQUESTS",
        "COLUMNS",
        "WIDTH",
    ]
)
def _paged_mqa_skip(
    Q,
    CACHE,
    WEIGHTS,
    OWNERS,
    UPPER,
    TABLE,
    OUT,
    Q0,
    Q1,
    W0,
    C0,
    T0,
    H,
    BH: tl.constexpr,
    PAGE,
    PAGES,
    REQUESTS,
    COLUMNS,
    WIDTH,
):
    # A tile whose first group is at or past the
    # row's bound, or whose owner row is invalid, scores -inf in the stock kernel
    # too (every column fails its valid mask): write that and skip the query
    # decode, key loads and dot. FULL graphs score max_model_len columns, so at
    # short contexts almost every tile is skipped.
    row = tl.program_id(0).to(tl.int64)
    first = tl.program_id(1) * 32
    owner = tl.load(OWNERS + row).to(tl.int64)
    upper = tl.load(UPPER + row)
    if (first >= upper) | (owner < 0) | (owner >= REQUESTS) | (first // PAGE >= COLUMNS):
        group = first + tl.arange(0, 32)
        tl.store(OUT + row * WIDTH + group, float("-inf"), group < WIDTH)
    else:
        _paged_mqa_tile(Q, CACHE, WEIGHTS, OWNERS, UPPER, TABLE, OUT, Q0, Q1, W0, C0, T0, H, BH, PAGE, PAGES, REQUESTS, COLUMNS, WIDTH)



def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _cuda_tensors(first, *others):
    _require(first.is_cuda and torch.version.hip is None, "requires NVIDIA CUDA")
    _require(
        torch.cuda.get_device_capability(first.device) >= (8, 0),
        "requires sm_80 or newer",
    )
    _require(all(x.device == first.device for x in others), "tensor devices differ")


def _integer_vector(x, size, name):
    _require(
        x.ndim == 1 and x.shape[0] == size and x.dtype in (torch.int32, torch.int64),
        f"{name} must be int32/int64 [{size}]",
    )


def _cache_layout(cache, width):
    _require(
        cache.ndim == 3 and cache.shape[1] > 0 and cache.shape[2] == width,
        f"cache must be [pages,page_size,{width}]",
    )
    _require(
        cache.stride(2) == 1
        and cache.stride(1) == width
        and cache.stride(0) >= cache.shape[1] * width,
        "cache rows must be contiguous and pages must not overlap",
    )
    if cache.dtype == torch.uint8:
        _require(cache.stride(0) % 4 == 0, "byte pages must be FP32 aligned")


class PartialConfig(NamedTuple):
    block_n: int
    num_warps: int
    num_stages: int
    shared: int  # bytes Triton allocated for this config
    limit: int  # the device's per-block shared-memory limit


# KV rows per iteration, preferred first. On sm_86 (101,376 B per block) with
# 128-wide latent chunks the preferred 32 needs 26,624 B (fp8_ds_mla) or
# 59,392 B (BF16/FP16 rows) at D=512; 16 is the fallback for smaller limits.
PARTIAL_BLOCK_N = (32, 16)
_partial_configs: dict[tuple, PartialConfig] = {}


def shared_memory_limit(device):
    index = device.index if device.index is not None else torch.cuda.current_device()
    properties = triton.runtime.driver.active.utils.get_device_properties(index)
    return int(properties["max_shared_mem"])


def partial_config(args, grid, device, d, r, packed, dtype):
    """Pick the first BLOCK_N whose compiled kernel fits this device's shared memory.

    Uses Triton's own allocation (``metadata.shared`` of the warmed-up binary),
    so the check is the one the launcher enforces, not an estimate.
    """
    key = (device.index, d, r, packed, dtype)
    config = _partial_configs.get(key)
    if config is not None:
        return config
    limit = shared_memory_limit(device)
    num_warps = 8 if d == 512 else 4
    tried = []
    for block_n in PARTIAL_BLOCK_N:
        compiled = _sparse_partial.warmup(
            *args,
            BLOCK_N=block_n,
            num_warps=num_warps,
            num_stages=1,
            grid=grid,
        )
        shared = int(compiled.metadata.shared)
        tried.append(f"BLOCK_N={block_n}: {shared} B")
        if shared <= limit:
            config = PartialConfig(block_n, num_warps, 1, shared, limit)
            _partial_configs[key] = config
            return config
    raise ValueError(
        f"no sparse MLA config fits {limit} B of shared memory ({', '.join(tried)})"
    )


# sparse_mla's FP32 split partials and logs live
# in one persistent buffer per device, reserved when the attention layers are built
# (inside model load, before the KV pin is allocated), instead of a fresh
# (t, h, splits, 512) FP32 tensor per layer call: 64 MiB at a 1,024-token chunk and
# 32 heads. The memory is resident and counted at boot, so a pin that leaves too
# little room fails at start, not at a later send, and the per-call churn behind
# allocator growth is gone. Same kernels, same values (every partial is written
# before it is read); calls are serialised on one stream, so the layers share it.
_SPARSE_WORKSPACE: dict = {}


def reserve_sparse_workspace(device, rows: int, heads: int, latent_dim: int = 512) -> None:
    """Grow the device's sparse_mla workspace to hold ``rows`` x ``heads`` split rows
    (rows already counted with their split factor) of ``latent_dim`` FP32 values."""
    device = torch.device(device)
    key = (device.type, device.index)
    need = int(rows) * int(heads)
    have = _SPARSE_WORKSPACE.get(key)
    if have is not None and have[1].numel() >= need and have[0].numel() >= need * latent_dim:
        return
    _SPARSE_WORKSPACE[key] = (
        torch.empty((need * latent_dim,), dtype=torch.float32, device=device),
        torch.empty((need,), dtype=torch.float32, device=device),
    )


def _sparse_scratch(device, t: int, h: int, splits: int, d: int):
    """(partial, logs) views of the reserved workspace, or fresh tensors when none is
    reserved or it is too small for this call."""
    have = _SPARSE_WORKSPACE.get((device.type, device.index))
    rows = t * h * splits
    if have is not None and have[1].numel() >= rows and have[0].numel() >= rows * d:
        return (have[0][: rows * d].view(t, h, splits, d), have[1][:rows].view(t, h, splits))
    return (torch.empty((t, h, splits, d), dtype=torch.float32, device=device),
            torch.empty((t, h, splits), dtype=torch.float32, device=device))


def sparse_mla(
    q,
    kv,
    indices,
    scale,
    *,
    latent_dim=512,
    sinks=None,
    sink_owner=True,
    lengths=None,
    num_splits=None,
    output_dtype=None,
):
    """Sparse MLA over physical rows; always returns sink-inclusive natural LSE."""
    _require(q.ndim == 3 and kv.ndim == 2 and indices.ndim == 2, "invalid ranks")
    t, h, width = q.shape
    d, r = latent_dim, width - latent_dim
    _require(
        q.dtype in (torch.bfloat16, torch.float16) and h > 0,
        "queries must be BF16/FP16 with nonempty heads",
    )
    _require(d in (16, 32, 64, 128, 256, 512), "unsupported latent dimension")
    _require(r in (0, 16, 32, 64, 128), "unsupported rotary dimension")
    packed = kv.dtype == torch.uint8
    _require(
        (packed and d == 512 and r == 64 and kv.shape[1] == 656)
        or (not packed and kv.dtype == q.dtype and kv.shape[1] == width),
        "incompatible KV layout or dtype",
    )
    _require(kv.stride(1) == 1, "KV columns must be contiguous")
    if packed:
        _require(kv.stride(0) % 4 == 0, "packed rows must be FP32 aligned")
    _require(
        indices.shape[0] == t and indices.dtype in (torch.int32, torch.int64),
        "indices must be int32/int64 [T,K]",
    )
    _require(math.isfinite(scale) and scale > 0, "scale must be finite and positive")
    tensors = [kv, indices]
    if sinks is not None:
        _require(sinks.shape == (h,) and sinks.dtype == torch.float32, "invalid sinks")
        tensors.append(sinks)
        sinks = sinks.contiguous()
    if lengths is not None:
        _integer_vector(lengths, t, "lengths")
        tensors.append(lengths)
        lengths = lengths.contiguous()
    _cuda_tensors(q, *tensors)
    q = q if q.stride(-1) == 1 else q.contiguous()
    indices = indices if indices.stride(-1) == 1 else indices.contiguous()
    k = indices.shape[1]
    splits = num_splits if num_splits is not None else (8 if t < 32 and k > 256 else 1)
    _require(splits in (1, 2, 4, 8), "num_splits must be 1,2,4,8")
    dtype = q.dtype if output_dtype is None else output_dtype
    _require(
        dtype in (torch.float32, torch.bfloat16, torch.float16), "invalid output dtype"
    )
    output = torch.empty((t, h, d), dtype=dtype, device=q.device)
    lse = torch.empty((t, h), dtype=torch.float32, device=q.device)
    if not t:
        return output, lse
    partial, logs = _sparse_scratch(q.device, t, h, splits, d)
    grid = (t, triton.cdiv(h, 16), splits)
    args = (
        q,
        kv,
        indices,
        lengths if lengths is not None else indices,
        partial,
        logs,
        q.stride(0),
        q.stride(1),
        kv.stride(0),
        indices.stride(0),
        kv.shape[0],
        k,
        h,
        d,
        r,
        splits,
        float(scale),
        packed,
        lengths is not None,
    )
    config = partial_config(args, grid, q.device, d, r, packed, q.dtype)
    _sparse_partial[grid](
        *args,
        BLOCK_N=config.block_n,
        num_warps=config.num_warps,
        num_stages=config.num_stages,
    )
    _sparse_finish[(t, h)](
        partial,
        logs,
        sinks if sinks is not None else q,
        output,
        lse,
        h,
        d,
        splits,
        sinks is not None and sink_owner,
        num_warps=4,
    )
    return output, lse


# ---------------------------------------------------------------------------
# A decode/spec-verify path for sparse_mla.
#
# The default kernel's decode grid is (rows, cdiv(heads, 16), 8): about 96 CTAs
# on 82 SMs, each walking a fixed 1/8 of all K index columns, -1 holes and
# non-local DCP entries included, one dependent gather per BLOCK_N. At 6 rows x
# 2,048 top-k that is latency-bound, not bandwidth-bound. This path:
#   1. compacts each row's valid indices to the front and writes its count
#      (one small kernel; the -1 holes of SWA and of the DCP filter drop out);
#   2. runs the partial over each row's *valid* entries only, split into up to
#      32 per-row spans sized to fill the SMs, with HEAD_BLOCK (16 or 32) heads
#      per CTA so one KV gather serves all heads of a row;
#   3. finishes with the unchanged _sparse_finish (same partial/log2-LSE layout,
#      same sinks, same sink-inclusive natural LSE for the DCP combine).
# Same rows, same softmax, same LSE contract as sparse_mla. Not bitwise with it:
# the FP32 summation order differs (split boundaries move), and the default split
# P (hi + lo) keeps more of each probability than stock's input-dtype P.
# ---------------------------------------------------------------------------


@triton.jit(do_not_specialize=["I0", "O0", "K", "N"])
def _compact_indices(
    IDX,
    LEN_IN,
    OUT,
    LEN_OUT,
    I0,
    O0,
    K,
    N,
    HAS_LENGTHS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    column = tl.arange(0, BLOCK)
    inside = column < K
    index = tl.load(IDX + row * I0 + column, inside, -1).to(tl.int64)
    valid = inside & (index >= 0) & (index < N)
    if HAS_LENGTHS:
        usable = tl.minimum(tl.maximum(tl.load(LEN_IN + row), 0), K)
        valid = valid & (column < usable)
    flag = valid.to(tl.int32)
    position = tl.cumsum(flag, 0) - 1
    count = tl.sum(flag, 0)
    tl.store(OUT + row * O0 + position, index.to(tl.int32), valid)
    tl.store(OUT + row * O0 + column, -1, inside & (column >= count))
    tl.store(LEN_OUT + row, count)


@triton.jit(do_not_specialize=["Q0", "Q1", "K0", "I0", "H"])
def _sparse_partial_decode(
    Q,
    KV,
    IDX,
    LENGTHS,
    PART,
    LOGS,
    Q0,
    Q1,
    K0,
    I0,
    H,
    D: tl.constexpr,
    R: tl.constexpr,
    SPLITS: tl.constexpr,
    SCALE: tl.constexpr,
    PACKED: tl.constexpr,
    HEAD_BLOCK: tl.constexpr,
    BLOCK_N: tl.constexpr,
    P_MODE: tl.constexpr = 2,
):
    C: tl.constexpr = D if D < 128 else 128
    NG: tl.constexpr = D // C
    row = tl.program_id(0).to(tl.int64)
    heads = tl.program_id(1) * HEAD_BLOCK + tl.arange(0, HEAD_BLOCK)
    split = tl.program_id(2)
    c = tl.arange(0, C)
    live = heads[:, None] < H
    q_row = Q + row * Q0 + heads[:, None] * Q1
    q0 = tl.load(q_row + c[None], live, 0.0)
    if NG >= 2:
        q1 = tl.load(q_row + C + c[None], live, 0.0)
    if NG >= 4:
        q2 = tl.load(q_row + 2 * C + c[None], live, 0.0)
        q3 = tl.load(q_row + 3 * C + c[None], live, 0.0)
    if R:
        r = tl.arange(0, R)
        qr = tl.load(q_row + D + r[None], live, 0.0)
    maximum = tl.full((HEAD_BLOCK,), float("-inf"), tl.float32)
    denominator = tl.zeros((HEAD_BLOCK,), tl.float32)
    acc0 = tl.zeros((HEAD_BLOCK, C), tl.float32)
    if NG >= 2:
        acc1 = tl.zeros((HEAD_BLOCK, C), tl.float32)
    if NG >= 4:
        acc2 = tl.zeros((HEAD_BLOCK, C), tl.float32)
        acc3 = tl.zeros((HEAD_BLOCK, C), tl.float32)
    # The row's compacted valid entries, split into SPLITS spans of whole tiles.
    usable = tl.load(LENGTHS + row)
    span = tl.cdiv(tl.cdiv(usable, SPLITS), BLOCK_N) * BLOCK_N
    begin = split * span
    end = tl.minimum(begin + span, usable)
    for start in range(begin, end, BLOCK_N):
        column = start + tl.arange(0, BLOCK_N)
        index = tl.load(IDX + row * I0 + column, column < end, -1).to(tl.int64)
        valid = (column < end) & (index >= 0)
        rows = KV + index * K0
        k0 = _latent_chunk(rows, valid, c, 0, C, PACKED, q0.dtype)
        score = tl.dot(q0, tl.trans(k0), out_dtype=tl.float32)
        if NG >= 2:
            k1 = _latent_chunk(rows, valid, c, 1, C, PACKED, q0.dtype)
            score = tl.dot(q1, tl.trans(k1), score, out_dtype=tl.float32)
        if NG >= 4:
            k2 = _latent_chunk(rows, valid, c, 2, C, PACKED, q0.dtype)
            score = tl.dot(q2, tl.trans(k2), score, out_dtype=tl.float32)
            k3 = _latent_chunk(rows, valid, c, 3, C, PACKED, q0.dtype)
            score = tl.dot(q3, tl.trans(k3), score, out_dtype=tl.float32)
        if R:
            if PACKED:
                rope_ptr = (rows + 528).to(tl.pointer_type(tl.bfloat16))
                kr = tl.load(rope_ptr[:, None] + r[None], valid[:, None], 0.0)
            else:
                kr = tl.load(rows[:, None] + D + r[None], valid[:, None], 0.0)
            score = tl.dot(qr, tl.trans(kr.to(qr.dtype)), score, out_dtype=tl.float32)
        score = tl.where(valid[None], score * SCALE * LOG2E, float("-inf"))
        next_max = tl.maximum(maximum, tl.max(score, 1))
        safe_max = tl.where(next_max == float("-inf"), 0.0, next_max)
        alpha = tl.exp2(maximum - safe_max)
        probability = tl.exp2(score - safe_max[:, None])
        if P_MODE == 2:
            # Split P: P = hi + lo
            # with hi = P rounded to the V dtype and lo = (P - hi) rounded
            # (P - hi is exact in FP32), so P keeps about 16 mantissa bits; both
            # halves go through tensor-core dots with FP32 accumulation. V is
            # exact in its dtype (the e4m3 x 2^n latent is exact in BF16).
            hi = probability.to(q0.dtype)
            lo = (probability - hi.to(tl.float32)).to(q0.dtype)
            acc0 = tl.dot(hi, k0, acc0 * alpha[:, None], out_dtype=tl.float32)
            acc0 = tl.dot(lo, k0, acc0, out_dtype=tl.float32)
            if NG >= 2:
                acc1 = tl.dot(hi, k1, acc1 * alpha[:, None], out_dtype=tl.float32)
                acc1 = tl.dot(lo, k1, acc1, out_dtype=tl.float32)
            if NG >= 4:
                acc2 = tl.dot(hi, k2, acc2 * alpha[:, None], out_dtype=tl.float32)
                acc2 = tl.dot(lo, k2, acc2, out_dtype=tl.float32)
                acc3 = tl.dot(hi, k3, acc3 * alpha[:, None], out_dtype=tl.float32)
                acc3 = tl.dot(lo, k3, acc3, out_dtype=tl.float32)
        elif P_MODE == 1:
            # The "ieee" fallback: P x V as an
            # FP32 FMA dot (never TF32); V upcast in registers. Head block 16
            # only (32 falls off a cliff on sm_86, measured 2026-11-20).
            acc0 = tl.dot(probability, k0.to(tl.float32), acc0 * alpha[:, None],
                          input_precision="ieee", out_dtype=tl.float32)
            if NG >= 2:
                acc1 = tl.dot(probability, k1.to(tl.float32), acc1 * alpha[:, None],
                              input_precision="ieee", out_dtype=tl.float32)
            if NG >= 4:
                acc2 = tl.dot(probability, k2.to(tl.float32), acc2 * alpha[:, None],
                              input_precision="ieee", out_dtype=tl.float32)
                acc3 = tl.dot(probability, k3.to(tl.float32), acc3 * alpha[:, None],
                              input_precision="ieee", out_dtype=tl.float32)
        else:  # the stock kernel's input-dtype P (the bench's comparison arm only)
            weights = probability.to(q0.dtype)
            acc0 = tl.dot(weights, k0, acc0 * alpha[:, None], out_dtype=tl.float32)
            if NG >= 2:
                acc1 = tl.dot(weights, k1, acc1 * alpha[:, None], out_dtype=tl.float32)
            if NG >= 4:
                acc2 = tl.dot(weights, k2, acc2 * alpha[:, None], out_dtype=tl.float32)
                acc3 = tl.dot(weights, k3, acc3 * alpha[:, None], out_dtype=tl.float32)
        denominator = denominator * alpha + tl.sum(probability, 1)
        maximum = next_max
    safe_denominator = tl.where(denominator > 0, denominator, 1.0)
    logsum = tl.where(
        denominator > 0,
        maximum + tl.log2(safe_denominator),
        float("-inf"),
    )
    state = (row * H + heads.to(tl.int64)) * SPLITS + split
    out = PART + state[:, None] * D + c[None]
    tl.store(out, acc0 / safe_denominator[:, None], live)
    if NG >= 2:
        tl.store(out + C, acc1 / safe_denominator[:, None], live)
    if NG >= 4:
        tl.store(out + 2 * C, acc2 / safe_denominator[:, None], live)
        tl.store(out + 3 * C, acc3 / safe_denominator[:, None], live)
    tl.store(LOGS + state, logsum, heads < H)


DECODE_SPLITS = (1, 2, 4, 8, 16, 32)
# P x V precision modes
P_MODES = {"input": 0, "ieee": 1, "split": 2}
DECODE_MAX_ROWS = 64  # the decode path serves decode and spec-verify batches
_decode_configs: dict[tuple, PartialConfig] = {}
_sm_counts: dict[int, int] = {}


def decode_splits(rows: int, head_groups: int, k: int, sms: int, block_n: int = 32) -> int:
    """Splits per row: the smallest power of two giving at least 4 CTAs per SM,
    capped at 32 and at one tile of top-k columns per split."""
    ctas = max(rows * head_groups, 1)
    want = 1
    while want < 32 and ctas * want < 4 * sms:
        want *= 2
    tiles = max(-(-k // block_n), 1)
    while want > 1 and want > tiles:
        want //= 2
    return want


def decode_head_block(h: int, requested: int | None = None) -> int:
    """16 or 32 heads per CTA (32 = one gather serves all DCP-gathered heads)."""
    block = requested if requested is not None else (32 if h >= 32 else 16)
    if block not in (16, 32):
        raise ValueError(f"head block must be 16 or 32 (got {block})")
    return block


def _sm_count(device) -> int:
    index = device.index if device.index is not None else torch.cuda.current_device()
    if index not in _sm_counts:
        _sm_counts[index] = torch.cuda.get_device_properties(index).multi_processor_count
    return _sm_counts[index]


def _decode_config(args, grid, device, d, r, packed, dtype, head_block, p_mode=2):
    key = (device.index, d, r, packed, dtype, head_block, p_mode)
    config = _decode_configs.get(key)
    if config is not None:
        return config
    limit = shared_memory_limit(device)
    num_warps = 8 if d == 512 else 4
    tried = []
    for block_n in PARTIAL_BLOCK_N:
        compiled = _sparse_partial_decode.warmup(
            *args,
            HEAD_BLOCK=head_block,
            BLOCK_N=block_n,
            P_MODE=p_mode,
            num_warps=num_warps,
            num_stages=1,
            grid=grid,
        )
        shared = int(compiled.metadata.shared)
        tried.append(f"BLOCK_N={block_n}: {shared} B")
        if shared <= limit:
            config = PartialConfig(block_n, num_warps, 1, shared, limit)
            _decode_configs[key] = config
            return config
    raise ValueError(
        f"no sparse MLA decode config fits {limit} B of shared memory ({', '.join(tried)})"
    )


def sparse_mla_decode(
    q,
    kv,
    indices,
    scale,
    *,
    latent_dim=512,
    sinks=None,
    sink_owner=True,
    lengths=None,
    num_splits=None,
    head_block=None,
    output_dtype=None,
    fp32_p=True,
    p_mode=None,
):
    """sparse_mla for decode/spec-verify batches (rows <= DECODE_MAX_ROWS):
    the same inputs, outputs and sink-inclusive natural LSE.

    P x V precision: ``p_mode`` "split" (the default and the
    engine's path: P = hi + lo in the V dtype, two tensor-core dots, FP32
    accumulation), "ieee" (an FP32 FMA dot; always head block 16) or "input"
    (the stock kernel's input-dtype P, the bench's comparison arm only).
    ``fp32_p=False`` is "input"."""
    _require(q.ndim == 3 and kv.ndim == 2 and indices.ndim == 2, "invalid ranks")
    t, h, width = q.shape
    d, r = latent_dim, width - latent_dim
    _require(
        q.dtype in (torch.bfloat16, torch.float16) and h > 0,
        "queries must be BF16/FP16 with nonempty heads",
    )
    _require(d in (16, 32, 64, 128, 256, 512), "unsupported latent dimension")
    _require(r in (0, 16, 32, 64, 128), "unsupported rotary dimension")
    packed = kv.dtype == torch.uint8
    _require(
        (packed and d == 512 and r == 64 and kv.shape[1] == 656)
        or (not packed and kv.dtype == q.dtype and kv.shape[1] == width),
        "incompatible KV layout or dtype",
    )
    _require(kv.stride(1) == 1, "KV columns must be contiguous")
    if packed:
        _require(kv.stride(0) % 4 == 0, "packed rows must be FP32 aligned")
    _require(
        indices.shape[0] == t and indices.dtype in (torch.int32, torch.int64),
        "indices must be int32/int64 [T,K]",
    )
    _require(math.isfinite(scale) and scale > 0, "scale must be finite and positive")
    _require(t <= DECODE_MAX_ROWS, f"the decode path takes at most {DECODE_MAX_ROWS} rows")
    tensors = [kv, indices]
    if sinks is not None:
        _require(sinks.shape == (h,) and sinks.dtype == torch.float32, "invalid sinks")
        tensors.append(sinks)
        sinks = sinks.contiguous()
    if lengths is not None:
        _integer_vector(lengths, t, "lengths")
        tensors.append(lengths)
        lengths = lengths.contiguous()
    _cuda_tensors(q, *tensors)
    q = q if q.stride(-1) == 1 else q.contiguous()
    indices = indices if indices.stride(-1) == 1 else indices.contiguous()
    k = indices.shape[1]
    dtype = q.dtype if output_dtype is None else output_dtype
    _require(
        dtype in (torch.float32, torch.bfloat16, torch.float16), "invalid output dtype"
    )
    output = torch.empty((t, h, d), dtype=dtype, device=q.device)
    lse = torch.empty((t, h), dtype=torch.float32, device=q.device)
    if not t:
        return output, lse
    mode = p_mode if p_mode is not None else ("split" if fp32_p else "input")
    _require(mode in P_MODES, f"p_mode must be one of {sorted(P_MODES)}")
    if mode == "ieee":
        head_block = 16  # the FP32 FMA dot falls off a cliff at 32 heads (sm_86)
    head_block = decode_head_block(h, head_block)
    groups = triton.cdiv(h, head_block)
    splits = (
        num_splits
        if num_splits is not None
        else decode_splits(t, groups, k, _sm_count(q.device))
    )
    _require(splits in DECODE_SPLITS, f"num_splits must be one of {DECODE_SPLITS}")
    compact = torch.empty((t, max(k, 1)), dtype=torch.int32, device=q.device)
    counts = torch.empty((t,), dtype=torch.int32, device=q.device)
    _compact_indices[(t,)](
        indices,
        lengths if lengths is not None else indices,
        compact,
        counts,
        indices.stride(0),
        compact.stride(0),
        k,
        kv.shape[0],
        lengths is not None,
        BLOCK=triton.next_power_of_2(max(k, 1)),
        num_warps=4,
    )
    partial = torch.empty((t, h, splits, d), dtype=torch.float32, device=q.device)
    logs = torch.empty((t, h, splits), dtype=torch.float32, device=q.device)
    grid = (t, groups, splits)
    args = (
        q,
        kv,
        compact,
        counts,
        partial,
        logs,
        q.stride(0),
        q.stride(1),
        kv.stride(0),
        compact.stride(0),
        h,
        d,
        r,
        splits,
        float(scale),
        packed,
    )
    config = _decode_config(args, grid, q.device, d, r, packed, q.dtype, head_block,
                            P_MODES[mode])
    _sparse_partial_decode[grid](
        *args,
        HEAD_BLOCK=head_block,
        BLOCK_N=config.block_n,
        P_MODE=P_MODES[mode],
        num_warps=config.num_warps,
        num_stages=config.num_stages,
    )
    _sparse_finish[(t, h)](
        partial,
        logs,
        sinks if sinks is not None else q,
        output,
        lse,
        h,
        d,
        splits,
        sinks is not None and sink_owner,
        num_warps=8 if splits >= 16 else 4,
    )
    return output, lse


def write_mla_cache(kv_c, k_pe, cache, slots):
    """Compatibility helper; the engine inherits its CUDA cache writer."""
    from vllm import _custom_ops as ops

    _require(
        kv_c.ndim == 2 and kv_c.shape[1] == 512 and k_pe.shape == (kv_c.shape[0], 64),
        "requires kv_c [T,512] and k_pe [T,64]",
    )
    _require(
        kv_c.dtype in (torch.bfloat16, torch.float16) and k_pe.dtype == kv_c.dtype,
        "cache inputs must have the same BF16/FP16 dtype",
    )
    packed = cache.dtype == torch.uint8
    _require(
        not packed or kv_c.dtype == torch.bfloat16,
        "fp8_ds_mla CUDA writer requires BF16 rotary input",
    )
    _require(packed or cache.dtype == kv_c.dtype, "cache dtype mismatch")
    _cache_layout(cache, 656 if packed else 576)
    _integer_vector(slots, slots.numel(), "slots")
    _require(slots.numel() <= kv_c.shape[0], "slots exceed input rows")
    _cuda_tensors(kv_c, k_pe, cache, slots)
    if not slots.numel() or not cache.numel():
        return
    kv_c = kv_c if kv_c.stride(-1) == 1 else kv_c.contiguous()
    k_pe = k_pe if k_pe.stride(-1) == 1 else k_pe.contiguous()
    # Keep this helper's invalid-slot contract; the engine supplies valid
    # int64 slot_mapping entries or PAD_ID directly to the inherited writer.
    capacity = cache.shape[0] * cache.shape[1]
    slots = torch.where((slots >= 0) & (slots < capacity), slots, -1).long()
    ops.concat_and_cache_mla(
        kv_c,
        k_pe,
        cache,
        slots.contiguous(),
        kv_cache_dtype="fp8_ds_mla" if packed else "auto",
        scale=torch.ones((), dtype=torch.float32, device=cache.device),
    )


def write_index_pool(cache, slots, quant, scales):
    _require(cache.dtype == torch.uint8, "IndexPool cache must contain bytes")
    _cache_layout(cache, 132)
    _require(
        quant.ndim == 2 and quant.shape[1] == 128 and quant.dtype == torch.uint8,
        "pooled keys must be uint8 [G,128]",
    )
    _require(
        scales.shape == (quant.shape[0],) and scales.dtype == torch.float32,
        "scales must be FP32 [G]",
    )
    _integer_vector(slots, quant.shape[0], "slots")
    _cuda_tensors(quant, cache, slots, scales)
    if not quant.shape[0]:
        return
    quant = quant if quant.stride(-1) == 1 else quant.contiguous()
    _write_pool[(quant.shape[0],)](
        cache,
        slots.contiguous(),
        quant,
        scales.contiguous(),
        cache.stride(0),
        quant.stride(0),
        cache.shape[1],
        cache.shape[0] * cache.shape[1],
        num_warps=4,
    )


def paged_mqa_logits(q, cache, weights, owners, upper, table, max_groups, *, early_exit=False):
    _require(
        q.ndim == 3
        and q.dtype == torch.uint8
        and q.shape[2] == 128
        and 1 <= q.shape[1] <= 64,
        "MQA queries must be uint8 [T,H,128], 1<=H<=64",
    )
    t, h, _ = q.shape
    _require(cache.dtype == torch.uint8, "IndexPool cache must contain bytes")
    _cache_layout(cache, 132)
    _require(
        weights.shape == (t, h) and weights.dtype == torch.float32,
        "weights must be FP32 [T,H]",
    )
    _integer_vector(owners, t, "owners")
    _integer_vector(upper, t, "upper")
    _require(
        table.ndim == 2 and table.dtype in (torch.int32, torch.int64),
        "table must be an integer matrix",
    )
    _require(isinstance(max_groups, int) and max_groups >= 0, "invalid max_groups")
    _cuda_tensors(q, cache, weights, owners, upper, table)
    output = torch.empty((t, max_groups), dtype=torch.float32, device=q.device)
    if not t or not max_groups:
        return output
    q = q if q.stride(-1) == 1 else q.contiguous()
    weights = weights.contiguous()
    table = table if table.stride(-1) == 1 else table.contiguous()
    kernel = _paged_mqa_skip if early_exit else _paged_mqa
    kernel[(t, triton.cdiv(max_groups, 32))](
        q,
        cache,
        weights,
        owners.contiguous(),
        upper.contiguous(),
        table,
        output,
        q.stride(0),
        q.stride(1),
        weights.stride(0),
        cache.stride(0),
        table.stride(0),
        h,
        max(16, triton.next_power_of_2(h)),
        cache.shape[1],
        cache.shape[0],
        table.shape[0],
        table.shape[1],
        max_groups,
        num_warps=4,
        num_stages=1,
    )
    return output


# hy4_reference.fwht_quantize in one kernel per
# row block, bit for bit: the same butterfly order (stride 1, 2, ..., 64; low
# half = a + b, high half = a - b, in FP32), the same FP32 scalar multiply by
# 128**-0.5, amax / 448 with IEEE division, ceil(log2) through libdevice log2f
# (torch's CUDA log2), a power-of-two scale (so x / scale is exact as x * 2**-k),
# and the same saturating round-to-nearest-even E4M3 bit arithmetic.
@triton.jit
def _fwht_quantize_kernel(X, SCRATCH, OUT, SCALE, X0, ROWS):
    row = tl.program_id(0).to(tl.int64)
    d = tl.arange(0, 128)
    x = tl.load(X + row * X0 + d).to(tl.float32)
    base = SCRATCH + row * 128
    for level in tl.static_range(7):
        stride = 1 << level
        tl.store(base + d, x)
        tl.debug_barrier()
        partner = tl.load(base + (d ^ stride))
        tl.debug_barrier()
        high = (d & stride) != 0
        x = tl.where(high, partner - x, x + partner)
    x = x * 0.08838834764831845
    amax = tl.max(tl.abs(x), 0)
    ratio = tl.math.div_rn(tl.maximum(amax, 1e-4), 448.0)
    exponent = libdevice.ceil(libdevice.log2(ratio))
    scale = libdevice.exp2(exponent)
    y = x * libdevice.exp2(-exponent)
    sign = libdevice.signbit(y) != 0
    magnitude = tl.minimum(tl.abs(y), 448.0)
    bits = magnitude.to(tl.int32, bitcast=True)
    rounded = ((bits + 0x7FFFF + ((bits >> 20) & 1)) >> 20) - 960
    subnormal = libdevice.rint(magnitude * 512.0).to(tl.int32)
    code = tl.where(magnitude < 0.015625, subnormal, rounded)
    code = tl.minimum(tl.maximum(code, 0), 126)
    code = tl.where(y != y, 127, code)
    code = code | tl.where(sign, 128, 0)
    tl.store(OUT + row * 128 + d, code.to(tl.uint8))
    tl.store(SCALE + row, scale)


def fwht_quantize_fused(vectors):
    """``hy4_reference.fwht_quantize`` on CUDA in one launch: ``vectors`` [N,128]
    BF16/FP16/FP32 -> (uint8 [N,128], FP32 scales [N]), bit for bit."""
    _require(vectors.ndim == 2 and vectors.shape[1] == 128, "IndexPool quantization requires [rows,128]")
    _cuda_tensors(vectors)
    n = vectors.shape[0]
    out = torch.empty((n, 128), dtype=torch.uint8, device=vectors.device)
    scale = torch.empty((n,), dtype=torch.float32, device=vectors.device)
    if not n:
        return out, scale
    x = vectors if vectors.stride(-1) == 1 else vectors.contiguous()
    scratch = torch.empty((n, 128), dtype=torch.float32, device=vectors.device)
    _fwht_quantize_kernel[(n,)](x, scratch, out, scale, x.stride(0), n, num_warps=1)
    return out, scale


def indexpool_select(
    q,
    cache,
    weights,
    owners,
    positions,
    table,
    *,
    max_groups,
    ratio=4,
    group_topk=512,
    logits_bytes=64 * 1024 * 1024,
    active=None,
):
    """Score past groups in bounded row chunks and append the causal tail."""
    _require(ratio >= 1 and group_topk >= 1, "invalid group geometry")
    _integer_vector(positions, q.shape[0], "positions")
    _cuda_tensors(q, positions)
    _require(isinstance(max_groups, int) and max_groups >= 0, "invalid max_groups")
    row_bytes = max(1, max_groups * 4)
    _require(logits_bytes >= row_bytes, "logits budget cannot hold one row")
    if active is not None:
        _require(
            active.shape == positions.shape and active.dtype == torch.bool,
            "active must be bool [T]",
        )
        _cuda_tensors(q, active)
    result = torch.full(
        (q.shape[0], group_topk * ratio), -1, dtype=torch.int32, device=q.device
    )
    chunk = max(1, logits_bytes // row_bytes)
    for begin in range(0, q.shape[0], chunk):
        end = min(q.shape[0], begin + chunk)
        pos = positions[begin:end]
        live = pos >= 0
        if active is not None:
            live = live & active[begin:end]
        upper = torch.where(live, pos // ratio, 0)
        logits = paged_mqa_logits(
            q[begin:end],
            cache,
            weights[begin:end],
            owners[begin:end],
            upper,
            table,
            max_groups,
        )
        groups = select_groups(logits, upper, group_topk - 1)
        result[begin:end].copy_(expand_groups(groups, pos, ratio, active=live))
    return result
