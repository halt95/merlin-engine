# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Triton kernels for vllm/merlin/hy4_group_select.py (imported on first use)."""

import torch

from vllm.triton_utils import tl, triton

_BLOCK = 1024


@triton.jit
def _mask_kernel(L, U, S, W, L0, S0, UNIT, SIZE, RANK,
                 SHARDED: tl.constexpr, BLOCK: tl.constexpr):
    # Hy4ShardLayout.num_local_groups on
    # upper[t] (non-negative, so // and % match Python), then the masked fill.
    t = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    ok = col < W
    n = tl.load(U + t).to(tl.int64)
    if SHARDED:
        span = UNIT * SIZE
        rem = n % span - RANK * UNIT
        rem = tl.minimum(tl.maximum(rem, 0), UNIT)
        n = (n // span) * UNIT + rem
    x = tl.load(L + t * L0 + col, mask=ok, other=0.0).to(tl.float32)
    x = tl.where(col.to(tl.int64) >= n, float("-inf"), x)
    tl.store(S + t * S0 + col, x, mask=ok)


@triton.jit
def _pack_kernel(V, I, OUT, COUNT, BUDGET, V0, I0, UNIT, SIZE, RANK,
                 SHARDED: tl.constexpr, BLOCK: tl.constexpr):
    # where(value != -inf, id, -1), then
    # local_to_group, packed with the score; -inf / -1 past COUNT.
    t = tl.program_id(0)
    k = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    inside = k < COUNT
    v = tl.load(V + t * V0 + k, mask=inside, other=float("-inf"))
    i = tl.load(I + t * I0 + k, mask=inside, other=-1).to(tl.int64)
    local = tl.where(inside & (v != float("-inf")), i, -1)
    gid = local
    if SHARDED:
        gid = ((local // UNIT) * SIZE + RANK) * UNIT + local % UNIT
    gid = tl.where(local >= 0, gid, -1)
    keep = k < BUDGET
    base = OUT + (t * BUDGET + k) * 2
    tl.store(base, v, mask=keep)
    tl.store(base + 1, gid.to(tl.float32), mask=keep)


@triton.jit
def _merge_kernel(G, S, GID, T, K, NK, G0, G1, G2, BLOCK: tl.constexpr):
    # Permute [N, T, K] -> [T, N*K], the gid
    # back to int64, scores of gid < 0 masked to -inf.
    t = tl.program_id(0)
    j = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    ok = j < NK
    n = j // K
    k = j % K
    src = G + n * G0 + t * G1 + k * G2
    score = tl.load(src, mask=ok, other=0.0)
    gid = tl.load(src + 1, mask=ok, other=-1.0).to(tl.int64)
    tl.store(S + t * NK + j, tl.where(gid < 0, float("-inf"), score), mask=ok)
    tl.store(GID + t * NK + j, gid, mask=ok)


@triton.jit
def _pick_kernel(V, P, GID, OUT, COUNT, NK, V0, P0, BLOCK: tl.constexpr):
    t = tl.program_id(0)
    k = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    ok = k < COUNT
    v = tl.load(V + t * V0 + k, mask=ok, other=float("-inf"))
    p = tl.load(P + t * P0 + k, mask=ok, other=0)
    g = tl.load(GID + t * NK + p, mask=ok, other=0)
    chosen = tl.where(v != float("-inf"), g, 2147483647).to(tl.int32)
    tl.store(OUT + t * COUNT + k, chosen, mask=ok)


@triton.jit
def _finish_kernel(C, OUT, COUNT, BUDGET, BLOCK: tl.constexpr):
    t = tl.program_id(0)
    k = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    inside = k < COUNT
    c = tl.load(C + t * COUNT + k, mask=inside, other=2147483647)
    c = tl.where(c == 2147483647, -1, c)
    tl.store(OUT + t * BUDGET + k, c, mask=k < BUDGET)


def mask(logits, upper, scores, sharded, unit, size, rank):
    t, w = logits.shape
    assert logits.stride(1) == 1 and scores.is_contiguous() and upper.stride(0) == 1
    _mask_kernel[(t, triton.cdiv(w, _BLOCK))](
        logits, upper, scores, w, logits.stride(0), scores.stride(0), unit, size, rank,
        SHARDED=sharded, BLOCK=_BLOCK)


def pack(values, ids, packed, count, budget, sharded, unit, size, rank):
    t = packed.shape[0]
    _pack_kernel[(t, triton.cdiv(budget, _BLOCK))](
        values, ids, packed, count, budget, values.stride(0) if count else 0,
        ids.stride(0) if count else 0, unit, size, rank, SHARDED=sharded, BLOCK=_BLOCK)


def merge(gathered, s, g):
    n, t, k, two = gathered.shape
    assert two == 2 and gathered.stride(3) == 1
    _merge_kernel[(t, triton.cdiv(n * k, _BLOCK))](
        gathered, s, g, t, k, n * k, gathered.stride(0), gathered.stride(1),
        gathered.stride(2), BLOCK=_BLOCK)


def pick(values, pick_ids, g, chosen):
    t, count = chosen.shape
    _pick_kernel[(t, triton.cdiv(count, _BLOCK))](
        values, pick_ids, g, chosen, count, g.shape[1], values.stride(0),
        pick_ids.stride(0), BLOCK=_BLOCK)


def finish(chosen, out, count, budget):
    t = out.shape[0]
    assert chosen.is_contiguous()
    _finish_kernel[(t, triton.cdiv(budget, _BLOCK))](
        chosen, out, count, budget, BLOCK=_BLOCK)

