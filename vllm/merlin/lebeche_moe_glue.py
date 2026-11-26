# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 lebeche hill-climb (fork of Merlin 3.0.2.2)
"""Small-batch MoE "glue" kernels for sm_80/86/89 (lebeche fork).

A decode step of Qwen3.8-Flash-Next runs every MoE layer with M <= 4 rows
(M <= 6 with deeper speculation). Around the two Marlin expert GEMMs the stock
path launches, per layer: ``topkGating``, ``moe_align_block_size_kernel``,
``count_and_sort_expert_tokens_kernel``, ``moe_sum_vec_dynamic_kernel``, the
shared-expert ``sigmoid`` and broadcast ``mul`` and the Inductor ``add`` that
joins the shared and routed outputs. For ``1 <= M <= MAX_M`` three Triton
kernels replace those seven:

* ``_route_kernel`` (one single-warp CTA per row, the stock
  ``topkGating<16,512,4,16,32,int,bf16>`` lane for lane): softmax over the 512
  router logits, top-k and renormalisation -> ``topk_weights``, ``topk_ids``.

  - every lane folds its 16 ``expf`` values (columns ``8t..8t+7`` then
    ``256+8t..``) in order, then the 32 lane sums are combined by a butterfly
    over lane xor 16, 8, 4, 2, 1;
  - ``expf`` is libdevice ``__nv_expf`` in both (Triton's ``tl.exp`` is
    ``__nv_fast_expf`` and is NOT used); the two reciprocals are correctly
    rounded (:func:`_recip_f32`; Triton's own fp32 division is not IEEE);
  - top-k picks the largest probability, lowest index first on ties; NaN/Inf
    probabilities count as 0 and padding rows get id -1, as in the stock kernel;
  - the weights are renormalised by the picks summed in pick order.

* ``_align_kernel`` (one CTA, up to 6 rows): the expert-parallel map and the
  block-size-8 alignment -> ``sorted_token_ids``, ``expert_ids``,
  ``num_tokens_post_padded`` (``_route_kernel`` pre-fills the first two).
  Blocks are in ascending local expert id and a block lists its flat
  ``token * top_k + k`` entries in ascending order. The stock
  ``count_and_sort`` kernel ranks the entries of one expert with ``atomicAdd``:
  ascending inside a warp of 32 entries, so identical for M <= 3, but a race
  between warps for M >= 4 (the same input gives different orders run to run,
  ``results/stock_sort_order.json`` in the run dir). Ascending is one of the
  orders it produces, and the Marlin GEMM result does not depend on the order
  inside a block (bitwise test). An expert holds at most M <= 8 = block size
  entries, i.e. one block: that is where ``MAX_M`` comes from. With 7 or 8
  rows the stock alignment kernels run (they are faster there).

* ``_post_kernel``: for every output element, the fp32 sum over the local
  top-k rows in k order rounded to bf16 (``moe_sum``), the shared-expert
  ``sigmoid(gate) * down`` rounded to bf16 (torch's bf16 sigmoid is the fp32
  ``1 / (1 + exp(-x))`` rounded once) and their bf16 sum.

Both Marlin GEMMs, the routed activation and the shared-expert GEMMs are
untouched and see bit-identical inputs.

Dispatch is a run-time Python test on the row count. That is safe here because
the whole MoE runner executes inside an opaque custom op
(``vllm.moe_forward[_shared]`` and the ``vllm.moe_forward_shared_glue`` twin
registered in ``runner/moe_runner.py``): torch.compile never traces it, and a
CUDA graph captures whichever kernels the capture-time M selects. The only
trace-time decision is which of the two ops ``MoERunner.forward`` calls; it
depends on the switches and on static layer properties, never on M.

Two independent pieces:

1. route + align: :func:`route` (called by ``MoERunner._apply_quant_method``
   instead of the router) leaves the alignment in a one-slot stash keyed on the
   identity of ``topk_ids``; ``fused_marlin_moe`` picks it up with
   :func:`take_align` instead of calling ``moe_align_block_size``.
2. post: inside ``vllm.moe_forward_shared_glue`` :func:`forward_combined`
   opens a context; ``MarlinExperts`` defers its ``moe_sum``
   (:func:`defer_sum`) and the shared-expert MLP defers its gate
   (:func:`defer_shared_gate`); the fused kernel then writes the combined
   output. Whatever was deferred but cannot be fused is completed with the
   stock ops, so every fallback is the stock arithmetic.

``MERLIN_LEBECHE_MOE_GLUE`` (default "1"; "0" = stock) is read once at import
and is a hashed compile factor; it also needs ``MERLIN_LEBECHE_SM86_LL`` and an
sm_80/86/89 device (:func:`glue_active`).
"""

from __future__ import annotations

from collections.abc import Iterable
from functools import partial
from typing import Any

import torch

from vllm.merlin.envs import read_registered
from vllm.merlin.lebeche_sm86 import SM86_LL_SWITCH, sm86_ll_active
from vllm.triton_utils import tl, triton

try:
    # tl.exp lowers to __nv_fast_expf; the stock kernels (and torch) call
    # __nv_expf, which is libdevice.exp.
    from triton.language.extra import libdevice
except ImportError:  # no Triton: the glue stays off
    libdevice = None

MOE_GLUE_SWITCH: bool = bool(read_registered("MERLIN_LEBECHE_MOE_GLUE"))
# Module bool for the cheap test on the hot path.
GLUE_SWITCH: bool = MOE_GLUE_SWITCH and SM86_LL_SWITCH and libdevice is not None
MOE_ROUTE_FUSE_SWITCH: bool = bool(read_registered("MERLIN_LEBECHE_MOE_ROUTE_FUSE"))
# Single-CTA CuTe-DSL fused route+align and hardware-PTX post combine.
ROUTE_FUSE_SWITCH: bool = MOE_ROUTE_FUSE_SWITCH and GLUE_SWITCH

MAX_M = 8  # one block of 8 per expert holds at most MAX_M entries
BLOCK_SIZE_M = 8  # the Marlin MoE block size fused_marlin_moe picks for M <= 8
NUM_EXPERTS = 512  # topkGating<16, 512, 4, 16, 32, int, bf16>
MAX_TOPK = 16

# Launch shapes (CUDA graph replay sweeps on an RTX 3090, see the run dir).
ALIGN_NUM_WARPS = 8
ALIGN_MAX_ENTRIES = 64  # fused alignment up to 6 rows of top-10
# Row-sum butterfly: True = Triton's own 32-lane reduction (shuffle-xor 16..1,
# the stock order; checked bit for bit by the unit test), False = the same
# adds spelled out through scratch memory (slower, no reliance on Triton).
LANE_SUM_REDUCE = True
POST_BLOCK = 256  # 10 single-warp CTAs per row of 2560 (fbc1a32)
POST_BLOCK_OPT = 128  # 20 single-warp CTAs per row -> 80 CTAs at M=4 (~82 SMs on 3090)
POST_NUM_WARPS = 1
ROUTE_FUSE_MAX_M = 6
# How torch computes sigmoid on bf16 (checked over all 65536 inputs by the test).
SIG_ROUNDED = 1  # round to bf16 after exp, after 1 + e and after the division
SIG_SINGLE = 2  # fp32 throughout, one rounding to bf16 (what torch does)
SIG_PRECOMPUTED = 0  # gate tensor already holds sigmoid(gate)
POST_SIG_MODE = SIG_SINGLE

# Kernel launch counters (tests assert on them); never read by the engine.
CALLS = {"route_align": 0, "route_align_cute": 0, "post": 0, "post_opt": 0, "take_align": 0}


def glue_active() -> bool:
    """Both switches on and the device is sm_80/86/89."""
    return GLUE_SWITCH and sm86_ll_active()


def route_fuse_active() -> bool:
    """Fused single-CTA route+align and hardware-PTX post combine active on sm_8x."""
    return ROUTE_FUSE_SWITCH and sm86_ll_active()


# --------------------------------------------------------------------- kernels
@triton.jit
def _recip_f32(y):
    """Correctly rounded fp32 ``1 / y`` (IEEE, like the stock ``1.f / y``).
    Triton's fp32 division is PTX ``div.full`` (up to 2 ulp off) and an fp64
    division costs ~0.35 us on an RTX 3090. The fp64 residual ``1 - q * y`` of
    an fp32 candidate ``q`` is exact (48-bit product, then Sterbenz), so the
    candidate within 2 ulp of the approximation with the smallest residual is
    the rounded quotient (a quotient is never an exact midpoint)."""
    q0 = 1.0 / y
    qb = q0.to(tl.int32, bitcast=True)
    yd = y.to(tl.float64)
    best_q = q0
    best_r = tl.abs(1.0 - q0.to(tl.float64) * yd)
    for j in tl.static_range(5):
        q = (qb + (j - 2)).to(tl.float32, bitcast=True)
        r = tl.abs(1.0 - q.to(tl.float64) * yd)
        take = r < best_r
        best_q = tl.where(take, q, best_q)
        best_r = tl.where(take, r, best_r)
    return best_q


@triton.jit
def _route_kernel(
    logits_ptr,  # bf16 [M, E]
    pad_ptr,  # uint8 [M] padding rows (ids = -1)
    w_ptr,  # f32 [M, TOPK]
    ids_ptr,  # int32 [M, TOPK]
    scr_ptr,  # f32 [M, 32] scratch (LANE_SUM_REDUCE == 0)
    sorted_ptr,  # int32 [M * TOPK * BS] (FILL)
    eids_ptr,  # int32 [M * TOPK] (FILL)
    E: tl.constexpr,
    TOPK: tl.constexpr,
    BS: tl.constexpr,
    FB: tl.constexpr,  # power of two >= TOPK * BS
    HAS_PAD: tl.constexpr,
    RENORM: tl.constexpr,
    FILL: tl.constexpr,
    LANE_SUM_REDUCE: tl.constexpr,
):
    """One CTA of ONE warp per row; lane t is lane t of the stock topkGating
    warp and owns the same 16 columns: 8t..8t+7 and 256+8t..256+8t+7."""
    m = tl.program_id(0)
    row = logits_ptr + m * E
    lane = tl.arange(0, 32)
    xs = ()
    for v in tl.static_range(16):
        xs += (tl.load(row + lane * 8 + ((v // 8) * (E // 2) + v % 8)).to(tl.float32),)  # type: ignore[assignment]
    lmax = xs[0]
    for v in tl.static_range(1, 16):
        lmax = tl.maximum(lmax, xs[v])
    mx = tl.max(lmax, axis=0)

    # Row sum in the stock order: every lane folds its 16 values in order,
    # then the lanes are combined by a butterfly over lane xor 16, 8, 4, 2, 1.
    es = ()
    for v in tl.static_range(16):
        es += (libdevice.exp(xs[v] - mx),)  # type: ignore[assignment]
    s = es[0]
    for v in tl.static_range(1, 16):
        s = s + es[v]
    if LANE_SUM_REDUCE:
        # Triton reduces a 32-element vector held one element per lane with
        # shuffle-xor 16, 8, 4, 2, 1: the stock butterfly itself.
        rsum = tl.sum(s, axis=0)
    else:
        # The same butterfly spelled out, through scratch memory.
        tl.store(scr_ptr + m * 32 + lane, s)
        tl.debug_barrier()
        l0 = ()
        for t in tl.static_range(32):
            l0 += (tl.load(scr_ptr + m * 32 + t),)  # type: ignore[assignment]
        l1 = ()
        for t in tl.static_range(16):
            l1 += (l0[t] + l0[t + 16],)  # type: ignore[assignment]
        l2 = ()
        for t in tl.static_range(8):
            l2 += (l1[t] + l1[t + 8],)  # type: ignore[assignment]
        l3 = ()
        for t in tl.static_range(4):
            l3 += (l2[t] + l2[t + 4],)  # type: ignore[assignment]
        rsum = (l3[0] + l3[2]) + (l3[1] + l3[3])
    recip = _recip_f32(rsum)

    # Keys: p >= +0, so its bit pattern orders like the value. NaN or Inf
    # (exponent all ones) -> 0, like the stock kernel.
    keys = ()
    for v in tl.static_range(16):
        pbits = (es[v] * recip).to(tl.int32, bitcast=True)
        keys += (tl.where((pbits & 0x7F800000) == 0x7F800000, 0, pbits),)  # type: ignore[assignment]

    # Top-k: the largest p, the lowest column on ties (two 32-bit reductions
    # per pick; one 64-bit reduction over a packed key costs 3x as much).
    ws = ()
    idl = ()
    for _k in tl.static_range(TOPK):
        lbest = keys[0]
        for v in tl.static_range(1, 16):
            lbest = tl.maximum(lbest, keys[v])
        best = tl.max(lbest, axis=0)
        lcol = tl.zeros([32], dtype=tl.int32) + E
        for u in tl.static_range(16):  # v = 15..0: the lowest column wins
            lcol = tl.where(
                keys[15 - u] == best,
                lane * 8 + (((15 - u) // 8) * (E // 2) + (15 - u) % 8),
                lcol,
            )
        cbest = tl.min(lcol, axis=0)
        nkeys = ()
        for v in tl.static_range(16):
            nkeys += (  # type: ignore[assignment]
                tl.where(lane * 8 + ((v // 8) * (E // 2) + v % 8) == cbest, -1, keys[v]),
            )
        keys = nkeys
        ws += (best.to(tl.float32, bitcast=True),)  # type: ignore[assignment]
        idl += (cbest,)  # type: ignore[assignment]

    scale = 1.0
    if RENORM:
        ssum = 0.0
        for k in tl.static_range(TOPK):
            ssum = ssum + ws[k]
        scale = _recip_f32(tl.where(ssum > 0.0, ssum, 1.0))
    is_pad = False
    if HAS_PAD:
        is_pad = tl.load(pad_ptr + m) != 0
    for k in tl.static_range(TOPK):
        tl.store(w_ptr + m * TOPK + k, ws[k] * scale)
        idk = idl[k]
        if HAS_PAD:
            idk = tl.where(is_pad, -1, idk)
        tl.store(ids_ptr + m * TOPK + k, idk)
    if FILL:
        # This row's share of the alignment buffers; _align_kernel (the next
        # launch on the stream) only scatters into them.
        n = tl.num_programs(0) * TOPK
        fi = tl.arange(0, FB)
        tl.store(sorted_ptr + m * (TOPK * BS) + fi, tl.zeros([FB], dtype=tl.int32) + n, mask=fi < TOPK * BS)
        tl.store(eids_ptr + m * TOPK + fi, tl.zeros([FB], dtype=tl.int32) - 1, mask=fi < TOPK)


@triton.jit
def _align_kernel(
    ids_ptr,  # int32 [M, TOPK]
    map_ptr,  # int32 [E] global -> local expert (-1 = not on this rank)
    sorted_ptr,  # int32 [M * TOPK * BS], pre-filled with n
    eids_ptr,  # int32 [M * TOPK], pre-filled with -1
    npost_ptr,  # int32 [1]
    n,  # M * TOPK entries
    NB: tl.constexpr,  # power of two >= n
    E: tl.constexpr,
    BS: tl.constexpr,
    HAS_MAP: tl.constexpr,
):
    """The scatter half of moe_align_block_size for n <= NB entries with at
    most BS entries per expert (one block each); _route_kernel did the fill."""
    ei = tl.arange(0, NB)
    emask = ei < n
    gid = tl.load(ids_ptr + ei, mask=emask, other=-1)
    gvalid = emask & (gid >= 0) & (gid < E)
    if HAS_MAP:
        loc = tl.load(map_ptr + gid, mask=gvalid, other=-1)
    else:
        loc = tl.where(gvalid, gid, -1)
    valid = loc >= 0
    li = loc[:, None]
    lj = loc[None, :]
    earlier = valid[None, :] & (lj == li) & (ei[None, :] < ei[:, None])
    rank = tl.sum(earlier.to(tl.int32), axis=1)
    first = valid & (rank == 0)
    blk = tl.sum((first[None, :] & (lj < li)).to(tl.int32), axis=1)
    nblk = tl.sum(first.to(tl.int32), axis=0)
    tl.store(sorted_ptr + blk * BS + rank, ei, mask=valid)
    tl.store(eids_ptr + blk, loc, mask=first)
    tl.store(npost_ptr, nblk * BS)


@triton.jit
def _round_bf16(x):
    """fp32 -> the fp32 value of its bf16 rounding (nearest even; a NaN keeps
    its sign and high payload bits, quieted). Explicit integer arithmetic: a
    ``.to(bfloat16).to(float32)`` round trip is not a reliable rounding point
    in Triton and would let the next add see the unrounded product."""
    u = x.to(tl.uint32, bitcast=True)
    r = (u + (((u >> 16) & 1) + 0x7FFF)) & 0xFFFF0000
    r = tl.where((u & 0x7FFFFFFF) > 0x7F800000, (u & 0xFFFF0000) | 0x00400000, r)
    return r.to(tl.float32, bitcast=True)


@triton.jit
def _post_kernel(
    moe_ptr,  # bf16 [M, TOPK, K] weighted GEMM2 rows
    ids_ptr,  # int32 [M, TOPK]
    map_ptr,  # int32 [E]
    sh_ptr,  # bf16 [M, K] shared-expert down projection
    gate_ptr,  # bf16 [M] shared-expert gate logits (or sigmoid, SIG_MODE 0)
    out_ptr,  # bf16 [M, K]
    K,
    TOPK: tl.constexpr,
    HAS_MAP: tl.constexpr,
    HAS_SHARED: tl.constexpr,
    SIG_MODE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    m = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    kmask = offs < K
    # Which of the top-k rows are local: all the id and map loads first, so
    # that they overlap instead of forming one dependent chain per k.
    gids = ()
    for k in tl.static_range(TOPK):
        gids += (tl.load(ids_ptr + m * TOPK + k),)  # type: ignore[assignment]
    takes = ()
    for k in tl.static_range(TOPK):
        take = gids[k] >= 0
        if HAS_MAP:
            take = take & (tl.load(map_ptr + tl.where(take, gids[k], 0)) >= 0)
        takes += (take,)  # type: ignore[assignment]
    # fp32 sum in k order. A skipped row is a masked-off load (no access)
    # that adds +0.0, which leaves the accumulator bit for bit: it starts at
    # +0.0 and therefore is never -0.0.
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for k in tl.static_range(TOPK):
        x = tl.load(
            moe_ptr + (m * TOPK + k) * K + offs, mask=kmask & takes[k], other=0.0
        )
        acc = acc + x.to(tl.float32)
    res = _round_bf16(acc)
    if HAS_SHARED:
        g = tl.load(gate_ptr + m).to(tl.float32)
        if SIG_MODE == 1:
            e = _round_bf16(libdevice.exp(-g))
            d = _round_bf16(1.0 + e)
            sig = _round_bf16(_recip_f32(d))
        elif SIG_MODE == 2:
            sig = _round_bf16(_recip_f32(1.0 + libdevice.exp(-g)))
        else:
            sig = g
        sh = tl.load(sh_ptr + m * K + offs, mask=kmask, other=0.0).to(tl.float32)
        res = _round_bf16(_round_bf16(sig * sh) + res)
    # res is bf16-representable: the conversion below is exact.
    tl.store(out_ptr + m * K + offs, res.to(tl.bfloat16), mask=kmask)


@triton.jit
def _round_bf16_hw(x):
    """Hardware round-to-nearest-even bf16 then upcast to f32 (prevents fusion)."""
    return tl.inline_asm_elementwise(
        asm="{ .reg .b16 tmp; cvt.rn.bf16.f32 tmp, $1; cvt.f32.bf16 $0, tmp; }",
        constraints="=r,r",
        args=[x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _rcp_rn_f32_hw(x):
    """Hardware IEEE round-to-nearest-even 1.0f / x (1 instruction on sm_8x)."""
    return tl.inline_asm_elementwise(
        asm="rcp.rn.f32 $0, $1;",
        constraints="=r,r",
        args=[x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _post_kernel_opt(
    moe_ptr,  # bf16 [M, TOPK, K]
    ids_ptr,  # int32 [M, TOPK]
    map_ptr,  # int32 [E]
    sh_ptr,  # bf16 [M, K]
    gate_ptr,  # bf16 [M]
    out_ptr,  # bf16 [M, K]
    K,
    TOPK: tl.constexpr,
    HAS_MAP: tl.constexpr,
    HAS_SHARED: tl.constexpr,
    SIG_MODE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    m = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    kmask = offs < K
    gids = ()
    for k in tl.static_range(TOPK):
        gids += (tl.load(ids_ptr + m * TOPK + k),)  # type: ignore[assignment]
    takes = ()
    for k in tl.static_range(TOPK):
        take = gids[k] >= 0
        if HAS_MAP:
            take = take & (tl.load(map_ptr + tl.where(take, gids[k], 0)) >= 0)
        takes += (take,)  # type: ignore[assignment]
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for k in tl.static_range(TOPK):
        x = tl.load(
            moe_ptr + (m * TOPK + k) * K + offs, mask=kmask & takes[k], other=0.0
        )
        acc = acc + x.to(tl.float32)
    res = _round_bf16_hw(acc)
    if HAS_SHARED:
        g = tl.load(gate_ptr + m).to(tl.float32)
        if SIG_MODE == 1:
            e = _round_bf16_hw(libdevice.exp(-g))
            d = _round_bf16_hw(1.0 + e)
            sig = _round_bf16_hw(_rcp_rn_f32_hw(d))
        elif SIG_MODE == 2:
            sig = _round_bf16_hw(_rcp_rn_f32_hw(1.0 + libdevice.exp(-g)))
        else:
            sig = g
        sh = tl.load(sh_ptr + m * K + offs, mask=kmask, other=0.0).to(tl.float32)
        res = _round_bf16_hw(sig * sh) + res
    tl.store(out_ptr + m * K + offs, res.to(tl.bfloat16), mask=kmask)


# (num_rows, top_k, num_experts, block_size_m)
_RouteKey = tuple[int, int, int, int]


class _RouteAlignFused:
    """Compile cache and CuTe-DSL warmup provider for CuteFusedRouteAlign."""

    def __init__(self) -> None:
        self._compiled: dict[_RouteKey, Any] = {}
        self._warmup: set[_RouteKey] = set()
        self._warmup_registered = False

    def _compile(self, key: _RouteKey) -> None:
        if key in self._compiled:
            return
        import cutlass.cute as cute
        from cutlass import BFloat16, Float32, Int32, Uint8
        from quack.compile_utils import make_fake_tensor

        from vllm.merlin._lebeche_moe_route_cute import CuteFusedRouteAlign
        from vllm.model_executor.kernels.linear.cute_dsl.skinny_gemm import (
            shape_dynamic_skinny_gemm,
        )

        m, top_k, e, bs = key
        gl = make_fake_tensor(BFloat16, (m, e), divisibility=8)
        gp = make_fake_tensor(Uint8, (cute.sym_int(),), divisibility=1)
        gm = make_fake_tensor(Int32, (cute.sym_int(),), divisibility=1)
        gw = make_fake_tensor(Float32, (m, top_k), divisibility=1)
        gi = make_fake_tensor(Int32, (m, top_k), divisibility=1)
        gs = make_fake_tensor(Int32, (m * top_k * bs,), divisibility=4)
        ge = make_fake_tensor(Int32, (m * top_k,), divisibility=1)
        gn = make_fake_tensor(Int32, (1,), divisibility=1)
        kern = CuteFusedRouteAlign(
            num_rows=m, top_k=top_k, num_experts=e, block_size_m=bs
        )
        self._compiled[key] = cute.compile(
            kern,
            gl,
            gp,
            gm,
            gw,
            gi,
            gs,
            ge,
            gn,
            Int32(0),
            Int32(1),
            Int32(1),
            shape_dynamic_skinny_gemm._stream(),
            options="--enable-tvm-ffi",
        )

    def request_warmup(self, keys: Iterable[_RouteKey]) -> None:
        self._warmup.update(keys)
        if self._warmup_registered or not self._warmup:
            return
        from vllm.model_executor.warmup.cutedsl_warmup import (
            register_cutedsl_warmup_provider,
        )

        register_cutedsl_warmup_provider(self)
        self._warmup_registered = True

    def get_cutedsl_warmup_compile_units(self):
        from vllm.model_executor.warmup.cutedsl_warmup import CuTeDSLCompileUnit

        return tuple(
            CuTeDSLCompileUnit(
                name=f"Lebeche MoE fused route+align M={k[0]} E={k[2]} K={k[1]}",
                key=("lebeche-moe-route-align", *k),
                compile=partial(self._compile, k),
            )
            for k in sorted(self._warmup)
        )

    def run(
        self,
        logits: torch.Tensor,
        expert_map: torch.Tensor | None,
        is_padding: torch.Tensor | None,
        top_k: int,
        renormalize: bool,
        w: torch.Tensor,
        ids: torch.Tensor,
        sorted_ids: torch.Tensor,
        eids: torch.Tensor,
        npost: torch.Tensor,
    ) -> None:
        from vllm.model_executor.kernels.linear.cute_dsl.skinny_gemm import (
            shape_dynamic_skinny_gemm,
        )

        m, e = logits.shape
        key: _RouteKey = (m, top_k, e, BLOCK_SIZE_M)
        if key not in self._compiled:
            self._compile(key)
        # Dynamic-1D gPad/gMap accept already-allocated tensors when unused (no alloc).
        pad_t = is_padding.view(torch.uint8) if is_padding is not None else npost.view(torch.uint8)
        map_t = expert_map if expert_map is not None else eids
        self._compiled[key](
            logits,
            pad_t,
            map_t,
            w,
            ids,
            sorted_ids,
            eids,
            npost,
            1 if is_padding is not None else 0,
            1 if expert_map is not None else 0,
            1 if renormalize else 0,
            shape_dynamic_skinny_gemm._stream(),
        )


route_align_fused = _RouteAlignFused()


def maybe_register_route_warmup(num_experts: int, top_k: int) -> None:
    """Register M=1..6 single-CTA route+align compile units for cutedsl_warmup."""
    if not route_fuse_active():
        return
    if num_experts != NUM_EXPERTS or not (1 <= top_k <= MAX_TOPK):
        return
    from vllm.model_executor.kernels.linear.cute_dsl.skinny_gemm import (
        shape_dynamic_skinny_gemm,
    )

    if not shape_dynamic_skinny_gemm.is_available():
        return
    keys = [
        (m, top_k, num_experts, BLOCK_SIZE_M)
        for m in range(1, ROUTE_FUSE_MAX_M + 1)
        if m * top_k <= ALIGN_MAX_ENTRIES
    ]
    route_align_fused.request_warmup(keys)


# -------------------------------------------------------------------- wrappers
def _pow2(n: int) -> int:
    return 1 << max(n - 1, 0).bit_length()


def route_align(
    logits: torch.Tensor,
    expert_map: torch.Tensor | None,
    is_padding: torch.Tensor | None,
    top_k: int,
    renormalize: bool,
    num_warps: int | None = None,
    lane_sum_reduce: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(topk_weights, topk_ids, sorted_token_ids, expert_ids,
    num_tokens_post_padded)`` for ``1 <= M <= MAX_M`` rows of bf16 logits: the
    shapes and bytes of ``fused_topk`` followed by ``moe_align_block_size(
    topk_ids, 8, E, expert_map, ignore_invalid_experts=True)`` (the entries of
    one block in ascending order, see the module docstring)."""
    m, e = logits.shape
    dev = logits.device
    n = m * top_k
    w = torch.empty((m, top_k), dtype=torch.float32, device=dev)
    ids = torch.empty((m, top_k), dtype=torch.int32, device=dev)
    sorted_ids = torch.empty((n * BLOCK_SIZE_M,), dtype=torch.int32, device=dev)
    eids = torch.empty((n,), dtype=torch.int32, device=dev)
    npost = torch.empty((1,), dtype=torch.int32, device=dev)
    CALLS["route_align"] += 1
    fused_align = n <= ALIGN_MAX_ENTRIES
    if (
        fused_align
        and route_fuse_active()
        and 1 <= m <= ROUTE_FUSE_MAX_M
        and e == NUM_EXPERTS
        and 1 <= top_k <= MAX_TOPK
        and num_warps is None
        and lane_sum_reduce is None
    ):
        from vllm.model_executor.kernels.linear.cute_dsl.skinny_gemm import (
            shape_dynamic_skinny_gemm,
        )

        if shape_dynamic_skinny_gemm.is_available():
            CALLS["route_align_cute"] += 1
            route_align_fused.run(
                logits,
                expert_map,
                is_padding,
                top_k,
                bool(renormalize),
                w,
                ids,
                sorted_ids,
                eids,
                npost,
            )
            return w, ids, sorted_ids, eids, npost
    lane_reduce = LANE_SUM_REDUCE if lane_sum_reduce is None else lane_sum_reduce
    _route_kernel[(m,)](
        logits,
        is_padding.view(torch.uint8) if is_padding is not None else None,
        w,
        ids,
        None if lane_reduce else torch.empty((m * 32,), dtype=torch.float32, device=dev),
        sorted_ids if fused_align else None,
        eids if fused_align else None,
        E=e,
        TOPK=top_k,
        BS=BLOCK_SIZE_M,
        FB=_pow2(top_k * BLOCK_SIZE_M),
        HAS_PAD=is_padding is not None,
        RENORM=bool(renormalize),
        FILL=fused_align,
        LANE_SUM_REDUCE=bool(lane_reduce),
        num_warps=1,
    )
    if not fused_align:
        # 7 or 8 rows: the pairwise alignment kernel is slower than the two
        # stock kernels there.
        from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
            moe_align_block_size,
        )

        return (w, ids) + moe_align_block_size(
            ids, BLOCK_SIZE_M, e, expert_map, ignore_invalid_experts=True
        )
    _align_kernel[(1,)](
        ids,
        expert_map,
        sorted_ids,
        eids,
        npost,
        n,
        NB=_pow2(n),
        E=e,
        BS=BLOCK_SIZE_M,
        HAS_MAP=expert_map is not None,
        num_warps=ALIGN_NUM_WARPS if num_warps is None else num_warps,
    )
    return w, ids, sorted_ids, eids, npost


def post_combine(
    moe_out: torch.Tensor,
    topk_ids: torch.Tensor,
    expert_map: torch.Tensor | None,
    shared: torch.Tensor | None,
    gate: torch.Tensor | None,
    out: torch.Tensor,
    block: int | None = None,
    num_warps: int | None = None,
    sig_mode: int | None = None,
) -> torch.Tensor:
    """``out = moe_sum(moe_out) [+ sigmoid(gate) * shared]`` with the stock
    bf16 roundings; ``moe_out`` is ``[M, top_k, K]`` and ``out`` ``[M, K]``."""
    m, top_k, k = moe_out.shape
    CALLS["post"] += 1
    if route_fuse_active():
        opt_block = POST_BLOCK_OPT if block is None else block
        CALLS["post_opt"] += 1
        _post_kernel_opt[(m, triton.cdiv(k, opt_block))](
            moe_out,
            topk_ids,
            expert_map,
            shared,
            gate if shared is not None else None,
            out,
            k,
            TOPK=top_k,
            HAS_MAP=expert_map is not None,
            HAS_SHARED=shared is not None,
            SIG_MODE=POST_SIG_MODE if sig_mode is None else sig_mode,
            BLOCK=opt_block,
            num_warps=POST_NUM_WARPS if num_warps is None else num_warps,
        )
        return out
    block = POST_BLOCK if block is None else block
    _post_kernel[(m, triton.cdiv(k, block))](
        moe_out,
        topk_ids,
        expert_map,
        shared,
        gate if shared is not None else None,
        out,
        k,
        TOPK=top_k,
        HAS_MAP=expert_map is not None,
        HAS_SHARED=shared is not None,
        SIG_MODE=POST_SIG_MODE if sig_mode is None else sig_mode,
        BLOCK=block,
        num_warps=POST_NUM_WARPS if num_warps is None else num_warps,
    )
    return out


# --------------------------------------------------------------- route + align
# (topk_ids, expert_map, num_experts, sorted_token_ids, expert_ids, npost)
_STASH: tuple | None = None


def _plain_i32(t: torch.Tensor, n: int, dev: torch.device) -> bool:
    return (
        t.dtype == torch.int32
        and t.dim() == 1
        and t.shape[0] == n
        and t.device == dev
        and t.is_contiguous()
    )


def route_core(
    router,
    router_logits: torch.Tensor,
    expert_map: torch.Tensor | None,
    topk_indices_dtype: torch.dtype | None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """``router.select_experts`` for a small bf16 batch plus the stashed
    alignment, or None when the stock path has to run."""
    global _STASH
    _STASH = None
    if router_logits.dim() != 2:
        return None
    m, e = router_logits.shape
    if not (1 <= m <= MAX_M) or e != NUM_EXPERTS:
        return None
    from vllm.model_executor.layers.fused_moe.router.fused_topk_router import (
        FusedTopKRouter,
        _get_padding_mask,
    )

    if (
        type(router) is not FusedTopKRouter
        or router.scoring_func != "softmax"
        or router.global_num_experts != e
        or not (1 <= router.top_k <= MAX_TOPK)
        or router.eplb_state is not None
        or router.capture_fn is not None
        or router._routing_replay_out is not None
        or topk_indices_dtype not in (None, torch.int32)
    ):
        return None
    if (
        router_logits.dtype != torch.bfloat16
        or not router_logits.is_cuda
        or not router_logits.is_contiguous()
    ):
        return None
    dev = router_logits.device
    if expert_map is not None and not _plain_i32(expert_map, e, dev):
        return None
    pad = _get_padding_mask(m)
    if pad is not None and not (
        pad.dtype == torch.bool
        and pad.dim() == 1
        and pad.shape[0] == m
        and pad.device == dev
        and pad.is_contiguous()
    ):
        return None
    w, ids, sorted_ids, eids, npost = route_align(
        router_logits, expert_map, pad, router.top_k, router.renormalize
    )
    _STASH = (ids, expert_map, e, sorted_ids, eids, npost)
    return w, ids


def route(
    runner, hidden_states: torch.Tensor, router_logits: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Hook of ``MoERunner._apply_quant_method`` (modular kernels)."""
    if not glue_active() or hidden_states.shape[0] != router_logits.shape[0]:
        return None
    import vllm.envs as envs

    if envs.VLLM_BATCH_INVARIANT or runner.enable_dbo:
        return None
    quant_method = runner._quant_method
    kernel = getattr(quant_method, "moe_kernel", None)
    if kernel is None:
        return None
    experts = kernel.fused_experts
    # Only the Marlin experts pick the stash up (fused_marlin_moe).
    if type(experts).__name__ != "MarlinExperts" or experts._lora_context is not None:
        return None
    return route_core(
        runner.router,
        router_logits,
        runner.routed_experts.expert_map,
        quant_method.topk_indices_dtype,
    )


def take_align(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    expert_map: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """The alignment :func:`route` computed for exactly this ``topk_ids``
    tensor, block size, expert count and expert map; None otherwise."""
    global _STASH
    stash = _STASH
    if stash is None:
        return None
    _STASH = None
    smap = stash[1]
    same_map = expert_map is smap or (
        expert_map is not None
        and smap is not None
        and expert_map.data_ptr() == smap.data_ptr()
        and expert_map.shape == smap.shape
        and expert_map.dtype == smap.dtype
    )
    if (
        stash[0] is not topk_ids
        or block_size != BLOCK_SIZE_M
        or num_experts != stash[2]
        or not same_map
    ):
        return None
    CALLS["take_align"] += 1
    return stash[3], stash[4], stash[5]


# ------------------------------------------------------------------------ post
class _Ctx:
    __slots__ = ("m", "sum", "gate")

    def __init__(self, m: int):
        self.m = m
        self.sum: tuple | None = None  # (moe_sum fn, moe_out, topk_ids, expert_map)
        self.gate: tuple | None = None  # (down output, gate logits)


_CTX: _Ctx | None = None


def defer_sum(
    moe_sum,
    moe_out: torch.Tensor,
    topk_ids: torch.Tensor,
    expert_map: torch.Tensor | None,
) -> bool:
    """``MarlinExperts``: leave the top-k reduction to the fused kernel."""
    ctx = _CTX
    if ctx is None or ctx.sum is not None:
        return False
    if (
        moe_out.dim() != 3
        or moe_out.shape[0] != ctx.m
        or topk_ids.shape != moe_out.shape[:2]
        or moe_out.shape[1] > MAX_TOPK
        or moe_out.dtype != torch.bfloat16
        or not moe_out.is_contiguous()
        or topk_ids.dtype != torch.int32
        or not topk_ids.is_contiguous()
        or topk_ids.device != moe_out.device
    ):
        return False
    # Without an expert map MarlinExperts.moe_sum is not pad-aware: stock.
    if expert_map is None or not (
        expert_map.dtype == torch.int32
        and expert_map.is_contiguous()
        and expert_map.device == moe_out.device
    ):
        return False
    ctx.sum = (moe_sum, moe_out, topk_ids, expert_map)
    return True


def defer_shared_gate(down: torch.Tensor, gate: torch.Tensor) -> bool:
    """Shared-expert MLP: leave ``sigmoid(gate) * down`` to the fused kernel."""
    ctx = _CTX
    if ctx is None or ctx.gate is not None:
        return False
    if (
        down.dim() != 2
        or down.shape[0] != ctx.m
        or down.dtype != torch.bfloat16
        or not down.is_contiguous()
        or gate.dtype != torch.bfloat16
        or gate.numel() != ctx.m
        or not gate.is_contiguous()
        or gate.device != down.device
    ):
        return False
    ctx.gate = (down, gate)
    return True


def forward_combined(
    layer,
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    shared_experts_input: torch.Tensor | None,
    input_ids: torch.Tensor | None,
) -> torch.Tensor:
    """``shared_output + fused_output`` of ``layer._forward_impl``: the body of
    ``vllm.moe_forward_shared_glue``. Above ``MAX_M`` rows (or whenever a piece
    cannot be fused) this is the stock arithmetic followed by torch's add."""
    global _CTX
    ctx = None
    m = hidden_states.shape[0]
    if (
        glue_active()
        and hidden_states.dim() == 2
        and 1 <= m <= MAX_M
        and hidden_states.dtype == torch.bfloat16
        and hidden_states.is_cuda
    ):
        ctx = _Ctx(m)
    _CTX = ctx
    try:
        shared_out, fused_out = layer._forward_impl(
            hidden_states, router_logits, shared_experts_input, input_ids
        )
    finally:
        _CTX = None
    if ctx is not None:
        dsum, dgate = ctx.sum, ctx.gate
        if (
            dsum is not None
            and dgate is not None
            and dgate[0] is shared_out
            and fused_out.shape == shared_out.shape
            and fused_out.shape == (m, dsum[1].shape[2])
            and fused_out.dtype == torch.bfloat16
            and fused_out.is_contiguous()
            and fused_out.device == shared_out.device == dsum[1].device
        ):
            return post_combine(
                dsum[1], dsum[2], dsum[3], shared_out, dgate[1], fused_out
            )
        if dsum is not None:
            dsum[0](dsum[1], fused_out, dsum[2], dsum[3])
        if dgate is not None:
            shared_out = torch.sigmoid(dgate[1]) * shared_out
    return shared_out + fused_out
