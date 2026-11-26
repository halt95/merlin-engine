# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 lebeche hill-climb (fork of Merlin 3.0.2.2)
"""Single-CTA CuTe-DSL fused small-M MoE router + block-size-8 alignment for sm_8x.

Replaces the two Triton launches (``_route_kernel`` + ``_align_kernel``) in
``vllm/merlin/lebeche_moe_glue.py`` for ``1 <= M <= 6`` with ONE single-CTA
kernel (``M`` warps = ``M * 32`` threads):

* Each warp ``m`` loads row ``m`` of ``logits`` (512 BF16) with two 128-bit
  vectorized loads while all threads overlap 128-bit pre-fill of ``sorted_ids``
  and ``eids``.
* Exact stock ``topkGating<16,512,4,16,32,int,bf16>`` arithmetic: 10-instruction
  PTX ``__nv_expf``, lane-order sum + 5-step ``shuffle_sync_bfly`` (xor 16..1),
  IEEE ``rcp.rn.f32`` reciprocal, hardware ``redux.sync.max.s32`` +
  ``vote_ballot_sync`` + ``brev``/``clz`` lowest-index tie-broken top-k, and
  pick-order renormalization.
* M == 1 runs a zero-smem, zero-barrier warp-only alignment (all top-k picks
  of one row are distinct). M = 2..6 align inside the same CTA via shared
  memory with same-row duplicate skipping.
* ``has_pad``, ``has_map`` and ``renormalize`` are runtime flags, so only ONE
  compile unit per M (6 units total for Flash-Next) is needed in
  ``cutedsl_warmup``.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cuda.bindings.driver import CUstream
from cutlass import BFloat16, Float32, Int32, Uint8, const_expr


def _ptx_f32(code: str, *args):
    return cute.arch.inline_ptx(
        code, write_only_types=[Float32], read_only_args=list(args)
    )


def _ptx_i32(code: str, *args):
    return cute.arch.inline_ptx(
        code, write_only_types=[Int32], read_only_args=list(args)
    )


def _nv_expf(x):
    # Exact sm_86 PTX sequence of libdevice.exp (__nv_expf).
    r5 = _ptx_f32("fma.rn.ftz.f32 {$w0}, {$r0}, 0f3BBB989D, 0f3F000000;", x)
    r6 = _ptx_f32("cvt.ftz.sat.f32.f32 {$w0}, {$r0};", r5)
    r9 = _ptx_f32("fma.rm.ftz.f32 {$w0}, {$r0}, 0f437C0000, 0f4B400001;", r6)
    r10 = _ptx_f32("add.f32 {$w0}, {$r0}, 0fCB40007F;", r9)
    r11 = _ptx_f32("neg.f32 {$w0}, {$r0};", r10)
    r13 = _ptx_f32("fma.rn.ftz.f32 {$w0}, {$r0}, 0f3FB8AA3B, {$r1};", x, r11)
    r15 = _ptx_f32("fma.rn.ftz.f32 {$w0}, {$r0}, 0f32A57060, {$r1};", x, r13)
    r16 = _ptx_f32("shl.b32 {$w0}, {$r0}, 23;", r9)
    r17 = _ptx_f32("ex2.approx.ftz.f32 {$w0}, {$r0};", r15)
    return _ptx_f32("mul.f32 {$w0}, {$r0}, {$r1};", r17, r16)


def _rcp_rn_f32(x):
    return _ptx_f32("rcp.rn.f32 {$w0}, {$r0};", x)


def _f32_to_i32_bits(x):
    return _ptx_i32("mov.b32 {$w0}, {$r0};", x)


def _i32_bits_to_f32(x):
    return _ptx_f32("mov.b32 {$w0}, {$r0};", x)


def _redux_max_s32(x):
    return _ptx_i32("redux.sync.max.s32 {$w0}, {$r0}, 0xffffffff;", x)


def _ffs0(mask):
    # 0-based index of lowest set bit (mask != 0).
    return _ptx_i32(
        "{\n.reg .b32 tmp;\nbrev.b32 tmp, {$r0};\nclz.b32 {$w0}, tmp;\n}", mask
    )


def _popc(mask):
    return _ptx_i32("popc.b32 {$w0}, {$r0};", mask)


class CuteFusedRouteAlign:
    def __init__(
        self,
        *,
        num_rows: int,
        top_k: int = 10,
        num_experts: int = 512,
        block_size_m: int = 8,
    ) -> None:
        if not (1 <= num_rows <= 6):
            raise ValueError(f"num_rows must be in 1..6, got {num_rows}")
        if num_experts != 512:
            raise ValueError(f"num_experts must be 512, got {num_experts}")
        if not (1 <= top_k <= 16):
            raise ValueError(f"top_k must be in 1..16, got {top_k}")
        if block_size_m % 4 != 0:
            raise ValueError(f"block_size_m must be a multiple of 4, got {block_size_m}")
        self.num_rows = num_rows
        self.top_k = top_k
        self.num_experts = num_experts
        self.block_size_m = block_size_m
        self.block_threads = num_rows * 32

    @cute.jit
    def __call__(
        self,
        gLogits: cute.Tensor,
        gPad: cute.Tensor,
        gMap: cute.Tensor,
        gW: cute.Tensor,
        gIds: cute.Tensor,
        gSorted: cute.Tensor,
        gEids: cute.Tensor,
        gNpost: cute.Tensor,
        has_pad: Int32,
        has_map: Int32,
        renormalize: Int32,
        stream: CUstream,
    ) -> None:
        copy_vec = cute.make_copy_atom(
            cute.nvgpu.CopyG2ROp(),
            BFloat16,
            num_bits_per_copy=128,
            load_cache_mode=cute.nvgpu.LoadCacheMode.ALWAYS,
        )
        copy_i32x4 = cute.make_copy_atom(
            cute.nvgpu.CopyR2GOp(), Int32, num_bits_per_copy=128
        )
        n_ent = self.num_rows * self.top_k
        smem_bytes = 0 if self.num_rows == 1 else (n_ent * 2 * 4 + 32)
        self.kernel(
            gLogits,
            gPad,
            gMap,
            gW,
            gIds,
            gSorted,
            gEids,
            gNpost,
            has_pad,
            has_map,
            renormalize,
            copy_vec,
            copy_i32x4,
        ).launch(
            grid=[1, 1, 1],
            block=[self.block_threads, 1, 1],
            smem=smem_bytes,
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        gLogits: cute.Tensor,
        gPad: cute.Tensor,
        gMap: cute.Tensor,
        gW: cute.Tensor,
        gIds: cute.Tensor,
        gSorted: cute.Tensor,
        gEids: cute.Tensor,
        gNpost: cute.Tensor,
        has_pad: Int32,
        has_map: Int32,
        renormalize: Int32,
        copy_vec: cute.CopyAtom,
        copy_i32x4: cute.CopyAtom,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        m = cute.arch.warp_idx()
        lane = cute.arch.lane_idx()

        M: cutlass.Constexpr = self.num_rows
        TOPK: cutlass.Constexpr = self.top_k
        E: cutlass.Constexpr = self.num_experts
        BS: cutlass.Constexpr = self.block_size_m
        N: cutlass.Constexpr = M * TOPK

        gL_vec = cute.logical_divide(gLogits, (None, 8))
        r0 = cute.make_rmem_tensor(cute.make_layout((8,), stride=(1,)), BFloat16)
        r1 = cute.make_rmem_tensor(cute.make_layout((8,), stride=(1,)), BFloat16)
        cute.copy(copy_vec, gL_vec[m, (None, lane)], r0)
        cute.copy(copy_vec, gL_vec[m, (None, lane + 32)], r1)

        # Overlap 128-bit vectorized pre-fill of gSorted + gEids while logits load.
        gS_v4 = cute.logical_divide(gSorted, (4,))
        fill_r = cute.make_rmem_tensor(cute.make_layout((4,), stride=(1,)), Int32)
        fill_r.fill(Int32(N))
        n_v4: cutlass.Constexpr = (N * BS) // 4
        if tidx < Int32(n_v4):
            cute.copy(copy_i32x4, fill_r, gS_v4[((None, tidx),)])
        if tidx < Int32(N):
            gEids[tidx] = Int32(-1)

        xs = cute.make_rmem_tensor(cute.make_layout((16,), stride=(1,)), Float32)
        for v in cutlass.range_constexpr(8):
            xs[v] = r0[v].to(Float32)
            xs[v + 8] = r1[v].to(Float32)

        lmax = xs[0]
        for v in cutlass.range_constexpr(1, 16):
            lmax = _ptx_f32("max.f32 {$w0}, {$r0}, {$r1};", lmax, xs[v])
        mx = lmax
        for off in cutlass.range_constexpr(5):
            sh = 16 >> off
            other = cute.arch.shuffle_sync_bfly(mx, sh)
            mx = _ptx_f32("max.f32 {$w0}, {$r0}, {$r1};", mx, other)

        # Row sum in stock lane order + butterfly (xor 16, 8, 4, 2, 1).
        es = cute.make_rmem_tensor(cute.make_layout((16,), stride=(1,)), Float32)
        for v in cutlass.range_constexpr(16):
            es[v] = _nv_expf(xs[v] - mx)
        s = es[0]
        for v in cutlass.range_constexpr(1, 16):
            s = s + es[v]
        s = s + cute.arch.shuffle_sync_bfly(s, 16)
        s = s + cute.arch.shuffle_sync_bfly(s, 8)
        s = s + cute.arch.shuffle_sync_bfly(s, 4)
        s = s + cute.arch.shuffle_sync_bfly(s, 2)
        s = s + cute.arch.shuffle_sync_bfly(s, 1)
        recip = _rcp_rn_f32(s)

        # NaN/Inf probabilities map to 0, like stock topkGating.
        keys = cute.make_rmem_tensor(cute.make_layout((16,), stride=(1,)), Int32)
        for v in cutlass.range_constexpr(16):
            pbits = _f32_to_i32_bits(es[v] * recip)
            is_bad = (pbits & Int32(0x7F800000)) == Int32(0x7F800000)
            keys[v] = Int32(0) if is_bad else pbits

        lbest = Int32(-1)
        lcol = Int32(E)
        for u in cutlass.range_constexpr(16):
            v: cutlass.Constexpr = 15 - u
            kv = keys[v]
            col_v = lane * Int32(8) + Int32((v // 8) * 256 + (v % 8))
            if kv >= lbest:
                lbest = kv
                lcol = col_v

        ssum = Float32(0.0)
        my_w = Float32(0.0)
        my_id = Int32(-1)

        for k in cutlass.range_constexpr(TOPK):
            best = _redux_max_s32(lbest)
            eq = lbest == best
            # First-half columns (0..255) strictly precede second-half (256..511),
            # and within each half column order is strictly monotonic in lane.
            m1 = cute.arch.vote_ballot_sync(eq and (lcol < Int32(256)))
            m2 = cute.arch.vote_ballot_sync(eq)
            wmask = m1 if m1 != Int32(0) else m2
            win_lane = _ffs0(wmask)
            cbest = cute.arch.shuffle_sync(lcol, win_lane)
            best_f = _i32_bits_to_f32(best)
            ssum = ssum + best_f
            if lane == Int32(k):
                my_w = best_f
                my_id = cbest
            if const_expr(k < TOPK - 1):
                if lane == win_lane:
                    nlbest = Int32(-1)
                    nlcol = Int32(E)
                    for u in cutlass.range_constexpr(16):
                        v: cutlass.Constexpr = 15 - u
                        col_v = lane * Int32(8) + Int32((v // 8) * 256 + (v % 8))
                        kv = Int32(-1) if col_v == cbest else keys[v]
                        keys[v] = kv
                        if kv >= nlbest:
                            nlbest = kv
                            nlcol = col_v
                    lbest = nlbest
                    lcol = nlcol

        scale = Float32(1.0)
        if renormalize != Int32(0):
            denom = ssum if ssum > Float32(0.0) else Float32(1.0)
            scale = _rcp_rn_f32(denom)

        is_pad = False
        if has_pad != Int32(0):
            is_pad = gPad[m] != Uint8(0)

        loc = Int32(-1)
        if lane < Int32(TOPK):
            idk = Int32(-1) if is_pad else my_id
            gW[m, lane] = my_w * scale
            gIds[m, lane] = idk
            if idk >= Int32(0) and idk < Int32(E):
                if has_map != Int32(0):
                    loc = gMap[idk]
                else:
                    loc = idk

        if const_expr(M == 1):
            # Single row: all TOPK picks are distinct, so zero smem & zero CTA barriers.
            cute.arch.sync_warp()
            valid_mask = cute.arch.vote_ballot_sync(
                (lane < Int32(TOPK)) and (loc >= Int32(0))
            )
            nblk = _popc(valid_mask)
            blk = Int32(0)
            for j in cutlass.range_constexpr(TOPK):
                lj = cute.arch.shuffle_sync(loc, Int32(j))
                if lj >= Int32(0) and lj < loc:
                    blk = blk + Int32(1)
            if lane < Int32(TOPK) and loc >= Int32(0):
                gSorted[blk * Int32(BS)] = lane
                gEids[blk] = loc
            if lane == Int32(0):
                gNpost[0] = nblk * Int32(BS)
        else:
            smem = cutlass.memory.SmemAllocator()
            smem_loc = smem.allocate_tensor(
                Int32, cute.make_layout((N,), stride=(1,)), byte_alignment=16
            )
            smem_first = smem.allocate_tensor(
                Int32, cute.make_layout((N,), stride=(1,)), byte_alignment=16
            )
            if lane < Int32(TOPK):
                smem_loc[m * Int32(TOPK) + lane] = loc
            cute.arch.sync_threads()

            li = Int32(-1)
            rank = Int32(0)
            is_first = Int32(0)
            if tidx < Int32(N):
                li = smem_loc[tidx]
                if li >= Int32(0):
                    # Picks within one row are distinct: only earlier rows can duplicate.
                    row_start = (tidx // Int32(TOPK)) * Int32(TOPK)
                    for ej in cutlass.range_constexpr(N - TOPK):
                        if Int32(ej) < row_start and smem_loc[ej] == li:
                            rank = rank + Int32(1)
                    is_first = Int32(1) if rank == Int32(0) else Int32(0)
                smem_first[tidx] = is_first
            cute.arch.sync_threads()

            if tidx < Int32(N):
                if li >= Int32(0):
                    blk = Int32(0)
                    for ej in cutlass.range_constexpr(N):
                        if smem_first[ej] != Int32(0) and smem_loc[ej] < li:
                            blk = blk + Int32(1)
                    gSorted[blk * Int32(BS) + rank] = tidx
                    if is_first != Int32(0):
                        gEids[blk] = li
            if tidx == Int32(0):
                nblk = Int32(0)
                for ej in cutlass.range_constexpr(N):
                    nblk = nblk + smem_first[ej]
                gNpost[0] = nblk * Int32(BS)
