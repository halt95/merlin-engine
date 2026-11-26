# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 lebeche hill-climb (fork of Merlin 3.0.2.2)
"""Cooperative single-launch CuTe-DSL fused router GEMM + softmax + top-k + align for sm_8x.

Replaces the two launches (``router_gate`` ``CuteSkinnyGemm`` ``(512, 2560)`` +
``CuteFusedRouteAlign``) in ``MoERunner`` for small decode batches
(``M in {1, 2, 3, 4, 6, 8}``) with ONE cooperative kernel:

* **Phase 1 (all CTAs):** Grouped skinny BF16 GEMM (``groups_per_cta``
  independent ``group_bs``-thread groups per CTA, each computing
  ``opb_per_group`` rows of ``W_gate [512, 2560]`` with ``vector_width``).
  At ``M = 1..4`` ``(group_bs, vector_width)`` match
  ``QWEN4_EXP_SM86_GEMM_PLANS[(512, 2560)]`` so every BF16 router logit is
  bit-identical to commit ``f332186``. Logits are written to a tiny persistent
  L2-resident scratch buffer ``[M, 512]`` (4 KB at M=4). CTA 0 concurrently
  pre-fills ``sorted_token_ids`` and ``expert_ids`` with 128-bit stores.
* **Barrier:** Every CTA issues ``fence_acq_rel_gpu`` + ``atomic_add`` on a
  persistent zero-initialized self-resetting counter ``gCtr[0]`` (matching the
  pattern of ``CuteGdnVerifyNorm``). The last CTA resets ``gCtr[0] = 0``.
* **Phase 2 (last CTA only):** Acquires ``fence_acq_rel_gpu``, loads ``gLogits``
  with L2-coherent ``LoadCacheMode.GLOBAL`` (``ld.global.cg`` to bypass stale
  L1), and runs exact ``topkGating<16,512,4,16,32,int,bf16>`` softmax + top-k
  (with zero-loop top-2 fast path on first lane win) + EP map + block-size-8
  alignment (with warp-ballot ``popc`` block counting) across ``M`` warps.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cuda.bindings.driver import CUstream
from cutlass import BFloat16, Float32, Int32, Uint8, const_expr

from vllm.merlin._lebeche_moe_route_cute import (
    _f32_to_i32_bits,
    _ffs0,
    _i32_bits_to_f32,
    _nv_expf,
    _popc,
    _ptx_f32,
    _rcp_rn_f32,
    _redux_max_s32,
)

_PREFETCH_TILES = 2


class CuteFusedRouterRouteAlign:
    """Single-launch cooperative fused router GEMM + route + block-size-8 align."""

    def __init__(
        self,
        *,
        num_rows: int,
        group_bs: int,
        groups_per_cta: int,
        opb_per_group: int,
        vector_width: int,
        top_k: int = 10,
        num_experts: int = 512,
        block_size_m: int = 8,
        static_k: int = 2560,
    ) -> None:
        if not (1 <= num_rows <= block_size_m):
            raise ValueError(f"num_rows must be in 1..{block_size_m}, got {num_rows}")
        if num_experts != 512:
            raise ValueError(f"num_experts must be 512, got {num_experts}")
        if not (1 <= top_k <= 16):
            raise ValueError(f"top_k must be in 1..16, got {top_k}")
        if block_size_m % 4 != 0:
            raise ValueError(f"block_size_m must be a multiple of 4, got {block_size_m}")
        if group_bs % cute.arch.WARP_SIZE != 0:
            raise ValueError("group_bs must be a multiple of warp size")
        self.num_rows = num_rows
        self.group_bs = group_bs
        self.group_nw = group_bs // cute.arch.WARP_SIZE
        self.groups_per_cta = groups_per_cta
        self.opb_per_group = opb_per_group
        self.cta_outputs = groups_per_cta * opb_per_group
        if num_experts % self.cta_outputs != 0:
            raise ValueError("num_experts must be divisible by cta_outputs")
        self.num_ctas = num_experts // self.cta_outputs
        self.block_size = groups_per_cta * group_bs
        self.num_warps = self.block_size // cute.arch.WARP_SIZE
        if self.num_warps < num_rows:
            raise ValueError(
                f"block_size/32 ({self.num_warps}) must be >= num_rows ({num_rows})"
            )
        if static_k % (group_bs * vector_width) != 0:
            raise ValueError("static_k must be a multiple of group_bs * vector_width")
        if static_k < _PREFETCH_TILES * group_bs * vector_width:
            raise ValueError("static_k must contain at least two k-tiles")
        self.vector_width = vector_width
        self.top_k = top_k
        self.num_experts = num_experts
        self.block_size_m = block_size_m
        self.static_k = static_k

    @cute.jit
    def __call__(
        self,
        gA: cute.Tensor,
        gW_mat: cute.Tensor,
        gLogits: cute.Tensor,
        gCtr: cute.Tensor,
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
        copy_a = cute.make_copy_atom(
            cute.nvgpu.CopyG2ROp(),
            BFloat16,
            num_bits_per_copy=self.vector_width * 16,
            load_cache_mode=cute.nvgpu.LoadCacheMode.ALWAYS,
        )
        copy_b = cute.make_copy_atom(
            cute.nvgpu.CopyG2ROp(),
            BFloat16,
            num_bits_per_copy=self.vector_width * 16,
            load_cache_mode=cute.nvgpu.LoadCacheMode.STREAMING,
        )
        copy_b_pf = cute.make_copy_atom(
            cute.nvgpu.CopyG2ROp(),
            BFloat16,
            num_bits_per_copy=self.vector_width * 16,
            load_cache_mode=cute.nvgpu.LoadCacheMode.ALWAYS,
        )
        # L2-coherent 128-bit load (ld.global.cg) bypasses stale L1 in last CTA.
        copy_logits_cg = cute.make_copy_atom(
            cute.nvgpu.CopyG2ROp(),
            BFloat16,
            num_bits_per_copy=128,
            load_cache_mode=cute.nvgpu.LoadCacheMode.GLOBAL,
        )
        copy_i32x4 = cute.make_copy_atom(
            cute.nvgpu.CopyR2GOp(), Int32, num_bits_per_copy=128
        )
        smem_gemm = (
            0
            if self.group_nw == 1
            else (
                self.groups_per_cta
                * self.num_rows
                * self.opb_per_group
                * self.group_nw
                * 4
            )
        )
        n_ent = self.num_rows * self.top_k
        align_warps = (n_ent + 31) // 32
        smem_align = 0 if self.num_rows == 1 else (n_ent * 2 * 4 + align_warps * 4 + 32)
        smem_bytes = smem_gemm + smem_align + 64
        self.kernel(
            gA,
            gW_mat,
            gLogits,
            gCtr,
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
            copy_a,
            copy_b,
            copy_b_pf,
            copy_logits_cg,
            copy_i32x4,
        ).launch(
            grid=[self.num_ctas, 1, 1],
            block=[self.block_size, 1, 1],
            smem=smem_bytes,
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        gA: cute.Tensor,
        gW_mat: cute.Tensor,
        gLogits: cute.Tensor,
        gCtr: cute.Tensor,
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
        copy_a: cute.CopyAtom,
        copy_b: cute.CopyAtom,
        copy_b_pf: cute.CopyAtom,
        copy_logits_cg: cute.CopyAtom,
        copy_i32x4: cute.CopyAtom,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        block_idx, _, _ = cute.arch.block_idx()
        warp_idx = cute.arch.warp_idx()
        lane = cute.arch.lane_idx()

        M: cutlass.Constexpr = self.num_rows
        GBS: cutlass.Constexpr = self.group_bs
        GNW: cutlass.Constexpr = self.group_nw
        GPC: cutlass.Constexpr = self.groups_per_cta
        OPB_G: cutlass.Constexpr = self.opb_per_group
        CTA_OUT: cutlass.Constexpr = self.cta_outputs
        NCTAS: cutlass.Constexpr = self.num_ctas
        VW: cutlass.Constexpr = self.vector_width
        TK: cutlass.Constexpr = self.top_k
        NE: cutlass.Constexpr = self.num_experts
        BSM: cutlass.Constexpr = self.block_size_m
        N: cutlass.Constexpr = M * TK
        NUM_TILES: cutlass.Constexpr = self.static_k // (GBS * VW)
        ALIGN_WARPS: cutlass.Constexpr = (N + 31) // 32

        smem = cutlass.memory.SmemAllocator()
        s_flag = smem.allocate_tensor(
            Int32, cute.make_layout((1,), stride=(1,)), byte_alignment=16
        )
        if const_expr(M > 1):
            smem_loc = smem.allocate_tensor(
                Int32, cute.make_layout((N,), stride=(1,)), byte_alignment=16
            )
            smem_first = smem.allocate_tensor(
                Int32, cute.make_layout((N,), stride=(1,)), byte_alignment=16
            )
            smem_wcnt = smem.allocate_tensor(
                Int32, cute.make_layout((ALIGN_WARPS,), stride=(1,)), byte_alignment=16
            )
        if const_expr(GNW > 1):
            partials = smem.allocate_tensor(
                Float32,
                cute.make_layout(
                    (GPC, M, OPB_G, GNW),
                    stride=(M * OPB_G * GNW, OPB_G * GNW, GNW, 1),
                ),
                byte_alignment=16,
            )

        # CTA 0 overlaps 128-bit pre-fill of gSorted & gEids before barrier.
        if block_idx == Int32(0):
            gS_v4 = cute.logical_divide(gSorted, (4,))
            fill_r = cute.make_rmem_tensor(cute.make_layout((4,), stride=(1,)), Int32)
            fill_r.fill(Int32(N))
            n_v4: cutlass.Constexpr = (N * BSM) // 4
            fill_iters: cutlass.Constexpr = (n_v4 + self.block_size - 1) // self.block_size
            for fi in cutlass.range_constexpr(fill_iters):
                fidx = tidx + Int32(fi * self.block_size)
                if fidx < Int32(n_v4):
                    cute.copy(copy_i32x4, fill_r, gS_v4[((None, fidx),)])
            if tidx < Int32(N):
                gEids[tidx] = Int32(-1)

        # Phase 1: Single-step grouped exact skinny GEMM.
        grp_id = tidx // Int32(GBS)
        grp_tid = tidx % Int32(GBS)
        grp_warp = grp_tid // Int32(32)
        n_base = block_idx * Int32(CTA_OUT) + grp_id * Int32(OPB_G)

        acc = cute.make_rmem_tensor(
            cute.make_layout((M, OPB_G), stride=(OPB_G, 1)), Float32
        )
        acc.fill(0.0)

        gA_vec = cute.logical_divide(gA, (None, VW))
        gB_vec = cute.logical_divide(gW_mat, (None, VW))
        tA_all = cute.logical_divide(gA_vec, (None, (None, GBS)))
        tB_all = cute.logical_divide(gB_vec, (None, (None, GBS)))
        tA = tA_all[None, (None, (grp_tid, None))]

        a_regs = cute.make_rmem_tensor(
            cute.make_layout((M, VW), stride=(VW, 1)), BFloat16
        )
        b_regs = cute.make_rmem_tensor(
            cute.make_layout((OPB_G, VW), stride=(VW, 1)), BFloat16
        )
        pf_b = cute.make_rmem_tensor(
            cute.make_layout(
                (_PREFETCH_TILES, OPB_G, VW), stride=(OPB_G * VW, VW, 1)
            ),
            BFloat16,
        )
        for kt in cutlass.range_constexpr(_PREFETCH_TILES):
            for ni in cutlass.range_constexpr(OPB_G):
                tB_s = tB_all[n_base + ni, (None, (grp_tid, None))]
                cute.copy(copy_b_pf, tB_s[None, kt], pf_b[kt, ni, None])

        for kt in cutlass.range_constexpr(NUM_TILES):
            if const_expr(kt < _PREFETCH_TILES):
                b_f32 = pf_b[kt, None, None].load().to(Float32)
            else:
                for ni in cutlass.range_constexpr(OPB_G):
                    tB_s = tB_all[n_base + ni, (None, (grp_tid, None))]
                    cute.copy(copy_b, tB_s[None, kt], b_regs[ni, None])
                b_f32 = b_regs.load().to(Float32)
            for mi in cutlass.range_constexpr(M):
                cute.copy(copy_a, tA[mi, None, kt], a_regs[mi, None])
            a_f32 = a_regs.load().to(Float32)
            for vi in cutlass.range_constexpr(VW):
                for mi in cutlass.range_constexpr(M):
                    for ni in cutlass.range_constexpr(OPB_G):
                        acc[mi, ni] += a_f32[mi, vi] * b_f32[ni, vi]

        for mi in cutlass.range_constexpr(M):
            for ni in cutlass.range_constexpr(OPB_G):
                acc[mi, ni] = cute.arch.warp_reduction_sum(acc[mi, ni])

        if const_expr(GNW == 1):
            if lane == Int32(0):
                for mi in cutlass.range_constexpr(M):
                    for ni in cutlass.range_constexpr(OPB_G):
                        gLogits[mi, n_base + ni] = Float32(acc[mi, ni]).to(BFloat16)
        else:
            if lane == Int32(0):
                for mi in cutlass.range_constexpr(M):
                    for ni in cutlass.range_constexpr(OPB_G):
                        partials[grp_id, mi, ni, grp_warp] = acc[mi, ni]
            cute.arch.sync_threads()
            if grp_tid == Int32(0):
                for mi in cutlass.range_constexpr(M):
                    for ni in cutlass.range_constexpr(OPB_G):
                        total = (
                            partials[grp_id, mi, ni, None]
                            .load()
                            .reduce(
                                cute.ReductionOp.ADD,
                                init_val=Float32(0.0),
                                reduction_profile=0,
                            )
                        )
                        gLogits[mi, n_base + ni] = Float32(total).to(BFloat16)

        # Cooperative self-resetting GPU barrier.
        cute.arch.sync_threads()
        if tidx == Int32(0):
            cute.arch.fence_acq_rel_gpu()
            old = cute.arch.atomic_add(gCtr.iterator, Int32(1))
            is_last = Int32(1) if old == Int32(NCTAS - 1) else Int32(0)
            if old == Int32(NCTAS - 1):
                gCtr[0] = Int32(0)
            s_flag[0] = is_last
        cute.arch.sync_threads()

        # Phase 2: Last CTA runs exact softmax + top-k + EP map + align.
        if s_flag[0] != Int32(0):
            cute.arch.fence_acq_rel_gpu()
            if warp_idx < Int32(M):
                m = warp_idx
                gL_vec = cute.logical_divide(gLogits, (None, 8))
                r0 = cute.make_rmem_tensor(cute.make_layout((8,), stride=(1,)), BFloat16)
                r1 = cute.make_rmem_tensor(cute.make_layout((8,), stride=(1,)), BFloat16)
                cute.copy(copy_logits_cg, gL_vec[m, (None, lane)], r0)
                cute.copy(copy_logits_cg, gL_vec[m, (None, lane + 32)], r1)

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

                for v in cutlass.range_constexpr(16):
                    xs[v] = _nv_expf(xs[v] - mx)
                s = xs[0]
                for v in cutlass.range_constexpr(1, 16):
                    s = s + xs[v]
                s = s + cute.arch.shuffle_sync_bfly(s, 16)
                s = s + cute.arch.shuffle_sync_bfly(s, 8)
                s = s + cute.arch.shuffle_sync_bfly(s, 4)
                s = s + cute.arch.shuffle_sync_bfly(s, 2)
                s = s + cute.arch.shuffle_sync_bfly(s, 1)
                recip = _rcp_rn_f32(s)

                keys = cute.make_rmem_tensor(cute.make_layout((16,), stride=(1,)), Int32)
                lbest = Int32(-1)
                lcol = Int32(NE)
                lbest2 = Int32(-1)
                lcol2 = Int32(NE)
                for u in cutlass.range_constexpr(16):
                    v: cutlass.Constexpr = 15 - u
                    pbits = _f32_to_i32_bits(xs[v] * recip)
                    is_bad = (pbits & Int32(0x7F800000)) == Int32(0x7F800000)
                    kv = Int32(0) if is_bad else pbits
                    keys[v] = kv
                    col_v = lane * Int32(8) + Int32((v // 8) * 256 + (v % 8))
                    if kv >= lbest:
                        lbest2 = lbest
                        lcol2 = lcol
                        lbest = kv
                        lcol = col_v
                    elif kv >= lbest2:
                        lbest2 = kv
                        lcol2 = col_v

                prev_win_col = Int32(-1)
                has_second = Int32(1)
                ssum = Float32(0.0)
                my_w = Float32(0.0)
                my_id = Int32(-1)
                for k in cutlass.range_constexpr(TK):
                    best = _redux_max_s32(lbest)
                    eq = lbest == best
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
                    if const_expr(k < TK - 1):
                        if lane == win_lane:
                            if has_second != Int32(0):
                                prev_win_col = cbest
                                lbest = lbest2
                                lcol = lcol2
                                has_second = Int32(0)
                            else:
                                nlbest = Int32(-1)
                                nlcol = Int32(NE)
                                for u in cutlass.range_constexpr(16):
                                    v: cutlass.Constexpr = 15 - u
                                    col_v = lane * Int32(8) + Int32(
                                        (v // 8) * 256 + (v % 8)
                                    )
                                    kv = (
                                        Int32(-1)
                                        if (col_v == cbest or col_v == prev_win_col)
                                        else keys[v]
                                    )
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
                if lane < Int32(TK):
                    idk = Int32(-1) if is_pad else my_id
                    gW[m, lane] = my_w * scale
                    gIds[m, lane] = idk
                    if idk >= Int32(0) and idk < Int32(NE):
                        if has_map != Int32(0):
                            loc = gMap[idk]
                        else:
                            loc = idk

                if const_expr(M == 1):
                    cute.arch.sync_warp()
                    valid_mask = cute.arch.vote_ballot_sync(
                        (lane < Int32(TK)) and (loc >= Int32(0))
                    )
                    nblk = _popc(valid_mask)
                    blk = Int32(0)
                    for j in cutlass.range_constexpr(TK):
                        lj = cute.arch.shuffle_sync(loc, Int32(j))
                        if lj >= Int32(0) and lj < loc:
                            blk = blk + Int32(1)
                    if lane < Int32(TK) and loc >= Int32(0):
                        gSorted[blk * Int32(BSM)] = lane
                        gEids[blk] = loc
                    if lane == Int32(0):
                        gNpost[0] = nblk * Int32(BSM)
                else:
                    if lane < Int32(TK):
                        smem_loc[m * Int32(TK) + lane] = loc

            if const_expr(M > 1):
                cute.arch.sync_threads()
                li = Int32(-1)
                rank = Int32(0)
                is_first = Int32(0)
                if tidx < Int32(N):
                    li = smem_loc[tidx]
                    if li >= Int32(0):
                        row_start = (tidx // Int32(TK)) * Int32(TK)
                        for ej in cutlass.range_constexpr(N - TK):
                            if Int32(ej) < row_start and smem_loc[ej] == li:
                                rank = rank + Int32(1)
                        is_first = Int32(1) if rank == Int32(0) else Int32(0)
                    smem_first[tidx] = is_first
                if warp_idx < Int32(ALIGN_WARPS):
                    wmask = cute.arch.vote_ballot_sync(is_first != Int32(0))
                    if lane == Int32(0):
                        smem_wcnt[warp_idx] = _popc(wmask)
                cute.arch.sync_threads()
                if tidx < Int32(N):
                    if li >= Int32(0):
                        blk = Int32(0)
                        for ej in cutlass.range_constexpr(N):
                            if smem_first[ej] != Int32(0) and smem_loc[ej] < li:
                                blk = blk + Int32(1)
                        gSorted[blk * Int32(BSM) + rank] = tidx
                        if is_first != Int32(0):
                            gEids[blk] = li
                if tidx == Int32(0):
                    nblk = Int32(0)
                    for wi in cutlass.range_constexpr(ALIGN_WARPS):
                        nblk = nblk + smem_wcnt[wi]
                    gNpost[0] = nblk * Int32(BSM)
