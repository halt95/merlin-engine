# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 lebeche hill-climb (fork of Merlin 3.0.2.2)
"""Direct-register Tensor Core small-M W8 channelwise GEMM over Marlin weights.

Reads Marlin-repacked ``uint8b128`` weights (``b_q_weight`` of shape
``[K // 16, N * 4]``, ``int32``) and Marlin-permuted channelwise BF16 scales
(``b_scales`` of shape ``[N]``) in-place without duplicating weights.

Each warp loads 16x64 (or 16x32) Marlin tiles directly into registers with
128-bit streaming loads (``LoadCacheMode.STREAMING``), dequantizes uint8b128
into BF16x2 via inline PTX ``prmt.b32`` + ``add.f32`` (matching Marlin's
``dequant.h``), and executes transposed ``mma.sync.aligned.m16n8k16`` so one
instruction accumulates all ``M <= 8`` batch rows against 16 output columns.
No shared memory staging of A/B, no ``cp.async``, no ``elect.sync`` (sm_8x
compatible), and no split-K global workspace or atomics.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cuda.bindings.driver import CUstream
from cutlass import const_expr


def _dequant_u8b128_to_bf16x2_pair(q_i32):
    """Dequantize 4 Marlin-packed uint8b128 bytes into 2 BF16x2 (int32) words.

    Marlin packs bytes ``[b0, b1, b2, b3]`` for ``k = tc_row + {0, 8, 1, 9}``.
    Returns ``(bf16x2(b0-128, b2-128), bf16x2(b1-128, b3-128))``, matching
    ``csrc/quantization/marlin/dequant.h`` (``nv_bfloat162, kU8B128``).
    """
    return cute.arch.inline_ptx(
        "{"
        " .reg .b32 t0, t1, t2, t3;"
        " .reg .f32 f0, f1, f2, f3;"
        " prmt.b32 t0, {$r0}, 0x4B000000, 0x7650;"
        " prmt.b32 t1, {$r0}, 0x4B000000, 0x7652;"
        " prmt.b32 t2, {$r0}, 0x4B000000, 0x7651;"
        " prmt.b32 t3, {$r0}, 0x4B000000, 0x7653;"
        " mov.b32 f0, t0;"
        " mov.b32 f1, t1;"
        " mov.b32 f2, t2;"
        " mov.b32 f3, t3;"
        " add.f32 f0, f0, 0fCB000080;"
        " add.f32 f1, f1, 0fCB000080;"
        " add.f32 f2, f2, 0fCB000080;"
        " add.f32 f3, f3, 0fCB000080;"
        " mov.b32 t0, f0;"
        " mov.b32 t1, f1;"
        " mov.b32 t2, f2;"
        " mov.b32 t3, f3;"
        " prmt.b32 {$w0}, t0, t1, 0x7632;"
        " prmt.b32 {$w1}, t2, t3, 0x7632;"
        "}",
        write_only_types=[cutlass.Int32, cutlass.Int32],
        read_only_args=[q_i32],
    )


def _mma_m16n8k16_bf16(a0, a1, a2, a3, b0, b1, c0, c1, c2, c3):
    return cute.arch.inline_ptx(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{ {$w0}, {$w1}, {$w2}, {$w3} }, "
        "{ {$r0}, {$r1}, {$r2}, {$r3} }, "
        "{ {$r4}, {$r5} }, "
        "{ {$r6}, {$r7}, {$r8}, {$r9} };",
        write_only_types=[
            cutlass.Float32,
            cutlass.Float32,
            cutlass.Float32,
            cutlass.Float32,
        ],
        read_only_args=[a0, a1, a2, a3, b0, b1, c0, c1, c2, c3],
    )


class CuteW8MarlinGemm:
    """Dynamic-M (1 <= M <= max_m <= 8) W8 channelwise GEMM on sm_8x."""

    def __init__(
        self,
        *,
        max_m: int,
        cols_per_block: int = 64,
        block_size: int = 256,
        k_unroll: int = 2,
        static_k: int | None = None,
        pingpong: bool = True,
        contig_k: bool = False,
    ) -> None:
        if not (1 <= max_m <= 8):
            raise ValueError(f"max_m must be in [1, 8], got {max_m}")
        if cols_per_block not in (32, 64):
            raise ValueError(f"cols_per_block must be 32 or 64, got {cols_per_block}")
        if block_size % cute.arch.WARP_SIZE != 0 or block_size < 64:
            raise ValueError(
                f"block_size must be a multiple of 32 and >= 64, got {block_size}"
            )
        num_warps = block_size // cute.arch.WARP_SIZE
        if static_k is not None:
            if static_k % (16 * num_warps) != 0:
                raise ValueError(
                    f"static_k={static_k} must be divisible by 16 * num_warps={16 * num_warps}"
                )
        smem_w_stride = num_warps | 1
        smem_bytes = max_m * cols_per_block * smem_w_stride * 4
        if smem_bytes > 48 * 1024:
            raise ValueError(f"smem {smem_bytes} exceeds 48 KiB default limit")

        self.max_m = max_m
        self.cols_per_block = cols_per_block
        self.n_mmas = cols_per_block // 16
        self.n_v4 = cols_per_block // 32
        self.block_size = block_size
        self.num_warps = num_warps
        self.k_unroll = k_unroll
        self.static_k = static_k
        self.pingpong = pingpong
        self.contig_k = contig_k
        self.smem_w_stride = smem_w_stride
        self.smem_bytes = smem_bytes

    @cute.jit
    def __call__(
        self,
        gA_u32: cute.Tensor,  # [M, K // 2] Int32 (BF16 reinterpreted as bf16x2)
        gB_i32: cute.Tensor,  # [K // 16, N * 4] Int32 (Marlin repacked W8)
        gS: cute.Tensor,  # [N] BFloat16 (Marlin permuted channelwise scales)
        gC: cute.Tensor,  # [M, N] BFloat16
        stream: CUstream,
    ) -> None:
        m = cute.size(gC, mode=[0])
        n = cute.size(gC, mode=[1])
        k_tiles = cute.size(gB_i32, mode=[0])
        copy_b = cute.make_copy_atom(
            cute.nvgpu.CopyG2ROp(),
            cutlass.Int32,
            num_bits_per_copy=128,
            load_cache_mode=cute.nvgpu.LoadCacheMode.STREAMING,
        )
        self.kernel(gA_u32, gB_i32, gS, gC, m, k_tiles, copy_b).launch(
            grid=[n // self.cols_per_block, 1, 1],
            block=[self.block_size, 1, 1],
            smem=self.smem_bytes,
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        gA_u32: cute.Tensor,
        gB_i32: cute.Tensor,
        gS: cute.Tensor,
        gC: cute.Tensor,
        m_extent: cutlass.Int32,
        k_tiles: cutlass.Int32,
        copy_b: cute.CopyAtom,
    ) -> None:
        max_m = const_expr(self.max_m)
        cols_per_block = const_expr(self.cols_per_block)
        n_mmas = const_expr(self.n_mmas)
        n_v4 = const_expr(self.n_v4)
        num_warps = const_expr(self.num_warps)
        w_stride = const_expr(self.smem_w_stride)

        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        warp_id = tidx >> 5
        lane = tidx & 31
        group_id = lane >> 2
        tid_in_g = lane & 3

        gB_v4 = cute.zipped_divide(gB_i32, (1, 4))
        acc = cute.make_rmem_tensor(
            cute.make_layout((n_mmas, 4), stride=(4, 1)),
            cutlass.Float32,
        )
        acc.fill(0.0)

        if const_expr(cols_per_block == 32):
            n_tile = bidx >> 1
            half = bidx & 1
            vec0 = n_tile * 64 + lane * 2 + half
        else:
            n_tile = bidx
            vec0 = n_tile * 64 + lane * 2

        total_k_tiles = (
            const_expr(self.static_k // 16)
            if const_expr(self.static_k is not None)
            else k_tiles
        )
        n_iters = total_k_tiles // num_warps
        valid_a_row = group_id < m_extent

        if const_expr(
            self.pingpong
            and self.static_k is not None
            and (n_iters % 2 == 0)
            and n_iters >= 2
        ):
            b0 = cute.make_rmem_tensor(
                cute.make_layout((n_v4, 4), stride=(4, 1)), cutlass.Int32
            )
            b1 = cute.make_rmem_tensor(
                cute.make_layout((n_v4, 4), stride=(4, 1)), cutlass.Int32
            )
            n_pairs = const_expr(n_iters // 2)
            kt_init = (warp_id * n_iters) if const_expr(self.contig_k) else warp_id
            if const_expr(cols_per_block == 32):
                cute.copy(copy_b, gB_v4[(0, None), (kt_init, vec0)], b0[0, None])
            else:
                cute.copy(copy_b, gB_v4[(0, None), (kt_init, vec0)], b0[0, None])
                cute.copy(copy_b, gB_v4[(0, None), (kt_init, vec0 + 1)], b0[1, None])

            for p in cutlass.range(n_pairs, unroll=self.k_unroll):
                it0 = p * 2
                it1 = it0 + 1
                kt0 = (
                    (warp_id * n_iters + it0)
                    if const_expr(self.contig_k)
                    else (warp_id + it0 * num_warps)
                )
                kt1 = (
                    (warp_id * n_iters + it1)
                    if const_expr(self.contig_k)
                    else (warp_id + it1 * num_warps)
                )
                if const_expr(cols_per_block == 32):
                    cute.copy(copy_b, gB_v4[(0, None), (kt1, vec0)], b1[0, None])
                else:
                    cute.copy(copy_b, gB_v4[(0, None), (kt1, vec0)], b1[0, None])
                    cute.copy(copy_b, gB_v4[(0, None), (kt1, vec0 + 1)], b1[1, None])

                k_pair0 = kt0 * 8 + tid_in_g
                xb0 = cutlass.Int32(0)
                xb1 = cutlass.Int32(0)
                if valid_a_row:
                    xb0 = gA_u32[group_id, k_pair0]
                    xb1 = gA_u32[group_id, k_pair0 + 4]
                for vi in cutlass.range_constexpr(n_v4):
                    for pair_i in cutlass.range_constexpr(2):
                        mi_mma = vi * 2 + pair_i
                        a0, a2 = _dequant_u8b128_to_bf16x2_pair(
                            b0[vi, pair_i * 2 + 0]
                        )
                        a1, a3 = _dequant_u8b128_to_bf16x2_pair(
                            b0[vi, pair_i * 2 + 1]
                        )
                        d0, d1, d2, d3 = _mma_m16n8k16_bf16(
                            a0,
                            a1,
                            a2,
                            a3,
                            xb0,
                            xb1,
                            acc[mi_mma, 0],
                            acc[mi_mma, 1],
                            acc[mi_mma, 2],
                            acc[mi_mma, 3],
                        )
                        acc[mi_mma, 0] = d0
                        acc[mi_mma, 1] = d1
                        acc[mi_mma, 2] = d2
                        acc[mi_mma, 3] = d3

                if p + 1 < n_pairs:
                    it2 = it0 + 2
                    kt2 = (
                        (warp_id * n_iters + it2)
                        if const_expr(self.contig_k)
                        else (warp_id + it2 * num_warps)
                    )
                    if const_expr(cols_per_block == 32):
                        cute.copy(copy_b, gB_v4[(0, None), (kt2, vec0)], b0[0, None])
                    else:
                        cute.copy(copy_b, gB_v4[(0, None), (kt2, vec0)], b0[0, None])
                        cute.copy(
                            copy_b, gB_v4[(0, None), (kt2, vec0 + 1)], b0[1, None]
                        )

                k_pair1 = kt1 * 8 + tid_in_g
                xb0 = cutlass.Int32(0)
                xb1 = cutlass.Int32(0)
                if valid_a_row:
                    xb0 = gA_u32[group_id, k_pair1]
                    xb1 = gA_u32[group_id, k_pair1 + 4]
                for vi in cutlass.range_constexpr(n_v4):
                    for pair_i in cutlass.range_constexpr(2):
                        mi_mma = vi * 2 + pair_i
                        a0, a2 = _dequant_u8b128_to_bf16x2_pair(
                            b1[vi, pair_i * 2 + 0]
                        )
                        a1, a3 = _dequant_u8b128_to_bf16x2_pair(
                            b1[vi, pair_i * 2 + 1]
                        )
                        d0, d1, d2, d3 = _mma_m16n8k16_bf16(
                            a0,
                            a1,
                            a2,
                            a3,
                            xb0,
                            xb1,
                            acc[mi_mma, 0],
                            acc[mi_mma, 1],
                            acc[mi_mma, 2],
                            acc[mi_mma, 3],
                        )
                        acc[mi_mma, 0] = d0
                        acc[mi_mma, 1] = d1
                        acc[mi_mma, 2] = d2
                        acc[mi_mma, 3] = d3
        else:
            b_regs = cute.make_rmem_tensor(
                cute.make_layout((n_v4, 4), stride=(4, 1)), cutlass.Int32
            )
            for it in cutlass.range(n_iters, unroll=self.k_unroll):
                kt = (
                    (warp_id * n_iters + it)
                    if const_expr(self.contig_k and self.static_k is not None)
                    else (warp_id + it * num_warps)
                )
                if const_expr(cols_per_block == 32):
                    cute.copy(copy_b, gB_v4[(0, None), (kt, vec0)], b_regs[0, None])
                else:
                    cute.copy(copy_b, gB_v4[(0, None), (kt, vec0)], b_regs[0, None])
                    cute.copy(
                        copy_b, gB_v4[(0, None), (kt, vec0 + 1)], b_regs[1, None]
                    )
                k_pair = kt * 8 + tid_in_g
                xb0 = cutlass.Int32(0)
                xb1 = cutlass.Int32(0)
                if valid_a_row:
                    xb0 = gA_u32[group_id, k_pair]
                    xb1 = gA_u32[group_id, k_pair + 4]
                for vi in cutlass.range_constexpr(n_v4):
                    for pair_i in cutlass.range_constexpr(2):
                        mi_mma = vi * 2 + pair_i
                        a0, a2 = _dequant_u8b128_to_bf16x2_pair(
                            b_regs[vi, pair_i * 2 + 0]
                        )
                        a1, a3 = _dequant_u8b128_to_bf16x2_pair(
                            b_regs[vi, pair_i * 2 + 1]
                        )
                        d0, d1, d2, d3 = _mma_m16n8k16_bf16(
                            a0,
                            a1,
                            a2,
                            a3,
                            xb0,
                            xb1,
                            acc[mi_mma, 0],
                            acc[mi_mma, 1],
                            acc[mi_mma, 2],
                            acc[mi_mma, 3],
                        )
                        acc[mi_mma, 0] = d0
                        acc[mi_mma, 1] = d1
                        acc[mi_mma, 2] = d2
                        acc[mi_mma, 3] = d3

            if const_expr(self.static_k is None):
                kt_tail = n_iters * num_warps + warp_id
                if kt_tail < k_tiles:
                    if const_expr(cols_per_block == 32):
                        cute.copy(
                            copy_b, gB_v4[(0, None), (kt_tail, vec0)], b_regs[0, None]
                        )
                    else:
                        cute.copy(
                            copy_b, gB_v4[(0, None), (kt_tail, vec0)], b_regs[0, None]
                        )
                        cute.copy(
                            copy_b,
                            gB_v4[(0, None), (kt_tail, vec0 + 1)],
                            b_regs[1, None],
                        )
                    k_pair = kt_tail * 8 + tid_in_g
                    xb0 = cutlass.Int32(0)
                    xb1 = cutlass.Int32(0)
                    if valid_a_row:
                        xb0 = gA_u32[group_id, k_pair]
                        xb1 = gA_u32[group_id, k_pair + 4]
                    for vi in cutlass.range_constexpr(n_v4):
                        for pair_i in cutlass.range_constexpr(2):
                            mi_mma = vi * 2 + pair_i
                            a0, a2 = _dequant_u8b128_to_bf16x2_pair(
                                b_regs[vi, pair_i * 2 + 0]
                            )
                            a1, a3 = _dequant_u8b128_to_bf16x2_pair(
                                b_regs[vi, pair_i * 2 + 1]
                            )
                            d0, d1, d2, d3 = _mma_m16n8k16_bf16(
                                a0,
                                a1,
                                a2,
                                a3,
                                xb0,
                                xb1,
                                acc[mi_mma, 0],
                                acc[mi_mma, 1],
                                acc[mi_mma, 2],
                                acc[mi_mma, 3],
                            )
                            acc[mi_mma, 0] = d0
                            acc[mi_mma, 1] = d1
                            acc[mi_mma, 2] = d2
                            acc[mi_mma, 3] = d3

        # Cross-warp FP32 reduction in bank-conflict-free padded smem
        col_block_base = bidx * cols_per_block
        smem = cutlass.memory.SmemAllocator()
        smem_layout = cute.make_layout(
            (max_m, cols_per_block, num_warps),
            stride=(cols_per_block * w_stride, w_stride, 1),
        )
        partials = smem.allocate_tensor(
            cutlass.Float32, smem_layout, byte_alignment=16
        )

        m0 = tid_in_g * 2
        m1 = m0 + 1
        if m0 < m_extent:
            for mi_mma in cutlass.range_constexpr(n_mmas):
                lc0 = mi_mma * 16 + group_id
                partials[m0, lc0, warp_id] = acc[mi_mma, 0]
                partials[m0, lc0 + 8, warp_id] = acc[mi_mma, 2]
        if m1 < m_extent:
            for mi_mma in cutlass.range_constexpr(n_mmas):
                lc0 = mi_mma * 16 + group_id
                partials[m1, lc0, warp_id] = acc[mi_mma, 1]
                partials[m1, lc0 + 8, warp_id] = acc[mi_mma, 3]

        cute.arch.sync_threads()

        # Parallel coalesced epilogue: unpermute Marlin channelwise scale, scale in FP32, cast BF16
        total_outs = m_extent * cols_per_block
        max_outs = const_expr(max_m * cols_per_block)
        n_ep_iters = const_expr((max_outs + self.block_size - 1) // self.block_size)
        for ei in cutlass.range_constexpr(n_ep_iters):
            out_id = tidx + ei * self.block_size
            if out_id < total_outs:
                mi = out_id // cols_per_block
                lc = out_id % cols_per_block
                col = col_block_base + lc
                # scale_perm_single is an involution over each 32-column group:
                # col = blk32*32 + c_mod4*8 + tc_c  <->  s_idx = blk32*32 + (tc_c>>1)*8 + c_mod4*2 + (tc_c&1)
                blk32 = col >> 5
                in32 = col & 31
                c_mod4 = in32 >> 3
                tc_c = in32 & 7
                s_idx = blk32 * 32 + (tc_c >> 1) * 8 + c_mod4 * 2 + (tc_c & 1)
                s_f32 = gS[s_idx].to(cutlass.Float32)
                total = (
                    partials[mi, lc, None]
                    .load()
                    .reduce(
                        cute.ReductionOp.ADD,
                        init_val=cutlass.Float32(0.0),
                        reduction_profile=0,
                    )
                )
                gC[mi, col] = cutlass.Float32(total * s_f32).to(cutlass.BFloat16)
