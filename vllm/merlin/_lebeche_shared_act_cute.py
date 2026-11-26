# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 lebeche hill-climb (fork of Merlin 3.0.2.2)
"""CuTe-DSL kernel for fused small-M shared-expert gate_up_proj + silu_and_mul."""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cuda.bindings.driver import CUstream
from cutlass import BFloat16, Float32, Int32, const_expr

_PREFETCH_TILES = 2


def _ptx_f32(code: str, *args):
    return cute.arch.inline_ptx(
        code, write_only_types=[Float32], read_only_args=list(args)
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


def _triton_silu_mul(g, u):
    # Exact match to Inductor triton_poi_fused_mul_silu_slice_0:
    # (g / (libdevice.exp(-g) + 1.0)) * u
    ng = _ptx_f32("neg.f32 {$w0}, {$r0};", g)
    e = _nv_expf(ng)
    d = _ptx_f32("add.f32 {$w0}, {$r0}, 0f3F800000;", e)
    s = _ptx_f32("div.full.f32 {$w0}, {$r0}, {$r1};", g, d)
    return _ptx_f32("mul.f32 {$w0}, {$r0}, {$r1};", s, u)


class CuteSharedGateUpSilu:
    def __init__(
        self,
        *,
        element_type=BFloat16,
        num_rows: int,
        block_size: int,
        outputs_per_block: int,
        vector_width: int = 8,
        k_unroll: int = 1,
        use_pdl: bool = False,
        static_k: int | None = None,
    ) -> None:
        if outputs_per_block % 2 != 0:
            raise ValueError("outputs_per_block must be even")
        if block_size % cute.arch.WARP_SIZE != 0:
            raise ValueError("block_size must be a multiple of warp size")
        if num_rows * outputs_per_block > cute.arch.WARP_SIZE:
            raise ValueError("num_rows * outputs_per_block must fit in one warp")
        self.element_type = element_type
        self.num_rows = num_rows
        self.block_size = block_size
        self.outputs_per_block = outputs_per_block
        self.vector_width = vector_width
        self.k_unroll = k_unroll
        self.use_pdl = use_pdl
        self.static_k = static_k
        self.num_warps = block_size // cute.arch.WARP_SIZE

    @cute.jit
    def __call__(
        self,
        gA: cute.Tensor,
        gB: cute.Tensor,
        gC: cute.Tensor,
        stream: CUstream,
    ) -> None:
        n = cute.size(gB, mode=[0])
        k = cute.size(gA, mode=[1])
        copy_a = cute.make_copy_atom(
            cute.nvgpu.CopyG2ROp(),
            self.element_type,
            num_bits_per_copy=self.vector_width * self.element_type.width,
            load_cache_mode=cute.nvgpu.LoadCacheMode.ALWAYS,
        )
        copy_b = cute.make_copy_atom(
            cute.nvgpu.CopyG2ROp(),
            self.element_type,
            num_bits_per_copy=self.vector_width * self.element_type.width,
            load_cache_mode=cute.nvgpu.LoadCacheMode.STREAMING,
        )
        copy_b_prefetch = cute.make_copy_atom(
            cute.nvgpu.CopyG2ROp(),
            self.element_type,
            num_bits_per_copy=self.vector_width * self.element_type.width,
            load_cache_mode=cute.nvgpu.LoadCacheMode.ALWAYS,
        )
        smem_bytes = (
            0
            if self.num_warps == 1
            else self.num_rows * self.outputs_per_block * self.num_warps * 4
        )
        self.kernel(
            gA, gB, gC, k, n // 2, copy_a, copy_b, copy_b_prefetch
        ).launch(
            grid=[cute.ceil_div(n, self.outputs_per_block), 1, 1],
            block=[self.block_size, 1, 1],
            smem=smem_bytes,
            stream=stream,
            use_pdl=self.use_pdl,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        gA: cute.Tensor,
        gB: cute.Tensor,
        gC: cute.Tensor,
        k_extent: Int32,
        half_n: Int32,
        copy_a: cute.CopyAtom,
        copy_b: cute.CopyAtom,
        copy_b_prefetch: cute.CopyAtom,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        block_idx, _, _ = cute.arch.block_idx()
        warp_idx = cute.arch.warp_idx()

        num_rows: cutlass.Constexpr = self.num_rows
        outputs_per_block: cutlass.Constexpr = self.outputs_per_block
        vector_width: cutlass.Constexpr = self.vector_width
        block_size: cutlass.Constexpr = self.block_size
        num_warps: cutlass.Constexpr = self.num_warps
        pairs_per_block: cutlass.Constexpr = outputs_per_block // 2

        acc = cute.make_rmem_tensor(
            cute.make_layout(
                (num_rows, outputs_per_block), stride=(outputs_per_block, 1)
            ),
            Float32,
        )
        acc.fill(0.0)

        pair_base = block_idx * pairs_per_block
        k_tile_size: cutlass.Constexpr = block_size * vector_width

        gA_vec = cute.logical_divide(gA, (None, vector_width))
        gB_vec = cute.logical_divide(gB, (None, vector_width))
        tA_all = cute.logical_divide(gA_vec, (None, (None, block_size)))
        tB_all = cute.logical_divide(gB_vec, (None, (None, block_size)))
        tA = tA_all[None, (None, (tidx, None))]

        a_regs = cute.make_rmem_tensor(
            cute.make_layout((num_rows, vector_width), stride=(vector_width, 1)),
            self.element_type,
        )
        b_regs = cute.make_rmem_tensor(
            cute.make_layout(
                (outputs_per_block, vector_width), stride=(vector_width, 1)
            ),
            self.element_type,
        )

        if const_expr(self.static_k is not None):
            num_k_tiles: cutlass.Constexpr = self.static_k // k_tile_size
            prefetched_b = cute.make_rmem_tensor(
                cute.make_layout(
                    (_PREFETCH_TILES, outputs_per_block, vector_width),
                    stride=(outputs_per_block * vector_width, vector_width, 1),
                ),
                self.element_type,
            )
            for k_tile in cutlass.range_constexpr(_PREFETCH_TILES):
                for ni in cutlass.range_constexpr(outputs_per_block):
                    n_idx = (ni % 2) * half_n + pair_base + (ni // 2)
                    tB_static = tB_all[n_idx, (None, (tidx, None))]
                    cute.copy(
                        copy_b_prefetch,
                        tB_static[None, k_tile],
                        prefetched_b[k_tile, ni, None],
                    )

            if const_expr(self.use_pdl):
                cute.arch.griddepcontrol_wait()

            for k_tile in cutlass.range_constexpr(num_k_tiles):
                if const_expr(k_tile < _PREFETCH_TILES):
                    b_f32 = prefetched_b[k_tile, None, None].load().to(Float32)
                else:
                    for ni in cutlass.range_constexpr(outputs_per_block):
                        n_idx = (ni % 2) * half_n + pair_base + (ni // 2)
                        tB_static = tB_all[n_idx, (None, (tidx, None))]
                        cute.copy(
                            copy_b, tB_static[None, k_tile], b_regs[ni, None]
                        )
                    b_f32 = b_regs.load().to(Float32)
                for mi in cutlass.range_constexpr(num_rows):
                    cute.copy(copy_a, tA[mi, None, k_tile], a_regs[mi, None])
                a_f32 = a_regs.load().to(Float32)
                for vi in cutlass.range_constexpr(vector_width):
                    for mi in cutlass.range_constexpr(num_rows):
                        for ni in cutlass.range_constexpr(outputs_per_block):
                            acc[mi, ni] += a_f32[mi, vi] * b_f32[ni, vi]
        else:
            if const_expr(self.use_pdl):
                cute.arch.griddepcontrol_wait()
            num_k_tiles = k_extent // k_tile_size
            for k_tile in cutlass.range(num_k_tiles, unroll=self.k_unroll):
                for mi in cutlass.range_constexpr(num_rows):
                    cute.copy(copy_a, tA[mi, None, k_tile], a_regs[mi, None])
                for ni in cutlass.range_constexpr(outputs_per_block):
                    n_idx = (ni % 2) * half_n + pair_base + (ni // 2)
                    tB_dyn = tB_all[n_idx, (None, (tidx, None))]
                    cute.copy(copy_b, tB_dyn[None, k_tile], b_regs[ni, None])
                for vi in cutlass.range_constexpr(vector_width):
                    for mi in cutlass.range_constexpr(num_rows):
                        for ni in cutlass.range_constexpr(outputs_per_block):
                            acc[mi, ni] = acc[mi, ni] + a_regs[mi, vi].to(
                                Float32
                            ) * b_regs[ni, vi].to(Float32)

        for mi in cutlass.range_constexpr(num_rows):
            for ni in cutlass.range_constexpr(outputs_per_block):
                acc[mi, ni] = cute.arch.warp_reduction_sum(acc[mi, ni])

        if const_expr(num_warps == 1):
            num_pairs_total: cutlass.Constexpr = num_rows * pairs_per_block
            if tidx < Int32(num_pairs_total):
                mi_p = tidx // Int32(pairs_per_block)
                pi_p = tidx % Int32(pairs_per_block)
                g_val = Float32(acc[mi_p, pi_p * 2]).to(self.element_type).to(Float32)
                u_val = (
                    Float32(acc[mi_p, pi_p * 2 + 1]).to(self.element_type).to(Float32)
                )
                gC[mi_p, pair_base + pi_p] = _triton_silu_mul(g_val, u_val).to(
                    self.element_type
                )
        else:
            smem = cutlass.memory.SmemAllocator()
            partials = smem.allocate_tensor(
                Float32,
                cute.make_layout(
                    (num_rows, outputs_per_block, num_warps),
                    stride=(outputs_per_block * num_warps, num_warps, 1),
                ),
                byte_alignment=16,
            )
            # Portable lane-0 check (no sm_90 elect.sync).
            if tidx % cute.arch.WARP_SIZE == 0:
                for mi in cutlass.range_constexpr(num_rows):
                    for ni in cutlass.range_constexpr(outputs_per_block):
                        partials[mi, ni, warp_idx] = acc[mi, ni]
            cute.arch.sync_threads()

            num_slots: cutlass.Constexpr = num_rows * outputs_per_block
            if warp_idx == 0:
                slot = tidx % num_slots
                mi_l = slot // outputs_per_block
                ni_l = slot % outputs_per_block
                total = (
                    partials[mi_l, ni_l, None]
                    .load()
                    .reduce(
                        cute.ReductionOp.ADD,
                        init_val=Float32(0.0),
                        reduction_profile=0,
                    )
                )
                y = Float32(total).to(self.element_type).to(Float32)
                u = cute.arch.shuffle_sync_down(y, 1)
                out_val = _triton_silu_mul(y, u)
                if tidx < num_slots and ni_l % 2 == 0:
                    gC[mi_l, pair_base + (ni_l // 2)] = out_val.to(self.element_type)

        if const_expr(self.use_pdl):
            cute.arch.griddepcontrol_launch_dependents()
