# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 lebeche hill-climb (fork of Merlin 3.0.2.2)
"""Dynamic-M CuTe-DSL HyperConnection combine + grouped GemmaRMSNorm for sm_8x.

Replaces Triton ``_hc_combine_norm_kernel`` (4 warps/block, 4096-wide padded tile)
with an exact-width 320-thread (10-warp) 128-bit vectorized kernel for the
Flash-Next HC geometry (hc=4, hc_dim=2560 -> 1 tile of 8 BF16 elements per thread,
zero masking, dynamic M via grid=[M * hc, 1, 1]).

Only two compile units are needed per deployment (``has_inj=True`` and
``has_inj=False``), covering all token counts M >= 1.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cuda.bindings.driver import CUstream
from cutlass import const_expr

from ._hc_small_fused import _ptx_f32, _triton_div, _triton_sigmoid

_SmemAllocator = getattr(getattr(cutlass, "memory", None), "SmemAllocator", None) or cutlass.utils.SmemAllocator


class CuteHcCombineNorm:
    """Fused residual combine + per-branch GemmaRMSNorm with dynamic batch size M."""

    def __init__(
        self,
        *,
        element_type,
        has_inj: bool = True,
        warps_per_stream: int = 10,
        hc: int = 4,
        hc_dim: int = 2560,
        eps: float = 1e-6,
    ) -> None:
        bs = warps_per_stream * 32
        if hc_dim % (bs * 8) != 0:
            raise ValueError("hc_dim must be divisible by warps_per_stream * 32 * 8")
        self.element_type = element_type
        self.has_inj = has_inj
        self.wps = warps_per_stream
        self.hc = hc
        self.hc_dim = hc_dim
        self.eps = eps

    @cute.jit
    def __call__(
        self,
        gRes: cute.Tensor,  # [M, hc * hc_dim]
        gBlk: cute.Tensor,  # [M, hc_dim]
        gInj: cute.Tensor,  # [M, hc] (unused when has_inj=False)
        gNW: cute.Tensor,   # [1, hc * hc_dim]
        gOut: cute.Tensor,  # [M, hc * hc_dim]
        gXn: cute.Tensor,   # [M, hc * hc_dim]
        stream: CUstream,
    ) -> None:
        m = cute.size(gRes, mode=[0])
        copy_g2r = cute.make_copy_atom(
            cute.nvgpu.CopyG2ROp(),
            self.element_type,
            num_bits_per_copy=128,
            load_cache_mode=cute.nvgpu.LoadCacheMode.ALWAYS,
        )
        copy_r2g = cute.make_copy_atom(
            cute.nvgpu.CopyR2GOp(),
            self.element_type,
            num_bits_per_copy=128,
        )
        self.kernel(
            gRes, gBlk, gInj, gNW, gOut, gXn, copy_g2r, copy_r2g
        ).launch(
            grid=[m * self.hc, 1, 1],
            block=[self.wps * 32, 1, 1],
            smem=self.wps * 4,
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        gRes: cute.Tensor,
        gBlk: cute.Tensor,
        gInj: cute.Tensor,
        gNW: cute.Tensor,
        gOut: cute.Tensor,
        gXn: cute.Tensor,
        copy_g2r: cute.CopyAtom,
        copy_r2g: cute.CopyAtom,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        pid, _, _ = cute.arch.block_idx()
        mi = pid // self.hc
        s = pid % self.hc

        wps: cutlass.Constexpr = self.wps
        bs: cutlass.Constexpr = wps * 32
        hc_dim: cutlass.Constexpr = self.hc_dim
        num_tiles: cutlass.Constexpr = hc_dim // (bs * 8)

        scale = cutlass.Float32(1.0)
        if const_expr(self.has_inj):
            scale = cutlass.Float32(2.0) * _triton_sigmoid(
                _triton_div(gInj[mi, s].to(cutlass.Float32), cutlass.Float32(float(self.hc)))
            )

        tRes = cute.logical_divide(
            cute.logical_divide(gRes, (None, 8)), (None, (None, bs))
        )[mi, (None, (tidx, None))]
        tBlk = cute.logical_divide(
            cute.logical_divide(gBlk, (None, 8)), (None, (None, bs))
        )[mi, (None, (tidx, None))]
        tOut = cute.logical_divide(
            cute.logical_divide(gOut, (None, 8)), (None, (None, bs))
        )[mi, (None, (tidx, None))]
        tXn = cute.logical_divide(
            cute.logical_divide(gXn, (None, 8)), (None, (None, bs))
        )[mi, (None, (tidx, None))]
        tNW = cute.logical_divide(
            cute.logical_divide(gNW, (None, 8)), (None, (None, bs))
        )[0, (None, (tidx, None))]

        out_all = cute.make_rmem_tensor(
            cute.make_layout((num_tiles, 8), stride=(8, 1)), self.element_type
        )
        res_reg = cute.make_rmem_tensor(cute.make_layout((8,)), self.element_type)
        blk_reg = cute.make_rmem_tensor(cute.make_layout((8,)), self.element_type)
        nw_reg = cute.make_rmem_tensor(cute.make_layout((8,)), self.element_type)

        sum_sq = cutlass.Float32(0.0)
        tile_off = s * num_tiles
        for tile in cutlass.range_constexpr(num_tiles):
            st = tile_off + tile
            cute.copy(copy_g2r, tRes[None, st], res_reg)
            cute.copy(copy_g2r, tBlk[None, tile], blk_reg)
            for vi in cutlass.range_constexpr(8):
                ov = cutlass.Float32(
                    res_reg[vi].to(cutlass.Float32)
                    + blk_reg[vi].to(cutlass.Float32) * scale
                ).to(self.element_type)
                out_all[tile, vi] = ov
                of = ov.to(cutlass.Float32)
                sum_sq = sum_sq + of * of
            cute.copy(copy_r2g, out_all[tile, None], tOut[None, st])

        sum_sq = cute.arch.warp_reduction_sum(sum_sq)
        smem = _SmemAllocator()
        partials = smem.allocate_tensor(
            cutlass.Float32, cute.make_layout((wps,)), byte_alignment=16
        )
        if tidx % 32 == 0:
            partials[tidx // 32] = sum_sq
        cute.arch.sync_threads()
        sum_sq = partials.load().reduce(
            cute.ReductionOp.ADD,
            init_val=cutlass.Float32(0.0),
            reduction_profile=0,
        )

        mean_sq = _triton_div(sum_sq, cutlass.Float32(float(hc_dim))) + cutlass.Float32(
            self.eps
        )
        rrms = _ptx_f32("rsqrt.approx.f32 {$w0}, {$r0};", mean_sq)

        for tile in cutlass.range_constexpr(num_tiles):
            st = tile_off + tile
            cute.copy(copy_g2r, tNW[None, st], nw_reg)
            for vi in cutlass.range_constexpr(8):
                y = out_all[tile, vi].to(cutlass.Float32) * rrms
                y = _ptx_f32(
                    "fma.rn.f32 {$w0}, {$r0}, {$r1}, {$r2};",
                    y,
                    nw_reg[vi].to(cutlass.Float32),
                    y,
                )
                res_reg[vi] = y.to(self.element_type)
            cute.copy(copy_r2g, res_reg, tXn[None, st])
