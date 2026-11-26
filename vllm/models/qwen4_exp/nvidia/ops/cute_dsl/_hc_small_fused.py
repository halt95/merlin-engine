# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 lebeche hill-climb (fork of Merlin 3.0.2.2)
"""Skinny BF16 GEMM with the HyperConnection glue fused in as an epilogue.

The GEMM body is the shape-dynamic skinny GEMM
(vllm/model_executor/kernels/linear/cute_dsl/_skinny_gemm.py): same loads, same
FP32 accumulation order for a given ``(block_size, vector_width)``, same
warp/block reduction. Only the tail differs. The block total is rounded to the
element type first (the rounding point of the unfused GEMM output) and then:

``EPILOGUE_DOWN_SILU``
    rows ``< rank`` store ``silu(bf16(total) / hc)`` (``_hc_silu_kernel``) into
    ``gC``; the following logit rows store ``bf16(total)`` into ``gL``.

``EPILOGUE_UP_GATE_MIX``
    the weight rows are stream-major ``[hc, hc_dim]``; a block owns
    ``outputs_per_block / hc`` inner indices of *all* streams and stores
    ``mean_s(sigmoid(bf16(gate_s)) * x_s)`` (``_hc_gate_mix_kernel``), so the
    ``[M, hc * hc_dim]`` gate never reaches memory.

The epilogue runs lane-parallel in warp 0 (one lane per output of the block),
which needs ``num_rows * outputs_per_block <= 32``. Lane 0 of each warp
publishes the warp partials (no ``elect.sync``: this kernel targets sm_8x).
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cuda.bindings.driver import CUstream
from cutlass import const_expr

_PREFETCH_TILES = 2

EPILOGUE_DOWN_SILU = "down_silu"
EPILOGUE_UP_GATE_MIX = "up_gate_mix"


# The epilogues must reproduce the Triton glue kernels (ops/hc.py) bit for bit.
# Triton lowers x / HC and 1 / (1 + exp(-x)) to div.full and
# ex2.approx(x * -log2(e)); the DSL's own / and exp lower to
# rcp.rn / an accurate expf, which differ in the last FP32 bit. So these
# steps are emitted as the same PTX instructions Triton generates.
def _ptx_f32(code: str, *args):
    return cute.arch.inline_ptx(
        code, write_only_types=[cutlass.Float32], read_only_args=list(args)
    )


def _triton_div(a, b):
    return _ptx_f32("div.full.f32 {$w0}, {$r0}, {$r1};", a, b)


def _triton_sigmoid(x):
    # 0fBFB8AA3B = -log2(e); 0f3F800000 = 1.0
    t = _ptx_f32("mul.f32 {$w0}, {$r0}, 0fBFB8AA3B;", x)
    e = _ptx_f32("ex2.approx.f32 {$w0}, {$r0};", t)
    d = _ptx_f32("add.f32 {$w0}, {$r0}, 0f3F800000;", e)
    return _triton_div(cutlass.Float32(1.0), d)


class CuteHcSmallFused:
    def __init__(
        self,
        *,
        element_type,
        epilogue: str,
        num_rows: int,
        block_size: int,
        outputs_per_block: int,
        hc: int,
        rank: int = 0,
        vector_width: int = 8,
        k_unroll: int = 1,
        use_pdl: bool = False,
        static_k: int | None = None,
    ) -> None:
        if epilogue not in (EPILOGUE_DOWN_SILU, EPILOGUE_UP_GATE_MIX):
            raise ValueError(f"unknown epilogue {epilogue!r}")
        if block_size % cute.arch.WARP_SIZE != 0:
            raise ValueError("block_size must be a multiple of the warp size")
        if num_rows * outputs_per_block > cute.arch.WARP_SIZE:
            raise ValueError("num_rows * outputs_per_block must fit in one warp")
        if epilogue == EPILOGUE_UP_GATE_MIX and outputs_per_block % hc:
            raise ValueError("outputs_per_block must be a multiple of hc")
        k_tile_size = block_size * vector_width
        if static_k is not None and (
            static_k % k_tile_size or static_k < _PREFETCH_TILES * k_tile_size
        ):
            raise ValueError("static K must contain at least two complete tiles")
        self.element_type = element_type
        self.epilogue = epilogue
        self.num_rows = num_rows
        self.block_size = block_size
        self.outputs_per_block = outputs_per_block
        self.hc = hc
        self.rank = rank
        self.vector_width = vector_width
        self.k_unroll = k_unroll
        self.use_pdl = use_pdl
        self.static_k = static_k
        self.num_warps = block_size // cute.arch.WARP_SIZE

    @cute.jit
    def __call__(
        self,
        gA: cute.Tensor,  # [M, K] GEMM input
        gB: cute.Tensor,  # [N, K] weight
        gX: cute.Tensor,  # [M, N] gate-mix operand (unused for down_silu)
        gC: cute.Tensor,  # [M, rank] (down_silu) or [M, N / hc] (up_gate_mix)
        gL: cute.Tensor,  # [M, N - rank] logits (unused for up_gate_mix)
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
        self.kernel(
            gA,
            gB,
            gX,
            gC,
            gL,
            k,
            n // self.hc,
            copy_a,
            copy_b,
            copy_b_prefetch,
        ).launch(
            grid=[cute.ceil_div(n, self.outputs_per_block), 1, 1],
            block=[self.block_size, 1, 1],
            smem=self.num_rows * self.outputs_per_block * self.num_warps * 4,
            stream=stream,
            use_pdl=self.use_pdl,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        gA: cute.Tensor,
        gB: cute.Tensor,
        gX: cute.Tensor,
        gC: cute.Tensor,
        gL: cute.Tensor,
        k_extent: cutlass.Int32,
        hc_dim: cutlass.Int32,
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
        hc: cutlass.Constexpr = self.hc
        gate_mix: cutlass.Constexpr = self.epilogue == EPILOGUE_UP_GATE_MIX
        # Inner indices per block (gate mix); weight row of output slot ni.
        inner_per_block: cutlass.Constexpr = outputs_per_block // hc if gate_mix else 0

        acc_layout = cute.make_layout(
            (num_rows, outputs_per_block), stride=(outputs_per_block, 1)
        )
        acc = cute.make_rmem_tensor(acc_layout, cutlass.Float32)
        acc.fill(0.0)

        n_base = block_idx * outputs_per_block
        inner_base = block_idx * inner_per_block
        k_tile_size: cutlass.Constexpr = block_size * vector_width

        gA_vec = cute.logical_divide(gA, (None, vector_width))
        gB_vec = cute.logical_divide(gB, (None, vector_width))
        # Layout after both divides is (M/N, K_TILE, K_LANE, K_VEC).
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
                    if const_expr(gate_mix):
                        n_idx = (ni % hc) * hc_dim + inner_base + ni // hc
                    else:
                        n_idx = n_base + ni
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
                    b_f32 = prefetched_b[k_tile, None, None].load().to(cutlass.Float32)
                else:
                    for ni in cutlass.range_constexpr(outputs_per_block):
                        if const_expr(gate_mix):
                            n_idx = (ni % hc) * hc_dim + inner_base + ni // hc
                        else:
                            n_idx = n_base + ni
                        tB_static = tB_all[n_idx, (None, (tidx, None))]
                        cute.copy(
                            copy_b,
                            tB_static[None, k_tile],
                            b_regs[ni, None],
                        )
                    b_f32 = b_regs.load().to(cutlass.Float32)

                for mi in cutlass.range_constexpr(num_rows):
                    cute.copy(copy_a, tA[mi, None, k_tile], a_regs[mi, None])
                a_f32 = a_regs.load().to(cutlass.Float32)

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
                    if const_expr(gate_mix):
                        n_idx = (ni % hc) * hc_dim + inner_base + ni // hc
                    else:
                        n_idx = n_base + ni
                    tB_dynamic = tB_all[n_idx, (None, (tidx, None))]
                    cute.copy(
                        copy_b,
                        tB_dynamic[None, k_tile],
                        b_regs[ni, None],
                    )

                for vi in cutlass.range_constexpr(vector_width):
                    for mi in cutlass.range_constexpr(num_rows):
                        for ni in cutlass.range_constexpr(outputs_per_block):
                            acc[mi, ni] = acc[mi, ni] + a_regs[mi, vi].to(
                                cutlass.Float32
                            ) * b_regs[ni, vi].to(cutlass.Float32)

        for mi in cutlass.range_constexpr(num_rows):
            for ni in cutlass.range_constexpr(outputs_per_block):
                acc[mi, ni] = cute.arch.warp_reduction_sum(acc[mi, ni])

        smem_layout = cute.make_layout(
            (num_rows, outputs_per_block, num_warps),
            stride=(outputs_per_block * num_warps, num_warps, 1),
        )
        smem = cutlass.utils.SmemAllocator()
        partials = smem.allocate_tensor(cutlass.Float32, smem_layout, byte_alignment=16)
        if tidx % cute.arch.WARP_SIZE == 0:
            for mi in cutlass.range_constexpr(num_rows):
                for ni in cutlass.range_constexpr(outputs_per_block):
                    partials[mi, ni, warp_idx] = acc[mi, ni]

        cute.arch.sync_threads()

        # Lane-parallel epilogue in warp 0: lane = mi * outputs_per_block + ni.
        # Lanes past the last slot recompute a valid slot and store nothing.
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
                    init_val=cutlass.Float32(0.0),
                    reduction_profile=0,
                )
            )
            # Rounding point of the unfused GEMM output.
            y_elem = cutlass.Float32(total).to(self.element_type)
            y = y_elem.to(cutlass.Float32)
            if const_expr(gate_mix):
                # _hc_gate_mix_kernel: acc = fma(sigmoid(g_s), x_s, acc) over
                # the streams in order (Triton contracts its mul+add), then
                # acc / HC. Stream s of an inner index sits s lanes above its
                # stream-0 lane, which gathers the sigmoids. Every lane runs
                # the (in-bounds) chain so it stays straight-line; only the
                # stream-0 lanes of real slots store.
                sig = _triton_sigmoid(y)
                mix = cutlass.Float32(0.0)
                inner = inner_base + ni_l // hc
                for s in cutlass.range_constexpr(hc):
                    sig_s = sig
                    if const_expr(s > 0):
                        sig_s = cute.arch.shuffle_sync_down(sig, s)
                    mix = _ptx_f32(
                        "fma.rn.f32 {$w0}, {$r0}, {$r1}, {$r2};",
                        sig_s,
                        gX[mi_l, s * hc_dim + inner].to(cutlass.Float32),
                        mix,
                    )
                if tidx < num_slots and ni_l % hc == 0:
                    gC[mi_l, inner] = _triton_div(mix, cutlass.Float32(hc)).to(
                        self.element_type
                    )
            else:
                # _hc_silu_kernel: x = y / HC; x * sigmoid(x).
                z = _triton_div(y, cutlass.Float32(hc))
                silu = _ptx_f32(
                    "mul.f32 {$w0}, {$r0}, {$r1};", z, _triton_sigmoid(z)
                )
                n_l = n_base + ni_l
                if tidx < num_slots:
                    if n_l < self.rank:
                        gC[mi_l, n_l] = silu.to(self.element_type)
                    else:
                        gL[mi_l, n_l - self.rank] = y_elem

        if const_expr(self.use_pdl):
            cute.arch.griddepcontrol_launch_dependents()
