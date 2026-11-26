# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 lebeche hill-climb (fork of Merlin 3.0.2.2)
"""Opt-in (LABELLED LOSSY) W8 Tensor Core fused HyperConnection kernels for sm_8x.

Four CuTe-DSL kernels cover the full batch-size spectrum for Flash-Next
``GatedResidual`` projections (``hc = 4``, ``rank = 320``, ``hidden_size = 2560``,
``hyper_hidden_size = 10240``) using on-the-fly ``uint8b128`` quantization:

* ``CuteHcDownSiluW8TC``:
    Single-kernel split-K ``mma.sync.aligned.m16n8k16`` fused ``down(+inject)``
    GEMM + ``SiLU`` epilogue over group-256 ``uint8b128`` weights repacked as
    ``[N // 16, K // 32, 32, 4]`` ``int32`` + BF16 scales ``[N, K // 256]``.
    Handles dynamic ``1 <= M <= max_m`` (up to ``M = 32``) and both merged
    ``(336, 10240)`` (``n_total = 324``) and final-mixer ``(320, 10240)``
    (``n_total = 320``). Uses a GPU-scope ``atom.acq_rel.gpu.global.add.s32``
    arrival counter with automatic zero-reset on the last arriving split-K CTA,
    so it requires no second kernel launch and no ``memset`` under CUDA graphs.

* ``CuteHcUpGateMixW8TC``:
    ``mma.sync.aligned.m16n8k16`` fused ``up`` GEMM + ``sigmoid`` + HC stream
    ``gate_mix`` epilogue over inner-major (``[hc_dim, hc, K]``) per-row
    ``uint8b128`` weights repacked as ``[N // 32, K // 16, 32, 4]`` ``int32`` +
    BF16 scales ``[N]``. Handles dynamic ``1 <= M <= max_m`` (up to ``M = 32``).

* ``CuteUnpackDownW8`` and ``CuteUnpackUpW8SM``:
    Coalesced 128-bit dequantizers (BF16 ``[N, K]``) used for prefill
    (``M > 32``) so no duplicate BF16 weight copy is retained in VRAM.
    ``CuteUnpackUpW8SM`` writes stream-major ``[hc, hc_dim, K]`` directly so
    prefill ``F.linear`` + ``hc_gate_mix`` avoids any ``[M, 10240]`` transpose.

All kernels are sm_80/86/89 compatible (no ``elect.sync``, no PDL, no
``cp.async``) and reproduce the exact Triton epilogue PTX instructions
(``div.full.f32``, ``ex2.approx.f32``, ``fma.rn.f32``) of ``_hc_small_fused.py``.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import torch
from cuda.bindings.driver import CUstream
from cutlass import const_expr

from ._hc_small_fused import _ptx_f32, _triton_div, _triton_sigmoid

HC_W8_DOWN_GROUP_SIZE = 256


def _dequant_u8b128_to_bf16x2_pair(q_i32):
    """Dequantize 4 packed uint8b128 bytes ``[b0, b1, b2, b3]`` -> 2 BF16x2 words.

    Returns ``(bf16x2(b0-128, b2-128), bf16x2(b1-128, b3-128))``, matching
    ``_w8_skinny_gemm.py`` / Marlin ``dequant.h``.
    """
    return cute.arch.inline_ptx(
        "{"
        " .reg .b32 t0, t1, t2, t3;"
        " .reg .f32 f0, f1, f2, f3;"
        " prmt.b32 t0, {$r0}, 0x4B000000, 0x7650;"
        " prmt.b32 t1, {$r0}, 0x4B000000, 0x7652;"
        " prmt.b32 t2, {$r0}, 0x4B000000, 0x7651;"
        " prmt.b32 t3, {$r0}, 0x4B000000, 0x7653;"
        " mov.b32 f0, t0; mov.b32 f1, t1; mov.b32 f2, t2; mov.b32 f3, t3;"
        " add.f32 f0, f0, 0fCB000080;"
        " add.f32 f1, f1, 0fCB000080;"
        " add.f32 f2, f2, 0fCB000080;"
        " add.f32 f3, f3, 0fCB000080;"
        " mov.b32 t0, f0; mov.b32 t1, f1; mov.b32 t2, f2; mov.b32 t3, f3;"
        " prmt.b32 {$w0}, t0, t1, 0x7632;"
        " prmt.b32 {$w1}, t2, t3, 0x7632;"
        "}",
        write_only_types=[cutlass.Int32, cutlass.Int32],
        read_only_args=[q_i32],
    )


def _dequant_u8b128_4f32(q_i32):
    """Dequantize 4 packed uint8b128 bytes ``[b0, b1, b2, b3]`` -> 4 Float32s."""
    return cute.arch.inline_ptx(
        "{"
        " .reg .b32 t0, t1, t2, t3;"
        " prmt.b32 t0, {$r0}, 0x4B000000, 0x7650;"
        " prmt.b32 t1, {$r0}, 0x4B000000, 0x7651;"
        " prmt.b32 t2, {$r0}, 0x4B000000, 0x7652;"
        " prmt.b32 t3, {$r0}, 0x4B000000, 0x7653;"
        " mov.b32 {$w0}, t0; mov.b32 {$w1}, t1; mov.b32 {$w2}, t2; mov.b32 {$w3}, t3;"
        " add.f32 {$w0}, {$w0}, 0fCB000080;"
        " add.f32 {$w1}, {$w1}, 0fCB000080;"
        " add.f32 {$w2}, {$w2}, 0fCB000080;"
        " add.f32 {$w3}, {$w3}, 0fCB000080;"
        "}",
        write_only_types=[
            cutlass.Float32,
            cutlass.Float32,
            cutlass.Float32,
            cutlass.Float32,
        ],
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


def _atomic_add_acq_rel_i32(ptr_i64, val):
    return cute.arch.inline_ptx(
        "atom.acq_rel.gpu.global.add.s32 {$w0}, [{$r0}], {$r1};",
        write_only_types=[cutlass.Int32],
        read_only_args=[ptr_i64, val],
    )


@torch.no_grad()
def quantize_pack_hc_down_w8(
    weight: torch.Tensor, group_size: int = HC_W8_DOWN_GROUP_SIZE
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``[N, K]`` BF16 down weight (group-256 symmetric ``uint8b128``)
    and pack into ``[N // 16, K // 32, 32, 4]`` ``int32`` for ``CuteHcDownSiluW8TC``.
    """
    size_n, size_k = weight.shape
    if size_n % 16 != 0 or size_k % group_size != 0 or group_size % 32 != 0:
        raise ValueError(
            f"HC down W8 requires N%16==0, K%{group_size}==0, got {size_n}x{size_k}"
        )
    wf = weight.float().view(size_n, size_k // group_size, group_size)
    max_abs = wf.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    scales = (max_abs / 127.0).to(torch.bfloat16)
    q_i32 = (
        torch.clamp(torch.round(wf / scales.float()), -128, 127).to(torch.int32) + 128
    ).view(size_n, size_k)
    scales_out = scales.squeeze(-1).contiguous()

    # Pack [N // 16, 2(r2), 8(g), K // 32, 2(k2), 2(kh), 4(t), 2(p)]
    w = q_i32.view(size_n // 16, 2, 8, size_k // 32, 2, 2, 4, 2)
    w = w.permute(0, 3, 2, 6, 4, 5, 1, 7).contiguous()
    packed = (
        w[..., 0, 0]
        | (w[..., 1, 0] << 8)
        | (w[..., 0, 1] << 16)
        | (w[..., 1, 1] << 24)
    ).view(size_n // 16, size_k // 32, 32, 4).contiguous()
    return packed, scales_out


@torch.no_grad()
def quantize_pack_hc_up_w8(
    weight: torch.Tensor, hc: int = 4
) -> tuple[torch.Tensor, torch.Tensor]:
    """Permute stream-major ``[hc * hc_dim, K]`` BF16 up weight to inner-major,
    quantize per-row symmetric ``uint8b128``, and pack into
    ``[N // 32, K // 16, 32, 4]`` ``int32`` for ``CuteHcUpGateMixW8TC``.
    """
    size_n, size_k = weight.shape
    if size_n % 32 != 0 or size_k % 16 != 0 or size_n % hc != 0:
        raise ValueError(
            f"HC up W8 requires N%32==0, K%16==0, N%{hc}==0, got {size_n}x{size_k}"
        )
    hc_dim = size_n // hc
    # Stream-major [hc, hc_dim, K] -> inner-major [hc_dim, hc, K] -> [N, K]
    w_im = weight.view(hc, hc_dim, size_k).transpose(0, 1).contiguous().view(size_n, size_k)
    wf = w_im.float()
    max_abs = wf.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
    scales = (max_abs / 127.0).to(torch.bfloat16)
    q_i32 = (
        torch.clamp(torch.round(wf / scales.float()), -128, 127).to(torch.int32) + 128
    )
    scales_out = scales.squeeze(-1).contiguous()

    # Pack [N // 32, 2(m2), 2(r2), 8(g), K // 16, 2(kh), 4(t), 2(p)]
    w = q_i32.view(size_n // 32, 2, 2, 8, size_k // 16, 2, 4, 2)
    w = w.permute(0, 4, 3, 6, 1, 5, 2, 7).contiguous()
    packed = (
        w[..., 0, 0]
        | (w[..., 1, 0] << 8)
        | (w[..., 0, 1] << 16)
        | (w[..., 1, 1] << 24)
    ).view(size_n // 32, size_k // 16, 32, 4).contiguous()
    return packed, scales_out


class CuteHcDownSiluW8TC:
    """Single-kernel split-K Tensor Core W8 ``down(+inject) + SiLU``."""

    def __init__(
        self,
        *,
        max_m: int,
        block_size: int = 128,
        split_k: int = 5,
        hc: int = 4,
        rank: int = 320,
        static_k: int = 10240,
        group_size: int = HC_W8_DOWN_GROUP_SIZE,
    ) -> None:
        if max_m not in (4, 8, 16, 32):
            raise ValueError(f"max_m must be in (4, 8, 16, 32), got {max_m}")
        if block_size % 32 != 0:
            raise ValueError(f"block_size must be multiple of 32, got {block_size}")
        num_warps = block_size // 32
        if (static_k // group_size) % split_k != 0:
            raise ValueError(
                f"num_groups={static_k // group_size} not divisible by split_k={split_k}"
            )
        if (group_size // 32) % num_warps != 0:
            raise ValueError(
                f"group_size//32={group_size // 32} not divisible by num_warps={num_warps}"
            )
        self.max_m = max_m
        self.m_tiles = (max_m + 7) // 8
        self.block_size = block_size
        self.num_warps = num_warps
        self.split_k = split_k
        self.hc = hc
        self.rank = rank
        self.static_k = static_k
        self.group_size = group_size
        self.groups_per_split = (static_k // group_size) // split_k
        self.w_stride = num_warps | 1
        self.smem_bytes = max_m * 16 * self.w_stride * 4 + 16
        if self.smem_bytes > 48 * 1024:
            raise ValueError(f"smem {self.smem_bytes} exceeds 48 KiB")

    @cute.jit
    def __call__(
        self,
        gA_u32: cute.Tensor,  # [M, K // 2] Int32 (BF16 reinterpreted as bf16x2)
        gB_i32: cute.Tensor,  # [N // 16, K // 32, 32, 4] Int32
        gS: cute.Tensor,  # [N, K // GS] BFloat16
        gPart: cute.Tensor,  # [max_sk, max_m_ws, max_n] Float32
        gCounter: cute.Tensor,  # [max_n_tiles] Int32
        gC: cute.Tensor,  # [M, rank] BFloat16
        gL: cute.Tensor,  # [M, N] BFloat16 (contiguous; logits written at [rank, n_total))
        n_total: cutlass.Int32,  # rank + n_logits (324 or 320)
        stream: CUstream,
    ) -> None:
        m = cute.size(gC, mode=[0])
        n_tiles = cute.size(gB_i32, mode=[0])
        copy_b = cute.make_copy_atom(
            cute.nvgpu.CopyG2ROp(),
            cutlass.Int32,
            num_bits_per_copy=128,
            load_cache_mode=cute.nvgpu.LoadCacheMode.STREAMING,
        )
        self.kernel(
            gA_u32, gB_i32, gS, gPart, gCounter, gC, gL, m, n_total, copy_b
        ).launch(
            grid=[n_tiles, self.split_k, 1],
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
        gPart: cute.Tensor,
        gCounter: cute.Tensor,
        gC: cute.Tensor,
        gL: cute.Tensor,
        m_extent: cutlass.Int32,
        n_total: cutlass.Int32,
        copy_b: cute.CopyAtom,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        n_tile, sk_id, _ = cute.arch.block_idx()

        MAX_M: cutlass.Constexpr = self.max_m
        MT: cutlass.Constexpr = self.m_tiles
        NW: cutlass.Constexpr = self.num_warps
        WSTR: cutlass.Constexpr = self.w_stride
        SK: cutlass.Constexpr = self.split_k
        GPS: cutlass.Constexpr = self.groups_per_split
        K32_PER_G: cutlass.Constexpr = self.group_size // 32
        K32_PER_W: cutlass.Constexpr = K32_PER_G // NW
        TOTAL_ITERS: cutlass.Constexpr = GPS * K32_PER_W
        HC: cutlass.Constexpr = self.hc

        warp_id = tidx >> 5
        lane = tidx & 31
        group_id = lane >> 2
        tid_in_g = lane & 3

        acc = cute.make_rmem_tensor(
            cute.make_layout((MT, 4), stride=(4, 1)), cutlass.Float32
        )
        acc.fill(0.0)
        gacc = cute.make_rmem_tensor(
            cute.make_layout((MT, 4), stride=(4, 1)), cutlass.Float32
        )
        gacc.fill(0.0)
        b0 = cute.make_rmem_tensor(cute.make_layout((4,), stride=(1,)), cutlass.Int32)
        b1 = cute.make_rmem_tensor(cute.make_layout((4,), stride=(1,)), cutlass.Int32)

        k32_first = (sk_id * GPS) * K32_PER_G + warp_id * K32_PER_W
        cute.copy(copy_b, gB_i32[n_tile, k32_first, lane, None], b0)

        for it in cutlass.range_constexpr(TOTAL_ITERS):
            gi: cutlass.Constexpr = it // K32_PER_W
            ki: cutlass.Constexpr = it % K32_PER_W
            g_id = sk_id * GPS + gi
            k32 = g_id * K32_PER_G + warp_id * K32_PER_W + ki
            cur = b0 if const_expr(it % 2 == 0) else b1
            nxt = b1 if const_expr(it % 2 == 0) else b0
            if const_expr(it + 1 < TOTAL_ITERS):
                gi_n: cutlass.Constexpr = (it + 1) // K32_PER_W
                ki_n: cutlass.Constexpr = (it + 1) % K32_PER_W
                k32_n = (sk_id * GPS + gi_n) * K32_PER_G + warp_id * K32_PER_W + ki_n
                cute.copy(copy_b, gB_i32[n_tile, k32_n, lane, None], nxt)

            base_u32 = k32 * 16 + tid_in_g
            w0, w1 = _dequant_u8b128_to_bf16x2_pair(cur[0])
            w2, w3 = _dequant_u8b128_to_bf16x2_pair(cur[1])
            w4, w5 = _dequant_u8b128_to_bf16x2_pair(cur[2])
            w6, w7 = _dequant_u8b128_to_bf16x2_pair(cur[3])
            for mt in cutlass.range_constexpr(MT):
                row = mt * 8 + group_id
                valid = row < m_extent
                srow = row if valid else cutlass.Int32(0)
                a0 = gA_u32[srow, base_u32] if valid else cutlass.Int32(0)
                a1 = gA_u32[srow, base_u32 + 4] if valid else cutlass.Int32(0)
                a2 = gA_u32[srow, base_u32 + 8] if valid else cutlass.Int32(0)
                a3 = gA_u32[srow, base_u32 + 12] if valid else cutlass.Int32(0)
                gacc[mt, 0], gacc[mt, 1], gacc[mt, 2], gacc[mt, 3] = _mma_m16n8k16_bf16(
                    w0, w1, w2, w3, a0, a1,
                    gacc[mt, 0], gacc[mt, 1], gacc[mt, 2], gacc[mt, 3],
                )
                gacc[mt, 0], gacc[mt, 1], gacc[mt, 2], gacc[mt, 3] = _mma_m16n8k16_bf16(
                    w4, w5, w6, w7, a2, a3,
                    gacc[mt, 0], gacc[mt, 1], gacc[mt, 2], gacc[mt, 3],
                )

            if const_expr(ki == K32_PER_W - 1):
                sc0 = gS[n_tile * 16 + group_id, g_id].to(cutlass.Float32)
                sc1 = gS[n_tile * 16 + group_id + 8, g_id].to(cutlass.Float32)
                for mt in cutlass.range_constexpr(MT):
                    acc[mt, 0] = acc[mt, 0] + gacc[mt, 0] * sc0
                    acc[mt, 1] = acc[mt, 1] + gacc[mt, 1] * sc0
                    acc[mt, 2] = acc[mt, 2] + gacc[mt, 2] * sc1
                    acc[mt, 3] = acc[mt, 3] + gacc[mt, 3] * sc1
                    gacc[mt, 0] = cutlass.Float32(0.0)
                    gacc[mt, 1] = cutlass.Float32(0.0)
                    gacc[mt, 2] = cutlass.Float32(0.0)
                    gacc[mt, 3] = cutlass.Float32(0.0)

        smem = cutlass.utils.SmemAllocator()
        partials = smem.allocate_tensor(
            cutlass.Float32,
            cute.make_layout((MAX_M, 16, WSTR), stride=(16 * WSTR, WSTR, 1)),
            byte_alignment=16,
        )
        flag_s = smem.allocate_tensor(
            cutlass.Int32, cute.make_layout((4,), stride=(1,)), byte_alignment=16
        )

        for mt in cutlass.range_constexpr(MT):
            m0 = mt * 8 + tid_in_g * 2
            m1 = m0 + 1
            if m0 < MAX_M:
                partials[m0, group_id, warp_id] = acc[mt, 0]
                partials[m0, group_id + 8, warp_id] = acc[mt, 2]
            if m1 < MAX_M:
                partials[m1, group_id, warp_id] = acc[mt, 1]
                partials[m1, group_id + 8, warp_id] = acc[mt, 3]
        cute.arch.sync_threads()

        n_out = n_tile * 16 + lane
        for mi_it in cutlass.range_constexpr((MAX_M + NW - 1) // NW):
            mi = warp_id + mi_it * NW
            if mi < m_extent and lane < 16:
                tot = cutlass.Float32(0.0)
                for w in cutlass.range_constexpr(NW):
                    tot = tot + partials[mi, lane, w]
                gPart[sk_id, mi, n_out] = tot
        cute.arch.sync_threads()

        if tidx == 0:
            c_ptr = gCounter.iterator.toint() + cutlass.Int64(n_tile) * cutlass.Int64(4)
            old = _atomic_add_acq_rel_i32(c_ptr, cutlass.Int32(1))
            is_last = cutlass.Int32(1) if old == SK - 1 else cutlass.Int32(0)
            if is_last == 1:
                gCounter[n_tile] = cutlass.Int32(0)
            flag_s[0] = is_last
        cute.arch.sync_threads()

        if flag_s[0] == 1:
            for mi_it in cutlass.range_constexpr((MAX_M + NW - 1) // NW):
                mi = warp_id + mi_it * NW
                if mi < m_extent and lane < 16 and n_out < n_total:
                    tot = cutlass.Float32(0.0)
                    for s in cutlass.range_constexpr(SK):
                        tot = tot + gPart[s, mi, n_out]
                    y_elem = cutlass.Float32(tot).to(cutlass.BFloat16)
                    y = y_elem.to(cutlass.Float32)
                    z = _triton_div(y, cutlass.Float32(HC))
                    silu = _ptx_f32(
                        "mul.f32 {$w0}, {$r0}, {$r1};", z, _triton_sigmoid(z)
                    )
                    if n_out < self.rank:
                        gC[mi, n_out] = silu.to(cutlass.BFloat16)
                    else:
                        gL[mi, n_out] = y_elem


class CuteHcUpGateMixW8TC:
    """Tensor Core W8 fused ``up + sigmoid + gate_mix`` over inner-major weights."""

    def __init__(
        self,
        *,
        max_m: int,
        block_size: int = 160,
        hc: int = 4,
        static_k: int = 320,
    ) -> None:
        if max_m not in (4, 8, 16, 32):
            raise ValueError(f"max_m must be in (4, 8, 16, 32), got {max_m}")
        if block_size % 32 != 0:
            raise ValueError(f"block_size must be multiple of 32, got {block_size}")
        num_warps = block_size // 32
        if (static_k // 16) % num_warps != 0:
            raise ValueError(
                f"static_k//16={static_k // 16} not divisible by num_warps={num_warps}"
            )
        self.max_m = max_m
        self.m_tiles = (max_m + 7) // 8
        self.block_size = block_size
        self.num_warps = num_warps
        self.hc = hc
        self.static_k = static_k
        self.w_stride = num_warps | 1
        self.smem_bytes = max_m * 32 * self.w_stride * 4
        if self.smem_bytes > 48 * 1024:
            raise ValueError(f"smem {self.smem_bytes} exceeds 48 KiB")

    @cute.jit
    def __call__(
        self,
        gA_u32: cute.Tensor,  # [M, K // 2] Int32
        gB_i32: cute.Tensor,  # [N // 32, K // 16, 32, 4] Int32
        gS: cute.Tensor,  # [N] BFloat16 (inner-major)
        gX: cute.Tensor,  # [M, N] BFloat16 (stream-major)
        gC: cute.Tensor,  # [M, N // hc] BFloat16
        stream: CUstream,
    ) -> None:
        m = cute.size(gC, mode=[0])
        hc_dim = cute.size(gC, mode=[1])
        copy_b = cute.make_copy_atom(
            cute.nvgpu.CopyG2ROp(),
            cutlass.Int32,
            num_bits_per_copy=128,
            load_cache_mode=cute.nvgpu.LoadCacheMode.STREAMING,
        )
        self.kernel(gA_u32, gB_i32, gS, gX, gC, m, hc_dim, copy_b).launch(
            grid=[hc_dim // 8, 1, 1],
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
        gX: cute.Tensor,
        gC: cute.Tensor,
        m_extent: cutlass.Int32,
        hc_dim: cutlass.Int32,
        copy_b: cute.CopyAtom,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        MAX_M: cutlass.Constexpr = self.max_m
        MT: cutlass.Constexpr = self.m_tiles
        NW: cutlass.Constexpr = self.num_warps
        WSTR: cutlass.Constexpr = self.w_stride
        HC: cutlass.Constexpr = self.hc
        N_ITERS: cutlass.Constexpr = (self.static_k // 16) // NW

        warp_id = tidx >> 5
        lane = tidx & 31
        group_id = lane >> 2
        tid_in_g = lane & 3

        acc = cute.make_rmem_tensor(
            cute.make_layout((MT, 2, 4), stride=(8, 4, 1)), cutlass.Float32
        )
        acc.fill(0.0)
        b0 = cute.make_rmem_tensor(cute.make_layout((4,), stride=(1,)), cutlass.Int32)
        b1 = cute.make_rmem_tensor(cute.make_layout((4,), stride=(1,)), cutlass.Int32)

        cute.copy(copy_b, gB_i32[bidx, warp_id, lane, None], b0)
        for it in cutlass.range_constexpr(N_ITERS):
            kt = warp_id + it * NW
            cur = b0 if const_expr(it % 2 == 0) else b1
            nxt = b1 if const_expr(it % 2 == 0) else b0
            if const_expr(it + 1 < N_ITERS):
                cute.copy(copy_b, gB_i32[bidx, kt + NW, lane, None], nxt)
            k_u32 = kt * 8 + tid_in_g
            w0, w1 = _dequant_u8b128_to_bf16x2_pair(cur[0])
            w2, w3 = _dequant_u8b128_to_bf16x2_pair(cur[1])
            w4, w5 = _dequant_u8b128_to_bf16x2_pair(cur[2])
            w6, w7 = _dequant_u8b128_to_bf16x2_pair(cur[3])
            for mt in cutlass.range_constexpr(MT):
                row = mt * 8 + group_id
                valid = row < m_extent
                srow = row if valid else cutlass.Int32(0)
                a_b0 = gA_u32[srow, k_u32] if valid else cutlass.Int32(0)
                a_b1 = gA_u32[srow, k_u32 + 4] if valid else cutlass.Int32(0)
                acc[mt, 0, 0], acc[mt, 0, 1], acc[mt, 0, 2], acc[mt, 0, 3] = (
                    _mma_m16n8k16_bf16(
                        w0, w1, w2, w3, a_b0, a_b1,
                        acc[mt, 0, 0], acc[mt, 0, 1], acc[mt, 0, 2], acc[mt, 0, 3],
                    )
                )
                acc[mt, 1, 0], acc[mt, 1, 1], acc[mt, 1, 2], acc[mt, 1, 3] = (
                    _mma_m16n8k16_bf16(
                        w4, w5, w6, w7, a_b0, a_b1,
                        acc[mt, 1, 0], acc[mt, 1, 1], acc[mt, 1, 2], acc[mt, 1, 3],
                    )
                )

        smem = cutlass.utils.SmemAllocator()
        partials = smem.allocate_tensor(
            cutlass.Float32,
            cute.make_layout((MAX_M, 32, WSTR), stride=(32 * WSTR, WSTR, 1)),
            byte_alignment=16,
        )
        for mt in cutlass.range_constexpr(MT):
            m0 = mt * 8 + tid_in_g * 2
            m1 = m0 + 1
            for tile in cutlass.range_constexpr(2):
                n0 = tile * 16 + group_id
                n1 = n0 + 8
                if m0 < MAX_M:
                    partials[m0, n0, warp_id] = acc[mt, tile, 0]
                    partials[m0, n1, warp_id] = acc[mt, tile, 2]
                if m1 < MAX_M:
                    partials[m1, n0, warp_id] = acc[mt, tile, 1]
                    partials[m1, n1, warp_id] = acc[mt, tile, 3]
        cute.arch.sync_threads()

        inner_idx = bidx * 8 + (lane >> 2)
        stream_id = lane & 3
        sc = gS[bidx * 32 + lane].to(cutlass.Float32)
        for mi_it in cutlass.range_constexpr((MAX_M + NW - 1) // NW):
            mi = warp_id + mi_it * NW
            if mi < m_extent:
                total = cutlass.Float32(0.0)
                for w in cutlass.range_constexpr(NW):
                    total = total + partials[mi, lane, w]
                y_elem = cutlass.Float32(total * sc).to(cutlass.BFloat16)
                sig = _triton_sigmoid(y_elem.to(cutlass.Float32))
                mix = cutlass.Float32(0.0)
                for s in cutlass.range_constexpr(HC):
                    sig_s = sig
                    if const_expr(s > 0):
                        sig_s = cute.arch.shuffle_sync_down(sig, s)
                    mix = _ptx_f32(
                        "fma.rn.f32 {$w0}, {$r0}, {$r1}, {$r2};",
                        sig_s,
                        gX[mi, s * hc_dim + inner_idx].to(cutlass.Float32),
                        mix,
                    )
                if stream_id == 0:
                    gC[mi, inner_idx] = _triton_div(mix, cutlass.Float32(HC)).to(
                        cutlass.BFloat16
                    )


class CuteUnpackDownW8:
    """Coalesced 128-bit dequantizer for packed HC down weights (prefill)."""

    def __init__(
        self, *, static_k: int = 10240, group_size: int = HC_W8_DOWN_GROUP_SIZE
    ) -> None:
        self.static_k = static_k
        self.group_size = group_size

    @cute.jit
    def __call__(
        self, gQ: cute.Tensor, gS: cute.Tensor, gOut: cute.Tensor, stream: CUstream
    ) -> None:
        n_tiles = cute.size(gQ, mode=[0])
        self.kernel(gQ, gS, gOut).launch(
            grid=[n_tiles, self.static_k // 128, 1],
            block=[128, 1, 1],
            smem=16 * 128 * 2,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self, gQ: cute.Tensor, gS: cute.Tensor, gOut: cute.Tensor
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        n_tile, k128, _ = cute.arch.block_idx()
        warp_id = tidx >> 5
        lane = tidx & 31
        g = lane >> 2
        t = lane & 3
        k32 = k128 * 4 + warp_id
        g_id = (k32 * 32) // self.group_size

        smem = cutlass.utils.SmemAllocator()
        tile = smem.allocate_tensor(
            cutlass.BFloat16,
            cute.make_layout((16, 128), stride=(128, 1)),
            byte_alignment=16,
        )
        s0 = gS[n_tile * 16 + g, g_id].to(cutlass.Float32)
        s1 = gS[n_tile * 16 + g + 8, g_id].to(cutlass.Float32)

        for w_idx in cutlass.range_constexpr(4):
            word = gQ[n_tile, k32, lane, w_idx]
            f0, f1, f2, f3 = _dequant_u8b128_4f32(word)
            k_loc = warp_id * 32 + (w_idx >> 1) * 16 + (w_idx & 1) * 8 + t * 2
            tile[g, k_loc] = (f0 * s0).to(cutlass.BFloat16)
            tile[g + 8, k_loc] = (f1 * s1).to(cutlass.BFloat16)
            tile[g, k_loc + 1] = (f2 * s0).to(cutlass.BFloat16)
            tile[g + 8, k_loc + 1] = (f3 * s1).to(cutlass.BFloat16)
        cute.arch.sync_threads()

        row = tidx >> 3
        col_base = (tidx & 7) * 16
        copy_s2g = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), cutlass.BFloat16, num_bits_per_copy=128
        )
        tmp = cute.make_rmem_tensor(
            cute.make_layout((8,), stride=(1,)), cutlass.BFloat16
        )
        gOut_v8 = cute.logical_divide(gOut, (None, 8))
        tile_v8 = cute.logical_divide(tile, (None, 8))
        for j in cutlass.range_constexpr(2):
            c8 = (col_base >> 3) + j
            cute.copy(copy_s2g, tile_v8[row, (None, c8)], tmp)
            cute.copy(copy_s2g, tmp, gOut_v8[n_tile * 16 + row, (None, k128 * 16 + c8)])


class CuteUnpackUpW8SM:
    """Coalesced 128-bit dequantizer for packed HC up weights directly to
    stream-major ``[hc * hc_dim, K]`` BF16 (prefill; zero activation transpose).
    """

    def __init__(self, *, static_k: int = 320, hc: int = 4) -> None:
        self.static_k = static_k
        self.hc = hc

    @cute.jit
    def __call__(
        self, gQ: cute.Tensor, gS: cute.Tensor, gOut: cute.Tensor, stream: CUstream
    ) -> None:
        n32 = cute.size(gQ, mode=[0])
        hc_dim = (n32 * 32) // self.hc
        self.kernel(gQ, gS, gOut, hc_dim).launch(
            grid=[n32, self.static_k // 80, 1],
            block=[160, 1, 1],
            smem=32 * 80 * 2,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        gQ: cute.Tensor,
        gS: cute.Tensor,
        gOut: cute.Tensor,
        hc_dim: cutlass.Int32,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        n32, k80, _ = cute.arch.block_idx()
        warp_id = tidx >> 5
        lane = tidx & 31
        g = lane >> 2
        t = lane & 3
        kt = k80 * 5 + warp_id

        smem = cutlass.utils.SmemAllocator()
        tile = smem.allocate_tensor(
            cutlass.BFloat16,
            cute.make_layout((32, 80), stride=(80, 1)),
            byte_alignment=16,
        )
        for w_idx in cutlass.range_constexpr(4):
            word = gQ[n32, kt, lane, w_idx]
            m2 = w_idx >> 1
            kh = w_idx & 1
            r0 = m2 * 16 + g
            r1 = r0 + 8
            s0 = gS[n32 * 32 + r0].to(cutlass.Float32)
            s1 = gS[n32 * 32 + r1].to(cutlass.Float32)
            f0, f1, f2, f3 = _dequant_u8b128_4f32(word)
            k_loc = warp_id * 16 + kh * 8 + t * 2
            tile[r0, k_loc] = (f0 * s0).to(cutlass.BFloat16)
            tile[r1, k_loc] = (f1 * s1).to(cutlass.BFloat16)
            tile[r0, k_loc + 1] = (f2 * s0).to(cutlass.BFloat16)
            tile[r1, k_loc + 1] = (f3 * s1).to(cutlass.BFloat16)
        cute.arch.sync_threads()

        # 32 inner-major rows x 80 cols = 320 vec8 chunks; 160 threads -> 2 stores/thread.
        # Map inner-major row r = inner_loc * 4 + stream -> stream-major row = stream * hc_dim + inner_idx.
        copy_s2g = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), cutlass.BFloat16, num_bits_per_copy=128
        )
        tmp = cute.make_rmem_tensor(
            cute.make_layout((8,), stride=(1,)), cutlass.BFloat16
        )
        gOut_v8 = cute.logical_divide(gOut, (None, 8))
        tile_v8 = cute.logical_divide(tile, (None, 8))
        for j in cutlass.range_constexpr(2):
            idx = tidx + j * 160
            row = idx // 10
            c8 = idx % 10
            inner_idx = n32 * 8 + (row >> 2)
            stream_id = row & 3
            out_row = stream_id * hc_dim + inner_idx
            cute.copy(copy_s2g, tile_v8[row, (None, c8)], tmp)
            cute.copy(copy_s2g, tmp, gOut_v8[out_row, (None, k80 * 10 + c8)])


__all__ = [
    "HC_W8_DOWN_GROUP_SIZE",
    "CuteHcDownSiluW8TC",
    "CuteHcUpGateMixW8TC",
    "CuteUnpackDownW8",
    "CuteUnpackUpW8SM",
    "quantize_pack_hc_down_w8",
    "quantize_pack_hc_up_w8",
]
