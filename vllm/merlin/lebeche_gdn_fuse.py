# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 lebeche hill-climb (fork of Merlin 3.0.2.2)
"""Small-batch (M <= 8) fused GDN RecoverSSM decode kernel on sm_80/86/89.

Stock Flash-Next (Qwen4Exp) GDN decode executes 7 kernels per layer between
``in_proj_qkvz``/``in_proj_ba`` and ``out_proj``:

1. ``triton_poi_fused_zeros_0`` (``core_attn_out = torch.zeros(...)``)
2. ``direct_copy_kernel`` (``b.contiguous()``)
3. ``direct_copy_kernel`` (``a.contiguous()``)
4. ``_causal_conv1d_update_kernel``
5. ``_gdn_recoverssm_verify_kernel``
6. ``direct_copy_kernel`` (``output_gate.reshape(-1, 128)``)
7. ``layer_norm_fwd_kernel`` (``RMSNormGated``)

When ``MERLIN_LEBECHE_GDN_FUSE`` is on (and ``MERLIN_LEBECHE_SM86_LL`` is on
on an sm_80/86/89 GPU), ``forward_cuda`` allocates ``core_attn_out`` with
``torch.empty`` instead of ``torch.zeros`` (a static boot-time choice, never
branched on ``M``), and inside the opaque
``vllm::qwen_gdn_attention_core_fused_norm_packed`` custom op, small-batch
RecoverSSM spec-only decode (``1 <= M <= MAX_M``) runs in 2 kernels with zero
intermediate activation buffers:

1. Stock ``causal_conv1d_update`` in-place on ``mixed_qkvz[:M_act, :qkv_dim]``;
2. ``CuteGdnVerifyNorm`` (CuTe DSL): reads ``q, k, v, z`` directly from
   ``mixed_qkvz`` and ``b, a`` directly from ``ba`` (eliminating all three
   ``.contiguous()``/``.reshape()`` copies), overlaps streaming ``LDG.E.128``
   checkpoint-state loads with vectorized ``bf16x4`` Q/K L2-norm and A/B
   gating into shared memory, runs sub-warp-per-row (``LPR=4``, ``BV=32``)
   recurrence with pairwise binary-tree reductions matching Triton butterfly
   order, writes coalesced ``gOut`` and ``gReplay`` tiles, uses a
   self-resetting per-``(b, hv)`` atomic counter with ``fence_acq_rel_gpu``
   and ``ld.global.cg`` L2-coherent loads so the last CTA for each head runs
   vectorized ``bf16x4`` ``RMSNormGated`` in-place, and zeroes any null blocks
   and CUDA-graph padded tokens ``[num_actual_tokens, num_padded_tokens)``.

Above ``MAX_M`` or whenever any runtime precondition is not met (prefill,
mixed batch, non-RecoverSSM, unsupported shapes/dtypes), the op zeroes
``core_attn_out`` and falls back to the stock sequence bit-identically.
"""


from functools import partial
from typing import Any

import torch

from vllm.merlin.envs import read_registered
from vllm.merlin.lebeche_sm86 import sm86_ll_active
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

GDN_FUSE_SWITCH: bool = bool(read_registered("MERLIN_LEBECHE_GDN_FUSE"))

MAX_M: int = 8
_HEAD_DIM: int = 128
_LPR: int = 4  # 4 lanes/row -> 8 rows/warp, 4 warps/CTA -> BV=32, 4 v-blocks/head
_MAX_SPEC_LEN: int = 8

# (H, HV, spec_len, is_sigmoid, alog_dtype, dt_bias_dtype, norm_w_dtype)
_CompileKey = tuple[int, int, int, bool, torch.dtype, torch.dtype, torch.dtype]


def gdn_fuse_active() -> bool:
    """True when the fused small-M GDN decode path should be active."""
    return GDN_FUSE_SWITCH and sm86_ll_active()


def _torch_to_cutlass_dtype(dt: torch.dtype):
    import cutlass

    if dt == torch.bfloat16:
        return cutlass.BFloat16
    if dt == torch.float32:
        return cutlass.Float32
    raise ValueError(f"unsupported dtype for GDN fuse: {dt}")


def _build_cute_gdn_verify_norm_cls():
    """Lazily define the CuTe DSL kernel class when CuTe DSL is imported."""
    import cutlass
    import cutlass.cute as cute
    from cuda.bindings.driver import CUstream
    from cutlass import const_expr

    def _ptx_f32(code: str, *args):
        return cute.arch.inline_ptx(
            code, write_only_types=[cutlass.Float32], read_only_args=list(args)
        )

    def _triton_div(a, b):
        return _ptx_f32("div.full.f32 {$w0}, {$r0}, {$r1};", a, b)

    def _triton_exp(x):
        # 0f3FB8AA3B = log2(e)
        t = _ptx_f32("mul.f32 {$w0}, {$r0}, 0f3FB8AA3B;", x)
        return _ptx_f32("ex2.approx.f32 {$w0}, {$r0};", t)

    def _triton_neg_exp(x):
        # 0fBFB8AA3B = -log2(e)
        t = _ptx_f32("mul.f32 {$w0}, {$r0}, 0fBFB8AA3B;", x)
        return _ptx_f32("ex2.approx.f32 {$w0}, {$r0};", t)

    def _triton_log(x):
        # 0f3F317218 = ln(2)
        l2 = _ptx_f32("lg2.approx.f32 {$w0}, {$r0};", x)
        return _ptx_f32("mul.f32 {$w0}, {$r0}, 0f3F317218;", l2)

    def _triton_rsqrt(x):
        return _ptx_f32("rsqrt.approx.ftz.f32 {$w0}, {$r0};", x)

    def _triton_sigmoid(x):
        e = _triton_neg_exp(x)
        d = _ptx_f32("add.f32 {$w0}, {$r0}, 0f3F800000;", e)
        return _triton_div(cutlass.Float32(1.0), d)

    def _mul_nf(a, b):
        return _ptx_f32("mul.rn.f32 {$w0}, {$r0}, {$r1};", a, b)

    def _warp_sum32(val):
        for mask in (16, 8, 4, 2, 1):
            val = val + cute.arch.shuffle_sync_bfly(val, offset=mask)
        return val

    class CuteGdnVerifyNorm:
        """Fused GDN RecoverSSM verify + RMSNormGated + padding zeroing."""

        def __init__(
            self,
            *,
            H: int,
            HV: int,
            spec_len: int,
            is_sigmoid: bool,
            lpr: int = _LPR,
        ) -> None:
            assert HV % H == 0
            assert lpr in (4, 8)
            self.H = H
            self.HV = HV
            self.K = _HEAD_DIM
            self.V = _HEAD_DIM
            self.ratio = HV // H
            self.qkv_dim = 2 * H * self.K + HV * self.V
            self.lpr = lpr
            self.nw = 4
            self.subwarps_per_warp = 32 // lpr
            self.bv = self.nw * self.subwarps_per_warp
            self.v4l = self.K // (lpr * 4)
            self.n_vblocks = self.V // self.bv
            self.spec_len = spec_len
            self.is_sigmoid = is_sigmoid

        @cute.jit
        def __call__(
            self,
            gQkvz: cute.Tensor,    # [M_pad, QKVZ_DIM] bf16
            gBa: cute.Tensor,      # [M_pad, BA_DIM] bf16
            gAlog: cute.Tensor,    # [HV] f32/bf16
            gDtBias: cute.Tensor,  # [HV] bf16/f32
            gState: cute.Tensor,   # [blocks, HV, 128, 128] f32
            gReplay: cute.Tensor,  # [blocks, HV, SPEC_LEN, 257] f32
            gNormW: cute.Tensor,   # [128] bf16/f32
            gOut: cute.Tensor,     # [M_pad, HV, 128] bf16
            gCtr: cute.Tensor,     # [MAX_M, HV] i32 (zero-init, self-resetting)
            gQsl: cute.Tensor,     # [B + 1] i32
            gSidx: cute.Tensor,    # [B] i32
            scale: cutlass.Float32,
            eps: cutlass.Float32,
            ba_b_off: cutlass.Int32,
            ba_a_off: cutlass.Int32,
            null_block_id: cutlass.Int32,
            num_actual_tokens: cutlass.Int32,
            num_padded_tokens: cutlass.Int32,
            batch: cutlass.Int32,
            stream: CUstream,
        ) -> None:
            copy_f4_state = cute.make_copy_atom(
                cute.nvgpu.CopyG2ROp(),
                cutlass.Float32,
                num_bits_per_copy=128,
                load_cache_mode=cute.nvgpu.LoadCacheMode.STREAMING,
            )
            copy_bf16_ca = cute.make_copy_atom(
                cute.nvgpu.CopyG2ROp(),
                cutlass.BFloat16,
                num_bits_per_copy=64,
                load_cache_mode=cute.nvgpu.LoadCacheMode.ALWAYS,
            )
            copy_bf16_cg = cute.make_copy_atom(
                cute.nvgpu.CopyG2ROp(),
                cutlass.BFloat16,
                num_bits_per_copy=64,
                load_cache_mode=cute.nvgpu.LoadCacheMode.GLOBAL,
            )
            copy_bf16_st = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                cutlass.BFloat16,
                num_bits_per_copy=64,
            )
            smem_bytes = (
                2 * self.spec_len * self.K * 4
                + 2 * self.spec_len * self.bv * 4
                + self.spec_len * self.bv * 2
                + 128
            )
            self.kernel(
                gQkvz,
                gBa,
                gAlog,
                gDtBias,
                gState,
                gReplay,
                gNormW,
                gOut,
                gCtr,
                gQsl,
                gSidx,
                scale,
                eps,
                ba_b_off,
                ba_a_off,
                null_block_id,
                num_actual_tokens,
                num_padded_tokens,
                copy_f4_state,
                copy_bf16_ca,
                copy_bf16_cg,
                copy_bf16_st,
            ).launch(
                grid=[self.n_vblocks, batch, self.HV],
                block=[128, 1, 1],
                smem=smem_bytes,
                stream=stream,
            )

        @cute.kernel
        def kernel(
            self,
            gQkvz: cute.Tensor,
            gBa: cute.Tensor,
            gAlog: cute.Tensor,
            gDtBias: cute.Tensor,
            gState: cute.Tensor,
            gReplay: cute.Tensor,
            gNormW: cute.Tensor,
            gOut: cute.Tensor,
            gCtr: cute.Tensor,
            gQsl: cute.Tensor,
            gSidx: cute.Tensor,
            scale: cutlass.Float32,
            eps: cutlass.Float32,
            ba_b_off: cutlass.Int32,
            ba_a_off: cutlass.Int32,
            null_block_id: cutlass.Int32,
            num_actual_tokens: cutlass.Int32,
            num_padded_tokens: cutlass.Int32,
            copy_f4_state: cute.CopyAtom,
            copy_bf16_ca: cute.CopyAtom,
            copy_bf16_cg: cute.CopyAtom,
            copy_bf16_st: cute.CopyAtom,
        ) -> None:
            tidx, _, _ = cute.arch.thread_idx()
            pid_v, pid_b, pid_hv = cute.arch.block_idx()

            H_c: cutlass.Constexpr = self.H
            HV_c: cutlass.Constexpr = self.HV
            K_c: cutlass.Constexpr = self.K
            V_c: cutlass.Constexpr = self.V
            RATIO: cutlass.Constexpr = self.ratio
            QKV_D: cutlass.Constexpr = self.qkv_dim
            LPR: cutlass.Constexpr = self.lpr
            V4L: cutlass.Constexpr = self.v4l
            BV: cutlass.Constexpr = self.bv
            NVB: cutlass.Constexpr = self.n_vblocks
            SPEC: cutlass.Constexpr = self.spec_len

            subwarp_id = tidx // LPR
            sublane_id = tidx % LPR
            warp_id = tidx // 32
            lane_id = tidx % 32
            pid_h = pid_hv // RATIO

            sidx = gSidx[pid_b]
            bos = gQsl[pid_b]
            eos = gQsl[pid_b + 1]
            qlen = eos - bos
            if qlen > SPEC:
                qlen = cutlass.Int32(SPEC)
            v_base = pid_v * BV
            vr = v_base + subwarp_id

            smem = cutlass.memory.SmemAllocator()
            s_q = smem.allocate_tensor(
                cutlass.Float32,
                cute.make_layout((SPEC, K_c), stride=(K_c, 1)),
                byte_alignment=16,
            )
            s_k = smem.allocate_tensor(
                cutlass.Float32,
                cute.make_layout((SPEC, K_c), stride=(K_c, 1)),
                byte_alignment=16,
            )
            s_corr = smem.allocate_tensor(
                cutlass.Float32,
                cute.make_layout((SPEC, BV), stride=(BV, 1)),
                byte_alignment=16,
            )
            s_out = smem.allocate_tensor(
                cutlass.BFloat16,
                cute.make_layout((SPEC, BV), stride=(BV, 1)),
                byte_alignment=16,
            )
            s_v = smem.allocate_tensor(
                cutlass.Float32,
                cute.make_layout((SPEC, BV), stride=(BV, 1)),
                byte_alignment=16,
            )
            s_dec = smem.allocate_tensor(
                cutlass.Float32, cute.make_layout((SPEC,)), byte_alignment=16
            )
            s_bet = smem.allocate_tensor(
                cutlass.Float32, cute.make_layout((SPEC,)), byte_alignment=16
            )
            s_flag = smem.allocate_tensor(
                cutlass.Int32, cute.make_layout((4,)), byte_alignment=16
            )

            # Zero CUDA-graph padded tokens [live_actual_tokens, num_padded_tokens).
            # Read gQsl[batch] from device memory so FULL CUDA graph replays with
            # fewer active tokens than captured never leave torch.empty() garbage.
            if pid_b == 0 and pid_v == 0:
                _, grid_b, _ = cute.arch.grid_dim()
                pt = gQsl[grid_b]
                if num_actual_tokens < pt:
                    pt = num_actual_tokens
                if pt < 0:
                    pt = cutlass.Int32(0)
                while pt < num_padded_tokens:
                    gOut[pt, pid_hv, tidx] = cutlass.BFloat16(0.0)
                    pt = pt + 1

            if sidx <= null_block_id:
                if tidx < BV:
                    for t in range(SPEC):
                        if t < qlen and bos + t < num_padded_tokens:
                            gOut[bos + t, pid_hv, v_base + tidx] = cutlass.BFloat16(0.0)
            else:
                # 1. Issue streaming 128-bit checkpoint state loads first so DRAM
                #    latency overlaps with Q/K L2-norm and A/B gating below.
                s_reg = cute.make_rmem_tensor(
                    cute.make_layout((V4L, 4), stride=(4, 1)), cutlass.Float32
                )
                for c in cutlass.range_constexpr(V4L):
                    tile_idx = c * LPR + sublane_id
                    cute.copy(
                        copy_f4_state,
                        cute.local_tile(
                            gState[sidx, pid_hv, vr, None], (4,), (tile_idx,)
                        ),
                        s_reg[c, None],
                    )

                # 2. Vectorized bf16x4 Q/K L2-norm + V load + A/B gating into SMEM
                neg_A = -_triton_exp(cutlass.Float32(gAlog[pid_hv]))
                dt_b = cutlass.Float32(gDtBias[pid_hv])
                q_tile_id = pid_h * 32 + lane_id
                k_tile_id = (H_c + pid_h) * 32 + lane_id
                tmp4 = cute.make_rmem_tensor((4,), cutlass.BFloat16)

                tw = warp_id
                while tw < qlen:
                    tok = bos + tw
                    cute.copy(
                        copy_bf16_ca,
                        cute.local_tile(gQkvz[tok, None], (4,), (q_tile_id,)),
                        tmp4,
                    )
                    q0 = cutlass.Float32(tmp4[0])
                    q1 = cutlass.Float32(tmp4[1])
                    q2 = cutlass.Float32(tmp4[2])
                    q3 = cutlass.Float32(tmp4[3])
                    qss = (_mul_nf(q0, q0) + _mul_nf(q1, q1)) + (
                        _mul_nf(q2, q2) + _mul_nf(q3, q3)
                    )
                    qinv = _triton_rsqrt(_warp_sum32(qss) + cutlass.Float32(1e-6)) * scale
                    s_q[tw, lane_id * 4 + 0] = q0 * qinv
                    s_q[tw, lane_id * 4 + 1] = q1 * qinv
                    s_q[tw, lane_id * 4 + 2] = q2 * qinv
                    s_q[tw, lane_id * 4 + 3] = q3 * qinv

                    cute.copy(
                        copy_bf16_ca,
                        cute.local_tile(gQkvz[tok, None], (4,), (k_tile_id,)),
                        tmp4,
                    )
                    k0 = cutlass.Float32(tmp4[0])
                    k1 = cutlass.Float32(tmp4[1])
                    k2 = cutlass.Float32(tmp4[2])
                    k3 = cutlass.Float32(tmp4[3])
                    kss = (_mul_nf(k0, k0) + _mul_nf(k1, k1)) + (
                        _mul_nf(k2, k2) + _mul_nf(k3, k3)
                    )
                    kinv = _triton_rsqrt(_warp_sum32(kss) + cutlass.Float32(1e-6))
                    s_k[tw, lane_id * 4 + 0] = k0 * kinv
                    s_k[tw, lane_id * 4 + 1] = k1 * kinv
                    s_k[tw, lane_id * 4 + 2] = k2 * kinv
                    s_k[tw, lane_id * 4 + 3] = k3 * kinv

                    if lane_id < BV:
                        s_v[tw, lane_id] = cutlass.Float32(
                            gQkvz[tok, 2 * H_c * K_c + pid_hv * V_c + v_base + lane_id]
                        )
                    if lane_id == 0:
                        av = cutlass.Float32(gBa[tok, ba_a_off + pid_hv])
                        bv_s = cutlass.Float32(gBa[tok, ba_b_off + pid_hv])
                        xv = av + dt_b
                        sp = (
                            _triton_log(cutlass.Float32(1.0) + _triton_exp(xv))
                            if xv <= cutlass.Float32(20.0)
                            else xv
                        )
                        s_dec[tw] = _triton_exp(neg_A * sp)
                        s_bet[tw] = _triton_sigmoid(bv_s)
                    tw = tw + 4
                cute.arch.barrier()

                # 3. Delta-rule verify recurrence over T=0..qlen-1 (zero global loads)
                for t in range(SPEC):
                    if t < qlen:
                        decay = s_dec[t]
                        beta = s_bet[t]
                        p = cute.make_rmem_tensor((V4L,), cutlass.Float32)
                        for c in cutlass.range_constexpr(V4L):
                            kb = (c * LPR + sublane_id) * 4
                            s0 = s_reg[c, 0] * decay
                            s1 = s_reg[c, 1] * decay
                            s2 = s_reg[c, 2] * decay
                            s3 = s_reg[c, 3] * decay
                            s_reg[c, 0] = s0
                            s_reg[c, 1] = s1
                            s_reg[c, 2] = s2
                            s_reg[c, 3] = s3
                            p[c] = (
                                _mul_nf(s0, s_k[t, kb + 0])
                                + _mul_nf(s1, s_k[t, kb + 1])
                            ) + (
                                _mul_nf(s2, s_k[t, kb + 2])
                                + _mul_nf(s3, s_k[t, kb + 3])
                            )
                        if const_expr(V4L == 8):
                            dot_k = ((p[0] + p[4]) + (p[2] + p[6])) + (
                                (p[1] + p[5]) + (p[3] + p[7])
                            )
                        else:
                            dot_k = (p[0] + p[2]) + (p[1] + p[3])
                        if const_expr(LPR >= 8):
                            dot_k = dot_k + cute.arch.shuffle_sync_bfly(dot_k, offset=4)
                        if const_expr(LPR >= 4):
                            dot_k = dot_k + cute.arch.shuffle_sync_bfly(dot_k, offset=2)
                        dot_k = dot_k + cute.arch.shuffle_sync_bfly(dot_k, offset=1)

                        corr = (s_v[t, subwarp_id] - dot_k) * beta

                        for c in cutlass.range_constexpr(V4L):
                            kb = (c * LPR + sublane_id) * 4
                            s0 = s_reg[c, 0] + corr * s_k[t, kb + 0]
                            s1 = s_reg[c, 1] + corr * s_k[t, kb + 1]
                            s2 = s_reg[c, 2] + corr * s_k[t, kb + 2]
                            s3 = s_reg[c, 3] + corr * s_k[t, kb + 3]
                            s_reg[c, 0] = s0
                            s_reg[c, 1] = s1
                            s_reg[c, 2] = s2
                            s_reg[c, 3] = s3
                            p[c] = (
                                _mul_nf(s0, s_q[t, kb + 0])
                                + _mul_nf(s1, s_q[t, kb + 1])
                            ) + (
                                _mul_nf(s2, s_q[t, kb + 2])
                                + _mul_nf(s3, s_q[t, kb + 3])
                            )
                        if const_expr(V4L == 8):
                            dot_q = ((p[0] + p[4]) + (p[2] + p[6])) + (
                                (p[1] + p[5]) + (p[3] + p[7])
                            )
                        else:
                            dot_q = (p[0] + p[2]) + (p[1] + p[3])
                        if const_expr(LPR >= 8):
                            dot_q = dot_q + cute.arch.shuffle_sync_bfly(dot_q, offset=4)
                        if const_expr(LPR >= 4):
                            dot_q = dot_q + cute.arch.shuffle_sync_bfly(dot_q, offset=2)
                        dot_q = dot_q + cute.arch.shuffle_sync_bfly(dot_q, offset=1)

                        if sublane_id == 0:
                            s_out[t, subwarp_id] = cutlass.BFloat16(dot_q)
                            s_corr[t, subwarp_id] = corr

                # 4. Coalesced gOut write first -> GPU fence -> atomic trigger for RMSNormGated
                cute.arch.barrier()
                idx = tidx
                while idx < SPEC * BV:
                    wt = idx // BV
                    wv = idx % BV
                    if wt < qlen:
                        gOut[bos + wt, pid_hv, v_base + wv] = s_out[wt, wv]
                    idx = idx + 128

                cute.arch.barrier()
                if tidx == 0:
                    cute.arch.fence_acq_rel_gpu()
                    old = cute.arch.atomic_add(
                        gCtr.iterator + pid_b * HV_c + pid_hv, cutlass.Int32(1)
                    )
                    is_last = cutlass.Int32(1) if old == NVB - 1 else cutlass.Int32(0)
                    if old == NVB - 1:
                        gCtr[pid_b, pid_hv] = cutlass.Int32(0)
                    s_flag[0] = is_last
                cute.arch.barrier()

                # Write replay corr + (pid_v == 0) replay k/decay concurrently with RMSNormGated
                idx2 = tidx
                while idx2 < SPEC * BV:
                    wt = idx2 // BV
                    wv = idx2 % BV
                    if wt < qlen:
                        gReplay[sidx, pid_hv, wt, v_base + wv] = s_corr[wt, wv]
                    idx2 = idx2 + 128
                if pid_v == 0:
                    tk = warp_id
                    while tk < qlen:
                        gReplay[sidx, pid_hv, tk, V_c + lane_id * 4 + 0] = s_k[
                            tk, lane_id * 4 + 0
                        ]
                        gReplay[sidx, pid_hv, tk, V_c + lane_id * 4 + 1] = s_k[
                            tk, lane_id * 4 + 1
                        ]
                        gReplay[sidx, pid_hv, tk, V_c + lane_id * 4 + 2] = s_k[
                            tk, lane_id * 4 + 2
                        ]
                        gReplay[sidx, pid_hv, tk, V_c + lane_id * 4 + 3] = s_k[
                            tk, lane_id * 4 + 3
                        ]
                        if lane_id == 0:
                            gReplay[sidx, pid_hv, tk, V_c + K_c] = s_dec[tk]
                        tk = tk + 4

                # 5. Last CTA for (pid_b, pid_hv) runs vectorized bf16x4 RMSNormGated in-place
                if s_flag[0] == 1:
                    cute.arch.fence_acq_rel_gpu()
                    x_reg = cute.make_rmem_tensor((4,), cutlass.BFloat16)
                    z_reg = cute.make_rmem_tensor((4,), cutlass.BFloat16)
                    y_reg = cute.make_rmem_tensor((4,), cutlass.BFloat16)
                    z_tile_id = (QKV_D // 4) + pid_hv * 32 + lane_id
                    base_c = lane_id * 4
                    w0 = cutlass.Float32(gNormW[base_c + 0])
                    w1 = cutlass.Float32(gNormW[base_c + 1])
                    w2 = cutlass.Float32(gNormW[base_c + 2])
                    w3 = cutlass.Float32(gNormW[base_c + 3])
                    tn = warp_id
                    while tn < qlen:
                        tok = bos + tn
                        cute.copy(
                            copy_bf16_cg,
                            cute.local_tile(gOut[tok, pid_hv, None], (4,), (lane_id,)),
                            x_reg,
                        )
                        cute.copy(
                            copy_bf16_ca,
                            cute.local_tile(gQkvz[tok, None], (4,), (z_tile_id,)),
                            z_reg,
                        )
                        x0 = cutlass.Float32(x_reg[0])
                        x1 = cutlass.Float32(x_reg[1])
                        x2 = cutlass.Float32(x_reg[2])
                        x3 = cutlass.Float32(x_reg[3])
                        ss = (_mul_nf(x0, x0) + _mul_nf(x1, x1)) + (
                            _mul_nf(x2, x2) + _mul_nf(x3, x3)
                        )
                        var = _triton_div(_warp_sum32(ss), cutlass.Float32(128.0))
                        rstd = _triton_rsqrt(var + eps)
                        z0 = cutlass.Float32(z_reg[0])
                        z1 = cutlass.Float32(z_reg[1])
                        z2 = cutlass.Float32(z_reg[2])
                        z3 = cutlass.Float32(z_reg[3])
                        if const_expr(self.is_sigmoid):
                            y_reg[0] = cutlass.BFloat16(
                                (x0 * rstd) * w0 * _triton_sigmoid(z0)
                            )
                            y_reg[1] = cutlass.BFloat16(
                                (x1 * rstd) * w1 * _triton_sigmoid(z1)
                            )
                            y_reg[2] = cutlass.BFloat16(
                                (x2 * rstd) * w2 * _triton_sigmoid(z2)
                            )
                            y_reg[3] = cutlass.BFloat16(
                                (x3 * rstd) * w3 * _triton_sigmoid(z3)
                            )
                        else:
                            y_reg[0] = cutlass.BFloat16(
                                (x0 * rstd) * w0 * (z0 * _triton_sigmoid(z0))
                            )
                            y_reg[1] = cutlass.BFloat16(
                                (x1 * rstd) * w1 * (z1 * _triton_sigmoid(z1))
                            )
                            y_reg[2] = cutlass.BFloat16(
                                (x2 * rstd) * w2 * (z2 * _triton_sigmoid(z2))
                            )
                            y_reg[3] = cutlass.BFloat16(
                                (x3 * rstd) * w3 * (z3 * _triton_sigmoid(z3))
                            )
                        cute.copy(
                            copy_bf16_st,
                            y_reg,
                            cute.local_tile(gOut[tok, pid_hv, None], (4,), (lane_id,)),
                        )
                        tn = tn + 4

    return CuteGdnVerifyNorm


class _GdnFuseRuntime:
    """Compile cache, CuTe-DSL warmup provider, and per-device counter store."""

    def __init__(self) -> None:
        self._compiled: dict[_CompileKey, Any] = {}
        self._warmup: set[_CompileKey] = set()
        self._warmup_registered = False
        self._counters: dict[tuple[int, int], torch.Tensor] = {}
        self._kernel_cls: Any = None

    def _compile(self, key: _CompileKey) -> Any:
        fn = self._compiled.get(key)
        if fn is not None:
            return fn
        import cutlass
        import cutlass.cute as cute
        from cuda.bindings.driver import CUstream
        from quack.compile_utils import make_fake_tensor

        if self._kernel_cls is None:
            self._kernel_cls = _build_cute_gdn_verify_norm_cls()

        H, HV, spec_len, is_sigmoid, alog_dt, dt_bias_dt, norm_w_dt = key
        qkvz_dim = 2 * H * _HEAD_DIM + 2 * HV * _HEAD_DIM
        m = cute.sym_int()
        nb = cute.sym_int()
        b = cute.sym_int()
        mb = cute.sym_int()
        args = (
            make_fake_tensor(cutlass.BFloat16, (m, qkvz_dim), divisibility=4),
            make_fake_tensor(cutlass.BFloat16, (m, cute.sym_int()), divisibility=1),
            make_fake_tensor(_torch_to_cutlass_dtype(alog_dt), (HV,), divisibility=1),
            make_fake_tensor(_torch_to_cutlass_dtype(dt_bias_dt), (HV,), divisibility=1),
            make_fake_tensor(
                cutlass.Float32, (nb, HV, _HEAD_DIM, _HEAD_DIM), divisibility=4
            ),
            make_fake_tensor(
                cutlass.Float32,
                (nb, HV, spec_len, 2 * _HEAD_DIM + 1),
                divisibility=1,
            ),
            make_fake_tensor(
                _torch_to_cutlass_dtype(norm_w_dt), (_HEAD_DIM,), divisibility=1
            ),
            make_fake_tensor(
                cutlass.BFloat16, (m, HV, _HEAD_DIM), divisibility=4
            ),
            make_fake_tensor(cutlass.Int32, (mb, HV), divisibility=1),
            make_fake_tensor(
                cutlass.Int32, (cute.sym_int(),), divisibility=1, leading_dim=None
            ),
            make_fake_tensor(
                cutlass.Int32, (b,), divisibility=1, leading_dim=None
            ),
            cutlass.Float32(_HEAD_DIM**-0.5),
            cutlass.Float32(1e-6),
            cutlass.Int32(0),
            cutlass.Int32(HV),
            cutlass.Int32(NULL_BLOCK_ID),
            cutlass.Int32(spec_len),
            cutlass.Int32(spec_len),
            cutlass.Int32(1),
            CUstream(0),
        )
        kernel = self._kernel_cls(
            H=H, HV=HV, spec_len=spec_len, is_sigmoid=is_sigmoid, lpr=_LPR
        )
        fn = cute.compile(kernel, *args, options="--enable-tvm-ffi")
        self._compiled[key] = fn
        return fn

    def request_warmup(self, key: _CompileKey) -> None:
        self._warmup.add(key)
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
                name=(
                    f"Lebeche GDN RecoverSSM verify+norm "
                    f"(H={k[0]}, HV={k[1]}, T={k[2]}, sig={k[3]})"
                ),
                key=(
                    "lebeche-gdn-verify-norm",
                    k[0],
                    k[1],
                    k[2],
                    k[3],
                    str(k[4]),
                    str(k[5]),
                    str(k[6]),
                ),
                compile=partial(self._compile, k),
            )
            for k in sorted(
                self._warmup,
                key=lambda x: (
                    x[0],
                    x[1],
                    x[2],
                    x[3],
                    str(x[4]),
                    str(x[5]),
                    str(x[6]),
                ),
            )
        )

    def get_counter(self, device: torch.device, HV: int) -> torch.Tensor:
        dev_idx = device.index if device.index is not None else 0
        key = (dev_idx, HV)
        ctr = self._counters.get(key)
        if ctr is None:
            ctr = torch.zeros((MAX_M, HV), dtype=torch.int32, device=device)
            self._counters[key] = ctr
        return ctr


_RUNTIME = _GdnFuseRuntime()


def request_gdn_fuse_warmup(layer: Any) -> None:
    """Register the GDN verify+norm kernel for this layer with CuTe-DSL warmup."""
    if not gdn_fuse_active():
        return
    if (
        not getattr(layer, "use_gdn_recoverssm", False)
        or not getattr(layer, "enable_fused_gdn_spec_decode", False)
        or layer.gqa_interleaved_layout
        or layer.head_k_dim != _HEAD_DIM
        or layer.head_v_dim != _HEAD_DIM
        or layer.norm.bias is not None
        or layer.norm.group_size is not None
        or not layer.norm.norm_before_gate
        or layer.norm.activation not in ("sigmoid", "silu", "swish")
    ):
        return
    spec_len = layer.num_spec + 1
    if not (1 <= spec_len <= _MAX_SPEC_LEN):
        return
    H = layer.num_k_heads // layer.tp_size
    HV = layer.num_v_heads // layer.tp_size
    if H <= 0 or HV < H or HV % H != 0:
        return
    state_dtypes = layer.get_state_dtype()
    if len(state_dtypes) < 2 or state_dtypes[1] != torch.float32:
        return
    if (
        layer.A_log.dtype not in (torch.bfloat16, torch.float32)
        or layer.dt_bias.dtype not in (torch.bfloat16, torch.float32)
        or layer.norm.weight.dtype not in (torch.bfloat16, torch.float32)
    ):
        return
    _RUNTIME.request_warmup(
        (
            H,
            HV,
            spec_len,
            layer.norm.activation == "sigmoid",
            layer.A_log.dtype,
            layer.dt_bias.dtype,
            layer.norm.weight.dtype,
        )
    )


def try_fused_recoverssm_decode(
    layer: Any,
    mixed_qkvz: torch.Tensor,
    ba: torch.Tensor,
    core_attn_out: torch.Tensor,
    attn_metadata: Any,
) -> bool:
    """Run the 2-kernel fused GDN RecoverSSM decode path when eligible.

    Returns True if executed (``core_attn_out``, ``conv_state``, and
    ``replay_cache`` are updated); returns False without mutating anything so
    the caller can fall back to the stock path.
    """
    if not gdn_fuse_active():
        return False
    if not getattr(layer, "use_gdn_recoverssm", False):
        return False
    if (
        attn_metadata.spec_sequence_masks is None
        or attn_metadata.num_prefills != 0
        or attn_metadata.num_decodes != 0
        or attn_metadata.num_spec_decodes <= 0
    ):
        return False

    num_padded_tokens = mixed_qkvz.shape[0]
    num_actual_tokens = attn_metadata.num_actual_tokens
    batch = attn_metadata.num_spec_decodes
    spec_len = layer.num_spec + 1
    if not (
        1 <= num_actual_tokens <= num_padded_tokens <= MAX_M
        and 1 <= batch <= MAX_M
        and 1 <= spec_len <= _MAX_SPEC_LEN
    ):
        return False

    if (
        layer.gqa_interleaved_layout
        or layer.head_k_dim != _HEAD_DIM
        or layer.head_v_dim != _HEAD_DIM
        or layer.norm.bias is not None
        or layer.norm.group_size is not None
        or not layer.norm.norm_before_gate
        or layer.norm.activation not in ("sigmoid", "silu", "swish")
    ):
        return False

    H = layer.num_k_heads // layer.tp_size
    HV = layer.num_v_heads // layer.tp_size
    if H <= 0 or HV < H or HV % H != 0:
        return False
    qkv_dim = 2 * H * _HEAD_DIM + HV * _HEAD_DIM
    z_dim = HV * _HEAD_DIM

    self_kv_cache = layer.kv_cache
    if len(self_kv_cache) < 3:
        return False
    conv_state_raw = self_kv_cache[0]
    ssm_state = self_kv_cache[1]
    replay_cache = self_kv_cache[2]

    spec_qsl = attn_metadata.spec_query_start_loc
    spec_sidx_tensor = attn_metadata.spec_state_indices_tensor
    if (
        spec_qsl is None
        or spec_sidx_tensor is None
        or spec_qsl.ndim != 1
        or spec_qsl.shape[0] < batch + 1
        or spec_sidx_tensor.ndim != 2
        or spec_sidx_tensor.shape[0] < batch
        or spec_sidx_tensor.shape[1] < 1
    ):
        return False
    qsl = spec_qsl[: batch + 1]
    sidx = spec_sidx_tensor[:batch, 0]

    # Layout, dtype, and 16-byte alignment checks for vectorized copies.
    # Note:
    # - MambaBase.bind_kv_cache() slices conv_state, ssm_state, and replay_cache
    #   out of a single per-block packed page buffer, so ssm_state.stride(0) and
    #   replay_cache.stride(0) exceed their inner block size (is_contiguous() is
    #   False). We check inner contiguity plus 16-byte alignment of ssm_state
    #   (data_ptr % 16 == 0 and stride(0) % 4 == 0) for LDG.E.128.
    # - GDNRecoverSSMMetadataBuilder slices spec_state_indices_tensor[:, :1]
    #   from a (max_bs, num_spec + 1) staging buffer, so sidx.stride(0) is
    #   typically num_spec + 1 (e.g. 4), never 1. CuteGdnVerifyNorm takes
    #   dynamic strides for both gQsl and gSidx (leading_dim=None).
    replay_dim = 2 * _HEAD_DIM + 1
    if (
        mixed_qkvz.dtype != torch.bfloat16
        or ba.dtype != torch.bfloat16
        or core_attn_out.dtype != torch.bfloat16
        or conv_state_raw.dtype != torch.bfloat16
        or ssm_state.dtype != torch.float32
        or replay_cache.dtype != torch.float32
        or layer.A_log.dtype not in (torch.bfloat16, torch.float32)
        or layer.dt_bias.dtype not in (torch.bfloat16, torch.float32)
        or layer.norm.weight.dtype not in (torch.bfloat16, torch.float32)
        or mixed_qkvz.ndim != 2
        or tuple(mixed_qkvz.shape) != (num_padded_tokens, qkv_dim + z_dim)
        or mixed_qkvz.stride(1) != 1
        or (mixed_qkvz.stride(0) % 4) != 0
        or ba.ndim != 2
        or ba.shape[0] != num_padded_tokens
        or ba.stride(1) != 1
        or tuple(core_attn_out.shape) != (num_padded_tokens, HV, _HEAD_DIM)
        or not core_attn_out.is_contiguous()
        or ssm_state.ndim != 4
        or tuple(ssm_state.shape[1:]) != (HV, _HEAD_DIM, _HEAD_DIM)
        or ssm_state.stride()[1:] != (_HEAD_DIM * _HEAD_DIM, _HEAD_DIM, 1)
        or ssm_state.stride(0) < 0
        or (ssm_state.stride(0) % 4) != 0
        or replay_cache.ndim != 4
        or tuple(replay_cache.shape)
        != (ssm_state.shape[0], HV, spec_len, replay_dim)
        or replay_cache.stride()[1:] != (spec_len * replay_dim, replay_dim, 1)
        or replay_cache.stride(0) < 0
        or tuple(layer.A_log.shape) != (HV,)
        or tuple(layer.dt_bias.shape) != (HV,)
        or tuple(layer.norm.weight.shape) != (_HEAD_DIM,)
        or not layer.A_log.is_contiguous()
        or not layer.dt_bias.is_contiguous()
        or not layer.norm.weight.is_contiguous()
        or qsl.dtype != torch.int32
        or sidx.dtype != torch.int32
        or qsl.stride(0) < 1
        or sidx.stride(0) < 1
        or (mixed_qkvz.data_ptr() % 16) != 0
        or (core_attn_out.data_ptr() % 16) != 0
        or (ssm_state.data_ptr() % 16) != 0
    ):
        return False

    if layer.disable_tp_for_ba_proj and layer.tp_size > 1:
        ba_b_off = layer.tp_rank * HV
        ba_a_off = layer.num_v_heads + layer.tp_rank * HV
    else:
        ba_b_off = 0
        ba_a_off = HV
    if ba.shape[1] < ba_a_off + HV:
        return False

    # 1. Causal conv1d update in-place into mixed_qkvz[:num_actual_tokens, :qkv_dim]
    conv_state = (
        conv_state_raw
        if is_conv_state_dim_first()
        else conv_state_raw.transpose(-1, -2)
    )
    conv_weights = layer.conv1d.weight.view(
        layer.conv1d.weight.size(0), layer.conv1d.weight.size(2)
    )
    causal_conv1d_update(
        mixed_qkvz[:num_actual_tokens, :qkv_dim],
        conv_state,
        conv_weights,
        layer.conv1d.bias,
        layer.activation,
        conv_state_indices=sidx,
        num_accepted_tokens=attn_metadata.num_accepted_tokens,
        query_start_loc=qsl,
        max_query_len=spec_len,
        validate_data=False,
    )

    # 2. Fused RecoverSSM verify + RMSNormGated + padding/null zeroing
    key: _CompileKey = (
        H,
        HV,
        spec_len,
        layer.norm.activation == "sigmoid",
        layer.A_log.dtype,
        layer.dt_bias.dtype,
        layer.norm.weight.dtype,
    )
    fn = _RUNTIME._compile(key)
    ctr = _RUNTIME.get_counter(core_attn_out.device, HV)

    from cuda.bindings.driver import CUstream

    stream = CUstream(torch.cuda.current_stream(core_attn_out.device).cuda_stream)
    fn(
        mixed_qkvz,
        ba,
        layer.A_log,
        layer.dt_bias,
        ssm_state,
        replay_cache,
        layer.norm.weight,
        core_attn_out,
        ctr,
        qsl,
        sidx,
        float(_HEAD_DIM**-0.5),
        float(layer.norm.eps),
        int(ba_b_off),
        int(ba_a_off),
        int(NULL_BLOCK_ID),
        int(num_actual_tokens),
        int(num_padded_tokens),
        int(batch),
        stream,
    )
    return True
