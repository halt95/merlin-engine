# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lebeche fork: sm_80/86 Triton Tensor-Core GEMM for MTP fc_hidden (MERLIN_LEBECHE_MTP_LL_EXTRA).

On Flash-Next (TP=2, hc_count=4), MTP step 0 feeds [T=4, 4, 2560] -> M_flat=16
into fc_hidden (weight [1280, 2560]). CuteSkinnyGemm (SIMT) loses to CUTLASS at
M >= 12, so M_flat=16 fell back to cutlass_80_tensorop + cublasLt::splitKreduce
(14.63 us cold-L2 on RTX 3090). A 1-pass Triton Tensor-Core GEMM (no split-K
reduce, BLOCK_N=32, BLOCK_K=128, num_warps=4, num_stages=5) runs in ~10.05 us
at M=16 (+4.57 us/step) and ~9.84 us at M=8 (+2.0 us over CuteSkinnyGemm),
with tighter FP32 accumulation accuracy vs FP64 than split-K cuBLASLt.

Inert on non-sm_80/86 devices and when MERLIN_LEBECHE_MTP_LL_EXTRA=0.
"""

from __future__ import annotations

from functools import lru_cache

import torch

from vllm.merlin.envs import read_registered
from vllm.model_executor.warmup.jit_warmup_triton_helper import TritonWarmupTensor
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON, tl, triton

MTP_LL_EXTRA_SWITCH: bool = bool(read_registered("MERLIN_LEBECHE_MTP_LL_EXTRA"))

MTP_FC_SHAPE = (1280, 2560)
MTP_TC_MIN_M = 5
MTP_TC_MAX_M = 32


@lru_cache(maxsize=1)
def mtp_tc_gemm_active() -> bool:
    if not MTP_LL_EXTRA_SWITCH or not HAS_TRITON or not current_platform.is_cuda():
        return False
    cap = current_platform.get_device_capability()
    return cap is not None and cap.to_int() in (80, 86)


@triton.jit(do_not_specialize=["M"])
def _mtp_tc_gemm_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
) -> None:
    """Packed row-major [M, K] @ [N, K]^T -> [M, N] BF16 GEMM (FP32 acc)."""
    pid_n = tl.program_id(0)
    rm = tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    m_mask = rm[:, None] < M
    n_mask = rn[None, :] < N
    for k_off in tl.range(0, K, BLOCK_K):
        k = k_off + rk
        a = tl.load(
            x_ptr + rm[:, None] * K + k[None, :],
            mask=m_mask,
            other=0.0,
        )
        b = tl.load(
            w_ptr + rn[None, :] * K + k[:, None],
            mask=n_mask,
            other=0.0,
        )
        acc = tl.dot(a, b, acc=acc)
    tl.store(
        out_ptr + rm[:, None] * N + rn[None, :],
        acc.to(tl.bfloat16),
        mask=m_mask & n_mask,
    )


def mtp_tc_gemm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """1-pass Triton Tensor-Core BF16 GEMM for contiguous (N=1280, K=2560), 5 <= M <= 32."""
    M, K = x.shape
    N = weight.shape[0]
    out = torch.empty((M, N), dtype=torch.bfloat16, device=x.device)
    block_m = 16 if M <= 16 else 32
    block_n = 32
    _mtp_tc_gemm_kernel[(triton.cdiv(N, block_n),)](
        x,
        weight,
        out,
        M,
        N,
        K,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=128,
        num_warps=4,
        num_stages=5,
    )
    return out


_warmed_up = False


def warmup_mtp_tc_gemm() -> None:
    """Compile and preload both BLOCK_M={16, 32} specializations at startup."""
    global _warmed_up
    if _warmed_up or not mtp_tc_gemm_active() or not torch.cuda.is_initialized():
        return
    from vllm.merlin.triton_preload import preload_triton_kernels

    N, K = MTP_FC_SHAPE
    w_ptr = TritonWarmupTensor(torch.bfloat16, shape=(N, K))
    for block_m in (16, 32):
        x_ptr = TritonWarmupTensor(torch.bfloat16, shape=(block_m, K))
        out_ptr = TritonWarmupTensor(torch.bfloat16, shape=(block_m, N))
        _mtp_tc_gemm_kernel.warmup(
            x_ptr,
            w_ptr,
            out_ptr,
            block_m,
            N,
            K,
            BLOCK_M=block_m,
            BLOCK_N=32,
            BLOCK_K=128,
            num_warps=4,
            num_stages=5,
            grid=(triton.cdiv(N, 32),),
        )
    preload_triton_kernels(
        _mtp_tc_gemm_kernel,
        label="lebeche MTP fc_hidden TC GEMM",
        expected=2,
    )
    _warmed_up = True
