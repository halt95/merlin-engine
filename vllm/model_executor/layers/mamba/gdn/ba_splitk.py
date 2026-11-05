# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
# --- GDN in_proj_ba split-K (sm_86, small M) ---
"""Split-K GEMM for the GDN in_proj_ba projection at small token counts.

For x[M <= 16, 5120] @ W[48, 5120]^T (the per-rank in_proj_ba at TP=2 during an MTP K=3
verify step, M = 4) cuBLAS on sm_86 picks a cutlass wmma kernel with a 4-CTA grid that
leaves most SMs idle. Splitting K 16 ways fills the SMs.
M > 16 and K not a multiple of 16 x 64 take F.linear unchanged (cuBLAS is already fast there).
Reduction order differs from cuBLAS, so the result is not bit-identical to it (fp32 accumulate).

Switch: VLLM_GDN_BA_SPLITK (registered in envs.py, on by default; 0 disables it).
Proof: "GDN in_proj_ba SPLIT-K ACTIVE: ..." once per process at the first dispatch.
"""
import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)

SPLITK_MAX_M = 16
SPLITK_S = 16
SPLITK_BK = 64
_fired = [False]


@triton.jit
def _ba_splitk_kernel(x_ptr, w_ptr, p_ptr, M, N, K: tl.constexpr, KCHUNK: tl.constexpr,
                      BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)
    rm = tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    k0 = pid_s * KCHUNK
    for k in range(k0, k0 + KCHUNK, BK):
        rk = k + tl.arange(0, BK)
        xa = tl.load(x_ptr + rm[:, None] * K + rk[None, :], mask=rm[:, None] < M, other=0.0)
        wa = tl.load(w_ptr + rn[:, None] * K + rk[None, :], mask=rn[:, None] < N, other=0.0)
        acc += tl.dot(xa, tl.trans(wa))
    tl.store(p_ptr + pid_s * M * N + rm[:, None] * N + rn[None, :], acc,
             mask=(rm[:, None] < M) & (rn[None, :] < N))


def gdn_ba_splitk(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    M, K = x.shape
    N = w.shape[0]
    if (
        M > SPLITK_MAX_M
        or K % (SPLITK_S * SPLITK_BK) != 0
        or not x.is_contiguous()
        or not w.is_contiguous()
    ):
        return torch.nn.functional.linear(x, w)
    if not _fired[0]:
        _fired[0] = True
        logger.info(
            "GDN in_proj_ba SPLIT-K ACTIVE: M=%d N=%d K=%d S=%d (dispatch M<=%d)",
            M, N, K, SPLITK_S, SPLITK_MAX_M)
    part = torch.empty((SPLITK_S, M, N), device=x.device, dtype=torch.float32)
    BN = 64
    _ba_splitk_kernel[(triton.cdiv(N, BN), SPLITK_S)](
        x, w, part, M, N, K=K, KCHUNK=K // SPLITK_S, BM=16, BN=BN, BK=SPLITK_BK)
    return part.sum(0).to(x.dtype)


def _gdn_ba_splitk_fake(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return x.new_empty((x.shape[0], w.shape[0]))


direct_register_custom_op(
    op_name="gdn_ba_splitk", op_func=gdn_ba_splitk, fake_impl=_gdn_ba_splitk_fake)
