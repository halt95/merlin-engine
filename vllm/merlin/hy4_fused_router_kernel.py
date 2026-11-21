# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""One-launch BF16 gate GEMV with FP32 products and accumulation."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _gate_gemv(
    X,
    W,
    Y,
    M: tl.constexpr,
    X0: tl.constexpr,
    W0: tl.constexpr,
    BM: tl.constexpr,
):
    expert = tl.program_id(0)
    rows = tl.program_id(1) * BM + tl.arange(0, BM)
    cols = tl.arange(0, 1024)
    acc = tl.full((BM, 1024), 0, tl.float32)
    # One CTA owns the entire reduction: no atomics or split-K workspace.
    # Reuse each weight tile across four tokens for the larger decode batches.
    for tile in range(3):
        k = tile * 1024 + cols
        x = tl.load(
            X + rows[:, None] * X0 + k[None, :], rows[:, None] < M, 0
        ).to(tl.float32)
        w = tl.load(W + expert * W0 + k).to(tl.float32)
        acc = tl.fma(x, w[None, :], acc)
    logits = tl.sum(acc, axis=1)
    tl.store(Y + rows * 512 + expert, logits, rows < M)


def gate_gemv(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    # Eligibility is checked by the opaque dispatcher before importing this file.
    rows = x.shape[0]
    out = torch.empty((rows, 512), dtype=torch.float32, device=x.device)
    block_m = 1 if rows <= 4 else 4
    _gate_gemv[(512, triton.cdiv(rows, block_m))](
        x, weight, out, rows, x.stride(0), weight.stride(0),
        block_m, num_warps=4,
    )
    return out
