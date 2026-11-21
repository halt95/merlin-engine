# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Two-launch Hy4 decode routing: Triton GEMV, then the stock fused top-k.

The existing MoE router owns sigmoid, selection-only bias, top-k, renormalization
and scaling. Keeping it also keeps its tie ordering, padding and capture hooks.
The GEMV changes FP32 reduction order versus cuBLAS: this is NOT bitwise and
needs a model-level KLD gate, even when the selection tests pass.

The custom op keeps the M <= MAX_ROWS decision out of Dynamo. vLLM can discard shape
guards, so a Python branch in the model forward could freeze decode dispatch
into a graph later used for prefill. No tensor values are read on the host.
"""

import os
from collections.abc import Mapping

import torch
import torch.nn.functional as F

from vllm.utils.torch_utils import direct_register_custom_op


def fused_router_requested(environ: Mapping[str, str] | None = None) -> bool:
    env = os.environ if environ is None else environ
    raw = (env.get("MERLIN_HY4_FUSED_ROUTER", "") or "").strip()
    if raw in ("", "0"):
        return False
    if raw == "1":
        return True
    raise ValueError(f"MERLIN_HY4_FUSED_ROUTER must be 0 or 1 (got {raw!r})")


# Measured 2026-11-20 on sm_86: fused 8.3 vs 29.3 us at M1, 34.0 vs
# 41.3 at M32, slower than stock at M64; the GEMV serves M <= 32 only.
MAX_ROWS = 32


def rows_supported(rows: int) -> bool:
    return 1 <= rows <= MAX_ROWS


def can_use_fused_router(x: torch.Tensor, weight: torch.Tensor) -> bool:
    """Metadata-only eligibility for the Hy4 Flash checkpoint shape."""
    return (
        x.is_cuda
        and torch.version.hip is None
        and weight.device == x.device
        and x.dtype == weight.dtype == torch.bfloat16
        and x.ndim == weight.ndim == 2
        and rows_supported(x.shape[0])
        and x.shape[1] == 3072
        and weight.shape == (512, 3072)
        and x.stride(1) == weight.stride(1) == 1
        and x.stride(0) >= 3072
        and weight.stride(0) >= 3072
    )


def _fused_router_gate_impl(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    if can_use_fused_router(x, weight):
        from vllm.merlin.hy4_fused_router_kernel import gate_gemv

        return gate_gemv(x, weight)
    # CheckpointDtypeGate's CUDA tier-7 operands and GEMM. This also gives
    # CPU callers a real fallback without importing Triton.
    return F.linear(x.float(), weight.float())


def _fused_router_gate_fake(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], weight.shape[0]), dtype=torch.float32)


direct_register_custom_op(
    op_name="hy4_fused_router_gate",
    op_func=_fused_router_gate_impl,
    fake_impl=_fused_router_gate_fake,
    dispatch_key="CompositeExplicitAutograd",
    tags=(torch.Tag.needs_fixed_stride_order,),
)


def fused_router_gate(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Inference-only gate dispatch; the caller owns opt-in and invariant mode."""
    return torch.ops.vllm.hy4_fused_router_gate(x, weight)
