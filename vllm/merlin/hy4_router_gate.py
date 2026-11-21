# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Hy4 MoE router gate held at the checkpoint's dtype (
hy4-gate-ckpt-dtype).

The Hy4 gate is a ``GateLinear`` with fp32 parameters and fp32 logits. The
checkpoint stores it BF16 (``mlp.gate.weight`` is BF16 in the tencent/Hy4-flash
safetensors headers; the hy4rtn packs leave the gate unquantized), so the fp32
parameter is an exact upcast of BF16 values held at twice the bytes: [512,
3072] fp32 = 6 MiB per MoE layer, 63 MiB on the 21-layer last stage.

Where ``GateLinear`` has no fp32-weight kernel for this device (no SM90+/gfx950
tier: the 3090 path), its forward is tier 7: ``x.to(float32)`` and the
unquantized GEMM on the fp32 weight. ``CheckpointDtypeGate`` keeps the weight
at the dtype the checkpoint delivers and runs that same tier-7 GEMM on
``weight.to(float32)``: the same fp32 operands, so the same logits, bit for bit.
A checkpoint that delivers fp32 is stored fp32. Where a specialized fp32-weight
tier is eligible the layer is the stock ``GateLinear``.

with MERLIN_HY4_FUSED_ROUTER=1, supported
decode inputs use a single Triton GEMV instead of the casts and GEMM. This
opt-in changes the FP32 reduction order; the stock selection kernel still
follows it in the MoE runner.
"""

from __future__ import annotations

import torch
from torch.nn import Parameter

import vllm.envs as envs
from vllm.merlin.hy4_fused_router import (
    fused_router_gate,
    fused_router_requested,
)
from vllm.model_executor.layers.fused_moe import GateLinear
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform

_STORABLE = (torch.bfloat16, torch.float16, torch.float32)


def stores_checkpoint_dtype(gate: GateLinear) -> bool:
    """True when ``gate``'s forward is the tier-7 fp32 GEMM (nothing else reads the
    fp32 weight), so storing it narrower and upcasting at compute is exact."""
    return bool(
        current_platform.is_cuda()
        and gate.is_unquantized
        and gate.bias is None
        and gate.weight.dtype == torch.float32
        and gate.out_dtype == torch.float32
        and not gate.allow_specialized_router_gemm
        and not gate.allow_fp32_router_gemm
        and not gate.allow_ll_fp32w_gemm
        and not gate.allow_ll_bf16_gemm
        and not gate.allow_bf16x3_router_gemm
        and not gate.allow_rocm_bf16x3_router_gemm
    )


class CheckpointDtypeGate(GateLinear):
    def __init__(self, *args, storage_dtype: torch.dtype = torch.bfloat16, **kwargs):
        super().__init__(*args, **kwargs)
        self._fused_router_enabled = fused_router_requested()
        self.stores_checkpoint_dtype = stores_checkpoint_dtype(self)
        if not self.stores_checkpoint_dtype:
            return
        old = self.weight
        new = Parameter(
            torch.empty(old.shape, dtype=storage_dtype, device=old.device),
            requires_grad=False,
        )
        attrs = dict(old.__dict__)
        attrs["weight_loader"] = self._checkpoint_dtype_loader
        self.register_parameter("weight", new)
        set_weight_attrs(new, attrs)

    def _checkpoint_dtype_loader(self, param: Parameter, loaded_weight: torch.Tensor):
        # Hold exactly what the checkpoint delivers: re-type the storage to its dtype.
        if loaded_weight.dtype != param.dtype:
            dtype = loaded_weight.dtype if loaded_weight.dtype in _STORABLE else torch.float32
            param.data = torch.empty(param.shape, dtype=dtype, device=param.device)
        self.weight_loader(param, loaded_weight)

    def forward(self, x: torch.Tensor):
        if not self.stores_checkpoint_dtype:
            return super().forward(x)
        # The stock MoE router remains launch two.
        # Keep the shape decision inside the custom op, outside Dynamo tracing.
        if (
            self._fused_router_enabled
            and not envs.VLLM_BATCH_INVARIANT
            and envs.VLLM_USE_FUSED_MOE_GROUPED_TOPK
        ):
            return self._return(fused_router_gate(x, self.weight))
        # GateLinear tier 7 on the fp32 weight it would have held.
        x = x.to(torch.float32)
        weight = self.weight.to(torch.float32)
        if envs.VLLM_BATCH_INVARIANT and (
            current_platform.is_cuda_alike() or current_platform.is_xpu()
        ):
            from vllm.model_executor.determinism.batch_invariant import (
                linear_batch_invariant,
            )

            output = linear_batch_invariant(x, weight, None)
        else:
            output = self.quant_method._gemm_impl(self, x, weight, None)
        return self._return(output)
