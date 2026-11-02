# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Weight-only quantized copy of a vocab-parallel lm_head for draft steps.

The speculator swaps this in as the draft model's ``lm_head``; the target keeps
the original BF16 module. ``LogitsProcessor`` drives it like any lm_head: it
calls ``quant_method.apply`` on the local vocab shard, then gathers across TP
and slices the vocab padding off.
"""

import torch
import torch.nn as nn

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    apply_gptq_marlin_linear,
    marlin_make_empty,
    marlin_make_workspace_new,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_test import (
    marlin_quantize,
)
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.scalar_type import scalar_types

logger = init_logger(__name__)

_GROUP_SIZE = 128
_QUANT_TYPES = {"int4": scalar_types.uint4b8, "int8": scalar_types.uint8b128}
_CHUNK_N = 8192  # vocab rows per CPU quantization block (multiple of 64)


class _MarlinHeadMethod:
    """The ``quant_method`` LogitsProcessor calls; ``layer`` is the head."""

    def apply(
        self,
        layer: "QuantizedDraftLMHead",
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return apply_gptq_marlin_linear(
            input=x,
            weight=layer.qweight,
            weight_scale=layer.scales,
            weight_zp=layer.zp,
            workspace=layer.workspace,
            wtype=layer.wtype,
            output_size_per_partition=layer.size_n,
            input_size_per_partition=layer.size_k,
            bias=bias,
        )


def _quantize_chunked(weight: torch.Tensor, wtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Marlin-pack a [N, K] head shard on the CPU, _CHUNK_N vocab rows at a time.

    Marlin's weight permutation works on 64-column groups of the [K, N] matrix
    and its packing and scale permutation stay inside those groups, so packing
    64-aligned column blocks and concatenating along dim 1 equals packing the
    whole matrix. Chunking keeps the CPU peak to a few hundred MiB per rank
    instead of several full fp32 copies of the shard.
    """
    size_n = weight.shape[0]
    q_parts, s_parts = [], []
    for n0 in range(0, size_n, _CHUNK_N):
        w_kn = weight[n0 : n0 + _CHUNK_N].to("cpu", torch.float32).t().contiguous()
        w_ref, q_w, s = marlin_quantize(w_kn, wtype, _GROUP_SIZE)
        del w_ref, w_kn
        q_parts.append(q_w)
        s_parts.append(s)
    return torch.cat(q_parts, dim=1), torch.cat(s_parts, dim=1)


class QuantizedDraftLMHead(nn.Module):
    def __init__(self, head: ParallelLMHead, mode: str):
        super().__init__()
        if mode not in _QUANT_TYPES:
            raise ValueError(f"draft_lm_head_quant must be one of {list(_QUANT_TYPES)}")
        weight = head.weight
        if weight.dim() != 2:
            raise ValueError("draft_lm_head_quant needs a plain 2-D lm_head weight")
        size_n, size_k = weight.shape  # local vocab shard (padded), hidden
        if size_k % _GROUP_SIZE or size_n % 64:
            raise ValueError(
                f"draft_lm_head_quant: shard {size_n}x{size_k} is not Marlin-"
                f"aligned (vocab shard % 64, hidden % {_GROUP_SIZE})"
            )
        self.mode = mode
        self.wtype = _QUANT_TYPES[mode]
        self.size_n, self.size_k = size_n, size_k
        self.tp_size = head.tp_size
        self.shard_indices = head.shard_indices
        self.quant_method = _MarlinHeadMethod()

        device, dtype = weight.device, weight.dtype
        q_w, s = _quantize_chunked(weight.detach(), self.wtype)
        self.register_buffer("qweight", q_w.to(device), persistent=False)
        self.register_buffer("scales", s.to(device=device, dtype=dtype), persistent=False)
        del q_w, s
        self.register_buffer("zp", marlin_make_empty(device).data, persistent=False)
        self.register_buffer(
            "workspace", marlin_make_workspace_new(device), persistent=False
        )
        self._log_built(dtype)

    def load_weights(self, weights):
        # AutoWeightsLoader hands a child module its weights through this hook.
        # The copy is derived from the target's head at load time; a reload
        # would leave it stale, so refuse rather than skip silently.
        raise NotImplementedError(
            "Reloading weights with draft_lm_head_quant set is not supported; "
            "restart the engine to rebuild the quantized draft head."
        )

    def _log_built(self, dtype: torch.dtype) -> None:
        size_n, size_k, mode = self.size_n, self.size_k, self.mode
        logger.info(
            "Draft lm_head: %s weight-only copy of the %dx%d shard (%.1f MiB; "
            "the target keeps its %s head)",
            mode.upper(),
            size_n,
            size_k,
            (self.qweight.numel() * self.qweight.element_size()
             + self.scales.numel() * self.scales.element_size()) / 2**20,
            dtype,
        )
