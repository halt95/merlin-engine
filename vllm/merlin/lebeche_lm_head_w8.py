# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 lebeche hill-climb (fork of Merlin 3.0.2.2)
"""Opt-in (LABELLED LOSSY, default OFF) on-the-fly per-channel W8 target lm_head.

Quantizes ``ParallelLMHead`` BF16 weights on-the-fly into symmetric per-channel
``uint8b128`` Marlin weights (``group_size=-1``) on sm_80/86/89 and runs the
logits projection through ``apply_gptq_marlin_linear`` (``vllm::marlin_gemm``).
At ``M <= 8`` decode, ``vllm::marlin_gemm`` dispatches to ``CuteW8MarlinGemm``
(``vllm/merlin/lebeche_w8_gemm.py``) at ~865-877 GB/s (~99% of RTX 3090 DRAM
ceiling, ~2x faster than BF16 cuBLAS); at ``M > 8`` it falls back to stock
Marlin. Frees ~303 MiB of BF16 lm_head weights per TP=2 PP-last rank.

Compatibility with ``draft_lm_head_quant`` (e.g. ``DRAFT_HEAD=int4``):
``speculator._maybe_quantize_draft_lm_head`` runs *after* target
``process_weights_after_loading`` and requires the target ``ParallelLMHead`` to
still be a plain ``ParallelLMHead`` with an ``UnquantizedEmbeddingMethod`` and
a floating-point ``weight``. When ``draft_lm_head_quant`` (or
``draft_lm_head_quantization``) is configured, quantization of the target head
is deferred until the first eager ``apply()`` call (which runs during
``profile_run``, after the draft head has been built from the original BF16
weight and before CUDA graph capture). When no draft head quantizer is active,
quantization completes inside ``process_weights_after_loading``.

``MERLIN_LEBECHE_LM_HEAD_W8`` defaults to ``"0"`` (OFF, bit-identical to
lossless BF16). Read once at import and hashed as a compile factor.
"""

from __future__ import annotations

import gc
import time

import torch
from torch import nn
from torch.nn.parameter import Parameter

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.merlin.envs import read_registered
from vllm.scalar_type import scalar_types

logger = init_logger(__name__)

SM8X_LM_HEAD_W8_CAPABILITIES: frozenset[tuple[int, int]] = frozenset(
    {(8, 0), (8, 6), (8, 9)}
)
LM_HEAD_W8_SWITCH: bool = bool(read_registered("MERLIN_LEBECHE_LM_HEAD_W8"))

_CHUNK_N = 16384  # vocab rows per GPU repack chunk (multiple of 64)
_active: bool | None = None


def lm_head_w8_active() -> bool:
    """Switch on and current CUDA device is sm_80/86/89."""
    global _active
    active = _active
    if active is None:
        active = False
        if LM_HEAD_W8_SWITCH:
            from vllm.platforms import current_platform

            if current_platform.is_cuda():
                cap = current_platform.get_device_capability()
                if cap is None:
                    return False
                active = (cap.major, cap.minor) in SM8X_LM_HEAD_W8_CAPABILITIES
        if not torch.compiler.is_compiling():
            _active = active
    return active


def _should_defer_for_draft_lm_head() -> bool:
    """Defer weight replacement when speculator needs the BF16 target weight."""
    try:
        from vllm.config import get_current_vllm_config

        cfg = get_current_vllm_config()
    except Exception:
        return False
    spec = getattr(cfg, "speculative_config", None) if cfg is not None else None
    if spec is None:
        return False
    return (
        getattr(spec, "draft_lm_head_quant", None) is not None
        or getattr(spec, "draft_lm_head_quantization", None) is not None
    )


def _is_eligible_lm_head(layer: nn.Module) -> bool:
    if envs.VLLM_BATCH_INVARIANT or not lm_head_w8_active():
        return False
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

    if type(layer) is not ParallelLMHead:
        return False
    if getattr(layer, "bias", None) is not None:
        return False
    weight = getattr(layer, "weight", None)
    if (
        not isinstance(weight, torch.Tensor)
        or weight.dim() != 2
        or not weight.is_cuda
        or weight.dtype != torch.bfloat16
    ):
        return False
    size_n, size_k = weight.shape
    return size_n > 0 and size_k > 0 and (size_n % 64 == 0) and (size_k % 128 == 0)


@torch.no_grad()
def quantize_lm_head_w8_marlin(
    weight: torch.Tensor, chunk_n: int = _CHUNK_N
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a ``[N, K]`` CUDA BF16 lm_head shard to channelwise Marlin W8.

    Uses chunked GPU quantization + ``ops.gptq_marlin_repack(..., num_bits=8)``
    to keep transient peak memory low (~120 MiB) and latency under ~200 ms.
    """
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        marlin_permute_scales,
    )

    if not (weight.is_cuda and weight.dim() == 2 and weight.dtype == torch.bfloat16):
        raise ValueError(
            f"quantize_lm_head_w8_marlin expects 2D CUDA BF16 weight, got "
            f"{tuple(weight.shape)} {weight.dtype} on {weight.device}"
        )
    size_n, size_k = weight.shape
    if size_n % 64 != 0 or size_k % 128 != 0:
        raise ValueError(
            f"quantize_lm_head_w8_marlin requires N%64==0 and K%128==0, got {size_n}x{size_k}"
        )
    if chunk_n % 64 != 0:
        raise ValueError(f"chunk_n must be a multiple of 64, got {chunk_n}")

    dtype = weight.dtype
    q_parts: list[torch.Tensor] = []
    s_parts: list[torch.Tensor] = []
    for n0 in range(0, size_n, chunk_n):
        w_chunk = weight[n0 : n0 + chunk_n]
        cn = w_chunk.shape[0]
        wf = w_chunk.float()
        max_abs = wf.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
        scale = (max_abs / 127.0).to(dtype)
        q_i = (
            torch.clamp(torch.round(wf / scale.float()), -128, 127).to(torch.int32)
            + 128
        )
        # Pack 4 uint8 bytes along K into int32: [K // 4, cn]
        q_t = q_i.t().contiguous().view(size_k // 4, 4, cn)
        q_packed = (
            q_t[:, 0, :]
            | (q_t[:, 1, :] << 8)
            | (q_t[:, 2, :] << 16)
            | (q_t[:, 3, :] << 24)
        ).contiguous()
        del wf, q_i, q_t
        q_marlin = ops.gptq_marlin_repack(
            b_q_weight=q_packed,
            size_k=size_k,
            size_n=cn,
            num_bits=8,
        )
        del q_packed
        s_marlin = marlin_permute_scales(
            scale.view(1, cn), size_k=size_k, size_n=cn, group_size=-1
        )
        q_parts.append(q_marlin)
        s_parts.append(s_marlin)

    qweight = torch.cat(q_parts, dim=1).contiguous()
    scales = torch.cat(s_parts, dim=1).contiguous()
    return qweight, scales


def finalize_parallel_lm_head_w8(layer: nn.Module) -> bool:
    """Quantize ``layer.weight`` to W8 Marlin buffers and release BF16 weight."""
    if getattr(layer, "_lebeche_lm_head_w8_ready", False):
        return True
    if not _is_eligible_lm_head(layer):
        layer._lebeche_lm_head_w8_pending = False
        return False

    from vllm.merlin.lebeche_w8_gemm import request_w8_small_m_warmup
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        marlin_make_empty,
        marlin_make_workspace_new,
    )

    weight = layer.weight
    size_n, size_k = weight.shape
    device, dtype = weight.device, weight.dtype
    orig_mib = weight.numel() * weight.element_size() / 2**20

    t0 = time.perf_counter()
    qweight, scales = quantize_lm_head_w8_marlin(weight.detach())
    zp = marlin_make_empty(device).data
    workspace = marlin_make_workspace_new(device)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    layer.register_buffer("_lebeche_w8_qweight", qweight, persistent=False)
    layer.register_buffer("_lebeche_w8_scales", scales, persistent=False)
    layer.register_buffer("_lebeche_w8_zp", zp, persistent=False)
    layer.register_buffer("_lebeche_w8_workspace", workspace, persistent=False)
    layer._lebeche_w8_size_n = size_n
    layer._lebeche_w8_size_k = size_k
    layer._lebeche_lm_head_w8_ready = True
    layer._lebeche_lm_head_w8_pending = False

    # Replace Parameter with an empty 0-byte placeholder so the BF16 shard is
    # freed if unshared (while leaving any tied embed_tokens tensor untouched).
    del weight
    layer.register_parameter(
        "weight",
        Parameter(torch.empty(0, dtype=dtype, device=device), requires_grad=False),
    )
    gc.collect()
    torch.cuda.empty_cache()

    request_w8_small_m_warmup(size_n, size_k)
    new_mib = (
        qweight.numel() * qweight.element_size()
        + scales.numel() * scales.element_size()
    ) / 2**20
    logger.info(
        "Target lm_head (MERLIN_LEBECHE_LM_HEAD_W8=1, LOSSY): quantized %dx%d "
        "shard BF16 (%.1f MiB) -> per-channel W8 Marlin (%.1f MiB) in %.1f ms",
        size_n,
        size_k,
        orig_mib,
        new_mib,
        elapsed_ms,
    )
    return True


def prepare_parallel_lm_head_w8(layer: nn.Module) -> None:
    """Hook called from ``UnquantizedEmbeddingMethod.process_weights_after_loading``."""
    if not _is_eligible_lm_head(layer):
        return
    from vllm.merlin.lebeche_w8_gemm import request_w8_small_m_warmup

    size_n, size_k = layer.weight.shape
    request_w8_small_m_warmup(size_n, size_k)
    if _should_defer_for_draft_lm_head():
        layer._lebeche_lm_head_w8_pending = True
        layer._lebeche_lm_head_w8_ready = False
    else:
        finalize_parallel_lm_head_w8(layer)


def maybe_apply_lm_head_w8(
    layer: nn.Module,
    x: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Run W8 Marlin lm_head projection if enabled/ready; else return ``None``."""
    if not LM_HEAD_W8_SWITCH or bias is not None or envs.VLLM_BATCH_INVARIANT:
        return None
    if getattr(layer, "_lebeche_lm_head_w8_pending", False):
        if torch.compiler.is_compiling() or torch.cuda.is_current_stream_capturing():
            return None
        finalize_parallel_lm_head_w8(layer)
    if not getattr(layer, "_lebeche_lm_head_w8_ready", False):
        return None

    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        apply_gptq_marlin_linear,
    )

    return apply_gptq_marlin_linear(
        input=x,
        weight=layer._lebeche_w8_qweight,
        weight_scale=layer._lebeche_w8_scales,
        weight_zp=layer._lebeche_w8_zp,
        workspace=layer._lebeche_w8_workspace,
        wtype=scalar_types.uint8b128,
        output_size_per_partition=layer._lebeche_w8_size_n,
        input_size_per_partition=layer._lebeche_w8_size_k,
        bias=None,
    )


__all__ = [
    "LM_HEAD_W8_SWITCH",
    "finalize_parallel_lm_head_w8",
    "lm_head_w8_active",
    "maybe_apply_lm_head_w8",
    "prepare_parallel_lm_head_w8",
    "quantize_lm_head_w8_marlin",
]
