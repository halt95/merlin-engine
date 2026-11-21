# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Chunked, in-place Marlin WNA16 MoE weight repack: a lower load-time peak.

Stock ``_process_weights_marlin`` (fused_moe/oracle/int_wna16.py) first makes a
full K-first copy of each layer's packed weights and scales (``transpose(1, 2)
.contiguous()``), then repacks them into a third full-size tensor, while the
layer's own parameters are still alive. The peak is the loaded layer plus two
more copies of it: +1.21 GiB (1,238 MiB measured) per Flash-Next MoE layer (EP2:
256 experts, hidden 2560, intermediate 640, W4 g128) and +3.48 GiB per Hy4 layer (256 x
3072 x 1536), which is where the Hy4 PP1 workers ran out of memory.

Here the experts are converted in byte-bounded chunks along ``E``. Each chunk is
transposed and handed to the same stock op (``gptq_marlin_moe_repack`` or
``marlin_moe_permute_scales``, both per-expert loops), and the chunk's result is
copied back over the chunk's own input rows. The Marlin output of one expert has
exactly as many elements of the same dtype as its N-first input, so the loaded
parameter's storage becomes the output: the transient is one input chunk plus
one output chunk (2 x ``MERLIN_MOE_REPACK_CHUNK_MB``) instead of two layers.
The bytes are those of the stock path (the same op on the same expert slices);
tests/merlin/test_moe_repack_chunk_gpu.py (development tree) checks that bit for bit.

``MERLIN_MOE_REPACK_CHUNK_MB`` (default 64, ``0`` = stock path) bounds a chunk's
input bytes; a chunk always holds at least one expert. When the in-place write
does not apply (non-contiguous source, element count or dtype change, an
inference tensor outside inference mode) a fresh output is filled chunk by
chunk instead, which still drops the full transposed copy.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

_MIB = 1 << 20
DEFAULT_CHUNK_MIB = 64


def chunk_bytes() -> int:
    """Input bytes per chunk from ``MERLIN_MOE_REPACK_CHUNK_MB``; 0 = stock path."""
    import vllm.envs as envs

    return int(envs.MERLIN_MOE_REPACK_CHUNK_MB) * _MIB


def expert_chunk_ranges(
    num_experts: int, bytes_per_expert: int, max_chunk_bytes: int
) -> list[tuple[int, int]]:
    """Contiguous ``[start, stop)`` expert ranges covering ``[0, num_experts)``.

    Each range holds ``max(1, max_chunk_bytes // bytes_per_expert)`` experts
    (the last one fewer when that does not divide ``num_experts``).
    """
    if num_experts < 0 or bytes_per_expert < 0 or max_chunk_bytes <= 0:
        raise ValueError(
            "expert_chunk_ranges needs num_experts >= 0, bytes_per_expert >= 0 "
            f"and max_chunk_bytes > 0, got {num_experts}, {bytes_per_expert}, "
            f"{max_chunk_bytes}"
        )
    per = max(1, max_chunk_bytes // max(1, bytes_per_expert))
    return [(s, min(s + per, num_experts)) for s in range(0, num_experts, per)]


def _shares_storage(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()


def can_write_in_place(
    src: torch.Tensor, out_tail: tuple[int, ...], out_dtype: torch.dtype
) -> bool:
    """True when the converted experts can overwrite ``src``'s own storage."""
    if src.dim() < 1 or src.shape[0] == 0 or not src.is_contiguous():
        return False
    if src.dtype != out_dtype:
        return False
    numel_out = 1
    for d in out_tail:
        numel_out *= d
    if src[0].numel() != numel_out:
        return False
    # An inference tensor cannot be written outside inference mode.
    return not (torch.is_inference(src) and not torch.is_inference_mode_enabled())


def map_experts_chunked(
    src: torch.Tensor,
    convert: Callable[[torch.Tensor], torch.Tensor],
    out_tail: tuple[int, ...],
    out_dtype: torch.dtype,
    max_chunk_bytes: int,
) -> torch.Tensor:
    """``convert`` over ``src``'s experts in chunks; returns ``[E, *out_tail]``.

    ``convert`` maps an expert chunk ``src[s:t]`` to ``[t - s, *out_tail]`` and
    must depend on each expert's rows only (the stock ops loop per expert). When
    :func:`can_write_in_place` holds the result is a view of ``src``'s storage:
    chunk ``[s, t)`` is fully converted (stream order) before its rows are
    overwritten, and no other chunk reads them.
    """
    src = src.detach()
    num_experts = src.shape[0]
    in_place = can_write_in_place(src, out_tail, out_dtype)
    if in_place:
        out = src.view(num_experts, *out_tail)
    else:
        out = torch.empty((num_experts, *out_tail), dtype=out_dtype, device=src.device)
    bytes_per_expert = src[0].numel() * src.element_size() if num_experts else 0
    for s, t in expert_chunk_ranges(num_experts, bytes_per_expert, max_chunk_bytes):
        res = convert(src[s:t])
        if tuple(res.shape) != (t - s, *out_tail) or res.dtype != out_dtype:
            raise RuntimeError(
                f"moe repack chunk [{s}, {t}): got {tuple(res.shape)} {res.dtype}, "
                f"expected {(t - s, *out_tail)} {out_dtype}"
            )
        if in_place and _shares_storage(res, src):
            res = res.clone()
        out[s:t].copy_(res)
        del res
    return out


def repack_qweight(
    w: torch.Tensor,
    *,
    num_bits: int,
    pack_factor: int,
    is_a_8bit: bool,
    fp8_preprocess: bool,
    max_chunk_bytes: int,
) -> torch.Tensor:
    """N-first packed ``[E, N, K / pack]`` -> Marlin ``[E, K / 16, N * bits / 2]``.

    Per chunk the stock steps of ``_process_weights_marlin`` in their order:
    K-first copy, the optional (elementwise) FP8 preprocess, then
    ``gptq_marlin_moe_repack``.
    """
    import vllm._custom_ops as ops

    _, size_n, k_packed = w.shape
    size_k = k_packed * pack_factor
    out_tail = (size_k // 16, size_n * (num_bits // 2))

    def convert(chunk: torch.Tensor) -> torch.Tensor:
        x = chunk.transpose(1, 2).contiguous()
        if fp8_preprocess:
            x = ops.marlin_int4_fp8_preprocess(x, inplace=False)
        return ops.gptq_marlin_moe_repack(
            x, size_k, size_n, num_bits, is_a_8bit=is_a_8bit
        )

    return map_experts_chunked(w, convert, out_tail, w.dtype, max_chunk_bytes)


def permute_scales(
    s: torch.Tensor,
    *,
    size_k: Callable[[int], int],
    group_size: int,
    is_a_8bit: bool,
    fp8_scale: bool,
    max_chunk_bytes: int,
) -> torch.Tensor:
    """N-first scales ``[E, N, G]`` -> Marlin-permuted ``[E, G, N]``.

    Per chunk the stock steps: K-first copy, the optional FP8 ``* 512``, then
    ``marlin_moe_permute_scales``. ``size_k(G)`` gives the stock call's
    ``size_k`` argument from the K-first group count.
    """
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        marlin_moe_permute_scales,
    )

    _, size_n, groups = s.shape
    k = size_k(groups)

    def convert(chunk: torch.Tensor) -> torch.Tensor:
        x = chunk.transpose(1, 2).contiguous()
        if fp8_scale:
            x = x * 512
        return marlin_moe_permute_scales(
            s=x, size_k=k, size_n=size_n, group_size=group_size, is_a_8bit=is_a_8bit
        )

    return map_experts_chunked(s, convert, (groups, size_n), s.dtype, max_chunk_bytes)
