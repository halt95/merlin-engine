# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Helpers shared by the packed PLE table formats (INT6 g32 and INT8-row).

Each format stores a table row as integer codes followed by FP16 scales in
one byte row, and dequantizes pulled rows with its own Triton kernel. The
checks and the reshape around that kernel are the same for every format.
"""

import sys
from collections.abc import Callable

import torch


def require_little_endian(label: str) -> None:
    """The FP16 scale bytes of a packed row are little-endian."""
    if sys.byteorder != "little":
        raise RuntimeError(f"{label} PLE rows are defined for little-endian hosts")


def dequantize_packed_rows(
    rows: torch.Tensor,
    output_dtype: torch.dtype,
    *,
    label: str,
    row_bytes: int,
    head_dim: int,
    cuda_op: str,
    reference: Callable[[torch.Tensor, torch.dtype], torch.Tensor],
) -> torch.Tensor:
    """U8 [..., k * row_bytes] pulled rows -> [..., k * head_dim].

    ``k`` is the number of n-gram heads. CUDA tensors run
    ``torch.ops.vllm.<cuda_op>`` (the format's registered custom op: static
    shape, no host sync); other devices run ``reference`` (tests, CPU
    tooling).
    """
    if rows.dtype != torch.uint8:
        raise TypeError(f"{label} PLE rows must be uint8, got {rows.dtype}")
    width = rows.shape[-1]
    if width % row_bytes:
        raise ValueError(
            f"{label} PLE row width {width} is not a multiple of {row_bytes}"
        )
    heads = width // row_bytes
    lead = rows.shape[:-1]
    flat = rows.reshape(-1, row_bytes)
    if flat.is_cuda:
        if not flat.is_contiguous():
            flat = flat.contiguous()
        out = torch.empty(
            (flat.shape[0], head_dim), dtype=output_dtype, device=flat.device
        )
        getattr(torch.ops.vllm, cuda_op)(flat, out)
    else:
        out = reference(flat, output_dtype)
    return out.reshape(*lead, heads * head_dim)
