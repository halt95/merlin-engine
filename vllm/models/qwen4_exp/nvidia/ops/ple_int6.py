# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""INT6 g32 PLE table rows (``ple_embedding_dtype == "int6_g32"``).

Fork module. The table
method that uses it is ``Qwen4ExpPLEInt6G32EmbeddingMethod`` in
``nvidia/ple_formats.py``; the pull transport wiring is in ``nvidia/ple_pull.py``.

One PLE table row is 160 values = 5 groups of 32. A group is stored as 32
six-bit codes plus one FP16 scale:

* ``q`` in [-31, 31], ``x[g * 32 + i] = q * scale[g]``; the code is ``u = q + 32``
  (6 bits, ``u = 0`` is never written);
* four codes per three bytes, little-endian: ``w = u0 | u1 << 6 | u2 << 12 | u3 << 18``,
  bytes ``w & 255, (w >> 8) & 255, (w >> 16) & 255``. A row therefore has 40 words =
  **120 code bytes**;
* the checkpoint stores ``codes`` U8 ``[rows, 120]`` and ``scales`` F16 ``[rows, 5]``
  (``<name>.shard_N.codes`` / ``<name>.shard_N.scales``).

The host table and the pull transport carry ONE interleaved U8 row of **130 bytes**:
bytes 0..119 are the codes, bytes 120..129 are the five scales as little-endian FP16
(2 bytes each). ``PleOffloadLayer`` only checks row byte counts, so a U8
``[tokens, ngram_heads * 130]`` output needs no transport change; the GPU expands the
rows at ``dequantize_pulled`` with ``dequantize_int6_rows`` (one Triton program per
row, static shapes, no allocation beyond the output, CUDA-graph safe). The arithmetic
is ``bf16(float32(q) * float32(scale))``: ``|q| <= 31`` and an FP16 scale make the
product exact in FP32, so the only rounding is the final cast.

The torch functions below are the reference and the loader's vectorised packing;
they need no GPU and no triton.
"""

import torch

from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

from .ple_common import dequantize_packed_rows, require_little_endian

INT6_G32_DTYPE_NAME = "int6_g32"
GROUP = 32
GROUPS = 5
HEAD_DIM = GROUP * GROUPS  # 160 values per table row
CODE_BYTES = HEAD_DIM * 6 // 8  # 120
SCALE_BYTES = GROUPS * 2  # 10
ROW_BYTES = CODE_BYTES + SCALE_BYTES  # 130


def _require_little_endian() -> None:
    require_little_endian("INT6 g32")


# ------------------------------------------------------------------ host side
def interleave_int6_rows(codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Codes U8 [R, 120] + scales F16 [R, 5] -> U8 [R, 130] rows (vectorised)."""
    _require_little_endian()
    if codes.dtype != torch.uint8 or codes.ndim != 2 or codes.shape[1] != CODE_BYTES:
        raise ValueError(
            f"int6 codes must be uint8 [rows, {CODE_BYTES}], got "
            f"{codes.dtype} {tuple(codes.shape)}"
        )
    if scales.dtype != torch.float16 or scales.ndim != 2 or scales.shape[1] != GROUPS:
        raise ValueError(
            f"int6 scales must be float16 [rows, {GROUPS}], got "
            f"{scales.dtype} {tuple(scales.shape)}"
        )
    if codes.shape[0] != scales.shape[0]:
        raise ValueError(
            "int6 codes/scales row counts differ: "
            f"{codes.shape[0]} != {scales.shape[0]}"
        )
    rows = torch.empty((codes.shape[0], ROW_BYTES), dtype=torch.uint8)
    rows[:, :CODE_BYTES] = codes
    rows[:, CODE_BYTES:] = (
        scales.contiguous().view(torch.uint8).reshape(codes.shape[0], SCALE_BYTES)
    )
    return rows


def split_int6_rows(rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Inverse of ``interleave_int6_rows``: U8 [R, 130] -> (codes, F16 scales)."""
    _require_little_endian()
    if rows.dtype != torch.uint8 or rows.shape[-1] != ROW_BYTES:
        raise ValueError(
            f"int6 rows must be uint8 [..., {ROW_BYTES}], "
            f"got {rows.dtype} {tuple(rows.shape)}"
        )
    flat = rows.reshape(-1, ROW_BYTES)
    codes = flat[:, :CODE_BYTES].contiguous()
    scales = flat[:, CODE_BYTES:].contiguous().view(torch.float16).reshape(-1, GROUPS)
    return codes, scales


def unpack_int6_codes(codes: torch.Tensor) -> torch.Tensor:
    """U8 [R, 120] -> int8 [R, 160] q in [-32, 31] (-32 is never written)."""
    n = codes.shape[0]
    t = codes.reshape(n, -1, 3).to(torch.int32)
    w = t[..., 0] | (t[..., 1] << 8) | (t[..., 2] << 16)
    u = torch.stack([(w >> (6 * i)) & 63 for i in range(4)], dim=-1)
    return (u - 32).to(torch.int8).reshape(n, -1)


def dequantize_int6_reference(
    rows: torch.Tensor, out_dtype: torch.dtype
) -> torch.Tensor:
    """Torch reference of the GPU kernel: bf16(float32(q) * float32(scale))."""
    lead = rows.shape[:-1]
    codes, scales = split_int6_rows(rows)
    q = unpack_int6_codes(codes).to(torch.float32).reshape(-1, GROUPS, GROUP)
    x = (q * scales.to(torch.float32)[..., None]).reshape(-1, HEAD_DIM)
    return x.to(out_dtype).reshape(*lead, HEAD_DIM)


# ------------------------------------------------------------------ GPU side
@triton.jit
def _int6_g32_dequant_kernel(
    rows_ptr,
    out_ptr,
    ROW_BYTES: tl.constexpr,
    CODE_BYTES: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One program per table row: 130 interleaved bytes -> 160 values."""
    row = tl.program_id(0).to(tl.int64)
    j = tl.arange(0, BLOCK)
    mask = j < HEAD_DIM
    base = rows_ptr + row * ROW_BYTES
    word = j // 4
    pos = j % 4
    b0 = tl.load(base + word * 3, mask=mask, other=0).to(tl.int32)
    b1 = tl.load(base + word * 3 + 1, mask=mask, other=0).to(tl.int32)
    b2 = tl.load(base + word * 3 + 2, mask=mask, other=0).to(tl.int32)
    packed = b0 | (b1 << 8) | (b2 << 16)
    q = (((packed >> (pos * 6)) & 63) - 32).to(tl.float32)
    group = j // 32
    s0 = tl.load(base + CODE_BYTES + group * 2, mask=mask, other=0).to(tl.int32)
    s1 = tl.load(base + CODE_BYTES + group * 2 + 1, mask=mask, other=0).to(tl.int32)
    scale = (s0 | (s1 << 8)).to(tl.int16).to(tl.float16, bitcast=True).to(tl.float32)
    value = q * scale
    tl.store(
        out_ptr + row * HEAD_DIM + j,
        value.to(out_ptr.dtype.element_ty),
        mask=mask,
    )


def _dequant_into(rows: torch.Tensor, out: torch.Tensor) -> None:
    """Rows U8 [R, 130] -> out [R, 160] (the custom op's body)."""
    n = rows.shape[0]
    if n == 0:
        return
    _int6_g32_dequant_kernel[(n,)](
        rows,
        out,
        ROW_BYTES=ROW_BYTES,
        CODE_BYTES=CODE_BYTES,
        HEAD_DIM=HEAD_DIM,
        BLOCK=triton.next_power_of_2(HEAD_DIM),
    )


def _dequant_into_fake(rows: torch.Tensor, out: torch.Tensor) -> None:
    return None


direct_register_custom_op(
    op_name="qwen4_exp_ple_int6_dequant",
    op_func=_dequant_into,
    mutates_args=["out"],
    fake_impl=_dequant_into_fake,
)


def dequantize_int6_rows(rows: torch.Tensor, output_dtype: torch.dtype) -> torch.Tensor:
    """U8 [..., k * 130] pulled rows -> [..., k * 160] in ``output_dtype``.

    ``k`` is the number of n-gram heads. CUDA tensors run the Triton kernel (static
    shape, no host sync); other devices run the torch reference (tests, CPU tooling).
    """
    return dequantize_packed_rows(
        rows,
        output_dtype,
        label="INT6",
        row_bytes=ROW_BYTES,
        head_dim=HEAD_DIM,
        cuda_op="qwen4_exp_ple_int6_dequant",
        reference=dequantize_int6_reference,
    )
