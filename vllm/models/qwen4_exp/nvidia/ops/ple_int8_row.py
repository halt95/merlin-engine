# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""INT8 per-row PLE table rows (``ple_embedding_dtype == "int8_row"``; a
reference table).

Fork module. One PLE table row is 160 values with ONE FP16 scale (symmetric):

* ``q`` in [-127, 127] (int8), ``x[i] = q[i] * scale``; the table is quantised
  with ``scale = absmax(row) / 127`` stored as FP16 (an all-zero row gets 1) and
  ``q = clamp(round(x / scale), -127, 127)`` by the table builder;
* the checkpoint stores ``codes`` I8 ``[rows, 160]`` and ``scales`` F16
  ``[rows, 1]`` (``<name>.shard_N.codes`` / ``<name>.shard_N.scales``); there is
  no global ``weight_scale``.

The host table and the pull transport carry ONE interleaved U8 row of **162
bytes**: bytes 0..159 are the codes (two's complement), bytes 160..161 the scale
as little-endian FP16. The GPU expands the rows at ``dequantize_pulled`` with
``dequantize_int8_row_rows`` (one Triton program per row, static shapes, no
allocation beyond the output, CUDA-graph safe). The arithmetic is
``out_dtype(float32(q) * float32(scale))``: ``|q| <= 127`` and an FP16 scale make
the product exact in FP32, so the only rounding is the final cast -- the same
value as the builder's ``q.float() * scale.float()`` before that cast.

The table method that uses it is ``Qwen4ExpPLEInt8RowEmbeddingMethod`` in
``nvidia/ple_formats.py``; the torch functions below are the reference and the
loader's vectorised packing and need no GPU and no triton.
"""

import torch

from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

from .ple_common import dequantize_packed_rows, require_little_endian

INT8_ROW_DTYPE_NAME = "int8_row"
HEAD_DIM = 160  # values per table row
CODE_BYTES = HEAD_DIM  # 160
SCALE_BYTES = 2  # one FP16
ROW_BYTES = CODE_BYTES + SCALE_BYTES  # 162


def _require_little_endian() -> None:
    require_little_endian("INT8 row")


# ------------------------------------------------------------------ host side
def interleave_int8_row_rows(codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Codes I8 [R, 160] + scales F16 [R, 1] -> U8 [R, 162] rows (vectorised)."""
    _require_little_endian()
    if codes.dtype != torch.int8 or codes.ndim != 2 or codes.shape[1] != CODE_BYTES:
        raise ValueError(
            f"int8_row codes must be int8 [rows, {CODE_BYTES}], got "
            f"{codes.dtype} {tuple(codes.shape)}"
        )
    if scales.dtype != torch.float16 or scales.ndim != 2 or scales.shape[1] != 1:
        raise ValueError(
            "int8_row scales must be float16 [rows, 1], got "
            f"{scales.dtype} {tuple(scales.shape)}"
        )
    if codes.shape[0] != scales.shape[0]:
        raise ValueError(
            "int8_row codes/scales row counts differ: "
            f"{codes.shape[0]} != {scales.shape[0]}"
        )
    rows = torch.empty((codes.shape[0], ROW_BYTES), dtype=torch.uint8)
    rows[:, :CODE_BYTES] = codes.contiguous().view(torch.uint8)
    rows[:, CODE_BYTES:] = (
        scales.contiguous().view(torch.uint8).reshape(codes.shape[0], SCALE_BYTES)
    )
    return rows


def split_int8_row_rows(rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Inverse of ``interleave_int8_row_rows``: U8 [R, 162] -> (codes, scales)."""
    _require_little_endian()
    if rows.dtype != torch.uint8 or rows.shape[-1] != ROW_BYTES:
        raise ValueError(
            f"int8_row rows must be uint8 [..., {ROW_BYTES}], "
            f"got {rows.dtype} {tuple(rows.shape)}"
        )
    flat = rows.reshape(-1, ROW_BYTES)
    codes = flat[:, :CODE_BYTES].contiguous().view(torch.int8)
    scales = flat[:, CODE_BYTES:].contiguous().view(torch.float16).reshape(-1, 1)
    return codes, scales


def dequantize_int8_row_reference(
    rows: torch.Tensor, out_dtype: torch.dtype
) -> torch.Tensor:
    """Torch reference of the GPU kernel: out_dtype(float32(q) * float32(scale))."""
    lead = rows.shape[:-1]
    codes, scales = split_int8_row_rows(rows)
    x = codes.to(torch.float32) * scales.to(torch.float32)
    return x.to(out_dtype).reshape(*lead, HEAD_DIM)


# ------------------------------------------------------------------ GPU side
@triton.jit
def _int8_row_dequant_kernel(
    rows_ptr,
    out_ptr,
    ROW_BYTES: tl.constexpr,
    CODE_BYTES: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One program per table row: 162 interleaved bytes -> 160 values."""
    row = tl.program_id(0).to(tl.int64)
    j = tl.arange(0, BLOCK)
    mask = j < HEAD_DIM
    base = rows_ptr + row * ROW_BYTES
    b = tl.load(base + j, mask=mask, other=0).to(tl.int32)
    q = tl.where(b >= 128, b - 256, b).to(tl.float32)
    s0 = tl.load(base + CODE_BYTES).to(tl.int32)
    s1 = tl.load(base + CODE_BYTES + 1).to(tl.int32)
    scale = (s0 | (s1 << 8)).to(tl.int16).to(tl.float16, bitcast=True).to(tl.float32)
    value = q * scale
    tl.store(
        out_ptr + row * HEAD_DIM + j,
        value.to(out_ptr.dtype.element_ty),
        mask=mask,
    )


def _dequant_into(rows: torch.Tensor, out: torch.Tensor) -> None:
    """Rows U8 [R, 162] -> out [R, 160] (the custom op's body)."""
    n = rows.shape[0]
    if n == 0:
        return
    _int8_row_dequant_kernel[(n,)](
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
    op_name="qwen4_exp_ple_int8_row_dequant",
    op_func=_dequant_into,
    mutates_args=["out"],
    fake_impl=_dequant_into_fake,
)


def dequantize_int8_row_rows(
    rows: torch.Tensor, output_dtype: torch.dtype
) -> torch.Tensor:
    """U8 [..., k * 162] pulled rows -> [..., k * 160] in ``output_dtype``.

    ``k`` is the number of n-gram heads. CUDA tensors run the Triton kernel (static
    shape, no host sync); other devices run the torch reference (tests, CPU tooling).
    """
    return dequantize_packed_rows(
        rows,
        output_dtype,
        label="INT8 row",
        row_bytes=ROW_BYTES,
        head_dim=HEAD_DIM,
        cuda_op="qwen4_exp_ple_int8_row_dequant",
        reference=dequantize_int8_row_reference,
    )
