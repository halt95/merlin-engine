# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Fork PLE table formats beyond upstream's FP8 / unquantized pair.

Three more tables ride the pull transport beside the shipped FP8 table:
INT6 g32 is the shipped opt-in (``--ple int6``); BF16 and INT8-row are
reference tables:

* ``ple_embedding_dtype == "bfloat16"``: the BF16 reference table, stored
  unquantized whatever the checkpoint's quant_config says about the other
  layers. Rows cross the transport as BF16 (5,120 B/token).
* ``"int6_g32"``: 6-bit codes + five FP16 group scales per row, one interleaved
  U8 row of 130 bytes (``ops/ple_int6.py``; 2,080 B/token).
* ``"int8_row"``: int8 codes + one FP16 row scale, one interleaved U8 row of
  162 bytes (``ops/ple_int8_row.py``; 2,592 B/token).

The two packed formats store a U8 ``[rows, row_bytes]`` table, cross the
transport as bytes and are expanded to the activation dtype on the GPU after
the pull. Every other ``ple_embedding_dtype`` -- the shipped ``float8_e4m3fn``
included -- goes to upstream's ``Qwen4ExpPLEEmbeddingMethod.from_quant_config``
unchanged, so the FP8 path resolves to the same method by the same code.
"""

from collections.abc import Callable
from typing import ClassVar

import torch
from torch import nn

from vllm.model_executor.parameter import ModelWeightParameter
from vllm.models.qwen4_exp.common.ngram_embedding import (
    Qwen4ExpPLEEmbedding,
    Qwen4ExpPLEEmbeddingMethod,
    Qwen4ExpPLEUnquantizedEmbeddingMethod,
)

from .ops import ple_int6, ple_int8_row

#: The BF16 reference table's ``text_config.ple_embedding_dtype``.
BF16_DTYPE_NAME = "bfloat16"
INT6_G32_DTYPE_NAME = ple_int6.INT6_G32_DTYPE_NAME
INT8_ROW_DTYPE_NAME = ple_int8_row.INT8_ROW_DTYPE_NAME


class PackedShardPairer:
    """Collects the ``.codes`` / ``.scales`` halves of checkpoint shards.

    Hands back the interleaved U8 ``[rows, row_bytes]`` shard once both halves
    have arrived, in either order and across several ``load_weights`` calls.
    ``pending()`` lists the shards that only got one half; ``rows`` counts the
    table rows handed back, so a shard missing both halves is caught too.
    """

    def __init__(
        self, interleave: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    ) -> None:
        self._interleave = interleave
        self._halves: dict[str, dict[str, torch.Tensor]] = {}
        self._done: set[str] = set()
        self.rows = 0

    def add(
        self, shard_key: str, part: str, tensor: torch.Tensor
    ) -> torch.Tensor | None:
        if part not in ("codes", "scales"):
            raise ValueError(f"packed PLE shard {shard_key}: unknown part .{part}")
        if not (shard_key.isdigit() and str(int(shard_key)) == shard_key):
            # canonical shard numbers only: shard_00 and shard_0 would both land
            # in row block 0
            raise ValueError(
                f"packed PLE shard {shard_key!r}: not a canonical shard number"
            )
        if shard_key in self._done:
            # a second copy of a completed shard would add its rows twice and
            # could mask a missing one in the row count
            raise ValueError(f"packed PLE shard {shard_key}: loaded twice")
        halves = self._halves.setdefault(shard_key, {})
        if part in halves:
            raise ValueError(f"packed PLE shard {shard_key}: duplicate .{part}")
        halves[part] = tensor
        if len(halves) < 2:
            return None
        del self._halves[shard_key]
        self._done.add(shard_key)
        rows = self._interleave(halves["codes"], halves["scales"])
        self.rows += rows.shape[0]
        return rows

    def pending(self) -> list[str]:
        return sorted(self._halves)


class Qwen4ExpPLEPackedRowEmbeddingMethod(Qwen4ExpPLEEmbeddingMethod):
    """A PLE table stored as one interleaved U8 row of ``ROW_BYTES`` per table row.

    The checkpoint carries each shard as ``shard_N.codes`` / ``shard_N.scales``,
    which ``Qwen4ExpNGramEmbeddingPull.load_weights`` interleaves through
    ``make_pairer()``. Pull transport only: rows cross as bytes and
    ``dequantize`` runs on the GPU (the torch reference elsewhere).
    """

    DTYPE_NAME: ClassVar[str]
    HEAD_DIM: ClassVar[int]
    ROW_BYTES: ClassVar[int]

    @staticmethod
    def interleave(codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    def dequantize_rows(rows: torch.Tensor, output_dtype: torch.dtype) -> torch.Tensor:
        raise NotImplementedError

    def make_pairer(self) -> PackedShardPairer:
        return PackedShardPairer(self.interleave)

    def create_weights(
        self,
        layer: Qwen4ExpPLEEmbedding,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        del input_size, output_size, params_dtype
        if input_size_per_partition != self.HEAD_DIM:
            raise ValueError(
                f"{self.DTYPE_NAME} PLE tables hold {self.HEAD_DIM} values per "
                f"row, this layer's head_dim is {input_size_per_partition}"
            )
        weight = ModelWeightParameter(
            data=layer.allocate_embedding_weight(
                sum(output_partition_sizes),
                self.ROW_BYTES,
                torch.uint8,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=extra_weight_attrs.get("weight_loader"),
        )
        layer.register_parameter("weight", weight)

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        """A shard whose codes or scales never arrived must not load silently,
        nor a shard missing both (its rows would stay uninitialised)."""
        pairer = layer.__dict__.get("_packed_pending")
        if pairer is not None and pairer.pending():
            raise ValueError(
                f"{self.DTYPE_NAME} PLE checkpoint: shard(s) "
                f"{pairer.pending()} have only one of .codes / .scales"
            )
        weight = getattr(layer, "weight", None)
        if pairer is not None and weight is not None and pairer.rows < weight.shape[0]:
            raise ValueError(
                f"{self.DTYPE_NAME} PLE checkpoint: the paired shards hold "
                f"{pairer.rows} rows, the table has {weight.shape[0]}; a shard "
                "is missing (both its .codes and .scales)"
            )

    def dequantize(
        self,
        layer: nn.Module,
        embeddings: torch.Tensor,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        del layer
        return self.dequantize_rows(embeddings, output_dtype)


class Qwen4ExpPLEInt6G32EmbeddingMethod(Qwen4ExpPLEPackedRowEmbeddingMethod):
    """INT6 g32: U8 ``[rows, 130]`` (codes 120 B + five FP16 scales 10 B)."""

    DTYPE_NAME = ple_int6.INT6_G32_DTYPE_NAME
    HEAD_DIM = ple_int6.HEAD_DIM
    ROW_BYTES = ple_int6.ROW_BYTES
    interleave = staticmethod(ple_int6.interleave_int6_rows)
    dequantize_rows = staticmethod(ple_int6.dequantize_int6_rows)


class Qwen4ExpPLEInt8RowEmbeddingMethod(Qwen4ExpPLEPackedRowEmbeddingMethod):
    """INT8 per row: U8 ``[rows, 162]`` (int8 codes 160 B + one FP16 scale 2 B)."""

    DTYPE_NAME = ple_int8_row.INT8_ROW_DTYPE_NAME
    HEAD_DIM = ple_int8_row.HEAD_DIM
    ROW_BYTES = ple_int8_row.ROW_BYTES
    interleave = staticmethod(ple_int8_row.interleave_int8_row_rows)
    dequantize_rows = staticmethod(ple_int8_row.dequantize_int8_row_rows)


_PACKED_METHODS: dict[str, type[Qwen4ExpPLEPackedRowEmbeddingMethod]] = {
    INT6_G32_DTYPE_NAME: Qwen4ExpPLEInt6G32EmbeddingMethod,
    INT8_ROW_DTYPE_NAME: Qwen4ExpPLEInt8RowEmbeddingMethod,
}


def select_ple_embedding_method(
    quant_config,
    prefix: str,
    embedding_dtype: str | None = None,
) -> Qwen4ExpPLEEmbeddingMethod:
    """The fork's PLE format selector: the fork formats (BF16, INT6 g32,
    INT8-row), then upstream's."""
    packed = _PACKED_METHODS.get(embedding_dtype)  # type: ignore[arg-type]
    if packed is not None:
        return packed()
    if embedding_dtype == BF16_DTYPE_NAME:
        return Qwen4ExpPLEUnquantizedEmbeddingMethod()
    return Qwen4ExpPLEEmbeddingMethod.from_quant_config(
        quant_config, prefix, embedding_dtype
    )


__all__ = [
    "BF16_DTYPE_NAME",
    "INT6_G32_DTYPE_NAME",
    "INT8_ROW_DTYPE_NAME",
    "PackedShardPairer",
    "Qwen4ExpPLEInt6G32EmbeddingMethod",
    "Qwen4ExpPLEInt8RowEmbeddingMethod",
    "Qwen4ExpPLEPackedRowEmbeddingMethod",
    "select_ple_embedding_method",
]
