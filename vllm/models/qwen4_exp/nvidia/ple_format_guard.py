# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Fail-closed check that a PLE table's stored format matches its resolved method.

Guard: ``Qwen4ExpPLEEmbeddingMethod.from_quant_config`` picks the table
format from config alone (``ple_embedding_dtype`` and the quantization config).
In 0.32 a mis-resolution is silent: an FP8 E4M3 table whose prefix the quant
config ignores resolves to the unquantized method, and the loader then copies
FP8 bytes into a BF16 parameter without the scale. This compares what the
checkpoint actually stores with what the resolved method expects, at table
load, and refuses on any mismatch.
"""

import torch

from vllm.models.qwen4_exp.common.ngram_embedding import (
    Qwen4ExpPLEFp8EmbeddingMethod,
    Qwen4ExpPLEUnquantizedEmbeddingMethod,
)
from vllm.models.qwen4_exp.nvidia.ple_formats import (
    Qwen4ExpPLEInt6G32EmbeddingMethod,
    Qwen4ExpPLEInt8RowEmbeddingMethod,
)

#: The only valid (method, stored dtype) pairs. FP8 also needs its global scale,
#: which ``Qwen4ExpPLEFp8EmbeddingMethod.process_weights_after_loading`` checks.
#: The packed tables (INT6 g32, INT8 row) are checked on their interleaved
#: U8 rows, after ``Qwen4ExpNGramEmbeddingPull.load_weights`` paired each shard's
#: ``.codes`` / ``.scales`` (whose own dtypes and shapes the interleave checks).
_VALID: tuple[tuple[type, frozenset[torch.dtype]], ...] = (
    (Qwen4ExpPLEUnquantizedEmbeddingMethod, frozenset({torch.bfloat16, torch.float16})),
    (Qwen4ExpPLEFp8EmbeddingMethod, frozenset({torch.float8_e4m3fn})),
    (Qwen4ExpPLEInt6G32EmbeddingMethod, frozenset({torch.uint8})),
    (Qwen4ExpPLEInt8RowEmbeddingMethod, frozenset({torch.uint8})),
)


def check_ple_table_format(method: object, stored_dtype: torch.dtype, name: str) -> None:
    for method_cls, dtypes in _VALID:
        if type(method) is method_cls:
            if stored_dtype in dtypes:
                return
            expected = " or ".join(sorted(str(d) for d in dtypes))
            break
    else:
        expected = "no supported format"
    raise ValueError(
        f"PLE table format mismatch for {name!r}: the checkpoint stores "
        f"{stored_dtype} but the resolved method {type(method).__name__} "
        f"expects {expected}. Refusing to load (a wrong method would decode "
        "the table silently wrong); check text_config.ple_embedding_dtype and "
        "the quantization config's ignore list."
    )
