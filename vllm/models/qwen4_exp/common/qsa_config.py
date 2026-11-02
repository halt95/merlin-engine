# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""CPU dispatch and warmup coverage for QSA sparse attention."""

#: Device capabilities (major * 10 + minor) with their own upstream tuning
#: table (vllm#57273). Every other capability, sm_8x included, takes the
#: default table.
QSA_TUNED_CAPABILITIES = (90, 120)


def _select_sm120_config(
    base_programs: int, use_prefill_config: bool, is_fp8: bool
) -> tuple[int, int, int]:
    """(block_n, target_splits, num_warps) retuned on sm_120 (RTX PRO 6000
    Blackwell), split by cache dtype: bf16 and fp8 favour different configs on
    sm_120, most visibly on the large-prefill region. Each entry is the fastest
    config that stays correct on its own path; the main win is more warps.

    Upstream vLLM 0.32 table, verbatim.
    """
    if is_fp8:
        if base_programs > 2048:
            return (32, 2, 1) if use_prefill_config else (64, 1, 2)
        if base_programs <= 24:
            return 64, 64, 8
        if base_programs <= 32:
            return 128, 8, 4
        if base_programs <= 64:
            return 64, 8, 4
        if base_programs <= 128:
            return 32, 4, 4
        if base_programs <= 256:
            return 128, 4, 4
        if base_programs <= 512:
            return 64, 4, 4
        return 32, 2, 1
    # bf16 K/V cache.
    if base_programs > 2048:
        return 64, 2, 2
    if base_programs <= 24:
        return 64, 64, 8
    if base_programs <= 64:
        return 64, 8, 8
    if base_programs <= 128:
        return 32, 8, 8
    if base_programs <= 256:
        return 32, 8, 1
    if base_programs <= 512:
        return 64, 4, 2
    return 32, 4, 1


def _select_sm90_config(
    base_programs: int, use_prefill_config: bool, is_fp8: bool
) -> tuple[int, int, int]:
    """(block_n, target_splits, num_warps) tuned on sm_90 (H20).

    Upstream vLLM 0.32 table, verbatim.
    """
    if base_programs > 2048:
        return (32, 1, 1) if use_prefill_config else (64, 1, 2)
    if is_fp8:
        if base_programs <= 24:
            return 64, 64, 2
        if base_programs <= 32:
            return 32, 16, 1
        if base_programs <= 64:
            return 32, 8, 1
        if base_programs <= 128:
            return 32, 4, 1
        if base_programs <= 256:
            return 32, 8, 1
        return 32, 4, 1
    if base_programs <= 32:
        return 64, 64, 2
    if base_programs <= 64:
        return 32, 16, 1
    if base_programs <= 256:
        return 32, 8, 1
    return 32, 4, 1


def select_qsa_sparse_config(
    num_rows: int,
    num_kv_heads: int,
    use_prefill_config: bool,
    num_columns: int,
    *,
    is_fp8: bool = False,
    capability: int | None = None,
) -> tuple[int, int, int, int]:
    """Select (block_n, num_warps, num_tiles, num_splits) for the kernel.

    Tuned on GB300 for the Qwen3.8-Flash-Next TP1/TP2/TP4 shapes, keyed on
    base_programs = num_rows * num_kv_heads. The bp > 2048 region splits on
    use_prefill_config (capture-stable: at FULL-graph capture max_query_len is the
    uniform decode/verify length).

    ``capability`` (major * 10 + minor) selects upstream's arch tables: 120
    (sm_120) and 90 (sm_90) dispatch to their own tables, split by ``is_fp8``
    (any FP8 K/V cache). Every other value, sm_8x included, and None take the
    default table, which ignores ``is_fp8``.
    """
    base_programs = num_rows * num_kv_heads
    if capability == 120:
        BLOCK_N, target_splits, num_warps = _select_sm120_config(
            base_programs, use_prefill_config, is_fp8
        )
    elif capability == 90:
        BLOCK_N, target_splits, num_warps = _select_sm90_config(
            base_programs, use_prefill_config, is_fp8
        )
    elif base_programs > 2048:
        BLOCK_N, target_splits, num_warps = (
            (32, 1, 1) if use_prefill_config else (64, 1, 2)
        )
    elif base_programs <= 24:
        BLOCK_N, target_splits, num_warps = 32, 64, 4
    elif base_programs <= 32:
        BLOCK_N, target_splits, num_warps = 32, 16, 1
    elif base_programs <= 64:
        BLOCK_N, target_splits, num_warps = 32, 8, 1
    elif base_programs <= 128:
        BLOCK_N, target_splits, num_warps = 32, 4, 1
    elif base_programs <= 256:
        BLOCK_N, target_splits, num_warps = 32, 8, 1
    elif base_programs <= 512:
        BLOCK_N, target_splits, num_warps = 64, 4, 2
    else:
        BLOCK_N, target_splits, num_warps = 64, 1, 2
    num_tiles = (num_columns + BLOCK_N - 1) // BLOCK_N
    # Never more splits than tiles, never empty.
    num_splits = min(target_splits, num_tiles)
    return BLOCK_N, num_warps, num_tiles, num_splits


def qsa_sparse_warmup_profiles(
    num_kv_heads: int,
    selection_width: int,
    *,
    max_num_rows: int = 8192,
    max_num_decode_rows: int | None = None,
    is_fp8: bool = False,
    capability: int | None = None,
) -> tuple[tuple[int, int, int, int], ...]:
    """Enumerate distinct configs over bounded prefill and decode rows.

    ``is_fp8`` and ``capability`` are forwarded to the selector, so warmup
    enumerates exactly what dispatch can pick on this device.
    """
    if max_num_decode_rows is None:
        max_num_decode_rows = max_num_rows
    if min(num_kv_heads, selection_width, max_num_rows, max_num_decode_rows) < 1:
        raise ValueError("QSA warmup bounds must be positive")
    if max_num_decode_rows > max_num_rows:
        raise ValueError("QSA decode warmup bound exceeds the token bound")
    return tuple(
        sorted(
            {
                select_qsa_sparse_config(
                    num_rows,
                    num_kv_heads,
                    use_prefill_config,
                    selection_width,
                    is_fp8=is_fp8,
                    capability=capability,
                )
                for use_prefill_config, row_limit in (
                    (False, max_num_decode_rows),
                    (True, max_num_rows),
                )
                for num_rows in range(1, row_limit + 1)
            }
        )
    )
