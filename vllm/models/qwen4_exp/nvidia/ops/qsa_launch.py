# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Launch parameters of the QSA split-K partial kernel on sm_8x.

The default dispatch table selects BLOCK_N 64, 1 split, 2 warps, 2 stages for
513 to 2048 programs (query rows x KV heads) and for some smaller launches, and
for more than 2048 programs outside the prefill table. On sm_86 with E4M3 K/V,
Triton 3.8.0 spills about 4x more registers than 3.7.1 on that launch, which
makes the 2-stage launch about 46 % slower per 1024-row prefill launch, so it
runs with 1 stage (2 warps, 1 stage) whichever path selects it: prefill chunks,
and decode or verify batches that reach it (the shipped Flash-Next profiles cap
the decode batch below it). Prefill launches above 2048 programs use the
prefill table and keep 2 stages. Kernel source, tiles, splits and the
tile/reduction order are unchanged, so the output bytes match the 2-stage
launch under the same Triton. Measured on sm_86 with E4M3 K/V, the 1-stage
prefill launch runs faster than the 3.7.1 2-stage launch; sm_80 and sm_89 take
the same launch unmeasured (same bytes, speed only).
"""

from __future__ import annotations

from functools import lru_cache

from vllm.platforms import current_platform

SM8X_PREFILL_NUM_STAGES = 1
DEFAULT_NUM_STAGES = 2


@lru_cache(maxsize=1)
def _is_sm8x() -> bool:
    capability = current_platform.get_device_capability()
    return capability is not None and capability.major == 8


def qsa_partial_num_stages(
    block_n: int, num_warps: int, num_splits: int, tuning_capability: int | None
) -> int:
    """``num_stages`` of the split-K partial launch (warps are unchanged).

    Serving and warm-up both call this, so the warmed binary is the served one.
    Only the sm_8x default-table config BLOCK_N 64, 1 split, 2 warps drops to
    one stage, for prefill and for a decode batch that selects the same config;
    every other launch keeps 2 stages.
    """
    if (
        tuning_capability is None
        and block_n == 64
        and num_splits == 1
        and num_warps == 2
        and _is_sm8x()
    ):
        return SM8X_PREFILL_NUM_STAGES
    return DEFAULT_NUM_STAGES
