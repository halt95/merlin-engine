# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Host-only verify fast-path eligibility checks; no device reads or RNG state."""

import numpy as np


def sampling_may_mask_logits(
    temperature: float, min_p: float, top_k: int, top_p: float, vocab_size: int
) -> bool:
    """Inputs are the normalized values in SamplingStates."""
    return bool(
        temperature != 0.0
        and (min_p != 0.0 or top_k != vocab_size or top_p != 1.0)
    )


def guard_spec_verify_fast_paths(
    all_greedy: bool,
    skip_masked_vocab_blocks: bool,
    *,
    synthetic: bool,
    step_keyed_rng: bool,
    watermark: bool,
) -> tuple[bool, bool]:
    """Keep Merlin's protected verification paths on the original kernels."""
    if synthetic or step_keyed_rng or watermark:
        return False, False
    return bool(all_greedy), bool(skip_masked_vocab_blocks)


def spec_verify_batch_flags(
    temperature: np.ndarray | None,
    may_mask: np.ndarray | None,
    idx_mapping: np.ndarray,
) -> tuple[bool, bool]:
    """Only the current chunk's persistent request slots determine eligibility.

    Without the host mirrors (``None``) both shortcuts stay off: the legacy launch.
    """
    if temperature is None or may_mask is None or idx_mapping.size == 0:
        return False, False
    return (
        bool(np.all(temperature[idx_mapping] == 0.0)),
        bool(np.any(may_mask[idx_mapping])),
    )
