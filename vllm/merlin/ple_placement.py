# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Where PLE layers land under pipeline parallelism (Qwen 4 readiness).

Flash-Next has one PLE layer (``ple_layer_ids`` [2], 1-based, so decoder layer
1) and TP2xPP2 keeps it on PP rank 0. The PLE pull transport, the runner's
``input_ids`` path and the offload worker all assume that, so the
config check refuses a PLE layer on a later rank. This module turns that
refusal into the first step of the rule: name the ``VLLM_PP_LAYER_PARTITION``
that moves every PLE layer onto rank 0, or say that none exists.

Pure arithmetic on layer counts; ``partition=None`` reads the engine's own
split (``get_pp_indices``, which honours ``VLLM_PP_LAYER_PARTITION``).
"""

from __future__ import annotations

from collections.abc import Sequence


def stage_partition(
    num_layers: int, pp_size: int, partition: Sequence[int] | None = None
) -> list[int]:
    """Layer count per PP stage: ``partition`` if given, else the engine's split."""
    if partition is not None:
        part = [int(n) for n in partition]
        if len(part) != pp_size or sum(part) != num_layers or min(part) < 1:
            raise ValueError(
                f"partition {part} must have {pp_size} positive entries "
                f"summing to {num_layers}"
            )
        return part
    from vllm.distributed.utils import get_pp_indices

    bounds = [get_pp_indices(num_layers, r, pp_size) for r in range(pp_size)]
    return [end - start for start, end in bounds]


def stage0_partition(
    num_layers: int,
    ple_idx: Sequence[int],
    pp_size: int,
    partition: Sequence[int] | None = None,
) -> list[int] | None:
    """Smallest change to ``partition`` that puts every PLE layer on stage 0.

    Stage 0 grows to ``max(ple_idx) + 1`` layers; the extra layers come off the
    later stages one at a time, always from the largest (ties: the last stage,
    which also carries MTP and the LM head). Every stage keeps >= 1 layer.
    Returns the partition unchanged when it already fits, and None when no
    partition can (the transport has to serve a later stage, or PP=1).
    """
    part = stage_partition(num_layers, pp_size, partition)
    if not ple_idx or pp_size == 1:
        return part
    need = max(int(i) for i in ple_idx) + 1
    if need <= part[0]:
        return part
    if need > num_layers - (pp_size - 1):
        return None
    part = list(part)
    while part[0] < need:
        donor = max(range(1, pp_size), key=lambda s: (part[s], s))
        part[donor] -= 1
        part[0] += 1
    return part


def ple_rank0_refusal(
    num_layers: int,
    ple_idx: Sequence[int],
    pp_size: int,
    rank0_start: int,
    rank0_end: int,
) -> str:
    """The config check's refusal text, with the partition that would fit."""
    fit = stage0_partition(num_layers, ple_idx, pp_size)
    if fit is None:
        hint = (
            "No partition puts them all on rank 0 (every later rank needs a "
            "layer); run with PP=1 (a multi-stage transport is not supported yet)."
        )
    else:
        hint = (
            "Give rank 0 more layers, e.g. "
            f"VLLM_PP_LAYER_PARTITION={','.join(map(str, fit))} (check that "
            "stage's VRAM), or run with PP=1."
        )
    return (
        "Qwen4Exp N-gram PLE embedding requires every PLE layer to live on "
        f"pipeline rank 0 (layers {rank0_start}..{rank0_end - 1}), but PLE "
        f"layers are at {sorted(ple_idx)}. {hint}"
    )
