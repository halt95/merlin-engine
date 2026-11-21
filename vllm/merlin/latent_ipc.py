# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""LatentMoE width on the FlashInfer PCIe IPC all-reduce (``MERLIN_LATENT_IPC``).

Stock, the PCIe IPC workspace is tuned and prepared for the model's hidden
width only, so any other all-reduce width falls back to NCCL. Hy4 Flash's
LatentMoE reduces its routed output at ``routed_expert_hidden_size`` (1536)
before the up-projection: ``MERLIN_LATENT_IPC=1`` tunes and prepares that width
on the same workspace too (its ``max_numel`` is sized by the hidden width, so
a narrower row fits). The reduction itself is the workspace's own: no change
of dtype or summation, the same kernel at another width.

The hooks in vllm/distributed/device_communicators/flashinfer_pcie_ipc_all_reduce.py
call :class:`Latent`; unset, blank or 0 leaves every shape and log line stock.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

from vllm.logger import init_logger

logger = init_logger(__name__)

ENV = "MERLIN_LATENT_IPC"


def latent_ipc_requested(environ=None) -> bool:
    """1 = also prepare the LatentMoE latent width; unset/blank/0 = stock."""
    raw = ((os.environ if environ is None else environ).get(ENV, "") or "").strip()
    if raw in ("", "0"):
        return False
    if raw == "1":
        return True
    raise ValueError(f"{ENV} must be 0 or 1 (got {raw!r})")


def latent_extra_widths(hf_text_config, environ=None) -> tuple[int, ...]:
    """The model's LatentMoE routed width when asked for (Hy4 Flash:
    ``routed_expert_hidden_size`` 1536), else nothing."""
    if not latent_ipc_requested(environ):
        return ()
    latent = getattr(hf_text_config, "routed_expert_hidden_size", None)
    return (int(latent),) if latent else ()


def latent_ipc_widths(hidden_dim: int, extra: Sequence[int]) -> tuple[int, ...]:
    """hidden_dim first, then each distinct positive extra width that fits the
    workspace (at most hidden_dim)."""
    out = [int(hidden_dim)]
    for width in extra:
        width = int(width)
        if width <= 0 or width > hidden_dim:
            logger.warning_once(
                "FlashInfer PCIe IPC all-reduce: extra width %d skipped (must be "
                "1..hidden_dim %d)",
                width,
                hidden_dim,
            )
            continue
        if width not in out:
            out.append(width)
    return tuple(out)


class Latent:
    """The hooks the PCIe IPC all-reduce wrapper calls."""

    extra = staticmethod(latent_extra_widths)
    widths = staticmethod(latent_ipc_widths)

    @staticmethod
    def shapes(batches: Sequence[int], widths: Sequence[int]) -> list[tuple[int, int]]:
        """The (batch, width) shapes to prepare: every capture size per width."""
        return [(batch, width) for width in widths for batch in batches]

    @staticmethod
    def fits(comm, width: int) -> bool:
        """``should_use``'s width test: one of the prepared widths (the hidden
        width alone before setup or when stock)."""
        return width in getattr(comm, "widths", (comm.hidden_dim,))

    @staticmethod
    def log(comm, batches: Sequence[int]) -> None:
        """One line per rank when extra widths are on (the wrapper's own init
        line is info_once, first rank only, and the latent layers sit on PP1 too)."""
        if len(comm.widths) > 1:
            import torch.distributed as dist

            logger.info(
                "MERLIN_LATENT_IPC: PCIe IPC all-reduce global rank %d, TP rank %d/%d, "
                "widths=%s (hidden %d + latent), max_tokens=%d",
                dist.get_rank() if dist.is_initialized() else 0,
                comm.rank,
                comm.world_size,
                comm.widths,
                comm.hidden_dim,
                batches[-1],
            )
