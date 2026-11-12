# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Per-module device-memory peaks during post-load weight processing.

Diagnostic only, default off. With ``MERLIN_LOAD_PEAK_LOG=1`` every module's
``process_weights_after_loading`` is bracketed: torch's peak allocated bytes
inside the call (``max_memory_allocated`` after a peak reset), reserved
bytes, and driver free memory (``mem_get_info``) before and after. At the end
the worst modules are logged, so a boot-time free-memory dip seen by an NVML
sampler can be attributed to live tensors (allocated) or to the caching
allocator (reserved > allocated, which torch frees on demand). No effect on
weights or numerics; off means a single env read per load.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_MIB = 1 << 20
_ROWS: list[tuple[int, str, int, int, int, int]] = []


def enabled() -> bool:
    return os.environ.get("MERLIN_LOAD_PEAK_LOG", "0") == "1"


@contextlib.contextmanager
def bracket(name: str, device: torch.device) -> Iterator[None]:
    """Record one module's processing peak (no-op unless enabled)."""
    if not enabled() or device.type != "cuda":
        yield
        return
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    alloc0 = torch.cuda.memory_allocated(device)
    free0 = torch.cuda.mem_get_info(device)[0]
    try:
        yield
    finally:
        torch.cuda.synchronize(device)
        peak = torch.cuda.max_memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
        free1 = torch.cuda.mem_get_info(device)[0]
        _ROWS.append((peak - alloc0, name, alloc0, peak, reserved, free0 - free1))


def report(device: torch.device, top: int = 8) -> None:
    """Log the modules with the largest transient peaks, then reset."""
    if not enabled() or not _ROWS:
        return
    rows = sorted(_ROWS, reverse=True)[:top]
    free = torch.cuda.mem_get_info(device)[0] if device.type == "cuda" else 0
    logger.info(
        "load peaks on %s: %d modules, free now %.0f MiB; top transients:",
        device, len(_ROWS), free / _MIB,
    )
    for over, name, alloc0, peak, reserved, dfree in rows:
        logger.info(
            "  %s: +%.0f MiB over start (allocated %.0f -> peak %.0f MiB), "
            "reserved %.0f MiB, driver free change %.0f MiB",
            name, over / _MIB, alloc0 / _MIB, peak / _MIB, reserved / _MIB,
            -dfree / _MIB,
        )
    _ROWS.clear()
