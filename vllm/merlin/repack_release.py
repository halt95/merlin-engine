# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Free the caching allocator after each module's weight repack.

Post-load processing (Marlin repack of a MoE layer) holds the new and the old
tensors at once, a ~1.24 GiB transient per Flash-Next MoE layer on a 4x20 GB
stage. Torch keeps the freed blocks cached, so across layers the reserved
pool grows ~2 GiB above the live peak and the driver's free memory (what
NVML reports) dips with it. After each
module, when cached-but-unused bytes reach ``_MIN_RELEASE_BYTES``, this
synchronises and calls ``empty_cache`` so the driver dip is bounded by the
live peak. No effect on weights or numerics. Default on;
``MERLIN_REPACK_RELEASE=0`` restores stock behaviour.
"""

from __future__ import annotations

import os

import torch

_MIB = 1 << 20
_MIN_RELEASE_BYTES = 256 * _MIB


def enabled() -> bool:
    return os.environ.get("MERLIN_REPACK_RELEASE", "1") != "0"


def release_after_repack(device: torch.device) -> int:
    """Return the cached bytes released (0 when nothing was done)."""
    if device.type != "cuda" or not enabled():
        return 0
    cached = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
    if cached < _MIN_RELEASE_BYTES:
        return 0
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    return cached
