# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""GPU memory that ``initialize_kv_cache`` allocates besides the KV pool.

``initialize_kv_cache`` creates the attention metadata builders, and with them
their FlashInfer float workspace (``share_workspace`` allocates it eagerly:
394 MiB or more for the target and, unless MERLIN_SHARE_DRAFT_WORKSPACE=1, as
much again for the MTP drafter's builders), plus the block tables and the
Mamba / RecoverSSM contexts. None of that is in the memory profile:
``profile_run`` runs before the KV cache exists (``skip_attn=True``), and the
CUDA-graph estimate measures from the start of ``capture_model``, after the
profiling bootstrap has already initialised its minimal KV cache. With
``--kv-cache-memory-bytes`` unset the KV pool is then sized into memory the
real ``initialize_kv_cache`` needs for these buffers, and the warm-up or the
graph capture after it runs out (seen on a 27B TP2 boot at util 0.96).

The profiling bootstrap runs the same initialisation against a minimal KV
cache, so its free-memory drop minus the minimal KV storage is the overhead.
None of these buffers scales with the number of KV blocks, so the minimal
cache measures what the full one needs.
"""

from __future__ import annotations

import torch


def measure_kv_init_overhead(free_before: int, kv_bytes: int) -> int:
    """Bytes the KV-cache initialisation took beyond ``kv_bytes`` of KV storage.

    ``free_before`` is the device free memory just before the initialisation,
    read after an ``empty_cache``; the allocator cache is emptied again here so
    cached-but-unused blocks are not counted."""
    torch.accelerator.empty_cache()
    free_after = torch.accelerator.get_memory_info()[0]
    return max(free_before - free_after - kv_bytes, 0)
