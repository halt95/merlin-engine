# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Copy-on-write block copies as row copies, with no gather temporary.

A ``blocks[dst] = blocks[src]`` copy gathers every source row into one
temporary first. With the side-cache layout each rank's KV pool is one storage,
so that temporary is n_copies x the rank's full bytes per block (up to
max_num_seqs x (KV groups - 1) rows per step), which a tight KV pin cannot
absorb. A basic-index row copy needs no temporary.
"""

from collections.abc import Sequence

import torch

from vllm.v1.core.kv_cache_utils import KVCacheBlockCopy


def copy_block_rows(
    blocks: torch.Tensor, kv_cache_block_copies: Sequence[KVCacheBlockCopy]
) -> bool:
    """Copy each ``(src, dst)`` row of ``blocks`` (dim 0 = scheduler blocks).

    Returns False, copying nothing, when a source is also a destination: the
    gather reads every source before any write, which row copies in list order
    would not. The scheduler never emits that (destinations are fresh blocks,
    sources stay retained); the caller's gather handles it if it ever does.
    Duplicate destinations keep the last copy.
    """
    pairs = [(int(src), int(dst)) for src, dst in kv_cache_block_copies]
    if {src for src, _ in pairs} & {dst for _, dst in pairs}:
        return False
    for src, dst in pairs:
        blocks[dst].copy_(blocks[src])
    return True
