# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Restore dependency geometry shared by capacity sizing and unit builders.

Boundaries are the accepted hit length, AFTER the EAGLE drop. Proof pages
are required sources even when the finder trims them from the returned table.
No pool, keys, tensors or CUDA state are needed to describe the geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_coordinator import KVCacheCoordinator
    from vllm.v1.kv_cache_interface import KVCacheConfig


@dataclass(frozen=True)
class RestoreGroupDependencies:
    group_id: int
    data_pages: range
    proof_pages: range

    @property
    def blocks(self) -> int:
        return len(self.data_pages) + len(self.proof_pages)


def sliding_window_dependency_pages(
    boundary: int,
    block_size: int,
    sliding_window: int,
    extra_retained_tokens: int = 0,
    eagle_margin_tokens: int = 0,
) -> tuple[range, range]:
    """Required data and proof-only pages at an accepted SW boundary.

    Mirrors SlidingWindowManager._find_fine_grained_hit: the retained start
    uses the post-drop boundary, whereas the probed tail uses B + margin.
    A witness inside the data tail consumes no additional physical row.
    The tail itself is required even for a one-token window.
    """
    if block_size <= 0 or sliding_window <= 0:
        raise ValueError("block size and sliding window must be positive")
    if boundary < 0 or extra_retained_tokens < 0 or eagle_margin_tokens < 0:
        raise ValueError("boundary, extra retention and EAGLE margin must be nonnegative")
    if boundary == 0:
        return range(0), range(0)
    end = -(-boundary // block_size)
    retained = sliding_window - 1 + extra_retained_tokens
    first = min(
        (boundary - 1) // block_size,
        max(0, boundary - retained) // block_size,
    )
    proof_end = -(-(boundary + eagle_margin_tokens) // block_size)
    return range(first, end), range(end, proof_end)


def group_dependency_pages(
    kind: str,
    boundary: int,
    block_size: int,
    *,
    sliding_window: int = 0,
    extra_retained_tokens: int = 0,
    margin: int = 0,
) -> tuple[range, range]:
    """Data and proof-only physical pages one group needs at accepted
    boundary ``boundary``: the single geometry the manifest (capacity, L_max)
    and the unit builder (which keys a unit requires) both use.

    ``kind``: "full", "sliding" or "recurrent". ``margin``: the EAGLE proof
    margin in tokens (0 without EAGLE). A full group needs pages [0, B) and
    the proof pages up to B + margin; a sliding group the retained tail
    (``sliding_window_dependency_pages``); a recurrent group exactly the
    state page of B.
    """
    if boundary <= 0:
        return range(0), range(0)
    if kind == "recurrent":
        page = (boundary - 1) // block_size
        return range(page, page + 1), range(0)
    if kind == "sliding":
        return sliding_window_dependency_pages(
            boundary, block_size, sliding_window, extra_retained_tokens, margin
        )
    end = -(-boundary // block_size)
    return range(end), range(end, -(-(boundary + margin) // block_size))


def fine_grained_lookup(coordinator, manager) -> bool:
    """Whether ``manager``'s group resolves hits below its block size: the
    coordinator's resolved policy, the one source the manifest, the
    unit builder and the finders' EAGLE proof margin all use; never an
    inference from the physical page size."""
    return bool(
        getattr(coordinator, "enable_partial_hash_hits", False)
        and getattr(manager, "supports_fine_grained_hash_lookup", False)
    )


def fine_grained_group_ids(coordinator) -> frozenset[int]:
    return frozenset(
        gid for gid, manager in enumerate(coordinator.single_type_managers)
        if fine_grained_lookup(coordinator, manager)
    )


def restore_dependency_manifest(
    cfg: KVCacheConfig,
    boundary: int,
    *,
    coordinator: KVCacheCoordinator | None = None,
    block_sizes: Sequence[int] | None = None,
) -> tuple[RestoreGroupDependencies, ...]:
    """Resolve per-group rows, including witnesses, without allocating them.

    Boot passes the actual coordinator: manager block sizes include DCP,
    use_eagle includes the legacy all-group fallback, and fine-grained lookup
    is the resolved policy (not an inference from the physical page size).
    Config-only callers use explicit group EAGLE flags and the resolved hit
    alignment; callers needing the legacy fallback must pass a coordinator.
    Recurrent groups always require one exact-B state, never a proof page.
    """
    from vllm.v1.core.kv_cache_utils import eagle_proof_margin
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        MambaSpec,
        SlidingWindowSpec,
    )

    if boundary < 0:
        raise ValueError("restore boundary must be nonnegative")
    if boundary == 0:
        return ()
    result = []
    for gid in cfg.prefix_cacheable_group_ids:
        group = cfg.kv_cache_groups[gid]
        spec = group.kv_cache_spec
        if coordinator is not None:
            manager = coordinator.single_type_managers[gid]
            size = manager.block_size
            # Unitary and no-prefix coordinators hash at the block size.
            hash_size = getattr(coordinator, "hash_block_size", size)
            fine = fine_grained_lookup(coordinator, manager)
            eagle = manager.use_eagle
        else:
            # Resolved per-group sizes (DCP/LCM) when the caller has them.
            size = block_sizes[gid] if block_sizes is not None else spec.block_size
            hash_size = cfg.hash_block_size or size
            fine = cfg.cache_hit_alignment_tokens == hash_size < size
            eagle = group.is_eagle_group
        if size <= 0:
            raise ValueError("restore block size must be positive")
        margin = eagle_proof_margin(size, hash_size, fine) if eagle else 0
        if isinstance(spec, MambaSpec):
            data, proof = group_dependency_pages("recurrent", boundary, size)
        elif isinstance(spec, SlidingWindowSpec):
            data, proof = group_dependency_pages(
                "sliding", boundary, size, sliding_window=spec.sliding_window,
                extra_retained_tokens=spec.extra_retained_tokens, margin=margin,
            )
        elif isinstance(spec, FullAttentionSpec):
            data, proof = group_dependency_pages(
                "full", boundary, size, margin=margin
            )
        else:
            # Existing singleton policy for other prefix-cacheable state.
            data, proof = range(1), range(0)
        result.append(RestoreGroupDependencies(gid, data, proof))
    return tuple(result)


def restore_capacity_blocks(
    dependency_blocks: int, *, assembly_blocks: int | None = None
) -> int:
    """One complete restore + the pool's null row (+ an explicit assembly
    allowance).

    The restore-unit warning fires below one 262K restore unit, so the
    default assembly allowance is zero. A caller that has resolved an
    assembly budget may pass it; this sizing helper does not enforce that
    budget.
    """
    if dependency_blocks < 0 or (assembly_blocks is not None and assembly_blocks < 0):
        raise ValueError("restore and assembly block counts must be nonnegative")
    if dependency_blocks == 0:
        return 0
    if assembly_blocks is None:
        assembly_blocks = 0
    return dependency_blocks + 1 + assembly_blocks
