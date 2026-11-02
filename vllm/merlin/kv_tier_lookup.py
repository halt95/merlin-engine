# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Read-only two-tier prefix resolution and the leases consumed by promotion.

Sources are typed handles, never GPU block-table entries. The allocation and
transaction owners consume the absolute slots in PrefixHitPlan; a scalar
local-prefix/external-suffix split cannot represent this result.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from functools import cached_property
from typing import TYPE_CHECKING, cast

from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    eagle_proof_margin,
    make_block_hash_with_group_id,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec

if TYPE_CHECKING:
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.kv_cache_coordinator import KVCacheCoordinator
    from vllm.v1.core.kv_cache_utils import KVCacheBlock


class Tier(Enum):
    GPU = "gpu"
    HOST = "host"


@dataclass(frozen=True)
class SourceVersion:
    """Both generations must change before storage or published bytes change."""

    allocation: int
    readiness: int


# None means unpublished, uncommitted, masked, or otherwise unreadable.
# The provider must check all-rank readiness even for referenced GPU blocks.
VersionReader = Callable[
    ["BlockPool", "KVCacheBlock", int, BlockHash], SourceVersion | None
]


def tier_version(
    pool: BlockPool, block: KVCacheBlock, group_id: int, key: BlockHash
) -> SourceVersion | None:
    """The VersionReader of the unified tier (``BlockTierState``).

    GPU pool with the tier: the content generation and its ready step, or
    None while a writer is pending on some rank or a commit was masked
    (readiness is a lookup predicate). Enforced from binding: the
    frontier starts at 0, so content no rank has acknowledged is a miss
    (there is no fail-open window before the first rank report). Host
    pool, or no tier: a key is published only once its copy completed, so a
    published block is ready.
    """
    tier = getattr(pool, "tier", None)
    if tier is None:
        return SourceVersion(0, 0)
    stamp = tier.stamp_of(block)
    if stamp is None:
        return None
    if tier.is_masked(block):
        return None
    if not tier.is_content_ready(block):  # Frontiers stay hittable
        return None
    return SourceVersion(stamp.gen, stamp.ready_step)


@dataclass(frozen=True, eq=False)
class PrefixSource:
    tier: Tier
    pool: BlockPool
    block: KVCacheBlock
    group_id: int
    key: BlockHash
    primary: Hashable
    version: SourceVersion
    coverage_start: int
    coverage_end: int

    @property
    def is_null(self) -> bool:
        return False


@dataclass(frozen=True)
class _NullSource:
    is_null: bool = True


_NULL = _NullSource()


@dataclass(frozen=True)
class PrefixSlot:
    group_id: int
    block_index: int
    source: PrefixSource
    valid_tokens: int
    # Continuation must use the manager's private-tail/CoW path.
    private_tail: bool


@dataclass(frozen=True)
class PrefixHitPlan:
    epoch: int
    boundary: int
    gpu_boundary: int
    shared_prefix_boundary: int
    # None is legitimate sparse padding, never an unresolved host hole.
    groups: tuple[tuple[PrefixSource | None, ...], ...]
    block_sizes: tuple[int, ...]
    gpu_pools: tuple[BlockPool, ...]
    proof_sources: tuple[PrefixSource, ...] = ()
    request_hashes: tuple[BlockHash, ...] = ()
    max_length: int = 0
    # Some probe met a keyed source it could not use for a reason that ends
    # by itself (not ready on every rank, a staged assembly member, this
    # step's CoW/publication exclusion): the plan may lengthen without any
    # new key, so it is never reused across passes.
    provisional: bool = False

    @property
    def union_gain(self) -> int:
        return max(0, self.boundary - self.gpu_boundary)

    # Built once per plan: validation, admission, the restore
    # slots and accounting read them repeatedly. The plan is frozen.
    @cached_property
    def slots(self) -> tuple[PrefixSlot, ...]:
        return tuple(
            PrefixSlot(
                gid,
                idx,
                source,
                min(size, self.boundary - idx * size),
                bool(self.boundary % size and idx == self.boundary // size),
            )
            for gid, group in enumerate(self.groups)
            for idx, source in enumerate(group)
            if source is not None
            for size in (self.block_sizes[gid],)
        )

    @cached_property
    def dependencies(self) -> tuple[PrefixSource, ...]:
        # Keep distinct keys even when they share storage: every proof must
        # survive revalidation. Physical leases are deduplicated separately.
        return tuple(dict.fromkeys(
            [slot.source for slot in self.slots] + list(self.proof_sources)
        ))


class _UnionView:
    """The read-only subset used by the existing group-specific finders."""

    null_block = _NULL

    def __init__(
        self,
        coordinator: KVCacheCoordinator,
        host_pool: BlockPool | None,
        block_hashes: Sequence[BlockHash],
        version: VersionReader,
        union: _UnionView | None = None,
    ):
        self.coordinator = coordinator
        self.host_pool = host_pool
        self.hashes = block_hashes
        self.hash_block_size = coordinator.block_pool.hash_block_size
        self.version = version
        self.indices = {key: i for i, key in enumerate(block_hashes)}
        self.cache: dict[tuple[int, int], PrefixSource | None] = {}
        self.used: list[PrefixSource] = []
        self.blocked = False
        # The finder's current max_length (tokens): no boundary it accepts
        # lies past it (GPU preference for sliding-window tails).
        self.bound: int | None = None
        # A GPU-only view beside a union view over the same hashes: the union
        # probes the GPU first, so its cached answer gives this view's (a
        # GPU source, else none) without probing again.
        self.union = union

    def _exact(self, gid: int, index: int) -> PrefixSource | None:
        cache_key = (gid, index)
        if (
            cache_key not in self.cache
            and self.union is not None
            and cache_key in self.union.cache
        ):
            shared = self.union.cache[cache_key]
            self.cache[cache_key] = (
                shared if shared is not None and shared.tier is Tier.GPU else None
            )
        if cache_key not in self.cache:
            self.cache[cache_key] = None
            manager = self.coordinator.single_type_managers[gid]
            key = self.hashes[index]
            pools = [(Tier.GPU, manager.block_pool)]
            if self.host_pool is not None:
                pools.append((Tier.HOST, self.host_pool))
            for tier, pool in pools:
                group_key = make_block_hash_with_group_id(key, gid)
                for block in pool.cached_block_hash_to_block.get_blocks(group_key):
                    # Preserve same-step CoW/publication restrictions in addition
                    # to all-rank content-generation readiness.
                    if tier is Tier.GPU and (
                        block.block_id in manager.published_block_ids_this_step
                        or block.block_hash in getattr(
                            manager, "cached_blocks_this_step", ()
                        )
                    ):
                        self.blocked = True
                        continue
                    stamp = self.version(pool, block, gid, key)
                    if stamp is None or block.is_null or block.block_hash is None:
                        self.blocked = True
                        continue
                    end = (index + 1) * self.hash_block_size
                    if (
                        block.block_hash_num_tokens is not None
                        and block.block_hash_num_tokens < end
                    ):
                        continue
                    size = manager.block_size
                    self.cache[cache_key] = PrefixSource(
                        tier, pool, block, gid, key, block.block_hash, stamp,
                        (end - 1) // size * size, end,
                    )
                    break
                if self.cache[cache_key] is not None:
                    break
        return self.cache[cache_key]

    def get_cached_block(
        self, block_hash: BlockHash, kv_cache_group_ids: list[int]
    ) -> list[PrefixSource] | None:
        index = self.indices[block_hash]
        result = []
        for gid in kv_cache_group_ids:
            source = self._exact(gid, index)
            manager = self.coordinator.single_type_managers[gid]
            spec = manager.kv_cache_spec
            # FA is append-only within a page. A later matching key covers an
            # earlier partial boundary, including differently keyed groups.
            # Recurrent states MUST NOT use this substitution.
            if isinstance(spec, FullAttentionSpec) and (
                source is None or source.tier is Tier.HOST
            ):
                size = manager.block_size
                end = (index + 1) * self.hash_block_size
                top = min(
                    ((end - 1) // size + 1) * size // self.hash_block_size,
                    len(self.hashes),
                )
                for later in range(top - 1, index, -1):
                    later_source = self._exact(gid, later)
                    if later_source is not None:
                        if source is None or later_source.tier is Tier.GPU:
                            source = later_source
                        if source.tier is Tier.GPU:
                            break
            # A sliding-window tail page keyed later on the host but
            # earlier on the GPU. Every boundary the finder can accept is at
            # most ``bound``, so a ready GPU key at or past it inside the
            # same page covers it: prefer it to a restore. Window pages
            # (wholly below ``bound``) never qualify.
            if (
                isinstance(spec, SlidingWindowSpec)
                and source is not None
                and source.tier is Tier.HOST
                and self.bound is not None
            ):
                size = manager.block_size
                page_start = index * self.hash_block_size // size * size
                lowest = max(
                    page_start // self.hash_block_size,
                    -(-self.bound // self.hash_block_size) - 1,
                )
                for earlier in range(index - 1, lowest - 1, -1):
                    gpu = self._exact(gid, earlier)
                    if gpu is not None and gpu.tier is Tier.GPU:
                        source = gpu
                        break
            if source is None:
                return None
            result.append(source)
        self.used.extend(result)
        return result


def closure_alignment(coordinator: KVCacheCoordinator, alignment: int) -> int:
    """The boundary grid that closes every compression group:
    lcm of the hit alignment and each prefix-cacheable spec's integer
    ``tokens_per_state`` (QSA compressed key/index, MLA indexers).

    ``resolve_kv_cache_block_sizes`` already refuses a config whose hit
    alignment does not close every ratio, so on a validated engine this is
    ``alignment`` itself; the resolver still enforces it, so a boundary
    that cannot close is lowered (replayed) and never restored.
    """
    from math import lcm

    from vllm.v1.kv_cache_interface import iter_layer_specs

    grid = alignment
    for manager in coordinator.single_type_managers:
        if not manager.kv_cache_spec.prefix_cacheable:
            continue
        for spec in iter_layer_specs(manager.kv_cache_spec):
            ratio = getattr(spec, "tokens_per_state", 1)
            if isinstance(ratio, int) and ratio > 1:
                grid = lcm(grid, ratio)
    return grid


def _resolve(
    coordinator: KVCacheCoordinator,
    view: _UnionView,
    max_length: int,
) -> tuple[tuple[tuple[PrefixSource | None, ...], ...], int, int,
           tuple[PrefixSource, ...]]:
    managers = coordinator.single_type_managers
    alignment = (
        view.hash_block_size
        if coordinator.enable_partial_hash_hits
        else coordinator.scheduler_block_size
    )
    active = [
        gid for gid, manager in enumerate(managers)
        if manager.kv_cache_spec.prefix_cacheable
    ]
    # FA first tightens the bound, but same-spec groups are deliberately
    # independent: their matching keys and residency need not coincide.
    active.sort(key=lambda gid: not isinstance(
        managers[gid].kv_cache_spec, FullAttentionSpec
    ))
    cap = min(max(0, max_length), len(view.hashes) * view.hash_block_size)
    # The boundary closes every compression group (QSA target/MTP).
    closure = closure_alignment(coordinator, alignment)
    cap -= cap % closure
    longest = 0
    groups: list[tuple[PrefixSource | None, ...]] = [()] * len(managers)
    if not active:
        cap = 0
    # A fresh verification at each reduced bound re-establishes sparse
    # endpoints and SW/EAGLE witnesses; no simple-hybrid early exit.
    while cap > 0:
        lengths = []
        dependencies = []
        # Within a pass each group is bounded by the lowest length an
        # earlier group returned (FA first), so a cold FA miss ends the pass
        # instead of running every recurrent group's backward scan from the
        # cap. The pass still returns only when every group reached ``cap``.
        run = cap
        for gid in active:
            if run <= 0:
                groups[gid] = ()
                lengths.append(0)
                continue
            manager = managers[gid]
            eagle = manager.use_eagle
            margin = eagle_proof_margin(
                manager.block_size,
                view.hash_block_size,
                coordinator.enable_partial_hash_hits
                and manager.supports_fine_grained_hash_lookup,
            ) if eagle else 0
            bound = run
            descents = 0
            while True:
                view.used = []
                view.bound = min(
                    bound + margin, len(view.hashes) * view.hash_block_size
                )
                blocks, length = manager.find_longest_cache_hit(
                    block_hashes=view.hashes,
                    max_length=view.bound,
                    kv_cache_group_ids=[gid],
                    block_pool=cast("BlockPool", view),
                    kv_cache_spec=manager.kv_cache_spec,
                    drop_eagle_block=eagle,
                    alignment_tokens=alignment,
                    dcp_world_size=manager.dcp_world_size,
                    pcp_world_size=manager.pcp_world_size,
                )
                if not 0 <= length <= bound:
                    raise ValueError("group finder exceeded the candidate boundary")
                if not length:
                    break
                # The block-granular SW finder omits
                # extra_retained_tokens. Every page the GPU-hit table retains
                # below the hit goes into the layout as data (restored like
                # the rest); if one is missing this candidate is not a
                # restore point and the search descends to the next lower
                # candidate, it does not end at zero.
                layout = _sw_retained_layout(
                    view, manager, gid, length, alignment, blocks[0]
                )
                if not isinstance(layout, int):
                    blocks = [layout]
                    break
                # Every candidate that still needs the missing page
                # fails too, so the next search ends below it; and the number
                # of descents is capped (each re-runs the finder at O(bound)),
                # a group past the cap takes a miss.
                bound = min(length - alignment, layout)
                descents += 1
                if bound <= 0 or descents > MAX_SW_DESCENTS:
                    blocks, length = ([[]], 0)
                    break
            groups[gid] = tuple(
                None if source.is_null else cast(PrefixSource, source)
                for source in blocks[0]
            )
            lengths.append(length)
            dependencies.extend(view.used)
            longest = max(longest, length)
            run = min(run, length - length % closure)
        next_cap = min(lengths)
        next_cap -= next_cap % closure
        if next_cap == cap:
            data = {source for group in groups for source in group if source}
            proofs = tuple(dict.fromkeys(
                source for source in dependencies if source not in data
            ))
            return tuple(groups), cap, longest, proofs
        cap = next_cap
    return tuple(() for _ in managers), 0, longest, ()


# Lower-boundary searches one block-granular sliding-window group may run in
# a resolve before it takes a miss.
MAX_SW_DESCENTS = 8


def _sw_retained_layout(view, manager, gid: int, length: int, alignment: int,
                        layout):
    """The group's restore layout at ``length``: the finder's own, except for
    a block-granular sliding-window group with extra retention, whose every
    data page the manifest requires (``group_dependency_pages``, the GPU-hit
    table's retained pages) is filled in from the union view. When such a
    page cannot be laid out, an int: the largest bound below which a
    candidate no longer needs it (``length - alignment`` when the page end is
    not hash-aligned). Fine-grained groups prove and lay out their retained
    pages inside their own finder."""
    from vllm.merlin.kv_restore_dependencies import group_dependency_pages
    from vllm.v1.kv_cache_interface import SlidingWindowSpec

    spec = manager.kv_cache_spec
    extra = getattr(spec, "extra_retained_tokens", 0)
    if not isinstance(spec, SlidingWindowSpec) or extra <= 0:
        return layout
    size, hbs = manager.block_size, view.hash_block_size
    if size > alignment and size % alignment == 0 and hbs == alignment:
        return layout  # fine-grained finder
    data, _ = group_dependency_pages(
        "sliding", length, size,
        sliding_window=spec.sliding_window, extra_retained_tokens=extra,
    )
    count = -(-length // size)
    out = list(layout[:count]) + [view.null_block] * max(0, count - len(layout))
    for page in data:
        if page >= count or not out[page].is_null:
            continue
        end = min((page + 1) * size, length)
        if end % hbs:
            return length - alignment
        found = view.get_cached_block(view.hashes[end // hbs - 1], [gid])
        if not found:
            # Cached in neither tier: every candidate above page * size
            # requires it (the required range only moves down with length).
            return page * size
        out[page] = found[0]
    return out


def find_longest_union_hit(
    coordinator: KVCacheCoordinator,
    block_hashes: Sequence[BlockHash],
    max_length: int,
    host_pool: BlockPool | None,
    *,
    version: VersionReader,
    epoch: int,
) -> PrefixHitPlan:
    """Resolve the greatest valid boundary across both tiers, from token zero.

    Callers enforce prefix_cache_lookup_enabled and pass num_tokens - 1.
    Resolution does not pin, allocate, publish, or count an admitted hit.
    ``host_pool=None`` resolves the GPU tier alone (the admission fallback
    when a restore cannot be materialized).
    """
    hashes = tuple(block_hashes)
    if (
        host_pool is not None
        and host_pool.hash_block_size != coordinator.block_pool.hash_block_size
    ):
        raise ValueError("GPU and host hash units differ")
    if host_pool is None:
        gpu_view = _UnionView(coordinator, None, hashes, version)
        groups, boundary, longest, proofs = _resolve(coordinator, gpu_view, max_length)
        gpu_boundary = boundary
        provisional = gpu_view.blocked
    else:
        view = _UnionView(coordinator, host_pool, hashes, version)
        groups, boundary, longest, proofs = _resolve(coordinator, view, max_length)
        provisional = view.blocked
        if not any(
            source is not None and source.tier is Tier.HOST
            for group in groups for source in group
        ) and not any(source.tier is Tier.HOST for source in proofs):
            # Every source is on the GPU: the GPU tier alone resolves the
            # same boundary (no second resolve).
            gpu_boundary = boundary
        else:
            # The GPU-only bound, sharing the union view's GPU probes.
            gpu_view = _UnionView(coordinator, None, hashes, version, union=view)
            _, gpu_boundary, _, _ = _resolve(coordinator, gpu_view, boundary)
            provisional = provisional or gpu_view.blocked
    return PrefixHitPlan(
        epoch, boundary, gpu_boundary,
        longest if longest > boundary else 0,
        groups, tuple(m.block_size for m in coordinator.single_type_managers),
        tuple(m.block_pool for m in coordinator.single_type_managers),
        proofs, hashes, max_length, provisional,
    )


class KeyLog:
    """Keys a pool inserted, in order (``BlockPool.key_log``). Bounded: a
    reader whose position fell off the front re-resolves."""

    LIMIT = 1 << 16

    def __init__(self) -> None:
        self.base = 0
        self.keys: list = []

    @property
    def end(self) -> int:
        return self.base + len(self.keys)

    def append(self, key) -> None:
        self.keys.append(key)
        if len(self.keys) > 2 * self.LIMIT:
            drop = len(self.keys) - self.LIMIT
            del self.keys[:drop]
            self.base += drop

    def since(self, position: int):
        """Keys inserted at or after ``position``; None if no longer known."""
        if position < self.base:
            return None
        return self.keys[position - self.base:]


class PrefixLease:
    """Whole-plan retention, separate from the bounded in-flight DMA credits.

    Used unchanged by waiter protection and admission. After issue, ownership
    passes to the transaction; cancellation cannot release it before quiescence.
    """

    def __init__(
        self,
        plan: PrefixHitPlan,
        version: VersionReader,
        current_epoch: Callable[[], int],
    ):
        self.plan = plan
        self.version = version
        self.current_epoch = current_epoch
        self._held: list[tuple[BlockPool, KVCacheBlock]] = []
        self.closed = False

    def valid(self) -> bool:
        if self.closed or self.current_epoch() != self.plan.epoch:
            return False
        for source in self.plan.dependencies:
            key = make_block_hash_with_group_id(source.key, source.group_id)
            if (
                source.pool.blocks[source.block.block_id] is not source.block
                or not source.pool.cached_block_hash_to_block.contain(
                    key, source.block.block_id
                )
                or source.block.block_hash != source.primary
                or self.version(
                    source.pool, source.block, source.group_id, source.key
                ) != source.version
            ):
                return False
        return True

    def acquire(self) -> bool:
        if self.closed:
            return False
        if self._held:
            valid = self.valid()
            if not valid:
                self.release()
            return valid
        if not self.valid():
            self.release()
            return False
        seen = set()
        try:
            for source in self.plan.dependencies:
                identity = (id(source.pool), id(source.block))
                if identity not in seen:
                    seen.add(identity)
                    source.pool.touch([source.block])
                    self._held.append((source.pool, source.block))
            if self.valid():
                return True
        except BaseException:
            self.release()
            raise
        self.release()
        return False

    def release(self) -> None:
        if self.closed:
            return
        for pool, block in reversed(self._held):
            pool.free_blocks([block])
        self._held.clear()
        self.closed = True


@dataclass
class PrefixHitAccounting:
    """Logical admission and successful transfer are different events."""

    union_hit_tokens: int = 0
    gpu_hit_tokens: int = 0
    union_gain_tokens: int = 0
    restored_requests: int = 0
    restored_host_blocks: int = 0
    _admitted: dict[Hashable, PrefixHitPlan] = field(default_factory=dict)
    _completed: set[Hashable] = field(default_factory=set)

    def admit(self, admission: Hashable, plan: PrefixHitPlan) -> None:
        # admission is (epoch, request_id, admission_generation), not a lookup.
        if admission in self._admitted:
            return
        self._admitted[admission] = plan
        self.union_hit_tokens += plan.boundary
        self.gpu_hit_tokens += plan.gpu_boundary
        self.union_gain_tokens += plan.union_gain

    def complete(self, admission: Hashable, *, success: bool) -> None:
        plan = self._admitted.get(admission)
        if plan is None or admission in self._completed:
            return
        self._completed.add(admission)
        if success:
            hosts = {
                (id(s.source.pool), id(s.source.block))
                for s in plan.slots if s.source.tier is Tier.HOST
            }
            if hosts:
                self.restored_requests += 1
                self.restored_host_blocks += len(hosts)

    def forget(self, admission: Hashable) -> None:
        """Retire bookkeeping after the owner has drained all callbacks."""
        self._admitted.pop(admission, None)
        self._completed.discard(admission)


def place_restore_destinations(
    coordinator: KVCacheCoordinator,
    request_id: str,
    restore_slots: Sequence[tuple[int, int]],
    num_computed_tokens: int,
    skip_zeroing_group_ids: Sequence[int] = (),
) -> None:
    """Materialize the host holes of a mixed two-tier hit.

    ``add_local_computed_blocks`` has adopted every leased GPU source and put
    a null placeholder at each host slot; this replaces each placeholder with
    a private, unhashed destination block (one pool draw each, already in the
    admission count). It runs after every group's GPU sources are touched,
    so a draw cannot evict a later group's source.

    A destination that holds the partial page at the boundary is private, so
    no CoW is needed (``_has_partial_local_hit`` sees the placeholder); its
    table index is left uncached so the page is cached once it fills.
    """
    if not restore_slots:
        return
    managers = coordinator.single_type_managers
    pool = coordinator.block_pool
    destinations = pool.get_new_blocks(len(restore_slots))
    for (gid, index), block in zip(restore_slots, destinations):
        manager = managers[gid]
        table = manager.req_to_blocks[request_id]
        if index >= len(table) or not table[index].is_null:
            raise RuntimeError(
                "host-tier restore slot has no placeholder in the block table"
            )
        table[index] = block
        if num_computed_tokens % manager.block_size and (
            index == num_computed_tokens // manager.block_size
        ):
            manager.num_cached_block[request_id] = min(
                manager.num_cached_block.get(request_id, index), index
            )
        # The async load writes the block; zeroing would race it.
        if manager._record_new_block_ids and gid not in skip_zeroing_group_ids:
            manager.new_block_ids.append(block.block_id)


def adopted_draws(
    coordinator: KVCacheCoordinator,
    new_computed_blocks: Sequence[Sequence[KVCacheBlock]],
    total_computed_tokens: int,
) -> int:
    """The cached-free hit blocks an admission takes off the free queue by
    adoption, counted exactly as each manager's ``get_num_blocks_to_allocate``
    counts them. They are draws, but not reclaims from the queue
    head, so the write-back gate's victim simulation must not pop them again.
    """
    from vllm.v1.core.single_type_kv_cache_manager import MambaManager

    total = 0
    for manager, blocks in zip(coordinator.single_type_managers, new_computed_blocks):
        if not blocks:
            continue
        if isinstance(manager, MambaManager) and manager.mamba_cache_mode == "align":
            window = blocks
        else:
            skipped = manager.get_num_skipped_tokens(total_computed_tokens)
            window = blocks[skipped // manager.block_size :]
        total += manager._get_num_evictable_blocks(window)
    return total
