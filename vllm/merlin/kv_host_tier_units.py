# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)

"""Host-RAM KV tier: restore-point units and host eviction.

A unit records the FA chain, boundary-window SW pages (including proof
pages), and exact recurrent endpoints needed for one restore boundary.
Members are found by key in the GPU/host union directory.

Write-back owns each unit transaction and its row references. Copied rows
stay unkeyed until the final all-rank acknowledgement. If a caller keys
them earlier, is_staged also hides rows belonging only to unpublished
units. Only publish_unit makes a new host unit visible and registers
its exclusive rows as a coherent host eviction group.

Host eviction is per unit over its EXCLUSIVE
rows: a row shared with another unit (a common system prompt, a fork, a
superseded predecessor) stays keyed while any owner remains and is freed
with its last owner, so a shared prefix never makes the whole host one
victim. Victim
classes, in order: idle ownerless
rows, units whose rows all duplicate GPU blocks held by running requests,
units by LRU of host use (optionally probation before protected,
``kv_host_tier_policy``), units whose twins are about to leave the GPU.
There is no retention interval. Lost units' exclusive host copies are
unkeyed; there is no fragment directory or cross-tier value order.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.v1.core.kv_cache_utils import (
    BlockHashWithGroupId,
    KVCacheBlock,
    get_block_hash,
    get_group_id,
    make_block_hash_with_group_id,
)
from vllm.v1.core.kv_tier_state import BlockState

if TYPE_CHECKING:
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.request import Request

logger = init_logger(__name__)

UnitId = int


class GroupKind(Enum):
    FULL = "full"
    SLIDING = "sliding"
    RECURRENT = "recurrent"


@dataclass(frozen=True)
class GroupInfo:
    group_id: int
    kind: GroupKind
    block_size: int
    sliding_window: int = 0
    # Sliding window: tokens retained below a hit beyond the window
    # (``SlidingWindowSpec.extra_retained_tokens``).
    extra_retained_tokens: int = 0
    # The lookup drops one hash unit from a hit in this group (an EAGLE
    # group, ``KVCacheCoordinator.eagle_group_ids``): the window it proves
    # starts one unit earlier.
    eagle: bool = False
    # The group's resolved fine-grained lookup policy
    # (``kv_restore_dependencies.fine_grained_lookup``); None only for
    # tables built without a coordinator: the page size decides then.
    fine: bool | None = None


def key_order(key: BlockHashWithGroupId) -> tuple[int, bytes]:
    """The total order of host-tier keys: (group id, hash
    bytes). Never set / frozenset / dict-of-set iteration order: bytes
    hashes are salted per process (PYTHONHASHSEED), so hash order made the
    same workload store, evict and restore differently per process."""
    return (get_group_id(key), bytes(get_block_hash(key)))


def ordered_keys(keys) -> list[BlockHashWithGroupId]:
    """``keys`` (any iterable, a set included) in ``key_order``."""
    return sorted(keys, key=key_order)


@dataclass
class RestoreUnit:
    uid: int
    boundary: int
    # Keys a restore at ``boundary`` needs, in every group.
    required: frozenset[BlockHashWithGroupId]
    # Recurrent endpoints (a subset of ``required``).
    endpoints: frozenset[BlockHashWithGroupId]
    # Kept with the unit when present. Empty now: the sliding-window
    # EAGLE proof page is a required member when the group drops a unit.
    optional: frozenset[BlockHashWithGroupId] = frozenset()
    host_group: bool = False
    # The keys in chain order (group by group as the table lists them, pages
    # ascending; endpoints where their group sits), as ``manifest`` derived
    # them. Empty for a unit built by hand: ``UnitTable.ordered``
    # falls back to ``key_order``. Decisions iterate this, never ``keys``.
    order: tuple[BlockHashWithGroupId, ...] = ()

    @property
    def keys(self) -> frozenset[BlockHashWithGroupId]:
        return self.required | self.optional


@dataclass
class UnitCounters:
    units_recorded: int = 0
    units_incomplete: int = 0
    units_superseded: int = 0
    units_host_registered: int = 0  # demoted_units
    units_lost: int = 0  # unit_lost
    unit_lost_host_blocks: int = 0
    dup_reclaimed: int = 0  # host twins reused while the GPU held the key
    # Eviction (the iface COUNTERS_V3 names).
    evict_exclusive_rows: int = 0  # rows made idle by ``apply``
    evict_shared_kept: int = 0  # shared rows a victim left to its co-owners
    evict_skipped_leased: int = 0  # candidates skipped: a pinned/protected row
    evict_running_dup_rows: int = 0
    evict_imminent_dup_rows: int = 0
    evict_units: int = 0
    evict_plans: int = 0
    evict_plans_short: int = 0  # plans that could not meet the need
    evict_apply_mismatch: int = 0  # apply freed other rows than planned
    supersede_deferred: int = 0  # published predecessors kept for a successor
    supersede_completed: int = 0  # ... dropped once the successor published
    constrained_successor: int = 0  # a predecessor evicted for its successor
    # Staged rows of an abandoned unit that lost their host keys before the unit
    # restarted (a stall release, or the per-unit stall release giving up its
    # start): the copy work the liveness release threw away.
    abandoned_fragment_evicted: int = 0
    # Fallback boundary (flag ``FALLBACK_BOUNDARY``; 0 with it off): a
    # published predecessor kept as its published successor's fallback, and
    # the times host eviction took a successor while its fallback stayed
    # restorable (the session's host prefix shrank to the predecessor's
    # boundary instead of going whole).
    fallback_kept: int = 0
    fallback_shrinks: int = 0


# Victim classes, in eviction order (iface ``EvictionPolicy``).
CLASS_RUNNING_DUP = "running_dup"
CLASS_LRU = "lru"
CLASS_IMMINENT_DUP = "imminent_dup"
_CLASS_RANK = {CLASS_RUNNING_DUP: 0, CLASS_LRU: 1, CLASS_IMMINENT_DUP: 2}


@dataclass(frozen=True)
class EvictionPlan:
    """A pure host-eviction selection (``UnitTable.plan``); nothing is
    unkeyed until ``UnitTable.apply``, which frees exactly ``rows``.

    ``victims``: units in application order. ``rows``: the unique host row
    ids the plan makes idle (each victim's exclusive rows at its turn; a row
    shared with a surviving unit is never listed). ``classes``: rows per
    class, keys ``idle`` (idle ownerless rows already available, freed by
    nobody), ``running_dup``, ``lru``, ``imminent_dup``. ``displace_published``:
    whether the plan may take published host-only content (a lane that may
    not gets idle rows and running-twin duplicates only)."""

    need_rows: int
    victims: list[int] = field(default_factory=list)
    rows: list[int] = field(default_factory=list)
    classes: dict[str, int] = field(default_factory=dict)
    displace_published: bool = True
    idle: int = 0
    version: int = -1
    step: int = 0
    for_units: tuple[int, ...] = ()

    @property
    def satisfied(self) -> bool:
        """The idle rows plus the plan's rows meet the need."""
        return self.idle + len(self.rows) >= self.need_rows

    @property
    def empty(self) -> bool:
        return not self.victims


@dataclass
class _Candidate:
    """One unit's eviction view, cached per (directory version, step)."""

    uid: int
    rows: tuple[KVCacheBlock, ...]  # unique host rows of the unit
    owners: tuple[frozenset[int], ...]  # owners of each row (aliases included)
    cls: str
    last_use: int
    published: bool
    abandoned: bool
    rank: int = 0  # admission-policy segment rank (0 probation, 1 protected)


def unit_key(
    block_hashes: Sequence, hash_block_size: int, tokens: int, group_id: int
) -> BlockHashWithGroupId:
    """The key of group ``group_id`` at hash boundary ``tokens``."""
    assert tokens > 0 and tokens % hash_block_size == 0
    return make_block_hash_with_group_id(
        block_hashes[tokens // hash_block_size - 1], group_id
    )


@dataclass(frozen=True)
class UnitManifest:
    """What a restore unit at ``boundary`` needs (``UnitTable.manifest``)."""

    boundary: int
    required: frozenset[BlockHashWithGroupId]
    endpoints: frozenset[BlockHashWithGroupId]
    optional: frozenset[BlockHashWithGroupId]
    # The recorded unit with exactly these required keys, if any:
    # ``UnitTable.manifest`` fills it from ``_recorded_uid``, so the planner does
    # not match keys itself.
    uid: int | None = None
    # ``required`` then ``optional`` in chain order.
    order: tuple[BlockHashWithGroupId, ...] = ()


class Incomplete(Enum):
    """``UnitTable.manifest``: a boundary was found but its unit is
    incomplete (proof past the chain, or a required key absent)."""

    INCOMPLETE = "incomplete"


INCOMPLETE = Incomplete.INCOMPLETE


class _ProofBeyondChain(Exception):
    """A unit's proof page lies past the request's recorded hash chain."""


class UnitTable:

    def __init__(
        self,
        gpu_pool: BlockPool,
        host_pool: BlockPool,
        groups: Sequence[GroupInfo],
        hash_block_size: int,
    ) -> None:
        from vllm.merlin.envs import read_registered

        self.gpu_pool = gpu_pool
        self.host_pool = host_pool
        self.groups = list(groups)
        self.hash_block_size = hash_block_size
        self._recurrent = [g for g in self.groups if g.kind is GroupKind.RECURRENT]
        self._full = [g for g in self.groups if g.kind is GroupKind.FULL]
        assert self._full, "a restore unit needs a full-attention group"
        self._kind = {g.group_id: g.kind for g in self.groups}
        self._sliding = [g for g in self.groups if g.kind is GroupKind.SLIDING]
        # Step retention is deleted. Host victims are chosen by
        # class (idle -> running-twin duplicates -> LRU of host use ->
        # imminent twins) over each unit's EXCLUSIVE rows; shared rows stay
        # while any owner remains (``plan`` / ``apply``).
        from vllm.merlin.kv_host_tier_policy import make_admission_policy

        mode = read_registered("MERLIN_KV_HOST_TIER__ADMISSION")
        self.policy = make_admission_policy(mode)
        # Fallback boundary (default off): a published successor keeps
        # its deepest published, host-complete predecessor as a live unit,
        # ordered right behind it for eviction. Evicting the successor then
        # frees only its own exclusive rows (the FA tail past the
        # predecessor's boundary, its recurrent endpoint and SW window) and
        # the predecessor stays restorable: the session's host prefix shrinks
        # to the predecessor boundary instead of going whole. No copy and no
        # publication: every kept row was published (all-rank acked) with
        # the predecessor, and only its ownership changes. Off: the
        # predecessor is dropped when its successor publishes, as before.
        self._fallback_on = bool(
            read_registered("MERLIN_KV_HOST_TIER__FALLBACK_BOUNDARY")
        )
        self._fallback_of: dict[int, int] = {}  # predecessor -> successor
        self._fallback: dict[int, int] = {}  # successor -> predecessor
        # Classifies a GPU twin: "running" (a running request's table holds
        # it), "imminent" (grace-held / in the free queue: it leaves the GPU
        # next). Installed by the copy scheduler (SessionYield knows graces);
        # the fallback reads the ref count alone (grace pins count as running).
        self.twin_kind: Callable[[KVCacheBlock], str] | None = None
        # Backing revocation (the copy scheduler's REPAIR lane): ``apply`` calls
        # it with the GPU twins of every host row it evicts while the GPU still
        # holds the key (duplicate victims), so that GPU content is dirty again
        # and eligible for a repair copy. ``hook(gpu_blocks, step)``.
        self.revoke_hook: Callable[[list[KVCacheBlock], int], None] | None = None
        # Directory version: bumped by every change of units, keys or host
        # ownership; the eviction candidate index is rebuilt only when it (or
        # the engine step) changes, never per call.
        self._dir_version = 0
        self._cand_cache: tuple[tuple[int, int], list[_Candidate]] | None = None
        # Supersede-at-publish: a published predecessor stays restorable until
        # its successor publishes (pred uid -> successor uid, and back).
        self._superseded_by: dict[int, int] = {}
        self._pending_preds: dict[int, set[int]] = {}
        # Miss causes: why each recent unit was
        # lost, by the plain hash of every key it required, so a later lookup
        # that recomputes the session can name the cause. Bounded.
        self.lost: OrderedDict[object, tuple[str, int]] = OrderedDict()
        self._loss_cause: str | None = None
        # Walker decisions on units it did not copy: uid -> cause.
        self._skip_cause: dict[int, str] = {}
        self.step = 0
        self.units: dict[int, RestoreUnit] = {}
        self._key_order: dict[int, list[BlockHashWithGroupId]] = {}
        self._ordered_cache: dict[int, tuple[RestoreUnit, list]] = {}
        self._key_units: dict[BlockHashWithGroupId, set[int]] = {}
        self._published: set[int] = set()
        # Previously published rows stay visible across unit supersession.
        self._visible: set[int] = set()
        self._last_use: dict[int, int] = {}
        # Rows of a unit still assembling are owned
        # by its transaction until it publishes or is lost: one ref each, so
        # the host LRU cannot pop a half-copied unit's rows for another copy.
        self._held: dict[int, list[KVCacheBlock]] = {}
        self._pending_unhashed: list[BlockHashWithGroupId] = []
        self._next_uid = 0
        self.counters = UnitCounters()
        self._reclaiming: int | None = None
        # Eviction callbacks run before the directory removes the block.
        self._unkeying: set[int] = set()
        # While a failed load is invalidated, a lost unit's
        # published rows outside the read set keep their keys (they are
        # intact, all-rank acked copies; ``keep_lost_remains``).
        self._keep_lost_remains = False
        # Host rows the write-back walker protects from
        # eviction (``protect``): every row a starting or in-progress unit
        # depends on, its predecessor's visible prefix rows included.
        self._protected_rows: frozenset[int] = frozenset()
        # Rule A, "complete units outrank fragments": while a unit-less
        # copy (an intermediate recurrent checkpoint) makes room
        # (``lone_victims``), no component with a published owner is a
        # victim. An intermediate boundary may then recompute on a host full
        # of complete units; output is identical.
        self._lone_mode = False
        # Units whose start the walker gave up (no copy
        # progress): landed rows are not held for them until they restart.
        self._abandoned: set[int] = set()

    # -- directory ------------------------------------------------------------

    def _gpu_block(self, key: BlockHashWithGroupId) -> KVCacheBlock | None:
        """The GPU block under ``key``. When several blocks carry the
        key (a partial-hit CoW twin registers its owner's prompt-tail key
        again), one that is not a running request's write frontier comes
        first, so a finished turn's unit resolves to its own copy, never to
        the page another request is still decoding into."""
        cache = self.gpu_pool.cached_block_hash_to_block
        block = cache.get_one_block(key)
        tier = self.gpu_pool.tier
        if block is None or tier is None or not tier.is_frontier(block):
            return block
        for twin in cache.get_blocks(key):
            if not tier.is_frontier(twin):
                return twin
        return block

    def _host_block(self, key: BlockHashWithGroupId) -> KVCacheBlock | None:
        return self.host_pool.cached_block_hash_to_block.get_one_block(key)

    def _present(self, key: BlockHashWithGroupId) -> bool:
        return self._gpu_block(key) is not None or self._host_block(key) is not None

    def _host_holds_other(
        self, key: BlockHashWithGroupId, block: KVCacheBlock
    ) -> bool:
        entry = self.host_pool.cached_block_hash_to_block._cache.get(key)
        if entry is None:
            return False
        if isinstance(entry, KVCacheBlock):
            return entry is not block
        return any(b is not block for b in entry.values())

    @staticmethod
    def _keys_of(pool: BlockPool, block: KVCacheBlock) -> list[BlockHashWithGroupId]:
        keys = [] if block.block_hash is None else [block.block_hash]
        # Aliases live in a set of bytes (``BlockPool``): hash order is per
        # process, so they are listed in ``key_order``.
        keys.extend(ordered_keys(
            pool.cached_block_hashes_by_block.get(block.block_id, ())
        ))
        return keys

    def units_of(self, key: BlockHashWithGroupId) -> list[RestoreUnit]:
        return [self.units[u] for u in sorted(self._key_units.get(key, ()))]

    def ordered(self, unit: RestoreUnit) -> list[BlockHashWithGroupId]:
        """The unit's keys in a total order: chain order when the
        manifest recorded it, else ``key_order``. Every member, row, copy
        and loss walk over a unit's keys goes through this."""
        got = self._ordered_cache.get(unit.uid)
        if got is not None and got[0] is unit:
            return got[1]
        keys = unit.keys
        if unit.order and set(unit.order) == keys:
            out = list(unit.order)
        else:
            out = ordered_keys(keys)
        self._ordered_cache[unit.uid] = (unit, out)
        return out

    def unit_of_block(self, block: KVCacheBlock) -> list[RestoreUnit]:
        """The units a GPU or host block's keys belong to (empty: a
        singleton eviction object). Includes a published predecessor kept
        for its successor (supersede-at-publish); ``is_superseded`` tells
        them apart (touch scoping should prefer the successor)."""
        pool = self.gpu_pool if block.pool is not self.host_pool else self.host_pool
        seen: dict[int, RestoreUnit] = {}
        for key in self._keys_of(pool, block):
            for unit in self.units_of(key):
                seen[unit.uid] = unit
        return list(seen.values())

    def is_superseded(self, unit_id: UnitId) -> bool:
        """A published predecessor kept until its successor publishes."""
        return unit_id in self._superseded_by

    def gpu_members(self, unit: RestoreUnit) -> list[KVCacheBlock]:
        """The unit's GPU-resident members, endpoints last (the walker issues
        the whole list; a missing member is host-resident or lost)."""
        out: list[KVCacheBlock] = []
        tail: list[KVCacheBlock] = []
        seen: set[int] = set()
        # The key order is immutable per unit: sorted once.
        order = self._key_order.get(unit.uid)
        if order is None:
            # Stable sort of the total order: endpoints last.
            order = self._key_order[unit.uid] = sorted(
                self.ordered(unit), key=lambda k: k in unit.endpoints
            )
        for key in order:
            block = self._gpu_block(key)
            if block is None or block.block_id in seen:
                continue
            seen.add(block.block_id)
            (tail if key in unit.endpoints else out).append(block)
        return out + tail

    @property
    def has_recurrent(self) -> bool:
        return bool(self._recurrent)

    def host_members(self, unit: RestoreUnit) -> list[KVCacheBlock]:
        out: list[KVCacheBlock] = []
        seen: set[int] = set()
        for key in self.ordered(unit):
            block = self._host_block(key)
            if block is not None and block.block_id not in seen:
                seen.add(block.block_id)
                out.append(block)
        return out

    def is_host_complete(self, unit: RestoreUnit) -> bool:
        return all(self._host_block(k) is not None for k in unit.required)

    # -- recording --------------------------------------------------------------

    def _find_boundary(self, block_hashes: Sequence, limit: int) -> int:
        hbs = self.hash_block_size
        anchor = self._recurrent or self._full[:1]
        for tokens in range(limit, 0, -hbs):
            if all(
                self._present(unit_key(block_hashes, hbs, tokens, g.group_id))
                for g in anchor
            ):
                return tokens
        return 0

    def _covering_key(
        self, block_hashes: Sequence, boundary: int, size: int, gid: int
    ) -> BlockHashWithGroupId:
        """The key that proves an append-only page covers ``boundary`` inside
        it: the key at ``boundary`` when present, else the highest
        present key later in the same page (as the lookup's finders accept,
        ``kv_tier_lookup._UnionView.get_cached_block`` and the sliding-window
        ``highest_keyed_in``), else the key at ``boundary`` (absent: the unit
        is incomplete)."""
        hbs = self.hash_block_size
        exact = unit_key(block_hashes, hbs, boundary, gid)
        if self._present(exact):
            return exact
        page_end = min((boundary // size + 1) * size, len(block_hashes) * hbs)
        for tokens in range(page_end, boundary, -hbs):
            key = unit_key(block_hashes, hbs, tokens, gid)
            if self._present(key):
                return key
        return exact

    def group_margin(self, g: GroupInfo) -> int:
        """Tokens past an accepted boundary an EAGLE group's lookup proves
        before dropping them (0 without EAGLE): the finder's own margin."""
        if not g.eagle:
            return 0
        from vllm.v1.core.kv_cache_utils import eagle_proof_margin

        fine = g.fine if g.fine is not None else g.block_size > self.hash_block_size
        return eagle_proof_margin(g.block_size, self.hash_block_size, fine)

    def _required_keys(
        self, block_hashes: Sequence, boundary: int
    ) -> tuple[set, set, set]:
        """Every key a restore at the ACCEPTED boundary reads, from the one
        geometry capacity sizing uses (``group_dependency_pages``): FA pages
        [0, B) and the page holding B proved by a covering key;
        sliding window the retained tail below B; EAGLE groups additionally
        the proof pages the finder probes up to B + margin;
        recurrent the endpoint at exactly B. A proof beyond the recorded hash
        chain cannot be keyed: ``_ProofBeyondChain``."""
        from vllm.merlin.kv_restore_dependencies import group_dependency_pages

        hbs = self.hash_block_size
        chain = len(block_hashes) * hbs
        # Insertion-ordered (dicts): the chain order ``manifest`` records
        # as ``order``. Callers that only test membership are fine.
        required: dict[BlockHashWithGroupId, None] = {}
        endpoints: dict[BlockHashWithGroupId, None] = {}
        for g in self.groups:
            gid, size = g.group_id, g.block_size
            if g.kind is GroupKind.RECURRENT:
                key = unit_key(block_hashes, hbs, boundary, gid)
                required[key] = None
                endpoints[key] = None
                continue
            margin = self.group_margin(g)
            data, proof = group_dependency_pages(
                "full" if g.kind is GroupKind.FULL else "sliding",
                boundary,
                size,
                sliding_window=g.sliding_window,
                extra_retained_tokens=g.extra_retained_tokens,
                margin=margin,
            )
            for page in (*data, *proof):
                end = (page + 1) * size
                # The EAGLE finder proves B + margin. When that lands inside
                # the data page holding B there is no separate proof page:
                # the data page itself must be covered through B + margin,
                # or the live lookup drops below B.
                # Pages ending at or below B are unaffected (end <= reach).
                reach = boundary + margin
                if reach > chain:
                    raise _ProofBeyondChain
                if end <= reach:
                    required[unit_key(block_hashes, hbs, end, gid)] = None
                else:
                    required[self._covering_key(block_hashes, reach, size, gid)] = None
        return required.keys(), endpoints.keys(), {}.keys()

    def manifest(self, request: Request) -> UnitManifest | Incomplete | None:
        """The manifest of ``request``'s deepest restore boundary as the
        directory proves it now: what ``record_finished`` records. Read-only:
        no flush, no counter, nothing recorded (the census reads it).
        None when there is no boundary; ``INCOMPLETE`` when the boundary's
        proof lies past the hash chain or a required key is absent."""
        block_hashes = request.block_hashes
        hbs = self.hash_block_size
        limit = min(request.num_computed_tokens, len(block_hashes) * hbs)
        # An EAGLE group proves B + margin: the accepted boundary leaves room
        # for its proof inside the keyed chain.
        limit -= max((self.group_margin(g) for g in self.groups), default=0)
        limit -= limit % hbs
        if limit <= 0:
            return None
        boundary = self._find_boundary(block_hashes, limit)
        if boundary == 0:
            return None
        try:
            required, endpoints, optional = self._required_keys(
                block_hashes, boundary
            )
        except _ProofBeyondChain:
            return INCOMPLETE
        if not all(self._present(k) for k in required):
            return INCOMPLETE
        present_optional = [k for k in optional if self._present(k)]
        order = (*required, *(k for k in present_optional if k not in required))
        required = frozenset(required)
        return UnitManifest(
            boundary,
            required,
            frozenset(endpoints),
            frozenset(present_optional),
            self._recorded_uid(required),
            order,
        )

    def _recorded_uid(self, required: frozenset) -> int | None:
        """The recorded unit whose required keys are exactly ``required``
        (the owners of one key, checked for equality)."""
        if not required:
            return None
        probe = next(iter(required))
        for uid in sorted(self._key_units.get(probe, ())):
            unit = self.units.get(uid)
            if unit is not None and unit.required == required:
                return uid
        return None

    def record_finished(self, request: Request) -> RestoreUnit | None:
        """Record the unit of ``request``'s deepest restore boundary."""
        self.flush()
        found = self.manifest(request)
        if found is None:
            return None
        if isinstance(found, Incomplete):
            self.counters.units_incomplete += 1
            return None
        unit = RestoreUnit(
            self._next_uid,
            found.boundary,
            found.required,
            found.endpoints,
            found.optional,
            order=found.order,
        )
        self._next_uid += 1
        self._bump()
        # Supersede older units this one contains (the session's next turn).
        older: set[int] = set()
        keys = self.ordered(unit)  # A total order, never set order
        for key in keys:
            older.update(self._key_units.get(key, ()))
        inherited_use: int | None = None
        inherited_publication = False
        # Landed rows the superseded units held: the successor owns them and
        # is unpublished, so their refs pass to it (else they leak).
        inherited_rows: list[KVCacheBlock] = []
        # Supersede-at-publish: a PUBLISHED predecessor stays a complete,
        # restorable unit until this successor publishes (a lost successor
        # then falls back to it: one turn of recompute, output identical).
        # An unpublished one is superseded now, as before (nothing restores
        # from it; its landed rows pass to the successor).
        keep_preds: set[int] = set()
        for uid in sorted(older):
            old = self.units[uid]
            if not (old.keys - old.endpoints) <= unit.keys:
                continue
            # The successor keeps the newest use of what it supersedes,
            # published or not, so LRU does not age it.
            last = self._last_use.get(old.uid)
            if last is not None:
                inherited_use = (
                    last if inherited_use is None else max(last, inherited_use)
                )
            self.counters.units_superseded += 1
            if old.uid in self._published:
                inherited_publication = True
                keep_preds.add(old.uid)
                continue
            inherited_rows.extend(self._held.pop(old.uid, ()))
            # Its own kept predecessors now wait for this successor.
            keep_preds |= self._pending_preds.pop(old.uid, set())
            self._drop_unit(old)
        groups = self.host_pool.evict_groups
        for key in keys:
            # Shared with a unit that stays (a fork), through this key or any
            # alias of its host block: out of that unit's group.
            host = self._host_block(key)
            if host is not None and self._block_owners(host):
                groups.leave(host)
            self._key_units.setdefault(key, set()).add(unit.uid)
        self.units[unit.uid] = unit
        self.counters.units_recorded += 1
        # Preserve visibility only when every member was already published
        # by an older unit transaction. New rows still require publication.
        if (
            inherited_publication and self.is_host_complete(unit)
            and all(b.block_id in self._visible for b in self.host_members(unit))
        ):
            self._published.add(unit.uid)
        if inherited_use is not None:
            self._last_use[unit.uid] = max(
                inherited_use, self._last_use.get(unit.uid, inherited_use)
            )
        if inherited_rows:
            self._hand_over(inherited_rows, exclude=None)
        keep_preds &= set(self.units)
        if keep_preds:
            if unit.uid in self._published:
                # Published at record (every member already visible): the
                # policy places it, inheriting its predecessors' segment,
                # then the predecessors are replaced now.
                self.policy.on_publish(
                    unit.uid, len(self.host_members(unit)),
                    inherit=keep_preds, keys=self._plain_keys(unit),
                )
                self._settle_preds(unit.uid, keep_preds)
            else:
                for pred in sorted(keep_preds):
                    prev = self._superseded_by.get(pred)
                    if prev is not None and prev != unit.uid:
                        self._pending_preds.get(prev, set()).discard(pred)
                    elif prev is None:
                        self.counters.supersede_deferred += 1
                    self._superseded_by[pred] = unit.uid
                self._pending_preds.setdefault(unit.uid, set()).update(keep_preds)
        self._maybe_register_host(unit)
        return unit

    def _complete_supersede(self, pred_uid: UnitId) -> None:
        """The successor published: drop its predecessor. Rows it shared
        with the successor stay (the successor owns them); its exclusive rows
        (the old recurrent endpoint) become ownerless LRU content."""
        pred = self.units.get(pred_uid)
        if pred is None:
            return
        self.counters.supersede_completed += 1
        self._drop_unit(pred)

    def retains(self, block: KVCacheBlock) -> bool:
        """SW pages are retained only in a recorded boundary's dependency
        window. The manifest already includes its data and proof pages;
        running requests need no incremental endpoint scan here."""
        key = block.block_hash
        if key is None or self._kind.get(get_group_id(key)) is not GroupKind.SLIDING:
            return True
        return any(
            self._key_units.get(k) for k in self._keys_of(self.gpu_pool, block)
        )

    def hold_unpublished(self, cpu_blocks: Iterable[KVCacheBlock]) -> None:
        """Store completion, before the transfer refs drop: keep one ref on
        each landed row of a unit that is not published yet."""
        self._bump()
        hold: list[KVCacheBlock] = []
        for block in cpu_blocks:
            if block.is_null:
                continue
            # One ref per unpublished owner (a shared row is held by each),
            # so every owner stays assembling and its refs balance on its own
            # publish, loss or reset, or on a co-owner's publish, which makes
            # the row visible (``_release_visible_holds``).
            for unit in self.unit_of_block(block):
                if unit.uid in self._published or unit.uid in self._abandoned:
                    continue
                rows = self._held.setdefault(unit.uid, [])
                if all(b.block_id != block.block_id for b in rows):
                    rows.append(block)
                    hold.append(block)
        if hold:
            self.host_pool.touch(hold)

    @property
    def assembling(self) -> int:
        """Units holding rows while they assemble."""
        return len(self._held)

    def _release_held(self, uid: int) -> None:
        rows = self._held.pop(uid, None)
        if rows:
            self._bump()
            self._hand_over(rows, exclude=uid)

    def _hand_over(self, rows, exclude: int | None) -> None:
        """Release one ref on each of ``rows``, except that a row still owned
        by another unpublished unit that holds no ref on it passes its ref
        to that unit (it stays assembling, never a ref-0 staged row). An
        abandoned owner is never an heir: it holds nothing
        until it restarts, so a row handed to it would stay ref 1 for good.
        A visible (published) row has no heir: it is a published
        unit's lookup source, never a staged row."""
        free: list[KVCacheBlock] = []
        for block in rows:
            if block.block_id in self._visible:
                free.append(block)
                continue
            heir = None
            for uid in sorted(self._block_owners(block)):
                if (
                    uid == exclude or uid in self._published
                    or uid in self._abandoned or uid not in self.units
                ):
                    continue
                held = self._held.get(uid, ())
                if all(b.block_id != block.block_id for b in held):
                    heir = uid
                    break
            if heir is None:
                free.append(block)
            else:
                self._held.setdefault(heir, []).append(block)
        if free:
            self.host_pool.free_blocks(free)

    def _release_visible_holds(self, rows: Iterable[KVCacheBlock]) -> None:
        """``rows`` were just published. ``hold_unpublished`` held each
        landed row once per UNPUBLISHED owner, so the co-owner of a fork's
        shared page (or of a shared prefix) held it too while nothing was
        copying that co-owner. It stayed assembling, holding the row and a
        MAX_ASSEMBLING slot, until its own members were copied or lost,
        possibly to the end of the run. A published row needs no assembly
        hold: it stays keyed while any owner remains, and the walker
        protects the host rows of every unit in progress. Staged (invisible)
        rows keep their holds."""
        ids = {b.block_id for b in rows}
        free: list[KVCacheBlock] = []
        for uid in sorted(self._held):
            held = self._held[uid]
            keep = [b for b in held if b.block_id not in ids]
            if len(keep) == len(held):
                continue
            free.extend(b for b in held if b.block_id in ids)
            if keep:
                self._held[uid] = keep
            else:
                del self._held[uid]
        if free:
            self._bump()
            self.host_pool.free_blocks(free)

    def release_all_held(self) -> None:
        rows = [b for held in self._held.values() for b in held]
        self._held.clear()
        if rows:
            self.host_pool.free_blocks(rows)

    def _drop_unit(self, unit: RestoreUnit) -> None:
        self._bump()
        if self._fallback or self._fallback_of:
            self._unlink_fallback(unit.uid)
        # Supersede-at-publish links: a dropped successor releases its kept
        # predecessors (they stay ordinary published units, the fallback
        # boundary); a dropped predecessor leaves its successor's list.
        succ = self._superseded_by.pop(unit.uid, None)
        if succ is not None:
            preds = self._pending_preds.get(succ)
            if preds is not None:
                preds.discard(unit.uid)
                if not preds:
                    del self._pending_preds[succ]
        for pred in self._pending_preds.pop(unit.uid, ()):
            self._superseded_by.pop(pred, None)
        self.policy.on_drop(unit.uid)
        self._release_held(unit.uid)
        self._abandoned.discard(unit.uid)
        if unit.host_group:
            self._ungroup_host(unit)
        self.units.pop(unit.uid, None)
        self._key_order.pop(unit.uid, None)
        self._published.discard(unit.uid)
        self._last_use.pop(unit.uid, None)
        self._skip_cause.pop(unit.uid, None)
        # The order first: ``ordered`` caches, so popping before it would keep
        # every dropped unit alive for the process.
        keys = self.ordered(unit)
        self._ordered_cache.pop(unit.uid, None)
        for key in keys:
            owners = self._key_units.get(key)
            if owners is not None:
                owners.discard(unit.uid)
                if not owners:
                    del self._key_units[key]

    def _unlink_fallback(self, uid: UnitId) -> None:
        """``uid`` leaves the table. A successor's kept fallback becomes
        the session's restore point: an ordinary unit with the successor's
        recency (it rode right behind it). A fallback leaving just unlinks."""
        pred = self._fallback.pop(uid, None)
        if pred is not None:
            self._fallback_of.pop(pred, None)
            if pred in self.units:
                if self._loss_cause == "host_evicted":
                    self.counters.fallback_shrinks += 1
                last = self._last_use.get(uid)
                if last is not None:
                    self._last_use[pred] = max(
                        last, self._last_use.get(pred, last)
                    )
        succ = self._fallback_of.pop(uid, None)
        if succ is not None:
            self._fallback.pop(succ, None)

    # -- transaction publication -----------------------------------------------

    def is_staged(self, block: KVCacheBlock) -> bool:
        """Only transaction-published host rows are lookup sources.
        Published shared rows stay visible across unit supersession."""
        # Rows no unit owns (seeded sources, non-unit copies) are not held
        # back: only a unit's own rows wait for its publication.
        return block.block_id not in self._visible and bool(self._block_owners(block))

    def publish_unit(self, unit_id: UnitId, step: int) -> bool:
        """Publish after write-back validates the final all-rank ack and keys
        every member. This table neither validates DMA nor owns row refs.
        False means the recorded unit was lost or is not host-complete."""
        self.flush()
        unit = self.units.get(unit_id)
        if unit is None or not self.is_host_complete(unit):
            return False
        self._bump()
        self._published.add(unit_id)
        members = self.host_members(unit)
        self._visible.update(b.block_id for b in members)
        # Visible first, so neither its own release nor a co-owner's
        # hold keeps a now-published row as an assembly hold.
        self._release_held(unit_id)
        self._release_visible_holds(members)
        self.touch(unit_id, step)
        self._maybe_register_host(unit)
        # The admission policy (flag) places the unit: a returning session's
        # successor inherits its predecessor's segment, a ghost hit enters
        # protected, anything else probation.
        preds = self._pending_preds.get(unit_id, set())
        self.policy.on_publish(
            unit_id, len(members), inherit=preds, keys=self._plain_keys(unit)
        )
        # Supersede-at-publish: the successor is restorable now.
        # The deepest restorable
        # predecessor stays as its fallback when the flag is on.
        preds = sorted(self._pending_preds.pop(unit_id, set()))
        keep = self._fallback_pick(unit_id, preds) if self._fallback_on else None
        for pred in preds:
            self._superseded_by.pop(pred, None)
            if pred != keep:
                self._complete_supersede(pred)
        if keep is not None:
            self._keep_fallback(unit_id, keep)
        return True

    def _settle_preds(self, succ_uid: UnitId, preds: Iterable[UnitId]) -> None:
        """The published successor ``succ_uid`` replaces ``preds``: drop
        them (flag off: every one), or with the fallback-boundary flag keep the
        deepest restorable one as its fallback and drop the rest."""
        preds = sorted(preds)
        keep = self._fallback_pick(succ_uid, preds) if self._fallback_on else None
        for pred in preds:
            if pred != keep:
                self._complete_supersede(pred)
        if keep is not None:
            self._keep_fallback(succ_uid, keep)

    def _fallback_pick(
        self, succ_uid: UnitId, preds: Sequence[UnitId]
    ) -> UnitId | None:
        """The deepest predecessor that is a restore point now (published,
        every required key on the host, a shorter boundary)."""
        succ = self.units.get(succ_uid)
        if succ is None:
            return None
        best: RestoreUnit | None = None
        for uid in preds:
            pred = self.units.get(uid)
            if (
                pred is None or uid not in self._published
                or pred.boundary >= succ.boundary
                or not self.is_host_complete(pred)
            ):
                continue
            if best is None or (pred.boundary, uid) > (best.boundary, best.uid):
                best = pred
        return None if best is None else best.uid

    def _keep_fallback(self, succ_uid: UnitId, pred_uid: UnitId) -> None:
        """Keep ``pred_uid`` as the one fallback of ``succ_uid`` (one level:
        the predecessor's own fallback, if any, becomes an ordinary unit)."""
        self._bump()
        old = self._fallback.pop(pred_uid, None)
        if old is not None:
            self._fallback_of.pop(old, None)
        prev = self._fallback.pop(succ_uid, None)
        if prev is not None:
            self._fallback_of.pop(prev, None)
        self._fallback_of[pred_uid] = succ_uid
        self._fallback[succ_uid] = pred_uid
        self.counters.fallback_kept += 1

    def fallback_of(self, unit_id: UnitId) -> UnitId | None:
        """The successor ``unit_id`` is the kept fallback of, or None."""
        return self._fallback_of.get(unit_id)

    def fallback(self, unit_id: UnitId) -> UnitId | None:
        """The kept fallback predecessor of ``unit_id``, or None."""
        return self._fallback.get(unit_id)

    # -- host use and eviction --------------------------------------------------

    def touch(self, unit_id: UnitId, step: int, *, host_use: bool = False) -> None:
        """Record a use in engine steps: the LRU position of host eviction.
        ``host_use`` (a host hit / restore) also tells the admission policy
        the unit earned its place (probation -> protected, flag); a GPU hit
        only moves LRU. Callers touch only the requester's own units, once
        per admission (the planner's scoping), never every owner of a shared page."""
        if unit_id in self.units:
            self._last_use[unit_id] = max(step, self._last_use.get(unit_id, step))
            self._bump()
            if host_use:
                rows = sum(1 for _ in self.host_members(self.units[unit_id]))
                self.policy.on_host_use(unit_id, rows)

    def fa_pages(self, unit_id: UnitId) -> int:
        """FA-page depth of the boundary, using the finest FA-class group.
        Admission (including the tiny predicate) belongs to the walker."""
        unit = self.units[unit_id]
        size = min(g.block_size for g in self._full)
        return (unit.boundary + size - 1) // size

    def _idle_ownerless(self, block: KVCacheBlock) -> bool:
        """A host row the next allocation may take without destroying a
        unit: free (in the queue), owned by no unit through any key, and in
        no host eviction group (a popped group member breaks its group)."""
        return (
            not block.is_null and block.ref_cnt == 0
            and block.prev_free_block is not None
            and block.block_id not in self.host_pool.evict_groups._block_group
            and not self._block_owners(block)
        )

    def idle_rows(
        self, limit: int, skip: Callable[[KVCacheBlock], bool] | None = None
    ) -> list[KVCacheBlock]:
        """Up to ``limit`` idle ownerless host rows, in free-queue order
        (passing over the rows ``skip`` names)."""
        out: list[KVCacheBlock] = []
        if limit <= 0:
            return out
        for block in self.host_pool.free_block_queue.iter_blocks_after(None):
            if self._idle_ownerless(block) and not (skip and skip(block)):
                out.append(block)
                if len(out) >= limit:
                    break
        return out

    def can_reserve(self, need_rows: int, step: int) -> bool:
        """Whether ``need_rows`` idle ownerless rows exist now or a plan
        can make them (honours ``lone_victims``). Evicts nothing and moves
        no counter (a gate poll is not a plan); like every plan
        it first runs the deferred un-key work (``flush``)."""
        if need_rows <= 0:
            return True
        return self.plan(
            need_rows, step, displace_published=True, count=False
        ).satisfied

    def front_rows(
        self, n: int, skip: Callable[[KVCacheBlock], bool] | None = None
    ) -> list[KVCacheBlock]:
        """Put the first ``n`` idle ownerless rows (``skip`` as in
        ``idle_rows``) at the head of the host free queue, in order, and
        return them: the very rows the walker's decision reserved are the
        next ``n`` pops. The caller allocates right after,
        with no intervening step."""
        rows = self.idle_rows(n, skip)
        queue = self.host_pool.free_block_queue
        for block in rows:
            queue.remove(block)
        queue.prepend_n(rows)
        return rows

    def lone_victims(self, on: bool) -> bool:
        """Set the unit-less-copy victim filter of ``plan`` (rule
        A): while on, a published unit is never a victim, not even a
        running-twin duplicate, so an intermediate checkpoint takes only idle
        ownerless rows (free-queue LRU order) or unpublished units by LRU.
        The trade-off: an intermediate boundary may recompute on a host full
        of complete units; output is identical. Returns the previous
        setting, for restoring."""
        prev, self._lone_mode = self._lone_mode, bool(on)
        return prev

    def protect(self, row_ids) -> frozenset[int]:
        """Set the host rows ``plan`` never frees (a unit is skipped when any
        of its EXCLUSIVE rows is protected; a protected shared row protects
        nobody else); returns the previous set, for roll-back."""
        prev, self._protected_rows = self._protected_rows, frozenset(row_ids)
        return prev

    def abandon(self, unit_id: UnitId) -> None:
        """The walker gave up this unit's start: release its landed rows
        (an abandoned staged fragment, an ordinary LRU victim) and hold none
        for it until ``resume``. A row it shares with another unpublished,
        non-abandoned owner is handed over to that owner, else freed."""
        self._bump()
        self._abandoned.add(unit_id)
        self._release_held(unit_id)

    def resume(self, unit_id: UnitId) -> None:
        self._bump()
        self._abandoned.discard(unit_id)

    def is_abandoned(self, unit_id: UnitId) -> bool:
        return unit_id in self._abandoned

    def is_published(self, unit_id: UnitId) -> bool:
        return unit_id in self._published

    def assembling_uids(self) -> list[UnitId]:
        """Units holding landed rows while they assemble."""
        return list(self._held)

    def host_evict_order(self, need_rows: int, step: int = 0) -> list[UnitId]:
        """The victims ``plan`` picks to make ``need_rows`` rows available
        (``displace_published=True``, honouring ``lone_victims``), or [] if
        idle rows suffice or the need cannot be met. Selection only: counts
        nothing."""
        plan = self.plan(need_rows, step, displace_published=True, count=False)
        return list(plan.victims) if plan.satisfied else []

    # -- eviction: plan / apply (iface ``EvictionPolicy``) ------------------

    def _bump(self) -> None:
        """Units, keys or host ownership changed: the candidate index is
        stale (rebuilt on the next plan, never per call)."""
        self._dir_version += 1

    def _plain_keys(self, unit: RestoreUnit) -> frozenset:
        return frozenset(get_block_hash(k) for k in unit.required)

    def _private_keys(self, unit: RestoreUnit) -> frozenset:
        """Required keys only this unit owns: the ghost list never learns a
        shared (system-prompt) key, so an unrelated session sharing it is
        not taken for a returning one."""
        return frozenset(
            get_block_hash(k) for k in unit.required
            if self._key_units.get(k, set()) <= {unit.uid}
        )

    def _twin_kind(self, host_block: KVCacheBlock) -> str:
        """Classify a duplicate host row: "running" when every GPU twin is held
        by a running request, else "imminent" (it leaves the GPU next)."""
        kind_of = self.twin_kind
        for key in self._keys_of(self.host_pool, host_block):
            twin = self._gpu_block(key)
            if twin is None:
                return "imminent"
            kind = kind_of(twin) if kind_of is not None else (
                "running" if twin.ref_cnt > 0 else "imminent"
            )
            if kind != "running":
                return "imminent"
        return "running"

    def _gpu_twins(self, host_rows: Iterable[KVCacheBlock]) -> list[KVCacheBlock]:
        """The GPU blocks holding a key of ``host_rows`` (deduplicated)."""
        out: dict[int, KVCacheBlock] = {}
        for row in host_rows:
            for key in self._keys_of(self.host_pool, row):
                twin = self._gpu_block(key)
                if twin is not None:
                    out[twin.block_id] = twin
        return list(out.values())

    def _victim_class(self, exclusive: Sequence[KVCacheBlock]) -> str:
        if not exclusive:
            return CLASS_LRU
        running = True
        for block in exclusive:
            if not self.is_duplicate(block):
                return CLASS_LRU
            if running and self._twin_kind(block) != "running":
                running = False
        return CLASS_RUNNING_DUP if running else CLASS_IMMINENT_DUP

    def _candidates(self, step: int) -> list[_Candidate]:
        """Every non-assembling unit with host rows, in victim order: class
        (running-twin duplicates, LRU, imminent twins), then the admission
        segment (probation before protected; flag off: one segment), then
        last use. Cached per (directory version, step): selection re-reads
        only the live pin/queue state of the rows it considers."""
        key = (self._dir_version, step)
        cached = self._cand_cache
        if cached is not None and cached[0] == key:
            return cached[1]
        out: list[_Candidate] = []
        for uid, unit in self.units.items():
            if uid in self._held:
                continue  # assembling: its landed rows are held (ref > 0)
            rows = tuple(self.host_members(unit))
            if not rows:
                continue
            owners = tuple(frozenset(self._block_owners(b)) for b in rows)
            exclusive = [b for b, o in zip(rows, owners) if o <= {uid}]
            out.append(_Candidate(
                uid, rows, owners, self._victim_class(exclusive),
                self._last_use.get(uid, 0), uid in self._published,
                uid in self._abandoned, self.policy.rank(uid),
            ))
        if self._fallback_of:
            out.sort(key=self._fallback_order)
        else:
            out.sort(key=lambda c: (
                _CLASS_RANK[c.cls], c.rank if c.cls == CLASS_LRU else 0,
                c.last_use, c.uid,
            ))
        self._cand_cache = (key, out)
        return out

    def _fallback_order(self, c: _Candidate) -> tuple:
        """Victim order with kept fallbacks: a fallback takes its
        successor's admission rank and recency and sorts right behind it
        within its own class, so the successor's exclusive rows go first
        (the shrink) and the fallback only after it. Every other unit keeps
        the plain order (class, rank, last use, uid)."""
        anchor, rank, last, behind = c.uid, c.rank, c.last_use, 0
        succ = self._fallback_of.get(c.uid)
        if succ is not None and succ in self.units:
            anchor, behind = succ, 1
            rank = self.policy.rank(succ)
            last = max(last, self._last_use.get(succ, 0))
        return (
            _CLASS_RANK[c.cls], rank if c.cls == CLASS_LRU else 0,
            last, anchor, behind,
        )

    def _row_free_now(self, block: KVCacheBlock) -> bool:
        return (
            block.ref_cnt == 0 and block.prev_free_block is not None
            and block.block_id not in self._protected_rows
        )

    def plan(
        self,
        need_rows: int,
        step: int = 0,
        *,
        displace_published: bool = True,
        for_units: Iterable[UnitId] = (),
        count: bool = True,
    ) -> EvictionPlan:
        """Selection of host victims for ``need_rows`` rows: it evicts
        nothing (``apply`` does), but first runs the deferred un-key work
        (``flush``) and, unless ``count`` is False (a probe such as
        ``can_reserve``), moves the plan counters.

        Order: idle ownerless rows (counted, freed by nobody) -> units whose
        exclusive rows all duplicate GPU blocks held by running requests ->
        units by LRU of host use (flag: probation before protected) -> units
        whose twins are about to leave the GPU. A victim frees only its
        EXCLUSIVE rows; a row shared with a unit that survives the plan is
        never listed. A unit is skipped when one of its exclusive rows is
        pinned, out of the free queue or protected: no lock spreads through
        a shared row.

        ``displace_published=False`` (a lane that may not displace published
        host content): idle rows and running-twin duplicates only.
        ``lone_victims`` (rule A) excludes every published unit.
        ``for_units``: the units this room is for; their kept predecessors
        (supersede-at-publish) are tried first, so a constrained successor
        replaces its own predecessor instead of waiting; with the admission
        flag on they also decide whether protected units may be displaced.
        The plan may be short (``satisfied`` False); ``apply`` it only when
        it is satisfied.
        """
        self.flush()
        # A probe counts into a scratch copy.
        c = self.counters if count else UnitCounters()
        c.evict_plans += 1
        for_units = tuple(for_units)
        idle = len(self.idle_rows(need_rows)) if need_rows > 0 else 0
        classes = {"idle": idle, CLASS_RUNNING_DUP: 0, CLASS_LRU: 0,
                   CLASS_IMMINENT_DUP: 0}
        victims: list[int] = []
        rows: list[int] = []
        plan_args = dict(
            need_rows=need_rows, displace_published=displace_published,
            idle=idle, version=self._dir_version, step=step,
            for_units=for_units,
        )
        if need_rows <= 0 or idle >= need_rows:
            return EvictionPlan(victims=victims, rows=rows, classes=classes, **plan_args)
        preferred: set[int] = set()
        for uid in for_units:
            preferred |= self._pending_preds.get(uid, set())
        may_protected = self.policy.admits_protected(
            [self._plain_keys(self.units[u]) for u in for_units if u in self.units],
            [u for u in preferred],
        )
        cands = self._candidates(step)
        if preferred:
            cands = [x for x in cands if x.uid in preferred] + [
                x for x in cands if x.uid not in preferred
            ]
        # Victims released so far: each gives up its host ownership (a
        # duplicate victim stays recorded, GPU-backed, but owns no host row
        # any more), so a shared row frees with the last of its owners
        # (shared roots of all-duplicate forks).
        gone: set[int] = set()
        freed: set[int] = set()
        remaining = list(cands)
        bypassed = False
        while idle + len(freed) < need_rows:
            progress = False
            retry: list[_Candidate] = []
            for cand in remaining:
                if idle + len(freed) >= need_rows:
                    break
                # A constrained successor with displacement rights may replace
                # its own kept predecessor (budgeted replacement): never a
                # deadlock. A lane without them (``displace_published=False``,
                # e.g. a deep-GRACE offer) gets no bypass.
                own = displace_published and cand.uid in preferred
                if self._lone_mode and cand.published and not own:
                    continue  # Rule A: complete units outrank fragments
                take = [
                    b for b, o in zip(cand.rows, cand.owners)
                    if b.block_id not in freed and (o - gone) <= {cand.uid}
                ]
                if not take:
                    # A possible zero-gain co-owner: the same permissions,
                    # judged on all its rows (it gives every one of them up).
                    if not displace_published and (
                        self._victim_class(cand.rows) != CLASS_RUNNING_DUP
                    ):
                        continue
                    # The admission filter as on the take path below: only the
                    # LRU class stays out of a protected unit.
                    if (
                        self._victim_class(cand.rows) == CLASS_LRU
                        and cand.rank > 0 and not own and not may_protected
                    ):
                        bypassed = True
                        continue
                    retry.append(cand)
                    continue
                if not all(self._row_free_now(b) for b in take):
                    c.evict_skipped_leased += 1
                    continue
                # Classify what is actually taken, live (rows a released
                # co-owner left; a GPU ref or grace change since the index was
                # cached), BEFORE the lane's permission and the admission
                # filter, exactly as ``apply``.
                cls = self._victim_class(take)
                if not displace_published and cls != CLASS_RUNNING_DUP:
                    continue
                if (
                    cls == CLASS_LRU and cand.rank > 0
                    and not own and not may_protected
                ):
                    # Admission flag: a probation copy never displaces
                    # protected content.
                    bypassed = True
                    continue
                victims.append(cand.uid)
                ids = [b.block_id for b in take]
                rows.extend(ids)
                freed.update(ids)
                classes[cls] += len(ids)
                gone.add(cand.uid)
                progress = True
            if idle + len(freed) >= need_rows:
                break
            if not progress:
                # Zero-exclusive owners (every row shared,
                # e.g. aliases of one physical row, or a shared prompt whose
                # private tails are gone): a shared row frees with its LAST
                # owner. ``retry`` holds the eligible candidates (filters
                # passed) with nothing exclusive left. Take the first of them
                # (victim order) with a free row whose other live owners are
                # ALL in ``retry``, fewest first: those co-owners leave the
                # table first (zero-gain victims), so the next pass frees the
                # row. A row with an owner outside ``retry`` (assembling,
                # filtered, pinned elsewhere) is never freed (alias-safe).
                ready = {x.uid for x in retry}
                blockers: set[int] | None = None
                for x in retry:
                    best: set[int] | None = None
                    for b, o in zip(x.rows, x.owners):
                        if b.block_id in freed or not self._row_free_now(b):
                            continue
                        others = set(o - gone) - {x.uid}
                        if others and others <= ready and (
                            best is None or len(others) < len(best)
                        ):
                            best = others
                    if best is not None:
                        blockers = best
                        break
                if not blockers:
                    break
                for x in [x for x in retry if x.uid in blockers]:
                    victims.append(x.uid)
                    gone.add(x.uid)
                    retry.remove(x)
            remaining = retry
        plan = EvictionPlan(victims=victims, rows=rows, classes=classes, **plan_args)
        if not plan.satisfied:
            c.evict_plans_short += 1
            if bypassed and count:
                self.policy.counters.probation_bypassed += 1
        return plan

    def apply(self, plan: EvictionPlan) -> int:
        """Evict ``plan``'s victims in order; returns the rows made idle
        (front-queued). Each victim loses exactly its exclusive rows at its
        turn; a unit whose exclusive rows were all GPU duplicates stays
        recorded (GPU-backed), a unit with a host-only row is lost (cause
        ``host_evicted``). Refuses (and counts ``evict_apply_mismatch``)
        whatever no longer matches the selection, so the freed rows are
        exactly the planned ones or a subset of them."""
        if plan.empty:
            return 0
        self.flush()
        c = self.counters
        planned = set(plan.rows)
        succs = set(plan.for_units)
        freed_ids: list[int] = []
        applied: set[int] = set()  # victims done (released host ownership)
        outer, self._loss_cause = self._loss_cause, "host_evicted"
        try:
            for uid in plan.victims:
                unit = self.units.get(uid)
                if unit is None:
                    c.evict_apply_mismatch += 1
                    continue
                members = self.host_members(unit)
                take = [
                    b for b in members
                    if self._block_owners(b) - applied <= {uid}
                ]
                if any(
                    b.block_id not in planned or not self._row_free_now(b)
                    for b in take
                ):
                    c.evict_apply_mismatch += 1
                    break
                own = (
                    plan.displace_published
                    and self._superseded_by.get(uid) in succs
                )
                cls = self._victim_class(take) if take else CLASS_LRU
                if take and (
                    not plan.displace_published and cls != CLASS_RUNNING_DUP
                ):
                    # The twin left the running set since ``plan``: this
                    # lane may not take it.
                    c.evict_apply_mismatch += 1
                    break
                applied.add(uid)
                c.evict_shared_kept += len(members) - len(take)
                if own:
                    c.constrained_successor += 1
                if not take:
                    # A zero-gain co-owner (``plan``): its shared rows free
                    # with their last owner, later in this plan.
                    if members and all(self.is_duplicate(b) for b in members):
                        # GPU-backed: stays recorded, unpublished.
                        self.policy.on_drop(uid)
                        self._published.discard(uid)
                        if unit.host_group:
                            self._ungroup_host(unit)
                        self._bump()
                        if uid in self._superseded_by:
                            self._complete_supersede(uid)
                        continue
                    self.policy.on_evict(uid, self._private_keys(unit))
                    c.units_lost += 1
                    self._record_loss(unit)
                    self._drop_unit(unit)
                    continue
                if cls == CLASS_RUNNING_DUP:
                    c.evict_running_dup_rows += len(take)
                    self.policy.on_drop(uid)  # re-enters on re-publication
                elif cls == CLASS_IMMINENT_DUP:
                    c.evict_imminent_dup_rows += len(take)
                    self.policy.on_drop(uid)
                else:
                    self.policy.on_evict(uid, self._private_keys(unit))
                self._published.discard(uid)
                if unit.host_group:
                    self._ungroup_host(unit)
                self._bump()
                ids = [b.block_id for b in take]
                twins = self._gpu_twins(take) if self.revoke_hook else []
                n = self._drop_host_remains(take)
                if twins:
                    self.revoke_hook(twins, plan.step)
                freed_ids.extend(ids[:n])
                if uid in self._superseded_by and uid in self.units:
                    # A kept predecessor lost its host copy before its
                    # successor published. Host-only content gone (LRU): the
                    # restore fallback is lost, so it is a loss,
                    # never a silent supersede. GPU-backed (a duplicate): the
                    # GPU still holds it; the supersede completes now.
                    if cls == CLASS_LRU:
                        c.units_lost += 1
                        self._record_loss(unit)
                        self._drop_unit(unit)
                    else:
                        self._complete_supersede(uid)
                elif cls == CLASS_LRU and uid in self.units:
                    # Its host-only rows were optional keys: it still leaves
                    # the table, as ``plan`` assumed (its shared rows become
                    # its co-owners' own).
                    c.units_lost += 1
                    self._record_loss(unit)
                    self._drop_unit(unit)
                c.evict_units += 1
        finally:
            self._loss_cause = outer
        if set(freed_ids) != planned:
            c.evict_apply_mismatch += int(bool(set(freed_ids) - planned))
        c.evict_exclusive_rows += len(freed_ids)
        return len(freed_ids)

    def is_duplicate(self, host_block: KVCacheBlock) -> bool:
        """Every key of ``host_block`` is also held by the GPU as a usable
        replacement: the GPU block is ready and is the very content this host
        copy was read from (generation-qualified). Pending or foreign
        same-key GPU content makes the host copy the only restore point, so
        displacing it would lose it."""
        keys = self._keys_of(self.host_pool, host_block)
        if not keys:
            return False
        tier = self.gpu_pool.tier
        for k in keys:
            g = self._gpu_block(k)
            if g is None:
                return False
            if tier is None:
                continue
            if not tier.is_ready(g):
                return False
            cid = tier.content_id(g)
            if cid is not None and tier.host_src.get(host_block.block_id) != cid:
                return False
        return True

    # -- host groups ------------------------------------------------------------

    def _block_owners(self, block: KVCacheBlock) -> set[int]:
        """Units that own any key ``block`` carries (primary and aliases):
        un-keying the block destroys all of them."""
        owners: set[int] = set()
        pool = self.gpu_pool if block.pool is not self.host_pool else self.host_pool
        for key in self._keys_of(pool, block):
            owners |= self._key_units.get(key, set())
        return owners

    def _exclusive(self, unit: RestoreUnit) -> list[KVCacheBlock]:
        """The unit's host blocks no other unit depends on through any of
        their keys: exclusivity is physical, since eviction un-keys a block
        with every alias it carries."""
        return [
            b for b in self.host_members(unit)
            if self._block_owners(b) <= {unit.uid}
        ]

    def _ungroup_host(self, unit: RestoreUnit) -> None:
        groups = self.host_pool.evict_groups
        groups.leave_many(self.host_members(unit))  # one rebuild
        unit.host_group = False

    def _maybe_register_host(self, unit: RestoreUnit) -> None:
        """Register a published, complete unit's exclusive host copies."""
        if (
            unit.host_group or unit.uid not in self._published
            or not self.is_host_complete(unit)
        ):
            return
        members = self._exclusive(unit)
        if self.host_pool.evict_groups.register(members) > 0:
            unit.host_group = True
            self.counters.units_host_registered += 1

    def on_host_registered(self, cpu_blocks: Iterable[KVCacheBlock]) -> None:
        """Refresh groups after store refs drop; publication is explicit."""
        self.flush()
        self._bump()
        pending: dict[int, RestoreUnit] = {}
        for block in cpu_blocks:
            for unit in self.unit_of_block(block):
                pending[unit.uid] = unit
        for unit in pending.values():
            self._maybe_register_host(unit)

    def on_host_released(self, cpu_block_ids: Iterable[int]) -> None:
        """Load refs dropped. Duplicate status is checked at eviction.
        Restore callers must also touch each hit unit with the engine step."""
        self.flush()
        self._bump()

    # -- loss and directory listeners ------------------------------------------

    LOST_KEYS_MAX = 1 << 16

    def note_skip(self, unit: RestoreUnit, cause: str) -> None:
        """The walker did not copy ``unit`` (skipped_tiny / never_fit): if
        its GPU copy is reclaimed, that is the loss cause."""
        self._skip_cause[unit.uid] = cause

    def _cause_of(self, unit: RestoreUnit) -> tuple[str, int]:
        age = self.step - self._last_use.get(unit.uid, self.step)
        if self._loss_cause == "host_evicted":
            return "host_evicted", age
        if self._loss_cause is not None:
            return self._loss_cause, age
        return self._skip_cause.get(unit.uid, "missed_copy"), age

    def _record_loss(self, unit: RestoreUnit) -> None:
        cause = self._cause_of(unit)
        for k in self.ordered(unit):
            if k not in unit.required:
                continue
            plain = get_block_hash(k)
            self.lost[plain] = cause
            self.lost.move_to_end(plain)
        while len(self.lost) > self.LOST_KEYS_MAX:
            self.lost.popitem(last=False)
        self._skip_cause.pop(unit.uid, None)

    def loss_cause(self, block_hashes: Sequence[object]) -> tuple[str, int] | None:
        """The recorded loss cause of the deepest lost unit on this prefix."""
        for h in reversed(block_hashes):
            got = self.lost.get(h)
            if got is not None:
                return got
        return None

    def _on_key_lost(self, key: BlockHashWithGroupId) -> None:
        for uid in list(self._key_units.get(key, ())):
            unit = self.units.get(uid)
            if unit is None or key not in unit.required:
                continue
            self.counters.units_lost += 1
            self._record_loss(unit)
            remains = self._exclusive(unit)
            if self._keep_lost_remains:
                # Only the failed load's read rows lose their keys; published
                # rows stay keyed and become ownerless LRU content. An
                # assembling unit's staged rows are still un-keyed.
                remains = [b for b in remains if b.block_id not in self._visible]
            self._drop_unit(unit)
            self.counters.unit_lost_host_blocks += self._drop_host_remains(remains)

    def keep_lost_remains(self, keep: bool) -> bool:
        """Set whether a lost unit's published remains keep their keys
        (failed-load invalidation); returns the previous value."""
        prev, self._keep_lost_remains = self._keep_lost_remains, keep
        return prev

    def _drop_host_remains(self, blocks: Iterable[KVCacheBlock]) -> int:
        """Unkey lost/evicted exclusive rows and front-queue idle ones.
        A pinned row loses its directory keys, never its bytes or refs."""
        self._bump()
        pool = self.host_pool
        front: list[KVCacheBlock] = []
        seen: set[int] = set()
        for block in blocks:
            bid = block.block_id
            if block.is_null or bid in seen or bid in self._unkeying:
                continue
            seen.add(bid)
            queued = block.ref_cnt == 0 and block.prev_free_block is not None
            if queued:
                pool.free_block_queue.remove(block)
            self._visible.discard(bid)
            keys = self._keys_of(pool, block)
            pool._maybe_evict_cached_block(block)
            # Also work with a bare pool (without TierRuntime's host hook).
            for key in keys:
                if not self._present(key):
                    self._on_key_lost(key)
            if queued:
                front.append(block)
        pool.free_block_queue.prepend_n(front)
        return len(front)

    def flush(self) -> None:
        # A CoW move unhashes its source before keying its destination.
        pending, self._pending_unhashed = self._pending_unhashed, []
        for key in pending:
            if self._gpu_block(key) is None and self._host_block(key) is None:
                self._on_key_lost(key)

    def on_gpu_keyed(self, block: KVCacheBlock) -> None:
        self._bump()  # a host row may have become a duplicate
        self._reclaiming = None
        self.flush()

    def on_gpu_unhashed(
        self, block: KVCacheBlock, keys: Sequence[BlockHashWithGroupId]
    ) -> None:
        reclaimed = self._reclaiming == block.block_id
        self._bump()
        self._reclaiming = None
        self.flush()
        self._pending_unhashed.extend(keys)
        if reclaimed:
            self.flush()

    def on_gpu_reclaim(self, block: KVCacheBlock, state: BlockState) -> None:
        self.flush()
        self._reclaiming = block.block_id

    def on_gpu_reset(self) -> None:
        self._reclaiming = None
        self._pending_unhashed.clear()
        self._loss_cause = "race"
        try:
            for key in list(self._key_units):
                if self._host_block(key) is None:
                    self._on_key_lost(key)
        finally:
            self._loss_cause = None

    def on_host_unkeyed(self, block: KVCacheBlock) -> None:
        """A host block is losing its keys (still in the directory here)."""
        bid = block.block_id
        if bid in self._unkeying:
            return
        self._bump()
        # Owners are looked up only for a staged row while
        # some unit is abandoned (rare), never on a published row's eviction.
        if (
            self._abandoned and bid not in self._visible
            and not self._abandoned.isdisjoint(self._block_owners(block))
        ):
            self.counters.abandoned_fragment_evicted += 1
        self._visible.discard(bid)
        self._unkeying.add(bid)
        outer, self._loss_cause = self._loss_cause, "host_evicted"
        try:
            self.flush()
            for key in self._keys_of(self.host_pool, block):
                if self._host_holds_other(key, block):
                    continue
                if self._gpu_block(key) is not None:
                    self.counters.dup_reclaimed += 1
                else:
                    self._on_key_lost(key)
        finally:
            self._loss_cause = outer
            self._unkeying.discard(bid)

    def on_host_reset(self) -> None:
        """Host keys are gone; transaction refs belong to write-back."""
        self._bump()
        self.release_all_held()
        self._published.clear()
        self._visible.clear()
        # No published predecessor survives a host reset.
        self._superseded_by.clear()
        self._pending_preds.clear()
        self._fallback_of.clear()
        self._fallback.clear()
        self.policy.clear()
        for unit in list(self.units.values()):
            unit.host_group = False
            if any(self._gpu_block(k) is None for k in unit.required):
                self._drop_unit(unit)

    def clear(self) -> None:
        self._reclaiming = None
        self._pending_unhashed.clear()
        self.units.clear()
        self._key_order.clear()
        self._ordered_cache.clear()
        self._key_units.clear()
        self._published.clear()
        self._visible.clear()
        self._last_use.clear()
        self._unkeying.clear()
        self._superseded_by.clear()
        self._pending_preds.clear()
        self._fallback_of.clear()
        self._fallback.clear()
        self.policy.clear()
        self._cand_cache = None
        self._bump()


def group_infos(
    kv_cache_config,
    group_ids: Iterable[int],
    group_block_sizes: Sequence[int],
    eagle_group_ids: Iterable[int] = (),
    fine_group_ids: Iterable[int] | None = None,
) -> list[GroupInfo]:
    """The unit-relevant groups among ``group_ids`` (the tier's prefix-cacheable
    groups). Groups of other kinds carry no restore-unit member.
    ``eagle_group_ids``: the groups whose lookup drops one hash unit
    (``KVCacheCoordinator.eagle_group_ids``)."""
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        MambaSpec,
        SlidingWindowSpec,
    )

    eagle = frozenset(eagle_group_ids)
    fine = None if fine_group_ids is None else frozenset(fine_group_ids)

    def is_fine(gid):
        return None if fine is None else gid in fine

    out: list[GroupInfo] = []
    for gid in group_ids:
        spec = kv_cache_config.kv_cache_groups[gid].kv_cache_spec
        size = group_block_sizes[gid]
        if isinstance(spec, MambaSpec):
            out.append(GroupInfo(gid, GroupKind.RECURRENT, size))
        elif isinstance(spec, SlidingWindowSpec):
            out.append(GroupInfo(
                gid, GroupKind.SLIDING, size, spec.sliding_window,
                spec.extra_retained_tokens, gid in eagle, is_fine(gid),
            ))
        elif isinstance(spec, FullAttentionSpec):
            out.append(GroupInfo(
                gid, GroupKind.FULL, size, eagle=gid in eagle, fine=is_fine(gid)
            ))
    return out


def mount(
    gpu_pool: BlockPool,
    host_pool: BlockPool,
    kv_cache_config,
    group_ids: Sequence[int],
    group_block_sizes: Sequence[int],
    hash_block_size: int,
    eagle_group_ids: Iterable[int] = (),
    fine_group_ids: Iterable[int] | None = None,
) -> UnitTable:
    """Mount the unified tier on ``gpu_pool``: its ``BlockTierState``
    (``gpu_pool.tier``) with the unit table as listener."""
    from vllm.v1.core.kv_tier_state import BlockTierState

    assert gpu_pool.tier is None
    units = UnitTable(
        gpu_pool,
        host_pool,
        group_infos(
            kv_cache_config, group_ids, group_block_sizes, eagle_group_ids,
            fine_group_ids,
        ),
        hash_block_size,
    )
    tier = BlockTierState(gpu_pool, group_ids, host_pool=host_pool)
    tier.listener = units
    if units._sliding:
        # Only a window's sliding-window pages are restore dependencies.
        tier.retains_fn = units.retains
    gpu_pool.tier = tier
    return units


__all__ = [
    "EvictionPlan",
    "GroupInfo",
    "GroupKind",
    "RestoreUnit",
    "UnitCounters",
    "UnitTable",
    "UnitId",
    "group_infos",
    "mount",
    "unit_key",
]
