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

import bisect
import heapq
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
    # Of ``units_lost``, the evicted covered
    # kept predecessors (``covered_early``): the session restores from its
    # published successor, only the predecessor's own boundary is gone.
    units_lost_covered_early: int = 0
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


@dataclass
class PlanCostCounters:
    """Planner work, observational only: never read by a decision,
    moved by probes too (``can_reserve``, ``host_evict_order``), kept apart
    from ``UnitCounters`` so a probe still moves no plan counter.
    Exported by ``TierRuntime.unit_counters`` as ``plancost_<name>``."""

    plans: int = 0  # ``plan`` calls, probes included
    plans_walked: int = 0  # ... that walked the candidates (idle short)
    class_passes: int = 0  # candidate class passes (new directory version or step)
    resorts: int = 0  # ... that re-sorted (a class or the structure changed)
    struct_rebuilds: int = 0  # structure rebuilds (a structural bump)
    idle_counts: int = 0  # ``idle_count`` calls
    unowned_rebuilds: int = 0  # unit-less row list rebuilds
    struct_bumps: int = 0  # ``_bump`` (structure and directory versions)
    gpu_bumps: int = 0  # ``_bump_gpu`` (directory version only)
    idle_lists: int = 0  # idle ownerless row lists rebuilt (queue moved)
    idle_order_walks: int = 0  # ... that walked the queue for their order
    idle_rows_cached: int = 0  # ``idle_rows`` answered from the list
    lone_views: int = 0  # lone-mode candidate views built
    witness_classes: int = 0  # class reads settled by an absent-key witness
    skip_indexes: int = 0  # first-pass skip indexes built (list/directory)
    skip_syncs: int = 0  # ... leased positions re-synced (a flag moved)
    leased_rebuilds: int = 0  # leased flags re-read whole (structure, lost log)
    leased_rereads: int = 0  # units re-read (their rows moved or were protected)
    struct_updates: int = 0  # structure versions met by an incremental update
    struct_units: int = 0  # ... units re-derived by them
    # ``_bump`` by call site (``struct_bumps`` is their sum):
    bump_record: int = 0
    bump_hold: int = 0
    bump_release_held: int = 0
    bump_hand_over: int = 0
    bump_release_visible: int = 0
    bump_drop_unit: int = 0
    bump_publish: int = 0
    bump_fallback: int = 0
    bump_touch: int = 0
    bump_abandon: int = 0
    bump_resume: int = 0
    bump_apply: int = 0
    bump_host_registered: int = 0
    bump_host_released: int = 0
    bump_host_remains: int = 0
    bump_host_unkeyed: int = 0
    bump_host_rekeyed: int = 0
    bump_host_reset: int = 0
    bump_clear: int = 0
    cands_visited: int = 0  # candidates a plan evaluated one by one
    cands_skipped: int = 0  # ... and settled in runs by the skip index


# Victim classes, in eviction order (iface ``EvictionPolicy``).
CLASS_RUNNING_DUP = "running_dup"
CLASS_LRU = "lru"
CLASS_IMMINENT_DUP = "imminent_dup"
_CLASS_RANK = {CLASS_RUNNING_DUP: 0, CLASS_LRU: 1, CLASS_IMMINENT_DUP: 2}


def _victim_key(c: "_Candidate") -> tuple:
    """Plain victim order: class, then inside the LRU class the admission
    segment (probation before protected), then last use, uid.

    No covered-first term: spending a covered kept predecessor early,
    instead of letting the turn after retire it for free (supersede),
    doubles host evictions and reactive re-copies. ``covered`` stays
    for the diag (``live_over_covered``, info) and the loss counter."""
    lru = c.cls == CLASS_LRU
    return (
        _CLASS_RANK[c.cls], c.rank if lru else 0, c.last_use, c.uid,
    )


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
    # MERLIN_KV_TIER_DIAG only (else 0). 1:
    # the plan took a live LRU unit (not a preferred predecessor) although
    # the same planner, on the same snapshot, without live units and
    # imminent twins (running-twin duplicates, covered units and preferred
    # predecessors only), meets the same need. 0 by design.
    live_over_covered: int = 0

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
    # The rows with no other owner (``owners`` <= {uid}), in ``rows`` order,
    # and each one's host keys (``_keys_of``): structure, cached with it.
    exclusive: tuple[KVCacheBlock, ...] = ()
    excl_keys: tuple[tuple[BlockHashWithGroupId, ...], ...] = ()
    # Every other unit owning one of ``rows``.
    others: frozenset[int] = frozenset()
    # An exclusive row is no duplicate whatever its twins: it has
    # no host key, or a key the GPU directory lacks. Set by the class pass,
    # valid for its directory version (a GPU key appears only through
    # ``on_gpu_keyed``, which moves it): the class is LRU, no twin is read.
    witness: bool = False
    # The same for a unit with no exclusive row, over all its rows (the
    # zero-gain path classifies ``rows``): ``_victim_class(rows)`` is LRU.
    rows_witness: bool = False
    witness_dv: int = -1  # the directory version both were read at
    # A published later unit extends this one
    # (``UnitTable._is_covered``); re-read per structure version.
    covered: bool = False


# Queue moves logged between two
# leased-flag reads before a full re-read is cheaper.
_QUEUE_LOG_CAP = 4096


class _SkipIndex:
    """What a plan's first pass does with each candidate of one victim-order
    list (``UnitTable._skip_index``). Its categories hold for one directory
    version:

    * no exclusive row: evaluated (the zero-gain path), unless its rows
      witness settles that path's class LRU: then as an LRU witness below;
    * an absent-key witness (LRU): skipped by a lane that may not displace
      published content; by one that may, skipped with ``bypassed`` when
      protected (rank > 0) and protected units are not admitted, else
      evaluated (taken);
    * any other: evaluated (its class is read live).

    On top, the leased flags (``_leased_flags``, synced per flag version):
    a skipped unit with an exclusive row not free now counts
    ``evict_skipped_leased`` instead of a bypass; an evaluated one is
    judged by the loop body itself. ``leased_pos`` and ``bypass_pos`` are
    sorted positions, counted per run by bisection.

    A leased skip holds whatever was taken before it: a unit with an
    exclusive row takes at least its exclusive rows (owned by it alone,
    never freed by another victim), so the row that is not free stays in its
    take. A witness skip does not: once a co-owner is taken, the unit's take
    gains the rows they shared (a unit with no exclusive row gets a take at
    all), and one of them may not be free, so ``_first_pass`` evaluates
    every later co-owner of a unit it takes."""

    __slots__ = ("n", "pos", "zero", "live", "lru", "_must", "_excl", "_by",
                 "leased_pos", "bypass_pos", "synced")

    def __init__(self, cands):
        self.n = len(cands)
        self.pos = {c.uid: i for i, c in enumerate(cands)}
        self.zero: list[int] = []
        self.live: list[int] = []
        self.lru: list[tuple[int, int]] = []
        self._must: dict[tuple[bool, bool], list[int]] = {}
        # (position, uid) of every unit with an exclusive row; (position,
        # uid or None) of every protected LRU witness.
        self._excl: list[tuple[int, int]] = []
        self._by: list[tuple[int, int | None]] = []
        for i, c in enumerate(cands):
            if c.exclusive:
                self._excl.append((i, c.uid))
                if c.witness:
                    self.lru.append((i, c.rank))
                    if c.rank > 0:
                        self._by.append((i, c.uid))
                else:
                    self.live.append(i)
            elif c.rows_witness:
                self.lru.append((i, c.rank))
                if c.rank > 0:
                    self._by.append((i, None))  # no exclusive row: never leased
            else:
                self.zero.append(i)
        self.leased_pos: list[int] = []
        self.bypass_pos: list[int] = []
        self.synced = -1

    def sync(self, flags: dict[int, bool], version: int) -> bool:
        if self.synced == version:
            return False
        self.leased_pos = [i for i, uid in self._excl if flags[uid]]
        self.bypass_pos = [
            i for i, uid in self._by if uid is None or not flags[uid]
        ]
        self.synced = version
        return True

    def must(self, displace: bool, may_protected: bool) -> list[int]:
        mode = (displace, displace and may_protected)
        got = self._must.get(mode)
        if got is None:
            got = self.zero + self.live
            if displace:
                got += [i for i, rank in self.lru if may_protected or rank == 0]
            got.sort()
            self._must[mode] = got
        return got


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
        # MERLIN_KV_TIER_DIAG=1: one victim line per eviction plan that takes
        # a host-only (LRU) unit, with the host row split. Debug only.
        try:
            self._diag = bool(read_registered("MERLIN_KV_TIER_DIAG"))
        except Exception:
            self._diag = False
        self.groups = list(groups)
        self.hash_block_size = hash_block_size
        self._recurrent = [g for g in self.groups if g.kind is GroupKind.RECURRENT]
        self._full = [g for g in self.groups if g.kind is GroupKind.FULL]
        assert self._full, "a restore unit needs a full-attention group"
        self._kind = {g.group_id: g.kind for g in self.groups}
        self._sliding = [g for g in self.groups if g.kind is GroupKind.SLIDING]
        # Host victims are chosen by class (idle -> running-twin
        # duplicates -> LRU of host use ->
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
        # predecessor is dropped when its successor publishes.
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
        # The index is split by what each part reads. The structure
        # (rows, owners, exclusive rows and their keys, last use, publication,
        # abandonment, admission rank) reads only this table and the HOST
        # directory: cached on ``_struct_version``, which every ``_bump`` moves
        # except the GPU-directory hooks (they change duplicate status only).
        # The class pass (GPU twins, read live) and the sort stay keyed on
        # (directory version, step), so a full host tier does not rebuild the
        # structure every engine step.
        self._struct_version = 0
        self._struct_cache: tuple[int, list[_Candidate]] | None = None
        # What changed since the
        # structure was last derived (``_bump``'s marks), and the indexes an
        # incremental update reads: uid -> candidate, host row -> the units
        # whose candidate lists it, host row -> its keys at derivation.
        self._dirty_uids: set[int] = set()
        self._dirty_keys: set = set()
        self._dirty_rows: set[int] = set()
        self._dirty_full = True
        self._cand_of: dict[int, _Candidate] = {}
        self._row_users: dict[int, set[int]] = {}
        self._row_keys: dict[int, tuple] = {}
        self._unowned_rows: set[int] | None = None
        self._synced_sv = -1
        # Tests: check every incremental structure against a full one.
        self.verify_structure = False
        self._cand_cache: (
            tuple[tuple[int, int], list[_Candidate], list[_Candidate]] | None
        ) = None
        # Host row id -> owned by a unit through any key; entries dropped for
        # the rows a structural change can touch (``_sync_structure``).
        self._owned_memo: tuple[int, dict[int, bool]] = (-1, {})
        self._unowned_cache: tuple[int, list[KVCacheBlock]] | None = None
        # ``_queue_epoch`` moves before every mutation of the host free queue
        # and of the host eviction groups (``_watch_host_queue``): with the
        # structure version it fixes ``_idle_ownerless`` for every row. Per
        # (structure version, queue epoch): the idle ownerless rows, and
        # their free-queue order once asked for (``_idle_list``).
        self._queue_epoch = 0
        # The host blocks the free
        # queue moved since the leased flags last read them (lost past a cap:
        # the flags are then rebuilt), and the flags: per structure version
        # and protection set, uid -> an exclusive row is not free now.
        self._queue_moved: list[KVCacheBlock] = []
        self._queue_log_lost = False
        self._leased: list | None = None
        self._leased_version = 0
        self._idle_cache: tuple[tuple[int, int], list[KVCacheBlock], bool] | None = (
            None
        )
        # Host row id -> (primary key, alias set, alias count, keys): the
        # ``_keys_of`` memo, validated against the row on every read.
        self._host_keys: dict[int, tuple] = {}
        # The lone-mode view of the current candidate list (``_lone_view``).
        self._lone_cache: tuple | None = None
        self._skip_cache: tuple | None = None  # the first-pass index (``_skip_index``)
        self._co_owner_cache: tuple | None = None
        self._watch_host_queue()
        # Supersede-at-publish: a published predecessor stays restorable until
        # its successor publishes (pred uid -> successor uid, and back).
        self._superseded_by: dict[int, int] = {}
        self._pending_preds: dict[int, set[int]] = {}
        # A kept immediate predecessor
        # (record_finished) and the later units proven at record to
        # extend it on their own hash chain (pred uid -> successor uids, and
        # back). Once one of them is published the predecessor is covered:
        # its session restores from the successor. The victim order has no
        # covered term (``_victim_key``); ``covered`` feeds the diag
        # (``live_over_covered``) and the loss counters.
        self._covered_by: dict[int, set[int]] = {}
        self._covers: dict[int, set[int]] = {}
        # Diag only (``plan``'s counterfactual, ``_cf_excluded``): exclude
        # uncovered live LRU units and imminent twins, except the successor's
        # preferred predecessors, to test whether the covered units alone
        # could serve the plan.
        self._cf_only = False
        # The host eviction group a
        # unit registered (group id, member row ids). ``_ungroup_host`` leaves
        # exactly these too: a member re-keyed away from the unit's key is no
        # longer found by key (``host_members``) but was still grouped.
        self._group_of: dict[int, tuple[int, list[int]]] = {}
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
        self.plan_cost = PlanCostCounters()
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

    def _keys_of(
        self, pool: BlockPool, block: KVCacheBlock
    ) -> Sequence[BlockHashWithGroupId]:
        """A block's keys: its primary key, then its aliases in ``key_order``.
        A host row's keys are memoised and
        re-validated on every read against the row itself (the primary key
        object, the alias set object and its size). Every host key change
        replaces the primary key (``set_block_hash`` / ``reset_hash``),
        adds to the alias set (``_insert_block_hash``) or replaces or drops
        the set (``_remove_cached_block_hashes``, ``supersede_host_copies``,
        ``reset_prefix_cache``), so a memo hit is the same key list."""
        if pool is not self.host_pool:
            return self._keys_raw(pool, block)
        aliases = pool.cached_block_hashes_by_block.get(block.block_id)
        size = -1 if aliases is None else len(aliases)
        got = self._host_keys.get(block.block_id)
        if (
            got is not None and got[0] is block.block_hash
            and got[1] is aliases and got[2] == size
        ):
            return got[3]
        keys = tuple(self._keys_raw(pool, block))
        self._host_keys[block.block_id] = (block.block_hash, aliases, size, keys)
        return keys

    @staticmethod
    def _keys_raw(pool: BlockPool, block: KVCacheBlock) -> list[BlockHashWithGroupId]:
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
        keys = self.ordered(unit)  # A total order, never set order
        self._bump("record", uids=(unit.uid,), keys=keys)
        # Supersede older units this one contains (the session's next turn).
        older: set[int] = set()
        for key in keys:
            older.update(self._key_units.get(key, ()))
        # A turn's unit keys the full-attention page holding its boundary by
        # a covering key inside that page; the next turn keys the same page
        # whole, under another key. So an older unit is also a prefix of this
        # turn when each of its non-endpoint keys this unit lacks is a
        # full-attention key on this request's own hash chain below this
        # boundary. Sliding-window keys still need exact containment.
        # The deepest such prefix (the immediate predecessor) is kept, so a
        # retry, regenerate or sibling of the last turn still restores at its
        # boundary; it is superseded by the turn after.
        chain: set | None = None  # built on the first chain candidate
        superseded: list[RestoreUnit] = []
        on_chain: list[RestoreUnit] = []
        inherited_use: int | None = None
        inherited_publication = False
        # Landed rows the superseded units held: the successor owns them and
        # is unpublished, so their refs pass to it (else they leak).
        inherited_rows: list[KVCacheBlock] = []
        # Supersede-at-publish: a PUBLISHED predecessor stays a complete,
        # restorable unit until this successor publishes (a lost successor
        # then falls back to it: one turn of recompute, output identical).
        # An unpublished one is superseded now (nothing restores
        # from it; its landed rows pass to the successor).
        keep_preds: set[int] = set()
        kept: RestoreUnit | None = None
        for uid in sorted(older):
            old = self.units[uid]
            rest = old.keys - old.endpoints
            if rest <= unit.keys:
                superseded.append(old)
                continue
            if old.boundary >= unit.boundary:
                continue
            if chain is None:
                chain = set(request.block_hashes[
                    : found.boundary // max(1, self.hash_block_size)])
            if self._prefix_on_chain(old, unit, chain):
                on_chain.append(old)
        if on_chain:
            # Kept unless an exactly contained unit lies between it and this
            # boundary. One at this boundary (a retry or identical regenerate
            # replacing it) does not count: the predecessor below stays.
            pred = max(on_chain, key=lambda u: (u.boundary, u.uid))
            if all((u.boundary, u.uid) < (pred.boundary, pred.uid)
                   for u in superseded if u.boundary < unit.boundary):
                on_chain.remove(pred)
                kept = pred
            superseded = sorted(superseded + on_chain, key=lambda u: u.uid)
        for old in superseded:
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
        # The kept predecessor is on this
        # request's chain below its boundary (``_prefix_on_chain``).
        if kept is not None and kept.uid in self.units:
            self._covered_by.setdefault(kept.uid, set()).add(unit.uid)
            self._covers.setdefault(unit.uid, set()).add(kept.uid)
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
            self._bump("hand_over", rows=[b.block_id for b in inherited_rows])
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

    def _prefix_on_chain(
        self, old: RestoreUnit, unit: RestoreUnit, chain: set
    ) -> bool:
        """``old`` is a prefix of the request ``unit`` was recorded from: a
        shorter boundary, and every non-endpoint key of ``old`` that ``unit``
        lacks is a full-attention key whose hash is on ``chain`` (that
        request's hashes below ``unit``'s boundary)."""
        return old.boundary < unit.boundary and all(
            k in unit.keys
            or (self._kind.get(get_group_id(k)) is GroupKind.FULL
                and get_block_hash(k) in chain)
            for k in old.keys - old.endpoints
        )

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
        cpu_blocks = list(cpu_blocks)
        self._bump("hold", rows=[b.block_id for b in cpu_blocks if not b.is_null])
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
            self._bump(
                "release_held", uids=(uid,), rows=[b.block_id for b in rows]
            )
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
            self._bump("release_visible", rows=ids)
            self.host_pool.free_blocks(free)

    def release_all_held(self) -> None:
        rows = [b for held in self._held.values() for b in held]
        self._held.clear()
        if rows:
            self.host_pool.free_blocks(rows)

    def _drop_unit(self, unit: RestoreUnit) -> None:
        self._bump("drop_unit", uids=(unit.uid,), keys=self.ordered(unit))
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
        # A dropped successor no longer covers
        # its predecessor (live again); a dropped predecessor leaves.
        for pred in self._covers.pop(unit.uid, ()):
            succs = self._covered_by.get(pred)
            if succs is not None:
                succs.discard(unit.uid)
                if not succs:
                    del self._covered_by[pred]
        for succ in self._covered_by.pop(unit.uid, ()):
            preds = self._covers.get(succ)
            if preds is not None:
                preds.discard(unit.uid)
                if not preds:
                    del self._covers[succ]
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
        self._bump("publish")
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
        self._bump("fallback")
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
            self._bump("touch")
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
            and not self._owned(block)
        )

    def _owned(self, block: KVCacheBlock) -> bool:
        """``bool(self._block_owners(block))`` for a host row, memoised per
        structure version (it reads the row's host keys and the unit keys
        only), so a full-queue ``idle_rows`` scan stops re-deriving owners."""
        if self._synced_sv != self._struct_version:
            self._sync_structure()
        memo = self._owned_memo
        got = memo[1].get(block.block_id)
        if got is None:
            got = memo[1][block.block_id] = bool(self._block_owners(block))
        return got

    def _watch_host_queue(self) -> None:
        """Count every mutation of the host free queue and of the host
        eviction groups (what ``_idle_ownerless`` and ``_row_free_now`` read
        live) in ``_queue_epoch``, before it runs. Wraps this pool's queue
        and group objects in place: both live as long as the pool."""
        def watch(obj, names):
            for name in names:
                fn = getattr(obj, name)

                def wrapped(*args, _fn=fn, **kwargs):
                    self._queue_epoch += 1
                    return _fn(*args, **kwargs)

                setattr(obj, name, wrapped)

        # The queue's moves are also
        # logged by block, for the leased flags (``_leased_flags``).
        queue = self.host_pool.free_block_queue
        log = self._queue_moved

        def note(blocks) -> None:
            if len(log) < _QUEUE_LOG_CAP:
                log.extend(blocks)
            else:
                self._queue_log_lost = True

        def logged(name, moved):
            fn = getattr(queue, name)

            def wrapped(*args, _fn=fn, **kwargs):
                self._queue_epoch += 1
                out = _fn(*args, **kwargs)
                note(moved(args, kwargs, out))
                return out

            setattr(queue, name, wrapped)

        def first(args, kwargs, out):
            return (args[0] if args else next(iter(kwargs.values())),)

        def first_list(args, kwargs, out):
            return args[0] if args else next(iter(kwargs.values()))

        logged("popleft", lambda a, k, out: (out,))
        logged("popleft_n", lambda a, k, out: out)
        logged("remove", first)
        logged("append", first)
        logged("prepend_n", first_list)
        logged("append_n", first_list)
        watch(self.host_pool.evict_groups, (
            "register", "leave", "leave_many", "clear", "_break",
        ))

    def _idle_list(self, ordered: bool) -> list[KVCacheBlock] | None:
        """The idle ownerless rows (``_idle_ownerless``), in free-queue order
        when ``ordered``; None until a full queue walk has cached the
        unit-less rows of this structure version (``_unowned_cache``: a
        tier with many idle rows keeps the early-stopping walks). Cached
        per (structure version, queue epoch): ownership is structure, refs,
        queue membership and eviction groups move only with the epoch."""
        if self._synced_sv != self._struct_version:
            self._sync_structure()
        sv = self._struct_version
        unowned = self._unowned_cache
        if unowned is None or unowned[0] != sv:
            return None
        key = (sv, self._queue_epoch)
        got = self._idle_cache
        if got is None or got[0] != key:
            self.plan_cost.idle_lists += 1
            grouped = self.host_pool.evict_groups._block_group
            rows = [
                b for b in unowned[1]
                if b.ref_cnt == 0 and b.prev_free_block is not None
                and b.block_id not in grouped
            ]
            got = self._idle_cache = (key, rows, len(rows) < 2)
        if ordered and not got[2]:
            # Their queue order: walk from the head until all are met.
            self.plan_cost.idle_order_walks += 1
            want = {b.block_id for b in got[1]}
            rows = []
            for block in self.host_pool.free_block_queue.iter_blocks_after(None):
                if block.block_id in want:
                    rows.append(block)
                    if len(rows) == len(want):
                        break
            got = self._idle_cache = (key, rows, True)
        return got[1]

    def _cache_unowned(self) -> None:
        """A walk met every queued row: cache the unit-less rows (queued or
        not: a pinned row re-enters the queue with no bump) for the next
        counts and lists of this structure version."""
        self.plan_cost.unowned_rebuilds += 1
        rows = [
            b for b in self.host_pool.blocks if not b.is_null and not self._owned(b)
        ]
        self._unowned_rows = {b.block_id for b in rows}
        self._unowned_cache = (self._struct_version, rows)

    def idle_count(self, limit: int) -> int:
        """``len(self.idle_rows(limit))``. A count needs no queue order. Until
        a walk has met the whole queue for this structure version, it walks
        as ``idle_rows`` does (ownership memoised, stopping at ``limit``);
        then it reads ``_idle_list``."""
        self.plan_cost.idle_counts += 1
        if limit <= 0:
            return 0
        rows = self._idle_list(False)
        if rows is not None:
            return min(limit, len(rows))
        n = 0
        for block in self.host_pool.free_block_queue.iter_blocks_after(None):
            if self._idle_ownerless(block):
                n += 1
                if n >= limit:
                    return n
        self._cache_unowned()
        return n

    def idle_rows(
        self, limit: int, skip: Callable[[KVCacheBlock], bool] | None = None
    ) -> list[KVCacheBlock]:
        """Up to ``limit`` idle ownerless host rows, in free-queue order
        (passing over the rows ``skip`` names).
        from ``_idle_list`` once established: the same rows in the same
        order, and ``skip`` is called on exactly the rows, in the order, the
        queue walk calls it on (idle ownerless ones, up to ``limit``)."""
        out: list[KVCacheBlock] = []
        if limit <= 0:
            return out
        rows = self._idle_list(True)
        if rows is not None:
            self.plan_cost.idle_rows_cached += 1
            for block in rows:
                if not (skip and skip(block)):
                    out.append(block)
                    if len(out) >= limit:
                        break
            return out
        for block in self.host_pool.free_block_queue.iter_blocks_after(None):
            if self._idle_ownerless(block) and not (skip and skip(block)):
                out.append(block)
                if len(out) >= limit:
                    return out
        self._cache_unowned()
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
        self._bump("abandon")
        self._abandoned.add(unit_id)
        self._release_held(unit_id)

    def resume(self, unit_id: UnitId) -> None:
        self._bump("resume")
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

    def _bump(
        self, site: str = "", *, uids=(), keys=(), rows=(), full: bool = False
    ) -> None:
        """Units, keys or host ownership changed: the candidate index is
        stale (rebuilt on the next plan, never per call).

        ``site`` counts the caller;
        ``uids`` / ``keys`` / ``rows`` (host block ids) name what changed, so
        ``_sync_structure`` re-derives only the units they can reach;
        ``full`` (or a site naming nothing structural: publication, use,
        abandonment, fallback, load refs) re-derives all, or only the cheap
        per-unit fields, respectively."""
        self._dir_version += 1
        self._struct_version += 1
        cost = self.plan_cost
        cost.struct_bumps += 1
        name = "bump_" + site
        if name in cost.__dict__:
            cost.__dict__[name] += 1
        if full:
            self._dirty_full = True
            return
        if uids:
            self._dirty_uids.update(uids)
        if keys:
            self._dirty_keys.update(keys)
        if rows:
            self._dirty_rows.update(rows)

    def _bump_gpu(self) -> None:
        """The GPU directory changed (a key added or removed): duplicate
        status may differ, the structure does not. The class pass reruns (it
        is keyed on the directory version); the structure is kept. A unit the
        change loses is dropped through ``_drop_unit``, which bumps both."""
        self._dir_version += 1
        self.plan_cost.gpu_bumps += 1

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
        only the live pin/queue state of the rows it considers.

        The per-unit structure comes from
        ``_structure`` (structure version); per (directory version, step)
        only the classes are re-derived, and the list is re-sorted only when
        the structure or a class changed. The result is the list the full
        rebuild gives: the same base order (``units`` order), the same
        classes (``_live_class``), the same sort key."""
        key = (self._dir_version, step)
        cached = self._cand_cache
        if cached is not None and cached[0] == key:
            return cached[1]
        self.plan_cost.class_passes += 1
        base = self._structure()
        changed = cached is None or cached[2] is not base
        for c in base:
            cls = self._live_class(c)
            if cls != c.cls:
                c.cls = cls
                changed = True
        if not changed:
            out = cached[1]
        elif self._fallback_of:
            self.plan_cost.resorts += 1
            out = sorted(base, key=self._fallback_order)
        else:
            self.plan_cost.resorts += 1
            out = sorted(base, key=_victim_key)
        self._cand_cache = (key, out, base)
        return out

    def _is_covered(self, uid: UnitId) -> bool:
        """``uid`` is a kept predecessor that a published, live later
        unit extends on its own hash chain: that unit restores the session,
        so losing ``uid`` costs only a restore at its own boundary (a retry
        or regenerate of the turn after it), never the session."""
        succs = self._covered_by.get(uid)
        if not succs:
            return False
        units, published, abandoned = self.units, self._published, self._abandoned
        return any(
            s in units and s in published and s not in abandoned for s in succs
        )

    def _structure(self) -> list[_Candidate]:
        """Every non-assembling unit with host rows, in ``units`` order, with
        its rows, owners, exclusive rows (and their host keys), last use,
        publication, abandonment and admission rank. Cached per structure
        version: all of it reads this table and the host directory only,
        whose every change bumps it (the GPU hooks do not: ``_bump_gpu``).
        ``cls`` is left to the class pass. A new version re-derives
        only the units its marks reach (``_sync_structure``)."""
        if self._synced_sv != self._struct_version:
            self._sync_structure()
        return self._struct_cache[1]

    def _derive(self, uid: int, unit: RestoreUnit) -> _Candidate | None:
        """One unit's candidate, from the table and the host directory now
        (None: assembling, or no host row). Records the rows it lists."""
        if uid in self._held:
            return None  # assembling: its landed rows are held (ref > 0)
        rows = tuple(self.host_members(unit))
        if not rows:
            return None
        host = self.host_pool
        owners = []
        row_keys = self._row_keys
        row_users = self._row_users
        key_units = self._key_units
        for b in rows:
            ks = tuple(self._keys_of(host, b))
            row_keys[b.block_id] = ks
            row_users.setdefault(b.block_id, set()).add(uid)
            o: set[int] = set()
            for k in ks:
                o |= key_units.get(k, set())
            owners.append(frozenset(o))
        owners = tuple(owners)
        exclusive = tuple(b for b, o in zip(rows, owners) if o <= {uid})
        return _Candidate(
            uid, rows, owners, "", 0, False, False, 0,
            exclusive=exclusive,
            excl_keys=tuple(row_keys[b.block_id] for b in exclusive),
            others=frozenset().union(*owners) - {uid},
        )

    def _sync_structure(self) -> None:
        """Bring the structure to the current version. A full derivation
        after a reset (or the first time); else only the units the marks of
        the bumps since can reach:

        * a marked unit (recorded, dropped, held or released);
        * every unit owning a marked key (its unit set changed);
        * every unit listing, or owning a key of, a marked host row or the
          row a marked key sat on or sits on now (its keys, aliases or
          owners changed).

        A unit's rows, owners and exclusive rows read only those (and the
        held set, whose changes mark the unit and its rows), so every other
        unit's candidate is unchanged. Last use, publication, abandonment
        and rank are re-read for all (cheap; touch, publish, abandon and the
        admission policy move them without marks). Row ownership
        (``_owned``) and the unit-less row set follow the same rows."""
        sv = self._struct_version
        cost = self.plan_cost
        units = self.units
        cand_of = self._cand_of
        if self._dirty_full or self._struct_cache is None:
            cost.struct_rebuilds += 1
            self._row_users = {}
            self._row_keys = {}
            cand_of = self._cand_of = {}
            for uid, unit in units.items():
                c = self._derive(uid, unit)
                if c is not None:
                    cand_of[uid] = c
            self._owned_memo = (sv, {})
            self._unowned_rows = None
            self._unowned_cache = None
        else:
            cost.struct_updates += 1
            host = self.host_pool
            key_units = self._key_units
            row_users, row_keys = self._row_users, self._row_keys
            rows = set(self._dirty_rows)
            affected = set(self._dirty_uids)
            key_row = None
            if self._dirty_keys:
                key_row = {}
                for r, ks in row_keys.items():
                    for k in ks:
                        key_row[k] = r
                # Every physical
                # copy of a marked key, not one: duplicate host copies are
                # legal, and each one's ownership memo and unit-less
                # membership follow the key's unit set.
                copies = host.cached_block_hash_to_block.get_blocks
                for k in self._dirty_keys:
                    affected |= key_units.get(k, set())
                    r = key_row.get(k)
                    if r is not None:
                        rows.add(r)
                    for b in copies(k):
                        rows.add(b.block_id)
            blocks = host.blocks
            for r in rows:
                affected |= row_users.get(r, set())
                for k in row_keys.get(r, ()):
                    affected |= key_units.get(k, set())
                for k in self._keys_of(host, blocks[r]):
                    affected |= key_units.get(k, set())
            for uid in affected:
                old = cand_of.pop(uid, None)
                if old is not None:
                    for b in old.rows:
                        users = row_users.get(b.block_id)
                        if users is not None:
                            users.discard(uid)
                            if not users:
                                del row_users[b.block_id]
                                row_keys.pop(b.block_id, None)
                unit = units.get(uid)
                if unit is not None:
                    c = self._derive(uid, unit)
                    if c is not None:
                        cand_of[uid] = c
            cost.struct_units += len(affected)
            memo = self._owned_memo[1]
            unowned = self._unowned_rows
            for r in rows:
                memo.pop(r, None)
            self._owned_memo = (sv, memo)
            if unowned is not None and self._unowned_cache is not None:
                for r in rows:
                    b = blocks[r]
                    if not b.is_null and not self._block_owners(b):
                        unowned.add(r)
                    else:
                        unowned.discard(r)
                self._unowned_cache = (sv, [blocks[r] for r in sorted(unowned)])
        self._dirty_uids = set()
        self._dirty_keys = set()
        self._dirty_rows = set()
        self._dirty_full = False
        last_use, published = self._last_use, self._published
        abandoned, rank = self._abandoned, self.policy.rank
        covered = self._is_covered
        out = []
        for uid in units:
            c = cand_of.get(uid)
            if c is None:
                continue
            c.last_use = last_use.get(uid, 0)
            c.published = uid in published
            c.abandoned = uid in abandoned
            c.rank = rank(uid)
            c.covered = covered(uid)
            out.append(c)
        self._struct_cache = (sv, out)
        self._synced_sv = sv
        if self.verify_structure:
            self._verify_structure(out)

    def _verify_structure(self, out: list[_Candidate]) -> None:
        """Debug (tests): the incremental structure equals a full one."""
        saved = (self._row_users, self._row_keys)
        self._row_users, self._row_keys = {}, {}
        try:
            full = []
            for uid, unit in self.units.items():
                c = self._derive(uid, unit)
                if c is not None:
                    c.last_use = self._last_use.get(uid, 0)
                    c.published = uid in self._published
                    c.abandoned = uid in self._abandoned
                    c.rank = self.policy.rank(uid)
                    c.covered = self._is_covered(uid)
                    full.append(c)
        finally:
            self._row_users, self._row_keys = saved

        def view(cs):
            return [(c.uid, tuple(b.block_id for b in c.rows), c.owners,
                     tuple(b.block_id for b in c.exclusive), c.excl_keys,
                     c.others, c.last_use, c.published, c.abandoned, c.rank,
                     c.covered)
                    for c in cs]

        assert view(out) == view(full), "incremental structure diverged"
        memo = self._owned_memo[1]
        for r, owned in memo.items():
            b = self.host_pool.blocks[r]
            assert owned == bool(self._block_owners(b)), f"owned memo {r}"
        # What the planner's idle
        # gate reads, against the directory: every row's ownership as
        # ``_owned`` answers it (memo or not), and no ownerless row is held
        # in a host eviction group (it could never be idle).
        for b in self.host_pool.blocks:
            if b.is_null:
                continue
            owners = self._block_owners(b)
            got = memo.get(b.block_id)
            if got is not None:
                assert got == bool(owners), f"owned memo {b.block_id}"
            if b.block_id in self.host_pool.evict_groups._block_group:
                assert owners, f"ownerless row {b.block_id} kept in a group"
        if self._unowned_rows is not None:
            want_rows = {
                b.block_id for b in self.host_pool.blocks
                if not b.is_null and not self._block_owners(b)
            }
            assert self._unowned_rows == want_rows, "unit-less row set diverged"
        cache = self._unowned_cache
        if cache is not None and cache[0] == self._struct_version:
            want = sorted(
                b.block_id for b in self.host_pool.blocks
                if not b.is_null and not self._block_owners(b)
            )
            got = [b.block_id for b in self._unowned_cache[1]]
            assert got == want, "unit-less rows diverged"

    def _live_class(self, c: _Candidate) -> str:
        """``self._victim_class(c.exclusive)``, read now, with the cached keys.

        ``twin_kind`` is a function of the live pins (the copy
        queue's pin census moves on every waiter pin change), so the calls
        made before the early exit are not observable and the
        absent-key witness is exact: ``_victim_class`` answers LRU as soon as
        it meets a row that is no duplicate, whatever the rows before it, and
        a row with no host key or with a key the GPU directory lacks is no
        duplicate. Such a row settles the class without a twin read; it
        stays absent until ``on_gpu_keyed`` moves the directory version."""
        dv = self._dir_version
        if c.witness_dv != dv:
            # Read once per directory version: absence ends only through
            # ``on_gpu_keyed`` (a new version).
            c.witness_dv = dv
            get = self.gpu_pool.cached_block_hash_to_block.get_one_block
            c.witness = any(
                not keys or any(get(k) is None for k in keys)
                for keys in c.excl_keys
            )
            c.rows_witness = not c.exclusive and any(
                not keys or any(get(k) is None for k in keys)
                for keys in (self._keys_of(self.host_pool, b) for b in c.rows)
            )
        if c.witness:
            self.plan_cost.witness_classes += 1
            return CLASS_LRU
        return self._class_of(c.exclusive, c.excl_keys)

    def _co_owners(self) -> dict[int, list[int]]:
        """Map uid -> the candidates (uids) sharing a row with it: the units
        whose take changes when ``uid`` is taken. Per structure version."""
        sv = self._struct_version
        got = self._co_owner_cache
        if got is not None and got[0] == sv:
            return got[1]
        out: dict[int, list[int]] = {}
        for c in self._structure():
            for o in c.others:
                out.setdefault(o, []).append(c.uid)
        self._co_owner_cache = (sv, out)
        return out

    def _leased_flags(self) -> dict[int, bool]:
        """Map uid -> whether one of the unit's exclusive rows is not free now
        (``_row_free_now``), for every candidate with an exclusive row.
        ``_row_free_now`` reads the row's ref count, its queue membership
        and the protection set; a host row is in the free queue only at ref
        0, so between two reads only the rows the queue moved (logged by
        ``_watch_host_queue``) and the rows that entered or left the
        protection set can have changed: those rows' units are re-read. The
        write-back sets a new protection set before nearly every ``room``,
        so it is diffed, not treated as a new world. A new structure or a
        lost log re-reads all."""
        sv, prot = self._struct_version, self._protected_rows
        got = self._leased
        log = self._queue_moved
        if got is None or got[0] != sv or self._queue_log_lost:
            log.clear()
            self._queue_log_lost = False
            free = self._row_free_now
            cands = {c.uid: c for c in self._structure() if c.exclusive}
            owner = {
                b.block_id: uid for uid, c in cands.items() for b in c.exclusive
            }
            flags = {
                uid: not all(free(b) for b in c.exclusive)
                for uid, c in cands.items()
            }
            self._leased = [sv, prot, flags, owner, cands]
            self._leased_version += 1
            self.plan_cost.leased_rebuilds += 1
            return flags
        flags, owner = got[2], got[3]
        moved: set = set()
        if got[1] is not prot:
            if got[1] != prot:
                moved.update(owner.get(r) for r in got[1] ^ prot)
            got[1] = prot
        if log:
            moved.update(owner.get(b.block_id) for b in log)
            log.clear()
        if moved:
            moved.discard(None)
            free, cands = self._row_free_now, got[4]
            self.plan_cost.leased_rereads += len(moved)
            for uid in moved:
                now = not all(free(b) for b in cands[uid].exclusive)
                if now != flags[uid]:
                    flags[uid] = now
                    self._leased_version += 1
        return flags

    def _skip_index(self, cands: list[_Candidate]) -> _SkipIndex:
        """The first-pass index of ``cands`` (``_SkipIndex``): categories per
        list (structure, order) and directory version (witnesses), leased
        positions per leased-flag version (``_leased_flags``)."""
        flags = self._leased_flags()
        dv = self._dir_version
        got = self._skip_cache
        if got is not None and got[0] is cands and got[1] == dv:
            idx = got[2]
        else:
            self.plan_cost.skip_indexes += 1
            idx = _SkipIndex(cands)
            self._skip_cache = (cands, dv, idx)
        if idx.sync(flags, self._leased_version):
            self.plan_cost.skip_syncs += 1
        return idx

    def _cand_class(self, c: _Candidate) -> str:
        """``_victim_class(c.exclusive)`` read now, inside a plan whose
        candidates came from this directory version's class pass: the pass's
        witness still holds (``_live_class``)."""
        if c.witness:
            self.plan_cost.witness_classes += 1
            return CLASS_LRU
        return self._class_of(c.exclusive, c.excl_keys)

    def _class_of(
        self,
        rows: Sequence[KVCacheBlock],
        row_keys: Sequence[Sequence[BlockHashWithGroupId]],
    ) -> str:
        """``_victim_class(rows)`` step for step, with each row's host keys
        given (``row_keys``, cached with the structure) instead of re-derived
        by ``is_duplicate`` and ``_twin_kind`` through ``_keys_of``."""
        if not rows:
            return CLASS_LRU
        tier = self.gpu_pool.tier
        kind_of = self.twin_kind
        running = True
        for block, keys in zip(rows, row_keys):
            if not keys:
                return CLASS_LRU
            for k in keys:  # ``is_duplicate``
                g = self._gpu_block(k)
                if g is None:
                    return CLASS_LRU
                if tier is None:
                    continue
                if not tier.is_ready(g):
                    return CLASS_LRU
                cid = tier.content_id(g)
                if cid is not None and tier.host_src.get(block.block_id) != cid:
                    return CLASS_LRU
            if running:  # ``_twin_kind``
                for k in keys:
                    twin = self._gpu_block(k)
                    kind = kind_of(twin) if kind_of is not None else (
                        "running" if twin.ref_cnt > 0 else "imminent"
                    )
                    if kind != "running":
                        running = False
                        break
        return CLASS_RUNNING_DUP if running else CLASS_IMMINENT_DUP

    def _lone_view(
        self, cands: list[_Candidate], preferred: set[int], displace: bool
    ) -> list[_Candidate]:
        """``cands`` reordered as ``plan`` does (``preferred`` first, then
        the rest, each in victim order), keeping only what lone mode does not
        skip: unpublished units, and preferred units when ``displace`` (they
        are the lane's own). The unpublished list and the uid positions are
        cached with the candidate list they were cut from."""
        got = self._lone_cache
        if got is None or got[0] is not cands:
            self.plan_cost.lone_views += 1
            got = self._lone_cache = (
                cands, [c for c in cands if not c.published],
                {c.uid: i for i, c in enumerate(cands)},
            )
        unpublished, pos = got[1], got[2]
        if not preferred:
            return unpublished
        first = [
            cands[i] for i in sorted(pos[u] for u in preferred if u in pos)
            if displace or not cands[i].published
        ]
        return first + [c for c in unpublished if c.uid not in preferred]

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
        self.plan_cost.plans += 1
        for_units = tuple(for_units)
        # The same count, without the queue walk.
        idle = self.idle_count(need_rows) if need_rows > 0 else 0
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
        self.plan_cost.plans_walked += 1
        cands = self._candidates(step)
        if self._lone_mode:
            # Lone mode visits only
            # the candidates the loop does not skip at its first test (an
            # unpublished unit, or a preferred one this lane may displace),
            # in the order below; a skipped candidate moves nothing.
            cands = self._lone_view(cands, preferred, displace_published)
        # Outside lone mode the preferred units
        # go first in ``_first_pass``.
        # Victims released so far: each gives up its host ownership (a
        # duplicate victim stays recorded, GPU-backed, but owns no host row
        # any more), so a shared row frees with the last of its owners
        # (shared roots of all-duplicate forks).
        gone: set[int] = set()
        freed: set[int] = set()
        remaining = list(cands)
        bypassed = False
        progress = False
        retry: list[_Candidate] = []

        # The loop body, unchanged,
        # as a function the indexed first pass calls too.
        def visit(cand: _Candidate) -> None:
            nonlocal bypassed, progress
            # A constrained successor with displacement rights may replace
            # its own kept predecessor (budgeted replacement): never a
            # deadlock. A lane without them (``displace_published=False``,
            # e.g. a deep-GRACE offer) gets no bypass.
            own = displace_published and cand.uid in preferred
            if self._lone_mode and cand.published and not own:
                return  # Rule A: complete units outrank fragments
            # No released victim co-owns a row of this unit (``gone``
            # disjoint from ``others``; ``cand.uid`` is never in ``gone``):
            # ``o - gone`` is ``o``, so the rows to take are exactly the
            # cached exclusive rows, none of which another victim can
            # have freed (each is owned by this unit alone).
            same = gone.isdisjoint(cand.others)
            if same:
                take = cand.exclusive
            else:
                take = [
                    b for b, o in zip(cand.rows, cand.owners)
                    if b.block_id not in freed and (o - gone) <= {cand.uid}
                ]
            if not take:
                # A possible zero-gain co-owner: the same permissions,
                # judged on all its rows (it gives every one of them up).
                # One pure read, or
                # the rows witness of this directory version's class pass.
                rows_cls = (
                    CLASS_LRU if cand.rows_witness
                    else self._victim_class(cand.rows)
                )
                if not displace_published and rows_cls != CLASS_RUNNING_DUP:
                    return
                if self._cf_only and self._cf_excluded(cand, rows_cls, preferred):
                    return
                # The admission filter as on the take path below: only the
                # LRU class stays out of a protected unit.
                if (
                    rows_cls == CLASS_LRU
                    and cand.rank > 0 and not own and not may_protected
                ):
                    bypassed = True
                    return
                retry.append(cand)
                return
            if not all(self._row_free_now(b) for b in take):
                c.evict_skipped_leased += 1
                return
            # Classify what is actually taken, live (rows a released
            # co-owner left; a GPU ref or grace change since the index was
            # cached), BEFORE the lane's permission and the admission
            # filter, exactly as ``apply``.
            # The same live check (same
            # calls, same order) on the cached keys of the exclusive rows.
            cls = (
                self._cand_class(cand) if same
                else self._victim_class(take)
            )
            if not displace_published and cls != CLASS_RUNNING_DUP:
                return
            if (
                cls == CLASS_LRU and cand.rank > 0
                and not own and not may_protected
            ):
                # Admission flag: a probation copy never displaces
                # protected content.
                bypassed = True
                return
            if self._cf_only and self._cf_excluded(cand, cls, preferred):
                return
            victims.append(cand.uid)
            ids = [b.block_id for b in take]
            rows.extend(ids)
            freed.update(ids)
            classes[cls] += len(ids)
            gone.add(cand.uid)
            progress = True

        # The first pass outside lone
        # mode visits only what the skip index cannot settle.
        indexed = not self._lone_mode
        while idle + len(freed) < need_rows:
            progress = False
            retry = []
            if indexed:
                indexed = False
                self._first_pass(
                    cands, preferred, displace_published, may_protected,
                    visit, lambda: idle + len(freed) >= need_rows,
                    victims, c,
                )
                bypassed = bypassed or self._first_pass_bypassed
            else:
                for cand in remaining:
                    if idle + len(freed) >= need_rows:
                        break
                    self.plan_cost.cands_visited += 1
                    visit(cand)
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
        # The counterfactual (diag only).
        if self._diag and not self._cf_only and self._takes_live(
            cands, victims, preferred
        ):
            self._cf_only = True
            try:
                alt = self.plan(
                    need_rows, step, displace_published=displace_published,
                    for_units=for_units, count=False,
                )
            finally:
                self._cf_only = False
            plan_args["live_over_covered"] = int(alt.satisfied)
        plan = EvictionPlan(victims=victims, rows=rows, classes=classes, **plan_args)
        if not plan.satisfied:
            c.evict_plans_short += 1
            if bypassed and count:
                self.policy.counters.probation_bypassed += 1
        return plan

    @staticmethod
    def _cf_excluded(cand: _Candidate, cls: str, preferred) -> bool:
        """The counterfactual's filter: an uncovered live LRU unit or an
        imminent twin, unless it is one of the successor's own preferred
        predecessors (visited first). Diag only: it tests whether the
        covered units alone could serve the plan."""
        if cand.uid in preferred:
            return False
        return cls == CLASS_IMMINENT_DUP or (cls == CLASS_LRU and not cand.covered)

    def _takes_live(self, cands, victims, preferred) -> bool:
        """The plan took a live (uncovered) LRU unit other than a preferred
        predecessor."""
        by = {c.uid: c for c in cands}
        return any(
            u not in preferred and (c := by.get(u)) is not None
            and c.cls == CLASS_LRU and not c.covered
            for u in victims
        )

    def _first_pass(
        self, cands, preferred, displace, may_protected, visit, satisfied,
        victims, counters,
    ) -> None:
        """``plan``'s first pass over ``cands`` outside lone mode (the
        preferred units first, then the rest, each in victim order), visiting
        only what ``_skip_index`` cannot settle: its evaluated positions, the
        preferred units, and every later co-owner of a unit taken here (its
        take gained their shared rows). The runs between are settled in bulk
        exactly as the loop would settle each candidate:
        ``evict_skipped_leased`` per leased one, ``bypassed`` for a protected
        witness. A run is never cut short: only a take satisfies the plan,
        and takes happen at visited positions."""
        self._first_pass_bypassed = False
        cost = self.plan_cost
        firsts = [x for x in cands if x.uid in preferred] if preferred else ()
        for cand in firsts:
            if satisfied():
                return
            cost.cands_visited += 1
            visit(cand)
        idx = self._skip_index(cands)
        pos = idx.pos
        stops = idx.must(displace, may_protected)
        if preferred:
            stops = sorted(set(stops).union(pos[u] for u in preferred if u in pos))
        co = self._co_owners() if victims else None
        extra: list[int] = []  # co-owners of a unit taken: a heap
        for v in victims:  # taken by the preferred visits
            extra.extend(pos[u] for u in co.get(v, ()) if u in pos)
        heapq.heapify(extra)
        leased, bypass, n = idx.leased_pos, idx.bypass_pos, idx.n
        bulk_bypass = displace and not may_protected and bool(bypass)
        k, cur = 0, 0
        while not satisfied():
            while extra and extra[0] < cur:
                heapq.heappop(extra)
            nxt = stops[k] if k < len(stops) else n
            if extra and extra[0] < nxt:
                nxt = extra[0]
            if leased:
                counters.evict_skipped_leased += (
                    bisect.bisect_left(leased, nxt) - bisect.bisect_left(leased, cur)
                )
            if bulk_bypass and (
                bisect.bisect_left(bypass, nxt) != bisect.bisect_left(bypass, cur)
            ):
                self._first_pass_bypassed = True
            cost.cands_skipped += nxt - cur
            if nxt == n:
                return
            cur = nxt + 1
            if k < len(stops) and stops[k] == nxt:
                k += 1
            cand = cands[nxt]
            if cand.uid in preferred:
                continue  # visited first
            before = len(victims)
            cost.cands_visited += 1
            visit(cand)
            if len(victims) > before:
                if co is None:
                    co = self._co_owners()
                for v in victims[before:]:
                    for u in co.get(v, ()):
                        j = pos.get(u)
                        if j is not None and j > nxt:
                            heapq.heappush(extra, j)

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
        if self._diag:
            self._diag_plan(plan)
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
                        self._bump("apply")
                        if uid in self._superseded_by:
                            self._complete_supersede(uid)
                        continue
                    self.policy.on_evict(uid, self._private_keys(unit))
                    self._count_evicted(unit)
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
                self._bump("apply")
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
                    self._count_evicted(unit)
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
        rows = list(self.host_members(unit))
        # Also the rows it registered that
        # are still in its group, whatever key now finds them.
        reg = self._group_of.pop(unit.uid, None)
        if reg is not None:
            gid, ids = reg
            seen = {b.block_id for b in rows}
            blocks = self.host_pool.blocks
            rows += [
                blocks[i] for i in ids
                if i not in seen and groups._block_group.get(i) == gid
            ]
        groups.leave_many(rows)  # one rebuild
        unit.host_group = False

    def _maybe_register_host(self, unit: RestoreUnit) -> None:
        """Register a published, complete unit's exclusive host copies."""
        if (
            unit.host_group or unit.uid not in self._published
            or not self.is_host_complete(unit)
        ):
            return
        members = self._exclusive(unit)
        groups = self.host_pool.evict_groups
        gid = groups._next_id
        if groups.register(members) > 0:
            unit.host_group = True
            self.counters.units_host_registered += 1
            self._group_of[unit.uid] = (gid, [
                b.block_id for b in members
                if groups._block_group.get(b.block_id) == gid
            ])

    def on_host_registered(self, cpu_blocks: Iterable[KVCacheBlock]) -> None:
        """Refresh groups after store refs drop; publication is explicit."""
        self.flush()
        cpu_blocks = list(cpu_blocks)
        self._bump(
            "host_registered", rows=[b.block_id for b in cpu_blocks if not b.is_null]
        )
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
        self._bump("host_released")

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

    def _is_stale(self, unit: RestoreUnit) -> bool:
        """Diagnosis only: a later unit on the same chain holds every
        non-endpoint key of ``unit`` but at most one (the page holding its
        boundary, which the later turn keys whole): an earlier turn of a
        session that has moved on."""
        rest = unit.keys - unit.endpoints
        if not rest:
            return False
        held: dict[int, int] = {}
        for key in rest:
            for uid in self._key_units.get(key, ()):
                if uid != unit.uid:
                    held[uid] = held.get(uid, 0) + 1
        return any(
            n >= len(rest) - 1 and (u := self.units.get(uid)) is not None
            and u.boundary > unit.boundary
            for uid, n in held.items()
        )

    def _diag_plan(self, plan: EvictionPlan) -> None:
        """MERLIN_KV_TIER_DIAG=1: the plan's victims (class, age in steps,
        stale or live) and the host row split, when the plan takes a unit
        with a host-only row (a loss). Reads only; changes nothing."""
        rows: list[str] = []
        lossy = False
        stale_cache: dict[int, bool] = {}

        def stale(uid: int) -> bool:
            if uid not in stale_cache:
                u = self.units.get(uid)
                stale_cache[uid] = bool(u is not None and self._is_stale(u))
            return stale_cache[uid]

        for uid in plan.victims:
            unit = self.units.get(uid)
            if unit is None:
                continue
            take = [b for b in self.host_members(unit) if self._block_owners(b) <= {uid}]
            cls = self._victim_class(take) if take else CLASS_LRU
            lossy |= cls == CLASS_LRU and bool(take)
            last = self._last_use.get(uid)
            age = plan.step - last if last is not None else -1
            rows.append(
                f"{uid}:{cls}:age={age}:{'stale' if stale(uid) else 'live'}"
                f":rows={len(take)}:tokens={unit.boundary}"
                f":covered={int(self._is_covered(uid))}"
            )
        if not lossy:
            return
        split = {"ownerless": 0, "twin": 0, "stale": 0, "live": 0, "idle": 0}
        # What the planner could take now:
        # the same classes counting free rows only (``_row_free_now``: no
        # ref, in the queue, not a protected restore row), and the free rows
        # whose every owner is a covered predecessor (``_is_covered``).
        free = {"ownerless_free": 0, "twin_free": 0, "stale_free": 0,
                "covered_free": 0}
        # Ownerless keyed rows (no unit owns a key of theirs) by the first
        # gate of the planner's idle test (``_idle_ownerless``) they fail: the
        # ownership memo still says owned (stale memo), ref'd, out of the
        # free queue, in a host eviction group, else idle (``idle_count``
        # takes it). ``own_prot``: of the idle ones, the protected restore
        # rows (``_row_free_now``'s gate only; idle for ``idle_count``).
        grouped = self.host_pool.evict_groups._block_group
        own = {"own_memo": 0, "own_ref": 0, "own_offq": 0, "own_group": 0,
               "own_idle": 0, "own_prot": 0}
        memo = self._owned_memo[1] if self._owned_memo else {}
        for block in self.host_pool.blocks:
            if block.is_null:
                continue
            if block.block_hash is None:
                split["idle"] += 1
                continue
            owners = self._block_owners(block)
            ok = self._row_free_now(block)
            if not owners:
                split["ownerless"] += 1
                free["ownerless_free"] += ok
                if memo.get(block.block_id):
                    own["own_memo"] += 1
                elif block.ref_cnt > 0:
                    own["own_ref"] += 1
                elif block.prev_free_block is None:
                    own["own_offq"] += 1
                elif block.block_id in grouped:
                    own["own_group"] += 1
                else:
                    own["own_idle"] += 1
                    own["own_prot"] += block.block_id in self._protected_rows
            elif self.is_duplicate(block):
                split["twin"] += 1
                free["twin_free"] += ok
            elif all(stale(uid) for uid in owners):
                split["stale"] += 1
                free["stale_free"] += ok
            else:
                split["live"] += 1
            if ok and owners and all(self._is_covered(u) for u in owners):
                free["covered_free"] += 1
        split.update(free)
        split.update(own)
        split["live_over_covered"] = plan.live_over_covered
        logger.info(
            "KV tier diag: host evict step=%d rows=%d victims=[%s] split %s",
            plan.step, len(plan.rows), " ".join(rows),
            " ".join(f"{k}={v}" for k, v in split.items()),
        )

    def _count_evicted(self, unit: RestoreUnit) -> None:
        """``unit`` is lost (a host eviction un-keyed it, or a lost key):
        ``units_lost``, and ``units_lost_covered_early`` when it was a
        covered kept predecessor. The miss cause is unchanged (a returning
        retry of that boundary did lose it)."""
        self.counters.units_lost += 1
        if self._is_covered(unit.uid):
            self.counters.units_lost_covered_early += 1

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
            self._count_evicted(unit)
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
        blocks = list(blocks)
        self._bump("host_remains", rows=[b.block_id for b in blocks])
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
        # Duplicate status only (``_bump_gpu``).
        self._bump_gpu()  # a host row may have become a duplicate
        self._reclaiming = None
        self.flush()

    def on_gpu_unhashed(
        self, block: KVCacheBlock, keys: Sequence[BlockHashWithGroupId]
    ) -> None:
        reclaimed = self._reclaiming == block.block_id
        self._bump_gpu()
        self._reclaiming = None
        self.flush()
        self._pending_unhashed.extend(keys)
        if reclaimed:
            self.flush()

    def on_gpu_reclaim(self, block: KVCacheBlock, state: BlockState) -> None:
        self.flush()
        self._reclaiming = block.block_id

    def on_gpu_reset(self) -> None:
        # Every host row stopped being a
        # duplicate; the class pass reruns even within this step.
        self._bump_gpu()
        self._reclaiming = None
        self._pending_unhashed.clear()
        self._loss_cause = "race"
        try:
            for key in list(self._key_units):
                if self._host_block(key) is None:
                    self._on_key_lost(key)
        finally:
            self._loss_cause = None

    def on_host_rekeyed(self, block: KVCacheBlock) -> None:
        """A host row lost only some of its keys (a newer copy superseded
        them) and stays keyed: its owners changed, so the units reading it
        re-derive. it also leaves
        its eviction group: the group is a unit's exclusive rows by key, and
        the superseded key now finds the newer row, so the unit's cleanup
        would not visit this one (an ownerless row kept grouped is never
        idle)."""
        self.host_pool.evict_groups.leave(block)
        self._bump("host_rekeyed", rows=(block.block_id,))

    def on_host_unkeyed(self, block: KVCacheBlock) -> None:
        """A host block is losing its keys (still in the directory here)."""
        bid = block.block_id
        if bid in self._unkeying:
            return
        self._bump("host_unkeyed", rows=(bid,))
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
        self._bump("host_reset", full=True)
        self.release_all_held()
        self._published.clear()
        self._visible.clear()
        # No published predecessor survives a host reset.
        self._superseded_by.clear()
        self._pending_preds.clear()
        self._fallback_of.clear()
        self._fallback.clear()
        self.policy.clear()
        self._group_of.clear()
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
        self._covered_by.clear()
        self._covers.clear()
        self._group_of.clear()
        self._fallback_of.clear()
        self._fallback.clear()
        self.policy.clear()
        self._cand_cache = None
        self._bump("clear", full=True)


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
