# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Host-RAM KV tier, write-back engine and admission gate.

GPU and host are two levels of one cache. The GPU side's block state is the
allocator's (``BlockPool.tier``, vllm/v1/core/kv_tier_state.py): ``DIRTY``
(hashed, stored by the tier, some key not on the host), ``CLEAN`` (host-backed
or not worth keeping) or ``DEMOTING`` (a copy is in flight; the source is
touched, so it is not in the free queue). The state is derived on every query
(host backing is revalidated against the host directory), so a host unkey or
reset needs no reverse transition here. This module decides and issues copies:

* Walker (``WriteBack.select``). Once per scheduler step it reads the next
  reclaims in the allocator's own pop order (``iter_pool_reclaims``:
  junction priority and ``EvictGroups`` expansion included) over a window of
  the running headroom plus the admission gate's ``demand_hint``. The first
  dirty member of a restore unit schedules the
  whole unit (``UnitTable.gpu_members``, endpoints last); members it cannot
  issue this step wait in a backlog and go first next step. Only ready
  content is copied (``BlockTierState.is_ready``: every rank retired its
  writer and no commit was masked). Copies are bounded by DMA credits (``P``)
  and by the running requests' allocation headroom: a copy pin never takes a
  free block a running request needs this step. With the host pool full and
  nothing evictable a dirty victim is left to be reclaimed (counted
  ``demote_dropped_host_full``): a host copy is optional, storage safety is
  not.
  A victim that is not ready yet does not use up the window: the walk reaches
  past it (up to ``P`` more victims), so a ready victim behind an unready head
  is still saved this step.
* Host eviction. A copy takes the host pool's next rows in its
  own pop order, which is LRU by last use (whole host units go together:
  ``EvictGroups``). Nothing is refused by recency or by a value rank. A unit
  with fewer than ``TINY_FA_PAGES`` full-attention pages is not copied
  (``demote_skipped_tiny``); a unit the host cannot hold now is not started
  (``demote_unit_too_large``). Host rows come from idle rows, else from
  ``UnitTable.plan``/``apply`` (exclusive rows only, no recency
  refusal). A unit-less copy never takes an idle row that is the only host
  copy of a free GPU block's content (un-keying it re-dirties that
  block, so the copy moves a dirty block instead of removing one); rows are
  judged as if the ones taken before them were gone, and a block that loses
  nothing (already dirty, or clean whatever the host holds) protects no row.
  The admission mirror applies the same decision.
* Transactions (``DemoteTxn``). One store event is one transaction with an
  id (the event index), the cache epoch at issue and the demotion tickets of
  its sources. It publishes only when every rank acked it (idempotent per
  rank) and its epoch is current; each host copy is registered only if its
  source still holds the content the copy read (``demote_still_valid``) and
  the keys snapshotted at issue (TierRuntime.keys_to_register). Published
  sources go to the FRONT of the GPU free queue (``end_demote``); anything
  else is released normally and stays dirty.
* Admission gate (``WriteBack.admit``, installed on KVCacheManager). For
  WAITING / PREEMPTED requests only: the gate simulates the allocation's pops
  (the next reclaims after the blocks it adopts) and, if they hold protected
  content (dirty and ready), returns a demotion wait (no blocks, shortfall 0,
  so no pin is yielded) and feeds ``demand_hint``. A real shortfall that
  in-flight demotions will cover is a demotion wait too. The gate escapes
  after ``ESCAPE_STEPS`` counted waits without progress on the request's
  own deficit, with a batch-scaled total-wait backstop, or at once when the
  walker cannot progress. The idle breaker also releases waits after
  ``IDLE_BREAK_STEPS`` consecutive walker steps with nothing running,
  selected or in flight (``gate_idle_breaks``), preserving real shortfalls;
  a re-copy of content already landed once in the current waiting episode,
  under no new key, is not progress for it.
  Dirty reclaims from escapes are counted ``demote_missed_escape``.
  Running requests never wait: other dirty reclaims are
  ``demote_missed_running``.

Accounts. There is no retention account: admitted restores are bounded by a
restores-in-flight cap in the planner, and the tier's minimum size is one
max-length restore unit (the math, ``kv_host_tier``).
Quarantine (``account_quarantine``): stale-epoch demotions still in flight
plus abandoned loads' destinations. ``H_run`` (``account_running``): the exact
block need of the running requests' next step, computed by the scheduler hook
after allocation and consumed by the next walk; ``select`` falls back to the R
estimate only when no hook ran (the walker driven directly).
DMA credits (``P``) are issued, not completed copies, in both directions: the
walker only issues demotions into credits that in-flight restores leave free.
Restores never wait on credits here, and their reservations (host hit pins,
GPU destinations) take none, so a restore larger than ``P`` is never
deadlocked by the cap. Pinned restore sources are reported as
``promote_pins``.
"""

import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from vllm.logger import init_logger
from vllm.merlin.kv_host_tier_lifecycle import TxnAcks, TxnState
from vllm.utils.math_utils import cdiv
from vllm.v1.core.kv_tier_state import BlockState, iter_pool_reclaims

if TYPE_CHECKING:
    from vllm.merlin.kv_host_tier_units import UnitTable
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.kv_cache_utils import KVCacheBlock
    from vllm.v1.core.kv_tier_state import BlockTierState, DemoteTicket
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)

# Gate tuning envs, documented under "KV RAM cache tuning knobs" in
# models/flash-next/docs/reference.md; refused with the tier off
# (kv_host_tier.post_init).
CREDITS_ENV = "MERLIN_KV_HOST_TIER__COPY_CREDITS"
ESCAPE_ENV = "MERLIN_KV_HOST_TIER__ESCAPE_STEPS"
DEFAULT_COPY_CREDITS = 32
DEFAULT_ESCAPE_STEPS = 8
# Idle walker steps (nothing running, selected or in flight) after
# which the admission gate stops holding waiters.
IDLE_BREAK_STEPS = 2
# A unit with fewer full-attention pages than this is not worth
# a host copy (a flush or a short probe); it is the next eviction instead.
TINY_FA_PAGES = 2
# Units assembling at once (their landed rows are held until publish);
# a new unit waits while this many are in progress, so held halves never fill
# the host. MERLIN_KV_HOST_TIER__MAX_ASSEMBLING overrides it.
MAX_ASSEMBLING = 2
ASSEMBLING_ENV = "MERLIN_KV_HOST_TIER__MAX_ASSEMBLING"
N_RESTORE_ENV = "MERLIN_KV_HOST_TIER__N_RESTORE"
DEFAULT_N_RESTORE = 2
# Per-direction DMA credits. The demotion direction (D) has its own row cap and
# an optional byte cap per rank; the restore direction (R) keeps
# ``COPY_CREDITS`` (``credits_cap``, read by the plans tick). One shared credit
# pool would throttle write-back to a fraction of the stream rate when acks
# take several steps.
CREDITS_D_ENV = "MERLIN_KV_HOST_TIER__COPY_CREDITS_D"
BYTES_D_ENV = "MERLIN_KV_HOST_TIER__COPY_BYTES_D_MIB"
# Optional total write-back source-reference cap (0 = off).
STORE_PINS_ENV = "MERLIN_KV_HOST_TIER__MAX_PENDING_STORE_BLOCKS"
# Keep a published successor's predecessor as its fallback restore point
# (``kv_host_tier_units``); 0/unset = off.
FALLBACK_ENV = "MERLIN_KV_HOST_TIER__FALLBACK_BOUNDARY"
# The D default equals COPY_CREDITS (32 rows); COPY_CREDITS_D sets another.
DEFAULT_COPY_CREDITS_D = DEFAULT_COPY_CREDITS
GATE_ENVS = (
    CREDITS_ENV, ESCAPE_ENV, N_RESTORE_ENV, ASSEMBLING_ENV,
    CREDITS_D_ENV, BYTES_D_ENV, STORE_PINS_ENV, FALLBACK_ENV,
)


def copy_credits() -> int:
    from vllm.merlin.envs import _e2_positive_int

    return _e2_positive_int(CREDITS_ENV, DEFAULT_COPY_CREDITS)


def copy_credits_d() -> int | None:
    """Demotion-direction row credits when set explicitly; None = follow
    ``credits_cap`` (COPY_CREDITS)."""
    import os

    from vllm.merlin.envs import _e2_positive_int

    if CREDITS_D_ENV in os.environ:
        return _e2_positive_int(CREDITS_D_ENV, DEFAULT_COPY_CREDITS_D)
    return None


def copy_bytes_d() -> int:
    """Demotion-direction byte cap per rank (0 = rows only)."""
    from vllm.merlin.envs import _e2_nonneg_int

    return _e2_nonneg_int(BYTES_D_ENV, default=0) << 20


def fallback_boundary() -> bool:
    """Whether a published successor keeps its predecessor as its
    fallback restore point (exactly 0 or 1; unset = off)."""
    from vllm.merlin.envs import _e2_strict_switch

    return bool(_e2_strict_switch(FALLBACK_ENV, default=False))


def max_pending_store_blocks() -> int:
    """Maximum distinct write-back source blocks, or zero for no cap."""
    from vllm.merlin.envs import _e2_nonneg_int

    return _e2_nonneg_int(STORE_PINS_ENV, default=0)


def escape_steps() -> int:
    from vllm.merlin.envs import _e2_positive_int

    return _e2_positive_int(ESCAPE_ENV, DEFAULT_ESCAPE_STEPS)


def n_restore() -> int:
    from vllm.merlin.envs import _e2_positive_int

    return _e2_positive_int(N_RESTORE_ENV, DEFAULT_N_RESTORE)


def max_assembling() -> int:
    from vllm.merlin.envs import _e2_positive_int

    return _e2_positive_int(ASSEMBLING_ENV, MAX_ASSEMBLING)


def derive_reserve(
    kv_cache_config: "KVCacheConfig",
    max_num_batched_tokens: int,
    cp_world_size: int = 1,
) -> tuple[int, int]:
    """``(R, alignment_groups)``: the running headroom bound. ``R`` is the
    step's token budget in blocks per group (plus the sliding window and the
    recurrent rotation pair), and ``alignment_groups`` the number of groups in
    which every running request can need one more block in one step (a block
    edge crossing, or an align-mode recurrent rotation); the walker adds one
    block per running request per such group, so independently aligned
    requests are not rounded down to one block."""
    from vllm.v1.core.kv_cache_utils import resolve_dcp_kv_block_size
    from vllm.v1.kv_cache_interface import MambaSpec, SlidingWindowSpec

    reserve = 0
    align_groups = 0
    for g in kv_cache_config.prefix_cacheable_groups:
        spec = g.kv_cache_spec
        block_size = resolve_dcp_kv_block_size(spec, cp_world_size)
        if isinstance(spec, MambaSpec):
            # Align-mode rotation: the running state block and its successor,
            # and a new state block per running request at a boundary.
            reserve += 2
            align_groups += 1
        elif isinstance(spec, SlidingWindowSpec):
            reserve += cdiv(spec.sliding_window, block_size) + 1
            align_groups += 1
        else:
            reserve += cdiv(max_num_batched_tokens, block_size)
            align_groups += 1
    return reserve, align_groups


@dataclass
class WritebackCounters:
    """Cumulative write-back counters, on the tier log line."""

    demoted: int = 0
    demoted_units: int = 0  # mirrors UnitCounters.units_host_registered
    # ``room()`` calls (observational).
    room_calls: int = 0
    demote_redemoted: int = 0  # mirrors TierBlockCounters.reclaimed_backed
    demote_missed_running: int = 0
    demote_missed_escape: int = 0
    unit_lost: int = 0  # mirrors UnitCounters.units_lost
    demote_dropped_host_full: int = 0
    # A victim met with every DMA credit taken (in-flight copies and
    # restores): not started, left to be reclaimed, counted once.
    demote_dropped_credits: int = 0
    # A unit under TINY_FA_PAGES full-attention pages is not copied
    # (the next eviction). Nothing is refused by recency or by a value rank
    # (no retention interval).
    demote_skipped_tiny: int = 0
    # A unit the host cannot hold now is not started.
    demote_unit_too_large: int = 0
    # A new unit waited for MAX_ASSEMBLING units in progress to publish, or
    # for their remaining rows (with its own) to fit the host cache.
    demote_assembly_waits: int = 0
    # A unit that fits the host cache waited because the rows it needs are
    # pinned now (leases, held or staged rows): a wait, not never_fit.
    demote_host_busy: int = 0
    # A unit not started because pinning all its
    # free GPU members would take the running headroom (a wait; its loss,
    # if reclaimed, is missed_copy).
    demote_gpu_tight: int = 0
    # Host rows reserved for started units' members not issued yet,
    # now and peak.
    reserved_rows: int = 0
    reserved_rows_peak: int = 0
    # Unit members pinned on the GPU until their copy issues, and the
    # peak held at once.
    demote_held: int = 0
    demote_held_peak: int = 0
    # Current gauges, set at snapshot time (not cumulative): GPU blocks
    # held for a pending copy, and the distinct restore units they belong
    # to (counted by unit, not by reservation key: one key can span
    # several units and a held referenced member need not be reserved).
    held_gpu_now: int = 0
    held_units_now: int = 0
    # A pin, or a unit start, refused by the hold
    # budget (the running headroom plus the waiting head's need, and at most
    # one restore unit of free blocks pinned); and pins released because a
    # held unit made no copy progress for HOLD_STALL steps.
    demote_held_capped: int = 0
    demote_held_released: int = 0
    # A copy not issued this step because its
    # new DMA source pin on a free block would breach H_run + the waiting
    # head's need (the walk stops there; counted once per stop).
    demote_copy_capped: int = 0
    # Cumulative time in the walker (select), microseconds.
    select_us: int = 0
    # Exact H_run skipped: free blocks far above the bound.
    h_run_skipped: int = 0
    demote_not_ready: int = 0
    demote_discarded: int = 0
    demote_wait_steps: int = 0
    demote_wait_max: int = 0
    # Escaping admissions: once per request until it waits again, ends or is
    # preempted (a re-poll before it allocates is the same escape).
    demote_escapes: int = 0
    # Waits released because nothing was running and the walker sat
    # idle with nothing in flight (no progress source; otherwise a livelock).
    gate_idle_breaks: int = 0
    credits: int = 0
    credits_peak: int = 0
    promote_pins: int = 0
    # Quarantine: storage of issued copies that no longer publish (stale
    # epoch demotions, abandoned loads) until quiescence; <= issued DMA.
    quarantine: int = 0
    quarantine_peak: int = 0
    # Exact running headroom H_run of the last walk, and its peak.
    h_run: int = 0
    h_run_peak: int = 0
    restore_failed: int = 0  # mirrors TierCounters.restore_failed
    stale_epoch_txns: int = 0
    # cache_thrash session breaks: host-backed (hashes kept) / destructive.
    thrash_demoted: int = 0
    thrash_unit_lost: int = 0
    # Stored-group pages reclaimed as no restore dependency (out-of-
    # window sliding-window pages); mirrors TierBlockCounters.
    reclaimed_unretained: int = 0
    # Copy rows per lane,
    # host-aware yield, preempt grace, backing repair, per-direction credits
    # and the unique-row assembly ledger.
    copy_grace_rows: int = 0
    copy_preempt_rows: int = 0
    copy_running_rows: int = 0
    copy_repair_rows: int = 0
    copy_reactive_rows: int = 0
    yield_incomplete: int = 0
    yield_waited_steps: int = 0
    grace_offers: int = 0  # GRACE offers a walk accepted
    preempt_grace_taken: int = 0
    preempt_grace_expired: int = 0  # past PREEMPT_GRACE_STEPS
    preempt_grace_shed: int = 0  # released early: free pool under the floor
    preempt_grace_refused: int = 0
    backing_revoked: int = 0
    repair_pending: int = 0
    repair_done: int = 0
    lane2_rows: int = 0
    lane2_late: int = 0
    credits_d_issued: int = 0
    credits_r_issued: int = 0
    bytes_d_inflight_peak: int = 0
    bytes_r_inflight_peak: int = 0
    assembly_unique_rows_peak: int = 0
    assembly_unique_bytes_peak: int = 0
    copyq_offer_skipped: int = 0  # offers the walker could not place


@dataclass
class DemoteTxn:
    """One demotion transaction = one store event."""

    txn_id: int
    epoch: int
    tickets: list["DemoteTicket"]
    acked_ranks: set[int] = field(default_factory=set)
    acks: "TxnAcks | None" = None


class WriteBack:
    def __init__(
        self,
        cpu_pool: "BlockPool",
        prefix_cacheable_group_ids: Iterable[int],
        reserve_blocks: int,
        alignment_groups: int,
        world_size: int,
        credits_cap: int | None = None,
        escape_after: int | None = None,
    ) -> None:
        self.cpu_pool = cpu_pool
        self.group_ids = frozenset(prefix_cacheable_group_ids)
        self.reserve = reserve_blocks
        self.alignment_groups = alignment_groups
        self.world_size = world_size
        self.credits_cap = copy_credits() if credits_cap is None else credits_cap
        # Per-direction credits. Demotion copies use
        # ``credits_d`` rows (and ``bytes_d`` bytes per rank when set) and
        # never subtract restore DMA; restores keep ``credits_cap`` (the
        # plans tick) and stop subtracting the walker's in-flight copies
        # (split directions). An explicit ``credits_cap`` sets both
        # directions.
        # None = the D credits follow ``credits_cap`` (read live, so a
        # caller that changes ``credits_cap`` changes both).
        self._credits_d: int | None = (
            copy_credits_d() if credits_cap is None else None
        )
        self._bytes_d = copy_bytes_d() if credits_cap is None else 0
        self.max_store_pins = max_pending_store_blocks()
        # Host row bytes per rank (set by the runtime; 0 = rows only).
        self._row_bytes = 0
        # True once the restore direction keeps its own credit account
        # (``plans.begin_tick`` does not subtract ``in_flight``; the planner
        # sets it). Until then the D budget also subtracts restore DMA.
        self.split_directions = False
        # The copy scheduler (``kv_host_tier_copyq.CopyQueue``), created by
        # ``bind`` with a unit table; offers flow only once ``bind_manager``
        # gave it the KVCacheManager (grace pins, SessionYield).
        self.copyq: Any = None
        # GPU block id -> the copy-queue offer its issued copy belongs to.
        self._offer_of: dict[int, Any] = {}
        # Lane-1 inputs captured by ``account_running`` (the scheduler's
        # running list and coordinator of the step just scheduled).
        self._running_reqs: Sequence[Any] = ()
        self._coordinator: Any = None
        # Lane-1 rows the copy scheduler may offer this walk (``_lane1_allow``,
        # set by ``_select``; 0 = none).
        self._lane1_allow = 0
        self.escape_after = escape_steps() if escape_after is None else escape_after
        self.gpu_pool: BlockPool | None = None
        self.units: UnitTable | None = None
        self.counters = WritebackCounters()
        # Units in progress at once (MAX_ASSEMBLING unless overridden).
        self.max_assembling = max_assembling()
        # Exact H_run of this step, set by ``account_running`` from the
        # scheduler hook and consumed by the next ``select`` (None: no hook ran
        # this step, ``select`` falls back to the reserve estimate).
        self.h_run: int | None = None
        # This step's running headroom as account_running computed it (exact,
        # or the conservative bound when the exact value was not needed);
        # read by prefetch, which runs between account_running and the walker,
        # and cleared when the walker consumes the step.
        self.headroom_now: int | None = None
        self.epoch = 0
        self.step = 0
        # Consecutive walker steps with nothing running, nothing in
        # flight and nothing selected (``select``); the gate holds no waiter
        # once it reaches IDLE_BREAK_STEPS.
        self.idle_steps = 0
        # GPU block id -> its demotion ticket, for every source still pinned.
        self.in_flight: dict[int, DemoteTicket] = {}
        # Tickets selected this step, awaiting their store event (``issue``).
        self._staged: list[DemoteTicket] = []
        self.txns: dict[int, DemoteTxn] = {}
        # Demotions unwound by a per-rank failure report.
        self.failed_txns = 0
        # Unit members not yet issued: they go first next step.
        self.backlog: list[KVCacheBlock] = []
        # Largest allocation the gate made wait since the last walk.
        self.demand_hint = 0
        # Restore (H2D) blocks issued and not completed: they hold credits.
        self.promote_dma = 0
        # Credits kept free for a restore
        # batch waiting to issue (credits never exceed P).
        self.promote_reserved = 0
        # Per waiting request: [total waited steps, last counted step, largest
        # allocation, lowest deficit of the current run of consecutive waits
        # (the previous progress value when unscoped), consecutive stalled
        # steps].
        self._waits: dict[str, list[int]] = {}
        # Dirty reclaims escaped admissions are about to make.
        self._escape_pending = 0
        # Request id -> its preemption count when an
        # escape was last counted (``demote_escapes`` once per escaping
        # admission, not per poll); dropped when it waits again or ends.
        self._escaped: dict[str, int] = {}
        # BlockTierState.counters.reclaimed_dirty already attributed.
        self._dirty_seen = 0
        # (block id, generation) already counted as dropped (host full).
        self._dropped: set[tuple[int, int]] = set()
        # (block id, generation) already counted as dropped (no credits).
        self._dropped_cr: set[tuple[int, int]] = set()
        # GPU members of a selected unit not copied
        # yet (out of credits, or not ready) are pinned here until their copy
        # is issued, so a running allocation cannot reclaim them first. Pinned
        # only while the free pool stays above the running headroom.
        self._held_gpu: dict[int, "KVCacheBlock"] = {}
        # (block id, generation) already counted as tiny (not copied).
        self._skipped: set[tuple[int, int]] = set()
        # (block id, generation) already counted as a too-large unit.
        self._too_large: set[tuple[int, int]] = set()
        # Per-step unit membership memo, cleared at each walk.
        self._members_memo: dict[int, list] = {}
        # Demotions issued or completed so far: the gate's progress signal.
        self._progress = 0
        # GPU block id -> (generation, keys the host held for it, walk
        # step) of its last published reactive / repair copy. A later copy of
        # that same content under no new key, landed in the head's waiting
        # episode, is a re-copy (its host copy was displaced while the source
        # stayed on the GPU): no progress for the idle breaker
        # (``_breaker_progress``).
        self._landed: dict[int, tuple[int, frozenset, int]] = {}
        # The current blocked head's deficit
        # episode. Per wait record (``_waits``, by identity): [record, walk
        # step first seen, walk step its current run of consecutive counted
        # waits began, its last counted wait step]. The episode starts at the
        # current head's run (``_ep_start``, None without a waiter): the
        # head is the record that counted a wait last, the oldest on a tie.
        # Other waiters' records coming and going, and stale records, never
        # anchor it (``_note_episode``).
        self._born: dict[str, list] = {}
        self._ep_start: int | None = None
        # The running headroom of the last walk (``_start_verdict`` in the
        # gate's mirror; 0 before any walk).
        self._last_headroom = 0
        # The host rows of the last ``select``'s picks, in pick order, each
        # held with one ref: the store transfer takes them as its
        # destinations (the manager allocates nothing).
        self.host_rows: list[KVCacheBlock] | None = None
        # Unit-wide reservation: ledger id -> (host rows held
        # with one ref each, {GPU member id: generation} still to issue), and
        # GPU member id -> ledger id. A started unit reserves a row for every
        # missing destination, unready members included; P bounds only the
        # copies issued per step.
        self._res: dict[int, tuple[list[KVCacheBlock], dict[int, int]]] = {}
        self._res_members: dict[int, int] = {}
        # The cap on charged holds (``_charged``; None: one
        # restore unit's worth, ``_hold_cap``), the waiting head's new-block
        # need (the latest admission since the last walk; the walk's copy in
        # ``_last_head_need``), and each reservation's last progress step.
        self.hold_cap: int | None = None
        self._cap_step = -1
        self._cap_now = 0
        self._head_need = 0
        self._last_head_need = 0
        self._res_progress: dict[int, int] = {}
        # Issued member -> its reservation key, so a landing counts as the
        # unit's progress too.
        self._res_issued: dict[int, int] = {}
        # Units started by a reservation, until published or lost.
        self._started: set[int] = set()
        # Unit id -> its last copy progress step (a
        # reservation, an issue or a finished copy of a member), kept apart
        # from the reservations: the sweep can drop a reservation whose unit
        # keeps assembling (``_release_stalled_units``).
        self._unit_progress: dict[int, int] = {}
        self._prune_step = 0

    # -- binding ------------------------------------------------------------
    def bind(self, gpu_pool: "BlockPool", units: "UnitTable | None") -> None:
        """Attach to the GPU pool's tier state (mounted by
        ``kv_host_tier_units.mount``) and the unit table."""
        assert gpu_pool.tier is not None, "mount the tier state first"
        self.gpu_pool = gpu_pool
        self.units = units
        self._dirty_seen = gpu_pool.tier.counters.reclaimed_dirty
        # cache_thrash: a yielded session's break demotes instead of
        # destroying when the session is host-backed (HostTierEviction).
        gpu_pool.evict_groups.host_tier = self
        if units is not None and self.copyq is None:
            from vllm.merlin.kv_host_tier_copyq import CopyQueue

            self.copyq = CopyQueue(self)
        if units is not None and self.copyq is not None:
            # Backing revocation: the unit table reports a host
            # twin it evicted while the GPU still holds the key, through
            # ``revoke_hook(gpu_blocks, step)`` when it defines one.
            if hasattr(units, "revoke_hook"):
                units.revoke_hook = self.copyq.revoke_backing
            # Duplicate classes (``UnitTable.plan``): "running" / "imminent" GPU
            # twins, from the copy scheduler (grace, waiter and tier pins).
            if hasattr(units, "twin_kind"):
                units.twin_kind = self.copyq.twin_kind

    def bind_manager(self, kv_cache_manager: Any) -> Any:
        """Give the copy scheduler the KVCacheManager (grace
        pins, SessionYield order, the coordinator) and return it for the
        manager's ``install_tier_hooks``. None without a unit table."""
        if self.copyq is None:
            return None
        self.copyq.bind_manager(kv_cache_manager)
        return self.copyq

    @property
    def tier(self) -> "BlockTierState":
        assert self.gpu_pool is not None and self.gpu_pool.tier is not None
        return self.gpu_pool.tier

    def _tiny(self, block: "KVCacheBlock") -> bool:
        """Whether every unit ``block`` belongs to has fewer than
        ``TINY_FA_PAGES`` full-attention pages (no unit: not tiny)."""
        if self.units is None:
            return False
        units = list(self.units.unit_of_block(block))
        return bool(units) and all(self.units.fa_pages(u.uid) < TINY_FA_PAGES for u in units)

    def _store_pins_fit(self, blocks, picked_ids=()) -> bool:
        """Count distinct store sources, including held, staged and issued.

        Count referenced sources too: their request can release
        them while the tier still owns a reference. Restore pins are separate.
        """
        cap = self.max_store_pins
        if not cap:
            return True
        # Bounded by the cap, not by the pin count: the union is built
        # source by source and the check fails as soon as it passes the
        # cap. Each source adds at most cap + 1 new ids before that and
        # repeats at most cap ids already seen, so one call is O(cap) however
        # many pins are held (the walk calls it per candidate block).
        seen: set[int] = set()

        def over(ids) -> bool:
            for bid in ids:
                if bid not in seen:
                    seen.add(bid)
                    if len(seen) > cap:
                        return True
            return False

        return not (
            over(self._held_gpu)
            or over(self.in_flight)
            or over(picked_ids)
            or over(t.block.block_id for t in self._staged)
            or over(b.block_id for b in blocks)
        )

    def _hold(
        self, blocks, gpu_free: int, headroom: int, pending: list,
        picked_ids=(),
    ) -> int:
        """Pin unit members that wait for a copy. A block already
        referenced costs no free block; a free one is pinned only while the
        free pool stays above the running headroom. Returns the free count.

        The pins go to ``pending`` and are applied (``touch``) after the walk:
        touching a free block unlinks it from the GPU free queue the walk is
        iterating (``iter_pool_reclaims``), which ended the walk there."""
        tier = self.tier
        pin = []
        floor = headroom + self._last_head_need
        cap = self._hold_cap()
        charged = self._charged()
        for b in blocks:
            if b.block_id in self._held_gpu or b.block_id in self.in_flight:
                continue
            if not tier.is_dirty(b) or not self._in_unit(b):
                continue
            if not self._store_pins_fit([b], picked_ids):
                continue
            if b.ref_cnt == 0:
                if gpu_free - 1 < headroom:
                    continue
                if gpu_free - 1 < floor or charged >= cap:
                    # The hold budget: never pin the blocks the
                    # waiting head needs, nor more than one unit's worth.
                    self.counters.demote_held_capped += 1
                    continue
                gpu_free -= 1
                charged += 1
            pin.append(b)
            self._held_gpu[b.block_id] = b
        if pin:
            pending.extend(pin)
            c = self.counters
            c.demote_held += len(pin)
            c.demote_held_peak = max(c.demote_held_peak, len(self._held_gpu))
        return gpu_free

    def _charged(self) -> int:
        """Held GPU blocks the hold budget charges: those whose only ref is
        the tier's pin (``ref_cnt == 1``), whether pinned free or held while
        referenced and since released by their request.
        In-flight copy sources are not holds (the DMA needs them) and are
        bounded by the credits instead."""
        return sum(
            1 for bid, b in self._held_gpu.items()
            if b.ref_cnt == 1 and bid not in self.in_flight
        )

    def _shed_holds(self, headroom: int) -> None:
        """Make the hold budget a real admission guarantee:
        re-measured every walk, while the charged holds exceed the
        cap or the free pool is below ``headroom`` plus the waiting head's
        need, release charged holds, the most recently started unit's
        not-yet-issued members first. A released member stays queued and
        reclaimable: a reclaim before its copy is a missed copy. Counted in
        ``demote_held_released``."""
        cap = self._hold_cap()
        floor = headroom + self._last_head_need
        charged = self._charged()
        gpu_free = self.gpu_pool.get_num_free_blocks()
        if charged <= cap and gpu_free >= floor:
            return
        order: list[int] = []
        seen: set[int] = set()
        for key in reversed(list(self._res)):
            for bid in reversed(list(self._res[key][1])):
                if bid not in seen:
                    seen.add(bid)
                    order.append(bid)
        for bid in reversed(list(self._held_gpu)):
            if bid not in seen:
                seen.add(bid)
                order.append(bid)
        shed = []
        for bid in order:
            if charged <= cap and gpu_free >= floor:
                break
            b = self._held_gpu.get(bid)
            if b is None or b.ref_cnt != 1 or bid in self.in_flight:
                continue
            shed.append(b)
            charged -= 1
            gpu_free += 1
        if shed:
            self.counters.demote_held_released += len(shed)
            self._release_held(shed)

    def _hold_cap(self) -> int:
        """Free GPU blocks the walker may hold at once: one restore unit's
        worth (the largest recorded unit, read once per walk step), or a
        third of the GPU pool before any unit is recorded."""
        if self.hold_cap is not None:
            return self.hold_cap
        if self._cap_step != self.step:
            self._cap_step = self.step
            largest = max(
                (len(u.required) for u in self.units.units.values()), default=0
            ) if self.units is not None else 0
            self._cap_now = largest or max(
                1, self.gpu_pool.num_gpu_blocks // (self.max_assembling + 1)
            )
        return self._cap_now

    def _hold_stall_steps(self) -> int:
        """Walker steps without copy progress after which a held unit's pins
        are released."""
        return 2 * max(1, self.escape_after)

    def _in_unit(self, block: "KVCacheBlock") -> bool:
        """``block`` still belongs to a recorded restore unit (always True
        without a unit table)."""
        return self.units is None or bool(self.units.unit_of_block(block))

    def _release_held(self, blocks) -> None:
        out = [b for b in blocks if self._held_gpu.pop(b.block_id, None) is not None]
        if out:
            self.gpu_pool.free_blocks(out)

    def _sweep_held(self, tier) -> None:
        """Drop holds whose block is no longer dirty (backed, superseded or
        unhashed), no longer in the backlog, or whose unit was lost: a lost
        unit's surviving members are no restore point, so copying them is
        waste and holding them pins GPU blocks for nothing."""
        if self.units is not None and self.backlog:
            self.backlog = [b for b in self.backlog if self._in_unit(b)]
        if not self._held_gpu:
            return
        queued = {b.block_id for b in self.backlog}
        stale = [
            b for bid, b in self._held_gpu.items()
            if bid not in queued or not tier.is_dirty(b) or not self._in_unit(b)
        ]
        self._release_held(stale)

    def _host_capacity(self) -> int:
        """Data rows of the host cache (the pool's null row excluded)."""
        return self.cpu_pool.num_gpu_blocks - 1

    def _unit_rows(self, units, gpu_blocks) -> int:
        """Host rows ``units`` occupy once complete: their rows already on
        the host plus each GPU member not backed there (ready or not, copies
        in flight included)."""
        tier = self.tier
        host: set[int] = set()
        for unit in units:
            host.update(b.block_id for b in self.units.host_members(unit))
        gpu = sum(
            1 for m in gpu_blocks
            if m.block_id in self.in_flight or tier.is_dirty(m)
        )
        return len(host) + gpu

    def _active_units(self) -> dict:
        """Units in progress (MAX_ASSEMBLING, protection, committed
        footprint): from the start of their reservation until they publish
        or are lost (a unit whose every member is issued but
        not yet acked stays in progress). Also any unit with rows landed and
        held, members in the backlog or reserved, or a copy in flight."""
        units = self.units
        out: dict = {}
        if units is None:
            return out
        for uid in units.assembling_uids():
            unit = units.units.get(uid)
            if unit is not None:
                out[uid] = unit
        for block in self.backlog:
            for unit in units.unit_of_block(block):
                out[unit.uid] = unit
        for bid in self._res_members:
            for unit in units.unit_of_block(self.gpu_pool.blocks[bid]):
                out[unit.uid] = unit
        for ticket in self.in_flight.values():
            for unit in units.unit_of_block(ticket.block):
                out[unit.uid] = unit
        # Started units leave only on publication or loss. One with no work
        # left of any kind (nothing reserved, queued, in flight or held:
        # completed by other copies) is dropped too, so it cannot pin the
        # host or an assembly slot forever.
        # A unit that gave up its start is not in progress.
        out = {uid: u for uid, u in out.items() if not units.is_abandoned(uid)}
        for uid in list(self._started):
            unit = units.units.get(uid)
            if unit is None or units.is_published(uid) or uid not in out:
                self._started.discard(uid)
        return out

    def _host_rows_of(self, units) -> set[int]:
        """Host row ids ``units`` depend on (an extending unit's keys include
        its predecessor's prefix, so its visible rows are among them)."""
        out: set[int] = set()
        for unit in units:
            out.update(b.block_id for b in self.units.host_members(unit))
        return out

    def _alloc_rows(self, n: int, guarded: bool = False) -> list["KVCacheBlock"]:
        """Allocate ``n`` host rows, exactly the idle ownerless rows the
        reservation counted (``front_rows``; ``guarded``: only rows a
        unit-less copy may take, ``_guarded_skip``), never an unselected
        unit's row; fewer when fewer are idle."""
        if n <= 0:
            return []
        if self.units is None:
            return self.cpu_pool.get_new_blocks(
                min(n, self.cpu_pool.get_num_free_blocks())
            )
        rows = self.units.front_rows(n, self._guarded_skip() if guarded else None)
        got = self.cpu_pool.get_new_blocks(len(rows)) if rows else []
        assert [b.block_id for b in got] == [b.block_id for b in rows]
        return got

    def _guarded_skip(self):
        """The un-keying guard as an ``idle_rows`` skip, judged in free-queue order as
        if every row it passed so far were already taken (two
        same-content copies each looked spare while the other existed, and
        one allocation took both). ``idle_rows`` calls it only for idle
        ownerless rows and keeps exactly the rows it passes, so the rows a
        scan returns may all be taken together. One scan per closure."""
        gone: set[int] = set()

        def skip(row: "KVCacheBlock") -> bool:
            if self._backs_free_twin(row, gone):
                return True
            gone.add(row.block_id)
            return False

        return skip

    def _guarded_rows(self, n: int) -> list["KVCacheBlock"]:
        """Up to ``n`` idle rows a unit-less copy may take (``_guarded_skip``),
        in the order ``_alloc_rows(n, guarded=True)`` takes them."""
        return self.units.idle_rows(n, self._guarded_skip())

    def _backs_free_twin(self, row: "KVCacheBlock", gone=frozenset()) -> bool:
        """Taking host ``row`` (with the rows ``gone`` taken already)
        would turn a reclaimable GPU block DIRTY again. The row is the only
        host copy of some key of a GPU block in the free queue (ref 0) that
        still holds the very content the row was copied from, and that block
        is CLEAN now by its host backing (backing is revalidated per query,
        so un-keying the row re-dirties it). A unit-less copy into such a row
        only moves a dirty block, never removes one: the next walk copies the
        twin back (the residue round robin).

        A twin that loses nothing is not protected: one already
        DIRTY (a key missing, e.g. a new alias: taking its own old row and
        copying it lands every key), or one CLEAN whatever the host holds
        (masked, or outside every retained window)."""
        if row.block_hash is None:
            return False
        tier = self.tier
        src = tier.host_src.get(row.block_id)
        if src is None:
            return False
        host = self.cpu_pool.cached_block_hash_to_block
        gpu = self.gpu_pool.cached_block_hash_to_block
        keys = [row.block_hash]
        keys.extend(self.cpu_pool.cached_block_hashes_by_block.get(row.block_id, ()))
        for key in keys:
            if any(
                h is not row and h.block_id not in gone
                and tier.host_src.get(h.block_id) == src
                for h in host.get_blocks(key)
            ):
                continue  # another host copy keeps the twin backed
            for twin in gpu.get_blocks(key):
                if (
                    twin.ref_cnt == 0 and tier.content_id(twin) == src
                    and self._backing_needed(twin, src, gone)
                ):
                    return True
        return False

    def _backing_needed(self, twin: "KVCacheBlock", cid: int, gone) -> bool:
        """``twin`` (content ``cid``) is CLEAN only because the host holds
        every key it carries in a copy of ``cid`` outside ``gone``
        (``BlockTierState.state`` without its counters): losing one of
        those copies would make it DIRTY."""
        tier = self.tier
        if not tier.is_eligible(twin) or tier.is_masked(twin):
            return False  # CLEAN whatever the host holds
        if (
            tier.retains_fn is not None
            and not tier.retains_fn(twin)
            and not tier._host_holds_foreign(twin)
        ):
            return False  # no restore reads it: CLEAN regardless
        host = self.cpu_pool.cached_block_hash_to_block
        src = tier.host_src
        return all(
            any(
                h.block_id not in gone and src.get(h.block_id) == cid
                for h in host.get_blocks(key)
            )
            for key in self._gpu_keys(twin)
        )

    def _reserve(self, key: int, members, rows) -> None:
        self._res_progress[key] = self.step
        self._note_unit_progress(members)
        held, todo = self._res.setdefault(key, ([], {}))
        held.extend(rows)
        gen = self.tier.generation
        for m in members:
            todo[m.block_id] = gen[m.block_id]
            self._res_members[m.block_id] = key
        self._note_reserved()

    def _note_reserved(self) -> None:
        c = self.counters
        c.reserved_rows = sum(len(rows) for rows, _ in self._res.values())
        c.reserved_rows_peak = max(c.reserved_rows_peak, c.reserved_rows)

    def _note_unit_progress(self, blocks) -> None:
        """Copy progress of the units ``blocks`` belong to."""
        if self.units is None:
            return
        for b in blocks:
            for unit in self.units.unit_of_block(b):
                self._unit_progress[unit.uid] = self.step

    def _draw_row(self, bid: int) -> "KVCacheBlock":
        """The reserved row of ``bid``'s unit that its copy issues into."""
        key = self._res_members.pop(bid)
        rows, todo = self._res[key]
        todo.pop(bid, None)
        row = rows.pop()
        self._res_progress[key] = self.step  # a copy issued: progress
        self._note_unit_progress((self.gpu_pool.blocks[bid],))
        self._res_issued[bid] = key
        if not rows and not todo:
            del self._res[key]
            self._res_progress.pop(key, None)
        return row

    def _sweep_reserved(self, tier) -> None:
        """Release reserved rows whose member no longer needs a copy: its
        content changed (generation), it is backed or clean, or its unit was
        lost, or it left the backlog (no copy will draw it). A unit lost or
        fully issued holds no row."""
        released: list[KVCacheBlock] = []
        queued = {b.block_id for b in self.backlog}
        for key in list(self._res):
            rows, todo = self._res[key]
            for bid, gen in list(todo.items()):
                b = self.gpu_pool.blocks[bid]
                if (
                    bid not in queued
                    or tier.generation[bid] != gen or not tier.is_dirty(b)
                    or not self._in_unit(b)
                ):
                    del todo[bid]
                    self._res_members.pop(bid, None)
            extra = len(rows) - len(todo)
            if extra > 0:
                released.extend(rows[len(todo):])
                del rows[len(todo):]
            if not rows and not todo:
                del self._res[key]
                self._res_progress.pop(key, None)
        if released:
            self.cpu_pool.free_blocks(released)
        self._note_reserved()

    def _release_stalled(self) -> None:
        """Liveness: a started unit whose copies made no
        progress (nothing issued or finished for it) for
        ``_hold_stall_steps`` walker steps, and that has NO copy in flight
        (the pipe stalled for it: credits never returned), gives up its
        start: its pins and reserved rows are released,
        its queued members leave the backlog, its landed rows are released
        (an abandoned fragment), and it is not restarted while a copy of it
        is still in flight. A reclaim of its members is a missed copy."""
        limit = self._hold_stall_steps()
        stalled = [
            key for key in self._res
            if self.step - self._res_progress.get(key, self.step) >= limit
            and not self._key_in_flight(key)
        ]
        for key in stalled:
            rows, todo = self._res.pop(key)
            self._res_progress.pop(key, None)
            for bid in [b for b, k in self._res_issued.items() if k == key]:
                del self._res_issued[bid]
            members = set(todo)
            for bid in members:
                self._res_members.pop(bid, None)
            blocks = [self.gpu_pool.blocks[bid] for bid in members]
            uids = {
                u.uid for b in blocks for u in self.units.unit_of_block(b)
            } if self.units is not None else set()
            held = [b for b in blocks if b.block_id in self._held_gpu]
            self.counters.demote_held_released += len(held)
            self._release_held(held)
            self.backlog = [b for b in self.backlog if b.block_id not in members]
            if rows:
                self.cpu_pool.free_blocks(rows)
            for uid in uids:
                self.units.abandon(uid)
                self._started.discard(uid)
        if stalled:
            self._note_reserved()
        if self.units is not None:
            self._release_stalled_units(limit)

    def _release_stalled_units(self, limit: int) -> None:
        """The per-unit half of the stall
        release. A unit assembling (holding landed, unpublished rows) that
        has no reserved member (a reservation's own stall release covers
        those) and no member copy in flight, and made no copy progress for
        ``limit`` walker steps, gives up its start as a stalled reservation
        does: its landed rows are released (handed to a co-owner still
        assembling, else an abandoned fragment), its queued members leave the
        backlog unpinned, and the walker restarts it when it meets it again.
        Reached when ``_sweep_reserved`` drops a reservation whose unit
        survives: a CoW hash move (``BlockPool.move_block_hashes``) carried
        its last reserved member's key to a block the running request keeps,
        outside the reactive window. The timeout frees its staged rows and its
        MAX_ASSEMBLING slot while that block is held."""
        units = self.units
        live = set(units.assembling_uids())
        for uid, since in list(self._unit_progress.items()):
            if (
                uid not in units.units or units.is_published(uid)
                or units.is_abandoned(uid)
                or (uid not in live and self.step - since >= limit)
            ):
                del self._unit_progress[uid]
        # Only a unit past the limit is judged (the common walk stops here).
        due = sorted(
            uid for uid in live
            if self.step - self._unit_progress.setdefault(uid, self.step) >= limit
        )
        if not due:
            return
        blocks = self.gpu_pool.blocks
        busy = {
            u.uid for bid in self._res_members
            for u in units.unit_of_block(blocks[bid])
        }
        busy.update(
            u.uid for t in self.in_flight.values()
            for u in units.unit_of_block(t.block)
        )
        released: list[KVCacheBlock] = []
        dropped: set[int] = set()  # their queued members, filtered once
        for uid in due:
            unit = units.units.get(uid)
            if (
                uid in busy or unit is None
                or units.is_published(uid) or units.is_abandoned(uid)
            ):
                continue
            members = units.gpu_members(unit)
            if any(m.block_id in self.in_flight for m in members):
                continue
            dropped.update(m.block_id for m in members)
            held = [m for m in members if m.block_id in self._held_gpu]
            self.counters.demote_held_released += len(held)
            released.extend(held)
            units.abandon(uid)
            self._started.discard(uid)
            del self._unit_progress[uid]
        if dropped:
            self.backlog = [b for b in self.backlog if b.block_id not in dropped]
        self._release_held(released)

    def _key_in_flight(self, key: int) -> bool:
        """A copy of reservation ``key``'s unit is in flight: one it issued,
        or any GPU member of the units its queued members belong to."""
        if any(
            k == key and bid in self.in_flight
            for bid, k in self._res_issued.items()
        ):
            return True
        if self.units is None or not self.in_flight:
            return False
        _, todo = self._res[key]
        seen: set = set()
        for bid in todo:
            for u in self.units.unit_of_block(self.gpu_pool.blocks[bid]):
                if u.uid in seen:
                    continue
                seen.add(u.uid)
                if any(
                    m.block_id in self.in_flight
                    for m in self.units.gpu_members(u)
                ):
                    return True
        return False

    def _release_reserved_all(self) -> None:
        rows = [r for held, _ in self._res.values() for r in held]
        self._res.clear()
        self._res_members.clear()
        self._res_progress.clear()
        self._res_issued.clear()
        if rows:
            self.cpu_pool.free_blocks(rows)
        self._note_reserved()

    def _note_skip(self, block: "KVCacheBlock", cause: str) -> None:
        if self.units is not None:
            for unit in self.units.unit_of_block(block):
                self.units.note_skip(unit, cause)

    def _unit_members(self, block: "KVCacheBlock") -> list["KVCacheBlock"]:
        if self.units is None:
            return [block]
        memo = self._members_memo.get(block.block_id)
        if memo is not None:
            return memo
        out = self._unit_members_uncached(block)
        self._members_memo[block.block_id] = out
        return out

    def _unit_members_uncached(self, block: "KVCacheBlock") -> list["KVCacheBlock"]:
        out: list[KVCacheBlock] = []
        seen: set[int] = set()
        for unit in self.units.unit_of_block(block):
            for member in self.units.gpu_members(unit):
                if member.block_id not in seen:
                    seen.add(member.block_id)
                    out.append(member)
        return out or [block]

    def _protected(self, block: "KVCacheBlock") -> bool:
        """A reclaim of ``block`` destroys a restore point the walker could
        save (ready, eligible, not backed)."""
        tier = self.tier
        return tier.is_dirty(block) and tier.is_ready(block)

    # -- accounts ------------------------------------------
    def account_running(
        self,
        coordinator: Any,
        empty_blocks: Sequence[Sequence[Any]],
        running: Sequence["Request"],
        num_scheduled: "dict[str, int]",
        *,
        token_budget: int,
        max_model_len: int,
        num_spec_tokens: int = 0,
        num_lookahead_tokens: int = 0,
        sampled_per_step: int = 1,
        long_prefill_threshold: int = 0,
        input_budget: int | None = None,
        draft_slots: int = 0,
        kv_holders: Sequence["Request"] = (),
    ) -> int:
        """Exact ``H_run``: the blocks the running requests allocate in the
        NEXT scheduler step, in running order, from the allocator's own
        accounting (``coordinator.get_num_blocks_to_allocate``), taken after
        this step's allocation is in the block table and before
        ``_update_after_schedule`` advances the requests.

        Per request: ``projected`` = computed + scheduled now (tokens whose
        slots exist); the known length grows by the one token this step
        samples when it reaches the end of the known tokens; the next chunk
        is that length (+ drafts) less ``projected``, capped as the scheduler
        caps it (long-prefill threshold, input budget, model length) and by
        the step token budget shared in running order. Per-request
        alignment, spec/lookahead slots, CoW and recurrent rotation come from
        the group managers. It over-estimates only where the scheduler would
        shrink a chunk (mamba block-aligned split, encoder inputs) or drafts
        are rejected; it never under-estimates.
        """
        # Lane 1 (RUNNING) reads the running requests and
        # the coordinator of the step just scheduled (the ``running_ready``
        # input; the copy scheduler polls them at the next walk).
        self._running_reqs = tuple(running)
        self._coordinator = coordinator
        # This step's write frontiers, before any early return: the
        # running requests' and the KV-holding waiters' (a parked
        # streaming session resumes writing its own table in place). Lane 1
        # and H_run read ``running`` only.
        self.note_frontiers(coordinator, (*running, *kv_holders))
        # Exact H_run only matters when the free pool is near the bound: with
        # the free blocks above the conservative estimate plus every pin the
        # walker could make this step, the estimate and the exact value stop
        # nothing differently.
        if self.gpu_pool is not None:
            bound = self.reserve + len(running) * self.alignment_groups
            if (
                self.gpu_pool.get_num_free_blocks()
                >= bound + max(self.credits_d, 0) + 1
            ):
                self.counters.h_run_skipped += 1
                self.h_run = None
                self.headroom_now = bound
                return bound
        budget = token_budget
        # The NEXT step's budgets, shared across requests in running order as
        # the scheduler shares them: callers pass the full per-step
        # limits, never the remainder of the step that just finished
        # scheduling (that is 0 after a long prefill).
        inputs = input_budget
        total = 0
        for req in running:
            if budget <= 0 or (inputs is not None and inputs <= draft_slots):
                break
            scheduled = num_scheduled.get(req.request_id, 0)
            projected = req.num_computed_tokens + scheduled
            known = req.num_tokens + req.num_output_placeholders
            samples = int(
                sampled_per_step > 0 and scheduled > 0 and projected >= known
            )
            want = (
                known
                + samples * sampled_per_step
                + (num_spec_tokens if samples else 0)
                - projected
            )
            if 0 < long_prefill_threshold < want:
                want = long_prefill_threshold
            want = min(want, budget)
            if inputs is not None:
                want = min(want, inputs - draft_slots)
            want = min(want, max_model_len - projected - sampled_per_step)
            if want <= 0:
                continue
            budget -= want
            if inputs is not None:
                inputs -= want
            main = projected + want
            total += coordinator.get_num_blocks_to_allocate(
                request_id=req.request_id,
                num_tokens=min(main + num_lookahead_tokens, max_model_len),
                new_computed_blocks=empty_blocks,
                num_encoder_tokens=0,
                total_computed_tokens=projected,
                num_local_computed_tokens=projected,
                num_tokens_main_model=main,
                prefill_end=max(req.num_prompt_tokens, req.num_tokens - 1),
            )
        self.h_run = total
        self.headroom_now = total
        return total

    def note_frontiers(self, coordinator: Any, running: Sequence["Request"]) -> None:
        """The write frontiers of ``running`` (the running
        requests and the KV-holding waiters, which may resume writing their
        own tables), replaced every scheduler step
        (``account_running``), so a finished request, or a preempted one that
        gave up its table, leaves none behind. Per request and KV group: every
        non-null block from the first position not yet written on every rank
        (``num_computed_tokens`` less the in-flight tokens of an async step)
        on, and always the last non-null block (a recurrent state written in
        place). These are never copy sources (``BlockTierState.is_ready``):
        GRACE / PREEMPT offers and the reactive unit expansion inherit lane
        1's rule, and a unit's key resolves to a twin off the frontier
        (``UnitTable._gpu_block``). The set is small (the tail page, plus
        lookahead / speculative slots, per group and running request).

        A frontier needs its writer's reference: a block table always holds
        one, so a frontier member is referenced, outside the free queue and
        never a victim; it costs a waiting head nothing and starves no
        escape. ``BlockTierState.is_frontier`` (on the set AND referenced) is
        the one predicate for the copy gate (``is_ready``), the unit's twin
        pick (``UnitTable._gpu_block``) and this waiver. A unit whose only
        not-ready members are such referenced frontier members counts as
        whole for the head-floor waiver (``_issues_whole``): its
        free members issue now and the frontier member waits in the backlog
        behind a referenced (uncharged) hold until its writer moves on or the
        stall release gives the start up. A block released after the frontier
        was taken (a writer that finished without a grace; the admission gate
        runs before ``account_running`` rebuilds the set) is ref 0, no longer
        written and ready (``BlockTierState.is_ready``), so its unit keeps its
        gate protection for that step, and its key keeps
        resolving to it, not to a twin.

        Block tables are positional: index = position //
        block_size, with null blocks where a group dropped one (sliding
        window, Mamba align). Every keyed manager keeps that layout; a
        compacting manager would need its own frontier."""
        if self.gpu_pool is None or self.gpu_pool.tier is None:
            return
        live: set[int] = set()
        managers = getattr(coordinator, "single_type_managers", ())
        sizes = []
        for m in managers:
            # Tokens per table entry (0 = unknown: the whole table is live).
            sizes.append(getattr(m, "block_size", None) or getattr(
                getattr(m, "kv_cache_spec", None), "block_size", 0
            ) or 0)
        for req in running:
            written = max(
                0,
                int(getattr(req, "num_computed_tokens", 0) or 0)
                - int(getattr(req, "num_in_flight_tokens", 0) or 0),
            )
            for g, blocks in enumerate(coordinator.get_blocks(req.request_id)):
                last = len(blocks) - 1
                while last >= 0 and blocks[last].is_null:
                    last -= 1
                if last < 0:
                    continue
                size = sizes[g] if g < len(sizes) else 0
                # A group of unknown page size: all of its blocks.
                start = written // size if size else 0
                for b in blocks[min(start, last):]:
                    if not b.is_null:
                        live.add(b.block_id)
        self.gpu_pool.tier.live_ids = frozenset(live)

    def account_quarantine(self, abandoned_loads: int = 0) -> None:
        """Quarantined storage: copies of an older epoch still in flight plus
        abandoned loads' destinations; released as each completes."""
        stale = sum(
            len(t.tickets) for t in self.txns.values() if t.epoch != self.epoch
        )
        c = self.counters
        c.quarantine = stale + abandoned_loads
        c.quarantine_peak = max(c.quarantine_peak, c.quarantine)

    # -- walker -------------------------------------------------------------
    def select(self, step: int, num_running: int) -> list["KVCacheBlock"]:
        """Blocks to demote this step (touched and marked demoting through
        ``begin_demote``). The caller allocates their host blocks right after
        and must call ``issue`` with the store event."""
        t0 = time.perf_counter()
        self._note_episode(step)
        picked = self._select(step, num_running)
        self.counters.select_us += int((time.perf_counter() - t0) * 1e6)
        # Only copies that can free a victim for a waiter
        # (reactive and repair: sources in the free queue) are a progress
        # source for the gate. A grace / preempt / lane-1 copy's source is
        # referenced and stays so: with nothing running it frees nothing, and
        # counting it held a waiter for as long as grace copies ran (a
        # replayed wait of 9 steps > IDLE_BREAK_STEPS + 1). Nor is a re-copy
        # of content already landed once (``_breaker_progress``).
        if (
            num_running == 0
            and not any(self._breaker_progress(b.block_id) for b in picked)
            and not any(self._breaker_progress(bid) for bid in self.in_flight)
            and not self.promote_dma
        ):
            self.idle_steps += 1
        else:
            self.idle_steps = 0
        return picked

    def _breaker_progress(self, block_id: int) -> bool:
        """A copy of ``block_id`` keeps the idle breaker from firing: it is
        gate-relevant (``_gate_relevant``) and not a re-copy. A re-copy is a
        copy of content whose earlier reactive / repair copy already landed and
        was displaced while the source stayed on the GPU: it frees nothing the
        head can use that the first landing did not, so it is not progress.
        Only a landing in the head's waiting episode counts
        (``_note_episode``), and only when it covered every key the block
        carries now (a copy that lands a new alias publishes a prefix the
        host never had: progress)."""
        if not self._gate_relevant(block_id):
            return False
        rec = self._landed.get(block_id)
        if rec is None or rec[0] != self.tier.generation[block_id]:
            return True
        start = self._ep_start
        if start is None or rec[2] < start:
            return True  # landed before this head waited
        return not self._gpu_keys(self.gpu_pool.blocks[block_id]) <= rec[1]

    def _note_episode(self, step: int) -> None:
        """The current blocked head's deficit episode, for the
        breaker's re-copy history. The head is the record that counted a
        demotion wait most recently (admission stops at the first request
        that cannot go), the oldest record on a tie: another waiter polled in
        the same pass, or coming and going, never takes it over.
        A stale record (a request no longer reached) keeps only its
        backstop: it stops counting waits, so it is not the head and never
        anchors another head's history (a stale oldest record
        made a new head's repair a re-copy). The episode starts with the
        head's current run of consecutive counted waits; the real-shortfall
        pass-through continues a run (the retry waits in the same pass), a
        gap of passes starts a new one (``_wait``'s run rule).

        Known limit (accepted): two waiters that
        count waits on alternate passes each restart their run every time, so
        every re-copy is progress and the re-copy breaker never fires for
        them. Each is still released by the total-wait backstop
        (``escape_after * batches * 2`` counted waits), no worse than the
        base; FCFS admission rarely alternates heads."""
        born = self._born
        waits = self._waits
        for rid, rec in waits.items():
            seen = born.get(rid)
            if seen is None or seen[0] is not rec:
                born[rid] = [rec, step, step, rec[1]]
            elif rec[1] != seen[3]:
                if rec[1] - seen[3] > 1:
                    seen[2] = step  # not consecutive: a new deficit episode
                seen[3] = rec[1]
        if len(born) > len(waits):
            for rid in [r for r in born if r not in waits]:
                del born[rid]
        head = None
        for seen in born.values():
            if step - seen[3] > 1:
                # It counted no wait on this pass or the one
                # before (``_waiting_now``'s rule): not the current blocked
                # head, even as the only record. No head: no re-copy history.
                continue
            if head is None or (seen[3], -seen[1]) > (head[3], -head[1]):
                head = seen
        self._ep_start = None if head is None else head[2]

    def _gpu_keys(self, block: "KVCacheBlock") -> set:
        """Every key GPU ``block`` carries now (primary and aliases)."""
        keys = set(self.gpu_pool.cached_block_hashes_by_block.get(block.block_id, ()))
        if block.block_hash is not None:
            keys.add(block.block_hash)
        return keys

    def _landed_keys(self, block: "KVCacheBlock", cid: int) -> frozenset:
        """The keys of GPU ``block`` the host holds in a copy of content
        ``cid`` (what its landing published, plus any older same-content
        copy's keys)."""
        host = self.cpu_pool.cached_block_hash_to_block
        src = self.tier.host_src
        return frozenset(
            k for k in self._gpu_keys(block)
            if any(src.get(h.block_id) == cid for h in host.get_blocks(k))
        )

    def _gate_relevant(self, block_id: int) -> bool:
        """A copy of ``block_id`` is gate progress: reactive (no copy-queue
        offer) or a REPAIR copy (its source is free, ref 0)."""
        offer = self._offer_of.get(block_id)
        if offer is None:
            return True
        from vllm.merlin.kv_host_tier_iface import CopyLane

        return offer.lane is CopyLane.REPAIR

    def gate_idle(self) -> bool:
        """No progress source for a demotion wait: nothing ran, the
        walker selected nothing and nothing is in flight for IDLE_BREAK_STEPS
        steps. Waiting then only holds the waiters' pins against each other
        (seen as Running 0, Waiting 18, GPU KV 93-96 % held for an hour)."""
        return self.idle_steps >= IDLE_BREAK_STEPS

    def _select(self, step: int, num_running: int) -> list["KVCacheBlock"]:
        self.host_rows = None
        if self.gpu_pool is None:
            return []
        tier = self.tier
        self.account_reclaims()
        self.step = step
        if self.units is not None:
            self.units.step = step
        self._members_memo.clear()
        self._prune_memos(tier)
        # Lookahead: the reserve (the step's reuse) plus per-request edges
        # plus the largest waiting admission. Headroom is
        # what running requests may allocate this step: none without them.
        reach = self.reserve + num_running * self.alignment_groups
        exact, self.h_run = self.h_run, None
        self.headroom_now = None
        if exact is None:
            headroom = reach if num_running > 0 else 0
        else:
            # Exact H_run (``account_running``), not the estimate.
            headroom = exact
            reach = max(reach, exact)
            self.counters.h_run = exact
            self.counters.h_run_peak = max(self.counters.h_run_peak, exact)
        window = reach + self.demand_hint
        demand = self.demand_hint
        self.demand_hint = 0
        # The admission gate judges starts against this walk's headroom.
        self._last_headroom = headroom
        # The waiting head's need since the last walk (the hold budget).
        self._last_head_need, self._head_need = self._head_need, 0

        # The demotion direction's own credits (rows, and
        # bytes per rank when ``bytes_d`` is set); restore DMA does not
        # share them.
        budget = self._d_budget()
        gpu_free = self.gpu_pool.get_num_free_blocks()
        picked: list[KVCacheBlock] = []
        picked_ids: set[int] = set()
        c = self.counters
        # Host rows. A started unit's members draw from its
        # reservation (``_res``). A lone victim's row is counted here
        # (``host_used``) and allocated after the walk; ``host_ok`` idle
        # ownerless rows are known to exist.
        host_used = 0
        host_ok = 0
        capped = False  # ``demote_copy_capped`` latch: once per walk
        # Unit-less picks this walk, the idle rows they may take (none
        # that backs a free GPU twin, ``_backs_free_twin``) known to exist,
        # and the count a scan last fell short of (until the host changes:
        # a plan, or a unit reservation taking rows, resets both).
        fresh_used = 0
        fresh_ok = 0
        fresh_short: int | None = None
        fresh_ids: set[int] = set()

        def room(
            n: int, evict: bool = True, src=None, units=(),
            displace: bool = True,
        ) -> bool:
            """Idle host rows for ``n`` more rows this step: ownerless rows,
            else (``evict``) the rows an ``UnitTable.plan`` frees, applied
            only when it meets the need (idle -> running-twin
            duplicates -> LRU -> imminent twins; exclusive rows only).
            ``units``: the unit ids this room is for (a constrained successor
            may take its own kept predecessor). ``displace``: whether the
            plan may take published host-only content (copy-queue lanes map
            ``may_displace`` here; False = idle rows and running-twin
            duplicates only).

            ``src``: a unit-less block (an intermediate checkpoint, a lookup-
            visible boundary of its own). Rule A, "complete units
            outrank fragments": its copy takes idle ownerless rows or evicts
            unpublished components by LRU, never a
            component with a published owner; otherwise it is dropped
            (host-full). The trade-off: an intermediate boundary may
            recompute on a host full of complete units; output identical."""
            nonlocal host_ok, fresh_ok, fresh_short
            c.room_calls += 1
            want = host_used + n
            if want <= host_ok:
                return True
            if self.units is None:
                host_ok = self.cpu_pool.get_num_free_blocks()
                return want <= host_ok
            if not evict:
                host_ok = self.units.idle_count(want)
                return want <= host_ok
            prev = None
            if src is not None:
                # Rule A: a unit-less copy never evicts a component
                # with a published owner.
                prev = self.units.lone_victims(True)
            try:
                # The idle check counts rows; it does not walk the queue.
                have = self.units.idle_count(want)
                if have < want:
                    plan = self.units.plan(
                        want, self.step, displace_published=displace,
                        for_units=units,
                    )
                    if plan.satisfied and not plan.empty:
                        self.units.apply(plan)
                        # The plan freed rows and un-keyed others: rescan.
                        fresh_ok, fresh_short = 0, None
                    have = self.units.idle_count(want)
            finally:
                if prev is not None:
                    self.units.lone_victims(prev)
            host_ok = have
            return want <= have

        def fresh_room() -> bool:
            """An idle row for one more unit-less copy that does not
            back a free GPU twin. Taking such a row only moves a dirty block
            (un-keying it re-dirties the twin, which the next walk copies
            back): with more dirty unit-less blocks than idle rows, the walker
            re-copied the same residue every step for as long as nothing ran.
            Without such a row the copy is not issued (a lone victim is
            reclaimed, never waited for), as on a full host. The admission
            mirror makes the same decision (``_unitless_row_ok``).

            One scan counts the rows for every unit-less pick
            the credits still allow, so a walk scans the guarded queue prefix
            once (O(S + P) guard evaluations for S guarded rows ahead of P
            picks), again only after the host changed under it (a plan or a
            unit reservation), not once per pick (O(P*S + P^2))."""
            nonlocal fresh_ok, fresh_short
            want = fresh_used + 1
            if want <= fresh_ok:
                return True
            if fresh_short is not None and want >= fresh_short:
                return False
            look = fresh_used + max(1, budget)
            fresh_ok = len(self._guarded_rows(look))
            if fresh_ok < look:
                fresh_short = fresh_ok + 1  # every such row was counted
            return want <= fresh_ok

        def take(block: "KVCacheBlock", lone: bool = True) -> bool:
            """Pick ``block`` if it needs and may get a copy; False stops the
            walk (out of credits, of running headroom, or of host rows for an
            unreserved unit member)."""
            nonlocal budget, host_used, gpu_free, capped, fresh_used
            bid = block.block_id
            if bid in picked_ids or bid in self.in_flight:
                return True
            if tier.state(block) is not BlockState.DIRTY:
                return True
            if not tier.is_ready(block):
                c.demote_not_ready += 1
                return True
            if budget <= 0 or not self._store_pins_fit([block], picked_ids):
                return False
            reserved = bid in self._res_members
            unitless = self.units is not None and not self.units.unit_of_block(block)
            if not reserved and (
                not room(1, src=block if unitless else None)
                or (unitless and not fresh_room())
            ):
                if not lone:
                    return False  # a unit member waits for its row
                # No host row is free or evictable (leases, in-flight
                # copies): a lone victim is reclaimed, never waited for.
                ident = (bid, tier.generation[bid])
                if ident not in self._dropped:
                    self._dropped.add(ident)
                    c.demote_dropped_host_full += 1
                return True
            if block.ref_cnt == 0 and bid not in self._held_gpu:
                # The pin takes the block out of the free queue: never below
                # what running requests allocate this step. A held member was
                # charged when its hold was decided.
                if gpu_free - 1 < headroom:
                    return False
                if bid not in whole and (
                    gpu_free - 1 < headroom + self._last_head_need
                ):
                    # A new source pin obeys
                    # the hold budget's floor too, or rolling one-row copies
                    # re-take every block the waiting head needs. A started
                    # unit's member just waits (the stall release bounds it).
                    # A non-atomic member loop may stop here once per offer;
                    # the walk counts one stop. Not for a
                    # unit started whole (``whole``), whose pins all issue now.
                    if not capped:
                        capped = True
                        c.demote_copy_capped += 1
                    return False
                gpu_free -= 1
            picked.append(block)
            picked_ids.add(bid)
            budget -= 1
            if not reserved:
                host_used += 1
                if unitless:
                    fresh_used += 1
                    fresh_ids.add(bid)
            return True

        def copies(order: list["KVCacheBlock"]) -> int:
            return sum(
                1
                for m in order
                if m.block_id not in picked_ids
                and m.block_id not in self.in_flight
                and tier.is_dirty(m)
                and tier.is_ready(m)
            )

        stopped = False
        step_seen: set[int] = set()
        keep: list[KVCacheBlock] = []
        # Holds decided during the walk, pinned once it is over (``_hold``).
        pins: list[KVCacheBlock] = []
        # Members of units started whole this walk (``_issues_whole``):
        # their source pins are not charged the waiting head's need.
        whole: set[int] = set()
        # The backlog is the rest of a unit already admitted: it goes first.
        self._sweep_held(tier)
        self._sweep_reserved(tier)
        self._release_stalled()
        self._shed_holds(headroom)
        self._sweep_offers()
        if self.copyq is not None:
            # Preempt-grace expiry / floor shedding runs every walk, credits
            # or not.
            self.copyq.sweep(step)
        # The host backing of every unit in progress is never
        # an eviction victim (kept until it publishes or is lost: it stays in
        # progress until then), so a reservation cannot evict the rows a
        # started unit depends on.
        protected: set[int] = set()
        if self.units is not None:
            protected = self._host_rows_of(self._active_units().values())
            self.units.protect(protected)
        backlog = self.backlog
        # Reserved members need no new row; one reservation for the rest.
        unreserved = [b for b in backlog if b.block_id not in self._res_members]
        if budget > 0 and unreserved:
            room(min(budget, copies(unreserved)), evict=False)
        for i, block in enumerate(backlog):
            lone = block.block_id not in self._res_members and not self._in_unit(block)
            if not take(block, lone=lone):
                keep = backlog[i:]
                stopped = True
                break
        # Members still waiting to be ready stay (held) for the next step.
        waiting = [
            b for b in backlog[: len(backlog) - len(keep)]
            if (b.block_id in self._held_gpu or b.block_id in self._res_members)
            and b.block_id not in picked_ids
            and tier.is_dirty(b) and not tier.is_ready(b)
        ]
        self.backlog = waiting + keep

        # Units in progress and the host rows they take once complete: a new
        # unit starts only beside fewer than MAX_ASSEMBLING of them and when
        # all of them fit the host cache together (exact with the unit-wide
        # reservation: a started unit owns its rows).
        active = self._active_units()
        committed = sum(
            self._unit_rows([u], self.units.gpu_members(u)) for u in active.values()
        )
        if self.units is not None:
            protected |= self._host_rows_of(active.values())
            self.units.protect(protected)

        def drop_credits(blocks) -> None:
            for m in blocks:
                ident = (m.block_id, tier.generation[m.block_id])
                if tier.is_dirty(m) and ident not in self._dropped_cr:
                    self._dropped_cr.add(ident)
                    c.demote_dropped_credits += 1

        if not stopped and window > 0:
            # Unready victims do not use up the window: reach
            # past them, by at most P more victims.
            limit = window
            cap = window + max(self.credits_d, 0)
            seen = 0

            def victims():
                """Window victims, each followed by the session peers its
                pop's destructive break would un-hash: they are
                destroyed by the same pop, so they are victims too."""
                nonlocal seen
                for victim, _keyed, destroyed in iter_pool_reclaims(
                    tier.pool, cap
                ):
                    if seen >= limit:
                        return
                    seen += 1
                    yield victim, True
                    yield from ((p, False) for p in destroyed)

            for node, primary in victims():
                if stopped:
                    break
                if node.block_id in picked_ids or not tier.is_dirty(node):
                    continue
                if node.block_id in step_seen:
                    continue  # its unit was already judged this step
                if primary and not tier.is_ready(node):
                    limit = min(limit + 1, cap)
                members = self._unit_members(node)
                # The met member first, then the rest of its unit.
                order = [node] + [m for m in members if m is not node]
                if len(order) > 1:
                    step_seen.update(m.block_id for m in order)
                if self._tiny(node):
                    # Too small to be worth a restore: the next eviction.
                    self._note_skip(node, "skipped_tiny")
                    for m in order:
                        ident = (m.block_id, tier.generation[m.block_id])
                        if tier.is_dirty(m) and ident not in self._skipped:
                            self._skipped.add(ident)
                            c.demote_skipped_tiny += 1
                    continue
                need = copies(order)
                lone = len(order) == 1 or self.units is None
                if lone:
                    # A unit's only GPU member evicts for its row like any
                    # unit; a unit-less block's copy never evicts a
                    # published unit (rule A: ``room(src=...)`` in ``take``).
                    node_uids = [
                        u.uid for u in self.units.unit_of_block(node)
                    ] if self.units is not None else []
                    if need and node_uids:
                        # Make its row now, for its own unit (``take`` finds
                        # it idle; a full host is counted there).
                        room(need, units=node_uids)
                    # A full host: ``take`` counts it host-full dropped.
                    if not take(node):
                        if budget <= 0 and tier.is_ready(node):
                            drop_credits([node])
                        stopped = True
                    continue
                verdict, start = self._start_verdict(
                    node, order, need, gpu_free=gpu_free, headroom=headroom,
                    budget=budget, active=active, committed=committed,
                    head_need=self._last_head_need,
                    picked_ids=picked_ids,
                )
                node_units, rows, key, owned, fresh, is_whole = start
                if verdict == "never_fit":
                    # Never start a unit the whole host cache cannot hold
                    # (against the cache, not today's free rows).
                    ident = (node.block_id, tier.generation[node.block_id])
                    if ident not in self._too_large:
                        self._too_large.add(ident)
                        c.demote_unit_too_large += 1
                    self._note_skip(node, "never_fit")
                    continue
                if verdict == "credits":
                    # Every credit is taken: not started (no rows reserved,
                    # nothing pinned); a running allocation may reclaim it.
                    drop_credits(order)
                    continue
                if verdict == "not_ready":
                    # Nothing copyable yet: not started, judged again when
                    # met (its unready members are counted).
                    for m in order:
                        take(m, lone=False)
                    continue
                if verdict == "assembly":
                    # Finish the units in progress first.
                    c.demote_assembly_waits += 1
                    continue
                if verdict in ("held_capped", "stalled"):
                    # The hold budget leaves the waiting head its blocks
                    # (the hold cap), or the unit gave up its start and a copy
                    # of it is still in flight: not started now.
                    if verdict == "held_capped":
                        c.demote_held_capped += 1
                    continue
                if verdict == "gpu_tight":
                    # Pinning every free member would take
                    # the running headroom. Not started (a wait); if a
                    # running allocation reclaims it, the loss is
                    # missed_copy, never a half copy.
                    c.demote_gpu_tight += 1
                    continue
                # ``is_whole`` is ``_start_verdict``'s own decision,
                # taken before ``room`` below reserves (an eviction it applies
                # would make the rows look idle).
                if fresh:
                    # The unit's own host backing (its predecessor's visible
                    # prefix included) is protected while it reserves.
                    own = self._host_rows_of(node_units)
                    self.units.protect(protected | own)
                    if not room(
                        len(fresh), units=[u.uid for u in node_units]
                    ):
                        self.units.protect(protected)  # roll back
                        # Fits the cache, but its rows are pinned now.
                        c.demote_host_busy += 1
                        continue
                    # Belt and braces: the eviction must not have changed
                    # which members need a destination (a lost backing row
                    # turns its GPU twin dirty).
                    again = [
                        m for m in order
                        if m.block_id not in picked_ids
                        and m.block_id not in self.in_flight
                        and m.block_id not in self._res_members
                        and tier.is_dirty(m)
                    ]
                    if [m.block_id for m in again] != [m.block_id for m in fresh]:
                        self.units.protect(protected)
                        c.demote_host_busy += 1
                        continue
                    got = self._alloc_rows(len(fresh))
                    if len(got) < len(fresh):  # cannot happen: room() counted
                        self.cpu_pool.free_blocks(got)
                        self.units.protect(protected)
                        c.demote_host_busy += 1
                        continue
                    host_ok -= len(got)
                    fresh_ok, fresh_short = 0, None  # rows taken: rescan
                    if key is None:
                        key = min(u.uid for u in node_units)
                    self._reserve(key, fresh, got)
                    self._started.update(u.uid for u in node_units)
                    for u in node_units:
                        self.units.resume(u.uid)
                    protected |= own
                    if not owned:
                        active.update((u.uid, u) for u in node_units)
                        committed += rows
                if is_whole:
                    # Started on ``_start_verdict``'s whole-unit rule;
                    # its pins skip the head floor in ``take`` as well.
                    whole.update(m.block_id for m in fresh)
                for member in order:
                    if not take(member, lone=False):
                        stopped = True
                        break
                # The reserved members not issued now wait in the backlog,
                # pinned (the slack check above covered every free one).
                rest = [m for m in fresh if m.block_id not in picked_ids]
                if rest:
                    self.backlog.extend(rest)
                    gpu_free = self._hold(rest, gpu_free, headroom, pins, picked_ids)

        # The copy scheduler's lanes (PREEMPT, GRACE,
        # REPAIR, RUNNING) after the reactive victims, within what is left of
        # the D credits. Every accepted member goes through ``take`` and
        # ``begin_demote`` like a reactive pick (the transfer-owned source
        # ref until all-rank quiescence; the demotion path is unchanged).
        copyq = self.copyq
        if (
            copyq is not None and copyq.manager is not None
            and self.units is not None and not stopped and budget > 0
        ):
            quiet = (
                not picked and not self._waiting_now(step) and not self.backlog
                and not self._res and demand == 0
                and self._last_head_need == 0 and self._head_need == 0
                and not self.promote_dma and not self.promote_reserved
            )
            self._lane1_allow = self._lane1_budget(quiet)
            lane1_left = self._lane1_allow
            offers = copyq.offers(
                step, rows_budget=budget, bytes_budget=0, busy=picked_ids
            )
            gen = tier.generation
            from vllm.merlin.kv_host_tier_iface import CopyLane

            for offer in offers:
                if offer.lane is CopyLane.RUNNING:
                    # Lane 1 never takes the credits, idle rows or GPU slack
                    # of anything else: its own sub-budget, idle host rows
                    # beyond one restore unit's worth, never evicting.
                    if lane1_left <= 0 or stopped:
                        continue
                    # The byte cap
                    # counts every copy, also those this walk picked before
                    # this offer (issued only after the walk): lane 1's
                    # allowance was read from the copies in flight alone.
                    cap = lane1_left
                    if self.bytes_d > 0 and self.row_bytes > 0:
                        used = (len(self.in_flight) + len(picked_ids)) * self.row_bytes
                        cap = min(cap, max(0, (self.bytes_d - used) // self.row_bytes))
                    if cap <= 0:
                        continue
                    members = [
                        b for b, g in zip(offer.blocks, offer.generations)
                        if b.block_id not in picked_ids
                        and b.block_id not in self.in_flight
                        and gen[b.block_id] == g
                        and tier.is_dirty(b) and tier.is_ready(b)
                        and b.ref_cnt > 0
                        and not self.units.unit_of_block(b)
                    ][:cap]
                    if not members:
                        continue
                    spare = self._hold_cap()
                    # The idle check is a count, not a queue walk.
                    idle = self.units.idle_count(host_used + len(members) + spare)
                    fit = max(0, idle - host_used - spare)
                    members = members[:fit]
                    if not members:
                        continue
                    host_ok = max(host_ok, host_used + len(members))
                    # ``take`` charges ``budget``: lane 1 is charged its own
                    # sub-budget instead, the D budget is left as it was.
                    saved, budget = budget, len(members)
                    for m in members:
                        if not take(m, lone=False):
                            break
                        if m.block_id in picked_ids:
                            self._offer_of[m.block_id] = offer
                            lane1_left -= 1
                    budget = saved
                    continue
                if budget <= 0 or stopped:
                    break
                members = [
                    b for b, g in zip(offer.blocks, offer.generations)
                    if b.block_id not in picked_ids
                    and b.block_id not in self.in_flight
                    and gen[b.block_id] == g
                    and tier.is_dirty(b) and tier.is_ready(b)
                ]
                if not members:
                    continue
                displace = copyq.may_displace(offer)
                unit = (
                    self.units.units.get(offer.uid)
                    if offer.uid is not None else None
                )
                if offer.txn_atomic and unit is not None:
                    fresh = [
                        m for m in members if m.block_id not in self._res_members
                    ]
                    if fresh:
                        if not self._store_pins_fit(fresh, picked_ids):
                            c.copyq_offer_skipped += 1
                            continue
                        # Assembly ledger (unique rows): every unit in
                        # progress plus this one fits the host cache.
                        extra = 0 if unit.uid in active else len(fresh)
                        if committed + extra > self._host_capacity():
                            c.demote_assembly_waits += 1
                            c.copyq_offer_skipped += 1
                            continue
                        own = self._host_rows_of([unit])
                        self.units.protect(protected | own)
                        if not room(
                            len(fresh), units=[unit.uid], displace=displace
                        ):
                            self.units.protect(protected)
                            c.copyq_offer_skipped += 1
                            continue
                        got = self._alloc_rows(len(fresh))
                        if len(got) < len(fresh):  # cannot happen: room() counted
                            self.cpu_pool.free_blocks(got)
                            self.units.protect(protected)
                            c.copyq_offer_skipped += 1
                            continue
                        host_ok -= len(got)
                        fresh_ok, fresh_short = 0, None  # rows taken: rescan
                        self._reserve(unit.uid, fresh, got)
                        self._started.add(unit.uid)
                        self.units.resume(unit.uid)
                        protected |= own
                        if unit.uid not in active:
                            active[unit.uid] = unit
                            committed += len(fresh)
                    # Lane provenance for every member, issued now or
                    # queued in the backlog for a later chunk): a GRACE/PREEMPT
                    # source is never front-requeued.
                    for m in members:
                        self._offer_of[m.block_id] = offer
                    for m in members:
                        if not take(m, lone=False):
                            stopped = budget > 0
                            break
                    rest = [
                        m for m in members
                        if m.block_id in self._res_members
                        and m.block_id not in picked_ids
                    ]
                    if rest:
                        queued = {b.block_id for b in self.backlog}
                        self.backlog.extend(
                            m for m in rest if m.block_id not in queued
                        )
                        gpu_free = self._hold(rest, gpu_free, headroom, pins, picked_ids)
                    continue
                # Non-atomic (REPAIR, RUNNING) or a unit no longer recorded:
                # idle rows or running-twin duplicates only, never published
                # content (``displace=False``); the rest waits for the next
                # offer.
                unreserved = sum(
                    1 for m in members if m.block_id not in self._res_members
                )
                fit = unreserved
                if unreserved and not room(unreserved, displace=False):
                    fit = max(0, host_ok - host_used)
                for m in members:
                    if m.block_id not in self._res_members:
                        if fit <= 0:
                            continue
                        fit -= 1
                    if not take(m, lone=False):
                        break
                    if m.block_id in picked_ids:
                        self._offer_of[m.block_id] = offer
            self._note_ledger()

        if pins:
            self.gpu_pool.touch(pins)
        if not picked:
            return []
        # Destinations: reserved members draw their unit's rows; lone victims
        # get exactly the idle rows ``room`` counted.
        # Unit-less copies first, into rows that back no free GPU twin
        # (``fresh_room`` counted them; a unit start since may have taken
        # one, so a shortfall drops the last of them, as host-full).
        lone_picks = [b for b in picked if b.block_id not in self._res_members]
        fresh_picks = [b for b in lone_picks if b.block_id in fresh_ids]
        other_picks = [b for b in lone_picks if b.block_id not in fresh_ids]
        fresh_rows = self._alloc_rows(len(fresh_picks), guarded=True)
        lone_rows = self._alloc_rows(len(other_picks))
        drop = {b.block_id for b in fresh_picks[len(fresh_rows):]}
        for bid in drop:
            ident = (bid, tier.generation[bid])
            if ident not in self._dropped:
                self._dropped.add(ident)
                c.demote_dropped_host_full += 1
        # The others cannot fall short: room() counted them.
        drop.update(b.block_id for b in other_picks[len(lone_rows):])
        if drop:
            picked = [b for b in picked if b.block_id not in drop]
        fresh_rows.reverse()
        lone_rows.reverse()
        out: list[KVCacheBlock] = []
        for b in picked:
            if b.block_id in self._res_members:
                out.append(self._draw_row(b.block_id))
            elif b.block_id in fresh_ids:
                out.append(fresh_rows.pop())
            else:
                out.append(lone_rows.pop())
        self._note_reserved()
        if not picked:
            return []
        self.host_rows = out
        self._staged = tier.begin_demote(picked)
        assert len(self._staged) == len(picked)
        c.credits_d_issued += len(picked)
        c.copy_reactive_rows += sum(
            1 for b in picked if b.block_id not in self._offer_of
        )
        # The demotion pin now protects them: drop the hold.
        self._release_held([b for b in picked if b.block_id in self._held_gpu])
        return picked

    @property
    def bytes_d(self) -> int:
        """Opt-in D2H byte cap per rank (``COPY_BYTES_D_MIB``; 0 = rows
        only)."""
        return self._bytes_d

    @bytes_d.setter
    def bytes_d(self, value: int) -> None:
        self._check_byte_cap(int(value), self._row_bytes)
        self._bytes_d = int(value)

    @property
    def row_bytes(self) -> int:
        """Host row bytes per rank (set at bind by the runtime)."""
        return self._row_bytes

    @row_bytes.setter
    def row_bytes(self, value: int) -> None:
        self._check_byte_cap(self._bytes_d, int(value))
        self._row_bytes = int(value)

    @staticmethod
    def _check_byte_cap(bytes_d: int, row_bytes: int) -> None:
        """A nonzero byte cap smaller than one physical row would
        make every demotion budget 0 (``left // row_bytes`` == 0) and
        silently disable the walker. Refused as soon as both are known."""
        if bytes_d > 0 and row_bytes > 0 and bytes_d < row_bytes:
            raise ValueError(
                f"{BYTES_D_ENV}: a byte cap of {bytes_d} B per rank is smaller "
                f"than one host row ({row_bytes} B); set 0 (rows only) or at "
                f"least {-(-row_bytes // (1 << 20))} MiB"
            )

    @property
    def credits_d(self) -> int:
        """Demotion-direction row credits (``credits_cap`` unless the D env
        or an assignment set its own)."""
        return self.credits_cap if self._credits_d is None else self._credits_d

    @credits_d.setter
    def credits_d(self, value: int | None) -> None:
        self._credits_d = value

    def _d_budget(self) -> int:
        """Demotion credits left: rows (``credits_d``), and, with a byte cap
        per rank (``bytes_d``), the rows the bytes left still cover."""
        # Lane-1 (RUNNING) copies in flight have their own sub-budget
        # (``_lane1_budget``) and never reduce this one: a slack copy must
        # never delay a reactive copy (lane-1 copies in flight
        # held the D credits, so a zero-running waiter's victims could not be
        # copied until the gate escaped, max_idle_wait 9).
        in_flight = len(self.in_flight) - self._lane1_in_flight()
        rows = self.credits_d - in_flight
        if not self.split_directions:
            # Shared credits until the restore side keeps its own
            # account (``plans.begin_tick``): restore DMA and the credits
            # reserved for a waiting restore batch still count here.
            rows -= self.promote_dma + self.promote_reserved
        if self.bytes_d > 0 and self.row_bytes > 0:
            # The opt-in byte cap is ONE D2H budget per rank.
            # Every demotion copy in flight is charged, lane 1 included
            # (lane 1 rows have their own credit share, its bytes do not).
            left = self.bytes_d - len(self.in_flight) * self.row_bytes
            rows = min(rows, left // self.row_bytes)
        return rows

    def _lane1_in_flight(self) -> int:
        if not self._offer_of:
            return 0
        from vllm.merlin.kv_host_tier_iface import CopyLane

        return sum(
            1 for bid in self.in_flight
            if (o := self._offer_of.get(bid)) is not None
            and o.lane is CopyLane.RUNNING
        )

    LANE1_CREDIT_SHARE = 4  # lane 1 may hold at most credits_d / 4 in flight

    def _lane1_budget(self, quiet: bool) -> int:
        """Rows lane 1 (RUNNING) may offer this walk: none unless the tier is
        quiet (no gate wait, no waiting head need, no demand hint, no restore
        DMA or reserved restore credits, no backlog or reservation, no
        reactive pick this walk), else up to its own sub-budget of
        ``credits_d / LANE1_CREDIT_SHARE`` rows in flight (at least one row
        while ``credits_d`` > 0, so a small P still makes progress), and,
        with a byte cap, never past ``bytes_d`` for all copies in flight."""
        if not quiet or self.credits_d <= 0:
            return 0
        share = max(1, self.credits_d // max(1, self.LANE1_CREDIT_SHARE))
        rows = max(0, share - self._lane1_in_flight())
        if self.bytes_d > 0 and self.row_bytes > 0:
            left = self.bytes_d - len(self.in_flight) * self.row_bytes
            rows = min(rows, max(0, left // self.row_bytes))
        return rows

    def _sweep_offers(self) -> None:
        """Forget offer membership of blocks no copy will issue any more
        (not in flight, queued or reserved)."""
        if not self._offer_of:
            return
        queued = {b.block_id for b in self.backlog}
        for bid in list(self._offer_of):
            if (
                bid not in self.in_flight and bid not in queued
                and bid not in self._res_members
            ):
                del self._offer_of[bid]

    def _note_ledger(self) -> None:
        """Assembly ledger peaks (unique host rows held for unpublished
        units, reserved or in flight)."""
        if self.copyq is None:
            return
        rows = self.copyq.assembly_rows()
        c = self.counters
        c.assembly_unique_rows_peak = max(c.assembly_unique_rows_peak, rows)
        if self.row_bytes > 0:
            c.assembly_unique_bytes_peak = max(
                c.assembly_unique_bytes_peak, rows * self.row_bytes
            )

    PRUNE_EVERY = 256

    def _prune_memos(self, tier) -> None:
        """Forget idents of content that no longer exists: the
        counted-once sets are keyed by (block id, generation) and a reused
        block's old generation can never come back."""
        if self.step - self._prune_step < self.PRUNE_EVERY:
            return
        self._prune_step = self.step
        gen = tier.generation
        for name in ("_dropped", "_dropped_cr", "_skipped", "_too_large"):
            ids = getattr(self, name)
            setattr(self, name, {i for i in ids if gen[i[0]] == i[1]})
        self._landed = {b: r for b, r in self._landed.items() if gen[b] == r[0]}

    def issue(self, txn_id: int, gpu_block_ids: list[int]) -> None:
        tickets, self._staged = self._staged, []
        assert [t.block.block_id for t in tickets] == list(gpu_block_ids)
        self.txns[txn_id] = DemoteTxn(
            txn_id, self.epoch, tickets,
            acks=TxnAcks(self.epoch, txn_id, 0, self.world_size),
        )
        for ticket in tickets:
            self.in_flight[ticket.block.block_id] = ticket
        self._progress += sum(
            1 for t in tickets if self._gate_relevant(t.block.block_id)
        )
        self._update_credits()
        if self.copyq is not None and self._offer_of:
            # Per offer, the rows THIS transaction issues: a chunked offer
            # (credits, backlog) issues over several transactions and each
            # row is counted once, here (chunked grace).
            issued: dict[int, tuple[Any, int]] = {}
            for ticket in tickets:
                offer = self._offer_of.get(ticket.block.block_id)
                if offer is not None:
                    prev = issued.get(offer.offer_id)
                    issued[offer.offer_id] = (offer, (prev[1] if prev else 0) + 1)
            for offer, rows in issued.values():
                self.copyq.note_issued_rows(offer, rows)
                self.copyq.on_issued(offer, txn_id)

    def note_restore_issued(self, rows: int) -> None:
        """Restore (H2D) rows issued this tick (``credits_r_issued``; called
        by the plans tick)."""
        if rows > 0:
            self.counters.credits_r_issued += rows

    def set_promote_dma(self, blocks: int) -> None:
        """Restore (H2D) blocks issued and not completed, after loads issue.
        Called after the tick's walk, so a restore in flight
        voids the idle verdict that walk left (the breaker would otherwise
        release a waiter on the next admission with the restore, an
        opportunity per ``_victims_may_change``, in flight)."""
        self.promote_dma = blocks
        if blocks > 0:
            self.idle_steps = 0
        self._update_credits()

    def _update_credits(self) -> None:
        c = self.counters
        c.credits = len(self.in_flight) + self.promote_dma
        c.credits_peak = max(c.credits_peak, c.credits)
        if self.row_bytes > 0:
            c.bytes_d_inflight_peak = max(
                c.bytes_d_inflight_peak, len(self.in_flight) * self.row_bytes
            )
            c.bytes_r_inflight_peak = max(
                c.bytes_r_inflight_peak, self.promote_dma * self.row_bytes
            )

    # -- completion ---------------------------------------------------------
    def owns(self, txn_id: int) -> bool:
        return txn_id in self.txns

    def ack(self, txn_id: int, ranks: Iterable[int]) -> bool:
        """Record successful rank acks (idempotent per rank); True once all
        ranks acked and the transaction can commit."""
        return self.report(
            txn_id, [(rank, True, True) for rank in ranks]
        ) is TxnState.COMMIT

    def report(
        self, txn_id: int, reports: Iterable[tuple[int, bool, bool]]
    ) -> TxnState | None:
        """Record per-rank ``(rank, success, quiescent)`` reports of a
        demotion transaction; None when ``txn_id`` is not (or no longer) one.

        COMMIT: every rank succeeded and is quiescent (publish). UNWIND: a
        rank failed and every rank is quiescent (publish nothing; sources
        stay GPU-resident, the staged host blocks are released). PENDING: keep
        everything pinned."""
        txn = self.txns.get(txn_id)
        if txn is None or txn.acks is None:
            return None
        for rank, success, quiescent in reports:
            txn.acks.report(rank, success, quiescent)
            if success:
                txn.acked_ranks.add(rank)
        return txn.acks.state

    def take_txn(self, txn_id: int) -> DemoteTxn | None:
        return self.txns.pop(txn_id, None)

    def is_current(self, txn: DemoteTxn) -> bool:
        if txn.epoch == self.epoch:
            return True
        self.counters.stale_epoch_txns += 1
        return False

    def still_valid(self, gpu_block_id: int) -> bool:
        """The source of an in-flight demotion still holds the content the
        copy read (only then may its host copy be published)."""
        ticket = self.in_flight.get(gpu_block_id)
        return ticket is not None and self.tier.demote_still_valid(ticket)

    def finish(self, txn: DemoteTxn, landed: Sequence[bool]) -> None:
        """``txn``'s copies are over; ``landed[i]``: ticket i's host copy was
        published. Releases every source pin (published ones to the front of
        the free queue)."""
        # A finished copy is its units' progress, landed or not.
        self._note_unit_progress(t.block for t in txn.tickets)
        for ticket in txn.tickets:
            bid = ticket.block.block_id
            self.in_flight.pop(bid, None)
            key = self._res_issued.pop(bid, None)
            if key is not None and key in self._res:
                # A landing (or a finished copy) is the unit's progress: a
                # slow but healthy pipe never stall-releases.
                self._res_progress[key] = self.step
        # Release order: only a reactive copy's published source
        # goes to the front of the free queue; a copy-queue lane's source was
        # copied ahead of eviction and keeps its place (tail on a free).
        front_ok = None
        acked: dict[int, tuple[Any, list[bool]]] = {}
        relevant = [self._gate_relevant(t.block.block_id) for t in txn.tickets]
        if self._offer_of:
            from vllm.merlin.kv_host_tier_iface import CopyLane

            front_ok = []
            for ticket, ok in zip(txn.tickets, landed, strict=True):
                offer = self._offer_of.pop(ticket.block.block_id, None)
                front_ok.append(offer is None or offer.lane is CopyLane.REACTIVE)
                if offer is not None:
                    acked.setdefault(offer.offer_id, (offer, []))[1].append(bool(ok))
        done = self.tier.end_demote(txn.tickets, landed, front_ok)
        # An old epoch's copy (kept lane identity across reset) is no
        # lane outcome: the copy queue it was offered by was reset.
        if self.copyq is not None and txn.epoch == self.epoch:
            for offer, oks in acked.values():
                self.copyq.on_acked(offer, oks)
        c = self.counters
        done_ids = {id(t) for t in done}
        self._progress += sum(
            1 for t, r in zip(txn.tickets, relevant) if r and id(t) in done_ids
        )
        for t, r in zip(txn.tickets, relevant):
            if r and id(t) in done_ids:
                # Re-copy memo, with the keys it covered
                self._landed[t.block.block_id] = (
                    t.gen, self._landed_keys(t.block, t.cid), self.step
                )
        c.demoted += len(done)
        c.demote_discarded += len(txn.tickets) - len(done)
        self._update_credits()
        if self.units is not None:
            c.demoted_units = self.units.counters.units_host_registered

    def release_stale(self, txn: DemoteTxn) -> None:
        """A transaction of an older epoch (reset) drained: nothing is
        published, the sources are released normally and stay dirty."""
        self.finish(txn, [False] * len(txn.tickets))

    # -- cache_thrash HostTierEviction -------------------
    def is_host_backed(self, blocks: Sequence["KVCacheBlock"]) -> bool:
        """Reusing ``blocks`` loses no restore dependency: each is CLEAN by
        the tier-state relation (host twin under every key it carries, revalidated
        now; or not worth keeping). A demotion in flight is not backing."""
        tier = self.tier
        return all(tier.state(b) is BlockState.CLEAN for b in blocks)

    def demote(self, blocks: Sequence["KVCacheBlock"]) -> None:
        self.counters.thrash_demoted += 1

    def unit_lost(self, blocks: Sequence["KVCacheBlock"]) -> None:
        if any(self.tier.is_dirty(b) for b in blocks):
            self.counters.thrash_unit_lost += 1

    # -- reclaim accounting --------------------------------------------------
    def account_reclaims(self) -> None:
        """Attribute dirty reclaims since the last call (``BlockTierState``
        counts them at the pop): to escaped admissions first, the rest to
        running allocations."""
        if self.gpu_pool is None or self.gpu_pool.tier is None:
            return
        tc = self.tier.counters
        delta = tc.reclaimed_dirty - self._dirty_seen
        self._dirty_seen = tc.reclaimed_dirty
        c = self.counters
        escaped = min(delta, self._escape_pending)
        c.demote_missed_escape += escaped
        c.demote_missed_running += delta - escaped
        self._escape_pending = 0
        c.demote_redemoted = tc.reclaimed_backed
        c.reclaimed_unretained = tc.reclaimed_unretained
        if self.units is not None:
            c.unit_lost = self.units.counters.units_lost
            c.demoted_units = self.units.counters.units_host_registered

    # -- admission gate -----------------------------------------------------
    def _soon_free(self) -> int:
        """In-flight sources that return to the free queue on completion."""
        return sum(1 for t in self.in_flight.values() if t.block.ref_cnt == 1)

    def _victims_may_change(self) -> bool:
        """A pending transfer could change an admission's victims
        or free a host row. It errs to waiting: any demotion
        copy in flight, of any lane, whatever its source's references (one
        returning to the free queue lands CLEAN, a reactive one at the front;
        a source another request also pins still releases its destination
        row, which then backs no free twin), or restore DMA in flight (its
        completion releases the host rows it reads). Restore DMA that frees
        no row a unit-less copy may take makes a futile wait: an accepted
        cost, bounded by the escape count. Without any, a victim the
        walker refuses stays refused.

        Known limit (bounded class): the check does not prove the
        transfer can spare this victim (a landing row the guard protects, a
        copy whose source stays pinned by a running request): such waits are
        futile but end at the escape count, charged as escapes."""
        # ``in_flight`` holds exactly the demotion copies issued and not yet
        # finished (``issue`` / ``finish``), every lane: a
        # GRACE / PREEMPT / RUNNING copy releases its destination row too.
        return bool(self.promote_dma or self.in_flight)

    def _charge_escape(
        self, request: "Request", num_new_blocks: int, exclude: frozenset[int]
    ) -> None:
        """An admission escapes: its reclaims of protected content are this
        escape's loss (``demote_missed_escape``). Every escape path
        charges the raw protected victims (``mirror=False``),
        not only those the walker would copy, so a refused unit-less victim is
        never a running miss. ``demote_escapes`` counts the escaping
        admission once (a request polled again before it
        allocates escapes again; the reclaim attribution is per poll, as
        ``account_reclaims`` clears it every walk)."""
        raw = self.protected_victims(num_new_blocks, exclude, mirror=False)
        if raw:
            self._escape_pending += raw
            rid = request.request_id
            npre = getattr(request, "num_preemptions", 0)
            if self._escaped.get(rid) != npre:
                self._escaped[rid] = npre
                self.counters.demote_escapes += 1

    def can_progress(self) -> bool:
        """A demotion is in flight or can be issued (credits come back as
        restores complete); False only when nothing will free a victim.
        Any demotion in flight counts, copy-queue lanes included: its landing
        can release a host row for a refused unit-less victim even when the host
        is full. Known limit: a slow copy-queue copy is not breaker progress, so
        the idle breaker can still release a refused victim before it lands."""
        if self.promote_dma or self.in_flight:
            return True
        return self.credits_cap > 0 and self.cpu_pool.get_num_free_blocks() > 0

    def _start_verdict(
        self, node, order, need: int, *, gpu_free: int, headroom: int,
        budget: int, active: dict, committed: int, picked_ids=frozenset(),
        head_need: int = 0, charged: int | None = None,
    ):
        """Whether the walker starts (or continues) the multi-member unit it
        meets at ``node`` now, before host rows: the ONE predicate the walk
        and the admission gate's mirror share. Returns
        ``(verdict, (node_units, rows, key, owned, fresh, whole))``; ``verdict``
        is
        None (go on to host rows) or why not: ``never_fit`` (larger than the
        cache), ``credits`` (every DMA credit taken), ``not_ready`` (nothing
        copyable yet), ``assembly`` (MAX_ASSEMBLING units in progress, or
        they and this one exceed the cache), ``gpu_tight`` (pinning its free
        members would take the running headroom), ``held_capped`` (or the
        waiting head's need, unless the unit issues whole this walk
        ``_issues_whole``; or exceed the hold cap),
        ``stalled`` (it gave up its start and a copy is still in flight).
        ``whole``: the head floor was waived for this unit (
        ``_issues_whole``), so the walk's ``take`` skips it for these pins;
        False whenever the floor does not bind."""
        tier = self.tier
        units = self.units
        node_units = units.unit_of_block(node)
        rows = self._unit_rows(node_units, order)
        key = next(
            (self._res_members[m.block_id] for m in order
             if m.block_id in self._res_members),
            None,
        )
        owned = key is not None or any(u.uid in active for u in node_units)
        fresh = [
            m for m in order
            if m.block_id not in picked_ids
            and m.block_id not in self.in_flight
            and m.block_id not in self._res_members
            and tier.is_dirty(m)
        ]
        whole = False
        start = (node_units, rows, key, owned, fresh, whole)
        if charged is None:
            charged = self._charged()
        if rows > self._host_capacity():
            return "never_fit", start
        if any(units.is_abandoned(u.uid) for u in node_units) and any(
            m.block_id in self.in_flight for m in order
        ):
            return "stalled", start
        if not owned:
            if need and budget <= 0:
                return "credits", start
            if not need:
                return "not_ready", start
            if (
                len(active) >= self.max_assembling
                or committed + rows > self._host_capacity()
            ):
                return "assembly", start
        if fresh:
            # Refuse before reserving any rows: no partially pinned unit.
            if not self._store_pins_fit(fresh, picked_ids):
                return "held_capped", start
            free_members = sum(
                1 for m in fresh
                if m.ref_cnt == 0 and m.block_id not in self._held_gpu
            )
            if gpu_free - free_members < headroom:
                return "gpu_tight", start
            floor = headroom + head_need
            if gpu_free - free_members < floor and self._issues_whole(
                fresh, need, budget, node_units=node_units
            ):
                # The head floor binds, and the unit's fresh
                # members all issue this walk: it is not charged the waiting
                # head's need. Judged only when the floor binds (the probe's
                # cost; the waiver is moot otherwise).
                floor = headroom
                whole = True
                start = (node_units, rows, key, owned, fresh, whole)
            if (
                gpu_free - free_members < floor
                or charged + free_members > self._hold_cap()
            ):
                return "held_capped", start
        return None, start

    def _issues_whole(self, fresh, need: int, budget: int, *, node_units=()) -> bool:
        """Whether every fresh member of the unit issues its copy in this walk
        (all ready, and the credits cover every copy the unit needs), so its
        pins are transfer refs that return to the front of the free queue CLEAN
        when the copies land, never holds left to later walks. The waiting
        head's need is then not a reason to refuse the start: that head is the
        admission about to reclaim these very members, so the floor would make
        the walker refuse them, leave the gate's mirror nothing to protect, and
        let the admission destroy them uncopied. The running headroom
        (``gpu_tight``), the hold cap, ``_hold``'s floor for members left to
        later walks and the lone-block floor are unchanged. Walker and gate
        mirror share it through ``_start_verdict``; the walk's ``take`` skips
        the head floor for exactly these members.

        Not when the unit's host rows would come from evicting a unit whose free
        GPU twin (ref 0) loses its last host copy. Un-keying it turns the twin
        DIRTY again, so the copy only moves a dirty block into the head's
        reclaims. The caller asks only when the head floor binds, i.e. when
        ``gpu_free < headroom + head_need + free members``: the head's path is
        then the whole free queue, so every free twin is in it. A twin counts
        only when EVERY host row holding one of its keys is a row the plan
        frees; a row shared with a surviving unit keeps it CLEAN, and refusing
        in that case would reopen a lost-copy case. ``_backs_free_twin`` is not
        reused for this check: it does not flag a unit victim's re-dirtied
        twins. A twin already DIRTY (e.g. the same tokens
        recomputed into a new generation under the same keys) loses nothing and
        is skipped, and only host rows holding the twin's own content
        (``host_src == content_id``) count as its backing, so a foreign copy
        under the same key neither vetoes nor masks.

        A member that is a live write frontier
        (``BlockTierState.is_frontier``: on ``live_ids`` and referenced, its
        writer's table) does not make the unit "not whole". It is not
        copyable now (``is_ready``), but it is never a free-queue victim and
        the walk never pins it from the free pool, so it costs the head
        nothing: the unit's free members issue now as for a whole unit, and
        the frontier member waits in the backlog (a referenced hold,
        uncharged) until its writer moves on or the stall release gives the
        start up. A not-ready member that is not a referenced frontier
        (unacked or masked content, a ref-0 block in a stale set included)
        still refuses the waiver.

        Known limits:
        * The gate's mirror judges ``need <= budget`` against the full
          ``_d_budget()``, while the walk spends credits on the backlog and
          earlier victims first, so it may refuse a unit the mirror counted.
          The head then waits on a copy that is not issued until the escape
          count (ESCAPE_STEPS consecutive waits without progress) releases it;
          the unit is lost (charged as an escape, not a running miss).
        * A whole unit's copy that never lands keeps its sources
          transfer-pinned, so a head with no other free room stays short of
          blocks for as long as the DMA is outstanding (as for any copy in
          flight; the stall release does not fire while a copy is in flight).
        * The walk calls this before ``room``, with rows its earlier lone
          picks counted but not yet allocated; the mirror cannot see those,
          so on a full host a walk whose lone picks take the last idle rows
          may evict where this probe saw idle rows (rare).
        * Head size: a plain-mode head (``note_head``) registers its
          full-sequence need, a coherent one (``admit``) only the chunk's fresh
          blocks, so a long plain head charges a larger floor and
          ``_shed_holds`` releases more holds. This is not starvation (the full
          sequence is the plain head's real admission requirement), but that
          head misses more copies."""
        tier = self.tier
        if not (need <= budget and all(
            tier.is_ready(m) or tier.is_frontier(m) for m in fresh
        )):
            return False
        units = self.units
        # The idle check is a count, not a queue walk.
        if units is not None and units.idle_count(len(fresh)) < len(fresh):
            # The rows would come from an eviction: probe it as the walk's
            # ``room`` would (the unit's own backing, its predecessor's
            # visible prefix included, protected while it reserves).
            prev = units.protect(
                units._protected_rows | self._host_rows_of(node_units)
            )
            try:
                plan = units.plan(
                    len(fresh), self.step,
                    for_units=[u.uid for u in node_units], count=False,
                )
            finally:
                units.protect(prev)
            gone = frozenset(plan.rows)
            host = self.cpu_pool.cached_block_hash_to_block
            for uid in plan.victims:
                u = units.units.get(uid)
                if u is None:
                    continue
                for m in units.gpu_members(u):
                    if m.ref_cnt != 0 or tier.is_dirty(m):
                        # An already-DIRTY twin (e.g. a recomputed
                        # generation under the same key) loses nothing.
                        continue
                    cid = tier.content_id(m)
                    if cid is None:
                        continue  # unstamped: no host copy can back it
                    for key in self._gpu_keys(m):
                        backing = [
                            h for h in host.get_blocks(key)
                            if tier.host_src.get(h.block_id) == cid
                        ]
                        if backing and all(h.block_id in gone for h in backing):
                            return False
        return True

    def _gate_context(self) -> dict:
        """The walker's start inputs as the next walk will see them if the
        waiter waits (no allocation): free GPU blocks now, the last walk's
        headroom, the credits left, the units in progress and their rows."""
        active = self._active_units()
        return dict(
            gpu_free=self.gpu_pool.get_num_free_blocks(),
            headroom=self._last_headroom,
            head_need=max(self._head_need, self._last_head_need),
            budget=self._d_budget(),
            active=active,
            committed=sum(
                self._unit_rows([u], self.units.gpu_members(u))
                for u in active.values()
            ),
        )

    def _unitless_row_ok(self) -> bool:
        """Whether the walker's ``take`` finds a row for one unit-less copy
        (``room(1, src=...)`` then ``fresh_room``), without side effects; the
        caller sets ``lone_victims`` as the walk does. It is the one
        guarded capacity decision the admission mirror shares with the walk.
        An idle row the guard passes; else, with no idle row at all, the row
        ``room``'s plan would free (un-keyed: it backs nothing); idle rows
        that all back a free twin make ``room`` content without a plan and
        ``fresh_room`` refuse."""
        if self._guarded_rows(1):
            return True
        if self.units.idle_count(1):
            return False
        return self.units.can_reserve(1, self.step)

    def _effective_copies(
        self, block: "KVCacheBlock", ctx: dict | None = None, why=None
    ) -> list["KVCacheBlock"]:
        """The members the walker would copy when it meets ``block`` (its
        selection, without side effects): the unit's ready dirty members, or
        nothing when the unit is tiny, the walk would stop at it (credits,
        running headroom), or its rows cannot be reserved now. ``why``: a
        list that gets ``"host_full"`` when a unit-less block's copy is
        refused for want of a row it may take (``_unitless_row_ok``)."""
        tier = self.tier

        def copyable(m: "KVCacheBlock") -> bool:
            return (
                m.block_id not in self.in_flight
                and tier.is_dirty(m)
                and tier.is_ready(m)
            )

        if self._tiny(block):
            return []
        members = self._unit_members_uncached(block) if self.units is not None else []
        order = [block] + [m for m in members if m is not block]
        copies = [m for m in order if copyable(m)]
        if self.units is None:
            if len(copies) > self.cpu_pool.get_num_free_blocks():
                return []
            return copies
        if len(order) == 1:
            # A lone block: the walker's ``take``. A unit-less one never evicts
            # a published unit (rule A, the same rule as ``room(src=...)``).
            if not copies:
                return []
            unitless = not self.units.unit_of_block(block)
            if unitless:
                # The walker's guarded decision, not any
                # free row. Judged before the transient stops below (a spent
                # credit or the pin floor hid the refusal, so
                # the victim was neither waited for nor charged as an escape);
                # ``admit`` then waits on it while a transfer is pending.
                prev = self.units.lone_victims(True)  # rule A, as the walk
                try:
                    if not self._unitless_row_ok():
                        if why is not None:
                            why.append("host_full")
                        return []
                finally:
                    self.units.lone_victims(prev)
            if ctx is None:
                ctx = self._gate_context()
            if not self._store_pins_fit([block]):
                return []
            if ctx["budget"] <= 0:
                # The walk stops: no credits. (A copyable victim is then
                # admitted silently, a running miss: pre-existing.)
                return []
            if block.ref_cnt == 0 and ctx["gpu_free"] - 1 < (
                ctx["headroom"] + ctx.get("head_need", 0)
            ):
                # The walk stops: its pin would take the headroom or the
                # waiting head's blocks (as ``take``).
                return []
            if not unitless and not self.units.can_reserve(1, self.step):
                return []
            return copies
        if ctx is None:
            ctx = self._gate_context()
        verdict, (node_units, _rows, _key, _owned, fresh, _whole) = self._start_verdict(
            block, order, len(copies), **ctx
        )
        if verdict is not None:
            # The walker will not start it now: not a reason to wait.
            return []
        if fresh:
            # The walker's ``room``: its own backing protected while it
            # reserves (no side effects here).
            prev = self.units.protect(
                self.units._protected_rows | self._host_rows_of(node_units)
            )
            try:
                if not self.units.can_reserve(len(fresh), self.step):
                    return []  # its rows are pinned: it waits
            finally:
                self.units.protect(prev)
        return copies

    def protected_victims(
        self,
        num_new_blocks: int,
        exclude: frozenset[int] = frozenset(),
        *,
        mirror: bool = True,
        refused: list[int] | None = None,
    ) -> int:
        """Protected blocks among the ``num_new_blocks`` reclaims an
        allocation makes after adopting ``exclude`` (allocation simulation in
        the allocator's pop order). ``mirror``: count only the
        members the walker would copy (``_effective_copies``); False counts
        every protected victim (what an escape is charged with).
        ``refused`` (mirror only): ``refused[0]`` is raised by the protected
        victims the walker refuses for want of a host row they may take
        (lost on admission as on a full host)."""
        if num_new_blocks <= 0:
            return 0
        n = 0
        protected = 0
        counted: set[int] = set()
        # One decision per restore unit, as the walker makes it: the members
        # it would copy (``_effective_copies``), and whether it refused them
        # for host rows.
        decided: dict[tuple, frozenset[int]] = {}
        host_full: set[tuple] = set()
        # The walker's start inputs, read once per call (``_gate_context``).
        ctx_box: list = [None]

        def judge(block: "KVCacheBlock") -> int:
            if block.block_id in counted or not self._protected(block):
                return 0
            counted.add(block.block_id)
            if self.units is None or not mirror:
                return 1
            owners = tuple(sorted(u.uid for u in self.units.unit_of_block(block)))
            ident = owners or ("block", block.block_id)
            chosen = decided.get(ident)
            if chosen is None:
                if ctx_box[0] is None and self.units is not None:
                    ctx_box[0] = self._gate_context()
                why: list[str] = []
                chosen = decided[ident] = frozenset(
                    m.block_id
                    for m in self._effective_copies(block, ctx_box[0], why)
                )
                if why:
                    host_full.add(ident)
            if block.block_id in chosen:
                return 1
            if refused is not None and ident in host_full:
                refused[0] += 1
            return 0

        # Adoption happens before any pop: the adopted hits leave the
        # queue and their groups in the shadow first, never as reclaims.
        for block, _keyed, destroyed in iter_pool_reclaims(
            self.tier.pool, num_new_blocks, exclude
        ):
            protected += judge(block)
            # The full destructive effect of the pop: a session break
            # un-hashes its peers whether or not they are reclaimed next.
            for peer in destroyed:
                if peer.block_id not in exclude:
                    protected += judge(peer)
            n += 1
            if n >= num_new_blocks:
                break
        return protected

    def note_head(
        self, request: "Request", num_new_blocks: int, shortfall: int
    ) -> None:
        """A WAITING / PREEMPTED admission refused before the gate
        (``allocate_slots``' non-coherent full-sequence check). Its need sets
        the next walk's head floor, as ``admit`` does, and a shortfall the
        in-flight copy sources cover is a shortfall wait counted as ``admit``'s
        real-shortfall branch counts it in coherent mode, so the refusal does
        not break the head's run of consecutive waits. No new state: every walk
        resets ``_head_need`` and the next admission overwrites it. The idle
        breaker is checked first, as in ``admit``. The need registered is the
        full sequence's (see ``_issues_whole``'s known limits, head size)."""
        if self.gpu_pool is None or num_new_blocks <= 0:
            return
        self._head_need = num_new_blocks
        rid = request.request_id
        if self.gate_idle():
            # The idle breaker first, as ``admit``: no wait
            # can make progress, so the record goes and the break is counted.
            self._waits.pop(rid, None)
            self.counters.gate_idle_breaks += 1
            return
        if 0 < shortfall <= self._soon_free() and not self._escaping(rid):
            self._wait(rid, num_new_blocks, scope=shortfall)

    def admit(
        self,
        request: "Request",
        num_new_blocks: int,
        shortfall: int,
        adopted: Sequence[Sequence["KVCacheBlock"]],
    ) -> tuple[int, bool]:
        """``(shortfall, demotion_wait)`` for a WAITING / PREEMPTED request's
        allocation of ``num_new_blocks`` fresh blocks.

        ``shortfall`` is the allocator's real shortfall; ``adopted`` the
        blocks the allocation adopts (they leave the free queue before the
        fresh blocks are popped, so they are not reclaims).
        """
        if self.gpu_pool is None:
            return shortfall, False
        rid = request.request_id
        # The waiting queue's head as of this admission pass: the
        # latest admission since the last walk; admission is
        # in queue order and stops at the first request that cannot go.
        self._head_need = num_new_blocks
        if self.gate_idle():
            # Idle breaker: no wait can make progress. Release this
            # request's wait and let the allocator (and the scheduler's pin
            # yield, which a demotion wait suppresses) act on the real
            # shortfall; dirty victims are reclaimed as an escape.
            self._waits.pop(rid, None)
            if shortfall <= 0 and num_new_blocks > 0:
                exclude = frozenset(
                    b.block_id for group in adopted for b in group if not b.is_null
                )
                self._charge_escape(request, num_new_blocks, exclude)
            self.counters.gate_idle_breaks += 1
            return shortfall, False
        if shortfall > 0:
            # Soon-free only for the real-shortfall test.
            if shortfall <= self._soon_free() and not self._escaping(rid):
                # Progress is the request's own deficit shrinking.
                return 0, self._wait(rid, num_new_blocks, scope=shortfall)
            # The wait record stays (an admission, an escape, the idle
            # breaker or the request's end drops it). The scheduler yields a
            # pin for this shortfall and retries in the same pass; a retry
            # that becomes a demotion wait continues this record. Dropping it
            # here restarted the count every pass when the yielded waiter
            # re-pinned at the end of it, so neither ESCAPE_STEPS nor the
            # backstop ever fired.
            return shortfall, False
        if num_new_blocks <= 0:
            self._waits.pop(rid, None)
            return 0, False
        exclude = frozenset(
            b.block_id for group in adopted for b in group if not b.is_null
        )
        # A walker that cannot progress (no credits, or the host full, with
        # nothing in flight) saves nothing: the mirror would admit silently
        # and the dirty reclaims would count as running misses. Count every
        # protected victim instead; ``_escaping`` then escapes at once.
        refused = [0]
        dirty = self.protected_victims(
            num_new_blocks, exclude, mirror=self.can_progress(), refused=refused
        )
        # A unit-less victim the walker refuses
        # (no row it may take) waits like any demotion wait, but only
        # while a pending transfer could change the victims or free a row
        # (``_victims_may_change``: a landing source goes CLEAN to the front
        # and is popped instead). The wait is counted by ``_wait`` and bound
        # by the escape count. With no such transfer it escapes at once.
        waitable = bool(refused[0]) and self._victims_may_change()
        if dirty == 0 and not waitable:
            self._waits.pop(rid, None)
            if refused[0]:
                # Nothing to wait for: the loss is this admission's, as on a
                # full host (``can_progress`` false), not a running miss.
                self._charge_escape(request, num_new_blocks, exclude)
            return 0, False
        if self._escaping(rid):
            self._waits.pop(rid, None)
            self._charge_escape(request, num_new_blocks, exclude)
            return 0, False
        # The progress scope keeps the refused
        # victims whenever the request waits (a copyable victim or an
        # opportunity justifies it), so it drops only when a victim is saved
        # or taken, never because an opportunity went away (that was a new
        # low and reset the stalled count).
        return 0, self._wait(rid, num_new_blocks, scope=dirty + refused[0])

    def _escaping(self, rid: str) -> bool:
        """Escape on NO PROGRESS: ``escape_after`` consecutive counted
        waiting steps without progress on the request's own deficit
        (``_wait``); unscoped waits use demotions issued or completed.
        A total-wait bound scaled by the batches the demand needs
        (``cdiv(need, P)``, doubled) is the backstop. Also escapes when
        ``can_progress`` is false."""
        waited = self._waits.get(rid)
        if waited is not None and waited[1] != self.step:
            batches = max(1, -(-waited[2] // max(1, self.credits_cap)))
            # Only consecutive waits count; a stretch of passes
            # without one (a real shortfall) ages the stalled count out.
            stalled = waited[4] if self.step - waited[1] <= 1 else 0
            if stalled >= self.escape_after or waited[0] >= (
                self.escape_after * batches * 2
            ):
                return True
        return not self.can_progress()

    def _waiting_now(self, step: int) -> bool:
        """A request counted a demotion wait on the pass before walk
        ``step``: a record kept across a real shortfall, or left by a request
        the scan no longer reaches, is not one."""
        return any(step - waited[1] <= 1 for waited in self._waits.values())

    def _wait(self, rid: str, num_new_blocks: int, scope: int | None = None) -> bool:
        """Count one waiting step. ``scope``: the request's protected victims
        still to save or real shortfall. Only a new low
        of the scope in the current run of consecutive waits resets the
        escape counter, so progress on other units, or a scope swinging back
        down to a level it already reached, cannot extend a
        stalled wait past ESCAPE_STEPS. A pass without a counted wait starts
        a new run (the total-wait backstop keeps counting). Without a scope,
        any demotion issued or completed counts as progress."""
        self.demand_hint = max(self.demand_hint, num_new_blocks)
        self._escaped.pop(rid, None)  # a later escape is a new one
        cur = self._progress if scope is None else scope
        waited = self._waits.setdefault(rid, [0, -1, num_new_blocks, cur, 0])
        waited[2] = max(waited[2], num_new_blocks)
        if waited[1] != self.step:
            if self.step - waited[1] > 1:
                waited[3], waited[4] = cur, 0  # not consecutive: a new run
            waited[1] = self.step
            waited[0] += 1
            if scope is None:
                moved = cur != waited[3]
                waited[3] = cur
            else:
                moved = cur < waited[3]
                waited[3] = min(waited[3], cur)
            if moved:
                waited[4] = 0  # the request's own demand moved
            else:
                waited[4] += 1
            c = self.counters
            c.demote_wait_steps += 1
            c.demote_wait_max = max(c.demote_wait_max, waited[0])
        return True

    def forget_request(self, rid: str) -> None:
        self._waits.pop(rid, None)
        self._escaped.pop(rid, None)

    # -- reset --------------------------------------------------------------
    def reset(self) -> None:
        """New cache epoch: in-flight transactions drain but never publish."""
        self.epoch += 1
        if self.gpu_pool is not None and self.gpu_pool.tier is not None:
            self.gpu_pool.tier.live_ids = frozenset()
        self.backlog = []
        if self.gpu_pool is not None:
            self._release_held(list(self._held_gpu.values()))
        self._release_reserved_all()
        self.idle_steps = 0
        self.demand_hint = 0
        self._waits.clear()
        self._escape_pending = 0
        self._escaped.clear()
        self._dropped.clear()
        self._dropped_cr.clear()
        self._landed.clear()  # host keys are gone: any copy is a first one
        self._born.clear()  # the wait records are gone too
        self._ep_start = None
        self._started.clear()
        self._cap_step = -1
        self._head_need = 0
        self._last_head_need = 0
        if self.units is not None:
            self.units.protect(())
        self._skipped.clear()
        self._unit_progress.clear()
        # Copies still in flight keep their lane: lane 1's
        # sub-budget and the gate's progress test classify them until they
        # drain; the rest of the offer map belongs to the old epoch.
        self._offer_of = {
            bid: o for bid, o in self._offer_of.items() if bid in self.in_flight
        }
        if self.copyq is not None:
            self.copyq.reset()

    def snapshot_counters(self) -> WritebackCounters:
        self.account_reclaims()
        c = self.counters
        c.held_gpu_now = len(self._held_gpu)
        c.held_units_now = len({
            u.uid for b in self._held_gpu.values()
            for u in self.units.unit_of_block(b)
        }) if self.units is not None else 0
        return replace(c)


__all__ = [
    "CREDITS_ENV",
    "ESCAPE_ENV",
    "GATE_ENVS",
    "DemoteTxn",
    "WriteBack",
    "WritebackCounters",
    "copy_credits",
    "derive_reserve",
    "escape_steps",
]
