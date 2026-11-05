# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Plan-aware admission for the host-RAM KV tier.

The ``PrefixHitPlan`` replaces the earlier lookup (GPU prefix, then a host
suffix from the scheduler-block boundary). One plan drives:

* the computed-token count at admission (its boundary B, from token zero);
* which table positions adopt a GPU source and which get a private GPU
  destination for a host restore, per group (a mixed layout);
* the destination draws, counted inside ``allocate_slots`` with the rest of
  the admission (``KVCacheManager.restore_slots``) and placed after every
  GPU source is touched (``place_restore_destinations``);
* the promotion load: one transfer per plan, built from the plan's slots,
  never from ``local_tokens // block + i``.

Leases (``PrefixLease``) are taken before anything is allocated and
never outlive one scheduling pass. A restore that cannot start under the
in-flight cap waits while restores make progress; after a stall it falls back
to the GPU tier alone, then cold.

Soft promotion: on a successful load every full restored page is published
on the GPU under its host copy's keys; the host copies stay keyed, and the
unit directory groups them again only once the whole unit is GPU-resident
(``units.on_host_released``). The partial boundary page is private to its
consumer and is not published from the restore.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any

from vllm.logger import init_logger
from vllm.merlin import kv_host_tier_lifecycle
from vllm.merlin.kv_host_tier_iface import AdmissionOutcome
from vllm.merlin.kv_host_tier_units import GroupKind, unit_key
from vllm.merlin.kv_tier_lookup import (
    PrefixHitAccounting,
    PrefixHitPlan,
    PrefixLease,
    PrefixSlot,
    Tier,
    tier_version,
)

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.request import Request
    from vllm.v1.simple_kv_offload.manager import SimpleCPUOffloadScheduler

# Fallback levels per waiting request (a restore that cannot fit
# misses before any DMA).
UNION = 0
GPU_ONLY = 1
# One re-resolution after a lease failed validation (key, generation or epoch
# moved between resolve and lease); a second failure is a cold lookup.
MAX_RESOLVE_ATTEMPTS = 2
# A valid host plan that finds the restore cap (``RestoresInFlight``) full WAITS,
# holding no lease, for as long as any cap holder makes real progress (a credit
# batch resolves on every rank, or a load retires): each holder's retirement
# frees a slot, FCFS, so no holder's progress is unrelated to a waiter.
# It falls back to the GPU-only plan (the miss cause ``restore_cap_wait``) only
# when no restore has progressed for a stall timeout derived from the observed
# restore latency, on BOTH clocks: scheduler steps, RESTORE_STALL_FACTOR x the
# EWMA of issue -> commit steps per batch (never less than
# RESTORE_STALL_MIN_STEPS), AND wall time, RESTORE_STALL_FACTOR x the EWMA of
# issue -> commit seconds per batch (never less than RESTORE_STALL_MIN_S). The
# engine loop steps continuously while a long-context restore takes seconds, so
# steps alone would call a progressing batch stalled before its DMA could land;
# the step clock stays so a rig that drives steps without real time still
# works. Never a pass count: a long restore that keeps landing batches never
# turns a restorable prefix into a recompute. A genuine transfer failure
# retires the load (progress) and frees its slot, so the waiter is admitted on
# the next pass.
RESTORE_STALL_MIN_STEPS = 8
RESTORE_STALL_MIN_S = 2.0
RESTORE_STALL_FACTOR = 4
# Prior for the batch latency before any batch was observed (DMA ack 1-3
# steps) and the EWMA weight of each new sample.
RESTORE_LATENCY_PRIOR_STEPS = 2.0
RESTORE_LATENCY_EWMA = 0.5
# Wall-clock prior: the physical cost of the last batch issued, row_bytes x
# batch rows / a conservative pinned host->GPU rate (RESTORE_H2D_BYTES_PER_S),
# or a flat RESTORE_LATENCY_PRIOR_S when no batch was issued yet / row_bytes is
# not visible (x RESTORE_STALL_FACTOR = the RESTORE_STALL_MIN_S floor).
RESTORE_LATENCY_PRIOR_S = 0.5
RESTORE_H2D_BYTES_PER_S = 2 * (1 << 30)

logger = init_logger(__name__)


@dataclass(frozen=True)
class _ZeroHit:
    """The plan fields ``_record_outcome`` reads, for an admission on no hit."""

    shared_prefix_boundary: int = 0
    boundary: int = 0
    gpu_boundary: int = 0


@dataclass
class PlanCounters:
    """The plan-admission counters."""

    plan_lookups: int = 0
    plan_revalidations: int = 0
    plan_cache_hits: int = 0
    mixed_layouts: int = 0
    restore_admitted: int = 0
    restore_coalesced_waits: int = 0
    restore_unfit: int = 0
    restore_published_blocks: int = 0
    private_tails: int = 0
    # Passes a valid host plan spent waiting for a restore
    # slot (no lease held), and admissions that fell back to GPU-only after
    # a stalled DMA (never on a pass count; once per admission).
    restore_cap_wait: int = 0
    restore_cap_fallback: int = 0
    # Scoped host-LRU touches at admission: units on the
    # requester's own chain touched / units met through a shared row skipped.
    touch_scoped: int = 0
    touch_skipped_offchain: int = 0
    # Once-per-admission outcomes handed to the metrics sink.
    admission_outcomes: int = 0


@dataclass
class PlanHit:
    """What the scheduler adopts for one admission attempt."""

    blocks: tuple[list[Any], ...]
    boundary: int
    gpu_boundary: int
    shared_prefix_boundary: int
    restore: bool


@dataclass
class _Pending:
    lease: PrefixLease
    plan: PrefixHitPlan
    restore: tuple[PrefixSlot, ...]
    # GPU dependencies outside the table (EAGLE/SW proof witnesses), leased
    # for this admitting pass only.
    pins: list = field(default_factory=list)
    # The plan's host source rows, ref-leased from ``admit``
    # until the hand-off pins them itself (``_issue``) or the pending plan is
    # released. Without retention nothing else keeps a looked-up unit from
    # being the LRU victim of the room-making walk that follows admission
    # (which shows as ``host_evicted`` between lookup and hand-off).
    # Per plan, per unit, released on every path: never a host-wide lock.
    host_pins: list = field(default_factory=list)


@dataclass
class _CachedPlan:
    signature: tuple
    plan: PrefixHitPlan
    hashes: frozenset
    positions: tuple[int, int]


@dataclass
class _Inflight:
    sources: frozenset
    waiters: set[str] = field(default_factory=set)


class TierPlans:
    """Owned by ``SimpleCPUOffloadScheduler`` when the tier is active."""

    def __init__(self, sched: SimpleCPUOffloadScheduler):
        self.sched = sched
        # Lookups that recompute a once-held session, by loss cause,
        # and time spent in the lookup path.
        self.miss_causes: dict[str, int] = {}
        # The per-admission miss line is a census line: INFO with
        # MERLIN_KV_HOST_TIER_CENSUS=1, DEBUG otherwise (the counts above
        # are kept either way).
        from vllm.merlin.envs import read_registered

        self._miss_at_info = bool(read_registered("MERLIN_KV_HOST_TIER_CENSUS"))
        self.lookup_seconds = 0.0
        self.kv_cache_manager: KVCacheManager | None = None
        self._lookup_coordinator = None
        self._pending: dict[str, _Pending] = {}
        self._inflight: dict[tuple, _Inflight] = {}
        self._plan_loads: dict[str, tuple] = {}
        # Per plan load: unissued (gpu, cpu) pairs, and the issued batch size.
        self._unissued: dict[str, list[tuple[int, int]]] = {}
        self._issued: dict[str, int] = {}
        # Per plan load: batches issued, receives seen, and whether the last
        # pair has been issued.
        self._batches: dict[str, int] = {}
        self._receives: dict[str, int] = {}
        self._final_issued: dict[str, bool] = {}
        # Per plan load: host rows its issued batches read. A failed restore
        # un-keys only these (``read_sources``); unread rows hold intact
        # content and stay keyed.
        self._read: dict[str, list[int]] = {}
        # Loads cut short by a cancel/reset are never published.
        self._cut: set[str] = set()
        # Per restore: GPU blocks reserved at hand-off for the consumer's
        # private recurrent/partial tails (None: none could be reserved).
        self._tails: dict[str, list | None] = {}
        self._tick_budget = 0
        self.fallback: dict[str, int] = {}
        # Per waiting request, its last resolved plan (never a lease).
        self._plans: dict[str, _CachedPlan] = {}
        self.counters = PlanCounters()
        self.accounting = PrefixHitAccounting()
        self._admissions = 0
        # Per waiting request, the restore progress
        # it last saw and the step and wall time it saw it at (the two stall
        # clocks); the requests that fell back after a stall (the outcome
        # label). Cleared on admission / forget / reset.
        self._cap_waits: dict[str, tuple[int, int, float]] = {}
        # Request id -> the steps its cap wait had stalled for at its first
        # fallback (the ``restore_cap_wait`` line's age).
        self._cap_fallback: dict[str, int] = {}
        # The requests whose latest ``_admit`` pass took the GPU-only cap
        # fallback branch: what that pass returned (its plan or its zero hit)
        # is the fallback. Only such an admission is a ``restore_cap_wait``
        # miss (a refused fallback whose later pass resolves
        # something else, e.g. after a host eviction, is not); ``_cap_fallback``
        # keeps the age and the once-per-admission dedup.
        self._fallback_pass: set[str] = set()
        # Request id -> shared-prefix boundary of the plan-less
        # hit ``_admit`` just returned; consumed by ``on_admitted``.
        self._zero_hit: dict[str, int] = {}
        # The requests whose ``lookups`` was counted
        # and that are not admitted yet: a pass retried after a GPU-room
        # refusal is the same logical admission, like a cap wait. It
        # gates ``lookups`` only: the hit counts at the hand-off
        # and the miss is settled once at admission (``_settle_miss``).
        # Cleared on admission / forget / reset.
        self._looked_up: set[str] = set()
        # Restore progress: +1 per credit batch resolved on every rank and per
        # load retired. The issue step / wall time of each request's batch in
        # flight and the EWMAs of issue -> commit latency in steps and in
        # seconds (the stall timeouts). The wall clock is injectable; the
        # seconds EWMA starts from the physical prior of the last batch issued
        # (``_latency_prior_s``) until its first sample.
        self._now: Callable[[], float] = time.monotonic
        self._restore_progress = 0
        self._batch_issue_step: dict[str, int] = {}
        self._batch_issue_time: dict[str, float] = {}
        self._restore_latency_steps = RESTORE_LATENCY_PRIOR_STEPS
        self._restore_latency_s: float | None = None
        self._latency_prior_s = RESTORE_LATENCY_PRIOR_S
        # Request id -> step of its last recorded admission
        # outcome and scoped touch, so a re-entered admission in the same
        # step (coalesced follower, retried pass) records and touches once.
        self._admitted_at: dict[str, int] = {}

    # -- resolution -------------------------------------------------------------
    def bind(self, kv_cache_manager: KVCacheManager) -> None:
        self.kv_cache_manager = kv_cache_manager
        # Per-direction credits. The walker's demotion
        # budget stops subtracting promotions once the planner stops
        # subtracting demotions (``begin_tick``).
        wb = self.sched.writeback
        if wb is not None:
            wb.split_directions = True

    def version(self, pool, block, group_id, key):
        """The lookup predicate on both tiers: ``tier_version`` (GPU
        readiness), and on the host a staged member of an assembling unit is
        unreadable until the unit commits, in resolution
        and in every validation."""
        rt = self.sched.tier_runtime
        units = rt.units if rt is not None else None
        if (
            units is not None
            and pool is self.sched.cpu_block_pool
            and units.is_staged(block)
        ):
            return None
        return tier_version(pool, block, group_id, key)

    def _epoch(self) -> int:
        return self.sched.cache_epoch

    def _restore_cap(self):
        runtime = self.sched.tier_runtime
        units = runtime.units if runtime is not None else None
        cap = getattr(units, "restores_in_flight", units)
        return cap if all(
            callable(getattr(cap, name, None))
            for name in ("can_start", "start", "done")
        ) else None

    def coordinator(self):
        """The GPU coordinator the plan's group finders run on. Without a
        bound KVCacheManager (connector-level rigs) a lookup-only coordinator
        over the bound GPU pool is built once."""
        if self.kv_cache_manager is not None:
            return self.kv_cache_manager.coordinator
        if self._lookup_coordinator is None:
            from vllm.v1.core.kv_cache_coordinator import get_kv_cache_coordinator
            from vllm.v1.core.kv_cache_utils import (
                sliding_window_context_kv_is_causal,
            )

            s = self.sched
            src = s.cpu_coordinator
            cfg = s.vllm_config
            spec = cfg.speculative_config
            coordinator = get_kv_cache_coordinator(
                kv_cache_config=s.kv_cache_config,
                max_model_len=cfg.model_config.max_model_len,
                max_in_flight_tokens=cfg.max_in_flight_tokens,
                use_eagle=spec is not None and spec.use_eagle_block_drop(),
                enable_caching=True,
                enable_kv_cache_events=False,
                dcp_world_size=s.cp_world_size,
                pcp_world_size=1,
                scheduler_block_size=s.block_size,
                hash_block_size=s.hash_block_size,
                allow_partial_hash_hits=src.enable_partial_hash_hits,
                num_prefill_lookahead=cfg.num_prefill_lookahead_tokens,
                causal_sliding_window_kv=sliding_window_context_kv_is_causal(spec),
            )
            pool = s._gpu_block_pool
            assert pool is not None, "bind the GPU block pool before a lookup"
            coordinator.block_pool = pool
            for manager in coordinator.single_type_managers:
                manager.block_pool = pool
                manager._null_block = pool.null_block
            self._lookup_coordinator = coordinator
        return self._lookup_coordinator

    # -- plan reuse across passes --------------------------------------
    def _key_logs(self):
        from vllm.merlin.kv_tier_lookup import KeyLog

        logs = []
        for pool in (self.sched._gpu_block_pool, self.sched.cpu_block_pool):
            if pool.key_log is None:
                pool.key_log = KeyLog()
            logs.append(pool.key_log)
        return logs

    def _signature(self, request: Request, level: int) -> tuple:
        # Readiness (the commit frontier, staged assembly members, this
        # step's exclusions) is not part of it: a resolve that met such a
        # source is provisional and never reused, and a probe it never made
        # lies behind a miss that only a new key or such a source can end.
        return (
            level, self._epoch(), request.num_tokens, len(request.block_hashes),
            self.lookup_enabled(request),
        )

    def _cached_plan(self, request: Request, level: int) -> PrefixHitPlan | None:
        """The request's previous plan when nothing could have lengthened it:
        same cap, epoch and fallback level, no key of this request inserted
        in either pool since, and it was not provisional. A shortened plan is caught by the lease
        validation every use runs. The plan holds no storage."""
        entry = self._plans.get(request.request_id)
        if entry is None or entry.signature != self._signature(request, level):
            return None
        from vllm.v1.core.kv_cache_utils import get_block_hash

        for log, position in zip(self._key_logs(), entry.positions):
            keys = log.since(position)
            if keys is None or any(get_block_hash(k) in entry.hashes for k in keys):
                return None
        return entry.plan

    def _remember(self, request: Request, level: int, plan: PrefixHitPlan) -> None:
        if plan.provisional:
            self._plans.pop(request.request_id, None)
            return
        old = self._plans.get(request.request_id)
        hashes = (
            old.hashes
            if old is not None and old.signature[3] == len(request.block_hashes)
            else frozenset(request.block_hashes)
        )
        gpu_log, host_log = self._key_logs()
        self._plans[request.request_id] = _CachedPlan(
            self._signature(request, level), plan, hashes,
            (gpu_log.end, host_log.end),
        )

    def lookup_enabled(self, request: Request) -> bool:
        if self.kv_cache_manager is not None:
            return self.kv_cache_manager.prefix_cache_lookup_enabled(request)
        return not request.skip_reading_prefix_cache

    def resolve(
        self, request: Request, max_tokens: int | None = None, *, gpu_only=False
    ) -> PrefixHitPlan:
        """Read-only: no pin, allocation, publication or admitted count."""
        cap = request.num_tokens - 1 if self.lookup_enabled(request) else 0
        if max_tokens is not None:
            cap = min(cap, max_tokens)
        self.counters.plan_lookups += 1
        return self.coordinator().find_longest_union_hit(
            request.block_hashes,
            max(0, cap),
            None if gpu_only else self.sched.cpu_block_pool,
            version=self.version,
            epoch=self._epoch(),
        )

    def restore_slots(self, plan: PrefixHitPlan) -> tuple[PrefixSlot, ...]:
        """The host slots that need a destination: inside every group's
        attention window at B (a position the manager skips holds null)."""
        managers = self.coordinator().single_type_managers
        out = []
        for slot in plan.slots:
            if slot.source.tier is not Tier.HOST:
                continue
            manager = managers[slot.group_id]
            skipped = manager.get_num_skipped_tokens(plan.boundary) // manager.block_size
            if slot.block_index >= skipped:
                out.append(slot)
        return tuple(out)

    @staticmethod
    def _sources(restore) -> frozenset:
        return frozenset(
            (slot.source.block.block_id, slot.source.primary) for slot in restore
        )

    def _coalesce(self, request_id: str, restore) -> bool:
        """A restore whose host sources an in-flight promotion already reads
        waits for it, then resolves again (its pages are then GPU hits)."""
        sources = self._sources(restore)
        for key, inflight in self._inflight.items():
            if key[1] == self._epoch() and sources <= inflight.sources:
                inflight.waiters.add(request_id)
                self.sched._promotion_waiters.setdefault(key, set()).add(request_id)
                self.counters.restore_coalesced_waits += 1
                return True
        return False

    # -- scheduler admission ----------------------------------------------------
    MISS_CAUSES = (
        "missed_copy", "race", "host_evicted", "never_fit", "skipped_tiny",
        # A host-capable plan recomputed after the bounded wait
        # for a restore slot (iface.MISS_CAUSES_V3).
        "restore_cap_wait",
    )

    def _reusable(self, request: Request) -> int | None:
        """The request's reusable prefix R from the outcome sink's request
        history (None: no sink, R unknown)."""
        sink = getattr(self.sched.tier_runtime, "outcome_sink", None)
        fn = getattr(sink, "reusable_tokens", None)
        return int(fn(request)) if callable(fn) else None

    def _miss_cause(
        self, request: Request, adopted: int, cap_age: int | None,
        reusable: int | None,
    ) -> tuple[str, int] | None:
        """Why an admission that adopted ``adopted`` tokens recomputes a
        session the tier once held, decided once from the
        admitted plan: no miss when it
        adopted the reusable prefix; ``restore_cap_wait`` when the GPU-only
        cap fallback is what it admitted (``cap_age``); else the loss cause
        recorded for the deepest lost unit on its prefix past ``adopted``
        (current evidence, whatever earlier passes saw); else ``unknown``
        when R says something reusable was not adopted."""
        if reusable is not None and reusable <= adopted:
            return None
        if cap_age is not None:
            return "restore_cap_wait", cap_age
        runtime = self.sched.tier_runtime
        units = runtime.units if runtime is not None else None
        if units is not None and units.lost:
            size = max(1, units.hash_block_size)
            past = request.block_hashes[adopted // size:]
            got = units.loss_cause(past) if past else None
            if got is not None:
                return got
        return ("unknown", -1) if reusable is not None else None

    def _settle_miss(
        self, request: Request, adopted: int, cap_age: int | None,
        reusable: int | None,
    ) -> None:
        """The admission's one miss line, which a test run attributes by
        request id, and its plans-line ``miss_<cause>`` count: once per
        admission, at the admission, from the plan it adopted. The line is
        logged at INFO only with the census on (DEBUG otherwise).
        A request that leaves or a cache reset before it is
        admitted recomputed nothing and books nothing."""
        got = self._miss_cause(request, adopted, cap_age, reusable)
        if got is None:
            return
        cause, age = got
        if cause in self.MISS_CAUSES:
            self.miss_causes[cause] = self.miss_causes.get(cause, 0) + 1
        (logger.info if self._miss_at_info else logger.debug)(
            "SimpleCPU tier miss: req=%s cause=%s age_steps=%d hit_tokens=%d "
            "rows_missing_tokens=%d",
            request.request_id, cause, age, adopted,
            max(0, request.num_tokens - 1 - adopted),
        )

    def _touch_units(self, plan, request: Request, *, restore=()) -> None:
        """Refresh the host LRU of the units this admission actually reuses,
        on GPU and host hits alike: a GPU-hot prefix is not
        the first host eviction.

        Scoped to the
        requester's OWN chain. A unit is touched only when its boundary lies
        at or below ``plan.boundary`` and the key the requester's hashes give
        at that boundary is one of the unit's required keys, i.e. the unit is
        a prefix of this request (its predecessor turns, or a shared-prefix
        unit recorded at the junction). Units met only through a shared
        physical row (every other agent's units, via the system-prompt
        pages) are not touched. Called once per admission from
        ``on_admitted`` (never on a lookup retry or a failed attempt)."""
        runtime = self.sched.tier_runtime
        units = runtime.units if runtime is not None else None
        if units is None:
            return
        step = self.sched._sched_step
        # Host use is per unit: only a unit owning a row this
        # admission restores from the host; a GPU-served unit on the same
        # chain (the shared-prefix unit at the junction) is not promoted.
        host_uids = {
            u.uid
            for slot in restore
            if slot.source.tier is not Tier.GPU
            for u in units.unit_of_block(slot.source.block)
        }
        hbs = max(1, units.hash_block_size)
        hashes = request.block_hashes
        chain = len(hashes) * hbs
        full = [g for g in units.groups if g.kind is GroupKind.FULL]
        seen: set[int] = set()
        for group in plan.groups:
            for source in group:
                if source is None:
                    continue
                for unit in units.unit_of_block(source.block):
                    if unit.uid in seen:
                        continue
                    seen.add(unit.uid)
                    b = unit.boundary
                    # Supersede-at-publish: a kept predecessor is still in
                    # ``unit_of_block``. Below the adopted boundary its
                    # successor carries the use; AT the adopted boundary the
                    # predecessor IS what this admission restores (the
                    # successor is still incomplete), so it is touched.
                    if units.is_superseded(unit.uid) and b < plan.boundary:
                        continue
                    on_chain = (
                        0 < b <= plan.boundary
                        and b <= chain
                        and b % hbs == 0
                        and any(
                            unit_key(hashes, hbs, b, g.group_id) in unit.required
                            for g in full
                        )
                    )
                    if not on_chain:
                        self.counters.touch_skipped_offchain += 1
                        continue
                    # The unit table's ``touch``: a host restore is host use; a GPU-only
                    # hit refreshes the LRU position only.
                    units.touch(unit.uid, step, host_use=unit.uid in host_uids)
                    self.counters.touch_scoped += 1

    # -- restore-cap wait -----------------------------------------------------
    def restore_stall_steps(self) -> int:
        """Steps without restore progress after which a cap waiter falls back
        to its GPU-only plan: RESTORE_STALL_FACTOR x the observed batch
        latency (EWMA, in steps), at least RESTORE_STALL_MIN_STEPS."""
        return max(
            RESTORE_STALL_MIN_STEPS,
            int(-(-RESTORE_STALL_FACTOR * self._restore_latency_steps // 1)),
        )

    def restore_stall_seconds(self) -> float:
        """Wall-clock twin of ``restore_stall_steps``: RESTORE_STALL_FACTOR x
        the observed batch latency in seconds (EWMA; before any sample the
        physical prior of the last batch issued), at least
        RESTORE_STALL_MIN_S. A waiter falls back only when both clocks have
        run out since the last progress."""
        latency = self._restore_latency_s
        if latency is None:
            latency = self._latency_prior_s
        return max(RESTORE_STALL_MIN_S, RESTORE_STALL_FACTOR * latency)

    def _note_batch_issued(self, request_id: str, rows: int) -> None:
        """A credit batch of ``rows`` host rows was issued for ``request_id``
        this tick: its issue step and wall time feed the two latency EWMAs
        when it resolves, and its physical cost (row_bytes x rows / a
        conservative host->GPU rate) is the wall-clock prior until then."""
        self._batch_issue_step[request_id] = self.sched._sched_step
        self._batch_issue_time[request_id] = self._now()
        wb = self.sched.writeback
        row_bytes = wb.row_bytes if wb is not None else 0
        if row_bytes > 0 and rows > 0:
            self._latency_prior_s = max(
                RESTORE_LATENCY_PRIOR_S, rows * row_bytes / RESTORE_H2D_BYTES_PER_S
            )

    def _note_restore_progress(self, request_id: str | None) -> None:
        """A restore made real progress: a batch resolved on every rank or a
        load retired (success or failure, both free the way for waiters).
        With the batch's issue step and time known, sample its latency for
        the two stall timeouts."""
        self._restore_progress += 1
        issued = self._batch_issue_step.pop(request_id, None) if request_id else None
        issued_at = self._batch_issue_time.pop(request_id, None) if request_id else None
        if issued is not None:
            sample = float(max(1, self.sched._sched_step - issued))
            self._restore_latency_steps += RESTORE_LATENCY_EWMA * (
                sample - self._restore_latency_steps
            )
        if issued_at is not None:
            seconds = max(0.0, self._now() - issued_at)
            if self._restore_latency_s is None:
                self._restore_latency_s = seconds
            else:
                self._restore_latency_s += RESTORE_LATENCY_EWMA * (
                    seconds - self._restore_latency_s
                )

    def _cap_wait_progressing(self, request_id: str) -> bool:
        """Whether a cap waiter keeps waiting: True while any cap holder has
        progressed since it last looked, or for less than the stall timeout
        since one last did on either clock (steps or wall time); False =
        stalled on both, fall back."""
        step = self.sched._sched_step
        wall = self._now()
        now = self._restore_progress
        seen = self._cap_waits.get(request_id)
        if seen is None or seen[0] != now:
            self._cap_waits[request_id] = (now, step, wall)
            return True
        if step - seen[1] < self.restore_stall_steps():
            return True
        return wall - seen[2] < self.restore_stall_seconds()

    def _record_outcome(
        self, request: Request, plan, restore, step: int, *,
        reusable: int | None, cap_fallback: bool = False,
    ) -> None:
        """The once-per-admission ``AdmissionOutcome`` for the
        metrics sink (``tier_runtime.outcome_sink``, installed by the census
        / runtime code). ``reusable`` (``_reusable``) comes from the sink's
        request history, never from the census union."""
        runtime = self.sched.tier_runtime
        sink = getattr(runtime, "outcome_sink", None)
        if sink is None:
            return
        reusable = reusable or 0
        host_tokens = plan.boundary - plan.gpu_boundary if restore else 0
        adopted = plan.gpu_boundary + host_tokens
        cause = None
        if reusable > adopted:
            # ``cap_fallback``: the admitted plan is the GPU-only cap
            # fallback, read before the wait state is cleared.
            cause = "restore_cap_wait" if cap_fallback else "unknown"
        sink.record(AdmissionOutcome(
            request_id=request.request_id,
            step=step,
            prompt_tokens=request.num_prompt_tokens,
            reusable_tokens=reusable,
            gpu_tokens=plan.gpu_boundary,
            host_tokens=host_tokens,
            shared_prefix_tokens=plan.shared_prefix_boundary,
            restore_requested=bool(restore),
            restore_completed=None if restore else True,
            miss_cause=cause,
            first_turn=reusable == 0,
            gpu_boundary=plan.gpu_boundary,
        ))
        self.counters.admission_outcomes += 1

    def _note_zero_hit_admission(self, request: Request, level: int) -> None:
        """An admission on no hit still records its outcome: a
        whole-prefix loss must reach R and ``lost_tokens``. Its routes: a
        boundary-0 plan, a lease that failed twice, a stalled cap wait whose
        GPU-only plan is empty (``_admit``), and a plan that went cold after
        it could not fit (``use_cold``, which never reaches ``_admit``). Any
        other ``on_admitted`` without a pending plan is no plan admission (a
        load that came back). Consumes the wait state, so a later re-admission
        is not labelled by it."""
        req_id = request.request_id
        shared = self._zero_hit.pop(req_id, None)
        if shared is None and level <= GPU_ONLY:
            return
        # Nothing came from the host. A ``restore_cap_wait`` miss only when
        # the pass that produced this zero hit took the cap fallback branch
        # (``use_cold`` admits after no ``_admit`` pass: the latest pass's
        # fallback, lowered to cold), not because an earlier one did.
        cap_age = self._cap_fallback.pop(req_id, None)
        if req_id not in self._fallback_pass:
            cap_age = None
        self._fallback_pass.discard(req_id)
        self._cap_waits.pop(req_id, None)
        step = self.sched._sched_step
        # A request that may not read the cache (prompt logprobs, ...) lost
        # nothing the tier could have served.
        if self._admitted_at.get(req_id) == step or not self.lookup_enabled(request):
            return
        self._admitted_at[req_id] = step
        reusable = self._reusable(request)
        self._record_outcome(
            request, _ZeroHit(shared or 0), (), step,
            reusable=reusable, cap_fallback=cap_age is not None,
        )
        self._settle_miss(request, 0, cap_age, reusable)

    def admit(self, request: Request) -> PlanHit | None:
        t0 = time.perf_counter()
        try:
            return self._admit(request)
        finally:
            self.lookup_seconds += time.perf_counter() - t0

    def _admit(self, request: Request) -> PlanHit | None:
        """Resolve and lease this request's plan for one admission attempt.

        None: wait (a coalesced promotion or a draining old incarnation).
        The pending lease is released by ``update_state_after_alloc`` (on
        admission) or ``release_pending`` (end of the pass), never later.
        """
        s = self.sched
        req_id = request.request_id
        self.release(req_id)
        self._zero_hit.pop(req_id, None)
        self._fallback_pass.discard(req_id)  # set again by this pass's fallback
        if req_id in s._abandoned_reqs_to_load:
            return None
        # A pass of an admission already waiting for a restore slot (that wait
        # has no pass bound) is the same lookup: it counts no lookup, miss or
        # census retry again, so the counters follow admissions, not wait
        # length. The plan is still re-resolved.
        rewait = req_id in self._cap_waits or req_id in self._cap_fallback
        # So is a pass retried after ``allocate_slots``
        # refused the plan for GPU room: ``lookups`` counts once per logical
        # admission (hits count at the hand-off, the miss is settled at the
        # admission). The census still sees the retry (``lookup_retries``,
        # the attempts counter).
        if not (rewait or req_id in self._looked_up):
            s.tier_stats.lookups += 1
            self._looked_up.add(req_id)
        level = self.fallback.get(req_id, UNION)
        lease = None
        plan = None
        cached = self._cached_plan(request, level)
        for attempt in range(MAX_RESOLVE_ATTEMPTS):
            if attempt == 0 and cached is not None:
                plan = cached
                self.counters.plan_cache_hits += 1
            else:
                plan = self.resolve(request, gpu_only=level >= GPU_ONLY)
                self._remember(request, level, plan)
            if plan.boundary == 0:
                break
            # Validate, but take no refs: within one pass nothing reclaims a
            # source before allocate_slots adopts the GPU pages (touched
            # before any draw) and the hand-off pins the host sources; a held
            # ref would only re-order the free queue on every retried pass.
            lease = PrefixLease(plan, self.version, self._epoch)
            if lease.valid():
                break
            lease = None
            self.counters.plan_revalidations += 1
        assert plan is not None
        if s.tier_runtime is not None:
            s._census().lookup(request, plan.gpu_boundary, retry=not rewait)
            s.tier_runtime.on_lookup(req_id, request.num_tokens - 1 - plan.gpu_boundary)
        empty = tuple([] for _ in plan.groups)
        if lease is None:
            self._zero_hit[req_id] = plan.shared_prefix_boundary
            return PlanHit(empty, 0, 0, plan.shared_prefix_boundary, False)
        restore = self.restore_slots(plan)
        # A follower whose host sources an
        # in-flight promotion already reads needs no restore slot of its own;
        # it coalesces BEFORE the cap gate, so a full cap never recomputes it.
        if restore and self._coalesce(req_id, restore):
            lease.release()
            return None
        cap = self._restore_cap()
        if restore and cap is not None and not cap.can_start():
            # The cap is full. Wait without a
            # lease (the same contract as a coalesced promotion: None = try
            # again next pass, nothing pinned, nothing reserved) while the
            # restores in flight make progress; a slot frees when one retires
            # (``on_load_retired`` -> ``cap.done``). Only a stalled DMA (no
            # batch commit or retire for the stall timeout) falls back to the
            # GPU-only plan, and that recompute is the ``restore_cap_wait`` miss.
            lease.release()
            if self._cap_wait_progressing(req_id):
                self.counters.restore_cap_wait += 1
                return None
            waited = self._cap_waits.pop(req_id, None)
            if req_id not in self._cap_fallback:
                # Once per admission: a fallback whose GPU-only admission
                # failed and waits again is the same fallback.
                # Whether it is a ``restore_cap_wait`` miss is settled at the
                # admission, from the plan admitted (a fallback
                # refused for GPU room can still restore on a later pass).
                # Age = steps since the stall clock started.
                self.counters.restore_cap_fallback += 1
                self._cap_fallback[req_id] = (
                    s._sched_step - waited[1] if waited is not None else 0
                )
            # What this pass returns (plan or zero hit) is the fallback.
            self._fallback_pass.add(req_id)
            plan = self.resolve(request, gpu_only=True)
            lease = PrefixLease(plan, self.version, self._epoch)
            # An invalid GPU-only lease recomputes from token 0.
            if plan.boundary == 0 or not lease.valid():
                self._zero_hit[req_id] = plan.shared_prefix_boundary
                return PlanHit(empty, 0, 0, plan.shared_prefix_boundary, False)
            restore = ()
        # Lease the plan's host sources for this admission.
        # The ref takes each row out of the host free queue, so the walker's
        # eviction (``host_evict_order`` skips referenced rows) cannot take
        # the unit this request is about to restore while the allocation
        # makes GPU room. Released by ``_unpin`` (release / wait / forget /
        # end of pass) or just before ``_issue`` pins the sources itself.
        host_pins: list = []
        if restore:
            seen_rows: set[int] = set()
            for slot in restore:
                src = slot.source
                blk = src.block
                if src.tier is Tier.GPU or blk.is_null or id(blk) in seen_rows:
                    continue
                seen_rows.add(id(blk))
                src.pool.touch([blk])
                host_pins.append((src.pool, blk))
        pool = s._gpu_block_pool
        assert pool is not None
        restore_positions = {(slot.group_id, slot.block_index) for slot in restore}
        blocks = tuple(
            [
                pool.null_block
                if source is None or (gid, idx) in restore_positions
                or source.tier is not Tier.GPU
                else source.block
                for idx, source in enumerate(group)
            ]
            for gid, group in enumerate(plan.groups)
        )
        # The host-LRU touch moved to ``on_admitted`` (once
        # per admission, scoped to the requester's chain); a lookup attempt
        # that does not admit touches nothing.
        # Lease, for this pass only, the plan's GPU dependencies the
        # table does not adopt (proof witnesses), before allocate_slots draws
        # destinations: a draw can then never reclaim one, and the gate's
        # simulation sees them out of the free queue with the destinations,
        # the private tails and the request's growth. Table pages are adopted
        # (touched) before any draw; host sources are pinned at the hand-off
        # in this same pass, and no GPU draw reaches the host pool.
        pins = self._pin_proofs(plan) if restore else []
        if restore:
            assert self.kv_cache_manager is not None
            self.kv_cache_manager.restore_slots[req_id] = tuple(
                sorted(restore_positions)
            )
            # The consumer's private continuation blocks are drawn with the
            # destinations, inside the admission gate.
            tails = sum(self._slot_needs_private(slot) for slot in restore)
            if tails:
                self.kv_cache_manager.restore_tails[req_id] = tails
        self._pending[req_id] = _Pending(lease, plan, restore, pins, host_pins)
        return PlanHit(
            blocks, plan.boundary, plan.gpu_boundary,
            plan.shared_prefix_boundary, bool(restore),
        )

    def _pin_proofs(self, plan: PrefixHitPlan) -> list:
        data = {
            id(source.block)
            for group in plan.groups for source in group if source is not None
        }
        pins, seen = [], set()
        for source in plan.proof_sources:
            block = source.block
            if source.tier is not Tier.GPU or id(block) in data or id(block) in seen:
                continue
            seen.add(id(block))
            source.pool.touch([block])
            pins.append(block)
        return pins

    def _unpin(self, pending: _Pending) -> None:
        pins, pending.pins = pending.pins, []
        if pins:
            self.sched._gpu_block_pool.free_blocks(pins)
        self._unpin_host(pending)

    def _unpin_host(self, pending: _Pending) -> None:
        """Release the admission's host-row lease. Idempotent:
        the rows go back to the host free queue's tail (most recently used),
        which is also the demand touch a looked-up unit earns."""
        host_pins, pending.host_pins = pending.host_pins, []
        for pool, block in host_pins:
            pool.free_blocks([block])

    def release(self, request_id: str) -> None:
        pending = self._pending.pop(request_id, None)
        if pending is not None:
            self._unpin(pending)
            pending.lease.release()
        if self.kv_cache_manager is not None:
            self.kv_cache_manager.restore_slots.pop(request_id, None)
            self.kv_cache_manager.restore_tails.pop(request_id, None)
            self._release_tails(
                self.kv_cache_manager.restore_tail_blocks.pop(request_id, None)
            )

    def release_pending(self) -> None:
        """End of a scheduling pass: no lease outlives it."""
        for request_id in list(self._pending):
            self.release(request_id)

    def note_unfit(
        self, request: Request, *, shortfall: int, num_running: int
    ) -> None:
        """``allocate_slots`` refused this request's plan. Lower its boundary
        (GPU tier only, then cold) when waiting cannot help: nothing is
        running that could free blocks, or it needs more than the pool."""
        pending = self._pending.get(request.request_id)
        if pending is None or pending.plan.boundary == 0:
            return
        pool = self.sched._gpu_block_pool
        assert pool is not None
        needed = shortfall + pool.get_num_free_blocks()
        if num_running == 0 or needed > pool.num_gpu_blocks - 1:
            level = self.fallback.get(request.request_id, UNION) + 1
            self.fallback[request.request_id] = level
            if pending.restore:
                self.counters.restore_unfit += 1

    def use_cold(self, request: Request) -> bool:
        return self.fallback.get(request.request_id, UNION) > GPU_ONLY

    def on_admitted(self, request: Request, blocks) -> bool:
        """``update_state_after_alloc`` for a plan admission. False: not a
        plan admission."""
        req_id = request.request_id
        level = self.fallback.pop(req_id, UNION)
        self._plans.pop(req_id, None)
        self._looked_up.discard(req_id)  # the logical admission ends
        pending = self._pending.pop(req_id, None)
        if pending is None:
            self._note_zero_hit_admission(request, level)
            return False
        s = self.sched
        tail_blocks = None
        if self.kv_cache_manager is not None:
            self.kv_cache_manager.restore_slots.pop(req_id, None)
            self.kv_cache_manager.restore_tails.pop(req_id, None)
            tail_blocks = self.kv_cache_manager.restore_tail_blocks.pop(req_id, None)
        plan, restore = pending.plan, pending.restore
        external = plan.boundary - plan.gpu_boundary if restore else 0
        # Once per admission (request id + step): the scoped
        # host-LRU touch and the metrics outcome. A re-entered admission of
        # the same request in the same step records nothing twice.
        step = s._sched_step
        # Whether the restore is handed off, decided once and before anything
        # books it: the source still valid and a restore slot free (the
        # hit, the Prometheus hit tokens and the miss all follow it).
        cap = self._restore_cap()
        handed = bool(restore) and pending.lease.valid() and (
            cap is None or cap.can_start()
        )
        # What the admission adopts: the whole plan, unless a restore it
        # needed is not handed off (then only its GPU pages).
        adopted = plan.boundary if handed or not restore else plan.gpu_boundary
        # Read the cap-wait state before clearing it. A
        # ``restore_cap_wait`` miss only when the pass that produced this plan
        # took the cap fallback branch (its GPU-only plan, which restores
        # nothing): not a later pass that restores after all or resolves
        # anything else (e.g. after a host eviction).
        cap_age = self._cap_fallback.pop(req_id, None)
        if req_id not in self._fallback_pass:
            cap_age = None
        self._fallback_pass.discard(req_id)
        self._cap_waits.pop(req_id, None)
        if self._admitted_at.get(req_id) != step:
            self._admitted_at[req_id] = step
            self._touch_units(plan, request, restore=restore)
            reusable = self._reusable(request)
            self._record_outcome(
                request, plan, restore, step,
                reusable=reusable, cap_fallback=cap_age is not None,
            )
            self._settle_miss(request, adopted, cap_age, reusable)
        if s.tier_runtime is not None:
            census = s._census()
            if census.enabled:  # the per-group lists feed the census only
                cpu_blocks = tuple(
                    [slot.source.block for slot in restore if slot.group_id == gid]
                    for gid in range(len(plan.groups))
                )
                census.admission(req_id, external, (cpu_blocks, external, 0))
            # Prometheus ``tokens_total{event=hit}`` counts only a restore
            # that was handed off, like ``hit_tokens``: a refused hand-off
            # books no hit.
            s.tier_runtime.on_admission(
                req_id, external, hit_tokens=external if handed else 0
            )
        self._admissions += 1
        admission = (plan.epoch, req_id, self._admissions)
        self.accounting.admit(admission, plan)
        if not restore:
            self._release_tails(tail_blocks)
            self._unpin(pending)
            pending.lease.release()
            self.accounting.complete(admission, success=True)
            self.accounting.forget(admission)
            return True
        if not handed:
            # The source changed or the restore cap filled before hand-off.
            # Nothing is issued; the request recomputes and resolves again.
            from vllm.v1.simple_kv_offload.manager import (
                LoadRequestState,
                TransferMeta,
            )

            self._release_tails(tail_blocks)
            self._unpin(pending)
            pending.lease.release()
            self.accounting.complete(admission, success=False)
            self.accounting.forget(admission)
            s._reqs_to_load[req_id] = LoadRequestState(
                request=request, transfer_meta=TransferMeta([], []), failed=True
            )
            return True
        ids = blocks.get_block_ids()
        destinations = [ids[slot.group_id][slot.block_index] for slot in restore]
        self.counters.mixed_layouts += any(
            source is not None and source.tier is Tier.GPU
            for group in plan.groups
            for source in group
        )
        # The hand-off pins the host sources itself; drop the
        # admission lease first so each row carries one ref, not two.
        self._unpin_host(pending)
        try:
            self._issue(
                request, plan, restore, destinations, pending.lease,
                table_owned=True, admission=admission, tail_blocks=tail_blocks,
            )
            # Once per admission, from the plan actually handed off
            # (not a wait pass's plan the host lost since,
            # nor a hand-off refused above).
            s.tier_stats.hit_lookups += 1
            s.tier_stats.hit_tokens += external
        finally:
            # Validated at the hand-off with its witnesses held.
            self._unpin(pending)
        return True

    # -- the promotion transfer -------------------------------------------------
    def _issue(
        self, request, plan, restore, destinations, lease, *, table_owned,
        admission=None, tail_blocks=None,
    ) -> None:
        s = self.sched
        gpu = s._gpu_block_pool
        cpu = s.cpu_block_pool
        req_id = request.request_id
        assert gpu is not None and s._reqs_to_load.get(req_id) is None
        if len(set(destinations)) != len(destinations):
            lease.release()
            raise RuntimeError("host-tier restore has aliased destinations")
        try:
            if table_owned and s.tier_runtime is not None:
                s.tier_runtime.check_load_destinations(gpu, destinations, req_id)
        except BaseException:
            lease.release()
            raise
        members = []
        sources = []
        for slot, dst in zip(restore, destinations):
            src = slot.source.block
            sources.append(src)
            members.append(
                kv_host_tier_lifecycle.RestoreMember(
                    src.block_id,
                    dst,
                    src.block_hash,
                    src.block_hash_num_tokens,
                    (
                        src.block_hash,
                        frozenset(cpu.cached_block_hashes_by_block.get(src.block_id, ())),
                    ),
                    private=slot.private_tail,
                )
            )
        # The plan lease hands its source refs to the transfer.
        cpu.touch(sources)
        if table_owned:
            gpu.touch([gpu.blocks[bid] for bid in destinations])
        lease.release()
        key = ("plan", plan.epoch, req_id, self._admissions)
        self._inflight[key] = _Inflight(self._sources(restore))
        cap = self._restore_cap()
        if cap is not None:
            cap.start()
        restore_lease = kv_host_tier_lifecycle.RestoreLease(
            s.cache_epoch, key, tuple(members)
        )
        s._promotions[key] = req_id
        from vllm.v1.simple_kv_offload.manager import LoadRequestState, TransferMeta

        s._reqs_to_load[req_id] = LoadRequestState(
            request=request,
            transfer_meta=TransferMeta(
                list(destinations),
                [src.block_id for src in sources],
                cache_epoch=s.cache_epoch,
            ),
            lease=restore_lease,
        )
        self._plan_loads[req_id] = (key, admission)
        if table_owned:
            self.reserve_private_tails(req_id, members, tail_blocks)
        else:
            self._release_tails(tail_blocks)
        self._unissued[req_id] = list(
            zip(destinations, [src.block_id for src in sources])
        )
        self.counters.restore_admitted += 1
        tokens = plan.boundary - plan.gpu_boundary
        if s.tier_runtime is not None:
            s._census().pending_loads[req_id] = (tokens, len(sources))
        s.tier_stats.loaded_tokens += tokens
        s.tier_stats.loaded_blocks += len(sources)

    # -- credit-bounded issue -------------------------------------
    def dma_blocks(self, state) -> int:
        """Copies this load has issued and not completed (DMA credits)."""
        req_id = state.request.request_id
        if req_id in self._plan_loads:
            return self._issued.get(req_id, 0)
        if state.load_event is not None and state.load_event >= 0:
            return len(state.transfer_meta.cpu_block_ids)
        return 0

    def begin_tick(self) -> None:
        """Credits left for promotion batches this tick. The whole manifest
        is retained by transfer refs from admission; credits only bound what
        is issued, so a unit larger than ``P`` progresses batch by batch."""
        wb = self.sched.writeback
        if wb is None:
            self._tick_budget = 1 << 62
            return
        # Credits never exceed P (one at least, so a restore can progress).
        # While a restore has unissued rows the walker leaves it a share of
        # P (``waiting_for_credit``), so batches keep that size as stores
        # complete without the total ever passing P.
        in_flight = sum(self._issued.values())
        cap = max(wb.credits_cap, 1)
        # The restore direction has its
        # own P_r; demotions in flight (``wb.in_flight``) no longer reduce it.
        self._tick_budget = cap - in_flight

    def waiting_for_credit(self) -> int:
        """Credits the walker must leave free for admitted restores."""
        unissued = sum(len(pairs) for pairs in self._unissued.values())
        if not unissued:
            return 0
        wb = self.sched.writeback
        cap = max(wb.credits_cap, 1) if wb is not None else 1
        return max(0, min(cap, unissued) - sum(self._issued.values()))

    def end_tick(self) -> None:
        """After this tick's loads are issued: the credit account counts
        them (``credits``/``credits_peak`` include H2D batches)."""
        s = self.sched
        wb = s.writeback
        if wb is None:
            return
        loads = (*s._reqs_to_load.values(), *s._abandoned_reqs_to_load.values())
        wb.set_promote_dma(sum(
            self.dma_blocks(state)
            for state in loads
            if state.transfer_meta is not None
            and state.load_event is not None
            and state.load_event >= 0
        ))

    def next_batch(self, request_id: str) -> tuple[list[int], list[int]] | None:
        """None: not a plan load (the caller issues it whole). Empty lists:
        nothing to issue this tick."""
        pending = self._unissued.get(request_id)
        if pending is None:
            return None
        budget = self._tick_budget
        n = min(len(pending), max(0, budget))
        if n == 0:
            return [], []
        batch, self._unissued[request_id] = pending[:n], pending[n:]
        self._tick_budget -= n
        # Restore rows issued feed ``credits_r_issued``.
        wb = self.sched.writeback
        if wb is not None:
            wb.note_restore_issued(n)
        # The batch's issue step and time, for the
        # restore latency EWMAs behind the stall timeouts.
        self._note_batch_issued(request_id, n)
        self._issued[request_id] = n
        self._batches[request_id] = self._batches.get(request_id, 0) + 1
        self._final_issued[request_id] = not self._unissued[request_id]
        cpu = [c for _, c in batch]
        self._read.setdefault(request_id, []).extend(cpu)
        return [g for g, _ in batch], cpu

    def read_sources(self, request_id: str) -> list[int] | None:
        """The host rows a plan load's issued batches read (None: not a plan
        load; the caller then invalidates the whole transfer). A failed
        restore un-keys exactly these, and only while each still holds the
        key it was read under. A cancel or reset holds the
        whole manifest until the load retires, so a read row cannot be
        reused before its failure lands; the key check stays as the guard."""
        if request_id not in self._plan_loads:
            return None
        s = self.sched
        state = s._reqs_to_load.get(request_id) or s._abandoned_reqs_to_load.get(
            request_id
        )
        keys = {}
        if state is not None and state.lease is not None:
            keys = {m.cpu_id: m.source_keys[0] for m in state.lease.members}
        host = s.cpu_block_pool
        return [
            cpu_id for cpu_id in self._read.get(request_id, ())
            if cpu_id in keys and host.blocks[cpu_id].block_hash == keys[cpu_id]
        ]

    def batch_committed(self, request_id: str, state, event: int, failed: bool) -> bool:
        """The load transaction of one credit batch resolved on every rank
        (``load_txn_key = (epoch, event, 0)``). True for a COMMIT of a
        non-final batch: the next batch issues on the next tick and the
        receive goes on. False lets the caller finish the load: the last
        batch committed, any batch unwound (the request recomputes; nothing
        later is issued), or the consumer was cancelled (nothing later is
        issued; nothing else is in flight, batches are sequential)."""
        if request_id not in self._plan_loads or state.load_event != event:
            return False
        # A batch resolved on every rank is restore
        # progress for the cap waiters, whatever its outcome.
        self._note_restore_progress(request_id)
        # The unissued remainder stays recorded: a cancelled restore that
        # stops early is incomplete (``on_load_retired``), never complete.
        if failed or state.finished or not self._unissued.get(request_id):
            return False
        self._issued.pop(request_id, None)
        reqs = self.sched._load_event_to_reqs.get(event)
        if reqs is not None:
            if request_id in reqs:
                reqs.remove(request_id)
            if not reqs:
                self.sched._load_event_to_reqs.pop(event, None)
        state.load_event = None
        return True

    def release_for_key(self, key) -> None:
        """The last follower of promotion ``key`` is gone: a finished leader
        whose staging was kept for it is cut back to its outstanding batch."""
        s = self.sched
        for rid, entry in list(self._plan_loads.items()):
            if entry[0] != key:
                continue
            state = s._reqs_to_load.get(rid)
            if state is not None and state.finished and state.load_event is not None:
                self.release_unoutstanding(rid, state)

    def release_unoutstanding(self, request_id: str, state, *, keep_complete: bool = True) -> None:
        """Keep the whole manifest leased until any active DMA has retired.

        Epoch validation below prevents stale publication. Holding the manifest
        avoids per-batch quarantine state while DMA may still write its rows.
        """
        if request_id not in self._plan_loads or state.transfer_meta is None:
            return
        key = self._plan_loads[request_id][0]
        if (
            keep_complete
            and self._final_issued.get(request_id)
            and self.sched._promotion_waiters.get(key)
        ):
            return
        self._cut.add(request_id)
        if self._unissued.get(request_id):
            self._unissued[request_id] = []
        self._release_tails(self._tails.pop(request_id, None))

    def withhold_receive(self, request_id: str, state, failed: bool) -> bool:
        """The worker reports a receive per batch event. Withhold it from the
        scheduler until every issued batch landed and the last pair was
        issued (or the consumer was cancelled, which issues nothing more). A
        failed receive is never withheld. Batches advance only on their
        per-rank transaction (``batch_committed``), never on a receive."""
        if request_id not in self._plan_loads or failed:
            return False
        seen = self._receives.get(request_id, 0) + 1
        self._receives[request_id] = seen
        if seen < self._batches.get(request_id, 0):
            return True
        return not (self._final_issued.get(request_id) or state.finished)

    # -- private tails -----------------------------------------
    def _needs_private(self, member) -> bool:
        """A restored recurrent state, or the page holding a boundary that is
        not page-aligned: its consumer would write it in place. Such a
        destination stays immutable staging; the consumer continues on a
        private copy.

        The boundary page is the slot's position (``private``), not its
        source's coverage: the union lookup may serve a mid-page boundary
        from a later, page-end FA key, whose destination is then
        published under that full-page key."""
        from vllm.v1.core.kv_cache_utils import get_group_id
        from vllm.v1.kv_cache_interface import MambaSpec

        manager = self.coordinator().single_type_managers[get_group_id(member.key)]
        return (
            member.private
            or isinstance(manager.kv_cache_spec, MambaSpec)
            or member.num_tokens % manager.block_size != 0
        )

    def _slot_needs_private(self, slot: PrefixSlot) -> bool:
        """``_needs_private`` of the member this host slot becomes."""
        from vllm.v1.kv_cache_interface import MambaSpec

        manager = self.coordinator().single_type_managers[slot.group_id]
        tokens = slot.source.block.block_hash_num_tokens
        return (
            slot.private_tail
            or isinstance(manager.kv_cache_spec, MambaSpec)
            or (tokens is not None and tokens % manager.block_size != 0)
        )

    def reserve_private_tails(self, request_id: str, members, drawn=None) -> None:
        """At hand-off, keep one GPU block per restored recurrent/partial tail
        for the consumer's private continuation. The admission drew them in
        ``allocate_slots`` under the admission gate (``drawn``); no
        block is drawn here. If they are missing the restore fails closed for
        those members: they are never published and its followers resolve
        again."""
        need = sum(self._needs_private(m) for m in members)
        drawn = list(drawn or ())
        if not need:
            self._release_tails(drawn)
            return
        if len(drawn) >= need:
            self._release_tails(drawn[need:])
            self._tails[request_id] = drawn[:need]
            return
        self._release_tails(drawn)
        self._tails[request_id] = None

    def _release_tails(self, tails) -> None:
        if tails:
            self.sched._gpu_block_pool.free_blocks(tails)

    def _privatize(self, request_id: str, state, tails) -> tuple[set[int], bool]:
        """Move the consumer onto private copies of its restored
        recurrent/partial tails. The staged destination keeps exactly the
        accepted-B content and is published; the consumer's table gets a
        reserved block that a GPU block copy (the CoW copy queue, run before
        the step's forward) fills from it. Nothing ever writes the staged
        block. Returns (privatized destination ids, closed): ``closed`` when
        some tail could not be privatized (not published, no followers)."""
        from vllm.v1.core.kv_cache_utils import get_group_id

        from vllm.v1.kv_cache_interface import MambaSpec

        need = [m for m in state.lease.members if self._needs_private(m)]
        kvm = self.kv_cache_manager
        if not need:
            self._release_tails(tails)
            return set(), False
        if not tails or kvm is None:
            self._release_tails(tails)
            return set(), True
        pool = self.sched._gpu_block_pool
        managers = kvm.coordinator.single_type_managers
        spare = list(tails)
        privatized: set[int] = set()
        for member in need:
            manager = managers[get_group_id(member.key)]
            table = manager.req_to_blocks.get(request_id, [])
            staged = pool.blocks[member.gpu_id]
            index = next((i for i, b in enumerate(table) if b is staged), None)
            if index is None or not spare:
                continue
            private = spare.pop()
            # The table's ref on the staged block becomes the copy's retention
            # ref (released by the scheduler once the copy ran); the reserved
            # block's ref becomes the table's, plus one retention ref.
            table[index] = private
            manager._pending_cow_copies.append((staged, private))
            private.ref_cnt += 1
            if isinstance(manager.kv_cache_spec, MambaSpec):
                # The staged block holds the state's key at B; the private
                # continuation is never keyed there nor CoW'd from.
                manager.restored_private_tails[request_id] = member.num_tokens
            privatized.add(member.gpu_id)
            self.counters.private_tails += 1
        self._release_tails(spare)
        return privatized, len(privatized) < len(need)

    def on_load_retired(self, request_id: str, state) -> None:
        """Before transfer refs drop, publish completed pages under host keys."""
        # A load cancelled before its first batch issued never reaches
        # ``on_load_done``.
        forget = getattr(self.sched.tier_runtime, "forget_load", None)
        if forget is not None:
            forget(request_id)
        entry = self._plan_loads.pop(request_id, None)
        remaining = self._unissued.pop(request_id, None)
        # Cut short by a cancel/reset: incomplete whatever is left.
        remaining = remaining or request_id in self._cut
        self._cut.discard(request_id)
        self._issued.pop(request_id, None)
        for book in (self._batches, self._receives, self._final_issued, self._read):
            book.pop(request_id, None)
        tails = self._tails.pop(request_id, None)
        if entry is None:
            self._release_tails(tails)
            return
        key, admission = entry
        self._inflight.pop(key, None)
        s = self.sched
        cap = self._restore_cap()
        if cap is not None:
            cap.done()
        # A retired load (complete, failed or cut) is
        # restore progress and frees a slot for the cap waiters.
        self._note_restore_progress(None)
        if remaining:
            # Incomplete (cancelled between or during batches): destinations
            # past the issued batches were never written. Nothing of it is
            # published, by this path or by the coalesced-waiter path
            # (publish_cancelled); its followers resolve again.
            s._promotion_waiters.pop(key, None)
        success = (
            not state.failed
            and not remaining
            and state.load_event is not None
            and state.load_event >= 0
            and state.lease is not None
            and state.lease.epoch == s.cache_epoch
        )
        if admission is not None:
            self.accounting.complete(admission, success=success)
            self.accounting.forget(admission)
        # Close the admission outcome (restore completed or
        # failed -> recomputed) for the metrics sink.
        sink = getattr(s.tier_runtime, "outcome_sink", None)
        retired = getattr(sink, "restore_retired", None)
        if callable(retired):
            retired(request_id, bool(success))
        # A cancelled consumer's restore is published only for its coalesced
        # waiters, by the reset path (RestoreLease.publish_cancelled).
        if not success or state.finished:
            self._release_tails(tails)
            return
        privatized, closed = self._privatize(request_id, state, tails)
        if closed:
            s._promotion_waiters.pop(key, None)
        pool = s._gpu_block_pool
        assert pool is not None
        for member in state.lease.members:
            block = pool.blocks[member.gpu_id]
            if (
                self._needs_private(member)
                and member.gpu_id not in privatized
            ) or block.block_hash is not None or block.ref_cnt == 0:
                continue
            primary, aliases = member.source_keys
            if primary is None:
                continue
            # Tier-aware: the quiesced restore is stamped ready.
            pool.publish_restored(
                primary, block, num_tokens=member.num_tokens,
                host_block_id=member.cpu_id,
            )
            for alias in sorted(aliases):  # Total key order
                pool.publish_restored(
                    alias, block, num_tokens=None, host_block_id=member.cpu_id
                )
            self.counters.restore_published_blocks += 1

    def log_counters(self) -> dict[str, int]:
        """Cumulative ``name=int`` counters for the ``SimpleCPU tier plans:``
        log line and the census events: plan admission, the
        union benefit (B, G, B - G), backing provenance and transactions."""
        s = self.sched
        out = {f.name: getattr(self.counters, f.name) for f in fields(PlanCounters)}
        for cause in self.MISS_CAUSES:
            out[f"miss_{cause}"] = self.miss_causes.get(cause, 0)
        out["lookup_evict_us"] = int(self.lookup_seconds * 1e6)
        a = self.accounting
        out["union_hit_tokens"] = a.union_hit_tokens
        out["gpu_only_tokens"] = a.gpu_hit_tokens
        out["union_gain"] = a.union_gain_tokens
        out["restored_requests"] = a.restored_requests
        out["restored_host_blocks"] = a.restored_host_blocks
        pool = s._gpu_block_pool
        tier = pool.tier if pool is not None else None
        for name in ("backing_unproven", "backing_foreign", "host_superseded"):
            out[name] = getattr(tier.counters, name, 0) if tier is not None else 0
        for name in (
            "load_txn_failures", "store_txn_failures", "txn_report_mismatches",
            "load_stale_epoch_completions",
        ):
            out[name] = getattr(s, name, 0)
        return out

    def forget(self, request_id: str) -> None:
        self.release(request_id)
        self.fallback.pop(request_id, None)
        self._plans.pop(request_id, None)
        self._zero_hit.pop(request_id, None)
        # Left before admission: nothing recomputed, no miss.
        self._looked_up.discard(request_id)
        self._cap_waits.pop(request_id, None)
        self._cap_fallback.pop(request_id, None)
        self._fallback_pass.discard(request_id)
        self._batch_issue_step.pop(request_id, None)
        self._batch_issue_time.pop(request_id, None)
        self._admitted_at.pop(request_id, None)
        for inflight in self._inflight.values():
            inflight.waiters.discard(request_id)

    def reset(self) -> None:
        self.release_pending()
        self.fallback.clear()
        self._plans.clear()
        self._zero_hit.clear()
        # The reset ends each logical lookup: a waiting request's next lookup
        # counts again; its miss is still settled once, at its admission.
        self._looked_up.clear()
        self._cap_waits.clear()
        self._cap_fallback.clear()
        self._fallback_pass.clear()
        self._batch_issue_step.clear()
        self._batch_issue_time.clear()
        self._admitted_at.clear()
        self._inflight.clear()
