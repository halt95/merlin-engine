# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Host-RAM KV tier: the shared contracts.

Import-only. This module defines the four frozen contracts the host tier's
components exchange and the counter names they share, so that the eviction
(units/policy), copy-scheduler (copyq/writeback) and metrics
(census/runtime) parts compile against one definition:

* ``CopyOffer``: one unit of copy work offered to the copy scheduler
  (grace-held finished turn, preempt grace, slack-only running write-through,
  backing repair, or the reactive fallback), with the source identity the
  transaction must re-validate at ack (``generation`` per block).
* ``BoundaryManifest``: the exact restore boundary a request's unit needs,
  derived from ``UnitTable.manifest`` and carried through finish/preempt and
  supersede-at-publish (the predecessor stays restorable until its successor
  publishes).
* ``EvictionPlan``: defined by the units/policy module in
  ``kv_host_tier_units``; re-exported here lazily so this module never
  imports engine code at load time.
* ``AdmissionOutcome``: recorded ONCE per admitted request (never per lookup
  retry) with the historically reusable tokens from request history,
  not from the census union, so E = sum(adopted) / sum(reusable) cannot be
  deflated by content the pipeline lost.

Also ``LifecycleHooks`` / ``CopyScheduler`` / ``EvictionPolicy`` /
``OutcomeSink`` protocols, which the hook lines in ``kv_cache_manager``,
``scheduler`` and ``simple_kv_offload.manager`` call through ``getattr``
(tier off = no object installed = byte-identical behaviour).

Nothing here changes the demotion path: every copy still goes
``begin_demote`` -> txn -> all-rank ack -> ``demote_still_valid`` -> publish.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal, Protocol, Sequence, runtime_checkable

if TYPE_CHECKING:
    from vllm.merlin.kv_host_tier_units import EvictionPlan, RestoreUnit
    from vllm.v1.core.kv_cache_utils import BlockHashWithGroupId, KVCacheBlock
    from vllm.v1.request import Request

__all__ = [
    "CopyLane",
    "CopyOffer",
    "BoundaryManifest",
    "AdmissionOutcome",
    "EvictionPlan",
    "GraceRelease",
    "LifecycleHooks",
    "CopyScheduler",
    "EvictionPolicy",
    "OutcomeSink",
    "MISS_CAUSES_V3",
    "COUNTERS_V3",
    "CENSUS_EVENTS_V3",
    "CENSUS_GAUGES_V3",
    "SERIES_FIELDS_V3",
]


# -- copy work ---------------------------------------------------------------

class CopyLane(Enum):
    """Why a copy is offered; also its priority class (highest first)."""

    REACTIVE = "reactive"  # the walker's reclaim-window victim (fallback)
    GRACE = "grace"  # a grace-held finished turn, SessionYield order
    PREEMPT = "preempt"  # a preempted request under its bounded grace
    REPAIR = "repair"  # host backing was revoked; the GPU twin is dirty again
    RUNNING = "running"  # lane 1: slack-only write-through of a running request


# Lanes that may displace published host-only content when they need rows.
# RUNNING never does (SGLang's host > device rule, applied per step); REPAIR
# and RUNNING take idle rows or disposable duplicates only.
DISPLACING_LANES: frozenset[CopyLane] = frozenset(
    {CopyLane.REACTIVE, CopyLane.GRACE, CopyLane.PREEMPT}
)


@dataclass(frozen=True)
class CopyOffer:
    """One unit of copy work: the GPU members of one restore unit (or one
    lone block) that lack a host copy, offered at ``step``.

    ``generations[i]`` is ``BlockTierState.generation[blocks[i].block_id]``
    at offer time; the transaction re-checks it at ack (``demote_still_valid``)
    so a reallocated source never publishes. ``yield_rank`` is the position in
    ``SessionYield._units()`` order (0 = yields next); lanes without a yield
    position use ``None``. ``bytes_per_rank`` is the host arena bytes this
    offer needs on one rank (rows x host_block_bytes). ``deadline_steps`` is
    the scheduler's estimate of steps before the first source is reclaimed
    (None = not at risk). ``txn_atomic`` asks the scheduler to issue every
    member in one transaction (a complete manifest lands or nothing does).
    """

    offer_id: int
    lane: CopyLane
    step: int
    blocks: tuple["KVCacheBlock", ...]
    generations: tuple[int, ...]
    keys: tuple["BlockHashWithGroupId", ...]
    request_id: str | None = None
    uid: int | None = None  # RestoreUnit.uid when the unit is recorded
    yield_rank: int | None = None
    bytes_per_rank: int = 0
    deadline_steps: int | None = None
    txn_atomic: bool = True

    def __post_init__(self) -> None:
        if not (len(self.blocks) == len(self.generations) == len(self.keys)):
            raise ValueError("CopyOffer: blocks, generations and keys must align")

    @property
    def rows(self) -> int:
        return len(self.blocks)


# -- restore boundaries -------------------------------------------------------

@dataclass(frozen=True)
class BoundaryManifest:
    """The exact restore boundary of one request at one point in its life.

    Built from ``UnitTable.manifest(request)`` (``UnitManifest``), plus the
    lifecycle fields the copy scheduler needs: ``predecessor_uid`` is the unit this
    boundary supersedes (kept restorable until this one publishes:
    supersede-at-publish); ``version`` increments on every re-derivation for
    the same request so a stale manifest from before a CoW move or a reset is
    never published; ``host_complete`` is the directory's view at build time.
    """

    request_id: str
    boundary: int
    required: frozenset["BlockHashWithGroupId"]
    endpoints: frozenset["BlockHashWithGroupId"]
    optional: frozenset["BlockHashWithGroupId"] = frozenset()
    uid: int | None = None
    predecessor_uid: int | None = None
    version: int = 0
    host_complete: bool = False
    # finish / preempt / grace / abort / reset (what produced this manifest)
    event: Literal["finish", "preempt", "grace", "abort", "reset"] = "finish"

    @property
    def keys(self) -> frozenset["BlockHashWithGroupId"]:
        return self.required | self.optional

    @classmethod
    def from_unit(
        cls, request_id: str, unit: "RestoreUnit", *, event: str = "finish",
        predecessor_uid: int | None = None, version: int = 0,
        host_complete: bool = False,
    ) -> "BoundaryManifest":
        return cls(
            request_id=request_id,
            boundary=unit.boundary,
            required=unit.required,
            endpoints=unit.endpoints,
            optional=unit.optional,
            uid=unit.uid,
            predecessor_uid=predecessor_uid,
            version=version,
            host_complete=host_complete,
            event=event,  # type: ignore[arg-type]
        )


# -- once-per-admission outcome ------------------------------------------------

# Every miss cause an ``AdmissionOutcome`` can carry; the planner books its own
# per-cause counts for the subset in ``TierPlans.MISS_CAUSES``. The census and
# runtime code report counts per cause.
MISS_CAUSES_V3: tuple[str, ...] = (
    # admission misses
    "missed_copy", "race", "host_evicted", "never_fit", "skipped_tiny",
    # restore-slot, grace and backing misses
    "restore_cap_wait",  # a valid host plan waited for a restore slot
    "yield_incomplete",  # yielded before its grace copy acked (destructive break)
    "lane2_late",  # finish/preempt copy issued but reclaimed before ack
    "backing_revoked",  # host twin evicted as a duplicate; repair not yet done
    "not_admitted",  # scan-resistance admission dropped the copy
    "host_busy",  # fits the cache, but every eligible row was leased
    "first_turn",  # nothing reusable (rho = 0); never counts against E
    "unknown",
)


@dataclass(frozen=True)
class AdmissionOutcome:
    """Recorded exactly once per admitted request, when its plan is adopted
    (``PrefixPlanner.on_admitted``), never on a lookup retry or a wait.

    ``reusable_tokens`` (R_i) comes from request history, independent of the
    tier: the longest exact prefix this request shares with any previously
    computed request of the same session/chain, rounded to the boundary grid
    the layout can restore at. It is what E's denominator sums. ``gpu_tokens``
    and ``host_tokens`` are the tokens actually adopted from each tier (A_i).
    ``restore_completed`` is filled by the sink when the load retires
    (None while in flight; False = failed, recomputed).
    """

    request_id: str
    step: int
    prompt_tokens: int
    reusable_tokens: int
    gpu_tokens: int
    host_tokens: int
    shared_prefix_tokens: int = 0
    restore_requested: bool = False
    restore_completed: bool | None = None
    miss_cause: str | None = None  # one of MISS_CAUSES_V3 when adopted < reusable
    first_turn: bool = False
    # Diagnostics (no effect on E): the census union at admission and the
    # plan's gpu boundary, for the split_lost / T* reports.
    union_tokens: int | None = None
    gpu_boundary: int | None = None

    @property
    def adopted_tokens(self) -> int:
        return self.gpu_tokens + self.host_tokens

    @property
    def lost_tokens(self) -> int:
        return max(0, self.reusable_tokens - self.adopted_tokens)


# -- protocols -----------------------------------------------------------------

GraceRelease = Literal["hit", "overflow", "yield", "abort", "reset"]


@runtime_checkable
class LifecycleHooks(Protocol):
    """What the engine's hook lines call (installed only with the tier on).

    ``kv_cache_manager``: ``grace_taken`` after ``_take_grace_pin``,
    ``grace_released`` before the grace pin's blocks are freed or yielded.
    ``scheduler``: ``preempted`` inside ``_preempt_request`` before
    ``_free_request_blocks``; ``finished`` from the connector's
    ``request_finished_all_groups``. ``running_ready`` is the lane-1 input:
    full, hashed, all-rank-ready FA blocks a running request referenced this
    step (the copy scheduler dedupes against host backing itself).
    """

    def grace_taken(
        self, request: "Request", blocks: Sequence["KVCacheBlock"], step: int
    ) -> None: ...

    def grace_released(self, request_id: str, reason: GraceRelease, step: int) -> None: ...

    def finished(self, request: "Request", manifest: BoundaryManifest | None, step: int) -> None: ...

    def preempted(self, request: "Request", manifest: BoundaryManifest | None, step: int) -> None: ...

    def running_ready(
        self, request: "Request", blocks: Sequence["KVCacheBlock"], step: int
    ) -> None: ...


@runtime_checkable
class CopyScheduler(Protocol):
    """The copy queue (``kv_host_tier_copyq``) as the walker sees it."""

    def offers(self, step: int, *, rows_budget: int, bytes_budget: int) -> list[CopyOffer]:
        """Offers to issue this step, highest priority first, within the
        demotion direction's row and byte credits."""
        ...

    def on_issued(self, offer: CopyOffer, txn_id: int) -> None: ...

    def on_acked(self, offer: CopyOffer, ok: Sequence[bool]) -> None: ...

    def revoke_backing(self, blocks: Sequence["KVCacheBlock"], step: int) -> None:
        """Host twins of ``blocks`` were evicted: the GPU content is dirty
        again and eligible for a REPAIR offer (never an immediate recopy)."""
        ...

    def host_complete(self, request_id: str) -> bool:
        """Host-aware yield asks this before yielding a grace."""
        ...


@runtime_checkable
class EvictionPolicy(Protocol):
    """Host victim selection (``kv_host_tier_units`` / ``kv_host_tier_policy``).

    ``plan`` is pure selection: nothing is unkeyed until ``apply``. The order
    is idle ownerless rows -> twins held by a running request -> ordinary LRU
    units (exclusive rows only; shared rows stay while any owner remains) ->
    twins about to leave the GPU (grace / free-queue head), last. A lane that
    may not displace published content gets a plan over the first two
    classes only.
    """

    def plan(
        self, need_rows: int, step: int, *, displace_published: bool
    ) -> "EvictionPlan": ...

    def apply(self, plan: "EvictionPlan") -> int:
        """Evict; returns rows made idle."""
        ...


@runtime_checkable
class OutcomeSink(Protocol):
    def record(self, outcome: AdmissionOutcome) -> None: ...

    def restore_retired(self, request_id: str, completed: bool) -> None: ...


# -- reserved counter names ----------------------------------------------------

# Walker / copy-scheduler counters (``WriteBack.counters`` names; the copy
# scheduler adds the fields, the metrics read them). Existing names kept as-is are not repeated here.
COUNTERS_V3: tuple[str, ...] = (
    # copy lanes, rows issued per lane
    "copy_grace_rows", "copy_preempt_rows", "copy_running_rows",
    "copy_repair_rows", "copy_reactive_rows",
    # grace copies / yield
    "yield_incomplete", "yield_waited_steps", "grace_offers",
    "preempt_grace_taken", "preempt_grace_expired", "preempt_grace_shed",
    # backing repair
    "backing_revoked", "repair_pending", "repair_done",
    # finish-time completion
    "lane2_rows", "lane2_late",
    # per-direction credits (rows and bytes), peaks
    "credits_d_issued", "credits_r_issued",
    "bytes_d_inflight_peak", "bytes_r_inflight_peak",
    # assembly ledger (unique rows/bytes), replaces the MAX_ASSEMBLING count
    "assembly_unique_rows_peak", "assembly_unique_bytes_peak",
    # eviction
    "evict_exclusive_rows", "evict_shared_kept", "evict_skipped_leased",
    "evict_running_dup_rows", "evict_imminent_dup_rows",
    "probation_admitted", "probation_bypassed",
    "supersede_deferred", "constrained_successor",
    # plans
    "touch_scoped", "touch_skipped_offchain", "restore_cap_wait",
    "admission_outcomes",
)

# Census additions (``kv_host_tier_census.GAUGES`` / ``EVENTS``).
CENSUS_GAUGES_V3: tuple[str, ...] = (
    "host_rows_published", "host_rows_provisional", "host_rows_idle",
    "host_rows_leased", "host_rows_duplicate_running",
    "host_rows_duplicate_imminent", "grace_pins_active",
)
CENSUS_EVENTS_V3: tuple[str, ...] = (
    "split_lost_gpu_ckpt_host_fa", "split_lost_host_ckpt_gpu_fa",
    "restore_cap_wait", "yield_incomplete", "lane2_late",
    "backing_revoked", "outcomes_recorded", "outcomes_first_turn",
    "reusable_tokens", "adopted_gpu_tokens", "adopted_host_tokens",
)

# Fields the 5-minute series line must carry so a whole-host freeze cannot
# hide: the series parser and the test scorer read these names.
SERIES_FIELDS_V3: tuple[str, ...] = (
    "stored_blocks", "demoted_units", "unit_lost", "demote_missed_running",
    "demote_missed_escape", "demote_host_busy", "demote_dropped_host_full",
    "restored_requests", "yield_incomplete", "restore_cap_wait",
    "lane2_late", "backing_revoked", "host_rows_idle", "host_rows_published",
    "credits_d_issued", "credits_r_issued", "select_us",
)


def __getattr__(name: str) -> Any:
    # Lazy re-export: ``EvictionPlan`` is owned by kv_host_tier_units.
    if name == "EvictionPlan":
        from vllm.merlin.kv_host_tier_units import EvictionPlan

        return EvictionPlan
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
