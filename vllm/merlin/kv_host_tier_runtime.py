# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Host-RAM KV tier, scheduler-side runtime.

Built by SimpleCPUOffloadScheduler only when the generated tier is active
(``kv_host_tier.is_active``); every other SimpleCPUOffload user gets
``tier_runtime = None`` and the upstream behaviour. It owns four things:

1. Store keys (partial blocks). A store copies a GPU block
   after the compute of the step that issued it, but upstream registers the
   host copy under whatever hashes the GPU block carries when the completion
   is processed. The fork re-keys hashed blocks in place: a partial tail is
   promoted to a full-block key (BlockPool.cache_full_blocks), a producer's
   live partial tail moves its key to a copy-on-write block
   (BlockPool.move_block_hashes), a transient Mamba checkpoint slot is
   evicted and re-keyed at another position. A copy taken before such a
   re-key holds the bytes of the OLD key; registering it under the new key
   would serve plausible nonsense (or, for a moved key, trip the completion's
   ``block_hash is not None`` assert). So the tier snapshots each source
   block's keys when the store is issued and registers the host copy only if
   the primary key is unchanged at completion, and then only under the
   secondary keys present at both points. Anything else is discarded (the
   host block goes back to the free list unhashed) and counted.
2. Load destinations. A restore must land only in blocks the loading request
   owns alone and the prefix cache cannot serve (allocated this step,
   caching delayed until the load lands). That is what the scheduler
   allocates; the tier checks it and fails loudly if it ever is not true.
3. Load failures (recompute). A host-to-GPU copy that fails is reported by the
   worker as a failed receive of its request (failed_recving, with the
   generated ``kv_load_failure_policy="recompute"``); the scheduler then
   recomputes the request from its local prefix-cache hit. Before that, the
   host blocks the failed load read are un-keyed, so the retry cannot hit
   them again.
4. Metrics: tier hit/miss/stored/evicted/load-failed tokens and the cached
   arena occupancy (blocks holding a key -- not the in-flight ``used``
   count), on the connector's existing Prometheus path.

Token counts are full-attention-group tokens (the coverage a lookup can hit);
other groups (Mamba state, folded PLE) count blocks.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from vllm.logger import init_logger
from vllm.merlin.kv_host_tier_iface import COUNTERS_V3
from vllm.merlin.kv_host_tier_census import (
    EVENTS,
    GAUGES,
    FlatStoreDetector,
    HistoryIndex,
    history_capacity,
    HostFlushDetector,
    OutcomeLedger,
    RestoreProgressDetector,
    TierCensus,
    efficiency,
    interval_delta,
)

if TYPE_CHECKING:
    from vllm.merlin.kv_host_tier_units import UnitTable
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.kv_cache_utils import BlockHashWithGroupId, KVCacheBlock

logger = init_logger(__name__)

# Store key snapshot of one source block: (primary key, secondary keys).
StoreKeys = tuple["BlockHashWithGroupId | None", frozenset]


class MetricName:
    TOKENS = "vllm:merlin_kv_host_tier_tokens_total"
    BLOCKS = "vllm:merlin_kv_host_tier_blocks_total"
    CACHED_BLOCKS = "vllm:merlin_kv_host_tier_cached_blocks"
    CAPACITY_BLOCKS = "vllm:merlin_kv_host_tier_capacity_blocks"
    CENSUS = "vllm:merlin_kv_host_tier_census"
    CENSUS_EVENTS = "vllm:merlin_kv_host_tier_census_events_total"
    RECLAIMS = "vllm:merlin_kv_host_tier_reclaims_total"
    WORKER = "vllm:merlin_kv_host_tier_worker_total"
    WORKER_COST = "vllm:merlin_kv_host_tier_copy_ms_per_block"


# Counter field -> (metric, event label).
_TOKEN_EVENTS = {
    "hit_tokens": "hit",
    "miss_tokens": "miss",
    "stored_tokens": "stored",
    "evicted_tokens": "evicted",
    "load_failed_tokens": "load_failed",
}
_BLOCK_EVENTS = {
    "stored_blocks": "stored",
    "evicted_blocks": "evicted",
    "load_failed_blocks": "load_failed",
    "discarded_rekeyed_blocks": "discarded_rekeyed",
}


@dataclass
class TierCounters:
    """Cumulative tier counters (drained as deltas into the stats payload).

    - hit_tokens / miss_tokens: at admission, the tokens past the local
      prefix-cache hit that the tier was asked for, split into those it
      served (accepted for loading) and those it did not have.
    - stored / evicted: host blocks that gained a key at store completion /
      lost it to LRU reuse of the host pool (a cache reset is not eviction).
    - load_failed: restores whose copy failed; their requests recompute.
    - discarded_rekeyed_blocks: host copies dropped because their source
      block was re-keyed while the copy was in flight (see module doc, 1).
    """

    hit_tokens: int = 0
    miss_tokens: int = 0
    stored_tokens: int = 0
    stored_blocks: int = 0
    evicted_tokens: int = 0
    evicted_blocks: int = 0
    load_failed_tokens: int = 0
    load_failed_blocks: int = 0
    load_failed_requests: int = 0
    discarded_rekeyed_blocks: int = 0
    # Host keys dropped by a coordinated reset (not evictions).
    reset_unkeyed_blocks: int = 0
    # Census observations never participate in allocation or lookup decisions.
    gpu_host_overlap: int = 0
    # keyed-union (raw residency, unfiltered) next to the usable ones
    keyed_union_units: int = 0
    keyed_union_tokens: int = 0
    restorable_units: int = 0
    restorable_tokens: int = 0
    orphan_blocks: int = 0
    # Orphans that are not recurrent checkpoints.
    orphan_unclassified: int = 0
    union_restorable_units: int = 0
    union_restorable_tokens: int = 0
    split_units: int = 0
    reclaim_step_max: int = 0
    in_flight_peak: int = 0
    readiness_observable: int = 0
    manifest_members: int = 0
    census_scans: int = 0
    manifest_dropped: int = 0
    reclaimed_blocks: int = 0
    collateral_blocks: int = 0
    commit_dropped_not_hashed: int = 0
    split_lost: int = 0
    split_lost_tokens: int = 0
    returns_cold: int = 0
    returns_gpu: int = 0
    returns_host: int = 0
    returns_mixed: int = 0
    checkpoint_replay: int = 0
    checkpoint_tail: int = 0
    checkpoint_both: int = 0
    checkpoint_unknown: int = 0
    restore_completed_requests: int = 0
    restore_completed_tokens: int = 0
    restore_completed_blocks: int = 0
    # Admitted restore attempts that fail, abort, or lose their plan/epoch.
    restore_failed: int = 0
    # A restore whose request went away: cancellation, not transfer failure.
    restore_cancelled: int = 0
    preemption_flushes: int = 0
    preempted_requests: int = 0
    lookup_retries: int = 0
    gpu_readiness_unknown_hits: int = 0
    # split_lost by census layout (counts and tokens).
    split_lost_gpu_ckpt_host_fa: int = 0
    split_lost_host_ckpt_gpu_fa: int = 0
    split_lost_mixed_positional: int = 0
    split_lost_none: int = 0
    split_lost_tokens_gpu_ckpt_host_fa: int = 0
    split_lost_tokens_host_ckpt_gpu_fa: int = 0
    split_lost_tokens_mixed_positional: int = 0
    # The once-per-admission ledger (``OutcomeLedger``; tokens come
    # from request history, not the census union).
    outcomes_recorded: int = 0
    outcomes_first_turn: int = 0
    outcomes_duplicate: int = 0
    outcome_prompt_tokens: int = 0
    outcome_prompt_tokens_returns: int = 0
    reusable_tokens: int = 0
    reusable_full_tokens: int = 0
    adopted_gpu_tokens: int = 0
    adopted_host_tokens: int = 0
    adopted_capped_tokens: int = 0
    host_failed_tokens: int = 0
    host_cancelled_tokens: int = 0
    lost_tokens: int = 0
    outcome_restores_requested: int = 0
    outcome_restores_completed: int = 0
    outcome_restores_failed: int = 0
    outcome_restores_cancelled: int = 0
    outcome_geometry_missing: int = 0
    # Hashes the bounded request history dropped (cumulative; non-zero
    # on the prompt basis invalidates the ceiling / E claims from then on).
    history_truncated: int = 0
    history_truncated_full: int = 0
    # Run-length detectors (``FlatStoreDetector`` etc.), episodes.
    flat_store_episodes: int = 0
    restore_stall_episodes: int = 0
    host_flush_events: int = 0
    # Census gauges: host rows by class, refreshed once per stats drain.
    host_rows_published: int = 0
    host_rows_provisional: int = 0
    host_rows_idle: int = 0
    host_rows_leased: int = 0
    host_rows_duplicate_running: int = 0
    host_rows_duplicate_imminent: int = 0
    grace_pins_active: int = 0
    flat_store_s: int = 0
    worker: dict = field(default_factory=dict)
    reclaims: dict = field(default_factory=dict)


class _EvictionHook:
    """BlockPool.metrics_collector for the host pool: sees every block the
    pool un-keys through ``_maybe_evict_cached_block`` (LRU reuse in
    get_new_blocks, evict_blocks) before its key is removed."""

    def __init__(self, runtime: "TierRuntime") -> None:
        self._rt = runtime

    def on_block_allocated(self, block: "KVCacheBlock") -> None:
        pass

    def on_block_accessed(self, block: "KVCacheBlock") -> None:
        pass

    def on_block_evicted(self, block: "KVCacheBlock") -> None:
        if (
            self._rt.units is not None
            and block.block_hash is not None
            and not self._rt._resetting
        ):
            self._rt.units.on_host_unkeyed(block)
        self._rt._on_unkeyed(block)

    def reset(self) -> None:
        # BlockPool.reset_prefix_cache dropped every key without eviction.
        self._rt.cached_blocks = 0
        if self._rt.units is not None:
            self._rt.units.on_host_reset()


class TierRuntime:
    def __init__(
        self,
        cpu_block_pool: "BlockPool",
        num_cpu_blocks: int,
        fa_gidx: int,
        fa_block_size: int,
    ) -> None:
        self.cpu_block_pool = cpu_block_pool
        # Block 0 is the pool's null block, never cached.
        self.capacity_blocks = max(0, num_cpu_blocks - 1)
        self.fa_gidx = fa_gidx
        self.fa_block_size = fa_block_size
        self.counters = TierCounters()
        self._drained = TierCounters()
        # Cumulative name -> int counters owned elsewhere (plan admission,
        # backing, transactions; ``TierPlans.log_counters``), exported as
        # census events by ``add_stats``.
        self.extra_counters: Callable[[], dict[str, int]] | None = None
        self._extra_drained: dict[str, int] = {}
        # Host blocks currently holding a key (arena occupancy).
        self.cached_blocks = 0
        # Tokens asked of the tier at the request's last lookup / planned for
        # its pending load (for the admission split and failure accounting).
        self._queried: dict[str, int] = {}
        self._load_tokens: dict[str, int] = {}
        self._invalidating = False
        # Set while a coordinated reset un-keys every host block.
        self._resetting = False
        self.census = TierCensus(self)
        # Request-history reuse index and the once-per-admission
        # outcome ledger. They run with the census flag OFF (production) and
        # never decide anything. ``plans._record_outcome`` reaches the ledger
        # through ``outcome_sink`` (``reusable_tokens(request)``, ``record``,
        # ``restore_retired``).
        self.hash_block_size = 0
        self.history = HistoryIndex()
        self.ledger = OutcomeLedger(
            self.counters, self.history, geometry=lambda: self.hash_block_size
        )
        self.outcome_sink = self.ledger
        # ``units.loss_cause`` resolves an ``unknown`` miss to the cause
        # the unit table recorded (pure lookup; ``units`` binds later).
        self.ledger.loss_lookup = self._loss_cause
        # Mappings of cumulative counters owned elsewhere (the write-back
        # counters: ``lambda: asdict(writeback.counters)``) read by the series
        # line and the run-length detectors.
        self.series_sources: list[Callable[[], dict]] = []
        self.flat_detector = FlatStoreDetector()
        self.restore_detector = RestoreProgressDetector()
        self.flush_detector = HostFlushDetector()
        # A coordinated reset un-keys every host row on purpose. The
        # epoch counts them (and the scheduler's cache epoch is read too) so
        # the flush detector re-arms instead of reading the reset as a flush.
        self.reset_epoch = 0
        self._flush_epoch_seen: tuple = (0, 0)
        self.last_v3_line = ""
        self.metrics_every_s = 60.0
        self._clock: Callable[[], float] = time.monotonic
        self._cpu_clock: Callable[[], float] = time.process_time
        self._metrics_prev: dict | None = None
        self._metrics_prev_t: float | None = None
        self._metrics_prev_cpu: float | None = None
        self._metrics_prev_flat: dict = {}
        self._admission_route_warned = False
        self.last_metrics_line = ""
        # Restore units and host order (kv_host_tier_units.UnitTable), bound
        # with the GPU pool; None until then.
        self.units: UnitTable | None = None
        assert cpu_block_pool.metrics_collector is None
        cpu_block_pool.metrics_collector = _EvictionHook(self)

    # -- tokens -------------------------------------------------------------
    def _fa_tokens(self, block_hash: "BlockHashWithGroupId", num_tokens) -> int:
        from vllm.v1.core.kv_cache_utils import get_group_id

        if num_tokens is None or get_group_id(block_hash) != self.fa_gidx:
            return 0
        return (num_tokens - 1) % self.fa_block_size + 1 if num_tokens > 0 else 0

    def _on_unkeyed(self, block: "KVCacheBlock") -> None:
        if block.block_hash is None or block.is_null:
            return
        self.cached_blocks -= 1
        tokens = self._fa_tokens(block.block_hash, block.block_hash_num_tokens)
        c = self.counters
        if self._invalidating:
            c.load_failed_blocks += 1
        elif self._resetting:
            # A reset is not an eviction (backing_invalidated{reset}).
            c.reset_unkeyed_blocks += 1
        else:
            c.evicted_blocks += 1
            c.evicted_tokens += tokens

    # -- lookups and admission ---------------------------------------------
    def on_lookup(self, req_id: str, queried_tokens: int) -> None:
        self._queried[req_id] = max(0, queried_tokens)

    def on_admission(
        self, req_id: str, num_external_tokens: int, *, hit_tokens: int | None = None
    ) -> None:
        """``hit_tokens``: the external tokens actually handed off to a restore
        (default: all of them). A plan whose hand-off was refused books no hit
        tokens, only miss tokens, as ``hit_tokens`` in the tier log line does; its load state is still tracked from
        ``num_external_tokens``."""
        queried = self._queried.pop(req_id, 0)
        booked = num_external_tokens if hit_tokens is None else hit_tokens
        hit = min(booked, queried) if queried else booked
        self.counters.hit_tokens += hit
        self.counters.miss_tokens += max(0, queried - hit)
        if num_external_tokens > 0:
            self._load_tokens[req_id] = num_external_tokens
            self.census.pending_loads.setdefault(req_id, (num_external_tokens, 0))

    def unit_counters(self) -> dict[str, int]:
        """Cumulative eviction / supersede / SLRU counters: the unit
        table's ``UnitCounters`` as named, the admission policy's
        ``AdmissionCounters`` as named when the name is in ``COUNTERS_V3``
        and ``slru_<name>`` otherwise (``demoted`` is also a write-back
        counter). Empty before the unit table binds."""
        units = self.units
        out: dict[str, int] = {}
        if units is None:
            return out
        sources = (
            (getattr(units, "counters", None), ""),
            (getattr(getattr(units, "policy", None), "counters", None), "slru_"),
        )
        for src, prefix in sources:
            if src is None:
                continue
            for name, value in vars(src).items():
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    continue
                out[name if not prefix or name in COUNTERS_V3 else prefix + name] = value
        return out

    def v3_counters(self, merged: dict | None = None) -> dict[str, int]:
        """Every ``COUNTERS_V3`` name some source provides, plus the unit and
        SLRU counters; ``merged`` is ``_series_pending()`` (write-back + plans)."""
        merged = self._series_pending() if merged is None else merged
        out = {n: merged[n] for n in COUNTERS_V3 if n in merged}
        out.update(self.unit_counters())
        return out

    def _loss_cause(self, block_hashes):
        units = self.units
        if units is None or not units.lost:
            return None
        return units.loss_cause(block_hashes)

    def on_request_gone(self, req_id: str) -> None:
        self._queried.pop(req_id, None)
        self.ledger.restore_retired(req_id, None)

    def forget_load(self, req_id: str) -> None:
        """The request's load state is cleaned up (``on_load_retired``): drop
        its token count. A completed or failed load already took it
        (``on_load_done`` / ``invalidate_failed_load`` run first); a load
        cancelled before it issued never did."""
        self._load_tokens.pop(req_id, None)

    def on_load_done(self, req_id: str) -> None:
        self._load_tokens.pop(req_id, None)
        # Idempotent: a failed load already retired through
        # ``invalidate_failed_load`` (it runs before this).
        self.ledger.restore_retired(req_id, True)

    # -- request history and outcomes --------------------------------
    def bind_geometry(
        self, hash_block_size: int, align: int | None = None, cache_tokens: int = 0
    ) -> None:
        """The scheduler's hash block size (and, optionally, the restore grid
        tokens are floored to). Without it ``reusable_tokens`` returns 0 and
        ``outcome_geometry_missing`` counts, so E reads None instead of a
        wrong number. ``cache_tokens``: what the GPU and host tiers hold
        together; with the hash unit it sizes the request history."""
        self.hash_block_size = int(hash_block_size)
        self.ledger.align = align
        if self.hash_block_size > 0:
            self.history.capacity = max(
                self.history.capacity,
                history_capacity(self.hash_block_size, cache_tokens),
            )
            logger.info(
                "SimpleCPU tier history: capacity=%d hashes per basis "
                "hash_block_size=%d", self.history.capacity, self.hash_block_size,
            )

    def on_request_finished(self, request: Any) -> None:
        """A request finished: its computed chain becomes reusable history."""
        self.ledger.note_finished(request)
        self.ledger.forget(request.request_id)

    @staticmethod
    def check_load_destinations(
        gpu_pool: "BlockPool", gpu_block_ids: list[int], req_id: str
    ) -> None:
        """Every restore target is private to the loader and not servable by
        the prefix cache (comment 4166, load side). Called before the tier's
        own touch, so the request's block table holds the only reference."""
        for bid in gpu_block_ids:
            block = gpu_pool.blocks[bid]
            if block.ref_cnt != 1 or block.block_hash is not None:
                raise RuntimeError(
                    f"host-RAM KV tier: refusing to restore request {req_id} "
                    f"into GPU block {bid} (ref_cnt={block.ref_cnt}, "
                    f"hashed={block.block_hash is not None}); a restore may "
                    "only fill blocks the request owns alone and the prefix "
                    "cache cannot serve yet"
                )

    # -- stores -------------------------------------------------------------
    @staticmethod
    def snapshot(gpu_pool: "BlockPool", gpu_block_ids: list[int]) -> list[StoreKeys]:
        """The keys each store source carries when the store is issued."""
        out: list[StoreKeys] = []
        for bid in gpu_block_ids:
            block = gpu_pool.blocks[bid]
            out.append(
                (
                    block.block_hash,
                    frozenset(gpu_pool.cached_block_hashes_by_block.get(bid, ())),
                )
            )
        return out

    def keys_to_register(
        self, gpu_pool: "BlockPool", gpu_block_id: int, issued: StoreKeys
    ) -> frozenset | None:
        """Secondary keys to register with the primary, or None to discard
        the host copy (its source was re-keyed while the copy was in flight)."""
        primary, secondaries = issued
        block = gpu_pool.blocks[gpu_block_id]
        if primary is None or block.block_hash != primary:
            self.counters.discarded_rekeyed_blocks += 1
            return None
        now = gpu_pool.cached_block_hashes_by_block.get(gpu_block_id, ())
        return secondaries.intersection(now)

    def on_registered(self, cpu_block: "KVCacheBlock", was_keyed: bool) -> None:
        if was_keyed or cpu_block.block_hash is None:
            return
        self.cached_blocks += 1
        self.counters.stored_blocks += 1
        self.counters.stored_tokens += self._fa_tokens(
            cpu_block.block_hash, cpu_block.block_hash_num_tokens
        )

    # -- load failures ------------------------------------------------------
    def unkey_all_for_reset(self) -> None:
        """Coordinated reset: host keys leave lookup visibility before
        any physical reuse; in-flight copies keep their storage refs."""
        pool = self.cpu_block_pool
        self.reset_epoch += 1
        self.flush_detector.rearm()
        self._resetting = True
        try:
            pool.evict_blocks(
                {b.block_id for b in pool.blocks if b.block_hash is not None}
            )
        finally:
            self._resetting = False
        # One bulk teardown, not one unit walk per evicted key.
        if self.units is not None:
            self.units.on_host_reset()

    def invalidate_failed_load(
        self, req_id: str, cpu_block_ids: list[int], *, stale: bool = False
    ) -> None:
        """Un-key the host blocks a failed load read, so the recompute retry
        cannot hit them again. The blocks keep the load's refs; the caller's
        cleanup releases them to the free list unhashed.

        ``stale``: the copy succeeded for an older cache epoch. It is
        invalidated the same way but is not a failed restore:
        the attempt settles as cancelled and no failure is counted; the
        manager counts it in ``load_stale_epoch_completions``."""
        if stale:
            self.census.load_done(req_id, failed=False, cancelled=True)
            self.ledger.restore_retired(req_id, None)
        else:
            self.census.load_done(req_id, failed=True)
            self.ledger.restore_retired(req_id, False)
        pool = self.cpu_block_pool
        units = self.units
        # Un-key only the rows this load read; a lost unit's other
        # published rows (intact, possibly leased by other loads) keep keys.
        prev = units.keep_lost_remains(True) if units is not None else None
        self._invalidating = True
        try:
            pool.evict_blocks(set(cpu_block_ids))
        finally:
            self._invalidating = False
            if units is not None:
                units.keep_lost_remains(prev)
        if stale:
            self._load_tokens.pop(req_id, None)
            logger.info(
                "host-RAM KV tier: stale-epoch completion of request %s; its "
                "%d host blocks are dropped (the cache was reset under the "
                "copy) and the request recomputes",
                req_id,
                len(cpu_block_ids),
            )
            return
        c = self.counters
        c.load_failed_requests += 1
        c.load_failed_tokens += self._load_tokens.pop(req_id, 0)
        logger.warning(
            "host-RAM KV tier: restore of request %s failed; its %d host "
            "blocks are dropped from the tier and the request recomputes "
            "(%d failed restores so far)",
            req_id,
            len(cpu_block_ids),
            c.load_failed_requests,
        )

    # -- metrics ------------------------------------------------------------
    def add_stats(self, stats: Any) -> None:
        """Drain the counter deltas and set the occupancy gauges on a
        SimpleCPUOffloadStats payload."""
        self.refresh_gauges()
        self.poll_metrics()
        cur, prev = self.counters, self._drained
        for cause, tokens in self.ledger.miss_tokens.items():
            name = "outcome_miss_tokens_" + cause
            delta = tokens - self._extra_drained.get(name, 0)
            if delta > 0:
                stats.increase_counter(MetricName.CENSUS_EVENTS, delta, (name,))
            self._extra_drained[name] = tokens
        for cause, n in self.ledger.miss_n.items():
            name = "outcome_miss_" + cause
            delta = n - self._extra_drained.get(name, 0)
            if delta > 0:
                stats.increase_counter(MetricName.CENSUS_EVENTS, delta, (name,))
            self._extra_drained[name] = n
        for name, event in _TOKEN_EVENTS.items():
            delta = getattr(cur, name) - getattr(prev, name)
            if delta > 0:
                stats.increase_counter(MetricName.TOKENS, delta, (event,))
        for name, event in _BLOCK_EVENTS.items():
            delta = getattr(cur, name) - getattr(prev, name)
            if delta > 0:
                stats.increase_counter(MetricName.BLOCKS, delta, (event,))
        for name in EVENTS:
            delta = getattr(cur, name) - getattr(prev, name)
            if delta > 0:
                stats.increase_counter(MetricName.CENSUS_EVENTS, delta, (name,))
        for name in GAUGES:
            stats.set_gauge(MetricName.CENSUS, getattr(cur, name), (name,))
        census = self.census
        for labels, value in census.reclaims.items():
            delta = value - census.drained_reclaims[labels]
            if delta > 0:
                stats.increase_counter(MetricName.RECLAIMS, delta, labels)
        census.drained_reclaims = census.reclaims.copy()
        for rank, values in census.worker.items():
            for name, value in values.items():
                labels = (rank, name)
                delta = value - census.drained_worker.get(labels, 0)
                if delta > 0:
                    stats.increase_counter(MetricName.WORKER, delta, labels)
                census.drained_worker[labels] = value
            for bucket in ("", "_n8", "_n32", "_n70", "_nother"):
                blocks = values.get("d2h_blocks" + bucket, 0)
                if blocks:
                    stats.set_gauge(
                        MetricName.WORKER_COST,
                        values.get("d2h_ms" + bucket, 0) / blocks,
                        (rank, bucket.removeprefix("_n") or "all"),
                    )
        self._drained = replace(cur)
        if self.extra_counters is not None:
            for name, value in self.extra_counters().items():
                delta = value - self._extra_drained.get(name, 0)
                if delta > 0:
                    stats.increase_counter(MetricName.CENSUS_EVENTS, delta, (name,))
                self._extra_drained[name] = value
        for name, value in self.unit_counters().items():
            key = "unit_" + name
            delta = value - self._extra_drained.get(key, 0)
            if delta > 0:
                stats.increase_counter(MetricName.CENSUS_EVENTS, delta, (name,))
            self._extra_drained[key] = value
        stats.set_gauge(MetricName.CACHED_BLOCKS, self.cached_blocks)
        stats.set_gauge(MetricName.CAPACITY_BLOCKS, self.capacity_blocks)

    # -- metrics ------------------------------------------------------
    def refresh_gauges(self) -> None:
        """Host rows by class (census gauges), from pool state only.

        ``idle``: free rows no unit owns (``UnitTable._idle_ownerless`` when the
        table has it, else free and unkeyed); ``leased``: a reader holds a ref;
        ``published``/``provisional``: keyed rows a lookup can / cannot serve
        yet (staged for an assembling unit). The duplicate classes come from
        ``UnitTable.duplicate_row_counts()`` when the eviction code provides it
        (``{"running": n, "imminent": n}``), else stay 0."""
        c = self.counters
        pool = self.cpu_block_pool
        units = self.units
        idle_fn = getattr(units, "_idle_ownerless", None) if units is not None else None
        staged_fn = getattr(units, "is_staged", None) if units is not None else None
        published = provisional = idle = leased = 0
        for b in pool.blocks:
            if b.is_null:
                continue
            if b.ref_cnt > 0:
                leased += 1
            keyed = b.block_hash is not None
            if keyed:
                try:
                    staged = staged_fn is not None and staged_fn(b)
                except Exception:  # a gauge must never break the stats drain
                    staged, staged_fn = False, None
                if staged:
                    provisional += 1
                else:
                    published += 1
            if b.ref_cnt == 0:
                try:
                    free = idle_fn(b) if idle_fn is not None else not keyed
                except Exception:
                    free, idle_fn = not keyed, None
                idle += 1 if free else 0
        c.host_rows_published, c.host_rows_provisional = published, provisional
        c.host_rows_idle, c.host_rows_leased = idle, leased
        dup = getattr(units, "duplicate_row_counts", None) if units is not None else None
        if callable(dup):
            counts = dup()
            c.host_rows_duplicate_running = int(counts.get("running", 0))
            c.host_rows_duplicate_imminent = int(counts.get("imminent", 0))
        sched = self.census.scheduler
        mgr = getattr(sched, "kv_cache_manager", None)
        c.grace_pins_active = int(getattr(mgr, "num_grace_pins", 0) or 0)
        c.flat_store_s = int(self.flat_detector.current_flat_s)

    def _series_pending(self) -> dict:
        """Cumulative counters from every registered source plus the plans
        counters (``extra_counters``), one flat mapping."""
        merged: dict = {}
        if self.extra_counters is not None:
            try:
                merged.update(self.extra_counters())
            except Exception:
                logger.debug("tier extra counters failed", exc_info=True)
        for source in self.series_sources:
            try:
                merged.update(source())
            except Exception:  # a metrics source must never break stats
                logger.debug("tier series source failed", exc_info=True)
        # The write-back counters are read here directly, so the
        # periodic metrics line (and the flat-store detector's pending-work
        # signal) carry them even when the manager's change-detected tier log
        # does not fire on a write-back-only change. Registered sources win.
        wb = getattr(self.census.scheduler, "writeback", None)
        counters = getattr(wb, "counters", None)
        if counters is not None:
            try:
                for name, value in vars(counters).items():
                    if isinstance(value, (int, float)) and name not in merged:
                        merged[name] = value
            except Exception:
                logger.debug("tier writeback counters failed", exc_info=True)
        return merged

    def restored_requests(self, merged: dict) -> int:
        got = merged.get("restored_requests")
        return int(got) if got is not None else self.counters.restore_completed_requests

    def poll_metrics(self, now: float | None = None) -> str | None:
        """Feed the run-length detectors (every stats drain, ~10 s) and, once
        per ``metrics_every_s``, log the ``SimpleCPU tier metrics:`` line.

        The line carries per-INTERVAL values (the server's own hit rate is a
        1,000-request window that can hide a whole-host freeze): E, rho_bar,
        g, h, T*, E_tier, combined, miss share, copy and CPU cost, the host row
        classes, the ``SERIES_FIELDS_V3`` counters that exist, and
        ``route_gap`` (outcomes with a restore minus ``restored_requests``
        over the interval: a non-zero value means an admission route does not
        record its outcome)."""
        now = self._clock() if now is None else now
        c = self.counters
        merged = self._series_pending()
        if self.flat_detector.feed(now, c.stored_blocks, merged or None):
            c.flat_store_episodes += 1
            logger.warning(
                "SimpleCPU tier flat-store: no host copy landed for %.0f s while "
                "copy work was eligible (host rows idle=%d published=%d); "
                "stored_blocks=%d",
                self.flat_detector.current_flat_s, c.host_rows_idle,
                c.host_rows_published, c.stored_blocks,
            )
        restored = self.restored_requests(merged)
        if self.restore_detector.feed(now, restored):
            c.restore_stall_episodes += 1
            logger.warning(
                "SimpleCPU tier restore-stall: restored_requests=%d has not risen "
                "for %.0f s", restored, self.restore_detector.window_s,
            )
        epoch = (self.reset_epoch, getattr(self.census.scheduler, "cache_epoch", 0))
        if epoch != self._flush_epoch_seen:
            self._flush_epoch_seen = epoch
            self.flush_detector.rearm()
        if self.flush_detector.feed(now, self.cached_blocks, self.capacity_blocks):
            c.host_flush_events += 1
            logger.warning(
                "SimpleCPU tier host flush: cached rows fell %d -> %d (capacity "
                "%d) between two samples",
                self.flush_detector.events[-1][1], self.cached_blocks,
                self.capacity_blocks,
            )
        if self._metrics_prev_t is not None and now - self._metrics_prev_t < self.metrics_every_s:
            return None
        return self._emit_metrics(now, merged, restored)

    def _emit_metrics(self, now: float, merged: dict, restored: int) -> str | None:
        c = self.counters
        snap = self.ledger.snapshot()
        cpu = self._cpu_clock()
        first = self._metrics_prev_t is None
        dt = None if first else now - self._metrics_prev_t
        # Intervals read settled admission cohorts; the
        # ``cum_*`` fields keep the provisional credit of in-flight restores.
        settled = self.ledger.settled_snapshot()
        d = interval_delta(settled, self._metrics_prev)
        truncated = snap.get("history_truncated", 0)  # cumulative: sticky
        eff = efficiency(d, truncated)
        cum = efficiency(snap, truncated)
        fields: dict[str, Any] = {"interval_s": dt}
        for name in ("e", "rho_bar", "g", "h", "t_star", "e_tier", "combined", "miss_share"):
            fields[name] = eff[name]
        for name in ("e", "rho_bar", "g", "h", "t_star", "e_tier"):
            fields["cum_" + name] = cum[name]
        fields["ceiling_valid"] = int(eff["ceiling_valid"])
        fields["history_truncated"] = truncated
        fields["outcomes"] = d.get("outcomes_recorded", 0)
        fields["outcomes_first_turn"] = d.get("outcomes_first_turn", 0)
        fields["restore_outcomes"] = d.get("outcome_restores_requested", 0)
        fields["restores_cancelled"] = d.get("outcome_restores_cancelled", 0)
        flat_prev = self._metrics_prev_flat
        fields["restored_d"] = restored - flat_prev.get("restored", restored)
        # A settled restore outcome that neither failed nor was cancelled must
        # show as a restored request.
        gap = (fields["restore_outcomes"] - fields["restored_d"]
               - d.get("outcome_restores_failed", 0) - fields["restores_cancelled"])
        fields["route_gap"] = gap
        if (not first and abs(gap) > max(2, 0.2 * max(fields["restored_d"], 1))
                and not self._admission_route_warned):
            self._admission_route_warned = True
            logger.warning(
                "SimpleCPU tier outcomes: %d restore outcomes vs %d restored "
                "requests over the last interval; an admission route does not "
                "record its AdmissionOutcome", fields["restore_outcomes"],
                fields["restored_d"],
            )
        fields["cached_blocks"] = self.cached_blocks
        fields["stored_blocks"] = c.stored_blocks
        for name in ("host_rows_idle", "host_rows_published", "host_rows_provisional",
                     "host_rows_leased", "grace_pins_active"):
            fields[name] = getattr(c, name)
        fields["flat_store_s"] = c.flat_store_s
        fields["flat_longest_s"] = int(self.flat_detector.longest_flat_s)
        # Copy cost per rank (worker census): ms per block and GB/s when the
        # arena row size is known (``row_bytes``).
        for rank, values in sorted(self.census.worker.items()):
            blocks, ms = values.get("d2h_blocks", 0), values.get("d2h_ms", 0)
            if blocks:
                fields[f"d2h_ms_per_block_r{rank}"] = ms / blocks
                row_bytes = getattr(self, "row_bytes", 0)
                if row_bytes and ms:
                    fields[f"d2h_gbps_r{rank}"] = blocks * row_bytes / (ms / 1e3) / 1e9
        if self._metrics_prev_cpu is not None and dt:
            fields["cpu_pct"] = 100.0 * (cpu - self._metrics_prev_cpu) / dt
        sched = self.census.scheduler
        step = getattr(sched, "_sched_step", None)
        if step is not None:
            fields["sched_step"] = step
            prev_step = flat_prev.get("step")
            sel = merged.get("select_us")
            if sel is not None and prev_step is not None and step > prev_step:
                fields["select_us_per_step"] = (sel - flat_prev.get("select_us", 0)) / (step - prev_step)
        for name in _SERIES_FIELDS:
            if name in merged and name not in fields:
                fields[name] = merged[name]
        fields["restored_requests"] = restored
        for cause, n in sorted(d.get("miss_n", {}).items()):
            fields["miss_" + cause] = n
        line = "SimpleCPU tier metrics: " + " ".join(
            f"{k}={_fmt(v)}" for k, v in fields.items() if v is not None
        )
        logger.info(line)
        self.last_metrics_line = line
        self._emit_v3_counters(merged)
        self._metrics_prev, self._metrics_prev_t, self._metrics_prev_cpu = settled, now, cpu
        self._metrics_prev_flat = {
            "restored": restored, "step": step, "select_us": merged.get("select_us", 0),
        }
        return line

    def _emit_v3_counters(self, merged: dict) -> None:
        """``SimpleCPU tier counters:`` - the cumulative eviction, supersede,
        SLRU, copy-lane, credit and assembly counters (``COUNTERS_V3`` plus
        the unit and SLRU counters), so a serving log shows exclusive-tail
        eviction working. ``missing`` counts ``COUNTERS_V3`` names no
        source provides."""
        values = self.v3_counters(merged)
        missing = [n for n in COUNTERS_V3 if n not in values]
        parts = [f"{k}={_fmt(v)}" for k, v in values.items()]
        parts.append(f"missing={len(missing)}")
        if missing:
            parts.append("missing_names=" + ",".join(missing))
        line = "SimpleCPU tier counters: " + " ".join(parts)
        logger.info(line)
        self.last_v3_line = line

    def snapshot_counters(self) -> TierCounters:
        return replace(self.counters,
                       worker={k: dict(v) for k, v in self.census.worker.items()},
                       reclaims=dict(self.census.reclaims))


_SERIES_FIELDS = (
    "stored_blocks", "demoted_units", "unit_lost", "demote_missed_running",
    "demote_missed_escape", "demote_host_busy", "demote_dropped_host_full",
    "restored_requests", "yield_incomplete", "restore_cap_wait",
    "lane2_late", "backing_revoked", "host_rows_idle", "host_rows_published",
    "credits_d_issued", "credits_r_issued", "select_us",
)  # = kv_host_tier_iface.SERIES_FIELDS_V3


def _fmt(v: Any) -> str:
    return f"{v:.4f}" if isinstance(v, float) else str(v)


def build_prom_metrics(base_cls: type, *args: Any) -> Any:
    """``base_cls`` (SimpleCPUOffloadPromMetrics) plus the tier's metrics."""

    class MerlinKVHostTierPromMetrics(base_cls):  # type: ignore[misc, valid-type]
        def __init__(self, vllm_config, metric_types, labelnames, per_engine):
            super().__init__(vllm_config, metric_types, labelnames, per_engine)
            tokens = self._counter_cls(
                name=MetricName.TOKENS,
                documentation=(
                    "Host-RAM KV tier tokens by event: hit / miss (at "
                    "admission, past the local prefix-cache hit), stored / "
                    "evicted (full-attention tokens gaining / losing a host "
                    "key), load_failed (restores that failed and were "
                    "recomputed)."
                ),
                labelnames=self._labelnames + ["event"],
            )
            blocks = self._counter_cls(
                name=MetricName.BLOCKS,
                documentation=(
                    "Host-RAM KV tier blocks by event: stored, evicted, "
                    "load_failed, discarded_rekeyed (host copy dropped: its "
                    "GPU source was re-keyed while the copy was in flight)."
                ),
                labelnames=self._labelnames + ["event"],
            )
            cached = self._gauge_cls(
                name=MetricName.CACHED_BLOCKS,
                documentation=(
                    "Host-RAM KV tier arena occupancy: host blocks holding a "
                    "cached key (in-flight transfers are not counted)."
                ),
                labelnames=self._labelnames,
                multiprocess_mode="mostrecent",
            )
            capacity = self._gauge_cls(
                name=MetricName.CAPACITY_BLOCKS,
                documentation="Host-RAM KV tier arena size in blocks.",
                labelnames=self._labelnames,
                multiprocess_mode="mostrecent",
            )
            for name, labels, gauge in (
                (MetricName.CENSUS, ["observation"], True),
                (MetricName.CENSUS_EVENTS, ["event"], False),
                (MetricName.RECLAIMS, ["group", "class", "cause"], False),
                (MetricName.WORKER, ["rank", "observation"], False),
                (MetricName.WORKER_COST, ["rank", "batch_blocks"], True),
            ):
                kwargs = {"multiprocess_mode": "mostrecent"} if gauge else {}
                self._defs[name] = (self._gauge_cls if gauge else self._counter_cls)(
                    name=name, documentation="Read-only host-tier census.",
                    labelnames=self._labelnames + labels, **kwargs,
                )
                self._num_label_values[name] = len(labels)
            self._defs.update(
                {
                    MetricName.TOKENS: tokens,
                    MetricName.BLOCKS: blocks,
                    MetricName.CACHED_BLOCKS: cached,
                    MetricName.CAPACITY_BLOCKS: capacity,
                }
            )
            self._num_label_values.update(
                {
                    MetricName.TOKENS: 1,
                    MetricName.BLOCKS: 1,
                    MetricName.CACHED_BLOCKS: 0,
                    MetricName.CAPACITY_BLOCKS: 0,
                }
            )

    return MerlinKVHostTierPromMetrics(*args)


__all__ = [
    "MetricName",
    "TierCounters",
    "TierRuntime",
    "build_prom_metrics",
]
