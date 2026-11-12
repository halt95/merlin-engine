# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Read-only host-tier census. None of these observations authorizes a cache hit.

Units are key manifests proved at finish, including the EAGLE/SW proof keys:
the restore unit table's own manifest when the tier has one, else an
exact-key proof at the replay/tail boundaries.
The scan is O(pool blocks + retained manifest members), with a bounded metadata
budget. Dropped manifests make the unit counts lower bounds and are reported.
The census does not certify readiness: it observes keys, not the per-rank
commit receipts.
"""

import time
from collections import Counter, OrderedDict, deque
from dataclasses import dataclass
from itertools import islice

from vllm.logger import init_logger
from vllm.merlin.kv_host_tier import RESTORE_UNIT_TOKENS
from vllm.merlin.kv_host_tier_units import UnitManifest
from vllm.v1.core.kv_cache_coordinator import replay_boundaries
from vllm.v1.core.kv_cache_utils import (
    eagle_proof_margin,
    get_group_id,
    make_block_hash_with_group_id,
)
from vllm.v1.kv_cache_interface import (
    MambaSpec,
    get_mamba_prefill_checkpoint_position,
)

logger = init_logger(__name__)

GAUGES = (
    "gpu_host_overlap", "restorable_units", "restorable_tokens", "orphan_blocks",
    "orphan_unclassified",
    "union_restorable_units", "union_restorable_tokens", "split_units",
    "reclaim_step_max", "in_flight_peak",
    "readiness_observable", "manifest_members",
    # Host rows by class
    "host_rows_published", "host_rows_provisional", "host_rows_idle",
    "host_rows_leased", "host_rows_duplicate_running",
    "host_rows_duplicate_imminent", "grace_pins_active",
    "flat_store_s",
)
EVENTS = (
    "census_scans", "manifest_dropped", "reclaimed_blocks", "collateral_blocks",
    "commit_dropped_not_hashed", "split_lost", "split_lost_tokens",
    "returns_cold", "returns_gpu", "returns_host", "returns_mixed",
    "checkpoint_replay", "checkpoint_tail", "checkpoint_both", "checkpoint_unknown",
    "restore_completed_requests", "restore_completed_tokens",
    "restore_completed_blocks", "restore_failed",
    "preemption_flushes", "preempted_requests", "lookup_retries",
    "gpu_readiness_unknown_hits",
    # split_lost by layout (the census union held a boundary the live
    # plan did not adopt), the once-per-admission outcome ledger, run-length
    # detectors. ``restore_cap_wait`` / ``yield_incomplete`` / ``lane2_late`` /
    # ``backing_revoked`` are owned by the plans / copy-scheduler counters and
    # reach the same metric through ``TierRuntime.extra_counters``.
    "split_lost_gpu_ckpt_host_fa", "split_lost_host_ckpt_gpu_fa",
    "split_lost_mixed_positional", "split_lost_none",
    "split_lost_tokens_gpu_ckpt_host_fa", "split_lost_tokens_host_ckpt_gpu_fa",
    "split_lost_tokens_mixed_positional",
    "outcomes_recorded", "outcomes_first_turn", "outcomes_duplicate",
    "outcome_prompt_tokens", "outcome_prompt_tokens_returns",
    "reusable_tokens", "reusable_full_tokens",
    "adopted_gpu_tokens", "adopted_host_tokens", "adopted_capped_tokens",
    "host_failed_tokens", "host_cancelled_tokens", "lost_tokens",
    "outcome_restores_requested", "outcome_restores_completed",
    "outcome_restores_failed", "outcome_restores_cancelled",
    "outcome_geometry_missing", "history_truncated", "history_truncated_full",
    "flat_store_episodes", "restore_stall_episodes", "host_flush_events",
)

# Cumulative per-admission ledger fields (``OutcomeLedger``); every name is a
# ``TierCounters`` field and a census event.
LEDGER_FIELDS = (
    "outcomes_recorded", "outcomes_first_turn", "outcomes_duplicate",
    "outcome_prompt_tokens", "outcome_prompt_tokens_returns",
    "reusable_tokens", "reusable_full_tokens",
    "adopted_gpu_tokens", "adopted_host_tokens", "adopted_capped_tokens",
    "host_failed_tokens", "host_cancelled_tokens", "lost_tokens",
    "outcome_restores_requested", "outcome_restores_completed",
    "outcome_restores_failed", "outcome_restores_cancelled",
    "outcome_geometry_missing", "history_truncated", "history_truncated_full",
)


@dataclass(frozen=True)
class Unit:
    owner: str
    tokens: int
    position: str
    keys: frozenset
    checkpoints: frozenset
    endpoint: bytes


class _ProofPool:
    """Only the read API used by the manager classmethods, with exact key tracing."""

    def __init__(self, gpu, host):
        self.gpu, self.host = gpu, host
        self.null_block = gpu.null_block
        self.hash_block_size = gpu.hash_block_size
        self.keys = set()

    def get_cached_block(self, block_hash, kv_cache_group_ids):
        out, keys = [], []
        for gid in kv_cache_group_ids:
            key = make_block_hash_with_group_id(block_hash, gid)
            block = None
            for pool in (self.gpu, self.host):
                block = pool.cached_block_hash_to_block.get_one_block(key)
                if block is not None:
                    break
            if block is None:
                return None
            out.append(block)
            keys.append(key)
        self.keys.update(keys)
        return out


def unit_blocks_by_group(scheduler, boundary=RESTORE_UNIT_TOKENS):
    """Rows per group of a complete restore unit at ``boundary``: the one
    restore dependency manifest (data and EAGLE proof pages), the same
    geometry capacity sizing and the unit builder use. Rings and other groups
    outside the tier contribute zero. Independent of current residency."""
    from vllm.merlin.kv_restore_dependencies import restore_dependency_manifest

    cfg = scheduler.cpu_kv_cache_config
    counts = [0] * len(cfg.kv_cache_groups)
    for group in restore_dependency_manifest(
        cfg, boundary, coordinator=getattr(scheduler, "cpu_coordinator", None),
        block_sizes=scheduler.group_block_sizes,
    ):
        counts[group.group_id] = group.blocks
    return counts


class _UnitDirectory(OrderedDict):
    """Census units by identity (their key set), oldest first, with an index
    from each member key to the identities holding it, kept on every
    mutation, so a reclaim finds a key's owners without scanning every unit."""

    def __init__(self):
        super().__init__()
        self.by_key: dict = {}
        # Endpoint hash -> identities: a lookup visits only the units whose
        # endpoint lies on the request's hash chain.
        self.by_endpoint: dict = {}

    def _index(self, identity, unit) -> None:
        for key in unit.keys:
            self.by_key.setdefault(key, {})[identity] = None
        self.by_endpoint.setdefault(unit.endpoint, {})[identity] = None

    def _unindex(self, identity, unit) -> None:
        for key in unit.keys:
            holders = self.by_key.get(key)
            if holders is not None:
                holders.pop(identity, None)
                if not holders:
                    del self.by_key[key]
        holders = self.by_endpoint.get(unit.endpoint)
        if holders is not None:
            holders.pop(identity, None)
            if not holders:
                del self.by_endpoint[unit.endpoint]

    def __setitem__(self, identity, unit) -> None:
        old = super().get(identity)
        if old is not None:
            self._unindex(identity, old)
        super().__setitem__(identity, unit)
        self._index(identity, unit)

    def __delitem__(self, identity) -> None:
        self._unindex(identity, self[identity])
        super().__delitem__(identity)

    _MISSING = object()

    def pop(self, identity, default=_MISSING):
        if identity in self:
            unit = super().pop(identity)
            self._unindex(identity, unit)
            return unit
        if default is self._MISSING:
            raise KeyError(identity)
        return default

    def popitem(self, last=True):
        identity, unit = super().popitem(last=last)
        self._unindex(identity, unit)
        return identity, unit

    def clear(self) -> None:
        super().clear()
        self.by_key.clear()
        self.by_endpoint.clear()

    def on_chain(self, block_hashes):
        """The units whose endpoint is one of ``block_hashes``."""
        for h in block_hashes:
            for identity in self.by_endpoint.get(h, ()):
                yield self[identity]

    def owners_of(self, key) -> set:
        return {self[i].owner for i in self.by_key.get(key, ())}


class TierCensus:
    def __init__(self, runtime):
        self.rt = runtime
        from vllm.merlin.envs import read_registered

        self.enabled = bool(read_registered("MERLIN_KV_HOST_TIER_CENSUS"))
        # Keep the production path lightweight. The directory, proof state and
        # histogram accumulators exist only when census debugging is requested.
        self.units = _UnitDirectory() if self.enabled else None
        self.members = 0
        self.lookups = {} if self.enabled else None
        self.admitted = set() if self.enabled else None
        # Runtime lifecycle callbacks share this transient request map even
        # when census diagnostics are disabled.
        self.pending_loads = {}
        self.reclaim_step = 0
        self.superseded_keys = set() if self.enabled else None
        self._remember_queue = deque(maxlen=64) if self.enabled else None
        self.worker = {}
        self.drained_worker = {}
        self.reclaims = Counter() if self.enabled else {}
        self.drained_reclaims = {}
        self._last_dropped = 0

        self.scheduler = None

    def bind(self, scheduler):
        # The scheduler reference is bound whatever the census flag says: the
        # metrics line reads the step counter and select time through it.
        # Only the census hooks below are debug-only.
        self.scheduler = scheduler
        self._bind_hooks(scheduler)

    def _bind_hooks(self, scheduler):
        gpu = scheduler._gpu_block_pool
        if gpu is not None:
            gpu.evict_groups.census = self
        if getattr(self, "_layout_logged", False):
            return
        self._layout_logged = True
        unit_blocks = unit_blocks_by_group(scheduler)
        for gid, group in enumerate(scheduler.cpu_kv_cache_config.kv_cache_groups):
            spec = group.kv_cache_spec
            tensors = [
                t for t in scheduler.cpu_kv_cache_config.kv_cache_tensors
                if set(t.layers).intersection(group.layer_names)
            ]
            logger.info(
                "SimpleCPU tier census layout: group=%d spec=%s cacheable=%s "
                "block_size=%d page_bytes=%d strides=%s dtype=%s "
                "unit_tokens=%d unit_blocks=%d",
                gid, type(spec).__name__, spec.prefix_cacheable,
                scheduler.group_block_sizes[gid], spec.page_size_bytes,
                [t.block_stride for t in tensors],
                getattr(spec, "dtypes", getattr(spec, "dtype", None)),
                RESTORE_UNIT_TOKENS, unit_blocks[gid],
            )

    def prove(self, request, boundary):
        """Verify each group at this exact boundary, never truncate sparse state."""
        sched = self.scheduler
        gpu = sched._gpu_block_pool
        if gpu is None or boundary <= 0:
            return None
        co = sched.cpu_coordinator
        proof = _ProofPool(gpu, sched.cpu_block_pool)
        checkpoints = set()
        for gid, (group, manager) in enumerate(zip(
            co.kv_cache_config.kv_cache_groups, co.single_type_managers
        )):
            spec = group.kv_cache_spec
            if not spec.prefix_cacheable:
                continue  # Ring slot ids are never indexed.
            eagle = gid in co.eagle_group_ids
            fine = (co.enable_partial_hash_hits
                    and manager.supports_fine_grained_hash_lookup)
            alignment = (self.scheduler.hash_block_size if co.enable_partial_hash_hits
                         else co.scheduler_block_size)
            margin = eagle_proof_margin(
                manager.block_size, self.scheduler.hash_block_size, fine
            )
            maximum = min(
                boundary + (margin if eagle else 0),
                len(request.block_hashes) * self.scheduler.hash_block_size,
            )
            before = proof.keys.copy()
            _, reach = type(manager).find_longest_cache_hit(
                block_hashes=request.block_hashes,
                max_length=maximum,
                kv_cache_group_ids=[gid],
                block_pool=proof,
                kv_cache_spec=spec,
                drop_eagle_block=eagle,
                alignment_tokens=alignment,
                dcp_world_size=manager.dcp_world_size,
                pcp_world_size=manager.pcp_world_size,
            )
            if reach != boundary:
                return None
            if isinstance(spec, MambaSpec):
                checkpoints.update(proof.keys - before)
        return frozenset(proof.keys), frozenset(checkpoints)

    def remember(self, request):
        """Queue the finished request; at most REMEMBER_PER_STEP are proved per
        scheduler step (a full lookup proof per finish spiked the
        step). The queue is bounded: the oldest unproved request is dropped."""
        if self.scheduler is None or self.scheduler._gpu_block_pool is None:
            return
        self._remember_queue.append(request)

    REMEMBER_PER_STEP = 2

    def _drain_remember(self):
        for _ in range(min(self.REMEMBER_PER_STEP, len(self._remember_queue))):
            self._remember_now(self._remember_queue.popleft())

    def _remember_now(self, request):
        if self.scheduler is None or self.scheduler._gpu_block_pool is None:
            return
        co = self.scheduler.cpu_coordinator
        alignment = (self.scheduler.hash_block_size if co.enable_partial_hash_hits
                     else co.scheduler_block_size)
        prompt = request.num_prompt_tokens
        positions = replay_boundaries(
            prompt, co.scheduler_block_size, bool(co.eagle_group_ids)
        )
        replay = positions[0] // alignment * alignment
        tail = (positions[-1] if co.eagle_group_ids else prompt)
        tail = tail // alignment * alignment
        table = getattr(self.rt, "units", None)
        if table is not None:
            # The unit is the restore's own manifest (UnitTable: the
            # deepest boundary whose recurrent endpoints are keyed, FA pages
            # proved by a covering key as the union lookup accepts). The
            # exact-key proof below never reaches a block-aligned boundary of
            # an EAGLE FA group whose block is wider than the hash unit (it
            # wants a key one unit past B inside a full page), so on
            # Flash-Next it recorded nothing: manifest_members 0.
            # The exact-key proof is skipped whenever a unit table exists.
            found = table.manifest(request)
            if isinstance(found, UnitManifest):
                # "tail" only at the prompt-tail checkpoint the manager
                # registers (one unit lower with EAGLE); any other boundary
                # (a decode-time state, a lower fallback) is "unknown".
                prompt_tail = get_mamba_prefill_checkpoint_position(
                    prompt, alignment, bool(co.eagle_group_ids)
                )
                b = found.boundary
                position = ("both" if b == replay == prompt_tail else
                            "replay" if b == replay else
                            "tail" if b == prompt_tail else "unknown")
                self._record(request, found.boundary, position,
                             found.required | found.optional, found.endpoints)
            self._bound_members()
            return
        for boundary in sorted({replay, tail}):
            if boundary > request.num_computed_tokens:
                continue
            proved = self.prove(request, boundary)
            if proved is None:
                continue
            keys, checkpoints = proved
            position = ("both" if replay == tail else
                        "replay" if boundary == replay else "tail")
            self._record(request, boundary, position, keys, checkpoints)
        self._bound_members()

    def _record(self, request, boundary, position, keys, checkpoints):
        endpoint = request.block_hashes[
            boundary // self.scheduler.hash_block_size - 1
        ]
        unit = Unit(request.request_id, boundary, position, frozenset(keys),
                    frozenset(checkpoints), endpoint)
        old = self.units.pop(unit.keys, None)
        self.members -= len(old.keys) if old else 0
        self.units[unit.keys] = unit
        self.members += len(unit.keys)

    def _bound_members(self):
        # Bound scheduler metadata independently of the number/length of sessions.
        limit = 8 * (len(self.rt.cpu_block_pool.blocks) +
                     len(self.scheduler._gpu_block_pool.blocks))
        while self.members > limit:
            _, old = self.units.popitem(last=False)
            self.members -= len(old.keys)
            self.rt.counters.manifest_dropped += 1

    @staticmethod
    def index(pool, usable=None):
        """Key -> block id of the keyed blocks; with ``usable`` only the
        blocks the live lookup could serve."""
        keys, primaries = {}, set()
        for block in pool.blocks:
            if block.is_null or block.block_hash is None:
                continue
            if usable is not None and not usable(block):
                continue
            primaries.add(block.block_hash)
            keys[block.block_hash] = block.block_id
            for key in pool.cached_block_hashes_by_block.get(block.block_id, ()):
                keys[key] = block.block_id
        return keys, primaries

    def scan(self):
        """On-demand scan; also called every 256 scheduler steps."""
        if self.scheduler is None or self.scheduler._gpu_block_pool is None:
            return
        gpu_pool = self.scheduler._gpu_block_pool
        # Residency (keyed) and usable capacity (what the union lookup can
        # serve: GPU content ready on every rank, host members not staged for
        # an assembling unit) are different gauges.
        gpu_all, gp = self.index(gpu_pool)
        host_all, hp = self.index(self.rt.cpu_block_pool)
        tier = gpu_pool.tier
        units = getattr(self.rt, "units", None)
        gpu, _ = self.index(
            gpu_pool, None if tier is None else tier.is_ready
        )
        host, _ = self.index(
            self.rt.cpu_block_pool,
            None if units is None else (lambda b: not units.is_staged(b)),
        )
        union = gpu.keys() | host.keys()
        union_all = gpu_all.keys() | host_all.keys()
        c = self.rt.counters
        c.gpu_host_overlap = len(gp & hp)
        c.keyed_union_units = c.keyed_union_tokens = 0
        c.restorable_units = c.restorable_tokens = 0
        c.union_restorable_units = c.union_restorable_tokens = c.split_units = 0
        covered = set()
        for identity, unit in list(self.units.items()):
            if not unit.keys.intersection(union_all):
                self.members -= len(unit.keys)
                del self.units[identity]
                continue
            if unit.keys <= union_all:
                c.keyed_union_units += 1
                c.keyed_union_tokens += unit.tokens
            if unit.keys <= host.keys():
                c.restorable_units += 1
                c.restorable_tokens += unit.tokens
                covered.update(host[key] for key in unit.keys)
            if unit.keys <= union:
                c.union_restorable_units += 1
                c.union_restorable_tokens += unit.tokens
                if not unit.keys <= gpu.keys() and not unit.keys <= host.keys():
                    c.split_units += 1
        self.superseded_keys.intersection_update(union_all)
        # Every keyed physical host block, not the key -> block map: two host
        # copies of one key (legal after supersede_host_copies) are two
        # blocks, and the map keeps only one of them.
        orphans = {
            b.block_id for b in self.rt.cpu_block_pool.blocks
            if not b.is_null and b.block_hash is not None
        } - covered
        c.orphan_blocks = len(orphans)
        c.orphan_unclassified = self._unclassified(orphans)
        c.manifest_members = self.members
        c.census_scans += 1

    def _unclassified(self, orphans) -> int:
        """Orphan host blocks that are neither a recurrent
        checkpoint (a recurrent-group key: a restore point on its own, the
        FA prefix may be on the GPU). A lost unit's pages are un-keyed,
        so no fragment pairs with an endpoint any more."""
        pool = self.rt.cpu_block_pool
        groups = self.scheduler.cpu_kv_cache_config.kv_cache_groups
        count = 0
        for block_id in orphans:
            block = pool.blocks[block_id]
            keys = [block.block_hash,
                    *pool.cached_block_hashes_by_block.get(block_id, ())]
            if any(key is not None and isinstance(
                groups[get_group_id(key)].kv_cache_spec, MambaSpec
            ) for key in keys):
                continue
            count += 1
        return count

    def superseded(self, block):
        if block.block_hash is not None:
            self.superseded_keys.add(block.block_hash)

    def on_reclaim(self, block, collateral=False):
        if block.block_hash is None or block.is_null:
            return
        sched = self.scheduler
        key = block.block_hash
        owners = self.units.owners_of(key)
        if sched._gpu_block_pool.evict_groups.junctions.is_junction(block):
            kind = "junction"
        elif len(owners) > 1:
            kind = "shared_prefix"
        elif owners:
            kind = "finished_unit"
        elif key in self.superseded_keys:
            kind = "superseded_checkpoint"
        elif isinstance(sched.cpu_kv_cache_config.kv_cache_groups[
            get_group_id(key)
        ].kv_cache_spec, MambaSpec):
            # No producer provenance here: do not call every orphan superseded.
            kind = "unowned_checkpoint"
        else:
            kind = "unclassified"
        cause = "collateral" if collateral else "pop"
        self.reclaims[(str(get_group_id(key)), kind, cause)] += 1
        c = self.rt.counters
        if collateral:
            c.collateral_blocks += 1
        else:
            c.reclaimed_blocks += 1
            self.reclaim_step += 1

    def step(self, output):
        self._drain_remember()
        c = self.rt.counters
        c.reclaim_step_max = max(c.reclaim_step_max, self.reclaim_step)
        if self.reclaim_step:
            logger.debug(
                "SimpleCPU tier census reclaim: step=%d blocks=%d cumulative=%s",
                self.scheduler._sched_step, self.reclaim_step, dict(self.reclaims),
            )
        self.reclaim_step = 0
        c.preemption_flushes += bool(output.preempted_req_ids)
        c.preempted_requests += len(output.preempted_req_ids or ())
        sched = self.scheduler
        dropped = sched.boundary_store_stats.dropped_not_hashed
        c.commit_dropped_not_hashed += max(0, dropped - self._last_dropped)
        self._last_dropped = dropped
        c.in_flight_peak = max(c.in_flight_peak,
            len(sched._store_event_to_blocks) + len(sched._load_event_to_reqs))
        if sched._sched_step % 256 == 0:
            self.scan()

    def _step_index(self, pool, usable=None, kind="keyed"):
        """``index(pool)`` at most once per scheduler step:
        every waiting request's admission attempt observes the same pools.
        A read-only observation, so within-step staleness is harmless."""
        step = self.scheduler._sched_step
        memo = getattr(self, "_index_memo", None)
        if memo is None or memo[0] != step:
            memo = self._index_memo = (step, {})
        got = memo[1].get((id(pool), kind))
        if got is None:
            got = memo[1][(id(pool), kind)] = self.index(pool, usable)
        return got

    def _usable_gpu(self, pool):
        tier = pool.tier
        return None if tier is None else tier.is_ready

    def _usable_host(self):
        units = getattr(self.rt, "units", None)
        return None if units is None else (lambda b: not units.is_staged(b))

    def _holding(self, keys):
        """The census units holding any of ``keys`` (through the key index,
        not a scan of every unit)."""
        if not isinstance(self.units, _UnitDirectory):
            return list(self.units.values())
        ids = {i for key in keys for i in self.units.by_key.get(key, ())}
        return [self.units[i] for i in ids]

    def _build_union(self, gpu, host) -> set:
        return gpu.keys() | host.keys()

    def _step_union(self, gpu, host) -> set:
        """The key universe of both pools, built at most once per scheduler
        step with the pool indexes, not per lookup."""
        memo = self._index_memo[1]
        union = memo.get("union")
        if union is None:
            union = memo["union"] = self._build_union(gpu, host)
        return union

    def lookup(self, request, local, *, retry: bool = True):
        """Observe a lookup of ``request`` (``local`` GPU-hit tokens).
        ``retry`` False: a re-observation of an admission still waiting for a
        restore slot, never a lookup retry."""
        total = max(1, request.num_tokens - 1)
        best, layout = local, "none"
        # The observation is built from what the live lookup could
        # serve (ready GPU content, host members not staged), not from raw
        # residency, so an unready checkpoint cannot inflate the hit. Its
        # key universe is built once per step and only the units on the
        # request's hash chain are visited.
        gpu_pool = self.scheduler._gpu_block_pool
        gpu, _ = self._step_index(gpu_pool, self._usable_gpu(gpu_pool), "usable")
        host, _ = self._step_index(
            self.rt.cpu_block_pool, self._usable_host(), "usable"
        )
        union = self._step_union(gpu, host)
        units = (
            self.units.on_chain(request.block_hashes)
            if isinstance(self.units, _UnitDirectory)
            else self.units.values()
        )
        for unit in units:
            index = unit.tokens // self.scheduler.hash_block_size - 1
            if (unit.tokens > total or index >= len(request.block_hashes)
                    or request.block_hashes[index] != unit.endpoint
                    or not unit.keys <= union or unit.tokens <= best):
                continue
            best = unit.tokens
            positional = unit.keys - unit.checkpoints
            if unit.checkpoints <= gpu.keys() and not positional <= gpu.keys():
                layout = "gpu_ckpt_host_fa"
            elif unit.checkpoints <= host.keys() and not positional <= host.keys():
                layout = "host_ckpt_gpu_fa"
            else:
                layout = "mixed_positional"
        if retry and request.request_id in self.lookups:
            self.rt.counters.lookup_retries += 1
        checkpoints = {
            make_block_hash_with_group_id(
                request.block_hashes[local // self.scheduler.hash_block_size - 1], gid
            )
            for gid, group in enumerate(
                self.scheduler.cpu_kv_cache_config.kv_cache_groups
            )
            if local > 0 and isinstance(group.kv_cache_spec, MambaSpec)
        }
        positions = {u.position for u in self._holding(checkpoints)
                     if u.checkpoints & checkpoints}
        position = next(iter(positions)) if len(positions) == 1 else "unknown"
        self.lookups[request.request_id] = (
            local, total, best, layout, position, bool(checkpoints)
        )

    def admission(self, req_id, external, pending):
        # A restore attempt is admitted even without a census lookup observation
        # (or when its pending plan was lost). Retries are separate attempts;
        # repeated observations of the same outstanding attempt are not.
        if external > 0:
            blocks = pending[0] if pending is not None else ()
            count = sum(not b.is_null for group in blocks for b in group)
            self.pending_loads.setdefault(req_id, (external, count))
        observation = self.lookups.pop(req_id, None)
        if observation is None or req_id in self.admitted:
            return
        self.admitted.add(req_id)
        local, total, union, layout, gpu_position, recurrent = observation
        if recurrent:
            self.rt.counters.gpu_readiness_unknown_hits += 1
        if union > local + external:
            self.rt.counters.split_lost += 1
            self.rt.counters.split_lost_tokens += union - local - external
            self.reclaims[("all", layout, "split_lost")] += 1
            # By layout, so a reciprocal-layout adoption gap is
            # visible without parsing the reclaim labels.
            name = "split_lost_" + layout
            if hasattr(self.rt.counters, name):
                setattr(self.rt.counters, name, getattr(self.rt.counters, name) + 1)
                tname = "split_lost_tokens_" + layout
                if hasattr(self.rt.counters, tname):
                    setattr(self.rt.counters, tname,
                            getattr(self.rt.counters, tname) + union - local - external)
        cls = ("mixed" if local and external else "host" if external
               else "gpu" if local else "cold")
        c = self.rt.counters
        name = "returns_" + cls
        setattr(c, name, getattr(c, name) + 1)
        position = gpu_position if local and recurrent else "none"
        if external and pending is not None:
            blocks, _, _ = pending
            keys = set()
            for group in blocks:
                for block in group:
                    if not block.is_null:
                        keys.add(block.block_hash)
                        keys.update(self.rt.cpu_block_pool.cached_block_hashes_by_block.get(
                            block.block_id, ()
                        ))
            has_recurrent = recurrent or any(
                isinstance(group.kv_cache_spec, MambaSpec)
                for group in self.scheduler.cpu_kv_cache_config.kv_cache_groups
            )
            if has_recurrent:
                positions = {u.position for u in self._holding(keys)
                             if u.checkpoints & keys}
                position = next(iter(positions)) if len(positions) == 1 else "unknown"
        if position != "none":
            name = "checkpoint_" + position
            setattr(c, name, getattr(c, name) + 1)
            self.reclaims[("all", cls, name)] += 1
        logger.info(
            "SimpleCPU tier return: req=%s cls=%s local_tokens=%d external_tokens=%d "
            "ext_frac=%.6f checkpoint=%s phase=admitted",
            req_id, cls, local, external, external / total, position,
        )

    def load_done(self, req_id, failed, cancelled=False):
        pending = self.pending_loads.pop(req_id, None)
        if pending is None:
            return
        # A request that went away (cancel, abort, finish) cancelled its
        # restore; that is not a transfer failure. Real failures:
        # a failed transaction or an invalid lease/epoch/source.
        if cancelled:
            self.rt.counters.restore_cancelled += 1
            return
        # Read the lease only for accounting: never change publication, refs,
        # copy issue, or the worker's result. A successful DMA can still belong
        # to an invalidated plan or an old epoch.
        sched = self.scheduler
        if not failed and sched is not None:
            state = (getattr(sched, "_reqs_to_load", {}).get(req_id)
                     or getattr(sched, "_abandoned_reqs_to_load", {}).get(req_id))
            if state is not None:
                failed = state.failed or state.finished
                lease = state.lease
                if lease is not None:
                    failed = (failed or lease.epoch != sched.cache_epoch
                              or not lease.sources_valid(self.rt.cpu_block_pool))
        if failed:
            self.rt.counters.restore_failed += 1
            # The write-back log enumerates this dataclass. Keep its cumulative
            # mirror in sync without adding a scheduler/allocator hook.
            writeback = getattr(sched, "writeback", None)
            if writeback is not None:
                writeback.counters.restore_failed = self.rt.counters.restore_failed
        else:
            tokens, blocks = pending
            c = self.rt.counters
            c.restore_completed_requests += 1
            c.restore_completed_tokens += tokens
            c.restore_completed_blocks += blocks

    def gone(self, req_id):
        self.load_done(req_id, failed=False, cancelled=True)
        self.lookups.pop(req_id, None)
        self.admitted.discard(req_id)

    def reset(self):
        self.units.clear()
        self.superseded_keys.clear()
        self.members = 0
        self.lookups.clear()
        self.admitted.clear()
        # Reset settles every admitted attempt now, as cancelled (an expected
        # reset is not a transfer failure). Late DMA
        # completions still release storage through the existing lifecycle,
        # but cannot count these attempts a second time.
        for req_id in tuple(self.pending_loads):
            self.load_done(req_id, failed=False, cancelled=True)
        self._last_dropped = 0


# These entrypoints are observational instrumentation. With the default-off
# flag, avoid pool scans, key proofs, unit bookkeeping and histogram updates.
def _census_debug_only(method):
    def guarded(self, *args, **kwargs):
        if not self.enabled:
            if method.__name__ == "reset":
                self.pending_loads.clear()
            elif method.__name__ in ("load_done", "gone") and args:
                self.pending_loads.pop(args[0], None)
            return None
        return method(self, *args, **kwargs)
    guarded.__name__ = method.__name__
    guarded.__doc__ = method.__doc__
    return guarded


for _method_name in (
    "_bind_hooks", "prove", "remember", "_drain_remember", "_remember_now", "scan",
    "_unclassified", "superseded", "on_reclaim", "step", "_step_index",
    "_holding", "_step_union", "lookup", "admission", "load_done", "gone",
    "reset",
):
    setattr(
        TierCensus,
        _method_name,
        _census_debug_only(getattr(TierCensus, _method_name)),
    )


# =============================================================================
# Metrics: the once-per-admission outcome ledger, the
# request-history reuse index, per-interval deltas, T*, and the run-length
# detectors for a whole-host freeze (flat stored, stalled restores,
# whole-host flush).
#
# Everything below is pure Python: no engine imports, no pool access, no
# decisions. It runs with the census flag OFF (the production path) and is
# fed by ``TierRuntime``. The reuse denominator comes from REQUEST HISTORY
# (what an earlier finished request computed), never from the census union,
# which shrinks with every block the pipeline loses (circular).
# =============================================================================

#: Write-back counters whose rise while ``stored_blocks`` is flat means copy
#: work was eligible and did not land.
FLAT_PENDING_KEYS = (
    "demote_missed_running", "demote_missed_escape", "demote_dropped_host_full",
    "demote_dropped_credits", "demote_host_busy",
    "demote_assembly_waits", "demote_gpu_tight",
)


# The request history is sized in the hash unit the server uses
# (``--prefix-match-unit``), never the page size.
# It holds at least HISTORY_MIN_CHAINS distinct 262K-token chains and
# HISTORY_CACHE_MARGIN x what the GPU and host tiers can hold together.
HISTORY_DEFAULT_CAPACITY = 1 << 17
HISTORY_MIN_CHAINS = 64
HISTORY_CHAIN_TOKENS = 262_144
HISTORY_CACHE_MARGIN = 4


def history_capacity(hash_block_size: int, cache_tokens: int = 0) -> int:
    """Distinct hashes in the shared prompt/full history for this geometry."""
    tokens = max(HISTORY_MIN_CHAINS * HISTORY_CHAIN_TOKENS,
                 HISTORY_CACHE_MARGIN * max(0, cache_tokens))
    return max(HISTORY_DEFAULT_CAPACITY, tokens // max(1, hash_block_size))


class HistoryIndex:
    """Which prefix chains an earlier FINISHED request already computed.

    ``reusable_tokens`` of a new request = the longest prefix of its block-hash
    chain found here, floored to the layout's restore grid and to
    ``num_tokens - 1`` (the last token is always recomputed). Hash chains are
    parent-dependent, so one hash present means its whole prefix was computed.

    One insertion-ordered dict holds at most ``capacity`` distinct hashes,
    refreshed on re-add. Its boolean value records whether a hash was inside
    any finished request's PROMPT (the conservative lower bound: RecoverSSM +
    PP publishes no decode checkpoint). All keys form the ``full`` basis
    (prompt + generated tokens; the upper bound). Re-adding a hash as decode
    cannot erase its prompt flag.

    ``TierRuntime.bind_geometry`` sizes the shared capacity with
    ``history_capacity``: one hash per ``--prefix-match-unit`` tokens, so a
    262K chain is 16,384 hashes on 27B and 4,096 on FN. Every eviction counts
    toward ``evicted_full``; evicting a flagged hash also counts toward
    ``evicted_prompt``. Decode can therefore displace prompt history sooner
    than with separate limits. A dropped prompt hash can only DEFLATE the
    denominator, so once the PROMPT basis truncated, every rho_bar / T* / E /
    E_tier claim from here on is invalid (``efficiency`` returns them None and
    ``ceiling_valid`` False) rather than flattered; ``evicted`` is the total
    number of basis entries lost.

    ``_tails`` remembers the hash of each finished request's last full PROMPT
    block (its resume point). A new request whose matched prefix passes one
    is a returning turn of that chain; a match that ends inside other
    requests' prompts without passing a tail is a shared prefix only (a first
    turn behind a common system prompt). Bounded like ``_hashes``.
    """

    def __init__(self, capacity: int = HISTORY_DEFAULT_CAPACITY) -> None:
        self.capacity = capacity
        self._hashes: dict[bytes, bool] = {}
        self._tails: dict[bytes, None] = {}
        self.evicted_prompt = 0
        self.evicted_full = 0
        self.finished = 0

    @property
    def evicted(self) -> int:
        return self.evicted_prompt + self.evicted_full

    def note_finished(self, block_hashes, hash_block_size: int,
                      prompt_tokens: int, computed_tokens: int) -> None:
        """A request finished having computed ``computed_tokens`` tokens."""
        size = max(1, hash_block_size)
        full = min(len(block_hashes), max(0, computed_tokens) // size)
        prompt = min(full, max(0, prompt_tokens) // size)
        index = self._hashes
        for i in range(full):
            h = block_hashes[i]
            in_prompt = index.pop(h, False) or i < prompt
            index[h] = in_prompt
        if prompt > 0:
            tails = self._tails
            tail = block_hashes[prompt - 1]
            tails.pop(tail, None)
            tails[tail] = None
            for t in list(islice(tails, max(0, len(tails) - self.capacity))):
                del tails[t]
        dropped = max(0, len(index) - self.capacity)
        # Collect the oldest keys in one pass before mutating the dict.
        # Repeated next(iter(index)) would rescan deleted slots for each key.
        for h in list(islice(index, dropped)):
            self.evicted_prompt += index.pop(h)
        self.evicted_full += dropped
        self.finished += 1

    def matched_blocks(self, block_hashes, basis: str = "prompt") -> int:
        index = self._hashes
        n = 0
        for h in block_hashes:
            if (not index.get(h, False) if basis == "prompt" else h not in index):
                break
            n += 1
        return n

    def reaches_tail(self, block_hashes) -> bool:
        """The chain's matched prompt-basis prefix (before any restore-grid
        floor) passes an earlier finished request's prompt tail: a returning
        turn, not just a shared prefix."""
        tails = self._tails
        if not tails or not block_hashes:
            return False
        n = self.matched_blocks(block_hashes, "prompt")
        return any(block_hashes[i] in tails for i in range(n))

    def reusable_tokens(self, block_hashes, num_tokens: int,
                        hash_block_size: int, align: int | None = None,
                        basis: str = "prompt") -> int:
        size = max(1, hash_block_size)
        tokens = min(self.matched_blocks(block_hashes, basis) * size,
                     max(0, num_tokens - 1))
        grid = max(1, align or size)
        return tokens // grid * grid


class _Fields:
    """Plain attribute bag with the ledger fields at zero (tests, no runtime)."""

    def __init__(self) -> None:
        for name in LEDGER_FIELDS:
            setattr(self, name, 0)


class OutcomeLedger:
    """Implements ``kv_host_tier_iface.OutcomeSink`` plus the request-history
    side (``reusable_tokens(request)`` / ``note_finished(request)``).

    ``record`` is idempotent per ``(request_id, step)``: a lookup retry or a
    second route to the same admission cannot count twice. Counters are written
    into ``counters`` (``TierCounters`` in production) so the existing delta
    export carries them; ``snapshot()`` reads them back.
    """

    SEEN_MAX = 8192
    PENDING_MAX = 4096

    def __init__(self, counters=None, history: HistoryIndex | None = None,
                 hash_block_size: int = 0, align: int | None = None,
                 basis: str = "prompt", geometry=None) -> None:
        self.c = counters if counters is not None else _Fields()
        self.history = history if history is not None else HistoryIndex()
        self.hash_block_size = hash_block_size
        self.geometry = geometry  # () -> hash block size, bound late by the runtime
        self.align = align
        self.basis = basis
        self.miss_tokens: dict[str, int] = {}
        self.miss_n: dict[str, int] = {}
        self._seen: OrderedDict = OrderedDict()
        # MERLIN_KV_TIER_DIAG=1: one "first-turn" line per first-turn outcome.
        self._diag = _diag_on()
        # request id -> the decode-inclusive reusable tokens computed beside R_i
        self._full: OrderedDict = OrderedDict()
        # request ids whose reusable prefix passes an earlier prompt tail
        # (``HistoryIndex.reaches_tail``): returning turns, never first turns
        self._returning: OrderedDict = OrderedDict()
        # request id -> (host tokens, host tokens counted in the capped numerator)
        self._pending_restore: OrderedDict = OrderedDict()
        # request id -> its block-hash chain (kept from ``reusable_tokens`` to
        # ``record`` so an ``unknown`` miss can be resolved)
        self._hashes: OrderedDict = OrderedDict()
        # (block hashes past the adopted boundary) -> (cause, age) | None:
        # ``UnitTable.loss_cause``, pure; bound late by the runtime. None:
        # causes stay as the plans recorded them.
        self.loss_lookup = None

    # -- history side ---------------------------------------------------------
    def _hbs(self) -> int:
        return self.hash_block_size or (self.geometry() if self.geometry else 0) or 0

    def _reusable(self, request, basis: str) -> int:
        hashes = getattr(request, "block_hashes", None)
        if not hashes:
            return 0
        hbs = self._hbs()
        if hbs <= 0:
            # Unknown geometry: say 0 (E reads None) rather than a wrong number.
            self.c.outcome_geometry_missing += 1
            return 0
        return self.history.reusable_tokens(
            hashes, request.num_tokens, hbs, self.align, basis
        )

    def reusable_tokens(self, request) -> int:
        """R_i for ``request`` (OutcomeSink hook used by ``plans._record_outcome``).
        Also remembers the decode-inclusive bound for the admission's record."""
        r = self._reusable(request, self.basis)
        self._full[request.request_id] = self._reusable(request, "full")
        while len(self._full) > self.PENDING_MAX:
            self._full.popitem(last=False)
        hashes = getattr(request, "block_hashes", None)
        if r > 0 and hashes and self.history.reaches_tail(hashes):
            self._returning[request.request_id] = None
            while len(self._returning) > self.PENDING_MAX:
                self._returning.popitem(last=False)
        if hashes and self.loss_lookup is not None:
            self._hashes[request.request_id] = hashes
            while len(self._hashes) > self.PENDING_MAX:
                self._hashes.popitem(last=False)
        return r

    def reusable_full_tokens(self, request) -> int:
        return self._reusable(request, "full")

    def note_finished(self, request) -> None:
        hashes = getattr(request, "block_hashes", None)
        hbs = self._hbs()
        if hashes and hbs > 0:
            self.history.note_finished(
                hashes, hbs, request.num_prompt_tokens, request.num_computed_tokens,
            )
            # Cumulative levels, exported as deltas like every other counter.
            self.c.history_truncated = self.history.evicted_prompt
            self.c.history_truncated_full = self.history.evicted_full

    def resolve_miss_cause(self, request_id: str, boundary: int) -> str | None:
        """The loss cause the unit table recorded for the deepest lost unit on
        this request's chain past ``boundary`` (tokens), the same lookup
        ``TierPlans._miss_cause`` makes. Pure: no counter, no state change.
        None when unknown (no chain, no lookup, geometry unbound, no loss
        record: e.g. never copied)."""
        hashes = self._hashes.get(request_id)
        lookup = self.loss_lookup
        hbs = self._hbs()
        if not hashes or lookup is None or hbs <= 0:
            return None
        past = hashes[max(0, boundary) // hbs:]
        if not past:
            return None
        try:
            got = lookup(past)
        except Exception:  # a metrics lookup must never break admission
            return None
        return got[0] if got else None

    # -- OutcomeSink ----------------------------------------------------------
    def record(self, outcome) -> bool:
        c = self.c
        key = (outcome.request_id, outcome.step)
        if key in self._seen:
            c.outcomes_duplicate += 1
            return False
        self._seen[key] = None
        while len(self._seen) > self.SEEN_MAX:
            self._seen.popitem(last=False)
        reusable = max(0, outcome.reusable_tokens)
        gpu, host = max(0, outcome.gpu_tokens), max(0, outcome.host_tokens)
        # A first turn: nothing reusable, or only a prefix it shares with other
        # requests (a common system prompt, R <= the shared-prefix boundary).
        # A request whose R passes an earlier prompt tail is its chain's next
        # turn even when R <= the junction (the junction can sit past the
        # previous prompt's tail, e.g. 8,000 vs 8,128 on a 4,096-token FA block).
        returning = outcome.request_id in self._returning
        self._returning.pop(outcome.request_id, None)
        first = bool(outcome.first_turn) or (
            not returning
            and reusable <= max(0, getattr(outcome, "shared_prefix_tokens", 0))
        )
        c.outcomes_recorded += 1
        c.outcome_prompt_tokens += outcome.prompt_tokens
        if first:
            c.outcomes_first_turn += 1
            if self._diag:
                logger.info(
                    "KV tier diag: first-turn t=%.3f req=%s prompt=%d "
                    "shared_prefix=%d reusable=%d gpu=%d host=%d cached=%d",
                    time.monotonic(), outcome.request_id, outcome.prompt_tokens,
                    max(0, getattr(outcome, "shared_prefix_tokens", 0)),
                    reusable, gpu, host, gpu + host,
                )
        else:
            c.outcome_prompt_tokens_returns += outcome.prompt_tokens
        c.reusable_tokens += reusable
        full = max(reusable, self._full.pop(outcome.request_id, 0))
        c.reusable_full_tokens += full
        c.adopted_gpu_tokens += gpu
        c.adopted_host_tokens += host
        gpu_capped = min(gpu, reusable)
        host_capped = min(gpu + host, reusable) - gpu_capped
        c.adopted_capped_tokens += gpu_capped + host_capped
        lost = max(0, reusable - gpu - host)
        c.lost_tokens += lost
        cause = None
        if lost:
            cause = outcome.miss_cause or "unknown"
            if cause == "unknown":
                # Plans picks restore_cap_wait or unknown; the loss cause
                # the unit table recorded for the lost unit is the real one.
                cause = self.resolve_miss_cause(outcome.request_id, gpu + host) or cause
            self.miss_tokens[cause] = self.miss_tokens.get(cause, 0) + lost
            self.miss_n[cause] = self.miss_n.get(cause, 0) + 1
        self._hashes.pop(outcome.request_id, None)
        if outcome.restore_requested:
            c.outcome_restores_requested += 1
            if outcome.restore_completed is None:
                self._pending_restore[outcome.request_id] = (
                    host, host_capped,
                    # What this admission added above, for ``settled_snapshot``.
                    {
                        "outcomes_recorded": 1,
                        "outcomes_first_turn": int(first),
                        "outcome_prompt_tokens": outcome.prompt_tokens,
                        "outcome_prompt_tokens_returns":
                            0 if first else outcome.prompt_tokens,
                        "reusable_tokens": reusable,
                        "reusable_full_tokens": full,
                        "adopted_gpu_tokens": gpu,
                        "adopted_host_tokens": host,
                        "adopted_capped_tokens": gpu_capped + host_capped,
                        "lost_tokens": lost,
                        "outcome_restores_requested": 1,
                    },
                    (cause, lost) if lost else None,
                )
                while len(self._pending_restore) > self.PENDING_MAX:
                    self._pending_restore.popitem(last=False)
            elif outcome.restore_completed:
                c.outcome_restores_completed += 1
            else:
                self._fail(host, host_capped)
        return True

    def _cancel(self, host: int, host_capped: int) -> None:
        """A cancelled restore keeps no host credit anywhere: not in the
        capped numerator (E) and not in the host share (h / combined / E_tier),
        which reads ``adopted_host_tokens - host_failed_tokens -
        host_cancelled_tokens``."""
        c = self.c
        c.outcome_restores_cancelled += 1
        c.host_cancelled_tokens += host
        c.adopted_capped_tokens -= host_capped
        c.lost_tokens += host_capped

    def _fail(self, host: int, host_capped: int) -> None:
        c = self.c
        c.outcome_restores_failed += 1
        c.host_failed_tokens += host
        c.adopted_capped_tokens -= host_capped
        c.lost_tokens += host_capped
        if host_capped:
            self.miss_tokens["restore_failed"] = (
                self.miss_tokens.get("restore_failed", 0) + host_capped
            )
            self.miss_n["restore_failed"] = self.miss_n.get("restore_failed", 0) + 1

    @property
    def pending_restores(self) -> int:
        """Admissions whose host restore was handed off and has not retired."""
        return len(self._pending_restore)

    def restore_retired(self, request_id: str, completed: bool | None) -> None:
        """The request's load retired. ``completed`` False: it failed and the
        request recomputes (its host tokens leave the adopted numerator);
        None: cancelled (no failure, no credit). Idempotent."""
        pending = self._pending_restore.pop(request_id, None)
        if pending is None:
            return
        host, host_capped = pending[:2]
        if completed:
            self.c.outcome_restores_completed += 1
        elif completed is False:
            self._fail(host, host_capped)
        else:
            self._cancel(host, host_capped)

    def forget(self, request_id: str) -> None:
        """The request finished or went away: an unretired restore is a
        cancelled one (no credit), never silently dropped with its credit."""
        self.restore_retired(request_id, None)
        self._hashes.pop(request_id, None)

    def snapshot(self) -> dict:
        snap = {name: getattr(self.c, name) for name in LEDGER_FIELDS}
        snap["miss_tokens"] = dict(self.miss_tokens)
        snap["miss_n"] = dict(self.miss_n)
        return snap

    def settled_snapshot(self) -> dict:
        """``snapshot`` without the admissions whose restore has not retired:
        each admission enters whole, numerator and
        denominator together, in the interval it settles in, so a later cancel
        or failure can never drive an interval's E or h below zero. The
        cumulative ``snapshot`` keeps the provisional credit; both agree once
        nothing is in flight."""
        snap = self.snapshot()
        for _host, _capped, added, missed in self._pending_restore.values():
            for name, n in added.items():
                snap[name] -= n
            if missed is not None:
                cause, lost = missed
                for book, n in ((snap["miss_tokens"], lost), (snap["miss_n"], 1)):
                    book[cause] -= n
                    if not book[cause]:
                        del book[cause]
        return snap


def interval_delta(cur: dict, prev: dict | None) -> dict:
    """Per-interval delta of two cumulative snapshots (numbers and one level of
    ``{name: number}`` dicts). The server's own hit-rate windows are 1,000
    requests, near-cumulative over a long run, and can hide a whole-host
    freeze for tens of minutes, so every qualification reads intervals."""
    prev = prev or {}
    out = {}
    for k, v in cur.items():
        if isinstance(v, dict):
            p = prev.get(k, {})
            out[k] = {n: x - p.get(n, 0) for n, x in v.items() if x - p.get(n, 0)}
        else:
            out[k] = v - prev.get(k, 0)
    return out


def _ratio(a, b):
    return (a / b) if b else None


def efficiency(d: dict, history_truncated: int | None = None) -> dict:
    """Reuse metrics from a (cumulative or interval) ledger mapping.

    ``history_truncated``: the CUMULATIVE count of prompt-basis hashes the
    bounded history dropped (an interval delta cannot say whether an earlier
    interval truncated); defaults to ``d['history_truncated']``. Non-zero
    means the reusable-token denominator is deflated from then on, so
    ``rho_bar`` / ``t_star`` / ``e`` / ``e_tier`` are None and
    ``ceiling_valid`` is False. ``g`` / ``h`` / ``combined`` /
    ``miss_share`` do not read the history and stay.

    * ``rho_bar``  = sum R / sum prompt           (token-weighted opportunity; first
      turns count with their shared-prefix R, ``outcome_prompt_tokens_returns``
      is the prompt volume of the returning turns alone)
    * ``g``        = sum GPU / sum prompt         (GPU adoption)
    * ``h``        = host-confirmed / (prompt - GPU)   (host share of GPU misses,
      the server's ``ext %`` definition)
    * ``t_star``   = (rho_bar - g) / (1 - g)      (the tier's ceiling on ``h``)
    * ``e``        = sum min(adopted, R) / sum R  (E; host failures removed)
    * ``e_tier``   = h / t_star
    * ``combined`` = (GPU + host-confirmed) / prompt; ``miss_share`` = 1 - combined
    """
    P = d.get("outcome_prompt_tokens", 0)
    R = d.get("reusable_tokens", 0)
    gpu = d.get("adopted_gpu_tokens", 0)
    host = (d.get("adopted_host_tokens", 0) - d.get("host_failed_tokens", 0)
            - d.get("host_cancelled_tokens", 0))
    out = {
        "rho_bar": _ratio(R, P),
        "g": _ratio(gpu, P),
        "h": _ratio(host, P - gpu),
        "e": _ratio(d.get("adopted_capped_tokens", 0), R),
        "combined": _ratio(gpu + host, P),
    }
    out["miss_share"] = None if out["combined"] is None else 1.0 - out["combined"]
    g, rho = out["g"], out["rho_bar"]
    out["t_star"] = (
        (rho - g) / (1.0 - g) if g is not None and rho is not None and g < 1.0 else None
    )
    out["e_tier"] = (
        out["h"] / out["t_star"]
        if out["h"] is not None and out["t_star"] and out["t_star"] > 0 else None
    )
    truncated = (d.get("history_truncated", 0) if history_truncated is None
                 else history_truncated)
    out["ceiling_valid"] = not truncated
    if truncated:
        for name in ("rho_bar", "t_star", "e", "e_tier"):
            out[name] = None
    return out


class FlatStoreDetector:
    """Fires when ``stored_blocks`` has not risen for ``flat_s`` seconds while
    copy work was eligible: a pending-work counter (``FLAT_PENDING_KEYS``) rose
    by at least ``min_work`` over the same ``flat_s`` window.

    The run-length bar: no flat stretch longer than one
    census interval (256 steps, ~5.2 min on the 27B; default 240 s here) with
    eligible work. Idle is not a freeze (nothing rose); a few missed blocks
    during a healthy lull are not one either (``min_work``). In a freeze the
    stores stop while the missed/dropped counters climb by hundreds; the
    detector fires one ``flat_s`` window after the last store. ``feed`` returns
    True on the sample that opens an episode; ``longest_flat_s`` is the worst
    eligible stretch seen (the report-only number every gate prints).
    """

    def __init__(self, flat_s: float = 240.0, min_work: int = 8) -> None:
        self.flat_s = flat_s
        self.min_work = min_work
        self.last_stored: int | None = None
        self.flat_since = 0.0
        self._window: deque = deque()
        self.episode_open = False
        self.episodes = 0
        self.longest_flat_s = 0.0
        self.current_flat_s = 0.0
        self.first_fire_t: float | None = None

    @staticmethod
    def pending_sum(pending) -> int | None:
        if pending is None:
            return None
        return sum(int(pending.get(k, 0) or 0) for k in FLAT_PENDING_KEYS)

    def feed(self, t: float, stored: int, pending=None) -> bool:
        work = self.pending_sum(pending)
        if self.last_stored is None or stored > self.last_stored:
            self.last_stored = stored
            self.flat_since = t
            self._window = deque([(t, work or 0)])
            self.episode_open = False
            self.current_flat_s = 0.0
            return False
        if work is None:
            return False
        w = self._window
        w.append((t, work))
        horizon = t - self.flat_s
        while len(w) > 1 and w[1][0] <= horizon:
            w.popleft()
        if work < w[0][1]:  # counters reset
            w.clear()
            w.append((t, work))
        eligible = work - w[0][1] >= self.min_work
        self.current_flat_s = (t - self.flat_since) if eligible else 0.0
        self.longest_flat_s = max(self.longest_flat_s, self.current_flat_s)
        if eligible and not self.episode_open and self.current_flat_s >= self.flat_s:
            self.episode_open = True
            self.episodes += 1
            if self.first_fire_t is None:
                self.first_fire_t = t
            return True
        return False


#: Miss causes that mean a returning chain wanted the host and did not get it
#: (``AdmissionOutcome.miss_cause``): with ``outcome_restores_requested`` they
#: make the restore demand the stall detector is conditioned on.
RESTORE_DEMAND_CAUSES = ("host_evicted", "missed_copy", "restore_cap_wait")


def restore_demand(restores_requested: int, miss_n: dict) -> int:
    """Cumulative restore demand: admissions that requested a host restore,
    plus admissions that lost reusable tokens the host should have served
    (``RESTORE_DEMAND_CAUSES``). Flat while every return is a GPU hit, so a
    flat ``restored_requests`` then is no stall."""
    return int(restores_requested) + sum(
        int(miss_n.get(cause, 0)) for cause in RESTORE_DEMAND_CAUSES
    )


def series_restore_demand(sample: dict, cum: int | None) -> int | None:
    """The cumulative restore demand after one ``SimpleCPU tier metrics:``
    sample. Lines that carry ``restore_demand`` give it directly; older lines
    carry the per-interval ``restore_outcomes`` and ``miss_<cause>`` counts,
    which are summed onto ``cum``. None when the line has neither."""
    if "restore_demand" in sample:
        return int(sample["restore_demand"])
    if "restore_outcomes" not in sample:
        return cum
    step = int(sample.get("restore_outcomes", 0)) + sum(
        int(sample.get("miss_" + cause, 0)) for cause in RESTORE_DEMAND_CAUSES
    )
    return (cum or 0) + step


def _diag_on() -> bool:
    """MERLIN_KV_TIER_DIAG=1 (debug log lines only)."""
    try:
        from vllm.merlin.envs import read_registered

        return bool(read_registered("MERLIN_KV_TIER_DIAG"))
    except Exception:  # a debug switch must never break the census
        return False


class RestoreProgressDetector:
    """Fires when ``restored_requests`` has not risen for ``window_s`` seconds
    after the first restore (strictly rising per 10 min) while restores were
    wanted.

    With ``demand`` (cumulative, ``restore_demand``) the stall clock starts at
    the first sample that shows demand above its level at the last restore,
    and an episode opens only if demand also rose within the last
    ``window_s``: a quiet spell where every return is a GPU hit is not a stall.
    ``pending`` (restores handed off and not retired) counts as demand at
    every sample, so a wedge whose waiters stop admitting still fires.
    Without ``demand`` (None) the clock runs from the last restore."""

    def __init__(self, window_s: float = 600.0) -> None:
        self.window_s = window_s
        self.last: int | None = None
        self.last_t = 0.0
        self.open = False
        self.episodes = 0
        self.first_fire_t: float | None = None
        self.demand_base: int | None = None
        self.demand_last: int | None = None
        self.demand_since: float | None = None
        self.demand_rose_t: float | None = None

    def _note_demand(self, t: float, demand: int | None, pending: int = 0) -> None:
        if demand is None:
            return
        if pending > 0:
            # a restore is in flight: someone waits on it now
            if self.demand_since is None:
                self.demand_since = t
            self.demand_rose_t = t
        if self.demand_base is None or demand < self.demand_base:
            # first sample, or the counters restarted
            self.demand_base = self.demand_last = demand
            return
        last = self.demand_last if self.demand_last is not None else demand
        self.demand_last = demand
        if demand <= last:
            return
        quiet = (self.demand_rose_t is not None
                 and t - self.demand_rose_t >= self.window_s)
        if self.demand_since is None or quiet:
            # the first unmet demand since the last restore, or the first
            # after a quiet spell: the stall clock starts here
            self.demand_since = t
        self.demand_rose_t = t

    def feed(self, t: float, restored: int, demand: int | None = None,
             pending: int = 0) -> bool:
        if self.last is None:
            self.last, self.last_t = restored, t
            self.demand_base = self.demand_last = demand
            return False
        if restored > self.last:
            self.last, self.last_t, self.open = restored, t, False
            self.demand_base = self.demand_last = demand
            self.demand_since = self.demand_rose_t = None
            return False
        self._note_demand(t, demand, pending)
        if self.last <= 0 or self.open:
            return False
        if demand is None:
            start = self.last_t
        elif self.demand_since is None or self.demand_rose_t is None:
            return False
        elif t - self.demand_rose_t >= self.window_s:
            return False  # demand went quiet again: nothing waits on a restore
        else:
            start = self.demand_since
        if t - start >= self.window_s:
            self.open = True
            self.episodes += 1
            if self.first_fire_t is None:
                self.first_fire_t = t
            return True
        return False


class HostFlushDetector:
    """Fires when the cached host rows fall by >= ``frac`` of capacity between
    two samples: whole-component eviction, not per-unit expiry (for example a
    large share of the rows dropping in one interval to admit one unit).

    Fixed at 50 % of the host rows, independent of unit size. So on a small
    tier one long-context admission that evicts most of the host fires too.

    ``rearm`` forgets the previous sample: a coordinated prefix-cache reset
    un-keys every row on purpose and must not read as a flush."""

    def __init__(self, frac: float = 0.5) -> None:
        self.frac = frac
        self.prev: int | None = None
        self.events: list[tuple[float, int, int]] = []
        self.rearms = 0

    def threshold(self, capacity: int) -> float:
        return self.frac * capacity

    def rearm(self) -> None:
        self.prev = None
        self.rearms += 1

    def feed(self, t: float, cached: int, capacity: int) -> bool:
        prev, self.prev = self.prev, cached
        if prev is None or capacity <= 0:
            return False
        if prev - cached >= self.threshold(capacity):
            self.events.append((t, prev, cached))
            return True
        return False
