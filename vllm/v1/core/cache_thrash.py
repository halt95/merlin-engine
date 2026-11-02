# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Prefix-cache thrash fixes for hybrid (Mamba/GDN) models.

A hybrid prefix hit resumes only where a session's whole chain of blocks (the
attention prefix and the GDN state) survives, so losing any one block of a
session loses all of it. Under oversubscription the stock policy breaks many
sessions a block at a time, and follow-up turns re-prefill in full. Two
scheduler-side rules, both default ON (see ``vllm/merlin/envs.py``):

* ``VLLM_E2_SCHED_COHERENT_YIELD``: need-sized, session-coherent yields.
  ``SessionYield`` picks ONE session whose release covers the shortfall (or
  several, never breaking a waiting session for less than half of what it
  frees), otherwise the head waits while running work can progress. The
  freed blocks of a yielded session are registered with the block pool as one
  ``EvictGroups`` group: the first block reused un-hashes its cached-free
  peers and moves them to the front of the free queue, so a broken session is
  consumed before any intact one.
* ``VLLM_E2_SCHED_MISS_DEFER_S``: cost-aware admission (scheduler.py). A
  young cold miss behind pinned waiters is parked for at most this many
  seconds (the fairness bound), and only when admitting it would need a pin
  yielded or would un-hash a cached block of a waiter's prompt past its pin
  (``SessionYield.evicts_needed``, counted over the whole prompt at the
  admission gate). Otherwise it is admitted at once.

Hook sites: ``BlockPool`` (evict_groups), ``KVCacheManager`` (thrash) and
``Scheduler``; everything with logic lives here.
"""

from __future__ import annotations

import itertools
import time
from collections.abc import Callable, Iterable, Sequence
from typing import TYPE_CHECKING, Protocol

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.v1.core.fgpc import JunctionCheckpoints
from vllm.v1.core.kv_cache_utils import KVCacheBlock, get_block_hash
from vllm.v1.core.sched.request_queue import SchedulingPolicy
from vllm.v1.kv_cache_interface import AttentionSpec, MambaSpec
from vllm.v1.request import Request, RequestStatus

if TYPE_CHECKING:
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.sched.scheduler import Scheduler

logger = init_logger(__name__)

# Bound (s) on a head's wait for natural frees when miss deferral is off.
DEFAULT_HEAD_BOUND_S = 20

# Decision log for diagnosis: None (the default) records nothing; a
# list collects ("yield" | "park" | "wait" | "break" | "admit", fields).
EVENTS: list[tuple[str, dict]] | None = None


def emit(ev: str, **fields) -> None:
    if EVENTS is not None:
        EVENTS.append((ev, fields))


# One holder of a session: ("grace", finished request id) or
# ("waiter", waiting request id).
Holder = tuple[str, str]


class HostTierEviction(Protocol):
    """Bound by the tier manager on the GPU pool's ``evict_groups``.

    All calls run on the scheduler ownership path without waiting or copying.
    ``is_host_backed`` is read-only (also used by the eviction predictor): it
    must validate current keys, aliases, content generations and cache epoch,
    including the complete restore-unit dependencies. A clean bit alone is
    insufficient. A one-member query asks whether reusing that member loses
    any restore dependency; a group query certifies the whole broken unit.

    Notifications must preserve host keys and must not mutate GPU hashes,
    refs or queues. ``demote`` records an already-backed unit leaving L1;
    ``unit_lost`` counts a destructive break. In-flight copies are not backing.
    """

    def is_host_backed(self, blocks: Sequence[KVCacheBlock]) -> bool: ...

    def demote(self, blocks: Sequence[KVCacheBlock]) -> None: ...

    def unit_lost(self, blocks: Sequence[KVCacheBlock]) -> None: ...


class EvictGroups:
    """Session-coherent eviction for one ``BlockPool``: the cached-free blocks
    of a session whose pin was yielded, registered as one group."""

    def __init__(self, pool: BlockPool) -> None:
        self._pool = pool
        self._groups: dict[int, list[KVCacheBlock]] = {}
        self._block_group: dict[int, int] = {}
        self._next_id = 0
        # GPU pool only; CPU eviction stays destructive.
        self.host_tier: HostTierEviction | None = None
        # Junction checkpoints (see vllm/v1/core/fgpc.py): evicted before any
        # other cached block, never grouped.
        self.junctions = JunctionCheckpoints(pool)

    def __bool__(self) -> bool:
        return bool(self._block_group)

    def register(self, blocks: Iterable[KVCacheBlock]) -> int:
        """Register the cached-free blocks among ``blocks`` (this pool's,
        hashed, ref 0, in no group yet, not a shared junction checkpoint) as
        one session. Returns the number registered (0, or at least 2: a
        single block needs no grouping)."""
        members: list[KVCacheBlock] = []
        seen: set[int] = set()
        for block in blocks:
            if (
                not block.is_null
                and block.ref_cnt == 0
                and block.block_hash is not None
                and (block.pool is None or block.pool is self._pool)
                and block.block_id not in self._block_group
                and block.block_id not in seen
                and not self.junctions.is_junction(block)
            ):
                seen.add(block.block_id)
                members.append(block)
        if len(members) < 2:
            return 0
        gid = self._next_id
        self._next_id += 1
        self._groups[gid] = members
        for block in members:
            self._block_group[block.block_id] = gid
        return len(members)

    def leave(self, block: KVCacheBlock) -> None:
        """The block was un-hashed or hit again: no longer a cached member."""
        gid = self._block_group.pop(block.block_id, None)
        if gid is None:
            return
        members = self._groups.get(gid)
        if members is not None:
            members[:] = [b for b in members if b is not block]
            if not members:
                del self._groups[gid]

    def leave_many(self, blocks: Iterable[KVCacheBlock]) -> None:
        """``leave`` for a batch: each affected group is rebuilt once, so
        detaching a whole U-row unit costs O(U), not O(U^2)."""
        gone: dict[int, set[int]] = {}
        for block in blocks:
            gid = self._block_group.pop(block.block_id, None)
            if gid is not None:
                gone.setdefault(gid, set()).add(block.block_id)
        for gid, ids in gone.items():
            members = self._groups.get(gid)
            if members is None:
                continue
            members[:] = [b for b in members if b.block_id not in ids]
            if not members:
                del self._groups[gid]

    def clear(self) -> None:
        self._groups.clear()
        self._block_group.clear()
        self.junctions.clear()

    def pop_free_blocks(self, num_blocks: int) -> list[KVCacheBlock]:
        """``free_block_queue.popleft_n``, plus: a free junction checkpoint
        is reused before any other cached block, and a popped member of a
        yielded session breaks it, so the next pops take the rest of that
        session."""
        queue = self._pool.free_block_queue
        junctions = self.junctions
        census = getattr(self, "census", None)  # read once, not per block
        if not self._block_group and not junctions:
            blocks = queue.popleft_n(num_blocks)
            if census:
                for block in blocks:
                    census.on_reclaim(block)
            return blocks
        ret: list[KVCacheBlock] = []
        for _ in range(num_blocks):
            block = junctions.take_evictable() if junctions else None
            if block is None:
                block = queue.popleft()
            if census:
                census.on_reclaim(block)
            if junctions:
                junctions.forget(block)
            ret.append(block)
            gid = self._block_group.get(block.block_id)
            if gid is not None:
                self._break(gid, block)
        return ret

    def is_host_backed(self, blocks: Sequence[KVCacheBlock]) -> bool:
        return (
            bool(blocks)
            and self.host_tier is not None
            and self.host_tier.is_host_backed(blocks)
        )

    def _break(self, gid: int, popped: KVCacheBlock) -> None:
        """Reuse a session first, retaining its peers' hashes if host-backed.

        Referenced members (waiter/grace pins, active tables or DMA leases)
        never move or lose a ref here. Hits leave the group via ``touch``.
        """
        pool = self._pool
        members = self._groups.pop(gid, ())
        # Notify before allocation unkeys the popped member.
        demoted = self.is_host_backed(members)
        if self.host_tier is not None:
            if demoted:
                self.host_tier.demote(members)
            else:
                self.host_tier.unit_lost(members)
        rest: list[KVCacheBlock] = []
        for block in members:
            if self._block_group.get(block.block_id) == gid:
                del self._block_group[block.block_id]
            if (
                block is popped
                or block.ref_cnt != 0
                or block.prev_free_block is None
                or block.next_free_block is None
            ):
                continue
            # a backed peer keeps its hashes: no loss.
            if not demoted and (census := getattr(self, "census", None)):
                census.on_reclaim(block, collateral=True)
            pool.free_block_queue.remove(block)
            if not demoted:
                pool._maybe_evict_cached_block(block)
            rest.append(block)
        pool.free_block_queue.prepend_n(rest)
        emit("break", gid=gid, block=popped.block_id, rest=len(rest), demoted=demoted)


class SessionYield:
    """Per-manager thrash state: each waiter pin's hit length, anchors and
    prompt hashes (what a young miss must not evict), and the need-sized,
    session-coherent yield policy over graces and waiter pins."""

    def __init__(self, manager: KVCacheManager, coherent: bool = False) -> None:
        self.manager = manager
        # Set by the scheduler (VLLM_E2_SCHED_COHERENT_YIELD, hybrid only).
        self.coherent = coherent
        # Per waiter pin: the hit length, and the deepest hit block of each
        # group (its anchors: the grace of the turn it resumes holds them, so
        # the two yield as one session).
        self.hit_tokens: dict[str, int] = {}
        self.anchors: dict[str, list[KVCacheBlock]] = {}
        # Per waiter pin: prefix hashes, and (built on first use) the set of
        # those at or past the pinned hit, which the pin does not hold.
        self.prefix_hashes: dict[str, Sequence] = {}
        self._needed_sets: dict[str, frozenset] = {}
        # The largest single session the last ``yield_need_sized`` could have
        # released (blocks).
        self.last_best_single_yield = 0
        # Where the scheduler is yielding from (site, step, req, hit); only
        # read for the decision log.
        self.ctx: dict = {}

    # -- pin bookkeeping ---------------------------------------------------
    def record_pin(
        self,
        request_id: str,
        num_hit: int,
        computed,
        block_hashes: Sequence = (),
    ):
        self.hit_tokens[request_id] = num_hit
        self.prefix_hashes[request_id] = block_hashes
        self._needed_sets.pop(request_id, None)
        self.anchors[request_id] = [
            deepest
            for group in computed.blocks
            if (deepest := next((b for b in reversed(group) if not b.is_null), None))
            is not None
        ]

    def forget(self, request_id: str) -> None:
        self.hit_tokens.pop(request_id, None)
        self.anchors.pop(request_id, None)
        self.prefix_hashes.pop(request_id, None)
        self._needed_sets.pop(request_id, None)

    def trim_extras(self, request_id: str, kept: list[KVCacheBlock]) -> None:
        """The pin was cut down to ``kept``. If an anchor went with the
        extras, the anchors no longer describe what the pin holds (and a
        released anchor may be reused by another conversation): drop them, so
        the pin pairs with no grace. The waiter stays protected: its hit
        boundary drops to 0, so every cached block of its prompt that the pin
        no longer holds counts as needed."""
        kept_ids = {id(block) for block in kept}
        if any(id(a) not in kept_ids for a in self.anchors.get(request_id, ())):
            self.anchors.pop(request_id, None)
            if request_id in self.hit_tokens:
                self.hit_tokens[request_id] = 0
            self._needed_sets.pop(request_id, None)

    def follow_cow(self, source_block: KVCacheBlock, cow_block: KVCacheBlock) -> None:
        # a grace / waiter pin moved to
        # the CoW copy: the copy scheduler's pin census is stale.
        copyq = self._copy_scheduler()
        if copyq is not None:
            copyq.note_pins_changed()
        for anchors in self.anchors.values():
            for i, block in enumerate(anchors):
                if block is source_block:
                    anchors[i] = cow_block

    def protected_waiters(self, request: Request) -> list[str]:
        """Pinned waiters a young miss ``request`` must not cost blocks: every
        one but itself. Two pinned misses that would each evict the other's
        blocks both park until the older reaches the bound."""
        return [rid for rid in self.hit_tokens if rid != request.request_id]

    def _needed_hashes(self, waiters: list[str]) -> list[frozenset]:
        """Per waiter, its prompt's hashes from the pinned hit on: the cached
        blocks it could still grow its hit into. Below the hit the pin holds
        what it needs."""
        needed: list[frozenset] = []
        unit = self.manager.block_pool.hash_block_size
        for rid in waiters:
            hashes = self._needed_sets.get(rid)
            if hashes is None:
                start = self.hit_tokens.get(rid, 0) // unit
                hashes = frozenset(
                    itertools.islice(self.prefix_hashes.get(rid, ()), start, None)
                )
                self._needed_sets[rid] = hashes
            if hashes:
                needed.append(hashes)
        return needed

    def _block_is_needed(self, block: KVCacheBlock, needed: list[frozenset]) -> bool:
        if block.block_hash is None:
            return False
        pool = self.manager.block_pool
        keys = [
            block.block_hash,
            *pool.cached_block_hashes_by_block.get(block.block_id, ()),
        ]
        return any(get_block_hash(key) in hashes for key in keys for hashes in needed)

    def evicts_needed(
        self,
        num_blocks_to_allocate: int,
        hit: tuple[Sequence[KVCacheBlock], ...],
        waiters: list[str],
    ) -> bool:
        """Forecast a waiter's last-copy loss.
        The allocation count includes the request's evictable hit blocks.
        Adopt those hits first, then replay junction priority and session
        expansion on a shadow queue. Backed groups retain their peer hashes;
        other groups unhash peers. Only destroying an unbacked needed block
        parks the miss. This query changes no refs, hashes or queue order.
        """
        needed = self._needed_hashes(waiters)
        if not needed:
            return False
        pool = self.manager.block_pool
        groups = pool.evict_groups
        out: set[int] = set()  # ids no longer in the shadow queue
        hit_free = 0
        for group in hit:
            for block in group:
                if block.is_null or id(block) in out:
                    continue
                out.add(id(block))
                if block.ref_cnt == 0:
                    hit_free += 1
        pops = num_blocks_to_allocate - hit_free
        if pops <= 0:
            return False
        front: list[KVCacheBlock] = []
        # Shadow the hashes as well as queue order.
        unhashed: set[int] = set()
        junctions = list(groups.junctions.free_evictable()) if groups.junctions else []
        broken: set[int] = set()
        queue = pool.free_block_queue
        cur = queue.fake_free_list_head.next_free_block
        tail = queue.fake_free_list_tail
        while pops > 0:
            while cur is not None and cur is not tail and id(cur) in out:
                cur = cur.next_free_block
            junctions = [b for b in junctions if id(b) not in out]
            # Hashed front peers still yield to junctions.
            head = front[0] if front else cur
            if head is None or head is tail:
                break
            if head.block_hash is not None and id(head) not in unhashed and junctions:
                block = junctions.pop(0)
            elif front:
                block = front.pop(0)
            else:
                block = cur
            out.add(id(block))
            pops -= 1
            # Only loss of the last copy parks a miss.
            if self._block_is_needed(block, needed) and not groups.is_host_backed(
                (block,)
            ):
                return True
            gid = groups._block_group.get(block.block_id)
            if gid is None or gid in broken:
                continue
            broken.add(gid)
            # Hit blocks have already left the group.
            members = [
                b
                for b in groups._groups.get(gid, ())
                if b is block or id(b) not in out
            ]
            demoted = groups.is_host_backed(members)
            for peer in members:
                if id(peer) in out or peer.ref_cnt != 0 or peer.prev_free_block is None:
                    continue
                # Backed peers remain cached until popped.
                if not demoted:
                    if self._block_is_needed(
                        peer, needed
                    ) and not groups.is_host_backed((peer,)):
                        return True
                    unhashed.add(id(peer))
                out.add(id(peer))
                front.append(peer)
        return False

    # -- yielding ------------------------------------------------------------
    def release_for_yield(self, unit: list[Holder]) -> int:
        """Release a session's holders (graces and/or a waiter pin) for a
        yield. With coherent yields, the blocks that land in the free queue
        are registered as one eviction group. Returns the number of blocks
        the holders held."""
        manager = self.manager
        held: list[KVCacheBlock] = []
        released = 0
        for kind, request_id in unit:
            if kind == "grace":
                blocks = manager.grace_pinned_blocks(request_id)
                # ``release_grace_pin`` (reason "yield") tells the copy
                # scheduler before the blocks are freed (kv-host-tier).
                released += manager.release_grace_pin(request_id)
            else:
                blocks = manager.pinned_hit_blocks(request_id)
                released += manager.release_pinned_hit(request_id)
            held.extend(blocks)
            if EVENTS is not None:
                session = next((r for k, r in unit if k == "waiter"), unit[0][1])
                emit(
                    "yield",
                    kind=kind,
                    victim=request_id,
                    session=session,
                    waited=any(k == "waiter" for k, _ in unit),
                    held=len(blocks),
                    freed=sum(1 for b in blocks if b.ref_cnt == 0),
                    short=manager.last_alloc_shortfall,
                    **self.ctx,
                )
        if self.coherent:
            manager.block_pool.evict_groups.register(held)
        return released

    def _units(self, keep: str | None) -> list[list[Holder]]:
        """The sessions that could yield, in yield order: finished turns no
        waiter resumes (oldest first), then each waiter pin (most recently
        pinned first) together with the turn it resumes -- the grace that
        holds every one of its anchors and whose prompt its hit reaches to
        within two prefix units, the same test as ``_release_grace_for_hit``.
        That grace frees nothing on its own while the pin holds the same
        blocks; a grace that only shares a prefix with the pin is a different
        conversation and stays its own session."""
        manager = self.manager
        paired: set[str] = set()
        waiter_units: list[list[Holder]] = []
        for request_id in reversed(list(manager._pinned_hits)):
            if request_id == keep:
                continue
            owners: set[str] | None = None
            for anchor in self.anchors.get(request_id, ()):
                holders = {
                    grace_id
                    for grace_id, block_ids in manager._grace_block_ids.items()
                    if id(anchor) in block_ids
                }
                owners = holders if owners is None else owners & holders
            reach = self.hit_tokens.get(request_id, 0) + (
                2 * manager.block_pool.hash_block_size
            )
            unit: list[Holder] = [("waiter", request_id)]
            # Grace insertion order, oldest first - never set order.
            for grace_id in [g for g in manager._grace_block_ids if g in (owners or ())]:
                if grace_id not in paired and reach >= (
                    manager._grace_prompt_tokens.get(grace_id, 0)
                ):
                    paired.add(grace_id)
                    unit.insert(0, ("grace", grace_id))
            waiter_units.append(unit)
        grace_units: list[list[Holder]] = [
            [("grace", grace_id)]
            for grace_id in manager._grace_pins
            if grace_id not in paired
        ]
        return grace_units + waiter_units

    # -- host-aware yield (kv-host-tier) ------------------------------
    def _copy_scheduler(self):
        """The host tier's copy scheduler (``kv_host_tier_copyq.CopyQueue``)
        when the tier is on and bound to this manager; None otherwise, and
        every yield decision is then exactly the tier-less one."""
        groups = getattr(self.manager.block_pool, "evict_groups", None)
        host_tier = getattr(groups, "host_tier", None)
        copyq = getattr(host_tier, "copyq", None)
        if copyq is None or getattr(copyq, "manager", None) is None:
            return None
        return copyq

    _STATE_RANK = {"complete": 0, "in_flight": 1, "dirty": 2}

    def _unit_state(self, copyq, unit: list[Holder]) -> str:
        """The worst grace state of a yield unit (a waiter pin alone is a
        live request's hit, nothing the host tier copies: ``complete``)."""
        worst = "complete"
        for kind, request_id in unit:
            if kind != "grace":
                continue
            state = copyq.grace_state(request_id)
            if self._STATE_RANK[state] > self._STATE_RANK[worst]:
                worst = state
        return worst

    def grace_yield_order(self) -> list[str]:
        """Grace request ids in the order a running request's one-pin yield
        (``KVCacheManager.yield_pinned_hits``) and the grace overflow should
        release them: host-complete graces first (their release is
        lossless), oldest first within each class. Without the tier: the
        plain oldest-first order."""
        order = list(self.manager._grace_pins)
        copyq = self._copy_scheduler()
        if copyq is None:
            return order
        complete = [rid for rid in order if copyq.grace_state(rid) == "complete"]
        done = set(complete)
        return complete + [rid for rid in order if rid not in done]

    def _freeable(self, unit: list[Holder], hit_ids: set[int]) -> int:
        """Blocks that releasing ``unit`` alone would return to the free
        queue (held by nothing else; never a block of ``hit_ids``)."""
        manager = self.manager
        counts: dict[int, list] = {}
        for kind, request_id in unit:
            blocks = (
                manager._grace_pins.get(request_id, ())
                if kind == "grace"
                else manager._pinned_hits.get(request_id, ())
            )
            for block in blocks:
                entry = counts.setdefault(id(block), [block, 0])
                entry[1] += 1
        return sum(
            1
            for block_id, (block, n) in counts.items()
            if block.ref_cnt == n and block_id not in hit_ids
        )

    def yield_need_sized(
        self,
        shortfall: int,
        keep: str | None = None,
        keep_hit: tuple[Sequence[KVCacheBlock], ...] | None = None,
    ) -> int:
        """Need-sized yield for a waiting head short of ``shortfall`` blocks.

        Release the smallest session that covers the shortfall on its own
        (its grace and waiter pin together), keeping yield order for ties.
        If none does, sessions are taken in yield order, and a session with a
        waiting follow-up is taken only if it is needed for at least half of
        what it frees: it is never broken to cover a small remainder (a
        hybrid session loses its whole hit to the first block reused, so the
        rest of it would be wasted). Finished turns nobody waits on are taken
        as needed.
        Otherwise nothing is released and the head waits.

        Returns the number of blocks the released holders held (0: nothing
        released; ``last_best_single_yield`` says how close the best single
        session came)."""
        # No shortfall (a demotion
        # wait) releases nothing.
        if shortfall <= 0:
            self.last_best_single_yield = 0
            return 0
        hit_ids = (
            {id(block) for group in keep_hit for block in group} if keep_hit else set()
        )
        units = [(unit, self._freeable(unit, hit_ids)) for unit in self._units(keep)]
        self.last_best_single_yield = max((f for _, f in units), default=0)
        # host-aware yield: prefer a
        # host-complete session (its break demotes, nothing is lost); hold
        # back briefly for one whose grace copy is in flight; else yield as
        # before and let the copy scheduler count ``yield_incomplete``.
        copyq = self._copy_scheduler()
        if copyq is not None:
            return self._yield_host_aware(copyq, units, shortfall, hit_ids)
        best = min(
            ((unit, freeable) for unit, freeable in units if freeable >= shortfall),
            key=lambda candidate: candidate[1],
            default=None,
        )
        if best is not None:
            return self.release_for_yield(best[0])
        # No single session covers it: the head needs several sessions' worth.
        chosen: list[list[Holder]] = []
        covered = 0
        for unit, freeable in units:
            if freeable == 0:
                continue
            total = self._freeable(
                [holder for taken in chosen for holder in taken] + unit, hit_ids
            )
            if total >= shortfall:
                waited_on = any(kind == "waiter" for kind, _ in unit)
                if waited_on and 2 * (shortfall - covered) < total - covered:
                    continue  # it would be broken for a remainder: try the next
                return sum(self.release_for_yield(taken) for taken in chosen + [unit])
            chosen.append(unit)
            covered = total
        return 0

    def _yield_host_aware(
        self,
        copyq,
        units: list[tuple[list[Holder], int]],
        shortfall: int,
        hit_ids: set[int],
    ) -> int:
        """``yield_need_sized`` with the host tier's copy state.

        Single session: the smallest HOST-COMPLETE one that covers the
        shortfall; else the smallest covering one, unless one of its graces
        has a copy in flight and the bounded wait (``CopyQueue.may_wait_for``:
        YIELD_WAIT_STEPS / YIELD_WAIT_S) has not run out, in which case
        nothing is released this time and the head waits (the scheduler's
        own fallbacks still bound it). Several sessions: the same walk as the
        tier-less rule, over host-complete sessions first."""
        states = {id(unit): self._unit_state(copyq, unit) for unit, _ in units}
        covering = [(u, f) for u, f in units if f >= shortfall]
        complete = [(u, f) for u, f in covering if states[id(u)] == "complete"]
        best = min(complete or covering, key=lambda c: c[1], default=None)
        if best is not None:
            unit = best[0]
            if states[id(unit)] == "in_flight" and any(
                kind == "grace" and copyq.may_wait_for(rid) for kind, rid in unit
            ):
                return 0
            return self.release_for_yield(unit)
        ordered = sorted(
            units, key=lambda c: self._STATE_RANK[states[id(c[0])]]
        )  # stable: yield order within each class
        chosen: list[list[Holder]] = []
        covered = 0
        for unit, freeable in ordered:
            if freeable == 0:
                continue
            total = self._freeable(
                [holder for taken in chosen for holder in taken] + unit, hit_ids
            )
            if total >= shortfall:
                waited_on = any(kind == "waiter" for kind, _ in unit)
                if waited_on and 2 * (shortfall - covered) < total - covered:
                    continue  # it would be broken for a remainder: try the next
                return sum(self.release_for_yield(taken) for taken in chosen + [unit])
            chosen.append(unit)
            covered = total
        return 0


class Admission:
    """Thrash policy for pinned hybrid hits.

    Parks young cold misses that would cost protected waiters blocks and
    runs coherent yields at admission. The host tier keeps prefix pins;
    other connectors retain their existing exemption.
    """

    def __init__(self, scheduler: Scheduler) -> None:
        self.scheduler = scheduler
        manager = scheduler.kv_cache_manager
        specs = [g.kv_cache_spec for g in scheduler.kv_cache_config.kv_cache_groups]
        # A hybrid layout: attention AND Mamba groups. Pure-Mamba and
        # attention-only models keep the stock rules.
        hybrid = any(isinstance(s, MambaSpec) for s in specs) and any(
            isinstance(s, AttentionSpec) for s in specs
        )
        # The scheduler recognizes the exact tier config.
        on = manager.enable_caching and scheduler.prefix_pins and hybrid
        # FCFS only: parking reorders the skipped queue, which the priority
        # policy orders by its own key.
        self.miss_defer_s = (
            envs.VLLM_E2_SCHED_MISS_DEFER_S
            if on and scheduler.policy == SchedulingPolicy.FCFS
            else 0
        )
        manager.thrash.coherent = on and envs.VLLM_E2_SCHED_COHERENT_YIELD
        # arrival_time is wall-clock; tests substitute their own.
        self.clock: Callable[[], float] = time.time
        if self.miss_defer_s:
            logger.info(
                "Prefix cache: cost-aware admission on (a cold miss waits up to "
                "%d s, and only when it would cost protected waiters blocks)",
                self.miss_defer_s,
            )
        if manager.thrash.coherent:
            logger.info(
                "Prefix cache: need-sized, session-coherent yields on (one "
                "session per shortfall; a broken session is reused first)"
            )

    def site(self, site: str, request: Request, hit: int) -> None:
        """Where a yield comes from, for the decision log."""
        sched = self.scheduler
        sched.kv_cache_manager.thrash.ctx = dict(
            site=site, step=sched.current_step, req=request.request_id, hit=hit
        )

    def pin_arrivals(self) -> None:
        """This step's arrivals pin their hits before anyone is looked at: a
        cold miss ahead of them must see them as waiting hits."""
        if self.miss_defer_s:
            sched = self.scheduler
            sched._pin_waiting_hits(max_requests=sched.kv_cache_manager.MAX_PINNED_WAITERS)

    def is_young_miss(self, request: Request, num_computed_tokens: int) -> bool:
        """A fresh waiting request whose hit covers less than half its prompt,
        younger than the cost-aware admission bound."""
        return (
            self.miss_defer_s > 0
            and request.status == RequestStatus.WAITING
            and 2 * num_computed_tokens < request.num_tokens
            and self.clock() - request.arrival_time < self.miss_defer_s
        )

    def park(self, request: Request, why: str, hit: int, protected: int) -> None:
        emit(
            "park",
            req=request.request_id,
            why=why,
            age=round(self.clock() - request.arrival_time, 3),
            hit=hit,
            prompt=request.num_tokens,
            protected=protected,
        )

    def protected_waiters(self, request: Request) -> list[str]:
        return self.scheduler.kv_cache_manager.thrash.protected_waiters(request)

    def park_over_budget(self, request: Request, hit: int, protected: int) -> None:
        """This step's token budget, Mamba alignment or encoder budget cannot
        schedule the young miss: it is skipped so the scan reaches the
        protected waiters behind it. No eviction was forecast."""
        self.park(request, "budget", hit, protected)

    def eviction_check(
        self, request: Request, waiters: list[str]
    ) -> Callable[..., bool]:
        """The ``allocate_slots`` hook for a young miss behind ``waiters``:
        True refuses the admission (it would un-hash a block they need)."""
        thrash = self.scheduler.kv_cache_manager.thrash

        def check(num_blocks_to_allocate: int, hit) -> bool:
            return thrash.evicts_needed(num_blocks_to_allocate, hit, waiters)

        return check

    def head_yield(
        self,
        request: Request,
        hit: int,
        keep_hit,
        num_scheduled_tokens,
        retry: Callable[[], object],
    ):
        """Need-sized yield for an admission head that did not fit. Returns
        ``(new_blocks, yield_for_head)``: the retried allocation (None if
        nothing was released or it still does not fit) and whether the
        one-pin-at-a-time fallback may go on.

        First try a need-sized yield: one session that covers the shortfall,
        or several only while each covers at least half of it; retry if freed.
        If allocation still fails, wait only while a running request can
        advance and the head is not aged. Otherwise allow the one-pin loop.
        Progress includes scheduled work, an in-flight step or PP cadence."""
        sched = self.scheduler
        manager = sched.kv_cache_manager
        # An aged head (past the same bound as the miss deferral; 20 s when
        # deferral is off) no longer waits for natural frees: it falls to the
        # one-pin-at-a-time fallback.
        bound = self.miss_defer_s or DEFAULT_HEAD_BOUND_S
        aged = self.clock() - request.arrival_time >= bound
        running_can_progress = not aged and (
            bool(num_scheduled_tokens)
            or any(
                r.num_in_flight_tokens > 0
                or sched.current_step < r.next_decode_eligible_step
                for r in sched.running
            )
        )
        new_blocks = None
        yield_for_head = True
        if manager.thrash.yield_need_sized(
            manager.last_alloc_shortfall, keep=request.request_id, keep_hit=keep_hit
        ):
            new_blocks = retry()
            yield_for_head = new_blocks is None and not running_can_progress
        elif running_can_progress:
            emit(
                "wait",
                req=request.request_id,
                hit=hit,
                short=manager.last_alloc_shortfall,
                best=manager.thrash.last_best_single_yield,
            )
            yield_for_head = False
        if yield_for_head:
            self.site("admit-idle", request, hit)
        return new_blocks, yield_for_head


__all__ = ["Admission", "EvictGroups", "HostTierEviction", "SessionYield"]
