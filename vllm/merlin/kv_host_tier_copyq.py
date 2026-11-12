# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Host-RAM KV tier: the copy scheduler.

The walker (``kv_host_tier_writeback.WriteBack._select``) asks this queue for
work after its reactive window, within the demotion direction's row and byte
credits, and issues every accepted offer through the unchanged demotion path:
``begin_demote`` (the transfer-owned source ref, held until every rank is
quiescent) -> one store txn -> all-rank ack -> ``demote_still_valid`` ->
publish. Nothing here copies, allocates host rows or unkeys anything; it only
decides WHAT to offer and in which order.

Lanes, highest priority first (``CopyLane``; the walker's own REACTIVE
victims come before all of them):

* PREEMPT: a preempted request's restore unit under a bounded "preempt
  grace" (one extra ref on its hashed blocks, taken only when the free pool
  covers the running headroom plus the waiting head's need without them, and
  at most one restore unit's worth in total; released at issue or after
  ``PREEMPT_GRACE_STEPS``).
* GRACE: every missing dependency of a grace-held finished turn, in
  SessionYield order (the turn that yields next first). Grace-held blocks are
  referenced and immutable, so the copy needs no free-block pin beyond the
  transfer ref; ``YOW_LOOKAHEAD`` turns at the head of the yield order may
  displace published host content (they are about to leave the GPU), the
  rest take idle rows or disposable duplicates only.
* REPAIR: a block whose host twin was evicted as a duplicate
  (``revoke_backing``) and that is now free (ref 0) and dirty: offered, never
  displacing; a referenced one is covered by GRACE/PREEMPT/RUNNING later.
* RUNNING (lane 1): full, hashed, all-rank-ready full-attention blocks of
  running requests, slack-only (idle rows or disposable duplicates; never a
  published unit), at most ``RUNNING_ROWS_PER_STEP`` rows a step.

State is recomputed from the engine each step (graces, units, block states),
so a missed hook costs one step, never a stale copy: the only state kept is
the preempt graces (refs), the repair set, offers in flight and counters.
"""

from __future__ import annotations

import itertools
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from vllm.merlin.kv_host_tier_iface import (
    DISPLACING_LANES,
    BoundaryManifest,
    CopyLane,
    CopyOffer,
    GraceRelease,
)

if TYPE_CHECKING:
    from vllm.merlin.kv_host_tier_units import RestoreUnit
    from vllm.merlin.kv_host_tier_writeback import WriteBack
    from vllm.v1.core.kv_cache_utils import KVCacheBlock
    from vllm.v1.request import Request

# Turns at the head of the yield order whose copy may displace published
# host content (SGLang's host > device rule, applied per step: only content
# about to leave the GPU outranks what the host already holds).
YOW_LOOKAHEAD = 2
# Steps a preempt grace holds the preempted request's hashed blocks waiting
# for its copy to issue.
PREEMPT_GRACE_STEPS = 4
# Lane-1 rows offered per step at most (slack-only write-through).
RUNNING_ROWS_PER_STEP = 32
# Host-aware yield: a session whose grace copy is in flight is waited for at
# most this many walker steps and this many seconds before it is yielded
# anyway (counted ``yield_incomplete``).
YIELD_WAIT_STEPS = 2
YIELD_WAIT_S = 2.0


@dataclass
class _PreemptGrace:
    blocks: list["KVCacheBlock"]
    expires: int
    uid: int | None


class CopyQueue:
    """``CopyScheduler`` + ``LifecycleHooks`` for one ``WriteBack``."""

    def __init__(self, wb: "WriteBack") -> None:
        self.wb = wb
        # The scheduler's KVCacheManager (grace pins, SessionYield, the
        # coordinator for preempted blocks); None until ``bind_manager``.
        self.manager: Any = None
        self._ids = itertools.count(1)
        # Request id -> its recorded unit uid (``finished`` / ``preempted``
        # hooks); a hint only, re-validated against the unit table.
        self._uid_of: dict[str, int] = {}
        self._preempt: dict[str, _PreemptGrace] = {}
        # GPU block id -> generation when its host backing was revoked.
        self._repair: dict[int, int] = {}
        # Offers issued and not yet acked, by offer id.
        self._issued: dict[int, CopyOffer] = {}
        # Host-aware yield: first step / time a session was held back.
        self._yield_wait: dict[str, tuple[int, float]] = {}
        self._fa_groups: tuple[int, ...] | None = None
        self._running_cursor = 0
        # Bumped by every grace / waiter / preempt pin change this queue
        # hears of (take, release, CoW move): the ``twin_kind`` cache key.
        self._pin_version = 0
        self._pins_cache: tuple[tuple, dict[int, int]] | None = None
        # Request id -> id() of the grace pin list already reported released
        # (``grace_released`` idempotence).
        self._released: dict[str, int] = {}

    # -- binding ------------------------------------------------------------
    def bind_manager(self, kv_cache_manager: Any) -> None:
        self.manager = kv_cache_manager
        self._pin_version += 1  # another manager's pins

    @property
    def _units(self):
        return self.wb.units

    # -- LifecycleHooks -----------------------------------------------------
    def grace_taken(
        self, request: "Request", blocks: Sequence["KVCacheBlock"], step: int
    ) -> None:
        # Polled each step from ``_grace_pins``; only the pin version moves.
        self._pin_version += 1

    def grace_released(
        self, request_id: str, reason: GraceRelease, step: int
    ) -> None:
        """A grace is about to be released (its blocks are still held). A
        yield (``SessionYield.release_for_yield``) or an overflow of a turn
        whose unit is not host-complete is counted ``yield_incomplete``; a
        hit (the turn resumed on the GPU) or an abort loses nothing.

        Idempotent per grace: a repeat for the same pinned grace (the same
        pin list object, still in ``_grace_pins`` until the release pops it)
        is ignored, so a caller that reports a release and then calls
        ``release_grace_pin`` (which reports it too) counts it once."""
        # Before the repeat check: the pin census moves either way.
        self._pin_version += 1
        m = self.manager
        pins = getattr(m, "_grace_pins", None) if m is not None else None
        ident = id(pins.get(request_id)) if pins is not None else None
        if ident is not None and self._released.get(request_id) == ident:
            return
        if ident is not None:
            self._released[request_id] = ident
        if len(self._released) > 4 * 16 + 64:
            live = pins or {}
            self._released = {
                r: i for r, i in self._released.items()
                if id(live.get(r)) == i
            }
        if reason in ("yield", "overflow"):
            self.note_grace_yield(request_id)
        self._yield_wait.pop(request_id, None)
        # The hint lives as long as the grace (or preempt grace) it serves:
        # a yield or an overflow ends it too.
        if request_id not in self._preempt:
            self._uid_of.pop(request_id, None)

    def finished(
        self, request: "Request", manifest: BoundaryManifest | None, step: int
    ) -> None:
        if manifest is not None and manifest.uid is not None:
            self._uid_of[request.request_id] = manifest.uid
            self._prune_uid_of(request.request_id)

    def _prune_uid_of(self, keep: str) -> None:
        """Bound the uid hints: past the threshold keep only
        ``keep`` (its grace may not be pinned yet) and requests that still
        hold a grace or a preempt grace (a finished turn may get no grace pin
        at all, so its hint is never released)."""
        if len(self._uid_of) <= 4 * 16 + 64:
            return
        m = self.manager
        pins = getattr(m, "_grace_pins", None) if m is not None else None
        live = set(pins or ()) | set(self._preempt) | {keep}
        self._uid_of = {r: u for r, u in self._uid_of.items() if r in live}

    def preempted(
        self, request: "Request", manifest: BoundaryManifest | None, step: int
    ) -> None:
        """Take a bounded preempt grace on the preempted request's hashed
        blocks, before the scheduler frees them, only when the free pool
        covers the next step's running need and the waiting head's need
        without them, and at most one restore unit's worth across all
        preempt graces (no unbudgeted long-lived pins)."""
        wb = self.wb
        rid = request.request_id
        if manifest is not None and manifest.uid is not None:
            self._uid_of[rid] = manifest.uid
            # Bounded as ``finished`` bounds it: a preemption whose grace is
            # refused (or never asked) leaves a hint no grace will release.
            self._prune_uid_of(rid)
        if self.manager is None or wb.gpu_pool is None or rid in self._preempt:
            return
        seen: set[int] = set()
        blocks: list[KVCacheBlock] = []
        for group in self.manager.coordinator.get_blocks(rid):
            for b in group:
                if b.is_null or b.block_hash is None or b.block_id in seen:
                    continue
                seen.add(b.block_id)
                blocks.append(b)
        if not blocks:
            return
        c = wb.counters
        held = sum(len(g.blocks) for g in self._preempt.values())
        floor = self._floor()
        if (
            held + len(blocks) > wb._hold_cap()
            or wb.gpu_pool.get_num_free_blocks() < floor
        ):
            c.preempt_grace_refused += 1
            return
        wb.gpu_pool.touch(blocks)
        self._pin_version += 1
        self._preempt[rid] = _PreemptGrace(
            blocks, wb.step + PREEMPT_GRACE_STEPS,
            manifest.uid if manifest is not None else self._uid_of.get(rid),
        )
        c.preempt_grace_taken += 1

    def holds_preempt_grace(self) -> bool:
        """Whether a preempt grace holds blocks it will give back: at its
        deadline (``PREEMPT_GRACE_STEPS``), when its unit completes, or at a
        sweep that finds the free pool below the floor (the waiting head's
        need). Pin yields never release one."""
        return bool(self._preempt)

    def running_ready(
        self, request: "Request", blocks: Sequence["KVCacheBlock"], step: int
    ) -> None:
        # Lane 1 polls the running requests the walker saw in
        # ``account_running``; the hook is accepted for the contract.
        return None

    def _release_preempt(self, rid: str) -> None:
        grace = self._preempt.pop(rid, None)
        if grace is not None:
            self._uid_of.pop(rid, None)  # the hint ends with it
        if grace is not None and self.wb.gpu_pool is not None:
            self._pin_version += 1
            self.wb.gpu_pool.free_blocks(grace.blocks)

    def _floor(self) -> int:
        """Free GPU blocks a preempt grace must leave: the running headroom
        (this step's when ``account_running`` ran, else the last walk's)
        plus the waiting head's need, the newer of the admission since the
        last walk and the one the last walk consumed (as the gate mirror)."""
        wb = self.wb
        headroom = wb.headroom_now if wb.headroom_now is not None else wb._last_headroom
        return headroom + max(wb._head_need, wb._last_head_need)

    def note_pins_changed(self) -> None:
        """A grace or waiter pin moved to another block (CoW), or a waiter
        pin was taken, trimmed or dropped: invalidate the ``twin_kind``
        cache."""
        self._pin_version += 1

    def sweep(self, step: int) -> None:
        """Every walker step, whatever the credits: expire
        preempt graces past their deadline, release those whose unit is gone
        or host-complete, and shed the newest while the free pool is below
        the floor (a preempt grace never holds what running requests or the
        waiting head need)."""
        wb = self.wb
        if not self._preempt or wb.gpu_pool is None:
            return
        for rid in list(self._preempt):
            grace = self._preempt[rid]
            if wb.step >= grace.expires:
                self._release_preempt(rid)
                wb.counters.preempt_grace_expired += 1
                continue
            unit = self._live_unit(grace.uid) or self._unit_for(rid, grace.blocks)
            if unit is None or self._unit_state(unit) == "complete":
                self._release_preempt(rid)
        floor = self._floor()
        for rid in reversed(list(self._preempt)):
            if wb.gpu_pool.get_num_free_blocks() >= floor:
                break
            self._release_preempt(rid)
            wb.counters.preempt_grace_shed += 1  # not an expiry

    # -- unit lookup ----------------------------------------------------------
    def _live_unit(self, uid: int | None) -> "RestoreUnit | None":
        units = self._units
        if uid is None or units is None:
            return None
        unit = units.units.get(uid)
        is_sup = getattr(units, "is_superseded", None)
        if unit is None or (is_sup is not None and is_sup(uid)):
            return None
        return unit

    def _unit_for(
        self, rid: str, blocks: Sequence["KVCacheBlock"]
    ) -> "RestoreUnit | None":
        """The request's own restore unit: the hook's uid when still live,
        else the unit owning most of ``blocks`` (deepest boundary on a tie).
        A unit that merely shares a prefix (another agent's system prompt)
        owns only those few blocks, never most of them."""
        unit = self._live_unit(self._uid_of.get(rid))
        if unit is not None:
            return unit
        units = self._units
        if units is None or not blocks:
            return None
        votes: dict[int, int] = {}
        found: dict[int, RestoreUnit] = {}
        for b in blocks:
            for u in units.unit_of_block(b):
                votes[u.uid] = votes.get(u.uid, 0) + 1
                found[u.uid] = u
        if not votes:
            return None
        uid = max(votes, key=lambda u: (votes[u], found[u].boundary))
        self._uid_of[rid] = uid
        return found[uid]

    def _grace_order(self) -> list[str]:
        """Grace request ids in SessionYield order (yields next first)."""
        m = self.manager
        if m is None:
            return []
        thrash = getattr(m, "thrash", None)
        try:
            units = thrash._units(None) if thrash is not None else None
        except (AttributeError, KeyError):
            units = None
        if units is None:
            return list(getattr(m, "_grace_pins", ()))
        out: list[str] = []
        for unit in units:
            out.extend(rid for kind, rid in unit if kind == "grace")
        return out

    def _grace_unit(self, rid: str) -> "RestoreUnit | None":
        m = self.manager
        blocks = m._grace_pins.get(rid, ()) if m is not None else ()
        return self._unit_for(rid, blocks)

    # -- host-aware yield ---------------------------------------------------
    def _unit_state(self, unit: "RestoreUnit | None") -> str:
        """``complete`` (nothing dirty, nothing in flight), ``in_flight``
        (a copy of it is in flight) or ``dirty``."""
        if unit is None or self._units is None:
            return "complete"
        tier = self.wb.tier
        in_flight = self.wb.in_flight
        dirty = False
        for m in self._units.gpu_members(unit):
            if m.block_id in in_flight:
                return "in_flight"
            if tier.is_dirty(m):
                dirty = True
        return "dirty" if dirty else "complete"

    def host_complete(self, request_id: str) -> bool:
        if self.manager is None or self._units is None:
            return True
        return self._unit_state(self._grace_unit(request_id)) == "complete"

    def grace_state(self, request_id: str) -> str:
        if self.manager is None or self._units is None:
            return "complete"
        return self._unit_state(self._grace_unit(request_id))

    def may_wait_for(self, request_id: str) -> bool:
        """Host-aware yield: True while a session whose copy is in flight may
        still be held back (bounded by YIELD_WAIT_STEPS and YIELD_WAIT_S)."""
        if self.grace_state(request_id) != "in_flight":
            self._yield_wait.pop(request_id, None)
            return False
        now = time.monotonic()
        first = self._yield_wait.setdefault(request_id, (self.wb.step, now))
        if (
            self.wb.step - first[0] >= YIELD_WAIT_STEPS
            or now - first[1] >= YIELD_WAIT_S
        ):
            return False
        self.wb.counters.yield_waited_steps += 1
        return True

    def pick_overflow_victim(self, candidates: list[str]) -> str | None:
        """Grace overflow (``KVCacheManager._take_grace_pin``): the oldest
        host-complete grace among ``candidates`` (oldest first); None = the
        oldest (no host-complete grace, or the tier is unbound)."""
        if self.manager is None or self._units is None:
            return None
        for rid in candidates:
            if self.grace_state(rid) == "complete":
                return rid
        return None

    # -- duplicate classes (``UnitTable.twin_kind``) -----------------------
    def _pin_counts(self) -> dict[int, int]:
        """id(block) -> refs held by grace pins, waiter pins and preempt
        graces (not by a running request's block table). Cached on the
        walker step and the pin version, which every pin change moves: grace
        take and release, preempt grace change, CoW pin move, manager
        binding, and every waiter pin take, trim and drop (``SessionYield``
        -> ``note_pins_changed``). Exact: an answer never depends on when the
        cache was filled."""
        m = self.manager
        key = (self.wb.step, self._pin_version)
        cached = self._pins_cache
        if cached is not None and cached[0] == key:
            return cached[1]
        counts: dict[int, int] = {}
        sources: list = []
        if m is not None:
            sources.extend(getattr(m, "_grace_pins", {}).values())
            sources.extend(getattr(m, "_pinned_hits", {}).values())
        sources.extend(g.blocks for g in self._preempt.values())
        for blocks in sources:
            for b in blocks:
                counts[id(b)] = counts.get(id(b), 0) + 1
        self._pins_cache = (key, counts)
        return counts

    def twin_kind(self, block: "KVCacheBlock") -> str:
        """"running" when a running request's block table holds ``block``
        (refs beyond grace, waiter, preempt-grace and tier pins); else
        "imminent" (grace-only, waiter-pinned or free: it leaves the GPU
        next). A grace-only twin is imminent (evicted last)."""
        wb = self.wb
        tier_refs = int(block.block_id in wb._held_gpu)
        if wb.gpu_pool is not None and wb.tier.is_demoting(block):
            tier_refs += 1
        other = block.ref_cnt - tier_refs - self._pin_counts().get(id(block), 0)
        return "running" if other > 0 else "imminent"

    def note_grace_yield(self, request_id: str) -> None:
        """A grace is being released by a yield or an overflow: count it
        ``yield_incomplete`` when its unit is not host-complete (a GPU
        break of it destroys content the host does not have)."""
        if self.manager is None or self._units is None:
            return
        if self.grace_state(request_id) != "complete":
            self.wb.counters.yield_incomplete += 1
        self._yield_wait.pop(request_id, None)

    # -- CopyScheduler --------------------------------------------------------
    def may_displace(self, offer: CopyOffer) -> bool:
        if offer.lane is CopyLane.GRACE:
            return offer.yield_rank is not None and offer.yield_rank < YOW_LOOKAHEAD
        return offer.lane in DISPLACING_LANES

    def _offer(
        self, lane: CopyLane, step: int, members: Sequence["KVCacheBlock"],
        busy: set[int], *, request_id: str | None = None,
        uid: int | None = None, yield_rank: int | None = None,
        txn_atomic: bool = True,
    ) -> CopyOffer | None:
        """An offer of ``members``' dirty, ready blocks; None when nothing is
        dirty, or (atomic) when a member is in flight or not ready yet."""
        tier = self.wb.tier
        blocks: list[KVCacheBlock] = []
        seen: set[int] = set()
        for m in members:
            bid = m.block_id
            if bid in seen:
                continue
            seen.add(bid)
            if bid in busy:
                if txn_atomic and tier.is_demoting(m):
                    return None  # finish the copy in flight first
                continue
            if not tier.is_dirty(m):
                continue
            if not tier.is_ready(m):
                if txn_atomic:
                    return None
                continue
            blocks.append(m)
        if not blocks:
            return None
        gen = tier.generation
        rb = max(0, int(getattr(self.wb, "row_bytes", 0) or 0))
        return CopyOffer(
            offer_id=next(self._ids),
            lane=lane,
            step=step,
            blocks=tuple(blocks),
            generations=tuple(gen[b.block_id] for b in blocks),
            keys=tuple(b.block_hash for b in blocks),
            request_id=request_id,
            uid=uid,
            yield_rank=yield_rank,
            bytes_per_rank=len(blocks) * rb,
            txn_atomic=txn_atomic,
        )

    def _tiny(self, unit: "RestoreUnit") -> bool:
        from vllm.merlin.kv_host_tier_writeback import TINY_FA_PAGES

        return self._units.fa_pages(unit.uid) < TINY_FA_PAGES

    def offers(
        self, step: int, *, rows_budget: int, bytes_budget: int,
        busy: set[int] | None = None,
    ) -> list[CopyOffer]:
        """Offers to issue this step, highest priority first. The walker
        re-checks every member and the budget when it accepts them; offers
        past the budget are dropped here except the first one when nothing
        is in flight (a unit larger than the credits must still progress)."""
        wb = self.wb
        units = self._units
        if units is None or wb.gpu_pool is None:
            return []
        busy = set(wb.in_flight) | (busy or set())
        out: list[CopyOffer] = []
        rows_left = rows_budget
        bytes_left = bytes_budget
        lone_ok = not wb.in_flight and not busy

        def add(offer: CopyOffer | None) -> bool:
            nonlocal rows_left, bytes_left, lone_ok
            if offer is None:
                return True
            fits = offer.rows <= rows_left and (
                bytes_budget <= 0 or offer.bytes_per_rank <= bytes_left
            )
            if not fits and not (lone_ok and not out):
                return rows_left > 0
            out.append(offer)
            if offer.lane is CopyLane.GRACE:
                # Offers this walk accepted, not built.
                wb.counters.grace_offers += 1
            busy.update(b.block_id for b in offer.blocks)
            rows_left -= offer.rows
            bytes_left -= offer.bytes_per_rank
            lone_ok = False
            return rows_left > 0

        # PREEMPT: bounded graces, oldest first (expiry and release run in
        # ``sweep``, every walker step).
        self.sweep(step)
        for rid in list(self._preempt):
            grace = self._preempt[rid]
            unit = self._live_unit(grace.uid) or self._unit_for(rid, grace.blocks)
            if unit is None:
                continue
            if not add(self._offer(
                CopyLane.PREEMPT, step, units.gpu_members(unit), busy,
                request_id=rid, uid=unit.uid,
            )):
                return out

        # GRACE: SessionYield order.
        for rank, rid in enumerate(self._grace_order()):
            unit = self._grace_unit(rid)
            if unit is None or self._tiny(unit):
                continue
            offer = self._offer(
                CopyLane.GRACE, step, units.gpu_members(unit), busy,
                request_id=rid, uid=unit.uid, yield_rank=rank,
            )
            if not add(offer):
                return out

        # REPAIR: revoked backing now free and dirty.
        tier = wb.tier
        blocks = wb.gpu_pool.blocks
        for bid, gen in list(self._repair.items()):
            b = blocks[bid]
            if tier.generation[bid] != gen:
                del self._repair[bid]
                continue
            if not tier.is_dirty(b):
                if not tier.is_demoting(b):
                    del self._repair[bid]
                    wb.counters.repair_done += 1
                continue
            if b.ref_cnt != 0:
                continue  # referenced: GRACE / PREEMPT / RUNNING cover it
            if not add(self._offer(CopyLane.REPAIR, step, [b], busy, txn_atomic=False)):
                return out
        wb.counters.repair_pending = len(self._repair)

        # RUNNING (lane 1): slack-only, round-robin over running requests.
        running = getattr(wb, "_running_reqs", None) or ()
        coord = getattr(wb, "_coordinator", None)
        if running and coord is not None:
            fa = self._fa_group_ids(coord)
            # Lane 1's own sub-budget (``WriteBack._lane1_budget``: 0 unless
            # the tier is quiet), never the rows left to the other lanes.
            cap = min(RUNNING_ROWS_PER_STEP, max(0, getattr(wb, "_lane1_allow", 0)))
            n = len(running)
            start = self._running_cursor % n
            for i in range(n):
                if cap <= 0:
                    break
                req = running[(start + i) % n]
                groups = coord.get_blocks(req.request_id)
                # Write on demoting: only FA/MLA pages
                # that are COMPLETE and strictly below the request's write
                # position, i.e. page i with (i + 1) * page_size <=
                # num_computed_tokens. Those positions are never written
                # again (this step's tokens land at num_computed_tokens and
                # beyond). Never the partial / last page, even when a
                # partial-block hash makes it look cached and ready, and
                # never a recurrent or conv state (not in ``fa``).
                # Unit-less blocks only: a member of a recorded unit is
                # copied by GRACE / PREEMPT / reactive as a whole unit; a
                # lane-1 copy of it would stage a partial unit (assembling,
                # a held host row, an assembly slot).
                computed = int(getattr(req, "num_computed_tokens", 0) or 0)
                members = []
                for g in fa:
                    if g >= len(groups):
                        continue
                    full = computed // self._page_size(coord, g)
                    members.extend(
                        b for b in groups[g][:full]
                        if not b.is_null and b.block_hash is not None
                        and not units.unit_of_block(b)
                    )
                offer = self._offer(
                    CopyLane.RUNNING, step, members, busy,
                    request_id=req.request_id, txn_atomic=False,
                )
                if offer is None:
                    continue
                if offer.rows > cap:
                    offer = self._offer(
                        CopyLane.RUNNING, step, list(offer.blocks[:cap]), busy,
                        request_id=req.request_id, txn_atomic=False,
                    )
                    if offer is None:
                        continue
                # Lane 1 is not charged against the other lanes' rows.
                cap -= offer.rows
                out.append(offer)
                busy.update(b.block_id for b in offer.blocks)
            self._running_cursor = start + 1
        return out

    @staticmethod
    def _page_size(coord: Any, gid: int) -> int:
        """Tokens per block of group ``gid`` (the manager's block size)."""
        m = coord.single_type_managers[gid]
        size = getattr(m, "block_size", None) or m.kv_cache_spec.block_size
        return max(1, int(size))

    def _fa_group_ids(self, coord: Any) -> tuple[int, ...]:
        if self._fa_groups is None:
            from vllm.v1.kv_cache_interface import (
                FullAttentionSpec,
                MLAAttentionSpec,
            )

            # Exact full attention only (a sliding or sink variant's pages
            # are not restore dependencies the way FA pages are).
            exact = (FullAttentionSpec, MLAAttentionSpec)
            self._fa_groups = tuple(
                i for i, m in enumerate(coord.single_type_managers)
                if type(m.kv_cache_spec) in exact and i in self.wb.group_ids
            )
        return self._fa_groups

    def note_issued_rows(self, offer: CopyOffer, rows: int) -> None:
        """Lane row counters: ``rows`` of ``offer`` issued in one
        transaction (the walker calls this once per transaction, so a
        chunked offer counts each issued row once, never ``offer.rows`` per
        chunk)."""
        c = self.wb.counters
        lane = offer.lane
        if lane is CopyLane.GRACE:
            c.copy_grace_rows += rows
            c.lane2_rows += rows
        elif lane is CopyLane.PREEMPT:
            c.copy_preempt_rows += rows
            c.lane2_rows += rows
        elif lane is CopyLane.REPAIR:
            c.copy_repair_rows += rows
        elif lane is CopyLane.RUNNING:
            c.copy_running_rows += rows

    def on_issued(self, offer: CopyOffer, txn_id: int) -> None:
        """A transaction carrying (some of) ``offer``'s rows was issued.
        Counting is per issued row (``note_issued_rows``), not here."""
        self._issued[offer.offer_id] = offer
        if offer.lane is CopyLane.PREEMPT and offer.request_id is not None:
            # The transfer refs (DEMOTING) hold the copied sources now; the
            # members still queued are held by the walker (``_hold``).
            self._release_preempt(offer.request_id)

    def on_acked(self, offer: CopyOffer, ok: Sequence[bool]) -> None:
        """The offer's transaction is over (every rank quiescent). A member
        not published (stale source: CoW move, reuse, epoch) stays dirty and
        is offered again while still valid."""
        self._issued.pop(offer.offer_id, None)
        if offer.lane in (CopyLane.GRACE, CopyLane.PREEMPT) and not all(ok):
            self.wb.counters.lane2_late += 1
        if offer.lane is CopyLane.REPAIR:
            for b, good in zip(offer.blocks, ok):
                if good and self._repair.pop(b.block_id, None) is not None:
                    self.wb.counters.repair_done += 1

    def revoke_backing(self, blocks: Sequence["KVCacheBlock"], step: int) -> None:
        tier = self.wb.tier
        for b in blocks:
            if b.is_null:
                continue
            self._repair[b.block_id] = tier.generation[b.block_id]
        self.wb.counters.backing_revoked += len(blocks)
        self.wb.counters.repair_pending = len(self._repair)

    # -- assembly ledger ------------------------------------------------------
    def assembly_rows(self) -> int:
        """Unique host rows held for units not yet published: landed and
        held (assembling), reserved by the reactive walker, and in flight."""
        wb = self.wb
        rows: set[int] = set()
        units = self._units
        if units is not None:
            held = getattr(units, "_held", {})
            for blocks in held.values():
                rows.update(b.block_id for b in blocks)
        for reserved, _ in wb._res.values():
            rows.update(b.block_id for b in reserved)
        return len(rows) + len(wb.in_flight)

    # -- lifecycle --------------------------------------------------------------
    def forget_request(self, rid: str) -> None:
        self._yield_wait.pop(rid, None)

    def reset(self) -> None:
        for rid in list(self._preempt):
            self._release_preempt(rid)
        self._repair.clear()
        self._issued.clear()
        self._yield_wait.clear()
        self._uid_of.clear()
        self._pin_version += 1
        self._pins_cache = None
        self._released.clear()


__all__ = [
    "CopyQueue",
    "PREEMPT_GRACE_STEPS",
    "RUNNING_ROWS_PER_STEP",
    "YIELD_WAIT_S",
    "YIELD_WAIT_STEPS",
    "YOW_LOOKAHEAD",
]
