# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Allocator-owned state of the GPU KV pool when the host-RAM tier is mounted.

The GPU pool and the host pool are two levels of one cache. This module is the
GPU side's half of the shared directory: for every GPU block it answers "can
the allocator reuse this block without destroying a restore point?" and "are
this block's bytes final on every rank?". ``BlockPool`` holds an instance in
``BlockPool.tier`` (None without the tier) and calls the ``on_*`` hooks; all
logic lives here.

Block state (``state``), derived on every query, never cached:

* ``DEMOTING``: a copy to the host is in flight. The source is touched
  (ref > 0), so it is not in the free queue and cannot be reclaimed.
* ``CLEAN``: reclaim is free. The block is unhashed, its group is not stored
  by the tier, its content was reported masked (never valid), the content is
  no restore dependency (``retains_fn``), or the host pool holds every key the
  block carries (primary and aliases).
* ``DIRTY``: hashed, stored by the tier, a restore dependency, and some key is
  not on the host. A reclaim destroys a restore point.

Restore dependency: ``retains_fn`` (installed by the unit table when a
sliding-window group exists, ``kv_host_tier_units.mount``) says whether a hashed
page can still be read by a restore. A sliding-window page is one only inside
the retained window of a boundary (a recorded unit, a superseded endpoint that
is still keyed, or a running request's prospective finish-time boundary); the
out-of-window pages a long prefill sheds are not, so they are neither copied by
the write-back walker nor protected by the admission gate. Without
``retains_fn`` (no sliding-window group: MTP lanes) every eligible page is a
dependency. A page the host holds under one of its keys in OTHER content (an
older generation) stays a dependency whatever ``retains_fn`` says
(``_host_holds_foreign``): the GPU generation must be copied over it.

Backing is revalidated against the host directory on every query, never kept as
a bit: a host unkey (LRU reuse, duplicate reclaim, a failed load's invalidation,
a reset) makes the GPU copy DIRTY. Equal keys do NOT mean equal content (a
recomputed generation of the same tokens has different bytes), so a host block
backs a GPU block only when it was copied from that very content: every stamped
GPU content has a content id (``ContentStamp.cid``), minted when it is
published, carried through a CoW hash move, and inherited by a restored
destination (``publish_restored``); a host copy records the id it read
(``note_host_copy``), and ``host_backs`` needs the host block under every key to
carry the GPU content's id. A GPU block with no stamp (e.g. a cancelled
restore's quiesced publication, whose bytes ARE the host copy's) falls back to
key equality and is counted ``backing_unproven``.

Readiness (``is_ready``) is a property of the content, not of the block id, and
is the lookup and publication predicate for both tiers (a demotion copies only
ready sources). Every hashed content gets a ``ContentStamp`` when it is
published (the scheduled step that writes it and the producing request), tied to
the block's ``generation``; any un-hash or reuse bumps the generation and
invalidates the stamp. A CoW hash move (``BlockPool.move_block_hashes``) carries
the stamp to the destination and delays readiness to the move's step. Content is
ready once the per-rank commit frontier (``set_acked_step``, the minimum over
ranks of the contiguous retired step) has passed its step and its
``(producer, produced_step)`` was not reported masked (``mark_masked``); until
the rank-ack feed sets both, nothing is ready.

Step ids: ``advance_step(s)`` names the step being scheduled next. The offload
manager calls it from ``build_connector_meta`` with its own ``_sched_step + 1``,
so a stamp's step is the ``_sched_step`` of the connector metadata that ships the
forward writing the content.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Iterable, Iterator, Sequence
from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING, NamedTuple, Protocol

from vllm.v1.core.kv_cache_utils import (
    BlockHashWithGroupId,
    KVCacheBlock,
    get_group_id,
)

if TYPE_CHECKING:
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.request import Request


class BlockState(IntEnum):
    CLEAN = 0
    DIRTY = 1
    DEMOTING = 2


class ContentStamp(NamedTuple):
    """Provenance of one hashed content of a block."""

    # BlockTierState.generation of the block when stamped.
    gen: int
    # Step whose forward finishes writing the bytes (ready once acked).
    ready_step: int
    # Producing request and step: the identity a masked commit is reported by.
    producer: str | None
    produced_step: int
    # Content id: names these bytes across block ids (CoW, host copies).
    cid: int = 0


class DemoteTicket(NamedTuple):
    """One source block of a demotion (``begin_demote``)."""

    block: KVCacheBlock
    gen: int
    primary: BlockHashWithGroupId
    # Content id the copy reads (ContentStamp.cid; 0 = unstamped).
    cid: int = 0


class TierBlockListener(Protocol):
    """Receives GPU directory changes (the host tier's unit table)."""

    def on_gpu_keyed(self, block: KVCacheBlock) -> None:
        """``block`` gained keys (publication or a CoW move)."""

    def on_gpu_unhashed(
        self, block: KVCacheBlock, keys: Sequence[BlockHashWithGroupId]
    ) -> None:
        """``keys`` were removed from ``block`` (already out of the map)."""

    def on_gpu_reclaim(self, block: KVCacheBlock, state: BlockState) -> None:
        """``block`` is popped for reuse; its keys are still present."""

    def on_gpu_reset(self) -> None:
        """``reset_prefix_cache`` dropped every GPU key."""


@dataclass
class TierBlockCounters:
    # Reclaims by state at the pop.
    reclaimed_dirty: int = 0  # demote_missed: a restore point destroyed
    reclaimed_backed: int = 0  # demote_redemoted: host twin present, no copy
    reclaimed_free: int = 0  # unhashed / not stored / masked / unretained
    # Of reclaimed_free: stored-group pages no restore reads (the
    # out-of-window sliding-window pages a long prefill sheds).
    reclaimed_unretained: int = 0
    demote_issued: int = 0
    demote_completed: int = 0
    demote_abandoned: int = 0
    front_requeued: int = 0
    masked_blocks: int = 0
    # Stamped GPU content compared by key equality only (no content id).
    backing_unproven: int = 0
    # Host copies that carry the keys but were read from other content.
    backing_foreign: int = 0
    # Older-generation host copies evicted by a newer copy of the same key.
    host_superseded: int = 0


class BlockTierState:
    def __init__(
        self,
        pool: BlockPool,
        eligible_group_ids: Collection[int] | None = None,
        host_pool: BlockPool | None = None,
    ) -> None:
        n = pool.num_gpu_blocks
        self.pool = pool
        self.host_pool = host_pool
        # None = every group is stored by the tier.
        self.eligible_group_ids = (
            None if eligible_group_ids is None else frozenset(eligible_group_ids)
        )
        self.generation: list[int] = [0] * n
        self._stamp: list[ContentStamp | None] = [None] * n
        self._demoting: dict[int, DemoteTicket] = {}
        self.step = 1
        # Contiguous commit frontier, min over ranks. 0 = none.
        self.acked_step = 0
        self._masked: set[tuple[str | None, int]] = set()
        self._by_producer: dict[tuple[str | None, int], set[int]] = {}
        # Content ids: next id to mint and the content id
        # each host block was copied from.
        self._next_cid = 1
        self.host_src: dict[int, int] = {}
        # Bumped by every GPU prefix-cache reset (lifecycle epoch).
        self.epoch = 0
        self.listener: TierBlockListener | None = None
        # Restore-dependency predicate; None = every eligible page.
        self.retains_fn: Callable[[KVCacheBlock], bool] | None = None
        self.counters = TierBlockCounters()
        # The running requests' write frontiers, set every
        # scheduler step by the write-back (``WriteBack.note_frontiers``):
        # blocks a running request may still write. Their content is not
        # final, so no lane copies them (``is_ready``, ``is_frontier``).
        self.live_ids: frozenset[int] = frozenset()

    # -- step / readiness inputs --------------------------------------
    def advance_step(self, step: int) -> None:
        self.step = step

    def set_acked_step(self, acked_step: int) -> None:
        """Every rank has retired every step <= ``acked_step`` (forward run,
        deferred commits executed or reported masked)."""
        self.acked_step = max(self.acked_step, acked_step)

    def mark_masked(self, producer: str | None, produced_step: int) -> list[int]:
        """A rank skipped the commit of ``producer``'s row at
        ``produced_step``: content stamped with it is never ready and never
        worth keeping. Returns the ids of the GPU blocks that carry it (the
        caller may un-hash them)."""
        ident = (producer, produced_step)
        bids = [
            bid
            for bid in self._by_producer.get(ident, ())
            if self._valid_stamp(bid) is not None
        ]
        if bids:
            self._masked.add(ident)
            self.counters.masked_blocks += len(bids)
        return bids

    # -- queries --------------------------------------------------------------
    def _valid_stamp(self, bid: int) -> ContentStamp | None:
        stamp = self._stamp[bid]
        if stamp is None or stamp.gen != self.generation[bid]:
            return None
        return stamp

    def stamp_of(self, block: KVCacheBlock) -> ContentStamp | None:
        """The stamp of the block's current content (None if unhashed)."""
        if block.is_null or block.block_hash is None:
            return None
        return self._valid_stamp(block.block_id)

    def is_masked(self, block: KVCacheBlock) -> bool:
        stamp = self.stamp_of(block)
        return stamp is not None and (stamp.producer, stamp.produced_step) in (
            self._masked
        )

    def is_ready(self, block: KVCacheBlock) -> bool:
        """The block's current content is final on every rank and no running
        request writes it any more: the copy predicate (a running
        request's write frontier is never a copy source, so a host copy matches
        its GPU source, on every lane and the reactive walk alike).

        A frontier needs its writer's reference: a table always holds one
        (``defer_block_free`` fences in-flight steps). A block released after
        the frontier was taken (a writer that finished without a grace; the
        gate in ``schedule()`` runs before ``account_running`` rebuilds
        ``live_ids``) is ref 0 and no longer written, so it stays ready:
        otherwise the just-freed writer's unit would lose its gate
        protection for one step."""
        return not self.is_frontier(block) and self.is_content_ready(block)

    def is_frontier(self, block: KVCacheBlock) -> bool:
        """The block is a running (or KV-holding) request's write frontier
        that is still being written: on the last step's frontier AND
        referenced (``ref_cnt > 0``, its writer's table). One predicate for
        every reader of ``live_ids``: ``is_ready`` (copy sources), the unit
        twin pick (``UnitTable._gpu_block``) and the whole-unit waiver
        (``WriteBack._issues_whole``). A ref-0 block in a stale set is no
        frontier for any of them (a stale twin must not resolve to
        its sibling while both count as copies)."""
        return block.block_id in self.live_ids and block.ref_cnt > 0

    def is_content_ready(self, block: KVCacheBlock) -> bool:
        """The block's current content is final on every rank (its stamped,
        keyed content: a running request's frontier page is still content-
        ready for its keyed slots). The prefix-lookup predicate."""
        stamp = self.stamp_of(block)
        return (
            stamp is not None
            and stamp.ready_step <= self.acked_step
            and (stamp.producer, stamp.produced_step) not in self._masked
        )

    def is_eligible(self, block: KVCacheBlock) -> bool:
        """Hashed and in a group the tier stores."""
        key = block.block_hash
        if block.is_null or key is None:
            return False
        return (
            self.eligible_group_ids is None
            or get_group_id(key) in self.eligible_group_ids
        )

    # -- content identity -----------------------------------------
    def _mint(self) -> int:
        cid = self._next_cid
        self._next_cid += 1
        return cid

    def content_id(self, block: KVCacheBlock) -> int | None:
        """The id of the block's current stamped content (None: unstamped)."""
        stamp = self.stamp_of(block)
        return None if stamp is None or not stamp.cid else stamp.cid

    def note_host_copy(self, host_block_id: int, cid: int) -> None:
        """A host block now holds a copy of content ``cid`` (0: unstamped
        source, which no GPU content can match)."""
        self.host_src[host_block_id] = cid

    def supersede_host_copies(
        self,
        host_pool: BlockPool,
        new: KVCacheBlock,
        keys: Iterable[BlockHashWithGroupId],
    ) -> int:
        """``new`` was just keyed under ``keys`` as a copy of a content: remove
        the keys of every other host block that were read from a DIFFERENT
        content (an older generation of the same tokens), so the host
        directory never offers older same-key bytes to a restore.

        Per physical block, and only the contested keys: a block whose
        every key is contested is evicted whole; otherwise only the contested
        keys leave its entries and its other keys, and the units that depend
        on them, stay. The same content copied twice is kept; a copy with no
        content id supersedes nothing. Returns the blocks touched."""
        cid = self.host_src.get(new.block_id, 0)
        if not cid:
            return 0
        keyset = set(keys)
        stale: dict[int, KVCacheBlock] = {}
        for key in sorted(keyset):  # total key order
            for other in host_pool.cached_block_hash_to_block.get_blocks(key):
                if other is not new and self.host_src.get(other.block_id) != cid:
                    stale[other.block_id] = other
        for other in stale.values():
            aliases = host_pool.cached_block_hashes_by_block.get(other.block_id, set())
            held = set(aliases)
            if other.block_hash is not None:
                held.add(other.block_hash)
            contested = held & keyset
            if held <= keyset:
                host_pool.evict_blocks({other.block_id})
                continue
            for key in contested:
                host_pool.cached_block_hash_to_block.pop(key, other.block_id)
            survivors = sorted(k for k in held if k not in contested)
            if other.block_hash in contested:
                other.reset_hash()
                other.set_block_hash(survivors[0], num_tokens=None)
                survivors = survivors[1:]
            if survivors:
                host_pool.cached_block_hashes_by_block[other.block_id] = set(survivors)
            else:
                host_pool.cached_block_hashes_by_block.pop(other.block_id, None)
            host_pool._emit_block_removed_events(sorted(contested, key=repr))
        self.counters.host_superseded += len(stale)
        return len(stale)

    def backs_key(self, block: KVCacheBlock, key: BlockHashWithGroupId) -> bool:
        """The host holds ``key`` in a block copied from ``block``'s current
        content (generation-qualified)."""
        host = self.host_pool
        if host is None:
            return False
        copies = host.cached_block_hash_to_block.get_blocks(key)
        if not copies:
            return False
        cid = self.content_id(block)
        if cid is None:
            # No identity to compare: fail closed. The block is dirty
            # until a copy of this stamped content lands.
            self.counters.backing_unproven += 1
            return False
        if any(self.host_src.get(h.block_id) == cid for h in copies):
            return True
        self.counters.backing_foreign += 1
        return False

    def host_backs(self, block: KVCacheBlock) -> bool:
        """The host holds every key ``block`` carries now, each in a copy of
        THIS content (same content id), not merely under an equal key."""
        key = block.block_hash
        if self.host_pool is None or key is None:
            return False
        if not self.backs_key(block, key):
            return False
        return all(
            self.backs_key(block, alias)
            for alias in self.pool.cached_block_hashes_by_block.get(
                block.block_id, ()
            )
        )

    def state(self, block: KVCacheBlock) -> BlockState:
        if block.block_id in self._demoting and not block.is_null:
            return BlockState.DEMOTING
        if not self.is_eligible(block) or self.is_masked(block):
            return BlockState.CLEAN
        if (
            self.retains_fn is not None
            and not self.retains_fn(block)
            and not self._host_holds_foreign(block)
        ):
            return BlockState.CLEAN  # no restore reads it
        return BlockState.CLEAN if self.host_backs(block) else BlockState.DIRTY

    def _host_holds_foreign(self, block: KVCacheBlock) -> bool:
        """The host holds a key ``block`` carries in a copy of OTHER content
        (an older generation of the same tokens, or one that cannot be
        compared). Such a page stays a dependency whatever ``retains_fn``
        says: host completeness is checked by key, so dropping the GPU copy
        undemoted would leave the older bytes to satisfy a restore; the walker
        copies it and the copy supersedes them
        (``supersede_host_copies``)."""
        host = self.host_pool
        if host is None:
            return False
        cid = self.content_id(block)
        keys = [block.block_hash]
        keys.extend(self.pool.cached_block_hashes_by_block.get(block.block_id, ()))
        for key in keys:
            copies = host.cached_block_hash_to_block.get_blocks(key)
            if copies and (
                cid is None
                or not any(self.host_src.get(h.block_id) == cid for h in copies)
            ):
                return True
        return False

    def is_dirty(self, block: KVCacheBlock) -> bool:
        return self.state(block) is BlockState.DIRTY

    # -- demotion (the write-back engine) -------------------------------
    def begin_demote(self, blocks: Iterable[KVCacheBlock]) -> list[DemoteTicket]:
        """Pin DIRTY sources for a host copy: touch them (out of the free
        queue) and mark them DEMOTING. Blocks that are not DIRTY are skipped.
        Readiness is the caller's check (``is_ready``)."""
        tickets: list[DemoteTicket] = []
        for block in blocks:
            if self.state(block) is not BlockState.DIRTY:
                continue
            assert block.block_hash is not None
            ticket = DemoteTicket(
                block,
                self.generation[block.block_id],
                block.block_hash,
                self.content_id(block) or 0,
            )
            self.pool.touch([block])
            self._demoting[block.block_id] = ticket
            tickets.append(ticket)
        self.counters.demote_issued += len(tickets)
        return tickets

    def demote_still_valid(self, ticket: DemoteTicket) -> bool:
        """The source still holds the content the copy read (same generation
        and primary): only then may the host copy be published."""
        block = ticket.block
        return (
            self.generation[block.block_id] == ticket.gen
            and block.block_hash == ticket.primary
        )

    def end_demote(
        self,
        tickets: Sequence[DemoteTicket],
        ok: Sequence[bool],
        front_ok: Sequence[bool] | None = None,
    ) -> list[DemoteTicket]:
        """Release demotion sources. ``ok[i]``: every rank copied ticket i and
        the caller published the host copy. A published source whose content
        is unchanged is CLEAN now; anything else is released normally and
        stays DIRTY. Returns the published (CLEAN) tickets.

        Release order: ``front_ok[i]`` says whether a published
        source may go to the FRONT of the free queue (the next reuse). Only a
        reactive copy may: its source was already at the reclaim head. A copy
        of a grace-held, preempted or running request's block (a grace or lane-1 copy)
        was made ahead of eviction, so a source that reaches ref 0 here goes
        to the tail like any free, keeping its GPU-hit value and never
        dragging its session group to the front. None = every published
        source to the front (the behaviour without the copy scheduler)."""
        front: list[KVCacheBlock] = []
        back: list[KVCacheBlock] = []
        normal: list[KVCacheBlock] = []
        done: list[DemoteTicket] = []
        if front_ok is None:
            front_ok = [True] * len(tickets)
        for ticket, good, to_front in zip(tickets, ok, front_ok, strict=True):
            block = ticket.block
            self._demoting.pop(block.block_id, None)
            if good and self.demote_still_valid(ticket):
                (front if to_front else back).append(block)
                done.append(ticket)
            else:
                normal.append(block)
        c = self.counters
        c.demote_completed += len(done)
        c.demote_abandoned += len(normal)
        # Normal releases first: an unhashed one is prepended too, and the
        # published sources must end up ahead of it.
        self.pool.free_blocks(normal)
        if back:
            self.pool.free_blocks(back)
        self.pool.free_blocks_to_front(front)
        return done

    def is_demoting(self, block: KVCacheBlock) -> bool:
        return block.block_id in self._demoting

    # -- BlockPool hooks -------------------------------------------------------
    def _new_content(self, bid: int, stamp: ContentStamp) -> None:
        self._stamp[bid] = stamp
        self._by_producer.setdefault((stamp.producer, stamp.produced_step), set()).add(
            bid
        )

    def _drop_content(self, bid: int) -> None:
        stamp = self._valid_stamp(bid)
        self.generation[bid] += 1
        if stamp is None:
            return
        ident = (stamp.producer, stamp.produced_step)
        owners = self._by_producer.get(ident)
        if owners is not None:
            owners.discard(bid)
            if not owners:
                del self._by_producer[ident]
                self._masked.discard(ident)

    def on_published(self, block: KVCacheBlock, request: Request) -> None:
        """``request`` published a key on ``block`` (cache_full_blocks /
        cache_partial_block). A first key stamps new content; an alias on
        content that is already stamped keeps its stamp."""
        if block.is_null or block.block_hash is None:
            return
        bid = block.block_id
        if self._valid_stamp(bid) is None:
            self._new_content(
                bid,
                ContentStamp(
                    self.generation[bid], self.step, request.request_id, self.step,
                    self._mint(),
                ),
            )
        if self.listener is not None:
            self.listener.on_gpu_keyed(block)

    def on_restored(
        self, block: KVCacheBlock, host_block_id: int | None = None
    ) -> None:
        """A quiesced host restore keyed ``block``: its bytes are final on
        every rank, so the content is stamped ready (no producing step that
        could be masked) and the directory is notified like any keying.

        The destination inherits the content id of the host copy it was read
        from (``host_block_id``), so that copy, and only that copy, backs it.
        With no known source the content gets a fresh id no host copy
        carries: it is never classed as backed (fail closed), never as
        backed-by-key-equality."""
        if block.is_null or block.block_hash is None:
            return
        bid = block.block_id
        if self._valid_stamp(bid) is None:
            cid = self.host_src.get(host_block_id, 0) if host_block_id is not None else 0
            self._new_content(
                bid,
                ContentStamp(
                    self.generation[bid], 0, None, 0, cid or self._mint()
                ),
            )
        if self.listener is not None:
            self.listener.on_gpu_keyed(block)

    def on_moved(self, src: KVCacheBlock, dst: KVCacheBlock) -> None:
        """``src``'s keys moved to ``dst`` (its CoW copy, written this step).
        ``src`` was already un-hashed (generation bumped), so its last stamp
        is read raw."""
        old = self._stamp[src.block_id]
        bid = dst.block_id
        self._drop_content(bid)
        if dst.block_hash is not None:
            self._new_content(
                bid,
                ContentStamp(
                    self.generation[bid],
                    max(old.ready_step, self.step) if old else self.step,
                    old.producer if old else None,
                    old.produced_step if old else self.step,
                    # CoW carries the content: same bytes, same identity.
                    old.cid if old and old.cid else self._mint(),
                ),
            )
        if self.listener is not None:
            self.listener.on_gpu_keyed(dst)

    def on_unhashed(
        self, block: KVCacheBlock, keys: Sequence[BlockHashWithGroupId]
    ) -> None:
        """``keys`` were removed from ``block``'s map entries; its own hash
        fields are reset right after."""
        if block.is_null:
            return
        self._drop_content(block.block_id)
        if self.listener is not None and keys:
            self.listener.on_gpu_unhashed(block, keys)

    def on_reclaim(self, block: KVCacheBlock) -> None:
        """``get_new_blocks`` hands ``block`` out; its keys are still there."""
        if block.is_null:
            return
        assert block.block_id not in self._demoting
        if block.block_hash is not None:
            st = self.state(block)
            c = self.counters
            if st is BlockState.DIRTY:
                c.reclaimed_dirty += 1
            elif self.is_eligible(block) and self.host_backs(block):
                c.reclaimed_backed += 1
            else:
                c.reclaimed_free += 1
                if (
                    self.retains_fn is not None
                    and self.is_eligible(block)
                    and not self.retains_fn(block)
                ):
                    c.reclaimed_unretained += 1
            if self.listener is not None:
                self.listener.on_gpu_reclaim(block, st)
        else:
            self.counters.reclaimed_free += 1
            # Allocation generation: unhashed reuse is new content too.
            self.generation[block.block_id] += 1

    def on_front_requeued(self, num_blocks: int) -> None:
        self.counters.front_requeued += num_blocks

    def on_reset(self) -> None:
        """Every GPU key was dropped (reset_prefix_cache)."""
        for bid in range(len(self.generation)):
            self.generation[bid] += 1
        self._by_producer.clear()
        self._masked.clear()
        self._demoting.clear()
        self.host_src.clear()
        self.epoch += 1
        if self.listener is not None:
            self.listener.on_gpu_reset()


def iter_pool_reclaims(
    pool: BlockPool, n: int, adopted: Collection[int] = ()
) -> Iterator[tuple[KVCacheBlock, bool, tuple[KVCacheBlock, ...]]]:
    """``(block, keyed, destroyed)`` for the next ``n`` blocks
    ``pool.get_new_blocks`` would hand out, in order, without changing
    anything. ``keyed``: the pop destroys content the block still holds
    (False for an unhashed block, or a peer an earlier pop of its session
    already un-hashed in shadow). ``destroyed``: the session peers this pop's
    destructive ``_break`` un-hashes (collateral: destroyed by this pop even
    when they are not among the next ``n`` reclaims). Shared by the GPU
    walker, the admission gate and the host order (one pop order
    read the same way on both tiers).

    ``adopted``: block ids an admission touches before it allocates. They
    leave the queue and their eviction groups first (``BlockPool.touch``),
    so the shadow removes them before simulating any pop or group break;
    they are never yielded."""
    groups = pool.evict_groups
    queue = pool.free_block_queue
    junctions = list(groups.junctions.free_evictable()) if groups.junctions else []
    out: set[int] = set(adopted)
    front: list[KVCacheBlock] = []
    # Front peers a destructive break un-hashes (shadowed, not applied).
    unhashed: set[int] = set()
    broken: set[int] = set()
    cur = queue.fake_free_list_head.next_free_block
    tail = queue.fake_free_list_tail
    while n > 0:
        while cur is not None and cur is not tail and cur.block_id in out:
            cur = cur.next_free_block
        head = front[0] if front else cur
        if head is None or head is tail:
            return
        junctions = [b for b in junctions if b.block_id not in out]
        if (
            head.block_hash is not None
            and head.block_id not in unhashed
            and junctions
        ):
            block = junctions.pop(0)
        elif front:
            block = front.pop(0)
        else:
            block = cur
        out.add(block.block_id)
        n -= 1
        keyed = block.block_hash is not None and block.block_id not in unhashed
        destroyed: list[KVCacheBlock] = []
        gid = groups._block_group.get(block.block_id)
        if gid is not None and gid not in broken:
            broken.add(gid)
            members = [
                m for m in groups._groups.get(gid, ()) if m.block_id not in adopted
            ]
            keep = groups.is_host_backed(members)
            for peer in members:
                if (
                    peer.block_id in out
                    or peer.ref_cnt != 0
                    or peer.prev_free_block is None
                ):
                    continue
                out.add(peer.block_id)
                front.append(peer)
                if not keep:
                    unhashed.add(peer.block_id)
                    if peer.block_hash is not None:
                        destroyed.append(peer)
        yield block, keyed, tuple(destroyed)


__all__ = [
    "BlockState",
    "BlockTierState",
    "ContentStamp",
    "DemoteTicket",
    "TierBlockCounters",
    "TierBlockListener",
    "iter_pool_reclaims",
]
