# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Storage leases for host-tier restore transactions.

Manifest retention is separate from DMA credits. The complete destination
layout is allocated by KVCacheManager before a lease is created; no copy may
be issued until every member has been validated and pinned.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm.v1.core.block_pool import BlockPool

# A missing completion cannot turn a DMA address into reusable storage.
TRANSFER_TIMEOUT_S = 300.0


class TxnState(Enum):
    """Resolution of a transaction from its per-rank reports."""

    PENDING = "pending"  # a rank is missing or has not proven quiescence
    COMMIT = "commit"  # every rank succeeded and is quiescent
    UNWIND = "unwind"  # some rank failed and every rank is quiescent


@dataclass
class TxnAcks:
    """Per-rank acks of one transaction, keyed (epoch, txn, batch).

    A rank reports ``(success, quiescent)``. Failure is sticky: a later
    success from the same rank cannot undo it. Quiescence is monotone: once a
    rank proved its copies stopped it cannot un-prove it, and a repeated
    report is idempotent. The transaction resolves only when every rank has
    reported and proven quiescence, so a rank that failed but may still be
    reading or writing keeps the storage quarantined (never reusable) until
    it reports quiescent or the age bound fails the engine stop.
    """

    epoch: int
    txn: int
    batch: int
    world_size: int
    ranks: dict[int, tuple[bool, bool]] = field(default_factory=dict)

    def report(self, rank: int, success: bool, quiescent: bool) -> None:
        old = self.ranks.get(rank)
        if old is not None:
            success = old[0] and success
            quiescent = old[1] or quiescent
        self.ranks[rank] = (success, quiescent)

    @property
    def failed(self) -> bool:
        return any(not ok for ok, _ in self.ranks.values())

    @property
    def state(self) -> TxnState:
        if len(self.ranks) < self.world_size or not all(
            quiet for _, quiet in self.ranks.values()
        ):
            return TxnState.PENDING
        return TxnState.UNWIND if self.failed else TxnState.COMMIT


@dataclass(frozen=True)
class RestoreMember:
    cpu_id: int
    gpu_id: int
    key: Any
    num_tokens: int
    source_keys: tuple[Any, frozenset]
    # A plan restore's partial boundary page. Its destination is staging the
    # consumer never writes (it continues on a private copy), so it is
    # published like any other member.
    private: bool = False


@dataclass(frozen=True)
class RestoreLease:
    epoch: int
    key: tuple
    members: tuple[RestoreMember, ...]

    def sources_valid(self, pool: BlockPool) -> bool:
        for member in self.members:
            source = pool.blocks[member.cpu_id]
            keys = (
                source.block_hash,
                frozenset(pool.cached_block_hashes_by_block.get(member.cpu_id, ())),
            )
            if keys != member.source_keys:
                return False
        return True

    def published(self, pool: BlockPool) -> bool:
        # Normal request CoW may already have moved the publication elsewhere.
        return all(
            pool.cached_block_hash_to_block.get_one_block(member.key) is not None
            for member in self.members
        )

    def publish_cancelled(
        self,
        pool: BlockPool,
        computed_tokens: int | None = None,
        state_groups: frozenset[int] = frozenset(),
    ) -> int:
        """Publish quiesced immutable content after its consumer was removed.

        Each key stops at the accepted boundary, even if the host source also
        proves a longer append-only prefix. A future consumer adopts this
        cached content through the ordinary private continuation/CoW path.

        A member of a recurrent-state group (``state_groups``) is overwritten
        in place by the leader's continuation, so its bytes only still equal
        the restored state while the leader has not computed past the member's
        boundary (``computed_tokens``). Past that, the key would name advanced
        bytes: that member is not published (the unit stays incomplete and
        its consumers recompute). Published members go through the tier-aware
        ``BlockPool.publish_restored``, which installs the content
        generation, readiness and directory notification with the key.

        Restored recurrent and partial destinations are immutable staging:
        their consumer continues on private copies (``TierPlans._privatize``),
        so the guard above only matters if a copy could not be made.

        Returns the number of members published."""
        from vllm.v1.core.kv_cache_utils import get_group_id

        members = list(self.members)
        for member in members:
            block = pool.blocks[member.gpu_id]
            if block.block_hash is not None and block.block_hash != member.key:
                raise RuntimeError("host-tier restore destination changed under lease")
        published = 0
        for member in members:
            if (
                get_group_id(member.key) in state_groups
                and computed_tokens is not None
                and computed_tokens > member.num_tokens
            ):
                continue
            pool.publish_restored(
                member.key, pool.blocks[member.gpu_id], member.num_tokens,
                host_block_id=member.cpu_id,
            )
            published += 1
        return published
