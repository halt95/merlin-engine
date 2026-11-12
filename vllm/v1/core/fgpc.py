# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Soft shared-prefix checkpoints for ``--enable-mamba-shared-prefix-checkpoint``
(FGPC).

With FGPC on, a Mamba "align" group publishes its recurrent state at a
shared-prefix junction (a sub-block position where full attention has cached
further than the GDN group has a state, ``Request.shared_prefix_boundary``),
so later siblings resume there instead of at the block floor. That state is a
pure cache entry: no request needs it once its copy-on-write copies have run.

The rules here make it cost no live capacity:

* No hold. Neither the producer (whose live state moved into a CoW copy) nor a
  consumer (which restores from it into its own CoW copy) keeps a reference on
  a junction block beyond the copy retention, so after the copy fence the
  block is ref 0, still hashed, in the free queue: cached and evictable,
  reported free by ``BlockPool.get_usage``. Holds on prompt-tail checkpoints
  (the multi-turn resume points) are unchanged.
* Ordinary LRU. A free junction block ages in the free queue like any other
  cached block: after every never-cached block, in least-recently-used order,
  and each sibling that resumes from it touches it again. A block with a reference (a
  table, a waiter pin, a pending copy, a grace) is never in the free queue,
  so it is never a candidate.
* One per prefix. A junction is published only when no block holds its key
  yet (concurrent siblings that observed the same junction in one step do not
  each materialize a copy).
* Never in a coherent eviction group (``cache_thrash.EvictGroups``): a
  junction is shared by every sibling, so it is not one session's block, and
  reusing it must not un-hash a session's peers.

A block counts as a junction only while its sole hash is the junction key it
was published under; provenance is dropped when the block is handed out for
reuse.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.v1.core.kv_cache_utils import (
    BlockHashWithGroupId,
    KVCacheBlock,
    get_block_hash,
    get_group_id,
    make_block_hash_with_group_id,
)
from vllm.v1.request import Request

logger = init_logger(__name__)


def _diag_on() -> bool:
    """MERLIN_KV_TIER_DIAG=1: log junction publish / pop lines (debug only)."""
    try:
        from vllm.merlin.envs import read_registered

        return bool(read_registered("MERLIN_KV_TIER_DIAG"))
    except Exception:  # a debug switch must never break the pool
        return False


def _diag(event: str, key: BlockHashWithGroupId, block: KVCacheBlock, **extra) -> None:
    tail = "".join(f" {k}={v}" for k, v in extra.items())
    logger.info(
        "KV tier diag: junction %s t=%.3f group=%d key=%s block=%d%s",
        event, time.monotonic(), get_group_id(key),
        bytes(get_block_hash(key))[:6].hex(), block.block_id, tail,
    )


if TYPE_CHECKING:
    from vllm.v1.core.block_pool import BlockPool


class JunctionCheckpoints:
    """Provenance and eviction order of one ``BlockPool``'s junction
    checkpoints."""

    def __init__(self, pool: BlockPool) -> None:
        self._pool = pool
        # block id -> the junction key the block was published under, in
        # publication order (oldest first: the eviction order among them).
        self._keys: dict[int, BlockHashWithGroupId] = {}
        self._diag = _diag_on()

    def __bool__(self) -> bool:
        return bool(self._keys)

    def __len__(self) -> int:
        return len(self._keys)

    def is_junction(self, block: KVCacheBlock) -> bool:
        """``block`` still holds exactly the junction state it was published
        as (no other hash was added, none was removed)."""
        key = self._keys.get(block.block_id)
        return (
            key is not None
            and block.block_hash == key
            and block.block_id not in self._pool.cached_block_hashes_by_block
        )

    def publish(
        self,
        request: Request,
        block: KVCacheBlock,
        num_tokens: int,
        kv_cache_group_id: int,
        block_size: int,
    ) -> BlockHashWithGroupId | None:
        """``BlockPool.cache_partial_block`` for a junction, at most one block
        per junction key. Returns the key, or None if not published. A block
        that already carries a hash is published without
        junction provenance (it is more than a junction)."""
        if block.block_hash is not None:
            return self._pool.cache_partial_block(
                request=request,
                block=block,
                num_tokens=num_tokens,
                kv_cache_group_id=kv_cache_group_id,
                block_size=block_size,
            )
        unit = self._pool.hash_block_size
        key = make_block_hash_with_group_id(
            request.block_hashes[num_tokens // unit - 1], kv_cache_group_id
        )
        if self._pool.cached_block_hash_to_block.get_one_block(key) is not None:
            return None
        published = self._pool.cache_partial_block(
            request=request,
            block=block,
            num_tokens=num_tokens,
            kv_cache_group_id=kv_cache_group_id,
            block_size=block_size,
        )
        if published is not None:
            self._keys[block.block_id] = published
            if self._diag:
                _diag("publish", published, block, tokens=num_tokens,
                      req=request.request_id)
        return published

    def follow(self, source: KVCacheBlock, cow: KVCacheBlock) -> bool:
        """``source``'s hashes were moved to ``cow`` (the producer keeps
        writing ``source``). Returns whether a junction moved."""
        key = self._keys.pop(source.block_id, None)
        if key is None or cow.block_hash != key:
            return False
        self._keys[cow.block_id] = key
        if self._diag:
            _diag("move", key, cow, source=source.block_id)
        return True

    def forget(self, block: KVCacheBlock) -> None:
        """``block`` is handed out for reuse."""
        key = self._keys.pop(block.block_id, None)
        if self._diag and key is not None and block.block_hash == key:
            _diag("pop", key, block, reason="reuse")

    def clear(self) -> None:
        if self._diag and self._keys:
            logger.info("KV tier diag: junction reset t=%.3f dropped=%d",
                        time.monotonic(), len(self._keys))
        self._keys.clear()

    def free_evictable(self) -> list[KVCacheBlock]:
        """The free junction blocks to take ahead of the free queue's order:
        none. Junctions age as ordinary cached blocks (module doc), so the
        GPU walker, the admission shadow and the host order, which all read
        this, follow plain queue order, as ``take_evictable`` does."""
        return []

    def take_evictable(self) -> KVCacheBlock | None:
        """The free junction block to reuse instead of the free queue's head:
        never (None pops the head). Kept with ``free_evictable`` so the real
        pop and the shadow walks stay one order."""
        return None


__all__ = ["JunctionCheckpoints"]
