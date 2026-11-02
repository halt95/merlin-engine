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
* Evict first. When an allocation would evict a cached block (the free
  queue's head is hashed, i.e. the never-cached blocks are used up), a free
  junction block is taken instead, oldest first: it is reused after every
  never-cached block and before any ordinary cached block. A block with a
  reference (a table, a waiter pin, a pending copy, a grace) is never in the
  free queue, so it is never a candidate.
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

from typing import TYPE_CHECKING

from vllm.v1.core.kv_cache_utils import (
    BlockHashWithGroupId,
    KVCacheBlock,
    make_block_hash_with_group_id,
)
from vllm.v1.request import Request

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
        that already carries a hash is published as before, without
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
        return published

    def follow(self, source: KVCacheBlock, cow: KVCacheBlock) -> bool:
        """``source``'s hashes were moved to ``cow`` (the producer keeps
        writing ``source``). Returns whether a junction moved."""
        key = self._keys.pop(source.block_id, None)
        if key is None or cow.block_hash != key:
            return False
        self._keys[cow.block_id] = key
        return True

    def forget(self, block: KVCacheBlock) -> None:
        """``block`` is handed out for reuse."""
        self._keys.pop(block.block_id, None)

    def clear(self) -> None:
        self._keys.clear()

    def free_evictable(self) -> list[KVCacheBlock]:
        """The free junction blocks ``take_evictable`` would hand out, in its
        order, without removing them."""
        out: list[KVCacheBlock] = []
        for block_id in self._keys:
            block = self._pool.blocks[block_id]
            if (
                self.is_junction(block)
                and block.ref_cnt == 0
                and block.prev_free_block is not None
            ):
                out.append(block)
        return out

    def take_evictable(self) -> KVCacheBlock | None:
        """The free junction block to reuse instead of the free queue's head,
        or None to pop the head: only when the head is a cached block (every
        never-cached block is gone) and a free junction block exists. The
        block is removed from the free queue."""
        queue = self._pool.free_block_queue
        head = queue.fake_free_list_head.next_free_block
        if head is None or head is queue.fake_free_list_tail or head.block_hash is None:
            return None
        for block_id in list(self._keys):
            block = self._pool.blocks[block_id]
            if not self.is_junction(block):
                del self._keys[block_id]
                continue
            if block.ref_cnt == 0 and block.prev_free_block is not None:
                queue.remove(block)
                return block
        return None


__all__ = ["JunctionCheckpoints"]
