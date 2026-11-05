# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Persistent snapshot storage for PP-deferred RecoverSSM commits.

A non-last PP rank snapshots the commit inputs of step T (state indices, query
offsets, request rows, block table, align counts) and commits them pp_size
steps later. The ring keeps ``depth`` slots of persistent buffers and copies the
inputs into them, so once a slot's buffers exist a snapshot into it allocates
nothing. A buffer is allocated on its first use, when an input outgrows it, and
again after ``reset``; a snapshot whose slot is still owned falls back to
``clone`` (see Lifetime below).

Each snapshot equals what ``Tensor.clone`` would return:

* values: ``dst.copy_(src)`` writes the source's values;
* layout: the destination is a view with the shape and the strides that
  ``clone`` (``torch.preserve_format``) gives the source, at offset 0 of a
  buffer from the caching allocator, so pointer alignment is that of a fresh
  allocation (the same Triton specialisation);
* sharing: a source copied twice in one snapshot (same data pointer, shape,
  strides and dtype, i.e. the same values) is copied once and the destination
  shared. Consumers only read snapshots, so sharing changes no value; it lets
  the align postprocess dedupe see one align count tensor per batch.

Lifetime: slot k is reused only when the snapshot that last used it is dead
(its commit has been enqueued on the stream, or the snapshot was dropped
unread) -- the owner is held through a weakref that the RecoverSSM state keeps
alive until the commit is enqueued. Copies and the commit run on the same
stream, so a later copy into the slot is ordered after the commit's reads. A
slot still owned when its turn comes is never overwritten: that snapshot falls
back to ``clone`` (fresh allocations).
"""

from __future__ import annotations

import weakref
from typing import Any

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


def _clone_strides(src: torch.Tensor) -> tuple[int, ...]:
    """The strides ``src.clone()`` would produce, without allocating device
    memory (a meta tensor computes the layout only)."""
    return torch.empty_like(src, device="meta").stride()


def _storage_numel(shape: tuple[int, ...], strides: tuple[int, ...]) -> int:
    if any(s == 0 for s in shape):
        return 0
    return 1 + sum((d - 1) * st for d, st in zip(shape, strides))


class _Slot:
    def __init__(self) -> None:
        self.owner: weakref.ref | None = None
        # One growable flat buffer per (copy position in the snapshot, dtype,
        # device): the snapshot copies the same fields in the same order every
        # step, so each position reuses its buffer.
        self.buffers: dict[tuple[int, torch.dtype, torch.device], torch.Tensor] = {}

    def busy(self) -> bool:
        return self.owner is not None and self.owner() is not None


class SnapshotCopier:
    """The ``copy`` callable of one snapshot, bound to one ring slot."""

    def __init__(self, slot: _Slot) -> None:
        self._slot = slot
        self._pos = 0
        self._done: list[tuple[tuple[Any, ...], torch.Tensor]] = []

    def __call__(self, src: torch.Tensor) -> torch.Tensor:
        strides = _clone_strides(src)
        key = (
            src.data_ptr(),
            tuple(src.shape),
            tuple(src.stride()),
            src.dtype,
            src.device,
        )
        for k, dst in self._done:
            if k == key:
                return dst
        need = _storage_numel(tuple(src.shape), strides)
        bkey = (self._pos, src.dtype, src.device)
        self._pos += 1
        buf = self._slot.buffers.get(bkey)
        if buf is None or buf.numel() < need:
            buf = torch.empty(max(need, 1), dtype=src.dtype, device=src.device)
            self._slot.buffers[bkey] = buf
        dst = buf.as_strided(tuple(src.shape), strides, 0)
        dst.copy_(src)
        self._done.append((key, dst))
        return dst


class RecoverSSMSnapshotRing:
    """``depth`` slots, taken round-robin. ``depth`` = the PP handler's queue
    depth (pp_size in-flight steps) + 1."""

    def __init__(self, depth: int) -> None:
        if depth < 1:
            raise ValueError(f"snapshot ring depth must be >= 1, got {depth}")
        self.depth = depth
        self._slots = [_Slot() for _ in range(depth)]
        self._next = 0
        self.fallbacks = 0  # proof counter: snapshots taken with clone()

    def acquire(self) -> tuple[_Slot | None, SnapshotCopier | None]:
        """The next slot and its copier, or (None, None) when that slot's
        previous snapshot is still alive (the caller then clones)."""
        slot = self._slots[self._next]
        if slot.busy():
            self.fallbacks += 1
            logger.warning_once(
                "RecoverSSM snapshot ring: slot still in use (deferred commit "
                "not yet enqueued); falling back to clone()"
            )
            return None, None
        self._next = (self._next + 1) % self.depth
        return slot, SnapshotCopier(slot)

    @staticmethod
    def bind(slot: _Slot, owner: Any) -> None:
        """Mark ``slot`` owned by ``owner`` (the deferred step) until it dies."""
        slot.owner = weakref.ref(owner)

    def reset(self) -> None:
        """Drop every buffer (KV cache re-initialization)."""
        self._slots = [_Slot() for _ in range(self.depth)]
        self._next = 0
