# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Census store-timing events, created once at backend init.

Building two timing events per census store on the copy thread
(``torch.Event(enable_timing=True)``) would make each store a cudaEventCreate
plus a cudaEventRecord in the middle of a forward pass, because torch creates
the CUDA event lazily at its first ``record``; such a create can stall the
copy thread in ``start.record(stream)`` next to a QSA module load. This pool
creates every pair up front, on the init thread. On the copy
thread a census store then costs two cudaEventRecord calls and no creation,
and both are ``torch.cuda.Event`` records, which run with the GIL released
(the generic ``torch.Event.record`` holds it).

A pair is reused only after its sample has left ``timings``. The worker's
``_collect_store_census`` reads ``elapsed_time`` and then deletes the entry,
after the event list has given up the end event, so a pair is never
re-recorded while anything still reads it. If more stores are outstanding than
the pool holds, the pool grows: that store builds its own pair, and
the pair joins the pool when it is reclaimed. The census numbers are unchanged:
the same events, recorded at the same points on the same stream.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)

#: Pairs created at init (128 events). A census store is one store event, at
#: most one per scheduler step, and stays in use until the worker polls it
#: complete. Past this many in flight the pool grows (warned once).
CENSUS_EVENT_PAIRS = 64


def _timing_event() -> Any:
    """``torch.cuda.Event``, not ``torch.Event``: torch 2.13's generic
    ``Event.record`` holds the GIL across cudaEventRecord (and the lazy
    create), the CUDA one releases it (torch/csrc/Event.cpp:97-129 vs
    torch/csrc/cuda/Event.cpp:127-144). Both events of a pair stay CUDA
    events: ``elapsed_time`` needs a CUDA peer."""
    import torch

    return torch.cuda.Event(enable_timing=True)


class CensusEventPool:
    """(start, end) timing-event pairs for the store copy thread.

    ``take``, ``commit`` and ``release`` are called only from the store copy
    thread. ``timings`` (the backend's ``store_census`` dict) is shared with
    the worker. The pool only tests membership in it, which is atomic under the
    GIL.
    """

    def __init__(
        self,
        stream: Any,
        size: int = CENSUS_EVENT_PAIRS,
        make_event: Callable[[], Any] = _timing_event,
    ) -> None:
        self._make_event = make_event
        self._free: deque[tuple[Any, Any]] = deque()
        # (event_idx, pair), in commit order.
        self._in_use: deque[tuple[int, tuple[Any, Any]]] = deque()
        self.grown = 0
        for _ in range(size):
            pair = (make_event(), make_event())
            # A torch event is created at its first record: do that here, on
            # the init thread, against the still-idle store stream.
            for event in pair:
                event.record(stream)
            self._free.append(pair)

    def take(self, timings: dict) -> tuple[Any, Any]:
        """A pair for the next census store; reclaims collected pairs first."""
        if self._in_use:
            pending: deque[tuple[int, tuple[Any, Any]]] = deque()
            for event_idx, pair in self._in_use:
                if event_idx in timings:
                    pending.append((event_idx, pair))
                else:
                    self._free.append(pair)
            self._in_use = pending
        if self._free:
            return self._free.popleft()
        self.grown += 1
        if self.grown == 1:
            logger.warning(
                "KV tier census: more than %d census stores in flight; the "
                "store thread creates timing events per store beyond that",
                len(self._in_use),
            )
        return (self._make_event(), self._make_event())

    def commit(self, event_idx: int, pair: tuple[Any, Any]) -> None:
        """``pair`` now times the sample ``timings[event_idx]``."""
        self._in_use.append((event_idx, pair))

    def release(self, pair: tuple[Any, Any]) -> None:
        """``pair`` was taken for a store that published no sample."""
        self._free.append(pair)


# Copy-loop entry points; ``pool`` is None when the backend was driven without
# one (``_run_copy_loop`` called directly), which keeps the per-store events.
def census_take(pool: CensusEventPool | None, timings: dict) -> tuple[Any, Any]:
    if pool is None:
        return (_timing_event(), _timing_event())
    return pool.take(timings)


def census_commit(pool: CensusEventPool | None, event_idx: int,
                  pair: tuple[Any, Any] | None) -> None:
    if pool is not None and pair is not None:
        pool.commit(event_idx, pair)


def census_release(pool: CensusEventPool | None,
                   pair: tuple[Any, Any] | None) -> None:
    if pool is not None and pair is not None:
        pool.release(pair)
