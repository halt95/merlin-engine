# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Slow-fill guard for tool-call grammars: a backstop, default on.

A grammar mask is filled on the engine-core thread, so one request whose fills take
seconds holds every stream. With ``VLLM_MERLIN_STRICT_STALL_GUARD=1`` (default) each
mask fill of a tool-call request (a structural-tag grammar) that takes at least
``VLLM_MERLIN_STRICT_STALL_FLOOR_MS`` of CPU time (default 300 ms) counts toward the
request's slow-fill time over its last ``VLLM_MERLIN_STRICT_STALL_WINDOW`` fills
(default 64); faster fills add nothing and reset nothing. When that time reaches
``VLLM_MERLIN_STRICT_STALL_BUDGET_MS`` (default 5000 ms) the request ends with an
error finish reason, the way a request whose tokens the grammar rejects ends; other
requests continue. Each end is a server log line naming the request and its slow
fills; ``fires`` counts them.

The clock is the calling thread's CPU time, so a loaded host cannot trip it. Healthy
fills stay under the floor (the slowest seen on today's budgeted grammar took 280 ms
of wall time on the test host), so healthy traffic adds nothing; a stall like the
whitespace one (fills growing past a second) is ended after at most the budget plus
one fill of slow-fill time. With the strict whitespace cap on that stall cannot
happen; the guard is there for pathologies not yet seen and for the cap off.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any

from vllm import envs
from vllm.logger import init_logger

logger = init_logger(__name__)

_lock = threading.Lock()
_settings: tuple[bool, float, float, int] | None = None
# per grammar (by id) the CPU time (ms) of each mask fill in the current step
_step: dict[int, list[float]] = {}
# per request with a slow fill in its window: [the window's counted times (0 for a
# fill under the floor), their sum]
_slow: dict[str, list[Any]] = {}
fires = 0
_MAX_TRACKED = 4096


def settings() -> tuple[bool, float, float, int]:
    """(on, floor ms, budget ms, window in fills), read once."""
    global _settings
    if _settings is None:
        _settings = (
            bool(envs.VLLM_MERLIN_STRICT_STALL_GUARD),
            float(envs.VLLM_MERLIN_STRICT_STALL_FLOOR_MS),
            float(envs.VLLM_MERLIN_STRICT_STALL_BUDGET_MS),
            int(envs.VLLM_MERLIN_STRICT_STALL_WINDOW),
        )
    return _settings


def on() -> bool:
    return settings()[0]


def timed_fill(grammar: Any, bitmask: Any, index: int) -> None:
    """``grammar.fill_bitmask`` with its compute time added to the grammar's step.

    The timer is the calling thread's CPU time (``time.thread_time``) around the fill
    call alone: it leaves out time the thread waits for the GIL or the CPU and, on the
    parallel path, the wait in the fill pool's queue (the timer starts in the worker).
    """
    start = time.thread_time()
    grammar.fill_bitmask(bitmask, index)
    spent = (time.thread_time() - start) * 1e3
    with _lock:
        _step.setdefault(id(grammar), []).append(spent)


def add_fills(req_id: str, fills: list[float]) -> None:
    """Count one request's fill times (ms) into its window."""
    _, floor, _, window = settings()
    entry = _slow.get(req_id)
    if entry is None:
        if max(fills) < floor:
            return
        entry = _slow[req_id] = [deque(maxlen=window), 0.0]
    times = entry[0]
    for ms in fills:
        counted = ms if ms >= floor else 0.0
        if len(times) == times.maxlen:
            entry[1] -= times[0]
        times.append(counted)
        entry[1] += counted
    if entry[1] <= 1e-9:
        del _slow[req_id]


def end_step(requests: dict[str, Any], request_ids: list[str]) -> None:
    """Close the step: add each tool-call request's fills to its window."""
    with _lock:
        for req_id in request_ids:
            request = requests.get(req_id)
            so = getattr(request, "structured_output_request", None)
            grammar = getattr(so, "grammar", None)
            fills = _step.pop(id(grammar), None) if grammar is not None else None
            if not fills:
                continue
            if req_id in _slow or (max(fills) >= settings()[1] and _guarded(so)):
                add_fills(req_id, fills)
        _step.clear()
        if len(_slow) > _MAX_TRACKED:  # requests that finished while slow
            _slow.clear()


def _guarded(so: Any) -> bool:
    from vllm.v1.structured_output.backend_types import StructuredOutputOptions

    key = getattr(so, "structured_output_key", None)
    return bool(key) and key[0] == StructuredOutputOptions.STRUCTURAL_TAG


def should_end(req_id: str) -> bool:
    """True once: the request's slow-fill time in its window reached the budget."""
    global fires
    if not _slow:
        return False
    _, floor, budget, _ = settings()
    with _lock:
        entry = _slow.get(req_id)
        if entry is None or entry[1] < budget:
            return False
        del _slow[req_id]
        fires += 1
        count = fires
    slow = [t for t in entry[0] if t > 0.0]
    logger.error(
        "Strict stall guard: request %s ended with an error: %d grammar mask fills "
        "of at least %d ms CPU took %.0f ms in its last %d fills (%s ms; budget %d "
        "ms, VLLM_MERLIN_STRICT_STALL_BUDGET_MS); %d request(s) ended so far.",
        req_id,
        len(slow),
        int(floor),
        entry[1],
        len(entry[0]),
        ", ".join(f"{t:.0f}" for t in slow),
        int(budget),
        count,
    )
    return True


def forget(req_id: str) -> None:
    with _lock:
        _slow.pop(req_id, None)
