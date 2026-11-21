# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""State of one background Responses stream, shared by its producer and readers.

The producer appends events and sets ``signal`` after each one. A reader drains
from its own cursor and then waits on ``signal``. ``done`` is the producer's
last word: it is set, before the final ``signal``, when the producer has ended
for any reason (finished, cancelled or failed), so a reader that has drained
everything stops instead of waiting for an event that cannot come. ``failure``
holds the producer's exception when it failed; a cancelled producer leaves it
``None`` and its readers end without an error.

The state is stored when the request is accepted, before the producer task is
scheduled, and the task's done callback calls ``end()``: a task cancelled before
its first step never runs the producer, and its readers must still end.
"""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field
from typing import Any

TERMINAL_EVENTS = ("response.completed", "response.incomplete")


@dataclass
class BackgroundStream:
    events: deque[Any] = field(default_factory=deque)
    signal: asyncio.Event = field(default_factory=asyncio.Event)
    done: bool = False
    failure: BaseException | None = None

    def end(self) -> None:
        """Mark the producer ended and wake the readers. Safe to call twice."""
        self.done = True
        self.signal.set()

    def completed(self) -> bool:
        """Whether the last stored event is one a finished response ends with."""
        last = self.events[-1] if self.events else None
        return getattr(last, "type", None) in TERMINAL_EVENTS


async def mark_failed(serving: Any, response_id: str) -> None:
    """Set the stored response to ``failed`` unless it already has a final
    status. A response that was never stored, or is gone, is left alone. When
    the stream's last event is an ``error`` event, its cause is kept on the
    stored response."""
    stream = serving.event_store.get(response_id)
    last = stream.events[-1] if stream is not None and stream.events else None
    async with serving.response_store_lock:
        stored = serving.response_store.get(response_id)
        if stored is not None and stored.status not in ("completed", "cancelled"):
            stored.status = "failed"
            if getattr(last, "type", None) == "error":
                from vllm.merlin.responses_errors import response_error

                stored.error = response_error(last)
