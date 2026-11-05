# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Client tool-call indexes for parsers that can reject a call.

The engine counts every call it opens, accepted or not, and streams that raw
count as ``tool_calls[].index``. Once ``validate_tool_names`` rejects raw call 0,
the surviving call reaches the client at index 1 with nothing at 0, and
``maybe_filter_parallel_tool_calls`` (which keeps index 0) drops it. Clients must
see consecutive indexes from 0, so each accepted call takes the number of
accepted calls before it. Nothing changes when no call was rejected.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from vllm.entrypoints.generate.base.protocol import DeltaMessage


class _Slot(Protocol):
    name_sent: bool


def renumber_accepted_calls(
    delta: DeltaMessage | None, slots: Sequence[_Slot]
) -> DeltaMessage | None:
    """Rewrite raw engine indexes in ``delta`` to client indexes, in place."""
    if delta is not None and delta.tool_calls:
        for call in delta.tool_calls:
            call.index = sum(slot.name_sent for slot in slots[: call.index])
    return delta
