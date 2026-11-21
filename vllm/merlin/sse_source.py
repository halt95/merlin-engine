# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Close an SSE encoder's source when the encoder's stream is closed.

An encoder written as ``async for event in source: yield frame`` does not close
``source`` when it is itself closed while parked at its ``yield`` (a client that
goes away as a frame is handed over): the source is then left to garbage
collection and its cleanup runs at an unknown time. ``closing_source`` passes
the frames through and, when it ends for any reason, closes the encoder and
then the source, so the handler's cleanup has run when the close returns.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any


async def closing_source(
    frames: AsyncGenerator[str, None],
    source: AsyncGenerator[Any, None],
) -> AsyncGenerator[str, None]:
    try:
        async for frame in frames:
            yield frame
    finally:
        try:
            await frames.aclose()
        finally:
            await source.aclose()
