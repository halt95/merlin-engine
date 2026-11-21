# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""The KV pool a frontend publishes: the minimum over the engines it routes to.

Each engine reports its pool (``kv_cache_size_tokens``, the figure its boot
line "KV cache size: N tokens" prints) in its ready response. The frontend
keeps every report by DP rank, the key its engine identities encode, and
recomputes the minimum over the ranks it currently routes to whenever an
engine reports or the routed set changes (startup, elastic scale-up or
scale-down). Null only when no routed engine has reported a pool.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


class RoutedKVPool:
    def __init__(self) -> None:
        self.by_rank: dict[int, int] = {}

    def report(self, rank: int, pool_tokens: int | None) -> None:
        """Record one engine's ready response; a rank re-reporting replaces it."""
        if pool_tokens is None:
            # An engine without KV cache groups has no pool.
            self.by_rank.pop(rank, None)
        else:
            self.by_rank[rank] = pool_tokens

    def forget(self, ranks: Iterable[int]) -> None:
        """Drop ranks that left; a replacement reports afresh."""
        for rank in ranks:
            self.by_rank.pop(rank, None)

    def publish(self, cache_config: Any, engine_identities: Iterable[bytes]) -> None:
        """Set ``kv_cache_capacity_tokens`` from the currently routed engines."""
        routed = {int.from_bytes(identity, "little") for identity in engine_identities}
        pools = [pool for rank, pool in self.by_rank.items() if rank in routed]
        cache_config.kv_cache_capacity_tokens = min(pools) if pools else None
