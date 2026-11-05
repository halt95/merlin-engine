# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""The restore admission cap of the host KV tier.

``RestoresInFlight`` counts the restores that are running and refuses a new one
once the cap (``MERLIN_KV_HOST_TIER__N_RESTORE``) is reached. Host rows are not
retained by a ledger, so this is the module's only content.
"""

from __future__ import annotations


class RestoresInFlight:
    def __init__(self, cap: int | None = None) -> None:
        if cap is None:
            from vllm.merlin.envs import read_registered

            cap = read_registered("MERLIN_KV_HOST_TIER__N_RESTORE")
        if cap < 1:
            raise ValueError("restore cap must be positive")
        self.cap = cap
        self.count = 0

    def can_start(self) -> bool:
        return self.count < self.cap

    def start(self) -> bool:
        """Reserve one restore slot; False leaves a full cap unchanged."""
        if not self.can_start():
            return False
        self.count += 1
        return True

    def done(self) -> None:
        """Release a slot once on completion, failure or cancellation.
        Epoch reset alone must not release a restore whose DMA is in flight."""
        if self.count == 0:
            raise RuntimeError("restore slot released without a matching start")
        self.count -= 1
