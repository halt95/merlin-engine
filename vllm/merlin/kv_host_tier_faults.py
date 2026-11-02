# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Debug-only host-tier transfer faults, for fault-injection tests.

MERLIN_KV_HOST_TIER_FAIL_LOADS=N fails the first N nonempty restore transfers
on each rank's backend; 0/unset is off. Restart to reset the quota. A transfer
can contain multiple requests and blocks. On multi-chunk transfers the error
is raised before the final chunk, after earlier chunks have been submitted,
so the real failed-event fence must protect partially written destinations.

MERLIN_KV_HOST_TIER_FAIL_STORES=N does the same for the first N nonempty
demotion (GPU to host) transfers on each rank's store stream, so a fault-injection
test can drive a real failed demotion through the per-rank ledger: the rank reports
success=False, the scheduler publishes nothing, the sources stay GPU-resident
and dirty and the staged host blocks are released. Each quota is independent.

The backend invokes this only for generated-tier restores (failed_events is
provided), inside the same try/except as copy_blocks. It does not fabricate
failed receives, invalidate host entries or change scheduler behavior.
"""


class RestoreFailureInjector:
    KNOB = "MERLIN_KV_HOST_TIER_FAIL_LOADS"
    WHAT = "restore"

    def _limit(self) -> int:
        import vllm.envs as envs

        return envs.MERLIN_KV_HOST_TIER_FAIL_LOADS

    def __init__(self) -> None:
        self.limit = self._limit()
        self.injected = 0
        self._remaining = self.limit
        self._selected = False
        if self.limit:
            from vllm.logger import init_logger

            init_logger(__name__).warning(
                "DEBUG FAULT INJECTION ACTIVE: %s=%d; the first %d host-tier "
                "%s transfers on this rank will fail (unset this knob after "
                "the test)",
                self.KNOB,
                self.limit,
                self.limit,
                self.WHAT,
            )

    def before_copy(
        self, event_idx: int, *, first_chunk: bool, last_chunk: bool
    ) -> None:
        if not self.limit:
            return
        if first_chunk:
            self._selected = self._remaining > 0
            self._remaining = max(0, self._remaining - 1)
        if self._selected and last_chunk:
            self.injected += 1
            raise OSError(
                f"{self.KNOB}: injected {self.WHAT}-copy "
                f"failure {self.injected}/{self.limit} for event {event_idx}"
            )


class StoreFailureInjector(RestoreFailureInjector):
    """FAIL_STORES: the same selection and chunk rule on the store stream."""

    KNOB = "MERLIN_KV_HOST_TIER_FAIL_STORES"
    WHAT = "demotion"

    def _limit(self) -> int:
        import vllm.envs as envs

        return envs.MERLIN_KV_HOST_TIER_FAIL_STORES
