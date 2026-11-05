# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Engine-progress watchdog behind ``/health``.

With stats logging on, the EngineCore sends the frontend an
``EngineCoreOutputs`` after every step, scheduler stats included when no request
produced a token, so a frontend with unfinished requests that has heard nothing
for longer than any legitimate step is looking at a stalled engine (a stuck
step or a deadlocked busy loop), unless it is holding the requests without
stepping on purpose: a
paused scheduler (pause_generation mode "keep"), a sleeping engine, or a
streaming-input session waiting for its next chunk. So after the silence the
watchdog sends the EngineCore a utility call, which its busy loop answers
between steps: an answer within HOLD_PROBE_S means the loop is turning (the
clock restarts), no answer means it is stuck in a step or deadlocked.
``/health`` then answers 503 (the error subclasses EngineDeadError, which is
what the endpoint maps to 503) until the next output.

Off (timeout 0) unless this frontend sees every step of a single engine: with
stats logging off, or with several API servers (stats go to one of them), a
long prefill sends a frontend nothing between chunks, and with several
data-parallel engines one engine's outputs would mask another's stall.
"""

import asyncio
from time import monotonic
from typing import Any

from vllm.exceptions import VLLMServerError
from vllm.logger import init_logger
from vllm.v1.engine.exceptions import EngineDeadError

logger = init_logger(__name__)

HOLD_PROBE_S = 5.0


class EngineStalledError(EngineDeadError):
    """The EngineCore is alive but has completed no step within the timeout."""

    def __init__(self, stalled_s: float, timeout_s: float):
        VLLMServerError.__init__(
            self,
            f"EngineCore has completed no step for {stalled_s:.0f}s with requests "
            f"in flight (VLLM_MERLIN_HEALTH_STALL_S={timeout_s:g})",
        )


class StallWatch:
    """Monotonic time of the engine's last sign of progress, per stall episode."""

    def __init__(self, timeout_s: float):
        self.timeout_s = timeout_s
        self._last = monotonic()
        self._episode = 0  # bumped by every sign of progress
        self._reported = False
        self._probe: asyncio.Future | None = None  # one in flight per episode

    def progress(self) -> None:
        """An output arrived from the EngineCore (or it answered a probe)."""
        self._last = monotonic()
        self._episode += 1
        self._reported = False
        self._probe = None

    def admit(self, was_idle: bool) -> None:
        """A request is being added: work that starts after an idle spell is timed
        from now, not from the last output before the idle spell."""
        if was_idle:
            self.progress()

    def _answered(self, probe: asyncio.Future, episode: int) -> None:
        # A reply that lands after the probe timed out still proves the loop turns
        # (an engine holding paused or streaming-input requests sends no output).
        if probe.cancelled() or probe.exception() is not None:  # retrieved either way
            return
        if episode == self._episode:
            self.progress()

    async def check(self, busy: bool, engine_core: Any) -> None:
        """Raise EngineStalledError when requests are in flight, the engine has been
        silent for longer than the timeout and its busy loop does not answer."""
        if self.timeout_s <= 0:
            return
        if not busy:
            self.progress()  # an idle engine sends nothing and is healthy
            return
        stalled_s = monotonic() - self._last
        if stalled_s <= self.timeout_s:
            return
        episode = self._episode
        if not self._reported:  # once reported, no further probes until progress
            if self._probe is None:  # concurrent /health calls share one probe
                self._probe = asyncio.ensure_future(engine_core.is_sleeping_async())
                self._probe.add_done_callback(
                    lambda done: self._answered(done, episode)
                )
            probe = self._probe
            # asyncio.wait never cancels the probe; a late answer reaches _answered.
            done, _ = await asyncio.wait({probe}, timeout=HOLD_PROBE_S)
            if done:
                self._answered(probe, episode)
            if self._episode != episode:  # it answered, or an output arrived
                return
            if not self._reported:  # a concurrent call may have reported it
                self._reported = True
                logger.error(
                    "EngineCore stalled: no step completed for %.0fs with requests "
                    "in flight; /health answers 503 until it makes progress",
                    stalled_s,
                )
        raise EngineStalledError(stalled_s, self.timeout_s)
