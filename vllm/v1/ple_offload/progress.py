# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Independent progress deadline for PLE transactions."""

import math
import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

from vllm.v1.ple_offload.fatal import ple_fail_closed

_DEFAULT_SERVING_DEADLINE_S = 300.0
_DEFAULT_STARTUP_DEADLINE_S = 1800.0
_MAX_PROGRESS_DEADLINE_S = 86400.0
_POLL_INTERVAL_S = 0.1


def _validate_deadline(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0 or value > _MAX_PROGRESS_DEADLINE_S:
        raise ValueError(
            f"{name} must be finite, greater than zero, and no greater than "
            f"{_MAX_PROGRESS_DEADLINE_S:g} seconds"
        )
    return value


def _environment_deadline(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        # Unset or empty means the default (matches the envs registry).
        return _validate_deadline(float(default), name)
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"{name}={raw!r} is not a number of seconds") from None
    return _validate_deadline(value, name)


@dataclass(frozen=True)
class _Submission:
    sequence: int
    started: float
    startup: bool
    num_tokens: int
    deadline: float
    depth: int


@dataclass(frozen=True)
class _PendingCompletion:
    sequence: int
    started: float
    startup: bool
    num_tokens: int
    deadline: float
    event: Any


class PleProgressMonitor:
    """Bound compilation/submission separately from ordered GPU completion.

    Only a first-seen shape or explicit startup gets the host compile budget.
    Warm serving submissions use the serving deadline. Once the release event
    is recorded, the GPU gets its own budget, independent of host compilation.
    Neither phase can wait forever.
    """

    def __init__(
        self,
        *,
        rank: int,
        layer_names: tuple[str, ...],
        serving_deadline_s: float,
        startup_deadline_s: float,
    ) -> None:
        serving_deadline_s = _validate_deadline(
            serving_deadline_s, "serving_deadline_s"
        )
        startup_deadline_s = _validate_deadline(
            startup_deadline_s, "startup_deadline_s"
        )
        self._rank = rank
        self._layer = ",".join(layer_names)
        self._serving_deadline_s = serving_deadline_s
        self._startup_deadline_s = startup_deadline_s
        self._condition = threading.Condition()
        self._submission: _Submission | None = None
        self._pending: deque[_PendingCompletion] = deque()
        self._seen_num_tokens: set[int] = set()
        self._sequence = 0
        self._stopping = False
        self._thread = threading.Thread(
            target=self._run,
            name=f"ple-progress-rank{rank}",
            daemon=True,
        )
        self._thread.start()

    @classmethod
    def from_environment(
        cls, *, rank: int, layer_names: tuple[str, ...]
    ) -> "PleProgressMonitor":
        return cls(
            rank=rank,
            layer_names=layer_names,
            serving_deadline_s=_environment_deadline(
                "VLLM_PLE_PROGRESS_DEADLINE_S",
                _DEFAULT_SERVING_DEADLINE_S,
            ),
            startup_deadline_s=_DEFAULT_STARTUP_DEADLINE_S,
        )

    @property
    def sequence(self) -> int:
        with self._condition:
            return self._sequence

    @property
    def has_pending_completions(self) -> bool:
        with self._condition:
            return bool(self._pending)

    def describe_submission(self, sequence: int) -> str:
        """Return the armed transaction descriptor for the active sequence."""
        with self._condition:
            active = self._submission
            if active is None or active.sequence != sequence:
                raise RuntimeError("PLE fault correlation has no active submission")
            return f"sequence:{active.sequence}@begin:{active.started:.9f}"

    def completion_is_pending(self, sequence: int) -> bool:
        """Whether this serving sequence has a retained, incomplete event.

        This method is called only behind the armed fault guard. An event that
        has completed on the GPU but has not yet been dequeued is not counted.
        """
        with self._condition:
            pending = next(
                (
                    item
                    for item in self._pending
                    if item.sequence == sequence and not item.startup
                ),
                None,
            )
        if pending is None:
            return False
        return not bool(pending.event.query())

    def begin(self, *, startup: bool, num_tokens: int) -> int:
        now = time.monotonic()
        with self._condition:
            active = self._submission
            if active is not None:
                if not startup or not active.startup:
                    raise RuntimeError(
                        "Overlapping PLE host submissions violate the serialized "
                        "single-slot protocol"
                    )
                sequence = active.sequence
                self._submission = _Submission(
                    sequence=sequence,
                    started=active.started,
                    startup=True,
                    num_tokens=num_tokens,
                    deadline=active.deadline,
                    depth=active.depth + 1,
                )
            else:
                # begin_forward receives num_tokens_after_padding: the eager
                # input size, or the FULL graph's validated captured copy size
                # (_prepare_ple_forward checks that equality). Keep exact sizes:
                # rounding into our own bucket could hide a cold specialization.
                # Model/layer/dtype are fixed for this connector; request count
                # is transport metadata, not the model's padded token dimension.
                budget = (
                    self._startup_deadline_s
                    if startup or num_tokens not in self._seen_num_tokens
                    else self._serving_deadline_s
                )
                self._sequence += 1
                sequence = self._sequence
                self._submission = _Submission(
                    sequence=sequence,
                    started=now,
                    startup=startup,
                    num_tokens=num_tokens,
                    deadline=now + budget,
                    depth=1,
                )
            # Admission consumes the one cold allowance (including dummy/capture
            # shapes). Host submissions are serialized and failures are fatal;
            # a later serving begin of this size follows its prior host submit.
            # GPU completion is tracked separately and need not have finished.
            self._seen_num_tokens.add(num_tokens)
            self._condition.notify_all()
            return sequence

    def track_completion(self, event: Any) -> None:
        """Retain a release event until the monitor observes GPU completion."""
        if not callable(getattr(event, "query", None)):
            raise TypeError("PLE completion event must provide query()")
        with self._condition:
            active = self._submission
            if active is None:
                raise RuntimeError("PLE completion has no matching submission")
            self._pending.append(
                _PendingCompletion(
                    sequence=active.sequence,
                    started=active.started,
                    startup=active.startup,
                    num_tokens=active.num_tokens,
                    deadline=time.monotonic() + (
                        self._startup_deadline_s
                        if active.startup
                        else self._serving_deadline_s
                    ),
                    event=event,
                )
            )
            self._condition.notify_all()

    def complete(self) -> None:
        with self._condition:
            active = self._submission
            if active is None or active.depth <= 0:
                raise RuntimeError("PLE release has no matching transaction")
            if active.depth == 1:
                self._submission = None
            else:
                self._submission = _Submission(
                    sequence=active.sequence,
                    started=active.started,
                    startup=active.startup,
                    num_tokens=active.num_tokens,
                    deadline=active.deadline,
                    depth=active.depth - 1,
                )
            self._condition.notify_all()

    def close(self) -> None:
        with self._condition:
            self._stopping = True
            self._condition.notify_all()
        self._thread.join(timeout=1)

    def _fail_if_running(self, **diagnostic: Any) -> None:
        # Serialize failure with close(): a teardown query must not kill the
        # worker after close has disarmed the monitor.
        with self._condition:
            if not self._stopping:
                ple_fail_closed(**diagnostic)

    def _run(self) -> None:
        try:
            self._monitor()
        except BaseException as error:
            with self._condition:
                if self._stopping:
                    return
            self._fail_if_running(
                where="progress-monitor-thread",
                rank=self._rank,
                layer=self._layer,
                sequence=self._sequence,
                error=error,
            )

    def _monitor(self) -> None:
        while True:
            with self._condition:
                if self._stopping:
                    return
                submission = self._submission
                pending = self._pending[0] if self._pending else None
                if submission is None and pending is None:
                    self._condition.wait(timeout=_POLL_INTERVAL_S)
                    continue

            # CUDA completion is observed only here. Never synchronize and never
            # use the host flag as evidence that the queued reset completed.
            if pending is not None and pending.event.query():
                with self._condition:
                    if self._pending and self._pending[0] is pending:
                        self._pending.popleft()
                        self._condition.notify_all()
                continue

            now = time.monotonic()
            if pending is not None and now >= pending.deadline:
                with self._condition:
                    if self._stopping:
                        return
                    if not self._pending or self._pending[0] is not pending:
                        continue
                mode = "startup" if pending.startup else "serving"
                self._fail_if_running(
                    where=(
                        f"progress-deadline:{mode}:gpu-completion:"
                        f"tokens={pending.num_tokens}"
                    ),
                    rank=self._rank,
                    layer=self._layer,
                    sequence=pending.sequence,
                    error=TimeoutError("PLE GPU completion made no bounded progress"),
                )

            if submission is not None and now >= submission.deadline:
                with self._condition:
                    if self._stopping:
                        return
                    if self._submission is not submission:
                        continue
                mode = "startup" if submission.startup else "serving"
                self._fail_if_running(
                    where=(
                        f"progress-deadline:{mode}:host-submission:"
                        f"tokens={submission.num_tokens}"
                    ),
                    rank=self._rank,
                    layer=self._layer,
                    sequence=submission.sequence,
                    error=TimeoutError("PLE host submission made no bounded progress"),
                )

            deadlines = []
            if pending is not None:
                deadlines.append(pending.deadline)
            if submission is not None:
                deadlines.append(submission.deadline)
            timeout = _POLL_INTERVAL_S
            if deadlines:
                timeout = min(
                    timeout,
                    max(0.0, min(deadlines) - time.monotonic()),
                )
            with self._condition:
                if self._stopping:
                    return
                if (
                    self._submission is not submission
                    or (self._pending[0] if self._pending else None) is not pending
                ):
                    continue
                self._condition.wait(timeout=timeout)
