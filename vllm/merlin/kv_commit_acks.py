# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Content-writer retirement for the unified KV tier.

Step numbers are scheduler-issued and never reset. A retired step includes
masked dispositions; retirement alone does not certify a recurrent row.
Callers must retain the actual writer's (request, step) on the content
generation, including across CoW, rather than stamp its final releaser.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any


def _record_event() -> Any:
    import torch

    event = torch.Event()
    event.record(torch.cuda.current_stream())
    return event


class WorkerCommitAcks:
    """A contiguous frontier of GPU-complete forwards and deferred writers."""

    def __init__(self, record_event: Callable[[], Any] = _record_event):
        self.committed_step = 0
        self.current_step = 0
        self._record_event = record_event
        self._events: dict[int, Any | None] = {}
        self._deferred: set[int] = set()
        self._masked: set[tuple[str, int]] = set()
        # Whether the runner's current execute began a scheduler step. An
        # execute without one (warm-up, profiling: no connector metadata)
        # writes no tracked content and must not defer under the previous
        # step's number: a second writer for one step retires twice.
        self._active = False
        # A non-last PP rank decides in sample_tokens, after the forward's
        # finish_forward and before the next step, whether the step's writer
        # is deferred; until then the frontier may not pass it.
        self._undecided: int | None = None

    def begin_step(self, step: int, acked: int = 0, *, hold: bool = False) -> None:
        if step != self.current_step + 1:
            raise RuntimeError("KV commit steps must be consecutive")
        self.current_step = step
        self._events[step] = None
        self._active = True
        self._undecided = step if hold else None
        # The scheduler has consumed every rank's dispositions through acked.
        self._masked = {row for row in self._masked if row[1] > acked}

    def skip_step(self) -> None:
        """The runner's execute carries no scheduler step (warm-up, profiling,
        no connector metadata): nothing it runs is tracked or deferred."""
        self._active = False
        self._undecided = None

    def defer(self, req_ids: Iterable[str]) -> DeferredCommitAck | None:
        """The current step's writer retires with its deferred postprocess
        (exactly one per step), not with its forward. None when the execute
        began no step."""
        if not self._active:
            return None
        step = self.current_step
        if step in self._deferred or step <= self.committed_step:
            raise RuntimeError("KV deferred writer registered twice")
        self._deferred.add(step)
        self._events[step] = None  # the forward's event no longer retires it
        self._undecided = None
        return DeferredCommitAck(self, step, tuple(req_ids))

    def not_deferred(self) -> None:
        """The current step has no deferred writer: its forward retires it."""
        self._undecided = None

    def finish_forward(self) -> None:
        step = self.current_step
        if (
            self._active
            and step in self._events
            and step not in self._deferred
            and self._events[step] is None
        ):
            self._events[step] = self._record_event()

    def finish_deferred(self, step: int) -> None:
        if step not in self._deferred:
            raise RuntimeError("KV deferred writer retired twice")
        self._deferred.remove(step)
        self._events[step] = self._record_event()

    def report(self) -> tuple[int, set[tuple[str, int]]]:
        while self.committed_step < self.current_step:
            step = self.committed_step + 1
            if step == self._undecided:
                break  # its writer may yet be deferred (non-last PP rank)
            event = self._events[step]
            if event is None or not event.query():
                break
            del self._events[step]
            self.committed_step = step
        # Repeat failures until the scheduler acknowledges them. A later
        # frontier report must not overtake an earlier masked-row report.
        return self.committed_step, set(self._masked)


@dataclass
class DeferredCommitAck:
    tracker: WorkerCommitAcks
    step: int
    req_ids: tuple[str, ...]

    def mask(self, excluded: Iterable[bool]) -> None:
        self.tracker._masked.update(
            (req_id, self.step)
            for req_id, masked in zip(self.req_ids, excluded)
            if masked
        )

    def finish(self) -> None:
        # Called after postprocess enqueues every backend's deferred commit,
        # or after all rows were excluded (there is no deferred writer).
        self.tracker.finish_deferred(self.step)


class SchedulerCommitAcks:
    """Distinct-rank retirement and failed recurrent producer identities.

    The allocator/union lookup should use is_ready for each content writer.
    Keep failed identities until their content generations are invalidated.
    Step numbers, rank frontiers and failures survive cache reset; this makes
    delayed reports harmless without depending on the transfer epoch.
    """

    def __init__(self, world_size: int):
        self._steps = [0] * world_size
        self.masked_rows: set[tuple[str, int]] = set()

    @property
    def acked(self) -> int:
        return min(self._steps)

    def update(
        self, committed_steps: dict[int, int], masked_rows: set[tuple[str, int]]
    ) -> None:
        if any(rank < 0 or rank >= len(self._steps) for rank in committed_steps):
            raise ValueError("KV commit ack from an unknown rank")
        # Dispositions are installed before advancing any readiness frontier.
        self.masked_rows.update(masked_rows)
        for rank, step in committed_steps.items():
            self._steps[rank] = max(self._steps[rank], step)

    def is_ready(self, req_id: str, step: int) -> bool:
        return 0 < step <= self.acked and (req_id, step) not in self.masked_rows

    def prune(self, live) -> int:
        """Drop masked rows nothing can ask about any more: their step
        is at or below the min-rank frontier (a rank only re-reports rows above
        its own frontier, which is at least this one) and ``live(row)`` says
        no GPU content or held hand-off still carries that producer identity.
        Returns the rows dropped."""
        acked = self.acked
        gone = {r for r in self.masked_rows if r[1] <= acked and not live(r)}
        self.masked_rows -= gone
        return len(gone)


def _worker(runner: Any, *, disabled_ok: bool = False) -> Any:
    """The tier worker whose tracker the runner reports to. None while the
    runner's connector is disabled (the scheduler warm-up): the warm-up's
    execute/sample pairs write no tracked content."""
    wrapper = getattr(runner, "kv_connector", None)
    if not disabled_ok and getattr(wrapper, "_disabled", False):
        return None
    connector = getattr(wrapper, "kv_connector", None)
    worker = getattr(connector, "worker_handler", None)
    return worker if getattr(worker, "commit_acks", None) is not None else None


def begin_step(runner: Any, scheduler_output: Any) -> None:
    worker = _worker(runner, disabled_ok=True)
    if worker is None:
        return
    metadata = getattr(scheduler_output, "kv_connector_metadata", None)
    if (
        getattr(runner.kv_connector, "_disabled", False)
        or metadata is None
        or not metadata.scheduled_step
    ):
        worker.commit_acks.skip_step()
        return
    pp = getattr(runner, "pp_handler", None)
    # A non-last PP rank whose execute runs a forward reaches sample_tokens,
    # which decides whether the step's writer is deferred (defer_step).
    hold = (
        pp is not None
        and not getattr(pp, "is_last_rank", True)
        and getattr(scheduler_output, "total_num_scheduled_tokens", 0) > 0
    )
    worker.commit_acks.begin_step(
        metadata.scheduled_step, metadata.acked_step, hold=hold
    )


def defer_step(runner: Any, input_batch: Any) -> None:
    worker = _worker(runner)
    if worker is None:
        return
    slot = runner.pp_handler.queue[-1]
    if slot is None:
        worker.commit_acks.not_deferred()
        return
    slot.commit_ack = worker.commit_acks.defer(input_batch.req_ids)
