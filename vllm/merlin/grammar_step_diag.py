# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Per-step grammar timing for structured-output requests (diagnostic, default off).

``VLLM_MERLIN_GRAMMAR_STEP_DIAG=1`` writes one NDJSON line per engine step that has
structured-output work, to ``<tempdir>/merlin-grammar-step-<pid>.ndjson``:

* engine core: the wall time of each request's ``accept_tokens`` after sampling
  (``accept``), of each request's draft trim (``trim``), and of every mask row the
  scheduler fills (``fill``: request, row, ms), plus five timestamps of the step:
  ``exec_issued`` (execute_model sent), ``prev_result`` (the previous step's output
  back), ``bitmask_start`` / ``bitmask_end`` (this step's masks) and ``sample_issued``
  (sample_tokens sent);
* worker (last pipeline stage): ``gpu_gap_ms``, the GPU time between the end of the
  forward pass and the start of sampling, measured with CUDA events. A gap close to
  the mask time means the forward pass hides none of the mask work.

Timestamps are ``time.time()`` seconds; durations are milliseconds. Nothing is
recorded with the switch off.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from collections import deque
from typing import Any

from vllm import envs

_on: bool | None = None
_lock = threading.Lock()
_record: dict[str, Any] = {}
_file = None
_worker: deque = deque()
_worker_step = 0


def on() -> bool:
    global _on
    if _on is None:
        _on = bool(envs.VLLM_MERLIN_GRAMMAR_STEP_DIAG)
    return _on


def _out():
    global _file
    if _file is None:
        path = os.path.join(
            tempfile.gettempdir(), f"merlin-grammar-step-{os.getpid()}.ndjson"
        )
        _file = open(path, "a", buffering=1, encoding="utf-8")  # noqa: SIM115
    return _file


def mark(name: str) -> None:
    """A step timestamp (engine core)."""
    if on():
        _record[name] = time.time()


def add(kind: str, request_id: str, seconds: float, row: int | None = None) -> None:
    """One timed grammar call of a request (engine core)."""
    if on():
        entry = [request_id, round(seconds * 1e3, 4)]
        if row is not None:
            entry.append(row)
        _record.setdefault(kind, []).append(entry)


def flush() -> None:
    """Write the step's line if it had grammar work, and start a new one."""
    global _record
    if not on():
        return
    record, _record = _record, {}
    if any(k in record for k in ("accept", "trim", "fill")):
        _out().write(json.dumps(record) + "\n")


def forward_end() -> None:
    """Worker: the forward pass has been enqueued; mark its end on the stream."""
    if not on():
        return
    import torch

    if _worker and _worker[-1][1] is None:
        _worker.pop()  # a forward pass with no sampling after it
    event = torch.cuda.Event(enable_timing=True)
    event.record()
    _worker.append([event, None, time.time(), None])


def sample_start(structured: bool) -> None:
    """Worker: sampling starts; mark it on the stream and report finished pairs."""
    global _worker_step
    if not on() or not _worker:
        return
    import torch

    entry = _worker[-1]
    if entry[1] is None:
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        entry[1] = event
        entry[3] = time.time()
        entry.append(structured)
    while _worker and _worker[0][1] is not None and _worker[0][1].query():
        fwd, smp, t_fwd, t_smp, flag = _worker.popleft()
        _worker_step += 1
        _out().write(
            json.dumps(
                {
                    "worker_step": _worker_step,
                    "structured": flag,
                    "forward_end": t_fwd,
                    "sample_start": t_smp,
                    "gpu_gap_ms": round(fwd.elapsed_time(smp), 4),
                }
            )
            + "\n"
        )
