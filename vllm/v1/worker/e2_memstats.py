# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Debug-only allocator telemetry, BOUNDARY-ONLY. Off unless
VLLM_E2_MEMSTATS=<ctl poll seconds> and VLLM_E2_MEMSTATS_DIR=<dir>.

WHY THIS IS NOT A SAMPLER. torch.cuda.memory_stats() and
reset_peak_memory_stats() take the per-device caching-allocator mutex and the
Python bindings do not release the GIL, so polling them shares a lock with the
model path and has no place inside a measured region. PyTorch maintains the
allocator high-water marks itself, so reading them once per window is exact.

WHAT THIS THREAD DOES. Between window boundaries it makes NO CUDA calls and
touches NO allocator state: its tick is a plain read of <dir>/window.ctl, which
holds "<seq> <name>". The measurement driver crosses a boundary only while the server
is quiet (no request running). On a new seq the thread performs these
boundary-only allocator reads:
  1. memory_stats()            -> the closing window's exact high-water marks
                                  and cumulative retry/OOM counters
  2. reset_peak_memory_stats() -> immediately after the read, with no file I/O
                                  in the gap
  3. mem_get_info()            -> device free/total AT THAT QUIESCENT INSTANT
  4. host_memory_stats()       -> pinned-host allocator current values, outside
                                  the read-to-reset gap
then writes the "close" record for the old window and the "ack" record for the
new one.

WHAT THIS MEASURES.
  - A row's allocator peak is EXACT for that row's window (not a sampled
    maximum; it cannot miss a transient).
  - Device free/total is observed ONLY at quiescent boundaries. No minimum-free
    figure exists for the inside of a row, and none may be reported.
  - Retry and OOM counters are cumulative and never reset; per-row deltas come
    from consecutive close records.
  - There is NO interval-level peak. A qualifying interval is narrower
    than its row and is identified only afterwards from the scheduler log, so
    the peak attributable to it is the ROW-level high-water mark, which
    over-includes the prefill ramp. Report this loss of resolution
    explicitly.

Records (one JSON per line): kind start | close | ack | error; rank, local,
dev, pid; window, seq; and for close: allocated_bytes.all.current,
reserved_bytes.all.current, alloc_peak_w, res_peak_w, retries, ooms, free,
total, run_alloc_peak, run_res_peak, and a host object containing every
host_memory_stats() key ending in ".current". Records are written with
synchronous file I/O, never inside the read-to-reset gap. Nothing here raises
into the worker.
"""

import json
import os
import threading
import time

import torch

from vllm.merlin.envs import read_registered

# The registry getter is the one reader of the name; this module owns only
# the float parse.
_SEG = float(read_registered("VLLM_E2_MEMSTATS") or 0)
_DIR = os.environ.get("VLLM_E2_MEMSTATS_DIR", "")
_started = []
VERSION = "e2_memstats/5"


def _close_metrics(dev):
    """Read one quiescent boundary; kept separate for CPU-only testing."""
    s = torch.cuda.memory_stats(dev)
    ap = int(s.get("allocated_bytes.all.peak", 0))
    rp = int(s.get("reserved_bytes.all.peak", 0))
    torch.cuda.reset_peak_memory_stats(dev)  # no file I/O in the gap
    free, total = torch.cuda.mem_get_info(dev)
    host_stats = torch.cuda.host_memory_stats()
    host = {
        key: int(value) for key, value in host_stats.items() if key.endswith(".current")
    }
    return {
        "allocated_bytes.all.current": int(s.get("allocated_bytes.all.current", 0)),
        "reserved_bytes.all.current": int(s.get("reserved_bytes.all.current", 0)),
        "alloc_peak_w": ap,
        "res_peak_w": rp,
        "retries": int(s.get("num_alloc_retries", 0)),
        "ooms": int(s.get("num_ooms", 0)),
        "free": int(free),
        "total": int(total),
        "host": host,
    }


def enabled() -> bool:
    return _SEG > 0 and bool(_DIR)


def ensure_started(worker, armed_by: str = "compile_or_warm_up_model") -> None:
    if _started:
        return
    dev = (
        torch.cuda.current_device()
    )  # captured on the worker's own thread: every call below passes it explicitly
    t = threading.Thread(
        target=_loop,
        name="e2-memstats",
        daemon=True,
        args=(
            dev,
            getattr(worker, "rank", -1),
            getattr(worker, "local_rank", -1),
            armed_by,
        ),
    )
    _started.append(t)
    t.start()


def _ctl():
    try:
        with open(os.path.join(_DIR, "window.ctl")) as f:
            seq, name = f.read().split(None, 1)
        return int(seq), name.strip()
    except (OSError, ValueError):
        return None


def _loop(dev, rank, local, armed_by):
    os.makedirs(_DIR, exist_ok=True)
    # noqa SIM115: the handle lives as long as the telemetry daemon thread
    # that writes to it; there is no scope to close it in.
    f = open(  # noqa: SIM115
        os.path.join(_DIR, f"memstats-rank{rank}-pid{os.getpid()}.jsonl"),
        "a",
        buffering=1,
    )
    base = {"rank": rank, "local": local, "dev": dev, "pid": os.getpid()}

    def w(rec):
        f.write(json.dumps({**base, **rec}, separators=(",", ":")) + "\n")

    run_ap = run_rp = 0
    seq, name = _ctl() or (0, "serve")
    torch.cuda.reset_peak_memory_stats(dev)
    w(
        {
            "kind": "start",
            "t": round(time.time(), 4),
            "m": round(time.monotonic(), 4),
            "armed_by": armed_by,
            "window": name,
            "seq": seq,
            "version": VERSION,
            "poll_s": _SEG,
        }
    )
    while True:
        time.sleep(_SEG)
        try:
            c = _ctl()  # file read only: no CUDA, no allocator lock, no GIL-held call
            if c is None or c[0] == seq:
                continue
            # Boundary. The driver only crosses one while the server is
            # quiet, so this is the single point at which this thread
            # touches the allocator at all.
            metrics = _close_metrics(dev)
            ap = metrics["alloc_peak_w"]
            rp = metrics["res_peak_w"]
            run_ap, run_rp = max(run_ap, ap), max(run_rp, rp)
            w(
                {
                    "kind": "close",
                    "t": round(time.time(), 4),
                    "m": round(time.monotonic(), 4),
                    "window": name,
                    "seq": seq,
                    **metrics,
                    "run_alloc_peak": run_ap,
                    "run_res_peak": run_rp,
                }
            )
            seq, name = c
            w(
                {
                    "kind": "ack",
                    "t": round(time.time(), 4),
                    "m": round(time.monotonic(), 4),
                    "window": name,
                    "seq": seq,
                }
            )
        except Exception as e:  # noqa: BLE001 - telemetry must never raise into the worker
            w(
                {
                    "kind": "error",
                    "t": round(time.time(), 4),
                    "m": round(time.monotonic(), 4),
                    "window": name,
                    "seq": seq,
                    "error": f"{type(e).__name__}: {e}"[:300],
                }
            )
