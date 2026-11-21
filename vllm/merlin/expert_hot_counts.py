# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Opt-in routed-expert hit counter for building a cold-expert hot profile.

Default OFF: ``MERLIN_EXPERT_HOT_COUNTS`` unset, blank or ``0``.
:func:`attach` then returns without binding anything, so the router's existing
``capture_fn is not None`` test stays False and the forward is unchanged.

ON (``MERLIN_EXPERT_HOT_COUNTS=<dir>``): every MoE runner whose router has the
upstream ``BaseRouter.set_capture_fn`` hook (called right after top-k, on the
logical expert ids, before EPLB mapping) gets a counter. Per call it enqueues a
few device ops into tensors allocated once at load:

* a call with ``M <= MERLIN_EXPERT_HOT_MMAX`` rows (decode or spec-verify) adds
  each routed (token, k) slot to ``slot[E]`` and the call's distinct routed set
  to ``step[E]`` (the distinct-fetch basis: what a cold read moves);
* a larger call (a prefill chunk) adds its slots to ``prefill[E]`` only.

No host sync, no host read of a device value and no allocation outside the
graph pool, so the ops are captured into CUDA graphs and every replay counts.
The ``M`` branch is taken on the Python shape at capture time, which is the
graph's (padded) size. A padded replay counts its padding rows too, so run the
calibration at c1, where a spec-verify step fills its graph exactly.

Dumps: a host thread polls ``<dir>/ctl/req-<seq>-<name>`` (every
``MERLIN_EXPERT_HOT_POLL_S`` seconds, default 0.5, 0 = no thread). The driver
writes one only while the server is idle. The thread copies the counters on a
side stream after an event on the main stream, writes
``<dir>/seg-<seq>-<name>-rank<r>-pid<p>.json``, zeroes the counters (each segment
is a delta) and writes ``<dir>/ctl/ack-<seq>-rank<r>-pid<p>``. A last segment
``exit`` is written at interpreter exit. ``tools/merlin/cold_expert_profile.py
--counts-dir <dir>`` turns the segments into a ``merlin-cold-experts/1``
profile.
"""

from __future__ import annotations

import atexit
import json
import logging
import math
import os
import re
import threading
import time

import torch

logger = logging.getLogger("vllm.merlin.expert_hot_counts")

FORMAT = "merlin-expert-hot-counts/1"
_REQ = re.compile(r"^req-(\d+)-([A-Za-z0-9_.:-]+)$")
_LAYER = re.compile(r"\.(\d+)\.")

_STATE: dict = {"counters": [], "dir": None, "thread": None, "atexit": False,
                "lock": threading.Lock(), "done": set()}


def counts_dir(environ=None) -> str | None:
    raw = (os.environ if environ is None else environ).get("MERLIN_EXPERT_HOT_COUNTS", "")
    raw = (raw or "").strip()
    return None if raw in ("", "0") else raw


def _int_env(environ, name: str, default: int, low: int) -> int:
    raw = (environ.get(name, "") or "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{name}={raw!r} is not an integer") from None
    if value < low:
        raise ValueError(f"{name}={value} is below {low}")
    return value


def _float_env(environ, name: str, default: float) -> float:
    raw = (environ.get(name, "") or "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"{name}={raw!r} is not a number") from None
    if not math.isfinite(value):  # inf / nan refused
        raise ValueError(f"{name}={raw!r} is not a finite number")
    if value < 0:
        raise ValueError(f"{name}={value} is negative")
    return value


def _layer_index(name: str) -> int | None:
    found = _LAYER.findall("." + name + ".")
    return int(found[-1]) if found else None


class Counter:
    """Device-side counts for one MoE layer; called by the router with topk_ids."""

    def __init__(self, layer: int, layer_name: str, model: str, experts: int, topk: int,
                 local_ids: list[int], mmax: int, device, window: int = 0,
                 windows: int = 64) -> None:
        self.layer, self.layer_name, self.model = layer, layer_name, model
        self.experts, self.topk, self.local_ids, self.mmax = experts, topk, local_ids, mmax
        self.device = torch.device(device)
        z = lambda n: torch.zeros(n, dtype=torch.int64, device=self.device)  # noqa: E731
        self.slot, self.step, self.prefill, self.mark = z(experts), z(experts), z(experts), z(experts)
        # decode calls, decode rows, prefill rows (rows at the graph's own size)
        self.meta = z(3)
        # Step windows inside a segment: decode call
        # c (counted from the segment start) adds its distinct set to row
        # min(c // window, windows - 1) of win (the last row absorbs any overflow).
        self.window, self.windows = window, windows
        self.win = (torch.zeros(windows * experts, dtype=torch.int32, device=self.device)
                    if window else None)
        self.span = torch.arange(experts, dtype=torch.int64, device=self.device) if window else None

    def __call__(self, topk_ids: torch.Tensor) -> None:
        rows = topk_ids.shape[0]
        ids = topk_ids.reshape(-1).to(torch.int64)
        live = (ids >= 0) & (ids < self.experts)  # a -1 (no expert) slot counts nothing
        ids = torch.where(live, ids, torch.zeros_like(ids))
        ones = live.to(torch.int64)
        if rows <= self.mmax:
            self.slot.index_add_(0, ids, ones)
            self.mark.zero_()
            self.mark.index_add_(0, ids, ones)
            self.step.add_(self.mark.clamp_(max=1))
            if self.win is not None:  # before the call count moves
                row = torch.div(self.meta[0:1], self.window, rounding_mode="floor").clamp_(
                    max=self.windows - 1)
                self.win.index_add_(0, row * self.experts + self.span, self.mark.to(torch.int32))
            self.meta[0:1].add_(1)
            self.meta[1:2].add_(rows)
        else:
            self.prefill.index_add_(0, ids, ones)
            self.meta[2:3].add_(rows)

    def tensors(self):
        if self.win is not None:
            return (self.slot, self.step, self.prefill, self.meta, self.win)
        return (self.slot, self.step, self.prefill, self.meta)


def _local_ids(module, experts: int) -> list[int]:
    manager = getattr(module, "expert_map_manager", None)
    expert_map = getattr(manager, "expert_map", None) if manager is not None else None
    if expert_map is None:
        expert_map = getattr(module, "expert_map", None)
    if expert_map is None:
        return list(range(experts))
    return [e for e, v in enumerate(expert_map.detach().to("cpu").tolist()) if v >= 0]


def attach(model: torch.nn.Module, environ=None) -> int:
    """Bind a counter to every MoE router of ``model`` when the knob is on.
    Returns the number bound (0 when off)."""
    environ = os.environ if environ is None else environ
    root = counts_dir(environ)
    if root is None:
        return 0
    mmax = _int_env(environ, "MERLIN_EXPERT_HOT_MMAX", 32, 1)
    # Per-segment step windows (0 = off)
    window = _int_env(environ, "MERLIN_EXPERT_HOT_WINDOW_STEPS", 0, 0)
    windows = _int_env(environ, "MERLIN_EXPERT_HOT_WINDOWS", 64, 1)
    poll = _float_env(environ, "MERLIN_EXPERT_HOT_POLL_S", 0.5)
    tag = type(model).__name__
    bound = 0
    for name, mod in model.named_modules():
        router = getattr(mod, "router", None)
        if router is None or not hasattr(router, "set_capture_fn"):
            continue
        layer_name = getattr(mod, "layer_name", None) or name
        layer = _layer_index(layer_name)
        if layer is None:
            continue
        if getattr(router, "capture_fn", None) is not None:
            raise RuntimeError(
                f"MERLIN_EXPERT_HOT_COUNTS: {layer_name} already has a routing capture "
                "(routed-experts return or another probe); run one at a time")
        experts = int(getattr(router, "global_num_experts"))
        topk = int(getattr(router, "top_k"))
        params = [p for p in mod.parameters(recurse=True)]
        device = params[0].device if params else torch.device("cpu")
        counter = Counter(layer, layer_name, tag, experts, topk, _local_ids(mod, experts),
                          mmax, device, window, windows)
        router.set_capture_fn(counter)
        _STATE["counters"].append(counter)
        bound += 1
    if not bound:
        logger.info("expert hot counts: no MoE router in %s", tag)
        return 0
    _STATE["dir"] = root
    os.makedirs(os.path.join(root, "ctl"), exist_ok=True)
    if not _STATE["atexit"]:
        atexit.register(_at_exit)
        _STATE["atexit"] = True
    if poll > 0 and _STATE["thread"] is None:
        t = threading.Thread(target=_poll, args=(root, poll), name="merlin-expert-hot",
                             daemon=True)
        _STATE["thread"] = t
        t.start()
    logger.warning(
        "MERLIN_EXPERT_HOT_COUNTS: counting routed experts on %d %s layers into %s "
        "(calibration only: never on a timed boot)", bound, tag, root)
    return bound


def _rank() -> dict:
    out = {"rank": 0, "pp": 0, "tp": 0}
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            out["rank"] = torch.distributed.get_rank()
        from vllm.distributed.parallel_state import get_pp_group, get_tp_group

        out["pp"] = get_pp_group().rank_in_group
        out["tp"] = get_tp_group().rank_in_group
    except Exception:  # noqa: BLE001 - tests and single-process runs
        pass
    return out


def _snapshot(reset: bool) -> list[dict]:
    """Host copies of every counter (then zeroed when ``reset``)."""
    counters = list(_STATE["counters"])
    cuda = [c for c in counters if c.device.type == "cuda"]
    side = None
    if cuda:
        torch.cuda.set_device(cuda[0].device)
        main = torch.cuda.default_stream(cuda[0].device)
        side = torch.cuda.Stream(cuda[0].device)
        event = torch.cuda.Event()
        event.record(main)
        side.wait_event(event)
    rows = []
    ctx = torch.cuda.stream(side) if side is not None else _null()
    with ctx:
        # copy=True: .to("cpu") of a CPU counter is the
        # counter itself, so the reset below zeroed the "copy" before it was written
        host = [[t.to("cpu", non_blocking=False, copy=True) for t in c.tensors()]
                for c in counters]
        if reset:
            for c in counters:
                for t in c.tensors():
                    t.zero_()
    if side is not None:
        side.synchronize()
    for c, (slot, step, prefill, meta, *win) in zip(counters, host):
        m = meta.tolist()
        row = {
            "layer": c.layer, "layer_name": c.layer_name, "model": c.model,
            "experts": c.experts, "topk": c.topk, "local": c.local_ids, "mmax": c.mmax,
            "decode_calls": m[0], "decode_rows": m[1], "prefill_rows": m[2],
            "slot": slot.tolist(), "step": step.tolist(), "prefill": prefill.tolist(),
        }
        if win:  # only the windows this segment filled
            used = min(-(-m[0] // c.window), c.windows)
            grid = win[0].view(c.windows, c.experts)[:used]
            row.update(window_steps=c.window, windows_overflow=m[0] > c.window * c.windows,
                       windows=grid.tolist())
        rows.append(row)
    return rows


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def dump_segment(seq: int, name: str, reset: bool = True) -> str:
    """Write one segment file (and its ack); returns the segment path."""
    root = _STATE["dir"]
    if root is None:
        raise RuntimeError("expert hot counts are not attached")
    with _STATE["lock"]:
        who = _rank()
        pid = os.getpid()
        body = {"format": FORMAT, "segment": name, "seq": seq, "pid": pid,
                "written": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **who,
                "layers": _snapshot(reset)}
        path = os.path.join(root, f"seg-{seq:04d}-{name}-rank{who['rank']}-pid{pid}.json")
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            json.dump(body, f, separators=(",", ":"))
            f.write("\n")
        os.replace(tmp, path)
        ack = os.path.join(root, "ctl", f"ack-{seq:04d}-rank{who['rank']}-pid{pid}")
        with open(ack, "w", encoding="utf-8", newline="\n") as f:
            f.write(os.path.basename(path) + "\n")
        return path


def poll_once(root: str) -> list[str]:
    """Serve every pending request in ``<root>/ctl`` once (oldest first)."""
    written = []
    try:
        names = sorted(os.listdir(os.path.join(root, "ctl")))
    except OSError:
        return written
    for n in names:
        m = _REQ.match(n)
        if not m or n in _STATE["done"]:
            continue
        # Marked served only once the dump is on disk,
        # so a failed write is retried on the next poll
        written.append(dump_segment(int(m.group(1)), m.group(2)))
        _STATE["done"].add(n)
    return written


def _poll(root: str, every: float) -> None:
    while True:
        try:
            poll_once(root)
        except Exception as e:  # noqa: BLE001 - never take the engine down
            logger.warning("expert hot counts: dump failed: %s", e)
        time.sleep(every)


def _at_exit() -> None:
    if _STATE["dir"] is None or not _STATE["counters"]:
        return
    try:
        dump_segment(9999, "exit", reset=False)
    except Exception as e:  # noqa: BLE001
        logger.warning("expert hot counts: exit dump failed: %s", e)
