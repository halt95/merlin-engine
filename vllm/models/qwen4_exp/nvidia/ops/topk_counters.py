# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Python reader for the medium-path exact-restart top-k counters.

The counters themselves are device-global uint64s owned by
`csrc/merlin/topk/topk.cu`; `torch.ops._C.merlin_persistent_topk_exact_counters`
snapshots them into an **int64** CUDA tensor. Without this module nothing in a
running engine ever calls that op, so the instrument records into a buffer no
one reads -- which measures nothing.

DTYPE IS PART OF THE PROTOCOL (large-path determinism change). Version 1 was u32
device counters read into a signed int32 tensor, which goes negative after
~33k all-equal rows at k=512 and wraps after ~66k, so a delta taken across the
boundary is a wrong number that looks like a measurement. Version 2 widens
both sides, and the op REJECTS an int32 buffer rather than filling it from half
of each counter. An older build whose op still demands int32 will therefore
raise here; that is the intended signal, not a regression to paper over.

SLOT LAYOUT is build-dependent, so nothing here hard-codes it. `slot_count()`
asks the build (the `..._slots` op); a build without that op has no readable
counters here. The op demands EXACTLY that many elements, so a guessed length
is an error, not a short read; names come from a table keyed by that length:

  5  rows overflow fallbacks faults switch
  10  rows overflow fallbacks faults rows_large ties_cut ties_dropped
             faults_large switch switch_large
  11  the ten above, plus decode_overflow

An unknown length degrades to `slot<i>` names rather than mislabelling.

WHY 11 AND NOT 10. The decode counter is numbered around the large-path block
instead of after the medium-path one, so nothing is renumbered: slots 4-7 are
the large-path counters, 8-9 the two switch flags, 10 the decode overflow.
Every one of the eleven is live on this build; on a build that
drives only some of them the rest are structurally zero and the line still
reports them by name -- they are the slots they will hold, not a
mislabelling. `decode_overflow` is the number that has to be read before any
claim that the DECODE path is deterministic: it counts rows whose buffered
candidates exceeded the 3,708-slot capacity before the tie ranking ran, which
ranking cannot repair.

HOW TO READ IT. Set `MERLIN_TOPK_EXACT_COUNTERS_LOG=<seconds>` on the
engine; each GPU worker then emits, at most that often and once more at
shutdown, one line into the ordinary engine log:

    topk-exact-counters rank=PP0_TP1 rows=... overflow=... fallbacks=...
    faults=... switch=... [rows_large=... ... decode_overflow=...]

The first five fields are the medium-path line and keep their order on every build;
any further slot is appended after them, so an 11-slot build reads as the medium-path
line plus the four large-path fields, `switch_large` and `decode_overflow`.

Counts are cumulative per worker process, so take the LATEST line
per rank -- `grep topk-exact-counters <engine log> | ...` keyed on the `rank=`
field -- and reads `faults` as the pass/fail number (it must stay 0) with
`rows` as the denominator. Earlier lines are only useful as a rate.

COST. `read()` passes `sync=1`, i.e. a full device synchronize, because a
stream-ordered snapshot can miss increments from work in flight on another
stream. That is acceptable ONLY because the cadence is coarse (>= 30 s in
practice): the time check happens before the op call, so a step that is not
due for a log touches nothing but `time.monotonic()`. The read is also refused
inside a cudagraph capture -- the op allocates and synchronizes, which would
abort the capture.
"""

import time

import torch

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)

_OP_NAME = "merlin_persistent_topk_exact_counters"

_MEDIUM_NAMES = ("rows", "overflow", "fallbacks", "faults", "switch")
_LARGEDET_NAMES = (
    "rows",
    "overflow",
    "fallbacks",
    "faults",
    "rows_large",
    "ties_cut",
    "ties_dropped",
    "faults_large",
    "switch",
    "switch_large",
)
# The decode-overflow counter comes after the large-path block; see the module
# docstring for why it is slot 10 and not slot 4.
_TIEDET_NAMES = _LARGEDET_NAMES + ("decode_overflow",)
_SLOT_NAMES: dict[int, tuple[str, ...]] = {
    5: _MEDIUM_NAMES,
    10: _LARGEDET_NAMES,
    11: _TIEDET_NAMES,
}

# The medium-path line contract. Any further name is appended after these, in slot
# order, so a 10-slot build still starts with exactly the medium-path line.
_LINE_ORDER = _MEDIUM_NAMES

_warned_absent = False
_warned_failed = False
_last_log_monotonic = 0.0


def _get_op(name: str):
    try:
        return getattr(torch.ops._C, name, None)
    except (AttributeError, RuntimeError):
        return None


def slot_count() -> int:
    """How many int64 slots this build's counter array carries.

    Asked of the build, never guessed: a length that does not match the
    extension is rejected by the op's own length check. (The
    MERLIN_TOPK_COUNTER_SLOTS fallback for builds older than the slot-count op
    was pruned; the merlin_* ops always come with it.)
    """
    query = _get_op(_OP_NAME + "_slots")
    if query is None:
        raise RuntimeError(f"torch.ops._C.{_OP_NAME}_slots is not in this build")
    return int(query())


def slot_names(count: int) -> tuple[str, ...]:
    names = _SLOT_NAMES.get(count)
    return names if names is not None else tuple(f"slot{i}" for i in range(count))


def read(reset: bool = False, sync: bool = True) -> dict[str, int]:
    """Snapshot the counters as {slot name: value}; {} when unreadable."""
    global _warned_absent
    op = _get_op(_OP_NAME)
    if op is None:
        if not _warned_absent:
            _warned_absent = True
            logger.warning(
                "torch.ops._C.%s is not in this build; the top-k exact-restart "
                "counters cannot be read.",
                _OP_NAME,
            )
        return {}
    if reset and not sync:
        raise ValueError("reset requires sync: zeroing races in-flight atomics")
    if not (torch.cuda.is_available() and torch.cuda.is_initialized()):
        return {}
    if torch.cuda.is_current_stream_capturing():
        # The op allocates and synchronizes; calling it here aborts the capture.
        return {}
    count = slot_count()
    device = torch.device("cuda", torch.cuda.current_device())
    # int64: the device counters are 64-bit and the op refuses any other
    # width. See the module docstring -- this is protocol version 2.
    out = torch.zeros(count, dtype=torch.int64, device=device)
    op(out, 1 if reset else 0, 1 if sync else 0)
    return dict(zip(slot_names(count), out.tolist(), strict=True))


def rank_tag() -> str:
    from vllm.distributed.parallel_state import get_pp_group, get_tp_group

    try:
        return f"PP{get_pp_group().rank_in_group}_TP{get_tp_group().rank_in_group}"
    except Exception:
        # Pre-init or single-process: say so rather than claim rank 0.
        return "PP?_TP?"


def log_line(tag: str) -> str:
    """The one-line form to grep; "" when the counters are unreadable."""
    counters = read()
    if not counters:
        return ""
    order = [name for name in _LINE_ORDER if name in counters]
    order += [name for name in counters if name not in _LINE_ORDER]
    fields = " ".join(f"{name}={counters[name]}" for name in order)
    return f"topk-exact-counters rank={tag} {fields}"


def maybe_log(force: bool = False) -> str:
    """Emit the line if due (or forced). Time check first: when not due this
    costs one clock read and never touches the device."""
    global _last_log_monotonic, _warned_failed
    period = envs.MERLIN_TOPK_EXACT_COUNTERS_LOG
    if period <= 0:
        return ""
    now = time.monotonic()
    if not force and now - _last_log_monotonic < period:
        return ""
    _last_log_monotonic = now
    try:
        line = log_line(rank_tag())
    except Exception as exc:  # never let an instrument break a step or shutdown
        if not _warned_failed:
            _warned_failed = True
            logger.warning("top-k exact-restart counter read failed: %s", exc)
        return ""
    if line:
        logger.info("%s", line)
    return line
