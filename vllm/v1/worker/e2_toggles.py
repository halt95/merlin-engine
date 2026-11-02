# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Per-rank determinism-toggle line (log-only; changes no behaviour).

Why this exists: the effective state of the
four determinism toggles and the three persistent_topk switches must be
attributable to a RANK and a PID. The kernel's own stderr line
(``[merlin] persistent_topk tie policy: ...``, topk.cu report_tie_policy_once)
is printed once per process with no tag, and FULL_K / QSA_SORT have no log
producer at all. Every GPU worker therefore logs exactly one line::

    e2-toggles[PP<p>_TP<t>]: pid=<pid> MERLIN_FULL_K=<raw> MERLIN_QSA_SORT=<raw>
    MERLIN_TIE_DET=<raw> MERLIN_TIE_RECENT=<raw> MERLIN_TOPK_EXACT_MEDIUM=<raw>
    MERLIN_TOPK_EXACT_DECODE=<raw> MERLIN_TOPK_DET_LARGE=<raw>
    MERLIN_TOPK_SORTED_EMIT=<raw>
    effective: full_k=<0|1> qsa_sort=<0|1|invalid> tie_det=<0|1> tie_recent=<0|1>
    tie_recent_effective=<0|1> exact_medium=<0|1> exact_decode=<0|1>
    det_large=<0|1> sorted_emit=<0|1> source=<libc-getenv|os-environ>
    cpp-cache: <full_k=... tie_det=... tie_recent=... tie_recent_effective=...
    exact_medium=... exact_decode=... det_large=... sorted_emit=...
    source=cpp-cache | n-a reason=<why>>
    python: qsa_sort=<0|1|n-a> sorted_emit_skip=<0|1|n-a>
    source=qsa_indexer._QSA_SORT
    device: switch=<0|1|n-a> switch_large=<0|1|n-a> source=counter-slots-8-9

(one line in the log). ``<raw>`` is ``unset``, a plain token, or a double-quoted
string with every byte outside ``[!-~]`` (and ``"`` / ``\\``) hex-escaped, so
the line always splits on whitespace.

WHAT EACH SECTION IS
* ``effective:`` -- what each switch resolves to. The seven C++ switches are
  folded in csrc (topk.cu, marlin_moe_wna16/ops.cu: no ``getenv`` remains), so
  they report the compiled value whatever the environment says; the raw
  section still shows a set variable, which the kernel ignores. QSA_SORT is
  the vllm/merlin/envs.py reader's result (``invalid`` where that reader
  raises). The raw C++ names are read through libc ``getenv``, not
  ``os.environ``, which can diverge from it after an ``os.putenv``. Nothing
  may branch on this section: this module decides nothing.
* ``cpp-cache:`` -- the kernel's switches as the C++ statics cached them, from
  ``torch.ops._C.merlin_persistent_topk_toggle_state`` and
  ``torch.ops._moe_C.marlin_moe_full_k_state`` (query ops that only LOAD a
  mirror the static initialisers publish; they never fill a cache). ``n-a``
  with a reason on a build without those ops --
  older extension builds have none.
* ``python:`` -- ``qsa_indexer._QSA_SORT``, the module constant the QSA sort
  actually branches on (looked up in ``sys.modules``, never imported here;
  ``n-a`` if the module was never loaded in this process), and
  ``sorted_emit_skip`` -- ``qsa_indexer._kernel_sorted_emit_state``, i.e.
  whether the Python sort is being SKIPPED because the kernel reported
  MERLIN_TOPK_SORTED_EMIT through ``persistent_topk_toggle_state`` field 6
  (``n-a`` until the first QSA top-k has run and latched it).
* ``device:`` -- counter slots [8]/[9], which the counter op fills from the
  C++ ``exact_medium_enabled()`` / ``det_large_enabled()`` statics
  (topk.cu persistent_topk_exact_counters).

WHEN. Once per worker, after ``compile_or_warm_up_model`` has run the MRV2
kernel warmup and cudagraph capture (``arm()``), i.e. after this process's
first persistent_topk and Marlin MoE launches -- NOT at the counter reader's
first read: the first ``maybe_log`` runs at the head of the warmup's first
execute step, before that step's forward. When the cpp-cache ops exist and a
static is still uninitialised at ``arm()``, the line is deferred to the head of
the first execute step where every static is filled, and forced out at shutdown
otherwise (with ``uninit`` values, which readers must treat as a failure). It
does not depend on ``MERLIN_TOPK_EXACT_COUNTERS_LOG``: it fires with the counter
log off.

WHAT THIS DOES NOT PROVE. Without the cpp-cache ops the C++ state is not
read: ``effective:`` then states the compiled constants of this source tree,
and the only C++-derived evidence is ``device:`` (EXACT_MEDIUM and DET_LARGE)
plus the untagged stderr policy line. A stale extension built from an older
tree (one that still read the variables) would not show up there. With the
cpp-cache ops that gap closes for every C++ name. Even then the line proves the
CONFIGURATION each rank ran, not that a switch changed any output: that is the
job of an output-equality run (identical sends -> one distinct output).
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable

from vllm.logger import init_logger
from vllm.v1.worker.e2_identity import _rank_tag

logger = init_logger(__name__)

# (variable, key, compiled value): the csrc getters return these constants
# (topk.cu *_enabled(), ops.cu merlin_full_k()); none reads the variable.
CXX_SWITCHES: tuple[tuple[str, str, bool], ...] = (
    ("MERLIN_FULL_K", "full_k", True),
    ("MERLIN_TIE_DET", "tie_det", True),
    ("MERLIN_TIE_RECENT", "tie_recent", True),
    ("MERLIN_TOPK_EXACT_MEDIUM", "exact_medium", True),
    ("MERLIN_TOPK_EXACT_DECODE", "exact_decode", True),
    ("MERLIN_TOPK_DET_LARGE", "det_large", True),
    ("MERLIN_TOPK_SORTED_EMIT", "sorted_emit", True),
)
QSA_SORT_NAME = "MERLIN_QSA_SORT"
RAW_ORDER = (
    "MERLIN_FULL_K",
    "MERLIN_QSA_SORT",
    "MERLIN_TIE_DET",
    "MERLIN_TIE_RECENT",
    "MERLIN_TOPK_EXACT_MEDIUM",
    "MERLIN_TOPK_EXACT_DECODE",
    "MERLIN_TOPK_DET_LARGE",
    "MERLIN_TOPK_SORTED_EMIT",
)
EFFECTIVE_ORDER = (
    "full_k",
    "qsa_sort",
    "tie_det",
    "tie_recent",
    "tie_recent_effective",
    "exact_medium",
    "exact_decode",
    "det_large",
    "sorted_emit",
)
CPP_ORDER = tuple(k for k in EFFECTIVE_ORDER if k != "qsa_sort")

_QSA_MODULE = "vllm.models.qwen4_exp.nvidia.ops.qsa_indexer"


# --------------------------------------------------------------- parsers ----
def qsa_sort_value(raw: str | None) -> str:
    """The vllm/merlin/envs.py MERLIN_QSA_SORT reader, read once at import by
    qsa_indexer.py (``_QSA_SORT``): unset is on, ``"0"``/``"1"`` as written,
    any other value goes through ``int()`` (the 3.0.0 legacy reading), which
    raises on ``"true"`` and ``""``. A raising value -- that engine would not
    have booted -- is reported as ``invalid`` rather than guessed."""
    if raw is None:
        return "1"
    try:
        return "1" if bool(int(raw)) else "0"
    except ValueError:
        return "invalid"


def effective_tie_recent(det_ties: bool, tie_recent: bool) -> bool:
    """csrc/merlin/topk/persistent_topk.cuh:65-68: ``det_ties ? tie_recent
    : true`` -- with TIE_DET off the restart paths keep most-recent-wins."""
    return tie_recent if det_ties else True


# --------------------------------------------------------------- sources ----
def _libc_getenv() -> tuple[Callable[[str], bytes | None], str]:
    """libc ``getenv``: the process ``environ`` a native reader would see.

    Falls back to ``os.environb`` / ``os.environ`` -- and says so in the line's
    ``source=`` -- where there is no C library to ask (an offline run off Linux).
    """
    try:
        import ctypes

        getenv = ctypes.CDLL(None).getenv
        getenv.restype = ctypes.c_char_p
        getenv.argtypes = [ctypes.c_char_p]

        def read(name: str) -> bytes | None:
            return getenv(name.encode("ascii"))

        return read, "libc-getenv"
    except (OSError, AttributeError, TypeError):
        pass
    environb = getattr(os, "environb", None)
    if environb is not None:
        return (lambda name: environb.get(name.encode("ascii"))), "os-environ"

    def read_environ(name: str) -> bytes | None:
        value = os.environ.get(name)
        return None if value is None else value.encode("utf-8", "surrogateescape")

    return read_environ, "os-environ"


def _printable(b: int) -> bool:
    return 0x21 <= b <= 0x7E and b not in b'"\\'


def encode_raw(raw: bytes | None) -> str:
    """``unset``, a plain token, or a quoted, hex-escaped string -- never a
    space, so the line splits on whitespace. The literal value ``unset`` is
    quoted so it cannot pass for an absent variable."""
    if raw is None:
        return "unset"
    if raw and raw != b"unset" and all(_printable(b) for b in raw):
        return raw.decode("ascii")
    return '"' + "".join(chr(b) if _printable(b) else f"\\x{b:02x}" for b in raw) + '"'


def environ_view(
    cxx_getenv: Callable[[str], bytes | None],
    py_getenv: Callable[[str], str | None],
) -> tuple[dict[str, str], dict[str, str]]:
    """(raw value by name, effective map) from one read of the environment."""
    raw: dict[str, str] = {}
    flags: dict[str, bool] = {}
    for name, key, compiled in CXX_SWITCHES:
        raw[name] = encode_raw(cxx_getenv(name))
        flags[key] = compiled
    qsa_raw = py_getenv(QSA_SORT_NAME)
    raw[QSA_SORT_NAME] = encode_raw(
        None if qsa_raw is None else qsa_raw.encode("utf-8", "surrogateescape")
    )
    eff = {key: "1" if flags[key] else "0" for key in flags}
    eff["qsa_sort"] = qsa_sort_value(qsa_raw)
    eff["tie_recent_effective"] = (
        "1" if effective_tie_recent(flags["tie_det"], flags["tie_recent"]) else "0"
    )
    return raw, {key: eff[key] for key in EFFECTIVE_ORDER}


_TOPK_STATE_FIELDS = (
    "tie_det",
    "tie_recent",
    "tie_recent_effective",
    "exact_medium",
    "exact_decode",
    "det_large",
    "sorted_emit",
)
# Known layouts: version -> field count. Layout 2 is layout 1
# plus field 6 (MERLIN_TOPK_SORTED_EMIT); fields 0-5 are unchanged.
_TOPK_STATE_LAYOUTS = {1: 6, 2: 7}


def decode_topk_state(state: int) -> dict[str, str] | str:
    """Unpack ``persistent_topk_toggle_state()``: 2 bits per field (0 off,
    1 on, 2 not initialised), field count at bits 48-55, layout version at
    bits 56-62 -- (1, 6) or (2, 7). Any other layout is refused with a reason
    string. A layout-1 build has no sorted emission, reported as ``n-a``."""
    version, count = (state >> 56) & 0x7F, (state >> 48) & 0xFF
    if _TOPK_STATE_LAYOUTS.get(version) != count:
        return f"unknown-layout-v{version}-{count}-fields"
    names = {0: "0", 1: "1", 2: "uninit"}
    out = {
        key: names.get((state >> (2 * i)) & 3, "invalid")
        for i, key in enumerate(_TOPK_STATE_FIELDS[:count])
    }
    for key in _TOPK_STATE_FIELDS[count:]:
        out[key] = "n-a"
    return out


def decode_full_k_state(state: int) -> str:
    """``marlin_moe_full_k_state()``: -1 not initialised, 0 off, 1 on."""
    return {-1: "uninit", 0: "0", 1: "1"}.get(int(state), "invalid")


def _op(namespace: str, name: str):
    try:
        import torch

        return getattr(getattr(torch.ops, namespace), name, None)
    except (AttributeError, RuntimeError, ImportError):
        return None


def cpp_cache() -> dict[str, str] | str:
    """The C++ cached switches, or a reason string when the build cannot say."""
    topk_op = _op("_C", "merlin_persistent_topk_toggle_state")
    full_k_op = _op("_moe_C", "marlin_moe_full_k_state")
    missing = [
        name
        for name, op in (
            ("_C.merlin_persistent_topk_toggle_state", topk_op),
            ("_moe_C.marlin_moe_full_k_state", full_k_op),
        )
        if op is None
    ]
    if missing:
        return "op-absent:" + ",".join(missing)
    decoded = decode_topk_state(int(topk_op()))
    if isinstance(decoded, str):
        return decoded
    out = dict(decoded, full_k=decode_full_k_state(int(full_k_op())))
    return {key: out[key] for key in CPP_ORDER}


def qsa_sort_live() -> str:
    module = sys.modules.get(_QSA_MODULE)
    if module is None or not hasattr(module, "_QSA_SORT"):
        return "n-a"
    return "1" if bool(module._QSA_SORT) else "0"


def sorted_emit_skip_live() -> str:
    """``qsa_indexer._kernel_sorted_emit_state``: 1 = the Python sort is being
    skipped because the kernel reported sorted emission, 0 = not skipped,
    ``n-a`` = not latched yet (or the module was never loaded)."""
    module = sys.modules.get(_QSA_MODULE)
    state = getattr(module, "_kernel_sorted_emit_state", None) if module else None
    if state is None:
        return "n-a"
    return "1" if state else "0"


def device_switches() -> dict[str, str]:
    """Counter slots [8]/[9] (switch / switch_large). One synchronizing read
    (topk_counters.read refuses inside a capture), so it is only called from
    arm() and the head of an execute step."""
    try:
        from vllm.models.qwen4_exp.nvidia.ops import topk_counters

        counters = topk_counters.read()
    except Exception:
        counters = {}
    return {
        key: str(counters[key]) if key in counters else "n-a"
        for key in ("switch", "switch_large")
    }


# ------------------------------------------------------------------ line ----
def toggles_line(
    tag: str,
    pid: int,
    raw: dict[str, str],
    effective: dict[str, str],
    env_source: str,
    cpp: dict[str, str] | str,
    qsa_live: str,
    device: dict[str, str],
    sorted_skip: str = "n-a",
) -> str:
    parts = [f"e2-toggles[{tag}]: pid={pid}"]
    parts += [f"{name}={raw[name]}" for name in RAW_ORDER]
    parts.append("effective:")
    parts += [f"{key}={effective[key]}" for key in EFFECTIVE_ORDER]
    parts.append(f"source={env_source}")
    parts.append("cpp-cache:")
    if isinstance(cpp, str):
        parts += ["n-a", f"reason={cpp}"]
    else:
        parts += [f"{key}={cpp[key]}" for key in CPP_ORDER]
        parts.append("source=cpp-cache")
    parts += [
        "python:",
        f"qsa_sort={qsa_live}",
        f"sorted_emit_skip={sorted_skip}",
        "source=qsa_indexer._QSA_SORT",
    ]
    parts.append("device:")
    parts += [f"{key}={device[key]}" for key in ("switch", "switch_large")]
    parts.append("source=counter-slots-8-9")
    return " ".join(parts)


# ----------------------------------------------------------------- state ----
_armed = False
_emitted = False
_deferral_noted = False


def arm(**kwargs) -> str:
    """Once per worker, at the end of compile_or_warm_up_model (warmup and
    capture done): emit now, unless a C++ static is still uninitialised."""
    global _armed
    _armed = True
    return maybe_emit(**kwargs)


def maybe_emit(
    force: bool = False,
    *,
    tag_fn: Callable[[], str] = _rank_tag,
    getenv_fn: Callable[[], tuple[Callable[[str], bytes | None], str]] = _libc_getenv,
    cpp_fn: Callable[[], dict[str, str] | str] = cpp_cache,
    device_fn: Callable[[], dict[str, str]] = device_switches,
    qsa_fn: Callable[[], str] = qsa_sort_live,
    sorted_skip_fn: Callable[[], str] = sorted_emit_skip_live,
) -> str:
    """Emit the line at most once per process. One bool check when already
    emitted or not yet armed, so it can sit at the head of every execute step.
    Never raises. Returns the line when this call emitted it, else ""."""
    global _emitted, _deferral_noted
    if _emitted or not _armed:
        return ""
    tag = "PP?_TP?"
    try:
        tag = tag_fn()
        cpp = cpp_fn()
        pending = (
            sorted(k for k, v in cpp.items() if v == "uninit")
            if isinstance(cpp, dict)
            else []
        )
        if pending and not force:
            if not _deferral_noted:
                _deferral_noted = True
                # Deliberately not 'e2-toggles[': log readers count those.
                logger.info(
                    "e2-toggles deferred on %s: C++ cache not initialised for %s",
                    tag,
                    ",".join(pending),
                )
            return ""
        cxx_getenv, env_source = getenv_fn()
        raw, effective = environ_view(cxx_getenv, os.environ.get)
        line = toggles_line(
            tag,
            os.getpid(),
            raw,
            effective,
            env_source,
            cpp,
            qsa_fn(),
            device_fn(),
            sorted_skip_fn(),
        )
        _emitted = True
        logger.info("%s", line)
        return line
    except Exception as error:  # log-only: must never break a boot or a step
        _emitted = True
        logger.warning(
            "e2-toggles[%s]: unavailable (%s: %s)", tag, type(error).__name__, error
        )
        return ""
