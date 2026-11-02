# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Fail-closed helpers for the PLE single-slot transport."""

import functools
import os
import time
from collections.abc import Callable
from types import TracebackType
from typing import Any, NoReturn, TypeVar

_F = TypeVar("_F", bound=Callable[..., Any])

_TRUSTED_ERROR_LABELS = (
    (RuntimeError, "RuntimeError"),
    (TimeoutError, "TimeoutError"),
    (ValueError, "ValueError"),
    (OSError, "OSError"),
    (EOFError, "EOFError"),
    (BrokenPipeError, "BrokenPipeError"),
    (AssertionError, "AssertionError"),
    (KeyError, "KeyError"),
    (IndexError, "IndexError"),
    (TypeError, "TypeError"),
    (MemoryError, "MemoryError"),
    (KeyboardInterrupt, "KeyboardInterrupt"),
)

_EXC_ARGS = BaseException.args.__get__
_EXC_TRACEBACK = BaseException.__traceback__.__get__
_EXC_CAUSE = BaseException.__cause__.__get__
_EXC_CONTEXT = BaseException.__context__.__get__
_EXC_SUPPRESS_CONTEXT = BaseException.__suppress_context__.__get__
_TYPE_QUALNAME = type.__dict__["__qualname__"].__get__
_TYPE_MODULE = type.__dict__["__module__"].__get__
_MAX_CHAIN = 16
_MAX_ARG_CHARS = 1024

_FATAL_NOTIFY_FD: int | None = None


def set_fatal_notify_fd(fd: int | None) -> None:
    """Register the fault pipe worker_main received from the executor."""
    global _FATAL_NOTIFY_FD
    _FATAL_NOTIFY_FD = fd


def notify_parent_fatal() -> None:
    """Tell the executor this worker is dying, before the process is gone.

    The executor's only other signal is the process sentinel, which the kernel
    closes when the LAST task of the worker leaves -- after the address-space
    teardown, not at os._exit. On a rank whose GPU work is paired with a live
    peer that teardown can wait for the peer's exit, while the peer waits on
    this rank in turn, and the engine hangs on the response queue. One byte
    here wakes the monitor at once so the failure ladder ends the peers,
    which is what frees the teardown. Nonblocking, never raises:
    the caller os._exits right after, whatever happens here.
    """
    fd = _FATAL_NOTIFY_FD
    if fd is None:
        return
    try:
        os.set_blocking(fd, False)
        os.write(fd, b"F")
    except BaseException:
        pass


def _safe_type_label(cls: type) -> str:
    """``module.qualname`` of ``cls`` without running class-defined code."""
    try:
        qualname = _TYPE_QUALNAME(cls)
        module = _TYPE_MODULE(cls)
    except BaseException:
        return "<unknown type>"
    if type(qualname) is not str:
        qualname = "<unknown type>"
    if type(module) is str and module != "builtins":
        return f"{module}.{qualname}"
    return qualname


def _safe_exception_line(error: BaseException) -> str:
    """Type label plus the exact-``str``/``int`` args; nothing else is shown."""
    label = _safe_type_label(type(error))
    args = _EXC_ARGS(error)
    if type(args) is not tuple or len(args) == 0:
        return label
    parts = []
    for arg in args[:4]:
        if type(arg) is str:
            parts.append(arg[:_MAX_ARG_CHARS])
        elif type(arg) is int and -(10**18) < arg < 10**18:
            parts.append(str(arg))
        else:
            parts.append(f"<{_safe_type_label(type(arg))}>")
    if len(args) > 4:
        parts.append("...")
    if len(args) == 1:
        return f"{label}: {parts[0]}"
    return f"{label}: ({', '.join(parts)})"


def _safe_frames(error: BaseException) -> list[str]:
    lines = []
    tb = _EXC_TRACEBACK(error)
    while type(tb) is TracebackType:
        code = tb.tb_frame.f_code
        filename = code.co_filename
        name = code.co_qualname
        lineno = tb.tb_lineno
        lines.append(
            f'  File "{filename if type(filename) is str else "?"}", '
            f"line {lineno if type(lineno) is int else '?'}, "
            f"in {name if type(name) is str else '?'}\n"
        )
        tb = tb.tb_next
    return lines


def _safe_format_exception(error: BaseException) -> str:
    """A traceback.format_exception look-alike that runs no exception code.

    Frames come from the traceback objects (file, line, function; no source
    lines, which would mean file I/O on the dying path); the message is the
    exception's exact-``str`` args, never ``str(error)``. Chains follow
    ``__cause__`` / ``__context__`` like the interpreter, bounded and
    cycle-safe.
    """
    chain: list[tuple[BaseException, str]] = []
    seen: set[int] = set()
    current: BaseException | None = error
    link = ""
    while current is not None and len(chain) < _MAX_CHAIN:
        if id(current) in seen:
            break
        seen.add(id(current))
        chain.append((current, link))
        cause = _EXC_CAUSE(current)
        if cause is not None:
            current = cause
            link = (
                "\nThe above exception was the direct cause of the "
                "following exception:\n\n"
            )
        elif _EXC_SUPPRESS_CONTEXT(current) is not True:
            current = _EXC_CONTEXT(current)
            link = (
                "\nDuring handling of the above exception, another "
                "exception occurred:\n\n"
            )
        else:
            current = None
    out: list[str] = []
    # Innermost first, as the interpreter prints; each outer exception's link
    # text sits between it and the exception it chains to.
    for index in range(len(chain) - 1, -1, -1):
        exc, _ = chain[index]
        out.append("Traceback (most recent call last):\n")
        out.extend(_safe_frames(exc))
        out.append(_safe_exception_line(exc) + "\n")
        if index > 0:
            out.append(chain[index][1])
    return "".join(out)


def ple_fail_closed(
    *,
    where: str,
    rank: int,
    layer: str,
    sequence: int,
    error: BaseException,
) -> NoReturn:
    """Emit one best-effort nonblocking diagnostic and terminate immediately."""
    try:
        os.set_blocking(2, False)
        error_type = type(error)
        error_name = "Exception"
        trusted_error = False
        for trusted_type, trusted_name in _TRUSTED_ERROR_LABELS:
            if error_type is trusted_type:
                error_name = trusted_name
                trusted_error = True
                break

        safe_where = where[:128] if type(where) is str else "unknown"
        safe_layer = layer[:256] if type(layer) is str else "unknown"
        safe_rank = rank if type(rank) is int else -1
        safe_sequence = sequence if type(sequence) is int else -1
        detail_suffix = ""
        if trusted_error:
            args = BaseException.args.__get__(error, error_type)
            if type(args) is tuple and len(args) == 1 and type(args[0]) is str:
                detail = args[0][:512].replace("\n", "\\n")
                detail_suffix = f":{detail}"
        message = (
            f"PLE fatal: where={safe_where} rank={safe_rank} layer={safe_layer} "
            f"sequence={safe_sequence} error={error_name}{detail_suffix} "
            f"pid={os.getpid()} "
            f"monotonic={time.monotonic():.9f} wall={time.time():.9f}\n"
        )
        os.write(2, message.encode("utf-8", "backslashreplace")[:2048])
    except BaseException:
        pass
    try:
        tb = _safe_format_exception(error)
        raw = tb.encode("utf-8", "backslashreplace")
        clipped = len(raw) > 8192
        raw = raw[-8192:]
        os.write(
            2,
            b"PLE fatal traceback (tail) follows:\n"
            if clipped
            else b"PLE fatal traceback follows:\n",
        )
        os.write(2, raw)
        if not raw.endswith(b"\n"):
            os.write(2, b"\n")
    except BaseException:
        try:  # noqa: SIM105
            os.write(2, b"PLE fatal: traceback unavailable\n")
        except BaseException:
            pass
    notify_parent_fatal()
    os._exit(1)


def fail_ple_execute_model(func: _F) -> _F:
    """Fail a non-shutdown worker-step escape only on PLE-owning ranks.

    This includes ``sample_tokens``, the second half of the split execute-model
    step. A rank without a PLE connector rethrows the original exception.
    """

    @functools.wraps(func)
    def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            return func(self, *args, **kwargs)
        except SystemExit:
            # WorkerProc's signal handler uses SystemExit for clean shutdown.
            raise
        except BaseException as error:
            connector = getattr(self, "_ple_offload_connector", None)
            if connector is None:
                model_runner = getattr(self, "model_runner", None)
                connector = getattr(model_runner, "_ple_offload_connector", None)
            if connector is None:
                raise
            connector._fail_closed(func.__name__, error)

    return wrapped  # type: ignore[return-value]
