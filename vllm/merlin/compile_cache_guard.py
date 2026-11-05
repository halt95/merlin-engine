# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Compile-cache fault guard.

A sticky CUDA fault (illegal memory access and kin) must not leave a compile
cache that a restart loads and crashes on again. The caches a run uses on vLLM
0.33 with torch 2.14 (AOT on by default):

- ``<VLLM_CACHE_ROOT>/torch_compile_cache/torch_aot_compile/<sha256>/``: the
  hash dir, shared by every rank. ``inductor_cache/`` under it is the
  process-wide ``TORCHINDUCTOR_CACHE_DIR`` (generated kernels, runtime autotune
  results) and ``rank_<r>_<dp>/model`` is the AOT artifact.
- ``<VLLM_CACHE_ROOT>/torch_compile_cache/<sha256[:10]>/rank_<r>_<dp>/<prefix>/``:
  VllmBackend's per-rank graph files.
- A caller-set ``TRITON_CACHE_DIR`` outside those (the container's
  ``/cache/triton``, the launcher's ``<cache root>/triton``): Triton binaries
  and persisted autotune results.

Three parts, all best-effort (a guard failure never masks the original error):

1. **Fault marker.** Each process registers the cache dirs it uses. When a
   sticky device fault reaches a worker or EngineCore handler, the process
   writes ``merlin-fault.json`` into each registered dir and stops persisting
   compile artifacts for the rest of its life.
2. **Quarantine on the next boot.** The first process to use a dir that carries
   a marker moves the whole dir to ``<parent>/.merlin-quarantine/`` (newest
   ``_KEEP`` kept per parent) and the run recompiles. Every process that uses
   a dir holds a shared lock on ``<parent>/.merlin-inuse-<name>.lock`` for its
   lifetime; a marked dir that another live process holds (another engine
   instance on the same cache root) is never moved: the guard warns and
   leaves it. The check and the move run under ``<parent>/.merlin-guard.lock``,
   so ranks starting together see either the old dir with its marker or a
   fresh one. Each dir is checked once per process.
3. **AOT artifact digest.** ``rank_*/model`` gets a ``model.sha256`` sidecar;
   a mismatch on load is a load failure (recompile), a missing sidecar (older
   caches) is accepted.

Quarantine needs POSIX ``flock``; without it the guard warns once and never
moves a dir (markers and digests still work).

``VLLM_MERLIN_COMPILE_CACHE_GUARD=0`` turns all three off (strict 0/1, parsed
by the fork env registry). Nothing here touches numerics: a quarantined cache
recompiles the same graphs.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import sys
import threading
import time
from collections.abc import Iterator

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)

ENV_NAME = "VLLM_MERLIN_COMPILE_CACHE_GUARD"
MARKER_NAME = "merlin-fault.json"
QUARANTINE_DIRNAME = ".merlin-quarantine"
GUARD_LOCK_NAME = ".merlin-guard.lock"
INUSE_LOCK_PREFIX = ".merlin-inuse-"
DIGEST_SUFFIX = ".sha256"
_KEEP = 2
_ERROR_CHARS = 2000

# Sticky device faults: after one of these the CUDA context is unusable, so any
# later work in the process (including cache writes) is suspect. Lower-case
# substrings of the CUDA runtime / driver error strings. Out-of-memory is not
# sticky and is deliberately absent.
STICKY_FAULT_SIGNATURES: tuple[str, ...] = (
    "illegal memory access",
    "illegal instruction",
    "misaligned address",
    "unspecified launch failure",
    "device-side assert triggered",
    "invalid program counter",
    "hardware stack error",
    "uncorrectable ecc error",
    "cudaerrorillegaladdress",
    "cudaerrorlaunchfailure",
)

_state_lock = threading.Lock()
_registered: dict[str, dict[str, object]] = {}
_prepared: set[str] = set()
_inuse_fds: dict[str, int] = {}
_fault_seen = False


def enabled() -> bool:
    """``VLLM_MERLIN_COMPILE_CACHE_GUARD`` through the fork env registry:
    unset or "1" is on, "0" is off, anything else is refused."""
    return bool(envs.VLLM_MERLIN_COMPILE_CACHE_GUARD)


def fault_seen() -> bool:
    """True once this process recorded a sticky device fault."""
    return _fault_seen


def _exception_texts(error: object) -> Iterator[str]:
    seen: set[int] = set()
    stack = [error]
    while stack:
        item = stack.pop()
        if item is None or id(item) in seen:
            continue
        seen.add(id(item))
        if isinstance(item, BaseException):
            yield f"{type(item).__name__}: {item}"
            for note in getattr(item, "__notes__", None) or ():
                yield str(note)
            stack.append(item.__cause__)
            stack.append(item.__context__)
        else:
            yield str(item)


def is_sticky_device_fault(error: object) -> bool:
    """Whether an exception (its cause/context chain included) or a failure
    string carries a sticky CUDA fault signature."""
    for text in _exception_texts(error):
        low = text.lower()
        if any(sig in low for sig in STICKY_FAULT_SIGNATURES):
            return True
    return False


# --- locks (POSIX flock; production platforms are Linux) --------------------


def _flock_module():
    try:
        import fcntl
    except ImportError:
        return None
    return fcntl


def locking_available() -> bool:
    return _flock_module() is not None


def _open_lock(path: str) -> int:
    return os.open(path, os.O_RDWR | os.O_CREAT, 0o644)


def _flock(fd: int, exclusive: bool, blocking: bool) -> bool:
    """Take (or convert) a lock on ``fd``; False when non-blocking and held."""
    fcntl = _flock_module()
    op = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
    if not blocking:
        op |= fcntl.LOCK_NB
    try:
        fcntl.flock(fd, op)
    except BlockingIOError:
        return False
    return True


def _close_lock(fd: int) -> None:
    os.close(fd)  # closing the descriptor releases its flock


@contextlib.contextmanager
def _guard_lock(parent: str) -> Iterator[None]:
    fd = _open_lock(os.path.join(parent, GUARD_LOCK_NAME))
    try:
        _flock(fd, exclusive=True, blocking=True)
        yield
    finally:
        _close_lock(fd)


def _inuse_lock_path(path: str) -> str:
    parent, name = os.path.split(path)
    return os.path.join(parent, f"{INUSE_LOCK_PREFIX}{name}.lock")


# --- quarantine --------------------------------------------------------------


def _read_marker(path: str) -> dict[str, object]:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {"raw": str(data)[:_ERROR_CHARS]}
    except (OSError, ValueError) as e:
        return {"unreadable": str(e)}


def _prune(quarantine_root: str, keep: int = _KEEP) -> None:
    import shutil

    try:
        entries = sorted(
            e for e in os.listdir(quarantine_root) if not e.startswith(".")
        )
    except OSError:
        return
    for name in entries[: max(0, len(entries) - keep)]:
        shutil.rmtree(os.path.join(quarantine_root, name), ignore_errors=True)


def _quarantine_locked(path: str, inuse_fd: int) -> str | None:
    """Move a marked dir aside. The caller holds the parent guard lock and
    ``inuse_fd`` exclusively (no live process uses the dir)."""
    marker = os.path.join(path, MARKER_NAME)
    if not os.path.isfile(marker):
        return None
    info = _read_marker(marker)
    quarantine_root = os.path.join(os.path.dirname(path), QUARANTINE_DIRNAME)
    os.makedirs(quarantine_root, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    dest = os.path.join(
        quarantine_root, f"{stamp}-{os.getpid()}-{os.path.basename(path)}"
    )
    os.replace(path, dest)
    logger.warning(
        "Compile cache %s was in use when a CUDA fault ended its last run "
        "(%s at %s, pid %s: %s); moved to %s and recompiling from scratch "
        "(%s=0 disables).",
        path,
        info.get("where", "?"),
        info.get("time_utc", "?"),
        info.get("pid", "?"),
        str(info.get("error", info))[:300],
        dest,
        ENV_NAME,
    )
    _prune(quarantine_root)
    return dest


def prepare_cache_dir(path: str, kind: str = "compile") -> str | None:
    """Call before a process first uses a cache dir.

    Quarantines the dir if a faulted run left a marker in it and no other live
    process holds it, then registers it for this process's fault markers and
    holds its in-use lock for the life of the process. Returns the quarantine
    path when the dir was moved, else None. Checks a given dir once per process.
    """
    if not path or not enabled():
        return None
    key = os.path.abspath(path)
    with _state_lock:
        if key in _prepared:
            return None
        _prepared.add(key)
        _registered.setdefault(key, {"kind": kind})
    if not locking_available():
        if os.path.isfile(os.path.join(key, MARKER_NAME)):
            logger.warning(
                "Compile cache %s carries a CUDA-fault marker, but this "
                "platform has no flock, so it is not quarantined; delete it "
                "by hand if the restart crashes.",
                key,
            )
        return None
    parent = os.path.dirname(key)
    try:
        os.makedirs(parent, exist_ok=True)
        with _guard_lock(parent):
            fd = _open_lock(_inuse_lock_path(key))
            dest = None
            if os.path.isfile(os.path.join(key, MARKER_NAME)):
                if _flock(fd, exclusive=True, blocking=False):
                    dest = _quarantine_locked(key, fd)
                else:
                    logger.warning(
                        "Compile cache %s carries a CUDA-fault marker but "
                        "another live process is using it, so it is left in "
                        "place. Stop every instance that shares this cache "
                        "root (or give each its own) to let it be quarantined.",
                        key,
                    )
            # Hold a shared in-use lock for the life of this process (an
            # exclusive lock taken above converts to shared).
            _flock(fd, exclusive=False, blocking=True)
            with _state_lock:
                _inuse_fds[key] = fd
            return dest
    except OSError as e:
        logger.warning("compile cache guard: could not check %s: %s", key, e)
        return None


def _inside(path: str, parent: str) -> bool:
    try:
        return os.path.commonpath([path, parent]) == parent
    except ValueError:  # different drives
        return False


def prepare_external_cache_dir(path: str | None, managed_root: str) -> str | None:
    """Cover a caller-set cache dir such as ``TRITON_CACHE_DIR``.

    It is skipped only when a dir this process already registered contains it
    (quarantining that dir moves it too). Being somewhere under ``managed_root``
    (the ``torch_compile_cache`` tree) is not enough: ``$ROOT/torch_compile_cache/
    triton`` sits beside the hash dirs, not inside one, so it is covered here.
    ``managed_root`` only scopes that check to the managed tree's dirs.
    """
    if not path:
        return None
    key = os.path.abspath(path)
    root = os.path.abspath(managed_root)
    with _state_lock:
        covering = [
            d for d, meta in _registered.items()
            if meta.get("kind") != "external" and _inside(d, root)
        ]
    if any(_inside(key, d) for d in covering):
        return None
    return prepare_cache_dir(key, kind="external")


def prepare_triton_cache_dir(path: str | None) -> str | None:
    """Cover a caller-set ``TRITON_CACHE_DIR`` (``path``) from worker device init.

    Workers use that dir from ``init_device`` on, long before their first
    compile, so its in-use lock is taken here: a restarting instance on the
    same cache root then cannot quarantine it under a live peer that has not
    compiled yet. The compile-time call finds it already prepared.
    """
    if not path:
        return None
    return prepare_external_cache_dir(
        path, os.path.join(envs.VLLM_CACHE_ROOT, "torch_compile_cache")
    )


def note_loaded_from_cache(path: str) -> None:
    """Record that this process served from an artifact loaded out of ``path``
    (diagnostic field in the fault marker)."""
    key = os.path.abspath(path)
    with _state_lock:
        if key in _registered:
            _registered[key]["loaded_from_cache"] = True


# --- fault recording ---------------------------------------------------------


def _write_json_atomic(path: str, payload: dict[str, object]) -> None:
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1, sort_keys=True)
    os.replace(tmp, path)


def _stop_inductor_cache_writes() -> None:
    torch = sys.modules.get("torch")
    if torch is None:
        return
    with contextlib.suppress(Exception):
        torch._inductor.config.force_disable_caches = True


def record_fault(error: object, where: str) -> bool:
    """Mark this process's cache dirs after a sticky device fault.

    Never raises. Returns True when the error was a sticky fault (markers are
    then written into every registered dir that still exists).
    """
    global _fault_seen
    try:
        if not enabled() or not is_sticky_device_fault(error):
            return False
        with _state_lock:
            first = not _fault_seen
            _fault_seen = True
            dirs = {k: dict(v) for k, v in _registered.items()}
        _stop_inductor_cache_writes()
        if not first:
            return True
        text = next(_exception_texts(error), "")[:_ERROR_CHARS]
        for path, meta in dirs.items():
            if not os.path.isdir(path):
                continue
            payload: dict[str, object] = {
                "version": 1,
                "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "pid": os.getpid(),
                "where": where,
                "error": text,
                **meta,
            }
            try:
                _write_json_atomic(os.path.join(path, MARKER_NAME), payload)
            except OSError as e:
                logger.warning("compile cache guard: no marker in %s: %s", path, e)
        if dirs:
            logger.error(
                "CUDA fault (%s): marked %d cache dir(s) for quarantine on the "
                "next boot: %s",
                where,
                len(dirs),
                ", ".join(sorted(dirs)),
            )
        return True
    except Exception:  # noqa: BLE001 -- never mask the original failure
        return False


# --- AOT artifact digest -----------------------------------------------------


def digest_path(artifact_path: str) -> str:
    return artifact_path + DIGEST_SUFFIX


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def publish_artifact(tmp_path: str, artifact_path: str) -> None:
    """Atomically publish a finished artifact with its digest sidecar.

    The stale sidecar goes first, so a crash between the steps leaves either
    the old pair, or the new artifact without a sidecar (accepted as legacy);
    never a new artifact under an old digest. With the guard off this is the
    plain ``os.replace``.
    """
    if not enabled():
        os.replace(tmp_path, artifact_path)
        return
    digest = file_sha256(tmp_path)
    sidecar = digest_path(artifact_path)
    with contextlib.suppress(FileNotFoundError):
        os.remove(sidecar)
    os.replace(tmp_path, artifact_path)
    tmp_sidecar = f"{sidecar}.{os.getpid()}.tmp"
    with open(tmp_sidecar, "w", encoding="ascii") as f:
        f.write(digest + "\n")
    os.replace(tmp_sidecar, sidecar)


def verify_artifact(artifact_path: str) -> None:
    """Raise ``RuntimeError`` when the artifact does not match its sidecar.

    A missing sidecar passes (caches written before this guard).
    """
    if not enabled():
        return
    try:
        with open(digest_path(artifact_path), encoding="ascii") as f:
            expected = f.read().strip()
    except FileNotFoundError:
        return
    actual = file_sha256(artifact_path)
    if actual != expected:
        raise RuntimeError(
            f"Compile cache artifact digest mismatch ({actual[:12]} != "
            f"{expected[:12]}); the artifact is damaged or partly written."
        )
