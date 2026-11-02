# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Minimal PLE child bootstrap; intentionally imports no torch or vLLM."""

import importlib.abc
import importlib.machinery
import os
import pickle
import struct
import sys
import threading
from multiprocessing.connection import Connection
from types import ModuleType
from typing import Any

_FAULT = None
# The same gate as vllm/v1/ple_offload/test_fault.py: unset, empty and "0" mean
# off; any other value arms the test-only injector.
if os.environ.get("VLLM_PLE_TEST_FAULT", "0").strip() not in ("", "0"):
    try:
        from vllm_ple_test_fault import (
            _FAULT as _FAULT,
        )
        from vllm_ple_test_fault import (
            _block_on_harness_gate,
        )
    except ImportError as exc:
        raise RuntimeError(
            "VLLM_PLE_TEST_FAULT is set but the test-only injector package "
            "'vllm_ple_test_fault' is not importable (put tests/v1/ple_offload "
            "on PYTHONPATH, or unset VLLM_PLE_TEST_FAULT / set it to 0)"
        ) from exc


def _send_partial_ready_frame(fd: int) -> tuple[int, int]:
    """Write the complete header and an incomplete body in one syscall."""
    body = pickle.dumps(
        {"status": "READY", "layer_names": []},
        protocol=pickle.HIGHEST_PROTOCOL,
    )
    frame = struct.pack("!i", len(body)) + body
    fragment = frame[: 4 + max(1, len(body) // 2)]
    try:
        return os.write(fd, fragment), len(frame)
    except OSError:
        return 0, len(frame)


def _send_partial_ready_header(fd: int) -> tuple[int, int]:
    """Write fewer than the four required header bytes in one syscall."""
    header = struct.pack("!i", 1)
    try:
        return os.write(fd, header[:2]), len(header)
    except OSError:
        return 0, len(header)


def _read_exact(fd: int, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = os.read(fd, size - len(chunks))
        if not chunk:
            raise EOFError("PLE launch payload was truncated")
        chunks.extend(chunk)
    return bytes(chunks)


def _recv_launch_payload(fd: int) -> dict:
    size = struct.unpack("!Q", _read_exact(fd, 8))[0]
    payload = pickle.loads(_read_exact(fd, size))
    if not isinstance(payload, dict):
        raise RuntimeError("invalid PLE bootstrap payload")
    return payload


def _fires(
    cut: str,
    *,
    unit: str = "bootstrap-launch",
    transaction: str = "launch:1",
    bytes_written: int | None = None,
    frame_size: int | None = None,
) -> bool:
    return _FAULT is not None and _FAULT.fires(
        cut,
        role="bootstrap",
        unit=unit,
        eligible=True,
        transaction_id=transaction,
        correlation_id="launch:1",
        bytes_written=bytes_written,
        frame_size=frame_size,
    )


def _write_and_ack_partial(
    fd: int,
    cut: str,
    sender: Any,
) -> bool:
    """A short or failed prefix write cannot acknowledge or enter the park."""
    if _FAULT is None or _FAULT.cut != cut:
        # Disarmed or armed for another cut: the control stream is untouched.
        return False
    written, frame_size = sender(fd)
    if written <= 0 or written >= frame_size:
        return False
    expected = 2 if cut == "launch_partial_ready_header" else None
    if expected is not None and written != expected:
        return False
    if expected is None and written <= 4:
        return False
    return _fires(
        cut,
        bytes_written=written,
        frame_size=frame_size,
    )


class _FaultImportLoader(importlib.abc.Loader):
    def __init__(self, delegate: importlib.abc.Loader) -> None:
        self._delegate = delegate

    def create_module(self, spec):
        create = getattr(self._delegate, "create_module", None)
        return None if create is None else create(spec)

    def exec_module(self, module: ModuleType) -> None:
        if _fires(
            "launch_block_worker_import",
            unit="bootstrap-import",
            transaction="module:vllm.v1.ple_offload.worker",
        ):
            _block_on_harness_gate()
        self._delegate.exec_module(module)


class _FaultImportFinder(importlib.abc.MetaPathFinder):
    _TARGET = "vllm.v1.ple_offload.worker"

    def find_spec(self, fullname: str, path, target=None):
        if fullname != self._TARGET:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot find {fullname}")
        if not isinstance(spec.loader, importlib.abc.Loader):
            raise ImportError(f"{fullname} has no executable loader")
        spec.loader = _FaultImportLoader(spec.loader)
        return spec


def _install_import_gate() -> None:
    target = _FaultImportFinder._TARGET
    if target in sys.modules:
        raise RuntimeError("PLE worker import gate was a cache hit")
    sys.meta_path.insert(0, _FaultImportFinder())


def main() -> None:
    """Receive launch state before importing torch or vLLM."""
    if len(sys.argv) != 3:
        raise SystemExit("expected inherited PLE control and parent-death fds")
    control_fd = int(sys.argv[1])
    death_fd = int(sys.argv[2])
    if control_fd == death_fd:
        raise RuntimeError("PLE control and parent-death fds must be distinct")

    os.set_inheritable(control_fd, False)
    os.set_inheritable(death_fd, False)
    death_pipe = Connection(death_fd)

    def monitor_parent() -> None:
        try:  # noqa: SIM105
            death_pipe.recv()
        except (EOFError, OSError):
            # SIM105 declined: this module's contract is the docstring above --
            # its import list is the audited surface of the pre-torch child.
            pass
        os._exit(1)

    threading.Thread(
        target=monitor_parent,
        daemon=True,
        name="PleOffloadBootstrapDeathMonitor",
    ).start()

    if _fires("launch_park_before_payload_read"):
        threading.Event().wait()

    # An armed reconstruction payload invokes its reducer inside this loads().
    payload = _recv_launch_payload(control_fd)

    if _write_and_ack_partial(
        control_fd,
        "launch_partial_ready",
        _send_partial_ready_frame,
    ):
        threading.Event().wait()

    if _write_and_ack_partial(
        control_fd,
        "launch_partial_ready_header",
        _send_partial_ready_header,
    ):
        threading.Event().wait()

    if _fires("launch_silent_ready"):
        threading.Event().wait()

    ready_pipe = Connection(control_fd)
    vllm_config = pickle.loads(payload["vllm_config_payload"])

    if _FAULT is not None and _FAULT.cut == "launch_block_worker_import":
        _install_import_gate()
    from vllm.v1.ple_offload.worker import PleOffloadWorker

    PleOffloadWorker.proc_main(
        vllm_config=vllm_config,
        num_workers=int(payload["num_workers"]),
        ipc_addr=str(payload["ipc_addr"]),
        ready_pipe=ready_pipe,
    )
