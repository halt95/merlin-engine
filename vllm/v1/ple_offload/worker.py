# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Dedicated CPU-offload process for PLE embedding layers.

This module implements a standalone process that:
1. Loads only the :class:`PleOffloadLayer` weights into CPU memory.
2. Accepts per-step computation requests from GPU worker processes.
3. Runs ``forward_impl()`` on CPU, writes the rows into every TP worker's
   shared host output buffer for the requesting DP rank and sets that
   worker's shared host flag (host-mapped pull design: this process holds no
   CUDA context; the GPU waits on the host flag and DMAs host -> GPU inside
   its captured forward).

The TP workers within one DP rank receive identical inputs, so the CPU result
is computed once per DP rank and fanned out to all of its TP ranks.

Class structure mirrors the GPU worker pattern in multiproc_executor.py:

  PleOffloadWorkerHandle -- handle held by the spawning GPU worker
  PleOffloadWorker       -- process lifecycle and READY handshake
  PleOffloadRunner       -- owns weights and serves inference requests
"""

import contextlib
import copy
import os
import pickle
import select
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass
from multiprocessing.connection import Connection
from typing import Any, cast

import msgspec
import torch
import torch.distributed as dist
import zmq

import vllm.envs as envs
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed.parallel_state import (
    ensure_model_parallel_initialized,
    init_distributed_environment,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.ple_offload_layer import (
    PleOffloadLayer,
    mark_as_offload_worker,
)
from vllm.model_executor.model_loader import get_model_loader
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.model_loader.dummy_loader import DummyModelLoader
from vllm.model_executor.model_loader.utils import (
    initialize_model,
    process_weights_after_loading,
)
from vllm.model_executor.model_loader.weight_utils import (
    initialize_single_dummy_weight,
)
from vllm.utils.system_utils import decorate_logs
from vllm.utils.torch_utils import set_default_torch_dtype
from vllm.v1.ple_offload.protocol import (
    _PLE_OFFLOAD_REQUEST_DECODER,
    PleOffloadRegistration,
    PleOffloadRequest,
)
from vllm.v1.ple_offload.test_fault import (
    _FAULT,
    _FaultReconstructionPayload,
)
from vllm.v1.utils import record_function_or_nullcontext

logger = init_logger(__name__)

_PUBLISH_FENCE = threading.Lock()


class _PleShutdownRequested(Exception):
    """Internal cancellation; never report an orderly stop as a failure."""


def _bind_private_ipc_socket(pull_socket: zmq.Socket, ipc_addr: str) -> None:
    """Create the endpoint owner-only, with no bind-to-chmod exposure window.

    Only the dedicated offload child calls this. Keep the configured address:
    all local PP/TP/DP processes of this user must connect to the same socket.
    """
    if not ipc_addr.startswith("ipc://") or not os.path.isabs(ipc_addr[6:]):
        raise ValueError("PLE requires an absolute filesystem IPC address")
    previous_umask = os.umask(0o177)
    try:
        pull_socket.bind(ipc_addr)
    finally:
        os.umask(previous_umask)


def _unlink_ipc_socket(ipc_addr: str) -> None:
    """Best-effort unlink of the exact filesystem IPC endpoint PLE owns."""
    if not ipc_addr.startswith("ipc://"):
        return
    path = ipc_addr.removeprefix("ipc://")
    # PLE receives an absolute UUID path from get_open_zmq_ipc_path(). Avoid
    # treating abstract or relative ZMQ addresses as filesystem ownership.
    if not os.path.isabs(path):
        return
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
    except OSError as error:
        logger.warning("Failed to unlink PLE IPC socket %s: %s", path, error)


_PROCESS_EXIT_IMMINENT = False
_EXIT_IMMINENT_CHILD_WAIT_S = 3.5
# procfs mount the predicate reads (tests point it at a fake tree).
_PROC_ROOT = "/proc"
# task_struct flag: set by do_exit() -> exit_signals() before exit_mm(); a task
# carrying it never returns to userspace.
_PF_EXITING = 0x4


def mark_process_exit_imminent() -> None:
    """This process os._exits right after its shutdown; see close()."""
    global _PROCESS_EXIT_IMMINENT
    _PROCESS_EXIT_IMMINENT = True


def _task_in_exit(stat: bytes) -> bool:
    """One /proc/<pid>/task/<tid>/stat: Z/X, or PF_EXITING (kernel exit path)."""
    # Fields after the parenthesised comm (which may contain spaces/parens):
    # [0]=state ... [6]=flags (stat field 9).
    fields = stat[stat.rfind(b")") + 2 :].split()
    if not fields:
        return True
    if fields[0] in (b"Z", b"X"):
        return True
    try:
        return bool(int(fields[6]) & _PF_EXITING)
    except (IndexError, ValueError):
        return False


def _child_thread_group_exiting(
    proc: subprocess.Popen, proc_root: str | None = None
) -> bool:
    """True once EVERY task of ``proc`` is exited or in the kernel exit path.

    A zombie leader alone is not enough (a thread can still run userspace
    after the leader exits). Never reaps: the child is unreaped, so its PID
    cannot be reused and <proc_root>/<pid>/task is its own. Off Linux, only a
    reaped child counts.
    """
    if proc.returncode is not None:
        return True
    task_dir = os.path.join(proc_root or _PROC_ROOT, str(proc.pid), "task")
    try:
        tids = os.listdir(task_dir)
    except FileNotFoundError:
        return True
    except OSError:
        return proc.poll() is not None
    for tid in tids:
        try:
            with open(os.path.join(task_dir, tid, "stat"), "rb") as fh:
                stat = fh.read()
        except FileNotFoundError:
            continue  # this task finished exiting between listdir and open
        except OSError:
            return False
        if not _task_in_exit(stat):
            return False
    return True


def _publish_fence() -> None:
    with _PUBLISH_FENCE:
        pass


def _offload_child_environment() -> dict[str, str]:
    """Build the offload environment without mutating the parent."""
    child_env = dict(os.environ)
    child_env["VLLM_DISABLE_SHARED_EXPERTS_STREAM"] = "1"
    # See _prepare_cuda_free_process item 1b: no GDN aux stream in the child.
    child_env["VLLM_GDN_BA_AUX_STREAM"] = "0"
    child_env.setdefault("CUTLASS_CUDA_DEVICE_ID", "0")
    child_env.pop("VLLM_PP_LAYER_PARTITION", None)

    home_device = child_env.get("VLLM_PLE_OFFLOAD_HOME_DEVICE", "").strip()
    if home_device:
        parent_visible = child_env.get("CUDA_VISIBLE_DEVICES", "").strip()
        try:
            physical = (
                parent_visible.split(",")[int(home_device)].strip()
                if parent_visible
                else home_device
            )
        except (IndexError, ValueError) as error:
            raise ValueError("Invalid VLLM_PLE_OFFLOAD_HOME_DEVICE") from error
        child_env["CUDA_VISIBLE_DEVICES"] = physical
    return child_env


def _prepare_cuda_free_process() -> None:
    """Keep the PLE offload process free of any CUDA context.

    The host-mapped pull design issues no CUDA work here, yet model
    construction reached CUDA incidentally:
    1. ``VLLM_DISABLE_SHARED_EXPERTS_STREAM=1``: each MoE layer's shared-experts
       runner calls ``aux_stream()`` -> ``torch.cuda.Stream()``, whose C++
       constructor creates the primary context without calling
       ``torch.cuda._lazy_init`` (``torch.cuda.is_initialized()`` stays False).
       This is the context a startup check caught (stack dumped inside
       ``Stream.__new__``). This process never runs an MoE layer, so it never
       needs the stream. vLLM reads the variable when the layer is built; this
       process does not enable vLLM's env cache.
    1b. ``VLLM_GDN_BA_AUX_STREAM=0``: the same for the GDN layers. With the
       parent's flag at 1 the meta-model construction would call
       ``aux_stream()`` per GDN layer, which the guard refuses; this process
       never runs a GDN projection, so the flag is forced off here only.
    2. ``CUTLASS_CUDA_DEVICE_ID``: CUTLASS python (``cutlass_cppgen``) runs
       ``initialize_cuda_context()`` -> ``cudart.cudaFree(0)`` at import unless
       it is set. Not observed on this import path; kept as a cheap defence.
    3. The torch guard (``_install_cuda_guard``).
    """
    os.environ["VLLM_DISABLE_SHARED_EXPERTS_STREAM"] = "1"
    os.environ["VLLM_GDN_BA_AUX_STREAM"] = "0"
    os.environ.setdefault("CUTLASS_CUDA_DEVICE_ID", "0")
    _install_cuda_guard()


def _install_cuda_guard() -> None:
    """Make torch-level CUDA initialization raise in the PLE offload process.

    An offload-startup check found the host-mapped pull still got a 322 MiB context
    on its home card: importing the Qwen4Exp model imports the vendored FLA
    ops, whose utils call triton's ``get_current_target()`` at import time,
    i.e. ``torch.cuda.current_device()`` -> lazy init. FLA catches
    ``RuntimeError`` and falls back to "cpu" (and then skips its Hopper
    probe), so raising here keeps this process context-free. Any other call
    site fails loudly at offload startup with its stack logged, instead of
    silently taking VRAM on a stage-0 card. The GPU workers are untouched.

    torch also has C++ entry points that create the context without calling
    ``torch.cuda._lazy_init``: stream construction (``torch.cuda.Stream``) and
    the pinned host allocator (``Tensor.pin_memory``). Both raise here too;
    this process needs neither (all of its buffers are plain CPU memory).
    ``torch.cuda.is_initialized()`` only reflects the Python-level path, so
    the per-PID GPU sampler, not that flag, is the authority on contexts.
    """
    import traceback

    if torch.cuda.is_initialized():
        # Fail closed: a context that already exists would stay on a
        # stage-0 card for the whole run.
        raise RuntimeError(
            "CUDA is already initialized in the PLE offload process before its "
            "guard; the host-mapped pull offload must stay CUDA-free"
        )

    def _refuse(what: str) -> None:
        stack = "".join(traceback.format_stack(limit=12)[:-2])
        logger.info(
            "PLE offload CUDA guard: blocked a CUDA initialization attempt "
            "(%s; the caller must fall back to CPU or fail); call site:\n%s",
            what,
            stack,
        )
        raise RuntimeError(
            f"CUDA is disabled in the PLE offload process (host-mapped pull): {what}"
        )

    def _blocked_lazy_init() -> None:
        _refuse("torch.cuda lazy init")

    def _blocked_stream(cls, *args, **kwargs):
        _refuse("torch.cuda.Stream construction")

    def _blocked_pin_memory(self, *args, **kwargs):
        _refuse("Tensor.pin_memory")

    torch.cuda._lazy_init = _blocked_lazy_init  # type: ignore[assignment]
    torch.cuda.Stream.__new__ = staticmethod(_blocked_stream)  # type: ignore[assignment]
    torch.Tensor.pin_memory = _blocked_pin_memory  # type: ignore[assignment]
    logger.info(
        "PLE offload CUDA guard installed (torch CUDA lazy init, stream "
        "construction and pin_memory raise)"
    )


@contextlib.contextmanager
def _platform_device_is_meta():
    """During the offload's meta-model construction, report the meta device.

    Some layer constructors pass an explicit ``current_platform.current_device()``
    (e.g. the GDN output norm in qwen_gdn_linear_attn.py), which overrides the
    ``torch.device("meta")`` context and would allocate on CUDA. In the offload
    process nothing but the CPU PLE layers is ever materialized, so those
    tensors belong on meta too. The patch is an instance attribute removed on
    exit, restoring the platform's own method.
    """
    from vllm.platforms import current_platform

    current_platform.current_device = lambda *args, **kwargs: torch.device("meta")
    try:
        yield
    finally:
        with contextlib.suppress(AttributeError):
            del current_platform.current_device


# Bounded wait for the GPU to release a result (seconds). A timed-out ownership
# transfer is fatal: the offload process raises and exits, never overwrites.
_PLE_WAIT_TIMEOUT_S = 120.0


@dataclass
class PleOffloadOutputTarget:
    """Host output destination and ready flag for one TP worker.

    Both are shared CPU tensors owned by the GPU worker, page-locked and
    device-mapped in that worker's context. This process only reads/writes
    them with plain CPU loads/stores (no CUDA): wait for ``flag == 0`` (the GPU
    released the previous result), write the rows, then store ``flag = 1``.
    """

    tp_rank: int
    host_output_buffer: torch.Tensor
    host_flag: torch.Tensor  # int32, element 0 is the flag


@dataclass
class PleOffloadInputBuffers:
    """Shared-memory input buffers registered for one DP rank."""

    input_ids_buf: torch.Tensor  # int32 (max_num_tokens,)
    query_start_loc_buf: torch.Tensor  # int32 (max_num_reqs + 1,)
    ngram_context_buf: torch.Tensor | None  # int32 (max_num_reqs, ngram_context_len)


@dataclass(frozen=True)
class _ReceivedPleOffloadRequest:
    request: PleOffloadRequest
    serving: bool
    transaction_id: str | None
    correlation_id: str | None


def _decode_offload_request(
    frames: list[bytes],
) -> _ReceivedPleOffloadRequest:
    if len(frames) != 2:
        raise RuntimeError("PLE offload request must contain two frames")
    request = _PLE_OFFLOAD_REQUEST_DECODER.decode(frames[0])
    metadata = msgspec.msgpack.decode(frames[1])
    if type(metadata) is not dict or set(metadata) != {
        "serving",
        "transaction",
        "correlation",
    }:
        raise RuntimeError("PLE offload request metadata is invalid")
    serving = metadata["serving"]
    transaction = metadata["transaction"]
    correlation = metadata["correlation"]
    if type(serving) is not bool:
        raise RuntimeError("PLE offload serving metadata is invalid")
    if transaction is not None and type(transaction) is not str:
        raise RuntimeError("PLE offload transaction metadata is invalid")
    if correlation is not None and type(correlation) is not str:
        raise RuntimeError("PLE offload correlation metadata is invalid")
    return _ReceivedPleOffloadRequest(
        request=request,
        serving=serving,
        transaction_id=transaction,
        correlation_id=correlation,
    )


def _receive_offload_request(
    socket: zmq.Socket, flags: int = 0
) -> _ReceivedPleOffloadRequest | PleOffloadRequest:
    """Disarmed: the unchanged single-frame request. Armed: request + metadata."""
    if _FAULT is None:
        return _PLE_OFFLOAD_REQUEST_DECODER.decode(socket.recv(flags))
    return _decode_offload_request(socket.recv_multipart(flags))


def _send_framed_payload(fd: int, payload: bytes, deadline: float) -> None:
    """Send one launch payload without blocking beyond an absolute deadline."""
    frame = memoryview(struct.pack("!Q", len(payload)) + payload)
    offset = 0
    os.set_blocking(fd, False)
    try:
        while offset < len(frame):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("PLE launch payload transmission timed out")
            _, writable, _ = select.select([], [fd], [], remaining)
            if not writable:
                raise TimeoutError("PLE launch payload transmission timed out")
            try:
                sent = os.write(fd, frame[offset:])
            except BlockingIOError:
                continue
            if sent <= 0:
                raise BrokenPipeError("PLE launch control socket closed")
            offset += sent
            if (
                _FAULT is not None
                and _FAULT.cut == "launch_park_before_payload_read"
            ):
                _FAULT.acknowledge_launch_send(
                    bytes_sent=offset,
                    frame_size=len(frame),
                )
    finally:
        with contextlib.suppress(OSError):
            os.set_blocking(fd, True)


_PLE_READY_MAX_PAYLOAD_BYTES = 1 << 20

class _BoundedStartupResponseWriter:
    """Preflight startup responses against the parent receiver's frame cap."""

    def __init__(self, writer: Connection) -> None:
        self._writer = writer

    def send(self, message: Any) -> None:
        payload = pickle.dumps(message, protocol=pickle.HIGHEST_PROTOCOL)
        if len(payload) > _PLE_READY_MAX_PAYLOAD_BYTES:
            payload = pickle.dumps(
                {
                    "status": "FAILURE",
                    "error": "PLE startup response exceeded size limit",
                },
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        if len(payload) > _PLE_READY_MAX_PAYLOAD_BYTES:
            raise RuntimeError(
                "Bounded PLE startup failure response exceeds its frame cap"
            )
        self._writer.send_bytes(payload)

    def close(self) -> None:
        self._writer.close()


def _read_exact_before_deadline(
    fd: int,
    size: int,
    deadline: float,
) -> bytes:
    """Read exactly ``size`` bytes without crossing an absolute deadline."""
    chunks = bytearray()
    while len(chunks) < size:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(
                "PLE offload worker did not become ready before its "
                "absolute launch deadline"
            )
        readable, _, _ = select.select([fd], [], [], remaining)
        if not readable:
            raise TimeoutError(
                "PLE offload worker did not become ready before its "
                "absolute launch deadline"
            )
        try:
            chunk = os.read(fd, size - len(chunks))
        except BlockingIOError:
            continue
        if not chunk:
            raise EOFError("PLE startup response was truncated")
        chunks.extend(chunk)
    return bytes(chunks)


def _recv_bounded_connection_message(
    reader: Connection,
    deadline: float,
) -> Any:
    """Receive one bounded multiprocessing Connection frame by ``deadline``."""
    fd = reader.fileno()
    os.set_blocking(fd, False)
    try:
        header = _read_exact_before_deadline(fd, 4, deadline)
        size = struct.unpack("!i", header)[0]
        if size == -1:
            size = struct.unpack(
                "!Q", _read_exact_before_deadline(fd, 8, deadline)
            )[0]
        elif size < 0:
            raise RuntimeError("PLE startup response has an invalid frame size")
        if size > _PLE_READY_MAX_PAYLOAD_BYTES:
            raise RuntimeError(
                f"PLE startup response exceeds "
                f"{_PLE_READY_MAX_PAYLOAD_BYTES} bytes"
            )
        frame = _read_exact_before_deadline(fd, size, deadline)
    finally:
        with contextlib.suppress(OSError):
            os.set_blocking(fd, True)
    try:
        return pickle.loads(frame)
    except BaseException as error:
        raise RuntimeError(
            "PLE startup response is not a valid serialized message"
        ) from error


@dataclass
class PleOffloadWorkerHandle:
    """Resources owned by the GPU worker that spawned the offload process."""

    proc: subprocess.Popen
    death_writer: Connection | None
    ready_pipe_reader: Connection | None
    launch_deadline: float
    ipc_addr: str | None = None

    def close(self, *, graceful: bool = True) -> None:
        """Release all process resources. Safe to call more than once."""
        if self.ready_pipe_reader is not None:
            self.ready_pipe_reader.close()
            self.ready_pipe_reader = None
        if graceful and _PROCESS_EXIT_IMMINENT:
            self._close_without_reap()
            return
        if graceful and self.proc.poll() is None:
            # Keep parent-death supervision connected while SIGTERM lets
            # proc_main run its shutdown event and finally cleanup.
            self.proc.terminate()
            with contextlib.suppress(subprocess.TimeoutExpired):
                self.proc.wait(timeout=5)
        if self.death_writer is not None:
            self.death_writer.close()
            self.death_writer = None
        if not graceful and self.proc.poll() is None:
            # Expired launch: retain the immediate EOF -> TERM path.
            self.proc.terminate()
            with contextlib.suppress(subprocess.TimeoutExpired):
                self.proc.wait(timeout=5)
        if self.proc.poll() is None:
            self.proc.kill()
            with contextlib.suppress(subprocess.TimeoutExpired):
                self.proc.wait(timeout=5)
        if self.proc.poll() is not None and self.ipc_addr is not None:
            _unlink_ipc_socket(self.ipc_addr)

    def _close_without_reap(self) -> None:
        """close() for a parent that os._exits next: wait (bounded) until every
        child task has left userspace, never for the reap.

        SIGTERM lets proc_main's finally unlink the IPC socket, as in the
        graceful path; parent-death supervision stays connected meanwhile.
        Once the child has run exit, the address-space teardown that keeps it
        unreapable needs nothing from this process: the orphan is reaped by
        the subreaper/init after this process exits.
        """
        if not _child_thread_group_exiting(self.proc):
            with contextlib.suppress(ProcessLookupError):
                os.kill(self.proc.pid, signal.SIGTERM)
            deadline = time.monotonic() + _EXIT_IMMINENT_CHILD_WAIT_S
            while not _child_thread_group_exiting(self.proc):
                if time.monotonic() >= deadline:
                    logger.warning(
                        "PLE offload process %d still running %.1f s after "
                        "SIGTERM; killing it without waiting (parent exits)",
                        self.proc.pid,
                        _EXIT_IMMINENT_CHILD_WAIT_S,
                    )
                    with contextlib.suppress(ProcessLookupError):
                        os.kill(self.proc.pid, signal.SIGKILL)
                    break
                time.sleep(0.01)
        exited = _child_thread_group_exiting(self.proc)
        if self.death_writer is not None:
            self.death_writer.close()
            self.death_writer = None
        # Same fallback as close(): the child has run exit, so this exact
        # PLE-owned address is no longer in use.
        if exited and self.ipc_addr is not None:
            _unlink_ipc_socket(self.ipc_addr)


def _init_offload_distributed() -> None:
    """Initialize the single-rank Gloo world required by TP-aware layers."""
    if dist.is_initialized():
        return

    store_dir = tempfile.mkdtemp(prefix="vllm_ple_offload_")
    init_distributed_environment(
        world_size=1,
        rank=0,
        distributed_init_method=f"file://{store_dir}/store",
        local_rank=0,
        backend="gloo",
    )
    offload_config = VllmConfig()
    offload_parallel_config = offload_config.parallel_config
    offload_parallel_config.data_parallel_size = 1
    offload_parallel_config.data_parallel_size_local = 1
    offload_parallel_config.data_parallel_rank = 0
    offload_parallel_config.data_parallel_rank_local = 0
    offload_parallel_config.data_parallel_index = 0
    offload_parallel_config.tensor_parallel_size = 1
    offload_parallel_config.pipeline_parallel_size = 1
    offload_parallel_config.prefill_context_parallel_size = 1
    offload_parallel_config.decode_context_parallel_size = 1
    offload_parallel_config.world_size = 1
    offload_parallel_config.nnodes = 1
    offload_parallel_config.node_rank = 0
    with set_current_vllm_config(offload_config):
        ensure_model_parallel_initialized(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            backend="gloo",
        )
    logger.info(
        "Distributed environment initialized (backend=gloo, rank=0, world_size=1)."
    )


def _offload_construction_config(vllm_config: VllmConfig) -> VllmConfig:
    """The config the child builds its discovery (meta) model under.

    The child runs a TP1/DP1/PP1 Gloo world, but the real config may enable
    TP/DP/EP features whose constructors read those sizes back from the
    process groups. Sequence-parallel MoE (TP>1, DP>1, EP and an SP all2all
    backend) is one: the MoE block takes ``use_sequence_parallel_moe=True``
    from the config while ``FusedMoE`` derives SP from the TP1 group, and
    asserts they agree. It also turns on GR/PLE SP. Build under a copy whose
    parallel config matches the child's world, with EP, EPLB and HC SP off.
    The PLE subtrees are unaffected: their table and hashing do not depend
    on these sizes, and the child never shards their rows. The real config
    stays on the runner for registrations and request routing.
    """
    construction_config = copy.copy(vllm_config)
    parallel_config = copy.copy(vllm_config.parallel_config)
    for name, value in (
        ("tensor_parallel_size", 1),
        ("pipeline_parallel_size", 1),
        ("data_parallel_size", 1),
        ("data_parallel_size_local", 1),
        ("data_parallel_rank", 0),
        ("data_parallel_rank_local", 0),
        ("data_parallel_index", 0),
        ("prefill_context_parallel_size", 1),
        ("decode_context_parallel_size", 1),
        ("world_size", 1),
        ("enable_expert_parallel", False),
        ("enable_eplb", False),
        ("enable_hc_sp", False),
    ):
        object.__setattr__(parallel_config, name, value)
    object.__setattr__(construction_config, "parallel_config", parallel_config)
    return construction_config


class PleOffloadWorker:
    """Manage process creation, READY handshake, and the child entry point."""

    READY_STR = "READY"

    @staticmethod
    def make_process(
        vllm_config: VllmConfig,
        num_workers: int,
        ipc_addr: str,
    ) -> PleOffloadWorkerHandle:
        """Start one CPU offload process in a fresh interpreter."""
        launch_deadline = (
            time.monotonic() + float(envs.VLLM_PLE_OFFLOAD_READY_TIMEOUT)
        )
        config_payload = pickle.dumps(
            vllm_config, protocol=pickle.HIGHEST_PROTOCOL
        )
        child_env = _offload_child_environment()

        # READY and parent-death supervision must never compete for reads.
        parent_control, child_control = socket.socketpair()
        parent_death, child_death = socket.socketpair()
        ready_reader = Connection(parent_control.detach())
        death_writer = Connection(parent_death.detach())
        try:
            proc = subprocess.Popen(
                [
                    sys.executable,
                    # Omit the implicit cwd import path, retaining PYTHONPATH installs.
                    "-P",
                    "-m",
                    "vllm_ple_offload_bootstrap",
                    str(child_control.fileno()),
                    str(child_death.fileno()),
                ],
                env=child_env,
                # The owning GPU worker drains consumers before stopping us.
                # Group SIGINT/SIGTERM must not tear down the producer early.
                start_new_session=True,
                close_fds=True,
                pass_fds=(child_control.fileno(), child_death.fileno()),
            )
        except BaseException:
            ready_reader.close()
            death_writer.close()
            raise
        finally:
            child_control.close()
            child_death.close()

        handle = PleOffloadWorkerHandle(
            proc=proc,
            death_writer=death_writer,
            ready_pipe_reader=ready_reader,
            launch_deadline=launch_deadline,
            ipc_addr=ipc_addr,
        )
        try:
            config_for_launch: bytes | _FaultReconstructionPayload = config_payload
            if (
                _FAULT is not None
                and _FAULT.cut == "launch_block_reconstruction"
            ):
                config_for_launch = _FaultReconstructionPayload(config_payload)
            launch_state = {
                "vllm_config_payload": config_for_launch,
                "num_workers": num_workers,
                "ipc_addr": ipc_addr,
            }
            if (
                _FAULT is not None
                and _FAULT.cut == "launch_park_before_payload_read"
            ):
                # Larger than ordinary AF_UNIX socket buffers: an acknowledged
                # non-reading child leaves the parent inside framed transmission.
                launch_state["_test_padding"] = b"\0" * (8 << 20)
            launch_payload = pickle.dumps(
                launch_state,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
            _send_framed_payload(
                ready_reader.fileno(), launch_payload, launch_deadline
            )
        except BaseException:
            handle.close(graceful=False)
            raise

        return handle

    @staticmethod
    def wait_for_ready(handle: PleOffloadWorkerHandle) -> None:
        """Wait until weights and all GPU registrations are ready to serve."""
        reader = handle.ready_pipe_reader
        if reader is None:
            return
        try:
            message = _recv_bounded_connection_message(
                reader, handle.launch_deadline
            )
            if type(message) is not dict:
                raise RuntimeError("PLE offload worker sent an invalid response")
            status = dict.get(message, "status")
            if type(status) is not str:
                raise RuntimeError(
                    "PLE offload worker sent an invalid response status"
                )
            if status == PleOffloadWorker.READY_STR:
                pass
            elif status == "FAILURE":
                error_detail = dict.get(message, "error")
                if type(error_detail) is not str:
                    error_detail = "unknown error"
                raise RuntimeError(
                    "PLE offload worker failed during startup: "
                    f"{error_detail}"
                )
            else:
                raise RuntimeError("PLE offload worker sent unsupported status")
            layer_names = dict.get(message, "layer_names")
            if (
                type(layer_names) is not list
                or any(
                    type(name) is not str for name in layer_names
                )
            ):
                raise RuntimeError(
                    "PLE offload worker sent invalid READY layer_names"
                )
            logger.info(
                "Worker ready - %d PleOffloadLayer(s): %s",
                len(layer_names),
                layer_names,
            )
        except EOFError as error:
            handle.close(graceful=False)
            raise RuntimeError("PLE offload worker exited during startup") from error
        except BaseException:
            handle.close(graceful=False)
            raise
        finally:
            if handle.ready_pipe_reader is not None:
                handle.ready_pipe_reader.close()
                handle.ready_pipe_reader = None

    @staticmethod
    def proc_main(
        vllm_config: VllmConfig,
        num_workers: int,
        ipc_addr: str,
        ready_pipe: Connection,
    ) -> None:
        """Load PLE weights, accept registrations, and run the request loop."""
        decorate_logs("PleOffloadWorker")
        ready_writer = _BoundedStartupResponseWriter(ready_pipe)
        shutdown_event = threading.Event()

        def handle_signal(_signum: int, _frame: object) -> None:
            shutdown_event.set()

        signal.signal(signal.SIGTERM, handle_signal)
        signal.signal(signal.SIGINT, handle_signal)

        zmq_context: zmq.Context | None = None
        pull_socket: zmq.Socket | None = None
        owns_ipc_socket = False
        try:
            # The flag lets PleOffloadLayer subclasses execute their complete
            # constructors instead of becoming empty GPU-worker placeholders.
            mark_as_offload_worker()

            _prepare_cuda_free_process()
            home_device = os.environ.get("VLLM_PLE_OFFLOAD_HOME_DEVICE", "").strip()
            if home_device:
                logger.info(
                    "PLE offload bootstrap visibility is CUDA_VISIBLE_DEVICES=%s "
                    "(host-mapped pull: no CUDA context expected)",
                    os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                )

            os.environ.pop("VLLM_PP_LAYER_PARTITION", None)
            _init_offload_distributed()

            # Model components read the active VllmConfig while the meta model
            # is constructed, so keep the context around runner initialization.
            with set_current_vllm_config(vllm_config):
                runner = PleOffloadRunner(vllm_config)

            zmq_context = zmq.Context()
            pull_socket = zmq_context.socket(zmq.PULL)
            _bind_private_ipc_socket(pull_socket, ipc_addr)
            owns_ipc_socket = True
            logger.info(
                "Bound IPC address %s; waiting for %d GPU worker registration(s).",
                ipc_addr,
                num_workers,
            )

            # READY means that the process can immediately serve requests. Wait
            # for every DP/TP worker to register before notifying the parent.
            runner.accept_registrations(pull_socket, num_workers, shutdown_event)
            if shutdown_event.is_set():
                raise _PleShutdownRequested
            # Checked: the host-mapped pull design must not create a context.
            logger.info(
                "PLE offload host-mapped pull: CUDA initialized in this process = %s",
                torch.cuda.is_initialized(),
            )
            ready_writer.send(
                {
                    "status": PleOffloadWorker.READY_STR,
                    "layer_names": sorted(runner.layer_names),
                }
            )
            ready_writer.close()
            ready_writer = None  # type: ignore[assignment]

            runner.busy_loop(pull_socket, shutdown_event)
        except _PleShutdownRequested:
            logger.debug("PLE offload shutdown requested")
        except Exception as error:
            logger.exception("Unexpected failure in PLE offload worker.")
            if ready_writer is not None:
                with contextlib.suppress(Exception):
                    ready_writer.send(
                        {"status": "FAILURE", "error": repr(error)[:4096]}
                    )
            raise
        finally:
            if pull_socket is not None:
                pull_socket.close(linger=0)
            if zmq_context is not None:
                zmq_context.term()
            if owns_ipc_socket:
                _unlink_ipc_socket(ipc_addr)
            if ready_writer is not None:
                ready_writer.close()


class PleOffloadRunner:
    """Own all discovered PLE tables and serve every local DP rank."""

    def __init__(self, vllm_config: VllmConfig) -> None:
        disabled = [
            name
            for name in (
                "VLLM_E2_PLE_FAST_IDS",
                "VLLM_E2_PLE_EARLY_REQUEST",
            )
            if not getattr(envs, name)
        ]
        if disabled:
            logger.warning(
                "PLE offload is using the slower pull path: %s disabled. "
                "Set VLLM_E2_PLE_FAST_IDS=1 and VLLM_E2_PLE_EARLY_REQUEST=1 "
                "to match the Merlin launcher.",
                ", ".join(disabled),
            )
        self.vllm_config = vllm_config
        self._clamp_input_ids = (
            getattr(vllm_config, "speculative_config", None) is not None
        )
        # name -> PleOffloadLayer (CPU)
        self._layers: dict[str, PleOffloadLayer] = {}
        # dp_rank -> layer_name -> one destination per TP rank
        self._worker_targets: dict[int, dict[str, list[PleOffloadOutputTarget]]] = {}
        self._scratch_bufs: dict[int, dict[str, torch.Tensor]] = {}
        self._shutdown_event: threading.Event | None = None
        self._step = 0  # number of handled request batches (diagnostics)
        self._request_index = 0
        # Shared-memory inputs are registered once per DP rank by TP rank zero.
        self._input_bufs: dict[int, PleOffloadInputBuffers] = {}
        self._load_weights()

    @property
    def layer_names(self) -> list[str]:
        """Return PleOffloadLayer names in model traversal order."""
        return list(self._layers)

    def _load_weights(self) -> None:
        """Load only :class:`PleOffloadLayer` subtrees into CPU memory.

        Strategy:
        1. Build the entire model on ``meta`` so non-offloaded parameters use no
           physical memory. PleOffloadLayer constructors explicitly target CPU.
        2. Discover all PleOffloadLayer modules from the complete model.
        3. Stream the checkpoint through a prefix filter so only matching PLE
           tensors are materialized and passed to ``model.load_weights``.
        4. Run post-load processing only on the CPU-owned PLE subtrees.
        """
        # The discovery model is built (and loaded) under a config matching
        # this process's TP1/DP1/PP1 world; see _offload_construction_config.
        construction_config = _offload_construction_config(self.vllm_config)
        with set_current_vllm_config(construction_config):
            self._load_weights_under(construction_config)

    def _load_weights_under(self, construction_config: VllmConfig) -> None:
        """``_load_weights`` body, under the child-local construction config."""
        model_config = construction_config.model_config
        load_config = construction_config.load_config

        # Step 1: build complete structure, while only PLE subtrees allocate CPU
        # memory. All transformer, MoE, and vision parameters remain on meta.
        logger.info("Initializing model structure for PLE weight discovery ...")
        model_dtype = cast(torch.dtype, model_config.dtype)
        with (
            set_default_torch_dtype(model_dtype),
            torch.device("meta"),
            _platform_device_is_meta(),
        ):
            model = initialize_model(
                vllm_config=construction_config,
                model_config=model_config,
            )

        # Step 2: preserve named_modules DFS order so CPU execution follows the
        # same layer order as the GPU model forward.
        offload_layers = {
            name: module
            for name, module in model.named_modules()
            if isinstance(module, PleOffloadLayer)
        }
        if not offload_layers:
            raise RuntimeError(
                "The PLE pull transport is on, but no PleOffloadLayer "
                "was found in the initialized model"
            )
        logger.info(
            "Found %d PleOffloadLayer(s): %s",
            len(offload_layers),
            sorted(offload_layers),
        )
        offload_prefixes = tuple(f"{name}." for name in offload_layers)

        mapper = getattr(model, "hf_to_vllm_mapper", None)
        matched_checkpoint_tensors = 0

        def offload_only_iter(
            weights: Iterable[tuple[str, torch.Tensor]],
        ) -> Iterable[tuple[str, torch.Tensor]]:
            nonlocal matched_checkpoint_tensors
            for weight_name, tensor in weights:
                mapped_name: str | None = weight_name
                if mapper is not None:
                    mapped_names = mapper.apply_list([weight_name])
                    mapped_name = mapped_names[0] if mapped_names else None
                if mapped_name is not None and mapped_name.startswith(offload_prefixes):
                    matched_checkpoint_tensors += 1
                    yield weight_name, tensor

        loader = get_model_loader(load_config)
        if isinstance(loader, DummyModelLoader):
            logger.info(
                "Initializing dummy weights for %d PleOffloadLayer(s) ...",
                len(offload_layers),
            )
            # 0.32 removed `initialize_dummy_weights(model, model_config)`; its
            # body was exactly this loop with the same defaults.
            for layer in offload_layers.values():
                for param in layer.state_dict().values():
                    initialize_single_dummy_weight(param)
        elif isinstance(loader, DefaultModelLoader):
            all_weights = loader.get_all_weights(model_config, model)
            loaded_params = model.load_weights(offload_only_iter(all_weights))
            if matched_checkpoint_tensors == 0:
                raise RuntimeError(
                    "PLE offload checkpoint filter matched no weights for "
                    f"layers: {sorted(offload_layers)}"
                )

            expected_offload_params = {
                f"{layer_name}.{param_name}"
                for layer_name, layer in offload_layers.items()
                for param_name, _ in layer.named_parameters()
            }
            loaded_offload_entries = {
                name for name in loaded_params if name.startswith(offload_prefixes)
            }
            loaded_expected_params = expected_offload_params.intersection(loaded_params)
            missing_offload_params = sorted(
                expected_offload_params.difference(loaded_expected_params)
            )
            if missing_offload_params:
                raise RuntimeError(
                    "PLE offload checkpoint did not load all materialized "
                    f"parameters: {missing_offload_params}"
                )
            logger.info(
                "PLE offload matched %d checkpoint tensor(s), loaded %d "
                "offload entries, and verified %d/%d materialized "
                "parameter(s) for layers: %s",
                matched_checkpoint_tensors,
                len(loaded_offload_entries),
                len(loaded_expected_params),
                len(expected_offload_params),
                sorted(offload_layers),
            )
        else:
            raise NotImplementedError(
                "PLE offload requires the default or dummy model loader, got "
                f"{type(loader).__name__}"
            )

        # Step 4: post-load processing is restricted to CPU-owned PLE modules;
        # the remainder of the model is still on meta and must not be visited.
        for layer in offload_layers.values():
            process_weights_after_loading(layer, model_config, torch.device("cpu"))

        self._layers.update(offload_layers)
        del model
        logger.info("PLE weight loading complete.")

    def accept_registrations(
        self,
        pull_socket: zmq.Socket,
        num_workers: int,
        shutdown_event: threading.Event | None = None,
    ) -> None:
        """Receive every local DP/TP worker's IPC and shared-memory buffers."""
        logger.info("Waiting for %d GPU worker registration(s) ...", num_workers)
        registrations: list[PleOffloadRegistration] = []
        for index in range(num_workers):
            while True:
                if shutdown_event is not None and shutdown_event.is_set():
                    raise _PleShutdownRequested
                if pull_socket.poll(timeout=100, flags=zmq.POLLIN):
                    break
            if shutdown_event is not None and shutdown_event.is_set():
                raise _PleShutdownRequested
            item = pickle.loads(pull_socket.recv())
            if not isinstance(item, PleOffloadRegistration):
                raise RuntimeError(
                    "Expected PleOffloadRegistration during setup, got "
                    f"{type(item).__name__} ({index + 1}/{num_workers})"
                )
            registrations.append(item)
            logger.info(
                "GPU worker %d registered (dp_rank=%d, tp_rank=%d, layers=%s).",
                item.worker_id,
                item.dp_rank,
                item.tp_rank,
                sorted(item.host_output_buffers),
            )

        dp_size = self.vllm_config.parallel_config.data_parallel_size
        tp_size = self.vllm_config.parallel_config.tensor_parallel_size
        if num_workers != dp_size * tp_size:
            raise RuntimeError(
                f"Expected {dp_size * tp_size} registrations for DP={dp_size}, "
                f"TP={tp_size}, got {num_workers}"
            )

        registrations_by_dp: dict[int, list[PleOffloadRegistration]] = {}
        for registration in registrations:
            registrations_by_dp.setdefault(registration.dp_rank, []).append(
                registration
            )
        if set(registrations_by_dp) != set(range(dp_size)):
            raise RuntimeError(
                f"Expected DP ranks {set(range(dp_size))}, "
                f"got {set(registrations_by_dp)}"
            )
        for dp_rank, dp_registrations in registrations_by_dp.items():
            tp_ranks = {registration.tp_rank for registration in dp_registrations}
            if tp_ranks != set(range(tp_size)):
                raise RuntimeError(
                    f"DP rank {dp_rank} expected TP ranks {set(range(tp_size))}, "
                    f"got {tp_ranks}"
                )

        for registration in registrations:
            if set(registration.host_output_buffers) != set(self.layer_names) or set(
                registration.host_flag_tensors
            ) != set(self.layer_names):
                raise RuntimeError(
                    "Registered PLE layers do not match CPU layers: "
                    f"registered={sorted(registration.host_output_buffers)} / "
                    f"{sorted(registration.host_flag_tensors)}, "
                    f"cpu={sorted(self.layer_names)}"
                )
            targets_for_dp = self._worker_targets.setdefault(registration.dp_rank, {})
            for layer_name, host_buffer in registration.host_output_buffers.items():
                host_flag = registration.host_flag_tensors[layer_name]
                for tensor in (host_buffer, host_flag):
                    if tensor.device.type != "cpu" or not tensor.is_shared():
                        raise RuntimeError(
                            f"PLE host buffer for {layer_name} (dp="
                            f"{registration.dp_rank}, tp={registration.tp_rank}) "
                            "is not a shared CPU tensor"
                        )
                if host_flag.dtype != torch.int32 or host_flag.numel() < 1:
                    raise RuntimeError(f"PLE host flag for {layer_name} must be int32")
                target = PleOffloadOutputTarget(
                    tp_rank=registration.tp_rank,
                    host_output_buffer=host_buffer,
                    host_flag=host_flag,
                )
                targets_for_dp.setdefault(layer_name, []).append(target)
            # All TP ranks in one DP group receive the same input, so buffers
            # registered by TP rank zero are sufficient for that DP rank.
            if registration.tp_rank == 0:
                self._input_bufs[registration.dp_rank] = PleOffloadInputBuffers(
                    input_ids_buf=registration.input_ids_buf,
                    query_start_loc_buf=registration.query_start_loc_buf,
                    ngram_context_buf=registration.ngram_context_buf,
                )

        if set(self._input_bufs) != set(range(dp_size)):
            raise RuntimeError(
                "TP rank zero did not register PLE input buffers for every DP "
                f"rank: expected={set(range(dp_size))}, got={set(self._input_bufs)}"
            )

        config = self.vllm_config.model_config.hf_text_config
        max_tokens = self.vllm_config.scheduler_config.max_num_batched_tokens
        for dp_rank, layer_targets in self._worker_targets.items():
            self._scratch_bufs[dp_rank] = {}
            for layer_name, targets in layer_targets.items():
                if len(targets) != tp_size:
                    raise RuntimeError(
                        f"PLE layer {layer_name} for DP rank {dp_rank} received "
                        f"{len(targets)} targets, expected {tp_size}"
                    )
                targets.sort(key=lambda target: target.tp_rank)
                out_dim = self._layers[layer_name].get_offload_output_dim(
                    int(config.ple_embed_dim)
                )
                out_dtype = self._layers[layer_name].get_offload_output_dtype(
                    self.vllm_config.model_config.dtype
                )
                for target in targets:
                    shape = tuple(target.host_output_buffer.shape)
                    if shape != (max_tokens, out_dim) or (
                        target.host_output_buffer.dtype != out_dtype
                    ):
                        raise RuntimeError(
                            f"PLE host buffer for {layer_name} (dp={dp_rank}, "
                            f"tp={target.tp_rank}) is {shape} "
                            f"{target.host_output_buffer.dtype}, expected "
                            f"{(max_tokens, out_dim)} {out_dtype}"
                        )
                self._scratch_bufs[dp_rank][layer_name] = torch.empty(
                    max_tokens, out_dim, dtype=out_dtype, device="cpu"
                )
        logger.info(
            "Registrations complete (dp_size=%d, tp_size=%d, layers=%s).",
            dp_size,
            tp_size,
            sorted(self.layer_names),
        )

    @torch.inference_mode()
    def busy_loop(
        self,
        pull_socket: zmq.Socket,
        shutdown_event: threading.Event,
    ) -> None:
        """Decode and batch available requests by DP rank until shutdown."""
        logger.info("Busy-loop started.")
        self._shutdown_event = shutdown_event
        poller = zmq.Poller()
        poller.register(pull_socket, zmq.POLLIN)
        while not shutdown_event.is_set():
            if pull_socket not in dict(poller.poll(timeout=100)):
                continue

            requests = []
            try:
                requests.append(_receive_offload_request(pull_socket))
                while True:
                    requests.append(
                        _receive_offload_request(pull_socket, zmq.NOBLOCK)
                    )
            except zmq.Again:
                pass
            except (msgspec.DecodeError, RuntimeError) as error:
                raise RuntimeError("Unexpected PLE offload request") from error

            self._handle_requests(requests)

    def _handle_requests(
        self, requests: list[_ReceivedPleOffloadRequest]
    ) -> None:
        """Run requests layer-first so each DP rank can resume promptly."""
        with record_function_or_nullcontext("ple_offload: handle_requests"):
            self._handle_requests_impl(requests)

    def _validate_requests_before_publication(
        self, requests: list[PleOffloadRequest]
    ) -> None:
        """Validate a drained request batch before computation or publication."""
        for request in requests:
            dp_rank = request.dp_rank
            if dp_rank not in self._worker_targets or dp_rank not in self._input_bufs:
                raise RuntimeError(
                    f"No PLE buffers or output targets for dp_rank={dp_rank}; "
                    "refusing request"
                )

            input_bufs = self._input_bufs[dp_rank]
            if not 0 < request.num_tokens <= input_bufs.input_ids_buf.numel():
                raise RuntimeError(
                    f"PLE num_tokens={request.num_tokens} is outside capacity "
                    f"{input_bufs.input_ids_buf.numel()} for dp_rank={dp_rank}"
                )
            max_reqs = input_bufs.query_start_loc_buf.numel() - 1
            if not 0 < request.num_reqs <= max_reqs:
                raise RuntimeError(
                    f"PLE num_reqs={request.num_reqs} is outside capacity "
                    f"{max_reqs} for dp_rank={dp_rank}"
                )
            if (
                input_bufs.query_start_loc_buf.dtype != torch.int32
                or input_bufs.query_start_loc_buf.dim() != 1
            ):
                raise RuntimeError(
                    f"PLE query_start_loc for dp_rank={dp_rank} must be 1-D int32"
                )
            if (
                input_bufs.ngram_context_buf is not None
                and request.num_reqs > input_bufs.ngram_context_buf.shape[0]
            ):
                raise RuntimeError(
                    f"PLE num_reqs={request.num_reqs} exceeds ngram context "
                    f"capacity for dp_rank={dp_rank}"
                )

            query_start_loc = input_bufs.query_start_loc_buf[
                : request.num_reqs + 1
            ].tolist()
            if (
                query_start_loc[0] != 0
                or query_start_loc[-1] > request.num_tokens
                or any(a > b for a, b in zip(query_start_loc, query_start_loc[1:]))
            ):
                raise RuntimeError(
                    f"Invalid PLE query_start_loc for dp_rank={dp_rank}: "
                    f"{query_start_loc}, num_tokens={request.num_tokens}"
                )

            for layer_name, targets in self._worker_targets[dp_rank].items():
                scratch = self._scratch_bufs[dp_rank][layer_name]
                if request.num_tokens > scratch.shape[0] or any(
                    request.num_tokens > target.host_output_buffer.shape[0]
                    for target in targets
                ):
                    raise RuntimeError(
                        f"PLE num_tokens={request.num_tokens} exceeds output "
                        f"capacity for {layer_name}, dp_rank={dp_rank}"
                    )

    def _handle_requests_impl(
        self,
        requests: list[_ReceivedPleOffloadRequest | PleOffloadRequest],
    ) -> None:
        """Process one batch of PLE requests."""
        received = [
            item
            if isinstance(item, _ReceivedPleOffloadRequest)
            else _ReceivedPleOffloadRequest(item, False, None, None)
            for item in requests
        ]
        plain_requests = [item.request for item in received]
        self._validate_requests_before_publication(plain_requests)
        received_by_dp: dict[int, _ReceivedPleOffloadRequest] = {}
        for item in received:
            request = item.request
            if request.dp_rank not in self._worker_targets:
                raise RuntimeError(
                    f"No PLE output targets for dp_rank={request.dp_rank}; "
                    "refusing request"
                )
            if request.dp_rank in received_by_dp:
                raise RuntimeError(
                    f"Two PLE requests for dp_rank={request.dp_rank} in one batch "
                    f"(step {self._step}); the single-slot protocol is violated"
                )
            received_by_dp[request.dp_rank] = item
        self._step += 1
        if _FAULT is not None:
            for item in received_by_dp.values():
                if not item.serving:
                    continue
                self._request_index += 1
                transaction = f"request:{self._request_index}"
                if item.transaction_id is not None:
                    transaction = f"{transaction}@{item.transaction_id}"
                for cut in ("park_before_publish", "park_before_publish_ack"):
                    if _FAULT is not None and _FAULT.fires(
                        cut,
                        role="offload",
                        unit="offload-request",
                        eligible=True,
                        dp_rank=item.request.dp_rank,
                        transaction_id=transaction,
                        correlation_id=item.correlation_id,
                    ):
                        # No multiprocessing or fault-state lock is held here.
                        threading.Event().wait()

        requests_by_dp = {
            dp_rank: item.request
            for dp_rank, item in received_by_dp.items()
        }

        # Speculative placeholders are not vocabulary IDs. Normalize each DP
        # input once before all PLE layers consume the shared buffer.
        if self._clamp_input_ids:
            for dp_rank, request in requests_by_dp.items():
                self._input_bufs[dp_rank].input_ids_buf[
                    : request.num_tokens
                ].clamp_min_(0)

        # The input packing and shifts depend only on request data and are
        # reused by every Qwen4Exp PLE layer in this serialized transaction.
        prepared_ngram_ids_by_dp: dict[int, Any] = {}
        first_layer = next(iter(self._layers.values()), None)
        prepare_ngram_ids = getattr(
            first_layer, "_prepare_ngram_ids_offload", None
        )
        if prepare_ngram_ids is not None:
            for dp_rank, request in requests_by_dp.items():
                input_bufs = self._input_bufs[dp_rank]
                ngram_context = (
                    input_bufs.ngram_context_buf[: request.num_reqs]
                    if input_bufs.ngram_context_buf is not None
                    else None
                )
                prepared_ngram_ids_by_dp[dp_rank] = prepare_ngram_ids(
                    input_bufs.input_ids_buf[: request.num_tokens],
                    input_bufs.query_start_loc_buf[: request.num_reqs + 1],
                    ngram_context,
                )

        for layer_name, layer in self._layers.items():
            for dp_rank, request in requests_by_dp.items():
                targets = self._worker_targets[dp_rank][layer_name]

                input_bufs = self._input_bufs[dp_rank]
                ngram_context = (
                    input_bufs.ngram_context_buf[: request.num_reqs]
                    if input_bufs.ngram_context_buf is not None
                    else None
                )
                prepared_ngram_ids = prepared_ngram_ids_by_dp.get(dp_rank)
                if prepared_ngram_ids is not None:
                    if not hasattr(layer, "_compute_ngram_ids_from_common"):
                        raise RuntimeError(
                            "PLE layers do not share a compatible n-gram ID "
                            "preparation path"
                        )
                    result = layer.forward_impl(
                        input_bufs.input_ids_buf[: request.num_tokens],
                        input_bufs.input_ids_buf[: request.num_tokens],
                        input_bufs.query_start_loc_buf[: request.num_reqs + 1],
                        ngram_context,
                        output_buffer=self._scratch_bufs[dp_rank][layer_name],
                        prepared_ngram_ids=prepared_ngram_ids,
                    )
                else:
                    result = layer.forward_impl(
                        input_bufs.input_ids_buf[: request.num_tokens],
                        input_bufs.input_ids_buf[: request.num_tokens],
                        input_bufs.query_start_loc_buf[: request.num_reqs + 1],
                        ngram_context,
                        output_buffer=self._scratch_bufs[dp_rank][layer_name],
                    )
                scratch = self._scratch_bufs[dp_rank][layer_name]
                if result.dim() != 2 or result.shape[0] != request.num_tokens or (
                    result.shape[1] != scratch.shape[1]
                ) or result.dtype != scratch.dtype or result.device.type != "cpu":
                    raise RuntimeError(
                        f"PLE result {tuple(result.shape)} {result.dtype} "
                        f"{result.device} for {layer_name} does not "
                        f"match the request ({request.num_tokens} rows x "
                        f"{scratch.shape[1]} {scratch.dtype} CPU); refusing to "
                        f"publish (step {self._step})"
                    )
                slices = tuple(slice(0, size) for size in result.shape)
                for target in targets:
                    self._wait_flag_released(target, layer_name, dp_rank)
                    target.host_output_buffer[slices].copy_(result[slices])
                    _publish_fence()
                    target.host_flag[0] = 1

    def _wait_flag_released(
        self, target: PleOffloadOutputTarget, layer_name: str, dp_rank: int
    ) -> None:
        """Spin until the GPU reset this target's flag (previous result used)."""
        flag = target.host_flag
        if int(flag[0]) == 0:
            return
        spins = 0
        warned = False
        start = time.monotonic()
        while int(flag[0]) != 0:
            spins += 1
            if self._shutdown_event is not None and self._shutdown_event.is_set():
                raise _PleShutdownRequested
            if spins > 200:
                time.sleep(0.0001)
                waited = time.monotonic() - start
                if not warned and waited > 5.0:
                    warned = True
                    logger.warning(
                        "PLE offload waited >5 s for the GPU to release %s "
                        "(dp=%d, tp=%d, step %d); flag=%d",
                        layer_name,
                        dp_rank,
                        target.tp_rank,
                        self._step,
                        int(flag[0]),
                    )
                if waited > _PLE_WAIT_TIMEOUT_S:
                    raise RuntimeError(
                        f"PLE offload: GPU did not release {layer_name} (dp={dp_rank}, "
                        f"tp={target.tp_rank}, step {self._step}) within "
                        f"{_PLE_WAIT_TIMEOUT_S:.0f} s; refusing to overwrite"
                    )
