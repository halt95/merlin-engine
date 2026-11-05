# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Persistent, cudagraph-safe counters for the bounds guards.

Every bounds guard MASKS an out-of-range access so the SM cannot fault on it, and
records that it did. A guard that masks silently turns a crash into a wrong
answer and tells nobody, so the recording is not optional.

Design (follows the QSA KV-clip counter, vllm/models/qwen4_exp/nvidia/qsa.py,
which survives FULL cudagraph replay):

  * counters are PERSISTENT int64 GPU tensors, allocated eagerly by prealloc()
    from the model runner constructor, so nothing is allocated inside a capture
    and the data_ptr is stable across replays;
  * the kernels do nothing but captured atomics, so replay keeps counting;
  * a daemon thread reads them on a cadence, OFF the engine thread, so no
    device->host sync is added on the step path. It starts only AFTER warmup and
    cudagraph capture, from gpu_worker beside enable_gpu_sync_check() (see
    start_reader()).

A counter allocated per launch and read back inline cannot work: the D2H aborts
capture, and under replay the Python host code never runs.

Negative values. Slots 1 and 2 are written with tl.atomic_max into a zeroed
buffer, so every recorded value is stored biased by GUARD_BIAS and the reader
subtracts it; a slot that is still 0 means "nothing recorded". Kernel modules
bind the bias as a tl.constexpr (a plain module-level int is not readable from a
@triton.jit body).

Modes (VLLM_E2_GUARD_MODE):
  raise  the reader logs and arms a flag; a LATER execute_model on the engine
         thread raises. The raise is late and local to one rank; read
         check_pending before using it.
  warn   (default) log only, except for the sites in _ALWAYS_FATAL_SITES, which
         arm the raise whatever the mode says, because their masked value has
         already been written somewhere durable.
  Any other value, "off" included, is a boot error.

Runner. prealloc() and check_pending() are called from the V2 runner
(vllm/v1/worker/gpu/). On a V1 runner the counters are allocated lazily by the
first guarded launch and nothing calls check_pending, so raise degrades to warn
and _ALWAYS_FATAL_SITES never stops anything. The guards still mask and count.

Idle quiesce (always on). A reader that copies every _INTERVAL forever keeps an
idle GPU out of its idle P-state, so the worker wraps its GPU entry points
(execute_model, sample_tokens, execute_dummy_batch) in engine_call, which keeps a
call generation and an open-call count, and the reader runs a completion-driven
state machine:

  ACTIVE  poll on the cadence. Detection is cadence-bound, not one step: the
          copy that sees a new count is consumed a tick or two later, where a
          fatal site's count is armed; the raise lands at the next
          execute_model, and under async scheduling several steps can run in
          that window.
  FENCE   no engine call open and none started for a full interval: record a
          CUDA event on every stream the guarded kernels run on (from the
          reader, so the engine thread gains no CUDA call) and keep polling
          until it completes.
  DRAIN   the fence completed, so every guarded write so far has finished: issue
          one fresh copy, report it directly, fold in any half-confirmed
          increase, leave nothing in flight.
  PARKED  wait on a condition variable, no CUDA call of any kind, until the next
          engine_call enter, suspend_reader() or stop_reader().

Single-stream invariant: every guarded kernel runs on the engine thread's fenced
stream. A violation falls back to polling forever with an ERROR line, never to a
silent miss. Worker.sleep suspends the reader (one settled read, then no CUDA
call until wake_up), whatever the idle switch says. A call that starts before the
park decision aborts the quiesce, so a hit from before the idle period is
reported (and, for a fatal site, armed) before the reader parks. The gate is
opt-in per process (enable_idle_gate).

Transport (VLLM_E2_GUARD_TRANSPORT, default "mapped"): how the host sees the
counters. Detection does not depend on it; cost and latency do.

  copy    the counters are device memory; the reader mirrors them with a
          non-blocking side-stream D2H copy every tick (NonBlockingReader).
          Fatal sites are armed by the cadence reader only.
  mapped  the counter buffer is page-locked host memory mapped into the device
          address space (cudaHostAlloc under UVA), exposed to the kernels as an
          ordinary [12, 3] int64 CUDA tensor. The guarded kernels are unchanged
          (same source, arguments, specialisation and captured graph), and a
          clean step issues no counter access at all, because every atomic is
          predicated on a hit.
          Host visibility is event-gated (_StepEvents): every engine call ends
          with a CUDA event on the guarded stream, and the host reads the mapped
          counters only after an event newer than its last read has completed.
          The engine thread checks the fatal sites' counts (every site under
          raise) at the top of every execute_model, so a fatal hit in engine
          call N stops the first execute_model after call N's event completes
          (contract in check_pending). A count is claimed and armed in one
          critical section before anything is logged (_claim_and_arm), and any
          failure in the observe path arms a contract-failure stop.
          The mapping is validated at allocation (before capture, on the engine
          thread): pointer identity, device ordinal, a host write read by a
          kernel, a kernel write, a contended run of the guards' own atomic
          sequences over every SM checked against a device-memory reference, and
          a D2H copy through the alias. Any failure, a capture in progress or an
          allocation after the reader started keeps that device on copy, loudly;
          the effective transport is logged per rank. The self-test makes a few
          small device allocations before model load, so the device caching
          allocator's history differs from copy: a temperature-0 identity check
          between the transports is part of qualifying the mapped transport.
          Assumes CPython with the GIL: the step check's lock-free fast path
          reads a fatal site's high-water mark and then _pending, which
          _claim_and_arm writes in the opposite order under _fatal_lock. A
          free-threaded build must take the slow path (_fatal_lock)
          unconditionally.
"""

import functools
import math
import os
import threading
import time
from datetime import timedelta

import torch

from vllm.logger import init_logger
from vllm.merlin.envs import read_registered

logger = init_logger(__name__)

# Bias applied to every recorded row/value so negatives survive atomic_max on a
# zero-initialised buffer. Any int32-range value plus this is > 0.
GUARD_BIAS = 1 << 31

# One (n_masked, row, value) triple per guarded kernel.
SLOTS_PER_SITE = 3

# Largest value the bias encoding can carry without overflowing the int64 slot.
GUARD_MAX_ENCODABLE = (1 << 63) - 1 - GUARD_BIAS
GUARD_MIN_ENCODABLE = -GUARD_BIAS + 1


def decode_slot(biased: int) -> int | None:
    """Reader side. ``None`` means "this slot recorded nothing".

    THE ENCODING'S LIMITS (closed here by test
    rather than by change). A slot is an int64 zeroed at allocation and only
    ever written with `tl.atomic_max`, so the recorded value must be biased to
    survive negatives, and 0 has to mean "unwritten". Three classes of extreme
    value are therefore not recoverable from the slot:

      * ``value == -GUARD_BIAS`` (INT32_MIN) encodes to exactly 0 and is
        FALSIFIED: it reads back as "nothing recorded".
      * ``value < -GUARD_BIAS`` encodes negative, `atomic_max` against 0 keeps
        0, and the value is LOST the same way.
      * ``value > GUARD_MAX_ENCODABLE`` overflows the int64 slot.

    In every one of those cases the COUNT slot, which carries no bias and is
    written with `atomic_add`, still advances.

    BUT `None` IS NOT A DIAGNOSIS OF "UNENCODABLE". The reader has
    NO ordering dependency on the producer -- it copies through a pinned mirror
    on a side stream with no dependency on the compute stream -- so a count
    that has become visible with its maxima not yet visible produces exactly
    the same reading. `_poll_once`'s one-tick confirm delay makes that less
    likely; it does not make it impossible, and a delay is not a happens-before
    relationship. So `None` means UNOBSERVED OR UNENCODABLE.

    What it never means is "this field recorded nothing while the count rose".
    Anything reading these slots must say so, and must not build a fatal
    decision on the maxima -- those decisions are count-based.
    """
    return None if biased == 0 else int(biased) - GUARD_BIAS

SITE_MAMBA_COPY = 0
SITE_GDN_DECODE_STATE = 1
SITE_QSA_PRE_INDEXER = 2
SITE_QSA_PREFILL = 3
SITE_QSA_DECODE = 4
SITE_BT_GATHER = 5
SITE_SLOT_MAPPING = 6
SITE_KV_ZEROER = 7
SITE_GDN_RECURRENT_STATE = 8
SITE_QSA_ROPE_POS = 9
SITE_QSA_RING_SKIP = 10
SITE_QSA_RING_UNCOMMITTED = 11
NUM_SITES = 12

# Bit layout of the SITE_QSA_RING_UNCOMMITTED record (before GUARD_BIAS).
RING_RECORD_POS_BITS = 32
RING_RECORD_REQ_BITS = 16
RING_RECORD_LAYER_BITS = 14


def decode_ring_record(biased: int) -> dict[str, int] | None:
    """Unpack slot 1 of SITE_QSA_RING_UNCOMMITTED. ``layer`` is -1 when the
    launcher was not given one. None = nothing recorded (or not yet visible)."""
    value = decode_slot(biased)
    if value is None:
        return None
    pos_mask = (1 << RING_RECORD_POS_BITS) - 1
    req_mask = (1 << RING_RECORD_REQ_BITS) - 1
    layer_mask = (1 << RING_RECORD_LAYER_BITS) - 1
    return {
        "layer": ((value >> (RING_RECORD_REQ_BITS + RING_RECORD_POS_BITS)) & layer_mask)
        - 1,
        "request_row": (value >> RING_RECORD_POS_BITS) & req_mask,
        "source_position": value & pos_mask,
    }


SITE_NAMES = {
    SITE_MAMBA_COPY: "_copy_mamba_state_block (block-table row/column)",
    SITE_GDN_DECODE_STATE: "fused_sigmoid_gating_delta_rule_update_kernel "
    "(ssm_state_indices column)",
    SITE_QSA_PRE_INDEXER: "_qsa_pre_indexer_kernel (k_work request row)",
    SITE_QSA_PREFILL: "_qsa_mqa_paged_prefill_kernel (page-table page id)",
    SITE_QSA_DECODE: "_qsa_mqa_paged_uniform_kernel (page-table page id)",
    SITE_BT_GATHER: "_gather_block_tables_kernel (request row / num_blocks)",
    SITE_SLOT_MAPPING: "_compute_slot_mappings_kernel (request-state row)",
    SITE_KV_ZEROER: "_zero_kv_blocks_kernel (KV pool block id)",
    SITE_GDN_RECURRENT_STATE: "fused_recurrent_gated_delta_rule_fwd_kernel "
    "(ssm_state_indices column)",
    SITE_QSA_ROPE_POS: "_norm_rope in _qsa_pre_indexer_kernel "
    "(rope position vs cos/sin table rows; slot 2 = axis 0=t 1=h 2=w)",
    SITE_QSA_RING_SKIP: "_qsa_pre_indexer_kernel suffix commit SKIPPED "
    "(state slot < 0 or >= ring capacity; slot 1 = token index, "
    "slot 2 = state slot value)",
    SITE_QSA_RING_UNCOMMITTED: "_qsa_pre_indexer_kernel historical ring row "
    "UNCOMMITTED (tag != logical_position + 1; group load, RoPE tail and "
    "compressed-slot store masked; slot 1 = packed first-offender record, "
    "slot 2 = the tag its row held)",
}


class GuardFatalError(RuntimeError):
    """drain(raise_on_fatal=True) found a fatal-site hit. A distinct type so
    the worker's shutdown can tell it from an incidental drain failure."""


_MODE = os.environ.get("VLLM_E2_GUARD_MODE", "warn").strip().lower()
if _MODE not in ("raise", "warn"):
    # fail closed: a set-but-broken gate must not silently disarm the instrument
    raise ValueError(f"VLLM_E2_GUARD_MODE must be raise|warn, got {_MODE!r}")


def _parse_interval(text: str) -> float:
    """Seconds between reader polls: a finite number > 0, or ValueError.

    float() accepts "nan", "inf" and negatives; the first two reach
    time.sleep() and kill the reader daemon, a negative one raises there
    instead. Reject them at import, where a broken
    setting is a loud boot failure instead of a silently absent instrument.
    """
    try:
        value = float(text)
    except ValueError as exc:
        raise ValueError("VLLM_E2_GUARD_INTERVAL must be seconds") from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError(
            f"VLLM_E2_GUARD_INTERVAL must be a finite number > 0, got {text!r}"
        )
    return value


# the registry getter carries the name and the
# default; _parse_interval stays the one validator.
_INTERVAL = _parse_interval(read_registered("VLLM_E2_GUARD_INTERVAL"))


def _parse_transport(text: str | None) -> str:
    """VLLM_E2_GUARD_TRANSPORT: unset or empty = "mapped" (default), "copy" =
    the side-stream D2H reader. Anything else is a ValueError at import."""
    value = "" if text is None else text.strip().lower()
    if value in ("", "mapped"):
        return "mapped"
    if value == "copy":
        return "copy"
    raise ValueError(f"VLLM_E2_GUARD_TRANSPORT must be mapped or copy, got {text!r}")


_TRANSPORT = _parse_transport(os.environ.get("VLLM_E2_GUARD_TRANSPORT"))
_DRAIN_TIMEOUT_S = 2.0

_cv = threading.Condition(threading.Lock())
_call_seq = 0  # bumped on every engine_call enter and exit
_calls_open = 0
_gate_armed = False  # enable_idle_gate() ran in this process
_gate_broken: str | None = None  # why the gate fell back to polling forever
_fence_events: dict[torch.device, list] = {}  # device -> [(stream, event)]
_reader_parked = False
_reader_thread: threading.Thread | None = None
_reader_stop = threading.Event()
_engine_thread: int | None = None  # threading.get_ident() of the arming thread
_engine_device: torch.device | None = None
_fence_keys: dict[torch.device, frozenset] = {}  # device -> fence stream identities
_drain_streams: dict[torch.device, list] = {}
_suspend_req = False  # suspend_reader() asked; resume_reader() clears it
_suspend_gen = 0  # bumped by every suspend_reader() request
_suspended_gen = 0  # the request the reader last acknowledged
_reader_suspended = False  # the reader acknowledged and makes no CUDA call
_asleep = False  # suspended for Worker.sleep: drain() must not read the device
_gate_gen = 0  # bumped by every arm and lifecycle reset
# Serialises start_reader / enable_idle_gate / stop_reader (never taken by the
# reader thread or on the step path).
_life_lock = threading.RLock()
# Serialises every counter read (cadence poll, idle drain, drain()): they share
# the pinned mirrors, events, _seen and _confirm_pending.
_io_lock = threading.Lock()
# Observability and tests: reader-issued CUDA operations and transitions.
_stats = {"polls": 0, "reads": 0, "fences": 0, "drains": 0, "parks": 0, "resumes": 0}

_buffers: dict[torch.device, torch.Tensor] = {}
# Devices the reader has given up on. They stay in _buffers: see _reader.
_read_disabled: set[torch.device] = set()
_lock = threading.Lock()
_reader_started = False
_pending: str | None = None
_seen: dict[tuple[torch.device, int], int] = {}

_host_views: dict[torch.device, torch.Tensor] = {}
# Engine-thread step-boundary check (check_pending): (device, numpy view of the
# count column) for every mapped device. Replaced, never mutated in place.
_step_views: tuple = ()
_step_check_on = False  # start_reader() turns it on
_fatal_hw: dict[torch.device, list] = {}
_fatal_lock = threading.Lock()


class NonBlockingReader:
    """Reads device counter tensors from a side thread without ever waiting on
    the compute stream.

    A synchronous, pageable device-to-host copy (``buf.tolist()``) on the
    compute stream is serviced by the driver holding the CUDA context lock until
    the stream's queued work has completed. Under the PLE pull transport the
    stream can hold a GPU-side wait whose producer needs the same lock (the
    connector thread's ``event.synchronize()`` and the main thread's next kernel
    launch), so such a read would deadlock the whole worker. The reader must
    never block on the compute stream.

    Every registered tensor gets a pinned host mirror and the copy runs
    on a dedicated side stream with ``non_blocking=True``; completion is
    observed with ``event.query()``. Nothing in this class blocks on the GPU.
    The side stream has no dependency on the compute stream, so a pending
    wait there cannot stall the copy, and a non-blocking pinned copy returns
    from the driver immediately. The rows a tick reports are therefore the
    values as of the previous tick's copy, and the three slots of one site
    may come from slightly different moments while kernels are still
    counting: the count is monotone and the maxima are lifetime maxima, so
    this changes nothing the reader claims.
    """

    def __init__(self) -> None:
        self._streams: dict[torch.device, torch.cuda.Stream] = {}
        # key -> [device_tensor, pinned_host_tensor | None, event | None, inflight]
        self._jobs: dict[object, list] = {}
        self._lock = threading.Lock()

    def register(self, key: object, tensor: torch.Tensor) -> None:
        with self._lock:
            if key in self._jobs:
                return
            if tensor.device.type != "cuda":
                self._jobs[key] = [tensor, None, None, False]
                return
            dev = tensor.device
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "guard_telemetry: NonBlockingReader.register() called during "
                    "cudagraph capture; prealloc() must run before capture"
                )
            if dev not in self._streams:
                with torch.cuda.device(dev):
                    self._streams[dev] = torch.cuda.Stream(device=dev)
            host = torch.empty(
                tensor.shape, dtype=tensor.dtype, device="cpu", pin_memory=True
            )
            with torch.cuda.device(dev):
                event = torch.cuda.Event()
            self._jobs[key] = [tensor, host, event, False]

    def registered(self, key: object) -> bool:
        return key in self._jobs

    def poll_one(self, key: object) -> list | None:
        """Return the rows of a completed copy for ``key`` and start the next
        one; return None when the previous copy is still in flight. Never
        blocks."""
        job = self._jobs[key]
        tensor, host, event, inflight = job
        if host is None:  # CPU tensor: nothing can block
            return tensor.tolist()
        rows = None
        if inflight:
            if not event.query():
                return None
            rows = host.tolist()
        stream = self._streams[tensor.device]
        with torch.cuda.device(tensor.device), torch.cuda.stream(stream):
            host.copy_(tensor, non_blocking=True)
            event.record(stream)
        job[3] = True
        return rows

    def poll(self) -> dict[object, list]:
        out = {}
        with self._lock:
            keys = list(self._jobs)
        for key in keys:
            rows = self.poll_one(key)
            if rows is not None:
                out[key] = rows
        return out

    def read_settled(self, key: object, timeout: float, cancel=None) -> list | None:
        """Final read for the idle drain, called only once a fence has shown
        that every producer of this buffer is complete. Waits for any copy
        still in flight and DISCARDS it (the counters are monotone, so the
        fresh copy carries everything it did), issues one fresh copy, waits for
        it and returns its rows, leaving nothing in flight. Waiting is
        event.query() polling on the reader thread, never a blocking driver
        wait (see the class docstring). None = a copy did not complete within
        ``timeout``, or ``cancel()`` turned true while waiting (a call resumed,
        a stop or suspend was asked); the copy is left in flight, exactly as
        the cadence path expects it, and the caller goes back to the
        non-blocking cadence."""
        job = self._jobs[key]
        tensor, host, event, inflight = job
        if host is None:  # CPU tensor: nothing can be in flight
            return tensor.tolist()
        deadline = time.monotonic() + timeout

        def landed() -> bool:
            while not event.query():
                if time.monotonic() >= deadline or (cancel is not None and cancel()):
                    return False
                time.sleep(0.0005)
            return True

        if inflight and not landed():
            return None
        stream = self._streams[tensor.device]
        with torch.cuda.device(tensor.device), torch.cuda.stream(stream):
            host.copy_(tensor, non_blocking=True)
            event.record(stream)
        job[3] = True
        if not landed():
            return None
        job[3] = False
        return host.tolist()


_nb_reader = NonBlockingReader()
# device -> the copy in which a count increase was first seen
# (reported on the next tick)
_confirm_pending: dict[torch.device, list] = {}

_ALWAYS_FATAL_SITES: set[int] = set()

_ALWAYS_FATAL_SITES.add(SITE_QSA_ROPE_POS)

_ALWAYS_FATAL_SITES.add(SITE_QSA_RING_UNCOMMITTED)


def _normalize(device: torch.device) -> torch.device:
    """torch.device("cuda") and torch.device("cuda:0") are different dict keys.

    A miss here would allocate a second buffer lazily, on whatever stream is
    current.
    """
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return device


def counter(site: int, device: torch.device) -> torch.Tensor:
    """Persistent [3] int64 slice for one guarded kernel.

    Allocation should already have happened in prealloc(); the lazy path is a
    fallback so a kernel reached from a context the runner did not construct
    still counts rather than crashes.
    """
    device = _normalize(device)
    buf = _buffers.get(device)
    if buf is None:
        with _lock:
            buf = _buffers.get(device)
            if buf is None:
                buf, host = _new_buffer(device)
                if host is not None:
                    _register_host(device, host)
                _buffers[device] = buf
    with _cv:
        snap = (
            (_gate_gen, _calls_open, _engine_thread)
            if _gate_armed and _gate_broken is None
            else None
        )
    if snap is not None:
        gen, calls_open, owner = snap
        if calls_open == 0 or threading.get_ident() != owner:
            _break_gate(
                f"guarded launch at {SITE_NAMES.get(site, site)} outside an "
                f"engine call or off the engine thread",
                gen,
                coverage=True,
            )
        elif _off_fence_stream(device):
            _break_gate(
                f"guarded launch at {SITE_NAMES.get(site, site)} on {device} on a "
                f"stream the fence does not cover",
                gen,
                coverage=True,
            )
    return buf[site]


def prealloc(device: torch.device) -> None:
    """Allocate every site counter eagerly, before any cudagraph capture.

    counter() allocates lazily on first use. A guarded kernel whose FIRST
    execution happens inside a capture would allocate on the capturing stream,
    which is exactly the failure mode to avoid. The model
    runner calls this at construction, so by capture time every buffer exists
    and counter() is a dict lookup plus a view.

    Under TRANSPORT=mapped on CUDA it first agrees the transport with this
    replica's TP and PP ranks (_agree_transport): one transport per replica,
    mapped only if every rank's self-test passed. That is a gloo collective, so
    every TP/PP rank must call prealloc() (the V2 runner constructor does).
    """
    device = _normalize(device)
    with _cv:
        # An earlier worker's fence streams do not cover this worker's warmup
        # and capture producers: no drain() before start_reader() may use them.
        _drain_streams.pop(device, None)
    if _agreement_applies(device):
        _agree_transport(device)  # collective; before any buffer and any capture
    for site in range(NUM_SITES):
        counter(site, device)
    if device not in _host_views and not _nb_reader.registered(device):
        _nb_reader.register(device, _buffers[device])


_SHAPE = (NUM_SITES, SLOTS_PER_SITE)


def _new_buffer(device: torch.device) -> tuple[torch.Tensor, torch.Tensor | None]:
    """(counter buffer, host view or None). The host view is returned only
    for TRANSPORT=mapped."""
    if device in _preagreed:
        # Agreed across the TP/PP ranks by prealloc(): honour it exactly (a
        # second self-test here could disagree with the first).
        pair = _preagreed.pop(device)
        if pair is not None:
            return pair
        return torch.zeros(_SHAPE, dtype=torch.int64, device=device), None
    if _TRANSPORT == "mapped" and _step_check_on:
        logger.warning(
            "bounds-guard counters on %s first allocated after the reader "
            "started: copy transport for this device (no step events).",
            device,
        )
    elif _TRANSPORT == "mapped" and _ubatching():
        logger.warning(
            "bounds-guard counters on %s: micro-batching (DBO / ubatch_size > 1) "
            "runs the forward on per-ubatch threads and streams; copy transport "
            "for this device (mapped needs one engine thread and stream).",
            device,
        )
    elif _TRANSPORT == "mapped":
        if device.type == "cpu":
            # CPU memory is host memory: the mapped transport is a direct read.
            buf = torch.zeros(_SHAPE, dtype=torch.int64, device=device)
            return buf, buf
        if device.type == "cuda":
            if torch.cuda.is_current_stream_capturing():
                logger.warning(
                    "bounds-guard counters on %s first allocated during a "
                    "cudagraph capture: copy transport for this device (the "
                    "mapped-transport self-test cannot run inside a capture).",
                    device,
                )
            else:
                pair = _alloc_mapped(device)
                if pair is not None:
                    return pair
    return torch.zeros(_SHAPE, dtype=torch.int64, device=device), None


def _ubatching() -> bool:
    """The current vLLM config runs micro-batches (enable_dbo or
    ubatch_size > 1). The worker constructs its model runner -- and prealloc()
    runs -- inside set_current_vllm_config. No config (tests): False."""
    try:
        from vllm.config.vllm import get_current_vllm_config_or_none
    except ImportError:
        return False
    try:
        cfg = get_current_vllm_config_or_none()
        return bool(cfg is not None and cfg.parallel_config.use_ubatching)
    except Exception:  # noqa: BLE001
        # Gates a cross-rank collective: never default to mapped; copy is safe.
        logger.exception(
            "bounds-guard: cannot read the vLLM config to detect micro-batching; "
            "assuming it is ON (copy transport, no transport agreement)"
        )
        return True


class _CudaArrayRef:
    """__cuda_array_interface__ over a mapped host allocation, so torch wraps
    it (zero-copy) as an ordinary CUDA tensor the Triton launchers accept."""

    def __init__(self, ptr: int) -> None:
        self.__cuda_array_interface__ = {
            "shape": _SHAPE,
            "typestr": "<i8",
            "data": (ptr, False),
            "version": 3,
            "strides": None,
        }


def _driver_device_pointer(ptr: int) -> int | None:
    """CU_POINTER_ATTRIBUTE_DEVICE_POINTER of a host pointer, from the driver
    (the same query Triton's launcher makes). None when it cannot be asked."""
    import ctypes
    import sys

    name = "nvcuda.dll" if sys.platform == "win32" else "libcuda.so.1"
    try:
        lib = ctypes.CDLL(name)
        out = ctypes.c_uint64(0)
        rc = lib.cuPointerGetAttribute(
            ctypes.byref(out), ctypes.c_int(3), ctypes.c_uint64(ptr)
        )
    except Exception:  # noqa: BLE001
        return None
    return int(out.value) if rc == 0 else None


# Every mapped host allocation stays referenced here for the life of the
# process: captured graphs bake its address (see _lose_device).
_mapped_keepalive: list = []


def _alloc_mapped(device: torch.device) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Mapped page-locked counters for ``device`` after a self-test, or None
    (copy transport) with a WARNING. Engine thread, before capture."""

    def refuse(why: str) -> None:
        logger.warning(
            "bounds-guard counters on %s: mapped transport unavailable (%s); "
            "using the copy transport (side-stream D2H reader). Detection is "
            "unchanged; the per-step fatal check is the cadence reader's.",
            device,
            why,
        )

    conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "").replace(" ", "").lower()
    if "pinned_use_cuda_host_register:true" in conf:
        # cudaHostRegister'd memory has no host==device pointer guarantee.
        refuse("PYTORCH_CUDA_ALLOC_CONF pinned_use_cuda_host_register")
        return None
    try:
        with torch.cuda.device(device):
            host = torch.zeros(_SHAPE, dtype=torch.int64, pin_memory=True)
            _mapped_keepalive.append(host)
            ptr = host.data_ptr()
            if _driver_device_pointer(ptr) != ptr:
                refuse("driver does not map the host pointer at the same address")
                return None
            ref = _CudaArrayRef(ptr)
            dev = torch.as_tensor(ref, device=device)
            _mapped_keepalive.append(ref)
            if dev.device != device or dev.data_ptr() != ptr:
                refuse(f"wrapped as {dev.device} @ {dev.data_ptr():#x}")
                return None
            # Self-test: every access class the kernels and this module use.
            host[0, 1] = 7  # host write ...
            dev[0, 2:3].add_(dev[0, 1:2])  # ... read and written by a kernel
            idx = torch.zeros(2, dtype=torch.int64, device=device)
            dev[1].index_add_(  # device atomic add
                0, idx, torch.tensor([2, 3], dtype=torch.int64, device=device)
            )
            dev[1].scatter_reduce_(  # device atomic max
                0,
                idx + 1,
                torch.tensor(
                    [GUARD_BIAS + 4, GUARD_BIAS - 2], dtype=torch.int64, device=device
                ),
                reduce="amax",
            )
            torch.cuda.synchronize(device)
            seen = (int(host[0, 2]), int(host[1, 0]), int(host[1, 1]))
            want = (7, 5, GUARD_BIAS + 4)
            via_copy = dev.tolist()  # D2H copy through the device alias
            if seen != want or via_copy[1][1] != GUARD_BIAS + 4:
                host.zero_()
                refuse(f"self-test read {seen} / copy {via_copy[1][:2]}, want {want}")
                return None
            host.zero_()
            sms = torch.cuda.get_device_properties(device).multi_processor_count
            programs, block = 4 * sms, 32
            ref = torch.zeros((2, SLOTS_PER_SITE), dtype=torch.int64, device=device)
            guard_atomic_probe(dev[0], dev[1], programs, block)
            guard_atomic_probe(ref[0], ref[1], programs, block)
            torch.cuda.synchronize(device)
            rows, ref_rows = host[:2].tolist(), ref.tolist()
            host.zero_()
            bad = probe_check(rows[0], rows[1], programs, block) or probe_check(
                ref_rows[0], ref_rows[1], programs, block
            )
            if bad is None and rows[0] != ref_rows[0]:
                bad = f"mapped {rows[0]} != device reference {ref_rows[0]}"
            if bad is not None:
                refuse(f"contended self-test: {bad}")
                return None
    except Exception as exc:  # noqa: BLE001 -- fall back, never fail the boot
        refuse(f"{type(exc).__name__}: {exc}")
        return None
    logger.info(
        "bounds-guard counters on %s: mapped host memory (no D2H copy; "
        "event-gated host reads; the engine thread checks fatal counts at every "
        "step boundary); self-test ok incl. %d programs x %d lanes contending on "
        "%d SMs.",
        device,
        programs,
        block,
        sms,
    )
    return dev, host


_AGREE_TIMEOUT_S = 300.0
# device -> agreed (dev, host) pair, or None = agreed copy. Written by prealloc()
# before its site loop, popped once by _new_buffer under _lock.
_preagreed: dict[torch.device, tuple | None] = {}


def _mp_cpu_groups() -> list:
    """This rank's TP gloo group, then its PP gloo group, each only with more
    than one member. [] without torch.distributed or vLLM's model-parallel
    groups (single GPU, tests, the PLE offload process). Every term is the same
    on all members of a group, so every member enters each collective or none
    does. No broad except: a failure here fails every rank alike."""
    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()):
        return []
    from vllm.distributed import parallel_state as ps

    if not ps.model_parallel_is_initialized():
        return []
    return [
        g.cpu_group for g in (ps.get_tp_group(), ps.get_pp_group()) if g.world_size > 1
    ]


def _agree_flags(local_ok: bool, locked: bool) -> tuple[bool, bool, bool]:
    """(all_ok, any_ok, any_locked) over this replica's TP x PP ranks. One MIN
    all-reduce of [ok, -ok, -locked] per group, TP first, then PP: after the TP
    reduce every rank holds its stage's extremes, and the PP reduce (over a fixed
    TP rank) gives every rank the replica's. CPU tensor on gloo: no CUDA stream,
    no NCCL, nothing a capture could see. NOT the world group: it spans DP
    replicas, which boot independently. After a wait times out the gloo worker
    thread stays inside the collective until the group's own op timeout, so a
    rank that raises here exits only because the executor kills unready workers;
    keep that dependency."""
    groups = _mp_cpu_groups()
    if not groups:
        return local_ok, local_ok, locked
    import torch.distributed as dist

    v = torch.tensor([int(local_ok), -int(local_ok), -int(locked)], dtype=torch.int64)
    for g in groups:
        work = dist.all_reduce(v, op=dist.ReduceOp.MIN, group=g, async_op=True)
        try:
            work.wait(timeout=timedelta(seconds=_AGREE_TIMEOUT_S))
        except Exception as exc:  # noqa: BLE001 -- re-raised with the reason
            raise RuntimeError(
                "bounds-guard transport agreement: a TP/PP peer did not reach "
                f"prealloc() within {_AGREE_TIMEOUT_S:.0f}s ({exc}); refusing to "
                "boot with possibly mixed guard transports"
            ) from exc
    return v[0].item() == 1, v[1].item() == -1, v[2].item() == -1


def _agreement_applies(device: torch.device) -> bool:
    """Whether prealloc() runs the agreement. ONLY terms that are equal on every
    TP/PP rank of a replica (env, config, device type). Rank-local facts (self-
    test, existing buffers, _step_check_on, capture) go into the agreed VALUE,
    never into this predicate: a rank that skipped the collective would hang its
    peers until _AGREE_TIMEOUT_S."""
    return (
        _TRANSPORT == "mapped"
        and device.type == "cuda"
        and not _ubatching()
    )


def _agree_transport(device: torch.device) -> None:
    """prealloc(), before the site loop, OUTSIDE _lock (a collective under a lock
    is a deadlock waiting for a second user of that lock)."""
    with _lock:
        existing = device in _buffers
        locked = device in _host_views  # mapped and possibly baked into graphs
    pair = None
    if not (existing or _step_check_on or torch.cuda.is_current_stream_capturing()):
        pair = _alloc_mapped(device)  # the self-test; never raises
    local_ok = locked or pair is not None
    all_ok, any_ok, any_locked = _agree_flags(local_ok, locked)
    split = any_ok and not all_ok
    if split and any_locked:
        # A mapped buffer in use cannot be demoted and its peers cannot be
        # promoted. Every rank holds the same flags, so every rank raises.
        raise RuntimeError(
            "bounds-guard transport split across TP/PP ranks with a mapped buffer "
            "already in use; restart with VLLM_E2_GUARD_TRANSPORT=copy"
        )
    if split and pair is not None:
        logger.warning(
            "bounds-guard counters on %s: the mapped self-test passed here but "
            "failed on a TP/PP peer; copy transport on every rank (one transport "
            "per replica). The mapped page stays allocated (keepalive).",
            device,
        )
        pair = None
    if not existing:
        with _lock:
            _preagreed[device] = pair if all_ok else None
    logger.info(
        "bounds-guard transport agreed on %s across the TP/PP ranks: %s",
        device,
        "mapped" if all_ok else "copy",
    )


_PROBE_KERNEL = None
_PROBE_HASH = (40503, 65521, 32768)


def _probe_kernel():
    global _PROBE_KERNEL
    if _PROBE_KERNEL is None:
        from vllm.triton_utils import tl, triton

        @triton.jit
        def _guard_atomic_probe(
            ctr_ptr, ring_ptr, BIAS: tl.constexpr, BLOCK: tl.constexpr
        ):
            pid = tl.program_id(0)
            lanes = tl.arange(0, BLOCK)
            zero = lanes * 0
            # 1. vector atomics, every lane of every program on ONE address
            #    (the masked vector form of the ring-skip site)
            tl.atomic_add(ctr_ptr + zero, 1)
            tl.atomic_max(ctr_ptr + 1 + zero, (pid * BLOCK + lanes).to(tl.int64) + BIAS)
            # 2. one scalar max per program (the hit path of every other guard),
            #    over a hashed value that is negative for some programs
            h = (pid.to(tl.int64) * 40503) % 65521 - 32768
            tl.atomic_max(ctr_ptr + 2, h + BIAS)
            # 3. the ring-record claim of qsa_pre_indexer (lines 477-486):
            #    count, first-offender CAS on slot 1, tag exchange by the winner
            tl.atomic_add(ring_ptr, 1)
            rec = (pid + 1).to(tl.int64) + BIAS
            prev = tl.atomic_cas(ring_ptr + 1, rec * 0, rec)
            if prev == 0:
                tl.atomic_xchg(ring_ptr + 2, pid.to(tl.int64) * 3 + 1 + BIAS)

        _PROBE_KERNEL = _guard_atomic_probe
    return _PROBE_KERNEL


def guard_atomic_probe(ctr: torch.Tensor, ring: torch.Tensor, programs: int, block: int) -> None:
    """Launch the contention probe on two [3] int64 rows (a guard triple and a
    ring triple). Used by the mapped self-test and the CUDA qualification."""
    _probe_kernel()[(programs,)](ctr, ring, BIAS=GUARD_BIAS, BLOCK=block)


def probe_check(ctr, ring, programs: int, block: int, replays: int = 1) -> str | None:
    """None when (ctr, ring) hold exactly what ``replays`` probe launches must
    leave; else what differs. The CAS winner is any program, but slot 2 must
    be THAT program's tag (one coherent record)."""
    mul, mod, off = _PROBE_HASH
    hmax = max((p * mul) % mod - off for p in range(programs))
    want = [replays * programs * block, GUARD_BIAS + programs * block - 1, GUARD_BIAS + hmax]
    if list(ctr) != want:
        return f"guard triple {list(ctr)} != {want}"
    if ring[0] != replays * programs:
        return f"ring count {ring[0]} != {replays * programs}"
    w = ring[1] - GUARD_BIAS - 1
    if not 0 <= w < programs:
        return f"ring owner {ring[1]} is no program's record"
    if ring[2] != GUARD_BIAS + 3 * w + 1:
        return f"ring tag {ring[2]} is not owner {w}'s tag"
    return None


def _register_host(device: torch.device, host: torch.Tensor) -> None:
    """Under _lock: make ``host`` the reader's and the step check's view."""
    global _step_views
    _host_views[device] = host
    counts = host.numpy()[:, 0]  # a strided view of the count column, no copy
    _step_views = _step_views + ((device, counts),)


# Event ring per mapped device. 16 >> the in-flight engine calls of any shape
# here (PP x async lookahead keeps <= ~4 outstanding per rank).
_STEP_RING = 16


class _HostEvent:
    """Step event of a CPU-device counter buffer: CPU memory needs no
    visibility event, so every record is complete at once."""

    def record(self) -> None:
        pass

    def query(self) -> bool:
        return True


def _new_step_event(device: torch.device):
    if device.type != "cuda":
        return _HostEvent()
    with torch.cuda.device(device):
        return torch.cuda.Event()


class _StepEvents:
    """HOST VISIBILITY of the mapped counters (TRANSPORT=mapped).

    The guarded kernels write the mapped page-locked counters with device-scope
    atomics from one GPU. The host reads them only for work it has observed
    complete through a CUDA event recorded after that work in stream order (the
    CUDA programming guide requires streams or events to order accesses to mapped
    memory). A successful ``cudaEventQuery`` is the observation point, the same
    one the COPY transport rests on (NonBlockingReader.poll_one: event.query(),
    then host.tolist() of a DMA-written pinned mirror), so mapped adds no new
    visibility assumption. The host never relies on counter writes for work in
    flight and never read-modify-writes a counter.

    The live buffer: event N complete does NOT mean nothing writes the buffer --
    call N+1's kernels may be running while the host loads it. The contract needs
    only this, and it holds:
      * every slot is MONOTONE NON-DECREASING over the buffer's life: counts only
        atomic_add positive amounts; maxima only atomic_max (biased, so never
        below 0); the ring record's slot 1 is written once by the first CAS from
        0 and slot 2 once by that CAS's winner (atomic_xchg from 0);
      * each host load is a naturally aligned 8-byte load (numpy / tolist on an
        int64 element), single-copy atomic on x86-64, and each device update is
        one naturally aligned 8-byte atomic, so a load returns some value the
        slot actually held, never a torn mix;
      * so any load after event N is observed complete returns a value >= the
        value at N's completion, possibly larger (N+1's increments early). That
        is all the fatal decision needs: it arms on count > last armed count.

    One event is recorded at the end of every engine call (engine_call ->
    _call_exit), on the calling thread's current stream = the stream every
    guarded kernel runs on (single-stream invariant, enforced when the idle gate
    is armed), plus one at creation that covers everything before the reader
    started (warmup, capture). The idle fence keeps its own events and its
    stronger drain: those are recorded by the reader only at idle, so they cannot
    serve as per-step markers.

    advance(observer) queries outstanding records newest first and returns True
    when a record newer than that observer's last read has completed: only then
    may the observer read. A ring slot re-recorded before its old record was
    queried is credited with the sequence number read BEFORE the query (record
    writes the event, then the number), which can only under-credit, never
    credit work that has not completed.
    """

    def __init__(self, device: torch.device) -> None:
        self.events = [_new_step_event(device) for _ in range(_STEP_RING)]
        self.slot_seq = [0] * _STEP_RING
        self.rec_seq = 0  # records issued (engine thread only)
        self.done_seq = 0  # newest record observed complete
        self.seen = {"engine": 0, "reader": 0}
        self.queries = 0  # cudaEventQuery calls issued (cost accounting)
        self.lock = threading.Lock()
        self.record()  # covers everything enqueued before the reader started

    def record(self) -> None:
        seq = self.rec_seq + 1
        i = seq % _STEP_RING
        self.events[i].record()  # 1. the event ...
        self.slot_seq[i] = seq  # 2. ... then its number
        self.rec_seq = seq

    def _complete(self, seq: int) -> bool:
        """Record ``seq`` observed complete. False when its slot no longer
        (or not yet) holds ``seq`` -- never credit work we did not observe."""
        i = seq % _STEP_RING
        if self.slot_seq[i] != seq:
            return False
        self.queries += 1
        return bool(self.events[i].query())

    def advance(self, observer: str) -> bool:
        """True when a record newer than ``observer``'s last read has been
        observed complete. The queries run OUTSIDE the lock (the lock guards
        only the snapshot and the commit), so neither observer ever waits on
        the other's CUDA calls. Records on one stream complete in order, so
        the newest complete record is found by one query of the newest record
        and, if that one is still running, a bisection of the outstanding
        window: at most 1 + log2(_STEP_RING) = 5 queries, typically 1-2."""
        with self.lock:
            done, top = self.done_seq, self.rec_seq
        lo = max(done, top - _STEP_RING)  # newest seq not to query again
        found = 0
        if top > lo:
            if self._complete(top):
                found = top
            else:
                hi = top  # known (or treated) incomplete
                while hi - lo > 1:
                    mid = (lo + hi) // 2
                    if self._complete(mid):
                        lo = found = mid
                    else:
                        hi = mid
        with self.lock:
            if found > self.done_seq:
                self.done_seq = found
            done = self.done_seq
            if done > self.seen[observer]:
                self.seen[observer] = done
                return True
            return False


# device -> _StepEvents, created by start_reader() (engine thread, after capture)
_step_events: dict[torch.device, _StepEvents] = {}


def _record_step_events() -> None:
    """_call_exit: one event per mapped device, after the call's work. A
    failure arms a contract-failure stop (the visibility rule cannot be kept)."""
    try:
        for se in list(_step_events.values()):
            se.record()
    except Exception as exc:  # noqa: BLE001
        _arm(
            f"bounds-guard step event record failed ({type(exc).__name__}: "
            f"{exc}); the fatal-site contract cannot be enforced on this rank."
        )


_post_capture_hooks: list = []
_post_capture_done = False


def register_post_capture_hook(fn) -> None:
    """Run fn() once after warmup and cudagraph capture (immediately if that already happened)."""
    if _post_capture_done:
        fn()
    else:
        _post_capture_hooks.append(fn)


def run_post_capture_hooks() -> None:
    global _post_capture_done
    _post_capture_done = True
    hooks, _post_capture_hooks[:] = list(_post_capture_hooks), []
    for fn in hooks:
        try:
            fn()
        except Exception:  # noqa: BLE001 -- one broken starter must not stop the others or the engine
            logger.exception("post-capture reader start failed: %r", fn)


def start_reader() -> None:
    """Start the cadence reader. Called AFTER warmup and cudagraph capture.

    Also runs the post-capture hooks of the other readers (register_post_capture_hook), whatever the guard mode.

    The reader must not exist during capture. It copies device counters to the
    host, and a device-to-host copy from a second thread while a capture is
    active is exactly the call class cudaStreamCaptureModeGlobal -- the default
    for torch.cuda.graph -- rejects. Starting after capture avoids the question.

    Between prealloc() and this call the kernels still count; nothing reads.
    """
    global _reader_started, _reader_thread, _step_check_on, _drain_streams
    run_post_capture_hooks()
    for device in list(_host_views):
        se = _step_events.get(device)
        if se is None:
            _step_events[device] = _StepEvents(device)
        else:
            se.record()  # a restarted reader: cover the gap since the last call
    _step_check_on = True
    with _life_lock:
        if _reader_started:
            t = _reader_thread
            if t is not None and t.is_alive():
                if _reader_stop.is_set():
                    logger.error(
                        "bounds-guard reader NOT started: a stopped reader from an "
                        "earlier worker in this process is still running; guard "
                        "telemetry for this worker is not read."
                    )
                return
            # A reader whose stop timed out has exited since: reap it.
            _reset_lifecycle()
        _drain_streams = {}  # an earlier worker's fence: never this one's
        _reader_started = True
        _reader_thread = threading.Thread(
            target=_reader, name="bounds-guard-reader", daemon=True
        )
        _reader_thread.start()
    logger.info(
        "bounds-guard reader started (mode=%s, interval=%.3gs, transport=%s).",
        _MODE,
        _INTERVAL,
        _transport_summary(),
    )


def _transport_summary() -> str:
    """"mapped", "copy", or a per-device mix (a device fell back)."""
    cuda = [d for d in _buffers if d.type == "cuda"]
    if not cuda:
        return _TRANSPORT
    kinds = {("mapped" if d in _host_views else "copy") for d in cuda}
    if len(kinds) == 1:
        return kinds.pop()
    return ",".join(
        f"{d}={'mapped' if d in _host_views else 'copy'}" for d in sorted(cuda, key=str)
    )


def _stream_key(stream) -> tuple:
    return (stream.stream_id, stream.device_index, stream.device_type)


def _current_stream_key(device: torch.device) -> tuple:
    """Identity of the calling thread's current stream on ``device``. Host-side
    only, no device work: the C++ binding returns the triple without building a
    Python Stream (torch.cuda.current_stream does build one)."""
    try:
        return tuple(torch._C._cuda_getCurrentStream(device.index))
    except AttributeError:
        return _stream_key(torch.cuda.current_stream(device))


def _off_fence_stream(device: torch.device) -> bool:
    """True when the calling thread's current stream on ``device`` is not a
    fence stream. False when there is nothing to compare (CPU device, gate not
    armed on this device). Never raises: an error counts as off-stream."""
    keys = _fence_keys.get(device)
    if not keys:
        return False
    try:
        return _current_stream_key(device) not in keys
    except Exception:  # noqa: BLE001
        return True


def enable_idle_gate(device: torch.device, streams) -> bool:
    """Arm the idle quiesce for this process (see IDLE QUIESCE).

    The worker calls this on its engine thread, after capture and AFTER
    start_reader() (which runs the post-capture hooks first; the reader polls
    on the cadence until the gate is armed), and only once its GPU entry points
    are wrapped in engine_call. ``streams`` must be every stream the guarded
    kernels launch on: for the V1 and V2 runners that is the engine thread's
    current stream (the forward and its graph replays, the MTP proposal and its
    graphs, the RecoverSSM commit chain and the replay of its commit graph, the
    block-table / slot-mapping / KV-zero / mamba-copy kernels; the runners' side
    streams only copy). SINGLE-STREAM INVARIANT, enforced at run time: every
    engine call and every eager guarded launch (counter()) checks that it runs
    on the engine thread and on a fence stream, else the gate falls back to
    polling forever with an ERROR line -- never a silent miss. Graph replays do
    not pass through counter(); they inherit the stream of the engine call that
    replays them, which the call check covers. One fence event per stream is
    created here, outside any capture, and only ever re-recorded by the reader.
    Returns whether the gate is armed.
    """
    global _gate_armed, _engine_thread, _engine_device, _gate_gen
    streams = list(streams)
    if _host_views and len(streams) > 1:
        # The step-event ring orders records on ONE stream (bisection).
        raise ValueError(
            "guard_telemetry: TRANSPORT=mapped supports exactly one fence "
            f"stream, got {len(streams)}; use VLLM_E2_GUARD_TRANSPORT=copy"
        )
    device = _normalize(device)
    pairs: list = []
    if device.type == "cuda":
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "guard_telemetry.enable_idle_gate() called during cudagraph capture"
            )
        with torch.cuda.device(device):
            pairs = [(s, torch.cuda.Event()) for s in streams]
        if not pairs:
            raise ValueError("enable_idle_gate needs at least one CUDA stream")
    with _life_lock, _cv:
        _gate_gen += 1
        _fence_events[device] = pairs
        _fence_keys[device] = frozenset(_stream_key(s) for s, _ in pairs)
        _engine_thread = threading.get_ident()
        _engine_device = device
        _gate_armed = True
    logger.info(
        "bounds-guard reader: idle quiesce armed on %s (%d fence stream(s)); the "
        "reader parks with no device work after a final drain once the engine "
        "is idle.",
        device,
        len(pairs),
    )
    return True


def engine_call(fn):
    """Decorator for the worker's GPU entry points: marks an engine call open
    for the idle quiesce. Engine-thread cost per call: two uncontended lock
    round trips, two integer updates and one host-side current-stream lookup --
    no device work, no sync -- plus, under TRANSPORT=mapped once the reader has
    started, one cudaEventRecord at exit (the step visibility marker,
    _StepEvents; never a wait). Applied unconditionally; the idle part only
    matters once enable_idle_gate() has run."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        _call_enter()
        try:
            return fn(*args, **kwargs)
        finally:
            _call_exit()

    return wrapper


def _call_enter() -> None:
    """Never raises (the wrapper's exit must always run)."""
    global _call_seq, _calls_open, _reader_parked
    with _cv:
        _calls_open += 1
        _call_seq += 1
        if _reader_parked:
            _reader_parked = False
            _cv.notify_all()
        snap = (
            (_gate_gen, _engine_thread, _engine_device)
            if _gate_armed and _gate_broken is None
            else None
        )
    if snap is not None:
        gen, owner, device = snap
        if threading.get_ident() != owner:
            _break_gate(
                "engine call entered on a thread other than the arming one",
                gen,
                coverage=True,
            )
        elif device is not None and _off_fence_stream(device):
            _break_gate(
                "engine call runs on a stream the fence does not cover",
                gen,
                coverage=True,
            )


def check_replay_stream() -> None:
    """Call immediately before a CUDA graph replay that may contain guarded
    kernels (the V2 runner's FULL graphs, CUDAGraphWrapper, the breakable
    graphs, the ubatch wrapper). A replay inherits the CALLER's current stream,
    and no guarded launch inside it passes through counter(), so the producer-
    coverage check has to happen here: a replay outside an engine call, off the
    engine thread, or on a stream the fence does not cover breaks coverage
    (and, under TRANSPORT=mapped, arms a contract-failure stop). Lock-free:
    a tuple of GIL-atomic reads and one host-side current-stream lookup."""
    if not (_gate_armed and _gate_broken is None):
        return
    gen, owner, device, calls_open = _gate_gen, _engine_thread, _engine_device, _calls_open
    if calls_open == 0 or threading.get_ident() != owner:
        _break_gate(
            "CUDA graph replay outside an engine call or off the engine thread",
            gen,
            coverage=True,
        )
    elif device is not None and _off_fence_stream(device):
        _break_gate(
            "CUDA graph replay on a stream the fence does not cover", gen, coverage=True
        )


def _call_exit() -> None:
    global _call_seq, _calls_open
    if _gate_armed and _gate_broken is None:
        gen, device = _gate_gen, _engine_device
        if device is not None and _off_fence_stream(device):
            _break_gate(
                "engine call exits on a stream the fence does not cover",
                gen,
                coverage=True,
            )
    if _step_check_on and _step_events:
        # TRANSPORT=mapped: the visibility marker for this call's work (one
        # cudaEventRecord on the engine thread per engine call).
        _record_step_events()
    with _cv:
        _calls_open -= 1
        _call_seq += 1


def _gate_live() -> bool:
    return _gate_armed and _gate_broken is None


def _coverage_live() -> bool:
    """The producer-coverage checks are armed and unbroken: only then may a
    mapped read be gated on the step events (every producer is then ordered
    before the engine-stream event that covers it)."""
    return _gate_armed and _gate_broken is None


def _break_gate(reason: str, gen: int | None = None, coverage: bool = False) -> None:
    """Fall back to polling forever (the plain cadence reader). Loud: a protocol
    violation must never turn into a silent miss. One device per worker process,
    so this is that device's gate. ``gen``: the gate generation the violation
    was observed under; a gate re-armed since is left alone.

    ``coverage``: the violation is a PRODUCER outside the covered stream /
    thread / engine call (it only changes the wording). Under TRANSPORT=mapped
    EVERY break arms a contract-failure stop FIRST, then logs (best effort):
    after a break the coverage checks stop, so the step contract cannot be kept
    whatever the cause (a fence record failure included). Chosen over joining every
    producer stream before the step event: the violating producer's stream is
    by definition unknown to the protocol, a join would add a GPU wait to the
    step path, and no production path launches a guarded kernel off the
    engine stream -- so
    the rule is fail-closed. Under copy the reader's cadence polling keeps
    detection as before and nothing is armed."""
    global _gate_broken
    with _cv:
        if _gate_broken is not None or (gen is not None and gen != _gate_gen):
            return
        _gate_broken = reason
        _cv.notify_all()
    if _host_views:
        kind = "producer-stream coverage" if coverage else "idle/coverage gate"
        _arm(
            f"bounds-guard {kind} broken under TRANSPORT=mapped ({reason}); the "
            f"mapped step-boundary contract cannot be kept on this rank (use "
            f"VLLM_E2_GUARD_TRANSPORT=copy for a workload with off-stream "
            f"guarded producers)."
        )
    try:
        _log_break(reason)
    except Exception:  # noqa: BLE001 -- the stop (if any) is armed already
        pass


def _log_break(reason: str) -> None:
    logger.error(
        "bounds-guard reader: idle quiesce DISABLED for this process (%s); "
        "the reader polls every %.3gs from now on, idle or not (no hit is "
        "missed; idle cards stay out of their idle P-state).%s",
        reason,
        _INTERVAL,
        (
            " TRANSPORT=mapped: coverage can no longer be shown; a "
            "contract-failure stop was ARMED (the next execute_model raises), "
            "and until then the reader reads the mapped counters UNGATED on "
            "every tick, so the step-boundary contract is CADENCE-BOUND for "
            "producers off the fenced stream (a hit there is read within one "
            "interval)."
            if _host_views
            else ""
        ),
    )


def _record_pairs(device: torch.device, pairs) -> list:
    """Record every (stream, event) pair on ``device``; the events, in order.
    Raises on a failed record: the caller decides (the reader breaks the gate,
    the shutdown drain only gives up)."""
    events = []
    for stream, event in pairs:
        with torch.cuda.device(device):
            event.record(stream)
        events.append(event)
    return events


def _record_fence() -> list | None:
    """Reader thread: one event on every fence stream of every device. The
    engine thread's streams accept the record from this thread; work enqueued
    before it is covered, and a call that enqueues after it bumps _call_seq
    first, which discards the fence. A failed record falls back to polling
    forever (power, not the rank, pays for it)."""
    for device in list(_buffers):
        if device.type == "cuda" and device not in _fence_events:
            _break_gate(f"counter buffer on {device} has no fence stream")
            return None
    events = []
    for device, pairs in list(_fence_events.items()):
        try:
            events.extend(_record_pairs(device, pairs))
        except Exception:  # noqa: BLE001
            # Publish the break (and, under mapped, arm the stop) FIRST; the
            # log is best effort and can neither delay nor skip either.
            _break_gate(f"fence record failed on {device}")
            try:
                logger.exception(
                    "bounds-guard reader: fence record failed on %s", device
                )
            except Exception:  # noqa: BLE001
                pass
            return None
    _stats["fences"] += 1
    return events


def _final_drain(fence_seq: int) -> bool:
    """The fence completed: read every counter once, coherently, and report.
    False = no park: a call started meanwhile (its kernels may run beside the
    copy, so those rows go through the ordinary confirm protocol, or the wait
    is cancelled with the copy left in flight for the cadence path), a stop or
    suspend was requested, or a copy did not land in time."""
    _stats["drains"] += 1

    def cancelled() -> bool:
        # Plain reads, a hint only: the authoritative check is under _cv below.
        return (
            _reader_stop.is_set()
            or _suspend_req
            or _call_seq != fence_seq
            or _calls_open != 0
        )

    for device, buf in list(_buffers.items()):
        if device in _read_disabled:
            continue
        try:
            host = _host_views.get(device)
            if host is not None:
                # mapped: the fence has completed, so every write of every
                # producer is visible in host memory -- nothing to copy or wait for
                rows = host.tolist()
            else:
                if not _nb_reader.registered(device):
                    _nb_reader.register(device, buf)  # reader thread: never capturing
                rows = _nb_reader.read_settled(device, _DRAIN_TIMEOUT_S, cancelled)
        except Exception:  # noqa: BLE001 -- same contract as _poll_once
            _lose_device(device)
            continue
        if rows is None:
            if not cancelled():
                logger.warning(
                    "bounds-guard reader: final drain copy on %s did not complete "
                    "in %.1fs; staying active",
                    device,
                    _DRAIN_TIMEOUT_S,
                )
            return False
        with _cv:
            quiet = _call_seq == fence_seq and not _calls_open
        if not quiet:
            # A call entered before the copy landed: not a coherent snapshot.
            _consume(device, rows)
            return False
        pend = _confirm_pending.pop(device, None)
        if pend is not None:
            rows = [[max(a[0], b[0]), b[1], b[2]] for a, b in zip(pend, rows)]
        _report_rows(device, rows)
    return True


_park_logged = False


def _park(fence_seq: int) -> bool:
    """Wait with no CUDA call until an engine call starts (or a stop or suspend
    request). False, without waiting, when a call started since the fence."""
    global _reader_parked, _park_logged
    with _cv:
        if (
            _call_seq != fence_seq
            or _calls_open
            or not _gate_live()
            or _reader_stop.is_set()
            or _suspend_req
        ):
            return False
        _reader_parked = True
        _stats["parks"] += 1
        t0 = time.monotonic()
        first = not _park_logged
        _park_logged = True
        try:
            while (
                _call_seq == fence_seq
                and _gate_broken is None
                and not _reader_stop.is_set()
                and not _suspend_req
            ):
                if first:
                    # Log once, outside the lock: an enter is never held up by
                    # logging I/O. A call that starts meanwhile is seen below.
                    first = False
                    _cv.release()
                    try:
                        logger.info(
                            "bounds-guard reader: engine idle; final drain read "
                            "every counter after the last call's GPU work "
                            "completed; parked with no device work until the "
                            "next engine call (logged once; resumes after >= 5 s "
                            "parked are logged)."
                        )
                    finally:
                        _cv.acquire()
                    continue
                _cv.wait()
        finally:
            _reader_parked = False
        _stats["resumes"] += 1
    parked = time.monotonic() - t0
    if parked >= 5.0 and not _reader_stop.is_set():
        logger.info(
            "bounds-guard reader resumed after %.1fs parked (parks %d, drains %d)",
            parked,
            _stats["parks"],
            _stats["drains"],
        )
    return True


def _suspend_point(gen: int) -> None:
    """Reader thread: honour suspend_reader() request ``gen``. The caller
    (Worker.sleep) has synchronised the device and no engine call runs, so one
    settled read per device is coherent: report it, acknowledge, then wait with
    no CUDA call until resume_reader(), a newer request, or stop_reader(). Never
    acknowledges with a copy still in flight: a cancelled or racing read (the
    sleeper gave up, a call started) returns without acknowledging."""
    global _asleep, _suspended_gen

    def cancelled() -> bool:
        return _reader_stop.is_set() or not _suspend_req or _suspend_gen != gen

    with _cv:
        seq0 = _call_seq
    with _io_lock:
        for device, buf in list(_buffers.items()):
            if device in _read_disabled:
                continue
            try:
                host = _host_views.get(device)
                if host is not None:
                    rows = host.tolist()  # mapped: the sleeper synchronised the device
                else:
                    if not _nb_reader.registered(device):
                        _nb_reader.register(device, buf)
                    # No deadline of its own: the sleeper's timeout cancels it.
                    rows = _nb_reader.read_settled(device, float("inf"), cancelled)
            except Exception:  # noqa: BLE001
                _lose_device(device)
                continue
            if rows is None:
                return  # cancelled; the copy stays in flight for the cadence path
            with _cv:
                coherent = (
                    _suspend_req
                    and _suspend_gen == gen
                    and _call_seq == seq0
                    and not _calls_open
                )
            if not coherent:
                _consume(device, rows)
                return
            pend = _confirm_pending.pop(device, None)
            if pend is not None:
                rows = [[max(a[0], b[0]), b[1], b[2]] for a, b in zip(pend, rows)]
            _report_rows(device, rows)
        # The barrier goes up while _io_lock is still held: drain() checks
        # _asleep under the same lock, so no read can slip in between.
        with _cv:
            if cancelled():
                return
            _asleep = True
            _suspended_gen = gen
            _cv.notify_all()
    _sleep_while_asleep()


def _sleep_while_asleep() -> None:
    """Reader thread: no CUDA call while the worker sleeps (also for a reader
    started while asleep), until resume_reader() or stop_reader()."""
    global _reader_suspended
    with _cv:
        _reader_suspended = True
        _cv.notify_all()
        try:
            while _asleep and not _reader_stop.is_set():
                _cv.wait()
        finally:
            _reader_suspended = False


def _reader() -> None:
    """Off-engine-thread reader; drain() is the only other counter read.

    ACTIVE -> FENCE -> DRAIN -> PARKED, see IDLE QUIESCE; SUSPENDED while the
    worker sleeps. Without a live gate it is the plain cadence reader."""
    seq_prev = -1
    fence = None  # (call generation, events)
    resumed = False
    while not _reader_stop.is_set():
        try:
            # Right after a resume the first copy starts at once (nothing is in
            # flight after a drain), so resume costs no extra tick.
            if not resumed and _reader_stop.wait(_INTERVAL):
                break
            resumed = False
            if _asleep:
                fence, seq_prev = None, -1
                _sleep_while_asleep()
                continue
            if _suspend_req:
                fence, seq_prev = None, -1
                _suspend_point(_suspend_gen)
                continue
            if not _gate_live():
                fence = None
                with _io_lock:
                    _poll_once()
                continue
            with _cv:
                seq, open_ = _call_seq, _calls_open
            if fence is not None and (fence[0] != seq or open_):
                fence = None  # a call started: the fence no longer covers it
            if fence is not None and all(e.query() for e in fence[1]):
                fence_seq, fence = fence[0], None
                seq_prev = -1
                with _io_lock:
                    drained = _final_drain(fence_seq)
                if drained:
                    resumed = _park(fence_seq)
                continue
            with _io_lock:
                _poll_once()
            if fence is None and open_ == 0 and seq == seq_prev:
                events = _record_fence()
                if events is not None:
                    fence = (seq, events)
            seq_prev = seq
        except Exception:  # noqa: BLE001 -- the daemon itself must survive
            fence = None
            seq_prev = -1
            _arm(  # first: logging below is best effort
                "bounds-guard reader loop failed; the fatal-site contract "
                "(_ALWAYS_FATAL_SITES) can no longer be enforced."
            )
            try:
                logger.exception("bounds-guard reader loop failed")
            except Exception:  # noqa: BLE001
                pass
            time.sleep(max(_INTERVAL, 0.05))


def wait_idle(timeout: float) -> bool:
    """Block the CALLER until the reader has parked after a final drain, so
    every counter write so far has been reported. Never for a step path: the
    caller must not be inside an engine call. False when the gate is not live
    (the reader polls forever) or on timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        t = _reader_thread
        if t is None or not t.is_alive() or not _gate_live():
            return False
        with _cv:
            if _reader_parked and not _calls_open:
                return True
        time.sleep(0.005)
    return False


def suspend_reader(timeout: float = 10.0) -> bool:
    """Lifecycle (Worker.sleep, after it synchronised the device): the reader
    reads every counter once more, reports, and then makes no CUDA call until
    resume_reader() -- whatever the idle switch or the gate state, so a sleeping
    worker never has its counters read. True when already asleep, when no
    reader runs, or when the reader has acknowledged THIS request with nothing
    in flight (the barrier _asleep is raised under _io_lock, so drain() cannot
    read either; a reader started while asleep stays suspended). False on
    timeout: the request is withdrawn (the reader goes back to work) and the
    caller must not treat the reader as quiescent (Worker.sleep refuses to
    sleep)."""
    global _suspend_req, _suspend_gen, _asleep
    with _life_lock:  # one suspend at a time, never beside a start/stop/resume
        if _asleep:
            return True  # already suspended: nothing new to read
        t = _reader_thread
        if t is None or not t.is_alive():
            with _io_lock, _cv:
                _asleep = True
            return True
        deadline = time.monotonic() + timeout
        with _cv:
            _suspend_gen += 1
            gen = _suspend_gen
            _suspend_req = True
            _cv.notify_all()
            try:
                while not (_asleep and _suspended_gen == gen):
                    remaining = deadline - time.monotonic()
                    if remaining <= 0 or not t.is_alive():
                        return False
                    _cv.wait(remaining)
            finally:
                _suspend_req = False  # served or withdrawn
                _cv.notify_all()
    return True


def resume_reader() -> None:
    """Lifecycle (Worker.wake_up, after the device state is restored)."""
    global _suspend_req, _asleep
    with _life_lock, _cv:
        _suspend_req = False
        _asleep = False
        _cv.notify_all()


def _reset_lifecycle() -> None:
    """Under _life_lock, with no reader thread running: back to the state of a
    fresh import, except the counters, their high-water marks and _asleep."""
    global _reader_started, _reader_thread, _gate_armed, _gate_broken, _gate_gen
    global _engine_thread, _engine_device, _suspend_req, _reader_parked
    global _reader_suspended
    with _cv:
        _reader_thread = None
        _reader_started = False
        _gate_gen += 1
        _gate_armed = False
        _gate_broken = None
        _engine_thread = None
        _engine_device = None
        _suspend_req = False
        _reader_suspended = False
        _reader_parked = False
        _fence_events.clear()
        _fence_keys.clear()
        _reader_stop.clear()


def stop_reader(timeout: float = 2 * _DRAIN_TIMEOUT_S + 1.0) -> bool:
    """Stop and join the reader (orderly shutdown, before drain()). Wakes a
    parked or suspended reader. After a successful join the lifecycle is reset
    (under the lifecycle lock, so a concurrent start or arm cannot be erased),
    and a later worker in this process starts a fresh reader and arms its own
    gate; the counter buffers and high-water marks stay (captured graphs may
    still hold their addresses, and a reported hit is not reported twice).
    False, loudly, when the thread did not stop; start_reader() then refuses
    to run a second one while it lives and reaps it once it has exited."""
    with _life_lock:
        _reader_stop.set()
        with _cv:
            _cv.notify_all()
        t = _reader_thread
        if t is not None:
            t.join(timeout)
            if t.is_alive():
                logger.error("bounds-guard reader did not stop within %.1fs", timeout)
                return False  # not reset: drain() finds the live fence
        _stash_drain_streams()
        _reset_lifecycle()
    return True


def _stash_drain_streams() -> None:
    """Under _life_lock, reader joined: keep the fence streams for drain(), but
    only while coverage is live (else they do not cover every producer)."""
    global _drain_streams
    with _cv:
        _drain_streams = (
            {d: [s for s, _ in p] for d, p in _fence_events.items() if p}
            if _coverage_live()
            else {}
        )


def _report_rows(device: torch.device, rows: list[list[int]]) -> None:
    """Report every (device, site) whose masked count advanced. Shared by the
    cadence reader, the drains and the confirmation path.

    Two passes: every fatal candidate in this snapshot is claimed and armed
    BEFORE any row is logged, so a blocked or raising logger on an earlier
    (even nonfatal) row cannot delay or skip a later fatal row's stop."""
    reports = []  # (site, n_masked, message, fatal)
    for site, (n_masked, biased_row, biased_val) in enumerate(rows):
        if n_masked <= _seen.get((device, site), 0):
            continue
        _seen[(device, site)] = n_masked
        row = decode_slot(biased_row)
        val = decode_slot(biased_val)
        if site == SITE_QSA_RING_UNCOMMITTED:
            # One coherent first-offender record, not independent maxima.
            record = decode_ring_record(biased_row)
            msg = (
                f"ring-row tag guard masked {n_masked} uncommitted historical "
                f"ring row(s) in {SITE_NAMES[site]} on {device}; first "
                f"offender {record}, its row held tag {val} (expected "
                f"source_position + 1). The group was not pooled and its "
                f"compressed slot was NOT written."
            )
            reports.append((site, n_masked, msg, True))
            continue
        msg = (
            f"bounds guard masked {n_masked} out-of-range access(es) "
            f"in {SITE_NAMES.get(site, site)} on {device} "
            f"(cumulative maxima: row-field {row}, index-field {val}; these "
            f"are independent lifetime maxima, not one observed pair). The "
            f"access was suppressed, so this is a MASKED FAULT and not a "
            f"correct result -- the metadata is wrong upstream."
        )
        if row is None or val is None:
            msg += (
                " A None field is UNOBSERVED OR UNENCODABLE, not absent: "
                "either this copy predates the slot's atomic_max becoming "
                "visible, or the value was == INT32_MIN, below it, or past "
                f"{GUARD_MAX_ENCODABLE}. The reader has no ordering "
                "dependency on the producer, so it cannot tell which; see "
                "decode_slot."
            )
        fatal = _MODE == "raise" or site in _ALWAYS_FATAL_SITES
        reports.append((site, n_masked, msg, fatal))
    # Pass 1: arm every fatal row of the snapshot (no logging inside).
    for site, n_masked, msg, fatal in reports:
        if fatal:
            _claim_and_arm(device, [(site, n_masked)], lambda _r, m=msg: m)
    # Pass 2: log.
    for _site, _n, msg, _fatal in reports:
        logger.warning("%s", msg)


def _poll_once() -> None:
    for device, buf in list(_buffers.items()):
        if device in _read_disabled:
            continue
        host = _host_views.get(device)
        if host is not None:
            _stats["polls"] += 1
            try:
                se = _step_events.get(device)
                if (
                    se is not None
                    and _coverage_live()
                    and not se.advance("reader")
                    and device not in _confirm_pending
                ):
                    continue
                rows = host.tolist()
            except Exception:  # noqa: BLE001 -- quarantine, as on the copy path
                _lose_device(device)
                continue
            _stats["reads"] += 1
            _consume(device, rows)
            continue
        try:
            # A non-blocking side-stream copy; None means the
            # previous copy is still in flight (report on a later tick).
            if not _nb_reader.registered(device):
                _nb_reader.register(device, buf)  # reader thread: never capturing
            _stats["polls"] += 1
            rows = _nb_reader.poll_one(device)
            if rows is None:
                continue
        except Exception:  # noqa: BLE001 -- a reader must never kill the run
            _lose_device(device)
            continue
        _consume(device, rows)


def _consume(device: torch.device, rows: list) -> None:
    """Cadence path for one copy that may have raced the producers."""
    pend = _confirm_pending.get(device)
    rising = [
        site
        for site, (n, _r, _v) in enumerate(rows)
        if n > _seen.get((device, site), 0)
    ]
    if pend is None:
        if rising:
            _confirm_pending[device] = rows
            candidates = [
                (s, rows[s][0])
                for s in rising
                if _MODE == "raise" or s in _ALWAYS_FATAL_SITES
            ]
            if candidates:
                _risen, msg = _claim_and_arm(
                    device,
                    candidates,
                    lambda risen: (
                        "bounds guard masked-access count rose at "
                        + ", ".join(
                            f"{SITE_NAMES.get(s, s)} (count {n})" for s, n in risen
                        )
                        + f" on {device} (first observation; fatal by count; the "
                        "recorded maxima are reported best effort on the "
                        "confirming copy)."
                    ),
                )
                if msg is not None:
                    logger.warning("%s", msg)
        return
    _confirm_pending.pop(device, None)
    # report with the CONFIRMING copy's values (never lower than the
    # pending copy's counts: the counters are monotone)
    rows = [[max(a[0], b[0]), b[1], b[2]] for a, b in zip(pend, rows)]
    _report_rows(device, rows)


def _lose_device(device: torch.device) -> None:
    """A counter read failed: stop polling the device, keep holding it."""
    _arm(
        f"bounds-guard reader lost the counter on {device}; the "
        f"fatal-site contract (_ALWAYS_FATAL_SITES) is no longer "
        f"enforced there."
    )
    _read_disabled.add(device)
    try:
        logger.exception(
            "bounds-guard reader failed to read the counter on %s; "
            "no further guard telemetry will be reported for it. The "
            "buffer is kept alive: captured graphs still write to it.",
            device,
        )
    except Exception:  # noqa: BLE001
        pass
    pend = _confirm_pending.pop(device, None)
    if pend is not None:
        try:
            _report_rows(device, pend)  # claims arm before they log
        except Exception:  # noqa: BLE001
            pass


def _arm_locked(msg: str) -> None:
    """Under _fatal_lock. Accumulates: a second stop armed before the first is
    consumed is appended, never dropped."""
    global _pending
    if _pending is None:
        _pending = msg
    elif msg not in _pending:
        _pending = f"{_pending} || {msg}"


def _arm(msg: str) -> None:
    with _fatal_lock:
        _arm_locked(msg)


def _claim_and_arm(device: torch.device, candidates, describe):
    """Publish a fatal stop for every (site, count) in ``candidates`` whose
    count is above the last one armed for (device, site) -- by either observer
    (the cadence reader or the engine-thread step check). ONE critical section:
    the pending stop is written first, then the high-water mark, and nothing in
    it logs, so a count is never claimed without being armed. (The step check's
    lock-free fast path reads the high-water mark and then _pending; under the
    GIL a mark it sees raised implies the stop it guards is already pending.)
    ``describe(risen)`` builds the message; if it raises, a generic message is
    armed instead. Returns (risen, message), ([], None) when nothing rose.
    Logging is the caller's, AFTER this returns."""
    with _fatal_lock:
        hw = _fatal_hw.get(device)
        if hw is None:
            hw = _fatal_hw[device] = [0] * NUM_SITES
        risen = [(s, int(n)) for s, n in candidates if n > hw[s]]
        if not risen:
            return [], None
        try:
            msg = describe(risen)
        except Exception as exc:  # noqa: BLE001 -- never strand a claim
            msg = (
                f"bounds guard fatal count rose on {device}: {risen} (message "
                f"build failed: {type(exc).__name__})"
            )
        _arm_locked(msg)  # 1. the stop
        for s, n in risen:  # 2. then the mark that suppresses re-arming
            hw[s] = n
    return risen, msg


def _fatal_sites() -> tuple:
    return tuple(range(NUM_SITES)) if _MODE == "raise" else tuple(sorted(_ALWAYS_FATAL_SITES))


_FATAL_SITES = _fatal_sites()


def _step_check() -> None:
    """Engine thread, top of execute_model (TRANSPORT=mapped): arm a stop for
    any fatal-site count above the last armed one, reading the mapped counters
    ONLY after a step event newer than the last check has completed (see
    _StepEvents). Cost: one or two cudaEventQuery on mapped CUDA devices, then
    one host-memory read per fatal site (two under warn, twelve under raise);
    one uncontended se.lock round trip per mapped device, _fatal_lock only when
    a count rose. Any exception arms a contract-failure stop
    (never a silent fall-through)."""
    try:
        for device, counts in _step_views:
            se = _step_events.get(device)
            if se is not None and _coverage_live() and not se.advance("engine"):
                continue  # no newly completed step work: nothing new is visible
            # (coverage not live: a stop is armed or IDLE=0 without coverage;
            # read ungated -- an early read is harmless, see _StepEvents)
            hw = _fatal_hw.get(device)
            for site in _FATAL_SITES:
                if counts[site] > (hw[site] if hw is not None else 0):
                    break
            else:
                continue
            _risen, msg = _claim_and_arm(
                device,
                [(site, int(counts[site])) for site in _FATAL_SITES],
                lambda risen, d=device: (
                    "bounds guard masked-access count rose at "
                    + ", ".join(f"{SITE_NAMES.get(s, s)} (count {n})" for s, n in risen)
                    + f" on {d} (step-boundary check on the engine thread; "
                    "fatal by count; the reader reports the recorded maxima)."
                ),
            )
            if msg is not None:
                logger.warning("%s", msg)  # after the stop is armed
    except Exception as exc:  # noqa: BLE001
        _arm(
            "bounds-guard step-boundary check failed "
            f"({type(exc).__name__}: {exc}); the fatal-site contract cannot be "
            "enforced on this rank."
        )
        try:
            logger.exception("bounds-guard step-boundary check failed")
        except Exception:  # noqa: BLE001 -- the stop is armed either way
            pass


def check_pending() -> None:
    """Engine-thread check. One Python bool per step; adds no sync.

    Called at the top of execute_model, ahead of anything that mutates state
    and ahead of any collective, so the step it lands on aborts cleanly.

    Two things it does not do:

    1. It does NOT raise on the step that fired. The reader polls on a cadence:
       the count is armed a tick or two after the hit, and the raise lands at the
       next execute_model after that -- several decode steps on, more under async
       scheduling. The scheduler state at the raise belongs to some later step,
       not the offending one. Attribution comes from the counter and the
       WARNING; the raise is a stop, not a snapshot.
    2. It does NOT coordinate across ranks. The counters are per process, so only
       the rank whose GPU fired arms. Its PP/TP peers proceed into the forward
       and block in recv/allreduce until the NCCL watchdog fires, which is a
       worse diagnostic than the Xid being chased.

    Hence VLLM_E2_GUARD_MODE=warn for long stress boots; raise only for a targeted
    repro where losing the group is acceptable and stopping is the point.

    TRANSPORT=mapped tightens point 1 (not point 2). The contract: every engine
    call on this rank ends with a CUDA event recorded on the stream the guarded
    kernels run on (_StepEvents). A fatal-site hit written by the work of engine
    call N is stopped by the first check_pending on this rank that runs after call
    N's event has completed: the check queries the event itself, reads the mapped
    counters only once it has completed, and arms and raises here. No step-count
    bound is part of the contract: how many calls run between N and that check
    depends on the scheduler (async PP admits several concurrent batches, and the
    output copy starts before the MTP proposal, so consuming an output does not
    imply the call's exit event has completed). It holds on every PP stage and
    relies on no PCIe ordering argument (see _StepEvents).

    Preconditions, each enforced fail-closed: every guarded producer runs on the
    engine thread, inside an engine call, on the fenced stream -- checked at every
    eager guarded launch, every engine call entry and every graph replay
    (check_replay_stream); a violation under mapped ARMS a contract-failure stop
    before it is logged. Bounded by the event ring: fewer than _STEP_RING (16)
    calls in flight on the rank, which is NOT enforced (beyond it detection is
    later, never earlier). The cadence reader reads under the same rule and is the
    backstop for the WARNING.
    """
    global _pending
    if _step_check_on:
        _step_check()
    if _pending is None:
        return
    with _fatal_lock:
        msg = _pending
        _pending = None
    if msg is None:
        return
    raise RuntimeError(
        f"{msg} (Sites in _ALWAYS_FATAL_SITES and reader failures stop the "
        f"rank under every mode; MODE=warn only softens the other "
        f"sites.)"
    )


def drain(
    *,
    raise_on_fatal: bool = False,
    raise_on_unsettled: bool = True,
    timeout: float | None = None,
) -> dict[str, list[int]]:
    """Synchronous final read of every counter, off the cadence.

    The cadence reader can miss a hit on the LAST step of a run: the forward
    completes, its result is consumed, no further execute_model polls
    check_pending, and the process exits before the reader's next tick. Call
    this at stage end / orderly shutdown, outside capture. It waits for the
    counters' producers, reports anything the reader has not, and returns the
    unbiased snapshot of the devices it read settled. With raise_on_fatal it
    raises for a hit at a site in _ALWAYS_FATAL_SITES or, under MODE=raise, at
    any site, and, unless raise_on_unsettled is False, for a device it could not
    read settled (reader lost, or producers not complete by the deadline: a
    busy or wedged device, which reports no hit of its own).

    It shares the reader's pinned mirrors, events and high-water marks, so stop
    the reader first (stop_reader(); the worker's shutdown does); every read is
    also serialised by one lock. ``timeout`` bounds the WHOLE drain: the wait
    for that lock (a reader stuck in a read; on expiry it logs and drains
    nothing), the wait for the counters' producers (event.query() polling on the
    reader's fence streams, or, when no gate was armed, a whole-device wait in a
    daemon helper the drain abandons at the deadline) and the copy-transport
    read. Producers not shown complete by then: FINAL DRAIN INCOMPLETE at ERROR,
    and the device's rows are never presented as final (mapped counts, host
    memory, are still reported as a LOWER BOUND). None = no deadline. Not bounded:
    a driver that blocks inside a host-side enqueue (event create/record, copy
    launch).
    """
    deadline = math.inf if timeout is None else time.monotonic() + timeout
    if not _io_lock.acquire(timeout=-1 if timeout is None else timeout):
        logger.error(
            "bounds-guard FINAL DRAIN skipped: the reader still holds the counters "
            "after %.1fs",
            timeout,
        )
        return {}
    try:
        if _asleep:
            logger.error(
                "bounds-guard FINAL DRAIN skipped: the worker is asleep (its "
                "counters were read when it went to sleep)"
            )
            return {}
        return _drain_locked(raise_on_fatal, raise_on_unsettled, deadline)
    finally:
        _io_lock.release()


def _drain_fence_streams(device: torch.device) -> list | None:
    """Streams whose completion shows every producer of ``device``'s counters
    complete, or None (no gate armed, or coverage broken)."""
    with _cv:  # under _io_lock (drain): the documented _io_lock -> _cv order
        pairs = _fence_events.get(device)
        if pairs and _coverage_live():
            return [s for s, _ in pairs]  # stop_reader() timed out: still live
        return _drain_streams.get(device)


def _drain_events(device: torch.device, streams) -> list:
    """Shutdown fence: FRESH events (never the reader's own -- a reader whose
    stop timed out may still query those), recorded on ``streams`` by the
    record loop of _record_fence, never by _record_fence itself (after the
    lifecycle reset that one breaks the gate and, under mapped, arms a stop).
    Test seam."""
    with torch.cuda.device(device):
        pairs = [(s, torch.cuda.Event()) for s in streams]
    return _record_pairs(device, pairs)


def _drain_sync(device: torch.device) -> None:
    """Whole-device wait for the unarmed fallback. Test seam."""
    torch.cuda.synchronize(device)


def _remaining(deadline: float) -> float:
    return math.inf if deadline == math.inf else max(0.0, deadline - time.monotonic())


def _producers_done(device: torch.device, deadline: float) -> bool:
    """True once every producer of ``device``'s counters has completed; False
    at ``deadline`` (math.inf = no deadline). Never blocks the caller past the
    deadline; raises on a CUDA error (sticky device state)."""
    streams = _drain_fence_streams(device)
    if streams:
        events = _drain_events(device, streams)
        while not all(e.query() for e in events):
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.0005)
        return True
    done, err = threading.Event(), []

    def run() -> None:
        try:
            _drain_sync(device)
        except BaseException as exc:  # noqa: BLE001 -- handed to the caller
            err.append(exc)
        finally:
            done.set()

    threading.Thread(target=run, name="bounds-guard-drain-sync", daemon=True).start()
    # never math.inf into Event.wait (it overflows); None waits unbounded
    if not done.wait(None if deadline == math.inf else _remaining(deadline)):
        return False
    if err:
        raise err[0]
    return True


def _snap_rows(device: torch.device, rows, out: dict) -> None:
    """Unbiased snapshot entries for one device's rows (shared with snapshot())."""
    for site, (n, row, val) in enumerate(rows):
        if n:
            out[f"{device}:{SITE_NAMES.get(site, site)}"] = [
                n,
                row - GUARD_BIAS if row else 0,
                val - GUARD_BIAS if val else 0,
            ]


def _fatal_rows(device: torch.device, rows, tag: str = "") -> list[str]:
    return [
        f"{SITE_NAMES.get(site, site)} on {device}: {n_masked}{tag}"
        for site, (n_masked, _r, _v) in enumerate(rows)
        if n_masked and (_MODE == "raise" or site in _ALWAYS_FATAL_SITES)
    ]


def _drain_locked(
    raise_on_fatal: bool, raise_on_unsettled: bool, deadline: float = math.inf
) -> dict[str, list[int]]:
    fatal: list[str] = []  # site hits, the unsettled lower bounds included
    unread: list[str] = []  # devices not read settled
    snap: dict[str, list[int]] = {}
    for device, buf in list(_buffers.items()):
        host = _host_views.get(device)
        rows = None
        try:
            settled = device.type != "cuda" or _producers_done(device, deadline)
            if host is not None:
                rows = host.tolist()  # mapped: host memory, readable settled or not
            elif settled:
                if not _nb_reader.registered(device):
                    _nb_reader.register(device, buf)
                # pinned mirror, side stream, event.query() polling: bounded
                rows = _nb_reader.read_settled(device, _remaining(deadline))
                settled = rows is not None
        except Exception:  # noqa: BLE001
            unread.append(f"reader lost {device}")  # first; log is best effort
            try:
                logger.exception(
                    "bounds-guard FINAL DRAIN INCOMPLETE: could not read %s", device
                )
            except Exception:  # noqa: BLE001
                pass
            continue
        if not settled:
            unread.append(f"drain incomplete on {device}")  # first; log best effort
            try:
                logger.error(
                    "bounds-guard FINAL DRAIN INCOMPLETE on %s: the counters' "
                    "producers were not shown complete before the deadline "
                    "(device busy or wedged); %s",
                    device,
                    "the mapped counts reported below are a LOWER BOUND, not final"
                    if rows is not None
                    else "not read (the copy transport needs the device)",
                )
            except Exception:  # noqa: BLE001
                pass
            if rows is not None:
                _report_rows(device, rows)  # monotone: any count seen is a real hit
                fatal.extend(_fatal_rows(device, rows, " (unsettled)"))
            continue  # never in the snapshot: not a final read
        _report_rows(device, rows)
        _snap_rows(device, rows, snap)  # no second sync, no pageable copy
        fatal.extend(_fatal_rows(device, rows))
    # Log readers key on this line, so an unread device with no hit gets it only where
    # it is fatal; its FINAL DRAIN INCOMPLETE line above is logged either way.
    # Real hits and unread devices are listed separately.
    if fatal or (unread and raise_on_unsettled):
        logger.error(
            "bounds-guard FINAL DRAIN: fatal-site hits %s; unread devices %s "
            "(snapshot %s)",
            fatal,
            unread,
            snap,
        )
        if raise_on_fatal:
            raise GuardFatalError(
                f"bounds-guard FINAL DRAIN: fatal-site hits {fatal}; "
                f"unread devices {unread}"
            )
    else:
        logger.info("bounds-guard FINAL DRAIN: no fatal-site hits (snapshot %s)", snap)
    return snap


def snapshot() -> dict[str, list[int]]:
    """Test/diagnostic helper: current counters, unbiased. Syncs -- not for the
    step path."""
    out: dict[str, list[int]] = {}
    for device, buf in _buffers.items():
        host = _host_views.get(device)
        if host is not None and device.type == "cuda":
            torch.cuda.synchronize(device)
        rows = host.tolist() if host is not None else buf.tolist()
        _snap_rows(device, rows, out)
    return out
