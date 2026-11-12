# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modified by the Merlin project; see NOTICE.
"""DMA copy backend for GPU<->CPU block transfers."""

from __future__ import annotations

import queue
import threading
import time

import torch

from vllm.logger import init_logger
from vllm.merlin.kv_host_tier_census_events import (
    CensusEventPool, census_commit, census_release, census_take)
from vllm.platforms import current_platform
from vllm.v1.simple_kv_offload.cuda_mem_ops import (
    CU_MEMCPY_SRC_ACCESS_ORDER_ANY,
    CU_MEMCPY_SRC_ACCESS_ORDER_STREAM,
    BatchMemcpyParams,
    build_params,
    copy_blocks,
)

logger = init_logger(__name__)


def teardown_deadline_s() -> float:
    """One deadline for the whole copy-backend teardown (lifecycle lock, both
    thread joins, both stream drains): inside the executor's worker shutdown
    grace (VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS, 5 s; 2 s after a failure)."""
    import vllm.envs as envs

    grace = float(envs.VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS)
    return max(0.5, min(2.0, grace - 1.0))


def _left(deadline: float) -> float:
    # A small floor so a direction that is already done is not called stuck
    # only because an earlier wait used up the budget.
    return max(0.05, deadline - time.monotonic())


def _bounded_call(fn, deadline: float, name: str) -> tuple[bool, BaseException | None]:
    """Run ``fn`` (a stream drain) on a helper thread; (finished, its error)."""
    errors: list[BaseException] = []

    def run() -> None:
        try:
            fn()
        except BaseException as exc:  # noqa: BLE001 - returned to the caller
            errors.append(exc)

    thread = threading.Thread(target=run, name=name, daemon=True)
    thread.start()
    thread.join(timeout=_left(deadline))
    if thread.is_alive():
        return False, None
    return True, errors[0] if errors else None


def blocks_per_call(bytes_per_block: int, max_call_bytes: int) -> int:
    """Blocks one batch-copy call may carry so that it moves at most
    ``max_call_bytes`` (always at least one block: a block is the unit of
    failure attribution, cancellation and fault injection never splits)."""
    return max(1, max_call_bytes // max(1, bytes_per_block))


_HOST_WAIT_POLL_S = 0.0002


def _host_wait(event, discard: threading.Event | None) -> bool:
    """Poll ``event`` until it completes, holding no driver lock between
    polls (``event.query()`` never waits on the GPU). False if ``discard``
    is set first (shutdown: the transfer is dropped, never issued)."""
    # query() holds the GIL in torch.Event and torch.cuda.Event alike
    # (torch 2.13); cudaEventQuery does not block, but is a known residual.
    while not event.query():
        if discard is not None and discard.is_set():
            return False
        time.sleep(_HOST_WAIT_POLL_S)
    return True


class DmaCopyBackend:

    """cuMemcpyBatchAsync copy backend (one background thread per direction).

    Loads and stores run on separate threads with separate FIFO queues: a copy
    to or from pageable host memory (the host-RAM KV tier arena) returns only once
    the driver has staged it, so a long store must not hold up the loads queued
    behind it. Each direction keeps its own stream and event list, so the
    per-direction FIFO order the worker's polling relies on is unchanged.

    ``chunk_blocks`` > 0 issues a transfer as consecutive batch calls of at
    most that many blocks on the same stream (same bytes, same order, one
    completion event after the last). The driver stages pageable copies one
    call at a time per context, so this bounds how long any other pageable
    copy in the process -- the other direction's, or the engine's own -- can
    wait behind a long one to a single chunk. 0 (pinned arenas) = one call.
    """

    def __init__(self) -> None:

        # Set by the worker whenever the host tier is on (the copy-cost
        # metric); None for other connectors.
        self.store_census: dict | None = None
        self._census_events: CensusEventPool | None = None
        self._store_params: BatchMemcpyParams | None = None
        self._load_params: BatchMemcpyParams | None = None
        self._load_stream: torch.cuda.Stream | None = None
        self._store_stream: torch.cuda.Stream | None = None

        self._queues: dict[bool, queue.SimpleQueue] = {}
        self._threads: list[threading.Thread] = []
        self._shutdown: bool = False

        self._chunk_blocks: int = 0
        # First exception raised on a copy thread; re-raised on the engine
        # thread (launch_copy / check_error / flush) instead of leaving the
        # transfer's event unpublished and its request pending forever.
        self._error: BaseException | None = None
        # flush() and shutdown() never overlap: a flush that saw the backend
        # live must not queue barriers behind shutdown's sentinels, and a
        # shutdown in progress must not let a flush return early.
        self._lifecycle_lock = threading.Lock()
        # Set when shutdown misses its deadline: the copy threads issue nothing
        # more (queued copies are dropped, their events never published) and a
        # flush still waiting raises instead of reporting completion.
        self._discard = threading.Event()
        # The first shutdown failure (missed deadline, stuck thread, failed
        # stream drain). A later flush() or shutdown() raises it again rather
        # than reporting a completion that was never established.
        self._teardown_error: BaseException | None = None

    def init(
        self,
        gpu_caches: dict[str, torch.Tensor],
        cpu_caches: dict[str, torch.Tensor],
        device: torch.device,
        load_stream: torch.cuda.Stream,
        store_stream: torch.cuda.Stream,

        chunk_blocks: int = 0,
    ) -> None:

        self._chunk_blocks = max(0, chunk_blocks)
        self._load_stream = load_stream
        self._store_stream = store_stream
        if self.store_census is not None:
            self._census_events = CensusEventPool(store_stream)

        # Stores read the live KV cache -> STREAM (paired with the compute-done

        # wait in get_finished); loads read stable host memory -> ANY.
        self._store_params = build_params(
            gpu_caches,
            cpu_caches,
            store_stream,
            src_access_order=CU_MEMCPY_SRC_ACCESS_ORDER_STREAM,
        )
        self._load_params = build_params(
            cpu_caches,
            gpu_caches,
            load_stream,
            src_access_order=CU_MEMCPY_SRC_ACCESS_ORDER_ANY,
        )


        for is_store, stream in ((False, load_stream), (True, store_stream)):
            q: queue.SimpleQueue = queue.SimpleQueue()
            self._queues[is_store] = q
            thread = threading.Thread(
                target=self._copy_loop,
                args=(q, device, stream, self._chunk_blocks),
                name="simple-kv-offload-" + ("store" if is_store else "load"),
                daemon=True,
            )
            self._threads.append(thread)
            thread.start()

    def launch_copy(
        self,
        src_blocks: list[int],
        dst_blocks: list[int],
        is_store: bool,
        event_idx: int,
        events_list: list[tuple[int, torch.Event]],
        wait_event: torch.Event | None = None,
        # With a set, a copy failure of this
        # transfer adds event_idx to it (the event is still published) instead
        # of failing the copy thread: the worker reports the restore failed.
        failed_events: set[int] | None = None,
    ) -> None:

        if self._shutdown:
            raise RuntimeError("KV copy backend is shut down")
        self.check_error()
        params = self._store_params if is_store else self._load_params

        q = self._queues.get(is_store)
        assert params is not None and q is not None
        q.put(
            (
                src_blocks,
                dst_blocks,
                params,

                event_idx,
                events_list,
                wait_event,

                failed_events,
            )
        )


    def check_error(self) -> None:
        if self._error is not None:
            raise RuntimeError("KV copy thread failed") from self._error

    def flush(self) -> None:
        """Wait until every transfer submitted before this call has been issued
        and has completed on its stream (a barrier through both queues), so a
        caller syncing the event lists afterwards also covers transfers that
        were still queued or mid-chunk when it was called. After shutdown()
        no thread could release a new barrier: return, or raise when that
        shutdown failed to join or drain, since completion is not established."""
        with self._lifecycle_lock:
            self._flush()

    def _flush(self) -> None:
        self.check_error()
        if self._shutdown:
            self._check_teardown()
            self._check_discard()
            return

        barriers: list[threading.Event] = []
        for q in self._queues.values():
            done = threading.Event()
            q.put(done)
            barriers.append(done)
        for done in barriers:
            while not done.wait(0.05):
                self.check_error()
                self._check_discard()
        self.check_error()
        self._check_discard()


    def _check_teardown(self) -> None:
        if self._teardown_error is not None:
            raise RuntimeError(
                "KV copy backend: an earlier shutdown did not complete; DMA "
                "completion is not established"
            ) from self._teardown_error


    def _check_discard(self) -> None:
        if self._discard.is_set():
            raise RuntimeError(
                "KV copy backend: shutdown discarded copies still in flight; "
                "this flush cannot establish their completion"
            )


    def shutdown(self, timeout_s: float | None = None) -> None:
        """Issue the queued copies, drain both directions, report errors.

        Every wait (the lifecycle lock a flush may hold, both thread joins,
        both stream drains) shares one deadline inside the executor's shutdown
        grace (``teardown_deadline_s``). When it passes, the copies still
        queued are discarded, never issued (their events are never published,
        so nothing is served from them), the stuck direction's stream drain is
        skipped and a TimeoutError is raised: that direction's DMA completion
        is not established and the worker keeps its arenas alive.
        """
        budget = teardown_deadline_s() if timeout_s is None else timeout_s
        deadline = time.monotonic() + budget
        if not self._lifecycle_lock.acquire(timeout=_left(deadline)):
            # A flush is still waiting on a copy: stop issuing, release it.
            self._discard.set()
            self._shutdown = True
            for q in self._queues.values():
                q.put(None)
            logger.error(
                "SimpleCPUOffload copy backend: a flush still waited on a copy "
                "at the %.1f s shutdown deadline; queued copies are discarded "
                "and DMA completion is not established",
                budget,
            )
            error = TimeoutError("KV copy backend flush did not finish by shutdown")
            if self._teardown_error is None:
                self._teardown_error = error
            raise error
        try:
            if self._shutdown:
                self.check_error()
                self._check_teardown()
                return
            self._shutdown = True
            for q in self._queues.values():
                q.put(None)
            error: BaseException | None = None
            stuck: set[str] = set()
            for thread in self._threads:
                thread.join(timeout=_left(deadline))
                if thread.is_alive():
                    stuck.add(thread.name)
                    logger.error(
                        "SimpleCPUOffload copy thread %s did not stop within "
                        "the %.1f s shutdown deadline; its queued copies are "
                        "discarded and its stream drain is skipped",
                        thread.name,
                        budget,
                    )
                    if error is None:
                        error = TimeoutError(f"KV copy thread {thread.name} stuck")
            if stuck:
                self._discard.set()
            # The arenas must outlive every issued copy, not only the threads.
            for direction, stream in (
                ("load", self._load_stream),
                ("store", self._store_stream),
            ):
                if stream is None or f"simple-kv-offload-{direction}" in stuck:
                    continue
                done, exc = _bounded_call(
                    stream.synchronize, deadline, f"simple-kv-offload-drain-{direction}"
                )
                if not done:
                    logger.error(
                        "SimpleCPUOffload %s stream did not drain within the "
                        "%.1f s shutdown deadline; DMA completion is not "
                        "established",
                        direction,
                        budget,
                    )
                    exc = TimeoutError(f"KV copy {direction} stream drain timed out")
                elif exc is not None:
                    logger.error(
                        "SimpleCPUOffload %s stream drain failed",
                        direction,
                        exc_info=exc,
                    )
                if exc is not None and error is None:
                    error = exc
            if error is not None:
                self._teardown_error = error
                raise error
        finally:
            self._lifecycle_lock.release()
        self.check_error()


    def _copy_loop(

        self,
        q: queue.SimpleQueue,
        device: torch.device,

        stream: torch.cuda.Stream,
        chunk_blocks: int = 0,
    ) -> None:
        # A copy-thread failure is kept for check_error.
        try:
            # Timings only on census stores.
            census_kw = {}
            if self.store_census is not None and stream is self._store_stream:
                census_kw["timings"] = self.store_census
                census_kw["census_events"] = self._census_events
            self._run_copy_loop(
                q, device, stream, chunk_blocks, self._discard,

                **(
                    {"fault_kind": "store"}
                    if self._store_stream is not None
                    and stream is self._store_stream
                    else {}
                ),
                **census_kw,
            )
        except BaseException as exc:  # noqa: BLE001 - surfaced by check_error
            logger.exception("SimpleCPUOffload copy thread failed")
            # Keep the first failure (root cause).
            if self._error is None:
                self._error = exc
            # Release flush() waiters; check_error raises for them.
            while True:
                try:
                    item = q.get_nowait()
                except queue.Empty:
                    break
                if isinstance(item, threading.Event):
                    item.set()

    @staticmethod
    def _run_copy_loop(
        q: queue.SimpleQueue,
        device: torch.device,
        stream: torch.cuda.Stream,
        chunk_blocks: int = 0,

        discard: threading.Event | None = None,

        timings: dict | None = None,

        fault_kind: str = "load",
        census_events: CensusEventPool | None = None,
    ) -> None:

        from vllm.merlin.kv_host_tier_faults import (
            RestoreFailureInjector,
            StoreFailureInjector,
        )

        injector: RestoreFailureInjector | StoreFailureInjector | None = None
        current_platform.set_device(device)
        while True:
            item = q.get()
            if item is None:
                return

            if isinstance(item, threading.Event):    # flush() barrier
                # After a missed shutdown deadline the flush raises instead.
                if discard is None or not discard.is_set():
                    stream.synchronize()
                item.set()
                continue
            # Discarded: never issued or published.
            if discard is not None and discard.is_set():
                continue
            (
                src_blocks,
                dst_blocks,
                params,

                event_idx,
                events_list,
                wait_event,

                failed_events,
            ) = item
            # Wait for the compute prerequisite on
            # the host, polling, before any driver call. The tier arena is
            # pageable: the driver services a pageable copy holding the CUDA
            # context lock until the copy's stream has run everything queued
            # before it, so a GPU-side stream.wait_event(compute_done) would
            # make this thread hold that lock while compute waits for work
            # (the PLE pull transport, the main thread's next launch) that
            # needs it: a whole-worker deadlock until the executor timeout.
            if wait_event is not None:
                # Discarded at shutdown: never issued.
                if not _host_wait(wait_event, discard):
                    continue
                stream.wait_event(wait_event)  # complete: a no-op ordering edge

            step = chunk_blocks if chunk_blocks > 0 else max(1, len(src_blocks))
            # The timing start is recorded after
            # the compute wait, with no synchronize.
            start = pair = None
            submitted = time.monotonic()
            if timings is not None:

                # The timing pair comes from the pool; no CUDA event is
                # created here.
                pair = census_take(census_events, timings)
                start = pair[0]
                start.record(stream)
            # failed_events is set only by the host
            # tier, so other connectors never arm a hook. A restore never consumes
            # the store quota, nor a store the restore quota.
            if failed_events is not None and injector is None:
                injector = (
                    StoreFailureInjector()
                    if fault_kind == "store"
                    else RestoreFailureInjector()
                )
            # An armed fault fires before the final
            # call, so at least one earlier call has been submitted whenever
            # the transfer has two blocks or more (partial writes are what the
            # failed-event fence is for), however large a call may be. Only a
            # transfer the quota can still select is split: with one load
            # event per request, the later transfers keep their calls.
            if injector is not None and injector._remaining > 0:
                step = min(step, max(1, len(src_blocks) - 1))
            abandoned = False
            try:
                for i in range(0, len(src_blocks), step):
                    # A missed shutdown deadline
                    # stops a transfer between chunks; its event is never
                    # published, so its partial bytes are never served.
                    if discard is not None and discard.is_set():
                        abandoned = True
                        break
                    # Inside the real copy error
                    # path, after earlier chunks but before the final copy.
                    if injector is not None:
                        injector.before_copy(
                            event_idx,
                            first_chunk=i == 0,
                            last_chunk=i + step >= len(src_blocks),
                        )
                    copy_blocks(
                        src_blocks[i : i + step], dst_blocks[i : i + step], params
                    )
            except Exception:
                if failed_events is None:
                    raise
                logger.exception(
                    "SimpleCPUOffload: restore copy for event %d failed; its "
                    "requests will recompute",
                    event_idx,
                )
                # Before the event is published: whoever sees it complete
                # also sees the failure. The event follows every chunk that
                # was enqueued, so the destination blocks are released only
                # once nothing writes them any more.
                failed_events.add(event_idx)
            # Discarded at shutdown mid-transfer:
            # its completion is never published, so nothing is served from it.
            if abandoned or (discard is not None and discard.is_set()):
                census_release(census_events, pair)
                continue
            # Retain the existing completion event.
            event = pair[1] if pair is not None else torch.cuda.Event()
            event.record(stream)

            if timings is not None:
                timings[event_idx] = (start, event, len(src_blocks),
                                      1000 * (time.monotonic() - submitted))
                census_commit(census_events, event_idx, pair)
            events_list.append((event_idx, event))
