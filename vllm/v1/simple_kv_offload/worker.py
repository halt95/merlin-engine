# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modified by the Merlin project; see NOTICE.
"""Worker-side handler for SimpleCPUOffloadConnector."""

from typing import TYPE_CHECKING
import time

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.merlin import kv_commit_acks, kv_host_tier
from vllm.utils.torch_utils import PIN_MEMORY
from vllm.v1.simple_kv_offload.copy_backend import (
    DmaCopyBackend,
    blocks_per_call,
)
from vllm.v1.simple_kv_offload.cuda_mem_ops import lock_host_memory, pin_tensor
from vllm.v1.simple_kv_offload.disk_backend import DiskBackend
from vllm.v1.simple_kv_offload.metadata import (
    SimpleCPUOffloadMetadata,
    SimpleCPUOffloadWorkerMetadata,
)
from vllm.v1.simple_kv_offload.sizing import sync_num_offload_blocks_across_workers

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import KVCacheConfig

logger = init_logger(__name__)

# Bytes one batch-copy call of the pageable arena
# may move. The driver stages pageable copies one call at a time per context,
# so a call bounds how long any other pageable copy waits behind it (at
# ~8 GB/s, 256 MiB is ~32 ms). Several blocks share one call up to this bound,
# so a small-block layout is not issued one NumPy descriptor build and driver
# call per block; a block larger than the bound is still one call per block.
# Internal: not a user knob.
HOST_TIER_COPY_CALL_BYTES = 256 << 20
# Failed teardown cannot prove DMA has stopped.
# Keep its arenas alive until process exit rather than free raw copy addresses.
_FAILED_SHUTDOWN_WORKERS: list[object] = []


class SimpleCPUOffloadWorker:
    """Worker-side handler for CPU offloading transfers."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: "KVCacheConfig | None",
        cpu_capacity_bytes: int,
        kv_offload_backend: str = "cpu",
        disk_path: str | None = None,
        disk_capacity_bytes: int = 0,
        disk_buffer_slots: int = 2,
        use_page_cache: bool = False,
        num_cpu_blocks: int | None = None,
    ):
        self.vllm_config = vllm_config
        self.kv_cache_config = kv_cache_config
        self.cpu_capacity_bytes = cpu_capacity_bytes
        self._num_cpu_blocks_override = num_cpu_blocks
        self.disk_path = disk_path
        self.disk_capacity_bytes = disk_capacity_bytes
        self.disk_buffer_slots = disk_buffer_slots
        self.use_page_cache = use_page_cache
        self.disk_mode = kv_offload_backend == "disk"
        # The host tier's arena is not registered
        # with CUDA (see _init_cpu_mode) and its loads wait for prior compute.
        self.host_tier = vllm_config is not None and kv_host_tier.is_active(
            vllm_config
        )

        # Commit acks for the writer events the runner
        # retires.
        self.commit_acks = kv_commit_acks.WorkerCommitAcks() if self.host_tier else None

        self.gpu_kv_caches: dict[str, torch.Tensor] | None = None
        self.cpu_kv_caches: dict[str, torch.Tensor] | None = None
        self.device: torch.device | None = None
        self.num_cpu_blocks: int = 0

        # CUDA streams for the async transfers
        self.load_stream: torch.cuda.Stream | None = None
        self.store_stream: torch.cuda.Stream | None = None

        self._backend: DmaCopyBackend | DiskBackend | None = None

        # Ordered (event_idx, Event). Events pre-allocated on main thread.
        self._load_events: list[tuple[int, torch.Event]] = []
        self._store_events: list[tuple[int, torch.Event]] = []
        # High-water marks: highest event_idx completed per stream.
        # When the event list is empty, the hwm covers all prior events.
        self._load_hwm: int = -1
        self._store_hwm: int = -1

        # Metadata for the current step
        self._connector_metadata: SimpleCPUOffloadMetadata | None = None

        # A compute-done event is made per store (in the store path), not kept here.
        # Pending event index sets, populated in bind_connector_metadata
        self._pending_load_event_indices: set[int] = set()
        self._pending_store_event_indices: set[int] = set()
        # Completed store events to report via build_connector_worker_meta
        self._completed_store_events: dict[int, int] = {}
        self._census: dict[str, float] = {}
        self._census_dirty = False
        self._census_rank = "0"
        # Load events whose copy failed (filled by
        # the load copy thread) and the requests to report as failed receives.
        self._failed_load_events: set[int] = set()
        self._failed_recving: set[str] = set()
        # Per-rank transaction outcomes,
        # reported once via build_connector_worker_meta:
        # (cache epoch at issue, event) -> success.
        self._failed_store_events: set[int] = set()
        self._store_event_epochs: dict[int, int] = {}
        self._store_outcomes: dict[tuple[int, int], bool] = {}
        self._load_outcomes: dict[tuple[int, int], bool] = {}
        # Load event -> request ids; kept across
        # clear/rebind on idle drain steps.
        self._load_event_to_reqs: dict[int, tuple[str, ...]] = {}
        self._load_event_epochs: dict[int, int] = {}
        self._cache_epoch = 0

    def register_kv_caches(
        self,
        kv_caches: dict[str, torch.Tensor],
    ) -> None:
        """Register GPU KV caches and allocate pinned CPU tensors.
        The worker will infer the underlying raw storage from the kv_caches.

        Args:
            kv_caches: Per-layer GPU KV caches. Values are either a single
                tensor (attention layers) or a list of tensors (Mamba layers
                in hybrid models). All values are included for offloading
                by resolving to their underlying raw storage.

        """
        if not kv_caches:
            logger.warning("No KV caches to offload.")
            return

        self.device = next(iter(kv_caches.values())).device

        assert self.kv_cache_config is not None
        # Before the arena is allocated.
        if self.host_tier:
            kv_host_tier.refuse_prefix_replay(self.kv_cache_config)
        num_blocks = self.kv_cache_config.num_blocks
        assert self.kv_cache_config.kv_cache_tensors

        # The DMA backend copies blocks as base + block_id * stride(0), so every
        # region is a [num_blocks, block_bytes] view. Block bytes come from each
        # tensor's placement -- layers in one allocation can differ in page size.
        unique_gpu_caches: dict[str, torch.Tensor] = {}
        seen: dict[tuple[torch.device, int], int] = {}
        for kv_cache_tensor in self.kv_cache_config.kv_cache_tensors:
            if kv_cache_tensor.slots_per_block > 1:
                # Side-cache layout: a slot-packed ring view (block stride
                # = one slot, ids pool_block * slots + slot). Its group is not
                # prefix-cacheable, so the scheduler never stores or loads its
                # ids, and its slots lie inside pool blocks the ring manager
                # owns: no other group's id addresses them. Copies are whole
                # pool blocks, so this view must not become a region (with a
                # one-slot row it would mis-size every copy and the CPU pool).
                continue
            name = next((n for n in kv_cache_tensor.layers if n in kv_caches), None)
            if name is None:
                continue
            num_layers = len(kv_cache_tensor.layers)
            block_bytes = kv_cache_tensor.block_stride
            blocks_span = num_blocks * block_bytes
            layer_outermost = kv_cache_tensor.layer_stride >= blocks_span
            if layer_outermost:
                # One region per layer, or per KV head group under LHBNC.
                start = kv_cache_tensor.offset
                span = num_layers * kv_cache_tensor.layer_stride
                layer_names = kv_cache_tensor.layers
            else:
                start = 0
                span = blocks_span
                layer_names = kv_cache_tensor.layers[:1]

            cache = kv_caches[name]
            raw = torch.empty(0, dtype=torch.int8, device=cache.device).set_(
                cache.untyped_storage()
            )
            assert raw.numel() >= start + span, (
                f"KV cache {name!r} storage has {raw.numel()} bytes, smaller than "
                f"the {span} bytes it places at offset {start}"
            )
            regions = raw[start : start + span].view(-1, num_blocks, block_bytes)
            groups_per_layer, remainder = divmod(len(regions), len(layer_names))
            assert remainder == 0, (
                f"KV cache {name!r} places {len(layer_names)} layers in "
                f"{len(regions)} block-strided regions"
            )
            for idx, region in enumerate(regions):
                key = (region.device, region.data_ptr())
                if key in seen:
                    # An aliased region (groups
                    # sharing one placement) must have the same block bytes, or
                    # the first one's copies would mis-size the others.
                    assert seen[key] == block_bytes, (
                        f"KV cache {name!r} aliases a region of "
                        f"{seen[key]} B blocks with {block_bytes} B blocks"
                    )
                    continue
                seen[key] = block_bytes
                layer_name = layer_names[idx // groups_per_layer]
                key_name = (
                    layer_name
                    if groups_per_layer == 1
                    else f"{layer_name}.{idx % groups_per_layer}"
                )
                unique_gpu_caches[key_name] = region

        # Compute per-tensor bytes_per_block. Tensors may have different
        # page_size_bytes (e.g., UniformTypeKVCacheSpecs with varying head_size).
        per_tensor_bpb = [
            t.stride(0) * t.element_size() for t in unique_gpu_caches.values()
        ]
        total_bytes_per_block = sum(per_tensor_bpb)

        # Host-tier block count resolved once for every rank
        if self.host_tier:
            # Ring views and narrower PP stages need
            # fewer bytes per block, but cannot use more IDs than the scheduler.
            # The engine resolved one count for every rank (widest PP stage), so
            # the tier skips the cross-rank MIN sync and the explicit override.
            self.num_cpu_blocks = kv_host_tier.worker_block_count(
                self.kv_cache_config,
                max(1, self.cpu_capacity_bytes // total_bytes_per_block),
            )
        else:
            self.num_cpu_blocks = self._resolve_num_cpu_blocks(
                self.disk_capacity_bytes if self.disk_mode else self.cpu_capacity_bytes,
                total_bytes_per_block,
            )

        # Use lowest priority so KV cache I/O yields to compute streams.
        low_pri, _ = torch.cuda.Stream.priority_range()
        self.load_stream = torch.cuda.Stream(priority=low_pri)
        self.store_stream = torch.cuda.Stream(priority=low_pri)

        self.gpu_kv_caches = unique_gpu_caches

        # Refuse an arena the host cannot hold:
        # check, allocate and lock it under one host-wide reservation so
        # concurrent ranks cannot all pass the check and then over-commit.
        if self.host_tier and not self.disk_mode:
            with kv_host_tier.arena_reservation(
                self.num_cpu_blocks * total_bytes_per_block
            ):
                self._init_cpu_mode(
                    unique_gpu_caches, total_bytes_per_block, self.device
                )
            return
        if self.disk_mode:
            self._init_disk_mode(unique_gpu_caches, total_bytes_per_block, self.device)
        else:
            self._init_cpu_mode(unique_gpu_caches, total_bytes_per_block, self.device)

    def _resolve_num_cpu_blocks(self, capacity_bytes: int, bytes_per_block: int) -> int:
        local_capacity = capacity_bytes // bytes_per_block
        requested = self._num_cpu_blocks_override
        if requested is None:
            return sync_num_offload_blocks_across_workers(max(1, local_capacity))
        if type(requested) is not int or not 0 < requested <= local_capacity:
            raise ValueError(
                f"num_cpu_blocks={requested!r} must be between 1 and "
                f"local offload capacity {local_capacity}"
            )
        return requested

    def _init_disk_mode(
        self,
        unique_gpu_caches: dict[str, torch.Tensor],
        total_bytes_per_block: int,
        device: torch.device,
    ) -> None:
        logger.info(
            "SimpleCPUOffloadWorker [DISK]: %d tensors, %d disk slots (%.2f GB)",
            len(unique_gpu_caches),
            self.num_cpu_blocks,
            (self.num_cpu_blocks * total_bytes_per_block) / (1024**3),
        )

        assert self.disk_path is not None
        rank_path = f"{self.disk_path}.rank_{device.index or 0}"
        self._backend = DiskBackend()
        self._backend.init(
            unique_gpu_caches,
            device,
            self.load_stream,
            self.store_stream,
            rank_path,
            self.num_cpu_blocks,
            total_bytes_per_block,
            self.disk_buffer_slots,
            self.use_page_cache,
        )

    def _init_cpu_mode(
        self,
        unique_gpu_caches: dict[str, torch.Tensor],
        total_bytes_per_block: int,
        device: torch.device,
    ) -> None:
        logger.info(
            "SimpleCPUOffloadWorker [CPU]: %d tensors, %d CPU blocks (%.2f GB)",
            len(unique_gpu_caches),
            self.num_cpu_blocks,
            (self.num_cpu_blocks * total_bytes_per_block) / (1024**3),
        )

        # The tier keeps its arena pageable. cudaHostRegister maps every
        # 4 KiB page of an anonymous host range into the GPU's virtual address
        # space (the driver cannot use larger GPU pages for user memory it did
        # not allocate), so the GPU's page tables grow by 8 B per 4 KiB page:
        # 2 MiB of device memory per GiB of arena, outside the torch allocator
        # (24 MiB per rank at 12 GiB). A pageable arena maps nothing: the
        # driver stages the same batched copies through its own pinned
        # buffers, byte for byte. mlock keeps the arena resident as
        # registration did.
        pin_memory = PIN_MEMORY and not self.host_tier
        if not pin_memory and not self.host_tier:
            logger.warning(
                "Pinned memory not available. CPU offload performance may be degraded."
            )

        self.cpu_kv_caches = {}
        for name, gpu_tensor in unique_gpu_caches.items():
            cpu_shape = (self.num_cpu_blocks,) + gpu_tensor.shape[1:]
            # Allocate non-pinned first, then pin via cudaHostRegister to
            # bypass PyTorch's CUDACachingHostAllocator which rounds up to
            # the next power of 2 (e.g. 100 GB -> 128 GB).
            # vllm#64345: CPU
            # blocks are loaded only after a completed store, so the arena
            # needs no zero fill (pin / mlock still fault in every page).
            tensor = torch.empty(cpu_shape, dtype=gpu_tensor.dtype, device="cpu")
            if pin_memory:
                pin_tensor(tensor)
            # Host RAM only: an arena that cannot
            # be locked could be swapped to disk, so the tier refuses to start.
            elif self.host_tier and not lock_host_memory(tensor):
                self.cpu_kv_caches = None
                raise kv_host_tier.mlock_refused(device, tensor.nbytes)
            self.cpu_kv_caches[name] = tensor
        if self.host_tier:
            kv_host_tier.log_arena(
                device, self.num_cpu_blocks, total_bytes_per_block, True
            )

        self._backend = DmaCopyBackend()
        # Store-copy timings feed the
        # tier's copy-cost metric, so they are collected whenever the tier is
        # on; MERLIN_KV_HOST_TIER_CENSUS gates only the debug census.
        if self.host_tier:
            self._census_rank = str(
                torch.distributed.get_rank() if torch.distributed.is_initialized()
                else getattr(self.vllm_config.parallel_config, "rank", 0)
            )
            self._backend.store_census = {}
            logger.info(
                "SimpleCPU tier census worker: rank=%s strides=%s dtypes=%s",
                self._census_rank,
                {name: cache.stride(0) * cache.element_size()
                 for name, cache in self.cpu_kv_caches.items()},
                {name: str(cache.dtype) for name, cache in self.cpu_kv_caches.items()},
            )
        self._backend.init(
            unique_gpu_caches,
            self.cpu_kv_caches,
            device,
            self.load_stream,
            self.store_stream,
            # Pageable arena: batch calls of at
            # most HOST_TIER_COPY_CALL_BYTES (see DmaCopyBackend).
            chunk_blocks=(
                blocks_per_call(total_bytes_per_block, HOST_TIER_COPY_CALL_BYTES)
                if self.host_tier
                else 0
            ),
        )

    def _init_event_leases(self) -> None:
        """Per-event load ownership, also for workers built without
        ``__init__`` (unit-test doubles). The ``getattr(self, ...)`` guards in
        this class exist for the same doubles; a real worker always has the
        attributes."""
        if not hasattr(self, "_cache_epoch"):
            self._load_event_to_reqs = {}
            self._load_event_epochs = {}
            self._cache_epoch = 0
            self._failed_store_events = set()
            self._store_event_epochs = {}
            self._store_outcomes = {}
            self._load_outcomes = {}

    def bind_connector_metadata(self, metadata: SimpleCPUOffloadMetadata) -> None:
        self._connector_metadata = metadata
        # Event ownership outlives step metadata.
        self._init_event_leases()
        self._cache_epoch = max(self._cache_epoch, metadata.cache_epoch)
        # One event per request with the tier.
        for event, _, _ in metadata.load_transfer_list():
            self._load_event_epochs[event] = metadata.cache_epoch
        for event_idx, req_ids in metadata.load_event_to_reqs.items():
            if event_idx > self._load_hwm:
                self._load_event_to_reqs.setdefault(event_idx, tuple(req_ids))
        # No DMA was issued for rejected loads.
        if getattr(self, "host_tier", False):
            self._failed_recving.update(metadata.failed_recving)
        for event, _, _ in metadata.load_transfer_list():
            self._pending_load_event_indices.add(event)
        if metadata.store_event >= 0:
            self._pending_store_event_indices.add(metadata.store_event)
            if getattr(self, "host_tier", False):
                self._store_event_epochs[metadata.store_event] = metadata.cache_epoch

    def clear_connector_metadata(self) -> None:
        self._connector_metadata = None

    def start_load_kv(self) -> None:
        # Invoked after the forward launch on steps without sync loads
        # (SchedulerOutput.has_sync_kv_loads), so the CPU-side block copy
        # op overhead (~5ms) stays hidden behind GPU compute. Stores are
        # issued in wait_for_save().
        metadata = self._connector_metadata
        if metadata is not None and metadata.load_cpu_blocks:
            backend = self._backend
            assert backend is not None
            # Order the restore after the compute
            # already queued (a reused block's last writer), as stores are.
            compute_done = None
            if self.host_tier:
                compute_done = torch.cuda.Event()  # wait() drops the GIL
                compute_done.record(torch.cuda.current_stream())
            # One copy (one event) per transfer.
            # An empty transfer still records its completion event.
            for event, gpu_blocks, cpu_blocks in metadata.load_transfer_list():
                backend.launch_copy(
                    cpu_blocks,
                    gpu_blocks,
                    is_store=False,
                    event_idx=event,
                    events_list=self._load_events,
                    wait_event=compute_done,
                    **(
                        {"failed_events": self._failed_load_events}
                        if self.host_tier
                        else {}
                    ),
                )

    def wait_for_save(self) -> None:
        """Submit async stores.

        Stores (GPU->CPU) read the live KV cache, which the compute stream may
        still be writing under v1 overlapped execution, so they are ordered
        after a compute-done event recorded on the current stream. See
        #45704 for the bug and #39306 for the srcAccessOrder rationale.
        """
        # Also retires empty/drain forwards.
        tracker = getattr(self, "commit_acks", None)
        if tracker is not None:
            tracker.finish_forward()
        metadata = self._connector_metadata
        if metadata is not None and metadata.store_gpu_blocks:
            backend = self._backend
            assert backend is not None
            # One compute-done event per store: the store thread may issue its
            # wait after the engine has moved on, and a reused event would by
            # then carry a later recording (still safe on one compute stream,
            # but not the dependency this store needs).
            compute_done = torch.cuda.Event()  # wait() drops the GIL
            compute_done.record(torch.cuda.current_stream())
            backend.launch_copy(
                metadata.store_gpu_blocks,
                metadata.store_cpu_blocks,
                is_store=True,
                event_idx=metadata.store_event,
                events_list=self._store_events,
                wait_event=compute_done,
                # A failed store copy is reported
                # per rank (success=False) instead of failing the engine; its
                # event still follows every enqueued chunk (quiescent).
                **(
                    {"failed_events": self._failed_store_events}
                    if getattr(self, "host_tier", False)
                    else {}
                ),
            )

    def shutdown(self) -> None:
        """Ask the copy backend to drain queued copies before teardown.
        Copy failures and join timeouts are logged so engine teardown goes on;
        a timeout means that direction's DMA completion is not established."""
        backend = self._backend
        if backend is None:
            return
        try:
            backend.shutdown()
        except Exception:  # noqa: BLE001
            if self not in _FAILED_SHUTDOWN_WORKERS:
                _FAILED_SHUTDOWN_WORKERS.append(self)
            logger.exception("SimpleCPUOffload copy backend failed before shutdown")

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        """Report completed transfer events to the scheduler.

        Loads are submitted in start_load_kv() and stores in wait_for_save().

        Returns:
            tuple of (finished_sending, finished_recving).
            - finished_sending: always None (stores use worker metadata).
            - finished_recving: req_ids whose loads have completed.

        """
        if isinstance(self._backend, DmaCopyBackend):
            self._backend.check_error()
        # Event leases survive metadata being cleared.
        finished_recving: set[str] = set()
        # Failed receives must also finish the wait.
        if getattr(self, "host_tier", False):
            finished_recving.update(self._failed_recving)

        if self._pending_load_event_indices:
            load_wm = self._poll_stream_events(is_store=False)
            for j in [j for j in self._pending_load_event_indices if j <= load_wm]:
                self._pending_load_event_indices.discard(j)
                # Retain cancelled transfers' acks.
                self._init_event_leases()
                req_ids = self._load_event_to_reqs.pop(j, ())
                event_epoch = self._load_event_epochs.pop(j, self._cache_epoch)
                if req_ids:
                    finished_recving.update(req_ids)
                if self.host_tier:
                    # The transport outcome only: the report carries the
                    # event's epoch and the scheduler invalidates an older
                    # epoch's load itself (a stale-epoch completion is not a
                    # transport failure).
                    ok = j not in self._failed_load_events
                    self._load_outcomes[(event_epoch, j)] = ok
                if self.host_tier and (
                    j in self._failed_load_events or event_epoch != self._cache_epoch
                ):
                    self._failed_load_events.discard(j)
                    self._failed_recving.update(req_ids or ())

        if self._pending_store_event_indices:
            store_wm = self._poll_stream_events(is_store=True)
            for j in [j for j in self._pending_store_event_indices if j <= store_wm]:
                self._pending_store_event_indices.discard(j)
                self._completed_store_events[j] = 1
                if getattr(self, "host_tier", False):
                    self._init_event_leases()
                    epoch = self._store_event_epochs.pop(j, self._cache_epoch)
                    self._store_outcomes[(epoch, j)] = (
                        j not in self._failed_store_events
                    )
                    self._failed_store_events.discard(j)

        return None, finished_recving or None

    def take_failed_recving(self) -> set[str]:
        """Requests whose restore failed since the last call (they are also in
        that step's finished_recving)."""
        failed, self._failed_recving = self._failed_recving, set()
        return failed

    def build_connector_worker_meta(self) -> SimpleCPUOffloadWorkerMetadata | None:
        """Return stores, census samples and content-writer retirement,
        even without stores."""
        tracker = getattr(self, "commit_acks", None)
        steps, masked = {}, set()
        if tracker is not None:
            committed, masked = tracker.report()
            steps[self.vllm_config.parallel_config.rank] = committed
        # Flush samples need no store receipt.
        census_dirty = getattr(self, "_census_dirty", False)
        outcomes = getattr(self, "_load_outcomes", None) or getattr(
            self, "_store_outcomes", None
        )
        if (
            not self._completed_store_events
            and tracker is None
            and not census_dirty
            and not outcomes
        ):
            return None
        meta = SimpleCPUOffloadWorkerMetadata(
            completed_store_events=self._completed_store_events,
            census={self._census_rank: dict(self._census)} if census_dirty else {},
            committed_steps=steps,
            masked_rows=masked,
        )
        # Per-rank acks for demotion txns.
        if getattr(self, "host_tier", False):
            rank = frozenset((self.vllm_config.parallel_config.rank,))
            meta.store_ack_ranks = dict.fromkeys(self._completed_store_events, rank)
            # The copy event completed, so the
            # rank is quiescent whether the copy succeeded or failed.
            my_rank = next(iter(rank))
            meta.store_reports = {
                (epoch, event, 0): {my_rank: (ok, True)}
                for (epoch, event), ok in self._store_outcomes.items()
            }
            meta.load_reports = {
                (epoch, event, 0): {my_rank: (ok, True)}
                for (epoch, event), ok in self._load_outcomes.items()
            }
            self._store_outcomes = {}
            self._load_outcomes = {}
        self._completed_store_events = {}
        self._census_dirty = False
        return meta

    def handle_preemptions(
        self, kv_connector_metadata: SimpleCPUOffloadMetadata
    ) -> None:
        """Protect preempted transfers until their storage can be reused."""
        # The manager retains per-transfer refs
        # through all-rank completion, including cancelled load destinations.
        if self.host_tier and kv_connector_metadata.transfer_leases:
            if isinstance(self._backend, DmaCopyBackend):
                self._backend.check_error()
            self._poll_stream_events(is_store=False)
            self._poll_stream_events(is_store=True)
            return
        if not kv_connector_metadata.need_flush:
            return
        started = time.monotonic()
        self._flush_and_sync_all()
        if getattr(self, "host_tier", False):
            self._census["flushes"] = self._census.get("flushes", 0) + 1
            elapsed = 1000 * (time.monotonic() - started)
            self._census["flush_ms"] = self._census.get("flush_ms", 0) + elapsed
            self._census_dirty = True

    def _flush_and_sync_all(self) -> None:
        """Synchronize all in-flight transfer events."""
        # First drain the copy queues: a transfer still queued or mid-chunk
        # has not appended its event yet.
        if isinstance(self._backend, DmaCopyBackend):
            self._backend.flush()
        # Pop rather than iterate + clear(): the copy threads append to these
        # lists concurrently, and an event appended between the loop and a
        # clear() would be dropped unsynchronized (its index never reported).
        while self._load_events:
            event_idx, event = self._load_events.pop(0)
            event.synchronize()
            self._load_hwm = event_idx
        # The pops above leave the list empty; no clear().

        while self._store_events:
            event_idx, event = self._store_events.pop(0)
            event.synchronize()
            self._store_hwm = event_idx
            self._collect_store_census(event_idx)

    def _poll_stream_events(self, is_store: bool) -> int:
        """Non-blocking poll for completed events and return the high-water mark."""
        events = self._store_events if is_store else self._load_events
        hwm = self._store_hwm if is_store else self._load_hwm
        while events:
            event_idx, event = events[0]
            if not event.query():
                break
            hwm = event_idx
            events.pop(0)
            if is_store:
                self._collect_store_census(event_idx)
        if is_store:
            self._store_hwm = hwm
        else:
            self._load_hwm = hwm
        return hwm

    def _collect_store_census(self, event_idx: int) -> None:
        timings = getattr(getattr(self, "_backend", None), "store_census", None)
        if timings is None or event_idx not in timings:
            return
        start, end, blocks, submit_ms = timings[event_idx]
        try:
            elapsed = start.elapsed_time(end)
        finally:  # popped after the read: the copy thread reuses the pair
            timings.pop(event_idx, None)
        bucket = str(blocks) if blocks in (8, 32, 70) else "other"
        # Called only after query() or the existing flush established completion.
        for name, value in (
            ("d2h_batches", 1), ("d2h_blocks", blocks),
            ("d2h_ms", elapsed), ("d2h_submit_ms", submit_ms),
            ("d2h_ms_n" + bucket, elapsed), ("d2h_blocks_n" + bucket, blocks),
        ):
            self._census[name] = self._census.get(name, 0) + value
        self._census_dirty = True
