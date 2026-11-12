# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modified by the Merlin project; see NOTICE.
"""Scheduler-side manager for SimpleCPUOffloadConnector."""

import contextlib
import time
from collections.abc import Iterable
from dataclasses import dataclass, field, fields, replace
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

from vllm.config import VllmConfig
from vllm.distributed.kv_events import (
    MEDIUM_CPU,
    MEDIUM_STORAGE,
    BlockRemoved,
    BlockStored,
    KVCacheEvent,
)
from vllm.distributed.kv_transfer.kv_connector.utils import yield_req_data
from vllm.logger import init_logger
from vllm.merlin import (
    kv_commit_acks,
    kv_host_tier,
    kv_host_tier_hooks,
    kv_host_tier_lifecycle,
    kv_host_tier_plans,
    kv_host_tier_retention,
    kv_host_tier_runtime,
    kv_host_tier_units,
    kv_host_tier_writeback,
    kv_restore_dependencies,
)
from vllm.utils.math_utils import cdiv
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_coordinator import (
    KVCacheCoordinator,
    get_kv_cache_coordinator,
)
from vllm.v1.core.kv_cache_utils import (
    BlockHashWithGroupId,
    ExternalBlockHash,
    generate_block_hash_extra_keys,
    get_block_hash,
    get_group_id,
    make_block_hash_with_group_id,
    maybe_convert_block_hash,
    resolve_block_hashes,
    resolve_dcp_kv_block_size,
    sliding_window_context_kv_is_causal,
    to_event_extra_keys,
)
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import (
    ChunkedLocalAttentionSpec,
    FullAttentionSpec,
    MambaSpec,
    SlidingWindowSpec,
    get_kv_cache_spec_kind,
    get_kv_cache_spec_sliding_window,
)
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.simple_kv_offload.metadata import (
    SimpleCPUOffloadMetadata,
    SimpleCPUOffloadWorkerMetadata,
)
from vllm.v1.simple_kv_offload.metrics import (
    OUTCOME_TO_FIELD,
    MetricName,
    SimpleCPUOffloadStats,
)
from vllm.v1.simple_kv_offload.sizing import gpu_total_bytes

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.kv_cache_utils import BlockHashList, KVCacheBlock
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)


@dataclass
class BlockStoreMeta:
    """Per-block metadata snapshot for BlockStored event emission.

    Captured at store-prep time (when the Request is available) and carried
    through TransferMeta to async completion, where the Request is gone.
    """

    token_ids: list[int]
    parent_block_hash: ExternalBlockHash | None
    lora_id: int | None
    lora_name: str | None
    extra_keys: tuple[Any, ...] | None
    block_size: int | None = None


@dataclass
class TransferMeta:
    gpu_block_ids: list[int]
    cpu_block_ids: list[int]
    block_meta: list[dict[BlockHashWithGroupId, BlockStoreMeta]] | None = None
    # Store keys at issue (TierRuntime.snapshot).
    gpu_block_keys: list[Any] | None = None
    # Old-epoch completions only release leases.
    cache_epoch: int = 0


@dataclass
class LoadRequestState:
    request: "Request"
    transfer_meta: TransferMeta
    load_event: int | None = None
    finished: bool = False
    # A rejected load has no blocks to transfer.
    failed: bool = False

    lease: kv_host_tier_lifecycle.RestoreLease | None = None
    issued_at: float | None = None
    # A handed-off plan restore cancelled or reset
    # with no batch in flight: issue nothing more, publish one no-DMA receive
    # notice (``failed_recving``) so the scheduler frees or resumes it.
    notify_only: bool = False


# NOTE: This per-request state is only used in eager mode.
@dataclass
class StoreRequestState:
    request: "Request"
    # Accumulated block IDs from scheduler_output via yield_req_data.
    block_ids: tuple[list[int], ...]
    # Per-group cursors tracking how many blocks have been stored/skipped.
    num_stored_blocks: list[int]
    store_events: set[int] = field(default_factory=set)
    finished: bool = False


class _StoreAdmission(Enum):
    READY = auto()
    NULL_BLOCK = auto()
    NOT_HASHED = auto()
    IN_FLIGHT = auto()
    ALREADY_CACHED = auto()


@dataclass
class BoundaryStoreStats:
    """Cumulative counters for boundary hand-off stores.

    Every branch that declines a hand-off is silent by design: the boundary is
    simply not offloaded and a later request misses. Without counters the only
    symptom is a lower cache hit rate with nothing in the logs, so each decline
    reason is counted and exposed through
    ``SimpleCPUOffloadConnector.get_boundary_store_stats()``. Reset by
    ``reset()``, which carries any not-yet-drained deltas; not otherwise
    cleared.
    """

    published: int = 0
    stored: int = 0
    dropped_cpu_full: int = 0
    dropped_request_gone: int = 0
    skipped_already_cached: int = 0
    skipped_in_flight: int = 0
    dropped_null_block: int = 0
    dropped_not_hashed: int = 0

    # A hand-off for a group the tier never offloads (not prefix-cacheable,
    # see prefix_cacheable_group_ids). Only Mamba "align" managers produce
    # hand-offs, so this stays 0; it guards the skip set, e.g. the slot-packed
    # QSA ring of the side-cache layout, whose ids are ring slots, not pool
    # blocks.
    dropped_skipped_group: int = 0


@dataclass
class TierStats:
    """Cumulative host-tier counters for the periodic tier log line.

    ``loaded_tokens`` are tokens accepted from the host tier (scheduler side,
    at allocation). Reset by ``reset()``.
    """

    lookups: int = 0
    hit_lookups: int = 0
    hit_tokens: int = 0
    loaded_tokens: int = 0
    loaded_blocks: int = 0
    stored_blocks: int = 0


# Seconds between "SimpleCPU tier:" INFO lines (only when something changed).
TIER_LOG_INTERVAL_S = 60.0
# Log at least this often even when nothing changed, so a
# frozen tier is a visible flat line in an offline replay, not a gap.
TIER_LOG_FORCE_S = 300.0


class SimpleCPUOffloadScheduler:
    """Scheduler-side manager for CPU offloading."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: "KVCacheConfig | None",
        cpu_capacity_bytes: int,
        scheduler_block_size: int,
        hash_block_size: int,
        lazy_offload: bool = False,
        disk_capacity_bytes: int = 0,
        use_page_cache: bool = False,
        num_cpu_blocks: int | None = None,
        aligned_num_cpu_blocks: int | None = None,
    ):
        self.vllm_config = vllm_config
        self.kv_cache_config = kv_cache_config
        # When disk mode is active, the offload pool size is disk-based.
        offload_capacity = (
            disk_capacity_bytes if disk_capacity_bytes > 0 else cpu_capacity_bytes
        )
        self.enable_kv_cache_events = (
            vllm_config.kv_events_config is not None
            and vllm_config.kv_events_config.enable_kv_cache_events
        )
        dcp_world_size = vllm_config.parallel_config.decode_context_parallel_size
        self.cp_world_size = dcp_world_size
        self.block_size = scheduler_block_size
        self.hash_block_size = hash_block_size
        assert self.block_size % self.hash_block_size == 0
        # Derive a CPU KVCacheConfig from the GPU config and build a coordinator
        assert kv_cache_config is not None
        # Before any host memory is sized.
        if kv_host_tier.is_active(vllm_config):
            kv_host_tier.refuse_prefix_replay(kv_cache_config)
        self.cpu_kv_cache_config = self._derive_cpu_config(
            kv_cache_config, offload_capacity, num_cpu_blocks, aligned_num_cpu_blocks
        )
        self.num_cpu_blocks = self.cpu_kv_cache_config.num_blocks
        self.prefix_cacheable_group_ids = (
            self.cpu_kv_cache_config.prefix_cacheable_group_ids
        )
        self.kv_event_medium = MEDIUM_STORAGE if disk_capacity_bytes > 0 else MEDIUM_CPU
        # Must track kv_event_medium.
        self._info_labelvalues: tuple[str, ...] = (
            "disk" if disk_capacity_bytes > 0 else "cpu",
            str(use_page_cache).lower(),
            str(lazy_offload).lower(),
            str(self.num_cpu_blocks),
        )
        # Find the full attention kv group for prefix cache matching.
        self.fa_gidx = -1
        for g_idx, g in enumerate(self.cpu_kv_cache_config.kv_cache_groups):
            if isinstance(g.kv_cache_spec, FullAttentionSpec):
                self.fa_gidx = g_idx
                break
        assert 0 <= self.fa_gidx < len(self.cpu_kv_cache_config.kv_cache_groups)
        # Recurrent state is overwritten in place.
        self._state_group_ids = frozenset(
            g_idx
            for g_idx, g in enumerate(self.cpu_kv_cache_config.kv_cache_groups)
            if isinstance(g.kv_cache_spec, MambaSpec)
        )
        logger.info(
            "SimpleCPUOffloadScheduler: Allocating %d offload blocks "
            "(%.2f GB, mode=%s, backend=%s)",
            self.num_cpu_blocks,
            offload_capacity / (1024**3),
            # The tier is write-back, not eager.
            "write-back"
            if not lazy_offload and kv_host_tier.is_active(vllm_config)
            else "lazy"
            if lazy_offload
            else "eager",
            "disk" if disk_capacity_bytes > 0 else "cpu",
        )

        spec_config = vllm_config.speculative_config
        use_eagle_block_drop = (
            spec_config is not None and spec_config.use_eagle_block_drop()
        )
        self.cpu_coordinator: KVCacheCoordinator = get_kv_cache_coordinator(
            kv_cache_config=self.cpu_kv_cache_config,
            max_model_len=vllm_config.model_config.max_model_len,
            max_in_flight_tokens=vllm_config.max_in_flight_tokens,
            use_eagle=use_eagle_block_drop,
            enable_caching=True,
            enable_kv_cache_events=self.enable_kv_cache_events,
            dcp_world_size=dcp_world_size,
            pcp_world_size=1,
            scheduler_block_size=self.block_size,
            hash_block_size=self.hash_block_size,
            allow_partial_hash_hits=not lazy_offload,
            num_prefill_lookahead=vllm_config.num_prefill_lookahead_tokens,
            causal_sliding_window_kv=sliding_window_context_kv_is_causal(
                vllm_config.speculative_config
            ),
        )
        # The tier uses resolved lookup dependencies.
        if kv_host_tier.is_active(vllm_config):
            min_rows = kv_host_tier.min_tier_rows(self.cpu_kv_cache_config)
            data_rows = self.num_cpu_blocks - 1  # the pool reserves its null row
            if data_rows < min_rows:
                raise RuntimeError(
                    f"host-RAM KV tier has {data_rows} host rows per rank; "
                    f"one 262K restore unit needs {min_rows}. Increase "
                    f"{kv_host_tier.KNOB} or set it to 0 to disable the tier."
                )
            kv_host_tier.warn_if_below_restore_unit(
                kv_cache_config,
                cpu_capacity_bytes,
                vllm_config.parallel_config.world_size,
                coordinator=self.cpu_coordinator,
            )
        self.group_block_sizes = self.cpu_coordinator.group_block_sizes
        # FA group's own resolved block_size; divides scheduler_block_size (the
        # LCM) but is NOT assumed to equal it.
        self.fa_block_size: int = self.group_block_sizes[self.fa_gidx]
        assert self.block_size % self.fa_block_size == 0
        self.cpu_block_pool: BlockPool = self.cpu_coordinator.block_pool
        # GPU block pool reference - bound after scheduler builds kv_cache_manager
        self._gpu_block_pool: BlockPool | None = None

        # Load metadata
        self._reqs_to_load: dict[str, LoadRequestState] = {}
        # Inverse map: load_event_idx -> req_ids. Keyed by load_event_idx because
        # the worker reports completions by event index, not request id.
        self._load_event_to_reqs: dict[int, list[str]] = {}

        # Pending (cpu_hit_blocks, hit_length, num_computed_tokens) tuples from
        # find_longest_cache_hit, kept pinned via touch() while awaiting
        # update_state_after_alloc(). ``num_computed_tokens`` is the local
        # prefix the hit was resolved against; the load path needs it to place
        # external blocks and must not re-derive it from block metadata.
        self._pending_cpu_hits: dict[
            str, tuple[tuple[list[KVCacheBlock], ...], int, int]
        ] = {}

        # Store metadata
        self._lazy_mode = lazy_offload
        # Lazy mode: use a cursor to track the last scanned block in the GPU free queue.
        self._cursor: KVCacheBlock | None = None
        if self._lazy_mode:
            self._target_free = self._estimate_lazy_target_blocks(
                kv_cache_config,
                vllm_config.scheduler_config.max_num_batched_tokens,
                self.cp_world_size,
            )
        else:
            self._target_free = 0
        self._store_event_to_blocks: dict[int, TransferMeta] = {}
        self._abandoned_store_event_to_blocks: dict[int, TransferMeta] = {}
        # Eager mode only
        self._reqs_to_store: dict[str, StoreRequestState] = {}
        self._store_event_to_reqs: dict[int, list[str]] = {}
        self._in_flight_store_gpu_blocks: set[int] = set()
        self._pending_finished_stores: list[TransferMeta] = []
        self._abandoned_reqs_to_load: dict[str, LoadRequestState] = {}

        # Event counters
        self._load_event_counter: int = 0
        self._store_event_counter: int = 0
        self.boundary_store_stats = BoundaryStoreStats()
        # Interval stats state drained by get_stats()
        self._boundary_stats_snapshot = BoundaryStoreStats()
        self._interval_load_blocks_completed = 0

        # Commit readiness is rank- and writer-specific, independent of PP depth.
        self.commit_acks = kv_commit_acks.SchedulerCommitAcks(
            vllm_config.parallel_config.world_size
        )
        self._sched_step = 0
        self._masked_pruned = 0
        # Lifecycle hooks (the copy
        # scheduler); None until ``install_tier_hooks``.
        self.tier_hooks: Any = None
        self.tier_stats = TierStats()
        self._tier_log_last = time.monotonic()
        self._tier_log_logged: TierStats | None = None
        self._wb_log_logged: Any = None
        self._tier_log_forced = time.monotonic()
        # The tier runtime (store keys, load
        # checks and failures, metrics); None for every other SimpleCPUOffload
        # user.
        self.tier_runtime: kv_host_tier_runtime.TierRuntime | None = (
            kv_host_tier_runtime.TierRuntime(
                self.cpu_block_pool,
                self.num_cpu_blocks,
                self.fa_gidx,
                self.fa_block_size,
            )
            if not lazy_offload and kv_host_tier.is_active(vllm_config)
            else None
        )
        # The tier is write-back: victims are
        # copied from the GPU free-queue head, never at compute time;
        # positional eager selection is not used.
        self.writeback: kv_host_tier_writeback.WriteBack | None = None
        if self.tier_runtime is not None:
            # The metrics ledger
            # needs the hash grid to round reusable tokens to boundaries.
            kv_host_tier_hooks.bind_tier_geometry(self, kv_cache_config)
            reserve, align = kv_host_tier_writeback.derive_reserve(
                kv_cache_config,
                vllm_config.scheduler_config.max_num_batched_tokens,
                self.cp_world_size,
            )
            self.writeback = kv_host_tier_writeback.WriteBack(
                self.cpu_block_pool,
                self.prefix_cacheable_group_ids,
                reserve,
                align,
                vllm_config.parallel_config.world_size,
            )
            logger.info(
                "host-RAM KV tier: write-back, reserve R=%d blocks (+%d per "
                "running request), host LRU, "
                "copy credits P=%d, escape after %d steps",
                reserve,
                align,
                self.writeback.credits_cap,
                self.writeback.escape_after,
            )
            # The walker's counters
            # feed the per-interval series line (SERIES_FIELDS_V3).
            self.tier_runtime.series_sources.append(
                lambda: vars(self.writeback.counters)
            )

        # For TP/PP: track partial store completions across steps.
        # Events must be reported by all world_size workers before considered complete.
        self._expected_worker_count = vllm_config.parallel_config.world_size
        self._store_event_pending_counts: dict[int, int] = {}
        # Per-rank load transaction acks, keyed
        # (epoch, load event, batch).
        self._load_acks: dict[
            tuple[int, int, int], kv_host_tier_lifecycle.TxnAcks
        ] = {}
        self._load_resolved: dict[tuple[int, int, int], None] = {}
        self.store_txn_failures = 0
        self.load_txn_failures = 0
        self.load_stale_epoch_completions = 0
        self.txn_report_mismatches = 0

        if self.tier_runtime is not None:
            self._census()
        # Monotonic epochs and transaction ownership.
        self.cache_epoch = 0
        self._promotions: dict[tuple, str] = {}
        self._promotion_waiters: dict[tuple, set[str]] = {}
        self._ready_promotions: dict[str, LoadRequestState] = {}
        self._store_issued_at: dict[int, float] = {}
        # PrefixHitPlan admission.
        self.tier_plans: kv_host_tier_plans.TierPlans | None = (
            kv_host_tier_plans.TierPlans(self) if self.tier_runtime is not None else None
        )
        if self.tier_plans is not None:
            self.tier_runtime.extra_counters = self.tier_plans.log_counters

    @staticmethod
    def _derive_cpu_config(
        gpu_config: "KVCacheConfig",
        cpu_capacity_bytes: int,
        num_cpu_blocks: int | None = None,
        aligned_num_cpu_blocks: int | None = None,
    ) -> "KVCacheConfig":
        """Derive a CPU KVCacheConfig from the GPU config.
        Same kv_cache_groups, num_blocks scaled by CPU/GPU memory ratio.

        An explicit ``num_cpu_blocks`` is checked against local capacity.
        ``aligned_num_cpu_blocks`` comes from the live worker allocations and
        is used as-is when no explicit count is configured."""
        # Import here to avoid potential circular imports
        from vllm.v1.kv_cache_interface import KVCacheTensor

        assert len(gpu_config.kv_cache_tensors) > 0

        num_gpu_blocks = gpu_config.num_blocks
        local_capacity = (
            num_gpu_blocks * cpu_capacity_bytes // gpu_total_bytes(gpu_config)
        )
        if num_cpu_blocks is None:
            num_cpu_blocks = (
                aligned_num_cpu_blocks
                if aligned_num_cpu_blocks is not None
                else max(1, local_capacity)
            )
        elif (
            type(num_cpu_blocks) is not int or not 0 < num_cpu_blocks <= local_capacity
        ):
            raise ValueError(
                f"num_cpu_blocks={num_cpu_blocks!r} must be between 1 and "
                f"local offload capacity {local_capacity}"
            )
        # Create CPU kv_cache_tensors mirroring GPU by scaling size proportionally.
        cpu_tensors = [
            KVCacheTensor(
                size=t.size // num_gpu_blocks * num_cpu_blocks,
                layers=list(t.layers),
                layer_stride=t.layer_stride,
                block_stride=t.block_stride,
                offset=t.offset,

                slots_per_block=t.slots_per_block,
            )
            for t in gpu_config.kv_cache_tensors
        ]

        return replace(
            gpu_config,
            num_blocks=num_cpu_blocks,
            kv_cache_tensors=cpu_tensors,
        )

    @staticmethod
    def _estimate_lazy_target_blocks(
        kv_cache_config: "KVCacheConfig",
        max_num_batched_tokens: int,
        cp_world_size: int = 1,
    ) -> int:
        """GPU blocks to keep available (free/offloaded) per step in lazy mode."""
        WATERMARK_RATIO = 1.0  # Reserve larger space to avoid running out of GPU blocks
        target = 0
        for g in kv_cache_config.prefix_cacheable_groups:
            spec = g.kv_cache_spec
            block_size = resolve_dcp_kv_block_size(spec, cp_world_size)
            if isinstance(spec, MambaSpec):
                target += 2
            elif isinstance(spec, SlidingWindowSpec):
                target += cdiv(spec.sliding_window, block_size) + 1
            else:
                target += cdiv(max_num_batched_tokens, block_size)
        return int(target * (1 + WATERMARK_RATIO))

    def bind_gpu_block_pool(self, gpu_block_pool: BlockPool) -> None:
        """Bind GPU block pool so that we can touch blocks during stores.
        Called by Scheduler after kv_cache_manager is ready."""
        self._gpu_block_pool = gpu_block_pool
        # Unified tier: GPU block state + units.
        if self.tier_runtime is not None and gpu_block_pool.tier is None:
            self.tier_runtime.units = kv_host_tier_units.mount(
                gpu_block_pool,
                self.cpu_block_pool,
                self.cpu_kv_cache_config,
                self.prefix_cacheable_group_ids,
                self.group_block_sizes,
                self.hash_block_size,
                # The lookup's EAGLE groups.
                self.cpu_coordinator.eagle_group_ids,
                # Its resolved fine lookup.
                kv_restore_dependencies.fine_grained_group_ids(
                    self.cpu_coordinator
                ),
            )
        # Write-back engine on the tier state.
        if self.writeback is not None:
            self.writeback.bind(gpu_block_pool, self.tier_runtime.units)
            # Admitted restores are bounded by a restores-in-flight
            # cap (the retention account is gone).
            if self.tier_runtime.units is not None:
                self.tier_runtime.units.restores_in_flight = (
                    kv_host_tier_retention.RestoresInFlight(
                        kv_host_tier_writeback.n_restore()
                    )
                )

        if self.tier_runtime is not None:
            self._census()



    def bind_kv_cache_manager(self, kv_cache_manager: Any) -> None:
        """Install the write-back admission gate."""
        if self.writeback is not None:
            kv_cache_manager.admission_gate = self.writeback.admit
            kv_cache_manager.running_headroom_hook = self.writeback.account_running
            kv_cache_manager.head_need_hook = self.writeback.note_head

        if self.tier_plans is not None:
            self.tier_plans.bind(kv_cache_manager)
        # The copy scheduler gets
        # the KVCacheManager (grace pins, SessionYield order) and the host
        # row bytes per rank for its byte credits; it then becomes the
        # lifecycle hook target. None without a unit table.
        if self.writeback is not None:
            row_bytes = kv_host_tier.host_block_bytes(self.cpu_kv_cache_config)
            self.writeback.row_bytes = row_bytes
            if self.tier_runtime is not None:
                # The same row size for the series' D2H GB/s per rank.
                self.tier_runtime.row_bytes = row_bytes
            cq = self.writeback.bind_manager(kv_cache_manager)
            if cq is not None:
                self.install_tier_hooks(cq)
        # Lifecycle: the KVCacheManager's grace
        # take/release reach the copy scheduler (adds the step). Tier only.
        if self.tier_runtime is not None:
            kv_cache_manager.tier_hooks = kv_host_tier_hooks.KVCMLifecycleAdapter(self)

    # Lifecycle plumbing (``kv_host_tier_hooks``): the
    # copy scheduler registers itself here; no-ops until it does.
    def install_tier_hooks(self, hooks: Any) -> None:
        self.tier_hooks = hooks

    def tier_preempted(self, request: "Request") -> None:
        kv_host_tier_hooks.notify_preempted(self, request)

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        """Return (num_new_tokens, is_async) from consecutive CPU cache hits."""
        # Pins found CPU blocks so they survive LRU eviction until
        # update_state_after_alloc() consumes them. Any pin from an earlier
        # call on the same request (e.g. retry after a failed allocate_slots)
        # is dropped first.
        if stale := self._pending_cpu_hits.pop(request.request_id, None):
            self._free_pending_cpu_hit(stale)

        if request.skip_reading_prefix_cache:
            return 0, False


        self.tier_stats.lookups += 1
        if self.tier_runtime is not None:
            self.tier_runtime.on_lookup(
                request.request_id, request.num_tokens - 1 - num_computed_tokens
            )
        if num_computed_tokens % self.block_size != 0:
            # Transfers are whole-block copies, so an external suffix cannot
            # start in the middle of a destination block. When the GPU-side
            # coordinator also serves fine-grained hits, the local prefix can
            # land on a hash boundary that is not a scheduler-block boundary;
            # such a request keeps its local hit and skips the external one.
            logger.warning_once(
                "SimpleCPU external lookup requires scheduler-block-aligned "
                "local tokens, got %d tokens with scheduler block size %d.",
                num_computed_tokens,
                self.block_size,
            )
            return 0, False

        # Suffix-relative chunk boundaries must match absolute token positions.
        for group_id in self.prefix_cacheable_group_ids:
            spec = self.cpu_kv_cache_config.kv_cache_groups[group_id].kv_cache_spec
            if (
                isinstance(spec, ChunkedLocalAttentionSpec)
                and num_computed_tokens % spec.attention_chunk_size
            ):
                return 0, False

        num_skipped_hashes = num_computed_tokens // self.hash_block_size
        remaining_hashes = request.block_hashes[num_skipped_hashes:]

        if not remaining_hashes:
            return 0, False
        # Must recompute at least the last token, matching the logic in
        # kv_cache_manager.get_computed_blocks().
        max_hit_len = request.num_tokens - 1 - num_computed_tokens
        if max_hit_len <= 0:
            return 0, False
        cpu_hit_blocks, hit_length, _ = self.cpu_coordinator.find_longest_cache_hit(
            remaining_hashes, max_hit_len
        )

        if hit_length > 0:
            pin_blocks = [
                blk for grp in cpu_hit_blocks for blk in grp if not blk.is_null
            ]
            self.cpu_block_pool.touch(pin_blocks)
            self._pending_cpu_hits[request.request_id] = (
                cpu_hit_blocks,
                hit_length,
                num_computed_tokens,
            )

            self.tier_stats.hit_lookups += 1
            self.tier_stats.hit_tokens += hit_length
            return hit_length, True
        return 0, False


    def _prune_masked_rows(self) -> None:
        """Every 256 steps drop masked (producer, step) rows no live GPU
        content carries any more."""
        if self._sched_step - self._masked_pruned < 256:
            return
        self._masked_pruned = self._sched_step
        pool = self._gpu_block_pool
        tier = pool.tier if pool is not None else None
        by_producer = tier._by_producer if tier is not None else {}
        self.commit_acks.prune(lambda row: row in by_producer)



    def _retire_ready_promotions(self) -> None:
        pool = self._gpu_block_pool
        if pool is None:
            return
        for req_id, state in list(self._ready_promotions.items()):
            lease = state.lease
            assert lease is not None
            valid = (
                lease.epoch == self.cache_epoch
                and lease.sources_valid(self.cpu_block_pool)
            )
            published = lease.published(pool)
            if valid and state.finished and not published:
                lease.publish_cancelled(
                    pool,
                    state.request.num_computed_tokens,
                    self._state_group_ids,
                )
                published = True
            if valid and not published and self._promotion_waiters.get(lease.key):
                continue
            self._ready_promotions.pop(req_id)
            self._promotions.pop(lease.key, None)
            self._promotion_waiters.pop(lease.key, None)
            self._release_transfer_refs(state.transfer_meta)
            # Soft promotion (units.on_host_released)
            # also for a coalesced restore retired here.
            if self.tier_runtime is not None and self.tier_runtime.units is not None:
                self.tier_runtime.units.on_host_released(
                    state.transfer_meta.cpu_block_ids
                )


    def _check_transfer_age(self) -> None:
        if self.tier_runtime is None:
            return
        now = time.monotonic()
        issued = list(self._store_issued_at.values()) + [
            state.issued_at
            for state in (
                *self._reqs_to_load.values(),
                *self._abandoned_reqs_to_load.values(),
                *self._ready_promotions.values(),
            )
            if state.issued_at is not None
        ]
        if any(
            now - start > kv_host_tier_lifecycle.TRANSFER_TIMEOUT_S
            for start in issued
        ):
            # No completion means no proof of quiescence. Keep all leases.
            raise RuntimeError("host-tier transfer timed out; storage quarantined")


    def _census(self):
        census = self.tier_runtime.census
        gpu = self._gpu_block_pool
        # With the census off only the scheduler reference is bound.
        if (census.scheduler is not self or
                (census.enabled and gpu is not None and
                 getattr(gpu.evict_groups, "census", None) is not census)):
            census.bind(self)
        return census

    def get_tier_stats(self) -> TierStats:
        return replace(self.tier_stats)

    def _cached_blocks_by_group(self) -> list[int]:
        """CPU blocks currently holding a cached hash, per KV cache group."""
        counts = [0] * len(self.cpu_kv_cache_config.kv_cache_groups)
        for blk in self.cpu_block_pool.blocks:
            bhash = blk.block_hash
            if bhash is not None and not blk.is_null:
                counts[get_group_id(bhash)] += 1
        return counts

    def _maybe_log_tier(self, now: float | None = None) -> None:
        """One INFO line per TIER_LOG_INTERVAL_S while the counters move.

        For log readers: occupancy per group (density), host-tier
        tokens (prefill by source), boundary-store drops and the grid-skip
        cost. Cumulative counters; take deltas between lines.
        """
        now = time.monotonic() if now is None else now
        if now - self._tier_log_last < TIER_LOG_INTERVAL_S:
            return
        self._tier_log_last = now
        stats = self.tier_stats
        census = (self.tier_runtime.snapshot_counters()
                  if self.tier_runtime is not None else None)
        # plan/backing/txn counters move too.
        plans = self.tier_plans.log_counters() if self.tier_plans else None
        # Write-back counters move (demote
        # misses, drops, credits) without TierStats / census / plans moving;
        # they are part of the change test, and a forced line every
        # TIER_LOG_FORCE_S keeps a frozen tier visible in a replay.
        wbc = (self.writeback.snapshot_counters()
               if self.writeback is not None else None)
        if (stats == self._tier_log_logged and
                census == getattr(self, "_census_log_logged", None)
                and plans == getattr(self, "_plans_log_logged", None)
                and wbc == self._wb_log_logged
                and now - self._tier_log_forced < TIER_LOG_FORCE_S):
            return
        self._tier_log_forced = now
        self._wb_log_logged = wbc
        self._plans_log_logged = plans
        self._tier_log_logged = replace(stats)
        self._census_log_logged = census
        by_group = self._cached_blocks_by_group()
        b = self.boundary_store_stats
        logger.info(
            "SimpleCPU tier: cpu_blocks used=%d/%d cached_by_group=%s "
            "fa_group=%d fa_block_size=%d "
            "lookups=%d hit_lookups=%d hit_tokens=%d loaded_tokens=%d "
            "loaded_blocks=%d stored_blocks=%d boundary_stored=%d "
            "boundary_dropped_cpu_full=%d census=%s",
            # The pool's block 0 is the null block, never free and never used.
            self.num_cpu_blocks - 1 - self.cpu_block_pool.get_num_free_blocks(),
            self.num_cpu_blocks - 1,
            by_group,
            self.fa_gidx,
            self.fa_block_size,
            stats.lookups,
            stats.hit_lookups,
            stats.hit_tokens,
            stats.loaded_tokens,
            stats.loaded_blocks,
            stats.stored_blocks,
            b.stored,
            b.dropped_cpu_full,
            vars(census) if census is not None else {},
        )
        # Write-back counters.
        if wbc is not None:
            logger.info(
                "SimpleCPU tier write-back: %s",
                " ".join(
                    f"{f.name}={getattr(wbc, f.name)}"
                    for f in fields(kv_host_tier_writeback.WritebackCounters)
                ),
            )
        # Plan admission, union benefit,
        # backing provenance, transactions, name=int.
        if plans is not None:
            logger.info(
                "SimpleCPU tier plans: %s",
                " ".join(f"{name}={value}" for name, value in plans.items()),
            )

    # TODO(yifan): this API now only matches the suffix part of the prefix cache. A more
    # general API should scan blocks in both GPU and CPU block pool in a single pass.
    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        req_id = request.request_id
        block_ids_by_group = blocks.get_block_ids()
        num_groups = len(block_ids_by_group)

        # Store tracking (eager mode only). Register the request;
        # block IDs are accumulated from scheduler_output in
        # _prepare_eager_store_specs via yield_req_data.
        # Write-back keeps no per-request store state.
        if (
            not self._lazy_mode
            and self.writeback is None
            and req_id not in self._reqs_to_store
        ):
            self._reqs_to_store[req_id] = StoreRequestState(
                request=request,
                block_ids=tuple([] for _ in range(num_groups)),
                num_stored_blocks=[0] * num_groups,
            )

        # A plan admission builds its load
        # from the plan's slots.
        if self.tier_plans is not None and self.tier_plans.on_admitted(
            request, blocks
        ):
            return
        # Pop the CPU hit cached by get_num_new_matched_tokens(). The
        # found blocks were pinned there to survive LRU eviction in the window
        # between get_num_new_matched_tokens() and this matching call.
        pending = self._pending_cpu_hits.pop(req_id, None)

        if self.tier_runtime is not None:
            self._census().admission(req_id, num_external_tokens, pending)
            self.tier_runtime.on_admission(req_id, num_external_tokens)

        if num_external_tokens == 0:
            if pending is not None:
                logger.warning(
                    "SimpleCPUOffloadScheduler: update_state_after_alloc "
                    "called for req_id=%s with no external tokens but "
                    "get_num_new_matched_tokens() unexpectedly recorded "
                    "a pending CPU hit; releasing the stale pin.",
                    req_id,
                )
                self._free_pending_cpu_hit(pending)
            return

        if pending is None:
            logger.warning(
                "SimpleCPUOffloadScheduler: update_state_after_alloc called "
                "for req_id=%s with num_external_tokens=%d but no pending "
                "CPU hit from get_num_new_matched_tokens(); skipping load.",
                req_id,
                num_external_tokens,
            )
            return

        cpu_hit_blocks_full, _, num_computed_tokens = pending

        assert num_computed_tokens % self.block_size == 0, (
            "SimpleCPU external loads must start at a scheduler-block boundary"
        )
        # Fine-grained hits are hash-block aligned, not group-block aligned, so
        # the per-group block count below rounds up instead of dividing exactly.
        # Hash alignment is still required: it is what makes the accepted
        # external suffix start on a boundary every group can address.
        assert num_external_tokens % self.hash_block_size == 0, (
            f"num_external_tokens={num_external_tokens} is not aligned to "
            f"hash_block_size={self.hash_block_size}"
        )

        # The scheduler may have accepted fewer blocks than
        # get_num_new_matched_tokens() reported.
        # (e.g. due to token budget in test_partial_gpu_prefix_plus_cpu_load).
        # Take only the leading N blocks per group matching num_external_tokens;
        # the rest will be released along with the temp pin below.
        cpu_hit_blocks: list[list[KVCacheBlock]] = []
        for g in range(num_groups):
            if g not in self.prefix_cacheable_group_ids:
                cpu_hit_blocks.append([])
                continue
            g_block_size = self.group_block_sizes[g]
            n_take_g = cdiv(num_external_tokens, g_block_size)
            cpu_hit_blocks.append(cpu_hit_blocks_full[g][:n_take_g])

        gpu_block_ids: list[int] = []
        cpu_block_ids: list[int] = []
        cpu_blocks_to_touch: list[KVCacheBlock] = []

        for g in range(num_groups):
            cpu_blocks_g = cpu_hit_blocks[g]
            n_ext_g = len(cpu_blocks_g)
            if n_ext_g == 0:
                continue

            g_block_size = self.group_block_sizes[g]
            gpu_ext_start = num_computed_tokens // g_block_size
            group_gpu_ids = block_ids_by_group[g]

            for i, cpu_blk in enumerate(cpu_blocks_g):
                # Skip null blocks (e.g. sliding window or mamba padding).
                if cpu_blk.is_null:
                    continue
                gpu_block_ids.append(group_gpu_ids[gpu_ext_start + i])
                cpu_block_ids.append(cpu_blk.block_id)
                cpu_blocks_to_touch.append(cpu_blk)

        # Touch CPU blocks to prevent eviction during async load.
        self.cpu_block_pool.touch(cpu_blocks_to_touch)
        # Release the temporary pin held since get_num_new_matched_tokens().
        self._free_pending_cpu_hit(pending)

        # Touch GPU blocks to prevent freeing during async load
        assert self._gpu_block_pool is not None

        if self.tier_runtime is not None:
            self.tier_runtime.check_load_destinations(
                self._gpu_block_pool, gpu_block_ids, req_id
            )
        self._gpu_block_pool.touch(
            [self._gpu_block_pool.blocks[bid] for bid in gpu_block_ids]
        )

        assert self._reqs_to_load.get(req_id) is None
        self._reqs_to_load[req_id] = LoadRequestState(
            request=request,
            transfer_meta=TransferMeta(
                gpu_block_ids, cpu_block_ids, cache_epoch=self.cache_epoch
            ),
        )

        self.tier_stats.loaded_tokens += num_external_tokens
        self.tier_stats.loaded_blocks += len(cpu_block_ids)

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> SimpleCPUOffloadMetadata:

        self._retire_ready_promotions()
        self._check_transfer_age()
        self._sched_step += 1
        if self.tier_runtime is not None:
            self._census().step(scheduler_output)
        self._maybe_log_tier()

        gpu_tier = self._gpu_block_pool.tier if self._gpu_block_pool else None
        if gpu_tier is not None:
            gpu_tier.advance_step(self._sched_step + 1)
        # --- Stores ---
        store_event = -1
        store_gpu, store_cpu, store_req_ids, store_meta = self.prepare_store_specs(
            scheduler_output
        )
        for transfer in self._pending_finished_stores:
            store_gpu.extend(transfer.gpu_block_ids)
            store_cpu.extend(transfer.cpu_block_ids)
            if store_meta is not None:
                assert transfer.block_meta is not None
                store_meta.extend(transfer.block_meta)
        self._pending_finished_stores.clear()

        if store_gpu:
            store_event = self._store_event_counter
            self._store_event_counter += 1

            self._store_issued_at[store_event] = time.monotonic()
            self._store_event_to_blocks[store_event] = TransferMeta(

                store_gpu, store_cpu, store_meta, cache_epoch=self.cache_epoch
            )

            if self.tier_runtime is not None:
                assert self._gpu_block_pool is not None
                self._store_event_to_blocks[store_event].gpu_block_keys = (
                    self.tier_runtime.snapshot(self._gpu_block_pool, store_gpu)
                )
            # One store event = one demotion txn.
            if self.writeback is not None:
                self.writeback.issue(store_event, store_gpu)
            if store_req_ids:  # For eager mode only, track req->blocks mapping
                self._store_event_to_reqs[store_event] = store_req_ids
                for req_id in store_req_ids:
                    store_state = self._reqs_to_store.get(req_id)
                    if store_state is not None:
                        store_state.store_events.add(store_event)

        # --- Loads ---
        load_event = -1
        load_gpu: list[int] = []
        load_cpu: list[int] = []
        load_req_ids: list[str] = []
        # Per request (gpu, cpu) of this tick.
        load_parts: list[tuple[list[int], list[int]]] = []
        failed_recving: set[str] = set()

        if self.tier_plans is not None:
            self.tier_plans.begin_tick()
        for req_id, load_state in self._reqs_to_load.items():
            if load_state.load_event is not None:
                continue
            # Publish once; -1 means no DMA.
            if load_state.failed or load_state.notify_only:
                load_state.load_event = -1
                failed_recving.add(req_id)
                continue
            assert load_state.transfer_meta is not None
            # Plan restores issue in
            # credit-bounded batches; the manifest stays leased throughout.
            batch = (
                self.tier_plans.next_batch(req_id)
                if self.tier_plans is not None
                else None
            )
            if batch is not None:
                if not batch[0]:
                    continue
                load_gpu.extend(batch[0])
                load_cpu.extend(batch[1])
                load_req_ids.append(req_id)

                load_parts.append(batch)
                continue
            load_gpu.extend(load_state.transfer_meta.gpu_block_ids)
            load_cpu.extend(load_state.transfer_meta.cpu_block_ids)
            load_req_ids.append(req_id)

            load_parts.append((
                list(load_state.transfer_meta.gpu_block_ids),
                list(load_state.transfer_meta.cpu_block_ids),
            ))
        # With the tier, one load event (one
        # per-rank load transaction) per request: a copy failure fails and
        # un-keys only that request's restore.
        load_transfers: list[tuple[int, list[int], list[int]]] = []
        if load_req_ids and self.tier_runtime is not None:
            for req_id, (gpu, cpu) in zip(load_req_ids, load_parts):
                event = self._load_event_counter
                self._load_event_counter += 1
                self._reqs_to_load[req_id].load_event = event
                self._reqs_to_load[req_id].issued_at = time.monotonic()
                self._load_event_to_reqs[event] = [req_id]
                load_transfers.append((event, gpu, cpu))
            load_event = load_transfers[0][0]
        elif load_req_ids:
            load_event = self._load_event_counter
            self._load_event_counter += 1
            for req_id in load_req_ids:
                self._reqs_to_load[req_id].load_event = load_event

                self._reqs_to_load[req_id].issued_at = time.monotonic()
            self._load_event_to_reqs[load_event] = load_req_ids
        # The credit account counts H2D.
        if self.tier_plans is not None:
            self.tier_plans.end_tick()

        result = SimpleCPUOffloadMetadata(
            load_event=load_event,
            load_gpu_blocks=load_gpu,
            load_cpu_blocks=load_cpu,
            # Several events this tick.
            load_transfers=load_transfers if len(load_transfers) > 1 else [],
            load_event_to_reqs={
                event_idx: list(req_ids)
                for event_idx, req_ids in self._load_event_to_reqs.items()
            },
            store_event=store_event,
            store_gpu_blocks=store_gpu,
            store_cpu_blocks=store_cpu,
            # Transfer leases replace the
            # synchronize-all flush (the worker already skips it).
            need_flush=bool(scheduler_output.preempted_req_ids)
            and self.tier_runtime is None,

            failed_recving=failed_recving,

            scheduled_step=self._sched_step,
            acked_step=self.acked,
            # Every DMA address owns a transfer ref.
            transfer_leases=self.tier_runtime is not None,
            cache_epoch=self.cache_epoch,
        )
        return result

    def prepare_store_specs(
        self, scheduler_output: SchedulerOutput
    ) -> tuple[
        list[int],
        list[int],
        list[str],
        list[dict[BlockHashWithGroupId, BlockStoreMeta]] | None,
    ]:
        """Prepare store specs for the store event."""

        if self.writeback is not None:
            return self._prepare_writeback_store_specs(scheduler_output)
        if self._lazy_mode:
            return self._prepare_lazy_store_specs()
        else:
            return self._prepare_eager_store_specs(scheduler_output)


    def _prepare_writeback_store_specs(
        self, scheduler_output: SchedulerOutput
    ) -> tuple[
        list[int],
        list[int],
        list[str],
        list[dict[BlockHashWithGroupId, BlockStoreMeta]] | None,
    ]:
        """Write-back victims from the GPU free-queue head. Boundary hand-offs
        and positional blocks are not copy triggers: a retained checkpoint is
        demoted with its unit once it is a victim.
        """
        wb = self.writeback
        assert wb is not None
        loads = (*self._reqs_to_load.values(), *self._abandoned_reqs_to_load.values())
        wb.counters.promote_pins = sum(
            len(s.transfer_meta.cpu_block_ids)
            for s in loads
            if s.transfer_meta is not None
        )
        wb.promote_dma = sum(
            # A plan load's issued batch.
            self.tier_plans.dma_blocks(s)
            if self.tier_plans is not None
            else len(s.transfer_meta.cpu_block_ids)
            for s in loads
            if s.transfer_meta is not None
            and s.load_event is not None
            and s.load_event >= 0
        )
        # Staged members of assembling units are retained too.
        # One credit stays free for a
        # waiting restore batch.
        wb.promote_reserved = (
            self.tier_plans.waiting_for_credit() if self.tier_plans is not None else 0
        )
        # Quarantine: abandoned loads (reset) and cancelled loads whose DMA is
        # still in flight; both hold only the pairs
        # their outstanding batch can still write.
        wb.account_quarantine(self._quarantined_pairs())
        picked = wb.select(
            self._sched_step, len(scheduler_output.num_scheduled_tokens)
        )
        if not picked:
            return [], [], [], None
        gpu_ids = [b.block_id for b in picked]
        # The destinations are the rows the walker
        # reserved (a started unit's own reservation, or the exact idle rows
        # of a lone victim), each held with the one ref this transfer owns.
        cpu_blocks = wb.host_rows
        assert cpu_blocks is not None and len(cpu_blocks) == len(picked)
        assert all(b.ref_cnt == 1 and b.block_hash is None for b in cpu_blocks)
        cpu_ids = [b.block_id for b in cpu_blocks]
        # Payload-less events, as hand-off stores emit: the block's own hashes
        # are registered (and emitted) at completion.
        meta = [{} for _ in gpu_ids] if self.enable_kv_cache_events else None
        return gpu_ids, cpu_ids, [], meta

    def _prepare_lazy_store_specs(
        self,
    ) -> tuple[
        list[int],
        list[int],
        list[str],
        list[dict[BlockHashWithGroupId, BlockStoreMeta]] | None,
    ]:
        """Single-pass cursor walk: offload cached GPU blocks near eviction.

        Walks the GPU free queue from the cursor, counting blocks that are
        free-or-offloaded (safe for the allocator to evict). Stops when
        target_free blocks are covered or CPU capacity is reached.
        """
        gpu_pool = self._gpu_block_pool
        if gpu_pool is None or self._target_free <= 0:
            return [], [], [], None

        free_queue = gpu_pool.free_block_queue
        cpu_pool = self.cpu_block_pool
        num_cpu_free = cpu_pool.get_num_free_blocks()

        # Validate cursor: stale if block was removed from free queue.
        if self._cursor is not None and self._cursor.ref_cnt > 0:
            self._cursor = None

        gpu_ids: list[int] = []
        last_visited = self._cursor

        for covered, node in enumerate(free_queue.iter_blocks_after(self._cursor)):
            if covered >= self._target_free or len(gpu_ids) >= num_cpu_free:
                break

            last_visited = node
            bhash = node.block_hash

            if (
                bhash is not None
                and not node.is_null
                and cpu_pool.cached_block_hash_to_block.get_one_block(bhash) is None
            ):
                gpu_ids.append(node.block_id)

        self._cursor = last_visited

        # Batch-allocate CPU blocks.
        if gpu_ids:
            cpu_blocks = cpu_pool.get_new_blocks(len(gpu_ids))
            cpu_ids = [blk.block_id for blk in cpu_blocks]
            # Touch GPU blocks to prevent eviction during async copy.
            gpu_pool.touch([gpu_pool.blocks[bid] for bid in gpu_ids])
        else:
            cpu_ids = []

        return gpu_ids, cpu_ids, [], None

    def _prepare_eager_store_specs(
        self, scheduler_output: SchedulerOutput
    ) -> tuple[
        list[int],
        list[int],
        list[str],
        list[dict[BlockHashWithGroupId, BlockStoreMeta]] | None,
    ]:
        """Identify newly computed blocks to offload from scheduler requests.

        Only considers blocks whose KV data has been **confirmed computed** by
        the GPU. Blocks from the current step are stored on a later step, or by
        the finish-time flush if the request completes first.

        Returns:
            (gpu_block_ids, cpu_block_ids, req_ids, block_meta) for the store
            event. ``block_meta`` is None when kv cache events are disabled.

        """
        merged_gpu_block_ids: list[int] = []
        merged_cpu_block_ids: list[int] = []
        req_ids: list[str] = []
        merged_block_meta: list[dict[BlockHashWithGroupId, BlockStoreMeta]] | None = (
            [] if self.enable_kv_cache_events else None
        )

        gpu_block_pool = self._gpu_block_pool
        if gpu_block_pool is None:
            return [], [], [], merged_block_meta
        cpu_block_pool = self.cpu_block_pool
        num_groups = len(self.cpu_kv_cache_config.kv_cache_groups)
        # Dedup against blocks already scheduled.
        in_flight = self._in_flight_store_gpu_blocks
        num_free = cpu_block_pool.get_num_free_blocks()

        block_state = scheduler_output.kv_connector_block_state
        boundary_offloads = (
            block_state.boundary_state_offloads if block_state is not None else {}
        )
        stats = self.boundary_store_stats
        preempted_req_ids = scheduler_output.preempted_req_ids or ()
        for req_id, entries in boundary_offloads.items():
            # The scheduler drains handoffs without filtering by request
            # liveness. A request that finished or was preempted in this step
            # is giving up its blocks, which may already have been reallocated
            # to another request; reading them would publish unrelated KV under
            # a valid hash. Drop conservatively, as the mooncake store does.
            store_state = self._reqs_to_store.get(req_id)
            if (
                store_state is None
                or store_state.finished
                or req_id in preempted_req_ids
                or req_id in scheduler_output.finished_req_ids
            ):
                stats.published += len(entries)
                stats.dropped_request_gone += len(entries)
                continue
            scheduled_for_req = False
            # ``boundary_tokens`` is not needed here: the cache key comes from
            # the handed-off block itself, not from the offered boundary.

            for group_id, gpu_block_id, _ in entries:
                stats.published += 1

                if group_id not in self.prefix_cacheable_group_ids:
                    # Checked before the id is used as a pool index.
                    stats.dropped_skipped_group += 1
                    continue
                gpu_block = gpu_block_pool.blocks[gpu_block_id]
                admission = self._classify_store_candidate(gpu_block)
                if admission is _StoreAdmission.NULL_BLOCK:
                    stats.dropped_null_block += 1
                    continue
                if admission is _StoreAdmission.NOT_HASHED:
                    stats.dropped_not_hashed += 1
                    continue
                if admission is _StoreAdmission.IN_FLIGHT:
                    stats.skipped_in_flight += 1
                    continue
                if admission is _StoreAdmission.ALREADY_CACHED:
                    stats.skipped_already_cached += 1
                    continue
                if num_free <= 0:
                    stats.dropped_cpu_full += 1
                    logger.warning_once(
                        "SimpleCPU dropped a boundary-state handoff because "
                        "the CPU block pool is full; this boundary will not be retried."
                    )
                    continue
                cpu_block = cpu_block_pool.get_new_blocks(1)[0]
                merged_gpu_block_ids.append(gpu_block_id)
                merged_cpu_block_ids.append(cpu_block.block_id)
                if merged_block_meta is not None:
                    # Keep the metadata list index-parallel with the block ids.
                    # The hand-off block's own hashes are resolved at completion,
                    # so there is nothing to capture here; emit the event without
                    # a payload, as the size-mismatch fallback already does.
                    merged_block_meta.append({})
                in_flight.add(gpu_block_id)
                gpu_block_pool.touch([gpu_block])
                num_free -= 1
                stats.stored += 1
                scheduled_for_req = True
            if scheduled_for_req:
                req_ids.append(req_id)

        # Preemption frees the request's blocks. It resumes with a full new
        # block table (MRV1 resumed-cached, MRV2 NewRequestData).
        for req_id in preempted_req_ids:
            state = self._reqs_to_store.get(req_id)
            if state is not None:
                state.block_ids = tuple([] for _ in range(num_groups))
                state.num_stored_blocks = [0] * num_groups

        for req_id, block_id_groups, is_full_table in yield_req_data(scheduler_output):
            state = self._reqs_to_store.get(req_id)
            if state is None or state.finished:
                continue

            if is_full_table:
                state.block_ids = tuple([] for _ in range(num_groups))

            if block_id_groups:
                for g in range(min(num_groups, len(block_id_groups))):
                    if block_id_groups[g] is not None:
                        # Accumulate new block IDs.
                        state.block_ids[g].extend(block_id_groups[g])

            num_new_tokens = scheduler_output.num_scheduled_tokens.get(req_id, 0)
            if num_new_tokens == 0:
                continue

            block_ids_by_group = state.block_ids
            if not block_ids_by_group:
                continue

            gpu_block_ids, advanced_per_group, block_meta = (
                self._select_eager_blocks_to_store(state, block_ids_by_group)
            )

            # Batch allocate the CPU destinations.
            n_to_alloc = len(gpu_block_ids)
            if n_to_alloc > 0:
                cpu_blocks_alloc = cpu_block_pool.get_new_blocks(n_to_alloc)
                cpu_block_ids = [blk.block_id for blk in cpu_blocks_alloc]
            else:
                cpu_block_ids = []

            if cpu_block_ids:
                req_ids.append(req_id)
                merged_gpu_block_ids.extend(gpu_block_ids)
                merged_cpu_block_ids.extend(cpu_block_ids)
                in_flight.update(gpu_block_ids)
                if merged_block_meta is not None:
                    assert block_meta is not None
                    merged_block_meta.extend(block_meta)

                # Touch GPU blocks to prevent freeing during async copy
                gpu_block_pool.touch(
                    [gpu_block_pool.blocks[bid] for bid in gpu_block_ids]
                )

                logger.debug(
                    "Request %s: Scheduling store of %d blocks to CPU (%d groups)",
                    req_id,
                    len(cpu_block_ids),
                    num_groups,
                )

            # Advance per-group cursors (includes cached hits + newly stored)
            for g in range(num_groups):
                state.num_stored_blocks[g] += advanced_per_group[g]

        # A request can contribute both boundary and positional blocks to the
        # same event. Completion tracking is set-based, so keep one request ID.
        req_ids = list(dict.fromkeys(req_ids))
        return merged_gpu_block_ids, merged_cpu_block_ids, req_ids, merged_block_meta

    # Shared with allocator/union readiness checks.
    @property
    def acked(self) -> int:
        return self.commit_acks.acked


    def _sync_tier_readiness(
        self, new_masked: set[tuple[str, int]], frontier_reported: bool = True
    ) -> None:
        """Feed rank acks into the GPU pool's content readiness (the
        ``BlockTierState``): masked dispositions first, then the frontier.
        Content a masked commit should have written is never valid, so its
        GPU blocks are un-hashed (never served, never demoted)."""
        pool = self._gpu_block_pool
        tier = pool.tier if pool is not None else None
        if tier is None:
            return
        stale: set[int] = set()
        for producer, step in new_masked:
            stale.update(tier.mark_masked(producer, step))
        if stale:
            pool.evict_blocks(stale)
        if frontier_reported:
            tier.set_acked_step(self.commit_acks.acked)

    def _cached_gpu_block(
        self, resolved_hashes: "BlockHashList", block_idx: int, group_id: int
    ) -> "KVCacheBlock | None":
        """Return the GPU block still cached under this group's block hash."""
        if block_idx >= len(resolved_hashes):
            return None
        assert self._gpu_block_pool is not None
        blocks = self._gpu_block_pool.get_cached_block(
            resolved_hashes[block_idx], [group_id]
        )
        return blocks[0] if blocks else None

    def _select_eager_blocks_to_store(
        self,
        state: StoreRequestState,
        block_ids_by_group: tuple[list[int], ...],
    ) -> tuple[
        list[int],
        list[int],
        list[dict[BlockHashWithGroupId, BlockStoreMeta]] | None,
    ]:
        """Return confirmed eager blocks, cursor advances, and event metadata."""
        assert self._gpu_block_pool is not None
        gpu_block_ids: list[int] = []
        advanced_per_group = [0] * len(self.cpu_kv_cache_config.kv_cache_groups)
        block_meta: list[dict[BlockHashWithGroupId, BlockStoreMeta]] | None = (
            [] if self.enable_kv_cache_events else None
        )
        request = state.request
        confirmed_tokens = request.num_computed_tokens - request.num_output_placeholders
        num_free = self.cpu_block_pool.get_num_free_blocks()

        for g, group_gpu_ids in enumerate(block_ids_by_group):
            if g not in self.prefix_cacheable_group_ids:
                continue
            if len(gpu_block_ids) >= num_free:
                break
            group_manager = self.cpu_coordinator.single_type_managers[g]
            if not group_manager.has_positionally_stable_blocks:
                continue
            # FIXME (yifan): handle CPU cache eviction, where
            # num_stored_blocks can be stale and omit evicted blocks in
            # the middle of the request.
            group_size = self.group_block_sizes[g]
            # Mimic the GPU KV cache manager's cacheable prefix.
            cacheable_tokens = self.cpu_coordinator.get_num_cacheable_tokens(
                confirmed_tokens, g
            )
            ready = min(len(group_gpu_ids), cacheable_tokens // group_size)
            resolved_hashes = resolve_block_hashes(
                request.block_hashes, self.hash_block_size, group_size
            )
            curr_mm_idx = 0
            secondary_mm_idx = 0
            start = state.num_stored_blocks[g]
            for i, gpu_block_id in enumerate(group_gpu_ids[start:ready], start=start):
                gpu_block = self._gpu_block_pool.blocks[gpu_block_id]
                if gpu_block.is_null:
                    # Sliding-window groups null pages that left the window
                    # before the connector sees the block table, but the
                    # retained prefix-cache tail stays hashed in the GPU free
                    # queue; store it from there.
                    recovered = self._cached_gpu_block(resolved_hashes, i, g)
                    if recovered is None:
                        advanced_per_group[g] += 1
                        continue
                    gpu_block = recovered
                    gpu_block_id = gpu_block.block_id
                if (
                    self._classify_store_candidate(gpu_block)
                    is not _StoreAdmission.READY
                ):
                    advanced_per_group[g] += 1
                    continue
                if len(gpu_block_ids) >= num_free:
                    break
                primary_block_hash = gpu_block.block_hash
                assert primary_block_hash is not None
                gpu_block_ids.append(gpu_block_id)
                advanced_per_group[g] += 1
                if block_meta is not None:
                    token_start = i * group_size
                    token_end = token_start + group_size
                    parent_hash = (
                        None
                        if i == 0
                        else maybe_convert_block_hash(resolved_hashes[i - 1])
                    )
                    lora_req = request.lora_request
                    extra_keys, curr_mm_idx = generate_block_hash_extra_keys(
                        request, token_start, token_end, curr_mm_idx
                    )
                    meta_by_hash = {
                        primary_block_hash: BlockStoreMeta(
                            token_ids=list(
                                request.all_token_ids[token_start:token_end]
                            ),
                            parent_block_hash=parent_hash,
                            lora_id=lora_req.adapter_id if lora_req else None,
                            lora_name=lora_req.name if lora_req else None,
                            extra_keys=extra_keys,
                        )
                    }
                    first_hash_idx = token_start // self.hash_block_size
                    last_hash_idx = token_end // self.hash_block_size
                    for hash_idx in range(first_hash_idx, last_hash_idx):
                        block_hash = make_block_hash_with_group_id(
                            request.block_hashes[hash_idx], g
                        )
                        if block_hash is None or block_hash == primary_block_hash:
                            continue
                        hash_start = hash_idx * self.hash_block_size
                        hash_end = hash_start + self.hash_block_size
                        secondary_extra_keys, secondary_mm_idx = (
                            generate_block_hash_extra_keys(
                                request,
                                hash_start,
                                hash_end,
                                secondary_mm_idx,
                            )
                        )
                        meta_by_hash[block_hash] = BlockStoreMeta(
                            token_ids=list(request.all_token_ids[hash_start:hash_end]),
                            parent_block_hash=(
                                None
                                if hash_idx == 0
                                else maybe_convert_block_hash(
                                    request.block_hashes[hash_idx - 1]
                                )
                            ),
                            lora_id=lora_req.adapter_id if lora_req else None,
                            lora_name=lora_req.name if lora_req else None,
                            extra_keys=secondary_extra_keys,
                            block_size=self.hash_block_size,
                        )
                    block_meta.append(meta_by_hash)

        return gpu_block_ids, advanced_per_group, block_meta

    def _classify_store_candidate(self, gpu_block: "KVCacheBlock") -> _StoreAdmission:
        if gpu_block.is_null:
            return _StoreAdmission.NULL_BLOCK
        if gpu_block.block_hash is None:
            return _StoreAdmission.NOT_HASHED
        if gpu_block.block_id in self._in_flight_store_gpu_blocks:
            return _StoreAdmission.IN_FLIGHT
        if (
            self.cpu_block_pool.cached_block_hash_to_block.get_one_block(
                gpu_block.block_hash
            )
            is not None
        ):
            return _StoreAdmission.ALREADY_CACHED
        return _StoreAdmission.READY

    def get_boundary_store_stats(self) -> BoundaryStoreStats:
        return replace(self.boundary_store_stats)

    def get_stats(self) -> SimpleCPUOffloadStats:
        """Drain per-step stats for the connector's stats hooks."""
        stats = SimpleCPUOffloadStats()

        current = self.boundary_store_stats
        for outcome, field_name in OUTCOME_TO_FIELD.items():
            delta = getattr(current, field_name) - getattr(
                self._boundary_stats_snapshot, field_name
            )
            if delta > 0:
                stats.increase_counter(MetricName.SAVE_OUTCOMES, delta, (outcome,))
        self._boundary_stats_snapshot = replace(current)

        if self._interval_load_blocks_completed:
            stats.increase_counter(
                MetricName.LOAD_BLOCKS,
                self._interval_load_blocks_completed,
            )
            self._interval_load_blocks_completed = 0

        stats.set_gauge(
            MetricName.USED_BLOCKS,
            self.num_cpu_blocks - self.cpu_block_pool.get_num_free_blocks(),
        )
        pending = sum(
            len(t.cpu_block_ids)
            for t in (
                *self._store_event_to_blocks.values(),
                *self._pending_finished_stores,
                *self._abandoned_store_event_to_blocks.values(),
            )
        )
        stats.set_gauge(MetricName.PENDING_STORE_BLOCKS, pending)
        stats.set_gauge(MetricName.INFO, 1, self._info_labelvalues)
        return stats

    def update_connector_output(self, connector_output: KVConnectorOutput) -> None:
        """Handle async transfer completions from worker.

        Load completions arrive via finished_recving (real req_ids).
        Store completions arrive via kv_connector_worker_meta as
        per-event worker counts. We accumulate across steps and process
        a store event only when all workers have reported completion.
        """
        # Per-rank load transaction reports.
        meta = connector_output.kv_connector_worker_meta
        if self.tier_runtime is not None and isinstance(
            meta, SimpleCPUOffloadWorkerMetadata
        ):
            for key, reports in meta.load_reports.items():
                self.report_load_txn(
                    key, [(r, ok, quiet) for r, (ok, quiet) in reports.items()]
                )

        # --- Load completions ---
        # A failed restore: un-key its host
        # blocks before cleanup; the scheduler recomputes the request.
        failed_recving = connector_output.failed_recving or ()
        for req_id in list(connector_output.finished_recving or []):
            # A batch of a plan restore
            # landed; the receive finishes with its last batch.
            state = self._reqs_to_load.get(req_id)
            if (
                self.tier_plans is not None
                and state is not None
                and self.tier_plans.withhold_receive(
                    req_id, state, req_id in failed_recving
                )
            ):
                connector_output.finished_recving.discard(req_id)
                continue
            # A no-DMA notice of a cancel/reset
            # with nothing in flight is no copy failure: no row is un-keyed.
            if self.tier_runtime is not None:

                if state is not None and state.notify_only:
                    # A restore that never ran: settled as cancelled in the
                    # census and the outcome ledger (idempotent after an
                    # abort's ``on_request_gone``), no host row un-keyed.
                    self.tier_runtime.ledger.restore_retired(req_id, None)
                    self._tier_load_finished(req_id, False, stale=True)
                else:
                    self._tier_load_finished(req_id, req_id in failed_recving)
            # A notice copied nothing.
            notice = state is not None and state.notify_only
            completed_blocks = self._cleanup_load_request(req_id)

            if completed_blocks and not notice:
                self._interval_load_blocks_completed += completed_blocks

        # --- Store completions ---
        if not isinstance(meta, SimpleCPUOffloadWorkerMetadata):
            return
        # Cumulative observations per rank.
        if self.tier_runtime is not None:
            self._census().worker.update(getattr(meta, "census", {}))
        # Install masked rows before readiness.
        new_masked = meta.masked_rows - self.commit_acks.masked_rows
        self.commit_acks.update(meta.committed_steps, meta.masked_rows)
        self._sync_tier_readiness(new_masked, bool(meta.committed_steps))
        # Bound the masked-row set.
        self._prune_masked_rows()

        wb = self.writeback
        # Events that only carry rank reports.
        events = dict(meta.completed_store_events)
        if wb is not None:
            for _, event_idx, _ in meta.store_reports:
                events.setdefault(event_idx, 0)
        for event_idx, count in events.items():
            # Demotion txns count ranks, not
            # reports, so a repeated report never completes a txn early.
            if wb is not None and wb.owns(event_idx):
                self._report_store_txn(event_idx, meta)
                continue
            # A settled or unknown demotion txn
            # (duplicate report) never reaches the report-count path.
            if wb is not None:
                continue
            total = self._store_event_pending_counts.get(event_idx, 0) + count
            if total >= self._expected_worker_count:
                self._store_event_pending_counts.pop(event_idx, None)
                self._process_store_event(event_idx)
            else:
                self._store_event_pending_counts[event_idx] = total


    def _report_store_txn(
        self, event_idx: int, meta: SimpleCPUOffloadWorkerMetadata
    ) -> None:
        """Apply one step's per-rank reports to a demotion transaction.

        COMMIT publishes; UNWIND (a rank failed, all quiescent) publishes
        nothing: the sources stay GPU-resident as before the demote and the
        staged host blocks are released. PENDING keeps every pin (a failed or
        unprovable rank stays quarantined until it proves quiescence or the
        age bound fails the engine stop)."""
        wb = self.writeback
        assert wb is not None
        transfer = self._store_event_to_blocks.get(
            event_idx
        ) or self._abandoned_store_event_to_blocks.get(event_idx)
        reports: list[tuple[int, bool, bool]] = []
        dropped = 0
        for (epoch, txn, _batch), ranks in meta.store_reports.items():
            if txn != event_idx:
                continue
            if transfer is not None and epoch != transfer.cache_epoch:
                # A report from another cache epoch proves nothing here.
                self.txn_report_mismatches += 1
                dropped += 1
                continue
            reports.extend((r, ok, quiet) for r, (ok, quiet) in ranks.items())
        legacy = meta.store_ack_ranks.get(event_idx)
        if legacy is not None:
            reports.extend((r, True, True) for r in legacy)
        if not reports:
            if event_idx in meta.completed_store_events and not dropped:
                raise RuntimeError(
                    f"host-RAM KV tier: store event {event_idx} completed "
                    "without rank acks"
                )
            return
        state = wb.report(event_idx, reports)
        if state is kv_host_tier_lifecycle.TxnState.COMMIT:
            self._process_store_event(event_idx)
        elif state is kv_host_tier_lifecycle.TxnState.UNWIND:
            self.store_txn_failures += 1
            wb.failed_txns += 1
            logger.warning(
                "host-RAM KV tier: demotion txn %d failed on rank(s) %s; "
                "nothing is published, sources stay GPU-resident",
                event_idx,
                sorted(r for r, ok, _ in reports if not ok),
            )
            self._process_store_event(event_idx, failed=True)


    def report_load_txn(
        self,
        key: tuple[int, int, int],
        reports: "Iterable[tuple[int, bool, bool]]",
    ) -> kv_host_tier_lifecycle.TxnState | None:
        """Per-rank reports ``(rank, success, quiescent)`` of one load
        (promotion) transaction ``key = (epoch, load event, batch)``.

        The transaction resolves when every rank reported quiescent: COMMIT
        completes its requests; UNWIND (a rank failed, or the report belongs
        to an older cache epoch) fails them so they recompute from the last
        good boundary, drops the host blocks the load read and releases the
        lease. PENDING changes nothing. Returns None for a transaction that
        already resolved (a duplicate report is a no-op)."""
        if key in self._load_resolved:
            return None
        epoch, event, batch = key
        acks = self._load_acks.get(key)
        if acks is None:
            acks = kv_host_tier_lifecycle.TxnAcks(
                epoch, event, batch, self._expected_worker_count
            )
            self._load_acks[key] = acks
        for rank, success, quiescent in reports:
            acks.report(rank, success, quiescent)
        state = acks.state
        if state is kv_host_tier_lifecycle.TxnState.PENDING:
            return state
        del self._load_acks[key]
        self._load_resolved[key] = None
        while len(self._load_resolved) > 8192:
            self._load_resolved.pop(next(iter(self._load_resolved)))
        # A transport failure is an UNWIND only. A load that succeeded for an
        # older cache epoch is invalidated (no publication; its requests take
        # the failed-receive cleanup) but is not a transport failure.
        unwound = state is kv_host_tier_lifecycle.TxnState.UNWIND
        failed = unwound or epoch != self.cache_epoch
        if unwound:
            self.load_txn_failures += 1
        elif failed:
            self.load_stale_epoch_completions += 1
        for req_id in list(self._load_event_to_reqs.get(event, ())):
            # A non-final credit batch of a
            # plan restore committed: the next batch issues, the receive goes on.
            state = self._reqs_to_load.get(req_id)
            if (
                self.tier_plans is not None
                and state is not None
                and self.tier_plans.batch_committed(req_id, state, event, failed)
            ):
                continue
            if self.tier_runtime is not None:
                self._tier_load_finished(req_id, failed, stale=failed and not unwound)
            completed = self._cleanup_load_request(req_id)
            if completed:
                self._interval_load_blocks_completed += completed
        return state


    def load_txn_key(self, req_id: str) -> tuple[int, int, int] | None:
        """The (epoch, load event, batch) key a request's issued promotion is
        acked under, or None before it was issued (or when it needs no DMA)."""
        state = self._reqs_to_load.get(req_id) or self._abandoned_reqs_to_load.get(
            req_id
        )
        if state is None or state.load_event is None or state.load_event < 0:
            return None
        return (state.transfer_meta.cache_epoch, state.load_event, 0)


    def _tier_load_finished(
        self, req_id: str, failed: bool, *, stale: bool = False
    ) -> None:
        assert self.tier_runtime is not None
        if failed:
            state = self._reqs_to_load.get(req_id) or self._abandoned_reqs_to_load.get(
                req_id
            )
            if state is not None and state.transfer_meta is not None:
                # Never publish a failed promotion.
                state.failed = True
                # A plan restore un-keys only
                # the host blocks its issued batches read.
                read = (
                    self.tier_plans.read_sources(req_id)
                    if self.tier_plans is not None
                    else None
                )
                self.tier_runtime.invalidate_failed_load(
                    req_id,
                    state.transfer_meta.cpu_block_ids if read is None else read,
                    stale=stale,
                )
        self._census().load_done(req_id, failed and not stale, cancelled=stale)
        self.tier_runtime.on_load_done(req_id)

    # ``failed``: a rank reported the copy failed.
    def _process_store_event(self, event_idx: int, failed: bool = False) -> None:
        """Process a fully-completed store event (``failed``: a rank reported
        the copy failed, so nothing is published)."""

        self._store_issued_at.pop(event_idx, None)
        transfer = self._store_event_to_blocks.pop(event_idx, None)
        # A txn from an older cache epoch never
        # publishes (reset() also moves it to the abandoned map).
        txn = self.writeback.take_txn(event_idx) if self.writeback else None
        # One stale path for both epoch fences
        # (the manager's cache epoch and the demotion txn's).
        if transfer is not None and (
            failed
            or transfer.cache_epoch != self.cache_epoch
            or (txn is not None and not self.writeback.is_current(txn))
        ):
            if failed and not self._lazy_mode:
                self._in_flight_store_gpu_blocks.difference_update(
                    transfer.gpu_block_ids
                )
            self._abandoned_store_event_to_blocks[event_idx] = transfer
            transfer = None
        if transfer is None:
            transfer = self._abandoned_store_event_to_blocks.pop(event_idx, None)
            if transfer is None:
                return  # guard stale events from before a reset() call
            # Demotion sources are released by
            # their tickets (end_demote), never twice.
            if txn is not None:
                self.cpu_block_pool.free_blocks(
                    self.cpu_block_pool.blocks[bid] for bid in transfer.cpu_block_ids
                )
                self.writeback.release_stale(txn)
                return
            self._release_transfer_refs(transfer)
            return

        if not self._lazy_mode:
            self._in_flight_store_gpu_blocks.difference_update(transfer.gpu_block_ids)

        self._process_store_completion(
            transfer.gpu_block_ids,
            transfer.cpu_block_ids,
            transfer.block_meta,

            gpu_block_keys=transfer.gpu_block_keys,
            demote_txn=txn,
        )
        logger.debug(
            "Store event %d completed: cached %d blocks to CPU",
            event_idx,
            len(transfer.cpu_block_ids),
        )

        # Eager only: update per-req state
        if not self._lazy_mode:
            for req_id in self._store_event_to_reqs.pop(event_idx, []):
                state = self._reqs_to_store.get(req_id)
                if state is None:
                    continue
                state.store_events.discard(event_idx)
                if state.finished and not state.store_events:
                    self._cleanup_store_request(req_id)

    def _process_store_completion(
        self,
        gpu_block_ids: list[int],
        cpu_block_ids: list[int],
        block_meta: list[dict[BlockHashWithGroupId, BlockStoreMeta]] | None = None,

        gpu_block_keys: list[Any] | None = None,
        demote_txn: "kv_host_tier_writeback.DemoteTxn | None" = None,
    ) -> None:
        """Register copied blocks in the CPU prefix cache and release refs."""
        assert len(cpu_block_ids) == len(gpu_block_ids)
        assert block_meta is None or len(block_meta) == len(gpu_block_ids)

        cpu_blocks = [self.cpu_block_pool.blocks[bid] for bid in cpu_block_ids]

        self.tier_stats.stored_blocks += len(cpu_blocks)

        assert self._gpu_block_pool is not None
        # Whether each source's copy was published.
        landed = [False] * len(gpu_block_ids)
        for i, (gpu_block_id, cpu_block) in enumerate(zip(gpu_block_ids, cpu_blocks)):
            gpu_block = self._gpu_block_pool.blocks[gpu_block_id]
            # A demotion publishes only the content
            # its copy read (same generation and key as at begin_demote).
            if demote_txn is not None and not self.writeback.still_valid(
                gpu_block_id
            ):
                continue
            # Register under the keys the source
            # held from issue to now, else discard (TierRuntime).
            tier_keys = None
            if gpu_block_keys is not None and self.tier_runtime is not None:
                tier_keys = self.tier_runtime.keys_to_register(
                    self._gpu_block_pool, gpu_block_id, gpu_block_keys[i]
                )
                if tier_keys is None:
                    continue

                landed[i] = True
            primary_hash = gpu_block.block_hash
            assert primary_hash is not None

            was_keyed = cpu_block.block_hash is not None
            self.cpu_block_pool._insert_block_hash(
                primary_hash,
                cpu_block,
                num_tokens=gpu_block.block_hash_num_tokens,
            )

            if self.tier_runtime is not None:
                self.tier_runtime.on_registered(cpu_block, was_keyed)
                # The host copy is of the content the demotion read (design
                # 2.2); a store outside a demotion proves no identity.
                gpu_tier = self._gpu_block_pool.tier
                if gpu_tier is not None:
                    ticket = (
                        self.writeback.in_flight.get(gpu_block_id)
                        if demote_txn is not None and self.writeback is not None
                        else None
                    )
                    gpu_tier.note_host_copy(
                        cpu_block.block_id, ticket.cid if ticket is not None else 0
                    )
            secondary_hashes = self._gpu_block_pool.cached_block_hashes_by_block.get(
                gpu_block_id, ()
            )

            if tier_keys is not None:
                secondary_hashes = tier_keys
            for block_hash in secondary_hashes:
                self.cpu_block_pool._insert_block_hash(
                    block_hash, cpu_block, num_tokens=None
                )
            # A newer generation supersedes the
            # older same-key host copies, so a restore can only
            # resolve the bytes a GPU hit would return.
            if self._gpu_block_pool.tier is not None:
                self._gpu_block_pool.tier.supersede_host_copies(
                    self.cpu_block_pool, cpu_block, (primary_hash, *secondary_hashes)
                )
            if self.enable_kv_cache_events:
                meta_by_hash = block_meta[i] if block_meta is not None else {}
                primary_group_idx = get_group_id(primary_hash)
                primary_block_size = self.group_block_sizes[primary_group_idx]
                events_to_emit: list[
                    tuple[BlockHashWithGroupId, BlockStoreMeta | None, int]
                ] = [
                    (
                        primary_hash,
                        meta_by_hash.get(primary_hash),
                        primary_block_size,
                    )
                ]
                for secondary_hash in secondary_hashes:
                    secondary_meta = meta_by_hash.get(secondary_hash)
                    events_to_emit.append(
                        (
                            secondary_hash,
                            secondary_meta,
                            secondary_meta.block_size
                            if secondary_meta is not None
                            and secondary_meta.block_size is not None
                            else 0,
                        )
                    )

                for block_hash, meta, event_block_size in events_to_emit:
                    group_idx = get_group_id(block_hash)
                    spec = self.cpu_kv_cache_config.kv_cache_groups[
                        group_idx
                    ].kv_cache_spec
                    if meta is not None and len(meta.token_ids) != event_block_size:
                        # token_ids were sliced with a different g_block_size
                        # (e.g. Mamba+DCP where capture uses spec.block_size *
                        # cp_world_size but the event emits spec.block_size * 1).
                        # Emit empty metadata until #49962 fixes the sizing.
                        token_ids = []
                        parent_block_hash = None
                        extra_keys = None
                    else:
                        token_ids = meta.token_ids if meta else []
                        parent_block_hash = meta.parent_block_hash if meta else None
                        extra_keys = meta.extra_keys if meta else None
                    self.cpu_block_pool.kv_event_queue.append(
                        BlockStored(
                            block_hashes=[
                                maybe_convert_block_hash(get_block_hash(block_hash))
                            ],
                            parent_block_hash=parent_block_hash,
                            token_ids=token_ids,
                            block_size=event_block_size,
                            lora_id=meta.lora_id if meta else None,
                            medium=self.kv_event_medium,
                            lora_name=meta.lora_name if meta else None,
                            extra_keys=to_event_extra_keys(extra_keys and [extra_keys]),
                            group_idx=group_idx,
                            kv_cache_spec_kind=get_kv_cache_spec_kind(spec).value,
                            kv_cache_spec_sliding_window=(
                                get_kv_cache_spec_sliding_window(spec)
                            ),
                            locality="LOCAL",
                        )
                    )

        # A unit still assembling keeps its landed
        # rows (one ref each) until it publishes or is lost.
        if self.tier_runtime is not None and self.tier_runtime.units is not None:
            self.tier_runtime.units.hold_unpublished(cpu_blocks)
        # Free CPU and GPU blocks' ref counts to turn them into prefix cache
        self.cpu_block_pool.free_blocks(cpu_blocks)
        # A published source is clean and goes to
        # the free-queue front (it is the next reuse); a discarded one stays
        # dirty (BlockTierState.end_demote).
        if demote_txn is not None:
            self.writeback.finish(demote_txn, landed)
        else:
            self._gpu_block_pool.free_blocks(
                self._gpu_block_pool.blocks[bid] for bid in gpu_block_ids
            )
        # A unit's last member landed: publish it
        # (a unit is a lookup source only once every member is on the
        # host and acked by every rank; until then its rows are invisible).
        if self.tier_runtime is not None and self.tier_runtime.units is not None:
            units = self.tier_runtime.units
            for uid in {u.uid for b in cpu_blocks for u in units.unit_of_block(b)}:
                units.publish_unit(uid, self._sched_step)
            units.on_host_registered(cpu_blocks)

    def _release_transfer_refs(self, transfer: TransferMeta) -> None:
        """Release transfer refs without making copied data cacheable."""
        cpu_blocks = [self.cpu_block_pool.blocks[bid] for bid in transfer.cpu_block_ids]
        self.cpu_block_pool.free_blocks(cpu_blocks)
        assert self._gpu_block_pool is not None
        self._gpu_block_pool.free_blocks(
            self._gpu_block_pool.blocks[bid] for bid in transfer.gpu_block_ids
        )

    def has_pending_stores(self) -> bool:
        """Return True if a store transfer is queued or in flight."""
        return bool(
            self._pending_finished_stores
            or self._store_event_to_blocks
            or self._abandoned_store_event_to_blocks
            # Idle and drain steps poll loads too.
            or self._reqs_to_load
            or self._abandoned_reqs_to_load
            or self._ready_promotions
        )

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        """Always returns (False, None). GPU blocks are protected by ref_cnt,
        so the scheduler can free blocks immediately."""
        req_id = request.request_id

        if self.tier_runtime is not None:
            self._census().gone(req_id)
            self.tier_runtime.on_request_gone(req_id)

        if self.writeback is not None:
            self.writeback.forget_request(req_id)

        if self.tier_plans is not None:
            self.tier_plans.forget(req_id)

        # Release any temp CPU hit pin from get_num_new_matched_tokens()
        # if request is canceled or preempted before update_state_after_alloc()
        pending = self._pending_cpu_hits.pop(req_id, None)
        if pending is not None:
            self._free_pending_cpu_hit(pending)

        # Dropping a follower never revokes DMA refs.
        emptied = []
        for key, waiters in self._promotion_waiters.items():
            if req_id in waiters:
                waiters.discard(req_id)
                if not waiters:
                    emptied.append(key)
        if emptied and self.tier_plans is not None:
            # The last follower of a finished leader is gone, so
            # the staging kept for it is cut back to the outstanding batch.
            for key in emptied:
                self.tier_plans.release_for_key(key)
        # Completion may precede request cleanup.
        ready = self._ready_promotions.get(req_id)
        if ready is not None:
            ready.finished = True
            self._retire_ready_promotions()
        # Handle load: defer cleanup if load is in-flight
        load_state = self._reqs_to_load.get(req_id)
        if load_state is not None:
            if load_state.load_event is not None:
                load_state.finished = True  # Defer: load in-flight
                # Quarantine only the
                # batch in flight; release the rest now.
                if self.tier_plans is not None:
                    self.tier_plans.release_unoutstanding(req_id, load_state)
            # A handed-off load with no batch in
            # flight (between credit batches, or waiting for credits): the
            # scheduler still waits for its receive, so notify, never drop.
            elif self._notify_unissued(req_id, load_state):
                load_state.finished = True
            else:
                self._cleanup_load_request(req_id)

        # Handle store (eager mode only): defer cleanup if stores in-flight
        if not self._lazy_mode:
            store_state = self._reqs_to_store.get(req_id)
            if store_state is not None:
                if store_state.store_events:
                    store_state.finished = True  # Defer: stores in-flight
                else:
                    self._cleanup_store_request(req_id)

        return False, None


    def _quarantined_pairs(self) -> int:
        """Pairs a DMA can still write: the outstanding batch of cancelled
        or abandoned loads. Staging a surviving follower keeps is retention,
        not quarantine."""
        total = 0
        entries = (
            *self._abandoned_reqs_to_load.items(),
            *(
                (rid, s) for rid, s in self._reqs_to_load.items()
                if s.finished and s.load_event is not None and s.load_event >= 0
            ),
        )
        for rid, s in entries:
            if s.transfer_meta is None:
                continue
            total += (
                self.tier_plans.dma_blocks(s)
                if self.tier_plans is not None
                else len(s.transfer_meta.cpu_block_ids)
            )
        return total

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:

        if self.tier_runtime is not None:
            self._census().remember(request)
        # Write-back: no finish-time flush; the
        # freed blocks are demoted from the free-queue head when they are
        # victims.
        if self.writeback is None:
            self._queue_finished_eager_store(request, block_ids)
        # The request-history ledger records the
        # finished turn before the unit does.
        if self.tier_runtime is not None:
            self.tier_runtime.on_request_finished(request)
        # The restore unit, rediscovered by key.
        unit = None
        if self.tier_runtime is not None and self.tier_runtime.units is not None:
            unit = self.tier_runtime.units.record_finished(request)
        # Lifecycle: the finished turn to the copy scheduler.
        kv_host_tier_hooks.notify_finished(self, request, unit)
        return self.request_finished(request, block_ids=[])

    def _queue_finished_eager_store(
        self, request: "Request", block_ids: tuple[list[int], ...]
    ) -> None:
        """Queue confirmed eager blocks omitted after a request enters decode."""
        state = self._reqs_to_store.get(request.request_id)
        gpu_pool = self._gpu_block_pool
        if state is None or gpu_pool is None:
            return
        gpu_ids, _, block_meta = self._select_eager_blocks_to_store(state, block_ids)
        partial_tail = self._find_fa_partial_tail_source(request, block_ids)
        if (
            partial_tail is not None
            # Decode tokens can fill the boundary block, in which case the
            # positional scan above already selected it. Appending again would
            # allocate two CPU blocks for one GPU block.
            and partial_tail not in gpu_ids
            and len(gpu_ids) < self.cpu_block_pool.get_num_free_blocks()
        ):
            gpu_ids.append(partial_tail)
            if block_meta is not None:
                # The block's own hashes are registered at completion, so no
                # capture is needed here; emit the event without a payload, as
                # the size-mismatch fallback above already does.
                block_meta.append({})
        if not gpu_ids:
            return
        cpu_blocks = self.cpu_block_pool.get_new_blocks(len(gpu_ids))
        self._pending_finished_stores.append(
            TransferMeta(gpu_ids, [b.block_id for b in cpu_blocks], block_meta)
        )
        self._in_flight_store_gpu_blocks.update(gpu_ids)
        gpu_pool.touch([gpu_pool.blocks[bid] for bid in gpu_ids])

    def _find_fa_partial_tail_source(
        self, request: "Request", block_ids: tuple[list[int], ...]
    ) -> int | None:
        """Locate the full-attention prompt-tail block for a fine-grained hit.

        Mirrors ``FullAttentionManager._cache_prompt_tail_block``: only the
        final prompt hash boundary is eligible, and boundaries that land on a
        physical block edge are already covered by the positional scan.

        Attention block tables are append-only, so the block is located
        positionally, as the mooncake store and offloading connector also do;
        no new core hand-off is required. The boundary key is registered at
        completion along with the block's other hashes, so nothing has to be
        captured here.
        """
        if not self.cpu_coordinator.enable_partial_hash_hits:
            return None
        assert self._gpu_block_pool is not None
        boundary_tokens = (
            request.num_prompt_tokens // self.hash_block_size * self.hash_block_size
        )
        if boundary_tokens == 0 or boundary_tokens > request.num_computed_tokens:
            return None
        if boundary_tokens % self.fa_block_size == 0:
            return None
        block_idx = boundary_tokens // self.fa_block_size
        fa_block_ids = block_ids[self.fa_gidx]
        if block_idx >= len(fa_block_ids):
            return None
        gpu_block_id = fa_block_ids[block_idx]
        gpu_block = self._gpu_block_pool.blocks[gpu_block_id]
        if gpu_block.is_null or gpu_block_id in self._in_flight_store_gpu_blocks:
            return None
        hash_idx = boundary_tokens // self.hash_block_size - 1
        if hash_idx >= len(request.block_hashes):
            return None
        block_hash = make_block_hash_with_group_id(
            request.block_hashes[hash_idx], self.fa_gidx
        )
        # Only worth copying when the boundary key is actually registered on
        # the source block; completion replays the block's own hashes.
        if (
            gpu_block.block_hash != block_hash
            and block_hash
            not in self._gpu_block_pool.cached_block_hashes_by_block.get(
                gpu_block_id, ()
            )
        ):
            return None
        if self.cpu_block_pool.cached_block_hash_to_block.get_one_block(block_hash):
            return None
        return gpu_block_id

    def _free_pending_cpu_hit(
        self, pending: tuple[tuple[list["KVCacheBlock"], ...], int, int]
    ) -> None:
        """Release the temporary CPU block pin taken in get_num_new_matched_tokens()."""
        cpu_hit_blocks, _hit_length, _num_computed_tokens = pending
        blocks_to_free = [
            blk for grp in cpu_hit_blocks for blk in grp if not blk.is_null
        ]
        if blocks_to_free:
            self.cpu_block_pool.free_blocks(blocks_to_free)


    def _notify_unissued(self, req_id: str, state: LoadRequestState) -> bool:
        """``state`` has no load in flight. A plan restore (handed off to the
        scheduler, which waits in WAITING_FOR_REMOTE_KVS for its receive) or
        an unpublished failed load keeps its state: nothing more is issued
        and the next ``build_connector_meta`` publishes one no-DMA receive.
        Its manifest stays leased until that receive retires it. False:
        not such a load (the caller drops it, as before)."""
        plan = self.tier_plans is not None and self.tier_plans.is_plan_load(req_id)
        if not (plan or state.failed):
            return False
        if plan:
            self.tier_plans.release_unoutstanding(req_id, state, keep_complete=False)
        if not state.failed:
            state.notify_only = True
        state.issued_at = None  # nothing is in flight
        return True

    def _cleanup_load_request(self, req_id: str) -> int:
        """Release all load resources for a request.

        Shared between request_finished() and update_connector_output() paths.
        Removes the request from _reqs_to_load, cleans up event mappings,
        and frees CPU/GPU touch refs.

        Returns the number of blocks in the load if it had been issued to
        the worker, else 0 (never-issued loads are not counted as completed
        by the caller).
        """
        state = self._reqs_to_load.pop(req_id, None)
        if state is None:
            state = self._abandoned_reqs_to_load.pop(req_id, None)
        if state is None:
            return 0
        # Publish before the refs drop.
        if self.tier_plans is not None:
            self.tier_plans.on_load_retired(req_id, state)
        # Remove from load event mapping (only this req, not whole event)
        if state.load_event is not None:
            reqs = self._load_event_to_reqs.get(state.load_event)
            if reqs is not None:
                with contextlib.suppress(ValueError):
                    reqs.remove(req_id)
                if not reqs:
                    self._load_event_to_reqs.pop(state.load_event, None)

        # Keep a completed immutable promotion
        # leased until normal cache publication, or publish it after cancellation.
        if (
            state.lease is not None
            and not state.failed
            and state.lease.epoch == self.cache_epoch
            and state.load_event is not None
            and state.load_event >= 0
            and self._promotion_waiters.get(state.lease.key)
        ):
            self._ready_promotions[req_id] = state
            self._retire_ready_promotions()
            return len(state.transfer_meta.gpu_block_ids)
        if state.lease is not None:
            self._promotions.pop(state.lease.key, None)
            self._promotion_waiters.pop(state.lease.key, None)
        if state.transfer_meta is not None:
            # Free CPU touch refs
            self.cpu_block_pool.free_blocks(
                self.cpu_block_pool.blocks[bid]
                for bid in state.transfer_meta.cpu_block_ids
            )
            # Soft promotion (units.on_host_released).
            if self.tier_runtime is not None and self.tier_runtime.units is not None:
                self.tier_runtime.units.on_host_released(
                    state.transfer_meta.cpu_block_ids
                )
            # Free GPU touch refs
            assert self._gpu_block_pool is not None
            self._gpu_block_pool.free_blocks(
                self._gpu_block_pool.blocks[bid]
                for bid in state.transfer_meta.gpu_block_ids
            )

        if state.load_event is None:
            return 0
        return len(state.transfer_meta.gpu_block_ids)

    def _cleanup_store_request(self, req_id: str) -> None:
        """Release store metadata for a request.

        Metadata-only cleanup but no block freeing. Job completion handles
        block caching and GPU ref freeing via _process_store_completion().
        """
        state = self._reqs_to_store.pop(req_id, None)
        if state is None:
            return
        for event_idx in list(state.store_events):
            if (reqs := self._store_event_to_reqs.get(event_idx)) is not None:
                with contextlib.suppress(ValueError):
                    reqs.remove(req_id)
                if not reqs:
                    self._store_event_to_reqs.pop(event_idx, None)
        state.store_events.clear()

    def take_events(self) -> Iterable[KVCacheEvent]:
        events = self.cpu_block_pool.take_events()
        for event in events:
            if isinstance(event, BlockRemoved):
                event.medium = self.kv_event_medium
                event.locality = "LOCAL"
        return events

    def reset(self) -> bool:
        """Abandon pending transfers and reset the CPU cache when safe.

        Worker-side DMA may still be using blocks after reset is requested.
        Keep those block refs pinned until the existing completion path reports
        the transfer finished, then release refs without caching abandoned
        store results.
        """
        # Visibility changes before physical reuse.
        self.cache_epoch += 1
        self._promotions.clear()
        self._promotion_waiters.clear()

        if self.tier_plans is not None:
            self.tier_plans.reset()
        for state in self._ready_promotions.values():
            self._release_transfer_refs(state.transfer_meta)
        self._ready_promotions.clear()
        if self.tier_runtime is not None:
            self.tier_runtime.unkey_all_for_reset()
        else:
            self.cpu_block_pool.evict_blocks({
                block.block_id for block in self.cpu_block_pool.blocks
                if block.block_hash is not None
            })
        for pending in self._pending_cpu_hits.values():
            self._free_pending_cpu_hit(pending)
        self._pending_cpu_hits.clear()

        self._abandoned_store_event_to_blocks.update(self._store_event_to_blocks)
        # New epoch: in-flight demotions drain
        # without publishing; their sources stay pinned until the DMA is done.
        if self.writeback is not None:
            self.writeback.reset()
        for transfer in self._pending_finished_stores:
            self._release_transfer_refs(transfer)
        self._pending_finished_stores.clear()

        self._store_event_to_blocks.clear()
        self._in_flight_store_gpu_blocks.clear()

        # Loads that have not been sent to the worker cannot have running DMA.
        # In-flight loads stay pinned and are cleaned up on completion.
        for req_id in list(self._reqs_to_load):
            state = self._reqs_to_load.pop(req_id)
            if state.load_event is None:
                self._reqs_to_load[req_id] = state
                # A waiting plan restore is
                # notified (the scheduler recomputes it), never dropped.
                if not self._notify_unissued(req_id, state):
                    self._cleanup_load_request(req_id)
            else:
                self._abandoned_reqs_to_load[req_id] = state
                # Only the batch in flight
                # stays quarantined.
                if self.tier_plans is not None:
                    self.tier_plans.release_unoutstanding(req_id, state)

        self._reqs_to_store.clear()
        self._store_event_to_reqs.clear()
        self._store_event_pending_counts = {
            event_idx: count
            for event_idx, count in self._store_event_pending_counts.items()
            if event_idx in self._abandoned_store_event_to_blocks
        }
        self._cursor = None
        # Seed the fresh counters with outcomes since the last drain so a
        # mid-interval reset() does not drop them from the next report.
        undrained = BoundaryStoreStats(
            **{
                f.name: max(
                    0,
                    getattr(self.boundary_store_stats, f.name)
                    - getattr(self._boundary_stats_snapshot, f.name),
                )
                for f in fields(BoundaryStoreStats)
            }
        )
        self.boundary_store_stats = undrained
        self._boundary_stats_snapshot = BoundaryStoreStats()

        self.tier_stats = TierStats()
        self._tier_log_logged = None
        self._wb_log_logged = None
        if self.tier_runtime is not None:
            self._census().reset()
        # NOTE: _load_event_counter / _store_event_counter are not
        # reset as they are monotonic and must stay ahead of the workers
        # high-water marks to avoid event index collisions

        if self._abandoned_store_event_to_blocks or self._abandoned_reqs_to_load:
            return False

        # No host copy survives; GPU backing is
        # revalidated on every query, so the twins are dirty again as is.
        return self.cpu_block_pool.reset_prefix_cache()
