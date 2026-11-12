# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modified by the Merlin project; see NOTICE.

import itertools
from collections.abc import Sequence
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, overload

from vllm.distributed.kv_events import BlockStored, KVCacheEvent
from vllm.logger import init_logger
from vllm.merlin.kv_tier_lookup import (
    KeyLog,
    adopted_draws,
)
from vllm.utils.math_utils import cdiv
from vllm.v1.core.cache_thrash import SessionYield
from vllm.v1.core.kv_cache_coordinator import (
    HybridKVCacheCoordinator,
    get_kv_cache_coordinator,
)
from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector
from vllm.v1.core.kv_cache_utils import KVCacheBlock, KVCacheBlockCopy, trim_null_head
from vllm.v1.core.single_type_kv_cache_manager import MambaManager
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    CrossAttentionSpec,
    EncoderOnlyAttentionSpec,
    KVCacheConfig,
    MambaSpec,
    get_kv_cache_spec_kind,
    get_kv_cache_spec_sliding_window,
)
from vllm.v1.metrics.stats import PrefixCacheStats
from vllm.v1.request import Request, RequestStatus

logger = init_logger(__name__)


@dataclass
class KVCacheBlocks:
    """The allocation result of KVCacheManager, work as the interface between
    Scheduler and KVCacheManager, to hide KVCacheManager's internal data
    structure from the Scheduler.
    """

    blocks: tuple[Sequence[KVCacheBlock], ...]
    """
    `blocks[i][j]` refers to the i-th kv_cache_group
    and the j-th block of tokens.We don't use block of
    tokens as the outer dimension because it assumes all
    kv_cache_groups have the same number of blocks, which is true for now but
    will be broken if we want to give different block_size to different
    kv_cache_groups in the future.

    Each single type KVCacheBlocks could be represented as:
    - list[KVCacheBlock] for more than one KVCacheBlock
    - an empty tuple for requests without KVCacheBlock
      (a precomputed KVCacheBlocks is in KVCacheManager to avoid GC overhead)
    """

    def __add__(self, other: "KVCacheBlocks") -> "KVCacheBlocks":
        """Adds two KVCacheBlocks instances."""
        return KVCacheBlocks(
            tuple(
                list(itertools.chain(blk1, blk2))
                for blk1, blk2 in zip(self.blocks, other.blocks)
            )
        )

    @overload
    def get_block_ids(
        self,
        allow_none: Literal[False] = False,
        *,
        group_ids: Sequence[int] | None = None,
    ) -> tuple[list[int], ...]: ...

    @overload
    def get_block_ids(
        self,
        allow_none: Literal[True] = True,
        *,
        group_ids: Sequence[int] | None = None,
    ) -> tuple[list[int], ...] | None: ...

    def get_block_ids(
        self,
        allow_none: bool = False,
        *,
        group_ids: Sequence[int] | None = None,
    ) -> tuple[list[int], ...] | None:
        """Converts the KVCacheBlocks instance to block_ids.

        Args:
            allow_none: Return None when every selected group is empty.
            group_ids: KV cache groups to include. Includes all groups by default.

        Returns:
            tuple[list[int], ...]: A tuple of lists where:
                - the outer tuple corresponds to KV cache groups
                - each inner list contains the block_ids of the blocks in that
                  group

        """
        groups = (
            self.blocks
            if group_ids is None
            else tuple(self.blocks[group_id] for group_id in group_ids)
        )
        if allow_none and all(len(group) == 0 for group in groups):
            return None
        return tuple([blk.block_id for blk in group] for group in groups)

    def get_unhashed_block_ids(self) -> list[int]:
        """Get block_ids of unhashed blocks from KVCacheBlocks instance."""
        assert len(self.blocks) == 1, "Only one group is supported"
        return [block.block_id for block in self.blocks[0] if block.block_hash is None]

    def get_unhashed_block_ids_all_groups(self) -> list[list[int]]:
        """Get block_ids of unhashed blocks from KVCacheBlocks instance."""
        # Skip padding blocks.
        return [
            [
                block.block_id
                for block in trim_null_head(group)
                if block.block_hash is None and not block.is_null
            ]
            for group in self.blocks
        ]

    def new_empty(self) -> "KVCacheBlocks":
        """Creates a new KVCacheBlocks instance with no blocks."""
        return KVCacheBlocks(tuple(() for _ in range(len(self.blocks))))


class KVCacheManager:
    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        scheduler_block_size: int,
        hash_block_size: int,
        max_in_flight_tokens: int | None = None,
        enable_caching: bool = True,
        use_eagle: bool = False,
        num_hash_lookahead_tokens: int = 0,
        num_prefill_lookahead: int = 0,
        log_stats: bool = False,
        enable_kv_cache_events: bool = False,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
        metrics_collector: KVCacheMetricsCollector | None = None,
        watermark: float = 0.0,
        enable_mamba_shared_prefix_checkpoint: bool = False,
        *,
        causal_sliding_window_kv: bool,
    ) -> None:
        self.max_model_len = max_model_len
        # When unset, fall back to `max_model_len` so the recycling-aware cap
        # collapses to the prior (uncapped) admission behavior. The scheduler
        # always supplies the real value at runtime.
        if max_in_flight_tokens is None:
            max_in_flight_tokens = max_model_len

        self.enable_caching = enable_caching
        self.enable_kv_cache_events = enable_kv_cache_events
        self.use_eagle = use_eagle
        self.log_stats = log_stats
        self.metrics_collector = metrics_collector
        # FIXME: make prefix cache stats conditional on log_stats. We still need
        # this comment because when the log stats is enabled there are still
        # potential configs we could expose in the future.
        self.prefix_cache_stats = PrefixCacheStats() if log_stats else None

        self.coordinator = get_kv_cache_coordinator(
            kv_cache_config=kv_cache_config,
            max_model_len=self.max_model_len,
            max_in_flight_tokens=max_in_flight_tokens,
            use_eagle=self.use_eagle,
            num_hash_lookahead_tokens=num_hash_lookahead_tokens,
            enable_caching=self.enable_caching,
            enable_kv_cache_events=enable_kv_cache_events,
            dcp_world_size=dcp_world_size,
            pcp_world_size=pcp_world_size,
            scheduler_block_size=scheduler_block_size,
            hash_block_size=hash_block_size,
            metrics_collector=self.metrics_collector,
            num_prefill_lookahead=num_prefill_lookahead,
            causal_sliding_window_kv=causal_sliding_window_kv,
        )
        # One predicate, read by both sides of the feature, so the scheduler
        # cannot end a chunk at a junction the manager would refuse -- a refused
        # junction costs a forward pass and displaces the block-boundary stop.
        # Multi-module MTP is excluded because ``cache_blocks`` then hands the
        # manager ``num_computed - num_reprefillable`` rather than the chunk end.
        self.mamba_shared_prefix_checkpoint = (
            enable_mamba_shared_prefix_checkpoint
            and bool(self.coordinator.eagle_group_ids)
            and self.coordinator.enable_partial_hash_hits
            and self.coordinator.num_reprefillable_tokens == 0
        )
        if self.mamba_shared_prefix_checkpoint:
            for manager in self.coordinator.single_type_managers:
                if isinstance(manager, MambaManager):
                    manager.shared_prefix_checkpoint = True
        self.num_kv_cache_groups = len(kv_cache_config.kv_cache_groups)
        self.block_pool = self.coordinator.block_pool
        self.retained_hit_group_ids = tuple(
            manager.kv_cache_group_id
            for manager in self.coordinator.single_type_managers
            if manager.retains_longer_hit
        )
        self.kv_cache_config = kv_cache_config

        # Watermark: minimum number of KV cache blocks to keep free when
        # admitting waiting/preempted requests, to avoid frequent preemptions.
        assert watermark >= 0.0, "watermark must be non-negative"
        self.watermark_blocks = int(watermark * kv_cache_config.num_blocks)
        self.kv_cache_event_metadata = tuple(
            (
                get_kv_cache_spec_kind(group.kv_cache_spec).value,
                get_kv_cache_spec_sliding_window(group.kv_cache_spec),
            )
            for group in kv_cache_config.kv_cache_groups
        )

        # Pre-constructed KVCacheBlocks with no blocks, callers should use this
        # via create_kv_cache_blocks instead of creating new ones to avoid GC
        # overhead.
        #
        # We use nested tuples to ensure the empty KVCacheBlocks is immutable.
        self.empty_kv_cache_blocks = KVCacheBlocks(
            tuple(() for _ in range(self.num_kv_cache_groups))
        )

        self._pinned_hits: dict[str, list[KVCacheBlock]] = {}
        # A waiter whose pin lookup found nothing is not looked up again
        # until a key of its own is inserted: request id ->
        # (signature, key-log position, own hashes or None until needed).
        self._pin_misses: dict[str, tuple] = {}
        self._step_key_mark = 0
        self.thrash = SessionYield(self)
        self._grace_pins: dict[str, list[KVCacheBlock]] = {}
        self._grace_block_ids: dict[str, frozenset[int]] = {}
        self._grace_prompt_tokens: dict[str, int] = {}
        # Lifecycle hooks (installed by the tier; None = no-op)
        self.tier_hooks = None
        self.enable_grace_pins = (
            self.enable_caching and self.coordinator.enable_partial_hash_hits
        )
        # Shortfall (blocks) of the last allocate_slots that returned None.
        self.last_alloc_shortfall = 0
        # Each check the last
        # allocation failed, as (shortfall, reservation it carried).
        self.last_alloc_checks: tuple[tuple[int, int], ...] = ()
        # Whether the last allocate_slots was refused by its evicts_needed check.
        self.last_alloc_evicts_needed = False
        # Write-back admission gate, installed by
        # the host-RAM KV tier at connector bind (vllm/merlin/
        # kv_host_tier_writeback.py WriteBack.admit): (request, fresh blocks,
        # real shortfall, adopted blocks) -> (shortfall, demotion_wait).
        self.admission_gate: (
            Callable[[Request, int, int, Sequence], tuple[int, bool]] | None
        ) = None
        # Exact H_run (WriteBack.account_running),
        # called by the scheduler once per step after allocation.
        self.running_headroom_hook: Callable[..., int] | None = None
        # A waiting/preempted admission
        # refused before the gate (non-coherent full-sequence check) still
        # names its need to the write-back (WriteBack.note_head): (request,
        # full-sequence blocks, shortfall) -> None.
        self.head_need_hook: Callable[[Request, int, int], None] | None = None
        # Per waiting request, the host
        # slots (group, table index) of its admitted PrefixHitPlan; set by the
        # host tier for one admission attempt, consumed by allocate_slots.
        self.restore_slots: dict[str, tuple[tuple[int, int], ...]] = {}
        # Per such request, the private
        # continuation blocks its restored recurrent/partial tails need: drawn
        # here with the destinations, inside the admission gate, and
        # taken by the host tier at the hand-off.
        self.restore_tails: dict[str, int] = {}
        self.restore_tail_blocks: dict[str, list[KVCacheBlock]] = {}
        # Whether the last allocate_slots returned None to wait for demotions
        # (no blocks short, so nothing is yielded).
        self.last_alloc_demotion_wait = False
        # A producer that moves its live GDN tail's hash to a CoW copy keeps
        # writing the source; pins and graces follow the hash to the copy.
        for manager in self.coordinator.single_type_managers:
            if isinstance(manager, MambaManager):
                manager.moved_hash_listeners.append(self._pins_follow_moved_hash)

    @property
    def usage(self) -> float:
        """Get the KV cache usage.

        Returns:
            The KV cache usage (between 0.0 and 1.0).

        """
        return self.block_pool.get_usage()

    def make_prefix_cache_stats(self) -> PrefixCacheStats | None:
        """Get (and reset) the prefix cache stats.

        Returns:
            The current prefix caching stats, or None if logging is disabled.

        """
        if not self.log_stats:
            return None
        stats = self.prefix_cache_stats
        self.prefix_cache_stats = PrefixCacheStats()
        return stats

    def prefix_cache_lookup_enabled(self, request: Request) -> bool:
        """Whether a local prefix cache lookup may be run for this request."""
        return self.enable_caching and not request.skip_reading_prefix_cache

    def record_prefix_cache_stats(self, request: Request, num_hits: int) -> None:
        # Don't count a request that skipped the cache lookup.
        if not self.log_stats or not self.prefix_cache_lookup_enabled(request):
            return
        assert self.prefix_cache_stats is not None
        self.prefix_cache_stats.record(
            num_tokens=request.num_tokens,
            num_hits=num_hits,
            preempted=request.num_preemptions > 0,
        )

    def get_computed_blocks(
        self, request: Request, max_cache_hit_length: int | None = None
    ) -> tuple[KVCacheBlocks, int, int]:
        """Get the computed (cached) blocks for the request.
        Note that the computed blocks must be full.

        Args:
            request: The request to get the computed blocks.
            max_cache_hit_length: Upper bound on the hit length. Defaults to
                ``request.num_tokens - 1``.

        Returns:
            A tuple containing:
                - A list of blocks that are computed for the request.
                - The number of computed tokens.
                - ``shared_prefix_boundary``: the block-aligned token position of
                  a shared prefix that a sparse-retention group (Mamba / sliding
                  window) has not cached yet (Marconi-style APC), or 0 if none.
                  Pinned so sparse prefix-cache retention does not drop
                  the junction and defeat cross-request reuse.

        """
        # We skip finding the prefix cache hit when prefix caching is
        # disabled or the request is marked as skipping kv cache read
        # (which happens when the request requires prompt logprobs
        # or calls a pooling model with all pooling).
        if not self.prefix_cache_lookup_enabled(request):
            return self.empty_kv_cache_blocks, 0, 0

        # NOTE: When all tokens hit the cache, we must recompute the last token
        # to obtain logits. Thus, set max_cache_hit_length to prompt_length - 1.
        # This can trigger recomputation of an entire block, rather than just
        # the single last token, because allocate_slots() requires
        # num_computed_tokens to be block-size aligned. Removing this limitation
        # could slightly improve performance in the future.
        if max_cache_hit_length is None:
            max_cache_hit_length = request.num_tokens - 1
        computed_blocks, num_new_computed_tokens, num_uncached = (
            self.coordinator.find_longest_cache_hit(
                request.block_hashes, max_cache_hit_length
            )
        )

        # When kv_cache_report_mode is "full", emit BlockStored events
        # for the reused prefix cache blocks so that external consumers
        # (e.g. gateway) can learn about them.
        if (
            num_new_computed_tokens > 0
            and self.enable_kv_cache_events
            and getattr(request, "kv_cache_report_mode", "incremental") == "full"
        ):
            self.coordinator.emit_cached_block_events(
                request, computed_blocks, num_new_computed_tokens
            )

        # The junction to pin is where the lagging sparse-retention group stops
        # (``num_new_computed_tokens``) plus the uncached shared prefix -- i.e.
        # the longest single-group hit. Sub-block gaps are left to the mask,
        # which floors to the alignment boundary (a no-op there).
        shared_prefix_boundary = (
            num_new_computed_tokens + num_uncached if num_uncached else 0
        )

        blocks = self.create_kv_cache_blocks(computed_blocks)
        return blocks, num_new_computed_tokens, shared_prefix_boundary

    def get_computed_blocks_for_connector(
        self, request: Request, max_cache_hit_length: int | None = None
    ) -> tuple[KVCacheBlocks, int, int, bool]:
        """Local prefix-cache lookup for a request scheduled with a KV connector.

        Hybrid (Mamba + full-attention) models can have per-group prefix hits
        diverge under block pressure: the full-attention tail may be evicted
        while a deeper Mamba state survives, or vice versa. Report the
        full-attention hit as the local prefix - the connector transfers the
        remaining suffix and the Mamba state is transferred unconditionally by
        nixl's ``_apply_prefix_caching`` - and flag when that hit ran deeper
        than a lagging group. Such a hit only has a valid Mamba state at its
        boundary if the connector supplies it, so the caller must fall back to
        ``get_computed_blocks`` to reconcile when no external tokens are found.

        Non-hybrid models and already-convergent hits use ``get_computed_blocks``.

        Returns:
            The ``get_computed_blocks`` triple (blocks, number of local computed
            tokens, shared-prefix boundary) plus ``hit_diverged``.

        """
        coordinator = self.coordinator
        if not (
            self.kv_cache_config.has_mamba_layers
            and isinstance(coordinator, HybridKVCacheCoordinator)
            and coordinator.full_attention_group_id is not None
        ):
            return *self.get_computed_blocks(request, max_cache_hit_length), False

        if not self.prefix_cache_lookup_enabled(request):
            return self.empty_kv_cache_blocks, 0, 0, False

        if max_cache_hit_length is None:
            max_cache_hit_length = request.num_tokens - 1
        fa_group_id = coordinator.full_attention_group_id
        computed, per_group_hits = coordinator.find_longest_cache_hit_per_group(
            request.block_hashes, max_cache_hit_length
        )
        num_local = per_group_hits[fa_group_id]
        if any(hit > num_local for hit in per_group_hits) or any(
            hit < num_local
            for hit, group in zip(per_group_hits, self.kv_cache_config.kv_cache_groups)
            if group.kv_cache_spec.prefix_cacheable
            and not isinstance(group.kv_cache_spec, MambaSpec)
        ):
            # A lagging group hit deeper than full attention means its
            # full-attention blocks were evicted; use the reconciled boundary
            # that every group agrees on. Only a Mamba group may lag behind
            # it: the connector restores that state, but not the window of
            # a sliding-window group, whose table would also be too short
            # for `truncate_computed_blocks`.
            return *self.get_computed_blocks(request, max_cache_hit_length), False

        blocks = self.create_kv_cache_blocks(computed)
        # Per-group lookups do not detect an uncached shared prefix (boundary 0).
        return blocks, num_local, 0, min(per_group_hits) < num_local

    def allocate_slots(
        self,
        request: Request,
        num_new_tokens: int,
        num_new_computed_tokens: int = 0,
        new_computed_blocks: KVCacheBlocks | None = None,
        num_lookahead_tokens: int = 0,
        num_external_computed_tokens: int = 0,
        delay_cache_blocks: bool = False,
        num_encoder_tokens: int = 0,
        full_sequence_must_fit: bool = False,
        reserved_blocks: int = 0,
        has_scheduled_reqs: bool = True,
        skip_zeroing_group_ids: tuple[int, ...] = (),
        # Young-miss eviction check (cache_thrash.Admission)
        evicts_needed: Callable[[int, tuple], bool] | None = None,
        full_sequence_reserved_blocks: int | None = None,
        full_sequence_lookahead_tokens: int | None = None,
    ) -> KVCacheBlocks | None:
        """Add slots for a request with new tokens to append.

        Args:
            request: The request to allocate slots.
            num_new_tokens: The number of new tokens to be allocated and computed.
            num_new_computed_tokens: The number of new computed tokens just
                hitting the prefix caching, excluding external tokens.
            new_computed_blocks: The cached blocks for the above new computed
                tokens, grouped as a tuple by kv cache groups.
            num_lookahead_tokens: The number of speculative tokens to allocate.
                This is used by spec decode proposers with kv-cache such
                as eagle.
            num_external_computed_tokens: The number of tokens that their
                KV caches are not cached by vLLM but cached by the connector.
            delay_cache_blocks: Whether to skip caching the blocks. This is
                used by P/D when allocating blocks used in a KV transfer
                which will complete in a future step.
            num_encoder_tokens: The number of encoder tokens to allocate for
                cross-attention in encoder-decoder models(e.g., Whisper).
                For decoder-only models, this should be 0.
            full_sequence_must_fit: Only allocate blocks if the KV cache has enough
                free blocks to hold the full sequence, accounting for prefix cache hits
                and sliding window. Used as an admission gate to prevent over-admitting
                requests when chunked prefill would otherwise only check the first chunk
            reserved_blocks: Number of free blocks that must be left available for
                other in-flight sequences to complete. The actual allocation is only
                made if it fits within
                (free - reserved_blocks), in the full-sequence check too unless
                ``full_sequence_reserved_blocks`` is given for that check. A
                check that draws no block passes whatever is reserved.
                ``full_sequence_lookahead_tokens``, if given, replaces
                ``num_lookahead_tokens`` in the full-sequence check only.
                ``last_alloc_checks`` records each failing check's shortfall
                and the reservation it carried.
            has_scheduled_reqs: Whether any requests are already scheduled to run
                this step, controls whether watermark is applied.
            skip_zeroing_group_ids: Groups whose external-token blocks will be
                written by an async load and must not be zeroed concurrently.

        Blocks layout:
        ```
        ----------------------------------------------------------------------
        | < comp > | < new_comp > | < ext_comp >  | < new >  | < lookahead > |
        ----------------------------------------------------------------------
                                                  |   < to be computed >     |
        ----------------------------------------------------------------------
                                  |            < to be allocated >           |
        ----------------------------------------------------------------------
                                  | < to be cached (roughly, |
                                  | details below)>          |
        ----------------------------------------------------------------------
        | Prefix-cached tokens from either vLLM   |
        | or connector. Can be safely removed if  |
        | they are outside sliding window.        |
        ----------------------------------------------------------------------
        |   < cached by vLLM >    | not cached by |
                                  | vLLM, but     |
        | ref_cnt  | ref_cnt not  | cached by     |
        | increased| increased yet| connector     |
        ----------------------------------------------------------------------
        ```

        Abbreviations:

        ```
        comp      = request.num_computed_tokens
        new_comp  = num_new_computed_tokens
                  = len(new_computed_blocks) * block_size
        ext_comp  = num_external_computed_tokens, cached by the connector
        new       = num_new_tokens, including unverified draft tokens
        lookahead = num_lookahead_tokens
        ```

        NOTE: for new tokens which include both verified and unverified draft
        tokens, we only cache the verified tokens (by capping the number at
        `request.num_tokens`).

        The allocation has three stages:
        - Free unnecessary blocks in `comp` and check
           if we have sufficient free blocks (return None if not).
        - Handle prefix tokens (`comp + new_comp + ext_comp`):
            - Free unnecessary blocks (e.g. outside sliding window)
            - Allocate new blocks for `ext_comp` tokens inside
              sliding window
        - Allocate new blocks for tokens to be computed (`new + lookahead`)

        Returns:
            A list of new allocated blocks.

        """
        # A step may need no slots of its own while still adopting computed
        # tokens: an async KV load, or a chunk that ends inside the replayed
        # range of a hit (SWA bounded replay).
        if (
            num_new_tokens == 0
            and num_external_computed_tokens == 0
            and num_new_computed_tokens == 0
        ):
            raise ValueError(
                "num_new_tokens must be greater than 0 when there are no "
                "computed tokens to adopt"
            )

        if new_computed_blocks is not None:
            new_computed_block_list = new_computed_blocks.blocks
            if self.retained_hit_group_ids:
                # Only retain the prefix covered by the local and external hits.
                reused = num_new_computed_tokens + num_external_computed_tokens
                groups = list(new_computed_block_list)
                for group_id in self.retained_hit_group_ids:
                    manager = self.coordinator.single_type_managers[group_id]
                    groups[group_id] = groups[group_id][
                        : cdiv(reused, manager.block_size)
                    ]
                new_computed_block_list = tuple(groups)
        else:
            new_computed_block_list = self.empty_kv_cache_blocks.blocks

        # The number of computed tokens is the number of computed tokens plus
        # the new prefix caching hits
        num_local_computed_tokens = (
            request.num_computed_tokens + num_new_computed_tokens
        )
        total_computed_tokens = min(
            num_local_computed_tokens + num_external_computed_tokens,
            self.max_model_len,
        )

        self.last_alloc_checks = ()
        full_reserved_blocks = (
            reserved_blocks
            if full_sequence_reserved_blocks is None
            else full_sequence_reserved_blocks
        )
        watermark_blocks = 0
        # The watermark is applied to waiting/preempted requests only, and only
        # when there's at least one request already scheduled.
        if has_scheduled_reqs and request.status in (
            RequestStatus.WAITING,
            RequestStatus.PREEMPTED,
        ):
            watermark_blocks = self.watermark_blocks
        # Each host slot of a mixed two-tier
        # hit draws one private destination, counted with the fixed reserve.
        restore_slots = (
            self.restore_slots.get(request.request_id, ())
            if new_computed_blocks is not None
            else ()
        )
        watermark_blocks += len(restore_slots)
        restore_tails = (
            self.restore_tails.get(request.request_id, 0) if restore_slots else 0
        )
        watermark_blocks += restore_tails

        # Matches the scheduler's own prefill boundary: `num_tokens - 1`
        # extends it to resumed requests replaying their output tokens.
        prefill_end = max(request.num_prompt_tokens, request.num_tokens - 1)

        full_shortfall = 0
        if full_sequence_must_fit:
            # First check and fail if the full request sequence won't fit.
            full_num_tokens = min(request.num_tokens, self.max_model_len)
            # Count the spec-decode lookahead slots past the prompt as the
            # chunk gate below does: without them a prompt that fits only
            # without them is admitted, and its last chunk can never allocate.
            full_num_slot_tokens = min(
                full_num_tokens
                + (
                    num_lookahead_tokens
                    if full_sequence_lookahead_tokens is None
                    else full_sequence_lookahead_tokens
                ),
                self.max_model_len,
            )

            num_blocks_to_allocate = self.coordinator.get_num_blocks_to_allocate(
                request_id=request.request_id,
                num_tokens=full_num_slot_tokens,
                new_computed_blocks=new_computed_block_list,
                num_encoder_tokens=num_encoder_tokens,
                total_computed_tokens=total_computed_tokens,
                num_local_computed_tokens=num_local_computed_tokens,
                num_tokens_main_model=full_num_tokens,
                apply_admission_cap=True,
                prefill_end=prefill_end,
            )
            required_blocks = num_blocks_to_allocate + watermark_blocks
            # The full prompt must fit
            # alongside other requests' reserved tails.
            available_blocks = self.block_pool.get_num_free_blocks() - (
                full_reserved_blocks if required_blocks else 0
            )
            if required_blocks > available_blocks:
                # A non-coherent refusal
                # still names the waiting head to the write-back (its head
                # floor and shortfall wait: head_need_hook -> WriteBack.note_head).
                full_shortfall = required_blocks - available_blocks
                if not self.thrash.coherent:
                    self.last_alloc_shortfall = full_shortfall
                    self.last_alloc_checks = (
                        (
                            full_shortfall,
                            full_reserved_blocks if required_blocks else 0,
                        ),
                    )
                    self.last_alloc_evicts_needed = False
                    if self.head_need_hook is not None and request.status in (
                        RequestStatus.WAITING,
                        RequestStatus.PREEMPTED,
                    ):
                        self.head_need_hook(
                            request, num_blocks_to_allocate, full_shortfall
                        )
                    return None
                # Coherent yields pick ONE session that must cover what the
                # admission needs, so measure the chunk gate below too (it can
                # need more: lookahead slots, reserved blocks).

        num_tokens_main_model = total_computed_tokens + num_new_tokens
        num_tokens_need_slot = min(
            num_tokens_main_model + num_lookahead_tokens, self.max_model_len
        )

        # the eviction check vetoes fresh admissions
        # only (no blocks: the cleanup below is a no-op), so a veto changes
        # nothing.
        if evicts_needed is not None and any(
            self.coordinator.get_blocks(request.request_id)
        ):
            evicts_needed = None
        # Free the blocks that are skipped during the attention computation
        # (e.g., tokens outside the sliding window).
        # We can do this even if we cannot schedule this request due to
        # insufficient free blocks.
        # Should call this function before allocating new blocks to reduce
        # the number of evicted blocks.
        # Free on the processed-token basis: in-flight steps' attention windows
        # still read blocks below the optimistic boundary, and rejected spec
        # tokens can roll it back.
        self.coordinator.remove_skipped_blocks(
            request.request_id,
            max(0, total_computed_tokens - request.num_in_flight_tokens),
            num_prompt_tokens=request.num_prompt_tokens,
        )
        # Re-measures the gate's slots
        if full_shortfall:
            # Coherent yields only: the cleanup may have freed blocks, so
            # measure the full gate again.
            full_shortfall = max(
                0,
                self.coordinator.get_num_blocks_to_allocate(
                    request_id=request.request_id,
                    num_tokens=full_num_slot_tokens,
                    new_computed_blocks=new_computed_block_list,
                    num_encoder_tokens=num_encoder_tokens,
                    total_computed_tokens=total_computed_tokens,
                    num_local_computed_tokens=num_local_computed_tokens,
                    num_tokens_main_model=full_num_tokens,
                    apply_admission_cap=True,
                    prefill_end=prefill_end,
                )
                + watermark_blocks
                + full_reserved_blocks
                - self.block_pool.get_num_free_blocks(),
            )
        # the eviction check counts the whole prompt
        # (later chunks of an admitted request are not checked). Counted before
        # the chunk: in align mode the count records the Mamba prefill
        # checkpoint the allocation below uses, so the chunk's must be the last.
        prompt_blocks = 0
        full_tokens = min(request.num_tokens, self.max_model_len)
        if evicts_needed is not None and full_tokens > num_tokens_main_model:
            prompt_blocks = self.coordinator.get_num_blocks_to_allocate(
                request_id=request.request_id,
                num_tokens=min(full_tokens + num_lookahead_tokens, self.max_model_len),
                new_computed_blocks=new_computed_block_list,
                num_encoder_tokens=num_encoder_tokens,
                total_computed_tokens=num_local_computed_tokens
                + num_external_computed_tokens,
                num_local_computed_tokens=num_local_computed_tokens,
                num_tokens_main_model=full_tokens,
                prefill_end=prefill_end,
            )

        num_blocks_to_allocate = self.coordinator.get_num_blocks_to_allocate(
            request_id=request.request_id,
            num_tokens=num_tokens_need_slot,
            new_computed_blocks=new_computed_block_list,
            num_encoder_tokens=num_encoder_tokens,
            total_computed_tokens=num_local_computed_tokens
            + num_external_computed_tokens,
            num_local_computed_tokens=num_local_computed_tokens,
            num_tokens_main_model=num_tokens_main_model,
            prefill_end=prefill_end,
        )

        # Keep `reserved_blocks` free for other in-flight sequences, and an
        # additional watermark of headroom for waiting/preempted admissions.
        available_blocks = self.block_pool.get_num_free_blocks() - reserved_blocks
        required_blocks = num_blocks_to_allocate + watermark_blocks
        # A step that draws no block
        # takes nothing from a reservation.
        if required_blocks == 0:
            available_blocks = max(available_blocks, 0)
        shortfall = max(full_shortfall, required_blocks - available_blocks)
        self.last_alloc_checks = tuple(
            check
            for check in (
                (full_shortfall, full_reserved_blocks),
                (
                    required_blocks - available_blocks,
                    reserved_blocks if required_blocks else 0,
                ),
            )
            if check[0] > 0
        )
        # New requests never reclaim un-copied
        # blocks; running requests are never gated.
        self.last_alloc_demotion_wait = False
        if self.admission_gate is not None and request.status in (
            RequestStatus.WAITING,
            RequestStatus.PREEMPTED,
        ):
            shortfall, self.last_alloc_demotion_wait = self.admission_gate(
                # The fresh pops: restore
                # destinations in, adopted cached-free hits out.
                request,
                num_blocks_to_allocate
                + len(restore_slots)
                + restore_tails
                - adopted_draws(
                    self.coordinator,
                    new_computed_block_list,
                    num_local_computed_tokens + num_external_computed_tokens,
                ),
                shortfall,
                new_computed_block_list,
            )
            if self.last_alloc_demotion_wait:
                self.last_alloc_shortfall = 0
                self.last_alloc_evicts_needed = False
                return None
        if shortfall > 0:
            # Cannot allocate new blocks (either gate)
            self.last_alloc_shortfall = shortfall
            self.last_alloc_evicts_needed = False
            return None
        self.last_alloc_shortfall = 0
        # the gate passed: refuse an admission that
        # would un-hash a block a protected waiter needs.
        # The restore destinations and
        # private tails allocated below are fresh pops too, drawn
        # before the chunk, as the write-back gate above counts them.
        self.last_alloc_evicts_needed = evicts_needed is not None and evicts_needed(
            max(num_blocks_to_allocate, prompt_blocks)
            + len(restore_slots) + restore_tails,
            new_computed_block_list,
        )
        if self.last_alloc_evicts_needed:
            return None

        if (
            new_computed_block_list is not self.empty_kv_cache_blocks.blocks
            or num_external_computed_tokens > 0
        ):
            # Append the new computed blocks to the request blocks until now to
            # avoid the case where the new blocks cannot be allocated.
            self.coordinator.allocate_new_computed_blocks(
                request_id=request.request_id,
                new_computed_blocks=new_computed_block_list,
                num_local_computed_tokens=num_local_computed_tokens,
                num_external_computed_tokens=num_external_computed_tokens,
                skip_zeroing_group_ids=skip_zeroing_group_ids,
                restore_slots=restore_slots,
            )
            # The slots are consumed.
            self.restore_slots.pop(request.request_id, None)
            self.restore_tails.pop(request.request_id, None)
            if restore_tails:
                self.restore_tail_blocks[request.request_id] = (
                    self.block_pool.get_new_blocks(restore_tails)
                )

        new_blocks = self.coordinator.allocate_new_blocks(
            request.request_id,
            num_tokens_need_slot,
            num_tokens_main_model,
            num_encoder_tokens,
        )

        # P/D: delay caching blocks if we have to recv from
        # remote. Update state for locally cached blocks.
        if delay_cache_blocks:
            self.release_pinned_hit(request.request_id)
            if self._grace_pins and num_new_computed_tokens > 0:
                self._release_grace_for_hit(
                    new_computed_block_list, num_local_computed_tokens
                )
            return self.create_kv_cache_blocks(new_blocks)

        # NOTE(woosuk): We want to commit (cache) up to num_local_computed_tokens
        # + num_external_computed_tokens + num_new_tokens, but must exclude
        # "non-committable" tokens (e.g., draft tokens that could be rejected).
        # Therefore, we cap the number at `request.num_tokens`, ensuring only
        # "finalized" tokens are cached.
        num_tokens_to_cache = min(
            total_computed_tokens + num_new_tokens,
            request.num_tokens,
        )
        self.coordinator.cache_blocks(request, num_tokens_to_cache)

        # Admitted: the table now references the hit, the pins are redundant.
        self.release_pinned_hit(request.request_id)
        if self._grace_pins and num_new_computed_tokens > 0:
            # ... and so does the grace pin of the turn it resumes.
            self._release_grace_for_hit(
                new_computed_block_list, num_local_computed_tokens
            )
        return self.create_kv_cache_blocks(new_blocks)

    def free(self, request: Request) -> None:
        """Free the blocks allocated for the request.
        We free the blocks in reverse order so that the tail blocks are evicted
        first when caching is enabled.

        Args:
            request: The request to free the blocks.

        """
        self.release_pinned_hit(request.request_id)
        if not self._grace_eligible(request):
            self.coordinator.free(request.request_id)
            return
        blocks = self.coordinator.pop_blocks_for_free(request.request_id)
        self._take_grace_pin(request, blocks)
        self.block_pool.free_blocks(blocks)

    MAX_GRACE_PINS = 16

    @property
    def num_grace_pins(self) -> int:
        return len(self._grace_pins)

    @property
    def num_grace_pinned_blocks(self) -> int:
        return sum(len(blocks) for blocks in self._grace_pins.values())

    def grace_pinned_blocks(self, request_id: str) -> list[KVCacheBlock]:
        return list(self._grace_pins.get(request_id, ()))

    def _grace_eligible(self, request: Request) -> bool:
        """A turn that finished with output (stop, length, repetition, or
        an abort after decoding) may have a next turn that resumes it. A
        preemption (still RUNNING), an aborted waiter (no output), an error
        or an ignored request leaves nothing worth holding."""
        return (
            self.enable_grace_pins
            and self.enable_caching
            and self.prefix_cache_lookup_enabled(request)
            and request.is_finished()
            and request.status
            not in (RequestStatus.FINISHED_ERROR, RequestStatus.FINISHED_IGNORED)
            and request.num_output_tokens > 0
        )

    def _take_grace_pin(self, request: Request, blocks: list[KVCacheBlock]) -> int:
        """Take one extra ref on every hashed block of a finished request's
        just-popped blocks (``blocks`` is in eviction order: each group's
        table tail first, the held GDN states last). The caller then drops
        the table refs, now or after the in-flight fence; the hashed blocks
        stay out of the free queue until the grace is released. Unhashed
        blocks (running state, MTP scratch, an unkeyed last block) go back
        with the table refs as before. Returns the number of blocks held."""
        keep: list[KVCacheBlock] = []
        seen: set[int] = set()
        for block in blocks:
            if block.is_null or block.block_hash is None or id(block) in seen:
                continue
            # Still referenced by the request's table or hold: the grace ref
            # never moves a block in or out of a free queue (any pool).
            assert block.ref_cnt > 0
            seen.add(id(block))
            keep.append(block)
        if not keep:
            return 0
        request_id = request.request_id
        self.release_grace_pin(request_id)  # a repeated free: replace
        for block in keep:
            block.ref_cnt += 1
        self._grace_pins[request_id] = keep
        self._grace_block_ids[request_id] = frozenset(seen)
        self._grace_prompt_tokens[request_id] = request.num_prompt_tokens
        # Lifecycle: the grace is a copy source; overflow is host-aware
        hooks = self.tier_hooks
        if hooks is not None:
            hooks.grace_taken(request, keep)
        while len(self._grace_pins) > self.MAX_GRACE_PINS:
            victim = hooks and hooks.pick_overflow_victim(list(self._grace_pins))
            self.release_grace_pin(victim or next(iter(self._grace_pins)), "overflow")
        return len(keep)

    def release_grace_pin(self, request_id: str, reason: str = "yield") -> int:
        """Return a finished request's grace-pinned blocks to the pool (in
        the eviction order a free would have used). Returns the count.

        ``reason`` (a ``GraceRelease`` value) is passed to the host tier's
        copy scheduler, for example "yield" (a waiter needed the blocks),
        "overflow" (more graces than MAX_GRACE_PINS) or "hit" (the turn
        resumed on the GPU)."""
        # Lifecycle: the copy scheduler sees every grace release
        if self.tier_hooks is not None and request_id in self._grace_pins:
            self.tier_hooks.grace_released(request_id, reason)
        blocks = self._grace_pins.pop(request_id, None)
        self._grace_block_ids.pop(request_id, None)
        self._grace_prompt_tokens.pop(request_id, None)
        if not blocks:
            return 0
        self.block_pool.free_blocks(blocks)
        return len(blocks)

    def _release_grace_for_hit(
        self,
        computed_blocks: tuple[Sequence[KVCacheBlock], ...],
        num_hit_tokens: int,
    ) -> None:
        """An admitted hit now references its blocks through the table, so
        the grace of the turn it resumes has done its job. That turn is a
        grace that holds the deepest hit block of every group whose deepest
        hit block any grace holds (the GDN state it resumes from, the
        attention block holding the key), and whose prompt the hit reaches
        to within two prefix units (the EAGLE-dropped tail key). A new turn
        that only shares a system prompt with finished conversations
        releases none of theirs."""
        candidates: set[str] | None = None
        for group in computed_blocks:
            deepest = next((b for b in reversed(group) if not b.is_null), None)
            if deepest is None:
                continue
            owners = {
                request_id
                for request_id, block_ids in self._grace_block_ids.items()
                if id(deepest) in block_ids
            }
            if owners:
                candidates = owners if candidates is None else candidates & owners
        if not candidates:
            return
        reach = num_hit_tokens + 2 * self.block_pool.hash_block_size
        for request_id in [r for r in self._grace_block_ids if r in candidates]:
            if reach >= self._grace_prompt_tokens.get(request_id, 0):
                # Lifecycle: a hit ends the grace (its copy is moot)
                self.release_grace_pin(request_id, "hit")

    def _pins_follow_moved_hash(
        self, source_block: KVCacheBlock, cow_block: KVCacheBlock
    ) -> None:
        """A Mamba producer moved ``source_block``'s hashes to ``cow_block``
        and keeps writing ``source_block``. A grace or waiter pin on the
        source follows to the copy (as a hold does), so it protects the
        hashed state and never leaves an extra ref on a block its owner must
        hold exclusively (speculative relocation asserts on that)."""
        for request_id, blocks in self._grace_pins.items():
            if id(source_block) in self._grace_block_ids[request_id]:
                self._swap_pinned_block(blocks, source_block, cow_block)
                self._grace_block_ids[request_id] = frozenset(
                    id(block) for block in blocks
                )
        for blocks in self._pinned_hits.values():
            self._swap_pinned_block(blocks, source_block, cow_block)
        self.thrash.follow_cow(source_block, cow_block)

    def _swap_pinned_block(
        self,
        blocks: list[KVCacheBlock],
        source_block: KVCacheBlock,
        cow_block: KVCacheBlock,
    ) -> None:
        for i, block in enumerate(blocks):
            if block is source_block:
                if any(other is cow_block for other in blocks):
                    del blocks[i]
                else:
                    assert cow_block.ref_cnt > 0
                    cow_block.ref_cnt += 1
                    blocks[i] = cow_block
                self.block_pool.free_blocks([source_block])
                return

    def num_yieldable_blocks(
        self,
        keep: str | None = None,
        keep_hit: tuple[Sequence[KVCacheBlock], ...] | None = None,
    ) -> int:
        """Blocks that would return to the free queue if every grace and
        every waiter pin except ``keep``'s were released now. With
        ``keep_hit`` (the hit ``keep`` is being admitted on), ``keep``'s
        pinned blocks outside that hit (the proof block past it) count too:
        ``release_pinned_extras`` can give them back. Blocks of that hit
        never count: freed, they are taken straight back by the admission
        (a hit that grew past the pin snapshot would otherwise be counted
        through the grace that still holds it)."""
        pins: dict[int, list] = {}
        holders = [*self._grace_pins.values()] + [
            blocks
            for request_id, blocks in self._pinned_hits.items()
            if request_id != keep
        ]
        hit_ids: set[int] = set()
        if keep is not None and keep_hit is not None:
            holders.append(self._pinned_extras(keep, keep_hit))
            hit_ids = {id(block) for group in keep_hit for block in group}
        for blocks in holders:
            for block in blocks:
                entry = pins.setdefault(id(block), [block, 0])
                entry[1] += 1
        return sum(
            1
            for block_id, (block, n) in pins.items()
            if block.ref_cnt == n and block_id not in hit_ids
        )

    def _pinned_extras(
        self, request_id: str, hit: tuple[Sequence[KVCacheBlock], ...]
    ) -> list[KVCacheBlock]:
        hit_ids = {id(block) for group in hit for block in group}
        return [
            block
            for block in self._pinned_hits.get(request_id, ())
            if id(block) not in hit_ids
        ]

    def release_pinned_extras(
        self, request_id: str, hit: tuple[Sequence[KVCacheBlock], ...]
    ) -> int:
        """Drop from the request's pin the blocks outside ``hit``: the proof
        block past it, and any hit block a later lookup added. The request
        is being admitted on ``hit`` now, so they protect nothing it still
        needs; they cost it blocks. Returns the number dropped."""
        extras = self._pinned_extras(request_id, hit)
        if not extras:
            return 0
        extra_ids = {id(block) for block in extras}
        rest = [
            block
            for block in self._pinned_hits[request_id]
            if id(block) not in extra_ids
        ]
        if rest:
            self._pinned_hits[request_id] = rest
            self.thrash.trim_extras(request_id, rest)
        else:
            del self._pinned_hits[request_id]
            self.thrash.forget(request_id)
        self.block_pool.free_blocks(extras)
        return len(extras)

    MAX_PINNED_WAITERS = 16
    # Waiting requests one pin pass inspects (each
    # a full prefix lookup): cold requests pin nothing, so the pin bound alone
    # does not bound the scan.
    MAX_INSPECTED_WAITERS = 2 * MAX_PINNED_WAITERS

    @property
    def num_pinned_waiters(self) -> int:
        return len(self._pinned_hits)

    @property
    def num_pinned_blocks(self) -> int:
        return sum(len(blocks) for blocks in self._pinned_hits.values())

    def pin_waiting_hit(self, request: Request) -> int:
        """Pin the request's current prefix-cache hit while it waits for
        admission: every non-null hit block in every group (the attention
        prefix up to the key and the GDN state it resumes from) takes a ref
        and leaves the free queue. Idempotent. Returns the number of blocks
        pinned now (0 if nothing hit, already pinned, caching is off, or the
        waiter bound is reached).

        Known gap: a pin is a snapshot of the hit at pin time. It is not
        refreshed if the hit grows while the request waits (a sibling
        computing more of the same prefix); the admission lookup still finds
        whatever is cached then, so the only cost is a smaller hit, never a
        wrong one. A producer moving the hash of a pinned live GDN block into
        its CoW copy is followed (``_pins_follow_moved_hash``)."""
        request_id = request.request_id
        if (
            request_id in self._pinned_hits
            or not self.enable_caching
            or not self.prefix_cache_lookup_enabled(request)
            or len(self._pinned_hits) >= self.MAX_PINNED_WAITERS
        ):
            return 0
        if self._pin_miss_holds(request):
            return 0
        computed, num_hit, _ = self.get_computed_blocks(request)
        blocks = [
            block
            for group in computed.blocks
            for block in group
            if not block.is_null
        ]
        if num_hit == 0 or not blocks:
            self._remember_pin_miss(request)
            return 0
        self._pin_misses.pop(request_id, None)
        unit = self.block_pool.hash_block_size
        proof_idx = num_hit // unit
        if proof_idx < len(request.block_hashes):
            seen = {block.block_id for block in blocks}
            for group_id in range(self.num_kv_cache_groups):
                cached = self.block_pool.get_cached_block(
                    request.block_hashes[proof_idx], [group_id]
                )
                if cached and cached[0].block_id not in seen:
                    seen.add(cached[0].block_id)
                    blocks.append(cached[0])
        # Blocks of a private-pool group (hisparse host tier) are touched in
        # their own pool, as free_blocks routes them.
        for block in blocks:
            (block.pool or self.block_pool).touch([block])
        self._pinned_hits[request_id] = blocks
        self.thrash.record_pin(request_id, num_hit, computed, request.block_hashes)
        return len(blocks)

    def _pin_signature(self, request: Request) -> tuple:
        return (
            request.num_tokens, len(request.block_hashes),
            self.prefix_cache_lookup_enabled(request),
        )

    def _pin_key_log(self) -> KeyLog:
        if self.block_pool.key_log is None:
            self.block_pool.key_log = KeyLog()
        return self.block_pool.key_log

    def _pin_miss_holds(self, request: Request) -> bool:
        """The request's last pin lookup found nothing, and nothing that
        could change that happened since: same cap, and no key of its own
        inserted in the GPU pool (a miss only ends with a new key; an
        eviction only shortens a hit). Same freshness rule as the plan cache,
        so the bounded pin pass scans a cold waiter once."""
        entry = self._pin_misses.get(request.request_id)
        if entry is None:
            return False
        signature, position, hashes = entry
        if signature != self._pin_signature(request):
            return False
        keys = self._pin_key_log().since(position)
        if keys is None:
            return False
        if not keys:
            return True
        if hashes is None:  # built once, on the first keys to check
            hashes = frozenset(request.block_hashes)
            self._pin_misses[request.request_id] = (signature, position, hashes)
        from vllm.v1.core.kv_cache_utils import get_block_hash

        return not any(get_block_hash(k) in hashes for k in keys)

    def _remember_pin_miss(self, request: Request) -> None:
        # Keys inserted this step may be excluded from this step's hit
        # (``published_block_ids_this_step`` / ``cached_blocks_this_step``):
        # read from the step's start so they end the miss next step.
        old = self._pin_misses.get(request.request_id)
        hashes = (
            old[2]
            if old is not None and old[0][1] == len(request.block_hashes)
            else None
        )
        position = min(self._step_key_mark, self._pin_key_log().end)
        self._pin_misses[request.request_id] = (
            self._pin_signature(request), position, hashes,
        )

    def pinned_hit_blocks(self, request_id: str) -> list[KVCacheBlock]:
        """The blocks pinned for ``request_id`` (empty if none)."""
        return list(self._pinned_hits.get(request_id, ()))

    def release_pinned_hit(self, request_id: str) -> int:
        """Drop the request's pin (admission, abort, finish). Returns the
        number of blocks released."""
        blocks = self._pinned_hits.pop(request_id, None)
        self._pin_misses.pop(request_id, None)
        self.thrash.forget(request_id)
        if not blocks:
            return 0
        # Reversed: the prefix stays an eviction candidate longer than the
        # tail, as after a free.
        self.block_pool.free_blocks(reversed(blocks))
        return len(blocks)

    def yield_pinned_hits(
        self, keep: str | None = None, limit: int | None = None
    ) -> int:
        """Drop pins (except ``keep``'s) so a running request can allocate,
        or a waiter ahead in line can be admitted: grace pins of finished
        turns first (oldest first), then the most recently pinned waiters, at
        most ``limit`` pins per call (all by default; the scheduler passes 1
        and retries, so a decode step that needs a block costs one pin, not
        every pin). Returns the number of blocks released; 0 means there was
        nothing to yield."""
        released = 0
        yielded = 0
        # Grace pins first (oldest finished conversation first), then the
        # waiters' pins (most recently pinned first).
        # Host-aware grace order (plain oldest-first without the tier)
        for request_id in self.thrash.grace_yield_order():
            if limit is not None and yielded >= limit:
                return released
            released += self.thrash.release_for_yield([("grace", request_id)])
            yielded += 1
        for request_id in reversed(list(self._pinned_hits)):
            if request_id == keep:
                continue
            if limit is not None and yielded >= limit:
                break
            released += self.thrash.release_for_yield([("waiter", request_id)])
            yielded += 1
        return released

    def remove_skipped_blocks(
        self,
        request_id: str,
        processed_computed_tokens: int,
        num_prompt_tokens: int | None = None,
    ) -> None:
        """Remove the blocks that are no longer needed from `blocks` and replace
        the removed blocks with null_block.

        Args:
            request_id: The request ID.
            processed_computed_tokens: Computed-token prefix length covering
                fully processed and committed tokens only (safe to free).
            num_prompt_tokens: Optional prompt length for R-SWA gap eviction.

        """
        self.coordinator.remove_skipped_blocks(
            request_id, processed_computed_tokens, num_prompt_tokens
        )

    def pop_blocks_for_free(self, request: Request) -> list[KVCacheBlock]:
        """Pop the request's bookkeeping and return its blocks without
        returning them to the block pool. The caller must eventually free
        them as-is: they are already ordered so that tail blocks are
        evicted first, within each KV cache group.

        Args:
            request: The request to pop the blocks for.

        Returns:
            The request's blocks in eviction order (each group's blocks
            reversed, then concatenated group by group).

        """
        # A pin is never written by an in-flight step (the waiter has no
        # table), so it is released now, not with the deferred blocks.
        self.release_pinned_hit(request.request_id)
        blocks = self.coordinator.pop_blocks_for_free(request.request_id)
        # A finished turn whose last step is still in flight: the grace takes
        # its ref now; the table refs go when the fence drains.
        if self._grace_eligible(request):
            self._take_grace_pin(request, blocks)
        return blocks

    def evict_blocks(self, block_ids: set[int]) -> None:
        """Evict blocks from the prefix cache by their block IDs.

        Args:
            block_ids: Set of block IDs to evict from cache.

        """
        self.block_pool.evict_blocks(block_ids)

    def reset_prefix_cache(self) -> bool:
        """Reset prefix cache. This function may be used in RLHF
        flows to invalidate prefix caching after the weights are updated,
        or used for resetting prefix caching status for benchmarking.

        Returns:
            bool: True if the prefix cache is successfully reset,
            False otherwise.

        """
        # Pinned waiters' hits are about to
        # be invalidated anyway, but only drop the pins when they are all that
        # stands in the reset's way: a reset that fails for live requests keeps
        # every grace and waiter pin.
        if (self._pinned_hits or self._grace_pins) and self._pins_alone_block_reset():
            self.yield_pinned_hits()
        if not self.coordinator.reset_prefix_cache():
            return False
        if self.log_stats:
            assert self.prefix_cache_stats is not None
            self.prefix_cache_stats.reset = True
        return True

    def _pins_alone_block_reset(self) -> bool:
        """Whether releasing every grace and waiter pin would leave each pool
        with only its null block in use (the condition
        ``BlockPool.reset_prefix_cache`` checks)."""
        counts: dict[int, list] = {}
        for blocks in (*self._grace_pins.values(), *self._pinned_hits.values()):
            for block in blocks:
                entry = counts.setdefault(id(block), [block, 0])
                entry[1] += 1
        pinned_only: dict[int, int] = {}
        for block, n in counts.values():
            if not block.is_null and block.ref_cnt == n:
                pool = block.pool or self.block_pool
                pinned_only[id(pool)] = pinned_only.get(id(pool), 0) + 1
        pools = dict.fromkeys(
            manager.block_pool for manager in self.coordinator.single_type_managers
        )
        return all(
            pool.num_gpu_blocks - pool.get_num_free_blocks() - 1
            == pinned_only.get(id(pool), 0)
            for pool in pools
        )

    def get_num_common_prefix_blocks(self, running_request_id: str) -> list[int]:
        """Deprecated: cascade attention has been removed. Always returns 0
        for each KV cache group. Kept for out-of-tree schedulers."""
        return [0] * self.num_kv_cache_groups

    def take_events(self) -> list[KVCacheEvent]:
        """Take the KV cache events from the block pool.

        Returns:
            A list of KV cache events.

        """
        events = self.block_pool.take_events()
        for event in events:
            if not isinstance(event, BlockStored):
                continue
            if event.group_idx is None:
                continue
            if event.group_idx < 0 or event.group_idx >= len(
                self.kv_cache_event_metadata
            ):
                logger.warning(
                    "Group index `%s` not in KV cache metadata", event.group_idx
                )
                continue
            # Annotate here so BlockPool can keep emitting structural cache
            # events without owning semantic KV cache spec metadata.
            kind, sliding_window = self.kv_cache_event_metadata[event.group_idx]
            event.kv_cache_spec_kind = kind
            event.kv_cache_spec_sliding_window = sliding_window
        return events

    def get_blocks(self, request_id: str) -> KVCacheBlocks:
        """Get the blocks of a request."""
        return self.create_kv_cache_blocks(self.coordinator.get_blocks(request_id))

    def get_block_ids(self, request_id: str) -> tuple[list[int], ...]:
        """Get the block ids of a request."""
        return self.get_blocks(request_id).get_block_ids()

    def get_block_ids_for_computed_tokens(
        self,
        request_id: str,
        num_computed_tokens: int,
    ) -> tuple[list[int], ...]:
        """Get block ids covering the request's computed tokens."""
        block_ids = self.get_block_ids(request_id)
        clipped_block_ids: list[list[int]] = []
        for group, ids in zip(self.kv_cache_config.kv_cache_groups, block_ids):
            spec = group.kv_cache_spec
            if not isinstance(spec, AttentionSpec) or isinstance(
                spec, (CrossAttentionSpec, EncoderOnlyAttentionSpec)
            ):
                clipped_block_ids.append(ids)
                continue

            num_valid_blocks = cdiv(num_computed_tokens, spec.block_size)
            clipped_block_ids.append(ids[:num_valid_blocks])
        return tuple(clipped_block_ids)

    def estimate_cached_tokens(self, request: Request) -> int:
        """Estimate the number of tokens cached by the request."""
        cached_tokens: int | None = None
        for group, blocks in zip(
            self.kv_cache_config.kv_cache_groups,
            self.get_blocks(request.request_id).blocks,
        ):
            if not group.kv_cache_spec.prefix_cacheable:
                continue
            if isinstance(
                group.kv_cache_spec,
                (CrossAttentionSpec, EncoderOnlyAttentionSpec),
            ):
                # Cross-attention and encoder-only groups are not prefix cached.
                continue

            group_cached_tokens = 0
            for block in blocks:
                group_cached_tokens = max(
                    group_cached_tokens,
                    block.block_hash_num_tokens or 0,
                )

            cached_tokens = (
                group_cached_tokens
                if cached_tokens is None
                else min(cached_tokens, group_cached_tokens)
            )

        return cached_tokens or 0

    def cache_blocks(self, request: Request, num_computed_tokens: int) -> None:
        """Cache the blocks for the request, if enabled.

        Args:
            request: The request to cache the blocks.
            num_computed_tokens: The number of computed tokens, including tokens
                that are already cached and tokens to be cached.

        """
        if self.enable_caching:
            self.coordinator.cache_blocks(request, num_computed_tokens)

    def create_kv_cache_blocks(
        self, blocks: tuple[list[KVCacheBlock], ...]
    ) -> KVCacheBlocks:
        # Only create new KVCacheBlocks for non-empty blocks
        return KVCacheBlocks(blocks) if any(blocks) else self.empty_kv_cache_blocks

    def truncate_computed_blocks(
        self, blocks: KVCacheBlocks, num_computed_tokens: int
    ) -> KVCacheBlocks:
        """Return a lookup-result view truncated at an aligned token endpoint.

        An external hit can supply the final Mamba state even when the local
        Mamba group ends before this endpoint. Other groups must cover it.
        Pure slicing: refcounts are untouched and ``blocks`` is not mutated.
        """
        truncated: list[list[KVCacheBlock]] = []
        for group_blocks, manager, group in zip(
            blocks.blocks,
            self.coordinator.single_type_managers,
            self.kv_cache_config.kv_cache_groups,
            strict=True,
        ):
            if not group.kv_cache_spec.prefix_cacheable:
                # Scratch groups (e.g. the CSA compressor ring) hold a fixed
                # block covering no token range, so a lookup result never
                # carries computed blocks for them and their block size does
                # not divide the endpoint.
                assert not group_blocks
                truncated.append([])
                continue
            assert num_computed_tokens % manager.block_size == 0
            num_blocks = num_computed_tokens // manager.block_size
            if isinstance(group.kv_cache_spec, MambaSpec):
                num_blocks = min(num_blocks, len(group_blocks))
            else:
                assert num_blocks <= len(group_blocks)
            truncated.append(list(group_blocks[:num_blocks]))
        return self.create_kv_cache_blocks(tuple(truncated))

    def take_new_block_ids(self) -> list[int]:
        """Drain and return new attention block IDs for zeroing."""
        ids: list[int] = []
        for mgr in self.coordinator.single_type_managers:
            ids.extend(mgr.take_new_block_ids())
        return ids

    def take_new_ring_block_ids(self) -> list[int]:
        """Drain ring blocks claimed since the last call (ring-row tag clear).
        Independent of KV zeroing: every claim is returned."""
        ids: list[int] = []
        for mgr in self.coordinator.single_type_managers:
            ids.extend(mgr.take_new_ring_block_ids())
        return ids

    def get_ring_block_ids(self, request_id: str) -> list[int]:
        """The request's ring (CircularBufferSpec) block ids, all groups."""
        ids: list[int] = []
        for mgr in self.coordinator.single_type_managers:
            ids.extend(mgr.get_ring_block_ids(request_id))
        return ids

    def record_blocks_for_zeroing(self, request_id: str, start_token: int) -> None:
        """Re-record the request's blocks from start_token onwards for
        zeroing, e.g. blocks a failed async KV load left unwritten.

        start_token must be block-aligned: zeroing a partially-valid block
        would wipe its valid prefix.
        """
        for mgr in self.coordinator.single_type_managers:
            if mgr.records_new_block_ids:
                assert start_token % mgr.block_size == 0
                start_idx = start_token // mgr.block_size
                blocks = mgr.req_to_blocks[request_id]
                mgr.new_block_ids.extend(blk.block_id for blk in blocks[start_idx:])

    def take_kv_cache_block_copies(
        self,
    ) -> tuple[list[KVCacheBlockCopy], list[KVCacheBlock]]:
        """Drain pending copies and return their retained endpoints."""
        pending_copies: list[tuple[KVCacheBlock, KVCacheBlock]] = []
        for mgr in self.coordinator.single_type_managers:
            pending_copies.extend(mgr.take_pending_cow_copies())
        copies = [
            KVCacheBlockCopy(
                src_block_id=source_block.block_id,
                dst_block_id=cow_block.block_id,
            )
            for source_block, cow_block in pending_copies
        ]
        retained_blocks = [block for pair in pending_copies for block in pair]
        return copies, retained_blocks

    def take_boundary_state_offloads(
        self,
    ) -> dict[str, list[tuple[int, int, int]]]:
        """Drain this step's boundary-state hand-offs for a KV connector.

        Returns ``{request_id: [(group_id, block_id, boundary_tokens), ...]}``
        for mamba "align" boundary states: the
        request's committed boundary-state snapshots and, on a sub-block partial
        hit, the CoW copy of its last-prompt-boundary state. Only mamba "align"
        groups contribute; empty otherwise. A connector reads the referenced
        blocks — never resolving them positionally — and offloads them so a
        later request can hit that prefix.
        """
        offloads: dict[str, list[tuple[int, int, int]]] = {}
        for mgr in self.coordinator.single_type_managers:
            for (
                req_id,
                group_id,
                block,
                boundary_tokens,
            ) in mgr.take_pending_boundary_state_offloads():
                offloads.setdefault(req_id, []).append(
                    (group_id, block.block_id, boundary_tokens)
                )
        return offloads

    def finalize_partial_tail_offloads(
        self, request: Request
    ) -> list[tuple[int, int, int]]:
        """Consume safe producer partial tails when a request finishes.

        A mamba align table block is a valid boundary source only if no later
        token was forwarded. The connector pins and queues the exact table
        block before request cleanup, then releases the pin when every worker
        reports the store job complete.
        """
        offloads: list[tuple[int, int, int]] = []
        for mgr in self.coordinator.single_type_managers:
            finalized = mgr.finalize_partial_tail_offload(
                request.request_id,
                request.num_computed_tokens,
                request.num_in_flight_tokens,
            )
            if finalized is None:
                continue
            group_id, block, boundary_tokens = finalized
            offloads.append((group_id, block.block_id, boundary_tokens))
        return offloads

    def new_step_starts(self) -> None:
        """Notify the coordinator that a new step is starting."""
        self.coordinator.new_step_starts()
        # Pin-miss freshness mark.
        if self.block_pool.key_log is not None:
            self._step_key_mark = self.block_pool.key_log.end

    def invalidate_step_publications(self, request: Request) -> None:
        """Withdraw the prefix-cache entries ``request`` published this step.

        Call it before freeing a request that was scheduled this step and is
        then removed from the schedule: ``allocate_slots`` keys its blocks
        ahead of a forward that will not run, so they hold no valid KV.
        """
        if self.enable_caching:
            self.coordinator.invalidate_step_publications(request.request_id)
