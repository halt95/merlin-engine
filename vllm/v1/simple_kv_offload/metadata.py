# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modified by the Merlin project; see NOTICE.
"""Metadata for SimpleCPUOffloadConnector."""

from dataclasses import dataclass, field

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata,
    KVConnectorWorkerMetadata,
)

INVALID_JOB_ID = -1


@dataclass
class SimpleCPUOffloadMetadata(KVConnectorMetadata):
    """Metadata passed from scheduler to worker for CPU offload operations.

    The worker receives flat block lists keyed by a monotonic event_idx.
    Job->req_id translation is handled by the scheduler-side manager
    (via inverse maps), so the worker never knows about request identities.
    """

    # Load event per step. INVALID_JOB_ID means no blocks to load this step.
    load_event: int = INVALID_JOB_ID
    load_gpu_blocks: list[int] = field(default_factory=list)
    load_cpu_blocks: list[int] = field(default_factory=list)
    # Reverse map: load_event->req_ids, for tracking requests with finished load events
    load_event_to_reqs: dict[int, list[str]] = field(default_factory=dict)

    # More than one load event this step: one
    # (event, gpu blocks, cpu blocks) per request, in issue order, replacing
    # load_event / load_*_blocks (which then hold the first event and the
    # concatenated lists). Empty: the single load_event above.
    load_transfers: list[tuple[int, list[int], list[int]]] = field(
        default_factory=list
    )

    # Store event per step. INVALID_JOB_ID means no blocks to store this step.
    store_event: int = INVALID_JOB_ID
    store_gpu_blocks: list[int] = field(default_factory=list)
    store_cpu_blocks: list[int] = field(default_factory=list)

    # Whether any requests were preempted this step and need flush pending transfers.
    need_flush: bool = False
    # Manager pins every DMA source/destination.
    transfer_leases: bool = False
    cache_epoch: int = 0
    # Rejected loads complete without a DMA event.
    failed_recving: set[str] = field(default_factory=set)
    # Content writer identity and ack feedback.
    scheduled_step: int = 0
    acked_step: int = 0


    def load_transfer_list(self) -> list[tuple[int, list[int], list[int]]]:
        """Every load event of this step with its block lists."""
        if self.load_transfers:
            return self.load_transfers
        if self.load_event < 0:
            return []
        return [(self.load_event, self.load_gpu_blocks, self.load_cpu_blocks)]


@dataclass
class SimpleCPUOffloadWorkerMetadata(KVConnectorWorkerMetadata):
    """Worker -> Scheduler metadata for completed store events.

    Each worker reports {event_idx: 1} for newly completed stores.
    ``aggregate()`` sums counts across workers within a step.
    The scheduler-side manager accumulates across steps and processes
    a store completion only when count reaches ``world_size``.
    """

    completed_store_events: dict[int, int]
    # cumulative, keyed by global rank.
    census: dict[str, dict[str, float]] = field(default_factory=dict)
    # Preserve rank identity through aggregation.
    committed_steps: dict[int, int] = field(default_factory=dict)
    masked_rows: set[tuple[str, int]] = field(default_factory=set)
    # The ranks that completed each store event
    # (host tier only): a demotion txn publishes once every rank acked it,
    # and a repeated report from one rank is idempotent.
    store_ack_ranks: dict[int, frozenset[int]] = field(default_factory=dict)
    # per-rank transaction reports:
    # (epoch, txn, batch) -> {rank: (success, quiescent)}. ``txn`` is the
    # store event / load event index. A failure is sticky and a rank counts
    # as quiescent only once its copies provably stopped; a transaction
    # resolves when every rank reported quiescent (kv_host_tier_lifecycle).
    store_reports: dict[tuple[int, int, int], dict[int, tuple[bool, bool]]] = (
        field(default_factory=dict)
    )
    load_reports: dict[tuple[int, int, int], dict[int, tuple[bool, bool]]] = (
        field(default_factory=dict)
    )

    def aggregate(
        self, other: "KVConnectorWorkerMetadata"
    ) -> "KVConnectorWorkerMetadata":
        assert isinstance(other, SimpleCPUOffloadWorkerMetadata)
        merged = dict(self.completed_store_events)
        for k, v in other.completed_store_events.items():
            merged[k] = merged.get(k, 0) + v
        # Ranks retain independent samples.
        census = {**self.census, **other.census}
        # Repeated reports cannot stand in for a rank.
        steps = dict(self.committed_steps)
        for rank, step in other.committed_steps.items():
            steps[rank] = max(steps.get(rank, 0), step)
        ranks = dict(self.store_ack_ranks)
        for k, v in other.store_ack_ranks.items():
            ranks[k] = ranks.get(k, frozenset()) | v
        store_reports = _merge_reports(self.store_reports, other.store_reports)
        load_reports = _merge_reports(self.load_reports, other.load_reports)
        return SimpleCPUOffloadWorkerMetadata(
            completed_store_events=merged,
            census=census,
            committed_steps=steps,
            masked_rows=self.masked_rows | other.masked_rows,
            store_ack_ranks=ranks,
            store_reports=store_reports,
            load_reports=load_reports,
        )


def _merge_reports(
    left: dict[tuple[int, int, int], dict[int, tuple[bool, bool]]],
    right: dict[tuple[int, int, int], dict[int, tuple[bool, bool]]],
) -> dict[tuple[int, int, int], dict[int, tuple[bool, bool]]]:
    """Union per (epoch, txn, batch, rank); failure sticky, quiescence monotone."""
    merged = {key: dict(ranks) for key, ranks in left.items()}
    for key, ranks in right.items():
        into = merged.setdefault(key, {})
        for rank, (ok, quiet) in ranks.items():
            old = into.get(rank)
            into[rank] = (ok, quiet) if old is None else (
                old[0] and ok, old[1] or quiet
            )
    return merged
