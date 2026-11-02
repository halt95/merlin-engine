# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""MerlinKVHostTierConnector: the host-RAM KV tier (MERLIN_KV_HOST_TIER_GIB).

SimpleCPUOffloadConnector in kv_both mode, run as a write-back tier: the
scheduler manager demotes victims from the GPU
free-queue head and gates admissions on un-copied blocks
(vllm/merlin/kv_host_tier_writeback.py); the transfers are the parent's. There
is no mode: lazy and eager are not selectable. It exists so the tier has one
name the engine builds from vllm/merlin/kv_host_tier.py, and so the tier's
policy is explicit:

- only the exact generated config with the knob on is accepted (its
  contract marker included), using the same recognition as the engine
  exemptions; disk, lazy, eager-contract and other overrides are refused
  before the parent connector can create a backend;
- at cache-manager bind the write-back admission gate is installed on the
  KVCacheManager;
- a failed restore copy is reported as a failed receive of its requests
  (get_transfer_results); with the generated
  kv_load_failure_policy="recompute" the scheduler recomputes them, after the
  tier un-keyed the host blocks the load read (vllm/merlin/
  kv_host_tier_runtime.py). A failed store still fails loudly;
- tier metrics (hit/miss/stored/evicted/load-failed tokens, cached arena
  occupancy) ride the SimpleCPUOffload stats payload and Prometheus class.

The upstream SimpleCPUOffloadConnector keeps upstream behaviour (RecoverSSM
refused, side-cache layout off, prefix-cache pins off).
"""

from typing import TYPE_CHECKING, Any

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorRole,
    KVConnectorTransferResults,
)
from vllm.distributed.kv_transfer.kv_connector.v1.simple_cpu_offload_connector import (  # noqa: E501
    SimpleCPUOffloadConnector,
)
from vllm.merlin import kv_host_tier, kv_host_tier_hooks, kv_host_tier_runtime

if TYPE_CHECKING:
    from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
        KVConnectorPromMetrics,
        KVConnectorStats,
        PromMetric,
        PromMetricT,
    )
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request


class MerlinKVHostTierConnector(SimpleCPUOffloadConnector):
    """SimpleCPUOffloadConnector (write-back) as the Merlin host-RAM KV tier."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ):
        if (
            not kv_host_tier.is_active(vllm_config)
            or vllm_config.kv_transfer_config.kv_load_failure_policy != "recompute"
        ):
            raise ValueError(
                "MerlinKVHostTierConnector only accepts the generated write-back "
                "host-RAM tier config with kv_load_failure_policy='recompute'. "
                f"Set {kv_host_tier.KNOB} to a positive GiB value and remove "
                "--kv-transfer-config; hand-written, lazy_offload, disk and "
                "module-path overrides are not supported."
            )
        super().__init__(vllm_config, role, kv_cache_config)

    # --- worker side ---
    def get_transfer_results(
        self, finished_req_ids: set[str]
    ) -> KVConnectorTransferResults:
        results = super().get_transfer_results(finished_req_ids)
        worker = getattr(self, "worker_handler", None)
        if worker is not None:
            results.failed_recving |= worker.take_failed_recving()
        return results

    # --- scheduler side ---
    def bind_kv_cache_manager(self, kv_cache_manager: "KVCacheManager") -> None:
        super().bind_kv_cache_manager(kv_cache_manager)
        manager = getattr(self, "scheduler_manager", None)
        if manager is not None:
            manager.bind_kv_cache_manager(kv_cache_manager)

    def record_preempted(self, request: "Request") -> None:
        """A preempted request's computed prefix is a restore unit, recorded
        as at finish (``UnitTable.record_finished``) before the scheduler
        frees its blocks, so its copy has a unit's eviction rights. A repeat
        preemption or the later finish supersedes the earlier record (the
        successor contains it)."""
        manager = getattr(self, "scheduler_manager", None)
        runtime = getattr(manager, "tier_runtime", None)
        units = getattr(runtime, "units", None)
        if units is not None:
            units.record_finished(request)

    def has_pending_block_frees(self) -> bool:
        """Only transfers whose completion frees a GPU block: a
        restore or a grace/lane-1 copy in flight must not stop a running
        request from yielding a grace pin or preempting
        (kv_host_tier_hooks.has_pending_block_frees)."""
        manager = getattr(self, "scheduler_manager", None)
        if manager is None:
            return False
        return kv_host_tier_hooks.has_pending_block_frees(manager)

    def get_kv_connector_stats(self) -> "KVConnectorStats | None":
        stats = super().get_kv_connector_stats()
        manager = getattr(self, "scheduler_manager", None)
        runtime = getattr(manager, "tier_runtime", None)
        if stats is not None and runtime is not None:
            runtime.add_stats(stats)
        return stats

    @classmethod
    def build_prom_metrics(
        cls,
        vllm_config: VllmConfig,
        metric_types: "dict[type[PromMetric], type[PromMetricT]]",
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[Any]],
    ) -> "KVConnectorPromMetrics":
        from vllm.v1.simple_kv_offload.metrics import SimpleCPUOffloadPromMetrics

        return kv_host_tier_runtime.build_prom_metrics(
            SimpleCPUOffloadPromMetrics,
            vllm_config,
            metric_types,
            labelnames,
            per_engine_labelvalues,
        )
