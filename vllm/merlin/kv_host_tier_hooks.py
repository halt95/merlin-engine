# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Host-RAM KV tier: lifecycle-hook glue, kept out of the upstream
files. The scheduler-side manager (``SimpleCPUOffloadScheduler``) calls these
through one-line hooks; the copy scheduler (``kv_host_tier_copyq``) is the
``LifecycleHooks`` target it installs with ``install_tier_hooks``. Everything
here is a no-op until a target is installed, so with no copy scheduler the
tier behaves as it does without one and with the tier off nothing is installed at all.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from vllm.merlin.kv_host_tier_iface import BoundaryManifest

if TYPE_CHECKING:
    from vllm.v1.request import Request


class KVCMLifecycleAdapter:
    """Installed as ``KVCacheManager.tier_hooks`` with the tier on. The
    KVCacheManager has no scheduler step, so this adds ``_sched_step`` and
    forwards to the manager's ``tier_hooks`` (the copy scheduler)."""

    def __init__(self, manager: Any) -> None:
        self._m = manager

    def grace_taken(self, request: "Request", blocks: Any) -> None:
        hooks = self._m.tier_hooks
        if hooks is not None:
            hooks.grace_taken(request, blocks, self._m._sched_step)

    def grace_released(self, request_id: str, reason: str) -> None:
        hooks = self._m.tier_hooks
        if hooks is not None:
            hooks.grace_released(request_id, reason, self._m._sched_step)

    def pick_overflow_victim(self, candidates: list[str]) -> str | None:
        """Host-aware grace overflow: the copy scheduler
        may name a host-complete grace to release first; None = oldest."""
        pick = getattr(self._m.tier_hooks, "pick_overflow_victim", None)
        victim = pick(candidates) if pick is not None else None
        return victim if victim in candidates else None


def boundary_manifest(units: Any, request: "Request", event: str) -> BoundaryManifest | None:
    """The request's ``BoundaryManifest`` from the unit table's view, or None
    (no table, no boundary, incomplete). ``uid`` is the recorded unit with
    the same boundary and required set (``record_finished`` /
    ``record_preempted`` ran before the hook), None if never recorded."""
    if units is None:
        return None
    from vllm.merlin.kv_host_tier_units import Incomplete, UnitManifest

    found = units.manifest(request)
    if found is None or isinstance(found, Incomplete):
        return None
    assert isinstance(found, UnitManifest)
    uid = next(
        (
            u.uid
            for key in found.required
            for u in units.units_of(key)
            if u.boundary == found.boundary and u.required == found.required
        ),
        None,
    )
    return BoundaryManifest(
        request_id=request.request_id,
        boundary=found.boundary,
        required=found.required,
        endpoints=found.endpoints,
        optional=found.optional,
        uid=uid,
        event=event,  # type: ignore[arg-type]
    )


def bind_tier_geometry(manager: Any, kv_cache_config: Any) -> None:
    """The ledger's hash grid and, with what the GPU and host tiers hold, the
    size of its request history."""
    manager.tier_runtime.bind_geometry(
        manager.hash_block_size,
        cache_tokens=(kv_cache_config.num_blocks + manager.num_cpu_blocks)
        * manager.block_size,
    )


def notify_finished(manager: Any, request: "Request", unit: Any) -> None:
    """``request_finished_all_groups``: the finished turn's manifest goes to
    the copy scheduler (its grace, taken by the KVCacheManager's free, is the
    grace copy source)."""
    hooks = manager.tier_hooks
    if hooks is None:
        return
    manifest = None
    if unit is not None:
        units = manager.tier_runtime.units
        manifest = BoundaryManifest.from_unit(
            request.request_id, unit, event="finish",
            host_complete=units.is_host_complete(unit),
        )
    hooks.finished(request, manifest, manager._sched_step)


def notify_preempted(manager: Any, request: "Request") -> None:
    """Scheduler ``_preempt_request`` hook, after ``record_preempted``
    recorded the unit and before the blocks are freed (bounded preempt
    grace)."""
    hooks = manager.tier_hooks
    if hooks is None:
        return
    runtime = manager.tier_runtime
    units = runtime.units if runtime is not None else None
    hooks.preempted(request, boundary_manifest(units, request, "preempt"), manager._sched_step)


def has_pending_block_frees(manager: Any) -> bool:
    """The tier's ``KVConnectorBase.has_pending_block_frees``: whether
    waiting for a transfer can satisfy a running request's allocation
    instead of a pin yield or a preemption. True only for transfers whose
    completion returns a GPU block to the free queue:

    * an in-flight demotion whose source only the copy still references
      (ref 1, the admission gate's ``_soon_free``), current or of an older
      epoch; a grace / lane-1 / preempt copy of a block a grace, a hold or a
      running table still references (ref >= 2) frees nothing;
    * a store without a demotion transaction (base semantics: its transfer
      refs are released at completion) and a finish-time store;
    * a cancelled load (abandoned at reset, or its request finished while
      the DMA was in flight): its destination blocks go back on completion.

    A live restore lands in its own request's blocks and a ready promotion
    waits on its followers, so neither counts (both still keep the engine
    stepping through ``has_pending_push_work``). Without the write-back
    walker this is the parent's answer, unchanged."""
    wb = manager.writeback
    if wb is None:
        return manager.has_pending_stores()
    return bool(
        manager._pending_finished_stores
        or wb._soon_free()
        or any(not wb.owns(e) for e in manager._store_event_to_blocks)
        or any(not wb.owns(e) for e in manager._abandoned_store_event_to_blocks)
        or manager._abandoned_reqs_to_load
        or any(s.finished for s in manager._reqs_to_load.values())
    )
