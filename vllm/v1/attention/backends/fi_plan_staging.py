# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Fence for FlashInfer's reused plan-staging buffer (flashinfer-ai/flashinfer#6533).

The defect. Every FlashInfer wrapper owns ONE page-locked host buffer,
``_pin_memory_int_workspace_buffer``. ``plan()`` writes the schedule (request /
q-tile / kv-tile indices, o_indptr, merge_indptr, kv_chunk_size, block_valid_mask)
into it on the host and enqueues ``cudaMemcpyAsync`` into the wrapper's device int
workspace (include/flashinfer/attention/scheduler.cuh, the MATERIALIZE tail of
DecodePlan and PrefillPlanImpl), with no sync afterwards. A pinned source is read
when the GPU reaches the copy, not when it is issued. If the host plans the same
wrapper again before then, the queued copy carries the LATER plan and the earlier
launch runs with it: wrong output, or a kernel that loops for minutes
(flashinfer#6533: 5/5 mismatches on SM86 with FlashInfer 0.7.0.post1;
vllm#61239: illegal address or GPU hang under async scheduling).

Who is exposed. A blocking ``seq_lens.cpu()`` before a plan drains the stream, so
the previous copy has run. Builds that skip it can re-plan a wrapper while its copy
is still queued: MTP draft steps >= 2 (VLLM_DRAFT_SEQ_LENS_NOSYNC, on by default for
the 27B series), VLLM_TARGET_SEQ_LENS_ASYNC, and the no-speculation
``seq_lens_cpu_upper_bound`` path under async scheduling (vllm#61239).

The fence. After a plan that staged bytes, record an event on the planning stream.
Before the next plan on that wrapper overwrites the buffer, wait for that event if
it has not completed. The wait happens only when the previous copy really is still
queued, which is exactly when skipping it would be wrong; a build that already
synced pays one event query. Wrappers that carry the draft bounded-plan staging
ring (``_vllm_pin_ring``, FlashInferMetadataBuilder._bp_ring_init) rotate buffers
instead and are left alone, as are plans the builder's target staging ring routes
into one of its own slots (``RING_STAGED_ATTR``). Nothing happens while a CUDA graph
is being captured (no event may be queried there; this engine plans outside the
graph anyway).

Imports only torch and vllm.utils.gpu_sync_debug, so CPU tests can load it alone.
"""

from __future__ import annotations

from typing import Any

import torch

from vllm.utils.gpu_sync_debug import gpu_sync_allowed

# records = plans that staged bytes; waits = plans that found the previous copy of
# the same wrapper still queued (each wait is a #6533 overwrite avoided).
STATS = {"records": 0, "waits": 0}

_EVENT_ATTR = "_vllm_fi_staging_event"
# Set by FlashInferMetadataBuilder's target plan-staging ring (_PlanStagingRing in
# flashinfer.py) while a plan stages into one of its slots instead of the wrapper's
# own buffer. The ring fences each slot with its own event, so this fence (which
# guards the wrapper's own buffer) must neither wait nor record for that plan:
# waiting would stall the host on every PP + async step for a buffer the plan does
# not touch, and the event would describe a copy out of a ring slot.
RING_STAGED_ATTR = "_vllm_plan_ring_staged"


def _exempt(wrapper: Any) -> bool:
    return (
        getattr(wrapper, "_vllm_pin_ring", None) is not None
        or getattr(wrapper, RING_STAGED_ATTR, False)
        or torch.cuda.is_current_stream_capturing()
    )


def staging_fence(wrapper: Any) -> None:
    """Call right before a FlashInfer plan() / fast_decode_plan() on ``wrapper``."""
    if _exempt(wrapper):
        return
    ev = getattr(wrapper, _EVENT_ATTR, None)
    if ev is not None and not ev.query():
        STATS["waits"] += 1
        with gpu_sync_allowed():
            ev.synchronize()


def staging_record(wrapper: Any) -> None:
    """Call right after a FlashInfer plan() / fast_decode_plan() on ``wrapper``
    returned (not after a plan-cache hit: nothing was staged then)."""
    if _exempt(wrapper):
        return
    ev = getattr(wrapper, _EVENT_ATTR, None)
    if ev is None:
        ev = torch.cuda.Event()
        setattr(wrapper, _EVENT_ATTR, ev)
    # FlashInfer issues the copy on the current stream of the wrapper's device
    # (get_stream(float_workspace_buffer.device()) -> TVMFFIEnvGetStream).
    ev.record(torch.cuda.current_stream(getattr(wrapper, "device", None)))
    STATS["records"] += 1
