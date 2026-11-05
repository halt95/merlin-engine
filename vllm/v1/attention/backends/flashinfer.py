# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modified by the Merlin project; see NOTICE.
"""Attention layer with FlashInfer."""

from dataclasses import dataclass, replace
from enum import Enum
import inspect
import os
from functools import cache, partial, wraps
from typing import Any, ClassVar

import numpy as np
import torch
from flashinfer import (
    BatchAttentionWithAttentionSinkWrapper,
    BatchDecodeWithPagedKVCacheWrapper,
    BatchPrefillWithPagedKVCacheWrapper,
    BatchPrefillWithRaggedKVCacheWrapper,
    get_seq_lens,
    nvfp4_quantize_append_paged_kv_cache_with_slot_mapping,
)
from flashinfer.decode import fast_decode_plan, trtllm_batch_decode_with_kv_cache
from flashinfer.prefill import trtllm_batch_context_with_kv_cache
from flashinfer.utils import (
    FP4Tensor,
    get_device_sm_count,
    get_trtllm_gen_multi_ctas_kv_counter_bytes,
)
from typing_extensions import override

from vllm.v1.qwen35_series import series_active
from vllm.v1.qwen35_series import uniform_verify_active
from vllm import _custom_ops as custom_ops
from vllm import envs
from vllm.config import (
    CUDAGraphMode,
    VllmConfig,
    get_current_vllm_config_or_none,
)
from vllm.config.cache import CacheDType
from vllm.distributed.parallel_state import (
    get_dcp_group,
    get_dcp_world_size_and_rank,
)
from vllm.logger import init_logger
from vllm.merlin.flashinfer_smallq import (
    decode_wrapper_key,
    note_event,
    select_smallq_mode,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8StaticTensorSym,
    kNvfp4Dynamic,
)
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.triton_utils import tl, triton
from vllm.utils.flashinfer import (
    can_use_trtllm_attention,
    flashinfer_xqa_batch_decode_with_kv_cache,
    force_use_trtllm_attention,
    pin_host_range_buf,
    supports_trtllm_attention,
    use_trtllm_attention,
)
from vllm.utils.gpu_sync_debug import gpu_sync_allowed
from vllm.utils.math_utils import cdiv, round_up
from vllm.utils.torch_utils import (
    PIN_MEMORY,
    canonicalize_singleton_dim_strides,
    get_dtype_size,
    is_strictly_contiguous,
    nvfp4_kv_cache_full_dim,
    nvfp4_split_data_scale,
)
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
    max_decode_query_len,
)
from vllm.v1.attention.backends.utils import (
    get_dcp_local_seq_lens,
    get_flashinfer_layout_string,
    get_num_attention_heads_from_layers,
    get_per_layer_parameters,
    infer_global_hyperparameters,
    log2_lse_to_ln,
    split_decodes_and_prefills,
)
from vllm.v1.attention.backends.draft_bounded_plan import (
    decide_bounded_plan,
    fa2_decode_schedule,
    kv_metadata_reference,
    num_pages,
    plan_bytes_schedule,
)
# flashinfer#6533 pinned plan-buffer fence
from vllm.v1.attention.backends.fi_plan_staging import (
    RING_STAGED_ATTR,
    staging_fence,
    staging_record,
)
from vllm.v1.attention.ops.dcp import (
    cp_lse_ag_out_rs,
    dcp_a2a_lse_reduce,
    mask_dcp_empty_shards_,
)
from vllm.v1.attention.ops.merge_attn_states import merge_attn_states
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheLayout,
    KVCacheSpec,
    KVQuantMode,
    is_quantized_kv_cache,
    iter_layer_specs,
)
from vllm.v1.utils import CpuGpuBuffer

FLASHINFER_WORKSPACE_BUFFER_SIZE_BATCH_INVARIANT = 2048 * 1024 * 1024
FLASHINFER_PREFILL_WORKSPACE_BYTES_PER_ELEM = 16

FP8_DTYPE = current_platform.fp8_dtype()
FP4_DTYPE = torch.uint8

logger = init_logger(__name__)

# --- MTP draft-step seq_lens without a device sync (VLLM_DRAFT_SEQ_LENS_NOSYNC) ---
# Shared with the V2 autoregressive speculator. When enabled, every build that syncs
# seq_lens keeps its CPU copy in "last"; for draft step >= 2 the speculator puts
# last + 1 (what _update_draft_inputs_kernel does on the GPU, padded rows included) in
# "next", and the next build takes it instead of calling seq_lens.cpu().
_DRAFT_SEQ_LENS = {
    "enabled": envs.VLLM_DRAFT_SEQ_LENS_NOSYNC,
    "verify": bool(envs.VLLM_DRAFT_SEQ_LENS_VERIFY),
    "next": None,
    "last": None,
    # device seq_lens buffer the CPU vector was taken from (ownership tag)
    "next_buf": None,
    "last_buf": None,
    "bypassed": 0,
    "synced": 0,
    "verified": 0,
    "mismatch": 0,
    # --- VLLM_DRAFT_BOUNDED_PLAN: per-row [lb, ub] (int64 numpy) for the next draft build,
    # offered by the V2 autoregressive speculator, tagged with the device seq_lens buffer.
    "bounded": envs.VLLM_DRAFT_BOUNDED_PLAN,
    "bounded_verify": bool(envs.VLLM_DRAFT_BOUNDED_PLAN_VERIFY),
    "bounds": None,
    "bounds_buf": None,
    # (exact CPU seq_lens, device buffer data_ptr) of the latest plain synced build
    "exact_last": None,
    "bp_offered": 0,
    "bp_nosync": 0,
    "bp_ambiguous_sync": 0,
    "bp_ineligible_sync": 0,
    "bp_plan_check_fail": 0,
    "bp_verified": 0,
    "bp_verify_mismatch": 0,
    "bp_ring_waits": 0,
}

# --- Target (verify) build with exact CPU seq_lens and no seq_lens.cpu() ---
# VLLM_TARGET_SEQ_LENS_ASYNC: the runner hands the build a lazy handle
# (CommonAttentionMetadata.seq_lens_cpu_exact) whose values are exactly what
# seq_lens.cpu() would return; waiting on it waits for the previous step's post_update only.
_TARGET_SEQ_LENS = {
    "enabled": envs.VLLM_TARGET_SEQ_LENS_ASYNC,
    "verify": bool(envs.VLLM_TARGET_SEQ_LENS_ASYNC_VERIFY),
    "used": 0,
    "no_handle": 0,
    "ineligible": 0,
    "stale": 0,
    "verified": 0,
    "mismatch": 0,
    "fence_waits": 0,
}


def _tsl_log_counters(reason: str) -> None:
    t = _TARGET_SEQ_LENS
    logger.info(
        "target seq_lens async (%s): used %d, no-handle %d, ineligible %d, stale %d, "
        "verified %d (mismatch %d), plan-fence waits %d",
        reason, t["used"], t["no_handle"], t["ineligible"], t["stale"], t["verified"],
        t["mismatch"], t["fence_waits"])


# Pinned host staging ring for FlashInfer plans on draft wrappers (see _bp_ring_after_plan).
_BP_RING_SLOTS = 4
_BP_RING_BYTES = 256 * 1024
_BP_MAX_PADDED_BATCH = 8192  # the plan's int region must fit in _BP_RING_BYTES

# Target prefill/decode plans: rotating pinned staging (see _PlanStagingRing).
_PLAN_STAGING_SLOTS = 8
_PLAN_STAGING = {"plans": 0, "hazards": 0, "waits": 0}


class _PlanStagingRing:
    """Rotating page-locked staging buffers for every target FlashInfer plan().

    FlashInfer's planner writes the schedule (request/kv-tile indices, o_indptr,
    kv chunk sizes) into the wrapper's ONE pinned host buffer and enqueues an async
    H2D copy of it on the current stream (scheduler.cuh DecodePlanImpl /
    PrefillPlanImpl: cudaMemcpyAsync(int_buffer, page_locked_int_buffer, ...)).
    Nothing fences that buffer, so when the CPU plans step N+1 before the GPU has
    executed step N's copy, step N's kernels read step N+1's schedule against step
    N's own plan_info offsets. With MRV2 + PP + async scheduling the batch queue is
    pp_size + 1 deep and each stage's stream waits on the other stage before it
    reaches the copy, so the CPU routinely plans two steps ahead of the copy.
    Each plan stages into its own slot; a slot is reused only after the copy that
    read it has executed (per-slot event; waits only if the ring is lapped).
    Draft wrappers keep their own ring (_bp_ring_*); they are skipped here.
    """

    def __init__(self, slots: int = _PLAN_STAGING_SLOTS):
        self._bufs: list[torch.Tensor | None] = [None] * slots
        self._events: list[torch.cuda.Event | None] = [None] * slots
        self._idx = 0
        self._last_event: torch.cuda.Event | None = None
        self._pending: list[tuple] = []

    def stage(self, wrapper) -> tuple | None:
        own = getattr(wrapper, "_pin_memory_int_workspace_buffer", None)
        if own is None or getattr(wrapper, "_vllm_pin_ring", None) is not None:
            return None
        i = self._idx
        ev = self._events[i]
        if ev is not None and not ev.query():
            _PLAN_STAGING["waits"] += 1
            with gpu_sync_allowed():
                ev.synchronize()
        buf = self._bufs[i]
        if buf is None or buf.numel() < own.numel():
            buf = self._bufs[i] = torch.empty(
                own.numel(), dtype=own.dtype, pin_memory=True
            )
        wrapper._pin_memory_int_workspace_buffer = buf
        # The #6533 fence (fi_plan_staging) guards the wrapper's own buffer only;
        # this plan stages into slot i, which the slot event above fences.
        setattr(wrapper, RING_STAGED_ATTR, True)
        token = (i, own, getattr(wrapper, "_plan_info", None))
        self._pending.append((wrapper, token))
        return token

    def abort_pending(self) -> None:
        """Restore the own buffer of every wrapper still staged (plan() raised
        before commit()); else the next stage() would adopt the ring slot as "own"."""
        pending, self._pending = self._pending, []
        for wrapper, token in pending:
            wrapper._pin_memory_int_workspace_buffer = token[1]
            setattr(wrapper, RING_STAGED_ATTR, False)

    def commit(self, wrapper, token: tuple | None) -> None:
        if token is None:
            return
        i, own, plan_info_before = token
        self._pending = [(w, t) for w, t in self._pending if t is not token]
        # Unhooked plan() calls keep using the wrapper's own buffer.
        wrapper._pin_memory_int_workspace_buffer = own
        setattr(wrapper, RING_STAGED_ATTR, False)
        if wrapper._plan_info is plan_info_before:
            return  # plan cache hit: nothing was staged
        d = _PLAN_STAGING
        d["plans"] += 1
        if self._last_event is not None and not self._last_event.query():
            # The old single buffer would have been overwritten before this copy ran.
            d["hazards"] += 1
        ev = self._events[i]
        if ev is None:
            ev = self._events[i] = torch.cuda.Event()
        ev.record()
        self._last_event = ev
        self._idx = (i + 1) % len(self._bufs)
        if d["plans"] in (1000, 10000) or d["plans"] % 100000 == 0:
            logger.info(
                "flashinfer plan staging: %d plans, %d single-buffer hazards, "
                "%d ring waits",
                d["plans"], d["hazards"], d["waits"],
            )


def _abort_plan_staging_on_error(build):
    @wraps(build)
    def wrapper(self, *args, **kwargs):
        try:
            return build(self, *args, **kwargs)
        finally:
            self._plan_staging.abort_pending()

    return wrapper


class _BoundedDraftPlan:
    """State of one draft build planned from CPU bounds (VLLM_DRAFT_BOUNDED_PLAN)."""

    __slots__ = ("lb", "ub", "ub_cpu", "decision", "wrapper", "q_tiles", "max_split",
                 "min_chunk")

    def __init__(self, lb, ub, ub_cpu, decision, wrapper, q_tiles, max_split, min_chunk):
        self.lb = lb
        self.ub = ub
        self.ub_cpu = ub_cpu
        self.decision = decision
        self.wrapper = wrapper
        self.q_tiles = q_tiles
        self.max_split = max_split
        self.min_chunk = min_chunk


def _bp_log_counters(reason: str) -> None:
    d = _DRAFT_SEQ_LENS
    logger.info(
        "draft bounded plan (%s): offered %d, no-sync %d, ambiguous->sync %d, "
        "ineligible->sync %d, plan-check fail %d, verified %d (mismatch %d), ring waits %d",
        reason, d["bp_offered"], d["bp_nosync"], d["bp_ambiguous_sync"],
        d["bp_ineligible_sync"], d["bp_plan_check_fail"], d["bp_verified"],
        d["bp_verify_mismatch"], d["bp_ring_waits"])


def _draft_seq_lens_publish(staged) -> None:
    """Publish a successful build's CPU seq_lens (+ device buffer tag) as "last"."""
    if staged is not None:
        _DRAFT_SEQ_LENS["last"], _DRAFT_SEQ_LENS["last_buf"] = staged


trtllm_workspace_buffer = None


def _nvfp4_kv_on_fa2() -> bool:
    return any(current_platform.is_device_capability_family(f) for f in (80, 90, 120))


def _use_fa2_native_decode() -> bool:
    # Below SM90, native decode runs fa2, which takes uniform multi-token queries.
    return not current_platform.has_device_capability(90)


_KVPair = tuple[torch.Tensor, torch.Tensor]


def _nvfp4_kv_views(
    kv_cache: torch.Tensor, num_kv_heads: int
) -> tuple[_KVPair, _KVPair, str]:
    """Views of NVFP4 pages laid out as [K data | K scale | V data | V scale]."""
    num_blocks, num_slots, block_size, full_dim = kv_cache.shape
    hnd = kv_cache.stride()[1:] == (block_size * full_dim, full_dim, 1)
    if not hnd and kv_cache.stride()[1:] != (full_dim, num_slots * full_dim, 1):
        raise ValueError(f"NVFP4 KV pages must be dense, got {kv_cache.stride()}")
    dims = (num_kv_heads, block_size) if hnd else (block_size, num_kv_heads)
    side_size = num_kv_heads * block_size * full_dim
    (k_data, k_scale), (v_data, v_scale) = (
        nvfp4_split_data_scale(
            kv_cache.as_strided(
                (num_blocks, *dims, full_dim),
                (kv_cache.stride(0), dims[1] * full_dim, full_dim, 1),
                kv_cache.storage_offset() + side * side_size,
            )
        )
        for side in range(2)
    )
    return (k_data, v_data), (k_scale, v_scale), "HND" if hnd else "NHD"


def _get_trtllm_workspace_buffer():
    global trtllm_workspace_buffer
    if trtllm_workspace_buffer is None:
        trtllm_workspace_buffer = torch.zeros(
            envs.VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE, dtype=torch.uint8, device="cuda"
        )
    return trtllm_workspace_buffer


_fi_multi_ctas_kv_counter: torch.Tensor | None = None


def _get_multi_ctas_kv_counter_buffer(
    min_bytes: int, device: torch.device
) -> torch.Tensor:
    """Shared multi-CTA-KV counter buffer; the kernels re-zero it after each launch."""
    global _fi_multi_ctas_kv_counter
    if (
        _fi_multi_ctas_kv_counter is None
        or _fi_multi_ctas_kv_counter.numel() < min_bytes
    ):
        _fi_multi_ctas_kv_counter = torch.zeros(
            min_bytes, dtype=torch.uint8, device=device
        )
    return _fi_multi_ctas_kv_counter


def _pack_draft_block_bool_mask(
    bool_mask: torch.Tensor, num_packed: int
) -> torch.Tensor:
    """Pack a boolean draft mask into XQA's uint16 layout."""
    num_rows = bool_mask.shape[0]
    bits = 1 << torch.arange(32, device=bool_mask.device, dtype=torch.int64)
    mask_u32 = (
        (bool_mask.view(num_rows, num_packed, 32).to(torch.int64) * bits)
        .sum(dim=-1)
        .to(torch.uint32)
    )
    return mask_u32.view(torch.uint16).reshape(num_rows, num_packed * 2)


def _make_xqa_draft_block_mask(
    q_len: int, causal: bool, device: torch.device
) -> torch.Tensor:
    """Build a packed XQA draft mask for a uniform request."""
    num_packed = (q_len + 31) // 32
    padded = num_packed * 32
    q_idx = torch.arange(q_len, device=device).unsqueeze(1)
    kv_idx = torch.arange(padded, device=device).unsqueeze(0)
    bool_mask = kv_idx <= q_idx if causal else (kv_idx < q_len).expand(q_len, padded)
    return _pack_draft_block_bool_mask(bool_mask, num_packed)


def _make_xqa_ragged_draft_block_mask(
    q_lens: list[int], max_q_len: int, causal: bool, device: torch.device
) -> torch.Tensor:
    """Build a packed XQA draft mask for a ragged batch."""
    num_packed = (max_q_len + 31) // 32
    padded = num_packed * 32
    q_lens_t = torch.tensor(q_lens, device=device)
    row_lens = torch.repeat_interleave(q_lens_t, q_lens_t)
    request_starts = torch.cumsum(q_lens_t, dim=0) - q_lens_t
    row_starts = torch.repeat_interleave(request_starts, q_lens_t)
    q_idx = torch.arange(sum(q_lens), device=device) - row_starts
    kv_idx = torch.arange(padded, device=device).unsqueeze(0)
    bool_mask = kv_idx < row_lens.unsqueeze(1)
    if causal:
        bool_mask &= kv_idx <= q_idx.unsqueeze(1)
    return _pack_draft_block_bool_mask(bool_mask, num_packed)


@triton.jit
def _trtllm_prefill_attn_kvfp8_dequant(
    kv_cache_ptr,
    block_tables_prefill_ptr,
    block_table_stride,
    mock_kv_cache_ptr,
    k_scale_ptr,
    v_scale_ptr,
    src_stride_page,
    src_stride_kv,
    src_stride_head,
    src_stride_block,
    src_stride_head_size,
    DST_K_CACHE_STRIDE: tl.constexpr,
    DST_KV_CACHE_STRIDE: tl.constexpr,
    HEAD_STRIDE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
):
    batch_idx = tl.program_id(0).to(tl.int64)
    mock_block_table_idx = tl.program_id(1).to(tl.int64)
    orig_page_num = tl.load(
        block_tables_prefill_ptr + batch_idx * block_table_stride + mock_block_table_idx
    ).to(tl.int64)
    if orig_page_num <= 0:
        return
    dequant_dtype = mock_kv_cache_ptr.dtype.element_ty

    k_scale_val = tl.load(k_scale_ptr)
    v_scale_val = tl.load(v_scale_ptr)

    mock_page_idx = batch_idx * block_table_stride + mock_block_table_idx + 1
    logical_offsets = tl.arange(0, HEAD_STRIDE)
    block_offsets = logical_offsets // HEAD_SIZE
    head_size_offsets = logical_offsets % HEAD_SIZE

    for h in range(NUM_KV_HEADS):
        h_off = tl.cast(h, tl.int64)

        # Read K from source (supports non-contiguous page/kv/head strides)
        src_k = (
            orig_page_num * src_stride_page
            + h_off * src_stride_head
            + block_offsets * src_stride_block
            + head_size_offsets * src_stride_head_size
        )
        fp8_k = tl.load(kv_cache_ptr + src_k)
        dequant_k = (fp8_k.to(tl.float32) * k_scale_val).to(dequant_dtype)

        # Write K to contiguous mock cache
        dst_k = mock_page_idx * DST_KV_CACHE_STRIDE + h * HEAD_STRIDE + logical_offsets
        tl.store(mock_kv_cache_ptr + dst_k, dequant_k)

        # Read V from source (offset by src_stride_kv for the V half)
        src_v = (
            orig_page_num * src_stride_page
            + src_stride_kv
            + h_off * src_stride_head
            + block_offsets * src_stride_block
            + head_size_offsets * src_stride_head_size
        )
        fp8_v = tl.load(kv_cache_ptr + src_v)
        dequant_v = (fp8_v.to(tl.float32) * v_scale_val).to(dequant_dtype)

        # Write V to contiguous mock cache
        dst_v = (
            mock_page_idx * DST_KV_CACHE_STRIDE
            + DST_K_CACHE_STRIDE
            + h * HEAD_STRIDE
            + logical_offsets
        )
        tl.store(mock_kv_cache_ptr + dst_v, dequant_v)


def trtllm_prefill_attn_kvfp8_dequant(
    kv_cache: torch.Tensor,
    block_tables_prefill: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    dequant_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, num_of_page_per_token = block_tables_prefill.shape
    s = kv_cache.shape
    assert s[1] == 2
    assert dequant_dtype in (torch.bfloat16, torch.float16)

    # Logical source layout is (B, 2, H, N, D). The tensor may be a
    # non-contiguous view, so the Triton kernel indexes it with actual strides.
    strides = kv_cache.stride()
    num_kv_heads, block_size, head_size = s[2], s[3], s[4]
    head_stride = block_size * head_size
    k_cache_stride = num_kv_heads * head_stride
    kv_cache_stride = k_cache_stride * s[1]

    new_s = (batch_size * num_of_page_per_token + 1, s[1], s[2], s[3], s[4])
    # mock kv cache contains just the pages needed by this prefill
    mock_kv_cache = torch.empty(new_s, dtype=dequant_dtype, device=kv_cache.device)
    # we simply sequentially index the pages needed by this prefill
    mock_block_table = torch.arange(
        start=1,
        end=batch_size * num_of_page_per_token + 1,
        dtype=torch.int32,
        device=block_tables_prefill.device,
    ).reshape(batch_size, num_of_page_per_token)
    grid = (batch_size, num_of_page_per_token)
    _trtllm_prefill_attn_kvfp8_dequant[grid](
        kv_cache,
        block_tables_prefill,
        num_of_page_per_token,
        mock_kv_cache,
        k_scale,
        v_scale,
        strides[0],
        strides[1],
        strides[2],
        strides[3],
        strides[4],
        k_cache_stride,
        kv_cache_stride,
        head_stride,
        head_size,
        num_kv_heads,
    )
    return mock_kv_cache, mock_block_table


class TrtllmDCPContextWrapper:
    """trtllm-gen stand-in for the native wrapper in the DCP context pass."""

    # Plan state checked by FlashInferImpl.forward(), as on the native wrapper.
    _window_left = -1
    _logits_soft_cap = 0.0
    _causal = False

    @staticmethod
    def fits_workspace(qo_indptr_cpu: torch.Tensor, num_qo_heads: int) -> bool:
        # Mirrors the LSE softmax-stats carve-out in FlashInfer's trtllm-gen
        # launcher: heads * batch * round_up(max_q_len, 256) float2s + 1 MiB.
        batch_size = qo_indptr_cpu.shape[0] - 1
        max_q_len = int((qo_indptr_cpu[1:] - qo_indptr_cpu[:-1]).max())
        stats_bytes = num_qo_heads * batch_size * round_up(max_q_len, 256) * 8
        return stats_bytes + (1 << 20) <= _get_trtllm_workspace_buffer().nbytes

    def plan(
        self,
        *,
        qo_indptr: torch.Tensor,
        paged_kv_indptr: torch.Tensor,
        seq_lens: torch.Tensor,
        block_tables: torch.Tensor,
        causal: bool,
        sm_scale: float,
        window_left: int,
        logits_soft_cap: float | None,
        **kwargs,
    ) -> None:
        assert not causal and window_left == -1 and not logits_soft_cap
        device = block_tables.device
        self._block_tables = block_tables
        self._seq_lens = seq_lens.to(device, non_blocking=True)
        self._cum_seq_lens_q = qo_indptr.to(device, non_blocking=True)
        self._cum_seq_lens_kv = paged_kv_indptr.to(device, non_blocking=True)
        self._max_q_len = int((qo_indptr[1:] - qo_indptr[:-1]).max())
        self._max_kv_len = int(seq_lens.max())
        self._has_empty_ctx = bool((seq_lens == 0).any())
        self._sm_scale = sm_scale

    def run(
        self,
        q: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor],
        k_scale: float,
        v_scale: float,
        return_lse: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert return_lse
        q = canonicalize_singleton_dim_strides(q.contiguous())
        # trtllm-gen writes neither the output nor the LSE of requests without
        # local context; the combine weights them by 0, so the output must be
        # finite and the LSE -inf.
        out = torch.zeros_like(q) if self._has_empty_ctx else torch.empty_like(q)
        lse = torch.empty(q.shape[:2], dtype=torch.float32, device=q.device)
        if self._max_kv_len > 0:
            trtllm_batch_context_with_kv_cache(
                query=q,
                kv_cache=kv_cache,
                workspace_buffer=_get_trtllm_workspace_buffer(),
                block_tables=self._block_tables,
                seq_lens=self._seq_lens,
                max_q_len=self._max_q_len,
                max_kv_len=self._max_kv_len,
                bmm1_scale=self._sm_scale * k_scale,
                bmm2_scale=v_scale,
                batch_size=self._seq_lens.shape[0],
                cum_seq_lens_q=self._cum_seq_lens_q,
                cum_seq_lens_kv=self._cum_seq_lens_kv,
                causal=False,
                return_lse=True,
                lse=lse,
                out=out,
            )
        if self._has_empty_ctx:
            mask_dcp_empty_shards_(lse, self._seq_lens, self._cum_seq_lens_q)
        return out, lse


class BatchDCPPrefillWrapper:
    def __init__(
        self,
        kv_layout: str,
        workspace_buffer: torch.Tensor | None = None,
        dcp_a2a: bool = False,
        use_trtllm_context: bool = False,
    ):
        if dcp_a2a:
            self._dcp_combine = partial(dcp_a2a_lse_reduce, is_lse_base_on_e=False)
        else:
            self._dcp_combine = partial(cp_lse_ag_out_rs, is_lse_base_on_e=False)
        self._native_context = BatchPrefillWithPagedKVCacheWrapper(
            workspace_buffer, kv_layout
        )
        self._trtllm_context = TrtllmDCPContextWrapper() if use_trtllm_context else None
        # The context-pass wrapper planned for this step.
        self._context: BatchPrefillWithPagedKVCacheWrapper | TrtllmDCPContextWrapper
        self._new_tokens = BatchPrefillWithRaggedKVCacheWrapper(workspace_buffer)

    def plan(
        self,
        qo_indptr_cpu: torch.Tensor,
        paged_kv_indptr_cpu: torch.Tensor,
        paged_kv_indices: torch.Tensor,
        paged_kv_last_page_len_cpu: torch.Tensor,
        kv_lens_cpu: torch.Tensor,
        page_size: int,
        num_qo_heads: int,
        dcp_world_size: int,
        num_kv_heads: int,
        head_dim: int,
        sm_scale: float,
        window_left: int,
        logits_soft_cap: float | None,
        q_data_type: torch.dtype,
        kv_cache_dtype: torch.dtype,
        prefill_fixed_split_size: int,
        disable_split_kv: bool,
        block_tables: torch.Tensor,
    ):
        """Plan the prefill operation with given parameters."""
        self._context = self._native_context
        if self._trtllm_context is not None and (
            TrtllmDCPContextWrapper.fits_workspace(
                qo_indptr_cpu, num_qo_heads * dcp_world_size
            )
        ):
            self._context = self._trtllm_context
        staging_fence(self._context)
        self._context.plan(
            qo_indptr=qo_indptr_cpu,
            paged_kv_indptr=paged_kv_indptr_cpu,
            paged_kv_indices=paged_kv_indices,
            paged_kv_last_page_len=paged_kv_last_page_len_cpu,
            seq_lens=kv_lens_cpu,
            num_qo_heads=num_qo_heads * dcp_world_size,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim,
            page_size=page_size,
            causal=False,  # This is context run
            sm_scale=sm_scale,
            window_left=window_left,
            logits_soft_cap=logits_soft_cap,
            q_data_type=q_data_type,
            kv_data_type=kv_cache_dtype,
            fixed_split_size=prefill_fixed_split_size,
            disable_split_kv=disable_split_kv,
            block_tables=block_tables,
        )
        staging_record(self._context)
        staging_fence(self._new_tokens)
        self._new_tokens.plan(
            qo_indptr=qo_indptr_cpu,
            kv_indptr=qo_indptr_cpu,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim,
            head_dim_vo=head_dim,
            causal=True,  # This is newtokens run
            sm_scale=sm_scale,
            window_left=window_left,
            logits_soft_cap=logits_soft_cap,
            q_data_type=q_data_type,
        )
        staging_record(self._new_tokens)

    def run(
        self,
        layer: torch.nn.Module,
        prefill_query: torch.Tensor,
        kv_cache_tuple: tuple[torch.Tensor, torch.Tensor],
        key: torch.Tensor,
        value: torch.Tensor,
        out: torch.Tensor,
    ):
        prefill_query_across_dcp = get_dcp_group().all_gather(
            prefill_query.contiguous(), dim=1
        )
        output_context_tmp, lse_context_tmp = self._context.run(
            prefill_query_across_dcp,
            kv_cache_tuple,
            k_scale=layer._k_scale_float,
            v_scale=layer._v_scale_float,
            return_lse=True,
        )
        output_context, lse_context = self._dcp_combine(
            output_context_tmp,
            lse_context_tmp,
            get_dcp_group(),
            return_lse=True,
        )
        lse_context = log2_lse_to_ln(lse_context.transpose(0, 1).contiguous())

        output_query, lse_query = self._new_tokens.run(
            prefill_query,
            key,
            value,
            return_lse=True,
        )
        lse_query = log2_lse_to_ln(lse_query.transpose(0, 1).contiguous())

        merge_attn_states(
            out,
            output_context,
            lse_context,
            output_query,
            lse_query,
        )
        return out


class FlashInferBackend(AttentionBackend):
    supports_kv_cache_update_bypass: bool = True

    @classmethod
    def customize_spec(cls, spec: "AttentionSpec") -> "AttentionSpec":
        """NVFP4 stores K and V as separate per-head slots of packed fp4 data
        plus fp8 block scales."""
        if spec.state_content_bytes is not None or not spec.kv_quant_mode.is_nvfp4:
            return spec
        hs_k = nvfp4_kv_cache_full_dim(spec.head_size)
        hs_v = nvfp4_kv_cache_full_dim(spec.head_size_v)
        assert hs_k == hs_v, "nvfp4 with asymmetric K/V head sizes not yet supported"
        return replace(
            spec,
            num_head_slots=2 * spec.num_kv_heads,
            state_content_bytes=hs_k * get_dtype_size(spec.dtype),
        )

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
        "fp8_e5m2",
        "nvfp4",
        "nvfp4_4over6",
    ]

    @staticmethod
    def get_supported_kernel_block_sizes(kv_cache_spec=None) -> list[int | MultipleOf]:
        # Page sizes >= 128 only run on the trtllm-gen dynamic kernel (GQA/MQA
        # on Blackwell); advertise them only when usable so selection never
        # picks a large kernel block we cannot serve.
        use_large_pages = False
        vllm_config = get_current_vllm_config_or_none()
        if vllm_config is not None and vllm_config.model_config is not None:
            pc = vllm_config.parallel_config
            mc = vllm_config.model_config
            num_qo_heads = mc.get_num_attention_heads(pc)
            num_kv_heads = mc.get_num_kv_heads(pc)
            use_large_pages = (
                num_kv_heads > 0
                and num_qo_heads // num_kv_heads > 1
                and current_platform.is_device_capability_family(100)
                and can_use_trtllm_attention(num_qo_heads, num_kv_heads)
            )
        if not use_large_pages:
            return [16, 32, 64]
        return [16, 32, 64, 128, 256, 512, 1024]

    @staticmethod
    def get_name() -> str:
        return "FLASHINFER"

    @classmethod
    def supports_non_causal(cls) -> bool:
        return True

    @classmethod
    def supports_device_cpu_query_lens_mismatch(cls) -> bool:
        # Only trtllm-gen decode (SM100 family) reads query offsets from the
        # device tensor; XQA and wrapper decode plan from qo_indptr_cpu, and
        # under DCP spec queries are routed to the prefill path.
        if not current_platform.is_device_capability_family(100):
            return False
        vllm_config = get_current_vllm_config_or_none()
        if vllm_config is None or vllm_config.model_config is None:
            return True
        pc = vllm_config.parallel_config
        mc = vllm_config.model_config
        return pc.decode_context_parallel_size == 1 and can_use_trtllm_attention(
            mc.get_num_attention_heads(pc), mc.get_num_kv_heads(pc), is_prefill=False
        )

    @classmethod
    def supports_sliding_window(cls) -> bool:
        return True

    @staticmethod
    def get_impl_cls() -> type["FlashInferImpl"]:
        return FlashInferImpl

    @staticmethod
    def get_builder_cls() -> type["FlashInferMetadataBuilder"]:
        return FlashInferMetadataBuilder

    @staticmethod
    def get_dtype_for_flashinfer(kv_cache_dtype: str) -> torch.dtype:
        if kv_cache_dtype in ("fp8", "fp8_e4m3"):
            return torch.float8_e4m3fn
        elif kv_cache_dtype == "fp8_e5m2":
            return torch.float8_e5m2
        elif kv_cache_dtype.startswith("nvfp4"):
            return torch.uint8
        else:
            raise ValueError(f"Unrecognized dtype: {kv_cache_dtype}")

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype: CacheDType | None) -> bool:
        if kv_cache_dtype is not None and kv_cache_dtype.startswith("nvfp4"):
            if _nvfp4_kv_on_fa2():
                # The slot-mapping writer has no scale search.
                return kv_cache_dtype == "nvfp4"
            return (
                current_platform.is_device_capability_family(100)
                and supports_trtllm_attention(is_prefill=True)
                and supports_trtllm_attention(is_prefill=False)
            )
        return super().supports_kv_cache_dtype(kv_cache_dtype)

    @classmethod
    def supports_combination(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: CacheDType | None,
        block_size: int | None,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        use_mm_prefix: bool,
        device_capability: DeviceCapability,
    ) -> str | None:
        if (
            kv_cache_dtype is not None
            and kv_cache_dtype.startswith("nvfp4")
            and device_capability.major in (8, 9, 12)
        ):
            if head_size == 64:
                return "fa2 prefill misreads an NVFP4 KV cache at head_size 64"
            if has_sink:
                return "the sink prefill wrapper cannot read an NVFP4 KV cache"
        return None

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        # https://github.com/flashinfer-ai/flashinfer/blob/3d55c71a62052c590c130897d3a3db49b14fcc34/include/flashinfer/utils.cuh#L157
        return [64, 128, 256, 512]

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        # FlashInfer supports SM75+, but is currently broken on SM75 (Turing):
        # https://github.com/flashinfer-ai/flashinfer/issues/3620 (fix:
        # https://github.com/flashinfer-ai/flashinfer/pull/3621). Temporarily
        # raise the floor to SM80 so it is not auto-selected on SM75 until
        # that fix lands; revert to DeviceCapability(7, 5) once it does.
        return capability >= DeviceCapability(8, 0) and capability <= DeviceCapability(
            12, 1
        )

    @classmethod
    def supports_sink(cls) -> bool:
        """Whether FlashInfer can serve attention sinks on this platform."""
        from vllm.utils.flashinfer import (
            force_use_trtllm_attention,
        )

        if force_use_trtllm_attention() is False:
            return False

        if current_platform.is_device_capability(
            90
        ) or current_platform.is_device_capability_family(120):
            return supports_trtllm_attention(is_prefill=False)

        if not current_platform.is_device_capability_family(100):
            return False

        return supports_trtllm_attention(
            is_prefill=False
        ) and supports_trtllm_attention(is_prefill=True)

    @classmethod
    def supported_kv_cache_layouts(cls) -> tuple[KVCacheLayout, ...] | None:
        capability = current_platform.get_device_capability()
        if capability is not None and capability.major == 10:
            # The trtllm-gen kernels consume head-major block interiors; the L/B
            # nesting outside the block is immaterial to them.
            return (KVCacheLayout.LBHNC, KVCacheLayout.BLHNC)
        return super().supported_kv_cache_layouts()

    forward_includes_kv_cache_update: bool = False


@dataclass
class FIPrefill:
    """Metadata for the native FlashInfer prefill pathway (non-TRTLLM)."""

    wrapper: BatchPrefillWithPagedKVCacheWrapper | BatchDCPPrefillWrapper


@dataclass
class FIDecode:
    """Metadata for the native FlashInfer decode pathway (non-TRTLLM)."""

    wrapper: BatchDecodeWithPagedKVCacheWrapper


class FlashInferDecodeKernel(Enum):
    """Decode kernels selected inside the FlashInfer backend."""

    XQA = "xqa"
    TRTLLM_GEN = "trtllm-gen"


def _is_xqa_head_dim_supported(head_dim: int) -> bool:
    return 16 <= head_dim <= 256 and head_dim % 16 == 0


@dataclass
class TRTLLMPrefill:
    """Metadata for the TRTLLM prefill pathway."""

    block_tables: torch.Tensor
    """
    The slice of the block table tensor corresponding *only* to prefill requests.
    Shape: [num_prefills, max_num_blocks_per_seq]
    """

    seq_lens: torch.Tensor
    """
    The slice of the sequence lengths tensor corresponding *only* to prefill requests.
    Shape: [num_prefills]
    """

    cum_seq_lens_q: torch.Tensor
    cum_seq_lens_kv: torch.Tensor

    max_q_len: int
    """
    The maximum query length *among prefill requests*.
    """

    max_seq_len: int
    """The maximum sequence length for KV Cache."""


@dataclass
class FlashInferTrtllmAPIDecode:
    """Metadata for XQA and trtllm-gen decode."""

    kernel: FlashInferDecodeKernel

    block_tables: torch.Tensor
    """
    The slice of the block table tensor corresponding *only* to decode requests.
    Shape: [num_decodes, max_num_blocks_per_seq]
    """

    seq_lens: torch.Tensor
    """
    The slice of the sequence lengths tensor corresponding *only* to decode requests.
    Shape: [num_decodes]
    """

    max_seq_len: int
    """The maximum sequence length for KV Cache."""

    q_len_per_req: int = 1
    """Query tokens per request for uniform speculative decode."""

    q_cu_seq_lens: torch.Tensor | None = None
    """Cumulative query lengths for ragged XQA decode."""

    mask: torch.Tensor | None = None
    """Packed XQA draft mask."""

    dcp_query_start_loc: torch.Tensor | None = None
    """Decode query boundaries for masking empty DCP shards during reduction."""


@dataclass
class FlashInferMetadata:
    kv_cache_layout: KVCacheLayout
    num_actual_tokens: int
    """Total number of tokens in the batch (excluding padding)."""

    slot_mapping: torch.Tensor
    """Tensor for writing K/V to the cache. Shape: [num_actual_tokens]"""

    # The data types of the query for prefill and decode.
    # On SM90, these two data types may be different.
    q_data_type_prefill: torch.dtype
    q_data_type_decode: torch.dtype

    num_decodes: int
    num_decode_tokens: int
    num_prefills: int
    num_prefill_tokens: int
    causal: bool

    prefill: FIPrefill | TRTLLMPrefill | None
    """
    Holds the metadata for the prefill portion of the batch.
    Will be `None` if `num_prefill_tokens == 0`.
    """

    decode: FIDecode | FlashInferTrtllmAPIDecode | None
    """
    Holds the metadata for the decode portion of the batch.
    Will be `None` if `num_decode_tokens == 0`.
    """

    flashinfer_smallq_mode: int = -1


class FlashInferMetadataBuilder(AttentionMetadataBuilder[FlashInferMetadata]):
    kv_cache_spec: AttentionSpec
    reorder_batch_threshold: int = 1

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        # 27B series patches are family/device-gated (dense Qwen3.5 on sm_86 only),
        # asked about this builder's own execution device.
        self._qwen35_series = series_active(vllm_config, device=device)
        if (
            envs.MERLIN_FLASHINFER_WORKLOAD_TILE
            and not vllm_config.use_v2_model_runner
        ):
            raise ValueError("MERLIN_FLASHINFER_WORKLOAD_TILE requires the V2 runner")
        self._workload_tile = (
            self._qwen35_series and envs.MERLIN_FLASHINFER_WORKLOAD_TILE
        )
        if self._workload_tile:
            api = getattr(
                BatchDecodeWithPagedKVCacheWrapper, "sm8x_smallq_plan_api", 0
            )
            if api not in (1, 2):
                raise RuntimeError(
                    "MERLIN_FLASHINFER_WORKLOAD_TILE requires FlashInfer patch 0002"
                )
            if vllm_config.parallel_config.use_ubatching:
                raise ValueError(
                    "MERLIN_FLASHINFER_WORKLOAD_TILE requires ubatching disabled"
                )
        # FLASHINFER_SM8X_FP8_SMALLQ belongs to the patched FlashInfer (its
        # planner reads it); it is read here only to mirror that tile choice,
        # so it is not a vLLM or Merlin registry knob.
        self._smallq_cta32 = self._workload_tile and (
            os.environ.get("FLASHINFER_SM8X_FP8_SMALLQ") == "cta32"
        )
        self._decode_wrappers_smallq: dict[int, Any] = {}
        # The wider uniform-verify gate (it also admits Qwen3.5 MoE with
        # MERLIN_MOE_UNIFORM_VERIFY=1); only the spec-as-decode clause below asks it.
        self._uniform_verify = uniform_verify_active(vllm_config, device=device)
        # VLLM_DRAFT_BOUNDED_PLAN (same family/device gate as the other series patches);
        # cleared for good if a plan ever disagrees with the CPU schedule replica.
        self._bounded_plan_ok = self._qwen35_series and bool(
            envs.VLLM_DRAFT_BOUNDED_PLAN
        )
        self._bp_fired = False
        # VLLM_TARGET_SEQ_LENS_ASYNC (same family/device gate as the other series patches);
        # cleared for good on a VERIFY mismatch.
        self._tsl_ok = self._qwen35_series and bool(envs.VLLM_TARGET_SEQ_LENS_ASYNC)
        self._tsl_fired = False
        self.cache_config = vllm_config.cache_config
        self.model_config = vllm_config.model_config
        self.attention_config = vllm_config.attention_config
        self._num_speculative_tokens = vllm_config.num_speculative_tokens
        self._workspace_buffer = None
        self._prefill_wrapper: (
            BatchPrefillWithPagedKVCacheWrapper | BatchDCPPrefillWrapper | None
        ) = None  # Wrapper for prefill/append
        self._noncausal_prefill_wrapper: BatchPrefillWithPagedKVCacheWrapper | None = (
            None  # Wrapper for non-causal prefill (DFlash)
        )
        self._decode_wrapper = None  # Wrapper for decode (general shape)

        if envs.VLLM_BATCH_INVARIANT:
            self.decode_fixed_split_size = 2048
            self.prefill_fixed_split_size = 4096
            self.disable_split_kv = True
        else:
            self.decode_fixed_split_size = -1
            self.prefill_fixed_split_size = -1
            self.disable_split_kv = False

        self.compilation_config = vllm_config.compilation_config
        self.max_num_batched_tokens = (
            vllm_config.scheduler_config.max_num_batched_tokens
        )
        max_num_pages_per_req = cdiv(
            self.model_config.max_model_len, self.kv_cache_spec.block_size
        )
        max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        self.max_num_reqs = max_num_reqs
        max_num_pages = max_num_reqs * max_num_pages_per_req
        # Persistent uniform masks keep stable addresses for CUDA graphs.
        self._decode_mask_cache: dict[tuple[int, bool], torch.Tensor] = {}
        speculative_config = vllm_config.speculative_config
        num_spec_tokens = (
            speculative_config.num_speculative_tokens
            if speculative_config is not None
            else 0
        )
        self.enable_cuda_graph = (
            self.compilation_config.cudagraph_mode.decode_mode() == CUDAGraphMode.FULL
        )
        if self.enable_cuda_graph:
            # For full cudagraph capture, one `decode_wrapper` for each batch
            # size and query length is needed for FlashInfer.
            self._decode_wrappers_cudagraph: dict[
                tuple[int, int] | tuple[int, int, int],
                BatchDecodeWithPagedKVCacheWrapper,
            ] = {}
            self._decode_cudagraph_max_bs = (1 + num_spec_tokens) * max_num_reqs
            if self.compilation_config.max_cudagraph_capture_size is not None:
                self._decode_cudagraph_max_bs = min(
                    self._decode_cudagraph_max_bs,
                    self.compilation_config.max_cudagraph_capture_size,
                )
        self.dcp_world_size, self.dcp_rank = get_dcp_world_size_and_rank(
            kv_cache_spec.dcp_sharded
        )
        self.dcp_kv_cache_interleave_size = (
            vllm_config.parallel_config.cp_kv_cache_interleave_size
        )
        self.use_dcp = self.dcp_world_size > 1
        self.dcp_a2a = (
            self.use_dcp and vllm_config.parallel_config.dcp_comm_backend == "a2a"
        )

        # Compatible with models with non-uniform per-layer head counts.
        self.num_qo_heads = get_num_attention_heads_from_layers(
            vllm_config, layer_names
        ) or self.model_config.get_num_attention_heads(self.vllm_config.parallel_config)

        self.num_kv_heads = self.kv_cache_spec.num_kv_heads
        self.head_dim = self.kv_cache_spec.head_size
        self.page_size = self.kv_cache_spec.block_size

        if self.kv_cache_spec.kv_quant_mode != KVQuantMode.NONE:
            self.cache_dtype = self.cache_config.cache_dtype
            # Cannot use self.kv_cache_spec.dtype here because kv_cache_spec
            # storage dtype may not be the same as the op dtype (uint8 vs fp8_e4m3)
            self.is_kvcache_nvfp4 = self.cache_dtype.startswith("nvfp4")
            # fa2 plans NVFP4 by its uint8 storage dtype.
            self.nvfp4_fa2 = self.is_kvcache_nvfp4 and _nvfp4_kv_on_fa2()
            self.nvfp4_trtllm = self.is_kvcache_nvfp4 and not self.nvfp4_fa2
            if self.nvfp4_fa2 and self.use_dcp:
                raise NotImplementedError(
                    "The DCP prefill wrapper cannot read NVFP4 block scales."
                )
            if self.nvfp4_fa2 and self.head_dim == 64:
                raise NotImplementedError("fa2 prefill misreads NVFP4 at head_size 64.")
            if self.nvfp4_trtllm:
                if (
                    force_use_trtllm_attention() is False
                    or not supports_trtllm_attention(is_prefill=True)
                    or not supports_trtllm_attention(is_prefill=False)
                ):
                    raise ValueError(
                        f"--kv-cache-dtype {self.cache_dtype} requires the "
                        "SM100 trtllm-gen "
                        "FlashInfer path."
                    )
                # The scale search only affects the store kernel. FlashInfer
                # reads both variants using the same NVFP4 layout.
                self.kv_cache_dtype = "nvfp4"
            else:
                self.kv_cache_dtype = FlashInferBackend.get_dtype_for_flashinfer(
                    self.cache_dtype
                )
        else:
            self.cache_dtype = "auto"
            self.is_kvcache_nvfp4 = False
            self.nvfp4_fa2 = self.nvfp4_trtllm = False
            assert self.kv_cache_spec.dtype == self.model_config.dtype
            self.kv_cache_dtype = self.kv_cache_spec.dtype

        # Compute per-phase Q dtype.  On SM90 (XQA decode), the prefill and
        # decode phases require different Q dtypes when the KV cache is FP8
        # (FP8-Q for the FI native prefill, BF16/FP16-Q for XQA decode),
        # so both values must be tracked independently.
        self.q_data_type_prefill = self.get_q_data_type(is_prefill=True)
        self.q_data_type_decode = self.get_q_data_type(is_prefill=False)

        # Prefer TRTLLM/XQA for decoding whenever supported. The decode kernel
        # must be selected statically for FULL cudagraph capture.
        # XQA reads an NVFP4 cache only on SM12x; SM90 decodes it with fa2.
        can_use_xqa_or_trtllm_gen_decode = can_use_trtllm_attention(
            self.num_qo_heads, self.num_kv_heads, is_prefill=False
        ) and (not self.nvfp4_fa2 or current_platform.is_device_capability_family(120))
        # Page sizes >= 128 require the trtllm-gen GQA/MQA path (guaranteed by
        # get_supported_kernel_block_sizes).
        assert self.page_size <= 64 or (
            current_platform.is_device_capability_family(100)
            and can_use_xqa_or_trtllm_gen_decode
            and self.num_qo_heads // self.num_kv_heads > 1
        ), f"Unexpected FlashInfer page size {self.page_size} without trtllm-gen GQA"
        self.use_trtllm_decode_attention = can_use_xqa_or_trtllm_gen_decode
        self.flashinfer_trtllm_api_decode_kernel: FlashInferDecodeKernel | None = (
            self._get_flashinfer_trtllm_api_decode_kernel()
            if can_use_xqa_or_trtllm_gen_decode
            else None
        )
        # The dedicated FlashInfer XQA API accepts head dimensions in
        # [16, 256] that are divisible by 16. Some hybrid-attention models
        # (for example Gemma 4) use XQA-compatible sliding-attention groups
        # alongside global-attention groups with head_dim=512. Resolve the
        # backend per KV-cache group so the eligible groups still use XQA and
        # the wider groups fall back to native FlashInfer decode.
        if (
            self.flashinfer_trtllm_api_decode_kernel == FlashInferDecodeKernel.XQA
            and not _is_xqa_head_dim_supported(self.head_dim)
        ):
            logger.warning_once(
                "FlashInfer XQA decode does not support head_dim=%d; "
                "reverting this KV-cache group to native FlashInfer decode.",
                self.head_dim,
            )
            self.use_trtllm_decode_attention = False
            self.flashinfer_trtllm_api_decode_kernel = None
        if (
            self.use_dcp
            and self.flashinfer_trtllm_api_decode_kernel == FlashInferDecodeKernel.XQA
        ):
            logger.warning_once(
                "FlashInfer XQA decode does not support returning LSE and "
                "therefore does not support DCP, reverting to native FlashInfer "
                "decode."
            )
            self.use_trtllm_decode_attention = False
            self.flashinfer_trtllm_api_decode_kernel = None
        self.use_xqa = (
            self.flashinfer_trtllm_api_decode_kernel == FlashInferDecodeKernel.XQA
        )
        self.use_trtllm_gen_varlen_decode = self._uses_trtllm_gen_varlen_decode(
            vllm_config, self.flashinfer_trtllm_api_decode_kernel, self.use_dcp
        )
        # Fused draft steps advance seq_lens in place and keep block tables fixed,
        # so one trtllm-gen decode build serves every step (not DCP-local seq_lens).
        self.supports_draft_decode_metadata_update = (
            self.flashinfer_trtllm_api_decode_kernel
            == FlashInferDecodeKernel.TRTLLM_GEN
            and not self.use_dcp
        )
        # No use_fa2_native_decode (upstream #60532); sm_8x verify keeps Merlin's gate.
        self._init_reorder_batch_threshold(
            1,
            supports_spec_as_decode=(
                self.flashinfer_trtllm_api_decode_kernel is not None
                or (
                    # Spec-as-decode on sm_8x via the FI native decode wrapper.
                    # Upstream gates spec-as-decode on a trtllm/XQA decode kernel,
                    # which is None on sm_86, so without this clause MTP verify
                    # batches take the prefill path and FULL cudagraphs cannot
                    # capture. The FI native
                    # decode wrapper can plan uniform verify batches when FlashInfer is
                    # new enough.
                    not self.use_trtllm_decode_attention
                    and self._uniform_verify
                    and flashinfer_supports_uniform_multi_token_decode()
                )
            ),
            # trtllm-gen decode receives no cp_rank/global-seq-len information,
            # so its end-aligned causal mask is wrong for q_len > 1 over the
            # DCP-interleaved local KV shard (spec token i misses up to
            # (dcp_world_size - 1) * (q_len - 1 - i) KV entries, including its
            # own). Keep the threshold at 1 under DCP so spec queries take the
            # DCP-aware prefill path, until the kernel is CP-aware (compare
            # flash_attn_varlen_func's cp_world_size/cp_rank/cp_tot_seqused_k).
            supports_dcp_with_varlen=False,
        )

        # Global hyperparameters shared by all attention layers
        # TODO: discard this for trtllm-gen backend
        per_layer_parameters = get_per_layer_parameters(
            vllm_config, layer_names, FlashInferImpl
        )
        self.global_hyperparameters = infer_global_hyperparameters(per_layer_parameters)
        self.sm_scale = self.global_hyperparameters.sm_scale
        self.window_left = self.global_hyperparameters.window_left
        self.logits_soft_cap = self.global_hyperparameters.logits_soft_cap
        # The DCP prefill context pass runs on trtllm-gen when available, with
        # an unquantized KV cache and no sliding window or soft-capping.
        self.use_trtllm_dcp_prefill = (
            self.use_dcp
            and can_use_trtllm_attention(
                self.num_qo_heads, self.num_kv_heads, is_prefill=True
            )
            and self.kv_cache_spec.kv_quant_mode == KVQuantMode.NONE
            and self.window_left == -1
            and not self.logits_soft_cap
        )
        if self.use_trtllm_dcp_prefill:
            logger.info_once("Using the trtllm-gen context kernel for DCP prefill.")
        self.has_sinks = self.global_hyperparameters.has_sinks
        if self.has_sinks and (self.nvfp4_fa2 or not FlashInferBackend.supports_sink()):
            raise NotImplementedError(
                "FlashInfer backend currently does not support attention "
                "sinks, please use trtllm on blackwell or flash attention on "
                "earlier GPUs."
            )
        capability = current_platform.get_device_capability()
        arch = f"sm{capability.major}{capability.minor}" if capability else "unknown"
        decode_backend = (
            self.flashinfer_trtllm_api_decode_kernel.value
            if self.flashinfer_trtllm_api_decode_kernel is not None
            else "flashinfer-native"
        )
        logger.info_once(
            "FlashInfer resolved query dtypes: prefill=%s, decode=%s, "
            "decode_backend=%s, kv_cache_dtype=%s, arch=%s",
            self.q_data_type_prefill,
            self.q_data_type_decode,
            decode_backend,
            self.kv_cache_dtype,
            arch,
        )
        # Preparing persistent buffers
        self.paged_kv_indptr = CpuGpuBuffer(
            max_num_reqs + 1, dtype=torch.int32, device=self.device, pin_memory=False
        )
        self.paged_kv_indices = torch.zeros(
            max_num_pages, dtype=torch.int32, device=self.device
        )
        self.paged_kv_last_page_len = CpuGpuBuffer(
            max_num_reqs, dtype=torch.int32, device=self.device, pin_memory=False
        )
        self._plan_staging = _PlanStagingRing()

    @property
    def kv_cache_layout(self) -> KVCacheLayout:
        return self.cache_config.get_resolved_kv_cache_layout()

    # Keep SM90 prefill/decode Q dtype selection in one place.
    def get_q_data_type(self, is_prefill: bool) -> torch.dtype:
        # The user sets --attention-config.disable_flashinfer_q_quantization
        # to 1 explicitly, use model dtype for query.
        if self.vllm_config.attention_config.disable_flashinfer_q_quantization:
            return self.model_config.dtype

        # self.cache_dtype is resolved per KV-cache group: it is "auto" when
        # this group is unquantized (e.g. --kv-cache-dtype-skip-layers), even
        # if cache_config requests a quantized dtype globally.
        cache_dtype = self.cache_dtype

        # On SM90/SM12x, XQA decode requires BF16/FP16-Q even with FP8 KV cache.
        # FI native prefill on SM90 still uses FP8-Q in that case; SM12x prefill
        # is fa2-only and keeps the model dtype (handled below).
        if (
            (
                current_platform.is_device_capability(90)
                or current_platform.is_device_capability_family(120)
            )
            and not is_prefill
            and force_use_trtllm_attention() is not False
            and cache_dtype.startswith("fp8")
        ):
            return self.model_config.dtype

        # Otherwise, match Q dtype to the KV cache dtype.
        if cache_dtype.startswith("fp8"):
            # FP8-Q requires an fp8 tensor-core attention path. SM90 uses
            # native FA3 for prefill; XQA decode on SM90/SM12x is handled
            # above with model-dtype Q, while SM100 uses trtllm-gen.
            # Architectures with only fa2 (e.g. SM89, SM120) cannot
            # consume FP8 queries, so keep the model dtype for Q there.
            if current_platform.is_device_capability(
                90
            ) or current_platform.is_device_capability_family(100):
                return FlashInferBackend.get_dtype_for_flashinfer(cache_dtype)
            return self.model_config.dtype
        if cache_dtype.startswith("nvfp4"):
            if self.nvfp4_fa2:
                return self.model_config.dtype
            return FlashInferBackend.get_dtype_for_flashinfer("fp8_e4m3")
        return self.kv_cache_spec.dtype

    @override  # type: ignore[misc]
    @classmethod
    def get_cudagraph_support(
        cls: type["FlashInferMetadataBuilder"],
        vllm_config: VllmConfig,
        kv_cache_spec: KVCacheSpec,
    ) -> AttentionCGSupport:
        """Get the cudagraph support level for FlashInfer attention."""
        # XQA lacks LSE for DCP; DCP also cannot graph variable-length trtllm-gen.
        if vllm_config.parallel_config.decode_context_parallel_size > 1:
            return AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE

        kv_specs = iter_layer_specs(kv_cache_spec)
        # Upstream #60532 returns UNIFORM_BATCH unconditionally below
        # SM90 here; our sm_8x verify path keeps its own gate below
        # (uniform_verify_active + FlashInfer multi-token decode support).

        num_qo_heads = vllm_config.model_config.get_num_attention_heads(
            vllm_config.parallel_config
        )
        is_xqa_arch = current_platform.is_device_capability(
            90
        ) or current_platform.is_device_capability_family(120)
        has_uniform_batch_support: bool = len(kv_specs) > 0
        for spec in kv_specs:
            if not isinstance(spec, AttentionSpec):
                # FlashInfer only applies to attention, so we don't consider other types
                # of KV spec (e.g. Mamba) here. This is mostly for type checking.
                continue
            if not can_use_trtllm_attention(
                num_qo_heads=num_qo_heads,
                num_kv_heads=spec.num_kv_heads,
                is_prefill=False,
            ) or (
                is_xqa_arch
                and (
                    (
                        spec.kv_quant_mode.is_nvfp4
                        and not current_platform.is_device_capability_family(120)
                    )
                    or not _is_xqa_head_dim_supported(spec.head_size)
                )
            ):
                has_uniform_batch_support = False
                break

        use_non_causal = vllm_config.attention_config.use_non_causal
        if (
            has_uniform_batch_support
            or (
                uniform_verify_active(vllm_config)
                and flashinfer_supports_uniform_multi_token_decode()
            )
        ) and (not use_non_causal or is_xqa_arch):
            return AttentionCGSupport.UNIFORM_BATCH
        else:
            return AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE

    @override  # type: ignore[misc]
    @classmethod
    def get_varlen_cudagraph_max_query_len(
        cls: type["FlashInferMetadataBuilder"],
        vllm_config: VllmConfig,
        kv_cache_spec: KVCacheSpec,
    ) -> int | None:
        # The uniform gate already covers DCP, head geometry, the XQA head-dim
        # limit and causality, and yields None for subclasses that force NEVER.
        # The native fa2 decode wrapper takes only uniform query lengths.
        if (
            _use_fa2_native_decode()
            or cls.get_cudagraph_support(vllm_config, kv_cache_spec)
            != AttentionCGSupport.UNIFORM_BATCH
        ):
            return None
        # The kernel selector asserts off the trtllm-capable architectures, so
        # call it only past the gate, as get_cudagraph_support does.
        if not cls._uses_trtllm_gen_varlen_decode(
            vllm_config,
            cls._get_flashinfer_trtllm_api_decode_kernel(),
            use_dcp=vllm_config.parallel_config.decode_context_parallel_size > 1,
        ):
            return None
        # Longer requests are split off as prefills.
        return max_decode_query_len(vllm_config)

    def _get_workspace_buffer(self):
        if self._workspace_buffer is None:
            self._workspace_buffer = torch.zeros(
                self._workspace_bytes(), dtype=torch.uint8, device=self.device
            )
        return self._workspace_buffer

    # Used by the size check of a shared buffer.
    def _workspace_bytes(self) -> int:
        """Bytes of the float workspace this builder would allocate for itself."""
        buffer_size = envs.VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE
        if envs.VLLM_BATCH_INVARIANT:
            buffer_size = FLASHINFER_WORKSPACE_BUFFER_SIZE_BATCH_INVARIANT
        else:
            # FlashInfer prefill temp buffers (batch_prefill_tmp_v, ...)
            # scale with the prefill chunk and query-head footprint, NOT
            # context length. The fixed ~394 MiB default is too small for
            # wide-head models at the default 8192-token chunk on some
            # archs (e.g. sm_120), where FlashInfer hard-errors instead of
            # growing. Size to the batch's head footprint; never shrink
            # below the configured default.
            est = (
                self.max_num_batched_tokens
                * self.num_qo_heads
                * self.head_dim
                * FLASHINFER_PREFILL_WORKSPACE_BYTES_PER_ELEM
            )
            buffer_size = max(buffer_size, est)
        return buffer_size

    def set_workspace_buffer(self, workspace_buffer: torch.Tensor):
        self._workspace_buffer = workspace_buffer

    @staticmethod
    def _uses_trtllm_gen_varlen_decode(
        vllm_config: VllmConfig,
        decode_kernel: FlashInferDecodeKernel | None,
        use_dcp: bool,
    ) -> bool:
        """Whether decode reads per-request query lengths from the device
        qo_indptr. Adaptive verification trims drafts on device, and only
        trtllm-gen supports that (the selector already rejects the other
        configurations)."""
        speculative_config = vllm_config.speculative_config
        return (
            speculative_config is not None
            and speculative_config.enable_adaptive_verification
            and decode_kernel == FlashInferDecodeKernel.TRTLLM_GEN
            and not use_dcp
        )

    @staticmethod
    def _get_flashinfer_trtllm_api_decode_kernel() -> FlashInferDecodeKernel:
        if current_platform.is_device_capability(
            90
        ) or current_platform.is_device_capability_family(120):
            return FlashInferDecodeKernel.XQA
        assert current_platform.is_device_capability_family(100)
        return FlashInferDecodeKernel.TRTLLM_GEN

    def _compute_decode_query_lens(
        self,
        qo_indptr: torch.Tensor,
        qo_indptr_cpu: torch.Tensor,
        num_decodes: int,
        num_decode_tokens: int,
    ) -> tuple[int, torch.Tensor | None, list[int] | None]:
        """Return the query width, ragged offsets, and effective query lengths."""
        if num_decodes == 0 or num_decode_tokens == 0:
            return 1, None, None

        if not self.use_xqa:
            decode_q_lens = (
                qo_indptr_cpu[1 : num_decodes + 1] - qo_indptr_cpu[:num_decodes]
            )
            q_len_per_req = num_decode_tokens // num_decodes
            if num_decode_tokens % num_decodes == 0 and bool(
                (decode_q_lens == q_len_per_req).all().item()
            ):
                return q_len_per_req, None, None

            # CUDA-graph padding can leave zero-length requests after a
            # uniform speculative decode batch. Keep the packed query and
            # describe it through the trtllm-gen varlen API.
            max_q_len = int(decode_q_lens.max().item())
            if max_q_len > 1:
                return max_q_len, qo_indptr[: num_decodes + 1], None
            return 1, None, None

        decode_q_lens = qo_indptr_cpu[1 : num_decodes + 1] - qo_indptr_cpu[:num_decodes]
        nonzero = decode_q_lens[decode_q_lens > 0]
        q_len_per_req = int(nonzero.max().item()) if nonzero.numel() > 0 else 1
        uniform = nonzero.numel() <= 1 or bool((nonzero == nonzero[0]).all().item())
        if uniform and num_decode_tokens == num_decodes * q_len_per_req:
            return q_len_per_req, None, None

        if q_len_per_req <= 1:
            return 1, None, None

        q_lens = decode_q_lens.tolist()
        return (
            q_len_per_req,
            qo_indptr[: num_decodes + 1],
            q_lens,
        )

    def _get_decode_mask(
        self,
        q_len_per_req: int,
        ragged_q_lens: list[int] | None,
        num_decodes: int,
        causal: bool,
    ) -> torch.Tensor | None:
        """Return the packed XQA draft mask for speculative decode."""
        if q_len_per_req <= 1:
            return None

        if ragged_q_lens is None:
            key = (q_len_per_req, causal)
            buf = self._decode_mask_cache.get(key)
            if buf is None:
                per_req = _make_xqa_draft_block_mask(q_len_per_req, causal, self.device)
                buf = (
                    per_req.unsqueeze(0).expand(self.max_num_reqs, -1, -1).contiguous()
                )
                self._decode_mask_cache[key] = buf
            return buf[:num_decodes]

        return _make_xqa_ragged_draft_block_mask(
            ragged_q_lens, q_len_per_req, causal, self.device
        )

    def _get_prefill_wrapper(
        self,
        causal: bool = True,
    ) -> BatchPrefillWithPagedKVCacheWrapper | BatchDCPPrefillWrapper:
        if not causal:
            if self.use_dcp:
                raise NotImplementedError(
                    "FlashInfer non-causal prefill is not supported with DCP yet."
                )
            if self.is_kvcache_nvfp4:
                raise NotImplementedError(
                    "FlashInfer non-causal attention is not supported with "
                    "NVFP4 KV cache."
                )
            if self._noncausal_prefill_wrapper is None:
                if self.has_sinks and (
                    current_platform.is_device_capability(90)
                    or current_platform.is_device_capability_family(120)
                ):
                    self._noncausal_prefill_wrapper = (
                        BatchAttentionWithAttentionSinkWrapper(
                            self._get_workspace_buffer(),
                            get_flashinfer_layout_string(self.kv_cache_layout),
                            backend="auto",
                            q_data_type=self.q_data_type_prefill,
                            kv_data_type=self.kv_cache_dtype,
                            head_dim_qk=self.head_dim,
                            head_dim_vo=self.head_dim,
                            window_left=self.window_left,
                        )
                    )
                else:
                    self._noncausal_prefill_wrapper = (
                        BatchPrefillWithPagedKVCacheWrapper(
                            self._get_workspace_buffer(),
                            get_flashinfer_layout_string(self.kv_cache_layout),
                            backend="auto",
                        )
                    )
            return self._noncausal_prefill_wrapper

        if self._prefill_wrapper is None:
            if self.use_dcp:
                self._prefill_wrapper = BatchDCPPrefillWrapper(
                    kv_layout=get_flashinfer_layout_string(self.kv_cache_layout),
                    workspace_buffer=self._get_workspace_buffer(),
                    dcp_a2a=self.dcp_a2a,
                    use_trtllm_context=self.use_trtllm_dcp_prefill,
                )
            else:
                if self.has_sinks and (
                    current_platform.is_device_capability(90)
                    or current_platform.is_device_capability_family(120)
                ):
                    assert not self.is_kvcache_nvfp4
                    self._prefill_wrapper = BatchAttentionWithAttentionSinkWrapper(
                        self._get_workspace_buffer(),
                        get_flashinfer_layout_string(self.kv_cache_layout),
                        backend="auto",
                        q_data_type=self.q_data_type_prefill,
                        kv_data_type=self.kv_cache_dtype,
                        head_dim_qk=self.head_dim,
                        head_dim_vo=self.head_dim,
                        window_left=self.window_left,
                    )
                else:
                    # NVFP4 KV cache requires the trtllm-gen backend inside
                    # the wrapper on SM100; SM8x, SM90 and SM12x read it with fa2.
                    backend = "trtllm-gen" if self.nvfp4_trtllm else "auto"
                    self._prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(
                        self._get_workspace_buffer(),
                        get_flashinfer_layout_string(self.kv_cache_layout),
                        backend=backend,
                    )
        assert self._prefill_wrapper is not None
        return self._prefill_wrapper

    def _get_decode_wrapper(
        self,
        batch_size: int,
        use_cudagraph: bool = False,
        q_len_per_req: int = 1,
        smallq_mode: int = -1,
    ):
        use_cudagraph = use_cudagraph and smallq_mode != 0
        key = decode_wrapper_key(batch_size, q_len_per_req, smallq_mode)
        if use_cudagraph:
            decode_wrapper = self._decode_wrappers_cudagraph.get(key)
        elif smallq_mode != -1:
            decode_wrapper = self._decode_wrappers_smallq.get(smallq_mode)
        else:
            decode_wrapper = self._decode_wrapper

        if decode_wrapper is None:
            if use_cudagraph:
                paged_kv_indptr = self.paged_kv_indptr.gpu[: batch_size + 1]
                paged_kv_indices = self.paged_kv_indices
                paged_kv_last_page_len = self.paged_kv_last_page_len.gpu[:batch_size]
            else:
                paged_kv_indptr = None
                paged_kv_indices = None
                paged_kv_last_page_len = None
            # NVFP4 KV cache requires the trtllm-gen backend inside
            # the wrapper on SM100; SM8x, SM90 and SM12x read it with fa2.
            # The decode backend stays "auto" (upstream #60532 forces "fa2" below SM90):
            # with use_tensor_cores=True FlashInfer resolves "auto" to fa2 on
            # sm_8x anyway, and the sm_8x small-q verify tile is planned on it.
            backend = "trtllm-gen" if self.nvfp4_trtllm else "auto"
            decode_wrapper = BatchDecodeWithPagedKVCacheWrapper(
                self._get_workspace_buffer(),
                get_flashinfer_layout_string(self.kv_cache_layout),
                use_cuda_graph=use_cudagraph,
                paged_kv_indptr_buffer=paged_kv_indptr,
                paged_kv_indices_buffer=paged_kv_indices,
                paged_kv_last_page_len_buffer=paged_kv_last_page_len,
                # Tensor cores are enabled by default because the perf would be
                # at least as good as cuda cores for all attention ops in latest
                # gpus.
                use_tensor_cores=True,
                backend=backend,
            )
            if smallq_mode != -1:
                decode_wrapper.sm8x_smallq_mode = smallq_mode
                note_event(
                    f"wrapper created mode={smallq_mode} graph={use_cudagraph} "
                    f"bs={batch_size} q={q_len_per_req}",
                    logger.info,
                )
            # plan cache (27B series) only on dense Qwen3.5 / sm_86 wrappers
            decode_wrapper._vllm_plan_cache_ok = self._qwen35_series

            # save the decode wrapper
            if use_cudagraph:
                self._decode_wrappers_cudagraph[key] = decode_wrapper
            elif smallq_mode != -1:
                self._decode_wrappers_smallq[smallq_mode] = decode_wrapper
            else:
                self._decode_wrapper = decode_wrapper

        return decode_wrapper

    def _compute_flashinfer_kv_metadata(
        self,
        num_blocks_np: np.ndarray,
        seq_lens_np: np.ndarray,
        block_table_tensor: torch.Tensor,
        num_reqs: int,
        page_size: int,
    ) -> torch.Tensor:
        """Compute paged_kv_indptr, paged_kv_indices and paged_kv_last_page_len.

        Results are stored in self.paged_kv_indptr,
        self.paged_kv_indices, self.paged_kv_last_page_len buffers.

        Returns paged_kv_indices, a GPU tensor with shape [num_actual_pages].
        """
        # write self.paged_kv_indptr_cpu inplace (0-index is always 0)
        np.cumsum(
            num_blocks_np, dtype=np.int32, out=self.paged_kv_indptr.np[1 : num_reqs + 1]
        )
        paged_kv_indptr = self.paged_kv_indptr.copy_to_gpu(num_reqs + 1)

        # write self.paged_kv_indices inplace
        num_actual_pages = self.paged_kv_indptr.np[num_reqs]
        paged_kv_indices = self.paged_kv_indices[:num_actual_pages]
        _copy_page_indices_kernel[(num_reqs,)](
            paged_kv_indices,
            block_table_tensor,
            block_table_tensor.stride(0),
            paged_kv_indptr,
            BLOCK_SIZE=1024,
        )

        # write self.paged_kv_last_page_len_cpu inplace
        paged_kv_last_page_len_np = seq_lens_np % page_size
        self.paged_kv_last_page_len.np[:num_reqs] = np.where(
            (paged_kv_last_page_len_np == 0) & (seq_lens_np != 0),
            page_size,
            paged_kv_last_page_len_np,
        )
        self.paged_kv_last_page_len.copy_to_gpu(num_reqs)
        return paged_kv_indices

    # ---------------- VLLM_DRAFT_BOUNDED_PLAN (draft decode without seq_lens sync) ------------
    def _compute_flashinfer_kv_metadata_bounded(
        self,
        num_blocks_ub_np: np.ndarray,
        seq_lens_ub_np: np.ndarray,
        block_table_tensor: torch.Tensor,
        seq_lens_gpu: torch.Tensor,
        num_reqs: int,
        page_size: int,
    ) -> torch.Tensor:
        """Bounded counterpart of _compute_flashinfer_kv_metadata.

        The CPU halves of the buffers get the UPPER-BOUND page table (only the planner reads
        them: fast_decode_plan in cudagraph mode does not copy them to the GPU, FlashInfer
        fast_decode_plan). The GPU buffers the kernel reads get the TRUE page table,
        computed on the device from the device seq_lens by _draft_bounded_kv_fixup_kernel
        with the same arithmetic as the synced path. No host<->device copy, no sync.
        """
        np.cumsum(
            num_blocks_ub_np,
            dtype=np.int32,
            out=self.paged_kv_indptr.np[1 : num_reqs + 1],
        )
        lpl = seq_lens_ub_np % page_size
        self.paged_kv_last_page_len.np[:num_reqs] = np.where(
            (lpl == 0) & (seq_lens_ub_np != 0), page_size, lpl
        )
        _draft_bounded_kv_fixup_kernel[(num_reqs,)](
            seq_lens_gpu,
            block_table_tensor,
            block_table_tensor.stride(0),
            self.paged_kv_indptr.gpu,
            self.paged_kv_indices,
            self.paged_kv_last_page_len.gpu,
            page_size,
            BLOCK_R=triton.next_power_of_2(max(num_reqs, 1)),
            BLOCK_SIZE=1024,
        )
        # ub page count >= true page count: the slice only feeds the planner's size check.
        return self.paged_kv_indices[: int(self.paged_kv_indptr.np[num_reqs])]

    def _draft_bounded_prepare(
        self,
        bounds,
        num_reqs: int,
        num_decodes: int,
        num_prefills: int,
        num_decode_tokens: int,
        decode_with_flashinfer_trtllm_api: bool,
        use_cascade: bool,
    ) -> "_BoundedDraftPlan | None":
        """Decide whether this draft build can plan from the CPU bounds.

        Returns None to take the synced path (counted). Eligible only for the exact shape the
        argument covers: pure q_len=1 decode on a cudagraph fa2 wrapper that has planned before.
        """
        d = _DRAFT_SEQ_LENS
        d["bp_offered"] += 1
        lb, ub = bounds
        wrapper = None
        if (
            self._bounded_plan_ok
            and not self.use_dcp
            and not use_cascade
            and num_prefills == 0
            and num_decodes == num_reqs
            and num_decode_tokens == num_decodes
            and not decode_with_flashinfer_trtllm_api
            and self.enable_cuda_graph
            and num_decode_tokens <= self._decode_cudagraph_max_bs
            and self.window_left < 0
            and self.decode_fixed_split_size <= 0
            and not self.disable_split_kv
            and lb.shape[0] == num_reqs
            and ub.shape[0] == num_reqs
            and not torch.cuda.is_current_stream_capturing()
        ):
            wrapper = self._decode_wrappers_cudagraph.get((num_decodes, 1))
        if (
            wrapper is None
            or getattr(wrapper, "vllm_first_call", True)
            or getattr(wrapper, "_backend", None) != "fa2"
            or getattr(wrapper, "_plan_info", None) is None
            # The graph replays read the WRAPPER's paged-KV buffers; the fix-up kernel
            # writes the builder's. They must be the same storage (_get_decode_wrapper
            # hands the builder buffers to cudagraph wrappers; fast_decode_plan never
            # copies into them in cudagraph mode).
            or wrapper._paged_kv_indptr_buf.data_ptr()
            != self.paged_kv_indptr.gpu.data_ptr()
            or wrapper._paged_kv_indices_buf.data_ptr()
            != self.paged_kv_indices.data_ptr()
            or wrapper._paged_kv_last_page_len_buf.data_ptr()
            != self.paged_kv_last_page_len.gpu.data_ptr()
            or not bool((lb <= ub).all())
        ):
            d["bp_ineligible_sync"] += 1
            return None
        plan_info = wrapper._plan_info
        padded = int(plan_info[0])
        cta_tile_q = int(plan_info[3])
        gqa = self.num_qo_heads // self.num_kv_heads
        # In cudagraph mode padded_batch_size = max(max_batch_size_if_split, total_num_tiles_q)
        # (scheduler.cuh PrefillSplitQOKVIndptr). When it exceeds the q-tile bound it
        # IS FlashInfer's max_batch_size_if_split, read back rather than re-derived
        # from SM counts.
        tiles_q_bound = -(-num_decodes * gqa // cta_tile_q) + num_decodes - 1
        if padded <= tiles_q_bound or padded > _BP_MAX_PADDED_BATCH or cta_tile_q <= 0:
            d["bp_ineligible_sync"] += 1
            return None
        q_tiles = -(-gqa // cta_tile_q)
        min_chunk = max(128 // self.page_size, 1)  # as in scheduler.cuh
        decision = decide_bounded_plan(
            lb, ub, self.page_size, q_tiles, padded, min_chunk
        )
        if not decision.exact:
            d["bp_ambiguous_sync"] += 1
            return None
        self._bp_ring_init(wrapper)
        return _BoundedDraftPlan(
            lb, ub, torch.from_numpy(ub.astype(np.int32)), decision, wrapper,
            q_tiles, padded, min_chunk,
        )

    @staticmethod
    def _bp_ring_init(wrapper) -> None:
        """Give a draft wrapper a ring of pinned host staging buffers for its plans.

        FlashInfer's planner writes the schedule into the wrapper's pinned host buffer and
        enqueues an async H2D copy of it (scheduler.cuh). With no sync between draft
        steps, the next step's plan would overwrite that host buffer before the GPU (still
        busy with the verify forward) has executed the copy. Rotating buffers, guarded by an
        event per slot, makes each in-flight copy read its own plan.
        """
        if getattr(wrapper, "_vllm_pin_ring", None) is not None:
            return
        bufs = [
            torch.empty(_BP_RING_BYTES, dtype=torch.uint8, pin_memory=True)
            for _ in range(_BP_RING_SLOTS)
        ]
        wrapper._vllm_pin_ring = {
            "bufs": bufs,
            "events": [torch.cuda.Event() for _ in bufs],
            "recorded": [False] * len(bufs),
            "idx": 0,
            # the buffer holding the plan currently in _plan_info (read by the plan check)
            "last_buf": wrapper._pin_memory_int_workspace_buffer,
        }
        wrapper._pin_memory_int_workspace_buffer = bufs[0]
        # the target staging ring (_PlanStagingRing)
        # leaves the current schedule in one of its slots and restores the wrapper's own
        # buffer, so "last_buf" above may not hold _plan_info's bytes. Drop the plan-cache
        # key: the first bounded plan then re-plans into this ring and _bp_ring_after_plan
        # points last_buf at the bytes _draft_bounded_check reads.
        wrapper._vllm_plan_key = None

    @staticmethod
    def _bp_ring_after_plan(wrapper, plan_info_before) -> None:
        ring = getattr(wrapper, "_vllm_pin_ring", None)
        if ring is None or wrapper._plan_info is plan_info_before:
            return  # no ring, or the plan cache hit (nothing was staged)
        slot = ring["idx"]
        ring["events"][slot].record()
        ring["recorded"][slot] = True
        ring["last_buf"] = ring["bufs"][slot]
        nxt = (slot + 1) % len(ring["bufs"])
        if ring["recorded"][nxt] and not ring["events"][nxt].query():
            _DRAFT_SEQ_LENS["bp_ring_waits"] += 1
            ring["events"][nxt].synchronize()
        ring["idx"] = nxt
        wrapper._pin_memory_int_workspace_buffer = ring["bufs"][nxt]

    def _draft_bounded_check(
        self,
        bp: "_BoundedDraftPlan",
        common_attn_metadata: CommonAttentionMetadata,
        block_table_tensor: torch.Tensor,
        num_reqs: int,
    ) -> bool:
        """After planning: the plan FlashInfer produced must be the replica's ub schedule
        (validates the replica, its max_batch_size_if_split and cta_tile_q, on every build,
        without a sync). VERIFY mode additionally syncs and checks bounds, the true-length
        schedule and the GPU page table. False = take the exact path."""
        d = _DRAFT_SEQ_LENS
        wrapper = bp.wrapper
        ring = wrapper._vllm_pin_ring
        got = plan_bytes_schedule(
            ring["last_buf"], wrapper._plan_info, num_reqs, self.page_size
        )
        if got != bp.decision.ub_schedule:
            d["bp_plan_check_fail"] += 1
            self._bounded_plan_ok = False
            logger.warning(
                "draft bounded plan DISABLED: FlashInfer planned chunk=%s pages chunks=%s, "
                "replica predicted chunk=%s chunks=%s (n=%d); this build re-plans exactly",
                got.chunk_pages, got.chunks[:16], bp.decision.ub_schedule.chunk_pages,
                bp.decision.ub_schedule.chunks[:16], num_reqs)
            return False
        if not self._bp_fired:
            self._bp_fired = True
            logger.info(
                "DRAFT BOUNDED PLAN ACTIVE: draft decode planned from CPU seq_lens bounds, "
                "true page table written on the GPU (no seq_lens sync)")
        if d["bp_nosync"] % 5000 == 1:
            _bp_log_counters("periodic")
        if not d["bounded_verify"]:
            return True
        with gpu_sync_allowed():
            true = common_attn_metadata.seq_lens[:num_reqs].cpu()
            n_true_pages = int(num_pages(true.numpy(), self.page_size).sum())
            g_indptr = self.paged_kv_indptr.gpu[: num_reqs + 1].cpu()
            g_lpl = self.paged_kv_last_page_len.gpu[:num_reqs].cpu()
            g_idx = self.paged_kv_indices[: int(g_indptr[-1])].cpu()
            bt = block_table_tensor[:num_reqs].cpu()
        d["bp_verified"] += 1
        t = true.numpy().astype(np.int64)
        in_bounds = bool(((t >= bp.lb) & (t <= bp.ub)).all())
        true_sched = fa2_decode_schedule(
            num_pages(t, self.page_size), bp.q_tiles, bp.max_split, bp.min_chunk
        )
        r_indptr, r_idx, r_lpl = kv_metadata_reference(true, bt, self.page_size)
        ok = (
            in_bounds
            and true_sched == bp.decision.ub_schedule
            and int(r_indptr[-1]) == n_true_pages
            and torch.equal(g_indptr, r_indptr)
            and torch.equal(g_lpl, r_lpl)
            and torch.equal(g_idx, r_idx)
        )
        if not ok:
            d["bp_verify_mismatch"] += 1
            if d["bp_verify_mismatch"] <= 8:
                logger.warning(
                    "draft bounded plan VERIFY MISMATCH #%d: in_bounds=%s sched_eq=%s "
                    "indptr_eq=%s lpl_eq=%s idx_eq=%s true=%s lb=%s ub=%s",
                    d["bp_verify_mismatch"], in_bounds,
                    true_sched == bp.decision.ub_schedule,
                    torch.equal(g_indptr, r_indptr), torch.equal(g_lpl, r_lpl),
                    torch.equal(g_idx, r_idx), t[:8].tolist(), bp.lb[:8].tolist(),
                    bp.ub[:8].tolist())
        if d["bp_verified"] == 1 or d["bp_verified"] % 1000 == 0:
            _bp_log_counters("verify")
        return ok

    # ---------------- VLLM_TARGET_SEQ_LENS_ASYNC (verify build without seq_lens sync) --------
    def _target_exact_seq_lens(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        num_reqs: int,
        num_decodes: int,
        num_prefills: int,
        num_decode_tokens: int,
        decode_with_flashinfer_trtllm_api: bool,
        use_cascade: bool,
    ) -> torch.Tensor | None:
        """Exact CPU seq_lens for a uniform verify batch (q_len > 1, pure native decode), or
        None to take the synced path. The values equal seq_lens.cpu() by construction
        (vllm/v1/worker/gpu/target_seq_lens.py), so everything derived from them -- page
        table, kv lens, the FlashInfer plan -- is identical to the synced build."""
        handle = common_attn_metadata.seq_lens_cpu_exact
        t = _TARGET_SEQ_LENS
        if handle is None:
            if self._tsl_ok:
                t["no_handle"] += 1
            return None
        if not (
            self._tsl_ok
            and not self.use_dcp
            and not use_cascade
            and num_prefills == 0
            and num_decodes == num_reqs
            and num_decodes > 0
            and num_decode_tokens % num_decodes == 0
            and num_decode_tokens // num_decodes > 1
            and not decode_with_flashinfer_trtllm_api
            and common_attn_metadata.seq_lens.shape[0] == num_reqs
            and not torch.cuda.is_current_stream_capturing()
        ):
            t["ineligible"] += 1
            return None
        with gpu_sync_allowed():
            # event wait on the previous step's post_update + snapshot copy (not the drafts)
            exact = handle.get(num_reqs)
        if exact is None:
            t["stale"] += 1
            return None
        t["used"] += 1
        if not self._tsl_fired:
            self._tsl_fired = True
            logger.info(
                "TARGET SEQ_LENS ASYNC ACTIVE: verify FlashInfer build uses exact seq_lens "
                "from the post_update snapshot (no seq_lens.cpu() behind the draft steps)")
        if t["used"] % 5000 == 1:
            _tsl_log_counters("periodic")
        if not t["verify"]:
            return exact
        with gpu_sync_allowed():
            real = common_attn_metadata.seq_lens.cpu().to(torch.int32)
        t["verified"] += 1
        if not torch.equal(real, exact):
            t["mismatch"] += 1
            self._tsl_ok = False
            d = (real != exact).nonzero().flatten().tolist()
            logger.warning(
                "target seq_lens async VERIFY MISMATCH #%d (path DISABLED): n=%d rows %s "
                "real %s exact %s", t["mismatch"], num_reqs, d[:16],
                real[d[:16]].tolist(), exact[d[:16]].tolist())
            return real
        if t["verified"] == 1 or t["verified"] % 1000 == 0:
            _tsl_log_counters("verify")
        return exact

    @staticmethod
    def _tsl_plan_fence(wrapper) -> None:
        """Before an unsynced re-plan of a verify wrapper: its previous plan's async H2D copy
        (from the wrapper's single pinned host buffer) must have executed before the planner
        overwrites that buffer. True in practice (the snapshot event the build just waited
        on is stream-ordered behind that copy); checked, not assumed."""
        ev = getattr(wrapper, "_vllm_tsl_plan_event", None)
        if ev is not None and not ev.query():
            _TARGET_SEQ_LENS["fence_waits"] += 1
            with gpu_sync_allowed():
                ev.synchronize()

    @staticmethod
    def _tsl_plan_record(wrapper) -> None:
        ev = getattr(wrapper, "_vllm_tsl_plan_event", None)
        if ev is None:
            ev = wrapper._vllm_tsl_plan_event = torch.cuda.Event()
        ev.record()

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> FlashInferMetadata:
        return self.build(
            common_prefix_len=0,
            common_attn_metadata=common_attn_metadata,
            workload_prefill_flags=(
                [False] * common_attn_metadata.num_reqs if self._workload_tile else None
            ),
        )

    @_abort_plan_staging_on_error
    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
        workload_prefill_flags: list[bool] | None = None,
    ) -> FlashInferMetadata:
        num_reqs = common_attn_metadata.num_reqs
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        causal = common_attn_metadata.causal
        route_decode = causal or self.use_xqa
        if route_decode:
            num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
                split_decodes_and_prefills(
                    common_attn_metadata,
                    decode_threshold=self.reorder_batch_threshold,
                    require_uniform=not (
                        self.use_xqa or self.use_trtllm_gen_varlen_decode
                    ),
                )
            )
        else:
            num_decodes = 0
            num_prefills = num_reqs
            num_decode_tokens = 0
            num_prefill_tokens = num_actual_tokens

        page_size = self.page_size
        max_seq_len = common_attn_metadata.max_seq_len
        seq_lens = common_attn_metadata.seq_lens
        block_table_tensor = common_attn_metadata.block_table_tensor
        qo_indptr = common_attn_metadata.query_start_loc
        qo_indptr_cpu = common_attn_metadata.query_start_loc_cpu

        # Step 1: Decide which dispatch modes to use:
        # - Prefill (FI native or TRTLLM)
        # - Decode (FI native, XQA, or trtllm-gen)
        uses_spec_reorder = self.reorder_batch_threshold > 1
        # Page sizes >= 128 must use trtllm-gen; force it for prefill too.
        prefill_force_trtllm = (
            True if page_size >= 128 else self.attention_config.use_trtllm_attention
        )
        prefill_use_trtllm = causal and use_trtllm_attention(
            self.num_qo_heads,
            self.num_kv_heads,
            num_prefill_tokens,
            max_seq_len,
            self.dcp_world_size,
            self.cache_dtype,
            self.q_data_type_prefill,
            is_prefill=True,
            force_use_trtllm=prefill_force_trtllm,
            has_sinks=self.has_sinks,
            has_spec=uses_spec_reorder,
        )
        decode_with_flashinfer_trtllm_api = self.use_trtllm_decode_attention and (
            causal or self.use_xqa
        )

        if not causal and self.use_dcp:
            raise NotImplementedError(
                "FlashInfer non-causal prefill is not supported with DCP yet."
            )
        if not causal and self.use_trtllm_decode_attention:
            logger.warning_once(
                "Using FlashInfer for draft model non-causal attention; TRTLLM "
                "can still be used for target model causal attention."
            )
        all_uses_trtllm = causal and (
            (num_prefills == 0 or prefill_use_trtllm)
            and (num_decodes == 0 or decode_with_flashinfer_trtllm_api)
        )

        if not all_uses_trtllm:
            if self.has_sinks and self.use_dcp:
                raise NotImplementedError(
                    "FlashInfer BatchDCPPrefillWrapper does not support attention "
                    "sinks. Please use TRTLLM on Blackwell or FlashAttention on "
                    "earlier GPUs."
                )

            if not self.global_hyperparameters.has_same_window_lefts:
                raise ValueError(
                    "Window left is not the same for all layers. "
                    "One potential fix is to set disable_sliding_window=True"
                )

            assert self.global_hyperparameters.has_same_all_params, (
                "FlashInfer backend currently only supports models in which "
                "all layers share the same values for the following "
                "hyperparameters: `window_left`, `logits_soft_cap`, "
                "`sm_scale`."
            )

        # Step 2: Initialize the output metadata
        # Leave prefill/decode empty, to be populated
        # case by case depending on the batch contents and backend selection.
        attn_metadata = FlashInferMetadata(
            kv_cache_layout=self.kv_cache_layout,
            num_actual_tokens=num_actual_tokens,
            slot_mapping=common_attn_metadata.slot_mapping,
            q_data_type_prefill=self.q_data_type_prefill,
            q_data_type_decode=self.q_data_type_decode,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            causal=causal,
            prefill=None,
            decode=None,
        )

        # Guard access to seq_lens_cpu, which may not always be needed
        # and can be expensive to retrieve in async mode.
        # When all attention (both prefill and decode) uses TRTLLM,
        # seq_lens_cpu is not needed since TRTLLM paths use GPU tensors
        # (block_tables, seq_lens) directly.
        needs_seq_lens_cpu = self.use_dcp or not all_uses_trtllm
        # draft no-sync handoff: published only once this whole build has succeeded
        _dsl_staged = None
        # VLLM_DRAFT_BOUNDED_PLAN: bounds are consumed by the first build after the offer,
        # whatever path it takes, so they can never reach a later, unrelated build.
        _bp_bounds = _DRAFT_SEQ_LENS["bounds"]
        _bp_bounds_buf = _DRAFT_SEQ_LENS["bounds_buf"]
        _DRAFT_SEQ_LENS["bounds"] = None
        _DRAFT_SEQ_LENS["bounds_buf"] = None
        _bp: _BoundedDraftPlan | None = None
        _tsl_used = False
        if needs_seq_lens_cpu:
            # --- MTP draft-step seq_lens without a device sync ---
            _dsl = _DRAFT_SEQ_LENS
            _dsl_next = _dsl["next"]
            _dsl["next"] = None
            _dsl_next_buf = _dsl.get("next_buf")
            _dsl["next_buf"] = None
            if (
                _bp_bounds is not None
                and _dsl_next is None
                and _bp_bounds_buf == common_attn_metadata.seq_lens.data_ptr()
            ):
                _bp = self._draft_bounded_prepare(
                    _bp_bounds,
                    num_reqs,
                    num_decodes,
                    num_prefills,
                    num_decode_tokens,
                    decode_with_flashinfer_trtllm_api,
                    False,  # use_cascade: cascade attention is not used by this build
                )
            if _bp is not None:
                # Upper bounds; the true lengths only ever exist on the GPU here.
                seq_lens_cpu = _bp.ub_cpu
                _dsl["bp_nosync"] += 1
            elif (
                self._qwen35_series
                and _dsl_next is not None
                and _dsl_next.shape[0] == common_attn_metadata.seq_lens.shape[0]
                and _dsl_next_buf == common_attn_metadata.seq_lens.data_ptr()
            ):
                seq_lens_cpu = _dsl_next
                _dsl["bypassed"] += 1
                if _dsl["verify"]:
                    with gpu_sync_allowed():
                        _dsl_real = common_attn_metadata.seq_lens.cpu()
                    _dsl["verified"] += 1
                    if not torch.equal(_dsl_real, _dsl_next):
                        _dsl["mismatch"] += 1
                        if _dsl["mismatch"] <= 8:
                            _dsl_d = (_dsl_real != _dsl_next).nonzero().flatten().tolist()
                            logger.warning(
                                "draft seq_lens MISMATCH #%d: n=%d rows %s real %s derived %s",
                                _dsl["mismatch"], _dsl_real.shape[0], _dsl_d[:16],
                                _dsl_real[_dsl_d[:16]].tolist(),
                                _dsl_next[_dsl_d[:16]].tolist(),
                            )
                        seq_lens_cpu = _dsl_real
                    if _dsl["verified"] == 1 or _dsl["verified"] % 1000 == 0:
                        logger.info(
                            "draft seq_lens verify: %d builds verified, %d mismatches",
                            _dsl["verified"], _dsl["mismatch"])
            elif (
                self._num_speculative_tokens == 0
                and common_attn_metadata.seq_lens_cpu_upper_bound is not None
            ):
                # No speculative token accounting, so the upper bound is exact.
                seq_lens_cpu = common_attn_metadata.seq_lens_cpu_upper_bound
            elif (
                _tsl_exact := self._target_exact_seq_lens(
                    common_attn_metadata,
                    num_reqs,
                    num_decodes,
                    num_prefills,
                    num_decode_tokens,
                    decode_with_flashinfer_trtllm_api,
                    False,  # use_cascade: cascade attention is not used by this build
                )
            ) is not None:
                # exact (== seq_lens.cpu()), read without waiting for the draft steps
                seq_lens_cpu = _tsl_exact
                _tsl_used = True
                if _dsl["bounded"]:
                    # Same publication as the synced branch below: these ARE the exact
                    # lengths (byte-equal to seq_lens.cpu(); VERIFY mode checks it), same
                    # device buffer tag, so the speculator keeps anchoring this step's
                    # draft bounds instead of syncing every draft step.
                    _dsl["exact_last"] = (
                        seq_lens_cpu,
                        common_attn_metadata.seq_lens.data_ptr(),
                    )
            else:
                with gpu_sync_allowed():
                    seq_lens_cpu = common_attn_metadata.seq_lens.cpu()
                _dsl["synced"] += 1
                if _dsl["bounded"]:
                    # Exact lengths, tagged with their device buffer: the speculator
                    # anchors this step's draft bounds to the verify build's copy.
                    _dsl["exact_last"] = (
                        seq_lens_cpu,
                        common_attn_metadata.seq_lens.data_ptr(),
                    )
            if _dsl["enabled"] and self._qwen35_series:
                # Invalidate now, publish at the end of a successful build: a build
                # that raises after this point must not leave its CPU lengths as the
                # source of the next derivation (a retried draft step would take them
                # + 1 although the device buffer never advanced; it syncs instead).
                _dsl["last"] = None
                _dsl["last_buf"] = None
                # bounded builds hold upper bounds, never a derivation source for + 1
                if _bp is None:
                    _dsl_staged = (
                        seq_lens_cpu,
                        common_attn_metadata.seq_lens.data_ptr(),
                    )
        else:
            seq_lens_cpu = None

        # Adjust seq_lens_cpu for DCP
        if self.use_dcp:
            assert seq_lens_cpu is not None
            if num_prefills > 0:
                # Other attention groups may reuse the same common metadata.
                seq_lens_cpu = seq_lens_cpu.clone()
                qo_indptr_prefill_cpu = (
                    qo_indptr_cpu[num_decodes:] - qo_indptr_cpu[num_decodes]
                )
                query_lens_prefill_cpu = (
                    qo_indptr_prefill_cpu[1:] - qo_indptr_prefill_cpu[:-1]
                )
                seq_lens_cpu[num_decodes:] = (
                    seq_lens_cpu[num_decodes:] - query_lens_prefill_cpu
                )

            seq_lens_cpu = get_dcp_local_seq_lens(
                seq_lens_cpu,
                self.dcp_world_size,
                self.dcp_rank,
                self.dcp_kv_cache_interleave_size,
            )

        # Native paged attention consumes rank-local lengths (context only
        # for DCP prefills), so derive its page counts after the conversion.
        if seq_lens_cpu is not None:
            seq_lens_np = seq_lens_cpu.numpy()
            num_blocks_np = (seq_lens_np + (page_size - 1)) // page_size
        else:
            seq_lens_np = None
            num_blocks_np = None

        # Compute paged_kv_indices if necessary
        # paged_kv_indices is only needed for FlashInfer native paths;
        # XQA/trtllm-gen paths use block_tables directly on GPU.
        needs_native_paged_prefill = num_prefills > 0 and not prefill_use_trtllm
        needs_native_paged_decode = (
            num_decodes > 0 and not decode_with_flashinfer_trtllm_api
        )
        needs_paged_kv_indices = needs_native_paged_prefill or needs_native_paged_decode
        if needs_paged_kv_indices and _bp is not None:
            # eligibility guaranteed pure native decode (no cascade / prefill / DCP)
            assert num_blocks_np is not None
            assert seq_lens_np is not None
            paged_kv_indices = self._compute_flashinfer_kv_metadata_bounded(
                num_blocks_np,
                seq_lens_np,
                block_table_tensor,
                common_attn_metadata.seq_lens,
                num_reqs,
                page_size,
            )
        elif needs_paged_kv_indices:
            assert num_blocks_np is not None
            assert seq_lens_np is not None
            paged_kv_indices = self._compute_flashinfer_kv_metadata(
                num_blocks_np,
                seq_lens_np,
                block_table_tensor,
                num_reqs,
                page_size,
            )
        else:
            paged_kv_indices = None

        # Step 3: Handle prefill and decode pathways case by case
        ## PREFILL PATHWAY
        if num_prefills > 0:
            # Slices for shared prefill metadata
            prefill_start = num_decodes
            qo_indptr_prefill_cpu = (
                qo_indptr_cpu[prefill_start:] - qo_indptr_cpu[prefill_start]
            )
            assert qo_indptr_prefill_cpu.shape[0] == num_prefills + 1

            if prefill_use_trtllm:
                # TRTLLM prefill has no cross-rank combine for DCP-sharded KV;
                # use_trtllm_attention never selects it when DCP is enabled.
                assert not self.use_dcp
                # Create GPU versions
                qo_indptr_prefill_gpu = (
                    qo_indptr[prefill_start:] - qo_indptr[prefill_start]
                )
                # Compute cum_seq_lens_kv on GPU to avoid CPU sync.
                # This is the cumulative sum of the number of KV cache
                # blocks per prefill request.
                prefill_seq_lens = seq_lens[prefill_start:]
                num_blocks_per_req = (prefill_seq_lens + page_size - 1) // page_size
                paged_kv_indptr_prefill_gpu = self.paged_kv_indptr.gpu[
                    prefill_start : num_reqs + 1
                ]
                # Assign to slice to avoid cpu sync.
                paged_kv_indptr_prefill_gpu[:1] = 0
                torch.cumsum(
                    num_blocks_per_req,
                    dim=0,
                    out=paged_kv_indptr_prefill_gpu[1:],
                )
                # Compute max_q_len for prefill requests
                query_lens_prefill_cpu = (
                    qo_indptr_prefill_cpu[1:] - qo_indptr_prefill_cpu[:-1]
                )
                max_q_len_prefill = int(query_lens_prefill_cpu.max().item())
                prefill_block_tables = block_table_tensor[prefill_start:]
                if (
                    self.q_data_type_prefill != FP8_DTYPE
                    and self.cache_dtype.startswith("fp8")
                ):
                    seq_lens_cpu_upper_bound = (
                        common_attn_metadata.seq_lens_cpu_upper_bound
                    )
                    max_prefill_seq_len = max_seq_len
                    if seq_lens_cpu_upper_bound is not None:
                        max_prefill_seq_len = int(
                            seq_lens_cpu_upper_bound[prefill_start:num_reqs]
                            .max()
                            .item()
                        )
                    # Dequantization allocates one page per table entry and
                    # indexes rows by their logical width.
                    prefill_block_tables = canonicalize_singleton_dim_strides(
                        prefill_block_tables[
                            :, : cdiv(max_prefill_seq_len, page_size)
                        ].contiguous()
                    )
                attn_metadata.prefill = TRTLLMPrefill(
                    block_tables=prefill_block_tables,
                    seq_lens=prefill_seq_lens,
                    cum_seq_lens_q=qo_indptr_prefill_gpu,
                    cum_seq_lens_kv=paged_kv_indptr_prefill_gpu,
                    max_q_len=max_q_len_prefill,
                    max_seq_len=max_seq_len,
                )
            else:
                prefill_wrapper = self._get_prefill_wrapper(causal=attn_metadata.causal)
                # Slicing CPU buffers that are only needed for FI native prefills
                paged_kv_last_page_len_prefill_cpu = self.paged_kv_last_page_len.cpu[
                    prefill_start:num_reqs
                ]
                assert paged_kv_last_page_len_prefill_cpu.shape[0] == num_prefills
                paged_kv_indptr_prefill_cpu = self.paged_kv_indptr.cpu[
                    prefill_start : num_reqs + 1
                ]
                assert paged_kv_indptr_prefill_cpu.shape[0] == num_prefills + 1
                # plan() copies these to the GPU with non_blocking=True;
                # stage them in pinned memory so the copies stay async
                # (the reused buffers themselves are intentionally not pinned).
                if PIN_MEMORY:
                    qo_indptr_prefill_cpu = qo_indptr_prefill_cpu.pin_memory()
                    paged_kv_indptr_prefill_cpu = (
                        paged_kv_indptr_prefill_cpu.pin_memory()
                    )
                    paged_kv_last_page_len_prefill_cpu = (
                        paged_kv_last_page_len_prefill_cpu.pin_memory()
                    )
                # plan() also copies kv_lens to the GPU with non_blocking=True,
                # deriving them via get_seq_lens() into unpinned memory when
                # seq_lens is not passed; pass an explicit pinned copy of the
                # same values so that copy stays async too.
                kv_lens_prefill_cpu = get_seq_lens(
                    paged_kv_indptr_prefill_cpu,
                    paged_kv_last_page_len_prefill_cpu,
                    page_size,
                )
                if PIN_MEMORY:
                    kv_lens_prefill_cpu = kv_lens_prefill_cpu.pin_memory()
                if self.use_dcp:
                    assert isinstance(prefill_wrapper, BatchDCPPrefillWrapper)
                    prefill_wrapper.plan(
                        qo_indptr_cpu=qo_indptr_prefill_cpu,
                        paged_kv_indptr_cpu=paged_kv_indptr_prefill_cpu,
                        paged_kv_indices=paged_kv_indices,
                        paged_kv_last_page_len_cpu=paged_kv_last_page_len_prefill_cpu,
                        kv_lens_cpu=kv_lens_prefill_cpu,
                        page_size=self.page_size,
                        num_qo_heads=self.num_qo_heads,
                        dcp_world_size=self.dcp_world_size,
                        num_kv_heads=self.num_kv_heads,
                        head_dim=self.head_dim,
                        sm_scale=self.sm_scale,
                        window_left=self.window_left,
                        logits_soft_cap=self.logits_soft_cap,
                        q_data_type=self.q_data_type_prefill,
                        kv_cache_dtype=self.kv_cache_dtype,
                        prefill_fixed_split_size=self.prefill_fixed_split_size,
                        disable_split_kv=self.disable_split_kv,
                        block_tables=block_table_tensor[prefill_start:],
                    )
                else:
                    assert isinstance(
                        prefill_wrapper,
                        BatchPrefillWithPagedKVCacheWrapper,
                    )
                    # NVFP4 trtllm kernel only supports FP8 output;
                    # use FP8 o_data_type so the wrapper matches the
                    # FP8 output buffer allocated in forward().
                    o_dtype = (
                        FP8_DTYPE if self.nvfp4_trtllm else self.model_config.dtype
                    )
                    _staged = self._plan_staging.stage(prefill_wrapper)
                    staging_fence(prefill_wrapper)
                    prefill_wrapper.plan(
                        qo_indptr=qo_indptr_prefill_cpu,
                        paged_kv_indptr=paged_kv_indptr_prefill_cpu,
                        paged_kv_indices=paged_kv_indices,
                        paged_kv_last_page_len=paged_kv_last_page_len_prefill_cpu,
                        seq_lens=kv_lens_prefill_cpu,
                        num_qo_heads=self.num_qo_heads,
                        num_kv_heads=self.num_kv_heads,
                        head_dim_qk=self.head_dim,
                        page_size=self.page_size,
                        causal=attn_metadata.causal,
                        sm_scale=self.sm_scale,
                        window_left=self.window_left,
                        logits_soft_cap=self.logits_soft_cap,
                        q_data_type=self.q_data_type_prefill,
                        kv_data_type=self.kv_cache_dtype,
                        o_data_type=o_dtype,
                        fixed_split_size=self.prefill_fixed_split_size,
                        disable_split_kv=self.disable_split_kv,
                    )
                    staging_record(prefill_wrapper)
                    self._plan_staging.commit(prefill_wrapper, _staged)
                attn_metadata.prefill = FIPrefill(wrapper=prefill_wrapper)

        ## DECODE PATHWAY
        if num_decodes > 0:
            if decode_with_flashinfer_trtllm_api:
                assert self.flashinfer_trtllm_api_decode_kernel is not None
                seq_lens_decode = seq_lens[:num_decodes]
                if self.use_dcp:
                    assert common_attn_metadata.dcp_local_seq_lens is not None
                    seq_lens_decode = common_attn_metadata.dcp_local_seq_lens[
                        :num_decodes
                    ]
                q_len_per_req, q_cu_seq_lens, ragged_q_lens = (
                    self._compute_decode_query_lens(
                        qo_indptr, qo_indptr_cpu, num_decodes, num_decode_tokens
                    )
                )
                decode_mask = None
                if self.use_xqa:
                    decode_mask = self._get_decode_mask(
                        q_len_per_req,
                        ragged_q_lens,
                        num_decodes,
                        bool(causal),
                    )
                elif self.use_trtllm_gen_varlen_decode:
                    # CPU placeholders may underestimate query lengths;
                    # use a safe upper bound.
                    q_len_per_req = min(
                        common_attn_metadata.max_query_len, self.reorder_batch_threshold
                    )
                    # Always pass GPU offsets so a graph captured with uniform
                    # query lengths can replay batches with variable query lengths.
                    q_cu_seq_lens = qo_indptr[: num_decodes + 1]
                attn_metadata.decode = FlashInferTrtllmAPIDecode(
                    kernel=self.flashinfer_trtllm_api_decode_kernel,
                    block_tables=block_table_tensor[:num_decodes],
                    seq_lens=seq_lens_decode,
                    max_seq_len=max_seq_len,
                    q_len_per_req=q_len_per_req,
                    q_cu_seq_lens=q_cu_seq_lens,
                    mask=decode_mask,
                    dcp_query_start_loc=(
                        qo_indptr[: num_decodes + 1] if self.use_dcp else None
                    ),
                )
            else:
                assert seq_lens_cpu is not None
                pure_decode = num_prefills == 0
                use_cudagraph = (
                    self.enable_cuda_graph
                    and pure_decode
                    and num_decode_tokens <= self._decode_cudagraph_max_bs
                )
                num_input_tokens = num_decode_tokens

                # Spec-as-decode verify batches carry a uniform
                # num_decode_tokens // num_decodes tokens per request; the wrapper's
                # batch size and kv metadata are per request, not per token.
                assert num_decode_tokens % num_decodes == 0
                decode_q_len = num_decode_tokens // num_decodes
                smallq_mode = -1
                if self._workload_tile and decode_q_len in (4, 5):
                    query_lens = (
                        qo_indptr_cpu[1 : num_reqs + 1] - qo_indptr_cpu[:num_reqs]
                    ).tolist()
                    smallq_mode = select_smallq_mode(
                        enabled=True,
                        q_len=decode_q_len,
                        query_lens=query_lens,
                        prefill_flags=workload_prefill_flags,
                        cta32_allowed=self._smallq_cta32,
                        has_prefill=num_prefill_tokens > 0 or num_prefills > 0,
                    )
                if self._workload_tile:
                    attn_metadata.flashinfer_smallq_mode = smallq_mode
                    note_event(f"plan mode={smallq_mode}", logger.info)
                use_cudagraph = use_cudagraph and smallq_mode != 0
                decode_wrapper = self._get_decode_wrapper(
                    num_decodes,
                    use_cudagraph,
                    q_len_per_req=decode_q_len,
                    smallq_mode=smallq_mode,
                )
                # Use the persistent buffer with padding length,
                # instead of the same address but chunked version
                # in atten_metadata when using cudagraph.
                # NVFP4 trtllm kernel only supports FP8 output;
                # use FP8 o_data_type so the wrapper matches the
                # FP8 output buffer allocated in forward().
                o_dtype = FP8_DTYPE if self.nvfp4_trtllm else self.model_config.dtype

                def _plan_decode(paged_kv_indices=paged_kv_indices):
                    # reads the CPU halves of the paged_kv buffers as they are now
                    paged_kv_indptr_cpu = self.paged_kv_indptr.cpu[: num_decodes + 1]
                    paged_kv_last_page_len_cpu = self.paged_kv_last_page_len.cpu[:num_decodes]
                    # plan() copies these to the GPU with non_blocking=True;
                    # stage them in pinned memory so the copies stay async
                    # (the reused buffers themselves are intentionally not pinned).
                    if PIN_MEMORY:
                        paged_kv_indptr_cpu = paged_kv_indptr_cpu.pin_memory()
                        paged_kv_last_page_len_cpu = paged_kv_last_page_len_cpu.pin_memory()
                    # plan() also copies kv_lens to the GPU with non_blocking=True
                    # (trtllm-gen/cute-dsl backends), deriving them via
                    # get_seq_lens() into unpinned memory when seq_lens is not
                    # passed; pass an explicit pinned copy of the same values
                    # so that copy stays async too.
                    kv_lens_decode_cpu = get_seq_lens(
                        paged_kv_indptr_cpu, paged_kv_last_page_len_cpu, page_size
                    )
                    if PIN_MEMORY:
                        kv_lens_decode_cpu = kv_lens_decode_cpu.pin_memory()
                    fast_plan_decode(
                        decode_wrapper,
                        indptr_cpu=paged_kv_indptr_cpu,
                        indices=paged_kv_indices,
                        last_page_len_cpu=paged_kv_last_page_len_cpu,
                        seq_lens_cpu=kv_lens_decode_cpu,
                        num_qo_heads=self.num_qo_heads * self.dcp_world_size,
                        num_kv_heads=self.num_kv_heads,
                        head_dim=self.head_dim,
                        page_size=self.page_size,
                        # Disable flashinfer's pos encoding and use vllm's rope.
                        pos_encoding_mode="NONE",
                        sm_scale=self.sm_scale,
                        window_left=self.window_left,
                        logits_soft_cap=self.logits_soft_cap,
                        q_data_type=self.q_data_type_decode,
                        kv_data_type=self.kv_cache_dtype,
                        o_data_type=o_dtype,
                        fixed_split_size=self.decode_fixed_split_size,
                        disable_split_kv=self.disable_split_kv,
                        q_len_per_req=decode_q_len,
                    )

                _bp_plan_before = getattr(decode_wrapper, "_plan_info", None)
                if _tsl_used:
                    self._tsl_plan_fence(decode_wrapper)
                _staged = self._plan_staging.stage(decode_wrapper)
                _plan_decode()
                self._plan_staging.commit(decode_wrapper, _staged)
                if _tsl_used:
                    self._tsl_plan_record(decode_wrapper)
                self._bp_ring_after_plan(decode_wrapper, _bp_plan_before)
                if _bp is not None and not (
                    decode_wrapper is _bp.wrapper
                    and self._draft_bounded_check(
                        _bp, common_attn_metadata, block_table_tensor, num_reqs
                    )
                ):
                    # Exact path, as without the patch: sync, page table, re-plan.
                    with gpu_sync_allowed():
                        seq_lens_cpu = common_attn_metadata.seq_lens.cpu()
                    _DRAFT_SEQ_LENS["synced"] += 1
                    seq_lens_np = seq_lens_cpu.numpy()
                    paged_kv_indices = self._compute_flashinfer_kv_metadata(
                        (seq_lens_np + (page_size - 1)) // page_size,
                        seq_lens_np,
                        block_table_tensor,
                        num_reqs,
                        page_size,
                    )
                    _bp_plan_before = getattr(decode_wrapper, "_plan_info", None)
                    _staged = self._plan_staging.stage(decode_wrapper)
                    _plan_decode(paged_kv_indices)
                    self._plan_staging.commit(decode_wrapper, _staged)
                    self._bp_ring_after_plan(decode_wrapper, _bp_plan_before)
                    if _DRAFT_SEQ_LENS["enabled"] and self._qwen35_series:
                        _dsl_staged = (
                            seq_lens_cpu,
                            common_attn_metadata.seq_lens.data_ptr(),
                        )
                attn_metadata.decode = FIDecode(wrapper=decode_wrapper)
        _draft_seq_lens_publish(_dsl_staged)
        return attn_metadata

    def update_draft_decode_metadata(self, _metadata: FlashInferMetadata) -> None:
        pass


class FlashInferImpl(AttentionImpl):
    can_return_lse_for_decode: bool = True
    supports_dcp: bool = True

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: int | None = None,
        sinks: torch.Tensor | None = None,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        if sliding_window is None:
            self.sliding_window = (-1, -1)
        else:
            self.sliding_window = (sliding_window - 1, 0)
        self.window_left = (
            self.sliding_window[0] if self.sliding_window is not None else -1
        )
        self.cache_dtype = kv_cache_dtype
        self.is_kvcache_nvfp4 = kv_cache_dtype.startswith("nvfp4")
        self.nvfp4_fa2 = self.is_kvcache_nvfp4 and _nvfp4_kv_on_fa2()
        self.nvfp4_trtllm = self.is_kvcache_nvfp4 and not self.nvfp4_fa2
        self.kv_cache_dtype = "nvfp4" if self.is_kvcache_nvfp4 else kv_cache_dtype
        self.fp4_data_dim = head_size // 2 if self.is_kvcache_nvfp4 else 0
        self.logits_soft_cap = logits_soft_cap
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name

        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "Encoder self-attention and "
                "encoder/decoder cross-attention "
                "are not implemented for "
                "FlashInferImpl"
            )

        self.sinks: torch.Tensor | None = None
        # Keep the source so RL weight updates can refresh the runtime tensor.
        self._sinks_source = sinks
        if sinks is not None:
            if sinks.shape[0] != num_heads:
                raise ValueError(
                    "Sinks must have the same number of heads as the number of "
                    f"heads in the layer. Expected {num_heads}, but got "
                    f"{sinks.shape[0]}."
                )
            self.sinks = sinks

        self.supports_xqa_or_trtllm_gen_decode = can_use_trtllm_attention(
            num_heads, num_kv_heads, is_prefill=False
        )
        vllm_config = get_current_vllm_config_or_none()
        # Query pre-quantization needs a single dtype for the whole query tensor.
        # SM90 XQA needs BF16/FP16-Q for decode and FP8 for prefill,
        # so only enable this for SM100 trtllm-gen where both use FP8-Q.
        self.supports_quant_query_input = (
            self.supports_xqa_or_trtllm_gen_decode
            and is_quantized_kv_cache(self.kv_cache_dtype)
            and current_platform.is_device_capability_family(100)
            and vllm_config is not None
            and not vllm_config.attention_config.disable_flashinfer_q_quantization
        )
        self.bmm1_scale: float | None = None
        self.bmm2_scale: float | None = None
        self.o_sf_scale: float | None = None

        # Allocated eagerly to stay out of CUDA-graph capture; DCP all-gathers
        # the decode query over heads.
        # Only the trtllm-gen/XQA calls (SM90+) take this buffer, so pre-SM90 skips
        # the allocation and keeps the KV memory budget.
        self._multi_ctas_kv_counter_buffer: torch.Tensor | None = None
        if vllm_config is not None and current_platform.has_device_capability(90):
            _sched = vllm_config.scheduler_config
            self._multi_ctas_kv_counter_buffer = _get_multi_ctas_kv_counter_buffer(
                get_trtllm_gen_multi_ctas_kv_counter_bytes(
                    _sched.max_num_batched_tokens or _sched.max_num_seqs,
                    num_heads * self.dcp_world_size,
                    get_device_sm_count(torch.device("cuda")),
                ),
                torch.device("cuda"),
            )

        # Pre-allocated FP8 output buffer for NVFP4 without fused output quant.
        if self.nvfp4_trtllm and vllm_config is not None:
            max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens
            self._nvfp4_fp8_out = torch.empty(
                (max_num_tokens, num_heads, head_size),
                dtype=FP8_DTYPE,
                device="cuda",
            )
        else:
            self._nvfp4_fp8_out = None

        dcp_a2a = (
            vllm_config is not None
            and vllm_config.parallel_config.decode_context_parallel_size > 1
            and vllm_config.parallel_config.dcp_comm_backend == "a2a"
        )
        if dcp_a2a:
            self.dcp_combine = partial(dcp_a2a_lse_reduce, is_lse_base_on_e=False)
        else:
            self.dcp_combine = partial(cp_lse_ag_out_rs, is_lse_base_on_e=False)

    def fused_output_quant_supported(self, quant_key: QuantKey):
        if quant_key == kNvfp4Dynamic and self.is_kvcache_nvfp4:
            logger.warning_once(
                "fuse_attn_quant: not registering the NVFP4-output attention "
                "fusion with an NVFP4 KV cache (no such trtllm-gen kernel). "
                "Layers whose o_proj takes NVFP4 input keep the unfused path "
                "(FP8 -> BF16 -> o_proj quant); FP8-output fusion is unaffected."
            )
            return False
        # XQA does not support FP8/NVFP4 output, so require trtllm-gen
        # (SM100+) here.  Without that we cannot fuse the output quant.
        return (
            self.supports_xqa_or_trtllm_gen_decode
            and is_quantized_kv_cache(self.kv_cache_dtype)
            and current_platform.is_device_capability_family(100)
            and quant_key in (kFp8StaticTensorSym, kNvfp4Dynamic)
        )

    # FlashInfer requires attention sinks to be float32
    def process_weights_after_loading(self, act_dtype: torch.dtype):
        source_sinks = self._sinks_source
        if source_sinks is None:
            return
        if source_sinks.dtype == torch.float32:
            self.sinks = source_sinks
        elif self.sinks is None or self.sinks.dtype != torch.float32:
            self.sinks = source_sinks.to(torch.float32)
        else:
            self.sinks.copy_(source_sinks)

    def get_xqa_bmm1_scale(self, layer: torch.nn.Module, q_data_type: torch.dtype):
        bmm1_scale = self.scale
        if is_quantized_kv_cache(self.kv_cache_dtype):
            if q_data_type in (torch.float8_e4m3fn, torch.float8_e5m2):
                bmm1_scale *= layer._q_scale_float
            bmm1_scale *= layer._k_scale_float
        return bmm1_scale

    # SM90 may need FP8-Q for native prefill and BF16/FP16-Q for XQA decode,
    # so quantize only the slice whose target dtype differs.
    def maybe_quant_query(
        self,
        query: torch.Tensor,
        q_data_type: torch.dtype,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        if query.dtype != q_data_type:
            assert query.dtype in [torch.float16, torch.bfloat16]
            assert q_data_type in [torch.float8_e4m3fn, torch.float8_e5m2]
            assert query.dim() == 3
            num_tokens = query.shape[0]
            num_heads = query.shape[1]
            head_size = query.shape[2]
            assert query.stride(2) == 1 and query.stride(1) == head_size
            query_quantized, _ = custom_ops.scaled_fp8_quant(
                query.view(num_tokens, num_heads * head_size), scale=scale
            )
            return query_quantized.view(num_tokens, num_heads, head_size)

        return query

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor | None,
        value: torch.Tensor | None,
        kv_cache: torch.Tensor,
        attn_metadata: FlashInferMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with FlashInfer.

        Args:
            layer: The attention layer, providing the q/k/v quantization scales.
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size], or None for a
                KV-sharing decoder layer.
            value: shape = [num_tokens, num_kv_heads, head_size], or None for a
                KV-sharing decoder layer.
            kv_cache: [num_blocks, num_kv_heads, block_size, 2*head_size]
            attn_metadata: Metadata for attention.
            output: Tensor that the attention result is written into.
            output_scale: Scale for fused output quantization. Enables the
                attention+quantization fusion path when provided.
            output_block_scale: Block scale for fused output quantization,
                required for nvfp4 output and rejected for fp8 output.

        Returns:
            shape = [num_tokens, num_heads * head_size]

        """
        if attn_metadata is None:
            # Profiling run.
            return output.fill_(0)

        if self.bmm1_scale is None:
            self.bmm1_scale = self.scale
            if is_quantized_kv_cache(self.kv_cache_dtype):
                self.bmm1_scale *= layer._q_scale_float * layer._k_scale_float

        if self.bmm2_scale is None:
            self.bmm2_scale = 1.0
            if is_quantized_kv_cache(self.kv_cache_dtype):
                self.bmm2_scale *= layer._v_scale_float

        prefill_use_trtllm = isinstance(attn_metadata.prefill, TRTLLMPrefill)
        decode_kernel = (
            attn_metadata.decode.kernel
            if isinstance(attn_metadata.decode, FlashInferTrtllmAPIDecode)
            else None
        )
        decode_with_xqa = decode_kernel == FlashInferDecodeKernel.XQA
        decode_with_trtllm_gen = decode_kernel == FlashInferDecodeKernel.TRTLLM_GEN
        decode_with_flashinfer_trtllm_api = decode_with_xqa or decode_with_trtllm_gen

        # The attn+quant fusion happens when output_scale is provided.
        if output_scale is None:
            assert output_block_scale is None, (
                "output_block_scale is not supported when fusion has not happened"
            )
        else:
            assert attn_metadata.q_data_type_prefill == FP8_DTYPE, (
                "Query must be FP8 when attn+quant fusion happened for prefill."
            )
            assert attn_metadata.q_data_type_decode == FP8_DTYPE, (
                "Query must be FP8 when attn+quant fusion happened for decode."
            )
            assert (attn_metadata.num_prefills == 0 or prefill_use_trtllm) and (
                attn_metadata.num_decodes == 0 or decode_with_trtllm_gen
            ), "Output quant fusion requires TRTLLM prefill/trtllm-gen decode"

            if output.dtype == FP8_DTYPE:
                assert output_block_scale is None, (
                    "output_block_scale should not be provided for fp8 output"
                )
            elif output.dtype == FP4_DTYPE:
                assert output_block_scale is not None, (
                    "output_block_scale is required for nvfp4 output"
                )
            else:
                raise ValueError(f"Unsupported output dtype: {output.dtype}")

            # TRTLLM attn kernel requires to scale to pass as a host scalar,
            # store the o scale as a host scalar in warmup run with cuda graph
            # not enabled
            if layer._o_scale_float is None:
                layer._o_scale_float = output_scale.cpu().item()
                if output.dtype == FP8_DTYPE:
                    self.bmm2_scale = self.bmm2_scale / layer._o_scale_float
                elif output.dtype == FP4_DTYPE:
                    self.o_sf_scale = layer._o_scale_float

        # IMPORTANT!
        # NOTE(woosuk): With piece-wise CUDA graphs, this method is executed in
        # eager-mode PyTorch. Thus, we need to be careful about any CPU overhead
        # in this method. For example, `view` and `slice` (or `[:n]`) operations
        # are surprisingly slow even in the case they do not invoke any GPU ops.
        # Minimize the PyTorch ops in this method as much as possible.
        # Whenever making a change in this method, please benchmark the
        # performance to make sure it does not introduce any overhead.

        num_actual_tokens = attn_metadata.num_actual_tokens
        kv_cache_layout = attn_metadata.kv_cache_layout

        # FlashInfer treats uint8 KV cache as NVFP4. vLLM stores FP8 KV cache
        # as uint8 bytes, so pass FP8 caches with their logical dtype.
        if not self.is_kvcache_nvfp4 and kv_cache.dtype == torch.uint8:
            fp8_view_dtype = None
            if self.kv_cache_dtype in ("fp8", "fp8_e4m3", torch.float8_e4m3fn):
                fp8_view_dtype = torch.float8_e4m3fn
            elif self.kv_cache_dtype in ("fp8_e5m2", torch.float8_e5m2):
                fp8_view_dtype = torch.float8_e5m2
            if fp8_view_dtype is not None:
                kv_cache = kv_cache.view(fp8_view_dtype)

        # Inputs and outputs may be padded for CUDA graphs. Keep the original
        # query for single-token XQA, whose non-ragged API requires one row for
        # every (including padded) decode request.
        query_padded = query
        query = query[:num_actual_tokens]
        output_padded = output
        output = output[:num_actual_tokens]

        # When using spec decoding, num_decodes can be < num_decode_tokens
        # because some decode requests may have more than one query token.
        num_decode_tokens = attn_metadata.num_decode_tokens
        num_prefill_tokens = attn_metadata.num_prefill_tokens

        stride_order = kv_cache_layout.layer_view_order
        kv_cache_permute = kv_cache.permute(*stride_order)  # HND and contiguous
        # Fix degenerate strides on any size-1 dimension (e.g. num_kv_heads=1
        # with TP=8).  PyTorch permits non-canonical strides on size-1 dims;
        # CUDA TMA requires ≥16-byte alignment on all non-outermost strides.
        # canonicalize_singleton_dim_strides patches metadata via as_strided —
        # zero-copy.  See vllm.utils.torch_utils.
        fixed = canonicalize_singleton_dim_strides(kv_cache_permute)
        if fixed is not kv_cache_permute:
            logger.debug(
                "Canonicalized degenerate KV cache strides (FlashInfer): "
                "shape=%s, strides before=%s, strides after=%s",
                kv_cache_permute.shape,
                kv_cache_permute.stride(),
                fixed.stride(),
            )
        kv_cache_permute = fixed

        # Split K/V — zero-copy views. NVFP4 stores K/V as separate head
        # groups; other dtypes pack K/V in the content dim.
        hs = self.head_size
        nvfp4_kv_data = None
        nvfp4_kv_block_scales = None
        if self.is_kvcache_nvfp4:
            k_cache, v_cache = kv_cache.split(self.num_kv_heads, dim=1)
            kv_cache_tuple = (
                canonicalize_singleton_dim_strides(k_cache.permute(*stride_order)),
                canonicalize_singleton_dim_strides(v_cache.permute(*stride_order)),
            )
            nvfp4_kv_data, nvfp4_kv_block_scales, _ = _nvfp4_kv_views(
                kv_cache, self.num_kv_heads
            )
        else:
            kv_cache_tuple = kv_cache_permute.split(hs, dim=-1)

        use_dcp = self.dcp_world_size > 1
        if decode_with_xqa:
            assert not use_dcp
            assert self.o_sf_scale is None
            assert output.dtype != FP4_DTYPE

        # Regular attention (common case).
        # Decodes are at the front and prefills are at the back.
        if num_prefill_tokens > 0:
            prefill_query = query[num_decode_tokens:]
            assert prefill_query.shape[0] == num_prefill_tokens

            # Convert query to the expected dtype for prefill if needed.
            prefill_query = self.maybe_quant_query(
                prefill_query,
                attn_metadata.q_data_type_prefill,
                layer._q_scale,
            )

            if not prefill_use_trtllm:
                assert isinstance(attn_metadata.prefill, FIPrefill)
                prefill_wrapper = attn_metadata.prefill.wrapper
                assert prefill_wrapper is not None
                if use_dcp:
                    if key is None or value is None:
                        raise NotImplementedError(
                            "FlashInfer DCP prefill does not support KV-sharing layers"
                        )
                    assert isinstance(prefill_wrapper, BatchDCPPrefillWrapper)
                    assert prefill_wrapper._context._window_left == self.window_left
                    assert prefill_wrapper._context._logits_soft_cap == (
                        self.logits_soft_cap or 0.0
                    )
                    assert prefill_wrapper._context._sm_scale == self.scale
                    assert not prefill_wrapper._context._causal
                    assert prefill_wrapper._new_tokens._window_left == self.window_left
                    assert prefill_wrapper._new_tokens._logits_soft_cap == (
                        self.logits_soft_cap or 0.0
                    )
                    assert prefill_wrapper._new_tokens._sm_scale == self.scale
                    assert prefill_wrapper._new_tokens._causal

                    prefill_wrapper.run(
                        layer,
                        prefill_query,
                        kv_cache_tuple,
                        key[num_decode_tokens:num_actual_tokens],
                        value[num_decode_tokens:num_actual_tokens],
                        out=output[num_decode_tokens:],
                    )
                else:
                    assert isinstance(
                        prefill_wrapper, BatchPrefillWithPagedKVCacheWrapper
                    )
                    assert prefill_wrapper._window_left == self.window_left
                    assert prefill_wrapper._logits_soft_cap == (
                        self.logits_soft_cap or 0.0
                    )
                    assert prefill_wrapper._sm_scale == self.scale
                    assert prefill_wrapper._causal == attn_metadata.causal

                    if self.is_kvcache_nvfp4:
                        kv_cache_for_fi = nvfp4_kv_data
                    else:
                        kv_cache_for_fi = kv_cache_tuple
                    kv_cache_sf = (
                        nvfp4_kv_block_scales if self.is_kvcache_nvfp4 else None
                    )

                    # NVFP4 trtllm kernel only supports FP8 output.
                    # Use a pre-allocated FP8 buffer and dequantize
                    # afterwards.
                    needs_fp8_out_prefill = (
                        self.nvfp4_trtllm and output.dtype != FP8_DTYPE
                    )
                    if needs_fp8_out_prefill:
                        out_prefill = self._nvfp4_fp8_out[:num_prefill_tokens]
                    else:
                        out_prefill = output[num_decode_tokens:]

                    if isinstance(
                        prefill_wrapper, BatchAttentionWithAttentionSinkWrapper
                    ):
                        assert self.sinks is not None
                        prefill_wrapper.run(
                            prefill_query,
                            kv_cache_for_fi,
                            self.sinks,
                            self.scale * layer._q_scale_float * layer._k_scale_float,
                            v_scale=layer._v_scale_float,
                            out=out_prefill,
                        )
                    else:
                        prefill_wrapper.run(
                            prefill_query,
                            kv_cache_for_fi,
                            # The fa2 NVFP4 query stays in the model dtype.
                            q_scale=1.0 if self.nvfp4_fa2 else layer._q_scale_float,
                            k_scale=layer._k_scale_float,
                            v_scale=layer._v_scale_float,
                            out=out_prefill,
                            kv_cache_sf=kv_cache_sf,
                        )

                    if needs_fp8_out_prefill:
                        output[
                            num_decode_tokens : num_decode_tokens + num_prefill_tokens
                        ].copy_(out_prefill)
            else:
                assert isinstance(attn_metadata.prefill, TRTLLMPrefill)
                # prefill_query may be non-contiguous or have degenerate strides
                # on size=1 dims. contiguous() ensures memory layout; then
                # canonicalize_singleton_dim_strides fixes any remaining
                # degenerate strides on size=1 dims for TMA alignment.
                prefill_query = prefill_query.contiguous()
                prefill_query = canonicalize_singleton_dim_strides(prefill_query)
                workspace_buffer = _get_trtllm_workspace_buffer()
                block_tables_prefill = attn_metadata.prefill.block_tables
                seq_lens_prefill = attn_metadata.prefill.seq_lens

                # This path needs to be enabled with VLLM_KV_CACHE_LAYOUT = HND
                assert get_flashinfer_layout_string(kv_cache_layout) == "HND"
                assert is_strictly_contiguous(prefill_query)
                assert is_strictly_contiguous(workspace_buffer)
                assert is_strictly_contiguous(block_tables_prefill)
                assert is_strictly_contiguous(seq_lens_prefill)

                if output.dtype == FP4_DTYPE:
                    assert self.o_sf_scale is not None
                    out = FP4Tensor(
                        data=output[num_decode_tokens:],
                        scale=output_block_scale,
                        scale_start_index=num_decode_tokens,
                        original_shape=prefill_query.shape,
                    )
                else:
                    assert self.o_sf_scale is None
                    out = output[num_decode_tokens:]

                # NVFP4 trtllm kernel only supports FP8 output.
                # Use a pre-allocated FP8 buffer and dequantize afterwards.
                needs_fp8_out = self.is_kvcache_nvfp4 and output.dtype != FP8_DTYPE
                if needs_fp8_out:
                    out = self._nvfp4_fp8_out[:num_prefill_tokens]

                prefill_kv_block_scales = None
                if self.is_kvcache_nvfp4:
                    # NVFP4 trtllm-gen kernel requires FP8 query.
                    assert attn_metadata.q_data_type_prefill == FP8_DTYPE, (
                        "NVFP4 KV cache requires FP8 quantized queries for "
                        "trtllm-gen prefill. Set "
                        "disable_flashinfer_q_quantization=False."
                    )
                    mock_kv_cache = nvfp4_kv_data
                    mock_block_table = block_tables_prefill
                    prefill_kv_block_scales = nvfp4_kv_block_scales
                elif (
                    attn_metadata.q_data_type_prefill != FP8_DTYPE
                    and self.kv_cache_dtype.startswith("fp8")
                ):
                    # TRTLLM prefill attention does not support BF16 Q
                    # and fp8 kv cache. So to enable prefill attention
                    # with fp8 kv cache, we can construct a mock block
                    # and mock kv cache with BF16 KV involved in the prefill.
                    kv_cache_permute = canonicalize_singleton_dim_strides(
                        kv_cache_permute
                    )
                    kv_strides = kv_cache_permute.stride()
                    assert (
                        kv_strides[-1] == 1
                        and kv_strides[-2] == kv_cache_permute.shape[-1]
                    ), (
                        "KV cache inner dims (block_size, head_size) must be "
                        f"contiguous, got strides {kv_strides}"
                    )
                    # fp8 uses (B, H, N, 2*hs); reshape to (B, 2, H, N, hs)
                    # for the dequant kernel — zero-copy view. The dequant
                    # kernel handles the interleaved K/V block stride.
                    B_kv, H_kv, N_kv = kv_cache_permute.shape[:3]
                    kv_cache_5d = kv_cache_permute.view(B_kv, H_kv, N_kv, 2, hs)
                    kv_cache_5d = kv_cache_5d.permute(0, 3, 1, 2, 4)
                    mock_kv_cache, mock_block_table = trtllm_prefill_attn_kvfp8_dequant(
                        kv_cache_5d,
                        block_tables_prefill,
                        layer._k_scale,
                        layer._v_scale,
                        attn_metadata.q_data_type_prefill,
                    )
                else:
                    mock_kv_cache = kv_cache_tuple
                    mock_block_table = block_tables_prefill

                trtllm_batch_context_with_kv_cache(
                    query=prefill_query,
                    kv_cache=mock_kv_cache,
                    workspace_buffer=workspace_buffer,
                    block_tables=mock_block_table,
                    seq_lens=seq_lens_prefill,
                    max_q_len=attn_metadata.prefill.max_q_len,
                    max_kv_len=attn_metadata.prefill.max_seq_len,
                    bmm1_scale=self.bmm1_scale,
                    bmm2_scale=self.bmm2_scale,
                    batch_size=attn_metadata.num_prefills,
                    cum_seq_lens_q=attn_metadata.prefill.cum_seq_lens_q,
                    cum_seq_lens_kv=attn_metadata.prefill.cum_seq_lens_kv,
                    window_left=self.window_left,
                    sinks=self.sinks,
                    o_sf_scale=self.o_sf_scale,
                    out=out,
                    kv_cache_sf=prefill_kv_block_scales,
                    multi_ctas_kv_counter_buffer=self._multi_ctas_kv_counter_buffer,
                )

                if needs_fp8_out:
                    output[
                        num_decode_tokens : num_decode_tokens + num_prefill_tokens
                    ].copy_(out[:num_prefill_tokens])

        if num_decode_tokens > 0:
            decode_query_tokens = num_decode_tokens
            if decode_with_xqa:
                assert isinstance(attn_metadata.decode, FlashInferTrtllmAPIDecode)
                if attn_metadata.decode.q_len_per_req == 1:
                    decode_query_tokens = attn_metadata.num_decodes
            decode_query = query_padded[:decode_query_tokens]
            assert decode_query.shape[0] == decode_query_tokens

            # Convert query to the expected dtype for decode if needed.
            decode_query = self.maybe_quant_query(
                decode_query,
                attn_metadata.q_data_type_decode,
                layer._q_scale,
            )

            if not decode_with_flashinfer_trtllm_api:
                assert isinstance(attn_metadata.decode, FIDecode)
                decode_wrapper = attn_metadata.decode.wrapper
                assert decode_wrapper is not None
                assert decode_wrapper._window_left == self.window_left
                assert decode_wrapper._logits_soft_cap == (self.logits_soft_cap or 0.0)
                assert decode_wrapper._sm_scale == self.scale

                if self.is_kvcache_nvfp4:
                    kv_cache_for_fi = nvfp4_kv_data
                else:
                    kv_cache_for_fi = kv_cache_tuple
                kv_cache_sf = nvfp4_kv_block_scales if self.is_kvcache_nvfp4 else None

                # NVFP4 kernel only supports FP8 output.
                # Use a pre-allocated FP8 buffer and dequantize afterwards.
                needs_fp8_out = self.nvfp4_trtllm and output.dtype != FP8_DTYPE
                if needs_fp8_out:
                    out_decode = self._nvfp4_fp8_out[:num_decode_tokens]
                else:
                    out_decode = output[:num_decode_tokens]

                if use_dcp:
                    decode_query = get_dcp_group().all_gather(
                        decode_query.contiguous(), dim=-2
                    )
                    output_tmp = torch.empty_like(decode_query)
                    lse = torch.empty(
                        (decode_query.size(0), decode_query.size(1)),
                        dtype=torch.float32,
                        device=decode_query.device,
                    )
                    decode_wrapper.run(
                        decode_query,
                        kv_cache_for_fi,
                        q_scale=layer._q_scale_float,
                        k_scale=layer._k_scale_float,
                        v_scale=layer._v_scale_float,
                        out=output_tmp,
                        lse=lse,
                        return_lse=True,
                        kv_cache_sf=kv_cache_sf,
                        sinks=self.sinks,
                    )
                    output[:num_decode_tokens] = self.dcp_combine(
                        output_tmp,
                        lse,
                        get_dcp_group(),
                    )
                else:
                    decode_wrapper.run(
                        decode_query,
                        kv_cache_for_fi,
                        q_scale=1.0 if self.nvfp4_fa2 else layer._q_scale_float,
                        k_scale=layer._k_scale_float,
                        v_scale=layer._v_scale_float,
                        out=out_decode,
                        kv_cache_sf=kv_cache_sf,
                        sinks=self.sinks,
                    )

                if needs_fp8_out:
                    output[:num_decode_tokens].copy_(out_decode)
            else:
                assert isinstance(attn_metadata.decode, FlashInferTrtllmAPIDecode)
                # decode_query may be non-contiguous or have degenerate strides
                # on size=1 dims. contiguous() ensures memory layout; then
                # canonicalize_singleton_dim_strides fixes any remaining
                # degenerate strides on size=1 dims for TMA alignment.
                decode_query = decode_query.contiguous()
                decode_query = canonicalize_singleton_dim_strides(decode_query)
                workspace_buffer = _get_trtllm_workspace_buffer()
                block_tables_decode = attn_metadata.decode.block_tables
                seq_lens_decode = attn_metadata.decode.seq_lens

                # trtllm-gen needs HND layout on SM100. XQA is selected
                # separately on SM90 and does not use this SM100 layout gate.
                if decode_with_trtllm_gen:
                    assert get_flashinfer_layout_string(kv_cache_layout) == "HND"
                else:
                    assert decode_with_xqa
                assert is_strictly_contiguous(decode_query)
                assert is_strictly_contiguous(workspace_buffer)
                assert is_strictly_contiguous(block_tables_decode)
                assert is_strictly_contiguous(seq_lens_decode)
                kv_cache_permute = canonicalize_singleton_dim_strides(kv_cache_permute)
                kv_strides = kv_cache_permute.stride()
                assert (
                    kv_strides[-1] == 1 and kv_strides[-2] == kv_cache_permute.shape[-1]
                ), (
                    "KV cache inner dims (block_size, head_size) must be "
                    f"contiguous, got strides {kv_strides}"
                )

                if use_dcp:
                    assert decode_with_trtllm_gen
                    if output.dtype == FP4_DTYPE:
                        raise NotImplementedError(
                            "DCP decode with FlashInfer trtllm-gen does not support "
                            "FP4 attention output yet."
                        )
                    decode_query = get_dcp_group().all_gather(
                        decode_query.contiguous(), dim=-2
                    )
                    decode_query = canonicalize_singleton_dim_strides(decode_query)

                if decode_with_xqa:
                    bmm1_scale = self.get_xqa_bmm1_scale(
                        layer, attn_metadata.q_data_type_decode
                    )
                    q_len_per_req: int | None = attn_metadata.decode.q_len_per_req

                    flashinfer_xqa_batch_decode_with_kv_cache(
                        query=decode_query,
                        kv_cache=(
                            nvfp4_kv_data if self.is_kvcache_nvfp4 else kv_cache_tuple
                        ),
                        kv_cache_sf=nvfp4_kv_block_scales,
                        workspace_buffer=workspace_buffer,
                        block_tables=block_tables_decode,
                        seq_lens=seq_lens_decode,
                        max_seq_len=attn_metadata.decode.max_seq_len,
                        bmm1_scale=bmm1_scale,
                        bmm2_scale=self.bmm2_scale,
                        window_left=self.window_left,
                        out=output_padded[:decode_query_tokens],
                        sinks=self.sinks,
                        kv_layout=get_flashinfer_layout_string(kv_cache_layout),
                        q_len_per_req=q_len_per_req,
                        mask=attn_metadata.decode.mask,
                        q_cu_seq_lens=attn_metadata.decode.q_cu_seq_lens,
                    )
                    return output_padded

                assert decode_with_trtllm_gen
                if output.dtype == FP4_DTYPE:
                    assert self.o_sf_scale is not None
                    out = FP4Tensor(
                        data=output[:num_decode_tokens],
                        scale=output_block_scale,
                        scale_start_index=0,
                        original_shape=decode_query.shape,
                    )
                else:
                    assert self.o_sf_scale is None
                    out = output[:num_decode_tokens]

                # NVFP4 trtllm kernel only supports FP8 output.
                # Use a pre-allocated FP8 buffer and dequantize afterwards.
                needs_fp8_out = self.is_kvcache_nvfp4 and output.dtype != FP8_DTYPE
                if needs_fp8_out:
                    out = self._nvfp4_fp8_out[:num_decode_tokens]

                q_cu_seq_lens = attn_metadata.decode.q_cu_seq_lens
                max_q_len = None
                q_len_per_req = None
                if q_cu_seq_lens is not None:
                    # q_len_per_req must stay None for the kernel's varlen path.
                    max_q_len = attn_metadata.decode.q_len_per_req
                elif num_decode_tokens % attn_metadata.num_decodes != 0:
                    # This gets triggered when the dummy_run forces
                    # attention to be initialized with q_len = 0
                    q_len_per_req = 1
                else:
                    q_len_per_req = num_decode_tokens // attn_metadata.num_decodes

                lse = None
                if use_dcp:
                    out = torch.empty(
                        decode_query.shape,
                        dtype=output.dtype,
                        device=decode_query.device,
                    )
                    lse = torch.empty(
                        (decode_query.size(0), decode_query.size(1)),
                        dtype=torch.float32,
                        device=decode_query.device,
                    )

                trtllm_batch_decode_with_kv_cache(
                    query=decode_query,
                    kv_cache=(
                        nvfp4_kv_data if self.is_kvcache_nvfp4 else kv_cache_tuple
                    ),
                    workspace_buffer=workspace_buffer,
                    block_tables=block_tables_decode,
                    seq_lens=seq_lens_decode,
                    max_seq_len=attn_metadata.decode.max_seq_len,
                    bmm1_scale=self.bmm1_scale,
                    bmm2_scale=self.bmm2_scale,
                    window_left=self.window_left,
                    sinks=self.sinks,
                    o_sf_scale=self.o_sf_scale,
                    out=out,
                    kv_layout=get_flashinfer_layout_string(kv_cache_layout),
                    backend=attn_metadata.decode.kernel.value,
                    q_len_per_req=q_len_per_req,
                    max_q_len=max_q_len,
                    cum_seq_lens_q=q_cu_seq_lens,
                    kv_cache_sf=(
                        nvfp4_kv_block_scales if self.is_kvcache_nvfp4 else None
                    ),
                    lse=lse,
                    return_lse=self.need_to_return_lse_for_decode,
                    multi_ctas_kv_counter_buffer=self._multi_ctas_kv_counter_buffer,
                )

                if use_dcp:
                    assert isinstance(out, torch.Tensor)
                    assert lse is not None
                    output[:num_decode_tokens] = self.dcp_combine(
                        out,
                        lse,
                        get_dcp_group(),
                        seq_lens=seq_lens_decode,
                        query_start_loc=attn_metadata.decode.dcp_query_start_loc,
                    )
                elif needs_fp8_out:
                    output[:num_decode_tokens].copy_(out)
        return output_padded

    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        if self.kv_sharing_target_layer_name is None:
            # Reshape the input keys and values and store them in the cache.
            # Skip this if sharing KV cache with an earlier attention layer.
            # NOTE(woosuk): Here, key and value are padded while slot_mapping is
            # not padded. However, we don't need to do key[:num_actual_tokens]
            # and value[:num_actual_tokens] because the reshape_and_cache_flash
            # op uses the slot_mapping's shape to determine the number of
            # actual tokens.
            if self.nvfp4_fa2:
                data, scales, kv_layout = _nvfp4_kv_views(kv_cache, self.num_kv_heads)
                nvfp4_quantize_append_paged_kv_cache_with_slot_mapping(
                    key,
                    value,
                    slot_mapping,
                    data,
                    scales,
                    layer._k_scale,
                    layer._v_scale,
                    kv_layout=kv_layout,
                )
                return
            if self.is_kvcache_nvfp4:
                # (B, 2*H, N, full_dim) -> ((B, N, H, full_dim),
                #                            (B, N, H, full_dim));
                # K heads first, then V heads.
                k_cache, v_cache = kv_cache.transpose(1, 2).split(
                    self.num_kv_heads, dim=-2
                )
            else:
                # (B, H, N, 2*hs) -> ((B, N, H, hs), (B, N, H, hs))
                k_cache, v_cache = kv_cache.transpose(1, 2).split(
                    self.head_size, dim=-1
                )
            torch.ops._C_cache_ops.reshape_and_cache_flash(
                key,
                value,
                k_cache,
                v_cache,
                slot_mapping,
                self.cache_dtype,
                layer._k_scale,
                layer._v_scale,
            )


@cache
def flashinfer_supports_uniform_multi_token_decode() -> bool:
    """Whether the installed FlashInfer can plan the tensor-core decode path
    for a uniform q_len_per_req > 1 (spec-decode verify) and keep the plan
    cudagraph-safe."""
    return "q_len_per_req" in inspect.signature(fast_decode_plan).parameters


def fast_plan_decode(
    self,  # decode wrapper
    indptr_cpu: torch.Tensor,
    indices: torch.Tensor,
    last_page_len_cpu: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    page_size: int,
    pos_encoding_mode: str = "NONE",
    window_left: int = -1,
    logits_soft_cap: float | None = None,
    q_data_type: str | torch.dtype | None = "float16",
    kv_data_type: str | torch.dtype | None = None,
    o_data_type: str | torch.dtype | None = None,
    data_type: str | torch.dtype | None = None,
    sm_scale: float | None = None,
    rope_scale: float | None = None,
    rope_theta: float | None = None,
    non_blocking: bool = True,
    fixed_split_size: int = -1,
    disable_split_kv: bool = False,
    q_len_per_req: int = 1,
) -> None:
    """A faster version of BatchDecodeWithPagedKVCacheWrapper::plan used for
    cudagraph capture/replay, while the no cudagraph version turns back
    to the original plan.
    using original plan after passing host-side buffers:
    - only host-to-device copy of indptr and last_page_len buffers
    Modifications for cudagraph:
    - only host-to-device copy of indptr and last_page_len buffers.
    - avoid device-to-device copy of indices buffer.

    Part of the code get inspiration from the original plan from FlashInfer repo
    and the implementation of fast_decode_plan for FlashInfer in SGlang repo.
    """
    supports_uniform_multi_token = flashinfer_supports_uniform_multi_token_decode()
    if q_len_per_req > 1 and not supports_uniform_multi_token:
        raise RuntimeError(
            "The installed FlashInfer does not support uniform multi-token decode."
        )
    smallq_mode = getattr(self, "sm8x_smallq_mode", -1)
    planned_mode = getattr(self, "_sm8x_smallq_planned_mode", smallq_mode)
    if self.is_cuda_graph_enabled and planned_mode != smallq_mode:
        raise ValueError("FlashInfer graph wrapper changed its small-q tile")
    uniform_decode_kwargs: dict[str, Any] = {}
    if supports_uniform_multi_token:
        uniform_decode_kwargs["q_len_per_req"] = q_len_per_req

    # Warm up with the original plan if it is first call, and always run the
    # original plan if we run for dynamic shape. For fixed shape (cudagraph),
    # this warm up is to generate the _cached_module for the decode wrapper.
    if not self.is_cuda_graph_enabled or getattr(self, "vllm_first_call", True):
        # plan() copies qo_indptr from flashinfer's internal cached host
        # arange buffer (_get_range_buf) with non_blocking=True; pin the
        # cache entry so the copy stays async.
        pin_host_range_buf(len(last_page_len_cpu) + 1)
        staging_fence(self)
        self.plan(
            indptr=indptr_cpu,
            indices=indices,
            last_page_len=last_page_len_cpu,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            page_size=page_size,
            pos_encoding_mode=pos_encoding_mode,
            window_left=window_left,
            logits_soft_cap=logits_soft_cap,
            q_data_type=q_data_type,
            kv_data_type=kv_data_type,
            o_data_type=o_data_type,
            data_type=data_type,
            sm_scale=sm_scale,
            rope_scale=rope_scale,
            rope_theta=rope_theta,
            non_blocking=non_blocking,
            block_tables=None,
            seq_lens=seq_lens_cpu,
            fixed_split_size=fixed_split_size,
            disable_split_kv=disable_split_kv,
            # q_len_per_req is passed only when FlashInfer supports it.
            **uniform_decode_kwargs,
        )
        staging_record(self)
        self.vllm_first_call = False
        return

    assert self.is_cuda_graph_enabled, "Should be cudagraph only here"

    # ================= FLASHINFER PLAN CACHE ==========
    # Skip the whole flashinfer call when the schedule provably cannot have changed.
    # kv_lens is excluded from the key STRUCTURALLY: the fa2 planner never reads kv_len_arr
    # (it re-derives KV length from kv_indptr in pages); indptr_cpu IS in the key.
    # >> fa2 ONLY. The SM90/SM100 planners DO read kv_len_arr per request. This path serves sm_86.
    # >> Single-entry by design: _plan_info is offsets into a per-wrapper workspace holding
    #    only the LAST plan. A dict cache would return offsets into overwritten bytes.
    # fa2 is enforced, not assumed: FlashInfer resolves backend="auto" to a
    # concrete name in the first plan() (always taken above), so anything other than
    # "fa2" here (fa3, trtllm-gen, cute-dsl, unresolved) bypasses the cache.
    _pending_plan_key = None
    if (
        envs.VLLM_FLASHINFER_PLAN_CACHE
        and getattr(self, "_vllm_plan_cache_ok", False)
        and getattr(self, "_backend", None) == "fa2"
    ):
        _key = (
            # the wrapper's resolved backend: plan_info layouts differ per backend
            getattr(self, "_backend", None),
            tuple(indptr_cpu.tolist()),
            len(last_page_len_cpu),
            q_len_per_req,
            smallq_mode,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            window_left,
            logits_soft_cap,
            fixed_split_size,
            disable_split_kv,
            str(q_data_type),
            str(kv_data_type),
            sm_scale,
        )
        # Report BEFORE branching so the line fires on the Nth call, hit or miss.
        _h = getattr(self, "_vllm_plan_cache_hits", 0)
        _m = getattr(self, "_vllm_plan_cache_misses", 0)
        _n = _h + _m
        if _n in (0, 1, 2, 9, 99) or _n % 1000 == 0:
            logger.info(
                "flashinfer plan cache: call %d, %d hits / %d misses (%.1f%% hit rate)",
                _n + 1, _h, _m, 100.0 * _h / max(1, _n))
        if getattr(self, "_vllm_plan_key", None) == _key and getattr(
            self, "_plan_info", None
        ) is not None:
            self._vllm_plan_cache_hits = _h + 1
            return
        # Publish the key only after the plan succeeds: a failed plan must not
        # leave a key that a retry would treat as a hit on the previous plan.
        self._vllm_plan_key = None
        _pending_plan_key = _key
        self._vllm_plan_cache_misses = _m + 1
    else:
        self._vllm_plan_key = None
    # =============== end FLASHINFER PLAN CACHE ===============

    staging_fence(self)
    fast_decode_plan(
        self,
        indptr=indptr_cpu,
        indices=indices,
        last_page_len=last_page_len_cpu,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        page_size=page_size,
        pos_encoding_mode=pos_encoding_mode,
        window_left=window_left,
        logits_soft_cap=logits_soft_cap,
        q_data_type=q_data_type,
        kv_data_type=kv_data_type,
        data_type=data_type,
        sm_scale=sm_scale,
        rope_scale=rope_scale,
        rope_theta=rope_theta,
        non_blocking=non_blocking,
        fixed_split_size=fixed_split_size,
        disable_split_kv=disable_split_kv,
        # q_len_per_req is passed only when FlashInfer supports it.
        **uniform_decode_kwargs,
    )
    staging_record(self)
    if _pending_plan_key is not None:
        self._vllm_plan_key = _pending_plan_key


@triton.jit
def _copy_page_indices_kernel(
    page_indices,
    block_table,
    block_table_stride,
    cu_num_blocks,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    row_ptr = block_table + req_idx * block_table_stride
    start_idx = tl.load(cu_num_blocks + req_idx)
    end_idx = tl.load(cu_num_blocks + req_idx + 1)
    num_blocks = end_idx - start_idx

    offset = tl.arange(0, BLOCK_SIZE)
    for i in tl.range(0, num_blocks, BLOCK_SIZE):
        block_ids = tl.load(row_ptr + i + offset, mask=i + offset < num_blocks)
        tl.store(
            page_indices + start_idx + i + offset,
            block_ids,
            mask=i + offset < num_blocks,
        )


@triton.jit
def _draft_bounded_kv_fixup_kernel(
    seq_lens,  # [num_reqs] int32, device: the TRUE draft-step lengths
    block_table,  # [num_reqs, max_blocks] int32
    block_table_stride,
    kv_indptr,  # [>= num_reqs + 1] int32 out
    kv_indices,  # [max_pages] int32 out
    kv_last_page_len,  # [>= num_reqs] int32 out
    page_size,
    BLOCK_R: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """VLLM_DRAFT_BOUNDED_PLAN: the page table _compute_flashinfer_kv_metadata would build
    from synced seq_lens, built on the device (one program per request; the prefix sum is
    recomputed per program, num_reqs <= max_num_seqs). Spec: draft_bounded_plan.
    kv_metadata_reference / fixup_kernel_emulation."""
    req = tl.program_id(0)
    rows = tl.arange(0, BLOCK_R)
    prev = tl.load(seq_lens + rows, mask=rows < req, other=0)
    start = tl.sum((prev + page_size - 1) // page_size, axis=0)
    seq = tl.load(seq_lens + req)
    nb = (seq + page_size - 1) // page_size
    if req == 0:
        tl.store(kv_indptr, 0)
    tl.store(kv_indptr + req + 1, start + nb)
    lpl = seq % page_size
    lpl = tl.where((lpl == 0) & (seq != 0), page_size, lpl)
    tl.store(kv_last_page_len + req, lpl)
    row_ptr = block_table + req * block_table_stride
    offset = tl.arange(0, BLOCK_SIZE)
    for i in tl.range(0, nb, BLOCK_SIZE):
        m = i + offset < nb
        ids = tl.load(row_ptr + i + offset, mask=m)
        tl.store(kv_indices + start + i + offset, ids, mask=m)
