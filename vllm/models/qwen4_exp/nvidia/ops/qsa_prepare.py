# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modified by the Merlin project; see NOTICE.
"""Fused QSA prepare kernel for Qwen4Exp."""

import torch

from vllm.models.qwen4_exp.common.qsa_cache import (
    QSA_RING_TAG_WORD,
    QSA_RING_TAIL_ELEMENTS,
)
from vllm.triton_utils import tl, triton
from vllm.v1.worker import guard_telemetry

_GUARD_BIAS = tl.constexpr(guard_telemetry.GUARD_BIAS)
# Int64 word of the ring tail that carries `logical_position + 1`.
_RING_TAG_WORD = tl.constexpr(QSA_RING_TAG_WORD)
# Ring-row tag record packing and the encodable range of the found-tag slot.
_REC_POS_BITS = tl.constexpr(guard_telemetry.RING_RECORD_POS_BITS)
_REC_REQ_BITS = tl.constexpr(guard_telemetry.RING_RECORD_REQ_BITS)
_REC_LAYER_BITS = tl.constexpr(guard_telemetry.RING_RECORD_LAYER_BITS)
_GUARD_MAX_ENC = tl.constexpr(guard_telemetry.GUARD_MAX_ENCODABLE)
_GUARD_MIN_ENC = tl.constexpr(guard_telemetry.GUARD_MIN_ENCODABLE)


@triton.jit
def _norm_rope(
    x,
    pos_t,
    pos_h,
    pos_w,
    cos_sin_ptr,
    cos_sin_stride,
    cos_sin_rows,  # bounds-guard instrumentation
    guard_ctr,  # bounds-guard persistent int64[3], SITE_QSA_ROPE_POS
    norm_weight_ptr,
    eps,
    IS_MROPE: tl.constexpr,
    MROPE_H: tl.constexpr,
    MROPE_W: tl.constexpr,
):
    """Apply Gemma RMSNorm and selected-axis NeoX RoPE to register rows."""
    # Bounds guard: the position is a row index into the cos/sin table that
    # nothing else bounds (the masks below bound the rotary pair, not the row).
    # On the pooled cross-chunk path it comes from a raw-key ring row that is
    # not zeroed on reuse, so it can be residue. Every mrope axis is a live row
    # index, so each is checked. The filler is strongly negative so a negative
    # offender never reads back as position 0.
    _FILL: tl.constexpr = -(1 << 62)
    bad_t = (pos_t < 0) | (pos_t >= cos_sin_rows)
    if IS_MROPE:
        bad_h = (pos_h < 0) | (pos_h >= cos_sin_rows)
        bad_w = (pos_w < 0) | (pos_w >= cos_sin_rows)
    else:
        # the non-mrope branch reads only pos_t, and with NO mask at
        # all, which is why this guard clamps rather than masks.
        # This mask is always false, but correctly shaped and typed for the
        # max below.
        bad_h = pos_h < pos_h
        bad_w = pos_w < pos_w
    guard_bad = bad_t | bad_h | bad_w
    guard_n = tl.sum(guard_bad.to(tl.int32), axis=0)
    if guard_n > 0:
        worst_t = tl.max(tl.where(bad_t, pos_t, _FILL), axis=0).to(tl.int64)
        worst_h = tl.max(tl.where(bad_h, pos_h, _FILL), axis=0).to(tl.int64)
        worst_w = tl.max(tl.where(bad_w, pos_w, _FILL), axis=0).to(tl.int64)
        worst = tl.maximum(worst_t, tl.maximum(worst_h, worst_w))
        # slot 2 names the axis: 0 = t, 1 = h, 2 = w. Derived from the MASKS
        # (deliberately), not by comparing the winner's VALUE against each
        # axis's max. Value comparison mis-attributes whenever two axes share
        # the worst value -- which is the common case here, because pos_h and
        # pos_w are frequently equal to pos_t (the non-2D path sets them so,
        # and a stale ring tail writes the same int64 into all three). Asking
        # "did this axis have any offender at all", in t/h/w order, cannot.
        any_t = tl.max(bad_t.to(tl.int32), axis=0) > 0
        any_h = tl.max(bad_h.to(tl.int32), axis=0) > 0
        axis = tl.where(any_t, 0, tl.where(any_h, 1, 2))
        tl.atomic_add(guard_ctr, guard_n.to(tl.int64))
        tl.atomic_max(guard_ctr + 1, worst + _GUARD_BIAS)
        tl.atomic_max(guard_ctr + 2, axis.to(tl.int64) + _GUARD_BIAS)
    # Clamp rather than skip: the rotation is then wrong for these rows, which
    # is what a masked fault IS. The counter is the result, not the output.
    pos_t = tl.minimum(tl.maximum(pos_t, 0), cos_sin_rows - 1)
    pos_h = tl.minimum(tl.maximum(pos_h, 0), cos_sin_rows - 1)
    pos_w = tl.minimum(tl.maximum(pos_w, 0), cos_sin_rows - 1)

    TILE_T: tl.constexpr = x.shape[0]
    TILE_H: tl.constexpr = x.shape[1]
    D: tl.constexpr = x.shape[2]
    ROWS: tl.constexpr = TILE_T * TILE_H
    HALF: tl.constexpr = D // 2
    QUARTER: tl.constexpr = D // 4
    pairs = tl.arange(0, QUARTER)
    if IS_MROPE:
        # Qwen interleaves temporal, height, and width rotary pairs. Each axis
        # still indexes the same position-major cos/sin table.
        h_mask = ((pairs % 3) == 1) & (pairs <= 3 * MROPE_H)
        w_mask = ((pairs % 3) == 2) & (pairs <= 3 * MROPE_W)
        t_mask = ~(h_mask | w_mask)
        base = cos_sin_ptr + pairs[None, :]
        pos_rows = (pos_t, pos_h, pos_w)
        axis_masks = (t_mask, h_mask, w_mask)
        cos = tl.zeros((TILE_T, QUARTER), dtype=cos_sin_ptr.dtype.element_ty)
        sin = tl.zeros((TILE_T, QUARTER), dtype=cos_sin_ptr.dtype.element_ty)
        for axis in tl.static_range(3):
            cos += tl.load(
                base + pos_rows[axis][:, None] * cos_sin_stride,
                mask=axis_masks[axis][None, :],
                other=0,
            )
            sin += tl.load(
                base + pos_rows[axis][:, None] * cos_sin_stride + QUARTER,
                mask=axis_masks[axis][None, :],
                other=0,
            )
    else:
        cos = tl.load(cos_sin_ptr + pos_t[:, None] * cos_sin_stride + pairs[None, :])
        sin = tl.load(
            cos_sin_ptr + pos_t[:, None] * cos_sin_stride + QUARTER + pairs[None, :]
        )

    cos = tl.reshape(
        tl.broadcast_to(cos[:, None, :], (TILE_T, TILE_H, QUARTER)),
        (ROWS, QUARTER),
    )
    sin = tl.reshape(
        tl.broadcast_to(sin[:, None, :], (TILE_T, TILE_H, QUARTER)),
        (ROWS, QUARTER),
    )
    x = tl.reshape(x, (ROWS, D)).to(tl.float32)
    weight = tl.load(norm_weight_ptr + tl.arange(0, D)).to(tl.float32) + 1.0
    rrms = tl.rsqrt(tl.sum(x * x, axis=1) / D + eps)
    y = (x * rrms[:, None] * weight[None, :]).to(cos.dtype)
    rotated, passthrough = tl.split(
        tl.permute(tl.reshape(y, (ROWS, 2, HALF)), (0, 2, 1))
    )
    r0, r1 = tl.split(tl.permute(tl.reshape(rotated, (ROWS, 2, QUARTER)), (0, 2, 1)))
    out0 = r0 * cos - r1 * sin
    out1 = r1 * cos + r0 * sin
    rotated = tl.reshape(tl.permute(tl.join(out0, out1), (0, 2, 1)), (ROWS, HALF))
    result = tl.reshape(tl.permute(tl.join(rotated, passthrough), (0, 2, 1)), (ROWS, D))
    return tl.reshape(result, (TILE_T, TILE_H, D))


@triton.jit
def _rope_pos_guard(
    pos_t,
    pos_h,
    pos_w,
    cos_sin_rows,
    guard_ctr,
    IS_MROPE: tl.constexpr,
):
    """The SITE_QSA_ROPE_POS check of ``_norm_rope`` without the rotation.

    Used on MTP steps that skip index-Q work (``WRITE_Q=False``): no cos/sin row
    is read there, but a bad position must still reach the same always-fatal
    guard site, with the same count, worst value and axis encoding.
    """
    _FILL: tl.constexpr = -(1 << 62)
    bad_t = (pos_t < 0) | (pos_t >= cos_sin_rows)
    if IS_MROPE:
        bad_h = (pos_h < 0) | (pos_h >= cos_sin_rows)
        bad_w = (pos_w < 0) | (pos_w >= cos_sin_rows)
    else:
        bad_h = pos_h < pos_h
        bad_w = pos_w < pos_w
    guard_bad = bad_t | bad_h | bad_w
    guard_n = tl.sum(guard_bad.to(tl.int32), axis=0)
    if guard_n > 0:
        worst_t = tl.max(tl.where(bad_t, pos_t, _FILL), axis=0).to(tl.int64)
        worst_h = tl.max(tl.where(bad_h, pos_h, _FILL), axis=0).to(tl.int64)
        worst_w = tl.max(tl.where(bad_w, pos_w, _FILL), axis=0).to(tl.int64)
        worst = tl.maximum(worst_t, tl.maximum(worst_h, worst_w))
        any_t = tl.max(bad_t.to(tl.int32), axis=0) > 0
        any_h = tl.max(bad_h.to(tl.int32), axis=0) > 0
        axis = tl.where(any_t, 0, tl.where(any_h, 1, 2))
        tl.atomic_add(guard_ctr, guard_n.to(tl.int64))
        tl.atomic_max(guard_ctr + 1, worst + _GUARD_BIAS)
        tl.atomic_max(guard_ctr + 2, axis.to(tl.int64) + _GUARD_BIAS)


@triton.jit
def _to_dst_dtype(x, dst, scale):
    """Round to BF16 like the unfused path, then scale for an FP8 destination."""
    out_ty = dst.dtype.element_ty
    x = x.to(tl.bfloat16)
    if out_ty == tl.float8e4nv:
        x = x.to(tl.float32) / scale
    return x.to(out_ty)


@triton.jit
def _store_rotated(dst, y, o1, o2, scale):
    """Store a normalized head whose first ``2 * len(o1)`` dims are rotated."""
    HALF: tl.constexpr = o1.shape[0]
    dims = tl.arange(0, y.shape[0])
    rot = tl.arange(0, HALF)
    tl.store(dst + dims, _to_dst_dtype(y, dst, scale), mask=dims >= 2 * HALF)
    tl.store(dst + rot, _to_dst_dtype(o1, dst, scale))
    tl.store(dst + HALF + rot, _to_dst_dtype(o2, dst, scale))


@triton.jit(
    do_not_specialize=[
        "num_tokens",
        "num_state_blocks",
        "num_compressed_blocks",
        "num_k_work",
        "num_requests",
        "layer_id",
    ]
)
def _qsa_prepare_kernel(
    q_ptr,
    q_stride_token,
    k_ptr,
    k_stride_token,
    pos_ptr,
    pos_stride_axis,
    pos_stride_token,
    cos_sin_ptr,
    cos_sin_rows,  # bounds-guard instrumentation
    rope_guard_ctr,  # bounds-guard persistent int64[3], SITE_QSA_ROPE_POS
    q_norm_weight_ptr,
    k_norm_weight_ptr,
    eps,
    q_out_ptr,
    q_out_stride_token,
    q_out_stride_head,
    state_cache_ptr,
    state_cache_stride_block,
    state_cache_stride_token,
    state_slots_ptr,
    state_table_ptr,
    state_table_stride_req,
    query_start_loc_ptr,
    logical_positions_ptr,
    compressed_slots_ptr,
    k_work_metadata_ptr,
    compressed_cache_ptr,
    compressed_cache_stride_block,
    compressed_cache_stride_token,
    num_tokens,
    num_state_blocks,
    num_compressed_blocks,
    num_k_work,
    num_requests,  # bounds-guard instrumentation
    guard_ctr,  # bounds-guard instrumentation: persistent int64[3]
    ring_skip_ctr,  # ring-skip telemetry, persistent int64[3], SITE_QSA_RING_SKIP
    ring_uncommitted_ctr,  # ring-row tag check, persistent int64[3], SITE_QSA_RING_UNCOMMITTED
    layer_id,  # named in the uncommitted-row record only
    HQ: tl.constexpr,
    D: tl.constexpr,
    TILE_T_Q: tl.constexpr,
    TILE_H_Q: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    STATE_SIZE: tl.constexpr,
    COMP_PAGE_SIZE: tl.constexpr,
    IS_2D_POSITIONS: tl.constexpr,
    IS_K_MROPE: tl.constexpr,
    CACHE_HAS_ROPE_POS: tl.constexpr,
    MROPE_H: tl.constexpr,
    MROPE_W: tl.constexpr,
    WRITE_Q: tl.constexpr,
    main_qkv_ptr,
    main_qkv_stride_token,
    main_q_norm_weight_ptr,
    main_k_norm_weight_ptr,
    main_eps,
    main_q_out_ptr,
    main_gate_out_ptr,
    main_cache_ptr,
    main_cache_stride_block,
    main_cache_stride_token,
    main_cache_stride_head,
    main_slots_ptr,
    main_k_scale,
    main_v_scale,
    MAIN_HQ: tl.constexpr,
    MAIN_HK: tl.constexpr,
    MAIN_D: tl.constexpr,
    MAIN_PAGE_SIZE: tl.constexpr,
    HAS_MAIN: tl.constexpr,
):
    pid = tl.program_id(0)
    if HAS_MAIN:
        num_index_work = num_k_work + tl.cdiv(num_tokens, TILE_T_Q) * tl.cdiv(
            HQ, TILE_H_Q
        )
        if pid >= num_index_work:
            # Main attention: one program per (token, Q or KV head) after the
            # indexer work. RoPE covers the first D // 2 dims, the width of the
            # shared cos/sin table.
            main_pid = pid - num_index_work
            token = (main_pid // (MAIN_HQ + MAIN_HK)).to(tl.int64)
            head = main_pid % (MAIN_HQ + MAIN_HK)
            HALF: tl.constexpr = D // 4
            dims = tl.arange(0, MAIN_D)
            rot = tl.arange(0, HALF)
            row = main_qkv_ptr + token * main_qkv_stride_token
            is_k = head >= MAIN_HQ
            kv_head = head - MAIN_HQ
            if is_k:
                src = row + (2 * MAIN_HQ + kv_head) * MAIN_D
                weight_ptr = main_k_norm_weight_ptr
            else:
                src = row + 2 * head * MAIN_D
                weight_ptr = main_q_norm_weight_ptr
            x = tl.load(src + dims).to(tl.float32)
            inv_rms = tl.rsqrt(tl.sum(x * x, axis=0) / MAIN_D + main_eps)
            w = tl.load(weight_ptr + dims).to(tl.float32) + 1.0
            y = x * inv_rms * w
            x1 = tl.load(src + rot).to(tl.float32)
            x2 = tl.load(src + HALF + rot).to(tl.float32)
            w1 = tl.load(weight_ptr + rot).to(tl.float32) + 1.0
            w2 = tl.load(weight_ptr + HALF + rot).to(tl.float32) + 1.0
            x1 = (x1 * inv_rms * w1).to(tl.bfloat16).to(tl.float32)
            x2 = (x2 * inv_rms * w2).to(tl.bfloat16).to(tl.float32)
            pos = tl.load(pos_ptr + token * pos_stride_token).to(tl.int64)
            if IS_2D_POSITIONS:
                pos_h = tl.load(pos_ptr + pos_stride_axis + token * pos_stride_token)
                pos_w = tl.load(
                    pos_ptr + 2 * pos_stride_axis + token * pos_stride_token
                )
                is_h = (rot % 3 == 1) & (rot < 3 * MROPE_H)
                is_w = (rot % 3 == 2) & (rot < 3 * MROPE_W)
                pos = tl.where(
                    is_h, pos_h.to(tl.int64), tl.where(is_w, pos_w.to(tl.int64), pos)
                )
            # Merlin bounds guard: these are the same positions the index-Q
            # programs below check and count at SITE_QSA_ROPE_POS; clamp here
            # so a bad one cannot form an out-of-table cos/sin address.
            pos = tl.minimum(tl.maximum(pos, 0), cos_sin_rows - 1)
            cos = tl.load(cos_sin_ptr + pos * (D // 2) + rot).to(tl.float32)
            sin = tl.load(cos_sin_ptr + pos * (D // 2) + HALF + rot).to(tl.float32)
            o1 = x1 * cos - x2 * sin
            o2 = x2 * cos + x1 * sin
            if is_k:
                slot = tl.load(main_slots_ptr + token).to(tl.int64)
                if slot >= 0:
                    dst = (
                        main_cache_ptr
                        + (slot // MAIN_PAGE_SIZE) * main_cache_stride_block
                        + (slot % MAIN_PAGE_SIZE) * main_cache_stride_token
                        + kv_head * main_cache_stride_head
                    )
                    _store_rotated(dst, y, o1, o2, main_k_scale)
                    v = tl.load(src + MAIN_HK * MAIN_D + dims)
                    tl.store(dst + MAIN_D + dims, _to_dst_dtype(v, dst, main_v_scale))
            else:
                out = (token * MAIN_HQ + head) * MAIN_D
                _store_rotated(main_q_out_ptr + out, y, o1, o2, None)
                gate = tl.load(src + MAIN_D + dims)
                tl.store(main_gate_out_ptr + out + dims, gate)
            return
    # K work occupies the first programs; the remaining programs tile Q. This
    # keeps both paths in one launch while leaving their register shapes
    # independent.
    if WRITE_Q:
        if pid >= num_k_work:
            q_pid = pid - num_k_work
            num_head_tiles: tl.constexpr = tl.cdiv(HQ, TILE_H_Q)
            token_tile = q_pid // num_head_tiles
            head_tile = q_pid % num_head_tiles
            tokens = (token_tile * TILE_T_Q + tl.arange(0, TILE_T_Q)).to(tl.int64)
            heads = head_tile * TILE_H_Q + tl.arange(0, TILE_H_Q)
            valid_tokens = tokens < num_tokens
            valid_heads = heads < HQ
            dims = tl.arange(0, D)
            mask = valid_tokens[:, None, None] & valid_heads[None, :, None]
            x = tl.load(
                q_ptr
                + tokens[:, None, None] * q_stride_token
                + heads[None, :, None] * D
                + dims[None, None, :],
                mask=mask,
                other=0.0,
            )
            pos_t = tl.load(pos_ptr + tokens * pos_stride_token, mask=valid_tokens, other=0)
            if IS_2D_POSITIONS:
                pos_h = tl.load(
                    pos_ptr + pos_stride_axis + tokens * pos_stride_token,
                    mask=valid_tokens,
                    other=0,
                )
                pos_w = tl.load(
                    pos_ptr + 2 * pos_stride_axis + tokens * pos_stride_token,
                    mask=valid_tokens,
                    other=0,
                )
            else:
                pos_h = pos_t
                pos_w = pos_t
            y = _norm_rope(
                x,
                pos_t,
                pos_h,
                pos_w,
                cos_sin_ptr,
                D // 2,
                cos_sin_rows,
                rope_guard_ctr,
                q_norm_weight_ptr,
                eps,
                IS_2D_POSITIONS,
                MROPE_H,
                MROPE_W,
            )
            tl.store(
                q_out_ptr
                + tokens[:, None, None] * q_out_stride_token
                + heads[None, :, None] * q_out_stride_head
                + dims[None, None, :],
                y,
                mask=mask,
            )
    else:
        if pid >= num_k_work:
            # Index-Q is unused on this step; keep only its position guard.
            q_pid = pid - num_k_work
            num_head_tiles_g: tl.constexpr = tl.cdiv(HQ, TILE_H_Q)
            token_tile = q_pid // num_head_tiles_g
            tokens = (token_tile * TILE_T_Q + tl.arange(0, TILE_T_Q)).to(tl.int64)
            valid_tokens = tokens < num_tokens
            pos_t = tl.load(pos_ptr + tokens * pos_stride_token, mask=valid_tokens, other=0)
            if IS_2D_POSITIONS:
                pos_h = tl.load(
                    pos_ptr + pos_stride_axis + tokens * pos_stride_token,
                    mask=valid_tokens,
                    other=0,
                )
                pos_w = tl.load(
                    pos_ptr + 2 * pos_stride_axis + tokens * pos_stride_token,
                    mask=valid_tokens,
                    other=0,
                )
            else:
                pos_h = pos_t
                pos_w = pos_t
            _rope_pos_guard(
                pos_t, pos_h, pos_w, cos_sin_rows, rope_guard_ctr, IS_2D_POSITIONS
            )

    if pid < num_k_work:
        # One work item owns one completed compression group. Work item zero
        # additionally commits the request's current raw-K suffix below.
        work_metadata = tl.load(k_work_metadata_ptr + pid * 2 + tl.arange(0, 2))
        request, work_in_request = tl.split(work_metadata)
        if request < 0:
            return
        # Second bounds guard: the early return above rejects only the -1
        # sentinel the builder writes for inactive work items. Any other
        # corrupted value was loaded straight out of query_start_loc, which has
        # num_requests + 1 entries -- 8 bytes for a single-request batch. The
        # chunk_end load below derives its index from query_end, so a bad row
        # also walked logical_positions; returning here stops both.
        if request >= num_requests:
            tl.atomic_add(guard_ctr, 1)
            tl.atomic_max(guard_ctr + 1, (request).to(tl.int64) + _GUARD_BIAS)
            tl.atomic_max(guard_ctr + 2, (work_in_request).to(tl.int64) + _GUARD_BIAS)
            return

        query_start = tl.load(query_start_loc_ptr + request)
        query_end = tl.load(query_start_loc_ptr + request + 1)
        query_len = query_end - query_start
        chunk_end = tl.load(logical_positions_ptr + query_end - 1)
        chunk_start = chunk_end - query_len + 1
        num_groups = (chunk_end + 1) // COMPRESS_RATIO - chunk_start // COMPRESS_RATIO
        dims = tl.arange(0, D)

        if work_in_request < num_groups:
            first_boundary = (
                (chunk_start + COMPRESS_RATIO) // COMPRESS_RATIO
            ) * COMPRESS_RATIO - 1
            end_position = first_boundary + work_in_request * COMPRESS_RATIO
            boundary_token = query_start + end_position - chunk_start
            valid_token = (
                (boundary_token >= query_start)
                & (boundary_token < query_end)
                & (boundary_token < num_tokens)
            )
            compressed_slot = tl.load(
                compressed_slots_ptr + boundary_token,
                mask=valid_token,
                other=-1,
            )
            valid = (
                valid_token
                & (compressed_slot >= 0)
                & (compressed_slot < num_compressed_blocks * COMP_PAGE_SIZE)
            )
            state_block = tl.load(state_table_ptr + request * state_table_stride_req)
            state_block_valid = (state_block >= 0) & (state_block < num_state_blocks)
            safe_state_block = tl.maximum(state_block, 0).to(tl.int64)
            group_offsets = tl.arange(0, COMPRESS_RATIO)
            source_positions = end_position - (COMPRESS_RATIO - 1) + group_offsets
            source_in_chunk = source_positions >= chunk_start
            source_tokens = query_start + source_positions - chunk_start
            source_tokens_valid = (
                (source_tokens >= query_start)
                & (source_tokens < query_end)
                & (source_tokens < num_tokens)
            )
            current_base = (
                k_ptr
                + tl.maximum(source_tokens, 0).to(tl.int64)[:, None] * k_stride_token
            )
            cached_base = (
                state_cache_ptr
                + safe_state_block * state_cache_stride_block
                + (source_positions % STATE_SIZE)[:, None] * state_cache_stride_token
            )
            # EVERY historical source row must carry the tag the writer stores
            # for exactly this logical position, before anything reads its key
            # or its position tail. A missing ring block (state table entry
            # out of range) reads tag 0 and fails the same check. In-chunk rows
            # come from this launch's own k and are never checked.
            historical = valid & ~source_in_chunk
            source_tags = tl.load(
                (
                    state_cache_ptr
                    + safe_state_block * state_cache_stride_block
                    + (source_positions % STATE_SIZE) * state_cache_stride_token
                    + D
                ).to(tl.pointer_type(tl.int64))
                + _RING_TAG_WORD,
                mask=historical & state_block_valid,
                other=0,
            )
            row_bad = historical & (source_tags != source_positions.to(tl.int64) + 1)
            num_bad_rows = tl.sum(row_bad.to(tl.int32), axis=0)
            group_ok = num_bad_rows == 0
            if num_bad_rows > 0:
                # One coherent record per site, first offender wins: the CAS
                # claims slot 1 and only its winner writes slot 2, so both
                # fields always describe the same row.
                bad_position = tl.min(
                    tl.where(row_bad, source_positions.to(tl.int64), 1 << 62),
                    axis=0,
                )
                bad_tag = tl.sum(
                    tl.where(
                        row_bad & (source_positions.to(tl.int64) == bad_position),
                        source_tags,
                        0,
                    ),
                    axis=0,
                )
                record = (
                    (
                        ((layer_id + 1).to(tl.int64) & ((1 << _REC_LAYER_BITS) - 1))
                        << (_REC_REQ_BITS + _REC_POS_BITS)
                    )
                    | (
                        (request.to(tl.int64) & ((1 << _REC_REQ_BITS) - 1))
                        << _REC_POS_BITS
                    )
                    | (bad_position & ((1 << _REC_POS_BITS) - 1))
                )
                tl.atomic_add(ring_uncommitted_ctr, num_bad_rows.to(tl.int64))
                zero64 = bad_position * 0
                prev = tl.atomic_cas(
                    ring_uncommitted_ctr + 1, zero64, record + _GUARD_BIAS
                )
                if prev == 0:
                    clamped_tag = tl.minimum(
                        tl.maximum(bad_tag, _GUARD_MIN_ENC), _GUARD_MAX_ENC
                    )
                    tl.atomic_xchg(ring_uncommitted_ctr + 2, clamped_tag + _GUARD_BIAS)
            # Only the first completed group can cross the chunk boundary. Select
            # historical rows from the ring without issuing two masked loads.
            source_base = tl.where(source_in_chunk[:, None], current_base, cached_base)
            # Pointer selection obscures alignment from Triton's analysis.
            source_base = tl.multiple_of(source_base, (8, 8))
            source_valid = tl.where(
                source_in_chunk, source_tokens_valid, state_block_valid
            )
            # A failed group loads nothing (the mask, not a tl.where on the
            # result, is what suppresses the memory operation).
            source = tl.load(
                source_base + dims[None, :],
                mask=valid & group_ok & source_valid[:, None],
                other=0.0,
            ).to(tl.float32)
            # Match the unfused path's BF16 pooled tensor before RMSNorm.
            pooled = (
                (tl.sum(source, axis=0) / COMPRESS_RATIO).to(tl.bfloat16).to(tl.float32)
            )

            first_position = end_position - (COMPRESS_RATIO - 1)
            if CACHE_HAS_ROPE_POS:
                # RoPE uses the first token in the pooled group. Its exact MRoPE
                # coordinates may live in this chunk or the raw-state ring.
                first_in_chunk = first_position >= chunk_start
                first_token = query_start + first_position - chunk_start
                first_token_valid = (
                    (first_token >= query_start)
                    & (first_token < query_end)
                    & (first_token < num_tokens)
                )
                safe_first_token = tl.maximum(first_token, 0)
                load_current_position = first_in_chunk & first_token_valid
                first_pos_t = tl.load(
                    pos_ptr + safe_first_token * pos_stride_token,
                    mask=load_current_position,
                    other=0,
                )
                if IS_2D_POSITIONS:
                    first_pos_h = tl.load(
                        pos_ptr + pos_stride_axis + safe_first_token * pos_stride_token,
                        mask=load_current_position,
                        other=0,
                    )
                    first_pos_w = tl.load(
                        pos_ptr
                        + 2 * pos_stride_axis
                        + safe_first_token * pos_stride_token,
                        mask=load_current_position,
                        other=0,
                    )
                else:
                    first_pos_h = first_pos_t
                    first_pos_w = first_pos_t
                tail = (
                    state_cache_ptr
                    + safe_state_block * state_cache_stride_block
                    + (first_position % STATE_SIZE) * state_cache_stride_token
                    + D
                ).to(tl.pointer_type(tl.int64))
                # The first row's tag was validated above with the rest;
                # a failed group never reads its tail (pos stays 0, in range).
                load_cached_position = ~first_in_chunk & state_block_valid & group_ok
                cached_pos_t = tl.load(tail, mask=load_cached_position, other=0)
                cached_pos_h = tl.load(tail + 1, mask=load_cached_position, other=0)
                cached_pos_w = tl.load(tail + 2, mask=load_cached_position, other=0)
                pos_t = tl.where(first_in_chunk, first_pos_t.to(tl.int64), cached_pos_t)
                pos_h = tl.where(first_in_chunk, first_pos_h.to(tl.int64), cached_pos_h)
                pos_w = tl.where(first_in_chunk, first_pos_w.to(tl.int64), cached_pos_w)
            else:
                pos_t = first_position
                pos_h = first_position
                pos_w = first_position
            y = _norm_rope(
                tl.reshape(pooled, (1, 1, D)),
                pos_t + tl.arange(0, 1),
                pos_h + tl.arange(0, 1),
                pos_w + tl.arange(0, 1),
                cos_sin_ptr,
                D // 2,
                cos_sin_rows,
                rope_guard_ctr,
                k_norm_weight_ptr,
                eps,
                IS_K_MROPE,
                MROPE_H,
                MROPE_W,
            )
            compressed_block = (compressed_slot // COMP_PAGE_SIZE).to(tl.int64)
            compressed_row = compressed_slot % COMP_PAGE_SIZE
            # A group built from an uncommitted row must never become
            # usable compressed state. The slot stays unwritten; the site is
            # fatal because the selection kernels will still read it.
            tl.store(
                compressed_cache_ptr
                + compressed_block * compressed_cache_stride_block
                + compressed_row * compressed_cache_stride_token
                + dims,
                tl.reshape(y, (D,)),
                mask=valid & group_ok,
            )

        if work_in_request == 0:
            # This CTA may have just read historical rows from the circular buffer.
            # Keep every lane past those loads before overwriting the same ring.
            tl.debug_barrier()
            # One CTA per request commits only the suffix retained by the ring.
            num_state_rows = tl.minimum(query_len, STATE_SIZE)
            for state_offset in tl.range(0, num_state_rows):
                token = (query_end - num_state_rows + state_offset).to(tl.int64)
                valid_token = (
                    (token >= query_start) & (token < query_end) & (token < num_tokens)
                )
                slot = tl.load(state_slots_ptr + token, mask=valid_token, other=-1)
                valid_slot = (
                    valid_token & (slot >= 0) & (slot < num_state_blocks * STATE_SIZE)
                )
                # Ring-skip telemetry: a live token whose slot fails the range check
                # has its ring row silently NOT written (every store below is
                # masked by valid_slot). That token's raw key is then absent
                # from the ring the next step reads. Count it; the write stays
                # skipped.
                # slot == -1 is the PAD sentinel (upstream #56734: idx_mapping == -1
                # is a CUDA-graph padding request row, whose state slot is -1). It
                # is still skipped (valid_slot is False) but is not a fault, so it
                # is not counted; < -1 and >= capacity are still counted.
                skipped = valid_token & (slot != -1) & ~valid_slot
                tl.atomic_add(ring_skip_ctr, 1, mask=skipped)
                tl.atomic_max(
                    ring_skip_ctr + 1, token.to(tl.int64) + _GUARD_BIAS, mask=skipped
                )
                tl.atomic_max(
                    ring_skip_ctr + 2, slot.to(tl.int64) + _GUARD_BIAS, mask=skipped
                )
                safe_slot = tl.maximum(slot, 0)
                state_row = (
                    state_cache_ptr
                    + (safe_slot // STATE_SIZE).to(tl.int64) * state_cache_stride_block
                    + (safe_slot % STATE_SIZE) * state_cache_stride_token
                )
                k = tl.load(
                    k_ptr + tl.maximum(token, 0) * k_stride_token + dims,
                    mask=valid_slot,
                    other=0.0,
                )
                tl.store(state_row + dims, k, mask=valid_slot)
                if CACHE_HAS_ROPE_POS:
                    pos_t = tl.load(
                        pos_ptr + tl.maximum(token, 0) * pos_stride_token,
                        mask=valid_slot,
                        other=0,
                    )
                    if IS_2D_POSITIONS:
                        pos_h = tl.load(
                            pos_ptr
                            + pos_stride_axis
                            + tl.maximum(token, 0) * pos_stride_token,
                            mask=valid_slot,
                            other=0,
                        )
                        pos_w = tl.load(
                            pos_ptr
                            + 2 * pos_stride_axis
                            + tl.maximum(token, 0) * pos_stride_token,
                            mask=valid_slot,
                            other=0,
                        )
                    else:
                        pos_h = pos_t
                        pos_w = pos_t
                    tail = (state_row + D).to(tl.pointer_type(tl.int64))
                    tl.store(tail, pos_t.to(tl.int64), mask=valid_slot)
                    tl.store(tail + 1, pos_h.to(tl.int64), mask=valid_slot)
                    tl.store(tail + 2, pos_w.to(tl.int64), mask=valid_slot)
                # Ring-row tag: the tag names the logical position this row now
                # holds, taken from the authoritative per-token metadata (not
                # re-derived from chunk arithmetic), stored after key + tail.
                # Every committed row gets one, with or without RoPE positions.
                token_logical = tl.load(
                    logical_positions_ptr + tl.maximum(token, 0),
                    mask=valid_slot,
                    other=-1,
                )
                tl.store(
                    (state_row + D).to(tl.pointer_type(tl.int64)) + _RING_TAG_WORD,
                    token_logical.to(tl.int64) + 1,
                    mask=valid_slot,
                )


def _qsa_prepare_work_counts(
    num_tokens: int,
    num_q_heads: int,
    num_k_work: int,
    tile_t_q: int,
    tile_h_q: int,
    *,
    write_q: bool,
) -> tuple[int, int]:
    """Return K and Q program counts; disabling Q leaves K work unchanged.

    With ``write_q`` off the Q programs still run, guard-only (positions are
    checked, nothing is normalised, rotated or stored), so the count is the same.
    """
    del write_q
    num_q_work = triton.cdiv(num_tokens, tile_t_q) * triton.cdiv(num_q_heads, tile_h_q)
    return num_k_work, num_q_work


def qsa_prepare(
    q: torch.Tensor,
    k: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    eps: float,
    q_out: torch.Tensor,
    state_cache: torch.Tensor,
    state_slots: torch.Tensor,
    state_block_table: torch.Tensor,
    query_start_loc: torch.Tensor,
    logical_positions: torch.Tensor,
    compressed_cache: torch.Tensor,
    compressed_slots: torch.Tensor,
    k_work_metadata: torch.Tensor,
    *,
    compress_ratio: int,
    mrope_section: tuple[int, int, int] | None,
    rope_pos_offset: int | None,
    layer_id: int = -1,
    write_q: bool = True,
    main_qkv: torch.Tensor | None = None,
    main_q_norm_weight: torch.Tensor | None = None,
    main_k_norm_weight: torch.Tensor | None = None,
    main_eps: float | None = None,
    main_kv_cache: torch.Tensor | None = None,
    main_slot_mapping: torch.Tensor | None = None,
    main_k_scale: float | None = None,
    main_v_scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Normalize Q, compress K, then update the circular raw state.

    ``state_cache`` rows are ``[key D bf16 | 4 int64: pos_t, pos_h, pos_w, tag]``
    (``QSAKeyStateCache``); the tail is present with or without
    ``rope_pos_offset``. ``layer_id`` only labels the uncommitted-row record.

    With the ``main_*`` operands (all or none), the same launch also prepares
    the main attention (QK-norm/RoPE, gate copy, K/V cache write) and returns
    its Q and gate (vllm#57097). Without them it is the indexer-only launch and
    returns None (the Merlin SM86 E4M3 path, whose main K/V write stays on
    reshape_and_cache_flash).
    """
    main_operands = (
        main_qkv,
        main_q_norm_weight,
        main_k_norm_weight,
        main_eps,
        main_kv_cache,
        main_slot_mapping,
        main_k_scale,
        main_v_scale,
    )
    has_main = main_qkv is not None
    assert all((op is not None) == has_main for op in main_operands), (
        "qsa_prepare: pass all main_* operands or none"
    )
    num_tokens = q.shape[0]
    main_outputs: tuple[torch.Tensor, torch.Tensor] | None = None
    if has_main:
        assert main_qkv is not None and main_kv_cache is not None
        main_head_dim = main_kv_cache.shape[-1] // 2
        num_main_kv_heads = main_kv_cache.shape[2]
        num_main_q_heads = (
            main_qkv.shape[1] // (2 * main_head_dim) - num_main_kv_heads
        )
        main_q_out = main_qkv.new_empty(num_tokens, num_main_q_heads, main_head_dim)
        main_gate_out = torch.empty_like(main_q_out)
        main_outputs = (main_q_out, main_gate_out)
    if num_tokens == 0:
        return main_outputs
    num_q_heads, head_dim = q_out.shape[1:]
    assert cos_sin_cache.shape[-1] * 2 == head_dim
    assert q.shape == (num_tokens, num_q_heads * head_dim)
    assert k.shape == (num_tokens, head_dim)
    assert q.stride(-1) == 1
    assert k.stride(-1) == 1
    assert q_out.stride(-1) == 1
    assert cos_sin_cache.is_contiguous()
    assert state_cache.stride(-1) == 1
    assert compressed_cache.stride(-1) == 1
    assert k_work_metadata.ndim == 2 and k_work_metadata.shape[1] == 2
    is_2d_positions = positions.ndim == 2
    is_k_mrope = bool(mrope_section)
    cache_has_rope_pos = rope_pos_offset is not None
    assert rope_pos_offset is None or rope_pos_offset == head_dim
    assert state_cache.shape[-1] == head_dim + QSA_RING_TAIL_ELEMENTS, (
        f"QSA ring row width {state_cache.shape[-1]} != head_dim + "
        f"{QSA_RING_TAIL_ELEMENTS} (key + pos_t/h/w + tag)"
    )
    assert state_cache.dtype == torch.bfloat16
    assert state_cache.stride(1) % 4 == 0 and state_cache.stride(0) % 4 == 0
    if has_main:
        assert main_qkv is not None and main_slot_mapping is not None
        qkv_width = 2 * (num_main_q_heads + num_main_kv_heads) * main_head_dim
        assert main_qkv.shape == (num_tokens, qkv_width)
        assert main_qkv.stride(-1) == 1
        assert main_slot_mapping.shape == (num_tokens,)
        num_main_work = num_tokens * (num_main_q_heads + num_main_kv_heads)
        main_args = dict(
            main_qkv_ptr=main_qkv,
            main_qkv_stride_token=main_qkv.stride(0),
            main_q_norm_weight_ptr=main_q_norm_weight,
            main_k_norm_weight_ptr=main_k_norm_weight,
            main_eps=main_eps,
            main_q_out_ptr=main_q_out,
            main_gate_out_ptr=main_gate_out,
            main_cache_ptr=main_kv_cache,
            main_cache_stride_block=main_kv_cache.stride(0),
            main_cache_stride_token=main_kv_cache.stride(1),
            main_cache_stride_head=main_kv_cache.stride(2),
            main_slots_ptr=main_slot_mapping,
            main_k_scale=main_k_scale,
            main_v_scale=main_v_scale,
            MAIN_HQ=num_main_q_heads,
            MAIN_HK=num_main_kv_heads,
            MAIN_D=main_head_dim,
            MAIN_PAGE_SIZE=main_kv_cache.shape[1],
        )
    else:
        # Indexer-only launch: the main block is compiled out (HAS_MAIN=False);
        # its pointer operands only need a valid tensor.
        num_main_work = 0
        main_args = dict(
            main_qkv_ptr=q,
            main_qkv_stride_token=0,
            main_q_norm_weight_ptr=q,
            main_k_norm_weight_ptr=q,
            main_eps=0.0,
            main_q_out_ptr=q,
            main_gate_out_ptr=q,
            main_cache_ptr=q,
            main_cache_stride_block=0,
            main_cache_stride_token=0,
            main_cache_stride_head=0,
            main_slots_ptr=q,
            main_k_scale=1.0,
            main_v_scale=1.0,
            MAIN_HQ=1,
            MAIN_HK=1,
            MAIN_D=1,
            MAIN_PAGE_SIZE=1,
        )
    if is_2d_positions:
        assert positions.shape == (3, num_tokens)
        assert is_k_mrope
        pos_stride_axis, pos_stride_token = positions.stride()
    else:
        assert positions.shape == (num_tokens,)
        pos_stride_axis, pos_stride_token = 0, positions.stride(0)
    section = mrope_section if mrope_section is not None else (0, 0, 0)
    assert len(section) == 3

    if num_tokens <= 4096:
        TILE_T_Q, TILE_H_Q = 2, 2
    else:
        TILE_T_Q, TILE_H_Q = 2, 4
    num_k_work = k_work_metadata.shape[0]
    num_k_work, num_q_work = _qsa_prepare_work_counts(
        num_tokens,
        num_q_heads,
        num_k_work,
        TILE_T_Q,
        TILE_H_Q,
        write_q=write_q,
    )
    _qsa_prepare_kernel[(num_k_work + num_q_work + num_main_work,)](
        q,
        q.stride(0),
        k,
        k.stride(0),
        positions,
        pos_stride_axis,
        pos_stride_token,
        cos_sin_cache,
        cos_sin_cache.shape[0],
        guard_telemetry.counter(guard_telemetry.SITE_QSA_ROPE_POS, q.device),
        q_norm_weight,
        k_norm_weight,
        eps,
        q_out,
        q_out.stride(0),
        q_out.stride(1),
        state_cache,
        state_cache.stride(0),
        state_cache.stride(1),
        state_slots,
        state_block_table,
        state_block_table.stride(0),
        query_start_loc,
        logical_positions,
        compressed_slots,
        k_work_metadata,
        compressed_cache,
        compressed_cache.stride(0),
        compressed_cache.stride(1),
        num_tokens,
        state_cache.shape[0],
        compressed_cache.shape[0],
        num_k_work,
        min(query_start_loc.numel() - 1, state_block_table.shape[0]),
        guard_telemetry.counter(guard_telemetry.SITE_QSA_PRE_INDEXER, q.device),
        # Ring-skip counter: the kernel exempts state slot -1 (the PAD sentinel,
        # upstream #56734) from the count, so a LIVE token with a missing (-1)
        # mapping is uncounted too. It cannot tell the two apart: padded rows
        # arrive with a zero-filled state-table row (gather_block_tables) and
        # the kernel is not handed idx_mapping. Separating them needs a new
        # kernel input (idx_mapping or a real-request flag); not done here.
        guard_telemetry.counter(guard_telemetry.SITE_QSA_RING_SKIP, q.device),
        guard_telemetry.counter(guard_telemetry.SITE_QSA_RING_UNCOMMITTED, q.device),
        int(layer_id),
        HQ=num_q_heads,
        D=head_dim,
        TILE_T_Q=TILE_T_Q,
        TILE_H_Q=TILE_H_Q,
        COMPRESS_RATIO=compress_ratio,
        STATE_SIZE=state_cache.shape[1],
        COMP_PAGE_SIZE=compressed_cache.shape[1],
        IS_2D_POSITIONS=is_2d_positions,
        IS_K_MROPE=is_k_mrope,
        CACHE_HAS_ROPE_POS=cache_has_rope_pos,
        MROPE_H=section[1],
        MROPE_W=section[2],
        WRITE_Q=write_q,
        **main_args,
        HAS_MAIN=has_main,
        num_warps=1,
    )
    return main_outputs


__all__ = ["qsa_prepare"]
