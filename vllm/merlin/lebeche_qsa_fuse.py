# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lebeche fork: sm_80/86 QSA decode glue chain fusion (MERLIN_LEBECHE_QSA_CHAIN_FUSE).

Bit-exact (0 ULP vs commit f332186) optimizations for Qwen4Exp QSA decode on
RTX 3090 (sm_80/86):
  1. Direct-block sparse paged attention + early-exit split-K + active-split
     merge: reads compressed ``block_indices`` directly (bypassing
     ``_expand_qsa_indices_kernel``, 4x fewer block-table gathers), exits
     inactive splits immediately before loading Q or writing partials, masks
     merge loads to ``active_splits``, uses ``num_warps=4, num_stages=1`` for
     sm_80/86 E4M3 K/V (avoiding Triton 3.8.0 register spills), and eliminates
     dead ``output.zero_()`` when ``num_tokens == output.shape[0]``.
  2. Bounded-grid decode MQA indexer with ``max_visible <= block_topk`` skip:
     launches at most 328 CTAs (vs 2048..16384 on 512K page tables) using
     ``tl.num_programs(1)`` (zero extra JIT specializations) and skips dead
     MQA scoring when ``persistent_topk`` takes its trivial branch.

Inert on non-sm_80/86 devices and when ``MERLIN_LEBECHE_QSA_CHAIN_FUSE=0``.
"""

from __future__ import annotations

from functools import lru_cache

import torch

from vllm.merlin.envs import read_registered
from vllm.model_executor.warmup.jit_warmup_triton_helper import (
    TritonWarmupTensor,
    triton_scalar_specialization_rep,
)
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON, tl, triton
from vllm.v1.worker import guard_telemetry

QSA_CHAIN_FUSE_SWITCH: bool = bool(read_registered("MERLIN_LEBECHE_QSA_CHAIN_FUSE"))
_GUARD_BIAS = tl.constexpr(guard_telemetry.GUARD_BIAS)

# Direct-block decode path covers base_programs = num_rows * num_kv_heads <= 24
# (M <= 24 at TP=2/1 KV head: covers single-stream M=4, MTP M=1, and c=2..6;
# at M >= 25 stock split-K with num_stages=2 remains faster at full vc=2048).
QSA_DIRECT_MAX_BASE_PROGRAMS = 24
# Bounded-grid target CTA budget for RTX 3090 (82 SMs * 4 CTAs/SM = 328).
QSA_MQA_BOUNDED_TOTAL_CTAS = 328
_MQA_DECODE_BLOCK_N = 64


@lru_cache(maxsize=1)
def qsa_chain_fuse_active() -> bool:
    if not QSA_CHAIN_FUSE_SWITCH or not HAS_TRITON or not current_platform.is_cuda():
        return False
    cap = current_platform.get_device_capability()
    return cap is not None and cap.to_int() in (80, 86)


@triton.jit
def _decode_e4m3_to_bf16(b):
    b = b.to(tl.uint16)
    bits = ((b & 0x80) << 8) | ((b & 0x7F) << 7)
    decoded = bits.to(tl.uint16).to(tl.float16, bitcast=True) * 256.0
    decoded = tl.where((b & 0x7F) == 0x7F, float("nan"), decoded)
    return decoded.to(tl.bfloat16)


@triton.jit(do_not_specialize=["num_rows", "num_requests"])
def _qsa_direct_splitk_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    k_scale_ptr,
    v_scale_ptr,
    block_indices_ptr,
    query_positions_ptr,
    visible_blocks_ptr,
    block_table_ptr,
    token_to_req_ptr,
    partial_output_ptr,
    partial_lse_ptr,
    output_ptr,
    output_gate_ptr,
    stride_q_row,
    stride_q_head,
    stride_k_block,
    stride_k_token,
    stride_k_head,
    stride_v_block,
    stride_v_token,
    stride_v_head,
    stride_blk_row,
    stride_table_req,
    stride_output_row,
    stride_output_head,
    stride_output_gate_row,
    stride_output_gate_head,
    num_rows,
    num_cache_blocks,
    num_requests,
    BLOCK_TOPK: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    TOPK: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_QUERY_HEADS: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    NUM_TILES: tl.constexpr,
    KV_DTYPE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    kv_head = tl.program_id(1)
    split_id = tl.program_id(2)

    query_position = tl.load(query_positions_ptr + row)
    visible_blocks = tl.load(visible_blocks_ptr + row)
    complete_blocks = tl.maximum(tl.minimum(visible_blocks, BLOCK_TOPK), 0)
    expanded_count = complete_blocks * COMPRESS_RATIO
    safe_qpos_p1 = tl.maximum(query_position + 1, 0)
    tail_start = (safe_qpos_p1 // COMPRESS_RATIO) * COMPRESS_RATIO
    tail_count = tl.where(query_position >= 0, safe_qpos_p1 - tail_start, 0)
    valid_count = expanded_count + tail_count
    tile_end = tl.minimum(NUM_TILES, tl.cdiv(tl.minimum(valid_count, TOPK), BLOCK_N))

    if NUM_SPLITS > 1 and split_id >= tile_end:
        return

    request = tl.load(token_to_req_ptr + row)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)
    req_valid = (request >= 0) & (request < num_requests)

    head_offsets = tl.arange(0, BLOCK_M)
    dim_offsets = tl.arange(0, HEAD_DIM)
    first_head = kv_head * GROUP_SIZE
    query = tl.load(
        q_ptr
        + row * stride_q_row
        + (first_head + head_offsets[:, None]) * stride_q_head
        + dim_offsets[None, :],
        mask=head_offsets[:, None] < GROUP_SIZE,
        other=0.0,
    )

    max_value = tl.full((BLOCK_M,), -1.0e20, dtype=tl.float32)
    normalizer = tl.zeros((BLOCK_M,), dtype=tl.float32)
    accumulator = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)
    softmax_scale_log2: tl.constexpr = (HEAD_DIM**-0.5) * 1.4426950408889634

    if KV_DTYPE != 0:
        k_scale_value = tl.load(k_scale_ptr)
        v_scale_value = tl.load(v_scale_ptr)

    BLK_PER_TILE: tl.constexpr = BLOCK_N // COMPRESS_RATIO
    blk_sub = tl.arange(0, BLK_PER_TILE)
    tok_sub = tl.arange(0, COMPRESS_RATIO)
    k_head_off = kv_head * stride_k_head
    v_head_off = kv_head * stride_v_head

    for tile in range(split_id, tile_end, NUM_SPLITS):
        blk_ranks = tile * BLK_PER_TILE + blk_sub
        is_exp_blk = blk_ranks < complete_blocks
        is_tail_blk = (blk_ranks == complete_blocks) & (tail_count > 0)
        safe_rank = tl.minimum(blk_ranks, BLOCK_TOPK - 1)
        blk_id = tl.load(
            block_indices_ptr + row * stride_blk_row + safe_rank,
            mask=is_exp_blk,
            other=-1,
        )
        base_tok = tl.where(is_exp_blk, blk_id * COMPRESS_RATIO, tail_start)
        blk_live = req_valid & (
            (is_exp_blk & (blk_id >= 0)) | (is_tail_blk & (tail_start >= 0))
        )
        safe_base = tl.maximum(base_tok, 0)
        logical_page = safe_base // PAGE_SIZE
        blk_live &= logical_page < PAGE_TABLE_WIDTH
        physical_page = tl.load(
            block_table_ptr
            + safe_request * stride_table_req
            + tl.minimum(logical_page, PAGE_TABLE_WIDTH - 1),
            mask=blk_live,
            other=-1,
        )
        blk_live &= (physical_page >= 0) & (physical_page < num_cache_blocks)
        safe_page_blk = tl.maximum(physical_page, 0).to(tl.int64)
        base_page_off = (safe_base % PAGE_SIZE).to(tl.int64)

        k_blk_base = (
            safe_page_blk * stride_k_block + base_page_off * stride_k_token + k_head_off
        )
        k_tok_off = tl.reshape(
            k_blk_base[:, None] + tok_sub[None, :].to(tl.int64) * stride_k_token,
            (BLOCK_N,),
        )
        cols_2d = blk_ranks[:, None] * COMPRESS_RATIO + tok_sub[None, :]
        tok_ok = (
            is_exp_blk[:, None]
            | (is_tail_blk[:, None] & (tok_sub[None, :] < tail_count))
        ) & (cols_2d < TOPK)
        valid = tl.reshape(blk_live[:, None] & tok_ok, (BLOCK_N,))

        if KV_DTYPE == 0:
            keys = tl.load(
                k_cache_ptr + k_tok_off[None, :] + dim_offsets[:, None],
                mask=valid[None, :],
                other=0.0,
            )
            v_blk_base = (
                safe_page_blk * stride_v_block
                + base_page_off * stride_v_token
                + v_head_off
            )
            v_tok_off = tl.reshape(
                v_blk_base[:, None] + tok_sub[None, :].to(tl.int64) * stride_v_token,
                (BLOCK_N,),
            )
            values = tl.load(
                v_cache_ptr + v_tok_off[:, None] + dim_offsets[None, :],
                mask=valid[:, None],
                other=0.0,
            )
        elif KV_DTYPE == 1:
            key_bytes = tl.load(
                k_cache_ptr + k_tok_off[None, :] + dim_offsets[:, None],
                mask=valid[None, :],
                other=0,
            )
            keys = _decode_e4m3_to_bf16(key_bytes)
        else:
            keys = tl.load(
                k_cache_ptr + k_tok_off[None, :] + dim_offsets[:, None],
                mask=valid[None, :],
                other=0.0,
            ).to(tl.bfloat16)

        scores = tl.dot(query, keys)
        if KV_DTYPE == 0:
            scores *= softmax_scale_log2
        else:
            scores *= softmax_scale_log2 * k_scale_value
        scores = tl.where(valid[None, :], scores, -1.0e20)
        next_max = tl.maximum(max_value, tl.max(scores, axis=1))
        alpha = tl.math.exp2(max_value - next_max)
        probabilities = tl.where(
            valid[None, :], tl.math.exp2(scores - next_max[:, None]), 0.0
        )

        if KV_DTYPE != 0:
            v_blk_base = (
                safe_page_blk * stride_v_block
                + base_page_off * stride_v_token
                + v_head_off
            )
            v_tok_off = tl.reshape(
                v_blk_base[:, None] + tok_sub[None, :].to(tl.int64) * stride_v_token,
                (BLOCK_N,),
            )
            if KV_DTYPE == 1:
                value_bytes = tl.load(
                    v_cache_ptr + v_tok_off[:, None] + dim_offsets[None, :],
                    mask=valid[:, None],
                    other=0,
                )
                values = _decode_e4m3_to_bf16(value_bytes)
            elif KV_DTYPE == 2:
                values = tl.load(
                    v_cache_ptr + v_tok_off[:, None] + dim_offsets[None, :],
                    mask=valid[:, None],
                    other=0.0,
                ).to(tl.bfloat16)
            else:
                values = tl.load(
                    v_cache_ptr + v_tok_off[:, None] + dim_offsets[None, :],
                    mask=valid[:, None],
                    other=0.0,
                ).to(tl.float16)

        accumulator = tl.dot(
            probabilities.to(values.dtype),
            values,
            acc=accumulator * alpha[:, None],
        )
        normalizer = normalizer * alpha + tl.sum(probabilities, axis=1)
        max_value = next_max

    has_values = normalizer > 0
    normalized_output = tl.where(
        has_values[:, None],
        accumulator / tl.maximum(normalizer[:, None], 1.0e-20),
        0.0,
    )
    if KV_DTYPE != 0:
        normalized_output *= v_scale_value
    output_mask = head_offsets[:, None] < GROUP_SIZE
    if NUM_SPLITS == 1:
        normalized_output = normalized_output.to(output_ptr.dtype.element_ty)
        output_gate = tl.load(
            output_gate_ptr
            + row * stride_output_gate_row
            + (first_head + head_offsets[:, None]) * stride_output_gate_head
            + dim_offsets[None, :],
            mask=output_mask,
            other=0.0,
        ).to(tl.float32)
        normalized_output = normalized_output.to(tl.float32) * tl.sigmoid(output_gate)
        tl.store(
            output_ptr
            + row * stride_output_row
            + (first_head + head_offsets[:, None]) * stride_output_head
            + dim_offsets[None, :],
            normalized_output,
            mask=output_mask,
        )
    else:
        partial_lse = tl.where(
            has_values,
            max_value + tl.math.log2(tl.maximum(normalizer, 1.0e-20)),
            -float("inf"),
        )
        partial_row = (split_id * num_rows + row).to(tl.int64)
        tl.store(
            partial_output_ptr
            + (partial_row * NUM_QUERY_HEADS + first_head + head_offsets[:, None])
            * HEAD_DIM
            + dim_offsets[None, :],
            normalized_output,
            mask=output_mask,
        )
        tl.store(
            partial_lse_ptr + partial_row * NUM_QUERY_HEADS + first_head + head_offsets,
            partial_lse,
            mask=head_offsets < GROUP_SIZE,
        )


@triton.jit(do_not_specialize=["num_rows"])
def _qsa_merge_fast_kernel(
    partial_output_ptr,
    partial_lse_ptr,
    output_ptr,
    output_gate_ptr,
    query_positions_ptr,
    visible_blocks_ptr,
    stride_output_row,
    stride_output_head,
    stride_output_gate_row,
    stride_output_gate_head,
    num_rows,
    BLOCK_TOPK: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_QUERY_HEADS: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    NUM_TILES: tl.constexpr,
    BLOCK_SPLITS: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    head = tl.program_id(1)
    query_position = tl.load(query_positions_ptr + row)
    visible_blocks = tl.load(visible_blocks_ptr + row)
    complete_blocks = tl.maximum(tl.minimum(visible_blocks, BLOCK_TOPK), 0)
    safe_qpos_p1 = tl.maximum(query_position + 1, 0)
    tail_start = (safe_qpos_p1 // COMPRESS_RATIO) * COMPRESS_RATIO
    tail_count = tl.where(query_position >= 0, safe_qpos_p1 - tail_start, 0)
    valid_count = complete_blocks * COMPRESS_RATIO + tail_count
    tile_end = tl.minimum(NUM_TILES, tl.cdiv(tl.minimum(valid_count, TOPK), BLOCK_N))
    active_splits = tl.minimum(NUM_SPLITS, tile_end)

    split_offsets = tl.arange(0, BLOCK_SPLITS)
    dim_offsets = tl.arange(0, HEAD_DIM)
    split_mask = split_offsets < active_splits
    lse = tl.load(
        partial_lse_ptr + (split_offsets * num_rows + row) * NUM_QUERY_HEADS + head,
        mask=split_mask,
        other=-float("inf"),
    )
    lse_max = tl.max(lse, axis=0)
    has_values = lse_max > -float("inf")
    shifted = tl.where(split_mask & has_values, lse - lse_max, -float("inf"))
    weights = tl.math.exp2(shifted)
    denominator = tl.sum(weights, axis=0)
    split_rows = split_offsets.to(tl.int64) * num_rows + row
    partial_output = tl.load(
        partial_output_ptr
        + (split_rows[:, None] * NUM_QUERY_HEADS + head) * HEAD_DIM
        + dim_offsets[None, :],
        mask=split_mask[:, None],
        other=0.0,
    )
    merged = tl.sum(partial_output * weights[:, None], axis=0)
    merged = tl.where(denominator > 0, merged / denominator, 0.0)
    merged = merged.to(output_ptr.dtype.element_ty)
    output_gate = tl.load(
        output_gate_ptr
        + row * stride_output_gate_row
        + head * stride_output_gate_head
        + dim_offsets
    ).to(tl.float32)
    merged = merged.to(tl.float32) * tl.sigmoid(output_gate)
    tl.store(
        output_ptr + row * stride_output_row + head * stride_output_head + dim_offsets,
        merged,
    )


def _direct_launch_params(
    num_rows: int,
    num_kv_heads: int,
    selection_width: int,
    kv_dtype: int,
) -> tuple[int, int, int, int, int, int]:
    """Return (block_n, partial_warps, num_tiles, num_splits, partial_stages, merge_warps).

    Preserves the exact (block_n, num_splits) of commit f332186 so output is
    bit-identical (0 ULP) while upgrading M=1..3 to 4 warps and setting
    num_stages=1 on sm_80/86 E4M3 K/V (avoiding Triton 3.8.0 register spills).
    """
    from vllm.models.qwen4_exp.nvidia.ops.qsa import _select_qsa_config_sm86

    block_n, partial_warps, num_tiles, num_splits = _select_qsa_config_sm86(
        num_rows,
        num_kv_heads,
        False,
        selection_width,
        is_fp8=kv_dtype != 0,
    )
    partial_warps = max(partial_warps, 4)
    partial_stages = 1 if kv_dtype == 1 else 2
    # Match stock merge num_warps=2 so tl.sum reduction order is 0-ULP bit-exact.
    merge_warps = 2
    return block_n, partial_warps, num_tiles, num_splits, partial_stages, merge_warps


def qsa_sparse_paged_attention_direct(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_indices: torch.Tensor,
    query_positions: torch.Tensor,
    visible_blocks: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    compress_ratio: int,
    token_topk: int,
    out: torch.Tensor | None = None,
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
    *,
    output_gate: torch.Tensor,
) -> torch.Tensor:
    """Direct-block sparse GQA over paged BF16/FP8 K/V caches (no expand launch)."""
    from vllm.models.qwen4_exp.nvidia.ops.qsa import qsa_e4m3_uses_native_cast

    if q.ndim != 3 or k_cache.ndim != 4 or v_cache.shape != k_cache.shape:
        raise ValueError("QSA direct sparse attention received invalid Q/K/V shapes")
    num_rows = q.shape[0]
    if out is None:
        out = torch.empty_like(q)
    if num_rows == 0:
        return out

    fp8_dtype = current_platform.fp8_dtype()
    if k_cache.dtype == torch.bfloat16:
        kv_dtype = 0
    elif k_cache.dtype in (torch.uint8, fp8_dtype):
        if qsa_e4m3_uses_native_cast():
            kv_dtype = 3
            k_cache = k_cache.view(torch.float8_e4m3fn)
            v_cache = v_cache.view(torch.float8_e4m3fn)
        else:
            kv_dtype = 1
            if k_cache.dtype == fp8_dtype:
                k_cache = k_cache.view(torch.uint8)
                v_cache = v_cache.view(torch.uint8)
    elif k_cache.dtype == torch.float8_e5m2:
        kv_dtype = 2
    else:
        raise ValueError("QSA supports only BF16, uint8/E4M3, and E5M2 K/V caches")

    assert q.dtype == torch.bfloat16
    assert block_indices.dtype == torch.int32 and block_indices.stride(1) == 1
    assert query_positions.dtype == torch.int32 and query_positions.stride(0) == 1
    assert visible_blocks.dtype == torch.int32 and visible_blocks.stride(0) == 1
    assert block_table.dtype == torch.int32 and block_table.stride(1) == 1
    assert token_to_req.dtype == torch.int32 and token_to_req.stride(0) == 1
    assert q.stride(2) == k_cache.stride(3) == v_cache.stride(3) == 1
    assert out.stride(2) == 1
    assert output_gate.stride(-1) == 1
    output_gate_view = (
        output_gate if output_gate.shape == q.shape else output_gate.view_as(q)
    )

    if kv_dtype == 0:
        k_scale_ptr = q
        v_scale_ptr = q
    else:
        if k_scale is None:
            k_scale = torch.tensor(1.0, dtype=torch.float32, device=q.device)
        if v_scale is None:
            v_scale = torch.tensor(1.0, dtype=torch.float32, device=q.device)
        k_scale_ptr = k_scale
        v_scale_ptr = v_scale

    num_kv_heads = k_cache.shape[2]
    group_size = q.shape[1] // num_kv_heads
    block_m = triton.next_power_of_2(group_size)
    block_topk = token_topk // compress_ratio
    selection_width = token_topk + compress_ratio - 1
    (
        block_n,
        partial_warps,
        num_tiles,
        num_splits,
        partial_stages,
        merge_warps,
    ) = _direct_launch_params(num_rows, num_kv_heads, selection_width, kv_dtype)

    if num_splits == 1:
        partial_output = out
        partial_lse = out
    else:
        partial_output = torch.empty(
            (num_splits, *q.shape), dtype=torch.float32, device=q.device
        )
        partial_lse = torch.empty(
            (num_splits, num_rows, q.shape[1]),
            dtype=torch.float32,
            device=q.device,
        )

    _qsa_direct_splitk_kernel[(num_rows, num_kv_heads, num_splits)](
        q,
        k_cache,
        v_cache,
        k_scale_ptr,
        v_scale_ptr,
        block_indices,
        query_positions,
        visible_blocks,
        block_table,
        token_to_req,
        partial_output,
        partial_lse,
        out,
        output_gate_view,
        q.stride(0),
        q.stride(1),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        block_indices.stride(0),
        block_table.stride(0),
        out.stride(0),
        out.stride(1),
        output_gate_view.stride(0),
        output_gate_view.stride(1),
        num_rows,
        k_cache.shape[0],
        block_table.shape[0],
        BLOCK_TOPK=block_topk,
        COMPRESS_RATIO=compress_ratio,
        TOPK=selection_width,
        PAGE_SIZE=k_cache.shape[1],
        PAGE_TABLE_WIDTH=block_table.shape[1],
        GROUP_SIZE=group_size,
        HEAD_DIM=q.shape[2],
        NUM_QUERY_HEADS=q.shape[1],
        NUM_SPLITS=num_splits,
        NUM_TILES=num_tiles,
        KV_DTYPE=kv_dtype,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=partial_warps,
        num_stages=partial_stages,
    )
    if num_splits == 1:
        return out

    _qsa_merge_fast_kernel[(num_rows, q.shape[1])](
        partial_output,
        partial_lse,
        out,
        output_gate_view,
        query_positions,
        visible_blocks,
        out.stride(0),
        out.stride(1),
        output_gate_view.stride(0),
        output_gate_view.stride(1),
        num_rows,
        BLOCK_TOPK=block_topk,
        COMPRESS_RATIO=compress_ratio,
        TOPK=selection_width,
        BLOCK_N=block_n,
        HEAD_DIM=q.shape[2],
        NUM_QUERY_HEADS=q.shape[1],
        NUM_SPLITS=num_splits,
        NUM_TILES=num_tiles,
        BLOCK_SPLITS=triton.next_power_of_2(num_splits),
        num_warps=merge_warps,
        num_stages=1,
    )
    return out


def warmup_qsa_direct_sparse_attention(
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    *,
    num_query_heads: int,
    selection_width: int,
    compress_ratio: int = 4,
) -> None:
    """Compile and preload every reachable sm_80/86 direct-block specialization."""
    from vllm.merlin.triton_preload import preload_triton_kernels
    from vllm.models.qwen4_exp.nvidia.ops.qsa import qsa_e4m3_uses_native_cast

    if not qsa_chain_fuse_active():
        return
    head_dim = kv_cache.shape[-1] // 2
    key_cache, value_cache = kv_cache.transpose(1, 2).split(head_dim, dim=-1)
    page_size = key_cache.shape[1]
    if page_size % compress_ratio != 0:
        return
    token_topk = selection_width - (compress_ratio - 1)
    if token_topk <= 0 or token_topk % compress_ratio != 0:
        return
    block_topk = token_topk // compress_ratio
    num_kv_heads = key_cache.shape[2]
    group_size = num_query_heads // num_kv_heads
    block_m = triton.next_power_of_2(group_size)

    if key_cache.dtype == torch.bfloat16:
        warm_kv_dtype = 0
        warm_cache_dtype = key_cache.dtype
    elif key_cache.dtype in (torch.uint8, current_platform.fp8_dtype()):
        if qsa_e4m3_uses_native_cast():
            warm_kv_dtype = 3
            warm_cache_dtype = torch.float8_e4m3fn
        else:
            warm_kv_dtype = 1
            warm_cache_dtype = torch.uint8
    elif key_cache.dtype == torch.float8_e5m2:
        warm_kv_dtype = 2
        warm_cache_dtype = key_cache.dtype
    else:
        return

    max_rows = max(1, QSA_DIRECT_MAX_BASE_PROGRAMS // num_kv_heads)
    partial_profiles = sorted(
        {
            _direct_launch_params(r, num_kv_heads, selection_width, warm_kv_dtype)[:5]
            for r in range(1, max_rows + 1)
        }
    )
    merge_profiles = sorted(
        {
            (p[0], p[2], p[3], p[5])
            for r in range(1, max_rows + 1)
            for p in (_direct_launch_params(r, num_kv_heads, selection_width, warm_kv_dtype),)
            if p[3] > 1
        }
    )

    num_rows = 16
    num_requests = 16
    q_ptr = TritonWarmupTensor(
        torch.bfloat16, shape=(num_rows, num_query_heads, head_dim)
    )
    k_scale_ptr = q_ptr if warm_kv_dtype == 0 else TritonWarmupTensor(torch.float32)
    v_scale_ptr = q_ptr if warm_kv_dtype == 0 else TritonWarmupTensor(torch.float32)
    k_cache_ptr = TritonWarmupTensor(
        warm_cache_dtype,
        shape=tuple(key_cache.shape),
        strides=tuple(key_cache.stride()),
    )
    v_cache_ptr = TritonWarmupTensor(
        warm_cache_dtype,
        shape=tuple(value_cache.shape),
        strides=tuple(value_cache.stride()),
    )
    blk_idx_ptr = TritonWarmupTensor(torch.int32, shape=(num_rows, block_topk))
    i32_1d = TritonWarmupTensor(torch.int32)
    block_table_ptr = TritonWarmupTensor(
        block_table.dtype,
        shape=tuple(block_table.shape),
        strides=tuple(block_table.stride()),
    )
    output_ptr = TritonWarmupTensor(
        torch.bfloat16, shape=(num_rows, num_query_heads, head_dim)
    )
    output_gate_ptr = TritonWarmupTensor(
        torch.bfloat16, shape=(num_rows, num_query_heads, head_dim)
    )
    head_stride = head_dim
    row_stride = num_query_heads * head_dim
    num_cache_blocks = triton_scalar_specialization_rep(kv_cache.shape[0])

    for block_n, partial_warps, num_tiles, num_splits, partial_stages in partial_profiles:
        if num_splits == 1:
            partial_output_ptr = output_ptr
            partial_lse_ptr = output_ptr
        else:
            partial_output_ptr = TritonWarmupTensor(
                torch.float32,
                shape=(num_splits, num_rows, num_query_heads, head_dim),
            )
            partial_lse_ptr = TritonWarmupTensor(
                torch.float32, shape=(num_splits, num_rows, num_query_heads)
            )
        _qsa_direct_splitk_kernel.warmup(
            q_ptr,
            k_cache_ptr,
            v_cache_ptr,
            k_scale_ptr,
            v_scale_ptr,
            blk_idx_ptr,
            i32_1d,
            i32_1d,
            block_table_ptr,
            i32_1d,
            partial_output_ptr,
            partial_lse_ptr,
            output_ptr,
            output_gate_ptr,
            row_stride,
            head_stride,
            key_cache.stride(0),
            key_cache.stride(1),
            key_cache.stride(2),
            value_cache.stride(0),
            value_cache.stride(1),
            value_cache.stride(2),
            block_topk,
            block_table.stride(0),
            row_stride,
            head_stride,
            row_stride,
            head_stride,
            num_rows,
            num_cache_blocks,
            num_requests,
            BLOCK_TOPK=block_topk,
            COMPRESS_RATIO=compress_ratio,
            TOPK=selection_width,
            PAGE_SIZE=page_size,
            PAGE_TABLE_WIDTH=block_table.shape[1],
            GROUP_SIZE=group_size,
            HEAD_DIM=head_dim,
            NUM_QUERY_HEADS=num_query_heads,
            NUM_SPLITS=num_splits,
            NUM_TILES=num_tiles,
            KV_DTYPE=warm_kv_dtype,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            num_warps=partial_warps,
            num_stages=partial_stages,
            grid=(num_rows, num_kv_heads, num_splits),
        )

    for block_n, num_tiles, num_splits, merge_warps in merge_profiles:
        partial_output_ptr = TritonWarmupTensor(
            torch.float32,
            shape=(num_splits, num_rows, num_query_heads, head_dim),
        )
        partial_lse_ptr = TritonWarmupTensor(
            torch.float32, shape=(num_splits, num_rows, num_query_heads)
        )
        _qsa_merge_fast_kernel.warmup(
            partial_output_ptr,
            partial_lse_ptr,
            output_ptr,
            output_gate_ptr,
            i32_1d,
            i32_1d,
            row_stride,
            head_stride,
            row_stride,
            head_stride,
            num_rows,
            BLOCK_TOPK=block_topk,
            COMPRESS_RATIO=compress_ratio,
            TOPK=selection_width,
            BLOCK_N=block_n,
            HEAD_DIM=head_dim,
            NUM_QUERY_HEADS=num_query_heads,
            NUM_SPLITS=num_splits,
            NUM_TILES=num_tiles,
            BLOCK_SPLITS=triton.next_power_of_2(num_splits),
            num_warps=merge_warps,
            num_stages=1,
            grid=(num_rows, num_query_heads),
        )

    preload_triton_kernels(
        _qsa_direct_splitk_kernel,
        _qsa_merge_fast_kernel,
        label="lebeche QSA direct sparse attention",
        expected=len(partial_profiles) + len(merge_profiles),
    )


@triton.jit(do_not_specialize=["block_topk"])
def _qsa_mqa_paged_uniform_bounded_kernel(
    q_ptr,
    k_cache_ptr,
    page_table_ptr,
    visible_blocks_ptr,
    logits_ptr,
    stride_q_row,
    stride_q_head,
    stride_cache_block,
    stride_cache_token,
    stride_table_req,
    stride_logits_row,
    num_cache_blocks,
    block_topk,
    guard_ctr,
    PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    DECODE_QUERY_LEN: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STAGES: tl.constexpr,
) -> None:
    NUM_COLUMNS: tl.constexpr = PAGE_TABLE_WIDTH * PAGE_SIZE
    DECODE_QUERY_LEN_PADDED: tl.constexpr = triton.next_power_of_2(DECODE_QUERY_LEN)
    NUM_HEADS_PADDED: tl.constexpr = triton.next_power_of_2(NUM_HEADS)
    BLOCK_D: tl.constexpr = max(16, triton.next_power_of_2(HEAD_DIM))
    request = tl.program_id(0)
    tile_pid = tl.program_id(1)
    num_tile_progs = tl.num_programs(1)
    query_offsets = tl.arange(0, DECODE_QUERY_LEN_PADDED)
    valid_query_offsets = query_offsets < DECODE_QUERY_LEN
    rows = request * DECODE_QUERY_LEN + query_offsets
    visible = tl.load(
        visible_blocks_ptr + rows,
        mask=valid_query_offsets,
        other=0,
    )
    max_visible = tl.max(visible, axis=0)
    # When max_visible <= block_topk, persistent_topk takes its trivial branch
    # and emits [0, visible) without reading logits.
    if max_visible <= block_topk:
        return
    total_tiles = tl.minimum(
        tl.cdiv(max_visible, BLOCK_N), tl.cdiv(NUM_COLUMNS, BLOCK_N)
    )
    if tile_pid >= total_tiles:
        return

    dims = tl.arange(0, BLOCK_D)
    n = tl.arange(0, DECODE_QUERY_LEN_PADDED * NUM_HEADS_PADDED)
    query_offset = n // NUM_HEADS_PADDED
    head = n % NUM_HEADS_PADDED
    valid_query = (query_offset < DECODE_QUERY_LEN) & (head < NUM_HEADS)
    query = tl.load(
        q_ptr
        + (request * DECODE_QUERY_LEN + query_offset)[None, :] * stride_q_row
        + head[None, :] * stride_q_head
        + dims[:, None],
        mask=valid_query[None, :] & (dims[:, None] < HEAD_DIM),
        other=0.0,
    )
    column_offsets = tl.arange(0, BLOCK_N)
    guard_bad_n = tl.zeros([BLOCK_N], dtype=tl.int32)
    guard_bad_col = tl.full([BLOCK_N], -1, tl.int32)
    for tile in tl.range(tile_pid, total_tiles, num_tile_progs, num_stages=STAGES):
        columns = tile * BLOCK_N + column_offsets
        live = columns < max_visible
        logical_page = tl.minimum(columns // PAGE_SIZE, PAGE_TABLE_WIDTH - 1)
        page_offset = columns % PAGE_SIZE
        physical_page = tl.load(
            page_table_ptr + request * stride_table_req + logical_page,
            mask=live,
            other=-1,
        )
        page_bad = live & ((physical_page < 0) | (physical_page >= num_cache_blocks))
        guard_bad_n += page_bad.to(tl.int32)
        guard_bad_col = tl.maximum(
            guard_bad_col, tl.where(page_bad, logical_page, -1).to(tl.int32)
        )
        live = live & (physical_page >= 0) & (physical_page < num_cache_blocks)
        safe_page = tl.maximum(physical_page, 0).to(tl.int64)
        keys = tl.load(
            k_cache_ptr
            + safe_page[:, None] * stride_cache_block
            + page_offset[:, None] * stride_cache_token
            + dims[None, :],
            mask=live[:, None] & (dims[None, :] < HEAD_DIM),
            other=0.0,
            eviction_policy="evict_first",
        )
        scores = tl.dot(keys, query, out_dtype=tl.float32)
        scores = tl.where(valid_query[None, :], tl.maximum(scores, 0.0), 0.0)
        scores = tl.reshape(
            scores,
            (BLOCK_N, DECODE_QUERY_LEN_PADDED, NUM_HEADS_PADDED),
        )
        score = tl.sum(scores, axis=2)
        tl.store(
            logits_ptr + rows[None, :] * stride_logits_row + columns[:, None],
            score,
            mask=valid_query_offsets[None, :]
            & (columns[:, None] < NUM_COLUMNS)
            & (columns[:, None] < visible[None, :]),
        )

    guard_n = tl.sum(guard_bad_n, axis=0)
    if guard_n > 0:
        tl.atomic_add(guard_ctr, guard_n.to(tl.int64))
        tl.atomic_max(guard_ctr + 1, (request).to(tl.int64) + _GUARD_BIAS)
        tl.atomic_max(
            guard_ctr + 2, (tl.max(guard_bad_col, axis=0)).to(tl.int64) + _GUARD_BIAS
        )


def qsa_mqa_paged_decode_bounded(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    visible_blocks: torch.Tensor,
    logits: torch.Tensor,
    decode_query_len: int,
    block_topk: int,
) -> None:
    num_requests = q.shape[0] // decode_query_len
    columns = page_table.shape[1] * k_cache.shape[1]
    max_tiles = triton.cdiv(columns, _MQA_DECODE_BLOCK_N)
    num_tile_progs = max(1, min(max_tiles, QSA_MQA_BOUNDED_TOTAL_CTAS // num_requests))
    _qsa_mqa_paged_uniform_bounded_kernel[(num_requests, num_tile_progs)](
        q,
        k_cache,
        page_table,
        visible_blocks,
        logits,
        *q.stride()[:-1],
        *k_cache.stride()[:2],
        *page_table.stride()[:-1],
        *logits.stride()[:-1],
        k_cache.shape[0],
        block_topk,
        guard_telemetry.counter(guard_telemetry.SITE_QSA_DECODE, q.device),
        PAGE_SIZE=k_cache.shape[1],
        PAGE_TABLE_WIDTH=page_table.shape[1],
        NUM_HEADS=q.shape[1],
        HEAD_DIM=q.shape[2],
        DECODE_QUERY_LEN=decode_query_len,
        BLOCK_N=_MQA_DECODE_BLOCK_N,
        STAGES=2,
        num_warps=1 if k_cache.dtype == torch.float8_e4m3fn else 2,
    )


def warmup_qsa_mqa_bounded_decode(
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    *,
    num_heads: int,
    head_dim: int,
    max_decode_query_len: int,
    max_num_reqs: int,
) -> None:
    """Compile and preload the bounded decode MQA scorer (one per decode_query_len)."""
    from vllm.merlin.triton_preload import preload_triton_kernels

    if not qsa_chain_fuse_active() or max_decode_query_len <= 0:
        return
    page_size = k_cache.shape[1]
    page_table_width = page_table.shape[1]
    columns = page_table_width * page_size
    k_cache_ptr = TritonWarmupTensor(k_cache.dtype, shape=tuple(k_cache.shape))
    page_table_ptr = TritonWarmupTensor(
        page_table.dtype, shape=(max_num_reqs, page_table_width)
    )
    visible_blocks_ptr = TritonWarmupTensor(torch.int32)
    guard_ptr = TritonWarmupTensor(torch.int64)
    num_tile_progs = max(1, min(triton.cdiv(columns, _MQA_DECODE_BLOCK_N), QSA_MQA_BOUNDED_TOTAL_CTAS))

    for dql in range(1, max_decode_query_len + 1):
        q_ptr = TritonWarmupTensor(k_cache.dtype, shape=(dql, num_heads, head_dim))
        logits_ptr = TritonWarmupTensor(torch.float32, shape=(dql, columns))
        _qsa_mqa_paged_uniform_bounded_kernel.warmup(
            q_ptr,
            k_cache_ptr,
            page_table_ptr,
            visible_blocks_ptr,
            logits_ptr,
            num_heads * head_dim,
            head_dim,
            k_cache.stride(0),
            k_cache.stride(1),
            page_table.stride(0),
            columns,
            k_cache.shape[0],
            512,
            guard_ptr,
            PAGE_SIZE=page_size,
            PAGE_TABLE_WIDTH=page_table_width,
            NUM_HEADS=num_heads,
            HEAD_DIM=head_dim,
            DECODE_QUERY_LEN=dql,
            BLOCK_N=_MQA_DECODE_BLOCK_N,
            STAGES=2,
            num_warps=1 if k_cache.dtype == torch.float8_e4m3fn else 2,
            grid=(1, num_tile_progs),
        )
    preload_triton_kernels(
        _qsa_mqa_paged_uniform_bounded_kernel,
        label="lebeche QSA bounded indexer decode",
        expected=max_decode_query_len,
    )
