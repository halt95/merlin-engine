# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modified by the Merlin project; see NOTICE.
"""DFly2 draft input preparation.

Adapted from #68296 ``vllm/models/hy_v4/nvidia/dfly_inputs.py:1-351``. Two
launches per draft step: one per request for the query block and sampling
state, and one for the sparse indices of every draft layer.

Differences from upstream:
- No query-slot kernel (#68296 :121-158, :295-317). The speculator computes the
  query block's slots with ``BlockTables.compute_slot_mappings``, the target's
  own slot-mapping kernel, which is DCP-aware: under decode context
  parallelism only the owner rank of a position gets a real slot, on its
  rank-local block. Upstream's kernel
  assumes DCP 1.
- No M-RoPE buffers (text lane; the Flash attention takes 1-D positions).
- ``last_sampled`` is the 1-D ``[max_num_reqs]`` buffer of this tree's runner.
- ``max_model_len`` is a host int.
- Each launcher has a torch reference (``*_torch``) with the same contract; it
  runs on CPU tensors and is what the CPU tests and the GPU parity test use.
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit(do_not_specialize=["num_reqs", "padded_batch", "max_model_len"])
def _prepare_dfly_inputs_kernel(
    # Outputs
    input_ids_ptr,
    positions_ptr,
    is_padding_ptr,
    query_start_loc_ptr,
    seq_lens_ptr,
    active_ptr,
    source_rows_ptr,
    anchors_ptr,
    hidden_out_ptr,
    hidden_out_stride,
    draft_idx_mapping_ptr,
    out_temperature_ptr,
    out_seeds_ptr,
    # Inputs
    target_query_start_loc_ptr,
    target_positions_ptr,
    idx_mapping_ptr,
    num_sampled_ptr,
    num_rejected_ptr,
    last_sampled_ptr,
    last_sampled_stride,
    hidden_in_ptr,
    hidden_in_stride,
    temperature_ptr,
    seeds_ptr,
    max_model_len,
    mask_token_id,
    hidden_size,
    num_reqs,
    padded_batch,
    QUERY_WIDTH: tl.constexpr,
    QUERY_BLOCK: tl.constexpr,
    HIDDEN_BLOCK: tl.constexpr,
):
    req_idx = tl.program_id(0)
    is_real = req_idx < num_reqs

    # Rows past num_reqs are padding: every load below is masked off for them.
    req_state_idx = tl.load(idx_mapping_ptr + req_idx, mask=is_real, other=-1).to(
        tl.int64
    )
    query_end = tl.load(target_query_start_loc_ptr + req_idx + 1, mask=is_real, other=0)
    num_rejected = tl.load(num_rejected_ptr + req_idx, mask=is_real, other=0)
    source_row = tl.where(is_real, query_end - num_rejected - 1, 0).to(tl.int64)
    anchor = tl.load(target_positions_ptr + source_row, mask=is_real, other=-1) + 1
    num_sampled = tl.load(num_sampled_ptr + req_idx, mask=is_real, other=0)
    active = is_real & (num_sampled > 0) & (anchor + QUERY_WIDTH <= max_model_len)
    bonus = tl.load(
        last_sampled_ptr + req_state_idx * last_sampled_stride, mask=active, other=0
    ).to(tl.int32)
    tl.store(
        out_temperature_ptr + req_state_idx,
        tl.load(temperature_ptr + req_state_idx, mask=is_real),
        mask=is_real,
    )
    tl.store(
        out_seeds_ptr + req_state_idx,
        tl.load(seeds_ptr + req_state_idx, mask=is_real),
        mask=is_real,
    )

    tl.store(active_ptr + req_idx, active)
    tl.store(source_rows_ptr + req_idx, source_row)
    tl.store(anchors_ptr + req_idx, anchor)
    tl.store(draft_idx_mapping_ptr + req_idx, tl.where(active, req_state_idx, -1))

    offsets = tl.arange(0, QUERY_BLOCK)
    in_block = offsets < QUERY_WIDTH
    token_idx = req_idx * QUERY_WIDTH + offsets
    positions = tl.where(active, anchor + offsets, offsets)
    input_ids = tl.where(offsets == 0, bonus, mask_token_id)
    tl.store(input_ids_ptr + token_idx, input_ids, mask=in_block)
    tl.store(positions_ptr + token_idx, positions, mask=in_block)
    tl.store(is_padding_ptr + token_idx, active == 0, mask=in_block)

    tl.store(seq_lens_ptr + req_idx, tl.where(active, anchor, 0) + QUERY_WIDTH)
    tl.store(query_start_loc_ptr + req_idx, req_idx * QUERY_WIDTH)
    if req_idx == padded_batch - 1:
        tl.store(query_start_loc_ptr + padded_batch, padded_batch * QUERY_WIDTH)

    out_row = hidden_out_ptr + req_idx * hidden_out_stride
    in_row = hidden_in_ptr + source_row * hidden_in_stride
    for i in range(0, hidden_size, HIDDEN_BLOCK):
        block = i + tl.arange(0, HIDDEN_BLOCK)
        mask = block < hidden_size
        hidden = tl.load(in_row + block, mask=mask & is_real, other=0.0)
        tl.store(out_row + block, hidden, mask=mask)


@triton.jit
def _build_dfly_reuse_indices_kernel(
    out_ptr,
    out_layer_stride,
    out_row_stride,
    snapshot_ptr,
    snapshot_slot_stride,
    snapshot_row_stride,
    layer_snapshot_slots_ptr,
    active_ptr,
    source_rows_ptr,
    anchors_ptr,
    TARGET_TOPK: tl.constexpr,
    LOCAL_WINDOW: tl.constexpr,
    QUERY_WIDTH: tl.constexpr,
    OUT_WIDTH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Per query row: target top-k history, local window, then causal block."""
    req_idx = tl.program_id(0)
    query_offset = tl.program_id(1)
    layer = tl.program_id(2)

    active = tl.load(active_ptr + req_idx)
    anchor = tl.load(anchors_ptr + req_idx)
    source_row = tl.load(source_rows_ptr + req_idx).to(tl.int64)
    slot = tl.load(layer_snapshot_slots_ptr + layer).to(tl.int64)
    history_ptr = (
        snapshot_ptr + slot * snapshot_slot_stride + source_row * snapshot_row_stride
    )
    out_row = (
        out_ptr
        + layer * out_layer_stride
        + (req_idx * QUERY_WIDTH + query_offset).to(tl.int64) * out_row_stride
    )
    window_start = anchor - LOCAL_WINDOW
    for i in range(0, OUT_WIDTH, BLOCK_SIZE):
        cols = i + tl.arange(0, BLOCK_SIZE)
        is_history = cols < TARGET_TOPK
        history = tl.load(history_ptr + cols, mask=is_history & active, other=-1)
        history = tl.where((history >= 0) & (history < window_start), history, -1)
        window_col = cols - TARGET_TOPK
        is_window = (window_col >= 0) & (window_col < LOCAL_WINDOW)
        window = tl.maximum(window_start + window_col, -1)
        block_col = window_col - LOCAL_WINDOW
        is_causal = (block_col >= 0) & (block_col <= query_offset)
        value = tl.where(
            is_history,
            history,
            tl.where(is_window, window, tl.where(is_causal, anchor + block_col, -1)),
        )
        value = tl.where(active, value, -1)
        tl.store(out_row + cols, value.to(tl.int32), mask=cols < OUT_WIDTH)


def dfly_topk_width(target_topk: int, local_window: int, query_width: int) -> int:
    """History + window + causal block, 128-aligned (#68296 dfly.py:215)."""
    return ((target_topk + local_window + query_width + 127) // 128) * 128


def prepare_dfly_inputs(
    *,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    is_padding: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    active: torch.Tensor,
    source_rows: torch.Tensor,
    anchors: torch.Tensor,
    target_hidden_out: torch.Tensor,
    draft_idx_mapping: torch.Tensor,
    out_temperature: torch.Tensor,
    out_seeds: torch.Tensor,
    target_query_start_loc: torch.Tensor,
    target_positions: torch.Tensor,
    idx_mapping: torch.Tensor,
    num_sampled: torch.Tensor,
    num_rejected: torch.Tensor,
    last_sampled: torch.Tensor,
    target_hidden: torch.Tensor,
    temperature: torch.Tensor,
    seeds: torch.Tensor,
    max_model_len: int,
    mask_token_id: int,
    query_width: int,
    num_reqs: int,
    padded_batch: int,
) -> None:
    """Fill the query block of ``padded_batch`` requests in place.

    Rows past ``num_reqs`` are inactive padding, so ``num_reqs=0`` gives an
    all-padding block for dummy runs. A request is active when it sampled a
    token and its whole block fits in ``max_model_len``; an inactive row has
    ``draft_idx_mapping = -1`` (no slots, no logits) and drafts -1.
    """
    if num_reqs > 0:
        assert target_hidden.shape[-1] == target_hidden_out.shape[-1], (
            "DFly fusion requires the target's final hidden states"
        )
    if padded_batch == 0:
        return
    kwargs = dict(
        input_ids=input_ids,
        positions=positions,
        is_padding=is_padding,
        query_start_loc=query_start_loc,
        seq_lens=seq_lens,
        active=active,
        source_rows=source_rows,
        anchors=anchors,
        target_hidden_out=target_hidden_out,
        draft_idx_mapping=draft_idx_mapping,
        out_temperature=out_temperature,
        out_seeds=out_seeds,
        target_query_start_loc=target_query_start_loc,
        target_positions=target_positions,
        idx_mapping=idx_mapping,
        num_sampled=num_sampled,
        num_rejected=num_rejected,
        last_sampled=last_sampled,
        target_hidden=target_hidden,
        temperature=temperature,
        seeds=seeds,
        max_model_len=max_model_len,
        mask_token_id=mask_token_id,
        query_width=query_width,
        num_reqs=num_reqs,
        padded_batch=padded_batch,
    )
    if not input_ids.is_cuda:
        prepare_dfly_inputs_torch(**kwargs)
        return
    _prepare_dfly_inputs_kernel[(padded_batch,)](
        input_ids,
        positions,
        is_padding,
        query_start_loc,
        seq_lens,
        active,
        source_rows,
        anchors,
        target_hidden_out,
        target_hidden_out.stride(0),
        draft_idx_mapping,
        out_temperature,
        out_seeds,
        target_query_start_loc,
        target_positions,
        idx_mapping,
        num_sampled,
        num_rejected,
        last_sampled,
        last_sampled.stride(0),
        target_hidden,
        target_hidden.stride(0),
        temperature,
        seeds,
        max_model_len,
        mask_token_id,
        target_hidden_out.shape[-1],
        num_reqs,
        padded_batch,
        QUERY_WIDTH=query_width,
        QUERY_BLOCK=triton.next_power_of_2(query_width),
        HIDDEN_BLOCK=1024,
    )


def prepare_dfly_inputs_torch(
    *,
    input_ids,
    positions,
    is_padding,
    query_start_loc,
    seq_lens,
    active,
    source_rows,
    anchors,
    target_hidden_out,
    draft_idx_mapping,
    out_temperature,
    out_seeds,
    target_query_start_loc,
    target_positions,
    idx_mapping,
    num_sampled,
    num_rejected,
    last_sampled,
    target_hidden,
    temperature,
    seeds,
    max_model_len,
    mask_token_id,
    query_width,
    num_reqs,
    padded_batch,
) -> None:
    """Torch reference of ``_prepare_dfly_inputs_kernel`` (same buffers and writes)."""
    device = input_ids.device
    real = torch.arange(padded_batch, device=device) < num_reqs
    n = num_reqs
    source = torch.zeros(padded_batch, dtype=torch.int64, device=device)
    anchor = torch.zeros(padded_batch, dtype=anchors.dtype, device=device)
    state = torch.full((padded_batch,), -1, dtype=torch.int64, device=device)
    sampled = torch.zeros(padded_batch, dtype=torch.int64, device=device)
    if n:
        source[:n] = (
            target_query_start_loc[1 : n + 1].long() - num_rejected[:n].long() - 1
        )
        anchor[:n] = target_positions[source[:n]].to(anchor.dtype) + 1
        state[:n] = idx_mapping[:n].long()
        sampled[:n] = num_sampled[:n].long()
        out_temperature[state[:n]] = temperature[state[:n]]
        out_seeds[state[:n]] = seeds[state[:n]]
    anchor = torch.where(real, anchor, torch.zeros_like(anchor))
    is_active = real & (sampled > 0) & (anchor + query_width <= max_model_len)
    bonus = torch.zeros(padded_batch, dtype=torch.int64, device=device)
    if n:
        flat = last_sampled.reshape(last_sampled.shape[0], -1)[:, 0]
        bonus[:n] = torch.where(is_active[:n], flat[state[:n].clamp_min(0)].long(), 0)
    active[:padded_batch] = is_active
    source_rows[:padded_batch] = source
    anchors[:padded_batch] = anchor
    draft_idx_mapping[:padded_batch] = torch.where(is_active, state, -1).to(
        draft_idx_mapping.dtype
    )
    offsets = torch.arange(query_width, device=device)
    pos = torch.where(is_active[:, None], anchor[:, None] + offsets, offsets)
    ids = torch.full(
        (padded_batch, query_width), mask_token_id, dtype=torch.int64, device=device
    )
    ids[:, 0] = bonus
    count = padded_batch * query_width
    input_ids[:count] = ids.flatten().to(input_ids.dtype)
    positions[:count] = pos.flatten().to(positions.dtype)
    is_padding[:count] = (~is_active)[:, None].expand(-1, query_width).flatten()
    seq_lens[:padded_batch] = (torch.where(is_active, anchor, 0) + query_width).to(
        seq_lens.dtype
    )
    query_start_loc[: padded_batch + 1] = (
        torch.arange(padded_batch + 1, device=device) * query_width
    ).to(query_start_loc.dtype)
    hidden = target_hidden_out.new_zeros((padded_batch, target_hidden_out.shape[-1]))
    if n:
        hidden[:n] = target_hidden[source[:n]].to(hidden.dtype)
    target_hidden_out[:padded_batch] = hidden


def build_dfly_reuse_indices(
    out: torch.Tensor,
    snapshots: torch.Tensor,
    layer_snapshot_slots: torch.Tensor,
    active: torch.Tensor,
    source_rows: torch.Tensor,
    anchors: torch.Tensor,
    target_topk: int,
    local_window: int,
    query_width: int,
    padded_batch: int,
) -> None:
    """Fill ``out[layer, :padded_batch * query_width]`` for every draft layer.

    Each row is ``[history (target top-k of the source row, before the
    window) | local window | causal query block]``, ``-1`` padded. The values
    are global per-request token positions; under DCP every rank holds the
    same rows and the draft impl filters them to its local slots, as for the
    target.
    """
    num_layers, _, out_width = out.shape
    if padded_batch == 0 or num_layers == 0:
        return
    if not out.is_cuda:
        build_dfly_reuse_indices_torch(
            out,
            snapshots,
            layer_snapshot_slots,
            active,
            source_rows,
            anchors,
            target_topk,
            local_window,
            query_width,
            padded_batch,
        )
        return
    assert out.stride(-1) == 1 and snapshots.stride(-1) == 1
    _build_dfly_reuse_indices_kernel[(padded_batch, query_width, num_layers)](
        out,
        out.stride(0),
        out.stride(1),
        snapshots,
        snapshots.stride(0),
        snapshots.stride(1),
        layer_snapshot_slots,
        active,
        source_rows,
        anchors,
        TARGET_TOPK=target_topk,
        LOCAL_WINDOW=local_window,
        QUERY_WIDTH=query_width,
        OUT_WIDTH=out_width,
        BLOCK_SIZE=1024,
    )


def build_dfly_reuse_indices_torch(
    out,
    snapshots,
    layer_snapshot_slots,
    active,
    source_rows,
    anchors,
    target_topk,
    local_window,
    query_width,
    padded_batch,
) -> None:
    """Torch reference of ``_build_dfly_reuse_indices_kernel``."""
    num_layers, _, width = out.shape
    batch = padded_batch
    device = out.device
    act = active[:batch].bool()
    anchor = anchors[:batch].long()
    history = snapshots[
        layer_snapshot_slots.long()[:, None], source_rows[:batch].long()[None, :]
    ][..., :target_topk].long()
    window_start = anchor - local_window
    history = torch.where(
        (history >= 0) & (history < window_start[None, :, None]), history, -1
    )
    rows = torch.full(
        (num_layers, batch, query_width, width), -1, dtype=torch.long, device=device
    )
    rows[..., :target_topk] = history[:, :, None, :]
    if local_window:
        offsets = torch.arange(local_window, device=device)
        window = (window_start[:, None] + offsets).clamp_min(-1)
        rows[..., target_topk : target_topk + local_window] = window[None, :, None, :]
    q = torch.arange(query_width, device=device)
    causal = anchor[:, None, None] + q[None, None, :]
    causal = causal.expand(batch, query_width, query_width).clone()
    causal.masked_fill_(q[None, :] > q[:, None], -1)
    end = min(target_topk + local_window + query_width, width)
    rows[..., target_topk + local_window : end] = causal[
        None, ..., : end - target_topk - local_window
    ]
    rows.masked_fill_(~act[None, :, None, None], -1)
    out[:, : batch * query_width] = rows.reshape(
        num_layers, batch * query_width, width
    ).to(out.dtype)
