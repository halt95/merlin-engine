# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modified by the Merlin project; see NOTICE.
"""Torch adaptation of #68296 indexpool_prepare.py and indexpool_ops.py."""

import torch


def prepare_groups(keys, positions, tail, table, starts, lengths, tokens, ratio):
    """Return fixed-capacity groups and logical query metadata.

    Read carry before write_tails replaces speculative rows. Tensor values
    never determine Python control flow or output allocation sizes.
    """
    batch = lengths.numel()
    if batch == 0:
        raise ValueError("IndexPool preparation requires request metadata")
    if keys.shape[0] == 0 or tail.shape[0] == 0 or table.shape[1] == 0:
        raise ValueError("IndexPool preparation requires bound storage")

    device = keys.device
    starts = starts.long()
    lengths = lengths.long()
    query_lengths = starts[1:] - starts[:-1]
    contexts = lengths - query_lengths
    request_live = (query_lengths > 0) & (contexts >= 0)
    counts = torch.where(
        request_live,
        lengths // ratio - contexts.clamp_min(0) // ratio,
        0,
    )
    ends = counts.cumsum(0)
    slots = (tokens + batch * (ratio - 1)) // ratio
    slot = torch.arange(slots, device=device)
    requests = torch.searchsorted(ends.contiguous(), slot, right=True)
    group_live = requests < batch
    requests = requests.clamp_max(batch - 1)
    previous = torch.cat((ends.new_zeros(1), ends[:-1]))
    groups = contexts[requests].clamp_min(0) // ratio
    groups = groups + slot - previous[requests]
    first = groups * ratio
    members = first[:, None] + torch.arange(ratio, device=device)
    old = members < contexts[requests, None]
    source = starts[requests, None] + members - contexts[requests, None]
    pages = table[requests, 0].long()
    group_live = group_live & (pages >= 0) & (pages < tail.shape[0])
    safe_pages = pages.clamp(0, tail.shape[0] - 1)
    retained = tail[safe_pages[:, None], members.remainder(tail.shape[1]), :128].to(
        keys.dtype
    )
    current = keys[source.clamp(0, keys.shape[0] - 1)]
    raw = torch.where(old[..., None], retained, current)
    raw = raw.masked_fill(~group_live[:, None, None], 0)

    source_first = starts[requests] + first - contexts[requests]
    source_first = source_first.clamp(0, keys.shape[0] - 1)
    retained_xyz = tail[safe_pages, first.remainder(tail.shape[1]), 128:131].long().T
    if positions.ndim == 1:
        current_xyz = positions[source_first][None].expand(3, -1)
    else:
        current_xyz = positions[:, source_first]
    coordinates = torch.where(
        (first < contexts[requests])[None], retained_xyz, current_xyz
    )
    coordinates = coordinates.masked_fill(~group_live[None], 0)

    rows = torch.arange(tokens, device=device)
    owner = torch.searchsorted(starts[1:].contiguous(), rows, right=True)
    token_live = owner < batch
    owner = owner.clamp_max(batch - 1)
    token_live = token_live & request_live[owner] & (rows < starts[-1])
    logical = contexts[owner].clamp_min(0) + rows - starts[owner]
    upper = torch.where(token_live, logical // ratio, 0).int()
    group_positions = coordinates if positions.ndim == 2 else coordinates[0]
    return (
        raw,
        group_positions,
        requests.int(),
        groups,
        group_live,
        owner,
        logical,
        upper,
        token_live,
    )


def rotate_indexer(x, positions, rope_dim, parameters):
    """Adapted from #68296 indexpool_ops.py:39-74."""
    if rope_dim == 0:
        return x
    if parameters.get("rope_type", "default") != "default":
        raise ValueError("Flash IndexPool requires default RoPE")
    theta = parameters.get("rope_theta", 10000.0)
    inverse = theta ** (
        -torch.arange(0, rope_dim, 2, device=x.device).float() / rope_dim
    )
    if positions.ndim == 1:
        frequency = positions.float()[:, None] * inverse
    else:
        section = parameters.get("mrope_section")
        if positions.shape[0] != 3 or section is None:
            raise ValueError("IndexPool requires three-axis MRoPE sections")
        if sum(section) * 2 != rope_dim:
            raise ValueError("Invalid IndexPool MRoPE section widths")
        planes = positions.float()[:, :, None] * inverse
        if parameters.get("mrope_interleaved", True):
            from vllm.model_executor.layers.rotary_embedding.mrope import (
                apply_interleaved_rope,
            )

            frequency = apply_interleaved_rope(planes, section)
        else:
            parts, offset = [], 0
            for axis, width in enumerate(section):
                parts.append(planes[axis, :, offset : offset + width])
                offset += width
            frequency = torch.cat(parts, -1)

    nope, pe = x.split((x.shape[-1] - rope_dim, rope_dim), -1)
    pe = pe.reshape(*pe.shape[:-1], rope_dim // 2, 2)
    pe = pe.transpose(-2, -1).flatten(-2)
    frequency = torch.cat((frequency, frequency), -1)
    if pe.ndim == 3:
        frequency = frequency[:, None]
    left, right = pe.float().chunk(2, -1)
    rotated = (
        pe.float() * frequency.cos() + torch.cat((-right, left), -1) * frequency.sin()
    )
    return torch.cat((nope, rotated.to(x.dtype)), -1)


def logical_window_indices(starts, lengths, rows, actual, window, width):
    if window < 1 or width < window:
        raise ValueError("SWA index width must cover the causal window")
    if lengths.numel() == 0:
        return starts.new_full((rows, width), -1, dtype=torch.int32)
    tokens = torch.arange(rows, device=starts.device)
    request = torch.searchsorted(starts[1:].contiguous(), tokens, right=True)
    request = request.clamp_max(lengths.numel() - 1)
    query_lengths = starts[request + 1] - starts[request]
    context = lengths[request] - query_lengths
    logical = context + tokens - starts[request]
    live = (
        (tokens < actual) & (tokens < starts[-1]) & (query_lengths > 0) & (context >= 0)
    )
    columns = torch.arange(width, device=starts.device)
    indices = logical[:, None] - window + 1 + columns
    valid = live[:, None] & (columns < window) & (indices >= 0)
    return torch.where(valid, indices, -1).int()
