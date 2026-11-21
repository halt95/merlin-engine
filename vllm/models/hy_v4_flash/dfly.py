# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modified by the Merlin project; see NOTICE.
"""DFly policy and torch references; this is not a registered speculator.

Adapted from #68296 hy_v4/dfly_config.py:96-179,
hy_v4_flash/dfly_config.py:1-63, and
hy_v4/nvidia/dfly_inputs.py:58-213.
"""

from copy import deepcopy
from dataclasses import dataclass

import torch

from vllm.models.hy_v4.checkpoint import draft_quantization_config
from vllm.models.hy_v4_flash.configs import SPARSE_ATTENTION
from vllm.transformers_utils.configs.hy_v4 import HYV4Config


def derive_dfly_config(target_config):
    target = target_config.get_text_config()
    if target.model_type != "hy_v4_flash":
        raise ValueError("DFly requires a hy_v4_flash target")
    count = getattr(target_config, "dfly_num_hidden_layers", None)
    if count is None:
        count = getattr(target, "dfly_num_hidden_layers", None)
    if type(count) is not int or count < 1:
        raise ValueError("DFly requires positive dfly_num_hidden_layers")

    def option(name, default=None):
        return deepcopy(
            getattr(
                target_config,
                f"dfly_{name}",
                getattr(target, f"dfly_{name}", default),
            )
        )

    kv_layers = option("target_kv_layers")
    index_layers = option("target_index_layer")
    if type(index_layers) is int:
        index_layers = [index_layers] * count
    ropes = []
    for name, layers in (
        ("target_kv_layers", kv_layers),
        ("target_index_layer", index_layers),
    ):
        if not isinstance(layers, list) or len(layers) != count:
            raise ValueError(f"DFly {name} requires one target layer per draft layer")
        for index in layers:
            if type(index) is not int or not 1 <= index < target.num_hidden_layers:
                raise ValueError(f"Invalid DFly target layer: {index}")
            if target.layer_types[index] != SPARSE_ATTENTION:
                raise ValueError(f"DFly target layer {index} must be sparse")
            ropes.append(target.rope_for_layer(index))
    if len(set(kv_layers)) != count:
        raise ValueError("DFly requires distinct target KV layers")
    if any(rope != ropes[0] for rope in ropes[1:]):
        raise ValueError("DFly selected target layers must use identical RoPE")
    if (
        option("enable_ihc", False) is not False
        or option("mlp_type", "dense") != "dense"
        or option("attn_causal", True) is not True
        or option("mtp_fusion", "slot0") != "slot0"
        or option("enable_hidden_correction", True) is not True
    ):
        raise ValueError(
            "DFly requires causal dense layers, slot0 fusion, and correction"
        )
    mask = option("mask_token_id")
    if type(mask) is not int or not 0 <= mask < target.vocab_size:
        raise ValueError("DFly mask token is outside the vocabulary")
    width = option("intermediate_size")
    if width is not None and (type(width) is not int or width < 1):
        raise ValueError("DFly intermediate size must be positive")
    window = option("context_local_window", 0)
    if window is None:
        window = 0
    if type(window) is not int or window < 0:
        raise ValueError("DFly context_local_window must be nonnegative")
    values = target.to_dict()
    values.update(
        model_type="hy_v4",
        architectures=["HYV4DFlyModel"],
        num_hidden_layers=count,
        n_routed_experts=0,
        n_shared_experts=0,
        num_nextn_predict_layers=0,
        enable_ihc=False,
        layer_types=[SPARSE_ATTENTION] * count,
        mlp_layer_types=["dense"] * count,
        indexer_types=["full"] * count,
        rope_parameters=ropes[0],
        quantization_config=draft_quantization_config(
            getattr(target, "quantization_config", None)
        ),
    )
    draft = HYV4Config.from_dict(values)
    draft.model_type = "hy_v4_dfly"
    draft.target_num_hidden_layers = target.num_hidden_layers
    draft.target_hidden_size = target.hidden_size
    draft.target_kv_layers = kv_layers
    draft.target_index_layer = index_layers
    draft.target_index_topk = option("target_index_topk", target.index_topk)
    draft.dfly_intermediate_size = width
    if width is not None:
        draft.intermediate_size = width
    draft.context_local_window = window
    draft.mask_token_id = mask
    draft.engram_layer_ids = None
    draft.routed_expert_hidden_size = None
    draft.attn_post_proj_norm = option("attn_post_proj_norm")
    if (
        draft.attn_post_proj_norm is not None
        and type(draft.attn_post_proj_norm) is not bool
    ):
        raise ValueError("DFly attention post-projection norm must be boolean")
    draft.hidden_correction_intermediate_size = option(
        "hidden_correction_intermediate_size", target.hidden_size
    )
    if type(draft.hidden_correction_intermediate_size) is not int or (
        draft.hidden_correction_intermediate_size < 1
    ):
        raise ValueError("DFly hidden correction width must be positive")
    return draft


@dataclass
class DFlyInputs:
    input_ids: torch.Tensor
    positions: torch.Tensor
    is_padding: torch.Tensor
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    active: torch.Tensor
    source_rows: torch.Tensor
    anchors: torch.Tensor
    target_hidden: torch.Tensor
    idx_mapping: torch.Tensor


def prepare_dfly_inputs(
    *,
    target_query_start_loc,
    target_positions,
    idx_mapping,
    num_sampled,
    num_rejected,
    last_sampled,
    target_hidden,
    mask_token_id: int,
    query_width: int,
    max_model_len: int,
    padded_batch: int,
):
    """Torch reference; allocating and validating, not capture-ready."""
    requests = num_sampled.numel()
    if query_width < 1 or padded_batch < requests:
        raise ValueError("Invalid DFly query width or padded batch")
    if (
        target_query_start_loc.numel() != requests + 1
        or num_rejected.numel() != requests
        or idx_mapping.numel() != requests
    ):
        raise ValueError("Invalid DFly request layout")
    device = target_positions.device
    source = target_query_start_loc[1:].long() - num_rejected.long() - 1
    if requests and bool(
        (
            (source < target_query_start_loc[:-1])
            | (source >= target_query_start_loc[1:])
        ).any()
    ):
        raise ValueError("DFly rejection count leaves no accepted source row")
    if requests and bool((idx_mapping < 0).any()):
        raise ValueError("Real DFly requests require valid state indices")
    source_rows = torch.zeros(padded_batch, dtype=torch.int64, device=device)
    anchors = torch.zeros(padded_batch, dtype=target_positions.dtype, device=device)
    active = torch.zeros(padded_batch, dtype=torch.bool, device=device)
    mapping = torch.full((padded_batch,), -1, dtype=torch.int32, device=device)
    hidden = target_hidden.new_zeros((padded_batch, target_hidden.shape[-1]))
    bonus = torch.zeros(padded_batch, dtype=torch.int32, device=device)
    if requests:
        source_rows[:requests] = source
        anchors[:requests] = target_positions[source] + 1
        active[:requests] = (num_sampled > 0) & (
            anchors[:requests] + query_width <= max_model_len
        )
        mapping[:requests] = torch.where(active[:requests], idx_mapping, -1).int()
        hidden[:requests] = target_hidden[source]
        sampled = last_sampled[:, 0] if last_sampled.ndim == 2 else last_sampled
        bonus[:requests] = torch.where(
            active[:requests], sampled[idx_mapping.long()], 0
        ).int()
    offsets = torch.arange(query_width, device=device)
    positions = torch.where(active[:, None], anchors[:, None] + offsets, offsets)
    ids = torch.full(
        (padded_batch, query_width), mask_token_id, dtype=torch.int32, device=device
    )
    ids[:, 0] = bonus
    return DFlyInputs(
        input_ids=ids.flatten(),
        positions=positions.flatten(),
        is_padding=(~active[:, None]).expand(-1, query_width).reshape(-1),
        query_start_loc=torch.arange(padded_batch + 1, dtype=torch.int32, device=device)
        * query_width,
        seq_lens=torch.where(active, anchors, 0).int() + query_width,
        active=active,
        source_rows=source_rows,
        anchors=anchors,
        target_hidden=hidden,
        idx_mapping=mapping,
    )


def build_dfly_reuse_indices(
    snapshots,
    layer_snapshot_slots,
    active,
    source_rows,
    anchors,
    *,
    local_window: int,
    query_width: int,
):
    """Return [draft layers, padded_batch * query_width, aligned width].

    snapshots contains logical positions, not physical cache offsets.
    """
    if local_window < 0 or query_width < 1:
        raise ValueError("Invalid DFly window or query width")
    layers = layer_snapshot_slots.numel()
    batch = active.numel()
    topk = snapshots.shape[-1]
    width = ((topk + local_window + query_width + 127) // 128) * 128
    out = snapshots.new_full((layers, batch, query_width, width), -1)
    if layers == 0 or batch == 0:
        return out.reshape(layers, batch * query_width, width)
    history = snapshots[
        layer_snapshot_slots.long()[:, None], source_rows.long()[None, :]
    ]
    window_start = anchors - local_window
    history = torch.where(
        (history >= 0) & (history < window_start[None, :, None]), history, -1
    )
    out[:, :, :, :topk] = history[:, :, None, :]
    if local_window:
        offsets = torch.arange(local_window, device=out.device)
        window = (window_start[:, None] + offsets).clamp_min(-1)
        out[:, :, :, topk : topk + local_window] = window[None, :, None, :]
    offsets = torch.arange(query_width, device=out.device)
    causal = anchors[:, None, None] + offsets[None, None, :]
    causal = causal.expand(batch, query_width, query_width)
    causal = causal.masked_fill(offsets[None, :] > offsets[:, None], -1)
    out[:, :, :, topk + local_window : topk + local_window + query_width] = causal
    out.masked_fill_(~active[None, :, None, None], -1)
    return out.reshape(layers, batch * query_width, width)
