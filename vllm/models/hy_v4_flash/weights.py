# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Checkpoint-name rules for the HYV4 Flash text model.

Kept free of model and kernel imports so CPU tests can run them against the
real checkpoint index. ``nvidia/model.py`` uses these and nothing else to
name, skip and require weights.
"""

import regex as re

from vllm.model_executor.models.utils import WeightsMapper

PACKED_MODULES_MAPPING = {"gate_up_proj": ["gate_proj", "up_proj"]}

# Attention-cache parameters a quantisation method may create on `mla_attn`.
# Each defaults correctly when the checkpoint has no value for it.
# `prob_scale` comes from BaseKVCacheMethod.create_weights under fp8 and
# compressed-tensors.
CACHE_DEFAULTS = frozenset(
    {
        "q_scale",
        "k_scale",
        "v_scale",
        "prob_scale",
        "q_zero_point",
        "k_zero_point",
        "v_zero_point",
    }
)

HF_TO_VLLM_MAPPER = WeightsMapper(
    orig_to_new_stacked={
        ".mlp.gate_proj": (".mlp.gate_up_proj", 0),
        ".mlp.up_proj": (".mlp.gate_up_proj", 1),
        ".shared_experts.gate_proj": (".shared_experts.gate_up_proj", 0),
        ".shared_experts.up_proj": (".shared_experts.gate_up_proj", 1),
    },
    orig_to_new_substr={
        ".compressed_indexer.": ".indexer.",
        ".mlp.gate.wg.": ".mlp.gate.",
    },
    # Tencent's FP8 checkpoint names `modules_to_not_convert` by the VL
    # wrapper's module paths (#68296 vl_model.py:74-75). Weight names never
    # carry this prefix, so the entries only rename the ignore list.
    orig_to_new_prefix={
        "language_model.model.model.": "model.",
        "language_model.model.lm_head": "lm_head",
    },
)

EXPERT_WEIGHT = re.compile(
    r"(.*\.experts)\.(?:(\d+)\.)?(gate_proj|up_proj|down_proj|gate_up_proj)"
    r"(\.weight(?:_packed|_scale(?:_inv)?|_shape|_zero_point)?|_scale(?:_inv)?)?"
)
_EXPERT_SHARDS = {
    "gate_proj": ("w1",),
    "up_proj": ("w3",),
    "down_proj": ("w2",),
    "gate_up_proj": ("w1", "w3"),
}
_ENGRAM = re.compile(r"model\.layers\.(\d+)\.(engram|engram_layer)\.(.+)")
_DFLY_SHARED = frozenset(
    {
        "mtp_fusion",
        "hidden_correction",
        "final_norm",
        "norm",
        "embed_tokens",
        "confidence_head",
    }
)


def is_cache_default(name: str) -> bool:
    return ".mla_attn." in name and name.rsplit(".", 1)[-1] in CACHE_DEFAULTS


def required_parameter_names(named_parameters) -> set[str]:
    """Parameters a complete checkpoint must load."""
    return {name for name, _ in named_parameters if not is_cache_default(name)}


def target_weight_name(config, name: str) -> str | None:
    """Checkpoint name for the text target, or None when another part owns it.

    The vision tower, DFly draft layers and MTP layers are skipped only when the
    config declares them. Engram config ids are 1-based: checkpoint
    ``layers.{k-1}.engram_layer`` (and legacy ``layers.k.engram``) belong to
    config id k.
    """
    if name.startswith("vit."):
        return None
    engram = _ENGRAM.fullmatch(name)
    if engram is not None:
        number, kind, leaf = engram.groups()
        layer_id = int(number) + (kind == "engram_layer")
        if layer_id not in (config.engram_layer_ids or ()):
            raise ValueError(f"Undeclared Engram checkpoint weight: {name}")
        return f"model.layers.{layer_id - 1}.engram_layer.{leaf}"
    for prefix, count in (
        ("model.mtp_layers.", getattr(config, "num_nextn_predict_layers", 0)),
        ("model.dfly_layers.", getattr(config, "dfly_num_hidden_layers", 0)),
    ):
        if not name.startswith(prefix):
            continue
        local = name[len(prefix) :].removeprefix("layers.")
        first = local.split(".", 1)[0]
        declared = count > 0 and first.isdigit() and 0 <= int(first) < count
        if prefix == "model.dfly_layers." and count > 0:
            declared |= first in _DFLY_SHARED
        if not declared:
            raise ValueError(f"Undeclared auxiliary Flash weight: {name}")
        return None
    if name.startswith("model.layers."):
        number = name.removeprefix("model.layers.").split(".", 1)[0]
        start = config.num_hidden_layers
        count = getattr(config, "num_nextn_predict_layers", 0)
        if number.isdigit() and start <= int(number) < start + count:
            return None
    if config.tie_word_embeddings and name == "lm_head.weight":
        return None
    return name


def expert_target(match) -> tuple[str, str | None, str, str, str]:
    """(module prefix, expert id, projection, suffix, routed_experts target)."""
    prefix, expert_id, projection, suffix = match.groups()
    suffix = (suffix or "").removeprefix(".weight")
    leaf = "w2_weight" if projection == "down_proj" else "w13_weight"
    return (
        prefix,
        expert_id,
        projection,
        suffix,
        f"{prefix}.routed_experts.{leaf}{suffix}",
    )


def expert_pieces(name, projection, value, expert_id, num_experts):
    """Yield (expert index, FusedMoE shard id, tensor) for one expert entry.

    Checkpoint tensors use nn.Linear's [out, in] layout: ``gate_up_proj`` is
    [E, 2 * moe_intermediate, latent] with gate rows first and ``down_proj``
    is [E, latent, moe_intermediate], the layout of FusedMoE's w13/w2.
    """
    shards = _EXPERT_SHARDS[projection]
    if expert_id is None:
        if value.ndim != 3 or value.shape[0] != num_experts:
            raise ValueError(f"Invalid fused expert shape: {name}")
        ids = range(num_experts)
    else:
        ids = [int(expert_id)]
    for index in ids:
        if not 0 <= index < num_experts:
            raise ValueError(f"Expert index out of range: {name}")
        tensor = value[index] if expert_id is None else value
        if len(shards) == 1:
            pieces = (tensor,)
        else:
            if tensor.ndim < 2 or tensor.shape[-2] % 2:
                raise ValueError(
                    "Export combined gate/up metadata as separate "
                    f"gate_proj/up_proj entries: {name}"
                )
            pieces = tensor.chunk(2, dim=-2)
        for shard, piece in zip(shards, pieces):
            yield index, shard, piece


def module_parameter_name(name: str) -> str:
    """Rename checkpoint leaves whose module attribute differs (non-expert)."""
    if name.endswith((".hc_fn", ".hc_head_fn")):
        name += ".weight"
    name = name.replace(".gate.e_score_correction_bias", ".expert_bias")
    return name.replace(".router.gate.", ".gate.")
