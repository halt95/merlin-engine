# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Checkpoint helpers shared by HYV4 target and draft models."""

from collections.abc import Mapping
from copy import deepcopy

import regex as re
import torch


def is_compressed_tensors(checkpoint) -> bool:
    return (
        isinstance(checkpoint, Mapping)
        and checkpoint.get("quant_method") == "compressed-tensors"
    )


def pack_module_names(tensor_names) -> tuple[str, ...]:
    """Module prefixes of a checkpoint's linear tensors, in first-seen order.

    The Hy4 quantization validator resolves a compressed-tensors pack's config
    groups against these (canonical checkpoint names, before any DFly prefix
    mapping). Leaves
    without a weight suffix (sinks, iHC params, fused BF16 expert tensors) are
    not quantisation targets and are skipped.
    """
    from vllm.merlin.hy4_quant import hy4_pack_module_name

    modules: dict[str, None] = {}
    for name in tensor_names:
        try:
            modules.setdefault(hy4_pack_module_name(name))
        except ValueError:
            continue
    return tuple(modules)


def checkpoint_module_names(model_config) -> tuple[str, ...]:
    """``pack_module_names`` of the checkpoint ``model_config`` loads, read from
    its safetensors headers (no tensor data)."""
    from vllm.transformers_utils.config import get_safetensors_params_metadata

    model = getattr(model_config, "model_weights", "") or model_config.model
    names = get_safetensors_params_metadata(
        model, revision=getattr(model_config, "revision", None)
    )
    modules = pack_module_names(names)
    if not modules:
        raise ValueError(
            f"Could not read the safetensors tensor names of {model!r}; the Hy4 "
            "compressed-tensors pack is validated against them"
        )
    return modules


def validate_hy4_checkpoint_quantization(
    checkpoint, model_config=None, *, module_names=None, **kwargs
) -> None:
    """The Hy4 quantization validator with the pack's module names resolved.

    A compressed-tensors pack is checked against ``module_names`` or, if not
    given, the names in ``model_config``'s safetensors headers; FP8 and BF16
    checkpoints need no names.
    """
    from vllm.merlin.hy4_quant import validate_checkpoint_quantization

    if checkpoint is None:
        return
    if (
        module_names is None
        and model_config is not None
        and is_compressed_tensors(checkpoint)
    ):
        module_names = checkpoint_module_names(model_config)
    validate_checkpoint_quantization(checkpoint, module_names=module_names, **kwargs)


def draft_quantization_config(checkpoint, *, module_names=None):
    """Select draft groups while retaining their original checkpoint targets."""
    if checkpoint is None:
        return None

    validate_hy4_checkpoint_quantization(checkpoint, module_names=module_names)
    result = deepcopy(checkpoint)
    if result["quant_method"] == "fp8":
        return result

    result["config_groups"] = {
        name: group
        for name, group in result["config_groups"].items()
        if name.startswith("dfly_")
    }
    if not result["config_groups"]:
        return None
    validate_hy4_checkpoint_quantization(
        result,
        module_names=None
        if module_names is None
        else [name for name in module_names if "dfly_layers." in name],
        require_routed_experts=False,
        require_dfly=True,
    )
    return result


def validate_expert_shape(config, projection, value, *, expert_id):
    """Consume pack metadata that WNA16 MoE does not register as a parameter."""
    latent = config.routed_expert_hidden_size or config.hidden_size
    intermediate = config.moe_intermediate_size
    shapes = {
        "gate_proj": (intermediate, latent),
        "up_proj": (intermediate, latent),
        "gate_up_proj": (2 * intermediate, latent),
        "down_proj": (latent, intermediate),
    }
    expected = shapes[projection]
    experts = config.n_routed_experts
    if value.dtype not in (torch.int32, torch.int64):
        raise ValueError("Expert weight_shape must contain integers")
    if expert_id is None:
        if tuple(value.shape) != (experts, 2):
            raise ValueError("Fused expert weight_shape must be [experts, 2]")
        wanted = value.new_tensor(expected).expand(experts, 2)
    else:
        if not 0 <= int(expert_id) < experts:
            raise ValueError("Expert index out of range")
        if tuple(value.shape) != (2,):
            raise ValueError("Per-expert weight_shape must be [2]")
        wanted = value.new_tensor(expected)
    if not torch.equal(value, wanted):
        raise ValueError(
            f"Expert weight_shape disagrees with {projection}: expected {expected}"
        )


_FFN_WEIGHT = re.compile(
    r"layers\.\d+\.mlp\.(gate_proj|up_proj|down_proj)\."
    r"(weight|weight_packed|weight_shape|weight_scale|weight_scale_inv)"
)


def probe_dfly_ffn_width(name, value, hidden_size, *, bits=None):
    """Infer the original FFN width from a streamed checkpoint entry.

    Packed INT4/INT8 tensors use compressed-tensors' [N, K / pack] layout.
    Block-FP8 scales alone do not determine the exact original width.
    """
    match = _FFN_WEIGHT.fullmatch(name)
    if match is None:
        return None
    projection, leaf = match.groups()
    down = projection == "down_proj"

    if leaf == "weight_shape":
        if value.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"Invalid DFly weight_shape dtype: {name}")
        if tuple(value.shape) != (2,):
            raise ValueError(f"Invalid DFly weight_shape: {name}")
        shape = tuple(int(x) for x in value.cpu().tolist())
    elif leaf == "weight":
        if value.ndim != 2:
            raise ValueError(f"Invalid DFly FFN weight: {name}")
        shape = tuple(value.shape)
    elif leaf == "weight_packed":
        if bits not in (4, 8) or value.dtype != torch.int32 or value.ndim != 2:
            raise ValueError(f"Invalid DFly packed FFN weight: {name}")
        shape = (value.shape[0], value.shape[1] * (32 // bits))
    elif leaf == "weight_scale" and bits in (4, 8):
        if value.ndim != 2:
            raise ValueError(f"Invalid DFly FFN scales: {name}")
        shape = (value.shape[0], value.shape[1] * 128)
    else:
        return None

    if min(shape) < 1 or shape[0 if down else 1] != hidden_size:
        raise ValueError(f"Invalid DFly FFN hidden size: {name}, {shape}")
    return shape[1 if down else 0]
