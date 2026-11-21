# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Hy4 Flash quantization validation and routed-expert weight budget helpers.

This module deliberately imports no torch or CUDA code so config validation and
the CPU budget check can run without initializing a CUDA device.
"""

import copy
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import regex as re

__all__ = [
    "hy4_dfly_intermediate_size",
    "hy4_pack_module_name",
    "hy4_pack_module_names",
    "hy4_routed_expert_weight_budget",
    "split_hy4_target_and_draft_quantization",
    "validate_checkpoint_quantization",
    "validate_hy4_pack_tensor_coverage",
]

_WNA16_GROUP_BITS = {
    "routed_experts_w4a16": 4,
    "dfly_w4a16": 4,
    "dfly_w8a16": 8,
}


def _has_fields(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    for key, value in expected.items():
        field = actual.get(key)
        if type(field) is not type(value) or field != value:
            return False
    return True


def _quantization_error() -> ValueError:
    return ValueError(
        "Hy4 Flash supports native blockwise FP8, compressed-tensors block-128 "
        "FP8, weight-only MXFP4, and the named group-128 W4A16/W8A16 pack groups; "
        "checkpoint KV-cache schemes and transforms are unsupported"
    )


def _validate_checkpoint_quantization_metadata(config: object) -> None:
    """Validate checkpoint quantization schemes supported by the Hy4 port.

    The W4A16/W8A16 group names are intentional: these formats are accepted
    only for the routed-expert and DFly groups emitted by the Hy4 pack.
    """
    if not isinstance(config, Mapping):
        raise _quantization_error()
    if config.get("kv_cache_scheme") is not None:
        raise _quantization_error()
    if config.get("transform_config") is not None:
        raise _quantization_error()

    if config.get("quant_method") == "fp8":
        if (
            config.get("activation_scheme") != "dynamic"
            or config.get("weight_block_size") != [128, 128]
            or config.get("fmt") not in (None, "e4m3")
            or config.get("store_dtype") not in (None, "fp8")
            or config.get("expert_dtype") not in (None, "fp8")
        ):
            raise _quantization_error()
        return

    quant_format = config.get("format")
    if config.get("quant_method") != "compressed-tensors" or quant_format not in (
        "float-quantized",
        "mxfp4-pack-quantized",
        "mixed-precision",
        "pack-quantized",
    ):
        raise _quantization_error()

    groups = config.get("config_groups")
    if not isinstance(groups, Mapping) or not groups:
        raise _quantization_error()

    expected_fp8_weight = {
        "num_bits": 8,
        "type": "float",
        "strategy": "block",
        "block_structure": [128, 128],
        "symmetric": True,
        "dynamic": False,
    }
    expected_fp8_activations = {
        "num_bits": 8,
        "type": "float",
        "strategy": "group",
        "group_size": 128,
        "symmetric": True,
        "dynamic": True,
    }
    expected_mxfp4_weight = {
        "num_bits": 4,
        "type": "float",
        "strategy": "group",
        "group_size": 32,
        "symmetric": True,
        "dynamic": False,
    }

    for name, group in groups.items():
        if not isinstance(group, Mapping):
            raise _quantization_error()

        weights = group.get("weights")
        activations = group.get("input_activations")
        if (
            not isinstance(weights, Mapping)
            or group.get("output_activations") is not None
        ):
            raise _quantization_error()

        group_format = group.get("format") or quant_format
        targets = group.get("targets")

        expected_bits = _WNA16_GROUP_BITS.get(name)
        is_hy4_wna16 = (
            expected_bits is not None
            and group_format == "pack-quantized"
            and activations is None
            and isinstance(targets, list)
            and bool(targets)
            and _has_fields(
                weights,
                {
                    "num_bits": expected_bits,
                    "type": "int",
                    "strategy": "group",
                    "group_size": 128,
                    "symmetric": True,
                    "dynamic": False,
                },
            )
        )
        is_fp8 = (
            group_format == "float-quantized"
            and isinstance(activations, Mapping)
            and _has_fields(weights, expected_fp8_weight)
            and _has_fields(activations, expected_fp8_activations)
        )
        is_mxfp4 = (
            group_format == "mxfp4-pack-quantized"
            and activations is None
            and _has_fields(weights, expected_mxfp4_weight)
            and weights.get("scale_dtype") in (None, "uint8", "torch.uint8")
        )
        if not (is_hy4_wna16 or is_fp8 or is_mxfp4):
            raise _quantization_error()


_PACK_TENSOR_SUFFIXES = (
    ".weight_zero_point",
    ".weight_scale_inv",
    ".weight_packed",
    ".weight_scale",
    ".weight_shape",
    ".weight",
)


def _unique_names(names: Iterable[str], label: str) -> tuple[str, ...]:
    if isinstance(names, (str, bytes)):
        raise ValueError(f"{label} must be an iterable of names, not a string")
    try:
        values = tuple(names)
    except TypeError as exc:
        raise ValueError(f"{label} must be an iterable of names") from exc
    if any(not isinstance(name, str) or not name for name in values):
        raise ValueError(f"{label} must contain non-empty strings")
    if len(values) != len(set(values)):
        raise ValueError(f"{label} contains duplicate names")
    return values


def hy4_pack_module_name(tensor_name: str) -> str:
    """Return the module prefix for a Hy4 checkpoint weight or quant tensor."""
    if not isinstance(tensor_name, str) or not tensor_name:
        raise ValueError("tensor_name must be a non-empty string")
    for suffix in _PACK_TENSOR_SUFFIXES:
        if tensor_name.endswith(suffix):
            return tensor_name[: -len(suffix)]
    raise ValueError(f"unrecognized Hy4 weight tensor suffix: {tensor_name!r}")


def hy4_pack_module_names(tensor_names: Iterable[str]) -> tuple[str, ...]:
    """Normalize pack tensor names, including weight_shape sidecars, to modules."""
    tensor_names = _unique_names(tensor_names, "tensor_names")
    modules = dict.fromkeys(hy4_pack_module_name(name) for name in tensor_names)
    return tuple(modules)


def validate_hy4_pack_tensor_coverage(
    expected_names: Iterable[str], loaded_names: Iterable[str]
) -> None:
    """Require each expected pack tensor to be loaded exactly once."""
    expected = set(_unique_names(expected_names, "expected_names"))
    loaded = set(_unique_names(loaded_names, "loaded_names"))
    missing = sorted(expected - loaded)
    unexpected = sorted(loaded - expected)
    if missing or unexpected:
        raise ValueError(
            f"Hy4 pack tensor coverage mismatch: missing={missing[:5]}, "
            f"unexpected={unexpected[:5]}"
        )


def _matches_targets(module_name: str, targets: Sequence[str]) -> bool:
    for target in targets:
        if target.startswith("re:"):
            try:
                if re.fullmatch(target[3:], module_name):
                    return True
            except re.error as exc:
                raise ValueError(
                    f"invalid quantization target regex {target!r}"
                ) from exc
        elif module_name == target:
            return True
    return False


def _is_routed_expert_module(module_name: str) -> bool:
    return ".mlp.experts." in module_name and module_name.rsplit(".", 1)[-1] in {
        "gate_proj",
        "up_proj",
        "down_proj",
        "gate_up_proj",
    }


def _is_dfly_module(module_name: str) -> bool:
    # The pack's DFly groups also target the draft-level modules
    # dfly_layers.hidden_correction.* and dfly_layers.mtp_fusion.eh_proj.
    return (
        re.search(
            r"(?:^|\.)dfly_layers\.(?:\d+|hidden_correction|mtp_fusion)(?:\.|$)",
            module_name,
        )
        is not None
    )


def validate_checkpoint_quantization(
    config: object,
    *,
    module_names: Iterable[str] | None = None,
    require_routed_experts: bool = True,
    require_dfly: bool = False,
) -> None:
    """Validate supported formats and resolve Hy4 groups against pack modules.

    module_names contains canonical checkpoint module prefixes before any
    DFly prefix mapping. WNA16 packs are rejected if validation is metadata-only.
    """
    _validate_checkpoint_quantization_metadata(config)
    if (
        not isinstance(config, Mapping)
        or config.get("quant_method") != "compressed-tensors"
    ):
        return
    groups = config.get("config_groups")
    if not isinstance(groups, Mapping) or not groups:
        return
    if any(not isinstance(name, str) for name in groups):
        raise _quantization_error()

    has_wna16_groups = any(name in _WNA16_GROUP_BITS for name in groups)
    if (
        require_routed_experts
        and has_wna16_groups
        and "routed_experts_w4a16" not in groups
    ):
        raise ValueError("Hy4 target config is missing routed_experts_w4a16")
    dfly_groups = [name for name in groups if name.startswith("dfly_")]
    if len(dfly_groups) > 1:
        raise ValueError("a Hy4 draft config must select one DFly quantization group")
    if require_dfly and not dfly_groups:
        raise ValueError("Hy4 draft config is missing a DFly quantization group")

    needs_resolution = any(
        name in _WNA16_GROUP_BITS
        or name.startswith("dfly_")
        or name == "attn_fp8_block128"
        for name in groups
    )
    if module_names is None:
        if needs_resolution:
            raise ValueError(
                "Hy4 pack quantization requires resolved module_names; "
                "metadata-only validation is insufficient"
            )
        return
    modules = _unique_names(module_names, "module_names")
    if not modules:
        raise ValueError("module_names must not be empty")

    resolved: dict[str, set[str]] = {}
    for name, group in groups.items():
        targets = group.get("targets") if isinstance(group, Mapping) else None
        if (
            targets is None
            and name not in _WNA16_GROUP_BITS
            and not name.startswith("dfly_")
        ):
            continue
        if (
            not isinstance(targets, list)
            or not targets
            or any(not isinstance(target, str) or not target for target in targets)
        ):
            raise ValueError(f"quantization group {name!r} has invalid targets")
        matched = {module for module in modules if _matches_targets(module, targets)}
        if not matched:
            raise ValueError(f"quantization group {name!r} matches no pack modules")
        if name.startswith("dfly_"):
            outside = sorted(
                module for module in matched if not _is_dfly_module(module)
            )
            if outside:
                raise ValueError(
                    f"DFly group {name!r} targets non-DFly modules: {outside[:5]}"
                )
        if name == "routed_experts_w4a16":
            outside = sorted(
                module for module in matched if not _is_routed_expert_module(module)
            )
            if outside:
                raise ValueError(
                    f"routed expert group targets non-experts: {outside[:5]}"
                )
        resolved[name] = matched

    for name, matched in resolved.items():
        for other_name, other_matched in resolved.items():
            if name >= other_name:
                continue
            overlap = sorted(matched & other_matched)
            if overlap:
                raise ValueError(
                    f"quantization groups {name!r} and {other_name!r} overlap: "
                    f"{overlap[:5]}"
                )

    if require_routed_experts and has_wna16_groups:
        expected_experts = {
            module for module in modules if _is_routed_expert_module(module)
        }
        if not expected_experts:
            raise ValueError("module_names contain no routed-expert pack modules")
        missing_experts = sorted(
            expected_experts - resolved.get("routed_experts_w4a16", set())
        )
        if missing_experts:
            raise ValueError(
                "routed_experts_w4a16 does not cover expert modules: "
                f"{missing_experts[:5]}"
            )


def split_hy4_target_and_draft_quantization(
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Separate target groups from the single DFly group before prefix mapping."""
    if not isinstance(config, Mapping):
        raise ValueError("quantization config must be a mapping")
    groups = config.get("config_groups")
    if not isinstance(groups, Mapping) or not groups:
        raise ValueError("quantization config has no config_groups")
    if any(not isinstance(name, str) for name in groups):
        raise ValueError("quantization group names must be strings")
    target_groups = {
        name: group for name, group in groups.items() if not name.startswith("dfly_")
    }
    draft_groups = {
        name: group for name, group in groups.items() if name.startswith("dfly_")
    }
    if not target_groups:
        raise ValueError("quantization config has no target groups")
    if len(draft_groups) > 1:
        raise ValueError("select one DFly group for each draft model")

    target_config = copy.deepcopy(dict(config))
    target_config["config_groups"] = copy.deepcopy(target_groups)
    draft_config = None
    if draft_groups:
        draft_config = copy.deepcopy(dict(config))
        draft_config["config_groups"] = copy.deepcopy(draft_groups)
    return target_config, draft_config


def _linear_logical_shape(
    packed_shape: Sequence[int],
    bits: int,
    weight_shape: Sequence[int] | None,
) -> tuple[int, int]:
    if type(bits) is not int or bits not in (4, 8):
        raise ValueError("bits must be 4 or 8")
    try:
        packed = tuple(packed_shape)
    except TypeError as exc:
        raise ValueError("packed_shape must contain two dimensions") from exc
    if len(packed) != 2 or any(type(dim) is not int or dim < 1 for dim in packed):
        raise ValueError("packed_shape must contain two positive integer dimensions")
    factor = 32 // bits
    if weight_shape is None:
        return packed[0], packed[1] * factor
    try:
        logical = tuple(weight_shape)
    except TypeError as exc:
        raise ValueError("weight_shape must contain two dimensions") from exc
    if len(logical) != 2 or any(type(dim) is not int or dim < 1 for dim in logical):
        raise ValueError("weight_shape must contain two positive integer dimensions")
    if logical[0] != packed[0] or logical[1] != packed[1] * factor:
        raise ValueError("weight_shape does not match the packed tensor shape")
    return logical[0], logical[1]


def hy4_dfly_intermediate_size(
    gate_proj_packed_shape: Sequence[int],
    down_proj_packed_shape: Sequence[int],
    *,
    bits: int,
    gate_proj_weight_shape: Sequence[int] | None = None,
    down_proj_weight_shape: Sequence[int] | None = None,
) -> int:
    """Probe DFly FFN width from packed linear shapes or weight_shape sidecars."""
    gate_shape = _linear_logical_shape(
        gate_proj_packed_shape, bits, gate_proj_weight_shape
    )
    down_shape = _linear_logical_shape(
        down_proj_packed_shape, bits, down_proj_weight_shape
    )
    if gate_shape[1] != down_shape[0]:
        raise ValueError("DFly gate/down projections disagree on hidden size")
    if gate_shape[0] != down_shape[1]:
        raise ValueError("DFly gate/down projections disagree on FFN width")
    return gate_shape[0]


def hy4_routed_expert_weight_budget(
    config: Mapping[str, object],
    *,
    tp_size: int = 2,
    pp_size: int = 2,
    bits: int = 4,
    group_size: int = 128,
    stage_layer_bounds: Sequence[tuple[int, int]] | None = None,
) -> tuple[float, ...]:
    """Return routed-expert weight GiB per TP rank for each PP stage.

    The routed Hy4 expert matrices have three latent-width square weights per
    expert. The result includes packed weight storage and one BF16 scale per
    group. Other model weights, runtime allocations, and KV cache are excluded.
    Supply actual contiguous half-open PP ranges in stage_layer_bounds when the
    pipeline split is uneven.
    """
    if not isinstance(config, Mapping):
        raise ValueError("config must be a mapping")
    text_config = config.get("text_config", config)
    if not isinstance(text_config, Mapping):
        raise ValueError("text_config must be a mapping")

    for name, value in (("tp_size", tp_size), ("pp_size", pp_size)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if bits not in (4, 8):
        raise ValueError("bits must be 4 or 8")
    if type(group_size) is not int or group_size < 1:
        raise ValueError("group_size must be a positive integer")

    num_layers = text_config.get("num_hidden_layers")
    num_experts = text_config.get("n_routed_experts")
    latent_size = text_config.get("routed_expert_hidden_size")
    if latent_size is None:
        latent_size = text_config.get("moe_intermediate_size")

    for name, value in (
        ("num_hidden_layers", num_layers),
        ("n_routed_experts", num_experts),
        ("routed_expert_hidden_size", latent_size),
    ):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")

    pack_factor = 32 // bits
    if latent_size % group_size or latent_size % pack_factor or latent_size % tp_size:
        raise ValueError(
            "routed expert latent size must divide group_size, the packed "
            "weight factor, and TP size"
        )

    mlp_layer_types = text_config.get("mlp_layer_types")
    if mlp_layer_types is None:
        dense_layers = text_config.get("first_k_dense_replace", 1)
        if type(dense_layers) is not int or not 0 <= dense_layers <= num_layers:
            raise ValueError("first_k_dense_replace is outside the model layer range")
        mlp_layer_types = ["dense"] * dense_layers + ["sparse"] * (
            num_layers - dense_layers
        )
    if (
        not isinstance(mlp_layer_types, Sequence)
        or isinstance(mlp_layer_types, (str, bytes))
        or len(mlp_layer_types) != num_layers
    ):
        raise ValueError("mlp_layer_types must contain one entry per model layer")

    if stage_layer_bounds is None:
        base_layers, extra_layers = divmod(num_layers, pp_size)
        ranges = []
        start = 0
        for stage in range(pp_size):
            stage_layers = base_layers + (1 if stage < extra_layers else 0)
            ranges.append((start, start + stage_layers))
            start += stage_layers
    else:
        if (
            not isinstance(stage_layer_bounds, Sequence)
            or isinstance(stage_layer_bounds, (str, bytes))
            or len(stage_layer_bounds) != pp_size
        ):
            raise ValueError("stage_layer_bounds must have one range per PP stage")
        ranges = []
        expected_start = 0
        for bounds in stage_layer_bounds:
            if (
                not isinstance(bounds, Sequence)
                or isinstance(bounds, (str, bytes))
                or len(bounds) != 2
                or type(bounds[0]) is not int
                or type(bounds[1]) is not int
                or bounds[0] != expected_start
                or bounds[1] <= bounds[0]
            ):
                raise ValueError(
                    "stage_layer_bounds must be contiguous, increasing, "
                    "half-open ranges"
                )
            ranges.append((bounds[0], bounds[1]))
            expected_start = bounds[1]
        if expected_start != num_layers:
            raise ValueError("stage_layer_bounds must cover every model layer")

    moe_by_stage: list[int] = []
    for start, end in ranges:
        types = mlp_layer_types[start:end]
        unknown = [kind for kind in types if kind not in ("dense", "sparse", "moe")]
        if unknown:
            raise ValueError(f"unsupported MLP layer type: {unknown[0]!r}")
        moe_by_stage.append(sum(kind != "dense" for kind in types))

    params_per_layer = num_experts * 3 * latent_size * latent_size
    bytes_per_param = bits / 8 + 2 / group_size
    gib_per_moe_layer_per_rank = (
        params_per_layer * bytes_per_param / tp_size / (1024**3)
    )
    return tuple(count * gib_per_moe_layer_per_rank for count in moe_by_stage)
