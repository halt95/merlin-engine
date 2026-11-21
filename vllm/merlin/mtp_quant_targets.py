# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Layer index of the Qwen3.5 MTP layers, decided at load.

A checkpoint names its MTP layer ``mtp.layers.0``; the drafter can build it at the
runtime index ``<num_hidden_layers + 0>`` (the index the runner's KV order and the
state-sharing map see). A quantisation config that matches layer names (a
compressed-tensors target or ignore entry ``mtp.layers.0.*``) then has to follow
the rename. This module renames the exact entries naming ``<prefix>.layers.N`` on a
copy of the config and VERIFIES the rename before any layer is built: for every MTP
module, every by-name outcome the engine derives from the config is computed with
the engine's own functions on the original config at the old name and on the
renamed copy at the runtime name. If all are equal the layers are built at the
runtime index from the copy. If any differs, or anything cannot be verified (a
quantisation method that is not compressed-tensors, an online quantisation config,
a declared expert-substitution spec, or any exception inside the check), this module
returns nothing and one warning names the reason: the caller then runs the base
engine's own construction, untouched (the caller's config object, the checkpoint
index, the base's order of side effects). The check only reads: the config under
test is probed on copies.

The by-name readers of a compressed-tensors config (all covered):

* ``get_scheme_dict``: the ignore list (``should_ignore_layer``), then the first
  target in map order that matches the name, then a target for every shard of a
  fused name, then the targets against the module's class name;
* ``get_linear_transform_schemes``: the transform config's targets and ignore, per
  partition name and against the module's class;
* ``get_moe_method``: the scheme of the routed experts, read through expert 0's
  three projections.

The expert-substitution spec of the model config is keyed by layer index, so a
config that declares it also keeps the checkpoint index.
"""

import copy
import importlib
from collections.abc import Mapping

import regex as re

from vllm.config import VllmConfig
from vllm.logger import init_logger

logger = init_logger(__name__)

# The modules of one MTP layer (a full-attention Qwen3.5 decoder layer) whose
# quantisation the engine resolves, with their classes (which the engine's matchers
# read); a test reads the layer classes' source and compares.
_ATTENTION = {
    "self_attn.qkv_proj": "linear:QKVParallelLinear",
    "self_attn.o_proj": "linear:RowParallelLinear",
}
_DENSE_MLP = {
    "mlp.gate_up_proj": "linear:MergedColumnParallelLinear",
    "mlp.down_proj": "linear:RowParallelLinear",
}
# The MoE block: the router, the shared expert and, as the engine resolves the
# routed experts, expert 0's projections.
_MOE_MLP = {
    "mlp.gate": "gate:GateLinear",
    "mlp.shared_expert_gate": "linear:ReplicatedLinear",
    "mlp.shared_expert.gate_up_proj": "linear:MergedColumnParallelLinear",
    "mlp.shared_expert.down_proj": "linear:RowParallelLinear",
    "mlp.experts.0.gate_proj": "experts:RoutedExperts",
    "mlp.experts.0.up_proj": "experts:RoutedExperts",
    "mlp.experts.0.down_proj": "experts:RoutedExperts",
}
_MODULE_HOMES = {
    "linear": "vllm.model_executor.layers.linear",
    "gate": "vllm.model_executor.layers.fused_moe.router.gate_linear",
    "experts": "vllm.model_executor.layers.fused_moe.routed_experts",
}
_MOE_MODEL_TYPE = "qwen3_5_moe_text"


def mtp_modules(model_type: str | None) -> dict[str, str]:
    """Module path under ``<prefix>.layers.<i>.`` -> ``home:ClassName``."""
    mlp = _MOE_MLP if model_type == _MOE_MODEL_TYPE else _DENSE_MLP
    return {**_ATTENTION, **mlp}


def _module_instance(spec: str):
    """An instance of the module's real class, with no state: the engine's matchers
    read only the class (its name and bases)."""
    home, _, class_name = spec.partition(":")
    cls = getattr(importlib.import_module(_MODULE_HOMES[home]), class_name)
    return object.__new__(cls)


def remap_mtp_target(
    name: str,
    mtp_start_layer_idx: int,
    num_mtp_layers: int,
    layer_prefix: str = "mtp",
) -> str:
    """``<prefix>.layers.N.*`` -> ``<prefix>.layers.(start+N).*`` for N below the MTP
    layer count; any other name (a pattern, a wrapper-prefixed or other index) as
    written."""
    if name.startswith("re:"):
        return name
    m = re.match(rf"{re.escape(layer_prefix)}\.layers\.(\d+)(?=\.)", name)
    if m is None or int(m.group(1)) >= num_mtp_layers:
        return name
    index = mtp_start_layer_idx + int(m.group(1))
    return f"{layer_prefix}.layers.{index}{name[m.end() :]}"


def renamed_quant_config(
    quant_config, mtp_start_layer_idx: int, num_mtp_layers: int, layer_prefix: str
):
    """A copy of a compressed-tensors config whose exact targets and ignore entries
    naming ``<prefix>.layers.N`` name the runtime index. The target map, its scheme
    dictionaries and the ignore list are new containers; patterns are not edited."""

    def rename(name: str) -> str:
        return remap_mtp_target(name, mtp_start_layer_idx, num_mtp_layers, layer_prefix)

    out = copy.copy(quant_config)
    out.target_scheme_map = {
        rename(key): dict(value) if isinstance(value, dict) else value
        for key, value in quant_config.target_scheme_map.items()
    }
    if isinstance(quant_config.ignore, list):
        out.ignore = [rename(entry) for entry in quant_config.ignore]
    return out


def _normal(value, quant_format):
    """Comparable form of a resolved scheme: a scheme without a format takes the
    config's, as the engine's lookup writes it."""
    if isinstance(value, dict):
        return {**value, "format": value.get("format") or quant_format}
    return value


def _call(read):
    try:
        return read()
    except Exception as e:  # compared as raised-or-not and the type
        return ("raises", type(e).__name__)


def _outcomes(config, spec: str, name: str) -> dict:
    """Every by-name outcome the engine derives for one module under ``config``."""
    from vllm.model_executor.layers.linear import LinearBase
    from vllm.model_executor.layers.quantization.compressed_tensors.transform.linear import (  # noqa: E501
        get_linear_transform_schemes,
    )
    from vllm.model_executor.layers.quantization.compressed_tensors.utils import (
        should_ignore_layer,
    )

    # Probe a copy: the engine's lookup writes a default format into the scheme
    # dictionaries, and the routed experts' lookup adds a target.
    probe = copy.copy(config)
    probe.target_scheme_map = {k: dict(v) for k, v in config.target_scheme_map.items()}
    module = _module_instance(spec)
    mapping = config.packed_modules_mapping
    out = {
        "ignored": _call(
            lambda: should_ignore_layer(
                name, ignore=config.ignore, fused_mapping=mapping
            )
        )
    }
    if spec.startswith("experts:"):
        probe._add_fused_moe_to_target_scheme_map()
    out["scheme"] = _call(
        lambda: _normal(probe.get_scheme_dict(module, name), config.quant_format)
    )
    if isinstance(module, LinearBase):
        out["transform"] = _call(
            lambda: get_linear_transform_schemes(
                module, name, config.transform_config, mapping
            )
        )
    return out


def _first_difference(
    quant_config,
    renamed,
    mtp_start_layer_idx: int,
    num_mtp_layers: int,
    layer_prefix: str,
    model_type: str | None,
) -> str | None:
    """The first module whose outcome differs between the old name on the original
    config and the runtime name on the renamed copy, described; None if none."""
    for index in range(num_mtp_layers):
        for path, spec in mtp_modules(model_type).items():
            old = f"{layer_prefix}.layers.{index}.{path}"
            runtime = f"{layer_prefix}.layers.{mtp_start_layer_idx + index}.{path}"
            before = _outcomes(quant_config, spec, old)
            after = _outcomes(renamed, spec, runtime)
            for reader, value in before.items():
                if value != after[reader]:
                    return (
                        f"{path}: {reader} {value!r} at {old} vs "
                        f"{after[reader]!r} at {runtime}"
                    )
    return None


def runtime_index_layers(
    vllm_config: VllmConfig,
    mtp_start_layer_idx: int,
    num_mtp_layers: int,
    layer_prefix: str,
    make_layer,
):
    """The MTP layers built at the runtime index from a renamed copy of the quant
    config, when the rename is verified; None otherwise, and the caller then builds
    the layers exactly as the base engine does (nothing here has touched
    ``vllm_config``). ``make_layer(config, prefix)`` builds one layer. Anything the
    verification raises is a reason to return None: the base path then raises
    whatever it would have raised, at its own place."""
    try:
        reason = _keep_index_reason(
            vllm_config, mtp_start_layer_idx, num_mtp_layers, layer_prefix
        )
    except Exception as e:
        reason = f"the check raised {type(e).__name__}: {e}"
    if isinstance(reason, str):
        logger.warning("MTP layers keep their checkpoint index: %s", reason)
        return None
    from torch import nn

    layer_config = copy.copy(vllm_config)  # a copy: the caller's is never modified
    if reason is not None:
        layer_config.quant_config = reason
    return nn.ModuleList(
        make_layer(layer_config, f"{layer_prefix}.layers.{mtp_start_layer_idx + i}")
        for i in range(num_mtp_layers)
    )


def _declares_expert_substitution(model_config) -> bool:
    """Whether the model config declares an expert-substitution spec (keyed by layer
    index), without parsing it: the base construction does any parsing."""
    configs = [getattr(model_config, n, None) for n in ("hf_text_config", "hf_config")]
    if any(getattr(c, "approximate_experts", None) is not None for c in configs):
        return True
    compression = getattr(configs[1], "compression_config", None)
    transform = (
        compression.get("transform_config")
        if isinstance(compression, Mapping)
        else None
    )
    return (
        isinstance(transform, Mapping)
        and transform.get("expert_substitution") is not None
    )


def _keep_index_reason(vllm_config, mtp_start_layer_idx, num_mtp_layers, layer_prefix):
    """A string (why the checkpoint index is kept), the renamed config (rename
    verified), or None (no quantisation config: nothing reads names). Reads only;
    the config under test is probed on copies."""
    model_config = vllm_config.model_config
    if _declares_expert_substitution(model_config):
        return "the model config declares expert substitution by layer index"
    pp_size = getattr(
        getattr(vllm_config, "parallel_config", None), "pipeline_parallel_size", 1
    )
    if isinstance(pp_size, int) and pp_size > 1:
        # The bit-identity proof covers single-stage layouts only; a multi-stage
        # layout keeps the checkpoint index, as before this hook.
        return "the runtime index is not verified on a pipeline-parallel layout"
    quant_config = vllm_config.quant_config
    if quant_config is None:
        return None
    if not isinstance(getattr(quant_config, "target_scheme_map", None), dict):
        return (
            f"{type(quant_config).__name__} reads layer names and is not verified "
            "for the runtime index"
        )
    if getattr(quant_config, "online_quantization_config", None) is not None:
        return "online quantisation resolves layer names with its own targets"
    renamed = renamed_quant_config(
        quant_config, mtp_start_layer_idx, num_mtp_layers, layer_prefix
    )
    model_type = getattr(model_config.hf_text_config, "model_type", None)
    difference = _first_difference(
        quant_config,
        renamed,
        mtp_start_layer_idx,
        num_mtp_layers,
        layer_prefix,
        model_type,
    )
    return difference if difference is not None else renamed
