# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modified by the Merlin project; see NOTICE.
"""HYV4 DFly2 draft model for the Hy4 Flash target.

Adapted from #68296 ``vllm/models/hy_v4/nvidia/dfly.py:1-492``.

Differences from upstream:
- Attention: each draft layer is a ``HYV4FlashAttention`` built by the same
  factory as the target's sparse layers (``hy_v4_flash.backend``), with
  ``dfly_external_topk`` so it builds no indexer and reads the indices the
  speculator writes. Upstream uses the hy_v4 (FlashMLA, sm_90+) attention.
  Same checkpoint leaves: separate ``q_a_proj``/``kv_a_proj_with_mqa``,
  ``linear_gate``, ``learnable_sink_param``, ``post_proj_norm``.
- Module paths equal checkpoint paths: the draft layers live at
  ``model.dfly_layers.N`` and the fusion/correction blocks at
  ``model.dfly_layers.{mtp_fusion,hidden_correction,final_norm}``. The pack's
  target FP8 regex ``(?<![A-Za-z_])layers\\.([1-9]|...)`` cannot match
  ``dfly_layers.N``, and its ``dfly_w4a16``/``dfly_w8a16`` groups match by name
  without upstream's prefix rewrite (#68296 dfly.py:47-74), which also refused
  any non-8-bit draft scheme.
- Quantised drafts: INT4/INT8 W*A16 group-128 compressed-tensors
  draft linears (Marlin) and FP8 block-128. Every quantised draft linear must
  match a ``dfly_*`` config group (``validate_dfly_quantization``).
- FFN width probe: learned from ``.weight`` (2-D BF16/FP8),
  ``.weight_packed`` (N unpacked, K = packed * 32 / bits) or ``.weight_shape``
  ([N, K]) of ``mlp.{gate,up,down}_proj``; upstream reads 2-D ``.weight`` only
  (#68296 dfly.py:440-451) and rejects an INT8 FFN (768 != 3072).
- Weights load through each parameter's loader (shard ids from the stacked
  mapper), like the Flash target; AutoWeightsLoader drops shard ids here.
"""

from collections.abc import Iterable, Mapping
from copy import deepcopy
from typing import Any

import torch
from torch import nn

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    LinearBase,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.quantization.compressed_tensors.utils import (
    is_equal_or_regex_match,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.utils import WeightsMapper
from vllm.models.hy_v4.dfly_config import DFLY_CHECKPOINT_PREFIX, DFLY_MODEL_TYPE
from vllm.models.hy_v4.nvidia.moe import HYV4FeedForward
from vllm.models.hy_v4_flash.nvidia.attention import HYV4FlashAttention
from vllm.utils.torch_utils import set_default_torch_dtype

from .dfly_inputs import dfly_topk_width

# Backend placeholders with no checkpoint tensor (no KV scheme is supported).
_CACHE_DEFAULTS = frozenset(
    {
        "q_scale",
        "k_scale",
        "v_scale",
        "q_zero_point",
        "k_zero_point",
        "v_zero_point",
        "prob_scale",
    }
)
_FFN_LEAVES = ("gate_proj", "up_proj", "down_proj")
_FUSED = {"gate_up_proj": ["gate_proj", "up_proj"]}
# Non-layer children of model.dfly_layers (checkpoint names).
DFLY_AUX_MODULES = ("mtp_fusion", "hidden_correction", "final_norm")
DFLY_IGNORED = ("confidence_head.",)


# ---------------------------------------------------------------- pure helpers
def dfly_quant_groups(
    quant_config: Mapping[str, Any] | None,
    layer_name: str,
    fused_mapping: Mapping[str, list[str]] = _FUSED,
) -> set[str]:
    """Names of the compressed-tensors config groups a draft linear falls in.

    Same matching as compressed-tensors: ``ignore`` first, then the first
    group whose targets match, a fused layer by each of its components.
    An empty set means the layer stays BF16.
    """
    if not quant_config:
        return set()
    groups = quant_config.get("config_groups") or {}
    ignore = quant_config.get("ignore") or []
    names = [layer_name]
    for fused, parts in fused_mapping.items():
        if layer_name.endswith(fused):
            names = [layer_name[: -len(fused)] + part for part in parts]
            break
    found: set[str] = set()
    for name in names:
        if any(is_equal_or_regex_match(name, target) for target in ignore):
            continue
        for group_name, group in groups.items():
            if any(
                is_equal_or_regex_match(name, target)
                for target in group.get("targets") or ()
            ):
                found.add(group_name)
                break
    return found


def dfly_group_bits(
    quant_config: Mapping[str, Any] | None, layer_name: str
) -> int | None:
    """Weight bits of the draft linear's group (None for BF16)."""
    groups = dfly_quant_groups(quant_config, layer_name)
    if not groups:
        return None
    bits = {
        quant_config["config_groups"][name]["weights"]["num_bits"] for name in groups
    }
    if len(bits) != 1:
        raise ValueError(f"DFly linear {layer_name} spans groups {sorted(groups)}")
    return bits.pop()


def probe_ffn_width(
    leaf: str, tensor: torch.Tensor, hidden_size: int, bits: int | None
) -> int | None:
    """The DFly FFN width a ``mlp.<proj>.<kind>`` tensor reveals, or None.

    ``leaf`` is e.g. ``gate_proj.weight_packed``. Gate/up are [N=width, K=hidden],
    down is [N=hidden, K=width]. ``weight_packed`` stores K * bits / 32 int32
    words per row; ``weight_shape`` is the unpacked [N, K]. Raises on a hidden
    size that disagrees with the draft.
    """
    proj, _, kind = leaf.partition(".")
    if proj not in _FFN_LEAVES:
        return None
    down = proj == "down_proj"
    if kind == "weight_shape":
        if tensor.numel() != 2:
            raise ValueError(f"Invalid DFly FFN weight_shape: {leaf}")
        n, k = (int(v) for v in tensor.flatten().tolist())
    elif kind in ("weight", "weight_packed"):
        if tensor.ndim != 2 or min(tensor.shape) < 1:
            raise ValueError(f"Invalid DFly FFN weight shape: {leaf}")
        n, k = int(tensor.shape[0]), int(tensor.shape[1])
        if kind == "weight_packed":
            if bits not in (4, 8):
                raise ValueError(
                    f"DFly FFN {leaf} is packed but its config group is not INT4/INT8"
                )
            k = k * 32 // bits
    else:
        return None
    if (n if down else k) != hidden_size:
        raise ValueError(
            f"Invalid DFly FFN hidden size: {leaf} {n}x{k}, hidden {hidden_size}"
        )
    return k if down else n


def checkpoint_to_local(name: str) -> str | None:
    """``model.dfly_layers.X`` -> ``model.dfly_layers.X``; others -> None.

    The draft's module tree is the checkpoint's, so names map 1:1 except the
    unused confidence head (and an optional ``layers.`` infix some exports use).
    """
    if not name.startswith(DFLY_CHECKPOINT_PREFIX):
        return None
    local = name[len(DFLY_CHECKPOINT_PREFIX) :].removeprefix("layers.")
    if local.startswith(DFLY_IGNORED):
        return None
    return DFLY_CHECKPOINT_PREFIX + local


def dfly_topk_rows(scheduler_config, query_width: int) -> int:
    """Rows of the draft's top-k buffer: one query block of ``query_width``
    (K + 1) rows per request, at most ``max_num_seqs`` requests
    (HYV4DFlySpeculator.propose runs ``num_reqs * width`` tokens and refuses
    more than max_num_batched_tokens). Sizing it by max_num_batched_tokens
    held 5 x 2048 x 2176 int32 = 85 MiB on the last stage for 48 live rows."""
    tokens = scheduler_config.max_num_batched_tokens
    seqs = getattr(scheduler_config, "max_num_seqs", None)
    if not seqs:
        return tokens
    return min(tokens, int(seqs) * query_width)


# ---------------------------------------------------------------------- modules
class HYV4DFlyFusion(nn.Module):
    """Slot-0 fusion of the bonus token embedding with the target hidden state."""

    def __init__(self, config, prefix: str, quant_config: QuantizationConfig | None):
        super().__init__()
        d = config.hidden_size
        self.enorm = RMSNorm(d, eps=config.rms_norm_eps)
        self.hnorm = RMSNorm(d, eps=config.rms_norm_eps)
        self.eh_proj = ReplicatedLinear(
            2 * d, d, bias=False, prefix=f"{prefix}.eh_proj", quant_config=quant_config
        )
        self.gate = nn.Parameter(torch.empty(d), requires_grad=False)

    def forward(
        self, anchors: torch.Tensor, target_hidden: torch.Tensor
    ) -> torch.Tensor:
        fused, _ = self.eh_proj(
            torch.cat((self.enorm(anchors), self.hnorm(target_hidden)), -1)
        )
        return anchors + self.gate * fused


class HYV4DFlyHiddenCorrection(nn.Module):
    """Corrects a query row's hidden state with the previous draft token."""

    def __init__(self, config, prefix: str, quant_config: QuantizationConfig | None):
        super().__init__()
        d = config.hidden_size
        intermediate = config.hidden_correction_intermediate_size
        self.hidden_norm = RMSNorm(d, eps=config.rms_norm_eps)
        self.embed_norm = RMSNorm(d, eps=config.rms_norm_eps)
        self.gate_up_proj = MergedColumnParallelLinear(
            2 * d,
            [intermediate, intermediate],
            bias=False,
            prefix=f"{prefix}.gate_up_proj",
            quant_config=quant_config,
        )
        self.act = SiluAndMul()
        self.down_proj = RowParallelLinear(
            intermediate,
            d,
            bias=False,
            prefix=f"{prefix}.down_proj",
            quant_config=quant_config,
        )

    def forward(
        self, hidden: torch.Tensor, previous_embed: torch.Tensor
    ) -> torch.Tensor:
        combined = torch.cat(
            (self.hidden_norm(hidden), self.embed_norm(previous_embed)), -1
        )
        gate_up, _ = self.gate_up_proj(combined)
        delta, _ = self.down_proj(self.act(gate_up))
        return hidden + delta


class HYV4DFlyFeedForward(nn.Module):
    """Dense SwiGLU whose width is read from the checkpoint (#68296 dfly.py:141-171)."""

    def __init__(self, hidden_size, hidden_act, quant_config, prefix):
        super().__init__()
        self._kwargs: dict[str, Any] = dict(
            hidden_size=hidden_size,
            hidden_act=hidden_act,
            quant_config=quant_config,
            prefix=prefix,
        )
        self.register_buffer("_placement", torch.empty(0), persistent=False)
        self.width: int | None = None

    def materialize(self, intermediate_size: int) -> None:
        with (
            torch.device(self._placement.device),
            set_default_torch_dtype(self._placement.dtype),
        ):
            mlp = HYV4FeedForward(intermediate_size=intermediate_size, **self._kwargs)
        self.gate_up_proj = mlp.gate_up_proj
        self.down_proj = mlp.down_proj
        self.act_fn = mlp.act_fn
        self.width = intermediate_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        out, _ = self.down_proj(self.act_fn(gate_up))
        return out


class HYV4DFlyDecoderLayer(nn.Module):
    def __init__(self, vllm_config, config, layer_idx, topk_indices_buffer, prefix):
        super().__init__()
        self.self_attn = HYV4FlashAttention(
            vllm_config, config, layer_idx, topk_indices_buffer, f"{prefix}.self_attn"
        )
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = HYV4DFlyFeedForward(
            config.hidden_size,
            config.hidden_act,
            vllm_config.quant_config,
            f"{prefix}.mlp",
        )

    def forward(self, positions, hidden_states, residual):
        if residual is not None:
            hidden_states = hidden_states + residual
        residual = hidden_states
        hidden_states = self.self_attn(positions, self.input_layernorm(hidden_states))
        hidden_states = hidden_states + residual
        residual = hidden_states
        hidden_states = self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, residual


class HYV4DFlyLayers(nn.Module):
    """``model.dfly_layers``: numbered layers plus fusion, correction, final norm."""

    def __init__(self, vllm_config, config, topk_indices, prefix):
        super().__init__()
        quant_config = vllm_config.quant_config
        self.num_layers = config.num_hidden_layers
        for i in range(self.num_layers):
            self.add_module(
                str(i),
                HYV4DFlyDecoderLayer(
                    vllm_config, config, i, topk_indices[i], f"{prefix}.{i}"
                ),
            )
        self.mtp_fusion = HYV4DFlyFusion(config, f"{prefix}.mtp_fusion", quant_config)
        self.hidden_correction = HYV4DFlyHiddenCorrection(
            config, f"{prefix}.hidden_correction", quant_config
        )
        self.final_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __len__(self) -> int:
        return self.num_layers

    def __getitem__(self, index: int) -> HYV4DFlyDecoderLayer:
        if not 0 <= index < self.num_layers:
            raise IndexError(index)
        return getattr(self, str(index))

    def __iter__(self):
        return (self[i] for i in range(self.num_layers))


class HYV4DFlyModel(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str):
        super().__init__()
        spec = vllm_config.speculative_config
        assert spec is not None
        config = deepcopy(vllm_config.model_config.hf_config)
        if config.model_type != DFLY_MODEL_TYPE:
            raise ValueError("Expected a derived hy_v4_dfly configuration")
        self.config = config
        self._vllm_config = vllm_config
        self.declared_ffn_width = getattr(config, "dfly_intermediate_size", None)
        self.declared_post_proj_norm = getattr(config, "dfly_attn_post_proj_norm", None)
        dummy = vllm_config.load_config.load_format == "dummy"
        self.query_width = spec.num_speculative_tokens + 1
        self.local_window = config.context_local_window or 0
        # The draft attends to target history + window + its own block.
        config.index_topk = dfly_topk_width(
            config.target_index_topk, self.local_window, self.query_width
        )
        config.dfly_external_topk = True
        if self.declared_post_proj_norm is False:
            config.sandwich_norm = False
        self.quant_config = vllm_config.quant_config
        self.embed_tokens: VocabParallelEmbedding | None = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size, prefix=f"{prefix}.embed_tokens"
        )
        # Set to False so the PP path loads the target table for this stage
        # (spec_decode/dflash/utils._maybe_load_target_embed_on_pp).
        self.loaded_embed_tokens = False
        # Live rows only (max_num_seqs x (K + 1))
        max_rows = dfly_topk_rows(vllm_config.scheduler_config, self.query_width)
        self.register_buffer(
            "topk_indices",
            torch.full(
                (config.num_hidden_layers, max_rows, config.index_topk),
                -1,
                dtype=torch.int32,
            ),
            persistent=False,
        )
        self.dfly_layers = HYV4DFlyLayers(
            vllm_config, config, self.topk_indices, f"{prefix}.dfly_layers"
        )
        if dummy or self.declared_ffn_width is not None:
            self.materialize_ffn(
                self.declared_ffn_width
                if self.declared_ffn_width is not None
                else config.intermediate_size
            )

    @property
    def layers(self) -> HYV4DFlyLayers:
        return self.dfly_layers

    @property
    def ffn_width(self) -> int | None:
        return self.dfly_layers[0].mlp.width if len(self.dfly_layers) else None

    def materialize_ffn(self, width: int) -> None:
        if width < 1 or width % get_tensor_model_parallel_world_size():
            raise ValueError("DFly FFN width must be positive and divisible by TP")
        if self.declared_ffn_width not in (None, width):
            raise ValueError("dfly_intermediate_size disagrees with the checkpoint")
        if self.ffn_width is not None:
            if self.ffn_width != width:
                raise ValueError("Inconsistent DFly FFN widths in checkpoint")
            return
        with set_current_vllm_config(self._vllm_config):
            for layer in self.dfly_layers:
                layer.mlp.materialize(width)
        self.config.intermediate_size = width

    def resolve_post_proj_norm(self, seen: set[str]) -> set[str]:
        """Drop the post-projection norms the checkpoint lacks (#68296 :281-295)."""
        names = {
            f"model.dfly_layers.{i}.self_attn.post_proj_norm.weight"
            for i in range(len(self.dfly_layers))
        }
        loaded = names & seen
        if loaded and loaded != names:
            raise ValueError("Inconsistent DFly attention post-projection norms")
        if not loaded and self.declared_post_proj_norm is not True:
            for layer in self.dfly_layers:
                layer.self_attn.post_proj_norm = None
            return names
        return set()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            assert self.embed_tokens is not None, "DFly embedding is not bound"
            inputs_embeds = self.embed_tokens(input_ids)
        hidden, residual = inputs_embeds, None
        for layer in self.dfly_layers:
            hidden, residual = layer(positions, hidden, residual)
        return self.dfly_layers.final_norm(hidden + residual)


def validate_dfly_quantization(
    model: nn.Module, quant_config: Any
) -> dict[str, int | None]:
    """Fail closed unless every quantised draft linear is in a ``dfly_*`` group.

    Returns ``{layer prefix: bits or None}`` for the draft linears. Under
    compressed-tensors a draft linear matching a target group (the FP8
    attention group or the expert group) would load an FP8/INT4 scheme over
    the pack's INT8 weights; the draft's module names prevent that and this
    check proves it on the built model.
    """
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
        CompressedTensorsConfig,
    )

    bits: dict[str, int | None] = {}
    raw = getattr(quant_config, "config", None)
    is_ct = isinstance(quant_config, CompressedTensorsConfig)
    for name, module in model.named_modules():
        if not isinstance(module, LinearBase) or ".dfly_layers." not in f".{name}":
            continue
        prefix = name
        if not is_ct:
            bits[prefix] = None
            continue
        groups = dfly_quant_groups(raw, prefix)
        stray = sorted(g for g in groups if not g.startswith("dfly_"))
        if stray:
            raise ValueError(
                f"DFly draft linear {prefix} matches target groups {stray}"
            )
        if groups:
            weights = raw["config_groups"][next(iter(groups))]["weights"]
            kind = (weights.get("type"), weights.get("num_bits"))
            if kind not in (("int", 4), ("int", 8), ("float", 8)):
                raise ValueError(
                    f"DFly draft linear {prefix} has unsupported scheme {kind}"
                )
            if weights.get("type") == "int" and weights.get("group_size") != 128:
                raise ValueError(
                    f"DFly draft linear {prefix} needs group-128 INT weights"
                )
        bits[prefix] = dfly_group_bits(raw, prefix)
    return bits


class HYV4DFlyForCausalLM(nn.Module):
    """``HYV4DFlyModel`` architecture: the DFly2 draft of a Hy4 Flash checkpoint."""

    checkpoint_prefix = DFLY_CHECKPOINT_PREFIX
    fall_back_to_pt_during_load = False
    allow_patterns_overrides = ["*.safetensors"]
    has_own_lm_head = False
    draft_id_to_target_id = None
    packed_modules_mapping = dict(_FUSED)
    _weight_mapper = WeightsMapper(
        orig_to_new_stacked={
            ".gate_proj.": (".gate_up_proj.", 0),
            ".up_proj.": (".gate_up_proj.", 1),
        }
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        if prefix:
            raise ValueError("The DFly draft keeps checkpoint module paths: prefix ''")
        self.model = HYV4DFlyModel(vllm_config, "model")
        self.config = self.model.config
        self.quant_config = vllm_config.quant_config
        self.quant_bits = validate_dfly_quantization(self, self.quant_config)
        self.has_own_embed_tokens = False
        self.lm_head: ParallelLMHead | None = None
        self.logits_processor = LogitsProcessor(self.config.vocab_size)
        self.logits_processor.use_all_gather = True
        self.loaded_parts: dict[str, set] = {}

    @classmethod
    def is_unused_checkpoint_weight(cls, name: str) -> bool:
        return checkpoint_to_local(name) is None

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        assert self.model.embed_tokens is not None, "DFly embedding is not bound"
        return self.model.embed_tokens(input_ids)

    def forward(self, input_ids, positions, inputs_embeds=None) -> torch.Tensor:
        return self.model(input_ids, positions, inputs_embeds)

    def apply_hidden_correction(self, hidden, previous_ids) -> torch.Tensor:
        return self.model.dfly_layers.hidden_correction(
            hidden, self.embed_input_ids(previous_ids)
        )

    def fuse(self, anchor_embeds, target_hidden) -> torch.Tensor:
        return self.model.dfly_layers.mtp_fusion(anchor_embeds, target_hidden)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.lm_head is None:
            raise RuntimeError(
                "DFly requires the target LM head before computing logits"
            )
        return self.logits_processor(self.lm_head, hidden_states)

    def get_draft_kv_cache_layer_names(self) -> list[str]:
        return [layer.self_attn.mla_attn.layer_name for layer in self.model.dfly_layers]

    # ------------------------------------------------------------------ loading
    def _ffn_bits(self, layer: int, proj: str) -> int | None:
        # From the config groups by name: the FFN modules may not exist yet.
        raw = getattr(self.quant_config, "config", None)
        return dfly_group_bits(raw, f"model.dfly_layers.{layer}.mlp.{proj}")

    def _record(self, name: str, part) -> None:
        parts = self.loaded_parts.setdefault(name, set())
        if part in parts or None in parts or (part is None and parts):
            raise ValueError(f"Duplicate DFly weight or shard: {name}")
        parts.add(part)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        self.loaded_parts.clear()
        hidden = self.config.hidden_size
        seen: set[str] = set()
        pending: list[tuple[str, torch.Tensor]] = []
        loaded: set[str] = set()

        def put(name: str, tensor: torch.Tensor) -> None:
            params = dict(self.named_parameters())
            for mapped, value in self._weight_mapper.apply([(name, tensor)]):
                if mapped not in params:
                    raise ValueError(
                        f"Unexpected DFly weight {name!r}, shape={tuple(tensor.shape)}"
                    )
                if value.is_floating_point() and value.dtype not in (
                    torch.bfloat16,
                    torch.float32,
                    torch.float8_e4m3fn,
                ):
                    raise ValueError(f"DFly weight {name} has dtype {value.dtype}")
                param = params[mapped]
                shard = getattr(value, "shard_id", None)
                self._record(mapped, shard)
                loader = getattr(param, "weight_loader", default_weight_loader)
                if shard is None:
                    loader(param, value)
                else:
                    loader(param, value, shard)
                loaded.add(mapped)

        for name, tensor in weights:
            local = checkpoint_to_local(name)
            if local is None:
                continue
            if local in seen:
                raise ValueError(f"Duplicate DFly weight {local!r}")
            seen.add(local)
            rest = local[len(DFLY_CHECKPOINT_PREFIX) :]
            if rest == "embed_tokens.weight":
                # A draft-owned table (the Hy4 Flash checkpoint has none).
                local = "model.embed_tokens.weight"
            index, _, tail = rest.partition(".")
            if index.isdigit() and tail.startswith("mlp."):
                if not 0 <= int(index) < len(self.model.dfly_layers):
                    raise ValueError(f"Unexpected DFly weight {local!r}")
                leaf = tail.removeprefix("mlp.")
                proj = leaf.partition(".")[0]
                width = probe_ffn_width(
                    leaf, tensor, hidden, self._ffn_bits(int(index), proj)
                )
                if width is not None:
                    self.model.materialize_ffn(width)
                if self.model.ffn_width is None:
                    pending.append((local, tensor))
                    continue
                for item in pending:
                    put(*item)
                pending.clear()
            put(local, tensor)
        if self.model.ffn_width is None:
            raise ValueError(
                "Missing required DFly FFN weights (no width-bearing tensor)"
            )
        optional = self.model.resolve_post_proj_norm(seen)
        required = {
            name
            for name, _ in self.named_parameters()
            if not (".mla_attn." in name and name.rsplit(".", 1)[-1] in _CACHE_DEFAULTS)
            and name != "model.embed_tokens.weight"
        }
        missing = required - loaded - optional
        if missing:
            raise ValueError(f"Missing required DFly weights: {sorted(missing)}")
        self._verify_parts()
        # The FFN linears exist only now: check their groups too.
        self.quant_bits = validate_dfly_quantization(self, self.quant_config)
        self.has_own_embed_tokens = "model.dfly_layers.embed_tokens.weight" in seen
        if not self.has_own_embed_tokens:
            # The speculator binds the target's table (#68296 dfly.py:489-491).
            self.model.embed_tokens = None
        return loaded

    def _verify_parts(self) -> None:
        for name, parts in self.loaded_parts.items():
            if None in parts:
                continue
            module = self.get_submodule(name.rpartition(".")[0])
            expected = set(range(len(getattr(module, "output_sizes", ()))))
            if parts != expected:
                raise ValueError(f"Incomplete DFly checkpoint shards: {name}")


def run_dfly_query(
    model: "HYV4DFlyForCausalLM",
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    target_hidden: torch.Tensor,
    active: torch.Tensor,
    draft_tokens: torch.Tensor,
    sample_fn,
) -> torch.Tensor:
    """One DFly query block (#68296 dfly_speculator.py:76-98).

    ``input_ids``/``positions``: [batch * W]; ``target_hidden``: [batch, d];
    ``active``: [batch]. Writes ``draft_tokens[:batch, :W-1]`` (-1 for inactive
    rows) and returns that view.
    """
    batch = active.shape[0]
    width = model.model.query_width
    embeddings = model.embed_input_ids(input_ids)
    embeddings[::width] = model.fuse(embeddings[::width], target_hidden)
    hidden = model(input_ids=input_ids, positions=positions, inputs_embeds=embeddings)
    hidden = hidden.view(batch, width, -1)
    previous = input_ids.view(batch, width)[:, 0].long()
    for step in range(width - 1):
        corrected = model.apply_hidden_correction(hidden[:, step + 1], previous)
        previous = sample_fn(step, corrected)
        draft_tokens[:batch, step] = torch.where(active, previous, -1)
    return draft_tokens[:batch]
