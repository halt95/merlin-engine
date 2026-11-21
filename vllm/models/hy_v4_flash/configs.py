# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modified by the Merlin project; see NOTICE.
"""Adapted from #68296 vllm/models/hy_v4_flash/configs.py:1-366."""

from copy import deepcopy
from typing import Any

from huggingface_hub.dataclasses import strict

from vllm.transformers_utils.configs.hy_v4 import HYV4Config

# Transformers >= 5.18 (the pin is 5.20) remaps "deepseek_sparse_attention" to
# "indexed_attention" at the end of PreTrainedConfig.__post_init__, so the
# canonical spelling must be the one the base class leaves in place.
SPARSE_ATTENTION = "indexed_attention"
SLIDING_ATTENTION = "sliding_attention"
_SPARSE_ALIASES = {
    "sparse",
    "sparse_attention",
    "deepseek_sparse_attention",
    SPARSE_ATTENTION,
}
# M-RoPE keys only the multimodal wrapper consumes. All three axes are equal for
# text positions, so dropping them is exact for the text architecture.
_MROPE_KEYS = ("mrope_section", "mrope_interleaved")
_BASE_ROPE_DEFAULTS = ("rope_theta", "rope_type", "partial_rotary_factor")


def quantization_group_sizes(quant: Any) -> set[int]:
    """Weight group/block sizes along the expert intermediate (TP) dimension."""
    if quant is None:
        return set()
    if not isinstance(quant, dict):
        quant = quant.to_dict() if hasattr(quant, "to_dict") else vars(quant)
    sizes = set()
    block = quant.get("weight_block_size")
    if block:
        sizes.update(int(size) for size in block)
    for group in (quant.get("config_groups") or {}).values():
        weights = (group or {}).get("weights") or {}
        size = weights.get("group_size")
        if size is not None and size > 0:
            sizes.add(int(size))
    return sizes


@strict
class HYV4FlashConfig(HYV4Config):
    model_type = "hy_v4_flash"
    base_config_key = "text_config"

    hidden_size: int = 3072
    intermediate_size: int = 13824
    moe_intermediate_size: int = 1536
    num_hidden_layers: int = 42
    n_routed_experts: int = 512
    num_experts_per_tok: int = 16
    q_lora_rank: int = 1024
    max_position_embeddings: int = 1048576
    routed_expert_hidden_size: int | None = 1536
    sandwich_norm: bool = True
    dsa_skip_attn_post_norm: bool = False
    swa_num_attention_heads: int | None = 64
    swa_qk_nope_head_dim: int | None = None
    swa_qk_rope_head_dim: int | None = None
    swa_v_head_dim: int | None = None
    sliding_window: int = 512
    window_size: list[int] | None = None
    use_compressed_indexer: bool = True
    compressed_indexer_compress_ratio: int = 4
    compressed_indexer_num_attention_heads: int = 16
    compressed_indexer_head_dim: int = 128
    compressed_indexer_group_topk: int = 512
    compressed_indexer_candidate_topk: int = 2048
    compressed_indexer_reuse_fine_indexer: bool = False
    compressed_indexer_skip_fine_select: bool = True
    compressed_indexer_key_pooling: str = "gated"
    engram_layer_ids: list[int] | None = None
    engram_max_ngram_size: int = 3
    engram_n_embed_per_ngram: int = 768
    engram_n_head_per_ngram: int = 8
    engram_pad_id: int = 120002
    engram_seed: int = 0
    engram_fp8_embedding: bool = False
    engram_layer_head_vocab_sizes: dict[str | int, list[list[int]]] | None = None
    vision_config: dict | None = None

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any], **kwargs: Any):
        values = deepcopy(config_dict)
        text = values.pop("text_config", None)
        if isinstance(text, dict):
            outer = values
            values = deepcopy(text)
            shared = ("architectures", "vision_config", "quantization_config")
            for name, value in outer.items():
                # to_dict() of a text config carries these keys as None; an
                # outer value must still win over that placeholder.
                if (name.startswith("dfly_") or name in shared) and values.get(
                    name
                ) is None:
                    values[name] = value
        values["model_type"] = cls.model_type
        values.pop("engram_enable", None)
        return super().from_dict(values, **kwargs)

    def __post_init__(self, **kwargs: Any) -> None:
        kwargs.pop("engram_enable", None)
        if self.layer_types is None:
            self.layer_types = [
                SPARSE_ATTENTION if i == 0 or (i - 1) % 4 == 0 else SLIDING_ATTENTION
                for i in range(self.num_hidden_layers)
            ]
        self._normalize_layer_types()
        if self.indexer_types is None:
            self.indexer_types = ["full"] * self.num_hidden_layers
        for name in (
            "num_attention_heads",
            "qk_nope_head_dim",
            "qk_rope_head_dim",
            "v_head_dim",
        ):
            if getattr(self, f"swa_{name}") is None:
                setattr(self, f"swa_{name}", getattr(self, name))
        self._normalize_rope()
        super().__post_init__(**kwargs)
        # The base class may remap layer_types (transformers >= 5.18); fold
        # every spelling back to the canonical one before validating.
        self._normalize_layer_types()
        self._normalize_rope(after_base=True)
        self.validate_flash_config()

    def _normalize_layer_types(self) -> None:
        if self.layer_types is None:
            return
        self.layer_types = [
            SPARSE_ATTENTION if kind in _SPARSE_ALIASES else kind
            for kind in self.layer_types
        ]

    def _is_text_architecture(self) -> bool:
        return not any(
            arch.endswith("ForConditionalGeneration")
            for arch in self.architectures or ()
        )

    def _normalize_rope(self, *, after_base: bool = False) -> None:
        rope = deepcopy(self.rope_parameters)
        if not rope:
            return
        if after_base and any(isinstance(value, dict) for value in rope.values()):
            # Transformers releases that do not know the canonical layer type
            # treat the per-layer dict as flat and add a global default.
            for key in _BASE_ROPE_DEFAULTS:
                if not isinstance(rope.get(key, {}), dict):
                    rope.pop(key)
        if self._is_text_architecture():
            for params in (rope, *rope.values()):
                if isinstance(params, dict):
                    for key in _MROPE_KEYS:
                        params.pop(key, None)
        for alias in _SPARSE_ALIASES - {SPARSE_ATTENTION}:
            if alias in rope:
                value = rope.pop(alias)
                if SPARSE_ATTENTION in rope and rope[SPARSE_ATTENTION] != value:
                    raise ValueError("Conflicting sparse attention RoPE parameters")
                rope[SPARSE_ATTENTION] = value
        if any(isinstance(value, dict) for value in rope.values()):
            if any(not isinstance(value, dict) for value in rope.values()):
                raise ValueError("Cannot mix flat and per-layer RoPE parameters")
            for kind in set(self.layer_types or ()):
                if kind not in rope:
                    raise ValueError(f"Missing rope_parameters for {kind}")
        self.rope_parameters = rope

    def rope_for_layer(self, layer_idx: int) -> dict[str, Any]:
        rope = self.rope_parameters or {}
        return deepcopy(rope.get(self.layer_types[layer_idx], rope))

    def validate_layer_type(self) -> None:
        self.validate_flash_config()

    def validate_flash_config(self) -> None:
        for name in (
            "hidden_size",
            "vocab_size",
            "num_hidden_layers",
            "intermediate_size",
            "moe_intermediate_size",
            "num_attention_heads",
            "swa_num_attention_heads",
            "q_lora_rank",
            "kv_lora_rank",
            "qk_rope_head_dim",
            "swa_qk_rope_head_dim",
            "v_head_dim",
            "swa_v_head_dim",
            "sliding_window",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name, choices in (
            ("layer_types", {SPARSE_ATTENTION, SLIDING_ATTENTION}),
            ("mlp_layer_types", {"dense", "sparse"}),
            ("indexer_types", {"full"}),
        ):
            values = getattr(self, name)
            if values is None or len(values) != self.num_hidden_layers:
                raise ValueError(f"{name} must have {self.num_hidden_layers} entries")
            if any(value not in choices for value in values):
                raise ValueError(f"Unsupported {name}: {values}")
        if self.window_size is not None and self.window_size != [
            self.sliding_window - 1,
            0,
        ]:
            raise ValueError("window_size must equal [sliding_window - 1, 0]")
        latent = self.routed_expert_hidden_size
        if latent is not None and (type(latent) is not int or latent < 1):
            raise ValueError("routed_expert_hidden_size must be positive")
        if not 1 <= self.num_experts_per_tok <= self.n_routed_experts:
            raise ValueError("num_experts_per_tok must be within the expert count")
        if (
            not self.use_compressed_indexer
            or not self.compressed_indexer_skip_fine_select
            or self.compressed_indexer_reuse_fine_indexer
            or self.compressed_indexer_key_pooling != "gated"
        ):
            raise ValueError("Flash requires independent gated coarse-only IndexPool")
        ratio = self.compressed_indexer_compress_ratio
        if type(ratio) is not int or ratio not in (2, 4, 8, 16):
            raise ValueError("compressed_indexer_compress_ratio must be 2/4/8/16")
        if (
            type(self.compressed_indexer_group_topk) is not int
            or self.compressed_indexer_group_topk < 2
        ):
            raise ValueError("compressed_indexer_group_topk must be at least 2")
        candidates = self.compressed_indexer_group_topk * ratio
        if (
            self.compressed_indexer_candidate_topk != candidates
            or self.index_topk != candidates
        ):
            raise ValueError(
                "IndexPool requires index_topk=candidate_topk=group_topk*ratio"
            )
        if self.compressed_indexer_head_dim != 128 or self.index_head_dim != 128:
            raise ValueError("IndexPool requires 128-dimensional keys")
        layers = self.engram_layer_ids or []
        if len(set(layers)) != len(layers) or any(
            type(i) is not int or not 1 <= i <= self.num_hidden_layers for i in layers
        ):
            raise ValueError("engram_layer_ids must be unique 1-based layer numbers")
        if not layers:
            return
        if not self.enable_ihc:
            raise ValueError("Flash Engram requires iHC")
        heads = self.engram_n_head_per_ngram
        if (
            type(heads) is not int
            or heads < 1
            or self.engram_n_embed_per_ngram % heads
            or self.engram_max_ngram_size < 2
        ):
            raise ValueError("Invalid Engram embedding/head dimensions")
        if not 0 <= self.engram_pad_id < self.vocab_size:
            raise ValueError("engram_pad_id is outside the vocabulary")
        groups = self.engram_layer_head_vocab_sizes or {}
        for layer in layers:
            sizes = groups.get(str(layer), groups.get(layer))
            if sizes is None or len(sizes) != self.engram_max_ngram_size - 1:
                raise ValueError(f"Missing Engram head vocabulary for layer {layer}")
            if any(
                len(group) != heads
                or any(type(size) is not int or size < 1 for size in group)
                for group in sizes
            ):
                raise ValueError(f"Invalid Engram head vocabulary for layer {layer}")
            if sum(size for group in sizes for size in group) > 2147483647:
                raise ValueError("Engram flattened bucket offsets must fit int32")

    def verify_parallel_config(self, parallel) -> None:
        # hy4-pp (PP runs on hy_v4's iHC path with
        # stage-local Engram: vllm/merlin/hy4flash_pp.py)
        if parallel.prefill_context_parallel_size != 1:
            raise ValueError("Flash prefill_context_parallel_size must be 1")
        if parallel.pipeline_parallel_size > self.num_hidden_layers:
            raise ValueError("Flash PP exceeds the number of decoder layers")
        if (
            parallel.data_parallel_size != 1
            or parallel.enable_eplb
            or parallel.enable_dbo
        ):
            raise ValueError("This Flash port requires DP=1 without EPLB or DBO")
        tp = parallel.tensor_parallel_size
        for name in (
            "num_attention_heads",
            "swa_num_attention_heads",
            "intermediate_size",
            "moe_intermediate_size",
        ):
            if getattr(self, name) % tp:
                raise ValueError(f"{name} must be divisible by TP")
        expert_shard = self.moe_intermediate_size // tp
        for group in quantization_group_sizes(
            getattr(self, "quantization_config", None)
        ):
            # TP8 gives 1536 // 8 = 192, which no g128 kernel (Marlin) or
            # 128-row FP8 block can tile; TP4 gives 384 and is fine.
            if expert_shard % group:
                raise ValueError(
                    f"moe_intermediate_size // TP = {expert_shard} is not a "
                    f"multiple of the checkpoint quantization group {group}"
                )
        if self.engram_layer_ids:
            heads = (self.engram_max_ngram_size - 1) * self.engram_n_head_per_ngram
            if tp > heads:
                raise ValueError("Engram requires one complete hash head per TP rank")

    def update(self, config_dict: dict[str, Any]) -> None:
        values = deepcopy(config_dict)
        values.pop("engram_enable", None)
        super().update(values)
        self._normalize_layer_types()
        self._normalize_rope()
        self.validate_flash_config()
