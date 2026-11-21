# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modified by the Merlin project; see NOTICE.
"""Adapted from #68296 nvidia/model.py:1-472 for Merlin's 0.33 interfaces."""

from collections.abc import Iterable

import regex as re
import torch
from torch import nn

from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.forward_context import get_forward_context

from vllm.merlin.hy4_kvshard import validate_hy4_dcp
from vllm.merlin.hy4flash_pp import (
    check_dfly_stage,
    local_engram_hash_rows,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    is_pp_missing_parameter,
)
from vllm.models.hy_v4.checkpoint import (
    validate_expert_shape,
    validate_hy4_checkpoint_quantization,
)
from vllm.models.hy_v4.nvidia.hc import HYV4HCLayer
from vllm.models.hy_v4.nvidia.model import (
    HYV4DecoderLayer,
    HYV4ForCausalLM,
    HYV4Model,
)
from vllm.models.hy_v4_flash.hashing import (
    FlashNgramHashState,
    batch_lookback_rows,
)
from vllm.models.hy_v4_flash.weights import (
    EXPERT_WEIGHT,
    HF_TO_VLLM_MAPPER,
    PACKED_MODULES_MAPPING,
    expert_pieces,
    expert_target,
    module_parameter_name,
    required_parameter_names,
    target_weight_name,
)
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors

from .attention import HYV4FlashAttention
from .engram import FlashEngram, dummy_table_dtype
from .moe import HYV4FlashFeedForward, HYV4FlashMoE


def validate_runtime(vllm_config):
    config = vllm_config.model_config.hf_text_config
    config.validate_flash_config()
    config.verify_parallel_config(vllm_config.parallel_config)
    # The quantization validator resolves a pack's
    # groups against the checkpoint's module names (safetensors headers).
    validate_hy4_checkpoint_quantization(
        getattr(config, "quantization_config", None), vllm_config.model_config
    )
    parallel = vllm_config.parallel_config
    validate_hy4_dcp(
        tp_size=parallel.tensor_parallel_size,
        dcp_size=parallel.decode_context_parallel_size,
        pcp_size=parallel.prefill_context_parallel_size,
        interleave=parallel.cp_kv_cache_interleave_size,
        block_size=vllm_config.cache_config.block_size or 64,
        ratio=config.compressed_indexer_compress_ratio,
        num_heads=config.num_attention_heads,
        swa_num_heads=config.swa_num_attention_heads,
    )
    if not current_platform.is_cuda():
        raise NotImplementedError("HYV4 Flash serving requires NVIDIA CUDA")
    if vllm_config.model_config.dtype != torch.bfloat16:
        raise ValueError("HYV4 Flash requires bfloat16 compute")
    if (
        vllm_config.lora_config is not None
        or vllm_config.kv_transfer_config is not None
    ):
        raise ValueError("This Flash port does not support LoRA or KV transfer")
    speculative = vllm_config.speculative_config
    # DFly2 is the one speculative method wired for Flash.
    if speculative is not None and speculative.method != "dfly":
        raise NotImplementedError(
            "HYV4 Flash supports speculative method 'dfly' only, "
            f"got {speculative.method!r}"
        )
    if speculative is not None:
        # The DFly2 source layers must be on the last stage.
        check_dfly_stage(vllm_config)
    if vllm_config.model_config.quantization not in (None, "fp8", "compressed-tensors"):
        raise ValueError("Flash supports BF16, FP8, or the compressed-tensors pack")
    if config.index_topk % 128 or config.index_topk < config.sliding_window:
        raise ValueError("index_topk must be 128-aligned and cover sliding_window")
    if config.engram_layer_ids:
        storage = vllm_config.engram_config
        if not storage.cpu_offload:
            raise ValueError(
                "Flash Engram requires --engram-config "
                '\'{"cpu_offload":true,"dp_shared_memory":false}\''
            )
        if storage.embedding_across_dp or storage.dp_shared_memory or storage.use_thp:
            raise ValueError("Flash currently uses private TP-local pinned host tables")


class HYV4FlashDecoderLayer(HYV4DecoderLayer):
    def __init__(
        self,
        config,
        vllm_config,
        cache_config=None,
        quant_config=None,
        prefix="",
        topk_indices_buffer=None,
    ):
        nn.Module.__init__(self)
        self.config = config
        self.hidden_size = config.hidden_size
        self.layer_idx = int(prefix.rsplit(".", 1)[1])
        self.enable_ihc = config.enable_ihc
        self.self_attn = HYV4FlashAttention(
            vllm_config,
            config,
            self.layer_idx,
            topk_indices_buffer,
            f"{prefix}.self_attn",
        )
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        dense = config.mlp_layer_types[self.layer_idx] == "dense"
        self.mlp = (HYV4FlashFeedForward if dense else HYV4FlashMoE)(
            config, quant_config, f"{prefix}.mlp"
        )
        self.block_type = "feedforward" if dense else "moe"
        self.hc_attn_layer = HYV4HCLayer(
            config, self.layer_idx, prefix=f"{prefix}.hc_attn_layer"
        )
        self.hc_mlp_layer = HYV4HCLayer(
            config, self.layer_idx, prefix=f"{prefix}.hc_mlp_layer"
        )

    # MERLIN_HY4_FUSE_IHC=1: the fused iHC boundaries
    # (vllm/merlin/hy4_ihc_fuse.py, bit-identical); anything they do not cover
    # takes the stock forward.
    def forward(self, positions, hidden_states, residual):
        from vllm.merlin import hy4_ihc_fuse

        if hy4_ihc_fuse.fuse_requested() and hidden_states.is_cuda:
            x3 = self.hc_attn_layer.prepare_input(hidden_states)
            if hy4_ihc_fuse.layer_supported(self, x3):
                return hy4_ihc_fuse.forward_layer(self, positions, x3)
        return super().forward(positions, hidden_states, residual)


class HYV4FlashModel(HYV4Model):
    decoder_layer_cls = HYV4FlashDecoderLayer

    def __init__(self, *, vllm_config, prefix=""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.loaded_parts = {}
        self.engram_hash = None
        # Stage-local Engram: (one-based layer ID,
        # checkpoint-global hash row) for this stage's Engram layers only. A
        # stage with none builds no hash state and hashes nothing.
        self.engram_rows = ()
        if self.config.engram_layer_ids:
            self.engram_rows = local_engram_hash_rows(
                self.config.engram_layer_ids,
                self.start_layer,
                self.end_layer,
                self.config.num_hidden_layers,
            )
        if self.engram_rows:
            dummy = vllm_config.load_config.load_format == "dummy"
            self.engram_hash = FlashNgramHashState(
                self.config, dummy=dummy, rows=[row for _, row in self.engram_rows]
            )
            # Dummy tables take the config's table
            # dtype, as in #68296 engram.py:367-371 (FP8 when engram_fp8_embedding).
            for layer_id, _ in self.engram_rows:
                layer = self.layers[layer_id - 1]
                layer.engram_layer = FlashEngram(
                    self.config,
                    self.engram_hash.layout,
                    layer_id,
                    tp_size=get_tensor_model_parallel_world_size(),
                    rank=get_tensor_model_parallel_rank(),
                    cpu_offload=True,
                    dummy_dtype=dummy_table_dtype(self.config) if dummy else None,
                )
        self.token_lookback_depth = (
            self.config.engram_max_ngram_size - 1 if self.engram_hash is not None else 0
        )
        # #68296 hy_v4/nvidia/model.py:268-271
        speculative = vllm_config.speculative_config
        self.dfly_enabled = speculative is not None and speculative.method == "dfly"

    def embed_input_ids(self, input_ids):
        # #68296 hy_v4/nvidia/model.py:389-392. A draft
        # the speculator marks -1 (inactive request) reaches the verify batch;
        # the sampler still sees -1 and rejects it, the embedding reads row 0.
        # The clamp also covers the host table (F.embedding on a UVA view has no
        # bounds check at TP 1; at TP > 1 the vocab mask already zeroes it).
        if self.dfly_enabled:
            input_ids = input_ids.clamp_min(0)
        return super().embed_input_ids(input_ids)

    @eager_break_during_capture
    def _prepare_hashes(self, input_ids, positions, lookback, layer_name, output):
        metadata = get_forward_context().attn_metadata
        metadata = metadata.get(layer_name) if isinstance(metadata, dict) else None
        if metadata is None or metadata.num_actual_tokens == 0:
            output.fill_(-1)
            return
        if lookback is None:
            raise ValueError("Flash Engram requires V2 lookback_token_ids")
        # Global 1-D positions, not metadata.seq_lens: under DCP the first
        # local layer's metadata may carry rank-local lengths.
        output.copy_(
            self.engram_hash(
                input_ids,
                # The model state hands the whole
                # persistent (max_num_reqs, depth) buffer, and a breakable graph's
                # replay calls this with the capture-time arguments but the live
                # metadata: take the rows of the live batch here, from that buffer.
                batch_lookback_rows(lookback, metadata.query_start_loc),
                metadata.query_start_loc,
                positions,
                metadata.num_actual_tokens,
            )
        )

    def forward(
        self,
        input_ids,
        positions,
        intermediate_tensors=None,
        inputs_embeds=None,
        lookback_token_ids=None,
    ):
        if self.engram_hash is None:
            return super().forward(
                input_ids, positions, intermediate_tensors, inputs_embeds
            )
        if input_ids is None:
            raise ValueError("Flash Engram requires original input token IDs")
        if get_pp_group().is_first_rank:
            hidden = (
                inputs_embeds
                if inputs_embeds is not None
                else self.embed_input_ids(input_ids)
            )
        else:
            if intermediate_tensors is None:
                raise ValueError("Missing Flash pipeline intermediate tensors")
            hidden = intermediate_tensors["hidden_states"]
        hashes = torch.empty(
            (
                len(self.engram_rows),
                input_ids.numel(),
                self.engram_hash.offsets.shape[1],
            ),
            dtype=torch.int32,
            device=input_ids.device,
        )
        self._prepare_hashes(
            input_ids,
            positions,
            lookback_token_ids,
            self.layers[self.start_layer].self_attn.layer_name,
            hashes,
        )
        by_layer = {
            layer_id - 1: slot for slot, (layer_id, _) in enumerate(self.engram_rows)
        }
        for index in range(self.start_layer, self.end_layer):
            layer = self.layers[index]
            if index in by_layer:
                hidden = layer.hc_attn_layer.prepare_input(hidden)
                hidden = layer.engram_layer(hidden, hashes[by_layer[index]])
            hidden, _ = layer(positions, hidden, None)
        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden.flatten(1)})
        return self.norm(self.hc_head(hidden))

    def _record_part(self, name, part):
        parts = self.loaded_parts.setdefault(name, set())
        if part in parts or None in parts or (part is None and parts):
            raise ValueError(f"Duplicate Flash checkpoint weight or shard: {name}")
        parts.add(part)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        parameters = dict(self.named_parameters())
        loaded = set()
        for name, value in weights:
            if is_pp_missing_parameter(name, self):
                continue
            engram = re.fullmatch(r"layers\.(\d+)\.engram_layer\.(.+)", name)
            if engram is not None:
                layer_number, leaf = engram.groups()
                layer = self.layers[int(layer_number)]
                if not hasattr(layer, "engram_layer"):
                    raise ValueError(f"Undeclared Engram weight: {name}")
                if leaf == "lookup_table":
                    self._record_part(name, None)
                    self.engram_hash.load_lookup(value)
                    continue
                if leaf.startswith("multi_head_embedding.embedding."):
                    tail = leaf.removeprefix("multi_head_embedding.embedding.")
                    if tail not in ("weight", "weight_scale_inv"):
                        raise ValueError(f"Unknown Engram table weight: {name}")
                    self._record_part(name, None)
                    layer.engram_layer.multi_head_embedding.load_weight(tail, value)
                    loaded.add(name)
                    continue
            expert = EXPERT_WEIGHT.fullmatch(name)
            if expert is not None:
                prefix, expert_id, projection, suffix, target = expert_target(expert)
                leaf = "w2_weight" if projection == "down_proj" else "w13_weight"
                if suffix == "_shape":
                    packed = target.removesuffix("_shape") + "_packed"
                    wrapped = packed.replace(
                        ".routed_experts.", ".routed_experts.base_layer."
                    )
                    if packed not in parameters and wrapped not in parameters:
                        raise ValueError(
                            f"Shape metadata without packed weight: {name}"
                        )
                    validate_expert_shape(
                        self.config, projection, value, expert_id=expert_id
                    )
                    self._record_part(name, None)
                    # The fused ``w13/w2_weight_shape`` parameter is metadata
                    # (compressed-tensors registers it, nothing reads it after
                    # loading); the per-expert shapes are validated above, so
                    # the fused parameter counts as loaded.
                    for shape_param in (
                        target,
                        target.replace(
                            ".routed_experts.", ".routed_experts.base_layer."
                        ),
                    ):
                        if shape_param in parameters:
                            loaded.add(shape_param)
                    continue
                if target not in parameters:
                    alternative = f"{prefix}.routed_experts.base_layer.{leaf}{suffix}"
                    if alternative not in parameters:
                        raise ValueError(f"Unknown Flash expert weight: {name}")
                    target = alternative
                parameter = parameters[target]
                for index, shard, piece in expert_pieces(
                    name, projection, value, expert_id, self.config.num_experts
                ):
                    success = parameter.weight_loader(
                        parameter,
                        piece,
                        target,
                        shard_id=shard,
                        expert_id=index,
                        return_success=True,
                    )
                    if success:
                        self._record_part(target, (index, shard))
                        loaded.add(target)
                continue
            name = module_parameter_name(name)
            if ".engram_layer.lookup_per_head_norms." in name:
                prefix, suffix = name.split(".lookup_per_head_norms.")
                head, leaf = suffix.split(".", 1)
                embedding = self.get_submodule(prefix).multi_head_embedding
                if not head.isdigit() or not 0 <= int(head) < embedding.n_hash_cols:
                    raise ValueError(f"Invalid Engram hash-head weight: {name}")
                if not embedding.head_start <= int(head) < embedding.head_end:
                    continue
                name = (
                    f"{prefix}.lookup_per_head_norms."
                    f"{int(head) - embedding.head_start}.{leaf}"
                )
            if name not in parameters:
                raise ValueError(f"Unknown Flash checkpoint weight: {name}")
            parameter = parameters[name]
            shard = getattr(value, "shard_id", None)
            self._record_part(name, shard)
            loader = getattr(parameter, "weight_loader", default_weight_loader)
            if shard is None:
                loader(parameter, value)
            else:
                loader(parameter, value, shard)
            loaded.add(name)
        return loaded

    def verify_loaded_parts(self):
        for layer_id in self.config.engram_layer_ids or ():
            layer = self.layers[layer_id - 1]
            if isinstance(layer, PPMissingLayer):
                continue
            prefix = f"layers.{layer_id - 1}.engram_layer."
            required = {
                prefix + "lookup_table",
                prefix + "multi_head_embedding.embedding.weight",
            }
            if layer.engram_layer.multi_head_embedding.fp8:
                required.add(prefix + "multi_head_embedding.embedding.weight_scale_inv")
            missing = required - self.loaded_parts.keys()
            if missing:
                raise ValueError(
                    f"Missing Engram checkpoint weights: {sorted(missing)}"
                )
        for name, parts in self.loaded_parts.items():
            if None in parts:
                continue
            module_name, _, leaf = name.rpartition(".")
            module = self.get_submodule(module_name)
            if leaf.startswith(("w13_weight", "w2_weight")):
                mapping = module.expert_map_manager.expert_map
                local = (
                    range(self.config.num_experts)
                    if mapping is None
                    else (mapping >= 0).nonzero().flatten().tolist()
                )
                shards = ("w1", "w3") if leaf.startswith("w13_") else ("w2",)
                expected = {(index, shard) for index in local for shard in shards}
            else:
                expected = set(range(len(module.output_sizes)))
            if parts != expected:
                raise ValueError(f"Incomplete Flash checkpoint shards: {name}")


class HYV4FlashForCausalLM(HYV4ForCausalLM):
    model_cls = HYV4FlashModel
    supports_lora = False
    supports_pp = True
    requires_raw_input_tokens = True
    packed_modules_mapping = PACKED_MODULES_MAPPING
    hf_to_vllm_mapper = HF_TO_VLLM_MAPPER

    def __init__(self, *, vllm_config, prefix=""):
        validate_runtime(vllm_config)
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.token_lookback_depth = self.model.token_lookback_depth

    @staticmethod
    def get_model_state_cls():
        from vllm.models.deepseek_v41.nvidia.model_state import DeepseekV41ModelState

        return DeepseekV41ModelState

    def forward(
        self,
        input_ids,
        positions,
        intermediate_tensors=None,
        inputs_embeds=None,
        lookback_token_ids=None,
    ):
        return self.model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
            lookback_token_ids,
        )

    def _target_weights(self, weights):
        seen = set()
        for name, value in weights:
            if name in seen:
                raise ValueError(f"Duplicate Flash checkpoint weight: {name}")
            seen.add(name)
            target = target_weight_name(self.config, name)
            if target is not None:
                yield target, value

    def load_weights(self, weights):
        self.model.loaded_parts.clear()
        loaded = AutoWeightsLoader(self).load_weights(
            self._target_weights(weights), mapper=self.hf_to_vllm_mapper
        )
        required = required_parameter_names(self.named_parameters())
        missing = required - loaded
        if missing:
            raise ValueError(f"Missing Flash checkpoint weights: {sorted(missing)}")
        self.model.verify_loaded_parts()
        return loaded
