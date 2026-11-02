# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modified by the Merlin project; see NOTICE.
"""Inference-only Qwen4Exp MTP (Multi-Token Predictor) model.

The MTP draft model reuses the Qwen4Exp backbone (PLE/HC/MoE) but:
  - drops all multi-modal handling (text-only),
  - forces PLE off while keeping the main model's HC stream count,
  - fuses the backbone hidden and the new-token embedding via
    ``residual_linear_shared`` (fc_embedding + shared fc_hidden) instead of
    the ``Linear(2H, H)`` + repeat used by other MTP variants,
  - emits TWO hidden streams per step (scheme A): a single stream [T, H]
    (final-mixer collapsed, fed to the LM head) and a pre-final-mixer
    multi stream [T, hc_count*H] (fed to the next draft step).
"""

from collections.abc import Iterable

import regex as re
import torch
from torch import nn
from transformers import Qwen4ExpTextConfig

from vllm import envs
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig, replace, set_current_vllm_config
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.utils import (
    is_model_fused_shared_expert_compatible,
)
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    PinnedHostVocabParallelEmbedding,
    VocabParallelEmbedding,
    should_use_host_embed_table,
)
from vllm.model_executor.model_loader.utils import configure_quant_config
from vllm.model_executor.models.interfaces import (
    LocalArgmaxMixin,
    SupportsHCSP,
    SupportsPP,
    SupportsReplaySSM,
)
from vllm.model_executor.models.qwen3_5 import Qwen3_5Model
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    WeightsMapper,
    get_draft_quant_config,
    make_empty_intermediate_tensors_factory,
    maybe_fuse_shared_experts,
    maybe_prefix,
)
from vllm.models.common.ops.sequence_parallel import (
    sp_all_gather,
    sp_padding_mask,
    sp_shard,
)
from vllm.sequence import IntermediateTensors
from vllm.v1.e2_guards import (
    assert_compiled_submodule,
    assert_graph_profile_is_executable,
    assert_ple_mode,
)

from .draft_head import (
    ENV_DRAFT_VOCAB_FILE,
    check_draft_vocab_agreement,
    d2t_offsets,
    draft_vocab_file,
    draft_vocab_fingerprint,
    load_draft_vocab_ids,
    scatter_to_full,
    select_head_rows,
)
from .hyperconnection import GatedResidual, HyperConnectionConfig
from .low_latency_gemm import enable_qwen4_exp_low_latency_gemm
from .model import (
    _EXTRA_WEIGHTS_MAPPER,
    _QWEN4_EXP_IGNORED_MISSING_SUFFIXES,
    Qwen4ExpDecoderLayer,
    Qwen4ExpMixtureOfExperts,
    Qwen4ExpSparseMoeBlock,
    is_hc_sequence_parallel_enabled,
)

from .mtp_kv_scales import maybe_load_mtp_qsa_static_kv_scales

logger = init_logger(__name__)


def _remap_ignored_layers(
    ignored_layers: list[str],
    mtp_start_layer_idx: int,
) -> list[str]:
    remapped: list[str] = []
    for name in ignored_layers:
        if name.startswith("mtp."):
            new_name = re.sub(
                r"(?<=\.layers\.)\d+",
                lambda m: str(mtp_start_layer_idx + int(m.group(0))),
                name,
            )
            remapped.append(new_name)
        else:
            remapped.append(name)
    return remapped


def _remap_quantized_layers(
    quantized_layers: dict[str, dict],
    mtp_start_layer_idx: int,
) -> dict[str, dict]:
    """Map checkpoint MTP layer indices to standalone draft indices."""
    return {
        _remap_ignored_layers([name], mtp_start_layer_idx)[0]: layer_info
        for name, layer_info in quantized_layers.items()
    }


def _remap_mtp_weight_name(name: str) -> str | None:
    """Map Qwen4Exp checkpoint paths into the standalone draft model."""
    for checkpoint_prefix in (
        "model.language_model.",
        "language_model.",
    ):
        if name.startswith(checkpoint_prefix):
            name = name.removeprefix(checkpoint_prefix)
            break

    if name.startswith("embed_tokens."):
        name = f"model.{name}"
    if name.startswith("model.mtp."):
        name = name.removeprefix("model.")
    if name.startswith("mtp.shared_head.head."):
        return name.replace("mtp.shared_head.head.", "lm_head.", 1)
    if name.startswith("model.shared_head.head."):
        return name.replace("model.shared_head.head.", "lm_head.", 1)
    if name.startswith("shared_head.head."):
        return name.replace("shared_head.head.", "lm_head.", 1)
    if name.startswith("model.lm_head."):
        return name.removeprefix("model.")
    if name.startswith("mtp."):
        return name.replace("mtp.", "model.", 1)
    if name.startswith("model.embed_tokens.") or name.startswith("lm_head."):
        return name
    return None


def _make_draft_vllm_config(
    vllm_config: VllmConfig,
    mtp_start_layer_idx: int,
) -> VllmConfig:
    """Ensure that the draft model config is set in the vLLM config."""
    speculative_config = vllm_config.speculative_config
    if speculative_config is None or speculative_config.draft_model_config is None:
        raise ValueError("speculative_config.draft_model_config must be set")

    draft_quant_config = get_draft_quant_config(vllm_config)

    # inject packed and ignored modules to the quantization config of draft model
    if draft_quant_config is not None:
        configure_quant_config(draft_quant_config, Qwen4ExpMTP)
        ignored_layers = getattr(draft_quant_config, "ignored_layers", None)
        if ignored_layers:
            setattr(  # noqa: B010
                draft_quant_config,
                "ignored_layers",
                _remap_ignored_layers(ignored_layers, mtp_start_layer_idx),
            )
        exclude_modules = getattr(draft_quant_config, "exclude_modules", None)
        if exclude_modules:
            setattr(  # noqa: B010
                draft_quant_config,
                "exclude_modules",
                _remap_ignored_layers(exclude_modules, mtp_start_layer_idx),
            )
        quantized_layers = getattr(draft_quant_config, "quantized_layers", None)
        if quantized_layers:
            setattr(  # noqa: B010
                draft_quant_config,
                "quantized_layers",
                _remap_quantized_layers(quantized_layers, mtp_start_layer_idx),
            )

    draft_vllm_config = replace(
        vllm_config,
        model_config=speculative_config.draft_model_config,
    )
    # VllmConfig post-init derives the target quant config, so restore the
    # independently resolved draft quant config after replacement.
    draft_vllm_config.quant_config = draft_quant_config
    return draft_vllm_config


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
        "hidden_states": 0,
    }
)
class Qwen4ExpMultiTokenPredictor(nn.Module):
    hf_to_vllm_mapper = Qwen3_5Model.hf_to_vllm_mapper | _EXTRA_WEIGHTS_MAPPER

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()

        model_config = vllm_config.model_config
        config: Qwen4ExpTextConfig = model_config.hf_text_config

        self.config = config
        # MoE SP also enables HC SP, while HC SP can be enabled independently.
        self.use_hc_sequence_parallel = is_hc_sequence_parallel_enabled(vllm_config)
        self.use_sequence_parallel = (
            vllm_config.parallel_config.use_sequence_parallel_moe
        )
        self.vocab_size = config.vocab_size

        self.mtp_start_layer_idx = config.num_hidden_layers
        self.num_mtp_layers = getattr(config, "mtp_num_hidden_layers", 1)

        self.hidden_size = config.hidden_size
        self.hc_count = config.hc_count

        self.uses_host_embed_table = should_use_host_embed_table()
        if self.uses_host_embed_table and getattr(
            config, "tie_word_embeddings", False
        ):
            raise ValueError(
                "VLLM_HOST_EMBED_TABLE=1 is incompatible with "
                "tie_word_embeddings=True"
            )
        embedding_cls = (
            PinnedHostVocabParallelEmbedding
            if self.uses_host_embed_table
            else VocabParallelEmbedding
        )
        self.embed_tokens = embedding_cls(self.vocab_size, self.hidden_size)
        draft_vllm_config = _make_draft_vllm_config(
            vllm_config,
            self.mtp_start_layer_idx,
        )
        with set_current_vllm_config(draft_vllm_config, prefix=prefix):
            # residual_linear_shared fusion: fc_embedding projects the token
            # embedding, fc_hidden (shared across HC branches) projects the
            # backbone hidden; the embedding is added as a residual to every
            # branch (see mtp_residual_linear_shared.md).
            self.fc_embedding = ColumnParallelLinear(
                self.hidden_size,
                self.hidden_size,
                gather_output=True,
                bias=False,
                return_bias=False,
                quant_config=draft_vllm_config.quant_config,
                prefix=f"{prefix}.fc_embedding",
            )
            self.fc_hidden = ColumnParallelLinear(
                self.hidden_size,
                self.hidden_size,
                gather_output=True,
                bias=False,
                return_bias=False,
                quant_config=draft_vllm_config.quant_config,
                prefix=f"{prefix}.fc_hidden",
            )
            self.layers = nn.ModuleList(
                Qwen4ExpDecoderLayer(
                    draft_vllm_config,
                    layer_type="qwen_sparse_attention",
                    prefix=f"{prefix}.layers.{self.mtp_start_layer_idx + idx}",
                )
                for idx in range(self.num_mtp_layers)
            )
        self.is_fused_shared_expert_enabled = is_model_fused_shared_expert_compatible(
            self.layers,
            Qwen4ExpSparseMoeBlock,
            "mlp",
        )

        self.pre_fc_norm_embedding = GemmaRMSNorm(
            self.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_fc_norm_hidden = GemmaRMSNorm(
            self.hidden_size * self.hc_count, eps=config.rms_norm_eps
        )
        # HC final mixer collapses the multi stream into [T, H] for the LM head.
        hc_config = HyperConnectionConfig(
            hc_count=config.hc_count,
            hidden_size=config.hidden_size,
            params_dtype=torch.bfloat16,
            hc_lowrank=config.hc_lowrank,
            rms_norm_eps=config.rms_norm_eps,
            hc_per_branch_norm=True,
        )
        self.hyper_connection_mixer = GatedResidual(
            hc_config,
            use_combine=False,
            prefix=maybe_prefix(prefix, "hyper_connection_mixer"),
        )
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states"], self.hidden_size * self.hc_count
        )

    def _iter_qsa_attentions(self):
        """Yield MTP attention modules that own a QSA indexer."""
        for layer in self.layers:
            attention = getattr(layer, "self_attn", None)
            if (
                attention is not None
                and getattr(attention, "indexer", None) is not None
            ):
                yield attention

    def set_skip_topk(self, skip: bool) -> None:
        """Select on MTP step 0 and reuse its QSA indices on later steps."""
        for attention in self._iter_qsa_attentions():
            attention.indexer.skip_topk = skip

    def compact_topk_indices(self, row_indices: torch.Tensor) -> None:
        """Keep each request's target-aligned step-0 sparse-index row."""
        num_rows = row_indices.numel()
        for attention in self._iter_qsa_attentions():
            buffer = attention.topk_indices_buffer
            selected = buffer.index_select(0, row_indices)
            buffer[:num_rows].copy_(selected)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        hidden_states: torch.Tensor | None = None,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | IntermediateTensors:
        hc_count = self.hc_count
        hidden_size = self.hidden_size
        prev_block_output: torch.Tensor | None = None

        if intermediate_tensors is None:
            assert hidden_states is not None
            if inputs_embeds is None:
                assert input_ids is not None
                inputs_embeds = self.embed_input_ids(input_ids)
            # Embedding branch: pre-norm -> fc_embedding -> [T, H].
            inputs_embeds = self.pre_fc_norm_embedding(inputs_embeds)
            inputs_embeds = self.fc_embedding(inputs_embeds)

            # Backbone hidden is multi-stream [T, hc_count*H] (scheme A:
            # the main model truly emits the pre-final-mixer multi stream
            # on the first step; subsequent steps reuse the prior draft
            # step's multi stream).
            num_tokens = hidden_states.shape[0]
            hidden_states = hidden_states.view(num_tokens, hc_count, hidden_size)
            hidden_states = self.pre_fc_norm_hidden(hidden_states.flatten(-2)).view(
                num_tokens, hc_count, hidden_size
            )
            hidden_states = self.fc_hidden(hidden_states)
            hidden_states = hidden_states.flatten(-2)
            prev_block_output = inputs_embeds
        else:
            hidden_states = intermediate_tensors["hidden_states"]

        current_step_idx = spec_step_idx % self.num_mtp_layers
        full_num_tokens = positions.shape[-1]
        if self.use_hc_sequence_parallel:
            # SP requires PP=1, so the first-rank branch sets prev_block_output.
            assert prev_block_output is not None
            if (
                self.use_sequence_parallel
                and envs.VLLM_MOE_SKIP_PADDING
                and is_forward_context_available()
            ):
                forward_context = get_forward_context()
                forward_context.is_padding = sp_padding_mask(
                    forward_context.is_padding, hidden_states
                )
            hidden_states = sp_shard(hidden_states)
            # The FC output is already complete and needs only a slice.
            prev_block_output = sp_shard(prev_block_output)
        layer = self.layers[current_step_idx]
        hidden_states, block_output, injection = layer(
            hidden_states=hidden_states,
            prev_block_output=prev_block_output,
            prev_injection=None,
            positions=positions,
            input_ids=None,
            query_start_loc=None,
            ngram_context=None,
        )
        if not get_pp_group().is_last_rank:
            # As in the target model, PP carries a materialized tensor rather
            # than the delayed hidden/output/injection tuple.
            hidden_states = layer.mlp_hyper_connection.combine(
                hidden_states, block_output, injection
            )
            return IntermediateTensors({"hidden_states": hidden_states})

        # Last PP rank finalize. Keep both:
        #   (A) sample_hidden_states [T, H]  -> single stream for the LM head
        #   (B) multi_hidden [T, hc_count*H] -> pre-final-mixer multi stream
        #       for the next draft step (zero extra compute, just kept).
        multi_hidden, sample_hidden_states, _ = (
            self.hyper_connection_mixer.combine_and_mix(
                hidden_states, block_output, injection
            )
        )
        if self.use_hc_sequence_parallel:
            # Gather both outputs together, then restore their contiguous layouts.
            packed_hidden_states = torch.cat(
                [sample_hidden_states, multi_hidden], dim=-1
            )
            packed_hidden_states = sp_all_gather(packed_hidden_states)[:full_num_tokens]
            sample_hidden_states, multi_hidden = packed_hidden_states.split(
                [hidden_size, hc_count * hidden_size], dim=-1
            )
            sample_hidden_states = sample_hidden_states.contiguous()
            multi_hidden = multi_hidden.contiguous()
        return sample_hidden_states, multi_hidden

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        weights = maybe_fuse_shared_experts(
            weights,
            enabled=self.is_fused_shared_expert_enabled,
            n_routed_experts=getattr(self.config, "num_experts", 0) or 0,
            n_shared_experts=1,
            ckpt_prefix="mlp.shared_expert",
        )
        mapper = self.hf_to_vllm_mapper | WeightsMapper(
            orig_to_new_substr={"hyper_connection_mixer.block_inject_weight": None}
        )
        loader = AutoWeightsLoader(
            self,
            ignore_unexpected_suffixes=_QWEN4_EXP_IGNORED_MISSING_SUFFIXES.copy(),
        )
        return loader.load_weights(weights, mapper=mapper)


def _max_draft_logits_rows(vllm_config: VllmConfig) -> int:
    """Rows of one draft compute_logits call: at most one per scheduled
    request (padded CUDA-graph rows included), i.e. max_num_seqs."""
    scheduler_config = getattr(vllm_config, "scheduler_config", None)
    return int(getattr(scheduler_config, "max_num_seqs", 0) or 0)


def _gather_tp_fingerprints(fingerprint: str) -> list[str]:
    """This rank's draft-vocab table fingerprint, all-gathered over the drafter's TP
    group (CPU group). Not the world group: the V2 runner builds the drafter on
    the last pipeline stage only, so a world collective here would hang."""
    if get_tensor_model_parallel_world_size() == 1:
        return [fingerprint]
    tp_group = get_tp_group()
    gathered: list[str | None] = [None] * tp_group.world_size
    torch.distributed.all_gather_object(
        gathered, fingerprint, group=tp_group.cpu_group
    )
    return [str(fp) for fp in gathered]


def _is_capturing(t: torch.Tensor) -> bool:
    return t.is_cuda and torch.cuda.is_current_stream_capturing()


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
        "hidden_states": 0,
    }
)
class Qwen4ExpMTP(
    LocalArgmaxMixin,
    nn.Module,
    SupportsHCSP,
    SupportsPP,
    Qwen4ExpMixtureOfExperts,
    SupportsReplaySSM,
):
    # compute_logits reads self.lm_head through the LogitsProcessor, so the
    # speculator can swap in a quantized draft-only head.
    supports_draft_lm_head_quant = True
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
        "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
        "in_proj_ba": ["in_proj_b", "in_proj_a"],
        "input_mix_weight_down_block_inject": [
            "input_mix_weight_down",
            "block_inject_weight",
            "_input_mix_padding",
        ],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        assert_graph_profile_is_executable(vllm_config, role="draft")
        assert_ple_mode(vllm_config, role="draft")
        config: Qwen4ExpTextConfig = vllm_config.model_config.hf_text_config
        self.vllm_config = vllm_config
        self.quant_config = vllm_config.quant_config

        super().__init__()
        self.config = config
        self.model = Qwen4ExpMultiTokenPredictor(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "mtp"),
        )
        assert_compiled_submodule(self.model, vllm_config, role="draft")

        draft_vocab_path = draft_vocab_file()
        self.draft_vocab_ids: torch.Tensor | None = None
        if draft_vocab_path is not None:
            # Recheck verification where the reduced draft vocabulary actually activates, with the path
            # being activated: the config may predate the knob.
            speculative_config = getattr(vllm_config, "speculative_config", None)
            if speculative_config is not None:
                speculative_config.verify_draft_vocab_verification(draft_vocab_path)
            if config.tie_word_embeddings:
                raise ValueError(
                    f"{ENV_DRAFT_VOCAB_FILE} needs untied embeddings "
                    "(tie_word_embeddings=True on this checkpoint)"
                )
            self.draft_vocab_ids = load_draft_vocab_ids(
                draft_vocab_path,
                config.vocab_size,
                get_tensor_model_parallel_world_size(),
            )
            # All drafter ranks must serve one table, before any capture.
            check_draft_vocab_agreement(
                _gather_tp_fingerprints(draft_vocab_fingerprint(self.draft_vocab_ids)),
                draft_vocab_path,
            )
        head_vocab_size = (
            config.vocab_size
            if self.draft_vocab_ids is None
            else int(self.draft_vocab_ids.numel())
        )

        if get_pp_group().is_last_rank:
            if config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    head_vocab_size,
                    config.hidden_size,
                    prefix=maybe_prefix(prefix, "lm_head"),
                )
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(head_vocab_size)
        if self.draft_vocab_ids is not None:
            self.has_own_lm_head = get_pp_group().is_last_rank
            if get_pp_group().is_last_rank:
                map_device = self.lm_head.weight.device
                # Buffers, not Parameters: the loader's not-initialised check
                # walks parameters. Filled in load_weights, after the rows load.
                self.register_buffer(
                    "_draft_ids",
                    torch.zeros(head_vocab_size, dtype=torch.int64, device=map_device),
                    persistent=False,
                )
                self.register_buffer(
                    "draft_id_to_target_id",
                    torch.zeros(head_vocab_size, dtype=torch.int64, device=map_device),
                    persistent=False,
                )
                self.register_buffer(
                    "_draft_logits_full",
                    self._new_draft_logits_buffer(
                        _max_draft_logits_rows(vllm_config),
                        getattr(vllm_config.model_config, "head_dtype", None)
                        or self.lm_head.weight.dtype,
                        map_device,
                    ),
                    persistent=False,
                )
            else:
                self.draft_id_to_target_id = None
            self._draft_head_loaded = False
            # Set after CUDA-graph capture: from then on graphs hold pointers
            # into _draft_logits_full, so it must never be re-allocated.
            self._draft_logits_sealed = False
        else:
            # Plain attribute (NOT a buffer): register_buffer refuses a name that
            # already exists as an attribute.
            self.draft_id_to_target_id = None
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )
        self.set_moe_parameters(self.model.layers)
        enable_qwen4_exp_low_latency_gemm(self, vllm_config.model_config.dtype)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        hidden_states: torch.Tensor | None = None,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | IntermediateTensors:
        return self.model(
            input_ids,
            positions,
            hidden_states,
            intermediate_tensors,
            inputs_embeds,
            spec_step_idx=spec_step_idx,
        )

    def compute_logits(
        self, hidden_states: torch.Tensor, spec_step_idx: int = 0
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        if self.draft_vocab_ids is None or logits is None:
            return logits
        return scatter_to_full(
            logits,
            self._draft_ids,
            self.config.vocab_size,
            out=self._draft_logits_out(logits),
        )

    def _new_draft_logits_buffer(
        self, rows: int, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        return torch.full(
            (rows, self.config.vocab_size), float("-inf"), dtype=dtype, device=device
        )

    def _draft_logits_out(self, logits: torch.Tensor) -> torch.Tensor:
        """The persistent scatter_to_full output buffer, checked against this
        call. A mismatch is re-allocated only before capture (the first eager
        warmup call); during a capture or once sealed after capture it raises,
        since captured graphs hold pointers into the buffer."""
        buf = self._draft_logits_full
        rows = int(logits.shape[0])
        if (
            rows <= buf.shape[0]
            and buf.dtype == logits.dtype
            and buf.device == logits.device
        ):
            return buf
        if self._draft_logits_sealed or _is_capturing(logits):
            when = (
                "after graph capture (sealed)"
                if self._draft_logits_sealed
                else "during graph capture"
            )
            raise RuntimeError(
                "draft logits buffer "
                f"{tuple(buf.shape)} {buf.dtype} {buf.device} does not fit "
                f"{rows} rows of {logits.dtype} on {logits.device} {when}; it "
                "must be sized before capture"
            )
        logger.warning(
            "draft logits buffer %s %s on %s re-allocated for %d rows of %s "
            "on %s (outside capture)",
            tuple(buf.shape),
            buf.dtype,
            buf.device,
            rows,
            logits.dtype,
            logits.device,
        )
        self._draft_logits_full = self._new_draft_logits_buffer(
            max(rows, int(buf.shape[0])), logits.dtype, logits.device
        )
        return self._draft_logits_full

    def seal_draft_logits_buffer(self) -> None:
        """Called by the worker after CUDA-graph capture: forbid any later
        re-allocation of the persistent draft-vocab logits buffer. No-op when the knob is
        off or this stage holds no head."""
        if hasattr(self, "_draft_logits_full"):
            self._draft_logits_sealed = True

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        draft_ids = self.draft_vocab_ids
        owns_head = draft_ids is not None and get_pp_group().is_last_rank

        def remap_weight_names():
            for name, weight in weights:
                remapped_name = _remap_mtp_weight_name(name)
                if remapped_name is None:
                    continue
                if owns_head and remapped_name == "lm_head.weight":
                    weight = select_head_rows(
                        weight,
                        draft_ids,
                        self.config.vocab_size,
                        self.config.hidden_size,
                    )
                yield remapped_name, weight

        mapper = WeightsMapper(
            orig_to_new_substr={"hyper_connection_mixer.block_inject_weight": None}
        )
        loader = AutoWeightsLoader(
            self,
            ignore_unexpected_suffixes=_QWEN4_EXP_IGNORED_MISSING_SUFFIXES.copy(),
        )
        # Before profiling/capture.
        loaded = loader.load_weights(remap_weight_names(), mapper=mapper)
        if owns_head:
            self._finalize_draft_head(loaded)
        maybe_load_mtp_qsa_static_kv_scales(self)
        return loaded

    def _finalize_draft_head(self, loaded: set[str]) -> None:
        """Proof of life. A silent knob is not a knob: refuse a random head."""
        if "lm_head.weight" not in loaded:
            raise RuntimeError(
                f"{ENV_DRAFT_VOCAB_FILE} is set but lm_head.weight was not "
                "loaded through Qwen4ExpMTP.load_weights - refusing to serve a "
                "random draft head"
            )
        self._draft_head_loaded = True
        ids = self.draft_vocab_ids
        self._draft_ids.copy_(ids)
        self.draft_id_to_target_id.copy_(d2t_offsets(ids))
        w = self.lm_head.weight
        logger.info(
            "REDUCED DRAFT VOCAB ACTIVE: tp_rank=%d N=%d of %d (file %s, "
            "table %s), lm_head local shard %s %s on %s, has_own_lm_head=%s",
            get_tensor_model_parallel_rank(),
            int(ids.numel()),
            self.config.vocab_size,
            draft_vocab_file(),
            draft_vocab_fingerprint(ids),
            tuple(w.shape),
            w.dtype,
            w.device,
            self.has_own_lm_head,
        )


__all__ = ["Qwen4ExpMTP", "Qwen4ExpMultiTokenPredictor"]
