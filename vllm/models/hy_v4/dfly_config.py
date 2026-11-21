# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modified by the Merlin project; see NOTICE.
"""DFly2 draft configuration and validation for HYV4-family targets.

Adapted from #68296 ``vllm/models/hy_v4/dfly_config.py:96-318`` (draft
derivation, ``adapt_draft_model_config``, ``apply_options``) and
``vllm/v1/worker/gpu/spec_decode/dfly/config.py:1-67`` (token count, options).

Differences from upstream, for this 0.33 tree:
- No speculator plugin API: ``vllm/config/speculative.py`` calls these
  classmethods directly through ``vllm.v1.worker.gpu.spec_decode.dfly.config``.
- The draft runs on the Flash attention factory (the Hy4 sm_86 sparse-MLA
  backend), so only the ``hy_v4_flash`` target is wired; the in-tree
  ``hy_v4`` preview target (FlashMLA, sm_90+) raises.
- Checkpoint quantization goes through
  ``vllm.merlin.hy4_quant.validate_checkpoint_quantization``, which accepts the
  pack's ``dfly_w4a16``/``dfly_w8a16`` groups as well as
  upstream's FP8/MXFP4.
- Decode context parallelism is allowed (the draft reads the TP-sharded target
  cache through the same DCP path) and pipeline
  parallelism is allowed when every target KV/index layer sits on the last
  stage; upstream requires PP = DCP = 1.
"""

from copy import deepcopy
from functools import partial
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from transformers import PreTrainedConfig

    from vllm.config import SpeculativeConfig

DFLY_METHOD = "dfly"
DFLY_MODEL_TYPE = "hy_v4_dfly"
DFLY_ARCHITECTURE = "HYV4DFlyModel"
DFLY_CHECKPOINT_PREFIX = "model.dfly_layers."
# KV cache dtypes the draft may share with the target (#68296 dfly_config.py:314).
DFLY_KV_CACHE_DTYPES = (None, "auto", "bfloat16", "fp8_ds_mla")
# The draft keys of a combined checkpoint (#68296 dfly_config.py:121-135).
_DFLY_OPTIONS = (
    "enable_ihc",
    "mtp_fusion",
    "target_index_topk",
    "context_local_window",
    "enable_hidden_correction",
    "mlp_type",
    "intermediate_size",
    "attn_post_proj_norm",
    "mask_token_id",
    "target_index_layer",
    "target_kv_layers",
    "hidden_correction_intermediate_size",
)


def _dfly_option(target_config, text_config, name: str, default: Any) -> Any:
    """``dfly_<name>`` from the outer config first, then the text config."""
    key = f"dfly_{name}"
    if hasattr(target_config, key):
        return deepcopy(getattr(target_config, key))
    return deepcopy(getattr(text_config, key, default))


def last_stage_first_layer(num_layers: int, pp_size: int) -> int:
    """First decoder layer of the last PP stage (honours VLLM_PP_LAYER_PARTITION)."""
    if pp_size <= 1:
        return 0
    from vllm.distributed.utils import get_pp_indices

    return get_pp_indices(num_layers, pp_size - 1, pp_size)[0]


class HYV4DFlySpeculatorConfig:
    """Method ``dfly`` for HYV4-family targets. Subclasses set the target family."""

    base_method = DFLY_METHOD
    target_model_type = "hy_v4"
    # Upstream DFlySpeculatorConfig (#68296 spec_decode/dfly/config.py:22-25).
    uses_target_hidden_states = True
    shares_target_positions = False
    parallel_drafting = True

    # ---------------------------------------------------------------- derivation
    @classmethod
    def derive_draft_hf_config(
        cls, hf_config: "PreTrainedConfig", *, target_config: "PreTrainedConfig"
    ) -> "PreTrainedConfig":
        raise NotImplementedError(
            "DFly is ported for the hy_v4_flash target only; the hy_v4 preview "
            "target's attention (FlashMLA) does not run on sm_86"
        )

    @classmethod
    def collect_options(cls, target_config: "PreTrainedConfig") -> dict[str, Any]:
        """Read and check the dfly_* keys (#68296 dfly_config.py:96-167)."""
        text = target_config.get_text_config()
        if text.model_type != cls.target_model_type:
            raise ValueError(f"DFly requires a {cls.target_model_type} target model")
        num_layers = _dfly_option(target_config, text, "num_hidden_layers", None)
        if type(num_layers) is not int or num_layers < 1:
            raise ValueError("DFly requires a positive dfly_num_hidden_layers")
        defaults = {
            "enable_ihc": False,
            "mtp_fusion": "slot0",
            "target_index_topk": text.index_topk,
            "context_local_window": None,
            "enable_hidden_correction": True,
            "mlp_type": "dense",
            "intermediate_size": None,
            "attn_post_proj_norm": None,
            "mask_token_id": None,
            "target_index_layer": None,
            "target_kv_layers": None,
            "hidden_correction_intermediate_size": text.hidden_size,
        }
        options = {
            name: _dfly_option(target_config, text, name, defaults[name])
            for name in _DFLY_OPTIONS
        }
        options["num_hidden_layers"] = num_layers
        options["attn_causal"] = _dfly_option(target_config, text, "attn_causal", True)
        width = options["intermediate_size"]
        if width is not None and (type(width) is not int or width < 1):
            raise ValueError("DFly intermediate size must be positive")
        norm = options["attn_post_proj_norm"]
        if norm is not None and type(norm) is not bool:
            raise ValueError("DFly attention post-projection norm must be boolean")
        if options["mlp_type"] != "dense" or options["enable_ihc"] is not False:
            raise ValueError("DFly requires dense FFNs and dfly_enable_ihc=false")
        if options["attn_causal"] is not True:
            raise ValueError("DFly requires causal MLA attention")
        if (
            options["mtp_fusion"] != "slot0"
            or options["enable_hidden_correction"] is not True
        ):
            raise ValueError("DFly requires slot0 fusion and hidden correction")
        mask = options["mask_token_id"]
        if type(mask) is not int or not 0 <= mask < text.vocab_size:
            raise ValueError("DFly dfly_mask_token_id is outside the vocabulary")
        topk = options["target_index_topk"]
        window = options["context_local_window"]
        if (
            type(topk) is not int
            or topk <= 0
            or (window is not None and (type(window) is not int or window < 0))
        ):
            raise ValueError("DFly needs positive target top-k and nonnegative window")
        correction = options["hidden_correction_intermediate_size"]
        if type(correction) is not int or correction <= 0:
            raise ValueError(
                "DFly hidden correction intermediate size must be positive"
            )
        index = options["target_index_layer"]
        if type(index) is int:
            options["target_index_layer"] = [index] * num_layers
        for name in ("target_index_layer", "target_kv_layers"):
            layers = options[name]
            if (
                not isinstance(layers, list)
                or len(layers) != num_layers
                or any(
                    type(i) is not int or not 1 <= i < text.num_hidden_layers
                    for i in layers
                )
            ):
                raise ValueError(
                    f"DFly {name} needs one valid target layer per draft layer"
                )
        if len(set(options["target_kv_layers"])) != num_layers:
            raise ValueError("DFly requires distinct target KV layers")
        return options

    @classmethod
    def draft_hf_overrides(cls, spec: "SpeculativeConfig"):
        """The picklable transform that turns the target config into the draft's."""
        target = spec.target_model_config
        if spec.model not in (target.model, target.model_weights):
            raise ValueError("DFly weights must be in the target checkpoint")
        return partial(
            derive_dfly_hf_config,
            config_cls=cls,
            target_config=deepcopy(target.hf_config),
        )

    @classmethod
    def target_rope_parameters(cls, target_text_config, draft_config) -> dict:
        return deepcopy(target_text_config.rope_parameters)

    # ------------------------------------------------------------ token count
    @classmethod
    def resolve_num_speculative_tokens(cls, spec: "SpeculativeConfig") -> None:
        """#68296 spec_decode/dfly/config.py:44-56: dfly_block_size - 1, or as given."""
        n = spec.num_speculative_tokens
        if n is None:
            block_size = getattr(
                spec.target_model_config.hf_config, "dfly_block_size", None
            )
            if type(block_size) is not int or not 2 <= block_size <= 17:
                raise ValueError(
                    "DFly needs num_speculative_tokens (5 for Hy4 Flash) or "
                    "dfly_block_size in [2, 17]"
                )
            # The checkpoint block includes the separate anchor query.
            n = block_size - 1
        if type(n) is not int or not 1 <= n <= 16:
            raise ValueError("DFly requires num_speculative_tokens in [1, 16]")
        spec.num_speculative_tokens = n

    # ---------------------------------------------------------------- options
    @classmethod
    def apply_options(cls, spec: "SpeculativeConfig") -> None:
        """#68296 dfly_config.py:312-318 and spec_decode/dfly/config.py:58-62."""
        if spec.kv_cache_dtype not in DFLY_KV_CACHE_DTYPES:
            raise ValueError("DFly requires BF16 or fp8_ds_mla draft MLA KV cache")
        if spec.attention_backend is not None:
            raise ValueError("DFly does not support a draft attention backend override")
        if spec.enable_adaptive_verification:
            raise ValueError("DFly does not support adaptive verification")
        if spec.rejection_sample_method != "standard":
            raise ValueError("DFly requires standard target verification")
        if getattr(spec, "use_heterogeneous_vocab", False):
            raise ValueError("DFly shares the target vocabulary")

    # ---------------------------------------------------------------- parallel
    @classmethod
    def verify_target_parallel_config(cls, spec: "SpeculativeConfig") -> None:
        parallel = spec.target_parallel_config
        if parallel.prefill_context_parallel_size != 1:
            raise ValueError("DFly requires prefill_context_parallel_size=1")
        if parallel.data_parallel_size != 1:
            raise ValueError("This DFly port runs one TP-local draft (DP=1)")
        draft = spec.draft_model_config.hf_config
        pp = parallel.pipeline_parallel_size
        if pp > 1:
            # The draft aliases its target layers' KV and reads their top-k
            # snapshots, so all of them must live on the stage that drafts.
            layers = spec.target_model_config.hf_text_config.num_hidden_layers
            first = last_stage_first_layer(layers, pp)
            used = set(draft.target_kv_layers) | set(draft.target_index_layer)
            if min(used) < first:
                raise ValueError(
                    f"DFly target layers {sorted(used)} must all be on the last "
                    f"pipeline stage (layers {first}-{layers - 1})"
                )

    # ------------------------------------------------------------ validation
    @classmethod
    def adapt_draft_model_config(cls, spec: "SpeculativeConfig") -> None:
        """#68296 dfly_config.py:222-310, adapted (see the module docstring)."""
        from vllm.merlin.hy4_quant import validate_checkpoint_quantization
        from vllm.models.hy_v4.checkpoint import checkpoint_module_names

        target = spec.target_model_config
        draft = spec.draft_model_config
        draft.model_weights = target.model_weights
        config = draft.hf_config
        if config.model_type != DFLY_MODEL_TYPE:
            raise ValueError("method='dfly' requires a HYV4 DFly checkpoint")
        text = target.hf_text_config
        if text.model_type != cls.target_model_type:
            raise ValueError(f"DFly requires a {cls.target_model_type} target model")
        if str(target.dtype) != "torch.bfloat16":
            raise ValueError("DFly requires a BF16 target dtype")
        if target.quantization in ("fp8", "compressed-tensors"):
            # A pack is resolved against the checkpoint's module
            # names (the draft reads the target checkpoint, so the same names).
            module_names = (
                checkpoint_module_names(target)
                if target.quantization == "compressed-tensors"
                else None
            )
            validate_checkpoint_quantization(
                target.model_arch_config.quantization_config,
                module_names=module_names,
            )
            if draft.quantization != target.quantization or spec.quantization not in (
                None,
                target.quantization,
            ):
                raise ValueError("DFly must use the checkpoint's quantization")
            validate_checkpoint_quantization(
                draft.model_arch_config.quantization_config,
                module_names=module_names,
            )
        elif target.quantization is not None:
            raise ValueError(
                "DFly supports BF16 targets and the checkpoint's FP8 or "
                "compressed-tensors quantization"
            )
        elif draft.quantization is not None or spec.quantization is not None:
            raise ValueError("DFly does not quantize an unquantized checkpoint's draft")
        parallel = spec.target_parallel_config
        cls.verify_target_parallel_config(spec)
        tp = parallel.tensor_parallel_size
        if spec.draft_tensor_parallel_size not in (None, tp):
            raise ValueError("DFly draft and target must use the same TP size")
        for size in (
            config.num_attention_heads,
            config.dfly_intermediate_size,
            config.hidden_correction_intermediate_size,
        ):
            if size is not None and size % tp:
                raise ValueError(
                    "DFly heads and FFN dimensions must be divisible by TP"
                )
        for field in ("vocab_size", "hidden_size", "kv_lora_rank", "qk_rope_head_dim"):
            if getattr(config, field) != getattr(text, field):
                raise ValueError(f"DFly and target must have the same {field}")
        if config.target_hidden_size != target.get_hidden_size():
            raise ValueError("DFly target_hidden_size disagrees with the target")
        if config.target_index_topk != text.index_topk:
            raise ValueError("DFly target_index_topk disagrees with the target")
        if _strip_mrope(config.rope_parameters) != _strip_mrope(
            cls.target_rope_parameters(text, config)
        ):
            raise ValueError("DFly and target must use identical RoPE parameters")
        if config.target_num_hidden_layers != text.num_hidden_layers:
            raise ValueError("DFly target_num_hidden_layers disagrees with the target")


def _strip_mrope(rope: dict | None) -> dict:
    """Text positions are equal on every M-RoPE axis, so the sections are inert."""
    rope = deepcopy(rope or {})
    for value in [rope, *[v for v in rope.values() if isinstance(v, dict)]]:
        value.pop("mrope_section", None)
        value.pop("mrope_interleaved", None)
    return rope


def derive_dfly_hf_config(
    hf_config: "PreTrainedConfig",
    *,
    config_cls: type[HYV4DFlySpeculatorConfig],
    target_config: "PreTrainedConfig",
) -> "PreTrainedConfig":
    """Module-level (picklable) entry for the draft ModelConfig's hf_overrides."""
    if hf_config.model_type.startswith("dummy_"):
        return hf_config
    return config_cls.derive_draft_hf_config(hf_config, target_config=target_config)


def dfly_config_class(target_model_type: str) -> type[HYV4DFlySpeculatorConfig]:
    """The DFly configuration class for a target text model type."""
    if target_model_type == "hy_v4_flash":
        from vllm.models.hy_v4_flash.dfly_config import HYV4FlashDFlySpeculatorConfig

        return HYV4FlashDFlySpeculatorConfig
    if target_model_type == "hy_v4":
        return HYV4DFlySpeculatorConfig
    raise ValueError(
        "No DFly configuration is registered for target model type "
        f"{target_model_type!r}"
    )
