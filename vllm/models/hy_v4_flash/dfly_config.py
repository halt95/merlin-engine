# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modified by the Merlin project; see NOTICE.
"""Flash target policy for the HYV4 DFly2 draft.

Adapted from #68296 ``vllm/models/hy_v4_flash/dfly_config.py:1-63``: the
selected target layers must be sparse (not SWA) and share one RoPE, which the
draft adopts. The draft config is a Flash config (model type ``hy_v4_dfly``)
so the draft layers can be built by the same attention factory as the
target's sparse layers.
"""

from copy import deepcopy
from typing import Any

from vllm.models.hy_v4.dfly_config import (
    DFLY_ARCHITECTURE,
    DFLY_MODEL_TYPE,
    HYV4DFlySpeculatorConfig,
    _strip_mrope,
)
from vllm.models.hy_v4_flash.configs import SPARSE_ATTENTION, HYV4FlashConfig


class HYV4FlashDFlyConfig(HYV4FlashConfig):
    """Flash-shaped config of the 5-layer DFly2 draft (dense, sparse attention)."""

    model_type = DFLY_MODEL_TYPE

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any], **kwargs: Any):
        values = deepcopy(config_dict)
        values.pop("text_config", None)
        values["model_type"] = cls.model_type
        # Skip HYV4FlashConfig.from_dict: the values are already flat.
        return super(HYV4FlashConfig, cls).from_dict(values, **kwargs)


class HYV4FlashDFlySpeculatorConfig(HYV4DFlySpeculatorConfig):
    target_model_type = "hy_v4_flash"

    @classmethod
    def derive_draft_hf_config(cls, hf_config, *, target_config):
        """#68296 hy_v4/dfly_config.py:96-179 + hy_v4_flash/dfly_config.py:20-31."""
        options = cls.collect_options(target_config)
        text = target_config.get_text_config()
        num_layers = options["num_hidden_layers"]
        rope = _strip_mrope(
            cls.select_rope(
                text, options["target_index_layer"], options["target_kv_layers"]
            )
        )
        values = text.to_dict()
        for name in ("architectures", "auto_map", "transformers_version"):
            values.pop(name, None)
        width = options["intermediate_size"]
        values.update(
            architectures=[DFLY_ARCHITECTURE],
            num_hidden_layers=num_layers,
            layer_types=[SPARSE_ATTENTION] * num_layers,
            mlp_layer_types=["dense"] * num_layers,
            indexer_types=["full"] * num_layers,
            enable_ihc=False,
            num_nextn_predict_layers=0,
            engram_layer_ids=None,
            engram_layer_head_vocab_sizes=None,
            rope_parameters=rope,
            # The FFN width is probed from the checkpoint when undeclared.
            intermediate_size=width if width is not None else text.intermediate_size,
            # Draft sparse layers keep their post-projection norm unless the
            # checkpoint omits it (resolved at load time); FFNs have none.
            sandwich_norm=True,
            dsa_skip_attn_post_norm=False,
            target_num_hidden_layers=text.num_hidden_layers,
            target_hidden_size=text.hidden_size,
            target_kv_layers=options["target_kv_layers"],
            target_index_layer=options["target_index_layer"],
            target_index_topk=options["target_index_topk"],
            context_local_window=options["context_local_window"],
            mask_token_id=options["mask_token_id"],
            dfly_intermediate_size=width,
            dfly_attn_post_proj_norm=options["attn_post_proj_norm"],
            hidden_correction_intermediate_size=options[
                "hidden_correction_intermediate_size"
            ],
        )
        return HYV4FlashDFlyConfig.from_dict(values)

    @classmethod
    def select_rope(cls, text, index_layers, kv_layers) -> dict:
        selected = []
        for name, layers in (
            ("target_index_layer", index_layers),
            ("target_kv_layers", kv_layers),
        ):
            for index in layers:
                if text.layer_types[index] != SPARSE_ATTENTION:
                    raise ValueError(
                        f"DFly {name} layer {index} must be sparse, not SWA"
                    )
                selected.append(text.rope_for_layer(index))
        if any(_strip_mrope(r) != _strip_mrope(selected[0]) for r in selected[1:]):
            raise ValueError("DFly selected target layers must use identical RoPE")
        return deepcopy(selected[0])

    @classmethod
    def target_rope_parameters(cls, target_text_config, draft_config) -> dict:
        return cls.select_rope(
            target_text_config,
            draft_config.target_index_layer,
            draft_config.target_kv_layers,
        )
