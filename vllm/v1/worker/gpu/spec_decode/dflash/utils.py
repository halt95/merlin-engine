# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modified by the Merlin project; see NOTICE.
import glob
import json
import os

import torch.nn as nn
from safetensors import safe_open

from vllm.config import VllmConfig, replace
from vllm.distributed import get_pp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.vocab_parallel_embedding import (
    PinnedHostVocabParallelEmbedding,
    should_use_host_embed_table,
)
from vllm.model_executor.model_loader import get_model
from vllm.model_executor.model_loader.utils import get_draft_load_config
from vllm.model_executor.models.utils import PPMissingLayer
from vllm.v1.worker.gpu.spec_decode.eagle.utils import (
    _should_share,
    get_target_lm_head,
    maybe_share_target_embed,
)
from vllm.v1.worker.gpu.spec_decode.utils import get_pp_safe_draft_load_config

logger = init_logger(__name__)


def load_dflash_model(target_model: nn.Module, vllm_config: VllmConfig) -> nn.Module:
    from vllm.compilation.backends import set_model_tag
    from vllm.model_executor.models.qwen3_dflash import (
        dflash_has_any_non_causal,
    )

    speculative_config = vllm_config.speculative_config
    assert speculative_config is not None
    draft_model_config = speculative_config.draft_model_config
    # Select an attention backend that supports the drafter's attention: mixing
    # a non-causal layer onto a causal-only backend would fail.
    draft_vllm_config = replace(
        vllm_config,
        attention_config=replace(
            vllm_config.attention_config,
            use_non_causal=dflash_has_any_non_causal(draft_model_config.hf_config),
            backend=speculative_config.attention_backend,
        ),
        cache_config=(
            replace(
                vllm_config.cache_config,
                cache_dtype=speculative_config.kv_cache_dtype,
            )
            if speculative_config.kv_cache_dtype is not None
            else vllm_config.cache_config
        ),
        load_config=get_pp_safe_draft_load_config(get_draft_load_config(vllm_config)),
    )
    with set_model_tag("dflash_head"):
        dflash_model = get_model(
            vllm_config=draft_vllm_config, model_config=draft_model_config
        )

    target_language_model = (
        target_model.get_language_model()
        if hasattr(target_model, "get_language_model")
        else target_model
    )
    # MuseGlimmerForCausalLM marks its inner MuseGlimmerModel as the language
    # model, so get_language_model() already returns the inner module and has
    # no .model of its own.
    target_inner = getattr(target_language_model, "model", target_language_model)
    draft_inner = dflash_model.model

    if not _maybe_load_target_embed_on_pp(draft_inner, vllm_config, target_inner):
        maybe_share_target_embed(dflash_model, draft_inner, target_inner)

    target_lm_head = get_target_lm_head(target_model, target_language_model)
    draft_lm_head = getattr(dflash_model, "lm_head", None)
    if (
        target_lm_head is not None
        and not getattr(dflash_model.config, "has_own_lm_head", False)
        and _should_share(
            dflash_model, "has_own_lm_head", draft_lm_head, target_lm_head
        )
    ):
        if draft_lm_head is not None:
            del dflash_model.lm_head
        dflash_model.lm_head = target_lm_head

    return dflash_model


def _find_target_embed_tensor(model_dir: str) -> tuple[str, str]:
    """(file, tensor name) of the target's input embedding in a local
    safetensors checkpoint. MTP and vision copies are skipped."""
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.isfile(index_path):
        with open(index_path, encoding="utf-8") as f:
            weight_map: dict[str, str] = json.load(f)["weight_map"]
    else:
        weight_map = {}
        for path in glob.glob(os.path.join(model_dir, "*.safetensors")):
            with safe_open(path, framework="pt") as f:
                weight_map.update(dict.fromkeys(f.keys(), os.path.basename(path)))
    names = sorted(
        (
            name
            for name in weight_map
            if name.endswith("embed_tokens.weight")
            and "mtp" not in name
            and "visual" not in name
        ),
        key=len,
    )
    if not names:
        raise RuntimeError(f"no input embedding found in the target at {model_dir}")
    return os.path.join(model_dir, weight_map[names[0]]), names[0]


def _resident_target_embed(target_inner: nn.Module) -> nn.Module | None:
    embed = getattr(target_inner, "embed_tokens", None) or getattr(
        target_inner, "embedding", None
    )
    return None if embed is None or isinstance(embed, PPMissingLayer) else embed


def _maybe_load_target_embed_on_pp(
    draft_inner: nn.Module, vllm_config: VllmConfig, target_inner: nn.Module
) -> bool:
    """Give a draft without its own embedding table the target's, under PP.

    With PP>1 the drafter runs on the last stage while the target embedding
    lives on the first, so maybe_share_target_embed cannot share it and the
    draft table would stay uninitialized. Load the target's table from its
    checkpoint instead (into pinned host memory under VLLM_HOST_EMBED_TABLE,
    matching the target). Returns whether it did; otherwise the caller shares
    the resident target table as before.
    """
    if get_pp_group().world_size == 1:
        return False
    if getattr(draft_inner, "loaded_embed_tokens", True):
        return False
    # A target that keeps its input embedding on this stage (e.g. Kimi for
    # DFlash) is shared by maybe_share_target_embed; load only when absent.
    if _resident_target_embed(target_inner) is not None:
        return False
    old = draft_inner.embed_tokens
    if isinstance(old, PinnedHostVocabParallelEmbedding):
        embed = old
    elif should_use_host_embed_table():
        embed = PinnedHostVocabParallelEmbedding(
            old.org_vocab_size,
            old.embedding_dim,
            params_dtype=old.weight.dtype,
        )
        del draft_inner.embed_tokens
        draft_inner.embed_tokens = embed
    else:
        embed = old

    model_dir = vllm_config.model_config.model
    if not os.path.isdir(model_dir):
        raise RuntimeError(
            "a DFlash draft without its own embedding needs a local target "
            f"checkpoint under pipeline parallelism; got {model_dir!r}"
        )
    path, name = _find_target_embed_tensor(model_dir)
    with safe_open(path, framework="pt") as f:
        weight = f.get_tensor(name)
    if weight.shape[0] != embed.org_vocab_size or (
        weight.shape[1] != embed.embedding_dim
    ):
        raise RuntimeError(
            f"target embedding {name} {tuple(weight.shape)} does not match the "
            f"draft table ({embed.org_vocab_size}, {embed.embedding_dim})"
        )
    embed.weight_loader(embed.weight, weight.to(embed.weight.dtype))
    logger.info(
        "DFlash draft: loaded the target input embedding %s for PP stage %d (%s)",
        name,
        get_pp_group().rank_in_group,
        "pinned host"
        if isinstance(embed, PinnedHostVocabParallelEmbedding)
        else "device",
    )
    return True
