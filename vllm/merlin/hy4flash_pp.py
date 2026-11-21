# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Pipeline-parallel stage ownership for Hy4 Flash (HYV4FlashForCausalLM).

Upstream (vllm-project/vllm#68296) refuses PP for this model. Merlin runs it as TP2xPP2 on
4x3090 (TP4xPP2 on 8 cards) on top of the in-tree ``hy_v4`` PP path, which
already ships the flattened ``[tokens, hc_mult * hidden_size]`` iHC transfer
(``vllm/models/hy_v4/nvidia/model.py`` ``HYV4Model.forward``).

This module holds the pure stage arithmetic the model and the config hook
share:

- ``stage_bounds``: the engine's own split (``get_pp_indices``, so
  ``VLLM_PP_LAYER_PARTITION`` is honoured exactly as ``make_layers`` sees it);
- ``local_engram_hash_rows``: which Engram layers a stage owns, with their
  checkpoint-global hash-row index (the multipliers/primes/offsets of a row
  are seeded by the layer ID, so a stage must keep the global row, not a
  renumbered one);
- ``validate_dfly_locality``: DFly2 reads target KV/index of layers
  25/29/33/37/41, so every source layer must sit on the last stage, where
  the drafter runs;
- ``verify_and_update_hy4flash_config``: the config-time hook (sparse MQA
  forced before the config is hashed; the PP split checked).

Bounds are zero-based half-open decoder-layer ranges. Engram layer IDs are
one-based (``engram_layer_ids`` in the checkpoint config). DFly2 source
layers are zero-based (``dfly_target_kv_layers``).
"""

from collections.abc import Sequence
from typing import Any

HY4FLASH_ARCHS = ("HYV4FlashForCausalLM", "HYV4FlashForConditionalGeneration")


def partition_bounds(
    partition: Sequence[int], num_layers: int
) -> tuple[tuple[int, int], ...]:
    """Validate whole-layer counts and return zero-based half-open stage ranges."""
    if type(num_layers) is not int or num_layers < 1:
        raise ValueError("num_layers must be a positive integer")
    if not partition or any(type(n) is not int or n < 1 for n in partition):
        raise ValueError("each PP stage must own a positive number of layers")
    if sum(partition) != num_layers:
        raise ValueError("PP partition must cover exactly num_layers")
    bounds = []
    start = 0
    for count in partition:
        bounds.append((start, start + count))
        start += count
    return tuple(bounds)


def stage_bounds(num_layers: int, pp_size: int) -> tuple[tuple[int, int], ...]:
    """Every stage's ``(start, end)`` as the engine computes it for ``make_layers``.

    Uses ``get_pp_indices`` so ``VLLM_PP_LAYER_PARTITION`` (the launcher's
    ``PP_PARTITION``) and the uneven-split rule are the engine's own.
    """
    from vllm.distributed.utils import get_pp_indices

    if type(pp_size) is not int or pp_size < 1:
        raise ValueError("pp_size must be a positive integer")
    if pp_size > num_layers:
        raise ValueError(f"PP={pp_size} exceeds the {num_layers} decoder layers")
    bounds = tuple(get_pp_indices(num_layers, r, pp_size) for r in range(pp_size))
    counts = [end - start for start, end in bounds]
    # Re-validate through partition_bounds: an env partition with a zero entry
    # would otherwise give a stage with no decoder layer.
    return partition_bounds(counts, num_layers)


def local_engram_layers(
    layer_ids: Sequence[int], start: int, end: int, num_layers: int
) -> tuple[int, ...]:
    """Return the one-based Engram IDs a stage owns, in checkpoint order."""
    if (
        type(num_layers) is not int
        or num_layers < 1
        or type(start) is not int
        or type(end) is not int
        or not 0 <= start < end <= num_layers
    ):
        raise ValueError("invalid PP stage bounds")
    if any(type(i) is not int or not 1 <= i <= num_layers for i in layer_ids):
        raise ValueError("Engram layer IDs must be one-based decoder layer numbers")
    if len(set(layer_ids)) != len(layer_ids):
        raise ValueError("Engram layer IDs must be unique")
    return tuple(i for i in layer_ids if start <= i - 1 < end)


def local_engram_hash_rows(
    layer_ids: Sequence[int], start: int, end: int, num_layers: int
) -> tuple[tuple[int, int], ...]:
    """Return ``(one-based layer ID, checkpoint-global hash row)`` for a stage.

    The hash row is the layer's index in the full ``engram_layer_ids`` list,
    which is what ``FlashNgramHashState`` seeds its multipliers and indexes its
    primes/offsets with. An empty tuple means the stage hashes nothing.
    """
    local = set(local_engram_layers(layer_ids, start, end, num_layers))
    return tuple(
        (layer_id, row) for row, layer_id in enumerate(layer_ids) if layer_id in local
    )


def validate_dfly_locality(
    bounds: Sequence[tuple[int, int]],
    target_kv_layers: Sequence[int],
    target_index_layers: Sequence[int],
) -> None:
    """Require each zero-based DFly2 source layer to reside on the last PP stage.

    Both source lists have one entry per draft layer. Index-layer entries may
    repeat; target KV layers must be distinct.
    """
    if not bounds:
        raise ValueError("DFly requires at least one PP stage")
    expected_start = 0
    for start, end in bounds:
        if (
            type(start) is not int
            or type(end) is not int
            or start != expected_start
            or end <= start
        ):
            raise ValueError("PP bounds must be contiguous nonempty ranges from zero")
        expected_start = end
    if not target_kv_layers or len(target_kv_layers) != len(target_index_layers):
        raise ValueError("DFly needs one KV and index source per draft layer")
    last_start, last_end = bounds[-1]
    for name, indices in (
        ("target_kv_layers", target_kv_layers),
        ("target_index_layers", target_index_layers),
    ):
        for index in indices:
            if type(index) is not int or not 0 <= index < last_end:
                raise ValueError(f"invalid DFly {name} layer: {index!r}")
            if index < last_start:
                raise ValueError(
                    f"DFly {name} layer {index} is not on the last PP stage "
                    f"(layers {last_start}-{last_end - 1}); move the split so "
                    "the drafter's source layers are local to it"
                )
    if len(set(target_kv_layers)) != len(target_kv_layers):
        raise ValueError("DFly needs distinct target KV layers")


def dfly_source_layers(
    text_config: Any, hf_config: Any = None
) -> tuple[list[int], list[int]] | None:
    """The checkpoint's DFly2 source layers (zero-based), or None without DFly2.

    Reads ``dfly_*`` from the outer config first, then the text config, as
    ``derive_dfly_config`` does.
    """

    def option(name):
        value = getattr(hf_config, f"dfly_{name}", None) if hf_config else None
        return (
            value if value is not None else getattr(text_config, f"dfly_{name}", None)
        )

    count = option("num_hidden_layers")
    if not count:
        return None
    kv = list(option("target_kv_layers") or [])
    index = option("target_index_layer")
    if type(index) is int:
        index = [index] * count
    return kv, list(index or [])


def check_dfly_stage(vllm_config: Any) -> None:
    """With a speculative config, the DFly2 sources must be on the last stage.

    Call before building the drafter (the DFly2 proposer calls it too). A
    config without DFly2 layers passes.
    """
    text = vllm_config.model_config.hf_text_config
    sources = dfly_source_layers(text, vllm_config.model_config.hf_config)
    if sources is None:
        return
    bounds = stage_bounds(
        text.num_hidden_layers, vllm_config.parallel_config.pipeline_parallel_size
    )
    validate_dfly_locality(bounds, *sources)


def verify_and_update_hy4flash_config(vllm_config: Any) -> None:
    """Config-time settings for HYV4FlashForCausalLM (``MODELS_CONFIG_MAP``).

    - ``attention_config.sparse_mla_force_mqa = True``: the learnable sink only
      exists on the sparse MQA path (the dense MLA prefill backends cannot
      apply it), and the Hy4 sparse builder/impl require it. Setting it here, before
      ``VllmConfig`` hashes itself, puts it in the compile-cache key; setting
      it later, while ``build_attention_parts`` builds the layers, does not.
    - the PP split: every stage owns at least one decoder layer under the
      engine's own partition rule.
    """
    model_config = vllm_config.model_config
    text = model_config.hf_text_config
    if getattr(text, "model_type", None) != "hy_v4_flash":
        return
    vllm_config.attention_config.sparse_mla_force_mqa = True
    pp = vllm_config.parallel_config.pipeline_parallel_size
    if pp > 1:
        stage_bounds(text.num_hidden_layers, pp)
