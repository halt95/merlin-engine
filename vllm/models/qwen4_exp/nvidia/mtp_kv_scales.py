# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Separate target and MTP calibration in the VLLM_QSA_KV_SCALES sidecar.

Existing top-level target entries are unchanged. Optional draft entries live in
an "mtp" object, keyed by the runtime suffix "mtp.layers.N.self_attn.attn".
N is num_hidden_layers + the draft layer index (48 for Flash-Next's first MTP
layer), not the checkpoint's zero-based MTP index. Any outer draft-model prefix
is stripped for lookup. Never fall back to a target layer's calibration.

For example, add this alongside the existing target entries, using measured
draft scales (the numbers below are illustrative, not a calibration):
    "mtp": {"mtp.layers.48.self_attn.attn": {"k_scale": 0.25, "v_scale": 2.0}}
"""

import logging
import os
import re
from pathlib import Path

logger = logging.getLogger(__name__)


def qsa_kv_scale_entries(raw: dict, path: Path, *, mtp: bool = False) -> dict:
    """Select one namespace; the existing QSA loader validates its entries."""
    draft = raw.get("mtp", {})
    if not isinstance(draft, dict):
        raise ValueError(f"{path}: mtp scales must be a JSON object keyed by layer name")
    if mtp:
        return draft
    return {name: entry for name, entry in raw.items() if name != "mtp"}


def mtp_qsa_scale_name(layer_name: str) -> str:
    """Resolve a registered draft layer name to its sidecar key."""
    match = re.search(r"(?:^|\.)mtp\.layers\.(\d+)\.self_attn\.attn$", layer_name)
    if match is None:
        raise ValueError(f"Unexpected MTP QSA layer name: {layer_name!r}")
    return f"mtp.layers.{match.group(1)}.self_attn.attn"


def maybe_load_mtp_qsa_static_kv_scales(model) -> None:
    """Apply draft-only scales after weights load, before profiling/capture."""
    path = os.environ.get("VLLM_QSA_KV_SCALES", "").strip()
    if not path:
        return

    # Lazy import: qsa.py also uses the namespace selector above.
    from .qsa import (
        Qwen4ExpQSAAttention,
        _is_qsa_fp8_cache_dtype,
        load_qsa_static_kv_scales,
    )

    layers = {
        mtp_qsa_scale_name(layer.layer_name): layer
        for layer in model.modules()
        if isinstance(layer, Qwen4ExpQSAAttention)
        and _is_qsa_fp8_cache_dtype(layer.kv_cache_dtype)
    }
    if any(layer._k_scale.device.type == "meta" for layer in layers.values()):
        logger.info(
            "Qwen4Exp MTP QSA: model is on meta; drafter K/V scales "
            "are applied by the GPU workers only"
        )
        return
    if not layers:
        logger.info("Qwen4Exp MTP QSA: 0 local FP8 drafter layers to calibrate")
        return

    applied = load_qsa_static_kv_scales(layers, path, strict=False, mtp=True)
    logger.info(
        "Qwen4Exp MTP QSA: applied static K/V scales from %s "
        "to %d of %d drafter layer(s): %s",
        path,
        len(applied),
        len(layers),
        ", ".join(applied) or "<none>",
    )
    missing = sorted(set(layers) - set(applied))
    if missing:
        logger.warning(
            "Qwen4Exp MTP QSA: missing static K/V scales in %s for "
            "drafter layer(s) %s; retaining existing scales (default 1.0)",
            path,
            ", ".join(missing),
        )
