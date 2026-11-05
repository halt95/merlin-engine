# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Boot-time checks for checkpoint-owned quantization scales."""

import torch


def validate_checkpoint_scale(
    layer: torch.nn.Module,
    name: str,
    *,
    sentinel: float | None = None,
) -> None:
    scale = getattr(layer, name)
    valid = torch.isfinite(scale).all()
    if sentinel is not None:
        valid &= (scale != sentinel).all()
    if not bool(valid):
        layer_name = getattr(
            layer, "layer_name", getattr(layer, "prefix", type(layer).__name__)
        )
        raise ValueError(
            f"Quantized checkpoint has missing or non-finite {name!r} "
            f"for layer {layer_name!r}"
        )
