# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Full layer names for selective weight offload.

``make_layers`` builds a PP stage's decoder layers and hands them to the
offloader through one ``wrap_modules`` generator. The UVA offloader matched
``--cpu-offload-params`` segments against layer-relative parameter names
(``mlp.experts.w13_qweight``), so a pattern could never name a layer
(``layers.3.mlp.experts``) or one PP stage's layers. Every rank received the
same selection.

``named_layer`` records each layer's full module name (``<prefix>.<idx>``, as
passed to ``layer_fn``) on the layer. The UVA offloader prepends it before
matching, so ``.layers.3.mlp.experts.`` style patterns select exactly those
layers on the rank that builds them. Existing patterns keep matching:
segment matching only gains leading segments. A pattern that is itself a
prefix segment (``model``, ``layers``) now matches every layer, which is what
it names. Placement only; no numerical effect.
"""

from __future__ import annotations

from collections.abc import Callable

from torch import nn

ATTR = "_merlin_offload_name"


def named_layer(layer_fn: Callable[..., nn.Module], name: str) -> nn.Module:
    """Build one layer via ``layer_fn(prefix=name)`` and record its full name."""
    layer = layer_fn(prefix=name)
    setattr(layer, ATTR, name)
    return layer


def layer_name_prefix(module: nn.Module) -> str:
    """``"<full name>."`` for a layer built by ``named_layer``, else ``""``."""
    name = getattr(module, ATTR, None)
    return f"{name}." if name else ""
