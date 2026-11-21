# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Which builder sizes an MLA layer's chunked-prefill workspace.

``MLAAttention.chunked_prefill_workspace_size`` sizes the profile run's dummy
``kv_b_proj`` output. The common builder reserves a 64K-token buffer for the
dense prefill-context gather. A backend whose builder declares
``never_gathers_prefill_context = True`` (the Hy4 sparse backend) never does
that gather, so its own ``determine_chunked_prefill_workspace_size`` sizes the
buffer instead. Tests: tests/merlin_cpu/test_mla_backend_workspace.py
(development tree).
"""

from typing import Any


def workspace_builder(layer: Any, default: Any) -> Any:
    """The layer's backend builder when it declares it never gathers a dense
    prefill context, else ``default`` (also when the layer has no backend or
    the backend has no builder)."""
    get_builder = getattr(getattr(layer, "attn_backend", None), "get_builder_cls", None)
    if get_builder is None:
        return default
    try:
        cls = get_builder()
    except NotImplementedError:
        return default
    if getattr(cls, "never_gathers_prefill_context", False):
        return cls
    return default
