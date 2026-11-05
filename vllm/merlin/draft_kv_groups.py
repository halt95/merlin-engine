# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""The KV cache groups that hold a DFlash drafter's layers.

``kv_cache_planning.get_global_kv_cache_groups`` flags the groups whose
prefix hits drop the EAGLE trailing unit (``KVCacheGroupSpec.is_eagle_group``)
from the proposer's draft-layer tags (``KVCacheSpec.is_draft``, vllm#61455).
This module is the fallback when no layer carries the tag: without it the
annotator finds drafters only by a spec marker (DSpark) or by position for
MTP. A DFlash drafter matches neither, so every DFlash boot falls
through to the coordinator's flag-all fallback
(``KVCacheCoordinator.eagle_group_ids`` = every group) and logs "no KV cache
group could be identified as the draft model's". On a hybrid target that
marks the target's own full-attention and recurrent groups as draft groups
too: their lookups drop a trailing unit they do not need, and the host tier
sizes restore units with proof pages the target never reads.

A DFlash drafter's KV layers are identifiable by construction:

* they are registered after every target layer (the proposer loads the
  drafter after the target, so they trail ``kv_cache_spec``, which keeps
  registration order);
* they are named ``layers.{i + N}`` with ``N`` the target's total hidden layer
  count (``DFlashQwen3ForCausalLM`` passes ``start_layer_id=N``,
  vllm/model_executor/models/qwen3_dflash.py), so every draft index is ``>= N``
  while every target decoder layer index is ``< N``.

The trailing run of layers whose ``layers.<i>`` index is ``>= N`` is the
drafter's. Every group holding one of them is a draft group; no other group is
flagged. When the names do not follow the convention (no trailing run, or
every layer would qualify) nothing is flagged and the caller's other rules and
the coordinator fallback apply as before.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

_LAYER_INDEX = re.compile(r"(?:^|\.)layers\.(\d+)(?=\.|$)")


def layer_index(name: str) -> int | None:
    """The decoder index ``i`` of the last ``layers.<i>`` segment, or None."""
    found = _LAYER_INDEX.findall(name)
    return int(found[-1]) if found else None


def dflash_draft_layer_names(
    vllm_config: Any, kv_cache_spec: Mapping[str, Any]
) -> list[str]:
    """The DFlash drafter's KV layers in ``kv_cache_spec`` (registration
    order), or [] when speculative decoding is not DFlash or the layers cannot
    be told apart."""
    spec_config = getattr(vllm_config, "speculative_config", None)
    if spec_config is None or getattr(spec_config, "method", None) != "dflash":
        return []
    model_config = getattr(vllm_config, "model_config", None)
    total = getattr(model_config, "get_total_num_hidden_layers", None)
    if total is None:
        return []
    try:
        num_target_layers = int(total())
    except (AttributeError, TypeError, ValueError):
        return []
    names = list(kv_cache_spec)
    draft: list[str] = []
    for name in reversed(names):
        index = layer_index(name)
        if index is None or index < num_target_layers:
            break
        draft.append(name)
    if not draft or len(draft) == len(names):
        return []
    draft.reverse()
    return draft


def flag_dflash_draft_groups(
    vllm_config: Any,
    kv_cache_spec: Mapping[str, Any],
    kv_cache_groups: Sequence[Any],
) -> bool:
    """Flag (``is_eagle_group``) every group holding a DFlash draft layer.
    Returns whether any group was flagged."""
    draft = set(dflash_draft_layer_names(vllm_config, kv_cache_spec))
    if not draft:
        return False
    flagged = False
    for group in kv_cache_groups:
        if draft.intersection(group.layer_names):
            group.is_eagle_group = True
            flagged = True
    return flagged


__all__ = [
    "dflash_draft_layer_names",
    "flag_dflash_draft_groups",
    "layer_index",
]
