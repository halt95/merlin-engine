# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Ring-row tag fix: worker side of the ring-row tag clear.

The scheduler reports every newly claimed ring (``CircularBufferSpec``) block
in ``SchedulerOutput.new_ring_block_ids_to_clear``, whether or not KV zeroing
is on. Before the step's forward, the runner hands them to every layer that
keeps a tagged ring (``clear_ring_row_tags``; today ``QSAKeyStateCache``),
which zeroes only the tag word of each row. A zero tag never validates, so the
new owner cannot pool a row it did not commit itself, whatever the previous
occupant (or another group that held the block) left there.

ORDER. The runners call this right after the KV zeroer and before the CoW
copies and the forward, on the same stream, inside the same execute_model as
the claim's first step. So: previous owner's work done (deferred free) ->
claim -> tag clear -> new owner's first read/write.
"""

from collections.abc import Iterable, Mapping

import torch


def ring_tag_layers(static_forward_context: Mapping[str, object]) -> list:
    """Layers of this worker that keep a tagged ring."""
    return [
        layer
        for layer in static_forward_context.values()
        if callable(getattr(layer, "clear_ring_row_tags", None))
    ]


def clear_ring_row_tags(
    layers: Iterable,
    block_ids: list[int],
    device: torch.device,
) -> int:
    """Zero the tag word of every row of ``block_ids`` in every layer.

    Returns the number of layers cleared. Raises (fail loud) on a block id
    outside a layer's pool: an index_fill_ with a bad index is a device-side
    assert, far from its cause."""
    if not block_ids:
        return 0
    layers = list(layers)
    if not layers:
        return 0
    lo, hi = min(block_ids), max(block_ids)
    index = torch.tensor(block_ids, dtype=torch.long, device=device)
    for layer in layers:
        tags = getattr(layer, "ring_tag_cache", None)
        if tags is None:
            raise RuntimeError(
                f"ring tag clear: layer {getattr(layer, 'prefix', layer)!r} has "
                "no bound ring_tag_cache"
            )
        if lo < 0 or hi >= tags.shape[0]:
            raise ValueError(
                f"ring tag clear: block ids [{lo}, {hi}] outside the pool of "
                f"{tags.shape[0]} blocks ({getattr(layer, 'prefix', layer)!r})"
            )
        layer.clear_ring_row_tags(index)
    return len(layers)
