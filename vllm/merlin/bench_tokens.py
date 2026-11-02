# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Token-weighted per-second output buckets for vllm bench serve.

The bench records one timestamp per streamed event, but on MTP / spec-decode
lanes one event carries several tokens (Flash-Next: 2.3-3.1), so counting 1 per
event reads peak and steady-state output tok/s 2-3x low. The stream gives no
per-event token count, so each event is weighted by the request's mean tokens
per event (``output_tokens / events``); the weights sum to ``output_tokens``,
the same total ``calculate_metrics`` takes from ``usage.completion_tokens``.
"""

from __future__ import annotations

from collections.abc import MutableSequence, Sequence


def event_weight(output_tokens: int | None, n_events: int) -> float:
    """Tokens credited to each streamed event; 1 when usage gave no count."""
    if not output_tokens or n_events <= 0:
        return 1.0
    return output_tokens / n_events


def add_event_tokens(
    buckets: MutableSequence[float],
    event_times: Sequence[float],
    output_tokens: int | None,
    min_start_time: float,
) -> None:
    """Add one request's streamed events to the per-second ``buckets``."""
    weight = event_weight(output_tokens, len(event_times))
    for event_time in event_times:
        second_bucket = int(event_time - min_start_time)
        if 0 <= second_bucket < len(buckets):
            buckets[second_bucket] += weight
