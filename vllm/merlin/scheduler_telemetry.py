# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Bounded scheduler policy counters, independent of the host-tier metrics."""

from __future__ import annotations

from typing import Any

POLICY_COUNTER_FIELDS = {
    "prefill_runs": (
        "runs",
        "windows_cut",
        "long_chunks",
        "long_deferred",
        "short_first",
    ),
    "decode_first": ("reserved_tokens", "shrunk_chunks", "kept_prefill"),
}
METRIC_NAME = "vllm:merlin_scheduler_policy_events_total"


class PolicyCounterDeltas:
    """Drain fixed cumulative fields without touching their log snapshots."""

    def __init__(self) -> None:
        self.previous: dict[tuple[str, str], int] = {}

    def snapshot(
        self, prefill_runs: Any, decode_first: Any
    ) -> dict[str, dict[str, int]]:
        sources = {
            "prefill_runs": prefill_runs.stats if prefill_runs is not None else None,
            "decode_first": decode_first,
        }
        out: dict[str, dict[str, int]] = {}
        for policy, fields in POLICY_COUNTER_FIELDS.items():
            source = sources[policy]
            if source is None:
                continue
            for event in fields:
                value = getattr(source, event)
                key = (policy, event)
                delta = value - self.previous.get(key, 0)
                if delta > 0:
                    out.setdefault(policy, {})[event] = delta
                self.previous[key] = value
        return out


class PolicyCounterProm:
    def __init__(self, counter_cls, labelnames, per_engine_labelvalues) -> None:
        self.counter = counter_cls(
            name=METRIC_NAME,
            documentation="Scheduler policy event deltas; policies are opt-in.",
            labelnames=labelnames + ["policy", "event"],
        )
        self.per_engine_labelvalues = per_engine_labelvalues

    def observe(self, deltas: dict[str, dict[str, int]], engine_idx: int) -> None:
        labels = self.per_engine_labelvalues[engine_idx]
        # Bound label cardinality even if a payload contains unknown keys.
        for policy, fields in POLICY_COUNTER_FIELDS.items():
            values = deltas.get(policy, {})
            for event in fields:
                value = values.get(event, 0)
                if value > 0:
                    self.counter.labels(*labels, policy, event).inc(value)
