# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Metric arithmetic for one (workload, agents) level. Pure functions; no I/O.

Token counts come only from the server's ``usage`` fields. Throughput is divided by the level's
wall time (all agents start together; the clock stops when the last agent finishes, before the
checkers run). The prefix-hit share is cached prompt tokens over prompt tokens, counted only over
steps whose response reported ``prompt_tokens_details.cached_tokens``.
"""

from __future__ import annotations

import math
from collections import Counter


def percentile(values: list[float], q: float) -> float | None:
    """Nearest-rank percentile (q in 0..100); None for an empty list."""
    if not values:
        return None
    s = sorted(values)
    k = max(1, math.ceil(q * len(s) / 100.0))
    return s[min(k, len(s)) - 1]


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def value_per_box_hour(*, prompt_tokens: int, completion_tokens: int, cached_tokens: int | None,
                       wall_s: float, agents: int, price_in: float | None, price_out: float | None,
                       price_cached: float | None, box_cost: float | None) -> dict | None:
    """Value of the tokens served per box-hour at user-supplied prices (per million tokens).

    Returns None when no price was given. A missing input or output price counts as 0; a missing cached
    price bills cached tokens at the input price. Without cached_tokens from the server every prompt
    token is billed as uncached.
    """
    if price_in is None and price_out is None and price_cached is None:
        return None
    notes = []
    p_in = price_in or 0.0
    p_out = price_out or 0.0
    if price_in is None:
        notes.append("no --price-in: input tokens valued at 0")
    if price_out is None:
        notes.append("no --price-out: output tokens valued at 0")
    if price_cached is None:
        p_cached = p_in
        notes.append("no --price-cached: cached input valued at the input price")
    else:
        p_cached = price_cached
    cached = cached_tokens or 0
    if cached_tokens is None:
        notes.append("server reported no cached_tokens: all input valued as uncached")
    uncached = max(0, prompt_tokens - cached)
    v_in, v_cached, v_out = uncached * p_in / 1e6, cached * p_cached / 1e6, completion_tokens * p_out / 1e6
    total = v_in + v_cached + v_out
    hours = wall_s / 3600.0
    per_hour = total / hours if hours > 0 else None
    net = per_hour - box_cost if per_hour is not None and box_cost is not None else None
    margin = net / per_hour if net is not None and per_hour else None
    return {"value_total": total, "value_per_box_hour": per_hour,
            "value_per_agent_hour": per_hour / agents if per_hour is not None and agents else None,
            "box_cost_per_hour": box_cost, "net_per_box_hour": net, "margin": margin,
            "cached_share_of_value": v_cached / total if total else None,
            "prices_per_million": {"input": p_in, "cached_input": p_cached, "output": p_out}, "notes": notes}


def summarize_level(workload: str, agents: int, episodes: list[dict], wall_s: float,
                    prices: dict | None = None) -> dict:
    steps = [s for e in episodes for s in e.get("steps", [])]
    prompt = sum(s["prompt_tokens"] for s in steps)
    completion = sum(s["completion_tokens"] for s in steps)
    reported = [s for s in steps if s.get("cached_tokens") is not None]
    if reported:
        cached = sum(s["cached_tokens"] for s in reported)
        prompt_reported = sum(s["prompt_tokens"] for s in reported)
        hit = cached / prompt_reported if prompt_reported else None
    else:
        cached, hit = None, None
    lat = [s["latency_s"] for s in steps]
    n_ep = len(episodes)
    successes = sum(1 for e in episodes if e.get("success"))
    out = {
        "workload": workload,
        "agents": agents,
        "episodes": n_ep,
        "successes": successes,
        "success_rate": successes / n_ep if n_ep else None,
        "mean_score": mean([float(e.get("score") or 0.0) for e in episodes]),
        "steps": len(steps),
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
        "cached_tokens": cached,
        "wall_s": wall_s,
        "output_tok_s": completion / wall_s if wall_s > 0 else None,
        "total_tok_s": (prompt + completion) / wall_s if wall_s > 0 else None,
        "prefix_hit_share": hit,
        "cached_tokens_reported_share": len(reported) / len(steps) if steps else 0.0,
        "step_latency_mean_s": mean(lat),
        "step_latency_p95_s": percentile(lat, 95),
        "avg_prompt_tokens_per_step": prompt / len(steps) if steps else None,
        "max_prompt_tokens": max((s["prompt_tokens"] for s in steps), default=None),
        "server_errors": sum(1 for e in episodes if e.get("end") == "server_error"),
        "episode_ends": dict(Counter(e.get("end") for e in episodes)),
    }
    if prices:
        out["value"] = value_per_box_hour(prompt_tokens=prompt, completion_tokens=completion, cached_tokens=cached,
                                          wall_s=wall_s, agents=agents, **prices)
    else:
        out["value"] = None
    return out
