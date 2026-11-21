# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""JSON report and markdown tables for the agentic benchmark."""

from __future__ import annotations

import json
from pathlib import Path

SCHEMA = "merlin-agentic-bench/1"
TITLES = {"tool_loop": "Synthetic tool loop", "repo_fix": "Repo bug fixes", "research": "Read-heavy research"}


def build_report(meta: dict, levels: list[dict], episodes: list[dict]) -> dict:
    return {"schema": SCHEMA, "meta": meta, "levels": levels, "episodes": episodes}


def _num(v, digits: int = 0) -> str:
    if v is None:
        return "n/a"
    return f"{v:,.{digits}f}"


def _pct(v) -> str:
    return "n/a" if v is None else f"{100 * v:.0f}%"


def _ctx(v) -> str:
    if v is None:
        return "n/a"
    return f"~{v / 1000:.1f}K" if v >= 1000 else f"~{v:.0f}"


def _value(level: dict, key: str):
    v = level.get("value") or {}
    return v.get(key)


def render_markdown(report: dict) -> str:
    meta, levels = report["meta"], report["levels"]
    has_value = any(level.get("value") for level in levels)
    has_net = any(_value(level, "net_per_box_hour") is not None for level in levels)
    lines = ["# Agentic concurrency benchmark", ""]
    lines.append(f"- model: `{meta.get('model')}`" + (f" ({meta['label']})" if meta.get("label") else ""))
    lines.append(f"- started (UTC): {meta.get('started_utc')}; seed {meta.get('seed')}; "
                 f"{meta.get('episodes_per_agent')} episode(s) per agent; temperature {meta.get('temperature')}; "
                 f"max {meta.get('max_steps')} steps per episode")
    lines.append("- tok/s use the server's usage fields over each level's wall time; prefix hit = cached prompt "
                 "tokens / prompt tokens (n/a: the server did not report cached_tokens)")
    if has_value:
        p = next(level["value"]["prices_per_million"] for level in levels if level.get("value"))
        lines.append(f"- value uses the prices given on the command line, per million tokens: input {p['input']}, "
                     f"cached input {p['cached_input']}, output {p['output']}; box cost "
                     f"{meta.get('prices', {}).get('box_cost')} per hour")
    lines.append("")

    workloads = list(dict.fromkeys(level["workload"] for level in levels))
    lines += ["## Summary", ""]
    head = ["Workload (avg context)", "Agents", "Success", "Output tok/s", "Total tok/s", "Prefix hit"]
    if has_value:
        head.append("Value/box-h")
    if has_net:
        head += ["Net/box-h", "Margin"]
    lines += ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for w in workloads:
        rows = sorted((level for level in levels if level["workload"] == w), key=lambda r: r["agents"])
        steps = sum(r["steps"] for r in rows)
        ctx = sum(r["prompt_tokens"] for r in rows) / steps if steps else None

        def col(fn):
            return " / ".join(fn(r) for r in rows)
        cells = [f"{TITLES.get(w, w)} ({_ctx(ctx)})", col(lambda r: str(r["agents"])),
                 col(lambda r: _pct(r["success_rate"])), col(lambda r: _num(r["output_tok_s"])),
                 col(lambda r: _num(r["total_tok_s"])), col(lambda r: _pct(r["prefix_hit_share"]))]
        if has_value:
            cells.append(col(lambda r: _num(_value(r, "value_per_box_hour"), 2)))
        if has_net:
            cells += [col(lambda r: _num(_value(r, "net_per_box_hour"), 2)), col(lambda r: _pct(_value(r, "margin")))]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    lines += ["## Per level", ""]
    head = ["Workload", "Agents", "Episodes", "Success", "Score", "Steps", "Avg context", "Output tok/s",
            "Total tok/s", "Prefix hit", "Step mean s", "Step p95 s", "Wall s", "Errors"]
    lines += ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in levels:
        lines.append("| " + " | ".join([
            r["workload"], str(r["agents"]), str(r["episodes"]), f"{r['successes']}/{r['episodes']}",
            _num(r["mean_score"], 2), str(r["steps"]), _num(r["avg_prompt_tokens_per_step"]),
            _num(r["output_tok_s"], 1), _num(r["total_tok_s"], 1), _pct(r["prefix_hit_share"]),
            _num(r["step_latency_mean_s"], 2), _num(r["step_latency_p95_s"], 2), _num(r["wall_s"], 1),
            str(r["server_errors"])]) + " |")
    notes = sorted({n for r in levels for n in ((r.get("value") or {}).get("notes") or [])})
    partial = [r for r in levels if 0 < r["cached_tokens_reported_share"] < 1]
    if notes or partial:
        lines += ["", "Notes:", ""]
        lines += [f"- {n}" for n in notes]
        lines += [f"- {r['workload']} x{r['agents']}: cached_tokens reported on "
                  f"{_pct(r['cached_tokens_reported_share'])} of steps" for r in partial]
    return "\n".join(lines) + "\n"


def write_report(report: dict, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    jp, mp = out_dir / "report.json", out_dir / "report.md"
    jp.write_text(json.dumps(report, indent=2, sort_keys=False) + "\n", encoding="utf-8", newline="\n")
    mp.write_text(render_markdown(report), encoding="utf-8", newline="\n")
    return jp, mp
