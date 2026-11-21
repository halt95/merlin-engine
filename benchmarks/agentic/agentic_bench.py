#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Agentic concurrency benchmark for any OpenAI-compatible chat server.

Runs 1, 2, 4, 8 (or any list of) agents at once on three offline workloads and reports output and
total tok/s from the server's usage fields, prefix-cache hit share, task success, step latency, wall
time and, with user-supplied prices, value per box-hour. Standard library only. See README.md.

    python benchmarks/agentic/agentic_bench.py --base-url http://127.0.0.1:8000 --agents 1,2,4,8
    python benchmarks/agentic/agentic_bench.py --self-check
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agentic_loop import ChatClient, ServerError, check_episode, run_episode  # noqa: E402
from agentic_metrics import summarize_level  # noqa: E402
from agentic_report import SCHEMA, build_report, render_markdown, write_report  # noqa: E402
from agentic_workloads import SYSTEM_PROMPTS, WORKLOADS, make_episode, self_check, tools_for  # noqa: E402

_PAD_VOCAB = ("signal", "junction", "gradient", "platform", "timetable", "freight", "viaduct", "harbour", "ledger",
              "member", "survey", "branch", "engine", "wagon", "quarry", "service", "station", "record", "board",
              "repair", "season", "route", "carriage", "contract", "dividend", "census", "bridge", "river")


def pad_text(seed: int, words: int) -> str:
    """Deterministic filler, used only to lengthen the shared system prompt (--system-pad-words)."""
    rng = random.Random(f"pad:{seed}")
    out = [rng.choice(_PAD_VOCAB) for _ in range(words)]
    return "\n".join(" ".join(out[i:i + 16]) for i in range(0, len(out), 16))


def build_system_prompt(workload: str, seed: int, agents: int, *, salt: bool = True, pad_words: int = 0) -> str:
    parts = []
    if salt:
        # A per-level first line keeps one level's prompts from hitting the prefix cache of the level before.
        parts.append(f"Run tag: {workload}-seed{seed}-agents{agents}.")
    parts.append(SYSTEM_PROMPTS[workload])
    if pad_words > 0:
        parts.append("Background notes (not needed for the task):\n" + pad_text(seed, pad_words))
    return "\n\n".join(parts)


def run_level(client: ChatClient, workload: str, agents: int, args: argparse.Namespace) -> tuple[dict, list]:
    system = build_system_prompt(workload, args.seed, agents, salt=not args.no_level_salt,
                                 pad_words=args.system_pad_words)
    tools = tools_for(workload)
    done: list[tuple] = []
    lock = threading.Lock()
    barrier = threading.Barrier(agents + 1)

    def worker(a: int) -> None:
        barrier.wait()
        for e in range(args.episodes_per_agent):
            index = a * args.episodes_per_agent + e
            ep, rec = None, None
            try:
                ep = make_episode(workload, index, args.seed, n_funcs=args.n_funcs, test_timeout_s=args.tool_timeout)
                rec = run_episode(client, ep, system, tools, max_steps=args.max_steps, max_nudges=args.max_nudges,
                                  max_tokens=args.max_tokens, temperature=args.temperature, seed=args.seed,
                                  agent=a, run_checker=False)
            except Exception as exc:  # keep the other agents running; the episode counts as failed
                rec = {"agent": a, "task_id": getattr(ep, "task_id", None), "end": "harness_error",
                       "error": repr(exc), "steps": [], "tool_counts": {}}
            with lock:
                done.append((ep, rec))

    threads = [threading.Thread(target=worker, args=(a,), daemon=True) for a in range(agents)]
    for t in threads:
        t.start()
    barrier.wait()
    t0 = time.perf_counter()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0

    episodes = []
    for ep, rec in done:  # checkers run after the clock stops
        if ep is not None and "success" not in rec and rec.get("end") != "harness_error":
            check_episode(ep, rec)
        rec.setdefault("success", False)
        rec.setdefault("score", 0.0)
        rec["workload"], rec["level_agents"] = workload, agents
        if ep is not None:
            ep.close(keep=args.keep_workspaces)
            if args.keep_workspaces and ep.workspace is not None:
                rec["workspace"] = str(ep.workspace.root)
        episodes.append(rec)
    episodes.sort(key=lambda r: (r["agent"], r.get("task_id") or ""))
    return summarize_level(workload, agents, episodes, wall, prices_from(args)), episodes


def prices_from(args: argparse.Namespace) -> dict | None:
    vals = {"price_in": args.price_in, "price_out": args.price_out, "price_cached": args.price_cached,
            "box_cost": args.box_cost}
    if all(v is None for k, v in vals.items() if k != "box_cost"):
        return None
    return vals


def parse_agents(text: str) -> list[int]:
    try:
        levels = [int(x) for x in text.split(",") if x.strip()]
    except ValueError:
        raise argparse.ArgumentTypeError(f"--agents takes a comma list of integers, got {text!r}") from None
    if not levels or any(n < 1 for n in levels):
        raise argparse.ArgumentTypeError("--agents levels must be positive integers")
    return levels


def parse_workloads(text: str) -> list[str]:
    names = [x.strip() for x in text.split(",") if x.strip()]
    bad = [n for n in names if n not in WORKLOADS]
    if bad or not names:
        raise argparse.ArgumentTypeError(f"unknown workload(s) {bad}; choose from {', '.join(WORKLOADS)}")
    return names


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--base-url", default="http://127.0.0.1:8000", help="server root or its /v1 URL")
    p.add_argument("--model", default=None, help="model id; default: the first one /v1/models lists")
    p.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"),
                   help="bearer token; default: $OPENAI_API_KEY")
    p.add_argument("--workloads", type=parse_workloads, default=list(WORKLOADS), help="comma list")
    p.add_argument("--agents", type=parse_agents, default=[1, 2, 4, 8], help="comma list of concurrency levels")
    p.add_argument("--episodes-per-agent", type=int, default=2)
    p.add_argument("--max-steps", type=int, default=40, help="chat requests per episode at most")
    p.add_argument("--max-nudges", type=int, default=2, help="replies without a tool call tolerated in a row")
    p.add_argument("--max-tokens", type=int, default=4096, help="max_tokens per request")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0, help="task selection, generated workspaces and request seed")
    p.add_argument("--n-funcs", type=int, default=6, help="stub functions per tool-loop workspace")
    p.add_argument("--system-pad-words", type=int, default=0,
                   help="deterministic filler words added to the shared system prompt, to emulate long agent prompts")
    p.add_argument("--no-level-salt", action="store_true",
                   help="drop the per-level first line, so a level may hit the prefix cache of the level before")
    p.add_argument("--tool-timeout", type=float, default=30.0, help="seconds per sandboxed test run")
    p.add_argument("--request-timeout", type=float, default=600.0, help="seconds per chat request")
    p.add_argument("--extra-body", type=json.loads, default=None,
                   help="JSON object merged into every request body, e.g. '{\"chat_template_kwargs\": {...}}'")
    p.add_argument("--warmup", type=int, default=1, help="untimed short requests before the first level")
    p.add_argument("--price-in", type=float, default=None, help="value per million uncached input tokens")
    p.add_argument("--price-cached", type=float, default=None, help="value per million cached input tokens")
    p.add_argument("--price-out", type=float, default=None, help="value per million output tokens")
    p.add_argument("--box-cost", type=float, default=None, help="cost of the serving box per hour, same currency")
    p.add_argument("--label", default=None, help="free text recorded in the report, e.g. the hardware")
    p.add_argument("--out", type=Path, default=None, help="report directory; default agentic-results/<UTC time>")
    p.add_argument("--keep-workspaces", action="store_true", help="keep the per-agent temp directories")
    p.add_argument("--self-check", action="store_true",
                   help="offline: check the bundled tasks fail as shipped and pass with their reference fixes")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_check:
        rows = self_check(n_funcs=args.n_funcs)
        for r in rows:
            print(f"{r['task']:<32} fails as shipped: {r['fails_as_shipped']!s:<5}  passes with fix: "
                  f"{r['passes_with_fix']}")
        return 0 if all(r["fails_as_shipped"] and r["passes_with_fix"] for r in rows) else 1
    if args.extra_body is not None and not isinstance(args.extra_body, dict):
        print("--extra-body must be a JSON object", file=sys.stderr)
        return 2
    try:
        client = ChatClient(args.base_url, args.model, args.api_key, args.request_timeout, args.extra_body)
        for _ in range(max(0, args.warmup)):
            client.chat([{"role": "user", "content": "Reply with the word ready."}], [], max_tokens=16,
                        temperature=0.0, seed=args.seed)
    except ServerError as e:
        print(f"server not usable: {e}", file=sys.stderr)
        return 2

    started = datetime.now(timezone.utc)
    meta = {"schema": SCHEMA, "model": client.model, "label": args.label,
            "started_utc": started.strftime("%Y-%m-%dT%H:%M:%SZ"), "seed": args.seed,
            "workloads": args.workloads, "agents": args.agents, "episodes_per_agent": args.episodes_per_agent,
            "max_steps": args.max_steps, "max_tokens": args.max_tokens, "temperature": args.temperature,
            "n_funcs": args.n_funcs, "system_pad_words": args.system_pad_words,
            "level_salt": not args.no_level_salt, "extra_body": args.extra_body,
            "prices": {"price_in": args.price_in, "price_cached": args.price_cached, "price_out": args.price_out,
                       "box_cost": args.box_cost},
            "client": {"python": platform.python_version(), "platform": platform.system()}}
    levels, episodes = [], []
    for workload in args.workloads:
        for agents in args.agents:
            print(f"[agentic] {workload} x{agents} ...", file=sys.stderr, flush=True)
            summary, eps = run_level(client, workload, agents, args)
            levels.append(summary)
            episodes += eps
            print(f"[agentic] {workload} x{agents}: success {summary['successes']}/{summary['episodes']}, "
                  f"output {summary['output_tok_s'] or 0:.1f} tok/s, wall {summary['wall_s']:.1f} s",
                  file=sys.stderr, flush=True)
    meta["finished_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    report = build_report(meta, levels, episodes)
    out = args.out or Path("agentic-results") / started.strftime("%Y%m%dT%H%M%SZ")
    jp, mp = write_report(report, out)
    print(render_markdown(report))
    print(f"[agentic] wrote {jp} and {mp}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
