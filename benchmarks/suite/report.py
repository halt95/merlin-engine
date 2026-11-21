# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""The report: a JSON document with the raw (unrounded) medians and a markdown
document in the layout of the published bench cards. Both are computed from the
per-send records alone; figures are rounded when the markdown is rendered,
never before."""

from __future__ import annotations

import math
import statistics

try:
    from . import protocol as P
except ImportError:  # run as a script from this directory
    import protocol as P

SCHEMA = "merlin-bench-suite-report/1"
MEDIAN_KEYS = ("decode_tok_s_T", "median_event_interval_ms", "tokens_per_event", "ttft_s", "prefill_tok_s")


def median(values):
    v = [x for x in values if isinstance(x, (int, float)) and not isinstance(x, bool)]
    return statistics.median(v) if v else None


def wilson(k: int, n: int, z: float = 1.959964):
    """Wilson score interval for k successes of n (95 % by default)."""
    if n <= 0:
        return None
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [max(0.0, centre - half), min(1.0, centre + half)]


def _cell(rows, depth, mode):
    rs = [r for r in rows if r.get("depth") == depth and r.get("mode") == mode]
    ok = [r for r in rs if r.get("valid")]
    cts = sorted(r["completion_tokens"] for r in rs if isinstance(r.get("completion_tokens"), int))
    return {
        "depth": depth, "mode": mode, "sends": len(rs), "valid": len(ok),
        "prompt_tokens": sorted({r["prompt_tokens"] for r in rs if r.get("prompt_tokens")}),
        "median": {k: median(r.get(k) for r in ok) for k in MEDIAN_KEYS},
        "send_decode_tok_s": [r.get("decode_tok_s_T") for r in rs],
        "completion_tokens": [cts[0], cts[-1]] if cts else None,
        "problems": sorted({p for r in rs for p in r.get("problems", [])}),
        "thinking_observed_sends": sum(1 for r in rs if r.get("thinking_observed")),
    }


def _exact_cells(rows):
    out = []
    for mode in ("on", "off"):
        rs = [r for r in rows if r.get("row") == "exact262" and r.get("mode") == mode]
        if not rs:
            continue
        rs.sort(key=lambda r: r.get("rep", 0))
        dec = [r.get("decode_tok_s_T") for r in rs]
        out.append({
            "mode": mode, "sends": len(rs), "valid": sum(1 for r in rs if r.get("valid")),
            "prompt_tokens": [r.get("prompt_tokens") for r in rs],
            "exact": all(r.get("exact") for r in rs),
            "median": {k: median(r.get(k) for r in rs) for k in MEDIAN_KEYS},
            "send_decode_tok_s": dec,
            "decode_tok_s_range": [min(d for d in dec if d is not None), max(d for d in dec if d is not None)]
            if any(d is not None for d in dec) else None,
            "cold_ttft_s": rs[0].get("ttft_s"),
            "cold_prefill_tok_s": rs[0].get("prefill_tok_s"),
            "cached_tokens": [r.get("cached_tokens") for r in rs],
            "needle_found": [r.get("needle_found") for r in rs] if mode == "off" else None,
            "problems": sorted({p for r in rs for p in r.get("problems", [])}),
            "thinking_observed_sends": sum(1 for r in rs if r.get("thinking_observed")),
        })
    return out


def _needle_summary(rows):
    out = []
    depths = sorted({r["depth_target"] for r in rows if r.get("row") == "needle"}, reverse=True)
    for depth in depths:
        rs = [r for r in rows if r.get("row") == "needle" and r.get("depth_target") == depth]
        if not rs:
            continue
        ok = [r for r in rs if r.get("valid")]
        exact = sum(1 for r in ok if r.get("exact_pass"))
        pts = [r["prompt_tokens"] for r in rs if isinstance(r.get("prompt_tokens"), int)]
        out.append({
            "depth": depth, "planned": len(rs), "completed_valid": len(ok),
            "completion_yield": len(ok) / len(rs), "exact_pass": exact,
            "wilson95": wilson(exact, len(ok)),
            "legacy_pass": sum(1 for r in ok if r.get("legacy_pass")),
            "incomplete": len(rs) - len(ok),
            "prompt_tokens_min": min(pts) if pts else None, "prompt_tokens_max": max(pts) if pts else None,
            "thinking_observed_cases": sum(1 for r in rs if r.get("thinking_observed")),
            "reasons": sorted({x for r in rs for x in r.get("problems", [])}),
        })
    return out


def build_report(meta: dict, rows: list[dict]) -> dict:
    """The JSON report. ``meta``: created_utc, command, profile (resolve_profile),
    server (base, model, models), effort, sections, boot, refused/warnings."""
    measured = [r for r in rows if r.get("row") in ("decode", "decode_off", "prefill", "ttft", "exact262", "needle")]
    invalid = [r for r in measured if not r.get("valid")]
    dec = [r for r in rows if r.get("row") in ("decode", "decode_off")]
    decode = {"cells": [_cell(dec, d, m) for m in ("on", "off") for d in sorted({r["depth"] for r in dec})]
              if dec else []}
    pre = [r for r in rows if r.get("row") == "prefill"]
    exact = _exact_cells(rows)
    prefill_rows = [{
        "depth": r["depth"], "prompt_tokens": r.get("prompt_tokens"), "prefill_tok_s": r.get("prefill_tok_s"),
        "ttft_s": r.get("ttft_s"), "valid": r.get("valid"), "source": "prefill send"} for r in pre]
    for c in exact:
        if c["mode"] == "on" and c["cold_ttft_s"]:
            prefill_rows.append({
                "depth": c["prompt_tokens"][0], "prompt_tokens": c["prompt_tokens"][0],
                "prefill_tok_s": c["cold_prefill_tok_s"], "ttft_s": c["cold_ttft_s"],
                "valid": c["valid"] == c["sends"], "source": "exact 262K cold thinking-on send"})
    tt = [r for r in rows if r.get("row") == "ttft"]
    ttft_cells = [_cell(tt, d, "on") for d in sorted({r["depth"] for r in tt})]
    # thinking, as the server's responses showed it
    on = [r for r in measured if (r.get("chat_template_kwargs") or {}).get("enable_thinking") is True]
    off = [r for r in measured if (r.get("chat_template_kwargs") or {}).get("enable_thinking") is False]
    needle_rows = [r for r in rows if r.get("row") == "needle"]
    off += needle_rows
    thinking = {
        "effort_sent": meta.get("effort"),
        "thinking_on_kwargs": P.thinking_kwargs(meta.get("effort") or P.DEFAULT_EFFORT),
        "thinking_off_kwargs": P.NO_THINKING,
        "on_sends": len(on), "on_sends_with_reasoning_in_response": sum(1 for r in on if r.get("thinking_observed")),
        "off_sends": len(off), "off_sends_with_reasoning_in_response": sum(1 for r in off if r.get("thinking_observed")),
    }
    warnings = list(meta.get("warnings") or [])
    if thinking["on_sends"] and not thinking["on_sends_with_reasoning_in_response"]:
        warnings.append("no thinking-on response showed reasoning (no reasoning field and no reasoning "
                        "token count): the server may not separate reasoning, or thinking was not applied")
    if thinking["off_sends_with_reasoning_in_response"]:
        warnings.append("a thinking-off response showed reasoning: the thinking switch was not honoured")
    return {
        "schema": SCHEMA, "suite_version": P.SUITE_VERSION, "created_utc": meta.get("created_utc"),
        "command": meta.get("command"), "profile": meta.get("profile"), "server": meta.get("server"),
        "thinking": thinking,
        "protocol": {
            "request_depths": {str(k): v for k, v in P.REQ.items()}, "sends": P.SENDS,
            "max_tokens_thinking_on": P.MAX_TOKENS, "max_tokens_thinking_off": P.OFF_MAX_TOKENS,
            "prefill_max_tokens": P.PREFILL_MAX_TOKENS, "temperature": 0, "cache_salt": "per request",
            "exact_target_prompt_tokens": P.EXACT_TARGET, "exact_out_tokens": P.EXACT_OUT_TOKENS,
            "exact_reps": P.EXACT_REPS, "needle_position": P.NEEDLE_POSITION,
            "needle_max_tokens": P.NEEDLE_MAX_TOKENS, "needle_seed": P.NEEDLE_SEED,
            "median_of": P.SENDS, "rounding": "at render time, from the raw medians in this file",
        },
        "sections_run": list(meta.get("sections") or []),
        "sections": {
            "decode": decode, "exact262": {"cells": exact}, "prefill": {"rows": prefill_rows},
            "ttft": {"cells": ttft_cells}, "needle": {"depths": _needle_summary(rows)},
        },
        "invalid_sends": len(invalid),
        "warnings": warnings,
    }


# -- markdown -------------------------------------------------------------------

def _f(v, spec="%.1f"):
    return spec % v if isinstance(v, (int, float)) and not isinstance(v, bool) else "n/a"


def _n(v):
    return "{:,}".format(v) if isinstance(v, int) else "n/a"


def _pt(c):
    pts = c.get("prompt_tokens") or []
    return ", ".join(_n(p) for p in pts) if pts else "n/a"


def _cells_table(cells, ttft_only=False):
    L = []
    if ttft_only:
        L += ["| prompt tokens | TTFT s | prefill tok/s | valid |", "|---|---|---|---|"]
        for c in cells:
            m = c["median"]
            L.append("| %s | %s | %s | %d/%d |" % (_pt(c), _f(m["ttft_s"], "%.2f"), _f(m["prefill_tok_s"], "%.0f"),
                                                   c["valid"], c["sends"]))
        return L
    L += ["| prompt tokens | thinking | decode tok/s | ITL ms | tokens/event | TTFT s | valid | completion tokens |",
          "|---|---|---|---|---|---|---|---|"]
    for c in cells:
        m = c["median"]
        extra = "" if c["valid"] == c["sends"] else " (" + "; ".join(c["problems"])[:120] + ")"
        ct = c.get("completion_tokens")
        L.append("| %s | %s | %s | %s | %s | %s | %d/%d%s | %s |" % (
            _pt(c), c["mode"], _f(m["decode_tok_s_T"]), _f(m["median_event_interval_ms"], "%.2f"),
            _f(m["tokens_per_event"], "%.2f"), _f(m["ttft_s"], "%.2f"), c["valid"], c["sends"], extra,
            "%s-%s" % (ct[0], ct[1]) if ct else "n/a"))
    return L


def render_markdown(rep: dict) -> str:
    prof, srv, th = rep.get("profile") or {}, rep.get("server") or {}, rep["thinking"]
    S = rep["sections"]
    L = [f"# {prof.get('name', '?')} - Merlin bench suite report, {(rep.get('created_utc') or '')[:10]}", ""]
    L += [
        "> **Provenance.** Run with `benchmarks/suite` against `%s`, model `%s`%s. Streams are client-timed from "
        "this machine; the server was not restarted for this run and its build, GPUs and load are whatever they "
        "were. See `benchmarks/suite/README.md` for the protocol and what differs from a reference-host card run."
        % (srv.get("base", "?"), srv.get("model", "?"),
           ", context %s tokens" % _n(srv["max_model_len"]) if srv.get("max_model_len") else ""),
        "> Profile row `%s` (%s)." % (prof.get("name", "?"), prof.get("known", "?")),
        "> Thinking on: effort `%s` sent with every thinking-on request (%d of %d responses showed reasoning); "
        "thinking off: `enable_thinking=false` (%d of %d responses showed reasoning)."
        % (th["effort_sent"], th["on_sends_with_reasoning_in_response"], th["on_sends"],
           th["off_sends_with_reasoning_in_response"], th["off_sends"]),
        "",
    ]
    for w in rep.get("warnings") or []:
        L.append("> **Warning.** " + w)
    if rep.get("warnings"):
        L.append("")
    if S["decode"]["cells"]:
        L += ["## 1. Single-stream decode ladder", "",
              "T=0, a needle prompt at depth with the image, every send a cold prefill (cache salt); thinking on "
              "<= 256 tokens, thinking off = the answer-plus-summary task (natural stop, <= 1,024 tokens); 3 sends "
              "per cell, each cell the median of the valid sends. tok/s = tokens after the first streamed event / "
              "time from the first to the last event. With MTP read the median event interval (ITL) together with "
              "tokens per event.", ""]
        L += _cells_table(S["decode"]["cells"]) + [""]
    if S["exact262"]["cells"]:
        L += ["## 2. Exact 262K", "",
              "Exactly %s prompt tokens with a %d-token output cap, both thinking modes, one stream, %d sends with "
              "one cache salt (send 1 cold); each send uses a different prompt text. Decode, ITL and tokens per "
              "event are the median of the sends; TTFT is send 1's." % (_n(P.EXACT_TARGET), P.EXACT_OUT_TOKENS, P.EXACT_REPS),
              "", "| prompt tokens | thinking | tok/s (median) | sends, tok/s | ITL ms | tokens/event | cold TTFT s | exact |",
              "|---|---|---|---|---|---|---|---|"]
        for c in S["exact262"]["cells"]:
            m, rg = c["median"], c["decode_tok_s_range"]
            L.append("| %s | %s | %s | %s | %s | %s | %s | %s |" % (
                _n(c["prompt_tokens"][0]), c["mode"], _f(m["decode_tok_s_T"]),
                "%s-%s" % (_f(rg[0]), _f(rg[1])) if rg else "n/a", _f(m["median_event_interval_ms"], "%.2f"),
                _f(m["tokens_per_event"], "%.2f"), _f(c["cold_ttft_s"]), "yes" if c["exact"] else "NO"))
        L.append("")
        for c in S["exact262"]["cells"]:
            L.append("Thinking %s: tokens served from the prefix cache per send: %s." % (
                c["mode"], ", ".join(_n(x or 0) for x in c["cached_tokens"])))
        L.append("")
    if S["prefill"]["rows"]:
        L += ["## 3. Cold prefill", "", "| prompt tokens | prefill | TTFT s |", "|---|---|---|"]
        for r in S["prefill"]["rows"]:
            L.append("| %s | %s tok/s | %s |" % (_n(r["prompt_tokens"]), _f(r["prefill_tok_s"], "%.0f"),
                                                _f(r["ttft_s"], "%.2f")))
        L += ["", "One cold send per depth (%d tokens out at 10K and 100K). A 262K row, when present, is the cold "
              "thinking-on send of section 2: prompt tokens / time to first token." % P.PREFILL_MAX_TOKENS, ""]
    if S["ttft"]["cells"]:
        L += ["## 4. Time to first token", "",
              "%d cold sends per depth, %d tokens out, thinking on: the median time from the request to the first "
              "streamed token event, and the prompt tokens per second of that interval." % (P.SENDS, P.PREFILL_MAX_TOKENS),
              ""]
        L += _cells_table(S["ttft"]["cells"], ttft_only=True) + [""]
    if S["needle"]["depths"]:
        L += ["## 5. Needle panel", "",
              "Single-occurrence needles (the code appears once in the prompt, %d %% into the haystack), an image in "
              "the prompt, thinking off, greedy, a fresh cache salt per request, sent one at a time. A case counts "
              "when it completes validly (HTTP 200, usage present, the depth reached, no cache hit); exact = the "
              "`ANSWER1` line equals the code." % round(P.NEEDLE_POSITION * 100), "",
              "| depth | planned | completed-valid | completion yield | exact pass (of valid) | Wilson 95 % CI | "
              "legacy pass | incomplete/invalid | prompt tokens (min-max) |", "|---|---|---|---|---|---|---|---|---|"]
        line = []
        for d in S["needle"]["depths"]:
            ci = d["wilson95"]
            L.append("| %s | %d | %d | %s | %d | %s | %d | %d | %s-%s |" % (
                _n(d["depth"]), d["planned"], d["completed_valid"], _f(100 * d["completion_yield"], "%.0f") + " %",
                d["exact_pass"], "%s-%s %%" % (_f(100 * ci[0]), _f(100 * ci[1])) if ci else "n/a",
                d["legacy_pass"], d["incomplete"], _n(d["prompt_tokens_min"]), _n(d["prompt_tokens_max"])))
            line.append("%d/%d at %dK" % (d["exact_pass"], d["completed_valid"], d["depth"] // 1000))
        L += ["", "Needle panel: " + ", ".join(line) + " found exactly.", ""]
    L += ["## 6. Coverage of this run", "", "| section | status |", "|---|---|"]
    for sec in ("decode", "prefill", "ttft", "exact262", "needle"):
        L.append("| %s | %s |" % (sec, "run" if sec in rep["sections_run"] else "not run"))
    L += ["", "Invalid sends in this run: %d (listed with their reasons in `rows.jsonl`)." % rep["invalid_sends"], ""]
    return "\n".join(L)


def write_reports(out_dir, report: dict) -> tuple:
    import json
    from pathlib import Path

    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    jp, mp = d / "report.json", d / "report.md"
    jp.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    mp.write_text(render_markdown(report), encoding="utf-8", newline="\n")
    return jp, mp
