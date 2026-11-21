#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Run the Merlin bench-card sections against an OpenAI-compatible server.

    python benchmarks/suite/run.py --base http://127.0.0.1:8000 \\
        --profile fn-quad-3090 --section decode
    python benchmarks/suite/run.py --profile 27b-tp4-3090 --all --out ./cards/27b-tp4

Sections: decode (the single-stream ladder, thinking on and off), prefill (cold
10K and 100K), ttft (time to first token), exact262 (exactly 261,888 prompt
tokens + 128 out) and needle (the single-occurrence needle panel). ``--all``
runs the five in that order. A report is written to ``--out``: ``report.json``
(raw medians), ``report.md`` (the published cards' layout) and ``rows.jsonl``
(one record per send).

Exit codes: 0 every send valid, 3 some send invalid (the report is still
written), 2 refused (the server lacks something the protocol needs, a section
could not run, or the arguments are wrong).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

try:
    from . import client as C
    from . import protocol as P
    from . import report as R
    from . import sections as S
except ImportError:  # run as a script from this directory or by path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import client as C
    import protocol as P
    import report as R
    import sections as S

SECTIONS = ("decode", "prefill", "ttft", "exact262", "needle")


def _ints(text: str) -> tuple:
    return tuple(int(x) for x in text.split(",") if x.strip())


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="run.py", description=__doc__.split("\n\n")[0])
    ap.add_argument("--base", default="http://127.0.0.1:8000", help="server base URL (default %(default)s)")
    ap.add_argument("--model", help="served model id (default: the first id of /v1/models)")
    ap.add_argument("--api-key", help="bearer token (default: none)")
    ap.add_argument("--profile", required=True,
                    help="the launch row the server was started with (e.g. fn-quad-3090, 27b-tp4-3090); "
                         "picks the card's rows and labels the report")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--section", action="append", metavar="SECTION",
                   help="decode|prefill|ttft|exact262|needle; repeat or comma-separate for several")
    g.add_argument("--all", action="store_true", help="run every section")
    ap.add_argument("--effort", default=P.DEFAULT_EFFORT,
                    help="reasoning effort sent with thinking-on requests (default %(default)s)")
    ap.add_argument("--out", help="output folder (default: ./bench-suite-<profile>-<utc>)")
    ap.add_argument("--boot", type=int, default=1,
                    help="boot index in the ladder nonces; 1 reproduces the published prompts byte for byte")
    ap.add_argument("--no-warm", action="store_true", help="skip the warm-up sends (a lab run never does)")
    ap.add_argument("--timeout", type=float, default=3600.0, help="seconds per request (default %(default)s)")
    # smoke-run overrides: a run that uses any of them is not a card run, and the report says so
    ap.add_argument("--depths", type=_ints, help="ladder/ttft depths, e.g. 4096,131072 (default: the profile's)")
    ap.add_argument("--prefill-depths", type=_ints, help="cold prefill depths (default 10240,102400)")
    ap.add_argument("--exact-target", type=int, help="exact-row prompt tokens (default 261888)")
    ap.add_argument("--needle-counts", help="DEPTH:N,DEPTH:N (default: the profile's, at 262144 and 131072)")
    return ap


def _needle_counts(text: str) -> dict:
    out = {}
    for part in text.split(","):
        d, n = part.split(":")
        out[int(d)] = int(n)
    return out


def main(argv=None, log=print) -> int:
    args = parser().parse_args(argv)
    chosen = []
    for item in (args.section or []):
        chosen += [x for x in item.split(",") if x]
    sections = list(SECTIONS) if args.all else [s for s in SECTIONS if s in chosen]
    bad = [s for s in chosen if s not in SECTIONS]
    if bad:
        print(f"refused: unknown section {bad}", file=sys.stderr)
        return 2
    profile = P.resolve_profile(args.profile)
    warnings = []
    overrides = []
    if args.depths:
        profile["decode_depths"] = args.depths
        overrides.append("--depths")
    prefill_depths = args.prefill_depths or P.PREFILL_DEPTHS
    if args.prefill_depths:
        overrides.append("--prefill-depths")
    exact_target = args.exact_target or P.EXACT_TARGET
    if args.exact_target:
        overrides.append("--exact-target")
    if args.needle_counts:
        counts = _needle_counts(args.needle_counts)
        overrides.append("--needle-counts")
    else:
        counts = {262144: profile["needle"]["n262"], 131072: profile["needle"]["n131"]}
    if args.boot != 1:
        overrides.append("--boot")
    if args.no_warm:
        overrides.append("--no-warm")
    if overrides:
        warnings.append("not a card run: " + ", ".join(overrides) + " changed the protocol")
    if profile["known"].startswith("generic"):
        warnings.append(f"profile {args.profile!r} is not a published row: the generic ladder was used")

    problems = P.selftest()
    if problems:
        print("refused: the frozen protocol inputs are altered: " + "; ".join(problems), file=sys.stderr)
        return 2

    server = C.Server(args.base, args.model, args.api_key, args.timeout)
    out_dir = Path(args.out) if args.out else Path(
        "bench-suite-%s-%s" % (args.profile, time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())))
    run = None
    try:
        models = C.get_models(server)
        if server.model is None:
            server.model = models["ids"][0]
        elif server.model not in models["ids"]:
            raise C.Refused(f"model {server.model!r} is not served; /v1/models lists {models['ids']}")
        max_len = models["max_model_len"]
        run = S.Run(server, args.effort, args.boot, out_dir, log=log)
        if not args.no_warm:
            S.warm(run)
    except C.Refused as e:
        if run is not None:
            run.close()
        print("refused: " + str(e), file=sys.stderr)
        return 2

    rc = 0
    ran = []
    try:
        for name in sections:
            try:
                if name == "decode":
                    S.decode(run, profile["decode_depths"])
                elif name == "prefill":
                    S.prefill(run, prefill_depths)
                elif name == "ttft":
                    S.ttft(run, profile["decode_depths"])
                elif name == "exact262":
                    need = exact_target + P.EXACT_OUT_TOKENS
                    if max_len is not None and max_len < need:
                        warnings.append(f"exact262 skipped: the server's context ({max_len:,}) is below {need:,}")
                        continue
                    base_depth = min(P.REQ, key=lambda d: abs(P.REQ[d] - exact_target))
                    S.exact262(run, exact_target, base_depth)
                elif name == "needle":
                    use = {d: n for d, n in counts.items() if max_len is None or d <= max_len}
                    for d in sorted(set(counts) - set(use)):
                        warnings.append(f"needle depth {d:,} skipped: the server's context is {max_len:,}")
                    if not use:
                        continue
                    S.needle(run, use)
                ran.append(name)
            except C.Refused as e:
                warnings.append(f"{name} refused: {e}")
                print(f"refused ({name}): {e}", file=sys.stderr)
                rc = 2
    finally:
        run.close()

    meta = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command": list(argv if argv is not None else sys.argv[1:]),
        "profile": profile, "effort": args.effort, "boot": args.boot, "sections": ran, "warnings": warnings,
        "server": {"base": server.base, "model": server.model, "models": models["ids"], "max_model_len": max_len},
    }
    report = R.build_report(meta, run.rows)
    jp, mp = R.write_reports(out_dir, report)
    log(f"report: {jp}\nreport: {mp}\ninvalid sends: {report['invalid_sends']}")
    if rc:
        return rc
    return 3 if report["invalid_sends"] else 0


if __name__ == "__main__":
    sys.exit(main())
