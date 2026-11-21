# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""The card sections, one function each: the single-stream decode ladder, cold
prefill, time to first token, the exact 262K row and the needle panel.

Every function sends its requests through ``client`` and appends one record per
send to ``run.rows`` (and to ``rows.jsonl`` when the run has an output folder).
A record carries ``valid`` and the reasons it is not; nothing is dropped, and the
report is computed from the records alone."""

from __future__ import annotations

import hashlib
import json
import random
import time
import uuid
from pathlib import Path

try:
    from . import client as C
    from . import protocol as P
except ImportError:  # run as a script from this directory
    import client as C
    import protocol as P

SEND_KEYS = ("decode_tok_s_T", "median_event_interval_ms", "ttft_s", "tokens_per_event")


class Run:
    """The state shared by the sections of one invocation."""

    def __init__(self, server: C.Server, effort: str, boot: int = 1,
                 out_dir: str | Path | None = None, log=print):
        self.server = server
        self.effort = effort
        self.boot = boot
        self.rows: list[dict] = []
        self.log = log
        self.out_dir = Path(out_dir) if out_dir else None
        self._fh = None
        if self.out_dir:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            self._fh = open(self.out_dir / "rows.jsonl", "w", encoding="utf-8", newline="\n")

    @property
    def thinking(self) -> dict:
        return P.thinking_kwargs(self.effort)

    def emit(self, row: dict) -> dict:
        self.rows.append(row)
        if self._fh:
            self._fh.write(json.dumps(row) + "\n")
            self._fh.flush()
        brief = {k: row.get(k) for k in ("row", "mode", "depth", "send", "valid", "problems",
                                         "prompt_tokens", "completion_tokens", "decode_tok_s_T",
                                         "ttft_s", "prefill_tok_s") if k in row}
        self.log("ROW " + json.dumps(brief))
        return row

    def close(self) -> None:
        if self._fh:
            self._fh.close()
            self._fh = None


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _check_send_metrics(r: dict) -> None:
    for k in SEND_KEYS:
        if not C.finite_pos(r.get(k)):
            r["problems"].append(f"{k} not a finite positive number")


def _stamp(run: Run, r: dict, **fields) -> dict:
    r.update(fields)
    r["valid"] = not r["problems"]
    return run.emit(r)


# -- warm-up ------------------------------------------------------------------

def warm(run: Run) -> dict:
    """The discarded warm-up sends: an 8K prompt with 16 tokens out, then a 4K
    prompt with 256 out. Raises ``Refused`` when the stream lacks a field the
    protocol needs (usage, logprobs): nothing measured on such a server would be
    comparable."""
    salt = f"warm-{time.time_ns()}"
    r = C.stream_chat(run.server, P.build_prompt(P.WARM_DEPTH, salt), P.WARM_MAX_TOKENS, salt, run.thinking)
    if r.get("http") != 200:
        raise C.Refused(f"the warm-up request failed ({'; '.join(r['problems']) or r.get('http')})")
    missing = C.missing_fields(r)
    if missing:
        raise C.Refused("the server's stream lacks " + " and ".join(missing)
                        + ": the card timing is defined on logprob token events and the usage block")
    nonce = f"G6-{run.boot}-warm"
    w = C.stream_chat(run.server, P.build_prompt(4096, nonce), P.MAX_TOKENS, nonce, run.thinking)
    return {"row": "warm", "depth": 4096, "http": w["http"], "finish_reason": w["finish_reason"],
            "completion_tokens": w["completion_tokens"]}


# -- 1. decode ladder -----------------------------------------------------------

def decode(run: Run, depths) -> None:
    """3 cold sends per cell, thinking on (<= 256 tokens) and thinking off (the
    answer-plus-summary task, natural stop, <= 1,024 tokens)."""
    for depth in depths:
        if depth not in P.REQ:
            raise C.Refused(f"depth {depth} is not a card depth ({sorted(P.REQ)})")
        for i in range(P.SENDS):
            nonce = f"G6-{run.boot}-{depth}-{i}"
            prompt = P.build_prompt(P.REQ[depth], nonce)
            r = C.stream_chat(run.server, prompt, P.MAX_TOKENS, nonce, run.thinking)
            if C.is_int(r.get("completion_tokens")) and r["completion_tokens"] < 2:
                r["problems"].append("fewer than two generated tokens")
            _check_send_metrics(r)
            _stamp(run, r, row="decode", mode="on", depth=depth, send=i, nonce=nonce,
                   prompt_sha256=_sha(prompt))
        for i in range(P.SENDS):
            nonce = f"G6-{run.boot}-{depth}-off{i}"
            prompt = P.build_off(depth, nonce)
            r = C.stream_chat(run.server, prompt, P.OFF_MAX_TOKENS, nonce, P.NO_THINKING, keep_text=True)
            if r["finish_reason"] != "stop":
                r["problems"].append(f"thinking-off send did not stop naturally ({r['finish_reason']!r})")
            if not C.is_int(r.get("completion_tokens")) or r["completion_tokens"] < P.OFF_MIN_TOKENS:
                r["problems"].append("fewer than 200 generated tokens")
            ok, why = P.content_valid(r.get("answer_text") or "")
            if not ok:
                r["problems"].append(f"content gate: {why}")
            _check_send_metrics(r)
            _stamp(run, r, row="decode_off", mode="off", depth=depth, send=i, nonce=nonce,
                   prompt_sha256=_sha(prompt))


# -- 3. cold prefill --------------------------------------------------------------

def prefill(run: Run, depths=P.PREFILL_DEPTHS) -> None:
    """One cold send per depth, 8 tokens out."""
    for depth in depths:
        if depth not in P.REQ:
            raise C.Refused(f"depth {depth} is not a card depth ({sorted(P.REQ)})")
        nonce = f"G6-{run.boot}-{depth}-p"
        prompt = P.build_prompt(P.REQ[depth], nonce)
        r = C.stream_chat(run.server, prompt, P.PREFILL_MAX_TOKENS, nonce, run.thinking)
        if not C.finite_pos(r.get("prefill_tok_s")) or not C.finite_pos(r.get("ttft_s")):
            r["problems"].append("prefill metrics not finite positive")
        _stamp(run, r, row="prefill", mode="on", depth=depth, send=0, nonce=nonce,
               prompt_sha256=_sha(prompt))


# -- 4. time to first token -------------------------------------------------------

def ttft(run: Run, depths) -> None:
    """3 cold sends per depth, 8 tokens out: the time to the first token event
    and the prompt tokens per second of that interval."""
    for depth in depths:
        if depth not in P.REQ:
            raise C.Refused(f"depth {depth} is not a card depth ({sorted(P.REQ)})")
        for i in range(P.SENDS):
            nonce = f"TT-{run.boot}-{depth}-{i}"
            prompt = P.build_prompt(P.REQ[depth], nonce)
            r = C.stream_chat(run.server, prompt, P.PREFILL_MAX_TOKENS, nonce, run.thinking)
            if not C.finite_pos(r.get("ttft_s")):
                r["problems"].append("ttft_s not a finite positive number")
            _stamp(run, r, row="ttft", mode="on", depth=depth, send=i, nonce=nonce,
                   prompt_sha256=_sha(prompt))


# -- 2. exact 262K ----------------------------------------------------------------

def _sized(run: Run, build, tail: str, nonce: str, need: int, base_depth: int):
    """The prompt text whose own token count (the server's ``/tokenize``) is
    ``need``: the card builder's long prompt, padded or trimmed by filler units."""
    t = build(base_depth, nonce)
    if not t.endswith(tail):
        raise C.Refused("prompt builder output changed; refusing an unfrozen exact-262K prompt")
    body = t[: -len(tail)]
    count = lambda text: C.tokenize_count(run.server, text)  # noqa: E731
    n = count(body + tail)
    unit = P.FILLER_UNIT
    for _ in range(60):
        d = need - n
        if d == 0:
            return body + tail, n
        if d >= 40:
            body += unit * max(1, int(d / P.TOKENS_PER_UNIT) - 1)
        elif d > 0:
            body += " ok"
        elif d <= -40:
            body = body[len(unit) * max(1, int(-d / P.TOKENS_PER_UNIT) - 1):]
        else:
            body = body[:-3] if body.endswith(" ok") else body[: body.rfind(" ")]
        n = count(body + tail)
    raise C.Refused(f"could not size the prompt to {need} text tokens (last {n})")


def exact262(run: Run, target: int = P.EXACT_TARGET, base_depth: int = 261120) -> None:
    """Exactly ``target`` prompt tokens (261,888) + 128 out, both thinking modes,
    3 reps: rep 1 cold (a fresh salt), reps 2-3 share its salt with a different
    needle nonce, so they reuse the prefix before the needle. A 4K probe per
    mode gives the template overhead (usage.prompt_tokens - /tokenize count)."""
    modes = {
        "on": (lambda d, n: P.build_prompt(P.REQ[d], n), P.QUESTION, run.thinking),
        "off": (lambda d, n: P.build_off(d, n), P.TASK_OFF, P.NO_THINKING),
    }
    for mode, (build, tail, kw) in modes.items():
        probe = build(4096, "X262-probe-" + mode)
        p = C.stream_chat(run.server, probe, 1, f"x262-probe-{mode}-{time.time_ns()}", kw)
        if not C.is_int(p.get("prompt_tokens")):
            raise C.Refused("the probe request returned no prompt_tokens; cannot size the exact row")
        overhead = p["prompt_tokens"] - C.tokenize_count(run.server, probe)
        run.log(f"exact262 {mode}: template overhead {overhead} tokens")
        adj = 0
        salt = f"x262-{mode}-{time.time_ns()}"
        for rep in range(P.EXACT_REPS):
            nonce = f"X262-{mode}-{int(time.time()) % 100000:05d}-{rep}"
            text, ntext = _sized(run, build, tail, nonce, target - overhead - adj, base_depth)
            r = C.stream_chat(run.server, text, P.EXACT_OUT_TOKENS, salt, kw, keep_text=(mode == "off"))
            r.pop("events", None)
            exact = r.get("prompt_tokens") == target
            if C.is_int(r.get("prompt_tokens")):
                adj += r["prompt_tokens"] - target
            if not exact:
                r["problems"].append(f"prompt_tokens {r.get('prompt_tokens')} != {target}")
            if rep > 0:  # reps 2-3 share rep 1's salt: reusing its prefix is expected, reported
                r["problems"] = [x for x in r["problems"] if not x.startswith("prefix-cache hit")]
            _check_send_metrics(r)
            _stamp(run, r, row="exact262", mode=mode, rep=rep, cold=(rep == 0),
                   effort=run.effort if mode == "on" else None, target_prompt_tokens=target,
                   exact=exact, text_tokens=ntext, nonce=nonce,
                   needle_found=(nonce in (r.get("answer_text") or "")) if mode == "off" else None)


# -- 5. needle panel ----------------------------------------------------------------

def _needle_text(n_units: int, k_prefix: int, nonce: str) -> str:
    k = max(0, min(n_units, k_prefix))
    return (P.FILLER_UNIT * k + P.NEEDLE_TEMPLATE.replace("{nonce}", nonce)
            + P.FILLER_UNIT * (n_units - k) + "\n\n" + P.QUESTION)


def _needle_body(server: C.Server, text: str, max_tokens: int, seed: int, salt: str) -> dict:
    return {
        "model": server.model,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": P.image_data_uri()}},
            {"type": "text", "text": text}]}],
        "temperature": 0, "seed": seed % (2 ** 31), "max_tokens": max_tokens, "stream": False,
        "chat_template_kwargs": dict(P.NO_THINKING), "cache_salt": salt,
    }


def _needle_row(case: dict, status, resp, err: str | None, max_tokens: int) -> dict:
    reasons = []
    if err or status != 200:
        reasons.append(err or f"HTTP {status}")
    choice = None
    if isinstance(resp, dict):
        ch = resp.get("choices")
        if isinstance(ch, list) and ch and isinstance(ch[0], dict):
            choice = ch[0]
    if choice is None and not reasons:
        reasons.append("no choices in response")
    msg = (choice or {}).get("message") or {}
    content = msg.get("content")
    finish = (choice or {}).get("finish_reason")
    usage = resp.get("usage") if isinstance(resp, dict) else None
    pt = ct = cached = rtok = None
    if isinstance(usage, dict):
        pt, ct = usage.get("prompt_tokens"), usage.get("completion_tokens")
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
        rtok = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
    if not (C.is_int(pt) and C.is_int(ct)):
        reasons.append("usage missing prompt/completion tokens")
    if choice is not None and finish not in ("stop", "length"):
        reasons.append(f"finish_reason={finish!r}")
    if C.is_int(pt):
        floor = int(P.NEEDLE_FLOOR_FRAC * case["prompt_target"])
        if pt < floor:
            reasons.append(f"prompt_tokens {pt} < floor {floor} (depth not reached)")
        if pt + max_tokens > case["depth_target"]:
            reasons.append(f"prompt_tokens {pt} + max_tokens exceeds depth {case['depth_target']}")
    if C.is_int(cached) and cached > 0:
        reasons.append(f"cached_tokens={cached} despite a fresh cache_salt")
    exact, legacy = P.score_answer(content, case["nonce"])
    reasoning = msg.get("reasoning_content") or msg.get("reasoning")
    return {
        "row": "needle", "id": case["id"], "depth": case["depth_target"], "depth_target": case["depth_target"],
        "prompt_target": case["prompt_target"], "nonce": case["nonce"], "http": status,
        "finish_reason": finish, "prompt_tokens": pt, "completion_tokens": ct, "cached_tokens": cached,
        "content": content, "usage": usage, "exact_pass": exact, "legacy_pass": legacy,
        "reasoning_tokens": rtok if C.is_int(rtok) else None,
        "thinking_observed": bool(reasoning) or (C.is_int(rtok) and rtok > 0),
        "problems": reasons, "valid": not reasons,
    }


def needle(run: Run, counts: dict, seed: int = P.NEEDLE_SEED, position: float = P.NEEDLE_POSITION,
           max_tokens: int = P.NEEDLE_MAX_TOKENS, max_consecutive_errors: int = 3) -> None:
    """The needle panel: one single-occurrence needle (the nonce, once, in the
    prompt) at ``position`` of the haystack, an image in the prompt, thinking
    off, temperature 0, a fresh cache salt per request, sent one at a time.
    ``counts`` maps a depth to its number of cases. Prompts are sized with the
    server's ``/tokenize`` plus the template overhead of a probe request."""
    if not 0.05 < position < 0.95:
        raise C.Refused("the needle position must be inside (0.05, 0.95)")
    rng = random.Random(seed)
    used: set[str] = set()
    probe_nonce = "NV00000-00"
    probe_text = _needle_text(100, 55, probe_nonce)
    st, resp, err = C.chat_once(run.server, _needle_body(run.server, probe_text, 1, seed, f"ndl-probe-{uuid.uuid4().hex}"))
    pt = ((resp or {}).get("usage") or {}).get("prompt_tokens") if isinstance(resp, dict) else None
    if not C.is_int(pt):
        raise C.Refused("the needle probe returned no usage.prompt_tokens: " + (err or f"HTTP {st}"))
    overhead = pt - C.tokenize_count(run.server, probe_text)
    t0 = C.tokenize_count(run.server, _needle_text(0, 0, probe_nonce))
    t1 = C.tokenize_count(run.server, _needle_text(400, 200, probe_nonce))
    unit = (t1 - t0) / 400.0
    if unit <= 0:
        raise C.Refused("a filler unit has no tokens on this server's tokenizer")
    consecutive = 0
    for depth in sorted(counts, reverse=True):
        n_cases = int(counts.get(depth, 0))
        if n_cases <= 0:
            continue
        target = depth - P.needle_reserve(depth)
        tol = max(8, round(0.001 * target))
        n = max(1, round((target - overhead - t0) / unit))
        for _ in range(8):
            got = C.tokenize_count(run.server, _needle_text(n, round(position * n), probe_nonce)) + overhead
            if abs(target - got) <= tol:
                break
            n = max(1, n + round((target - got) / unit))
        k = round(position * n)
        run.log(f"needle {depth}: {n} filler units, template overhead {overhead} tokens")
        for i in range(n_cases):
            while True:
                nonce = f"NV{rng.randrange(10000, 100000):05d}-{i:02d}"
                if nonce[:7] not in used:
                    used.add(nonce[:7])
                    break
            case = {"id": f"{depth}-{i:02d}", "depth_target": depth, "prompt_target": target, "nonce": nonce}
            text = _needle_text(n, k, nonce)
            salt = f"ndl-{run.boot}-{case['id']}-{uuid.uuid4().hex}"
            status, resp, err = C.chat_once(run.server, _needle_body(run.server, text, max_tokens, seed, salt))
            row = _needle_row(case, status, resp, err, max_tokens)
            row.update({"cache_salt": salt, "filler_units": n, "prefix_units": k,
                        "position_target": position, "max_tokens": max_tokens})
            run.emit(row)
            consecutive = consecutive + 1 if status is None else 0
            if consecutive >= max_consecutive_errors:
                raise C.Refused(f"{consecutive} consecutive requests failed to reach the server; stopping the panel")
