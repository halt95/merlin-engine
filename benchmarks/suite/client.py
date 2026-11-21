# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""A small client for an OpenAI-compatible Merlin server (standard library only):
the streamed chat request the cards time, the token-event metrics, the send
validity rules, a non-streamed request for the needle panel, and the server
probes (``/v1/models``, ``/tokenize``)."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import time
import urllib.error
import urllib.request

try:
    from . import protocol as P
except ImportError:  # run as a script from this directory
    import protocol as P


class Refused(Exception):
    """The server cannot be measured with the card protocol; the message says why."""


class Server:
    def __init__(self, base: str, model: str | None = None, api_key: str | None = None,
                 timeout: float = 3600.0):
        self.base = base.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    def headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = "Bearer " + self.api_key
        return h

    def request(self, path: str, payload: dict | None = None, timeout: float | None = None):
        data = None if payload is None else json.dumps(payload).encode()
        req = urllib.request.Request(self.base + path, data=data, headers=self.headers())
        return urllib.request.urlopen(req, timeout=timeout or self.timeout)


def is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def finite_pos(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v > 0


def entry_ok(e) -> bool:
    return (isinstance(e, dict) and isinstance(e.get("token"), str)
            and isinstance(e.get("logprob"), (int, float)) and not isinstance(e.get("logprob"), bool)
            and math.isfinite(e.get("logprob")))


# -- server probes ------------------------------------------------------------

def get_models(server: Server) -> dict:
    """``/v1/models``: the served ids and, when reported, the context length."""
    try:
        with server.request("/v1/models", timeout=60) as resp:
            doc = json.load(resp)
    except (urllib.error.URLError, OSError, ValueError) as e:
        raise Refused(f"cannot read {server.base}/v1/models: {e}") from e
    data = doc.get("data") if isinstance(doc, dict) else None
    if not isinstance(data, list) or not data or not isinstance(data[0], dict) or not data[0].get("id"):
        raise Refused(f"{server.base}/v1/models lists no model: is this an OpenAI-compatible server?")
    ids = [d.get("id") for d in data if isinstance(d, dict)]
    lens = [d.get("max_model_len") for d in data if is_int(d.get("max_model_len"))]
    return {"ids": ids, "max_model_len": lens[0] if lens else None}


def tokenize_count(server: Server, text: str) -> int:
    """Prompt tokens of ``text`` on the server's own tokenizer (``/tokenize``)."""
    try:
        with server.request("/tokenize", {"model": server.model, "prompt": text}, timeout=600) as resp:
            doc = json.load(resp)
    except (urllib.error.URLError, OSError, ValueError) as e:
        raise Refused(f"{server.base}/tokenize failed ({e}); the exact-262K and needle sections size "
                      "their prompts with it") from e
    if is_int(doc.get("count")):
        return doc["count"]
    toks = doc.get("tokens")
    if isinstance(toks, list):
        return len(toks)
    raise Refused(f"{server.base}/tokenize returned neither a count nor tokens")


# -- the streamed card request ------------------------------------------------

def metrics_from_events(events, pt):
    """Card metrics from the token events ``[t_rel, tokens, chars]``.
    tok/s = tokens after the first event / time from the first to the last event."""
    m = {}
    if not events:
        return m
    m["ttft_s"] = events[0][0]
    if is_int(pt) and pt > 0 and events[0][0] > 0:
        m["prefill_tok_s"] = pt / events[0][0]
    if len(events) >= 2:
        t1, tn = events[0][0], events[-1][0]
        gaps = [events[i][0] - events[i - 1][0] for i in range(1, len(events))]
        m["measured_interval_s"] = tn - t1
        m["tokens_after_first_event"] = sum(e[1] for e in events[1:])
        if tn > t1:
            m["decode_tok_s_T"] = m["tokens_after_first_event"] / (tn - t1)
            m["median_event_interval_ms"] = statistics.median(gaps) * 1000
        m["tokens_per_event"] = sum(e[1] for e in events) / len(events)
    return m


def stream_chat(server: Server, prompt: str, max_tokens: int, salt: str, kwargs: dict,
                keep_text: bool = False) -> dict:
    """One streamed chat send with the image and the prompt, temperature 0, a
    per-request cache salt (a cold prefill), logprobs on so that every streamed
    chunk with a token is one timed event. Returns the send record; its
    ``problems`` list is empty for a valid send."""
    body = {
        "model": server.model,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": P.image_data_uri()}},
            {"type": "text", "text": prompt}]}],
        "temperature": 0, "max_tokens": max_tokens, "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": kwargs, "cache_salt": salt,
        "logprobs": True, "top_logprobs": 0,
    }
    t_req = time.perf_counter()
    events, text, ep = [], [], []
    finish = usage = http = None
    terminal = False
    reasoning_deltas = 0
    transport = None
    try:
        with server.request("/v1/chat/completions", body) as resp:
            http = resp.status
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    break
                try:
                    ev = json.loads(payload)
                except ValueError:
                    ep.append("malformed SSE event")
                    continue
                if ev.get("usage"):
                    usage = ev["usage"]
                for ch in ev.get("choices", []):
                    d = ch.get("delta", {}) or {}
                    content = d.get("content")
                    if d.get("reasoning") or d.get("reasoning_content"):
                        reasoning_deltas += 1
                    lp = (ch.get("logprobs") or {}).get("content")
                    entries = lp if isinstance(lp, list) else []
                    if lp is not None and not isinstance(lp, list):
                        ep.append("logprobs.content is not a list")
                    if any(not entry_ok(e) for e in entries):
                        ep.append("malformed logprob entry")
                    n_lp = len(entries)
                    has_content = isinstance(content, str) and content != ""
                    if terminal and (has_content or n_lp):
                        ep.append("content or entries after the terminal event")
                    if has_content and n_lp < 1:
                        ep.append("content without logprob entries")
                    if n_lp >= 1:
                        events.append([time.perf_counter() - t_req, n_lp,
                                       len(content) if has_content else 0])
                    if has_content:
                        text.append(content)
                    if ch.get("finish_reason"):
                        if terminal:
                            ep.append("more than one terminal event")
                        terminal = True
                        finish = ch["finish_reason"] if finish is None else finish
    except urllib.error.HTTPError as e:
        http = e.code
        transport = f"http {e.code}"
    except (urllib.error.URLError, OSError) as e:
        transport = f"transport error: {e}"
    full = "".join(text)
    u = usage if isinstance(usage, dict) else {}
    ct, pt = u.get("completion_tokens"), u.get("prompt_tokens")
    cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens")
    rtok = (u.get("completion_tokens_details") or {}).get("reasoning_tokens")
    rec = {
        "http": http, "finish_reason": finish, "completion_tokens": ct, "prompt_tokens": pt,
        "cached_tokens": cached, "cache_salt": salt, "max_tokens": max_tokens,
        "chat_template_kwargs": kwargs, "answer_text": full if keep_text else None,
        "n_events": len(events), "tokens_in_events": sum(e[1] for e in events), "events": events,
        "answer_sha256": hashlib.sha256(full.encode()).hexdigest(), "event_problems": sorted(set(ep)),
        "reasoning_deltas": reasoning_deltas,
        "reasoning_tokens": rtok if is_int(rtok) else None,
        "thinking_observed": bool(reasoning_deltas or (is_int(rtok) and rtok > 0)),
        "usage": usage,
    }
    rec.update(metrics_from_events(events, pt))
    problems = list(rec["event_problems"])
    if transport:
        problems.append(transport)
    elif http != 200:
        problems.append(f"http {http}")
    if not transport:
        if finish not in ("stop", "length"):
            problems.append(f"finish {finish!r}")
        if usage is None:
            problems.append("usage missing from the stream")
        if not events:
            problems.append("no token events: the stream carries no logprob entries")
        if not is_int(ct):
            problems.append("completion_tokens not an integer")
        elif events and sum(e[1] for e in events) != ct:
            problems.append(f"tokens_in_events {sum(e[1] for e in events)} != completion_tokens {ct}")
        if not is_int(pt) or pt <= 0:
            problems.append("prompt_tokens not a positive integer")
        if cached is not None and cached != 0:
            problems.append(f"prefix-cache hit {cached} tokens on a salted request")
        if any(events[i][0] <= events[i - 1][0] for i in range(1, len(events))):
            problems.append("event arrivals not strictly increasing")
    rec["problems"] = problems
    return rec


def missing_fields(rec: dict) -> list[str]:
    """Fields the card protocol needs and a send did not carry."""
    out = []
    if rec.get("http") != 200:
        return out
    if rec.get("usage") is None:
        out.append("usage (stream_options.include_usage)")
    if not rec.get("n_events"):
        out.append("logprobs (logprobs=true, top_logprobs=0 on the stream)")
    return out


# -- the non-streamed request (needle panel) ----------------------------------

def chat_once(server: Server, body: dict) -> tuple[int | None, dict | None, str | None]:
    """POST a chat body; returns (http status, parsed JSON or None, error text)."""
    try:
        with server.request("/v1/chat/completions", body) as resp:
            raw = resp.read()
            status = resp.status
    except urllib.error.HTTPError as e:
        return e.code, None, f"http {e.code}"
    except (urllib.error.URLError, OSError) as e:
        return None, None, f"transport error: {e}"
    try:
        return status, json.loads(raw), None
    except ValueError as e:
        return status, None, f"response not JSON: {e}"
