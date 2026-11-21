# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Minimal OpenAI-compatible chat client and the self-contained agent loop.

Standard library only. One step = one non-streamed ``/v1/chat/completions`` request with the
workload's tools; every tool call in the reply is executed in order and its result is sent back as a
``tool`` message. An episode ends when the model calls the workload's finishing tool, after
``max_steps`` requests, or after ``max_nudges`` replies in a row without a tool call.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

from agentic_sandbox import SandboxError

NUDGE = ("Please continue the task by calling one of the provided tools. "
         "Call the finishing tool (finish or submit_answer) when you are done.")


class ServerError(RuntimeError):
    pass


def api_base(url: str) -> str:
    base = url.rstrip("/")
    return base if base.endswith("/v1") else base + "/v1"


class ChatClient:
    def __init__(self, base_url: str, model: str | None = None, api_key: str | None = None,
                 timeout_s: float = 600.0, extra_body: dict | None = None):
        self.base = api_base(base_url)
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.extra_body = dict(extra_body or {})
        self.model = model or self.first_model()

    def _request(self, method: str, path: str, body: dict | None = None) -> dict:
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(self.base + path, data=data, method=method)
        req.add_header("Content-Type", "application/json")
        if self.api_key:
            req.add_header("Authorization", f"Bearer {self.api_key}")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")[:500]
            raise ServerError(f"HTTP {e.code} from {path}: {detail}") from e
        except (urllib.error.URLError, TimeoutError, ValueError) as e:
            raise ServerError(f"request to {path} failed: {e}") from e

    def first_model(self) -> str:
        data = self._request("GET", "/models").get("data") or []
        if not data:
            raise ServerError("the server lists no models; pass --model")
        return data[0]["id"]

    def chat(self, messages: list, tools: list, *, max_tokens: int, temperature: float,
             seed: int | None) -> tuple[dict, float]:
        body = {"model": self.model, "messages": messages, "max_tokens": max_tokens,
                "temperature": temperature, "stream": False}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        if seed is not None:
            body["seed"] = seed
        body.update(self.extra_body)
        t0 = time.perf_counter()
        resp = self._request("POST", "/chat/completions", body)
        return resp, time.perf_counter() - t0


def parse_usage(resp: dict) -> dict:
    """Token counts from a response; cached_tokens is None when the server does not report it."""
    u = resp.get("usage") or {}
    details = u.get("prompt_tokens_details")
    cached = details.get("cached_tokens") if isinstance(details, dict) else None
    return {"prompt_tokens": int(u.get("prompt_tokens") or 0),
            "completion_tokens": int(u.get("completion_tokens") or 0),
            "cached_tokens": int(cached) if cached is not None else None}


def _assistant_message(msg: dict, step: int) -> tuple[dict, list]:
    out = {"role": "assistant", "content": msg.get("content") or ""}
    calls = []
    for k, tc in enumerate(msg.get("tool_calls") or []):
        fn = tc.get("function") or {}
        calls.append({"id": tc.get("id") or f"call_{step}_{k}", "type": "function",
                      "function": {"name": fn.get("name") or "", "arguments": fn.get("arguments") or "{}"}})
    if calls:
        out["tool_calls"] = calls
    return out, calls


def run_episode(client: ChatClient, episode, system_prompt: str, tools: list, *, max_steps: int = 40,
                max_nudges: int = 2, max_tokens: int = 4096, temperature: float = 0.0,
                seed: int | None = 0, agent: int = 0, run_checker: bool = True) -> dict:
    """Drive one episode to its end. Returns the episode record with its per-step records.

    With ``run_checker=False`` the caller runs ``check_episode`` later, outside its timed window.
    """
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": episode.user_message}]
    steps: list[dict] = []
    end, error, nudges = "max_steps", None, 0
    t_start = time.perf_counter()
    for step in range(max_steps):
        try:
            resp, latency = client.chat(messages, tools, max_tokens=max_tokens, temperature=temperature,
                                        seed=seed)
        except ServerError as e:
            end, error = "server_error", str(e)
            break
        usage = parse_usage(resp)
        choice = (resp.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        assistant, calls = _assistant_message(msg, step)
        messages.append(assistant)
        rec = {"agent": agent, "step": step, "latency_s": latency, **usage,
               "finish_reason": choice.get("finish_reason"), "tool_calls": len(calls), "tool_s": 0.0}
        steps.append(rec)
        if not calls:
            nudges += 1
            if nudges > max_nudges:
                end = "no_tool_call"
                break
            messages.append({"role": "user", "content": NUDGE})
            continue
        nudges = 0
        t_tool = time.perf_counter()
        for call in calls:
            name = call["function"]["name"]
            try:
                args = json.loads(call["function"]["arguments"] or "{}")
                if not isinstance(args, dict):
                    raise ValueError("arguments must be a JSON object")
                result = episode.call_tool(name, args)
            except (ValueError, TypeError, OSError, SandboxError) as e:
                result = f"error: {e}"
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": result})
        rec["tool_s"] = time.perf_counter() - t_tool
        if episode.done:
            end = "finished"
            break
    rec = {"agent": agent, "task_id": episode.task_id, "end": end, "error": error,
           "elapsed_s": time.perf_counter() - t_start, "steps": steps, "tool_counts": dict(episode.tool_counts)}
    if run_checker:
        check_episode(episode, rec)
    return rec


def check_episode(episode, rec: dict) -> dict:
    """Run the workload's checker and merge its verdict into the episode record."""
    try:
        check = episode.verify()
    except Exception as e:  # a checker crash is a failed episode, not a crashed benchmark
        check = {"success": False, "score": 0.0, "checker_error": repr(e)}
    rec.update(check)
    return rec
