# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Chat-template / parser round-trip self-test. It only reports: it never refuses boot and never changes an output.

Nothing else checks that the served chat template and the selected reasoning and
tool parsers read each other. ``ParserManager.validate_parser_tokenizer`` only
constructs the parsers against the tokenizer. This probe goes further:

1. It renders a fixed conversation (one user turn, then an assistant turn with
   reasoning and one tool call) through the template actually served, with the
   served ``chat_template_kwargs``.
2. It checks that the generation prompt is a prefix of the rendered assistant
   turn, and takes the rest as what the model would have generated.
3. It parses that text with the selected parsers, whole and streamed one token
   at a time, and compares reasoning, tool name and arguments with the fixture
   and the two parses with each other.
4. It lists every ``token_id_terminals`` marker the tokenizer does not hold as
   one token. The parser then matches that marker as text, and the id-keyed
   handling for it (literal-marker spans, reasoning end by id) is off.

Every finding is a warning or a note about coverage; a probe that cannot run
(no template, a tokenizer without ``apply_chat_template``, a parser the fixture
does not fit) says so and stops. Boot logs each finding once per tokenizer and
parser pair. The fixture is deliberately well formed: the probe feeds the
parsers nothing a normal model turn would not contain, and it runs on a
daemon thread with a deadline, so a parser that stops returning cannot hold up
boot.

Doctor use, tokenizer files only (no weights)::

    python -m vllm.merlin.parser_selftest --model CKPT_DIR \\
        [--chat-template FILE] --reasoning-parser qwen3 \\
        --tool-call-parser qwen3_coder [--chat-template-kwargs JSON] [--strict]
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import struct
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)

DEADLINE_S = 20.0
STOP_GRACE_S = 2.0
CLOSE_WAIT_S = 5.0  # how long closing the helper waits for a killed process before reaping it elsewhere

_REASONING = "Plan the lookup."
_USER = "What is the weather in Tokyo for 3 days?"
_CALL_NAME = "get_weather"
_CALL_ARGS = {"city": "Tokyo", "days": 3}
_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": _CALL_NAME,
            "description": "Get the weather.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "days": {"type": "integer"},
                },
                "required": ["city"],
            },
        },
    }
]

# Both spellings of the reasoning field: templates read one or the other.
_ASSISTANT = {
    "role": "assistant",
    "content": "",
    "reasoning_content": _REASONING,
    "reasoning": _REASONING,
    "tool_calls": [
        {
            "id": "call_0",
            "type": "function",
            "function": {"name": _CALL_NAME, "arguments": dict(_CALL_ARGS)},
        }
    ],
}
_USER_MSG = {"role": "user", "content": _USER}


@dataclass
class Finding:
    # "warn": a mismatch or unresolved marker; "note": coverage the probe could not give;
    # "info": a difference already known and tracked elsewhere (reported, not a warning)
    level: str
    text: str


@dataclass
class Report:
    findings: list[Finding] = field(default_factory=list)
    covered: list[str] = field(default_factory=list)
    elapsed_ms: float = 0.0

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.level == "warn"]

    @property
    def notes(self) -> list[Finding]:
        return [f for f in self.findings if f.level == "note"]

    @property
    def infos(self) -> list[Finding]:
        return [f for f in self.findings if f.level == "info"]

    def info(self, text: str) -> None:
        self.findings.append(Finding("info", text))

    def warn(self, text: str) -> None:
        self.findings.append(Finding("warn", text))

    def note(self, text: str) -> None:
        self.findings.append(Finding("note", text))


def _clean_kwargs(kwargs: dict[str, Any] | None) -> dict[str, Any]:
    return {
        k: v
        for k, v in (kwargs or {}).items()
        if k not in ("chat_template", "tokenize", "tools", "add_generation_prompt")
    }


def _render(tokenizer, template, messages, generation_prompt, kwargs) -> str:
    out = tokenizer.apply_chat_template(
        messages,
        tools=_TOOLS,
        chat_template=template,
        tokenize=False,
        add_generation_prompt=generation_prompt,
        **kwargs,
    )
    if not isinstance(out, str):
        raise TypeError(f"apply_chat_template returned {type(out).__name__}, not text")
    return out


def _end_of_turn(tokenizer, template, kwargs) -> str:
    """What the template writes after an assistant turn's text, found by rendering a marked plain turn."""
    mark = "SELFTESTMARK"
    out = _render(
        tokenizer,
        template,
        [_USER_MSG, {"role": "assistant", "content": mark}],
        False,
        kwargs,
    )
    return out[out.rindex(mark) + len(mark) :].strip() if mark in out else ""


def _unresolved_markers(parser, tokenizer) -> list[tuple[str, str]]:
    """``token_id_terminals`` markers that are not one token in the tokenizer's vocabulary."""
    vocab = tokenizer.get_vocab()
    seen: dict[tuple[str, str], None] = {}
    for sub in (parser.reasoning_parser, parser.tool_parser):
        config = getattr(getattr(sub, "_parser_engine", None), "parser_engine_config", None)
        for terminal, text in (getattr(config, "token_id_terminals", None) or {}).items():
            if vocab.get(text) is None:
                seen.setdefault((terminal, text), None)
    return list(seen)


def _norm_calls(calls) -> list[tuple[str, Any]]:
    out = []
    for c in calls or []:
        try:
            args = json.loads(c.arguments) if c.arguments else {}
        except (TypeError, ValueError):
            args = c.arguments
        out.append((c.name, args))
    return out


def _stream(parser, tokenizer, request, ids, prompt_ids):
    reasoning: list[str] = []
    content: list[str] = []
    slots: dict[int, list] = {}
    prev = ""
    for n, tid in enumerate(ids):
        text = tokenizer.decode(ids[: n + 1])
        delta_text = text[len(prev) :] if text.startswith(prev) else tokenizer.decode([tid])
        prev = text
        delta = parser.parse_delta(
            delta_text, [tid], request, prompt_ids, finished=n == len(ids) - 1
        )
        if delta is None:
            continue
        if delta.reasoning:
            reasoning.append(delta.reasoning)
        if delta.content:
            content.append(delta.content)
        for tc in delta.tool_calls or []:
            slot = slots.setdefault(tc.index, ["", ""])
            if tc.function and tc.function.name:
                slot[0] += tc.function.name
            if tc.function and tc.function.arguments:
                slot[1] += tc.function.arguments
    calls = []
    for i in sorted(slots):
        name, raw = slots[i]
        try:
            args = json.loads(raw) if raw else {}
        except ValueError:
            args = raw
        calls.append((name, args))
    return "".join(reasoning), "".join(content), calls


def _probe(
    report: Report,
    tokenizer,
    chat_template: str | None,
    parser_cls,
    chat_template_kwargs: dict[str, Any] | None,
    model_config,
) -> None:
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest

    kwargs = _clean_kwargs(chat_template_kwargs)
    request = ChatCompletionRequest(
        model="selftest", messages=[_USER_MSG], tools=_TOOLS
    )

    def make():
        return parser_cls(
            tokenizer,
            request.tools,
            chat_template_kwargs=kwargs,
            model_config=model_config,
        )

    try:
        parser = make()
    except Exception as e:  # noqa: BLE001 - report, never raise
        report.warn(f"the parsers cannot be built against this tokenizer: {type(e).__name__}: {e}")
        return
    for terminal, text in _unresolved_markers(parser, tokenizer):
        report.warn(
            f"marker {text!r} ({terminal}) is not one token in this tokenizer; "
            "the parser matches it as text, so handling keyed on its token id is off"
        )
    has_reasoning = parser.reasoning_parser is not None
    has_tools = parser.tool_parser is not None
    if not (has_reasoning or has_tools):
        report.note("no reasoning or tool parser selected: the round trip was not run")
        return
    if chat_template is None:
        report.note("no chat template resolved for this tokenizer: the round trip was not run")
        return
    if not hasattr(tokenizer, "apply_chat_template"):
        report.note("this tokenizer cannot render a chat template: the round trip was not run")
        return

    try:
        head = _render(tokenizer, chat_template, [_USER_MSG], True, kwargs)
        full = _render(tokenizer, chat_template, [_USER_MSG, _ASSISTANT], False, kwargs)
        with_reasoning = True
        if not full.startswith(head):
            # Served kwargs may close reasoning in the prompt (thinking off): retry without it.
            bare = {k: v for k, v in _ASSISTANT.items() if not k.startswith("reasoning")}
            retry = _render(tokenizer, chat_template, [_USER_MSG, bare], False, kwargs)
            if retry.startswith(head):
                full, with_reasoning = retry, False
                report.note(
                    "the served kwargs close reasoning in the generation prompt: "
                    "the reasoning half of the round trip was not run"
                )
            else:
                report.warn(
                    "the generation prompt is not a prefix of the rendered assistant turn: "
                    "the template re-renders the turn the model produced"
                )
                return
    except Exception as e:  # noqa: BLE001
        report.note(f"the fixed conversation could not be rendered: {type(e).__name__}: {e}")
        return

    body = full[len(head) :].rstrip()
    try:
        eot = _end_of_turn(tokenizer, chat_template, kwargs)
    except Exception:  # noqa: BLE001 - the fixed conversation already rendered; keep the tail
        eot = ""
    if eot and body.endswith(eot):
        body = body[: -len(eot)].rstrip()
    check_reasoning = has_reasoning and with_reasoning and _REASONING in body
    check_tools = has_tools and _CALL_NAME in body and "Tokyo" in body
    if has_reasoning and with_reasoning and not check_reasoning:
        report.note("the template does not render reasoning: the reasoning half was not run")
    if has_tools and not check_tools:
        report.note("the template does not render tool calls: the tool half was not run")
    if not (check_reasoning or check_tools):
        return
    report.covered.extend(
        n for n, on in (("reasoning", check_reasoning), ("tool call", check_tools)) if on
    )

    ids = list(tokenizer.encode(body, add_special_tokens=False))
    prompt_ids = list(tokenizer.encode(head, add_special_tokens=False))
    expected = [(_CALL_NAME, _CALL_ARGS)]

    def judge(label, reasoning, calls):
        if check_reasoning and (reasoning or "").strip() != _REASONING:
            report.warn(
                f"{label}: the reasoning parser did not return the reasoning the template "
                f"rendered (got {(reasoning or '')[:60]!r})"
            )
        if check_tools and calls != expected:
            report.warn(
                f"{label}: the tool-call parser did not return the call the template "
                f"rendered (got {calls!r})"
            )

    whole = make()
    whole.set_prompt_token_ids(prompt_ids)
    reasoning, content, tool_calls = whole.parse(
        body, request, enable_auto_tools=True, model_output_token_ids=ids
    )
    whole_calls = _norm_calls(tool_calls)
    judge("whole output", reasoning, whole_calls)

    # The streamed parser is seeded the way serving seeds it: the prompt ids go in with every delta and
    # ``parse_delta`` applies them to the parser state before its first delta.
    s_reasoning, s_content, s_calls = _stream(make(), tokenizer, request, ids, prompt_ids)
    judge("streamed output", s_reasoning, s_calls)
    whole_text = (reasoning or "", content or "")
    if whole_calls != s_calls or whole_text[0].strip() != s_reasoning.strip() or (
        whole_text[1].strip() != s_content.strip()
    ):
        report.warn("streamed and whole-output parses of the same text differ")
    elif whole_text != (s_reasoning, s_content):
        report.warn(
            "streamed and whole-output parses of the same text differ only in whitespace: "
            f"whole {whole_text!r}, streamed {(s_reasoning, s_content)!r}"
        )


class _Cancelled(BaseException):
    """Raised inside an abandoned probe so it stops; not an Exception, so a parser cannot swallow it."""


def _stop_thread(thread: threading.Thread, grace_s: float) -> bool:
    """Ask a running probe thread to stop and wait for it. True when it has ended."""
    import ctypes

    ctypes.pythonapi.PyThreadState_SetAsyncExc(
        ctypes.c_ulong(thread.ident or 0), ctypes.py_object(_Cancelled)
    )
    thread.join(grace_s)
    return not thread.is_alive()


class _HelperError(Exception):
    """The tokenizer helper process could not be started or stopped answering."""


_CHILD_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "parser_selftest_child.py")


def _helper_command() -> list[str]:
    """The helper's command line: ``-I -S`` plus the one directory ``tokenizers`` is imported from.

    ``-I`` drops the environment and the user site, ``-S`` skips the ``site`` module, so no ``.pth`` file and
    no ``sitecustomize`` of the serving environment runs in the helper (such code could import torch or vllm,
    or write to the helper's framed stdout). The directory is resolved here, in the server, without importing
    the package, and the helper puts it in front of the standard library on its ``sys.path``.
    """
    import importlib.util

    spec = importlib.util.find_spec("tokenizers")
    locations = list(spec.submodule_search_locations or ()) if spec is not None else []
    if not locations:
        raise _HelperError("the tokenizers package was not found for the tokenizer helper process")
    return [sys.executable, "-I", "-S", _CHILD_SCRIPT, os.path.dirname(os.path.abspath(locations[0]))]


class _TokenizerHelper:
    """A short-lived child process that holds the native tokenizer, so the serving process never does.

    The child imports only ``tokenizers`` (no vllm, no torch, no CUDA; see parser_selftest_child.py). The
    serving process hands it the serialised tokenizer and later asks it to encode and decode; closing it
    kills the child and reaps it, so a request that never returns cannot outlive the probe.
    """

    def __init__(self, serialised: str) -> None:
        self._serialised: bytes | None = serialised.encode("utf-8")  # dropped once the helper has it
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._closed = False

    def start(self) -> None:
        env = dict(os.environ, TOKENIZERS_PARALLELISM="false", RAYON_NUM_THREADS="1")
        try:
            command = _helper_command()
            proc = subprocess.Popen(  # noqa: S603 - the same interpreter, our own script
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                env=env,
            )
        except OSError as e:
            raise _HelperError(f"the tokenizer helper process could not be started: {e}") from e
        with self._lock:
            if self._closed:
                proc.kill()
                proc.wait()
                raise _HelperError("the tokenizer helper was closed before it started")
            self._proc = proc
        payload, self._serialised = self._serialised, None
        if payload is None or self._exchange(payload) != b"ok":
            raise _HelperError("the tokenizer helper did not accept the tokenizer")

    def _exchange(self, payload: bytes) -> bytes:
        proc = self._proc
        if proc is None or proc.stdin is None or proc.stdout is None:
            raise _HelperError("the tokenizer helper is not running")
        try:
            proc.stdin.write(struct.pack(">I", len(payload)))  # header and payload apart: no second copy
            proc.stdin.write(payload)
            proc.stdin.flush()
            head = proc.stdout.read(4)
            if len(head) < 4:
                raise _HelperError("the tokenizer helper stopped answering")
            (size,) = struct.unpack(">I", head)
            data = proc.stdout.read(size)
        except (OSError, ValueError) as e:  # a closed pipe: the helper was killed or died
            raise _HelperError(f"the tokenizer helper stopped answering: {e}") from e
        if len(data) != size:
            raise _HelperError("the tokenizer helper stopped answering")
        return data

    def call(self, request: dict[str, Any]) -> dict[str, Any]:
        reply = json.loads(self._exchange(json.dumps(request).encode("utf-8")))
        if "error" in reply:
            raise _HelperError(f"the tokenizer helper failed: {reply['error']}")
        return reply

    def close(self) -> None:
        with self._lock:
            self._closed = True
            proc, self._proc = self._proc, None
        if proc is None:
            return
        try:  # a launcher stub (a Windows venv python) has the real interpreter as its child
            import psutil

            for child in psutil.Process(proc.pid).children(recursive=True):
                child.kill()
        except Exception:  # noqa: BLE001 - best effort; the pipes closing also ends the helper
            pass
        proc.kill()  # first: the kill ends a read, and breaks a write, that a worker is blocked in
        try:
            proc.wait(CLOSE_WAIT_S)
        except subprocess.TimeoutExpired:
            # Cannot happen for a killed process short of a stuck kernel call. Closing the buffered pipes now
            # could wait on a worker that still holds their lock, so reap and close off this thread instead.
            logger.warning(
                "Parser/template self-test: helper process %s did not exit %g s after a kill; "
                "it is reaped in the background",
                proc.pid,
                CLOSE_WAIT_S,
            )
            threading.Thread(target=self._reap, args=(proc,), name="parser-selftest-reap", daemon=True).start()
            return
        self._close_pipes(proc)

    def _reap(self, proc: subprocess.Popen) -> None:
        try:
            proc.wait()
        finally:
            self._close_pipes(proc)

    @staticmethod
    def _close_pipes(proc: subprocess.Popen) -> None:
        for stream in (proc.stdin, proc.stdout):
            try:
                if stream is not None:
                    stream.close()
            except (OSError, ValueError):
                pass


class _NoBackend:
    """Stands in for the native backend on the probe's tokenizer: any use is an error, never a shared call."""

    def __getattr__(self, name):
        raise RuntimeError(f"the probe's tokenizer has no native backend ({name} is not available)")


@dataclass(frozen=True)
class _AddedToken:
    """An added token as the parsers read it (its text and whether it is
    flagged special), copied out of the serving tokenizer so the probe holds
    no native object."""

    content: str
    special: bool

    def __str__(self) -> str:
        return self.content


def _added_tokens(serialised: str) -> dict[int, _AddedToken]:
    """The serving tokenizer's added tokens, read from the serialised form the
    boot thread already took (no second call into its native backend); empty
    when it lists none."""
    at = serialised.find('"added_tokens":')
    if at < 0:
        return {}
    start = serialised.index("[", at)
    try:
        listed, _ = json.JSONDecoder().raw_decode(serialised, start)
    except ValueError:
        return {}
    return {
        int(t["id"]): _AddedToken(str(t["content"]), bool(t.get("special", False)))
        for t in listed
        if isinstance(t, dict) and "id" in t and "content" in t
    }


class _RemoteTokenizer:
    """The tokenizer the probe uses: encode and decode go to the helper process, the rest is Python state
    cloned from the serving tokenizer (templates, special tokens, added
    tokens) plus its read-only vocabulary dict."""

    def __init__(
        self,
        clone,
        vocab,
        special_tokens,
        special_ids,
        helper: _TokenizerHelper,
        added=None,
    ) -> None:
        self._clone = clone
        self._vocab = vocab
        self._special_tokens = special_tokens
        self._special_ids = special_ids
        self._helper = helper
        self._added = added if added is not None else {}

    @property
    def all_special_tokens(self) -> list[str]:
        return self._special_tokens

    @property
    def all_special_ids(self) -> list[int]:
        return self._special_ids

    @property
    def added_tokens_decoder(self) -> dict[int, _AddedToken]:
        # The parsers read which added tokens are special (the Qwen3 line-start
        # test ignores them): the probe must build the same parser as serving,
        # so this comes from the serving tokenizer too, never from the clone,
        # whose property would reach the missing native backend.
        return self._added

    def __getattr__(self, name):  # only reached for attributes not defined here
        return getattr(self._clone, name)

    def __len__(self) -> int:
        return len(self._clone)

    def get_vocab(self) -> dict[str, int]:
        return self._vocab

    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, str):
            return self._vocab.get(tokens)
        return [self._vocab.get(t) for t in tokens]

    def convert_ids_to_tokens(self, ids, skip_special_tokens=False):
        single = isinstance(ids, int)
        tokens = self._helper.call({"op": "id_to_token", "ids": [int(i) for i in ([ids] if single else ids)]})["tokens"]
        return tokens[0] if single else tokens

    def encode(self, text, add_special_tokens=True, **_):
        return self._helper.call({"op": "encode", "text": text, "special": bool(add_special_tokens)})["ids"]

    def decode(self, token_ids, skip_special_tokens=False, **_):
        ids = [int(i) for i in token_ids]
        text = self._helper.call({"op": "decode", "ids": ids, "skip": bool(skip_special_tokens)})["text"]
        if getattr(self._clone, "clean_up_tokenization_spaces", False):
            text = self._clone.clean_up_tokenization(text)
        return text


class _HelperTokenizerBuilder:
    def __init__(self, tokenizer, serialised: str) -> None:
        # Boot-thread part: the only reads of the serving tokenizer (its serialised form, its Python
        # attributes and its cached vocabulary). The clone's native backend is replaced before any use.
        self._vocab = tokenizer.get_vocab()
        self._special_tokens = list(tokenizer.all_special_tokens)
        self._special_ids = list(tokenizer.all_special_ids)
        self._added = _added_tokens(serialised)
        self._clone = object.__new__(type(tokenizer))
        self._clone.__dict__.update(tokenizer.__dict__)
        self._clone._tokenizer = _NoBackend()
        self._helper = _TokenizerHelper(serialised)

    def build(self):
        self._helper.start()
        return _RemoteTokenizer(
            self._clone,
            self._vocab,
            self._special_tokens,
            self._special_ids,
            self._helper,
            self._added,
        )

    def close(self) -> None:
        self._helper.close()


class _CopiedTokenizerBuilder:
    def __init__(self, tokenizer) -> None:
        self._private = copy.deepcopy(tokenizer)

    def build(self):
        return self._private

    def close(self) -> None:
        pass


def _private_tokenizer(tokenizer):
    """Boot-thread step: everything read from the serving tokenizer is read here, once, synchronously.

    Returns a builder: ``build()`` (called by the probe worker) gives the tokenizer the probe uses, and
    ``close()`` releases it. A Rust-backed tokenizer is serialised here (about 0.1 s on a 248k-entry
    vocabulary) and loaded by a short-lived helper process, so its native memory (about 65 MB plus the
    interpreter) is outside the serving process and the helper can be killed. Any other tokenizer is
    deep-copied here.
    """
    backend = getattr(tokenizer, "_tokenizer", None)
    to_str = getattr(backend, "to_str", None)
    serialised = to_str() if callable(to_str) else None
    if isinstance(serialised, str):
        return _HelperTokenizerBuilder(tokenizer, serialised)
    return _CopiedTokenizerBuilder(tokenizer)


def run(
    tokenizer,
    chat_template: str | None,
    parser_cls,
    chat_template_kwargs: dict[str, Any] | None = None,
    *,
    model_config=None,
    deadline_s: float | None = None,
    resolve_template: bool = False,
    private_builder=None,
) -> Report:
    """Run the probe and return its findings. Never raises.

    ``private_builder`` (see ``_private_tokenizer``) makes the worker use its own tokenizer, whose native
    part lives in a helper process that is killed at the deadline; a worker that outlives it holds no
    native object the server uses.

    ``resolve_template`` resolves ``chat_template`` the way the renderer does (a name, the tokenizer's own
    template, or the fallback) inside the same bounded worker. Past ``deadline_s`` the worker is stopped
    and joined before this returns, so nothing keeps using the tokenizer or the report afterwards; if it
    cannot be stopped, the report returned is a copy it does not hold.
    """
    work_report = Report()
    started = time.perf_counter()

    def work() -> None:
        try:
            tok = private_builder.build() if private_builder is not None else tokenizer
            template = (
                _resolve_template(tok, chat_template, model_config)
                if resolve_template
                else chat_template
            )
            _probe(work_report, tok, template, parser_cls, chat_template_kwargs, model_config)
        except _Cancelled:
            pass
        except _HelperError as e:
            work_report.note(str(e))
        except Exception as e:  # noqa: BLE001
            work_report.note(f"the probe stopped on {type(e).__name__}: {e}")

    thread = threading.Thread(target=work, name="parser-selftest", daemon=True)
    thread.start()
    deadline_s = DEADLINE_S if deadline_s is None else deadline_s
    thread.join(deadline_s)
    report = work_report
    if thread.is_alive():
        report = Report(list(work_report.findings), list(work_report.covered))
        if private_builder is not None:
            private_builder.close()  # kills the helper process: a request waiting on it ends at once
            thread.join(STOP_GRACE_S)
        stopped = not thread.is_alive() or _stop_thread(thread, STOP_GRACE_S)
        report.note(
            f"the probe did not finish within {deadline_s:g} s and was "
            + ("stopped" if stopped else "abandoned and could not be stopped")
        )
    if private_builder is not None:
        private_builder.close()
    report.elapsed_ms = (time.perf_counter() - started) * 1000.0
    return report


_REPORTED: set[tuple[str, str, str]] = set()


def check_at_boot(
    tokenizer,
    chat_template: str | None,
    parser_cls,
    chat_template_kwargs: dict[str, Any] | None,
    *,
    tool_parser_name: str | None,
    reasoning_parser_name: str | None,
    model_config=None,
    background: bool = False,
) -> Report | None:
    """Boot hook: probe the served pairing once per tokenizer and parser pair and log the findings.

    Nothing in here raises: the whole body is guarded. The template lookup and the probe run in one
    deadline-bounded worker on a private tokenizer, so nothing that outlives the deadline holds anything
    the server uses. With ``background`` the worker is not waited for: the hook returns after the
    tokenizer snapshot and the findings are logged when the probe ends (``wait_for_boot_checks`` joins it).
    Without it the report is returned.
    """
    try:
        return _check_at_boot(
            tokenizer,
            chat_template,
            parser_cls,
            chat_template_kwargs,
            tool_parser_name,
            reasoning_parser_name,
            model_config,
            background,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("Parser/template self-test could not run: %s: %s", type(e).__name__, e)
        return None


_PENDING: list[threading.Thread] = []


def wait_for_boot_checks(timeout: float = 60.0) -> None:
    """Join the background boot checks started so far (tests and shutdown paths)."""
    for thread in list(_PENDING):
        thread.join(timeout)
        if not thread.is_alive():
            _PENDING.remove(thread)


def _check_at_boot(
    tokenizer,
    chat_template,
    parser_cls,
    chat_template_kwargs,
    tool_parser_name,
    reasoning_parser_name,
    model_config,
    background,
) -> Report | None:
    name = "" if tokenizer is None else str(getattr(tokenizer, "name_or_path", "") or type(tokenizer).__name__)
    key = (
        name,
        reasoning_parser_name or "",
        tool_parser_name or "",
    )
    if key in _REPORTED:
        return None
    _REPORTED.add(key)
    pair = f"reasoning parser {reasoning_parser_name or 'none'!r}, tool parser {tool_parser_name or 'none'!r}"
    if parser_cls is None or tokenizer is None:
        report = Report()
        report.note(
            "no reasoning or tool parser is selected"
            if parser_cls is None
            else "the server runs without a tokenizer"
        )
        _log_report(report, pair)
        return report
    build = _private_tokenizer(tokenizer)

    def job() -> Report:
        report = run(
            tokenizer,
            chat_template,
            parser_cls,
            chat_template_kwargs,
            model_config=model_config,
            resolve_template=True,
            private_builder=build,
        )
        _log_report(report, pair)
        return report

    if not background:
        return job()

    def guarded() -> None:
        try:
            job()
        except Exception as e:  # noqa: BLE001
            logger.warning("Parser/template self-test could not run: %s: %s", type(e).__name__, e)

    thread = threading.Thread(target=guarded, name="parser-selftest-boot", daemon=True)
    _PENDING.append(thread)
    thread.start()
    return None


def _log_report(report: Report, pair: str) -> None:
    for f in report.warnings:
        logger.warning("Parser/template self-test (%s): %s", pair, f.text)
    # A note is a coverage gap: a warning when nothing was checked, an info line when part of it was.
    for f in report.notes:
        log = logger.info if report.covered else logger.warning
        log("Parser/template self-test (%s): not covered: %s", pair, f.text)
    for f in report.infos:
        logger.info("Parser/template self-test (%s): %s", pair, f.text)
    if not report.findings:
        logger.info(
            "Parser/template self-test passed (%s; %s) in %.0f ms",
            pair,
            " and ".join(report.covered),
            report.elapsed_ms,
        )


def _resolve_template(tokenizer, chat_template: str | None, model_config) -> str | None:
    from vllm.renderers.hf import resolve_chat_template

    try:
        return resolve_chat_template(
            tokenizer, chat_template, _TOOLS, model_config=model_config
        )
    except Exception:  # noqa: BLE001 - no template is a coverage note, not an error
        return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="parser_selftest",
        description="Round-trip a fixed conversation through a chat template and the selected parsers.",
    )
    ap.add_argument("--model", required=True, help="checkpoint directory (tokenizer files are enough)")
    ap.add_argument("--chat-template", help="template file; default: the checkpoint's own")
    ap.add_argument("--reasoning-parser", required=True, help="parser name, or none")
    ap.add_argument("--tool-call-parser", required=True, help="parser name, or none")
    ap.add_argument(
        "--chat-template-kwargs",
        default='{"enable_thinking": true, "reasoning_effort": "medium"}',
        help="JSON; default: the served Qwen3.8 lane defaults",
    )
    ap.add_argument("--strict", action="store_true", help="exit 1 when there is a warning")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    from pathlib import Path

    from transformers import AutoTokenizer

    from vllm.parser.parser_manager import ParserManager

    try:
        kwargs = json.loads(args.chat_template_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        template = (
            Path(args.chat_template).read_text(encoding="utf-8")
            if args.chat_template
            else getattr(tokenizer, "chat_template", None)
        )
        reasoning = None if args.reasoning_parser == "none" else args.reasoning_parser
        tool = None if args.tool_call_parser == "none" else args.tool_call_parser
        parser_cls = ParserManager.get_parser(
            tool_parser_name=tool,
            reasoning_parser_name=reasoning,
            enable_auto_tools=tool is not None,
            model_name=args.model,
            tokenizer=tokenizer,
        )
    except Exception as e:  # noqa: BLE001
        print(f"parser_selftest: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    report = run(tokenizer, template, parser_cls, kwargs)
    if args.json:
        print(
            json.dumps(
                {
                    "findings": [f.__dict__ for f in report.findings],
                    "covered": report.covered,
                    "elapsed_ms": round(report.elapsed_ms, 1),
                },
                indent=1,
            )
        )
    else:
        for f in report.findings:
            print(f"{ {'warn': 'WARN', 'note': 'NOTE'}.get(f.level, 'INFO') }: {f.text}")
        print(
            f"parser self-test: {len(report.warnings)} warning(s), {len(report.notes)} note(s); "
            f"covered: {', '.join(report.covered) or 'nothing'}; {report.elapsed_ms:.0f} ms"
        )
    return 1 if args.strict and report.warnings else 0


if __name__ == "__main__":
    sys.exit(main())
