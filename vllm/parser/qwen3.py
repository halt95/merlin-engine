# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modified by the Merlin project; see NOTICE.
"""Qwen3 parser for tool calls and reasoning.

Qwen3 XML tool call format::

    <tool_call>
    <function=func_name>
    <parameter=key>value</parameter>
    </function>
    </tool_call>

The argument body consists of ``<parameter=NAME>VALUE</parameter>`` tags.
The ``_qwen3_arg_converter`` parses these into a JSON object.
"""

from __future__ import annotations

import functools
import json
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import regex as re

from vllm.merlin.tool_call_index import renumber_accepted_calls
from vllm.parser.engine.events import EventType, SemanticEvent
from vllm.parser.engine.parser_engine import ParserEngine
from vllm.parser.engine.parser_engine_config import (
    ArgValueExtender,
    ParserEngineConfig,
    ParserState,
    Transition,
)  # the two imports below
from vllm.parser.engine.streaming_parser_engine import StreamingParserEngine
from vllm.parser.engine.token_id_scanner import (
    DROP_TERMINAL,
    PreLexedTerminal,
)

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
    )
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
    from vllm.tokenizers import TokenizerLike
    from vllm.tool_parsers.abstract_tool_parser import Tool

THINK_START = "<think>"
THINK_END = "</think>"
TOOL_CALL_START = "<tool_call>"
TOOL_CALL_END = "</tool_call>"
CHATML_TURN_BOUNDARIES = frozenset(("<|im_start|>", "<|im_end|>"))
FUNC_PREFIX = "<function="
FUNC_END = "</function>"
PARAM_START = "<parameter="
PARAM_END = "</parameter>"

# These two are used only by mimo.py.
_PARAM_RE = re.compile(
    r"<\s*parameter\s*=\s*([^>]*)>"
    r"(.*?)"
    r"(?:<\s*/\s*parameter\s*>|(?=<\s*parameter\s*=))",
    re.DOTALL,
)
_PARTIAL_PARAM_RE = re.compile(r"<\s*parameter\s*=\s*([^>]+)>(.*)$", re.DOTALL)
# Structural delimiters.
_PARAM_OPEN_RE = re.compile(r"<\s*parameter\s*=\s*([^>]*)>")
_PARAM_CLOSE_RE = re.compile(r"<\s*/\s*parameter\s*>")
# A closer is structural before the next parameter, a stray call closer or the end.
_PARAM_CLOSE_TAIL_RE = re.compile(r"\s*(?:<\s*parameter\s*=|<\s*/\s*tool_call\s*>|$)")
_PARAM_OPEN_START_RE = re.compile(r"<\s*parameter\s*=")


def _trim_wrapping_newlines(value: str) -> str:
    """Strip one leading and one trailing newline (the Qwen3 template markup)."""
    if value.startswith("\n"):
        value = value[1:]
    if value.endswith("\n"):
        value = value[:-1]
    return value


def _qwen3_arg_converter(raw_args: str, partial: bool) -> str:
    params: dict[str, object] = {}
    # a value ends at the
    # first ``</parameter>`` followed by the next opener or the body end, so
    # literal ``</parameter>`` text (and ``<parameter=`` text next to it, as in
    # quoted XML) inside a value is kept.
    # A value that never closes (max_tokens, or a missing closer before
    # ``</function>``) is kept in both modes instead of dropped, so the
    # non-stream result matches what streaming already emitted.
    opener = _PARAM_OPEN_RE.search(raw_args)
    while opener is not None:
        name = opener.group(1)
        start = opener.end()
        end = resume = None
        # While streaming, a closer at the end of the text so far may still
        # turn out to be literal.
        tentative = False
        for closer in _PARAM_CLOSE_RE.finditer(raw_args, start):
            tail = _PARAM_CLOSE_TAIL_RE.match(raw_args, closer.end())
            if tail is not None:
                end, resume = closer.start(), closer.end()
                tentative = partial and not tail.group(0).strip()
                break
        value = raw_args[start:end]
        if not _PARAM_CLOSE_RE.search(value):
            # No literal closer inside: the next opener starts the next
            # parameter after a missing closer.
            next_open = _PARAM_OPEN_START_RE.search(value)
            if next_open is not None:
                settled = end is not None and not tentative
                end = resume = start + next_open.start()
                value = raw_args[start:end]
                if partial and not settled:
                    # Not settled yet: do not stream a parameter that may be
                    # value text; the value is cut here, which stays a prefix
                    # of either reading.
                    resume = None
        if name:
            params[name] = _trim_wrapping_newlines(value)
        if resume is None:
            break
        opener = _PARAM_OPEN_RE.search(raw_args, resume)

    return json.dumps(params, ensure_ascii=False)


# What ends an open value in ``_PARAM_RE``: a closing tag or the next opener.
_PARAM_VALUE_END_RE = re.compile(r"<\s*/\s*parameter\s*>|<\s*parameter\s*=")


def _param_value_end_window(text: str) -> str:
    """Return the suffix of *text* that may still grow into a value end."""
    start = text.rfind("<")
    if start == -1:
        return ""
    tail = text[start:]
    return tail if _PARAM_VALUE_END_RE.fullmatch(tail, partial=True) else ""


def _param_value_extender(
    raw_args: str, trim_newlines: bool
) -> ArgValueExtender | None:
    """Extend the open trailing ``<parameter=...>`` value of *raw_args*.

    The extender streams exactly what the converter would for the same text.
    With *trim_newlines* it follows ``_trim_wrapping_newlines``: it drops a
    leading newline of the value and holds back a trailing one.
    """
    match = _PARTIAL_PARAM_RE.search(_PARAM_RE.sub("", raw_args))
    if match is None:
        return None
    name, value = match.group(1), match.group(2)
    # A repeated key keeps its first position, so its value is not last.
    if any(m.group(1) == name for m in _PARAM_RE.finditer(raw_args)):
        return None

    return _value_extender(value, trim_newlines)


# Shared by both readings' extenders.
def _value_extender(
    value: str, trim_newlines: bool, end_re: re.Pattern = _PARAM_VALUE_END_RE
) -> ArgValueExtender:
    """Stream deltas appended to the open *value* until *end_re* completes."""
    window = _end_window(value, end_re)
    emit = _value_emitter(value, trim_newlines)

    def extend(delta: str) -> str | None:
        nonlocal window
        text = window + delta
        if end_re.search(text):
            return None
        window = _end_window(text, end_re)
        return emit(delta)

    return extend


def _end_window(text: str, end_re: re.Pattern) -> str:
    """The suffix of *text* that may still grow into an *end_re* match."""
    start = text.rfind("<")
    if start == -1:
        return ""
    tail = text[start:]
    return tail if end_re.fullmatch(tail, partial=True) else ""


def _value_emitter(value: str, trim_newlines: bool) -> Callable[[str], str]:
    """JSON-escaped text for value text appended after *value*; with
    *trim_newlines* it follows ``_trim_wrapping_newlines`` (drops a leading
    newline of the value, holds back a trailing one)."""
    started = bool(value)
    held = "\n" if trim_newlines and value.endswith("\n") and value != "\n" else ""

    def emit(delta: str) -> str:
        nonlocal started, held
        if trim_newlines:
            text = held + delta
            if not started and text:
                started = True
                text = text.removeprefix("\n")
            held = "\n" if text.endswith("\n") else ""
            text = text[: len(text) - len(held)]
        else:
            text = delta
        return json.dumps(text, ensure_ascii=False)[1:-1]

    return emit


def _qwen3_arg_value_extender(raw_args: str) -> ArgValueExtender | None:
    # The extender follows upstream's
    # _PARAM_RE reading; use it only where the anchored converter agrees.
    anchored = _qwen3_arg_converter(raw_args, True)
    if _upstream_param_json(raw_args) != anchored:
        # Literal tags: extend the
        # anchored reading's open value instead of reconverting every delta.
        return _anchored_value_extender(raw_args, anchored)
    return _param_value_extender(raw_args, trim_newlines=True)


def _merge_arg_chunks(events: list[SemanticEvent]) -> list[SemanticEvent]:
    """Join each run of argument chunks of one call after its first chunk.

    A deferred ``</function>`` replays its whole value in one delta, a chunk per
    token; converting the arguments once per chunk made that replay quadratic.
    The first chunk of a run stays alone (it may send the call name first), and
    the joined rest gives the delta the same arguments the chunks would.
    """
    merged: list[SemanticEvent] = []
    parts: list[SemanticEvent] = []  # the run's chunks after its first

    def flush() -> None:
        if len(parts) == 1:
            merged.append(parts[0])
        elif parts:
            merged.append(
                SemanticEvent(
                    EventType.ARG_VALUE_CHUNK,
                    "".join(p.value for p in parts),
                    parts[0].tool_index,
                    sum(p.token_count for p in parts),
                )
            )
        parts.clear()

    run_index: int | None = None  # tool index of the current run
    for event in events:
        if event.type == EventType.ARG_VALUE_CHUNK and event.tool_index == run_index:
            parts.append(event)
            continue
        flush()
        merged.append(event)
        is_chunk = event.type == EventType.ARG_VALUE_CHUNK
        run_index = event.tool_index if is_chunk else None
    flush()
    return merged


# Text that can change the anchored
# reading: a parameter tag, or a call closer (it can make an earlier literal
# ``</parameter>`` structural). Other text only extends the open value.
_ANCHORED_VALUE_END_RE = re.compile(
    r"<\s*/\s*parameter\s*>|<\s*parameter\s*=|<\s*/\s*(?:tool_call|function)\s*>"
)


def _anchored_value_extender(raw_args: str, anchored: str) -> ArgValueExtender | None:
    """Extend the anchored converter's open value, the last one, which runs to
    the end of *raw_args*; None when the reading has no such value.

    Appended text with none of ``_ANCHORED_VALUE_END_RE`` adds no tag, so the
    reading keeps its structure and only that value grows; a probe confirms
    it for the current text (a closer at the end could still turn literal).
    """
    current = json.loads(anchored)
    if not current:
        return None
    name, value = next(reversed(current.items()))
    if not isinstance(value, str):
        return None
    for opener in reversed(list(_PARAM_OPEN_RE.finditer(raw_args))):
        if opener.group(1) == name:
            raw = raw_args[opener.end() :]
            if _trim_wrapping_newlines(raw) == value:
                break
    else:
        return None
    probe = json.loads(_qwen3_arg_converter(raw_args + "x", True))
    expected = {**current, name: _trim_wrapping_newlines(raw + "x")}
    if list(probe.items()) != list(expected.items()):
        return None
    return _anchored_extender(raw)


# a ``</parameter>`` and the whitespace after it.
_PARAM_CLOSE_WS_RE = re.compile(r"<\s*/\s*parameter\s*>\s*")
_PARAM_CLOSE_END_RE = re.compile(r"<\s*/\s*parameter\s*>\s*$")


def _anchored_extender(raw: str) -> ArgValueExtender:
    """Stream text appended to the anchored reading's open value *raw*.

    It follows ``_qwen3_arg_converter`` without reconverting:

    * ``</parameter>`` is held with the whitespace after it. Other text after it
      makes it literal value text (the converter's closer-tail rule) and both
      stream; a ``<`` there may open the next parameter or end the call, so the
      extender stops and the engine reconverts.
    * ``<parameter=`` is literal value text once the value holds a literal
      ``</parameter>`` (the converter splits a value at an opener only when the
      value has no closer); before that it stops the extender.
    * A call closer stops it (it can make an earlier closer structural).
    """
    emit = _value_emitter(raw, True)
    literal_closer = bool(_PARAM_CLOSE_RE.search(raw))
    buf = _end_window(raw, _ANCHORED_VALUE_END_RE)
    sent = len(buf)  # buf's first chars are value text already streamed
    pending = False  # buf starts with a held ``</parameter>``
    # buf follows a ``</parameter>`` (and whitespace): a tag completing there
    # may make that closer structural.
    after_closer = bool(_PARAM_CLOSE_END_RE.search(raw, 0, len(raw) - len(buf)))

    def take(k: int) -> str:
        # Stream buf[:k] (less what was already sent) and drop it from buf.
        nonlocal buf, sent, after_closer
        if buf[:k].strip():
            after_closer = False
        done = min(sent, k)
        text, buf, sent = buf[done:k], buf[k:], sent - done
        return emit(text) if text else ""

    def extend(delta: str) -> str | None:
        nonlocal buf, pending, literal_closer
        buf += delta
        out: list[str] = []
        while True:
            if pending:
                held = _PARAM_CLOSE_WS_RE.match(buf)
                assert held is not None
                if held.end() == len(buf):
                    break  # only whitespace after the closer so far
                if buf[held.end()] == "<":
                    return None
                pending, literal_closer = False, True
                out.append(take(held.end()))
                continue
            match = _ANCHORED_VALUE_END_RE.search(buf)
            if match is None:
                window = _end_window(buf, _ANCHORED_VALUE_END_RE)
                out.append(take(len(buf) - len(window)))
                break
            out.append(take(match.start()))
            if _PARAM_CLOSE_RE.match(buf):
                pending = True
            elif after_closer:
                return None
            elif literal_closer and _PARAM_OPEN_START_RE.match(buf):
                out.append(take(match.end() - match.start()))
            else:
                return None
        return "".join(out)

    return extend


# upstream's partial conversion (the
# reading ``_param_value_extender`` streams), kept to fence the extender off
# values that hold literal tags: there the anchored converter reads a different
# open value, and a delta streamed into the wrong one breaks the JSON.
def _upstream_param_json(raw_args: str) -> str:
    params: dict[str, object] = {}
    for match in _PARAM_RE.finditer(raw_args):
        params[match.group(1)] = _trim_wrapping_newlines(match.group(2))
    match = _PARTIAL_PARAM_RE.search(_PARAM_RE.sub("", raw_args))
    if match and match.group(1):
        params[match.group(1)] = _trim_wrapping_newlines(match.group(2))
    return json.dumps(params, ensure_ascii=False)


@functools.cache
def qwen3_config(
    thinking: bool = True,
    *,
    name: str = "qwen3",
    think_start: str = THINK_START,
    think_end: str = THINK_END,
    tool_start: str = TOOL_CALL_START,
    tool_end: str = TOOL_CALL_END,
    turn_boundary_tokens: frozenset[str] = frozenset(),
) -> ParserEngineConfig:
    return ParserEngineConfig(
        name=name,
        initial_state=ParserState.REASONING if thinking else ParserState.CONTENT,
        wait_for_reasoning=thinking,
        turn_boundary_tokens=turn_boundary_tokens,
        terminals={
            # Reasoning terminals
            "THINK_START": think_start,
            "THINK_END": think_end,
            # Tool call terminals
            "TOOL_START": tool_start,
            "TOOL_END": tool_end,
            "FUNC_PREFIX": FUNC_PREFIX,
            "FUNC_END": FUNC_END,
            "PARAM_START": PARAM_START,
            "PARAM_END": PARAM_END,
            "CLOSE_ANGLE": ">",
        },
        token_id_terminals={
            "THINK_START": think_start,
            "THINK_END": think_end,
            "TOOL_START": tool_start,
            "TOOL_END": tool_end,
        },
        transitions={
            # -- Reasoning transitions --
            (ParserState.REASONING, "THINK_START"): Transition(
                ParserState.REASONING,
                (),
            ),
            (ParserState.REASONING, "THINK_END"): Transition(
                ParserState.CONTENT,
                (EventType.REASONING_END,),
            ),
            # Absorb duplicate </think> — model may emit it after
            # already transitioning to CONTENT; drop it silently.
            (ParserState.CONTENT, "THINK_END"): Transition(
                ParserState.CONTENT,
                (),
            ),
            # Tool call directly from reasoning (implicit end)
            (ParserState.REASONING, "TOOL_START"): Transition(
                ParserState.TOOL_PREAMBLE,
                (EventType.REASONING_END, EventType.TOOL_CALL_START),
            ),
            # -- Tool call transitions --
            (ParserState.CONTENT, "TOOL_START"): Transition(
                ParserState.TOOL_PREAMBLE,
                (EventType.REASONING_END, EventType.TOOL_CALL_START),
            ),
            # Fallback: <function= without a preceding <tool_call>
            (ParserState.CONTENT, "FUNC_PREFIX"): Transition(
                ParserState.TOOL_NAME,
                (EventType.TOOL_CALL_START,),
            ),
            (ParserState.TOOL_PREAMBLE, "TOOL_END"): Transition(
                ParserState.CONTENT,
                (EventType.TOOL_CALL_END,),
            ),
            (ParserState.TOOL_PREAMBLE, "FUNC_PREFIX"): Transition(
                ParserState.TOOL_NAME,
                (),
            ),
            (ParserState.TOOL_NAME, "CLOSE_ANGLE"): Transition(
                ParserState.TOOL_ARGS,
                (),
            ),
            # Malformed: </function> while still in TOOL_NAME (no closing >)
            (ParserState.TOOL_NAME, "FUNC_END"): Transition(
                ParserState.TOOL_BETWEEN,
                (EventType.TOOL_CALL_END,),
            ),
            # regression issue 58227
            # <function=...</tool_call>
            # function name is not closed, go to content
            (ParserState.TOOL_NAME, "TOOL_END"): Transition(
                ParserState.CONTENT,
                (EventType.TOOL_CALL_END,),
            ),
            # regression issue 58227
            # <function=...<tool_call>
            # unclosed tool call followed by another tool call
            (ParserState.TOOL_NAME, "TOOL_START"): Transition(
                ParserState.TOOL_PREAMBLE,
                (EventType.TOOL_CALL_END, EventType.TOOL_CALL_START),
            ),
            # regression issue 58227
            # <function=...<function=
            # unclosed function name followed by another function name
            (ParserState.TOOL_NAME, "FUNC_PREFIX"): Transition(
                ParserState.TOOL_NAME,
                (EventType.TOOL_CALL_END, EventType.TOOL_CALL_START),
            ),
            (ParserState.TOOL_ARGS, "FUNC_END"): Transition(
                ParserState.TOOL_BETWEEN,
                (EventType.TOOL_CALL_END,),
            ),
            # regression issue 58227
            # <function=abc>...<function=
            (ParserState.TOOL_ARGS, "FUNC_PREFIX"): Transition(
                ParserState.TOOL_NAME,
                (EventType.TOOL_CALL_END, EventType.TOOL_CALL_START),
            ),
            (ParserState.TOOL_ARGS, "PARAM_START"): Transition(
                ParserState.TOOL_ARGS,
                (EventType.ARG_VALUE_CHUNK,),
            ),
            (ParserState.TOOL_ARGS, "PARAM_END"): Transition(
                ParserState.TOOL_ARGS,
                (EventType.ARG_VALUE_CHUNK,),
            ),
            (ParserState.TOOL_BETWEEN, "TOOL_END"): Transition(
                ParserState.CONTENT,
                (),
            ),
            # Consecutive tool call without closing </tool_call>
            (ParserState.TOOL_BETWEEN, "TOOL_START"): Transition(
                ParserState.TOOL_PREAMBLE,
                (EventType.TOOL_CALL_START,),
            ),
            (ParserState.TOOL_BETWEEN, "FUNC_PREFIX"): Transition(
                ParserState.TOOL_NAME,
                (EventType.TOOL_CALL_START,),
            ),
        },
        arg_converter=_qwen3_arg_converter,
        arg_value_extender=_qwen3_arg_value_extender,
        stream_arg_deltas=True,
        strip_trailing_reasoning_whitespace=False,
        tool_args_json=False,
        # No calls outside request.tools.
        validate_tool_names=True,
    )


class Qwen3StreamingParserEngine(StreamingParserEngine):
    """Qwen3 engine that holds two ambiguous terminals until the next one.

    - ``<tool_call>`` while reasoning ends reasoning only when ``<function=``
      or ``</tool_call>`` follows after whitespace. Other text or terminals
      (``</think>``) mean the tag was a mention: it stays reasoning text and
      the rest of the turn is not swallowed. Content keeps the upstream rule
      (a tool block there is dropped when unparsable, #47137).
    - ``</function>`` inside a parameter value, or right after a ``</parameter>``
      (that closer stays tentative: it may be literal value text such as a
      quoted ``<function=x><parameter=y>z</parameter></function>``), is held
      together with everything after it until the question is settled:

      * a later ``</function>`` followed (after whitespace) by ``</tool_call>``,
        ``<tool_call>`` or ``<function=`` is the real call end, so everything
        before it, including the first ``</function>``, was literal value text;
      * ``</tool_call>``, ``<tool_call>`` or the end of output without such a
        later closer means the first ``</function>`` was the real call end: the
        call ends there and what followed is handled as after any call end
        (the way main handles it), never appended to an argument.

      The structural ``</parameter>``/``</function>`` tags therefore never end
      up inside an argument value unless a later closer confirmed them as
      literal.

    At the end of output a held terminal takes its structural meaning, so a
    truncated opener is still dropped. Well-formed calls take the same
    transitions, a few tokens later.
    """

    _HOLD_UNTIL: dict[str, frozenset[str]] = {
        "TOOL_START": frozenset({"FUNC_PREFIX", "TOOL_END"}),
    }
    # what may follow a real call-ending ``</function>`` (after whitespace)
    _CALL_FOLLOWERS = frozenset({"TOOL_END", "TOOL_START", "FUNC_PREFIX"})
    # Set per request, vllm/merlin/tool_none.py
    literal_tool_markup = False
    keep_think_tags = False

    def reset(self, initial_state: ParserState | None = None) -> None:
        super().reset(initial_state=initial_state)
        self._held_terminal: str | None = None
        self._held: list[tuple[str, int]] = []
        # Deferred ``</function>``: items ("term", terminal, value, tokens) or
        # ("text", text, None, tokens); the first one is always the ``</function>``.
        self._defer: list[tuple[str, str, str | None, int]] = []
        self._value_open = False
        self._value_closed = False
        # The call was opened by ``<function=`` with no ``<tool_call>``.
        self._bare_call = False
        # Set when a held ``<tool_call>`` turns out to open a call.
        self.opened_held_call = False
        self._finishing = False
        self._wrote_text = False
        # The content so far ends in a
        # ``<tool_call>`` spelled in ordinary tokens (plus whitespace).
        self._after_spelled_start = False

    def _should_hold(self, terminal: str) -> bool:
        if (self.state, terminal) not in self.config.transitions:
            return False
        if terminal == "TOOL_START":
            return self.state == ParserState.REASONING
        if terminal == "FUNC_END":
            # A closed parameter is still tentative until the call closer is
            # confirmed, so a nested literal ``</function>`` after it is held too.
            # A bare call has no wrapper to confirm a later closer against, so its
            # first ``</function>`` ends it: what follows may be another call.
            return (
                self.state == ParserState.TOOL_ARGS
                and not self._bare_call
                and (self._value_open or self._value_closed)
            )
        return False

    def _track_value(self, terminal: str) -> None:
        if self.state != ParserState.TOOL_ARGS:
            self._value_open = self._value_closed = False
        elif terminal == "PARAM_START":
            self._value_open, self._value_closed = True, False
        elif terminal == "PARAM_END":
            self._value_open, self._value_closed = False, True

    def _release(self, structural: bool) -> list[SemanticEvent]:
        terminal = self._held_terminal
        assert terminal is not None
        (tag, tag_tokens), *rest = self._held
        self._held_terminal, self._held = None, []
        if structural:
            events = super()._on_terminal(terminal, tag, tag_tokens)
            self._track_value(terminal)
            self.opened_held_call = terminal == "TOOL_START"
        else:
            events = super()._emit_for_state(tag, tag_tokens)
        for text, token_count in rest:
            events.extend(self._emit_for_state(text, token_count))
        return events

    def _replay(self, items: list, literal: bool) -> list[SemanticEvent]:
        """Feed deferred items back in order; ``literal`` keeps a ``</function>`` as value text."""
        events: list[SemanticEvent] = []
        for kind, a, b, token_count in items:
            if kind == "text":
                events.extend(self._emit_for_state(a, token_count))
            elif literal and a == "FUNC_END":
                events.extend(super()._emit_for_state(b, token_count))
            else:
                events.extend(self._on_terminal(a, b, token_count))
        return events

    def _resolve(self, k: int) -> list[SemanticEvent]:
        """The deferred ``</function>`` at item ``k`` is the real call end; items before it were literal value text."""
        items, self._defer = self._defer, []
        events = self._replay(items[:k], literal=True)
        _, terminal, tag, tag_tokens = items[k]
        events.extend(super()._on_terminal(terminal, tag, tag_tokens))
        self._track_value(terminal)
        events.extend(self._replay(items[k + 1 :], literal=False))
        return events

    def _clean_closer(self) -> int | None:
        """Index of the last deferred ``</function>`` when only whitespace follows it, else None."""
        for k in range(len(self._defer) - 1, -1, -1):
            kind, a, _, _ = self._defer[k]
            if kind == "term" and a == "FUNC_END":
                return k
            if kind != "text" or a.strip():
                return None
        return None

    def _defer_terminal(
        self, terminal: str, value: str, token_count: int
    ) -> list[SemanticEvent]:
        if self._has_drops and terminal == DROP_TERMINAL:
            return []
        if terminal in self._CALL_FOLLOWERS:
            k = self._clean_closer()
            if k is not None:
                events = self._resolve(k)
            elif terminal == "FUNC_PREFIX":
                # ``<function=`` inside a value (a nested quote): literal text
                self._defer.append(("term", terminal, value, token_count))
                return []
            else:
                events = self._resolve(0)
            events.extend(self._on_terminal(terminal, value, token_count))
            return events
        self._defer.append(("term", terminal, value, token_count))
        return []

    # tool_choice=none.
    def _stays_text(self, terminal: str) -> bool:
        """A tool opener, or (keep_think_tags) a think tag the grammar would swallow."""
        transition = self.config.transitions.get((self.state, terminal))
        if transition is None:
            return False
        if transition.next_state in self._TOOL_STATES:
            return True
        return (
            self.keep_think_tags
            and transition.next_state == self.state
            and not transition.events
            # A ``<think>`` before any text only opens the reasoning block.
            and (terminal != "THINK_START" or self._wrote_text)
        )

    # a special token returned as text counts
    # as one token; the base loop passes none for it, which auto keeps.
    def _process_scanner_items(self, items) -> list[SemanticEvent]:
        if not self.literal_tool_markup:
            return super()._process_scanner_items(items)
        events: list[SemanticEvent] = []
        for item in items:
            if isinstance(item, PreLexedTerminal):
                events.extend(self._process_lex_tokens(self._lexer.flush()))
                events.extend(self._on_terminal(item.terminal, item.text, 1))
            else:
                events.extend(super()._process_scanner_items([item]))
        return events

    # With token ids, a ``<tool_call>``
    # spelled in ordinary tokens is text (the base engine's strict mode); the
    # ``<function=`` written after it is the same quoted markup, not the bare-call
    # fallback, so it stays text too. A real ``<tool_call>`` token, or
    # ``<function=`` anywhere else, is parsed normally.
    def _process_lex_tokens(self, tokens) -> list[SemanticEvent]:
        if not self._ever_had_token_ids:
            return super()._process_lex_tokens(tokens)
        events: list[SemanticEvent] = []
        for tok in tokens:
            if (
                tok.terminal == "FUNC_PREFIX"
                and self._after_spelled_start
                and self.state == ParserState.CONTENT
            ):
                events.extend(self._emit_for_state(tok.value, tok.token_count))
                self._after_spelled_start = False
                continue
            events.extend(super()._process_lex_tokens([tok]))
            if (
                tok.terminal == "TOOL_START"
                and tok.terminal in self._token_id_terminal_names
                and self.state == ParserState.CONTENT
            ):
                self._after_spelled_start = True
        return events

    def _on_terminal(
        self, terminal: str, value: str, token_count: int = 0
    ) -> list[SemanticEvent]:
        self._after_spelled_start = False
        if self.literal_tool_markup and self._stays_text(terminal):
            return self._emit_for_state(value, token_count)
        if self._defer:
            return self._defer_terminal(terminal, value, token_count)
        held = self._held_terminal
        if held is not None:
            if self._has_drops and terminal == DROP_TERMINAL:
                return []
            events = self._release(terminal in self._HOLD_UNTIL[held])
            events.extend(self._on_terminal(terminal, value, token_count))
            return events
        if self._should_hold(terminal):
            if terminal == "FUNC_END":
                self._defer = [("term", terminal, value, token_count)]
            else:
                self._held_terminal = terminal
                self._held = [(value, token_count)]
            return []
        if (
            terminal == "FUNC_PREFIX"
            and self.state == ParserState.TOOL_ARGS
            and (self._value_open or self._value_closed)
        ):
            # 0.33's (TOOL_ARGS, FUNC_PREFIX) recovery (#58227) is for a call
            # with no argument yet; inside a value ``<function=`` is quoted text.
            return self._emit_for_state(value, token_count)
        if terminal == "FUNC_PREFIX" and self.state != ParserState.TOOL_ARGS:
            self._bare_call = self.state != ParserState.TOOL_PREAMBLE
        events = super()._on_terminal(terminal, value, token_count)
        self._track_value(terminal)
        return events

    def _emit_for_state(self, text: str, token_count: int = 0) -> list[SemanticEvent]:
        self._wrote_text = self._wrote_text or bool(text.strip())
        if text.strip():
            self._after_spelled_start = False
        if self._defer:
            self._defer.append(("text", text, None, token_count))
            return []
        if self._held_terminal is not None:
            if self._finishing or not text.strip():
                self._held.append((text, token_count))
                return []
            events = self._release(structural=False)
            events.extend(self._emit_for_state(text, token_count))
            return events
        if self._value_closed and self.state == ParserState.TOOL_ARGS and text.strip():
            # Text after ``</parameter>``: that closer was literal value text.
            self._value_open, self._value_closed = True, False
        return super()._emit_for_state(text, token_count)

    def finish(self) -> list[SemanticEvent]:
        self._finishing = True
        pending = self._scanner.flush_pending()
        # a marker still deferred here never
        # lined up with text. Its text, if any, is already in the pending text, so
        # returning it as text as well would write what the model did not.
        if self.literal_tool_markup:
            pending = [p for p in pending if not isinstance(p, PreLexedTerminal)]
        events = self._process_scanner_items(pending)
        events.extend(self._process_lex_tokens(self._lexer.flush()))
        self._finishing = False
        if self._held_terminal is not None:
            events.extend(self._release(structural=True))
        while self._defer:
            # the output ended with no later closer: the first deferred </function> was real
            events.extend(self._resolve(0))
        self._record_reasoning_tokens(events)
        events.extend(super().finish())
        return events


class Qwen3Parser(ParserEngine):
    """Qwen3 parser: ``<think>``/``</think>`` reasoning +
    ``<tool_call>`` XML tool calls in a single engine.

    - ``<tool_call>`` as implicit reasoning end (a grammar transition, so it
      also feeds ``is_reasoning_end`` and the structured-output gate)

    Subclasses that share the grammar but differ only in the four wrapper
    token strings (reasoning + tool-call) override the class attributes
    below; everything else is inherited unchanged.
    """

    CONFIG_NAME = "qwen3"
    THINK_START = THINK_START
    THINK_END = THINK_END
    TOOL_START = TOOL_CALL_START
    TOOL_END = TOOL_CALL_END
    TURN_BOUNDARIES: frozenset[str] = CHATML_TURN_BOUNDARIES
    engine_cls = Qwen3StreamingParserEngine

    def __init__(
        self,
        tokenizer: TokenizerLike,
        tools: list[Tool] | None = None,
        **kwargs,
    ) -> None:
        chat_kwargs = kwargs.get("chat_template_kwargs", {}) or {}
        self.thinking_enabled = chat_kwargs.get("enable_thinking", True)
        kwargs.setdefault(
            "parser_engine_config",
            qwen3_config(
                thinking=self.thinking_enabled,
                name=self.CONFIG_NAME,
                think_start=self.THINK_START,
                think_end=self.THINK_END,
                tool_start=self.TOOL_START,
                tool_end=self.TOOL_END,
                turn_boundary_tokens=self.TURN_BOUNDARIES,
            ),
        )
        self._held_call_ids: list[int] = []
        self._content_ids_prefix: list[int] = []
        super().__init__(
            tokenizer,
            tools,
            **kwargs,
        )

    # a held ``<tool_call>`` can end
    # reasoning one or more deltas after its token. The DelegatingParser hands
    # the tool parser only the confirming delta's ids, so prepend the ids from
    # the ``<tool_call>`` token on; without them the tag reads as plain text.
    def _feed(
        self,
        delta_text: str,
        delta_token_ids: Sequence[int],
    ) -> list[SemanticEvent]:
        engine = self._engine
        held_ids = self._held_call_ids if engine._held_terminal == "TOOL_START" else []
        events = super()._feed(delta_text, delta_token_ids)
        self._content_ids_prefix = held_ids if engine.opened_held_call else []
        engine.opened_held_call = False
        if engine._held_terminal != "TOOL_START":
            self._held_call_ids = []
        elif held_ids:
            # In place: a long run of whitespace deltas stays linear.
            held_ids.extend(delta_token_ids)
        else:
            ids = list(delta_token_ids)
            start = self.vocab.get(self.TOOL_START)
            at = len(ids) - 1 - ids[::-1].index(start) if start in ids else len(ids)
            self._held_call_ids = ids[at:]
        return events

    # Strip as non-stream does: a padded
    # offered name is not out of set.
    def _handle_tool_name(self, event: SemanticEvent) -> None:
        super()._handle_tool_name(event)
        slot = self._tool_slots[event.tool_index]
        slot.name = slot.name.strip()

    # a rejected call leaves no index gap.
    def _events_to_delta(self, events, finished=False):
        events = _merge_arg_chunks(events)
        delta = super()._events_to_delta(events, finished=finished)
        return renumber_accepted_calls(delta, self._tool_slots)

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        prefix, self._content_ids_prefix = self._content_ids_prefix, []
        if prefix:
            return prefix + list(input_ids)
        return super().extract_content_ids(input_ids)

    def extract_reasoning(
        self,
        model_output: str,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> tuple[str | None, str | None]:
        if not self.thinking_enabled:
            return None, model_output
        return super().extract_reasoning(model_output, request)
