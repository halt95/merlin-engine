# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Content starts at its first visible character after a reasoning block.

A model that reasons writes ``...</think>\\n\\nAnswer``. The newlines belong to
neither the reasoning nor the answer, so they are dropped: streamed or not, with
or without tool calls, the content starts at ``Answer``. Whitespace later in the
content (between text and tool calls, at the end) is not touched, and content
that is nothing but that whitespace is no content.

The rule applies where the output itself closes a reasoning block. When the
prompt already closed it (thinking switched off), or the parser has no
reasoning, the output has no such boundary and nothing is removed.

This module is the one place that holds the rule. The parser front ends
(``DelegatingParser`` for a reasoning parser paired with a tool parser,
``ParserEngine`` for one engine doing both) tell it when reasoning is open and
when it ends, and pass their content through it.
"""

from __future__ import annotations

from typing import Any


def reasoning_boundary(
    parser: Any, model_output: str, reasoning: str | None, content: str | None
) -> bool:
    """A reasoning split, including an empty block with an explicit closer.
    Wrapper removal alone is not evidence of reasoning (thinking may be off)."""
    closer = getattr(parser, "reasoning_end_str", None)
    return content is not None and (
        reasoning is not None
        or (bool(closer) and closer in model_output and content != model_output)
    )


def content_after_reasoning(
    content: str | None, closed: bool, tool_parser: Any = None
) -> str | None:
    """Prepare split content without changing the text/token alignment of an
    engine tool pass. Other paths consume the boundary here."""
    if not closed:
        return content
    state = getattr(
        getattr(tool_parser, "_parser_engine", None), "_content_start", None
    )
    if isinstance(state, ContentStart):
        state.arm()
        return content
    state = ContentStart(reasoning_open=True)
    state.reasoning_ended()
    return state.take(content)


class ContentStart:
    """The same rule for text that arrives in pieces (one stream, or the
    events of one parse)."""

    __slots__ = ("_armed", "_open", "_pending")

    def __init__(self, reasoning_open: bool = False) -> None:
        self._armed = False
        self.reset(reasoning_open)

    def reset(self, reasoning_open: bool = False) -> None:
        self._open = reasoning_open
        self._pending = self._armed
        self._armed = False

    def arm(self) -> None:
        """The text that comes next directly follows a reasoning block that
        another parser closed. Survives the reset that starts the stream."""
        self._armed = self._pending = True

    def reasoning_open(self) -> None:
        self._open = True

    def reasoning_ended(self) -> None:
        """Arm the strip for an open block; an end event in content does nothing."""
        if self._open:
            self._pending = True
        self._open = False

    def tool_call(self) -> None:
        """A tool call follows the block: the whitespace after the call is
        whitespace beside a tool call, which is kept."""
        self._armed = self._pending = False

    def take(self, text: str | None) -> str | None:
        """*text* without the boundary whitespace; None when nothing is left."""
        if not text:
            return text
        self._armed = False
        if self._pending:
            text = text.lstrip()
            if not text:
                return None
            self._pending = False
        return text

    def take_delta(self, delta: Any, finished: bool) -> Any:
        """Apply :meth:`take` to a delta message's content. A delta left with
        nothing is dropped unless it is the last one."""
        if delta is None:
            return delta
        had_content = bool(delta.content)
        if had_content:
            delta.content = self.take(delta.content)
        if delta.tool_calls:
            self.tool_call()
        if (
            had_content
            and delta.content is None
            and not finished
            and not delta.reasoning
            and not delta.tool_calls
        ):
            return None
        return delta


def content_start_of(parser: Any) -> ContentStart:
    """The stream state of *parser*, made on first use."""
    state = parser.__dict__.get("_merlin_content_start")
    if state is None:
        state = parser.__dict__["_merlin_content_start"] = ContentStart()
    return state


def reasoning_closed_by_output(parser: Any, tool_parser: Any) -> None:
    """The stream of *parser* (a reasoning parser paired with *tool_parser*)
    just left its reasoning block. An engine tool parser reads the content
    from here on and drops the whitespace as it reads, so that whitespace it
    holds back for its own rules is not the boundary's. Otherwise the content
    is cleaned on its way out of *parser*."""
    engine = getattr(tool_parser, "_parser_engine", None)
    state = getattr(engine, "_content_start", None)
    if isinstance(state, ContentStart):
        state.arm()
        return
    own = content_start_of(parser)
    own.reasoning_open()
    own.reasoning_ended()
