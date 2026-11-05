# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Lossless ``tool_choice="none"``, Qwen-family parsers.

A caller who asked for no tools gets no tool call, and gets the tool markup the
model wrote back as text in the field it was written in. Markup that would open
a call (``<tool_call>``, ``<function=``) is plain text in this mode, in reasoning
as well as content: it neither ends reasoning nor enters the tool states, so
nothing after it can be consumed as arguments.

The mode lives on ``Qwen3StreamingParserEngine``: the families built on that
engine are the ones this has been proven on. Other engines have no such
attributes and keep upstream's behaviour, no calls with the markup removed. Their
grammars differ in ways the Qwen rule does not fit (Inkling shares its block
closer with text blocks and buffers a message header; Gemma 4 flushes deferred
markers at the end).

A request with no tools is ``tool_choice="none"`` at the API, so tool markup stays
text there too. Think tags the grammar swallows
are kept only when the request offers tools: a repeated ``</think>`` is a known
model habit, and plain chat should not show it.
"""

from __future__ import annotations

from typing import Any


def literal_tool_markup(request: Any) -> bool:
    """Whether this request's output keeps tool markup as text."""
    return getattr(request, "tool_choice", None) == "none"


def keep_think_tags(request: Any) -> bool:
    """Whether this request also keeps a later ``<think>`` and a second ``</think>``."""
    return literal_tool_markup(request) and bool(getattr(request, "tools", None))


def set_mode(engine: Any, request: Any) -> None:
    """Set the mode from the request, on a ``StreamingParserEngine`` that has it."""
    if hasattr(engine, "literal_tool_markup"):
        engine.literal_tool_markup = literal_tool_markup(request)
        engine.keep_think_tags = keep_think_tags(request)


def copy_mode(source: Any, target: Any) -> None:
    """Give a second engine the mode of the one that parsed the output."""
    if hasattr(source, "literal_tool_markup"):
        target.literal_tool_markup = source.literal_tool_markup
        target.keep_think_tags = source.keep_think_tags


def mark_delegates(parser: Any, request: Any) -> None:
    """Set the mode on the engines behind a ``DelegatingParser``'s two passes.

    The reasoning adapter's streaming entry point is not given the request, so
    both passes are set here.
    """
    for delegate in (parser.reasoning_parser, parser.tool_parser):
        parser_engine = getattr(delegate, "_parser_engine", None)
        if parser_engine is not None:
            set_mode(parser_engine._engine, request)
