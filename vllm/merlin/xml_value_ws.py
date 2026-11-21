# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""One parse for the whitespace around a free-form tool argument value.

A tool without a schema the grammar enforces (``"strict"`` not true, with
``tool_choice`` "required" or a named tool) gets xgrammar's generic argument object
(a JSON-schema element whose schema is ``true``). In the XML styles below, xgrammar
writes each value as ``[ws]* xml_any [ws]*`` between the opening and closing tags,
with ``xml_any ::= xml_string | basic_array | basic_object``, and the free text
``xml_string`` takes whitespace too. A run of n whitespace characters at the start
(or end) of a value then has n + 1 parses, the matcher keeps every one, and its
memory grows with the square of the run (2.9 GB at 12,288 characters on xgrammar
0.2.8) while its fills slow down.

``one_value_parse`` compiles the same language with the value written as::

    xml_value ::= xml_string | [ws]* basic_array [ws]* | [ws]* basic_object [ws]*

``xml_string`` is any text without the closing tag, so whitespace around one is
part of the string: ``[ws]* xml_string [ws]*`` and ``xml_string`` are the same set
of texts. Arrays and objects keep their padding, which only they can take. The
element becomes xgrammar's own grammar for it, read back from its printout with that
one rule changed. Elements of any other shape, style or whitespace cap are compiled
as given.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

import regex as re

# the styles whose generic object pads a value that may be free text
_STYLES = frozenset({"qwen_xml", "minimax_xml", "glm_xml", "kimi_k3_xml"})
_WS = "[ \\n\\r\\t]*"
_ANY = "((xml_string) | (basic_array) | (basic_object))"
_PADDED = f"{_WS} xml_any {_WS}"
_VALUE = f"((xml_string) | ({_WS} basic_array {_WS}) | ({_WS} basic_object {_WS}))"


def _rules(text: str) -> list[tuple[str, str]]:
    """(name, body) of each rule of an xgrammar printout, in order."""
    starts = list(re.finditer(r"(?m)^([A-Za-z_][A-Za-z0-9_]*) ::= ", text))
    out = []
    for i, m in enumerate(starts):
        end = starts[i + 1].start() if i + 1 < len(starts) else len(text)
        out.append((m.group(1), text[m.end() : end].rstrip("\n")))
    return out


@lru_cache(maxsize=64)
def _value_grammar(element_json: str) -> str | None:
    """The generic object's grammar with one parse per value, or None."""
    import xgrammar as xgr

    try:
        text = str(
            xgr.Grammar.from_structural_tag(
                json.dumps(
                    {"type": "structural_tag", "format": json.loads(element_json)}
                )
            )
        )
    except (RuntimeError, ValueError):
        return None
    rules = _rules(text)
    bodies = dict(rules)
    if bodies.get("xml_any") != _ANY or "xml_value" in bodies:
        return None
    users = [name for name, body in rules if "xml_any" in body and name != "xml_any"]
    if sorted(users) != ["xml_object", "xml_object_properties"]:
        return None
    for name in users:
        if bodies[name].count(_PADDED) != 1 or bodies[name].count("xml_any") != 1:
            return None
        bodies[name] = bodies[name].replace(_PADDED, "xml_value")
    lines = [f"{name} ::= {bodies[name]}" for name, _ in rules if name != "xml_any"]
    grammar = "\n".join(lines + [f"xml_value ::= {_VALUE}"])
    try:
        xgr.Grammar.from_ebnf(grammar)
    except (RuntimeError, ValueError):
        return None
    return grammar


def _generic(node: dict) -> bool:
    return (
        node.get("type") == "json_schema"
        and node.get("json_schema") is True
        and node.get("style") in _STYLES
        and node.get("max_whitespace_cnt") is None
    )


def _walk(node: Any) -> int:
    count = 0
    if isinstance(node, dict):
        if _generic(node):
            grammar = _value_grammar(json.dumps(node, sort_keys=True))
            if grammar is not None:
                node.clear()
                node.update({"type": "grammar", "grammar": grammar})
                return 1
        for value in node.values():
            count += _walk(value)
    elif isinstance(node, list):
        for value in node:
            count += _walk(value)
    return count


@lru_cache(maxsize=256)
def one_value_parse(spec: str) -> str:
    """``spec`` with every generic argument object written with one parse for the
    whitespace around a value; the same language. Unchanged when nothing applies."""
    try:
        doc = json.loads(spec)
    except ValueError:
        return spec
    if not isinstance(doc, dict) or doc.get("type") != "structural_tag":
        return spec
    if not _walk(doc):
        return spec
    return json.dumps(doc)
