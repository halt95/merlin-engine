# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Strict bounded strings with the bound kept out of the grammar.

A strict Qwen tool's top-level string parameter with ``maxLength N`` is a grammar rule
with a character budget (``root[max_chars=N]``, structural_tag_registry.py). xgrammar
cannot cache the token masks of a budgeted rule: every fill inside the value checks the
vocabulary against the remaining count, and in the last characters before the bound a
single fill costs up to a few hundred milliseconds. The engine fills those masks one
request and one row after another on the engine-core thread, so each strict stream adds
its fill time to every step.

With ``VLLM_MERLIN_STRICT_LENGTH_TABLE=1`` the engine compiles the same tag with the
budget attribute removed (the "twin": the same rule, so the same language except the
bound) and applies the bound itself:

* ``LengthTable``, once per tokenizer: for every token, how many code points it adds to
  a bounded value, counted the way the budget counts them (a character counts at its
  lead byte; continuation bytes count nothing), with the token split where it completes
  the closing sequence. Token masks for "at most r more" are built from it once and
  reused.
* ``CallTracker``, per request: a mirror of the tag's own structure (call opener token,
  function header, the parameters in declared order, the value automaton of each
  bounded block, the call closer token). It advances with every token the matcher
  accepts and rolls back with it, so it always knows whether the matcher is inside a
  bounded value and how many characters are left.
* A row is the twin's mask AND the table's mask for the characters left. A token that
  leaves the value and could reach another bounded value within itself is checked
  exactly by running the tracker over it. Accepting a token runs the tracker too, so a
  draft the bound refuses is refused.

A string nested inside a JSON-valued parameter with ``maxLength N`` (and no other
keyword) is the same case in xgrammar's JSON grammar: a quote, ``{0, N}`` repetitions
of one character class, a quote (no escapes in a bounded string). The twin replaces
that parameter's element by xgrammar's own grammar text for it with each such
repetition made unbounded, and the table counts a token's characters up to the
closing quote.

The tracker is exact only where the tag's structure is unambiguous: a bounded value can
be entered only after parameters whose end cannot be mistaken for text (other bounded
blocks, JSON-valued parameters, enum strings without ``<``, plain strings, whose value
ends at the first ``</parameter>``). A bound behind any other string parameter (whose
value may contain the closing tag as text) stays in the grammar, and so
does every bound in a tag shape this module does not recognise. Every such case is a log
line and a counter (``fallback_counts``), never an approximation. With the switch off,
nothing here runs and the engine compiles the tag it was given.
"""

from __future__ import annotations

import copy
import json
import threading
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from functools import cache
from typing import TYPE_CHECKING, Any

import numpy as np
import regex as re

from vllm import envs
from vllm.logger import init_logger
from vllm.merlin.tag_whitespace import one_run_per_gap, space_bound
from vllm.v1.structured_output.backend_xgrammar import XgrammarGrammar

if TYPE_CHECKING:
    import torch
    import xgrammar as xgr

logger = init_logger(__name__)

CLOSER = b"\n</parameter>"
_CLOSER_LEN = len(CLOSER)
_BLOCK_END = "\n</parameter>"
_RULE_HEAD = "root[max_chars={}] ::= s0\n"
_TWIN_HEAD = "root ::= s0\n"

# Tracker modes.
FREE, HEAD, ARGS, PRE_BLOCK, BLOCK, VALUE, ENUM, END, DONE, DEAD, LOST, TEXT = range(12)

fallback_counts: Counter[str] = Counter()
fast_count = 0
_counts_lock = threading.Lock()


def enabled() -> bool:
    return bool(envs.VLLM_MERLIN_STRICT_LENGTH_TABLE)


def _count_fallback(reason: str) -> None:
    with _counts_lock:
        fallback_counts[reason] += 1
        n = fallback_counts[reason]
    logger.info(
        "Strict length table: a bound stays in the grammar (%s); %d request(s) so far.",
        reason,
        n,
    )


# ------------------------------------------------------------------ tag analysis


@dataclass(frozen=True)
class Node:
    """A JSON schema node the tracker follows.

    kind: "o" object, "a" array, "s" string, "x" anything else.
    """

    kind: str
    props: dict[str, int] | None = None
    items: int = -1
    bound: int | None = None


@dataclass(frozen=True)
class Param:
    name: str
    kind: str  # "block", "json", "enum", "text" or "raw"
    bound: int | None  # applied by the table; None: unbounded or kept in the grammar
    nodes: tuple[Node, ...] = ()  # JSON parameter: its schema nodes, root first
    low: int | None = None  # JSON parameter: its smallest table bound


@dataclass(frozen=True)
class Tool:
    name: str
    params: dict[str, tuple[int, Param]]
    # smallest table bound among the parameters at index >= i (None: no table bound)
    min_bound_from: tuple[int | None, ...]
    # Whitespace runs the tracker caps (None: none, or the cap stays in the grammar):
    # every site holds ws_cap characters; where a parameter written by xgrammar ends,
    # its trailing site meets the next one and a run holds after_value characters.
    ws_cap: int | None = None
    after_value: int = 0
    # can_end[i]: the call may end once the parameters before index i are done
    can_end: tuple[bool, ...] = ()


@dataclass(frozen=True)
class Plan:
    twin: str
    tools: dict[str, Tool]
    triggered: bool
    opener: int
    closer: int
    kept: tuple[str, ...]
    min_bound: int | None


class _Unrecognised(Exception):
    pass


# A plain string parameter: xgrammar writes its value as text that ends at the first
# "</parameter>" (a TagDispatch excluding it), so the tracker can follow it exactly.
_TEXT_KEYS = frozenset({"type", "description", "title", "default", "examples"})
_TEXT_END = b"</parameter>"


def _schema_kind(schema: Any) -> str:
    """How a top-level parameter's value is written in the qwen_xml style."""
    if not isinstance(schema, dict):
        return "raw"
    kind = schema.get("type")
    if kind in ("object", "array", "integer", "number", "boolean", "null"):
        return "json"
    if kind == "string":
        if set(schema) <= _TEXT_KEYS:
            return "text"
        values = schema.get("enum")
        if (
            isinstance(values, list)
            and values
            and all(isinstance(v, str) and "<" not in v for v in values)
        ):
            return "enum"
    return "raw"


def _is_space(element: Any) -> bool:
    """The whitespace run between parameters, capped or not."""
    return space_bound(element) is not None


def _block_bound(element: Any) -> tuple[str, int] | None:
    """(name, N) when ``element`` is exactly the bounded string block, else None."""
    from vllm.tool_parsers.structural_tag_registry import _bounded_value_rule

    if not isinstance(element, dict) or element.get("type") != "sequence":
        return None
    parts = element.get("elements")
    if not isinstance(parts, list) or len(parts) != 2:
        return None
    space, tag = parts
    if not _is_space(space):
        return None
    if not isinstance(tag, dict) or tag.get("type") != "tag":
        return None
    begin, end, content = tag.get("begin"), tag.get("end"), tag.get("content")
    if not isinstance(begin, str) or end != _BLOCK_END or not isinstance(content, dict):
        return None
    if not (begin.startswith("<parameter=") and begin.endswith(">\n")):
        return None
    name = begin[len("<parameter=") : -2]
    if content.get("type") != "grammar" or set(content) != {"type", "grammar"}:
        return None
    text = content["grammar"]
    if not isinstance(text, str) or not text.startswith("root[max_chars="):
        return None
    try:
        limit = int(text[len("root[max_chars=") : text.index("]")])
    except ValueError:
        return None
    if text != _bounded_value_rule(limit):
        return None
    return name, limit


_ANNOTATIONS = frozenset({"description", "title", "default", "examples"})


def _count_max_length(node: Any) -> int:
    if isinstance(node, dict):
        return ("maxLength" in node) + sum(_count_max_length(v) for v in node.values())
    if isinstance(node, list):
        return sum(_count_max_length(v) for v in node)
    return 0


def _json_nodes(schema: Any) -> tuple[tuple[Node, ...], list[int]]:
    """The nodes of a JSON parameter's schema and the bounds of its modelled strings.

    Followed: closed objects, arrays with one item schema, strings. A string node is
    bounded when its schema is exactly ``type: string`` + ``maxLength`` (annotations
    aside). Anything else is an opaque node ("x"); a maxLength under it is not modelled.
    """
    nodes: list[Node] = []
    bounds: list[int] = []

    def build(sub: Any) -> int:
        index = len(nodes)
        nodes.append(Node("x"))
        if not isinstance(sub, dict):
            return index
        kind = sub.get("type")
        keys = set(sub) - _ANNOTATIONS
        if (
            kind == "object"
            and sub.get("additionalProperties") is False
            and isinstance(sub.get("properties"), dict)
            and keys <= {"type", "properties", "required", "additionalProperties"}
        ):
            props = {name: build(child) for name, child in sub["properties"].items()}
            nodes[index] = Node("o", props=props)
        elif (
            kind == "array"
            and isinstance(sub.get("items"), dict)
            and keys
            <= {
                "type",
                "items",
                "minItems",
                "maxItems",
            }
        ):
            nodes[index] = Node("a", items=build(sub["items"]))
        elif kind == "string":
            limit = sub.get("maxLength")
            least = sub.get("minLength", 0)
            if (
                keys in ({"type", "maxLength"}, {"type", "maxLength", "minLength"})
                and isinstance(limit, int)
                and not isinstance(limit, bool)
                and 1 <= limit <= 10**6
                # the minimum stays in the twin's repetition; the table applies the max
                and isinstance(least, int)
                and not isinstance(least, bool)
                and 0 <= least <= limit
            ):
                nodes[index] = Node("s", bound=limit)
                bounds.append(limit)
            else:
                nodes[index] = Node("s")
        return index

    build(schema)
    return tuple(nodes), bounds


@cache
def _plain_class() -> str:
    """The body xgrammar 0.2.8 gives the character rule of a bounded JSON string."""
    import xgrammar as xgr

    text = str(xgr.Grammar.from_json_schema('{"type": "string", "maxLength": 3}'))
    return re.search(r"(?m)^root_characters ::= (.*)$", text).group(1)


def _rules(text: str) -> list[tuple[str, str]]:
    """(name, body) of each rule of an xgrammar grammar printout, in order."""
    starts = list(re.finditer(r"(?m)^([A-Za-z_][A-Za-z0-9_]*) ::= ", text))
    out = []
    for i, m in enumerate(starts):
        end = starts[i + 1].start() if i + 1 < len(starts) else len(text)
        out.append((m.group(1), text[m.end() : end].rstrip("\n")))
    return out


def _twin_element(element: Any, bounds: list[int]) -> dict | None:
    """The element as xgrammar's own grammar text with its bounded string repetitions
    unbounded, or None when the repetitions found are not exactly ``bounds``."""
    import xgrammar as xgr

    text = str(
        xgr.Grammar.from_structural_tag(
            json.dumps({"type": "structural_tag", "format": element})
        )
    )
    rules = _rules(text)
    plain = {name for name, body in rules if body == _plain_class()}
    if not plain:
        return None
    use = re.compile(
        r"\b(" + "|".join(map(re.escape, sorted(plain))) + r")\{(\d+), (\d+)\}"
    )
    found = sorted(int(m.group(3)) for _, body in rules for m in use.finditer(body))
    if found != sorted(bounds):
        return None

    def unbounded(m: Any) -> str:
        # a minimum (minLength) stays in the grammar; only the maximum moves out
        return m.group(1) + ("*" if m.group(2) == "0" else "{" + m.group(2) + ", -1}")

    names = {name for name, _ in rules}
    bodies = {name: use.sub(unbounded, body) for name, body in rules}
    keep, todo = set(), ["root"]
    while todo:
        name = todo.pop()
        if name in keep or name not in bodies:
            continue
        keep.add(name)
        todo += [
            w for w in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", bodies[name]) if w in names
        ]
    grammar = "\n".join(
        f"{name} ::= {bodies[name]}" for name, _ in rules if name in keep
    )
    # The printout must read back (TagDispatch rules included); the masks of the
    # result are what the oracle tests compare.
    try:
        xgr.Grammar.from_ebnf(grammar)
    except (RuntimeError, ValueError):
        return None
    return {"type": "grammar", "grammar": grammar}


def _arguments(
    args: Any, eligible: bool, kept: list[str]
) -> list[tuple[str, str, int | None, Any]]:
    """The parameters of one call body as (name, kind, table bound, element)."""
    out: list[tuple[str, str, int | None, Any]] = []
    if isinstance(args, dict) and args.get("type") == "json_schema":
        schema = args.get("json_schema")
        if args.get("style") != "qwen_xml" or args.get("any_order") is not False:
            raise _Unrecognised("argument element style")
        if schema is True or schema == {}:
            return [("*", "raw", None, args)]
        if not isinstance(schema, dict) or schema.get("type") != "object":
            raise _Unrecognised("argument schema")
        props = schema.get("properties") or {}
        if not isinstance(props, dict):
            raise _Unrecognised("argument properties")
        if schema.get("additionalProperties") is not False:
            return [("*", "raw", None, args)]
        # The generic argument grammar: xgrammar writes the whole object. Its JSON
        # parameters' bounded strings are taken out together (see _analyse); a raw
        # string parameter makes every bound after it ambiguous, as in a sequence.
        ambiguous = False
        for name, sub in props.items():
            kind = _schema_kind(sub)
            if kind == "raw":
                ambiguous = True
            if kind != "json":
                out.append((name, kind, None, args))
                continue
            nodes, bounds = _json_nodes(sub)
            if bounds and len(bounds) != _count_max_length(sub):
                kept.append(
                    "a nested bound with other keywords or under an unfollowed schema"
                )
                bounds = []
            elif bounds and (not eligible or ambiguous):
                kept.append("a nested bound after a raw string parameter")
                bounds = []
            out.append((name, kind, (nodes, bounds), args))
        return out
    if not isinstance(args, dict) or args.get("type") != "sequence":
        raise _Unrecognised("argument element")
    elements = args.get("elements")
    if not isinstance(elements, list) or not elements:
        raise _Unrecognised("argument sequence")
    if not _is_space(elements[-1]):
        raise _Unrecognised("argument sequence end")
    ambiguous = False
    for element in elements[:-1]:
        inner = element
        if isinstance(element, dict) and element.get("type") == "optional":
            inner = element.get("content")
        block = _block_bound(inner)
        if block is not None:
            name, limit = block
            if eligible and not ambiguous:
                out.append((name, "block", limit, inner))
            else:
                kept.append("a bounded string after a raw string parameter")
                out.append((name, "block", None, inner))
            continue
        if not isinstance(inner, dict) or inner.get("type") != "json_schema":
            raise _Unrecognised("argument part")
        schema = inner.get("json_schema")
        if (
            inner.get("style") != "qwen_xml"
            or inner.get("any_order") is not False
            or not isinstance(schema, dict)
            or schema.get("type") != "object"
            or schema.get("additionalProperties") is not False
        ):
            raise _Unrecognised("argument part schema")
        props = schema.get("properties")
        if not isinstance(props, dict) or len(props) != 1:
            raise _Unrecognised("argument part properties")
        ((name, sub),) = props.items()
        kind = _schema_kind(sub)
        if kind == "raw":
            ambiguous = True
        if kind == "json" and _count_max_length(sub):
            nodes, bounds = _json_nodes(sub)
            if len(bounds) != _count_max_length(sub):
                kept.append(
                    "a nested bound with other keywords or under an unfollowed schema"
                )
                bounds = []
            elif not eligible or ambiguous:
                kept.append("a nested bound after a raw string parameter")
                bounds = []
            out.append((name, kind, (nodes, bounds), inner))
            continue
        if kind == "json":
            out.append((name, kind, (_json_nodes(sub)[0], []), inner))
            continue
        out.append((name, kind, None, inner))
    return out


_WS = b" \t\n\r"


def _json_elements(node: Any) -> list[dict]:
    """The JSON-schema elements of an argument grammar, outermost first."""
    if isinstance(node, dict):
        own = [node] if node.get("type") == "json_schema" else []
        return own + [e for v in node.values() for e in _json_elements(v)]
    if isinstance(node, list):
        return [e for v in node for e in _json_elements(v)]
    return []


def _uncap(args: Any) -> None:
    """Take the whitespace cap out of the twin's JSON elements and its runs between
    parameters; the tracker holds it. With every run unbounded the twin is written
    with one run per gap (tag_whitespace), so its matcher stays flat however long a
    run grows before the tracker refuses it."""
    from vllm.tool_parsers.structural_tag_registry import _QWEN_XML_SPACE

    for element in _json_elements(args):
        element["max_whitespace_cnt"] = None
    for element in _space_elements(args):
        element["pattern"] = _QWEN_XML_SPACE


def _space_elements(node: Any) -> list[dict]:
    """The whitespace-run elements of an argument grammar."""
    if isinstance(node, dict):
        own = [node] if _is_space(node) else []
        return own + [e for v in node.values() for e in _space_elements(v)]
    if isinstance(node, list):
        return [e for v in node for e in _space_elements(v)]
    return []


def _enum_text_ok(args: Any, cap: int) -> bool:
    """Enum values whose own whitespace cannot be mistaken for a site's run: none at
    either end, and no run longer than the cap inside."""
    for element in _json_elements(args):
        for value in _enum_values(element.get("json_schema")):
            if not isinstance(value, str) or value != value.strip(" \t\n\r"):
                return False
            if any(len(m) > cap for m in re.findall(r"[ \t\n\r]+", value)):
                return False
    return True


def _enum_values(node: Any) -> list[Any]:
    if isinstance(node, dict):
        own = node.get("enum") if isinstance(node.get("enum"), list) else []
        return list(own) + [v for x in node.values() for v in _enum_values(x)]
    if isinstance(node, list):
        return [v for x in node for v in _enum_values(x)]
    return []


def _whitespace(
    args: Any, params: list[tuple[str, str, Any, Any]], kept: list[str]
) -> tuple[int, int, tuple[bool, ...]] | None:
    """(cap, run after a written parameter, can_end) when the tracker holds the
    whitespace cap of this call body, or None: no cap, or it stays in the grammar."""
    caps = {e.get("max_whitespace_cnt") for e in _json_elements(args)}
    if caps in (set(), {None}):
        return None
    if len(caps) != 1 or not isinstance(next(iter(caps)), int):
        kept.append("a whitespace cap the tracker does not model")
        return None
    cap = next(iter(caps))
    if any(kind not in ("block", "json", "enum", "text") for _, kind, _, _ in params):
        kept.append("a whitespace cap with a parameter the tracker does not follow")
        return None
    if not _enum_text_ok(args, cap):
        kept.append("a whitespace cap with enum values holding whitespace runs")
        return None
    if args.get("type") == "json_schema":
        schema = args.get("json_schema") or {}
        required = set(schema.get("required") or [])
        optional = [name not in required for name, _, _, _ in params]
        after = cap  # one site between parameters of the whole object
    else:
        parts = args["elements"][:-1]
        optional = [isinstance(e, dict) and e.get("type") == "optional" for e in parts]
        after = 2 * cap  # a parameter element's trailing site meets the next one
    if len(optional) != len(params):
        kept.append("a whitespace cap the tracker does not model")
        return None
    can_end = tuple(all(optional[i:]) for i in range(len(optional) + 1))
    return cap, after, can_end


def _min_bound_from(table: dict[str, tuple[int, Param]]) -> tuple[int | None, ...]:
    bounds = [p.bound if p.kind == "block" else p.low for _, p in table.values()]
    mins: list[int | None] = []
    low: int | None = None
    for b in reversed(bounds):
        if b is not None:
            low = b if low is None else min(low, b)
        mins.append(low)
    return tuple(reversed(mins)) + (None,)


def _whole_object(
    params: list[tuple[str, str, Any, Any]], twin_args: dict, kept: list[str]
) -> tuple[dict[str, tuple[int, Param]], int]:
    """The parameter table of an argument object xgrammar writes whole, and how many
    bounds the twin's copy of it (``twin_args``, replaced in place) no longer has."""
    limits = [
        b
        for _, kind, bound, _ in params
        if kind == "json" and isinstance(bound, tuple)
        for b in bound[1]
    ]
    twin_json = _twin_element(twin_args, limits) if limits else None
    if limits and twin_json is None:
        kept.append("a nested bound xgrammar does not write as one repetition")
    if twin_json is not None:
        twin_args.clear()
        twin_args.update(twin_json)
    table: dict[str, tuple[int, Param]] = {}
    for i, (pname, kind, bound, _) in enumerate(params):
        if pname in table:
            raise _Unrecognised("two parameters with one name")
        if kind != "json":
            table[pname] = (i, Param(pname, kind, None))
            continue
        nodes, own = bound
        if twin_json is None or not own:
            nodes = tuple(Node(n.kind, n.props, n.items, None) for n in nodes)
            table[pname] = (i, Param(pname, kind, None, nodes=nodes))
            continue
        table[pname] = (i, Param(pname, kind, None, nodes=nodes, low=min(own)))
    return table, len(limits) if twin_json is not None else 0


def _call(tag: Any) -> tuple[str, Any]:
    """(function name, argument element) of one call tag body."""
    content = tag.get("content") if isinstance(tag, dict) else None
    if not isinstance(content, dict) or content.get("type") != "sequence":
        raise _Unrecognised("call body")
    parts = content.get("elements")
    if not isinstance(parts, list) or len(parts) != 3:
        raise _Unrecognised("call body parts")
    head, args, tail = parts
    if tail != {"type": "const_string", "value": "\n</function>\n"}:
        raise _Unrecognised("call body tail")
    value = head.get("value") if isinstance(head, dict) else None
    if head.get("type") != "const_string" or not isinstance(value, str):
        raise _Unrecognised("call body head")
    if not (value.startswith("\n<function=") and value.endswith(">\n")):
        raise _Unrecognised("call body head text")
    return value[len("\n<function=") : -2], args


def _bodies(content: Any) -> list[Any]:
    if isinstance(content, dict) and content.get("type") == "or":
        elements = content.get("elements")
        if not isinstance(elements, list):
            raise _Unrecognised("or")
        return [{"content": e} for e in elements]
    return [{"content": content}]


def _has_bound(node: Any) -> bool:
    if isinstance(node, dict):
        grammar = node.get("grammar")
        if isinstance(grammar, str) and "max_chars=" in grammar:
            return True
        return "maxLength" in node or any(_has_bound(v) for v in node.values())
    if isinstance(node, list):
        return any(_has_bound(v) for v in node)
    return False


def analyse(spec: str) -> tuple[Plan | None, str | None]:
    """The plan for a tag, or (None, reason); (None, None) when nothing is bounded."""
    try:
        doc = json.loads(spec)
    except ValueError:
        return None, None
    if not _has_bound(doc):
        return None, None
    try:
        return _analyse(doc), None
    except _Unrecognised as exc:
        return None, f"tag shape not modelled: {exc}"


def _analyse(doc: Any) -> Plan:
    if not isinstance(doc, dict) or doc.get("type") != "structural_tag":
        raise _Unrecognised("not a structural tag")
    fmt = doc.get("format")
    if not isinstance(fmt, dict):
        raise _Unrecognised("format")
    triggered = fmt.get("type") == "token_triggered_tags"
    if triggered:
        tags = fmt.get("tags")
        triggers = fmt.get("trigger_tokens")
        if (
            not isinstance(tags, list)
            or len(tags) != 1
            or not isinstance(triggers, list)
        ):
            raise _Unrecognised("triggered tags")
        tag = tags[0]
    elif fmt.get("type") == "tag":
        tag = fmt
    else:
        raise _Unrecognised("top-level format")
    begin, end = tag.get("begin"), tag.get("end")
    if not (
        isinstance(begin, dict)
        and begin.get("type") == "token"
        and isinstance(begin.get("token"), int)
        and isinstance(end, dict)
        and end.get("type") == "token"
        and isinstance(end.get("token"), int)
    ):
        raise _Unrecognised("call delimiters are not token ids")
    if triggered and fmt["trigger_tokens"] != [begin["token"]]:
        raise _Unrecognised("trigger tokens")
    calls = [_call(b) for b in _bodies(tag.get("content"))]
    # A raw value (a string with keywords xgrammar writes as its own rule) may hold a
    # later parameter's text, so a bound after one stays in the grammar; with several
    # calls a raw value anywhere can hide the next call's start, so every bound stays.
    # Plain strings are "text", which the tracker follows to the first "</parameter>".
    eligible = True
    if triggered:
        for _, args in calls:
            if any(kind == "raw" for _, kind, _, _ in _arguments(args, False, [])):
                eligible = False
    kept: list[str] = []
    ws_kept: list[str] = []  # where the whitespace cap stays in the twin
    if not eligible:
        kept.append("a raw string parameter in a tag with several calls")
    twin = copy.deepcopy(doc)
    tools: dict[str, Tool] = {}
    twin_calls = [_call(b) for b in _bodies(_twin_tag(twin, triggered).get("content"))]
    stripped = 0
    for (name, args), (_, twin_args) in zip(calls, twin_calls):
        if name in tools:
            raise _Unrecognised("two tools with one name")
        params = _arguments(args, eligible, kept)
        twin_params = _arguments(twin_args, eligible, [])
        ws = _whitespace(args, params, ws_kept)
        if ws is not None:
            _uncap(twin_args)
        if args.get("type") == "json_schema":
            table, n = _whole_object(params, twin_args, kept)
            stripped += n
            tools[name] = Tool(name, table, _min_bound_from(table), *(ws or (None,)))
            continue
        table: dict[str, tuple[int, Param]] = {}
        for i, ((pname, kind, bound, _), (_, _, _, twin_element)) in enumerate(
            zip(params, twin_params)
        ):
            if pname in table:
                raise _Unrecognised("two parameters with one name")
            if kind == "json" and isinstance(bound, tuple):
                nodes, limits = bound
                twin_json = _twin_element(twin_element, limits) if limits else None
                if limits and twin_json is None:
                    kept.append(
                        "a nested bound xgrammar does not write as one repetition"
                    )
                if twin_json is None:
                    # the bounds stay in the grammar; the tracker follows it unbounded
                    nodes = tuple(Node(n.kind, n.props, n.items, None) for n in nodes)
                    table[pname] = (i, Param(pname, kind, None, nodes=nodes))
                    continue
                twin_element.clear()
                twin_element.update(twin_json)
                stripped += len(limits)
                table[pname] = (
                    i,
                    Param(pname, kind, None, nodes=nodes, low=min(limits)),
                )
                continue
            if kind == "json":
                table[pname] = (i, Param(pname, kind, None, nodes=(Node("x"),)))
                continue
            if bound is not None:
                content = twin_element["elements"][1]["content"]
                content["grammar"] = (
                    _TWIN_HEAD + content["grammar"][len(_RULE_HEAD.format(bound)) :]
                )
                stripped += 1
            table[pname] = (i, Param(pname, kind, bound))
        tools[name] = Tool(name, table, _min_bound_from(table), *(ws or (None,)))
    if not stripped:
        raise _Unrecognised(
            "no bound the table models" + (f" ({kept[0]})" if kept else "")
        )
    if _has_bound(twin):
        kept.append("bounds of a kind the table does not model")
    lows = [
        m[0] for t in tools.values() for m in [t.min_bound_from] if m[0] is not None
    ]
    return Plan(
        twin=json.dumps(twin),
        tools=tools,
        triggered=triggered,
        opener=begin["token"],
        closer=end["token"],
        kept=tuple(dict.fromkeys(kept + ws_kept)),
        min_bound=min(lows) if lows else None,
    )


def _twin_tag(doc: Any, triggered: bool) -> Any:
    fmt = doc["format"]
    return fmt["tags"][0] if triggered else fmt


# ------------------------------------------------------------------ the table


def _closer_state(data: bytes) -> int:
    """Length of the longest suffix of ``data`` that is a proper prefix of CLOSER.

    CLOSER holds one newline, its first byte, so such a suffix starts at the last one.
    """
    j = data.rfind(b"\n")
    if j < 0:
        return 0
    tail = data[j:]
    return len(tail) if len(tail) < _CLOSER_LEN and CLOSER.startswith(tail) else 0


_CONTINUATION = bytes(range(0x80, 0xC0))


def _chars(data: bytes) -> int:
    """Characters a byte run adds to a value: its bytes that are not continuations."""
    return len(data.translate(None, _CONTINUATION))


class LengthTable:
    """Per-tokenizer token lengths and the cached "at most r more" masks."""

    def __init__(self, vocab: list[bytes], vocab_size: int):
        self.vocab = vocab
        size = max(vocab_size, len(vocab))
        self.words = (size + 31) // 32
        self.size = self.words * 32
        blen = np.zeros(self.size, dtype=np.int32)
        m0 = np.zeros(self.size, dtype=np.int32)
        mj = np.zeros(self.size, dtype=np.int32)
        n = len(vocab)
        blen[:n] = [len(data) for data in vocab]
        chars = [_chars(data) for data in vocab]
        m0[:n] = chars
        mj[:n] = chars
        closing = []
        newline = []
        quoted = []
        for i, data in enumerate(vocab):
            if b"\n" in data:
                newline.append(i)
                j = data.find(CLOSER)
                if j >= 0:
                    m0[i] = _chars(data[:j])
                    closing.append(i)
                else:
                    m0[i] = chars[i] - _closer_state(data)
            if b'"' in data:
                # a bounded JSON string has no escapes: it ends at the first quote
                mj[i] = _chars(data[: data.find(b'"')])
                quoted.append(i)
        self.blen = blen
        self.m0 = m0
        self.mj = mj
        self.max_m0 = int(m0.max()) if len(vocab) else 0
        self.max_mj = int(mj.max()) if len(vocab) else 0
        by_bytes: dict[bytes, list[int]] = {}
        for i, data in enumerate(vocab):
            if data and len(data) <= _CLOSER_LEN:
                by_bytes.setdefault(data, []).append(i)
        # Tokens that continue the closer from k: a proper prefix of what is left of it,
        # or the rest of it followed by more bytes. Both put -k into the value count.
        self.continue_ids: list[np.ndarray] = [np.zeros(0, dtype=np.int64)]
        self.closing_ids: list[np.ndarray] = [self._by_len(closing)]
        prefixed: dict[int, list[int]] = {k: [] for k in range(1, _CLOSER_LEN)}
        starts: dict[int, list[int]] = {}
        for k in range(1, _CLOSER_LEN):
            starts.setdefault(CLOSER[k], []).append(k)
        for i, data in enumerate(vocab):
            for k in starts.get(data[0], ()) if data else ():
                if data.startswith(CLOSER[k:]):
                    prefixed[k].append(i)
        for k in range(1, _CLOSER_LEN):
            rest = CLOSER[k:]
            cont = [i for n in range(1, len(rest)) for i in by_bytes.get(rest[:n], [])]
            cont += prefixed[k]
            self.continue_ids.append(np.array(sorted(set(cont)), dtype=np.int64))
            self.closing_ids.append(
                self._by_len(sorted(set(closing) | set(prefixed[k])))
            )
        self.newline_ids = self._by_len(newline)
        self.quote_ids = self._by_len(quoted)
        self.entry_ids = self._by_len(sorted(set(newline) | set(quoted)))
        self._by_bytes = by_bytes
        self._masks: OrderedDict[tuple, np.ndarray] = OrderedDict()
        self._lock = threading.Lock()
        self._ws: tuple | None = None
        self._ws_tables()  # with the table, not at a request's first fill

    def _by_len(self, ids: list[int]) -> np.ndarray:
        """``ids`` sorted by byte length, longest first."""
        arr = np.array(ids, dtype=np.int64)
        if not len(arr):
            return arr
        return arr[np.argsort(-self.blen[arr], kind="stable")]

    def longer_than(self, ids: np.ndarray, n: int) -> np.ndarray:
        """The prefix of a longest-first id array of tokens longer than n bytes."""
        if not len(ids):
            return ids
        lens = self.blen[ids]
        return ids[: int(np.searchsorted(-lens, -n, side="left"))]

    def _ws_tables(self) -> tuple:
        """(leading whitespace per token, its maximum, ids of tokens holding a run
        longer than 64 after other text, ids that may carry a run's extra newline into
        the call's end), built at first use."""
        with self._lock:
            if self._ws is not None:
                return self._ws
        lead = np.zeros(self.size, dtype=np.int32)
        long_runs: list[int] = []
        ends: list[int] = []
        for i, data in enumerate(self.vocab):
            rest = data.lstrip(_WS)
            lw = len(data) - len(rest)
            lead[i] = lw
            if len(rest) > 64 and any(
                len(m) > 64 for m in re.findall(rb"[ \t\n\r]+", rest)
            ):
                long_runs.append(i)
            if (
                lw
                and data[lw - 1] == 10
                and (b"\n" + _FUNCTION_END).startswith(data[lw - 1 :])
            ):
                ends.append(i)
        built = (
            lead,
            int(lead.max()) if len(self.vocab) else 0,
            np.array(long_runs, dtype=np.int64),
            np.array(ends, dtype=np.int64),
        )
        with self._lock:
            self._ws = built
        return built

    def ws_long_runs(self) -> np.ndarray:
        return self._ws_tables()[2]

    def ws_mask(self, k: int, may_end: bool) -> np.ndarray | None:
        """uint32 words of the tokens whose leading whitespace fits in ``k`` more
        characters; with ``may_end`` also those that end the run with the newline
        that opens the call's end one character past it. None when all fit."""
        lead, top, _, ends = self._ws_tables()
        if k >= top:
            return None
        key = ("ws", k, may_end)
        with self._lock:
            hit = self._masks.get(key)
            if hit is not None:
                self._masks.move_to_end(key)
                return hit
        allowed = lead <= k
        if may_end and len(ends):
            allowed[ends[lead[ends] == k + 1]] = True
        words = np.packbits(allowed, bitorder="little").view("<u4").copy()
        with self._lock:
            self._masks[key] = words
        return words

    def end_mask(self, j: int) -> np.ndarray:
        """uint32 words of the tokens that continue "</function>\n" from byte j."""
        key = ("end", j)
        with self._lock:
            hit = self._masks.get(key)
            if hit is not None:
                return hit
        allowed = np.zeros(self.size, dtype=bool)
        rest = _FUNCTION_END[j:]
        for n in range(1, len(rest) + 1):
            for i in self._by_bytes.get(rest[:n], ()):
                allowed[i] = True
        words = np.packbits(allowed, bitorder="little").view("<u4").copy()
        with self._lock:
            self._masks[key] = words
        return words

    def mask(self, k: int, r: int) -> np.ndarray | None:
        """uint32 words of the tokens that fit in ``r`` more characters: in a bounded
        block at closer state ``k`` (k >= 0), or in a bounded JSON string (k = -1).
        None when every token fits."""
        if r >= (self.max_mj if k < 0 else self.max_m0):
            return None
        key = (k, r)
        with self._lock:
            hit = self._masks.get(key)
            if hit is not None:
                self._masks.move_to_end(key)
                return hit
        allowed = (self.mj if k < 0 else self.m0) <= r
        if k > 0:
            allowed[self.continue_ids[k]] = True
        words = np.packbits(allowed, bitorder="little").view("<u4").copy()
        with self._lock:
            self._masks[key] = words
            while len(self._masks) > 512:
                self._masks.popitem(last=False)
        return words


_tables: dict[int, LengthTable] = {}
_tables_lock = threading.Lock()


def table_for(owner: Any, info: xgr.TokenizerInfo, vocab_size: int) -> LengthTable:
    """The table of ``info``, built once per ``owner`` (the backend's compiler)."""
    key = id(owner)
    with _tables_lock:
        table = _tables.get(key)
        if table is None:
            import time

            start = time.perf_counter()
            table = LengthTable([bytes(b) for b in info.decoded_vocab], vocab_size)
            _tables[key] = table
            logger.info(
                "Strict length table built in %.2f s (%d tokens, %.1f MiB).",
                time.perf_counter() - start,
                len(table.vocab),
                (table.blen.nbytes + table.m0.nbytes + table.mj.nbytes) / 2**20,
            )
        return table


# ------------------------------------------------------------------ the tracker

# State: (mode, tool, param, a, b, buf, ctx)
#   HEAD/ARGS: buf collects the header or tag text
#   BLOCK: a = characters in the value so far, b = closer state; param = name
#   VALUE (a JSON parameter): a = lexer state (_J_*), b = characters of a bounded
#   string, buf = the key being read, ctx = (frames, bound of the open string); a
#   frame is (kind, node, phase, key node): kind "t" the parameter value (phase 0
#   value, 1 after), "o" an object (phase 0 key, 1 colon, 2 value, 3 comma), "a" an
#   array (phase 0 value, 1 comma).
_J_OUT, _J_STR, _J_STR_ESC, _J_KEY, _J_KEY_ESC, _J_BOUNDED = range(6)
# What follows the newline that opens the end of a call ("\n</function>\n").
_FUNCTION_END = b"</function>\n"
_LOST_STATE = (LOST, None, None, 0, 0, b"", None)


class CallTracker:
    """The tag's structure, advanced one accepted token at a time."""

    def __init__(self, plan: Plan, table: LengthTable):
        self.plan = plan
        self.table = table
        self.start = (FREE, None, None, 0, 0, b"", None)

    def bound(self, state: tuple) -> int | None:
        """The bound of the value ``state`` is inside, if the table applies one."""
        if state[0] == BLOCK:
            return self.plan.tools[state[1]].params[state[2]][1].bound
        if state[0] == VALUE and state[3] == _J_BOUNDED:
            return state[6][1]
        return None

    def step(self, state: tuple, token: int) -> tuple[tuple, bool]:
        """(next state, the bound holds); a state nothing explains is LOST."""
        mode = state[0]
        if mode in (DONE, DEAD, LOST):
            return state, True
        if mode == FREE:
            if token == self.plan.opener:
                return (HEAD, None, None, 0, 0, b"", None), True
            return state, True
        if mode == END and token == self.plan.closer:
            after = FREE if self.plan.triggered else DONE
            return (after, None, None, 0, 0, b"", None), True
        data = self.table.vocab[token] if token < len(self.table.vocab) else b""
        return self.feed(state, data)

    def feed(self, state: tuple, data: bytes) -> tuple[tuple, bool]:
        mode, tool, param, a, b, buf, ctx = state
        plan = self.plan
        i = 0
        n = len(data)
        while i < n:
            if mode == BLOCK:
                limit = plan.tools[tool].params[param][1].bound
                # The value automaton: a = characters, b = closer state.
                while i < n:
                    c = data[i]
                    i += 1
                    if not 0x80 <= c <= 0xBF:
                        a += 1
                    if c == CLOSER[b]:
                        b += 1
                    else:
                        b = 1 if c == 10 else 0
                    if b == _CLOSER_LEN:
                        if limit is not None and a - _CLOSER_LEN > limit:
                            return (mode, tool, param, a, b, buf, ctx), False
                        last = plan.tools[tool].params[param][0]
                        ctx = (last, -1)
                        mode, param, a, buf = ARGS, None, 0, b""
                        b = plan.tools[tool].ws_cap or 0
                        break
                    if limit is not None and a > limit and (b == 0 or a - b > limit):
                        return (mode, tool, param, a, b, buf, ctx), False
                continue
            c = data[i]
            i += 1
            if mode == HEAD:
                buf += bytes((c,))
                if (
                    buf.startswith(b"\n<function=")
                    and buf.endswith(b">\n")
                    and len(buf) > 13
                ):
                    name = buf[len(b"\n<function=") : -2].decode("utf-8", "replace")
                    if name not in plan.tools:
                        return _LOST_STATE, True
                    mode, tool, buf = ARGS, name, b""
                    a, b, ctx = 0, plan.tools[name].ws_cap or 0, (-1, -1)
                elif len(buf) > 4096:
                    return _LOST_STATE, True
            elif mode == ARGS:
                if buf:
                    buf += bytes((c,))
                    over = ctx is not None and ctx[1] >= 0
                    if over and not _FUNCTION_END.startswith(buf):
                        # after a run's extra newline only the call's end may follow
                        return (mode, tool, param, a, b, buf, ctx), False
                    if c == 62:  # ">"
                        tag = buf.decode("utf-8", "replace")
                        buf = b""
                        if tag == "</parameter>":
                            a, b = 0, plan.tools[tool].after_value
                            continue
                        if tag == "</function>":
                            mode = END
                            continue
                        if not (tag.startswith("<parameter=") and tag.endswith(">")):
                            return _LOST_STATE, True
                        entry = plan.tools[tool].params.get(
                            tag[len("<parameter=") : -1]
                        )
                        if entry is None:
                            if "*" in plan.tools[tool].params:
                                return (DEAD, None, None, 0, 0, b"", None), True
                            return _LOST_STATE, True
                        p = entry[1]
                        param = p.name
                        if p.kind == "block":
                            mode = PRE_BLOCK
                        elif p.kind == "json":
                            mode, a, b = VALUE, _J_OUT, 0
                            ctx = ((("t", 0, 0, -1),), None)
                        elif p.kind == "enum":
                            mode, a = ENUM, 0
                        elif p.kind == "text":
                            mode, a = TEXT, 0
                        else:
                            return (DEAD, None, None, 0, 0, b"", None), True
                    elif len(buf) > 4096:
                        return _LOST_STATE, True
                elif c == 60:  # "<"
                    buf = b"<"
                elif c not in _WS:
                    return _LOST_STATE, True
                elif plan.tools[tool].ws_cap:
                    # a = the run so far, b = what the run may hold, ctx = (index of
                    # the last parameter, -1 or 0 once the run's extra newline is in)
                    last, over = ctx
                    a += 1
                    if over >= 0:
                        return (mode, tool, param, a, b, buf, ctx), False
                    if a > b:
                        t = plan.tools[tool]
                        if a == b + 1 and c == 10 and t.can_end[last + 1]:
                            ctx = (last, 0)  # the newline that opens the call's end
                        else:
                            return (mode, tool, param, a, b, buf, ctx), False
            elif mode == PRE_BLOCK:
                if c != 10:
                    return _LOST_STATE, True
                mode, a, b = BLOCK, 0, 0
            elif mode == VALUE:
                if a == _J_BOUNDED:
                    if c == 34:
                        a, b, ctx = _J_OUT, 0, (ctx[0], None)
                    elif not 0x80 <= c <= 0xBF:
                        b += 1
                        if b > ctx[1]:
                            return (mode, tool, param, a, b, buf, ctx), False
                    continue
                if a in (_J_STR, _J_KEY):
                    if c == 92:
                        a += 1
                    elif c == 34:
                        if a == _J_KEY:
                            frames = ctx[0]
                            kind, node, _, _ = frames[-1]
                            nodes = plan.tools[tool].params[param][1].nodes
                            props = (nodes[node].props if node >= 0 else None) or {}
                            key = props.get(buf.decode("utf-8", "replace"), -1)
                            ctx = (frames[:-1] + ((kind, node, 1, key),), None)
                            buf = b""
                        a = _J_OUT
                    elif a == _J_KEY:
                        buf += bytes((c,))
                    continue
                if a in (_J_STR_ESC, _J_KEY_ESC):
                    a -= 1
                    if a == _J_KEY:
                        buf += bytes((c,))
                    continue
                nodes = plan.tools[tool].params[param][1].nodes
                after = self._json_out(nodes, ctx[0], c)
                if after is None:
                    return _LOST_STATE, True
                if after == "end":
                    ctx = (plan.tools[tool].params[param][0], -1)
                    mode, param, a, b, buf = ARGS, None, 0, 0, b"<"
                    continue
                frames, a, bound = after
                cap = plan.tools[tool].ws_cap
                if c in _WS and a == _J_OUT and cap:
                    b += 1  # b = the whitespace run outside strings
                    if b > cap:
                        return (mode, tool, param, a, b, buf, (frames, bound)), False
                else:
                    b = 0
                ctx = (frames, bound)
            elif mode == TEXT:
                # a = how much of "</parameter>" the text ends with
                if c == _TEXT_END[a]:
                    a += 1
                    if a == len(_TEXT_END):
                        ctx = (plan.tools[tool].params[param][0], -1)
                        mode, param, a, buf = ARGS, None, 0, b""
                        b = plan.tools[tool].after_value
                else:
                    a = 1 if c == 60 else 0
            elif mode == ENUM:
                # a = the whitespace run around the value
                if c == 60:
                    ctx = (plan.tools[tool].params[param][0], -1)
                    mode, param, a, b, buf = ARGS, None, 0, 0, b"<"
                elif c in _WS:
                    a += 1
                    cap = plan.tools[tool].ws_cap
                    if cap and a > cap:
                        return (mode, tool, param, a, b, buf, ctx), False
                else:
                    a = 0
            elif mode == END:
                if c not in b" \t\n\r":
                    return _LOST_STATE, True
            else:  # DONE, DEAD, LOST, FREE: no structure to follow
                break
        return (mode, tool, param, a, b, buf, ctx), True

    @staticmethod
    def _json_out(nodes: tuple[Node, ...], frames: tuple, c: int) -> Any:
        """One byte outside JSON strings: (frames, lexer state, bound of an opened
        string), "end" for the "<" that closes the parameter, or None if unexplained."""
        kind, node, phase, key = frames[-1]
        if c in (32, 9, 10, 13):
            return frames, _J_OUT, None
        if kind == "o" and phase == 0 and c == 34:
            return frames, _J_KEY, None
        if kind == "o" and phase == 1:
            return (
                (frames[:-1] + ((kind, node, 2, key),), _J_OUT, None)
                if c == 58
                else None
            )
        if c == 44:  # ","
            if (kind == "o" and phase == 3) or (kind == "a" and phase == 1):
                return frames[:-1] + ((kind, node, 0, -1),), _J_OUT, None
            return None
        if c in (125, 93):  # "}" "]"
            if (kind == "o" and phase in (0, 3)) or (kind == "a" and phase in (0, 1)):
                return frames[:-1], _J_OUT, None
            return None
        if kind == "t" and phase == 1:
            return "end" if c == 60 else (frames, _J_OUT, None)
        if (kind == "o" and phase == 3) or (kind == "a" and phase == 1):
            return frames, _J_OUT, None  # the rest of a number or a literal
        # a value starts: its node, and the frame now waits for what follows it
        if kind == "t":
            value = node
        elif kind == "o":
            value = key
        else:
            value = nodes[node].items if node >= 0 else -1
        frames = frames[:-1] + ((kind, node, 1 if kind != "o" else 3, key),)
        target = nodes[value] if value >= 0 else None
        if c == 123:  # "{"
            inner = value if target is not None and target.kind == "o" else -1
            return frames + (("o", inner, 0, -1),), _J_OUT, None
        if c == 91:  # "["
            inner = value if target is not None and target.kind == "a" else -1
            return frames + (("a", inner, 0, -1),), _J_OUT, None
        if c == 34:
            bound = target.bound if target is not None and target.kind == "s" else None
            return frames, (_J_STR if bound is None else _J_BOUNDED), bound
        return frames, _J_OUT, None

    def ws_words(self, state: tuple) -> np.ndarray | None:
        """uint32 words of the tokens the whitespace caps allow next, or None."""
        mode, tool = state[0], state[1]
        t = self.plan.tools.get(tool) if tool is not None else None
        if t is None or not t.ws_cap:
            return None
        if mode == ARGS:
            last, over = state[6]
            if over >= 0:
                return self.table.end_mask(len(state[5]))
            if state[5]:
                return None  # inside a tag
            return self.table.ws_mask(state[4] - state[3], t.can_end[last + 1])
        if mode == VALUE and state[3] == _J_OUT:
            return self.table.ws_mask(t.ws_cap - state[4], False)
        if mode == ENUM:
            return self.table.ws_mask(t.ws_cap - state[3], False)
        return None

    def min_bound(self, state: tuple) -> int | None:
        """The smallest bound a token could still reach from ``state``."""
        mode, tool = state[0], state[1]
        if mode in (FREE, END, DONE, DEAD, LOST):
            return None
        if tool is None:
            return self.plan.min_bound
        t = self.plan.tools[tool]
        if state[2] is None:
            # between parameters: any parameter may follow (optionals may be skipped)
            return t.min_bound_from[0]
        index = t.params[state[2]][0]
        return t.min_bound_from[index + 1] if mode == BLOCK else t.min_bound_from[index]


# ------------------------------------------------------------------ the grammar


def _bit_view(bitmask: torch.Tensor, index: int) -> np.ndarray:
    return bitmask[index].numpy().view(np.uint32)


def compile_fast(backend: Any, spec: str, stop_token_ids: set[int] | None) -> Any:
    """A table-bounded grammar for ``spec``, or None to compile ``spec`` as given."""
    global fast_count
    if not enabled():
        return None
    plan, reason = analyse(spec)
    if plan is None:
        if reason is not None:
            _count_fallback(reason)
        return None
    import xgrammar as xgr

    compiler = backend.full_vocab_compiler
    ctx = compiler.compile_structural_tag(one_run_per_gap(plan.twin))
    table = table_for(compiler, ctx.tokenizer_info, backend.vocab_size)
    for reason in plan.kept:
        _count_fallback(reason)
    with _counts_lock:
        fast_count += 1
    override = list(stop_token_ids) if stop_token_ids else None
    return TableBoundedGrammar(
        matcher=xgr.GrammarMatcher(
            ctx,
            override_stop_tokens=override,
            max_rollback_tokens=backend.num_speculative_tokens,
        ),
        vocab_size=backend.vocab_size,
        ctx=ctx,
        tracker=CallTracker(plan, table),
        spec=spec,
        compiler=compiler,
        override_stop=override,
        max_rollback=backend.num_speculative_tokens,
    )


@dataclass
class TableBoundedGrammar(XgrammarGrammar):
    """The twin matcher plus the call tracker; masks and accepts equal the bounded
    grammar's."""

    tracker: CallTracker | None = field(default=None, repr=False, hash=False)
    spec: str = field(default="", repr=False, hash=False)
    compiler: Any = field(default=None, repr=False, hash=False)
    override_stop: list[int] | None = field(default=None, repr=False, hash=False)
    max_rollback: int = 0
    states: list = field(default_factory=list, repr=False, hash=False)
    history: list = field(default_factory=list, repr=False, hash=False)
    real: Any = field(default=None, repr=False, hash=False)
    twin_ctx: Any = field(default=None, repr=False, hash=False)

    def __post_init__(self):
        assert self.tracker is not None
        self.states = [self.tracker.start]
        self.twin_ctx = self.ctx

    def _to_real(self) -> None:
        """Continue on the bounded grammar; only when the tracker cannot follow."""
        import xgrammar as xgr

        logger.error(
            "Strict length table: the call structure could not be followed; this "
            "request continues on the bounded grammar."
        )
        _count_fallback("the tracker lost the structure")
        ctx = self.compiler.compile_structural_tag(one_run_per_gap(self.spec))
        real = xgr.GrammarMatcher(
            ctx,
            override_stop_tokens=self.override_stop,
            max_rollback_tokens=self.max_rollback,
        )
        for token in self.history:
            if not real.accept_token(token):
                raise RuntimeError("the bounded grammar refused an accepted token")
        self.real = real
        self.ctx = ctx

    def _matcher(self):
        return self.real if self.real is not None else self.matcher

    def _accept_one(self, token: int) -> bool:
        if self.real is not None:
            if not self.real.accept_token(token):
                return False
            self.history.append(token)
            return True
        state, ok = self.tracker.step(self.states[-1], token)
        if not ok or not self.matcher.accept_token(token):
            return False
        self.history.append(token)
        self.states.append(state)
        if state[0] == LOST:
            self._to_real()
        return True

    def _undo(self, num_tokens: int) -> None:
        self._matcher().rollback(num_tokens)
        del self.history[-num_tokens:]
        if self.real is None:
            del self.states[-num_tokens:]

    def accept_tokens(self, request_id: str, tokens: list[int]) -> bool:
        if self._is_terminated:
            return True
        for num_accepted, token in enumerate(tokens):
            if not self._accept_one(token):
                if num_accepted:
                    self.rollback(num_accepted)
                return False
            self.num_processed_tokens += 1
            self._is_terminated = self._matcher().is_terminated()
            if self._is_terminated:
                break
        return True

    def validate_tokens(self, tokens: list[int]) -> list[int]:
        if self._is_terminated:
            return []
        accepted: list[int] = []
        for token in tokens:
            if not self._accept_one(token):
                break
            accepted.append(token)
            if self._matcher().is_terminated():
                break
        if accepted:
            self._undo(len(accepted))
        return accepted

    def rollback(self, num_tokens: int) -> None:
        self._undo(num_tokens)
        self.num_processed_tokens -= num_tokens
        self._is_terminated = self._matcher().is_terminated()

    def reset(self):
        self.matcher.reset()
        self.real = None
        self.ctx = self.twin_ctx
        self.states = [self.tracker.start]
        self.history = []
        self.num_processed_tokens = 0
        self._is_terminated = False

    def fill_bitmask(self, bitmask: torch.Tensor, idx: int) -> None:
        if self.real is not None:
            self.real.fill_next_token_bitmask(bitmask, idx)
            return
        self.matcher.fill_next_token_bitmask(bitmask, idx)
        state = self.states[-1]
        if state[0] in (FREE, END, DONE, DEAD):
            return
        tracker = self.tracker
        table = tracker.table
        row = None
        words = tracker.ws_words(state)
        if words is not None:
            row = _bit_view(bitmask, idx)
            row &= words
        tool = tracker.plan.tools.get(state[1]) if state[1] is not None else None
        ws_runs = table.ws_long_runs() if tool is not None and tool.ws_cap else None
        limit = tracker.bound(state)
        if limit is not None:
            if state[0] == BLOCK:
                words = table.mask(state[4], limit - state[3])
                candidates = table.closing_ids[state[4]]
            else:
                words = table.mask(-1, limit - state[4])
                candidates = table.quote_ids
            if words is not None:
                if row is None:
                    row = _bit_view(bitmask, idx)
                row &= words
        else:
            candidates = table.entry_ids
        low = tracker.min_bound(state)
        candidates = (
            table.longer_than(candidates, low) if low is not None else candidates[:0]
        )
        if ws_runs is not None and len(ws_runs):
            # tokens holding a whitespace run past the cap after other text
            candidates = np.union1d(candidates, ws_runs)
        if not len(candidates):
            return
        if row is None:
            row = _bit_view(bitmask, idx)
        shifts = (candidates & 31).astype(np.uint32)
        live = candidates[((row[candidates >> 5] >> shifts) & 1) == 1]
        for token in live.tolist():
            if not tracker.step(state, token)[1]:
                row[token >> 5] &= np.uint32(~(1 << (token & 31)) & 0xFFFFFFFF)
