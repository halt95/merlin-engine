# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""One whitespace run per gap in a split tool-call argument grammar.

A strict tool with a bounded string parameter gets its arguments as a sequence of
parts, one per parameter (see ``_qwen_3_coder_arguments``): a bounded string block
(a whitespace run, then the tag) or a one-property JSON-schema element, which
xgrammar writes with a whitespace run on each side, then a final whitespace run. Two
neighbouring parts put two whitespace runs next to each other, so a run of n
whitespace characters between parameters has n + 1 parses. The matcher keeps every
one: its memory grows with the square of the run (about 20 bytes x n^2 on xgrammar
0.2.8, 4.5 GB at 15,360 characters) and its fills slow down with it.

``one_run_per_gap`` compiles the same language with one run in every gap. Each part
loses its own edge runs (a JSON-schema element becomes xgrammar's own grammar for it,
read back from its printout, with the two edge runs taken off; a block keeps only
its tag), and the sequence becomes::

    run(L), [part_1, run(t_1 + L)], [part_2, run(t_2 + L)], ..., [part_n, run(t_n + L)]

with each bracket optional where the part was. L is the bound of the leading run
every part and the end have (unbounded, or the strict whitespace cap) and t_i the
bound of part i's trailing run (0 for a block). A gap between two written parameters
held up to t_i + L characters before (part i's trailing run and the next part's
leading run, or the final run) and holds exactly that now, in one run, so the
accepted text is the same, character for character. When the parts' leading runs
differ, or a part is not one of these shapes, the sequence is left as it is.

The value inside a parameter tag has the same problem one level down. xgrammar writes
it with a run on each side, ``W{0,L} V W{0,R}``, and when V can be empty (a string
pattern such as ``^[a-z/._]*$``) a whitespace-only text can be split between the two
runs in up to L + R + 1 ways: the matcher keeps them all and a fill takes 300 ms at
about 90 characters. For a whitespace-free nullable pattern ``nullable_value`` keeps a
single path through each leading whitespace prefix. After n whitespace characters the
value may close (an empty value) when n <= L + R, or start a non-empty value V' (V
without the empty text) when n <= L, and may take one more whitespace character when
n < L + R; a non-empty value then has its own trailing run of at most R characters.
Only one alternative consumes whitespace and closing is inside each state, so there is
no second run to split with. An unbounded left run is one loop; a finite left bound
with an unbounded right one ends in an overflow loop that can only close. The accepted
text is the same: an empty V gave exactly the whitespace texts of at most L + R
characters, and a non-empty V that neither starts nor ends with whitespace has one
split. V' is written out for a string pattern that is an anchored sequence of simple
atoms (a plain character, an escape, a character class without negation) with a
quantifier each, none of which can match whitespace; any other value is left as it is.
Patterns whose atoms overlap keep their own ambiguity inside V'.

A class pattern ``^[C]*$`` whose class holds whitespace but not ``<`` has a value that
can itself start and end with whitespace, so the split between padding and value is
also ambiguous. ``_class_space_star`` writes ``W{0,L} C* W{0,R}`` and the closing tag
as L + R + 1 states with disjoint first characters (the longest leading padding, then
the longest value, then the trailing padding after the first whitespace character the
class lacks). A class with ``<``, and ``^[C]+$``, are left as xgrammar writes them (the
plus form needs thousands of states to stay exact).

A string-or-null union (exactly ``anyOf`` of string and null) is written as the string:
xgrammar pads a union but not a plain string, whose raw text takes any whitespace
itself, so the same text is accepted (``_raw_null``).

A tool whose arguments are not split (no bounded string) is one JSON-schema element;
``_whole_element`` reads xgrammar's whole grammar for it from its printout and swaps
only the padded value fragments, leaving the one capped gap between parameters as it
is (splitting it would put two runs side by side).
"""

from __future__ import annotations

import copy
import json
from functools import lru_cache
from typing import Any

import regex as re

_RUN = "(?:[ ]|\\t|\\n|\\r)"
# one edge run in xgrammar's printout: unbounded, or at most N
_EDGE = r"(?:\[ \\n\\r\\t\]\*|whitespace_\d+\{0, (\d+)\})"
_INF = -1


def _space(bound: int) -> dict:
    tail = "*" if bound == _INF else "{0," + str(bound) + "}"
    return {"type": "regex", "pattern": _RUN + tail}


def space_bound(element: Any) -> int | None:
    """The bound of a whitespace-run element (``_INF`` unbounded), or None."""
    if not isinstance(element, dict) or set(element) != {"type", "pattern"}:
        return None
    if element.get("type") != "regex" or not isinstance(element.get("pattern"), str):
        return None
    pattern = element["pattern"]
    if pattern == _RUN + "*":
        return _INF
    m = re.fullmatch(re.escape(_RUN) + r"\{0,(\d+)\}", pattern)
    return int(m.group(1)) if m else None


def _rules(text: str) -> list[tuple[str, str]]:
    """(name, body) of each rule of an xgrammar printout, in order."""
    starts = list(re.finditer(r"(?m)^([A-Za-z_][A-Za-z0-9_]*) ::= ", text))
    out = []
    for i, m in enumerate(starts):
        end = starts[i + 1].start() if i + 1 < len(starts) else len(text)
        out.append((m.group(1), text[m.end() : end].rstrip("\n")))
    return out


def _bound(m: Any, group: int) -> int:
    return _INF if m.group(group) is None else int(m.group(group))


_WHITESPACE = " \t\r\n"
_PLAIN = r"A-Za-z0-9/_:,;@#%&=~"
_ESCAPE = r"\\[dw./_\-]"
# one atom of a pattern: an escape, a character class (no negation, no nesting) or a
# plain character, then its quantifier
_PATTERN_ATOM = re.compile(
    r"(?P<atom>"
    + _ESCAPE
    + r"|\[(?:["
    + _PLAIN
    + r".\-]|"
    + _ESCAPE
    + r")+\]|["
    + _PLAIN
    + r"])"
    r"(?P<quant>[*+?]|\{(?P<lo>\d{1,4})(?P<comma>,(?P<hi>\d{0,4}))?\})?"
)
_ANNOTATIONS = frozenset(
    {"type", "pattern", "description", "title", "default", "examples"}
)


def _class_items(text: str) -> str:
    """The inside of an EBNF character class for the inside of a pattern's class."""
    out = []
    i = 0
    while i < len(text):
        if text[i] == "\\":
            e = text[i + 1]
            out.append({"d": "0-9", "w": "a-zA-Z0-9_", "-": "\\-"}.get(e, e))
            i += 2
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


def _atom_ebnf(atom: str) -> str | None:
    """EBNF for one pattern atom, or None when it can match whitespace."""
    text = "[" + _class_items(atom[1:-1] if atom.startswith("[") else atom) + "]"
    check = re.compile(atom if atom.startswith(("[", "\\")) else re.escape(atom))
    if any(check.fullmatch(w) for w in _WHITESPACE):
        return None
    return text


def _repeat(text: str, lo: int, hi: int) -> str:
    """``text`` repeated lo to hi times (hi ``_INF``: unbounded), written for EBNF."""
    if hi == _INF:
        return text + ("*" if lo == 0 else "+" if lo == 1 else "{" + str(lo) + ",}")
    if (lo, hi) == (1, 1):
        return text
    if (lo, hi) == (0, 1):
        return text + "?"
    if lo == hi:
        return text + "{" + str(lo) + "}"
    return text + "{" + f"{lo},{hi}" + "}"


def nonempty_pattern(pattern: Any) -> str | None:
    """EBNF for the non-empty texts of an anchored pattern that can match the empty
    text, or None: not nullable, a shape this does not model, or an atom that can
    match whitespace."""
    if not isinstance(pattern, str) or len(pattern) < 3:
        return None
    if (
        not pattern.startswith("^")
        or not pattern.endswith("$")
        or pattern.endswith("\\$")
    ):
        return None
    body = pattern[1:-1]
    atoms: list[tuple[str, int, int]] = []
    pos = 0
    while pos < len(body):
        m = _PATTERN_ATOM.match(body, pos)
        if m is None:
            return None
        text = _atom_ebnf(m.group("atom"))
        if text is None:
            return None
        quant = m.group("quant")
        if quant is None:
            lo = hi = 1
        elif quant == "*":
            lo, hi = 0, _INF
        elif quant == "+":
            lo, hi = 1, _INF
        elif quant == "?":
            lo, hi = 0, 1
        else:
            lo = int(m.group("lo"))
            if m.group("comma") is None:
                hi = lo
            else:
                hi = _INF if m.group("hi") == "" else int(m.group("hi"))
            if hi != _INF and hi < lo:
                return None
        atoms.append((text, lo, hi))
        pos = m.end()
    if not atoms or any(lo != 0 for _, lo, _ in atoms):
        return None
    # the non-empty texts, split by the first atom that matches something
    options = []
    for i, (text, _, hi) in enumerate(atoms):
        if hi == 0:
            continue
        parts = [_repeat(text, 1, hi)]
        parts += [_repeat(t, 0, h) for t, _, h in atoms[i + 1 :] if h != 0]
        options.append(" ".join(parts))
    return " | ".join(options) if options else None


def _run(bound: int) -> str:
    if bound == 0:
        return '""'
    return "[ \\n\\r\\t]" + ("*" if bound == _INF else "{0," + str(bound) + "}")


def _padding_rules(left: int, right: int) -> list[str]:
    """One path per whitespace prefix; these rules include the closing delimiter."""
    one = "[ \\n\\r\\t]"
    close = '"</parameter>"'
    rules = [f"trail ::= {_run(right)} {close}"]
    if left == _INF:
        rules.append(f"value ::= {close} | nonempty trail | {one} value")
        return rules
    last = left if right == _INF else left + right
    for n in range(last + 1):
        name = "value" if n == 0 else f"pad_{n}"
        options = [close]
        if n <= left:
            options.append("nonempty trail")
        if n < last:
            options.append(f"{one} pad_{n + 1}")
        elif right == _INF:
            options.append(f"{one} pad_over")
        rules.append(f"{name} ::= " + " | ".join(options))
    if right == _INF:
        rules.append(f"pad_over ::= {close} | {one} pad_over")
    return rules


_CLASS_STAR = re.compile(
    r"\^(?P<atom>\[(?:[ A-Za-z0-9/_:,;@#%&=~.\-]|\\[dwtnr./_\-])+\])\*\$"
)


def _class_space_star(pattern: Any, left: int, right: int) -> list[str] | None:
    """Rules for ``W{0,left} C* W{0,right}`` and the closing tag, with one path per
    prefix, when the class C holds whitespace and not ``<`` (the closing tag then
    starts with a character no value or padding character has). None otherwise.

    With A = C minus whitespace and D = whitespace not in C, the machine takes as much
    leading whitespace as the left run allows, then value characters of C; the first D
    character starts the trailing run, from which only whitespace and the closing tag
    follow (K is the closing tag, W whitespace)::

        P_i ::= K | W P_(i+1) | A Q      (i < left;  P_left is Q)
        Q   ::= K | C Q | D T_(right-1)
        T_j ::= K | W T_(j-1)            (T_0 ::= K)

    The text accepted is that of ``W{0,left} C* W{0,right}``: any split of it into
    padding and value extends to this one by taking the longest leading padding and
    then the longest value. That moves only a prefix of the value into the leading run,
    or a prefix of the trailing run into the value."""
    if not isinstance(pattern, str):
        return None
    match = _CLASS_STAR.fullmatch(pattern)
    if match is None:
        return None
    try:
        check = re.compile(match.group("atom"), re.ASCII)
    except re.error:
        return None
    chars = {chr(i) for i in range(128) if check.fullmatch(chr(i))}
    ws = set(_WHITESPACE)
    if not chars & ws or "<" in chars:
        return None
    ordinary, outside = chars - ws, ws - chars

    def cc(items: set[str]) -> str:
        return "[" + "".join(f"\\u{ord(c):04x}" for c in sorted(items)) + "]"

    one, close = "[ \\n\\r\\t]", '"</parameter>"'
    core = "value" if left == 0 else "pv_core"
    lines = []
    if left == _INF:
        options = [close, f"{one} value"]
        if ordinary:
            options.append(f"{cc(ordinary)} {core}")
        lines.append("value ::= " + " | ".join(options))
    else:
        for i in range(left):
            name = "value" if i == 0 else f"pv_p_{i}"
            nxt = core if i + 1 == left else f"pv_p_{i + 1}"
            options = [close, f"{one} {nxt}"]
            if ordinary:
                options.append(f"{cc(ordinary)} {core}")
            lines.append(f"{name} ::= " + " | ".join(options))
    options = [close, f"{cc(chars)} {core}"]
    if outside and right != 0:
        tail = "pv_tail" if right == _INF else f"pv_t_{right - 1}"
        options.append(f"{cc(outside)} {tail}")
    lines.append(f"{core} ::= " + " | ".join(options))
    if outside and right == _INF:
        lines.append(f"pv_tail ::= {close} | {one} pv_tail")
    elif outside and right != 0:
        for i in range(right):
            body = close if i == 0 else f"{close} | {one} pv_t_{i - 1}"
            lines.append(f"pv_t_{i} ::= {body}")
    return lines


def _raw_null(element: dict) -> dict:
    """``element`` with a property that is exactly a string or null (``anyOf`` of
    ``{"type": "string"}`` and ``{"type": "null"}``, either order, annotations only
    besides) written as the string. xgrammar pads a union with whitespace but not a
    plain string, whose raw text takes any whitespace itself. U, the raw strings that
    do not contain the closing tag, holds "null" and is closed under adding or removing
    whitespace at an end (the closing tag has none), so ``W U W`` is ``U`` and
    ``W (U or "null") W`` is ``U``. Nothing else is touched."""
    if element.get("style") != "qwen_xml" or element.get("excludes"):
        return element
    schema = element.get("json_schema")
    props = schema.get("properties") if isinstance(schema, dict) else None
    if not isinstance(props, dict):
        return element
    notes = _ANNOTATIONS - {"type", "pattern"}
    changed = {}
    for name, prop in props.items():
        if not isinstance(prop, dict) or not set(prop) <= notes | {"anyOf"}:
            continue
        branches = prop.get("anyOf")
        if not isinstance(branches, list) or len(branches) != 2:
            continue
        kinds = [
            b.get("type") if isinstance(b, dict) and set(b) == {"type"} else None
            for b in branches
        ]
        if sorted(k for k in kinds if k) != ["null", "string"]:
            continue
        changed[name] = {
            **{k: v for k, v in prop.items() if k in notes},
            "type": "string",
        }
    if not changed:
        return element
    return {**element, "json_schema": {**schema, "properties": {**props, **changed}}}


_NAME = r"[A-Za-z0-9_.\-]+"
_FRAGMENT = re.compile(
    r'"<parameter=('
    + _NAME
    + r')" ">" '
    + _EDGE
    + r" [A-Za-z_][A-Za-z0-9_]* "
    + _EDGE
    + r' "</parameter>"'
)
_EBNF_TOKEN = re.compile(r'"[^"]*"|\[[^\]]*\]|[A-Za-z_][A-Za-z0-9_]*')


def _namespaced(grammar: str, prefix: str) -> str:
    """``grammar`` with every rule name prefixed (quoted text and classes stay)."""
    return _EBNF_TOKEN.sub(
        lambda m: m.group(0) if m.group(0)[0] in '"[' else prefix + m.group(0), grammar
    )


def _candidate(element: dict) -> bool:
    """Whether a property of the element might be a padded value this rewrites."""
    schema = element.get("json_schema")
    props = schema.get("properties") if isinstance(schema, dict) else None
    if not isinstance(props, dict):
        return False
    return any(
        isinstance(p, dict) and p.get("type") == "string" and "pattern" in p
        for p in props.values()
    )


def _whole_element(element: dict) -> dict | None:
    """A qwen_xml JSON-schema element (the whole argument object of a tool that is not
    split) with the padding of each nullable string-pattern parameter written as in
    ``nullable_value``, and a string-or-null union as the string (``_raw_null``); the
    rest of xgrammar's own grammar for it, read back from its printout, untouched. None
    when nothing applies."""
    if element.get("type") != "json_schema" or element.get("style") != "qwen_xml":
        return None
    if not isinstance(element.get("json_schema"), dict):
        return None
    new = _raw_null(element)
    kept = new if new is not element else None
    if not _candidate(new):
        return kept
    import xgrammar as xgr

    try:
        text = str(
            xgr.Grammar.from_structural_tag(
                json.dumps({"type": "structural_tag", "format": new})
            )
        )
    except (RuntimeError, ValueError):
        return kept
    extra: list[str] = []

    def swap(m: Any) -> str:
        grammar = nullable_value(new, m.group(1), m.group(0))
        if grammar is None:
            return m.group(0)
        prefix = f"nv{len(extra)}_"
        extra.append(_namespaced(grammar, prefix))
        return prefix + "root"

    lines = [f"{name} ::= {_FRAGMENT.sub(swap, body)}" for name, body in _rules(text)]
    if not extra:
        return kept
    grammar = "\n".join(lines + extra)
    try:
        xgr.Grammar.from_ebnf(grammar)
    except (RuntimeError, ValueError):
        return kept
    return {"type": "grammar", "grammar": grammar}


def nullable_value(element: dict, name: str, middle: str) -> str | None:
    """The grammar of one parameter element with one path through the padding around an
    empty value, or None when its value is not a nullable string pattern this writes.
    ``middle`` is the element's printout between its own edge runs."""
    schema = element.get("json_schema")
    props = schema.get("properties") if isinstance(schema, dict) else None
    prop = props.get(name) if isinstance(props, dict) else None
    if not isinstance(prop, dict) or prop.get("type") != "string":
        return None
    if not set(prop) <= _ANNOTATIONS or element.get("excludes"):
        return None
    value = nonempty_pattern(prop.get("pattern"))
    m = re.fullmatch(
        r'"<parameter='
        + re.escape(name)
        + r'" ">" '
        + _EDGE
        + r" [A-Za-z_][A-Za-z0-9_]* "
        + _EDGE
        + r' "</parameter>"',
        middle,
    )
    if m is None:
        return None
    left, right = _bound(m, 1), _bound(m, 2)
    if value is None:
        rules = _class_space_star(prop.get("pattern"), left, right)
        if rules is None:
            return None
        return "\n".join([f'root ::= "<parameter={name}>" value', *rules])
    return "\n".join(
        [
            f'root ::= "<parameter={name}>" value',
            *_padding_rules(left, right),
            f"nonempty ::= {value}",
        ]
    )


def _strip_element(element: dict, name: str) -> tuple[dict, int, int] | None:
    """(grammar part without its edge runs, leading bound, trailing bound) of a
    one-property JSON-schema element, or None when its printout is not that shape."""
    import xgrammar as xgr

    element = _raw_null(element)
    try:
        text = str(
            xgr.Grammar.from_structural_tag(
                json.dumps({"type": "structural_tag", "format": element})
            )
        )
    except (RuntimeError, ValueError):
        return None
    rules = _rules(text)
    bodies = dict(rules)
    m = re.fullmatch(r"\(\(([A-Za-z_][A-Za-z0-9_]*)\)\)", bodies.get("root", ""))
    if m is None or m.group(1) not in bodies:
        return None
    top = m.group(1)
    m = re.fullmatch(
        r"\(\(" + _EDGE + " "
        r'("<parameter=' + re.escape(name) + r'" ">"(?: [^\s|()]+)* "</parameter>")'
        " " + _EDGE + r"\)\)",
        bodies[top],
    )
    if m is None:
        return None
    lead, middle, trail = _bound(m, 1), m.group(2), _bound(m, 3)
    bodies[top] = "((" + middle + "))"
    keep, todo = set(), ["root"]
    while todo:
        rule = todo.pop()
        if rule in keep or rule not in bodies:
            continue
        keep.add(rule)
        todo += re.findall(r"[A-Za-z_][A-Za-z0-9_]*", bodies[rule])
    grammar = "\n".join(f"{r} ::= {bodies[r]}" for r, _ in rules if r in keep)
    for candidate in (nullable_value(element, name, middle), grammar):
        if candidate is None:
            continue
        try:
            xgr.Grammar.from_ebnf(candidate)
        except (RuntimeError, ValueError):
            continue
        return {"type": "grammar", "grammar": candidate}, lead, trail
    return None


def _one_property(element: Any) -> str | None:
    """The parameter name of a one-property JSON-schema argument element, else None."""
    if not isinstance(element, dict) or element.get("type") != "json_schema":
        return None
    schema = element.get("json_schema")
    if element.get("style") != "qwen_xml" or element.get("any_order") is not False:
        return None
    if not isinstance(schema, dict) or schema.get("type") != "object":
        return None
    props = schema.get("properties")
    if schema.get("additionalProperties") is not False or not isinstance(props, dict):
        return None
    if len(props) != 1 or schema.get("required") != list(props):
        return None
    (name,) = props
    if not re.fullmatch(r"[A-Za-z0-9_.\-]+", name):
        return None
    return name


def _block(element: Any) -> tuple[dict, int] | None:
    """(tag, leading bound) of a bounded string block: a run, then the tag."""
    if not isinstance(element, dict) or element.get("type") != "sequence":
        return None
    parts = element.get("elements")
    if not isinstance(parts, list) or len(parts) != 2:
        return None
    lead = space_bound(parts[0])
    tag = parts[1]
    if lead is None or not isinstance(tag, dict) or tag.get("type") != "tag":
        return None
    begin = tag.get("begin")
    if not isinstance(begin, str) or not begin.startswith("<parameter="):
        return None
    return tag, lead


def _plus(a: int, b: int) -> int:
    return _INF if _INF in (a, b) else a + b


def _arguments(seq: dict) -> dict | None:
    """The sequence with one run per gap, or None when it is not a split argument
    sequence this rewrites."""
    elements = seq.get("elements")
    if not isinstance(elements, list) or len(elements) < 2:
        return None
    end = space_bound(elements[-1])
    if end is None:
        return None
    parts = []
    for element in elements[:-1]:
        optional = isinstance(element, dict) and element.get("type") == "optional"
        inner = element.get("content") if optional else element
        block = _block(inner)
        if block is not None:
            tag, lead = block
            parts.append((optional, copy.deepcopy(tag), lead, 0))
            continue
        name = _one_property(inner)
        if name is None:
            return None
        stripped = _strip_element(inner, name)
        if stripped is None:
            return None
        part, lead, trail = stripped
        parts.append((optional, part, lead, trail))
    if any(lead != end for _, _, lead, _ in parts):
        return None
    out: list[dict] = [_space(end)]
    for optional, part, _, trail in parts:
        pair = {"type": "sequence", "elements": [part, _space(_plus(trail, end))]}
        out.append({"type": "optional", "content": pair} if optional else pair)
    return {"type": "sequence", "elements": out}


def _walk(node: Any) -> int:
    """Rewrite every split argument sequence under ``node`` in place; how many."""
    count = 0
    if isinstance(node, dict):
        if node.get("type") == "sequence":
            new = _arguments(node)
            if new is not None:
                node.clear()
                node.update(new)
                return 1
        if node.get("type") == "json_schema":
            new = _whole_element(node)
            if new is not None:
                node.clear()
                node.update(new)
                return 1
        for value in node.values():
            count += _walk(value)
    elif isinstance(node, list):
        for value in node:
            count += _walk(value)
    return count


@lru_cache(maxsize=256)
def one_run_per_gap(spec: str) -> str:
    """``spec`` with every split argument sequence written with one whitespace run per
    gap; the same language. Returned unchanged when nothing applies."""
    try:
        doc = json.loads(spec)
    except ValueError:
        return spec
    if not isinstance(doc, dict) or doc.get("type") != "structural_tag":
        return spec
    if not _walk(doc):
        return spec
    return json.dumps(doc)
