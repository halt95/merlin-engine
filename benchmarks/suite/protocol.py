# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""The frozen protocol of the Merlin bench cards: prompts, depths, output caps,
thinking settings, validity rules and the profile rows.

Nothing here talks to a server. The prompt text, the request sizes and the
checks are the ones the published cards were measured with, so a run of this
suite sends the same prompts (up to the per-send nonce) and judges a send the
same way. ``selftest()`` checks the prompt builder and the image against the
frozen hashes; edit nothing in this file without moving them on purpose.
"""

from __future__ import annotations

import base64
import functools
import hashlib
import re
from pathlib import Path

SUITE_VERSION = "1"

# -- prompt material (the card prompt builder, byte for byte) -----------------

FILLER_UNIT = (
    "The archivist logged another routine morning in the stacks; "
    "the catalog remained stable and no volume required attention. "
)
QUESTION = (
    "What is the session access code stated in the late text, "
    "and what is the order number printed in the image? "
    "Answer on two lines prefixed 'ANSWER1: '/'ANSWER2: '."
)
NEEDLE_TEMPLATE = " NOTE: session access code {nonce}. "
# Tokens one FILLER_UNIT takes on the reference tokenizer (the builder's own constant).
TOKENS_PER_UNIT = 20.05

IMAGE_PATH = Path(__file__).with_name("ocr_text.png")
IMAGE_SHA256 = "f61f9c7a4312cb23dd50b013df0b514a489d2010ec9869daf3ca567f82551e57"
# sha256 of build_prompt(REQ[4096], "G6-1-4096-0"). The nonce prefix is the cards' own: it keeps the
# prompt text byte-identical to theirs.
PROMPT_SELFTEST_NONCE = "G6-1-4096-0"
PROMPT_SELFTEST_SHA256 = "0ff28dcaa054c43c99a52926197f82f5b11f81926f9bffe3973e26c84737e236"


def build_prompt(depth_tokens: int, nonce: str) -> str:
    """A needle prompt of about ``depth_tokens`` reference-tokenizer tokens: a
    filler haystack with one needle (the nonce) 55 % in, then the question.
    The cards were measured with this builder uncalibrated; see ``REQ``."""
    reserve = 3500 if depth_tokens > 250000 else 400
    budget = max(64, depth_tokens - reserve)
    n_units = max(1, int(budget / TOKENS_PER_UNIT))
    prefix = FILLER_UNIT * (n_units * 55 // 100)
    suffix = FILLER_UNIT * (n_units - n_units * 55 // 100)
    needle = f" NOTE: session access code {nonce}. "
    return f"SESSION {nonce}. " + prefix + needle + suffix + "\n\n" + QUESTION


# -- the card instrument ------------------------------------------------------

# The builder overshoots its nominal depth by 2.6-4.7 %, so every row requests a
# fixed builder depth that lands at about 0.995 x the nominal depth.
REQ = {4096: 3974, 10240: 9811, 32768: 31211, 102400: 97359, 131072: 124597, 261120: 251033}
LADDER_DEPTHS = (4096, 32768, 131072, 261120)
PREFILL_DEPTHS = (10240, 102400)
SENDS = 3  # sends per ladder cell and reps of the exact row: medians of 3
MAX_TOKENS = 256  # thinking on
PREFILL_MAX_TOKENS = 8
OFF_MAX_TOKENS = 1024  # thinking off: the answer-plus-summary task, natural stop
OFF_MIN_TOKENS = 200
OFF_MIN_WORDS = 150
WARM_DEPTH = 8192
WARM_MAX_TOKENS = 16
EXACT_TARGET = 261888  # prompt tokens of the exact 262K row
EXACT_OUT_TOKENS = 128
EXACT_REPS = 3
DEFAULT_EFFORT = "medium"


def thinking_kwargs(effort: str) -> dict:
    return {"enable_thinking": True, "reasoning_effort": effort}


NO_THINKING = {"enable_thinking": False}

# The thinking-off rows: the answer lines plus about 300 words of prose.
TASK_OFF = (
    "What is the session access code stated in the late text, and what is the order "
    "number printed in the image? Answer on two lines prefixed 'ANSWER1: '/'ANSWER2: '. "
    "Then, on the following lines, write a summary of the passage above in plain prose "
    "of about 300 words: describe the archive, the archivist's routine, what the "
    "catalog records, and what would change if a volume did require attention. No "
    "lists, no headings."
)


def build_off(depth: int, nonce: str) -> str:
    p = build_prompt(REQ[depth], nonce)
    if not p.endswith(QUESTION):
        raise RuntimeError("prompt builder output changed; refusing an unfrozen thinking-off prompt")
    return p[: -len(QUESTION)] + TASK_OFF


def content_valid(text: str) -> tuple[bool, str]:
    """The thinking-off content gate: two answer lines, then prose."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) < 3:
        return False, "fewer than three non-empty lines"
    if not re.match(r"^ANSWER1: \S", lines[0]) or not re.match(r"^ANSWER2: \S", lines[1]):
        return False, "first two lines are not 'ANSWER1: ...' then 'ANSWER2: ...'"
    words = sum(len(ln.split()) for ln in lines[2:])
    if words < OFF_MIN_WORDS:
        return False, f"only {words} words after the answer lines"
    for ln in set(lines):
        if lines.count(ln) > 2:
            return False, "a line repeats more than twice"
    return True, "ok"


# -- the needle panel ---------------------------------------------------------

NEEDLE_DEPTHS = (262144, 131072)
NEEDLE_POSITION = 0.55
NEEDLE_MAX_TOKENS = 256
NEEDLE_SEED = 13520261013
NEEDLE_FIELD = "ANSWER1"
NEEDLE_FLOOR_FRAC = 0.98  # a prompt shorter than this share of its target is invalid


def needle_reserve(depth: int) -> int:
    return 3500 if depth > 250000 else 400


_FIELD_RE = re.compile(r"^[\s>*_`#\-]*" + NEEDLE_FIELD + r"[\s*_`]*:[\s*_`]*(.*)$", re.I)


def score_answer(content: str | None, nonce: str) -> tuple[bool, bool]:
    """(exact, legacy). exact: the first ANSWER1 line's value equals the nonce
    after markdown decoration and trailing punctuation are stripped. legacy: the
    nonce occurs anywhere in the content."""
    content = content or ""
    legacy = bool(nonce) and nonce in content
    exact = False
    for line in content.splitlines():
        m = _FIELD_RE.match(line)
        if m:
            val = m.group(1).strip().strip("*_`\"'[]()<> ").rstrip(".,;:!").strip("*_`\"' ")
            exact = val == nonce
            break
    return exact, legacy


# -- profile rows -------------------------------------------------------------
# A profile names the launch row a server was started with and picks the rows
# its card carries. It does not start or configure a server.

_FULL = {"n262": 23, "n131": 10}
_HALF = {"n262": 12, "n131": 5}
_LADDER_2 = (4096, 131072)

PROFILES = {
    "fn-quad-3090": {"decode_depths": _LADDER_2, "needle": _FULL},
    "fn-quad-20g-sm86": {"decode_depths": _LADDER_2, "needle": _FULL},
    "27b-tp4-3090": {"decode_depths": _LADDER_2, "needle": _HALF},
    "27b-tp2-3090": {"decode_depths": _LADDER_2, "needle": _HALF},
    "27b-tp1-3090": {"decode_depths": LADDER_DEPTHS, "needle": _HALF},
}
GENERIC = {"decode_depths": LADDER_DEPTHS, "needle": _HALF}
FAMILIES = {"fn-": PROFILES["fn-quad-3090"], "27b-": PROFILES["27b-tp1-3090"]}


def resolve_profile(name: str) -> dict:
    """The rows for a profile name: a published row, else its family (``fn-``,
    ``27b-``), else the generic ladder. ``known`` says which."""
    if name in PROFILES:
        row, known = PROFILES[name], "published row"
    else:
        row, known = None, None
        for prefix, fam in FAMILIES.items():
            if name.startswith(prefix):
                row, known = fam, "family " + prefix
                break
        if row is None:
            row, known = GENERIC, "generic (not a published row)"
    return {
        "name": name,
        "known": known,
        "decode_depths": tuple(row["decode_depths"]),
        "needle": dict(row["needle"]),
    }


# -- self-test ------------------------------------------------------------------

def image_bytes() -> bytes:
    return IMAGE_PATH.read_bytes()


@functools.lru_cache(maxsize=1)
def image_data_uri() -> str:
    return "data:image/png;base64," + base64.b64encode(image_bytes()).decode()


def selftest() -> list[str]:
    """Problems with the frozen inputs (empty = intact)."""
    bad = []
    try:
        if hashlib.sha256(image_bytes()).hexdigest() != IMAGE_SHA256:
            bad.append("the card image differs from the frozen one")
    except OSError as e:
        bad.append(f"the card image is unreadable: {e}")
    p = build_prompt(REQ[4096], PROMPT_SELFTEST_NONCE)
    if hashlib.sha256(p.encode()).hexdigest() != PROMPT_SELFTEST_SHA256:
        bad.append("the prompt builder output differs from the frozen one")
    return bad
