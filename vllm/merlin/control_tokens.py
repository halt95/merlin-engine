# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Extra added tokens in the chat-content control set.

Merlin hands chat-content sanitising to Transformers (>= 5.20):
``apply_chat_template(..., sanitize_control_tokens=True)`` encodes, as plain
text, every control token found in messages, tools and documents. Transformers
5.20.2 takes that control set from
``PreTrainedTokenizerBase._get_control_tokens(chat_template)``: the tokenizer's
special tokens, plus either
the added tokens listed in ``tokenizer.chat_control_tokens`` or, when that is
unset, the added tokens that appear in the template's literal text.

Added tokens that are neither special nor in the template stay outside that
set, so a client can still type them and get the real token ids. On Ornith
these are the six FIM tokens.

``VLLM_MERLIN_EXTRA_CONTROL_TOKENS`` (comma-separated; set by the Ornith
launcher) adds such tokens to Transformers' own set for whatever template is
rendered. The library's per-template inference stays as it is. Only tokens the
tokenizer really has as added tokens are added. With the knob empty, or
``sanitize_control_tokens`` off, nothing changes.
"""

from collections.abc import Callable
from functools import wraps
from typing import Any

# The Ornith 1.5 (Qwen3.5) tokenizer's non-special added FIM / repo tokens.
ORNITH_FIM_CONTROL_TOKENS = (
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|fim_suffix|>",
    "<|fim_pad|>",
    "<|repo_name|>",
    "<|file_sep|>",
)


def extra_control_tokens() -> tuple[str, ...]:
    """The configured extra control tokens, in order, without blanks or repeats."""
    import vllm.envs as envs

    raw = envs.VLLM_MERLIN_EXTRA_CONTROL_TOKENS or ""
    return tuple(dict.fromkeys(t.strip() for t in raw.split(",") if t.strip()))


def merge_control_tokens(
    base: tuple[str, ...], added_token_contents: set[str], extras: tuple[str, ...]
) -> tuple[str, ...]:
    """``base`` (Transformers' set, order kept) plus each extra that is an added
    token of this tokenizer and not already in ``base``."""
    have = set(base)
    return tuple(base) + tuple(
        t for t in extras if t in added_token_contents and t not in have
    )


def wrap_get_control_tokens(
    get_control_tokens: Callable[[Any, str], tuple[str, ...]],
) -> Callable[[Any, str], tuple[str, ...]]:
    """Wrap ``PreTrainedTokenizerBase._get_control_tokens`` to add the extras."""

    @wraps(get_control_tokens)
    def _get_control_tokens(self, chat_template: str) -> tuple[str, ...]:
        base = tuple(get_control_tokens(self, chat_template))
        extras = extra_control_tokens()
        if not extras:
            return base
        added = {
            getattr(t, "content", str(t)) for t in self.added_tokens_decoder.values()
        }
        return merge_control_tokens(base, added, extras)

    _get_control_tokens.__merlin_extra_control_tokens__ = True  # type: ignore[attr-defined]
    return _get_control_tokens


def install(tokenizer_class: type) -> None:
    """Install the wrapper on ``tokenizer_class`` (vLLM's per-tokenizer cached
    subclass) when its Transformers base has the 5.20 hook."""
    original = getattr(tokenizer_class, "_get_control_tokens", None)
    if original is None or getattr(original, "__merlin_extra_control_tokens__", False):
        return
    tokenizer_class._get_control_tokens = wrap_get_control_tokens(original)
