# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modified by the Merlin project; see NOTICE.

from __future__ import annotations

import codecs
import json
import re

from vllm.tokenizers import TokenizerLike


def _replace_none_with_empty(tokens: list[str | None]):
    for i, token in enumerate(tokens):
        if token is None:
            tokens[i] = ""


def _convert_tokens_to_string_with_added_encoders(
    tokenizer: TokenizerLike,
    output_tokens: list[str],
    skip_special_tokens: bool,
    spaces_between_special_tokens: bool,
) -> str:
    # Adapted from
    # https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/tokenization_utils.py#L921
    # NOTE(woosuk): The following code is slow because it runs a for loop over
    # the output_tokens. In Python, running a for loop over a list can be slow
    # even when the loop body is very simple.
    # Performance improvements: avoid repeated attribute and function lookups;
    # localize frequently used objects;

    sub_texts: list[str] = []
    current_sub_text: list[str] = []
    convert_tokens_to_string = tokenizer.convert_tokens_to_string
    added_vocab_set = set(tokenizer.get_added_vocab())
    all_special_tokens = (
        set(tokenizer.all_special_tokens) if skip_special_tokens else ()
    )

    for token in output_tokens:
        # Use precomputed set for skip-special check
        if token in all_special_tokens:
            continue
        if token in added_vocab_set:
            if current_sub_text:
                sub_texts.append(convert_tokens_to_string(current_sub_text))
                current_sub_text.clear()
            sub_texts.append(token)
        else:
            current_sub_text.append(token)
    if current_sub_text:
        sub_texts.append(convert_tokens_to_string(current_sub_text))
    if spaces_between_special_tokens:
        return " ".join(sub_texts)
    return "".join(sub_texts)


# 5 is an arbitrary value that should work for all
# tokenizers (bigger = more conservative).
INITIAL_INCREMENTAL_DETOKENIZATION_OFFSET = 5
# Bound undecodable runs; ordinary UTF-8 needs at most four byte tokens.
MAX_PENDING_TOKENS = 16
# Lossless flush: at the cap only text that no
# later token can change is emitted: the window is split before at least
# HELD_TOKENS held tokens (a UTF-8 decoder reads a byte with at most the 3 bytes
# after it), at a point where the text so far, plus the held tokens decoded on
# their own, is exactly the whole window's text. Valid UTF-8 (whose window only
# stays open past the cap on literal U+FFFD characters) is never cut inside a
# character. If no such point exists yet the run waits, up to twice the cap.
HELD_TOKENS = 4
# ByteFallback pieces. That decoder renders a whole run of byte pieces at once:
# valid, or one U+FFFD per piece, so a run's text is only final once its bytes
# are complete UTF-8 or can no longer become valid.
_BYTE_PIECE = re.compile(r"<0x([0-9A-Fa-f]{2})>")


def _byte_run_is_final(pieces: list) -> bool:
    run = bytearray()
    for piece in reversed(pieces):
        m = _BYTE_PIECE.fullmatch(piece) if isinstance(piece, str) else None
        if m is None:
            break
        run.append(int(m.group(1), 16))
    if not run:
        return True
    run.reverse()
    try:
        bytes(run).decode("utf-8")
        return True  # complete and valid
    except UnicodeDecodeError:
        pass
    try:
        codecs.getincrementaldecoder("utf-8")().decode(bytes(run), final=False)
    except UnicodeDecodeError:
        return True  # invalid whatever follows
    return False  # an incomplete character later bytes may still complete


def byte_runs_complete(pieces: list) -> bool:
    """True when every run of ByteFallback byte pieces is complete, valid UTF-8,
    so decoding the pieces on their own renders them as in a longer text."""
    run = bytearray()
    for piece in [*pieces, None]:
        m = _BYTE_PIECE.fullmatch(piece) if isinstance(piece, str) else None
        if m is not None:
            run.append(int(m.group(1), 16))
            continue
        if run:
            try:
                bytes(run).decode("utf-8")
            except UnicodeDecodeError:
                return False
            run.clear()
    return True


def settled_split(decode, seq, start: int, read: int, end: int, piece=None):
    """``(j, text)``: emit ``text`` (the window's text from ``read`` to ``j``) and
    hold ``seq[j:end]``; None when no split point is settled yet.
    ``decode`` maps a slice of ``seq`` to its text; ``piece`` maps an item to
    its token string (default: the items are token strings)."""
    before = decode(seq[start:read])
    full = decode(seq[start:end])
    for j in range(end - HELD_TOKENS, read, -1):
        settled = decode(seq[start:j])
        if (
            settled.startswith(before)
            and full.startswith(settled)
            and full[len(settled) :] == decode(seq[j:end])
            and _byte_run_is_final(
                [piece(x) if piece else x for x in seq[start:j]]
            )
        ):
            return j, settled[len(before) :]
    return None


_CACHED_MARKER_KEY = "_vllm_space_marker_cache"
_NOT_CACHED = "__not_computed__"


def get_leading_space_marker(tokenizer: TokenizerLike) -> str | None:
    """Read the space marker from the tokenizer's pre_tokenizer config.

    Only Metaspace pre_tokenizers (used by SentencePiece-based models like
    Llama, Mistral, T5) have a replacement character whose leading instance
    gets stripped by decode(). ByteLevel (GPT-2), BertPreTokenizer (BERT),
    and others do not have this issue.

    Returns the marker character, or None if decode() is safe for single
    tokens.
    """
    cached = getattr(tokenizer, _CACHED_MARKER_KEY, _NOT_CACHED)
    if cached is not _NOT_CACHED:
        return cached  # type: ignore[return-value]

    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None:
        result = None
    else:
        result = None
        try:
            config = json.loads(backend.to_str())
        except Exception:
            pass
        else:
            pre = config.get("pre_tokenizer") or {}
            pre_type = pre.get("type")
            if pre_type == "Metaspace":
                result = pre.get("replacement", "▁")
            elif pre_type == "Sequence":
                for sub in pre.get("pretokenizers", []):
                    if sub.get("type") == "Metaspace":
                        result = sub.get("replacement", "▁")
                        break

    setattr(tokenizer, _CACHED_MARKER_KEY, result)
    return result


def _restore_leading_spaces(raw_token: str, token_str: str, marker: str) -> str:
    """Restore leading spaces that decode() stripped from a raw vocab piece."""
    num_markers = 0
    for ch in raw_token:
        if ch != marker:
            break
        num_markers += 1
    if num_markers == 0:
        return token_str
    existing = len(token_str) - len(token_str.lstrip(" "))
    missing = num_markers - existing
    return " " * missing + token_str if missing > 0 else token_str


def convert_prompt_ids_to_tokens(
    tokenizer: TokenizerLike,
    prompt_ids: list[int],
    skip_special_tokens: bool = False,
) -> tuple[list[str], int, int]:
    """Converts the prompt ids to tokens and returns the tokens and offsets
    for incremental detokenization.

    Note that not all tokens are converted to strings. Only the tokens that
    are necessary for incremental detokenization are converted to strings.
    """
    # We do not need to convert the whole prompt to tokens.
    # Offset a little more in case we have special tokens.
    new_tokens = tokenizer.convert_ids_to_tokens(
        prompt_ids[-INITIAL_INCREMENTAL_DETOKENIZATION_OFFSET - 2 :],
        skip_special_tokens=skip_special_tokens,
    )
    read_offset = len(new_tokens)
    prefix_offset = max(read_offset - INITIAL_INCREMENTAL_DETOKENIZATION_OFFSET, 0)
    # This is required to guard against out-of-vocab prompt token ids
    _replace_none_with_empty(new_tokens)  # type: ignore[arg-type]
    return new_tokens, prefix_offset, read_offset


def convert_ids_list_to_tokens(
    tokenizer: TokenizerLike,
    token_ids: list[int],
) -> list[str]:
    """Detokenize the input ids individually.

    Uses decode() for human-readable output, then checks the raw vocab
    piece via convert_ids_to_tokens() to restore any leading spaces that
    decode() stripped (SentencePiece add_dummy_prefix inverse).

    Args:
      tokenizer: tokenizer used by model under test
      token_ids: convert these tokens (Python list form)

    Returns:
      Python list of token string representations

    """
    if not token_ids:
        return []
    marker = get_leading_space_marker(tokenizer)
    if marker is None:
        return [tokenizer.decode([tid]) or "" for tid in token_ids]
    raw_tokens = tokenizer.convert_ids_to_tokens(token_ids)
    return [
        _restore_leading_spaces(raw, tokenizer.decode([tid]) or "", marker)
        for tid, raw in zip(token_ids, raw_tokens)
    ]


# Based on
# https://github.com/huggingface/text-generation-inference/blob/v0.9.4/server/text_generation_server/models/model.py#L62C9-L62C15
# under Apache 2.0 license
def detokenize_incrementally(
    tokenizer: TokenizerLike,
    all_input_ids: list[int],
    prev_tokens: list[str] | None,
    prefix_offset: int,
    read_offset: int,
    skip_special_tokens: bool = False,
    spaces_between_special_tokens: bool = True,
) -> tuple[list[str], str, int, int]:
    """Detokenizes the input ids incrementally and returns the new tokens
    and the new text.

    If `prev_tokens` is None, this function will convert the input ids to
    tokens and return the tokens and the new text. Otherwise, it will return the
    new tokens and the new text.

    This function will also return the new prefix offset and the new read
    offset to be used in the next iteration.

    The offsets are necessary to defeat cleanup algorithms in the decode which
    decide to add a space or not depending on the surrounding ids.

    Args:
        tokenizer: The tokenizer to use.
        all_input_ids: The input ids. The last id is the new token id.
        prev_tokens: The previous tokens. If None, this function will convert
            the input ids to tokens and return the tokens and the new text.
        prefix_offset: The prefix offset.
        read_offset: The read offset.
        skip_special_tokens: Whether to skip special tokens.
        spaces_between_special_tokens: Whether to add spaces between special
            tokens.

    """
    new_token_id = all_input_ids[-1]
    # This is the first iteration for this sequence
    is_first_iter = prev_tokens is None
    if is_first_iter:
        (prev_tokens, prefix_offset, read_offset) = convert_prompt_ids_to_tokens(
            tokenizer, all_input_ids[:-1], skip_special_tokens=skip_special_tokens
        )
    assert prev_tokens is not None

    # If the new token id is out of bounds, return an empty string.
    if 0 <= new_token_id < len(tokenizer):
        # Put new_token_id in a list so skip_special_tokens is respected
        new_tokens = tokenizer.convert_ids_to_tokens(
            [new_token_id], skip_special_tokens=skip_special_tokens
        )
        if isinstance(new_tokens, str):
            new_tokens = [new_tokens]
        else:
            # This is required to guard against out-of-vocab prompt token ids
            # (for example when using dummy weights)
            _replace_none_with_empty(new_tokens)  # type: ignore[arg-type]
    else:
        new_tokens = [""]
    output_token_count = len(prev_tokens) + len(new_tokens)
    window_start = (
        0 if is_first_iter else min(prefix_offset, read_offset, len(prev_tokens))
    )
    output_tokens = (
        prev_tokens[window_start:] if window_start else prev_tokens
    ) + new_tokens

    # If this is the first iteration, return all tokens.
    if is_first_iter:
        new_tokens = output_tokens

    # The prefix text is necessary only to defeat cleanup algorithms in
    # the decode which decide to add a space or not depending on the
    # surrounding ids.
    if tokenizer.is_fast or not tokenizer.get_added_vocab():
        prefix_text = tokenizer.convert_tokens_to_string(
            output_tokens[prefix_offset - window_start : read_offset - window_start]
        )
        new_text = tokenizer.convert_tokens_to_string(
            output_tokens[prefix_offset - window_start :]
        )
    else:
        prefix_text = _convert_tokens_to_string_with_added_encoders(
            tokenizer,
            output_tokens[prefix_offset - window_start : read_offset - window_start],
            skip_special_tokens=skip_special_tokens,
            spaces_between_special_tokens=spaces_between_special_tokens,
        )
        new_text = _convert_tokens_to_string_with_added_encoders(
            tokenizer,
            output_tokens[prefix_offset - window_start :],
            skip_special_tokens=skip_special_tokens,
            spaces_between_special_tokens=spaces_between_special_tokens,
        )

    if len(new_text) <= len(prefix_text) or new_text.endswith("�"):
        # utf-8 char at the end means it's a potential unfinished byte sequence
        # from byte fallback tokenization.
        # If it's in the middle, it's probably a real invalid id generated
        # by the model
        pending = output_token_count - read_offset
        if pending < MAX_PENDING_TOKENS:
            return new_tokens, "", prefix_offset, read_offset
        # Emit only the settled text.
        if tokenizer.is_fast or not tokenizer.get_added_vocab():
            to_text = tokenizer.convert_tokens_to_string
        else:

            def to_text(tokens: list[str]) -> str:
                return _convert_tokens_to_string_with_added_encoders(
                    tokenizer,
                    tokens,
                    skip_special_tokens=skip_special_tokens,
                    spaces_between_special_tokens=spaces_between_special_tokens,
                )

        split = settled_split(
            to_text,
            output_tokens,
            prefix_offset - window_start,
            read_offset - window_start,
            len(output_tokens),
        )
        if split is not None:
            j, text = split
            return new_tokens, text, read_offset, j + window_start
        if pending < 2 * MAX_PENDING_TOKENS:
            return new_tokens, "", prefix_offset, read_offset
        # Never settled: flush the whole run, bounding the work.

    new_text = new_text[len(prefix_text) :]
    return new_tokens, new_text, read_offset, output_token_count
