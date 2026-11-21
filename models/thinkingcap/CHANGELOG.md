# Changelog

Release notes of the ThinkingCap recipe, newest first. See the [README](README.md).

## Merlin 3.0.2.2

The recipe runs on **vLLM 0.33.0**, as in 3.0.2.1; the recipe version stays 3.0.0. See the
[engine changelog](../../docs/CHANGELOG.md#merlin-3022) for the full release.

- **Qwen3 parser:** the end-of-output hang with tools offered is fixed; content starts at its first visible
  character after the reasoning, streamed or not (`"\n\nAnswer"` becomes `"Answer"`); non-streaming replies keep
  the whitespace beside tool calls and return text written after a call; streamed content after whitespace that
  follows the reasoning no longer depends on where the stream is cut; forced and strict tool calls open with the tool-call token.
- **A Qwen3 tool call opens only at a line start** (the start of the answer, a line start after optional
  whitespace, or right after a call) and never inside a code fence or an inline code span; elsewhere the opener
  is text (`MERLIN_QWEN3_LITERAL_SPANS=0` leaves this as in 3.0.2.1).
- **Strict and free-form tool-call grammars** bound each string from a length table, write one whitespace run
  per gap, and a slow-fill guard ends a tool-call request whose mask fills keep taking long (see the engine
  changelog's known limits for the two value shapes that still hit it).
- **The four-card launcher with P2P on** uses the Qwen3.8-27B profile, so it gets the four decode settings too: at 4K,
  one-stream ms per output token 16.6 % (thinking on) and 16.8 % (thinking off) lower, output identical.

## Merlin 3.0.2.1

The recipe runs on **vLLM 0.33.0**, as in 3.0.2. See the [engine changelog](../../docs/CHANGELOG.md#merlin-3021) for
the full release. The Merlin 3.0.2.1 results are in the [2026-11-12 bench card](benchmarks/2026-11-12/BENCH-CARD.md); the
2026-11-05 card stays as the 3.0.2 comparator.

- **Two-card launcher with P2P on** uses the Qwen3.8-27B profile, so it gets the same six launcher defaults; decode is
  +19 % / +6 % at 4K / 131K against 3.0.2 and level at 262K (step time 29.26 ms against 29.34 ms; the 128-token rate varies with the text sent, 12 sends on the published wheel), and eight agents +3 %, thinking on.
- **Quoted tool calls stay text** on the Qwen3 parser (`MERLIN_QWEN3_LITERAL_SPANS=0` turns it off).
