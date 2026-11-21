# Changelog

Release notes of the Ornith 1.5 recipe, newest first. See the [README](README.md).

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

## Merlin 3.0.2.1

The recipe runs on **vLLM 0.33.0**, as in 3.0.2. See the [engine changelog](../../docs/CHANGELOG.md#merlin-3021) for
the full release. The Merlin 3.0.2.1 results are in the [2026-11-12 bench card](benchmarks/2026-11-12/BENCH-CARD.md); the
2026-11-04 card stays as the 3.0.2 comparator.
