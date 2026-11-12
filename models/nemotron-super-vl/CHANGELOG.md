# Changelog

Release notes of the Nemotron 3.5 Super VL recipe, newest first. See the [README](README.md).

## Merlin 3.0.2.1

The recipe runs on **vLLM 0.33.0**, as in 3.0.2. See the [engine changelog](../../docs/CHANGELOG.md#merlin-3021) for
the full release. The Merlin 3.0.2.1 results are in the [2026-11-12 bench card](benchmarks/2026-11-12/BENCH-CARD.md); the
2026-11-04 card stays as the 3.0.2 comparator.

- **Earlier turns keep their thinking in the prompt by default**, so the prefix cache no longer misses at the previous
  answer. A request's own `chat_template_kwargs` can still set `truncate_history_thinking: true`.
