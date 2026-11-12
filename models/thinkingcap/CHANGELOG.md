# Changelog

Release notes of the ThinkingCap recipe, newest first. See the [README](README.md).

## Merlin 3.0.2.1

The recipe runs on **vLLM 0.33.0**, as in 3.0.2. See the [engine changelog](../../docs/CHANGELOG.md#merlin-3021) for
the full release. The Merlin 3.0.2.1 results are in the [2026-11-12 bench card](benchmarks/2026-11-12/BENCH-CARD.md); the
2026-11-05 card stays as the 3.0.2 comparator.

- **Two-card launcher with P2P on** uses the Qwen3.8-27B profile, so it gets the same six launcher defaults; decode is
  +19 % / +6 % at 4K / 131K against 3.0.2 and level at 262K (step time 29.26 ms against 29.34 ms; the 128-token rate varies with the text sent, 12 sends on the published wheel), and eight agents +3 %, thinking on.
- **Quoted tool calls stay text** on the Qwen3 parser (`MERLIN_QWEN3_LITERAL_SPANS=0` turns it off).
