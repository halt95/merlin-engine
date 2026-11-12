# Changelog

Release notes of the Swift1.5-Huihui recipe, newest first. See the [README](README.md).

## Merlin 3.0.2.1

The recipe runs on **vLLM 0.33.0**, as in 3.0.2; the recipe version stays 3.0.0. See the
[engine changelog](../../docs/CHANGELOG.md#merlin-3021) for the full release. The Merlin 3.0.2.1 results are in the
[2026-11-12 bench card](benchmarks/2026-11-12/BENCH-CARD.md); the 2026-11-05 card stays as the 3.0.2 comparator.

- **Inherited from the Flash-Next launcher:** FGPC on (`-- --no-enable-mamba-shared-prefix-checkpoint` turns it off),
  `--long-prefill-token-threshold 512` (`LPT=0` turns it off), the prefill runs (`PREFILL_RUNS=0` turns them off),
  quoted tool calls kept as text
  (`MERLIN_QWEN3_LITERAL_SPANS=0` turns it off) and the KV RAM cache fixes.
