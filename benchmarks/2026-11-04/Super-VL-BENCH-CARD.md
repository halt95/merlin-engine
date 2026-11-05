# Nemotron 3.5 Super VL — Merlin 3.0.2 bench card, 2026-11-04

> **Provenance.** Maintainer measurements on the reference host. Measured on 4 x RTX 3090 at 220 W with P2P enabled (aikitoria's open-gpu-kernel-modules P2P driver) and Zagitta's gpu-unlock. PCIe Gen4, no NVLink, driver 610.43.02.
> Engine: Merlin 3.0.2 on vLLM 0.33.0 (wheel `0.33.0+merlin.665f43d2cf`), on the release candidate; the changes after it
> touch only Flash-Next code paths. Served by `merlin serve` with the shipped defaults, one fresh boot per layout;
> measured 2026-11-04 (UTC).
> Streams are client-timed; drivers, raw streams and boot logs are not published.
>
> Checkpoint: [halt95/Nemotron-3.5-Super-VL-120B-A12B-W4A16-Merlin](https://huggingface.co/halt95/Nemotron-3.5-Super-VL-120B-A12B-W4A16-Merlin). The 3.0.1 column is the [2026-11-02 card](../2026-11-02/Super-VL-BENCH-CARD.md).
> Layout: four cards (`svl-quad-3090`), MTP, FP8 KV, the shipped 1M context; thinking on (this template has no effort levels); boot 376 s; KV pool 1,856,367 tokens.

## 1. Single-stream decode ladder

Same frozen ladder driver as the 3.0.1 card: T=0, a needle prompt at depth, every send a cold prefill, ≤ 256 tokens
with thinking on and the no-think answer-plus-summary task with thinking off (natural stop, ≤ 1,024 tokens); 3 sends
per cell, each cell the median of the valid sends. tok/s = tokens after the first streamed event / time from the first
to the last event. With MTP, read the median event interval (ITL) together with tokens per event. The 3.0.2 run measured
the 4K and 131K rows; the 32K and 259K rows of the 3.0.1 card were not re-run.

| prompt tokens | thinking | 3.0.2 tok/s | 3.0.1 tok/s | change | ITL ms | tokens/event | TTFT s | valid |
|---|---|---|---|---|---|---|---|---|
| 4,269 | on | 101.2 | 102.6 | −1 % | 36.42 | 3.66 | 1.04 | 3/3 |
| 136,625 | on | 100.6 | 101.5 | −1 % | 36.92 | 3.61 | 27.73 | 3/3 |
| 4,328 | off | 86.6 | 85.8 | +1 % | 36.31 | 3.14 | 1.05 | 3/3 |
| 136,684 | off | 77.3 | 80.8 | −4 % | 36.94 | 2.78 | 27.91 | 1/3 |

Super VL only, as on the 3.0.1 card: with thinking off, a send counts if it returned HTTP 200, generated at least 200
tokens and had no streaming problems. The answer-format, word-count and natural-stop checks are not applied, because
at short context the model writes its deliberation as visible text and runs to the 1,024-token limit (all sends at
4K). Those rows time that text, not a finished answer. At 131K two sends with under 200 tokens are excluded.

## 2. Exact 262K

Exactly 261,888 prompt tokens + 128 out, both thinking modes, 3 sends with one cache salt (send 1 cold). Decode, ITL
and tokens per event are the median of the 3 sends; TTFT is send 1's. The sends column gives the spread of the three
sends: MTP acceptance moves decode by several percent between sends and between runs, so a difference to 3.0.1 smaller
than that spread is not resolved by these runs.
In this run send 2 was served cold as well. Thinking on: no send reused the prefix; thinking off: no send reused the prefix.

| prompt tokens | thinking | 3.0.2 tok/s (median) | 3.0.2 sends, tok/s | 3.0.1 tok/s | change | ITL ms | tokens/event | cold TTFT s (3.0.2 / 3.0.1) |
|---|---|---|---|---|---|---|---|---|
| 261,888 | on | 105.7 | 99.7–108.0 | 92.7 | +14 % | 37.15 | 3.66 | 62.1 / 66.9 |
| 261,888 | off | 86.3 | 85.0–87.1 | 81.8 | +5 % | 37.19 | 3.05 | 62.2 / 66.9 |

## 3. Cold prefill

| prompt tokens (3.0.2 / 3.0.1) | 3.0.2 prefill | 3.0.1 prefill | change | 3.0.2 TTFT s |
|---|---|---|---|---|
| 10,671 / 10,671 | 4,914 tok/s | 4,545 tok/s | +8 % | 2.17 |
| 106,725 / 106,725 | 5,111 tok/s | 4,689 tok/s | +9 % | 20.88 |
| 261,888 / 261,888 | 4,217 tok/s | 3,916 tok/s | +8 % | 62.1 |

One cold send per depth (8 tokens out at 10K and 100K). The 3.0.2 262K row is the cold thinking-on send of table 2:
prompt tokens / time to first token. The 3.0.1 262K row is the 2026-11-01 run's cold send, as the 3.0.1 card notes, so
its time to first token can differ slightly from that card's table 2.

## 4. Not re-run

The 8-agent runs, GSM8K-200, sessions held at once, images and the needle panel were not re-run for 3.0.2; the [3.0.1 card](../2026-11-02/Super-VL-BENCH-CARD.md) and the [3.0.0 card](../2026-10-20/Super-VL-BENCH-CARD.md) have them.
