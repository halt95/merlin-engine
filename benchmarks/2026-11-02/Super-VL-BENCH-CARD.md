# Nemotron 3.5 Super VL — Merlin 3.0.1 bench card, 2026-11-02

> **Provenance.** Maintainer measurements on the reference host. Measured on 4 x RTX 3090 at 220 W with P2P enabled (aikitoria's open-gpu-kernel-modules P2P driver) and Zagitta's gpu-unlock. PCIe Gen4, no NVLink, driver 610.43.02.
> Engine: Merlin 3.0.1 (the engine wheel is unchanged from 3.0.0). Served by `merlin serve` with the shipped
> defaults, one fresh boot per layout; the 3.0.1 bench runs were measured 2026-11-01 to 2026-11-02 (UTC); this template has no effort levels, so thinking-on rows are its full thinking, as on the 3.0.0 card.
> Streams are client-timed; drivers, raw streams and boot logs are not published.
>
> Checkpoint: [halt95/Nemotron-3.5-Super-VL-120B-A12B-W4A16-Merlin](https://huggingface.co/halt95/Nemotron-3.5-Super-VL-120B-A12B-W4A16-Merlin). The 3.0.0 column is the [2026-10-20 card](../2026-10-20/Super-VL-BENCH-CARD.md).
> Layout: four cards (`svl-quad-3090`), MTP, FP8 KV, the shipped 1M context; thinking on (this template has no effort levels; the driver's `medium` has no effect); boot 392 s; KV pool 1,856,367 tokens.
> The 3.0.0 card's KV pool (1,392,275 tokens) was measured at a 262K context length; this card serves the shipped 1M context.

## 1. Single-stream decode ladder

Same frozen ladder driver as the 3.0.0 card: T=0, a needle prompt at depth, every send a cold prefill, ≤ 256 tokens
with thinking on and the no-think answer-plus-summary task with thinking off (natural stop, ≤ 1,024 tokens); 3 sends
per cell, each cell the median of the valid sends. tok/s = tokens after the first streamed event / time from the first
to the last event. With MTP, read the median event interval (ITL) together with tokens per event.

| prompt tokens | thinking | 3.0.1 tok/s | 3.0.0 tok/s | change | ITL ms | tokens/event | TTFT s | valid |
|---|---|---|---|---|---|---|---|---|
| 4,269 | on | 102.6 | 96.6 | +6.2 % | 36.01 | 3.66 | 1.14 | 3/3 |
| 34,147 | on | 98.5 | 101.2 | −2.7 % | 36.06 | 3.51 | 6.81 | 3/3 |
| 136,625 | on | 101.5 | 101.3 | +0.2 % | 35.53 | 3.51 | 30.17 | 3/3 |
| 262,839 | on | 104.5 | — | — | 36.11 | 3.66 | 67.38 | 3/3 |
| 4,328 | off | 85.8 | — | — | 36.09 | 3.08 | 1.14 | 3/3 |
| 34,206 | off | 87.3 | — | — | 35.60 | 3.09 | 6.83 | 3/3 |
| 136,684 | off | 80.8 | — | — | 35.80 | 2.82 | 30.18 | 2/3 |
| 262,898 | off | 80.5 | — | — | 36.32 | 2.87 | 67.36 | 2/3 |

A cell with 2/3 valid sends is the median of its valid sends only.

Super VL only: with thinking off, a send counts if it returned HTTP 200, generated at least 200 tokens and had no
streaming problems. The answer-format, word-count and natural-stop checks are not applied, because at short context
the model writes its deliberation as visible text and runs to the 1,024-token limit (all sends at 4K and 32K). Those
rows time that text, not a finished answer. Two sends with under 200 tokens are excluded. The 3.0.0 card's 1K to 256K
ladder measured thinking on only.

## 2. Exact 262K

Exactly 261,888 prompt tokens + 128 out, both thinking modes, 3 sends (send 1 cold; sends 2 and 3 reuse the prefix
before the needle). Decode is the median of the 3 sends; TTFT is send 1's.

| prompt tokens | thinking | 3.0.1 tok/s | 3.0.0 tok/s | change | ITL ms | tokens/event | cold TTFT s (3.0.1 / 3.0.0) |
|---|---|---|---|---|---|---|---|
| 261,888 | on | 92.7 | 91.5 | +1.3 % | 36.44 | 3.20 | 66.9 / 88.8 |
| 261,888 | off | 81.8 | 82.8 | −1.2 % | 36.06 | 2.78 | 66.9 / 88.5 |

## 3. Cold prefill

| prompt tokens | 3.0.1 prefill | 3.0.0 prefill | change | 3.0.1 TTFT s |
|---|---|---|---|---|
| 10,671 | 4,545 tok/s | 3,737 tok/s | +21.6 % | 2.35 |
| 106,725 | 4,689 tok/s | 3,744 tok/s | +25.2 % | 22.76 |
| 261,888 | 3,916 tok/s | 2,951 tok/s | +32.7 % | 66.9 |

One cold send per depth (8 tokens out at 10K and 100K; the 2026-11-01 run's cold thinking-on send at 262K, so it can differ slightly from the table above).

## 4. Agents at once

Same method as the 3.0.0 card: eight 16K conversations, then eight simultaneous 512-token follow-ups, thinking
off, temperature 0. Total tok/s = tokens decoded while all eight stream / that overlap window.

| layout | 3.0.1 total tok/s | 3.0.0 total tok/s | change | overlap window s |
|---|---|---|---|---|
| Four cards | 432 | 429 | +0.7 % | 8.89 |

**Thinking on** (this template has no effort levels). The same eight conversations and follow-ups, with thinking on. A 512-token follow-up may end inside the reasoning, and reasoning tokens count, so these totals measure decoded tokens, not finished answers.

| layout | 3.0.1 total tok/s, thinking on | overlap window s |
|---|---|---|
| Four cards | 432 | 8.63 |

## 5. GSM8K-200 with thinking on

The 3.0.0 set (the same 200 problems, 5-shot, same scorer, temperature 0), with thinking on, on four cards: **197/200** correct, one cut at the 8,192-token cap with no answer and counted as wrong (mean 580 tokens). The thinking-off score on the 3.0.0 card is 195/200.

Sessions held at once, images, the thinking-off GSM8K-200 and the needle panel were not re-run; the [2026-10-20 card](../2026-10-20/Super-VL-BENCH-CARD.md) has them.
