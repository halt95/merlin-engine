# Ornith 1.5 — Merlin 3.0.1 bench card, 2026-11-02

> **Provenance.** Maintainer measurements on the reference host. Measured on 4 x RTX 3090 at 220 W with P2P enabled (aikitoria's open-gpu-kernel-modules P2P driver) and Zagitta's gpu-unlock. PCIe Gen4, no NVLink, driver 610.43.02.
> Engine: Merlin 3.0.1 (the engine wheel is unchanged from 3.0.0). Served by `merlin serve` with the shipped
> defaults, one fresh boot per layout; the 3.0.1 bench runs were measured 2026-11-01 to 2026-11-02 (UTC); this template has no effort levels, so thinking-on rows are its full thinking, as on the 3.0.0 card.
> Streams are client-timed; drivers, raw streams and boot logs are not published.
>
> Checkpoint: [halt95/Ornith-1.5-122B-A10B-W4A16-Merlin](https://huggingface.co/halt95/Ornith-1.5-122B-A10B-W4A16-Merlin). The 3.0.0 column is the [3.0.0 card (run 2026-10-21, filed under 2026-10-20)](../2026-10-20/Ornith-1.5-BENCH-CARD.md).
> Layout: four cards (`ornith-quad-3090`), MTP, FP8 KV; thinking on (this template has no effort levels; the driver's `medium` has no effect); boot 292 s; KV pool 894,618 tokens.
> Ornith's 3.0.0 card was measured on the 3.0.0 release with the Ornith-specific engine changes listed on that card;
> the comparison is against that build.

## 1. Single-stream decode ladder

Same frozen ladder driver as the 3.0.0 card: T=0, a needle prompt at depth, every send a cold prefill, ≤ 256 tokens
with thinking on and the no-think answer-plus-summary task with thinking off (natural stop, ≤ 1,024 tokens); 3 sends
per cell, each cell the median of the valid sends. tok/s = tokens after the first streamed event / time from the first
to the last event. With MTP, read the median event interval (ITL) together with tokens per event.

| prompt tokens | thinking | 3.0.1 tok/s | 3.0.0 tok/s | change | ITL ms | tokens/event | TTFT s | valid |
|---|---|---|---|---|---|---|---|---|
| 4,046 | on | 145.6 | 143.7 | +1.3 % | 19.12 | 2.75 | 0.71 | 3/3 |
| 32,566 | on | 131.3 | 127.5 | +3.0 % | 19.84 | 2.56 | 5.59 | 3/3 |
| 130,386 | on | 115.0 | 105.7 | +8.8 % | 21.01 | 2.37 | 37.13 | 3/3 |
| 259,557 | on | 99.3 | 89.6 | +10.8 % | 22.57 | 2.20 | 114.30 | 3/3 |
| 4,105 | off | 109.3 | 105.8 | +3.3 % | 19.08 | 2.08 | 0.72 | 3/3 |
| 32,625 | off | 105.3 | 103.4 | +1.8 % | 19.79 | 2.07 | 5.58 | 3/3 |
| 130,445 | off | 95.2 | 98.7 | −3.5 % | 20.98 | 1.98 | 37.18 | 3/3 |
| 259,616 | off | 95.7 | 84.2 | +13.7 % | 22.55 | 2.14 | 114.21 | 3/3 |

## 2. Exact 262K

Exactly 261,888 prompt tokens + 128 out, both thinking modes, 3 sends (send 1 cold; sends 2 and 3 reuse the prefix
before the needle). Decode is the median of the 3 sends; TTFT is send 1's.

| prompt tokens | thinking | 3.0.1 tok/s | 3.0.0 tok/s | change | ITL ms | tokens/event | cold TTFT s (3.0.1 / 3.0.0) |
|---|---|---|---|---|---|---|---|
| 261,888 | on | 104.6 | 99.8 | +4.8 % | 22.56 | 2.25 | 115.8 / 153.7 |
| 261,888 | off | 95.9 | 88.8 | +8.0 % | 22.57 | 2.10 | 115.8 / 153.5 |

## 3. Cold prefill

| prompt tokens | 3.0.1 prefill | 3.0.0 prefill | change | 3.0.1 TTFT s |
|---|---|---|---|---|
| 10,157 | 6,218 tok/s | 5,148 tok/s | +20.8 % | 1.63 |
| 101,845 | 3,994 tok/s | 3,049 tok/s | +31.0 % | 25.50 |
| 261,888 | 2,261 tok/s | 1,704 tok/s | +32.7 % | 115.8 |

One cold send per depth (8 tokens out at 10K and 100K; the 2026-11-01 run's cold thinking-on send at 262K, so it can differ slightly from the table above).

## 4. Agents at once

Same method as the 3.0.0 card: eight 16K conversations, then eight simultaneous 512-token follow-ups, thinking
off, temperature 0. Total tok/s = tokens decoded while all eight stream / that overlap window.

| layout | 3.0.1 total tok/s | 3.0.0 total tok/s | change | overlap window s |
|---|---|---|---|---|
| Four cards | 486 | 468 | +3.9 % | 8.09 |

**Thinking on** (this template has no effort levels). The same eight conversations and follow-ups, with thinking on. A 512-token follow-up may end inside the reasoning, and reasoning tokens count, so these totals measure decoded tokens, not finished answers.

| layout | 3.0.1 total tok/s, thinking on | overlap window s |
|---|---|---|
| Four cards | 528 | 7.36 |

## 5. GSM8K-200 with thinking on

The 3.0.0 set (the same 200 problems, 5-shot, same scorer, temperature 0), with thinking on, on four cards: **196/200** correct, none cut at the 8,192-token cap (mean 261 tokens, longest 3,634). The thinking-off score on the 3.0.0 card is 190/200.

Sessions held at once, images, the thinking-off GSM8K-200 and the needle panel were not re-run; the [3.0.0 card](../2026-10-20/Ornith-1.5-BENCH-CARD.md) has them. The KV pool and the sessions it holds are unchanged.
