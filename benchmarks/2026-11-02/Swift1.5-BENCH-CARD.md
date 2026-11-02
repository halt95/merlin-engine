# Swift1.5 — Merlin 3.0.1 bench card, 2026-11-02

> **Provenance.** Maintainer measurements on the reference host. Measured on 4 x RTX 3090 at 220 W with P2P enabled (aikitoria's open-gpu-kernel-modules P2P driver) and Zagitta's gpu-unlock. PCIe Gen4, no NVLink, driver 610.43.02.
> Engine: Merlin 3.0.1 (the engine wheel is unchanged from 3.0.0). Served by `merlin serve` with the shipped
> defaults, one fresh boot per layout; the 3.0.1 bench runs were measured 2026-11-01 to 2026-11-02 (UTC); thinking-on rows use reasoning effort medium, the server default.
> Streams are client-timed; drivers, raw streams and boot logs are not published.
>
> Checkpoint: [halt95/Swift1.5-Qwen3.8-Flash-Next-W4A16-Merlin](https://huggingface.co/halt95/Swift1.5-Qwen3.8-Flash-Next-W4A16-Merlin). The 3.0.0 column is the [2026-10-20 card](../2026-10-20/Swift1.5-BENCH-CARD.md).
> Layout: TP2 × PP2 + EP on four cards (`fn-swift1.5-quad-3090`), MTP K=3 probabilistic, FP8 KV; thinking-on effort `medium` (the server default; thinking-on rows re-measured at medium on a fresh boot, 2026-11-02); boot 382 s on 2026-11-01, 362 s on 2026-11-02; KV pool 924,993 tokens.

## 1. Single-stream decode ladder

Same frozen ladder driver as the 3.0.0 card: T=0, a needle prompt at depth, every send a cold prefill, ≤ 256 tokens
with thinking on and the no-think answer-plus-summary task with thinking off (natural stop, ≤ 1,024 tokens); 3 sends
per cell, each cell the median of the valid sends. tok/s = tokens after the first streamed event / time from the first
to the last event. With MTP, read the median event interval (ITL) together with tokens per event.

| prompt tokens | thinking | 3.0.1 tok/s | 3.0.0 tok/s | change | ITL ms | tokens/event | TTFT s | valid |
|---|---|---|---|---|---|---|---|---|
| 4,046 | on | 175.9 | 177.4 | −0.8 % | 17.74 | 3.07 | 0.67 | 3/3 |
| 32,566 | on | 176.2 | 175.3 | +0.5 % | 17.97 | 3.12 | 4.58 | 3/3 |
| 130,386 | on | 186.1 | 180.2 | +3.3 % | 18.01 | 3.18 | 18.42 | 3/3 |
| 259,557 | on | 184.4 | 180.4 | +2.2 % | 18.47 | 3.33 | 38.02 | 3/3 |
| 4,105 | off | 126.6 | 123.3 | +2.7 % | 17.94 | 2.25 | 0.70 | 3/3 |
| 32,625 | off | 129.2 | 125.9 | +2.6 % | 17.96 | 2.30 | 4.59 | 3/3 |
| 130,445 | off | 127.0 | 125.5 | +1.2 % | 18.33 | 2.30 | 18.54 | 3/3 |
| 259,616 | off | 132.2 | 131.1 | +0.8 % | 18.53 | 2.42 | 38.16 | 3/3 |

Thinking-on rows: reasoning effort medium (the server default), re-measured on 2026-11-02 on a fresh boot of a later release candidate; the 3.0.0 column was measured at effort `xhigh`, so the thinking-on changes compare different efforts. Thinking-off rows, prefill and the thinking-off agents rows are from the 2026-11-01 runs, some on an earlier release candidate. None of these runs used the final 3.0.1 commit.

## 2. Exact 262K

Exactly 261,888 prompt tokens + 128 out, both thinking modes, 3 sends (send 1 cold; sends 2 and 3 reuse the prefix
before the needle). Decode is the median of the 3 sends; TTFT is send 1's.

| prompt tokens | thinking | 3.0.1 tok/s | 3.0.0 tok/s | change | ITL ms | tokens/event | cold TTFT s (3.0.1 / 3.0.0) |
|---|---|---|---|---|---|---|---|
| 261,888 | on | 185.6 | 176.7 | +5.0 % | 18.43 | 3.20 | 38.3 / 44.9 |
| 261,888 | off | 145.4 | 164.1 | −11.4 % | 18.42 | 2.56 | 38.3 / 44.7 |

Thinking off at exact 262K reads −11.4 % against the 3.0.0 card. The time per decode step is
18.42 ms against 18.62 ms; the difference is MTP acceptance (2.56 against 2.91 tokens per
step), which varies over a 128-token sample. The 259K thinking-off ladder cell reads +0.8 %.

## 3. Cold prefill

| prompt tokens | 3.0.1 prefill | 3.0.0 prefill | change | 3.0.1 TTFT s |
|---|---|---|---|---|
| 10,199 | 6,623 tok/s | 5,676 tok/s | +16.7 % | 1.54 |
| 101,887 | 7,043 tok/s | 6,055 tok/s | +16.3 % | 14.47 |
| 261,888 | 6,783 tok/s | 5,834 tok/s | +16.3 % | 38.6 |

One cold send per depth (8 tokens out at 10K and 100K; the 2026-11-01 run's cold thinking-on send at 262K, so it can differ slightly from the table above).

## 4. Agents at once

Same method as the 3.0.0 card: eight 16K conversations, then eight simultaneous 512-token follow-ups, thinking
off, temperature 0. Total tok/s = tokens decoded while all eight stream / that overlap window.

| layout | 3.0.1 total tok/s | 3.0.0 total tok/s | change | overlap window s |
|---|---|---|---|---|
| Four cards | 650 | 643 | +1.1 % | 5.90 |

**Thinking on, reasoning effort medium** (the server default). The same eight conversations and follow-ups, with thinking on and each request setting medium. A 512-token follow-up may end inside the reasoning, and reasoning tokens count, so these totals measure decoded tokens, not finished answers.

| layout | 3.0.1 total tok/s, thinking on | overlap window s |
|---|---|---|
| Four cards | 673 | 5.27 |

## 5. GSM8K-200 at reasoning effort medium

The 3.0.0 set (the same 200 problems, 5-shot, same scorer, temperature 0), with thinking on at medium, on four cards: **199/200** correct, none cut at the 8,192-token cap (mean 280 tokens, longest 1,946). The thinking-off score on the 3.0.0 card is 197/200.

Sessions held at once, images, the thinking-off GSM8K-200 and the needle panel were not re-run; the [2026-10-20 card](../2026-10-20/Swift1.5-BENCH-CARD.md) has them. The KV pool and the sessions it holds are unchanged.
