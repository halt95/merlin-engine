# Swift1.5-Huihui — Merlin 3.0.1 bench card, 2026-11-02

> **Provenance.** Maintainer measurements on the reference host. Measured on 4 x RTX 3090 at 220 W with P2P enabled (aikitoria's open-gpu-kernel-modules P2P driver) and Zagitta's gpu-unlock. PCIe Gen4, no NVLink, driver 610.43.02.
> Engine: Merlin 3.0.1 (the engine wheel is unchanged from 3.0.0). Served by `merlin serve` with the shipped
> defaults, one fresh boot per layout; the 3.0.1 bench runs were measured 2026-11-01 to 2026-11-02 (UTC); thinking-on rows use reasoning effort medium, the server default.
> Streams are client-timed; drivers, raw streams and boot logs are not published.
>
> Checkpoint: [halt95/Swift1.5-Huihui-Qwen3.8-Flash-Next-W4A16-Merlin](https://huggingface.co/halt95/Swift1.5-Huihui-Qwen3.8-Flash-Next-W4A16-Merlin). The 3.0.0 column is the [2026-10-20 card](../2026-10-20/BENCH-CARD.md).
> Layout: TP2 × PP2 + EP on four cards (`fn-swift1.5-huihui-quad-3090`), MTP K=3 probabilistic, FP8 KV; thinking-on effort `medium` (the server default; thinking-on rows re-measured at medium on a fresh boot, 2026-11-02); boot 177 s; KV pool 924,993 tokens.

## 1. Single-stream decode ladder

Same frozen ladder driver as the 3.0.0 card: T=0, a needle prompt at depth, every send a cold prefill, ≤ 256 tokens
with thinking on and the no-think answer-plus-summary task with thinking off (natural stop, ≤ 1,024 tokens); 3 sends
per cell, each cell the median of the valid sends. tok/s = tokens after the first streamed event / time from the first
to the last event. With MTP, read the median event interval (ITL) together with tokens per event.

| prompt tokens | thinking | 3.0.1 tok/s | 3.0.0 tok/s | change | ITL ms | tokens/event | TTFT s | valid |
|---|---|---|---|---|---|---|---|---|
| 4,046 | on | 177.7 | 187.3 | −5.1 % | 18.00 | 3.15 | 0.67 | 3/3 |
| 32,566 | on | 181.4 | 192.0 | −5.5 % | 17.93 | 3.20 | 4.57 | 3/3 |
| 130,386 | on | 188.8 | 188.5 | +0.2 % | 18.06 | 3.32 | 18.36 | 3/3 |
| 259,557 | on | 186.4 | 188.6 | −1.2 % | 18.50 | 3.37 | 37.88 | 3/3 |
| 4,105 | off | 136.0 | 132.3 | +2.8 % | 17.97 | 2.43 | 0.71 | 3/3 |
| 32,625 | off | 129.9 | 128.1 | +1.4 % | 18.03 | 2.33 | 4.59 | 3/3 |
| 130,445 | off | 130.6 | 126.4 | +3.3 % | 18.22 | 2.36 | 18.36 | 3/3 |
| 259,616 | off | 129.4 | 126.7 | +2.1 % | 18.60 | 2.37 | 37.83 | 3/3 |

Thinking-on rows: reasoning effort medium (the server default), re-measured on 2026-11-02 on a fresh boot of the same release tree; the 3.0.0 column was measured at effort `xhigh`, so the thinking-on changes compare different efforts. Thinking-off rows, prefill and the thinking-off agents rows are unchanged from the earlier runs.

## 2. Exact 262K

Exactly 261,888 prompt tokens + 128 out, both thinking modes, 3 sends (send 1 cold; sends 2 and 3 reuse the prefix
before the needle). Decode is the median of the 3 sends; TTFT is send 1's.

| prompt tokens | thinking | 3.0.1 tok/s | 3.0.0 tok/s | change | ITL ms | tokens/event | cold TTFT s (3.0.1 / 3.0.0) |
|---|---|---|---|---|---|---|---|
| 261,888 | on | 196.0 | 189.5 | +3.4 % | 18.41 | 3.37 | 38.0 / 44.5 |
| 261,888 | off | 162.2 | 156.6 | +3.6 % | 18.29 | 2.84 | 38.1 / 44.5 |

## 3. Cold prefill

| prompt tokens | 3.0.1 prefill | 3.0.0 prefill | change | 3.0.1 TTFT s |
|---|---|---|---|---|
| 10,199 | 6,612 tok/s | 5,706 tok/s | +15.9 % | 1.54 |
| 101,887 | 7,075 tok/s | 6,097 tok/s | +16.0 % | 14.40 |
| 261,888 | 6,866 tok/s | 5,880 tok/s | +16.8 % | 38.1 |

One cold send per depth (8 tokens out at 10K and 100K; the exact-262K row's cold thinking-on send at 262K).

## 4. Agents at once

Same method as the 3.0.0 card: eight 16K conversations, then eight simultaneous 512-token follow-ups, thinking
off, temperature 0. Total tok/s = tokens decoded while all eight stream / that overlap window.

| layout | 3.0.1 total tok/s | 3.0.0 total tok/s | change | overlap window s |
|---|---|---|---|---|
| Four cards | 681 | 660 | +3.2 % | 5.60 |

This is a second pass. In the first, one agent stopped after one token (a natural end of sequence), so the
eight never streamed together; the 3.0.0 card was re-run the same way.

**Thinking on, reasoning effort medium** (the server default). The same eight conversations and follow-ups, with thinking on and each request setting medium. A 512-token follow-up may end inside the reasoning, and reasoning tokens count, so these totals measure decoded tokens, not finished answers.

| layout | 3.0.1 total tok/s, thinking on | overlap window s |
|---|---|---|
| Four cards | 703 | 5.11 |

## 5. GSM8K-200 at reasoning effort medium

The 3.0.0 set (the same 200 problems, 5-shot, same scorer, temperature 0), with thinking on at medium, on four cards: **198/200** correct, none cut at the 8,192-token cap (mean 282 tokens, longest 1,969). The thinking-off score on the 3.0.0 card is 198/200.

Sessions held at once, images, the thinking-off GSM8K-200 and the needle panel were not re-run; the [2026-10-20 card](../2026-10-20/BENCH-CARD.md) has them. The KV pool and the sessions it holds are unchanged.
