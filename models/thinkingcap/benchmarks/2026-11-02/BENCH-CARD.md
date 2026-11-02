# ThinkingCap — Merlin 3.0.1 bench card, 2026-11-02

> **Provenance.** Maintainer measurements on the reference host. Measured on 4 x RTX 3090 at 220 W with P2P enabled (aikitoria's open-gpu-kernel-modules P2P driver) and Zagitta's gpu-unlock. PCIe Gen4, no NVLink, driver 610.43.02.
> Engine: Merlin 3.0.1 (the engine wheel is unchanged from 3.0.0). Served by `merlin serve` with the shipped
> defaults, one fresh boot per layout; the 3.0.1 bench runs were measured 2026-11-01 to 2026-11-02 (UTC); thinking-on rows use reasoning effort medium, the server default.
> Streams are client-timed; drivers, raw streams and boot logs are not published.
>
> Checkpoint: [halt95/ThinkingCap-Qwen3.8-27B-W4A16-Merlin](https://huggingface.co/halt95/ThinkingCap-Qwen3.8-27B-W4A16-Merlin). The 3.0.0 column is the [2026-10-20 card](../2026-10-20/BENCH-CARD.md).
> Four cards: TP4, `27b-tp4-3090`; thinking-on effort `medium` (the server default; thinking-on rows re-measured at medium on a fresh boot, 2026-11-02); boot 171 s; KV pool 2,012,105 tokens.
> Two cards: TP2, `27b-tp2-3090`; thinking-on effort `medium` (the server default; thinking-on rows re-measured at medium on a fresh boot, 2026-11-02); boot 171 s; KV pool 826,689 tokens.
> MTP K=4, FP8 KV, RecoverSSM on.

## 1. Single-stream decode ladder

Same frozen ladder driver as the 3.0.0 card: T=0, a needle prompt at depth, every send a cold prefill, ≤ 256 tokens
with thinking on and the no-think answer-plus-summary task with thinking off (natural stop, ≤ 1,024 tokens); 3 sends
per cell, each cell the median of the valid sends. tok/s = tokens after the first streamed event / time from the first
to the last event. With MTP, read the median event interval (ITL) together with tokens per event.

### Four cards (TP4)

| prompt tokens | thinking | 3.0.1 tok/s | 3.0.0 tok/s | change | ITL ms | tokens/event | TTFT s | valid |
|---|---|---|---|---|---|---|---|---|
| 4,046 | on | 243.8 | 252.9 | −3.6 % | 15.89 | 3.80 | 1.08 | 3/3 |
| 32,566 | on | 241.0 | 252.1 | −4.4 % | 16.45 | 3.85 | 8.81 | 3/3 |
| 130,386 | on | 230.3 | 254.6 | −9.5 % | 17.78 | 3.87 | 46.55 | 3/3 |
| 259,557 | on | 222.3 | 228.6 | −2.8 % | 19.69 | 4.15 | 122.22 | 3/3 |
| 4,105 | off | 189.1 | 188.2 | +0.5 % | 15.95 | 3.00 | 1.14 | 3/3 |
| 32,625 | off | 164.8 | 162.9 | +1.2 % | 16.65 | 2.71 | 8.89 | 3/3 |
| 130,445 | off | 156.2 | 151.2 | +3.3 % | 17.99 | 2.76 | 46.86 | 3/3 |
| 259,616 | off | 146.0 | 138.2 | +5.6 % | 19.83 | 2.85 | 122.82 | 3/3 |

### Two cards (TP2)

| prompt tokens | thinking | 3.0.1 tok/s | 3.0.0 tok/s | change | ITL ms | tokens/event | TTFT s | valid |
|---|---|---|---|---|---|---|---|---|
| 4,046 | on | 203.0 | 198.4 | +2.3 % | 19.85 | 3.95 | 1.74 | 3/3 |
| 32,566 | on | 185.0 | 199.4 | −7.2 % | 21.05 | 3.80 | 15.13 | 3/3 |
| 130,386 | on | 162.3 | 182.5 | −11.1 % | 24.04 | 3.70 | 85.39 | 3/3 |
| 259,557 | on | 142.8 | 146.2 | −2.3 % | 29.33 | 3.95 | 231.14 | 3/3 |
| 4,105 | off | 145.1 | 143.0 | +1.5 % | 19.91 | 2.87 | 1.77 | 3/3 |
| 32,625 | off | 137.4 | 135.9 | +1.1 % | 21.15 | 2.88 | 15.18 | 3/3 |
| 130,445 | off | 120.4 | 115.9 | +3.9 % | 24.21 | 2.87 | 85.11 | 3/3 |
| 259,616 | off | 99.9 | 94.6 | +5.6 % | 28.84 | 2.87 | 232.16 | 3/3 |

Thinking-on rows: reasoning effort medium (the server default), re-measured on 2026-11-02 on a fresh boot of the same release tree; the 3.0.0 column was measured at effort `xhigh`, so the thinking-on changes compare different efforts. Thinking-off rows, prefill and the thinking-off agents rows are unchanged from the earlier runs.

## 2. Exact 262K

Exactly 261,888 prompt tokens + 128 out, both thinking modes, 3 sends (send 1 cold; sends 2 and 3 reuse the prefix
before the needle). Decode is the median of the 3 sends; TTFT is send 1's.

### Four cards (TP4)

| prompt tokens | thinking | 3.0.1 tok/s | 3.0.0 tok/s | change | ITL ms | tokens/event | cold TTFT s (3.0.1 / 3.0.0) |
|---|---|---|---|---|---|---|---|
| 261,888 | on | 238.0 | 248.2 | −4.1 % | 19.56 | 4.27 | 124.1 / 164.3 |
| 261,888 | off | 147.2 | 146.0 | +0.8 % | 19.90 | 2.78 | 124.6 / 164.3 |

### Two cards (TP2)

| prompt tokens | thinking | 3.0.1 tok/s | 3.0.0 tok/s | change | ITL ms | tokens/event | cold TTFT s (3.0.1 / 3.0.0) |
|---|---|---|---|---|---|---|---|
| 261,888 | on | 159.7 | 164.8 | −3.1 % | 29.67 | 4.36 | 235.4 / 304.6 |
| 261,888 | off | 97.4 | 88.1 | +10.5 % | 28.95 | 2.72 | 235.5 / 304.6 |

## 3. Cold prefill

### Four cards (TP4)

| prompt tokens | 3.0.1 prefill | 3.0.0 prefill | change | 3.0.1 TTFT s |
|---|---|---|---|---|
| 10,199 | 3,869 tok/s | 3,013 tok/s | +28.4 % | 2.64 |
| 101,887 | 2,990 tok/s | 2,295 tok/s | +30.3 % | 34.08 |
| 261,888 | 2,099 tok/s | 1,594 tok/s | +31.7 % | 124.7 |

### Two cards (TP2)

| prompt tokens | 3.0.1 prefill | 3.0.0 prefill | change | 3.0.1 TTFT s |
|---|---|---|---|---|
| 10,199 | 2,319 tok/s | 1,829 tok/s | +26.8 % | 4.40 |
| 101,887 | 1,679 tok/s | 1,297 tok/s | +29.5 % | 60.70 |
| 261,888 | 1,114 tok/s | 860 tok/s | +29.5 % | 235.2 |

One cold send per depth (8 tokens out at 10K and 100K; the exact-262K row's cold thinking-on send at 262K).

## 4. Agents at once

Same method as the 3.0.0 card: eight 16K conversations, then eight simultaneous 512-token follow-ups, thinking
off, temperature 0. Total tok/s = tokens decoded while all eight stream / that overlap window.

| layout | 3.0.1 total tok/s | 3.0.0 total tok/s | change | overlap window s |
|---|---|---|---|---|
| Four cards | 832 | 733 | +13.5 % | 4.48 (short) |
| Two cards | 633 | 502 | +26.1 % | 5.71 |

A window under 5 s is short: read that total loosely, as on the 3.0.0 card.

**Thinking on, reasoning effort medium** (the server default). The same eight conversations and follow-ups, with thinking on and each request setting medium. A 512-token follow-up may end inside the reasoning, and reasoning tokens count, so these totals measure decoded tokens, not finished answers.

| layout | 3.0.1 total tok/s, thinking on | overlap window s |
|---|---|---|
| Four cards | 952 | 3.57 (short) |
| Two cards | 697 | 5.15 |

## 5. GSM8K-200 at reasoning effort medium

The 3.0.0 set (the same 200 problems, 5-shot, same scorer, temperature 0), with thinking on at medium, on four cards: **198/200** correct, none cut at the 8,192-token cap (mean 265 tokens, longest 1,689). The thinking-off score on the 3.0.0 card is 195/200.

Sessions held at once, images, the thinking-off GSM8K-200 and the needle panel were not re-run; the [2026-10-20 card](../2026-10-20/BENCH-CARD.md) has them. The KV pool and the sessions it holds are unchanged.
