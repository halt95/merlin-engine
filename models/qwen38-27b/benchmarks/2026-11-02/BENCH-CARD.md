# Qwen3.8-27B — Merlin 3.0.1 bench card, 2026-11-02

> **Provenance.** Maintainer measurements on the reference host. Measured on 4 x RTX 3090 at 220 W with P2P enabled (aikitoria's open-gpu-kernel-modules P2P driver) and Zagitta's gpu-unlock. PCIe Gen4, no NVLink, driver 610.43.02.
> Engine: Merlin 3.0.1 (the engine wheel is unchanged from 3.0.0). Served by `merlin serve` with the shipped
> defaults, one fresh boot per layout; the 3.0.1 bench runs were measured 2026-11-01 to 2026-11-02 (UTC); the reasoning effort is set per request, as on the 3.0.0 card.
> Streams are client-timed; drivers, raw streams and boot logs are not published.
>
> Checkpoint: [halt95/Qwen3.8-27B-W4A16-Merlin](https://huggingface.co/halt95/Qwen3.8-27B-W4A16-Merlin). The 3.0.0 column is the [2026-10-20 card](../2026-10-20/BENCH-CARD.md).
> Four cards: TP4, `27b-tp4-3090`; thinking-on effort `medium` (the server default); boot 182 s; KV pool 2,012,105 tokens.
> Two cards: TP2 on GPUs 0 and 2, `27b-tp2-3090`; thinking-on effort `medium` (the server default); boot 321 s; KV pool 826,689 tokens.
> One card: TP1, `27b-tp1-3090`; thinking-on effort `medium` (the server default); boot 131 s; KV pool 271,506 tokens.
> MTP K=4, FP8 KV, RecoverSSM on.

## 1. Single-stream decode ladder

Same frozen ladder driver as the 3.0.0 card: T=0, a needle prompt at depth, every send a cold prefill, ≤ 256 tokens
with thinking on and the no-think answer-plus-summary task with thinking off (natural stop, ≤ 1,024 tokens); 3 sends
per cell, each cell the median of the valid sends. tok/s = tokens after the first streamed event / time from the first
to the last event. With MTP, read the median event interval (ITL) together with tokens per event.

### Four cards (TP4)

| prompt tokens | thinking | 3.0.1 tok/s | 3.0.0 tok/s | change | ITL ms | tokens/event | TTFT s | valid |
|---|---|---|---|---|---|---|---|---|
| 4,046 | on | 250.6 | 250.3 | +0.1 % | 16.11 | 4.00 | 1.08 | 3/3 |
| 32,566 | on | 258.6 | 252.4 | +2.5 % | 16.60 | 4.20 | 8.86 | 3/3 |
| 130,386 | on | 242.6 | 235.4 | +3.1 % | 18.06 | 4.20 | 46.78 | 3/3 |
| 259,557 | on | 238.2 | 224.0 | +6.3 % | 19.80 | 4.49 | 122.86 | 3/3 |
| 4,105 | off | 185.6 | 186.9 | −0.7 % | 16.02 | 2.96 | 1.14 | 3/3 |
| 32,625 | off | 169.3 | 168.5 | +0.5 % | 16.71 | 2.80 | 8.90 | 3/3 |
| 130,445 | off | 169.6 | 164.8 | +2.9 % | 18.01 | 2.99 | 46.84 | 3/3 |
| 259,616 | off | 140.2 | 133.0 | +5.4 % | 19.90 | 2.73 | 122.84 | 3/3 |

### Two cards (TP2, GPUs 0 and 2)

| prompt tokens | thinking | 3.0.1 tok/s | 3.0.0 tok/s | change | ITL ms | tokens/event | TTFT s | valid |
|---|---|---|---|---|---|---|---|---|
| 4,046 | on | 210.5 | 206.6 | +1.9 % | 19.88 | 4.13 | 1.73 | 3/3 |
| 32,566 | on | 199.3 | 196.1 | +1.6 % | 21.04 | 4.13 | 15.03 | 3/3 |
| 130,386 | on | 182.1 | 173.3 | +5.1 % | 24.01 | 4.20 | 84.67 | 3/3 |
| 259,557 | on | 162.3 | 153.6 | +5.7 % | 29.06 | 4.57 | 231.89 | 3/3 |
| 4,105 | off | 144.2 | 140.8 | +2.4 % | 19.89 | 2.85 | 1.77 | 3/3 |
| 32,625 | off | 134.5 | 130.7 | +2.9 % | 21.03 | 2.79 | 15.06 | 3/3 |
| 130,445 | off | 125.5 | 119.0 | +5.5 % | 24.05 | 2.95 | 85.02 | 3/3 |
| 259,616 | off | 96.0 | 91.6 | +4.8 % | 29.06 | 2.78 | 232.41 | 3/3 |

### One card (TP1)

| prompt tokens | thinking | 3.0.1 tok/s | 3.0.0 tok/s | change | ITL ms | tokens/event | TTFT s | valid |
|---|---|---|---|---|---|---|---|---|
| 4,046 | on | 125.2 | 123.8 | +1.1 % | 32.99 | 4.06 | 3.32 | 3/3 |
| 32,566 | on | 120.2 | 121.3 | −0.9 % | 35.98 | 4.27 | 29.13 | 3/3 |
| 130,386 | on | 95.2 | 94.8 | +0.4 % | 43.33 | 4.13 | 163.58 | 3/3 |
| 259,557 | on | 80.2 | 80.5 | −0.4 % | 54.83 | 4.34 | 446.56 | 3/3 |
| 4,105 | off | 85.1 | 84.3 | +0.9 % | 33.12 | 2.79 | 3.40 | 3/3 |
| 32,625 | off | 79.3 | 79.0 | +0.4 % | 35.68 | 2.82 | 29.24 | 3/3 |
| 130,445 | off | 67.3 | 67.8 | −0.7 % | 44.18 | 2.96 | 163.54 | 3/3 |
| 259,616 | off | 50.1 | 50.4 | −0.6 % | 56.03 | 2.78 | 446.65 | 3/3 |

## 2. Exact 262K

Exactly 261,888 prompt tokens + 128 out, both thinking modes, 3 sends (send 1 cold; sends 2 and 3 reuse the prefix
before the needle). Decode is the median of the 3 sends; TTFT is send 1's.

### Four cards (TP4)

| prompt tokens | thinking | 3.0.1 tok/s | 3.0.0 tok/s | change | ITL ms | tokens/event | cold TTFT s (3.0.1 / 3.0.0) |
|---|---|---|---|---|---|---|---|
| 261,888 | on | 238.4 | 214.8 | +11.0 % | 19.93 | 4.41 | 124.7 / 164.1 |
| 261,888 | off | 149.1 | 158.1 | −5.7 % | 20.00 | 2.84 | 124.6 / 164.0 |

Thinking off at exact 262K reads −5.7 % against the 3.0.0 card. The time per decode step is
20.00 ms against 20.78 ms; the difference is MTP acceptance (2.84 against 3.12 tokens per
step), which varies over a 128-token sample. The 259K thinking-off ladder cell reads +5.4 %.

### Two cards (TP2, GPUs 0 and 2)

| prompt tokens | thinking | 3.0.1 tok/s | 3.0.0 tok/s | change | ITL ms | tokens/event | cold TTFT s (3.0.1 / 3.0.0) |
|---|---|---|---|---|---|---|---|
| 261,888 | on | 143.5 | 142.5 | +0.7 % | 29.89 | 4.00 | 234.9 / 305.3 |
| 261,888 | off | 108.7 | 111.1 | −2.1 % | 29.17 | 3.05 | 235.6 / 305.1 |

### One card (TP1)

| prompt tokens | thinking | 3.0.1 tok/s | 3.0.0 tok/s | change | ITL ms | tokens/event | cold TTFT s (3.0.1 / 3.0.0) |
|---|---|---|---|---|---|---|---|
| 261,888 | on | 71.4 | 72.9 | −2.1 % | 56.00 | 3.76 | 452.4 / 574.1 |
| 261,888 | off | 50.5 | 49.3 | +2.4 % | 55.84 | 2.67 | 452.2 / 574.2 |

## 3. Cold prefill

### Four cards (TP4)

| prompt tokens | 3.0.1 prefill | 3.0.0 prefill | change | 3.0.1 TTFT s |
|---|---|---|---|---|
| 10,157 | 3,866 tok/s | 3,009 tok/s | +28.5 % | 2.63 |
| 101,845 | 2,995 tok/s | 2,295 tok/s | +30.5 % | 34.01 |
| 261,888 | 2,101 tok/s | 1,596 tok/s | +31.6 % | 124.7 |

### Two cards (TP2, GPUs 0 and 2)

| prompt tokens | 3.0.1 prefill | 3.0.0 prefill | change | 3.0.1 TTFT s |
|---|---|---|---|---|
| 10,157 | 2,301 tok/s | 1,823 tok/s | +26.2 % | 4.41 |
| 101,845 | 1,672 tok/s | 1,297 tok/s | +28.9 % | 60.93 |
| 261,888 | 1,115 tok/s | 858 tok/s | +29.9 % | 234.9 |

### One card (TP1)

| prompt tokens | 3.0.1 prefill | 3.0.0 prefill | change | 3.0.1 TTFT s |
|---|---|---|---|---|
| 10,157 | 1,200 tok/s | 1,021 tok/s | +17.5 % | 8.47 |
| 101,845 | 871 tok/s | 708 tok/s | +23.0 % | 116.87 |
| 261,888 | 579 tok/s | 456 tok/s | +27.0 % | 452.4 |

One cold send per depth (8 tokens out at 10K and 100K; the 2026-11-01 run's cold thinking-on send at 262K, so it can differ slightly from the table above).

## 4. Agents at once

Same method as the 3.0.0 card: eight 16K conversations, then eight simultaneous 512-token follow-ups, thinking
off, temperature 0. Total tok/s = tokens decoded while all eight stream / that overlap window.

| layout | 3.0.1 total tok/s | 3.0.0 total tok/s | change | overlap window s |
|---|---|---|---|---|
| Four cards | 854 | 752 | +13.6 % | 4.30 (short) |
| Two cards | 615 | 517 | +18.9 % | 6.08 |

A window under 5 s is short: read that total loosely, as on the 3.0.0 card.

**Thinking on, reasoning effort medium** (the server default). The same eight conversations and follow-ups, with thinking on and each request setting medium. A 512-token follow-up may end inside the reasoning, and reasoning tokens count, so these totals measure decoded tokens, not finished answers.

| layout | 3.0.1 total tok/s, thinking on | overlap window s |
|---|---|---|
| Four cards | 954 | 3.55 (short) |
| Two cards | 705 | 4.92 (short) |

## 5. GSM8K-200 at reasoning effort medium

The 3.0.0 set (the same 200 problems, 5-shot, same scorer, temperature 0), with thinking on at medium, on four cards: **198/200** correct, none cut at the 8,192-token cap (mean 301 tokens, longest 1,933). The thinking-off score on the 3.0.0 card is 197/200.

Sessions held at once, images, the thinking-off GSM8K-200 and the needle panel were not re-run; the [2026-10-20 card](../2026-10-20/BENCH-CARD.md) has them. The KV pool and the sessions it holds are unchanged.
