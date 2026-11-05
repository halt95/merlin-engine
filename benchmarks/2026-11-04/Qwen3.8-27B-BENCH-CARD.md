# Qwen3.8-27B — Merlin 3.0.2 bench card, 2026-11-04

> **Provenance.** Maintainer measurements on the reference host. Measured on 4 x RTX 3090 at 220 W with P2P enabled (aikitoria's open-gpu-kernel-modules P2P driver) and Zagitta's gpu-unlock. PCIe Gen4, no NVLink, driver 610.43.02.
> Engine: Merlin 3.0.2 on vLLM 0.33.0 (wheel `0.33.0+merlin.665f43d2cf`), on the release candidate; the changes after it
> touch only Flash-Next code paths. Served by `merlin serve` with the shipped defaults, one fresh boot per layout;
> measured 2026-11-04 (UTC).
> Streams are client-timed; drivers, raw streams and boot logs are not published.
>
> Checkpoint: [halt95/Qwen3.8-27B-W4A16-Merlin](https://huggingface.co/halt95/Qwen3.8-27B-W4A16-Merlin). The 3.0.1 column is the [2026-11-02 card](../2026-11-02/Qwen3.8-27B-BENCH-CARD.md).
> Four cards: TP4, `27b-tp4-3090`; thinking-on effort `medium` (the server default); boot 236 s; KV pool 2,012,105 tokens.
> Two cards: TP2 on GPUs 0 and 2, `27b-tp2-3090`; thinking-on effort `medium` (the server default); boot 165 s; KV pool 826,689 tokens.
> One card: TP1, `27b-tp1-3090`; thinking-on effort `medium` (the server default); boot 146 s; KV pool 271,506 tokens.
> MTP K=4, FP8 KV, RecoverSSM on.

## 1. Single-stream decode ladder

Same frozen ladder driver as the 3.0.1 card: T=0, a needle prompt at depth, every send a cold prefill, ≤ 256 tokens
with thinking on and the no-think answer-plus-summary task with thinking off (natural stop, ≤ 1,024 tokens); 3 sends
per cell, each cell the median of the valid sends. tok/s = tokens after the first streamed event / time from the first
to the last event. With MTP, read the median event interval (ITL) together with tokens per event. The 3.0.2 run measured
the 4K and 131K rows; the 32K and 259K rows of the 3.0.1 card were not re-run.

### Four cards (TP4)

| prompt tokens | thinking | 3.0.2 tok/s | 3.0.1 tok/s | change | ITL ms | tokens/event | TTFT s | valid |
|---|---|---|---|---|---|---|---|---|
| 4,046 | on | 258.1 | 250.6 | +3 % | 15.69 | 4.00 | 1.06 | 3/3 |
| 130,386 | on | 254.6 | 242.6 | +5 % | 17.70 | 4.34 | 46.35 | 3/3 |
| 4,105 | off | 181.4 | 185.6 | −2 % | 15.68 | 2.84 | 1.14 | 3/3 |
| 130,445 | off | 169.0 | 169.6 | 0 % | 17.63 | 2.93 | 46.55 | 3/3 |

### Two cards (TP2, GPUs 0 and 2)

| prompt tokens | thinking | 3.0.2 tok/s | 3.0.1 tok/s | change | ITL ms | tokens/event | TTFT s | valid |
|---|---|---|---|---|---|---|---|---|
| 4,046 | on | 210.8 | 210.5 | 0 % | 19.82 | 4.13 | 1.74 | 3/3 |
| 130,386 | on | 183.9 | 182.1 | +1 % | 23.86 | 4.27 | 85.20 | 3/3 |
| 4,105 | off | 139.4 | 144.2 | −3 % | 19.80 | 2.75 | 1.77 | 3/3 |
| 130,445 | off | 125.9 | 125.5 | 0 % | 23.93 | 2.96 | 84.93 | 3/3 |

### One card (TP1)

| prompt tokens | thinking | 3.0.2 tok/s | 3.0.1 tok/s | change | ITL ms | tokens/event | TTFT s | valid |
|---|---|---|---|---|---|---|---|---|
| 4,046 | on | 130.9 | 125.2 | +5 % | 32.23 | 4.13 | 3.27 | 3/3 |
| 130,386 | on | 99.7 | 95.2 | +5 % | 42.78 | 4.27 | 161.46 | 3/3 |
| 4,105 | off | 82.1 | 85.1 | −4 % | 32.58 | 2.66 | 3.33 | 3/3 |
| 130,445 | off | 66.7 | 67.3 | −1 % | 43.23 | 2.89 | 161.55 | 3/3 |

## 2. Exact 262K

Exactly 261,888 prompt tokens + 128 out, both thinking modes, 3 sends with one cache salt (send 1 cold). Decode, ITL
and tokens per event are the median of the 3 sends; TTFT is send 1's. The sends column gives the spread of the three
sends: MTP acceptance moves decode by several percent between sends and between runs, so a difference to 3.0.1 smaller
than that spread is not resolved by these runs.
In this run send 2 was served cold as well. Four cards, thinking on: send 3 reused 142,320 tokens; thinking off: no send reused the prefix. Two cards, thinking on: send 3 reused 142,064 tokens; thinking off: send 3 reused 142,064 tokens. One card, thinking on: send 3 reused 141,120 tokens; thinking off: send 3 reused 141,120 tokens.

### Four cards (TP4)

| prompt tokens | thinking | 3.0.2 tok/s (median) | 3.0.2 sends, tok/s | 3.0.1 tok/s | change | ITL ms | tokens/event | cold TTFT s (3.0.2 / 3.0.1) |
|---|---|---|---|---|---|---|---|---|
| 261,888 | on | 249.9 | 231.7–250.5 | 238.4 | +5 % | 19.49 | 4.41 | 124.1 / 124.7 |
| 261,888 | off | 163.0 | 159.0–163.2 | 149.1 | +9 % | 19.41 | 2.98 | 123.9 / 124.6 |

### Two cards (TP2, GPUs 0 and 2)

| prompt tokens | thinking | 3.0.2 tok/s (median) | 3.0.2 sends, tok/s | 3.0.1 tok/s | change | ITL ms | tokens/event | cold TTFT s (3.0.2 / 3.0.1) |
|---|---|---|---|---|---|---|---|---|
| 261,888 | on | 145.8 | 145.7–156.8 | 143.5 | +2 % | 29.44 | 4.00 | 235.4 / 234.9 |
| 261,888 | off | 107.0 | 98.5–108.1 | 108.7 | −2 % | 28.86 | 2.98 | 235.8 / 235.6 |

### One card (TP1)

| prompt tokens | thinking | 3.0.2 tok/s (median) | 3.0.2 sends, tok/s | 3.0.1 tok/s | change | ITL ms | tokens/event | cold TTFT s (3.0.2 / 3.0.1) |
|---|---|---|---|---|---|---|---|---|
| 261,888 | on | 76.8 | 74.5–76.8 | 71.4 | +8 % | 55.59 | 4.00 | 447.3 / 452.4 |
| 261,888 | off | 53.8 | 49.0–55.0 | 50.5 | +7 % | 54.26 | 2.78 | 446.2 / 452.2 |

## 3. Cold prefill

### Four cards (TP4)

| prompt tokens (3.0.2 / 3.0.1) | 3.0.2 prefill | 3.0.1 prefill | change | 3.0.2 TTFT s |
|---|---|---|---|---|
| 10,157 / 10,157 | 3,904 tok/s | 3,866 tok/s | +1 % | 2.60 |
| 101,845 / 101,845 | 3,015 tok/s | 2,995 tok/s | +1 % | 33.78 |
| 261,888 / 261,888 | 2,111 tok/s | 2,101 tok/s | 0 % | 124.1 |

### Two cards (TP2, GPUs 0 and 2)

| prompt tokens (3.0.2 / 3.0.1) | 3.0.2 prefill | 3.0.1 prefill | change | 3.0.2 TTFT s |
|---|---|---|---|---|
| 10,157 / 10,157 | 2,331 tok/s | 2,301 tok/s | +1 % | 4.36 |
| 101,845 / 101,845 | 1,677 tok/s | 1,672 tok/s | 0 % | 60.72 |
| 261,888 / 261,888 | 1,112 tok/s | 1,115 tok/s | 0 % | 235.4 |

### One card (TP1)

| prompt tokens (3.0.2 / 3.0.1) | 3.0.2 prefill | 3.0.1 prefill | change | 3.0.2 TTFT s |
|---|---|---|---|---|
| 10,157 / 10,157 | 1,219 tok/s | 1,200 tok/s | +2 % | 8.33 |
| 101,845 / 101,845 | 884 tok/s | 871 tok/s | +1 % | 115.26 |
| 261,888 / 261,888 | 586 tok/s | 579 tok/s | +1 % | 447.3 |

One cold send per depth (8 tokens out at 10K and 100K). The 3.0.2 262K row is the cold thinking-on send of table 2:
prompt tokens / time to first token. The 3.0.1 262K row is the 2026-11-01 run's cold send, as the 3.0.1 card notes, so
its time to first token can differ slightly from that card's table 2.

## 4. Agents at once

Same method as the 3.0.1 card: eight 16K conversations, then eight simultaneous 512-token follow-ups, temperature 0.
**Thinking on, reasoning effort medium** (the server default), each request setting medium. A 512-token follow-up may
end inside the reasoning, and reasoning tokens count, so these totals measure decoded tokens, not finished answers.
Total tok/s = tokens decoded while all eight stream / that overlap window. The thinking-off 8-agent run was not repeated
for 3.0.2; the [3.0.1 card](../2026-11-02/Qwen3.8-27B-BENCH-CARD.md) has it.

| layout | 3.0.2 total tok/s, thinking on | 3.0.1 | change | overlap window s |
|---|---|---|---|---|
| Four cards | 940 | 954 | −1 % | 2.89 (short) |
| Two cards | 682 | 705 | −3 % | 3.89 (short) |

A window under 5 s is short: read that total loosely, as on the 3.0.1 card. One card's pool (271,506 tokens) holds one 262K conversation or two 131K ones, and its layout admits two sequences at
most, so it has no 8-agent row.

## 5. Not re-run

GSM8K-200, sessions held at once, images and the needle panel were not re-run for 3.0.2; the [3.0.1 card](../2026-11-02/Qwen3.8-27B-BENCH-CARD.md) and the [3.0.0 card](../2026-10-20/Qwen3.8-27B-BENCH-CARD.md) have them.
