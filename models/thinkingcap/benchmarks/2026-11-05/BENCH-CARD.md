# ThinkingCap — Merlin 3.0.2 bench card, 2026-11-05

> **Provenance.** Maintainer measurements on the reference host. Measured on 4 x RTX 3090 at 220 W with P2P enabled (aikitoria's open-gpu-kernel-modules P2P driver) and Zagitta's gpu-unlock. PCIe Gen4, no NVLink, driver 610.43.02.
> Engine: Merlin 3.0.2 on vLLM 0.33.0, the 3.0.2 release tree. Served by `merlin serve` with the shipped
> defaults, one fresh boot per layout for the decode and prefill rows and another for the agents and GSM8K rows; measured 2026-11-05 (UTC).
> Streams are client-timed; drivers, raw streams and boot logs are not published.
>
> Checkpoint: [halt95/ThinkingCap-Qwen3.8-27B-W4A16-Merlin](https://huggingface.co/halt95/ThinkingCap-Qwen3.8-27B-W4A16-Merlin). The 3.0.1 column is the [2026-11-02 card](../2026-11-02/BENCH-CARD.md).
> Four cards: TP4, `27b-tp4-3090`; thinking-on effort `medium` (the server default); boot 216 s (agents and GSM8K boot 216 s); KV pool 2,012,105 tokens.
> Two cards: TP2 on GPUs 0 and 2, `27b-tp2-3090`; thinking-on effort `medium` (the server default); boot 146 s (agents boot 126 s); KV pool 826,689 tokens.
> MTP K=4, FP8 KV, RecoverSSM on.

## 1. Single-stream decode ladder

Same frozen ladder driver as the 3.0.1 card: T=0, a needle prompt at depth, every send a cold prefill, ≤ 256 tokens
with thinking on and the no-think answer-plus-summary task with thinking off (natural stop, ≤ 1,024 tokens); 3 sends
per cell, each cell the median of the valid sends. tok/s = tokens after the first streamed event / time from the first
to the last event. With MTP, read the median event interval (ITL) together with tokens per event. The 3.0.2 run measured
all four depths; the 3.0.1 card's thinking-on rows were measured at effort medium, as here, and its thinking-off rows
are from its 2026-11-01 runs, as that card notes.

### Four cards (TP4)

| prompt tokens | thinking | 3.0.2 tok/s | 3.0.1 tok/s | change | ITL ms | tokens/event | TTFT s | valid |
|---|---|---|---|---|---|---|---|---|
| 4,046 | on | 237.8 | 243.8 | −2 % | 15.61 | 3.65 | 1.07 | 3/3 |
| 32,566 | on | 240.1 | 241.0 | 0 % | 16.19 | 3.78 | 8.80 | 3/3 |
| 130,386 | on | 234.5 | 230.3 | +2 % | 17.64 | 3.86 | 46.54 | 3/3 |
| 259,557 | on | 240.8 | 222.3 | +8 % | 19.59 | 4.26 | 122.33 | 3/3 |
| 4,105 | off | 184.2 | 189.1 | −3 % | 15.56 | 2.85 | 1.14 | 3/3 |
| 32,625 | off | 175.5 | 164.8 | +6 % | 16.25 | 2.82 | 8.83 | 3/3 |
| 130,445 | off | 158.1 | 156.2 | +1 % | 17.80 | 2.76 | 46.59 | 3/3 |
| 259,616 | off | 142.9 | 146.0 | −2 % | 19.48 | 2.74 | 122.34 | 3/3 |

### Two cards (TP2, GPUs 0 and 2)

| prompt tokens | thinking | 3.0.2 tok/s | 3.0.1 tok/s | change | ITL ms | tokens/event | TTFT s | valid |
|---|---|---|---|---|---|---|---|---|
| 4,046 | on | 204.8 | 203.0 | +1 % | 19.81 | 3.98 | 1.76 | 3/3 |
| 32,566 | on | 182.7 | 185.0 | −1 % | 20.88 | 3.72 | 15.20 | 3/3 |
| 130,386 | on | 168.8 | 162.3 | +4 % | 23.90 | 3.78 | 84.70 | 3/3 |
| 259,557 | on | 158.1 | 142.8 | +11 % | 29.23 | 4.38 | 231.91 | 3/3 |
| 4,105 | off | 149.8 | 145.1 | +3 % | 19.78 | 2.95 | 1.78 | 3/3 |
| 32,625 | off | 140.6 | 137.4 | +2 % | 20.89 | 2.91 | 15.08 | 3/3 |
| 130,445 | off | 115.8 | 120.4 | −4 % | 23.97 | 2.74 | 84.83 | 3/3 |
| 259,616 | off | 97.9 | 99.9 | −2 % | 28.81 | 2.80 | 232.29 | 3/3 |

## 2. Exact 262K

Exactly 261,888 prompt tokens + 128 out, both thinking modes, 3 sends with one cache salt (send 1 cold). Decode, ITL
and tokens per event are the median of the 3 sends; TTFT is send 1's. The sends column gives the spread of the three
sends: MTP acceptance moves decode by several percent between sends and between runs, so a difference to 3.0.1 smaller
than that spread is not resolved by these runs.
In this run send 2 was served cold as well. Four cards, thinking on: send 3 reused 142,320 tokens; thinking off: send 3 reused 142,320 tokens. Two cards, thinking on: send 3 reused 142,064 tokens; thinking off: send 3 reused 142,064 tokens. Two cards, thinking on: sends 2 and 3 ended at 122 tokens (natural stop) instead of 128.

### Four cards (TP4)

| prompt tokens | thinking | 3.0.2 tok/s (median) | 3.0.2 sends, tok/s | 3.0.1 tok/s | change | ITL ms | tokens/event | cold TTFT s (3.0.2 / 3.0.1) |
|---|---|---|---|---|---|---|---|---|
| 261,888 | on | 223.0 | 222.4–225.1 | 238.0 | −6 % | 19.60 | 4.00 | 124.2 / 124.1 |
| 261,888 | off | 153.3 | 145.7–153.7 | 147.2 | +4 % | 19.66 | 2.84 | 123.9 / 124.6 |

### Two cards (TP2, GPUs 0 and 2)

| prompt tokens | thinking | 3.0.2 tok/s (median) | 3.0.2 sends, tok/s | 3.0.1 tok/s | change | ITL ms | tokens/event | cold TTFT s (3.0.2 / 3.0.1) |
|---|---|---|---|---|---|---|---|---|
| 261,888 | on | 160.4 | 151.6–160.8 | 159.7 | 0 % | 29.34 | 4.36 | 234.9 / 235.4 |
| 261,888 | off | 100.4 | 95.9–102.6 | 97.4 | +3 % | 28.73 | 2.78 | 235.2 / 235.5 |

Four cards, thinking on at exact 262K reads −6 % against the 3.0.1 card. The time per decode step is 19.60 ms against
19.56 ms; the difference is MTP acceptance (4.00 against 4.27 tokens per step). All three sends of this run read below
the 3.0.1 figure. The 259K thinking-on ladder cell on four cards reads +8 % (4.26 against 4.15 tokens per step).

## 3. Cold prefill

### Four cards (TP4)

| prompt tokens (3.0.2 / 3.0.1) | 3.0.2 prefill | 3.0.1 prefill | change | 3.0.2 TTFT s |
|---|---|---|---|---|
| 10,157 / 10,199 | 3,885 tok/s | 3,869 tok/s | 0 % | 2.61 |
| 101,845 / 101,887 | 3,010 tok/s | 2,990 tok/s | +1 % | 33.84 |
| 261,888 / 261,888 | 2,108 tok/s | 2,099 tok/s | 0 % | 124.2 |

### Two cards (TP2, GPUs 0 and 2)

| prompt tokens (3.0.2 / 3.0.1) | 3.0.2 prefill | 3.0.1 prefill | change | 3.0.2 TTFT s |
|---|---|---|---|---|
| 10,157 / 10,199 | 2,322 tok/s | 2,319 tok/s | 0 % | 4.37 |
| 101,845 / 101,887 | 1,679 tok/s | 1,679 tok/s | 0 % | 60.67 |
| 261,888 / 261,888 | 1,115 tok/s | 1,114 tok/s | 0 % | 234.9 |

One cold send per depth (8 tokens out at 10K and 100K). The 3.0.2 262K row is the cold thinking-on send of table 2:
prompt tokens / time to first token. The 3.0.1 262K row is the 2026-11-01 run's cold send, as the 3.0.1 card notes, so
its time to first token can differ slightly from that card's table 2.

## 4. Agents at once

Same method as the 3.0.1 card: eight 16K conversations, then eight simultaneous 512-token follow-ups, temperature 0.
**Thinking on, reasoning effort medium** (the server default), each request setting medium. A 512-token follow-up may
end inside the reasoning, and reasoning tokens count, so these totals measure decoded tokens, not finished answers.
Total tok/s = tokens decoded while all eight stream / that overlap window. The thinking-off 8-agent run was not repeated
for 3.0.2; the [3.0.1 card](../2026-11-02/BENCH-CARD.md) has it.

| layout | 3.0.2 total tok/s, thinking on | 3.0.1 | change | overlap window s |
|---|---|---|---|---|
| Four cards | 939 | 952 | −1 % | 3.86 (short) |
| Two cards | 708 | 697 | +2 % | 4.82 (short) |

A window under 5 s is short: the replies ended before all eight had overlapped for long, so read that total loosely,
as on the 3.0.1 card (its four-card window was 3.57 s, its two-card window 5.15 s).

## 5. GSM8K-200 at reasoning effort medium

The 3.0.1 card's set and method (the same 200 problems, 5-shot, same scorer, temperature 0), with thinking on at
medium, on four cards: **198/200** correct, none cut at the 8,192-token cap (mean 266 tokens, longest 1,499). The
3.0.1 card scored 198/200; the thinking-off score on the 3.0.0 card is 195/200. GSM8K runs once per checkpoint, on
the four-card boot, so the two-card layout has no GSM8K row.

## 6. Not re-run

The thinking-off 8-agent run, the thinking-off GSM8K-200, sessions held at once, images and the needle panel were not re-run for 3.0.2; the [3.0.1 card](../2026-11-02/BENCH-CARD.md) and the [3.0.0 card](../2026-10-20/BENCH-CARD.md) have them.
