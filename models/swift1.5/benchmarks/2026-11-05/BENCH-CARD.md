# Swift1.5 — Merlin 3.0.2 bench card, 2026-11-05

> **Provenance.** Maintainer measurements on the reference host. Measured on 4 x RTX 3090 at 220 W with P2P enabled (aikitoria's open-gpu-kernel-modules P2P driver) and Zagitta's gpu-unlock. PCIe Gen4, no NVLink, driver 610.43.02.
> Engine: Merlin 3.0.2 on vLLM 0.33.0, the 3.0.2 release tree. Served by `merlin serve` with the shipped
> defaults, one fresh boot for the decode and prefill rows and another for the agents and GSM8K rows; measured 2026-11-05 (UTC).
> Streams are client-timed; drivers, raw streams and boot logs are not published.
>
> Checkpoint: [halt95/Swift1.5-Qwen3.8-Flash-Next-W4A16-Merlin](https://huggingface.co/halt95/Swift1.5-Qwen3.8-Flash-Next-W4A16-Merlin). The 3.0.1 column is the [2026-11-02 card](../2026-11-02/BENCH-CARD.md).
> Layout: TP2 × PP2 + EP on four cards (`fn-swift1.5-quad-3090`), MTP K=3 probabilistic, FP8 KV; thinking-on effort `medium` (the server default); boot 407 s (agents and GSM8K boot 372 s); KV pool 924,993 tokens.

## 1. Single-stream decode ladder

Same frozen ladder driver as the 3.0.1 card: T=0, a needle prompt at depth, every send a cold prefill, ≤ 256 tokens
with thinking on and the no-think answer-plus-summary task with thinking off (natural stop, ≤ 1,024 tokens); 3 sends
per cell, each cell the median of the valid sends. tok/s = tokens after the first streamed event / time from the first
to the last event. With MTP, read the median event interval (ITL) together with tokens per event. The 3.0.2 run measured
all four depths; the 3.0.1 card's thinking-off rows are from its 2026-11-01 runs, as that card notes.

| prompt tokens | thinking | 3.0.2 tok/s | 3.0.1 tok/s | change | ITL ms | tokens/event | TTFT s | valid |
|---|---|---|---|---|---|---|---|---|
| 4,046 | on | 180.8 | 175.9 | +3 % | 17.81 | 3.15 | 0.67 | 3/3 |
| 32,566 | on | 181.0 | 176.2 | +3 % | 17.79 | 3.16 | 4.65 | 3/3 |
| 130,386 | on | 186.9 | 186.1 | 0 % | 18.00 | 3.25 | 18.85 | 3/3 |
| 259,557 | on | 191.8 | 184.4 | +4 % | 18.20 | 3.38 | 38.51 | 3/3 |
| 4,105 | off | 130.7 | 126.6 | +3 % | 17.78 | 2.31 | 0.69 | 3/3 |
| 32,625 | off | 136.1 | 129.2 | +5 % | 17.85 | 2.41 | 4.66 | 3/3 |
| 130,445 | off | 130.0 | 127.0 | +2 % | 18.00 | 2.32 | 18.76 | 3/3 |
| 259,616 | off | 129.8 | 132.2 | −2 % | 18.39 | 2.36 | 38.55 | 3/3 |

## 2. Exact 262K

Exactly 261,888 prompt tokens + 128 out, both thinking modes, 3 sends with one cache salt (send 1 cold). Decode, ITL
and tokens per event are the median of the 3 sends; TTFT is send 1's. The sends column gives the spread of the three
sends: MTP acceptance moves decode by several percent between sends and between runs, so a difference to 3.0.1 smaller
than that spread is not resolved by these runs.
In this run send 2 was served cold as well. Thinking on: send 3 reused 135,168 tokens; thinking off: send 3 reused 135,168 tokens.

| prompt tokens | thinking | 3.0.2 tok/s (median) | 3.0.2 sends, tok/s | 3.0.1 tok/s | change | ITL ms | tokens/event | cold TTFT s (3.0.2 / 3.0.1) |
|---|---|---|---|---|---|---|---|---|
| 261,888 | on | 186.1 | 169.1–191.8 | 185.6 | 0 % | 18.35 | 3.20 | 38.7 / 38.3 |
| 261,888 | off | 143.8 | 139.3–160.7 | 145.4 | −1 % | 18.23 | 2.51 | 38.7 / 38.3 |

## 3. Cold prefill

| prompt tokens (3.0.2 / 3.0.1) | 3.0.2 prefill | 3.0.1 prefill | change | 3.0.2 TTFT s |
|---|---|---|---|---|
| 10,157 / 10,199 | 6,569 tok/s | 6,623 tok/s | −1 % | 1.55 |
| 101,845 / 101,887 | 6,932 tok/s | 7,043 tok/s | −2 % | 14.69 |
| 261,888 / 261,888 | 6,762 tok/s | 6,783 tok/s | 0 % | 38.7 |

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
| Four cards | 659 | 673 | −2 % | 5.59 |

## 5. GSM8K-200 at reasoning effort medium

The 3.0.1 card's set and method (the same 200 problems, 5-shot, same scorer, temperature 0), with thinking on at
medium, on four cards: **199/200** correct, none cut at the 8,192-token cap (mean 277 tokens, longest 2,322). The
3.0.1 card scored 199/200; the thinking-off score on the 3.0.0 card is 197/200.

## 6. Not re-run

The thinking-off 8-agent run, the thinking-off GSM8K-200, sessions held at once, images and the needle panel were not re-run for 3.0.2; the [3.0.1 card](../2026-11-02/BENCH-CARD.md) and the [3.0.0 card](../2026-10-20/BENCH-CARD.md) have them.
