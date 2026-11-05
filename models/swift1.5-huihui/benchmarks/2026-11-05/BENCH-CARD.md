# Swift1.5-Huihui — Merlin 3.0.2 bench card, 2026-11-05

> **Provenance.** Maintainer measurements on the reference host. Measured on 4 x RTX 3090 at 220 W with P2P enabled (aikitoria's open-gpu-kernel-modules P2P driver) and Zagitta's gpu-unlock. PCIe Gen4, no NVLink, driver 610.43.02.
> Engine: Merlin 3.0.2 on vLLM 0.33.0, the 3.0.2 release tree. Served by `merlin serve` with the shipped
> defaults, one fresh boot for the decode and prefill rows and another for the agents and GSM8K rows; measured 2026-11-05 (UTC).
> Streams are client-timed; drivers, raw streams and boot logs are not published.
>
> Checkpoint: [halt95/Swift1.5-Huihui-Qwen3.8-Flash-Next-W4A16-Merlin](https://huggingface.co/halt95/Swift1.5-Huihui-Qwen3.8-Flash-Next-W4A16-Merlin). The 3.0.1 column is the [2026-11-02 card](../2026-11-02/BENCH-CARD.md).
> Layout: TP2 × PP2 + EP on four cards (`fn-swift1.5-huihui-quad-3090`), MTP K=3 probabilistic, FP8 KV; thinking-on effort `medium` (the server default); boot 397 s (agents and GSM8K boot 352 s); KV pool 924,993 tokens.

## 1. Single-stream decode ladder

Same frozen ladder driver as the 3.0.1 card: T=0, a needle prompt at depth, every send a cold prefill, ≤ 256 tokens
with thinking on and the no-think answer-plus-summary task with thinking off (natural stop, ≤ 1,024 tokens); 3 sends
per cell, each cell the median of the valid sends. tok/s = tokens after the first streamed event / time from the first
to the last event. With MTP, read the median event interval (ITL) together with tokens per event. The 3.0.2 run measured
all four depths; the 3.0.1 card's thinking-off rows are from its 2026-11-01 runs, as that card notes.

| prompt tokens | thinking | 3.0.2 tok/s | 3.0.1 tok/s | change | ITL ms | tokens/event | TTFT s | valid |
|---|---|---|---|---|---|---|---|---|
| 4,046 | on | 179.9 | 177.7 | +1 % | 17.60 | 3.14 | 0.67 | 3/3 |
| 32,566 | on | 189.4 | 181.4 | +4 % | 17.68 | 3.28 | 4.60 | 3/3 |
| 130,386 | on | 188.9 | 188.8 | 0 % | 18.10 | 3.32 | 18.50 | 3/3 |
| 259,557 | on | 183.8 | 186.4 | −1 % | 18.41 | 3.24 | 38.18 | 3/3 |
| 4,105 | off | 135.2 | 136.0 | −1 % | 17.54 | 2.36 | 0.69 | 3/3 |
| 32,625 | off | 131.0 | 129.9 | +1 % | 17.82 | 2.30 | 4.64 | 3/3 |
| 130,445 | off | 136.9 | 130.6 | +5 % | 18.18 | 2.46 | 18.57 | 3/3 |
| 259,616 | off | 128.8 | 129.4 | 0 % | 18.40 | 2.35 | 38.18 | 3/3 |

## 2. Exact 262K

Exactly 261,888 prompt tokens + 128 out, both thinking modes, 3 sends with one cache salt (send 1 cold). Decode, ITL
and tokens per event are the median of the 3 sends; TTFT is send 1's. The sends column gives the spread of the three
sends: MTP acceptance moves decode by several percent between sends and between runs, so a difference to 3.0.1 smaller
than that spread is not resolved by these runs.
In this run send 2 was served cold as well. Thinking on: send 3 reused 135,168 tokens; thinking off: send 3 reused 135,168 tokens.

| prompt tokens | thinking | 3.0.2 tok/s (median) | 3.0.2 sends, tok/s | 3.0.1 tok/s | change | ITL ms | tokens/event | cold TTFT s (3.0.2 / 3.0.1) |
|---|---|---|---|---|---|---|---|---|
| 261,888 | on | 192.6 | 191.8–197.0 | 196.0 | −2 % | 18.19 | 3.28 | 38.3 / 38.0 |
| 261,888 | off | 145.8 | 144.0–163.6 | 162.2 | −10 % | 18.34 | 2.56 | 38.2 / 38.1 |

Thinking off at exact 262K reads −10 % against the 3.0.1 card. The time per decode step is 18.34 ms against 18.29 ms;
the difference is MTP acceptance (2.56 against 2.84 tokens per step), and the 3.0.1 figure lies inside the spread of
this run's three sends. The 259K thinking-off ladder cell reads 0 %.

## 3. Cold prefill

| prompt tokens (3.0.2 / 3.0.1) | 3.0.2 prefill | 3.0.1 prefill | change | 3.0.2 TTFT s |
|---|---|---|---|---|
| 10,157 / 10,199 | 6,608 tok/s | 6,612 tok/s | 0 % | 1.54 |
| 101,845 / 101,887 | 7,008 tok/s | 7,075 tok/s | −1 % | 14.53 |
| 261,888 / 261,888 | 6,830 tok/s | 6,866 tok/s | −1 % | 38.3 |

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
| Four cards | 718 | 703 | +2 % | 5.01 |

## 5. GSM8K-200 at reasoning effort medium

The 3.0.1 card's set and method (the same 200 problems, 5-shot, same scorer, temperature 0), with thinking on at
medium, on four cards: **198/200** correct, none cut at the 8,192-token cap (mean 278 tokens, longest 1,437). The
3.0.1 card scored 198/200; the thinking-off score on the 3.0.0 card is 198/200.

## 6. Not re-run

The thinking-off 8-agent run, the thinking-off GSM8K-200, sessions held at once, images and the needle panel were not re-run for 3.0.2; the [3.0.1 card](../2026-11-02/BENCH-CARD.md) and the [3.0.0 card](../2026-10-20/BENCH-CARD.md) have them.
