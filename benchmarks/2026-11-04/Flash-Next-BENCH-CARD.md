# Qwen3.8-Flash-Next — Merlin 3.0.2 bench card, 2026-11-04

> **Provenance.** Maintainer measurements on the reference host. Measured on 4 x RTX 3090 at 220 W with P2P enabled (aikitoria's open-gpu-kernel-modules P2P driver) and Zagitta's gpu-unlock. PCIe Gen4, no NVLink, driver 610.43.02.
> Engine: Merlin 3.0.2 on vLLM 0.33.0 (wheel `0.33.0+merlin.665f43d2cf`), the final 3.0.2 engine. Served by `merlin serve` with the shipped
> defaults, one fresh boot; measured 2026-11-04 (UTC).
> Streams are client-timed; drivers, raw streams and boot logs are not published.
>
> Checkpoint: [halt95/Qwen3.8-Flash-Next-W4A16-Merlin](https://huggingface.co/halt95/Qwen3.8-Flash-Next-W4A16-Merlin). The 3.0.1 column is the [2026-11-02 card](../2026-11-02/Flash-Next-BENCH-CARD.md).
> Layout: TP2 × PP2 + EP on four cards (`fn-quad-3090`), MTP K=3 probabilistic, FP8 KV; thinking-on effort `medium` (the server default); boot 211 s; KV pool 924,993 tokens.

## 1. Single-stream decode ladder

Same frozen ladder driver as the 3.0.1 card: T=0, a needle prompt at depth, every send a cold prefill, ≤ 256 tokens
with thinking on and the no-think answer-plus-summary task with thinking off (natural stop, ≤ 1,024 tokens); 3 sends
per cell, each cell the median of the valid sends. tok/s = tokens after the first streamed event / time from the first
to the last event. With MTP, read the median event interval (ITL) together with tokens per event. The 3.0.2 run measured
the 4K and 131K rows; the 32K and 259K rows of the 3.0.1 card were not re-run.

| prompt tokens | thinking | 3.0.2 tok/s | 3.0.1 tok/s | change | ITL ms | tokens/event | TTFT s | valid |
|---|---|---|---|---|---|---|---|---|
| 4,046 | on | 172.4 | 177.5 | −3 % | 17.58 | 3.02 | 0.66 | 3/3 |
| 130,386 | on | 188.7 | 182.9 | +3 % | 17.94 | 3.28 | 18.62 | 3/3 |
| 4,105 | off | 128.8 | 134.9 | −4 % | 17.52 | 2.25 | 0.69 | 3/3 |
| 130,445 | off | 134.5 | 125.2 | +7 % | 17.96 | 2.39 | 18.66 | 3/3 |

## 2. Exact 262K

Exactly 261,888 prompt tokens + 128 out, both thinking modes, 3 sends with one cache salt (send 1 cold). Decode, ITL
and tokens per event are the median of the 3 sends; TTFT is send 1's. The sends column gives the spread of the three
sends: MTP acceptance moves decode by several percent between sends and between runs, so a difference to 3.0.1 smaller
than that spread is not resolved by these runs.
In this run send 2 was served cold as well. Thinking on: send 3 reused 135,168 tokens; thinking off: send 3 reused 135,168 tokens.

| prompt tokens | thinking | 3.0.2 tok/s (median) | 3.0.2 sends, tok/s | 3.0.1 tok/s | change | ITL ms | tokens/event | cold TTFT s (3.0.2 / 3.0.1) |
|---|---|---|---|---|---|---|---|---|
| 261,888 | on | 206.7 | 183.7–212.6 | 197.2 | +5 % | 18.04 | 3.56 | 38.6 / 38.4 |
| 261,888 | off | 152.9 | 144.2–157.3 | 148.4 | +3 % | 18.23 | 2.67 | 38.5 / 38.2 |

## 3. Cold prefill

| prompt tokens (3.0.2 / 3.0.1) | 3.0.2 prefill | 3.0.1 prefill | change | 3.0.2 TTFT s |
|---|---|---|---|---|
| 10,157 / 10,187 | 6,585 tok/s | 6,628 tok/s | −1 % | 1.54 |
| 101,845 / 101,875 | 6,994 tok/s | 7,064 tok/s | −1 % | 14.56 |
| 261,888 / 261,888 | 6,782 tok/s | 6,803 tok/s | 0 % | 38.6 |

One cold send per depth (8 tokens out at 10K and 100K). The 3.0.2 262K row is the cold thinking-on send of table 2:
prompt tokens / time to first token. The 3.0.1 262K row is the 2026-11-01 run's cold send, as the 3.0.1 card notes, so
its time to first token can differ slightly from that card's table 2.

## 4. Agents at once

Same method as the 3.0.1 card: eight 16K conversations, then eight simultaneous 512-token follow-ups, temperature 0.
**Thinking on, reasoning effort medium** (the server default), each request setting medium. A 512-token follow-up may
end inside the reasoning, and reasoning tokens count, so these totals measure decoded tokens, not finished answers.
Total tok/s = tokens decoded while all eight stream / that overlap window. The thinking-off 8-agent run was not repeated
for 3.0.2; the [3.0.1 card](../2026-11-02/Flash-Next-BENCH-CARD.md) has it.

| layout | 3.0.2 total tok/s, thinking on | 3.0.1 | change | overlap window s |
|---|---|---|---|---|
| Four cards | 654 | 677 | −3 % | 5.51 |

## 5. Quality A/B against 3.0.1 (same session)

One session on the same host: 3.0.1, then 3.0.2, then 3.0.1 again, each a fresh boot of `fn-quad-3090`. GSM8K is the
full test set (1,319 problems, 5-shot, thinking off, temperature 0, 512 tokens). The needle cells are nine 32K prompts
with a follow-up question for a needle placed in the context.

| measurement | 3.0.1 (first run) | 3.0.2 | 3.0.1 (second run) |
|---|---|---|---|
| GSM8K, correct of 1,319 | 1,276 (96.7 %) | 1,281 (97.1 %) | 1,275 (96.7 %) |
| 32K needle prompts answered, of 9 | 6 | 4 | 6 |

Every needle miss on every run is a refusal: the needle is phrased as a secret passphrase, and the model declined to
give it. 3.0.2 refused two more prompts than 3.0.1; every answer given was correct, on every run.

## 6. Not re-run

GSM8K-200 (thinking on), sessions held at once, images and the needle panel were not re-run for 3.0.2; the [3.0.1 card](../2026-11-02/Flash-Next-BENCH-CARD.md) and the [3.0.0 card](../2026-10-20/Flash-Next-BENCH-CARD.md) have them.
