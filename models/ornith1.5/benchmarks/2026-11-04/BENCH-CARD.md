# Ornith 1.5 — Merlin 3.0.2 bench card, 2026-11-04

> **Provenance.** Maintainer measurements on the reference host. Measured on 4 x RTX 3090 at 220 W with P2P enabled (aikitoria's open-gpu-kernel-modules P2P driver) and Zagitta's gpu-unlock. PCIe Gen4, no NVLink, driver 610.43.02.
> Engine: Merlin 3.0.2 on vLLM 0.33.0 (wheel `0.33.0+merlin.665f43d2cf`), on the release candidate; the changes after it
> touch only Flash-Next code paths. Served by `merlin serve` with the shipped defaults, one fresh boot per layout;
> measured 2026-11-04 (UTC).
> Streams are client-timed; drivers, raw streams and boot logs are not published.
>
> Checkpoint: [halt95/Ornith-1.5-122B-A10B-W4A16-Merlin](https://huggingface.co/halt95/Ornith-1.5-122B-A10B-W4A16-Merlin). The 3.0.1 column is the [2026-11-02 card](../2026-11-02/BENCH-CARD.md).
> Layout: four cards (`ornith-quad-3090`), MTP, FP8 KV; thinking on (this template has no effort levels); boot 311 s; KV pool 894,618 tokens.

## 1. Single-stream decode ladder

Same frozen ladder driver as the 3.0.1 card: T=0, a needle prompt at depth, every send a cold prefill, ≤ 256 tokens
with thinking on and the no-think answer-plus-summary task with thinking off (natural stop, ≤ 1,024 tokens); 3 sends
per cell, each cell the median of the valid sends. tok/s = tokens after the first streamed event / time from the first
to the last event. With MTP, read the median event interval (ITL) together with tokens per event. The 3.0.2 run measured
the 4K and 131K rows; the 32K and 259K rows of the 3.0.1 card were not re-run.

| prompt tokens | thinking | 3.0.2 tok/s | 3.0.1 tok/s | change | ITL ms | tokens/event | TTFT s | valid |
|---|---|---|---|---|---|---|---|---|
| 4,046 | on | 145.0 | 145.6 | 0 % | 19.01 | 2.72 | 0.69 | 3/3 |
| 130,386 | on | 121.7 | 115.0 | +6 % | 20.83 | 2.49 | 36.44 | 3/3 |
| 4,105 | off | 106.0 | 109.3 | −3 % | 18.96 | 2.00 | 0.70 | 3/3 |
| 130,445 | off | 97.7 | 95.2 | +3 % | 20.90 | 2.02 | 36.54 | 3/3 |

## 2. Exact 262K

Exactly 261,888 prompt tokens + 128 out, both thinking modes, 3 sends with one cache salt (send 1 cold). Decode, ITL
and tokens per event are the median of the 3 sends; TTFT is send 1's. The sends column gives the spread of the three
sends: MTP acceptance moves decode by several percent between sends and between runs, so a difference to 3.0.1 smaller
than that spread is not resolved by these runs.
In this run send 2 was served cold as well. Thinking on: send 3 reused 133,632 tokens; thinking off: send 3 reused 133,632 tokens.

| prompt tokens | thinking | 3.0.2 tok/s (median) | 3.0.2 sends, tok/s | 3.0.1 tok/s | change | ITL ms | tokens/event | cold TTFT s (3.0.2 / 3.0.1) |
|---|---|---|---|---|---|---|---|---|
| 261,888 | on | 111.0 | 109.1–130.0 | 104.6 | +6 % | 22.43 | 2.42 | 114.5 / 115.8 |
| 261,888 | off | 92.8 | 88.3–97.1 | 95.9 | −3 % | 22.48 | 2.00 | 114.4 / 115.8 |

## 3. Cold prefill

| prompt tokens (3.0.2 / 3.0.1) | 3.0.2 prefill | 3.0.1 prefill | change | 3.0.2 TTFT s |
|---|---|---|---|---|
| 10,157 / 10,157 | 6,385 tok/s | 6,218 tok/s | +3 % | 1.59 |
| 101,845 / 101,845 | 4,068 tok/s | 3,994 tok/s | +2 % | 25.03 |
| 261,888 / 261,888 | 2,288 tok/s | 2,261 tok/s | +1 % | 114.5 |

One cold send per depth (8 tokens out at 10K and 100K). The 3.0.2 262K row is the cold thinking-on send of table 2:
prompt tokens / time to first token. The 3.0.1 262K row is the 2026-11-01 run's cold send, as the 3.0.1 card notes, so
its time to first token can differ slightly from that card's table 2.

## 4. Not re-run

The 8-agent runs, GSM8K-200, sessions held at once, images and the needle panel were not re-run for 3.0.2; the [3.0.1 card](../2026-11-02/BENCH-CARD.md) and the [3.0.0 card](../2026-10-21/BENCH-CARD.md) have them.
