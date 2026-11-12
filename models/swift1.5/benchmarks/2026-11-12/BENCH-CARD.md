# Swift1.5 — Merlin 3.0.2.1 bench card, 2026-11-12

> **Provenance.** Maintainer measurements on the reference host: 4 x RTX 3090 at 220 W with P2P enabled (aikitoria's open-gpu-kernel-modules P2P driver) and Zagitta's gpu-unlock, PCIe Gen4, no NVLink, driver 610.43.02. Streams are client-timed; drivers, raw streams and boot logs are not published.
> Engine: Merlin 3.0.2.1 on vLLM 0.33.0. Every 3.0.2.1 row on this card (sections 1–5) was measured on the published 3.0.2.1 wheel `0.33.0+merlin.c56eeed392` on 2026-11-11. Served by `merlin serve` with the shipped defaults, on three separate fresh boots: decode, exact-262K and prefill; thinking-on agents and GSM8K-200; thinking-off agents and GSM8K-200. Measured 2026-11-11 (UTC). Decode comparisons use medians; cold prefill is one request per depth and each 8-agent figure is one overlap run.
>
> Checkpoint: [halt95/Swift1.5-Qwen3.8-Flash-Next-W4A16-Merlin](https://huggingface.co/halt95/Swift1.5-Qwen3.8-Flash-Next-W4A16-Merlin). The 3.0.2 column is the [2026-11-05 card](../2026-11-05/BENCH-CARD.md).
> Layout: TP2 × PP2 + EP on four cards (`fn-swift1.5-quad-3090`), MTP K=3 probabilistic, FP8 KV; thinking-on effort `medium` (the server default); boot 432 s (first start of this configuration: graphs compiled, none loaded from cache); KV pool 924,993 tokens.

## 1. Single-stream decode ladder

Same frozen ladder driver as the 3.0.2 card: T=0, a needle prompt at depth, every send a cold prefill, ≤ 256 tokens
with thinking on and the no-think answer-plus-summary task with thinking off (natural stop, ≤ 1,024 tokens); 3 sends
per cell, each cell the median of the valid sends. tok/s = tokens after the first streamed event / time from the first
to the last event. With MTP, read the median event interval (ITL) together with tokens per event.

| prompt tokens | thinking | 3.0.2.1 tok/s | 3.0.2 tok/s | change | ITL ms | tokens/event | TTFT s | valid |
|---|---|---|---|---|---|---|---|---|
| 4,046 | on | 179.8 | 180.8 | −1 % | 17.83 | 3.15 | 0.66 | 3/3 |
| 32,566 | on | 179.2 | 181.0 | −1 % | 17.93 | 3.16 | 4.66 | 3/3 |
| 130,386 | on | 185.0 | 186.9 | −1 % | 18.11 | 3.25 | 18.75 | 3/3 |
| 259,557 | on | 190.5 | 191.8 | −1 % | 18.36 | 3.42 | 38.36 | 3/3 |
| 4,105 | off | 130.5 | 130.7 | 0 % | 17.77 | 2.31 | 0.69 | 3/3 |
| 32,625 | off | 136.0 | 136.1 | 0 % | 17.90 | 2.41 | 4.65 | 3/3 |
| 130,445 | off | 128.7 | 130.0 | −1 % | 18.16 | 2.32 | 18.85 | 3/3 |
| 259,616 | off | 130.0 | 129.8 | 0 % | 18.40 | 2.38 | 38.38 | 3/3 |

## 2. Exact 262K

Exactly 261,888 prompt tokens with a 128-token output cap, both thinking modes, one stream, 3 sends with one cache salt
(send 1 cold); each send uses a different prompt text. Decode, ITL and tokens per event are the median of the 3 sends;
TTFT is send 1's. The sends column gives the range of the three sends.

With speculative decoding (MTP), the number of tokens produced per decode step depends on the text being generated, so
over 128 tokens the rate varies from send to send with the text; the range is given beside each median. Measured on
this release on the 27B two-card layout with thinking off: the same prompt text sent 100 times, across four server
starts, gave identical output every time and 100.0–103.5 tok/s, at 2.783 tokens per step on every send; three other
texts gave 106.9–110.3 tok/s; the time per decode step was flat, 27.5–27.9 ms at the client and 28.3–28.6 ms in the
server. On that row the previous development build and this release gave identical output and the same rate (a
paired difference of +0.3 to +0.5 %, inside the difference of about 1 % between two starts of the same build). This
was measured on that one layout and thinking mode only.

| prompt tokens | thinking | 3.0.2.1 tok/s (median) | 3.0.2.1 sends, tok/s | 3.0.2 tok/s | change | ITL ms | tokens/event | cold TTFT s |
|---|---|---|---|---|---|---|---|---|
| 261,888 | on | 182.1 | 173.4–192.8 | 186.1 | −2 % | 18.26 | 3.20 | 38.6 |
| 261,888 | off | 160.1 | 127.5–160.7 | 143.8 | +11 % | 18.35 | 2.78 | 38.7 |

In this run send 2 was served cold as well. Thinking on: send 3 reused 139,200 tokens; thinking off: send 3 reused 139,200 tokens.

Thinking off at 262K is +11 % against 3.0.2 (sends 127.5–160.7 here, 139.3–160.7 on 3.0.2). Median event interval 18.35 ms against 18.23 ms; tokens per event 2.78 against 2.51. The step time is within 1 %, so the difference follows the tokens produced per step, which depend on the text sent.

## 3. Cold prefill

| prompt tokens | 3.0.2.1 prefill | 3.0.2 prefill | change | TTFT s |
|---|---|---|---|---|
| 10,157 | 6,604 tok/s | 6,569 tok/s | +1 % | 1.54 |
| 101,845 | 6,964 tok/s | 6,932 tok/s | 0 % | 14.62 |
| 261,888 | 6,793 tok/s | 6,762 tok/s | 0 % | 38.6 |

One cold send per depth (8 tokens out at 10K and 100K). The 262K row is the cold thinking-on send of table 2:
prompt tokens / time to first token.

## 4. Agents at once

Eight 16K conversations, then eight simultaneous 512-token follow-ups, temperature 0. Thinking on is reasoning effort
medium (the server default), each request setting medium. A 512-token follow-up may end inside the reasoning, and
reasoning tokens count, so these totals measure decoded tokens, not finished answers. Total tok/s = tokens decoded
while all eight stream / that overlap window. Each row is one run of 8; a window under 5 s is short, so read that
total loosely, as on the 3.0.2 card.

| layout | thinking | 3.0.2.1 total tok/s | 3.0.2 tok/s | change | overlap window s |
|---|---|---|---|---|---|
| Four cards | on | 688 | 659 | +4 % | 5.56 |
| Four cards | off | 661 | 650 (the 3.0.2 thinking-off figure is the 3.0.1 run the 3.0.2 card carried over) | +2 % | 5.91 |

Each row is one run of eight. The same thinking-on row on Flash-Next spread 5 % across six repeated runs (661–697 tok/s), so the change column here is not a gain or a loss against 3.0.2.

## 5. GSM8K-200

The 3.0.1 card's set and method (200 problems, 5-shot, temperature 0), with thinking on at effort medium on four cards and thinking off: **199/200** thinking on, none cut at the 8,192-token cap; **198/200** thinking off, 1 cut at the 512-token limit. The 3.0.2 card scored 199/200 with thinking on; the thinking-off score carried on the 3.0.2 card is 197/200 (3.0.0 run).

## 6. Coverage of this card

| figure | status on 3.0.2.1 |
|---|---|
| Decode ladder, exact 262K, cold prefill | Measured 2026-11-11, published wheel `0.33.0+merlin.c56eeed392` |
| 8 agents, thinking on | Measured 2026-11-11, published wheel `0.33.0+merlin.c56eeed392` |
| 8 agents, thinking off | Measured 2026-11-11, published wheel `0.33.0+merlin.c56eeed392` |
| GSM8K-200, thinking on | Measured 2026-11-11, published wheel `0.33.0+merlin.c56eeed392` |
| GSM8K-200, thinking off | Measured 2026-11-11, published wheel `0.33.0+merlin.c56eeed392` |
| Sessions held at once | Not re-run; [3.0.0 card](../2026-10-20/BENCH-CARD.md) section 4, 2026-10-20 |
| 8 MP images | Not re-run; [3.0.0 card](../2026-10-20/BENCH-CARD.md) section 5, 2026-10-20 |
| Needle panel | Not re-run; [3.0.0 card](../2026-10-20/BENCH-CARD.md) section 6, pre-release build, 2026-10-17 |

Other CPU load on the host: none during any row.
