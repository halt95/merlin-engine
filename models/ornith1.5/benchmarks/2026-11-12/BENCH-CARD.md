# Ornith 1.5 — Merlin 3.0.2.1 bench card, 2026-11-12

> **Provenance.** Maintainer measurements on the reference host: 4 x RTX 3090 at 220 W with P2P enabled (aikitoria's open-gpu-kernel-modules P2P driver) and Zagitta's gpu-unlock, PCIe Gen4, no NVLink, driver 610.43.02. Streams are client-timed; drivers, raw streams and boot logs are not published.
> Engine: Merlin 3.0.2.1 on vLLM 0.33.0. Every 3.0.2.1 row on this card ran on the previous build (the 3.0.2.1 wheel `0.33.0+merlin.c56eeed392` as built on 2026-11-10, before the prefill runs change); the published wheel differs from it in three engine files that carry the new prefill schedule, which is off on this model. Served by `merlin serve` with the shipped defaults, on three separate fresh boots: decode, exact-262K and prefill; thinking-on agents and GSM8K-200; thinking-off agents and GSM8K-200. Measured 2026-11-10 and 2026-11-11 (UTC). Decode comparisons use medians; cold prefill is one request per depth and each 8-agent figure is one overlap run.
>
> Checkpoint: [halt95/Ornith-1.5-122B-A10B-W4A16-Merlin](https://huggingface.co/halt95/Ornith-1.5-122B-A10B-W4A16-Merlin). The 3.0.2 column is the [2026-11-04 card](../2026-11-04/BENCH-CARD.md).
> Layout: four cards (`ornith-quad-3090`), MTP, FP8 KV; thinking on (this template has no effort levels); boot 301 s (first start of this configuration: graphs compiled, none loaded from cache); KV pool 894,618 tokens.

## 1. Single-stream decode ladder

Same frozen ladder driver as the 3.0.2 card: T=0, a needle prompt at depth, every send a cold prefill, ≤ 256 tokens
with thinking on and the no-think answer-plus-summary task with thinking off (natural stop, ≤ 1,024 tokens); 3 sends
per cell, each cell the median of the valid sends. tok/s = tokens after the first streamed event / time from the first
to the last event. With MTP, read the median event interval (ITL) together with tokens per event.

| prompt tokens | thinking | 3.0.2.1 tok/s | 3.0.2 tok/s | change | ITL ms | tokens/event | TTFT s | valid |
|---|---|---|---|---|---|---|---|---|
| 4,046 | on | 148.7 | 145.0 | +3 % | 18.94 | 2.78 | 0.69 | 3/3 |
| 32,566 | on | 136.2 | - | - | 19.40 | 2.61 | 5.39 | 3/3 |
| 130,386 | on | 122.3 | 121.7 | 0 % | 20.77 | 2.49 | 36.48 | 3/3 |
| 259,557 | on | 97.3 | - | - | 22.38 | 2.15 | 113.21 | 3/3 |
| 4,105 | off | 109.4 | 106.0 | +3 % | 18.93 | 2.06 | 0.70 | 3/3 |
| 32,625 | off | 106.9 | - | - | 19.49 | 2.07 | 5.41 | 3/3 |
| 130,445 | off | 102.2 | 97.7 | +5 % | 20.82 | 2.10 | 36.61 | 3/3 |
| 259,616 | off | 92.7 | - | - | 22.43 | 2.07 | 113.22 | 3/3 |

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
| 261,888 | on | 107.0 | 96.4–113.1 | 111.0 | −4 % | 22.39 | 2.29 | 114.7 |
| 261,888 | off | 98.2 | 95.1–99.4 | 92.8 | +6 % | 22.29 | 2.13 | 114.4 |

In this run send 2 was served cold as well. Thinking on: send 3 reused 133,632 tokens; thinking off: send 3 reused 133,632 tokens.

Thinking off at 262K is +6 % against 3.0.2 (sends 95.1–99.4 here, 88.3–97.1 on 3.0.2). Median event interval 22.29 ms against 22.48 ms; tokens per event 2.13 against 2.00. The step time is within 1 %, so the difference follows the tokens produced per step, which depend on the text sent.

## 3. Cold prefill

| prompt tokens | 3.0.2.1 prefill | 3.0.2 prefill | change | TTFT s |
|---|---|---|---|---|
| 10,157 | 6,411 tok/s | 6,385 tok/s | 0 % | 1.58 |
| 101,845 | 4,059 tok/s | 4,068 tok/s | 0 % | 25.09 |
| 261,888 | 2,284 tok/s | 2,288 tok/s | 0 % | 114.7 |

One cold send per depth (8 tokens out at 10K and 100K). The 262K row is the cold thinking-on send of table 2:
prompt tokens / time to first token.

## 4. Agents at once

Eight 16K conversations, then eight simultaneous 512-token follow-ups, temperature 0. Thinking on (this template has no effort levels; the driver's `medium` has no effect). A 512-token follow-up may end inside the reasoning, and
reasoning tokens count, so these totals measure decoded tokens, not finished answers. Total tok/s = tokens decoded
while all eight stream / that overlap window. Each row is one run of 8; a window under 5 s is short, so read that
total loosely, as on the 3.0.2 card.

| layout | thinking | 3.0.2.1 total tok/s | 3.0.2 tok/s | change | overlap window s |
|---|---|---|---|---|---|
| Four cards | on | 516 | 528 (the 3.0.2 thinking-on figure is the 3.0.1 run the 3.0.2 card carried over) | −2 % | 7.08 |
| Four cards | off | 484 | 486 (the 3.0.2 thinking-off figure is the 3.0.1 run the 3.0.2 card carried over) | 0 % | 7.48 |

## 5. GSM8K-200

The 3.0.1 card's set and method (200 problems, 5-shot, temperature 0), with thinking on (this template has no effort levels) and thinking off on four cards: **195/200** thinking on, none cut at the 8,192-token cap; **191/200** thinking off, none cut at the 512-token limit. The 3.0.2 card carried 196/200 (thinking on, 3.0.1 run) and 190/200 (thinking off, 3.0.0 run).

## 6. Coverage of this card

| figure | status on 3.0.2.1 |
|---|---|
| Decode ladder, exact 262K, cold prefill | Measured 2026-11-10, previous build (not re-measured on the published wheel) |
| 8 agents, thinking on and off | Measured 2026-11-11, previous build (not re-measured on the published wheel) |
| GSM8K-200, thinking on and off | Measured 2026-11-11, previous build (not re-measured on the published wheel) |
| Three ~262K prompts submitted together (not a residency test) | Not re-run; [3.0.0 card](../2026-10-21/BENCH-CARD.md) section 4, 2026-10-21 |
| Images | Not re-run; [3.0.0 card](../2026-10-21/BENCH-CARD.md) section 5, 2026-10-21 |
| Needle panel | Not re-run; [3.0.0 card](../2026-10-21/BENCH-CARD.md) section 6, 2026-10-21 |

Repeatability at temperature 0: on this layout Ornith's output repeats within one install, but not across installs or rebuilds of the engine's tuning cache. Five cache states gave five different outputs on 2026-11-11, on the previous and the published code alike; for the first generated token the two leading candidates are close, so small numeric differences between tuning states can pick either. Every figure on this card was measured on one such state (tuned 2026-11-10).

Other CPU load on the host: none during any row.
