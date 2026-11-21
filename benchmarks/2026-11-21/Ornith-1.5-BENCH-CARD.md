# Ornith 1.5 — Merlin 3.0.2.2 bench card, 2026-11-21

> **Provenance.** Maintainer measurements on the reference host: 4 x RTX 3090 at 220 W with P2P enabled (aikitoria's open-gpu-kernel-modules P2P driver) and Zagitta's gpu-unlock, PCIe Gen4, no NVLink, driver 610.43.02. Streams are client-timed; drivers, raw streams and boot logs are not published.
> Engine: Merlin 3.0.2.2 on vLLM 0.33.0, measured on 2026-11-21 (UTC) on an install of this release's engine code.
> Why this card: the 3.0.2.2 spot check of this lane (one 4K thinking-on decode row and one 100K prefill row against the 3.0.2.1 card) fell outside the 4K decode row's band of ±3.6 %, so every section below was run again on this release's engine code.
>
> Checkpoint: [halt95/Ornith-1.5-122B-A10B-W4A16-Merlin](https://huggingface.co/halt95/Ornith-1.5-122B-A10B-W4A16-Merlin). The 3.0.2.1 column is the [2026-11-12 card](../2026-11-12/Ornith-1.5-BENCH-CARD.md).
> Layout: four cards (`ornith-quad-3090`), MTP, FP8 KV; thinking on (this template has no effort levels); boot 306 s; KV pool 894,618 tokens.

## 1. Single-stream decode ladder

Same frozen ladder driver as the 3.0.2.1 card: T=0, a needle prompt at depth, every send a cold prefill, ≤ 256 tokens
with thinking on and the no-think answer-plus-summary task with thinking off (natural stop, ≤ 1,024 tokens); 3 sends
per cell, each cell the median of the valid sends. tok/s = tokens after the first streamed event / time from the first
to the last event. With MTP, read the median event interval (ITL) together with tokens per event.

| prompt tokens | thinking | 3.0.2.2 tok/s | 3.0.2.1 tok/s | change | ITL ms | tokens/event | TTFT s | valid |
|---|---|---|---|---|---|---|---|---|
| 4,046 | on | 152.1 | 148.7 | +2 % | 19.00 | 2.85 | 0.68 | 3/3 |
| 32,566 | on | 134.5 | 136.2 | −1 % | 19.51 | 2.58 | 5.36 | 3/3 |
| 130,386 | on | 106.4 | 122.3 | −13 % | 20.91 | 2.16 | 36.16 | 3/3 |
| 259,557 | on | 101.5 | 97.3 | +4 % | 22.34 | 2.23 | 112.28 | 3/3 |
| 4,105 | off | 110.5 | 109.4 | +1 % | 18.93 | 2.08 | 0.70 | 3/3 |
| 32,625 | off | 106.6 | 106.9 | 0 % | 19.49 | 2.07 | 5.38 | 3/3 |
| 130,445 | off | 103.0 | 102.2 | +1 % | 20.83 | 2.12 | 36.36 | 3/3 |
| 259,616 | off | 96.0 | 92.7 | +4 % | 22.29 | 2.12 | 112.20 | 3/3 |

The 131K thinking-on cell is the median of sends of 97.9, 106.4 and 124.5 tok/s. Two of the three replies stopped by
themselves at 186 and 188 tokens, with fewer tokens per event (2.02 and 2.16 against 2.53 for the reply that ran to 256
tokens); the median event interval is level with 3.0.2.1 (20.91 against 20.77 ms). On the same engine path, six other
131K texts gave 117.5 tok/s (115.9–121.0). With MTP the number of tokens produced per step follows the text generated,
so at 131K the rate moves with the replies' text.

## 2. Exact 262K

Exactly 261,888 prompt tokens with a 128-token output cap, both thinking modes, one stream, 3 sends with one cache salt
(send 1 cold); each send uses a different prompt text. Decode, ITL and tokens per event are the median of the 3 sends;
TTFT is send 1's. The sends column gives the range of the three sends.

With speculative decoding (MTP), the number of tokens produced per decode step depends on the text being generated, so
over 128 tokens the rate varies from send to send with the text; the range is given beside each median.

| prompt tokens | thinking | 3.0.2.2 tok/s (median) | 3.0.2.2 sends, tok/s | 3.0.2.1 tok/s | change | ITL ms | tokens/event | cold TTFT s |
|---|---|---|---|---|---|---|---|---|
| 261,888 | on | 105.8 | 101.7–112.2 | 107.0 | −1 % | 22.33 | 2.29 | 113.8 |
| 261,888 | off | 93.6 | 92.5–97.2 | 98.2 | −5 % | 22.25 | 2.03 | 113.7 |

Thinking on: sends 2 and 3 were served cold as well (TTFT 113.7 s on every send); thinking off: send 3 reused the
cached prefix (TTFT 76.7 s). The thinking-off median is just below the 3.0.2.1 sends' range (95.1–99.4 tok/s); the
event interval is level (22.25 against 22.29 ms) and the tokens per event are lower (2.03 against 2.13).

## 3. Cold prefill

| prompt tokens | 3.0.2.2 prefill | 3.0.2.1 prefill | change | TTFT s |
|---|---|---|---|---|
| 10,157 | 6,484 tok/s | 6,411 tok/s | +1 % | 1.57 |
| 101,845 | 4,101 tok/s | 4,059 tok/s | +1 % | 24.84 |
| 261,888 | 2,302 tok/s | 2,284 tok/s | +1 % | 113.8 |

One cold send per depth (8 tokens out at 10K and 100K). The 262K row is the cold thinking-on send of table 2:
prompt tokens / time to first token.

## 4. Agents at once

Eight 16K conversations, then eight simultaneous 512-token follow-ups, temperature 0. Thinking on (this template has no effort levels; the driver's `medium` has no effect). A 512-token follow-up may end inside the reasoning, and
reasoning tokens count, so these totals measure decoded tokens, not finished answers. Total tok/s = tokens decoded
while all eight stream / that overlap window. Each row is one run of 8.

| layout | thinking | 3.0.2.2 total tok/s | 3.0.2.1 tok/s | change | overlap window s |
|---|---|---|---|---|---|
| Four cards | on | 522 | 516 | +1 % | 6.96 |
| Four cards | off | 481 | 484 | −1 % | 8.16 |

The 3.0.2.1 figures in this table are from that release's previous build (the 3.0.2.1 wheel as built on 2026-11-10 and
2026-11-11), as the coverage table of the [3.0.2.1 card](../2026-11-12/Ornith-1.5-BENCH-CARD.md) records.

The same runs also measured 2 and 4 agents: thinking on 184 tok/s (window 5.42 s) and 313 tok/s (6.20 s); thinking off
171 tok/s (5.77 s) and 286 tok/s (6.85 s).

## 5. GSM8K-200

The 3.0.1 card's set and method (200 problems, 5-shot, temperature 0) on four cards: **195/200** thinking on, none cut
at the 8,192-token cap (3.0.2.1: 195/200). **192/200** thinking off, 2 cut at the token limit (3.0.2.1: 191/200).

## 6. Coverage of this card

| figure | status on 3.0.2.2 |
|---|---|
| Decode ladder, exact 262K, cold prefill | Measured 2026-11-21 |
| 8 agents and GSM8K-200, thinking on and off | Measured 2026-11-21 |
| KV pool | Measured 2026-11-21: 894,618 tokens, as on 3.0.2.1 |
| Three ~262K prompts submitted together (not a residency test) | Not re-run; [3.0.0 card](../../models/ornith1.5/benchmarks/2026-10-21/BENCH-CARD.md) section 4, 2026-10-21 |
| Images | Not re-run; [3.0.0 card](../../models/ornith1.5/benchmarks/2026-10-21/BENCH-CARD.md) section 5, 2026-10-21 |
| Needle panel | Spot-checked 2026-11-21 on 3.0.2.2, before the MTP layer-index guard: 6/6 |

Repeatability at temperature 0: on this layout Ornith's output repeats within one install, but not across installs or
rebuilds of the engine's tuning cache (see the [3.0.2.1 card](../2026-11-12/Ornith-1.5-BENCH-CARD.md) section 6), so the tokens
produced per event, and with them the decode rate, can differ between two installs of the same code.

Other CPU load on the host: not recorded on these rows.
