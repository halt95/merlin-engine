# Qwen3.8-Flash-Next — Merlin 3.0.2.2 bench card, 2026-11-21

> **Provenance.** Maintainer measurements on the reference host: 4 x RTX 3090 at 220 W with P2P enabled (aikitoria's open-gpu-kernel-modules P2P driver) and Zagitta's gpu-unlock, PCIe Gen4, no NVLink, driver 610.43.02. Streams are client-timed; drivers, raw streams and boot logs are not published.
> Engine: Merlin 3.0.2.2 on vLLM 0.33.0, measured on 2026-11-21 (UTC) on a tree that carries this release's engine code; the only later non-comment engine change is the MTP layer-index guard on pipeline-parallel layouts, which affects only Ornith 1.5 (re-measured on it).
> What this card covers: the 4 x 20 GB row only. Its spot check read 154.0 tok/s at 4K thinking on against 147.5 tok/s on the 3.0.2.1 card, outside that row's band of ±3 %, so the decode ladder, the exact 262K row and the agents row were run again (the 3.0.2.1 20 GB rows were measured on the previous build of that release, before its prefill runs change). The 4 x 24 GB row was inside its bands and keeps its [3.0.2.1 card](../2026-11-12/Flash-Next-BENCH-CARD.md) figures (section 6).
>
> Checkpoint: [halt95/Qwen3.8-Flash-Next-W4A16-Merlin](https://huggingface.co/halt95/Qwen3.8-Flash-Next-W4A16-Merlin). The 3.0.2.1 column is the [2026-11-12 card](../2026-11-12/Flash-Next-BENCH-CARD.md).
> 20 GB row: `fn-quad-20g-sm86` on 4 x RTX 3090 emulating 20 GB cards (4,096 MiB held back per card), cold experts on layers 11, 12, 19, 20 and 37 (the default), MTP K=3 probabilistic, FP8 KV; thinking-on effort `medium` (the server default); boot 196 s; KV pool 265,888 tokens (1.01× at 262K). Not measured on real 20 GB cards.

## 1. Single-stream decode ladder

Same frozen ladder driver as the 3.0.2.1 card: T=0, a needle prompt at depth, every send a cold prefill, ≤ 256 tokens
with thinking on and the no-think answer-plus-summary task with thinking off (natural stop, ≤ 1,024 tokens); 3 sends
per cell, each cell the median of the valid sends. tok/s = tokens after the first streamed event / time from the first
to the last event. With MTP, read the median event interval (ITL) together with tokens per event.

### 4 x 20 GB (emulated)

The 20 GB figures are not comparable with the 24 GB card.

| prompt tokens | thinking | 3.0.2.2 tok/s | 3.0.2.1 tok/s | change | ITL ms | tokens/event | TTFT s | valid |
|---|---|---|---|---|---|---|---|---|
| 4,046 | on | 153.2 | 147.5 | +4 % | 19.56 | 3.02 | 0.98 | 3/3 |
| 130,386 | on | 159.2 | 158.4 | 0 % | 20.61 | 3.24 | 26.52 | 3/3 |
| 4,105 | off | 119.3 | 119.3 | 0 % | 18.86 | 2.25 | 1.02 | 3/3 |
| 130,445 | off | 125.6 | 124.8 | +1 % | 19.10 | 2.39 | 26.52 | 3/3 |

At 4K thinking on the sends were 146.4–160.5 tok/s, at the same tokens per event as on 3.0.2.1 (3.02) and a median
event interval of 19.56 against 19.96 ms.

## 2. Exact 262K

Exactly 261,888 prompt tokens with a 128-token output cap, both thinking modes, one stream, 3 sends with one cache salt
(send 1 cold); each send uses a different prompt text. Decode, ITL and tokens per event are the median of the 3 sends;
TTFT is send 1's. The sends column gives the range of the three sends.

With speculative decoding (MTP), the number of tokens produced per decode step depends on the text being generated, so
over 128 tokens the rate varies from send to send with the text; the range is given beside each median.

### 4 x 20 GB (emulated)

| prompt tokens | thinking | 3.0.2.2 tok/s (median) | 3.0.2.2 sends, tok/s | 3.0.2.1 tok/s | change | ITL ms | tokens/event | cold TTFT s |
|---|---|---|---|---|---|---|---|---|
| 261,888 | on | 170.8 | 163.4–176.8 | 165.1 | +3 % | 20.34 | 3.28 | 54.1 |
| 261,888 | off | 138.6 | 132.0–166.0 | 122.5 | +13 % | 19.11 | 2.61 | 54.2 |

No send reused the prefix in either thinking mode. Thinking off is +13 % with a median event interval of 19.11 against
19.34 ms and 2.61 tokens per event against 2.33: the difference follows the tokens produced per step, which depend on
the text sent (the three sends span 132.0–166.0 tok/s).

## 3. Cold prefill

### 4 x 20 GB (emulated)

| prompt tokens | 3.0.2.2 prefill | 3.0.2.1 prefill | change | TTFT s |
|---|---|---|---|---|
| 10,157 | 4,639 tok/s | 4,434 tok/s | +5 % | 2.19 |
| 101,845 | 4,932 tok/s | 4,656 tok/s | +6 % | 20.65 |
| 261,888 | 4,838 tok/s | 4,555 tok/s | +6 % | 54.1 |

One cold send per depth (8 tokens out at 10K and 100K). The 262K row is the cold thinking-on send of table 2:
prompt tokens / time to first token. The 3.0.2.1 20 GB rows were measured on the previous build of that release,
which did not have its prefill runs change (on by default on this row); this card is the first measurement of the
20 GB row with it.

## 4. Agents at once

The 4 x 20 GB row has no 8-agent figure, as on the 3.0.2.1 card: its 265,888-token pool held six of the eight 16K
follow-ups on this run (two lost their cached prefix and had to prefill again, TTFT 19.0 s each), so the "all eight streaming"
window cannot be measured.

## 5. GSM8K

Not run on the 20 GB row, as on the 3.0.2.1 card.

## 6. Coverage of this card

| figure | status on 3.0.2.2 |
|---|---|
| Decode ladder, exact 262K, cold prefill, 4 x 20 GB | Measured 2026-11-21 |
| 8 agents, 4 x 20 GB | Not applicable: the pool held six of the eight follow-ups on this run |
| KV pool and needle panel, 4 x 20 GB | Spot-checked 2026-11-21: pool 265,888 tokens as on 3.0.2.1; needle panel 6/6 |
| 4 x 24 GB: every section | Carried from the [3.0.2.1 card](../2026-11-12/Flash-Next-BENCH-CARD.md). Spot check on 3.0.2.2: 4K thinking-on decode 171.6 tok/s against 174.0 (band ±3 %), 100K prefill 6,999 against 7,036 tok/s (±3 %), KV pool 924,993 tokens as on 3.0.2.1, needle panel 6/6 |
| 8 MP images, sessions, quality panel | Not re-run; see the [3.0.2.1 card](../2026-11-12/Flash-Next-BENCH-CARD.md) section 6 |

Other CPU load on the host: not recorded on these rows.
