# Qwen3.8-Flash-Next — Merlin 3.0.0 bench card, 2026-10-20

> **Provenance.** Maintainer measurements on the reference host: 4× RTX 3090 (24 GB, sm_86) at 220 W, PCIe Gen4 with P2P,
> no NVLink, driver 610.43.02. Engine: the Merlin 3.0.0 release wheel (`vllm-0.32.0+merlin`, built for sm_80/86/89), installed the way a user
> installs it with `release/install-env.sh`; each configuration served by `merlin serve` with
> the release defaults, one fresh boot per configuration (2026-10-20), except the section 6 needle panel, marked "needle panel: a pre-release engine build, 2026-10-17". Streams are client-timed; drivers, raw streams and boot logs are not published.
>
> Checkpoint: [halt95/Qwen3.8-Flash-Next-W4A16-Merlin](https://huggingface.co/halt95/Qwen3.8-Flash-Next-W4A16-Merlin). Layout: TP2 × PP2 + EP on four cards, MTP K=3 probabilistic, FP8 KV; thinking-on effort `low` (the default at measurement time); boot 231 s (compile cache carried over from an earlier boot); KV pool 924,993 tokens.
>
> **Draft head.** Measured with the BF16 draft lm_head that 3.0.0 ships, as every earlier Flash-Next release did. The INT8
> copy (`DRAFT_HEAD=int8`, 6.5–7.2 % lower ms per token at the same MTP acceptance) is a candidate default for a later release; later releases keep `bf16` as the default.

## 1. Single-stream decode ladder

Same frozen ladder driver as the Flash-Next v2.5.1 and v2.5.2 benchmarks: T=0, a needle prompt at depth, every send
a cold prefill, ≤ 256 tokens with thinking on (the served default reasoning effort) and the no-think answer-plus-summary task
with thinking off (natural stop, ≤ 1,024 tokens); one discarded warm-up send per boot, then 3 sends per cell, each cell the
median of the valid sends. tok/s = tokens after the first streamed event / time from the first to the last event. With
MTP, read the median event interval (ITL) together with tokens per event, not tok/s alone.

**262K row (new in 3.0.0).** Exactly 261,888 prompt tokens + 128 out, both thinking modes, 3 sends (sends 1 and 2 cold;
send 3 reused the prefix before the needle from the prefix cache). The table gives the median decode and send 1's TTFT.

| prompt tokens | thinking | 3.0.0 tok/s | v2.5.2 tok/s | change | ITL ms (3.0.0 / v2.5.2) | tokens/event (3.0.0 / v2.5.2) | TTFT s | valid |
|---|---|---|---|---|---|---|---|---|
| 4,076 | on | 167.7 | 162.0 | +3.5 % | 18.37 / 19.04 | 3.06 / 3.04 | 0.79 | 3/3 |
| 32,596 | on | 167.6 | 164.7 | +1.8 % | 18.40 / 18.96 | 3.03 / 3.07 | 5.36 | 3/3 |
| 130,416 | on | 170.3 | 168.0 | +1.4 % | 18.76 / 19.55 | 3.10 / 3.20 | 21.57 | 3/3 |
| 259,587 | on | 187.4 | 177.2 | +5.8 % | 18.94 / 19.60 | 3.44 / 3.41 | 44.58 | 3/3 |
| 4,105 | off | 131.0 | 121.3 | +8.0 % | 18.31 / 19.06 | 2.39 / 2.30 | 0.82 | 3/3 |
| 32,625 | off | 126.8 | 118.0 | +7.5 % | 18.43 / 18.92 | 2.33 / 2.21 | 5.39 | 3/3 |
| 130,445 | off | 121.3 | 120.3 | +0.8 % | 18.77 / 19.55 | 2.25 / 2.33 | 21.61 | 3/3 |
| 259,616 | off | 125.0 | 119.7 | +4.4 % | 18.78 / 19.60 | 2.34 / 2.33 | 44.54 | 3/3 |

## 2. Exact 262K

| prompt tokens | thinking | decode tok/s (median of 3) | ITL ms | tokens/event | cold TTFT s | needle found |
|---|---|---|---|---|---|---|
| 261,888 | on | 183.6 | 18.98 | 3.28 | 44.9 | — (see note) |
| 261,888 | off | 149.0 | 18.91 | 2.67 | 44.7 | 3/3 |

**262K needle.** Each prompt is exactly 261,888 tokens with a needle inside it; the thinking-off sends answer directly and the
needle column counts the sends whose answer contains it (sends 1 and 2 cold, send 3 with the prefix before the needle
served from the prefix cache). The thinking-on sends stop at 128 tokens inside the reasoning, before any answer, so they measure decode
speed only.

## 3. Cold prefill

| prompt tokens | 3.0.0 prefill | v2.5.2 prefill | change | 3.0.0 TTFT s |
|---|---|---|---|---|
| 10,187 | 5,660 tok/s | 5,020 tok/s | +12.7 % | 1.80 |
| 101,875 | 6,051 tok/s | 5,417 tok/s | +11.7 % | 16.84 |
| 261,888 | 5,838 tok/s | 5,236 tok/s<sup>1</sup> | +11.5 % | 44.9 |

Prefill is one cold send per depth (8 tokens out); the 262K row is the exact-262K run's cold send 1 (section 2).
<sup>1</sup> v2.5.2 re-ran prefill at 10K and 100K only (within 0.1 % of v2.5.1 there); its 262K figure is the v2.5.1 card's
(2026-10-06, 261K, median of 3 cold sends).

Prefill over context depth: prompt tokens / median cold TTFT of the thinking-on ladder cells (3 sends each), and the
exact-262K row's cold send.

| prompt tokens | prefill | TTFT s |
|---|---|---|
| 4,076 | 5,159 tok/s | 0.79 |
| 32,596 | 6,081 tok/s | 5.36 |
| 130,416 | 6,046 tok/s | 21.57 |
| 259,587 | 5,823 tok/s | 44.58 |
| 261,888 | 5,838 tok/s | 44.9 |

## 4. Sessions held at once

| sessions | needles found | re-sent prompts served from cache (lowest) | all held at once |
|---|---|---|---|
| 3 × 262K | 3/3 | 99.95 % | yes |
| 6 × 131K | 6/6 | 99.91 % | yes |

Peak KV pool use during the test 100.0 %; preemptions 0.

Each stage sends N needle prompts of the stated length at once (each a distinct haystack, thinking off, greedy), then
re-sends the same N prompts one by one. The engine admits the first sends as prefill allows, so the re-send is the
residency proof: every prompt still served from the prefix cache means the KV pool held all N sessions together.

## 5. Images

| case | answer correct | prompt tokens | lowest free memory on any card |
|---|---|---|---|
| one 8 MP image | 1/1 | 8250 | 1,028 MiB |
| two 8 MP requests at once | 2/2 | 8250/8250 | 1,026 MiB |
| four 8 MP images in one prompt | 1/1 | 32842 | 1,026 MiB |

A 4096 × 2048 PNG (8.4 MP) with two solid halves; the model names both colours (thinking off; with four images, those of
the last one), graded on its final `left=… right=…` answer. Free memory is
sampled every 100 ms on every card in use while the request runs.

## 6. Real-text decode, agents, GSM8K and needles

Source: the Merlin 3.0.0 release wheel, 2026-10-20; needle panel: a pre-release engine build, 2026-10-17. Thinking off, temperature 0 unless stated.

**Real-text decode** (one conversation: a natural-text prompt, 512 tokens out, then a follow-up turn)

| prompt tokens | decode tok/s | TTFT s | prefill tok/s | follow-up TTFT s | follow-up cache reuse |
|---|---|---|---|---|---|
| 32,062 | 150.79 | 5.387 | 5943.6 | 0.369 | 97.9 % |
| 131,065 | 145.06 | 22.093 | 5931.5 | 0.779 | 99.5 % |
| 258,210 | 146.38 | 45.361 | 5692.3 | 1.122 | 99.8 % |

**Agents at once** (N 16K conversations, then N simultaneous 512-token follow-ups; total tok/s while all N decode)

| agents | total tok/s | overlap window s |
|---|---|---|
| 2 | 221.52 | 4.55 |
| 4 | 378.17 | 5.22 |
| 8 | 638.16 | 5.97 |

A short overlap window (a few seconds) means the replies ended before all N overlapped for long; read those rows loosely.

**GSM8K-200** (5-shot, thinking off): 199/200 correct.

**Needle panel** (single-occurrence needles at varied positions, an image in the prompt, thinking off, greedy, cold cache): 10/10 at 131K, 23/23 at 262K found exactly.


