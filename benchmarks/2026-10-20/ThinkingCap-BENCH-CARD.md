# ThinkingCap — Merlin 3.0.0 bench card, 2026-10-20

> **Provenance.** Maintainer measurements on the reference host: 4× RTX 3090 (24 GB, sm_86) at 220 W, PCIe Gen4 with P2P,
> no NVLink, driver 610.43.02. Engine: the Merlin 3.0.0 release wheel (`vllm-0.32.0+merlin`, built for sm_80/86/89), installed the way a user
> installs it with `release/install-env.sh`; each configuration served by `merlin serve` with
> the release defaults, one fresh boot per configuration (2026-10-20), except the sections marked "Source: a pre-release engine build, 2026-10-17". Streams are client-timed; drivers, raw streams and boot logs are not published.
>
> Checkpoint: [halt95/ThinkingCap-Qwen3.8-27B-W4A16-Merlin](https://huggingface.co/halt95/ThinkingCap-Qwen3.8-27B-W4A16-Merlin).
> MTP K=4, FP8 KV, RecoverSSM on, thinking-on effort `xhigh` (the default at measurement time). Two cards (TP2) and four cards (TP4).
> The two-card comparison is against the 2026-10-12 ThinkingCap card (v1.3.0 profile), whose ladder sampled at T 1.0 with
> effort `low` (this one: T=0, effort `xhigh`): compare ITL, TTFT and prefill like for like, tok/s and tokens per step
> only loosely.

## 1. Single-stream decode ladder

Same frozen ladder driver as the Flash-Next v2.5.1 and v2.5.2 benchmarks: T=0, a needle prompt at depth, every send
a cold prefill, ≤ 256 tokens with thinking on (the served default reasoning effort) and the no-think answer-plus-summary task
with thinking off (natural stop, ≤ 1,024 tokens); one discarded warm-up send per boot, then 3 sends per cell, each cell the
median of the valid sends. tok/s = tokens after the first streamed event / time from the first to the last event. With
MTP, read the median event interval (ITL) together with tokens per event, not tok/s alone.

**262K row (new in 3.0.0).** Exactly 261,888 prompt tokens + 128 out, both thinking modes, 3 sends (sends 1 and 2 cold;
send 3 reused the prefix before the needle from the prefix cache). The table gives the median decode and send 1's TTFT.

### Two cards (TP2) — KV pool 826,689 tokens, boot 178 s

| prompt tokens | thinking | 3.0.0 tok/s | 10-12 card tok/s | change | ITL ms (3.0.0 / 10-12 card) | tokens/event (3.0.0 / 10-12 card) | TTFT s | valid |
|---|---|---|---|---|---|---|---|---|
| 4,088 | on | 198.4 | 211.3 | -6.1 % | 20.08 / 20.0 | 3.93 / 4.12 | 2.19 | 3/3 |
| 32,608 | on | 199.4 | 172.8 | +15.4 % | 21.60 / 21.5 | 4.19 / 3.62 | 19.37 | 3/3 |
| 130,428 | on | 182.5 | 149.9 | +21.7 % | 25.34 / 25.4 | 4.39 / 3.53 | 109.70 | 3/3 |
| 259,599 | on | 146.2 | 157.7 | -7.3 % | 30.55 / 31.0 | 4.34 / 4.54 | 301.85 | 3/3 |
| 4,105 | off | 143.0 | 140.1 | +2.1 % | 20.13 / 20.1 | 2.87 / 2.80 | 2.20 | 3/3 |
| 32,625 | off | 135.9 | 126.3 | +7.6 % | 21.43 / 21.4 | 2.88 / 2.69 | 19.45 | 3/3 |
| 130,445 | off | 115.9 | 107.7 | +7.6 % | 25.25 / 25.2 | 2.87 / 2.68 | 109.73 | 3/3 |
| 259,616 | off | 94.6 | 93.6 | +1.1 % | 30.51 / 30.6 | 2.86 / 2.84 | 301.85 | 3/3 |

### Four cards (TP4) — KV pool 2,012,105 tokens, boot 209 s

| prompt tokens | thinking | tok/s | ITL ms | tokens/event | TTFT s | valid |
|---|---|---|---|---|---|---|
| 4,088 | on | 252.9 | 16.03 | 4.00 | 1.37 | 3/3 |
| 32,608 | on | 252.1 | 16.83 | 4.10 | 11.49 | 3/3 |
| 130,428 | on | 254.6 | 18.59 | 4.49 | 61.29 | 3/3 |
| 259,599 | on | 228.6 | 20.86 | 4.53 | 161.66 | 3/3 |
| 4,105 | off | 188.2 | 16.00 | 3.00 | 1.43 | 3/3 |
| 32,625 | off | 162.9 | 16.82 | 2.71 | 11.52 | 3/3 |
| 130,445 | off | 151.2 | 18.50 | 2.76 | 61.29 | 3/3 |
| 259,616 | off | 138.2 | 20.88 | 2.85 | 161.63 | 3/3 |

## 2. Exact 262K

**Two cards**

| prompt tokens | thinking | decode tok/s (median of 3) | ITL ms | tokens/event | cold TTFT s | needle found |
|---|---|---|---|---|---|---|
| 261,888 | on | 164.8 | 30.72 | 4.57 | 304.6 | — (see note) |
| 261,888 | off | 88.1 | 30.51 | 2.56 | 304.4 | 3/3 |

**262K needle.** Each prompt is exactly 261,888 tokens with a needle inside it; the thinking-off sends answer directly and the
needle column counts the sends whose answer contains it (sends 1 and 2 cold, send 3 with the prefix before the needle
served from the prefix cache). The thinking-on sends stop at 128 tokens inside the reasoning, before any answer, so they measure decode
speed only.

**Four cards**

| prompt tokens | thinking | decode tok/s (median of 3) | ITL ms | tokens/event | cold TTFT s | needle found |
|---|---|---|---|---|---|---|
| 261,888 | on | 248.2 | 20.73 | 4.74 | 164.3 | — (see note) |
| 261,888 | off | 146.0 | 20.91 | 2.91 | 164.2 | 3/3 |

**262K needle.** Each prompt is exactly 261,888 tokens with a needle inside it; the thinking-off sends answer directly and the
needle column counts the sends whose answer contains it (sends 1 and 2 cold, send 3 with the prefix before the needle
served from the prefix cache). The thinking-on sends stop at 128 tokens inside the reasoning, before any answer, so they measure decode
speed only.

## 3. Cold prefill

**Two cards**

| prompt tokens | prefill | TTFT s |
|---|---|---|
| 10,199 | 1,829 tok/s | 5.58 |
| 101,887 | 1,297 tok/s | 78.57 |
| 261,888 | 860 tok/s | 304.6 |

Prefill is one cold send per depth (8 tokens out); the 262K row is the exact-262K run's cold send 1.

**Four cards**

| prompt tokens | prefill | TTFT s |
|---|---|---|
| 10,199 | 3,013 tok/s | 3.38 |
| 101,887 | 2,295 tok/s | 44.40 |
| 261,888 | 1,594 tok/s | 164.3 |

Prefill is one cold send per depth (8 tokens out); the 262K row is the exact-262K run's cold send 1.

### Prefill over context depth

**Two cards**

Prefill over context depth: prompt tokens / median cold TTFT of the thinking-on ladder cells (3 sends each), and the
exact-262K row's cold send.

| prompt tokens | 3.0.0 prefill | 10-12 card prefill | change | 3.0.0 TTFT s |
|---|---|---|---|---|
| 4,088 | 1,867 tok/s | 1,749 tok/s | +6.7 % | 2.19 |
| 32,608 | 1,683 tok/s | 1,593 tok/s | +5.7 % | 19.37 |
| 130,428 | 1,189 tok/s | 1,162 tok/s | +2.3 % | 109.70 |
| 259,599 | 860 tok/s | 859 tok/s | +0.1 % | 301.85 |
| 261,888 | 860 tok/s | — | — | 304.6 |

**Four cards**

Prefill over context depth: prompt tokens / median cold TTFT of the thinking-on ladder cells (3 sends each), and the
exact-262K row's cold send.

| prompt tokens | prefill | TTFT s |
|---|---|---|
| 4,088 | 2,984 tok/s | 1.37 |
| 32,608 | 2,838 tok/s | 11.49 |
| 130,428 | 2,128 tok/s | 61.29 |
| 259,599 | 1,606 tok/s | 161.66 |
| 261,888 | 1,594 tok/s | 164.3 |

## 4. Sessions held at once

**Two cards**

| sessions | needles found | re-sent prompts served from cache (lowest) | all held at once |
|---|---|---|---|
| 3 × 262K | 3/3 | 99.99 % | yes |
| 6 × 131K | 6/6 | 99.98 % | yes |

Peak KV pool use during the test 100.0 %; preemptions 0.

Each stage sends N needle prompts of the stated length at once (each a distinct haystack, thinking off, greedy), then
re-sends the same N prompts one by one. The engine admits the first sends as prefill allows, so the re-send is the
residency proof: every prompt still served from the prefix cache means the KV pool held all N sessions together.

**Four cards**

| sessions | needles found | re-sent prompts served from cache (lowest) | all held at once |
|---|---|---|---|
| 7 × 262K | 7/7 | 99.99 % | yes |
| 14 × 131K | 14/14 | 99.98 % | yes |

Peak KV pool use during the test 100.0 %; preemptions 0.

Each stage sends N needle prompts of the stated length at once (each a distinct haystack, thinking off, greedy), then
re-sends the same N prompts one by one. The engine admits the first sends as prefill allows, so the re-send is the
residency proof: every prompt still served from the prefix cache means the KV pool held all N sessions together.

## 5. Images

**Two cards**

| case | answer correct | prompt tokens | lowest free memory on any card |
|---|---|---|---|
| one 8 MP image | 1/1 | 8250 | 886 MiB |
| two 8 MP requests at once | 2/2 | 8250/8250 | 886 MiB |
| four 8 MP images in one prompt | 1/1 | 32842 | 886 MiB |

A 4096 × 2048 PNG (8.4 MP) with two solid halves; the model names both colours (thinking off; with four images, those of
the last one), graded on its final `left=… right=…` answer. Free memory is
sampled every 100 ms on every card in use while the request runs.

**Four cards**

| case | answer correct | prompt tokens | lowest free memory on any card |
|---|---|---|---|
| one 8 MP image | 1/1 | 8250 | 862 MiB |
| two 8 MP requests at once | 2/2 | 8250/8250 | 862 MiB |
| four 8 MP images in one prompt | 1/1 | 32842 | 862 MiB |

A 4096 × 2048 PNG (8.4 MP) with two solid halves; the model names both colours (thinking off; with four images, those of
the last one), graded on its final `left=… right=…` answer. Free memory is
sampled every 100 ms on every card in use while the request runs.

## 6. Real-text decode, agents, GSM8K and needles

### Two cards

Source: a pre-release engine build, 2026-10-17. Thinking off, temperature 0 unless stated.

**Real-text decode** (one conversation: a natural-text prompt, 512 tokens out, then a follow-up turn)

| prompt tokens | decode tok/s | TTFT s | prefill tok/s | follow-up TTFT s | follow-up cache reuse |
|---|---|---|---|---|---|
| 32,062 | 154.44 | 18.75 | 1706.5 | 0.557 | 98.2 % |
| 131,065 | 122.14 | 109.885 | 1192.8 | 1.221 | 99.6 % |
| 258,210 | 94.42 | 297.847 | 867.2 | 2.151 | 99.8 % |

**Agents at once** (N 16K conversations, then N simultaneous 512-token follow-ups; total tok/s while all N decode)

| agents | total tok/s | overlap window s |
|---|---|---|
| 2 | 278.95 | 3.58 |
| 4 | 361.96 | 5.34 |
| 8 | 501.99 | 7.29 |

A short overlap window (a few seconds) means the replies ended before all N overlapped for long; read those rows loosely.

**GSM8K-200** (5-shot, thinking off): 193/200 correct.

**Needle panel** (single-occurrence needles at varied positions, an image in the prompt, thinking off, greedy, cold cache): 5/5 at 131K, 12/12 at 262K found exactly.


### Four cards

Source: the Merlin 3.0.0 release wheel, 2026-10-20. Thinking off, temperature 0 unless stated.

**Real-text decode** (one conversation: a natural-text prompt, 512 tokens out, then a follow-up turn)

| prompt tokens | decode tok/s | TTFT s | prefill tok/s | follow-up TTFT s | follow-up cache reuse |
|---|---|---|---|---|---|
| 32,062 | 203.11 | 11.192 | 2859.0 | 0.382 | 98.2 % |
| 131,065 | 190.07 | 61.419 | 2134.0 | 0.881 | 99.6 % |
| 258,210 | 160.11 | 160.226 | 1611.7 | 1.416 | 99.8 % |

**Agents at once** (N 16K conversations, then N simultaneous 512-token follow-ups; total tok/s while all N decode)

| agents | total tok/s | overlap window s |
|---|---|---|
| 2 | 364.64 | 2.76 |
| 4 | 503.9 | 3.81 |
| 8 | 732.73 | 5.13 |

A short overlap window (a few seconds) means the replies ended before all N overlapped for long; read those rows loosely.

**GSM8K-200** (5-shot, thinking off): 195/200 correct.

**Needle panel** (single-occurrence needles at varied positions, an image in the prompt, thinking off, greedy, cold cache): 5/5 at 131K, 12/12 at 262K found exactly.


