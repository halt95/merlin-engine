# Qwen3.8-27B — Merlin 3.0.0 bench card, 2026-10-20

> **Provenance.** Maintainer measurements on the reference host: 4× RTX 3090 (24 GB, sm_86) at 220 W, PCIe Gen4 with P2P,
> no NVLink, driver 610.43.02. Engine: the Merlin 3.0.0 release wheel (`vllm-0.32.0+merlin`, built for sm_80/86/89), installed the way a user
> installs it with `release/install-env.sh`; each configuration served by `merlin serve` with
> the release defaults, one fresh boot per configuration (2026-10-20), except the sections marked "Source: a pre-release engine build, 2026-10-17". Streams are client-timed; drivers, raw streams and boot logs are not published.
>
> Checkpoint: [halt95/Qwen3.8-27B-W4A16-Merlin](https://huggingface.co/halt95/Qwen3.8-27B-W4A16-Merlin). MTP K=4, FP8 KV,
> RecoverSSM on, thinking-on effort `medium` (the default). Two cards (TP2, GPUs 0 and 2), four cards (TP4) and,
> new in 3.0.0, one card (TP1).
>
> **Against v1.3.0.** The v1.3.0 card's ladder sampled at the server's T 1.0 with reasoning effort `low`; this ladder runs
> at T=0 with the served default effort `medium`, so tok/s and tokens per step are not like for like; the step time (ITL),
> TTFT and prefill are. v1.3.0's four-card layout ran without RecoverSSM; 3.0.0 runs it on every layout. Step time is
> unchanged; prefill and TTFT are faster, most on four cards. Boot times are with a compile cache carried over.

## 1. Single-stream decode ladder

Same frozen ladder driver as the Flash-Next v2.5.1 and v2.5.2 benchmarks: T=0, a needle prompt at depth, every send
a cold prefill, ≤ 256 tokens with thinking on (the served default reasoning effort) and the no-think answer-plus-summary task
with thinking off (natural stop, ≤ 1,024 tokens); one discarded warm-up send per boot, then 3 sends per cell, each cell the
median of the valid sends. tok/s = tokens after the first streamed event / time from the first to the last event. With
MTP, read the median event interval (ITL) together with tokens per event, not tok/s alone.

**262K row (new in 3.0.0).** Exactly 261,888 prompt tokens + 128 out, both thinking modes, 3 sends (sends 1 and 2 cold;
send 3 reused the prefix before the needle from the prefix cache). The table gives the median decode and send 1's TTFT.

### Two cards (TP2) — KV pool 826,689 tokens, boot 177 s

| prompt tokens | thinking | 3.0.0 tok/s | v1.3.0 tok/s | change | ITL ms (3.0.0 / v1.3.0) | tokens/event (3.0.0 / v1.3.0) | TTFT s | valid |
|---|---|---|---|---|---|---|---|---|
| 4,046 | on | 206.6 | 205.6 | +0.5 % | 20.21 / 20.0 | 4.13 / 4.06 | 2.17 | 3/3 |
| 32,566 | on | 196.1 | 197.8 | -0.9 % | 21.52 / 21.3 | 4.13 / 4.11 | 19.40 | 3/3 |
| 130,386 | on | 173.3 | 171.6 | +1.0 % | 25.41 / 25.4 | 4.20 / 4.03 | 109.53 | 3/3 |
| 259,557 | on | 153.6 | 144.3 | +6.4 % | 30.66 / 31.1 | 4.57 / 4.31 | 301.35 | 3/3 |
| 4,105 | off | 140.8 | 149.3 | -5.7 % | 20.23 / 19.9 | 2.85 / 2.96 | 2.21 | 3/3 |
| 32,625 | off | 130.7 | 134.8 | -3.0 % | 21.60 / 21.3 | 2.79 / 2.85 | 19.49 | 3/3 |
| 130,445 | off | 119.0 | 119.6 | -0.5 % | 25.27 / 25.3 | 2.95 / 2.95 | 109.79 | 3/3 |
| 259,616 | off | 91.6 | 89.0 | +2.9 % | 30.57 / 30.7 | 2.78 / 2.70 | 301.81 | 3/3 |

### Four cards (TP4) — KV pool 2,012,105 tokens, boot 219 s

| prompt tokens | thinking | 3.0.0 tok/s | v1.3.0 tok/s | change | ITL ms (3.0.0 / v1.3.0) | tokens/event (3.0.0 / v1.3.0) | TTFT s | valid |
|---|---|---|---|---|---|---|---|---|
| 4,046 | on | 250.3 | 257.8 | -2.9 % | 16.02 / 15.7 | 4.00 / 4.00 | 1.35 | 3/3 |
| 32,566 | on | 252.4 | 248.9 | +1.4 % | 17.01 / 16.6 | 4.20 / 4.04 | 11.46 | 3/3 |
| 130,386 | on | 235.4 | 238.2 | -1.2 % | 18.62 / 18.3 | 4.20 / 4.14 | 61.19 | 3/3 |
| 259,557 | on | 224.0 | 226.0 | -0.9 % | 20.82 / 20.9 | 4.49 / 4.49 | 161.60 | 3/3 |
| 4,105 | off | 186.9 | 180.7 | +3.4 % | 15.93 / 15.8 | 2.96 / 2.83 | 1.42 | 3/3 |
| 32,625 | off | 168.5 | 172.6 | -2.4 % | 16.83 / 16.7 | 2.80 / 2.84 | 11.52 | 3/3 |
| 130,445 | off | 164.8 | 164.5 | +0.2 % | 18.52 / 18.4 | 2.99 / 2.98 | 61.27 | 3/3 |
| 259,616 | off | 133.0 | 140.2 | -5.1 % | 20.82 / 20.8 | 2.73 / 2.86 | 161.59 | 3/3 |

### One card (TP1) — KV pool 271,506 tokens, boot 162 s

| prompt tokens | thinking | tok/s | ITL ms | tokens/event | TTFT s | valid |
|---|---|---|---|---|---|---|
| 4,046 | on | 123.8 | 33.21 | 4.06 | 3.88 | 3/3 |
| 32,566 | on | 121.3 | 36.30 | 4.27 | 35.01 | 3/3 |
| 130,386 | on | 94.8 | 45.25 | 4.13 | 202.41 | 3/3 |
| 259,557 | on | 80.5 | 55.47 | 4.34 | 566.69 | 3/3 |
| 4,105 | off | 84.3 | 33.32 | 2.79 | 3.96 | 3/3 |
| 32,625 | off | 79.0 | 36.03 | 2.82 | 35.18 | 3/3 |
| 130,445 | off | 67.8 | 44.45 | 2.96 | 202.44 | 3/3 |
| 259,616 | off | 50.4 | 55.74 | 2.78 | 567.01 | 3/3 |

## 2. Exact 262K

**Two cards**

| prompt tokens | thinking | decode tok/s (median of 3) | ITL ms | tokens/event | cold TTFT s | needle found |
|---|---|---|---|---|---|---|
| 261,888 | on | 142.5 | 30.76 | 4.13 | 305.3 | — (see note) |
| 261,888 | off | 111.1 | 30.75 | 3.28 | 305.1 | 3/3 |

**262K needle.** Each prompt is exactly 261,888 tokens with a needle inside it; the thinking-off sends answer directly and the
needle column counts the sends whose answer contains it (sends 1 and 2 cold, send 3 with the prefix before the needle
served from the prefix cache). The thinking-on sends stop at 128 tokens inside the reasoning, before any answer, so they measure decode
speed only.

**Four cards**

| prompt tokens | thinking | decode tok/s (median of 3) | ITL ms | tokens/event | cold TTFT s | needle found |
|---|---|---|---|---|---|---|
| 261,888 | on | 214.8 | 20.74 | 4.13 | 164.1 | — (see note) |
| 261,888 | off | 158.1 | 20.78 | 3.12 | 164.0 | 3/3 |

**262K needle.** Each prompt is exactly 261,888 tokens with a needle inside it; the thinking-off sends answer directly and the
needle column counts the sends whose answer contains it (sends 1 and 2 cold, send 3 with the prefix before the needle
served from the prefix cache). The thinking-on sends stop at 128 tokens inside the reasoning, before any answer, so they measure decode
speed only.

**One card**

| prompt tokens | thinking | decode tok/s (median of 3) | ITL ms | tokens/event | cold TTFT s | needle found |
|---|---|---|---|---|---|---|
| 261,888 | on | 72.9 | 57.39 | 3.76 | 574.1 | — (see note) |
| 261,888 | off | 49.3 | 56.56 | 2.67 | 574.2 | 3/3 |

**262K needle.** Each prompt is exactly 261,888 tokens with a needle inside it; the thinking-off sends answer directly and the
needle column counts the sends whose answer contains it (sends 1 and 2 cold, send 3 with the prefix before the needle
served from the prefix cache). The thinking-on sends stop at 128 tokens inside the reasoning, before any answer, so they measure decode
speed only.

## 3. Cold prefill

**Two cards**

| prompt tokens | 3.0.0 prefill | v1.3.0 prefill | change | 3.0.0 TTFT s |
|---|---|---|---|---|
| 10,157 | 1,823 tok/s | 1,739 tok/s | +4.8 % | 5.57 |
| 101,845 | 1,297 tok/s | 1,267 tok/s | +2.4 % | 78.51 |
| 261,888 | 858 tok/s | not measured | — | 305.3 |

Prefill is one cold send per depth (8 tokens out); the 262K row is the exact-262K run's cold send 1. v1.3.0 has no exact-262K
send; like for like at 259,557 tokens (median of 3 cold sends) it is in the depth table below.

**Four cards**

| prompt tokens | 3.0.0 prefill | v1.3.0 prefill | change | 3.0.0 TTFT s |
|---|---|---|---|---|
| 10,157 | 3,009 tok/s | 2,483 tok/s | +21.2 % | 3.38 |
| 101,845 | 2,295 tok/s | 1,989 tok/s | +15.4 % | 44.37 |
| 261,888 | 1,596 tok/s | not measured | — | 164.1 |

Prefill is one cold send per depth (8 tokens out); the 262K row is the exact-262K run's cold send 1. v1.3.0 has no exact-262K
send; like for like at 259,557 tokens (median of 3 cold sends) it is in the depth table below.

**One card**

| prompt tokens | prefill | TTFT s |
|---|---|---|
| 10,157 | 1,021 tok/s | 9.94 |
| 101,845 | 708 tok/s | 143.85 |
| 261,888 | 456 tok/s | 574.1 |

Prefill is one cold send per depth (8 tokens out); the 262K row is the exact-262K run's cold send 1. v1.3.0 has no exact-262K
send; like for like at 259,557 tokens (median of 3 cold sends) it is in the depth table below.

### Prefill over context depth

**Two cards**

Prefill over context depth: prompt tokens / median cold TTFT of the thinking-on ladder cells (3 sends each), and the
exact-262K row's cold send.

| prompt tokens | 3.0.0 prefill | v1.3.0 prefill | change | 3.0.0 TTFT s |
|---|---|---|---|---|
| 4,046 | 1,865 tok/s | 1,766 tok/s | +5.6 % | 2.17 |
| 32,566 | 1,679 tok/s | 1,608 tok/s | +4.4 % | 19.40 |
| 130,386 | 1,190 tok/s | 1,158 tok/s | +2.8 % | 109.53 |
| 259,557 | 861 tok/s | 854 tok/s | +0.9 % | 301.35 |
| 261,888 | 858 tok/s | — | — | 305.3 |

**Four cards**

Prefill over context depth: prompt tokens / median cold TTFT of the thinking-on ladder cells (3 sends each), and the
exact-262K row's cold send.

| prompt tokens | 3.0.0 prefill | v1.3.0 prefill | change | 3.0.0 TTFT s |
|---|---|---|---|---|
| 4,046 | 2,997 tok/s | 2,555 tok/s | +17.3 % | 1.35 |
| 32,566 | 2,842 tok/s | 2,373 tok/s | +19.8 % | 11.46 |
| 130,386 | 2,131 tok/s | 1,868 tok/s | +14.1 % | 61.19 |
| 259,557 | 1,606 tok/s | 1,448 tok/s | +10.9 % | 161.60 |
| 261,888 | 1,596 tok/s | — | — | 164.1 |

**One card**

Prefill over context depth: prompt tokens / median cold TTFT of the thinking-on ladder cells (3 sends each), and the
exact-262K row's cold send.

| prompt tokens | prefill | TTFT s |
|---|---|---|
| 4,046 | 1,043 tok/s | 3.88 |
| 32,566 | 930 tok/s | 35.01 |
| 130,386 | 644 tok/s | 202.41 |
| 259,557 | 458 tok/s | 566.69 |
| 261,888 | 456 tok/s | 574.1 |

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

**One card**

| sessions | needles found | re-sent prompts served from cache (lowest) | all held at once |
|---|---|---|---|
| 1 × 262K | 1/1 | 99.99 % | yes |
| 2 × 131K | 2/2 | 99.98 % | yes |

Peak KV pool use during the test 99.3 %; preemptions 0.

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

**One card**

| case | answer correct | prompt tokens | lowest free memory on any card |
|---|---|---|---|
| one 8 MP image | 0/1 (FAIL: engine out of memory) | n/a | 510 MiB |

A 4096 × 2048 PNG (8.4 MP) with two solid halves; the model names both colours (thinking off; with four images, those of
the last one), graded on its final `left=… right=…` answer. Free memory is
sampled every 100 ms on every card in use while the request runs. The engine stopped on the first 8 MP image (out of GPU memory in the vision encoder); the later rows ran against the stopped server.

One-card layouts ship with a 4 MP image cap. Measured 2026-10-21 on the released build with the single-card memory settings: one 4 MP image and two at once correct, also after a 262K needle, lowest free memory 558-563 MiB (bar 524 MiB).

## 6. Real-text decode, agents, GSM8K and needles

### Two cards

Source: a pre-release engine build, 2026-10-17. Thinking off, temperature 0 unless stated.

**Real-text decode** (one conversation: a natural-text prompt, 512 tokens out, then a follow-up turn)

| prompt tokens | decode tok/s | TTFT s | prefill tok/s | follow-up TTFT s | follow-up cache reuse |
|---|---|---|---|---|---|
| 32,062 | 146.77 | 18.813 | 1700.8 | 0.559 | 98.2 % |
| 131,065 | 122.16 | 109.965 | 1191.9 | 1.235 | 99.6 % |
| 258,210 | 97.19 | 297.55 | 867.8 | 2.149 | 99.8 % |

**Agents at once** (N 16K conversations, then N simultaneous 512-token follow-ups; total tok/s while all N decode)

| agents | total tok/s | overlap window s |
|---|---|---|
| 2 | 280.7 | 3.38 |
| 4 | 372.84 | 4.91 |
| 8 | 517.48 | 7.35 |

A short overlap window (a few seconds) means the replies ended before all N overlapped for long; read those rows loosely.

**GSM8K-200** (5-shot, thinking off): 194/200 correct.

**Needle panel** (single-occurrence needles at varied positions, an image in the prompt, thinking off, greedy, cold cache): 5/5 at 131K, 12/12 at 262K found exactly.


### Four cards

Source: the Merlin 3.0.0 release wheel, 2026-10-20. Thinking off, temperature 0 unless stated.

**Real-text decode** (one conversation: a natural-text prompt, 512 tokens out, then a follow-up turn)

| prompt tokens | decode tok/s | TTFT s | prefill tok/s | follow-up TTFT s | follow-up cache reuse |
|---|---|---|---|---|---|
| 32,062 | 200.91 | 11.188 | 2862.6 | 0.379 | 98.2 % |
| 131,065 | 185.13 | 61.414 | 2134.1 | 0.867 | 99.6 % |
| 258,210 | 162.8 | 160.163 | 1612.9 | 1.396 | 99.8 % |

**Agents at once** (N 16K conversations, then N simultaneous 512-token follow-ups; total tok/s while all N decode)

| agents | total tok/s | overlap window s |
|---|---|---|
| 2 | 361.33 | 2.64 |
| 4 | 549.39 | 3.39 |
| 8 | 751.77 | 4.86 |

A short overlap window (a few seconds) means the replies ended before all N overlapped for long; read those rows loosely.

**GSM8K-200** (5-shot, thinking off): 197/200 correct.

**Needle panel** (single-occurrence needles at varied positions, an image in the prompt, thinking off, greedy, cold cache): 5/5 at 131K, 12/12 at 262K found exactly.


### One card (two sequences at most, so no 4- or 8-agent rows)

Source: a pre-release engine build, 2026-10-17. Thinking off, temperature 0 unless stated.

**Real-text decode** (one conversation: a natural-text prompt, 512 tokens out, then a follow-up turn)

| prompt tokens | decode tok/s | TTFT s | prefill tok/s | follow-up TTFT s | follow-up cache reuse |
|---|---|---|---|---|---|
| 32,062 | 91.12 | 34.348 | 931.6 | 0.906 | 98.2 % |
| 131,065 | 72.93 | 205.98 | 636.5 | 1.793 | 99.6 % |
| 258,210 | 51.56 | 563.187 | 458.7 | 3.08 | 99.8 % |

**Agents at once** (N 16K conversations, then N simultaneous 512-token follow-ups; total tok/s while all N decode)

| agents | total tok/s | overlap window s |
|---|---|---|
| 2 | 163.76 | 5.73 |
| 4 | n/a | -1.37 |
| 8 | n/a | -15.21 |

A short overlap window (a few seconds) means the replies ended before all N overlapped for long; read those rows loosely.

**GSM8K-200** (5-shot, thinking off): 195/200 correct.

**Needle panel** (single-occurrence needles at varied positions, an image in the prompt, thinking off, greedy, cold cache): 5/5 at 131K, 12/12 at 262K found exactly.


