# Swift1.5-Huihui — Merlin 3.0.0 bench card, 2026-10-20

> **Provenance.** Maintainer measurements on the reference host: 4× RTX 3090 (24 GB, sm_86) at 220 W, PCIe Gen4 with P2P,
> no NVLink, driver 610.43.02. Engine: the Merlin 3.0.0 release wheel (`vllm-0.32.0+merlin`, built for sm_80/86/89), installed the way a user
> installs it with `release/install-env.sh`; each configuration served by `merlin serve` with
> the release defaults, one fresh boot per configuration (2026-10-20), except the section 6 needle panel, marked "needle panel: a pre-release engine build, 2026-10-17". Streams are client-timed; drivers, raw streams and boot logs are not published.
>
> Checkpoint: [halt95/Swift1.5-Huihui-Qwen3.8-Flash-Next-W4A16-Merlin](https://huggingface.co/halt95/Swift1.5-Huihui-Qwen3.8-Flash-Next-W4A16-Merlin). Layout: TP2 × PP2 + EP on four cards, MTP K=3 probabilistic, FP8 KV; thinking-on effort `xhigh` (the default at measurement time); boot 408 s (compile cache carried over from an earlier boot); KV pool 924,993 tokens.
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

| prompt tokens | thinking | tok/s | ITL ms | tokens/event | TTFT s | valid |
|---|---|---|---|---|---|---|
| 4,088 | on | 187.3 | 18.14 | 3.37 | 0.79 | 3/3 |
| 32,608 | on | 192.0 | 18.41 | 3.46 | 5.34 | 3/3 |
| 130,428 | on | 188.5 | 18.71 | 3.37 | 21.47 | 3/3 |
| 259,599 | on | 188.6 | 18.95 | 3.46 | 44.27 | 3/3 |
| 4,105 | off | 132.3 | 18.27 | 2.41 | 0.82 | 3/3 |
| 32,625 | off | 128.1 | 18.21 | 2.32 | 5.34 | 3/3 |
| 130,445 | off | 126.4 | 18.79 | 2.34 | 21.46 | 3/3 |
| 259,616 | off | 126.7 | 18.94 | 2.37 | 44.13 | 3/3 |

## 2. Exact 262K

| prompt tokens | thinking | decode tok/s (median of 3) | ITL ms | tokens/event | cold TTFT s | needle found |
|---|---|---|---|---|---|---|
| 261,888 | on | 189.5 | 18.93 | 3.37 | 44.5 | — (see note) |
| 261,888 | off | 156.6 | 18.81 | 2.78 | 44.4 | 3/3 |

**262K needle.** Each prompt is exactly 261,888 tokens with a needle inside it; the thinking-off sends answer directly and the
needle column counts the sends whose answer contains it (sends 1 and 2 cold, send 3 with the prefix before the needle
served from the prefix cache). The thinking-on sends stop at 128 tokens inside the reasoning, before any answer, so they measure decode
speed only.

## 3. Cold prefill

| prompt tokens | 3.0.0 prefill | v2.5.2 prefill | change | 3.0.0 TTFT s |
|---|---|---|---|---|
| 10,199 | 5,706 tok/s | 5,028 tok/s | +13.5 % | 1.79 |
| 101,887 | 6,097 tok/s | 5,446 tok/s | +12.0 % | 16.71 |
| 261,888 | 5,880 tok/s | not measured | — | 44.5 |

Prefill is one cold send per depth (8 tokens out); the 262K row is the exact-262K run's cold send 1 (section 2).
The v2.5.2 card has no 262K prefill for this checkpoint.

Prefill over context depth: prompt tokens / median cold TTFT of the thinking-on ladder cells (3 sends each), and the
exact-262K row's cold send.

| prompt tokens | prefill | TTFT s |
|---|---|---|
| 4,088 | 5,175 tok/s | 0.79 |
| 32,608 | 6,106 tok/s | 5.34 |
| 130,428 | 6,075 tok/s | 21.47 |
| 259,599 | 5,864 tok/s | 44.27 |
| 261,888 | 5,880 tok/s | 44.5 |

The v2.5.2 decode figures for this checkpoint used a different send protocol
(a warm-up send, then the mean of two timed sends), so they are not set against this ladder; prefill is like for like.

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
| one 8 MP image | 1/1 | 8250 | 1,136 MiB |
| two 8 MP requests at once | 2/2 | 8250/8250 | 1,136 MiB |

A 4096 × 2048 PNG (8.4 MP) with two solid halves; the model names both colours (thinking off; with four images, those of
the last one), graded on its final `left=… right=…` answer. Free memory is
sampled every 100 ms on every card in use while the request runs. The four-image prompt ran on the Flash-Next checkpoint; Swift1.5 and Swift1.5-Huihui share its engine and vision path.

## 6. Real-text decode, agents, GSM8K and needles

Source: the Merlin 3.0.0 release wheel, 2026-10-20; needle panel: a pre-release engine build, 2026-10-17. Thinking off, temperature 0 unless stated.

**Real-text decode** (one conversation: a natural-text prompt, 512 tokens out, then a follow-up turn)

| prompt tokens | decode tok/s | TTFT s | prefill tok/s | follow-up TTFT s | follow-up cache reuse |
|---|---|---|---|---|---|
| 32,062 | 151.12 | 5.385 | 5946.9 | 0.363 | 97.9 % |
| 131,065 | 146.99 | 22.07 | 5938.5 | 0.767 | 99.5 % |
| 258,210 | 145.66 | 45.311 | 5701.4 | 1.116 | 99.8 % |

**Agents at once** (N 16K conversations, then N simultaneous 512-token follow-ups; total tok/s while all N decode)

| agents | total tok/s | overlap window s |
|---|---|---|
| 2 | 218.37 | 4.53 |
| 4 | n/a | n/a |
| 8 | n/a | n/a |

A short overlap window (a few seconds) means the replies ended before all N overlapped for long; read those rows loosely.

8 agents, re-run on the released 3.0.0 install, 2026-10-21: 660 total tok/s (the 2026-10-20 run above did not overlap, hence n/a).

**GSM8K-200** (5-shot, thinking off): 198/200 correct.

**Needle panel** (single-occurrence needles at varied positions, an image in the prompt, thinking off, greedy, cold cache): 10/10 at 131K, 23/23 at 262K found exactly.


