# Nemotron 3.5 Super VL — Merlin 3.0.0 bench card, 2026-10-20

> **Provenance.** Maintainer measurements on the reference host: 4× RTX 3090 (24 GB, sm_86) at 220 W, PCIe Gen4 with P2P,
> no NVLink, driver 610.43.02. Engine: the Merlin 3.0.0 release wheel (`vllm-0.32.0+merlin`, built for sm_80/86/89), installed the way a user
> installs it with `release/install-env.sh`; each configuration served by `merlin serve` with
> the release defaults, one fresh boot per configuration (2026-10-20), except the sections marked "Source: a pre-release engine build, 2026-10-17". Streams are client-timed; drivers, raw streams and boot logs are not published.
>
> Checkpoint: [halt95/Nemotron-3.5-Super-VL-120B-A12B-W4A16-Merlin](https://huggingface.co/halt95/Nemotron-3.5-Super-VL-120B-A12B-W4A16-Merlin). Four cards, MTP, FP8 KV, full thinking (the driver sent
> `reasoning_effort: medium`, which this model's template does not read: its effort switch is `medium_effort`, left unset); KV pool 1,392,275 tokens; boot 391 s.
> The shipped 1M default (max_model_len 1,048,576) allocates 1,856,367 tokens at the same KV pin; this card's pool was measured at a 262K context length.
>
> **Ladder notes.** The ladder's prompt sizes are counted in Qwen tokens, and Nemotron's tokenizer counts the same text
> about 4.8 % longer, so the 261,120 rung does not fit the 262,144 context; the 262K decode figures are the exact 262K rows
> below (261,888 tokens counted by Nemotron's own tokenizer). The cold-prefill rows were re-run with the request sizes
> scaled by 1/1.0476 so the prompt token counts match the Qwen checkpoints'. The thinking-off ladder cells fail the ladder's
> answer-format check (the model formats the no-think task differently), so thinking-off speed is read from the exact
> 262K rows, which every send passed. The engine takes one image per prompt.

## 1. Single-stream decode ladder (thinking on)

Same frozen ladder driver as the Flash-Next v2.5.1 and v2.5.2 benchmarks: T=0, a needle prompt at depth, every send
a cold prefill, ≤ 256 tokens with thinking on (full thinking, the template's default) and the no-think answer-plus-summary task
with thinking off (natural stop, ≤ 1,024 tokens); one discarded warm-up send per boot, then 3 sends per cell, each cell the
median of the valid sends. tok/s = tokens after the first streamed event / time from the first to the last event. With
MTP, read the median event interval (ITL) together with tokens per event, not tok/s alone.

**262K row (new in 3.0.0).** Exactly 261,888 prompt tokens + 128 out, both thinking modes, 3 sends (sends 1 and 2 cold;
send 3 reused the prefix before the needle from the prefix cache). The table gives the median decode and send 1's TTFT.

| prompt tokens | thinking | tok/s | ITL ms | tokens/event | TTFT s | valid |
|---|---|---|---|---|---|---|
| 4,269 | on | 96.6 | 36.14 | 3.46 | 1.37 | 3/3 |
| 34,147 | on | 101.2 | 36.04 | 3.61 | 8.42 | 3/3 |
| 136,625 | on | 101.3 | 36.35 | 3.61 | 38.84 | 3/3 |
| 261,888 | on | 91.5 | 37.06 | 3.20 | 88.8 | 3 sends (section 2) |

<sub>262K row added on 2026-10-22, after first publication, from the same runs.</sub>

## 2. Exact 262K

| prompt tokens | thinking | decode tok/s (median of 3) | ITL ms | tokens/event | cold TTFT s | needle found |
|---|---|---|---|---|---|---|
| 261,888 | on | 91.5 | 37.06 | 3.20 | 88.8 | — (see note) |
| 261,888 | off | 82.8 | 36.13 | 2.84 | 88.5 | 3/3 |

**262K needle.** Each prompt is exactly 261,888 tokens with a needle inside it; the thinking-off sends answer directly and the
needle column counts the sends whose answer contains it (sends 1 and 2 cold, send 3 with the prefix before the needle
served from the prefix cache). The thinking-on sends stop at 128 tokens inside the reasoning, before any answer, so they measure decode
speed only.

## 3. Cold prefill

| prompt tokens | prefill | TTFT s |
|---|---|---|
| 10,187 | 3,737 tok/s | 2.73 |
| 101,885 | 3,744 tok/s | 27.21 |
| 261,888 | 2,951 tok/s | 88.8 |

<sub>262K row added on 2026-10-22, after first publication, from the same runs.</sub>

Prefill is one cold send per depth (8 tokens out); the 262K row is the exact-262K run's cold send 1 (section 2).

Prefill over context depth: prompt tokens / median cold TTFT of the thinking-on ladder cells (3 sends each), and the
exact-262K row's cold send.

| prompt tokens | prefill | TTFT s |
|---|---|---|
| 4,269 | 3,116 tok/s | 1.37 |
| 34,147 | 4,055 tok/s | 8.42 |
| 136,625 | 3,518 tok/s | 38.84 |
| 261,888 | 2,951 tok/s | 88.8 |

## 4. Sessions held at once

| sessions | needles found | re-sent prompts served from cache (lowest) | all held at once |
|---|---|---|---|
| 5 × 262K | 5/5 | 99.96 % | yes |
| 9 × 131K | 9/9 | 99.91 % | yes |

Peak KV pool use during the test 95.0 %; preemptions 0.

Each stage sends N needle prompts of the stated length at once (each a distinct haystack, thinking off, greedy), then
re-sends the same N prompts one by one. The engine admits the first sends as prefill allows, so the re-send is the
residency proof: every prompt still served from the prefix cache means the KV pool held all N sessions together. The 131K stage holds 9, the engine's maximum number of sequences.

## 5. Images

| case | answer correct | prompt tokens | lowest free memory on any card |
|---|---|---|---|
| one 8 MP image | 1/1 | 3344 | 686 MiB |

A 4096 × 2048 PNG (8.4 MP) with two solid halves; the model names both colours (thinking off; with four images, those of
the last one), graded on its final `left=… right=…` answer. Free memory is
sampled every 100 ms on every card in use while the request runs. Nemotron 3.5 Super VL downscales the 8 MP input to its native ~3.4 MP cap; the engine takes one image per prompt, so the four-image row does not apply.

## 6. Real-text decode, agents, GSM8K and needles

Source: a pre-release engine build, 2026-10-17. Thinking off, temperature 0 unless stated.

**Real-text decode** (one conversation: a natural-text prompt, 512 tokens out, then a follow-up turn)

| prompt tokens | decode tok/s | TTFT s | prefill tok/s | follow-up TTFT s | follow-up cache reuse |
|---|---|---|---|---|---|
| 32,032 | 85.95 | 8.439 | 3792.7 | 0.477 | 98.0 % |
| 131,075 | 93.11 | 39.912 | 3284.6 | 0.818 | 99.5 % |
| 258,204 | 83.35 | 93.912 | 2749.2 | 1.253 | 99.7 % |

**Agents at once** (N 16K conversations, then N simultaneous 512-token follow-ups; total tok/s while all N decode)

| agents | total tok/s | overlap window s |
|---|---|---|
| 2 | 129.79 | 7.34 |
| 4 | 233.31 | 8.61 |
| 8 | 428.74 | 8.82 |

A short overlap window (a few seconds) means the replies ended before all N overlapped for long; read those rows loosely.

**GSM8K-200** (5-shot, thinking off): 195/200 correct.

**Needle panel** (single-occurrence needles at varied positions, an image in the prompt, thinking off, greedy, cold cache): 10/10 at 131K, 23/23 at 262K found exactly.


