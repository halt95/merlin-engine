# Ornith 1.5 — Merlin 3.0.0 bench card, 2026-10-21

> **Provenance.** Maintainer measurements on the reference host: 4× RTX 3090 (24 GB, sm_86) at 220 W, PCIe Gen4 with P2P,
> no NVLink, driver 610.43.02. Engine for sections 1–5 and the needle panel: the Merlin 3.0.0 release install (wheel
> `vllm-0.32.0+merlin.7002f56537`, built for sm_80/86/89) with Ornith's engine changes added before they merged into 3.0.0.
> That build also carried a prefill experiment that did not ship.
> Served with the profile's flags, one fresh boot for the full card (2026-10-21); GSM8K-200 ran on a separate boot of the
> same setup. Agents and a second GSM8K-200 ran on the released 3.0.0 install, served by `merlin serve` with
> the `ornith-quad-3090` profile, one fresh boot (2026-10-21); the released install resolved the same 894,618-token KV pool.
> Streams are client-timed; drivers, raw streams and boot logs are not published.
>
> Checkpoint: [halt95/Ornith-1.5-122B-A10B-W4A16-Merlin](https://huggingface.co/halt95/Ornith-1.5-122B-A10B-W4A16-Merlin).
> Layout: TP2 × PP2 + EP on four cards, Gated DeltaNet RecoverSSM, MTP K=3 probabilistic from the model's own head,
> uniform MoE verify on, FP8 KV with calibrated scales, 1,024-token prefill chunks, 8 sequences; KV pool 894,618 tokens.
>
> **Thinking.** Ornith's chat template reads only `enable_thinking`; it has no `reasoning_effort`, so the driver's effort
> field has no effect and every thinking-on send ran at full thinking.
>
> **MTP acceptance** (tokens per step, K=3, served sampling, 10 prompts per set, a separate run on the shipping
> configuration): thinking on, prose 2.48 at 4K and 2.39 at 32K, code 2.56, GSM8K 2.58; thinking off 2.19 / 2.18 / 2.72 / 2.59.
> The INT4 MTP experts accept as the BF16 ones do (mean acceptance length 2.943 vs 2.947, measured on the no-P2P host).

## 1. Single-stream decode ladder

T=0, a needle prompt at depth, every send a cold prefill (nothing cached), ≤ 256 tokens with thinking on and the no-think
answer-plus-summary task with thinking off (natural stop, ≤ 1,024 tokens); one discarded warm-up send per boot, then 3 sends
per cell, each cell the median of the valid sends. tok/s = tokens after the first streamed event / time from the first to the
last event. With MTP, read the median event interval (ITL) together with tokens per event, not tok/s alone.

The ladder's top rung is 259,557 prompt tokens with thinking on (259,616 off); the 262K decode figures are the exact 262K
rows below.

| prompt tokens | thinking | tok/s | ITL ms | tokens/event | TTFT s | valid |
|---|---|---|---|---|---|---|
| 4,046 | on | 143.7 | 19.60 | 2.78 | 0.84 | 3/3 |
| 32,566 | on | 127.5 | 20.34 | 2.56 | 7.09 | 3/3 |
| 130,386 | on | 105.7 | 22.22 | 2.31 | 48.95 | 3/3 |
| 259,557 | on | 89.6 | 24.37 | 2.13 | 151.76 | 3/3 |
| 4,105 | off | 105.8 | 19.61 | 2.07 | 0.86 | 3/3 |
| 32,625 | off | 103.4 | 20.28 | 2.09 | 7.12 | 3/3 |
| 130,445 | off | 98.7 | 22.14 | 2.16 | 48.89 | 3/3 |
| 259,616 | off | 84.2 | 24.31 | 2.03 | 151.78 | 3/3 |

## 2. Exact 262K

Exactly 261,888 prompt tokens + 128 out, both thinking modes, 3 sends (sends 1 and 2 had nothing cached; send 3 reused
133,632 tokens from the prefix cache). The table gives the median decode and send 1's TTFT.

| prompt tokens | thinking | decode tok/s (median of 3) | ITL ms | tokens/event | cold TTFT s | needle found |
|---|---|---|---|---|---|---|
| 261,888 | on | 99.8 | 24.43 | 2.33 | 153.7 | — (see note) |
| 261,888 | off | 88.8 | 24.41 | 2.10 | 153.5 | 3/3 |

**262K needle.** Each prompt is exactly 261,888 tokens with a needle inside it; the thinking-off sends answer directly and the
needle column counts the sends whose answer contains it. The thinking-on sends stop at 128 tokens inside the reasoning,
before any answer, so they measure decode speed only.

## 3. Cold prefill

| prompt tokens | prefill | TTFT s |
|---|---|---|
| 10,157 | 5,148 tok/s | 1.97 |
| 101,845 | 3,049 tok/s | 33.40 |
| 261,888 | 1,704 tok/s | 153.7 |

Prefill is one cold send per depth (thinking on, 8 tokens out); the 262K row is the exact-262K run's cold send 1.

Prefill over context depth: prompt tokens / median cold TTFT of the thinking-on ladder cells (3 sends each), and the
exact-262K row's cold send.

| prompt tokens | prefill | TTFT s |
|---|---|---|
| 4,046 | 4,791 tok/s | 0.84 |
| 32,566 | 4,590 tok/s | 7.09 |
| 130,386 | 2,664 tok/s | 48.95 |
| 259,557 | 1,710 tok/s | 151.76 |
| 261,888 | 1,704 tok/s | 153.7 |

## 4. Prompts sent at once

| prompt | prompt tokens | first token s | needle found |
|---|---|---|---|
| 1 | 261,338 | 155.5 | yes |
| 2 | 259,261 | 305.0 | yes |
| 3 | 261,565 | 453.9 | yes |

Three ~262K needle prompts (259,261–261,565 tokens, each with its own needle, 128 tokens out) sent at once. The engine ran
all three (3 running at the peak, peak KV pool use 100 %, errors 0, 455.4 s wall) and found every needle. First tokens came
about one 262K prefill apart, and each reply finished before the next prompt's first token, so the decode windows do not
overlap and no combined decode rate is given.

This is not the residency test of the other cards: nothing was re-sent from the prefix cache, so it does not show that the
pool held all three sessions together. The 894,618-token pool is 3.41 × 262,144 tokens (3 × 262K by the engine's fit);
6 × 131K is only the pool ratio (6.83 × 131,072), not a measured load.

## 5. Images

| case | lowest free memory on any card |
|---|---|
| one 8 MP image (8,388,608 px) with a 262K prompt | 1,058 MiB |

Measured with uniform MoE verify off, on the same install and KV pool, when the pool size was set; the full card ran with it
on, and its own image headroom was not measured separately. At ready, the full-card boot had 2,067 / 2,067 / 1,307 / 1,307 MiB
free per card. The two-colour answer check, the two-requests-at-once case and the four-image case of the other cards were not
run for this checkpoint. The profile allows up to 8 images per prompt.

## 6. Agents, agent-trace replay, GSM8K and needles

Source: agents and the second GSM8K-200, the released 3.0.0 install, 2026-10-21; agent-trace replay, the first
GSM8K-200 and the needle panel, the full-card setup above, 2026-10-21. Thinking off unless stated.

**Agents at once** (N 16K conversations, then N simultaneous 512-token follow-ups; total tok/s while all N decode)

| agents | total tok/s | overlap window s |
|---|---|---|
| 2 | 167.22 | 5.56 |
| 4 | 274.93 | 6.96 |
| 8 | 467.92 | 7.79 |

Errors 0. A short overlap window (a few seconds) means the replies ended before all N overlapped for long; read those rows loosely.

**Agent-trace replay** (a recorded tool-use session replayed, 27 requests): 52/52 tool calls valid, 0 errors; decode median
148.559 tok/s, TTFT median 0.637 s, prefix-cache hit 99.9 % of the ideal (median).

**GSM8K-200** (5-shot, thinking off): 190/200 correct; 190/200 again on the released 3.0.0 install.

**Needle panel** (single-occurrence needles, thinking off, cold cache): 10/10 at 131K (130,663-token prompts), 23/23 at 262K
(258,616-token prompts) found exactly.

## 7. Without P2P

A second host: 4× RTX 3090, PCIe 4.0 x8, P2P off, 2026-10-21, on the same engine tree as the full card (the 3.0.0 release
install with Ornith's engine changes and the same unshipped prefill experiment), profile `ornith-quad-3090-nop2p`.

| | without P2P (second host) | with P2P (reference host) |
|---|---|---|
| KV pool | 977,838 tokens (3.73 × 262K; 3 × 262K by the engine's fit) | 894,618 tokens |
| lowest free memory, 8 MP image + 262K prompt (uniform verify off) | 1,057 MiB | 1,058 MiB |
| decode tok/s at 4K / 32K / 131K / ~262K, median per request | 118.2 / 113.8 / 100.6 / 80.0 | 137.2 / 123.9 / 109.4 / 84.7 |
| GSM8K-200 | 193/200 | 190/200 |

The decode row is a short per-request decode set run on the GSM8K boot, not the ladder of section 1. Both columns ran with
uniform MoE verify on. Sessions held at once were not run on the second host.
