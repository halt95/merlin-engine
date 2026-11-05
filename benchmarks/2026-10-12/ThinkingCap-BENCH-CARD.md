# Bench card, ThinkingCap-Qwen3.8-27B profile of v1.3.0: dual card (TP=2) and quad card (TP=4)

`halt95/ThinkingCap-Qwen3.8-27B-W4A16-Merlin` = our W4A16 quantisation of BottleCap AI's ThinkingCap-Qwen3.8-27B, served by
this repository's engine, patch series and scripts through `serve/serve-thinkingcap.sh` (TP=2) and
`serve/serve-thinkingcap-tp4.sh` (TP=4), which change only the default reasoning effort (`xhigh`). **Only rows measured on this
checkpoint appear in this card;** nothing here is inferred from the base checkpoint's results in
[`BENCH-CARD.md`](Qwen3.8-27B-BENCH-CARD.md) (the Qwen3.8-27B v1.3.0 card, same folder), which this card cites for comparison only.

**Reference host.** 4x RTX 3090 (24 GB, sm_86), PCIe Gen4 x16 to every card, P2P-enabled driver 610.43.02, 220 W cap. The dual card profile uses one pair of the four cards.

**Units.** Context depths are prompt targets: 4K = 4,096; 32K = 32,768; 128K = 131,072; 255K = 261,120 tokens. The actual prompt lengths are in the rows. "cN" = N concurrent streams. GB = 10^9 bytes, GiB = 2^30 bytes.

**Evidence.** Every figure is a maintainer measurement on the reference host, by the harnesses named in each section; the raw
run artifacts are not published. Measured 2026-10-08 to 2026-10-12 (UTC); where each row was measured:

- **2026-10-08 gate** (sections 1-2 and the gate rows of 4-6): dual card, the base profile's `serve/serve.sh` with its `medium`
  default (each harness sets its own per-request effort), the build directory's earlier `tokenizer.json`; 2026-10-08
  21:05-22:52 UTC. The section 1 thinking-token run ran on tree 2 of the Qwen3.8-27B card's section 7.
- **2026-10-09** (section 7 and the quad rows of sections 4-5): dual and quad card with the `xhigh` default, on an earlier tree (before the final patch revision), the upstream `tokenizer.json` staged.
- **2026-10-11** (sections 3 and 8, and the 2026-10-11 column of section 5): tree 5's engine, patches and `serve/serve.sh` with
  `THINKING` at the `xhigh` default (what the wrappers set), dual card only.
- **2026-10-12** (the quad card rows of sections 3, 4 and 8, and section 9): this release's tree (Merlin engine v2.5.2) freshly installed with
  `release/install-env.sh`, the checkpoint as published (`models/thinkingcap/checkpoint.sha256` 11 of 11, upstream
  `tokenizer.json`), served through `serve/serve-thinkingcap-tp4.sh` and `serve/serve-thinkingcap.sh` with no `THINKING` set.

## 1. Thinking tokens against the base checkpoint

**Design.** GSM8K test items, the same items for both arms, a per-item seed, served sampling, the harness sets the reasoning
effort on every request. ThinkingCap ran on one card pair and the Qwen3.8-27B v1.3.0 checkpoint on the other, each on the
shipped `serve/serve.sh`. Thinking tokens = completion tokens minus the answer's tokens (they include the closing marker).
**Time to answer** = the time to the first answer token. Because the two arms ran on different card pairs (which differ by about
5 % in speed), **time is indicative; the token counts are what decides.** The accuracy guardrail is non-inferiority within 2
percentage points.

| effort | arm | items | accuracy | truncated | thinking tokens, median / mean | median time to answer | median wall | MTP tokens/step |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| medium | Qwen3.8-27B v1.3.0 | 1,319 | 97.27 % | 0 | 157 / 205.9 | 3.7 s | 6.9 s | 4.19 |
| medium | ThinkingCap | 1,319 | 97.27 % | 0 | 135 / 170.2 | 3.3 s | 6.0 s | 4.22 |
| `xhigh` | Qwen3.8-27B v1.3.0 | 400 | 96.75 % | 0 | 140 / 458.4 | 3.8 s | 5.9 s | 3.48 |
| `xhigh` | ThinkingCap | 400 | 97.75 % | 0 | 81 / 176.4 | 2.8 s | 4.2 s | 3.57 |

- **Medium, paired** (ThinkingCap right and base wrong on 10, the reverse on 10): accuracy +0.00 pp, non-inferiority at 2 pp
  PASS (one-sided exact p = 0.001); thinking tokens, ratio of paired medians **x0.860** (95 % CI 0.840-0.879), median paired
  difference -12 tokens, Wilcoxon p 3.01e-94; time to answer x0.891 (0.871-0.915), indicative only.
- **`xhigh`, paired** (5 and 1): accuracy +1.00 pp, non-inferiority at 2 pp report-only (no pre-registered cell): one-sided exact
  p = 0.004; thinking tokens **x0.579** (0.533-0.625), median paired difference -50 tokens, Wilcoxon p 2.65e-60; time to answer
  x0.754 (0.692-0.796), indicative only.
- The `xhigh` cell is a report-only cell with its own item count. The comparison tool marks the accuracy guardrail
  "provisional" because the repetition screen is not one of its inputs; the screen's own result is in section 5. The mean is the mean
  thinking tokens per item (the table's median is rounded by the tool). One task family (GSM8K-style arithmetic word problems),
  one sampling setup; the saving on code, long-context or agentic work was not measured.

## 2. Single stream over context depth (dual card, 2026-10-08 gate)

![ThinkingCap-Qwen3.8-27B over context depth: cold prefill, single-stream decode and decode step time, dual and quad card](thinkingcap-v1.3.0-ctx-pp-tg-itl.png)

Medians of 3 cold requests per depth, thinking on (effort `low`, 256-token outputs): dual card from this section's ladder
(2026-10-08 gate), quad card from section 3's ladder (2026-10-12). Prefill = prompt tokens / time to first token; decode moves
with how many drafted tokens each text accepts (the 255K point accepts more, 4.5-4.8 per step); step time is per decode step.

Same harness and method as the Qwen3.8-27B card's section 1 (c1, 3 cold requests per depth and mode; thinking on: effort `low`, `max_tokens` 256;
thinking off: `max_tokens` 1,024, `min_tokens` 200). The base columns are the gate's own dual card ladder (the Qwen3.8-27B card, section 1), run on
the same card pair. tokens/step follows what each text accepts, so the ITL column is the one to compare.

**Thinking on**

| depth | prompt tokens | ThinkingCap ITL | base ITL | ITL ratio | ThinkingCap tok/step | base tok/step | ThinkingCap decode | base decode | ThinkingCap TTFT | base TTFT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4K | 4,076 | **20.0 ms** | **20.0 ms** | 1.00x | 4.12 | 4.06 | 211.3 tok/s | 205.6 tok/s | 2.3 s | 2.3 s |
| 32K | 32,596 | **21.5 ms** | **21.3 ms** | 1.01x | 3.62 | 4.11 | 172.8 tok/s | 197.8 tok/s | 20.5 s | 20.3 s |
| 128K | 130,416 | **25.4 ms** | **25.4 ms** | 1.00x | 3.53 | 4.03 | 149.9 tok/s | 171.6 tok/s | 112.2 s | 112.6 s |
| 255K | 259,587 | **31.0 ms** | **31.1 ms** | 0.99x | 4.54 | 4.31 | 157.7 tok/s | 144.3 tok/s | 302.3 s | 303.9 s |

**Thinking off**

| depth | prompt tokens | ThinkingCap ITL | base ITL | ITL ratio | ThinkingCap tok/step | base tok/step | ThinkingCap decode | base decode | ThinkingCap TTFT | base TTFT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4K | 4,105 | **20.1 ms** | **19.9 ms** | 1.01x | 2.80 | 2.96 | 140.1 tok/s | 149.3 tok/s | 2.4 s | 2.3 s |
| 32K | 32,625 | **21.4 ms** | **21.3 ms** | 1.00x | 2.69 | 2.85 | 126.3 tok/s | 134.8 tok/s | 20.5 s | 20.4 s |
| 128K | 130,445 | **25.2 ms** | **25.3 ms** | 1.00x | 2.68 | 2.95 | 107.7 tok/s | 119.6 tok/s | 112.3 s | 112.1 s |
| 255K | 259,616 | **30.6 ms** | **30.7 ms** | 1.00x | 2.84 | 2.70 | 93.6 tok/s | 89.0 tok/s | 303.7 s | 302.8 s |

Output lengths (completion tokens, 3 requests): thinking on: 4K [164, 114, 165]; 32K [169, 170, 169]; 128K [120, 120, 125]; 255K
[114, 119, 118]; thinking off: 4K [400, 310, 370]; 32K [308, 377, 295]; 128K [388, 366, 299]; 255K [396, 425, 403]. Requests
ending at the token cap: 0 of 12 thinking-on, 0 of 12 thinking-off.

The step time equals the base checkpoint's; fewer drafted tokens are accepted per step at 32K and 128K, so decode there is lower
than the base's. n = 3 per row.

## 3. Decode on the shipped engine (dual card 2026-10-11, quad card 2026-10-12)

One fresh-cache boot per profile at the `xhigh` default (dual: tree 5's `serve/serve.sh` with `THINKING` set; quad:
`serve/serve-thinkingcap-tp4.sh`):

| | dual card (TP=2) | quad card (TP=4) |
|---|---|---|
| single stream at 32K (the section 2 ladder: effort `low` per request, 256-token outputs, 3 cold requests) | **172.9-180.2 tok/s** (3.62-3.76 tokens per step) | **223.4-227.6 tok/s** (3.62 tokens per step, median) |
| mean acceptance over that ladder (the engine's speculative-decoding windows) | 3.43 tokens per step | 3.53 tokens per step |
| mean acceptance over the same ladder at `xhigh` per request with 1,024-token outputs | 3.55 tokens per step | 3.57 tokens per step |
| replay of two recorded agent sessions (35 and 40 tool-calling turns, prompts growing from about 13K to 94K tokens, original output lengths): effective rate, the prefill of each turn included | **113.6** and **112.3 tok/s** | **156.6** and **148.3 tok/s** (median TTFT 0.95 s / 1.11 s, median decode 228.8 / 213.9 tok/s) |

**Quad card ladder over context depth** (2026-10-12; same harness, requests and method as section 2, effort `low`, 256-token
outputs; medians of 3 cold requests; the base columns are the Qwen3.8-27B card's section 1 quad card ladder):

| depth | ThinkingCap ITL | base ITL | ThinkingCap tok/step | base tok/step | ThinkingCap decode | base decode | ThinkingCap TTFT | base TTFT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 4K, thinking on | **15.8 ms** | **15.7 ms** | 4.00 | 4.00 | 257.1 tok/s | 257.8 tok/s | 1.6 s | 1.6 s |
| 32K, thinking on | **16.7 ms** | **16.6 ms** | 3.62 | 4.04 | 223.4 tok/s | 248.9 tok/s | 13.7 s | 13.7 s |
| 128K, thinking on | **18.4 ms** | **18.3 ms** | 3.56 | 4.14 | 201.5 tok/s | 238.2 tok/s | 70.2 s | 69.8 s |
| 255K, thinking on | **20.8 ms** | **20.9 ms** | 4.75 | 4.49 | 251.5 tok/s | 226.0 tok/s | 179.7 s | 179.3 s |
| 4K, thinking off | **15.8 ms** | **15.8 ms** | 2.97 | 2.83 | 188.4 tok/s | 180.7 tok/s | 1.6 s | 1.6 s |
| 32K, thinking off | **16.7 ms** | **16.7 ms** | 2.95 | 2.84 | 178.6 tok/s | 172.6 tok/s | 13.7 s | 13.7 s |
| 128K, thinking off | **18.6 ms** | **18.4 ms** | 2.80 | 2.98 | 152.5 tok/s | 164.5 tok/s | 70.3 s | 70.0 s |
| 255K, thinking off | **20.8 ms** | **20.8 ms** | 2.75 | 2.86 | 133.7 tok/s | 140.2 tok/s | 179.5 s | 179.2 s |

The step time matches the base checkpoint within 1 % at every depth, as on dual card; decode differs only through the tokens
each text accepts per step. Output lengths (completion tokens): thinking on 4K [168, 168, 165], 32K [169, 170, 170], 128K [171,
122, 120], 255K [114, 118, 114]; thinking off 4K [370, 341, 330], 32K [358, 345, 296], 128K [394, 412, 392], 255K [420, 454,
374]. Cold prefill: 10,187 tokens in 4.1 s (2,461 tok/s), 101,875 tokens in 51.3 s (1,986 tok/s). The same ladder at `xhigh`
with 1,024-token outputs: thinking on 243.9 / 259.2 / 256.3 / 225.0 tok/s at 4K / 32K / 128K / 255K (3.83 / 4.19 / 4.49 / 4.51
tokens per step), thinking off 179.8 / 178.8 / 151.8 / 143.0 tok/s. 0 invalid requests in either ladder.

## 4. 262K needle, sessions resident at once and eight agent-style conversations

| profile | test | result | wall |
|---|---|---|---:|
| dual (gate) | 262K needle, 261,568-token cold prompt | **pass** | 306.9 s |
| dual (gate) | 3 x 261,600-token sessions, 512 generated each | 3 of 3 codes, 3 of 3 cold prefills, **0 preemptions**, peak KV pool 95.4 % | 920 s |
| dual (gate) | 6 x 130,800-token sessions, 512 generated each | 6 of 6 codes, 6 of 6 cold prefills, **0 preemptions**, peak KV pool 99.9 % | 686 s |
| quad (2026-10-09) | 7 x 261,600-token sessions, 512 generated each | 7 of 7 correct, **0 preemptions**, peak KV pool 90.6 % | 1,242 s |
| quad (2026-10-12) | 262K needle, 261,632-token cold prompt | **pass** | 180.0 s |

Eight agent-style conversations (dual, gate; same load and definitions as the Qwen3.8-27B card's section 3): 8 conversations, 1,362 s window, 206
requests (0 errors), largest prompt 143,793 tokens, prefix-cache hit rate **84.1 %**, TTFT p50 / p99 9.8 s / 283.8 s, 0
preemptions, peak KV pool usage 100.0 %; multi-turn replay first turn 22.9 s (35,868 tokens), median follow-up **2.79 s**.

Eight agent-style conversations (quad, 2026-10-12; same load generator and settings as the base checkpoint's 2026-10-11 quad
run in the Qwen3.8-27B card's section 11: 8 conversations, 1,200 s, prefix seed 22): 443 requests (0 errors, 0 engine errors), prefix-cache hit rate
**96.5 %**, TTFT p50 / p99 5.5 s / 101.5 s, 0 preemptions, peak KV pool usage 80.9 %; multi-turn replay median follow-up
**1.80 s**.

## 5. Quality gates

| | dual, gate (`medium` default) | dual, 2026-10-11 (`xhigh` default) | quad, 2026-10-09 (`xhigh` default) |
|---|---|---|---|
| GSM8K-200 (as served) | **195 / 200** (0 errors, 4 cut at the token cap; the base checkpoint at the same gate 196: ThinkingCap-only right 0, base-only right 1, exact McNemar p = 1) | **195 / 200** (0 errors, 3 cut at the token cap) | **193 / 200** |
| loop gate, 72 generations (12 x 6), c6: engine errors / loose flags / genuine collapses / length-capped | 0 / 0 / **0** / 0 | 0 / 0 / **0** / 6 | 0 loopers, **0** genuine collapses |
| loop gate: mean repeat-8, worst 12-token window repeat | 0.0016, 1 | 0.001, 3 | - |
| 262K needle | pass (section 4) | pass, 310 s | - |

Loop protocol as in the Qwen3.8-27B card's section 4. The thinking-token run's truncation counts (section 1: 0 of 1,319 at medium, 0 of 400 at `xhigh`) are
the long-thought check.

## 6. KV pool, memory and boot (2026-10-08 gate)

| | dual (boot 1) | dual (boot 2) |
|---|---:|---:|
| KV pool, attention block | **826,689 tokens**, 960 | **826,689 tokens**, 960 |
| boot to ready (fresh compile cache) | 392 s | 407 s |
| idle / peak VRAM per card | 23,474 / 23,622 MiB | 23,474 / 23,702 MiB |
| tile proof lines (ranks) | 2 of 2 | 2 of 2 |
| patch proof lines (embed / plan cache / split-K / seq-lens), RecoverSSM proof lines | 2 / 100 / 2 / 2, 32 | 2 / 100 / 2 / 2, 32 |
| boot errors | 0 | 0 |

A dual card boot of tree 2 (the Qwen3.8-27B card, section 7) with this checkpoint printed the tile proof line on 2 of 2 ranks with the same
pool and answered its first generation.

## 7. Smokes, tokenizer, images and quad card (2026-10-09, earlier tree, `xhigh` default)

| | dual card (TP=2) | quad card (TP=4) |
|---|---|---|
| KV pool, attention block | 826,689 tokens, 960 | 1,974,072 tokens, 848 |
| tile proof line | 2 of 2 ranks | 4 of 4 ranks |
| boot to ready, peak VRAM per card | 392 s, 23,844 MiB | 392 s, 23,552 MiB |
| smokes: tool call, reasoning split, vision, JSON-schema output, warm prefix | all passed | - |
| served tokenizer against the upstream one | identical on 153 of 153 items (the frozen combining-mark set), controls included | - |
| images | one 16 MP image: HTTP 200 in 3.7 s; eight distinct 4 MP images: HTTP 200 in 28.1 s | - |

A dash means no row was recorded for that profile in this run.

## 8. Tree 5 boot (dual card, 2026-10-11, `xhigh` default)

Fresh compile cache, on one card pair while the base checkpoint ran the Qwen3.8-27B card's section 11 on the other: pool 826,689 tokens (block 960, as
frozen), boot to ready 438 s, tile proof once on each of devices 0-1, patch proof lines 2 / 100 / 2 / 2 / 32 (embed / plan
cache / split-K / seq-lens / RecoverSSM), 0 boot errors, 0 errors at shutdown. Its GSM8K-200, needle and loop rows are in section 5.

Through the wrappers (2026-10-12, fresh compile caches, no `THINKING` set): `serve/serve-thinkingcap-tp4.sh` booted in 391 s
(pool 1,974,072 tokens) and `serve/serve-thinkingcap.sh` in 391 s (pool 826,689 tokens), both with the `xhigh` default in the
server's chat-template defaults. Smokes (warm prefix 8,384 of 8,411 tokens hit on the repeat, tool call, reasoning split, vision
4 of 4) passed on both. Idle after the last request: all cards in P8, 23-33 W per card.

## 9. Eight agents at once (2026-10-12)

Eight agents run concurrently, each replaying one of the two recorded Hermes agent sessions of section 3 turn by turn (four copies
of each; each copy's first user message carries a unique tag, so all eight share the system prompt and tool schema and diverge
after it). Every turn is re-sent as the agent sent it, with the original output length (capped at 4,000 tokens), through the
serving proxy exactly as an agent client reaches the server: `serve/serve-thinkingcap.sh` (dual card) and
`serve/serve-thinkingcap-tp4.sh` (quad card), this release's tree, the checkpoint as published. 300 turns per profile, 0 errors.
Unlike the eight conversations of section 4 (a load test that grows each conversation until the KV pool is under pressure),
this measures how fast real agent sessions run when eight share the server.

| | dual card (TP=2) | quad card (TP=4) |
|---|---:|---:|
| **total output, all eight agents** | **215.7 tok/s** | **333.5 tok/s** |
| per agent, effective (median / slowest; the prefill of each turn included) | 28.1 / 26.6 tok/s | 42.6 / 41.1 tok/s |
| per-turn decode, median | 32.5 tok/s | 50.8 tok/s |
| turn TTFT p50 / p90 | 1.95 s / 8.5 s | 1.58 s / 4.93 s |
| prefix-cache hit (`cached_tokens` / prompt tokens) | 95.4 % | 95.4 % |
| time for all eight sessions | 1,297 s | 839 s |
| one agent alone (section 3) | 113.6 and 112.3 tok/s | 156.6 and 148.3 tok/s |

Eight agents give about twice the total output of one (x1.9 dual, x2.2 quad), each agent at about a quarter of the single-agent
rate; the eight agents' rates differ by under 7 % (dual 26.6-28.4, quad 41.1-43.7 tok/s), and no request was preempted.

## 10. Not measured on this checkpoint

The club-3090 method (the Qwen3.8-27B card, section 9). At `xhigh` an answer can be longer, which means more tokens, a longer context and more KV
pressure; the per-step figures above were measured at the stated output lengths and do not by themselves say how a long `xhigh`
answer changes speed or residency.

## Notes and caveats

- The notes of the Qwen3.8-27B card ([`BENCH-CARD.md`](Qwen3.8-27B-BENCH-CARD.md#notes-and-caveats)) apply here too: reference host only
  (RTX 3090, PCIe Gen4 x16, P2P), three requests per single-stream cell, tokens per step counted per streamed chunk.
