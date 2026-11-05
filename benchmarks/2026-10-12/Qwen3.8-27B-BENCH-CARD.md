# Bench card, v1.3.0: dual card (TP=2) and quad card (TP=4)

v1.3.0 = Merlin engine v2.5.2 (vLLM 0.30.0 plus the Flash-Next fork's commits and the guard fix) + the 27B patch series + a FlashInfer 32-row verify tile, MTP K=4, served by the shipped `serve/serve.sh` (dual card, TP=2) and `serve/serve-tp4.sh` (quad card, TP=4) with no knob set.

**Reference host.** 4x RTX 3090 (24 GB, sm_86), PCIe Gen4 x16 to every card, P2P-enabled driver 610.43.02, 220 W cap. The dual card profile uses one pair of the four cards.

**Served as shipped.** The dual card line (`serve/serve.sh`):

- thinking on, default effort **medium** (the server default; the per-request efforts used by each harness are stated in its section);
- server-side sampling: T 1.0 / top-p 0.95 / top-k 20 (a server-side override);
- FP8 E4M3 KV with the checkpoint's calibrated scales (fp8_e4m3), fp16 SSM state (float16);
- MTP with K=4 draft tokens per step, `--max-num-seqs 16 --max-num-batched-tokens 2048 --long-prefill-token-threshold 832 --prefix-match-unit 16`, prefix caching with the fine-grained mamba cache, RecoverSSM on;
- `--gpu-memory-utilization 0.96` with the KV pin of the shipped script;
- cold nonce prompts in every single-stream cell.

- **Quad card (`serve/serve-tp4.sh`)** differs in: `--tensor-parallel-size 4`, `--max-num-seqs 24`, KV pin 18,231,050,240 B per GPU, no RecoverSSM; everything else as above.

**Units.** Context depths are prompt targets: 4K = 4,096; 32K = 32,768; 128K = 131,072; 255K = 261,120 tokens. The actual prompt lengths are in the rows. "cN" = N concurrent streams. GB = 10^9 bytes, GiB = 2^30 bytes.

**Evidence.** Every figure is a maintainer measurement on the reference host, served as shipped, by the harnesses named in each section. The results and the setup are documented here; the raw run artifacts (rows, harnesses, gate records, boot logs) are not published. **Measured** (2026-10-08 to 2026-10-09, UTC): dual card 2026-10-08 20:23-22:20; quad card 2026-10-08 22:52 to 2026-10-09 00:23.

> **Trees.** Sections 1-6 were measured on tree 1; the later trees and what each re-checked are in section 7.
>
> **ThinkingCap.** The ThinkingCap checkpoint profile (`serve/serve-thinkingcap*.sh`) has its own card in this folder:
> [`ThinkingCap-BENCH-CARD.md`](ThinkingCap-BENCH-CARD.md).

## At a glance

|  | dual card (TP=2) | quad card (TP=4) |
|---|---:|---:|
| KV pool, shipped pin | **826,689** tokens (block 960) | **1,974,072** tokens (block 848) |
| full 262K sessions resident, 0 preemptions | 3 (preemptions 0) (6 x 131K, preemptions 0) | 7 (preemptions 0) |
| single-stream decode step (thinking on), 4K -> 255K | 20.0 -> 31.1 ms | 15.7 -> 20.9 ms |
| tokens per decode step (thinking on) | 4.06 / 4.31 (4K / 255K) | 4.00 / 4.49 (4K / 255K) |
| single-stream decode (thinking on), 4K -> 255K | 205.6 -> 144.3 tok/s | 257.8 -> 226.0 tok/s |
| single-stream decode (thinking off), 4K -> 255K | 149.3 -> 89.0 tok/s | 180.7 -> 140.2 tok/s |
| cold prefill, 4K / 255K prompt (prompt tokens / TTFT) | 1,766 / 854 tok/s | 2,555 / 1,448 tok/s |
| cold 262K needle | pass, 307 s | pass, 181 s |
| 8-conversation load: prefix-cache hit rate, TTFT p50 / p99 | 92.6 %, 9.0 / 282.9 s | 88.6 %, 7.7 / 101.5 s |
| GSM8K-200 | 196 / 200 | 194 / 200 |
| loop gate: genuine collapses | 0 of 72 | 0 of 72 |
| peak VRAM per card over the gate steps | 23,844 MiB | 23,736 MiB |

## 1. Single stream over context depth

- c1, 3 requests per depth and mode; every request is a cold prefill (a distinct cache salt per request; the harness also records the API's cached-token count, which the gate's serve line did not report (the shipped scripts add `--enable-prompt-tokens-details`), so the cold property rests on the salt).
- **Thinking on:** the harness requests effort `low` with `max_tokens` 256, so short answers finish and long ones hit the cap. **Thinking off:** `enable_thinking` false, `max_tokens` 1,024, `min_tokens` 200.
- Median ITL = the median over the 3 requests of each request's median interval between streamed chunks (one chunk = one decode step). tokens/step = tokens per streamed chunk. Decode = tokens after the first chunk / time from the first to the last chunk (median of 3); it moves with how many drafted tokens each text accepts, so ITL is the steadier figure.
- TTFT = median of 3. Prefill = prompt tokens / median TTFT (it includes scheduling).

![Qwen3.8-27B v1.3.0 over context depth: cold prefill, single-stream decode and decode step time, dual and quad card](qwen38-27b-v1.3.0-ctx-pp-tg-itl.png)


**Thinking on**

| depth | prompt tokens | dual ITL | quad ITL | step-time ratio (dual / quad) | dual tok/step | quad tok/step | dual decode | quad decode | dual TTFT | quad TTFT | dual prefill | quad prefill |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4K | 4,076 | **20.0 ms** | **15.7 ms** | 1.27x | 4.06 | 4.00 | 205.6 tok/s | 257.8 tok/s | 2.3 s | 1.6 s | 1,766 tok/s | 2,555 tok/s |
| 32K | 32,596 | **21.3 ms** | **16.6 ms** | 1.28x | 4.11 | 4.04 | 197.8 tok/s | 248.9 tok/s | 20.3 s | 13.7 s | 1,608 tok/s | 2,373 tok/s |
| 128K | 130,416 | **25.4 ms** | **18.3 ms** | 1.39x | 4.03 | 4.14 | 171.6 tok/s | 238.2 tok/s | 112.6 s | 69.8 s | 1,158 tok/s | 1,868 tok/s |
| 255K | 259,587 | **31.1 ms** | **20.9 ms** | 1.49x | 4.31 | 4.49 | 144.3 tok/s | 226.0 tok/s | 303.9 s | 179.3 s | 854 tok/s | 1,448 tok/s |

**Thinking off**

| depth | prompt tokens | dual ITL | quad ITL | step-time ratio (dual / quad) | dual tok/step | quad tok/step | dual decode | quad decode | dual TTFT | quad TTFT | dual prefill | quad prefill |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4K | 4,105 | **19.9 ms** | **15.8 ms** | 1.26x | 2.96 | 2.83 | 149.3 tok/s | 180.7 tok/s | 2.3 s | 1.6 s | 1,765 tok/s | 2,543 tok/s |
| 32K | 32,625 | **21.3 ms** | **16.7 ms** | 1.28x | 2.85 | 2.84 | 134.8 tok/s | 172.6 tok/s | 20.4 s | 13.7 s | 1,599 tok/s | 2,374 tok/s |
| 128K | 130,445 | **25.3 ms** | **18.4 ms** | 1.37x | 2.95 | 2.98 | 119.6 tok/s | 164.5 tok/s | 112.1 s | 70.0 s | 1,164 tok/s | 1,863 tok/s |
| 255K | 259,616 | **30.7 ms** | **20.8 ms** | 1.48x | 2.70 | 2.86 | 89.0 tok/s | 140.2 tok/s | 302.8 s | 179.2 s | 858 tok/s | 1,449 tok/s |

Output lengths (completion tokens, 3 requests):

- dual, thinking on: 4K [256, 256, 256]; 32K [218, 141, 200]; 128K [143, 141, 139]; 255K [181, 181, 184].
- dual, thinking off: 4K [334, 316, 395]; 32K [387, 299, 319]; 128K [301, 301, 301]; 255K [417, 421, 451].
- quad, thinking on: 4K [256, 256, 256]; 32K [218, 200, 141]; 128K [143, 174, 211]; 255K [181, 184, 184].
- quad, thinking off: 4K [361, 379, 291]; 32K [386, 368, 292]; 128K [302, 301, 301]; 255K [338, 385, 381].
- Requests that ended at the token cap (`length`) instead of a natural stop: dual 3 of 12 thinking-on and 0 of 12 thinking-off; quad 3 of 12 and 0 of 12.

Two more cold-prefill points (one request each, 8-token output):

| profile | depth | prompt tokens | TTFT | prefill |
|---|---|---:|---:|---:|
| dual | 10K target | 10,187 | 5.9 s | 1,739 tok/s |
| dual | 100K target | 101,875 | 80.4 s | 1,267 tok/s |
| quad | 10K target | 10,187 | 4.1 s | 2,483 tok/s |
| quad | 100K target | 101,875 | 51.2 s | 1,989 tok/s |

## 2. 262K needle and sessions resident at once

- **Needle:** one cold prompt of about 262K tokens with an access code planted early and the question last; the answer's first line must be the code.
- **Residency:** N concurrent cold chat sessions, each prompt sized by the server's tokenizer, a code near the start and the question last, temperature 0, generation forced to the stated length. A metrics sampler records the peak `kv_cache_usage` and the change in the preemption counter.

| profile | prompt tokens | exact code returned | wall |
|---|---:|---:|---:|
| dual | 261,568 | **pass** | 307.1 s |
| quad | 261,568 | **pass** | 180.8 s |

| profile | sessions x prompt tokens | generated per session | correct access code | cold prefills | preemptions | peak KV pool usage | wall |
|---|---|---:|---:|---:|---:|---:|---:|
| dual, 3 x 262K | 3 x 261,600 | 512 | 3 of 3 | 3 of 3 | 0 | 100.0 % | 912 s |
| dual, 6 x 131K | 6 x 130,800 | 512 | 6 of 6 | 6 of 6 | 0 | 99.9 % | 679 s |
| quad, 7 x 262K | 7 x 261,600 | 512 | 7 of 7 | 7 of 7 | 0 | 100.0 % | 1,237 s |

## 3. Eight agent-style conversations

**Shape:** 8 concurrent conversations that share one tool/system prefix, each with its own seeded history, growing turn by turn until the KV pool is under pressure; one fixed duration. The hit rate is the mean of the engine's periodic prefix-cache hit-rate readings over the last 20 log intervals of the run. The multi-turn replay is a separate probe: two long sessions replayed turn by turn after a warm-up.

|  | dual | quad |
|---|---:|---:|
| conversations, metrics window | 8, 1,362 s | 8, 1,202 s |
| requests (errors) | 217 (0) | 466 (0) |
| largest prompt | 154,261 tokens | 234,185 tokens |
| prefix-cache hit rate | **92.6 %** | **88.6 %** |
| TTFT p50 / p99 | 9.0 s / 282.9 s | 7.7 s / 101.5 s |
| preemptions | 0 | 0 |
| peak KV pool usage | 99.9 % | 99.2 % |
| multi-turn replay: first turn TTFT (prompt) | 22.9 s (35,890 tokens) | 15.3 s (35,906 tokens) |
| multi-turn replay: median follow-up TTFT | **2.79 s** | **1.78 s** |

## 4. Quality and functional gates

**GSM8K-200** (the first 200 test items, as-served sampling, exact-number match):

| profile | correct | other |
|---|---:|---|
| dual | **196 / 200** | 0 errors, 0 without an answer, 3 cut at the token cap |
| quad | **194 / 200** | 0 errors, 0 without an answer, 4 cut at the token cap |

Dual against quad on identical questions: dual-only right 2, quad-only right 0 (of 200 common questions), exact McNemar p = 0.5.

**Repetition loops** (protocol: [`docs/loop-check.md`](https://github.com/halt95/qwen38-27b-3090s/blob/5faac4c452dbd95989f1d034d4cf251dd74d5457/docs/loop-check.md); loose screen: repeat-8 > 0.15, a 12-token window repeated 4 or more times, or tail diversity < 0.45; genuine collapse: tail diversity < 0.15 or a window repeated 20 or more times). MTP on, as served:

|  | dual | quad |
|---|---:|---:|
| generations (prompts x seeds), concurrency | 72 (12 x 6), c6 | 72 (12 x 6), c6 |
| engine errors | 0 | 0 |
| flagged by the loose screen | 0 | 0 |
| genuine collapses | **0** | **0** |
| length-capped | 0 | 0 |
| mean repeat-8 / mean tail diversity | 0.0015 / 0.912 | 0.0020 / 0.914 |
| worst 12-token window repeat | 2 | 2 |

**Smokes and images:**

- **dual:**
  - warm prefix: a prompt of 8,410 tokens sent twice; the second request hit 8,384 cached tokens; output identical: yes
  - tool call: pass; reasoning split: pass; vision (4 colour images): 4 of 4 correct
  - one 16 MP image (downscaled to the 4 MP cap): HTTP 200, 4,123 prompt tokens, 3.7 s; 8 distinct 4 MP images in one prompt: HTTP 200, 32,809 prompt tokens, 28.6 s
  - text decode while distinct 4 MP images are encoded back to back (3 streams, thinking off): 172.5 tok/s alone, 47.9 tok/s with the image load (72.3 % slower)
- **quad:**
  - warm prefix: a prompt of 8,413 tokens sent twice; the second request hit 8,384 cached tokens; output identical: yes
  - tool call: pass; reasoning split: pass; vision (4 colour images): 4 of 4 correct
  - one 16 MP image (downscaled to the 4 MP cap): HTTP 200, 4,123 prompt tokens, 2.8 s; 8 distinct 4 MP images in one prompt: HTTP 200, 32,809 prompt tokens, 19.9 s
  - text decode while distinct 4 MP images are encoded back to back (3 streams, thinking off): 235.8 tok/s alone, 61.5 tok/s with the image load (73.9 % slower)

## 5. KV pool, memory and boot

|  | dual (boot 1) | dual (boot 2) | quad |
|---|---:|---:|---:|
| KV pin per GPU (`KV_BYTES`) | 14,756,413,440 B | 14,756,413,440 B | 18,231,050,240 B |
| KV pool | **826,689 tokens** | **826,689 tokens** | **1,974,072 tokens** |
| attention block | 960 tokens | 960 tokens | 848 tokens |
| 262,144-token sessions the pool holds | 3.15x | 3.15x | 7.53x |
| boot to ready (fresh compile cache) | 403 s | 403 s | 393 s |
| idle VRAM per card, ready | 23,474 MiB | 23,474 MiB | 23,408 MiB |
| peak VRAM per card over the boot's steps | 23,702 MiB | 23,844 MiB | 23,736 MiB |
| tile proof lines (ranks) | 2 of 2 | 2 of 2 | 4 of 4 |
| host embedding table per GPU | 1.184 GiB | 1.184 GiB | 0.592 GiB |
| patch proof lines (embed / plan cache / split-K / seq-lens) | 2 / 100 / 2 / 2 | 2 / 100 / 2 / 2 | 4 / 208 / 4 / 4 |
| RecoverSSM proof lines | 32 | 32 | 0 |
| boot errors | 0 | 0 | 0 |

- The pool figure is vLLM's hybrid capacity (tokens of 262,144-token sessions it can hold, counting the attention blocks and the GDN state blocks), not blocks x block size.
- Every boot logs the expected pool and block next to the measured one: all three matched the frozen values.
- Boot time is a fresh-compile-cache boot (a warm cache is faster).

**Software:**

| piece | version |
|---|---|
| engine | Merlin engine v2.5.2 (Flash-Next v2.5.1, i.e. vLLM 0.30.0 plus the fork's commits, plus the guard fix) |
| 27B patch series | `patches/merlin-engine-v2.5.2/` 0001-0011, plus the FlashInfer 32-row verify tile patch. The measured trees carried the guard fix as series patch 0012 on Flash-Next v2.5.1; the final `vllm/` tree is identical |
| torch | 2.13.0 |
| flashinfer-python | 0.6.18.post1 |
| transformers | 5.18.0 |
| triton | 3.7.1 |
| NCCL | 2.29.7 |
| Python | 3.13 |
| CUDA toolkit | 13.3 |
| NVIDIA driver | 610.43.02 |

## 6. Verify tile off

The tile (`FLASHINFER_SM8X_FP8_SMALLQ=cta32`, default on) is switched off by setting the variable empty. One dual card boot with it off against the shipped boot, same scripts (the two boots ran on the two different card pairs of the host):

| boot | pool / block | boot to ready | tile proof lines | GSM8K-200 |
|---|---:|---:|---:|---:|
| tile on (shipped) | 826,689 / 960 | 403 s | 2 of 2 ranks | 196 / 200 |
| tile off | 826,689 / 960 | 397 s | none | 194 / 200 |

Tile on against tile off on identical questions: on-only right 2, off-only right 0 (of 200), exact McNemar p = 0.5. The off boot logged no tile line (0 matching lines). Decode speed with the tile off was not re-measured in this gate.

## 7. Later trees

Sections 1-6 were measured on tree 1. The commits after it were re-checked on the trees below; nothing here replaces a section 1-6 measurement.

| tree | changed since the one above | re-checked on it |
|---|---|---|
| 1. tree 1 | the serve configurations as frozen | everything in sections 1-6 |
| 2. tree 2 (identifiers renamed) | the tile's internal identifiers and proof-line text, container route removed, checksums and docs; one comment in `serve/serve.sh` | clean install, tile proof on every rank at dual and quad card (section 8); the ThinkingCap thinking-token run ([ThinkingCap card](ThinkingCap-BENCH-CARD.md), section 1) |
| 3. tree 3 (device-labelled proof) | device-labelled tile proof line, the scale argument behind the tile path, apply-script import and lock checks, RecoverSSM commit storage binding, the split-KV merge test's criterion | tile proof with device labels (dual, quad), RecoverSSM boot evidence, served tokenizer parity, GSM8K-200, greedy tokens against tree 2, the split-KV merge test, release audit tests, the club-3090 method (section 9) |
| 4. tree 4 (series 0012, first revision) | series patch 0012, first revision (idle quiesce of the guard reader), regenerated checksums | idle power on dual card (section 10, first revision) |
| 5. tree 5 (release) | series patch 0012, second revision (run-time stream checks, fatal-arm and lifecycle fixes on the same idle path), its tests; later only comments (serve-script headers, an installer comment), docs and checksums | idle power on dual and quad card (section 10, second revision); boot, tile proof, GSM8K-200, 262K needle, 8-conversation load, multi-turn replay, loop gate and smokes on dual and quad card (section 11) |

**Tree 3.** Clean install, shipped scripts, Qwen3.8-27B v1.3.0 checkpoint with the served upstream `tokenizer.json`:

| boot | KV pool, block | boot to ready | tile proof |
|---|---:|---:|---:|
| dual card | 826,689 tokens, block 960 (as frozen) | 400 s | 2 of 2 ranks, one device-labelled line each |
| quad card | 1,974,072 tokens, block 848 (as frozen) | 395 s | 4 of 4 ranks, one device-labelled line each |

- RecoverSSM boot evidence (dual card): 6 commit contexts, 0 stale contexts, 0 boot errors.
- Served tokenizer against the upstream `tokenizer.json`: **PASS** on the 153-item combining-mark set plus controls (9 sets).
- GSM8K-200 on this tree (**thinking off, temperature 0, 5-shot**, a different protocol from section 4's as-served run, so the two are not compared): **193 / 200**, 5 cut at the token cap.
- Greedy tokens (3 prompts x 512 tokens, temperature 0, thinking off): this tree against tree 2 on the same checkpoint: 0 of 3 identical. The control: tree 2 booted twice with fresh compile caches gave 0 of 3 identical across the two boots, and 3 of 3 (and 3 of 3) identical within each boot. Greedy output is exact within one compile cache and not across caches, even on the same tree, so cross-tree token equality is not a usable check.
- Split-KV merge test (the tile's own error against an fp64 reference): passes (44 cells ok, 0 failures).
- Release audit tests: tile-proof ranks 31 passed, 0 failed; patch-hash test **skipped** (it needs a wheel download that was not available on the host).

## 8. Tile proof on tree 2

After the identifier rename, a clean install of the release tree was booted from the shipped scripts and its own `release/check-tile-proof.sh` was run against each boot log:

| boot | tile proof lines | KV pool | attention block | first generation |
|---|---:|---:|---:|---|
| dual card | 2 of 2 ranks | 826,689 tokens | 960 tokens | answered (finish `stop`) |
| quad card | 4 of 4 ranks | 1,974,072 tokens | 848 tokens | answered (finish `stop`) |

## 9. Decode throughput by the club-3090 method


**Method.** The public club-3090 `bench.sh` narrative + code decode test (our re-implementation; method version `3fd0fab`): two very short prompts (an 8-word narrative request and an 11-word code request), `max_tokens` 1,000 (narrative) and 800 (code), sampler T 0.6 / top-p 0.95 / top-k 20 sent with each request, **3 warm-up runs then 5 measured runs per prompt**, mean (and CV) of the 5. **Decode TPS = completion tokens / (wall - TTFT)**, TTFT being the first answer token; wall TPS = completion tokens / wall. The canonical profile is **thinking off, no effort** (`enable_thinking` false). Measured on the shipped `serve/serve.sh` (dual card) and `serve/serve-tp4.sh` (quad card) with no knob set, at a per-card power cap of **220 W**.

| tokens per second | dual narrative | dual code | quad narrative | quad code |
|---|---:|---:|---:|---:|
| **Canonical** (thinking off): decode TPS | **142.8** (CV 5.7 %) | **200.1** (CV 2.4 %) | **182.2** (CV 2.2 %) | **251.9** (CV 2.8 %) |
| Canonical: wall TPS | 141.3 (CV 5.6 %) | 195.7 (CV 2.3 %) | 179.8 (CV 2.1 %) | 244.1 (CV 3.7 %) |
| **Ours** (thinking on, effort medium): **wall TPS** | **140.1** (CV 1.7 %) | **169.7** (CV 3.0 %) | **179.9** (CV 1.4 %) | **218.0** (CV 2.1 %) |
| Canonical: TTFT (first answer token) | 74 ms | 79 ms | 70 ms | 72 ms |
| Ours: mean completion tokens (every measured run reached the cap) | 1,000 | 800 | 1,000 | 800 |

- The **ours** row is thinking on at effort `medium` and is quoted as **wall TPS**. That method's timer takes TTFT as the first *answer* token, so with thinking on it skips vLLM's reasoning stream (the first answer token arrives after about 1.4 s of reasoning on the dual card); a decode figure from that timer would overstate. Wall TPS includes prefill and the time to the first token. These are reasoning tokens plus answer tokens, capped at `max_tokens`.
- Short prompts, 5 measured runs: the CV column shows the run-to-run spread; the means are not a context-depth measurement (see section 1 for depth).


## 10. Idle power


Series patch 0012: the guard's counter reader stops polling the device once the engine is idle, and resumes with the next request. Hardware: 220 W per-card cap, driver 610.43.02.

**Second revision (the one shipped), dual and quad card** (tree 5's serve scripts, 2026-10-09; 20 requests, then idle; P-states and draw sampled on every card of the profile):

| step | dual card (TP=2) | quad card (TP=4) |
|---|---:|---:|
| idle quiesce armed | 2 of 2 ranks | 4 of 4 ranks |
| 20 requests | 20 of 20 ok | 20 of 20 ok |
| all cards in P8, after the last response | **18.4 s** | **19.0 s** |
| all cards still in P8 from 25 s after the last response to the end of the 60 s window | yes | yes |
| draw of all the profile's cards over that stretch (mean) | 55.0 W (2 cards) | 113.0 W (4 cards) |
| reader parked / resumed | parked on 2 ranks; 4 resume log lines after one request | parked on 4 ranks; 8 resume log lines after one request |
| all cards in P8 again, after the resume request (draw as above) | **18.2 s** (54.7 W) | **18.7 s** (112.3 W) |
| guard hits; quiesce turned off by a run-time check | 0; 0 | 0; 0 |
| final drain at shutdown | complete on 2 of 2 ranks | complete on 4 of 4 ranks |

The rest of this section is the **first revision**, measured on its own tree before the second revision existed; the second revision keeps the same idle path.

**What holds a card out of P8.** One card, no engine, 40 s windows; each arm adds one activity to an otherwise idle process. The driver's own idle delay is the `quiet` row.

| activity | first P8 after the last work | P8 share of the tail | mean draw over the tail (one card) | P-states seen in the tail |
|---|---:|---:|---:|---|
| quiet (no activity) (run 1) | 17.1 s | 88 % | 46.1 W | P2, P5, P8 |
| host wake only | 17.1 s | 88 % | 45.3 W | P2, P5, P8 |
| `event.query` polling | 17.1 s | 88 % | 45.2 W | P2, P5, P8 |
| `event.record` polling | 18.2 s | 84 % | 46.7 W | P2, P5, P8 |
| a device-to-host copy every 50 ms (the old reader) | never | 0 % | 118.5 W | P2 |
| quiet (no activity) (run 2) | 18.2 s | 84 % | 46.7 W | P2, P5, P8 |

Only the periodic device-to-host copy holds the card in P2. The other arms reach P8 after the driver's own idle delay.

**End to end on the dual card profile** (shipped `serve/serve.sh` of this tree, 20 requests, then idle):

| step | result | pair draw |
|---|---:|---|
| idle quiesce armed | 2 of 2 ranks | - |
| 20 requests | 20 of 20 ok | - |
| both cards in P8, after the last response | **18.5 s** | 94.2 W (pair, mean over the window from 5 s to 60 s, which includes the drop) |
| reader parked / resumed | parked on 2 ranks; 4 resume log lines after one request | - |
| both cards in P8 again, after the resume request | **18.3 s** | - |
| **control**: same boot, idle quiesce not armed (reader polls forever) | never (stays P2) | **223.7 W** (pair, same window) |

- **Idle cards reach P8 as fast as a fully quiet GPU** (17.1-18.2 s for the quiet baseline, the driver's own idle delay on this host), both at the first idle and after a resume.
- The control shows the effect is the reader: with it polling, the pair stays in P2 and draws the figure above.
- The draw figures are window means over a stretch that includes the P2 to P8 drop, so they are comparisons between rows, not steady-state idle watts; the steady quiet draw is the ablation's tail mean.

## 11. Tree 5 under load (series 0012, second revision; 2026-10-11)

A targeted re-check of the shipped engine after the gate: an install of tree 5's engine, patches and serve scripts
(the later release commits change only comments, docs and checksums), fresh compile caches, the shipped `serve/serve.sh` and
`serve/serve-tp4.sh` with no knob set, the same harnesses as sections 1-4. Dual card ran on one card pair while the ThinkingCap
checkpoint ran on the other ([ThinkingCap card](ThinkingCap-BENCH-CARD.md), section 8); quad card ran alone afterwards.

|  | dual card (TP=2) | quad card (TP=4) |
|---|---:|---:|
| KV pool, attention block (as frozen) | 826,689 tokens, 960 | 1,974,072 tokens, 848 |
| boot to ready (fresh compile cache) | 412 s | 397 s |
| tile proof (`release/check-tile-proof.sh`) | once on each of devices 0-1 | once on each of devices 0-3 |
| patch proof lines (embed / plan cache / split-K / seq-lens / RecoverSSM) | 2 / 100 / 2 / 2 / 32 | 4 / 208 / 4 / 4 / 0 |
| GSM8K-200 (as served) | **195 / 200** (0 errors, 3 cut at the token cap) | **196 / 200** (0 errors, 3 cut at the token cap) |
| 262K needle (261,568-token cold prompt) | pass, 307 s | pass, 181 s |
| 8-conversation load: requests (errors) | 222 (0) | 488 (0) |
| largest prompt | 146,202 tokens | 237,113 tokens |
| prefix-cache hit rate | 90.6 % | 96.5 % |
| TTFT p50 / p99 | 8.3 s / 281.6 s | 9.0 s / 101.6 s |
| preemptions; peak KV pool usage | 0; 100.0 % | 0; 85.9 % |
| multi-turn replay: first turn TTFT, median follow-up TTFT | 22.8 s, **2.79 s** | 15.3 s, **1.78 s** |
| loop gate (72 generations, c6): engine errors / loose flags / genuine collapses / length-capped | 0 / 0 / **0** / 1 | 0 / 0 / **0** / 0 |
| loop gate: mean repeat-8, worst 12-token window repeat | 0.002, 3 | 0.0017, 2 |
| smokes: warm prefix (8,414 / 8,413-token prompt, 8,384 cached, output identical), tool call, reasoning split, vision 4 of 4 | all passed | all passed |
| boot errors | 0 | 0 |

- At shutdown (SIGTERM after the last step), the quad card log carries one `EngineDeadError` traceback from the API server's
  output handler, written after the engine's teardown had completed: every worker reported a clean final guard drain and exited
  with code 0, and no request failed. The dual card boot shut down without it.
- Against the gate (sections 1-4, earlier engine): GSM8K-200 195 against 196 on dual card and 196 against 194 on quad card; the
  needle, pool, loop gate and smokes repeat (residency was not re-run); the 8-conversation load is a fixed-duration run whose
  request count and hit rate depend on how far the conversations grow, so its rows are a re-check, not a comparison.

## Notes and caveats

- Reference host only: RTX 3090 (sm_86), PCIe Gen4 x16, P2P only. Without P2P these profiles are unmeasured.
- The dual card rows of sections 1, 2 (needle, residency) and 4 (GSM8K-200) come from one dual card boot; the load, loop gate, smokes and images from a second boot on the other card pair of the host (the two pairs differ by about 5 % in decode speed). The pool and boot figures of both boots are in section 5.
- This gate measured the shipped K=4 profile only. K=1 / K=3 ladders, the c1-c32 decode throughput table and the long-prefill probe were not re-run; the v1.0.0 figures for those are on the [2026-09-26 card](../2026-09-26/Qwen3.8-27B-BENCH-CARD.md) (a different engine and harness).
- Single-stream cells are 3 requests per depth and mode, so each median is one of three samples. The thinking-on harness caps its output at 256 tokens at effort low; ITL and tokens/step are per-step figures and do not depend on the cap.
- tokens/step here is tokens per streamed chunk of the request (the harness's own count); earlier cards took it from the engine's speculative-decoding counters. The two agree to the extent that one chunk is one step.
