# v2.5.1 card — 2026-10-06 (against v2.2.0: decode ladder, multi-turn cache, capacity, depth chart)

> **Provenance.** Maintainer measurements on the reference host (4× RTX 3090, 220 W, PCIe P2P, no NVLink; driver
> 610.43.02). v2.5.1 runs vLLM 0.30.0 + the v2.5.1 commits on python 3.13.5, torch 2.13.0 (CUDA 13.0), triton 3.7.1,
> flashinfer 0.6.18.post1, transformers 5.18.0, with the reference host's serve command (determinism switches on, guard
> telemetry `warn`, the clip counter on), `--block-size 4096` and `--prefix-match-unit 64`. "Release code" below marks
> runs on the published source, `96a0d13b6e`: section 5's two follow-up runs (8 conversations; one conversation).
> The section 1 decode check and every other run marked "v2.5.1" (prefill, decode ladder, depth, capacity, bench, quality,
> cache validity) ran on two pre-release builds of the same series (the decode check on the later one) that differ only in the
> waiting-request admission path, which none of those runs reach. The logging-only
> exact-counter reader was off on the two decode-ladder and depth boots (its variable was renamed in this release) and
> on for the v2.2.0 and v2.5.0 comparator boots and the section 1 check. Streams are client-timed. The drivers, raw streams and boot
> logs are not published (as for the earlier cards).
>
> v2.5.1 is a multi-turn cache release: in a single conversation a follow-up turn re-prefills a median of 3,202 tokens
> instead of 9,326 on v2.5.0 and waits 0.96 s instead of 1.97 s for its first token; with 8 deep conversations at once,
> 92.5 % of follow-up prompt tokens come from the cache (v2.5.0: 90.6 %) and the median follow-up waits 1.61 s instead
> of 3.54 s, though 4 of 99 follow-ups re-prefilled the whole conversation with the pool near full (section 5).
> Decode, prefill, capacity and quality read as v2.5.0's within run-to-run noise.

## 1. Single-stream decode ladder against v2.2.0

One frozen ladder on one host: decode is tok/s over the streamed answer, T=0 with a seed, needle prompt at depth,
≤ 256 tokens with thinking on and ≤ 1,024 with thinking off, 3 sends per cell; each column is the mean of the per-boot medians. The reference column is 7 boots: 2 of
v2.2.0 on 2026-10-04, interleaved with the release-candidate boots, and the 5-boot v2 gate of 2026-09-17, which
v2.2.0 matched at parity ([`../2026-09-24/BENCH-CARD.md`](../2026-09-24/Flash-Next-BENCH-CARD.md)). With MTP, read the median
event interval (ITL) together with tokens per event, not tok/s alone.

Two boots, exact-counter reader off:

| prompt tokens | thinking | v2.5.1 tok/s | reference tok/s | change | ITL ms (v2.5.1 / reference) | tokens/event (v2.5.1 / reference) |
|---|---|---|---|---|---|---|
| 4,096 | on | 157.9 | 162.0 | −2.5 % | 19.44 / 18.84 | 3.03 / 3.05 |
| 4,096 | off | 117.4 | 119.2 | −1.6 % | 19.36 / 18.76 | 2.27 / 2.24 |
| 32,768 | on | 159.2 | 165.1 | −3.6 % | 19.59 / 18.91 | 3.04 / 3.08 |
| 32,768 | off | 115.2 | 121.1 | −4.9 % | 19.49 / 18.91 | 2.24 / 2.29 |
| 131,072 | on | 161.8 | 168.4 | −3.9 % | 19.85 / 19.25 | 3.13 / 3.15 |
| 131,072 | off | 117.6 | 122.1 | −3.7 % | 19.87 / 19.22 | 2.31 / 2.33 |
| 261,120 | on | 174.3 | 177.0 | −1.5 % | 20.03 / 19.42 | 3.39 / 3.32 |
| 261,120 | off | 117.6 | 124.2 | −5.3 % | 20.04 / 19.39 | 2.32 / 2.38 |

**Pre-registered measure** (the v2.5.0 card's, unchanged): one index per boot, the mean over the four thinking-on
cells of the boot's tok/s divided by the reference mean of that cell; Welch t-test, two-sided. These two boots read
−0.01 % against the v2.5.0 final build (95 % CI −4.29 % to +4.27 %, p = 0.996, 2 boots against 2) and −2.87 % against
the reference (95 % CI −6.84 % to +1.09 %, p = 0.085, 2 boots against 7); v2.5.0's release candidate measured −2.81 %
against the same reference with 4 boots. A third boot, on the later pre-release build with the exact-counter reader on as on
the v2.2.0 and v2.5.0 comparator boots, read −1.12 % against the reference (the seven reference boots' own indices span −1.33 % to
+1.58 %) and +1.8 % against the v2.5.0 final build's mean; switching the reader off did not flatter the two boots
above. v2.5.1 changes the prefix cache and the scheduler, not the decode path.
Per-cell confidence intervals with 2 boots are wide and are not shown. Tokens per event match; the event interval is
0.6–0.7 ms longer, as on v2.5.0: RecoverSSM (the GDN state recovery for MTP speculative decoding, upstream #58863,
enabled by `--use-replayssm`) is in use, and its speculative verify runs the Triton GDN decode kernel and replays the
state on every step.

## 2. Aggregate throughput, multi-turn cache, accuracy and quality

| | v2.5.1 | v2.2.0 |
|---|---|---|
| N=8 aggregate | 533.7 tok/s (v2.5.1) | 531.5 tok/s (final tree, 2026-09-24) |
| N=1 aggregate | 124.4 tok/s (v2.5.1) | 117.8 tok/s (final tree, 2026-09-24) |
| multi-turn: follow-up cache reuse, 8 concurrent agent sessions (section 5's load, temperature 0) | **92.5 %** (release code) | 7.5 % (sessions growing to 80K, temperature 0.6) |
| multi-turn: median time to first token on follow-up turns, same load | **1.61 s** (release code) | 52.7 s (sessions growing to 80K, temperature 0.6) |
| one conversation alone, 26 follow-up turns (36K to 76K): median time to first token | **0.96 s** (release code) | 1.56 s |
| GSM8K-200, thinking off | 198/200 (v2.5.1) | 198/200 |
| MTP mean acceptance length (one 10-second serving log window) | 3.45 (v2.5.1) | 3.31 |
| divergence from the BF16 teacher (24 held-out prompts; lower is closer; ceiling 0.0388) | 0.0334 (v2.5.1, 4 captures) | 0.0338 (1 boot) |
| structured output, 80 requests over four load cells | 80/80 (v2.5.1) | 80/80 |
| 262,144-token needle (261,568 prompt tokens) | exact (v2.5.1) | exact |
| mixed soak | not re-run (v2.5.0 release candidate: 90 minutes, 11,002 requests, 0 errors, 0 HTTP 5xx) | 90 minutes: 12,115 requests, 0 errors |

The aggregate rows are one boot each and not same-day (v2.2.0's are from its own card); read them as unchanged
within run-to-run noise (v2.5.0 measured 535.2 and 127.8 tok/s). The multi-turn rows are section 5's load at
temperature 0: each session is a shared tool prefix, a unique history and turns that append a reply and a new tool
result, 8 sessions at once. v2.2.0 was measured on the same shape sampled at temperature 0.6 (the v2.5.0 card's load,
the one that reproduced the cache issue reported on Hugging Face, discussion #2), where v2.5.0 measured 82.0 % and
3.9 s; at temperature 0 v2.5.0 measured 90.6 % and 3.54 s (section 5). The soak was not re-run (v2.5.1 does not
change the decode path). The one-conversation row is a single agent session with no other traffic: a follow-up
re-prefills a median of 3,202 tokens on v2.5.1, 9,326 on v2.5.0 (1.97 s) and 6,931 on v2.2.0. The divergence metric is defined in the v2.2.0
card ([`../2026-09-24/BENCH-CARD.md`](../2026-09-24/Flash-Next-BENCH-CARD.md)); the four captures of the one boot agree to six
decimal places.

## 3. Capacity

| | v2.5.1 | v2.2.0 |
|---|---|---|
| KV pool, FP8 tokens, `--kv-cache-memory 4100000000` | **924,993** (attention block 4,096; 248 blocks; as v2.5.0) | 806,792 (block 3,200) |
| full 262,144-token sessions resident at once | 3, 0 preemptions; a fourth not resident (v2.5.1) | 3 on v2; v2.2.0 re-tested at 3 × 256K |
| 131,072-token sessions resident at once | **6**, 0 preemptions; a seventh not resident (v2.5.1) | 5 |
| full-pool pressure, 8 sessions | multi-turn load of section 5: pool at 99.2 %, 2 preemptions, 107/107 requests completed (release code) | pool driven to 100 % with preemptions, no failed request |
| without MTP (same command minus `--speculative-config`) | not re-measured (v2.5.0 release candidate, computed attention block: pool 1,057,314) | — |

v2.5.1 does not change the pool size or its layout. It changes which cached blocks may be evicted, and a request
resumed mid-block holds one extra state block (its tail checkpoint) until it is freed. Session
token totals overstate capacity: each session also holds fixed Gated-DeltaNet state blocks, so count sessions, not
tokens.

## 4. Prefill and decode over context depth (the chart)

![ctx_pp vs ctx_tg over context depth, with step time, v2.2.0 medians as the dashed line](flashnext-v2.5.1-ctx-pp-tg-itl.png)

The serve environment above (exact-counter reader off, see the provenance note); prefill on one boot, decode on the
second, all seven depths (2026-10-06). Same driver, cells and prompts as the v2.2.0 chart, whose medians are the
dashed line (10K–200K from 2026-09-24, 261K from 2026-10-05, as on the v2.5.0 chart).

- **ctx_pp** is cold prefill: 3 sends per depth, each with a unique nonce at the start of the first user turn, and
  every send asserted `cached_tokens == 0`.
- **ctx_tg** is decode with thinking on (`reasoning_effort` low), 512 tokens. At each depth, 3 different prompts (the
  same three as v2.2.0), 2 sends each; a prompt's value is the median of its 2 sends.
- **Step time** is the median inter-event interval of the same decode sends.
- In the chart every axis starts at zero; each faint dot is one send (ctx_pp) or one prompt (ctx_tg, step time), the
  line joins the medians, whose values are printed, and the dashed line is the v2.2.0 median.

| depth | ctx_pp tok/s, median [min–max] | vs v2.2.0 | ctx_tg tok/s, median [min–max] | per-prompt ctx_tg | step time ms, median [min–max] | tokens/event per prompt |
|---|---|---|---|---|---|---|
| 10K | 5,219 [5,146–5,225] | +1.9 % | 152.1 [149.4–180.4] | 149.4 / 152.1 / 180.4 | 19.36 [19.26–19.59] | 2.88 / 2.91 / 3.51 |
| 25K | 5,441 [5,430–5,442] | +1.0 % | 152.1 [150.3–161.2] | 150.3 / 152.1 / 161.2 | 19.37 [19.32–19.50] | 2.89 / 2.93 / 3.12 |
| 50K | 5,533 [5,525–5,536] | +2.3 % | 167.4 [165.6–168.2] | 168.2 / 167.4 / 165.6 | 19.61 [19.52–19.62] | 3.27 / 3.25 / 3.21 |
| 100K | 5,484 [5,478–5,485] | +2.7 % | 182.2 [180.9–184.5] | 184.5 / 182.2 / 180.9 | 19.88 [19.84–19.90] | 3.63 / 3.58 / 3.56 |
| 150K | 5,404 [5,395–5,405] | +2.7 % | 178.3 [161.0–185.7] | 185.7 / 178.3 / 161.0 | 19.58 [19.58–19.89] | 3.61 / 3.46 / 3.18 |
| 200K | 5,324 [5,318–5,330] | +2.8 % | 149.7 [145.1–179.7] | 149.7 / 145.1 / 179.7 | 19.92 [19.87–20.06] | 2.96 / 2.86 / 3.58 |
| 261K | 5,236 [5,234–5,238] | +2.3 % | 155.5 [139.5–160.8] | 160.8 / 139.5 / 155.5 | 20.04 [19.95–20.06] | 3.20 / 2.77 / 3.10 |

**Cold prefill is 1.0–2.8 % above v2.2.0** at every depth, and 0.2–1.6 % below the v2.5.0 final build's single boot
(−1.6 % at 10K, −1.4 % at 25K, 0.2–0.4 % from 50K to 261K). **Read the decode column through the step-time column.**
Step time is 19.4–20.0 ms from 10K to 261K, 0.2–0.7 ms above v2.2.0 at each depth; the ctx_tg spread tracks how many
drafted tokens each text accepts (2.77–3.63 per event). One boot per series, so a single depth's decode difference
against v2.2.0 is a content draw; section 1's repeat boots are the measure of the decode cost.

## 5. What changed in v2.5.1: follow-up turns

v2.5.0 kept a running request's Gated-DeltaNet checkpoint, but a follow-up turn could only resume at a 4,096-token
block boundary, and under pressure a conversation's cached prefix could still be evicted between its turns. v2.5.1:

- **resumes a follow-up on a 64-token grid** (`--prefix-match-unit 64`): it re-prefills from the last 64-token
  boundary of the cached prefix instead of the last 4,096-token block (with MTP, one 64-token unit earlier, for the
  draft); in one conversation, a median of 3,202 uncached tokens per follow-up instead of 9,326;
- **holds the GDN tail checkpoint until the request is freed**, so the state at the end of a turn cannot be evicted
  while that turn is still decoding;
- **keys a tail that ends exactly on a block boundary**, so the next turn can find it;
- **pins a waiting turn's cached prefix** (up to 16 waiting turns), so other requests' allocations do not evict it
  while the turn queues;
- **keeps a finished turn's prefix until that conversation's next turn** (up to 16 conversations, oldest released
  first).

The last two are best-effort: when the pool is full and a running request or the next admission needs blocks, the
protection is released rather than preempting a running request.

Eight concurrent agent conversations (about 76K tokens at the first turn, growing past 120K), temperature 0, 300 s,
the shape of section 2's multi-turn rows:

| | v2.5.1, release code | v2.5.0 |
|---|---|---|
| follow-up cache reuse | **92.5 %** | 90.6 % |
| median time to first token, follow-up turns | **1.61 s** | 3.54 s |
| requests started within 300 s, all completed (of them follow-ups) | **107** (99) | 88 (80) |
| follow-ups that re-prefilled the whole conversation | 4 | 0 |
| KV pool peak | 99.2 % | 99.2 % |
| preemptions | 2 | 0 |
| errors | 0 | 0 |

One conversation alone (36K growing to 76K, 26 follow-up turns):

| | v2.5.1 | v2.5.0 | v2.2.0 |
|---|---|---|---|
| median time to first token on follow-ups (p90) | **0.96 s** (1.02 s) | 1.97 s (2.23 s) | 1.56 s (2.02 s) |
| uncached tokens per follow-up, median | **3,202** | 9,326 | 6,931 |

Reuse is cached over prompt tokens, summed over the follow-up turns; a full re-prefill is a follow-up with under 50 %
of its prompt cached (every such turn had 0 cached, on prompts of 104,785–115,735 tokens). The v2.5.0 columns are the
same drivers on the v2.5.0 release code (block 4,096, no prefix-match unit), 2026-10-05. One boot of the release code.

Under 8 deep conversations, reuse is close to v2.5.0's (92.5 % against 90.6 %), but the median follow-up waits about
55 % less (1.61 s against 3.54 s) and 22 % more requests start within 300 s (107 against 88): v2.5.0 re-prefilled part
of every follow-up (a median of 9,326 tokens in the single-conversation run), while v2.5.1 resumes most follow-ups on
the 64-token grid. **What is left:** with the pool near full (99.2 %), 4 of 99 follow-ups found no cached prefix
and re-prefilled the whole conversation (105K–116K tokens); v2.5.0 had none at this load. On the two earlier cuts,
the same load gave 90.7 %, 1.95 s, 101 requests and 6 full re-prefills (earlier cut) and 91.9 %, 1.65 s, 110
requests and 5 (later cut): one boot each, so the three read as the same within run-to-run noise. When a waiting turn's protected
prefix and a running request's need for blocks collide, the protection yields rather than preempting the running
request, and the prefix is evicted.

## 6. Cache validity

Twenty prompt pairs, each sent cold (0 cached tokens) and again warm after a shorter turn primed the cache: every warm
send hit at the expected 64-token boundary (up to 236,864 cached tokens), the T=0 output was identical to the cold
send's and the functional answer correct, 20/20. An image check (the same text with two images in either order) answered
from the images, 4/4.

| tree | identical | correct | hit at the expected boundary | image check |
|---|---|---|---|---|
| v2.5.1 | 20/20 | 20/20 | 20/20 | 4/4 |
| v2.5.0 (control, block-aligned hits) | 20/20 | 20/20 | 20/20 | 4/4 |

Greedy repeatability at depth was not re-run for v2.5.1 (the decode path is v2.5.0's); see
[the v2.5.0 card, section 6](https://github.com/halt95/merlin-engine/blob/merlin-v3.0.1/benchmarks/2026-10-05/Flash-Next-BENCH-CARD_OLD.md#6-greedy-repeatability-at-depth-release-candidate).

## 7. Fault containment and shutdown

- Not re-run for v2.5.1: fault injection, the shipped-default-variables boot and the tool-call, reasoning-split and
  chat-template smoke (v2.5.1 changes the prefix cache and
  the scheduler only);
  [the v2.5.0 card, section 7](https://github.com/halt95/merlin-engine/blob/merlin-v3.0.1/benchmarks/2026-10-05/Flash-Next-BENCH-CARD_OLD.md#7-fault-containment-and-shutdown-release-candidate),
  has the release-candidate results.
- Known issue, not reached by the shipped configuration: with pipeline parallelism, blocks whose release is deferred
  can make the shared-prefix shortcut over-report a common prefix; only cascade attention on the V1 runner consumes it,
  and the shipped configuration uses the V2 runner with cascade attention off.
- 0 engine errors over the three v2.5.1 boots (ladder, depth, residency and bench; divergence, needle, structured
  output, follow-up and cache validity), the later-cut boot (decode check) and the release-code boot (one conversation,
  8-conversation load) while serving; 0 request errors. One of those boots logged the API server's output handler raising
  `EngineDeadError` during the scripted drained shutdown, after every request had finished and all four workers had
  exited cleanly.
