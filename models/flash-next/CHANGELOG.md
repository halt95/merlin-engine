# Changelog

Release notes of the Flash-Next recipe, newest first. See the [README](README.md) for current serving status
and [reference](docs/reference.md) for configuration and evidence boundaries.

## Merlin 3.0.2.1

The Flash-Next recipe runs on **vLLM 0.33.0**, as in 3.0.2. The recipe version stays 3.0.0 (the VERSION file is
unchanged). See the [engine changelog](../../docs/CHANGELOG.md#merlin-3021) for the full release. This
recipe's results are in the [2026-11-12 bench card](benchmarks/2026-11-12/BENCH-CARD.md); the 2026-11-04 card stays as the
3.0.2 comparator.

- **FGPC on:** the launcher passes `--enable-mamba-shared-prefix-checkpoint`.
  - A first turn behind a shared prompt of at least one 4,096-token attention block resumes from the junction, and
    junctions age like any cached block.
  - On the swarm launch with a ~9K shared prompt, 43 of 87 and 41 of 85 first turns were served from cache in two
    runs, against 0 of 82 with FGPC off.
  - `-- --no-enable-mamba-shared-prefix-checkpoint` turns it off.
- **`--long-prefill-token-threshold 512` by default** (`LPT`; `LPT=0` = off), so a long cold prefill holds back
  new turns and running decoders less (they still slow down; see the prefill runs below) while a session slot is free
  (`SEQS`). Requested in GitHub #6 (item 1).
- **Prefill runs by default** (`PREFILL_RUNS`; `PREFILL_RUNS=0` = off): while sessions decode, a long cold prompt takes
  turns with them instead of sharing every step, and `--max-num-batched-tokens` becomes 1,056. Measured in four runs of the shipped
  setting on 4 x RTX 3090 with three sessions decoding through a cold prefill: they keep 31-38 % of their decode speed
  during a cold 64K-128K prefill (14-16 % with the slices alone); the cold prompt's first token comes 17-19 % later at
  64K, 12-14 % at 128K and 19-24 % at 262K, and the p99 gap between streamed tokens is 0.40 s in three of the four runs
  (0.58 s in one, a client-side reading), against 0.23-0.35 s without the runs. With no session decoding, three cold
  64K prompts sent together reach their first token (p95) in 34.5-34.7 s against 28.3-28.9 s. A new prompt still waits
  for a free session slot. When `CHUNK` and the block size do not tile (neither divides the other) or the block is
  below the hybrid-state floor, the default switches the runs off with a note at start-up; `PREFILL_RUNS=1` refuses. `VLLM_MERLIN_DECODE_FIRST=1` needs
  `PREFILL_RUNS=0`. See the
  [reference](docs/reference.md#merlin-301-options). GitHub #6 (item 1), reported by Sudzpel.
- **4 × 20 GB rows (`fn-quad-20g-sm86`, `fn-quad-20g-sm89`; untested ⚫):**
  - **Layout:** the quad layout with the KV pin cut to one 262K session.
  - **Static cold-expert offload** (`MERLIN_COLD_EXPERTS`) on layers 11, 12, 19, 20 and 37 at hot fraction 0.5.
  - **Row engine settings:** 64 MiB prefill-logits caps, `NCCL_BUFFSIZE=524288`, `NCCL_MAX_NCHANNELS=1`, expandable
    segments.
  - **Evidence:** emulated on 4 × RTX 3090 with 4 GiB held back per card.
- **KV RAM cache:** a full tier keeps its decode speed (GitHub #11), plus the restore, reclaim and forecast fixes in the
  engine changelog.
- **Quoted tool calls stay text** on the `qwen3` parser; the launcher passes `MERLIN_QWEN3_LITERAL_SPANS=0|1` through
  (any other value stops the launch). `MERLIN_KV_TIER_DIAG=1` also passes through.
- **Decode top-k** reads only valid positions on rows whose length is not a multiple of 4 (port of vllm#62467).

## Merlin 3.0.2

The Flash-Next recipe is **built on vLLM 0.33.0**; the recipe version stays 3.0.0 (the VERSION file is unchanged). See the
[engine changelog](../../docs/CHANGELOG.md#merlin-302) for the full release. This recipe's results are in the
[2026-11-04 bench card](benchmarks/2026-11-04/BENCH-CARD.md); the 2026-11-02 card stays as the 3.0.1 comparator.

- **`--ple-store disk` (experimental):** reads the PLE table's rows from the checkpoint's files on NVMe instead of holding
  the table in host RAM (47.7–47.8 GiB freed with FP8, 38.8–38.9 GiB with `--ple int6`). Measured on the 0.32 tree; on 0.33 the FP8 table was re-measured for 3.0.2 against `ram`: output identical, and decode, warm first token and 262K unchanged within run-to-run noise. See the [3.0.2 options](docs/reference.md#merlin-302-options).
- **`--context SIZE` (experimental):** YaRN past 262K built from the checkpoint's config; see
  [Context size](docs/reference.md#context-size-302).
- **Pipeline-parallel decode:** commit kernels warmed on every PP rank at boot, a persistent snapshot ring that reuses its storage instead of allocating it each step, one align-postprocess launch per distinct input set, and two optional off-by-default switches
  (`VLLM_RECOVERSSM_PP_SPLIT_RECV`, `VLLM_MERLIN_PP_DECODE_BALANCE`). Tokens unchanged by construction; the on-versus-off comparison has not been done for 3.0.2.
- **Decode-first budget (experimental, off by default):** `VLLM_MERLIN_DECODE_FIRST=1`; see the [3.0.2 options](docs/reference.md#merlin-302-options). Reported by Sudzpel.
- **Registry:** the CMP 170HX and RTX 4090 rows, whose evidence is P2P-off only, report "untested" when the resolved P2P mode is on
  (`match.status_p2p_modes`).
- **Variant scales:** a Swift1.5 or Swift1.5-Huihui name on a directory that has the right `config.json` but neither the release
  scales file nor the verified marker is refused; pass `SCALES=<the recipe's scales file>` or run `merlin pull`.
- **Held back from 0.33:** lookahead block hashes, DFlash/DSpark trailing-block drop, small-batch split top-p, step-keyed RNG
  default. See the engine changelog.

## Merlin 3.0.1

- **DFlash2, opt-in:** `--spec dflash2` on `merlin serve` or `./setup.sh`; `SPEC=dflash2` for a direct
  recipe launch. Only the three Flash-Next-family 4 x 3090 P2P rows have qualified geometry.
  Decode measured +5–18 % versus MTP. At introduction, the DFlash2 pin was 3,468,165,120 B and the pool was
  789,668 tokens. Since Merlin 3.0.2.1, the pin is 3,630,000,000 B and the measured pool with prefill runs on
  is 815,203 tokens (FP32 GDN state, align-mode prefix caching); the engine fits 3 x 262K or 5 x 131K
  sessions, not 6 x 131K.
  Setup downloads and verifies the drafter; serving requires it at `DFLASH2_MODEL` or
  `~/models/flash-next-dflash2`. MTP remains the default.
- **Host-RAM KV tier, opt-in:** `--kv-ram-cache SIZE` caches evicted prefix KV blocks for restore
  instead of recompute; it adds no GPU context or simultaneous-session capacity. SIZE is GiB per server,
  split into whole GiB per TP x PP rank, and needs additional host RAM with locked memory allowed.
  On the reference 4 x RTX 3090 host a 262K restore took 1.2–1.4 s, against about 45 s to recompute. The cache is off by default and experimental in 3.0.1. See the
  [3.0.1 options](docs/reference.md#merlin-301-options) for controls and limits.
- **Reduced draft vocabulary removed.** The `REDUCED_DRAFT_VOCAB` / `DRAFT_VOCAB` switches, the bundled table and its
  builder are gone; setting either switch stops the launch (an empty value, or `REDUCED_DRAFT_VOCAB=0`, is accepted). `DRAFT_HEAD` still sets the draft head's precision.

## Merlin 3.0.0

The Flash-Next recipe is **built on vLLM 0.32.0** in the shared Merlin engine repository.

- **No reduced draft vocabulary by default.** Merlin 3.0.0 does not carry the reduced draft vocabulary of the v3.0.0 bundle (40,960-id table plus INT4 draft head, below). The launcher scores the full vocabulary; `DRAFT_HEAD=bf16|int8|int4` sets the draft lm_head precision independently of any table (shipped default `bf16`, the v2.5.x drafter; `DRAFT_HEAD=int8` measured 6.5–7.2 % lower ms per output token than BF16 at 32K/131K/258K with the same MTP acceptance in the 2026-10-17 A/B and is a candidate default for a later release; 3.0.1 keeps `bf16`). No reduced table shipped in 3.0.0, and the switch was removed in 3.0.1.

- **Concurrent decode, 4× RTX 3090, pre-release vLLM 0.32 build, 2026-10-15:** six 126,976-token sessions, 574.5 tok/s aggregate,
  10.78 ms/tok per agent, cold prefill 5,423 tok/s, 0 preemption.
- **Cache hit with MTP off:** hit ratio 0.998; 32K TTFT 5.78 → 0.315 s (cold → cached).
  The launcher defaults to MTP K=3; this measurement does not establish its cache-hit performance.
- **Vision:** 8 MP per image (`IMG_CAP_PX=8388608`) at the full
  `--kv-cache-memory-bytes 4100000000` pin, with stage-0 minimum free memory **902 MiB**.
  The 1x M-RoPE cache is automatic on this engine.
- **Pool:** **924,993 tokens** measured on 0.32.
- **FGPC off:** omit `--enable-mamba-shared-prefix-checkpoint` for Flash-Next; prefix caching remains on.
- **Layout:** the engine is at the repository root and the canonical launcher is
  `models/flash-next/serve.sh`. The shared installer and container integrations have their own pending qualification.

The [0.32 measurement summary](benchmarks/2026-10-15/BENCH-CARD.md) records the supplied results and limits.
The design and cache-hit runs used a pre-release vLLM 0.32 build; later release-candidate and merged-source checks matched its
T=0 tokens and logprobs on four configurations. Those identity checks did not rerun throughput.
This docs refresh adds no GPU run. The old single-stream ladder, quality, needle, multi-turn reuse,
idle-power, warmup, boot-time, YaRN and residency results below are historical, not current 0.32 claims.
The old bundle install commands are removed from the current README and reference.

Historical entries retain their original version labels, including pre-0.30 releases. The
**v3.0.0 (0.30) figure** marker identifies figures carried in the imported v3.0.0 documentation, not a
claim that every older result was originally measured on v3.0.0.

## v3.0.0

> **Historical measurements — v3.0.0 (0.30) figure.** This section retains the original release and engine labels;
> its performance, memory, quality, timing and test-count figures have not been re-measured on 0.32.

v3.0.0 was built and gated on 2026-10-10 but never published or tagged on GitHub.

v3.0.0 = stock vLLM v0.30.0 + 98 commits (v2.5.1's 97 and one for v3.0.0; the patch series, the bundle and the combined
diff were built for that release and are not files in this checkout). Same checkpoint. Release check on the reference host (2026-10-10; 16 of 17 items PASS, and the
VRAM-headroom item passed on a warm-against-warm re-run): single-stream decode 186–206 tok/s with thinking on, +17.5 to
+19.7 % per depth against v2.5.1's release ladder and +12.8 % against v2.5.1 booted the same day; the 924,993-token pool
holds 3 × 262K or 6 × 131K resident with 0 preemptions; GSM8K-200 198/200 (v2.5.1 198/200 on the same items); the 262K
needle found exactly;
6 deterministic 131K conversations reuse 98.97 %, identical to v2.5.1 turn for turn. Bench card:
[`benchmarks/2026-10-09/BENCH-CARD.md`](benchmarks/2026-10-09/BENCH-CARD.md).

### In brief

- **Reduced-vocabulary INT4 draft head, on by default.** The MTP drafter scores a 40,960-id draft vocabulary
  (`assets/draft-vocab.pt`, checked against its sha256 by `scripts/serve.sh`) through an INT4 draft-only head; the target
  verifies every draft, so output is unchanged. `REDUCED_DRAFT_VOCAB=0` serves the v2.5.1 drafter; `DRAFT_VOCAB=<path>`
  points at another copy of the same table. The release check measures the shipped configuration as a whole; it does not isolate the draft head's decode effect.
- **Idle power.** The bounds-guard reader parks once the engine drains (`VLLM_E2_GUARD_IDLE=1`, the default), so idle
  cards can drop to P8. The launcher keeps an inherited guard mode and interval, validates them, and warns when the
  mode resolves to `off`.
- **Start-up.** Bounded QSA warmup, reused GDN prefill warmup and persisted Triton autotune (on by default; `WARM=0`
  serves the v2.5.1 start-up path). Launch to the API answering: 268 s with an empty compile cache, 158–172 s on later
  boots (v2.5.1 from its cache on the same host: 153 s).
- RecoverSSM metadata reuse is always on.
- **Determinism.** At temperature 0, output could differ between compile caches: the persisted Triton autotune winners for
  the FLA GDN prefill kernels varied from cache to cache. The eight kernels now launch with one fixed config each on
  Ampere, and fresh caches match 23/23. This also holds for installs upgrading from v2.5.1.
- **PLE switches.** `VLLM_E2_PLE_FAST_IDS` and `VLLM_E2_PLE_EARLY_REQUEST` ship on together, about 0.11-0.16 ms per output
  token faster at identical output.
- **Guard.** The shutdown drain is bounded, TP and PP ranks agree on one counter transport, and the reader recovers
  after a failed sleep or wake-up.
- **Long-prompt memory knob.** `VLLM_E2_QSA_LOGITS_GRANULE_MB` (QSA prefill logits backing) is present and off by default.
- **Static names.** `scripts/serve.sh` and `scripts/build.sh` build into `vllm/` and `venv/` with the compile cache in
  `.vllm-cache`; the container image is `qwen38-flash-next-3090s` and its cache volume `flash-next-cache`.
  `scripts/serve-v2.5.sh` and `scripts/build-v2.5.sh` remain for this release as deprecation shims. A tree built by the
  pre-v3 `build-v2.5.sh` is v2.5.1 source and is refused by `scripts/serve.sh`; build v3.0.0 with `scripts/build.sh`.
- Credits for v3.0.0: brandon-nuttall (local-argmax mixin), CoolShades (idle-power report and contribution).


## v2.5.3

> **Historical measurements — v3.0.0 (0.30) figure.** This section retains the original release and engine labels;
> its performance, memory, quality, timing and test-count figures have not been re-measured on 0.32.

Release, 2026-10-12: a second checkpoint for this engine. **No engine change**: Merlin engine v2.5.2, the same bundle
(`v2.5.2-from-upstream-v0.30.0.bundle`, from the v2.5.2 release) and the same compiled ops (the v2.2.0 build-artifacts
asset).

- **Swift1.5-Huihui (abliterated).**
[halt95/Swift1.5-Huihui-Qwen3.8-Flash-Next-W4A16-Merlin](https://huggingface.co/halt95/Swift1.5-Huihui-Qwen3.8-Flash-Next-W4A16-Merlin),
the Swift 1.5 W4A16 checkpoint with huihui-ai's abliteration transplanted onto its 96 dense output projections (routed
experts unchanged). Treat it as a model that does not refuse: on StrongREJECT it refused 29 of the 278 prompts it
finished answering (10.4 %, one judged run; 35 cut off). Quality check against Swift 1.5 in the same run (on v2.5.1):
GSM8K 96.29 % vs 95.98 %, agent-turn tokens 0.86× (95 % CI includes parity). It ships Qwen's upstream `tokenizer.json`
(the check ran with Swift 1.5's earlier file; smoke-tested since). New files:
`scripts/serve-huihui.sh` (serve-v2.5.sh with this checkpoint's KV-scale sidecar, served names and `xhigh` default),
`scales/qsa_kv_scales_swift_huihui_dense.json`, `models/swift1.5-huihui/checkpoint.sha256`, `licenses/Huihui-LICENSE` and
`licenses/Huihui-NOTICE`, and its bench card
[`Huihui-BENCH-CARD.md`](../swift1.5-huihui/benchmarks/2026-10-12/Huihui-BENCH-CARD.md). In this tree the Huihui files live in
[`models/swift1.5-huihui/`](../swift1.5-huihui/README.md).

## v2.5.2

> **Historical measurements — v3.0.0 (0.30) figure.** This section retains the original release and engine labels;
> its performance, memory, quality, timing and test-count figures have not been re-measured on 0.32.

Point release, 2026-10-12: **Merlin engine v2.5.2** = the v2.5.1 engine plus one commit, the bounds-guard reader fix.
New bundle `v2.5.2-from-upstream-v0.30.0.bundle`, patch series 0001..0098 (0098 is the fix), `SHA256SUMS.v2.5.2` and
`upstream/PIN-v2.5` (commit `de6e2c2fe9`, tree `05e6d1a5…`). The compiled ops are unchanged (the fix is Python only), so
the v2.2.0 build-artifacts asset is reused again. To run it: the Quick start of the v2.5.3 README (tag `v2.5.3`).

- **Idle cards can drop to P8 with the bounds guards on.** With `GUARD=warn` (the default) the guard reader copied its
  counters to the host every 50 ms forever, and that copy alone holds an idle RTX 3090 in P2 (about 450 W idle on four
  cards against about 86 W in P8, as reported by a community member on GitHub; measured on the reference host: 446 W in P2 against
  110 W in P8). The reader now parks once the engine drains: when no
  engine call has been open for a full interval it fences the engine stream, waits for the fence, reads every counter
  one last time (nothing is writing them by then, so the read is complete), reports, and then does no device work at
  all until the next request arrives.
- **`GUARD=warn` keeps every bounds check.** A hit from the last step before idle is read and reported, and a fatal
  site is armed, before the reader parks; while serving, detection stays cadence-bound as before (the reader polls every
  `VLLM_E2_GUARD_INTERVAL`, 0.05 s). If any engine call or guarded launch runs off the engine thread or on an unfenced
  stream, the reader stops parking and polls forever, with an ERROR line. Sleep mode suspends the reader (one settled
  read, then no reads until wake-up).
- **New switch `VLLM_E2_GUARD_IDLE`** (default `1`; `0` polls forever, the v2.5.1 behaviour). `scripts/serve-v2.5.sh`
  sets it to 1 unless you set it, keeps an inherited value through its knob reset, and refuses anything but 0 or 1.
  It is not a compile factor: changing it does not trigger a recompile.
- **The same engine change as the Qwen3.8-27B releases.** The fix is the guard commit the 27B v1.3.0 and ThinkingCap
  v1.0.0 releases are built on, applied from the same patch file. Thanks to [CoolShades](https://github.com/CoolShades)'s
  [idle-power report](https://github.com/halt95/merlin-engine/issues/2) and [contribution](https://github.com/halt95/merlin-engine/pull/3), which proposed pausing the reader after a fixed 2 s of idle; the release parks it once the engine
  has drained instead, so it can share one implementation with the 27B releases.
- Nothing else changed: no kernel, scheduler or numerics code. Single-stream decode and cold prefill read as v2.5.1's
  within run-to-run noise (ladder cell medians within ±1.8 %; median tokens per event match to within 0.01); concurrent
  load and cache behaviour were not re-measured. Bench card: [`benchmarks/2026-10-12/BENCH-CARD.md`](https://github.com/halt95/merlin-engine/blob/v2.5.3/benchmarks/2026-10-12/BENCH-CARD.md).
  The v2.5.1 stability results were not re-run (the change is Python only, in the guard reader).

## v2.5.1

> **Historical measurements — v3.0.0 (0.30) figure.** This section retains the original release and engine labels;
> its performance, memory, quality, timing and test-count figures have not been re-measured on 0.32.

Released 2026-10-06: [v2.5.1 on GitHub](https://github.com/halt95/qwen38-flash-next-3090s/releases/tag/v2.5.1). Bench
card: [`benchmarks/2026-10-06/BENCH-CARD.md`](https://github.com/halt95/merlin-engine/blob/v2.5.3/benchmarks/2026-10-06/BENCH-CARD.md). To run it:
[Quick start](README.md#quick-start), or [Build and serve](docs/reference.md#build-and-serve) for the detail.

v2.5.0 was not published: v2.5.1 is v2.5.0 plus `--prefix-match-unit 64` and the prefix-cache fixes, so these notes
cover everything since v2.2.0, v2.5.0's changes included.

### In brief

- **Agent follow-up turns resume from cache.** A follow-up resumes on a 64-token grid (`--prefix-match-unit 64`) instead of the 4,096-token block, and the cached prefix a conversation needs is held until its next turn. One conversation alone: median time to first token on follow-ups 0.96 s (v2.5.0: 1.97 s; v2.2.0: 1.56 s).
- **Eight deep conversations at once.** Cache reuse is close to v2.5.0's (92.5 % against 90.6 % of follow-up prompt tokens), the median follow-up time to first token is about 55 % lower (1.61 s against 3.54 s) and 22 % more requests started within 5 minutes, all completed (107 against 88); 4 of 99 follow-ups still re-prefilled the whole conversation with the pool near full (99.2 %). v2.2.0 served 7.5 % from cache, with a 52.7 s median, on the same shape of load at temperature 0.6 with sessions growing to 80K, the load that reproduced the cache issue reported on Hugging Face.
- **More KV capacity.** The pool grows from 806,792 to 924,993 tokens (+15 %). Six 131K sessions and three 262,144-token sessions were each measured resident at once with 0 preemptions (v2.2.0: five 131K).
- **Tokenizer and image fixes.** Transformers 5.18.0 handles combining marks as intended, with parity on 92 items. Prompts can contain up to 42 images, each capped at 4 MP.
- **Defaults match the shipped launcher.** Unset `VLLM_E2_PLE_PULL_TRANSPORT` and `MERLIN_` kernel toggles use the shipped value of 1; `=0` opts out when you run the engine directly (`scripts/serve.sh` always sets the shipped values). `VLLM_USE_BREAKABLE_CUDAGRAPH=0` remains required.
- **Faster cold prefill.** 5,219–5,533 tok/s from 10K to 261K tokens, 1.0–2.8 % above v2.2.0 at every depth.
- **Stability checks passed.** GSM8K-200 scored 198/200, structured output passed 80/80 cases, the 262K needle was found exactly, cache hits were valid on 20 of 20 prompt pairs, and 0 engine errors occurred while serving (one boot logged an `EngineDeadError` during the scripted shutdown, after every request had finished); divergence from the BF16 teacher is 0.0334 (v2.2.0: 0.0338).

Single-stream decode is about 3 % slower than v2.2.0: v2.5.0's release candidate measured −2.81 % (95 % CI −4.60 % to −1.03 %, p = 0.0087), and v2.5.1, which does not change the decode path, reads −2.87 % (95 % CI −6.84 % to +1.09 %, 2 boots against 7) and −0.01 % against v2.5.0 (95 % CI −4.29 % to +4.27 %, 2 boots each). Eight concurrent requests are effectively unchanged: 533.7 against 531.5 tok/s. The trade is faster agent follow-ups, higher cache reuse and a 15 % larger pool. Bench card: [`benchmarks/2026-10-06/BENCH-CARD.md`](https://github.com/halt95/merlin-engine/blob/v2.5.3/benchmarks/2026-10-06/BENCH-CARD.md).

### What changed in v2.5.1

**v2.5.1 is v2.5.0 plus a prefix-cache fix.** v2.5.1 is v2.5.0's source plus five commits
that touch only the scheduler and the KV-cache manager (no compiled code), served with one more option,
`--prefix-match-unit 64`. v2.5.0 kept a running request's Gated-DeltaNet (GDN) prefix checkpoint, which ended the full
re-prefill of follow-up turns: on v2.2.0 most follow-ups re-prefilled the whole conversation (with 8 concurrent 80K
sessions, cache reuse 7.5 %; v2.5.0: 82.0 %). v2.5.1 removes most of what remained:

- **A follow-up turn resumes on a 64-token grid** instead of the 4,096-token block. `--prefix-match-unit` is a stock
  vLLM 0.30 option and changes only the matching granularity; the pool is unchanged. In one conversation a follow-up
  re-prefills a median of 3,202 tokens (v2.5.0: 9,326; v2.2.0: 6,931), and its median time to first token is 0.96 s
  (v2.5.0: 1.97 s; v2.2.0: 1.56 s). Hits stay valid: 20 prompt pairs, each sent cold and again warm, gave identical
  T=0 output, 20/20, every warm send hitting at the expected 64-token boundary.
- **The GDN checkpoint for a partial tail is held until the request is freed.** Upstream releases its copy after one
  step, so another request's allocation could evict it while the turn was still decoding.
- **A prompt tail that ends on a block boundary is keyed**, so the next turn can find it.
- **A waiting request's cached prefix is pinned until admission**, so other requests' allocations do not evict it
  while it queues.
- **A finished turn's prefix stays referenced until its next turn is admitted.**

With 8 concurrent conversations (temperature 0; 5 minutes),
cache reuse is close to v2.5.0's (92.5 % against 90.6 % of follow-up prompt tokens), the median follow-up waits about
55 % less (1.61 s against 3.54 s), and 22 % more requests start within the 5 minutes, all completed (107 against 88):
v2.5.0 re-prefilled part of every
follow-up, v2.5.1 resumes most of them on the 64-token grid. v2.2.0 was measured on the same shape of load at
temperature 0.6, the load that reproduced the multi-turn cache issue reported on Hugging Face (the multi-turn cache discussion): 7.5 %
reuse and a 52.7 s median follow-up (v2.5.0 on it: 82.0 %, 3.9 s). **What is left:** with the KV pool near full (99.2 %), 4
of 99 follow-ups in the 5 minutes found no cached prefix and re-prefilled the whole conversation (v2.5.0: none), because
the protection yields rather than preempting a running request. Each fix logs a line at startup ([Check it's working](docs/reference.md#check-its-working)).
Figures below marked v2.5.0 were measured on that build and not re-run; v2.5.1 changes neither the model path nor the
pool.

**Capacity increases, with a measured limit.** The KV pool grows from 806,792 to 924,993 tokens (+15 %), with an
attention block size of 4,096 tokens. Six 131K sessions were measured resident at once with 0 preemptions (v2.2.0:
five), and so were three 262,144-token sessions. Under full-pool pressure (the 8-conversation load above, the pool at
99.2 %), 2 preemptions occurred and 107 of 107 requests completed. Session token
totals overstate capacity because each session also holds fixed GDN state blocks.

**Combining marks tokenize as intended.** The transformers pin moves to 5.18.0. On v2.5.0, tokenizer output matched the reference tokenizer on all 92 test strings, and 102 of 102 generation prompts gave
bit-identical output to the previous pin in a fresh compile. The checkpoint's own `tokenizer.json` on Hugging Face, inherited from the Intel AutoRound release, is replaced with the upstream Qwen3.8-Flash-Next file (vocabulary, merges and special tokens identical), so tools that read it directly (the `tokenizers` library, GGUF converters) match upstream too; vLLM serving on 5.18.0 tokenizes identically with either file.

**Image prompts support more images.** Prompts can contain up to 42 images, each capped at 4 MP. On v2.5.0, every card stayed above the 512 MiB safety floor at the limit, and image tile reading was 41 of 42 on the test set, identical to v2.2.0; this is a base-model property.

**Unset environment variables now use the shipped defaults.** `VLLM_E2_PLE_PULL_TRANSPORT` and the five kernel toggles with the `MERLIN_` prefix default to 1; setting one to 0 opts out when the engine is run directly (`serve-v2.5.sh` clears inherited values and
exports the shipped ones). `VLLM_E2_GUARD_MODE` also defaults to `warn` in code now; its reader costs about 0.3 ms per
decode step on TP2 × PP2 (`off` skips it; the published numbers were measured with `warn`). A boot of v2.5.0's release candidate with these variables absent used the same pool, enabled the toggles on every rank, and passed needle, thinking, and no-thinking checks. `VLLM_USE_BREAKABLE_CUDAGRAPH=0` remains required; the engine refuses to start without it.

**Single-stream decode is about 3 % slower than v2.2.0.** v2.5.0's release candidate measured −2.81 % (95 % CI −4.60 % to −1.03 %, p = 0.0087; 4 boots against 7 reference boots: 2 of v2.2.0 and the 5-boot v2 gate that v2.2.0 matched). v2.5.1 changes the prefix cache and the scheduler, not the decode path: it reads −2.87 % against the reference (95 % CI −6.84 % to +1.09 %, p = 0.085, 2 boots against 7) and −0.01 % against v2.5.0 (95 % CI −4.29 % to +4.27 %, 2 boots against 2); a third boot on the release code, with the logging-only exact-counter reader on as on the v2.2.0 and v2.5.0 comparator boots, read −1.12 %. RecoverSSM (the GDN state recovery for MTP speculative decoding, [upstream RecoverSSM](https://github.com/vllm-project/vllm/pull/58863), enabled by `--use-replayssm`) is in use, and its speculative verify runs the Triton GDN decode kernel and replays the state each step. Eight concurrent requests are effectively unchanged at 533.7 against 531.5 tok/s.

**Cold prefill is faster than v2.2.0.** 5,219–5,533 tok/s from 10K to 261K, 1.0–2.8 % above v2.2.0 at every depth. During v2.5 development a release candidate with RecoverSSM and the default 3,392-token block was 6–12 % slower; two changes recovered it. The block is set to 4,096 tokens (`--block-size 4096`, four 1,024-token prefill chunks per block), which costs 0.5 % of the pool, and the fused packed GDN forward stays on under RecoverSSM (it had been switched off for the whole layer, prefill included).

**Stability and quality checks passed.** Structured output 80/80, GSM8K-200 198/200, the 262,144-token needle found exactly, three 262K and six 131K sessions resident with 0 preemptions, the quality measure 0.0334 against a ceiling of 0.0388 (v2.2.0: 0.0338), and 0 engine errors while serving over every run (the ladder, depth, residency and bench boots and the boot of the quality, needle, structured-output, follow-up and cache-validity runs); one boot logged the API server's `EngineDeadError` during the scripted drained shutdown, after every request had finished and all four workers had exited cleanly. On v2.5.0's release candidate, not re-run: a 90-minute mixed soak handled 11,002 requests with 0 errors and 0 HTTP 5xx responses, orderly shutdown completed on all 4 ranks on both boots, and injected PLE faults on rank 0 and rank 1 failed closed, with workers gone and GPUs free in 3.1–5.0 s.

**Published source.** The published commits are the tested series with comments, docstrings, messages and a few
identifiers tidied for release, and upstream ports credited by PR number (authors in NOTICE); see also
[docs/history.md](docs/history.md#what-changed-in-v220).

**Greedy output repeats at full context, with and without MTP** (v2.5.0's release candidate; not re-run, the decode
path is unchanged). Eight identical T=0 requests per depth (prefix cache off, 256 tokens generated) gave one distinct output at 65,536, 130,948, 134,997, 200,000 and 261,800 prompt
tokens, with MTP and without it. The control boot with the five kernel switches set to 0 gave 5 of 8 distinct outputs
at 65K and 8 of 8 at every deeper cell, so the test can fail; the guard counters stayed at zero on every cell. This
holds within one compile cache (see [Known behaviours](docs/reference.md#known-behaviours-of-the-qwen38-flash-next-architecture-in-vllm)).

**Without MTP** (v2.5.0's release candidate; not re-measured). `--use-replayssm` without a speculative config falls
back to the stock GDN path (a warning says so) and the pool is 1,057,314 tokens (measured at the computed attention
block, not v2.5.1's block 4,096 with `--prefix-match-unit 64`). Three 262,144-token sessions and seven
131,072-token sessions were each measured resident at once with 0 preemptions (a fourth and an eighth were not);
single-stream decode is about 82 tok/s at 65K–262K, against 158–174 tok/s with MTP (thinking on, the v2.5.1 decode ladder). `serve-v2.5.sh` always serves with MTP; the no-MTP line is the same
command without `--speculative-config`.


Earlier releases (v2.2.0, v2.0.1, v2, the v1 TP4 layout): [docs/history.md](docs/history.md).

## v2.2.0

> **Historical measurements — v3.0.0 (0.30) figure.** This section retains the original release and engine labels;
> its performance, memory, quality, timing and test-count figures have not been re-measured on 0.32.

Released 2026-09-25. The same shape moved onto the public vLLM `v0.30.0` tag plus 58 commits; a reliability release at
parity with v2: greedy repeatability, any-rank fault detection and faster shutdown. Full notes:
[docs/history.md](docs/history.md#what-changed-in-v220).

## v2.0.1

> **Historical measurements — v3.0.0 (0.30) figure.** This section retains the original release and engine labels;
> its performance, memory, quality, timing and test-count figures have not been re-measured on 0.32.

Released 2026-09-18. The v2 shape plus one engine fix, structured output under concurrency. Full notes, build and
stack: [docs/history.md](docs/history.md#what-changed-in-v201).

## v2

> **Historical measurements — v3.0.0 (0.30) figure.** This section retains the original release and engine labels;
> its performance, memory, quality, timing and test-count figures have not been re-measured on 0.32.

Released 2026-09-17. Tensor parallel 2 × pipeline parallel 2 with expert parallel, the host-mapped n-gram table and
host-resident token embeddings: the KV pool grows from 342,912 to 806,792 tokens. Full notes:
[docs/history.md](docs/history.md#v2-against-v1-at-a-glance).

## v1

> **Historical measurements — v3.0.0 (0.30) figure.** This section retains the original release and engine labels;
> its performance, memory, quality, timing and test-count figures have not been re-measured on 0.32.

The TP4 layout: [docs/history.md](docs/history.md#v1-the-tp4-layout).
