# Changelog

## Merlin 3.0.1

The engine-side subset; the full 3.0.1 list is in the [main changelog](../CHANGELOG.md#merlin-301).

### Flash-Next opt-ins

- **DFlash2:** `merlin serve <model> <dir> --spec dflash2` or `./setup.sh --spec dflash2`
  selects the drafter on the three Flash-Next-family 4 x 3090 P2P rows with qualified geometry.
  Setup downloads and verifies it; serving takes `DFLASH2_MODEL` or `~/models/flash-next-dflash2`.
  Decode measured +5–18 % versus MTP, but the 789,668-token GPU KV pool holds 3 x 262K sessions,
  not 6 x 131K. Other rows and the 27B, Super VL and Ornith families refuse it. MTP remains the default.
- **Host-RAM KV cache:** `--kv-ram-cache SIZE` on setup or serve restores evicted prefix KV blocks
  from RAM instead of recomputing them. It adds no GPU context or simultaneous-session capacity.
  SIZE is GiB per server, split into whole GiB per TP x PP rank, in addition to the server's host-RAM
  budget; locked memory must be allowed. The tier is off by default and experimental in 3.0.1 on the Flash-Next
  family (MTP drafter only) and on the 27B and ThinkingCap (RecoverSSM path only). Ornith 1.5 accepts it, untested ⚫;
  Super VL refuses it. On the reference 4 x RTX 3090 host a 262K restore took 1.2–1.4 s, against about 56 s to recompute. Known limits: a 262K session evicted during heavy
  prefill may be recomputed instead of restored (output identical), and host memory grows in steps over the first
  262K requests while the request history fills (bounded: about 70 MiB on Flash-Next, 280 MiB on the 27B).

See the [Flash-Next controls and limits](../../models/flash-next/docs/reference.md#merlin-301-options).

### Install and deployment diagnostics

- Container image tags use the same `SOURCE_REF` that the build verifies. Direct Docker examples,
  like Compose, publish the API on host loopback.
- A failed uv size probe in `setup.sh` reports the URL and an offline/proxy hint before exiting.
- PLE offload warns when either pull-speed control is disabled, including direct `vllm serve` runs
  that omit the launcher's `VLLM_E2_PLE_FAST_IDS=1` and `VLLM_E2_PLE_EARLY_REQUEST=1`.

## Merlin 3.0.0

**Built on vLLM 0.32.0** (upstream commit `5df63a7203`). Merlin 3.0.0 is the public release after v2.5.3.

### Default reasoning effort: medium on every Qwen3.8 checkpoint

- Every Qwen3.8 checkpoint now serves `reasoning_effort: medium` by default: Flash-Next (was `low`), Swift 1.5 and Swift 1.5
  Huihui (were `xhigh`), the 27B (unchanged) and ThinkingCap (was `xhigh`). A request still chooses its own effort
  through `chat_template_kwargs` (`low` / `medium` / `xhigh`). A trailing `--default-chat-template-kwargs` overrides the
  server default on any checkpoint, and `THINKING` does so on the 27B / ThinkingCap launchers. On the reference host, at the
  shipped sampling, `xhigh` drafted fewer tokens per MTP step than `medium`: Flash-Next prose 2.29 vs 2.66 and code 2.35 vs
  3.06; ThinkingCap prose 2.78 vs 3.27 and code 2.65 vs 3.55 (14-25 % fewer). GSM8K-200 with thinking on scored the same or
  better at `medium` (Swift 198 vs 198, Flash-Next 197 vs 197, ThinkingCap 197 vs 195), and `xhigh` writes longer reasoning.

### 27B on one card: 4 MP images

- The 27B / ThinkingCap single-card rows (`serve-1card.sh` and every TP1 hardware row) take images up to 4 MP
  (4,194,304 pixels) and keep the 262K KV pin; the multi-card layouts keep 8 MP. Three output-identical memory settings
  are on by default for these rows only: a 96 MiB FlashInfer workspace (`VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE`), the MTP
  drafter sharing the target's workspace (`MERLIN_SHARE_DRAFT_WORKSPACE`) and the vision rotary applied in place in the
  QKV buffer (`VLLM_MERLIN_VISION_ROTARY_INPLACE`). On an RTX 3090, 4 MP passed at one and two sequences and after a
  262K needle with 558-563 MiB minimum free (bar 524 MiB), with identity and GSM8K-200 194/200; without the settings the
  card ran out of memory at 3 MP.

### One command: setup.sh

- `git clone ... && cd merlin-engine && ./setup.sh` installs and serves. The bootstrap (`setup.sh`) fetches uv and a
  Python 3.13 into `.merlin/` when the host has none and hands over to `tools/merlin/setup_cli.py` (standard library
  only): host checks with the exact fix for each failure (driver R580+, GPUs and P2P through preflight, GCC/G++ >= 13,
  git history, disk against the plan, host RAM for the model), then the questions one at a time with Enter taking the
  recommended answer computed from the inventory (model: Flash-Next on 4 x 24 GB with >= 96 GiB RAM, 27B when the RAM
  or the cards are short; layout: what `merlin serve` picks, `--tp` / `--instances` where the host allows; models
  folder; local or network access with a generated API key; port), every download with its size (wheel, the 222 locked
  packages, the checkpoint from the Hugging Face API or `release/download-sizes.json`), one confirmation, the engine
  from the release wheel through `release/install-env.sh` + `--verify` (about 1.5 minutes), the checkpoint through the
  venv's resumable `hf download` verified against `models/<model>/checkpoint.sha256` with a `.merlin-verified.json`
  marker (no full re-hash while files are unchanged), `merlin doctor`, the endpoint and a curl test, then `merlin serve`
  in the foreground. `./setup.sh` again starts with the saved answers; `git pull && ./setup.sh` reinstalls only when
  the wheel, lock or commit moved and never re-downloads the model; `--reconfigure`, `--yes --model M [--models-dir D
  --host H --api-key K --port P --tp N --instances N --from-source --no-start --dry-run --simulate SPEC]`;
  `merlin pull <model>` downloads and verifies only. Documented in `docs/SETUP.md`; nothing is installed system-wide.
- Setup hardening: answers are kept per model (another model never inherits a folder or layout, and a
  folder holding another checkpoint is refused); the repo-local Python is on PATH for `install-env.sh` (no system
  `python3` needed); the API key lives in `.merlin/api-key` (0600, `.merlin/` 0700), is shown once and never printed
  in commands; uv's archive is checked against its published sha256 and the pinned digest; disk is budgeted per
  filesystem (wheel, uv cache, new and previous venv, the checkpoint's remaining bytes); `--dry-run` fetches and
  writes nothing; every download is pinned to one published revision;
  unknown sizes stay unknown; saved re-runs ask nothing; `--rehash` forces a full hash and the marker's trade-off is
  documented; `merlin pull` needs no GPU and takes only the model and folder; `--tp` labels are real TP sizes; a
  failed reinstall restores the previous venv; a found `gcc-13` / `g++-13` pair is remembered; a no-P2P single-card
  27B pick recommends `--tp 1 --instances N` (setup only); the wheel URL tries the `%2B` and `.` asset names.

### One launcher

- `tools/merlin/launch.py` is the one launcher for every checkpoint and hardware row: it builds the environment and
  `vllm serve` line from the profile registry and runs the shared start-up checks (venv-toolkit `CUDA_HOME`
  default, build-product re-hash, PLE readiness, image cap, topology selection) once. Every `models/<recipe>/serve*.sh`
  and `tools/merlin/launchers/*.sh` is a short shim for it with the same arguments and knobs. 27B / ThinkingCap TP4
  default to RecoverSSM on, with block 1024, chunk 4096, LPT 832 and a 17,923,833,856 B pin (pool 2,012,105 tokens,
  7.68 × 262K); the 8 MP image headroom at chunk 4096 measured 862 MiB lowest free (2026-10-20 release benchmarks). `CHUNK`, `BLOCK_SIZE` and `LPT` are knobs on
  every checkpoint.
- `merlin serve <checkpoint-dir>` identifies the checkpoint, picks the largest tested layout that fits the host (subsets
  of the idle cards on larger or mixed hosts, `--tp`, `--profile`, `--instances N`, free-card hints) and starts through
  the launcher. `scripts/serve-v2.5.sh`, `scripts/serve-v2.sh` and `scripts/build-v2.5.sh` forward to the current
  entry points with a notice (removed in the release after Merlin 3.0.0).
- Card selection follows the host topology: preflight reads `nvidia-smi topo -m` (PCIe switch / host bridge / NUMA
  links, kept in the inventory JSON) and the kernel's NUMA node per card, and `merlin serve` orders cards by tested
  name, model, NUMA node, host bridge, PCIe switch and index. `--instances N` cuts contiguous sets over that order
  (the 4 x 3090 pairs 0,2 / 1,3 stay), fills every card class with its own row (8 x 4090 24 GB + 8 x 4090 48 GB
  -> 16 single-card servers of two rows), keeps 3090 and 3090 Ti sets same-model where the counts allow, and names
  per class what fits when fewer instances do. `--simulate` takes mixed hosts (`8x4090+8x4090-48g`, `2x3090+2x3090ti`),
  up to 32 cards, and `:numa=K` / `:switch=S` / `:p2p=node|switch` topology options.
- Selector review fixes: a 27B / Super VL row qualified with P2P is never picked, recommended or started without it
  (a small-BAR1 card has no peer; the TP1 row and `--tp 1 --instances N` are named), `merlin doctor` honours `GPUS` /
  `CUDA_VISIBLE_DEVICES` and prints `GPUS=` on the start line of a subset; available host memory is MemAvailable
  capped by the cgroup's remaining allowance and the Flash-Next RAM floor follows the layout (an octo = 77 GiB);
  rows that keep the 24 GB pin (sm_89 32 / 48 GB) carry `pin_vram_gib` so an L40S loses nothing; a no-P2P
  multi-card 27B pick prints the measured aggregate note (4 x 170HX) and the `--tp 1 --instances N` command; `--instances`
  sizes by real partition feasibility; free-card hints name only what runs (busy cards apart, `--tp` / `--profile`
  and a `CACHE_ROOT` carried); explicit `GPUS=` order is kept; the launcher banner shows the evaluated label;
  `merlin list [checkpoint] [--arch] [--cards] [--vram]` gains a hardware column.
- The Swift 1.5, Huihui and Super VL 4 x RTX 3090 rows are labelled tested (2026-10-20 release bench cards, after the 2026-10-17 pre-release bench run); registry
  rows for 27B, Super VL, Swift and Huihui on sm_80 (A100 / A30 / CMP 170HX) and sm_86 48 GB (A6000 / A40) are added
  untested with estimated pins, and cards under their VRAM class get a reduced pool instead of a refusal.

### All-reduce on P2P hosts

- Flash-Next rows with P2P on use FlashInfer's PCIe IPC all-reduce for decode-sized messages
  (`VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=1`) and NCCL's `Simple` protocol for prefill, with vLLM's custom
  all-reduce off (`--disable-custom-all-reduce`). Earlier releases used custom all-reduce with `NCCL_PROTO=LL`. Super VL,
  P2P-off rows and the 27B profiles are unchanged; a P2P-off launch clears an inherited `VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC`.

### One engine and seven recipes

- Consolidates the Merlin vLLM fork at the repository root and recipes under [models/](../../models/README.md). Checkpoints remain on Hugging Face.
- Every recipe (Flash-Next, Swift1.5, Swift1.5-Huihui, Qwen3.8-27B, ThinkingCap, Nemotron Super VL, Ornith 1.5) is version 3.0.0.
- New in 3.0.0: [Ornith 1.5](../../models/ornith1.5/README.md) (ornith-ai's Qwen3.5-122B-A10B fine-tune, W4A16 with INT4 MTP experts) on
  4 × RTX 3090: TP2×PP2 + EP, RecoverSSM, MTP K=3 with uniform verify, 3 × 262K resident. Rows `ornith-quad-3090` (with P2P, tested on
  the reference host) and `ornith-quad-3090-nop2p` (tested on a 4 × RTX 3090 host without P2P); `merlin serve ornith1.5 <dir>`.
- Host-resident embedding table: unset `VLLM_HOST_EMBED_TABLE` now also enables it for Qwen3.5 MoE (the Ornith 1.5 engine support;
  set `VLLM_HOST_EMBED_TABLE=0` to keep the table on the GPU).
- Uses the shared [release/install-env.sh](../../release/install-env.sh) path in place of separate engine bundles and per-package patch series. The 0.32 installer and tile tooling are integrated; `release/requirements-0.32.lock` is complete with 222 exact pins and SHA-256 hashes, resolved with uv and checked by a dry-run `--require-hashes` install. The CUDA toolkit wheels are the 13.0 series matching torch's cuda-toolkit pin (see "Why the toolkit wheels are 13.0" in [release/install-0.32.md](../../release/install-0.32.md)); the lock is compile-qualified on CPU, and the GPU serve check is pending.
- Ships the engine as one prebuilt wheel for sm_80, sm_86 and sm_89 (`release/build-wheel.sh`). `MERLIN_WHEEL=<wheel> release/install-env.sh <venv>` installs it without compiling; the installer accepts only the wheel named in `release/merlin-wheel.sha256`, built from the checkout's native sources. A source install builds the same three architectures. The separate compiled-ops tarball and the experimental native-Marlin build option are removed.
- Links the [upstream vLLM README](https://github.com/vllm-project/vllm/blob/v0.32.0/README.md) and [vLLM documentation](https://docs.vllm.ai/) from Merlin's root [README](../../README.md), with [NOTICE](../../NOTICE) and [security policy](../../SECURITY.md).

### Engine and launchers

- Rebases the fork onto vLLM 0.32.0 (an upstream release candidate plus five upstream commits of the next release candidate, then the final release's one commit, [a macOS-only socket-family fix](https://github.com/vllm-project/vllm/pull/62177) in the Rust engine-core client). One of them, [the upstream /v1/messages protocol change](https://github.com/vllm-project/vllm/pull/61427), changes a file Merlin also edits (the /v1/messages request protocol); the others touch no Merlin-edited file. Eleven previous backports are supplied by upstream, including prefix-cache/MTP fixes, GDN metadata reuse and the 1× M-RoPE cache.
- Retains the sm_86 FP8 QSA reader, exact-need logits backing, PLE offload, RecoverSSM, cache retention, guard telemetry and persistent top-k.
- Replaces AllSpark with Marlin W8A16 for the Flash-Next INT8 GDN projections. The 0.32 requirements include humming-kernels 0.1.17 and instanttensor ≥ 0.2.2.
- Uses `--kv-cache-memory-bytes` and explicit align-mode Mamba caching. Flash-Next FGPC remains off; 27B FGPC remains on via `--enable-mamba-shared-prefix-checkpoint`.
- Bounds prompt-logprob GPU workspace with a default 256-row logits chunk, down from 1,024, while retaining the Flash-Next 4.1e9-byte KV pin and 924,993-token pool.
- Carries fixes for the padded QSA guard sentinel, DCP/EAGLE prefix-cache keys, RecoverSSM pipeline broadcast, PLE boot and table-format validation, PLE event lifetime/watchdog notification and prefill logits lifetime. Main includes the PLE crash fix, the constant QSA logits workspace and the FlashInfer staging fence; the three fixes passed verification on the reference host (for the staging fence, one stress generator was corrected and the run repeated).
- Changes sequence-parallel startup refusal to a warning. SP remains default-off and has layout-equivalent rounding differences.
- Adds generated 0.32 Swift/Huihui launchers with checkpoint-specific KV scales. A qualification run validated the generated Swift and Huihui launchers on 2026-10-16, measured with the BF16 draft head, the shipped default. Swift and Huihui then passed their Merlin 3.0.0 release bench cards (2026-10-20; Swift also the 2026-10-17 pre-release bench run); both are tested on the reference host. Swift's 27 T=0 outputs matched the Flash-Next quad despite argv differences; Huihui was checked on its own boot, image, and GSM8K run.
- Scores the **full vocabulary** in every Flash-Next MTP drafter; no reduced draft vocabulary is on by default (the reduced-vocabulary table is not shipped; the switch was removed in 3.0.1). The draft lm_head precision is one launcher knob, `DRAFT_HEAD=bf16|int8|int4`, independent of any vocabulary table; the shipped default is a single constant per launcher, `bf16`, the v2.5.x drafter. `DRAFT_HEAD=int8` is available: the 2026-10-17 BF16/INT8/INT4 A/B on the vanilla checkpoint measured it at 6.5–7.2 % lower ms per output token than BF16 at 32K/131K/258K with the same MTP acceptance, and it is a candidate default for a later release (3.0.1 keeps `bf16`).
- **`use_local_argmax_reduction` works with the Flash-Next MTP drafter.** `Qwen4ExpMTP` (nvidia and amd) mixes in `LocalArgmaxMixin`, so the flag no longer stops the engine at start-up. Contributed by [brandon-nuttall](https://github.com/brandon-nuttall) in [a pull request to qwen38-flash-next-3090s](https://github.com/halt95/qwen38-flash-next-3090s/pull/1). It is off in every shipped profile: it needs `"draft_sample_method":"greedy"`, which changes how draft tokens are picked, and on the contributor's 7 x 3090 + 1 x 3090 Ti TP2 x PP4 rig it gave +10-20 % 8-way aggregate decode and -8 to -10 % single stream at 4K. On the reference 4 x 3090 Flash-Next layout it gave no gain: 8-way aggregate 459 tok/s with greedy drafting and local argmax, 462 with greedy drafting alone, 483 with the shipped profile.
- Builds target sm_80, sm_86 and sm_89. Reference-host qualification is on RTX 3090; other GPU architectures and eight-card layouts remain untested / community-validated.

- **Fixes a RecoverSSM commit bug in align mode (GDN).** With prefix caching, RecoverSSM and MTP, a window whose accepted tokens crossed a block boundary, while its starting state sat in the block that also received the final state, could store a wrong block-boundary state (the final state was always correct). A later prefix-cache hit restoring that block could then diverge from a cold run. The commit now stores the boundary state and the final state in one traversal. Affects earlier 27B TP2 and Flash-Next releases served with prefix caching; at block size 4,096 and K=3, roughly 1 window in 1,365 could write such a state, and only a prefix hit landing exactly on that block could use it. Upgrading is recommended for deployments that serve with prefix caching.
- **Super VL (Nemotron) ships with RecoverSSM** on the Mamba-2 path; its first-boot front end no longer opens a CUDA context on GPU 0 (the image preprocessor runs eagerly), which returns ~260 MiB to rank 0.
- **Cache-thrash fixes on by default** for hybrid models: cost-aware admission (`VLLM_E2_SCHED_MISS_DEFER_S`, default 120 s) and session-coherent yields. Measured on Flash-Next under 8×131K and 12×80K oversubscription: follow-up reuse 74.7 % and 82.1 %.
- **Cold requests behind cached follow-ups:** a cold request is held back behind waiting prefix-cache hits only when admitting its whole prompt would need a waiting request's pin released or would evict cached blocks a waiting request still needs; otherwise it is admitted at once. A prefix-cache reset that cannot complete keeps the retained follow-up prefixes.
- **Every Qwen recipe serves Froggeric's fixed Qwen3.8 template (v22.5)**: Flash-Next, Swift and Huihui now use the template 27B and ThinkingCap already shipped. It accepts system messages after the first turn, so `/v1/messages` keeps coding agents' per-turn notes in place instead of merging them into the leading prompt, which broke the prefix cache (reported against Flash-Next). `CHAT_TEMPLATE=<path>` serves another template; `CHAT_TEMPLATE=` serves the checkpoint's own.
- **Checkpoint-local FP8 KV scales:** a Flash-Next-family checkpoint that ships `qsa_kv_scales.json` beside its weights is served with its own scales (`SCALES=` still overrides), so a new variant of a supported family runs on this release with `merlin serve flash-next <dir>` and no repository update.
- **27B shared-prefix checkpoints (FGPC)** are reworked as evict-first soft junction checkpoints, one per prefix, with no extra blocks held at rest; 27B prefix-hit counts are unchanged.

### Corrected 2026-10-15 measurements

Design-point and cache rows were measured on RTX 3090 GPUs with a pre-release vLLM 0.32 build. The tile ITL and GSM8K rows used a separate 0.32 candidate build. Engine identity checks are described below; these results do not qualify the final release assembly.

| Measurement | Result |
|---|---|
| Flash-Next 6×127K design | 574.5 tok/s aggregate; 10.78 ms/tok per agent; cold prefill 5,423 tok/s; 0 preemptions |
| 27B TP4 14×127K design (RecoverSSM-off line) | 439.5 tok/s aggregate; 31.79 ms/tok per agent; 0 preemptions |
| Flash-Next, MTP off, 32K follow-up | 0.998 cache-hit ratio; TTFT 5.78 → 0.315 s |
| 27B TP2, MTP off, 32K follow-up | 0.9996 cache-hit ratio; TTFT 18.25 → 0.52 s |
| KV pools | Flash-Next 924,993; 27B TP2 826,689 tokens (measured on 0.32; equal to the earlier pools); 27B TP4 2,012,105 tokens with RecoverSSM on (release bench card PASS), was 1,974,072 on the RecoverSSM-off line |
| 27B tile-on ITL at 4K / 32K / 131K | TP2 19.99 / 21.41 / 25.27 ms; TP4 15.83 / 16.62 / 18.37 ms (RecoverSSM-off line) |
| 27B tile-on GSM8K-200 | 192.0/200 mean |

The FlashInfer **0.7.0.post1** sm_86 FP8 small-q verify tile restores 27B TP4 step-time parity with 0.30. It uses `FLASHINFER_SM8X_FP8_SMALLQ=cta32` and the JIT suffix `_sm8x_smallq_070_v1`. The old untiled 0.32 slowdown figures are superseded. The tile patch and install/proof tooling must accompany the 27B launchers.

### Vision

The release cap is **8 MP (8,388,608 pixels)** across the served recipes, replacing the older 4 MP caps (Flash-Next and 27B), except Nemotron Super VL, which keeps its model-native cap of about 3.4 MP.

27B TP2 serves images up to 8 MP. With the vision-peak option on (`VLLM_MERLIN_VISION_PEAK=1`, the TP2 launcher default), the 50-item TP2 stress sweep's worst minimum free was 806 MiB per card (2026-10-15), above the 524 MiB floor. A separate knob OFF/ON identity check matched tokens and logprobs on 6/6 prompts. 27B TP4 (608-690 MiB free across runs: 690 MiB on 2026-10-15, 608 MiB in the 2026-10-16 mini gate) and Flash-Next (902 MiB) meet the target.

Swift1.5's 2026-10-15 window left 902 MiB free on stage 0; its generated launcher's 2026-10-16 8 MP burst after 8K prefill left 544 MiB minimum free (BF16 draft head, the shipped default), 20 MiB above the 524 MiB floor. Huihui's generated launcher independently measured 902 MiB minimum free. The chunked vision encoder is in the engine and inert at ≤8 MP; a larger cap is deferred to separate qualification.

### Evidence and remaining work

The release-candidate engine and then the merged release-candidate engine were temperature-0 token- and logprob-identical to the pre-release reference on four configurations, with 28/28 documents plus four controls per configuration. A fresh install (2026-10-21) and a from-source build (2026-10-22) then passed on the reference host; the container image is not yet qualified.

Native 262K configurations retain their intended numerics. No shipped profile enables YaRN; a YaRN extension set by hand changes numerics versus the 0.30 releases, and past 262K it needs the 3.0.1 drafter fix. Layout changes and SP can change rounding. Prefix-cache thrash under oversubscription and the upstream JSON-form tool-call parser limitation remain. Two-card Flash-Next is not implemented (it would need a host-RAM expert tier); single-card 27B is tested on the reference host (2026-10-21: 4 MP images, one 262K or two 131K sessions).

### Earlier releases

- Flash-Next: [recipe history](../../models/flash-next/CHANGELOG.md) and [v2.5.3](https://github.com/halt95/qwen38-flash-next-3090s/releases/tag/v2.5.3), including the Huihui twin and v2.5.2 guard idle fix.
- Qwen3.8-27B: [recipe history](../../models/qwen38-27b/CHANGELOG.md) and [v1.3.1](https://github.com/halt95/qwen38-27b-3090s/releases/tag/v1.3.1).
- Swift1.5: [recipe history](../../models/swift1.5/CHANGELOG.md) and [v2.5.1-swift1.5](https://github.com/halt95/Swift1.5-qwen38-flash-next-3090s/releases/tag/v2.5.1-swift1.5).

Historical benchmark cards retain their original release dates and toolchains; 0.30 results are not new 0.32 measurements.
