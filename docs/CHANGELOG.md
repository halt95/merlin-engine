# Changelog

Every Merlin release in one place, newest first. Until 3.0.0 each model shipped from its own repository with its own
version numbers; here each sits under the engine release it ran on, with its original tag named. Figures under the 2.x
releases were measured on vLLM 0.28-0.30 when they shipped and are kept as published; they are not re-measured on 3.0.0.

| Release | Date | Engine | Models |
|---|---|---|---|
| [3.0.1](#merlin-301) | 2026-11-02 | vLLM 0.32.0 | Flash-Next, Swift1.5, Swift1.5-Huihui, Qwen3.8-27B, ThinkingCap, Nemotron 3.5 Super VL, Ornith 1.5 |
| [3.0.0](#merlin-300) | 2026-10-22 | vLLM 0.32.0 | Flash-Next, Swift1.5, Swift1.5-Huihui, Qwen3.8-27B, ThinkingCap, Nemotron 3.5 Super VL, Ornith 1.5 |
| [2.5.3](#v253) | 2026-10-12 | Merlin engine v2.5.2 (vLLM 0.30.0) | Flash-Next, Swift1.5-Huihui (new); Qwen3.8-27B v1.3.0 / v1.3.1 and ThinkingCap (new) |
| [2.5.2](#v252) | 2026-10-12 | Merlin engine v2.5.2 (vLLM 0.30.0) | Flash-Next |
| [2.5.1](#v251) | 2026-10-06 | vLLM 0.30.0 + Merlin commits | Flash-Next, Swift1.5 |
| [2.2.0](#v220) | 2026-09-25 | vLLM 0.30.0 + 58 commits | Flash-Next, Swift1.5 (new); Qwen3.8-27B v1.0.0 (new) |
| [2.0.1](#v201) | 2026-09-18 | vLLM pre-0.30 | Flash-Next |
| [2](#v2) | 2026-09-17 | vLLM pre-0.30 | Flash-Next |
| [1](#v1) | 2026-09 | vLLM 0.28.0 + patches | Flash-Next |

## Merlin 3.0.1

2026-11-02. Built on vLLM 0.32.0, with the same seven models as 3.0.0. Every figure in the 3.0.1 bench tables was re-measured for this release
with the GPU unlock and P2P on release candidates just before the final commit (see below), except rows marked
(3.0.0 run). The DFlash2 and KV RAM cache figures were measured separately on earlier builds.

### Highlights

- **KV RAM cache (experimental).** `--kv-ram-cache 24g` keeps conversation prefixes that the GPU evicts in host RAM,
  and restores them instead of recomputing them. A restore gives exactly the same output as a GPU cache hit. It works on
  the Flash-Next family (MTP drafter only) and Qwen3.8-27B. It's off by default; Ornith 1.5 accepts it but is untested.
  On Flash-Next a 262K restore takes 1.2–1.4 s instead of about 45 s to recompute. Known limits: a 262K session
  evicted during heavy prefill may be recomputed instead of restored, and host memory grows in steps over the first
  262K requests while the request history fills (bounded: about 70 MiB on Flash-Next, 280 MiB on the 27B; in our soaks
  0.18–0.38 % per hour of resident-memory growth while it fills, above the 0.1 % per hour soak limit).
- **More sessions with `--swarm` (experimental).** Raises the session cap on four RTX 3090s, with or without
  P2P: from 8 to 24 on Flash-Next, Swift1.5 and Swift1.5-Huihui, and from 24 to 32 on Qwen3.8-27B and ThinkingCap. It
  turns on a 48 GB KV RAM cache with it. Tested on Flash-Next and the 27B with P2P; Swift1.5 and Swift1.5-Huihui, and
  every model without P2P, use the same cap untested (⚫, expected to work), and the server prints a warning on the
  no-P2P rows. Default caps are unchanged, and other models and layouts don't have a swarm
  cap yet.
- **DFlash2 drafter for the Flash-Next family.** `--spec dflash2` on 4× 3090 with P2P. Decode is 5–18 % faster than MTP,
  and the server holds three 262K sessions instead of six 131K ones. MTP stays the default.
- **GPU unlock support.** [Zagitta's gpu-unlock](https://github.com/Zagitta/gpu-unlock) is documented as an optional
  host upgrade, like P2P (proposed, with before/after measurements, by Sudzpel). In a stock-versus-unlocked A/B on the reference host at 220 W it measured about 16 % faster
  prefill on Flash-Next and 29 % on Qwen3.8-27B (two cards, 262K), with decode unchanged and identical output. From 3.0.1 on, our bench numbers are
  measured with it.
- **Longer context with YaRN.** Flash-Next serves beyond 262K with a YaRN override, tested on the reference host at
  393K and 524K (MTP and DFlash2) and 917K (MTP). The 1M recipe and YaRN on the 27B are untested. Asking for more than
  262K without YaRN is now refused at startup. This applies to every model: a context longer than the model's RoPE
  table without YaRN is refused, and `VLLM_ALLOW_LONG_MAX_MODEL_LEN` no longer overrides it.
- **INT6 embedding table.** `--ple int6` loads a smaller embedding table for Flash-Next and Swift1.5, saving about
  9 GiB of host RAM per server with quality checks passed. It is one shared pack laid over your FP8 checkpoint, which
  stays untouched; `merlin verify <dir> --ple int6` checks it, and `MERLIN_PLE_PACK_DIR` points at a local copy. FP8
  stays the default. Swift1.5-Huihui is refused everywhere (setup, pull, serve and its launcher) and serves with
  `--ple fp8`, because its MTP acceptance drops too far.

### Measured on Merlin 3.0.1

4x RTX 3090 without NVLink, P2P on, Zagitta's gpu-unlock, 220 W, 2026-11-02 bench cards (runs measured 2026-11-01 to 2026-11-02 UTC). Decode: one stream, thinking on (reasoning effort medium where the checkpoint has one; the Nemotron 3.5 Super VL and Ornith 1.5 templates have no effort levels), median of 3, at a 4K / 131K prompt and the full 262,144-token context (261,888 prompt + 128 out). 8 agents: total throughput of eight simultaneous follow-ups on 16K conversations; with thinking on a 512-token follow-up may end inside the reasoning, and reasoning tokens count. GSM8K-200 is correct answers out of 200: thinking on at effort medium on this release (on four cards), thinking off from the 3.0.0 runs, as the weights are unchanged.

| Checkpoint | GPUs | KV pool, tokens | Decode, tok/s, 4K / 131K / 262K | 262K prefill | 8 agents, tok/s, thinking on / off | GSM8K-200, thinking on / off |
|---|---|---|---|---|---|---|
| Flash-Next | 4x | 924,993 | 178 / 183 / 197 | 6,803 tok/s | 677 / 664 | 197 / 199 |
| Swift1.5 | 4x | 924,993 | 176 / 186 / 186 | 6,783 tok/s | 673 / 650 | 199 / 197 |
| Swift1.5-Huihui | 4x | 924,993 | 178 / 189 / 196 | 6,866 tok/s | 703 / 681 | 198 / 198 |
| Qwen3.8-27B | 1x | 271,506 | 125 / 95 / 71 | 579 tok/s | n/a (two at most) | - / 195 |
| Qwen3.8-27B | 2x | 826,689 | 210 / 182 / 143 | 1,115 tok/s | 705 / 615 | - / 194 |
| Qwen3.8-27B | 4x | 2,012,105 | 251 / 243 / 238 | 2,101 tok/s | 954 / 854 | 198 / 197 |
| ThinkingCap | 2x | 826,689 | 203 / 162 / 160 | 1,114 tok/s | 697 / 633 | - / 193 |
| ThinkingCap | 4x | 2,012,105 | 244 / 230 / 238 | 2,099 tok/s | 952 / 832 | 198 / 195 |
| Nemotron 3.5 Super VL | 4x | 1,856,367 (the shipped 1M context) | 103 / 102 / 93 | 3,916 tok/s | 432 / 432 | 197 / 195 |
| Ornith 1.5 | 4x | 894,618 | 146 / 115 / 105 | 2,261 tok/s | 528 / 486 | 196 / 190 |

Against 3.0.0, 262K prefill is 16-17 % faster on the Flash-Next family and 27-33 % faster on the other checkpoints, and four-card 27B time to first token at 262K is 125 s (164 s in 3.0.0). The four-card 27B and ThinkingCap agent totals and the two-card 27B thinking-on total come from overlap windows under 5 s; read them loosely, as on the 3.0.0 cards. Each checkpoint's card is under `models/<checkpoint>/benchmarks/2026-11-02/`, for example [Flash-Next](../models/flash-next/benchmarks/2026-11-02/BENCH-CARD.md) and [Qwen3.8-27B](../models/qwen38-27b/benchmarks/2026-11-02/BENCH-CARD.md).

### Hardware

- **More hardware.** On 4× RTX 3090 without P2P, Ornith 1.5 and Flash-Next's PP4 layout are tested; Flash-Next's
  default layout and the 27B on four cards were boot-checked with `--swarm` (2026-11-01). 4× RTX 4090 and 2× / 4× CMP 170HX
  64 GB are 🟡 testing: full model runs passed on cloud-hosted cards and kernel checks are still to run, so `merlin serve`
  picks them like untested profiles. Tuning continues on all three. 4× CMP 170HX (`fn-sm80-64g-x4-pp4`) was 🟢 tested
  in 3.0.0 and moves to 🟡 testing until its kernel checks run; on a host that matches both a tested and a testing
  profile, `merlin serve` picks the tested one.
- **Flash-Next PP4 on four cards without P2P**, by name: `--profile fn-quad-3090-nop2p-pp4`. Prefill is 47–53 % faster
  and decode 14 % slower than the default TP2×PP2, which stays the default. Its KV pin is now 4.15 GiB per card: a
  906,581-token pool that holds three 262K sessions (one before), tested on the reference host on 2026-10-19.
  `--profile fn-quad-24g-sm89-pp4` is the same layout on 4× RTX 4090 without P2P, by name only and untested ⚫.
- **4× RTX 4090 prefills in 2,048-token chunks** (`fn-quad-24g-sm89`; 1,024 before). On cloud-hosted 4090s this cut the
  131K time to first token from 24.4–27.0 s to 22.5 s (about 8–20 % faster prefill), with decode unchanged. The 8 MP
  image and 262K checks ran at 1,024 and have not run at 2,048 yet; `CHUNK=1024` goes back to the checked chunk.
- **FP8 KV cache for the Flash-Next family on sm_80 (A100, A30, CMP 170HX)** as an opt-in with `KV_DTYPE=fp8`. BF16
  stays its default there (the 27B already uses FP8). Untested.
- **Host RAM for Flash-Next on four cards:** 72 GiB is the tested minimum (64 GiB ran out of memory). Startup is
  refused below about 69 GiB, plus the KV RAM cache size when it's on; with `--ple int6` the floor is about 60 GiB
  (estimated, not boot-tested). 96 GB is the reference configuration; a host with at least 92 GiB MemTotal counts as
  that class. `MERLIN_SKIP_RAM_CHECK=1` keeps a RAM refusal as a warning only (see SETUP).
- **Qwen3.8-27B on WSL2 and on cards without P2P.** On WSL2, Merlin turns on pinned memory, which the 27B,
  Nemotron 3.5 Super VL and Ornith 1.5 need for their host embedding tables, and checks for WSL before starting;
  on Flash-Next it prints how to turn it on instead. `P2P=off` selects the no-P2P layouts.

### Before you upgrade

- **The engine wheel is unchanged from 3.0.0** (no compiled-code changes in 3.0.1; the native sources changed only in comments and message text), so `vllm --version` still prints
  `0.32.0+merlin.7002f56537`; the 3.0.1 Python comes from the checkout.

- **Request limits.** `allowed_token_ids` or `logit_bias` over 1,024 entries, `bad_words` over 128 sequences or 1,024
  tokens in total, or `stop_token_ids` over 128 with `min_tokens` set now return HTTP 400 instead of crashing the
  engine. With speculative decoding, `logit_bias` is refused. Generative
  scoring takes up to 32 items and 32,768 tokens (`MERLIN_GENERATIVE_SCORING_MAX_ITEMS`, `..._MAX_TOKENS`).
- **The API key now also covers** `/tokenize`, `/detokenize` and `/generative_scoring`. `/tokenize` refuses media
  parts.
- **An explicit `VLLM_HOST_EMBED_TABLE=1` is refused at boot** on a model whose embeddings are tied or quantized
  (it used to be skipped silently): unset it or set it to `0`. Unset, the default, is unchanged, and Flash-Next is
  unaffected.
- **The first Qwen3.8-27B start recompiles**, because its compile cache moved into the install (`.vllm-cache`);
  `VLLM_CACHE_ROOT` overrides it.
- **`REDUCED_DRAFT_VOCAB` and `DRAFT_VOCAB` now stop the launch** when set (an empty value, or `REDUCED_DRAFT_VOCAB=0`,
  is accepted). Reduced draft vocabulary is gone (see Removed); unset them.
- **Thinking-off decode at 262K measured lower on five layouts:** Swift1.5 −11.4 %, Qwen3.8-27B on four cards −5.7 %
  and on two cards −2.1 %, Nemotron 3.5 Super VL −1.2 % and Flash-Next −0.4 % against 3.0.0. Step time is the same or better; the gap is MTP draft acceptance over a 128-token sample, and
  the 259K thinking-off cells read +0.8 % and +5.4 % (see the bench cards).

### Fixes

- **`/health` watches for stalls**, on by default with stats logging, one API server and one engine: it returns 503
  when the engine completes no step for 120 s with requests in flight (`VLLM_MERLIN_HEALTH_STALL_S`; `0` turns it off). A failed request logs its cause at ERROR
  and returns a clear error message.
- A single request too large for the KV cache now fails with "KV cache capacity exhausted" instead of hanging.
- The RAM check now respects memory limits set on a container or a systemd service, so a server that can't fit is
  refused before the weights load instead of being killed partway through.
- A worker that dies during startup now fails the boot instead of hanging it.
- Structured output (JSON mode, strict tools, grammars) with speculative decoding is now exact. Draft acceptance
  for these requests reads lower as a result; this is expected.
- Flash-Next refuses to run FP8 KV with a recipe's automatic scales when the checkpoint's config doesn't match them; scales you set yourself with `SCALES` aren't checked. Calibrate a new checkpoint with
  `tools/qsa_calibrate.py` and pass `SCALES=`; the calibration now records which checkpoint it was made on, and the
  launcher warns when the scales name another checkpoint or were not calibrated under the YaRN length being served.
- A Flash-Next checkpoint whose config doesn't set `mamba_ssm_dtype` now keeps its GDN state in float32, as every shipped
  checkpoint does, instead of falling back to bf16.
- Bad sampler settings are rejected with an error before the request is admitted.
- An FP8 embedding table with broken scales (NaN, infinite or zero) and incompletely packed shards are refused at
  load.
- `--skip-tokenizer-init` is refused, and Merlin notes when extra arguments override a recipe value.
- `include_reasoning=false` no longer marks reasoning finished before generation starts; structured output now waits
  for the reasoning to end.
- With thinking off, a non-streaming response now puts the whole output in `content` and reports zero reasoning
  tokens, as streaming does.
- `/v1/messages`: a `tool_result` that carries a `tool_reference` part (deferred tool search) is now rendered as
  `Tool reference: <name>` text instead of failing with HTTP 400, so a history that contains one replays.
- **A boot hang on the Flash-Next family is fixed.** A Triton kernel loading while the host n-gram table was being
  pulled could stall both first-stage ranks. Every model now preloads its boot kernels and synchronises before a
  later kernel load. Two new warnings: `Triton kernel first compiled while serving` (a kernel compiled after startup,
  which costs that request time) and `Triton preload short for ...` (fewer kernels preloaded than expected).
- A dangling link in the vLLM cache directory is caught at startup and by `merlin doctor`, and the 27B keeps its compile
  cache inside the install.
- Sessions that stop while preempted, or on a hybrid model, resume correctly from the prefix cache.
- `n > 1` requests with a large seed no longer overflow.
- `merlin serve --instances N` starts all N servers or none. With `--wait-ready SECONDS` it waits until every
  instance answers `/health` and is still running; if one dies, even after answering while others load, or misses the
  deadline, all are stopped and the command exits 1. If it cannot confirm the stopped servers are gone, it names them
  so you can stop them by hand.
- **YaRN past 262K no longer overruns the drafter.** With a YaRN override, the MTP and DFlash2 drafters kept the
  checkpoint's 262,144-position table, so a longer prompt could stop the engine. The drafter now follows
  `--max-model-len`, and the draft `max_position_embeddings` override covers every eagle-family drafter (MTP,
  DFlash, DSpark) and the text config, so drafts follow the target's YaRN table past 262K (a port of the open upstream
  PR vllm#63432). Reported by brandon-nuttall and R-omk, whose 524K recipes we then tested.
- A failed `generate()` request is always logged at ERROR, and `EngineGenerateError` carries a default message.
- The "Unknown Merlin environment variable" warning no longer fires for setup and launcher `MERLIN_*` names, and the
  per-admission `SimpleCPU tier miss` log is DEBUG unless `MERLIN_KV_HOST_TIER_CENSUS=1`.
- A clean shutdown no longer prints an `EngineDeadError` traceback.
- An always-fatal bounds-guard hit (QSA rope position, uncommitted ring) found at shutdown now fails the rank with
  exit code 1 in every guard mode, including the default `warn`.
- The host n-gram table no longer times out on a cold kernel compile and leaves no orphaned process when killed.
- `install-env.sh --verify` and `--check` work without `uv` on the PATH. With an inherited `CUDA_HOME` that has no
  `nvcc`, it now prints the refusal and exits 1 instead of failing silently, and the FlashInfer patcher checks for GNU
  `patch` before downloading anything.
- The benchmark tool counts speculative-decoding output correctly, so its reported output rate changes.
- **Fixes from upstream vLLM:**
  - top-p no longer emits token 0 when a logit is forced very high, as at the end of a thinking budget;
  - split top-p sampling fixes;
  - hybrid models no longer write state through stale block-table rows left by an earlier, larger batch;
  - hybrid prefix hits: requests sharing an already-cached prompt boundary are no longer admitted one per step;
  - resumable and streaming sessions resume correctly;
  - cuBLAS initialises before memory profiling;
  - pipeline-parallel draft tokens are gathered in order;
  - adaptive verification no longer shrinks the KV cache by its profiling activations;
  - third-party Flash-Next quantizations with quantized q/k/v and a BF16 QSA indexer (such as AWQ builds) load.

### Engine

- The MTP drafter's attention layers now apply the FP8 KV scales. Output is unchanged.
- Qwen 4 readiness: Flash-Next family model types load from HF config, and the config check names a pipeline split
  that keeps the n-gram layers on the first card.
- YaRN M-RoPE with video pruning keeps the checkpoint's original correction window; the cos/sin cache still covers
  the longer positions. No shipped profile reaches this path.
- When freeing KV blocks by yielding one session, the scheduler now picks the smallest session that covers the
  shortfall instead of the first. Output is unchanged.
- With `VLLM_HOST_EMBED_TABLE=1`, the DFlash2 drafter's codebooks also move to host memory.

### Setup and defaults

- **`./setup.sh` recommends the 27B over ThinkingCap on 4× RTX 3090** when Flash-Next doesn't fit. At the served default (reasoning
  effort medium) ThinkingCap is not faster; its earlier lead was measured at xhigh. ThinkingCap stays available with `--model 27b-thinkingcap`.
- **SSE keep-alive every 15 s** on `/v1/chat/completions` and `/v1/completions`, so idle streams survive common proxy timeouts. `-- --sse-keep-alive-interval 0`
  turns it off.
- **Cleaner `./setup.sh` output**: titled sections, aligned options and a downloads table with a total. `NO_COLOR` is
  honoured. The existing questions, their defaults and saved answers are unchanged. Two questions are new and are
  asked only where they apply: **Drafter** on Flash-Next family rows with DFlash2 (default MTP) and **Swarm mode** on
  rows with a swarm cap (default no).
- **`merlin doctor` shows each GPU's PCIe link** and the IOMMU groups.
- **`merlin doctor` reports the GPU unlock state** per card (`on`, `off`, or `unknown` with the reason) and whether the `gpu_unlock` module is loaded,
  on the GeForce Ampere and Ada cards the module supports (tested on RTX 3090; the others untested ⚫). It reads the card's registers in a short-lived child process
  with a 5 s timeout and never writes them; it needs root on the host itself, and in a container, VM or WSL2 it says to check on the host. It runs on a live
  `merlin doctor` only, not with `--dry-run`, `--simulate` or `--inventory`.
- **DFlash2 with a KV RAM cache is refused** before anything downloads. The `--spec dflash2` hint now appears only on
  layouts that can run it.
- **First serve honours `CC` and `CXX`** (setup passes the compiler it found) and uses the venv's Ninja, so a host with
  only `gcc-14`, or with no system Ninja, no longer stops at `first serve needs ...`.
- **A complete checkpoint on a read-only mount** verifies instead of crashing while writing its verified marker.
- The suggested second-server command is quoted, so it works from a checkout path with spaces.
- **`merlin verify <dir>`** re-hashes every file of a checkpoint against its published checksums. `merlin serve`
  only checks file sizes at startup, and suggests `verify` when a checkpoint hasn't been verified.
- **The Flash-Next systemd unit restarts after the engine dies** (`Restart=always`), and stops retrying after five
  failed starts within an hour.
- **Containers:** the image tag matches the source release, and the Docker example binds to `127.0.0.1` instead of
  every interface.
- **CUDA 13.x other than 13.0** is accepted for an inherited `CUDA_HOME`; 13.0 is the tested toolkit.
- **Ornith 1.5 `SEQS=`** now sizes its CUDA-graph capture to match.
- **Launcher defaults:** `VLLM_NO_USAGE_STATS=1` on the Flash-Next family and `HF_HUB_OFFLINE=1` on the 27B and
  Ornith 1.5.
- **Launcher knobs (thanks to brandon-nuttall):** `SEQS` and `MAXLEN` now also apply on Flash-Next and the 27B (the CUDA-graph capture follows
  `SEQS`, and a value off the tested row prints a note), and Flash-Next takes `PP_PARTITION` and passes
  `MERLIN_MARLIN_MOE_CFG` through. An extra `--speculative-config` after `--` (any spelling vLLM accepts: `--speculative_config`, `-sc` or `--speculative-config.KEY`) merges with the
  recipe's; changing its method or K is refused. `SPEC=none` turns speculation off for diagnosis only; it is not a serving lane.
- **Launcher checks:** `/dev/shm` is checked for free space rather than its total size; a `MODEL` and a
  `MERLIN_CHECKPOINT_DIR` that name different folders are refused; the 27B with P2P on drops an inherited
  `NCCL_P2P_DISABLE=1`; and a server on a non-loopback `HOST` warns that the API key does not guard `/invocations`
  or the ops routes.
- **Opt-in FlashInfer workload tile for the 27B** (default off): `MERLIN_FLASHINFER_WORKLOAD_TILE=1
  release/install-env.sh` applies FlashInfer patch 0002 after 0001; see the
  [patch notes](../release/patches/flashinfer-0.7.0.post1/README.md). The Flash-Next family ignores the knob and prints a note saying so.
- **`./setup.sh` takes an install lock** (`.merlin/setup.lock`): a second run in the same checkout stops with a
  message. A successful install removes only its own rollback environment and keeps older `venv-old-*` ones. A failed
  uv size probe reports the URL and an offline/proxy hint.
- **The old `scripts/serve*.sh` shims are kept through Merlin 3.0.x** (the 3.0.0 notes said the next release would
  remove them).
- **ThinkingCap containers serve as `qwen38-27b`**, as bare metal does; `SERVED=thinkingcap` gives the old name.
- The LXC `provision.sh` no longer needs `TILE_PROOF_LOG` to reuse an environment; `install-env.sh --verify` is the
  check.
- PLE offload warns when either pull-speed control is off, including a direct `vllm serve` without the launcher's
  `VLLM_E2_PLE_FAST_IDS=1` and `VLLM_E2_PLE_EARLY_REQUEST=1`.

### Removed

- `VLLM_SPEC_DRAFT_UNDER_TARGET_TOPK`, which was off by default and measured 2–2.5 % slower. Setting it prints a note
  that it is ignored.
- The reduced MTP draft vocabulary: the `REDUCED_DRAFT_VOCAB` / `DRAFT_VOCAB` switches, the bundled table and its
  builder. It was off by default in 3.0.0. Setting either switch (other than an empty value or `REDUCED_DRAFT_VOCAB=0`) now stops the launch with a
  message; `DRAFT_HEAD` still sets the draft head's precision. A direct `vllm serve` still honours
  `VLLM_DRAFT_VOCAB_FILE`; that engine path is dormant and goes in a later release.

### Docs

- The 3.0.0 Flash-Next prefill gain (+12–13 %) is now stated against v2.5.2, which it was measured against.
- The Nemotron 3.5 Super VL host-RAM embedding table is documented. `VLLM_HOST_EMBED_TABLE=0` keeps it on the GPU.
- The 3.0.0 bench cards gained 262K rows on 2026-10-22, from the same runs; each such table says so.
- In the 3.0.0 notes below, one dead link now points at How to serve.

## Merlin 3.0.0

2026-10-22. Built on vLLM 0.32.0. One engine, one install and one command for all seven checkpoints, in one repository.

### In brief

- **Seven checkpoints, one engine.** Flash-Next, Swift1.5, Swift1.5-Huihui, Qwen3.8-27B, ThinkingCap and two new ones,
  Nemotron 3.5 Super VL and Ornith 1.5 (ornith-ai's Qwen3.5-122B-A10B fine-tune), each with a recipe under `models/`.
  Every one runs at its native context (262,144 tokens; Nemotron 3.5 Super VL 1,048,576) with an FP8 KV cache, MTP
  speculative decoding and vision.
- **One command.** `./setup.sh` checks the host, recommends a checkpoint and layout for your GPUs, shows every download
  with its size, installs the prebuilt engine (about 1.5 minutes, no compiling, nothing system-wide) and starts the
  server. Answers can be changed before anything downloads. Walkthrough: [How to serve](how-to-serve.md).
- **One launcher.** `merlin serve` identifies the checkpoint and picks the largest tested layout for the cards it finds
  (`--tp`, `--profile`, `--instances N` for several servers on one host); `merlin doctor` checks the host first.
- **Prebuilt engine for Ampere and Ada.** One wheel for sm_80, sm_86 and sm_89, checked against its published sha256.
- **More hardware support.** Profiles for RTX 3090 without P2P, eight RTX 3090s, RTX 4090 (24 and 48 GB), RTX A6000 / A40,
  A100, A30 and CMP 170HX; see [Which checkpoint runs where](how-to-serve.md#which-checkpoint-runs-where).
- **Faster prefill.** Flash-Next cold prefill 5,020 → 5,660 tok/s at 10K and 5,417 → 6,051 tok/s at 100K against
  v2.5.2 (+12–13 %); Qwen3.8-27B on four cards 2,483 → 3,009 tok/s at 10K (+21 %) against v1.3.0.
- **RecoverSSM on every checkpoint**, including 27B on four cards (pool 1,974,072 → 2,012,105 tokens) and both new
  ones, and **27B on one RTX 3090**: a 271,506-token pool, one 262K or two 131K sessions, images up to 4 MP.

### Measured on Merlin 3.0.0

4× RTX 3090 without NVLink, P2P on, 220 W, 2026-10-20 bench cards (27B one- and two-card agents/GSM8K, ThinkingCap two-card and Nemotron agents/GSM8K are from a
2026-10-17 pre-release build). Decode: one stream, thinking on, median of 3, at a
4K / 131K prompt and the full 262,144-token context (261,888 prompt + 128 out).

| Checkpoint | GPUs | KV pool, tokens | Decode, tok/s, 4K / 131K / 262K | 262K prefill | 8 agents | GSM8K-200 |
|---|---|---|---|---|---|---|
| Flash-Next | 4× | 924,993 | 168 / 170 / 184 | 5,838 tok/s | 638 tok/s | 199 |
| Swift1.5 | 4× | 924,993 | 177 / 180 / 177 | 5,834 tok/s | 643 tok/s | 197 |
| Swift1.5-Huihui | 4× | 924,993 | 187 / 188 / 189 | 5,880 tok/s | 660 | 198 |
| Qwen3.8-27B | 1× | 271,506 | 124 / 95 / 73 | 456 tok/s | n/a (two at most) | 195 |
| Qwen3.8-27B | 2× | 826,689 | 207 / 173 / 142 | 858 tok/s | 517 tok/s | 194 |
| Qwen3.8-27B | 4× | 2,012,105 | 250 / 235 / 215 | 1,596 tok/s | 752 tok/s | 197 |
| ThinkingCap | 2× | 826,689 | 198 / 182 / 165 | 860 tok/s | 502 tok/s | 193 |
| ThinkingCap | 4× | 2,012,105 | 253 / 255 / 248 | 1,594 tok/s | 733 tok/s | 195 |
| Nemotron 3.5 Super VL | 4× | 1,392,275 at 262K (1,856,367 at the shipped 1M context) | 97 / 101 / 91 | 2,951 tok/s | 429 tok/s | 195 |
| Ornith 1.5 | 4× | 894,618 | 144 / 106 / 100 | 1,704 tok/s | 468 | 190 |

Ornith 1.5 was measured on 2026-10-21 with its engine changes (uniform MoE verify on) before they merged into 3.0.0;
its bench card is [models/ornith1.5/benchmarks/2026-10-21](../models/ornith1.5/benchmarks/2026-10-21/BENCH-CARD.md).

### New checkpoints

- **Nemotron 3.5 Super VL** (NVIDIA, 120B-A12B MoE): W4A16 routed experts with an INT4 MTP head, tensor parallel 2 ×
  pipeline parallel 2 with expert parallel on four cards, RecoverSSM on its Mamba-2 path. It serves its native
  1,048,576-token context by default (pool 1,856,367 tokens): a 1,048,052-token needle came back exact and, re-sent, was
  served from the resident cache (1,047,936 tokens cached, 3 s to first token). At a 262K context the pool held 5 × 262K
  and 9 × 131K sessions at once with 0 preemptions; needles 23/23 at 262K and 10/10 at 131K (2026-10-17 pre-release build). One image per prompt,
  downscaled to the model's native ~3.4 MP.
- **Ornith 1.5** (ornith-ai's Qwen3.5-122B-A10B fine-tune, 122B MoE): INT4 routed experts, INT4 MTP experts, INT8
  GDN projections and calibrated FP8 KV scales; tensor parallel 2 × pipeline parallel 2 with expert parallel,
  RecoverSSM, MTP K=3 with uniform MoE verify (the verify batch through one MoE graph). The INT4 MTP experts accept as the BF16 ones do (mean acceptance length 2.943 against 2.947). Pool
  894,618 tokens (3 × 262K); three ~262K prompts sent at once all came back with their needles; a recorded tool-use
  session replayed with 52/52 tool calls valid. Tested with P2P on the reference host and without P2P on a second
  4× RTX 3090 host (pool 977,838 tokens, GSM8K-200 193/200). Checkpoint about 66.4 GiB.
- **Qwen3.8-27B on one card.** The single-card layout returns, tested on an RTX 3090 with Qwen3.8-27B: pool 271,506
  tokens, decode 124 / 95 / 73 tok/s at 4K / 131K / 262K, 262K prefill 456 tok/s and, on the 2026-10-17 pre-release build, GSM8K-200 195/200 and needles 5/5 at
  131K and 12/12 at 262K. ThinkingCap serves on the same profile but was not run on one card.

### One install, one command

- **`./setup.sh`** fetches uv and a Python 3.13 into the checkout when the host has none, then checks the host and
  names the exact fix for each failure: driver R580 or later, GPUs and P2P, GCC/G++ 13 or newer, disk against the
  plan, host RAM for the model. It asks one question at a time, with Enter taking the answer it recommends for the
  hardware it found: model, layout, models folder, local or network access (with a generated API key) and port.
- **Every download shown before it starts:** the engine wheel, the 222 locked packages and the checkpoint, each with
  its size, then one confirmation. The checkpoint download resumes, is pinned to one published revision and is checked
  against the release checksums; a marker skips the full re-hash while the files are unchanged (`--rehash` forces it).
  Setup ends with `merlin doctor`, the endpoint, a curl test and the server in the foreground.
- **Re-runs and updates.** `./setup.sh` again starts with the saved answers and asks nothing; `git pull && ./setup.sh`
  reinstalls only when the wheel, lock or commit moved and never downloads the model again; a failed reinstall
  restores the previous environment. `--reconfigure`, `--yes --model M` and `--dry-run` (fetches and writes nothing)
  are there for scripted installs, and `merlin pull <model>` downloads and verifies a checkpoint without a GPU.
- **Careful with your host.** Answers are kept per model, and a folder that holds another checkpoint is refused. The
  API key is stored owner-only, shown once and never printed in commands. uv's archive is checked against its
  published sha256. Disk is budgeted per filesystem.
- **`merlin serve <checkpoint>`** identifies the checkpoint, picks the largest tested layout that fits the idle cards
  and starts the server through one launcher. Every layout is a profile; every knob (`PORT`, `HOST`, `KV_BYTES`,
  `MAX_PIXELS`, `CHUNK`, `BLOCK_SIZE`, `LPT`, ...) is an environment variable. `--dry-run` prints the resolved command.
- **Card selection follows the board.** Cards are ordered by tested model, NUMA node, host bridge and PCIe switch, read
  from `nvidia-smi topo -m`. `--instances N` cuts separate card sets over that order, each with its own ports and
  compile cache, and fills mixed hosts class by class. A layout qualified with P2P is never picked without it; the
  one-card layout and `--tp 1 --instances N` are named instead. On 4× CMP 170HX, eight agents at 16K measured 873 tok/s
  on `--tp 1 --instances 4`, 523 on two TP=2 servers and 371 on one TP=4 server.
- **`merlin doctor`** checks GPUs, driver, host RAM and shared memory, honours `GPUS` / `CUDA_VISIBLE_DEVICES`, and
  reads available memory within a container's limit. `merlin list` shows every profile with its hardware.
- **Hardware profiles** beyond the reference host: RTX 3090 without P2P, eight RTX 3090s, RTX 4090 (24 and 48 GB),
  RTX A6000 / A40, A100, A30 and CMP 170HX. 4× CMP 170HX (64 GB) is tested; the others ship untested and expected to
  work. A card below its VRAM class gets a smaller pool instead of a refusal.
- **Prebuilt wheel.** One wheel for sm_80, sm_86 and sm_89 installs without compiling. The installer accepts only the
  wheel named in the published sha256, built from the checkout's native sources; a source install builds the same three
  architectures (about 1.5 hours on 8 cores). Requirements are locked to 222 exact pins with SHA-256 hashes, with the
  CUDA 13.0 toolkit wheels. A fresh install (2026-10-21) and a from-source build (2026-10-22) passed on the reference
  host.

### Engine

- **Rebased onto vLLM 0.32.0** (an upstream release candidate, the five upstream commits of the next release candidate, then the final release's one commit,
  [a macOS-only socket-family fix](https://github.com/vllm-project/vllm/pull/62177)). Eleven earlier backports now come
  from upstream, including prefix-cache and MTP fixes, GDN metadata reuse and the 1× M-RoPE cache. The rebased engine
  was temperature-0 token- and logprob-identical to the reference build on four configurations.
- **FP8 KV cache on Ampere.** Merlin's sm_86 FP8 attention reader for Flash-Next is kept, and 27B gets the FlashInfer
  0.7.0.post1 sm_86 FP8 small-query verify tile, which restores four-card step-time parity with vLLM 0.30: inter-token
  latency at 4K / 32K / 131K 19.99 / 21.41 / 25.27 ms on two cards and 15.83 / 16.62 / 18.37 ms on four.
- **Tables in host RAM.** Flash-Next's n-gram embedding table (about 47.7 GiB) lives in system memory, one copy shared
  by every GPU. The token embedding tables of the main model and its MTP drafter are read from pinned host memory:
  frees VRAM, no speed loss (about 1.2 GiB per card on 27B at TP=2). It is now on by default for Qwen3.5 MoE too, the
  Ornith 1.5 support (`VLLM_HOST_EMBED_TABLE=0` keeps the table on the GPU).
- **RecoverSSM on every checkpoint.** [RecoverSSM](https://github.com/vllm-project/vllm/pull/58863) recovers the GDN and
  Mamba-2 state for MTP speculative decoding, on by default for all seven checkpoints.
- **Cache retention and thrash protection.** Follow-ups resume from cache on a 64-token grid. Cache-thrash fixes are on
  by default for hybrid models: cost-aware admission and session-coherent yields. Under 8 × 131K and 12 × 80K
  oversubscription, follow-up turns reused 74.7 % and 82.1 % of their prompt from cache. A cold request waits behind
  cached follow-ups only when admitting it would evict blocks they still need.
- **PCIe all-reduce.** Flash-Next on P2P hosts uses FlashInfer's PCIe all-reduce for decode and NCCL `Simple` for
  prefill, in place of vLLM's custom all-reduce with `NCCL_PROTO=LL`.
- **Marlin W8A16** replaces AllSpark for the Flash-Next INT8 DeltaNet projections.
- **Bounds guards with idle P8.** The runtime bounds guards stay on, and idle cards drop to P8: 110 W idle on four RTX
  3090s, against 446 W when the guard reader kept them awake.
- **Leaner memory use.** The prompt-logprob GPU workspace is bounded by a 256-row logits chunk (was 1,024), with the
  Flash-Next pool unchanged at 924,993 tokens. 27B shared-prefix checkpoints are evict-first, one per prefix, with no
  extra blocks held at rest. Nemotron's image preprocessor no longer opens a CUDA context on GPU 0, returning about
  260 MiB to the first card.
- **New fine-tunes run as is.** A Flash-Next-family checkpoint that ships its own FP8 KV scales is served with them
  (`SCALES=` still overrides), with no repository update.
- **Upstream protocol change.** The upstream [/v1/messages protocol change](https://github.com/vllm-project/vllm/pull/61427)
  is merged with Merlin's own edits to that endpoint.
- `use_local_argmax_reduction` works with the Flash-Next MTP drafter, contributed by
  [brandon-nuttall](https://github.com/brandon-nuttall). It is off in every shipped profile, since it needs greedy
  drafting: it gave +10–20 % eight-way aggregate decode on the contributor's eight-card rig and no gain on the
  reference layout (459 tok/s against 483 for the shipped profile).

### Per-model recipes

- **Flash-Next, Swift1.5 and Swift1.5-Huihui.** Every MTP drafter scores the full vocabulary with a BF16 draft head;
  `DRAFT_HEAD=int8` is available (6.5–7.2 % lower ms per output token at the same MTP acceptance) and planned as the
  default in a later release (3.0.1 keeps `bf16`), and no reduced draft-vocabulary table ships (`REDUCED_DRAFT_VOCAB=1` is unavailable). The Swift1.5 and Huihui launchers are
  generated from the checkpoint profile with their own KV scales, both tested on the reference host. The pool holds
  3 × 262K and 6 × 131K sessions at once, 0 preemptions, and an 8 MP image fits at the full pin. Flash-Next decode
  with thinking on is 1.4–5.8 % faster than v2.5.2 at every depth (187.4 against 177.2 tok/s at 259,587 tokens).
- **Qwen3.8-27B and ThinkingCap.** RecoverSSM on every layout, MTP K=4 with an INT4 MTP head, shared-prefix checkpoints
  on, and the 0.7.0.post1 verify tile. Four cards: pool 2,012,105 tokens (7.68 × 262K), cold prefill +15 % at 100K and
  time to first token at 255K 179.3 → 161.6 s against v1.3.0, step time unchanged. Two cards take 8 MP images (806 MiB
  worst-case free per card over a 50-image sweep, above the 524 MiB floor).
- **27B on one card at 4 MP.** The single-card layout caps images at 4 MP (4,194,304 pixels) and keeps the 262K KV
  pin, with three output-identical memory settings on by default for it alone: a 96 MiB FlashInfer workspace, the MTP
  drafter sharing the target's workspace, and the vision rotary applied in place. On an RTX 3090, 4 MP passed at one and
  two sequences and after a 262K needle with 558–563 MiB minimum free (bar 524 MiB); without the settings the card ran
  out of memory at 3 MP.
- **Nemotron 3.5 Super VL at its native 1M.** The 1,048,576-token context is the default, with RecoverSSM and MTP K=3
  at the same KV pin as the 262K pool; one full 1M session fits on four RTX 3090s, or five 262K sessions.
- **Ornith 1.5.** Two profiles, with and without P2P, both tested: `merlin serve ornith1.5 <dir>`. Uniform MoE verify is
  on (`UV=0` turns it off) and was bitwise identical to verify off at one session.

### Defaults

- **Reasoning effort `medium` on every Qwen3.8 checkpoint** (Flash-Next was `low`; Swift1.5, Swift1.5-Huihui and
  ThinkingCap were `xhigh`). A request still picks its own effort through `chat_template_kwargs`. At `medium` the MTP
  drafter lands more tokens per step than at `xhigh` and GSM8K-200 scored the same or better: Flash-Next prose 2.66
  against 2.29 and code 3.06 against 2.35; ThinkingCap prose 3.27 against 2.78 and code 3.55 against 2.65. Nemotron's
  template has no `reasoning_effort` (its own switch is `medium_effort`); Ornith 1.5's has only `enable_thinking`.
- **One chat template.** Every Qwen recipe serves froggeric's fixed Qwen3.8 template (v22.5), which accepts system
  messages after the first turn, so coding agents keep their prefix cache across turns. `CHAT_TEMPLATE=<path>` serves
  another, `CHAT_TEMPLATE=` the checkpoint's own.
- **8 MP image cap** on every checkpoint (was 4 MP), except Nemotron (its native ~3.4 MP) and one-card 27B rows (4 MP).
- **Full-vocabulary MTP drafter** with a BF16 draft head on the Flash-Next family (`DRAFT_HEAD=bf16|int8|int4`).
  Nemotron and the 27B family use INT4 MTP heads, Ornith 1.5 INT4 MTP experts.

### Docs and repository

- **One repository.** The Merlin vLLM fork sits at the root and each checkpoint has a recipe under `models/`, every one
  at version 3.0.0; checkpoints stay on Hugging Face. One installer replaces the separate engine bundles, per-package
  patch series and compiled-ops tarball of earlier releases.
- **[How to serve](how-to-serve.md)** walks through setup end to end: the questions, talking to the server, serving on
  your network, choosing GPUs, switching models and fixing problems.
- **Bench cards** for every checkpoint, dated 2026-10-20 (Ornith 1.5: 2026-10-21): decode ladder with thinking on and
  off, exact 262K, cold prefill, sessions held at once, images, agents and GSM8K-200, with the previous release beside
  it where one exists. Earlier cards keep their original dates and toolchains.
- The README links the upstream vLLM README and documentation, with a NOTICE and a security policy.

### Fixes

- **RecoverSSM in align-mode prefix caching.** With prefix caching, RecoverSSM and MTP, a window whose accepted tokens
  crossed a block boundary could store a wrong block-boundary state (the final state was always correct), and a later
  cache hit on that block could diverge from a cold run. The boundary and final states are now stored in one pass. It
  affects earlier 27B two-card and Flash-Next releases served with prefix caching: at block size 4,096 and K=3,
  roughly 1 window in 1,365 could write such a state.
- **Engine fixes carried into 3.0.0:** the padded guard sentinel, prefix-cache keys for DCP and EAGLE, the RecoverSSM
  pipeline broadcast, the host n-gram table's boot, format validation and event lifetime, prefill logits lifetime and a
  FlashInfer staging fence.
- A prefix-cache reset that cannot complete keeps the retained follow-up prefixes.
- Sequence parallelism warns instead of refusing to start; it stays off by default.

### Upgrading

Make a fresh checkout and run `./setup.sh`: a models folder that already holds a checkpoint is checked against the
release checksums and reused rather than downloaded again. The old serve scripts (`scripts/serve-v2.5.sh` and
others) forward to the new launcher with a notice and are removed in the next release. Merlin 3.0.0 does not carry the
reduced-vocabulary INT4 draft head of an earlier, unpublished Flash-Next build.
Upgrading is recommended for anyone serving with prefix caching, for the RecoverSSM fix above. A clone of
qwen38-flash-next-3090s can point at the new name with
`git remote set-url origin https://github.com/halt95/merlin-engine.git`.

## v2.5.3

2026-10-12. Engine unchanged: Merlin engine v2.5.2.

- **New: Swift1.5-Huihui** (Flash-Next tag `v2.5.3`), Swift1.5 with huihui-ai's abliteration on its 96 dense output
  projections. It does not refuse: 29 of 278 StrongREJECT prompts refused (10.4 %). GSM8K 96.29 % against Swift1.5's
  95.98 % in the same run.
- **New: Qwen3.8-27B v1.3.0** on Merlin engine v2.5.2 plus 27B patches, two and four cards. Single-stream thinking
  decode at 4K / 32K / 128K / 255K: 206 / 198 / 172 / 144 tok/s on two cards, 258 / 249 / 238 / 226 on four.
  RecoverSSM on two cards (pool 826,689, three 262K sessions resident), pool 1,974,072 on four (seven 262K sessions).
  FlashInfer 32-row verify tile, MTP K=4, cached-token counts in every response, a per-image pixel cap.
- **New: ThinkingCap**, BottleCap AI's reasoning fine-tune of Qwen3.8-27B, served as a profile of 27B v1.3.0.
- **Qwen3.8-27B v1.3.1** (2026-10-13), documentation only: the glibc floor is 2.34 (Ubuntu 22.04, Debian 12, RHEL 9).

## v2.5.2

2026-10-12. Merlin engine v2.5.2 = v2.5.1 plus one fix.

- **Idle cards drop to P8 with the bounds guards on.** The guard reader polled every 50 ms forever, holding idle RTX
  3090s in P2: 446 W against 110 W in P8 on four cards. It now parks once the engine drains, after a final read of every
  counter, so no bounds check is lost. `VLLM_E2_GUARD_IDLE=0` restores the old behaviour. The same fix ships in
  Qwen3.8-27B v1.3.0. Thanks to the community member who reported it and proposed the first fix.
- No kernel, scheduler or numerics change; decode and prefill read as v2.5.1's within run-to-run noise.

## v2.5.1

2026-10-06. Flash-Next and Swift1.5 (tag `v2.5.1-swift1.5`). v2.5.0 was not published.

- **Agent follow-up turns resume from cache** on a 64-token grid instead of the 4,096-token block: median time to
  first token on follow-ups 0.96 s (v2.2.0: 1.56 s). With eight deep conversations at once, 92.5 % of follow-up prompt
  tokens came from cache.
- **KV pool 806,792 → 924,993 tokens (+15 %)**: six 131K or three 262K sessions resident with 0 preemptions.
- Cold prefill 5,219-5,533 tok/s from 10K to 261K, 1.0-2.8 % above v2.2.0; single-stream decode about 3 % below it.
- Transformers 5.18.0 (combining marks tokenize as upstream), up to 42 images per prompt at 4 MP each.
- GSM8K-200 198/200, structured output 80/80, the 262K needle found exactly.

## v2.2.0

2026-09-25.

- **Flash-Next** moves onto the public vLLM 0.30.0 tag plus 58 commits: a reliability release at parity with v2
  (greedy repeatability, any-rank fault detection, faster shutdown).
- **New: Swift1.5** (tag `v2.2.0-swift1.5`, 2026-09-28), the Merlin checkpoint with the 300 dense tensors Swift 1.5
  changed, its own FP8 KV scales and `xhigh` effort by default.
- **New: Qwen3.8-27B v1.0.0** (2026-09-27), the first 27B release: stock vLLM 0.30.0 plus five patches, two cards (pool
  794,351) or four (1,949,844), FP8 KV, MTP K=3, froggeric's chat template.

## v2.0.1

2026-09-18. The v2 shape plus one engine fix: structured output under concurrency.

## v2

2026-09-17. Flash-Next moves from four-way tensor parallel to **tensor parallel 2 × pipeline parallel 2** with expert
parallel, the FP8 n-gram table served over a host-mapped pull transport and the token embeddings moved into pinned
host memory. The KV pool grows from 342,912 to **806,792 tokens**: three full 262K sessions resident at once.
Single-stream decode 162 / 166 / 169 / 176 tok/s at 4K / 32K / 131K / 261K (parity with v1); prefill 10-26 % faster.

## v1

2026-09. The first Flash-Next release: vLLM 0.28.0 plus community commits and six patches, four-way tensor parallel with
expert parallel, MTP K=3, KV pool 342,912 tokens.
