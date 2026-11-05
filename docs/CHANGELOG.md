# Changelog

Every Merlin release in one place, newest first. Until 3.0.0 each model shipped from its own repository with its own
version numbers; here each sits under the engine release it ran on, with its original tag named. Figures under the 2.x
releases were measured on vLLM 0.28-0.30 when they shipped and are kept as published; they are not re-measured on 3.0.0.
Figures under 3.0.0 and 3.0.1 were measured on vLLM 0.32; the 3.0.2 figures are measured on 0.33 except rows marked (3.0.1 run) or (0.32 tree).

| Release | Date | Engine | Models |
|---|---|---|---|
| [3.0.2](#merlin-302) | 2026-11-05 | vLLM 0.33.0 | Flash-Next, Swift1.5, Swift1.5-Huihui, Qwen3.8-27B, ThinkingCap, Nemotron 3.5 Super VL, Ornith 1.5 |
| [3.0.1](#merlin-301) | 2026-11-02 | vLLM 0.32.0 | Flash-Next, Swift1.5, Swift1.5-Huihui, Qwen3.8-27B, ThinkingCap, Nemotron 3.5 Super VL, Ornith 1.5 |
| [3.0.0](#merlin-300) | 2026-10-22 | vLLM 0.32.0 | Flash-Next, Swift1.5, Swift1.5-Huihui, Qwen3.8-27B, ThinkingCap, Nemotron 3.5 Super VL, Ornith 1.5 |
| [2.5.3](#v253) | 2026-10-12 | Merlin engine v2.5.2 (vLLM 0.30.0) | Flash-Next, Swift1.5-Huihui (new); Qwen3.8-27B v1.3.0 / v1.3.1 and ThinkingCap (new) |
| [2.5.2](#v252) | 2026-10-12 | Merlin engine v2.5.2 (vLLM 0.30.0) | Flash-Next |
| [2.5.1](#v251) | 2026-10-06 | vLLM 0.30.0 + Merlin commits | Flash-Next, Swift1.5 |
| [2.2.0](#v220) | 2026-09-25 | vLLM 0.30.0 + 58 commits | Flash-Next, Swift1.5 (new); Qwen3.8-27B v1.0.0 (new) |
| [2.0.1](#v201) | 2026-09-18 | vLLM pre-0.30 | Flash-Next |
| [2](#v2) | 2026-09-17 | vLLM pre-0.30 | Flash-Next |
| [1](#v1) | 2026-09 | vLLM 0.28.0 + patches | Flash-Next |

## Unreleased

Nothing yet.

## Merlin 3.0.2

2026-11-05. Built on vLLM 0.33.0 (upstream tag `v0.33.0`, 943 commits after 0.32.0), with the same seven models as 3.0.1. Every figure in the 3.0.2 bench tables was measured for this release on the host described under "Measured on Merlin 3.0.2", except rows marked (3.0.1 run) or (0.32 tree).

### Highlights

- **vLLM 0.33.0 base.** The engine moves from 0.32.0 to 0.33.0: torch 2.14.0 (NCCL 2.30.7), torchvision 0.29.0, Triton 3.8.0,
  FlashInfer 0.7.2.post1, humming-kernels 0.1.19, Transformers 5.20.2 and Pillow 13.0.0. The CUDA 13.0 toolkit pins and the
  sm_80, sm_86 and sm_89 build are unchanged (upstream dropped its CUDA 12.9 wheels; the 0.32 lock already used CUDA 13.0). The Merlin
  changes were carried onto 0.33 by hand; the upstream changes we did not take are listed under "Held back from 0.33"
  below. The release gate re-ran load and a bench card on Flash-Next, Qwen3.8-27B (1, 2 and 4 GPUs), Nemotron 3.5 Super VL
  and Ornith 1.5, plus the KV host-tier hit rate and the 27B tile proof, on the new stack: each of these lanes booted and
  served a 262K request; the KV host tier reached 98.3 % (Flash-Next) and 98.2 % (Qwen3.8-27B) of the trace ceiling; the
  tile proof passed. Token identity against 3.0.1 at temperature 0 ran on Flash-Next, Qwen3.8-27B on 2 GPUs and Super VL
  (Ornith 1.5 and the 1- and 4-GPU 27B layouts had no identity run). The 27B stayed within the identity bar (KL at most
  1e-3). Super VL stayed inside the spread 3.0.1 shows between its own boots. Flash-Next's temperature-0 output differs
  from 3.0.1's, within the spread 3.0.1 itself shows between serving one request and eight at once (KL 2.66e-2 against
  3.99e-2; 3.0.1 serving one request alone is bit-exact). A Flash-Next quality A/B in one session (3.0.1, then 3.0.2, then
  3.0.1 again) scored the full GSM8K test set at 1,281 of 1,319 (97.1 %) on 3.0.2 against 1,276 and 1,275 (96.7 %) on
  3.0.1. On the nine 32K needle prompts 3.0.2 refused two more requests than 3.0.1 (it answered 4 of 9, 3.0.1 6 of 9 in
  both runs; the needle is phrased as a secret passphrase), with no wrong recall on any side: every miss was a refusal.
  Swift1.5 and Swift1.5-Huihui booted, served a 262K request and had their bench cards re-run on 3.0.2 on 2026-11-05
  (no identity run); ThinkingCap was not run on the 3.0.2 wheel before release. The NCCL and P2P sweep, the
  vision check, the Triton requalification and the heavier A/Bs were not run for 3.0.2.
- **FlashInfer 0.7.2.post1 with the sm_86 FP8 patches re-ported.** The small-query verify tile (patch 0001) and the
  workload plan mode (patch 0002) are rebuilt for the new FlashInfer and keep their switches and defaults: the tile is on
  only where the launcher sets `FLASHINFER_SM8X_FP8_SMALLQ=cta32` (the 27B), and the workload plan mode stays off unless
  `MERLIN_FLASHINFER_WORKLOAD_TILE=1` is set at install. The JIT module names changed (`_sm8x_smallq_072_v1`, `..._v2`), so
  no 0.32-era compiled module is reused. The 3.0.1 tile figures do not carry over; the 3.0.2 ones are in the bench cards.
  The live tile proof passed on the 27B TP2 boot of the release gate (2026-11-04). The knob-off v1-versus-v2 parity
  check has not been done for 3.0.2.
- **PLE table on NVMe (experimental; measured on the 0.32 tree, not run on 0.33 before release).** `--ple-store disk` on the Flash-Next family reads the embedding table's rows from
  the checkpoint's own files instead of holding the table in host RAM. Free memory after startup rises by 47.7–47.8 GiB with
  `--ple fp8` and 38.8–38.9 GiB with `--ple int6`, and the output was token for token identical to `ram` (304 of 304
  comparisons with FP8, 225 of 225 with INT6, temperature 0). The costs against `ram`, on 4× RTX 3090 with one Gen4 NVMe:
  decode is +1.8 % (FP8) and +2.3 % (INT6) once the table is cached, up to about +5 % on all-new random text at 262K, with
  the first token unchanged; the first send of a text whose rows are not cached yet decodes 9 % (FP8) and 11 % (INT6)
  slower, and the second send about 3 % slower; the first token of a cold 262K prompt takes +7 % to +16 % longer on real text and
  +53 % on all-new random text (FP8), or +16 % (INT6, chat). With `--kv-ram-cache 96g` the table never fully warms: about
  +10–11 % decode at 262K, +3.5 % on steady text and +15–30 % first token, for as long as it runs. Other disks and layouts
  are untested ⚫, and a slower disk will cost more. **These figures were measured on the 0.32 tree**; the 0.33 re-run
  has not been done for 3.0.2. The server stops if a table file is written, touched, `chmod`ed, linked, moved or removed while it
  runs. `ram` stays the default, and Enter at setup's question means RAM. Setup asks after the models folder, on local NVMe
  only; `merlin serve` warns on other storage and runs. The warm-up read after startup only runs if the free memory, less
  any KV RAM cache and 4 GiB, can hold the whole table. Costs and limits:
  [How to serve](how-to-serve.md#the-ple-table-on-nvme-experimental).
- **`--context SIZE` (experimental).** Serves past 262K with YaRN built from the checkpoint's config, for `merlin serve`
  and `./setup.sh`; a bare `--max-model-len` above native now adds the same recipe. A size above the layout's KV pool is
  refused. The recipes named in the Flash-Next reference were tested on the 0.32 tree, none on 0.33 before release; the rest are
  untested ⚫. The generated factor-2 recipe with the DFlash2 drafter is labelled "untested pending gate"; that gate was not run for
  3.0.2.
- **Changes to pipeline-parallel decode with RecoverSSM.** Five changes on the Flash-Next TP2×PP2 and other
  PP layouts, none of which changes the tokens by construction (the on-versus-off comparison has not been done for 3.0.2): the commit kernels are
  warmed on every pipeline rank at boot, so the first real commit no longer loads a Triton kernel mid-serve (a hang seen in testing); rank 0 waits only for the token counts before it commits the deferred step (`VLLM_RECOVERSSM_PP_SPLIT_RECV`, default
  off); on the last pipeline rank the commit can run after the drafter has proposed (the existing
  `VLLM_RECOVERSSM_COMMIT_AFTER_PROPOSE=1`, which the Flash-Next launcher still does not set); duplicate align-postprocess
  launches in one chain are dropped; and the deferred commit inputs are held in a persistent snapshot ring of `pp_size + 1` slots that reuses its
  storage instead of allocating a clone each step. Effect on decode: not yet measured, the A/B has not been done for 3.0.2; the 262K and eight-agent rows are in
  the bench cards.
- **Balanced PP decode batches (experimental, off by default).** `VLLM_MERLIN_PP_DECODE_BALANCE=1` caps decode rows per step
  at `ceil(decoding / pp_size)` so the pipeline's microbatches stay even; only the step a row runs in changes, never its
  tokens. A no-op with one pipeline stage, synchronous scheduling or the V1 runner. Not yet run on 0.33; the one 0.32 run was inconclusive. The A/B has not been done for 3.0.2.
- **Decode-first step budget (experimental, off by default).** `VLLM_MERLIN_DECODE_FIRST=1` reserves the tokens of every
  running decoder that is eligible in a step before a running prefill chunk is sized, so a session admitted after a long cold
  prefill keeps decoding while the prefill runs instead of stalling until it ends. A prefill chunk is never cut to nothing, and
  the knob changes nothing unless a decoder sits behind a prefill chunk. It is not batch-invariant: a smaller chunk moves the
  prefill's chunk boundaries, so temperature-0 outputs are not proven identical across the knob, which is why it is off by
  default. Pair it with `-- --prefill-schedule-interval N` for a decode-only duty cycle. No figures yet; the measurement has not been done
  for 3.0.2. Reported by Sudzpel (halt95/merlin-engine#6).
- **Ornith 1.5 treats its six FIM tokens as control tokens in chat content.** `<|fim_prefix|>`, `<|fim_middle|>`,
  `<|fim_suffix|>`, `<|fim_pad|>`, `<|repo_name|>` and `<|file_sep|>` are real token ids when typed into a message. The
  Ornith launcher now sets `VLLM_MERLIN_EXTRA_CONTROL_TOKENS` to those six, so the sanitizer encodes them as plain text.
  The knob is empty by default and acts only on the sanitize path. See the known limits for the other lanes.
- **Forged control tokens are encoded as text.** Chat messages, tools and documents that spell a control token
  (for example `<|im_end|>`) are encoded as plain text on the Hugging Face renderer; only the chat template can emit real
  control tokens. It is on by default (`--sanitize-control-tokens`, switched off with `--no-sanitize-control-tokens`), and
  a request cannot turn it off. `--reject-mm-with-string-template` (off by default) refuses multimodal input with a
  string-format template, whose inserted placeholders the sanitizer would also encode. Needs Transformers 5.20, now in the
  lock. A leading think block in replayed assistant content stays reasoning.
- **`tool_choice="none"` keeps the text the model wrote.** On the Qwen-family parsers (`qwen3`, `qwen3_coder`,
  `nemotron_v3` and the parsers built on them) tool-call markup that the model wrote comes back in the field it was written
  in, stream and non-stream, chat and Responses, where it was dropped before, and markup inside reasoning no longer ends
  reasoning early. Other parsers are unchanged. Details and limits: [engine changelog](merlin/ENGINE-CHANGELOG.md#merlin-302).
  Checked on the reference host against the release build (2026-11-04): passed on Flash-Next (chat and Responses, stream and
  non-stream), Qwen3.8-27B and Ornith 1.5. On Nemotron 3.5 Super VL the echo checks passed; its two reasoning checks used up
  the check's token budget inside the reasoning, so they show nothing either way.
- **A Ready line, a bug-report file and client settings.** `merlin serve` prints `Ready: http://HOST:PORT/v1 · model ·
  context · key` (and the Anthropic-compatible endpoint) once `/health` answers, with a `starting...` line every minute before it;
  `MERLIN_READY_BANNER=0` turns it off. `merlin doctor --report [FILE] [--log FILE] [--lines N]` writes preflight, the serve
  plan and the server log's tail to one 0600 file with keys, tokens and auth headers redacted. [Connect a client](clients.md)
  gives settings for OpenAI-compatible and Anthropic-compatible clients such as opencode, Cline/Roo, Aider and Continue.
- **`merlin serve --instances N` stays in the foreground.** It no longer returns while the servers run, with or without
  `--wait-ready`: Ctrl+C, SIGTERM or a closed terminal stops them all, and if one exits the others stop. A script that used
  `--wait-ready S && next` must start it in the background and poll `/health`.
- **A compile-cache fault guard, on by default.** After a sticky CUDA fault (illegal memory access, device-side assert,
  uncorrectable ECC and the like; not an out-of-memory or an NCCL timeout) the process writes a `merlin-fault.json` marker
  into each compile-cache directory it used and stops saving artifacts. The next start moves a marked directory to
  `.merlin-quarantine/` beside the compile cache (it keeps the newest two) and recompiles. The ahead-of-time compiled model
  file also gets a SHA-256 sidecar (`model.sha256`); a mismatch counts as a load failure and recompiles. Numerics are
  untouched, because a quarantined cache recompiles the same graphs. `VLLM_MERLIN_COMPILE_CACHE_GUARD=0` turns all of it
  off. After any sticky fault the next start pays one full recompile, whether or not the cache was the cause; loading now
  hashes each compiled artifact.
- **QSA indexer scratch reserve (opt-in).** `VLLM_MERLIN_QSA_SCRATCH` takes `off` (the default, as in 3.0.1), `reserve` (one
  transient allocation during the profile run, which lets the KV sizing see the indexer's logits buffer, as upstream
  vllm#60490 does) or `workspace` (the logits are reserved in the shared model-runner workspace with the MoE and
  GDN-checkpoint scratch instead of a separate block). All three run the same kernels on the same shapes, so only KV
  capacity moves. KV capacity figures for `reserve` and `workspace` have not been measured for 3.0.2.
- **Smaller changes, each off by default unless stated.** `VLLM_FLA_PACKED_DECODE_VERIFY_PARITY=1` makes the packed GDN
  decode kernel normalise like the speculative verify kernel (vllm#65338); it applies only where
  `VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE=1` already runs the packed kernel, which no RecoverSSM lane does, and with the
  knob off the kernel is the 3.0.1 one. `VLLM_SPEC_STABLE_RESIDUAL=1` computes the rejection sampler's residual in a form that
  cannot underflow to all `-inf` when the draft puts almost no mass outside the target's support; it changes numerics (token identity with it on is not
  claimed) and has not been timed, so it stays off by default. `MERLIN_MARLIN_DENSE_CFG` takes exact-shape tile rules (for example
  `sm=86,sms=82,m=1,n=8192,k=2560,g=64x128,ack=bitwise`, rules separated by `;`) for the BF16 channelwise-INT8 GDN
  projections; no tuning entry ships and unset means off. `MERLIN_KV_HOST_TIER__MAX_PENDING_STORE_BLOCKS` (default 0, off)
  caps the source references that host write-back holds, and `MERLIN_KV_HOST_TIER__FALLBACK_BOUNDARY=1` keeps a published
  successor's predecessor as its fallback restore point (default off). No default changes.
- **Qwen3.5-class models expose auxiliary hidden states over pipeline parallelism** (vllm#61180), which drafters that tap
  hidden states (such as EAGLE3) need on a PP layout; with one stage, or with no taps requested, the same operations run as before.
- **Speculative verification fast paths (vllm#65489, vllm#65100).** A batch whose rows are all greedy skips the random
  draws, and a vocabulary block whose target logits are all masked is skipped. Both are bit-identical by construction
  and were not tested separately.

### Measured on Merlin 3.0.2

4x RTX 3090 without NVLink, P2P on, Zagitta's gpu-unlock, 220 W, 2026-11-04 bench cards (runs measured
2026-11-04 UTC; Swift1.5 and Swift1.5-Huihui 2026-11-05). Decode: one stream, thinking on (reasoning effort medium where the checkpoint has one; the
Nemotron 3.5 Super VL and Ornith 1.5 templates have no effort levels), median of 3, at a 4K / 131K prompt and the full
262,144-token context (261,888 prompt + 128 out). 8 agents: total throughput of eight simultaneous follow-ups on 16K
conversations; with thinking on a 512-token follow-up may end inside the reasoning, and reasoning tokens count. GSM8K-200 is
correct answers out of 200. The GSM8K-200 figures are from the 3.0.1 run on vLLM 0.32 (the 3.0.1 card's protocol) and were not
re-run for 3.0.2: thinking on at effort medium on four cards, thinking off from the 3.0.0 run, as the weights are unchanged.
On the full GSM8K test set (1,319 problems, 5-shot, thinking off, temperature 0, 512 tokens) Flash-Next on 3.0.2 scored
1,281 (97.1 %), against 1,276 and 1,275 (96.7 %) for 3.0.1 in two runs of the same session.

| Checkpoint | GPUs | KV pool, tokens | Decode, tok/s, 4K / 131K / 262K | 262K prefill | 8 agents, tok/s, thinking on / off | GSM8K-200, thinking on / off |
|---|---|---|---|---|---|---|
| Flash-Next | 4x | 924,993 | 172 / 189 / 207 | 6,782 tok/s | 654 / 664 (3.0.1 run) | 197 / 199 (both 3.0.1 run) |
| Swift1.5 | 4x | 924,993 | 181 / 187 / 186 | 6,762 tok/s | 673 / 650 (both 3.0.1 run) | 199 / 197 (both 3.0.1 run) |
| Swift1.5-Huihui | 4x | 924,993 | 180 / 189 / 193 | 6,830 tok/s | 703 / 681 (both 3.0.1 run) | 198 / 198 (both 3.0.1 run) |
| Qwen3.8-27B | 1x | 271,506 | 131 / 100 / 77 | 586 tok/s | n/a (two at most) | - / 195 (3.0.1 run) |
| Qwen3.8-27B | 2x | 826,689 | 211 / 184 / 146 | 1,112 tok/s | 682 / 615 (3.0.1 run) | - / 194 (3.0.1 run) |
| Qwen3.8-27B | 4x | 2,012,105 | 258 / 255 / 250 | 2,111 tok/s | 940 / 854 (3.0.1 run) | 198 / 197 (both 3.0.1 run) |
| ThinkingCap (3.0.1 run) | 2x | 826,689 | 203 / 162 / 160 | 1,114 tok/s | 697 / 633 | - / 193 |
| ThinkingCap (3.0.1 run) | 4x | 2,012,105 | 244 / 230 / 238 | 2,099 tok/s | 952 / 832 | 198 / 195 |
| Nemotron 3.5 Super VL | 4x | 1,856,367 (the shipped 1M context) | 101 / 101 / 106 | 4,217 tok/s | 432 / 432 (both 3.0.1 run) | 197 / 195 (both 3.0.1 run) |
| Ornith 1.5 | 4x | 894,618 | 145 / 122 / 111 | 2,288 tok/s | 528 / 486 (both 3.0.1 run) | 196 / 190 (both 3.0.1 run) |

Against 3.0.1, on the checkpoints that have a 3.0.2 card: decode and prefill are within run-to-run spread on every lane
except Nemotron 3.5 Super VL, whose 262K decode is +14 % (thinking off +5 %) and whose cold prefill is +8 to +9 % at every
depth. Card-to-card decode figures move with MTP acceptance, which varies by several percent between sends and between runs (Flash-Next's three
262K sends ran 184 to 213 tok/s, the four-card 27B's 232 to 251), so differences smaller than that are not resolved by
these runs and are neither gains nor losses; read decode by the step time instead: on Flash-Next it is 1.2-1.9 % shorter
than on 3.0.1 in every row. Flash-Next cold prefill is level with 3.0.1 in a same-session A/B (+0.5 % at 10K, -0.2 % at 100K). Rows 2 % or
more below the 3.0.1 card: Flash-Next 4K decode -3 % (-4 % with thinking off); Qwen3.8-27B thinking-off 4K decode -2 %,
-3 % and -4 % on 4, 2 and 1 GPUs and thinking-off 262K -2 % on 2 GPUs; Ornith 1.5 thinking-off 4K and 262K -3 %; Swift1.5 thinking-off 259K -2 % and cold prefill -2 % at 100K; Swift1.5-Huihui 262K -2 % and thinking-off 262K -10 %
(the 3.0.1 figure lies inside the spread of the three 3.0.2 sends, 144 to 164 tok/s; the step time is level and MTP
acceptance lower); Super VL
thinking-off 131K -4 % (one valid send); 8 agents with thinking on -3 % on Flash-Next and on two-card 27B (the 27B totals
come from overlap windows under 4 s; read them loosely). ThinkingCap was not re-run. Each checkpoint's card is under `models/<checkpoint>/benchmarks/2026-11-04/` (Swift1.5's and Swift1.5-Huihui's under `2026-11-05/`),
for example [Flash-Next](../models/flash-next/benchmarks/2026-11-04/BENCH-CARD.md) and
[Qwen3.8-27B](../models/qwen38-27b/benchmarks/2026-11-04/BENCH-CARD.md). Figures marked (3.0.1 run) are carried over
from the 3.0.1 cards for rows that were not re-run: every ThinkingCap row (that checkpoint was not benchmarked for
3.0.2); every GSM8K-200 figure; every thinking-off 8-agent figure; the 8-agent figures for Swift1.5, Swift1.5-Huihui, Nemotron 3.5 Super VL and Ornith 1.5.

### Before you upgrade

- **The engine wheel changes.** 3.0.2 is built on 0.33.0: the wheel file is
  `vllm-0.33.0+merlin.665f43d2cf-cp313-cp313-linux_x86_64.whl`, and after the install `vllm --version` prints `0.33.0`; the
  checkout's commit and tree are recorded in `.build-source`. Use the
  3.0.2 wheel with the 3.0.2 checkout; a 3.0.1 wheel does not run it. `./setup.sh` and `release/install-env.sh` build a new
  venv from `release/requirements-0.33.lock` (torch 2.14.0, Triton 3.8.0); they do not upgrade an old venv in place.
- **The first start compiles again.** torch 2.14 changes the compile-cache keys and Triton 3.8 the autotune cache, so every
  persisted cache is cold: expect the first start of each model to compile its kernels again. `CACHE_ROOT` moves the cache on every Merlin launcher;
  `VLLM_CACHE_ROOT` moves it only for the Qwen3.8-27B launcher (or for a direct engine start).
- **`--mamba-cache-mode` has no effect on 0.33.** vLLM deprecated it: Mamba prefix caching now follows
  `--enable-prefix-caching`. The Merlin launchers still pass `--mamba-cache-mode align`, so the server logs one deprecation
  warning at start; it is harmless, and the launchers will stop passing it in a later release. If you pass the flag
  yourself, drop it.
- **A checkpoint that boots with missing quantization scales is refused** (wNa16 and ModelOpt FP8): the server stops
  at load instead of serving with unset scales. Re-download or re-quantize the checkpoint.
- **`SCALES` provenance is enforced.** A Flash-Next-family directory that has the right `config.json` but neither the
  release scales file nor the verified marker is refused where it used to be warned. `config.json` is identical across
  Flash-Next, Swift1.5 and Swift1.5-Huihui, so it does not identify the weights. Pass `SCALES=<the recipe's scales file>`
  (the refusal prints it) or run `merlin pull` / `./setup.sh` for the directory.
- **The P2P mode alone decides the NCCL and IPC environment.** The launcher now clears an inherited
  `NCCL_P2P_DISABLE`, `NCCL_P2P_LEVEL`, `NCCL_PROTO`, `NCCL_ALGO` and `VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC` on every
  multi-card profile before it sets the row's own values, and prints a note for each value it replaced. A shell that exported
  one of them on purpose no longer reaches the server. Choose the mode with `P2P=on|off|ipc` or a `--profile`; the row's
  interconnect settings then set these variables.
- **`VLLM_SERVER_DEV_MODE` is stripped.** It is dropped from the inherited environment on all four launcher families. To
  keep the development routes, set `MERLIN_ALLOW_SERVER_DEV_MODE=1`; the launcher then warns that those routes have no
  API-key protection.
- **The engine's reduced-draft-vocabulary path is removed.** `VLLM_DRAFT_VOCAB_FILE` (dormant since 3.0.1) and `VLLM_DRAFT_VOCAB_SHA256` no longer do
  anything: the engine ignores them with vLLM's "Unknown environment variable" warning, and the launcher still strips both.
- **Ornith's `serve.sh` probes peer access.** It runs `launch.py --profile ornith-quad-3090 --p2p-probe`: `P2P=on` or
  `P2P=off` picks the row without probing; otherwise the cards decide (`ornith-quad-3090` when every card pair has peer
  access, `ornith-quad-3090-nop2p` otherwise). If peer access cannot be read, the launch stops with "set P2P=on or
  P2P=off" instead of guessing.
- **Tested status follows the P2P mode.** Registry rows whose evidence is P2P-off only (the three CMP 170HX rows
  and the two RTX 4090 rows) report "untested" with a `WARN status-p2p` line when the resolved P2P mode is on.
- **The KV RAM cache "untested" note starts at 48g.** 48 GiB is the largest size with a recorded run; `--kv-ram-cache`
  above it prints the untested NOTE where the line used to be 160 GiB. Nothing is refused.
- **`--tokens-only`, `--skip-tokenizer-init` and their abbreviations are refused**, so a launch cannot bypass the tokenizer.
- **`n` is bounded by the server's `max_num_seqs`.** A chat or completion request with `n` above
  `min(VLLM_MAX_N_SEQUENCES, max_num_seqs)` is answered with HTTP 400 (`param: "n"`) before it is rendered (vllm#65355).
- **The PCIe IPC all-reduce checks peer access across the whole tensor-parallel group.** If any rank pair lacks it, every
  rank takes the same fallback and logs one warning, which covers hosts where only some links have peer access.
- **The host KV arena is no longer zero-filled at allocation,** so allocating it no longer touches every page.
  Nothing reads rows that were never stored.
- **The top-k softmax kernel sets its programmatic-launch attribute only on sm_90 and newer** (vllm#65059), so sm_80, sm_86
  and sm_89 CUDA graphs no longer carry the extra edge. Launch attribute only; outputs are identical.
- **A start after a CUDA fault recompiles.** With the compile-cache guard on (the default), a cache directory marked by a
  fault is moved to `.merlin-quarantine/` beside the compile cache and rebuilt, which costs one full recompile.
  Set `VLLM_MERLIN_COMPILE_CACHE_GUARD=0` to keep the 3.0.1 behaviour.
- **Prefix-cache hashes and hit lengths are the 3.0.1 ones.** Upstream's lookahead block hashes are pinned off (see "Held
  back from 0.33"), so a prefix cached by 3.0.1 hashes the same way.

### Fixes

- **The container builds without an NVIDIA driver.** A source build in `docker build` failed at CMake configure because
  `CUDA::cuda_driver` could not be resolved; the image now links the CUDA toolkit's own `libcuda` stub (link time only),
  as the wheel build does, and can install the release wheel instead of compiling (`MERLIN_WHEEL_URL`). Found and fixed by
  brandon-nuttall (halt95/merlin-engine#10), who also community-qualified the 3.0.1 image on 8 x RTX 3090 at **P2P=off**
  (octo TP4xPP2); the 3.0.2 image is not yet requalified.
- **A TP group of mixed card models (an RTX 3090 with an RTX 3090 Ti) boots more than once with the PCIe-IPC all-reduce.**
  FlashInfer's tune cache recorded rank 0's card name, so on the second boot the other rank rejected the file, the ranks
  disagreed on the tuned table and the workspace check failed. The ranks now take one load decision together, a mixed
  group's cache is named after the exact card set it was tuned on, and an unusable cache is set aside and re-tuned once.
  Homogeneous groups keep their cache path and reload as before. Reported by
  [brandon-nuttall](https://github.com/brandon-nuttall).
- **`P2P=ipc`:** a third interconnect mode on the Flash-Next rows that run the PCIe-IPC all-reduce: NCCL without its P2P
  transport (shared memory between ranks) plus the PCIe-IPC all-reduce, for hosts where NCCL's P2P bootstrap hangs but
  CUDA peer access works. Opt-in (never a default), refused on rows without it and by `merlin doctor --p2p ipc` without
  peer access or with BAR1 under 1 GiB; untested, so a tested row reports untested under it. Thanks to
  [brandon-nuttall](https://github.com/brandon-nuttall).
- **Transient generation failures return HTTP 503 instead of 500:** a multimodal cache miss, and, on setups with
  those connectors, a KV load failure under `kv_load_failure_policy=fail` or an encoder input the engine can no longer
  obtain. Shipped lanes recompute on a KV load failure, so that case stays out of reach there. Grammar rejection stays 500.
  Merged with upstream's paused-generation 503 (vllm#51488).
- **Full-mode KV reuse events no longer raise `IndexError` on a partial Mamba hit, and no longer report the partial tail
  under the hash of a full block.**
- **One request can no longer end a lane through the sampler limits.** The frontend already answered HTTP 400 for
  oversize `allowed_token_ids`, `logit_bias`, `bad_words` and `stop_token_ids`; EngineCore now applies the same limits at
  admission, so a request that arrives by another route (offline `LLM`, another frontend) gets a per-request error
  (`FinishReason.ERROR`) instead of an error inside the worker step that stops the engine. Limits are unchanged: more than
  1,024 `allowed_token_ids` or `logit_bias` entries, `bad_words` over 128 sequences or 1,024 tokens, `stop_token_ids` over 128
  with `min_tokens`.
- **Top-k logprob counts above the vocabulary size are rejected at admission** (`logprobs` and `prompt_logprobs`;
  vllm#64291; `prompt_logprob_token_ids` widths keep the configured limit), and **pending UTF-8 detokenization work is
  bounded** (vllm#63667): a malformed byte run is decoded over a bounded window instead of being re-decoded, ever longer,
  on the API event loop.
- **The HistoryIndex is slimmed to one shared dictionary,** and it keeps prompt
  provenance for the shared-cap accounting. The long plateau soak was not run for 3.0.2.
- **Security fixes (ports of vllm#52120 and vllm#64503):** the path-masking pattern in error messages no longer
  rescans a slash run from every slash (a slow-match input; it is one linear pass now), and the structured-decisions
  endpoint validates `chat_template_kwargs` against the server's `--trust-request-chat-template` setting.
- **`qwen3_coder` tool parser:** literal tags inside parameter values and in reasoning are kept as text,
  truncated arguments of a cut-off call are handled, a bare call ends at its first `</function>`, and
  `validate_tool_names=True` removes calls to names the request did not offer, with consecutive stream indexes after a
  rejected call. Held token ids no longer copy the whole prefix on every delta. The same configuration reaches
  `nemotron_v3`, MiMo, Step3p5 and MiniCPM-V.
- **KV code re-homed by hand.** Upstream moved most of `kv_cache_utils.py` into `kv_cache_planning.py`; Merlin's prefix-match
  unit, retention, deferred block free and host-tier sizing were re-applied on the new layout, and the draft-block drop set is
  read live again as in 3.0.1. Hit rate against the trace ceiling: Flash-Next reused 87.8 % of prompt tokens against
  a ceiling of 89.3 % (98.3 % of the ceiling); Qwen3.8-27B on 4 GPUs 90.5 % against 92.1 % (98.2 %); release
  configuration, 2026-11-04.
- **A Super VL host with P2P on keeps vLLM's custom all-reduce and sets `NCCL_PROTO=Simple` instead of `LL`.**
  On 4× RTX 3090, measured on torch 2.13: one-stream decode unchanged, a new 20K prefill about 6 % faster and decode during
  it about 10 % faster, with identical temperature-0 output. P2P-off rows and the other families are unchanged. Not re-measured
  on torch 2.14 (NCCL 2.30.7); that re-measure has not been done for 3.0.2.
- **Literal markers stay text in non-streaming responses too (vllm#64326).** The Qwen-family parsers now parse a
  non-streaming reply with its token ids, as streaming already did, so a tool-call or reasoning marker spelled out in
  ordinary tokens stays text instead of being read as markup. Parsing a long streamed tool call no longer re-reads the
  whole argument on every delta.
- **Int64 address arithmetic in the long-context kernels (vllm#63761, vllm#64820):** in-range addresses are identical, so
  outputs are bit-identical. Also from upstream: the KV
  block-copy dedup (vllm#55112), `needs_kv_cache_zeroing` combined across pipeline stages (vllm#64472), the DFlash `fc`
  input cast (vllm#63436), the Nemotron vision-hidden-size fallback (vllm#64506) and prompt-embed sentinel ids (vllm#64367).
- **Setup and launcher text:** the unverified-weights warning names `./setup.sh` and `merlin pull`; the Flash-Next 4090
  profile note gives the 2,048-token chunk's prefill gain as 8–20 %; the 27B profile label names the 3.0.1 recipe and its
  pinned checkpoint; an interrupted `hf download` leaves no partial files behind before the next download.
- **Upstream 0.33 fixes now in the engine** (taken with the merge): cuBLAS handles are created before profiling, the
  speculative warm-up with its pipeline-parallel guard, the pipeline-parallel draft broadcast (vllm#64458, replacing our
  port), EAGLE/MTP drafts skipped on steps where no request can use them (vllm#62784), sliding-window block nulling in the
  gather, and the connector-ordering fixes (vllm#62041).

### Engine

- Base: vLLM 0.33.0 (upstream commit `4ba787ec99`).
- The new engine switches include `--sanitize-control-tokens` / `--reject-mm-with-string-template`,
  `VLLM_MERLIN_EXTRA_CONTROL_TOKENS`, `VLLM_MERLIN_DECODE_FIRST`, `VLLM_MERLIN_PP_DECODE_BALANCE`,
  `VLLM_RECOVERSSM_PP_SPLIT_RECV`, `VLLM_MERLIN_COMPILE_CACHE_GUARD` (on by default), `VLLM_MERLIN_QSA_SCRATCH`,
  `VLLM_FLA_PACKED_DECODE_VERIFY_PARITY`, `VLLM_SPEC_STABLE_RESIDUAL`, `MERLIN_MARLIN_DENSE_CFG`,
  `MERLIN_KV_HOST_TIER__MAX_PENDING_STORE_BLOCKS`, `MERLIN_KV_HOST_TIER__FALLBACK_BOUNDARY` and `VLLM_E2_PLE_STORE` /
  `VLLM_E2_PLE_WARM` / `VLLM_E2_PLE_DISK_THREADS`; see "New flags and environment variables" for all of them.
- The NVFP4 PLE embedding method upstream added (vllm#56273) is accepted by the PLE format guard with `uint8` tables; every
  other method and dtype stays refused.
- Qwen3.8 Flash-Next family: Merlin's own sm_86 FP8 QSA reader stays; upstream's Ampere FP8 main KV (vllm#60602) and
  four-bytes-per-register E4M3 decode (vllm#60992) were not adopted. No same-card A/B against upstream was run for 3.0.2.
- The persistent top-k kernel got the float4 tail fix (vllm#62467) in our vendored copy.

### New flags and environment variables

| Name | Where | Default | What it does |
|---|---|---|---|
| `--ple-store ram\|disk` | `merlin serve`, `./setup.sh` | `ram` | **Experimental.** Where the Flash-Next family's PLE table lives. |
| `VLLM_E2_PLE_STORE`, `VLLM_E2_PLE_WARM`, `VLLM_E2_PLE_DISK_THREADS` | engine | `ram`, on, 32 | The engine side of `--ple-store`: the store, the background warm-up read, and the reader threads. |
| `PLE_STORE`, `PLE_WARM` | launcher | `ram`, `1` | The launcher's controls: `PLE_STORE` is `--ple-store`, and `PLE_WARM=0` turns the warm-up read off. The launcher sets the engine variables above from them, so setting `VLLM_E2_PLE_WARM` yourself has no effect through `merlin serve`. |
| `--context SIZE` | `merlin serve`, `./setup.sh` | the checkpoint's native context | **Experimental.** Serves SIZE tokens (`512k` = 524,288); past 262K it adds YaRN from the checkpoint's config. |
| `--instances N` (foreground) | `merlin serve` | one server | Now runs in the foreground and stops all N servers together. |
| `merlin doctor --report [FILE] [--log FILE] [--lines N]` | `merlin doctor` | off | Writes one redacted bug-report file (mode 0600). |
| `MERLIN_READY_BANNER=0` | launcher | on | Turns the Ready line off. |
| `--sanitize-control-tokens` / `--no-sanitize-control-tokens` | vLLM | on | Encodes control tokens in chat content as text. |
| `--reject-mm-with-string-template` | vLLM | off | Refuses multimodal input with a string-format chat template while sanitizing. |
| `VLLM_MERLIN_EXTRA_CONTROL_TOKENS` | engine | empty | Comma-separated added tokens the sanitizer also treats as control tokens. The Ornith launcher sets the six FIM tokens. |
| `VLLM_MERLIN_DECODE_FIRST=1` | engine | off | **Experimental.** Reserves running decoders' tokens before a prefill chunk is sized. Not proven token-identical. |
| `VLLM_MERLIN_PP_DECODE_BALANCE=1` | engine | off | **Experimental.** Caps decode rows per step to even out pipeline microbatches. |
| `VLLM_RECOVERSSM_PP_SPLIT_RECV=1` | engine | off | Rank 0 commits the deferred RecoverSSM step after the counts arrive, before the drafts. |
| `MERLIN_ALLOW_SERVER_DEV_MODE=1` | launcher | off | Lets `VLLM_SERVER_DEV_MODE` through (the development routes have no API-key protection). |
| `--p2p-probe` | `tools/merlin/launch.py` | off | Chooses the P2P or no-P2P twin of the Ornith profile from the cards' peer access; used by `models/ornith1.5/serve.sh`. |
| `"disable_lookahead_block_hashes": false` | `--speculative-config` | `true` in Merlin (upstream `false`) | Turns upstream's lookahead block hashes on. Not a supported lane in 3.0.2. |
| `VLLM_MERLIN_COMPILE_CACHE_GUARD` | engine | **on** (`0` turns it off) | Quarantines a compile-cache directory marked by a sticky CUDA fault into `.merlin-quarantine/` beside the cache and checks each compiled artifact's SHA-256. |
| `VLLM_MERLIN_QSA_SCRATCH=off\|reserve\|workspace` | engine | `off` | Reserves the QSA indexer's logits scratch during the profile run, as a transient allocation or in the shared workspace. |
| `VLLM_FLA_PACKED_DECODE_VERIFY_PARITY=1` | engine | off | Makes the packed GDN decode kernel normalise like the speculative verify kernel. |
| `VLLM_SPEC_STABLE_RESIDUAL=1` | engine | off | Underflow-safe residual in the rejection sampler. Changes numerics. |
| `MERLIN_MARLIN_DENSE_CFG` | engine | unset (off) | Exact-shape tile rules for the dense W8A16 Marlin GDN projections. No entry ships. |
| `MERLIN_KV_HOST_TIER__MAX_PENDING_STORE_BLOCKS` | engine | `0` (off) | Caps the source references host write-back holds. |
| `MERLIN_KV_HOST_TIER__FALLBACK_BOUNDARY=1` | engine | off | Keeps a published successor's predecessor as its fallback restore point. |
| `P2P=ipc` (and `merlin doctor --p2p ipc`) | launcher | off (opt-in, never a default) | A third interconnect mode on the Flash-Next rows that run the PCIe-IPC all-reduce: NCCL without its P2P transport plus the PCIe-IPC all-reduce. Untested. |
| `match.status_p2p_modes` | profile registry | absent | The P2P modes a tested or testing row's evidence covers. |
| `MERLIN_WHEEL_URL` | `container/` build argument | empty | Installs the release wheel into the image instead of compiling; the installer checks it against `release/merlin-wheel.sha256`. Thanks to brandon-nuttall (halt95/merlin-engine#10). |
| `MERLIN_PROFILE` | container entrypoint | empty | Flash-Next family: starts `merlin serve --profile <id>` instead of the recipe's default launcher. Thanks to brandon-nuttall (halt95/merlin-engine#10). |
| `MERLIN_LIBCUDA` | `release/install-env.sh` | empty | Path of the CUDA toolkit's `stubs/libcuda.so`, linked under the toolkit's `lib/stubs` for a source build with no driver (the container build sets it). |

### Held back from 0.33

Upstream 0.33 changes that Merlin does not take in 3.0.2, with the reason:

- **Lookahead block hashes (vllm#61589), on by default upstream, pinned off.** They change the prefix-hash format and the hit
  length on MTP lanes. Merlin sets `disable_lookahead_block_hashes=True`, so hashes and hit lengths stay as in 3.0.1. Whether
  it is taken later depends on a KV host-tier hit-rate comparison against the trace ceiling, which has not been done for 3.0.2.
- **DFlash and DSpark keep dropping the trailing block (vllm#54163 not taken).** The 3.0.1 rule stands. The same comparison decides it; it has not been done for 3.0.2.
- **Small-batch split top-p (vllm#61996) is reverted as a unit,** because it may change the reduction order. It stays
  reverted in 3.0.2; taking it needs a token-identity run, which has not been done for 3.0.2.
- **The step-keyed RNG fix for block verification (vllm#61161) is not the default.** Merlin keeps its 3.0.1 keying bit for
  bit, with `VLLM_SPEC_STEP_KEYED_RNG` as the opt-in. On the default path block verification keeps the draw reuse that
  vllm#61161 fixes; DFlash2 and the watermark skip path are restored to 3.0.1 keying.
- **Upstream's Ampere FP8 QSA paths (vllm#60602, vllm#60992)** are not adopted; Merlin's own reader stays.
- **`--mamba-cache-mode`** is deprecated upstream and has no effect (see "Before you upgrade").
- **Not in 0.33, still carried by Merlin:** ports of upstream changes that 0.33 does not contain, such as vllm#61557,
  vllm#62318, vllm#52120, vllm#64503, vllm#64673 and vllm#64675. They move out when upstream ships them.

### Known limits

- **PLE on NVMe and `--context`** figures above were measured on the 0.32 tree; their 0.33 re-runs have not been done for 3.0.2.
- **`tool_choice="auto"` can still return quoted tool-call markup as a real tool call (open).** An agent harness would
  execute it. For an explain-only request, send `tool_choice="none"`. `response_format` with thinking on is not covered: a
  quoted `<tool_call>` in reasoning can start the grammar early, which leaves the JSON in `reasoning`.
- **Only Ornith treats the six FIM tokens as control tokens.** The 2026-11-03 check found the same six tokens forgeable on
  the 27B, Flash-Next, ThinkingCap, Swift1.5 and Swift1.5-Huihui; opting them in is a one-line launcher change per family
  and is not part of 3.0.2.
- **Quoted control markers can re-encode differently on replay (transformers#49891).** If a user message spells a marker next
  to punctuation or code, a replayed conversation can differ from the first pass (4 of 6 quoted-marker cases on Ornith in
  the CPU check). The CPU check stays as a monitor; nothing in the engine works around it.
- **The compile-cache guard cannot see a fault that kills the process before Python does** (for example a native abort with no
  Python exception on the faulting rank). A Triton cache you set yourself with `TRITON_CACHE_DIR` is marked and quarantined
  like the compile cache.
- **KV RAM cache limits from 3.0.1 still apply** (a 262K session evicted during heavy prefill may be recomputed instead of
  restored; host memory grows in steps while the request history fills). Experimental labels are unchanged.
- **The FlashInfer 0.7.2.post1 patches have no figure older than this release.** The 0.7.0 tile numbers do not carry over.
- **The Rust frontend needs `--trust-request-chat-template`** (upstream vllm#62673) and refuses unsupported flags; Merlin
  launches the Python frontend.

### Docs

- Every 0.32 install line, version badge, lock reference and FlashInfer 0.7.0.post1 mention in the README, setup guide,
  engine README, container README and the model cards now names 0.33.0, `release/requirements-0.33.lock` and FlashInfer
  0.7.2.post1. Statements about earlier measurements keep their "0.32" or "3.0.1" labels.
- The model cards gain a 3.0.2 bench-card section beside the 3.0.1 one; the 3.0.1 cards are unchanged.

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

- **The engine wheel is unchanged from 3.0.0** (no compiled-code changes in 3.0.1; the native sources changed only in comments and message text), so the wheel file is still
  `vllm-0.32.0+merlin.7002f56537-…`, and after the install `vllm --version` prints `0.32.0`; the 3.0.1 Python comes from the checkout.

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
- The 3.0.0 bench cards gained 262K rows on 2026-10-22, from the same runs.

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
- **One chat template.** Every Flash-Next-family and Qwen3.8-27B-family recipe serves froggeric's fixed Qwen3.8 template (v22.5), which accepts system
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
