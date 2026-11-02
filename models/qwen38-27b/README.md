# Qwen3.8-27B and ThinkingCap on 2× or 4× RTX 3090

Every 27B launcher serves Froggeric's fixed Qwen3.8 template,
[`templates/froggeric-qwen-v22.5.jinja`](templates/froggeric-qwen-v22.5.jinja), as do ThinkingCap and the
Flash-Next recipes. It accepts system messages after the first turn, so coding agents' per-turn notes stay in
place on `/v1/messages` and the prefix cache keeps hitting. `CHAT_TEMPLATE=<path>` serves another template;
`CHAT_TEMPLATE=` (empty) serves the checkpoint's own.

**Recipe 3.0.0 targets the Merlin engine on vLLM 0.32.0**, with full 262K context, vision, tools,
FP8 E4M3 KV and MTP K=4. ThinkingCap is the [checkpoint profile](../thinkingcap/README.md), served at `medium`
reasoning effort by default; Qwen3.8-27B defaults to `medium`.

[![GPU](https://img.shields.io/badge/GPU-2x_or_4x_RTX_3090-76B900?logo=nvidia&logoColor=white)](docs/reference.md#hardware)
[![Context](https://img.shields.io/badge/context-262K_per_request-ffb000)](docs/reference.md#the-kv-budget)
[![KV pool](https://img.shields.io/badge/KV_pool-826%2C689_%2F_2%2C012%2C105_tokens-0969da)](docs/reference.md#the-kv-budget)
[![Engine](https://img.shields.io/badge/vLLM-0.32.0-6f42c1)](docs/reference.md#stack)

The two W4A16 checkpoints are
[halt95/Qwen3.8-27B-W4A16-Merlin](https://huggingface.co/halt95/Qwen3.8-27B-W4A16-Merlin) (Apache-2.0) and
[halt95/ThinkingCap-Qwen3.8-27B-W4A16-Merlin](https://huggingface.co/halt95/ThinkingCap-Qwen3.8-27B-W4A16-Merlin)
(BottleCap AI's reasoning fine-tune; [weights licence](#license)). Both use an AWQ int4 body, int4 `lm_head`
and MTP drafter, calibrated FP8 KV and fp16 SSM state. ThinkingCap is an independent quantised derivative,
not published, reviewed or endorsed by BottleCap AI or Qwen. Bare metal, sm_86 (RTX 3090) is the measured setup.

## Models

| Profile | Checkpoint | Default effort | Launchers |
|---|---|---|---|
| Qwen3.8-27B | Qwen3.8-27B-W4A16-Merlin | `medium` | [TP2](serve.sh), [TP4](serve-tp4.sh) |
| ThinkingCap | ThinkingCap-Qwen3.8-27B-W4A16-Merlin | `medium` | [TP2](../thinkingcap/serve.sh), [TP4](../thinkingcap/serve-tp4.sh) |

Both checkpoints share the engine, port 8100, served name `qwen38-27b` and profile settings.
The wrappers select the ThinkingCap checkpoint and forward the other variables; both checkpoints default to reasoning effort `medium`.

## Merlin 3.0.1 bench card (2026-11-02)

Maintainer measurements on 4× RTX 3090 (220 W, PCIe Gen4) with P2P (the aikitoria patched driver) and the Zagitta [gpu-unlock](https://github.com/Zagitta/gpu-unlock) kernel module, one fresh `merlin serve` boot per configuration, 2026-11-01 to 2026-11-02; method and every cell in the [bench card](benchmarks/2026-11-02/BENCH-CARD.md). Decode is one stream with thinking on at reasoning effort medium, the server default (tok/s after the first streamed token); prefill is one cold request. With thinking on, a 512-token agent follow-up may end inside the reasoning, and reasoning tokens count. GSM8K at medium ran on four cards. Thinking-off GSM8K is from the 3.0.0 runs (two-card and one-card: a 2026-10-17 pre-release build). The agent overlap windows on four cards and the two-card thinking-on window are under 5 s: read those totals loosely. Rows marked (3.0.0 run) were not re-run for 3.0.1; they and the rest of the 3.0.0 figures are on the [3.0.0 bench card](benchmarks/2026-10-20/BENCH-CARD.md).

| Measurement | 2× RTX 3090 (TP2) | 4× RTX 3090 (TP4) | 1× RTX 3090 |
|---|---|---|---|
| Decode, 4K / 131K / 262K prompt | 210 / 182 / 143 tok/s | 251 / 243 / 238 tok/s | 125 / 95 / 71 tok/s |
| Cold prefill, 10K / 100K / 262K prompt | 2,301 / 1,672 / 1,115 tok/s | 3,866 / 2,995 / 2,101 tok/s | 1,200 / 871 / 579 tok/s |
| Agents at once (8 × 16K conversations), total, thinking on / off | 705 / 615 tok/s | 954 / 854 tok/s | n/a |
| GSM8K-200 (5-shot), correct of 200, thinking on at medium / thinking off | – / 194 | 198 / 197 | – / 195 |
| Sessions held at once (3.0.0 run) | 3 × 262K and 6 × 131K, every session resident (re-sent prompts ≥ 99.98 % cached), 0 preemptions | 7 × 262K and 14 × 131K, every session resident (re-sent prompts ≥ 99.98 % cached), 0 preemptions | 1 × 262K and 2 × 131K, every session resident, 0 preemptions |
| 8 MP image (3.0.0 run) | one, two at once and four in one prompt (32,842 tokens) correct; lowest free memory 886 MiB on any card | one, two at once and four in one prompt (32,842 tokens) correct; lowest free memory 862 MiB on any card | 8 MP: out of memory on one card. One card takes 4 MP instead: one and two at once correct with the single-card memory settings, lowest free memory 558-563 MiB (2026-10-21) |

The [2026-10-15 bench card](benchmarks/2026-10-15/BENCH-CARD.md) records the 0.32 tile reclaim.
With patched **FlashInfer 0.7.0.post1** and `FLASHINFER_SM8X_FP8_SMALLQ=cta32`, the median decode-step
interval (ITL) at 4K / 32K / 131K is:

| Profile | 4K | 32K | 131K |
|---|---:|---:|---:|
| TP2 | **19.99 ms** | **21.41 ms** | **25.27 ms** |
| TP4 (RecoverSSM-off line) | **15.83 ms** | **16.62 ms** | **18.37 ms** |

The tile restores 0.30 depth parity. TP2 GSM8K-200, tile on, scored 192 / 193 / 191, mean **192.0/200**,
against v1.3.0's **196/200**: −2.0 pp, 95 % Newcombe CI **[−5.89, +1.61] pp**, within CI and not
significantly worse. Tile on and off are not bitwise identical. The tile run used a 0.32 candidate;
it was not a rerun of the final release assembly. The card explains that boundary and the cache-provenance check.

- **KV pools:** TP2 **826,689 tokens** at the shipped K=4 pin. TP4 serves RecoverSSM on with the shipped TP4 geometry (block 1024, chunk 4096, LPT 832, pin 17,923,833,856 B = 2,011 blocks): **2,012,105 tokens** (7.68 × 262K), all correctness checks passed, and the 2026-10-20 release bench card measured decode 252.4 / 235.4 / 224.0 tok/s at 32K / 131K / 259K (thinking on). The TP4 ITL and concurrent-decode figures in this section were measured on the retired RecoverSSM-off line (1,974,072 tokens).
- **FGPC ON** on both profiles: `--enable-mamba-shared-prefix-checkpoint`, inherited by ThinkingCap.
- 27B TP2 serves images up to 8 MP. With the vision-peak option on (`VLLM_MERLIN_VISION_PEAK=1`, the TP2 launcher default), the 50-item TP2 stress sweep's worst minimum free was 806 MiB per card (2026-10-15), above the 524 MiB floor. A separate knob OFF/ON identity check matched tokens and logprobs on 6/6 prompts. Flash-Next (902 MiB) meets the target. 27B TP4 left 690 MiB free on the retired RecoverSSM-off pin; at the shipped chunk 4096 with RecoverSSM on, the 2026-10-20 bench card measured 862 MiB lowest free (one 8 MP image, two at once, and four in one prompt; 2 cards: 886 MiB), above the 524 MiB floor.
  These base-checkpoint measurements do not constitute a new ThinkingCap benchmark.
- **Concurrent decode, RTX 3090, pre-release vLLM 0.32 build, 2026-10-15:** TP4 14×127K (RecoverSSM-off line): **439.5 tok/s aggregate**, **31.79 ms/tok per agent**, 0 preemption.
  TP2 with MTP off: prefix-cache hit 0.9996, TTFT 18.25→0.52 s at 32K. The merged release-candidate engine subsequently matched that pre-release reference's T=0 tokens and logprobs on four configurations.

The 2026-10-12 base and ThinkingCap cards are **historical vLLM 0.30 measurements**. Their single-stream
tok/s, token savings and quality figures are not new 3.0.0 results.

## What you need

Two or four 24 GB RTX 3090s, Linux x86_64, and the shared 0.32 environment described in
[release/README.md](../../release/README.md); it carries its own CUDA 13.0 toolkit for FlashInfer JIT, so no system
toolkit is needed. The tested numbers use GPU peer-to-peer (P2P). Without it, `merlin serve 27b <dir> --tp 2` picks the
no-P2P row, or run `P2P=off bash models/qwen38-27b/serve.sh`; RTX 3090s without P2P are untested on this release (⚫); the four-card
no-P2P row (`27b-tp4-3090-nop2p`) was boot-checked with `--swarm` on 2026-11-01.
Host-resident embeddings require about 2.4 GiB additional pinned host RAM (1.18 GiB per TP2 rank, 0.59 per TP4 rank).

`merlin serve 27b <dir> --kv-ram-cache SIZE` turns on the host-RAM KV cache (evicted prefix KV blocks restored from RAM
instead of recomputed; SIZE GiB per server, pinned, on top of the RAM above). A drain-ahead write-back stores whole
restore units before GPU blocks are reclaimed; host eviction takes idle rows first, then copies that duplicate blocks
running requests hold on the GPU, then the least recently used content, and last copies whose GPU twins are about to
be reclaimed; an evicted unit gives up only the rows no other session shares. There is no retention
interval; tuning knobs are listed in the
[Flash-Next reference](../flash-next/docs/reference.md#kv-ram-cache-tuning-knobs). The tier needs at least one 262K restore
unit per rank: the smallest size is `9g` on one card, `10g` on two and `12g` on four. It does not change the session cap: on the quad card `--swarm` serves up to 32
sessions instead of 24 (experimental), setting this cache to the tested 48g when `--kv-ram-cache` is not given. It runs on the RecoverSSM path only and remains
**experimental** on the 27B in 3.0.1; see [SETUP](../../docs/SETUP.md). Under heavy concurrent prefill a session may
recompute instead of restoring (output identical).

**Multi-session behaviour on TP4 (3.0.1).** While one session prefills a long prompt, the other sessions' decode slows
sharply and a burst of new sessions waits longer for its first token, with the KV RAM cache off or on. Long prefills
are not interleaved fairly with decode yet.
The [reference](docs/reference.md#requirements) describes the measured host and separates the old install requirements.

## Quick start

Use the repository-wide [release install route](../../release/README.md) for this checkout and vLLM **0.32.0**.
The shared installer and FlashInfer 0.7.0.post1 tile tooling are integrated.
The dependency lock is complete: 222 exact pins with artifact SHA-256 hashes, resolved with uv; a dry-run `--require-hashes` install resolves.
The installer applies the tile; the environment knob alone has no effect on unpatched FlashInfer.
The archived v1.3.x engine bundle and separate 27B series describe the historical 0.30 route.

Once the shared environment and tile patch are installed, run from the Merlin-engine repository root:

```bash
HF_HUB_DISABLE_TELEMETRY=1 venv/bin/hf download halt95/Qwen3.8-27B-W4A16-Merlin --local-dir ./ckpt
(cd ckpt && sha256sum -c ../models/qwen38-27b/checkpoint.sha256)
# Or download ThinkingCap (its weights have a separate licence):
HF_HUB_DISABLE_TELEMETRY=1 venv/bin/hf download halt95/ThinkingCap-Qwen3.8-27B-W4A16-Merlin --local-dir ./ckpt-thinkingcap
(cd ckpt-thinkingcap && sha256sum -c ../models/thinkingcap/checkpoint.sha256)

# Explicitly select the launchers' 8 MP default:
MAX_PIXELS=8388608 MODEL=$PWD/ckpt models/qwen38-27b/serve.sh
MAX_PIXELS=8388608 MODEL=$PWD/ckpt models/qwen38-27b/serve-tp4.sh
MAX_PIXELS=8388608 MODEL=$PWD/ckpt-thinkingcap models/thinkingcap/serve.sh
MAX_PIXELS=8388608 MODEL=$PWD/ckpt-thinkingcap models/thinkingcap/serve-tp4.sh
```

Choose one of the four launch commands. The server listens on `127.0.0.1:8100`; `HOST=0.0.0.0` exposes it to
other machines, so set `VLLM_API_KEY` when appropriate. A fresh compile cache takes several minutes.
At K=4 the expected pool is `826,689 tokens` (TP2) or `2,012,105 tokens` (TP4).
Wait for `Application startup complete`, then send a real generation.

### Container

The repository-wide [container recipe](../../container/README.md) is unqualified for this 27B release.
The old container at the archived package's `v1.0.0` tag serves the old engine.

## Talk to it

Any OpenAI-compatible client works. With curl:

```bash
curl http://127.0.0.1:8100/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen38-27b","messages":[{"role":"user","content":"Explain the difference between a stack and a queue."}]}'
```

Or from Python with the `openai` package (set `OPENAI_API_KEY` to your `VLLM_API_KEY`, or to any value if you set none):

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8100/v1")
reply = client.chat.completions.create(model="qwen38-27b", messages=[{"role": "user", "content": "Hello!"}])
print(reply.choices[0].message.content)
```

- Both checkpoints answer as `qwen38-27b` (`SERVED=` changes the name). Thinking is on by default, at `medium` effort for
  Qwen3.8-27B and for ThinkingCap; a request can choose its own effort (`xhigh` included) with
  `"chat_template_kwargs": {"reasoning_effort": "medium"}`.
- A short `max_tokens` can end inside the reasoning with empty `content` and `finish_reason: "length"`; raise it or send
  `"chat_template_kwargs": {"enable_thinking": false}`.
- Tools use the standard `tools` field (`qwen3_coder` parser, auto tool choice); images go in `image_url` content parts,
  up to 64 per prompt, each downscaled to the 8 MP cap (`MAX_PIXELS=8388608`).
- Every response reports `usage.prompt_tokens_details.cached_tokens`, the prompt tokens served from the prefix cache.
- Do not use `/v1/models` as a health check: it keeps answering 200 after the engine has died. A one-token generation
  is the check.

## 3.0.0 launcher migration (vLLM 0.32)

The TP2/TP4 launchers and ThinkingCap wrappers use `--attention-backend FLASHINFER`,
`--mamba-cache-mode align` and **FGPC ON** via `--enable-mamba-shared-prefix-checkpoint`.
The launchers default to **8,388,608 pixels (8 MP)**; the examples select that value explicitly.
The cap is a served default, not an enforced limit: a client that raises it per request
(`mm_processor_kwargs`) goes past the measured headroom and can run the engine out of memory.
Keep client harnesses at or below the cap.
The folded environment controls and the old attention environment variable are removed;
see [0.32 launcher settings](docs/reference.md#vllm-032-launcher-settings).
The shared installer, FlashInfer 0.7.0.post1 tile patch and 8 MP launcher default are integrated;
a fresh install (2026-10-21) and a from-source build (2026-10-22) passed on the reference host.

## What's new in v1.3.0

Historical release notes for vLLM 0.30; the install and performance claims in this section describe that release.

- **Engine.** Stock vLLM 0.30.0 plus five patches is replaced by Merlin engine v2.5.2 plus the 27B series
  `patches/merlin-engine-v2.5.2/0001-0011` ([What is in v1.3.0](docs/reference.md#what-is-in-v130)). The series is
  active only for a dense "Qwen3.5" target on sm_86; every other model takes the stock path. "Qwen3.5" here is the name
  of the architecture class vLLM loads (`Qwen3_5ForCausalLM` / `Qwen3_5ForConditionalGeneration`), which Qwen3.8-27B
  uses; it is not a different model, and the patches apply to this checkpoint.
- **Second checkpoint: ThinkingCap.** BottleCap AI's reasoning fine-tune of Qwen3.8-27B, in the same 4-bit format,
  served on this same engine at `medium` effort by default (`models/thinkingcap/serve.sh`,
  `models/thinkingcap/serve-tp4.sh`, checkpoint list `models/thinkingcap/checkpoint.sha256`). Its weights carry their own
  licence ([License](#license)); details in [ThinkingCap checkpoint profile](docs/reference.md#thinkingcap-checkpoint-profile).
- **FlashInfer 32-row verify tile, on by default** (`FLASHINFER_SM8X_FP8_SMALLQ=cta32`). The MTP verify step at K=4 has
  5 query rows per sequence; the tile runs that attention with 32-row tiles instead of stock FlashInfer's smaller ones,
  and because it acts on attention its effect grows with context depth. The release check measured the shipped profiles
  with the tile on; tile-off decode speed was not re-measured (bench card section 6 has a tile-off boot and its
  GSM8K-200). `release/install-env.sh` patches FlashInfer with
  `release/apply-flashinfer-patch.sh`; each rank prints a proof line when the tile is in use
  ([Check it's working](docs/reference.md#check-its-working)).
- **MTP K=4 by default** (was K=3): the drafter proposes four tokens per step instead of three. The release check measured
  K=4 only; `SPEC_K=3` and `SPEC_K=1` remain accepted ([SPEC_K](docs/reference.md#spec_k)).
- **`--prefix-match-unit 16`.** The hybrid KV block is 848 tokens at K=4 (832 at K≤3) and 960 on dual card with
  RecoverSSM; 64 does not divide 848, 16 divides all three.
- **Dual card runs RecoverSSM** (`--use-replayssm`) with shared prefix checkpoints (`--enable-mamba-shared-prefix-checkpoint` in vLLM 0.32), so three 262K or six
  131K sessions stay fully resident; the pin is 883 whole blocks. Patch 0011 runs the RecoverSSM commit from CUDA graphs
  after the MTP proposal to keep its per-step cost down; the release-check decode figures in
  [numbers](docs/reference.md#v130-numbers) are with RecoverSSM on.
- **Quad card** uses shared prefix checkpoints (`--enable-mamba-shared-prefix-checkpoint` in vLLM 0.32) (no RecoverSSM) and holds seven 262K sessions.
- **Scheduler limits tightened:** dual card `--max-num-seqs 16 --max-num-batched-tokens 2048` (was 32 / 4096), quad card
  `--max-num-seqs 24 --max-num-batched-tokens 2048`.
- **fp16 SSM state, unchanged.**
- **Idle cards drop to P8.** The engine's bounds-guard reader parks with no device work while the server is idle, so
  idle cards can drop to P8 instead of holding P2. On vLLM 0.32 idle parking is built in and has no environment switch. Thanks to
  the community [Flash-Next idle-power report](https://github.com/halt95/merlin-engine/issues/2)
  and [contribution](https://github.com/halt95/qwen38-flash-next-3090s/pull/3), which led to this fix.
- **Cached tokens in every response** (`--enable-prompt-tokens-details`), and a per-image pixel cap, `MAX_PIXELS`
  (v1.3.0 default 4,194,304 = 2048×2048; 3.0.0 release cap is 8,388,608 = 8 MP).
- **Combining marks tokenize as upstream.** transformers is pinned to 5.18.0: with 5.17.0 and earlier the served
  tokenizer split combining-mark scripts (Hindi, Thai, Arabic) differently from upstream; with 5.18.0 it matches upstream
  on a 153-item combining-mark set. The checkpoint's `tokenizer.json` is also replaced by the upstream Qwen3.8-27B file
  (sha256 `0997f410c57a1f4e…`; the file shipped before lacked `\p{M}` in its pre-tokenizer), for tools that read
  `tokenizer.json` directly; `models/qwen38-27b/checkpoint.sha256` lists the upstream file
  ([Checkpoint](docs/reference.md#checkpoint)). No weights changed.
- **Single card: 4 MP images.** `serve-1card.sh` (one RTX 3090, TP1) keeps the 262K KV pin and takes images up to 4 MP
  (the multi-card layouts take 8 MP). Three output-identical memory settings make that fit and are on by default: a
  96 MiB FlashInfer workspace, the MTP drafter sharing it, and the vision rotary applied in place. Measured 2026-10-21:
  4 MP passes at one and two sequences and after a 262K needle with 558-563 MiB free (bar 524 MiB); 8 MP stays short on
  one card. The row passed its Merlin 3.0.0 TP1 check on 2026-10-21 (pool 271,506 tokens); the KV pin is the checked value. The container
  route is removed for this release.

Upgrading from v1.0.0, and every change: [CHANGELOG.md](CHANGELOG.md#v130).

## What's new in v1.0.0

The first public release (2026-09-27): stock vLLM 0.30.0 from PyPI plus patches 0001-0005, the dual card and quad card
serve profiles at 262K with MTP K=3, and a container recipe. Its numbers: [docs/history.md](docs/history.md); its notes:
[CHANGELOG.md](CHANGELOG.md#v100).

## Documentation

- [CHANGELOG.md](CHANGELOG.md): release notes, newest first.
- [docs/reference.md](docs/reference.md): what the build and serve scripts do, serve-script variables and knobs,
  requirements, health checks, troubleshooting, the ThinkingCap profile, what is in v1.3.0, how the model sits on the
  cards, capacity, benchmarks, known behaviours and a glossary.
- Bench cards: [3.0.0 results, 2026-10-20](benchmarks/2026-10-20/BENCH-CARD.md) and [0.32 tile reclaim, 2026-10-15](benchmarks/2026-10-15/BENCH-CARD.md); historical 0.30 cards: [v1.3.0](benchmarks/2026-10-12/BENCH-CARD.md) and
  [ThinkingCap](../thinkingcap/benchmarks/2026-10-12/ThinkingCap-BENCH-CARD.md) (2026-10-12), [v1.0.0](https://github.com/halt95/qwen38-27b-3090s/blob/v1.3.1/benchmarks/2026-09-26/BENCH-CARD.md)
  (2026-09-26), [the earlier three-patch build](https://github.com/halt95/qwen38-27b-3090s/blob/v1.3.1/benchmarks/2026-09-25/BENCH-CARD.md) (2026-09-25).
- [docs/loop-check.md](docs/loop-check.md): the repetition-loop check.
- Patch notes: [patches/merlin-engine-v2.5.2/README.md](https://github.com/halt95/qwen38-27b-3090s/blob/v1.3.1/patches/merlin-engine-v2.5.2/README.md) (the 27B series) and
  [patches/flashinfer-0.6.18.post1/README.md](https://github.com/halt95/qwen38-27b-3090s/blob/v1.3.1/patches/flashinfer-0.6.18.post1/README.md) (the verify tile).
- [docs/history.md](docs/history.md): v1.0.0 on stock vLLM 0.30.0, with its numbers.

## Known limitations

- Tested on one setup: 4× RTX 3090 at 220 W, PCIe Gen4 x16, with peer-to-peer. sm_86 only; no other GPU, distribution,
  driver or non-P2P host has been run on our hardware. On cloud-hosted cards without P2P the 2- and 4-card RTX 4090 24 GB
  rows and the 1-, 2- and 4-card CMP 170HX 64 GB rows passed full-model runs: they are 🟡 testing (kernel parity
  qualification still to run). On the older v1.3.0 engine, two cloud-hosted RTX 3090s without P2P ran the TP2 layout
  with MTP (2026-10-09: 262K 3-needle 3/3, 255K decode 81.5 tok/s; not comparable to this release's numbers). The
  RTX 3090 no-P2P rows stay ⚫ until this release runs there (the four-card row was boot-checked with `--swarm` on
  2026-11-01). Other 24 GB-class sm_86 cards (RTX A5000, 3090 Ti)
  resolve to the same rows but
  are reported untested / community-validated; cards reporting under 24 GiB (A10, A10G, L4 at 22.5 GiB; 12 and 20 GB
  cards) are refused by preflight, because the launchers pin KV bytes for 24 GiB and the pool is not cut. See the
  [hardware matrix](../../docs/merlin/hardware-matrix.md).
- The container route returns in a later release.
- Windows: WSL2 only, and untested (⚫). See [WSL2](../../docs/merlin/hardware-matrix.md#wsl2) for the settings it needs.
- Switching the verify tile (`FLASHINFER_SM8X_FP8_SMALLQ`) or `SPEC_K`, or reusing prefix-cache blocks, can change output
  at near-tied tokens. The 0.30 check did not establish repeatability across fresh compile caches.
- Changing `--limit-mm-per-prompt` needs a fresh compile cache (`VLLM_CACHE_ROOT`).
- A same-day 0.32 tile-off GSM8K-200 remains pending.
- JSON-schema structured output was not tested at the historical 0.30 base release check; no new long-soak result is claimed here.
- The ThinkingCap token saving was measured on arithmetic word problems only.

The full list, with causes: [Known behaviours](docs/reference.md#known-behaviours).

## Credits

Qwen ([Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)), philbert
([AWQ int4 base checkpoint](https://huggingface.co/philbert440/Qwen3.8-27B-W4A16-AWQ)), BottleCap AI
([ThinkingCap-Qwen3.8-27B](https://huggingface.co/bottlecapai/ThinkingCap-Qwen3.8-27B); the ThinkingCap profile serves a
quantised derivative of it, not affiliated with or endorsed by BottleCap AI), [vLLM](https://github.com/vllm-project/vllm)
and lucifer1004 ([uniform multi-token decode](https://github.com/vllm-project/vllm/pull/47979), which patch 0001 ports),
[FlashInfer](https://github.com/flashinfer-ai/flashinfer) (which the verify tile patches),
[compressed-tensors](https://github.com/vllm-project/compressed-tensors) (the checkpoint format; `quant/` packs with it),
froggeric ([Qwen-Fixed-Chat-Templates](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates) v22.5),
[aikitoria's open-kernel-module patch](https://github.com/aikitoria/open-gpu-kernel-modules) (P2P on consumer Ampere),
Zagitta's [gpu-unlock](https://github.com/Zagitta/gpu-unlock) kernel module (the bench basis from 3.0.1; its NVIDIA firmware is personal-use and not shipped; [Sudzpel](https://github.com/Sudzpel), who proposed it with the first measurements on Merlin),
and a community Flash-Next
[idle-power report](https://github.com/halt95/merlin-engine/issues/2) and
[contribution](https://github.com/halt95/qwen38-flash-next-3090s/pull/3), behind the guard fix in Merlin engine v2.5.2.
Third-party notices are in the repository's `NOTICE`; the list with links per item: [Credit](docs/reference.md#credit).

## License

Code in this repository (patches, install and serve scripts, quantisation tools, checksums): Apache-2.0
([LICENSE](../../LICENSE)); the patches modify the Apache-2.0 engine (vLLM and Merlin engine v2.5.2) and FlashInfer, see `NOTICE` (repository root).
Model weights are not part of this repository. The checkpoint `halt95/Qwen3.8-27B-W4A16-Merlin` is Apache-2.0 like its
bases. The ThinkingCap checkpoint `halt95/ThinkingCap-Qwen3.8-27B-W4A16-Merlin` is not:

**Licence of the ThinkingCap weights.** ThinkingCap is licensed by BottleCap AI under the
[PolyForm Small Business License 1.0.0](https://polyformproject.org/licenses/small-business/1.0.0), plus BottleCap AI's
additional permission for personal, non-commercial use by an individual, free of charge. A business may use it only if it
qualifies as a "Small Business" under the PolyForm text (fewer than 100 employees and independent contractors in
total, and under US$1 million in prior-year revenue, in 2019 dollars adjusted for inflation); any other commercial use needs a separate licence from BottleCap AI. **There is no
unrestricted commercial use.** Required Notice: Copyright 2026 BottleCap AI (https://bottlecapai.com). These terms cover
the ThinkingCap weights on Hugging Face, which carry the licence files; they do not cover this repository's code, which
stays Apache-2.0. Copies of BottleCap AI's `LICENSE` and `NOTICE`, the PolyForm text and Qwen's Apache-2.0 text are in
[`licenses/`](licenses/) and [`../thinkingcap/licenses/`](../thinkingcap/licenses/); the changes made to the checkpoint are listed in `NOTICE` (repository root). This is a summary, not legal
advice.
