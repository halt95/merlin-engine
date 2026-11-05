<!-- Modified by the Merlin project; see NOTICE. -->
<div align="center">

<img src="docs/images/merlin-hero.svg" alt="Merlin: open models at their full 262K context on consumer GPUs" width="100%">

[![vLLM](https://img.shields.io/badge/based_on-vLLM_0.33.0-30A2FF)](https://github.com/vllm-project/vllm/tree/v0.33.0)
[![GPUs](https://img.shields.io/badge/GPUs-Ampere_·_Ada-76B900?logo=nvidia&logoColor=white)](#hardware)
[![Context](https://img.shields.io/badge/context-262K-ffb000)](#models)
[![Licence](https://img.shields.io/badge/licence-Apache--2.0-lightgrey)](LICENSE)
[![Support](https://img.shields.io/badge/support-Ko--fi-FF5E5B?logo=kofi&logoColor=white)](https://ko-fi.com/halt95)
[![Discord](https://img.shields.io/badge/discord-join_the_community-5865F2?logo=discord&logoColor=white)](https://discord.gg/jVDYAwdGzt)

**[Quick start](#quick-start) · [Models](#models) · [Features](#features) · [Performance](#performance) · [Hardware](#hardware) · [How to serve](docs/how-to-serve.md) · [Changelog](docs/CHANGELOG.md)**

</div>

<br>

Merlin is a [vLLM](https://github.com/vllm-project/vllm) fork tuned for NVIDIA Ampere and Ada GPUs. It runs seven open
models at their full 262K context (Nemotron 3.5 Super VL at its 1M context) on four RTX 3090s, behind one OpenAI-compatible API with vision, reasoning and
tool calling. One install, one command with pre-tuned profiles for your detected GPUs.

## Quick start

```bash
git clone https://github.com/halt95/merlin-engine.git && cd merlin-engine && ./setup.sh
```

The setup script checks your GPUs, suggests a model, shows each download and its size before it starts, then
launches the server on `127.0.0.1` (port 8000; 8100 for the 27B and ThinkingCap). Nothing is installed system-wide. Run it again to restart the server.

The [How to serve](docs/how-to-serve.md) guide covers the rest: [Merlin's commands and flags](docs/how-to-serve.md#2-merlin-commands),
talking to the server, serving on your network, picking GPUs and switching models.

<details>
<summary><b>Manual install</b></summary>

```bash
git clone https://github.com/halt95/merlin-engine.git && cd merlin-engine
curl -LO https://github.com/halt95/merlin-engine/releases/download/merlin-v3.0.2/vllm-0.33.0+merlin.665f43d2cf-cp313-cp313-linux_x86_64.whl
MERLIN_WHEEL="$PWD/vllm-0.33.0+merlin.665f43d2cf-cp313-cp313-linux_x86_64.whl" bash release/install-env.sh ./venv

venv/bin/hf download halt95/Qwen3.8-Flash-Next-W4A16-Merlin --local-dir ~/models/flash-next
(cd ~/models/flash-next && sha256sum -c "$OLDPWD/models/flash-next/checkpoint.sha256")
venv/bin/merlin serve ~/models/flash-next
```

The wheel is checked against `release/merlin-wheel.sha256`. Without `MERLIN_WHEEL` the installer builds the engine
from source, which takes about 1.5 hours on 8 cores. The first start compiles kernels; the server is ready when
`merlin: Ready: ...` prints.

</details>

## Models

| Model | Type | GPUs | KV pool, tokens | Decode at 262K | 8 agents, thinking on | 8 agents, thinking off (3.0.1 run) | GSM8K-200, thinking on | GSM8K-200, thinking off (3.0.1 run) |
|---|---|---|---|---|---|---|---|---|
| [Flash-Next](https://huggingface.co/halt95/Qwen3.8-Flash-Next-W4A16-Merlin) | 125B MoE | 4 | 924,993 | 207 tok/s | 654 tok/s | 664 tok/s | 197 (3.0.1 run) | 199 |
| [Swift1.5](https://huggingface.co/halt95/Swift1.5-Qwen3.8-Flash-Next-W4A16-Merlin) | Flash-Next fine-tune | 4 | 924,993 | 186 tok/s | 659 tok/s | 650 tok/s | 199 | 197 |
| [Swift1.5-Huihui](https://huggingface.co/halt95/Swift1.5-Huihui-Qwen3.8-Flash-Next-W4A16-Merlin) | Swift1.5, abliterated | 4 | 924,993 | 193 tok/s | 718 tok/s | 681 tok/s | 198 | 198 |
| [Qwen3.8-27B](https://huggingface.co/halt95/Qwen3.8-27B-W4A16-Merlin) | 27B dense | 1 / 2 / 4 | 271,506 / 826,689 / 2,012,105 | 77 / 146 / 250 tok/s | – / 682 / 940 tok/s | – / 615 / 854 tok/s | 198 (4 GPUs; 3.0.1 run) | 195 / 194 / 197 |
| [ThinkingCap](https://huggingface.co/halt95/ThinkingCap-Qwen3.8-27B-W4A16-Merlin) | 27B fine-tune | 2 / 4 | 826,689 / 2,012,105 | 160 / 223 tok/s | 708 / 939 tok/s | 633 / 832 tok/s | 198 (4 GPUs) | 193 / 195 |
| [Nemotron 3.5 Super VL](https://huggingface.co/halt95/Nemotron-3.5-Super-VL-120B-A12B-W4A16-Merlin) | 120B MoE, 1M context | 4 | 1,856,367 | 106 tok/s | 432 tok/s (3.0.1 run) | 432 tok/s | 197 (3.0.1 run) | 195 |
| [Ornith 1.5](https://huggingface.co/halt95/Ornith-1.5-122B-A10B-W4A16-Merlin) | 122B MoE | 4 | 894,618 | 111 tok/s | 528 tok/s (3.0.1 run) | 486 tok/s | 196 (3.0.1 run) | 190 |

<sub>Merlin 3.0.2. Measured on 4 x RTX 3090 at 220 W with P2P enabled (aikitoria's open-gpu-kernel-modules P2P driver) and Zagitta's gpu-unlock. Decode is one stream with thinking on (reasoning effort medium where the checkpoint has one; the Nemotron 3.5 Super VL and Ornith 1.5 templates have no effort levels), at the full
262,144-token context. 8 agents is total throughput for eight simultaneous follow-ups on 16K conversations; with thinking on (effort medium) a 512-token follow-up may end inside the reasoning, and reasoning tokens count. GSM8K-200 is correct answers out
of 200. GSM8K-200 follows the 3.0.1 card's protocol: thinking on at effort medium on four cards, re-run on 3.0.2 for Swift1.5, Swift1.5-Huihui and ThinkingCap; thinking off is from the 3.0.0 run, as the weights are unchanged. On the full GSM8K test set (1,319 problems, 5-shot, thinking off, temperature 0, 512 tokens) Flash-Next on 3.0.2 scored 1,281 (97.1 %), against 1,276 and 1,275 (96.7 %) for 3.0.1 in two runs of the same session. (3.0.1 run): figure from Merlin 3.0.1 on vLLM 0.32, not re-run for 3.0.2. All models use an FP8 KV cache and MTP speculative decoding. The bench cards
have the full figures and method.</sub>

## Features

- **More context in the same VRAM.** Embedding tables live in host RAM, so the cards hold more KV cache.
  Flash-Next keeps three full 262K conversations in memory on four 3090s.
- **FP8 KV cache on Ampere,** with our own kernels for cards that lack native FP8.
- **Faster prefill.** Up to 33 % faster at 262K in Merlin 3.0.1 than in 3.0.0, with the optional gpu-unlock host upgrade;
  3.0.2 against 3.0.1: unchanged within run-to-run spread, except Nemotron 3.5 Super VL +8 %.
- **On vLLM 0.33.** torch 2.14, Triton 3.8 and FlashInfer 0.7.2.post1, with the sm_86 FP8 patches carried over.
- **Built for agents.** Follow-up turns resume from cache, even when more conversations are open than the cache can
  hold.
- **KV RAM cache (experimental).** `--kv-ram-cache 24g` keeps conversation prefixes the GPU evicts in host RAM and
  restores them with identical output: on Flash-Next a 262K restore takes 1.2–1.4 s instead of about 45 s to recompute (Merlin 3.0.1, vLLM 0.32).
- **Swarm mode (experimental).** `--swarm` raises the session cap on four RTX 3090s, from 8 to 24 on the Flash-Next
  family and from 24 to 32 on the 27B family, with a 48 GB KV RAM cache. Tested on Flash-Next and the 27B with P2P (Merlin 3.0.1, vLLM 0.32).
- **PLE table on NVMe (experimental).** `--ple-store disk` reads the Flash-Next family's embedding table from the model
  files on NVMe instead of holding it in host RAM: free memory rises by 47.7–47.8 GiB (38.8–38.9 GiB with `--ple int6`),
  and output matched RAM token for token. It costs about 2 % of decode speed once the table is cached, 9–11 % on text
  whose rows are not cached yet, and the first token of a cold 262K prompt takes 7–53 % longer; cached, it is unchanged.
  With a large `--kv-ram-cache` the table never fully warms (+10–11 % decode at 262K with 96g). Measured on 4 x RTX 3090
  with one Gen4 NVMe; other disks and layouts are untested ⚫. The figures are from the 0.32 tree; the 0.33 re-run has not been run for 3.0.2. `ram` stays the default. [Costs and limits](docs/how-to-serve.md#the-ple-table-on-nvme-experimental).
- **DFlash2 for the Flash-Next family.** `--spec dflash2` on four RTX 3090s with P2P decodes 5–18 % faster than MTP
  (Merlin 3.0.1, vLLM 0.32), which stays the default.
- **Vision, reasoning and tool calling** on every model, with images up to 8 MP (Nemotron: its native ~3.4 MP; the 27B
  on one card: 4 MP; not re-measured on Pillow 13).
- **Low idle power.** Four idle 3090s draw 110 W (Merlin 3.0.0, vLLM 0.32).
- **One command.** `merlin serve <model>` picks the best layout for the GPUs it finds.
- **A Ready line and a bug-report file.** `merlin serve` prints the URL, model, context length and key location once the
  server answers (`MERLIN_READY_BANNER=0` turns it off); `merlin doctor --report [FILE] [--log FILE] [--lines N]` writes one
  redacted file for bug reports; [client settings](docs/clients.md) for OpenAI-compatible and Anthropic-compatible clients such as opencode, Cline/Roo, Aider and Continue.

<details>
<summary><b>What Merlin adds to vLLM</b></summary>

**More context in the same VRAM**

- **Token embedding tables in host RAM.** The token embedding tables of the main model and its MTP drafter are read
  from pinned host memory, on by default for the Flash-Next and Qwen3.8-27B families (about 1.2 GiB per card freed on
  27B at TP=2), with no speed loss.
- **N-gram embedding table in host RAM.** Flash-Next's n-gram embedding table (about 47.7 GiB) lives in system
  memory, one copy shared by every GPU, so the VRAM goes to KV cache: a 924,993-token pool on four RTX 3090s, three
  full 262K sessions resident at once. With `--ple-store disk` (experimental) it is read from NVMe instead.
- **FP8 KV cache on Ampere.** An sm_86 FP8 QSA attention reader for Flash-Next, and an sm_86 FP8 small-query verify
  tile for 27B, so every checkpoint keeps an FP8 KV cache on cards with no native FP8.
- **QSA indexer scratch reserve (opt-in).** `VLLM_MERLIN_QSA_SCRATCH=reserve|workspace` lets the KV sizing see
  Flash-Next's QSA indexer scratch; `off` stays the default, and KV capacity figures for the two modes have not been measured for 3.0.2.

**Faster decode and prefill**

- **RecoverSSM on every checkpoint.** RecoverSSM recovers the GDN and Mamba-2 state for MTP speculative decoding,
  on by default for Flash-Next, Swift1.5, Swift1.5-Huihui, Qwen3.8-27B, ThinkingCap, Nemotron 3.5 Super VL and
  Ornith 1.5. The Flash-Next family drafts over the full vocabulary with a BF16 draft head.
- **Faster prefill, with the optional gpu-unlock host upgrade.** Cold prefill of a full 262K prompt on four
  RTX 3090s, Merlin 3.0.1 against 3.0.0: +17 % on Flash-Next, +32 % on Qwen3.8-27B and ThinkingCap, +33 % on Nemotron 3.5
  Super VL and Ornith 1.5 (measured on vLLM 0.32). A stock-versus-unlocked A/B ties the gain to the unlocked host on Flash-Next and the 27B; Super VL and Ornith 1.5 had no A/B.
  Merlin 3.0.2 on vLLM 0.33, against 3.0.1: unchanged within run-to-run spread, except Nemotron 3.5 Super VL +8 %.
- **Changes to pipeline-parallel decode with RecoverSSM.** Commit kernels are warmed on every pipeline rank at boot, and the
  deferred commit reuses a persistent snapshot ring instead of allocating snapshot storage at every step; output is unchanged. Optional switches,
  both off by default: `VLLM_RECOVERSSM_PP_SPLIT_RECV=1` and `VLLM_MERLIN_PP_DECODE_BALANCE=1`. Effect on decode:
  not yet measured; the A/B has not been run for 3.0.2.
- **Decode-first step budget (experimental, off by default).** `VLLM_MERLIN_DECODE_FIRST=1` keeps running sessions
  decoding while a long cold prefill runs, instead of stalling until it ends. Temperature-0 output is not proven identical
  with it on, so it stays off.
- **DFlash2 drafter for the Flash-Next family.** `--spec dflash2` runs incoai's block-diffusion drafter on four RTX 3090s
  with P2P: decode 5–18 % faster than MTP (Merlin 3.0.1, vLLM 0.32), and the server holds three 262K sessions instead of
  six 131K ones. MTP stays the default.
- **All-reduce tuned for PCIe P2P.** Flash-Next on P2P hosts uses a PCIe IPC all-reduce for decode-sized messages
  and NCCL's Simple protocol for prefill.
- **Marlin W8A16** for Flash-Next's INT8 GDN projections.

**Built for agents**

- **Follow-ups resume from cache.** Prefix-cache retention keeps each conversation's prefix on a 64-token grid
  instead of the 4,096-token block, so the next turn starts from cache instead of re-reading the whole conversation
  (Flash-Next v2.5.1: median follow-up time to first token 0.96 s, was 1.56 s).
- **Cache-thrash protection, on by default.** Cost-aware admission and session-coherent yields keep follow-ups
  cached when more sessions are open than the pool holds: under 8 × 131K and 12 × 80K oversubscription, follow-up
  turns reused 74.7 % and 82.1 % of their prompt from cache (Merlin 3.0.0, vLLM 0.32).
- **KV RAM cache (experimental).** `--kv-ram-cache SIZE` keeps prefixes the GPU evicts in host RAM and restores them
  instead of recomputing them, with exactly the same output as a GPU cache hit. Flash-Next family with MTP and
  Qwen3.8-27B; off by default.
- **Longer context (experimental).** `--context 512k` serves past 262K with YaRN built from the checkpoint's config,
  and refuses a size above the layout's KV pool. Most combinations are untested ⚫; see the
  [Flash-Next reference](models/flash-next/docs/reference.md#context-size-302).
- **Swarm mode (experimental).** `--swarm` raises the session cap on four RTX 3090s, with or without P2P: 8 to 24 on
  Flash-Next, Swift1.5 and Swift1.5-Huihui, 24 to 32 on Qwen3.8-27B and ThinkingCap, with a 48 GB KV RAM cache. Tested
  on Flash-Next and the 27B with P2P (Merlin 3.0.1, vLLM 0.32); Swift1.5, Swift1.5-Huihui and every model without P2P use the same cap untested
  (⚫, expected to work).
- **A chat template that keeps the cache.** Every Flash-Next-family and Qwen3.8-27B-family recipe serves a fixed Qwen3.8 template that accepts system
  messages after the first turn, so coding agents' per-turn notes no longer break the prefix cache.
- **Vision, reasoning and tool calling on every checkpoint,** with images up to 8 MP (Nemotron its native ~3.4 MP,
  one-card 27B 4 MP) and a per-image pixel cap (`MAX_PIXELS`).
- **Forged control tokens stay text.** Chat messages, tools and documents that spell a control token such as `<|im_end|>`
  are encoded as plain text (`--sanitize-control-tokens`, on by default; `--reject-mm-with-string-template`, off by
  default, refuses multimodal input with a string-format template); Ornith 1.5 also covers its six FIM tokens
  (`VLLM_MERLIN_EXTRA_CONTROL_TOKENS`). `tool_choice="none"` keeps tool-call markup the model wrote as text on the
  Qwen-family parsers.

**Steady in long runs**

- **Repeatable greedy output and fault detection on any rank**, with faster shutdown.
- **Idle cards drop to P8** with the runtime bounds guards on (Merlin 3.0.0, vLLM 0.32): 110 W idle on four RTX 3090s, against 446 W when the
  guard reader kept them awake.
- **Compile-cache fault guard, on by default.** After a sticky CUDA fault the next start sets the used compile cache aside
  and recompiles once, and the compiled model file is checked against its SHA-256 at load.
  `VLLM_MERLIN_COMPILE_CACHE_GUARD=0` restores the 3.0.1 behaviour.

**One install, one command**

- **`./setup.sh`** checks the host, recommends a checkpoint and layout for your GPUs, shows every download with its
  size, installs the prebuilt engine (no compiling, nothing system-wide; about 1.5 minutes on 3.0.0, and the 3.0.2 fresh
  install has not been timed) and starts the server.
- **`merlin serve <checkpoint>`** recognises the checkpoint, picks the largest tested layout for the GPUs it finds,
  and starts the server; `merlin doctor` checks the host first. Every layout is a profile of `tools/merlin/launch.py`;
  every knob (`PORT`, `HOST`, `KV_BYTES`, `MAX_PIXELS`, ...) is an environment variable.
- **`merlin serve --instances N`** now stays in the foreground and stops all N servers together.
- **`P2P=ipc` (opt-in, untested):** a third interconnect mode on the Flash-Next rows with the PCIe-IPC all-reduce, for
  hosts where NCCL's P2P bootstrap hangs but CUDA peer access works.
- **Container:** `MERLIN_WHEEL_URL` installs the release wheel into the image instead of compiling, and `MERLIN_PROFILE`
  starts a chosen Flash-Next-family profile (thanks to brandon-nuttall); the 3.0.2 image is not yet requalified.
- **Profiles beyond the reference host:** RTX 3090 without P2P, eight RTX 3090s, RTX 4090 (24 and 48 GB),
  RTX A6000 / A40, A100, A30 and CMP 170HX.
- **One prebuilt wheel** for sm_80, sm_86 and sm_89, installed from a hash-locked requirements file.
- **New variants just work.** A Flash-Next-family checkpoint that ships its own FP8 KV scales is served with them,
  with no repository update.

</details>

## Performance

<img src="docs/images/decode.svg" alt="Single-stream decode at 4K, 131K and 262K context for every model on Merlin 3.0.2 (vLLM 0.33)" width="100%">

<details>
<summary><b>Prefill and time to first token at 262K</b></summary>

| Model | GPUs | Prefill | Time to first token |
|---|---|---|---|
| Flash-Next | 4 | 6,782 tok/s | 38.6 s |
| Swift1.5 | 4 | 6,762 tok/s | 38.7 s |
| Swift1.5-Huihui | 4 | 6,830 tok/s | 38.3 s |
| Qwen3.8-27B | 1 / 2 / 4 | 586 / 1,112 / 2,111 tok/s | 447.3 / 235.4 / 124.1 s |
| ThinkingCap | 2 / 4 | 1,115 / 2,108 tok/s | 234.9 / 124.2 s |
| Nemotron 3.5 Super VL | 4 | 4,217 tok/s | 62.1 s |
| Ornith 1.5 | 4 | 2,288 tok/s | 114.5 s |

<sub>One stream, cold cache, 261,888-token prompt.</sub>

</details>

**Bench cards:** [Flash-Next](models/flash-next/benchmarks/2026-11-04/BENCH-CARD.md) · [Swift1.5](models/swift1.5/benchmarks/2026-11-05/BENCH-CARD.md) · [Swift1.5-Huihui](models/swift1.5-huihui/benchmarks/2026-11-05/BENCH-CARD.md) · [Qwen3.8-27B](models/qwen38-27b/benchmarks/2026-11-04/BENCH-CARD.md) · [ThinkingCap](models/thinkingcap/benchmarks/2026-11-05/BENCH-CARD.md) · [Nemotron 3.5 Super VL](models/nemotron-super-vl/benchmarks/2026-11-04/BENCH-CARD.md) · [Ornith 1.5](models/ornith1.5/benchmarks/2026-11-04/BENCH-CARD.md)

## Hardware

| GPUs | Status |
|---|---|
| 2× or 4× RTX 3090 | 🟢 Tested and tuned |
| 4× RTX 3090 without P2P | 🟠 Partly tested (Ornith 1.5, Flash-Next PP4), tuning in progress |
| 4× RTX 4090 | 🟡 Testing on cloud-hosted cards, tuning in progress |
| 2× or 4× CMP 170HX 64 GB | 🟡 Testing on cloud-hosted cards, tuning in progress |
| 8× RTX 3090, RTX 4090 48 GB, A100, A30 | ⚫ Profile included, not yet tested by us |

Any NVIDIA sm_80, sm_86 or sm_89 card can run Merlin. The [How to serve](docs/how-to-serve.md#6-which-gpus-run-what)
guide lists which models fit on which cards, and how many full-length conversations each setup holds.

On RTX 3090s, PCIe P2P and Zagitta's [gpu-unlock](docs/merlin/hardware-matrix.md#optional-host-upgrade-gpu-unlock-geforce-ga102) are optional host upgrades: Merlin runs without
them, and our bench numbers (3.0.1 and 3.0.2) are measured with both. In a stock-versus-unlocked A/B the unlock sped up prefill by
about 16–29 %; decode and output are unchanged.

## Requirements

| | |
|---|---|
| OS | Linux x86-64, glibc 2.38 or newer |
| Driver | NVIDIA R580 or later (CUDA 13) |
| Python | 3.13 with [uv](https://docs.astral.sh/uv/) |
| Compiler | GCC 13 or newer |
| Host RAM | 72 GiB for the Flash-Next family on four cards (tested minimum; more with `--kv-ram-cache` or `--swarm`; `--ple-store disk`, experimental, moves the 47.7 GiB table to NVMe) |
| Disk | About 115 GiB for Flash-Next, 66 GiB for Ornith 1.5 |

`merlin doctor` checks your system before anything starts.

## Credits

**Models:** [Qwen](https://huggingface.co/Qwen) · [UkisAI](https://huggingface.co/ukisai/Swift1.5-Qwen3.8-Flash-Next) (Swift 1.5) ·
[huihui-ai](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated) (abliteration) · [BottleCap AI](https://huggingface.co/bottlecapai/ThinkingCap-Qwen3.8-27B) (ThinkingCap) ·
[NVIDIA](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16) (Nemotron) · [ornith-ai](https://huggingface.co/ornith-ai/Ornith-1.5-122B) (Ornith 1.5, a fine-tune of Qwen's [Qwen3.5-122B-A10B](https://huggingface.co/Qwen/Qwen3.5-122B-A10B))

**Quantisation:** [Intel](https://huggingface.co/Intel/Qwen3.8-Flash-Next-W4A16-AutoRound) (AutoRound) ·
[RadixArk](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4) (FP8 PLE table) · DominikBucko (MTP packing) ·
[philbert](https://huggingface.co/philbert440) (AWQ)

**Engine:** [vLLM](https://github.com/vllm-project/vllm), FlashInfer, compressed-tensors and the vLLM community's
PLE-offload work · [Sumandora](https://github.com/Sumandora/remove-refusals-with-transformers) (abliteration method) ·
[froggeric](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates) (chat template) · [aikitoria](https://github.com/aikitoria/open-gpu-kernel-modules)
(P2P on consumer Ampere) · [Zagitta](https://github.com/Zagitta/gpu-unlock) (GPU unlock) · [incoai](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2) (DFlash2 drafter) ·
[club-3090](https://github.com/noonghunna/club-3090/discussions/1735) (sm_80 FP8 KV decoder port) · [Sudzpel](https://github.com/Sudzpel) (GPU unlock proposal and measurements) ·
[brandon-nuttall](https://github.com/brandon-nuttall) (YaRN past 262K report, `SEQS`/`MAXLEN` launcher fix, `use_local_argmax_reduction`, TP2×PP4 community rig) · [R-omk](https://github.com/R-omk) (YaRN past 262K crash report and recipes) · [CoolShades](https://github.com/CoolShades) (idle-power report and first fix behind the v2.5.2 engine fix) ·
[Neiko2002](https://huggingface.co/Neiko2002) · [Yarikdevcom](https://github.com/yarikdevcom) (independent work on the same hardware class)

Upstream fixes are credited in the engine repository.

## Support

Merlin is built and tested on one home machine. If it saves you time, you can support it on
[Ko-fi](https://ko-fi.com/halt95).

## Licence

The engine is Apache-2.0, like vLLM; see [LICENSE](LICENSE) and [NOTICE](NOTICE). Each model has its own licence
on its Hugging Face card.

<sub>Coming from `qwen38-flash-next-3090s`? This repository replaces it. Run
`git remote set-url origin https://github.com/halt95/merlin-engine.git`, then reinstall with `./setup.sh`.</sub>
