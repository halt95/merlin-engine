<!-- Modified by the Merlin project from the upstream vLLM README; see NOTICE. -->
# Merlin engine

Merlin is a vLLM fork tuned for NVIDIA Ampere and Ada (**sm_80, sm_86 and sm_89**) GPUs, with model recipes for Qwen3.8-Flash-Next and Qwen3.8-27B. **Merlin 3.0.1 is built on vLLM 0.32.0.** It combines the engine, launchers, checkpoint checksums and installation tooling in one repository. Model weights stay on Hugging Face.

The fork carries the sm_86 FP8 QSA attention reader, host-memory PLE embedding offload, RecoverSSM, prefix-cache retention, persistent top-k and guard telemetry. It serves an OpenAI-compatible API with vision, reasoning and tool calling. See the [upstream vLLM README](https://github.com/vllm-project/vllm/blob/v0.32.0/README.md) and [vLLM documentation](https://docs.vllm.ai/); Merlin changes are described in [CHANGELOG.md](../CHANGELOG.md) and [NOTICE](../../NOTICE).

## Models

The [models/ index](../../models/README.md) leads to each recipe's launchers, requirements, checksum list and licences.

| Recipe | Recipe version | Hugging Face checkpoint | Reference layout |
|---|---|---|---|
| [Flash-Next](../../models/flash-next/README.md) | 3.0.0 | [Qwen3.8-Flash-Next-W4A16-Merlin](https://huggingface.co/halt95/Qwen3.8-Flash-Next-W4A16-Merlin) | 4× RTX 3090, TP2 × PP2 + EP |
| [Swift1.5](../../models/swift1.5/README.md) | 3.0.0 | [Swift1.5-Qwen3.8-Flash-Next-W4A16-Merlin](https://huggingface.co/halt95/Swift1.5-Qwen3.8-Flash-Next-W4A16-Merlin) | Same quad layout; tested on the reference host (Merlin 3.0.0 release bench card) |
| [Swift1.5-Huihui](../../models/swift1.5-huihui/README.md) | 3.0.0 | [Swift1.5-Huihui-Qwen3.8-Flash-Next-W4A16-Merlin](https://huggingface.co/halt95/Swift1.5-Huihui-Qwen3.8-Flash-Next-W4A16-Merlin) | Same quad layout; tested on the reference host (Merlin 3.0.0 release bench card) |
| [Qwen3.8-27B](../../models/qwen38-27b/README.md) | 3.0.0 | [Qwen3.8-27B-W4A16-Merlin](https://huggingface.co/halt95/Qwen3.8-27B-W4A16-Merlin) | 2× RTX 3090 TP2 or 4× RTX 3090 TP4 |
| [ThinkingCap](../../models/thinkingcap/README.md) | 3.0.0 | [ThinkingCap-Qwen3.8-27B-W4A16-Merlin](https://huggingface.co/halt95/ThinkingCap-Qwen3.8-27B-W4A16-Merlin) | Profile of the 27B recipe; separate weight licence |

Recipe versions stay at 3.0.0 in Merlin 3.0.1 (the VERSION files are unchanged); the recipe changes are in the
[changelog](../CHANGELOG.md). The previous public release is Merlin 3.0.0.

## Install and serve

Use Linux x86-64 with glibc 2.38 or newer (Ubuntu 24.04, Debian 13, RHEL 10 or newer; the release wheel's `_C_stable_libtorch.abi3.so` needs GLIBC_2.38), a CUDA 13.x NVIDIA driver (R580 or later), Python 3.13 with venv support and headers, and uv. The engine installs from the prebuilt release wheel (sm_80, sm_86 and sm_89); building it from source instead needs C/C++ and Rust build tools and the prerequisites in [release/](../../release/README.md). For Flash-Next, plan for a 96 GB host-RAM allocation and at least 1 GiB of shared memory. The checkpoint is ~115 GiB; allow room for it plus caches, the build environment and any container image. Keep the GPUs free of other CUDA processes. The hardware matrix describes PCIe, BAR1 and P2P requirements.

Reference host: NVIDIA driver 610.43.02 (reports CUDA 13.3). The release lock selects a matched CUDA 13.0 toolkit (nvcc, crt, nvvm, nvJitLink 13.0.88, cccl 13.0.85, runtime 13.0.96, the pair torch 2.13.0 cu130 ships with); the release wheel is served on it by every recipe (2026-10-20). Other CUDA 13.x drivers (R580 or later) are expected to work through CUDA minor-version compatibility, but are untested. See the [driver requirements](../../release/install-0.32.md).

**Install status:** the shared 0.32 installer, engine wheel and FlashInfer 0.7.0.post1 tile tooling are integrated. The dependency lock is complete: 222 exact pins with artifact SHA-256 hashes, resolved with uv. A fresh install from the release wheel was verified on the reference host and on a second host with 4× RTX 4090 (2026-10-20, about 1.5 minutes, no native build); the container image is not yet qualified. Historical recipe sections and bench cards describe vLLM 0.30; the Merlin 3.0.0 notes use the 2026-10-15 measurements.

**Quick start** (one command: host checks, a few questions with recommended defaults, every download shown with its size, the prebuilt wheel in about 1.5 minutes, the checkpoint verified, the server started):


```bash
git clone https://github.com/halt95/merlin-engine.git && cd merlin-engine && ./setup.sh
```

Ctrl+C stops the server; `./setup.sh` again starts it straight away with the saved answers; `git pull && ./setup.sh` updates the engine without re-downloading the model. `--yes --model 27b` answers nothing, `--dry-run` shows the plan, `--reconfigure` asks again; every option, the update flow, where things live and how to uninstall are in [docs/SETUP.md](../../docs/SETUP.md), and [docs/how-to-serve.md](../../docs/how-to-serve.md) walks through it end to end. Nothing is installed system-wide (uv and Python 3.13 live under `.merlin/`, the engine in `venv/`).

**Manual route.** From the Merlin 3.0.1 checkout, install once with the shared [release/install-env.sh](../../release/install-env.sh), download a checkpoint, then serve it:

```bash
# Run from the repository root.
curl -LO https://github.com/halt95/merlin-engine/releases/download/merlin-v3.0.1/vllm-0.32.0+merlin.7002f56537-cp313-cp313-linux_x86_64.whl
MERLIN_WHEEL="$PWD/vllm-0.32.0+merlin.7002f56537-cp313-cp313-linux_x86_64.whl" bash release/install-env.sh ./venv   # prebuilt engine for sm_80, sm_86 and sm_89, checked against release/merlin-wheel.sha256
# without MERLIN_WHEEL the installer compiles the engine instead (about 1.5 hours on 8 cores)

# Download the checkpoint separately with the Hugging Face CLI.
venv/bin/hf download halt95/Qwen3.8-Flash-Next-W4A16-Merlin --local-dir /path/to/flash-next
(cd /path/to/flash-next && sha256sum -c /path/to/merlin-engine/models/flash-next/checkpoint.sha256)

# Serve it.
python3 merlin_cli.py serve /path/to/flash-next          # identifies the checkpoint, picks the profile for this host, starts vllm
python3 merlin_cli.py serve /path/to/flash-next --dry-run   # show the resolved cards, environment and vllm command first
```

`merlin serve` works for every checkpoint (Flash-Next, Swift1.5, Swift1.5-Huihui, 27B, ThinkingCap, Nemotron Super VL, Ornith 1.5): it reads `config.json` and the recipes' checksum lists to name the checkpoint (or take the name first: `merlin serve 27b /path/to/27b`), picks the largest **tested** layout that fits the GPUs it finds (a bigger host gets a card subset and a hint for the free cards; `--tp N`, `--profile ID`, `--instances N` and `GPUS=` narrow or multiply the choice), and starts the server through the one launcher, `tools/merlin/launch.py`. Every launcher knob (`PORT`, `HOST`, `KV_BYTES`, `MAX_PIXELS`, `RECOVERSSM`, `LPT`, `CHUNK`, `BLOCK_SIZE`, ...) is an environment variable, as before.

**Advanced: the recipe launchers.** `models/<recipe>/serve.sh` (and `serve-tp4.sh` / `serve-1card.sh` for 27B, ThinkingCap's forwarding pair) still exist and take the same arguments and knobs as in every earlier release; since Merlin 3.0.0 each is a short shim that calls `tools/merlin/launch.py --profile <its reference profile>`, so `bash models/flash-next/serve.sh /path/to/flash-next` and `MODEL=/path/to/27b bash models/qwen38-27b/serve.sh` run through the same launcher with that recipe's reference profile. The generated launchers under `tools/merlin/launchers/` pin the other hardware rows the same way.

The release dependency requirements include **FlashInfer 0.7.0.post1**, **humming-kernels 0.1.17** and **instanttensor ≥ 0.2.2**. The 27B sm_86 profiles require the small-q FP8 verify tile and `FLASHINFER_SM8X_FP8_SMALLQ=cta32`; use the release patch/proof tooling, rather than a stock FlashInfer installation. The installer and its lock file own the full dependency set.

## Hardware

Summary of the [hardware matrix](../../docs/merlin/hardware-matrix.md). A build target or community profile is not a hardware qualification.

| Hardware and model | Layout | KV pool (tokens) | Evidence |
|---|---|---|---|
| 4× RTX 3090: Flash-Next / Nemotron Super VL / Ornith 1.5 | TP2 × PP2 + EP (partition 25/23; 46/42; 25/23) | 924,993 / 1,856,367 (1 × 1M) / 894,618 | 🟢 Tested on the reference host |
| 4× RTX 3090 without P2P: Flash-Next; 27B TP4; Ornith 1.5 | TP2 × PP2 (the default); TP1 × PP4 by name (`--profile fn-quad-3090-nop2p-pp4`, partition 12/13/13/10: prefill +47-53 %, decode -14 %); Ornith, same recipe with P2P off | TP2 × PP2: ~924,993 (estimate); PP4: 906,581; Ornith 977,838 | ⚫ Untested (Flash-Next TP2 × PP2, 27B TP4; boot-checked with `--swarm` on 4× RTX 3090 without P2P, 2026-11-01); 🟢 Tested on the reference host with P2P off, 2026-10-19 (Flash-Next PP4); 🟢 Tested on a 4× RTX 3090 host without P2P, 2026-10-21 (Ornith) |
| 8× RTX 3090, Flash-Next | TP2 × PP4; TP4 × PP2 alternative | Estimated; confirm at boot | ⚫ Untested |
| 1× or 2× RTX 3090, 27B | TP1 (4 MP images) or TP2 | 271,506 / 826,689 | 🟢 Tested on the reference host |
| 4× RTX 3090, 27B | TP4, RecoverSSM on | 2,012,105 | 🟢 Tested on the reference host |
| sm_80 (A100, A30, CMP 170HX) and sm_89 (Lovelace) | Hardware-specific profiles | 4× CMP 170HX 64 GB Flash-Next (PP4 × TP1): 4,638,492; others unmeasured here | 🟡 Testing (full-model runs on cloud-hosted cards): 4× RTX 4090 and CMP 170HX 64 GB rows (4× CMP 170HX 64 GB Flash-Next measured 2026-10-19); ⚫ Untested: every other profile |

Flash-Next on two 24 GB cards is not implemented (it would need a host-RAM expert tier); single-card 27B is tested on the reference host (2026-10-21: 4 MP images, one 262K or two 131K sessions; 8 MP needs two or more cards). The generated Swift launcher passed its qualification run on 2026-10-16 and the Merlin 3.0.0 release bench card (2026-10-20, after the 2026-10-17 pre-release bench run): the Swift quad is tested on the reference host. The generated Huihui launcher passed its qualification run on 2026-10-16 and the Merlin 3.0.0 release bench card (2026-10-20): the Huihui quad is tested on the reference host. Swift's 27 T=0 outputs matched the Flash-Next quad despite argv differences; Huihui was checked on its own boot, image, and GSM8K run. Broader hardware and performance claims remain separately scoped.

## Measured on RTX 3090

Design-point and cache results were measured on RTX 3090 GPUs with a pre-release vLLM 0.32 build on 2026-10-15. The release-candidate engine and then the merged release-candidate engine matched that reference's T=0 tokens and logprobs on four configurations, with 28/28 documents plus four controls per configuration. Tile results used a separate 0.32 candidate build. These results describe the served configurations; other GPUs and the final assembled release tree require separate qualification. Aggregate throughput and inter-token latency describe different benchmark windows.

| Design point | Aggregate decode | Per-agent time | Preemptions |
|---|---|---|---|
| Flash-Next, 6×127K | 574.5 tok/s | 10.78 ms/tok | 0 |
| 27B TP4, 14×127K | 439.5 tok/s | 31.79 ms/tok | 0 |

Flash-Next cold prefill at that design point was **5,423 tok/s**. With MTP off, the 32K follow-up cache-hit ratios were **0.998** for Flash-Next (TTFT **5.78 → 0.315 s**) and **0.9996** for 27B TP2 (**18.25 → 0.52 s**).

On a vLLM 0.32 candidate build with the FlashInfer 0.7.0.post1 tile enabled, 27B inter-token latency at **4K / 32K / 131K** was **19.99 / 21.41 / 25.27 ms** on TP2 and **15.83 / 16.62 / 18.37 ms** on TP4. The tile-on GSM8K-200 mean was **192.0/200**. These values supersede the earlier untiled 0.32 measurements; they do not qualify the final release assembly.

The Merlin image cap is **8,388,608 pixels (8 MP)**. 27B TP2 serves images up to 8 MP. With the vision-peak option on (`VLLM_MERLIN_VISION_PEAK=1`, the TP2 launcher default), the 50-item TP2 stress sweep's worst minimum free was 806 MiB per card (2026-10-15), above the 524 MiB floor. A separate knob OFF/ON identity check matched tokens and logprobs on 6/6 prompts. Flash-Next (902 MiB) meets the target. 27B TP4 left 690 MiB free on the retired RecoverSSM-off pin; at the shipped chunk 4096 with RecoverSSM on, the 2026-10-20 bench card measured 862 MiB lowest free (one 8 MP image, two at once, and four in one prompt; 2 cards: 886 MiB), above the 524 MiB floor. Swift's 2026-10-15 window left 902 MiB free on stage 0; the generated Swift launcher passed the 2026-10-16 8 MP burst after 8K prefill with 544 MiB stage-0 minimum free, 20 MiB above the 524 MiB floor. The generated Huihui launcher independently passed the same gate with 902 MiB minimum free. The chunked vision encoder is inert at this cap; larger caps need separate qualification.

## Moving from qwen38-flash-next-3090s v2.5.x

The engine now lives at the repository root, and recipes live under `models/<model>/`. Use `release/install-env.sh` with a fresh environment for the new checkout; the old bundle-and-patch installation and versioned script paths belong to their old release tags. If your clone's remote points at the old URL:

```bash
git remote set-url origin https://github.com/halt95/merlin-engine.git
```

Flash-Next FGPC stays off. The 27B keeps FGPC on under its 0.32 name, `--enable-mamba-shared-prefix-checkpoint`. The native context remains 262,144 tokens and no shipped profile enables YaRN. A YaRN extension is opt-in through `--hf-overrides` (recipe in [How to serve](../how-to-serve.md#longer-context-with-yarn-flash-next)), is tested at 393,216 tokens on Flash-Next only, and changes numerics from the 0.30 releases. Sequence parallelism is off by default and can change output rounding. Oversubscribing the KV pool can still thrash prefix caching, and the upstream `qwen3_coder` parser can drop JSON-form `<tool_call>` blocks.

## Licence and security

The engine is Apache-2.0; retain [LICENSE](../../LICENSE), [NOTICE](../../NOTICE) and applicable third-party notices. Weight licences vary by checkpoint: see each recipe's `licenses/` directory and HF card, especially the Swift and ThinkingCap terms. See [SECURITY.md](../../SECURITY.md) for private vulnerability reporting and deployment assumptions.
