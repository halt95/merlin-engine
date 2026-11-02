# Model recipes for Merlin 3.0.1

Each directory contains a model launcher, KV-scale sidecar, checkpoint checksums,
licences, benchmark cards and documentation. The repository root contains the
engine, based on **vLLM 0.32.0**. Checkpoints remain on Hugging Face; shared
installation and build instructions are in [release/](../release/README.md).

| Recipe | VERSION | Hugging Face checkpoint | Hardware | Launcher | Status |
|---|---|---|---|---|---|
| [flash-next/](flash-next/README.md) | 3.0.0 | [Flash-Next](https://huggingface.co/halt95/Qwen3.8-Flash-Next-W4A16-Merlin) | 4× RTX 3090 (sm_86), peer-to-peer | [serve.sh](flash-next/serve.sh) | 0.32 CLI, 8,388,608-pixel image cap, shared-prefix checkpointing off; measured 924,993-token KV pool. |
| [swift1.5/](swift1.5/README.md) | 3.0.0 | [Swift1.5](https://huggingface.co/halt95/Swift1.5-Qwen3.8-Flash-Next-W4A16-Merlin) | 4× RTX 3090 recipe | [serve.sh](swift1.5/serve.sh) | Generated 0.32 recipe with checkpoint-specific scales and medium reasoning by default; full-vocabulary MTP head (`DRAFT_HEAD` default `bf16`), shared-prefix checkpointing off, 8 MP default. A qualification run validated this generated launcher on 2026-10-16; the 3.0.0 release bench card passed it: tested on the reference host. |
| [swift1.5-huihui/](swift1.5-huihui/README.md) | 3.0.0 | [Swift1.5-Huihui](https://huggingface.co/halt95/Swift1.5-Huihui-Qwen3.8-Flash-Next-W4A16-Merlin) | 4× RTX 3090 recipe | [serve.sh](swift1.5-huihui/serve.sh) | Generated 0.32 recipe with dense recalibrated scales, distinct model aliases and medium reasoning by default; full-vocabulary MTP head (`DRAFT_HEAD` default `bf16`), shared-prefix checkpointing off, 8 MP default. A qualification run validated this generated launcher on 2026-10-16, and the Merlin 3.0.0 release bench card (2026-10-20) tested it on the release wheel. |
| [qwen38-27b/](qwen38-27b/README.md) | 3.0.0 | [27B](https://huggingface.co/halt95/Qwen3.8-27B-W4A16-Merlin) | 2× or 4× RTX 3090 (sm_86), peer-to-peer | [serve.sh](qwen38-27b/serve.sh), [serve-tp4.sh](qwen38-27b/serve-tp4.sh) | 0.32 CLI, 8 MP default and shared-prefix checkpointing on; FlashInfer 0.7.0.post1 small-query verify tile. KV pools: 826,689 tokens TP2 / 2,012,105 TP4 (RecoverSSM on). 27B TP2 serves images up to 8 MP. With the vision-peak option on (`VLLM_MERLIN_VISION_PEAK=1`, the TP2 launcher default), the 50-item TP2 stress sweep's worst minimum free was 806 MiB per card (2026-10-15), above the 524 MiB floor. A separate knob OFF/ON identity check matched tokens and logprobs on 6/6 prompts. Flash-Next (902 MiB) meets the target. 27B TP4 left 690 MiB free on the retired RecoverSSM-off pin; at the shipped chunk 4096 with RecoverSSM on, the 2026-10-20 bench card measured 862 MiB lowest free (one 8 MP image, two at once, and four in one prompt; 2 cards: 886 MiB), above the 524 MiB floor. |
| [thinkingcap/](thinkingcap/README.md) | 3.0.0 | [ThinkingCap](https://huggingface.co/halt95/ThinkingCap-Qwen3.8-27B-W4A16-Merlin) | Same 27B recipe | [serve.sh](thinkingcap/serve.sh), [serve-tp4.sh](thinkingcap/serve-tp4.sh) | Forwards to the 27B launchers with its own checkpoint (medium reasoning by default, like every checkpoint). Inherits their settings and limits; 27B measurements do not establish checkpoint-specific quality. |
| [nemotron-super-vl/](nemotron-super-vl/README.md) | 3.0.0 | [Super VL](https://huggingface.co/halt95/Nemotron-3.5-Super-VL-120B-A12B-W4A16-Merlin) | 4× RTX 3090 (sm_86), peer-to-peer | [serve.sh](nemotron-super-vl/serve.sh) | New in 3.0.0, **tested on the reference host** (release bench card): TP2×PP2+EP, RecoverSSM, MTP K=3, FP8 KV pinned at a provisional 2.50 GiB per card (1,856,367-token pool at the shipped 1M default: one 1M session or 5 × 262K resident). |
| [ornith1.5/](ornith1.5/README.md) | 3.0.0 | [Ornith 1.5](https://huggingface.co/halt95/Ornith-1.5-122B-A10B-W4A16-Merlin) | 4× RTX 3090 (sm_86), with or without peer-to-peer | [serve.sh](ornith1.5/serve.sh) | New in 3.0.0, **tested on the reference host** (`ornith-quad-3090`); the no-P2P row `ornith-quad-3090-nop2p` is tested on a 4× RTX 3090 host without P2P. Ornith's own chat template and sampling; uniform verify on; 3 × 262K resident |

Recipe versions stay at 3.0.0 in Merlin 3.0.1 (the VERSION files are unchanged); the recipe changes are in the
[changelog](../docs/CHANGELOG.md). A hardware recipe is not a claim of validation on every GPU
with the same memory capacity. Swift-family pools of 924,993 tokens were measured on both generated launchers (2026-10-16); session-residency targets inherited from
0.30 are not new 0.32 measurements.

Conventions:

- A recipe's `serve.sh` is its canonical entry point and, since Merlin 3.0.0, a
  short shim for the one launcher `tools/merlin/launch.py --profile <id>` (same
  arguments and knobs).
  `merlin serve <checkpoint-dir>` is the front door. Repository-root
  `scripts/serve.sh` and `scripts/serve-huihui.sh` forward to those entry points
  for Merlin 3.0.0. `scripts/build.sh` forwards to the wheel builder
  `release/build-wheel.sh`; use `release/install-env.sh` to create a serving environment.
- Image caps are served defaults, not enforced limits. A client that raises one per request
  (`mm_processor_kwargs`) goes past the measured memory headroom and can run the engine out
  of memory; keep client harnesses at or below each recipe's documented cap.
- `checkpoint.sha256` checks an existing download: run `sha256sum -c` with the
  recipe's list from inside the checkpoint directory. Flash-Next's list uses
  revision `cec53e9f90e5995b81691604ec0b32191debf797` and omits the mutable card.
- `benchmarks/<date>/` identifies dated evidence. Check each card's engine
  version, checkpoint and configuration before comparing numbers; older cards
  document historical 0.30 releases.
- Recipe changelogs preserve package history. Repository-wide release notes,
  notices and installation instructions describe the combined engine release.
- The 0.32 installer requires a complete qualified dependency hash lock and the
  matching FlashInfer patch tooling. It rejects an incomplete lock; recipe
  availability does not imply that installation or GPU qualification has passed.

## Checking the recipes

Use a prepared test environment at `venv/` from the repository root. These
checks use local files and CPU stubs; they do not qualify GPU serving.

```bash
venv/bin/python models/check_links.py
venv/bin/python models/check_links.py --list-retired
```
