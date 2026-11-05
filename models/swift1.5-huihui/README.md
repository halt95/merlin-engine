# Swift1.5-Huihui · Merlin 3.0.2

The launcher serves Froggeric's fixed Qwen3.8 template,
[`froggeric-qwen-v22.5.jinja`](../qwen38-27b/templates/froggeric-qwen-v22.5.jinja), the same template as the
27B recipes. Unlike the checkpoint's own `chat_template.jinja` (vendored in
[`models/flash-next/templates/`](../flash-next/templates/chat_template.jinja)), it accepts system messages after
the first turn, so coding agents' per-turn notes stay in place on `/v1/messages` and the prefix cache keeps
hitting. `CHAT_TEMPLATE=<path>` serves another template; `CHAT_TEMPLATE=` (empty) serves the checkpoint's own.

Recipe version: [3.0.0](VERSION). This is the abliterated Swift 1.5 W4A16 checkpoint, served by the engine at the repository root (currently vLLM 0.33.0 plus Merlin changes). The weights stay on Hugging Face:

[halt95/Swift1.5-Huihui-Qwen3.8-Flash-Next-W4A16-Merlin](https://huggingface.co/halt95/Swift1.5-Huihui-Qwen3.8-Flash-Next-W4A16-Merlin).

## Qualification status

The canonical [serve.sh](serve.sh) is generated from `tools/merlin/profiles/flash-next.json` by `tools/merlin/preflight.py`, using profile `fn-swift1.5-huihui-quad-3090`. It starts this checkout directly. A boot-and-check run of the generated launcher on 2026-10-16 gave: boot 398 s, GSM8K-20 20/20, ITL 4K / 32K 18.78 / 18.87 ms, and the Merlin 3.0.0 release bench card on the release wheel (2026-10-20): the Huihui quad is tested on the reference host.

Huihui has its own [3.0.2 bench card](benchmarks/2026-11-05/BENCH-CARD.md) and [3.0.1 bench card](benchmarks/2026-11-02/BENCH-CARD.md) (both below) and its own [3.0.0 bench card](benchmarks/2026-10-20/BENCH-CARD.md) (2026-10-20: decode, prefill, residency, 8 MP image, 2-agent, GSM8K-200 and needles, the needle panel from a 2026-10-17 pre-release build; its 8 agents were re-run on the released install, 2026-10-21); the remaining heavy results are **inherited from Swift**. Swift at 8 MP (8,388,608 pixels per image) on 4× RTX 3090 / 0.32 measured **stage-0 minimum free memory 902 MiB** on 2026-10-15. Huihui's generated launcher independently passed the 2026-10-16 8 MP check with 902 MiB minimum free; broader heavy results remain inherited from Swift. See [Swift's qualification status](../swift1.5/README.md#qualification-status).

The 924,993-token KV pool was measured on this generated launcher (2026-10-16). Earlier quality and speed numbers are recorded in the [2026-10-12 bench card](benchmarks/2026-10-12/Huihui-BENCH-CARD.md); those vLLM 0.30 results do not qualify Merlin 3.0.0.

## Merlin 3.0.2 bench card (2026-11-05)

Maintainer measurements on 4× RTX 3090 (220 W, PCIe Gen4) with P2P (the aikitoria patched driver) and the Zagitta [gpu-unlock](https://github.com/Zagitta/gpu-unlock) kernel module, on vLLM 0.33.0, one fresh `merlin serve` boot, 2026-11-05; method and every cell in the [bench card](benchmarks/2026-11-05/BENCH-CARD.md). Decode is one stream with thinking on at reasoning effort medium, the server default (tok/s after the first streamed token); prefill is one cold request. With thinking on, a 512-token agent follow-up may end inside the reasoning, and reasoning tokens count. Rows marked (3.0.1 run) were not re-run for 3.0.2 and are on the 3.0.1 card below.

| Measurement | Swift1.5-Huihui |
|---|---|
| Decode, 4K / 131K / 262K prompt | 180 / 189 / 193 tok/s |
| Cold prefill, 10K / 100K / 262K prompt | 6,608 / 7,008 / 6,830 tok/s |
| Agents at once (8 × 16K conversations), total, thinking on / off | 703 / 681 tok/s (3.0.1 run) |
| GSM8K-200 (5-shot), correct of 200, thinking on at medium / thinking off | 198 / 198 (3.0.1 run; thinking off from the 3.0.0 run) |
| Sessions held at once | Not re-measured on 3.0.2. 3.0.0 run: 3 × 262K and 6 × 131K, every session resident (re-sent prompts ≥ 99.91 % cached), 0 preemptions. |
| 8 MP image, one and two requests at once | Not re-measured on 3.0.2 (Pillow 13). 3.0.0 run, Pillow 12: correct; lowest free memory 1,136 MiB on any card. |
| KV pool | 924,993 tokens |
| Against 3.0.1 | Within run-to-run spread: decode +1 % / 0 % / -2 % at 4K / 131K / 262K, 262K prefill -1 %. Thinking off at 262K reads -10 %, with the 3.0.1 figure inside the spread of the three 3.0.2 sends (144 to 164 tok/s): the step time is level and MTP acceptance lower. |

## Merlin 3.0.1 bench card (2026-11-02)

Maintainer measurements on 4× RTX 3090 (220 W, PCIe Gen4) with P2P (the aikitoria patched driver) and the Zagitta [gpu-unlock](https://github.com/Zagitta/gpu-unlock) kernel module, one fresh `merlin serve` boot per configuration, 2026-11-01 to 2026-11-02; method and every cell in the [bench card](benchmarks/2026-11-02/BENCH-CARD.md). Decode is one stream with thinking on at reasoning effort medium, the server default (tok/s after the first streamed token); prefill is one cold request. With thinking on, a 512-token agent follow-up may end inside the reasoning, and reasoning tokens count. Thinking-off GSM8K is from the 3.0.0 run. Rows marked (3.0.0 run) were not re-run for 3.0.1; they and the rest of the 3.0.0 figures are on the [3.0.0 bench card](benchmarks/2026-10-20/BENCH-CARD.md).

| Measurement | Swift1.5-Huihui |
|---|---|
| Decode, 4K / 131K / 262K prompt | 178 / 189 / 196 tok/s |
| Cold prefill, 10K / 100K / 262K prompt | 6,612 / 7,075 / 6,866 tok/s |
| Agents at once (8 × 16K conversations), total, thinking on / off | 703 / 681 tok/s |
| GSM8K-200 (5-shot), correct of 200, thinking on at medium / thinking off | 198 / 198 |
| Sessions held at once (3.0.0 run) | 3 × 262K and 6 × 131K, every session resident (re-sent prompts ≥ 99.91 % cached), 0 preemptions |
| 8 MP image, one and two requests at once (3.0.0 run) | correct; lowest free memory 1,136 MiB on any card |
| KV pool | 924,993 tokens |

## Checkpoint

[halt95/Swift1.5-Huihui-Qwen3.8-Flash-Next-W4A16-Merlin](https://huggingface.co/halt95/Swift1.5-Huihui-Qwen3.8-Flash-Next-W4A16-Merlin) is the
[Swift 1.5](https://huggingface.co/halt95/Swift1.5-Qwen3.8-Flash-Next-W4A16-Merlin) W4A16 checkpoint (UkisAI's fine-tune, in the Merlin
format) with [huihui-ai's abliteration](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated) carried onto 96 dense modules:
`self_attn.o_proj` (12 layers) and `mlp.shared_expert.down_proj` (48 layers) in BF16, and `linear_attn.out_proj` (36 GDN layers) re-packed to
INT8 per channel; 132 stored tensors, listed in the checkpoint's `EDITED_TENSORS.txt`. The routed experts, the MTP draft head, the PLE table, the
tokenizer (Qwen's upstream `tokenizer.json`, as the Swift 1.5 W4A16 checkpoint ships it since 2026-10-06) and every configuration file are
Swift 1.5's, byte for byte.

The checkpoint is intended to reduce refusals. The historical [bench card](benchmarks/2026-10-12/Huihui-BENCH-CARD.md) reports the refusal evaluation and its limitations; refusal behaviour has not been re-evaluated here on 0.32.

## Quick start

The generated 0.33 recipe and shared installer are integrated in this checkout. Build and install the engine using the repository-wide [release instructions](../../release/README.md). The dependency lock is complete: 223 exact pins with artifact SHA-256 hashes, resolved with uv; a dry-run `--require-hashes` install resolves. From the repository root:

```bash
venv/bin/hf download halt95/Swift1.5-Huihui-Qwen3.8-Flash-Next-W4A16-Merlin \
  --revision 736e7fb2e95775693f7cc3c1bf525ef0f566818d --local-dir ./ckpt-swift-huihui
(cd ckpt-swift-huihui && sha256sum -c ../models/swift1.5-huihui/checkpoint.sha256)

# The generated launcher inherits CUDA_HOME; select the locked venv toolkit.
export CUDA_HOME="$(venv/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')/nvidia/cu13"
export PATH="$CUDA_HOME/bin:$PATH"
TREE="$PWD" VENV="$PWD/venv" CACHE_ROOT="$PWD/.vllm-cache" \
  bash models/swift1.5-huihui/serve.sh "$PWD/ckpt-swift-huihui"
```

Set `VENV` to the environment in which this checkout was installed. Reference host: NVIDIA driver 610.43.02 (reports CUDA 13.3). The release lock selects a matched CUDA 13.0 toolkit (nvcc, crt, nvvm, nvJitLink 13.0.88, cccl 13.0.85, runtime 13.0.96, the pair torch 2.14.0 cu130 ships with); the 3.0.0 release wheel was served on it by every recipe (2026-10-20); the 3.0.2 wheel was served on it on 2026-11-04 by Flash-Next, Qwen3.8-27B (1, 2 and 4 GPUs), Nemotron 3.5 Super VL and Ornith 1.5, each with a 262K request. Other CUDA 13.x drivers (R580 or later) are expected to work through CUDA minor-version compatibility, but are untested. See the [driver requirements](../../release/install-0.33.md). The launcher requires `EDITED_TENSORS.txt` in the checkpoint directory. Requests use `flash-next-swift-huihui` or `flash-next-abliterated`; `MODEL_NAME` overrides those names.

## Launcher defaults

The layout and engine settings follow [Swift](../swift1.5/README.md#launcher-defaults): 4× RTX 3090 with P2P, TP2 × PP2 + EP (25,23), 262,144-token per-request limit, 8 sequences, FP8 E4M3 KV at 4.1e9 bytes per card, 42 images with an 8 MP cap, MTP K=3 and FGPC off. Thinking defaults to `medium` (`xhigh` per request). The MTP head scores the full vocabulary (`DRAFT_HEAD` default `bf16`); reduced draft vocabulary was removed in 3.0.1, and `REDUCED_DRAFT_VOCAB` or `DRAFT_VOCAB` set in the environment stops the launch (an empty value, or `REDUCED_DRAFT_VOCAB=0`, is accepted).

Huihui uses its own [dense-recalibrated scale sidecar](scales/qsa_kv_scales_swift_huihui_dense.json), distinct served names and checkpoint marker. The [checksum list](checkpoint.sha256) covers its 38 checkpoint files. `SCALES` can override the sidecar; other environment defaults are documented in the [Swift reference](../swift1.5/docs/reference.md#build-and-serve). Select the Huihui profile explicitly; automatic hardware selection continues to choose Flash-Next.

## Known limitations

A boot-and-check run (2026-10-16) and the Merlin 3.0.0 release bench card (2026-10-20) ran this generated launcher on the release wheel. Swift's measured 8 MP margin and inherited heavy results do not establish Huihui-specific quality, refusal, cache, MTP or capacity results on this release. Other hardware and no-P2P operation need separate qualification; the Huihui profile is a P2P-on 4 x RTX 3090 row and preflight refuses it on hosts without P2P (see the [hardware matrix](../../docs/merlin/hardware-matrix.md)). The repository-wide [container route](../../container/README.md) and LXC serving with this recipe are untested here.

## Credits

UkisAI ([Swift 1.5](https://huggingface.co/ukisai/Swift1.5-Qwen3.8-Flash-Next)) and huihui-ai
([the abliteration](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated), method from
[remove-refusals-with-transformers](https://github.com/Sumandora/remove-refusals-with-transformers)) for the weights this checkpoint derives
from; the rest of the credit list is the [Flash-Next one](../flash-next/README.md#credits).

## License

This checkpoint's weights are a derivative of UkisAI's Swift 1.5 and carry the Swift Open License v1.0
([`licenses/Huihui-LICENSE`](licenses/Huihui-LICENSE)) as well as the Qwen Community License 1.0; its KV-scale sidecar
[`scales/qsa_kv_scales_swift_huihui_dense.json`](scales/qsa_kv_scales_swift_huihui_dense.json) is model-derived data under the same licences,
not Apache-2.0. The Swift Open License does not cover commercial use by an entity with US$1,000,000 or more in yearly gross revenue (with its
affiliates); UkisAI offers a separate licence (§§1 and 5). Provenance and the upstream notices: [`licenses/Huihui-NOTICE`](licenses/Huihui-NOTICE).
The launcher in this directory is Apache-2.0 (the repository's [LICENSE](../../LICENSE)).
