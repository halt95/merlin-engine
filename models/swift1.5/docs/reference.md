# Swift 1.5 reference · Merlin 3.0.1

This reference describes the generated 0.32 recipe. See the [README](../README.md#qualification-status) for the qualification boundary and [changelog](../CHANGELOG.md) for earlier releases. The current engine base is vLLM 0.32.0; the generated launcher passed its qualification run (2026-10-16) and the 2026-10-20 release bench card (after the 2026-10-17 pre-release bench run), and is tested on the reference host.

## Build and serve

Build this repository's engine using the [release instructions](../../../release/README.md), then follow the [quick start](../README.md#quick-start). The generated `models/swift1.5/serve.sh` is included in this checkout. There is no separate Swift engine clone.

The generator source is `tools/merlin/registry.py`, with profile `fn-swift1.5-quad-3090` in `tools/merlin/profiles/flash-next.json`. Change the profile and regenerate rather than editing `serve.sh` by hand. Select this profile explicitly with `tools/merlin/preflight.py --profile fn-swift1.5-quad-3090`; automatic hardware selection continues to choose Flash-Next.

| Variable | Generated default or meaning |
|---|---|
| `TREE` | Repository root, used as `PYTHONPATH` |
| `VENV` | `<repository>/venv`; must contain this checkout's installed engine |
| `CACHE_ROOT` | `<repository>/.vllm-cache`; use a fresh cache for a changed engine |
| `SCALES` | `<repository>/models/swift1.5/scales/kv_scales-swift-e4m3.json` |
| `MODEL_NAME` | `flash-next-v2 flash-next flash-mtp flash-next-mtp` |
| `HOST` / `PORT` | `127.0.0.1` / `8000` |
| `KV_BYTES` | `4100000000` per card, passed through `--kv-cache-memory-bytes` |
| `IMG_CAP_PX` | `8388608` pixels per image |
| `REDUCED_DRAFT_VOCAB`, `DRAFT_VOCAB` | Removed in 3.0.1: setting either stops the launch (an empty value, or `REDUCED_DRAFT_VOCAB=0`, is accepted) |
| `WARM` | `1` enables the generated kernel warmup configuration; `0` disables it |
| `GUARD` | `warn` unless inherited; validated as `raise` or `warn` |
| `COUNTERS` | `0`; `1` enables logging-only exact/clip counters |
| `PLE_HOME` | `0` |
| `CUDA_VISIBLE_DEVICES` | `0,1,2,3` |

The checkpoint directory is the required first argument. Extra vLLM arguments come last. The launcher sets thinking on at `medium`; a later `--default-chat-template-kwargs` can override that default. FGPC stays off; do not add `--enable-mamba-shared-prefix-checkpoint` to this launcher.

## Container

The repository-wide [container recipe](../../../container/README.md) has its own qualification check. No Swift container build or GPU-serving result is claimed here. The old per-package image and installer remain historical material in the [v2.5.1-swift1.5 tag](https://github.com/halt95/Swift1.5-qwen38-flash-next-3090s/tree/v2.5.1-swift1.5).

## Check it's working

Confirm that the generated launcher reports the Swift profile (qualified by the launcher qualification, 2026-10-16). Check that the log names Swift's scale sidecar and `medium`, then wait for startup and send a real generation as shown in [Talk to it](../README.md#talk-to-it). Record the actual KV pool; the launcher qualification measured 924,993 tokens on the generated launcher.

The launcher qualification exercised the generated launcher's boot, pool, vision and GSM8K-20 paths; the 2026-10-20 release bench card then ran it on the assembled tree.

## Troubleshooting

- An error requesting `ENGINE=` or `scripts/serve-v2.5.sh` identifies an archived launcher. Use this checkout's generated `models/swift1.5/serve.sh`.
- A missing scale sidecar requires checking `SCALES`; a relative override is relative to the current working directory.
- A missing `config.json` or checksum failure requires checking the downloaded checkpoint and pinned revision.
- `REDUCED_DRAFT_VOCAB` or `DRAFT_VOCAB` set in the environment stops the launch (removed in 3.0.1; an empty value, or
  `REDUCED_DRAFT_VOCAB=0`, is accepted); unset them.
- Empty answer content with `finish_reason: "length"` may mean the output cap ended inside reasoning; increase the cap or lower/disable reasoning per request.

## Hardware

The generated profile targets four RTX 3090 24 GB (sm_86), P2P enabled, TP2 × PP2 + EP with layer partition 25,23. It does not qualify other hardware, no-P2P operation or topology auto-selection. Budget about 115 GiB for the checkpoint; use the repository's preflight for host RAM, shared-memory and build-product checks.

## The KV budget

The profile pins 4.1e9 bytes of FP8 E4M3 KV per card, uses 4,096-token blocks and a 64-token prefix-match unit, and admits up to 8 sequences with a 262,144-token per-request limit. **924,993 tokens was measured on the generated launcher (2026-10-16); three 262K or six 131K resident sessions are inherited targets.**

## Stack

| Piece | Generated 0.32 recipe |
|---|---|
| Engine | Merlin 3.0.1 at repository root; vLLM 0.32.0 base |
| MTP | K=3, full-vocabulary head (`DRAFT_HEAD`, default bf16); reduced draft vocabulary removed in 3.0.1 |
| KV scales | [Swift-specific sidecar](../scales/kv_scales-swift-e4m3.json) |
| Caching | Prefix caching, align mode, RecoverSSM; FGPC off |
| Images/tools | Up to 42 images, 8,388,608-pixel default per image; Qwen3 coder and reasoning parsers |

## Vision evidence

On 2026-10-15, Swift at **8 MP** on the 4× RTX 3090 / 0.32 reference setup measured **902 MiB minimum free memory on stage 0**; the generated launcher's 2026-10-16 8 MP burst after 8K prefill left **544 MiB**, 20 MiB above the 524 MiB floor. These are the measured image-cap margins. It does not by itself qualify every generated-launcher path or a different image cap, topology or hardware setup.

[Huihui](../../swift1.5-huihui/README.md#qualification-status) inherits Swift's heavy results; its generated launcher independently measured 902 MiB minimum free at 8 MP (2026-10-16).

## Checkpoint


[halt95/Swift1.5-Qwen3.8-Flash-Next-W4A16-Merlin](https://huggingface.co/halt95/Swift1.5-Qwen3.8-Flash-Next-W4A16-Merlin)
is the Merlin checkpoint with the 300 tensors Swift 1.5 changed swapped in. We compared Swift 1.5 (revision
`0bd4fe22431372cdad1979267d3ab45aa7e6150a`) with Qwen/Qwen3.8-Flash-Next (revision
`de4b8e4d43b917e7706784d8bb445c9af86a3540`) tensor by tensor: changes were found in the attention `q/k/v/o_proj`
(12 layers), the GDN `in_proj_qkv` / `in_proj_z` / `out_proj` (36 layers) and the shared experts (48 layers), 5.42 GiB in
BF16. The routed experts, routers, MTP head, PLE table, embeddings, `lm_head` and vision tower were reported unchanged
(the MTP head with a full SHA-256 of every tensor; the other groups with whole-shard or sampled hashes, three 512 KiB
chunks per large tensor; a sampled match is strong evidence but not proof). Attention and shared experts stay in BF16;
the GDN projections are re-packed to INT8 per-channel symmetric in the Merlin checkpoint's format. The routed experts
(Intel AutoRound INT4 g128), MTP head and FP8 PLE table are the Merlin checkpoint's, unmodified, and the checkpoint
carries the `ple_embedding_dtype` config key the engine needs.

The KV scale sidecar was recalibrated on this checkpoint (`scales/kv_scales-swift-e4m3.json`, sha256 `d211be1f…`, the
same bytes as `qsa_kv_scales_swift.json` in the checkpoint). Swift 1.5 changed `k_proj` / `v_proj`, so the Merlin
sidecar does not carry over: use each checkpoint's own recipe and scales.

The checkpoint's own `tokenizer.json` on Hugging Face, the file Swift 1.5 and the Intel AutoRound release ship, is
replaced with the upstream Qwen3.8-Flash-Next file (vocabulary, merges and special tokens identical), so tools that
read it directly (the `tokenizers` library, GGUF converters) match upstream too.

## Benchmarks

The [2026-10-06 bench card](../benchmarks/2026-10-06/BENCH-CARD.md) and [2026-09-27 loop check](loop-check.md) describe earlier engine releases. Their prefill, decode, quality, pool and follow-up figures remain historical vLLM 0.30 evidence. They do not certify Merlin 3.0.0 or the generated 0.32 launcher.

The launcher qualification (2026-10-16) measured boot 443 s, GSM8K-20 19/20 and ITL 4K / 32K 18.78 / 18.84 ms for the generated launcher; the [2026-10-20 release bench card](../benchmarks/2026-10-20/BENCH-CARD.md) measured decode 177 / 180 / 177 tok/s at 4K / 131K / 262K and GSM8K-200 197/200, and the earlier pre-release bench run (2026-10-17) decode 156.9 / 159.7 / 147.2 tok/s at 32K / 131K / 258K and GSM8K 198/200 (see the [README](../README.md#qualification-status)). No session-residency result is asserted here. The 8 MP memory result above is maintainer-reported. No hardware run accompanied this update.

## Credit

Qwen, UkisAI, Intel AutoRound, RadixArk, DominikBucko and the Merlin/vLLM contributors supplied the model and serving work. See the [README credits](../README.md#credits) and the dated benchmark records for provenance.

## Licence detail


- **Code** in this recipe and the engine at the repository root is under Apache-2.0 (the repository's `LICENSE`).
- **The Swift 1.5 KV-scale sidecar** (`scales/kv_scales-swift-e4m3.json`) is model-derived data, not under Apache-2.0.
  It is subject to the Swift Open License v1.0 (`licenses/LICENSE-SWIFT`) and the Qwen Community License 1.0 (`licenses/LICENSE-QWEN`).
- **The checkpoint you serve**
  ([halt95/Swift1.5-Qwen3.8-Flash-Next-W4A16-Merlin](https://huggingface.co/halt95/Swift1.5-Qwen3.8-Flash-Next-W4A16-Merlin))
  is subject to the same two licences; it is not part of this repository.
  - Commercial Use by a Legal Entity whose gross revenue meets or exceeds US$1,000,000 is not licensed under the
    Swift Open License. Revenue is measured over the most recently completed fiscal year across that entity and all
    entities controlling, controlled by, or under common control with it. A Qualified Non-Profit Organization's
    Non-Commercial or Research Purposes are exempt from the threshold. An entity above the threshold may obtain a
    separate written Swift Enterprise License from UkisAI. See Swift Open License §§1 and 5.
  - If the Software or a derivative is used for a licensee's commercial product or service with more than
    100,000,000 monthly active users or more than US$20,000,000 monthly revenue, the respective model name must be
    prominently displayed in that product's or service's user interface. If the licensee or an affiliate conducts a
    Model as a Service or AI Work Assistant business, it must obtain a separate Qwen licence before any commercial
    use, except for qualifying internal use that makes neither the software, its outputs, nor its model capabilities
    available to a third party. See Qwen Community License §§1–2.
- "UkisAI" and "Swift" are UkisAI's names. They are used here only to describe where the weights come from.
- This section summarises the licences. It is not legal advice, and the licence texts govern.

This is an independent derivative made by halt95. It is not made, reviewed or endorsed by UkisAI or by Qwen.
