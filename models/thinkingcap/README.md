# ThinkingCap-Qwen3.8-27B on 2× or 4× RTX 3090

Both wrappers serve 27B's
[`froggeric-qwen-v22.5.jinja`](../qwen38-27b/templates/froggeric-qwen-v22.5.jinja), the template every Qwen
recipe ships. It accepts system messages after the first turn, so coding agents' per-turn notes stay in place
and the prefix cache keeps hitting. `CHAT_TEMPLATE=<path>` serves another template; `CHAT_TEMPLATE=` (empty)
serves the checkpoint's own.

**Serve BottleCap AI's ThinkingCap fine-tune of Qwen3.8-27B (our W4A16 build) at the full 262K context on two or four RTX 3090s,
with vision, tool calls and MTP speculative decoding, behind one OpenAI-compatible endpoint.**

[![GPU](https://img.shields.io/badge/GPU-2x_or_4x_RTX_3090-76B900?logo=nvidia&logoColor=white)](../qwen38-27b/docs/reference.md#hardware)
[![Context](https://img.shields.io/badge/context-262K_per_request-ffb000)](../qwen38-27b/docs/reference.md#the-kv-budget)
[![ThinkingCap](https://img.shields.io/badge/%F0%9F%A4%97_checkpoint-ThinkingCap--Qwen3.8--27B--W4A16--Merlin-ffd21e)](https://huggingface.co/halt95/ThinkingCap-Qwen3.8-27B-W4A16-Merlin)

> **Recipe 3.0.0 / Merlin 3.0.2:** **vLLM 0.33.0**. ThinkingCap is a profile of the
> [Qwen3.8-27B recipe](../qwen38-27b/README.md). Its wrappers forward to the TP2/TP4 base launchers.
> Both use **FGPC ON** (`--enable-mamba-shared-prefix-checkpoint`), pools **826,689 / 2,012,105 tokens** (TP4 = RecoverSSM on; ThinkingCap TP4 check PASS, 32/32 checks),
> and the **8 MP (8,388,608 pixel)** release cap. The integrated launchers default to this cap.
> The FlashInfer **0.7.2.post1** tile patch (a port of the measured 0.7.0.post1 one) is packaged in the shared installer; the dependency lock is complete with 223 exact pins and SHA-256 hashes.

The [2026-10-15 base-checkpoint card](../qwen38-27b/benchmarks/2026-10-15/BENCH-CARD.md) records TP2 ITL at
4K / 32K / 131K of **19.99 / 21.41 / 25.27 ms**, TP4 **15.83 / 16.62 / 18.37 ms**,
and GSM8K-200 tile on mean **192.0/200** against v1.3.0's **196/200** (−2.0 pp,
95 % Newcombe CI **[−5.89, +1.61] pp**, within CI and not significantly worse).
27B TP2 serves images up to 8 MP. With the vision-peak option on (`VLLM_MERLIN_VISION_PEAK=1`, the TP2 launcher default), the 50-item TP2 stress sweep's worst minimum free was 806 MiB per card (2026-10-15), above the 524 MiB floor. A separate knob OFF/ON identity check matched tokens and logprobs on 6/6 prompts. This was measured on the Qwen3.8-27B checkpoint (TP2); on the ThinkingCap checkpoint itself (TP2, 2026-10-16 release mini check), four ~8 MP images after an 8K prefill left a minimum of 806 MiB free per card, sampled every 100 ms, above the 524 MiB floor. Flash-Next (902 MiB) meets the target. 27B TP4 left 690 MiB free on the retired RecoverSSM-off pin; at the shipped chunk 4096 with RecoverSSM on, the 2026-10-20 bench card measured 862 MiB lowest free (one 8 MP image, two at once, and four in one prompt; 2 cards: 886 MiB), above the 524 MiB floor.
These are base-checkpoint results. ThinkingCap's own decode, prefill, agents and GSM8K at medium were measured for 3.0.1 on
2026-11-01 to 2026-11-02 (below); its **token savings have not been remeasured on 0.32**, and the table and the 2026-10-12 card keep historical **vLLM 0.30** measurements.

This directory holds the ThinkingCap launchers, the checkpoint checksum list, the licences and the bench card. The checkpoint is
[halt95/ThinkingCap-Qwen3.8-27B-W4A16-Merlin](https://huggingface.co/halt95/ThinkingCap-Qwen3.8-27B-W4A16-Merlin): BottleCap AI's
reasoning fine-tune of Qwen3.8-27B, which reasons in fewer tokens, quantised by us to W4A16 in the same format as the base checkpoint
(an AWQ int4 body made with philbert's recipe unchanged, plus the base release's calibrated FP8 E4M3 KV scales, int4 `lm_head` and int4 MTP
drafter reused byte for byte; `tokenizer.json` is the upstream file; 11 files, about 17 GB). It is an independent quantised derivative:
it is **not** published, reviewed or endorsed by BottleCap AI or Qwen.

| | Qwen3.8-27B | ThinkingCap |
|---|---|---|
| Default reasoning effort | `medium` | `medium` (was `xhigh` before 2026-10-21; `xhigh` per request) |
| Serve scripts (dual / quad card) | [`../qwen38-27b/serve.sh`](../qwen38-27b/serve.sh) / [`serve-tp4.sh`](../qwen38-27b/serve-tp4.sh) | [`serve.sh`](serve.sh) / [`serve-tp4.sh`](serve-tp4.sh) |
| Checkpoint list | [`checkpoint.sha256`](../qwen38-27b/checkpoint.sha256) | [`checkpoint.sha256`](checkpoint.sha256) |
| Weights licence | Apache-2.0 | PolyForm Small Business 1.0.0 plus BottleCap AI's personal-use permission ([License](#license)) |
| Historical 0.30 numbers | single-stream thinking decode at 32K: 198 tok/s dual card, 249 quad card | x0.860 of the base's median thinking tokens at medium effort at the same GSM8K accuracy (97.27 %, 1,319 items); 172.8 tok/s at 32K on dual card (2026-10-08 check) and 223.4 on quad card (2026-10-12) |

Both checkpoints run on the same engine, pins, port (8100) and served name (`qwen38-27b`). The 3.0.2 and 3.0.1 measurements are in the
[2026-11-02 card](benchmarks/2026-11-02/BENCH-CARD.md), and the older ones, dated per row, in the [2026-10-12 card](benchmarks/2026-10-12/ThinkingCap-BENCH-CARD.md); the profile's detail (what differs, token cost by effort,
capacity rows) is in [ThinkingCap checkpoint profile](../qwen38-27b/docs/reference.md#thinkingcap-checkpoint-profile).

## Merlin 3.0.2

This checkpoint was not benchmarked for 3.0.2. It was not run on the 3.0.2 wheel before release. The 3.0.1 bench card below is the latest measurement and was taken on vLLM 0.32. There is no 3.0.2 card.

## Merlin 3.0.1 bench card (2026-11-02)

Maintainer measurements on 4× RTX 3090 (220 W, PCIe Gen4) with P2P (the aikitoria patched driver) and the Zagitta [gpu-unlock](https://github.com/Zagitta/gpu-unlock) kernel module, one fresh `merlin serve` boot per configuration, 2026-11-01 to 2026-11-02; method and every cell in the [bench card](benchmarks/2026-11-02/BENCH-CARD.md). Decode is one stream with thinking on at reasoning effort medium, the server default (tok/s after the first streamed token); prefill is one cold request. With thinking on, a 512-token agent follow-up may end inside the reasoning, and reasoning tokens count. GSM8K at medium ran on four cards. Thinking-off GSM8K is from the 3.0.0 runs (two-card: a 2026-10-17 pre-release build). The agent overlap windows on four cards are under 5 s: read those totals loosely. Rows marked (3.0.0 run) were not re-run for 3.0.1; they and the rest of the 3.0.0 figures are on the [3.0.0 bench card](benchmarks/2026-10-20/BENCH-CARD.md).

| Measurement | 2× RTX 3090 (TP2) | 4× RTX 3090 (TP4) |
|---|---|---|
| Decode, 4K / 131K / 262K prompt | 203 / 162 / 160 tok/s | 244 / 230 / 238 tok/s |
| Cold prefill, 10K / 100K / 262K prompt | 2,319 / 1,679 / 1,114 tok/s | 3,869 / 2,990 / 2,099 tok/s |
| Agents at once (8 × 16K conversations), total, thinking on / off | 697 / 633 tok/s | 952 / 832 tok/s |
| GSM8K-200 (5-shot), correct of 200, thinking on at medium / thinking off | – / 193 | 198 / 195 |
| Sessions held at once (3.0.0 run) | 3 × 262K and 6 × 131K, every session resident (re-sent prompts ≥ 99.98 % cached), 0 preemptions | 7 × 262K and 14 × 131K, every session resident (re-sent prompts ≥ 99.98 % cached), 0 preemptions |
| 8 MP image (3.0.0 run) | one, two at once and four in one prompt (32,842 tokens) correct; lowest free memory 886 MiB on any card | one, two at once and four in one prompt (32,842 tokens) correct; lowest free memory 862 MiB on any card |

## Quick start

Install the shared engine and its environment as described in the
[Qwen3.8-27B Quick start](../qwen38-27b/README.md#quick-start). The shared installer and 0.7.2.post1 tile
are integrated; the archived 0.30 bundle does not install 3.0.0. Then, from the repository root:

```bash
HF_HUB_DISABLE_TELEMETRY=1 venv/bin/hf download halt95/ThinkingCap-Qwen3.8-27B-W4A16-Merlin --local-dir ./ckpt-thinkingcap
(cd ckpt-thinkingcap && sha256sum -c ../models/thinkingcap/checkpoint.sha256)   # the 11 checkpoint files
MAX_PIXELS=8388608 MODEL=$PWD/ckpt-thinkingcap models/thinkingcap/serve.sh        # dual card, TP=2
MAX_PIXELS=8388608 MODEL=$PWD/ckpt-thinkingcap models/thinkingcap/serve-tp4.sh    # quad card, TP=4
```

Both wrappers set `THINKING` to `{"enable_thinking": true, "reasoning_effort": "medium"}` unless you set it yourself, then run the 27B
launcher with every other variable passed through (`PORT`, `HOST`, `KV_BYTES`, `SPEC_K`, `MAX_PIXELS`, ...). A request can still choose its
own effort with `"chat_template_kwargs": {"reasoning_effort": "xhigh"}`, and
`THINKING='{"enable_thinking": true, "reasoning_effort": "xhigh"}'` makes xhigh the server default.

## Known limitations

- The historical 0.30 ThinkingCap token saving was measured on arithmetic word problems only.
- No ThinkingCap token-savings result is claimed on 0.33 beyond the 3.0.2 card. The 27B TP2 8 MP headroom result above applies, since ThinkingCap forwards to the 27B launchers.
- Everything in the [Qwen3.8-27B known limitations](../qwen38-27b/README.md#known-limitations) applies.

## Credits

BottleCap AI ([ThinkingCap-Qwen3.8-27B](https://huggingface.co/bottlecapai/ThinkingCap-Qwen3.8-27B); this profile serves a quantised
derivative of it, not affiliated with or endorsed by BottleCap AI), Qwen ([Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)), philbert
([AWQ int4 base checkpoint](https://huggingface.co/philbert440/Qwen3.8-27B-W4A16-AWQ)); the rest of the credit list is the
[Qwen3.8-27B one](../qwen38-27b/README.md#credits).

## License

Code in this directory (the two launchers, the checksum list): Apache-2.0 (the repository's [LICENSE](../../LICENSE)). Model weights are
not part of this repository. The ThinkingCap checkpoint `halt95/ThinkingCap-Qwen3.8-27B-W4A16-Merlin` is not Apache-2.0:

**Licence of the ThinkingCap weights.** ThinkingCap is licensed by BottleCap AI under the
[PolyForm Small Business License 1.0.0](https://polyformproject.org/licenses/small-business/1.0.0), plus BottleCap AI's
additional permission for personal, non-commercial use by an individual, free of charge. A business may use it only if it
qualifies as a "Small Business" under the PolyForm text (fewer than 100 employees and independent contractors in
total, and under US$1 million in prior-year revenue, in 2019 dollars adjusted for inflation); any other commercial use needs a separate licence from BottleCap AI. **There is no
unrestricted commercial use.** Required Notice: Copyright 2026 BottleCap AI (https://bottlecapai.com). These terms cover
the ThinkingCap weights on Hugging Face, which carry the licence files; they do not cover this repository's code, which
stays Apache-2.0. Copies of BottleCap AI's `LICENSE` and `NOTICE` and the PolyForm text are in [`licenses/`](licenses/) (Qwen's
Apache-2.0 text is [`../qwen38-27b/licenses/LICENSE-Apache-2.0-Qwen.txt`](../qwen38-27b/licenses/LICENSE-Apache-2.0-Qwen.txt)); the changes made to
the checkpoint are listed in the repository's `NOTICE`. This is a summary, not legal advice.
