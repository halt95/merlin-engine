# v2.5.1-swift1.5 card — 2026-10-06 (the Swift 1.5 checkpoint on the Flash-Next v2.5.1 engine: release check)

> **Historical vLLM 0.30 evidence.** The results and commands below belong to the named v2.x engine. They do not qualify the generated 0.32 launcher or assert current performance/capacity. Current status and the measured 8 MP result: [Swift README](../../README.md#qualification-status).

> *File names in this record (`serve/serve.sh`, `scripts/...`, `release/...`, `patches/...`) are from the measurement time; current locations are in [models/README.md](../../../README.md).*

> **Provenance.** Maintainer measurements on the reference host (4× RTX 3090, 220 W, PCIe P2P, no NVLink; driver
> 610.43.02). The engine is Flash-Next v2.5.1: vLLM 0.30.0 + the v2.5.1 commits, the published source `96a0d13b6e`, on
> python 3.13.5, torch 2.13.0 (CUDA 13.0), triton 3.7.1, flashinfer 0.6.18.post1, transformers 5.18.0. The checkpoint
> is the reference host's serving copy: the same 27 weight files as the published checkpoint (byte-identical, checked
> 2026-09-27) and the same weight map. KV-scale sidecar: sha256 `d211be1f…`, the bytes of
> `quant/kv_scales-swift-e4m3.json`. `xhigh` as the server's default reasoning effort,
> one boot with an empty compile cache, 2026-10-06.
>
> **Serve command.** The reference host's serving entry for this checkpoint: the flags of `scripts/serve-v2.5.sh`
> (including `--block-size 4096 --prefix-match-unit 64 --use-replayssm`, the side-cache layout, Model Runner V2 and
> checkpoint retention) plus the `xhigh` default, which is what `serve/serve.sh` produces, with three differences in
> the environment: the FP8 KV clip counter was on (`serve/serve.sh` leaves it off unless `COUNTERS=1`; it only logs),
> FlashInfer and Triton compiled with the host's CUDA 13.3 toolkit instead of the venv's CUDA 13.0 wheels, and the
> served model names included two extra aliases. The logging-only exact-counter reader was off, as on the engine's
> v2.5.1 decode-ladder boots. The boot confirmed the pool (924,993 tokens), the block size and prefix-match unit, the
> sidecar applied on every rank and `reasoning_effort: xhigh` among the server defaults.
>
> **Not run on this checkpoint:** the single-stream decode ladder, the depth chart, session residency, the divergence
> measure against the BF16 teacher, the mixed soak and the `xhigh` GSM8K and loop checks. Decode is the engine's, as
> in the [engine's v2.5.1 bench card](https://github.com/halt95/qwen38-flash-next-3090s/blob/v2.5.1/benchmarks/2026-10-06/BENCH-CARD.md)
> (measured on the Merlin checkpoint); on v2.2.0 this checkpoint's step time matched the Merlin checkpoint's within
> run-to-run noise ([`../2026-09-27/BENCH-CARD.md`](https://github.com/halt95/Swift1.5-qwen38-flash-next-3090s/blob/v2.5.1-swift1.5/benchmarks/2026-09-27/BENCH-CARD.md)). Streams are client-timed. The
> drivers, raw streams and boot log are not published (as for the earlier cards).

## 1. Results

The drivers are the engine's v2.5.1 release-gate drivers. The Merlin column is the engine's v2.5.1 bench card, for
reference: separate boots, and its rows ran on the cuts that card names.

| | this checkpoint, v2.5.1 | Merlin checkpoint, v2.5.1 (engine card) |
|---|---|---|
| KV pool, FP8 tokens, `--kv-cache-memory 4100000000` | **924,993** (attention block 4,096) | 924,993 |
| needle, thinking off, temperature 0, cold: 131,008 / 261,568 prompt tokens | found exactly / found exactly | — / found exactly (261,568) |
| structured output, 80 requests over four load cells | **80/80** | 80/80 |
| GSM8K-200, thinking off, temperature 0 | **197/200** | 198/200 |
| cold prefill at 10K / 100K tokens, median of 3 sends | **5,206 / 5,473 tok/s** | 5,219 at 10K; 5,484 at 100K (depth chart) |
| one conversation alone, 26 follow-up turns (36K to 76K): median time to first token (p90) | **0.96 s** (1.005 s) | 0.96 s (1.02 s) |
| same: uncached tokens per follow-up, median | 3,199 | 3,202 |
| cache validity: prompt pairs identical cold and warm, hit at the expected 64-token boundary | **20/20**, 20/20 (images 4/4) | 20/20, 20/20 (images 4/4) |
| 8 concurrent conversations, temperature 0, 300 s: follow-up cache reuse | **93.0 %** | 92.5 % |
| same: median time to first token, follow-up turns | **1.28 s** | 1.61 s |
| same: requests started within 300 s, all completed (of them follow-ups) | 90 (82) | 107 (99) |
| same: follow-ups that re-prefilled the whole conversation | 3 | 4 |
| same: KV pool peak; preemptions | 98.0 %; 0 | 99.2 %; 2 |
| engine errors while serving; request errors | 0; 0 | 0; 0 |

- **Needle:** a code placed in filler text at the given depth, thinking off, temperature 0, sent cold (0 cached
  tokens); the reply's first line had to equal the code.
- **Structured output:** 80 requests with a JSON schema, 40 with the server idle and 40 under load, in four cells
  (three thinking on, at the server's `xhigh` default, one thinking off), at the served sampling settings (temperature
  1.0, top-p 0.95, top-k 20); every reply parsed and validated, 0 non-200 responses.
- **GSM8K-200:** the first 200 GSM8K test questions, 5-shot, thinking off, temperature 0, up to 512 tokens, scored by
  exact match on the final number (one answer hit the 512-token cap).
- **Cold prefill:** 3 sends per depth (10,026–10,032 and 100,036–100,044 prompt tokens), every send asserted
  `cached_tokens == 0`, 1 output token; the range was 5,206–5,217 tok/s at 10K and 5,457–5,481 tok/s at 100K.
- **One conversation:** two sessions run one after the other with no other traffic, each a cold first turn and 13
  follow-ups that append a reply and a new tool result; thinking off, temperature 0, 48-token replies.
- **Cache validity:** twenty prompt pairs, each sent cold and again warm after a shorter turn primed the cache,
  temperature 0, thinking off: the warm send hit at the expected 64-token boundary and gave output identical to the
  cold send's, and the functional answer was correct, 20/20. An image check (the same text with two images in either
  order) answered from the images, 4/4.
- **Eight conversations:** the engine card's multi-turn load: each session is a shared tool prefix, a unique history
  (about 76K tokens at the first turn) and turns that append a reply and a new tool result, growing to 114K; 8
  sessions at once, temperature 0 with a fixed seed, up to 512 generated tokens per turn, 300 s. Reuse is cached over
  prompt tokens, summed over the follow-up turns. The three full re-prefills had 0 cached tokens on prompts of
  96,579–104,836 tokens.
- The load appends the model's own replies, so the two checkpoints did not serve identical conversations; read the
  request count and the reuse as properties of this run, not as a speed comparison between checkpoints.

## 2. An earlier cut

The same gate ran earlier the same day on a pre-release engine build, 2026-10-06, an earlier cut of the v2.5.1 series that differs only in the
waiting-request admission path: GSM8K-200 198/200, prefill 5,225 / 5,463 tok/s, one conversation 0.96 s (p90 0.999 s),
cache validity 20/20, and with 8 conversations 93.2 % reuse, a 1.29 s median follow-up, 93 requests (85 follow-ups),
3 full re-prefills, 1 preemption, 0 errors. One boot each: the two read as the same within run-to-run noise.

## 3. Evidence carried over from v2.2.0-swift1.5

These are properties of the checkpoint measured on the v2.2.0 engine and not re-run on v2.5.1
([`../2026-09-27/BENCH-CARD.md`](https://github.com/halt95/Swift1.5-qwen38-flash-next-3090s/blob/v2.5.1-swift1.5/benchmarks/2026-09-27/BENCH-CARD.md)): step time at four depths and aggregate decode at 8
streams within run-to-run noise of the Merlin checkpoint's, MTP acceptance 2.92 against 2.90, GSM8K-200 at `xhigh`
198/200 against 197, completion tokens at `xhigh` 304 against 416 per answer, the scored retrieval to 256K 3 of 3, and
the repetition-loop check at `xhigh` 0 of 72 ([`../../docs/loop-check.md`](../../docs/loop-check.md)).
