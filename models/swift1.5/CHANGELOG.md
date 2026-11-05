# Changelog

> Recipe release notes, followed by historical package releases. Earlier tags stay in
> [halt95/Swift1.5-qwen38-flash-next-3090s](https://github.com/halt95/Swift1.5-qwen38-flash-next-3090s).

Release notes, newest first. To install and run the current release, see the [README](README.md); reference detail
(scripts, environment variables, capacity, known behaviours, measurement method) is in
[docs/reference.md](docs/reference.md).

## Merlin 3.0.2

The Swift recipe runs on **vLLM 0.33.0** in Merlin 3.0.2; the recipe version stays 3.0.0. See the
[engine changelog](../../docs/CHANGELOG.md#merlin-302). This recipe was not benchmarked for 3.0.2; the latest card is the
[2026-11-02 bench card](benchmarks/2026-11-02/BENCH-CARD.md) (the 3.0.1 card; no 3.0.2 card). Swift1.5 and Swift1.5-Huihui now refuse a
directory whose `config.json` matches but that has neither the release scales file nor the verified marker (pass `SCALES=` or run
`merlin pull`), because `config.json` is identical across Flash-Next, Swift1.5 and Swift1.5-Huihui.

## Merlin 3.0.0

The Swift recipe now belongs to Merlin 3.0.0 in this repository, on the vLLM 0.32 line (currently 0.32.0). The canonical `serve.sh` is generated from the Swift checkpoint profile and starts the engine at the repository root; it does not wrap a separately installed v2.5.1 engine.

- Generated launcher status: **tested on the reference host**: the launcher qualification (2026-10-16) and the [2026-10-20 release bench card](benchmarks/2026-10-20/BENCH-CARD.md).
- Default image cap: **8 MP (8,388,608 pixels)**. Swift measured **902 MiB stage-0 minimum free memory** on 4× RTX 3090 / 0.32 on 2026-10-15; the generated launcher's 2026-10-16 8 MP burst after 8K prefill left 544 MiB, 20 MiB above the 524 MiB floor.
- MTP K=3 uses the full-vocabulary head (`DRAFT_HEAD=bf16|int8|int4`, default bf16); no reduced draft-vocabulary table ships (the switch was removed in 3.0.1). FGPC stays off.
- Huihui inherits Swift's heavy results. The 924,993-token pool was measured on both generated launchers (2026-10-16); session residency is an inherited target; earlier speed and quality figures below are historical, not new 0.32 measurements.
- Current install and serving instructions use the repository checkout; the retired package installer and container commands are kept only in the historical releases.

See the [README](README.md#qualification-status) for the evidence boundary.

## Historical releases

The entries below describe the released vLLM 0.30 packages and their original paths, settings and measurements. They are not current installation instructions or Merlin 3.0.0 qualification.

## v2.5.1-swift1.5

Released 2026-10-06: [v2.5.1-swift1.5 on GitHub](https://github.com/halt95/Swift1.5-qwen38-flash-next-3090s/releases/tag/v2.5.1-swift1.5).
Bench card: [`benchmarks/2026-10-06/BENCH-CARD.md`](benchmarks/2026-10-06/BENCH-CARD.md). To run it:
[Quick start](README.md#quick-start), or [Build and serve](docs/reference.md#build-and-serve) for the detail.

The same checkpoint on the Flash-Next v2.5.1 engine (v2.5.0 was not published), so these notes cover everything since
v2.2.0-swift1.5.

### In brief

- **Agent follow-up turns resume from cache.** A follow-up resumes on a 64-token grid (`--prefix-match-unit 64`)
  instead of the 4,096-token block, and the cached prefix a conversation needs is held until its next turn. One
  conversation alone: median time to first token on follow-ups 0.96 s (p90 1.005 s), a median of 3,199 uncached tokens
  per follow-up.
- **Eight deep conversations at once.** 93.0 % of follow-up prompt tokens came from cache and the median follow-up
  started in 1.28 s; 90 requests started within 5 minutes, all completed; 3 of 82 follow-ups re-prefilled the whole
  conversation with the pool near full (98.0 %), 0 preemptions.
- **More KV capacity.** The pool grows from 806,792 to 924,993 tokens (+15 %), with an attention block size of 4,096
  tokens.
- **Tokenizer and image fixes.** Transformers 5.18.0 handles combining marks as intended, with parity on 92 items.
  Prompts can contain up to 42 images, each capped at 4 MP.
- **Cold prefill.** 5,206 tok/s at 10K and 5,473 tok/s at 100K tokens; the engine measures v2.5.1 1.0–2.8 % faster
  than v2.2.0 on the Merlin checkpoint (not compared on this one).
- **Stability checks passed.** GSM8K-200 scored 197/200 (thinking off), structured output passed 80/80 cases, the
  131K and 262K needles were found exactly, cache hits were valid on 20 of 20 prompt pairs, and 0 engine errors
  occurred while serving.

Single-stream decode was not re-measured on this checkpoint. The engine's v2.5.1 card measures it about 3 % slower
than v2.2.0 on the Merlin checkpoint (−2.87 %, 95 % CI −6.84 % to +1.09 %; v2.5.0's release candidate −2.81 %); on
v2.2.0 this checkpoint's step time matched the Merlin checkpoint's within run-to-run noise. Bench card:
[`benchmarks/2026-10-06/BENCH-CARD.md`](benchmarks/2026-10-06/BENCH-CARD.md).

### What changed in v2.5.1-swift1.5

**The engine moves to v2.5.1.** `release/install-env.sh` checks out `halt95/qwen38-flash-next-3090s` tag `v2.5.1`
(commit `352e0065bd28686d60d8b13c1d2b16783ade6b2d`) and asserts `scripts/serve-v2.5.sh` (sha256 `22eb5201…`); it
downloads the v2.5.1 delta bundle from the v2.5.1 release and the compiled-ops tarball from the v2.2.0 release (v2.5.1
changes no compiled source; the tarball is reused, pinned by sha256), verifies both against `upstream/PIN-v2.5`, and
runs `scripts/build-v2.5.sh ./vllm-v2.5 ./venv-v2.5`. The engine's changes are in its
[CHANGELOG](https://github.com/halt95/qwen38-flash-next-3090s/blob/v2.5.1/CHANGELOG.md#v251).

**The serve entry wraps the v2.5.1 launcher.** `serve/serve.sh` starts `scripts/serve-v2.5.sh` with this checkpoint's
sidecar and the `xhigh` default, as before. The launcher now passes `--block-size 4096 --prefix-match-unit 64
--use-replayssm` and exports the side-cache layout, Model Runner V2 and checkpoint retention; the wrapper adds nothing
else. Use a new compile cache: v2.2 artefacts do not carry over.

**Combining marks tokenize as intended.** The transformers pin moves to 5.18.0. On v2.5.0, tokenizer output matched
the reference tokenizer on all 92 test strings. The checkpoint's own `tokenizer.json` on Hugging Face, the file Swift
1.5 and the Intel AutoRound release ship, is replaced with the upstream Qwen3.8-Flash-Next file (vocabulary, merges
and special tokens identical), so tools that read it directly (the `tokenizers` library, GGUF converters) match
upstream too; vLLM serving on 5.18.0 tokenizes identically with either file. The weights and every other checkpoint
file are unchanged; `release/checkpoint.sha256` and the download revision follow the new Hugging Face revision.

**Container.** `Dockerfile`, `docker-compose.yml` and `serve/docker-entrypoint.sh` follow the new engine paths and
image tag; the compose cache volume is `swift15-flash-next-cache-v2.5`. The image has not been rebuilt or GPU-served on
v2.5.1.

**Documentation.** A shorter README; the detail moves to `docs/reference.md`; `NOTICE` names the new engine pin and
the tokenizer change.

Upgrading from v2.2.0-swift1.5: install into a fresh `ENGINE=` (or move `engine/` away; `install-env.sh` refuses an
engine at another commit), re-download `tokenizer.json`, `README.md`, `NOTICE` and `SHA256SUMS` from the new revision
and re-run the checksum step.

## v2.2.0-swift1.5

Released 2026-09-28 (measured 2026-09-27). First public release of this repository. There is no earlier public version and no upgrade path.

- Serves `halt95/Swift1.5-Qwen3.8-Flash-Next-W4A16-Merlin` (41 files, checksums in
  `release/checkpoint.sha256`): the Merlin checkpoint with the 300 dense tensors Swift 1.5 changed.
- Engine: Flash-Next v2.2.0, unchanged and not copied here
  (`halt95/qwen38-flash-next-3090s` tag `v2.2.0` = commit `cda592711239fa5061d59ef4858db4e68b217445`).
- Install route: `release/install-env.sh` clones the engine at that commit, asserts it and the launcher's sha256,
  downloads and verifies the two v2.2.0 release assets against the engine's `upstream/PIN-v2.2`, and runs the
  engine's own `scripts/build-v2.2.sh`.
- Serve entry: `serve/serve.sh`, the v2.2.0 launcher (`scripts/serve-v2.2.sh`, sha256 checked before start) with
  `SCALES` set to this checkpoint's sidecar and `xhigh` as the server default reasoning effort.
- KV-scale sidecar: `quant/kv_scales-swift-e4m3.json`, static FP8 K/V scales calibrated on this checkpoint.
- Evidence: bench card `benchmarks/2026-09-27/`, loop check `docs/loop-check.md`.
- Container route (added 2026-09-28, same version): `Dockerfile`, `docker-compose.yml`, `.dockerignore`,
  `serve/docker-entrypoint.sh`. The image runs `release/install-env.sh` and serves through `serve/serve.sh` with
  the checkpoint mounted at `/model`; built and entrypoint-checked on a host without GPUs, not yet GPU-served in a
  container.
- Benchmarks (added 2026-09-28, same version): completion tokens by reasoning effort, the smoke set and a scored
  retrieval to 256K (bench card sections 7 and 8; two rows in the README table).
- Licences: `LICENSE` (Apache-2.0), `LICENSE-SWIFT` (Swift Open License v1.0), `LICENSE-QWEN` (Qwen Community
  License 1.0), `NOTICE`.
