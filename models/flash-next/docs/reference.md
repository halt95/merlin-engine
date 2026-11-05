# Reference

The detail behind the [README](../README.md) for Merlin 3.0.2, Merlin engine **built on vLLM 0.33.0**.
The current measured claims are in the [3.0.2 bench card](../benchmarks/2026-11-04/BENCH-CARD.md) (the [3.0.1 card](../benchmarks/2026-11-02/BENCH-CARD.md) has the previous release's); see also [Benchmarks](#benchmarks). Historical measurements are explicitly marked
**v3.0.0 (0.30) figure**; their original release labels are retained, and they are not re-measured 0.32 or 0.33 results.

- [Overview](#overview)
- [Build and serve](#build-and-serve)
- [Merlin 3.0.2 options](#merlin-302-options), [Merlin 3.0.1 options](#merlin-301-options)
- [Container](#container) and [Proxmox LXC](#proxmox-lxc)
- [Check it's working](#check-its-working), [Troubleshooting](#troubleshooting)
- [Multi-GPU hosts and topology (opt-in)](#multi-gpu-hosts-and-topology-opt-in), [Running notes](#running-notes)
- [How it works](#how-it-works): [Hardware](#hardware), [The KV budget](#the-kv-budget),
  [Host-resident embeddings](#host-resident-embeddings-inside-the-cudagraph), [Stack](#stack),
  [Checkpoint](#checkpoint), [Serving an agent](#serving-an-agent)
- [Benchmarks](#benchmarks), [Known behaviours](#known-behaviours-of-the-qwen38-flash-next-architecture-in-vllm)
- [What ships next](#what-ships-next), [Credit](#credit)

## Overview

Qwen3.8-Flash-Next is served at its native 262,144-token context limit on four RTX 3090s:
TP2 × PP2 with expert parallel, MTP K=3 by default, and a measured **924,993-token FP8 KV pool**.
The host-mapped PLE table and host-resident token embeddings leave VRAM for model state and the KV cache.
The Merlin 3.0.2 engine is built on vLLM 0.33.0; its source lives at the repository root.

## Build and serve

**The shared 0.33 build and install routes are integrated.**
Use [release/README.md](../../../release/README.md). The dependency lock is complete
with 223 exact pins and artifact SHA-256 hashes; its uv dry-run hash-checked install resolves.
`release/install-env.sh` installs this checkout editable, with the compiled engine from
the release wheel or built from source. Historical bundles and binary assets do not
qualify this engine.

| Path | Purpose |
|---|---|
| `release/` | Shared installer, environment pins and release tooling; see its current status above |
| `models/flash-next/serve.sh` | Canonical 0.33 launcher; requires a verified build-product marker |
| `models/flash-next/make-e1-config.py` | Adds the checkpoint's PLE dtype key if missing |
| `models/flash-next/make-ple-int6-pack.py` | Builds the INT6 PLE pack from the FP8 table or, with `--from-bf16` (needs NumPy), from the BF16 one; a dry run unless `--build`. Setup downloads the published pack, so most users never run it |
| `models/flash-next/scales/qsa_kv_scales_262k.json` | Shipped FP8 KV-scale sidecar |
| `models/flash-next/lxc/` | Imported Proxmox deployment recipe; build integration unqualified on 0.33 |

After the installer has produced a verified build and environment, run from the repository root:

```bash
TREE="$PWD" VENV="$PWD/venv" CACHE_ROOT="$PWD/.vllm-cache" \
  models/flash-next/serve.sh /path/to/Qwen3.8-Flash-Next-W4A16-Merlin
```

The launcher defaults to the engine checkout, `venv/` and `.vllm-cache/` at the repository root.
It verifies build-product hashes before starting. A shadowing `vllm/` checkout in an unrelated working
directory can override the intended import; use the launcher's working-directory checks.

Keep the default `HOST=127.0.0.1` behind a proxy, or set `VLLM_API_KEY` when using `HOST=0.0.0.0`.
Extra arguments after the checkpoint go to `vllm serve`.

**Requirements and reproduction boundary.** The old Python, CUDA, driver, RAM, disk, startup and no-P2P
qualification numbers are **v3.0.0 (0.30) figure** guidance, not 0.33 install qualification.
Use the shared install documentation and `release/requirements-0.33.lock` for the current toolchain and pins. In particular, the old
compiled-ops reuse claim is not carried forward. The supplied current measurements use the reference
4×3090 host with working P2P; no no-P2P throughput result was supplied for 0.32 or 0.33.

## Merlin 3.0.2 options

Everything under 3.0.1 options below still applies. New in 3.0.2:

| Option | Default | What it does |
|---|---|---|
| `--ple-store ram\|disk` (`VLLM_E2_PLE_STORE`) | `ram` | **Experimental.** `disk` reads the PLE table's rows from the checkpoint's files. `PLE_WARM` (default `1`; the launcher sets the engine's `VLLM_E2_PLE_WARM` from it) controls the background warm-up read; `VLLM_E2_PLE_DISK_THREADS` (default 32) the reader threads. Costs and limits: [How to serve](../../../docs/how-to-serve.md#the-ple-table-on-nvme-experimental). |
| `--context SIZE` | native 262,144 | **Experimental.** See [Context size](#context-size-302). |
| `VLLM_RECOVERSSM_PP_SPLIT_RECV=1` | off | Rank 0 waits only for the token counts before it commits the deferred RecoverSSM step; the drafts are waited for afterwards. PP layouts only. Bit-identical by construction; effect on decode not yet measured. |
| `VLLM_RECOVERSSM_COMMIT_AFTER_PROPOSE=1` | off (the launcher does not set it) | On the last pipeline rank the commit runs after the drafter has proposed. 3.0.2 extends it to the last rank of a PP layout; the other ranks stay refused. |
| `VLLM_MERLIN_PP_DECODE_BALANCE=1` | off | **Experimental.** Caps decode rows per step at `ceil(decoding / pp_size)`; tokens are unchanged. Untested on 0.33; the A/B has not been done for 3.0.2. |
| `VLLM_MERLIN_DECODE_FIRST=1` | off | **Experimental.** Reserves running decoders' tokens before a prefill chunk is sized, so a session admitted after a long cold prefill keeps decoding while it runs. Pair with `-- --prefill-schedule-interval N` for a decode-only duty cycle. Not proven token-identical (chunk boundaries move). |

Always on, with no switch: commit kernels are warmed on every pipeline rank at boot, duplicate align-postprocess launches in
one commit chain are dropped, and the deferred commit uses a persistent snapshot ring of `pp_size + 1` slots that reuses its storage instead of
allocating it each step. None of the three changes the tokens by construction (the on-versus-off comparison has not been done for 3.0.2).

The lookahead block hashes added upstream in 0.33 are pinned off (`disable_lookahead_block_hashes=True`), so prefix hashes and
hit lengths are the 3.0.1 ones; DFlash and DSpark keep dropping the trailing block, as in 3.0.1. See
[Held back from 0.33](../../../docs/CHANGELOG.md#held-back-from-033).

## Merlin 3.0.1 options

These are opt-ins; the default is MTP with the host-RAM KV cache off.

Launcher knobs new in 3.0.1: `SEQS` (sessions at once; the CUDA-graph capture follows it), `MAXLEN` (context
length), `PP_PARTITION` (the pipeline split) and `MERLIN_MARLIN_MOE_CFG` (passed through). A value off the tested
row prints a note. `SPEC=none` turns speculation off for diagnosis only; it is not a serving lane.

### DFlash2

Use `merlin serve flash-next <checkpoint-dir> --spec dflash2`, or
`./setup.sh --model flash-next --spec dflash2` to download and verify the drafter before serving.
The drafter is `incoai/Qwen3.8-Flash-Next-DFlash2`, about 1.2 GB, checked against
[`dflash2.sha256`](../dflash2.sha256). Serving uses `DFLASH2_MODEL` when set, otherwise
`~/models/flash-next-dflash2`, and refuses to start if the drafter is missing.
For a direct recipe launch, set `SPEC=dflash2 DFLASH2_MODEL=/path/to/drafter`.

Only the three 4 x 3090 P2P rows have qualified DFlash2 geometry: `fn-quad-3090`,
`fn-swift1.5-quad-3090` and `fn-swift1.5-huihui-quad-3090`. Select the corresponding checkpoint name for a
variant.
Other rows and the 27B, Super VL and Ornith families refuse DFlash2. The qualified geometry uses a
3,468,165,120-byte KV pin per rank instead of the MTP pin. Decode measured +5–18 % versus MTP on
4 x 3090, but the 789,668-token pool holds 3 x 262K sessions, not 6 x 131K.

### Host-RAM KV cache

`merlin serve flash-next <checkpoint-dir> --kv-ram-cache 64g` enables the host-RAM tier for evicted
prefix KV blocks; the same flag is accepted by `./setup.sh`. It restores a matching prefix from RAM
instead of recomputing it. It adds no GPU context or simultaneous-session capacity, does not change the session cap
(`merlin serve --swarm` does: 24 sessions instead of 8 on the 4 x 3090 profiles, with the cache at the tested 48g), and
never uses disk.

SIZE is whole GiB per server, split across TP x PP ranks: `64g` on TP2 x PP2 is 16 GiB per rank
(`MERLIN_KV_HOST_TIER_GIB=16` for a direct recipe launch). A size that does not split evenly is
rounded down to whole GiB per rank; a size that cannot hold one 262K restore unit per rank is refused (on
4 x 3090 the smallest is `8g`, 2 GiB per rank; [SETUP](../../../docs/SETUP.md) lists the other layouts). With `--instances N`,
each server gets SIZE. `--kv-ram-cache 0` disables the tier. **On the Flash-Next family the tier is
experimental** (the server logs a warning when it is on): a session evicted while other
requests are prefilling heavily may recompute instead of restoring from the cache (output identical);
a fix is planned. It runs with the MTP drafter only: `--kv-ram-cache` with `--spec dflash2`
is refused in 3.0.1 (with DFlash2 the cache restored nothing on the reference host).

On the reference 4 x RTX 3090 host a 262K restore took 1.2–1.4 s, against about 45 s to recompute. Host
memory grows in steps over the first 262K requests while the request history fills (bounded: about 70 MiB).
These are measured results for the reference layout, not an increase in the GPU pool or a promise for every
session length and layout.

Add the cache allocation to the server's host-RAM requirement. The tier locks its memory and refuses
to start if it cannot: use `LimitMEMLOCK=infinity` under systemd or `--ulimit memlock=-1` with Docker.
See [host-RAM sizing](../../../docs/SETUP.md#flash-next-host-ram-ple-table-and-kv-ram-cache)
for the table and worker memory budget.

#### KV RAM cache tuning knobs

The defaults are the served configuration; these environment variables exist for testing and diagnosis, and apply
to every family that runs the cache (Flash-Next, the 27B, Ornith). Each needs the cache on (`MERLIN_KV_HOST_TIER_GIB`
> 0, which `--kv-ram-cache` sets): they combine with `--kv-ram-cache SIZE` and are refused with `--kv-ram-cache 0`
or at boot without the cache. `merlin serve` checks the values before the boot;
blank means unset.

| Variable | Default | Meaning |
| --- | --- | --- |
| `MERLIN_KV_HOST_TIER__COPY_CREDITS` | `32` | Restore (host to GPU) rows in flight per rank; also the default for the copy credits below. Positive integer. |
| `MERLIN_KV_HOST_TIER__COPY_CREDITS_D` | `COPY_CREDITS` | Copy (GPU to host) rows in flight per rank. Positive integer. Running-session copies take at most a quarter of it (at least one row). |
| `MERLIN_KV_HOST_TIER__COPY_BYTES_D_MIB` | `0` | Optional cap on copy bytes in flight per rank, in MiB; every copy counts, running-session copies included. `0` = rows only; a nonzero cap below one row is refused at boot. |
| `MERLIN_KV_HOST_TIER__MAX_PENDING_STORE_BLOCKS` | `0` | Experimental, off by default. Optional cap on the distinct GPU blocks the copy path holds at once (held, staged and in flight, including blocks a request still references). `0` = off. A value below the largest restore unit stops that unit's copies, so it is recomputed on its return; size it from the measured unit geometry. Untested on a box. |
| `MERLIN_KV_HOST_TIER__FALLBACK_BOUNDARY` | `0` | Experimental, off by default. `1` keeps a session's previous restore point (its predecessor unit) on the host behind its newest one, so host eviction takes the newest unit's own rows first and the session still restores at the previous boundary instead of recomputing from zero. No copy and no new publication: the kept rows were already published. It costs one recurrent endpoint and sliding window per live session on the host. Exactly `0` or `1`; needs the tier on. Untested on a box. |
| `MERLIN_KV_HOST_TIER__N_RESTORE` | `2` | Restores in flight at once. A request that finds them all busy waits while they progress (see below). Positive integer. |
| `MERLIN_KV_HOST_TIER__MAX_ASSEMBLING` | `2` | Units copied at once whose landed rows are held until the unit is complete; a new unit waits while this many are in progress. Positive integer. |
| `MERLIN_KV_HOST_TIER__ESCAPE_STEPS` | `8` | Waits without progress after which an admission held for a pending copy goes ahead anyway (the copy is then missed). Positive integer. |
| `MERLIN_KV_HOST_TIER__ADMISSION` | off | `off`, `0` or `lru`: plain LRU. `slru` (experimental): new units enter probation, a host restore promotes a unit to protected (at most 80 % of published rows), and a copy for a session that is not returning never displaces protected content. |
| `MERLIN_KV_HOST_TIER_CENSUS` | `0` | `1` turns on the debug census (a per-step occupancy scan); off on the served path. |
| `VLLM_RECOVERSSM_CHECK_OWNERSHIP` | `0` | Debug only: `1` checks RecoverSSM state ownership, with a host sync per commit. |
| `MERLIN_KV_HOST_TIER_FAIL_LOADS`, `MERLIN_KV_HOST_TIER_FAIL_STORES` | `0` | Debug only: the first N restore / copy transfers per rank fail, to drive the recompute path. Never serve with them set. |

Fixed rules, not knobs: a request waiting for a restore slot recomputes its host part only when no restore has
progressed for 4 x the measured batch latency, on both clocks: at least 8 scheduler steps and at least 2 s (before the
first measurement the latency is the larger of 0.5 s and the last batch issued at 2 GiB/s, and 0.5 s before any batch
was issued). Running-session copies (the lowest-priority lane) run
only when no other copy, restore or admission is waiting.

## Engine defaults

Engine switches that are on by default in every launch, for every family. The defaults are the served
configuration; change them only to diagnose a problem.

| knob | default | what it does |
|---|---|---|
| `VLLM_MERLIN_VISION_CHUNK_ROWS` | `8192` | Runs the vision encoder's per-row operations in chunks of this many rows on large images, to lower the memory peak; `0` turns chunking off. |
| `VLLM_MERLIN_VISION_CHUNK_MIN_PIXELS` | `8388608` | Smallest image (in pixels) that is chunked; values below 8 MP are raised to 8 MP. |
| `VLLM_MERLIN_PROMPT_LOGPROBS_CHUNK_SIZE` | `256` | Prompt tokens scored per chunk for `prompt_logprobs` requests. |
| `VLLM_SPEC_PROFILE_SHAPES` | on | Runs the drafter's MoE shapes during startup memory profiling; `0` skips them. |
| `VLLM_GDN_BA_SPLITK` | on | Split-K GEMM for the GDN `in_proj_ba`; `0` off. |
| `VLLM_DRAFT_SEQ_LENS_NOSYNC` | on | No device sync for the second MTP draft step; `0` off. |
| `VLLM_E2_GUARD_TRANSPORT` | `mapped` | How the bounds-guard counters reach the host: `mapped` (page-locked host memory, checked at every step) or `copy`. |
| `VLLM_PLE_PROGRESS_DEADLINE_S` | `300` | Seconds a host-embedding (PLE) transaction may take while serving before the engine reports a progress-deadline failure; greater than 0, up to 86,400. |

`VLLM_E2_MEMSTATS` (default `0`, off) is debug-only allocator telemetry: a poll interval in seconds, written under
`VLLM_E2_MEMSTATS_DIR`. Never serve with it set.

## Context size (3.0.2)

`merlin serve flash-next /path/to/checkpoint --context 512k` serves 524,288
tokens using YaRN factor 2. Plain integers are tokens; `k`/`K` mean 1,024
and `m`/`M` mean 1,048,576 on **--context**. The displayed 393K/524K/917K
row names below are rounded labels: use their exact token counts.
The equivalent bare `-- --max-model-len 524288` now adds auto-YaRN too.
Extra vLLM arguments retain vLLM's case-sensitive units (`512k` = 512,000,
`512K` = 524,288). Native/default serving adds no YaRN.

The launcher selects the smallest covering factor from 1.5, 2, 4 and reads
the original window, theta, partial rotary and M-RoPE fields from the
checkpoint's `config.json`. It never guesses those fields. The generated
override sets `text_config.max_position_embeddings = native * factor` and
preserves the checkpoint's rotary fields with `rope_type=yarn`, `factor`
and `original_max_position_embeddings=native`. For a flat config the same
keys live at the root. It enables the existing
`VLLM_ALLOW_LONG_MAX_MODEL_LEN` only for the generated recipe, because a
tokenizer can still advertise the native limit. An explicit
`--hf-overrides` (including dotted forms) always wins; checkpoint-defined
scaling is also preserved.

`./setup.sh --context 512k` saves the choice per model. Interactive setup
offers native 262K, 393,216, 524,288 and 1,048,576 with recipe test status;
Enter and `--yes` keep native unless a saved/explicit choice says otherwise.
Before downloading, setup checks the selected layout's pool. Recipe
construction waits for the downloaded config at serve time.

At the current Flash-Next quad pins the reference pools are 924,993 (MTP)
and 789,668 (DFlash2). A request limit above the resolved pool is refused.
The estimate follows a changed KV pin in whole reference blocks, prints
resident full-context sessions and the scheduler cap, and never counts
the KV RAM cache as extra request context. The engine's boot capacity
remains authoritative; allocator capacity can change with context length.
Custom geometry needs a matching profile. A 1M context does not fit either
current quad pin.

Recipe evidence, for the generated override: 393,216 / factor 1.5 passed
MTP and DFlash2; 524,288 / factor 2 passed MTP. Factor 4 with MTP reached
917,504 (7/8 needles over two runs). DFlash2 at factor 2 passed only with
`max_position_embeddings` left as the checkpoint has it, a different
override, so the generated recipe is **⚫ untested** there. DFlash2 factor 4,
27B YaRN, other checkpoint/layout combinations (the eight-card rows
included), and static recipes remain **⚫ untested** on this release until
their gate passes. A recipe test label is not
qualification of a custom checkpoint or different hardware. The drafter
retains native RoPE, so acceptance can drop beyond native; FP8 KV scales
were not calibrated under YaRN.

### Context-size test plan

The rows below are not yet run in full, which is why the context recipes beyond those tested are labelled untested. To
qualify a row, pin the final source commit, checkpoint hashes,
profile, GPU IDs, driver, kernel and boot state, KV pin and complete resolved
argv and environment, then run Flash-Next on the 4x3090 with **MTP and DFlash2**:

| Row | Served limit | Target RoPE | Required measurement |
|---|---:|---|---|
| **262K native control** | 262,144 | native, no YaRN | 261,888 prompt + 128 output; needles 15/55 percent, T=0 identity vs base |
| 393K | 393,216 | factor 1.5 | needles 15/55/85 percent, including positions past native |
| 524K | 524,288 | factor 2 | needles 15/55/85 percent; published recipe |
| ~917K | 917,504 | factor 4 | needles 15/55/85 percent, at least three repeats, preserve every answer |
| 262K under YaRN | 524,288 | factor 2 | 261,888 + 128 control inside extended boot |
| ~500K under factor 4 | 524,288 | factor 4, explicit override | separates factor effects from long-depth effects |
| Static variant 2 | 524,288 | factor 2, original 262,144, mpe 524,288 | compare against rope-only variant 1 |
| Static variant 3 | 524,288 | factor 2, original 524,288, mpe 524,288 | numerical/quality comparison; do not silently normalize its reference window |

Measure actual token counts after templating/tokenization, exact needle
answers, time to first token, valid decode tokens/elapsed decode time,
draft attempts/accepted tokens/per-position acceptance, preemptions and
resident concurrency. Do not infer token throughput from SSE chunk counts.
Record READY time, actual KV pool, target/draft table rows, draft length
raise and the selected YaRN correction window. Include a cold ~900K +
one 8MP image factor-4 row and a 524K factor-2 + 8MP row, recording image
answer correctness and per-rank minimum free memory (524 MiB floor).
Keep generation plus image tokens inside the served limit.

Hash every file in the compile and Triton caches under `CACHE_ROOT` before
the first boot and again after the run, including after any forced crash,
and record the changed file count. A changed file is a finding.

With the current DFlash2 pin, the 917,504 row must be refused during
preflight: record that refusal as such, never as a retrieval pass. A
successful DFlash2 917K leg requires a separately approved/qualified
larger pool; the launcher does not raise the pin. Likewise the advertised
1,048,576 choice should refuse at both default quad pins.

For each supported recipe send an over-limit prompt and a prompt whose
requested output exceeds the remaining budget to chat and completions:
require HTTP 400, then a successful small generation and health check.
Verify no out-of-table request reaches a worker. Try a missing
`original_max_position_embeddings`: require a named YaRN configuration
error before serving, not a bare KeyError. Every successful leg requires
no traceback, CUDA error, Xid or worker death, drafting beyond native,
and saved retrieval/identity evidence. Report all misses; the historical
7/8 factor-4 result is not an all-pass bar. Keep 27B YaRN untested until a
27B test covers native 262K plus extended retrieval,
acceptance, table coverage and image headroom on its own layout.

## Container

The shared [container recipe](../../../container/README.md) builds this checkout through
the 0.33 installer. The route and complete 223-pin SHA-256 dependency lock are
integrated; on the 3.0.0 release (vLLM 0.32) a fresh install (2026-10-21) and a from-source build (2026-10-22) passed on the reference host, and the 3.0.2 fresh install has not been done; the container image is not yet qualified.

## Proxmox LXC

The imported recipe under `models/flash-next/lxc/` describes a privileged Debian LXC with the four cards
passed through. `lxc/pve-create.sh` creates the container; `lxc/provision.sh` installs userspace and the service;
`lxc/flash-next.service` starts serving, and `lxc/nvidia-prestart.sh` recreates the host device nodes at boot.
The device nodes must exist on the Proxmox host before the container starts; NVIDIA userspace in the
container must match the host module.

This provisioning route uses the shared 0.33 installer and verifies reused environments;
it has not been re-qualified on hardware with the final release tree.
The old allocation and timing measurements are **v3.0.0 (0.30) figure** guidance.
For current installation status, see [Build and serve](#build-and-serve).

## Check it's working

Wait for `Application startup complete` and verify `GPU KV cache size: 924,993 tokens` with the shipped pin.
Use a real generation to check health; `/v1/models` alone can answer even when the engine has died.

```bash
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"flash-next","messages":[{"role":"user","content":"ok"}],"max_tokens":16,"chat_template_kwargs":{"enable_thinking":false}}'
```

Add `Authorization: Bearer <key>` if `VLLM_API_KEY` is set.
First requests can compile kernels. The old readiness and warmup timings have not been re-measured on 0.32.

## Troubleshooting

- CUDA out-of-memory or a boot free-memory refusal: check for other GPU processes or a display server.
  A smaller `--kv-cache-memory-bytes` value can free memory, but changes capacity and needs qualification.
- A host OOM-killer event can indicate insufficient RAM for the host-resident tables.
  The legacy resident-floor estimate is not a new 0.32 measurement; see [Hardware](#hardware).
- `Bus error` at boot can indicate undersized `/dev/shm`.
- For import failures, check whether a stray `vllm/` directory shadows the intended engine checkout.
- `Unsupported .version` from PTX: compare the first-request compiler toolkit with the driver.
- A profile-guard refusal: check compilation mode, breakable graphs and other overrides against `serve.sh`.
- A missing `/dev/nvidia-uvm` at LXC start: recreate device nodes on the host and check the pre-start hook.
- `/v1/models` answers but generations fail: read the log from the first named guard, PLE or worker-exit cause.
- Empty content with `finish_reason: "length"`: thinking may have consumed the output budget.
  Raise `max_tokens` or send `"chat_template_kwargs":{"enable_thinking":false}`.
- Empty content with `finish_reason: "stop"` and zero tokens: retry once; see the historical observation below.

## Multi-GPU hosts and topology (opt-in)

Qwen4Exp sequence parallelism stays default-off. When effective SP is enabled, startup warns about rounding differences;
independent topology checks still refuse SP with pipeline parallelism. The warning and checks are in
[`Qwen4ExpForConditionalGenerationConfig.verify_and_update_config`](../../../vllm/model_executor/models/config.py),
with test coverage in the engine's development tree.

By default `serve.sh` uses the settings the release was qualified with: GPUs 0-3 in PCI bus order, vLLM's P2P
check skipped, `NCCL_P2P_LEVEL=SYS`, `NCCL_PROTO=Simple`, FlashInfer's PCIe IPC all-reduce for decode
(`VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=1`) and vLLM's custom all-reduce off. On a host that does not look like the reference one, set
`AUTO_TOPO=1`: serve then runs `models/flash-next/topo_select.py`, which reads the GPU topology from NVML and `/sys` and sets the
GPU order and the P2P and NCCL variables from it, falling back to the fixed settings on any doubt.

- **NVLink pairs**: the selector reorders the four GPUs so that each NVLink-bridged pair is a tensor-parallel pair,
  even when the bridged cards are not next to each other in bus order. This rule is derived from the topology NVML
  reports and covered by unit tests on a synthetic bridged topology; it has not been run on NVLink-bridged hardware.
- **More than four GPUs**: pick four by UUID (`nvidia-smi -L`), ideally on one NUMA node,
  `CUDA_VISIBLE_DEVICES=GPU-aaaa...,GPU-bbbb...,GPU-cccc...,GPU-dddd...` (format placeholders; substitute actual UUIDs).
  UUIDs do not depend on enumeration order.
  This works with the selector off too; with `AUTO_TOPO=1` it also checks the order and the P2P settings.

![Pairing tensor-parallel ranks with NVLink bridges on four GPUs bridged 0-2 and 1-3: in the default order 0,1,2,3 the tensor-parallel pairs are (0,1) and (2,3) and their all-reduce goes over PCIe; with CUDA_VISIBLE_DEVICES=0,2,1,3 or AUTO_TOPO=1 the pairs are (0,2) and (1,3) and each all-reduce goes over its NVLink bridge](images/flashnext-nvlink-pairing.png)

[docs/topology-selector.md](topology-selector.md) has the decision rules, `AUTO_TOPO=strict` and the other controls.

## Running notes

Use a release-specific environment and the installer-selected toolchain.
The launcher clears inherited fork knobs and exports its profile explicitly.
`GUARD` takes precedence over an inherited `VLLM_E2_GUARD_MODE`, then defaults to `warn`;
the current launcher accepts `raise|warn` and refuses `off`.
`VLLM_E2_GUARD_INTERVAL` must be finite and positive. The reader parks after the engine drains.
Idle parking is built in on vLLM 0.32 and 0.33; `VLLM_E2_GUARD_IDLE` is no longer registered, so setting it does not disable parking.
See [`guard_telemetry.py`](../../../vllm/v1/worker/guard_telemetry.py) (`enable_idle_gate`, `_park`) and
[`envs.py`](../../../vllm/envs.py). Mentions of this switch in older bench cards describe those historical engines.

The PLE offload process deliberately refuses a CUDA context. Its blocked-CUDA-initialization messages are
expected. A named bounds-guard or transport failure is a reason to inspect the first error in the log.
The old allocator-warning counts, minimum free-memory observations and step-cost estimates are not
current measurements; retained historical detail is in the [release notes](../CHANGELOG.md).

## How it works

TP2 pairs form two pipeline stages. `VLLM_PP_LAYER_PARTITION=25,23` assigns layers to those stages.
Stage 0 owns the vision tower and target embedding; stage 1 owns the MTP drafter and vocabulary head.
The PLE table stays in host RAM, with rows pulled over PCIe, and token embeddings are host-resident. With `--ple-store disk` (experimental) the table is read from the checkpoint's files on NVMe instead; see [How to serve](../../../docs/how-to-serve.md#the-ple-table-on-nvme-experimental).

![Flash-Next on four RTX 3090s, Merlin 3.0.0: TP2 x PP2 + EP, 25 and 23 layers per pipeline stage, host-resident PLE and embedding tables](images/flashnext-merlin-3.0.0-layout.png)

The diagram shows the launcher's partition and the 924,993-token pool of the 2026-10-15 card; the retired v2.5 diagram's
host-memory sizes were 0.30 figures and are not carried over. See [The KV budget](#the-kv-budget) for the measured pool.

### Hardware

The supplied measurements use the reference 4× RTX 3090 host with PCIe peer-to-peer.
Legacy provisioning used 96 GB host RAM and at least 1 GB `/dev/shm`.
The checkpoint is approximately 115 GiB; allow additional disk space for the environment, build products and caches.
Those allocations, the approximately 69 GiB resident floor and the old 220 W power-cap qualification are
**v3.0.0 (0.30) figure** guidance; a 0.32 host-memory floor, boot peak and power measurement were not supplied.
No new hardware qualification is implied for no-P2P or NVLink configurations.

### The KV budget

The measured pool on 0.32 is **924,993 tokens** (3.0.2 on 0.33: 924,993 tokens). The launcher pins
`--kv-cache-memory-bytes 4100000000` per rank with `--block-size 4096`.
The per-request context limit is 262,144 tokens. The pool is aggregate capacity, and hybrid state and
scheduling overhead also consume it.

The pre-release vLLM 0.32 run measured **six 126,976-token sessions**, with **0 preemption**, 574.5 tok/s aggregate,
10.78 ms/tok per agent and cold prefill 5,423 tok/s. Later release-candidate and merged-source checks matched its
T=0 tokens and logprobs; they did not rerun throughput. See the [measurement summary](../benchmarks/2026-10-15/BENCH-CARD.md).
The old three-262K and six-131K residency, no-MTP pool size, higher-pin ceilings and full-pool pressure
results are **v3.0.0 (0.30) figure** history in the [CHANGELOG](../CHANGELOG.md);
they have not been re-measured on 0.32 in the supplied evidence.

The current **8 MP per-image cap** (`IMG_CAP_PX=8388608`) at the shipped pin measured
**902 MiB stage-0 minimum free memory**. The engine uses the 1x M-RoPE cache automatically.
That headroom result does not qualify every image-count and concurrency combination.
Larger caps and a reduced-pin configuration have no new qualification here.

### Host-resident embeddings inside the cudagraph

`VLLM_HOST_EMBED_TABLE=1` keeps token embeddings in pinned host memory. Device-mapped lookups can replay
inside the draft cudagraph, unlike a CPU gather. The flag affects the compiled graph.
Old per-rank allocation sizes, VRAM savings and decode-neutrality comparisons are
**v3.0.0 (0.30) figure** history, not re-measured performance claims.
The implementation history is in [history.md](history.md).

### Stack

| Piece | Merlin 3.0.2 |
|---|---|
| Engine | Merlin, built on vLLM 0.33.0; source at the repository root |
| Environment and compiled ops | Shared 0.33 installer and engine wheel integrated; `release/requirements-0.33.lock` is complete with 223 exact pins and SHA-256 hashes (torch 2.14.0, Triton 3.8.0, FlashInfer 0.7.2.post1); a fresh install (2026-10-21) and a from-source build (2026-10-22) passed on the reference host on the 3.0.0 release (vLLM 0.32); the 3.0.2 fresh install has not been done |
| Release dependency facts | humming-kernels 0.1.19; instanttensor ≥ 0.2.2 floor |
| Parallel shape | TP2 × PP2 + EP; layer partition 25,23 |
| Speculation | MTP K=3, probabilistic; full-vocabulary draft head, precision `DRAFT_HEAD=bf16\|int8\|int4` (default `bf16`); reduced draft vocabulary removed in 3.0.1 (`REDUCED_DRAFT_VOCAB` / `DRAFT_VOCAB` stop the launch; an empty value, or `REDUCED_DRAFT_VOCAB=0`, is accepted) |
| Context and scheduling | 262,144 tokens per request; up to 8 sequences; 1,024-token prefill chunk |
| KV | 4.1e9 bytes per rank, FP8 e4m3, shipped scale sidecar; block size 4,096 |
| Prefix caching | On (`--enable-prefix-caching`; the launcher also passes the deprecated `--mamba-cache-mode align`, which has no effect on 0.33), match unit 64; **FGPC off** (omit `--enable-mamba-shared-prefix-checkpoint`) |
| Vision | Up to 42 images per request; 8 MP per image; no video |
| Compilation | FULL_AND_PIECEWISE, captures through 32; breakable graphs off; V2 runner and RecoverSSM |
| Guard | `warn` by default; `raise` accepted; `off` refused by the launcher |
| Warm start | `WARM=1` by default; `WARM=0` opts out; no new 0.33 warmup-time claim |

These are launcher/configuration facts, distinct from measured performance.
[Earlier stacks](history.md#v201-stack) retain their original release scope.

### Checkpoint

The checkpoint remains [halt95/Qwen3.8-Flash-Next-W4A16-Merlin](https://huggingface.co/halt95/Qwen3.8-Flash-Next-W4A16-Merlin).
It needs `ple_embedding_dtype: float8_e4m3fn` inside `text_config` in `config.json`.
For an older download, `models/flash-next/make-e1-config.py` adds the key without changing weights.
The shipped sidecar is `scales/qsa_kv_scales_262k.json`. A Flash-Next-family checkpoint that ships its own scales as
`qsa_kv_scales.json` beside its weights is served with them (the launcher prints a note); `SCALES=<path>` overrides
both. Serve such a variant with `merlin serve flash-next <checkpoint-dir>`. Reduced draft vocabulary was removed in Merlin 3.0.1: the MTP drafter scores the full vocabulary, and `REDUCED_DRAFT_VOCAB` or `DRAFT_VOCAB` set in the environment stops the launch (an empty value, or `REDUCED_DRAFT_VOCAB=0`, is accepted).
Checkpoint checksum lists are in `models/flash-next/checkpoint.sha256`.

### Serving an agent

| Need | Current configuration or measured result |
|---|---|
| Several long sessions | Concurrent decode: six 127K sessions, 0 preemption |
| Re-sent context | Prefix caching on, FGPC off; MTP-off 32K cache hit 0.998 and TTFT 5.78 → 0.315 s |
| Tools and thinking | Qwen3 coder tool parser, Qwen3 reasoning parser; thinking on at medium effort |
| Vision | 8 MP cap; 902 MiB measured stage-0 minimum free |
| Health | Check an actual generation and read the first named failure in the log |

The cache-hit row uses MTP off, unlike the launcher's default. It does not re-establish the old
multi-conversation follow-up reuse percentages under MTP.

## Benchmarks

The [2026-11-02 bench card](../benchmarks/2026-11-02/BENCH-CARD.md) has the Merlin 3.0.1 results. The
[2026-10-15 0.32 summary](../benchmarks/2026-10-15/BENCH-CARD.md) records maintainer-supplied rc-build concurrent decode, cache-hit,
image-cap and pool results; its raw streams, boot logs and a full release-check card are not included.

The [2026-10-09 v3.0.0 card](../benchmarks/2026-10-09/BENCH-CARD.md) is historical:
**v3.0.0 (0.30) figure** throughout, including its v2.5.1 comparator and optional YaRN entry.
Single-stream throughput, GSM8K, needle, teacher-divergence, idle power, boot timing, multi-turn reuse and
fault campaigns from that era are not current 0.32 qualification.
Older cards remain linked from the [CHANGELOG](../CHANGELOG.md) and [history](history.md).

## Known behaviours of the Qwen3.8-Flash-Next architecture in vLLM

These observations and mitigations were carried from the older serving path.
Their incidence and qualification results are **v3.0.0 (0.30) figure** history;
no new rates, fault campaign or fresh-compile determinism claim are established here.

- Keep a custom KV pool larger than the per-request context limit plus scheduling overhead.
- An empty warm completion with `finish_reason: "stop"` and zero tokens was observed with cached prefixes.
  Retrying once helped in the historical runs; it is not a guarantee.
- Earlier hybrid-cache versions dropped finished sessions' prefixes while other sessions decoded.
  The later retention fixes and their original results are described in the
  [v2.5.1 notes](../CHANGELOG.md#what-changed-in-v251). The current MTP-off cache-hit measurement does not
  cover every full-pool, multi-turn workload.
- Bounds guards and PLE ownership checks report named failures. Older illegal-address incidents and
  containment tests do not establish absence of faults in this release.
- Greedy output can depend on GPU order, compile cache, layout and kernel build. Earlier within-cache and
  fresh-cache identity counts have not been re-measured for this refresh.
- Old thinking-off short-context comparisons and draft-acceptance differences remain historical results.

## What ships next

Merlin 3.0.2 ships the 0.33 launcher, 8 MP cap, FGPC off and the defaults listed in [Stack](#stack).
RecoverSSM metadata reuse and warm-start controls remain configured; the draft head scores the full vocabulary (`DRAFT_HEAD` sets its precision).
The old 34 MiB INT4 reduced-vocabulary draft-head cost and 22.5–33.0 W idle-power measurements are
**v3.0.0 (0.30) figure** results, not re-measured on 0.32 or 0.33.
A fresh install of the 3.0.2 wheel, container qualification and larger image caps require their own qualification;
`--context` (YaRN) is described under [Context size](#context-size-302), with most combinations untested. Local-argmax remains opt-in; no new performance result is claimed for it here.

## Credit

- Model: [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
- vLLM model support and PLE offload: the [`peakcrosser7` Flash-Next branch](https://github.com/peakcrosser7/vllm/commits/release/qwen38next_offload)
  behind the [model-support](https://github.com/vllm-project/vllm/pull/53896) and [PLE-offload](https://github.com/vllm-project/vllm/pull/53899) contributions
  (the latter closed without merging; upstream took a [UVA PLE offload](https://github.com/vllm-project/vllm/pull/54371) instead, which is in `v0.30.0`);
  in the v2.0.1 base and re-ported since v2.2.0: pipeline changes by saichowdary007;
  carried ahead of the base as five upstream-authored prefix-cache patches: work by ZeldaHuang, yewentao256,
  akshaver, tobymao and lucamotz. The [v2.0.1 structured-output fixes](history.md#what-changed-in-v201)
  and other carried changes retain their upstream authorship. The tree works around the
  [PP>1 refusal for PLE checkpoints](https://github.com/vllm-project/vllm/issues/54709).
  v2.2.0 builds on public vLLM `v0.30.0`, including eastwood-c's model-support work; its stage-ownership
  pattern is applied to this model. Re-ported changes also credit woosebastian, amittell, ArcheyChen,
  Prudctual, Karl0007, semerandre and wzhao18. v2.5.1 adds ten more upstream ports, including
  [RecoverSSM](https://github.com/vllm-project/vllm/pull/58863), the GDN state recovery for speculative
  decoding behind the larger pool. Original authorship remains in upstream history and `NOTICE`.
- Reports and contributions: a community idle-power report and contribution, brandon-nuttall (YaRN past 262K report, `SEQS`/`MAXLEN` launcher fix, local-argmax mixin), R-omk (YaRN past 262K crash report and recipes), mgabor3141 (club-3090 [discussion](https://github.com/noonghunna/club-3090/discussions/1572))
- Quantised experts: [Intel/Qwen3.8-Flash-Next-W4A16-AutoRound](https://huggingface.co/Intel/Qwen3.8-Flash-Next-W4A16-AutoRound);
  FP8 PLE table: [RadixArk/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4);
  MTP INT4 packing recipe adapted from [DominikBucko/qwen38-flash-next-2x3090](https://github.com/DominikBucko/qwen38-flash-next-2x3090);
  P2P on consumer Ampere: aikitoria's open-kernel-module patch; bench basis from 3.0.1: Zagitta's [gpu-unlock](https://github.com/Zagitta/gpu-unlock) kernel module (its NVIDIA firmware is personal-use and not shipped; [Sudzpel](https://github.com/Sudzpel), who proposed it with the first measurements on Merlin)
- Independent work on the same hardware class: Neiko2002,
  [noonghunna/club-3090](https://github.com/noonghunna/club-3090), [tfriedel/qwen3.6-rtx3090-lab](https://github.com/tfriedel/qwen3.6-rtx3090-lab)
