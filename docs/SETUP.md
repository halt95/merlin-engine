# setup.sh: install and serve in one command

```bash
git clone https://github.com/halt95/merlin-engine.git && cd merlin-engine && ./setup.sh
```

`setup.sh` checks the host, asks a few questions (Enter takes the recommended answer, computed from the GPUs and RAM it
finds), shows every download with its size, installs the engine from the prebuilt release wheel (about 1.5 minutes;
nothing system-wide, no sudo), downloads and verifies the checkpoint, and starts the server in the foreground.
Ctrl+C stops it; `./setup.sh` again starts it straight away with the saved answers and asks nothing;
`git pull && ./setup.sh` updates the engine without re-downloading the model.

New to serving? [How to serve](how-to-serve.md) walks through it end to end, from the first run to a networked endpoint.

## What a run looks like

```
Merlin 3.0.1 setup (by Halt95)
  repo     ~/merlin-engine
  host     GPUs 4 x NVIDIA GeForce RTX 3090 (24 GB), P2P on, driver 595.84, RAM 96 GiB

Model  (Enter = recommended)
  1) flash-next         115.3 GiB  Qwen3.8 Flash-Next W4A16 (the reference recipe)  (recommended)
  2) swift1.5           115.3 GiB  Swift 1.5 Flash-Next W4A16 (UkisAI fine-tune)
  3) swift1.5-huihui    115.3 GiB  Swift 1.5 Huihui Flash-Next W4A16 (abliterated)
  4) 27b                 15.9 GiB  Qwen3.8 27B W4A16 (dense; 1, 2 or 4 cards)
  5) 27b-thinkingcap     15.9 GiB  ThinkingCap Qwen3.8 27B W4A16 (BottleCap AI; separate weight
                                   licence)
  6) nemotron-super-vl   67.9 GiB  Nemotron 3.5 Super VL 120B-A12B W4A16 (4 cards)
  7) ornith1.5           66.4 GiB  Ornith 1.5 122B-A10B W4A16 (4 x 24 GB; with or without P2P)
     why flash-next: 4 x GeForce RTX 3090, P2P on, 96 GiB RAM: fn-quad-3090 [tested] fits with the
     qualified RAM

Model [flash-next]:

Layout  (Enter = recommended)
  1) default  fn-quad-3090 [tested] on GPUs 0,1,2,3  (recommended)
     why default: what merlin serve picks

Layout [default]:

Models folder (an existing download is reused) [~/models/flash-next]:

Access
  1) local    this machine only (127.0.0.1)
  2) network  every host that can reach this machine (0.0.0.0), an API key is required

Access [local]:

Port [8000]:

Your answers  (nothing is downloaded or saved yet)
  1) model   flash-next
  2) layout  default layout
  3) folder  ~/models/flash-next
  4) access  local (127.0.0.1)
  5) port    8000

Change an answer? (Enter = no, or 1-5) [no]:

Downloads
  engine wheel                                                  217 MB
    github.com/halt95/merlin-engine release merlin-v3.0.1
  Python packages (222 from release/requirements-0.32.lock)    3.4 GiB
    PyPI via uv (cached after the first install); about 10.2 GiB installed
  checkpoint flash-next                                      115.3 GiB
    Hugging Face halt95/Qwen3.8-Flash-Next-W4A16-Merlin @ cec53e9 -> ~/models/flash-next
    40 files, 115.3 GiB in all, size from Hugging Face API
  --------------------------------------------------------------------
  total                                                      118.9 GiB

Serve
  HOST=127.0.0.1 PORT=8000 ~/merlin-engine/venv/bin/python ~/merlin-engine/merlin_cli.py serve flash-next       ~/models/flash-next

Proceed with the downloads above? (y/n) [y]:
```

Then the engine installs (`release/install-env.sh` with the wheel, then `--verify`), `hf download` fetches the
checkpoint (resumable), every file is checked against `models/<model>/checkpoint.sha256`, `merlin doctor` prints the
hardware report, and the server starts:

```
Endpoint: http://127.0.0.1:8000/v1  (OpenAI-compatible; /v1/messages for /v1/messages clients)
Test (once the log says `Application startup complete`):
  curl -s http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{"model": "flash-next-v2", "messages": [{"role": "user", "content": "Say hello."}], "max_tokens": 32}'
Ctrl+C stops the server; ./setup.sh starts it again without questions.
```

## Host checks (before anything is downloaded)

Every failure prints the exact fix and stops. Nothing is installed system-wide; the fixes are the only lines that
mention `sudo`. The bootstrap itself (uv and Python 3.13 into `.merlin/`, about 60 MB, both checksum-verified) runs
before the Python checks because the checks are written in Python; a `--dry-run` fetches neither and runs on any
`python3` 3.10+ it finds, or says what a real run would fetch.

| Check | Requirement | Fix printed |
|---|---|---|
| platform | Linux x86-64 | (the wheel and the lock are built for it) |
| curl, tar, git, sha256sum | present | `sudo apt install curl tar git coreutils` |
| NVIDIA driver | `nvidia-smi` present, R580 or later (CUDA 13) | `sudo apt install nvidia-driver-580-open` (or the .run installer), reboot |
| GPUs | at least one card with a Merlin layout | `merlin doctor` explains what fits |
| compiler | GCC/G++ >= 13: `CC` / `CXX`, else the remembered pair, else `cc` / `c++`; a `gcc-13` / `g++-13` (or 14, 15) pair on PATH is found by itself and remembered in `.merlin/setup.json` | `sudo apt install gcc-13 g++-13`, then `./setup.sh` again |
| git | a git clone (not a ZIP), tracked tree clean | `git clone`; commit or revert changes |
| host RAM | Flash-Next: the resident need of the PLE table and KV RAM cache the flags ask for (FP8, cache off: 69 GiB floor, 96 GiB qualified; [the table below](#flash-next-host-ram-ple-table-and-kv-ram-cache)) against host RAM (MemTotal, lowered to the cgroup `memory.max`) and, on real hardware, the memory available now (MemAvailable, capped by what the cgroup has left); 27B / Super VL: a warning below 32 GiB | a smaller `--kv-ram-cache`, add RAM, or `--model 27b` |
| disk | per filesystem: the wheel download, uv's package cache (up to 3.4 GiB), the new venv (about 10 GiB), the previous venv while the new one verifies, and the checkpoint's remaining bytes (a resume needs only what is still missing) | free space or `--models-dir` |
| models folder | empty, or this model's checkpoint; a folder holding another checkpoint (by its `config.json` or verified marker) is refused | `--models-dir` another folder |
| `--from-source` only | cargo, cmake, ninja | rustup; `sudo apt install cmake ninja-build` |

## Questions

| Question | Default (Enter) | Flag |
|---|---|---|
| model | recommended from the hardware: Flash-Next on 4 x 24 GB (or 2 x 48 GB) with >= 96 GiB RAM; otherwise the 27B; the reason is shown | `--model flash-next\|swift1.5\|swift1.5-huihui\|27b\|27b-thinkingcap\|nemotron-super-vl\|ornith1.5` |
| layout | what `merlin serve` picks (the largest tested layout that fits), with every other tensor-parallel size and every `--instances N` partition the host allows listed; one exception: a single-card 27B pick on a multi-card class without P2P recommends `--tp 1 --instances N` over all idle cards, with the measured reason (4 x CMP 170HX: TP1 x 4 = 873 tok/s, TP2 x 2 = 523, TP4 = 371; each single-card server takes images up to 4 MP, the multi-card layouts 8 MP) -- this is setup's recommendation only, `merlin serve`'s own default is unchanged | `--tp N`, `--instances N` |
| drafter (Flash-Next family) | asked only on the rows with a qualified DFlash2 geometry (the three 4 x 3090 P2P rows: Flash-Next, Swift1.5, Swift1.5-Huihui): `mtp` (recommended; the bigger pool: 6 x 131K or 3 x 262K) or `dflash2` (+5-18 % decode, 3 x 262K only; downloads the drafter); `--yes`, a piped run and every other row take `mtp` | `--spec mtp\|dflash2` |
| swarm (Flash-Next 4 x 3090 and 27B quad-card rows) | asked only on the rows that have a swarm cap: no (the default session cap, 8 on Flash-Next and 24 on the 27B quad card); yes serves up to 24 / 32 sessions with the KV RAM cache set to 48g (experimental); `--yes`, a piped run and every other row take no | `--swarm` / `--no-swarm` |
| models folder | `~/models/<model>` (with `--ple int6` still the FP8 checkpoint; the INT6 overlay is `<folder>-int6`); an existing folder is reused and skipped when its checksums pass | `--models-dir DIR` |
| access | this machine only (127.0.0.1); network (0.0.0.0) generates an API key, shows it once, and keeps it in `.merlin/api-key` (mode 0600); clients send `Authorization: Bearer <key>` | `--host 127.0.0.1\|0.0.0.0`, `--api-key KEY` |
| port | the model's default: 8000 for Flash-Next / Super VL, 8100 for 27B | `--port P` |

After the last question setup lists the answers. A number asks that question again (another model also asks its own layout, folder and port again); Enter, or `n`, carries on. Answers given by flags are listed but not offered. Nothing is downloaded or saved before the download confirmation, so `n` there, or Ctrl+C at any point before it, leaves nothing behind.

Answers are saved per model: the folder, port and layout of Flash-Next are never applied to 27B. `--model 27b` after a
Flash-Next run takes 27B's own saved answers, or its defaults. The PLE table and the KV RAM cache are flags, never
questions: [Flash-Next host RAM](#flash-next-host-ram-ple-table-and-kv-ram-cache).

## Network access and API-key scope

The generated key (`VLLM_API_KEY` or `--api-key`) protects the route prefixes `/v1`, `/v2`, `/inference`, `/cohere`,
`/generative_scoring`, `/tokenize` and `/detokenize`. It does **not** authenticate `/invocations` (the SageMaker generation
endpoint), or ops routes such as `/metrics`, `/version`, `/health` and `/ping`. A network bind (`HOST=0.0.0.0`, `--host ::`,
or a LAN address) therefore remains reachable through these unguarded routes even when an API key is set. The launcher
warns on a non-loopback bind, including when a key is configured.

Keep the backend on `HOST=127.0.0.1` behind a reverse proxy that authenticates every published route, including
`/invocations` and the ops routes, or restrict network access with a firewall or private network. An API key alone
does not secure the whole listener.

Image and media URLs in requests are fetched as in upstream vLLM: any host, with redirects followed. If untrusted
clients can reach the server, restrict fetches with `--allowed-media-domains`.

On a shared server, upstream also trusts client-supplied `multi_modal_uuids` / content-part `uuid` values as
multimodal cache identities when no processor or media kwargs are set. Reusing a UUID can reuse another request's
cached image embedding; an API key does not provide per-client cache isolation. Give mutually untrusted clients
separate server instances, or have a trusted gateway strip or reject these UUID fields before forwarding requests.

## Flash-Next host RAM: PLE table and KV RAM cache

Flash-Next holds its PLE ("n-gram") embedding table in host RAM, one copy per server. Setup asks nothing about the
table or the KV RAM cache (the answers list shows both, with the flag that changes each): the FP8 table and the cache off are the defaults, and two flags opt in, on `./setup.sh` and
on `merlin serve` alike (so the cache can be turned on without running setup again):

- `--kv-ram-cache SIZE` turns on a host-RAM cache for the prefix KV blocks the GPU pool evicts: a later request that
  shares the prefix gets them restored from RAM instead of recomputed. It does not add context (the GPU KV pool is
  unchanged) and it does not change the session cap (`--swarm`, below, does); it is host RAM only, never disk. SIZE is the host RAM the cache takes per server,
  in whole GiB: `24g`, `24G`, `24gib` or `24`; `0` turns a saved one off. It is split over the server's ranks
  (TP x PP) into the engine knob `MERLIN_KV_HOST_TIER_GIB` (the KV host tier, GiB per rank): `--kv-ram-cache 24g` on
  the four-rank TP2 x PP2 layout is 6 GiB per rank. With `--instances N` every server gets SIZE. A size that does not
  split into whole GiB per rank is rounded down (`--kv-ram-cache 22g` on four ranks is 20g, 5 GiB per rank) and the
  plan says so. The smallest size holds one 262K restore unit on every rank, rounded up to whole GiB per rank; a
  smaller one is refused before anything starts, and the refusal names the smallest size or `0` (off). The smallest
  size per server:

  | Layout | Default KV | sm_80 with `KV_DTYPE=fp8` |
  |---|---|---|
  | Flash-Next family on four cards: RTX 3090 (with or without P2P, PP4 too), RTX 4090 (PP4 too), 32 GB, 48 GB | `8g` | |
  | Flash-Next family, 8 x RTX 3090: TP2 x PP4 / TP4 x PP2 | `8g` / `16g` | |
  | Flash-Next family, 2 x 48 GB PP2 (sm_86, sm_89) | `6g` | |
  | Flash-Next, one 80 GB sm_80 card | `8g` | `4g` |
  | Flash-Next, 2 x 80 GB sm_80: TP2 / PP2 | `8g` | `4g` / `6g` |
  | Flash-Next family, 2 x 64 GB sm_80 PP2 | `8g` | `6g` |
  | Flash-Next family, 3 x 32 GB sm_80 PP3 | `9g` | `6g` |
  | Flash-Next family, four sm_80 cards (24, 32, 40, 64 or 80 GB) | `12g` | `8g` |
  | Qwen3.8-27B and ThinkingCap, one / two / four cards, any hardware | `9g` / `10g` / `12g` | |
  | Ornith 1.5, with or without P2P | `4g` (fallback) | |

  The BF16 KV variant of an FP8 row (a generated `-kv-bf16` launcher) needs more, for example `12g` on 4 x RTX 3090;
  the refusal names it.
  Ornith 1.5 has no recorded block count, so it keeps a `4g` fallback (1 GiB per rank) that is not yet verified to
  hold one restore unit. A direct recipe launch checks `MERLIN_KV_HOST_TIER_GIB` against the same per-rank minimum.
  The plan prints the per-server, per-rank and total sizes.
  It is experimental in 3.0.1 on the Flash-Next family (MTP drafter only) and on the 27B and ThinkingCap (RecoverSSM
  path only); Ornith 1.5 accepts it on its RecoverSSM path, untested ⚫, and the plan says so. Setup saves the flag with the other answers and passes it to
  `merlin serve`; it never exports the knob itself, and `merlin serve --kv-ram-cache` overrides an exported one.
  Capacity is computed from the resolved KV layout and the restore-unit manifest. The tier is one cache with the GPU pool (the allocator demotes a block to RAM before reusing it),
  so it has no mirror floor: there is no minimum size below which it holds nothing, and its capacity is about the
  size you give it, less a few blocks of in-flight transfer credit per rank. What it restores is one restore unit
  per prefix boundary: the full-attention chain (the QSA main and compressed-index groups included) up to 262,144
  tokens, any sliding-window tail, and one recurrent (GDN/PLE) state per recurrent group; the QSA ring slots never
  move. The unit size comes from the resolved KV layout, so it differs per profile (about 4 GiB over four ranks on
  `fn-swift1.5-quad-3090`; the engine logs the exact figure). The cache needs room for at least one 262K restore
  unit. Sizes below the minimum above are refused. Sizes above 160g are untested ⚫ and the plan says so.
  The cache locks its memory and fails closed when it cannot (the refusal says so): under systemd set
  `LimitMEMLOCK=infinity`; in a container run `ulimit -l unlimited`, or use a privileged LXC or CAP_IPC_LOCK.
- `--swarm` (experimental) serves the row's tested high-concurrency cap instead of its default: up to 24 sessions on the
  Flash-Next 4 x 3090 profiles (default 8), 32 on the 27B quad card, TP4 (default 24). The cap was tested with the KV RAM
  cache at 48g (12 GiB per rank over four ranks), so `--swarm` needs that: without `--kv-ram-cache` setup and `merlin
  serve` set the cache to 48g and say so, and the RAM check counts it; a smaller `--kv-ram-cache` is refused, never
  raised, and so is `--kv-ram-cache 0` beside swarm mode (a saved answer included: setup says which flag to change; the
  question is not asked while `--kv-ram-cache 0` is given). It is refused on rows with no swarm cap (the message names the
  rows that have one) and with `--spec dflash2`. `merlin serve` counts an exported `SWARM=1` as `--swarm` and checks an
  exported `MERLIN_KV_HOST_TIER_GIB` instead of overwriting it. An exported `SEQS` still wins over every cap. A row whose
  swarm cap has not been run on the reference host says "untested ⚫ (expected to work)" in the notes. Setup asks once, default no, on rows that have a swarm cap; `--no-swarm`
  turns a saved answer off. On a profile that has a swarm cap, setup's notes and the plan say which cap applies (default or `--swarm`; setup does not
  read an exported `SEQS`, `merlin serve` does and warns when it is above the default).
- `--ple fp8|int6` picks the PLE table: `fp8` (47.68 GiB, the default) or `int6`, the INT6 g32 table (38.74 GiB; it
  passed the 3.0.1 quality checks against FP8 on flash-next and swift1.5). There is no second checkpoint: flash-next,
  swift1.5 and swift1.5-huihui carry byte-identical FP8 PLE shards, and one INT6 PLE pack serves flash-next and swift1.5.
  swift1.5-huihui is refused for now: its MTP acceptance with the INT6 table fell past the threshold; serve it with FP8.
  With `--ple int6` setup (and `merlin pull <model> --ple int6`)
  verifies the FP8 checkpoint in `--models-dir`, downloads the pack once into `flash-next-ple-int6` beside it
  (`MERLIN_PLE_PACK_DIR` points at a local copy instead), and builds
  `<folder>-int6`: relative symlinks to the FP8 checkpoint's files and the pack's shards plus a generated `config.json`
  and index (`ple_embedding_dtype` `int6_g32`). The FP8 files are never changed. On a filesystem without symlinks the
  overlay uses hard links when everything shares one filesystem, else setup refuses (it never copies the table). A
  re-run keeps a current overlay and rebuilds a stale one (a changed FP8 checkpoint, pack or checksum list).
  `merlin serve <model> <folder>-int6 --ple int6` refuses an overlay that is stale or whose checkpoint or pack is
  unverified; `merlin verify <folder>-int6 --ple int6` re-hashes the checkpoint and the pack and checks the overlay.
  A container must mount the common parent of the checkpoint, the pack and the overlay (e.g. `-v ~/models:/models`),
  or the relative links do not resolve. `merlin serve` refuses an INT6 checkpoint without `--ple int6`; `--ple fp8`
  drops a saved `int6`. There is no INT4 table, and the table is never read from disk or the page cache.

The host-RAM check adds, per server, the table, the rest of the server (the measured FP8 floor less the FP8 table: about
21.3 GiB at four workers, plus 2 GiB per worker above four) and the cache, and refuses flags that do not fit, naming the
largest `--kv-ram-cache` that would:

| PLE table | `--kv-ram-cache` (per rank, 4 ranks) | resident need, 4 workers | smallest container |
|---|---|---|---|
| FP8 | off | 69 GiB | 72 GiB tested minimum (96 GiB qualified) |
| FP8 | 16g (4 GiB) | 85 GiB | 96 GiB |
| FP8 | 24g (6 GiB) | 93 GiB | 96 GiB |
| INT6 g32 | off | 60.06 GiB (estimate) | unmeasured (no capped-host boot yet) |
| INT6 g32 | 32g (8 GiB) | 92 GiB (estimate) | 96 GiB |

Only FP8 with the cache off has measured boots (the 69 GiB floor, 96 GiB qualified; tested minimum 72 GiB, where a capped
host booted and passed 262K, while 64 GiB was OOM-killed at boot, so a 64 GB host is unsupported and the launcher refuses
below the floor; `MERLIN_SKIP_RAM_CHECK=1` keeps it a warning). The other rows are estimates from
that floor and the measured 9 GiB table difference (67 vs 58 GiB used on the reference host); none has booted
under a capped container yet, so setup warns on them.

## Drafter: MTP or DFlash2

`--spec mtp|dflash2` (on `./setup.sh` and `merlin serve`; default `mtp`) picks the speculative drafter. `dflash2` is
incoai's DFlash2 block-diffusion drafter: +5-18 % decode against MTP on 4 x 3090, but its KV pool is 789,668 tokens,
which holds 3 x 262K and not 6 x 131K. It runs only on the Flash-Next family rows
with a qualified DFlash2 geometry, the three 4 x 3090 P2P rows; every other row, and 27B, Super VL and Ornith (which run
their own MTP head), refuse it, as does the launcher for a raw `SPEC=dflash2`. On cards that report a little less VRAM
than their class the DFlash2 pin is lowered the same way as the MTP one and the plan marks it untested ⚫.
`./setup.sh --spec dflash2` downloads `incoai/Qwen3.8-Flash-Next-DFlash2` at its pinned revision (about 1.2 GB) to
`~/models/flash-next-dflash2` (or `DFLASH2_MODEL`), checks `config.json` and `model.safetensors` against
`models/flash-next/dflash2.sha256`, and starts the server with `DFLASH2_MODEL` set. `merlin serve --spec dflash2` takes
`DFLASH2_MODEL`, else that folder, and refuses before boot when the drafter is not there.

## Flags

| Flag | Effect |
|---|---|
| `--yes` / `-y` | no questions: the recommended (or given) answers, downloads confirmed |
| `--reconfigure` | ask again (the saved answers are the defaults) |
| `--dry-run` | print the checks, answers, download plan and serve command; download nothing, write nothing (not even the answers) |
| `--simulate SPEC` | run the checks and plan against preflight's fabricated hardware (`4x3090`, `2x3090:ram=64`, `4x4090:nop2p`, ...); implies `--dry-run` unless `--no-start` (nothing ever starts on simulated hardware) |
| `--no-start` | install and download, then stop before `merlin serve` |
| `--rehash` | hash every checkpoint file again even when the verified marker is current |
| `--kv-ram-cache SIZE` | Flash-Next and the 27B (both experimental in 3.0.1: a session evicted under heavy concurrent prefill may recompute, output identical) and Ornith 1.5 (untested ⚫): a host-RAM cache for evicted prefix KV blocks (restored instead of recomputed; no extra context or sessions; never disk), SIZE GiB per server (`24g`, `24gib`, `24`; `0` = off), split evenly over its ranks; also on `merlin serve` |
| `--swarm` / `--no-swarm` | Flash-Next 4 x 3090 and 27B quad-card rows (experimental): the row's tested high-concurrency cap, 24 / 32 sessions instead of 8 / 24, with the KV RAM cache set to the tested 48g unless `--kv-ram-cache` gives at least that; a smaller cache is refused; also on `merlin serve --swarm` |
| `--spec mtp\|dflash2` | Flash-Next: the drafter, `mtp` (default) or `dflash2` (+5-18 % decode, 3 x 262K only; downloads the drafter); also on `merlin serve` |
| `--ple fp8\|int6` | Flash-Next: the PLE table, `fp8` (default) or `int6` (INT6 g32: the family pack laid over the FP8 checkpoint as `<folder>-int6`; flash-next and swift1.5 only, swift1.5-huihui not yet qualified); also on `merlin serve` and `merlin pull` |
| `--from-source` | compile the engine (about 1.5 h on 8 cores) instead of taking the release wheel |
| `--help` | the option list |

Non-interactive example: `./setup.sh --yes --model 27b --models-dir /data/27b --host 0.0.0.0 --api-key "$KEY" --port 8100 --tp 2`.

Environment overrides: `MERLIN_WHEEL=<file>` installs a local copy of the release wheel, `MERLIN_WHEEL_URL=<url>` fetches
it from elsewhere (both still checked against `release/merlin-wheel.sha256`); `MERLIN_RELEASE_VERSION` overrides the
release version read from the `## Merlin <version>` heading of `docs/CHANGELOG.md`; `MERLIN_UV_VERSION` another uv (checked against its published `.sha256` only);
`HF_TOKEN` is passed to the Hugging Face API and to `hf download` when set (no Merlin checkpoint needs it); `CC` / `CXX` name the compiler
pair (remembered).

## Downloads

The plan is shown with sizes before anything starts and confirmed once, after the questions (`--yes` skips the
question; a saved re-run asks nothing). An unknown size is shown as unknown, never as zero.

- uv (26 MB) and Python 3.13 (33 MB) into `.merlin/`, only when the host has none: uv's archive is checked against the
  release's published `.sha256` and, for the pinned version, the digest recorded in `setup.sh`;
- the engine wheel from the GitHub release `merlin-v<version>` named by `release/merlin-wheel.sha256`, verified against
  that sha256 (a wrong file is deleted, not installed); `curl -C -` resumes a partial download; the asset name with
  `+`, its `%2B` form and the `.` form GitHub may rename it to are all tried;
- the Python packages of `release/requirements-0.32.lock` (222 pins with hashes), fetched by uv and cached, so a
  reinstall takes them from the cache;
- the checkpoint, by the venv's `hf download` (resumable), pinned to one revision: the one `checkpoint.sha256` or
  `release/download-sizes.json` records. A checkpoint whose Hugging Face repository is not public yet
  (`pending_publication` in `release/download-sizes.json`) is served from a local copy only
  (`--models-dir`); nothing is downloaded from a moving `main`.

Checkpoint sizes come from the Hugging Face API, or from `release/download-sizes.json` when it is unreachable.

## Verification

After the download every file in `models/<model>/checkpoint.sha256` is hashed. A pass writes
`<models folder>/.merlin-verified.json` (the model, the checksum list's hash, each file's size and mtime). Later runs
skip the full re-hash while those match, and re-hash whenever the list, the file set, a size or an mtime changed, or
with `--rehash`. The trade-off: a corruption that keeps a file's size and mtime passes until `--rehash` (a 115 GiB
re-hash takes minutes, which is why the marker exists). A failure names the files; delete them and run again
(`hf download` fetches them anew).

`merlin pull <model> [--models-dir DIR] [--rehash] [--dry-run]` runs only the download and verification: no GPU,
driver or compiler check, no serving choice. It uses the engine venv's `hf` when there is one, else a small
`.merlin/hf` environment with the lock's huggingface-hub, so it works on a host that only downloads.

## Re-runs and updates

- `./setup.sh` again: the saved answers (`.merlin/setup.json`) are used, nothing is asked (not even the download
  confirmation), the venv is reused when `.merlin/state.json` records the same wheel sha, lock hash and commit, the
  checkpoint is skipped when its marker is current, and the server starts.
- `git pull && ./setup.sh`: a moved commit, changed lock or changed wheel reinstalls the engine: the working venv is
  moved aside, the new one is installed and verified, then the old one is removed; if the install or the verify
  fails, the old venv comes back and the engine still serves. The packages come from uv's cache; the model is never
  re-downloaded. The install holds `.merlin/setup.lock`: a second `./setup.sh` in the same checkout stops at
  once with a message instead of moving the first one's venv. A `.merlin/venv-old-*` left by an
  install that was killed is named at the end of the next successful install and kept until you remove it.
- `./setup.sh --reconfigure`: the questions again, with the saved answers as defaults.

## Where things live

| Path | Content |
|---|---|
| `.merlin/bin/` | uv (fetched, or a link to the one on PATH) and a `python3` link to the repo-local Python, for `release/install-env.sh` |
| `.merlin/python/` | the uv-managed Python 3.13 (`UV_PYTHON_INSTALL_DIR`) |
| `.merlin/downloads/` | the release wheel |
| `.merlin/setup.json` | the answers (model, host, compiler; folder, port and layout per model, and the flags that apply to the model: `--ple` for Flash-Next, `--kv-ram-cache` and `--swarm` for Flash-Next and the 27B), mode 0600 |
| `.merlin/api-key` | the API key for network access, mode 0600 (`.merlin/` itself is 0700) |
| `.merlin/state.json` | the install record (wheel sha, lock hash, commit, venv) |
| `.merlin/setup.lock` | held while the engine installs, so two `./setup.sh` runs never install over each other |
| `.merlin/hf/` | the download-only `hf` environment (`merlin pull` without the engine) |
| `venv/` | the engine environment `release/install-env.sh` created |
| `.build-source`, `.build-products`, `.build-products.sha256` | install-env's provenance and product hashes, in the checkout |
| `.vllm-cache/` | compile / Triton caches the launcher writes |
| `~/.cache/uv/` | uv's package cache (about 3.4 GiB; `uv cache clean` empties it) |
| `<models folder>` | the checkpoint and its `.merlin-verified.json` marker |

`.merlin/`, `venv/`, `.vllm-cache/` and the `.build-*` files are git-ignored.

## Uninstall

Delete `.merlin/`, `venv/`, `.vllm-cache/` and the `.build-*` files inside the checkout, the models folder
(`~/models/<model>` by default), and `~/.cache/uv` if you do not use uv otherwise. Nothing else was written: no
packages, services, shell profiles or system paths.

## Manual route

The pieces `setup.sh` drives are documented on their own: [release/install-env.sh](../release/README.md) (the venv),
`hf download` plus `sha256sum -c models/<model>/checkpoint.sha256` (the checkpoint), `merlin doctor` and
`merlin serve` ([README](../README.md#quick-start)).
