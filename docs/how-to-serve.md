# How to serve

From GPUs to a running, OpenAI-compatible endpoint in one command, then everything you can change afterwards.
Merlin runs on Linux x86-64 with glibc 2.38 or newer (Ubuntu 24.04, Debian 13, RHEL 10 or newer) and NVIDIA Ampere or Ada cards (sm_80, sm_86, sm_89). Which GPUs run which checkpoint,
and every profile the engine ships, is in [Which GPUs run what](#6-which-gpus-run-what); what each checkpoint needs in
RAM and disk is in [SETUP.md](SETUP.md), and setup prints it for your choice.

## 1. Install and start

```bash
git clone https://github.com/halt95/merlin-engine.git && cd merlin-engine && ./setup.sh
```

`./setup.sh` does the whole job and asks before anything large happens:

1. **Checks the host**: GPUs, NVIDIA driver (R580 or later), RAM, disk, compiler (GCC/G++ 13 or newer). A problem
   prints a `FAIL` line with the fix, and nothing is downloaded.
2. **Asks five questions**, each with a recommended answer that Enter takes, and two more only where they apply:

   | Question | Recommended on 4× RTX 3090 | Flag |
   |---|---|---|
   | Model | `flash-next` (chosen from your GPUs and RAM, with the reason shown) | `--model` |
   | Layout | the largest tested layout for your cards | `--tp`, `--instances` |
   | Models folder | `~/models/<model>`; an existing download is reused | `--models-dir` |
   | Access | `local` (this machine only) | `--host`, `--api-key` |
   | Port | 8000 (27B: 8100) | `--port` |
   | Drafter (Flash-Next family rows with DFlash2) | `mtp` | `--spec` |
   | Swarm mode (rows with a swarm cap) | no | `--swarm` |

3. **Lets you change an answer**: it lists them, a number asks that question again, Enter carries on.
4. **Shows every download with its size** (the prebuilt engine, the Python packages, the checkpoint) and asks once.
   `n` stops with nothing downloaded and nothing saved.
5. **Installs and starts**: the prebuilt engine into `venv/`, the checkpoint checked file by file against its
   checksums, then the server on `127.0.0.1` (port 8000; 8100 for the 27B and ThinkingCap). It is ready when the log says `Application startup complete`; the
   first start compiles kernels and takes longer than later ones.

Nothing is installed system-wide and nothing needs sudo: uv and Python 3.13 live in `.merlin/`, the engine in `venv/`.

| Later | Command |
|---|---|
| Start again (no questions) | `./setup.sh` |
| Update the engine (the model is not downloaded again) | `git pull && ./setup.sh` |
| Change your answers | `./setup.sh --reconfigure` |
| See the plan, change nothing | `./setup.sh --dry-run` |
| No questions at all | `./setup.sh --yes --model 27b --port 8100` |
| Stop the server | Ctrl+C |

## 2. Merlin commands

`./setup.sh` remembers its answers per model. `merlin serve` takes the same choices as flags, so you can change a
setting without running setup again. The full setup option list is in [SETUP.md](SETUP.md#flags).

| Command | What it does |
|---|---|
| `./setup.sh` | check the host, ask a few questions, download, verify and start |
| `./setup.sh --yes --model <name>` | the same with the recommended answers and no questions |
| `./setup.sh --reconfigure` | ask again, with the saved answers as defaults |
| `./setup.sh --dry-run` | print the checks, answers, downloads and serve command; change nothing |
| `venv/bin/merlin serve <model>` | serve on the largest tested layout for the GPUs found |
| `venv/bin/merlin serve <model> --dry-run` | print the profile, cards, environment and command; start nothing |
| `venv/bin/merlin list [model]` | every profile: checkpoint, status, hardware, launcher |
| `venv/bin/merlin pull <model>` | download and verify only |
| `venv/bin/merlin verify <dir>` | re-hash every file of a checkpoint against its published checksums |
| `venv/bin/merlin doctor` | the hardware report: GPUs, PCIe links, P2P, driver, RAM, shared memory, GPU unlock state per card |

### Turn on the Merlin features

```bash
./setup.sh --swarm                # experimental: more sessions on 4× RTX 3090, with a 48 GB KV RAM cache
./setup.sh --kv-ram-cache 24g     # experimental: keep evicted prefixes in host RAM and restore them
./setup.sh --spec dflash2         # Flash-Next family on 4× RTX 3090 with P2P: faster decode (downloads the 1.2 GB drafter)
./setup.sh --ple int6             # Flash-Next and Swift1.5: about 9 GiB less host RAM
venv/bin/merlin serve ~/models/flash-next --swarm   # the same flags without running setup
```

The table below has each flag's limits.

### `merlin serve` options

`./setup.sh` takes `--tp`, `--instances`, `--spec`, `--kv-ram-cache`, `--swarm` and `--ple` too.

| Option | Effect |
|---|---|
| `--tp N` | pick the layout with this tensor-parallel size |
| `--instances N` | start N servers on separate cards, on consecutive ports; all start or none do |
| `--wait-ready SECONDS` | with `--instances`: wait until every instance answers `/health` and is still running; if one dies (even after answering) or misses the deadline, all are stopped and the command exits 1 |
| `--profile <id>` | force one profile (checked against your hardware) |
| `--spec mtp\|dflash2` | Flash-Next family: the drafter. `mtp` is the default. `dflash2` decodes 5–18 % faster and holds three 262K sessions instead of six 131K ones |
| `--kv-ram-cache SIZE` | **Experimental.** Keeps conversation prefixes the GPU evicts in host RAM (for example `24g`) and restores them instead of recomputing them; output is identical. Flash-Next family with MTP and Qwen3.8-27B; Ornith 1.5 accepts it, untested ⚫. Off by default; adds no context or sessions. Not with `--spec dflash2`. Tuning knobs: [Flash-Next reference](../models/flash-next/docs/reference.md#kv-ram-cache-tuning-knobs) |
| `--swarm` | **Experimental.** Raises the session cap on 4× RTX 3090, with or without P2P: 8 → 24 on Flash-Next, Swift1.5 and Swift1.5-Huihui, 24 → 32 on Qwen3.8-27B and ThinkingCap (four cards). Tested on Flash-Next and the 27B with P2P; untested ⚫ on Swift1.5 and Swift1.5-Huihui and on every row without P2P (the server prints a warning there). Sets a 48g KV RAM cache; a smaller `--kv-ram-cache` is refused. Refused on profiles without a swarm cap and with `--spec dflash2`. An exported `SEQS` overrides every cap. `./setup.sh --no-swarm` turns off a saved answer |
| `--ple fp8\|int6` | Flash-Next and Swift1.5: the embedding table. `fp8` is the default; `int6` saves about 9 GiB of host RAM; `merlin verify <dir> --ple int6` checks the pack and `MERLIN_PLE_PACK_DIR` points at a local copy. Swift1.5-Huihui serves `fp8` only |
| `-- <vLLM args>` | pass anything after `--` straight to vLLM. Merlin notes a recipe value you override; an extra `--speculative-config` (any spelling vLLM accepts: `--speculative_config`, `-sc` or `--speculative-config.KEY`) merges with the recipe's, and changing its method or K is refused |

Environment: `GPUS=0,1` picks the cards, `PORT=8101` the port (a port already in use stops the launch;
`MERLIN_SKIP_PORT_CHECK=1` skips that check), `HOST=0.0.0.0` with `VLLM_API_KEY` serves on your
network ([step 4](#4-serve-on-your-network)). `SWARM=1` is the same as `--swarm`; `KV_DTYPE=fp8` opts sm_80 cards
into FP8 KV for the Flash-Next family. `SEQS` and `MAXLEN` override the session cap and context length on
Flash-Next family models, the 27B and Ornith 1.5 (a note is printed when either is off the tested row); `PP_PARTITION` sets Flash-Next's pipeline split.

### Longer context with YaRN (Flash-Next)

No profile turns YaRN on; you add it as a vLLM override. Asking for more than 262K without it is refused. 393,216
tokens:

```bash
venv/bin/merlin serve ~/models/flash-next -- --max-model-len 393216 --hf-overrides '{"text_config": {"max_position_embeddings": 393216, "rope_parameters": {"rope_type": "yarn", "factor": 1.5, "original_max_position_embeddings": 262144, "rope_theta": 10000000, "partial_rotary_factor": 0.25, "mrope_section": [11, 11, 10], "mrope_interleaved": true}}}'
```

| Length | `factor` | `max_position_embeddings` | `--max-model-len` | Tested with |
|---|---|---|---|---|
| 393K | 1.5 | 393216 | 393216 | 🟢 MTP and DFlash2 |
| 524K | 2.0 | 524288 | 524288 | 🟢 MTP and DFlash2 |
| 917K | 4.0 | 1048576 | 917504 | 🟢 MTP |
| 1M | 4.0 | 1048576 | 1048576 | ⚫ untested |

YaRN on Qwen3.8-27B is untested. The drafter follows the target's YaRN table past 262K.

## 3. Talk to it

The server speaks the OpenAI API at `http://127.0.0.1:8000/v1`, and `/v1/messages` for clients that use that API.
Each checkpoint answers to a model name:

| Checkpoint | `--model` | Model name in requests | Default port |
|---|---|---|---|
| Flash-Next | `flash-next` | `flash-next-v2` | 8000 |
| Swift1.5 | `swift1.5` | `flash-next-v2` | 8000 |
| Swift1.5-Huihui | `swift1.5-huihui` | `flash-next-swift-huihui` | 8000 |
| Qwen3.8-27B | `27b` | `qwen38-27b` | 8100 |
| ThinkingCap | `27b-thinkingcap` | `qwen38-27b` | 8100 |
| Nemotron 3.5 Super VL | `nemotron-super-vl` | `svl` | 8000 |
| Ornith 1.5 | `ornith1.5` | `ornith` | 8000 |

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model": "flash-next-v2", "messages": [{"role": "user", "content": "Say hello."}], "max_tokens": 64}'
```

```python
from openai import OpenAI

# The client reads OPENAI_API_KEY: set it to anything when served locally, to the key setup generated on a network (step 4).
client = OpenAI(base_url="http://127.0.0.1:8000/v1")
reply = client.chat.completions.create(model="flash-next-v2", messages=[{"role": "user", "content": "Say hello."}])
print(reply.choices[0].message.content)
```

**Thinking.** The Qwen3.8 checkpoints think before answering by default. Turn it off for one request with
`"chat_template_kwargs": {"enable_thinking": false}`; with thinking on, a short `max_tokens` can end inside the
reasoning with empty `content`. Each checkpoint's card lists its recommended sampling settings.

**Images.** Send them as OpenAI image content parts, up to 8 MP per image (Nemotron: its native ~3.4 MP; the one-card 27B layout: 4 MP).
Image and media URLs behave as in upstream vLLM: any host, and redirects are followed. If untrusted clients can reach
the API, restrict fetches with `merlin serve <model> -- --allowed-media-domains <host>`.

**Health.** `/health` returns 503 when the engine completes no step for 120 s with requests in flight
(`VLLM_MERLIN_HEALTH_STALL_S` changes it, `0` turns it off), so a supervisor or load balancer can restart a stuck
server. The stall watch runs when the server logs stats (the default) with one API server and one engine; with more
than one of either it is off. A failed request logs its cause at ERROR and returns a clear error message.

**Request limits.** `allowed_token_ids` and `logit_bias` take up to 1,024 entries; `bad_words` takes up to 128
sequences with at most 1,024 tokens between them; `stop_token_ids` takes up to 128 when `min_tokens` is set. More
returns HTTP 400. With speculative decoding (every shipped profile), `logit_bias` is refused.
Generative scoring takes up to 32 items and 32,768 tokens per request.

## 4. Serve on your network

Answer `network` at the Access question (or pass `--host 0.0.0.0`). Setup generates a key, shows it once and keeps
it in `.merlin/api-key`; clients send it as a bearer token:

```bash
export OPENAI_API_KEY="$(cat .merlin/api-key)"
curl -s http://<this-machine>:8000/v1/models -H "Authorization: Bearer $OPENAI_API_KEY"
```

Started by hand instead (`merlin serve`, step 2), set `HOST=0.0.0.0` and `VLLM_API_KEY`. Without a key the launcher
warns that every client that can reach the machine gets in. Merlin is meant for your own network: put a reverse
proxy with TLS in front before you open it to anything wider.

Streams send a keep-alive comment every 15 s while they wait (a queued request, a long prefill), so proxies with a
30–60 s timeout keep the connection open. `merlin serve <model> -- --sse-keep-alive-interval 0` turns it off; any
other number of seconds replaces the 15.

## 5. Choose your GPUs

**Optional host upgrades on RTX 3090s.** PCIe P2P and Zagitta's GPU unlock both speed Merlin up, and neither is
required. The unlock lifts the GeForce limit on FP16/BF16 math: about 16 % faster prefill on Flash-Next and 29 % on the
27B, with decode and output unchanged. It needs a cold boot to take effect, and `merlin doctor` reports each card's
unlock state (run it as root on the host). See the
[hardware matrix](merlin/hardware-matrix.md#optional-host-upgrade-gpu-unlock-geforce-ga102) before installing it; our 3.0.1 bench numbers are measured with it.

Setup's Layout question covers the common cases. On a bigger host, or to run things by hand, `merlin serve` takes
the same choices directly:

```bash
GPUS=5,6 venv/bin/merlin serve ~/models/qwen38-27b                  # 27B on GPUs 5 and 6 (TP=2)
GPUS=9,10,11,12 PORT=8101 venv/bin/merlin serve ~/models/qwen38-27b # 27B on GPUs 9-12 (TP=4), its own port
venv/bin/merlin serve ~/models/qwen38-27b --tp 1 --instances 4      # four one-GPU servers, ports 8100-8103
venv/bin/merlin serve ~/models/flash-next --dry-run                 # print the plan, start nothing
venv/bin/merlin list                                                # every checkpoint, profile and status
```

- **`GPUS=`** (or `CUDA_VISIBLE_DEVICES=`) uses exactly those cards; the layout follows the count.
- **`--tp N`** picks the layout by tensor-parallel size; **`--profile <id>`** forces one profile.
- **`--instances N`** starts N servers on separate cards, each on its own port.
- **Flash-Next on four cards without P2P** can also run PP4: `--profile fn-quad-3090-nop2p-pp4` (RTX 3090, tested:
  prefill 47–53 % faster, decode 14 % slower) or `--profile fn-quad-24g-sm89-pp4` (RTX 4090, untested). TP2×PP2
  stays the default.
- **More sessions, less RAM, longer context:** `--swarm`, `--kv-ram-cache`, `--ple int6` and YaRN are in the
  [Merlin commands](#2-merlin-commands).
- **Without peer-to-peer** (most RTX 4090 and CMP 170HX hosts), several one-GPU 27B servers serve many users faster
  than one large one; one large server holds the most context. `merlin serve` starts the
  large one by default and prints the `--tp 1 --instances N` command for the other. On RTX 3090s without P2P the 27B,
  Nemotron and Flash-Next run their no-P2P profiles: the same layouts with P2P off. On four 3090s Ornith 1.5 and
  Flash-Next's PP4 layout are tested that way; Flash-Next's default layout and the 27B were boot-checked with
  `--swarm` (2026-11-01), and Nemotron is not yet run.

## 6. Which GPUs run what

### Hardware

| Hardware | VRAM per card | Arch | Profiles | Testing | Performance |
|---|---|---|---|---|---|
| 4× RTX 3090, PCIe p2p | 24 GB | sm_86 | Included | 🟢 Tested | 🟢 Optimised |
| 2× RTX 3090 | 24 GB | sm_86 | Included | 🟢 Tested | 🟢 Optimised |
| 4× RTX 3090 w/o p2p | 24 GB | sm_86 | Included | 🟠 Partly tested | 🟠 Optimising |
| 8× RTX 3090 | 24 GB | sm_86 | Included | ⚫ Untested | 🟠 Optimising |
| 4× RTX 4090 | 24 GB | sm_89 | Included | 🟡 Testing | 🟠 Optimising |
| 4× RTX 4090 | 48 GB (modded) | sm_89 | Included | ⚫ Untested | 🔴 Unoptimised |
| 4× CMP 170HX | 64 GB (unlocked) | sm_80 | Included | 🟡 Testing | 🟠 Optimising |
| 2× CMP 170HX | 64 GB (unlocked) | sm_80 | Included | 🟡 Testing | 🟠 Optimising |
| A100 | 40 / 80 GB | sm_80 | Included | ⚫ Untested | 🔴 Unoptimised |
| A30 | 24 GB | sm_80 | Included | ⚫ Untested | 🔴 Unoptimised |

🟢 **Tested**: measured on our hardware. 🟡 **Testing**: full-model serving runs passed on cloud-hosted cards; kernel parity qualification is still to run. 🟠 **Partly tested**: some models or layouts measured on our hardware (here Ornith 1.5 and Flash-Next PP4). ⚫ **Untested**: a profile ships; not run by us yet, expected to work.
A row shows the best state of any checkpoint's profile for those cards; the tables below give it per checkpoint.
🟢 **Optimised**: measured and tuned on this hardware. 🟠 **Optimising**: works, tuning in progress. 🔴 **Unoptimised**: runs on the stock
profile, not tuned yet.

### Which checkpoint runs where

Find your GPUs and card count. Each cell says how the checkpoint runs there and how many full 262K-token
conversations it holds at once; `merlin serve` and `./setup.sh` pick the setup from the GPUs they find. Nemotron,
native 1M, has its own table below.

| Setup | Flash-Next, Swift1.5, Huihui | Qwen3.8-27B, ThinkingCap | Ornith 1.5 |
|---|---|---|---|
| 1× RTX 3090 | needs 4 cards | 🟢 1 session | needs 4 cards |
| 2× RTX 3090 | needs 4 cards | 🟢 3 sessions | needs 4 cards |
| 4× RTX 3090 | 🟢 3 sessions | 🟢 7 sessions | 🟢 3 sessions |
| 8× RTX 3090 | ⚫ 17 sessions (Flash-Next only) | 🟢 2 servers × 7 sessions (4 cards each) | 🟢 2 servers × 3 sessions (4 cards each) |
| 2× RTX 3090, no P2P | needs 4 cards | ⚫ 3 sessions | needs 4 cards |
| 4× RTX 3090, no P2P | 🟢 3 sessions (PP4 layout) | ⚫ 7 sessions | 🟢 3 sessions |
| 8× RTX 3090, no P2P | ⚫ 17 sessions (Flash-Next only) | ⚫ 2 servers × 7 sessions (4 cards each) | 🟢 2 servers × 3 sessions (4 cards each) |

<details>
<summary><b>Other Ampere and Ada GPUs</b>: RTX A6000 / A40, RTX 4090, 32 and 48 GB Ada cards, A30, A100, CMP 170HX</summary>

| Setup | Flash-Next, Swift1.5, Huihui | Qwen3.8-27B, ThinkingCap | Ornith 1.5 |
|---|---|---|---|
| 1× RTX A6000 / A40, 48 GB | needs 2 cards | ⚫ 3 sessions | no profile yet |
| 2× RTX A6000 / A40, 48 GB | ⚫ 5 sessions | ⚫ 8 sessions | no profile yet |
| 4× RTX A6000 / A40, 48 GB | ⚫ 25 sessions | ⚫ 18 sessions | no profile yet |
| 1× RTX 4090, 24 GB | needs 4 cards | ⚫ 1 session | no profile yet |
| 2× RTX 4090, 24 GB | needs 4 cards | 🟡 3 sessions | no profile yet |
| 4× RTX 4090, 24 GB | 🟡 3 sessions | 🟡 7 sessions | no profile yet |
| 1× RTX 4080 Super, 32 GB (modified) | needs 4 cards | ⚫ 1 session | no profile yet |
| 2× RTX 4080 Super, 32 GB (modified) | needs 4 cards | ⚫ 3 sessions | no profile yet |
| 4× RTX 4080 Super, 32 GB (modified) | ⚫ 10 sessions | ⚫ 7 sessions | no profile yet |
| 1× RTX 6000 Ada / L40S / RTX 4090 48 GB | needs 2 cards | ⚫ 1 session | no profile yet |
| 2× RTX 6000 Ada / L40S / RTX 4090 48 GB | ⚫ 5 sessions | ⚫ 3 sessions | no profile yet |
| 4× RTX 6000 Ada / L40S / RTX 4090 48 GB | ⚫ 25 sessions | ⚫ 7 sessions | no profile yet |
| 1× A30, 24 GB | needs 4 cards | ⚫ 1 session | no profile yet |
| 2× A30, 24 GB | needs 4 cards | ⚫ 3 sessions | no profile yet |
| 4× A30, 24 GB | ⚫ 1 session | ⚫ 7 sessions | no profile yet |
| 1× CMP 170HX, 32 GB | needs 3 cards | ⚫ 1 session | no profile yet |
| 2× CMP 170HX, 32 GB | needs 3 cards | ⚫ 4 sessions | no profile yet |
| 4× CMP 170HX, 32 GB | ⚫ 4 sessions | ⚫ 11 sessions | no profile yet |
| 1× A100 40 GB / CMP 170HX 40 GB | needs 4 cards | ⚫ 2 sessions | no profile yet |
| 2× A100 40 GB / CMP 170HX 40 GB | needs 4 cards | ⚫ 6 sessions | no profile yet |
| 4× A100 40 GB / CMP 170HX 40 GB | ⚫ 7 sessions | ⚫ 14 sessions | no profile yet |
| 1× CMP 170HX, 64 GB | needs 2 cards | 🟡 5 sessions | no profile yet |
| 2× CMP 170HX, 64 GB | 🟡 6 sessions | 🟡 11 sessions | no profile yet |
| 4× CMP 170HX, 64 GB | 🟡 17 sessions (8 at once; Flash-Next only) | 🟡 25 sessions | no profile yet |
| 1× A100, 80 GB | ⚫ 1 session | ⚫ 7 sessions | no profile yet |
| 2× A100, 80 GB | ⚫ 11 sessions | ⚫ 15 sessions | no profile yet |
| 4× A100, 80 GB | ⚫ 28 sessions | ⚫ 32 sessions | no profile yet |

</details>

**Nemotron 3.5 Super VL** serves its native 1,048,576-token context, with the 262K count beside it. It needs four cards (or two servers of four). The 1M counts are
the KV pool divided by 1,048,576; on 4× RTX 3090 (a 1,856,367-token pool) a 1,048,052-token needle came back exact, and
an identical re-send was served with 1,047,936 tokens from the cache, 3.05 s to the first token, with no preemptions.

| Setup | Full 1M conversations | Full 262K conversations |
|---|---|---|
| 4× RTX 3090 | 🟢 1 session | 🟢 5 sessions |
| 8× RTX 3090 | 🟢 2 servers × 1 session (4 cards each) | 🟢 2 servers × 5 sessions (4 cards each) |
| 4× RTX 3090, no P2P | ⚫ 1 session | ⚫ 5 sessions |
| 8× RTX 3090, no P2P | ⚫ 2 servers × 1 session (4 cards each) | ⚫ 2 servers × 5 sessions (4 cards each) |
| 4× RTX A6000 / A40, 48 GB | ⚫ 13 sessions | ⚫ 54 sessions (32 at once) |
| 4× RTX 4090, 24 GB | ⚫ 1 session | 🟡 5 sessions |
| 4× RTX 4080 Super, 32 GB (modified) | ⚫ 1 session | ⚫ 5 sessions |
| 4× RTX 6000 Ada / L40S / RTX 4090 48 GB | ⚫ 1 session | ⚫ 5 sessions |
| 4× A30, 24 GB | ⚫ 1 session | ⚫ 5 sessions |
| 4× CMP 170HX, 32 GB | ⚫ 5 sessions | ⚫ 21 sessions |
| 4× A100 40 GB / CMP 170HX 40 GB | ⚫ 9 sessions | ⚫ 37 sessions (32 at once) |
| 4× CMP 170HX, 64 GB | ⚫ 21 sessions | 🟡 86 sessions (32 at once) |
| 4× A100, 80 GB | ⚫ 29 sessions | ⚫ 119 sessions (32 at once) |

<sub>🟢 tested on our hardware · 🟡 testing: full-model runs passed on cloud-hosted cards, kernel parity qualification still to run · ⚫ expected to run: the engine's own estimate, not yet run by us.
On 4× RTX 4090 and 2× CMP 170HX 64 GB the Flash-Next column's 🟡 is Flash-Next and Swift1.5; Swift1.5-Huihui has not run there (⚫). The single RTX 4090 27B row stays ⚫: its full-model run failed on the cloud-hosted cards (262K prefill out of memory at chunk 1024); at the shipped chunk 512 only the 262K and 131K needles have run. Nemotron ran on the cloud-hosted cards in its 262K mode, so the 1M column stays ⚫ there.
"2 servers × 7 sessions" means two separate servers on the cards (`merlin serve --instances 2`). On 8× RTX 3090 with P2P, `merlin serve` picks the tested four-card Flash-Next layout (3 sessions); the 17-session eight-card layout is `--profile fn-octo-3090-tp2pp4` (untested). Without P2P it is the default.
"N sessions (M at once)" means the KV cache holds N full 262K conversations but the server
admits M requests at a time; the rest queue. Without P2P every model runs the same layouts with P2P off; on four
RTX 3090s Ornith 1.5 and Flash-Next's PP4 layout are tested that way (Flash-Next's default layout and the 27B were
boot-checked with `--swarm`, 2026-11-01), while Nemotron and the two-card and single-server eight-card setups
are not yet (⚫). The 🟢 8× cells are two separate 4-card servers of a profile tested on four cards, not an 8-card qualification. The 3090 Ti and
RTX A5000 run the RTX 3090 profiles, untested on those cards (⚫). KV cache: FP8 everywhere, except Flash-Next and its fine-tunes on sm_80 cards (A30, A100, CMP 170HX),
which default to BF16, so fewer sessions fit; `KV_DTYPE=fp8` opts them into FP8 KV (untested ⚫). On 4× CMP 170HX 64 GB the 27B (one, two and four cards) and Nemotron ran with
FP8 KV, 262K needles exact. Source: the engine's profile registry (`tools/merlin/profiles/`).</sub>

### Every profile

A profile is one tested or estimated setup of a checkpoint on one kind of hardware: the layout, the KV pin and the
batch limits. `merlin serve` and `./setup.sh` pick one from the GPUs they find; `--profile <id>` forces one, and
`venv/bin/merlin list` prints the same list on your machine. `~N` is the engine's estimate for a profile whose pool
the registry does not record as measured. 🟡 testing: full-model runs passed on cloud-hosted cards of that kind; kernel parity
qualification is still to run, and on other cards the profile counts as untested.

<details>
<summary><b>Flash-Next, Swift1.5, Swift1.5-Huihui</b>: 56 profiles, 4 tested, 5 testing</summary>

| Profile | Hardware and layout | Status | 262K sessions |
|---|---|---|---|
| `fn-sm80-64g-x4-pp4` | 4 x 64 GB sm_80 (CMP 170HX dual-rank), PP4 x TP1, bf16 KV, chunk 2048 | 🟡 testing | 17 (8 at once) |
| `fn-quad-3090` | 4 x RTX 3090 24 GB, P2P, TP2xPP2+EP (the default profile) | 🟢 tested | 3 |
| `fn-swift1.5-huihui-quad-3090` | swift1.5-huihui on 4 x RTX 3090 24 GB, TP2xPP2+EP | 🟢 tested | 3 |
| `fn-swift1.5-quad-3090` | swift1.5 on 4 x RTX 3090 24 GB, TP2xPP2+EP | 🟢 tested | 3 |
| `fn-quad-24g-sm80` | 4 x 24 GB sm_80 (A30), TP2xPP2+EP, bf16 KV | ⚫ untested | ~1 |
| `fn-swift1.5-huihui-quad-24g-sm80` | swift1.5-huihui: 4 x 24 GB sm_80 (A30), TP2xPP2+EP, bf16 KV | ⚫ untested | ~1 |
| `fn-swift1.5-quad-24g-sm80` | swift1.5: 4 x 24 GB sm_80 (A30), TP2xPP2+EP, bf16 KV | ⚫ untested | ~1 |
| `fn-sm80-32g-x3-pp3` | 3 x 32 GB sm_80 (CMP 170HX single-rank), PP3 x TP1, bf16 KV | ⚫ untested | ~1 |
| `fn-swift1.5-huihui-sm80-32g-x3-pp3` | swift1.5-huihui: 3 x 32 GB sm_80 (CMP 170HX single-rank), PP3 x TP1, bf16 KV | ⚫ untested | ~1 |
| `fn-swift1.5-sm80-32g-x3-pp3` | swift1.5: 3 x 32 GB sm_80 (CMP 170HX single-rank), PP3 x TP1, bf16 KV | ⚫ untested | ~1 |
| `fn-sm80-32g-x4-pp4` | 4 x 32 GB sm_80 (CMP 170HX single-rank), PP4 x TP1, bf16 KV | ⚫ untested | ~4 |
| `fn-swift1.5-huihui-sm80-32g-x4-pp4` | swift1.5-huihui: 4 x 32 GB sm_80 (CMP 170HX single-rank), PP4 x TP1, bf16 KV | ⚫ untested | ~4 |
| `fn-swift1.5-sm80-32g-x4-pp4` | swift1.5: 4 x 32 GB sm_80 (CMP 170HX single-rank), PP4 x TP1, bf16 KV | ⚫ untested | ~4 |
| `fn-sm80-40g-x4-pp4` | 4 x 40 GB sm_80 (CMP 170HX 10 GB SKU / A100 40 GB), PP4 x TP1, bf16 KV | ⚫ untested | ~7 |
| `fn-swift1.5-huihui-sm80-40g-x4-pp4` | swift1.5-huihui: 4 x 40 GB sm_80 (CMP 170HX 10 GB SKU / A100 40 GB), PP4 x TP1, bf16 KV | ⚫ untested | ~7 |
| `fn-swift1.5-sm80-40g-x4-pp4` | swift1.5: 4 x 40 GB sm_80 (CMP 170HX 10 GB SKU / A100 40 GB), PP4 x TP1, bf16 KV | ⚫ untested | ~7 |
| `fn-sm80-64g-x2-pp2` | 2 x 64 GB sm_80 (CMP 170HX dual-rank), PP2 x TP1, bf16 KV | 🟡 testing | ~6 |
| `fn-swift1.5-huihui-sm80-64g-x2-pp2` | swift1.5-huihui: 2 x 64 GB sm_80 (CMP 170HX dual-rank), PP2 x TP1, bf16 KV | ⚫ untested | ~6 |
| `fn-swift1.5-sm80-64g-x2-pp2` | swift1.5: 2 x 64 GB sm_80 (CMP 170HX dual-rank), PP2 x TP1, bf16 KV | 🟡 testing | ~6 |
| `fn-sm80-80g-x1` | 1 x 80 GB sm_80 (A100), single card, bf16 KV | ⚫ untested | ~1 |
| `fn-swift1.5-huihui-sm80-80g-x1` | swift1.5-huihui: 1 x 80 GB sm_80 (A100), single card, bf16 KV | ⚫ untested | ~1 |
| `fn-swift1.5-sm80-80g-x1` | swift1.5: 1 x 80 GB sm_80 (A100), single card, bf16 KV | ⚫ untested | ~1 |
| `fn-sm80-80g-x2-pp2` | 2 x 80 GB sm_80 (A100) without P2P, PP2 x TP1, bf16 KV | ⚫ untested | ~10 |
| `fn-sm80-80g-x2-tp2` | 2 x 80 GB sm_80 (A100), TP2+EP, bf16 KV | ⚫ untested | ~11 |
| `fn-swift1.5-huihui-sm80-80g-x2-pp2` | swift1.5-huihui: 2 x 80 GB sm_80 (A100) without P2P, PP2 x TP1, bf16 KV | ⚫ untested | ~10 |
| `fn-swift1.5-huihui-sm80-80g-x2-tp2` | swift1.5-huihui: 2 x 80 GB sm_80 (A100), TP2+EP, bf16 KV | ⚫ untested | ~11 |
| `fn-swift1.5-sm80-80g-x2-pp2` | swift1.5: 2 x 80 GB sm_80 (A100) without P2P, PP2 x TP1, bf16 KV | ⚫ untested | ~10 |
| `fn-swift1.5-sm80-80g-x2-tp2` | swift1.5: 2 x 80 GB sm_80 (A100), TP2+EP, bf16 KV | ⚫ untested | ~11 |
| `fn-sm80-80g-x4-tp2pp2` | 4 x 80 GB sm_80 (A100), TP2xPP2+EP, bf16 KV | ⚫ untested | ~28 |
| `fn-swift1.5-huihui-sm80-80g-x4-tp2pp2` | swift1.5-huihui: 4 x 80 GB sm_80 (A100), TP2xPP2+EP, bf16 KV | ⚫ untested | ~28 |
| `fn-swift1.5-sm80-80g-x4-tp2pp2` | swift1.5: 4 x 80 GB sm_80 (A100), TP2xPP2+EP, bf16 KV | ⚫ untested | ~28 |
| `fn-quad-3090-nop2p` | 4 x RTX 3090 24 GB, no P2P, the same TP2xPP2+EP launcher; boot-checked with `--swarm` on 4× RTX 3090 without P2P, 2026-11-01 | ⚫ untested | ~3 |
| `fn-quad-3090-nop2p-pp4` | 4 x RTX 3090 24 GB, no P2P, PP4 x TP1 (prefill alternative; TP2xPP2 stays the default) | 🟢 tested | 3 |
| `fn-swift1.5-huihui-quad-3090-nop2p` | swift1.5-huihui on 4 x RTX 3090 24 GB, no P2P, same layout, pin and partition as fn-quad-3090-nop2p | ⚫ untested | ~3 |
| `fn-swift1.5-quad-3090-nop2p` | swift1.5 on 4 x RTX 3090 24 GB, no P2P, same layout, pin and partition as fn-quad-3090-nop2p | ⚫ untested | ~3 |
| `fn-octo-3090-tp2pp4` | 8 x RTX 3090 24 GB, TP2xPP4+EP, max-num-seqs 32 (octo default; P2P on or off) | ⚫ untested | ~17 |
| `fn-octo-3090-tp4pp2` | 8 x RTX 3090 24 GB, TP4xPP2+EP, max-num-seqs 20 (latency alternative; P2P on or off) | ⚫ untested | ~11 |
| `fn-2x48g-pp2-sm86` | 2 x 48 GB sm_86 (RTX A6000 / A40), PP2 x TP1, fp8 KV | ⚫ untested | ~5 |
| `fn-swift1.5-2x48g-pp2-sm86` | swift1.5: 2 x 48 GB sm_86 (RTX A6000 / A40), PP2 x TP1, fp8 KV | ⚫ untested | ~5 |
| `fn-swift1.5-huihui-2x48g-pp2-sm86` | swift1.5-huihui: 2 x 48 GB sm_86 (RTX A6000 / A40), PP2 x TP1, fp8 KV | ⚫ untested | ~5 |
| `fn-4x48g-tp2pp2-sm86` | 4 x 48 GB sm_86 (RTX A6000 / A40), TP2xPP2+EP, fp8 KV | ⚫ untested | ~25 |
| `fn-swift1.5-4x48g-tp2pp2-sm86` | swift1.5: 4 x 48 GB sm_86 (RTX A6000 / A40), TP2xPP2+EP, fp8 KV | ⚫ untested | ~25 |
| `fn-swift1.5-huihui-4x48g-tp2pp2-sm86` | swift1.5-huihui: 4 x 48 GB sm_86 (RTX A6000 / A40), TP2xPP2+EP, fp8 KV | ⚫ untested | ~25 |
| `fn-quad-24g-sm89` | 4 x 24 GB sm_89 (RTX 4090), TP2xPP2+EP, fp8 KV | 🟡 testing | ~3 |
| `fn-quad-24g-sm89-pp4` | 4 x 24 GB sm_89 (RTX 4090), no P2P, PP4 x TP1, fp8 KV (prefill alternative) | ⚫ untested | ~1 |
| `fn-swift1.5-huihui-quad-24g-sm89` | swift1.5-huihui on 4 x 24 GB sm_89 (RTX 4090), same layout, pin and partition as fn-quad-24g-sm89 | ⚫ untested | ~3 |
| `fn-swift1.5-quad-24g-sm89` | swift1.5 on 4 x 24 GB sm_89 (RTX 4090), same layout, pin and partition as fn-quad-24g-sm89 | 🟡 testing | ~3 |
| `fn-quad-32g-sm89` | 4 x 32 GB sm_89 (modified RTX 4080 Super 32 GB), TP2xPP2+EP, fp8 KV | ⚫ untested | ~10 |
| `fn-swift1.5-huihui-quad-32g-sm89` | swift1.5-huihui on 4 x 32 GB sm_89 (RTX 4080 32 GB class), same layout, pin and partition as fn-quad-32g-sm89 | ⚫ untested | ~10 |
| `fn-swift1.5-quad-32g-sm89` | swift1.5 on 4 x 32 GB sm_89 (RTX 4080 32 GB class), same layout, pin and partition as fn-quad-32g-sm89 | ⚫ untested | ~10 |
| `fn-2x48g-pp2-sm89` | 2 x 48 GB sm_89 (RTX 6000 Ada, L40S), PP2 x TP1, fp8 KV | ⚫ untested | ~5 |
| `fn-swift1.5-2x48g-pp2-sm89` | swift1.5 on 2 x 48 GB sm_89 (RTX 4090 48 GB class), same layout, pin and partition as fn-2x48g-pp2-sm89 | ⚫ untested | ~5 |
| `fn-swift1.5-huihui-2x48g-pp2-sm89` | swift1.5-huihui on 2 x 48 GB sm_89 (RTX 4090 48 GB class), same layout, pin and partition as fn-2x48g-pp2-sm89 | ⚫ untested | ~5 |
| `fn-4x48g-tp2pp2-sm89` | 4 x 48 GB sm_89 (RTX 6000 Ada, L40S), TP2xPP2+EP, fp8 KV | ⚫ untested | ~25 |
| `fn-swift1.5-4x48g-tp2pp2-sm89` | swift1.5 on 4 x 48 GB sm_89 (RTX 4090 48 GB class), same layout, pin and partition as fn-4x48g-tp2pp2-sm89 | ⚫ untested | ~25 |
| `fn-swift1.5-huihui-4x48g-tp2pp2-sm89` | swift1.5-huihui on 4 x 48 GB sm_89 (RTX 4090 48 GB class), same layout, pin and partition as fn-4x48g-tp2pp2-sm89 | ⚫ untested | ~25 |

</details>

<details>
<summary><b>Qwen3.8-27B, ThinkingCap</b>: 32 profiles, 3 tested, 5 testing</summary>

| Profile | Hardware and layout | Status | 262K sessions |
|---|---|---|---|
| `27b-tp1-3090` | Qwen3.8-27B on 1 x RTX 3090 24 GB (TP1, with vision, 2 sequences, 4 MP images) | 🟢 tested | 1 |
| `27b-tp2-3090` | Qwen3.8-27B on 2 x RTX 3090 24 GB (TP2) | 🟢 tested | 3 |
| `27b-tp4-3090` | Qwen3.8-27B on 4 x RTX 3090 24 GB (TP4) | 🟢 tested | 7 |
| `27b-tp1-sm80-24g` | Qwen3.8-27B on 1 x A30 24 GB (sm_80, TP1), estimated pin | ⚫ untested | ~1 |
| `27b-tp2-sm80-24g` | Qwen3.8-27B on 2 x A30 24 GB (sm_80, TP2), estimated pin | ⚫ untested | ~3 |
| `27b-tp4-sm80-24g` | Qwen3.8-27B on 4 x A30 24 GB (sm_80, TP4), estimated pin | ⚫ untested | ~7 |
| `27b-tp1-sm80-32g` | Qwen3.8-27B on 1 x CMP 170HX 32 GB (sm_80, TP1), estimated pin | ⚫ untested | ~1 |
| `27b-tp2-sm80-32g` | Qwen3.8-27B on 2 x CMP 170HX 32 GB (sm_80, TP2), estimated pin | ⚫ untested | ~4 |
| `27b-tp4-sm80-32g` | Qwen3.8-27B on 4 x CMP 170HX 32 GB (sm_80, TP4), estimated pin | ⚫ untested | ~11 |
| `27b-tp1-sm80-40g` | Qwen3.8-27B on 1 x A100 40 GB (sm_80, TP1), estimated pin | ⚫ untested | ~2 |
| `27b-tp2-sm80-40g` | Qwen3.8-27B on 2 x A100 40 GB (sm_80, TP2), estimated pin | ⚫ untested | ~6 |
| `27b-tp4-sm80-40g` | Qwen3.8-27B on 4 x A100 40 GB (sm_80, TP4), estimated pin | ⚫ untested | ~14 |
| `27b-tp1-sm80-64g` | Qwen3.8-27B on 1 x CMP 170HX 64 GB (sm_80, TP1), estimated pin | 🟡 testing | ~5 |
| `27b-tp2-sm80-64g` | Qwen3.8-27B on 2 x CMP 170HX 64 GB (sm_80, TP2), estimated pin | 🟡 testing | ~11 |
| `27b-tp4-sm80-64g` | Qwen3.8-27B on 4 x CMP 170HX 64 GB (sm_80, TP4), estimated pin | 🟡 testing | ~25 |
| `27b-tp1-sm80-80g` | Qwen3.8-27B on 1 x A100 80 GB (sm_80, TP1), estimated pin | ⚫ untested | ~7 |
| `27b-tp2-sm80-80g` | Qwen3.8-27B on 2 x A100 80 GB (sm_80, TP2), estimated pin | ⚫ untested | ~15 |
| `27b-tp4-sm80-80g` | Qwen3.8-27B on 4 x A100 80 GB (sm_80, TP4), estimated pin | ⚫ untested | ~32 |
| `27b-tp2-3090-nop2p` | Qwen3.8-27B on 2 x RTX 3090 24 GB without P2P (TP2) | ⚫ untested | ~3 |
| `27b-tp4-3090-nop2p` | Qwen3.8-27B on 4 x RTX 3090 24 GB without P2P (TP4); boot-checked with `--swarm` on 4× RTX 3090 without P2P, 2026-11-01 | ⚫ untested | ~7 |
| `27b-tp1-sm86-48g` | Qwen3.8-27B on 1 x RTX A6000 / A40 48 GB (sm_86, TP1), estimated pin | ⚫ untested | ~3 |
| `27b-tp2-sm86-48g` | Qwen3.8-27B on 2 x RTX A6000 / A40 48 GB (sm_86, TP2), estimated pin | ⚫ untested | ~8 |
| `27b-tp4-sm86-48g` | Qwen3.8-27B on 4 x RTX A6000 / A40 48 GB (sm_86, TP4), estimated pin | ⚫ untested | ~18 |
| `27b-tp1-sm89-24g` | Qwen3.8-27B on 1 x RTX 4090 24 GB (sm_89, TP1), the 27b-tp1-3090 pins | ⚫ untested | ~1 |
| `27b-tp2-sm89-24g` | Qwen3.8-27B on 2 x RTX 4090 24 GB (sm_89, TP2), the 27b-tp2-3090 pins | 🟡 testing | ~3 |
| `27b-tp4-sm89-24g` | Qwen3.8-27B on 4 x RTX 4090 24 GB (sm_89, TP4), the 27b-tp4-3090 pins | 🟡 testing | ~7 |
| `27b-tp1-sm89-32g` | Qwen3.8-27B on 1 x RTX 4080 32 GB class (modified) (sm_89, TP1), the 27b-tp1-3090 pins | ⚫ untested | ~1 |
| `27b-tp2-sm89-32g` | Qwen3.8-27B on 2 x RTX 4080 32 GB class (modified) (sm_89, TP2), the 27b-tp2-3090 pins | ⚫ untested | ~3 |
| `27b-tp4-sm89-32g` | Qwen3.8-27B on 4 x RTX 4080 32 GB class (modified) (sm_89, TP4), the 27b-tp4-3090 pins | ⚫ untested | ~7 |
| `27b-tp1-sm89-48g` | Qwen3.8-27B on 1 x RTX 4090 48 GB class (modified) (sm_89, TP1), the 27b-tp1-3090 pins | ⚫ untested | ~1 |
| `27b-tp2-sm89-48g` | Qwen3.8-27B on 2 x RTX 4090 48 GB class (modified) (sm_89, TP2), the 27b-tp2-3090 pins | ⚫ untested | ~3 |
| `27b-tp4-sm89-48g` | Qwen3.8-27B on 4 x RTX 4090 48 GB class (modified) (sm_89, TP4), the 27b-tp4-3090 pins | ⚫ untested | ~7 |

</details>

<details>
<summary><b>Nemotron 3.5 Super VL</b>: 11 profiles, 1 tested, 2 testing</summary>

| Profile | Hardware and layout | Status | 262K sessions |
|---|---|---|---|
| `svl-quad-3090` | Nemotron Super VL 120B-A12B (W4A16 + INT4 MTP head) on 4 x RTX 3090 24 GB (TP2 x PP2 + EP, RecoverSSM) | 🟢 tested | 5 |
| `svl-quad-sm80-24g` | Nemotron Super VL 120B-A12B on 4 x A30 24 GB (sm_80), TP2 x PP2 + EP, RecoverSSM, estimated pin | ⚫ untested | ~5 |
| `svl-quad-sm80-32g` | Nemotron Super VL 120B-A12B on 4 x CMP 170HX 32 GB (sm_80), TP2 x PP2 + EP, RecoverSSM, estimated pin | ⚫ untested | ~21 |
| `svl-quad-sm80-40g` | Nemotron Super VL 120B-A12B on 4 x A100 40 GB (sm_80), TP2 x PP2 + EP, RecoverSSM, estimated pin | ⚫ untested | ~37 |
| `svl-quad-sm80-64g` | Nemotron Super VL 120B-A12B on 4 x CMP 170HX 64 GB (sm_80), TP2 x PP2 + EP, RecoverSSM, estimated pin | 🟡 testing | ~86 |
| `svl-quad-sm80-80g` | Nemotron Super VL 120B-A12B on 4 x A100 80 GB (sm_80), TP2 x PP2 + EP, RecoverSSM, estimated pin | ⚫ untested | ~119 |
| `svl-quad-3090-nop2p` | Nemotron Super VL 120B-A12B (W4A16 + INT4 MTP head) on 4 x RTX 3090 24 GB without P2P (TP2 x PP2 + EP, RecoverSSM) | ⚫ untested | ~5 |
| `svl-quad-sm86-48g` | Nemotron Super VL 120B-A12B on 4 x RTX A6000 / A40 48 GB (sm_86), TP2 x PP2 + EP, RecoverSSM, estimated pin | ⚫ untested | ~54 |
| `svl-quad-sm89-24g` | Nemotron Super VL 120B-A12B (W4A16 + INT4 MTP head) on 4 x RTX 4090 24 GB (sm_89), TP2 x PP2 + EP, RecoverSSM, the svl-quad-3090 pin | 🟡 testing | ~5 |
| `svl-quad-sm89-32g` | Nemotron Super VL 120B-A12B (W4A16 + INT4 MTP head) on 4 x RTX 4080 32 GB class (modified) (sm_89), TP2 x PP2 + EP, RecoverSSM, the svl-quad-3090 pin | ⚫ untested | ~5 |
| `svl-quad-sm89-48g` | Nemotron Super VL 120B-A12B (W4A16 + INT4 MTP head) on 4 x RTX 4090 48 GB class (modified) (sm_89), TP2 x PP2 + EP, RecoverSSM, the svl-quad-3090 pin | ⚫ untested | ~5 |

</details>

<details>
<summary><b>Ornith 1.5</b>: 2 profiles, 2 tested</summary>

| Profile | Hardware and layout | Status | 262K sessions |
|---|---|---|---|
| `ornith-quad-3090` | Ornith 1.5 122B-A10B (W4A16 + INT4 MTP experts) on 4 x RTX 3090 24 GB with P2P (TP2 x PP2 + EP, RecoverSSM) | 🟢 tested | 3 |
| `ornith-quad-3090-nop2p` | Ornith 1.5 122B-A10B (W4A16 + INT4 MTP experts) on 4 x RTX 3090 24 GB without P2P (TP2 x PP2 + EP, RecoverSSM) | 🟢 tested | 3 |

</details>

## 7. Switch or add a model

One server runs one model. To add another:

```bash
./setup.sh --model ornith1.5      # its own questions and folder; only its checkpoint is downloaded
./setup.sh --model swift1.5       # back again: saved answers, nothing to download
venv/bin/merlin pull 27b          # download and verify only, without serving
```

Answers are kept per model, so one model's folder, port and layout never apply to another.

## 8. When something is wrong

| Symptom | What to do |
|---|---|
| Setup stops with `FAIL` lines | Each line carries its fix; nothing was downloaded. Fix it and run `./setup.sh` again. |
| Unsure the host is ready | `venv/bin/merlin doctor` prints the hardware report: GPUs, P2P, driver, RAM, shared memory, GPU unlock state. |
| Unsure what will start | `venv/bin/merlin serve <model> --dry-run` prints the profile, cards, environment and command. |
| First start seems stuck | It compiles kernels first; wait for `Application startup complete`. The first 27B start after upgrading to 3.0.1 compiles again. |
| `/health` returns 503 | The engine has stalled for 120 s or hit an error; the log has it at ERROR. Restart the server. |
| A download broke off | Run `./setup.sh` again; downloads resume. |
| Checksum check failed | The message names the files: delete them and run `./setup.sh` again to fetch them anew. |
| Slow on cards without P2P | See [Choose your GPUs](#5-choose-your-gpus): several smaller servers may serve you better. |

Every option, the download and verification details, and where each file lives: [SETUP.md](SETUP.md).

## 9. Uninstall

Delete `.merlin/`, `venv/`, `.vllm-cache/` and the `.build-*` files in the checkout, the models folder
(`~/models/<model>` by default), and `~/.cache/uv` if nothing else uses uv. Nothing else was written: no packages,
services, shell profiles or system paths.
