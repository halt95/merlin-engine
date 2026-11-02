# container/ — Merlin 3.0.1

> **unqualified until qualified on hardware**

The image installs **this repository checkout**, built on vLLM **0.32.0**, through
[release/install-env.sh](../release/install-env.sh). That shared installer owns the 0.32
hash lock, CUDA toolchain, native build and [FlashInfer tile patch](../release/apply-flashinfer-patch.sh).
The image requires FlashInfer **0.7.0.post1**, humming-kernels **0.1.17** and instanttensor
**>= 0.2.2**. It does not fetch the retired engine bundle or 0.30 compiled artifacts.

The 0.32 installer, 0.7.0.post1 tile tooling and generated Swift/Huihui launchers
are integrated. `release/requirements-0.32.lock` is complete with 222 exact pins
and artifact SHA-256 hashes, resolved with uv; a dry-run `--require-hashes`
install resolves. A fresh install (2026-10-21) and a from-source build (2026-10-22) passed on the reference host;
the container image itself is not yet qualified. The build context is the repository root;
the image retains the checkout because the engine install is editable. The host's private `.git` directory is excluded.
The build fetches source history from `SOURCE_URL` (default: the public repository)
at `SOURCE_REF` (default: `merlin-v3.0.1`), checks that every copied tracked file matches
that ref, and lets the installer independently verify the vLLM 0.32.0 ancestry. That fetched
history stays in the image for subsequent source-identity verification.
Build from the matching clean ref. Use a Docker-tag-safe published tag or full commit SHA as
`SOURCE_REF`, since it also names the local image tag. For Compose, set `SOURCE_REF` in the host shell
or a Compose `.env` file; Compose passes the same value to the build. For direct builds, pass
`--build-arg SOURCE_REF="$SOURCE_REF"` and tag the image `merlin-engine:$SOURCE_REF`, as shown below.
For a custom source repository, also pass `--build-arg SOURCE_URL=<url>` to `docker build`.
The image bootstraps
[uv 0.12.13](https://github.com/astral-sh/uv/releases/tag/0.12.13) and installs the Rust
version selected by the checkout's `rust-toolchain.toml`.
Debian 13 supplies [GCC/G++ 14](https://packages.debian.org/trixie/g%2B%2B-14),
selected explicitly through `CC`, `CXX` and `CUDAHOSTCXX`. The CUDA source build
requires GCC/G++ >=13 and C++20 for sm_80/86/89, as enforced by this
checkout's `CMakeLists.txt`; GCC 14 is the image's selected compiler. The host
compiler must also be supported by the selected CUDA toolkit. The lock pins CUDA
CRT, nvcc, nvJitLink and NVVM to 13.0.88 (CCCL 13.0.85).

Reference host: NVIDIA driver 610.43.02 (reports CUDA 13.3). The release lock selects a matched CUDA 13.0 toolkit (nvcc, crt, nvvm, nvJitLink 13.0.88, cccl 13.0.85, runtime 13.0.96, the pair torch 2.13.0 cu130 ships with); the release wheel is served on it by every recipe (2026-10-20). Other CUDA 13.x drivers (R580 or later) are expected to work through CUDA minor-version compatibility, but are untested.
See the [driver requirements](../release/install-0.32.md).

The base is pinned to the multi-platform index digest of
[python:3.13-slim-trixie on Docker Hub](https://hub.docker.com/layers/library/python/3.13-slim-trixie/images/sha256-7f7f96867e2bf7bee2e2d11d5d46d65760f3eca347141a2f1410d98088b0e338?tab=layers),
looked up read-only on 2026-10-15:
```text
sha256:70729b46c69b4f1e97c4822c1af3df53a1476cf5ddc6c087c0c10bc3a5678c2f
```

From the repository root, with Docker Compose v2 and the NVIDIA Container Toolkit
configured on the host:

```bash
MODEL_DIR=/absolute/path/to/Qwen3.8-Flash-Next-W4A16-Merlin \
  docker compose -f container/docker-compose.yml up --build -d
```

The local image is named `merlin-engine:merlin-v3.0.1` by default, with its tag taken from
`SOURCE_REF`. Compose builds from `..` using `container/Dockerfile` and reserves all NVIDIA GPUs.
The default Flash-Next recipe
uses four RTX 3090s. An A100 run must choose a compatible model and parallel
configuration; the default four-card configuration is not an A100 qualification command.

Choose the checkpoint and canonical [recipe](../models/README.md) with environment:

| `MERLIN_MODEL` | Launcher | Devices in its default layout |
|---|---|---|
| `flash-next` (default) | `models/flash-next/serve.sh` | 4 GPUs |
| `swift1.5` | `models/swift1.5/serve.sh` | 4 GPUs |
| `swift1.5-huihui` | `models/swift1.5-huihui/serve.sh` | 4 GPUs |
| `qwen38-27b` | `models/qwen38-27b/serve.sh` | 2 GPUs |
| `thinkingcap` | `models/thinkingcap/serve.sh` | 2 GPUs |

```bash
MERLIN_MODEL=qwen38-27b MODEL_DIR=/absolute/path/to/Qwen3.8-27B-W4A16-Merlin \
  docker compose -f container/docker-compose.yml up --build -d
```

Both image-cap variables are set to **8,388,608 pixels** (8 MP): `IMG_CAP_PX` for the
Flash-Next family and `MAX_PIXELS` for the 27B family. Launchers own the 0.32 CLI flags
and `FLASHINFER_SM8X_FP8_SMALLQ`: the 27B launcher sets it to `cta32` by default (set it empty for stock
FlashInfer); the image installs the patch without enabling the tile globally. On unsupported architectures the patch keeps its stock path.

For direct Docker use:

```bash
SOURCE_REF=merlin-v3.0.1
docker build -f container/Dockerfile --build-arg SOURCE_REF="$SOURCE_REF" -t "merlin-engine:$SOURCE_REF" .
docker run --gpus all --stop-timeout 70 --shm-size=8g --ulimit memlock=-1 \
  -p 127.0.0.1:8000:8000 -v /absolute/path/to/checkpoint:/models/checkpoint \
  -v merlin-cache:/cache -e MERLIN_MODEL=flash-next "merlin-engine:$SOURCE_REF"
```

`CHECKPOINT_DIR` defaults to `/models/checkpoint`; an initial positional path overrides
it. Extra positional flags pass through to the Flash-Next family launchers. The 27B
family takes environment settings (`GPUS`, `KV_BYTES`, `SERVED`, etc.); extra flags are
rejected because those launchers do not forward them.

Compose forwards `MODEL_NAME`, `SERVED`, `GPUS`, `KV_BYTES` and `WARM` from the host shell
or a compose `.env` file (unset or empty keeps the launcher default). `KV_BYTES` applies to
Swift, Swift1.5-Huihui and the 27B recipes; Flash-Next sizes its own pool. `WARM=0` skips the
Flash-family warm-up and is the setting for exploratory non-sm_86 runs, for example
`WARM=0 MODEL_DIR=... docker compose -f container/docker-compose.yml up --build -d`.

Flash-Next family served names
are the launcher's defaults (`flash-next-v2` for Flash-Next and Swift1.5, `flash-next-swift-huihui` for Swift1.5-Huihui), with `MODEL_NAME` overriding them; the 27B and ThinkingCap use `SERVED` (default `qwen38-27b`).

The writable checkpoint mount lets the entrypoint prepare an older Flash-Next family
`config.json` with `models/flash-next/make-e1-config.py` (backup kept). Prepare that key
on the host first if using a read-only mount. No weights are changed.

HTTP listens on `0.0.0.0:8000` inside the container so Docker can forward to it. Both examples
publish port 8000 on the host's `127.0.0.1` only. Before publishing on a network address, set a
bearer key by uncommenting Compose's `VLLM_API_KEY` entry or passing `-e VLLM_API_KEY` to
`docker run` with the key exported in the host shell. Clients then send `Authorization: Bearer <key>`
for API requests. With no key the launcher prints an exposure warning at start, inside the container too:
it cannot see which host address the port is published on. The health check discovers a served name through `/v1/models`, then requires
HTTP 200 from both endpoints and a chat completion with a nonempty `choices`
array and an integer `usage.completion_tokens >= 1`, including the API key when set. It allows 30 minutes
for initial compilation. CUDA JIT tools remain in the image; `/cache` retains FlashInfer,
Triton and vLLM caches. Host RAM and GPU capacity must satisfy the selected recipe.
`MAX_JOBS` defaults to 4; the build targets sm_80, sm_86 and sm_89.

On the 27B, the installer's `--check <venv> <server-log> [q_len] --tp N` mode must also be run
after a tile-enabled MTP request, using the current boot's log, for the live tile proof. The tile
proof does not apply to the other models, including the Flash-Next family: check those with `--verify`.

The CPU readiness regression suite can be run with:

```bash
venv/bin/python -m unittest discover -s container -p test_healthcheck.py -v
```

No container build, regression suite or GPU boot has been run for this recipe. Hardware qualification must
record a successful build, model load, `/v1/models`, one-token completion, needle probes
and image size on a supported GPU configuration before this label can be removed.
