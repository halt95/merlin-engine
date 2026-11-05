# Install Merlin on vLLM 0.33.0

Use a fresh checkout of this repository and a new venv. The installer builds the
CUDA extensions and installs the checkout editable; it never fetches an engine
bundle, applies an engine patch series, or installs a stock vLLM wheel.

The dependency lock is complete: 223 exact pins with artifact SHA-256 hashes for
CPython 3.13 on Linux x86_64 with glibc 2.31 or newer (llguidance 1.7.6 ships its x86_64
wheel only as manylinux_2_31; every other locked wheel needs glibc 2.28 or older), resolved with uv.
The lock alone resolves on glibc 2.31; the release engine wheel's compiled library needs glibc 2.38,
so running Merlin from the wheel needs glibc 2.38 or newer. A dry-run
`--require-hashes` install resolves; the installer uses that same hash check.
The lock includes serving and build dependencies: torch 2.14.0, triton 3.8.0,
FlashInfer 0.7.2.post1, humming-kernels 0.1.19 and instanttensor 0.2.2.
The CUDA CRT, nvcc, nvJitLink and NVVM wheels are pinned to 13.0.88 and CCCL to 13.0.85,
matching runtime 13.0.96 and torch's CUDA 13.0 (cu130) build, so compiler, headers and
torch share one CUDA release (see "Why 13.0" below).
vllm itself and flashinfer-jit-cache are excluded.

Prerequisites: Linux x86_64, uv, Git (a clean checkout of this repository; a shallow clone is fine for installing and for
building the wheel), GCC/G++ >=13
(the installer checks this before installing anything; the reference host uses GCC 14) and C++20 for native ops (sm_80/86/89), Cargo/Rust, Python 3.13 development
headers, and a CUDA 13.x NVIDIA driver (R580 or later).

Reference host: NVIDIA driver 610.43.02 (reports CUDA 13.3). The release lock selects a matched CUDA 13.0 toolkit (nvcc, crt, nvvm, nvJitLink 13.0.88, cccl 13.0.85, runtime 13.0.96, the pair torch 2.14.0 cu130 ships with); the 3.0.0 release wheel (vLLM 0.32.0) was served on it by every recipe (2026-10-20); the 3.0.2 wheel was served on it on 2026-11-04 by Flash-Next, Qwen3.8-27B (1, 2 and 4 GPUs), Nemotron 3.5 Super VL and Ornith 1.5, each with a 262K request. Other CUDA 13.x drivers (R580 or later) are expected to work through CUDA minor-version compatibility, but are untested.
The reference host reports CUDA UMD Version 13.3 in `nvidia-smi`.

### Why the toolkit wheels are 13.0

torch 2.14.0 requires `cuda-toolkit==13.0.3`, whose `nvcc` extra is 13.0.88. A newer compiler
beside the 13.0 runtime and headers (`CUDA_VERSION 13000`) fails: FlashInfer's bundled CCCL
(`cuda/std/__cccl/cuda_toolkit.h`) stops every JIT compile with "CUDA compiler and CUDA toolkit
headers are incompatible". Overriding `CUDA_HOME` with another toolkit is accepted only with a
warning and is untested. The lock holds one CUDA release: nvcc, crt, nvvm and nvJitLink 13.0.88, CCCL 13.0.85, runtime
13.0.96. A 13.0 compiler is also below the reference driver's CUDA 13.3, so its PTX loads.

Qualification status of this locked route (vLLM 0.33.0, torch 2.14.0, FlashInfer 0.7.2.post1): at the 3.0.2 release
gate Flash-Next, Qwen3.8-27B (1, 2 and 4 GPUs), Nemotron 3.5 Super VL and Ornith 1.5 booted and served on it
(2026-11-04), building their FlashInfer JIT modules on sm_86, and the live tile proof passed on the 27B TP2 boot; a
fresh install from the release wheel has not been done. The ReplaySSM `checkpointing_ssu` module does not compile for sm_86 on any toolkit; it needs sm_90.
For the previous route (vLLM 0.32.0, FlashInfer 0.7.0.post1), a CPU-only venv built from its lock with
`--require-hashes` compiled the served configurations' FlashInfer JIT modules for sm_80, sm_86 and sm_89 (2026-10-17),
and release-wheel installs were then served on it (2026-10-20).

Select a host compiler supported by the toolkit; the container uses GCC/G++ 14.
The locked wheels provide nvcc. An inherited `CUDA_HOME` replaces them; CUDA 13.0 is the tested
toolkit and passes silently. Any other 13.x (13.3 and 13.4 included) is accepted with a warning and is
untested: a compiler newer than the 13.0 runtime headers can fail FlashInfer's CCCL version check.
A non-13 toolkit (12.x, 14.x) is refused. Unset `CUDA_HOME` to use the locked toolkit. An external
toolkit needs its own build and hardware qualification.
Compilation covers sm_80, sm_86 and sm_89 (`TORCH_CUDA_ARCH_LIST=8.0;8.6;8.9`); `MAX_JOBS`
limits build concurrency. With `MERLIN_WHEEL` set to the release wheel the installer
compiles nothing: it checks the wheel against `release/merlin-wheel.sha256` and takes the
compiled engine from it.

The FlashInfer 0.7.2.post1 tile tooling is integrated. From the repository root:

```bash
release/install-env.sh "$PWD/venv"
```

The venv lives inside the checkout, at `venv/`. `merlin serve`, `setup.sh` and the
model launchers use `$TREE/venv` unless `VENV=<path>` is set on every launch, so
install it there. `.gitignore` ignores `venv` whether it is a directory or a symlink.

The optional second argument selects another existing checkout of this repo.
Before building, the installer requires a clean tracked tree (a shallow clone is
fine). The selected commit and tree are recorded in `.build-source`; the version
override is applied only after this independent provenance check.
The installer refuses an existing venv, inherited precompiled-wheel switches,
`CMAKE_ARGS`, and native dependency `*_SRC_DIR` / `FETCHCONTENT_*` overrides.
It installs the hash-locked dependencies with uv, checks the required
versions, calls `apply-flashinfer-patch.sh <venv>`, and builds editable with
`--no-deps --no-build-isolation`. It checks the actual vllm import path and 0.33.0
version with Python's working-directory and user-site imports disabled.
It also creates the CUDA `lib64` links and the `.build-products` /
`.build-products.sha256` files consumed by the model launchers.

A failed install leaves the new venv and any build outputs for diagnosis.
For a retry, move the failed `venv` aside and run the installer again. Existing
source checkouts are never deleted.

The live tile proof is a second step for the **27B lane only** because it needs a
completed tile-enabled MTP generation, a checkpoint and GPUs. Installation prints
`LIVE TILE PROOF PENDING (27B only)`; it does not claim a live tile pass from import
or patch hashes. Non-27B lanes should use `--verify`: this proof checks the 27B
FlashInfer FP8 small-q decode tile (`FLASHINFER_SM8X_FP8_SMALLQ`), which Flash-Next's
QSA attention path does not exercise. Launch the 27B TP2 recipe with the newly
installed venv and save the entire boot log:

```bash
MODEL=/path/to/checkpoint VENV="$PWD/venv" TREE="$PWD" \
  FLASHINFER_SM8X_FP8_SMALLQ=cta32 models/qwen38-27b/serve.sh >server.log 2>&1
```

After a generation request completes, run in another shell from this checkout:

```bash
release/install-env.sh --check "$PWD/venv" server.log 5 --tp 2
```

For a read-only check without a running server:

```bash
release/install-env.sh --verify "$PWD/venv"
```

`--verify` rechecks the source identity, every dependency pin, exact installed
tile hashes, import locations, dependency consistency and native product hashes.
It refuses a pristine, missing or edited tile without changing installed files.
This check does not establish live tile selection. Neither `--verify` nor `--check`
requires uv on PATH: dependency consistency uses uv when available, otherwise the
venv's Python checks installed package metadata and requirement markers with the
locked `packaging` library (pip need not be installed). Installation still requires uv.

Use q_len 4 or 5 and the server's actual TP size. The log must belong to this
installation and its current boot. `--check` performs the same read-only
verification as `--verify`, then delegates to
`check-tile-proof.sh <server-log> [q_len] --tp N`. It prints
`POST-INSTALL SELF-CHECK OK` only if every check passes. The tile knob remains
off unless a launcher explicitly sets it.

The proof checker requires one complete planner selection line per device in the
current boot. This proves tile selection; it is not a numerical correctness test
or proof that a kernel completed. Capture the log after a successful request.

A fresh install from the 3.0.0 release wheel (vLLM 0.32.0) was verified on the reference host and on a second host with 4× RTX 4090 (2026-10-20); a fresh install of the vLLM 0.33.0 wheel has not been done; the container image is not yet qualified.
