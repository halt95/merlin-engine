# Install Merlin on vLLM 0.32.0

Use a fresh checkout of this repository and a new venv. The installer builds the
CUDA extensions and installs the checkout editable; it never fetches an engine
bundle, applies an engine patch series, or installs a stock vLLM wheel.

The dependency lock is complete: 222 exact pins with artifact SHA-256 hashes for
CPython 3.13 on Linux x86_64 (manylinux_2_28), resolved with uv. A dry-run
`--require-hashes` install resolves; the installer uses that same hash check.
The lock includes serving and build dependencies: torch 2.13.0, triton 3.7.1,
FlashInfer 0.7.0.post1, humming-kernels 0.1.17 and instanttensor 0.2.2.
The CUDA CRT, nvcc, nvJitLink and NVVM wheels are pinned to 13.0.88 and CCCL to 13.0.85,
matching runtime 13.0.96 and torch's CUDA 13.0 (cu130) build, so compiler, headers and
torch share one CUDA release (see "Why 13.0" below).
vllm itself and flashinfer-jit-cache are excluded.

Prerequisites: Linux x86_64, uv, Git with full checkout history, GCC/G++ >=13
(the installer checks this before installing anything; the reference host uses GCC 14) and C++20 for native ops (sm_80/86/89), Cargo/Rust, Python 3.13 development
headers, and a CUDA 13.x NVIDIA driver (R580 or later).

Reference host: NVIDIA driver 610.43.02 (reports CUDA 13.3). The release lock selects a matched CUDA 13.0 toolkit (nvcc, crt, nvvm, nvJitLink 13.0.88, cccl 13.0.85, runtime 13.0.96, the pair torch 2.13.0 cu130 ships with); the release wheel is served on it by every recipe (2026-10-20). Other CUDA 13.x drivers (R580 or later) are expected to work through CUDA minor-version compatibility, but are untested.
The reference host reports CUDA UMD Version 13.3 in `nvidia-smi`.

### Why the toolkit wheels are 13.0 (qualification record, 2026-10-17)

The first lock copied the reference host's venv, which carried nvcc/crt/nvvm/nvJitLink
13.4.92 and CCCL 13.3 beside runtime/headers 13.0.96 (`CUDA_VERSION 13000`). torch 2.13.0
requires `cuda-toolkit==13.0.3`, whose `nvcc` extra is 13.0.88, so nothing required 13.4.
That mix fails: FlashInfer's bundled CCCL (`cuda/std/__cccl/cuda_toolkit.h:41`) stops every
JIT compile with "CUDA compiler and CUDA toolkit headers are incompatible" (nvcc 13.4 vs
CUDART 13.0). Builds then had to set `CUDA_HOME=/usr/local/cuda` (13.3.73) to avoid it; that override is now accepted only with a warning and is untested. The lock
now holds one CUDA release: nvcc, crt, nvvm and nvJitLink 13.0.88, CCCL 13.0.85, runtime
13.0.96. A 13.0 compiler is also below the reference driver's CUDA 13.3, so its PTX loads.

Qualification status of this locked route on 2026-10-17: **compile-qualified on CPU, GPU serve pending.**
On a CPU-only host, a venv built from this lock with `--require-hashes` compiled the
FlashInfer 0.7.0.post1 JIT modules the served configurations build (batch prefill with FP8 E4M3 KV at
head dim 128 and 256, and sampling) for sm_80, sm_86 and sm_89, and vLLM's Marlin and
CUTLASS 2.x kernel sources for each of the three. The ReplaySSM `checkpointing_ssu`
module does not compile for sm_86 on any toolkit (13.3 included); it needs sm_90.
No GPU boot, first-request JIT run, or full vLLM native build had been done on this
lock by then; the 2026-10-20 release-wheel installs below were then served on it.

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

The FlashInfer 0.7.0.post1 tile tooling is integrated. From the repository root:

```bash
release/install-env.sh "$PWD/venv"
```

The optional second argument selects another existing checkout of this repo.
Before building, the installer requires a clean tracked tree (a shallow clone is
fine). The selected commit and tree are recorded in `.build-source`; the version
override is applied only after this independent provenance check.
The installer refuses an existing venv, inherited precompiled-wheel switches,
`CMAKE_ARGS`, and native dependency `*_SRC_DIR` / `FETCHCONTENT_*` overrides.
It installs the hash-locked dependencies with uv, checks the required
versions, calls `apply-flashinfer-patch.sh <venv>`, and builds editable with
`--no-deps --no-build-isolation`. It checks the actual vllm import path and 0.32.0
version with Python's working-directory and user-site imports disabled.
It also creates the CUDA `lib64` links and the `.build-products` /
`.build-products.sha256` files consumed by the model launchers.

A failed install leaves the new venv and any build outputs for diagnosis.
Choose a fresh venv path for a retry. Existing source checkouts are never deleted.

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

A fresh install from the release wheel was verified on the reference host and on a second host with 4× RTX 4090 (2026-10-20); the container image is not yet qualified.
