# Merlin 3.0.2 install and release tooling

Merlin 3.0.2 builds this checkout on **vLLM 0.33.0**. All recipes under
[`models/`](../models/README.md) share one environment installer and dependency
lock. Engine bundles and per-model engine patch series are historical artifacts.

**The installer and the dependency lock are integrated.**
[`requirements-0.33.lock`](requirements-0.33.lock) holds the exact versions of the
qualified environment with artifact SHA-256 hashes; the closure resolves on
CPython 3.13 / Linux x86_64 with glibc 2.31 or newer (llguidance 1.7.6 ships its x86_64 wheel
only as manylinux_2_31; every other locked wheel needs glibc 2.28 or older). The lock alone needs 2.31; the release engine
wheel's compiled library needs glibc 2.38, so running Merlin from the wheel needs 2.38 or newer. Install with `--require-hashes`.

| Files | Current behavior |
|---|---|
| [`install-env.sh`](install-env.sh), [`install-0.33.md`](install-0.33.md) | Creates a new Python 3.13 venv with uv, installs the single hash lock, applies the FlashInfer tile patch and installs this checkout editable with no dependency resolution or build isolation, taking the compiled engine from the release wheel (`MERLIN_WHEEL`) or building it for sm_80, sm_86 and sm_89. Verifies the checkout's commit and tree identity, source/import identity, dependency versions and native product hashes. Creates CUDA library links and the launcher build-product markers. `--verify` checks an existing install; `--check` adds live tile selection proof from a server log. |
| [`requirements-0.33.lock`](requirements-0.33.lock) | Exact serving, CUDA toolkit and build-tool pins, including torch 2.14.0, triton 3.8.0, FlashInfer 0.7.2.post1, humming-kernels 0.1.19, Pillow 13.0.0, Transformers 5.20.2 and instanttensor 0.2.2. Every package carries artifact SHA-256 hashes; the closure resolves on CPython 3.13 / Linux x86_64. Excludes vllm itself and flashinfer-jit-cache. |
| [`build-wheel.sh`](build-wheel.sh), [`wheel.py`](wheel.py), `merlin-wheel.sha256` | The engine wheel: one build for sm_80, sm_86 and sm_89 from a clean HEAD against the release lock, with build-host paths mapped out and audited (`wheel.py audit`). `merlin-wheel.sha256` names the one wheel a checkout accepts; `install-env.sh` refuses any other wheel, and one whose commit's native sources differ from the checkout's. |
| [`apply-flashinfer-patch.sh`](apply-flashinfer-patch.sh), [`check-tile-proof.sh`](check-tile-proof.sh), [`patches/flashinfer-0.7.2.post1/`](patches/flashinfer-0.7.2.post1/README.md) | Version-specific 0.7.2.post1 small-query FP8 verify tile (a port of the 0.7.0.post1 series, kept under `patches/flashinfer-0.7.0.post1/` for history). Checks pristine/patched hashes, serializes idempotent application and uses JIT suffix `_sm8x_smallq_072_v1`. Installation leaves the knob off; launchers select it with `FLASHINFER_SM8X_FP8_SMALLQ=cta32`. The proof checker requires a current-boot selection line for every TP rank after a completed MTP request; it does not prove numerical correctness. |
| [`apply-flashinfer-workload-patch.py`](apply-flashinfer-workload-patch.py) | Opt-in, 27B only, default off: with `MERLIN_FLASHINFER_WORKLOAD_TILE=1`, `install-env.sh` applies FlashInfer patch 0002 (workload-aware tile) after 0001; see the [patch notes](patches/flashinfer-0.7.2.post1/README.md) |
| [`make_manifest.py`](make_manifest.py) | Generates `PACKAGE-MANIFEST.md` and `SHA256SUMS` from all staged Git blobs and modes, including each recipe's checkpoint checksum list. Generate these on the final release tree after cleanup; downloaded checkpoint bytes are verified separately. |

Use Linux x86_64, Python 3.13 headers, uv, Git (a clean checkout; a shallow clone is fine for installing and for building
the wheel, which is named after the checkout's commit),
Cargo/Rust as selected by `rust-toolchain.toml`, CMake and ninja. Native CUDA
ops require **GCC/G++ >=13 and C++20**, as enforced by this checkout's
`CMakeLists.txt`; use a host compiler supported by the selected CUDA toolkit.
The [container](../container/README.md) selects GCC/G++ 14. These requirements
apply to sm_80, sm_86 and sm_89; native SASS coverage remains kernel-specific.
The lock pins the CUDA CRT, nvcc, nvJitLink and NVVM wheels to **13.0.88** (CCCL 13.0.85),
matching torch's CUDA 13.0 build.
Reference host: NVIDIA driver 610.43.02 (reports CUDA 13.3). The release lock selects a matched CUDA 13.0 toolkit (nvcc, crt, nvvm, nvJitLink 13.0.88, cccl 13.0.85, runtime 13.0.96, the pair torch 2.14.0 cu130 ships with); the 3.0.0 release wheel was served on it by every recipe (2026-10-20); the 3.0.2 wheel was served on it on 2026-11-04 by Flash-Next, Qwen3.8-27B (1, 2 and 4 GPUs), Nemotron 3.5 Super VL and Ornith 1.5, each with a 262K request. Other CUDA 13.x drivers (R580 or later) are expected to work through CUDA minor-version compatibility, but are untested.
See the [installation requirements](install-0.33.md).

The lock is complete: 223 exact pins with SHA-256 hashes, resolved with uv, and
a dry-run `--require-hashes` install resolves. From the repository root:

```bash
curl -LO https://github.com/halt95/merlin-engine/releases/download/merlin-v3.0.2/vllm-0.33.0+merlin.665f43d2cf-cp313-cp313-linux_x86_64.whl
MERLIN_WHEEL="$PWD/vllm-0.33.0+merlin.665f43d2cf-cp313-cp313-linux_x86_64.whl"   bash release/install-env.sh "$PWD/venv"
bash release/install-env.sh --verify "$PWD/venv"
```

Without `MERLIN_WHEEL` the installer compiles the engine (about 1.5 hours on 8 cores).
`release/build-wheel.sh` builds the wheel itself.

A fresh install from the 3.0.0 release wheel on the 0.32 lock was verified on the reference host and on
a second host with 4× RTX 4090 (2026-10-20); a fresh install of the 3.0.2 wheel on this lock has not been done; the container image is not yet qualified. A tile
hash check or import check does not replace a successful model request and the
separate live tile proof. See [install-0.33.md](install-0.33.md) for the full
procedure.

Checkpoint checksum lists live at `models/<model>/checkpoint.sha256`. Run
`sha256sum -c /path/to/checkout/models/<model>/checkpoint.sha256` inside the
downloaded checkpoint directory. Historical engine bundles, combined diffs,
per-model pin sets and package manifests remain at their original release tags.
