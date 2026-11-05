#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
# Install this Merlin checkout on vLLM 0.33.0 (Linux x86_64, CPython 3.13).
#   release/install-env.sh <new-venv-path> [engine-source-path]
# With MERLIN_WHEEL=<path to the release wheel> the compiled engine comes from that wheel (no
# native build); release/merlin-wheel.sha256 in the engine checkout must name it, and the checkout's
# native-source tree ids must match release/merlin-wheel.native.
# Verify an existing environment without serving or changing installed files:
#   release/install-env.sh --verify <venv-path> [engine-source-path]
# Opt-in: MERLIN_FLASHINFER_WORKLOAD_TILE=1 installs and verifies the FlashInfer
# 0001+0002 series (workload-aware tile; FLASHINFER_WHEEL=<pinned wheel> keeps it offline).
# Unset or 0 keeps the default 0001 tile exactly as before. Use the same value for --verify.
# 27B only: after serving a tile-enabled MTP request with this venv:
#   release/install-env.sh --check <venv-path> <server-log> [q_len] --tp N
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
REQ="$REPO/release/requirements-0.33.lock"
FLASHINFER_PATCH="$REPO/release/apply-flashinfer-patch.sh"
TILE_PROOF="$REPO/release/check-tile-proof.sh"
die() { echo "install-env: $*" >&2; exit 1; }

MODE=install
case "${1:-}" in
  --check) MODE=check; shift ;;
  --verify) MODE=verify; shift ;;
esac
VENV="${1:?usage: install-env.sh <new-venv-path> [engine-source-path] | --verify <venv-path> [engine-source-path] | --check <venv-path> <server-log> [q_len] --tp N}"
shift
if [ "$MODE" = check ]; then
  [ "$#" -ge 3 ] || die "--check needs <server-log> [q_len] --tp N"
  ENGINE="$REPO"
else
  [ "$#" -le 1 ] || die "too many arguments"
  ENGINE="${1:-$REPO}"
fi
ENGINE="$(cd "$ENGINE" && pwd -P)"
case "$VENV" in /*) ;; *) VENV="$PWD/$VENV" ;; esac
[ -f "$ENGINE/setup.py" ] && [ -f "$ENGINE/vllm/__init__.py" ] \
  || die "$ENGINE is not a Merlin-engine source checkout"
[ -r "$REQ" ] || die "missing $REQ"
if grep -nE 'TODO-(PIN|HASH)' "$REQ" >&2; then
  die "requirements-0.33.lock contains unresolved pins or hashes; restore the complete release lock"
fi
[ -r "$FLASHINFER_PATCH" ] && [ -r "$TILE_PROOF" ] \
  || die "missing FlashInfer tile tooling"

WHEEL="${MERLIN_WHEEL:-}"
[ -z "$WHEEL" ] || [ "$MODE" = install ] || die "MERLIN_WHEEL applies to an install only"
# The engine parses the same knob strictly (vllm/merlin/envs.py): exactly 0 or 1, or unset.
case "${MERLIN_FLASHINFER_WORKLOAD_TILE-0}" in
  0) TILE_SERIES=v1 ;;
  1) TILE_SERIES=v2 ;;
  *) die "MERLIN_FLASHINFER_WORKLOAD_TILE must be exactly 0 or 1 (or unset)" ;;
esac
# A source build must never silently extract an unqualified upstream wheel.
for v in VLLM_PRECOMPILED_WHEEL_LOCATION VLLM_USE_PRECOMPILED VLLM_USE_PRECOMPILED_RUST VLLM_DOCKER_BUILD_CONTEXT CMAKE_ARGS; do
  [ -z "${!v+x}" ] || die "$v is set; unset it to build from this checkout"
done
# Local dependency overrides would bypass the checkout's native source pins.
while IFS= read -r v; do
  case "$v" in
    *_SRC_DIR|FETCHCONTENT_*) die "$v is set; external sources must use the checkout's CMake pins" ;;
  esac
done < <(compgen -e)
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1

# Record the source independently of the package version that setup.py will stamp. The engine is vLLM 0.33.0
# (upstream commit 4ba787ec99, named in NOTICE); the public history does not carry vLLM's commits, and the
# wheel check pins the native sources by tree id.
command -v git >/dev/null || die "git is required"
[ "$(git -C "$ENGINE" rev-parse --show-toplevel)" = "$ENGINE" ] \
  || die "engine-source-path must be the Git checkout root"
SOURCE_BASE="vllm-0.33.0"
SOURCE_COMMIT="$(git -C "$ENGINE" rev-parse --verify HEAD)"
SOURCE_TREE="$(git -C "$ENGINE" rev-parse --verify 'HEAD^{tree}')"
[ -z "$(git -C "$ENGINE" status --porcelain --untracked-files=no)" ] \
  || die "selected checkout has modified tracked source; commit it before installing"
SOURCE_ID="$(printf 'upstream-base %s\ncommit %s\ntree %s' "$SOURCE_BASE" "$SOURCE_COMMIT" "$SOURCE_TREE")"
if [ "$MODE" != install ]; then
  # A wheel install records a fourth line naming the wheel.
  [ -r "$ENGINE/.build-source" ] && [ "$(sed -n 1,3p "$ENGINE/.build-source")" = "$SOURCE_ID" ] \
    || die "build source identity is missing or differs from the selected checkout; rebuild"
fi
echo "SOURCE OK: commit=$SOURCE_COMMIT tree=$SOURCE_TREE base=$SOURCE_BASE"
if [ -n "$WHEEL" ]; then
  [ -f "$WHEEL" ] || die "MERLIN_WHEEL=$WHEEL is not a file"
  WHEEL="$(cd "$(dirname "$WHEEL")" && pwd -P)/$(basename "$WHEEL")"
  command -v python3 >/dev/null || die "python3 is required to verify MERLIN_WHEEL"
  python3 -I "$REPO/release/wheel.py" verify "$WHEEL" "$ENGINE/release/merlin-wheel.sha256" "$ENGINE" \
    || die "MERLIN_WHEEL is not this checkout's release wheel"
  SOURCE_ID="$SOURCE_ID
wheel $(sha256sum "$WHEEL" | cut -d' ' -f1)  $(basename "$WHEEL")"
fi

if [ "$MODE" = install ]; then
  command -v uv >/dev/null || die "uv is required"
  [ ! -e "$VENV" ] && [ ! -L "$VENV" ] || die "$VENV already exists; choose a new venv path"
  # The native CUDA ops need GCC/G++ >= 13 and C++20 (the reference host uses GCC 14). Check before the long install.
  export CC="${CC:-cc}" CXX="${CXX:-c++}"
  export CUDAHOSTCXX="${CUDAHOSTCXX:-$CXX}"
  for t in "$CC" "$CXX"; do command -v "$t" >/dev/null || die "the install needs $t"; done
  [ -n "$WHEEL" ] || command -v cargo >/dev/null || die "source build needs cargo"
  for t in "$CC" "$CXX"; do
    v="$("$t" -dumpversion 2>/dev/null | cut -d. -f1)"
    case "$v" in ''|*[!0-9]*) die "cannot read the version of $t (GCC/G++ >= 13 required)" ;; esac
    [ "$v" -ge 13 ] || die "$t is version $v; the CUDA ops need GCC/G++ >= 13 (set CC/CXX to a newer compiler)"
  done
  uv venv --python 3.13 "$VENV"
  # One lock covers the frozen dependencies and the no-isolation build tools.
  uv pip install --python "$VENV/bin/python" --require-hashes --only-binary :all: -r "$REQ"
fi
PY="$VENV/bin/python"
[ -x "$PY" ] || die "no venv Python at $PY"
if [ "$MODE" != install ]; then
  [ -x "$VENV/bin/vllm" ] && [ -s "$ENGINE/.build-products" ] \
    && [ -s "$ENGINE/.build-products.sha256" ] || die "incomplete serving installation"
fi

# Validate every frozen pin, including the required serving dependencies.
"$PY" -P - "$REQ" <<'PY'
import importlib.metadata as md
import re
import sys
from pathlib import Path

from packaging.version import Version

required = {"torch", "triton", "flashinfer-python", "humming-kernels", "instanttensor"}
pins = {}
for line in Path(sys.argv[1]).read_text().splitlines():
    match = re.match(r"^([A-Za-z0-9_.-]+)(?:\[[A-Za-z0-9_,.-]+\])?==([^\s\\;]+)", line)
    if match:
        pins[match[1].lower().replace("_", "-")] = match[2]
for name in sorted(required):
    if name not in pins:
        sys.exit(f"lock must pin {name} with ==")
for name in sorted(pins):
    got = md.version(name)
    if Version(got) != Version(pins[name]):
        sys.exit(f"{name}: installed {got}, lock requires {pins[name]}")
if pins["flashinfer-python"] != "0.7.2.post1":
    sys.exit("FlashInfer tile requires flashinfer-python==0.7.2.post1")
if pins["humming-kernels"] != "0.1.19":
    sys.exit("0.33.0 requires humming-kernels==0.1.19")
if Version(pins["instanttensor"]) < Version("0.2.2"):
    sys.exit("0.33.0 requires instanttensor>=0.2.2 (exact version in the lock)")
try:
    md.distribution("flashinfer-jit-cache")
except md.PackageNotFoundError:
    pass
else:
    sys.exit("flashinfer-jit-cache would shadow the patched JIT build")
print("0.33.0 DEPENDENCY PINS OK")
PY

if [ "$TILE_SERIES" = v2 ]; then
  # Opt-in series: authenticated 0001+0002 (v2); --verify is offline (embedded hashes).
  WORKLOAD_PATCH="$REPO/release/apply-flashinfer-workload-patch.py"
  if [ "$MODE" = install ]; then
    "$PY" -I "$WORKLOAD_PATCH" ${FLASHINFER_WHEEL:+"$FLASHINFER_WHEEL"}
  fi
  "$PY" -I "$WORKLOAD_PATCH" --verify
else
if [ "$MODE" = install ]; then
  bash "$FLASHINFER_PATCH" "$VENV"
fi
# Verification must reject a missing or edited tile rather than apply it.
"$PY" -I - <<'PY' || die "the default 0001 tile does not verify; a venv installed with the opt-in series verifies with MERLIN_FLASHINFER_WORKLOAD_TILE=1"
import hashlib
import sys
import sysconfig
from pathlib import Path

root = Path(sysconfig.get_paths()["purelib"]) / "flashinfer"
expected = {
    "data/csrc/batch_prefill.cu":
        "6f3a7033af4426889585e6a5ab2fb959c4a965a98e106072c175631ed4d3a68e",
    "data/csrc/batch_prefill_paged_kernel_inst.jinja":
        "500fcecb03e267e6c7bd85c1240ba34581e5e1f9b1c82dd671f1a669f740e041",
    "data/include/flashinfer/attention/prefill.cuh":
        "4e371ab99e5170f6c76d9556f48cd8297e85254ce35c43edfa6cebdc0f401062",
    "data/include/flashinfer/attention/scheduler.cuh":
        "f45a12be6583888bb23a6fe4fed5d192c67757dbb58e9d209e8efd3a8f59ba5b",
    "jit/attention/modules.py":
        "36127818bd5c0ef66947105248398810c30c4c1999c4eb3dd0c0cfd98e75c4f4",
}
for name, digest in expected.items():
    path = root / name
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        sys.exit(f"installed FlashInfer tile is missing or changed: {path}")
print("FLASHINFER TILE HASHES OK")
PY
fi

if [ "$MODE" = install ]; then
  CU="$("$PY" -P -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')/nvidia/cu13"
  [ -d "$CU/lib" ] || die "lock must include the qualified CUDA 13.0 compiler toolkit wheels"
  [ -e "$CU/lib64" ] || ln -s lib "$CU/lib64"
  for so in "$CU"/lib/lib*.so.[0-9]*; do
    [ -e "$so" ] || continue
    base="${so%%.so.*}.so"
    [ -e "$base" ] || ln -s "$(basename "$so")" "$base"
  done
  if [ -n "${CUDA_HOME:-}" ] && [ "$CUDA_HOME" != "$CU" ]; then
    # An inherited toolkit replaces the locked one; only another CUDA 13 toolkit is accepted, and only 13.0 matches torch's
    # cu130 build without a warning (a newer nvcc beside the 13.0 runtime headers can fail FlashInfer's CCCL version check).
    # || true: a missing or failing nvcc must reach the refusal below, not end the script under pipefail
    rel="$({ "$CUDA_HOME/bin/nvcc" --version 2>/dev/null || true; } | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p')"
    case "$rel" in
      13.0) ;;
      13.*) echo "install-env: WARN: CUDA_HOME=$CUDA_HOME is CUDA $rel; the locked toolkit and torch's cu130 build are 13.0, so this is untested and a mixed-version CCCL build failure is possible; unset CUDA_HOME to use the locked toolkit at $CU" >&2 ;;
      *) die "CUDA_HOME=$CUDA_HOME is not a CUDA 13 toolkit (nvcc release: ${rel:-none}); unset CUDA_HOME to use the locked toolkit at $CU" ;;
    esac
    echo "install-env: note: building with CUDA_HOME=$CUDA_HOME from the environment instead of the locked toolkit $CU" >&2
  fi
  export CUDA_HOME="${CUDA_HOME:-$CU}"
  [ -x "$CUDA_HOME/bin/nvcc" ] || die "source build needs CUDA_HOME/bin/nvcc"
  # Driver-less hosts (docker build): the extensions link the driver API (CUDA::cuda_driver) and the toolkit wheels ship no
  # libcuda. MERLIN_LIBCUDA names the CUDA toolkit's stubs/libcuda.so; it is linked under the toolkit's lib/stubs exactly as
  # release/build-wheel.sh does, so it is a link-time file only and never on the loader path. Unset: nothing changes (a host
  # with the driver needs no stub). The container sets it for the wheel path too, the configuration that was qualified.
  if [ -n "${MERLIN_LIBCUDA:-}" ]; then
    [ -e "$MERLIN_LIBCUDA" ] || die "MERLIN_LIBCUDA=$MERLIN_LIBCUDA does not exist"
    mkdir -p "$CUDA_HOME/lib/stubs"
    ln -sfn "$MERLIN_LIBCUDA" "$CUDA_HOME/lib/stubs/libcuda.so"
  fi
  export PATH="$VENV/bin:$CUDA_HOME/bin:$PATH"
  # No dependency resolution or isolated build may replace the locked torch/triton. setup.py loads vllm/envs.py by
  # path, and that imports vllm.merlin.envs, so the checkout must be importable for this one command.
  if [ -n "$WHEEL" ]; then
    # Upstream's precompiled route: the editable install takes the compiled products from the wheel.
    (cd "$ENGINE" && PYTHONPATH="$ENGINE" VLLM_TARGET_DEVICE=cuda VLLM_VERSION_OVERRIDE=0.33.0 \
      VLLM_USE_PRECOMPILED=1 VLLM_PRECOMPILED_WHEEL_LOCATION="$WHEEL" \
      uv pip install --python "$PY" --no-deps --no-build-isolation --editable .)
  else
    (cd "$ENGINE" && PYTHONPATH="$ENGINE" VLLM_TARGET_DEVICE=cuda VLLM_VERSION_OVERRIDE=0.33.0 \
      TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;8.6;8.9}" \
      uv pip install --python "$PY" --no-deps --no-build-isolation --editable .)
  fi
fi

# -P and the cleared Python environment force the editable install to supply vllm.
"$PY" -P - "$ENGINE" <<'PY'
import importlib.metadata as md
import sys
import sysconfig
from pathlib import Path

import flashinfer
import torch
import triton
import vllm
from packaging.version import Version

engine = Path(sys.argv[1]).resolve()
origin = Path(vllm.__file__).resolve()
if origin != engine / "vllm" / "__init__.py":
    sys.exit(f"vllm imported from {origin}, expected {engine}/vllm/__init__.py")
if Version(vllm.__version__).public != "0.33.0":
    sys.exit(f"vllm is {vllm.__version__}, expected 0.33.0")
if Path(flashinfer.__file__).resolve() != (
    Path(sysconfig.get_paths()["purelib"]) / "flashinfer" / "__init__.py"
).resolve():
    sys.exit("flashinfer imports outside the verified venv package")
for name, module in (
    ("torch", torch), ("triton", triton), ("flashinfer-python", flashinfer)
):
    # the lock's PyPI torch reports 2.14.0+cu130 against distribution metadata 2.14.0: compare public versions
    if Version(module.__version__).public != Version(md.version(name)).public:
        sys.exit(f"{name}: imported module disagrees with installed distribution")
print(f"ENGINE IMPORT OK: {vllm.__version__} from {origin}")
PY
# Check dependency consistency without requiring uv in read-only modes.
if [ "$MODE" = install ] || command -v uv >/dev/null; then
  uv pip check --python "$PY"
else
  # uv-created venvs need not contain pip; use their Python and locked packaging.
  "$PY" -I - <<'PY'
import importlib.metadata as md
import sys

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

installed = {canonicalize_name(d.metadata["Name"]): d for d in md.distributions()}
errors = []
for dist in installed.values():
    for raw in dist.requires or ():
        req = Requirement(raw)
        if req.marker and not req.marker.evaluate({"extra": ""}):
            continue
        dep = installed.get(canonicalize_name(req.name))
        if dep is None:
            errors.append(f"{dist.metadata['Name']} requires {req}, which is not installed")
        elif req.specifier and not req.specifier.contains(dep.version, prereleases=True):
            errors.append(f"{dist.metadata['Name']} requires {req}, but {dep.version} is installed")
if errors:
    sys.exit("\n".join(errors))
print("DEPENDENCY CONSISTENCY OK (venv Python; uv unavailable)")
PY
fi

if [ "$MODE" = install ]; then
  # Keep the launcher contract: hash the native and generated Python products.
  "$PY" -P - "$ENGINE" "${WHEEL:+wheel $(basename "$WHEEL")}" <<'PY'
import hashlib
import sys
from pathlib import Path

engine = Path(sys.argv[1])
package = engine / "vllm"
for pattern in (
    "_C_stable_libtorch*.so", "_moe_C_stable_libtorch*.so",
    "cumem_allocator*.so", "vllm_flash_attn/_vllm_fa2_C*.so",
):
    if not list(package.glob(pattern)):
        sys.exit(f"source build did not produce vllm/{pattern}")
products = set(package.rglob("*.so")) | set(package.rglob("*.so.*"))
for name in ("triton_kernels", "vllm_flash_attn", "third_party"):
    products.update(
        p for p in (package / name).rglob("*")
        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
    )
lines = [
    f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(engine)}\n"
    for p in sorted(products)
]
sums = engine / ".build-products.sha256"
sums.write_text("".join(lines))
(engine / ".build-products").write_text((sys.argv[2] or "source 0.33.0") + "\n")
print(f"BUILD PRODUCTS RECORDED: {len(products)} files")
PY
fi

if [ "$MODE" = install ]; then
  printf '%s\n' "$SOURCE_ID" > "$ENGINE/.build-source"
else
  (cd "$ENGINE" && sha256sum -c --quiet --strict .build-products.sha256)
fi

if [ "$MODE" = check ]; then
  # Use a 27B log from this installation's current boot, after an MTP request completes.
  echo "LIVE TILE PROOF: 27B only; for non-27B lanes (including Flash-Next), use --verify."
  bash "$TILE_PROOF" "$@"
  echo "POST-INSTALL SELF-CHECK OK: checkout import and live 27B tile proof"
elif [ "$MODE" = verify ]; then
  echo "ENVIRONMENT VERIFIED: source, dependency pins, patch and native products"
else
  echo "ENGINE INSTALLED: TREE=$ENGINE VENV=$VENV"
  if [ "$TILE_SERIES" = v2 ]; then
    echo "FLASHINFER: opt-in 0001+0002 series (v2); the workload tile is selected at serve time by MERLIN_FLASHINFER_WORKLOAD_TILE=1 with FLASHINFER_SM8X_FP8_SMALLQ=cta32."
  fi
  echo "LIVE TILE PROOF PENDING (27B only): serve an MTP request with FLASHINFER_SM8X_FP8_SMALLQ=cta32,"
  echo "then run $ENGINE/release/install-env.sh --check '$VENV' <server-log> [q_len] --tp N"
  echo "Non-27B lanes (including Flash-Next): use --verify; this tile proof does not apply."
fi
