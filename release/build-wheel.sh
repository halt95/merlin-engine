#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
# Build the Merlin engine wheel (sm_80, sm_86, sm_89) from this checkout's HEAD.
#   release/build-wheel.sh [output-dir]          (default: ./dist under this checkout)
# Builds in a scratch copy against the release lock, refuses a wheel that carries
# build-host paths, and prints the line for release/merlin-wheel.sha256.
# Needs Linux x86_64, git, uv, readelf, GCC/G++ >= 13 and cargo; MAX_JOBS defaults to 8.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
REQ="$REPO/release/requirements-0.33.lock"
die() { echo "build-wheel: $*" >&2; exit 1; }

[ "$#" -le 1 ] || die "usage: build-wheel.sh [output-dir]"
OUT="${1:-$REPO/dist}"
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd -P)"
[ "$(uname -s)-$(uname -m)" = Linux-x86_64 ] || die "the release wheel is built on Linux x86_64"
for t in git uv readelf cargo; do command -v "$t" >/dev/null || die "missing $t"; done
export CC="${CC:-cc}" CXX="${CXX:-c++}"
export CUDAHOSTCXX="${CUDAHOSTCXX:-$CXX}"
for t in "$CC" "$CXX"; do
  v="$("$t" -dumpversion 2>/dev/null | cut -d. -f1)"
  case "$v" in ''|*[!0-9]*) die "cannot read the version of $t" ;; esac
  [ "$v" -ge 13 ] || die "$t is version $v; the CUDA ops need GCC/G++ >= 13"
done
for v in VLLM_PRECOMPILED_WHEEL_LOCATION VLLM_USE_PRECOMPILED VLLM_USE_PRECOMPILED_RUST CMAKE_ARGS CUDA_HOME TORCH_CUDA_ARCH_LIST CFLAGS CXXFLAGS CUDAFLAGS RUSTFLAGS CARGO_TARGET_DIR; do
  [ -z "${!v+x}" ] || die "$v is set; the release wheel uses the locked toolchain, fixed flags and the fixed architecture list"
done
while IFS= read -r v; do
  case "$v" in *_SRC_DIR|FETCHCONTENT_*) die "$v is set; external sources must use the checkout's CMake pins" ;; esac
done < <(compgen -e)
[ -z "$(git -C "$REPO" status --porcelain --untracked-files=no)" ] \
  || die "commit or discard tracked changes before building the release wheel"
COMMIT="$(git -C "$REPO" rev-parse --short=10 HEAD)"
VERSION="0.33.0+merlin.$COMMIT"
NAME="vllm-$VERSION-cp313-cp313-linux_x86_64.whl"
[ ! -e "$OUT/$NAME" ] || die "$OUT/$NAME already exists"

# Vendored OpenSSL bakes its install directory into the Rust frontend, so cargo builds at a fixed neutral path.
CARGO_TARGET=/tmp/merlin-wheel-cargo
mkdir -m 700 "$CARGO_TARGET" 2>/dev/null || die "$CARGO_TARGET exists (an earlier build, or another user); remove it"
[ -O "$CARGO_TARGET" ] && [ ! -L "$CARGO_TARGET" ] || die "$CARGO_TARGET is not a directory this user owns"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/merlin-wheel.XXXXXX")"
trap 'rm -rf -- "$WORK" "$CARGO_TARGET"' EXIT
SRC="$WORK/src" VENV="$WORK/venv"
mkdir "$SRC" "$WORK/tmp" "$WORK/dist"
git -C "$REPO" archive HEAD | tar -xf - -C "$SRC"
uv venv -q --python 3.13 "$VENV"
uv pip install -q --python "$VENV/bin/python" --require-hashes --only-binary :all: -r "$REQ"
PY="$VENV/bin/python"
CU="$("$PY" -P -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')/nvidia/cu13"
[ -x "$CU/bin/nvcc" ] || die "lock must include the CUDA 13.0 compiler toolkit wheels"
[ -e "$CU/lib64" ] || ln -s lib "$CU/lib64"
for so in "$CU"/lib/lib*.so.[0-9]*; do
  [ -e "$so" ] || continue
  base="${so%%.so.*}.so"
  [ -e "$base" ] || ln -s "$(basename "$so")" "$base"
done
# The extensions link the driver API (CUDA::cuda_driver), and the toolkit wheels ship no libcuda to link against.
stub="${MERLIN_LIBCUDA:-}"
[ -n "$stub" ] || stub="$({ ldconfig -p || /sbin/ldconfig -p; } 2>/dev/null | awk '$1 == "libcuda.so" {print $NF; exit}' || true)"
[ -n "$stub" ] || stub="$(ls /usr/local/cuda*/targets/x86_64-linux/lib/stubs/libcuda.so 2>/dev/null | head -n 1 || true)"
[ -n "$stub" ] && [ -e "$stub" ]   || die "no libcuda.so to link against: install the NVIDIA driver or a CUDA toolkit, or set MERLIN_LIBCUDA"
mkdir -p "$CU/lib/stubs"
ln -s "$stub" "$CU/lib/stubs/libcuda.so"

# Map build-host paths out of __FILE__ strings in host and device code. The map goes to CMake only:
# vendored OpenSSL in the Rust frontend records CFLAGS verbatim, paths included. setup.py splits
# CMAKE_ARGS on whitespace, hence one space-free map for the whole scratch directory.
case "$WORK" in *[[:space:]]*) die "the scratch directory $WORK contains whitespace; set TMPDIR to a plain path" ;; esac
MAP="-ffile-prefix-map=$WORK=/build"
CMAKE_MAP="-DCMAKE_C_FLAGS=$MAP -DCMAKE_CXX_FLAGS=$MAP -DCMAKE_CUDA_FLAGS=-Xcompiler=$MAP"
echo "== build $NAME (TORCH_CUDA_ARCH_LIST=8.0;8.6;8.9) $(date -u +%FT%TZ)"
(cd "$SRC" && env PATH="$VENV/bin:$CU/bin:$PATH" CUDA_HOME="$CU" CUDACXX="$CU/bin/nvcc" \
  TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9" VLLM_TARGET_DEVICE=cuda \
  MAX_JOBS="${MAX_JOBS:-8}" CARGO_BUILD_JOBS="${MAX_JOBS:-8}" NVCC_THREADS=1 CMAKE_BUILD_TYPE=Release \
  SETUPTOOLS_SCM_PRETEND_VERSION="$VERSION" VLLM_VERSION_OVERRIDE="$VERSION" \
  CMAKE_ARGS="$CMAKE_MAP" CARGO_TARGET_DIR="$CARGO_TARGET" \
  RUSTFLAGS="--remap-path-prefix=$WORK=/build --remap-path-prefix=${CARGO_HOME:-$HOME/.cargo}=/cargo --remap-path-prefix=${RUSTUP_HOME:-$HOME/.rustup}=/rustup" \
  PYTHONPATH="$SRC" PYTHONDONTWRITEBYTECODE=1 TMPDIR="$WORK/tmp" \
  "$PY" -m build --wheel --no-isolation --outdir "$WORK/dist" .)
[ -f "$WORK/dist/$NAME" ] || die "build did not produce $NAME: $(ls "$WORK/dist")"
if ! "$PY" -P "$REPO/release/wheel.py" audit "$WORK/dist/$NAME" --readme "$REPO/README.md" --forbid "$WORK" --forbid "$REPO" --forbid "$HOME"; then
  mv "$WORK/dist/$NAME" "$OUT/$NAME.rejected"
  die "the wheel failed its audit; kept for inspection as $OUT/$NAME.rejected"
fi
mv "$WORK/dist/$NAME" "$OUT/$NAME"
line="$(cd "$OUT" && sha256sum "$NAME")"
printf '%s\n' "$line" > "$OUT/$NAME.sha256"
echo "== built $OUT/$NAME ($(du -h "$OUT/$NAME" | cut -f1)) $(date -u +%FT%TZ)"
echo "release/merlin-wheel.sha256: $line"
