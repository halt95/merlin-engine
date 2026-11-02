#!/usr/bin/env bash
# One-command install and serve for the Merlin engine:
#   git clone <repo> && cd merlin-engine && ./setup.sh
# Every option: docs/SETUP.md (or ./setup.sh --help). This file only bootstraps a repo-local uv and
# Python 3.13 under .merlin/ and hands over to tools/merlin/setup_cli.py, which holds the logic.
# Nothing is installed system-wide and nothing here needs sudo.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
MERLIN_DIR="$REPO/.merlin"
UV_VERSION_DEFAULT="0.11.29"
UV_SHA256_DEFAULT="04f8b82f5d47f0512dcd32c67a4a6f16a0ea27c81537c338fd0ad6b23cebe829"  # uv-x86_64-unknown-linux-gnu.tar.gz of 0.11.29
UV_VERSION="${MERLIN_UV_VERSION:-$UV_VERSION_DEFAULT}"
UV_MIN_VERSION="0.8.0"  # `uv python install --no-bin` and `uv python find` as used here
UV_URL="https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/uv-x86_64-unknown-linux-gnu.tar.gz"
PYTHON_VERSION="3.13"

PRINTED=0
say() { printf '%s\n' "$*"; PRINTED=1; }
# One blank line between the lines above and setup_cli.py's own output (only when this script printed any).
handover_gap() { [ "$PRINTED" = 0 ] || printf '\n'; }
die() { printf 'setup: %s\n' "$*" >&2; exit 1; }

case "$(uname -s)/$(uname -m)" in
  Linux/x86_64) ;;
  *) die "Merlin runs on Linux x86-64 only (this is $(uname -s)/$(uname -m)); the release wheel and the dependency lock are built for it." ;;
esac

# A system python3 (3.10+) can show --help and run a dry run without any download.
system_py() {
  local p
  for p in python3 python3.13 python3.12 python3.11 python3.10; do
    if command -v "$p" >/dev/null 2>&1 && "$p" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
      command -v "$p"; return 0
    fi
  done
  return 1
}

SIMULATE=0 DRY_RUN=0 HELP=0 NO_START=0
for a in "$@"; do
  case "$a" in
    -h|--help) HELP=1 ;;
    --simulate|--simulate=*) SIMULATE=1 ;;  # testing on fabricated hardware: no nvidia-smi needed
    --dry-run) DRY_RUN=1 ;;
    --no-start) NO_START=1 ;;
  esac
done
[ "$SIMULATE" = 0 ] || [ "$NO_START" = 1 ] || DRY_RUN=1  # --simulate alone is a dry run (setup_cli does the same)
if [ "$HELP" = 1 ]; then
  if PY="$(system_py)"; then exec "$PY" -I "$REPO/tools/merlin/setup_cli.py" --help; fi
  sed -n '2,6p' "$0"; say "(python3 is not on PATH: ./setup.sh fetches Python 3.13 itself, then --help prints every option)"; exit 0
fi

# Cheap host checks that cost no download: the full set runs in setup_cli.py before the big downloads.
command -v curl >/dev/null || die "curl is missing. Fix (Debian/Ubuntu): sudo apt install curl"
command -v tar >/dev/null || die "tar is missing. Fix (Debian/Ubuntu): sudo apt install tar"
command -v git >/dev/null || die "git is missing. Fix (Debian/Ubuntu): sudo apt install git"
command -v sha256sum >/dev/null || die "sha256sum is missing. Fix (Debian/Ubuntu): sudo apt install coreutils"
if [ "$SIMULATE" = 0 ] && ! command -v nvidia-smi >/dev/null; then
  die "nvidia-smi is missing: no NVIDIA driver is installed or it is not on PATH. Install the NVIDIA driver (R580 or later, CUDA 13) from your distribution or https://www.nvidia.com/drivers, reboot, then run ./setup.sh again."
fi

# uv: a recent one on PATH, else the repo-local copy, else fetch it into .merlin/bin (URL, size and sha256 checked first).
uv_ok() { "$1" --version 2>/dev/null | awk -v min="$UV_MIN_VERSION" '{split($2,a,"."); split(min,m,"."); exit !(a[1]>m[1] || (a[1]==m[1] && (a[2]>m[2] || (a[2]==m[2] && a[3]>=m[3]))))}'; }
UV=""
for cand in "$(command -v uv || true)" "$MERLIN_DIR/bin/uv"; do
  if [ -n "$cand" ] && [ -x "$cand" ] && uv_ok "$cand"; then UV="$cand"; break; fi
done
if [ -z "$UV" ]; then
  size="$(curl -sIL "$UV_URL" | tr -d '\r' | awk 'tolower($1)=="content-length:"{s=$2} END{print s+0}')" \
    || die "cannot reach $UV_URL (offline/proxy?)"
  if [ "$size" -gt 0 ] 2>/dev/null; then size="$((size / 1024 / 1024)) MB"; else size="size unknown"; fi
  if [ "$DRY_RUN" = 1 ]; then
    say "setup: dry run: would fetch uv $UV_VERSION ($size) -> .merlin/bin/uv and Python $PYTHON_VERSION (about 35 MB) -> .merlin/python"
    say "       $UV_URL"
    if PY="$(system_py)"; then
      say "setup: dry run continues on $PY (a real run uses the fetched Python $PYTHON_VERSION)"
      handover_gap
      exec "$PY" -I "$REPO/tools/merlin/setup_cli.py" "$@"
    fi
    say "setup: no python3 (3.10+) on PATH, so the dry run stops here; a real run fetches the two above and then checks the host before any other download."
    exit 0
  fi
  say "setup: fetching uv $UV_VERSION ($size) -> .merlin/bin/uv"
  say "       from github.com/astral-sh/uv release $UV_VERSION (sha256 checked before use)"
  mkdir -p "$MERLIN_DIR/bin"; chmod 700 "$MERLIN_DIR"
  tmp="$(mktemp -d "$MERLIN_DIR/uv.XXXXXX")"
  curl -fL -# --retry 3 -C - -o "$tmp/uv.tar.gz" "$UV_URL" || die "could not download uv from $UV_URL (no network? put uv >= $UV_MIN_VERSION on PATH and run again)"
  # Integrity: the release's published .sha256, and for the default version the digest recorded in this file.
  curl -fsSL -o "$tmp/uv.tar.gz.sha256" "$UV_URL.sha256" || die "could not download $UV_URL.sha256 (needed to check the archive)"
  published="$(awk '{print $1}' "$tmp/uv.tar.gz.sha256")"
  got="$(sha256sum "$tmp/uv.tar.gz" | awk '{print $1}')"
  [ "$got" = "$published" ] || { rm -rf "$tmp"; die "uv archive sha256 $got does not match the published $published; not used"; }
  if [ "$UV_VERSION" = "$UV_VERSION_DEFAULT" ]; then
    [ "$got" = "$UV_SHA256_DEFAULT" ] || { rm -rf "$tmp"; die "uv $UV_VERSION archive sha256 $got does not match the digest recorded in setup.sh ($UV_SHA256_DEFAULT); not used"; }
  else
    say "setup: note: MERLIN_UV_VERSION=$UV_VERSION is checked against its published .sha256 only (setup.sh records the digest of $UV_VERSION_DEFAULT)"
  fi
  tar -xzf "$tmp/uv.tar.gz" -C "$tmp"
  mv "$tmp"/uv-x86_64-unknown-linux-gnu/uv "$MERLIN_DIR/bin/uv"
  rm -rf "$tmp"
  UV="$MERLIN_DIR/bin/uv"
  uv_ok "$UV" || die "the fetched uv reports $("$UV" --version 2>/dev/null); expected >= $UV_MIN_VERSION"
fi

# Python 3.13, uv-managed, inside the repo (install-env.sh's `uv venv --python 3.13` finds it the same way).
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$MERLIN_DIR/python}"
export UV_PYTHON_PREFERENCE="${UV_PYTHON_PREFERENCE:-managed}"
PY="$("$UV" python find "$PYTHON_VERSION" 2>/dev/null || true)"
if [ -z "$PY" ] || [ ! -x "$PY" ]; then
  if [ "$DRY_RUN" = 1 ]; then
    say "setup: dry run: would fetch Python $PYTHON_VERSION with uv (about 35 MB) -> .merlin/python"
    if SPY="$(system_py)"; then
      say "setup: dry run continues on $SPY"
      handover_gap
      exec "$SPY" -I "$REPO/tools/merlin/setup_cli.py" "$@"
    fi
    say "setup: no python3 (3.10+) on PATH, so the dry run stops here."
    exit 0
  fi
  say "setup: fetching Python $PYTHON_VERSION with uv (about 35 MB) -> .merlin/python"
  mkdir -p "$MERLIN_DIR"; chmod 700 "$MERLIN_DIR"
  "$UV" python install --no-bin "$PYTHON_VERSION" || die "uv could not install Python $PYTHON_VERSION (no network?)"
  PY="$("$UV" python find "$PYTHON_VERSION")"
fi

# .merlin/bin carries uv and a python3 for release/install-env.sh (which calls `uv` and `python3` from PATH), so a host
# without a system python3 installs the same way.
if [ "$DRY_RUN" = 1 ]; then  # a dry run writes nothing, not even the two links
  export PATH="$(dirname "$UV"):$PATH"
else
  mkdir -p "$MERLIN_DIR/bin"; chmod 700 "$MERLIN_DIR"
  [ "$UV" = "$MERLIN_DIR/bin/uv" ] || ln -sfn "$UV" "$MERLIN_DIR/bin/uv"
  ln -sfn "$PY" "$MERLIN_DIR/bin/python3"
  export PATH="$MERLIN_DIR/bin:$PATH"
fi
handover_gap
exec "$PY" -I "$REPO/tools/merlin/setup_cli.py" "$@"
