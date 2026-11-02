#!/usr/bin/env bash
# Select a canonical recipe; its launcher owns the serving flags and tile opt-in.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
recipe="${MERLIN_MODEL:-flash-next}"
case "$recipe" in
  flash-next|swift1.5|swift1.5-huihui) family=flash ;;
  qwen38-27b|thinkingcap) family=dense ;;
  *) echo "MERLIN_MODEL must be flash-next, swift1.5, swift1.5-huihui, qwen38-27b or thinkingcap" >&2; exit 2 ;;
esac
CKPT="${CHECKPOINT_DIR:-/models/checkpoint}"
if [ "$#" -gt 0 ]; then
  case "$1" in -*) ;; "") shift ;; *) CKPT="$1"; shift ;; esac
fi
[ -f "$CKPT/config.json" ] || {
  echo "checkpoint not found at $CKPT; mount the checkpoint directory at /models/checkpoint" >&2
  exit 1
}
export VENV="${VENV:-/opt/venv}" TREE="${TREE:-$REPO}"
export CACHE_ROOT="${CACHE_ROOT:-/cache}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$CACHE_ROOT}"
if [ "$family" = flash ]; then
  # MODEL_NAME only when supplied: the launcher's default names (flash-next-v2 flash-next flash-mtp flash-next-mtp,
  # or the checkpoint's own) stay the same inside and outside the container
  if ! "$VENV/bin/python" -c 'import json,sys; c=json.load(open(sys.argv[1], encoding="utf-8")); sys.exit(0 if c.get("text_config",{}).get("ple_embedding_dtype")=="float8_e4m3fn" else 1)' "$CKPT/config.json"; then
    if [ -w "$CKPT" ] && [ -w "$CKPT/config.json" ]; then
      "$VENV/bin/python" "$REPO/models/flash-next/make-e1-config.py" "$CKPT"
    else
      echo "config.json needs text_config.ple_embedding_dtype=float8_e4m3fn; prepare it with models/flash-next/make-e1-config.py" >&2
      exit 1
    fi
  fi
else
  [ "$#" = 0 ] || {
    echo "27B/ThinkingCap launchers take environment settings rather than extra vLLM arguments" >&2
    exit 2
  }
  # SERVED only when supplied: the launcher's default name (qwen38-27b for both recipes) stays the same as on bare metal
  export MODEL="$CKPT"
fi
if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi -L >/dev/null 2>&1; then
  echo "WARNING: no NVIDIA GPU visible; use --gpus all and configure nvidia-container-toolkit on the host" >&2
fi
cd "$REPO"
if [ "$family" = flash ]; then
  exec bash "$REPO/models/$recipe/serve.sh" "$CKPT" "$@"
fi
exec bash "$REPO/models/$recipe/serve.sh"
