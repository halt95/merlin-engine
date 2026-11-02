#!/usr/bin/env bash
# Shim for the one Merlin launcher, tools/merlin/launch.py --profile 27b-tp1-3090.
# TP=1 on one card, with vision (SPEC_K fixed at 4; tested 2026-10-21, 4 MP images). Knobs as serve.sh minus P2P/RECOVERSSM/SPEC_K.
#   MODEL=/path/to/checkpoint PORT=8100 bash models/qwen38-27b/serve-1card.sh
set -euo pipefail
REPO="${MERLIN_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
exec "${MERLIN_PYTHON:-python3}" "$REPO/tools/merlin/launch.py" --profile 27b-tp1-3090 -- "$@"
