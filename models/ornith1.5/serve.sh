#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
# Shim for the one Merlin launcher, tools/merlin/launch.py --profile ornith-quad-3090 --p2p-probe: the cards' peer
# access (or P2P=on|off) picks ornith-quad-3090 or ornith-quad-3090-nop2p.
# TP2 x PP2 + EP with RecoverSSM. Knobs: MODEL GPUS PORT HOST SERVED VENV TREE CACHE_ROOT P2P KV_BYTES CHUNK SEQS MAXLEN BLOCK_SIZE LPT PP_PARTITION MAX_PIXELS UV CUDA_HOME.
#   MODEL=/path/to/Ornith-1.5-122B-A10B-W4A16-Merlin PORT=8000 bash models/ornith1.5/serve.sh [extra vllm args]
set -euo pipefail
REPO="${MERLIN_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
exec "${MERLIN_PYTHON:-python3}" "$REPO/tools/merlin/launch.py" --profile ornith-quad-3090 --p2p-probe -- "$@"
