#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
# Shim for the one Merlin launcher, tools/merlin/launch.py --profile svl-quad-3090.
# TP2 x PP2 + EP with RecoverSSM. Knobs: MODEL GPUS PORT HOST SERVED VENV TREE CACHE_ROOT P2P KV_BYTES MNBT/CHUNK SEQS MAXLEN BLOCK/BLOCK_SIZE LPT PP_PARTITION MAX_PIXELS CUDA_HOME.
#   MODEL=~/models/nemotron-super-vl PORT=8000 bash models/nemotron-super-vl/serve.sh [extra vllm args]
set -euo pipefail
REPO="${MERLIN_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
exec "${MERLIN_PYTHON:-python3}" "$REPO/tools/merlin/launch.py" --profile svl-quad-3090 -- "$@"
