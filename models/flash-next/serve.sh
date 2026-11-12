#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
# Shim for the one Merlin launcher, tools/merlin/launch.py --profile fn-quad-3090.
# Knobs: PORT HOST MODEL_NAME PLE_HOME SCALES GUARD CACHE_ROOT TREE VENV WARM COUNTERS IMG_CAP_PX DRAFT_HEAD KV_BYTES CHUNK BLOCK_SIZE LPT PREFILL_RUNS SEQS MAXLEN AUTO_TOPO CHAT_TEMPLATE CUDA_HOME (docs/reference.md).
#   bash models/flash-next/serve.sh /path/to/Qwen3.8-Flash-Next-W4A16-Merlin [extra vllm args]
set -euo pipefail
REPO="${MERLIN_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
exec "${MERLIN_PYTHON:-python3}" "$REPO/tools/merlin/launch.py" --profile fn-quad-3090 -- "$@"
