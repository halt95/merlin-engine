#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
# Shim for the one Merlin launcher, tools/merlin/launch.py --profile 27b-tp4-3090.
# TP=4 on four cards: RecoverSSM on, block 1024, chunk 4096. Knobs as serve.sh (README.md).
#   MODEL=/path/to/checkpoint PORT=8100 bash models/qwen38-27b/serve-tp4.sh
set -euo pipefail
REPO="${MERLIN_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
exec "${MERLIN_PYTHON:-python3}" "$REPO/tools/merlin/launch.py" --profile 27b-tp4-3090 -- "$@"
