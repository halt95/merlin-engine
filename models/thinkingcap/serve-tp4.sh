#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
# Shim for the one Merlin launcher, tools/merlin/launch.py --profile 27b-tp4-3090 --checkpoint 27b-thinkingcap.
# ThinkingCap on four cards: the 27B TP4 shape with the ThinkingCap checkpoint (reasoning effort medium by default; a THINKING you set yourself wins).
#   MODEL=/path/to/thinkingcap-checkpoint PORT=8100 bash models/thinkingcap/serve-tp4.sh
set -euo pipefail
REPO="${MERLIN_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
exec "${MERLIN_PYTHON:-python3}" "$REPO/tools/merlin/launch.py" --profile 27b-tp4-3090 --checkpoint 27b-thinkingcap -- "$@"
