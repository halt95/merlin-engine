#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
# Deprecated entry point (the v2 launcher name): forwards to models/flash-next/serve.sh with the same arguments and
# environment. Kept through Merlin 3.0.x; to be removed in a later release.
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "note: scripts/serve-v2.sh is deprecated; forwarding to models/flash-next/serve.sh" >&2
exec bash "$here/models/flash-next/serve.sh" "$@"
