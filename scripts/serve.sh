#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
# Compatibility shim (since Merlin 3.0.0): the Flash-Next launcher moved to models/flash-next/serve.sh. Same arguments,
# same environment. Kept through Merlin 3.0.x; to be removed in a later release.
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "note: scripts/serve.sh moved to models/flash-next/serve.sh; forwarding" >&2
exec bash "$here/models/flash-next/serve.sh" "$@"
