#!/usr/bin/env bash
# Merlin 3.0.0 compatibility entry point for the generated vLLM 0.32.0
# Swift1.5-Huihui recipe at models/swift1.5-huihui/serve.sh.
# Arguments and environment pass through unchanged.
# Kept through Merlin 3.0.x; to be removed in a later release.
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "note: scripts/serve-huihui.sh moved to models/swift1.5-huihui/serve.sh; forwarding" >&2
exec bash "$here/models/swift1.5-huihui/serve.sh" "$@"
