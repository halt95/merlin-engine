#!/usr/bin/env bash
# Merlin 3.0.0 compatibility entry point for release/build-wheel.sh (the engine wheel).
# Arguments: [OUTPUT_DIR].
# Environment setup is release/install-env.sh; this command does not create a venv.
# Kept through Merlin 3.0.x; to be removed in a later release.
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "note: scripts/build.sh moved to release/build-wheel.sh; forwarding" >&2
exec bash "$here/release/build-wheel.sh" "$@"
