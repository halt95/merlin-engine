#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
# Provision a Debian 13 LXC created by lxc/pve-create.sh. Run inside it as root.
# Builds the Merlin checkout (vLLM 0.33.0) with release/install-env.sh.
# Prepare uv and the Rust toolchain selected by rust-toolchain.toml first.
# The release dependency hash lock is complete (223 exact pins with SHA-256 hashes).
# The installer rejects a modified lock containing unresolved markers. No legacy engine bundle or compiled-artifact download is used.
# Re-runs verify an existing installation; failed/partial venvs require a fresh
# venv before retrying. Needs network for apt, source and build dependencies.
# The live tile proof (release/check-tile-proof.sh) is 27B-only; a Flash-Next boot never emits it, so reuse is
# gated by install-env.sh --verify alone.
#
# NVIDIA_RUN=/models/NVIDIA-Linux-x86_64-<host-version>.run bash provision.sh
# NVIDIA userspace must match the host kernel module. The installer below uses
# --no-kernel-module; peer-to-peer support remains a host-driver property.
set -euo pipefail
REPO_URL="${REPO_URL:-https://github.com/halt95/merlin-engine.git}"
REPO_REF="${REPO_REF:-merlin-v3.0.2}"
APP="${APP:-/opt/flash-next}"
VENV="${VENV:-/opt/venv}"
[ "$(id -u)" = 0 ] || { echo "run provision.sh as root" >&2; exit 1; }
export DEBIAN_FRONTEND=noninteractive
# Immediately after container start, DHCP/name resolution may still be pending.
for _ in $(seq 60); do getent hosts deb.debian.org >/dev/null 2>&1 && break; sleep 1; done
apt-get update -q
apt-get install -y -q --no-install-recommends git ca-certificates python3.13 python3.13-venv python3.13-dev \
  kmod pciutils procps build-essential ninja-build cmake pkg-config libssl-dev libnuma1 patch util-linux
if ! command -v nvidia-smi >/dev/null 2>&1; then
  : "${NVIDIA_RUN:?nvidia-smi not present: set NVIDIA_RUN=/path/to/NVIDIA-Linux-x86_64-<host-version>.run}"
  [ -f "$NVIDIA_RUN" ] || { echo "NVIDIA_RUN=$NVIDIA_RUN not found" >&2; exit 1; }
  sh "$NVIDIA_RUN" --silent --no-kernel-module --no-drm --no-x-check --no-nouveau-check --no-backup
fi
nvidia-smi -L || { echo "check the container GPU device mappings and the host driver" >&2; exit 1; }
# Reference host: NVIDIA driver 610.43.02 (reports CUDA 13.3). The release lock selects a matched CUDA 13.0 toolkit (nvcc, crt, nvvm, nvJitLink 13.0.88, cccl 13.0.85, runtime 13.0.96, the pair torch 2.13.0 cu130 ships with); the release wheel is served on it by every recipe (2026-10-20).
# Other CUDA 13.x drivers (R580 or later) are expected to work through CUDA minor-version
# compatibility, but are untested.
# Require the CUDA 13.x floor; see release/install-0.33.md.
cuda_release="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA \(UMD \)\{0,1\}Version: *\([0-9][0-9]*\.[0-9][0-9]*\).*/\2/p' | head -n 1 || true)"
if [ -n "$cuda_release" ] && [ "$(printf '%s\n%s\n' 13.0 "$cuda_release" | sort -V | tail -n 1)" != "$cuda_release" ]; then
  echo "the host driver supports CUDA $cuda_release; the locked CUDA wheels require CUDA 13.0 or later (R580 or later)" >&2
  exit 1
fi
CKPT="${CKPT:-/models/Qwen3.8-Flash-Next-W4A16-Merlin}"
[ -f "$CKPT/config.json" ] || {
  echo "checkpoint not found at $CKPT: mount an existing checkpoint before provisioning" >&2; exit 1;
}
if [ ! -d "$APP/.git" ]; then git clone -q --branch "$REPO_REF" "$REPO_URL" "$APP"; fi
cd "$APP"
# Refuse to reuse a checkout of a different release.
want="$(git rev-parse --verify "$REPO_REF^{commit}")"
[ "$(git rev-parse HEAD)" = "$want" ] || {
  echo "$APP is not at $REPO_REF; use a checkout of the requested release" >&2; exit 1;
}
[ -z "$(git status --porcelain --untracked-files=no)" ] || {
  echo "$APP has modified tracked files; review them before provisioning" >&2; exit 1;
}
[ -f release/requirements-0.33.lock ] || {
  echo "this checkout lacks the 0.33 dependency lock required by release/install-env.sh" >&2; exit 1;
}
if [ ! -e "$VENV" ] && [ ! -L "$VENV" ]; then
  for tool in uv cargo rustc; do
    command -v "$tool" >/dev/null || {
      echo "source installation requires $tool; prepare uv and the repository's Rust toolchain first" >&2
      exit 1
    }
  done
  bash release/install-env.sh "$VENV" "$APP"
fi
# Every path verifies source identity, every lock pin, dependency consistency,
# exact installed tile hashes and native products before service enablement.
bash release/install-env.sh --verify "$VENV" "$APP"
"$VENV/bin/python" models/flash-next/make-e1-config.py "$CKPT"
install -m 0644 models/flash-next/lxc/flash-next.service /etc/systemd/system/flash-next.service
# Preserve the user's settings while replacing every old TREE/VENV assignment.
"$VENV/bin/python" -I - "$APP" "$VENV" "$CKPT" <<'PY'
import os
import re
import stat
import sys
import tempfile
from pathlib import Path

app, venv, checkpoint = sys.argv[1:]
config = Path("/etc/default/flash-next")


def quote(value):
    if any(c in value for c in ("\n", "\r", "\0")):
        sys.exit("service paths must not contain newlines or NUL")
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


if config.is_symlink():
    sys.exit("service config must be a regular file")
previous = config.stat() if config.exists() else None
text = config.read_text(encoding="utf-8") if previous else (
    f"CKPT={quote(checkpoint)}\n"
    "HOST=127.0.0.1\nPORT=8000\nCACHE_ROOT=/cache\n"
    "# To serve other hosts, set HOST=0.0.0.0 together with VLLM_API_KEY (bearer authentication).\n"
)
lines = [
    line for line in text.splitlines()
    if not re.match(r"^\s*(?:export\s+)?(?:TREE|VENV)\s*=", line)
]
lines.extend((f"TREE={quote(app)}", f"VENV={quote(venv)}"))
fd, temporary = tempfile.mkstemp(prefix=".flash-next-", dir=config.parent)
try:
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        os.fchmod(stream.fileno(), stat.S_IMODE(previous.st_mode) if previous else 0o600)
        if previous:
            os.fchown(stream.fileno(), previous.st_uid, previous.st_gid)
        stream.write("\n".join(lines) + "\n")
    os.replace(temporary, config)
finally:
    if os.path.exists(temporary):
        os.unlink(temporary)
PY
systemctl daemon-reload
systemctl enable --now flash-next.service
echo "provisioned; follow startup with: journalctl -fu flash-next"
echo "the reference 4x RTX 3090 profile has a 924,993-token KV pool"
echo "readiness requires a completed generation; /v1/models alone is insufficient:"
echo "  curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{\"model\":\"flash-next\",\"messages\":[{\"role\":\"user\",\"content\":\"ok\"}],\"max_tokens\":1,\"chat_template_kwargs\":{\"enable_thinking\":false}}'"
