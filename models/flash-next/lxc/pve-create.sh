#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
# Create a Proxmox VE LXC container for Flash-Next, modelled on the reference host: a PRIVILEGED Debian 13
# container with the four cards passed through as devices. Run ON THE PROXMOX HOST as root:
#
#   CTID=201 MODELS=/path/to/models CACHE=/path/to/flash-next-cache lxc/pve-create.sh
#
# Then: install -m 0755 lxc/nvidia-prestart.sh /var/lib/vz/snippets/nvidia-prestart.sh;
#       pct set $CTID --hookscript local:snippets/nvidia-prestart.sh;
#       pct start $CTID; pct push $CTID lxc/provision.sh /root/provision.sh;
#       pct exec $CTID -- env NVIDIA_RUN=/models/NVIDIA-Linux-x86_64-<host version>.run bash /root/provision.sh
#
# Needs Proxmox VE 8.1 or later (`pct set --devN` device passthrough) and a PVE whose appliance index lists the
# debian-13-standard template.
#
# Variables: CTID (required), TEMPLATE (debian-13-standard, downloaded if absent), STORAGE (rootfs storage, local-lvm),
# ROOTFS_GB (32), CORES (12), MEMORY_MB (98304 = 96 GB, the qualified allocation: host RAM holds the ~48 GiB FP8 n-gram
# table (ordinary memory) and ~4.2 GiB of pinned embedding tables; measured resident floor ~69 GiB, the rest is
# reclaimable page cache),
# BRIDGE (vmbr0), MODELS (host directory holding Qwen3.8-Flash-Next-W4A16-Merlin/; mounted at /models),
# CACHE (host directory for the compile cache; mounted at /cache), NVIDIA_DEVS (the /dev/nvidia* nodes to pass).
# The NVIDIA kernel module runs on the host; only its userspace goes inside (provision.sh).
set -euo pipefail
: "${CTID:?set CTID}"
TEMPLATE="${TEMPLATE:-debian-13-standard}"; STORAGE="${STORAGE:-local-lvm}"; ROOTFS_GB="${ROOTFS_GB:-32}"
CORES="${CORES:-12}"; MEMORY_MB="${MEMORY_MB:-98304}"; BRIDGE="${BRIDGE:-vmbr0}"
# PRIVILEGED by default, which is what the reference host runs. In an unprivileged container root is mapped to a
# high uid, and the NVIDIA device nodes are not all world-accessible (/dev/nvidia-uvm-tools is 0660 root:root on a
# stock driver install), so the container's root is denied access even though `pct set --devN` adds the cgroup rule.
# Making unprivileged work needs idmap entries and device-node ownership changes this script does not perform and we
# have not qualified. Set UNPRIVILEGED=1 only if you intend to do that yourself.
UNPRIVILEGED="${UNPRIVILEGED:-0}"
MODELS="${MODELS:?set MODELS to the host directory that contains Qwen3.8-Flash-Next-W4A16-Merlin/}"
CACHE="${CACHE:-/var/lib/flash-next-cache}"
NVIDIA_DEVS="${NVIDIA_DEVS:-/dev/nvidia0 /dev/nvidia1 /dev/nvidia2 /dev/nvidia3 /dev/nvidiactl /dev/nvidia-uvm /dev/nvidia-uvm-tools}"
# The device nodes must already exist ON THE HOST: the kernel module stays there and only its userspace goes inside
# the container. `nvidia-modprobe -u -c0` creates /dev/nvidia0 and nvidiactl but NOT the uvm nodes; running
# `nvidia-smi -L` once as root creates the full set and is the simplest reliable preparation. If the cards are bound
# to vfio-pci for a VM, unbind them first. None of this survives a host reboot by itself: attach a Proxmox pre-start
# hookscript that runs `nvidia-smi -L` (lxc/nvidia-prestart.sh).
for d in $NVIDIA_DEVS; do [ -e "$d" ] || {
  echo "missing $d on the host."
  echo "  Load the NVIDIA kernel driver and create its device nodes on the HOST first:  nvidia-smi -L"
  echo "  (that creates /dev/nvidia* including the uvm nodes; nvidia-modprobe -u -c0 alone does not)"
  echo "  If the cards are bound to vfio-pci for a VM, unbind them from vfio-pci first."
  echo "  To survive host reboots, attach lxc/nvidia-prestart.sh as the CT's pre-start hookscript (docs/reference.md, Proxmox LXC)."
  exit 1; }; done
mkdir -p "$CACHE"
[ -f "$MODELS/Qwen3.8-Flash-Next-W4A16-Merlin/config.json" ] || echo "note: $MODELS/Qwen3.8-Flash-Next-W4A16-Merlin/config.json not found; download the checkpoint there before provision.sh (it checks for it)"
# (|| true: under pipefail awk's early exit can fail the pipeline; an empty result is handled below)
tpl="$(pveam list local | awk -v t="$TEMPLATE" '$1 ~ t {print $1; exit}' || true)"
if [ -z "$tpl" ]; then
  name="$(pveam available --section system | awk -v t="$TEMPLATE" '$2 ~ t {print $2; exit}' || true)"
  [ -n "$name" ] || { echo "no $TEMPLATE template available (pveam available)"; exit 1; }
  pveam download local "$name"; tpl="local:vztmpl/$name"
fi
pct create "$CTID" "$tpl" \
  --hostname flash-next --ostype debian --unprivileged "$UNPRIVILEGED" --features nesting=1 \
  --cores "$CORES" --memory "$MEMORY_MB" --swap 0 \
  --rootfs "$STORAGE:$ROOTFS_GB" \
  --net0 "name=eth0,bridge=$BRIDGE,ip=dhcp" \
  --mp0 "$MODELS,mp=/models" --mp1 "$CACHE,mp=/cache" \
  --onboot 0
i=0
for d in $NVIDIA_DEVS; do pct set "$CTID" --dev$i "$d"; i=$((i + 1)); done
echo "created CT $CTID: $CORES cores, $((MEMORY_MB / 1024)) GB, $i NVIDIA device nodes, /models=$MODELS, /cache=$CACHE, unprivileged=$UNPRIVILEGED"
echo "reboot-safe device nodes: install -m 0755 $(dirname "$0")/nvidia-prestart.sh /var/lib/vz/snippets/nvidia-prestart.sh && pct set $CTID --hookscript local:snippets/nvidia-prestart.sh"
echo "next: pct start $CTID && pct push $CTID $(dirname "$0")/provision.sh /root/provision.sh && pct exec $CTID -- env NVIDIA_RUN=/models/NVIDIA-Linux-x86_64-<host version>.run bash /root/provision.sh"
