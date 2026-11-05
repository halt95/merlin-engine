#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
# Proxmox VE hookscript for the Flash-Next container (lxc/pve-create.sh): create the NVIDIA device nodes on the HOST
# before the container starts. After a host reboot the /dev/nvidia* nodes (the uvm ones in particular) do not exist
# until something opens the driver, and `pct start` then fails on the missing device passthrough.
#
# Proxmox calls a hookscript as `<script> <vmid> <phase>`, phase = pre-start | post-start | pre-stop | post-stop;
# a non-zero exit in pre-start aborts the start. Install on the Proxmox host (the storage must have the Snippets
# content type enabled, and the script must be executable):
#
#   install -m 0755 lxc/nvidia-prestart.sh /var/lib/vz/snippets/nvidia-prestart.sh
#   pct set <CTID> --hookscript local:snippets/nvidia-prestart.sh
vmid="$1"
phase="$2"
[ "$phase" = pre-start ] || exit 0
# `nvidia-smi -L` creates the full set, including /dev/nvidia-uvm and /dev/nvidia-uvm-tools
# (`nvidia-modprobe -u -c0` alone does not create the uvm nodes).
if ! nvidia-smi -L >/dev/null 2>&1; then
  echo "nvidia-prestart: nvidia-smi cannot reach the NVIDIA driver on the host; not starting CT $vmid" >&2
  exit 1
fi
for d in /dev/nvidia-uvm /dev/nvidia-uvm-tools; do
  [ -e "$d" ] || { echo "nvidia-prestart: $d is missing after nvidia-smi -L; not starting CT $vmid" >&2; exit 1; }
done
exit 0
