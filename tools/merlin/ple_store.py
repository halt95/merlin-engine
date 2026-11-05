# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""``--ple-store ram|disk``: what the tools know about the storage under the PLE table.

Setup offers disk on local NVMe only. ``merlin serve --ple-store disk`` on SATA, rotational,
network or unknown storage warns and runs. Every table file is checked by its real path, since an INT6
overlay is links and its pack can sit on another disk. Standard library only: the launcher imports this.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

STORE_ENV = "PLE_STORE"
STORES = ("ram", "disk")
CACHE_WARN_GIB = 16  # warn below the floor plus this much page cache
EXPERIMENTAL_NOTE = ("note: --ple-store disk is experimental: the PLE table is read from the checkpoint's files, so a cold table makes prefill "
                     "and decode slower until the page cache holds it (docs/how-to-serve.md)")
INDEX = "model.safetensors.index.json"
NETWORK_FS = ("nfs", "nfs4", "cifs", "smb3", "smbfs", "9p", "ceph", "lustre", "gpfs", "glusterfs", "afs", "beegfs", "virtiofs")
SYS_DEV_BLOCK = "/sys/dev/block"
SYS_BLOCK = "/sys/block"
MOUNTINFO = "/proc/self/mountinfo"
NAMES = {"nvme": "NVMe", "sata": "a SATA or other non-NVMe SSD", "rotational": "a rotating disk", "network": "a network filesystem",
         "unknown": "storage of a type that could not be read"}


def _fs_type(path: str, mountinfo: str = MOUNTINFO) -> str | None:
    """The filesystem type of the longest mount point holding ``path``."""
    best, kind = -1, None
    try:
        with open(mountinfo, encoding="utf-8", errors="replace") as f:
            for line in f:
                fields = line.split(" - ", 1)
                left = fields[0].split()
                if len(fields) != 2 or len(left) < 5:
                    continue
                mount = left[4].replace("\\040", " ")
                inside = path == mount or path.startswith(mount.rstrip("/") + "/")
                if inside and len(mount) >= best:
                    best, kind = len(mount), fields[1].split()[0]
    except OSError:
        return None
    return kind


def _leaf_disks(dev: str, sys_block: str, seen: set[str] | None = None) -> list[str]:
    """The whole disks under a block device name: a partition's parent, dm / md through ``slaves``."""
    seen = seen if seen is not None else set()
    if dev in seen:
        return []
    seen.add(dev)
    slaves = os.path.join(sys_block, dev, "slaves")
    try:
        under = sorted(os.listdir(slaves))
    except OSError:
        under = []
    if under:
        return [d for s in under for d in _leaf_disks(s, sys_block, seen)]
    if os.path.isdir(os.path.join(sys_block, dev)):
        return [dev]
    # a partition is not in /sys/block; its sysfs node sits under the parent disk
    try:
        for parent in os.listdir(sys_block):
            if os.path.isdir(os.path.join(sys_block, parent, dev)):
                return _leaf_disks(parent, sys_block, seen)
    except OSError:
        pass
    return []


def _disk_class(dev: str, sys_block: str) -> str:
    try:
        rotational = Path(sys_block, dev, "queue", "rotational").read_text().strip()
    except OSError:
        return "unknown"
    if rotational == "1":
        return "rotational"
    return "nvme" if dev.startswith("nvme") else "sata"


def device_class(path: str, *, sys_dev_block: str = SYS_DEV_BLOCK, sys_block: str = SYS_BLOCK, mountinfo: str = MOUNTINFO) -> str:
    """nvme | sata | rotational | network | unknown for the storage under ``path``. Several disks (LVM, md) take the
    slowest. ZFS, overlay and other filesystems with no block device behind ``st_dev`` are unknown."""
    real = os.path.realpath(path)
    fs = _fs_type(real, mountinfo)
    if fs and (fs in NETWORK_FS or fs.startswith("fuse")):
        return "network"
    try:
        st = os.stat(real)
    except OSError:
        return "unknown"
    node = os.path.join(sys_dev_block, f"{os.major(st.st_dev)}:{os.minor(st.st_dev)}")
    try:
        dev = os.path.basename(os.path.realpath(node, strict=True))
    except OSError:
        return "unknown"
    disks = _leaf_disks(dev, sys_block)
    if not disks:
        return "unknown"
    classes = {_disk_class(d, sys_block) for d in disks}
    for slowest in ("unknown", "rotational", "sata"):
        if slowest in classes:
            return slowest
    return "nvme"


def table_files(ckpt: str | Path) -> list[str]:
    """The real paths of the checkpoint's PLE table files, from its index; empty when it has none readable."""
    try:
        weight_map = json.loads((Path(ckpt) / INDEX).read_text(encoding="utf-8"))["weight_map"]
    except (OSError, ValueError, KeyError, TypeError):
        return []
    names = sorted({f for k, f in weight_map.items() if ".ngram_embedding.shard_" in k})
    return [os.path.realpath(os.path.join(str(ckpt), n)) for n in names]


def storage_classes(ckpt: str | Path, **probe: str) -> dict[str, str]:
    """{real table file, or the folder when it holds none yet: its device class}."""
    files = table_files(ckpt)
    if not files and Path(ckpt).is_dir():  # a pack folder: shards and a manifest, no index
        files = sorted(os.path.realpath(p) for p in Path(ckpt).glob("*.safetensors"))
    files = files or [str(ckpt)]
    return {f: device_class(f, **probe) for f in files}


def storage_warnings(ckpt: str | Path, **probe: str) -> list[str]:
    """``merlin serve --ple-store disk`` on storage setup would not offer: what to tell the user; it still runs."""
    by_class: dict[str, list[str]] = {}
    for path, cls in storage_classes(ckpt, **probe).items():
        if cls != "nvme":
            by_class.setdefault(cls, []).append(path)
    return [f"--ple-store disk: the PLE table is on {NAMES[cls]} ({paths[0]}{f' and {len(paths) - 1} more' if len(paths) > 1 else ''}); "
            "only local NVMe is expected to keep up, so decode and prefill can be far slower; serving anyway"
            for cls, paths in sorted(by_class.items())]


def offer_reason(ckpt: Path, **probe: str) -> str | None:
    """Why setup does not offer NVMe for the table in this folder; None when it does. A folder not there yet is
    judged by its nearest existing parent, where the download will land."""
    at = Path(ckpt)
    while not at.exists() and at != at.parent:
        at = at.parent
    for cls in storage_classes(at, **probe).values():
        if cls != "nvme":
            return f"{at} is on {NAMES[cls]}, and the disk option needs local NVMe"
    return None
