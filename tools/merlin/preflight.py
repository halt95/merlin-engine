#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Merlin preflight: read the hardware, pick a launch profile, print the label.

Standard library only, no GPU and no CUDA context. It reads ``nvidia-smi`` and ``/proc`` (or a described
inventory), compares them with the profile registry in ``tools/merlin/profiles/`` and prints

* what it found (GPU arch and count, VRAM, BAR1, P2P, host RAM, driver, virtualisation),
* which profile it picked (or why none fits) and that profile's status label:
  ``tested on the reference host``, ``🟡 testing: ...`` (full-model runs passed on
  cloud-hosted hardware; kernel parity qualification still to run) or
  ``untested / community-validated``,
* PASS / WARN / FAIL checks for the host prerequisites.

It never starts a server, never writes outside ``--write-launchers`` and ``--emit``, and measures nothing.

    python tools/merlin/preflight.py                       # probe this host
    python tools/merlin/preflight.py --dry-run             # no host access; simulated reference host
    python tools/merlin/preflight.py --simulate 8x3090     # described hardware, no host access
    python tools/merlin/preflight.py --inventory host.json # a captured inventory
    python tools/merlin/preflight.py --model 27b --simulate 2x3090
    python tools/merlin/preflight.py --list
    python tools/merlin/preflight.py --emit launcher --profile fn-octo-3090-tp2pp4

Exit codes: 0 a profile was picked and no check failed; 1 a check failed (or warned under --strict);
2 usage or input error; 3 no profile fits this hardware.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import subprocess
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gpu_unlock as U  # noqa: E402
import ple_store as PS  # noqa: E402
import registry as R  # noqa: E402

GIB = 2**30
CTX_262K = 262_144
CTX_131K = 131_072
MIB = 2**20
SCHEMA = 1

# Thresholds follow models/flash-next/docs/reference.md.
RAM_WARN_GIB = 96.0  # qualified allocation
RAM_FLOOR_GIB = 69.0  # measured resident floor at four workers: below it the load OOMs
# Capped-host bisection (systemd scope memory.max, swap 0, tier off, fn-swift1.5-quad-3090, the reference host):
# 64 GiB was OOM-killed at boot, 72 and 76 GiB booted and passed 262K identity + needle; the uncapped footprint was 68.4 GiB. The floor lies in (64, 72]; 72 is the
# tested minimum. launch.py carries both (FN_RAM_FLOOR_KB, FN_RAM_TESTED_MIN_GIB) and refuses below the floor.
RAM_PER_EXTRA_WORKER_GIB = (
    2.0  # estimate: ~8 GiB across four workers and the engine
)
# The PLE-table term inside RAM_FLOOR_GIB: the FP8 table every Flash-Next row pins today (~47.7 GiB). This is the one
# place the floor's table size lives. A checkpoint carrying the INT6 g32 table (--ple int6: config.json
# text_config.ple_embedding_dtype) swaps it for FN_PLE_OPT_IN_GIB's; the rest of the floor is unchanged.
FN_PLE_TABLE_GIB = 47.68
FN_PLE_OPT_IN_GIB = {"int6_g32": 38.74}  # launch.PLE_OPT_IN_TABLES (the launcher's floor swaps the same term)
FN_PLE_TEST_GIB = {"bfloat16": 95.37, "int8_row": 48.28}  # launch.PLE_TEST_TABLES (boot only on request)
# The host-RAM KV tier: GiB of host RAM per rank (TP x PP), 0 = off. Every RAM gate adds it on top
# of the floor, per server; merlin serve --kv-ram-cache sets it per rank and launch.py passes it to the engine.
KV_HOST_TIER_ENV = "MERLIN_KV_HOST_TIER_GIB"
SHM_MIN_MIB = 1024
BAR1_MIN_MIB = 1024  # a card below 1 GiB BAR1 hangs NCCL warmup on any P2P pair that includes it
BUSY_MIB = 1024  # the KV pool is pinned in bytes and leaves ~1 GB per card
KV_MIN_BYTES = 1 << 30  # the 27B / Super VL launcher refuses KV_BYTES under 1 GiB (launch.py); preflight refuses the same pin


def min_restore_unit_gib_per_rank(ranks: int, model: str | None = None, profile: dict | None = None) -> int:
    """Whole GiB per rank needed for one 262K restore unit: from the profile's host row size when it is recorded
    (registry.restore_unit_min_gib_per_rank), else 4 GiB per server."""
    return R.restore_unit_min_gib_per_rank(model, profile, ranks)


DRIVER_MIN_MAJOR = 580  # CUDA 13.0 driver: the 580 series or later


# --------------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------------
@dataclass
class Gpu:
    index: int
    name: str
    arch: str | None  # "sm_86", or None when unknown
    vram_mib: int
    bar1_mib: int | None = None
    mem_used_mib: int | None = None
    pcie_gen: int | None = None  # maximum generation the link can train to
    pcie_width: int | None = None
    persistence: bool | None = None
    power_limit_w: float | None = None
    bus_id: str | None = None
    numa_node: int | None = None  # the NUMA node the card hangs off (nvidia-smi topo -m, else /sys), None when unknown
    pcie_link: dict[str, int | None] | None = None  # sysfs current/max gen and width; information only


@dataclass
class Host:
    ram_mib: int | None = None
    shm_mib: int | None = None
    virt: str | None = None  # "bare", "container", "vm", or None when unknown
    driver: str | None = None
    system: str = "linux"
    iommu_cmdline: list[str] | None = None  # requested settings, not proof of the effective mode
    iommu_groups: int | None = None  # visible groups; zero does not prove IOMMU is disabled
    iommu_group_types: list[str] | None = None  # kernel-reported domain types (e.g. identity, DMA)


@dataclass
class Inventory:
    gpus: list[Gpu]
    host: Host
    p2p: str = "unknown"  # driver view: "on" (every pair OK), "off" (no pair OK), "partial", "unknown"
    source: str = "probe"  # "probe", "inventory:<file>", "simulate:<spec>", "dry-run"
    notes: list[str] = field(default_factory=list)
    p2p_ok: list[list[int]] | None = None  # the GPU index pairs the driver reports P2P OK (None = not read)
    links: list[list[Any]] | None = None  # [a, b, kind] per GPU pair from nvidia-smi topo -m (PIX / PXB / PHB / NODE / SYS / NV#)
    kv_tier_gib: int = 0  # the serve setting the RAM gates count: MERLIN_KV_HOST_TIER_GIB per rank (kv_host_tier_gib), 0 = off
    ple_table_gib: float = FN_PLE_TABLE_GIB  # the checkpoint's PLE table in host RAM (ple_table_gib_of): FP8, or INT6 g32

    def link(self, a: int, b: int) -> str | None:
        """The PCIe / NVLink relation between two cards as ``nvidia-smi topo -m`` prints it; None when not read."""
        if self.links is None:
            return None
        for x, y, kind in self.links:
            if (x, y) == (a, b) or (x, y) == (b, a):
                return str(kind)
        return None

    def p2p_among(self, gpus: list[int], driver_view: bool = False) -> str:
        """The effective P2P view restricted to ``gpus``: on / off / partial, or the host view when pairs are unknown.
        A card with BAR1 under 1 GiB has no usable peer (NCCL hangs at warmup on any P2P pair with it), so its pairs
        count as off whatever the driver says; ``driver_view`` gives the driver's own answer for the messages."""
        small = set() if driver_view else {g.index for g in self.gpus if g.bar1_mib is not None and g.bar1_mib < BAR1_MIN_MIB}
        if len(gpus) < 2:
            return self.p2p
        if self.p2p_ok is None:
            return "off" if (small & set(gpus) and self.p2p == "on") else self.p2p
        ok = {(a, b) for a, b in self.p2p_ok if a not in small and b not in small}
        ok |= {(b, a) for a, b in ok}
        states = [(a, b) in ok for a in gpus for b in gpus if a != b]
        if all(states):
            return "on"
        if not any(states):
            return "off"
        return "partial"


@dataclass
class Finding:
    level: str  # PASS, INFO, WARN, FAIL
    code: str
    text: str


@dataclass
class Report:
    model: str
    inventory: Inventory
    profile_id: str | None = None
    profile_title: str | None = None
    status: str | None = None  # "tested" | "testing" | "untested" (| "parked")
    label: str | None = None
    p2p_mode: str | None = None  # "on" | "off" | None
    effective_p2p: str = "unknown"
    layout: dict[str, Any] | None = None
    serve: dict[str, Any] | None = None
    capacity: dict[str, Any] | None = None
    launcher: str | None = None
    start: str | None = None
    differs: list[str] = field(default_factory=list)
    alternatives: list[str] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    no_profile_reason: str | None = None
    adjusted: dict[str, Any] | None = None  # capacity re-scaled to the reported VRAM (an estimate)
    gpu_unlock: dict[str, Any] | None = None  # live probes only: gpu_unlock.probe (informational, never gates)
    exit_code: int = 0

    def add(self, level: str, code: str, text: str) -> None:
        self.findings.append(Finding(level, code, text))


# --------------------------------------------------------------------------------------------
# Known cards (simulation and arch fallback by name)
# --------------------------------------------------------------------------------------------
# key: (display name, arch, VRAM MiB as nvidia-smi reports it, stock driver has P2P?)
KNOWN_CARDS: dict[str, tuple[str, str, int, bool]] = {
    "3090": ("NVIDIA GeForce RTX 3090", "sm_86", 24576, True),
    "3090ti": ("NVIDIA GeForce RTX 3090 Ti", "sm_86", 24564, True),
    # Cloud / small-VRAM cards. MiB are what nvidia-smi reports: A10/A10G/L4 ~22.5 GiB (below the 24 GiB class, so the
    # KV pin is lowered by the shortfall); the 12 GB cards fall outside every class and are refused. The 20 GB cards are
    # the 20 GiB class: untested 27B TP2 / TP4 rows, and untested Flash-Next quad rows with the static cold-expert offload.
    "a10": ("NVIDIA A10", "sm_86", 23028, False),
    "a10g": ("NVIDIA A10G", "sm_86", 23028, False),
    "a4500": ("NVIDIA RTX A4500", "sm_86", 20470, True),
    "3080-20g": ("NVIDIA GeForce RTX 3080", "sm_86", 20480, True),
    "4000ada": ("NVIDIA RTX 4000 Ada Generation", "sm_89", 20475, False),
    "3080ti": ("NVIDIA GeForce RTX 3080 Ti", "sm_86", 12288, True),
    "3080-12g": ("NVIDIA GeForce RTX 3080", "sm_86", 12288, True),
    "l4": ("NVIDIA L4", "sm_89", 23034, False),
    # Modified consumer Ada cards: the 32 GB RTX 4080 and the 48 GB RTX 4090 report ~32,760 / ~49,140 MiB (inside the 32 / 48 GiB classes).
    "4080-32g": ("NVIDIA GeForce RTX 4080", "sm_89", 32760, False),
    "4090-48g": ("NVIDIA GeForce RTX 4090", "sm_89", 49140, False),
    "a5000": ("NVIDIA RTX A5000", "sm_86", 24564, True),
    "a6000": ("NVIDIA RTX A6000", "sm_86", 49140, True),
    "a40": ("NVIDIA A40", "sm_86", 46068, True),
    "a30": ("NVIDIA A30", "sm_80", 24576, True),
    "a100-40g": ("NVIDIA A100-PCIE-40GB", "sm_80", 40960, True),
    "a100-80g": ("NVIDIA A100 80GB PCIe", "sm_80", 81920, True),
    "170hx-32g": ("NVIDIA CMP 170HX", "sm_80", 32768, False),
    "170hx-40g": ("NVIDIA CMP 170HX", "sm_80", 40960, False),
    "170hx-64g": ("NVIDIA CMP 170HX", "sm_80", 65536, False),
    "4090": ("NVIDIA GeForce RTX 4090", "sm_89", 24564, False),
    "4080s-32g": ("NVIDIA GeForce RTX 4080 SUPER", "sm_89", 32760, False),
    "l40s": ("NVIDIA L40S", "sm_89", 46068, False),
    "6000ada": ("NVIDIA RTX 6000 Ada Generation", "sm_89", 49140, False),
    "2080ti": ("NVIDIA GeForce RTX 2080 Ti", "sm_75", 11264, False),
    "h100": ("NVIDIA H100 80GB HBM3", "sm_90", 81559, True),
}

_NAME_TO_ARCH = [
    (r"RTX\s*30\d\d|RTX\s*A\d{4}|\bA40\b|\bA10G?\b|\bA16\b", "sm_86"),
    (r"\bA100\b|\bA30\b|CMP\s*170HX", "sm_80"),
    (r"RTX\s*40\d\d|\bL40S?\b|\bL4\b|RTX\s*6000\s*Ada|RTX\s*5000\s*Ada", "sm_89"),
    (r"RTX\s*20\d\d|\bT4\b|TITAN RTX", "sm_75"),
    (r"\bH100\b|\bH200\b|\bGH200\b", "sm_90"),
]


def arch_from_name(name: str) -> str | None:
    for pat, arch in _NAME_TO_ARCH:
        if re.search(pat, name, re.IGNORECASE):
            return arch
    return None


def arch_from_cc(cc: str | None) -> str | None:
    m = re.fullmatch(r"\s*(\d+)\.(\d+)\s*", cc or "")
    return f"sm_{m.group(1)}{m.group(2)}" if m else None


# --------------------------------------------------------------------------------------------
# Parsers (pure; exercised by the tests with captured outputs)
# --------------------------------------------------------------------------------------------
def _num(s: str) -> float | None:
    s = s.strip()
    if not s or s.startswith("[") or s.lower() in ("n/a", "not supported"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


QUERY_FIELDS = (
    "index,name,memory.total,memory.used,pci.bus_id,pcie.link.gen.max,pcie.link.width.current,"
    "persistence_mode,power.limit,compute_cap,driver_version"
)


def parse_query_csv(text: str) -> tuple[list[Gpu], str | None]:
    """Parse ``nvidia-smi --query-gpu=<QUERY_FIELDS> --format=csv,noheader,nounits``."""
    gpus: list[Gpu] = []
    driver: str | None = None
    for row in csv.reader(io.StringIO(text.strip())):
        if len(row) < 11:
            continue
        r = [c.strip() for c in row]
        idx = _num(r[0])
        vram = _num(r[2])
        if idx is None or vram is None:
            continue
        arch = arch_from_cc(r[9]) or arch_from_name(r[1])
        gen, width, pl = _num(r[5]), _num(r[6]), _num(r[8])
        used = _num(r[3])
        gpus.append(
            Gpu(
                index=int(idx),
                name=r[1],
                arch=arch,
                vram_mib=int(vram),
                mem_used_mib=int(used) if used is not None else None,
                bus_id=r[4] or None,
                pcie_gen=int(gen) if gen is not None else None,
                pcie_width=int(width) if width is not None else None,
                persistence={"enabled": True, "disabled": False}.get(r[7].lower()),
                power_limit_w=pl,
            )
        )
        driver = driver or (r[10] or None)
    return gpus, driver


def parse_bar1(text: str) -> list[int]:
    """BAR1 total (MiB) per GPU, in order, from ``nvidia-smi -q -d MEMORY``."""
    out: list[int] = []
    for block in re.split(r"(?m)^GPU\s+[0-9A-Fa-f:.]+\s*$", text)[1:]:
        m = re.search(r"BAR1 Memory Usage\s*\n\s*Total\s*:\s*(\d+)\s*MiB", block)
        if m:
            out.append(int(m.group(1)))
    return out


def parse_p2p_pairs(text: str) -> list[list[int]] | None:
    """The (row, col) GPU index pairs ``nvidia-smi topo -p2p r`` reports OK; None when nothing parsed."""
    pairs: list[list[int]] = []
    seen = False
    for line in text.splitlines():
        m = re.match(r"\s*GPU(\d+)\s+(.*)$", line)
        if not m:
            continue
        row = int(m.group(1))
        toks = m.group(2).split()
        if any(re.fullmatch(r"GPU\d+", t) for t in toks):
            continue
        seen = True
        for col, tok in enumerate(toks):
            if col != row and tok.upper() == "OK":
                pairs.append([row, col])
    return pairs if seen else None


def parse_p2p_matrix(text: str) -> str:
    """Summarise ``nvidia-smi topo -p2p r`` as on / off / partial / unknown."""
    states: list[bool] = []
    for line in text.splitlines():
        m = re.match(r"\s*GPU(\d+)\s+(.*)$", line)
        if not m:
            continue
        row = int(m.group(1))
        toks = m.group(2).split()
        if any(
            re.fullmatch(r"GPU\d+", t) for t in toks
        ):  # the header row: GPU0 GPU1 GPU2 ...
            continue
        for col, tok in enumerate(toks):
            if col == row or tok == "X":
                continue
            states.append(tok.upper() == "OK")
    if not states:
        return "unknown"
    if all(states):
        return "on"
    if not any(states):
        return "off"
    return "partial"


def parse_topo_matrix(text: str) -> tuple[list[list[Any]], dict[int, int]]:
    """Parse ``nvidia-smi topo -m``: the GPU-to-GPU link kinds ([a, b, kind], a < b) and the NUMA affinity per GPU.

    The table is tab-separated: a header of column labels (GPU0.. GPUn, NICs, then ``CPU Affinity``, ``NUMA Affinity``,
    ``GPU NUMA ID``) and one row per device; the legend below is ignored. Returns ([], {}) when nothing parsed."""
    links: list[list[Any]] = []
    numa: dict[int, int] = {}
    cols: list[str] | None = None
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("Legend"):
            if cols is not None and not line.strip():
                break
            continue
        cells = [c.strip() for c in line.split("\t")] if "\t" in line else line.split()
        if cols is None:
            if any(re.fullmatch(r"GPU\d+", c) for c in cells):
                cols = cells if cells and cells[0] == "" else ["", *cells]
            continue
        m = re.fullmatch(r"GPU(\d+)", cells[0])
        if not m:
            continue
        row = int(m.group(1))
        for label, cell in zip(cols[1:], cells[1:]):
            g = re.fullmatch(r"GPU(\d+)", label)
            if g and int(g.group(1)) > row and cell and cell != "X":
                links.append([row, int(g.group(1)), cell])
            elif label == "NUMA Affinity" and re.fullmatch(r"\d+", cell):
                numa[row] = int(cell)
    return links, numa


def parse_meminfo_mib(text: str) -> int | None:
    m = re.search(r"(?m)^MemTotal:\s+(\d+)\s*kB", text)
    return int(m.group(1)) // 1024 if m else None


def classify_virt(detected: str | None) -> str | None:
    """Map ``systemd-detect-virt`` output to bare / container / vm."""
    if detected is None:
        return None
    d = detected.strip().lower()
    if d in ("none", ""):
        return "bare"
    if d == "wsl":
        return "wsl"
    if d in (
        "lxc",
        "lxc-libvirt",
        "docker",
        "podman",
        "openvz",
        "systemd-nspawn",
        "rkt",
        "proot",
        "pouch",
        "container-other",
    ):
        return "container"
    return "vm"


# --------------------------------------------------------------------------------------------
# Probing (real host) and simulation
# --------------------------------------------------------------------------------------------
Runner = Callable[[list[str]], "str | None"]


def default_runner(cmd: list[str]) -> str | None:
    try:
        cp = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if cp.returncode == 0:
        return cp.stdout
    # A failed nvidia-smi may still have printed part of the answer; a partial GPU probe would build a silently short
    # inventory. Other tools (systemd-detect-virt prints "none" and exits 1) keep their stdout.
    return cp.stdout if cp.stdout.strip() and cmd[:1] != ["nvidia-smi"] else None


def _read(path: str) -> str | None:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError:
        return None


def _listdir(path: str) -> list[str] | None:
    try:
        return os.listdir(path)
    except OSError:
        return None


def pcie_generation(speed: str | None) -> int | None:
    """Decode sysfs GT/s values; unsupported or unknown rates stay unknown."""
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s+GT/s(?:\s+PCIe)?\s*", speed or "")
    if not match:
        return None
    return {2.5: 1, 5.0: 2, 8.0: 3, 16.0: 4, 32.0: 5, 64.0: 6}.get(float(match[1]))


def probe_interconnect(
    inv: Inventory,
    read: Callable[[str], str | None] = _read,
    listdir: Callable[[str], list[str] | None] = _listdir,
) -> None:
    """Reads the per-card link and IOMMU snapshots for the report only."""
    for gpu in inv.gpus:
        link: dict[str, int | None] = {}
        base = f"/sys/bus/pci/devices/{sysfs_bus_id(gpu.bus_id)}/" if gpu.bus_id else None
        for state in ("current", "max"):
            speed = read(base + state + "_link_speed") if base else None
            width = (read(base + state + "_link_width") or "").strip() if base else ""
            link[state + "_gen"] = pcie_generation(speed)
            link[state + "_width"] = int(width) if width.isdecimal() and int(width) > 0 else None
        gpu.pcie_link = link

    host = inv.host
    cmdline = read("/proc/cmdline")
    host.iommu_cmdline = None if cmdline is None else [
        token for token in cmdline.split()
        if token.split("=", 1)[0] in ("iommu", "intel_iommu", "amd_iommu", "iommu.passthrough")
    ]
    root = "/sys/kernel/iommu_groups"
    entries = listdir(root)
    if entries is not None:
        groups = [name for name in entries if name.isdecimal()]
        host.iommu_groups = len(groups)
        types = [(read(f"{root}/{name}/type") or "").strip() for name in groups]
        host.iommu_group_types = sorted(set(t for t in types if t))
        if any(not t for t in types):
            host.iommu_group_types.append("unknown")


def interconnect_text(inv: Inventory) -> list[str]:
    """Per-GPU link snapshots and requested/visible IOMMU state, with explicit unknowns."""
    out = []
    for gpu in inv.gpus:
        # Older inventories only captured maximum generation and current width.
        link = gpu.pcie_link if gpu.pcie_link is not None else {
            "current_width": gpu.pcie_width, "max_gen": gpu.pcie_gen,
        }
        current = f"Gen{link.get('current_gen') or '?'} x{link.get('current_width') or '?'}"
        maximum = f"Gen{link.get('max_gen') or '?'} x{link.get('max_width') or '?'}"
        out.append(f"  GPU{gpu.index} ({gpu.bus_id or 'bus unknown'}): PCIe current {current}; max {maximum} (? = unknown)")
    host = inv.host
    cmdline = ("unknown" if host.iommu_cmdline is None
               else " ".join(host.iommu_cmdline) or "no explicit setting (kernel default unknown)")
    groups = "unknown" if host.iommu_groups is None else str(host.iommu_groups)
    types = ", ".join(host.iommu_group_types or []) or "unknown"
    out.append(f"  IOMMU: cmdline {cmdline}; visible groups {groups}; group types {types}")
    out.append("  PCIe current links may downshift at idle; IOMMU cmdline settings describe requested mode.")
    return out


def sysfs_bus_id(bus_id: str) -> str:
    """``00000000:41:00.0`` as nvidia-smi prints it -> ``0000:41:00.0`` as /sys/bus/pci/devices names it."""
    parts = bus_id.strip().lower().split(":")
    if len(parts) == 3 and re.fullmatch(r"[0-9a-f]+", parts[0]):
        return f"{int(parts[0], 16):04x}:{parts[1]}:{parts[2]}"
    return bus_id.strip().lower()


def _cgroup_bytes(read: Callable[[str], str | None], path: str) -> int | None:
    """A cgroup memory file as an integer; None when missing, ``max``, or a limit so large it means unlimited."""
    v = (read(path) or "").strip()
    if not v.isdigit() or int(v) >= 1 << 60:
        return None
    return int(v)


# The v2 and v1 memory-controller file sets: (mount, limit, usage, memory.stat key of the reclaimable page cache).
CGROUP_MEMORY_SETS = (
    ("/sys/fs/cgroup", "memory.max", "memory.current", "inactive_file"),
    ("/sys/fs/cgroup/memory", "memory.limit_in_bytes", "memory.usage_in_bytes", "total_inactive_file"),
)


def cgroup_memory_dirs(read: Callable[[str], str | None], mount: str, limit_file: str) -> list[str]:
    """The memory cgroup directories that bound this process under ``mount``, its own first and the mount (the cgroup
    namespace's root) last. The process's cgroup comes from /proc/self/cgroup (v2: the ``0::<path>`` line; v1: the
    line whose controllers include ``memory``), so a systemd ``MemoryMax`` on a unit or scope, or a nested slice's
    limit, is seen as well as one set at the container root (docker --memory, LXC). Just the mount, as before, when
    /proc/self/cgroup is unreadable, names no such line, names the root, or names a path that is not a cgroup under
    the mount (neither its limit file nor cgroup.procs reads: v1 without a cgroup namespace, say)."""
    v2 = mount == "/sys/fs/cgroup"
    rel = None
    for ln in (read("/proc/self/cgroup") or "").splitlines():
        parts = ln.strip().split(":", 2)
        if len(parts) != 3:
            continue
        if (v2 and parts[0] == "0" and parts[1] == "") or (not v2 and "memory" in parts[1].split(",")):
            rel = parts[2]
            break
    segs = [x for x in (rel or "").split("/") if x and x not in (".", "..")]
    dirs = [mount + "/" + "/".join(segs[:i]) for i in range(len(segs), 0, -1)]
    if not dirs or (read(f"{dirs[0]}/{limit_file}") is None and read(f"{dirs[0]}/cgroup.procs") is None):
        return [mount]
    return dirs + [mount]


def host_ram_mib(read: Callable[[str], str | None] = _read) -> int | None:
    """MemTotal, lowered to the smallest finite cgroup memory limit on the way from this process's cgroup up to the
    root (as serve.sh does at the root; ``cgroup_memory_dirs``)."""
    total = parse_meminfo_mib(read("/proc/meminfo") or "")
    for mount, lim_name, _use, _key in CGROUP_MEMORY_SETS:
        for d in cgroup_memory_dirs(read, mount, lim_name):
            lim = _cgroup_bytes(read, f"{d}/{lim_name}")
            if lim is not None:
                lim_mib = lim // MIB
                if total is None or lim_mib < total:
                    total = lim_mib
    return total


def _cgroup_stat(read: Callable[[str], str | None], path: str, key: str) -> int | None:
    """One ``key value`` line of a cgroup ``memory.stat``; None when the file or the key is missing."""
    m = re.search(rf"(?m)^{re.escape(key)}\s+(\d+)\s*$", read(path) or "")
    return int(m.group(1)) if m else None


def mem_available_mib(read: Callable[[str], str | None] = _read) -> int | None:
    """What a new server can still take: the host's MemAvailable, capped by what every cgroup level from this process's
    own up to the root has left (``cgroup_memory_dirs``; the smallest wins). A cgroup's usage counts the page cache it
    holds, which the kernel reclaims on demand, so the reclaimable part (``inactive_file`` in v2 ``memory.stat``,
    ``total_inactive_file`` in v1) is taken out first, the way container runtimes report working-set memory:
    left = max - max(0, current - inactive_file). A missing stat subtracts nothing; an unlimited or unreadable level
    adds no cap; None when nothing could be read."""
    avail: int | None = None
    m = re.search(r"(?m)^MemAvailable:\s+(\d+)\s*kB", read("/proc/meminfo") or "")
    if m:
        avail = int(m.group(1)) // 1024
    for mount, lim_name, use_name, key in CGROUP_MEMORY_SETS:
        for d in cgroup_memory_dirs(read, mount, lim_name):
            lim, use = _cgroup_bytes(read, f"{d}/{lim_name}"), _cgroup_bytes(read, f"{d}/{use_name}")
            if lim is not None and use is not None:
                use = max(0, use - (_cgroup_stat(read, f"{d}/memory.stat", key) or 0))
                left = max(0, lim - use) // MIB
                if avail is None or left < avail:
                    avail = left
    return avail


def ple_pin_text(ple_gib: float) -> str:
    """How the RAM messages name the table: pinned in RAM, or on disk (``--ple-store disk``, ``ple_gib`` 0)."""
    return f"pins a ~{ple_gib:.1f} GiB PLE table" if ple_gib else "reads its PLE table from disk"


def ram_available_text(cards: int, avail_mib: int, tier_gib: int = 0, ple_gib: float = FN_PLE_TABLE_GIB) -> str:
    """serve's refusal (and doctor's FAIL) when the memory a host has left is under a Flash-Next server's floor (with
    the KV host tier, ``tier_gib`` per rank, when on)."""
    return (f"host memory available now is {_gib(avail_mib):.0f} GiB (MemAvailable, capped by what the cgroup has left); a "
            f"{cards}-card Flash-Next server {ple_pin_text(ple_gib)} and needs "
            f"~{ram_floor_gib(cards, tier_gib, ple_gib):.0f} GiB resident{ram_tier_text(cards, tier_gib)} "
            f"(another Merlin server or a large page cache user?)")


def ram_floor_gib(cards: int, tier_gib: int = 0, ple_gib: float = FN_PLE_TABLE_GIB) -> float:
    """The resident host RAM one Flash-Next server needs: the measured four-worker floor plus one allowance per worker
    above four (a TP2xPP4 octo = 69 + 4 x 2 = 77 GiB), plus the KV host tier's arena, ``tier_gib`` per rank
    (``cards`` = TP x PP ranks; 0 = off adds nothing), with the FP8 table term swapped for ``ple_gib`` (INT6 g32:
    69 - 47.68 + 38.74 = 60.06 GiB at four workers; an estimate, no capped-host boot record yet; 0 under
    --ple-store disk: 21.32 GiB)."""
    return RAM_FLOOR_GIB - FN_PLE_TABLE_GIB + ple_gib + max(0, cards - 4) * RAM_PER_EXTRA_WORKER_GIB + max(0, tier_gib) * cards


def ple_table_gib_of(ckpt: str | None, default: float = FN_PLE_TABLE_GIB, store: str | None = None) -> float:
    """The host-RAM size of the PLE table the checkpoint in ``ckpt`` carries (text_config.ple_embedding_dtype, read the
    way the launcher reads it): the INT6 g32 opt-in table's or a test table's, else the FP8 one; ``default`` when there
    is no readable config.json (a folder setup has not built yet: setup passes the table it will build). 0 under
    ``store`` disk: no table in RAM, so every RAM gate takes the full credit."""
    if store == "disk":
        return 0.0
    if not ckpt:
        return default  # no folder: never the current directory's config.json
    try:
        with open(os.path.join(ckpt, "config.json"), encoding="utf-8") as f:
            ple = (json.load(f).get("text_config") or {}).get("ple_embedding_dtype")
    except (OSError, ValueError, AttributeError, TypeError):
        return default
    return {**FN_PLE_OPT_IN_GIB, **FN_PLE_TEST_GIB}.get(ple, FN_PLE_TABLE_GIB)


def ram_tier_text(cards: int, tier_gib: int) -> str:
    """The clause the RAM messages add when the KV host tier is on; empty when it is off."""
    if tier_gib <= 0:
        return ""
    return f" (including a KV host tier of {tier_gib} GiB x {cards} ranks = {tier_gib * cards} GiB)"


def dflash2_tier_conflicts(model: str | None, env: dict[str, str], saved: dict | None) -> list[str]:
    """A KV RAM cache with DFlash2 on the Flash-Next family, which merlin serve refuses: from the
    environment (SPEC=dflash2 + MERLIN_KV_HOST_TIER_GIB) or from setup's saved answers (.merlin/setup.json)."""
    out = []
    if model == "flash-next" and (env.get("SPEC") or "").strip() == "dflash2" and kv_host_tier_gib(env) > 0:
        out.append(f"SPEC=dflash2 with {KV_HOST_TIER_ENV}={kv_host_tier_gib(env)}: the KV RAM cache is qualified with MTP "
                   "only; merlin serve refuses the pair. Unset SPEC (MTP) or set "
                   f"{KV_HOST_TIER_ENV}=0")
    saved = saved if isinstance(saved, dict) else {}
    models = saved.get("models") if isinstance(saved.get("models"), dict) else {}
    name = saved.get("model")
    block = models.get(name) if isinstance(name, str) else None
    gib = block.get("kv_ram_cache_gib") if isinstance(block, dict) else None
    if (isinstance(block, dict) and block.get("spec") == "dflash2" and isinstance(gib, int)
            and not isinstance(gib, bool) and gib > 0):
        out.append(f"saved setup answers for {name} (.merlin/setup.json) pair --spec dflash2 with --kv-ram-cache {gib}g; "
                   "merlin serve refuses the pair. Run ./setup.sh --kv-ram-cache 0 or ./setup.sh --spec mtp")
    return out


def saved_ple_table_gib(model: str | None, saved: dict | None) -> float:
    """doctor's PLE table: setup's saved answer for the current model (.merlin/setup.json ple_table int6 -> INT6 g32),
    else the FP8 table; merlin serve reads the served checkpoint's config instead (ple_table_gib_of)."""
    saved = saved if isinstance(saved, dict) else {}
    models = saved.get("models") if isinstance(saved.get("models"), dict) else {}
    block = models.get(saved.get("model")) if isinstance(saved.get("model"), str) else None
    if model == "flash-next" and isinstance(block, dict) and block.get("ple_store") == "disk":
        return 0.0
    if model == "flash-next" and isinstance(block, dict) and block.get("ple_table") == "int6":
        return FN_PLE_OPT_IN_GIB["int6_g32"]
    return FN_PLE_TABLE_GIB


def _saved_setup(repo: Path) -> dict | None:
    try:
        return json.loads((repo / ".merlin" / "setup.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def kv_host_tier_gib(env: dict[str, str] | None = None) -> int:
    """The KV host tier the environment asks for, GiB per rank (``MERLIN_KV_HOST_TIER_GIB``); 0 when unset, 0 or not a
    whole number (launch.py refuses a malformed value itself)."""
    v = ((os.environ if env is None else env).get(KV_HOST_TIER_ENV) or "").strip()
    return int(v) if v.isdigit() else 0


def probe(
    run: Runner = default_runner,
    read: Callable[[str], str | None] = _read,
    env: dict[str, str] | None = None,
    listdir: Callable[[str], list[str] | None] = _listdir,
) -> Inventory:
    env = os.environ if env is None else env
    host = Host(system="linux" if sys.platform.startswith("linux") else sys.platform)
    inv = Inventory(gpus=[], host=host, source="probe")
    out = run(
        ["nvidia-smi", f"--query-gpu={QUERY_FIELDS}", "--format=csv,noheader,nounits"]
    )
    if (
        out is None
    ):  # older drivers reject compute_cap: retry without it and leave that column empty
        fields = QUERY_FIELDS.replace("compute_cap,", "")
        out2 = run(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"]
        )
        if out2:
            buf = io.StringIO()
            w = csv.writer(buf, lineterminator="\n")
            for row in csv.reader(io.StringIO(out2.strip())):
                w.writerow(row[:9] + [""] + row[9:])
            out = buf.getvalue()
    if not out:
        inv.notes.append(
            "nvidia-smi is missing or returned nothing: no NVIDIA GPU is visible to this host"
        )
        return inv
    gpus, host.driver = parse_query_csv(out)
    inv.gpus = gpus
    probe_interconnect(inv, read, listdir)
    mem = run(["nvidia-smi", "-q", "-d", "MEMORY"])
    bars = parse_bar1(mem or "")
    if len(bars) == len(gpus):
        for g, b in zip(gpus, bars):
            g.bar1_mib = b
    if len(gpus) > 1:
        topo = run(["nvidia-smi", "topo", "-p2p", "r"])
        inv.p2p = parse_p2p_matrix(topo or "")
        inv.p2p_ok = parse_p2p_pairs(topo or "")
        links, numa = parse_topo_matrix(run(["nvidia-smi", "topo", "-m"]) or "")
        if links:
            inv.links = links
        for g in gpus:
            if g.index in numa:
                g.numa_node = numa[g.index]
    else:
        inv.p2p = "unknown"
    for g in gpus:  # the kernel's view, for cards nvidia-smi gives no affinity (-1 = no NUMA information)
        if g.numa_node is None and g.bus_id:
            node = (read(f"/sys/bus/pci/devices/{sysfs_bus_id(g.bus_id)}/numa_node") or "").strip()
            if re.fullmatch(r"\d+", node):
                g.numa_node = int(node)
    host.ram_mib = host_ram_mib(read)
    try:
        st = os.statvfs("/dev/shm")
        host.shm_mib = st.f_bavail * st.f_frsize // MIB  # free space an unprivileged writer has, not the mount size
    except (OSError, AttributeError):
        host.shm_mib = None
    virt = run(["systemd-detect-virt"])
    if virt is not None:
        host.virt = classify_virt(virt)
    else:
        cpuinfo = read("/proc/cpuinfo") or ""
        if re.search(r"(?m)^flags\s*:.*\bhypervisor\b", cpuinfo):
            host.virt = "vm"
        elif "container=" in (read("/proc/1/environ") or ""):
            host.virt = "container"
    if "microsoft" in (read("/proc/sys/kernel/osrelease") or "").lower():
        # the WSL2 kernel wins: systemd-detect-virt reports the innermost layer (docker inside WSL2), and without it the
        # hypervisor flag would read as VM passthrough
        host.virt = "wsl"
    cvd = env.get("CUDA_VISIBLE_DEVICES", "").strip()
    if cvd and re.fullmatch(r"\d+(,\d+)*", cvd):
        want = [int(x) for x in cvd.split(",")]
        inv.notes.append(f"CUDA_VISIBLE_DEVICES={cvd}: only those GPUs are considered")
        inv.gpus = [g for g in inv.gpus if g.index in want]
    elif cvd:
        raise ValueError(
            f"CUDA_VISIBLE_DEVICES={cvd}: only comma-separated GPU indices are supported "
            "(UUID and MIG selections are not; nvidia-smi -L lists the indices)"
        )
    return inv


_SIM_RE = re.compile(
    r"^(?P<n>\d+)x(?P<card>[a-z0-9-]+)(?P<opts>(?::[a-z0-9=._]+)*)$", re.IGNORECASE
)
SIM_MAX_CARDS = 32
_SIM_HOST_OPTS = ("ram", "shm", "virt", "numa", "switch", "p2p")
_SIM_CARD_OPTS = ("bar1", "used")


def simulate(spec: str) -> Inventory:
    """Build an inventory from a hardware spec.

    ``<N>x<card>[:opts][+<N>x<card>[:opts]...]``: several ``+``-joined parts describe a mixed host (``8x4090+8x4090-48g``,
    ``2x3090+2x3090ti``), indices run in order over the parts, at most 32 cards in all. Part options: ``:p2p`` /
    ``:nop2p`` (the driver's peer access for those cards; the card's stock default otherwise), ``:bar1=MIB``,
    ``:used=MIB``. Host options (on any part, the last wins): ``:ram=GIB``, ``:shm=MIB``, ``:virt=vm|lxc|bare``,
    ``:numa=K`` (K NUMA nodes, the cards in K consecutive blocks), ``:switch=S`` (S consecutive cards per PCIe switch),
    ``:p2p=node`` / ``:p2p=switch`` (peer access only inside one node / one switch). With ``numa`` or ``switch`` the
    inventory carries ``nvidia-smi topo -m``-style links (PIX inside a switch, PHB inside a node, SYS across nodes)."""
    parts: list[tuple[int, str, dict[str, str]]] = []
    host_opts: dict[str, str] = {}
    for raw in spec.strip().split("+"):
        m = _SIM_RE.match(raw.strip())
        if not m:
            raise ValueError(
                f"bad --simulate spec {spec!r}; expected e.g. 8x3090, 4x4090:nop2p:ram=64 or 8x4090+8x4090-48g:numa=2"
            )
        n, card = int(m["n"]), m["card"].lower()
        if card not in KNOWN_CARDS:
            raise ValueError(
                f"unknown card {card!r}; known: " + ", ".join(sorted(KNOWN_CARDS))
            )
        if n < 1:
            raise ValueError(f"card count must be 1..{SIM_MAX_CARDS}")
        opts: dict[str, str] = {}
        for o in filter(None, m["opts"].split(":")):
            if o in ("p2p", "nop2p"):
                opts["p2p"] = "on" if o == "p2p" else "off"
            elif "=" in o:
                k, v = o.split("=", 1)
                if k in _SIM_HOST_OPTS:
                    if k == "p2p" and v not in ("node", "switch"):
                        raise ValueError(f"bad --simulate option {o!r}: p2p=node or p2p=switch")
                    host_opts[k] = v
                elif k in _SIM_CARD_OPTS:
                    opts[k] = v
                else:
                    raise ValueError(f"bad --simulate option {o!r}")
            else:
                raise ValueError(f"bad --simulate option {o!r}")
        parts.append((n, card, opts))
    total = sum(n for n, _, _ in parts)
    if not 1 <= total <= SIM_MAX_CARDS:
        raise ValueError(f"card count must be 1..{SIM_MAX_CARDS} (spec has {total})")
    try:
        numa = int(host_opts.get("numa", 1))
        switch = int(host_opts.get("switch", 0))
    except ValueError:
        raise ValueError("--simulate numa= and switch= take whole numbers") from None
    if numa < 1 or numa > total or switch < 0:
        raise ValueError(f"--simulate numa= must be 1..{total} and switch= a positive card count")
    gpus: list[Gpu] = []
    p2p_cap: list[bool] = []
    for n, card, opts in parts:
        name, arch, vram, p2p_stock = KNOWN_CARDS[card]
        p2p = opts.get("p2p", "on" if p2p_stock else "off")
        bar1 = int(opts.get("bar1", 32768 if p2p == "on" else 256))
        used = int(opts.get("used", 0))
        for _ in range(n):
            i = len(gpus)
            gpus.append(
                Gpu(
                    index=i,
                    name=name,
                    arch=arch,
                    vram_mib=vram,
                    bar1_mib=bar1,
                    mem_used_mib=used,
                    pcie_gen=4,
                    pcie_width=16,
                    persistence=True,
                    power_limit_w=220.0,
                    bus_id=f"00000000:{0x01 + i:02X}:00.0",
                )
            )
            p2p_cap.append(p2p == "on")
    per_node = -(-total // numa)  # consecutive blocks, the last one shorter
    node_of = [i // per_node for i in range(total)]
    switch_of = [i // switch if switch else None for i in range(total)]
    if numa > 1:
        for g in gpus:
            g.numa_node = node_of[g.index]
    links: list[list[Any]] | None = None
    if numa > 1 or switch:
        links = []
        for a in range(total):
            for b in range(a + 1, total):
                if switch and switch_of[a] == switch_of[b]:
                    kind = "PIX"
                elif node_of[a] == node_of[b]:
                    kind = "PHB"
                else:
                    kind = "SYS"
                links.append([a, b, kind])
    local = host_opts.get("p2p")
    if local == "switch" and not switch:
        raise ValueError("--simulate p2p=switch needs switch=<cards per switch>")

    def peers(a: int, b: int) -> bool:
        if not (p2p_cap[a] and p2p_cap[b]):
            return False
        if local == "node" and node_of[a] != node_of[b]:
            return False
        if local == "switch" and switch_of[a] != switch_of[b]:
            return False
        return True

    pairs = [[i, j] for i in range(total) for j in range(total) if i != j and peers(i, j)]
    all_pairs = total * (total - 1)
    p2p_state = "on" if pairs and len(pairs) == all_pairs else "off" if not pairs else "partial"
    host = Host(
        ram_mib=int(float(host_opts.get("ram", 96)) * 1024),
        shm_mib=int(host_opts.get("shm", 8192)),
        virt=host_opts.get("virt", "container").replace("lxc", "container"),
        driver="595.84",
        system="linux",
    )
    return Inventory(
        gpus=gpus, host=host, p2p=p2p_state if total > 1 else "unknown", source=f"simulate:{spec}",
        p2p_ok=pairs if total > 1 else None, links=links,
    )


def reference_host() -> Inventory:
    inv = simulate("4x3090")
    inv.source = "dry-run"
    inv.notes.append(
        "DRY RUN: no host access; this is the simulated reference host (4 x RTX 3090, P2P on, 96 GiB). "
        "Use --simulate or --inventory to describe other hardware."
    )
    return inv


def load_inventory(path: str) -> Inventory:
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    gpus = [Gpu(**g) for g in d["gpus"]]
    host = Host(**d.get("host", {}))
    return Inventory(
        gpus=gpus,
        host=host,
        p2p=d.get("p2p", "unknown"),
        source=f"inventory:{path}",
        notes=d.get("notes", []),
        p2p_ok=d.get("p2p_ok"),
        links=d.get("links"),
    )


# --------------------------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------------------------
def effective_p2p(inv: Inventory, forced: str, report: Report) -> str:
    """The interconnect mode the launcher will use: the driver's view, narrowed by BAR1, or the user's override."""
    n = len(inv.gpus)
    if n < 2:
        return "n/a"
    bars = [g.bar1_mib for g in inv.gpus if g.bar1_mib is not None]
    small = [
        g.index
        for g in inv.gpus
        if g.bar1_mib is not None and g.bar1_mib < BAR1_MIN_MIB
    ]
    driver = inv.p2p
    # Ipc keeps NCCL off its P2P transport, but the
    # PCIe-IPC all-reduce maps peer buffers, so it needs the same peer access as on.
    if forced in ("on", "off", "ipc"):
        if forced in ("on", "ipc") and driver in ("off", "partial"):
            report.add(
                "FAIL",
                "p2p",
                f"--p2p {forced} was requested but the driver reports P2P '{driver}' between the selected GPUs",
            )
        if forced in ("on", "ipc") and small:
            report.add(
                "FAIL",
                "bar1",
                f"--p2p {forced} was requested but GPU(s) {small} have BAR1 below 1 GiB: NCCL hangs at warmup on any "
                "P2P pair that includes them",
            )
        return "on" if forced == "ipc" else forced
    raw = inv.p2p_among([g.index for g in inv.gpus], driver_view=True)
    if small and raw != "off":
        report.add(
            "WARN",
            "bar1",
            f"GPU(s) {small} have BAR1 {min(bars)} MiB (< 1 GiB): raw cudaMemcpy P2P works but NCCL hangs at warmup on "
            f"any P2P pair that includes them, so the effective P2P is off (the driver's view: {raw}); flash a ReBAR "
            "vBIOS and enable Resizable BAR in the board BIOS to get 32 GB BAR1",
        )
        return "off"
    if small:
        report.add("INFO", "bar1", f"GPU(s) {small} have BAR1 {min(bars)} MiB (< 1 GiB); P2P is off among the selected cards anyway")
        return "off"
    if driver == "on":
        if not bars:
            report.add(
                "INFO",
                "bar1",
                "BAR1 size was not readable (nvidia-smi -q -d MEMORY); P2P is taken from the driver view only",
            )
        return "on"
    if driver == "partial":
        report.add(
            "WARN",
            "p2p",
            "P2P works between some GPU pairs only; treating P2P as off (the TP pairs must all have it)",
        )
        return "off"
    if driver == "off":
        return "off"
    report.add(
        "WARN",
        "p2p",
        "the driver's P2P state could not be read (nvidia-smi topo -p2p r); assuming off. "
        "Pass --p2p on if peer-to-peer is known to work on this host",
    )
    return "off"


# Families whose rows are qualified per interconnect (a P2P-on row and a P2P-off row): preflight matches P2P exactly.
STRICT_P2P_MODELS = ("flash-next", "ornith")
# merlin serve's checkpoint name and the placeholder directory a start line shows, per non-Flash-Next family
CHECKPOINT_NAMES = {"flash-next": "flash-next", "27b": "27b", "svl": "nemotron-super-vl", "ornith": "ornith1.5"}
CKPT_DIRS = {"27b": "Qwen3.8-27B-checkpoint", "svl": "Nemotron-3.5-Super-VL-120B-A12B-W4A16-Merlin", "ornith": "Ornith-1.5-122B-A10B-W4A16-Merlin"}


def _gib(mib: float) -> float:
    return mib / 1024.0


def explain_no_profile(
    model: str, n: int, arch: str | None, vclass: int | None, total_gib: float
) -> str:
    small = explain_small_card(model, n, arch, total_gib / max(n, 1))
    if small:
        return small
    if model == "ornith":
        return (
            "the Ornith 1.5 profiles cover 4 cards of 24 GB sm_86 (RTX 3090 class), with peer-to-peer (ornith-quad-3090) or "
            "without it (ornith-quad-3090-nop2p): TP2 x PP2 + EP with the byte-pinned KV pool (merlin list ornith1.5). "
            f"Found {n} card(s), arch {arch}, VRAM class {vclass}."
        )
    if model == "svl":
        return (
            "the Super VL profiles cover 4 cards of 24 GB sm_86 (RTX 3090 class, peer-to-peer) and, untested, 4 cards of "
            "48 GB sm_86, 24 / 32 / 48 GB sm_89 and 24 / 32 / 40 / 64 / 80 GB sm_80 (TP2 x PP2 + EP with the byte-pinned KV pool; "
            f"merlin list nemotron-super-vl). Found {n} card(s), arch {arch}, VRAM class {vclass}."
        )
    if model == "27b":
        return (
            "the 27B profiles cover 1, 2 or 4 cards of 24 GB sm_86 (RTX 3090 class) and, untested, of 48 GB sm_86, 24 / 32 / 48 GB "
            "sm_89, 24 / 32 / 40 / 64 / 80 GB sm_80 and 2 or 4 cards of 20 GB sm_86 / sm_89 (the series patches are gated to "
            f"dense Qwen3.5 on sm_86; the other arches take the stock paths; merlin list 27b). Found {n} card(s), arch {arch}, VRAM class {vclass}."
        )
    if n == 2 and vclass == 24:
        return (
            "2 x 24 GB cannot hold Flash-Next (W4A16 weights ~67.5 GiB across two cards plus the KV pool). A 2-card profile needs "
            "the host-RAM expert tier, which is design-only and not part of this release."
        )
    if n == 1 and (vclass or 0) < 80:
        return f"one {vclass or '?'} GB card cannot hold the ~67.5 GiB of weights; the smallest single-card row is 1 x 80 GB (sm_80)."
    if n == 2 and vclass == 40:
        return (
            "2 x 40 GB is too small: the weights leave ~3 GiB per card, which cannot hold one 262K session at bf16 "
            "(see the capacity table in docs/merlin/hardware-matrix.md)."
        )
    if n == 8 and vclass == 24 and arch == "sm_89":
        return (
            "8 x 24 GB sm_89 (for example 8 x L4) has no profile: the octo rows are sm_86 only and the FP8 KV path on "
            "sm_89 has not run on any Ada card. merlin serve takes four of the cards for fn-quad-24g-sm89 (or set GPUS=)."
        )
    if n == 3 and vclass == 24:
        return "3 x 24 GB has no layout: 48 layers do not split into three stages of ~16 that each fit beside a KV pool."
    if total_gib < 96 and n <= 4:
        return (
            f"{total_gib:.0f} GiB of VRAM in {n} card(s) is below the ~96 GiB total that Flash-Next needs at 262K "
            "(weights ~67.5 GiB + KV + per-card overhead)."
        )
    return (
        f"no profile for {n} x {vclass or '?'} GB {arch or 'unknown-arch'}. Profiles cover: 4/8 x 24 GB sm_86, "
        "2/4 x 48 GB, 4 x 24/32 GB sm_80/sm_89, A100 80 GB x1/2/4, CMP 170HX 32/40/64 GB rows, and 4 x 20 GB rows with the static "
        "cold-expert offload (see --list)."
    )


def _candidates(
    reg: dict[str, Any], model: str, n: int, arch: str, vclass: int | None, p2p: str, checkpoint: str | None = None
) -> list[dict[str, Any]]:
    out = []
    # automatic selection never crosses checkpoints: the requested checkpoint's rows only (the split merlin serve uses)
    for p in R.profiles_for(reg, checkpoint or model):
        m = p["match"]
        if arch not in m["arch"] or m["cards"] != n or m["vram_gib"] != vclass:
            continue
        if (
            model in STRICT_P2P_MODELS
            and n >= 2
            and m["p2p"] in ("on", "off")
            and m["p2p"] != p2p
        ):
            continue
        # a 27B / Super VL row qualified with peer-to-peer never runs without it (merlin serve refuses it too)
        if model not in STRICT_P2P_MODELS and n >= 2 and m["p2p"] == "on" and p2p != "on":
            continue
        out.append(p)
    return out


def _p2p_qualified_rows(reg: dict[str, Any], n: int, arch: str, vclass: int | None) -> list[dict[str, Any]]:
    return [p for p in reg["profiles"] if arch in p["match"]["arch"] and p["match"]["cards"] == n
            and p["match"]["vram_gib"] == vclass and p["match"]["p2p"] == "on"]


def no_p2p_hint(reg: dict[str, Any], model: str, arch: str, vclass: int | None, gpus: list[int], p2p: str,
                inv: Inventory | None = None) -> str:
    """Why a multi-card 27B / Super VL set without peer-to-peer has no row, and what runs instead: the single-card row
    is named only when it passes preflight on one of these cards (a 22.5 GiB card's reduced pin holds no 262K pool)."""
    name = "nemotron-super-vl" if model == "svl" else "27b"
    tp1 = [p for p in reg["profiles"] if arch in p["match"]["arch"] and p["match"]["cards"] == 1 and p["match"]["vram_gib"] == vclass]
    text = (f"the {len(gpus)}-card {'Super VL' if model == 'svl' else '27B'} rows of this class were qualified with peer-to-peer "
            f"and P2P is {p2p} on GPUs {','.join(map(str, gpus))}")
    twin = next((p for p in reg["profiles"] if arch in p["match"]["arch"] and p["match"]["cards"] == len(gpus)
                 and p["match"]["vram_gib"] == vclass and p["match"].get("p2p") == "off"), None)
    if twin is not None:  # the no-P2P row of the same layout (untested on these cards)
        return (f"the {len(gpus)}-card {'Super VL' if model == 'svl' else '27B'} row qualified with peer-to-peer needs it and P2P is "
                f"{p2p} on GPUs {','.join(map(str, gpus))}; {twin['id']} is the same layout without P2P: "
                f"GPUS={','.join(map(str, gpus))} merlin serve {name} <dir> --profile {twin['id']}")
    if not tp1:
        return text + "; no single-card Super VL row exists"
    if inv is not None:
        one = evaluate(inv, reg, model=model, want_profile=tp1[0]["id"], gpus=[gpus[0]])
        if one.exit_code != 0 or not one.profile_id:
            why = one.no_profile_reason or next((f.text for f in one.findings if f.level == "FAIL"), "")
            return text + f"; the single-card row {tp1[0]['id']} does not fit these cards either: {why}"
    return text + (f"; the single-card row {tp1[0]['id']} runs without it: GPUS={gpus[0]} merlin serve {name} <dir> --tp 1 (one server), "
                   f"or merlin serve {name} <dir> --tp 1 --instances {len(gpus)} ({len(gpus)} servers, the measured best without P2P)")


def ipc_refusal(reg: dict[str, Any], prof: dict[str, Any]) -> str:
    rows = [q["id"] for q in reg["profiles"] if "ipc" in (q.get("interconnect") or {}).get("modes", {})]
    return (f"P2P=ipc was requested but {prof['id']} has no ipc interconnect mode"
            + (f" (rows with one: {', '.join(rows[:6])}{', ...' if len(rows) > 6 else ''})" if rows else
               " (no row of this model has one)"))


def _mode_for(p: dict[str, Any], p2p: str) -> str | None:
    modes = p.get("interconnect", {}).get("modes")
    if not modes:
        return None
    if p2p in modes:
        return p2p
    return p["interconnect"]["default"]


def kv_bytes_for_vram(
    profile: dict[str, Any], actual_gib: float, nominal: int
) -> int | None:
    """Per-card KV pin lowered by the VRAM a card lacks against what the pin was sized for: the row's class, or
    ``serve.pin_vram_gib`` where the row keeps a smaller class's pin (the sm_89 32 / 48 GB 27B and Super VL rows keep
    the 24 GB pin and treat the rest as headroom, so an L40S at 45 GiB loses nothing)."""
    nominal = min(nominal, int(profile["serve"].get("pin_vram_gib") or nominal))
    delta = nominal - actual_gib
    if delta < 0.05:
        return None
    return max(0, int(profile["serve"]["kv_cache_memory"] - delta * GIB))


def adjusted_capacity(profile: dict[str, Any], pin_bytes: int) -> dict[str, Any]:
    """Pool and fits at a lowered per-card KV pin, scaled linearly from the profile's sizing [inference].

    The reference pool is calibrated at the profile pin. A card with less VRAM gets the pin the sizing gives it
    (pin = card VRAM - binding-stage weights - overhead) and the pool shrinks in proportion. Fits reuse the profile's
    own pool-per-session ratio, so they are conservative. This is sizing for the hardware, not a cut on a given machine.
    """
    cap, ref = profile["capacity"], profile["serve"]["kv_cache_memory"]
    pin_bytes = max(pin_bytes, 0)
    pool = cap["pool_tokens"] * pin_bytes // ref
    # Fits scale the profile's own (rounded-down) session counts, so they under-report; a pool that still holds the
    # tokens of one session is never shown as zero, and the refusal predicate is the pool itself, not a scaled count.
    fit262 = pool * cap["fit_262k"] // cap["pool_tokens"]
    fit131 = pool * cap["fit_131k"] // cap["pool_tokens"]
    out = {
        "estimate": True,
        "pin_bytes": pin_bytes,
        "pool_tokens": pool,
        "holds_one_262k": pool >= CTX_262K,
        "fit_262k": max(fit262, 1) if pool >= CTX_262K else fit262,
        "fit_131k": max(fit131, 1) if pool >= CTX_131K else fit131,
    }
    if cap.get("kv_blocks"):
        out["kv_blocks"] = cap["kv_blocks"] * pin_bytes // ref
    return out


def explain_small_card(model: str, n: int, arch: str | None, gib: float) -> str | None:
    """Why a card below the 24 GiB class is refused when no row of its class fits: a 20 GB card (RTX 3080 20GB, A4500,
    RTX 4000 Ada) on a card count without a 20 GiB row, or a smaller card (RTX 3080 Ti / 3080 12 GB, 16 GB cards)."""
    if gib >= 24 * 0.93:
        return None
    tail = " Nothing is squeezed (no pool cut, no lower quant, no offload) to make it fit."
    if R.vram_class(gib) == 20:
        head = f"{n} x {gib:.1f} GiB {arch or 'unknown-arch'} (the 20 GiB class) does not fit:"
        if model == "27b":
            return (head + " the 20 GiB class has 27B rows for 2 cards (TP2) and 4 cards (TP4) only, both untested; a single"
                    " 20 GB card has no TP1 row (no run at that size)." + tail)
        return (head + " the 20 GiB class has one Flash-Next layout, 4 cards TP2xPP2+EP (fn-quad-20g-sm86 / fn-quad-20g-sm89,"
                " and the same rows for Swift1.5 and Swift1.5-Huihui),"
                " untested, with the static cold-expert offload; no other card count of this class has a row." + tail)
    if gib > 20 * 1.07:
        head = f"{n} x {gib:.1f} GiB {arch or 'unknown-arch'} does not fit: between the 20 and 24 GiB VRAM classes, so no profile applies."
    else:
        head = f"{n} x {gib:.1f} GiB {arch or 'unknown-arch'} does not fit: below the smallest VRAM class (20 GiB), so no profile applies."
    if model == "27b":
        return head + " The 27B launchers pin KV bytes for 20 GiB cards and up." + tail
    ledger = (
        " Sizing (flash-next.json): the octo's binding stage needs ~10 GiB of weights + 2.75 GiB overhead = ~12.75 GiB"
        " before any KV, more than a 12 GB card holds."
    )
    return head + ledger + tail


def _alt_fits(reg: dict[str, Any], pid: str, actual_gib: float) -> bool:
    """False for an alternative whose pin, lowered to the reported VRAM, holds no 262K session."""
    try:
        p = R.get_profile(reg, pid)
    except R.ProfileError:
        return True
    adj = kv_bytes_for_vram(p, actual_gib, p["match"]["vram_gib"])
    return adj is None or (adj > 0 and adjusted_capacity(p, adj)["holds_one_262k"])


def _names_ok(pats: list[str], inv: Inventory) -> bool:
    """Every card's name matches one of ``pats`` (regexes)."""
    return all(any(re.search(p, g.name) for p in pats) for g in inv.gpus)


def _tested_name_ok(prof: dict[str, Any], inv: Inventory) -> bool:
    """A tested row is 'tested' only on the cards it was tested on (match.tested_gpu_names: regexes on the GPU name)."""
    pats = prof["match"].get("tested_gpu_names")
    if not pats:
        return True
    return all(any(re.search(p, g.name) for p in pats) for g in inv.gpus)


def evaluate(
    inv: Inventory,
    reg: dict[str, Any],
    model: str = "flash-next",
    want_profile: str | None = None,
    p2p: str = "auto",
    gpus: list[int] | None = None,
    prefer_latency: bool = False,
    kv_dtype: str | None = None,
    checkpoint: str | None = None,
) -> Report:
    # checkpoint: a derived Flash-Next checkpoint (swift1.5, swift1.5-huihui) limits the automatic pick to its own rows;
    # unset, the pick is over the model's own rows (the vanilla rows on the Flash-Next registry). A named
    # want_profile is taken as given.
    rep = Report(model=model, inventory=inv)
    try:
        reg = R.with_kv_dtype(reg, kv_dtype)
    except R.ProfileError as e:
        rep.add("FAIL", "kv-dtype", str(e))
        rep.exit_code = 2
        return rep
    subset = gpus is not None and len(gpus) < len(inv.gpus)
    if gpus is not None:
        if len(set(gpus)) != len(gpus):
            rep.add("FAIL", "inventory", f"--gpus {','.join(map(str, gpus))}: a GPU index is given twice")
            rep.exit_code = 2
            return rep
        by_index = {g.index: g for g in inv.gpus}
        keep = [by_index[i] for i in gpus if i in by_index]  # the caller's order: it is the rank / stage order
        if len(keep) != len(gpus):
            rep.add(
                "FAIL",
                "inventory",
                f"--gpus {','.join(map(str, gpus))}: GPU index not present "
                f"(found {','.join(str(g.index) for g in inv.gpus) or 'none'})",
            )
            rep.exit_code = 2
            return rep
        inv = Inventory(
            gpus=keep, host=inv.host, p2p=inv.p2p_among(gpus), source=inv.source, notes=inv.notes,
            p2p_ok=inv.p2p_ok, links=inv.links, kv_tier_gib=inv.kv_tier_gib, ple_table_gib=inv.ple_table_gib,
        )
        rep.inventory = inv
    for note in inv.notes:
        rep.add("INFO", "inventory", note)
    n = len(inv.gpus)
    if inv.host.system != "linux":
        rep.add(
            "FAIL",
            "os",
            f"host system is {inv.host.system}: the engine builds and runs on Linux x86-64 only "
            "(on Windows use WSL2 with the NVIDIA WSL driver; WSL2 itself is untested for this engine)",
        )
        rep.no_profile_reason = f"host system is {inv.host.system}, not Linux"
        rep.exit_code = 3
        return rep
    if n == 0:
        rep.no_profile_reason = "no NVIDIA GPU was found"
        rep.add("FAIL", "inventory", rep.no_profile_reason)
        rep.exit_code = 3
        return rep

    # homogeneity (refuse mixed-VRAM sets; a 3090 Ti next to 3090s is fine)
    archs = {g.arch for g in inv.gpus}
    classes = {R.vram_class(_gib(g.vram_mib)) for g in inv.gpus}
    if len(archs) > 1 or len(classes) > 1:
        rep.no_profile_reason = (
            "mixed GPUs: arch "
            + "/".join(sorted(map(str, archs)))
            + ", VRAM class "
            + "/".join(sorted(map(str, classes)))
            + ". Select a homogeneous set with --gpus (GPUS= for merlin serve, which picks the largest homogeneous subset itself)."
        )
        rep.add("FAIL", "mixed", rep.no_profile_reason)
        rep.exit_code = 3
        return rep
    arch = inv.gpus[0].arch
    vclass = next(iter(classes))
    total_gib = sum(_gib(g.vram_mib) for g in inv.gpus)
    if arch not in R.SUPPORTED_ARCHS:
        what = (
            "outside the Merlin 3.x targets (Ampere sm_80/sm_86 and Lovelace sm_89 only; sm_90+ is out of scope)"
            if arch and int(arch[3:]) >= 90
            else "unknown or older than Ampere"
        )
        rep.no_profile_reason = f"GPU arch {arch}: {what}"
        rep.add("FAIL", "arch", rep.no_profile_reason)
        rep.exit_code = 3
        return rep
    rep.add("PASS", "arch", f"{n} x {inv.gpus[0].name}, {arch}, VRAM class {vclass} GB")

    rep.effective_p2p = effective_p2p(inv, p2p, rep)

    if want_profile:
        try:
            prof = R.get_profile(reg, want_profile)
        except R.ProfileError as e:
            rep.add("FAIL", "profile", str(e))
            rep.exit_code = 2
            return rep
        cands = [prof]
        m = prof["match"]
        mism = []
        if arch not in m["arch"]:
            mism.append(f"arch {arch} not in {m['arch']}")
        if m["cards"] != n:
            mism.append(f"{n} cards, profile is for {m['cards']}")
        if m["vram_gib"] != vclass:
            mism.append(f"VRAM class {vclass}, profile is for {m['vram_gib']}")
        if (
            model in STRICT_P2P_MODELS
            and n >= 2
            and m["p2p"] in ("on", "off")
            and m["p2p"] != rep.effective_p2p
        ):
            mism.append(f"P2P is {rep.effective_p2p}, profile is for {m['p2p']}")
        if model not in STRICT_P2P_MODELS and n >= 2 and m["p2p"] == "on" and rep.effective_p2p != "on" and not mism:
            rep.no_profile_reason = no_p2p_hint(reg, model, arch, vclass, [g.index for g in inv.gpus], rep.effective_p2p, inv)
            rep.add("FAIL", "p2p", f"{want_profile} was qualified with peer-to-peer and P2P is {rep.effective_p2p} on these cards; "
                    f"refusing: {rep.no_profile_reason}")
            rep.exit_code = 3
            _host_checks(rep, inv, None)
            return rep
        if mism:
            rep.add(
                "FAIL",
                "profile",
                f"--profile {want_profile} does not match this hardware: "
                + "; ".join(mism),
            )
    else:
        cands = _candidates(reg, model, n, arch, vclass, rep.effective_p2p, checkpoint)
        if cands:
            def keyf(p: dict[str, Any]) -> tuple[Any, ...]:
                exact = 0 if p["match"]["p2p"] == rep.effective_p2p else 1
                lat = 0 if (prefer_latency and p.get("latency")) else 1
                tested = 0 if p["status"] == "tested" else 1  # tested rows first
                # a named alternative (auto_select false) runs by --profile
                named = 0 if R.auto_selectable(p) else 1
                blocked = 1 if R.blocked_reason(p) else 0  # a blocked row is picked only to say why it refuses
                return (blocked, named, lat, tested, p.get("rank", 9), exact)

            cands.sort(key=keyf)
    if not cands:
        rep.no_profile_reason = explain_no_profile(model, n, arch, vclass, total_gib)
        if model not in STRICT_P2P_MODELS and n >= 2 and rep.effective_p2p != "on" and _p2p_qualified_rows(reg, n, arch, vclass):
            rep.no_profile_reason = no_p2p_hint(reg, model, arch, vclass, [g.index for g in inv.gpus], rep.effective_p2p, inv)
        rep.add(
            "FAIL", "profile", "no profile fits this hardware: " + rep.no_profile_reason
        )
        rep.exit_code = 3
        _host_checks(rep, inv, None)
        return rep

    prof = cands[0]
    blocked = R.blocked_reason(prof)
    if blocked:  # the row matches the hardware but has no proven fit: refuse it instead of starting it to OOM
        rep.no_profile_reason = f"{prof['id']} matches these cards but is blocked: {blocked}"
        rep.add("FAIL", "blocked", rep.no_profile_reason)
        rep.exit_code = 3
        _host_checks(rep, inv, None)
        return rep

    def compatible(p: dict[str, Any]) -> bool:
        return not (
            model in STRICT_P2P_MODELS
            and n >= 2
            and p["match"]["p2p"] in ("on", "off")
            and p["match"]["p2p"] != rep.effective_p2p
        )

    rep.alternatives = [
        f"{c['id']} ({R.label_text(reg, c)}): {c['title']}" for c in cands[1:]
    ]
    if want_profile is None:
        listed = {prof["id"], *[c["id"] for c in cands]}
        rep.alternatives += [
            f"{p['id']} ({R.label_text(reg, p)}): {p['title']}"
            for p in reg["profiles"]
            if p["id"] not in listed
            and compatible(p)
            and p["match"]["cards"] == n
            and p["match"]["vram_gib"] == vclass
            and arch in p["match"]["arch"]
        ]
    rep.profile_id, rep.profile_title = prof["id"], prof["title"]
    status = prof["status"]
    if model == "flash-next":
        rep.p2p_mode = _mode_for(prof, rep.effective_p2p) if n >= 2 else None
        # P2P=ipc is opt-in, only on rows that declare it
        if p2p == "ipc" and n >= 2:
            if "ipc" in prof["interconnect"]["modes"]:
                rep.p2p_mode = "ipc"
            else:
                rep.add("FAIL", "p2p", ipc_refusal(reg, prof))
    else:
        rep.p2p_mode = rep.effective_p2p if n >= 2 else None
        if p2p == "ipc":
            rep.add("FAIL", "p2p", ipc_refusal(reg, prof))
    if status == "tested" and not _tested_name_ok(prof, inv):
        status = "untested"
        other = sorted({g.name for g in inv.gpus if not any(re.search(p, g.name) for p in prof["match"]["tested_gpu_names"])})
        rep.add(
            "WARN",
            "tested-hw",
            f"{prof['id']} was tested on {', '.join(prof['match']['tested_gpu_names'])} only; "
            f"{' + '.join(other)} resolves to the same row but is untested / community-validated",
        )
    if status in (R.LABEL_TESTED, R.LABEL_TESTING) and inv.host.virt == "wsl":
        status = R.LABEL_UNTESTED  # WSL2 is untested for the engine: no tested or testing row there
    run_on = prof["match"].get("testing_gpu_names")
    if status == R.LABEL_TESTING and run_on and not _names_ok(run_on, inv):
        status = R.LABEL_UNTESTED  # the cloud-hosted runs used other cards
        other = sorted({g.name for g in inv.gpus
                        if not any(re.search(p, g.name) for p in run_on)})
        rep.add(
            "WARN",
            "testing-hw",
            f"{prof['id']} ran on {', '.join(run_on)} only; {' + '.join(other)} "
            "resolves to the same row but is untested / community-validated",
        )
    # A tested/testing row's evidence covers
    # only the P2P modes in match.status_p2p_modes; the other mode of a two-mode row is
    # untested (e.g. the CMP 170HX rows were measured P2P off only).
    covered = prof["match"].get("status_p2p_modes")
    if (
        status in (R.LABEL_TESTED, R.LABEL_TESTING)
        and covered
        and rep.p2p_mode
        and rep.p2p_mode not in covered
    ):
        status = R.LABEL_UNTESTED
        rep.add(
            "WARN",
            "status-p2p",
            f"{prof['id']} was measured with P2P {' / '.join(covered)} only; this host resolves "
            f"P2P={rep.p2p_mode}, which runs the same row but is untested / community-validated",
        )
    rep.status = status
    rep.label = R.label_text(reg, prof, status)
    rep.layout = prof.get("layout") or {"tp": prof["serve"]["tensor_parallel"], "pp": 1}
    rep.serve = prof["serve"]
    rep.capacity = prof["capacity"]
    warning = R.sm80_fp8_warning(model, prof)
    if warning:
        rep.add("WARN", "kv-dtype", warning)

    if model == "flash-next":
        shared = (
            prof["id"] == R.REFERENCE_ID
            or prof.get("same_launcher_as") == R.REFERENCE_ID
        )
        name = R.launcher_name(prof["id"])
        ck = prof.get("checkpoint") or "flash-next"
        ck_dir = (
            reg["checkpoints"][prof["checkpoint"]]["hf_repo"].split("/")[-1]
            if prof.get("checkpoint") else "Qwen3.8-Flash-Next-W4A16-Merlin"
        )
        prefix = ""
        if rep.p2p_mode and (R.auto_p2p_switch(prof) or rep.p2p_mode == "ipc"):
            prefix = f"P2P={rep.p2p_mode} "
        # every launcher file is a shim for tools/merlin/launch.py; name the file that pins this profile
        if shared:
            rep.launcher = "models/flash-next/serve.sh"
        elif prof.get("recipe_launcher"):
            rep.launcher = prof["recipe_launcher"]
        elif prof["id"] in reg.get("launchers", []):
            rep.launcher = f"tools/merlin/launchers/{name}"
        else:
            rep.launcher = "tools/merlin/launch.py"
        if rep.launcher == "tools/merlin/launch.py":
            rep.start = f"{prefix}merlin serve {ck} /path/to/{ck_dir} --profile {prof['id']}"
        else:
            rep.start = f"{prefix}bash {rep.launcher} /path/to/{ck_dir}"
        rep.differs = [] if shared else R.diff_vs_reference(reg, prof)
        if (
            rep.p2p_mode
            and R.auto_p2p_switch(prof)
            and rep.p2p_mode not in (rep.effective_p2p, "ipc")
        ):
            rep.add(
                "INFO",
                "p2p",
                f"profile has no '{rep.effective_p2p}' mode; using '{rep.p2p_mode}'",
            )
        nominal = prof["match"]["vram_gib"]
        actual = _gib(min(g.vram_mib for g in inv.gpus))  # the smallest card binds the shared pin
        adj = kv_bytes_for_vram(prof, actual, nominal)
        if adj is not None:
            rep.add(
                "WARN",
                "kv-vram",
                f"cards report {actual:.1f} GiB, {nominal - actual:.1f} GiB under the {nominal} GiB class: "
                f"export KV_BYTES={adj} (profile default {prof['serve']['kv_cache_memory']})",
            )
            rep.adjusted = adjusted_capacity(prof, adj)
            if adj <= 0 or not rep.adjusted["holds_one_262k"]:
                rep.profile_id = rep.profile_title = rep.label = rep.status = None
                rep.start = rep.launcher = None
                rep.no_profile_reason = (
                    f"{prof['id']} does not fit: at {actual:.1f} GiB per card the sizing leaves a KV pin of "
                    f"{max(adj, 0) / GIB:.2f} GiB, a ~{rep.adjusted['pool_tokens']:,}-token pool that holds no 262K "
                    "session. The pool is not cut further and no quant is lowered to make it fit."
                )
                rep.add("FAIL", "profile", "no profile fits this hardware: " + rep.no_profile_reason)
                rep.exit_code = 3
                return rep
            a = rep.adjusted
            rep.add(
                "INFO",
                "kv-vram",
                f"at that pin the pool is ~{a['pool_tokens']:,} tokens = ~{a['fit_262k']} x 262K or "
                f"~{a['fit_131k']} x 131K (ESTIMATE: scaled linearly from the profile sizing, not measured)",
            )
            ck = prof.get("checkpoint", "flash-next")
            ck_dir = (
                reg["checkpoints"][prof["checkpoint"]]["hf_repo"].split("/")[-1]
                if prof.get("checkpoint")
                else "Qwen3.8-Flash-Next-W4A16-Merlin"
            )
            rep.start = (
                f"merlin serve {ck} /path/to/{ck_dir} --profile {prof['id']}   "
                f"(sets KV_BYTES={adj}; the plain launcher needs that export too)"
            )
            rep.alternatives = [
                x
                for x in rep.alternatives
                if _alt_fits(reg, x.split(" ", 1)[0], actual)
            ]
    else:
        have = _gib(min(g.vram_mib for g in inv.gpus))
        nominal = prof["match"]["vram_gib"]
        adj = kv_bytes_for_vram(prof, have, nominal)
        if adj is not None:
            # a card under its class (A10 / A10G / L4 report 22.5 GiB): the pin shrinks by the shortfall, rounded down to whole
            # blocks where the row records them; quality is unchanged, only capacity. Refused only below one 262K session.
            blocks = prof["capacity"].get("kv_blocks")
            if blocks:
                per_block = prof["serve"]["kv_cache_memory"] // blocks
                adj = max(0, adj // per_block * per_block)
            rep.adjusted = adjusted_capacity(prof, adj)
            if adj < KV_MIN_BYTES or not rep.adjusted["holds_one_262k"]:
                rep.profile_id = rep.profile_title = rep.label = rep.status = None
                why = (f"a ~{rep.adjusted['pool_tokens']:,}-token pool that holds no 262K session"
                       if not rep.adjusted["holds_one_262k"] else "under the 1 GiB the launcher accepts (KV_BYTES >= 1 GiB)")
                rep.no_profile_reason = (
                    f"{prof['id']} does not fit: these cards report {have:.1f} GiB ({nominal - have:.1f} GiB under the "
                    f"{nominal} GiB class); the sizing leaves a KV pin of {max(adj, 0) / GIB:.2f} GiB, "
                    f"{why}. The quant is not lowered to make it fit."
                )
                rep.add("FAIL", "profile", "no profile fits this hardware: " + rep.no_profile_reason)
                rep.exit_code = 3
                return rep
            a = rep.adjusted
            rep.add(
                "WARN",
                "kv-vram",
                f"reduced pool: cards report {have:.1f} GiB, {nominal - have:.1f} GiB under the {nominal} GiB class, so the KV pin is "
                f"KV_BYTES={adj} (profile default {prof['serve']['kv_cache_memory']}): ~{a['pool_tokens']:,} tokens = "
                f"~{a['fit_262k']} x 262K or ~{a['fit_131k']} x 131K resident (ESTIMATE from the row's sizing; quality unchanged, capacity only)",
            )
        launcher_path = prof["launcher"].get("path")  # set only on the rows a shim pins exactly (the 3090 rows)
        ckpt = CKPT_DIRS[model]
        if launcher_path:
            rep.launcher = launcher_path
            rep.start = f"MODEL=/path/to/{ckpt} bash {launcher_path}"
        else:
            rep.launcher = "tools/merlin/launch.py"
            rep.start = f"merlin serve {CHECKPOINT_NAMES[model]} /path/to/{ckpt} --profile {prof['id']}"
        if n >= 2 and prof["match"]["p2p"] == "any":
            rep.start = f"P2P={rep.effective_p2p} " + rep.start
        prov = prof["serve"].get("provisional")
        if prov:
            basis = prof["serve"].get("provisional_basis") or "not measured on this engine yet (an estimate)"
            text = f"{prof['id']}: {', '.join(prov)} {basis}; KV_BYTES overrides the pin"
            if prof["serve"].get("image_max_pixels"):
                text += (f". The {prof['serve']['image_max_pixels']:,}-pixel image cap is the shipped default and is not "
                         "lowered here; check the free memory beside a full-size image on this card")
            rep.add("WARN", "provisional", text)
    if prof.get("fp8_opt_in") and rep.start:
        rep.start = "KV_DTYPE=fp8 " + rep.start
    _host_checks(rep, inv, prof)
    _label_finding(rep, prof)
    if subset and rep.start:  # the checked set is part of a bigger host: the start line must name it
        rep.start = f"GPUS={','.join(str(g.index) for g in inv.gpus)} {rep.start}"
    rep.exit_code = 1 if any(f.level == "FAIL" for f in rep.findings) else 0
    return rep


def _label_finding(rep: Report, prof: dict[str, Any]) -> None:
    if rep.status == "tested":
        rep.add("PASS", "label", f"{prof['id']}: {rep.label}")
    elif rep.status == "parked":
        rep.add(
            "WARN",
            "label",
            f"{prof['id']}: {rep.label.upper()}. This profile is parked and has no launcher in this release",
        )
    else:
        what = (prof.get("evidence") or ["none"])[0]
        if len(what) > 150:  # cut between words, never inside one or between a number and its unit
            cut = what[:147].rsplit(None, 1)[0]
            while cut[-1:].isdigit() and " " in cut:
                cut = cut.rsplit(None, 1)[0]
            what = cut.rstrip(" ,;:(") + " ..."
        rep.add(
            "WARN",
            "label",
            f"{prof['id']}: {rep.label.upper()}. Evidence so far: {what} (full list in docs/merlin/hardware-matrix.md)",
        )


def _host_checks(rep: Report, inv: Inventory, prof: dict[str, Any] | None) -> None:
    h = inv.host
    n = len(inv.gpus)
    if h.driver:
        try:
            major = int(h.driver.split(".")[0])
        except ValueError:
            major = None
        if major is not None and major < DRIVER_MIN_MAJOR:
            rep.add(
                "FAIL",
                "driver",
                f"driver {h.driver}: CUDA 13.0 needs the {DRIVER_MIN_MAJOR} series or later",
            )
        elif major is not None:
            rep.add("PASS", "driver", f"driver {h.driver}")
    else:
        rep.add("INFO", "driver", "driver version unknown")
    tier = inv.kv_tier_gib
    if tier > 0:
        ranks = int(prof["match"]["cards"]) if prof is not None else max(n, 1)
        minimum = min_restore_unit_gib_per_rank(ranks, rep.model, prof)
        if tier < minimum:
            rep.add(
                "FAIL",
                "kv-tier",
                f"{KV_HOST_TIER_ENV}={tier} GiB per rank cannot hold one 262K restore unit "
                f"for this {ranks}-rank layout; needs at least {minimum} GiB per rank.",
            )
    if rep.model == "flash-next":
        tier = inv.kv_tier_gib
        floor = ram_floor_gib(n, tier, inv.ple_table_gib)
        warn = RAM_WARN_GIB + max(0, tier) * n  # the qualified allocation is for the tier off: the tier comes on top
        if not inv.ple_table_gib:  # --ple-store disk: warn below the floor + the page cache allowance
            warn = floor + PS.CACHE_WARN_GIB
        t = ram_tier_text(n, tier)
        if h.ram_mib is None:
            rep.add(
                "WARN",
                "ram",
                f"host RAM unknown; Flash-Next needs {'~' if inv.ple_table_gib else 'a computed ~'}{floor:.0f} GiB resident ("
                + (f"~{inv.ple_table_gib:.1f} GiB pinned PLE table" if inv.ple_table_gib else "PLE table on disk") + f"{t}) "
                + (f"and {warn:.0f} GiB is qualified" if inv.ple_table_gib else f"and {warn:.0f} GiB is the computed floor plus page cache"),
            )
        else:
            ram = _gib(h.ram_mib)
            if ram < floor:
                rep.add(
                    "FAIL",
                    "ram",
                    f"host RAM {ram:.0f} GiB is below the ~{floor:.0f} GiB resident floor{t}: the model load is "
                    "expected to be OOM-killed",
                )
            elif ram < warn:
                rep.add(
                    "WARN",
                    "ram",
                    f"host RAM {ram:.0f} GiB is below the qualified {warn:.0f} GiB (floor ~{floor:.0f} GiB{t}; "
                    "the boot peak is unmeasured)"
                    if inv.ple_table_gib
                    else f"host RAM {ram:.0f} GiB leaves under {PS.CACHE_WARN_GIB} GiB of page cache for the PLE table above "
                    f"the ~{floor:.0f} GiB floor{t}; decode and prefill slow down",
                )
            else:
                rep.add(
                    "PASS",
                    "ram",
                    f"host RAM {ram:.0f} GiB (>= {warn:.0f} GiB {'qualified' if inv.ple_table_gib else 'floor + page cache'}{t})",
                )
    elif rep.model in ("27b", "ornith") and inv.kv_tier_gib > 0:
        # The 27B and Ornith have no measured resident floor; the KV host tier's arenas (GiB per rank, pinned)
        # come on top of it. Ranks = TP x PP (Ornith: TP2 x PP2 = 4; the 27B is TP only)
        srv = (prof or {}).get("serve", {})
        ranks = int(srv.get("tensor_parallel") or n) * int(srv.get("pipeline_parallel") or 1)
        need = inv.kv_tier_gib * ranks
        own = "Ornith's" if rep.model == "ornith" else "the 27B's"
        what = (f"the KV host tier needs {inv.kv_tier_gib} GiB x {ranks} ranks = {need} GiB of pinned host RAM on top of "
                f"{own} own")
        if h.ram_mib is None:
            rep.add("WARN", "ram", f"host RAM unknown; {what}")
        elif _gib(h.ram_mib) < need:
            rep.add("FAIL", "ram", f"host RAM {_gib(h.ram_mib):.0f} GiB is too small: {what}")
        else:
            rep.add("INFO", "ram", f"host RAM {_gib(h.ram_mib):.0f} GiB; {what}")
    if h.shm_mib is not None:
        if h.shm_mib < SHM_MIN_MIB:
            rep.add(
                "WARN",
                "shm",
                f"/dev/shm is {h.shm_mib} MiB; the PLE transport needs >= 1 GiB (docker --shm-size=8g)",
            )
        else:
            rep.add("PASS", "shm", f"/dev/shm {h.shm_mib} MiB")
    if h.virt == "vm":
        rep.add(
            "WARN",
            "virt",
            "the GPUs sit behind a virtual machine (passthrough): serving was measured on bare metal and in an "
            "LXC, and VM passthrough is measurably slower",
        )
    elif h.virt == "wsl":
        rep.add(
            "WARN",
            "virt",
            "WSL2: untested for the Merlin engine; the WSL2 driver's free-memory reading is reported unreliable by the "
            "community, and vLLM's profiling still reads it. Run a pool-only boot first. Host-resident embeddings need "
            "pinned memory: VLLM_WSL2_ENABLE_PIN_MEMORY=1 on a current WSL2 kernel (the launcher sets it, except for "
            "Flash-Next). Peer access is usually absent there (an unreadable P2P probe counts as off), and nvidia-smi "
            "cannot change persistence mode under WSL. See docs/merlin/hardware-matrix.md#wsl2",
        )
    elif h.virt in ("bare", "container"):
        rep.add(
            "PASS", "virt", "bare metal" if h.virt == "bare" else "container / LXC host"
        )
    busy = [g.index for g in inv.gpus if (g.mem_used_mib or 0) > BUSY_MIB]
    if busy:
        rep.add(
            "WARN",
            "busy",
            f"GPU(s) {busy} already hold more than 1 GiB: the KV pool is pinned in bytes and leaves ~1 GiB per "
            "card, so another process (a display server, a running server) can stop the boot",
        )
    off_pm = [g.index for g in inv.gpus if g.persistence is False]
    if off_pm and inv.host.virt != "wsl":  # WSL: nvidia-smi cannot change persistence; the WSL WARN covers it
        rep.add(
            "WARN",
            "persistence",
            f"persistence mode is off on GPU(s) {off_pm}: `nvidia-smi -pl` silently reverts when the last "
            "client exits. Run `nvidia-smi -pm 1` (or nvidia-persistenced) before setting a power cap",
        )
    narrow = [
        g.index for g in inv.gpus if g.pcie_width is not None and g.pcie_width <= 4
    ]
    x8 = [
        g.index for g in inv.gpus if g.pcie_width is not None and 4 < g.pcie_width < 16
    ]
    if narrow:
        rep.add(
            "WARN",
            "pcie",
            f"PCIe link width x{inv.gpus[0].pcie_width} on GPU(s) {narrow}: the PLE pull and PP traffic share this "
            "link; decode cost is unmeasured (170HX Gen2 x16 width is itself unresolved)",
        )
    elif x8:
        rep.add(
            "INFO",
            "pcie",
            f"PCIe link width below x16 on GPU(s) {x8}: expect lower prefill (the no-P2P test host ran x8)",
        )
    low_gen = [g.index for g in inv.gpus if g.pcie_gen is not None and g.pcie_gen < 3]
    if low_gen:
        rep.add(
            "INFO",
            "pcie",
            f"PCIe generation {inv.gpus[0].pcie_gen} on GPU(s) {low_gen}: ~6 GB/s class link",
        )
    if rep.p2p_mode in ("on", "ipc"):
        bars = [g.bar1_mib for g in inv.gpus if g.bar1_mib is not None]
        if bars:
            rep.add(
                "PASS",
                "bar1",
                f"BAR1 {min(bars)} MiB or more on every GPU (>= 1 GiB needed for NCCL P2P)",
            )


# --------------------------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------------------------
def _fmt_tokens(n: int) -> str:
    return f"{n:,}"


def render_text(rep: Report, reg: dict[str, Any]) -> str:
    inv, out = rep.inventory, []
    out.append(f"Merlin preflight ({rep.model})    [source: {inv.source}]")
    out.append("")
    out.append("Hardware")
    if inv.gpus:
        g = inv.gpus[0]
        homo = len({(x.name, x.vram_mib) for x in inv.gpus}) == 1
        if homo:
            bar = f", BAR1 {g.bar1_mib} MiB" if g.bar1_mib is not None else ""
            out.append(
                f"  {len(inv.gpus)} x {g.name}  {g.arch or 'arch unknown'}  {_gib(g.vram_mib):.1f} GiB{bar}"
            )
        else:
            for x in inv.gpus:
                out.append(
                    f"  GPU{x.index}: {x.name}  {x.arch or 'arch unknown'}  {_gib(x.vram_mib):.1f} GiB"
                )
    else:
        out.append("  no NVIDIA GPU found")
    if len(inv.gpus) > 1:
        out.append(f"  P2P: driver view '{inv.p2p}', effective '{rep.effective_p2p}'"
                   + (f", mode '{rep.p2p_mode}'" if rep.p2p_mode == "ipc" else ""))
    out.extend(interconnect_text(inv))
    h = inv.host
    ram = f"{_gib(h.ram_mib):.0f} GiB" if h.ram_mib else "unknown"
    shm = f"{h.shm_mib} MiB" if h.shm_mib is not None else "unknown"
    out.append(
        f"  host RAM {ram}, /dev/shm {shm}, virtualisation {h.virt or 'unknown'}, driver {h.driver or 'unknown'}"
    )
    out.extend(U.text_lines(rep.gpu_unlock))
    out.append("")
    out.append("Decision")
    if rep.profile_id:
        out.append(f"  profile   {rep.profile_id}: {rep.profile_title}")
        out.append(f"  STATUS    {rep.label.upper()}")
        if rep.model == "flash-next":
            lay, srv, cap = rep.layout, rep.serve, rep.capacity
            layout = f"TP{lay['tp']} x PP{lay['pp']}" + (
                " + EP" if lay.get("ep") else ""
            )
            if lay.get("partition"):
                layout += f", layer partition {','.join(map(str, lay['partition']))}"
            out.append(
                f"  layout    {layout}, max-num-seqs {srv['max_num_seqs']}, KV {srv['kv_cache_dtype']} "
                f"{srv['kv_cache_memory'] / GIB:.2f} GiB per card"
            )
        else:
            srv, cap = rep.serve, rep.capacity
            lay = rep.layout
            layout = f"TP{srv['tensor_parallel']}" + (
                f" x PP{lay['pp']}" + (" + EP" if lay.get("ep") else "")
                if lay.get("pp", 1) > 1
                else ""
            )
            if lay.get("partition"):
                layout += f", layer partition {','.join(map(str, lay['partition']))}"
            out.append(
                f"  layout    {layout}, max-num-seqs {srv['max_num_seqs']}, KV {srv['kv_cache_dtype']} "
                f"{srv['kv_cache_memory'] / GIB:.2f} GiB per card, MTP K={srv['spec_k']}"
            )
        basis = (
            "ESTIMATE, not measured on this hardware"
            if cap.get("estimate")
            else "measured on the reference host"
            if rep.status == "tested"
            else "measured on cloud-hosted cards; kernel parity not yet qualified"
            if rep.status == R.LABEL_TESTING
            else "measured on the reference host with RTX 3090s; this card is untested"
        )
        out.append(
            f"  capacity  {_fmt_tokens(cap['pool_tokens'])}-token pool = {cap['fit_262k']} x 262K or {cap['fit_131k']} x 131K "
            f"resident sessions ({basis})"
        )
        if rep.adjusted:
            a = rep.adjusted
            out.append(
                f"  at the reported VRAM: KV {a['pin_bytes'] / GIB:.2f} GiB per card, ~{_fmt_tokens(a['pool_tokens'])}-token "
                f"pool = ~{a['fit_262k']} x 262K or ~{a['fit_131k']} x 131K (ESTIMATE; the line above is the profile's "
                "nominal-class figure)"
            )
        out.append(f"  start     {rep.start}")
        if rep.differs:
            out.append("  differs from the tested quad profile:")
            out.extend(f"    - {d}" for d in rep.differs)
        if rep.alternatives:
            out.append("  alternatives (same hardware):")
            out.extend(f"    - {a}" for a in rep.alternatives)
    else:
        out.append(f"  NO PROFILE: {rep.no_profile_reason}")
    out.append("")
    out.append("Checks")
    for f in rep.findings:
        out.append(f"  [{f.level}] {f.text}")
    out.append("")
    if rep.profile_id and rep.status == "untested":
        out.append(
            "This hardware row is UNTESTED / COMMUNITY-VALIDATED: it ships so that it can be tried, and results (pool, decode at"
        )
        out.append(
            "4K/131K/262K, needles) are the validation. See docs/merlin/hardware-matrix.md."
        )
    if rep.profile_id and rep.status == R.LABEL_TESTING:
        out.append(
            "This hardware row is 🟡 TESTING: full-model serving runs passed on cloud-hosted"
        )
        out.append(
            "hardware; kernel parity qualification is still to run, and merlin serve"
        )
        out.append(
            "picks it like an untested row. See docs/merlin/hardware-matrix.md."
        )
    out.append(f"exit {rep.exit_code}")
    return "\n".join(out) + "\n"


def report_dict(rep: Report) -> dict[str, Any]:
    inv = rep.inventory
    return {
        "schema": SCHEMA,
        "model": rep.model,
        "source": inv.source,
        "exit_code": rep.exit_code,
        "hardware": {
            "gpus": [asdict(g) for g in inv.gpus],
            "host": asdict(inv.host),
            "p2p_driver": inv.p2p,
            "p2p_effective": rep.effective_p2p,
        },
        "profile": rep.profile_id,
        "title": rep.profile_title,
        "status": rep.status,
        "label": rep.label,
        "p2p_mode": rep.p2p_mode,
        "layout": rep.layout,
        "serve": rep.serve,
        "capacity": rep.capacity,
        "capacity_at_reported_vram": rep.adjusted,
        "launcher": rep.launcher,
        "start": rep.start,
        "differs": rep.differs,
        "alternatives": rep.alternatives,
        "no_profile_reason": rep.no_profile_reason,
        "findings": [asdict(f) for f in rep.findings],
        "gpu_unlock": rep.gpu_unlock,
    }


BLOCKED_TAG = "; BLOCKED, refused until a fit is proven"


def render_list(reg: dict[str, Any]) -> str:
    rows = []
    for p in reg["profiles"]:
        m = p["match"]
        lay = p.get("layout")
        shape = (
            f"TP{lay['tp']}xPP{lay['pp']}"
            if lay
            else f"TP{p['serve']['tensor_parallel']}"
        )
        rows.append(
            f"{p['id']:28s} {m['cards']} x {m['vram_gib']:>2} GB  {'/'.join(m['arch']):12s} p2p={m['p2p']:3s} "
            f"{shape:8s} {R.label_text(reg, p)}{BLOCKED_TAG if R.blocked_reason(p) else ''}"
        )
    return "\n".join(rows) + "\n"


def render_matrix(reg: dict[str, Any]) -> str:
    """The Markdown capacity table that docs/merlin/hardware-matrix.md embeds."""
    lines = [
        "| Profile | Cards | Arch | P2P | Layout | seqs | KV | Pool (tokens) | 262K / 131K | Status |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    rows = []
    for p in reg["profiles"]:
        rows.append((p["id"], p))
        for dtype in p["serve"].get("kv_alt", {}):
            rows.append((f"{p['id']} kv-{R.KV_TAG[dtype]}", R.variant_profile(p, dtype)))
    for rid, p in rows:
        m, lay, srv, cap = p["match"], p["layout"], p["serve"], p["capacity"]
        shape = f"TP{lay['tp']}xPP{lay['pp']}" + ("+EP" if lay["ep"] else "")
        est = "" if not cap["estimate"] else "~"
        kv = "fp8" if srv["kv_cache_dtype"] == "fp8_e4m3" else "bf16"
        lines.append(
            f"| `{rid}` | {m['cards']} x {m['vram_gib']} GB | {'/'.join(m['arch'])} | {m['p2p']} | "
            f"{shape} | {srv['max_num_seqs']} | {kv} | {est}{cap['pool_tokens']:,} | {est}{cap['fit_262k']} / {cap['fit_131k']} | "
            f"{R.label_text(reg, p)}{BLOCKED_TAG if R.blocked_reason(p) else ''} |"
        )
    return "\n".join(lines) + "\n"


def write_launchers(reg: dict[str, Any], directory: Path) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, text in R.render_all(reg).items():
        path = directory / name
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        paths.append(path)
    return paths


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="preflight.py",
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Exit codes: 0 ok; 1 a check failed (or warned with --strict); 2 usage/input error; "
        "3 no profile fits this hardware.",
    )
    ap.add_argument("--model", choices=("flash-next", "27b", "svl", "ornith"), default="flash-next")
    src = ap.add_mutually_exclusive_group()
    src.add_argument(
        "--dry-run",
        action="store_true",
        help="touch nothing on this host (no nvidia-smi, no /proc); use --simulate/--inventory, or the simulated reference host",
    )
    src.add_argument(
        "--simulate",
        metavar="SPEC",
        help="describe hardware, e.g. 8x3090, 4x4090:nop2p:ram=64 or 8x4090+8x4090-48g:numa=2 (implies no host access)",
    )
    src.add_argument(
        "--inventory",
        metavar="FILE",
        help="read a captured inventory JSON instead of probing",
    )
    ap.add_argument(
        "--gpus",
        help="comma-separated GPU indices to consider (default: all, or CUDA_VISIBLE_DEVICES indices)",
    )
    ap.add_argument(
        "--p2p",
        choices=("auto", "on", "off", "ipc"),
        default="auto",
        help="override the detected peer-to-peer state (ipc: NCCL without P2P transport + PCIe-IPC all-reduce)",
    )
    ap.add_argument(
        "--profile", help="force a profile id (checked against the hardware)"
    )
    ap.add_argument(
        "--latency",
        action="store_true",
        help="prefer the single-stream-latency layout where one exists",
    )
    ap.add_argument(
        "--kv",
        choices=("fp8", "bf16"),
        help="with --emit launcher: render the fp8 or bf16 KV variant of the profile (a profile lists the "
        "variants it sizes; the default KV dtype is its plain launcher)",
    )
    ap.add_argument("--strict", action="store_true", help="exit 1 on warnings too")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--list", action="store_true", help="list the profiles and exit")
    ap.add_argument(
        "--emit",
        choices=("launcher", "matrix", "inventory"),
        help="print a generated launcher (needs --profile), the Markdown matrix table, or the probed inventory as JSON",
    )
    ap.add_argument(
        "--write-launchers",
        nargs="?",
        const="",
        metavar="DIR",
        help="regenerate the checked-in launchers (default tools/merlin/launchers)",
    )
    return ap


def serve_choice(reg: dict[str, Any], model: str, inv: Inventory, profile: str | None) -> list[int] | None:
    """The cards ``merlin serve`` would pick on this inventory, so doctor reports the same layout; None when serve
    finds nothing (the full-inventory report then explains why)."""
    import serve_cli as S  # imported here: serve_cli imports this module

    name = CHECKPOINT_NAMES.get(model, "flash-next")
    if profile and model == "flash-next":  # a Swift / Huihui row belongs to its checkpoint's name
        try:
            name = R.get_profile(reg, profile).get("checkpoint") or name
        except R.ProfileError:
            return None
    try:
        if S._gpu_selection(dict(os.environ)) is not None:  # serve takes exactly the GPUS / CUDA_VISIBLE_DEVICES set
            return None
        return S.choose(reg, name, inv, profile, None).cards
    except (S.CliError, R.ProfileError):
        return None


def main(
    argv: list[str] | None = None, run: Runner = default_runner, stdout: Any = None,
    read: Callable[[str], str | None] = _read,
) -> int:
    out = stdout or sys.stdout
    args = build_parser().parse_args(argv)
    try:
        reg = R.load_registry(args.model)
    except (R.ProfileError, OSError, json.JSONDecodeError) as e:
        print(f"preflight: cannot load the profile registry: {e}", file=sys.stderr)
        return 2
    try:
        serving_reg = R.with_kv_dtype(reg, os.environ.get("KV_DTYPE"))
    except R.ProfileError as e:
        print(f"preflight: {e}", file=sys.stderr)
        return 2
    if args.list:
        out.write(render_list(reg))
        return 0
    if args.write_launchers is not None:
        if args.model != "flash-next":
            print(
                "preflight: launchers are generated for flash-next only",
                file=sys.stderr,
            )
            return 2
        d = Path(args.write_launchers) if args.write_launchers else R.LAUNCHERS_DIR
        for p in write_launchers(reg, d):
            out.write(f"wrote {p}\n")
        return 0
    if args.emit == "matrix":
        out.write(render_matrix(reg))
        return 0
    if args.emit == "launcher":
        if not args.profile or args.model != "flash-next":
            print(
                "preflight: --emit launcher needs --profile ID (flash-next)",
                file=sys.stderr,
            )
            return 2
        try:
            prof = R.get_profile(reg, args.profile)
            if args.kv:
                dtype = {v: k for k, v in R.KV_TAG.items()}[args.kv]
                prof = R.variant_profile(prof, dtype)
            out.write(R.render_launcher(reg, prof))
        except R.ProfileError as e:
            print(f"preflight: {e}", file=sys.stderr)
            return 2
        return 0
    reg = serving_reg
    try:
        gpus = [int(x) for x in args.gpus.split(",")] if args.gpus else None
    except ValueError:
        print(
            "preflight: --gpus must be a comma-separated list of integers",
            file=sys.stderr,
        )
        return 2
    if gpus is None:  # GPUS / CUDA_VISIBLE_DEVICES win for merlin serve, so doctor reports that set
        import serve_cli as S

        try:
            gpus = S._gpu_selection(dict(os.environ))
        except S.CliError as e:
            print(f"preflight: {e}", file=sys.stderr)
            return 2
    try:
        if args.simulate:
            inv = simulate(args.simulate)
        elif args.inventory:
            inv = load_inventory(args.inventory)
        elif args.dry_run:
            inv = reference_host()
        else:
            inv = probe(run, read)
    except (ValueError, OSError, KeyError, TypeError, json.JSONDecodeError) as e:
        print(f"preflight: {e}", file=sys.stderr)
        return 2
    inv.kv_tier_gib = kv_host_tier_gib()  # the RAM gates count the KV host tier merlin serve would start with
    inv.ple_table_gib = saved_ple_table_gib(args.model, _saved_setup(Path(__file__).resolve().parents[2]))
    if args.emit == "inventory":
        out.write(
            json.dumps(
                {
                    "gpus": [asdict(g) for g in inv.gpus],
                    "host": asdict(inv.host),
                    "p2p": inv.p2p,
                    "p2p_ok": inv.p2p_ok,
                    "links": inv.links,
                    "notes": inv.notes,
                },
                indent=1,
            )
            + "\n"
        )
        return 0
    picked = None
    if gpus is None and not args.latency and args.p2p == "auto":
        picked = serve_choice(reg, args.model, inv, args.profile)
    rep = evaluate(
        inv,
        reg,
        model=args.model,
        want_profile=args.profile,
        p2p=args.p2p,
        gpus=gpus if picked is None else picked,
        prefer_latency=args.latency,
    )
    if picked is not None and len(picked) < len(inv.gpus):
        rep.add("INFO", "selection", f"merlin serve would use GPUs {','.join(map(str, picked))} of "
                f"{len(inv.gpus)}; the report covers that set (--gpus checks another)")
    if inv.source == "probe" and rep.inventory.gpus:  # read-only and informational: it never changes the exit code
        cards = [(g.index, sysfs_bus_id(g.bus_id) if g.bus_id else None) for g in rep.inventory.gpus]
        try:
            rep.gpu_unlock = U.probe(cards, rep.inventory.host.virt, read, system=rep.inventory.host.system)
        except Exception as e:  # noqa: BLE001 - informational: doctor must never fail here
            rep.gpu_unlock = U.failed(cards, e)
    if inv.source == "probe":
        import launch as L

        try:
            L.check_cache_links(L.cache_root_for(args.model, dict(os.environ)))
        except L.LaunchError as e:
            rep.add("FAIL", "cache-link", str(e))
            rep.exit_code = rep.exit_code or 1
    if inv.source == "probe" and rep.profile_id and args.model == "flash-next":
        # the same live gate merlin serve applies (serve_cli.resolve): what the host has left now, against this layout
        avail, tier, ple = mem_available_mib(read), rep.inventory.kv_tier_gib, rep.inventory.ple_table_gib
        if avail is not None and _gib(avail) < ram_floor_gib(len(rep.inventory.gpus), tier, ple):
            rep.add("FAIL", "ram-available", ram_available_text(len(rep.inventory.gpus), avail, tier, ple)
                    + "; merlin serve refuses to start")
            rep.exit_code = rep.exit_code or 1
    for text in dflash2_tier_conflicts(args.model, dict(os.environ), _saved_setup(Path(__file__).resolve().parents[2])):
        rep.add("FAIL", "kv-tier", text)
        rep.exit_code = rep.exit_code or 1
    if (
        args.strict
        and rep.exit_code == 0
        and any(f.level == "WARN" for f in rep.findings)
    ):
        rep.exit_code = 1
    out.write(
        json.dumps(report_dict(rep), indent=1) + "\n"
        if args.json
        else render_text(rep, reg)
    )
    return rep.exit_code


if __name__ == "__main__":
    R.safe_stdio()
    sys.exit(main())
