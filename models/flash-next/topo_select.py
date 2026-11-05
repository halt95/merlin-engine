#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)

# topo_select.py -- chooses the GPU order, the P2P/NCCL settings and (opt-in) a NUMA binding for the
# 4-GPU layout (tensor parallel 2 x pipeline parallel 2) before the serve script launches vLLM.
#
#   python3 models/flash-next/topo_select.py --dry-run      # show what it would decide on this machine
#
# The serve script calls it only when AUTO_TOPO=1 or AUTO_TOPO=strict (opt-in; unset or 0 keeps the fixed
# settings and never runs this file). Run by hand, it treats an unset AUTO_TOPO as 1.
# CUDA_VISIBLE_DEVICES=GPU-<uuid>,... picks the four GPUs on a machine with more than 4 GPUs.
# It never creates a CUDA context: it reads NVML, sysfs and /proc only.
"""Topology selector for the 4-GPU serving layout.

Purpose
-------
The serve script's fixed settings cover one kind of host (4 GPUs on one NUMA node, PCIe
peer-to-peer working, no NVLink). This module inspects the layout's GPUs and writes a small shell env
file that the serve script sources. It decides:

* the rank order of the layout's four GPUs (TP pairs (a,b),(c,d); PP edges (a,c),(b,d)), reordering only
  when another order is strictly better on a simple link-locality score;
* whether vLLM should verify P2P itself (``VLLM_SKIP_P2P_CHECK``) and the NCCL variables;
* optionally (``AUTO_TOPO_BIND=1``) a ``numactl`` launch prefix binding the server to the GPUs' NUMA node.

On the reference layout (4 GPUs, one NUMA node, P2P everywhere, no NVLink, bus order) the result is
byte-identical to the serve script's fixed settings. Any doubt falls back to those fixed settings.

Usage
-----
``python3 topo_select.py [--env-out PATH] [--json-dir DIR] [--dry-run] [--quiet]``

Caller intent is read only from the snapshot the serve script takes before applying its own defaults:
``TOPO_CALLER_SET_<V>=1`` + ``TOPO_CALLER_<V>=<value>`` for V in CUDA_VISIBLE_DEVICES, CUDA_DEVICE_ORDER,
VLLM_SKIP_P2P_CHECK, NCCL_P2P_LEVEL, NCCL_PROTO, PLE_HOME. When run by hand, export those to mimic a caller.

A caller ``CUDA_VISIBLE_DEVICES`` that is set but empty (or only whitespace) counts as unset, as it did in the
serve script's own ``${CUDA_VISIBLE_DEVICES:-0,1,2,3}`` default.

Opt-in: the serve script runs this module only when ``AUTO_TOPO`` is ``1`` or ``strict``; with
``AUTO_TOPO`` unset or ``0`` serve never invokes it and keeps its fixed settings. The module's own default is
different on purpose: run by hand (or by anything else), an unset or empty ``AUTO_TOPO`` means ``1``, so
``--dry-run`` shows a decision without extra flags. ``AUTO_TOPO=0`` given to the module itself writes the fixed
settings.

Controls: ``AUTO_TOPO`` (1 | strict | 0; unset = 1 here, see above), ``AUTO_TOPO_NCCL`` (legacy | default),
``AUTO_TOPO_BIND`` (0 | 1), ``GPU_SET`` (auto), ``CPU_SET`` (cpulist; set but empty means no CPUs, so binding is skipped),
``TOPO_TIMEOUT`` (seconds, default 20, at most ``threading.TIMEOUT_MAX``).
Diagnostics go to ``--json-dir``, else ``$VLLM_CACHE_ROOT/topology``, else ``$CACHE_ROOT/topology``, else
``~/.cache/flash-next/topology``. The serve script passes ``--json-dir`` explicitly: the ``topology/`` directory under
the same cache root it exports as ``VLLM_CACHE_ROOT``.

Test-only inputs: ``TOPO_FIXTURE`` (a JSON file replacing every hardware read) and ``TOPO_SYSFS_ROOT`` (a directory
prefixed to the ``/sys`` and ``/proc`` paths the real-hardware adapter reads, so the tests can run that adapter
against a fake sysfs tree and a fake ``pynvml`` module). Neither is meant for production use.

Exit codes: 0 decision written, 2 usage error, 3 strict-mode refusal, 4 timeout, 5 internal error.
Requires Python >= 3.10; ``pynvml`` (package ``nvidia-ml-py``) is imported only for real-host discovery.
"""

from __future__ import annotations

import argparse
import dataclasses
import glob
import hashlib
import itertools
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any

VERSION = 1
LEGACY_CVD = "0,1,2,3"
SNAPSHOT_VARS = ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "VLLM_SKIP_P2P_CHECK",
                 "NCCL_P2P_LEVEL", "NCCL_PROTO", "PLE_HOME")
ENV_VARS = ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "VLLM_SKIP_P2P_CHECK", "NCCL_P2P_LEVEL", "NCCL_PROTO")
LEGACY_ENV = {"CUDA_VISIBLE_DEVICES": LEGACY_CVD, "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
              "VLLM_SKIP_P2P_CHECK": "1", "NCCL_P2P_LEVEL": "SYS", "NCCL_PROTO": "Simple"}
ANCESTORS = ("INTERNAL", "SINGLE", "MULTIPLE", "HOSTBRIDGE", "NODE", "SYSTEM")
IDLE_MEM_MIB = 1024
CVD_UNSET_EQUIVALENT = re.compile(r"\s*")   # set-but-empty or whitespace-only CUDA_VISIBLE_DEVICES == unset
BIND_MIN_CPUS = 8
BIND_MIN_FREE_KIB = 1.2 * 69 * 1024 * 1024  # 1.2 x the layout's ~69 GiB resident host memory
EXIT_OK, EXIT_USAGE, EXIT_STRICT, EXIT_TIMEOUT, EXIT_INTERNAL = 0, 2, 3, 4, 5


class UsageError(Exception):
    pass


class StrictRefusal(Exception):
    pass


class NvmlUnavailable(Exception):
    pass


# ----------------------------------------------------------------------------------------------------------------
# small helpers

def normalise_bus_id(raw: Any) -> str:
    """'00000000:01:00.0' / '0000:01:00.0' / bytes -> '00000000:01:00.0' (lowercase, 8-digit domain)."""
    if isinstance(raw, bytes):
        raw = raw.decode("ascii", "replace")
    s = str(raw).strip().rstrip("\x00").lower()
    m = re.fullmatch(r"(?:([0-9a-f]+):)?([0-9a-f]{1,2}):([0-9a-f]{1,2})\.([0-7])", s)
    if not m:
        raise ValueError(f"unrecognised PCI bus id {raw!r}")
    dom = int(m.group(1) or "0", 16)
    return f"{dom:08x}:{int(m.group(2), 16):02x}:{int(m.group(3), 16):02x}.{m.group(4)}"


def sysfs_bus_id(bus_id: str) -> str:
    """Normalised id -> the sysfs spelling '0000:01:00.0'."""
    dom, rest = bus_id.split(":", 1)
    return f"{int(dom, 16):04x}:{rest}"


def parse_cpulist(text: str) -> set[int]:
    """'0-3,8,10-11' -> {0,1,2,3,8,10,11}. Raises ValueError on malformed input."""
    out: set[int] = set()
    text = text.strip()
    if not text:
        return out
    for part in text.split(","):
        part = part.strip()
        m = re.fullmatch(r"(\d+)(?:-(\d+))?(?::(\d+))?", part)
        if not m:
            raise ValueError(f"bad cpulist element {part!r}")
        lo = int(m.group(1))
        hi = int(m.group(2)) if m.group(2) else lo
        step = int(m.group(3)) if m.group(3) else 1
        if hi < lo or step < 1:
            raise ValueError(f"bad cpulist range {part!r}")
        out.update(range(lo, hi + 1, step))
    return out


def format_cpulist(cpus: set[int]) -> str:
    """{0,1,2,3,8} -> '0-3,8'."""
    items = sorted(cpus)
    parts = []
    i = 0
    while i < len(items):
        j = i
        while j + 1 < len(items) and items[j + 1] == items[j] + 1:
            j += 1
        parts.append(str(items[i]) if i == j else f"{items[i]}-{items[j]}")
        i = j + 1
    return ",".join(parts)


def host_path(path: str) -> str:
    """'/sys/...' or '/proc/...' under TOPO_SYSFS_ROOT when that is set (tests only), else unchanged. On Windows,
    where a file name cannot contain ':', the fake tree spells the ':' of PCI addresses as '_'."""
    root = os.environ.get("TOPO_SYSFS_ROOT")
    if not root:
        return path
    rel = path.lstrip("/")
    return os.path.join(root, rel.replace(":", "_") if os.name == "nt" else rel)


def short_uuid(uuid: str) -> str:
    return uuid[:8] + ".." if len(uuid) > 10 else uuid


def fmt_class(c: float | None) -> str:
    if c is None:
        return "?"
    return str(int(c)) if float(c).is_integer() else str(c)


def atomic_write(path: str, text: str) -> None:
    d = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(prefix=".topo_select.", dir=d)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# ----------------------------------------------------------------------------------------------------------------
# facts

@dataclasses.dataclass
class Gpu:
    uuid: str
    bus_id: str                      # normalised, the sort key
    key: Any                         # fixture index or NVML handle index
    ordinal: int = -1                # CUDA ordinal under PCI_BUS_ID order
    numa: int | None = None
    pcie_gen_max: int | None = None
    pcie_gen_cur: int | None = None
    pcie_width_cur: int | None = None
    pcie_width_max: int | None = None
    details_read: bool = False


@dataclasses.dataclass
class PairFacts:
    ancestor: str | None             # one of ANCESTORS, or None = unknown
    p2p_read: str | None             # "OK" or another status; None = unknown
    p2p_write: str | None
    nvlink: bool

    @property
    def p2p_ok(self) -> bool | None:
        vals = (self.p2p_read, self.p2p_write)
        if any(v is None for v in vals):
            return None          # one direction unknown: the pair is unknown, never "not supported"
        return all(v == "OK" for v in vals)


@dataclasses.dataclass
class CpuFacts:
    affinity: set[int] | None
    mems_allowed: set[int] | None
    nodes: dict[int, tuple[set[int] | None, int | None]]   # node -> (cpus, mem_free_kib)


class Topology:
    """Common interface over fixture and NVML facts. Everything beyond identity is read lazily, per GPU/pair."""

    source = "?"

    def __init__(self) -> None:
        self.gpus: list[Gpu] = []            # sorted by bus id
        self.warnings: list[str] = []
        self._pairs: dict[tuple[str, str], PairFacts] = {}

    def _finish_identity(self) -> None:
        self.gpus.sort(key=lambda g: g.bus_id)
        for k, g in enumerate(self.gpus):
            g.ordinal = k

    def details(self, g: Gpu) -> Gpu:
        if not g.details_read:
            self._read_details(g)
            g.details_read = True
        return g

    def pair(self, a: Gpu, b: Gpu) -> PairFacts:
        k = tuple(sorted((a.bus_id, b.bus_id)))
        if k not in self._pairs:
            self._pairs[k] = self._read_pair(a, b)
        return self._pairs[k]

    # implemented by subclasses
    def _read_details(self, g: Gpu) -> None: ...
    def _read_pair(self, a: Gpu, b: Gpu) -> PairFacts: ...
    def occupancy(self, g: Gpu) -> tuple[int | None, int | None]: ...
    def system_nodes(self) -> list[int] | None: ...
    def cpu(self) -> CpuFacts: ...
    def numactl_present(self) -> bool: ...
    def numactl_dry_run(self, cpus: str, node: int) -> bool: ...

    def resolve_numa(self, raw: int | None) -> int | None:
        """sysfs numa_node -> node; -1 on a single-node system means that node, otherwise unknown."""
        if raw is None:
            return None
        if raw >= 0:
            return raw
        nodes = self.system_nodes()
        if raw == -1 and nodes is not None and len(nodes) == 1:
            return nodes[0]
        return None


def _opt_int(v: Any) -> int | None:
    if v is None or isinstance(v, bool):
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


class FixtureTopology(Topology):
    source = "fixture"

    def __init__(self, data: dict) -> None:
        super().__init__()
        self.data = data
        self.fixture_name = data.get("name")
        for i, g in enumerate(data.get("gpus") or []):
            self.gpus.append(Gpu(uuid=str(g["uuid"]), bus_id=normalise_bus_id(g["bus_id"]), key=i))
        self._finish_identity()
        self._cpu_raw = data.get("cpu") or {}
        self._numactl = data.get("numactl") or {}

    def _raw(self, g: Gpu) -> dict:
        return self.data["gpus"][g.key]

    def _read_details(self, g: Gpu) -> None:
        r = self._raw(g)
        g.numa = self.resolve_numa(_opt_int(r.get("numa")))
        g.pcie_gen_max = _opt_int(r.get("pcie_gen_max"))
        g.pcie_gen_cur = _opt_int(r.get("pcie_gen_cur"))
        g.pcie_width_cur = _opt_int(r.get("pcie_width_cur"))
        g.pcie_width_max = _opt_int(r.get("pcie_width_max"))

    def _read_pair(self, a: Gpu, b: Gpu) -> PairFacts:
        pairs = self.data.get("pairs") or {}
        p = pairs.get(f"{a.key}-{b.key}") or pairs.get(f"{b.key}-{a.key}")
        if p is None:
            return PairFacts(None, None, None, False)
        anc = p.get("ancestor")
        anc = str(anc).upper() if anc is not None else None
        rd, wr = p.get("p2p_read"), p.get("p2p_write")
        return PairFacts(anc if anc in ANCESTORS else None,
                         None if rd is None else str(rd), None if wr is None else str(wr),
                         bool(p.get("nvlink", False)))

    def occupancy(self, g: Gpu) -> tuple[int | None, int | None]:
        r = self._raw(g)
        return _opt_int(r.get("procs")), _opt_int(r.get("mem_used_mib"))

    def system_nodes(self) -> list[int] | None:
        nodes = self._cpu_raw.get("nodes")
        if not isinstance(nodes, dict):
            return None
        return sorted(int(k) for k in nodes)

    def cpu(self) -> CpuFacts:
        c = self._cpu_raw

        def cl(v: Any) -> set[int] | None:
            return None if v is None else parse_cpulist(str(v))
        nodes = {int(k): (cl(v.get("cpulist")), _opt_int(v.get("mem_free_kib")))
                 for k, v in (c.get("nodes") or {}).items()}
        return CpuFacts(cl(c.get("affinity")), cl(c.get("mems_allowed")), nodes)

    def numactl_present(self) -> bool:
        return bool(self._numactl.get("present", False))

    def numactl_dry_run(self, cpus: str, node: int) -> bool:
        return bool(self._numactl.get("dry_run_ok", False))


class NvmlTopology(Topology):
    """Real-host facts from NVML (no CUDA context), sysfs and /proc."""

    source = "nvml"

    def __init__(self) -> None:
        super().__init__()
        try:
            import pynvml  # from nvidia-ml-py; imported lazily on purpose
            pynvml.nvmlInit()
        except Exception as e:  # ImportError, NVMLError, OSError (library missing)
            raise NvmlUnavailable(f"{type(e).__name__}: {e}") from e
        self.nv = pynvml
        self._handles: dict[int, Any] = {}
        self._nvlink_peers: dict[str, set[str] | None] = {}
        self._nodes_cache: list[int] | None | bool = False
        try:
            for i in range(pynvml.nvmlDeviceGetCount()):
                h = pynvml.nvmlDeviceGetHandleByIndex(i)
                uuid = pynvml.nvmlDeviceGetUUID(h)
                uuid = uuid.decode() if isinstance(uuid, bytes) else str(uuid)
                bus = normalise_bus_id(pynvml.nvmlDeviceGetPciInfo(h).busId)
                self._handles[i] = h
                self.gpus.append(Gpu(uuid=uuid, bus_id=bus, key=i))
        except Exception as e:
            self.close()
            raise NvmlUnavailable(f"device enumeration failed: {type(e).__name__}: {e}") from e
        self._finish_identity()

    def close(self) -> None:
        try:
            self.nv.nvmlShutdown()
        except Exception:
            pass

    def const(self, name: str, default: int) -> int:
        """Some nvidia-ml-py versions define NVML_* constants as 1-tuples; unwrap before passing to NVML."""
        c = getattr(self.nv, name, default)
        return c[0] if isinstance(c, tuple) else c

    def _try(self, fn, *args):
        try:
            return fn(*args)
        except Exception:
            return None

    def _read_details(self, g: Gpu) -> None:
        h = self._handles[g.key]
        nv = self.nv
        g.pcie_gen_max = self._try(nv.nvmlDeviceGetMaxPcieLinkGeneration, h)
        g.pcie_gen_cur = self._try(nv.nvmlDeviceGetCurrPcieLinkGeneration, h)
        g.pcie_width_max = self._try(nv.nvmlDeviceGetMaxPcieLinkWidth, h)
        g.pcie_width_cur = self._try(nv.nvmlDeviceGetCurrPcieLinkWidth, h)
        raw = None
        try:
            with open(host_path(f"/sys/bus/pci/devices/{sysfs_bus_id(g.bus_id)}/numa_node"),
                      encoding="ascii") as f:
                raw = int(f.read().strip())
        except (OSError, ValueError):
            raw = None
        g.numa = self.resolve_numa(raw)

    def _ancestor(self, ha, hb) -> str | None:
        try:
            level = self.nv.nvmlDeviceGetTopologyCommonAncestor(ha, hb)
        except Exception:
            return None
        level = level[0] if isinstance(level, tuple) else level
        names = {self.const(f"NVML_TOPOLOGY_{n}", v): n
                 for n, v in (("INTERNAL", 0), ("SINGLE", 10), ("MULTIPLE", 20),
                              ("HOSTBRIDGE", 30), ("NODE", 40), ("SYSTEM", 50))}
        return names.get(level)

    def _p2p(self, ha, hb, index_name: str, index_default: int) -> str | None:
        """'OK', another status name, or None when the query itself failed (unknown, not unsupported)."""
        try:
            st = self.nv.nvmlDeviceGetP2PStatus(ha, hb, self.const(index_name, index_default))
        except Exception:
            return None
        st = st[0] if isinstance(st, tuple) else st
        if st == self.const("NVML_P2P_STATUS_OK", 0):
            return "OK"
        names = {1: "CNS", 2: "GNS", 3: "TNS", 4: "DISABLED_BY_REGKEY", 5: "NS", 6: "UNKNOWN"}
        return names.get(st, f"STATUS_{st}")

    def _peers(self, g: Gpu) -> set[str] | None:
        """Bus ids at the far end of this GPU's active NVLinks (empty on cards without NVLink)."""
        if g.bus_id in self._nvlink_peers:
            return self._nvlink_peers[g.bus_id]
        nv, h = self.nv, self._handles[g.key]
        peers: set[str] = set()
        enabled = self.const("NVML_FEATURE_ENABLED", 1)
        remote_fn = (getattr(nv, "nvmlDeviceGetNvLinkRemotePciInfo_v2", None)
                     or getattr(nv, "nvmlDeviceGetNvLinkRemotePciInfo", None))
        for link in range(self.const("NVML_NVLINK_MAX_LINKS", 18)):
            try:
                state = nv.nvmlDeviceGetNvLinkState(h, link)
            except Exception as e:
                code = getattr(e, "value", None)
                if code not in (self.const("NVML_ERROR_NOT_SUPPORTED", 3),
                                self.const("NVML_ERROR_INVALID_ARGUMENT", 2)):
                    self.warnings.append(f"NVLink state query failed on {g.uuid} link {link}: {e}")
                break
            state = state[0] if isinstance(state, tuple) else state
            if state != enabled or remote_fn is None:
                continue
            try:
                peers.add(normalise_bus_id(remote_fn(h, link).busId))
            except Exception:
                continue
        self._nvlink_peers[g.bus_id] = peers
        return peers

    def _read_pair(self, a: Gpu, b: Gpu) -> PairFacts:
        ha, hb = self._handles[a.key], self._handles[b.key]
        nvlink = b.bus_id in (self._peers(a) or ()) or a.bus_id in (self._peers(b) or ())
        return PairFacts(self._ancestor(ha, hb),
                         self._p2p(ha, hb, "NVML_P2P_CAPS_INDEX_READ", 0),
                         self._p2p(ha, hb, "NVML_P2P_CAPS_INDEX_WRITE", 1),
                         nvlink)

    def occupancy(self, g: Gpu) -> tuple[int | None, int | None]:
        h = self._handles[g.key]
        procs = self._try(self.nv.nvmlDeviceGetComputeRunningProcesses, h)
        mem = self._try(self.nv.nvmlDeviceGetMemoryInfo, h)
        return (None if procs is None else len(procs),
                None if mem is None else int(mem.used) // (1024 * 1024))

    def system_nodes(self) -> list[int] | None:
        if self._nodes_cache is False:
            paths = glob.glob(os.path.join(host_path("/sys/devices/system/node"), "node[0-9]*"))
            self._nodes_cache = sorted(int(os.path.basename(p)[4:]) for p in paths) or None
        return self._nodes_cache  # type: ignore[return-value]

    def cpu(self) -> CpuFacts:
        try:
            affinity: set[int] | None = set(os.sched_getaffinity(0))
        except (AttributeError, OSError):
            affinity = None
        mems = None
        try:
            with open(host_path("/proc/self/status"), encoding="ascii", errors="replace") as f:
                for line in f:
                    if line.startswith("Mems_allowed_list:"):
                        mems = parse_cpulist(line.split(":", 1)[1])
        except (OSError, ValueError):
            mems = None
        nodes: dict[int, tuple[set[int] | None, int | None]] = {}
        for n in self.system_nodes() or []:
            base = host_path(f"/sys/devices/system/node/node{n}")
            try:
                with open(f"{base}/cpulist", encoding="ascii") as f:
                    cpus: set[int] | None = parse_cpulist(f.read())
            except (OSError, ValueError):
                cpus = None
            free = None
            try:
                with open(f"{base}/meminfo", encoding="ascii") as f:
                    for line in f:
                        m = re.match(r"Node\s+\d+\s+MemFree:\s+(\d+)\s*kB", line)
                        if m:
                            free = int(m.group(1))
            except OSError:
                free = None
            nodes[n] = (cpus, free)
        return CpuFacts(affinity, mems, nodes)

    def numactl_present(self) -> bool:
        return shutil.which("numactl") is not None

    def numactl_dry_run(self, cpus: str, node: int) -> bool:
        exe = shutil.which("numactl")
        if exe is None:
            return False
        try:
            r = subprocess.run([exe, f"--physcpubind={cpus}", f"--preferred={node}", "true"],
                               stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               timeout=10, check=False)
            return r.returncode == 0
        except (OSError, subprocess.SubprocessError):
            return False


def read_facts_fixture(path: str) -> Topology:
    """Load a fixture. faults.nvml 'error' behaves like a failed NVML init; 'timeout' hangs (the caller times out)."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    fault = (data.get("faults") or {}).get("nvml")
    if fault == "error":
        raise NvmlUnavailable("fixture fault: nvml error")
    if fault == "timeout":
        while True:
            time.sleep(3600)
    return FixtureTopology(data)


def read_facts_nvml() -> Topology:
    return NvmlTopology()


# ----------------------------------------------------------------------------------------------------------------
# decisions

@dataclasses.dataclass
class Controls:
    auto_topo: str          # "1" | "strict" | "0"
    nccl: str               # "legacy" | "default"
    bind: bool
    lane_auto: bool
    lane_cpus: str | None
    caller: dict[str, str | None]   # snapshot: value, or None when the caller did not set it


@dataclasses.dataclass
class Decision:
    status: str = "legacy"
    reason: str = ""
    facts_source: str = "?"
    visible: int = 0
    lane: list[Gpu] = dataclasses.field(default_factory=list)          # final order
    input_order: list[Gpu] = dataclasses.field(default_factory=list)
    reordered: bool = False
    cvd_numeric_ok: bool = False     # caller gave numeric tokens or nothing (numeric form may be kept)
    auto_lane: bool = False
    classes: dict[tuple[int, int], float | None] = dataclasses.field(default_factory=dict)  # over final order
    p2p: dict[tuple[int, int], PairFacts] = dataclasses.field(default_factory=dict)
    candidates: list[dict] = dataclasses.field(default_factory=list)
    chosen_score: tuple | None = None
    input_score: tuple | None = None
    env: dict[str, tuple[str | None, str]] = dataclasses.field(default_factory=dict)  # value, provenance
    prefix: list[str] = dataclasses.field(default_factory=list)
    binding: dict = dataclasses.field(default_factory=lambda: {"enabled": False, "command": None,
                                                                "skipped_reason": "not_requested"})
    warnings: list[str] = dataclasses.field(default_factory=list)


def resolve_lane(topo: Topology, ctl: Controls, dec: Decision) -> list[Gpu] | None:
    """Map the caller's CUDA_VISIBLE_DEVICES (or the visible set) to 4 distinct GPUs, or set a legacy reason."""
    gpus = topo.gpus
    cvd = caller_cvd(ctl)
    if cvd is not None:
        lane: list[Gpu] = []
        numeric = True
        for tok in (t.strip() for t in cvd.split(",")):
            if re.fullmatch(r"\d+", tok):
                k = int(tok)
                if k >= len(gpus):
                    dec.reason = "cvd_unresolved"
                    return None
                lane.append(gpus[k])
            elif tok.startswith("GPU-"):
                numeric = False
                t = tok.lower()
                exact = [g for g in gpus if g.uuid.lower() == t]
                match = exact or [g for g in gpus if g.uuid.lower().startswith(t)]
                if len(match) != 1:
                    dec.reason = "cvd_unresolved"
                    return None
                lane.append(match[0])
            else:
                dec.reason = "cvd_unresolved"
                return None
        if len({g.bus_id for g in lane}) != len(lane):
            dec.reason = "cvd_duplicate"
            return None
        if len(lane) != 4:
            dec.reason = "gpus_not_4"
            return None
        dec.cvd_numeric_ok = numeric
        return lane

    dec.cvd_numeric_ok = True
    if len(gpus) == 4:
        return list(gpus)
    if len(gpus) < 4:
        dec.reason = "fewer_than_4_visible"
        return None
    # more than 4 visible, no GPU set given. Strict mode refuses before anything else, GPU_SET=auto included: an idle
    # observation is a guess about which GPUs are free, and strict means "do not guess".
    if ctl.auto_topo == "strict":
        raise StrictRefusal(f"topo: {len(gpus)} GPUs visible and CUDA_VISIBLE_DEVICES is not set; AUTO_TOPO=strict "
                            "refuses (GPU_SET=auto is not used in strict mode). "
                            "Set CUDA_VISIBLE_DEVICES=GPU-<uuid>,... (4 GPUs).")
    if ctl.lane_auto:
        return auto_lane(topo, ctl, dec)
    dec.reason = "more_than_4_visible"
    dec.warnings.append(f"{len(gpus)} GPUs visible and CUDA_VISIBLE_DEVICES is not set: using GPUs 0,1,2,3 in "
                        "PCI bus order. Pick the four GPUs explicitly with CUDA_VISIBLE_DEVICES=GPU-<uuid>,... "
                        "(4 GPUs; list them with nvidia-smi -L) or set GPU_SET=auto.")
    return None


def auto_lane(topo: Topology, ctl: Controls, dec: Decision) -> list[Gpu] | None:
    """GPU_SET=auto: best 4-subset of idle GPUs (all on one NUMA node first, then score, then smallest bus ids)."""
    dec.auto_lane = True
    eligible = []
    for g in topo.gpus:
        procs, mem = topo.occupancy(g)
        if procs == 0 and mem is not None and mem <= IDLE_MEM_MIB:
            eligible.append(g)
    if len(eligible) < 4:
        dec.reason = "auto_insufficient_idle"
        return None
    best_key, best = None, None
    for subset in itertools.combinations(eligible, 4):
        cls = lane_classes(topo, list(subset))
        if cls is None:
            continue
        top = max(order_score(p, cls) for p in itertools.permutations(range(4)))
        numas = {topo.details(g).numa for g in subset}
        one_node = len(numas) == 1 and None not in numas
        key = (one_node, top, tuple(-int(c, 16) for g in subset for c in re.split(r"[:.]", g.bus_id)))
        if best_key is None or key > best_key:
            best_key, best = key, list(subset)
    if best is None:
        dec.reason = "auto_facts_unknown"
        return None
    dec.warnings.append("GPU_SET=auto picked " + ",".join(g.uuid for g in best) +
                        " (idle when checked; an idle observation is not a reservation)")
    return best


def link_class(p: PairFacts) -> float | None:
    """NVLink 5; same switch 4 (P2P) / 2.5; host bridge or same node 3 (P2P) / 2; cross-socket 1; unknown None."""
    if p.nvlink:
        return 5
    if p.ancestor is None or p.p2p_ok is None:
        return None
    if p.ancestor == "SYSTEM":
        return 1
    if p.ancestor in ("INTERNAL", "SINGLE", "MULTIPLE"):
        return 4 if p.p2p_ok else 2.5
    return 3 if p.p2p_ok else 2      # HOSTBRIDGE, NODE


def lane_classes(topo: Topology, lane: list[Gpu]) -> dict[tuple[int, int], float] | None:
    """Link class for all six pairs (indices into lane); None when any class is unknown."""
    out = {}
    for i, j in itertools.combinations(range(4), 2):
        c = link_class(topo.pair(lane[i], lane[j]))
        if c is None:
            return None
        out[(i, j)] = c
    return out


def order_score(order: tuple[int, ...], cls: dict[tuple[int, int], float]) -> tuple:
    def c(x: int, y: int) -> float:
        return cls[(min(x, y), max(x, y))]
    a, b, cc, d = order
    tp = (c(a, b), c(cc, d))
    return (min(tp), sum(tp), min(c(a, cc), c(b, d)))


def choose_order(topo: Topology, lane: list[Gpu], dec: Decision) -> list[Gpu] | None:
    """Keep the input order unless a permutation scores strictly higher; None when facts are unknown."""
    cls = lane_classes(topo, lane)
    if cls is None:
        return None
    identity = (0, 1, 2, 3)
    dec.input_score = order_score(identity, cls)
    scored = [(order_score(p, cls), p) for p in itertools.permutations(range(4))]
    scored.sort(key=lambda sp: (tuple(-x for x in sp[0]), tuple(lane[i].bus_id for i in sp[1])))
    dec.candidates = [{"order": [lane[i].uuid for i in p], "ordinals": [lane[i].ordinal for i in p],
                       "score": list(s)} for s, p in scored[:5]]
    best_score, best = scored[0]
    if best_score > dec.input_score:
        dec.reordered = True
        dec.chosen_score = best_score
        return [lane[i] for i in best]
    dec.chosen_score = dec.input_score
    return list(lane)


def is_reference(topo: Topology, dec: Decision) -> bool:
    lane = dec.lane
    if dec.visible != 4 or not dec.cvd_numeric_ok or dec.reordered or dec.auto_lane:
        return False
    for a, b in itertools.combinations(lane, 2):
        p = topo.pair(a, b)
        if p.nvlink or p.p2p_ok is not True:
            return False
    numas = {g.numa for g in lane}
    return len(numas) == 1 and None not in numas


def caller_cvd(ctl: Controls) -> str | None:
    """The caller's CUDA_VISIBLE_DEVICES, or None when unset, empty or whitespace-only (the serve script's
    ${CUDA_VISIBLE_DEVICES:-0,1,2,3} default treats an empty value as unset; whitespace is no more usable)."""
    cvd = ctl.caller["CUDA_VISIBLE_DEVICES"]
    if cvd is None or CVD_UNSET_EQUIVALENT.fullmatch(cvd):
        return None
    return cvd


def decide_env(topo: Topology | None, ctl: Controls, dec: Decision) -> None:
    """Fill dec.env (value, provenance). Caller-set values always win; CUDA_VISIBLE_DEVICES may be reordered."""
    caller = ctl.caller
    cvd = caller_cvd(ctl)
    legacy_cvd = cvd if cvd is not None else LEGACY_CVD
    cvd_prov = "caller" if cvd is not None else "legacy"

    if dec.status == "legacy" or dec.reason == "reference":
        wanted = dict(LEGACY_ENV)
        prov = "legacy"
        wanted["CUDA_VISIBLE_DEVICES"] = legacy_cvd
    else:
        prov = "selector"
        tp_pairs = ((dec.lane[0], dec.lane[1]), (dec.lane[2], dec.lane[3]))
        tp_p2p = any(topo.pair(a, b).p2p_ok is True for a, b in tp_pairs)  # type: ignore[union-attr]
        wanted = {"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "NCCL_P2P_LEVEL": "SYS", "NCCL_PROTO": "Simple",
                  "VLLM_SKIP_P2P_CHECK": "0" if tp_p2p else "1"}
        if not tp_p2p and ctl.nccl == "default":
            wanted["NCCL_PROTO"] = None
        if not dec.reordered and dec.cvd_numeric_ok and not dec.auto_lane:
            wanted["CUDA_VISIBLE_DEVICES"] = legacy_cvd
        else:
            wanted["CUDA_VISIBLE_DEVICES"] = ",".join(g.uuid for g in dec.lane)
            cvd_prov = "selector"

    for name in ENV_VARS:
        if name == "CUDA_VISIBLE_DEVICES":
            dec.env[name] = (wanted[name], cvd_prov if wanted[name] == legacy_cvd else "selector")
        elif caller[name] is not None:
            dec.env[name] = (caller[name], "caller")
        else:
            dec.env[name] = (wanted[name], prov)


def decide_binding(topo: Topology, ctl: Controls, dec: Decision) -> None:
    """Tier-1 binding: all four GPUs on one known node N; every precondition must pass or no prefix."""
    if not ctl.bind:
        return
    b = dec.binding
    b["enabled"] = True

    def skip(reason: str) -> None:
        b["enabled"], b["skipped_reason"] = False, reason

    # The prefix binds the whole server process tree. Never combine it with vLLM's own --numa-bind: that option
    # auto-detects from the (already constrained) affinity and refuses to start, or binds a second time.
    if dec.status != "ok":
        return skip("legacy_decision")
    numas = {g.numa for g in dec.lane}
    if None in numas:
        return skip("numa_unknown")
    if len(numas) != 1:
        return skip("gpu_set_spans_nodes")
    node = numas.pop()
    cpu = topo.cpu()
    if not topo.numactl_present():
        return skip("numactl_missing")
    if cpu.affinity is None:
        return skip("affinity_unknown")
    node_cpus, node_free = cpu.nodes.get(node, (None, None))
    if node_cpus is None:
        return skip("node_cpulist_unknown")
    chosen = cpu.affinity & node_cpus
    if ctl.lane_cpus is not None:
        try:
            chosen &= parse_cpulist(ctl.lane_cpus)
        except ValueError:
            return skip("cpu_set_invalid")
    if len(chosen) < BIND_MIN_CPUS:
        return skip(f"too_few_cpus({len(chosen)})")
    if cpu.mems_allowed is None or node not in cpu.mems_allowed:
        return skip("node_not_in_mems_allowed")
    if node_free is None or node_free < BIND_MIN_FREE_KIB:
        return skip("node_mem_free_below_1.2x69GiB")
    cpus = format_cpulist(chosen)
    if not topo.numactl_dry_run(cpus, node):
        return skip("numactl_dry_run_failed")
    dec.prefix = ["numactl", f"--physcpubind={cpus}", f"--preferred={node}", "--"]
    b["command"] = " ".join(dec.prefix)
    b["skipped_reason"] = None


def decide(topo_loader, ctl: Controls) -> tuple[Decision, Topology | None]:
    """Run discovery and every decision. Runs inside the timeout guard."""
    dec = Decision()
    if ctl.auto_topo == "0":
        dec.reason = "auto_topo_off"
        decide_env(None, ctl, dec)
        return dec, None
    order = ctl.caller["CUDA_DEVICE_ORDER"]
    if order is not None and order != "PCI_BUS_ID":
        # numeric tokens would mean a different enumeration than the one resolved here: do not touch the GPU set
        dec.reason = "device_order_not_pci_bus_id"
        decide_env(None, ctl, dec)
        return dec, None
    try:
        topo = topo_loader()
    except NvmlUnavailable as e:
        dec.reason = "nvml_unavailable"
        dec.warnings.append(f"NVML unavailable ({e}); using the legacy settings")
        decide_env(None, ctl, dec)
        return dec, None
    dec.facts_source = topo.source
    dec.visible = len(topo.gpus)
    if len({g.bus_id for g in topo.gpus}) != len(topo.gpus) or len({g.uuid for g in topo.gpus}) != len(topo.gpus):
        dec.reason = "facts_duplicate"
    else:
        lane = resolve_lane(topo, ctl, dec)
        if lane is not None:
            for g in lane:
                topo.details(g)
            dec.input_order = list(lane)
            final = choose_order(topo, lane, dec)
            if final is None:
                dec.lane = list(lane)
                dec.reason = "facts_unknown"
            else:
                dec.lane = final
                dec.status = "ok"
                if is_reference(topo, dec):
                    dec.reason = "reference"
                else:
                    dec.reason = "reordered" if dec.reordered else "kept"
            for (i, a), (j, b) in itertools.combinations(enumerate(dec.lane), 2):
                p = topo.pair(a, b)
                dec.p2p[(i, j)] = p
                dec.classes[(i, j)] = link_class(p)
    decide_env(topo, ctl, dec)
    decide_binding(topo, ctl, dec)
    dec.warnings.extend(topo.warnings)
    return dec, topo


# ----------------------------------------------------------------------------------------------------------------
# output

def summary_line(dec: Decision) -> str:
    env = {k: v for k, (v, _) in dec.env.items()}
    proto = env["NCCL_PROTO"] if env["NCCL_PROTO"] is not None else "unset"
    bind = dec.binding
    bind_s = "on" if dec.prefix else ("off" if bind["skipped_reason"] == "not_requested"
                                      else f"off ({bind['skipped_reason']})")
    if dec.status != "ok":
        return (f"topo: legacy ({dec.reason}) | CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']} | "
                f"skip_p2p={env['VLLM_SKIP_P2P_CHECK']} | NCCL_PROTO={proto} | bind {bind_s}")
    lane = dec.lane
    numas = sorted({str(g.numa) if g.numa is not None else "?" for g in lane})
    ok = sum(1 for p in dec.p2p.values() if p.p2p_ok)
    p2p_s = "none" if ok == 0 else ("all" if ok == len(dec.p2p) else f"{ok}/{len(dec.p2p)} pairs")
    tp = f"({lane[0].ordinal},{lane[1].ordinal})({lane[2].ordinal},{lane[3].ordinal})"
    cls = dec.classes
    return (f"topo: 4 GPUs [{' '.join(short_uuid(g.uuid) for g in lane)}] numa{{{','.join(numas)}}} | "
            f"TP {tp} classes {fmt_class(cls[(0, 1)])}/{fmt_class(cls[(2, 3)])} | "
            f"PP {fmt_class(min(cls[(0, 2)], cls[(1, 3)]))} | "
            f"order {'reordered' if dec.reordered else 'kept'} | P2P {p2p_s} | "
            f"skip_p2p={env['VLLM_SKIP_P2P_CHECK']} | NCCL_PROTO={proto} | bind {bind_s} | reason={dec.reason}")


def render_env(dec: Decision, json_path: str | None) -> str:
    lines = [f"# topo_select v{VERSION} status={dec.status} reason={dec.reason}"]
    for name in ENV_VARS:
        value = dec.env[name][0]
        lines.append(f"unset {name}" if value is None else f"export {name}={shlex.quote(value)}")
    prefix = " ".join(shlex.quote(w) for w in dec.prefix)
    lines.append(f"export TOPO_LAUNCH_PREFIX={shlex.quote(prefix)}")
    lines.append(f"export TOPO_DECISION_JSON={shlex.quote(json_path or '')}")
    return "\n".join(lines) + "\n"


def diagnostics(dec: Decision, ctl: Controls) -> dict:
    ple = ctl.caller["PLE_HOME"]
    try:
        ple_idx = int(ple) if ple is not None else 0
    except ValueError:
        ple_idx = -1
    return {
        "version": VERSION,
        "status": dec.status,
        "reason": dec.reason,
        "gpu_set": [{"uuid": g.uuid, "bus_id": g.bus_id, "numa": g.numa, "pcie_gen_max": g.pcie_gen_max,
                     "pcie_width_cur": g.pcie_width_cur, "pcie_width_max": g.pcie_width_max} for g in dec.lane],
        "input_order": [g.uuid for g in dec.input_order],
        "link_matrix": {f"{i}-{j}": c for (i, j), c in sorted(dec.classes.items())},
        "p2p_matrix": {f"{i}-{j}": {"read": p.p2p_read, "write": p.p2p_write, "nvlink": p.nvlink,
                                    "ancestor": p.ancestor} for (i, j), p in sorted(dec.p2p.items())},
        "candidates": dec.candidates,
        "chosen_score": list(dec.chosen_score) if dec.chosen_score else None,
        "input_score": list(dec.input_score) if dec.input_score else None,
        "env": {k: {"value": v, "provenance": p} for k, (v, p) in dec.env.items()},
        "ple_home_uuid": dec.lane[ple_idx].uuid if 0 <= ple_idx < len(dec.lane) else None,
        "p2p_probe_cache_hint": dec.env["CUDA_VISIBLE_DEVICES"][0],
        "binding": dict(dec.binding),
        "warnings": list(dec.warnings),
        "facts_source": dec.facts_source,
    }


def write_outputs(dec: Decision, ctl: Controls, args: argparse.Namespace) -> None:
    """Diagnostics JSON first (non-fatal), then the env file (atomic); or print both for --dry-run."""
    for w in dec.warnings:
        print(f"topo: warning: {w}", file=sys.stderr)
    line = summary_line(dec)
    if args.dry_run:
        if not args.quiet:
            print(line)
        sys.stdout.write(render_env(dec, None))
        return
    json_path = None
    try:
        json_dir = args.json_dir or os.path.join(
            os.environ.get("VLLM_CACHE_ROOT") or os.environ.get("CACHE_ROOT")
            or os.path.join(os.path.expanduser("~"), ".cache", "flash-next"),
            "topology")
        os.makedirs(json_dir, exist_ok=True)
        lane_id = hashlib.sha256(",".join(sorted(g.uuid for g in dec.lane)).encode()).hexdigest()[:12]
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        path = os.path.join(json_dir, f"{lane_id}-{stamp}.json")
        atomic_write(path, json.dumps(diagnostics(dec, ctl), indent=2) + "\n")
        json_path = os.path.abspath(path)
    except Exception as e:
        print(f"topo: could not write diagnostics ({type(e).__name__}: {e}); continuing", file=sys.stderr)
    atomic_write(args.env_out, render_env(dec, json_path))
    if not args.quiet:
        print(line, file=sys.stderr)


# ----------------------------------------------------------------------------------------------------------------
# entry point

def read_controls(environ: dict[str, str]) -> Controls:
    def get(name: str) -> str:
        return environ.get(name, "").strip()

    # unset means on HERE: serve only calls this module with AUTO_TOPO=1 or strict, so this
    # default only matters when the selector is run by hand.
    auto = get("AUTO_TOPO") or "1"
    if auto not in ("1", "0", "strict"):
        raise UsageError(f"AUTO_TOPO must be 1, 0 or strict (got {auto!r})")
    nccl = get("AUTO_TOPO_NCCL") or "legacy"
    if nccl not in ("legacy", "default"):
        raise UsageError(f"AUTO_TOPO_NCCL must be legacy or default (got {nccl!r})")
    bind = get("AUTO_TOPO_BIND") or "0"
    if bind not in ("0", "1"):
        raise UsageError(f"AUTO_TOPO_BIND must be 0 or 1 (got {bind!r})")
    # GPU_SET / CPU_SET; the older LANE_GPUS / LANE_CPUS names are read when the new one is unset
    gpu_var = "GPU_SET" if "GPU_SET" in environ else "LANE_GPUS"
    cpu_var = "CPU_SET" if "CPU_SET" in environ else "LANE_CPUS"
    gpu_set = get(gpu_var)
    if gpu_set not in ("", "auto"):
        raise UsageError(f"{gpu_var} supports only 'auto' (got {gpu_set!r}); pick the GPUs with CUDA_VISIBLE_DEVICES")
    caller: dict[str, str | None] = {}
    for v in SNAPSHOT_VARS:
        caller[v] = environ.get(f"TOPO_CALLER_{v}", "") if environ.get(f"TOPO_CALLER_SET_{v}") == "1" else None
    # CPU_SET: unset -> no restriction; set (even to '') -> exactly that list (empty -> binding is skipped)
    return Controls(auto, nccl, bind == "1", gpu_set == "auto", environ.get(cpu_var), caller)


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace", newline="\n")  # type: ignore[attr-defined]
        except Exception:
            pass
    ap = argparse.ArgumentParser(description="Choose GPU order, P2P/NCCL settings and optional NUMA binding "
                                             "for the 4-GPU serving layout.")
    ap.add_argument("--env-out", help="write the decision env file here (required unless --dry-run)")
    ap.add_argument("--json-dir", help="diagnostics directory (default $VLLM_CACHE_ROOT/topology, else "
                                       "$CACHE_ROOT/topology, else ~/.cache/flash-next/topology)")
    ap.add_argument("--dry-run", action="store_true", help="print the summary and env file; write nothing")
    ap.add_argument("--quiet", action="store_true", help="no summary line")
    args = ap.parse_args(argv)   # exits 2 on bad arguments
    try:
        if not args.dry_run and not args.env_out:
            raise UsageError("--env-out PATH is required unless --dry-run")
        ctl = read_controls(dict(os.environ))
        raw_timeout = os.environ.get("TOPO_TIMEOUT", "").strip() or "20"
        try:
            timeout = float(raw_timeout)
            if not (math.isfinite(timeout) and 0 < timeout <= threading.TIMEOUT_MAX):
                raise ValueError
        except ValueError:
            raise UsageError(f"TOPO_TIMEOUT must be a finite number of seconds above 0 and at most "
                             f"{threading.TIMEOUT_MAX:g} (got {raw_timeout!r})") from None
    except UsageError as e:
        print(f"topo_select: {e}", file=sys.stderr)
        return EXIT_USAGE

    fixture = os.environ.get("TOPO_FIXTURE")
    loader = (lambda: read_facts_fixture(fixture)) if fixture else read_facts_nvml

    result: dict[str, Any] = {}

    def work() -> None:
        topo = None
        try:
            result["value"] = decide(loader, ctl)
            topo = result["value"][1]
        except BaseException as e:  # handed to the main thread
            result["error"] = e
        finally:
            if isinstance(topo, NvmlTopology):
                topo.close()

    try:
        t = threading.Thread(target=work, name="topo-discovery", daemon=True)
        t.start()
        t.join(timeout)
        alive = t.is_alive()
    except Exception as e:
        print(f"topo: internal error: discovery thread failed: {type(e).__name__}: {e}", file=sys.stderr)
        return EXIT_INTERNAL
    if alive:
        print(f"topo: discovery timed out after {raw_timeout}s", file=sys.stderr)
        sys.stderr.flush()
        os._exit(EXIT_TIMEOUT)   # a hung NVML call cannot be interrupted; leave without waiting for it
    if "error" in result:
        e = result["error"]
        if isinstance(e, StrictRefusal):
            print(str(e), file=sys.stderr)
            return EXIT_STRICT
        print(f"topo: internal error: {type(e).__name__}: {e}", file=sys.stderr)
        return EXIT_INTERNAL
    dec, _ = result["value"]
    try:
        write_outputs(dec, ctl, args)
    except Exception as e:
        print(f"topo: internal error while writing outputs: {type(e).__name__}: {e}", file=sys.stderr)
        return EXIT_INTERNAL
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
