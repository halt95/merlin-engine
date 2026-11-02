# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Read-only GPU unlock detection for ``merlin doctor`` (GeForce Ampere and Ada cards).

Per card: ``on``, ``off`` or ``unknown (why)``, plus whether the gpu_unlock kernel module is loaded. Informational only:
it never fails doctor, never raises and never writes anything. On/off come from three 32-bit BAR0 feature registers,
read through a read-only mapping of ``/sys/bus/pci/devices/<bdf>/resource0`` (O_RDONLY | O_SYNC, PROT_READ |
MAP_SHARED, one page, one aligned load per register, then munmap and close) in a short-lived child process, so a bus
fault cannot take doctor down. An unlocked card reads SS0 = 0x88888888 and SS1 = 0x8 (the values the module writes);
FR1 is a speed readout, reported but not gating. Registers are read only on bare metal, as root, from a supported
GeForce Ampere or Ada card bound to nvidia (the unlock finishes before nvidia can bind). The module state and the card
state are independent: a card stays unlocked until its next power-off even after the module is unloaded. The device
table, offsets and values are facts from the module's published device and register tables; this file contains no
code from the module.
"""

from __future__ import annotations

import mmap
import os
import re
import signal
import subprocess
import sys
from collections.abc import Callable
from typing import Any

MODULE = "gpu_unlock"
NVIDIA_VENDOR = 0x10DE
# The module's GeForce Ampere and Ada device ids, by chip family. Turing uses other register offsets and is not listed.
FAMILIES = {
    "GA102": (0x2200, 0x2203, 0x2204, 0x2205, 0x2206, 0x2207, 0x2208, 0x220A, 0x220D, 0x2216, 0x222B, 0x222F),
    "GA103": (0x2414,),
    "GA104": (0x2480, 0x2482, 0x2483, 0x2484, 0x2486, 0x2487, 0x2488, 0x2489, 0x248A, 0x248C, 0x248D, 0x248E, 0x249C,
              0x249D, 0x249F, 0x24A0, 0x24A4, 0x24AC, 0x24AD, 0x24AF, 0x24BF, 0x24C0, 0x24C7, 0x24C8, 0x24C9, 0x24DC,
              0x24DD, 0x24DF, 0x24E0),
    "GA106": (0x2501, 0x2503, 0x2504, 0x2505, 0x2507, 0x2508, 0x2509, 0x252F, 0x2544),
    "AD102": (0x2684, 0x2685, 0x2689),
    "AD103": (0x2702, 0x2704, 0x2705, 0x2709, 0x2757),
    "AD104": (0x2782, 0x2783, 0x2786),
    "AD106": (0x2803, 0x2805),
    "AD107": (0x2882, 0x28A0, 0x28E0, 0x28E1),
}
FAMILY_OF = {dev: fam for fam, devs in FAMILIES.items() for dev in devs}
REGS = (("FR1", 0x00823818), ("SS0", 0x0082381C), ("SS1", 0x00823820))
UNLOCKED = {"SS0": 0x88888888, "SS1": 0x00000008}  # FR1 (a speed readout) does not gate the result
TIMEOUT_S = 5.0
O_SYNC = getattr(os, "O_SYNC", 0o4010000)  # Linux values; the fallbacks only let CPU tests import this elsewhere
PROT_READ = getattr(mmap, "PROT_READ", 1)
MAP_SHARED = getattr(mmap, "MAP_SHARED", 1)
OFF_HINT = "the unlock module is loaded; check `journalctl -k -b | grep gpu_unlock`"
_BDF_RE = re.compile(r"[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]")
_CONTAINERS = {"lxc", "lxc-libvirt", "docker", "podman", "openvz", "systemd-nspawn", "rkt", "proot", "pouch",
               "container-other"}
_WHERE_WHY = {"container": "inside a container; check on the host", "vm": "inside a VM; check on the host",
              "wsl": "inside WSL2; check on the host", "unknown": "cannot confirm bare metal"}
# -I: no PYTHON* environment, user site or script directory; -S: no site hooks. Only this file and the stdlib load.
_CHILD = "import sys; sys.path.insert(0, sys.argv[1]); import gpu_unlock; sys.exit(gpu_unlock.child_main(sys.argv[2]))"
_run = subprocess.run
Reader = Callable[[str], "str | None"]


def _libc() -> Any:
    import ctypes

    libc = ctypes.CDLL(None, use_errno=True)
    libc.mmap.restype = ctypes.c_void_p
    libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
    libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    return libc


def _load32(addr: int) -> int:  # one aligned 32-bit load
    import ctypes

    return ctypes.c_uint32.from_address(addr).value


def read_bar0(bdf: str, offsets: list[int]) -> dict[int, int]:
    """{offset: value} through a read-only mapping of BAR0 (child side). Never writes; OSError when refused."""
    if not _BDF_RE.fullmatch(bdf):
        raise OSError(f"not a PCI address: {bdf!r}")
    page = min(offsets) & ~(mmap.PAGESIZE - 1)
    length = -(-(max(offsets) + 4 - page) // mmap.PAGESIZE) * mmap.PAGESIZE
    libc = _libc()
    fd = os.open(f"/sys/bus/pci/devices/{bdf}/resource0", os.O_RDONLY | O_SYNC)
    try:
        addr = libc.mmap(None, length, PROT_READ, MAP_SHARED, fd, page)
        if addr is None or addr in (-1, 2**64 - 1, 2**32 - 1):  # MAP_FAILED
            raise OSError("mmap of resource0 failed")
        try:
            return {o: _load32(addr + o - page) for o in offsets}
        finally:
            libc.munmap(addr, length)
    finally:
        os.close(fd)


def child_main(bdf: str) -> int:
    """Child entry: ``REGS <FR1> <SS0> <SS1>`` and exit 0, or ``REFUSED <reason>`` and exit 3."""
    try:
        raw = read_bar0(bdf, [o for _, o in REGS])
    except OSError as e:
        print(f"REFUSED {e.strerror or e}", flush=True)
        return 3
    print("REGS " + " ".join(f"{raw[o]:#010x}" for _, o in REGS), flush=True)
    return 0


def read_in_child(bdf: str, run: Callable[..., Any]) -> tuple[dict[str, int] | None, str | None]:
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        r = run([sys.executable, "-I", "-S", "-c", _CHILD, here, bdf], capture_output=True, text=True,
                timeout=TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return None, "register read failed: timed out"
    except (OSError, subprocess.SubprocessError) as e:
        return None, f"register read failed: {e}"
    lines = (r.stdout or "").strip().splitlines()
    if r.returncode < 0:
        try:
            return None, f"register read failed: killed by {signal.Signals(-r.returncode).name}"
        except ValueError:
            return None, f"register read failed: killed by signal {-r.returncode}"
    if r.returncode == 3 and len(lines) == 1 and lines[0].startswith("REFUSED "):
        return None, f"register access refused: {lines[0][8:].strip()}"
    if r.returncode != 0:
        return None, f"register read failed: exit code {r.returncode}"
    m = re.fullmatch(r"REGS (0x[0-9a-f]{8}) (0x[0-9a-f]{8}) (0x[0-9a-f]{8})", lines[0]) if len(lines) == 1 else None
    if not m:
        return None, "register read failed: malformed output"
    return {name: int(m[i + 1], 16) for i, (name, _) in enumerate(REGS)}, None


def where(virt: str | None, run: Callable[..., Any]) -> str:
    """bare / container / vm / wsl / unknown. Bare metal = systemd-detect-virt printing exactly "none" with exit 1."""
    if virt in ("container", "vm", "wsl"):
        return virt
    try:
        r = run(["systemd-detect-virt"], capture_output=True, text=True, timeout=TIMEOUT_S)
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    out = (r.stdout or "").strip().lower()
    if r.returncode == 1 and out == "none":
        return "bare"
    if r.returncode == 0 and out != "none" and re.fullmatch(r"[a-z0-9_-]+", out):
        return "wsl" if out == "wsl" else "container" if out in _CONTAINERS else "vm"
    return "unknown"


def module_state(read: Reader, system: str) -> tuple[str, str | None]:
    if system != "linux":
        return "unknown", "not a Linux host"
    mods = read("/proc/modules")
    if mods is not None:
        loaded = any(line.split(" ", 1)[0] == MODULE for line in mods.splitlines())
        return ("loaded" if loaded else "not loaded"), None
    state = read(f"/sys/module/{MODULE}/initstate")
    if state is not None:
        return ("loaded" if state.strip() == "live" else "not loaded"), None
    return "unknown", "cannot read /proc/modules"


def classify(vals: dict[str, int]) -> tuple[str, str | None]:
    text = " ".join(f"{k}={v:#010x}" for k, v in vals.items())
    if any((v & 0xFFFF0000) == 0xBADF0000 or v == 0xFFFFFFFF for v in vals.values()):
        return "unknown", f"register read returned an error pattern: {text}"
    if all(vals[k] == v for k, v in UNLOCKED.items()):
        return "on", None
    if vals["SS0"] == UNLOCKED["SS0"]:
        return "unknown", f"registers partly unlocked: {text}"
    return "off", None


def failed(gpus: list[tuple[int, str | None]], err: BaseException) -> dict[str, Any]:
    why = f"internal error: {err}"
    return {"module": "unknown", "module_name": MODULE, "module_why": why,
            "cards": [{"index": i, "bus_id": b, "pci_id": None, "family": None, "state": "unknown", "why": why,
                       "registers": None} for i, b in gpus]}


def probe(gpus: list[tuple[int, str | None]], virt: str | None, read: Reader, system: str | None = None,
          euid: Callable[[], int | None] | None = None, run: Callable[..., Any] | None = None) -> dict[str, Any]:
    """GPU unlock state per card plus the module state. ``gpus`` = [(index, sysfs bus id or None)]. Never raises."""
    try:
        system = system if system is not None else ("linux" if sys.platform.startswith("linux") else sys.platform)
        euid = euid or (lambda: getattr(os, "geteuid", lambda: None)())
        run = run or _run
        try:
            mod, mod_why = module_state(read, system)
        except Exception as e:  # noqa: BLE001
            mod, mod_why = "unknown", f"internal error: {e}"
        res: dict[str, Any] = {"module": mod, "module_name": MODULE, "module_why": mod_why, "cards": []}
        site: list[str] = []
        for index, bdf in gpus:
            card: dict[str, Any] = {"index": index, "bus_id": bdf, "pci_id": None, "family": None, "registers": None}
            try:
                card["state"], card["why"] = _card(card, bdf, virt, system, read, euid, run, site)
            except Exception as e:  # noqa: BLE001
                card["state"], card["why"] = "unknown", f"internal error: {e}"
            if card["state"] == "off" and mod == "loaded":
                card["why"] = OFF_HINT
            res["cards"].append(card)
        return res
    except Exception as e:  # noqa: BLE001 - informational: doctor must never fail here
        return failed(gpus, e)


def _card(card, bdf, virt, system, read, euid, run, site):
    if system != "linux":
        return "unknown", "not a Linux host"
    if not bdf or not _BDF_RE.fullmatch(bdf):
        return "unknown", "no PCI bus id from nvidia-smi"
    uevent = read(f"/sys/bus/pci/devices/{bdf}/uevent") or ""
    m = re.search(r"(?m)^PCI_ID=([0-9A-Fa-f]{4}):([0-9A-Fa-f]{4})\s*$", uevent)
    if not m:
        return "unknown", "cannot read the PCI device id"
    vendor, device = int(m[1], 16), int(m[2], 16)
    card["pci_id"] = f"{vendor:04x}:{device:04x}"
    if vendor != NVIDIA_VENDOR or device not in FAMILY_OF:
        return "unknown", f"{card['pci_id']} is not a supported GeForce Ampere or Ada card"
    card["family"] = FAMILY_OF[device]
    if not site:
        site.append(where(virt, run))
    if site[0] != "bare":
        return "unknown", _WHERE_WHY[site[0]]
    if euid() != 0:
        return "unknown", "needs root to read the card's registers"
    drv = re.search(r"(?m)^DRIVER=(\S+)\s*$", uevent)
    if not drv or drv[1] != "nvidia":
        return "unknown", f"driver is {drv[1] if drv else 'none'}, not nvidia; registers not read"
    bar = re.match(r"\s*0x([0-9a-fA-F]+)\s+0x([0-9a-fA-F]+)", read(f"/sys/bus/pci/devices/{bdf}/resource") or "")
    size = int(bar[2], 16) - int(bar[1], 16) + 1 if bar and int(bar[2], 16) > int(bar[1], 16) else None
    if size is not None and size < max(o for _, o in REGS) + 4:
        return "unknown", f"BAR0 is {size:#x} bytes, too small for the feature registers; registers not read"
    vals, why = read_in_child(bdf, run)
    if vals is None:
        return "unknown", why
    card["registers"] = {k: f"{v:#010x}" for k, v in vals.items()}
    return classify(vals)


def card_line(c: dict[str, Any]) -> str:
    return f"GPU{c['index']} ({c.get('bus_id') or 'bus unknown'}): GPU unlock: {c['state']}" + (
        f" ({c['why']})" if c.get("why") else "")


def module_line(res: dict[str, Any]) -> str:
    return f"GPU unlock module {MODULE}: {res['module']}" + (f" ({res['module_why']})" if res.get("module_why") else "")


def text_lines(res: dict[str, Any] | None) -> list[str]:
    return [f"  {card_line(c)}" for c in res["cards"]] + [f"  {module_line(res)}"] if res else []
