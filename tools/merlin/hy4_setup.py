# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Hy4 Flash in ``./setup.sh``: the row for the cards found, the host checks, the
verified pack download and the hints.

Hy4 Flash is served by ``tools/merlin/launch.py --profile <row>`` (``merlin serve`` does
not list it): setup picks the row from the card count (4 x 24 GB -> hy4flash-quad-3090,
8 x 24 GB -> hy4flash-octo-3090, refused otherwise), checks host RAM, the memlock limit
and the disk for the Engram variant asked (``--engram fp8``, the default, or ``bf16``),
downloads the W4A16 pack of that variant with every LFS file checked against
the sha256 and size the hub itself states (``X-Linked-Etag`` / ``X-Linked-Size`` on a
HEAD that does not follow the redirect; the CDN's own ETag after the redirect is never
used), and prints the launcher's command, taken from the launcher's plan
(``launch.build``). Standard library only, like setup_cli, which calls it."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import launch as L
import preflight as pf
import registry as R

MODEL = "hy4flash"
TITLE = "Hy4 Flash W4A16 + FP8 or BF16 Engram (4 or 8 x 24 GB, P2P; 4-card testing, 8-card untested)"
# Two packs: FP8 Engram tables (the default) and BF16
# Engram tables (./setup.sh --model hy4flash --engram bf16). Both publish in one
# Hugging Face repository, PACK_REPO: the main revision is FP8, the PACK_BRANCH_BF16
# branch is BF16. Each slot pins the commit setup downloads and verifies.
PACK_REPO = "halt95/Hy4-flash-W4A16-Merlin"
PACK_BRANCH_BF16 = "engram-bf16"
ENGRAMS = tuple(L.HY4_ENGRAM_GIB)  # ("fp8", "bf16")
ENGRAM_DEFAULT = L.HY4_ENGRAM_DEFAULT
# The FP8 pack's repository and the revision setup pins (a commit on main).
HF_REPO: str | None = PACK_REPO
HF_REVISION: str | None = "45a685b169f418b0f8e22867fc631ab68d107610"
# The BF16 pack's (the same repository, a commit on the engram-bf16 branch).
HF_REPO_BF16: str | None = PACK_REPO
HF_REVISION_BF16: str | None = "fb3973c7bec167e1fe79e1ebf7a07d8feb35b885"
HF_BASE = "https://huggingface.co"  # HF_ENDPOINT overrides it, as for the hf tool
USER_AGENT = "merlin-setup/1.0"
# the whole pack (weights + Engram tables + small files), decimal bytes. BF16 holds the
# same Engram rows at 192 B instead of 98 B: 1.2672e11 - 6.468e10 = 6.204e10 B more.
PACK_BYTES = 152_600_000_000
PACK_BYTES_BF16 = 214_640_000_000
# Host RAM. The Engram tables are pinned in host RAM; the engine processes take about
# 18 GiB beside them; the 4-card FP8 boots needed this much available (MemAvailable)
# before the start, and BF16 adds its tables' difference from FP8
# (launch.hy4_ram_need_gib). The KV RAM cache would come on top; it is refused on Hy4
# Flash for now, so the size terms below stay at 0.
ENGRAM_FP8_GIB = L.HY4_ENGRAM_GIB["fp8"]
ENGRAM_BF16_GIB = L.HY4_ENGRAM_GIB["bf16"]
ENGINE_PROCS_GIB = L.HY4_ENGINE_PROCS_GIB
RAM_AVAIL_MIN_GIB = L.HY4_RAM_AVAIL_MIN_GIB


def engram_gib(engram: str = ENGRAM_DEFAULT) -> float:
    return L.HY4_ENGRAM_GIB[engram]


def pack_bytes(engram: str = ENGRAM_DEFAULT) -> int:
    return PACK_BYTES_BF16 if engram == "bf16" else PACK_BYTES


def hub_source(engram: str = ENGRAM_DEFAULT) -> tuple[str | None, str | None]:
    """(repository, revision) of the ``engram`` pack: None while unpublished."""
    if engram == "bf16":
        return HF_REPO_BF16, HF_REVISION_BF16
    return HF_REPO, HF_REVISION
# A file the hub gives no LFS sha256 for is accepted only when it is small (config,
# tokenizer, chat template) and not a weight shard; anything else is refused unverified.
SMALL_FILE_MAX = 64 << 20
SHARD_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".gguf")
REASONING_EFFORTS = ("no_think", "high")  # the Hy4 Flash template takes only these
CHUNK = 16 << 20


class HubError(Exception):
    """The hub did not give what a verified download needs; nothing unverified is
    kept."""


# --------------------------------------------------------------------------------------
# Rows
# --------------------------------------------------------------------------------------
def registry() -> dict[str, Any]:
    return R.load_registry(MODEL)


# The row setup starts for each card count, named
# explicitly: two rows share 4 cards (the TP4 DIAG row is launch.py --profile only)
INSTALLER_ROWS = {4: "hy4flash-quad-3090", 8: "hy4flash-octo-3090"}


def rows(reg: dict[str, Any] | None = None) -> dict[int, dict[str, Any]]:
    """Card count -> the installer's row for it (INSTALLER_ROWS), read from the profile
    file. A named row that is missing, or matches another card count, is refused."""
    reg = reg or registry()
    by_id = {p["id"]: p for p in reg["profiles"]}
    out: dict[int, dict[str, Any]] = {}
    for n, pid in INSTALLER_ROWS.items():
        p = by_id.get(pid)
        if p is None:
            raise ValueError(f"hy4flash installer row {pid} is not in the profile file")
        if int(p["match"]["cards"]) != n:
            raise ValueError(f"hy4flash installer row {pid} matches {p['match']['cards']} cards, not {n}")
        out[n] = p
    if len({p["id"] for p in out.values()}) != len(out):
        raise ValueError("hy4flash installer rows repeat a profile")
    return out


def supported_text(reg: dict[str, Any] | None = None) -> str:
    parts = []
    for n, p in sorted(rows(reg).items()):
        m = p["match"]
        parts.append(
            f"{n} x {m['vram_gib']} GB {'/'.join(m['arch'])} (RTX 3090 class), P2P "
            f"{m['p2p']} -> {p['id']}"
        )
    return "; ".join(parts)


@dataclass(frozen=True)
class RowChoice:
    profile: dict[str, Any] | None
    why: str

    @property
    def id(self) -> str | None:
        return self.profile["id"] if self.profile else None


def select_row(inv: pf.Inventory, reg: dict[str, Any] | None = None) -> RowChoice:
    """The row for every visible card, or None with the reason. Exactly 4 or 8 cards of
    the rows' arch and 24 GiB VRAM class with P2P on among them; any other count, size,
    arch or P2P state is refused, naming what is supported."""
    reg = reg or registry()
    table = rows(reg)
    ok = f"Hy4 Flash runs on {supported_text(reg)}"
    gpus = inv.gpus
    if not gpus:
        return RowChoice(None, f"no NVIDIA GPU found. {ok}")
    n = len(gpus)
    sample = next(iter(table.values()))["match"]
    off = [
        g
        for g in gpus
        if g.arch not in sample["arch"]
        or R.vram_class(pf._gib(g.vram_mib)) != sample["vram_gib"]
    ]
    if off:
        seen = ", ".join(
            sorted(
                {
                    f"{g.name} ({pf._gib(g.vram_mib):.0f} GB, "
                    f"{g.arch or 'arch unknown'})"
                    for g in off
                }
            )
        )
        return RowChoice(None, f"{seen} is not a supported card. {ok}")
    if n not in table:
        return RowChoice(
            None,
            f"{n} x {gpus[0].name}: the rows take exactly "
            f"{' or '.join(map(str, sorted(table)))} cards. {ok}",
        )
    p = table[n]
    p2p = inv.p2p_among([g.index for g in gpus])
    if p2p != p["match"].get("p2p", "on"):
        return RowChoice(
            None,
            f"P2P is {p2p} among the {n} cards; the hy4flash rows are P2P-on only (no "
            f"no-P2P row yet). {ok}",
        )
    untested = UNTESTED_ON_HARDWARE.get(p["id"])
    return RowChoice(
        p,
        f"{n} x {gpus[0].name}, P2P on: {p['id']} [{R.label_text(reg, p)}]"
        + (f" - {untested}" if untested else ""),
    )


# Rows that ship without a boot on their hardware:
# the selection line says so
UNTESTED_ON_HARDWARE = {"hy4flash-octo-3090": "untested on hardware (pool is an estimate)"}


def ranks(profile: dict[str, Any]) -> int:
    lay = profile["layout"]
    return int(lay["tp"]) * int(lay.get("pp", 1))


# --------------------------------------------------------------------------------------
# Host checks: (level, code, text, fix) tuples, setup_cli wraps them as Problems
# --------------------------------------------------------------------------------------
Check = tuple[str, str, str, str]


def ram_need_gib(tier_total_gib: int = 0, engram: str = ENGRAM_DEFAULT) -> float:
    """Host RAM the server needs available: the 4-card FP8 boot's measured need (which
    covers the pinned Engram FP8 tables and the engine processes), moved by the
    ``engram`` variant's table difference, plus the KV RAM cache when asked."""
    return L.hy4_ram_need_gib(engram, tier_total_gib)


def ram_check(
    have: float | None,
    what: str,
    tier_total_gib: int,
    cards: int,
    engram: str = ENGRAM_DEFAULT,
) -> Check | None:
    need = ram_need_gib(tier_total_gib, engram)
    parts = (
        f"Engram {engram.upper()} tables ~{engram_gib(engram):.0f} GiB pinned + engine "
        f"processes ~{ENGINE_PROCS_GIB:.0f} GiB; "
        f"the 4-card boots needed {RAM_AVAIL_MIN_GIB:.0f} GiB available"
    )
    if engram != ENGRAM_DEFAULT:
        parts += (
            f" with the FP8 tables, + {engram_gib(engram) - ENGRAM_FP8_GIB:.0f} GiB for "
            f"the {engram.upper()} ones"
        )
    if tier_total_gib:
        parts += f"; + the KV RAM cache {tier_total_gib} GiB"
    if have is None:
        return (
            "WARN",
            "ram",
            f"host RAM unknown; Hy4 Flash needs ~{need:.0f} GiB available ({parts})",
            "",
        )
    if have < need:
        fix = "add RAM (or raise the container's memory limit)"
        if tier_total_gib:
            fix = "--kv-ram-cache 0 (the KV RAM cache off), or " + fix
        return (
            "FAIL",
            "ram",
            f"{what} {have:.0f} GiB is below the ~{need:.0f} GiB Hy4 Flash needs "
            f"({parts}): the load "
            "would be OOM-killed",
            fix,
        )
    if cards != 4:
        return (
            "WARN",
            "ram",
            f"{what} {have:.0f} GiB holds the ~{need:.0f} GiB the 4-card boots needed; "
            f"the {cards}-card "
            "row's need is unmeasured (its engine processes are more)",
            "",
        )
    return None


def memlock_limit(
    inv: pf.Inventory, read: Callable[[str], str | None] | None = None
) -> tuple[int | None, bool]:
    """(bytes, known): the pinned-memory limit the server will run under; None bytes =
    unlimited (or CAP_IPC_LOCK). A simulated host gives its own (unlimited by default);
    a probed host is read from this process, whose limit the server it starts
    inherits."""
    if inv.source != "probe":
        b = inv.host.memlock_bytes
        if b is None:
            return None, False
        return (None if b == pf.MEMLOCK_UNLIMITED else b), True
    try:
        import resource
    except ImportError:  # no such limit on this platform
        return None, False
    soft, _ = resource.getrlimit(resource.RLIMIT_MEMLOCK)
    if soft == resource.RLIM_INFINITY:
        return None, True
    status = (read or pf._read)("/proc/self/status") or ""
    m = re.search(r"(?m)^CapEff:\s*([0-9a-fA-F]+)", status)
    if m and (int(m.group(1), 16) >> 14) & 1:  # CAP_IPC_LOCK
        return None, True
    return int(soft), True


MEMLOCK_FIX = (
    "raise the memlock limit for the server: ulimit -l unlimited in this shell; "
    "systemd LimitMEMLOCK=infinity; "
    "docker --ulimit memlock=-1; LXC lxc.prlimit.memlock = unlimited"
)


def memlock_check(
    limit: int | None,
    known: bool,
    tier_per_rank_gib: int,
    engram: str = ENGRAM_DEFAULT,
) -> Check | None:
    """The KV RAM cache arena is locked per rank and the engine refuses one above the
    limit (FAIL); the Engram tables are pinned too, so a limit below them is a warning
    (how much of them one process locks is unmeasured)."""
    if not known:
        return ("WARN", "memlock", "the pinned-memory limit is unknown", MEMLOCK_FIX)
    if limit is None:
        return None
    gib = limit / pf.GIB
    if tier_per_rank_gib and limit < tier_per_rank_gib * pf.GIB:
        return (
            "FAIL",
            "memlock",
            f"the memlock limit is {gib:.2f} GiB; the KV RAM cache locks "
            f"{tier_per_rank_gib} GiB per "
            "rank and the engine refuses it",
            MEMLOCK_FIX + "; or --kv-ram-cache 0",
        )
    if limit < engram_gib(engram) * pf.GIB:
        return (
            "WARN",
            "memlock",
            f"the memlock limit is {gib:.2f} GiB; the Engram {engram.upper()} tables "
            f"(~{engram_gib(engram):.0f} GiB) "
            "are pinned in host RAM",
            MEMLOCK_FIX,
        )
    return None


# --------------------------------------------------------------------------------------
# CUDA: the NVRTC the humming JIT uses
# --------------------------------------------------------------------------------------
def lock_nvrtc(lock: Path) -> str | None:
    """The nvidia-cuda-nvrtc pin of the release lock (``13.0.88``), or None."""
    try:
        m = re.search(
            r"(?m)^nvidia-cuda-nvrtc==(\d+\.\d+[\w.]*)",
            lock.read_text(encoding="utf-8"),
        )
    except OSError:
        return None
    return m.group(1) if m else None


def system_cuda(
    read: Callable[[str], str | None] | None = None, root: str = "/usr/local/cuda"
) -> str | None:
    """The system CUDA's version (``13.3.0``) from its version.json, or None."""
    raw = (read or pf._read)(f"{root}/version.json")
    try:
        return str(json.loads(raw)["cuda"]["version"]) if raw else None
    except (ValueError, KeyError, TypeError):
        return None


def _major_minor(v: str | None) -> str | None:
    m = re.match(r"(\d+)\.(\d+)", v or "")
    return f"{m.group(1)}.{m.group(2)}" if m else None


def nvrtc_text(
    lock: Path,
    venv_python: Path | None,
    run: Callable[[list[str]], str | None] | None,
    probe: bool,
    read: Callable[[str], str | None] | None = None,
) -> str:
    """Which NVRTC the humming JIT compiles with: torch's own CUDA major.minor
    (vllm/merlin/humming_cuda.py drops a system CUDA of the same major and another
    minor). From the installed venv's torch when there is one on a probed host, else
    from the release lock's nvidia-cuda-nvrtc pin."""
    torch_cuda = src = None
    if probe and venv_python is not None and venv_python.exists() and run is not None:
        torch_cuda = _major_minor(
            run(
                [
                    str(venv_python),
                    "-c",
                    "import torch; print(torch.version.cuda or '')",
                ]
            )
        )
        src = "the installed venv's torch.version.cuda" if torch_cuda else None
    if not torch_cuda:
        pin = lock_nvrtc(lock)
        torch_cuda = _major_minor(pin)
        src = (
            f"release/{lock.name} nvidia-cuda-nvrtc=={pin}, torch's "
            f"cu{torch_cuda.replace('.', '')} runtime"
            if pin
            else None
        )
    if not torch_cuda:
        return "humming JIT: torch's own NVRTC (its CUDA version could not be read)"
    text = f"humming JIT: NVRTC {torch_cuda} (torch's CUDA, from {src})"
    sys_v = _major_minor(system_cuda(read)) if probe else None
    if (
        sys_v
        and sys_v != torch_cuda
        and sys_v.split(".")[0] == torch_cuda.split(".")[0]
    ):
        text += (
            f"; the system CUDA {sys_v} at /usr/local/cuda is ignored (another minor)"
        )
    elif sys_v and sys_v == torch_cuda:
        text += f"; the system CUDA {sys_v} matches it"
    return text


# --------------------------------------------------------------------------------------
# Hints
# --------------------------------------------------------------------------------------
def serve_hints(dfly_on: bool, engram: str = ENGRAM_DEFAULT) -> list[str]:
    default = L.HY4_REASONING_EFFORT
    effort = (
        f"thinking control: send reasoning_effort {' / '.join(REASONING_EFFORTS)} per "
        "request "
        f'(chat_template_kwargs {{"reasoning_effort": "high"}}); the served default is '
        f"{default}"
    )
    dfly = (
        "DFly2 speculative decoding is on by default (SPEC_K sets K); DFLY=0 in the "
        "server's environment turns it off"
        if dfly_on
        else "DFly2 speculative decoding is on by default but off here: the KV RAM "
        "cache keeps it off; DFLY=0 turns it off "
        "explicitly"
    )
    # The other pack, named
    if engram == ENGRAM_DEFAULT:
        pack = (
            f"Engram tables: FP8 (~{ENGRAM_FP8_GIB:.0f} GiB pinned, the default); the "
            f"BF16 pack is --engram bf16 (~{ENGRAM_BF16_GIB:.0f} GiB pinned, "
            f"~{ram_need_gib(0, 'bf16'):.0f} GiB host RAM available)"
        )
    else:
        pack = (
            f"Engram tables: {engram.upper()} (--engram {engram}, "
            f"~{engram_gib(engram):.0f} GiB pinned; the launcher runs with "
            f"{L.HY4_ENGRAM_ENV}={engram} and refuses a pack whose config.json says "
            "otherwise); not benched yet"
        )
    return [effort, dfly, pack]


# --------------------------------------------------------------------------------------
# The launcher's plan
# --------------------------------------------------------------------------------------
def launch_env(
    host: str,
    port: int,
    api_key: str | None,
    tier_per_rank_gib: int,
    engram: str = ENGRAM_DEFAULT,
) -> dict[str, str]:
    """What setup exports for launch.py: the bind, the Engram variant and the KV RAM
    cache. The cards, KV pin, chunk and every expert placement knob are the row's (the
    launcher resolves them)."""
    env = {"HOST": host, "PORT": str(port), L.HY4_ENGRAM_ENV: engram}
    if api_key:
        env["VLLM_API_KEY"] = api_key
    if tier_per_rank_gib:
        env[L.FN_KV_HOST_TIER] = str(tier_per_rank_gib)
    return env


def launcher_argv(repo: Path, profile_id: str, ckpt: Path) -> list[str]:
    return [
        str(repo / "venv" / "bin" / "python"),
        str(repo / "tools" / "merlin" / "launch.py"),
        "--profile",
        profile_id,
        "--",
        str(ckpt),
    ]


# The row's shape knobs: setup starts the row as the profile has it, so a stale export
# of one never reaches the plan it prints or the launcher it starts (an operator who
# wants one runs launch.py by hand).
ROW_KNOBS = (
    "DFLY",
    "SPEC_K",
    "SEQS",
    "MAXLEN",
    "KV_BYTES",
    "CHUNK",
    "GPUS",
    "CUDA_VISIBLE_DEVICES",
    "PP_PARTITION",
    "P2P",
    "BLOCK_SIZE",
    "EAGER",
    L.FN_KV_HOST_TIER,
    L.SWARM_ENV,
    L.HY4_ENGRAM_ENV,  # setup's --engram replaces a stale export
)


def launcher_environ(
    setup_env: dict[str, str], environ: dict[str, str] | None = None
) -> dict[str, str]:
    """The environment launch.py gets: the caller's without the row knobs, then setup's
    own (``launch_env``)."""
    base = dict(os.environ if environ is None else environ)
    for k in ROW_KNOBS:
        base.pop(k, None)
    return {**base, **setup_env}


def launcher_plan(
    repo: Path, profile_id: str, ckpt: Path, environ: dict[str, str]
) -> L.Plan:
    """The launcher's own plan for the row (pure: nothing runs). Raises L.LaunchError
    with the launcher's refusal."""
    return L.build(MODEL, profile_id, str(ckpt), dict(environ), repo=repo)


def dfly_on(plan: L.Plan) -> bool:
    return any(a == "--speculative-config" for a in plan.args)


# --------------------------------------------------------------------------------------
# The verified download
# --------------------------------------------------------------------------------------
def hub_base(environ: dict[str, str] | None = None) -> str:
    return (
        (environ if environ is not None else os.environ).get("HF_ENDPOINT") or HF_BASE
    ).rstrip("/")


def _headers(extra: dict[str, str] | None = None) -> dict[str, str]:
    h = {"User-Agent": USER_AGENT}
    tok = os.environ.get("HF_TOKEN")
    if tok:
        h["Authorization"] = f"Bearer {tok}"
    return {**h, **(extra or {})}


def resolve_url(base: str, repo: str, revision: str | None, name: str) -> str:
    return f"{base}/{repo}/resolve/{revision or 'main'}/{urllib.parse.quote(name)}"


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # surface the 3xx so the hub's own headers (X-Linked-*) are read, not the CDN's
        return None


def head_noredirect(url: str, timeout: float = 60) -> Any:
    """The headers of the hub's own answer to HEAD ``url``, without following its
    redirect to the CDN."""
    opener = urllib.request.build_opener(_NoRedirect)
    try:
        with opener.open(
            urllib.request.Request(url, headers=_headers(), method="HEAD"),
            timeout=timeout,
        ) as r:
            return r.headers
    except urllib.error.HTTPError as e:
        if 300 <= e.code < 400:
            return e.headers
        raise


def safe_name(name: Any) -> bool:
    """A repository path that stays inside the destination folder."""
    if (
        not isinstance(name, str)
        or not name
        or "\\" in name
        or "\x00" in name
        or name.startswith("/")
    ):
        return False
    parts = name.split("/")
    return all(p not in ("", ".", "..") for p in parts)


def list_files(
    base: str, repo: str, revision: str | None, timeout: float = 60
) -> list[str]:
    url = f"{base}/api/models/{repo}" + (f"/revision/{revision}" if revision else "")
    with urllib.request.urlopen(
        urllib.request.Request(url, headers=_headers()), timeout=timeout
    ) as r:
        data = json.loads(r.read())
    names = [
        s.get("rfilename") for s in data.get("siblings", []) if isinstance(s, dict)
    ]
    bad = [n for n in names if not safe_name(n)]
    if bad:
        raise HubError(
            f"{repo}: the hub lists a file name outside the folder ({bad[0]!r}); "
            "nothing downloaded"
        )
    if not names:
        raise HubError(f"{repo}: the hub lists no files")
    return sorted(names)


@dataclass(frozen=True)
class Remote:
    name: str
    sha256: str | None  # the hub's LFS sha256; None for a small non-LFS file
    size: int | None


def is_shard(name: str) -> bool:
    return name.endswith(SHARD_SUFFIXES)


def lfs_meta(
    base: str,
    repo: str,
    revision: str | None,
    name: str,
    head: Callable[[str], Any] = head_noredirect,
) -> Remote:
    """The file's sha256 and size as the hub states them: X-Linked-Etag / X-Linked-Size
    on its no-redirect answer. A weight shard without them, or a file over
    SMALL_FILE_MAX without them, is refused (HubError)."""
    hd = head(resolve_url(base, repo, revision, name))
    et = (hd.get("X-Linked-Etag") or "").strip().removeprefix("W/").strip('"')
    if re.fullmatch(r"[0-9a-f]{64}", et):
        sz = hd.get("X-Linked-Size") or ""
        if not sz.isdigit():
            raise HubError(
                f"{name}: the hub gives an LFS sha256 but no X-Linked-Size; refusing "
                "an unverified file"
            )
        return Remote(name, et, int(sz))
    if is_shard(name):
        raise HubError(
            f"{name}: the hub gives no LFS sha256 (X-Linked-Etag) for this weight "
            "shard; refusing an "
            "unverified shard"
        )
    cl = hd.get("Content-Length") or ""
    size = int(cl) if cl.isdigit() and not hd.get("Location") else None
    if size is not None and size > SMALL_FILE_MAX:
        raise HubError(
            f"{name}: {size} bytes and no LFS sha256 from the hub; refusing an "
            "unverified file"
        )
    return Remote(name, None, size)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(CHUNK), b""):
            h.update(b)
    return h.hexdigest()


def download_file(
    url: str,
    dst: Path,
    want: Remote,
    say: Callable[[str], None] = print,
    tries: int = 6,
    timeout: float = 300,
    sleep: Callable[[float], None] = time.sleep,
) -> str:
    """Fetch ``url`` (redirects followed) into ``dst`` through ``dst.part`` with resume,
    hashing as it writes; kept only when the size and the sha256 match what the hub
    stated (``want``). Returns the sha256 of the file kept."""
    part = dst.with_name(dst.name + ".part")
    dst.parent.mkdir(parents=True, exist_ok=True)
    cap = want.size if want.size is not None else SMALL_FILE_MAX
    last: Exception | None = None
    for attempt in range(tries):
        try:
            have = part.stat().st_size if part.exists() else 0
            if have > cap:
                part.unlink()
                have = 0
            h = hashlib.sha256()
            if have:
                with open(part, "rb") as f:
                    for b in iter(lambda: f.read(CHUNK), b""):
                        h.update(b)
            hdr = _headers({"Range": f"bytes={have}-"} if have else None)
            with urllib.request.urlopen(
                urllib.request.Request(url, headers=hdr), timeout=timeout
            ) as r:
                if have and r.status != 206:  # the server ignored the range: start over
                    have, h = 0, hashlib.sha256()
                    part.unlink()
                with open(part, "ab") as out:
                    for b in iter(lambda: r.read(CHUNK), b""):
                        have += len(b)
                        if have > cap:
                            raise HubError(
                                f"{want.name}: more than the {cap} bytes the hub stated"
                            )
                        out.write(b)
                        h.update(b)
            got = part.stat().st_size
            if want.size is not None and got != want.size:
                raise HubError(f"{want.name}: size {got} != {want.size} from the hub")
            digest = h.hexdigest()
            if want.sha256 is not None and digest != want.sha256:
                part.unlink()
                raise HubError(
                    f"{want.name}: sha256 {digest} != {want.sha256} from the hub (LFS)"
                )
            os.replace(part, dst)
            return digest
        except (OSError, HubError) as e:
            last = e
            if isinstance(e, HubError) and "sha256" in str(e) and attempt >= 1:
                break  # the same bytes twice: the source is wrong, not the line
            if attempt + 1 < tries:
                say(f"  {want.name}: {e}; retry {attempt + 1}")
                sleep(min(60, 2**attempt))
    if part.exists() and want.size is not None and part.stat().st_size >= want.size:
        part.unlink()
    raise HubError(f"{want.name}: download failed ({last})")


def marker_path(dest: Path) -> Path:
    return dest / L.VERIFIED_MARKER


def read_marker(dest: Path) -> dict[str, Any] | None:
    try:
        m = json.loads(marker_path(dest).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return m if isinstance(m, dict) and isinstance(m.get("files"), dict) else None


def marker_current(
    dest: Path,
    repo: str | None = None,
    revision: str | None = None,
    engram: str | None = None,
) -> bool:
    """A Hy4 Flash pack this installer verified, unchanged since (every listed file at
    its recorded size and mtime), and of ``repo`` / ``revision`` / the ``engram``
    variant when given (a marker without the field is an FP8 pack, written before the
    BF16 pack existed)."""
    m = read_marker(dest)
    if not m or m.get("model") != MODEL:
        return False
    if engram is not None and m.get("engram", ENGRAM_DEFAULT) != engram:
        return False
    if (
        repo is not None
        and m.get("repo") != repo
        or revision is not None
        and m.get("revision") != revision
    ):
        return False
    for name, rec in m["files"].items():
        if not safe_name(name) or not isinstance(rec, dict):
            return False
        try:
            st = (dest / name).stat()
        except OSError:
            return False
        if st.st_size != rec.get("size") or st.st_mtime_ns != rec.get("mtime_ns"):
            return False
    return True


def folder_conflict(dest: Path) -> str | None:
    """Why ``dest`` must not receive the pack: it holds another model's verified
    checkpoint."""
    m = read_marker(dest)
    if m and m.get("model") and m["model"] != MODEL:
        return f"holds a verified {m['model']} checkpoint"
    return None


def present_bytes(dest: Path) -> int:
    """Bytes of pack files already in ``dest`` (finished files and partial ones), for
    the disk check."""
    if not dest.is_dir():
        return 0
    total = 0
    for p in dest.rglob("*"):
        try:
            if p.is_file() and p.name != L.VERIFIED_MARKER:
                total += p.stat().st_size
        except OSError:
            pass
    return total


def pull(
    dest: Path,
    repo: str,
    revision: str | None,
    base: str | None = None,
    rehash: bool = False,
    say: Callable[[str], None] = print,
    head: Callable[[str], Any] = head_noredirect,
    fetch: Callable[[str, Path, Remote], str] | None = None,
    engram: str = ENGRAM_DEFAULT,
) -> dict[str, Any]:
    """Download and verify the ``engram`` pack into ``dest``; writes the verified marker
    last. A current marker is trusted (no network) unless ``rehash``, which hashes every
    file against its recorded sha256 again."""
    base = base or hub_base()
    if marker_current(dest, repo, revision, engram) and not rehash:
        say(f"  {dest}: verified earlier ({L.VERIFIED_MARKER}); nothing to download")
        return read_marker(dest) or {}
    if rehash and marker_current(dest, repo, revision, engram):
        m = read_marker(dest) or {}
        bad = [
            n
            for n, rec in m["files"].items()
            if sha256_of(dest / n) != rec.get("sha256")
        ]
        if not bad:
            say(f"  {dest}: every file hashed again and matches")
            return m
        say(f"  {len(bad)} file(s) no longer match; fetching them again")
    fetch = fetch or (lambda url, dst, want: download_file(url, dst, want, say))
    names = list_files(base, repo, revision)
    remotes = [
        lfs_meta(base, repo, revision, n, head) for n in names
    ]  # every refusal before the first byte
    files: dict[str, Any] = {}
    marker_path(dest).unlink(missing_ok=True)
    for i, r in enumerate(remotes, 1):
        dst = dest / r.name
        digest = None
        if dst.is_file() and r.size is not None and dst.stat().st_size == r.size:
            digest = sha256_of(dst)
            if r.sha256 is not None and digest != r.sha256:
                say(
                    f"  {r.name}: present but its sha256 differs from the hub's; "
                    "fetching it again"
                )
                dst.unlink()
                digest = None
        if digest is None:
            say(
                f"  [{i}/{len(remotes)}] {r.name}"
                + (f" ({r.size} bytes)" if r.size is not None else "")
            )
            digest = fetch(resolve_url(base, repo, revision, r.name), dst, r)
        st = dst.stat()
        files[r.name] = {
            "sha256": digest,
            "size": st.st_size,
            "mtime_ns": st.st_mtime_ns,
            "lfs": r.sha256 is not None,
        }
    marker = {
        "model": MODEL,
        "repo": repo,
        "revision": revision,
        "engram": engram,
        "verified_at": int(time.time()),
        "files": files,
    }
    tmp = marker_path(dest).with_suffix(".tmp")
    tmp.write_text(
        json.dumps(marker, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(tmp, marker_path(dest))
    return marker
