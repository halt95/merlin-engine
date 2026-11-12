# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""The logic behind ``./setup.sh``: host checks, a few questions, the download plan, install, checkpoint, start.

    setup.sh [--yes] [--model M] [--models-dir D] [--host H] [--api-key K] [--port P] [--tp N] [--instances N]
             [--ple fp8|int6] [--kv-ram-cache SIZE] [--swarm | --no-swarm] [--spec mtp|dflash2]
             [--from-source] [--no-start] [--dry-run] [--reconfigure] [--rehash] [--simulate SPEC]
    merlin pull <model> [--models-dir D] [--rehash]        (download + verify only; no GPU needed)

Answers live in ``.merlin/setup.json`` per model (a re-run asks nothing), the API key in ``.merlin/api-key`` (0600),
the install record in ``.merlin/state.json`` (wheel sha + lock hash + commit: a matching venv is reused), and a
verified checkpoint carries ``.merlin-verified.json`` (model, file list, sizes and mtimes: the full re-hash runs when
anything in that list changed, or with ``--rehash``). Standard library only; ``setup.sh`` runs it on the repo-local
Python 3.13 and the venv's ``hf`` does the (resumable) checkpoint download.
"""

from __future__ import annotations

import argparse
import contextlib
import errno
import hashlib
import json
import os
import re
import secrets
import shlex
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
import urllib.request
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))
import launch as L  # noqa: E402
import ple_pack as PP  # noqa: E402
import ple_store as PS  # noqa: E402
import preflight as pf  # noqa: E402
import registry as R  # noqa: E402
import serve_cli as S  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
# The release dependency lock install-env.sh installs (release/install-env.sh REQ).
RELEASE_LOCK = "requirements-0.33.lock"
# Checkpoints the launcher serves that setup cannot install or verify yet: no checkpoint.sha256 in the engine (it is written
# when the checkpoint is staged for publication). `merlin serve <name> <dir>` still serves a local copy.
SETUP_WITHHELD: dict[str, str] = {}  # none in this release
# flash-next, swift1.5, swift1.5-huihui, 27b, 27b-thinkingcap, nemotron-super-vl, ornith1.5
MODELS = [m for m in L.CHECKPOINTS if m not in SETUP_WITHHELD]
FAMILY = {k: v[0] for k, v in L.CHECKPOINTS.items()}
RECIPE = {k: v[1] for k, v in L.CHECKPOINTS.items()}
FAMILY_DIR = {k: Path(v[1]).name for k, v in L.CHECKPOINTS.items()}
TITLE = {
    "flash-next": "Qwen3.8 Flash-Next W4A16 (the reference recipe)",
    "swift1.5": "Swift 1.5 Flash-Next W4A16 (UkisAI fine-tune)",
    "swift1.5-huihui": "Swift 1.5 Huihui Flash-Next W4A16 (abliterated)",
    "27b": "Qwen3.8 27B W4A16 (dense; 1, 2 or 4 cards)",
    "27b-thinkingcap": "ThinkingCap Qwen3.8 27B W4A16 (BottleCap AI; separate weight licence)",
    "nemotron-super-vl": "Nemotron 3.5 Super VL 120B-A12B W4A16 (4 cards)",
    "ornith1.5": "Ornith 1.5 122B-A10B W4A16 (4 x 24 GB; with or without P2P)",
}
WHEEL_RELEASE_URL = "https://github.com/halt95/merlin-engine/releases/download/merlin-v{version}/{wheel}"
SETUP_JSON = "setup.json"
STATE_JSON = "state.json"
SETUP_LOCK = "setup.lock"  # held for the whole engine install
KEY_FILE = "api-key"
VERIFIED_MARKER = L.VERIFIED_MARKER
MIN_GCC = 13
GCC_CANDIDATES = ("14", "13", "15")  # gcc-N / g++-N looked up on PATH when cc is too old
RAM_OTHER_WARN_GIB = 32.0  # 27B / Super VL / Ornith: no measured host-RAM floor; below this the load is unproven
APT = "sudo apt install"
NOP2P_TP1_WHY = ("P2P is off among these cards: the measured best aggregate without P2P is one single-card server per card "
                 "(measured on 4 x CMP 170HX: TP1 x 4 = 873 tok/s, TP2 x 2 = 523, TP4 = 371), so setup recommends "
                 "--tp 1 --instances {n}; `merlin serve` alone would start one TP1 server on GPU {g}. Each single-card "
                 "server takes images up to 4 MP (the multi-card layouts take 8 MP)")


@dataclass(frozen=True)
class PleTable:
    """One of Flash-Next's PLE ("n-gram") embedding tables, held in host RAM (one copy per server)."""
    key: str  # the saved answer (setup.json ple_table); --ple fp8|int6 picks it
    dtype: str  # text_config.ple_embedding_dtype of a checkpoint carrying it
    gib: float  # the table alone in host RAM
    title: str


# FP8 is the default PLE table and nothing is asked; INT6 g32 is an opt-in flag (--ple int6), refused (int6_reason)
# unless the engine has its loader (ple_formats.py, booted by launch.py as PLE_OPT_IN_TABLES, not a test table) and
# the family's INT6 PLE pack is recorded (release/download-sizes.json ple_packs.int6 +
# models/flash-next/ple-int6-pack.sha256). There is no second checkpoint per model: setup lays the one pack over the
# user's verified FP8 checkpoint (ple_pack.py). No INT4. INT6 g32 measured no quality loss vs FP8 and is the route for
# smaller hosts. Where the table lives is separate: --ple-store ram|disk, asked after the Models folder.
PLE_TABLES = {
    "fp8": PleTable("fp8", "float8_e4m3fn", pf.FN_PLE_TABLE_GIB, "FP8 (the shipped table)"),
    "int6": PleTable("int6", "int6_g32", 38.74, "INT6 g32 (same quality and speed as FP8 in a side-by-side comparison)"),
}
PLE_DEFAULT = "fp8"
# The members whose checksum lists carry the ten model-plefp8-* shards the pack replaces, byte for byte (the same sha256
# in each list; int6_reason re-checks it).
PLE_INT6_MODELS = ("flash-next", "swift1.5")
# Members whose FP8 PLE shards are the pack's but whose INT6 boot failed a quality gate: refused with the reason (Huihui's
# INT6 MTP acceptance was 3.253 against FP8's 3.399 on the qualification run, past the -0.10 bar; the pack is unchanged).
# One table, shared with the launcher, which refuses the same checkpoints.
PLE_INT6_UNQUALIFIED = L.PLE_INT6_UNQUALIFIED
# The KV RAM cache = the KV host tier: off unless --kv-ram-cache SIZE, host RAM per server, which merlin serve splits
# evenly over the server's ranks (TP x PP) into this engine knob (GiB per rank). Setup never exports the knob itself.
KV_TIER_ENV = L.FN_KV_HOST_TIER
# The drafter: MTP by default; --spec dflash2 (or the one Drafter question, asked only on
# Flash-Next rows with a qualified DFlash2 geometry) downloads incoai's DFlash2 drafter and serves with it.
SPEC_DEFAULT = "mtp"
DFLASH2_TITLES = {"mtp": "MTP (recommended): bigger pool: 6x131K or 3x262K",
                  "dflash2": "DFlash2: +5-18 % decode, 3x262K only; downloads the drafter"}


class SetupError(Exception):
    def __init__(self, msg: str, code: int = 1):
        super().__init__(msg)
        self.code = code


# --------------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------------
def gib(n: int | float | None) -> str:
    if n is None:
        return "size unknown"
    if n < 1 << 20:
        return f"{n / 1024:.0f} KB"
    if n < 1 << 30:
        return f"{n / (1 << 20):.0f} MB"
    return f"{n / (1 << 30):.1f} GiB"


WIDTH = 100  # the output width (a terminal narrower than this wraps at its own width)


def wrap(text: str, width: int, first: str = "", rest: str | None = None) -> list[str]:
    """``text`` wrapped to ``width`` after the prefix ``first``; continuation lines start with ``rest`` (default:
    spaces as wide as ``first``). Words never break, so URLs, paths and profile names stay whole; each line of a
    multi-line text keeps its own leading spaces under the hanging indent. ``width`` 0: no wrapping."""
    rest = " " * len(first) if rest is None else rest
    out: list[str] = []
    for i, line in enumerate(text.split("\n")):
        body = line.lstrip(" ")
        own = line[: len(line) - len(body)]
        lead, sub = (first, rest) if i == 0 else (rest + own, rest + own + ("  " if own else ""))
        if width <= 0:
            out.append((lead + body).rstrip())
            continue
        out += textwrap.wrap(body, max(width, len(lead) + 20), initial_indent=lead, subsequent_indent=sub,
                             break_long_words=False, break_on_hyphens=False) or [lead.rstrip()]
    return out


def shell_wrap(words: list[str], width: int, indent: str = "  ", cont: str = "      ") -> list[str]:
    """A shell command over several lines joined by `` \\`` continuations when it is wider than ``width``; a
    ``-flag`` stays with its value and no word is split (pasted back, it is the same command)."""
    units: list[str] = []
    for w in words:
        if units and units[-1].startswith("-") and " " not in units[-1] and not w.startswith("-"):
            units[-1] += " " + w
        else:
            units.append(w)
    lines, cur = [], ""
    for u in units:
        if cur and len(cur) + 1 + len(u) + 2 > width:
            lines.append(cur + " \\")
            cur = cont + u
        else:
            cur = (cur + " " + u) if cur else indent + u
    return lines + [cur]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def release_version(repo: Path = REPO) -> str:
    """The release line of docs/CHANGELOG.md (``## Merlin <version>``) names the GitHub release tag ``merlin-v<version>``."""
    env = os.environ.get("MERLIN_RELEASE_VERSION")
    if env:
        return env
    for line in (repo / "docs" / "CHANGELOG.md").read_text(encoding="utf-8").splitlines():
        # Three or four parts (3.0.2, 3.0.2.1); a longer or dotted tail is no release line.
        m = re.match(r"^## Merlin (\d+(?:\.\d+){2,3})(?![\d.])", line)
        if m:
            return m[1]
    raise SetupError("docs/CHANGELOG.md names no `## Merlin <version>` release; set MERLIN_RELEASE_VERSION")


def wheel_record(repo: Path = REPO) -> tuple[str, str]:
    """(sha256, filename) from release/merlin-wheel.sha256."""
    text = (repo / "release" / "merlin-wheel.sha256").read_text(encoding="utf-8")
    m = re.search(r"(?m)^([0-9a-f]{64})\s+\*?(\S+\.whl)\s*$", text)
    if not m:
        raise SetupError("release/merlin-wheel.sha256 names no wheel")
    return m[1], m[2]


def wheel_urls(repo: Path = REPO) -> list[str]:
    """The release asset's URL and the forms GitHub may give a name with ``+`` (``%2B`` encoded, or ``.`` after an
    upload rename); MERLIN_WHEEL_URL replaces the first. The sha256 check applies to whichever answers."""
    _, wheel = wheel_record(repo)
    url = os.environ.get("MERLIN_WHEEL_URL") or WHEEL_RELEASE_URL.format(version=release_version(repo), wheel=wheel)
    base, _, name = url.rpartition("/")
    out = [url]
    for alt in (name.replace("+", "%2B"), name.replace("+", ".")):
        if alt != name and f"{base}/{alt}" not in out:
            out.append(f"{base}/{alt}")
    return out


def lock_hash(repo: Path = REPO) -> str:
    return sha256_file(repo / "release" / RELEASE_LOCK)


def sizes_table(repo: Path = REPO) -> dict[str, Any]:
    return json.loads((repo / "release" / "download-sizes.json").read_text(encoding="utf-8"))


def fn_family(model: str | None) -> bool:
    return bool(model) and FAMILY.get(model) == "flash-next"


def tier_family(model: str | None) -> bool:
    """The models --kv-ram-cache applies to: the Flash-Next family, the 27B and Ornith 1.5 (the last two
    untested until their qualification runs)."""
    return bool(model) and FAMILY.get(model) in S.KV_TIER_FAMILIES


def swarm_family(model: str | None) -> bool:
    """The models --swarm applies to: the families with rows that have a qualified swarm cap: Flash-Next and the 27B."""
    return bool(model) and FAMILY.get(model) in S.SWARM_FAMILIES


def is_variant(ple: str | None) -> bool:
    """A PLE table other than the one the model's own checkpoint carries (FP8)."""
    return ple not in (None, PLE_DEFAULT)


def ck_entry(model: str, table: dict[str, Any]) -> dict[str, Any]:
    """The release/download-sizes.json entry of the model's checkpoint (the FP8 one: the INT6 table is a pack laid over
    it, never a checkpoint of its own)."""
    return table["checkpoints"][model]


def hf_repo(model: str, repo: Path = REPO) -> str:
    return ck_entry(model, sizes_table(repo))["repo"]


def checksum_list(model: str, repo: Path = REPO) -> Path:
    return repo / RECIPE[model] / "checkpoint.sha256"


def pack_entry(table: dict[str, Any]) -> dict[str, Any] | None:
    """release/download-sizes.json ``ple_packs.int6``: the Flash-Next family's INT6 PLE pack (None when not recorded)."""
    e = (table.get("ple_packs") or {}).get(PP.PACK_KEY)
    return e if isinstance(e, dict) else None


def pack_list(repo: Path = REPO) -> Path:
    return repo / PP.PACK_LIST


def fp8_ple_sums(model: str, repo: Path = REPO) -> dict[str, str]:
    """The model-plefp8-* lines of a model's checksum list ({} without a list)."""
    lst = checksum_list(model, repo)
    return {n: h for n, h in parse_checksums(lst)[0].items() if n.startswith(PP.FP8_SHARD_PREFIX)} if lst.is_file() else {}


def int6_engine_reason(repo: Path = REPO) -> str | None:
    """Why this checkout cannot serve an INT6 g32 table (None when it can): the engine must carry the INT6 g32 loader, and its
    launcher boots the format only as an opt-in test table until it is promoted."""
    try:
        loader = (repo / "vllm" / "models" / "qwen4_exp" / "nvidia" / "ple_formats.py").read_text(encoding="utf-8")
    except OSError:
        loader = ""
    if "int6_g32" not in loader:
        return "this engine has no INT6 g32 PLE loader (the engine build is older than the INT6 table)"
    if "int6_g32" in getattr(L, "PLE_TEST_TABLES", {}):
        return "the launcher still treats int6_g32 as a test table (MERLIN_PLE_TEST_TABLE), not a shipped one"
    return None


def int6_reason(model: str | None, repo: Path = REPO, table: dict[str, Any] | None = None) -> str | None:
    """Why the INT6 g32 table cannot be chosen for ``model`` here; None when it can (the pack may still be pending
    publication: a local copy then serves)."""
    why = L.int6_unqualified_reason(model)
    if why:
        return why
    if model not in PLE_INT6_MODELS:
        return f"the INT6 g32 PLE pack does not cover {model} (it covers {', '.join(PLE_INT6_MODELS)})"
    why = int6_engine_reason(repo)
    if why:
        return why
    if pack_entry(table or sizes_table(repo)) is None or not pack_list(repo).is_file():
        return (f"the INT6 g32 PLE pack is not recorded in this release (release/download-sizes.json has no ple_packs.{PP.PACK_KEY} "
                f"entry or {PP.PACK_LIST} is missing)")
    problem = checksum_list_problem(pack_list(repo))
    if problem:
        return problem
    ref = fp8_ple_sums("flash-next", repo)
    if not ref or fp8_ple_sums(model, repo) != ref:
        return (f"{model}'s FP8 PLE shards ({PP.FP8_SHARD_PREFIX}* in {RECIPE[model]}/checkpoint.sha256) are not the ones the "
                "pack replaces, so the pack does not apply to it")
    return None


def int6_checkpoint_reason(ckpt_dir: Path) -> str | None:
    """Why the checkpoint in ``ckpt_dir`` does not carry the INT6 g32 table (None when its config.json says it does)."""
    cfg = read_json(ckpt_dir / "config.json") or {}
    dtype = (cfg.get("text_config") or cfg).get("ple_embedding_dtype")
    if dtype != PLE_TABLES["int6"].dtype:
        return f"{ckpt_dir} carries the {dtype or 'FP8'} PLE table, not INT6 g32 (ple_embedding_dtype in its config.json)"
    return None


CHECKSUM_LINE = re.compile(r"^([0-9a-fA-F]{64})\s+\*?(.+?)\s*$")


def parse_checksums(path: Path) -> tuple[dict[str, str], str | None]:
    """{filename: sha256} and the Hugging Face revision the header names (None when it names none)."""
    sums: dict[str, str] = {}
    rev = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#"):
            m = re.search(r"revision ([0-9a-f]{40})", line)
            if m:
                rev = m[1]
            continue
        m = CHECKSUM_LINE.match(line)
        if m:
            sums[m[2]] = m[1].lower()
    return sums, rev


def checksum_list_problem(path: Path) -> str | None:
    """Why ``path`` cannot verify anything (None when every non-comment line parses and at least one does): an empty,
    truncated or LFS-stub list would otherwise hash nothing and still report the checkpoint as verified."""
    lines = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#")]
    bad = [l for l in lines if not CHECKSUM_LINE.match(l)]
    if bad:
        return f"{path} has {len(bad)} line(s) that are not '<sha256>  <file>' (first: {bad[0][:60]!r})"
    if not lines:
        return f"{path} lists no files"
    return None


def pending_publication(model: str, repo: Path = REPO, table: dict[str, Any] | None = None) -> bool:
    """True for a checkpoint whose Hugging Face repository is not public yet (release/download-sizes.json
    ``pending_publication``). The marker is only valid with no recorded revision: a pinned revision means it is published."""
    ck = ck_entry(model, table or sizes_table(repo))
    if not ck.get("pending_publication"):
        return False
    if ck.get("revision") or pinned_revision(model, repo, table):
        raise SetupError(f"{model}: release/download-sizes.json marks it pending_publication but a revision is recorded")
    return True


def not_published_msg(model: str) -> str:
    return (f"{model} is not on Hugging Face yet; pass --models-dir with a folder that holds a local copy of the checkpoint "
            f"(the files models/{FAMILY_DIR.get(model, model)}/checkpoint.sha256 lists)")


def pinned_revision(model: str, repo: Path = REPO, table: dict[str, Any] | None = None) -> str | None:
    """The one revision a model is downloaded at: the checksum list's header, else release/download-sizes.json; both
    present and different is an inconsistency in the repository. None for a repository whose revision is not recorded
    yet (pending_publication): the plan then resolves it through the Hugging Face API or refuses."""
    _, rev = parse_checksums(checksum_list(model, repo))
    table_rev = ck_entry(model, table or sizes_table(repo)).get("revision")
    if rev and table_rev and rev != table_rev:
        raise SetupError(f"{model}: checkpoint.sha256 names revision {rev} but release/download-sizes.json {table_rev}")
    return rev or table_rev


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return d if isinstance(d, dict) else None


def write_json(path: Path, data: dict[str, Any], mode: int = 0o644) -> None:
    write_private(path, json.dumps(data, indent=1, sort_keys=True) + "\n", mode)


def write_private(path: Path, text: str, mode: int = 0o600) -> None:
    """Create the file with ``mode`` from the first byte (no world-readable window) under a 0700 .merlin."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.name == ".merlin":
        try:
            os.chmod(path.parent, 0o700)
        except OSError:
            pass
    # a unique temp file per writer (mkstemp creates it 0600): concurrent writers never share an inode
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        try:
            os.fchmod(fd, mode)  # mkstemp's mode is 0600; the file is ours, so set exactly what was asked
        except (OSError, AttributeError):
            pass
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def which(name: str, env: dict[str, str] | None = None) -> str | None:
    return shutil.which(name, path=(env or os.environ).get("PATH"))


def tool_major(tool: str, run: Callable[[list[str]], str | None]) -> int | None:
    out = run([tool, "-dumpversion"])
    if not out:
        return None
    head = out.strip().split(".")[0]
    return int(head) if head.isdigit() else None


def default_runner(cmd: list[str]) -> str | None:
    try:
        cp = subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return cp.stdout if cp.returncode == 0 else None


def nearest_existing(path: Path) -> Path:
    p = path
    while not p.exists() and p != p.parent:
        p = p.parent
    return p


def disk_free(path: Path) -> int | None:
    """Free bytes on the filesystem that holds ``path`` (its nearest existing parent)."""
    try:
        return shutil.disk_usage(nearest_existing(path)).free
    except OSError:
        return None


def device_of(path: Path) -> Any:
    try:
        return os.stat(nearest_existing(path)).st_dev
    except OSError:
        return str(path)


def uv_cache_dir() -> Path:
    env = os.environ.get("UV_CACHE_DIR")
    if env:
        return Path(env)
    xdg = os.environ.get("XDG_CACHE_HOME")
    return (Path(xdg) if xdg else Path.home() / ".cache") / "uv"


# --------------------------------------------------------------------------------------------
# Arguments and saved answers (per model)
# --------------------------------------------------------------------------------------------
PER_MODEL = ("models_dir", "port", "tp", "instances", "ple_table", "ple_store", "kv_ram_cache_gib", "spec", "swarm", "context")


@dataclass
class Answers:
    model: str | None = None
    models_dir: str | None = None
    host: str = "127.0.0.1"
    api_key: str | None = None  # never saved in setup.json: .merlin/api-key (0600)
    port: int | None = None
    tp: int | None = None
    instances: int | None = None
    from_source: bool = False
    cc: str | None = None
    cxx: str | None = None
    ple_table: str | None = None  # Flash-Next family: fp8, or int6 with --ple int6 (None elsewhere)
    ple_store: str | None = None  # Flash-Next family: ram, or disk with --ple-store disk (None elsewhere)
    kv_ram_cache_gib: int | None = None  # Flash-Next, 27B, Ornith: --kv-ram-cache, GiB per server, 0 = off (else None)
    spec: str | None = None  # Flash-Next family: the drafter, mtp / dflash2 (None elsewhere: their MTP head)
    swarm: bool | None = None  # Flash-Next, 27B: --swarm, the row's tested high-concurrency cap (False = the default cap; else None)

    context: int | None = None  # tokens; None keeps the native/default context

    def base_dir(self) -> Path:
        """The FP8 checkpoint folder (--models-dir, else ~/models/<model>), with --ple int6 too: the overlay is built
        beside it and the base is never written."""
        return Path(self.models_dir or default_dir(self.model)).expanduser()

    def checkpoint_dir(self) -> Path:
        """The folder merlin serve gets: the base checkpoint, or with --ple int6 its overlay ``<base>-int6``."""
        return PP.overlay_dir(self.base_dir()) if fn_family(self.model) and is_variant(self.ple_table) else self.base_dir()

    def pack_dir(self) -> Path:
        return PP.pack_dir(self.base_dir())


def default_dir(model: str | None) -> str:
    return f"~/models/{model}"


def normalize_fn(a: Answers) -> Answers:
    """The PLE table belongs to the Flash-Next family, the KV tier to it, the 27B and Ornith: FP8 and the tier off
    unless a flag (or a saved flag) says otherwise, never asked; nothing elsewhere."""
    if fn_family(a.model):
        a.ple_table = a.ple_table or PLE_DEFAULT
        a.ple_store = a.ple_store or "ram"
        a.spec = a.spec or SPEC_DEFAULT
    else:
        a.ple_table = a.ple_store = a.spec = None
    a.kv_ram_cache_gib = (a.kv_ram_cache_gib or 0) if tier_family(a.model) else None
    a.swarm = bool(a.swarm) if swarm_family(a.model) else None
    return a


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="setup.sh", description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Every option and the update / uninstall flow: docs/SETUP.md",
    )
    ap.add_argument("--yes", "-y", action="store_true", help="no questions: the recommended (or given) answers, downloads confirmed")
    ap.add_argument("--model", choices=MODELS, help="checkpoint to install and serve")
    ap.add_argument("--models-dir", metavar="DIR", help="checkpoint folder (default ~/models/<model>; an existing verified copy is reused). "
                    "With --ple int6 it is still the FP8 checkpoint: the INT6 overlay is DIR-int6 and the pack is shared beside it")
    ap.add_argument("--host", help="127.0.0.1 (this machine only, default) or 0.0.0.0 (network: needs --api-key or one is generated)")
    ap.add_argument("--api-key", metavar="KEY", help="bearer key the server requires (stored in .merlin/api-key, 0600)")
    ap.add_argument("--port", type=int, help="HTTP port (default: the model's, 8000 Flash-Next / Super VL, 8100 27B)")
    ap.add_argument("--tp", type=int, help="pick the layout with this tensor-parallel size")
    ap.add_argument("--instances", type=int, help="start N servers on N disjoint card sets")
    ap.add_argument("--ple", choices=list(PLE_TABLES), default=None,
                    help=f"Flash-Next: the PLE table in host RAM, fp8 ({PLE_TABLES['fp8'].gib} GiB, the default) or int6 "
                    f"(INT6 g32, {PLE_TABLES['int6'].gib} GiB): one pack for the family, downloaded once beside the FP8 checkpoint, "
                    "which stays untouched; the served folder is <checkpoint>-int6 (links + a generated config.json and index)")
    ap.add_argument("--ple-store", choices=PS.STORES, default=None,
                    help="Flash-Next: where the PLE table lives, ram (the default) or disk (experimental: read from the checkpoint's own files, "
                    "which frees the table's host RAM, but first reads are much slower). Asked after the models folder when it is on local NVMe")
    ap.add_argument("--kv-ram-cache", metavar="SIZE", help="Flash-Next (experimental, MTP only), 27B (experimental) and Ornith 1.5 (untested): a host-RAM cache for prefix KV blocks the GPU pool "
                    "evicts (restored instead of recomputed; adds no context and does not change the session cap (--swarm does); RAM only, never disk; ~66K tokens/GiB on Flash-Next TP2xPP2; needs memlock: LimitMEMLOCK=infinity or ulimit -l unlimited). SIZE = "
                    "host RAM per server (24g, 24gib or 24 = GiB; 0 = off, the default), split over its TP x PP ranks (rounded down to whole GiB per rank)")
    ap.add_argument("--context", type=L.C.parse_size, metavar="SIZE",
                    help="tokens or binary k/m (512k = 524288); default native, auto-YaRN past native")
    ap.add_argument("--swarm", action=argparse.BooleanOptionalAction, default=None,
                    help="the row's tested high-concurrency cap instead of its default (experimental): the Flash-Next 4 x 3090 profiles 24 "
                    "sessions (default 8), the 27B 4 x 3090 (TP4) profiles 32 (default 24). Needs the KV RAM cache at the size that cap was "
                    "qualified at (48g per server): without --kv-ram-cache the cache is set to that size, a smaller one is refused. Asked once, "
                    "default no, on rows that have a swarm cap; --no-swarm turns a saved one off. An exported SEQS still wins over every cap")
    ap.add_argument("--spec", choices=S.SPEC_CHOICES, help="Flash-Next: the drafter, mtp (default; bigger pool: 6 x 131K or "
                    f"3 x 262K) or dflash2 ({S.dflash2_why(strict=False).replace(chr(37), chr(37) * 2)}; downloads {S.DFLASH2_REPO})")
    ap.add_argument("--from-source", action="store_true", help="compile the engine instead of taking the release wheel")
    ap.add_argument("--no-start", action="store_true", help="install and download, then stop before `merlin serve`")
    ap.add_argument("--dry-run", action="store_true", help="print checks, answers, download plan and serve command; download and write nothing")
    ap.add_argument("--reconfigure", action="store_true", help="ask the questions again (saved answers are the defaults)")
    ap.add_argument("--rehash", action="store_true", help="hash every checkpoint file again even when the verified marker is current")
    ap.add_argument("--simulate", metavar="SPEC", help="preflight's hardware spec (4x3090, 2x3090:ram=64, ...): implies --dry-run unless --no-start")
    ap.add_argument("--pull-only", action="store_true", help=argparse.SUPPRESS)  # merlin pull
    return ap


def parse_args(argv: list[str]) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    if args.simulate and not args.no_start:  # nothing ever starts on fabricated hardware; --no-start still installs
        args.dry_run = True
    if args.host and args.host not in ("127.0.0.1", "0.0.0.0", "localhost", "::1"):
        if not re.fullmatch(r"[0-9.]+|[0-9a-fA-F:]+", args.host):
            raise SetupError(f"--host {args.host!r}: give an IP address (127.0.0.1 or 0.0.0.0)", 2)
    if args.port is not None and not 1 <= args.port <= 65535:
        raise SetupError(f"--port {args.port}: 1..65535", 2)
    if args.tp is not None and args.tp < 1 or args.instances is not None and args.instances < 1:
        raise SetupError("--tp and --instances take whole numbers >= 1", 2)
    args.ple_table = args.ple
    try:
        args.kv_ram_cache_gib = None if args.kv_ram_cache is None else S.parse_kv_ram_cache(args.kv_ram_cache)
    except S.CliError as e:
        raise SetupError(str(e), 2) from None
    if args.model and not fn_family(args.model) and args.spec == "dflash2":
        raise SetupError(f"--spec dflash2: DFlash2 is only available for the Flash-Next family ({L.FN_FAMILY_NAMES}); "
                         f"{args.model} uses its MTP head", 2)
    if args.model and not fn_family(args.model) and is_variant(args.ple):
        raise SetupError(f"--ple int6 applies to the Flash-Next family only, not {args.model}", 2)
    if args.model and not fn_family(args.model) and args.ple_store == "disk":
        raise SetupError(f"--ple-store disk applies to the Flash-Next family only, not {args.model}", 2)
    if args.model and not tier_family(args.model) and args.kv_ram_cache_gib:
        raise SetupError(f"--kv-ram-cache applies to the Flash-Next family, the 27B and Ornith 1.5 only, not {args.model}", 2)
    if args.swarm:
        if args.model and not swarm_family(args.model):
            raise SetupError(f"--swarm applies to the Flash-Next family and the 27B only, not {args.model}", 2)
        if args.kv_ram_cache_gib == 0:
            raise SetupError("--swarm needs the KV RAM cache (it is sized for the swarm cap when --kv-ram-cache is not given); "
                             "--kv-ram-cache 0 turns it off", 2)
        if args.spec == "dflash2":
            raise SetupError("--swarm with --spec dflash2: the KV RAM cache --swarm needs is qualified with MTP only", 2)
    return args


def _valid_saved(saved: dict[str, Any] | None) -> dict[str, Any]:
    """The saved file with anything malformed dropped (a bad model name, port or block never raises later)."""
    if not saved:
        return {"models": {}}
    out: dict[str, Any] = {}
    if saved.get("model") in MODELS:
        out["model"] = saved["model"]
    if saved.get("host") in ("127.0.0.1", "0.0.0.0", "localhost", "::1"):
        out["host"] = saved["host"]
    if isinstance(saved.get("from_source"), bool):
        out["from_source"] = saved["from_source"]
    for k in ("cc", "cxx"):
        if isinstance(saved.get(k), str) and saved[k]:
            out[k] = saved[k]
    models = saved.get("models")
    out["models"] = {}
    if isinstance(models, dict):
        for m, block in models.items():
            if m in MODELS and isinstance(block, dict):
                b: dict[str, Any] = {}
                if isinstance(block.get("models_dir"), str) and block["models_dir"]:
                    b["models_dir"] = block["models_dir"]
                for k in ("port", "tp", "instances", "context"):
                    v = block.get(k)
                    if isinstance(v, int) and not isinstance(v, bool) and v >= 1 and (k != "port" or v <= 65535):
                        b[k] = v
                if fn_family(m) and isinstance(block.get("ple_table"), str) and block["ple_table"] in PLE_TABLES:
                    b["ple_table"] = block["ple_table"]
                if fn_family(m) and block.get("ple_store") in PS.STORES:
                    b["ple_store"] = block["ple_store"]
                if fn_family(m) and block.get("spec") in S.SPEC_CHOICES:
                    b["spec"] = block["spec"]
                v = block.get("kv_ram_cache_gib")
                if tier_family(m) and isinstance(v, int) and not isinstance(v, bool) and v >= 0:
                    b["kv_ram_cache_gib"] = v
                if swarm_family(m) and isinstance(block.get("swarm"), bool):
                    b["swarm"] = block["swarm"]
                out["models"][m] = b
    return out


def merge_answers(saved: dict[str, Any] | None, args: argparse.Namespace, key: str | None = None) -> Answers:
    """Saved answers for the chosen model (the saved model unless --model names another), then every flag given on the
    command line. The folder, port and layout belong to a model: another model starts from its own defaults."""
    s = _valid_saved(saved)
    model = args.model or s.get("model")
    block = s["models"].get(model, {}) if model else {}
    a = Answers(model=model, host=s.get("host", "127.0.0.1"), from_source=s.get("from_source", False), cc=s.get("cc"), cxx=s.get("cxx"))
    a.api_key = key
    for k in PER_MODEL:
        setattr(a, k, block.get(k))
    for k in ("models_dir", "host", "api_key", "port", "tp", "instances", "ple_table", "ple_store", "kv_ram_cache_gib", "spec", "swarm", "context"):
        v = getattr(args, k)
        if v is not None:
            setattr(a, k, v)
    if args.from_source:
        a.from_source = True
    return normalize_fn(a)


def saved_block(a: Answers) -> dict[str, Any]:
    return {k: getattr(a, k) for k in PER_MODEL}


def save_answers(path: Path, saved: dict[str, Any] | None, a: Answers) -> dict[str, Any]:
    s = _valid_saved(saved)
    s.update({"model": a.model, "host": a.host, "from_source": a.from_source})
    for k in ("cc", "cxx"):
        if getattr(a, k):
            s[k] = getattr(a, k)
        else:
            s.pop(k, None)
    s["models"][a.model] = saved_block(a)
    write_json(path, s, 0o600)
    return s


def answers_complete(a: Answers, saved: dict[str, Any] | None) -> bool:
    """A saved set exists for this model (folder, port, layout) and network access has its key."""
    s = _valid_saved(saved)
    return bool(a.model) and a.model in s["models"] and a.port is not None and (a.host == "127.0.0.1" or bool(a.api_key))


def read_key(path: Path) -> str | None:
    try:
        key = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return key or None


# --------------------------------------------------------------------------------------------
# Host checks
# --------------------------------------------------------------------------------------------
@dataclass
class Problem:
    level: str  # FAIL or WARN
    code: str
    text: str
    fix: str = ""


def find_gcc(run: Callable[[list[str]], str | None], env: dict[str, str] | None = None) -> tuple[str, str] | None:
    """A ``gcc-N`` / ``g++-N`` pair (N >= 13) on PATH, newest first."""
    for n in sorted(GCC_CANDIDATES, reverse=True):
        cc, cxx = f"gcc-{n}", f"g++-{n}"
        if which(cc, env) and which(cxx, env) and (tool_major(cc, run) or 0) >= MIN_GCC and (tool_major(cxx, run) or 0) >= MIN_GCC:
            return cc, cxx
    return None


def gcc_problem(run: Callable[[list[str]], str | None], env: dict[str, str] | None = None,
                a: Answers | None = None) -> Problem | None:
    """GCC/G++ >= 13 for the install (release/install-env.sh checks CC / CXX, default cc / c++). The pair comes from
    CC / CXX in the environment, else the saved answers, else cc / c++; when those are too old a ``gcc-N`` / ``g++-N``
    pair on PATH is taken and remembered in the answers (so one fix lasts across re-runs)."""
    env = env or dict(os.environ)
    cc = env.get("CC") or (a.cc if a else None) or "cc"
    cxx = env.get("CXX") or (a.cxx if a else None) or "c++"
    fix = (f"{APT} gcc-{MIN_GCC} g++-{MIN_GCC}   (Debian 12 / Ubuntu 24.04 ship GCC 13 or 14; on Ubuntu 22.04 first "
           f"`sudo add-apt-repository ppa:ubuntu-toolchain-r/test`), then run ./setup.sh again (it finds gcc-{MIN_GCC} on PATH "
           f"and remembers it; CC=gcc-{MIN_GCC} CXX=g++-{MIN_GCC} ./setup.sh names another pair)")

    def problem(pair: tuple[str, str]) -> Problem | None:
        for t in pair:
            if which(t, env) is None:
                return Problem("FAIL", "compiler", f"{t} is missing (the install needs a C/C++ compiler >= GCC {MIN_GCC})", fix)
            major = tool_major(t, run)
            if major is None:
                return Problem("FAIL", "compiler", f"cannot read the version of {t} (GCC/G++ >= {MIN_GCC} required)", fix)
            if major < MIN_GCC:
                return Problem("FAIL", "compiler", f"{t} is GCC {major}; the engine's CUDA ops need GCC/G++ >= {MIN_GCC}", fix)
        return None

    p = problem((cc, cxx))
    if p is None:
        if a is not None and (env.get("CC") or env.get("CXX")):
            a.cc, a.cxx = cc, cxx  # an explicit pair is remembered
        return None
    if not (env.get("CC") or env.get("CXX")):
        found = find_gcc(run, env)
        if found:
            if a is not None:
                a.cc, a.cxx = found
            return None
    return p


def compiler_env(a: Answers, env: dict[str, str]) -> dict[str, str]:
    env = dict(env)
    if a.cc and not env.get("CC"):
        env["CC"] = a.cc
    if a.cxx and not env.get("CXX"):
        env["CXX"] = a.cxx
    return env


def git_problem(repo: Path, run: Callable[[list[str]], str | None], env: dict[str, str] | None = None) -> Problem | None:
    if which("git", env) is None:
        return Problem("FAIL", "git", "git is missing", f"{APT} git")
    dirty = run(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"])
    if dirty is None:
        return Problem("FAIL", "git", f"{repo} is not a git checkout (a ZIP download?)",
                       "clone the repository with git (git clone ...) and run ./setup.sh inside it")
    if dirty.strip():
        return Problem("FAIL", "git", "tracked files are modified; the installer records a clean commit as the engine's source",
                       f"commit or revert them (cd {repo} && git status)")
    return None


def driver_problem(inv: pf.Inventory) -> Problem | None:
    fix = ("install the NVIDIA driver R580 or later (CUDA 13): Ubuntu `sudo apt install nvidia-driver-580-open` (or the "
           ".run installer from https://www.nvidia.com/drivers), reboot, run ./setup.sh again")
    if not inv.gpus:
        return Problem("FAIL", "gpu", "no NVIDIA GPU is visible (nvidia-smi is missing or lists none)", fix)
    d = inv.host.driver
    major = int(d.split(".")[0]) if d and d.split(".")[0].isdigit() else None
    if major is not None and major < pf.DRIVER_MIN_MAJOR:
        return Problem("FAIL", "driver", f"driver {d} is older than the {pf.DRIVER_MIN_MAJOR} series: CUDA 13 needs R580 or later", fix)
    return None


def source_build_problems(run: Callable[[list[str]], str | None], env: dict[str, str] | None = None) -> list[Problem]:
    out = []
    if which("cargo", env) is None:
        out.append(Problem("FAIL", "cargo", "--from-source needs cargo (Rust)", "curl https://sh.rustup.rs -sSf | sh   (per user, no sudo)"))
    if which("cmake", env) is None or which("ninja", env) is None:
        out.append(Problem("FAIL", "cmake", "--from-source needs cmake and ninja", f"{APT} cmake ninja-build"))
    return out


def ram_budget(inv: pf.Inventory, avail: Callable[[], int | None] | None = None) -> tuple[float | None, str]:
    """(GiB, what it is) a new server can have: host RAM (MemTotal, lowered to a cgroup memory.max), and on a probed
    host also what is free now (MemAvailable, capped by what the cgroup has left), whichever is smaller."""
    total = pf._gib(inv.host.ram_mib) if inv.host.ram_mib is not None else None
    if inv.source != "probe":
        return total, "host RAM"
    now = (avail or pf.mem_available_mib)()
    if now is not None and (total is None or pf._gib(now) < total):
        return pf._gib(now), "host memory available now (MemAvailable, capped by the cgroup)"
    return total, "host RAM (MemTotal or the cgroup memory.max)"


def fn_ram_need_gib(ple: str | None, tier_gib: int | None, sizes: list[int], store: str | None = None) -> float:
    """Resident host RAM of Flash-Next servers with ``sizes`` workers each (TP x PP; one entry per --instances
    server): per server the measured floor with the FP8 table swapped for ``ple``'s (none under ``store`` disk, the full
    credit), plus the KV tier's arena of ``tier_gib`` per rank. The floor is preflight's (69 GiB at four
    workers, +2 GiB per worker above four)."""
    delta = (0.0 if store == "disk" else PLE_TABLES[ple or PLE_DEFAULT].gib) - PLE_TABLES[PLE_DEFAULT].gib
    return sum(pf.ram_floor_gib(n, tier_gib or 0) + delta for n in sizes)  # the floor serve's gates use, tier included


def fn_ram_parts(ple: str | None, tier_gib: int | None, sizes: list[int], store: str | None = None) -> str:
    t = PLE_TABLES[ple or PLE_DEFAULT]
    per = f"{len(sizes)} servers x " if len(sizes) > 1 else ""
    table = f"{t.key.upper()} PLE table on disk" if store == "disk" else f"{t.key.upper()} PLE table {t.gib:.1f} GiB"
    parts = f"{per}{table} + the rest of the server ~{fn_ram_need_gib(PLE_DEFAULT, 0, sizes) / len(sizes) - PLE_TABLES[PLE_DEFAULT].gib:.0f} GiB"
    if tier_gib:
        parts += f", KV host tier {tier_gib * sizes[0]} GiB per server ({tier_gib} GiB x {sizes[0]} ranks)"
    return parts


def max_kv_ram_cache_gib(ple: str | None, sizes: list[int], have: float | None, store: str | None = None) -> int | None:
    """The largest --kv-ram-cache (GiB per server, a whole number of GiB per rank) that fits ``have`` beside the servers;
    None when RAM is unknown."""
    if have is None:
        return None
    per_rank = max(0, int((have - fn_ram_need_gib(ple, 0, sizes, store)) // max(1, sum(sizes))))
    return per_rank * (sizes[0] if sizes else 4)


def ram_problem(model: str, inv: pf.Inventory, cards: int, ple: str | None = None, tier_gib: int | None = None,
                sizes: list[int] | None = None, budget: tuple[float | None, str] | None = None,
                int6_why: str | None = "not checked", store: str | None = None, disk_why: str | None = "not checked") -> Problem | None:
    """The host-RAM check of the chosen flags (nothing is picked here). Flash-Next: the servers' resident need for the
    PLE table (FP8, or INT6 with --ple int6) and the KV tier (``tier_gib`` = GiB per rank, from --kv-ram-cache) against
    ``ram_budget`` (FAIL when it cannot fit, with what would); other models: a warning below 32 GiB."""
    if not fn_family(model):
        ram = inv.host.ram_mib
        tier = (tier_gib or 0) * sum(sizes or [cards])  # 27B / Ornith KV RAM cache, pinned, on an unmeasured floor
        if ram is None:
            return Problem("WARN", "ram", "host RAM unknown" + (f"; the KV RAM cache needs {tier} GiB of it" if tier else ""))
        if tier and pf._gib(ram) < tier:
            return Problem("FAIL", "ram", f"host RAM {pf._gib(ram):.0f} GiB is below the {tier} GiB the KV RAM cache pins",
                           "a smaller --kv-ram-cache, or --kv-ram-cache 0 (off)")
        if tier and pf._gib(ram) < RAM_OTHER_WARN_GIB + tier:
            return Problem("WARN", "ram", f"host RAM {pf._gib(ram):.0f} GiB: {model} is unmeasured below {RAM_OTHER_WARN_GIB:.0f} GiB "
                           f"plus the {tier} GiB KV RAM cache")
        if pf._gib(ram) < RAM_OTHER_WARN_GIB:
            return Problem("WARN", "ram", f"host RAM {pf._gib(ram):.0f} GiB: {model} is unmeasured below {RAM_OTHER_WARN_GIB:.0f} GiB")
        return None
    sizes = sizes or [cards]
    ple, tier_gib = ple or PLE_DEFAULT, tier_gib or 0
    have, what = budget or ram_budget(inv)
    if have is None:
        return Problem("WARN", "ram", f"host RAM unknown; Flash-Next needs ~{fn_ram_need_gib(ple, tier_gib, sizes, store):.0f} GiB resident here")
    need = fn_ram_need_gib(ple, tier_gib, sizes, store)
    if have < need:
        fixes = []
        if tier_gib:
            top = max_kv_ram_cache_gib(ple, sizes, have, store)
            fixes.append(f"a smaller KV host tier (--kv-ram-cache {top}g or less)" if top else "the KV host tier off (--kv-ram-cache 0)")
        if ple == "fp8" and int6_why is None and have >= fn_ram_need_gib("int6", tier_gib, sizes):
            fixes.append(f"the INT6 g32 PLE table (--ple int6, ~{fn_ram_need_gib('int6', tier_gib, sizes):.0f} GiB)")
        if store != "disk" and disk_why is None and have >= fn_ram_need_gib(ple, tier_gib, sizes, "disk"):
            fixes.append(f"the PLE table on NVMe (--ple-store disk, ~{fn_ram_need_gib(ple, tier_gib, sizes, 'disk'):.0f} GiB; first reads are much slower)")
        fixes += ["add RAM (or raise the container's memory limit)", "choose 27b (--model 27b), which needs no PLE table"]
        return Problem("FAIL", "ram", f"{what} {have:.0f} GiB is below the ~{need:.0f} GiB Flash-Next needs resident here "
                       f"({fn_ram_parts(ple, tier_gib, sizes, store)}): the load would be OOM-killed", "; or ".join(fixes))
    if store == "disk":  # the full credit, with a warning below floor + 16 GiB of page cache
        if have < need + PS.CACHE_WARN_GIB:
            return Problem("WARN", "ram", f"{what} {have:.0f} GiB leaves under {PS.CACHE_WARN_GIB} GiB of page cache for the PLE table above the "
                           f"~{need:.0f} GiB {fn_ram_parts(ple, tier_gib, sizes, store)} need: decode and prefill slow down")
        return None
    if ple == PLE_DEFAULT and not tier_gib:
        if have < pf.RAM_WARN_GIB:
            return Problem("WARN", "ram", f"{what} {have:.0f} GiB is below the qualified {pf.RAM_WARN_GIB:.0f} GiB "
                           f"(floor ~{need:.0f} GiB; the boot peak is unmeasured)")
        return None
    margin = pf.RAM_WARN_GIB - pf.RAM_FLOOR_GIB  # the qualified allocation's headroom over the FP8 floor
    if have < need + margin:
        return Problem("WARN", "ram", f"{what} {have:.0f} GiB holds the ~{need:.0f} GiB estimate for {fn_ram_parts(ple, tier_gib, sizes)}, "
                       f"but not the {margin:.0f} GiB headroom the qualified {pf.RAM_WARN_GIB:.0f} GiB gives the FP8 table: this "
                       "combination has no capped-container boot record yet (the boot peak is unmeasured)")
    return None


# --------------------------------------------------------------------------------------------
# Recommendation and layouts
# --------------------------------------------------------------------------------------------
def _choose(model: str, inv: pf.Inventory, tp: int | None = None) -> S.Choice | None:
    try:
        return S.choose(S._load(model), model, inv, None, tp)
    except (S.CliError, R.ProfileError):
        return None


THINKINGCAP_LICENCE = ("ThinkingCap's weights are BottleCap AI's, under the PolyForm Small Business 1.0.0 licence plus their personal-use "
                       "permission (not Apache-2.0): models/thinkingcap/licenses/ and the Hugging Face card state the terms; "
                       "downloading them means accepting those terms.")


def recommend(inv: pf.Inventory) -> tuple[str | None, str]:
    """(model, why): Flash-Next when a Flash-Next layout fits the cards and the host has its qualified RAM; else 27B
    when a 27B layout fits; else Flash-Next under its RAM floor (an OOM is expected); else the first model with any
    layout; else None."""
    names = sorted({g.name.replace("NVIDIA ", "") for g in inv.gpus})
    cards = f"{len(inv.gpus)} x {' + '.join(names)}" if inv.gpus else "no GPU"
    ram = inv.host.ram_mib
    ram_s = f"{pf._gib(ram):.0f} GiB RAM" if ram is not None else "RAM unknown"
    # the card fit alone: preflight refuses Flash-Next under the RAM floor, which is the second question here
    fn = _choose("flash-next", replace(inv, host=replace(inv.host, ram_mib=int(pf.RAM_WARN_GIB * 1024))))
    if fn is not None and (ram is None or pf._gib(ram) >= pf.RAM_WARN_GIB):
        return "flash-next", (f"{cards}, P2P {fn.rep.effective_p2p}, {ram_s}: {fn.prof['id']} [{fn.rep.status}] fits"
                              + (" with the qualified RAM" if ram is not None else f" (check for >= {pf.RAM_WARN_GIB:.0f} GiB RAM)"))
    b27 = _choose("27b", inv)
    if b27 is not None:
        if fn is not None:
            why = f"{cards} fit Flash-Next ({fn.prof['id']}) but {ram_s} is below its qualified {pf.RAM_WARN_GIB:.0f} GiB"
        else:
            why = f"{cards} hold no Flash-Next layout (it needs 4 x 24 GB or 2 x 48 GB)"
        return "27b", f"{why}; 27B {b27.prof['id']} [{b27.rep.status}] fits, {ram_s}"
    if fn is not None:
        return "flash-next", (f"{cards} fit {fn.prof['id']} but {ram_s} is below the ~{pf.ram_floor_gib(len(fn.cards)):.0f} GiB "
                              "floor: expect an OOM; 27B does not fit either")
    for m in MODELS:
        if _choose(m, inv) is not None:
            return m, f"{cards}: only {m} has a layout here"
    return None, f"{cards}, {ram_s}: no Merlin layout fits (run `merlin doctor` for the report)"


@dataclass
class Layout:
    label: str  # "default", "--tp 2", "--instances 2", "--tp 1 --instances 4"
    tp: int | None
    instances: int | None
    text: str
    recommended: bool = False
    why: str = ""


def _profile_tp(p: dict[str, Any]) -> int | None:
    return (p.get("layout") or {}).get("tp", p["serve"].get("tensor_parallel"))


def layouts(model: str, inv: pf.Inventory, repo: Path = REPO) -> list[Layout]:
    """What this host offers for ``model``: serve's own pick first, then every other tensor-parallel size whose row
    resolves (``--tp`` takes the row's real TP, not its card count), then every ``--instances`` count that partitions.
    One option carries ``recommended``: serve's pick, except a single-card 27B pick on a multi-card class without P2P,
    where setup recommends ``--tp 1 --instances N`` over all idle cards (the measured aggregate advice; serve's own
    default is unchanged)."""
    out: list[Layout] = []
    seen: set[str] = set()
    try:
        d = S.resolve(model, None, None, inv, [], repo, {})
    except S.CliError:
        return out
    out.append(Layout("default", None, None, f"{d.profile_id} [{d.status}] on GPUs {','.join(map(str, d.cards))}", True,
                      "what merlin serve picks"))
    seen.add(d.profile_id)
    reg = S._load(model)
    tps = sorted({t for t in (_profile_tp(p) for p in S._profiles_for(model, reg)) if t})
    for tp in tps:
        try:
            p = S.resolve(model, None, None, inv, [], repo, {}, tp=tp)
        except S.CliError:
            continue
        if p.profile_id in seen:
            continue
        seen.add(p.profile_id)
        out.append(Layout(f"--tp {tp}", tp, None, f"{p.profile_id} [{p.status}] on GPUs {','.join(map(str, p.cards))}"))
    for n in range(2, len(inv.gpus) + 1):
        try:
            plans = S.plan_instances(model, None, None, inv, [], n, repo, {})
        except S.CliError:
            continue
        sets = " / ".join(",".join(map(str, p.cards)) for p in plans)
        ports = f"ports {plans[0].env.get('PORT')}-{plans[-1].env.get('PORT')}"
        tp1 = all(len(p.cards) == 1 for p in plans)
        out.append(Layout(f"--tp 1 --instances {n}" if tp1 else f"--instances {n}", 1 if tp1 else None, n,
                          f"{n} x {plans[0].profile_id} on GPUs {sets}, {ports}"))
    # the no-P2P single-card case: all idle cards of the class, each its own server
    if FAMILY[model] == "27b" and len(d.cards) == 1:
        first = next(g for g in inv.gpus if g.index == d.cards[0])
        cls = [g for g in inv.gpus if S._class_of(g) == S._class_of(first)]
        idle = S._idle(cls)
        if len(idle) > 1 and inv.p2p_among([g.index for g in idle]) != "on":
            pick = next((o for o in out if o.instances == len(idle) and o.tp == 1), None)
            if pick is not None:
                out[0].recommended, out[0].why = False, ""
                pick.recommended, pick.why = True, NOP2P_TP1_WHY.format(n=len(idle), g=d.cards[0])
    return out


def serve_plan(a: Answers, inv: pf.Inventory, repo: Path = REPO) -> list[S.Plan]:
    env = serve_env(a)
    if a.context is not None:
        env.pop("MAXLEN", None)  # explicit --context replaces the inherited limit
    if fn_family(a.model):  # the table the served folder will carry (an INT6 overlay may not be built yet)
        inv = replace(inv, ple_table_gib=0.0 if a.ple_store == "disk" else PLE_TABLES[a.ple_table or PLE_DEFAULT].gib)
    # the layout is resolved with the KV host tier off: ram_problem counts the tier (and the table) itself and names the
    # fixes, which serve's tier-aware RAM gates would otherwise pre-empt with a bare layout refusal
    env.pop(KV_TIER_ENV, None)
    for k in L.FN_KV_HOST_TIER_NEEDS_ON:  # refused with the tier off; serve checks them against the real size
        env.pop(k, None)
    if a.instances and a.instances > 1:
        plans = S.plan_instances(a.model, str(a.checkpoint_dir()), None, inv, [], a.instances, repo, env, tp=a.tp)
    else:
        plans = [S.resolve(a.model, str(a.checkpoint_dir()), None, inv, [], repo, env, tp=a.tp)]
    # Setup may not have downloaded config.json.
    if a.context is not None:
        for plan in plans:
            if plan.vllm is not None:
                try:
                    flags = L._flag_values(plan.vllm.args + plan.vllm.extra)
                    plan.notes.append(L.C.pool_note(plan.vllm, a.context, flags))
                except ValueError as e:
                    raise SetupError(str(e), 2) from None
                plan.notes.append("context: YaRN recipe resolves from config.json at serve time")
    return plans


def serve_env(a: Answers) -> dict[str, str]:
    env = compiler_env(a, dict(os.environ))
    env["HOST"] = a.host
    if a.port is not None:
        env["PORT"] = str(a.port)
    if a.api_key:
        env["VLLM_API_KEY"] = a.api_key
    if tier_family(a.model):  # the tier comes from the answers (serve's --kv-ram-cache), never from a stale export
        env.pop(KV_TIER_ENV, None)
        env.pop(L.SWARM_ENV, None)  # so does the swarm cap (serve's --swarm)
    if fn_family(a.model):
        env["PLE_STORE"] = a.ple_store or "ram"  # the answer, never a stale export
        env["SPEC"] = a.spec or SPEC_DEFAULT  # the resolved choice wins over an ambient SPEC (serve_plan resolves from this env)
        if a.spec == "dflash2":
            env["DFLASH2_MODEL"] = str(drafter_dir())  # the copy setup downloaded and verified
    env["MERLIN_HOME"] = str(REPO)
    return env


def serve_argv(a: Answers, repo: Path = REPO) -> list[str]:
    argv = [str(repo / "venv" / "bin" / "python"), str(repo / "merlin_cli.py"), "serve", a.model or "", str(a.checkpoint_dir())]
    if a.tp:
        argv += ["--tp", str(a.tp)]
    if a.instances and a.instances > 1:
        argv += ["--instances", str(a.instances)]
    if fn_family(a.model) and is_variant(a.ple_table):
        argv += ["--ple", a.ple_table]
    if fn_family(a.model) and a.ple_store == "disk":
        argv += ["--ple-store", "disk"]
    if tier_family(a.model) and a.kv_ram_cache_gib:
        argv += ["--kv-ram-cache", f"{a.kv_ram_cache_gib}g"]
    if a.context is not None:
        argv += ["--context", str(a.context)]
    if swarm_family(a.model) and a.swarm:
        argv += ["--swarm"]
    if fn_family(a.model):
        argv += ["--spec", a.spec or SPEC_DEFAULT]
    return argv


def serve_words(a: Answers, repo: Path = REPO) -> list[str]:
    """The serve command's shell words (environment first; the key is read from its file, never printed)."""
    pre = [f"HOST={a.host}", f"PORT={a.port}"] + (["VLLM_API_KEY=\"$(cat .merlin/api-key)\""] if a.api_key else [])
    if fn_family(a.model) and a.spec == "dflash2":
        pre.append(f"DFLASH2_MODEL={shlex.quote(str(drafter_dir()))}")
    return pre + [shlex.quote(x) for x in serve_argv(a, repo)]


def default_port(model: str) -> int:
    return S._default_port(S._load(model))


def served_name(model: str) -> str:
    reg = S._load(model)
    if FAMILY[model] == "flash-next":
        ck = reg.get("checkpoints", {}).get(model, {})
        return (ck.get("served_names") or reg["base"]["served_names"]).split()[0]
    if FAMILY[model] in ("svl", "ornith"):
        return reg["launch"]["served"]
    return reg["launch"]["checkpoints"][model]["served"]


# --------------------------------------------------------------------------------------------
# Install state and the verified-checkpoint marker
# --------------------------------------------------------------------------------------------
def head_commit(repo: Path, run: Callable[[list[str]], str | None] = default_runner) -> str | None:
    out = run(["git", "-C", str(repo), "rev-parse", "HEAD"])
    return out.strip() if out else None


def install_state(repo: Path, from_source: bool, commit: str | None) -> dict[str, Any]:
    sha, name = wheel_record(repo)
    return {"wheel_sha": "source" if from_source else sha, "wheel": name, "lock_sha": lock_hash(repo), "commit": commit,
            "venv": str(repo / "venv")}


def venv_current(repo: Path, want: dict[str, Any], saved: dict[str, Any] | None) -> bool:
    """The recorded install matches this checkout (wheel sha, lock hash, commit) and the venv is still there."""
    if not saved:
        return False
    if any(saved.get(k) != want[k] for k in ("wheel_sha", "lock_sha", "commit", "venv")):
        return False
    venv = Path(want["venv"])
    return (venv / "bin" / "vllm").exists() and (venv / "bin" / "python").exists()


def marker_data(ckpt_dir: Path, model: str, sums: dict[str, str], list_sha: str,
                stats: dict[str, tuple[int, int]] | None = None) -> dict[str, Any]:
    """The verified marker. ``stats`` (name -> (size, mtime_ns)) must be taken before the files were hashed, so a file
    replaced during verification is never certified with its new size and mtime; without it the files are stat'ed now."""
    files = {}
    for name in sorted(sums):
        if stats is not None and name in stats:
            files[name] = list(stats[name])
        else:
            st = (ckpt_dir / name).stat()
            files[name] = [st.st_size, st.st_mtime_ns]
    return {"model": model, "list_sha256": list_sha, "files": files, "verified_at": int(time.time())}


def marker_ok(ckpt_dir: Path, model: str, sums: dict[str, str], list_sha: str) -> bool:
    """True when the marker records this model and checksum list and every file still has the recorded size and
    mtime. A trade-off for speed (a 115 GiB re-hash takes minutes): same-size, same-mtime corruption passes until
    ``--rehash``; a changed list, file set, size or mtime re-hashes."""
    m = read_json(ckpt_dir / VERIFIED_MARKER)
    files = m.get("files") if m else None
    if not m or m.get("model") != model or m.get("list_sha256") != list_sha or not isinstance(files, dict) or set(files) != set(sums):
        return False
    for name, rec in files.items():
        if not (isinstance(rec, (list, tuple)) and len(rec) == 2):
            return False  # a malformed record re-hashes
        size, mtime = rec
        try:
            st = (ckpt_dir / name).stat()
        except OSError:
            return False
        if st.st_size != size or st.st_mtime_ns != mtime:
            return False
    return True


def verify_checkpoint(ckpt_dir: Path, list_path: Path, say: Callable[[str], None] = print, model: str = "",
                      rehash: bool = False) -> tuple[bool, list[str]]:
    """Hash every listed file (skipped when the marker is current, unless ``rehash``). (ok, missing-or-wrong files)."""
    problem = checksum_list_problem(list_path)
    if problem:
        return False, [problem]
    sums, _ = parse_checksums(list_path)
    list_sha = sha256_file(list_path)
    if not rehash and marker_ok(ckpt_dir, model, sums, list_sha):
        say(f"checkpoint {ckpt_dir}: verified earlier ({len(sums)} files unchanged; --rehash hashes them again)")
        return True, []
    bad = []
    stats: dict[str, tuple[int, int]] = {}
    for i, (name, want) in enumerate(sorted(sums.items()), 1):
        p = ckpt_dir / name
        if not p.is_file():
            bad.append(f"{name}: missing")
            continue
        before = p.stat()  # stat before hashing; the marker records this, and the file is re-checked afterwards
        say(f"  hashing {i}/{len(sums)} {name} ({gib(before.st_size)})")
        if sha256_file(p) != want:
            bad.append(f"{name}: sha256 differs")
            continue
        try:
            after = p.stat()
        except OSError:
            bad.append(f"{name}: changed while it was being verified")
            continue
        if (after.st_size, after.st_mtime_ns) != (before.st_size, before.st_mtime_ns):
            bad.append(f"{name}: changed while it was being verified")
            continue
        stats[name] = (before.st_size, before.st_mtime_ns)
    if bad:
        try:
            (ckpt_dir / VERIFIED_MARKER).unlink()
        except OSError:
            pass
        return False, bad
    try:
        write_json(ckpt_dir / VERIFIED_MARKER, marker_data(ckpt_dir, model, sums, list_sha, stats))
    except OSError as e:
        if e.errno not in (errno.EROFS, errno.EACCES):
            raise
        say(f"checkpoint {ckpt_dir}: verified, but {VERIFIED_MARKER} cannot be written there ({e.strerror}); later checks will rehash")
    return True, []


def checkpoint_missing(ckpt_dir: Path, list_path: Path) -> list[str]:
    sums, _ = parse_checksums(list_path)
    return [n for n in sums if not (ckpt_dir / n).is_file()]


def folder_conflict(ckpt_dir: Path, model: str, repo: Path = REPO) -> str | None:
    """Why ``ckpt_dir`` must not receive the FP8 checkpoint of ``model``: it holds another checkpoint (its config.json
    or verified marker belongs to another model, or to no known revision) or an INT6 overlay. None when the folder is
    empty, missing, or this checkpoint's."""
    if not ckpt_dir.is_dir():
        return None
    if (ckpt_dir / PP.OVERLAY_MARKER).is_file():
        return "holds an INT6 overlay (links over another folder), not a checkpoint"
    key = model
    m = read_json(ckpt_dir / VERIFIED_MARKER)
    if m and m.get("model") and m["model"] != key:
        return f"holds a verified {m['model']} checkpoint"
    cfg = ckpt_dir / "config.json"
    if cfg.is_file():
        digest = sha256_file(cfg)
        lists = [(o, checksum_list(o, repo)) for o in MODELS]
        owners = [o for o, lst in lists if lst.is_file() and parse_checksums(lst)[0].get("config.json") == digest]
        if key in owners:
            return None
        if owners:
            return f"holds a {' / '.join(owners)} checkpoint (its config.json)"
        return "holds a checkpoint that is not this release's revision of any Merlin model (its config.json matches none)"
    return None


# --------------------------------------------------------------------------------------------
# The INT6 PLE pack and the overlay
# --------------------------------------------------------------------------------------------
def overlay_sources(model: str, base: Path, pack: Path, repo: Path = REPO) -> PP.Sources:
    lst, plst = checksum_list(model, repo), pack_list(repo)
    return PP.Sources(model=model, base=base, pack=pack, base_files=sorted(parse_checksums(lst)[0]),
                      base_list_sha256=sha256_file(lst), pack_list_sha256=sha256_file(plst),
                      pack_files=sorted(parse_checksums(plst)[0]))


def pack_marker_ok(pack: Path, repo: Path = REPO) -> bool:
    plst = pack_list(repo)
    return plst.is_file() and marker_ok(pack, PP.PACK_MARKER_MODEL, parse_checksums(plst)[0], sha256_file(plst))


def overlay_reason(model: str, out: Path, repo: Path = REPO) -> str | None:
    """Why the INT6 overlay ``out`` cannot be served as ``model`` (None when it can): not an overlay, built for another
    model, its base or pack not verified (current marker), or stale against them (ple_pack.overlay_problems)."""
    m = PP.read_overlay_marker(out)
    if m is None:
        return f"{out} is not an INT6 overlay (no {PP.OVERLAY_MARKER}); setup builds one: ./setup.sh --model {model} --ple int6"
    if m.get("model") != model:
        return f"{out} is the INT6 overlay of {m.get('model')}, not {model}"
    base, pack = Path(str(m.get("base"))), Path(str(m.get("pack")))
    fix = f"merlin pull {model} --models-dir {shlex.quote(str(base))} --ple int6 rebuilds it"
    lst = checksum_list(model, repo)
    if not marker_ok(base, model, parse_checksums(lst)[0], sha256_file(lst)):
        return f"its FP8 base {base} is not verified as {model} (no current marker); {fix}"
    if not pack_marker_ok(pack, repo):
        return f"its INT6 pack {pack} is not verified (no current marker); {fix}"
    probs = PP.overlay_problems(out, overlay_sources(model, base, pack, repo))
    if probs:
        return f"{out} is stale ({'; '.join(probs[:3])}); {fix}"
    return None


def pack_item(a: Answers, repo: Path, table: dict[str, Any], fetch: Callable[[str], bytes] | None,
              online: bool) -> tuple[Download, bool, int | None]:
    """The pack's plan line: (item, needs download, remaining bytes or None when unknown)."""
    d, e = a.pack_dir(), pack_entry(table) or {}
    sums, rev = parse_checksums(pack_list(repo))
    missing = [n for n in sums if not (d / n).is_file()]
    what = "INT6 PLE pack (Flash-Next family)"
    if not missing:
        return Download(what, str(d), 0, "models", "present; verified before the overlay is built"), False, 0
    if e.get("pending_publication") or not e.get("repo"):
        return (Download(what, f"local copy only -> {d}", None, "models", pack_not_published_msg(d)), True, None)
    rev = rev or e.get("revision")
    present = sum((d / n).stat().st_size for n in sums if (d / n).is_file())
    remaining = max(0, int(e.get("bytes") or 0) - present) if e.get("bytes") else None
    note = f"{e.get('files', len(sums))} files, {gib(e.get('bytes'))} in all, once for every family member using {d}"
    return Download(what, f"hf download {e['repo']} --revision {rev} -> {d}", remaining, "models", note), True, remaining


def pack_not_published_msg(d: Path) -> str:
    return (f"the INT6 PLE pack is not on Hugging Face yet; put a local copy (the files {PP.PACK_LIST} lists) in {d} "
            f"({PP.PACK_DIR_ENV}=<folder> names another place)")


def pull_pack(repo: Path, a: Answers, env: dict[str, str], table: dict[str, Any], rehash: bool = False,
              runner: Callable[..., int] | None = None) -> None:
    """--ple int6: the pack into its shared folder (once), verified against models/flash-next/ple-int6-pack.sha256."""
    runner = runner or sh
    d, e, lst = a.pack_dir(), pack_entry(table) or {}, pack_list(repo)
    if checkpoint_missing(d, lst):
        if e.get("pending_publication") or not e.get("repo"):
            raise SetupError(pack_not_published_msg(d))
        _, rev = parse_checksums(lst)
        rev = rev or e.get("revision")
        if not rev:
            raise SetupError("the INT6 PLE pack has no pinned revision (record it in release/download-sizes.json); nothing downloaded")
        hf = hf_tool(repo, env, runner)
        d.mkdir(parents=True, exist_ok=True)
        rc = hf_download(hf, e["repo"], rev, d, env, runner)
        if rc != 0:
            raise SetupError(f"hf download of the INT6 PLE pack failed (rc {rc}); run again: finished files stay, partial ones start over")
    ok, bad = verify_checkpoint(d, lst, model=PP.PACK_MARKER_MODEL, rehash=rehash)
    if not ok:
        raise SetupError(f"INT6 PLE pack verification failed in {d}:\n  " + "\n  ".join(bad)
                         + "\n  delete the listed files and run again")
    print(f"setup: INT6 PLE pack verified in {d}", flush=True)


def build_int6_overlay(repo: Path, a: Answers) -> Path:
    """The overlay ``<base>-int6`` over the verified base and pack (built, rebuilt when stale, or kept when current)."""
    out = a.checkpoint_dir()
    try:
        how = PP.build_overlay(out, overlay_sources(a.model, a.base_dir(), a.pack_dir(), repo))
    except PP.PackError as e:
        raise SetupError(f"--ple int6: {e}; the FP8 checkpoint is unchanged") from None
    print(f"setup: INT6 overlay {how}: {out} (FP8 {a.base_dir()} + pack {a.pack_dir()})", flush=True)
    return out


# --------------------------------------------------------------------------------------------
# Download plan and disk
# --------------------------------------------------------------------------------------------
@dataclass
class Download:
    what: str
    source: str
    bytes: int | None  # None = unknown, 0 = nothing to fetch
    where: str  # "repo" or "models"
    note: str = ""


@dataclass
class DiskNeed:
    path: Path
    bytes: int
    what: str


@dataclass
class DownloadPlan:
    items: list[Download] = field(default_factory=list)
    need_install: bool = False
    need_checkpoint: bool = False
    revision: str | None = None  # the checkpoint revision hf download is pinned to
    disk: list[DiskNeed] = field(default_factory=list)

    def known_total(self) -> int:
        return sum(d.bytes or 0 for d in self.items)

    def unknown(self) -> int:
        return sum(1 for d in self.items if d.bytes is None)

    def pending(self) -> bool:
        return any(d.bytes is None or d.bytes > 0 for d in self.items)


@dataclass
class HfInfo:
    bytes: int
    files: int
    sha: str | None
    sizes: dict[str, int]


def hf_info(repo_id: str, revision: str | None, fetch: Callable[[str], bytes] | None = None, timeout: float = 10) -> HfInfo | None:
    """Sizes, file count and resolved commit of a Hugging Face repository (its API); None when unreachable or gated
    without HF_TOKEN."""
    url = f"https://huggingface.co/api/models/{repo_id}" + (f"/revision/{revision}" if revision else "") + "?blobs=true"
    try:
        if fetch is None:
            req = urllib.request.Request(url, headers={"User-Agent": "merlin-setup"})
            tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
            if tok:
                req.add_header("Authorization", f"Bearer {tok}")
            with urllib.request.urlopen(req, timeout=timeout) as r:
                raw = r.read()
        else:
            raw = fetch(url)
        d = json.loads(raw)
        sib = d.get("siblings") or []
        sizes = {s["rfilename"]: int((s.get("lfs") or {}).get("size") or s.get("size") or 0) for s in sib}
        total = sum(sizes.values())
        return HfInfo(total, len(sib), d.get("sha"), sizes) if total else None
    except Exception:  # noqa: BLE001 - any network / JSON failure means "use the table"
        return None


def checkpoint_item(model: str, ckpt_dir: Path, repo: Path, table: dict[str, Any], fetch: Callable[[str], bytes] | None,
                    online: bool) -> tuple[Download, bool, str | None, int | None]:
    """The checkpoint's plan line: (item, needs download, pinned revision, remaining bytes or None when unknown)."""
    what = f"checkpoint {model}"
    list_path = checksum_list(model, repo)
    ck = ck_entry(model, table)
    sums, _ = parse_checksums(list_path)
    missing = [n for n in sums if not (ckpt_dir / n).is_file()] if ckpt_dir.is_dir() else list(sums)
    rev = pinned_revision(model, repo, table)
    if not missing:
        return Download(what, str(ckpt_dir), 0, "models", "present; verified before serving"), False, rev, 0
    if pending_publication(model, repo, table):
        return (Download(what, f"local copy only -> {ckpt_dir}", None, "models", not_published_msg(model)),
                True, None, None)
    info = hf_info(ck["repo"], rev, fetch) if online else None
    if rev is None and info is not None and info.sha:
        rev = info.sha  # a repository without a recorded revision: pinned to the commit the API resolves now
    if info is not None:
        remaining: int | None = sum(info.sizes.get(n, 0) for n in missing)  # the listed files still to fetch
        n_bytes, n_files, src = info.bytes, info.files, "Hugging Face API"
    elif ck.get("bytes"):
        present = sum((ckpt_dir / n).stat().st_size for n in sums if (ckpt_dir / n).is_file()) if ckpt_dir.is_dir() else 0
        remaining = max(0, ck["bytes"] - present)
        n_bytes, n_files, src = ck["bytes"], ck.get("files"), "release/download-sizes.json"
    else:
        remaining, n_bytes, n_files, src = None, None, ck.get("files"), None
    if n_bytes is None:
        note = "size unknown (the Hugging Face API did not answer)"
    else:
        note = f"{n_files} files, {gib(n_bytes)} in all, size from {src}"
    if len(missing) < len(sums):
        note += f"; {len(missing)} of {len(sums)} listed files still missing in {ckpt_dir} (the download fetches them; finished files stay)"
    cmd = f"hf download {ck['repo']}" + (f" --revision {rev}" if rev else " (no pinned revision yet)") + f" -> {ckpt_dir}"
    return Download(what, cmd, remaining, "models", note), True, rev, remaining


def download_plan(a: Answers, repo: Path, state: dict[str, Any] | None, commit: str | None,
                  fetch: Callable[[str], bytes] | None = None, online: bool = True,
                  table: dict[str, Any] | None = None, engine: bool = True) -> DownloadPlan:
    """What a run fetches and the disk it needs (per path; ``disk_problems`` groups by filesystem). ``engine=False``
    plans the checkpoint only (merlin pull)."""
    table = table or sizes_table(repo)
    plan = DownloadPlan()
    sha, wheel = wheel_record(repo)
    venv = repo / "venv"
    factor = table.get("venv_factor", 3)
    pk = table["packages"]
    if engine and not venv_current(repo, install_state(repo, a.from_source, commit), state):
        plan.need_install = True
        if a.from_source:
            plan.items.append(Download("engine source build", "compiles csrc/ (about 1.5 h on 8 cores)", 0, "repo"))
        else:
            local = os.environ.get("MERLIN_WHEEL")
            cached = repo / ".merlin" / "downloads" / wheel
            if local and Path(local).is_file():
                plan.items.append(Download("engine wheel", f"MERLIN_WHEEL={local}", 0, "repo", "local file, no download"))
            elif cached.is_file() and sha256_file(cached) == sha:
                plan.items.append(Download("engine wheel", str(cached), 0, "repo", "already downloaded"))
            else:
                plan.items.append(Download("engine wheel", wheel_urls(repo)[0], table["wheel"]["bytes"], "repo"))
                plan.disk.append(DiskNeed(repo / ".merlin" / "downloads", table["wheel"]["bytes"], "the wheel download"))
        plan.items.append(Download(f"Python packages ({pk['count']} from release/{RELEASE_LOCK})", "PyPI via uv (cached after the first install)",
                                   pk["bytes"], "repo", f"about {gib(pk['bytes'] * factor)} installed"))
        plan.disk.append(DiskNeed(uv_cache_dir(), pk["bytes"], "uv's package cache (up to; less when it already holds them)"))
        plan.disk.append(DiskNeed(venv, pk["bytes"] * factor, "the new venv"))
        # the previous venv is moved aside into .merlin (a rename on the same filesystem: it already occupies its space),
        # so it adds a need only when .merlin is on another device and the move copies it there
        if venv.exists() and device_of(repo / ".merlin") != device_of(venv):
            plan.disk.append(DiskNeed(repo / ".merlin", pk["bytes"] * factor, "the previous venv, kept until the new one verifies"))
    item, need, rev, remaining = checkpoint_item(a.model, a.base_dir(), repo, table, fetch, online)
    plan.items.append(item)
    plan.need_checkpoint, plan.revision = need, rev
    if need and remaining:
        plan.disk.append(DiskNeed(a.base_dir(), remaining, f"the rest of the {a.model} checkpoint"))
    if fn_family(a.model) and is_variant(a.ple_table) and not int6_reason(a.model, repo, table):
        item, need, remaining = pack_item(a, repo, table, fetch, online)
        plan.items.append(item)
        if need and remaining:
            plan.disk.append(DiskNeed(a.pack_dir(), remaining, "the INT6 PLE pack"))
        plan.items.append(Download("INT6 overlay", f"links -> {a.checkpoint_dir()}", 0, "models",
                                   "links to the FP8 checkpoint and the pack + a generated config.json and index; the FP8 files are not changed"))
    if fn_family(a.model) and a.spec == "dflash2":
        d, ck = drafter_dir(), dflash2_entry(table)
        if not dflash2_present(d, repo):
            plan.items.append(Download("drafter DFlash2", f"hf download {ck['repo']} --revision {ck['revision']} -> {d}", ck["bytes"],
                                       "models", f"{ck['files']} files; config.json and model.safetensors checked ({DFLASH2_LIST})"))
            plan.disk.append(DiskNeed(d, ck["bytes"], "the DFlash2 drafter"))
        else:
            plan.items.append(Download("drafter DFlash2", str(d), 0, "models", "present; verified before serving"))
    return plan


DFLASH2_LIST = "models/flash-next/dflash2.sha256"


def dflash2_entry(table: dict[str, Any]) -> dict[str, Any]:
    """release/download-sizes.json's pinned DFlash2 drafter (repo, revision, files, bytes)."""
    ck = (table.get("drafters") or {}).get("dflash2")
    if not ck or not ck.get("revision"):
        raise SetupError("release/download-sizes.json has no pinned drafters.dflash2 entry; --spec dflash2 cannot download it")
    return ck


def drafter_dir() -> Path:
    return S.dflash2_dir()


def dflash2_present(d: Path, repo: Path = REPO) -> bool:
    return d.is_dir() and not checkpoint_missing(d, repo / DFLASH2_LIST)


def pull_drafter(repo: Path, a: Answers, env: dict[str, str], table: dict[str, Any], rehash: bool = False,
                 runner: Callable[..., int] | None = None) -> None:
    """--spec dflash2: the drafter at its pinned revision, verified against models/flash-next/dflash2.sha256."""
    if not (fn_family(a.model) and a.spec == "dflash2"):
        return
    runner = runner or sh
    d, ck, lst = drafter_dir(), dflash2_entry(table), repo / DFLASH2_LIST
    if checkpoint_missing(d, lst):
        hf = hf_tool(repo, env, runner)
        d.mkdir(parents=True, exist_ok=True)
        rc = hf_download(hf, ck["repo"], ck["revision"], d, env, runner)
        if rc != 0:
            raise SetupError(f"hf download of the DFlash2 drafter failed (rc {rc}); run again: finished files stay, partial ones start over")
    ok, bad = verify_checkpoint(d, lst, model="dflash2-drafter", rehash=rehash)
    if not ok:
        raise SetupError(f"DFlash2 drafter verification failed in {d}:\n  " + "\n  ".join(bad)
                         + "\n  delete the listed files and run again (hf download fetches them anew)")
    print(f"setup: DFlash2 drafter verified in {d}", flush=True)


def dflash2_qualified(a: Answers, inv: pf.Inventory, repo: Path = REPO) -> bool:
    """A Flash-Next-family layout whose every server row has a qualified DFlash2 geometry (serve.dflash2)."""
    if not fn_family(a.model):
        return False
    try:
        plans = serve_plan(a, inv, repo)
    except (S.CliError, R.ProfileError):
        return False
    rows = set(L.dflash2_rows(R.load_registry("flash-next")))
    return bool(plans) and all(p.profile_id in rows for p in plans)


def disk_problems(plan: DownloadPlan, free: Callable[[Path], int | None] | None = None,
                  dev: Callable[[Path], Any] | None = None) -> list[Problem]:
    """One FAIL per filesystem whose free space is below everything the plan puts there."""
    free, dev = free or disk_free, dev or device_of  # looked up at call time (tests replace them)
    groups: dict[Any, list[DiskNeed]] = {}
    for n in plan.disk:
        groups.setdefault(dev(n.path), []).append(n)
    out = []
    for needs in groups.values():
        total = sum(n.bytes for n in needs)
        have = free(needs[0].path)
        if have is not None and have < total:
            parts = ", ".join(f"{n.what} {gib(n.bytes)}" for n in needs)
            where = sorted({str(nearest_existing(n.path)) for n in needs})
            out.append(Problem("FAIL", "disk", f"the filesystem of {' and '.join(where)} has {gib(have)} free; the plan puts {gib(total)} there ({parts})",
                               "free the space" + (" or choose another models folder (--models-dir)" if any("checkpoint" in n.what for n in needs) else "")))
    return out


def short_source(src: str) -> str:
    """A plan line's source as a reader needs it: the release or the repository and revision, not the full URL or
    command (``--dry-run`` and the command a run prints carry those)."""
    m = re.fullmatch(r"https://github\.com/([^/]+/[^/]+)/releases/download/([^/]+)/[^/]+", src)
    if m:
        return f"github.com/{m[1]} release {m[2]}"
    m = re.fullmatch(r"hf download (\S+) --revision ([0-9a-f]{7})[0-9a-f]* -> (.+)", src)
    if m:
        return f"Hugging Face {m[1]} @ {m[2]} -> {m[3]}"
    m = re.fullmatch(r"hf download (\S+) (\(no pinned revision yet\)) -> (.+)", src)
    if m:
        return f"Hugging Face {m[1]} {m[2]} -> {m[3]}"
    return src


def render_plan(plan: DownloadPlan, detail: bool = False, color: bool = False, width: int = WIDTH) -> str:
    """The download table: item and size, the source and note on dimmer lines below, a total under a rule.
    ``detail`` (--dry-run) adds the full URL / command under a shortened source."""
    if not plan.pending():
        return "Nothing to download: the engine and the checkpoint are in place.\n"

    def sgr(t: str, code: str) -> str:
        return f"\033[{code}m{t}\033[0m" if color and t else t

    sizes = [gib(d.bytes) if d.bytes else ("-" if d.bytes == 0 else "size unknown") for d in plan.items]
    total = gib(plan.known_total()) + (f" + {plan.unknown()} of unknown size" if plan.unknown() else "")
    iw = max(len(x) for x in [d.what for d in plan.items] + ["total"])
    sw = max(len(x) for x in sizes + [total])
    lines = [sgr("Downloads", "1")]
    for d, size in zip(plan.items, sizes):
        lines.append(f"  {d.what:<{iw}}  {size:>{sw}}")
        src = short_source(d.source)
        info = [src] + ([d.note] if d.note else [])
        joined = "; ".join(info)
        for part in [joined] if len(joined) + 4 <= width else info:
            lines += [sgr(x, "2") for x in wrap(part, width, "    ")]
        if detail and src != d.source:  # whole, on one line: a URL or command is copied, not read
            lines.append(sgr(f"    {d.source}", "2"))
    lines.append("  " + "-" * (iw + 2 + sw))
    lines.append(f"  {'total':<{iw}}  {total:>{sw}}")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------------------------
# Actions
# --------------------------------------------------------------------------------------------
def sh(cmd: list[str], env: dict[str, str] | None = None, cwd: Path | None = None) -> int:
    print("+ " + " ".join(shlex.quote(c) for c in cmd), flush=True)
    return subprocess.call(cmd, env=env, cwd=cwd)


def fetch_wheel(urls: list[str], dest: Path, sha: str, runner: Callable[[list[str]], int] = sh) -> Path:
    """The first URL that answers, resumed into ``dest.part``, checked against ``sha`` (a mismatch is deleted)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and sha256_file(dest) == sha:
        return dest
    part = dest.with_suffix(dest.suffix + ".part")
    rc = 1
    for url in urls:
        rc = runner(["curl", "-fL", "--retry", "3", "-C", "-", "-o", str(part), url])
        if rc == 0:
            break
        print(f"setup: {url}: not available (rc {rc})" + ("; trying the next name" if url != urls[-1] else ""), flush=True)
    if rc != 0:
        raise SetupError("wheel download failed from every URL:\n  " + "\n  ".join(urls)
                         + "\n  MERLIN_WHEEL=<file> or MERLIN_WHEEL_URL=<url> override it")
    got = sha256_file(part)
    if got != sha:
        part.unlink()
        raise SetupError(f"downloaded wheel sha256 {got} differs from release/merlin-wheel.sha256 ({sha}); deleted, not installed")
    os.replace(part, dest)
    return dest


_LOCKS_HELD: set[str] = set()  # lock files this process holds (install_lock is re-entrant within one process)


def _try_lock(fd: int) -> bool:
    """An exclusive, non-blocking lock on ``fd``; False when another open file holds it. ``flock`` on Linux (the
    kernel drops it when the holder exits, however it exits); a byte lock elsewhere, so the contract is the same."""
    try:
        import fcntl
    except ImportError:  # not Linux: only the CPU tests get here
        import msvcrt
        try:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        except OSError:
            return False
        return True
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return False
    return True


@contextlib.contextmanager
def install_lock(merlin_dir: Path):
    """Hold ``.merlin/setup.lock`` exclusively, or fail fast: two overlapping installs would move each other's venv
    aside and delete each other's rollback copy. Re-entrant within one process."""
    path = merlin_dir / SETUP_LOCK
    key = str(path.resolve())
    if key in _LOCKS_HELD:
        yield
        return
    merlin_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        if not _try_lock(fd):
            holder = ""
            try:
                with open(path, encoding="utf-8") as f:
                    pid = f.read().strip()
                holder = f", held by pid {pid}" if pid.isdigit() else ""
            except OSError:
                pass
            raise SetupError(f"another ./setup.sh is installing the engine in this checkout ({path} is locked{holder}); "
                             "wait for it to finish, then run ./setup.sh again", 1)
        os.ftruncate(fd, 0)
        os.write(fd, f"{os.getpid()}\n".encode())
        _LOCKS_HELD.add(key)
        try:
            yield
        finally:
            _LOCKS_HELD.discard(key)
    finally:
        os.close(fd)  # closing the descriptor releases the lock


def install_engine(repo: Path, a: Answers, env: dict[str, str], runner: Callable[..., int] = sh) -> None:
    """install-env.sh into ``venv`` then ``--verify``, holding ``.merlin/setup.lock`` throughout. An existing venv is
    moved aside first (the editable install binds the venv to its path, so the new one must take that path), and
    comes back when the install or the verify fails: a failed update never leaves the host without its working
    engine. Success removes only this run's old venv; a ``venv-old-*`` an interrupted earlier run left is named,
    not deleted (it may be someone's only rollback copy)."""
    with install_lock(repo / ".merlin"):
        _install_engine_locked(repo, a, env, runner)


def _install_engine_locked(repo: Path, a: Answers, env: dict[str, str], runner: Callable[..., int]) -> None:
    sha, wheel = wheel_record(repo)
    venv = repo / "venv"
    old_dir = repo / ".merlin"
    old = None
    if venv.exists():
        old = old_dir / f"venv-old-{time.strftime('%Y%m%d-%H%M%S')}"
        print(f"setup: moving the working venv aside -> {old} (restored if the install fails, removed when it verifies)", flush=True)
        old_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(venv), str(old))
    env = compiler_env(a, env)
    env.pop("MERLIN_WHEEL", None)
    t0 = time.time()
    try:
        if not a.from_source:
            local = os.environ.get("MERLIN_WHEEL")
            if local and Path(local).is_file():
                path = Path(local).resolve()  # install-env.sh runs from another cwd, so the path must be absolute
            else:
                path = fetch_wheel(wheel_urls(repo), repo / ".merlin" / "downloads" / wheel, sha)
            env["MERLIN_WHEEL"] = str(path)
        rc = runner(["bash", str(repo / "release" / "install-env.sh"), str(venv)], env=env, cwd=repo)
        if rc != 0:
            raise SetupError(f"release/install-env.sh failed (rc {rc})")
        env.pop("MERLIN_WHEEL", None)  # --verify refuses it (an install-only variable)
        rc = runner(["bash", str(repo / "release" / "install-env.sh"), "--verify", str(venv)], env=env, cwd=repo)
        if rc != 0:
            raise SetupError(f"release/install-env.sh --verify failed (rc {rc})")
    except BaseException as e:  # noqa: BLE001 - restore on any failure, Ctrl+C included
        if old is not None:
            shutil.rmtree(venv, ignore_errors=True)
            shutil.move(str(old), str(venv))
            msg = "; the previous venv is back in place, the engine still serves"
        else:
            msg = f"; the partial venv is left at {venv} for diagnosis"
        if isinstance(e, SetupError):
            raise SetupError(str(e) + msg, e.code) from None
        print(f"setup: install interrupted{msg}", file=sys.stderr, flush=True)
        raise
    if old is not None:
        shutil.rmtree(old, ignore_errors=True)
    left = sorted(p.name for p in old_dir.glob("venv-old-*"))
    if left:
        print(f"setup: note: {', '.join(left)} in {old_dir} left by an earlier interrupted install; remove them once "
              "no rollback is needed", flush=True)
    print(f"setup: engine installed and verified in {time.time() - t0:.0f} s -> {venv}", flush=True)


def hf_tool(repo: Path, env: dict[str, str], runner: Callable[..., int] = sh) -> Path:
    """The ``hf`` CLI: the engine venv's, else a small ``.merlin/hf`` environment with the lock's huggingface-hub (so
    ``merlin pull`` works on a host that only downloads)."""
    hf = repo / "venv" / "bin" / "hf"
    if hf.exists():
        return hf
    hf = repo / ".merlin" / "hf" / "bin" / "hf"
    if hf.exists():
        return hf
    if which("uv", env) is None:
        raise SetupError("no `hf` (install the engine first, or run through ./setup.sh so uv is on PATH)")
    pins = {}
    for line in (repo / "release" / RELEASE_LOCK).read_text(encoding="utf-8").splitlines():
        m = re.match(r"^([A-Za-z0-9_.-]+)(?:\[[^\]]*\])?==([^\s\\;]+)", line)
        if m:
            pins[m[1]] = m[2]
    cons = repo / ".merlin" / "constraints.txt"
    write_private(cons, "".join(f"{k}=={v}\n" for k, v in sorted(pins.items())), 0o644)
    venv = repo / ".merlin" / "hf"
    if runner(["uv", "venv", "--python", "3.13", str(venv)], env=env) != 0:
        raise SetupError("uv venv for the hf tool failed")
    if runner(["uv", "pip", "install", "--python", str(venv / "bin" / "python"), "--only-binary", ":all:", "-c", str(cons),
               f"huggingface-hub=={pins['huggingface-hub']}"], env=env) != 0:
        raise SetupError("installing huggingface-hub for the hf tool failed")
    return hf


def clear_partial_downloads(dest: Path) -> int:
    """Delete the ``*.incomplete`` files an interrupted ``hf download --local-dir dest`` left under
    ``dest/.cache/huggingface/download``; returns how many. huggingface-hub 1.x writes each attempt to a name of its
    own and never resumes an earlier one, so a leftover is dead weight that the next run neither reuses nor removes."""
    cache, n, size = dest / ".cache" / "huggingface" / "download", 0, 0
    leftovers = sorted(cache.rglob("*.incomplete")) if cache.is_dir() else []
    for f in leftovers:
        try:
            st = f.stat()
            f.unlink()
        except OSError:
            continue
        n, size = n + 1, size + st.st_size
    if n:
        print(f"setup: removed {n} partial download{'s' if n != 1 else ''} ({gib(size)}) left in {cache} by an "
              "interrupted run; hf download starts those files again", flush=True)
    return n


def hf_download(hf: Path, repo_id: str, revision: str, dest: Path, env: dict[str, str],
                runner: Callable[..., int] = sh) -> int:
    """``hf download repo_id --revision revision --local-dir dest``, with the partial files of an interrupted earlier
    attempt removed before it and, if it fails, after it. ``.merlin-download.lock`` in hf's cache folder serializes
    downloads into one folder for the whole download and its cleanup, so no setup can delete another's partial files:
    a second one fails fast. Where the lock can't be created (a read-only cache folder) nothing is cleaned."""
    cache = dest / ".cache" / "huggingface" / "download"
    fd = None
    mine = False
    try:
        cache.mkdir(parents=True, exist_ok=True)
        fd = os.open(cache / ".merlin-download.lock", os.O_RDWR | os.O_CREAT, 0o600)
        mine = _try_lock(fd)
    except OSError:
        pass
    else:
        if not mine:
            os.close(fd)
            raise SetupError(f"another ./setup.sh is downloading into {dest}; wait for it to finish, then run ./setup.sh again", 1)
    try:
        if mine:
            clear_partial_downloads(dest)
        rc = runner([str(hf), "download", repo_id, "--revision", revision, "--local-dir", str(dest)],
                    env={**env, "HF_HUB_DISABLE_TELEMETRY": "1"})
        if mine and rc != 0:
            clear_partial_downloads(dest)
        return rc
    finally:
        if fd is not None:
            os.close(fd)  # closing the descriptor releases the lock


def pull_checkpoint(repo: Path, a: Answers, env: dict[str, str], revision: str | None, rehash: bool = False,
                    runner: Callable[..., int] = sh) -> None:
    """The FP8 checkpoint (downloaded or reused, verified); with --ple int6 then the pack and the overlay over both."""
    ckpt_dir = a.base_dir()
    int6 = fn_family(a.model) and is_variant(a.ple_table)
    if int6 and int6_reason(a.model, repo):
        raise SetupError(f"{a.model} with the {a.ple_table} PLE table: {int6_reason(a.model, repo)}; nothing downloaded")
    if int6 and PP.overlay_conflict(a.checkpoint_dir()):
        raise SetupError(f"{PP.overlay_conflict(a.checkpoint_dir())}; nothing downloaded. Choose another folder (--models-dir) or remove it.")
    list_path = checksum_list(a.model, repo)
    conflict = folder_conflict(ckpt_dir, a.model, repo)
    if conflict:
        raise SetupError(f"{ckpt_dir} {conflict}; nothing downloaded there. Choose another folder (--models-dir) or remove it.")
    sums, _ = parse_checksums(list_path)
    if checkpoint_missing(ckpt_dir, list_path):
        if pending_publication(a.model, repo, None):
            raise SetupError(not_published_msg(a.model))
        if not revision:
            raise SetupError(f"{a.model} has no pinned revision (record it in release/download-sizes.json when its Hugging Face "
                             "repository is published); nothing downloaded")
        hf = hf_tool(repo, env, runner)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        rc = hf_download(hf, hf_repo(a.model, repo), revision, ckpt_dir, env, runner)
        if rc != 0:
            raise SetupError(f"hf download failed (rc {rc}); run again: finished files stay, partial ones start over")
    ok, bad = verify_checkpoint(ckpt_dir, list_path, model=a.model, rehash=rehash)
    if not ok:
        raise SetupError(f"checkpoint verification failed in {ckpt_dir}:\n  " + "\n  ".join(bad)
                         + "\n  delete the listed files and run again (hf download fetches them anew)")
    print(f"setup: checkpoint verified: {len(sums)} files in {ckpt_dir}", flush=True)
    if int6:
        pull_pack(repo, a, env, sizes_table(repo), rehash, runner)
        build_int6_overlay(repo, a)


# --------------------------------------------------------------------------------------------
# Questions
# --------------------------------------------------------------------------------------------
class Asker:
    def __init__(self, yes: bool, inp: Any = None, out: Any = None):
        self.yes = yes
        self.inp = inp or sys.stdin
        self.out = out or sys.stdout
        try:
            tty = bool(self.out.isatty())
        except (AttributeError, ValueError):
            tty = False
        # bold / dim only on a terminal, never in a pipe or a log; NO_COLOR (any value) and TERM=dumb turn it off
        self.color = tty and "NO_COLOR" not in os.environ and os.environ.get("TERM") != "dumb"
        # tables and commands fit WIDTH (or a narrower terminal); messages wrap only on a terminal: in a pipe or a log
        # each stays one line (grep finds it whole)
        self.width = max(60, min(WIDTH, shutil.get_terminal_size((WIDTH, 24)).columns)) if tty else WIDTH
        self.prose = self.width if tty else 0
        self._blank = True  # the last line written was blank (or nothing is written yet): gap() adds none

    def say(self, text: str = "") -> None:
        self.out.write(text + "\n")
        self.out.flush()
        self._blank = text == "" or text.endswith("\n")

    def style(self, text: str, code: str = "1") -> str:
        """Bold (1) or dim (2) on a terminal; the text unchanged otherwise."""
        return f"\033[{code}m{text}\033[0m" if self.color and text else text

    def gap(self) -> None:
        """One blank line, unless the last line already is one."""
        if not self._blank:
            self.say()

    def section(self, title: str, hint: str = "") -> None:
        self.gap()
        self.say(self.style(title) + (f"  {hint}" if hint else ""))

    def para(self, text: str, first: str = "", rest: str | None = None, code: str = "", head: bool = False) -> None:
        """``text`` wrapped to the width with a hanging indent; ``code`` styles every line, ``head`` bolds the
        prefix word of the first line."""
        lines = wrap(text, self.prose, first, rest)
        word = first.strip()
        for i, line in enumerate(lines):
            if head and i == 0 and word and line.startswith(word):
                self.say(self.style(word) + line[len(word):])
            else:
                self.say(self.style(line, code) if code else line)

    def row(self, key: str, value: str, width: int = 12, code: str = "") -> None:
        """A key / value row (``key`` padded to ``width``), the value wrapped under itself."""
        self.para(value, f"{key:<{width}} ", None, code)

    def note(self, level: str, text: str, fix: str = "") -> None:
        """FAIL / WARN / NOTE: the level, the text wrapped under itself, the fix one step further in."""
        self.para(text, f"{level} ", " " * (len(level) + 1), head=True)
        if fix:
            self.para(f"fix: {fix}", " " * (len(level) + 1), " " * (len(level) + 6))

    def ask(self, prompt: str, default: str, choices: list[str] | None = None) -> str:
        if self.yes:
            return default
        tty = getattr(self.inp, "isatty", lambda: False)()
        self.gap()
        while True:
            self.out.write(f"{prompt} [{default}]: ")
            self.out.flush()
            self._blank = False
            line = self.inp.readline()
            if not line:  # EOF: the default
                self.out.write(default + "\n")
                return default
            v = line.strip() or default
            if choices and v.isascii() and v.isdigit() and len(v) < 6 and 1 <= int(v) <= len(choices):
                v = choices[int(v) - 1]
            if not tty:  # piped answers: show what was taken, as a terminal would
                self.out.write(v + "\n")
            if not choices or v in choices:
                return v
            self.say(f"  choose one of: {', '.join(choices)}")


def layout_label(a: Answers) -> str:
    if a.tp and a.instances and a.instances > 1:
        return f"--tp {a.tp} --instances {a.instances}"
    if a.tp:
        return f"--tp {a.tp}"
    if a.instances and a.instances > 1:
        return f"--instances {a.instances}"
    return "default layout"


def ask_questions(a: Answers, inv: pf.Inventory, ui: Asker, args: argparse.Namespace, repo: Path, table: dict[str, Any],
                  saved: dict[str, Any] | None, key_path: Path, only: set[str] | None = None) -> Answers:
    """Ask every question whose flag is unset; ``only`` limits it to those names (review_answers re-asks one)."""
    def want(name: str) -> bool:
        return only is None or name in only

    if want("model"):
        a = _ask_model(a, inv, ui, args, table, saved)
    if want("layout"):
        a = _ask_layout(a, inv, ui, args, repo)
    a = normalize_fn(a)  # the PLE table and the KV tier are flags only (--ple, --kv-ram-cache): never asked
    if want("drafter"):
        a = _ask_spec(a, inv, ui, args, repo)
    if want("swarm"):
        a = _ask_swarm(a, inv, ui, args, repo)
    if want("folder") and args.models_dir is None:
        a.models_dir = ui.ask("Models folder (an existing download is reused)", a.models_dir or default_dir(a.model))
    if want("folder") and not ui.yes:  # --yes: the folder is in the answers block below
        ui.say(f"  checkpoint -> {a.checkpoint_dir()}")
    if want("folder"):
        a = _ask_ple_store(a, inv, ui, args, repo)
    if want("access"):
        a = _ask_access(a, ui, args, key_path)
    if want("port") and args.port is None:
        p = ui.ask("Port", str(a.port or default_port(a.model)))
        if not (p.isascii() and p.isdigit() and len(p) <= 5) or not 1 <= int(p) <= 65535:
            raise SetupError(f"port {p!r}: 1..65535", 2)
        a.port = int(p)
    if want("context"):
        a = _ask_context(a, inv, ui, args, repo)
    if ui.yes:
        answers_block(a, inv, ui, "Answers", repo=repo)
    return a


def ple_store_reason(a: Answers) -> str | None:
    """Why setup does not offer NVMe for the table these answers read; None when it does. INT6 reads the pack, which
    MERLIN_PLE_PACK_DIR can put on another disk, whether or not the overlay is built yet."""
    return PS.offer_reason(a.pack_dir() if is_variant(a.ple_table) else a.checkpoint_dir())


def reconcile_ple_store(a: Answers, args: argparse.Namespace, ui: Asker) -> Answers:
    """A saved disk answer stays only while the table is still on NVMe (every run, questions or not); a flag always stays."""
    if fn_family(a.model) and args.ple_store is None and a.ple_store == "disk":
        why = ple_store_reason(a)
        if why is not None:
            ui.say(f"  PLE store: RAM; {why}")
            a.ple_store = "ram"
    return a


def _nvme_only_fits(a: Answers, inv: pf.Inventory, args: argparse.Namespace, repo: Path) -> bool:
    """Whether these flags fail with the table in RAM and fit with it on disk: said in the question, never its default."""
    def fails(store: str) -> bool:
        b = replace(a, ple_store=store)
        try:
            plans = serve_plan(b, inv, repo)
        except (S.CliError, R.ProfileError):
            return True  # no layout under this RAM gate
        sizes = [len(p.cards) for p in plans]
        per_rank = kv_ram_cache_per_rank(effective_answers(b, plans, sizes, args), sizes)
        p = ram_problem(a.model, inv, sizes[0], a.ple_table, per_rank, sizes, store=store)
        return p is not None and p.level == "FAIL"

    return fails("ram") and not fails("disk")


def _ask_ple_store(a: Answers, inv: pf.Inventory, ui: Asker, args: argparse.Namespace, repo: Path) -> Answers:
    """The PLE store question: Flash-Next family only, after the folder (the answer depends on where the table is); NVMe
    only (``merlin serve`` warns and runs elsewhere); never with --ple-store; never under --yes, where a saved disk answer
    stays while the table still qualifies, else RAM. Enter means RAM."""
    if not fn_family(a.model) or args.ple_store is not None:
        return a
    why = ple_store_reason(a)
    if why is not None:
        if a.ple_store == "disk" or not ui.yes:
            ui.say(f"  PLE store: RAM; {why}")
        a.ple_store = "ram"
        return a
    if ui.yes:
        return a
    t = PLE_TABLES[a.ple_table or PLE_DEFAULT]
    ui.section("PLE table", "(Enter = RAM)")
    ui.para(f"RAM holds the table in host memory (~{t.gib:.1f} GiB). NVMe (experimental) reads its rows from the model files and frees "
            "that memory, for about 2 % slower decode once the table is cached; text it has not read yet is slower (decode 9-11 %, "
            "and a cold 262K prompt takes 7-53 % longer to its first token).", "  ")
    if _nvme_only_fits(a, inv, args, repo):
        ui.para("These flags would not fit with the table in RAM, and would fit with it on NVMe.", "  ")
    answer = ui.ask("PLE table: RAM or NVMe? (ram/nvme)", "ram", ["ram", "nvme"])
    a.ple_store = "disk" if answer == "nvme" else "ram"
    return a


def _ask_context(a: Answers, inv: pf.Inventory, ui: Asker, args: argparse.Namespace, repo: Path) -> Answers:
    if args.context is not None or ui.yes or not swarm_family(a.model):
        return a
    ui.section("Context", "(Enter = native; pool fit checked before download)")
    try:
        plans = serve_plan(replace(a, context=None), inv, repo)
        profile = plans[0].profile_id
    except (S.CliError, SetupError):
        profile = ""  # the normal pre-download report explains a layout refusal
    ui.say("  native: 262K (262,144), no YaRN")
    for size, factor in ((393216, 1.5), (524288, 2.0), (1048576, 4.0)):
        label = L.C.qualification(a.model, profile, a.spec or "mtp", factor)
        ui.say(f"  {size}: YaRN {factor:g}, {label}")
    ui.say("  1M needs a pool that fits it; current Flash-Next 4x3090 pins do not.")
    value = ui.ask("Context (native, tokens, or binary k/m)", str(a.context) if a.context else "native")
    try:
        a.context = None if value == "native" else L.C.parse_size(value)
    except ValueError as e:
        raise SetupError(str(e), 2) from None
    return a


def _ask_spec(a: Answers, inv: pf.Inventory, ui: Asker, args: argparse.Namespace, repo: Path) -> Answers:
    """The one Drafter question: Flash-Next rows with a qualified DFlash2 geometry only, never under --yes or with
    --spec; elsewhere MTP (the only drafter there)."""
    if not fn_family(a.model) or args.spec is not None:
        return a
    qualified = dflash2_qualified(a, inv, repo)
    if ui.yes or not qualified:
        a.spec = "dflash2" if a.spec == "dflash2" and qualified else SPEC_DEFAULT  # a saved DFlash2 answer stays where it can run
        return a
    ui.section("Drafter", "(Enter = recommended)")
    for i, k in enumerate(S.SPEC_CHOICES, 1):
        first = f"  {i}) {k:<8} "
        ui.para(DFLASH2_TITLES[k], first, " " * len(first))
    a.spec = ui.ask("Drafter", a.spec or SPEC_DEFAULT, list(S.SPEC_CHOICES))
    return a


def swarm_row(a: Answers, inv: pf.Inventory, repo: Path = REPO) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """(profile, swarm cap block) when every server this layout starts is a row that has a qualified swarm cap, else None."""
    if not swarm_family(a.model):
        return None
    try:
        plans = serve_plan(a, inv, repo)
    except (S.CliError, R.ProfileError):
        return None
    profs = [p.vllm.profile for p in plans if p.vllm is not None]
    if not profs or len(profs) != len(plans) or not all(L.swarm_gate(pr) for pr in profs):
        return None
    return profs[0], L.swarm_gate(profs[0])


def _ask_swarm(a: Answers, inv: pf.Inventory, ui: Asker, args: argparse.Namespace, repo: Path) -> Answers:
    """The one Swarm question: asked only on rows that have a swarm cap (and not beside DFlash2, whose cache is
    refused), never under --yes or with --swarm / --no-swarm / an explicit --kv-ram-cache 0; default no. When the questions run
    (first run, --reconfigure) a saved swarm answer stays where its row has a cap and is reset to no where it has none; a
    saved set that skips the questions reaches the plan check instead, which refuses a swarm answer on a row without a cap
    (fails closed: nothing starts at a cap the user did not choose)."""
    if not swarm_family(a.model) or args.swarm is not None or args.kv_ram_cache_gib == 0:
        return a  # a flag decided it, or --kv-ram-cache 0 turned the cache (which the swarm cap needs) off on purpose
    row = swarm_row(a, inv, repo)
    if row is None or a.spec == "dflash2":
        a.swarm = False
        return a
    if ui.yes:
        return a
    prof, gate = row
    cap, need, ranks = int(gate["max_num_seqs"]), int(gate["min_gib_per_rank"]), int(prof["match"]["cards"])
    ui.section("Swarm", "(experimental)")
    ui.para(f"Up to {cap} sessions at once instead of {prof['serve']['max_num_seqs']}. Needs the KV RAM cache at {need * ranks}g of host "
            f"RAM per server ({need} GiB per rank over {ranks} ranks), which setup sets for you unless --kv-ram-cache says more.", "  ")
    a.swarm = ui.ask("Swarm mode? (y/n)", "y" if a.swarm else "n", ["y", "n", "yes", "no"]) in ("y", "yes")
    return a


def flag_rows(a: Answers, inv: pf.Inventory, repo: Path = REPO) -> list[tuple[str, str, str]]:
    """The Flash-Next answers set by flags (or the Drafter question): (name, value, the flag that changes it); for the
    27B and Ornith 1.5 only the KV RAM cache, and only when it is on (untested)."""
    context_rows = [("Context", f"{a.context:,} tokens", "--context SIZE")] if a.context else []
    swarm_rows = context_rows + [("Swarm", "on (experimental)", "--no-swarm for the default session cap")] if swarm_family(a.model) and a.swarm else context_rows
    if not fn_family(a.model):
        rows = []
        if tier_family(a.model) and a.kv_ram_cache_gib:
            label = "untested ⚫ on Ornith 1.5" if FAMILY[a.model] == "ornith" else "experimental"  # 27B
            rows.append(("KV RAM cache", f"{a.kv_ram_cache_gib}g per server ({label})", "--kv-ram-cache 0 to turn it off"))
        return rows + swarm_rows
    spec = "DFlash2" if a.spec == "dflash2" else "MTP"
    if a.spec == "dflash2":
        spec_hint = "--spec mtp for the bigger pool (6x131K)"
    else:
        spec_hint = "--spec dflash2 for faster decode (3x262K)" if dflash2_qualified(a, inv, repo) else ""
    if a.kv_ram_cache_gib:
        cache = f"{a.kv_ram_cache_gib}g per server" + (f" (untested \u26ab above {S.KV_RAM_CACHE_TESTED_GIB}g)"
                                                      if S.kv_ram_cache_untested(a.kv_ram_cache_gib) else " (experimental)")
        cache_hint = "--kv-ram-cache 0 to turn it off"
    elif a.swarm:
        cache, cache_hint = "sized by --swarm", "--kv-ram-cache SIZE to set another size"
    else:
        # The cache is qualified with MTP only: no hint to turn it on beside DFlash2.
        cache, cache_hint = "off", "" if a.spec == "dflash2" else "--kv-ram-cache 24g to keep evicted prefixes in RAM"
    ple = "INT6 g32" if is_variant(a.ple_table) else "FP8"
    store = ("NVMe (experimental)", "--ple-store ram") if a.ple_store == "disk" else ("RAM", "--ple-store disk")
    return [("Drafter", spec, spec_hint), ("KV RAM cache", cache, cache_hint), *swarm_rows,
            ("PLE table", ple, "--ple fp8 for the FP8 table" if is_variant(a.ple_table) else ""),
            ("PLE store", *store)]


def say_flag_rows(a: Answers, inv: pf.Inventory, ui: Asker, repo: Path = REPO) -> None:
    rows = flag_rows(a, inv, repo)
    if not rows:
        return
    vw = max(len(v) for _, v, _ in rows)
    for k, v, hint in rows:
        ui.row(f"  {k}", f"{v:<{vw}}  {hint}".rstrip(), 14)


def answers_block(a: Answers, inv: pf.Inventory, ui: Asker, title: str, hint: str = "", repo: Path = REPO) -> None:
    """The answers as aligned rows (the --yes summary and a saved re-run); for Flash-Next also the drafter, KV RAM
    cache and PLE table rows with the flag that changes each."""
    access = "local (127.0.0.1)" if a.host == "127.0.0.1" else f"network ({a.host})"
    ui.section(title, hint)
    rows = [("model", a.model), ("layout", layout_label(a))]
    rows += [("folder", str(a.checkpoint_dir())), ("access", access + (", API key" if a.api_key else "")), ("port", str(a.port))]
    for k, v in rows:
        ui.row(f"  {k}", v or "", 14 if fn_family(a.model) else 9)
    say_flag_rows(a, inv, ui, repo)


def _ask_model(a: Answers, inv: pf.Inventory, ui: Asker, args: argparse.Namespace, table: dict[str, Any],
               saved: dict[str, Any] | None) -> Answers:
    rec, why = recommend(inv)
    menus = not ui.yes and args.model is None
    if menus:
        ui.section("Model", "(Enter = recommended)")
        sizes = {m: gib(table["checkpoints"][m].get("bytes")) if table["checkpoints"][m].get("bytes") else "" for m in MODELS}
        nw, sw, iw = max(map(len, MODELS)), max(map(len, sizes.values())), len(f"{len(MODELS)})")
        for i, m in enumerate(MODELS, 1):
            first = f"  {f'{i})':>{iw}} {m:<{nw}}  {sizes[m]:>{sw}}  "
            ui.para(TITLE[m] + ("  (recommended)" if m == rec else ""), first, " " * len(first))
        ui.para(f"why {rec}: {why}" if rec else f"no recommendation: {why}", " " * (iw + 3), code="2")
    else:
        ui.gap()
        ui.row("Recommended", f"{rec} - {why}" if rec else f"none - {why}")
    if args.model is None:
        before = a.model
        a.model = ui.ask("Model", a.model or rec or "27b", MODELS)
        if a.model != before:  # another model: its own saved block, or its defaults (never the other model's folder)
            block = _valid_saved(saved)["models"].get(a.model, {})
            for k in PER_MODEL:
                if getattr(args, k) is None:  # a flag given on the command line stays
                    setattr(a, k, block.get(k))
    else:
        ui.row("Model", f"{a.model} (--model)")
    if _choose(a.model, inv) is None:
        ui.note("NOTE", f"no {a.model} layout fits this hardware (merlin doctor --model {FAMILY[a.model]} explains)")
    return a


def _ask_layout(a: Answers, inv: pf.Inventory, ui: Asker, args: argparse.Namespace, repo: Path) -> Answers:
    opts = layouts(a.model, inv, repo)
    if opts and args.tp is None and args.instances is None:
        labels = [o.label for o in opts]
        rec_opt = next(o for o in opts if o.recommended)
        if ui.yes:
            ui.row("Layout", f"{rec_opt.label}: {rec_opt.text} ({rec_opt.why}"
                   + (f"; {len(opts) - 1} other option(s) with --reconfigure)" if len(opts) > 1 else ")"))
        else:
            ui.section("Layout", "(Enter = recommended)")
            lw, iw = max(len(o.label) for o in opts), len(f"{len(opts)})")
            for i, o in enumerate(opts, 1):
                first = f"  {f'{i})':>{iw}} {o.label:<{lw}}  "
                ui.para(o.text + ("  (recommended)" if o.recommended else ""), first, " " * len(first))
            ui.para(f"why {rec_opt.label}: {rec_opt.why}", " " * (iw + 3), code="2")
        cur = layout_label(a) if (a.tp or a.instances) else rec_opt.label  # a saved layout stays the default; else the recommendation
        pick = ui.ask("Layout", cur if cur in labels else rec_opt.label, labels)
        chosen = next(o for o in opts if o.label == pick)
        a.tp, a.instances = chosen.tp, chosen.instances
    elif args.tp is not None or args.instances is not None:
        ui.row("Layout", f"{layout_label(a)} (flags)")
    notes = [n for n in layout_notes(a, inv, repo) if n.startswith("P2P is off")]  # the selector's measured note
    if notes:  # every note repeats under the serve command
        ui.gap()
    for note in notes:
        ui.note("NOTE", note)
    return a


def _ask_access(a: Answers, ui: Asker, args: argparse.Namespace, key_path: Path) -> Answers:
    if args.host is None:
        if not ui.yes:
            ui.section("Access")
            ui.say("  1) local    this machine only (127.0.0.1)")
            ui.para("every host that can reach this machine (0.0.0.0), an API key is required", "  2) network  ")
        cur = "local" if a.host == "127.0.0.1" else "network"
        pick = ui.ask("Access", cur, ["local", "network"])
        a.host = "127.0.0.1" if pick == "local" else "0.0.0.0"
    if a.host != "127.0.0.1" and not a.api_key:
        a.api_key = secrets.token_urlsafe(32)
        ui.gap()
        ui.para(f"API key, shown this once (kept in {key_path}, mode 0600):", "  ")
        ui.say("      " + ui.style(a.api_key))
        ui.para("Clients send `Authorization: Bearer <key>`; `cat` that file prints it again.", "  ")
    if a.host == "127.0.0.1" and not args.api_key:
        a.api_key = None
    return a


REVIEW_ITEMS = ("model", "layout", "folder", "access", "port")


def review_answers(a: Answers, inv: pf.Inventory, ui: Asker, args: argparse.Namespace, repo: Path, table: dict[str, Any],
                   saved: dict[str, Any] | None, key_path: Path) -> Answers:
    """After the questions and before any check or download: list the answers and re-ask any one of them, as often as
    wanted, until Enter. Answers given by flags are shown but not offered. Another model asks its own layout, folder
    and port again (they belong to the model)."""
    flagged = {"model": args.model is not None, "layout": args.tp is not None or args.instances is not None,
               "folder": args.models_dir is not None, "access": args.host is not None, "port": args.port is not None}
    while True:
        values = {"model": a.model, "layout": layout_label(a), "folder": str(a.checkpoint_dir()),
                  "access": "local (127.0.0.1)" if a.host == "127.0.0.1" else f"network ({a.host}), API key",
                  "port": str(a.port or default_port(a.model))}
        open_ = [k for k in REVIEW_ITEMS if not flagged[k]]
        if not open_:
            return a
        ui.section("Your answers", "(nothing is downloaded or saved yet)")
        for i, k in enumerate(open_, 1):
            ui.say(f"  {i}) {k:<7} {values[k]}")
        for k in REVIEW_ITEMS:
            if flagged[k]:
                ui.say(f"     {k:<7} {values[k]} (flag)")
        say_flag_rows(a, inv, ui, repo)
        pick = ui.ask(f"Change an answer? (Enter = no, or 1-{len(open_)})", "no", open_ + ["no", "n"])
        if pick in ("no", "n"):
            return a
        before = a.model
        a = ask_questions(a, inv, ui, args, repo, table, saved, key_path, only={pick})
        if pick == "model" and a.model != before:
            a = ask_questions(a, inv, ui, args, repo, table, saved, key_path, only={"layout", "drafter", "swarm", "context", "folder", "port"})
        elif pick == "layout":
            a = ask_questions(a, inv, ui, args, repo, table, saved, key_path, only={"drafter", "swarm", "context"})  # the row may have changed


def kv_ram_cache_per_rank(a: Answers, sizes: list[int], rows: list[str] | None = None) -> int:
    """--kv-ram-cache as the engine's GiB per rank over ``sizes`` (TP x PP of each server), rounded down; 0 when off or
    below one 262K restore unit on the ``rows`` (profile ids) the layout starts (ple_tier_problems refuses that)."""
    if not tier_family(a.model) or not a.kv_ram_cache_gib:
        return 0
    try:
        return S.kv_ram_cache_per_rank(a.kv_ram_cache_gib, sizes, restore_unit_minimum(a.model, rows, sizes))
    except S.CliError:
        return 0


def restore_unit_minimum(model: str | None, rows: list[str] | None, sizes: list[int]) -> int | None:
    """GiB per rank one 262K restore unit needs on every row (profile id) the layout starts, on the KV dtype serve will
    use (an exported KV_DTYPE=fp8 swaps in the row's FP8 capacity, as serve_plan does); None (serve's 4 GiB per server
    rule) when the rows are not known."""
    family = S.CHECKPOINTS.get(model or "")
    if not rows or not family:
        return None
    try:
        reg = R.with_kv_dtype(R.load_registry(family), os.environ.get("KV_DTYPE"))
        return max(R.restore_unit_min_gib_per_rank(family, R.get_profile(reg, pid), n) for pid, n in zip(rows, sizes))
    except (R.ProfileError, OSError, ValueError):
        return None


DFLASH2_TIER_FIX = "--kv-ram-cache 0 (off; it also drops a saved size), or --spec mtp (the default)"


def dflash2_tier_text(a: Answers, args: argparse.Namespace | None = None) -> str:
    """Why a KV RAM cache with --spec dflash2 is refused: the cache is qualified with MTP only, and
    merlin serve refuses the pair, so setup says so before anything downloads."""
    saved = " (saved)" if args is None or getattr(args, "kv_ram_cache_gib", None) is None else ""
    return (f"--kv-ram-cache {a.kv_ram_cache_gib}g{saved} with --spec dflash2: the KV RAM cache is qualified with MTP "
            "only; merlin serve refuses the pair")


def ple_tier_problems(a: Answers, args: argparse.Namespace, repo: Path, table: dict[str, Any], sizes: list[int],
                      rows: list[str] | None = None) -> list[Problem]:
    """--ple int6 / --kv-ram-cache (given now or saved) refused before anything downloads: INT6 where this build cannot serve
    it (int6_reason), a tier too small for one 262K restore unit over ``sizes`` (TP x PP of each server), and
    --ple int6 on a model outside the Flash-Next family, or --kv-ram-cache on one outside the Flash-Next, 27B and Ornith
    families."""
    out = []
    if not fn_family(a.model):
        if is_variant(args.ple) or (args.kv_ram_cache_gib and not tier_family(a.model)):
            flag = "--ple int6" if is_variant(args.ple) else "--kv-ram-cache"
            out.append(Problem("FAIL", "ple", f"{flag} does not apply to {a.model}",
                               "drop the flag, or choose " + " / ".join(PLE_INT6_MODELS if is_variant(args.ple)
                                                                         else ("flash-next", "swift1.5", "swift1.5-huihui"))))
            return out
        if a.kv_ram_cache_gib:
            try:
                S.kv_ram_cache_per_rank(a.kv_ram_cache_gib, sizes, restore_unit_minimum(a.model, rows, sizes))
            except S.CliError as e:
                out.append(Problem("FAIL", "kv-tier", str(e).removesuffix(" Nothing started."),
                                   "give a larger size, or --kv-ram-cache 0 (off)"))
        return out
    if is_variant(a.ple_table):
        why = int6_reason(a.model, repo, table)
        if why:
            out.append(Problem("FAIL", "ple", f"--ple int6: the INT6 g32 PLE table cannot be served here: {why}",
                               "--ple fp8 (the default; it also drops a saved int6)"))
    qualified = L.dflash2_rows(R.load_registry("flash-next"))
    if a.spec == "dflash2" and not all(r in qualified for r in rows or []):
        out.append(Problem("FAIL", "spec", f"--spec dflash2: {', '.join(rows or [])} has no qualified DFlash2 geometry (its KV pin "
                           f"is the MTP one); DFlash2 rows: {', '.join(qualified)}", "--spec mtp (the default) on this hardware"))
    if a.spec == "dflash2" and a.kv_ram_cache_gib:
        out.append(Problem("FAIL", "kv-tier", dflash2_tier_text(a, args), DFLASH2_TIER_FIX))
    if a.kv_ram_cache_gib:
        try:
            S.kv_ram_cache_per_rank(a.kv_ram_cache_gib, sizes, restore_unit_minimum(a.model, rows, sizes))
        except S.CliError as e:
            out.append(Problem("FAIL", "kv-tier", str(e).removesuffix(" Nothing started."),
                               "give a larger size, or --kv-ram-cache 0 (off)"))
    return out


def cache_explicitly_off(args: argparse.Namespace | None) -> bool:
    """--kv-ram-cache 0 was given on this command line: an explicit cache-off, which is not the same as no cache given (the
    answers carry 0 for both, so the difference lives in the flag)."""
    return args is not None and getattr(args, "kv_ram_cache_gib", None) == 0


def effective_answers(a: Answers, plans: list[S.Plan], sizes: list[int], args: argparse.Namespace | None = None) -> Answers:
    """The answers as merlin serve will run them: --swarm without --kv-ram-cache gets the KV RAM cache sized for the cap
    (never when the cache was explicitly turned off: swarm_problems refuses that pair)."""
    if not a.swarm or a.kv_ram_cache_gib or len(set(sizes)) != 1 or cache_explicitly_off(args):
        return a
    gates = [L.swarm_gate(p.vllm.profile) for p in plans if p.vllm is not None]
    if not gates or not all(gates):
        return a
    return replace(a, kv_ram_cache_gib=int(gates[0]["min_gib_per_rank"]) * sizes[0])


SWARM_RAM_FIX = "--no-swarm (the default session cap needs no KV RAM cache); or add RAM (or raise the container's memory limit)"


def swarm_problems(a: Answers, plans: list[S.Plan], sizes: list[int], args: argparse.Namespace | None = None) -> list[Problem]:
    """--swarm refused before anything downloads, after the saved answers, the flags and the questions are merged: an explicit
    ``--kv-ram-cache 0`` beside swarm mode (a saved answer, --swarm or the question: the user is not silently given a 48g
    cache, nor silently dropped to the default cap), a row with no swarm cap (naming the rows that have one), DFlash2 (whose
    cache is refused), or a --kv-ram-cache under the size the cap was qualified with (never raised)."""
    if not a.swarm:
        return []
    if cache_explicitly_off(args):
        src = "--swarm" if getattr(args, "swarm", None) else "your saved answer"
        return [Problem("FAIL", "swarm", f"--kv-ram-cache 0 turns the KV RAM cache off, but swarm mode is on ({src}) and its session cap "
                        "needs the cache (tested with 48g); nothing is changed for you",
                        "--no-swarm to keep the cache off at the default session cap, or leave out --kv-ram-cache 0 "
                        "(the cache is then sized for the swarm cap) or give --kv-ram-cache 48g or more")]
    profs = [p.vllm.profile for p in plans if p.vllm is not None]
    for pr in profs:
        if not L.swarm_gate(pr):
            return [Problem("FAIL", "swarm", L.swarm_not_qualified(pr["id"]), "--no-swarm (the default session cap), or a layout whose row has a swarm cap")]
    out = []
    if a.spec == "dflash2":
        out.append(Problem("FAIL", "swarm", "--swarm with --spec dflash2: the KV RAM cache --swarm needs is qualified with MTP only",
                           "--no-swarm, or --spec mtp (the default)"))
    if a.kv_ram_cache_gib and profs and len(set(sizes)) == 1:
        need = int(L.swarm_gate(profs[0])["min_gib_per_rank"])
        if a.kv_ram_cache_gib // sizes[0] < need:
            out.append(Problem("FAIL", "swarm", L.swarm_cache_refusal(profs[0], a.kv_ram_cache_gib // sizes[0]),
                               f"--kv-ram-cache {need * sizes[0]}g or more, or --no-swarm"))
    return out


def layout_notes(a: Answers, inv: pf.Inventory, repo: Path) -> list[str]:
    try:
        plans = serve_plan(a, inv, repo)
    except S.CliError as e:
        return [f"{a.model} cannot be served on this hardware: {e}"]
    return [n for p in plans for n in p.notes if not n.startswith("no checkpoint directory")]


# --------------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------------
def host_report(inv: pf.Inventory, ui: Asker) -> None:
    names: dict[str, int] = {}
    for g in inv.gpus:
        names[g.name] = names.get(g.name, 0) + 1
    cards = ", ".join(f"{n} x {k} ({pf._gib(next(g.vram_mib for g in inv.gpus if g.name == k)):.0f} GB)" for k, n in names.items()) or "none"
    ram = f"{pf._gib(inv.host.ram_mib):.0f} GiB" if inv.host.ram_mib else "unknown"
    ui.row("  host", f"GPUs {cards}, P2P {inv.p2p}, driver {inv.host.driver or 'unknown'}, RAM {ram}", 10)
    if inv.source != "probe":
        ui.row("  source", inv.source, 10)


def report_problems(problems: list[Problem], ui: Asker) -> bool:
    """Print; True when a FAIL is among them."""
    if problems:
        ui.gap()
    for p in problems:
        ui.note(p.level, f"{p.code}: {p.text}", p.fix)
    return any(p.level == "FAIL" for p in problems)


def main(argv: list[str] | None = None, inp: Any = None, out: Any = None, run: Callable[[list[str]], str | None] = default_runner) -> int:
    try:  # keep our lines in order with the subprocesses' output when stdout is a pipe or a log
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):
        pass
    try:
        return _main(list(sys.argv[1:] if argv is None else argv), inp, out, run)
    except SetupError as e:
        print(f"setup: {e}", file=sys.stderr)
        return e.code
    except KeyboardInterrupt:
        print("\nsetup: interrupted; ./setup.sh again continues (finished downloads stay)", file=sys.stderr)
        return 130


def _main(argv: list[str], inp: Any, out: Any, run: Callable[[list[str]], str | None]) -> int:
    args = parse_args(argv)
    repo = REPO
    merlin_dir = repo / ".merlin"
    key_path = merlin_dir / KEY_FILE
    ui = Asker(args.yes, inp, out)
    table = sizes_table(repo)
    saved = read_json(merlin_dir / SETUP_JSON)
    a = merge_answers(saved, args, args.api_key or read_key(key_path))
    if args.pull_only:
        return pull_main(args, a, repo, table, ui)
    inv = pf.simulate(args.simulate) if args.simulate else pf.probe()
    ui.say(ui.style(f"Merlin {release_version(repo)} setup") + " (by Halt95)")
    ui.row("  repo", str(repo), 10)
    host_report(inv, ui)

    # Host checks that cost no download.
    env = dict(os.environ)
    problems = [p for p in (git_problem(repo, run, env), gcc_problem(run, env, a), driver_problem(inv)) if p]
    if a.from_source:
        problems += source_build_problems(run, env)
    if report_problems(problems, ui):
        ui.gap()
        ui.say("setup: fix the FAIL lines above and run ./setup.sh again; nothing was downloaded.")
        return 1
    if a.cc and not os.environ.get("CC"):
        ui.row("  compiler", f"{a.cc} / {a.cxx} (remembered in .merlin/setup.json)", 10)

    a = reconcile_ple_store(a, args, ui)
    # Questions (a complete saved set for this model is reused unless --reconfigure).
    asked = False
    if not answers_complete(a, saved) or args.reconfigure:
        a = ask_questions(a, inv, ui, args, repo, table, saved, key_path)
        if not ui.yes:
            a = review_answers(a, inv, ui, args, repo, table, saved, key_path)
        asked = True
    else:
        answers_block(a, inv, ui, "Saved answers", "(from .merlin/setup.json; --reconfigure asks again)", repo)
    if a.port is None:
        a.port = default_port(a.model)

    # Model-specific checks and the plan.
    try:
        plans = serve_plan(a, inv, repo)
    except S.CliError as e:
        report_problems([Problem("FAIL", "layout", f"{a.model} with {layout_label(a)} does not fit this host: {e}",
                                 "--spec mtp (the default) on this hardware" if fn_family(a.model) and a.spec == "dflash2"
                                 else "./setup.sh --reconfigure picks a layout for the hardware found now")], ui)
        return 3
    cards = len(plans[0].cards)
    sizes = [len(p.cards) for p in plans]  # one table and one tier arena per server
    a_run = effective_answers(a, plans, sizes, args)  # --swarm without --kv-ram-cache: the cache sized for the cap
    swarm_probs = swarm_problems(a, plans, sizes, args)
    flag_problems = ple_tier_problems(a, args, repo, table, sizes, [p.profile_id for p in plans]) + swarm_probs
    per_rank = kv_ram_cache_per_rank(a_run, sizes, [p.profile_id for p in plans])
    ram = ram_problem(a.model, inv, cards, a.ple_table, per_rank, sizes,
                      int6_why=int6_reason(a.model, repo, table) if fn_family(a.model) else None,
                      store=a.ple_store, disk_why=ple_store_reason(a) if fn_family(a.model) else None)
    if ram is not None and ram.level == "FAIL" and a.swarm and not a.kv_ram_cache_gib:
        ram = replace(ram, fix=SWARM_RAM_FIX)  # the cache is the one --swarm sized: a smaller one would not serve the swarm cap
    # which session cap applies and why (default / --swarm), as the launcher says it at boot
    cap_said = [] if plans[0].vllm is None or any(p.code == "swarm" for p in swarm_probs) else \
        L.cap_report(plans[0].vllm.profile, bool(a.swarm), "", per_rank)[1]
    more = ([ram] if ram else []) + flag_problems + [Problem("WARN", "swarm", n.removeprefix("WARNING: ")) for n in cap_said
                                                      if n.startswith("WARNING: ")]
    for w in dict.fromkeys(w for p in plans for w in p.warnings):  # once each: instances repeat their row's warnings
        # Setup already reports its RAM check; preflight also carries it in the serve plan.
        if ram is not None and w.startswith("ram: "):
            continue
        more.append(Problem("WARN", *w.split(": ", 1)) if ": " in w else Problem("WARN", "serve", w))
    conflict = folder_conflict(a.base_dir(), a.model, repo)
    if conflict:
        more.append(Problem("FAIL", "folder", f"{a.base_dir()} {conflict}; nothing is downloaded over another checkpoint",
                            "choose another folder (--models-dir, or --reconfigure) or remove that one"))
    if fn_family(a.model) and is_variant(a.ple_table) and PP.overlay_conflict(a.checkpoint_dir()):
        more.append(Problem("FAIL", "folder", f"--ple int6: {PP.overlay_conflict(a.checkpoint_dir())}",
                            "move that folder away, or choose another FP8 checkpoint folder (--models-dir)"))
    commit = head_commit(repo, run)
    state = read_json(merlin_dir / STATE_JSON)
    plan = download_plan(a, repo, state, commit, table=table)
    more += disk_problems(plan)
    failed = report_problems(more, ui)
    ui.gap()
    ui.say(render_plan(plan, args.dry_run, ui.color, ui.width).rstrip("\n"))
    ui.section("Serve")
    for line in shell_wrap(serve_words(a, repo), ui.width):
        ui.say(line)
    notes = [n for p in plans for n in p.notes]
    if tier_family(a.model) and a_run.kv_ram_cache_gib and per_rank:
        notes.append(S.kv_ram_cache_text(a_run.kv_ram_cache_gib, per_rank, sizes[0], len(sizes)))
        if a.swarm and not a.kv_ram_cache_gib and plans[0].vllm is not None and L.swarm_gate(plans[0].vllm.profile):
            gate = L.swarm_gate(plans[0].vllm.profile)
            notes.append(S.swarm_cache_set_text(a_run.kv_ram_cache_gib, int(gate["min_gib_per_rank"]), sizes[0], int(gate["max_num_seqs"])))
        if S.kv_ram_cache_untested(a_run.kv_ram_cache_gib):
            notes.append(S.kv_ram_cache_untested(a_run.kv_ram_cache_gib))
        if FAMILY[a.model] in S.KV_TIER_UNTESTED:
            notes.append(S.KV_TIER_UNTESTED[FAMILY[a.model]])
    notes += [n.removeprefix("note: ") for n in cap_said if n.startswith("note: ")]
    if fn_family(a.model) and a.spec == "dflash2":
        notes.append(f"--spec dflash2: drafter {S.DFLASH2_REPO} -> {drafter_dir()}; {S.dflash2_why()}")
    if notes:
        ui.gap()
    for n in notes:
        ui.note("NOTE", n)
    if failed:
        ui.gap()
        ui.say("setup: fix the FAIL lines above and run ./setup.sh again; nothing was downloaded.")
        return 1
    if a.model == "27b-thinkingcap":
        ui.gap()
        ui.para(f"Licence: {THINKINGCAP_LICENCE}", "", " " * 9)
    if args.dry_run:
        ui.gap()
        ui.say("Dry run: nothing downloaded, nothing installed, nothing written, nothing started.")
        return 0
    # Confirmation: once, after the questions; a saved re-run asks nothing (--yes neither).
    if asked and plan.pending() and ui.ask("Proceed with the downloads above? (y/n)", "y", ["y", "n", "yes", "no"]) in ("n", "no"):
        ui.say("setup: stopped; nothing downloaded and nothing saved.")
        return 1
    if a.api_key and (asked or args.api_key):
        write_private(key_path, a.api_key + "\n", 0o600)
    save_answers(merlin_dir / SETUP_JSON, saved, a)

    # Install, checkpoint, state.
    env["PATH"] = f"{merlin_dir / 'bin'}:{env.get('PATH', '')}"
    if plan.need_install:
        with install_lock(merlin_dir):  # the install record is written under the same lock as the venv it describes
            install_engine(repo, a, env)
            write_json(merlin_dir / STATE_JSON, {**install_state(repo, a.from_source, commit), "installed_at": int(time.time())})
    else:
        ui.gap()
        ui.para("Engine: venv is current (wheel + lock + commit unchanged since the last install); nothing to install.")
    pull_checkpoint(repo, a, env, plan.revision, args.rehash)
    pull_drafter(repo, a, env, table, args.rehash)
    if args.no_start:
        ui.gap()
        ui.say("Done. Start it with ./setup.sh")
        return 0

    # Doctor, endpoint, start.
    ui.say("")
    ui.say(ui.style(f"merlin doctor --model {FAMILY[a.model]}"))
    S.main(["doctor", "--model", FAMILY[a.model]], out)
    print_endpoint(a, ui, key_path)
    sys.stdout.flush()
    os.execvpe(serve_argv(a, repo)[0], serve_argv(a, repo), serve_env(a))
    return 0  # pragma: no cover


def pull_main(args: argparse.Namespace, a: Answers, repo: Path, table: dict[str, Any], ui: Asker) -> int:
    """``merlin pull <model> [--models-dir D] [--spec dflash2]``: the checkpoint and requested drafter. No GPU,
    driver or compiler check, no serving choice; the folder is this model's unless --models-dir names another."""
    if not a.model:
        raise SetupError("merlin pull needs a model name", 2)
    ckpt_dir = a.checkpoint_dir()
    ui.say(ui.style(f"Merlin {release_version(repo)} pull") + f": {a.model} -> {ckpt_dir}")
    conflict = folder_conflict(a.base_dir(), a.model, repo)
    if conflict:
        report_problems([Problem("FAIL", "folder", f"{a.base_dir()} {conflict}",
                                 "choose another folder (--models-dir) or remove that one")], ui)
        return 1
    if fn_family(a.model) and is_variant(a.ple_table):
        why = int6_reason(a.model, repo, table) or PP.overlay_conflict(ckpt_dir)
        if why:
            report_problems([Problem("FAIL", "ple", f"--ple int6: {why}", "--ple fp8 (the default)")], ui)
            return 1
    plan = download_plan(a, repo, None, None, table=table, engine=False)
    failed = report_problems(disk_problems(plan), ui)
    ui.gap()
    ui.say(render_plan(plan, args.dry_run, ui.color, ui.width).rstrip("\n"))
    if failed:
        return 1
    if a.model == "27b-thinkingcap":
        ui.gap()
        ui.para(f"Licence: {THINKINGCAP_LICENCE}", "", " " * 9)
    if args.dry_run:
        ui.gap()
        ui.say("Dry run: nothing downloaded.")
        return 0
    if not ui.yes and plan.pending() and ui.ask("Proceed with the download above? (y/n)", "y", ["y", "n", "yes", "no"]) in ("n", "no"):
        ui.say("pull: stopped; nothing downloaded.")
        return 1
    env = dict(os.environ)
    env["PATH"] = f"{repo / '.merlin' / 'bin'}:{env.get('PATH', '')}"
    pull_checkpoint(repo, a, env, plan.revision, args.rehash)
    pull_drafter(repo, a, env, table, args.rehash)
    ui.gap()
    ui.say("Done.")
    return 0


def print_endpoint(a: Answers, ui: Asker, key_path: Path) -> None:
    host = "127.0.0.1" if a.host in ("0.0.0.0", "::") else a.host
    auth = ["-H", f'"Authorization: Bearer $(cat {shlex.quote(str(key_path))})"'] if a.api_key else []
    n = a.instances if a.instances and a.instances > 1 else 1
    ports = ", ".join(str(a.port + i) for i in range(n))
    body = json.dumps({"model": served_name(a.model), "messages": [{"role": "user", "content": "Say hello."}], "max_tokens": 32})
    ui.gap()
    ui.para(f"http://{host}:{a.port}/v1  (OpenAI-compatible; /v1/messages for /v1/messages clients)"
            + (f"; instances on ports {ports}" if n > 1 else ""), "Endpoint  ", head=True)
    if a.host == "0.0.0.0":
        ui.para(f"reachable from the network on every address of this machine, port {a.port}; the API key is in {key_path} (0600)",
                " " * 10)
    ui.section("Test", "(once the log says `Application startup complete`)")
    for line in shell_wrap(["curl", "-s", f"http://{host}:{a.port}/v1/chat/completions", "-H", "'Content-Type: application/json'",
                            *auth, "-d", shlex.quote(body)], ui.width):
        ui.say(line)
    ui.gap()
    if n > 1:
        ui.para(f"--instances {n}: the servers start in the background (own process groups), logs merlin-<profile>-<port>.log in the "
                "current directory; stop them by PID.")
    else:
        ui.say("Ctrl+C stops the server; ./setup.sh starts it again without questions.")
    ui.say("")


if __name__ == "__main__":
    R.safe_stdio()
    sys.exit(main())
