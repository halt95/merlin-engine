# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""The INT6 g32 PLE table pack: one pack for the whole Flash-Next family, laid over
the user's verified FP8 checkpoint. There is no second checkpoint per model.

The ten ``model-plefp8-*`` shards are byte-identical across flash-next, swift1.5 and swift1.5-huihui (same sha256 in
their checksum lists: the fine-tunes never touched the PLE table), so one INT6 g32 table (38.74 GiB, measured
equal to FP8) applies to every family member whose list carries those ten hashes. Setup offers it only for the
qualified members (setup_cli.PLE_INT6_MODELS: flash-next and swift1.5; swift1.5-huihui is not offered yet).

    pack (shared, downloaded once)        <models>/flash-next-ple-int6/
        model-pleint6-0000N-of-0000M.safetensors   <prefix>.shard_K.codes U8 [rows,120] + .scales F16 [rows,5]
        ple-int6-pack.json                        manifest: dtype, n-gram prefix, shard count, weight_map
    overlay (per model, links only)       <base>-int6/
        every file of the base checkpoint list except model-plefp8-*, config.json and the index -> link to <base>/
        every pack shard                                                                  -> link to the pack
        config.json                       generated: text_config.ple_embedding_dtype float8_e4m3fn -> int6_g32
        model.safetensors.index.json      generated: the 129 plefp8 keys (128 shard weights + weight_scale) out,
                                          the pack's 256 codes/scales keys in
        .merlin-overlay.json              what it was built from (base + pack lists, base config/index sha256,
                                          links, generated files): any change there makes it stale

Links are relative symlinks (the overlay, the base and the pack resolve together wherever their common parent is
mounted); on a filesystem without symlinks, hard links when the overlay shares the files' filesystem; else a refusal
(never a silent 40 GiB copy). The overlay is staged in a temporary sibling and renamed into place, so an interrupted
build never leaves a partial overlay that looks current. FP8 files and folders are never written. Standard library only.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PACK_KEY = "int6"  # --ple int6; release/download-sizes.json ple_packs.int6
PACK_DIRNAME = "flash-next-ple-int6"  # beside the base checkpoint folder: one copy for every family member there
PACK_DIR_ENV = "MERLIN_PLE_PACK_DIR"  # another place for the pack (a bigger disk); the links stay relative
PACK_LIST = "models/flash-next/ple-int6-pack.sha256"  # the pack's checksum list (written by make-ple-int6-pack.py)
PACK_MANIFEST = "ple-int6-pack.json"
PACK_MARKER_MODEL = "flash-next-ple-int6-pack"  # the pack folder's verified-marker name
PACK_FORMAT = 1
OVERLAY_SUFFIX = "-int6"
OVERLAY_MARKER = ".merlin-overlay.json"
FP8_SHARD_PREFIX = "model-plefp8-"
INT6_SHARD_PREFIX = "model-pleint6-"
INDEX = "model.safetensors.index.json"
CONFIG = "config.json"
FP8_DTYPE = "float8_e4m3fn"
INT6_DTYPE = "int6_g32"  # launch.PLE_OPT_IN_TABLES / vllm/models/qwen4_exp/nvidia/ple_formats.py INT6_G32_DTYPE_NAME
INT6_ROW = (120, 5)  # codes U8 [rows,120] + five F16 scales: the 130-byte row ple_formats interleaves
NGRAM_SHARDS = 128
NGRAM_PREFIX = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding"


class PackError(Exception):
    """The pack, the base checkpoint or the overlay folder cannot give a consistent INT6 overlay."""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _abs(p: Path | str) -> Path:
    """Absolute without resolving symlinks: a ~/models that is itself a link keeps its name in the relative targets."""
    return Path(os.path.abspath(os.path.expanduser(str(p))))


def pack_dir(base: Path, environ: dict[str, str] | None = None) -> Path:
    env = os.environ if environ is None else environ
    return _abs(env[PACK_DIR_ENV]) if env.get(PACK_DIR_ENV) else _abs(base).parent / PACK_DIRNAME


def overlay_dir(base: Path) -> Path:
    b = _abs(base)
    return b.with_name(b.name + OVERLAY_SUFFIX)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


# --------------------------------------------------------------------------------------------
# The pack manifest and the config / index delta
# --------------------------------------------------------------------------------------------
def read_manifest(pack: Path) -> dict[str, Any]:
    m = _read_json(pack / PACK_MANIFEST)
    if not isinstance(m, dict) or m.get("format") != PACK_FORMAT or m.get("ple_embedding_dtype") != INT6_DTYPE:
        raise PackError(f"{pack / PACK_MANIFEST} is not an INT6 g32 PLE pack manifest (format {PACK_FORMAT})")
    wm, prefix, n = m.get("weight_map"), m.get("prefix"), m.get("shards")
    if not isinstance(wm, dict) or not isinstance(prefix, str) or not isinstance(n, int) or n < 1:
        raise PackError(f"{pack / PACK_MANIFEST}: weight_map, prefix and shards are required")
    want = {f"{prefix}.shard_{k}.{part}" for k in range(n) for part in ("codes", "scales")}
    if set(wm) != want:
        raise PackError(f"{pack / PACK_MANIFEST}: weight_map must hold exactly the {2 * n} codes/scales keys of {n} shards")
    if not all(isinstance(f, str) and f.startswith(INT6_SHARD_PREFIX) and "/" not in f for f in wm.values()):
        raise PackError(f"{pack / PACK_MANIFEST}: weight_map names a file outside the pack's {INT6_SHARD_PREFIX}* shards")
    return m


def overlay_config(base_cfg: dict[str, Any]) -> dict[str, Any]:
    """The base config with the one key the engine reads changed: text_config.ple_embedding_dtype -> int6_g32. A
    base without the key carries the FP8 table (the published files predate the key); any other table is refused."""
    text = base_cfg.get("text_config")
    if not isinstance(text, dict):
        raise PackError("the base config.json has no text_config object; it is not a Flash-Next checkpoint")
    have = text.get("ple_embedding_dtype", FP8_DTYPE)
    if have != FP8_DTYPE:
        raise PackError(f"the base config.json says ple_embedding_dtype={have!r}; the pack overlays the FP8 ({FP8_DTYPE}) table only")
    cfg = copy.deepcopy(base_cfg)
    cfg["text_config"]["ple_embedding_dtype"] = INT6_DTYPE
    return cfg


def overlay_index(base_index: dict[str, Any], manifest: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """(the overlay's safetensors index, the base keys it drops). Dropped: every key the base maps to a model-plefp8-*
    shard, which must be exactly the manifest's n-gram table (its N ``.shard_K.weight`` keys and one ``weight_scale``)
    and nothing else; added: the manifest's codes/scales keys. Every other key keeps its base file."""
    wm = base_index.get("weight_map")
    if not isinstance(wm, dict):
        raise PackError(f"the base {INDEX} has no weight_map")
    prefix, n = manifest["prefix"], manifest["shards"]
    dropped = sorted(k for k, f in wm.items() if str(f).startswith(FP8_SHARD_PREFIX))
    weights = {f"{prefix}.shard_{k}.weight" for k in range(n)}
    extra = [k for k in dropped if k not in weights]
    if not weights <= set(dropped) or len(extra) != 1 or not extra[0].startswith(prefix + ".") or not extra[0].endswith("weight_scale"):
        raise PackError(f"the base {INDEX} maps {len(dropped)} keys to {FP8_SHARD_PREFIX}* shards; the pack replaces exactly "
                        f"{n + 1} ({n} {prefix}.shard_K.weight + its weight_scale)")
    stray = [k for k, f in wm.items() if k.startswith(prefix + ".") and k not in dropped]
    if stray:
        raise PackError(f"the base {INDEX} maps n-gram keys outside the {FP8_SHARD_PREFIX}* shards (first: {stray[0]})")
    new = {k: f for k, f in wm.items() if k not in set(dropped)}
    clash = set(new) & set(manifest["weight_map"])
    if clash:
        raise PackError(f"the base {INDEX} already has pack keys (first: {sorted(clash)[0]})")
    new.update(manifest["weight_map"])
    return {"metadata": base_index.get("metadata", {}), "weight_map": dict(sorted(new.items()))}, dropped


def tensor_bytes(path: Path) -> dict[str, int]:
    """{tensor name: data bytes} from a safetensors header (the header alone is read)."""
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        if n <= 0 or n > 100 << 20:
            raise PackError(f"{path} is not a safetensors file")
        try:
            h = json.loads(f.read(n))
        except ValueError:
            raise PackError(f"{path} has an unreadable safetensors header") from None
    return {k: int(v["data_offsets"][1]) - int(v["data_offsets"][0]) for k, v in h.items() if k != "__metadata__"}


def _total_size(index: dict[str, Any], base: Path, pack: Path, dropped: list[str], wm_base: dict[str, str],
                wm_pack: dict[str, str]) -> None:
    """Recompute metadata.total_size (when the base records one) from the tensor headers: minus the dropped FP8
    table, plus the pack's codes/scales, and check every pack tensor the manifest names is in its shard."""
    have: dict[str, int] = {}
    for f in sorted(set(wm_pack.values())):
        got = tensor_bytes(pack / f)
        want = {k for k, v in wm_pack.items() if v == f}
        if set(got) != want:  # exactly the manifest's tensors: a stray (say a zero-valued shard_00.codes) is refused
            odd = sorted(set(got) ^ want)
            raise PackError(f"{pack / f} holds tensors the manifest does not map to it, or lacks some (first: {odd[0]})")
        have.update(got)
    meta = index.get("metadata")
    if not isinstance(meta, dict) or "total_size" not in meta:
        return
    gone: dict[str, int] = {}
    for f in sorted({wm_base[k] for k in dropped}):
        gone.update(tensor_bytes(base / f))
    index["metadata"] = {**meta, "total_size": int(meta["total_size"]) - sum(gone.get(k, 0) for k in dropped)
                         + sum(have[k] for k in wm_pack)}


def _dump(obj: Any) -> str:
    return json.dumps(obj, indent=2) + "\n"


# --------------------------------------------------------------------------------------------
# Links
# --------------------------------------------------------------------------------------------
def link_mode(folder: Path, targets: list[Path]) -> str:
    """'symlink' when ``folder``'s filesystem takes symlinks, else 'hardlink' when every target shares its filesystem,
    else a PackError that says why (no copy fallback: the table alone is ~40 GiB)."""
    folder.mkdir(parents=True, exist_ok=True)
    probe = folder / ".merlin-symlink-probe"
    try:  # a FILE link to a real target, followed: what the overlay is made of
        os.symlink(os.path.relpath(os.path.realpath(targets[0]), os.path.realpath(folder)), probe)
        ok = probe.is_file()
        os.unlink(probe)
        if ok:
            return "symlink"
    except (OSError, NotImplementedError, AttributeError, IndexError):
        try:
            os.unlink(probe)
        except OSError:
            pass
    dev = _device(folder)
    off = [t for t in targets if _device(t) != dev]
    if not off:
        return "hardlink"
    raise PackError(f"the filesystem of {folder} has no symlinks, and hard links cannot reach {off[0].parent} (another "
                    "filesystem): the INT6 overlay is links only and is never a copy. Put the models folder (--models-dir) "
                    f"on a filesystem with symlinks (ext4, xfs, btrfs, zfs), or keep the base checkpoint, the pack "
                    f"({PACK_DIR_ENV}) and the overlay on one filesystem")


def _device(p: Path) -> int:
    return os.stat(p).st_dev


def _link(mode: str, target: Path, at: Path) -> str:
    """Create ``at`` -> ``target``; returns what the marker records (the relative symlink text, or the target path)."""
    if mode == "symlink":  # from the RESOLVED parent: the kernel follows ".." from where the link physically is
        try:
            rel = os.path.relpath(os.path.realpath(target), os.path.realpath(at.parent))
        except ValueError:  # another drive (Windows): no relative path exists
            raise PackError(f"{target} has no relative path from {at.parent} (another drive)") from None
        os.symlink(rel, at)
        return rel
    os.link(target, at)
    return str(target)


def _link_ok(mode: str, at: Path, recorded: str, target: Path) -> bool:
    try:
        if mode == "symlink":  # the text recorded AND the file it reaches is the expected target (a copied tree fails)
            return at.is_symlink() and os.readlink(at) == recorded and os.path.samefile(at, target)
        a, b = os.stat(at), os.stat(target)
        return not at.is_symlink() and (a.st_dev, a.st_ino) == (b.st_dev, b.st_ino)
    except OSError:
        return False


# --------------------------------------------------------------------------------------------
# The overlay
# --------------------------------------------------------------------------------------------
@dataclass
class Sources:
    """What an overlay is built from: the verified base checkpoint and pack, and the checksum lists they verified with."""
    model: str
    base: Path
    pack: Path
    base_files: list[str]  # the base checksum list's file names
    base_list_sha256: str
    pack_list_sha256: str
    pack_files: list[str] = field(default_factory=list)  # the pack checksum list's file names


def _expected(src: Sources) -> tuple[dict[str, Path], dict[str, str], dict[str, Any]]:
    """(links name -> absolute target, generated name -> text, the source digests the marker records)."""
    base, pack = _abs(src.base), _abs(src.pack)
    manifest = read_manifest(pack)
    cfg = _read_json(base / CONFIG)
    idx = _read_json(base / INDEX)
    if not isinstance(cfg, dict) or not isinstance(idx, dict):
        raise PackError(f"{base} has no readable {CONFIG} and {INDEX}")
    index, dropped = overlay_index(idx, manifest)
    _total_size(index, base, pack, dropped, idx["weight_map"], manifest["weight_map"])
    gen = {CONFIG: _dump(overlay_config(cfg)), INDEX: _dump(index)}
    links: dict[str, Path] = {}
    for name in sorted(src.base_files):
        if name.startswith(FP8_SHARD_PREFIX) or name in gen or "/" in name:
            continue
        links[name] = base / name
    for p in sorted(base.iterdir()):  # licence and notice files a list leaves out (card edits change them): kept, recorded
        if p.is_file() and p.name not in links and p.name.upper().startswith(("LICENSE", "NOTICE")):
            links[p.name] = p
    shards = set(manifest["weight_map"].values())
    if set(src.pack_files) != shards | {PACK_MANIFEST}:
        raise PackError(f"the pack checksum list ({PACK_LIST}) must cover exactly the manifest's {len(shards)} shards and "
                        f"{PACK_MANIFEST}: a shard it does not list would be served unverified")
    for name in sorted(shards):
        links[name] = pack / name
    missing = [str(t) for t in links.values() if not t.is_file()]
    if missing:
        raise PackError(f"the overlay would link {len(missing)} missing file(s) (first: {missing[0]})")
    digests = {"base_config_sha256": sha256_file(base / CONFIG), "base_index_sha256": sha256_file(base / INDEX),
               "manifest_sha256": sha256_file(pack / PACK_MANIFEST)}
    return links, gen, digests


def overlay_problems(out: Path, src: Sources) -> list[str]:
    """Why the overlay in ``out`` is not a current build of ``src`` (empty when it is): no marker, another model,
    base, pack or checksum list, a changed base config/index or pack manifest, a link gone, retargeted or dangling, or
    a generated file edited."""
    out = _abs(out)
    m = _read_json(out / OVERLAY_MARKER)
    if not isinstance(m, dict):
        return [f"{out} is not built (no {OVERLAY_MARKER})"]
    want = {"model": src.model, "base": str(_abs(src.base)), "pack": str(_abs(src.pack)),
            "base_list_sha256": src.base_list_sha256, "pack_list_sha256": src.pack_list_sha256}
    out_probs = [f"{k} changed ({m.get(k)!r} -> {v!r})" for k, v in want.items() if m.get(k) != v]
    if out_probs:
        return out_probs
    try:
        links, gen, digests = _expected(src)
    except PackError as e:
        return [str(e)]
    probs = [f"{k} changed (the base config.json, index or pack manifest)" for k, v in digests.items() if m.get(k) != v]
    rec_links, mode = m.get("links") or {}, m.get("mode")
    if set(rec_links) != set(links):
        probs.append("the set of linked files changed")
    for name, target in links.items():
        if name in rec_links and not _link_ok(mode, out / name, rec_links[name], target):
            probs.append(f"{name}: link missing, retargeted or dangling")
    for name, text in gen.items():
        f = out / name
        if not f.is_file() or f.is_symlink() or sha256_file(f) != hashlib.sha256(text.encode()).hexdigest():
            probs.append(f"{name}: not the generated file")
    return probs


def overlay_conflict(out: Path) -> str | None:
    """Why ``out`` must not be (re)built: it exists and is not an INT6 overlay (no marker but other entries), or it is
    not a folder. None when it is missing, empty or an overlay."""
    out = _abs(out)
    if not out.exists() and not out.is_symlink():
        return None
    if not out.is_dir() or out.is_symlink() or not real_dir(out):
        return f"{out} exists and is not a real folder (a file, a symlink or a junction); nothing is written there"
    if (out / OVERLAY_MARKER).is_file() or not any(out.iterdir()):
        return None
    return f"{out} holds files and is not an INT6 overlay (no {OVERLAY_MARKER}); nothing is written there"


@contextlib.contextmanager
def _build_lock(out: Path):
    """One builder per overlay folder at a time (``.<name>.lock`` beside it; flock where the platform has it: on Windows,
    where these tools are only developed and tested, there is no lock and concurrent setups are not supported)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(out.with_name(f".{out.name}.lock"), os.O_CREAT | os.O_RDWR, 0o644)
    try:
        try:
            import fcntl

            fcntl.flock(fd, fcntl.LOCK_EX)
        except ImportError:
            pass
        yield
    finally:
        os.close(fd)  # closing releases the flock


def build_overlay(out: Path, src: Sources) -> str:
    """Build (or rebuild when stale) the overlay of ``src`` in ``out``; 'current' when it already is, else 'built' /
    'rebuilt'. Staged in a temporary sibling and renamed into place under a lock: an interrupted build leaves no
    partial overlay, and two setups never build the same one at once."""
    out, base, pack = _abs(out), _abs(src.base), _abs(src.pack)
    rout, rbase, rpack = (Path(os.path.realpath(p)) for p in (out, base, pack))
    for o, b, k in ((out, base, pack), (rout, rbase, rpack)):  # as given and resolved; before the lock writes anything
        if o in (b, k) or b in o.parents or k in o.parents or o in b.parents or o in k.parents:
            raise PackError(f"the overlay folder {out} must be outside the base checkpoint and the pack")
    with _build_lock(out):
        return _build_overlay(out, src)


def _build_overlay(out: Path, src: Sources) -> str:
    base, pack = _abs(src.base), _abs(src.pack)
    why = overlay_conflict(out)
    if why:
        raise PackError(why)
    existed = (out / OVERLAY_MARKER).is_file()
    if existed and not overlay_problems(out, src):
        return "current"
    links, gen, digests = _expected(src)
    out.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{out.name}.", suffix=".building", dir=out.parent))
    try:
        mode = link_mode(stage, list(links.values()))
        rec = {name: _link(mode, target, stage / name) for name, target in links.items()}
        for name, text in gen.items():
            (stage / name).write_bytes(text.encode("utf-8"))  # byte-exact (no newline translation): the marker hashes it
        marker = {"model": src.model, "ple": PACK_KEY, "base": str(base), "pack": str(pack), "mode": mode,
                  "base_list_sha256": src.base_list_sha256, "pack_list_sha256": src.pack_list_sha256, **digests,
                  "links": rec, "generated": {n: hashlib.sha256(t.encode()).hexdigest() for n, t in gen.items()}}
        (stage / OVERLAY_MARKER).write_text(json.dumps(marker, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        old = None
        try:  # both renames under one guard: an interrupt between them puts the previous overlay back
            if out.exists():
                old = out.with_name(f".{out.name}.{os.getpid()}.stale")
                os.replace(out, old)
            os.replace(stage, out)
        except BaseException:
            if old is not None and old.exists() and not out.exists():
                os.replace(old, out)  # never no overlay where there was one
            raise
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    if old is not None:
        _remove_overlay(old)
    return "rebuilt" if existed else "built"


def _remove_overlay(folder: Path) -> None:
    """Remove a replaced overlay: its links (never their targets), generated files and marker, then the folder when
    nothing else is in it. A file the overlay did not make is left where it is."""
    if not real_dir(folder):
        return  # a junction or link renamed into place is never emptied: its entries belong to another folder
    m = _read_json(folder / OVERLAY_MARKER) or {}
    for name in [*(m.get("links") or {}), *(m.get("generated") or {}), OVERLAY_MARKER]:
        if not _plain_name(name):
            continue  # a marker naming ../flash-next/config.json (or an absolute path) never reaches outside the overlay
        try:
            os.unlink(folder / name)  # unlink removes a symlink or a hard link, never the file it points to
        except OSError:
            pass
    try:
        folder.rmdir()
    except OSError:
        pass


def real_dir(p: Path) -> bool:
    """A folder that is itself, not a symlink or a Windows junction / reparse point to another one."""
    p = Path(p)
    if p.is_symlink() or (hasattr(os.path, "isjunction") and os.path.isjunction(p)):
        return False
    try:
        if getattr(os.lstat(p), "st_file_attributes", 0) & 0x400:  # FILE_ATTRIBUTE_REPARSE_POINT
            return False
    except OSError:
        return False
    return os.path.normcase(os.path.realpath(p)) == os.path.normcase(os.path.join(os.path.realpath(p.parent), p.name))


def _plain_name(name: Any) -> bool:
    return (isinstance(name, str) and name not in ("", ".", "..") and "/" not in name and "\\" not in name
            and os.path.basename(name) == name and not os.path.isabs(name) and ":" not in name)


def read_overlay_marker(out: Path) -> dict[str, Any] | None:
    m = _read_json(_abs(out) / OVERLAY_MARKER)
    return m if isinstance(m, dict) else None
