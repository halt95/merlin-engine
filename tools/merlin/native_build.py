# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Installed build identity and checksum checks, without importing the engine."""

from __future__ import annotations

import base64
import hashlib
import importlib.machinery
import importlib.metadata as md
import importlib.util
import json
import os
import re
import runpy
import sys
from pathlib import Path


class BuildError(ValueError):
    def __init__(self, message: str, remedy: str = "run ./setup.sh"):
        super().__init__(message)
        self.remedy = remedy


def _wheel_helper() -> dict:
    # TREE can predate these helpers or be a relocated installation. Its files
    # are inputs to the current launcher's checker, never executable helpers.
    return runpy.run_path(str(Path(__file__).resolve().parents[2] / "release" / "wheel.py"))


def verify_hashes(
    root: Path, sums: Path, *, target_sets: tuple[tuple[str, ...], ...] | None = None,
) -> None:
    """Verify a sha256sum manifest, optionally requiring a complete supported target set."""
    lines = [line for line in sums.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise BuildError(f"{sums.name} records no files")
    entries = []
    for line in lines:
        match = re.fullmatch(r"([0-9a-fA-F]{64}) [ *](.+)", line)
        if not match:
            raise BuildError(f"malformed checksum in {sums.name}")
        entries.append((match[1], match[2]))
    if target_sets is not None:
        paths = [path for _, path in entries]
        if len(paths) != len(set(paths)) or not any(set(paths) == set(targets) for targets in target_sets):
            raise BuildError("FlashInfer patch record must list exactly one complete supported variant without duplicates")
    for digest, name in entries:
        path = root / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest().lower() != digest.lower():
            raise BuildError(f"{path} is missing or changed")


def verify_flashinfer(root: Path, sums: Path) -> None:
    """Require all targets of the recorded variant before checking their bytes."""
    wheel = _wheel_helper()
    target_sets = tuple(wheel["flashinfer_targets"](variant) for variant in ("v1", "v2"))
    verify_hashes(root, sums, target_sets=target_sets)


_GIT_REMEDY = (
    "restore Git access before retrying (for container --user, use the checkout owner's UID "
    "or explicitly trust only this checkout with Git safe.directory)"
)


def _saved_commit(tree: Path, wheel: dict) -> tuple[str, Path] | None:
    """The saved build commit and the file that names it, or None."""
    source = tree / ".build-source"
    if source.is_file():
        lines = source.read_text(encoding="utf-8").splitlines()
        fields = lines[1].split() if len(lines) > 1 else []
        if len(fields) != 2 or fields[0] != "commit" or not wheel["OBJECT_ID_RE"].fullmatch(fields[1]):
            raise BuildError("no saved source commit for this legacy install")
        return fields[1], source
    # Released setup also saved the install commit here. HEAD alone cannot
    # identify an installed binary after a pull, so never treat it as a build ID.
    state = tree / ".merlin" / "state.json"
    if state.is_file():
        try:
            saved = json.loads(state.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise BuildError(f"{state}: unreadable or malformed saved build state") from error
        if not isinstance(saved, dict):
            raise BuildError(f"{state}: saved build state must be a JSON object")
        commit = saved.get("commit")
        if commit is not None:
            if not isinstance(commit, str) or not wheel["OBJECT_ID_RE"].fullmatch(commit):
                raise BuildError(f"{state}: invalid saved build commit {commit!r}")
            return commit, state
    return None


def _git_problem(tree: Path, git) -> str | None:
    """None when Git can read TREE as the root of its checkout; otherwise why it cannot."""
    try:
        checkout = git(tree, "rev-parse", "--show-toplevel")
    except OSError as error:
        return f"the git program cannot run ({error})"
    if checkout.returncode or Path(checkout.stdout.strip()).resolve() != tree.resolve():
        return checkout.stderr.strip() or "TREE is not the root of a Git checkout"
    return None


def _flashinfer_series() -> dict:
    return runpy.run_path(str(Path(__file__).resolve().parents[2] / "release" / "apply-flashinfer-workload-patch.py"))


def flashinfer_state(package: Path) -> str | None:
    """Name the authenticated state of an installed FlashInfer package's patch targets, or None."""
    series = _flashinfer_series()
    digests = tuple(
        hashlib.sha256((package / name).read_bytes()).hexdigest() if (package / name).is_file() else ""
        for name in series["TARGETS"]
    )
    return next((state for state, want in series["STATE_SHA"].items() if digests == tuple(want)), None)


def _working_id(path: Path, algorithm: str, files: dict | None = None) -> bytes:
    """Git object identity for an exported native input, without a repository."""
    def digest(kind: str, data: bytes) -> bytes:
        header = f"{kind} {len(data)}\0".encode()
        return hashlib.new(algorithm, header + data).digest()

    if path.is_symlink():
        return digest("blob", os.fsencode(os.readlink(path)))
    if path.is_dir():
        entries = []
        # Git sorts a directory name as though it ended in '/'.
        children = sorted((path / name for name in files) if files is not None else path.iterdir(),
                          key=lambda p: os.fsencode(p.name) + (
            b"/" if p.is_dir() and not p.is_symlink() else b""))
        for child in children:
            mode = ("120000" if child.is_symlink() else "40000" if child.is_dir()
                    else "100755" if child.stat().st_mode & 0o100 else "100644")
            identity = _working_id(child, algorithm, files[child.name] if files is not None else None)
            if mode == "40000" and identity == digest("tree", b""):
                continue  # Git does not track empty directories.
            entries.append(mode.encode() + b" " + os.fsencode(child.name) + b"\0" + identity)
        return digest("tree", b"".join(entries))
    if not path.is_file():
        raise BuildError(f"native input {path} is missing")
    return digest("blob", path.read_bytes())


def verify_native(tree: Path, installed_ids: dict[str, str] | None = None, kind: str | None = None) -> list[str]:
    """Compare known build IDs; return verification gaps for one launch warning.

    installed_ids are release/merlin-wheel.native when the imported engine is a pip wheel already
    verified from its own metadata. kind is where the engine is imported from ("source" = this TREE;
    None is treated as "source").
    """
    wheel = _wheel_helper()
    git = wheel["git"]
    record = tree / ".build-native"
    verified_wheel = installed_ids is not None
    gaps = []
    if not verified_wheel and not (tree / ".build-products.sha256").is_file():
        gaps.append("installed native module hashes (no .build-products.sha256)")
    try:
        snapshot = wheel["read_native"](record) if record.is_file() else None
        files = wheel["read_tracked"](record) if record.is_file() else None
        saved = _saved_commit(tree, wheel) if snapshot is None else None
        commit, saved_in = saved if saved is not None else (None, None)
        want = installed_ids if verified_wheel else snapshot
        has_git = (tree / ".git").exists()
        # Never borrow a containing repository's HEAD or guess its ignore rules.
        problem = _git_problem(tree, git) if has_git else None
        if verified_wheel and commit is not None:
            found = problem is None and has_git and not git(tree, "cat-file", "-e", f"{commit}^{{commit}}").returncode
            if not found:
                gaps.append(f"the stale build record {saved_in.name} (its build commit {commit} cannot be found in this "
                            "checkout; ignored because the imported wheel was verified from its own metadata; "
                            f"delete {saved_in.name} or run ./setup.sh to clear it)")
        if not has_git or problem is not None:
            why = "TREE has no .git" if not has_git else f"Git history unavailable: {problem}"
            if want is None:
                if commit is not None and not has_git:
                    # Without .git the saved commit cannot be looked up at all (unlike a repository that lost
                    # it): judge what is on disk. Recorded product hashes are verified by the host phase.
                    gaps.append(f"native sources were not compared with the saved build commit {commit} in "
                                f"{saved_in.name} ({why})")
                    return gaps
                if commit is not None:  # .git exists but Git cannot read it: restore access, the commit may be there
                    raise BuildError(f"saved build commit {commit} cannot be resolved ({why})", _GIT_REMEDY)
                gaps.append(f"native modules were not checked against a build record (no saved build commit or native snapshot; {why})")
                return gaps
            if files is None:
                if verified_wheel:
                    gaps.append(f"checkout native sources were not compared with release/merlin-wheel.native ({why})")
                else:
                    gaps.append(f"native modules were not checked against a build record ({why}, and the native "
                                "snapshot has no tracked file list)")
                return gaps
            tracked = {}
            for name in files:
                node = tracked
                parts = name.split("/")
                for part in parts[:-1]:
                    node = node.setdefault(part, {})
                node[parts[-1]] = None
            have = {}
            for path, digest in want.items():
                node = tracked
                for part in path.split("/"):
                    node = node[part]
                have[path] = _working_id(tree / path, "sha1" if len(digest) == 40 else "sha256", node).hex()
            gaps.append(f"Git history ({why}; native sources checked against the tracked file list)" if not has_git
                        else f"{why} (native sources checked against the tracked file list instead)")
        else:
            if want is None and commit is not None:
                found = git(tree, "cat-file", "-e", f"{commit}^{{commit}}")
                if found.returncode:
                    detail = f": {found.stderr.strip()}" if found.stderr.strip() else ""
                    raise BuildError(f"saved build commit {commit} is no longer available{detail}")
                want = wheel["native_ids"](tree, commit)
            have = wheel["native_ids"](tree)
            if want is None:
                gaps.append("native modules were not checked against a build record (no saved build commit or native snapshot)")
            dirty = git(tree, "status", "--porcelain", "--untracked-files=all", "--", *wheel["NATIVE_SOURCES"])
            if dirty.returncode:
                raise BuildError(f"cannot inspect native inputs: {dirty.stderr.strip()}", _GIT_REMEDY)
            if dirty.stdout.strip():
                # Uncommitted native inputs refuse only when the imported engine is this TREE's build AND a
                # record says what it was built from; otherwise there is nothing to compare them with.
                if not verified_wheel and want is not None and kind in (None, "source"):
                    raise BuildError(
                        f"native inputs have uncommitted changes:\n{dirty.stdout.rstrip()}",
                        "commit or remove the listed native inputs, then run ./setup.sh; "
                        "for deliberate source-build leftovers such as a regenerated Cargo.lock, use the explicit override below",
                    )
                changed = sorted({line[3:] for line in dirty.stdout.splitlines() if line.strip()})
                why = ("which the imported engine is not built from" if verified_wheel or kind not in (None, "source")
                       else "with no build record to compare them with")
                gaps.append(f"uncommitted native inputs in TREE, {why} (" + ", ".join(changed) + ")")
        if want is not None:
            origin = "release/merlin-wheel.native" if verified_wheel else "installed build"
            for path, digest in want.items():
                if have[path] != digest:
                    raise BuildError(f"native input {path} differs from the {origin}: "
                                     f"{path} is {have[path]}, recorded {digest}")
    except wheel["WheelError"] as error:
        raise BuildError(str(error)) from error
    return gaps


def _module_name(member: str) -> str:
    """The dotted module a native member provides, whatever its extension suffix ('vllm/_C.abi3.so' -> 'vllm._C')."""
    head, _, base = member.rpartition("/")
    return ".".join([*head.split("/"), base.split(".", 1)[0]]) if head else base.split(".", 1)[0]


def _finder_spec(directory: Path, fullname: str):
    """Find fullname in one directory exactly as a package __path__ entry would: through this interpreter's
    sys.path_hooks (suffix precedence, packages before modules), without importing any parent package."""
    for hook in sys.path_hooks:
        try:
            finder = hook(str(directory))
        except ImportError:
            continue
        return finder.find_spec(fullname) if hasattr(finder, "find_spec") else None
    return None


def _resolve_module(package_dir: Path, module: str, remedy: str = "run ./setup.sh"):
    """Resolve module ('vllm.x.y') below vllm's own directory with this interpreter's import system, without
    importing anything. Returns the module spec (None if nothing resolves); refuses a subpackage found elsewhere."""
    parts = module.split(".")
    search = package_dir
    for depth in range(1, len(parts) - 1):
        package = _finder_spec(search, ".".join(parts[:depth + 1]))
        locations = [Path(p).resolve() for p in (package.submodule_search_locations or [])] if package else []
        if locations != [(search / parts[depth]).resolve()]:
            if not locations and not (search / parts[depth]).exists():
                return None
            raise BuildError(f"{'.'.join(parts[:depth + 1])}: the serving interpreter would import it from "
                             f"{locations or 'nowhere'}, not {search / parts[depth]}", remedy)
        search = locations[0]
    return _finder_spec(search, module)


def _source_modules(tree: Path) -> dict:
    """For an engine imported from TREE: the file each required extension resolves to in the serving interpreter
    (TREE-relative, every suffix in its own order), or None when no compiled extension resolves."""
    wheel = _wheel_helper()
    package_dir = (tree / "vllm").resolve()
    modules = {}
    for member in wheel["REQUIRED"]:
        module = _module_name(member)
        spec = _resolve_module(package_dir, module)
        if spec is None or not spec.origin:
            modules[module] = None
            continue
        path = Path(spec.origin).resolve()
        if not isinstance(spec.loader, importlib.machinery.ExtensionFileLoader):
            stem = module.rsplit(".", 1)[-1]
            containing = path.parent.parent if path.name.startswith("__init__.") else path.parent
            shadowed = sorted(p.name for p in containing.glob(f"{stem}.*")
                              if any(p.name == stem + s for s in importlib.machinery.EXTENSION_SUFFIXES))
            if shadowed:  # a same-named package or module outranks the compiled extension
                raise BuildError(f"{module}: the serving interpreter would import {path}, not the compiled "
                                 f"{', '.join(shadowed)}")
            modules[module] = None
            continue
        if not path.is_relative_to(tree.resolve()):
            raise BuildError(f"{module}: the serving interpreter would import {path}, outside TREE={tree}")
        modules[module] = path.relative_to(tree.resolve()).as_posix()
    return modules


_FILE_LOADERS = (importlib.machinery.SourceFileLoader, importlib.machinery.SourcelessFileLoader,
                 importlib.machinery.ExtensionFileLoader)


def _flashinfer_import() -> dict:
    """What this interpreter would import as FlashInfer (PYTHONPATH included), without importing it.

    "flashinfer" is the package directory when it is one directory on disk whose files can be checked (a regular
    package, or a namespace package with a single portion). Otherwise "flashinfer" is None and "flashinfer_unverified"
    says what would be imported instead (a single module, several namespace portions, an archive, a custom finder).
    Both are None when nothing named flashinfer can be imported."""
    def unverified(reason: str) -> dict:
        return {"flashinfer": None, "flashinfer_unverified": reason}

    try:
        spec = importlib.util.find_spec("flashinfer")
    except (ImportError, OSError, ValueError) as error:
        return unverified(f"the import lookup failed: {error}")
    if spec is None:
        return {"flashinfer": None, "flashinfer_unverified": None}
    locations = list(spec.submodule_search_locations or [])
    if spec.submodule_search_locations is None:
        return unverified(f"{spec.origin or spec.name} is a single module, not the FlashInfer package")
    if len(locations) != 1:
        return unverified("a namespace package with portions " + (", ".join(locations) or "none"))
    package = Path(locations[0])
    if spec.origin is not None and not isinstance(spec.loader, _FILE_LOADERS):
        return unverified(f"{package} is loaded by {type(spec.loader).__name__}, not from files on disk")
    if not package.is_dir():
        return unverified(f"{package} is not a directory on disk")
    return {"flashinfer": str(package.resolve()), "flashinfer_unverified": None}


def installed_engine(tree: Path, venv: Path) -> dict:
    """Run in the serving interpreter: inspect imports and distribution metadata, never import vllm.

    Every result says what this interpreter would import as FlashInfer (see _flashinfer_import)."""
    def unknown(reason: str) -> dict:
        return {"kind": "unknown", "gap": reason, **_flashinfer_import()}

    try:
        spec = importlib.util.find_spec("vllm")
        origin = Path(spec.origin).resolve() if spec and spec.origin else None
    except (ImportError, OSError, ValueError) as error:
        return unknown(f"cannot locate the imported engine: {error}")
    if origin == (tree / "vllm" / "__init__.py").resolve():
        return {"kind": "source", "modules": _source_modules(tree), **_flashinfer_import()}
    if origin is None:
        return unknown("no installed engine import location")
    if not origin.is_relative_to(venv.resolve()):
        return unknown(f"engine import location {origin} is outside VENV={venv}")
    try:
        dist = md.distribution("vllm")
        direct = dist.read_text("direct_url.json")
        if not direct:
            return unknown("installed distribution has no direct_url.json")
        url = json.loads(direct)
        if not isinstance(url, dict):
            return unknown("installed direct_url.json is not a JSON object")
        if url.get("dir_info", {}).get("editable"):
            return unknown("installed distribution is editable")
        archive = url.get("archive_info", {})
        archive_hash = archive.get("hashes", {}).get("sha256")
        legacy_hash = archive.get("hash", "")
        if archive_hash is None and isinstance(legacy_hash, str) and legacy_hash.startswith("sha256="):
            archive_hash = legacy_hash[len("sha256="):]
        if not isinstance(archive_hash, str):
            return unknown("installed direct_url.json has no archive sha256")
        version = dist.version
        if not isinstance(version, str) or not version:
            return unknown("installed distribution has no readable METADATA version")
        entries = dist.files
        if not entries:
            return unknown("installed distribution has no readable RECORD")
    except (md.PackageNotFoundError, OSError, ValueError, TypeError, AttributeError) as error:
        return unknown(f"installed distribution metadata is unreadable: {error}")

    wheel = _wheel_helper()
    pin = tree / "release" / "merlin-wheel.sha256"
    fields = pin.read_text(encoding="utf-8").split()
    if len(fields) != 2 or not re.fullmatch(r"[0-9a-f]{64}", fields[0]):
        raise BuildError(f"{pin}: expected one '<sha256>  <wheel filename>' record")
    want_hash, filename = fields
    try:
        want_version = wheel["version"](*wheel["parse_name"](Path(filename)))
    except wheel["WheelError"] as error:
        raise BuildError(str(error)) from error
    reinstall = (
        f"install the wheel this checkout pins ({filename}, release/merlin-wheel.sha256): "
        f"MERLIN_WHEEL=<path to {filename}> bash release/install-env.sh {venv} (the README's manual install), "
        f"or {venv}/bin/pip install <path to {filename}> for a pip-only venv; or run ./setup.sh"
    )

    def metadata_path(name: str) -> str:
        return next((str(dist.locate_file(entry)) for entry in entries
                     if str(entry).endswith(".dist-info/" + name)), f"installed vllm {name}")

    if version != want_version:
        raise BuildError(f"{metadata_path('METADATA')}: version {version!r}; {pin} pins {want_version!r}", reinstall)
    if archive_hash.lower() != want_hash:
        raise BuildError(f"{metadata_path('direct_url.json')}: archive sha256 {archive_hash}; {pin} pins {want_hash}",
                         reinstall)
    recorded_origin = Path(dist.locate_file("vllm/__init__.py")).resolve()
    if recorded_origin != origin:
        raise BuildError(f"vllm/__init__.py import location {origin}; distribution records {recorded_origin}", reinstall)
    # Every native member (compiled extensions and the Rust frontend), and the ones every wheel carries.
    native_entries = [entry for entry in entries if wheel["is_native"](str(entry))]
    # A required extension is recognised by the module it provides, whatever its suffix; each RECORD entry
    # below must then be the file the import system loads.
    for name in wheel["REQUIRED"]:
        if not any(_module_name(str(entry)) == _module_name(name) for entry in native_entries):
            raise BuildError(f"{name}: RECORD entries 0 for module {_module_name(name)}; expected 1", reinstall)
    count = sum(1 for entry in native_entries if str(entry) == wheel["FRONTEND"])
    if count != 1:
        raise BuildError(f"{wheel['FRONTEND']}: RECORD entries {count}; expected 1", reinstall)
    frontend_sha256 = None
    seen = set()
    for entry in native_entries:
        name = str(entry)
        if name in seen:
            raise BuildError(f"{name}: RECORD lists it more than once", reinstall)
        seen.add(name)
        recorded = entry.hash
        if recorded is None or recorded.mode != "sha256":
            raise BuildError(f"{name}: RECORD hash {recorded}; expected sha256", reinstall)
        path = Path(dist.locate_file(entry))
        try:
            hexdigest = wheel["sha256"](path)
            have = base64.urlsafe_b64encode(bytes.fromhex(hexdigest)).rstrip(b"=").decode()
        except OSError as error:
            raise BuildError(f"{path}: disk sha256 <unreadable>; RECORD sha256={recorded.value}", reinstall) from error
        if have != recorded.value:
            raise BuildError(f"{path}: disk sha256={have}; RECORD sha256={recorded.value}", reinstall)
        if name == wheel["FRONTEND"]:
            frontend_sha256 = hexdigest.lower()
        # The hashed file must be the one this interpreter's import system loads: a higher-priority
        # extension suffix, a same-named package or a source module beside it would win the import.
        suffix = next((s for s in sorted(importlib.machinery.EXTENSION_SUFFIXES, key=len, reverse=True)
                       if name.endswith(s)), None)
        module = name[:-len(suffix)].replace("/", ".") if suffix else ""
        parts = module.split(".")
        if suffix is None or parts[0] != "vllm" or len(parts) < 2 or not all(p.isidentifier() for p in parts):
            continue  # not an importable engine extension here (the frontend executable, dlopen'd libraries)
        # vllm's own directory is already matched to the distribution above.
        found = _resolve_module(origin.parent, module, reinstall)
        loaded = Path(found.origin).resolve() if found is not None and found.origin else None
        if loaded != path.resolve():
            raise BuildError(f"{module}: the serving interpreter would import {loaded or 'nothing'}, "
                             f"not the RECORD-verified {path}", reinstall)
    native = tree / "release" / "merlin-wheel.native"
    try:
        ids = wheel["read_native"](native)
    except wheel["WheelError"] as error:
        raise BuildError(str(error)) from error
    # FlashInfer is a separate package: report where it is so the launcher can name its patch state.
    # frontend_sha256 is the RECORD-verified hash of the packaged Rust frontend.
    return {"kind": "wheel", "native": ids, **_flashinfer_import(), "frontend_sha256": frontend_sha256}
