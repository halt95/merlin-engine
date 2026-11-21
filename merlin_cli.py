# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Console-script entry for ``merlin``.

The launchers, profiles and model assets live in the Merlin-engine checkout, not in the wheel, so this entry point
finds the checkout (``$MERLIN_HOME``, else this file's directory, else the current directory or a parent that is owned
by the user and not group or world writable) and says so clearly when there is none.
"""

import os
import sys
from pathlib import Path

_MARK = Path("tools") / "merlin" / "serve_cli.py"


def _euid() -> int | None:
    """The user this process runs as; None where there is no POSIX ownership (Windows)."""
    return os.geteuid() if hasattr(os, "geteuid") else None


def _owner_and_mode(path: Path) -> tuple[int, int]:
    st = path.stat()
    return st.st_uid, st.st_mode & 0o7777


MAX_MODULES = 500  # the most files of one kind read in one directory; more is refused rather than walked
# Engine files the launcher loads from the checkout in the same process: runpy.run_path (the Flash-Next family parser)
# and importlib exec_module (the cold-expert profile checker).
ENGINE_FILES = ("vllm/flashnext_family.py", "vllm/merlin/cold_experts_plan.py")


def _scan(directory: Path, suffix: str) -> list[Path] | str:
    """The files in ``directory`` ending in ``suffix`` (none when it does not exist), or why they cannot be listed."""
    try:
        names = sorted(e.name for e in os.scandir(directory) if e.name.endswith(suffix))
    except FileNotFoundError:
        return []
    except OSError as e:
        return f"{directory} cannot be read ({e.strerror or e})"
    if len(names) > MAX_MODULES:
        return f"{directory} holds more than {MAX_MODULES} {suffix} files"
    return [directory / n for n in names]


def untrusted_reason(root: Path) -> str | None:
    """Why a checkout found from the current directory must not be run, or None when it may be.

    The launch path imports and runs code from the checkout, so everything it loads from there is checked: each must
    be owned by the user running ``merlin`` and writable by that user alone (no group or other write bit), a symlink
    is followed to a target that must lie inside the checkout and pass the same test, and every directory is bounded
    (``MAX_MODULES`` files of a kind). Covered:

    - the checkout directory, ``tools`` and ``tools/merlin``, and every ``*.py`` in ``tools/merlin`` (``serve_cli``
      imports ``launch``, ``preflight``, ``registry``, ``setup_cli`` and the rest from there);
    - ``vllm/flashnext_family.py`` and ``vllm/merlin/cold_experts_plan.py`` (the launcher runs both from the checkout),
      with the ``vllm`` and ``vllm/merlin`` directories that hold them;
    - the bytecode Python would read for those: ``tools/merlin/__pycache__`` and ``vllm/merlin/__pycache__`` (the
      directory and every ``*.pyc`` in it), so a planted cache cannot stand in for a checked source file.

    Not covered, because nothing from it is loaded on this path: ``vllm/flashnext_family.py`` runs from source, so
    ``vllm/__pycache__`` is never read; the ``vllm`` package the launcher imports lazily comes from the Python
    environment (only ``tools/merlin`` is put on ``sys.path``); data files (profiles, templates, recipes, checksums) and
    shell launchers are not imported.

    Where there is no POSIX ownership (Windows: no ``geteuid``, and ``st_uid`` / mode bits do not describe who can
    write) the test cannot be made and does not apply: the working-directory lookup is accepted as before.
    """
    uid = _euid()
    if uid is None:
        return None
    pkg = root / "tools" / "merlin"
    paths = [root, root / "tools", pkg]
    found = _scan(pkg, ".py")
    if isinstance(found, str):
        return found
    paths += found
    for rel in ENGINE_FILES:
        f = root / rel
        if os.path.lexists(f):
            paths += [root / "vllm", f.parent, f]
    for cache in (pkg / "__pycache__", root / "vllm" / "merlin" / "__pycache__"):
        if os.path.lexists(cache):
            found = _scan(cache, ".pyc")
            if isinstance(found, str):
                return found
            paths += [cache, *found]
    inside = root.resolve()
    checked: set[Path] = set()

    def problem(p: Path, shown: Path) -> str | None:
        try:
            owner, mode = _owner_and_mode(p)
        except (OSError, RuntimeError) as e:
            return f"{shown} cannot be read ({getattr(e, 'strerror', None) or e})"
        if owner != uid:
            return f"{shown} is owned by uid {owner}, not by you (uid {uid})"
        if mode & 0o022:
            return f"{shown} is writable by its group or by everyone (mode {mode:04o})"
        return None

    for p in dict.fromkeys(paths):
        try:
            real = p.resolve(strict=True)
        except (OSError, RuntimeError) as e:  # RuntimeError: a link loop
            return f"{p} cannot be read ({getattr(e, 'strerror', None) or e})"
        if real != inside and inside not in real.parents:
            return f"{p} is a link to {real}, outside the checkout"
        why = problem(p, p)
        if why:
            return why
        d = real.parent  # a link's target sits in directories of its own: each one from the target up to the checkout
        while real != inside and d != inside and d not in checked:
            checked.add(d)
            why = problem(d, d)
            if why:
                return f"{why} (it holds {real}, the target of {p})" if real != p else why
            d = d.parent
    return None


def find_checkout(refused: list[tuple[Path, str]] | None = None) -> Path | None:
    """``$MERLIN_HOME``, else this file's directory, else the current directory or a parent that passes
    ``untrusted_reason``. A working-directory candidate that fails it is skipped and, when ``refused`` is given, the
    reason is appended."""
    here = Path(__file__).resolve().parent
    cands = [(Path(os.environ["MERLIN_HOME"]), False)] if os.environ.get("MERLIN_HOME") else []
    cwd = Path.cwd().resolve()
    cands += [(here, False), *((c, True) for c in (cwd, *cwd.parents))]
    for c, from_cwd in cands:
        if not (c / _MARK).is_file():
            continue
        why = untrusted_reason(c) if from_cwd else None
        if why is None:
            return c
        if refused is not None:
            refused.append((c, why))
    return None


def main() -> int:
    refused: list[tuple[Path, str]] = []
    root = find_checkout(refused)
    if root is None:
        for c, why in refused:
            print(
                f"merlin: not running the checkout at {c} (found from the current directory): {why}. If it is yours, "
                f"run: chmod -R go-w {c}. Otherwise use a checkout you own, or set MERLIN_HOME={c} to use this one as it is.",
                file=sys.stderr,
            )
        print(
            "merlin: no Merlin-engine checkout found. Set MERLIN_HOME=<checkout> (it holds tools/merlin, models/ and "
            "the launchers) or run from inside it.",
            file=sys.stderr,
        )
        return 2
    sys.path.insert(0, str(root / "tools" / "merlin"))
    import serve_cli

    serve_cli.R.safe_stdio()
    return serve_cli.main()


if __name__ == "__main__":
    sys.exit(main())
