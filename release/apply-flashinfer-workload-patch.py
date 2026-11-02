#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Opt-in: install both small-q patches from the pinned 0.7.0.post1 wheel.

The default release install applies 0001 only (release/apply-flashinfer-patch.sh).
release/install-env.sh runs this series installer only with
MERLIN_FLASHINFER_WORKLOAD_TILE=1. Run with the stopped engine's venv Python:
    python -I release/apply-flashinfer-workload-patch.py [WHEEL]
    python -I release/apply-flashinfer-workload-patch.py --verify
Installing without WHEEL downloads the exact pinned wheel from PyPI.
Verification is offline: it compares the eight installed files with the
embedded v2 sha256 table and changes nothing.
"""

import argparse
import fcntl
import hashlib
import importlib.metadata as md
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import urllib.request
import zipfile
from pathlib import Path

WHEEL_SHA = "c7adf826568d61fc1b7d3aadd4cae387a35a138bfc08e2deca2f34ecfa280716"
TARGETS = (
    "data/csrc/batch_prefill.cu",
    "data/csrc/batch_prefill_paged_kernel_inst.jinja",
    "data/csrc/batch_prefill_jit_binding.cu",
    "data/include/flashinfer/attention/prefill.cuh",
    "data/include/flashinfer/attention/scheduler.cuh",
    "jit/attention/modules.py",
    "decode.py",
    "prefill.py",
)
# sha256 of every TARGETS file, in order: the pinned wheel (pristine), after 0001
# (v1: the default release install; its five touched files match
# apply-flashinfer-patch.sh), and after 0001+0002 (v2, this opt-in series).
STATE_SHA = {
    "pristine": (
        "4190105e8b5c93621af44356b9aa50861560a5fbf8ff3279301487860604d932",
        "1e41a0ea9795847272674c1058a9d72f220facf296037cbf1f95ce6b77225a63",
        "dad0d81c55b92f71a634cb8240dcb0d150aa6ffd391ef3ef5d87b6207eeca4fc",
        "e66bce2652c0a3c5b54f88510aff2b4b48ff6a52c4ee23fa97f91fa06a935b87",
        "425ba78815ba77ae848c67507d1af6b024d05edd257a7d9b331166afcef9b17c",
        "747c92bd33ecc40297bddfd4b9d388e86e1e70aac5d2d71a37bdfa752c19cca6",
        "df105a90074609961925686572b7b7e4930bf7e065003829f788452607932d6c",
        "bd4aaa0a24462efbb6e6ddb12b3c98da9507e7b6ff6b22da3f40167ca66beb12",
    ),
    "v1": (
        "0e669eee017eb01ef5c4b23915a026c9ad4217a5edce0c99be1926a58b3824c8",
        "500fcecb03e267e6c7bd85c1240ba34581e5e1f9b1c82dd671f1a669f740e041",
        "dad0d81c55b92f71a634cb8240dcb0d150aa6ffd391ef3ef5d87b6207eeca4fc",
        "49cf57d1718eefe01d164a91ce36dd6bcbdf1b831d175fe5723b62a8dcc3ef61",
        "a8a7ccf79e488e7ff41dca6512f4c9bf6cb075ca4dfed0c9fb392de61eef80a4",
        "c38076b0c3c8a1d0e802dc6fc02a68e6da36604210bb03110df8774e220f37b5",
        "df105a90074609961925686572b7b7e4930bf7e065003829f788452607932d6c",
        "bd4aaa0a24462efbb6e6ddb12b3c98da9507e7b6ff6b22da3f40167ca66beb12",
    ),
    "v2": (
        "0a726df1e91cfc814ed606480678050ba90b51902e04650d596eeca16ebf0038",
        "500fcecb03e267e6c7bd85c1240ba34581e5e1f9b1c82dd671f1a669f740e041",
        "a07b3316dd116eac4930ae86fdbe89b7cf5984d1544f248356a7ba071a666a7b",
        "49cf57d1718eefe01d164a91ce36dd6bcbdf1b831d175fe5723b62a8dcc3ef61",
        "c78c74142474f37675a5df3f3ba5b77f8dabc3618e7c6d046279ee438ec194d9",
        "2a44d7b0e60acf3036048dd7ccd14876d60129939341a1cf8bc26b4a9f094765",
        "21202d646f1772844339b0f93e92eb820e08655c171eab6d1391f1f7e8dfcf5f",
        "912b5d7fb801e2af162c0743e079fd9070d27973a28d0deb71257f80bb478151",
    ),
}
PATCHES = (
    "0001-sm8x-fp8-smallq-verify-tile-070.patch",
    "0002-workload-smallq-plan-mode.patch",
)


def tree_state(root: Path) -> str | None:
    """Name the complete state of the eight files under root, or None."""
    digests = tuple(
        hashlib.sha256((root / name).read_bytes()).hexdigest()
        if (root / name).is_file()
        else ""
        for name in TARGETS
    )
    return next((s for s, want in STATE_SHA.items() if digests == want), None)


def check_target() -> tuple[Path, Path]:
    """The venv's purelib and FlashInfer package, after the origin checks."""
    if sys.prefix == sys.base_prefix:
        raise RuntimeError("run with the target venv's Python")
    purelib = Path(sysconfig.get_path("purelib")).resolve()
    package = purelib / "flashinfer"
    distribution = md.distribution("flashinfer-python")
    if (
        distribution.version != "0.7.0.post1"
        or Path(distribution.locate_file("flashinfer")).resolve() != package
    ):
        raise RuntimeError("target must contain FlashInfer 0.7.0.post1 in purelib")
    if '__version__ = "0.7.0.post1"' not in (package / "_build_meta.py").read_text(
        encoding="utf-8"
    ):
        raise RuntimeError("FlashInfer build metadata does not say 0.7.0.post1")
    spec = importlib.util.find_spec("flashinfer")
    if (
        spec is None
        or not spec.origin
        or Path(spec.origin).resolve() != package / "__init__.py"
    ):
        raise RuntimeError("flashinfer import is shadowed")
    if importlib.util.find_spec("flashinfer_jit_cache") is not None:
        raise RuntimeError("uninstall flashinfer-jit-cache")
    try:
        md.distribution("flashinfer-jit-cache")
    except md.PackageNotFoundError:
        pass
    else:
        raise RuntimeError("uninstall flashinfer-jit-cache")
    if list(purelib.glob("flashinfer_jit_cache*")):
        raise RuntimeError("remove stale flashinfer_jit_cache files")

    # Match the serving entry point's import path as well as isolated mode.
    subprocess.run(
        [
            sys.executable,
            "-P",
            "-c",
            (
                "import importlib.util as u, sys\n"
                "from pathlib import Path\n"
                "s = u.find_spec('flashinfer')\n"
                "if s is None or not s.origin or Path(s.origin).resolve() != "
                "Path(sys.argv[1]) / '__init__.py':\n"
                "    sys.exit('flashinfer import is shadowed')\n"
                "if u.find_spec('flashinfer_jit_cache') is not None:\n"
                "    sys.exit('uninstall flashinfer-jit-cache')\n"
            ),
            str(package),
        ],
        check=True,
    )
    return purelib, package


def verify() -> None:
    """Offline: the eight installed files must be the authenticated v2 bytes."""
    purelib, package = check_target()
    lock_path = purelib / ".flashinfer-sm8x-patch.lock"
    if not lock_path.is_file():
        raise RuntimeError(
            "FlashInfer patch series 0001+0002 is missing: no release patch "
            f"installer has run on this venv ({lock_path} is absent)"
        )
    with lock_path.open("r", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_SH)
        state = tree_state(package)
    if state != "v2":
        raise RuntimeError(
            "FlashInfer patch series 0001+0002 is missing (installed tree: "
            f"{state or 'edited or mixed'}); install it with "
            "MERLIN_FLASHINFER_WORKLOAD_TILE=1 release/install-env.sh"
        )
    print("FLASHINFER TILE HASHES OK: authenticated v2, all eight files")


def need_patch_tool() -> None:
    if shutil.which("patch") is None:
        raise RuntimeError("GNU patch is required to apply the series; install the patch package and re-run")


def install(wheel: Path) -> None:
    need_patch_tool()
    if hashlib.sha256(wheel.read_bytes()).hexdigest() != WHEEL_SHA:
        raise RuntimeError("wheel is not the pinned FlashInfer 0.7.0.post1 wheel")
    purelib, package = check_target()
    patches = Path(__file__).resolve().parent / "patches" / "flashinfer-0.7.0.post1"
    lock_path = purelib / ".flashinfer-sm8x-patch.lock"
    with lock_path.open("a", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with tempfile.TemporaryDirectory(prefix="flashinfer-smallq-") as tmp:
            reference = Path(tmp)
            with zipfile.ZipFile(wheel) as archive:
                for name in TARGETS:
                    target = reference / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(archive.read("flashinfer/" + name))

            def snapshot(root):
                return {name: (root / name).read_bytes() for name in TARGETS}

            states = [snapshot(reference)]
            for patch in PATCHES:
                subprocess.run(
                    [
                        "patch",
                        "-p1",
                        "--batch",
                        "--forward",
                        "--fuzz=0",
                        "--no-backup-if-mismatch",
                        "-d",
                        str(reference),
                        "-i",
                        str(patches / patch),
                    ],
                    check=True,
                )
                states.append(snapshot(reference))
            if tree_state(reference) != "v2":
                raise RuntimeError("the patch series does not reproduce the v2 table")
            current = snapshot(package)
            if current not in states:
                raise RuntimeError(
                    "edited or mixed FlashInfer tree; reinstall the pinned wheel"
                )
            expected = states[-1]
            if current == expected:
                print("FlashInfer workload tile patch already installed")
                return

            def replace_file(name, data):
                target = package / name
                with tempfile.NamedTemporaryFile(
                    dir=target.parent, delete=False
                ) as out:
                    temporary = Path(out.name)
                    out.write(data)
                try:
                    os.chmod(temporary, target.stat().st_mode)
                    os.replace(temporary, target)
                finally:
                    temporary.unlink(missing_ok=True)

            try:
                for name, data in expected.items():
                    replace_file(name, data)
                if snapshot(package) != expected or tree_state(package) != "v2":
                    raise RuntimeError("post-install byte comparison failed")
            except BaseException:
                for name, data in current.items():
                    replace_file(name, data)
                raise
            for cache in package.rglob("__pycache__"):
                shutil.rmtree(cache)
    print("FlashInfer workload tile patch installed; selection remains opt-in")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("wheel", nargs="?", type=Path)
    args = parser.parse_args()
    if args.verify:
        if args.wheel is not None:
            parser.error("--verify is offline and takes no wheel")
        verify()
        return
    if args.wheel is not None:
        install(args.wheel.resolve())
        return

    need_patch_tool()  # before the download
    # A local wheel (FLASHINFER_WHEEL for install-env.sh) keeps installs offline.
    with tempfile.TemporaryDirectory(prefix="flashinfer-wheel-") as tmp:
        with urllib.request.urlopen(
            "https://pypi.org/pypi/flashinfer-python/0.7.0.post1/json",
            timeout=60,
        ) as response:
            files = json.load(response)["urls"]
        name = "flashinfer_python-0.7.0.post1-py3-none-any.whl"
        matches = [
            row
            for row in files
            if row["filename"] == name
            and row["digests"]["sha256"] == WHEEL_SHA
            and row["url"].startswith("https://files.pythonhosted.org/")
        ]
        if len(matches) != 1:
            raise RuntimeError("PyPI does not advertise the pinned FlashInfer wheel")
        wheel = Path(tmp) / name
        with (
            urllib.request.urlopen(matches[0]["url"], timeout=60) as source,
            wheel.open("wb") as target,
        ):
            shutil.copyfileobj(source, target)
        install(wheel)


if __name__ == "__main__":
    main()
