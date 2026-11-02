# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Check the Merlin engine wheel (stdlib only).

  wheel.py audit  WHEEL [--forbid TEXT ...]   release gate run by build-wheel.sh
  wheel.py verify WHEEL RECORD ENGINE         install gate run by install-env.sh

The wheel's local version names the commit it was built from; RECORD
(release/merlin-wheel.sha256) holds "<sha256>  <filename>" for the one wheel
this checkout accepts. release/merlin-wheel.native, next to RECORD, holds one
"<git object id>  <path>" line per NATIVE_SOURCES path, taken at that commit:
verify accepts the wheel only while `git rev-parse HEAD:<path>` in the checkout
gives the same ids, so the build commit itself need not be in the checkout's
history.
"""

import argparse
import email.parser
import hashlib
import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

NAME_RE = re.compile(
    r"vllm-0\.32\.0\+merlin\.(?P<commit>[0-9a-f]{10})-cp313-cp313-linux_x86_64\.whl"
)
REQUIRED = (
    "vllm/_C_stable_libtorch.abi3.so",
    "vllm/_moe_C_stable_libtorch.abi3.so",
    "vllm/cumem_allocator.abi3.so",
    "vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so",
)
# A change under these paths changes the compiled products (CUDA/C++ extensions and the Rust frontend).
# release/merlin-wheel.native records their git object ids at the wheel's build commit.
NATIVE_SOURCES = (
    "csrc", "cmake", "CMakeLists.txt", "setup.py",
    "rust", "rust-toolchain.toml", "tools/build_rust.py",
)
# Authorship and project fields must be exactly upstream vLLM's plus the Merlin modifications author
# (pyproject.toml): the fork adds no maintainer or URL. The package carries LICENSE and the Merlin NOTICE.
METADATA_EXPECTED = {
    "Author": ["vLLM Team, halt95 (Merlin modifications)"],
    "License-File": ["LICENSE", "NOTICE"],
    "Author-email": [],
    "Maintainer": [],
    "Maintainer-email": [],
    "Summary": ["A high-throughput and memory-efficient inference and serving engine for LLMs"],
    "Home-page": [],
    "Project-URL": [
        "Homepage, https://github.com/vllm-project/vllm",
        "Documentation, https://docs.vllm.ai/en/latest/",
        "Slack, https://slack.vllm.ai/",
    ],
}
FORBIDDEN = ("/home/", "/root/")


class WheelError(Exception):
    pass


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def parse_name(wheel: Path) -> str:
    m = NAME_RE.fullmatch(wheel.name)
    if not m:
        raise WheelError(
            f"{wheel.name} is not a Merlin engine wheel name "
            "(vllm-0.32.0+merlin.<commit>-cp313-cp313-linux_x86_64.whl)"
        )
    return m["commit"]


def members(wheel: Path) -> list[zipfile.ZipInfo]:
    try:
        with zipfile.ZipFile(wheel) as z:
            return z.infolist()
    except zipfile.BadZipFile as e:
        raise WheelError(f"{wheel.name} is not a zip archive: {e}") from e


def check_members(names: set[str]) -> None:
    missing = [n for n in REQUIRED if n not in names]
    if missing:
        raise WheelError(f"wheel lacks {', '.join(missing)}")
    cached = sorted(n for n in names if n.endswith(".pyc") or "/__pycache__/" in n)
    if cached:
        raise WheelError(f"wheel carries bytecode: {cached[0]} (+{len(cached) - 1})")


def runpaths(so: Path) -> list[str]:
    out = subprocess.run(
        ["readelf", "-d", str(so)], capture_output=True, text=True, check=True
    ).stdout
    return re.findall(r"\((?:RPATH|RUNPATH)\)\s+Library r(?:un)?path: \[(.*)\]", out)


def audit(wheel: Path, forbid: list[str], readme: Path | None = None) -> list[str]:
    parse_name(wheel)
    infos = members(wheel)
    check_members({i.filename for i in infos})
    needles = [t.encode() for t in (*FORBIDDEN, *forbid)]
    problems = []
    with zipfile.ZipFile(wheel) as z, tempfile.TemporaryDirectory() as tmp:
        for info in infos:
            name = info.filename
            if name.endswith(".dist-info/METADATA"):
                raw = z.read(info)
                msg = email.parser.BytesParser().parsebytes(raw)
                for field, want in METADATA_EXPECTED.items():
                    have = msg.get_all(field, [])
                    if have != want:
                        problems.append(f"{name}: {field} is {have!r}, expected {want!r}")
                # the description must be the release tree's README byte for byte (the public-tree scan covers it)
                body = raw.split(b"\n\n", 1)[1] if b"\n\n" in raw else b""
                if readme is not None and body != readme.read_bytes():
                    problems.append(f"{name}: the description is not {readme}")
                problems += [f"{name}: contains {n.decode()!r}" for n in needles if n in raw]
            if not (name.endswith(".so") or ".so." in name or name == "vllm/vllm-rs"):
                continue
            data = z.read(info)
            for needle in needles:
                if needle in data:
                    problems.append(
                        f"{name}: contains {needle.decode()!r} ({data.count(needle)}x)"
                    )
            so = Path(tmp) / "lib.so"
            so.write_bytes(data)
            for rp in runpaths(so):
                if any(p and not p.startswith("$ORIGIN") for p in rp.split(":")):
                    problems.append(f"{name}: RPATH/RUNPATH {rp}")
    return problems


def git(engine: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(engine), *args], capture_output=True, text=True
    )


OBJECT_ID_RE = re.compile(r"[0-9a-f]{40}(?:[0-9a-f]{24})?")


def read_native(native: Path) -> dict[str, str]:
    """{path: git object id} from release/merlin-wheel.native, one entry per NATIVE_SOURCES path."""
    if not native.is_file():
        raise WheelError(
            f"{native} is missing; it records the native source tree ids the wheel was built from"
        )
    ids: dict[str, str] = {}
    for n, line in enumerate(native.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 2 or not OBJECT_ID_RE.fullmatch(fields[0]):
            raise WheelError(f"{native}:{n} must be one '<git object id>  <path>' line")
        if fields[1] in ids:
            raise WheelError(f"{native}:{n} lists {fields[1]} twice")
        ids[fields[1]] = fields[0]
    missing = [p for p in NATIVE_SOURCES if p not in ids]
    extra = sorted(set(ids) - set(NATIVE_SOURCES))
    if missing or extra:
        raise WheelError(
            f"{native} must list exactly the native source paths {', '.join(NATIVE_SOURCES)}"
            + (f"; missing {', '.join(missing)}" if missing else "")
            + (f"; unexpected {', '.join(extra)}" if extra else "")
        )
    return ids


def check_native(engine: Path, native: Path, commit: str) -> None:
    """The checkout's native sources (at HEAD) must be the ones the wheel was built from."""
    want = read_native(native)
    for path in NATIVE_SOURCES:
        have = git(engine, "rev-parse", "--verify", "--quiet", f"HEAD:{path}")
        if have.returncode:
            raise WheelError(
                f"cannot read {path} at HEAD in {engine} (is it a git checkout with {path} committed?); "
                "build from source instead"
            )
        got = have.stdout.strip()
        if got != want[path]:
            raise WheelError(
                f"native sources differ from the ones wheel commit {commit} was built from: "
                f"{path} is {got} in this checkout, {want[path]} in {native.name}; "
                "build from source or use the wheel built from this checkout"
            )


def verify(wheel: Path, record: Path, engine: Path) -> str:
    if not record.is_file() or not record.read_text(encoding="utf-8").strip():
        raise WheelError(
            f"{record} records no release wheel for this checkout; "
            "build from source instead"
        )
    fields = record.read_text(encoding="utf-8").split()
    if len(fields) != 2 or not re.fullmatch(r"[0-9a-f]{64}", fields[0]):
        raise WheelError(f"{record} must hold one '<sha256>  <filename>' line")
    want, name = fields
    if wheel.name != name:
        raise WheelError(f"wheel is {wheel.name}; this checkout accepts {name}")
    got = sha256(wheel)
    if got != want:
        raise WheelError(f"{wheel.name}: sha256 {got} does not match the recorded {want}")
    commit = parse_name(wheel)
    check_native(engine, record.with_name("merlin-wheel.native"), commit)
    check_members({i.filename for i in members(wheel)})
    return commit


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("audit")
    a.add_argument("wheel", type=Path)
    a.add_argument("--forbid", action="append", default=[])
    a.add_argument("--readme", type=Path, help="the README the wheel's description must equal")
    v = sub.add_parser("verify")
    v.add_argument("wheel", type=Path)
    v.add_argument("record", type=Path)
    v.add_argument("engine", type=Path)
    args = ap.parse_args(argv)
    try:
        if args.cmd == "audit":
            problems = audit(args.wheel, args.forbid, args.readme)
            for p in problems:
                print(f"wheel audit: {p}", file=sys.stderr)
            if problems:
                return 1
            print(f"WHEEL AUDIT OK: {args.wheel.name}")
        else:
            commit = verify(args.wheel, args.record, args.engine)
            print(f"WHEEL OK: {args.wheel.name} (commit {commit})")
    except (WheelError, OSError, subprocess.CalledProcessError) as e:
        print(f"wheel: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
