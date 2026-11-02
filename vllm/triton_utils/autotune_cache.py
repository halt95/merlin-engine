# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Portable, fingerprinted native Triton autotune records (no GPU imports)."""

import argparse
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

from filelock import FileLock

SCHEMA = 2
META_FILE = "flash_next.autotune.meta.json"


def cache_root() -> Path:
    home = Path(os.environ.get("TRITON_HOME", Path.home() / ".triton"))
    return Path(os.environ.get("TRITON_CACHE_DIR") or home / "cache").expanduser()


def publication_lock(root: Path) -> FileLock:
    root.mkdir(parents=True, exist_ok=True)
    return FileLock(str(root / ".flash_next_autotune.publication.lock"))


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def namespace(native_key: str) -> str:
    return digest(f"flash-next-autotune-{SCHEMA}:{native_key}".encode())


def validate_record(record: dict, fingerprint: dict | None = None) -> None:
    metadata = record["metadata"]
    if metadata["schema"] != SCHEMA:
        raise ValueError("autotune schema mismatch")
    if fingerprint is not None and metadata["fingerprint"] != fingerprint:
        raise ValueError("autotune fingerprint mismatch")
    compatibility = metadata["fingerprint"]
    if not isinstance(compatibility.get("driver_version"), str) or not compatibility[
        "driver_version"
    ]:
        raise ValueError("missing NVIDIA driver fingerprint")
    toolchain = compatibility.get("cuda_toolchain")
    if not isinstance(toolchain, dict) or not all(
        isinstance(toolchain.get(field), str) and toolchain[field]
        for field in ("ptxas_version", "ptxas_sha256")
    ):
        raise ValueError("missing resolved CUDA toolchain fingerprint")
    if digest(record["payload"].encode()) != metadata["payload_sha256"]:
        raise ValueError("autotune payload checksum mismatch")
    json.loads(record["payload"])


def read_records(root: Path) -> list[dict]:
    records = []
    for path in sorted(root.glob(f"*/{META_FILE}")):
        metadata = json.loads(path.read_text(encoding="utf-8"))
        payload = (path.parent / safe_filename(metadata["filename"])).read_text(encoding="utf-8")
        record = {
            "directory": path.parent.name,
            "metadata": metadata,
            "payload": payload,
        }
        validate_record(record)
        records.append(record)
    return records


def cache_identity(root: Path | None = None) -> str:
    # Include corrupt entries too: replacing any owned record invalidates AOT.
    entries = []
    for path in sorted((root or cache_root()).glob(f"*/{META_FILE}")):
        files = sorted(path.parent.glob("*.json"))
        entries.append([
            path.parent.name,
            [(f.name, digest(f.read_bytes())) for f in files],
        ])
    return digest(json.dumps(entries, sort_keys=True).encode())


def safe_filename(name: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or name in {".", ".."}:
        raise ValueError("invalid autotune filename")
    return name


def atomic_write(path: Path, content: str) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as file:
        temporary = Path(file.name)
        file.write(content)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def export_records(root: Path, destination: Path) -> None:
    destination.write_text(json.dumps(
        {"schema": SCHEMA, "records": read_records(root)}, indent=2
    ) + "\n", encoding="utf-8")


def import_records(root: Path, source: Path) -> int:
    bundle = json.loads(source.read_text(encoding="utf-8"))
    if bundle["schema"] != SCHEMA:
        raise ValueError("autotune bundle schema mismatch")
    records = bundle["records"]
    seen = set()
    # Validate the entire bundle before writing anything.
    for record in records:
        validate_record(record)
        directory = record["directory"]
        if not re.fullmatch(r"[A-Za-z0-9_-]+", directory):
            raise ValueError("invalid autotune directory")
        metadata = record["metadata"]
        safe_filename(metadata["filename"])
        if not metadata["filename"].endswith(".autotune.json"):
            raise ValueError("invalid native autotune filename")
        if metadata["fingerprint"]["triton_version"] != "3.7.1":
            raise ValueError("release winners require Triton 3.7.1")
        if metadata["fingerprint"]["arch"] != "sm_86":
            raise ValueError("release winners require sm_86")
        if directory in seen:
            raise ValueError("duplicate autotune directory")
        seen.add(directory)
    with publication_lock(root):
        pending = []
        # Publishers take this same root lock, so this preflight stays valid.
        for record in records:
            directory = root / record["directory"]
            existing = directory / META_FILE
            if existing.exists():
                old_metadata = json.loads(existing.read_text(encoding="utf-8"))
                old_payload = (
                    directory / safe_filename(old_metadata["filename"])
                ).read_text(encoding="utf-8")
                if (
                    old_metadata != record["metadata"]
                    or old_payload != record["payload"]
                ):
                    raise ValueError("conflicting canonical winner; use an empty cache")
            else:
                payload_path = directory / record["metadata"]["filename"]
                if payload_path.exists():
                    raise ValueError("incomplete canonical winner; use an empty cache")
                pending.append(record)
        for record in pending:
            directory = root / record["directory"]
            directory.mkdir(parents=True, exist_ok=True)
            atomic_write(directory / record["metadata"]["filename"], record["payload"])
            atomic_write(directory / META_FILE, json.dumps(
                record["metadata"], sort_keys=True
            ))
    return len(records)


def printable_records(root: Path) -> list[dict]:
    records = read_records(root)
    for record in records:
        native = json.loads(record.pop("payload"))
        timings = native.get("configs_timings") if isinstance(native, dict) else None
        if timings:
            # Reporting only; native Triton still selects the serving choice.
            record["minimum_config"], record["timing"] = min(
                timings, key=lambda entry: entry[1]
            )
        else:
            record["native_record"] = native
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=["export", "import", "print", "native-source"]
    )
    parser.add_argument("bundle", nargs="?", type=Path)
    parser.add_argument("--cache-dir", type=Path, default=cache_root())
    args = parser.parse_args()
    if args.action == "native-source":
        import inspect

        import triton
        import triton.runtime.autotuner as native

        print("Triton", triton.__version__)
        print("cache dir:", cache_root())
        print("autotune signature:", inspect.signature(triton.autotune))
        print("autotuning.cache:", triton.knobs.autotuning.cache)
        print(inspect.getsource(native.Autotuner.check_disk_cache))
    elif args.action == "print":
        print(json.dumps({"identity": cache_identity(args.cache_dir),
                          "records": printable_records(args.cache_dir)}, indent=2))
    elif args.bundle is None:
        parser.error("export/import requires a bundle filename")
    elif args.action == "export":
        export_records(args.cache_dir, args.bundle)
    else:
        print("Imported", import_records(args.cache_dir, args.bundle), "records")


if __name__ == "__main__":
    main()
