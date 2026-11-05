# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Group-consistent tune cache for the FlashInfer PCIe IPC all-reduce.

FlashInfer (0.7.2.post1) persists the PCIe IPC all-reduce tuning in one JSON
file per TP group, written by rank 0 with rank 0's ``_metadata``. Each rank's
``AutoTuner.load_configs`` then compares that metadata with its own, including
``gpu`` = ``torch.cuda.get_device_name()`` (autotuner.py ``_collect_metadata``).
On a TP group that mixes card models (an RTX 3090 with an RTX 3090 Ti: same
arch and VRAM, so one layout) rank 0 accepts its own file and the other rank
rejects it, the per-rank table digests differ, and the workspace's joint check
fails ("every rank must build the workspace with identical arguments"): the
first boot works, every later boot fails.

The fix keeps the tuned choice and makes the load decision a group decision:

1. **The cache path names the group's hardware.** A homogeneous group keeps
   the existing path (so existing caches and the boot-2 reload are unchanged).
   A mixed group gets ``<stem>_mixed-<digest><suffix>``, where the digest
   covers every rank's device name, compute capability and SM count in group
   rank order: exactly the composition the tuning measured.
2. **One load decision for the group.** Before the workspace is built, every
   rank classifies the file against its own environment with FlashInfer's own
   metadata check; the verdicts are gathered over the CPU group. All accept:
   everyone loads. Any rank would reject (or cannot parse) the file: rank 0
   moves it aside and every rank tunes afresh. No group can again split into
   loaders and tuners.
3. **A mixed group's file is readable by every rank.** After tuning, rank 0
   sets ``_metadata.gpu`` to FlashInfer's wildcard ``"*"``. That field is the
   only per-rank one; the path (1) already pins the exact per-rank hardware,
   and every other metadata field (FlashInfer, CUDA, cuBLAS, cuDNN, CuTe-DSL
   versions) is still checked on load.

Lossless: a loaded configuration is the one measured on the same ordered card
composition and software stack; nothing is approximated. A per-boot re-tune
happens only when a file is unusable, and then once (the next save heals it).
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)

METADATA_KEY = "_metadata"
WILDCARD = "*"

ABSENT = "absent"
ACCEPT = "accept"
REJECT = "reject"

LOAD = "load"
TUNE = "tune"
DISCARD = "discard"


def device_identity(device_index: int) -> dict[str, Any]:
    """What about one card can change the tuned choice."""
    import torch

    props = torch.cuda.get_device_properties(device_index)
    return {
        "name": torch.cuda.get_device_name(device_index),
        "cc": f"{props.major}.{props.minor}",
        "sms": int(props.multi_processor_count),
    }


def is_mixed(identities: Sequence[dict[str, Any]]) -> bool:
    return len({json.dumps(i, sort_keys=True) for i in identities}) > 1


def group_digest(identities: Sequence[dict[str, Any]]) -> str:
    """Digest of the per-rank identities in group rank order."""
    blob = json.dumps(list(identities), sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def cache_path_for(base: Path, identities: Sequence[dict[str, Any]]) -> Path:
    """The existing path for a homogeneous group; a hardware-named one for a
    mixed group."""
    if not is_mixed(identities):
        return base
    return base.with_name(f"{base.stem}_mixed-{group_digest(identities)}{base.suffix}")


def _flashinfer_metadata_check() -> tuple[Callable[[], dict], Callable] | None:
    try:
        from flashinfer.autotuner.autotuner import (
            _classify_metadata_mismatches,
            _collect_metadata,
        )
    except ImportError:
        return None
    return _collect_metadata, _classify_metadata_mismatches


def local_verdict(
    path: Path,
    check: tuple[Callable[[], dict], Callable] | None = None,
    device_name: Callable[[], str] | None = None,
) -> str:
    """How this rank's ``AutoTuner.load_configs`` would treat ``path``.

    ``check`` is FlashInfer's ``(_collect_metadata, _classify_metadata_mismatches)``;
    without it (a FlashInfer that moved them) only the per-rank ``gpu`` field
    is compared, which is the field that differs inside one process group.
    """
    if not path.is_file():
        return ABSENT
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return REJECT
    if not isinstance(data, dict):
        return REJECT
    saved = data.get(METADATA_KEY)
    if saved is None:
        return ACCEPT  # FlashInfer loads a file without metadata unchecked
    if check is None:
        check = _flashinfer_metadata_check()
    if check is not None:
        collect, classify = check
        hard, soft = classify(saved, collect())
        return REJECT if (hard or soft) else ACCEPT
    if device_name is None:
        import torch

        def device_name() -> str:
            return torch.cuda.get_device_name(torch.cuda.current_device())

    gpu = saved.get("gpu") if isinstance(saved, dict) else None
    return ACCEPT if gpu in (WILDCARD, device_name()) else REJECT


def agree(verdicts: Sequence[str]) -> str:
    """The group's action from every rank's verdict."""
    if all(v == ACCEPT for v in verdicts):
        return LOAD
    if all(v == ABSENT for v in verdicts):
        return TUNE
    return DISCARD


def set_aside(path: Path) -> Path | None:
    """Move an unusable cache out of the way (one kept, for inspection)."""
    if not path.exists():
        return None
    dest = path.with_name(path.name + ".rejected")
    os.replace(path, dest)
    return dest


def wildcard_gpu(path: Path) -> bool:
    """Set ``_metadata.gpu`` to the wildcard, atomically; True if changed."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return False
    meta = data.get(METADATA_KEY) if isinstance(data, dict) else None
    if not isinstance(meta, dict) or meta.get("gpu") == WILDCARD:
        return False
    meta["gpu"] = WILDCARD
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{time.monotonic_ns()}.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)
    return True


def prepare(
    base: Path,
    rank: int,
    identity: dict[str, Any],
    gather: Callable[[Any], list[Any]],
    barrier: Callable[[], None],
    check: tuple[Callable[[], dict], Callable] | None = None,
) -> tuple[Path, str, bool]:
    """Collective, before the workspace is built. Returns ``(path, action,
    mixed)``; every rank gets the same three values.

    ``gather`` is an all-gather of one picklable object over the TP group's CPU
    group, in group rank order; ``barrier`` a barrier on it.
    """
    identities = gather(identity)
    mixed = is_mixed(identities)
    path = cache_path_for(base, identities)
    try:
        verdict = local_verdict(path, check)
    except Exception as e:  # noqa: BLE001 -- an unreadable file is a reject
        logger.warning("PCIe IPC tune cache %s: check failed (%s)", path, e)
        verdict = REJECT
    verdicts = gather(verdict)
    action = agree(verdicts)
    if action == DISCARD:
        if rank == 0:
            dest = set_aside(path)
            logger.warning(
                "PCIe IPC tune cache %s is not usable on every rank of this TP "
                "group (verdicts %s); moved to %s and re-tuning on every rank.",
                path,
                verdicts,
                dest,
            )
        barrier()
    if mixed and rank == 0:
        logger.info(
            "PCIe IPC all-reduce: mixed TP group (%s); tune cache %s.",
            ", ".join(i["name"] for i in identities),
            path,
        )
    return path, action, mixed


def finish(path: Path, rank: int, mixed: bool, barrier: Callable[[], None]) -> None:
    """Collective, after ``tune()`` (which saved on rank 0): make a mixed
    group's file loadable by every rank."""
    if mixed and rank == 0 and path.is_file():
        wildcard_gpu(path)
    barrier()
