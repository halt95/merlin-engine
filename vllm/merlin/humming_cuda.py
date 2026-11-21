# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Humming's JIT on torch's own CUDA.

humming-kernels 0.1.19 (``humming/utils/cuda.py``) lists CUDA environments in
``find_all_cuda_paths``: ``/usr/local/cuda`` and ``/usr/local/cuda-*`` first,
then CUDA_HOME / CUDA_PATH, then the PyPI ``nvidia/cu13`` tree (its minor
unknown). ``filter_cuda_paths`` keeps torch's CUDA *major* only and takes the
highest known minor, so a system CUDA 13.3 wins over the venv's 13.0 runtime
that torch (cu130) ships with. NVRTC 13.3 then fails to open
``libnvrtc-builtins.so.13.3`` in the compile helper (``/usr/local/cuda/lib64``
is not on the loader path), and a different-minor NVRTC and headers would be
the wrong toolchain even if it loaded.

``install()`` runs before the first humming symbol is used (vllm/utils/humming.py)
and does two things:

1. Wraps ``find_all_cuda_paths`` so a system environment of torch's major with
   a known, different minor is dropped (a system CUDA matching torch's
   major.minor stays, and is still preferred; the PyPI tree is next). The
   wrapper replaces the module attribute, which ``filter_cuda_paths`` and
   ``get_cuda_library_path`` look up at call time, and every already-imported
   ``humming.*`` module that bound the original by ``from ... import``.
   Filtering the one list every chooser reads is what keeps NVRTC, its
   headers and the launcher's headers on one toolchain; patching only the
   NVRTC path would mix a 13.0 compiler with 13.3 headers.
2. Prepends the directory holding torch's ``libnvrtc-builtins.so.M.m`` to
   ``LD_LIBRARY_PATH`` so the ``nvrtc_compile`` helper subprocess, which
   inherits the environment, resolves the builtins NVRTC dlopens by soname.

Pure helpers (``torch_cuda_version``, ``keep_matching``, ``builtins_dir``) carry
the rules and are tested on CPU (tests/merlin_cpu/test_humming_cuda_match.py,
development tree).
"""

from __future__ import annotations

import functools
import glob
import os
import re
import sys
from collections.abc import Callable, MutableMapping
from typing import Any

_MARK = "_merlin_cuda_match"
_LIB_DIRS = ("lib64", "lib")


def torch_cuda_version() -> tuple[int, int] | None:
    """torch's CUDA (major, minor), or None for a CPU / ROCm build."""
    try:
        import torch
    except ImportError:
        return None
    return parse_version(torch.version.cuda)


def parse_version(text: str | None) -> tuple[int, int] | None:
    m = re.match(r"\s*(\d+)\.(\d+)", text or "")
    return (int(m.group(1)), int(m.group(2))) if m else None


def keep_matching(
    envs: list[dict[str, Any]], target: tuple[int, int]
) -> list[dict[str, Any]]:
    """``envs`` without the system environments of torch's major whose known minor
    differs. Other majors and environments of unknown minor (the PyPI trees) are
    kept: humming's own filter already rejects other majors."""
    major, minor = target
    out = []
    for env in envs:
        if (
            env.get("source") == "system"
            and env.get("major") == major
            and env.get("minor") is not None
            and env.get("minor") != minor
        ):
            continue
        out.append(env)
    return out


def builtins_dir(envs: list[dict[str, Any]], target: tuple[int, int]) -> str | None:
    """The first lib directory of an environment of torch's major that holds
    ``libnvrtc-builtins.so.M.m`` for torch's major.minor (system environments of
    that minor first, as humming orders them), or None."""
    major, minor = target
    for env in envs:
        if env.get("major") != major or env.get("minor") not in (None, minor):
            continue
        root = env.get("path") or ""
        for sub in _LIB_DIRS:
            d = os.path.join(root, sub)
            if glob.glob(os.path.join(d, f"libnvrtc-builtins.so.{major}.{minor}*")):
                return d
    return None


def _prepend_ld_path(directory: str, environ: MutableMapping[str, str]) -> None:
    parts = [p for p in (environ.get("LD_LIBRARY_PATH") or "").split(os.pathsep) if p]
    if parts and parts[0] == directory:
        return
    environ["LD_LIBRARY_PATH"] = os.pathsep.join(
        [directory, *[p for p in parts if p != directory]]
    )


def _wrap(original: Callable[[], list], target: tuple[int, int]) -> Callable[[], list]:
    @functools.lru_cache(maxsize=1)
    def find_all_cuda_paths():
        return keep_matching(list(original()), target)

    setattr(find_all_cuda_paths, _MARK, True)
    find_all_cuda_paths.__wrapped__ = original
    return find_all_cuda_paths


def install(
    modules: MutableMapping[str, Any] | None = None,
    environ: MutableMapping[str, str] | None = None,
    target: tuple[int, int] | None = None,
) -> bool:
    """Point humming's CUDA discovery at torch's major.minor; True when it is (now or
    already) in place. A no-op (False) without humming, without a CUDA torch, or
    when humming has no ``find_all_cuda_paths``."""
    modules = sys.modules if modules is None else modules
    environ = os.environ if environ is None else environ
    target = target or torch_cuda_version()
    if target is None:
        return False
    cuda = modules.get("humming.utils.cuda")
    if cuda is None:
        try:
            import importlib

            cuda = importlib.import_module("humming.utils.cuda")
        except Exception:
            return False
    original = getattr(cuda, "find_all_cuda_paths", None)
    if original is None:
        return False
    if getattr(original, _MARK, False):
        return True
    wrapped = _wrap(original, target)
    for name, module in list(modules.items()):
        bound = getattr(module, "find_all_cuda_paths", None)
        if (name == "humming" or name.startswith("humming.")) and bound is original:
            module.find_all_cuda_paths = wrapped
    cuda.find_all_cuda_paths = wrapped
    try:
        directory = builtins_dir(wrapped(), target)
    except Exception:
        # discovery itself failing is humming's error to raise, at its own call
        directory = None
    if directory:
        _prepend_ld_path(directory, environ)
    return True
