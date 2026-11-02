# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Opt-in validation and telemetry around Triton's native disk autotuner."""

import inspect
import json
import sys
from collections import defaultdict
from contextlib import ExitStack
from contextvars import ContextVar
from functools import wraps
from pathlib import Path

from filelock import FileLock

from vllm.logger import init_logger
from vllm.triton_utils.autotune_cache import (
    META_FILE,
    SCHEMA,
    cache_identity,
    cache_root,
    digest,
    namespace,
    publication_lock,
    validate_record,
)

logger = init_logger(__name__)
_context: ContextVar = ContextVar("flash_next_autotune", default=None)
_counts: dict[str, dict[str, set]] = defaultdict(
    lambda: {"loaded": set(), "fresh": set()}
)
_installed = False


def record_result(family: str, specialization: str, fresh: bool) -> None:
    counts = _counts[family]
    if fresh:
        counts["fresh"].add(specialization)
        counts["loaded"].discard(specialization)
    elif specialization not in counts["fresh"]:
        counts["loaded"].add(specialization)


def log_autotune_summary() -> None:
    for family, counts in sorted(_counts.items()):
        logger.info(
            "autotune: %d specializations loaded from cache / %d tuned fresh "
            "family=%s",
            len(counts["loaded"]), len(counts["fresh"]), family,
        )


class GuardedCache:
    def __init__(self, manager, fingerprint: dict, family: str, kernel: str):
        self.manager = manager
        self.fingerprint = fingerprint
        self.family = family
        self.kernel = kernel

    def get_file(self, filename: str):
        path = self.manager.get_file(filename)
        if path is None:
            return None
        try:
            metadata_path = self.manager.get_file(META_FILE)
            if metadata_path is None:
                raise ValueError("missing fingerprint")
            record = {
                "metadata": json.loads(Path(metadata_path).read_text(encoding="utf-8")),
                "payload": Path(path).read_text(encoding="utf-8"),
            }
            validate_record(record, self.fingerprint)
            if record["metadata"]["filename"] != filename:
                raise ValueError("autotune filename mismatch")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            logger.warning(
                "autotune: cached winner rejected; retuning family=%s kernel=%s: %s",
                self.family, self.kernel, exc,
            )
            return None
        return path

    def put(self, data, filename: str, *args, **kwargs):
        path = self.manager.put(data, filename, *args, **kwargs)
        payload = Path(path).read_text(encoding="utf-8")
        json.loads(payload)
        metadata = {
            "schema": SCHEMA,
            "fingerprint": self.fingerprint,
            "family": self.family,
            "kernel": self.kernel,
            "filename": filename,
            "payload_sha256": digest(payload.encode()),
        }
        self.manager.put(json.dumps(metadata, sort_keys=True), META_FILE, binary=False)
        return path


def _patch_native(native, fingerprint_for, is_capturing) -> None:
    cls = native.Autotuner
    original_init = cls.__init__
    original_check = cls.check_disk_cache
    original_manager = native.get_cache_manager
    init_signature = inspect.signature(original_init)
    if "cache_results" not in init_signature.parameters:
        raise RuntimeError("unsupported Triton autotuner: no cache_results")
    # Triton 3.7.1 names the second parameter ``tuning_key``; the call site in
    # Autotuner.run passes all three positionally, so only the shape matters.
    check_params = list(inspect.signature(original_check).parameters)
    if len(check_params) != 4 or check_params[0] != "self" or check_params[1] not in (
        "key", "tuning_key"
    ) or check_params[2:] != ["configs", "bench_fn"]:
        raise RuntimeError(
            f"unsupported Triton check_disk_cache signature {check_params}"
        )
    source = inspect.getsource(original_check)
    if not all(token in source for token in ["get_cache_manager", ".get_file", ".put"]):
        raise RuntimeError("unsupported Triton native disk cache implementation")

    @wraps(original_init)
    def initialize(self, *args, **kwargs):
        bound = init_signature.bind(self, *args, **kwargs)
        bound.arguments["cache_results"] = True
        original_init(*bound.args, **bound.kwargs)
        if not getattr(self, "cache_results", False):
            raise RuntimeError("Triton did not enable native disk autotuning")

    def get_manager(native_key):
        context = _context.get()
        if context is None:
            return original_manager(native_key)
        manager = original_manager(namespace(native_key))
        directory = Path(manager.cache_dir)
        if directory.parent.resolve() != cache_root().resolve():
            raise RuntimeError("native Triton cache root differs from config snapshot")
        context["stack"].enter_context(
            FileLock(str(directory / ".flash_next_autotune.lock"))
        )
        context["managers"].add(native_key)
        return GuardedCache(
            manager, {**context["fingerprint"], "native_key": native_key},
            context["family"], context["kernel"]
        )

    @wraps(original_check)
    def check(self, key, configs, bench_fn):
        if is_capturing():
            raise RuntimeError("autotune cache miss during CUDA capture; warm eagerly")
        fingerprint, family, kernel = fingerprint_for(self)
        fresh = False

        def bench(*args, **kwargs):
            nonlocal fresh
            fresh = True
            return bench_fn(*args, **kwargs)

        with ExitStack() as stack:
            # Lock order is root publication lock, then specialization lock.
            stack.enter_context(publication_lock(cache_root()))
            context = {
                "stack": stack, "fingerprint": fingerprint,
                "family": family, "kernel": kernel, "managers": set(),
            }
            token = _context.set(context)
            try:
                result = original_check(self, key, configs, bench)
                if not context["managers"]:
                    raise RuntimeError("Triton bypassed native disk autotuning")
                for specialization in context["managers"]:
                    record_result(family, specialization, fresh)
            finally:
                _context.reset(token)
        logger.debug(
            "autotune: %s family=%s kernel=%s key=%s",
            "tuned fresh" if fresh else "loaded from cache", family, kernel, key,
        )
        return result

    # Enable already decorated kernels as well as later/lazily imported ones.
    existing = []
    for module in list(sys.modules.values()):
        if module is None:
            continue
        for value in list(vars(module).values()):
            seen = set()
            while isinstance(value, (cls, native.Heuristics)):
                if id(value) in seen:
                    break
                seen.add(id(value))
                if isinstance(value, cls):
                    existing.append(value)
                value = getattr(value, "fn", None)
    for tuner in existing:
        if tuner.cache:
            raise RuntimeError(
                "install persisted autotuning before the first kernel run"
            )
    native.get_cache_manager = get_manager
    cls.__init__ = initialize
    cls.check_disk_cache = check
    for tuner in existing:
        tuner.cache_results = True


def _runtime_compatibility(target) -> dict:
    from triton.backends.nvidia import compiler

    from vllm.third_party import pynvml

    pynvml.nvmlInit()
    try:
        driver_version = pynvml.nvmlSystemGetDriverVersion()
    finally:
        pynvml.nvmlShutdown()
    if isinstance(driver_version, bytes):
        driver_version = driver_version.decode("ascii")
    if not isinstance(driver_version, str) or not driver_version:
        raise RuntimeError("cannot fingerprint the NVIDIA driver")

    # Use the compiler's resolver, including Triton's tool/arch overrides.
    # torch.version.cuda and a PATH nvcc do not identify this compiler.
    resolver = getattr(compiler, "get_ptxas", None)
    if not callable(resolver) or "get_ptxas" not in inspect.getsource(
        compiler.CUDABackend.make_cubin
    ):
        raise RuntimeError("unsupported Triton CUDA toolchain resolver")
    tool = resolver(target.arch)
    path = getattr(tool, "path", None)
    version = getattr(tool, "version", None)
    if not isinstance(path, str) or not isinstance(version, str) or not version:
        raise RuntimeError("unsupported Triton resolved ptxas identity")
    return {
        "driver_version": driver_version,
        "cuda_toolchain": {
            "ptxas_version": version,
            "ptxas_sha256": digest(Path(path).read_bytes()),
        },
    }


def enable_persisted_autotune(config) -> None:
    global _installed
    if not config.persist_triton_autotune:
        return
    if _installed:
        return
    import torch
    import triton
    import triton.runtime.autotuner as native

    if triton.__version__ != "3.7.1":
        raise RuntimeError("persisted autotuning requires qualified Triton 3.7.1")
    target = triton.runtime.driver.active.get_current_target()
    if target.backend != "cuda" or target.arch != 86:
        raise RuntimeError("persisted autotuning requires qualified CUDA sm_86")
    if cache_identity() != config.triton_autotune_cache_identity:
        raise RuntimeError("autotune cache changed after KernelConfig initialization")
    compatibility = _runtime_compatibility(target)
    native_source_hash = digest(inspect.getsource(native).encode())
    compiler_key = getattr(native, "triton_key", None)
    if not callable(compiler_key):
        raise RuntimeError("unsupported Triton native compiler fingerprint")
    compiler_hash = compiler_key()
    if not isinstance(compiler_hash, str):
        raise RuntimeError("unsupported Triton compiler fingerprint format")

    def fingerprint_for(tuner):
        jit = tuner.fn
        while not hasattr(jit, "cache_key"):
            jit = jit.fn
        base = jit.fn
        current = triton.runtime.driver.active.get_current_target()
        if current != target:
            raise RuntimeError("autotune target changed after worker initialization")
        fingerprint = {
            **compatibility,
            "triton_version": triton.__version__,
            "arch": "sm_86",
            "target": str(current),
            "device": torch.cuda.get_device_name(),
            "kernel_source_hash": digest(str(jit.cache_key).encode()),
            "native_autotuner_hash": native_source_hash,
            "compiler_hash": compiler_hash,
        }
        return fingerprint, base.__module__, base.__name__

    _patch_native(native, fingerprint_for, torch.cuda.is_current_stream_capturing)
    _installed = True
    logger.info(
        "autotune: native persistence enabled triton=3.7.1 arch=sm_86 "
        "cache_identity=%s",
        config.triton_autotune_cache_identity,
    )
