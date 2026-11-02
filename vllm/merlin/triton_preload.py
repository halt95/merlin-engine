# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Load warmed Triton kernels onto the device at boot.

``JITFunction.warmup()`` compiles a specialization and caches it, but Triton
loads the binary (``CompiledKernel._init_handles`` -> ``load_binary``:
cuModuleLoadData, cuModuleGetFunction, cuFuncGetAttribute/SetAttribute) only
at the first real launch, from ``launch_metadata``. That first launch can come
in the middle of a forward pass. Under CUDA lazy module loading it then runs
while device work is pending, such as the PLE host-mapped pull's host-flag
wait, and hang the worker (both PP0 ranks stopped in ``load_binary`` under
``qsa_sparse_paged_attention``). Loading every warmed specialization while
the boot warm-up still owns an idle device removes that first-launch load.

Loading only creates the module and function handles and sets the function's
shared-memory attribute. Nothing is launched and no tensor is read or
written, so no state is left behind.

Also here: the boot launch of the Mamba-hybrid accepted-count kernels, which
no warm-up reached (``warm_num_accepted_kernels``), and the post-warm-up
kernel-load monitor (``install_kernel_load_monitor``).

This is general: ``preload_boot_kernels`` loads
every specialization any boot compile left in a JITFunction cache: the
functions in Triton's registry plus those a ``jit_post_compile_hook``
recorder saw (``install_compile_recorder``, from ``Worker.load_model``).
A specialization first compiled while serving (one no warm-up produced)
still compiles and loads in the forward that needs it; the recorder logs it
(once per specialization, every rank) and nothing is written to disk.
"""

from __future__ import annotations

import contextlib
import functools
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)

_load_monitor_installed = False
_load_monitor_warns = True
_loads_seen: set[tuple[str, Any]] = set()

_recorder_installed = False
# id -> JITFunction compiled since the recorder went in (kept alive here).
_recorded: dict[int, Any] = {}
# After warm-up: a compile is a serving compile, logged once per key.
_serving_record = False
_serving_compiles_seen: set[tuple[str, str]] = set()
# Quiesce-before-load hook state.
_quiesce_installed = False
_serving_started = False
_quiesced_seen: set[tuple[str, Any]] = set()
_quiesce_failures_seen: set[str] = set()


def _current_device() -> Any:
    """The key Triton files this process's compiled kernels under: its
    ``driver.active.get_current_device()`` (the current CUDA device index)."""
    try:
        from triton.runtime import driver

        return driver.active.get_current_device()
    except Exception:
        import torch

        return torch.cuda.current_device()


def _compiled_kernels(jit_fn: Any, device: Any) -> list[Any]:
    """Every specialization a Triton ``JITFunction`` has cached for
    ``device``. ``load_binary`` loads into whatever CUDA context is current,
    so a kernel compiled for another device is never touched here."""
    fn = jit_fn
    # Autotuner / Heuristics wrap the JITFunction in ``.fn``.
    while not hasattr(fn, "device_caches") and hasattr(fn, "fn"):
        fn = fn.fn
    caches = getattr(fn, "device_caches", None)
    if not caches:
        return []
    # ``.get``: ``device_caches`` is a defaultdict; indexing would add a key.
    entry = caches.get(device)
    # (kernel_cache, kernel_key_cache, target, backend, binder) in Triton
    # 3.4+ (3.7.1 jit.py:671-682); the kernel cache is the first element.
    kernel_cache = entry[0] if isinstance(entry, tuple) else None
    return list(kernel_cache.values()) if isinstance(kernel_cache, dict) else []


def preload_triton_kernels(
    *jit_fns: Any, label: str = "", expected: int | None = None
) -> int:
    """Load each specialization of ``jit_fns`` cached for the current device
    now; return how many are loaded (already-loaded ones count, a second call
    is a no-op).

    A specialization that fails to load is logged and left to load at its
    first launch, as it would without the preload; boot does not fail for
    it. A
    shortfall (any failure, or fewer loaded than ``expected``, the number the
    warm-up compiled) is a WARNING with the counts, so a serving log shows when
    the preload did not take effect.
    """
    loaded, failed = _preload(jit_fns, _current_device())
    if not label:
        return loaded
    if failed or (expected is not None and loaded < expected):
        logger.warning(
            "Triton preload short for %s: loaded=%d failed=%d "
            "expected=%s; the rest load at their first launch",
            label, loaded, failed, expected,
        )
    else:
        logger.info(
            "Loaded %d warmed %s kernel binaries at boot.", loaded, label
        )
    return loaded


def preload_qsa_sparse_kernels(warmed: Any = None) -> int:
    """End of ``warmup_qsa_sparse_paged_attention``: the split-K partial
    kernel and its merge. ``warmed`` is the warm-up's (block_n, splits, warps)
    list: one partial per entry plus one merge per distinct split count > 1."""
    from vllm.models.qwen4_exp.nvidia.ops import qsa

    expected = None
    if warmed is not None:
        expected = len(warmed) + len({s for _, s, _ in warmed if s > 1})
    return preload_triton_kernels(
        qsa._qsa_sparse_paged_gqa_splitk_kernel,
        qsa._qsa_merge_splitk_kernel,
        label="Qwen4Exp QSA sparse attention",
        expected=expected,
    )


def preload_qsa_indexer_decode_kernels(expected: int | None = None) -> int:
    """End of ``warmup_qsa_mqa_paged_decode``: the uniform decode scorer,
    one specialization per warm-up profile."""
    from vllm.models.qwen4_exp.nvidia.ops import qsa_indexer

    return preload_triton_kernels(
        qsa_indexer._qsa_mqa_paged_uniform_kernel,
        label="Qwen4Exp QSA indexer decode",
        expected=expected,
    )


def _preload(
    jit_fns: tuple[Any, ...],
    device: Any,
    *,
    new_names: list[str] | None = None,
    failed_names: list[str] | None = None,
) -> tuple[int, int]:
    """Load every cached specialization of ``jit_fns`` on ``device``.
    Returns (loaded incl. already resident, failed). ``new_names`` collects
    the kernels this call loaded; with ``failed_names`` given, failures are
    collected there and logged at DEBUG instead of one WARNING each."""
    loaded = failed = 0
    for jit_fn in jit_fns:
        for cached in _compiled_kernels(jit_fn, device):
            kernel = None
            try:
                # An async compile (triton.AsyncCompileMode) caches a
                # FutureKernel, which forwards attribute reads to result():
                # resolve it explicitly, inside the try, so a failed compile
                # is caught here and the reset below lands on the real kernel.
                kernel = (cached.result()
                          if callable(getattr(type(cached), "result", None))
                          else cached)
                init_handles = getattr(kernel, "_init_handles", None)
                if init_handles is None:
                    continue
                if getattr(kernel, "module", None) is not None:
                    # Already loaded: never re-enter _init_handles for a
                    # resident key (Triton keeps that module for good).
                    loaded += 1
                    continue
                init_handles()
                if new_names is not None:
                    new_names.append(str(getattr(kernel, "name", jit_fn)))
            except Exception:
                # Triton sets ``_run`` before ``load_binary`` (3.7.1
                # compiler.py:455/:468): when the load itself failed (module
                # still None) clear it, so the first launch rebuilds the
                # handles and retries. A failure after the module is set (the
                # thread-count check, a kernel_load_end_hook: :471-474) keeps
                # ``_run``: the launch then skips _init_handles and must still
                # have its launcher.
                if kernel is not None and getattr(kernel, "module", None) is None:
                    with contextlib.suppress(Exception):
                        kernel._run = None
                failed += 1
                if failed_names is not None:
                    failed_names.append(str(getattr(kernel, "name", jit_fn)))
                    logger.debug("Triton boot load of %s failed",
                                 failed_names[-1], exc_info=True)
                    continue
                logger.warning(
                    "Triton preload: %s did not load at boot; it loads at its "
                    "first launch instead",
                    getattr(kernel, "name", jit_fn),
                    exc_info=True,
                )
                continue
            loaded += 1
    return loaded, failed


def warm_num_accepted_kernels(runner: Any) -> bool:
    """Compile and load the Mamba-hybrid accepted-count kernels at boot.

    ``MambaHybridModelState.postprocess_state`` launches
    ``_fill_num_accepted_kernel`` on a non-last PP rank after every non-final
    prefill chunk (model_runner.py ``sample_tokens``: ``postprocess_state(
    idx_mapping, 0)``), and ``_scatter_num_accepted_kernel`` with a sampled
    count tensor. The boot warm-up prefills one-chunk prompts, so neither ran
    before serving, and the first real one could compile the fill kernel on
    PP0 inside the same window as the QSA load. A
    real launch with a ``-1`` request index returns before its store, so it
    changes nothing and leaves both kernels compiled and loaded.

    The argument types are the production ones, because Triton keys each
    specialization on them: ``idx_mapping`` is int64 (model_runner.py builds
    it with ``dtype=torch.int64``; the PP receive filter's ``async_tensor_h2d``
    of an intp array is int64 too), ``num_accepted_tokens_gpu`` is the state's
    own int32 buffer, the fill value is ``max(num_sampled, 1)`` = 1 (Triton
    specializes the int 1 on its own), and the scatter's sampled counts are
    int32 (sampler / rejection sampler).
    """
    state = getattr(runner, "model_state", None)
    accepted = getattr(state, "num_accepted_tokens_gpu", None)
    if accepted is None:
        return False
    import torch

    from vllm.v1.worker.gpu.model_states import mamba_hybrid

    skip = torch.full((1,), -1, dtype=torch.int64, device=accepted.device)
    sampled = torch.ones(1, dtype=torch.int32, device=accepted.device)
    mamba_hybrid._fill_num_accepted_kernel[(1,)](skip, accepted, 1)
    mamba_hybrid._scatter_num_accepted_kernel[(1,)](skip, sampled, accepted)
    torch.cuda.synchronize(accepted.device)
    logger.info(
        "Warmed Mamba-hybrid fill/scatter num_accepted kernels at boot."
    )
    return True


# --- Every boot compile loaded at boot -------------------------------------


def _jit_function_registry() -> dict[str, Any]:
    """Triton's own map of every JITFunction constructed, keyed
    ``module:qualname`` (3.7.1 jit.py ``JITFunction.__init__``)."""
    try:
        from triton.runtime import jit as triton_jit
    except Exception:
        return {}
    registry = getattr(triton_jit, "_triton_jit_function_registry", None)
    return registry if isinstance(registry, dict) else {}


def _guard_specialization_serializer(jit_module: Any = None) -> None:
    """With any post-compile hook set, Triton serializes each compile's
    specialization (3.7.1 jit.py:636) and a non-JSON constexpr raises out of
    the compile. Same guard as ``vllm.utils.jit_monitor`` (whose marker it
    sets, so neither wraps twice), needed earlier: the recorder runs during
    warm-up, before the JIT monitor."""
    if jit_module is None:
        try:
            from triton.runtime import jit as jit_module
        except Exception:
            return
    original = getattr(jit_module, "serialize_specialization_data", None)
    if original is None or getattr(original, "_vllm_guarded", False):
        return

    @functools.wraps(original)
    def _guarded(*args: Any, **kwargs: Any) -> Any:
        try:
            return original(*args, **kwargs)
        except (TypeError, ValueError):
            return None

    _guarded._vllm_guarded = True  # type: ignore[attr-defined]
    jit_module.serialize_specialization_data = _guarded


def _log_serving_compile(kwargs: dict) -> None:
    name = str(getattr(kwargs.get("fn"), "name", "<unknown>"))
    ident = (name, str(kwargs.get("key")))
    if ident in _serving_compiles_seen:
        return
    _serving_compiles_seen.add(ident)
    logger.warning(
        "Triton kernel first compiled while serving: %s (no warm-up "
        "produced this specialization; it loads in this forward)", name,
    )


def _on_compile(**kwargs: Any) -> None:
    # Never let the recorder break a compile.
    with contextlib.suppress(Exception):
        jit_fn = getattr(kwargs.get("fn"), "jit_function", None)
        if jit_fn is None:
            return
        _recorded.setdefault(id(jit_fn), jit_fn)
        if _serving_record:
            _log_serving_compile(kwargs)


def install_compile_recorder(knobs: Any = None, *, jit_module: Any = None) -> bool:
    """Record every JITFunction compiled from now on (call before the model
    loads): ``preload_boot_kernels`` loads what they cached, and once armed
    (``arm_serving_recorder``) a compile is logged as a serving compile.
    Chains any existing ``knobs.runtime.jit_post_compile_hook``; the JIT
    monitor installed later chains this one. Idempotent. ``jit_module``
    (default ``triton.runtime.jit``) is where the serializer is guarded."""
    global _recorder_installed
    if _recorder_installed:
        return True
    if knobs is None:
        try:
            from triton import knobs
        except Exception:
            return False
    # Before the hook goes in: from then on every compile serializes.
    _guard_specialization_serializer(jit_module)
    runtime = getattr(knobs, "runtime", None)
    if runtime is None or not hasattr(runtime, "jit_post_compile_hook"):
        return False
    hook = runtime.jit_post_compile_hook
    if callable(getattr(hook, "add", None)):
        hook.add(_on_compile)
    elif hook is None:
        runtime.jit_post_compile_hook = _on_compile
    else:
        previous = hook

        def _chained(**kwargs: Any) -> Any:
            _on_compile(**kwargs)
            return previous(**kwargs)

        runtime.jit_post_compile_hook = _chained
    _recorder_installed = True
    return True


def arm_serving_recorder(enabled: bool) -> None:
    """From here on a compile is a serving compile: log it (``enabled``),
    or only record it (JIT warm-up off)."""
    global _serving_record
    _serving_record = bool(enabled)


def preload_boot_kernels() -> int:
    """Load every specialization a boot compile left cached for this device:
    Triton's registry plus the recorder's functions (a factory closure's
    function is shadowed in the registry). Call after the last warm-up
    launch, before the monitors. Returns how many it newly loaded; one that
    cannot load (e.g. a too-large autotune config) stays for its first
    launch, which fails or loads as it did before."""
    functions: dict[int, Any] = {}
    for jit_fn in list(_jit_function_registry().values()):
        functions.setdefault(id(jit_fn), jit_fn)
    for jit_fn in list(_recorded.values()):
        functions.setdefault(id(jit_fn), jit_fn)
    new_names: list[str] = []
    failed_names: list[str] = []
    free_before = _device_free_bytes()
    loaded, failed = _preload(tuple(functions.values()), _current_device(),
                              new_names=new_names, failed_names=failed_names)
    free_after = _device_free_bytes()
    cost = ("n/a" if free_before is None or free_after is None
            else f"{(free_before - free_after) / 2**20:.1f} MiB")
    args = (len(new_names), loaded, cost,
            ", ".join(sorted(set(new_names))) or "-")
    if failed:
        logger.warning(
            "Triton boot load: loaded=%d new of %d resident, device "
            "memory %s (%s); failed=%d (%s), those load at their first launch",
            *args, failed, ", ".join(sorted(set(failed_names))),
        )
    else:
        logger.info(
            "Triton boot load: loaded=%d new of %d resident, device "
            "memory %s (%s)", *args,
        )
    return len(new_names)


def _device_free_bytes() -> int | None:
    """Free device memory, so the boot log shows what the loads cost."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return int(torch.cuda.mem_get_info()[0])
    except Exception:
        return None


def _hook_kernel_name(args: tuple, kwargs: dict) -> str:
    name = kwargs.get("name")
    if isinstance(name, str):
        return name
    # Triton calls the hook positionally: (module, function, name,
    # metadata_group, hash).
    if len(args) > 2 and isinstance(args[2], str):
        return args[2]
    return next((a for a in args if isinstance(a, str)), "<unknown>")


def _hook_kernel_hash(args: tuple, kwargs: dict) -> Any:
    if "hash" in kwargs:
        return kwargs["hash"]
    return args[4] if len(args) > 4 else None


def _on_kernel_load(*args: Any, **kwargs: Any) -> None:
    # Never let logging break a kernel load.
    with contextlib.suppress(Exception):
        name = _hook_kernel_name(args, kwargs)
        key = (name, _hook_kernel_hash(args, kwargs))
        # Once per kernel binary: a load that keeps failing re-fires the hook
        # on every launch attempt.
        if key in _loads_seen:
            return
        _loads_seen.add(key)
        if _load_monitor_warns:
            logger.warning(
                "Triton kernel load after warm-up: %s (a module load in the "
                "middle of a forward can hang a pipeline-parallel step; warm "
                "it)", name,
            )
        else:
            logger.info(
                "Triton kernel load after warm-up: %s (JIT warm-up is off, so "
                "first launches load lazily)", name,
            )


def install_kernel_load_monitor(knobs: Any = None, *, warn: bool = True) -> bool:
    """Log every Triton kernel binary load from now on (call once warm-up and
    graph capture are done), once per (name, hash). Uses
    ``knobs.runtime.kernel_load_start_hook`` (Triton 3.7.1: a HookChain fired
    in ``CompiledKernel._init_handles`` before ``load_binary``). ``warn``
    False (JIT warm-up disabled, so lazy first loads are expected) logs at
    INFO; the monitor stays installed either way. Returns False when this
    Triton has no such hook. Idempotent: a second call adds no second hook."""
    global _load_monitor_installed, _load_monitor_warns, _serving_started
    _load_monitor_warns = warn
    # Warm-up and capture are over: from here a load is a serving load,
    # which the quiesce hook drains the stream for.
    _serving_started = True
    # Compiles from here on are serving compiles: log them, unless
    # JIT warm-up is off.
    install_compile_recorder(knobs)
    arm_serving_recorder(warn)
    if _load_monitor_installed:
        return True
    if knobs is None:
        try:
            from triton import knobs
        except Exception:
            return False
    runtime = getattr(knobs, "runtime", None)
    if runtime is None or not hasattr(runtime, "kernel_load_start_hook"):
        logger.warning("Triton has no kernel_load_start_hook; the kernel-load "
                       "monitor is not installed")
        return False
    _chain_load_start_hook(runtime, _on_kernel_load)
    _load_monitor_installed = True
    logger.info("Triton kernel-load monitor armed: any later binary "
                "load logs 'Triton kernel load after warm-up'.")
    return True


def _chain_load_start_hook(runtime: Any, fn: Any) -> None:
    """Add ``fn`` to ``runtime.kernel_load_start_hook``: a 3.7.1 HookChain
    (``add``), else become the hook or wrap the existing one, so the monitor
    and the quiesce hook both run whichever is installed first."""
    hook = runtime.kernel_load_start_hook
    if callable(getattr(hook, "add", None)):
        hook.add(fn)
    elif hook is None:
        runtime.kernel_load_start_hook = fn
    else:
        previous = hook

        def _chained(*args: Any, **kwargs: Any) -> Any:
            fn(*args, **kwargs)
            return previous(*args, **kwargs)

        runtime.kernel_load_start_hook = _chained


# --- Quiesce the compute stream before a serving-time load ----------------


def _torch_cuda() -> Any:
    import torch

    return torch.cuda


def _gpu_sync_allowed() -> Any:
    try:
        from vllm.utils.gpu_sync_debug import gpu_sync_allowed
    except Exception:
        return contextlib.nullcontext()
    return gpu_sync_allowed()


def _quiesce_before_load(*args: Any, **kwargs: Any) -> None:
    """``kernel_load_start_hook``: runs on the loading thread just before
    ``load_binary`` (3.7.1 compiler.py:465-468). Once serving has started,
    drain this thread's current stream first, so no PLE host-flag wait (or
    other host-dependent work) is pending on it while the module loads (a
    wait cycle). ``Stream.synchronize`` releases the GIL
    (torch cuda/Stream.cpp:134-142), so the PLE connector can publish the
    flag meanwhile. Only the current stream: a device-wide sync would also
    wait on the PP broadcast stream, which depends on PP1. Never while the
    stream is capturing (a sync there invalidates the capture). Never
    raises: a failed quiesce leaves the load as it was before."""
    if not _serving_started:
        return
    name = "<unknown>"
    with contextlib.suppress(Exception):
        name = _hook_kernel_name(args, kwargs)
    try:
        cuda = _torch_cuda()
        if cuda.is_current_stream_capturing():
            return
        # A deliberate sync: inside VLLM_GPU_SYNC_CHECK's allowance, which
        # the worker turns on after warm-up.
        with _gpu_sync_allowed():
            cuda.current_stream().synchronize()
    except Exception as e:
        cls = type(e).__name__
        if cls not in _quiesce_failures_seen:
            _quiesce_failures_seen.add(cls)
            logger.warning(
                "Triton quiesce before loading %s failed: %s; loading without "
                "a drain", name, cls, exc_info=True,
            )
        return
    with contextlib.suppress(Exception):
        key = (name, _hook_kernel_hash(args, kwargs))
        if key not in _quiesced_seen:
            _quiesced_seen.add(key)
            logger.info(
                "Quiesced the current stream before loading Triton kernel %s",
                name,
            )


def install_quiesce_before_load(knobs: Any = None) -> bool:
    """Install ``_quiesce_before_load`` on ``kernel_load_start_hook``
    (chained with the load monitor in either order). It acts only once
    ``install_kernel_load_monitor`` has started serving. Idempotent."""
    global _quiesce_installed
    if _quiesce_installed:
        return True
    if knobs is None:
        try:
            from triton import knobs
        except Exception:
            return False
    runtime = getattr(knobs, "runtime", None)
    if runtime is None or not hasattr(runtime, "kernel_load_start_hook"):
        return False
    _chain_load_start_hook(runtime, _quiesce_before_load)
    _quiesce_installed = True
    return True
