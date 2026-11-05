# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Per-rank launch-identity log line (log-only; changes no behaviour).

After model loading -- and after the PLE offload placeholders are armed -- every
GPU worker rank logs one line per model role it holds::

    e2-identity[PP<p>_TP<t>]: ple_backend=<class>@<transport> ple_method=<class>
    ple_weight_dtype=<dtype> breakable=<ON|OFF> compile_mode=<mode>
    cudagraph_mode=<mode> role=<target|draft>

``role=target`` is logged after model loading; ``role=draft`` only where a draft
model is loaded (the last PP stage), once the drafter's graph modes are resolved
(after the KV cache is initialised, from ``Worker.initialize_from_config``).
Every value is read from the live objects:

* ``ple_backend``: the class of each PLE layer's ``ple_embedding`` module, and
  the transport that serves its rows on this rank: ``host-mapped-pull``
  (offloaded with a host output pointer), ``ipc-push`` (offloaded, no host
  pointer) or ``local:<device>`` (the rank owns the table).
* ``ple_method``: the embedding's live ``quant_method``, or, on an offloaded
  placeholder, the ``_offload_quant_method`` the rank dequantizes with.
* ``ple_weight_dtype``: the table's dtype where the rank owns it; on an
  offloaded placeholder, the dtype of the armed transfer buffer
  (``get_offload_output_dtype``).
* ``breakable``: ``is_breakable_cudagraph_enabled()``, the switch the capture
  path itself reads.
* ``compile_mode`` / ``cudagraph_mode``, ``role=target``: the engine's
  ``CompilationConfig`` (config level, before KV-cache-time capture-mode
  resolution).
* ``compile_mode`` / ``cudagraph_mode``, ``role=draft``: the DRAFTER's own
  resolved state, because the engine has one ``CompilationConfig`` and its
  fields name the target's profile. ``compile_mode`` is the mode of the draft
  model's live ``@support_torch_compile`` wrapper(s) (``NONE`` when none is
  live); ``cudagraph_mode`` is the mode each speculator cudagraph manager holds
  after ``init_cudagraph_manager`` (``prefill:<m>,decode:<m>`` for the
  autoregressive / MTP speculator, the bare mode for a single-manager
  speculator; a V1 drafter reports its dispatcher's mode), or ``unresolved`` if
  a manager does not exist yet.

A role with no PLE layer reports ``none`` for the three PLE fields; several
distinct values on one rank are joined with ``,``.
"""

from typing import Any

from torch import nn

from vllm.logger import init_logger

logger = init_logger(__name__)

_NONE = "none"


def _join(values: set[str]) -> str:
    return ",".join(sorted(values)) if values else _NONE


def _enum_name(value: Any) -> str:
    if value is None:
        return _NONE
    return str(getattr(value, "name", value))


def _ple_embeddings(model: nn.Module | None) -> list[nn.Module]:
    """The ``ple_embedding`` module of every PLE layer in ``model``."""
    if model is None:
        return []
    found = []
    for module in model.modules():
        embedding = module.__dict__.get("_modules", {}).get("ple_embedding")
        if isinstance(embedding, nn.Module):
            found.append(embedding)
    return found


def _transport(embedding: nn.Module) -> str:
    if getattr(embedding, "_is_cpu_offloaded", False):
        if getattr(embedding, "_host_output_ptr", None) is not None:
            return "host-mapped-pull"
        return "ipc-push"
    weight = getattr(getattr(embedding, "ngram_embedding", None), "weight", None)
    if weight is not None:
        return f"local:{weight.device.type}"
    return "local:unknown"


def _method(embedding: nn.Module) -> str:
    table = getattr(embedding, "ngram_embedding", None)
    method = getattr(table, "quant_method", None)
    if method is None:
        method = getattr(embedding, "_offload_quant_method", None)
    return type(method).__name__ if method is not None else _NONE


def _weight_dtype(embedding: nn.Module) -> str:
    weight = getattr(getattr(embedding, "ngram_embedding", None), "weight", None)
    if weight is not None:
        return str(weight.dtype)
    if getattr(embedding, "_is_cpu_offloaded", False):
        buffer = getattr(embedding, "_gpu_output_buffer", None)
        if buffer is not None:
            return str(buffer.dtype)
    return "unknown"


def _rank_tag() -> str:
    from vllm.distributed.parallel_state import get_pp_group, get_tp_group

    return f"PP{get_pp_group().rank_in_group}_TP{get_tp_group().rank_in_group}"


def _draft(model_runner: Any) -> tuple[nn.Module | None, Any]:
    """The loaded draft model and its holder (V2 speculator or V1 drafter)."""
    for name in ("speculator", "drafter"):
        holder = getattr(model_runner, name, None)
        model = getattr(holder, "model", None)
        if isinstance(model, nn.Module):
            return model, holder
    return None, None


#: Written where a draft graph mode has no resolved source yet.
UNRESOLVED = "unresolved"
#: Speculator cudagraph-manager attributes (the set ``e2-guard[runtime]``
#: reads), with the phase label each carries on the draft line.
_DRAFT_MANAGER_PHASES = (
    ("prefill_cudagraph_manager", "prefill"),
    ("decode_cudagraph_manager", "decode"),
    ("cudagraph_manager", None),
)


def _draft_compile_mode(model: nn.Module) -> str:
    """Compile mode of the draft model's live ``@support_torch_compile`` wrappers.

    ``NONE`` when no wrapper is live: the drafter then runs eager, whatever
    the shared ``CompilationConfig.mode`` says.
    """
    from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper

    modes = {
        _enum_name(getattr(getattr(m, "compilation_config", None), "mode", None))
        for m in model.modules()
        if isinstance(m, TorchCompileWithNoGuardsWrapper)
        and not getattr(m, "do_not_compile", True)
    }
    return _join(modes) if modes else "NONE"


def _draft_cudagraph_mode(holder: Any) -> str:
    """The drafter's resolved cudagraph mode(s), read from its managers."""
    parts = []
    for attr, phase in _DRAFT_MANAGER_PHASES:
        if not hasattr(holder, attr):
            continue
        manager = getattr(holder, attr)
        mode = (
            UNRESOLVED
            if manager is None
            else _enum_name(getattr(manager, "cudagraph_mode", None))
        )
        parts.append(mode if phase is None else f"{phase}:{mode}")
    if parts:
        return ",".join(parts)
    # V1 drafter: one dispatcher, keyed in _check_and_update_cudagraph_mode.
    dispatcher = getattr(holder, "cudagraph_dispatcher", None)
    if dispatcher is not None and getattr(dispatcher, "keys_initialized", False):
        return _enum_name(getattr(dispatcher, "cudagraph_mode", None))
    return UNRESOLVED


def identity_line(
    tag: str,
    model: nn.Module | None,
    vllm_config: Any,
    role: str,
    *,
    compile_mode: str | None = None,
    cudagraph_mode: str | None = None,
) -> str:
    """One identity line. ``compile_mode`` / ``cudagraph_mode`` override the
    config-level values (the draft role passes its resolved state)."""
    from vllm.compilation.breakable_cudagraph import is_breakable_cudagraph_enabled

    embeddings = _ple_embeddings(model)
    backends = {f"{type(e).__name__}@{_transport(e)}" for e in embeddings}
    compilation = getattr(vllm_config, "compilation_config", None)
    if compile_mode is None:
        compile_mode = _enum_name(getattr(compilation, "mode", None))
    if cudagraph_mode is None:
        cudagraph_mode = _enum_name(getattr(compilation, "cudagraph_mode", None))
    return (
        f"e2-identity[{tag}]: "
        f"ple_backend={_join(backends)} "
        f"ple_method={_join({_method(e) for e in embeddings})} "
        f"ple_weight_dtype={_join({_weight_dtype(e) for e in embeddings})} "
        f"breakable={'ON' if is_breakable_cudagraph_enabled() else 'OFF'} "
        f"compile_mode={compile_mode} "
        f"cudagraph_mode={cudagraph_mode} "
        f"role={role}"
    )


def draft_identity_line(tag: str, draft_model: nn.Module, holder: Any) -> str:
    """The ``role=draft`` line: PLE fields from the draft modules, graph fields
    from the drafter's resolved state (never the shared CompilationConfig)."""
    return identity_line(
        tag,
        draft_model,
        getattr(holder, "vllm_config", None),
        "draft",
        compile_mode=_draft_compile_mode(draft_model),
        cudagraph_mode=_draft_cudagraph_mode(holder),
    )


def log_launch_identity(model_runner: Any, vllm_config: Any) -> None:
    """Log this rank's ``role=target`` line after model load. Never raises.

    The ``role=draft`` line comes later, from :func:`log_draft_identity`, once
    the drafter's graph modes exist.
    """
    tag = "PP?_TP?"
    try:
        tag = _rank_tag()
        logger.info(
            "%s", identity_line(tag, model_runner.get_model(), vllm_config, "target")
        )
    except Exception as error:  # log-only: identity must never break a boot
        logger.warning(
            "e2-identity[%s]: unavailable (%s: %s)", tag, type(error).__name__, error
        )


def log_draft_identity(model_runner: Any) -> None:
    """Log this rank's ``role=draft`` line, if it holds a drafter. Never raises.

    Call after ``initialize_kv_cache``: that is where the V2 runner resolves
    the target's mode and ``init_cudagraph_manager`` derives the drafter's
    (and where the V1 runner keys the drafter's dispatcher).
    """
    tag = "PP?_TP?"
    try:
        draft_model, holder = _draft(model_runner)
        if draft_model is None:
            return
        tag = _rank_tag()
        logger.info("%s", draft_identity_line(tag, draft_model, holder))
    except Exception as error:  # log-only: identity must never break a boot
        logger.warning(
            "e2-identity[%s]: unavailable (%s: %s)", tag, type(error).__name__, error
        )
