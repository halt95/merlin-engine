# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Family gates for the Qwen3.5 sm_86 patches.

The Qwen3.8-27B series patches (uniform multi-token FlashInfer decode, the decode-plan
cache, the MTP draft seq_lens no-sync and the GDN in_proj_ba split-K) were validated
only for the dense Qwen3.5 architectures on Ampere sm_86. On the shared engine every
other family (Flash-Next = Qwen4Exp included) and every other device must take the
stock path, so each patch asks series_active() and fails closed. The FULL-graph MTP
verify sites ask uniform_verify_active(), which also admits Qwen3.5 MoE when
MERLIN_MOE_UNIFORM_VERIFY=1.

The architecture names come from the one Qwen3.5-lineage table
(vllm/qwen35_family.py): today's Qwen3_5* names, the default Qwen 4 candidates
and MERLIN_QWEN35_ARCH_ALIASES, resolved once at import.
"""

from vllm import qwen35_family

DENSE_QWEN35_ARCHS = qwen35_family.dense_archs()


def is_dense_qwen35(architecture: str | None) -> bool:
    return architecture in DENSE_QWEN35_ARCHS


# Qwen3.5 MoE (Ornith-1.5-122B) may take the
# FULL-graph MTP verify path (and only that path; see uniform_verify_active).
QWEN35_MOE_ARCHS = qwen35_family.moe_archs()


_UNRESOLVED_WARNED = False
_UV_LOGGED = False
_CAP_QUERY_WARNED = False
# series_active() calls that failed closed because the capability query raised.
CAP_QUERY_FAILURES = 0


def execution_device_index(device=None) -> int | None:
    """CUDA ordinal of the device the caller executes on, or None when it is unknown.

    ``device`` is the caller's own execution device (``torch.device``, ``"cuda:N"``
    or an int ordinal) for callers that own one, e.g. an attention metadata builder.
    Without it the process's current device counts only once CUDA is initialised:
    before that, ``current_device()`` reports the default ordinal 0, which need not be
    the card this worker will run on. Never substitutes device 0 for an unknown device.
    """
    try:
        import torch

        if device is not None:
            if isinstance(device, bool):
                return None
            if isinstance(device, int):
                return device if device >= 0 else None
            dev = torch.device(device)
            if dev.type != "cuda":
                return None
            if dev.index is not None:
                return dev.index
            # "cuda" without an index means the current device: resolve it below.
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            return torch.cuda.current_device()
    except Exception:
        return None
    return None


def series_active(vllm_config, device=None) -> bool:
    """True only for a dense Qwen3.5 target executing on an sm_86 device.

    Fails closed (False) when the executing device cannot be resolved; pass ``device``
    from callers that own one.
    """
    model_config = getattr(vllm_config, "model_config", None)
    if model_config is None or not is_dense_qwen35(model_config.architecture):
        return False
    return _executes_on_sm86(device)


def uniform_verify_active(vllm_config, device=None) -> bool:
    """Gate of the FULL-graph MTP verify path (uniform multi-token FlashInfer decode).

    The 27B series as before, plus Qwen3.5 MoE on sm_86 when MERLIN_MOE_UNIFORM_VERIFY=1
    (opt-in; off by default). Only the spec-as-decode / uniform-batch sites ask this;
    every other series patch stays dense-only via series_active().
    """
    if series_active(vllm_config, device=device):
        return True
    model_config = getattr(vllm_config, "model_config", None)
    if model_config is None or model_config.architecture not in QWEN35_MOE_ARCHS:
        return False
    import vllm.envs as envs

    if not envs.MERLIN_MOE_UNIFORM_VERIFY:
        return False
    active = _executes_on_sm86(device)
    global _UV_LOGGED
    if device is not None and not _UV_LOGGED:
        _UV_LOGGED = True
        from vllm.logger import init_logger

        init_logger(__name__).info(
            "MERLIN_MOE_UNIFORM_VERIFY=1: FULL-graph MTP verify %s",
            "active"
            if active
            else "OFF (sm_86 only; device is not sm_86 or unresolved)",
        )
    return active


def _executes_on_sm86(device=None) -> bool:
    """True when the executing device resolves to sm_86; fails closed otherwise."""
    global _UNRESOLVED_WARNED, _CAP_QUERY_WARNED, CAP_QUERY_FAILURES
    from vllm.platforms import current_platform

    device_id = execution_device_index(device)
    if device_id is None:
        if not _UNRESOLVED_WARNED:
            _UNRESOLVED_WARNED = True
            from vllm.logger import init_logger

            init_logger(__name__).warning(
                "Qwen3.5 sm_86 gate: executing device unresolved (device=%r, CUDA "
                "not initialised or no CUDA device); the gated patches stay OFF for "
                "this caller",
                device,
            )
        return False
    try:
        cap = current_platform.get_device_capability(device_id=device_id)
    except Exception as exc:  # noqa: BLE001
        CAP_QUERY_FAILURES += 1
        if not _CAP_QUERY_WARNED:
            _CAP_QUERY_WARNED = True
            from vllm.logger import init_logger

            init_logger(__name__).warning(
                "Qwen3.5 sm_86 gate: device capability query failed for device %r "
                "(%s: %s); the gated patches stay OFF (failing closed). Counted in "
                "qwen35_series.CAP_QUERY_FAILURES; this warning is logged once.",
                device_id,
                type(exc).__name__,
                exc,
            )
        return False
    return cap is not None and cap.to_int() == 86
