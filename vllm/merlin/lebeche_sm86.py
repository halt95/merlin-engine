# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 lebeche hill-climb (fork of Merlin 3.0.2.2)
"""sm_80/86/89 port of the low-latency CuTe-DSL decode kernels (lebeche fork).

Stock 3.0.2.2 only runs the skinny BF16 GEMM plans and the fused
hyper-connection down+SiLU kernel on sm_90+. Nothing in their FMA kernels
needs Hopper except ``cute.arch.elect_one()`` (``elect.sync``) and PDL
(``griddepcontrol``). With the switch on, on a capability listed below:

* the skinny GEMM / fused FMA kernels are built with a lane-0 test instead of
  ``elect_one`` and without PDL;
* ``QWEN4_EXP_SM86_GEMM_PLANS`` (low_latency_gemm.py) is registered;
* ``GatedResidual`` uses the fused down+SiLU kernel for M <= 4 (FMA backend
  only; the clustered split-K MMA backend stays Hopper+).

``MERLIN_LEBECHE_SM86_LL`` (default "1"; "0" = exactly stock 3.0.2.2) is read
once at import. It is a hashed compile factor (it changes the captured
kernels). Other capabilities are never affected.
"""

import torch

from vllm.merlin.envs import read_registered

SM86_LL_CAPABILITIES: frozenset[tuple[int, int]] = frozenset({(8, 0), (8, 6), (8, 9)})
SM86_LL_SWITCH: bool = bool(read_registered("MERLIN_LEBECHE_SM86_LL"))
# Largest token count the fused HC down+SiLU FMA kernel beats the stock GEMM at
# on an RTX 3090 (measured: wins at M <= 4, loses from M = 8).
SM86_FUSED_HC_MAX_M = 4

_active: bool | None = None


def sm86_ll_active() -> bool:
    """Switch on and the current CUDA device is sm_80/86/89."""
    global _active
    active = _active
    if active is None:
        active = False
        if SM86_LL_SWITCH:
            from vllm.platforms import current_platform

            if current_platform.is_cuda():
                cap = current_platform.get_device_capability()
                if cap is None:
                    return False
                active = (cap.major, cap.minor) in SM86_LL_CAPABILITIES
        # Plain module bool (same reason as CudaPlatformBase._arch_support_pdl):
        # no write while dynamo traces.
        if not torch.compiler.is_compiling():
            _active = active
    return active
