# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 lebeche hill-climb (fork of Merlin 3.0.2.2)
"""sm_80/86/89 port of the low-latency CuTe-DSL skinny GEMM (lebeche fork).

Stock 3.0.2.2 only runs the skinny BF16 GEMM plans on sm_90+. Nothing in the
FMA kernel needs Hopper except ``cute.arch.elect_one()`` (``elect.sync``); PDL
is already conditional. With the switch on, on a capability listed below:

* the skinny GEMM kernel is built with a lane-0 test instead of ``elect_one``;
* ``QWEN4_EXP_SM86_GEMM_PLANS`` (low_latency_gemm.py) is registered, so
  ``enable_qwen4_exp_low_latency_gemm`` patches the matching unquantized
  Linear modules. Dispatch on the token count happens inside the opaque
  ``qwen4_exp_low_latency_gemm`` custom op, i.e. at run time.

The fused HC down+SiLU kernel (ops/cute_dsl/hc_down_silu.py) stays Hopper+: its
``1 <= M <= MAX_FUSED_M`` test is a Python branch in the traced forward, which
torch.compile resolves once at trace time (guards are dropped), so it cannot
select M <= 4 at run time. The merged HC down+inject projection is served by a
skinny plan instead, followed by the stock ``hc_silu`` kernel. With
``MERLIN_LEBECHE_HC_FUSE`` (default on) the SiLU and the gate mix run as
epilogues of the skinny down/up GEMMs, dispatched at run time inside custom ops
(vllm/models/qwen4_exp/nvidia/hc_small_fused.py).

``MERLIN_LEBECHE_SM86_LL`` (default "1"; "0" = exactly stock 3.0.2.2) is read
once at import. It is a hashed compile factor. Other capabilities are never
affected. The Flash-Next launcher forwards it (tools/merlin/launch.py).
"""

import torch

from vllm.merlin.envs import read_registered

SM86_LL_CAPABILITIES: frozenset[tuple[int, int]] = frozenset({(8, 0), (8, 6), (8, 9)})
SM86_LL_SWITCH: bool = bool(read_registered("MERLIN_LEBECHE_SM86_LL"))
SM86_LL_EXTRA_SWITCH: bool = bool(read_registered("MERLIN_LEBECHE_SM86_LL_EXTRA"))
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
