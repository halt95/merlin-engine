# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen4Exp decode GEMM selection on Hopper and Blackwell.

Dispatch follows Kimi-K3 and uses the local ``(N, K)`` shape and token count.
Plans contain measured CUDA graph capture sizes; other token counts use the
standard linear implementation.
"""

import torch
from torch import nn

import vllm.envs as envs
from vllm.model_executor.kernels.linear.cute_dsl.skinny_gemm import (
    SkinnyGemmConfig,
    row_stride_ok,
    shape_dynamic_skinny_gemm,
)
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    UnquantizedEmbeddingMethod,
)
from vllm.merlin.lebeche_sm86 import (
    SM86_LL_CAPABILITIES,
    SM86_LL_EXTRA_SWITCH,
    SM86_LL_SWITCH,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

QWEN4_EXP_SM103_GEMM_PLANS: dict[tuple[int, int], dict[int, SkinnyGemmConfig]] = {
    # GDN fused QKVZ projection, TP=4.
    (4096, 2560): {
        1: SkinnyGemmConfig(1, 64, 4, k_unroll=4),
        2: SkinnyGemmConfig(2, 64, 4, k_unroll=4),
    },
    # GDN and QSA output projections, TP=4.
    (2560, 1536): {
        1: SkinnyGemmConfig(1, 128, 2, k_unroll=2, vector_width=4),
        2: SkinnyGemmConfig(2, 128, 2, k_unroll=2, vector_width=4),
        4: SkinnyGemmConfig(4, 64, 2, k_unroll=2),
    },
    # GDN fused B/A projection, TP=4.
    (24, 2560): {
        1: SkinnyGemmConfig(1, 128, 2, k_unroll=4, vector_width=4),
        2: SkinnyGemmConfig(2, 128, 2, k_unroll=4, vector_width=4),
        4: SkinnyGemmConfig(4, 128, 2, k_unroll=4, vector_width=4),
        8: SkinnyGemmConfig(8, 128, 1, k_unroll=4, vector_width=4),
        16: SkinnyGemmConfig(16, 128, 1, k_unroll=4, vector_width=4),
    },
    # QSA fused QKV/gate + replicated indexer Q/K, TP=4 (GB300).
    (4224, 2560): {
        1: SkinnyGemmConfig(1, 128, 4, vector_width=4, static_k=2560),
        2: SkinnyGemmConfig(2, 128, 2, vector_width=4, static_k=2560),
        4: SkinnyGemmConfig(4, 64, 4, vector_width=4, static_k=2560),
    },
    # Shared-expert fused gate/up projection, TP=4.
    (320, 2560): {
        1: SkinnyGemmConfig(1, 128, 2, k_unroll=4, vector_width=4),
        2: SkinnyGemmConfig(2, 128, 2, k_unroll=4, vector_width=4),
        4: SkinnyGemmConfig(4, 128, 2, k_unroll=4, vector_width=4),
        8: SkinnyGemmConfig(8, 64, 1, k_unroll=4),
        16: SkinnyGemmConfig(16, 128, 2, k_unroll=4, vector_width=4),
    },
    # LM head, TP=4.
    (62080, 2560): {
        1: SkinnyGemmConfig(1, 64, 4, k_unroll=2),
        2: SkinnyGemmConfig(2, 32, 4, k_unroll=2),
    },
}

# H200 plans selected by exhaustive CUDA graph replay measurements over
# M={1, 2, 4, 8, 16}. Only points that beat the standard linear implementation
# in both hot-cache and L2-flush measurements are retained; other token counts
# keep the standard implementation and its GEMM heuristics.
QWEN4_EXP_SM90_GEMM_PLANS: dict[tuple[int, int], dict[int, SkinnyGemmConfig]] = {
    # GDN fused QKVZ projection, TP=4.
    (4096, 2560): {
        1: SkinnyGemmConfig(1, 128, 2, vector_width=4, static_k=2560),
        2: SkinnyGemmConfig(2, 64, 4, vector_width=4, static_k=2560),
    },
    # QSA fused QKV/gate + replicated indexer Q/K, TP=4.
    (4224, 2560): {
        1: SkinnyGemmConfig(1, 128, 2, k_unroll=6, vector_width=2, static_k=2560),
        2: SkinnyGemmConfig(2, 128, 2, k_unroll=3, vector_width=2, static_k=2560),
        4: SkinnyGemmConfig(4, 64, 4, k_unroll=1, vector_width=8),
    },
    # GDN and QSA output projections, TP=4.
    (2560, 1536): {
        1: SkinnyGemmConfig(1, 128, 4, vector_width=2, static_k=1536),
        2: SkinnyGemmConfig(2, 128, 4, vector_width=4, static_k=1536),
        4: SkinnyGemmConfig(4, 64, 4, k_unroll=6, vector_width=4),
    },
    # GDN fused B/A projection, TP=4.
    (24, 2560): {
        1: SkinnyGemmConfig(1, 128, 3, vector_width=4, static_k=2560),
        2: SkinnyGemmConfig(2, 64, 2, vector_width=4, static_k=2560),
        4: SkinnyGemmConfig(4, 64, 1, static_k=2560),
        8: SkinnyGemmConfig(8, 128, 1, vector_width=4, static_k=2560),
        16: SkinnyGemmConfig(16, 128, 1, vector_width=4, static_k=2560),
    },
    # Shared-expert fused gate/up projection, TP=4.
    (320, 2560): {
        1: SkinnyGemmConfig(1, 64, 2, vector_width=4, static_k=2560),
        2: SkinnyGemmConfig(2, 128, 4, k_unroll=5, vector_width=4),
        4: SkinnyGemmConfig(4, 160, 1, k_unroll=2),
        8: SkinnyGemmConfig(8, 128, 1, vector_width=4, static_k=2560),
        16: SkinnyGemmConfig(16, 128, 1, vector_width=4, static_k=2560),
    },
    # LM head, TP=4.
    (62080, 2560): {
        1: SkinnyGemmConfig(1, 64, 2, vector_width=2, static_k=2560),
        2: SkinnyGemmConfig(2, 64, 2, vector_width=2, static_k=2560),
    },
    # Final HC down projection, replicated in a TP=4 deployment.
    (320, 10240): {
        1: SkinnyGemmConfig(1, 256, 1, static_k=10240),
        2: SkinnyGemmConfig(2, 128, 1, k_unroll=10),
        4: SkinnyGemmConfig(4, 128, 1, k_unroll=10),
        8: SkinnyGemmConfig(8, 128, 1, k_unroll=10),
    },
}


# B200 plans; the winning configs differ from the B300 table.
QWEN4_EXP_SM100_GEMM_PLANS: dict[tuple[int, int], dict[int, SkinnyGemmConfig]] = {
    # GDN fused B/A projection, TP=4.
    (24, 2560): {
        1: SkinnyGemmConfig(1, 128, 1, k_unroll=4, vector_width=4, static_k=2560),
        2: SkinnyGemmConfig(2, 128, 2, k_unroll=4, vector_width=4, static_k=2560),
        4: SkinnyGemmConfig(4, 128, 1, k_unroll=4, vector_width=4, static_k=2560),
        8: SkinnyGemmConfig(8, 128, 1, k_unroll=4, vector_width=4, static_k=2560),
        16: SkinnyGemmConfig(16, 128, 1, k_unroll=2, vector_width=4, static_k=2560),
    },
    # Shared-expert fused gate/up projection, TP=4.
    (320, 2560): {
        1: SkinnyGemmConfig(1, 128, 1, k_unroll=4, vector_width=4, static_k=2560),
        2: SkinnyGemmConfig(2, 128, 1, k_unroll=4, vector_width=4, static_k=2560),
        4: SkinnyGemmConfig(4, 160, 1, k_unroll=2, static_k=2560),
        8: SkinnyGemmConfig(8, 128, 1, k_unroll=4, vector_width=4, static_k=2560),
        16: SkinnyGemmConfig(16, 128, 1, vector_width=4, static_k=2560),
    },
    # Final HC down projection, replicated in a TP=4 deployment.
    (320, 10240): {
        1: SkinnyGemmConfig(1, 256, 1, k_unroll=4, static_k=10240),
        2: SkinnyGemmConfig(2, 256, 1, k_unroll=4, static_k=10240),
        4: SkinnyGemmConfig(4, 256, 1, k_unroll=2),
        8: SkinnyGemmConfig(8, 256, 1),
    },
    # GDN and QSA output projections, TP=4.
    (2560, 1536): {
        1: SkinnyGemmConfig(1, 128, 2, k_unroll=2, vector_width=4, static_k=1536),
        2: SkinnyGemmConfig(2, 64, 2, k_unroll=2, static_k=1536),
        4: SkinnyGemmConfig(4, 64, 2, k_unroll=2, static_k=1536),
    },
    # QSA fused QKV/gate + replicated indexer Q/K, TP=4.
    (4224, 2560): {
        1: SkinnyGemmConfig(1, 128, 4, vector_width=4, static_k=2560),
        2: SkinnyGemmConfig(2, 128, 2, vector_width=4, static_k=2560),
        4: SkinnyGemmConfig(4, 64, 4, vector_width=4, static_k=2560),
    },
    # GDN fused QKVZ projection, TP=4.
    (4096, 2560): {
        1: SkinnyGemmConfig(1, 128, 2, vector_width=4, static_k=2560),
        2: SkinnyGemmConfig(2, 64, 2, k_unroll=2, static_k=2560),
        4: SkinnyGemmConfig(4, 64, 2, k_unroll=4, vector_width=4, static_k=2560),
    },
    # LM head, TP=4.
    (62080, 2560): {
        1: SkinnyGemmConfig(1, 128, 2, k_unroll=4, vector_width=4),
    },
}


# DGX Spark (GB10) plans for TP=1 and TP=2 local shapes.
QWEN4_EXP_SM121_GEMM_PLANS: dict[tuple[int, int], dict[int, SkinnyGemmConfig]] = {
    # Shared-expert gate.
    (1, 2560): {
        1: SkinnyGemmConfig(1, 64, 1, k_unroll=2, static_k=2560),
        2: SkinnyGemmConfig(2, 64, 1, k_unroll=2, vector_width=4, static_k=2560),
        4: SkinnyGemmConfig(4, 128, 1, vector_width=2, static_k=2560),
        8: SkinnyGemmConfig(8, 128, 1, k_unroll=4, vector_width=4, static_k=2560),
        16: SkinnyGemmConfig(16, 128, 1, vector_width=4, static_k=2560),
    },
    # GDN fused B/A projection, TP=2.
    (48, 2560): {
        1: SkinnyGemmConfig(1, 128, 1, k_unroll=4, vector_width=4, static_k=2560),
        2: SkinnyGemmConfig(2, 128, 2, k_unroll=4, vector_width=4, static_k=2560),
        4: SkinnyGemmConfig(4, 128, 1, k_unroll=4, vector_width=4, static_k=2560),
        8: SkinnyGemmConfig(8, 128, 1, k_unroll=4, vector_width=4, static_k=2560),
        16: SkinnyGemmConfig(16, 128, 1, k_unroll=2, vector_width=4, static_k=2560),
    },
    # GDN fused B/A projection.
    (96, 2560): {
        1: SkinnyGemmConfig(1, 128, 4, k_unroll=2, vector_width=2, static_k=2560),
        2: SkinnyGemmConfig(2, 64, 4, vector_width=2, static_k=2560),
        4: SkinnyGemmConfig(4, 128, 4, vector_width=2, static_k=2560),
        8: SkinnyGemmConfig(8, 128, 4, k_unroll=2, vector_width=4, static_k=2560),
        16: SkinnyGemmConfig(16, 64, 1),
    },
    # Router.
    (512, 2560): {
        1: SkinnyGemmConfig(1, 256, 1, k_unroll=4, vector_width=2, static_k=2560),
        2: SkinnyGemmConfig(2, 256, 1, k_unroll=4, vector_width=2, static_k=2560),
        4: SkinnyGemmConfig(4, 64, 1, k_unroll=2, static_k=2560),
        8: SkinnyGemmConfig(8, 64, 2, static_k=2560),
        16: SkinnyGemmConfig(16, 64, 2, k_unroll=2, static_k=2560),
    },
    # Shared-expert fused gate/up, TP=2; retain after indexer fusion.
    (640, 2560): {
        1: SkinnyGemmConfig(1, 256, 1, vector_width=2, static_k=2560),
        2: SkinnyGemmConfig(2, 256, 1, vector_width=2, static_k=2560),
        4: SkinnyGemmConfig(4, 128, 1, k_unroll=2, vector_width=4, static_k=2560),
        8: SkinnyGemmConfig(8, 128, 1, vector_width=4, static_k=2560),
        16: SkinnyGemmConfig(16, 64, 1, static_k=2560),
    },
    # Shared-expert fused gate/up projection.
    (1280, 2560): {
        1: SkinnyGemmConfig(1, 32, 1, k_unroll=2, static_k=2560),
        2: SkinnyGemmConfig(2, 64, 1, vector_width=4, static_k=2560),
        4: SkinnyGemmConfig(4, 128, 1, k_unroll=4, vector_width=2, static_k=2560),
        8: SkinnyGemmConfig(8, 128, 1, k_unroll=2, vector_width=2, static_k=2560),
        16: SkinnyGemmConfig(16, 32, 1, k_unroll=2, static_k=2560),
    },
    # Shared-expert down projection.
    (2560, 640): {
        1: SkinnyGemmConfig(1, 64, 1, k_unroll=4, vector_width=2, static_k=640),
        2: SkinnyGemmConfig(2, 64, 1, vector_width=2, static_k=640),
        4: SkinnyGemmConfig(4, 64, 1, vector_width=2, static_k=640),
        8: SkinnyGemmConfig(8, 32, 1, k_unroll=4, vector_width=4, static_k=640),
        16: SkinnyGemmConfig(16, 32, 1, k_unroll=2, vector_width=4, static_k=640),
    },
    # GDN and QSA output projections, TP=2.
    (2560, 3072): {
        1: SkinnyGemmConfig(1, 128, 2, k_unroll=2, vector_width=4, static_k=3072),
        2: SkinnyGemmConfig(2, 64, 2, k_unroll=2, static_k=3072),
        4: SkinnyGemmConfig(4, 64, 2, k_unroll=2, static_k=3072),
    },
    # GDN and QSA output projections.
    (2560, 6144): {
        1: SkinnyGemmConfig(1, 64, 2, vector_width=4, static_k=6144),
        2: SkinnyGemmConfig(2, 64, 1, k_unroll=4, vector_width=4, static_k=6144),
        4: SkinnyGemmConfig(4, 64, 1, k_unroll=4, vector_width=4, static_k=6144),
        8: SkinnyGemmConfig(8, 128, 1, vector_width=2, static_k=6144),
        16: SkinnyGemmConfig(16, 32, 1, static_k=6144),
    },
    # QSA fused QKV/gate projection, TP=2 (LoRA keeps the indexer projection separate).
    (6656, 2560): {
        1: SkinnyGemmConfig(1, 128, 4, k_unroll=2, vector_width=4, static_k=2560),
        2: SkinnyGemmConfig(2, 128, 4, k_unroll=2, vector_width=4, static_k=2560),
        4: SkinnyGemmConfig(4, 64, 2, k_unroll=4, vector_width=4, static_k=2560),
    },
    # QSA fused QKV/gate + replicated indexer Q/K projection, TP=2.
    (7296, 2560): {
        1: SkinnyGemmConfig(1, 32, 1, vector_width=2, static_k=2560),
        2: SkinnyGemmConfig(2, 32, 1, vector_width=2, static_k=2560),
        4: SkinnyGemmConfig(4, 32, 1, vector_width=2, static_k=2560),
        8: SkinnyGemmConfig(8, 32, 1, vector_width=2, static_k=2560),
        16: SkinnyGemmConfig(16, 32, 4, vector_width=2, static_k=2560),
    },
    # GDN fused QKVZ projection, TP=2.
    (8192, 2560): {
        1: SkinnyGemmConfig(1, 128, 2, vector_width=4, static_k=2560),
        2: SkinnyGemmConfig(2, 64, 2, k_unroll=2, static_k=2560),
        4: SkinnyGemmConfig(4, 64, 2, k_unroll=4, vector_width=4, static_k=2560),
    },
    # HC up projection.
    (10240, 320): {
        1: SkinnyGemmConfig(1, 32, 1, k_unroll=2, vector_width=2, static_k=320),
        2: SkinnyGemmConfig(2, 32, 1, vector_width=2, static_k=320),
        4: SkinnyGemmConfig(4, 32, 1, k_unroll=2, vector_width=2, static_k=320),
        8: SkinnyGemmConfig(8, 32, 1, vector_width=2, static_k=320),
    },
    # QSA fused QKV/gate + replicated indexer Q/K projection.
    (13952, 2560): {
        1: SkinnyGemmConfig(1, 32, 1, vector_width=4, static_k=2560),
        2: SkinnyGemmConfig(2, 32, 1, k_unroll=2, vector_width=4, static_k=2560),
        4: SkinnyGemmConfig(4, 32, 1, vector_width=4, static_k=2560),
        8: SkinnyGemmConfig(8, 64, 1, vector_width=2, static_k=2560),
        16: SkinnyGemmConfig(16, 32, 1, k_unroll=2, vector_width=4, static_k=2560),
    },
    # GDN fused QKVZ projection.
    (16384, 2560): {
        1: SkinnyGemmConfig(1, 32, 1, k_unroll=4, vector_width=2, static_k=2560),
        2: SkinnyGemmConfig(2, 32, 1, vector_width=2, static_k=2560),
        4: SkinnyGemmConfig(4, 32, 1, k_unroll=2, vector_width=2, static_k=2560),
        8: SkinnyGemmConfig(8, 64, 1, vector_width=2, static_k=2560),
        16: SkinnyGemmConfig(16, 32, 1, k_unroll=2, vector_width=2, static_k=2560),
    },
    # LM head, TP=2.
    (124160, 2560): {
        1: SkinnyGemmConfig(1, 128, 2, k_unroll=4, vector_width=4),
        2: SkinnyGemmConfig(2, 64, 2, k_unroll=2),
    },
    # LM head.
    (248320, 2560): {
        1: SkinnyGemmConfig(1, 128, 1, k_unroll=2, vector_width=4, static_k=2560),
        2: SkinnyGemmConfig(2, 128, 1, k_unroll=2, vector_width=4, static_k=2560),
        4: SkinnyGemmConfig(4, 32, 1, vector_width=4, static_k=2560),
        8: SkinnyGemmConfig(8, 64, 1, k_unroll=2, vector_width=4, static_k=2560),
        16: SkinnyGemmConfig(16, 128, 1, k_unroll=2, vector_width=4, static_k=2560),
    },
}


# lebeche fork: RTX 3090 (sm_86) plans for the TP=2 local shapes of
# Qwen3.8-Flash-Next, selected by CUDA graph replay sweeps (24 distinct
# weights, best of 40) over block_size x outputs_per_block x vector_width x
# {static K, k_unroll}; `stock -> skinny` us per call. Only points that beat
# the standard linear are listed (HC and shared-expert shapes lose from M=8;
# the LM head is bandwidth-bound and stays on the standard path). The merged
# HC down+inject projection (336, 10240) is planned here because the fused
# down+SiLU kernel cannot dispatch on M at run time under torch.compile; the
# router gate (512, 2560) reaches this op through GateLinear's tier 7. The
# kernels need the portable-elect build, see vllm/merlin/lebeche_sm86.py.
QWEN4_EXP_SM86_GEMM_PLANS: dict[tuple[int, int], dict[int, SkinnyGemmConfig]] = {
# skipped (no win) 96 2560 16 5.96 6.3
    # Merged HC down+inject projection (replicated, padded to 16 rows).
    (336, 10240): {
        1: SkinnyGemmConfig(1, 64, 1, static_k=10240),  # 15.51 -> 9.66 us
        2: SkinnyGemmConfig(2, 64, 1, static_k=10240),  # 15.2 -> 9.84 us
        3: SkinnyGemmConfig(3, 160, 2, static_k=10240),  # 15.21 -> 10.45 us
        4: SkinnyGemmConfig(4, 128, 2, static_k=10240),  # 15.35 -> 10.99 us
    },
    # MoE router gate (replicated; reached through GateLinear tier 7).
    (512, 2560): {
        1: SkinnyGemmConfig(1, 32, 1, vector_width=2, static_k=2560),  # 6.17 -> 4.9 us
        2: SkinnyGemmConfig(2, 160, 1, static_k=2560),  # 8.45 -> 5.09 us
        3: SkinnyGemmConfig(3, 128, 1, vector_width=4, static_k=2560),  # 8.48 -> 5.36 us
        4: SkinnyGemmConfig(4, 64, 1, vector_width=4, static_k=2560),  # 8.49 -> 5.74 us
    },
    # Shared-expert gate (replicated).
    (1, 2560): {
        1: SkinnyGemmConfig(1, 256, 1, vector_width=2, static_k=2560),  # 3.6 -> 1.86 us
        2: SkinnyGemmConfig(2, 320, 1, vector_width=4, static_k=2560),  # 3.64 -> 2.03 us
        3: SkinnyGemmConfig(3, 128, 1, vector_width=4, static_k=2560),  # 3.61 -> 2.21 us
        4: SkinnyGemmConfig(4, 128, 1, vector_width=4, static_k=2560),  # 3.62 -> 2.41 us
    },
    # HC up projection (replicated).
    (10240, 320): {
        1: SkinnyGemmConfig(1, 32, 2, vector_width=2, static_k=320),  # 11.03 -> 9.43 us
        2: SkinnyGemmConfig(2, 32, 2, vector_width=2, static_k=320),  # 11.08 -> 9.54 us
        3: SkinnyGemmConfig(3, 32, 2, vector_width=2, static_k=320),  # 11.09 -> 9.67 us
        4: SkinnyGemmConfig(4, 32, 2, vector_width=2, static_k=320),  # 11.12 -> 10.29 us
    },
    # Final HC mixer down projection (replicated).
    (320, 10240): {
        1: SkinnyGemmConfig(1, 64, 2, vector_width=2, static_k=10240),  # 15.16 -> 9.31 us
        2: SkinnyGemmConfig(2, 160, 1, static_k=10240),  # 14.69 -> 9.55 us
        3: SkinnyGemmConfig(3, 256, 2, k_unroll=2),  # 14.72 -> 10.01 us
        4: SkinnyGemmConfig(4, 128, 2, static_k=10240),  # 14.86 -> 10.8 us
    },
    # Shared-expert fused gate/up projection, TP=2.
    (640, 2560): {
        1: SkinnyGemmConfig(1, 320, 1, vector_width=4, static_k=2560),  # 6.77 -> 5.68 us
        2: SkinnyGemmConfig(2, 64, 2, vector_width=4, static_k=2560),  # 9.64 -> 5.79 us
        3: SkinnyGemmConfig(3, 32, 1, static_k=2560),  # 9.63 -> 6.04 us
        4: SkinnyGemmConfig(4, 64, 1, vector_width=4, static_k=2560),  # 9.66 -> 6.15 us
    },
    # Shared-expert down projection, TP=2.
    (2560, 320): {
        1: SkinnyGemmConfig(1, 32, 2, vector_width=2, static_k=320),  # 5.9 -> 3.83 us
        2: SkinnyGemmConfig(2, 32, 2, vector_width=2, static_k=320),  # 6.09 -> 3.91 us
        3: SkinnyGemmConfig(3, 32, 2, vector_width=2, static_k=320),  # 6.12 -> 4.07 us
        4: SkinnyGemmConfig(4, 32, 2, vector_width=2, static_k=320),  # 6.13 -> 4.29 us
    },
    # GDN fused B/A projection, TP=2.
    (48, 2560): {
        1: SkinnyGemmConfig(1, 320, 2, vector_width=2, static_k=2560),  # 4.93 -> 2.2 us
        2: SkinnyGemmConfig(2, 128, 2, vector_width=2, static_k=2560),  # 16.77 -> 2.5 us
        3: SkinnyGemmConfig(3, 128, 2, vector_width=4, static_k=2560),  # 19.02 -> 2.73 us
        4: SkinnyGemmConfig(4, 128, 2, vector_width=4, static_k=2560),  # 18.88 -> 2.98 us
        8: SkinnyGemmConfig(8, 128, 2, vector_width=4, static_k=2560),  # 17.81 -> 3.92 us
        16: SkinnyGemmConfig(16, 128, 2, vector_width=2, static_k=2560),  # 18.87 -> 5.38 us
    },
    # GDN fused B/A projection, replicated.
    (96, 2560): {
        1: SkinnyGemmConfig(1, 128, 2, vector_width=4, static_k=2560),  # 4.24 -> 2.58 us
        2: SkinnyGemmConfig(2, 128, 4, vector_width=4, static_k=2560),  # 20.37 -> 3.11 us
        3: SkinnyGemmConfig(3, 128, 4, vector_width=4, static_k=2560),  # 20.37 -> 3.33 us
        4: SkinnyGemmConfig(4, 64, 2, vector_width=2, static_k=2560),  # 20.39 -> 3.59 us
        8: SkinnyGemmConfig(8, 128, 2, vector_width=4, static_k=2560),  # 20.37 -> 4.62 us
    },
    # QSA fused QKV/gate + replicated indexer Q/K projection, TP=2.
    (7296, 2560): {
        1: SkinnyGemmConfig(1, 64, 1, k_unroll=2),  # 47.88 -> 44.14 us
        2: SkinnyGemmConfig(2, 64, 2),  # 52.01 -> 44.45 us
        3: SkinnyGemmConfig(3, 64, 1),  # 52.01 -> 44.43 us
        4: SkinnyGemmConfig(4, 64, 1),  # 52.06 -> 44.74 us
    },
    # QSA output projection, TP=2.
    (2560, 3072): {
        1: SkinnyGemmConfig(1, 128, 1, k_unroll=2),  # 27.84 -> 19.58 us
        2: SkinnyGemmConfig(2, 128, 1),  # 23.89 -> 19.69 us
        3: SkinnyGemmConfig(3, 128, 1, k_unroll=2),  # 23.98 -> 19.9 us
        4: SkinnyGemmConfig(4, 32, 1, static_k=3072),  # 24.04 -> 20.09 us
    },
    # MTP fc_embedding / fc_hidden, TP=2.
    (1280, 2560): {
        1: SkinnyGemmConfig(1, 64, 1, k_unroll=2),  # 10.67 -> 9.34 us
        2: SkinnyGemmConfig(2, 64, 1, k_unroll=2),  # 14.4 -> 9.45 us
        3: SkinnyGemmConfig(3, 64, 1, k_unroll=2),  # 14.45 -> 9.55 us
        4: SkinnyGemmConfig(4, 32, 1, static_k=2560),  # 14.45 -> 9.77 us
    },
}

# lebeche fork (MERLIN_LEBECHE_SM86_LL_EXTRA): remaining decode BF16 fallback
# shapes on RTX 3090 (PLE kv_proj and c=2 MTP fc_hidden at M_flat=8).
if SM86_LL_EXTRA_SWITCH:
    QWEN4_EXP_SM86_GEMM_PLANS[(1280, 2560)][8] = SkinnyGemmConfig(
        8, 32, 2, static_k=2560
    )  # 14.65 -> 12.33 us
    # PLE fused key/value projection (replicated, layer 2).
    QWEN4_EXP_SM86_GEMM_PLANS[(12800, 2560)] = {
        1: SkinnyGemmConfig(1, 64, 1, k_unroll=2),  # 85.56 -> 75.64 us
        2: SkinnyGemmConfig(2, 64, 1, k_unroll=2),  # 79.64 -> 75.66 us
        3: SkinnyGemmConfig(3, 64, 1, k_unroll=2),  # 80.28 -> 75.95 us
        4: SkinnyGemmConfig(4, 64, 1, k_unroll=2),  # 80.66 -> 76.29 us
    }


QWEN4_EXP_GEMM_PLANS_BY_CAPABILITY: dict[
    tuple[int, int] | None, dict[tuple[int, int], dict[int, SkinnyGemmConfig]]
] = {
    (10, 3): QWEN4_EXP_SM103_GEMM_PLANS,
    (10, 0): QWEN4_EXP_SM100_GEMM_PLANS,
    (9, 0): QWEN4_EXP_SM90_GEMM_PLANS,
    (12, 1): QWEN4_EXP_SM121_GEMM_PLANS,
}
if SM86_LL_SWITCH:
    # Tuned on sm_86 only; sm_80/89 reuse the table (same kernel build).
    for _capability in sorted(SM86_LL_CAPABILITIES):
        QWEN4_EXP_GEMM_PLANS_BY_CAPABILITY[_capability] = QWEN4_EXP_SM86_GEMM_PLANS


def _is_packed_row_major(tensor: torch.Tensor) -> bool:
    return tensor.dim() == 2 and tensor.stride() == (tensor.shape[1], 1)


def _runtime_ok(
    x: torch.Tensor, weight: torch.Tensor, config: SkinnyGemmConfig
) -> bool:
    return (
        not envs.VLLM_BATCH_INVARIANT
        and x.dim() == 2
        and row_stride_ok(x, config)
        and _is_packed_row_major(weight)
        and x.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and x.is_cuda
        and weight.is_cuda
        and x.device == weight.device
        and x.shape[1] == weight.shape[1]
    )


class _Qwen4ExpLowLatencyApply:
    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if bias is None and not envs.VLLM_BATCH_INVARIANT:
            return torch.ops.vllm.qwen4_exp_low_latency_gemm(x, layer.weight)
        return super().apply(layer, x, bias)  # type: ignore[misc]


class Qwen4ExpLowLatencyLinearMethod(_Qwen4ExpLowLatencyApply, UnquantizedLinearMethod):
    pass


class Qwen4ExpLowLatencyEmbeddingMethod(
    _Qwen4ExpLowLatencyApply, UnquantizedEmbeddingMethod
):
    pass


def _qwen4_exp_low_latency_gemm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    plans = QWEN4_EXP_GEMM_PLANS_BY_CAPABILITY.get(
        current_platform.get_device_capability(), {}
    )
    plan = plans.get((weight.shape[0], weight.shape[1]))
    if plan is not None:
        if x.dim() == 2:
            config = plan.get(x.shape[0])
            if (
                config is not None
                and _runtime_ok(x, weight, config)
                and shape_dynamic_skinny_gemm.is_available()
            ):
                return shape_dynamic_skinny_gemm(x, weight, config)
        elif SM86_LL_EXTRA_SWITCH and x.dim() > 2 and x.is_contiguous():
            # MTP fc_hidden passes a contiguous 3D tensor [T, hc_count, H].
            x_2d = x.view(-1, x.shape[-1])
            config = plan.get(x_2d.shape[0])
            if (
                config is not None
                and _runtime_ok(x_2d, weight, config)
                and shape_dynamic_skinny_gemm.is_available()
            ):
                return shape_dynamic_skinny_gemm(x_2d, weight, config).view(
                    *x.shape[:-1], weight.shape[0]
                )
    return torch.nn.functional.linear(x, weight)


def _qwen4_exp_low_latency_gemm_fake(
    x: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], weight.shape[0]))


direct_register_custom_op(
    op_name="qwen4_exp_low_latency_gemm",
    op_func=_qwen4_exp_low_latency_gemm,
    fake_impl=_qwen4_exp_low_latency_gemm_fake,
)


def enable_qwen4_exp_low_latency_gemm(
    module: nn.Module,
    dtype: torch.dtype,
) -> None:
    plans = QWEN4_EXP_GEMM_PLANS_BY_CAPABILITY.get(
        current_platform.get_device_capability(), {}
    )
    if dtype != torch.bfloat16 or not plans:
        return
    if not shape_dynamic_skinny_gemm.is_available():
        return

    warmup_configs: set[SkinnyGemmConfig] = set()
    for child in module.modules():
        is_linear = (
            isinstance(child, LinearBase)
            and type(child.quant_method) is UnquantizedLinearMethod
        )
        is_head = (
            isinstance(child, ParallelLMHead)
            and type(child.quant_method) is UnquantizedEmbeddingMethod
        )
        if not (is_linear or is_head):
            continue
        weight = getattr(child, "weight", None)
        if weight is None or weight.dim() != 2:
            continue
        plan = plans.get((weight.shape[0], weight.shape[1]))
        if plan is None:
            continue
        if is_linear:
            child.quant_method = Qwen4ExpLowLatencyLinearMethod()
        else:
            child.quant_method = Qwen4ExpLowLatencyEmbeddingMethod()
        warmup_configs.update(plan.values())

    if warmup_configs:
        shape_dynamic_skinny_gemm.request_warmup_configs(dtype, warmup_configs)

    # lebeche fork: fused small-M HC epilogues on top of the patched Linears.
    from .hc_small_fused import enable_qwen4_exp_hc_small_fused

    enable_qwen4_exp_hc_small_fused(module, dtype)
