# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Token-chunked Marlin MoE workspaces.

``MarlinExperts`` sizes its two intermediate caches for every token of the
step: ``workspace1 = (M * topk, max(N, K))`` and ``workspace2 = (M * topk *
max(2N, K),)``. The workspace manager keeps the high-water mark for the life
of the process, so a wide-topk model pays it before the KV cache is
allocated: Hy4 Flash (topk 16, N = K = 1536, CHUNK 2048) holds 96 + 192 MiB
on every rank, which is most of what PyTorch allocates over the weights.

With ``MERLIN_MOE_CHUNK_TOKENS`` = C > 0, a step of M > C tokens runs the same
``fused_marlin_moe`` over consecutive C-token slices of the hidden states,
top-k ids and weights and the output, reusing caches sized for C tokens.
Each token's expert outputs and its top-k sum depend only on that token's
row (``fused_marlin_moe`` slices its caches to the call's M), so the result is
the same computation at the same precision; only the GEMM tiling a kernel
picks for a smaller M can change the reduction order, as any batch-size change
does. Steps of at most C tokens (decode, spec-verify, CUDA-graph sizes below C)
take the stock path unchanged.

The output buffer the modular kernel hands to ``apply`` shares the start of
workspace1 (``_allocate_buffers`` reuses it "since there is only one chunk"),
so in chunked mode workspace1 also reserves the M x K output rows and the
cache is placed after them: no chunk can overwrite an earlier chunk's output.

0 (the default) is the stock path. A lane turns it on by exporting it; the
hy4flash launcher passes an exported value through and sets none itself.
"""

from __future__ import annotations

ENV = "MERLIN_MOE_CHUNK_TOKENS"


def chunk_tokens() -> int:
    """Tokens per Marlin MoE call from ``MERLIN_MOE_CHUNK_TOKENS``; 0 = stock path."""
    import vllm.envs as envs

    return int(envs.MERLIN_MOE_CHUNK_TOKENS)


def is_chunked(m: int, chunk: int) -> bool:
    return chunk > 0 and m > chunk


def marlin_workspace_shapes(
    m: int, n: int, k: int, topk: int, chunk: int
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """``MarlinExperts.workspace_shapes``: the stock shapes, or with chunking
    workspace1 = output rows (M x K) + one chunk's cache, flat."""
    if not is_chunked(m, chunk):
        return (m * topk, max(n, k)), (m * topk * max(2 * n, k),), (m, k)
    return (
        (m * k + chunk * topk * max(n, k),),
        (chunk * topk * max(2 * n, k),),
        (m, k),
    )


def output_rows_offset(m: int, k: int) -> int:
    """Elements of workspace1 the chunked layout keeps for the output."""
    return m * k


def chunk_slices(m: int, chunk: int) -> list[slice]:
    return [slice(s, min(s + chunk, m)) for s in range(0, m, chunk)]


def chunk_tokens_for(experts, hooked: bool = False) -> int:
    """The chunk size for one ``MarlinExperts`` call: 0 (the stock path) with a
    post-GEMM2 hook or a LoRA context, which both read whole-step rows."""
    if hooked or experts._lora_context is not None:
        return 0
    if experts.post_gemm2_hook is not None:
        return 0
    return chunk_tokens()


def run_chunked(fused_marlin_moe, chunk: int, **kw) -> None:
    """``fused_marlin_moe(**kw)``, or with M > chunk the same call over token
    slices of the hidden states, top-k weights and ids and the output, with the
    per-chunk cache placed after the output rows workspace13 shares."""
    x, out = kw["hidden_states"], kw["output"]
    m = x.size(0)
    if not is_chunked(m, chunk):
        fused_marlin_moe(**kw)
        return
    cache2 = kw["intermediate_cache2"].view(-1)[output_rows_offset(m, out.size(-1)) :]
    for s in chunk_slices(m, chunk):
        piece = {
            "hidden_states": x[s],
            "topk_weights": kw["topk_weights"][s],
            "topk_ids": kw["topk_ids"][s],
            "output": out[s],
            "intermediate_cache2": cache2,
        }
        fused_marlin_moe(**{**kw, **piece})
