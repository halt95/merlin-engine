# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Row-chunked Qwen3 vision tower.

Only per-row operations are sliced. Each block still makes one attention call
over the entire batch. Smaller GEMMs can change rounding, so large-image
support is qualified by a numerics identity check and a mixed-size memory
sweep on the GPU, not by CPU allclose alone.
"""

import logging
from collections.abc import Callable, Sequence
from typing import Any

import torch

logger = logging.getLogger(__name__)

MIN_CHUNK_PIXELS = 8_388_608
# Process-local path entries, including profiling and failed encoder calls.
chunked_vision_calls = 0
_encoder_cudagraph_warned = False


def warn_encoder_cudagraphs(enabled: bool) -> None:
    """Warn once when encoder graphs can bypass enabled vision chunking."""
    from vllm import envs

    global _encoder_cudagraph_warned
    if (
        enabled
        and envs.VLLM_MERLIN_VISION_CHUNK_ROWS > 0
        and not _encoder_cudagraph_warned
    ):
        logger.warning(
            "Encoder CUDA graphs are enabled with Merlin vision chunking; "
            "metadata-only graph calls retain the plain tower and do not get "
            "the chunked memory benefit. Disable cudagraph_mm_encoder before "
            "qualifying a larger image cap."
        )
        _encoder_cudagraph_warned = True


def chunked_rows(
    fn: Callable, x: torch.Tensor, rows: int, unit: int = 1
) -> torch.Tensor:
    """Write row-local results into one output; keep merger groups intact."""
    rows = max(unit, rows // unit * unit)
    n = x.shape[0]
    if n % unit:
        raise ValueError("Vision rows must be divisible by the merge unit")
    if n <= rows:
        return fn(x)
    first = fn(x[:rows])
    out = first.new_empty((n // unit, *first.shape[1:]))
    out[: rows // unit].copy_(first)
    del first
    for start in range(rows, n, rows):
        stop = min(start + rows, n)
        out[start // unit : stop // unit].copy_(fn(x[start:stop]))
    return out


def chunked_block(
    block: Any, x: torch.Tensor, metadata: dict, rows: int
) -> torch.Tensor:
    """Chunk norm/QKV/rotary/proj and both residuals; attend once globally."""
    attn = block.attn
    n, batch, _ = x.shape
    heads = attn.num_attention_heads_per_partition
    dim = attn.hidden_size_per_attention_head
    q, k, v = (x.new_empty((batch, n, heads, dim)) for _ in range(3))
    cos = metadata["rotary_pos_emb_cos"]
    sin = metadata["rotary_pos_emb_sin"]
    for start in range(0, n, rows):
        stop = min(start + rows, n)
        qkv, _ = attn.qkv(block.norm1(x[start:stop]))
        qkv = qkv.view(stop - start, batch, 3, heads, dim)
        qk = (
            qkv[:, :, :2]
            .permute(2, 1, 0, 3, 4)
            .reshape(2 * batch, stop - start, heads, dim)
            .contiguous()
        )
        if cos is not None and sin is not None:
            qk = attn.apply_rotary_emb(qk, cos[start:stop], sin[start:stop])
        qk = qk.view(2, batch, stop - start, heads, dim)
        q[:, start:stop].copy_(qk[0])
        k[:, start:stop].copy_(qk[1])
        v[:, start:stop].copy_(qkv[:, :, 2].permute(1, 0, 2, 3))
        del qkv, qk

    cu_seqlens = metadata["cu_seqlens"]
    # CuDNN BF16 metadata encodes V's upstream interleaved-QKV 3x stride.
    # Our independent contiguous V buffer has the same stride as Q/K.
    backend = attn.attn.attn_backend
    if backend.name == "FLASHINFER" and not attn.attn.fp8_enabled:
        split = cu_seqlens.shape[0] // 2
        cu_seqlens = cu_seqlens.clone()
        cu_seqlens[split:] //= 3
    context = attn.attn(
        query=q,
        key=k,
        value=v,
        cu_seqlens=cu_seqlens,
        max_seqlen=metadata["max_seqlen"],
        sequence_lengths=metadata.get("sequence_lengths"),
    )
    del q, k, v
    out = torch.empty_like(x)
    for start in range(0, n, rows):
        stop = min(start + rows, n)
        projected, _ = attn.proj(
            context[:, start:stop]
            .permute(1, 0, 2, 3)
            .reshape(stop - start, batch, heads * dim)
            .contiguous()
        )
        out[start:stop].copy_(x[start:stop] + projected)
        del projected
    del context
    for start in range(0, n, rows):
        stop = min(start + rows, n)
        out[start:stop] += block.mlp(block.norm2(out[start:stop]))
    return out


def maybe_chunked_vision(
    tower: Any,
    x: torch.Tensor,
    grid_thw: torch.Tensor | Sequence[Sequence[int]] | None,
    encoder_metadata: dict | None,
) -> torch.Tensor | None:
    """Return None to execute the untouched upstream forward.

    The threshold covers all patch rows in this call, across images/frames.
    Precomputed-metadata CUDA graphs have no grid and keep the upstream path;
    use eager encoding for the large-image memory benefit.
    """
    from vllm import envs

    global chunked_vision_calls
    rows = envs.VLLM_MERLIN_VISION_CHUNK_ROWS
    threshold = max(MIN_CHUNK_PIXELS, envs.VLLM_MERLIN_VISION_CHUNK_MIN_PIXELS)
    if rows == 0 or grid_thw is None:
        return None
    # The encoder budget bounds total rows, not each image/frame separately.
    n = x.shape[0]
    if n * tower.patch_size**2 <= threshold:
        return None
    grid = grid_thw if not isinstance(grid_thw, torch.Tensor) else grid_thw.tolist()
    chunked_vision_calls += 1
    if chunked_vision_calls == 1:
        logger.info(
            "Merlin vision chunk path active: patch_rows=%d chunk_rows=%d "
            "chunks=%d items=%d frames=%d threshold_pixels=%d",
            n,
            rows,
            (n + rows - 1) // rows,
            len(grid),
            sum(t for t, _, _ in grid),
            threshold,
        )
    logger.debug(
        "Merlin vision chunk call=%d patch_rows=%d chunk_rows=%d items=%d",
        chunked_vision_calls,
        n,
        rows,
        len(grid),
    )

    hidden = tower.input_norm(
        x.to(device=tower.device, non_blocking=True), tower.dtype
    )
    hidden = chunked_rows(tower.patch_embed, hidden, rows)
    metadata = encoder_metadata
    if metadata is None:
        metadata = tower.prepare_encoder_metadata(grid)
    hidden = (hidden + metadata["pos_embeds"]).unsqueeze(1)
    features = []
    for layer_num, block in enumerate(tower.blocks):
        hidden = chunked_block(block, hidden, metadata, rows)
        if layer_num in tower.deepstack_visual_indexes:
            idx = tower.deepstack_visual_indexes.index(layer_num)
            features.append(
                chunked_rows(
                    tower.deepstack_merger_list[idx],
                    hidden,
                    rows,
                    tower.spatial_merge_unit,
                )
            )
    main = chunked_rows(tower.merger, hidden, rows, tower.spatial_merge_unit)
    if not features:
        return main
    out = main.new_empty((main.shape[0], sum(f.shape[1] for f in [main, *features])))
    offset = 0
    for feature in [main, *features]:
        out[:, offset : offset + feature.shape[1]].copy_(feature)
        offset += feature.shape[1]
    return out
