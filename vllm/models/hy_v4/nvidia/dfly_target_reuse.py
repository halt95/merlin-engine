# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modified by the Merlin project; see NOTICE.
"""DFly2 reuse of the target's sparse-layer KV and top-k indices.

Adapted from #68296 ``vllm/models/hy_v4/nvidia/dfly_target_reuse.py:1-176``.

Each draft layer aliases one target sparse layer's KV storage (layers
25/29/33/37/41 on Hy4 Flash) and builds its sparse indices from the top-k the
target index layers selected for the request's last accepted token.

Split cache: the draft reads the TP-sharded (DCP)
target cache through the same DCP path the target uses; nothing is
replicated. That holds because the draft attention is built by the same
factory (equal ``MLAAttentionSpec``, ``dcp_sharded`` included, checked below),
runs at the target's DCP size and rank (checked per pair), writes its query
block through the target group's DCP-aware slot mapping (the speculator), and
reads global per-request positions that the impl filters to local slots.

Differences from upstream:
- Target layers are ``HYV4FlashAttention`` (Merlin's Flash port).
- Specs come from ``MLAAttention.get_kv_cache_spec`` (0.33 API) and are
  compared with ``topk_buffer_width``, ``is_index_group_leader`` and
  ``is_draft`` cleared.
- DCP checks per aliased pair (``vllm.merlin.hy4_kvshard.validate_dfly_kv_alias``
  plus a local size/rank/combine check).
"""

from dataclasses import replace

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.merlin.hy4_kvshard import validate_dfly_kv_alias
from vllm.models.hy_v4_flash.nvidia.attention import HYV4FlashAttention
from vllm.v1.kv_cache_interface import KVCacheConfig, MLAAttentionSpec
from vllm.v1.worker.utils import customize_attention_spec

from .dfly import HYV4DFlyForCausalLM
from .dfly_inputs import build_dfly_reuse_indices
from .dfly_snapshot import snapshot_rows, source_snapshot_rows


def check_dcp_alias(source_mla: nn.Module, dest_mla: nn.Module) -> None:
    """A draft layer may alias a target layer only at the same DCP layout."""
    s_impl, d_impl = getattr(source_mla, "impl", None), getattr(dest_mla, "impl", None)
    for attr in ("dcp_world_size", "dcp_rank"):
        s, d = (
            getattr(s_impl, attr, 1 if attr == "dcp_world_size" else 0),
            getattr(d_impl, attr, 1 if attr == "dcp_world_size" else 0),
        )
        if s != d:
            raise ValueError(
                f"DFly draft and target KV layer differ in {attr} ({d} vs {s}); a "
                "replicated draft cannot read a DCP-sharded target cache"
            )
    if (getattr(source_mla, "dcp_manager", None) is None) != (
        getattr(dest_mla, "dcp_manager", None) is None
    ):
        raise ValueError("DFly draft and target must both run the DCP gather/combine")
    # The shared-cache guard: sink layout, replication, gather/combine.
    validate_dfly_kv_alias(source_mla, dest_mla)


def storage_spec(mla_attn: nn.Module, vllm_config: VllmConfig) -> MLAAttentionSpec:
    spec = mla_attn.get_kv_cache_spec(vllm_config)
    if not isinstance(spec, MLAAttentionSpec):
        raise ValueError("DFly requires MLA KV cache specs")
    spec = customize_attention_spec(mla_attn.get_attn_backend(), spec)
    if not isinstance(spec, MLAAttentionSpec):
        raise ValueError("DFly backend must preserve the MLA KV cache spec")
    return replace(
        spec, topk_buffer_width=None, is_index_group_leader=False, is_draft=False
    )


class HYV4DFlyTargetReuse:
    def __init__(
        self,
        target: nn.Module,
        draft: HYV4DFlyForCausalLM,
        vllm_config: VllmConfig,
        draft_vllm_config: VllmConfig | None = None,
    ):
        config = draft.config
        if vllm_config.cache_config.cache_dtype not in (
            "auto",
            "bfloat16",
            "fp8_ds_mla",
        ):
            raise ValueError("DFly requires BF16 or fp8_ds_mla MLA KV cache")
        language_model = (
            target.get_language_model()
            if hasattr(target, "get_language_model")
            else target
        )
        target_layers = language_model.model.layers
        self.index_layers: dict[int, HYV4FlashAttention] = {}
        self.layer_pairs: list[tuple[HYV4FlashAttention, HYV4FlashAttention]] = []
        self.aliases: dict[str, str] = {}
        self.draft = draft
        self.config = config

        def target_attention(index: int) -> HYV4FlashAttention:
            attn = getattr(target_layers[index], "self_attn", None)
            if not isinstance(attn, HYV4FlashAttention):
                raise ValueError(
                    f"DFly target layer {index} is not a local HYV4 Flash MLA layer "
                    "(is it on this pipeline stage?)"
                )
            if attn.is_window_layer:
                raise ValueError(f"DFly target layer {index} must be sparse, not SWA")
            return attn

        for index in config.target_index_layer:
            attn = target_attention(index)
            if attn.indexer is None or attn.topk_indices_buffer is None:
                raise ValueError(
                    f"DFly target index layer {index} must run its own indexer"
                )
            if attn.topk_indices_buffer.shape[-1] != config.target_index_topk:
                raise ValueError(
                    f"DFly target index layer {index} has incompatible top-k width"
                )
            self.index_layers[index] = attn
        draft_config = draft_vllm_config or vllm_config
        for layer, index in zip(draft.model.dfly_layers, config.target_kv_layers):
            source, dest = target_attention(index), layer.self_attn
            if type(source.rotary_emb) is not type(dest.rotary_emb):
                raise ValueError(f"DFly target KV layer {index} uses incompatible RoPE")
            if source.num_local_heads != dest.num_local_heads:
                raise ValueError("DFly target and draft must use matching attention TP")
            if storage_spec(source.mla_attn, vllm_config) != storage_spec(
                dest.mla_attn, draft_config
            ):
                raise ValueError(
                    f"DFly target KV layer {index} has an incompatible cache spec"
                )
            check_dcp_alias(source.mla_attn, dest.mla_attn)
            self.layer_pairs.append((source, dest))
            self.aliases[dest.mla_attn.layer_name] = source.mla_attn.layer_name
        if len(set(self.aliases.values())) != len(self.aliases):
            raise ValueError("DFly needs a distinct target KV layer per draft layer")
        # One stacked snapshot buffer, so one launch builds every draft layer's
        # indices; each draft layer reads the slot of its target index layer.
        index_layer_ids = list(self.index_layers)
        first = next(iter(self.index_layers.values())).topk_indices_buffer
        # Each request's last K + 1 target
        # rows only (the source row is one of them; dfly_snapshot.py)
        self.snapshot_width = draft.model.query_width
        self.snapshots = torch.full(
            (
                len(index_layer_ids),
                snapshot_rows(vllm_config.scheduler_config, self.snapshot_width),
                first.shape[-1],
            ),
            -1,
            dtype=first.dtype,
            device=first.device,
        )
        for slot, index in enumerate(index_layer_ids):
            self.index_layers[index].topk_snapshot = self.snapshots[slot]
            self.index_layers[index].topk_snapshot_width = self.snapshot_width
        self.layer_snapshot_slots = torch.tensor(
            [index_layer_ids.index(i) for i in config.target_index_layer],
            dtype=torch.int32,
            device=first.device,
        )
        self._set_storage_aliases(True)

    def _set_storage_aliases(self, enabled: bool) -> None:
        """Toggle the KV-sharing alias on the draft layers.

        The alias makes KV cache allocation bind each draft layer to its target
        layer's storage (``attn_utils.get_shared_kv_cache_layers``). The draft
        must still write its query block KV, so the alias is only kept while
        the cache is allocated and bound (#68296 dfly_target_reuse.py:117-128).
        """
        for source, dest in self.layer_pairs:
            dest.mla_attn.kv_sharing_target_layer_name = (
                source.mla_attn.layer_name if enabled else None
            )

    def validate_groups(self, config: KVCacheConfig) -> None:
        self._set_storage_aliases(True)
        groups = {
            name: i
            for i, group in enumerate(config.kv_cache_groups)
            for name in group.layer_names
        }
        for draft, target in self.aliases.items():
            if draft not in groups or groups[draft] != groups.get(target):
                raise ValueError(
                    f"DFly alias {draft} and {target} must share a KV group"
                )

    def validate_storage(self) -> None:
        for source, dest in self.layer_pairs:
            a, b = source.mla_attn.kv_cache, dest.mla_attn.kv_cache
            if isinstance(a, (list, tuple)):
                a, b = a[0], b[0]
            if (
                a.numel() == 0
                or a.data_ptr() != b.data_ptr()
                or a.shape != b.shape
                or a.stride() != b.stride()
                or a.dtype != b.dtype
            ):
                raise ValueError(
                    "DFly query and target KV must alias the same local layout"
                )
        self._set_storage_aliases(False)

    def update_indices(
        self,
        active: torch.Tensor,
        source_rows: torch.Tensor,
        anchors: torch.Tensor,
        padded_batch: int,
        target_query_start_loc: torch.Tensor,
    ) -> None:
        # The target source row's snapshot row
        rows = source_snapshot_rows(
            source_rows,
            target_query_start_loc,
            padded_batch,
            self.snapshot_width,
            self.snapshots.shape[1],
        )
        build_dfly_reuse_indices(
            self.draft.model.topk_indices,
            self.snapshots,
            self.layer_snapshot_slots,
            active,
            rows,
            anchors,
            self.config.target_index_topk,
            self.draft.model.local_window,
            self.draft.model.query_width,
            padded_batch,
        )
