# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modified by the Merlin project; see NOTICE.
"""Adapted from #68296 nvidia/moe.py:1-87."""

from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.models.hy_v4.nvidia.moe import HYV4FeedForward, HYV4MoEFused


class HYV4FlashFeedForward(HYV4FeedForward):
    def __init__(self, config, quant_config, prefix):
        super().__init__(
            config.hidden_size,
            config.intermediate_size,
            config.hidden_act,
            quant_config=quant_config,
            prefix=prefix,
        )
        self.post_fc2_norm = (
            RMSNorm(config.hidden_size, config.rms_norm_eps)
            if config.sandwich_norm
            else None
        )

    def forward(self, hidden, skip_norm=False):
        out = super().forward(hidden)
        if skip_norm:  # the fused iHC post applies post_fc2_norm
            return out
        return self.post_fc2_norm(out) if self.post_fc2_norm is not None else out


class HYV4FlashMoE(HYV4MoEFused):
    def __init__(self, config, quant_config, prefix):
        latent = config.routed_expert_hidden_size
        down = up = None
        if latent is not None:
            down = ReplicatedLinear(
                config.hidden_size,
                latent,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.routed_expert_down_proj",
            )
            up = ReplicatedLinear(
                latent,
                config.hidden_size,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.routed_expert_up_proj",
            )
        super().__init__(
            config,
            quant_config=quant_config,
            prefix=prefix,
            routed_hidden_size=latent,
            routed_input_transform=down,
            routed_output_transform=up,
        )
        mapping = self.experts.routed_experts.expert_map_manager
        self.ep_size = mapping.ep_size
        self.ep_rank = mapping.ep_rank
        self.n_physical_experts = mapping.global_num_experts
        self.n_local_physical_experts = mapping.local_num_experts
        local_ids = (
            list(range(mapping.global_num_experts))
            if mapping.expert_map is None
            else (mapping.expert_map >= 0).nonzero().flatten().tolist()
        )
        self.physical_expert_start = min(local_ids, default=0)
        self.physical_expert_end = max(local_ids, default=-1) + 1
        self.combine_output_norm = (
            RMSNorm(config.hidden_size, config.rms_norm_eps)
            if config.sandwich_norm
            else None
        )

    def forward(self, hidden, skip_norm=False):
        out = super().forward(hidden)
        if skip_norm:  # the fused iHC post applies combine_output_norm
            return out
        return (
            self.combine_output_norm(out)
            if self.combine_output_norm is not None
            else out
        )
