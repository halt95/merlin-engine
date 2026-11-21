# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modified by the Merlin project; see NOTICE.
"""Adapted from #68296 hy_v4/nvidia/dfly.py:47-74."""

from copy import deepcopy

from torch import nn

from vllm.merlin.hy4_quant import validate_checkpoint_quantization
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
    CompressedTensorsConfig,
)


class HYV4DFlyQuantizationConfig(CompressedTensorsConfig):
    @classmethod
    def for_model(cls, checkpoint, model_prefix, *, module_names=None):
        # A draft-only selection (draft_quantization_config): no expert group.
        validate_checkpoint_quantization(
            checkpoint, module_names=module_names, require_routed_experts=False
        )
        config = cls.from_config(deepcopy(checkpoint))
        config.model_prefix = f"{model_prefix}."
        config.packed_modules_mapping = {
            "gate_up_proj": ["gate_proj", "up_proj"],
        }
        return config

    def get_scheme_dict(self, layer, layer_name=None):
        if layer_name is not None:
            if not layer_name.startswith(self.model_prefix):
                raise ValueError(f"DFly quantization outside draft: {layer_name}")
            local = layer_name.removeprefix(self.model_prefix)
            local = local.removeprefix("layers.")
            layer_name = f"model.dfly_layers.{local}"
        return super().get_scheme_dict(layer, layer_name)

    def weight_bits(self, layer_name):
        # Construct a real, allocation-free linear for CT's target matcher.
        layer = nn.Linear(1, 1, bias=False, device="meta")
        scheme = self.get_scheme_dict(layer, layer_name)
        if scheme is None or scheme["weights"].type != "int":
            return None
        return scheme["weights"].num_bits
