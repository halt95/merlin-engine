# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modified by the Merlin project; see NOTICE.
"""Method ``dfly`` (DFly2) for ``SpeculativeConfig``.

#68296 adds ``spec_decode/dfly/config.py`` (``DFlySpeculatorConfig``) on the
0.34 speculator plugin API, keyed by the target's text model type. This tree
has no plugin API, so ``vllm/config/speculative.py`` calls ``configure_dfly``
and the method-level checks live in the target family's class
(``vllm/models/hy_v4/dfly_config.py``).
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.config import SpeculativeConfig
    from vllm.models.hy_v4.dfly_config import HYV4DFlySpeculatorConfig


def get_dfly_config_class(
    spec: "SpeculativeConfig",
) -> "type[HYV4DFlySpeculatorConfig]":
    from vllm.models.hy_v4.dfly_config import dfly_config_class

    if spec.target_model_config is None:
        raise ValueError("DFly requires target_model_config")
    return dfly_config_class(spec.target_model_config.hf_text_config.model_type)


def default_dfly_model(spec: "SpeculativeConfig") -> str:
    """DFly weights ship inside the target checkpoint."""
    target = spec.target_model_config
    if target is None:
        raise ValueError("DFly requires target_model_config")
    return target.model


def dfly_num_lookahead_tokens(num_speculative_tokens: int) -> int:
    """KV slots past the target's range: the bonus query plus one per draft token."""
    return num_speculative_tokens + 1
