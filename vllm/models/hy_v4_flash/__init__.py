# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HYV4 Flash entry point; config imports do not import GPU model code."""

from importlib import import_module
from typing import Any

__all__ = ["HYV4FlashForCausalLM"]


def __getattr__(name: str) -> Any:
    if name != "HYV4FlashForCausalLM":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    model = getattr(import_module(".nvidia.model", __name__), name)
    globals()[name] = model
    return model
