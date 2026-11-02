# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Model-independent RecoverSSM Python helpers: the activation predicate shared by
layers, models and backends, checkpoint storage identity, and argument validation
that raises ``ValueError`` before any raw-pointer kernel launch."""

from typing import Any


def recoverssm_binding_signature(checkpoints: Any) -> tuple:
    """Storage identity of every layer's checkpoint: (data_ptr, shape, stride).

    The context's address table (base addresses, block strides) is derived from
    exactly these values. Tensor identity alone does not pin them: ``set_()``
    (or an in-place resize) moves a tensor's storage while keeping the object."""
    return tuple((t.data_ptr(), t.shape, t.stride()) for t in checkpoints)


def recoverssm_require(cond: bool, msg: str, component: str = "RecoverSSM") -> None:
    if not cond:
        raise ValueError(f"{component}: {msg}")


def uses_recoverssm(cache_config, num_speculative_tokens: int) -> bool:
    """True when this model instance runs RecoverSSM for its recurrent states.

    VllmConfig sets ``cache_config.use_recoverssm`` for every model whose recurrent
    states support it (Kimi-K3 KDA, Qwen GDN/PLE). Without speculative tokens there is
    nothing to recover, so the stock path runs."""
    return (
        bool(getattr(cache_config, "use_recoverssm", False))
        and num_speculative_tokens > 0
    )


__all__ = ["recoverssm_binding_signature", "recoverssm_require", "uses_recoverssm"]
