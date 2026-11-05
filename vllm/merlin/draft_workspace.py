# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""One FlashInfer float workspace for the MTP drafter and its target.

Opt-in with MERLIN_SHARE_DRAFT_WORKSPACE=1. The drafter and the target issue attention
serially on one compute stream, and the float buffer is scratch that each run rewrites
before reading, so one buffer serves both. With ``shared=None``,
``share_workspace`` only shares one workspace across a model's own builders.
"""

import logging

import torch

logger = logging.getLogger(__name__)


def share_workspace(
    builders, workspace: torch.Tensor | None, shared: torch.Tensor | None
) -> torch.Tensor | None:
    """Give ``builders`` one float workspace; returns the buffer the next ones share.

    The first builder that owns a workspace allocates it when none is given. A buffer
    handed in from another model (``shared``) is used only by builders it is large
    enough for; a builder it is too small for keeps its own, and later builders share
    that one.
    """
    for builder in builders:
        if workspace is None:
            if hasattr(builder, "_get_workspace_buffer"):
                workspace = builder._get_workspace_buffer()
        elif hasattr(builder, "set_workspace_buffer"):
            # A builder that cannot state its size (e.g. a composite) keeps its own.
            if workspace is shared and hasattr(builder, "_get_workspace_buffer"):
                need = (
                    builder._workspace_bytes()
                    if hasattr(builder, "_workspace_bytes")
                    else None
                )
                if need is None or workspace.numel() < need:
                    logger.info(
                        "draft workspace share refused for %s: shared buffer %d B, "
                        "needs %s; it keeps its own",
                        type(builder).__name__,
                        workspace.numel(),
                        "an unknown size" if need is None else f"{need} B",
                    )
                    workspace = builder._get_workspace_buffer()
                    continue
            builder.set_workspace_buffer(workspace)
    return workspace


def attn_workspace(attn_groups) -> torch.Tensor | None:
    """The float workspace a model's builders share (None if no builder owns one)."""
    for groups in attn_groups:
        for group in groups:
            for builder in group.metadata_builders:
                workspace = getattr(builder, "_workspace_buffer", None)
                if workspace is not None:
                    return workspace
    return None


def _enabled() -> bool:
    from vllm import envs

    return envs.MERLIN_SHARE_DRAFT_WORKSPACE


def target_workspace(target_attn_groups) -> torch.Tensor | None:
    """The buffer the drafter's builders start from: the target's with the knob on."""
    return attn_workspace(target_attn_groups) if _enabled() else None


def log_share(target_attn_groups, draft_attn_groups) -> None:
    """With the knob on, log whether the drafter ended up on the target's buffer."""
    if not _enabled():
        return
    target = attn_workspace(target_attn_groups)
    draft = attn_workspace(draft_attn_groups)
    shared = (
        target is not None
        and draft is not None
        and draft.data_ptr() == target.data_ptr()
    )
    logger.info(
        "MERLIN_SHARE_DRAFT_WORKSPACE: drafter %s the target FlashInfer workspace "
        "(%d B, target %#x, draft %#x)",
        "shares" if shared else "does NOT share",
        target.numel() if target is not None else 0,
        target.data_ptr() if target is not None else 0,
        draft.data_ptr() if draft is not None else 0,
    )
