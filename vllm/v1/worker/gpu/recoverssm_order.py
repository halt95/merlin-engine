# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Eligibility of VLLM_RECOVERSSM_COMMIT_AFTER_PROPOSE (the RecoverSSM GDN
commit enqueued after the MTP proposal). Kept free of the runner so it can be
tested without the CUDA stack."""

from typing import Any

from vllm.v1.kv_cache_interface import MambaSpec


def commit_after_propose_eligibility(
    speculator: Any, model_state: Any, pp_handler: Any, pcp_manager: Any
) -> tuple[bool, str]:
    """(eligible, reason). Eligible only when the proposal provably cannot read
    what the deferred tail writes or write what it reads:

    - a RecoverSSM model state that supports the deferral;
    - no PP, or the LAST PP rank (non-last ranks defer their commits to the
      PP broadcast pp_size steps later; the last rank's commit is an ordinary,
      non-deferred commit, so the proof below holds there unchanged);
    - no PCP (the proposal re-gathers the batch-order block tables, a commit
      input);
    - a model-based drafter with an INITIALIZED attention-group inventory
      (``set_attn`` ran) in which no group holds Mamba/GDN state. A missing,
      empty or differently shaped inventory is not evidence of safety."""
    if speculator is None:
        return False, "no speculator"
    # Only the non-last PP ranks are refused.
    if pp_handler is not None and not getattr(pp_handler, "is_last_rank", False):
        return False, "pipeline parallelism (non-last rank)"
    if pcp_manager is not None:
        return False, "prefill context parallelism"
    if getattr(model_state, "recoverssm", None) is None:
        return False, "RecoverSSM is off"
    if not getattr(model_state, "supports_deferred_postprocess", False):
        return False, "model state cannot defer its postprocess"
    groups = getattr(speculator, "attn_groups", None)
    if not isinstance(groups, list) or not groups:
        return False, "drafter attention groups not initialized"
    seen = 0
    for group_list in groups:
        if not isinstance(group_list, list):
            return False, "unrecognized drafter attention-group inventory"
        for group in group_list:
            seen += 1
            spec = getattr(group, "kv_cache_spec", None)
            if spec is None:
                return False, "drafter attention group without a KV cache spec"
            if isinstance(spec, MambaSpec):
                return False, "drafter holds Mamba/GDN state"
    if not seen:
        return False, "drafter attention groups not initialized"
    return True, "active"


class CommitAfterProposeGate:
    """Caches the eligibility verdict per (drafter attention-group inventory,
    PP handler, PCP manager) identity: a drafter ``set_attn`` re-run (KV cache
    re-initialization) or a changed parallel setup re-evaluates it."""

    def __init__(self, logger: Any) -> None:
        self._logger = logger
        self._key: tuple[Any, ...] | None = None
        self._ok = False

    def eligible(
        self, speculator: Any, model_state: Any, pp_handler: Any, pcp_manager: Any
    ) -> bool:
        key = (
            None if speculator is None else getattr(speculator, "attn_groups", None),
            model_state,
            pp_handler,
            pcp_manager,
        )
        if self._key is not None and all(a is b for a, b in zip(self._key, key)):
            return self._ok
        ok, reason = commit_after_propose_eligibility(
            speculator, model_state, pp_handler, pcp_manager
        )
        self._key, self._ok = key, ok
        if ok:
            self._logger.info(
                "RecoverSSM commit after propose: active (GDN commit and align "
                "postprocess enqueued after the MTP proposal, same stream)"
            )
        else:
            self._logger.warning(
                "VLLM_RECOVERSSM_COMMIT_AFTER_PROPOSE=1 ignored: %s", reason
            )
        return ok
