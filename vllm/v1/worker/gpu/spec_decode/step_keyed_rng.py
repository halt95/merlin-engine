# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Step-keyed random draws for block verification.

The defect
----------
MRV2 keys its spec-decode draws by request seed and absolute position:
the acceptance uniform is ``tl_rand32(seed, pos)`` and the draft Gumbel noise
is ``murmur3(seed, pos + _DRAFT_NOISE_SALT, token)``. Standard verification
stops at the first rejection, so every acceptance draw it reads decides once
and the draft noise past the rejection never reaches the output. Block
verification (Sun et al. 2024, ``rejection_sample_method="block"``) reads the
draw at EVERY drafted row. After a partial accept the next round re-verifies the
positions past the resampled token with the same uniforms (already known to
have exceeded their thresholds) and re-drafts them with the same noise, so the
rounds are not independent and the output distribution is biased (CPU joint
chi-square 1,828 / 8,160 against a 488 threshold; upstream vLLM #61114).

The fix
-------
With ``VLLM_SPEC_STEP_KEYED_RNG=1`` and block verification, the two draws a
later round could reuse are keyed by the ROUND:

* acceptance uniform of row ``i``: Philox4x32-10, key = request seed,
  counter = (anchor, i, _STEP_KEY_ACCEPT_PURPOSE, 0);
* draft Gumbel noise of draft step ``j`` for token ``t``: murmur3 over
  (_STEP_KEY_DRAFT_DOMAIN, seed, anchor, j, t).

``anchor`` is the key position of the round's first row (``pos[start]`` in the
rejection kernel, ``sample_src_positions - draft_step`` in the drafter; both are
the position before the first draft token). Every round commits at least one
token, so a request's anchors strictly increase and no (anchor, offset) pair is
ever drawn by two rounds. The keys depend only on the request seed, the anchor
and the offset, never on the batch row, the request-state slot, launch order or
a host RNG, so the draws are invariant to batch composition, preemption/resume
(seeded requests) and CUDA-graph replay: the anchor comes from the positions
buffer and the draft step from the speculator's persistent ``current_draft_step``
tensor, both device buffers the captured kernels read at replay.

What stays as it was
--------------------
* Greedy (T=0): no noise is drawn and the greedy branch never reads ``u``.
* Standard and synthetic verification: their keying is unchanged bit for bit;
  the flag only takes effect together with block verification.
* Resample / bonus noise stays position-keyed: each committed position is
  resampled exactly once, in the round that commits it.
* Seed semantics: seeded requests stay reproducible run to run. Unseeded
  requests keep the stock seed (drawn at admission, redrawn when a preempted
  request is re-added), which keeps the output exact but not reproducible
  across preemption, exactly as upstream.

Deviation from upstream: under block verification a seeded request produces a
different (still deterministic) token sequence with the flag on than upstream's
position-keyed block mode, which is itself not distribution-exact.

Flag: ``VLLM_SPEC_STEP_KEYED_RNG`` -- strict "0"/"1", default OFF, read once at
init (rejection sampler and speculator). OFF launches every kernel with the
stock constexprs, so all draws are byte-identical to the unflagged engine.
"""

from __future__ import annotations

from vllm import envs
from vllm.logger import init_logger

logger = init_logger(__name__)

FLAG_NAME = "VLLM_SPEC_STEP_KEYED_RNG"


def step_keyed_rng_requested() -> bool:
    return bool(envs.VLLM_SPEC_STEP_KEYED_RNG)


def resolve_step_keyed_rng(rejection_sample_method: str) -> bool:
    """Return whether step-keyed draws are in effect for this engine.

    Effective only for block verification. Requested with any other method it
    is a logged no-op: the other methods are already exact under their keying,
    which must stay bit-identical.
    """
    warn_block_without_step_keyed_rng(rejection_sample_method)
    if not step_keyed_rng_requested():
        return False
    if rejection_sample_method != "block":
        logger.warning_once(
            "%s=1 has no effect with rejection_sample_method=%r: only block "
            "verification re-reads draws across rounds.",
            FLAG_NAME,
            rejection_sample_method,
        )
        return False
    return True


_block_unset_warned = False


def warn_block_without_step_keyed_rng(rejection_sample_method: str) -> None:
    """Log one WARNING per process for block verification without the flag.

    Block verification re-reads the position-keyed draws across rounds, so its
    output distribution is biased without ``VLLM_SPEC_STEP_KEYED_RNG=1``. Called
    from ``resolve_step_keyed_rng``, which every block-verification init reaches.
    Only logs: behaviour is unchanged and nothing is raised.
    """
    global _block_unset_warned
    if rejection_sample_method != "block" or step_keyed_rng_requested():
        return
    if _block_unset_warned:
        return
    _block_unset_warned = True
    logger.warning(
        "rejection_sample_method='block' without %s=1: block verification "
        "is biased (draws are re-read across rounds). Set %s=1 for an "
        "unbiased output distribution.",
        FLAG_NAME,
        FLAG_NAME,
    )


def check_step_keyed_rng_rejection_support(*, watermark: bool) -> None:
    """Refuse (raise) target-side combinations that are not validated while ON."""
    if watermark:
        raise ValueError(
            f"{FLAG_NAME}=1 is not compatible with watermarking: the watermark "
            "PRF keys its draws by context, not by round."
        )


def check_step_keyed_rng_draft_support(
    *,
    speculator_name: str,
    supported: bool,
    has_watermarker: bool,
    probabilistic_draft: bool,
) -> None:
    """Refuse (raise) a drafter whose noise would stay position-keyed.

    Called at speculator init only when the flag is in effect (block
    verification). A drafter that samples its drafts by argmax draws no noise,
    so the rejection-side keying alone makes it exact. A probabilistic drafter
    must pass the persistent draft-step buffer to the sampler; only classes
    validated for that (supports_step_keyed_rng) are accepted. Raising rather
    than warning: a lever that silently leaves block mode biased is worse than
    a boot error.
    """
    if not probabilistic_draft:
        return
    if not supported:
        raise ValueError(
            f"{FLAG_NAME}=1 is not supported by {speculator_name} with "
            "probabilistic drafting: only speculators whose draft sampling "
            "passes the persistent draft-step buffer (supports_step_keyed_rng) "
            "are audited."
        )
    if has_watermarker:
        raise ValueError(f"{FLAG_NAME}=1 is not compatible with draft watermarking.")
