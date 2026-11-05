# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Sampler admission gate, as defence in depth.

The Model Runner V2 sampler enforces per-request capacity limits inside the
worker's ``add_request`` (``bad_words.py``, ``logit_bias.py``), which runs
inside ``execute_model``. A ``ValueError`` there fails the whole step and
EngineCore takes its fatal path, so one request kills the lane.

The frontend rejects these requests with a 400
(``SamplingParams._validate_sampler_caps``). This module mirrors the worker's
own admission checks, with the worker's own limits, so EngineCore can refuse an
inadmissible request on its per-request error path
(``VLLMValidationError`` -> ``FinishReason.ERROR``) before the worker ever
sees it, whatever entry point produced it (OpenAI server, offline ``LLM``,
another frontend, or a frontend whose check drifted). The worker's own raises
are unchanged; an admissible request is untouched.
"""

from typing import TYPE_CHECKING

import vllm.envs as envs

if TYPE_CHECKING:
    from vllm.sampling_params import SamplingParams


def sampler_admission_error(
    sampling_params: "SamplingParams",
) -> tuple[str, str, int] | None:
    """The first worker ``add_request`` limit this request breaks, as
    ``(parameter, message, count)``, or None if the V2 sampler would accept it.

    Same predicates, in the same order, as ``BadWordsState.add_request`` and
    ``LogitBiasState.add_request``.
    """
    # Lazy: the worker module imports torch/triton.
    from vllm.v1.worker.gpu.sample.logit_bias import (
        MAX_NUM_ALLOWED_TOKEN_IDS,
        MAX_NUM_LOGIT_BIAS_TOKENS,
        MAX_NUM_STOP_TOKEN_IDS,
    )

    bad_words_token_ids = sampling_params.bad_words_token_ids
    if bad_words_token_ids:
        num_bad_words = len(bad_words_token_ids)
        max_num_bad_words = envs.VLLM_MAX_NUM_BAD_WORDS
        if num_bad_words > max_num_bad_words:
            return (
                "bad_words",
                f"Too many bad words: {num_bad_words}. "
                f"The max number is {max_num_bad_words}.",
                num_bad_words,
            )
        total = sum(len(word) for word in bad_words_token_ids)
        max_total_tokens = envs.VLLM_MAX_BAD_WORDS_TOTAL_TOKENS
        if total > max_total_tokens:
            return (
                "bad_words",
                f"Too many total bad word tokens: {total}. "
                f"The max is {max_total_tokens}.",
                total,
            )

    allowed_token_ids = sampling_params.allowed_token_ids
    if allowed_token_ids and len(allowed_token_ids) > MAX_NUM_ALLOWED_TOKEN_IDS:
        count = len(allowed_token_ids)
        return (
            "allowed_token_ids",
            f"Too many allowed token IDs: {count}. "
            f"The max size is {MAX_NUM_ALLOWED_TOKEN_IDS}.",
            count,
        )

    logit_bias = sampling_params.logit_bias
    if logit_bias and len(logit_bias) > MAX_NUM_LOGIT_BIAS_TOKENS:
        count = len(logit_bias)
        return (
            "logit_bias",
            f"Too many logit bias tokens: {count}. "
            f"The max size is {MAX_NUM_LOGIT_BIAS_TOKENS}.",
            count,
        )

    stop_token_ids = sampling_params.all_stop_token_ids
    if (
        sampling_params.min_tokens > 0
        and stop_token_ids
        and len(stop_token_ids) > MAX_NUM_STOP_TOKEN_IDS
    ):
        count = len(stop_token_ids)
        return (
            "stop_token_ids",
            f"Too many stop tokens: {count}. "
            f"The max size is {MAX_NUM_STOP_TOKEN_IDS}.",
            count,
        )
    return None


def check_engine_admission(
    use_v2_model_runner: bool, sampling_params: "SamplingParams | None"
) -> None:
    """Raise ``VLLMValidationError`` for a request the V2 worker would refuse.

    Only the V2 model runner has these limits; V1 and pooling requests pass.
    """
    if sampling_params is None or not use_v2_model_runner:
        return
    error = sampler_admission_error(sampling_params)
    if error is None:
        return
    from vllm.exceptions import VLLMValidationError

    parameter, message, count = error
    raise VLLMValidationError(
        f"{message} (Model Runner V2 sampler limit; request refused before "
        "the worker)",
        parameter=parameter,
        value=count,
    )
