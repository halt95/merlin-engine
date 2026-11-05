# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Explicit transient generation causes; unknown failures stay 500."""

from vllm.outputs import RequestError
from vllm.v1.engine import EngineCoreOutput, FinishReason

_RETRYABLE_MESSAGES = {
    "kv_load_failure": "KV cache transfer failed; please retry.",
    "encoder_input_unavailable": "Encoder input unavailable; please retry.",
    "multimodal_cache_miss": "Multi-modal processor cache miss; please retry.",
}


def get_retryable_generation_error(output: EngineCoreOutput) -> RequestError | None:
    if output.finish_reason != FinishReason.ERROR:
        return None
    # The frontend invalidates these hashes before yielding to the generate task,
    # so a retry resends the data instead of repeating the stale cache reference.
    code = (
        "multimodal_cache_miss"
        if output.mm_cache_miss_hashes
        else output.retryable_error_code
    )
    if code is None:
        return None
    message = _RETRYABLE_MESSAGES.get(code)
    if message is None:
        return None
    return RequestError(code=code, message=message, retryable=True)
