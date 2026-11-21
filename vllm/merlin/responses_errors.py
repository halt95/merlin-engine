# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Typed errors on a Responses stream.

A stream has already sent ``response.created`` when generation fails, so the
failure can only travel as an event. ``error_event`` builds the Responses API's
``error`` event from the same typed error every other endpoint answers with
(type, sanitized message), so a request the engine refused as a client fault is
a ``BadRequestError`` here too and a genuine internal failure stays the fixed
internal message. ``response_error`` is the matching ``error`` of a stored
response that ended in failure.
"""

from __future__ import annotations

from typing import Any

from openai.types.responses import ResponseError, ResponseErrorEvent

from vllm.entrypoints.serve.exception_handling.error_response import (
    create_error_response,
)

# The only client-fault type a stream's generation can raise mid-stream.
_CLIENT_ERROR_TYPES = ("BadRequestError",)


def error_event(exc: Exception) -> ResponseErrorEvent:
    """The ``error`` event for ``exc``; the caller sets ``sequence_number``."""
    info = create_error_response(exc).error
    return ResponseErrorEvent(
        type="error",
        code=info.type,
        message=info.message,
        param=info.param,
        sequence_number=-1,
    )


def response_error(event: Any) -> ResponseError:
    """The stored response's ``error`` for an ``error`` event."""
    code = (
        "invalid_prompt"
        if getattr(event, "code", None) in _CLIENT_ERROR_TYPES
        else "server_error"
    )
    return ResponseError(code=code, message=getattr(event, "message", None) or "")
