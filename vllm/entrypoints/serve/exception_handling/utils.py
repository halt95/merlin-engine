# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modified by the Merlin project; see NOTICE.

import regex as re


def sanitize_message(message: str) -> str:
    """Strip memory addresses, tracebacks, and file paths from error messages."""
    message = re.sub(r" at 0x[0-9a-fA-F]+>", ">", message)
    message = re.sub(r'\n?\s*File "[^"]+", line \d+, in \S+(\n\s+.*)?', "", message)
    message = re.sub(
        r"/(?:home|usr|opt|var|tmp|root|lib|mnt|srv)(?:/[\w.\-]+)+", "<path>", message
    )
    # Match each slash run once and decide in the replacement, so the scan stays linear.
    message = re.sub(
        r"(?:/[\w\-]+)+(?:\.\w+)?",
        lambda m: "<path>" if "." in m[0] and m[0].count("/") >= 2 else m[0],
        message,
    )
    return message.strip()
