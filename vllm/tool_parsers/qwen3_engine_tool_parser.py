# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modified by the Merlin project; see NOTICE.

from typing import TYPE_CHECKING

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.parser.engine.registered_adapters import Qwen3ParserToolAdapter
from vllm.tool_parsers.tool_strict_level import (
    ToolStrictLevel,
)

if TYPE_CHECKING:
    from xgrammar import StructuralTag

    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest

logger = init_logger(__name__)


class Qwen3EngineToolParser(Qwen3ParserToolAdapter):  # type: ignore[valid-type, misc]
    structural_tag_model = "qwen_3_coder"
    # A forced, required or strict call is
    # opened and closed by the two special tokens, so the grammar cannot let the
    # model spell the opener in ordinary tokens (the parser reads that as text and
    # the call came back as raw markup). A tokenizer whose four tokens cannot be named
    # unambiguously keeps the builtin text form, with one warning.

    def get_structural_tag(
        self,
        request: "ChatCompletionRequest | ResponsesRequest",
        *,
        reasoning: bool = False,
        strict_level: ToolStrictLevel = ToolStrictLevel.AUTO,
    ) -> "StructuralTag | None":
        if not envs.VLLM_ENFORCE_STRICT_TOOL_CALLING:
            return None
        from vllm.tool_parsers.structural_tag_registry import (
            QWEN_3_CODER_TOKEN_MODEL,
            get_model_structural_tag,
            qwen_3_coder_token_ids,
            resolve_structural_tag_tokens,
        )

        model = self.structural_tag_model
        ids, problems = qwen_3_coder_token_ids(self.model_tokenizer)
        if problems:
            logger.warning_once(
                "The tool-call grammar keeps the text form of the call delimiters: %s.",
                "; ".join(problems),
            )
        else:
            model = QWEN_3_CODER_TOKEN_MODEL
        tag = get_model_structural_tag(
            model=model,
            tools=request.tools,
            tool_choice=request.tool_choice,
            reasoning=reasoning,
            strict_level=strict_level,
        )
        # The request is validated without a tokenizer: carry ids, not names.
        if tag is not None and model == QWEN_3_CODER_TOKEN_MODEL:
            resolve_structural_tag_tokens(tag, ids)
        return tag
