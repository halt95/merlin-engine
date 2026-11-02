# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Host-retained prompt scoring and a per-step GPU tensor budget."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import vllm.envs as envs
from vllm.exceptions import VLLMValidationError
from vllm.logger import init_logger

if TYPE_CHECKING:
    import torch

    from vllm.config import VllmConfig
    from vllm.sampling_params import SamplingParams
    from vllm.v1.outputs import LogprobsTensors


logger = init_logger(__name__)

LEGACY_SCORE_CHUNK_SIZE = 128
# A fitted chunk keeps top-20 requests (OpenAI's top_logprobs cap) admissible.
FIT_NUM_LOGPROBS = 20


@dataclass(frozen=True)
class PromptLogprobsBudget:
    vocab_size: int
    chunk_size: int
    logits_element_size: int
    max_num_batched_tokens: int
    tensor_parallel_size: int = 1
    tensor_budget_bytes: int = 384 * 1024**2
    use_v2_model_runner: bool = False
    logits_mode: bool = False
    rows: int = field(init=False)
    workspace: int = field(init=False)
    # ``chunk_size`` when its workspace fits; otherwise the largest row count
    # (>= 1) whose top-20 bound fits, so FP32 heads and huge vocabularies boot.
    effective_chunk_size: int = field(init=False)

    def _workspace(self, rows: int) -> int:
        alignment = 64 * self.tensor_parallel_size
        padded_vocab = (self.vocab_size + alignment - 1) // alignment * alignment
        # Projection can hold three TP local/gather/layout buffers.
        workspace = 3 * rows * padded_vocab * self.logits_element_size
        if not self.use_v2_model_runner:
            score_rows = min(rows, LEGACY_SCORE_CHUNK_SIZE)
            # BF16 log_softmax(dtype=float32) creates an FP32 input copy
            # and output while the full projection remains live. Charge the
            # same conservative conversion peak for other non-FP32 heads.
            score_bytes = 4
            if not self.logits_mode and self.logits_element_size != 4:
                score_bytes += 4
            scoring_workspace = padded_vocab * (
                rows * self.logits_element_size + score_rows * score_bytes
            )
            workspace = max(workspace, scoring_workspace)
        return workspace

    def _bound(self, rows: int, num_logprobs: int | None, fixed_width: int) -> int:
        temporary = rows * fixed_width * (8 + self.logits_element_size + 4)
        if num_logprobs is not None:
            k = self.vocab_size if num_logprobs == -1 else num_logprobs
            row_bytes = (k + 1) * 12 + 8
            temporary += 2 * self.max_num_batched_tokens * row_bytes
            temporary += rows * k * (max(self.logits_element_size, 4) + 8)
        # Reserve separate allowances for the maximum top-k and fixed-ID
        # widths, which can come from different concurrently scheduled requests.
        return self._workspace(rows) + 2 * temporary

    def __post_init__(self) -> None:
        rows = min(self.chunk_size, self.max_num_batched_tokens)
        effective = self.chunk_size
        if self._workspace(rows) > self.tensor_budget_bytes:
            # Monotonic in rows. If one row does not fit, validate() refuses
            # scoring per request; startup never fails.
            lo, hi = 1, rows
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if self._bound(mid, FIT_NUM_LOGPROBS, 0) <= self.tensor_budget_bytes:
                    lo = mid
                else:
                    hi = mid - 1
            rows = effective = lo
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "effective_chunk_size", effective)
        object.__setattr__(self, "workspace", self._workspace(rows))

    @property
    def chunk_reduced(self) -> bool:
        return self.effective_chunk_size != self.chunk_size

    def memory_bound(self, num_logprobs: int | None, fixed_width: int = 0) -> int:
        """Bound a whole scheduler step; retained results reside on the host.

        Workspace covers TP projection buffers and the legacy conversion/
        softmax peak separately. Count both sides of batch output concatenation
        and top-k temporaries. Fixed-ID requests transfer one chunk at a time.
        Backend scratch and allocator fragmentation require a measurement on the target GPU.
        """
        return self._bound(self.rows, num_logprobs, fixed_width)

    def validate(self, params: SamplingParams) -> None:
        fixed_ids = params.prompt_logprob_token_ids
        bound = self.memory_bound(
            params.prompt_logprobs,
            fixed_ids.shape[1] if fixed_ids is not None else 0,
        )
        if bound > self.tensor_budget_bytes:
            parameter = (
                "prompt_logprobs"
                if params.prompt_logprobs is not None
                else "prompt_logprob_token_ids"
            )
            raise VLLMValidationError(
                f"{parameter} per-step prompt-scoring tensor bound is "
                f"{bound} bytes, exceeding the {self.tensor_budget_bytes}-byte "
                f"({self.tensor_budget_bytes / 1024**2:g} MiB) limit. "
                "Reduce the requested score count or "
                "VLLM_MERLIN_PROMPT_LOGPROBS_CHUNK_SIZE, or calibrate "
                "VLLM_MERLIN_PROMPT_LOGPROBS_TENSOR_BUDGET_MIB. "
                "Prompt length and the KV pin do not need to change.",
                parameter=parameter,
                value=bound,
            )


def prompt_logprobs_budget_from_config(
    config: VllmConfig,
) -> PromptLogprobsBudget | None:
    if config.model_config.runner_type != "generate":
        return None
    model = config.model_config
    budget = PromptLogprobsBudget(
        vocab_size=model.get_vocab_size(),
        chunk_size=envs.VLLM_MERLIN_PROMPT_LOGPROBS_CHUNK_SIZE,
        logits_element_size=(model.head_dtype or model.dtype).itemsize,
        max_num_batched_tokens=config.scheduler_config.max_num_batched_tokens,
        tensor_parallel_size=config.parallel_config.tensor_parallel_size,
        use_v2_model_runner=config.use_v2_model_runner,
        logits_mode=model.logprobs_mode in ("raw_logits", "processed_logits"),
        tensor_budget_bytes=(
            envs.VLLM_MERLIN_PROMPT_LOGPROBS_TENSOR_BUDGET_MIB * 1024**2
        ),
    )
    if budget.chunk_reduced:
        logger.warning_once(
            "Prompt-scoring chunk reduced from %d to %d rows so the %d-byte "
            "workspace fits the %d-byte tensor budget (vocab %d, %d-byte head).",
            budget.chunk_size,
            budget.effective_chunk_size,
            budget.workspace,
            budget.tensor_budget_bytes,
            budget.vocab_size,
            budget.logits_element_size,
        )
    return budget


def prompt_logprobs_chunk_size(config: VllmConfig) -> int:
    """Projection rows per scoring chunk; the workers and admission share it."""
    budget = prompt_logprobs_budget_from_config(config)
    if budget is None:
        return envs.VLLM_MERLIN_PROMPT_LOGPROBS_CHUNK_SIZE
    return budget.effective_chunk_size


def prompt_logprobs_to_host(
    logprobs: LogprobsTensors, *, pin_memory: bool = False
) -> LogprobsTensors:
    """Start compact, non-blocking host copies of one request's slice.

    Host values are valid only after the step's AsyncOutput copy event. The
    device sources may be freed at once: the caching allocator reuses them only
    behind the copy on the same stream.
    """
    import torch

    from vllm.v1.outputs import LogprobsTensors

    def copy(tensor: torch.Tensor) -> torch.Tensor:
        out = torch.empty(
            tensor.shape,
            dtype=tensor.dtype,
            pin_memory=pin_memory and tensor.device.type == "cuda",
        )
        return out.copy_(tensor.contiguous(), non_blocking=True)

    return LogprobsTensors(
        copy(logprobs.logprob_token_ids),
        copy(logprobs.logprobs),
        copy(logprobs.selected_token_ranks),
    )


class PendingPromptLogprobs:
    """Host partials of a finished prompt whose copies may still be in flight."""

    __slots__ = ("partials",)

    def __init__(self, partials: list[LogprobsTensors]) -> None:
        self.partials = tuple(partials)

    def to_cpu_nonblocking(self) -> PendingPromptLogprobs:
        return self

    def resolve(self) -> LogprobsTensors:
        from vllm.v1.outputs import LogprobsTensors

        return LogprobsTensors.cat(self.partials)


def resolve_prompt_logprobs(
    prompt_logprobs: dict[str, LogprobsTensors | PendingPromptLogprobs | None],
) -> dict[str, LogprobsTensors | None]:
    """Concatenate pending partials; call only after the output copy event."""
    return {
        req_id: value.resolve() if isinstance(value, PendingPromptLogprobs) else value
        for req_id, value in prompt_logprobs.items()
    }


def score_legacy_prompt_chunks(
    runner,
    hidden_states: torch.Tensor,
    req_id: str,
    num_logits: int,
) -> None:
    """Copy chunked scores into the legacy runner's existing host buffers."""
    import torch

    request = runner.requests[req_id]
    output = request.in_progress_prompt_logprobs_cpu
    start_idx = request.num_computed_tokens
    req_idx = runner.input_batch.req_id_to_index[req_id]
    offset = runner.query_start_loc.np[req_idx].item()
    num_logprobs = runner.num_prompt_logprobs[req_id]
    for start in range(0, num_logits, runner.prompt_logprobs_chunk_size):
        end = min(start + runner.prompt_logprobs_chunk_size, num_logits)
        logits = runner.model.compute_logits(
            hidden_states[offset + start : offset + end]
        )
        # Keep the projection cap independent of legacy FP32 scoring storage.
        for score_start in range(0, end - start, LEGACY_SCORE_CHUNK_SIZE):
            score_end = min(score_start + LEGACY_SCORE_CHUNK_SIZE, end - start)
            a, b = start + score_start, start + score_end
            chunk_logits = logits[score_start:score_end]
            targets = torch.tensor(
                request.prompt_token_ids[start_idx + 1 + a : start_idx + 1 + b],
                dtype=torch.int64,
                device=runner.device,
            )
            if runner.model_config.logprobs_mode in ("raw_logits", "processed_logits"):
                scores = chunk_logits.to(torch.float32)
            else:
                scores = runner.sampler.compute_logprobs(chunk_logits)
            token_ids, logprobs, ranks, *_ = runner.sampler.gather_logprobs(
                scores, num_logprobs, targets
            )
            chunk_slice = slice(start_idx + a, start_idx + b)
            # Complete each host copy before allocating the next score chunk.
            output.logprob_token_ids[chunk_slice].copy_(token_ids)
            output.logprobs[chunk_slice].copy_(logprobs)
            output.selected_token_ranks[chunk_slice].copy_(ranks)
            del chunk_logits, scores, token_ids, logprobs, ranks, targets
        del logits
