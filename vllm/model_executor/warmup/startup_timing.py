# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)

import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)


@contextmanager
def startup_warmup_phase(
    config: "VllmConfig", phase: str, detail: str = ""
) -> Iterator[None]:
    """Log host elapsed time around an existing startup phase, without syncing."""
    if not config.kernel_config.log_startup_warmup:
        yield
        return
    started = time.perf_counter()
    logger.info("Startup warmup %s begin %s", phase, detail)
    try:
        yield
    finally:
        logger.info(
            "Startup warmup %s end host_elapsed=%.3fs %s",
            phase,
            time.perf_counter() - started,
            detail,
        )
