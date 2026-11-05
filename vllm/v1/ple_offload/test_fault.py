# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Hook slot for the test-only PLE fault injector.

The injector package ``vllm_ple_test_fault`` belongs to the test suite and is
not shipped. Every injection site in the product reads ``_FAULT`` from here and
short-circuits on ``_FAULT is None``.
``_FAULT`` stays None unless ``VLLM_PLE_TEST_FAULT`` is set, and only then is
the injector imported (a test harness that sets it puts the package on
``PYTHONPATH``). The PLE offload bootstrap applies the same gate.
"""

import os

_FAULT = None
_FaultReconstructionPayload = None
# Unset, empty and "0" mean off; any other value arms the injector.
if os.environ.get("VLLM_PLE_TEST_FAULT", "0").strip() not in ("", "0"):
    try:
        from vllm_ple_test_fault import (  # type: ignore[no-redef]
            _FAULT as _FAULT,
        )
        from vllm_ple_test_fault import (  # type: ignore[no-redef]
            _FaultReconstructionPayload as _FaultReconstructionPayload,
        )
    except ImportError as exc:
        raise RuntimeError(
            "VLLM_PLE_TEST_FAULT is set but the test-only injector package "
            "'vllm_ple_test_fault' is not importable (put tests/v1/ple_offload "
            "on PYTHONPATH, or unset VLLM_PLE_TEST_FAULT / set it to 0)"
        ) from exc
