# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
# Merlin-owned sources of the _C_stable_libtorch extension, included right after the initial
# set(VLLM_STABLE_EXT_SRC ...) so set_gencode_flags_for_srcs() gives them the same arch flags.
# Upstream csrc/topk.cu stays compiled and owns _C::persistent_topk; the Merlin top-k lives in
# namespace merlin and is reached only through _C::merlin_persistent_topk. The sampled kernels
# get their own TU because sharing one changed nvcc's SASS for the vendored kernels.
set(MERLIN_UPSTREAM_TOPK_SRC "csrc/topk.cu")
list(FIND VLLM_STABLE_EXT_SRC "${MERLIN_UPSTREAM_TOPK_SRC}" MERLIN_TOPK_INDEX)
if(MERLIN_TOPK_INDEX EQUAL -1)
  message(FATAL_ERROR
    "merlin.cmake: ${MERLIN_UPSTREAM_TOPK_SRC} is not in VLLM_STABLE_EXT_SRC; "
    "upstream moved or renamed its top-k source. Check that _C::persistent_topk "
    "still has an upstream definition before building the Merlin top-k beside "
    "it.")
endif()
list(APPEND VLLM_STABLE_EXT_SRC
  "csrc/merlin/topk/topk.cu"
  "csrc/merlin/topk/bindings.cpp")
# sampled_topk.cu is CUDA-only (PTX asm, __trap, 64-bit warp masks); topk.cu
# already guards its include of the launch interface with #ifndef USE_ROCM.
if(VLLM_GPU_LANG STREQUAL "CUDA")
  list(APPEND VLLM_STABLE_EXT_SRC "csrc/merlin/topk/sampled_topk.cu")
endif()
