# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Merlin (fork) environment registry, moved out of vllm/envs.py.

Declarations (TYPE_CHECKING), the fork parsers, the fork getters
(MERLIN_ENV), the fork's runtime-only names (MERLIN_IGNORED) and the
derived compile factors. vllm/envs.py merges them with three hook lines:
``environment_variables.update(MERLIN_ENV)``, ``ignored_factors |=
MERLIN_IGNORED`` and ``add_merlin_compile_factors(factors)``. The Python 0/1
switches share one parser (``_e2_strict_switch``) and the determinism toggles
default to ``SHIPPED_KERNEL_TOGGLE_DEFAULTS``. Imports nothing but the
standard library at module level, so vllm.envs can import it first.
"""

import math
import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # API-server request limits only. AOT: no. CAPTURE: no.
    MERLIN_GENERATIVE_SCORING_MAX_ITEMS: int = 32
    MERLIN_GENERATIVE_SCORING_MAX_TOKENS: int = 32768
    # Lower-peak eager Qwen3 vision tower (exactly "1"; default off).
    # AOT: no -- eager tower dispatch; compiled vision blocks keep the plain path.
    # CAPTURE: no -- metadata-only encoder graphs keep the plain path.
    VLLM_MERLIN_VISION_PEAK: bool = False
    # Prompt scoring runs eagerly after forward, outside AOT and CUDA graphs.
    VLLM_MERLIN_PROMPT_LOGPROBS_CHUNK_SIZE: int = 256
    # API server: /health answers 503 after this many seconds without an
    # EngineCore step while requests are in flight; 0 = off.
    VLLM_MERLIN_HEALTH_STALL_S: int = 120
    # API-server chat rendering: comma-separated added
    # tokens that sanitize_control_tokens also encodes as text in client content,
    # on top of Transformers' own control set. Empty = Transformers' set only.
    # The Ornith launcher sets the six FIM tokens. AOT: no. CAPTURE: no.
    VLLM_MERLIN_EXTRA_CONTROL_TOKENS: str = ""
    VLLM_MERLIN_PROMPT_LOGPROBS_TENSOR_BUDGET_MIB: int = 384
    # AOT: no -- eager tower dispatch, outside compiled decoder regions.
    # CAPTURE: no -- metadata-only encoder graphs keep the plain path.
    # Both controls are runtime-only (MERLIN_IGNORED); A/B arms reuse AOT caches.
    VLLM_MERLIN_VISION_CHUNK_ROWS: int = 8192
    VLLM_MERLIN_VISION_CHUNK_MIN_PIXELS: int = 8_388_608
    VLLM_E2_QSA_LOGITS_GRANULE_MB: int = 0
    # Prefix-cache thrash fixes for hybrid (Mamba/GDN) models, default ON; see
    # vllm/v1/core/cache_thrash.py. Cost-aware admission: seconds a young cold
    # miss may be parked behind waiting resident hits (0 = strict arrival
    # order). Coherent yields: need-sized, session-coherent yields (strict
    # "0"/"1"; "0" restores the old one-pin-at-a-time yields). AOT: no.
    # CAPTURE: no -- scheduler-side admission order and yield policy only.
    VLLM_E2_SCHED_MISS_DEFER_S: int = 120
    VLLM_E2_SCHED_COHERENT_YIELD: bool = True
    # V2 runner + PP + async scheduling: cap the decode rows of one step
    # at ceil(decoding requests / pp_size), deferring the rest by one step, so
    # the pp_size decode microbatches stay balanced instead of keeping the split
    # admission gave them (e.g. 7/1). Strict "0"/"1", default off. AOT: no.
    # CAPTURE: no -- scheduler-side step composition only.
    VLLM_MERLIN_PP_DECODE_BALANCE: bool = False
    # Decode-first step budget: running decoders' tokens are
    # reserved before prefill chunks are sized, so a decoder behind a long
    # prefill still gets its step. Strict "0"/"1", default off. AOT: no.
    # CAPTURE: no -- scheduler-side step composition only.
    VLLM_MERLIN_DECODE_FIRST: bool = False
    # Prefill runs (vllm/merlin/prefill_runs.py): while sessions decode, a long
    # prefill advances in runs of CHUNKS full chunks of CHUNK_TOKENS, then
    # yields a decode-only window sized from DECODE_SHARE and the TICK_COSTS
    # constants ("prefill tick ms,decode round ms,fill ticks"). Strict
    # "0"/"1", default off. AOT: no. CAPTURE: no -- scheduler-side step
    # composition only.
    VLLM_MERLIN_PREFILL_RUNS: bool = False
    VLLM_MERLIN_PREFILL_RUNS_CHUNKS: int = 16
    VLLM_MERLIN_PREFILL_RUNS_CHUNK_TOKENS: int = 1024
    VLLM_MERLIN_PREFILL_RUNS_DECODE_SHARE: float = 0.30
    VLLM_MERLIN_PREFILL_RUNS_TICK_COSTS: tuple[float, float, float] = (139.0, 25.0, 1.0)
    # Compile-cache fault guard, default on; exactly "0" disables. After
    # a sticky CUDA fault the run's compile cache dirs are marked, and the next
    # boot moves a marked dir to .merlin-quarantine/ and recompiles; AOT
    # artifacts carry a sha256 sidecar. AOT: no (never a cache-key factor).
    # CAPTURE: no.
    VLLM_MERLIN_COMPILE_CACHE_GUARD: bool = True
    # QSA indexer selection scratch in the memory-profile run (a port
    # of vllm#60490): off (default) | reserve (one transient
    # allocation of the bound) | workspace (the logits + top-k part lives in
    # the shared MRV2 workspace, which serving then slices). KV capacity moves;
    # outputs do not. AOT: no. CAPTURE: no (never used inside graph capture).
    VLLM_MERLIN_QSA_SCRATCH: str = "off"
    # Drafter MoE profile forwards at startup (exactly "0" skips them; any
    # other value runs them). Host-side dummy forwards in the memory-profiling
    # window only: AOT: no. CAPTURE: no. Runtime-only (MERLIN_IGNORED).
    VLLM_SPEC_PROFILE_SHAPES: bool = True
    # Profiling-teardown KV retention check (vllm/v1/worker/gpu/cudagraph_utils.py):
    # "1" turns the boot abort into a warning; default "0" aborts. AOT: no.
    # CAPTURE: no. Runtime-only (MERLIN_IGNORED).
    MERLIN_ALLOW_PROFILING_KV_RETENTION: bool = False
    # Host-RAM KV tier (vllm/merlin/kv_host_tier.py): GiB of host memory PER
    # RANK, 0 = off (the default). A positive value makes the engine configure
    # MerlinKVHostTierConnector (a write-back SimpleCPUOffload) itself. Strict
    # non-negative integer. AOT: no -- the connector's per-layer hooks are
    # no-ops and the tier's KVTransferConfig is left out of the config hash.
    # CAPTURE: no -- no op enters the forward. Runtime-only (MERLIN_IGNORED).
    MERLIN_KV_HOST_TIER_GIB: int = 0
    # Host admission policy for new copies, "off"/"lru" (plain
    # LRU, the default) or "slru" (byte-weighted probation/protected);
    # validated by kv_host_tier_policy.admission_mode. Runtime-only.
    MERLIN_KV_HOST_TIER__ADMISSION: str | None = None
    # Write-back pacing: copy credits, admission escape steps, the
    # restores-in-flight cap and units assembling at once (see the getters).
    # Runtime-only: scheduler-side copy issue.
    MERLIN_KV_HOST_TIER__COPY_CREDITS: int = 32
    MERLIN_KV_HOST_TIER__ESCAPE_STEPS: int = 8
    MERLIN_KV_HOST_TIER__N_RESTORE: int = 2
    MERLIN_KV_HOST_TIER__MAX_ASSEMBLING: int = 2
    # Demotion-direction DMA credits (rows; default = the
    # shared COPY_CREDITS) and an optional byte cap per rank in MiB (0 = rows
    # only). Runtime-only: scheduler-side copy issue.
    MERLIN_KV_HOST_TIER__COPY_CREDITS_D: int = 32
    MERLIN_KV_HOST_TIER__COPY_BYTES_D_MIB: int = 0
    MERLIN_KV_HOST_TIER__MAX_PENDING_STORE_BLOCKS: int = 0
    # Keep a published successor's predecessor as its fallback restore
    # point (host eviction order only; 0/unset = off). Runtime-only.
    MERLIN_KV_HOST_TIER__FALLBACK_BOUNDARY: bool = False
    # DEBUG ONLY: the tier census (unit directory, proofs, histograms and the
    # per-request "tier return" lines a test run reads); 0/unset = off. The
    # per-rank store-copy timings behind the copy-cost metric
    # (vllm:merlin_kv_host_tier_copy_ms_per_block) are collected whenever the
    # tier is on, whatever this knob says: one pooled CUDA event pair per
    # store batch, with no extra synchronisation.
    # Runtime-only: AOT: no. CAPTURE: no.
    MERLIN_KV_HOST_TIER_CENSUS: bool = False
    # DEBUG ONLY: KV tier diagnosis log lines (FGPC junction publish / pop,
    # first-turn cached tokens, host eviction victims with the host row
    # split); 0/unset = off, and nothing else changes. Runtime-only: AOT: no.
    # CAPTURE: no.
    MERLIN_KV_TIER_DIAG: bool = False
    # DEBUG ONLY: fail the first N restore transfers per rank (0/unset = off).
    # Requires the host tier on. Runtime-only: AOT: no. CAPTURE: no.
    MERLIN_KV_HOST_TIER_FAIL_LOADS: int = 0
    # DEBUG ONLY: fail the first N demotion (store) transfers per rank
    # (0/unset = off). Requires the host tier on. Runtime-only: AOT: no.
    # CAPTURE: no.
    MERLIN_KV_HOST_TIER_FAIL_STORES: int = 0
    # --- 27B lane knobs -------------------
    # Hashed unless listed in MERLIN_IGNORED below.
    VLLM_FLASHINFER_PLAN_CACHE: bool = True
    # Opt-in workload-selected FlashInfer verify tile (default off). Needs
    # the 0001+0002 FlashInfer series (release/install-env.sh with this knob at 1).
    # Hashed only when on (add_merlin_compile_factors): off keeps main's key.
    MERLIN_FLASHINFER_WORKLOAD_TILE: bool = False
    VLLM_GDN_BA_SPLITK: bool = True
    VLLM_DRAFT_SEQ_LENS_NOSYNC: bool = True
    VLLM_DRAFT_SEQ_LENS_VERIFY: bool | None = None
    VLLM_DRAFT_BOUNDED_PLAN: bool = False
    VLLM_DRAFT_BOUNDED_PLAN_VERIFY: bool | None = None
    VLLM_TARGET_SEQ_LENS_ASYNC: bool = False
    VLLM_TARGET_SEQ_LENS_ASYNC_VERIFY: bool | None = None
    # GDN (RecoverSSM) host metadata trims, decode steps only. Both are
    # host-side only: the staged buffers keep their addresses and receive the
    # same values, so neither AOT nor capture identity depends on them (ignored
    # factors). Default off.
    VLLM_GDN_SHARED_ALIGN_INDEX: bool = False
    VLLM_GDN_DECODE_FASTPATH: bool = False
    # RecoverSSM: enqueue the post-sampling GDN commit (and the mamba align
    # postprocess that consumes it) after the MTP proposal instead of before it,
    # on the same stream. AOT: no -- Python call order after sampling, outside
    # the compiled forward. CAPTURE: no -- not inside any captured graph.
    # Ignored factor, so an A/B flip reuses one AOT cache.
    VLLM_RECOVERSSM_COMMIT_AFTER_PROPOSE: bool = False
    # RecoverSSM + PP, non-last ranks: record a second receive event after the
    # sampled-token/count broadcasts so the deferred commit waits only for the
    # counts, not for the drafts (post_update still waits for everything).
    # Same kernels and arguments; reordered only across data-disjoint writers.
    # AOT: no -- eager Python/stream order after sampling. CAPTURE: no.
    # Ignored factor. Default off.
    VLLM_RECOVERSSM_PP_SPLIT_RECV: bool = False
    # RecoverSSM post-sampling commit execution mode
    # (vllm/v1/worker/gpu/model_states/recoverssm_commit_graph.py): "eager"/"0"/unset
    # = the unchanged launch chain, "staged" = staged inputs + eager chain (control
    # arm), "graph"/"1" = staged inputs + a replayed standalone CUDA graph of the same
    # chain. Default only: the worker method set_recoverssm_commit_mode flips it in a
    # running process; the boot value also decides whether the CUDA-graph capture
    # phase captures the commit graphs (not for eager). AOT: no -- read by Python
    # after sampling, outside the compiled forward. CAPTURE: no for the forward
    # graphs; the commit graphs are separate in-process graphs, keyed on the step's
    # tensors, never cached across processes, and bit-identical to the eager chain.
    # Ignored factor, so an A/B flip reuses one AOT cache. Unknown values raise at
    # boot.
    VLLM_RECOVERSSM_COMMIT_GRAPH: str = ""
    # Debug only: assert the source-block ownership invariant on every
    # Mamba2 RecoverSSM commit (a host sync per commit; never for serving).
    # Strict "0"/"1", default off. AOT: no; CAPTURE: no -- a host-side check
    # after the commit. Ignored factor.
    VLLM_RECOVERSSM_CHECK_OWNERSHIP: bool = False
    # --- Opt-in Qwen3.5 MoE / single-card memory knobs (off by default) ----
    # FULL-graph MTP verify on Qwen3.5 MoE / sm_86
    # (v1/qwen35_series.uniform_verify_active). Runtime-only (MERLIN_IGNORED).
    MERLIN_MOE_UNIFORM_VERIFY: bool = False
    # MTP drafter reuses the target's FlashInfer float workspace
    # (merlin/draft_workspace.py). Runtime-only (MERLIN_IGNORED).
    MERLIN_SHARE_DRAFT_WORKSPACE: bool = False
    # VISION_PEAK path: rotate Q/K inside the QKV buffer (no QK copy). Runtime-only.
    # Inert unless VLLM_MERLIN_VISION_PEAK=1 runs the eager tower on the flash-attn
    # ViT backend.
    VLLM_MERLIN_VISION_ROTARY_INPLACE: bool = False
    # -----------------------------------------------------------------------
    # None selects the model/device default; explicit values retain exact "1" parsing.
    # None and False must be distinct compile factors.
    VLLM_HOST_EMBED_TABLE: bool | None = None
    VLLM_PLE_OFFLOAD_READY_TIMEOUT: float = 600.0
    # "1" keys the block-verification acceptance draws and
    # the draft Gumbel noise by round (anchor position + in-round offset)
    # instead of absolute position, which makes rejection_sample_method="block"
    # distribution-exact. Effective only with block verification (a logged
    # no-op otherwise); standard/synthetic keying is never changed. Exactly
    # "0"/"1"; default OFF (byte-identical draws). Read once at rejection
    # sampler and speculator init (static).
    # AOT: no -- the rejection kernels and sample_draft run outside the
    # compiled model regions; Triton keys its own cache on the new constexprs.
    # CAPTURE: yes -- the draft sampler is CAPTURED under FULL draft graphs
    # with the STEP_KEYED constexpr; the keys it reads (positions, the
    # persistent current_draft_step, seeds) are device buffers, so replay needs
    # no host RNG. Frozen at init, graphs never persisted.
    VLLM_SPEC_STEP_KEYED_RNG: bool = False
    # Default-off numerical repair.
    VLLM_SPEC_STABLE_RESIDUAL: bool = False
    # --- Flash-Next knobs ---------------------------------------------
    # Registered so compile_factors() hashes them; the read sites still parse
    # and fail closed themselves. Each entry states two identities: AOT (does it
    # change the compiled artifact; the only one compile_factors() keys) and
    # CAPTURE (does it change what a cudagraph bakes in; protected only by not
    # changing it within one process). Where either is not established from the
    # code, the knob is hashed anyway: a spurious miss costs a recompile.
    #
    # COMPILE/CAPTURE FACTORS (hashed):
    # CAPTURE: yes, established -- clip counting allocates persistent per-layer
    # int64 counters and its increment ops are CAPTURED into the cudagraph (see
    # the graph-safety note in models/qwen4_exp/nvidia/qsa.py); unset, those ops
    # are absent. AOT: not established (the QSA forward is a custom op); hashed
    # conservatively.
    VLLM_QSA_KV_CLIP_COUNT: str = ""
    # CAPTURE: yes -- the resolved value is a launch constant baked into the
    # graph at capture (csrc/merlin/topk/persistent_topk.cuh,
    # PersistentTopKParams::exact_medium).
    # AOT: not established. Hashed conservatively.
    # Raw passthrough kept for the compile-cache hash: no behaviour reads it
    # (vllm/v1/worker/e2_toggles.py only logs the raw value), and the
    # medium-path exact restart is fixed on in csrc/merlin/topk/topk.cu. Do
    # not use envs.MERLIN_TOPK_EXACT_MEDIUM to decide behaviour.
    MERLIN_TOPK_EXACT_MEDIUM: str | None = None
    #
    # RUNTIME-ONLY (registered for validate_environ + a single cached read):
    # Static K/V scale sidecar: fill_()ed into existing fp32 device buffers
    # after load and before capture, and the kernel does tl.load(k_scale_ptr),
    # so the value is never a constexpr and the graph is identical.
    VLLM_QSA_KV_SCALES: str = ""
    # Guard telemetry: mode raise|warn and the reader poll interval in
    # seconds. counter() returns a real tensor in every mode, so the Triton
    # arguments and the captured graph do not depend on either.
    VLLM_E2_GUARD_MODE: str = "warn"
    VLLM_E2_GUARD_INTERVAL: str = "0.05"
    # Guard counter transport: "mapped" (default; counters in mapped page-locked
    # host memory, read without a D2H copy, fatal counts checked at every step
    # boundary) or "copy" (device counters, side-stream D2H reader).
    VLLM_E2_GUARD_TRANSPORT: str = "mapped"
    # Debug allocator telemetry (poll seconds + output directory).
    VLLM_E2_MEMSTATS: str = "0"
    VLLM_E2_MEMSTATS_DIR: str = ""
    # PLE offload child placement: rewrites CUDA_VISIBLE_DEVICES for the child
    # only (CUDA_VISIBLE_DEVICES is itself an ignored factor).
    VLLM_PLE_OFFLOAD_HOME_DEVICE: str = ""
    # PLE timeouts / deadlines, in seconds.
    VLLM_PLE_PROGRESS_DEADLINE_S: float = 300.0
    # PLE offload process only: compute the n-gram row IDs with NumPy instead
    # of ~50 tiny torch CPU ops (stage-0 graph stall). Same int64
    # arithmetic, same IDs, same rows; read once at import by
    # vllm/models/qwen4_exp/nvidia/ple_pull.py. AOT: no. CAPTURE: no -- the
    # offload process has no CUDA context. Strict "0"/"1", default off.
    VLLM_E2_PLE_FAST_IDS: bool = False
    # Stage-0 GPU worker: issue the PLE request (D2H of the step's PLE inputs +
    # the offload notification) right after prepare_inputs instead of just
    # before the forward, on real batches only. Same bytes, earlier on the
    # stream; dummy/profile/capture forwards keep the late issue, so no graph
    # changes. Read once at import by vllm/v1/worker/gpu/model_runner.py.
    # AOT: no. CAPTURE: no. Strict "0"/"1", default off.
    VLLM_E2_PLE_EARLY_REQUEST: bool = False
    # PLE offload process only: where the table lives, "ram" (default) or "disk"
    # (read from the checkpoint's safetensors files); whether a disk table
    # is warmed after READY; and the threads that take its page faults. Read once
    # at import by vllm/models/qwen4_exp/nvidia/ple_pull.py. AOT: no. CAPTURE: no --
    # the offload process has no CUDA context, and the rows are the same bytes.
    VLLM_E2_PLE_STORE: str = "ram"
    VLLM_E2_PLE_WARM: bool = True
    VLLM_E2_PLE_DISK_THREADS: int = 32
    #
    # The tie-determinism toggles are registered and classified in the
    # "determinism toggles" block below.
    MERLIN_TOPK_EXACT_COUNTERS_LOG: int = 0
    # Raw passthrough; runtime-only (ignored factor) and never consulted:
    # readers take the slot count from the build (persistent_topk_exact_counters_slots).
    MERLIN_TOPK_COUNTER_SLOTS: str | None = None
    # Flash-Next family-name aliases, "alias=canonical[,alias=canonical...]"
    # (vllm/flashnext_family.py). Default unset/"" (the resolved name sets are
    # exactly the historical Qwen4Exp literals). flashnext_family is stdlib-only
    # (imported by the config layer and the model registry), so it parses
    # os.environ itself; this entry is the registration, not a second parser.
    # AOT: no. CAPTURE: no -- it only changes which architecture / model_type
    # names resolve to the family at config and registry time; no kernel, op or
    # captured graph depends on it. Runtime-only (ignored_factors).
    VLLM_FLASHNEXT_FAMILY_ALIASES: str = ""
    # Qwen3.5-lineage (dense 27B / MoE) architecture and model_type names,
    # "Name,alias=canonical,-Name" (vllm/qwen35_family.py is the parser and the
    # documentation; docs/merlin/qwen4-27b-day-one.md is the day-one use). Default
    # unset/"": the canonical sets are the historical Qwen3_5* literals plus the
    # default Qwen 4 candidates, which match no shipped checkpoint. qwen35_family
    # is stdlib-only, so it parses os.environ itself; this entry is the
    # registration. AOT: no (ignored_factors), on the same rule as the Flash-Next
    # aliases: it decides which names resolve to the lineage at config and
    # registry time, and registering it must not move the release AOT cache key.
    # An entry changes the compiled graph only for the checkpoint it names, which
    # could not boot as a Qwen3.5-lineage model before; after changing an entry
    # for a checkpoint that already booted, use a fresh VLLM_CACHE_ROOT.
    # CAPTURE: no.
    MERLIN_QWEN35_ARCH_ALIASES: str = ""
    #
    # --- determinism toggles -----------------------------------------
    # FULL_K, QSA_SORT and TIE_DET/TIE_RECENT default to the shipped value 1
    # (SHIPPED_KERNEL_TOGGLE_DEFAULTS is the single source; vllm/env_override.py
    # puts it into the environment at import). All are hashed (none is in
    # ignored_factors) because each reaches a capture. Only MERLIN_QSA_SORT is
    # still read; FULL_K / TIE_DET / TIE_RECENT are fixed on in C++ and these
    # entries are raw passthroughs for the hash, never a behaviour switch.
    #
    # Marlin MoE split-K reduction order; the CUDA half of upstream PR #46639
    # (`use_full_k`).
    MERLIN_FULL_K: str | None = "1"
    # persistent_topk tie-break at the k-th cutoff, and its direction.
    MERLIN_TIE_DET: str | None = "1"
    MERLIN_TIE_RECENT: str | None = "1"
    #
    # Unlike the three above, a Python read site: parsed here and read
    # once at import by models/qwen4_exp/nvidia/ops/qsa_indexer.py, which adds
    # a torch.sort + two torch.where ops to the QSA selection path.
    MERLIN_QSA_SORT: bool = True
    # GDN in_proj_ba GEMM on the shared aux stream, overlapped with
    # in_proj_qkvz. Read at layer build by
    # model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py; CUDA only, and
    # serial under breakable CUDA graph capture. Same kernels, same inputs,
    # disjoint outputs, so the result is bit-identical. CAPTURE: yes (adds a
    # fork/join to the captured graph); default OFF, hashed by
    # compile_factors (not in ignored_factors).
    VLLM_GDN_BA_AUX_STREAM: bool = False
    # Packed GDN decode (VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE) with the
    # numerics of the kernels that verify speculative tokens (vllm#65338:
    # rsqrt L2 norm, 4 warps), so plain decode matches the verify recurrence.
    # Changes plain-decode bits; RecoverSSM lanes never run the packed kernel.
    # Read at layer build by
    # model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py. AOT: no -- the
    # kernel runs inside the opaque qwen_gdn_attention_core op. CAPTURE: yes,
    # in-process only (graphs capture the selected launch). Ignored factor,
    # so an A/B flip reuses one AOT cache. Default OFF.
    VLLM_FLA_PACKED_DECODE_VERIFY_PARITY: bool = False
    #
    # --- persistent_topk kernel switches ------------------------------
    # Same classification as the three raw passthroughs above and registered
    # for the same single reason: compile_factors() is a deny-list, so a name
    # it does not know cannot be hashed, and a Ray deployment that propagates
    # registered variables can otherwise drop an explicitly-disabled switch
    # and leave a worker on the default. Each policy is fixed in
    # csrc/merlin/topk/topk.cu (its *_enabled() getter returns a constant and
    # reads no variable); the value reaches the GPU as a kernel launch
    # argument, so a cudagraph bakes it in.
    # CAPTURE: yes, all three. AOT: not established (these kernels are built
    # into the extension, not by the AOT compiler); hashed conservatively
    # anyway, same rule as above. None is in ignored_factors.
    #
    # Raw passthroughs, so "unset" reads as None here; the kernel ignores the
    # variable either way (EXACT_MEDIUM, EXACT_DECODE and DET_LARGE are on).
    #
    # Take the exact restart on an overflowing MEDIUM row:
    # MERLIN_TOPK_EXACT_MEDIUM, declared once in the cross-change knob block
    # above (same raw passthrough, same hashing).
    # The same restart on an overflowing DECODE row.
    MERLIN_TOPK_EXACT_DECODE: str | None = None
    # Deterministic placement and tie membership on the LARGE path. Always on:
    # det_large_enabled() in csrc/merlin/topk/topk.cu returns a constant (no
    # variable is read), and the value reaches the large-path radix_topk as a
    # launch argument, so a cudagraph bakes it in. CAPTURE: yes. AOT: not
    # established (built into the extension, not by the AOT compiler); hashed
    # conservatively like the switches above.
    MERLIN_TOPK_DET_LARGE: str | None = None
    #
    # MERLIN_TOPK_SORTED_EMIT -- persistent_topk emits every row in the
    # canonical order MERLIN_QSA_SORT imposes (ascending index, padding last as
    # -1), and qsa_indexer.py then skips its Python sort. Same classification
    # as the kernel switches above: sorted_emit_enabled() in
    # csrc/merlin/topk/topk.cu returns a constant, and the value reaches the
    # GPU as a kernel launch argument AND removes three torch ops from the
    # captured sequence, so a cudagraph bakes it in. CAPTURE: yes.
    # AOT: not established for the kernel half; hashed conservatively (a fork
    # env knob must be a compile factor: an os.environ knob that is not
    # registered is not in the cache key). Raw passthrough: Python never parses
    # it to decide anything -- qsa_indexer.py reads the C++ cache through
    # persistent_topk_toggle_state() field 6. Not in ignored_factors.
    MERLIN_TOPK_SORTED_EMIT: str | None = "1"
    #
    # MERLIN_MARLIN_MOE_CFG: per-shape Marlin MoE block_size_m and
    # thread config, chosen among kernels that are already compiled. Unset (the
    # default) = the engine's own heuristic, byte-identical. The raw string is
    # parsed in Python by vllm/merlin/marlin_moe_cfg.py (the single read site and
    # parser; this entry is a passthrough). Syntax and the exactness classes are
    # documented there. IGNORED as a compile factor, see MERLIN_IGNORED.
    MERLIN_MARLIN_MOE_CFG: str | None = None
    # Exact-shape W8A16 tiles; default OFF.
    MERLIN_MARLIN_DENSE_CFG: str | None = None
    # Static cold-expert offload: profile path (unset, blank or 0 = OFF), optional
    # layer list and hot-fraction overrides. Parsed only by
    # vllm/merlin/cold_experts_plan.py, at load time. IGNORED as compile
    # factors, see MERLIN_IGNORED.
    MERLIN_COLD_EXPERTS: str | None = None
    MERLIN_COLD_EXPERTS_LAYERS: str | None = None
    MERLIN_COLD_EXPERTS_HOT_FRAC: str | None = None
    # Boot-memory helpers, both around post-load weight processing:
    # MERLIN_REPACK_RELEASE (default on, 0 = stock) empty_caches the allocator
    # after a module's repack (vllm/merlin/repack_release.py);
    # MERLIN_LOAD_PEAK_LOG=1 logs per-module load peaks, a diagnostic
    # (vllm/merlin/load_peak_log.py). Both readers parse os.environ themselves;
    # these entries are the registration. IGNORED as compile factors.
    MERLIN_REPACK_RELEASE: bool = True
    MERLIN_LOAD_PEAK_LOG: bool = False
    # Qwen3 tool-call parser literal spans, default on, 0 = off: fenced
    # spans and tool blocks quoted in closed reasoning stay text. Parser-only
    # (vllm/parser/qwen3.py reads it). IGNORED as a compile factor.
    MERLIN_QWEN3_LITERAL_SPANS: bool = True
    #
    # NOT REGISTERED: MERLIN_TOPK_EXACT_TRAP is a build-time -D define in
    # persistent_topk.cuh, not an environment variable.
    #
    # --- UNAUDITED: host-side knobs with no read site in this tree ----------
    # Names only, one per line, inside the fence; a knob leaves the fence when
    # its AOT and capture identity are written down against its read site.
    # The fence is empty: every host-side knob has its identity written
    # down.
    # --8<-- [start:unaudited-host-side-knobs]
    # --8<-- [end:unaudited-host-side-knobs]


# Compat flag for the 0/1 switch family. False: a value outside {"0", "1"}
# logs one warning naming the knob and the value, then keeps the
# interpretation that knob had before the helper existed, so no served
# behaviour changes. True makes every out-of-set value a boot error.
STRICT_SWITCH_ENFORCE = False

_switch_warned: set[tuple[str, str]] = set()


def _ple_store_value() -> str:
    value = os.getenv("VLLM_E2_PLE_STORE", "ram")
    if value not in ("ram", "disk"):
        raise ValueError(f"VLLM_E2_PLE_STORE must be ram or disk, got {value!r}")
    return value


def _ple_disk_threads() -> int:
    raw = os.getenv("VLLM_E2_PLE_DISK_THREADS", "32")
    if not (raw.isascii() and raw.isdigit()) or int(raw) < 1:
        raise ValueError(f"VLLM_E2_PLE_DISK_THREADS must be a whole number >= 1, got {raw!r}")
    return int(raw)


def _e2_strict_switch(
    name: str,
    default: bool = False,
    *,
    legacy: str = "strict",
    none_if_unset: bool = False,
) -> bool | None:
    """Parse a fork on/off switch; the one parser for every Python 0/1 knob.

    Accepted: exactly ``"0"`` and ``"1"``; unset means ``default`` (``None``
    when ``none_if_unset``, for knobs where unset and explicit 0 differ).
    Anything else is handled per ``STRICT_SWITCH_ENFORCE``: raise when True;
    otherwise warn once and apply the knob's pre-helper reading, ``legacy``:
    ``"strict"`` raises (it always did), ``"int"`` is ``bool(int(raw))``
    (``"2"`` is on, ``"true"`` and ``""`` raise), ``"eq1"`` is
    ``raw == "1"`` (everything else off). An empty value is out of set like
    any other: it never silently means the default.
    """
    raw = os.environ.get(name)
    if raw is None:
        return None if none_if_unset else default
    if raw == "0":
        return False
    if raw == "1":
        return True
    msg = f'{name} must be exactly "0" or "1" (or unset), got {raw!r}'
    if STRICT_SWITCH_ENFORCE:
        raise ValueError(msg)
    if (name, raw) not in _switch_warned:
        _switch_warned.add((name, raw))
        import logging

        logging.getLogger(__name__).warning(
            "%s; keeping the legacy reading (%s) for this release, "
            "this becomes a boot error in a later release",
            msg,
            legacy,
        )
    if legacy == "eq1":
        return raw == "1"
    if legacy == "int":
        try:
            return bool(int(raw))
        except ValueError:
            raise ValueError(msg) from None
    raise ValueError(msg)


def _qsa_scratch_mode() -> str:
    raw = os.getenv("VLLM_MERLIN_QSA_SCRATCH", "off").strip().lower()
    if raw not in ("off", "reserve", "workspace"):
        raise ValueError(
            "VLLM_MERLIN_QSA_SCRATCH must be off, reserve or workspace, "
            f"got {raw!r}"
        )
    return raw


def _prefill_runs_share() -> float:
    """VLLM_MERLIN_PREFILL_RUNS_DECODE_SHARE: a decimal in [0, 1); 0 keeps the
    full-chunk cap and drops the decode-only windows."""
    name = "VLLM_MERLIN_PREFILL_RUNS_DECODE_SHARE"
    raw = os.environ.get(name)
    if raw is None:
        return 0.30
    try:
        value = float(raw)
    except ValueError:
        value = -1.0
    if not (raw.strip() == raw and math.isfinite(value) and 0.0 <= value < 1.0):
        raise ValueError(f"{name} must be a decimal in [0, 1), got {raw!r}")
    return value


def _prefill_runs_tick_costs() -> tuple[float, float, float]:
    """VLLM_MERLIN_PREFILL_RUNS_TICK_COSTS: "prefill tick ms,decode round
    ms,fill ticks", e.g. "139,25,1"; the first two > 0, the third >= 0."""
    name = "VLLM_MERLIN_PREFILL_RUNS_TICK_COSTS"
    raw = os.environ.get(name)
    if raw is None:
        return (139.0, 25.0, 1.0)
    parts = raw.split(",")
    try:
        values = tuple(float(p) for p in parts)
    except ValueError:
        values = ()
    if (
        len(values) != 3
        or any(p.strip() != p or not p for p in parts)
        or not all(math.isfinite(v) for v in values)
        or not (values[0] > 0 and values[1] > 0 and values[2] >= 0)
    ):
        raise ValueError(
            f'{name} must be "prefill tick ms,decode round ms,fill ticks" '
            f"(two positive finite numbers and one >= 0), got {raw!r}"
        )
    return values  # type: ignore[return-value]


def _e2_nonneg_int(name: str, default: int = 0) -> int:
    """Parse a fork knob that accepts only a plain non-negative decimal integer.

    Unset means ``default``. Anything else (``""``, ``" 16"``, ``"-1"``,
    ``"1.5"``) is a boot error.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    if not (raw.isascii() and raw.isdigit()):
        raise ValueError(f"{name} must be a non-negative integer, got {raw!r}")
    return int(raw)


def _kv_host_tier_fault_quota(name: str) -> int:
    value = _e2_nonneg_int(name)
    if value and _e2_nonneg_int("MERLIN_KV_HOST_TIER_GIB") == 0:
        raise ValueError(f"{name} requires MERLIN_KV_HOST_TIER_GIB > 0")
    return value


def kv_host_tier_admission() -> str | None:
    """The host admission policy flag, normalised by
    ``kv_host_tier_policy.admission_mode`` ("off" / "slru"); None when unset.
    A bad value is a boot error."""
    raw = os.environ.get("MERLIN_KV_HOST_TIER__ADMISSION")
    if raw is None:
        return None
    from vllm.merlin.kv_host_tier_policy import admission_mode

    return admission_mode(raw)


def kv_host_tier_fail_loads() -> int:
    """Debug fault injection: first N restore transfers per rank, never CUDA chunks.

    Refuse a nonzero setting with the tier off, before any copy is issued.
    """
    return _kv_host_tier_fault_quota("MERLIN_KV_HOST_TIER_FAIL_LOADS")


def kv_host_tier_fail_stores() -> int:
    """Debug fault injection: first N demotion transfers per rank (same rules)."""
    return _kv_host_tier_fault_quota("MERLIN_KV_HOST_TIER_FAIL_STORES")


def _e2_positive_int(name: str, default: int) -> int:
    value = _e2_nonneg_int(name, default)
    if value == 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


# Architectures whose embedding tables may be kept in pinned host memory
# (VLLM_HOST_EMBED_TABLE): host_embed_archs(). Other architectures never take
# this path.
def host_embed_archs() -> tuple[str, ...]:
    """Architectures whose embedding shards may live on the host: the Qwen3.5 lineage
    (vllm/qwen35_family.py, dense and MoE) and Nemotron 3.5 Super VL."""
    from vllm import qwen35_family

    return (
        *qwen35_family.gdn_archs(),
        "NemotronH_Omni_Reasoning_V3",  # Nemotron 3.5 Super VL
    )


def host_embed_table_enabled_for_model(architecture: str) -> bool:
    """Resolve VLLM_HOST_EMBED_TABLE for one architecture.

    Only ``host_embed_archs()`` are eligible. For those, ``1`` turns
    host embedding on, ``0`` turns it off and unset means on (the default).
    """
    if architecture not in host_embed_archs():
        return False
    # Read through vllm.envs so the value comes from the same (possibly cached)
    # module __getattr__ as every other reader; keep module imports stdlib-only.
    import vllm.envs as envs

    setting = envs.VLLM_HOST_EMBED_TABLE
    return True if setting is None else setting


# Shipped value of each determinism toggle. The registry getters default to
# these and vllm/env_override.py setdefaults them into os.environ at import,
# so there is one table, not two.
SHIPPED_KERNEL_TOGGLE_DEFAULTS: dict[str, str] = {
    "MERLIN_FULL_K": "1",
    "MERLIN_QSA_SORT": "1",
    "MERLIN_TIE_DET": "1",
    "MERLIN_TIE_RECENT": "1",
    "MERLIN_TOPK_SORTED_EMIT": "1",
}

MERLIN_ENV: dict[str, Callable[[], Any]] = {
    # Per-request scoring limits; zero cannot disable either guard.
    "MERLIN_GENERATIVE_SCORING_MAX_ITEMS": lambda: _e2_positive_int(
        "MERLIN_GENERATIVE_SCORING_MAX_ITEMS", 32
    ),
    "MERLIN_GENERATIVE_SCORING_MAX_TOKENS": lambda: _e2_positive_int(
        "MERLIN_GENERATIVE_SCORING_MAX_TOKENS", 32768
    ),
    "VLLM_MERLIN_VISION_PEAK": lambda: _e2_strict_switch(
        "VLLM_MERLIN_VISION_PEAK", default=False, legacy="eq1"
    ),
    "VLLM_MERLIN_PROMPT_LOGPROBS_TENSOR_BUDGET_MIB": lambda: _e2_positive_int(
        "VLLM_MERLIN_PROMPT_LOGPROBS_TENSOR_BUDGET_MIB", 384
    ),
    "VLLM_MERLIN_PROMPT_LOGPROBS_CHUNK_SIZE": lambda: _e2_positive_int(
        "VLLM_MERLIN_PROMPT_LOGPROBS_CHUNK_SIZE", 256
    ),
    # 0 disables the /health stall check; negative/malformed values fail closed.
    "VLLM_MERLIN_HEALTH_STALL_S": lambda: _e2_nonneg_int(
        "VLLM_MERLIN_HEALTH_STALL_S", 120
    ),
    "VLLM_MERLIN_EXTRA_CONTROL_TOKENS": lambda: os.environ.get(
        "VLLM_MERLIN_EXTRA_CONTROL_TOKENS", ""
    ),
    # 0 disables chunking; negative/malformed values fail closed.
    "VLLM_MERLIN_VISION_CHUNK_ROWS": lambda: _e2_nonneg_int(
        "VLLM_MERLIN_VISION_CHUNK_ROWS", default=8192
    ),
    # The execution gate always floors this at today's 8 MP cap.
    "VLLM_MERLIN_VISION_CHUNK_MIN_PIXELS": lambda: _e2_nonneg_int(
        "VLLM_MERLIN_VISION_CHUNK_MIN_PIXELS", default=8_388_608
    ),
    # QSA prefill indexer logits backing granule in MiB (default 0 = off: the
    # logits are allocated at exactly the size each chunk needs). A positive
    # value rounds the allocation up to a multiple of it (16-64 recommended) and
    # reuses one backing for every chunk of a call. Strict non-negative integer.
    "VLLM_E2_QSA_LOGITS_GRANULE_MB": lambda: _e2_nonneg_int(
        "VLLM_E2_QSA_LOGITS_GRANULE_MB"
    ),
    # --- 27B lane knobs ---
    # --- FlashInfer decode-plan cache (fa2/sm_86) ---
    # Single-entry cache that skips fast_decode_plan when the schedule cannot have
    # changed. On by default; VLLM_FLASHINFER_PLAN_CACHE=0 disables it. Registered here
    # (not read from os.environ) so it is part of the compile cache key.
    "VLLM_FLASHINFER_PLAN_CACHE": lambda: _e2_strict_switch(
        "VLLM_FLASHINFER_PLAN_CACHE", default=True, legacy="int"
    ),
    "MERLIN_FLASHINFER_WORKLOAD_TILE": lambda: _e2_strict_switch(
        "MERLIN_FLASHINFER_WORKLOAD_TILE", default=False
    ),
    # --- GDN in_proj_ba split-K (sm_86, small M) ---
    # Triton split-K GEMM for the unquantized GDN in_proj_ba at M <= 16 tokens. On by
    # default; VLLM_GDN_BA_SPLITK=0 keeps cuBLAS. Registered here (not read from
    # os.environ) so it is part of the compile cache key.
    "VLLM_GDN_BA_SPLITK": lambda: _e2_strict_switch(
        "VLLM_GDN_BA_SPLITK", default=True, legacy="int"
    ),
    # --- MTP draft-step seq_lens without a device sync ---
    # Draft step >= 2 takes the previous draft build's CPU seq_lens + 1 instead of a
    # blocking seq_lens.cpu(). On by default; VLLM_DRAFT_SEQ_LENS_NOSYNC=0 disables it.
    # VLLM_DRAFT_SEQ_LENS_VERIFY=1 also syncs, compares, logs mismatches and uses the
    # synced value (diagnostic; off by default). Registered here (not read from
    # os.environ) so both are part of the compile cache key.
    "VLLM_DRAFT_SEQ_LENS_NOSYNC": lambda: _e2_strict_switch(
        "VLLM_DRAFT_SEQ_LENS_NOSYNC", default=True, legacy="int"
    ),
    # None when unset so unset and explicit 0 are distinct compile factors.
    "VLLM_DRAFT_SEQ_LENS_VERIFY": lambda: _e2_strict_switch(
        "VLLM_DRAFT_SEQ_LENS_VERIFY", legacy="int", none_if_unset=True
    ),
    # --- MTP draft decode planned from CPU bounds, no seq_lens sync (draft step 1+) ---
    # Plans the FlashInfer fa2 draft decode from the CPU upper bound on seq_lens when the
    # bounds provably give the same schedule as the true lengths, and writes the true page
    # table on the GPU; otherwise syncs exactly as before. Off by default;
    # VLLM_DRAFT_BOUNDED_PLAN=1 enables it. VLLM_DRAFT_BOUNDED_PLAN_VERIFY=1 also syncs,
    # checks bounds / schedule / page table against the synced lengths, logs mismatches and
    # uses the exact path on a mismatch (diagnostic). Both are host-side builder knobs and
    # are listed in compile_factors' ignored set, so a flip reuses the AOT cache.
    "VLLM_DRAFT_BOUNDED_PLAN": lambda: _e2_strict_switch(
        "VLLM_DRAFT_BOUNDED_PLAN", default=False, legacy="int"
    ),
    # None when unset (diagnostic knob; not a compile factor, see compile_factors).
    "VLLM_DRAFT_BOUNDED_PLAN_VERIFY": lambda: _e2_strict_switch(
        "VLLM_DRAFT_BOUNDED_PLAN_VERIFY", legacy="int", none_if_unset=True
    ),
    # --- Target (verify) FlashInfer build without the seq_lens sync ---
    # Right after post_update the runner enqueues a pinned non-blocking D2H copy of
    # num_computed_tokens + an event; the next verify build waits on that event only (not on
    # the draft steps queued behind it) and computes the EXACT seq_lens on the CPU (same
    # values seq_lens.cpu() returns, so the page table and FlashInfer plan are identical).
    # Off by default; VLLM_TARGET_SEQ_LENS_ASYNC=1 enables it. _VERIFY=1 also syncs, compares
    # against seq_lens.cpu(), logs and uses the synced lengths on a mismatch (diagnostic).
    # Both are registered runtime controls (not read from os.environ) and are excluded from the
    # compile cache key (MERLIN_IGNORED below).
    "VLLM_TARGET_SEQ_LENS_ASYNC": lambda: _e2_strict_switch(
        "VLLM_TARGET_SEQ_LENS_ASYNC", default=False, legacy="int"
    ),
    # None when unset, keeping it distinct from an explicit 0; a runtime control,
    # excluded from the compile cache key (MERLIN_IGNORED).
    "VLLM_TARGET_SEQ_LENS_ASYNC_VERIFY": lambda: _e2_strict_switch(
        "VLLM_TARGET_SEQ_LENS_ASYNC_VERIFY", legacy="int", none_if_unset=True
    ),
    # See its declaration. Default off.
    "VLLM_RECOVERSSM_COMMIT_AFTER_PROPOSE": lambda: _e2_strict_switch(
        "VLLM_RECOVERSSM_COMMIT_AFTER_PROPOSE", default=False, legacy="int"
    ),
    # See its declaration. Default off.
    "VLLM_RECOVERSSM_PP_SPLIT_RECV": lambda: _e2_strict_switch(
        "VLLM_RECOVERSSM_PP_SPLIT_RECV", default=False
    ),
    # --- GDN metadata host trims (see their declaration). Default off. ---
    # VLLM_GDN_SHARED_ALIGN_INDEX=1: the align-mode state-index gather index is
    # computed once per batch and reused by every GDN group (update_block_table)
    # instead of being re-derived per group.
    "VLLM_GDN_SHARED_ALIGN_INDEX": lambda: _e2_strict_switch(
        "VLLM_GDN_SHARED_ALIGN_INDEX", default=False, legacy="int"
    ),
    # VLLM_GDN_DECODE_FASTPATH=1: pure spec-decode FULL-graph batches whose spec
    # rows are a prefix of the batch stage the GDN metadata with slices and
    # fills instead of CPU-mask advanced indexing and H2D copies, and skip the
    # accepted-count work RecoverSSM overrides with ones.
    "VLLM_GDN_DECODE_FASTPATH": lambda: _e2_strict_switch(
        "VLLM_GDN_DECODE_FASTPATH", default=False, legacy="int"
    ),
    # RecoverSSM commit execution mode (see its declaration); runtime only.
    "VLLM_RECOVERSSM_COMMIT_GRAPH": lambda: os.getenv(
        "VLLM_RECOVERSSM_COMMIT_GRAPH", ""
    ).strip(),
    # Debug-only RecoverSSM ownership check (see its declaration).
    "VLLM_RECOVERSSM_CHECK_OWNERSHIP": lambda: _e2_strict_switch(
        "VLLM_RECOVERSSM_CHECK_OWNERSHIP", default=False, legacy="int"
    ),
    # --- end of the 27B lane knobs ---
    # --- Opt-in Qwen3.5 MoE / single-card memory knobs (off by default) ---
    "MERLIN_MOE_UNIFORM_VERIFY": lambda: _e2_strict_switch("MERLIN_MOE_UNIFORM_VERIFY"),
    "MERLIN_SHARE_DRAFT_WORKSPACE": lambda: _e2_strict_switch(
        "MERLIN_SHARE_DRAFT_WORKSPACE"
    ),
    "VLLM_MERLIN_VISION_ROTARY_INPLACE": lambda: _e2_strict_switch(
        "VLLM_MERLIN_VISION_ROTARY_INPLACE"
    ),
    "VLLM_HOST_EMBED_TABLE": lambda: _e2_strict_switch(
        "VLLM_HOST_EMBED_TABLE", legacy="eq1", none_if_unset=True
    ),
    # Timeout for PLE weight loading and TP worker registration.
    "VLLM_PLE_OFFLOAD_READY_TIMEOUT": lambda: float(
        os.getenv("VLLM_PLE_OFFLOAD_READY_TIMEOUT", "600")
    ),
    # Step-keyed block-verification draws; see the declaration.
    "VLLM_SPEC_STEP_KEYED_RNG": lambda: _e2_strict_switch(
        "VLLM_SPEC_STEP_KEYED_RNG", default=False
    ),
    # Reject misspelled boolean values.
    "VLLM_SPEC_STABLE_RESIDUAL": lambda: _e2_strict_switch(
        "VLLM_SPEC_STABLE_RESIDUAL", default=False
    ),
    # --- Flash-Next fork knobs; see the TYPE_CHECKING block for
    # which of these are compile/capture factors and which are ignored.
    # Compile/capture factors.
    "VLLM_QSA_KV_CLIP_COUNT": lambda: os.getenv("VLLM_QSA_KV_CLIP_COUNT", "").strip(),
    # Capture-configuration factor, fixed in C++; raw passthrough for the hash.
    "MERLIN_TOPK_EXACT_MEDIUM": lambda: os.getenv("MERLIN_TOPK_EXACT_MEDIUM", None),
    # Runtime only; all of the following are in compile_factors.ignored_factors.
    "VLLM_QSA_KV_SCALES": lambda: os.getenv("VLLM_QSA_KV_SCALES", "").strip(),
    "VLLM_E2_GUARD_MODE": lambda: (
        os.getenv("VLLM_E2_GUARD_MODE", "warn").strip().lower()
    ),
    "VLLM_E2_GUARD_INTERVAL": lambda: os.getenv("VLLM_E2_GUARD_INTERVAL", "0.05"),
    "VLLM_E2_GUARD_TRANSPORT": lambda: (
        os.getenv("VLLM_E2_GUARD_TRANSPORT", "mapped").strip().lower()
    ),
    "VLLM_E2_MEMSTATS": lambda: os.getenv("VLLM_E2_MEMSTATS", "0"),
    "VLLM_E2_MEMSTATS_DIR": lambda: os.getenv("VLLM_E2_MEMSTATS_DIR", ""),
    "VLLM_PLE_OFFLOAD_HOME_DEVICE": lambda: os.getenv(
        "VLLM_PLE_OFFLOAD_HOME_DEVICE", ""
    ).strip(),
    "VLLM_PLE_PROGRESS_DEADLINE_S": lambda: float(
        os.getenv("VLLM_PLE_PROGRESS_DEADLINE_S", "").strip() or "300"
    ),
    # PLE offload process: NumPy n-gram ID path (default off). Strict "0"/"1".
    "VLLM_E2_PLE_FAST_IDS": lambda: _e2_strict_switch(
        "VLLM_E2_PLE_FAST_IDS", default=False
    ),
    # Stage-0 runner: early PLE request after prepare_inputs (default off).
    "VLLM_E2_PLE_EARLY_REQUEST": lambda: _e2_strict_switch(
        "VLLM_E2_PLE_EARLY_REQUEST", default=False
    ),
    # PLE offload process: where the table lives. Strict ram|disk.
    "VLLM_E2_PLE_STORE": lambda: _ple_store_value(),
    "VLLM_E2_PLE_WARM": lambda: _e2_strict_switch("VLLM_E2_PLE_WARM", default=True),
    "VLLM_E2_PLE_DISK_THREADS": lambda: _ple_disk_threads(),
    # Seconds between `topk-exact-counters` lines from each GPU worker
    # (0, the default, disables the reader entirely). The line reports the
    # medium-path exact-restart counters; see
    # vllm/models/qwen4_exp/nvidia/ops/topk_counters.py.
    "MERLIN_TOPK_EXACT_COUNTERS_LOG": lambda: int(
        os.getenv("MERLIN_TOPK_EXACT_COUNTERS_LOG", "0")
    ),
    # Fallback slot count for the counter reader on a build WITHOUT the
    # persistent_topk_exact_counters_slots op (pre-tie-determinism). This tree registers
    # that op, so topk_counters.slot_count() never consults the variable here.
    # Raw passthrough: the reader's int(os.getenv(..., "5")) is the parser.
    "MERLIN_TOPK_COUNTER_SLOTS": lambda: os.getenv("MERLIN_TOPK_COUNTER_SLOTS", None),
    # Family-name aliases; raw passthrough, flashnext_family.py is the parser.
    "VLLM_FLASHNEXT_FAMILY_ALIASES": lambda: os.getenv(
        "VLLM_FLASHNEXT_FAMILY_ALIASES", ""
    ),
    # Qwen3.5-lineage names; raw passthrough, qwen35_family.py is the parser.
    "MERLIN_QWEN35_ARCH_ALIASES": lambda: os.getenv("MERLIN_QWEN35_ARCH_ALIASES", ""),
    # Determinism toggles. See the declaration block above for the AOT /
    # capture classification and for why three of these are raw passthroughs.
    # All four are absent from ignored_factors on purpose, so all four are
    # hashed into the compile cache key.
    "MERLIN_FULL_K": lambda: os.getenv(
        "MERLIN_FULL_K", SHIPPED_KERNEL_TOGGLE_DEFAULTS["MERLIN_FULL_K"]
    ),
    "MERLIN_TIE_DET": lambda: os.getenv(
        "MERLIN_TIE_DET", SHIPPED_KERNEL_TOGGLE_DEFAULTS["MERLIN_TIE_DET"]
    ),
    "MERLIN_TIE_RECENT": lambda: os.getenv(
        "MERLIN_TIE_RECENT", SHIPPED_KERNEL_TOGGLE_DEFAULTS["MERLIN_TIE_RECENT"]
    ),
    "MERLIN_QSA_SORT": lambda: _e2_strict_switch(
        "MERLIN_QSA_SORT", default=True, legacy="int"
    ),
    # GDN in_proj_ba on the aux stream; see the declaration block above.
    "VLLM_GDN_BA_AUX_STREAM": lambda: _e2_strict_switch(
        "VLLM_GDN_BA_AUX_STREAM", default=False
    ),
    # See the declaration.
    "VLLM_FLA_PACKED_DECODE_VERIFY_PARITY": lambda: _e2_strict_switch(
        "VLLM_FLA_PACKED_DECODE_VERIFY_PARITY", default=False
    ),
    # persistent_topk kernel switches. Raw passthroughs kept for the hash
    # (the C++ policy is fixed and reads none of them); see the declaration block.
    # MERLIN_TOPK_EXACT_MEDIUM is registered once, earlier in this dict.
    "MERLIN_TOPK_EXACT_DECODE": lambda: os.getenv("MERLIN_TOPK_EXACT_DECODE", None),
    "MERLIN_TOPK_DET_LARGE": lambda: os.getenv("MERLIN_TOPK_DET_LARGE", None),
    # In-kernel canonical emission; see the declaration block. Hashed: it
    # changes the captured sequence.
    "MERLIN_TOPK_SORTED_EMIT": lambda: os.getenv(
        "MERLIN_TOPK_SORTED_EMIT",
        SHIPPED_KERNEL_TOGGLE_DEFAULTS["MERLIN_TOPK_SORTED_EMIT"],
    ),
    # Marlin MoE per-shape block size / thread config. Raw
    # passthrough; marlin_moe_cfg.py parses it.
    "MERLIN_MARLIN_MOE_CFG": lambda: os.getenv("MERLIN_MARLIN_MOE_CFG", None),
    # Blank/unset is preserved as OFF.
    "MERLIN_MARLIN_DENSE_CFG": lambda: os.getenv("MERLIN_MARLIN_DENSE_CFG", None),
    # Static cold-expert offload: raw passthroughs; cold_experts_plan.py parses them.
    "MERLIN_COLD_EXPERTS": lambda: os.getenv("MERLIN_COLD_EXPERTS", None),
    "MERLIN_COLD_EXPERTS_LAYERS": lambda: os.getenv("MERLIN_COLD_EXPERTS_LAYERS", None),
    "MERLIN_COLD_EXPERTS_HOT_FRAC": lambda: os.getenv("MERLIN_COLD_EXPERTS_HOT_FRAC", None),
    # Boot-memory helpers and the Qwen3 parser literal spans: the
    # same parse as their readers (repack_release.py, load_peak_log.py, parser/qwen3.py).
    "MERLIN_REPACK_RELEASE": lambda: os.getenv("MERLIN_REPACK_RELEASE", "1") != "0",
    "MERLIN_LOAD_PEAK_LOG": lambda: os.getenv("MERLIN_LOAD_PEAK_LOG", "0") == "1",
    "MERLIN_QWEN3_LITERAL_SPANS": lambda: os.getenv("MERLIN_QWEN3_LITERAL_SPANS", "1") != "0",
    # Prefix-cache thrash fixes (default ON). Read in cache_thrash.py.
    "VLLM_E2_SCHED_MISS_DEFER_S": lambda: _e2_nonneg_int(
        "VLLM_E2_SCHED_MISS_DEFER_S", default=120
    ),
    "VLLM_E2_SCHED_COHERENT_YIELD": lambda: _e2_strict_switch(
        "VLLM_E2_SCHED_COHERENT_YIELD", default=True
    ),
    # Balanced PP decode microbatches, default off. Read in
    # async_scheduler.py.
    "VLLM_MERLIN_PP_DECODE_BALANCE": lambda: _e2_strict_switch(
        "VLLM_MERLIN_PP_DECODE_BALANCE", default=False
    ),
    # Decode-first step budget, default off. Read in
    # vllm/merlin/decode_first.py.
    "VLLM_MERLIN_DECODE_FIRST": lambda: _e2_strict_switch(
        "VLLM_MERLIN_DECODE_FIRST", default=False
    ),
    # Prefill runs and their tunables, default off. Read in
    # vllm/merlin/prefill_runs.py.
    "VLLM_MERLIN_PREFILL_RUNS": lambda: _e2_strict_switch(
        "VLLM_MERLIN_PREFILL_RUNS", default=False
    ),
    "VLLM_MERLIN_PREFILL_RUNS_CHUNKS": lambda: _e2_positive_int(
        "VLLM_MERLIN_PREFILL_RUNS_CHUNKS", 16
    ),
    "VLLM_MERLIN_PREFILL_RUNS_CHUNK_TOKENS": lambda: _e2_positive_int(
        "VLLM_MERLIN_PREFILL_RUNS_CHUNK_TOKENS", 1024
    ),
    "VLLM_MERLIN_PREFILL_RUNS_DECODE_SHARE": lambda: _prefill_runs_share(),
    "VLLM_MERLIN_PREFILL_RUNS_TICK_COSTS": lambda: _prefill_runs_tick_costs(),
    # Compile-cache fault guard, default on. Read in
    # vllm/merlin/compile_cache_guard.py.
    "VLLM_MERLIN_COMPILE_CACHE_GUARD": lambda: _e2_strict_switch(
        "VLLM_MERLIN_COMPILE_CACHE_GUARD", default=True
    ),
    # QSA profile reserve mode; parsed here, by _qsa_scratch_mode.
    "VLLM_MERLIN_QSA_SCRATCH": lambda: _qsa_scratch_mode(),
    # Same semantics as the former os.environ read: only "0" disables.
    "VLLM_SPEC_PROFILE_SHAPES": lambda: (
        os.getenv("VLLM_SPEC_PROFILE_SHAPES", "1") != "0"
    ),
    "MERLIN_ALLOW_PROFILING_KV_RETENTION": lambda: _e2_strict_switch(
        "MERLIN_ALLOW_PROFILING_KV_RETENTION"
    ),
    # Host-RAM KV tier, GiB per rank (0 = off). Read in vllm/merlin/kv_host_tier.py.
    "MERLIN_KV_HOST_TIER_GIB": lambda: _e2_nonneg_int("MERLIN_KV_HOST_TIER_GIB"),
    # Host admission policy flag; None when unset.
    "MERLIN_KV_HOST_TIER__ADMISSION": kv_host_tier_admission,
    "MERLIN_KV_HOST_TIER_CENSUS": lambda: _e2_strict_switch(
        "MERLIN_KV_HOST_TIER_CENSUS", default=False
    ),
    # Read in vllm/v1/core/fgpc.py, kv_host_tier_census.py, kv_host_tier_units.py.
    "MERLIN_KV_TIER_DIAG": lambda: _e2_strict_switch("MERLIN_KV_TIER_DIAG", default=False),
    "MERLIN_KV_HOST_TIER_FAIL_LOADS": kv_host_tier_fail_loads,
    "MERLIN_KV_HOST_TIER_FAIL_STORES": kv_host_tier_fail_stores,
    # Host-tier write-back pacing (copy credits P, admission escape steps,
    # the restores-in-flight cap N_restore, units assembling at once). Read
    # in vllm/merlin/kv_host_tier_writeback.py (GATE_ENVS) and
    # kv_host_tier_retention.RestoresInFlight; the defaults here are the
    # ones they apply.
    "MERLIN_KV_HOST_TIER__COPY_CREDITS": lambda: _e2_positive_int(
        "MERLIN_KV_HOST_TIER__COPY_CREDITS", 32
    ),
    "MERLIN_KV_HOST_TIER__ESCAPE_STEPS": lambda: _e2_positive_int(
        "MERLIN_KV_HOST_TIER__ESCAPE_STEPS", 8
    ),
    "MERLIN_KV_HOST_TIER__N_RESTORE": lambda: _e2_positive_int(
        "MERLIN_KV_HOST_TIER__N_RESTORE", 2
    ),
    "MERLIN_KV_HOST_TIER__MAX_ASSEMBLING": lambda: _e2_positive_int(
        "MERLIN_KV_HOST_TIER__MAX_ASSEMBLING", 2
    ),
    # Per-direction credits. The demotion rows default to the
    # shared COPY_CREDITS; the byte cap is MiB per rank, 0 = rows only.
    "MERLIN_KV_HOST_TIER__COPY_CREDITS_D": lambda: _e2_positive_int(
        "MERLIN_KV_HOST_TIER__COPY_CREDITS_D",
        _e2_positive_int("MERLIN_KV_HOST_TIER__COPY_CREDITS", 32),
    ),
    "MERLIN_KV_HOST_TIER__COPY_BYTES_D_MIB": lambda: _e2_nonneg_int(
        "MERLIN_KV_HOST_TIER__COPY_BYTES_D_MIB", 0
    ),
    "MERLIN_KV_HOST_TIER__MAX_PENDING_STORE_BLOCKS": lambda: _e2_nonneg_int(
        "MERLIN_KV_HOST_TIER__MAX_PENDING_STORE_BLOCKS", 0
    ),
    "MERLIN_KV_HOST_TIER__FALLBACK_BOUNDARY": lambda: _e2_strict_switch(
        "MERLIN_KV_HOST_TIER__FALLBACK_BOUNDARY", default=False
    ),
}


MERLIN_IGNORED: frozenset[str] = frozenset(
    {
    # API-server admission limits; no compiled or captured operations.
    "MERLIN_GENERATIVE_SCORING_MAX_ITEMS",
    "MERLIN_GENERATIVE_SCORING_MAX_TOKENS",
    # Eager prompt scoring after forward; no compiled or captured operations.
    "VLLM_MERLIN_PROMPT_LOGPROBS_CHUNK_SIZE",
    # API-server /health watchdog; no compiled or captured operations.
    "VLLM_MERLIN_HEALTH_STALL_S",
    # API-server chat rendering only (sanitize_control_tokens); no compiled or
    # captured operations.
    "VLLM_MERLIN_EXTRA_CONTROL_TOKENS",
    "VLLM_MERLIN_PROMPT_LOGPROBS_TENSOR_BUDGET_MIB",
    # AOT: no -- row slicing is eager, outside compiled decoder regions.
    # CAPTURE: no -- encoder graphs omit the grid and keep plain execution.
    "VLLM_MERLIN_VISION_CHUNK_ROWS",
    "VLLM_MERLIN_VISION_CHUNK_MIN_PIXELS",
    # AOT: no -- eager tower dispatch; compiled vision blocks keep the plain path.
    # CAPTURE: no -- encoder graphs omit the grid and keep plain execution.
    "VLLM_MERLIN_VISION_PEAK",
    # AOT: no -- buffer ownership only. CAPTURE: the drafter's graphs record the shared
    # pointer instead of a second buffer; same kernels and shapes.
    "MERLIN_SHARE_DRAFT_WORKSPACE",
    # AOT: no -- read only by the FlashInfer metadata builder (cudagraph-support
    # class and reorder threshold), outside compiled regions. CAPTURE: yes, like
    # VLLM_DRAFT_BOUNDED_PLAN.
    "MERLIN_MOE_UNIFORM_VERIFY",
    # AOT: no -- read by the FlashInfer metadata builder and the V2 graph
    # managers, outside compiled regions. CAPTURE: yes, so the ON value is added
    # back as a compile factor by add_merlin_compile_factors; unset and 0 leave
    # the key exactly as it is without the knob (knob-off parity).
    "MERLIN_FLASHINFER_WORKLOAD_TILE",
    # AOT: no -- eager tower path only. CAPTURE: no -- encoder graphs keep the plain
    # path.
    "VLLM_MERLIN_VISION_ROTARY_INPLACE",
    # --- Flash-Next fork knobs that are runtime-only ------
    # Registering a name in environment_variables makes it a compile
    # factor by default (this function is a deny-list), so every fork knob
    # that cannot change a compiled or captured artifact is listed here
    # with its reason. Hashing them would invalidate the whole AOT cache on
    # a log path, a timeout or a guard-mode flip.
    #
    # Static QSA K/V scales: fill_()ed into existing fp32 device buffers
    # after load and before capture; the kernel reads them with
    # tl.load(k_scale_ptr), never as a constexpr.
    "VLLM_QSA_KV_SCALES",
    # Guard telemetry: counter() returns a real [3] int64 view in every
    # mode, so the Triton arguments and the captured graph are identical;
    # the mode only gates the side reader thread and the check_pending
    # raise, and the interval is that thread's poll cadence.
    "VLLM_E2_GUARD_MODE",
    "VLLM_E2_GUARD_INTERVAL",
    # Transport only moves the counter buffer to mapped host memory: the
    # guarded kernels, their arguments, Triton specialisation (16-byte
    # aligned pointer either way) and the captured graph topology are
    # unchanged; a clean step touches no counter. Not a compile factor.
    "VLLM_E2_GUARD_TRANSPORT",
    # Debug telemetry cadence and output locations.
    "VLLM_E2_MEMSTATS",
    "VLLM_E2_MEMSTATS_DIR",
    # Scheduler-side admission order and yield policy; the scheduler is never
    # compiled and the worker sees ordinary block tables either way, so an A/B
    # flip reuses one AOT cache.
    "VLLM_E2_SCHED_MISS_DEFER_S",
    "VLLM_E2_SCHED_COHERENT_YIELD",
    # Balanced PP decode microbatches: which decode rows a step
    # schedules; shapes stay within the captured batch sizes either way.
    "VLLM_MERLIN_PP_DECODE_BALANCE",
    # Decode-first step budget: how many tokens a prefill chunk takes
    # this step; shapes stay within the captured batch sizes either way.
    "VLLM_MERLIN_DECODE_FIRST",
    # Prefill runs: which rows a step schedules and how large a prefill chunk
    # is (never above --max-num-batched-tokens); shapes stay within the
    # configured budget either way.
    "VLLM_MERLIN_PREFILL_RUNS",
    "VLLM_MERLIN_PREFILL_RUNS_CHUNKS",
    "VLLM_MERLIN_PREFILL_RUNS_CHUNK_TOKENS",
    "VLLM_MERLIN_PREFILL_RUNS_DECODE_SHARE",
    "VLLM_MERLIN_PREFILL_RUNS_TICK_COSTS",
    # Compile-cache fault guard: where the cache lives and whether it is
    # trusted, never what is compiled; must not move the AOT cache key.
    "VLLM_MERLIN_COMPILE_CACHE_GUARD",
    # QSA profile reserve: eager allocations in the dummy run and the
    # eager selection op; same kernels, shapes and strides; never used while a
    # graph is captured.
    "VLLM_MERLIN_QSA_SCRATCH",
    # Startup profiling only: extra dummy forwards before KV sizing, and the
    # severity of a post-profiling teardown check; neither touches a compiled
    # or captured artifact.
    "VLLM_SPEC_PROFILE_SHAPES",
    "MERLIN_ALLOW_PROFILING_KV_RETENTION",
    # Host-RAM KV tier size: adds a scheduler/worker-side connector whose
    # per-layer hooks are no-ops; the forward, its compiled artifacts and
    # captured graphs are the same with the tier on or off.
    "MERLIN_KV_HOST_TIER_GIB",
    # Host admission policy flag (scheduler-side eviction only).
    "MERLIN_KV_HOST_TIER__ADMISSION",
    # Debug census of the tier (scheduler-side bookkeeping and log lines).
    "MERLIN_KV_HOST_TIER_CENSUS",
    # Debug diagnosis log lines of the KV tiers (scheduler-side only).
    "MERLIN_KV_TIER_DIAG",
    # Debug copy-thread errors only; no compiled/captured operations.
    "MERLIN_KV_HOST_TIER_FAIL_LOADS",
    "MERLIN_KV_HOST_TIER_FAIL_STORES",
    # Write-back pacing: scheduler-side copy issue and admission waits only.
    "MERLIN_KV_HOST_TIER__COPY_CREDITS",
    "MERLIN_KV_HOST_TIER__ESCAPE_STEPS",
    "MERLIN_KV_HOST_TIER__N_RESTORE",
    "MERLIN_KV_HOST_TIER__MAX_ASSEMBLING",
    "MERLIN_KV_HOST_TIER__COPY_CREDITS_D",
    "MERLIN_KV_HOST_TIER__COPY_BYTES_D_MIB",
    "MERLIN_KV_HOST_TIER__MAX_PENDING_STORE_BLOCKS",
    # Scheduler-side host eviction order only.
    "MERLIN_KV_HOST_TIER__FALLBACK_BOUNDARY",
    # Placement only: rewrites CUDA_VISIBLE_DEVICES for the PLE offload
    # child, and CUDA_VISIBLE_DEVICES is already ignored above.
    "VLLM_PLE_OFFLOAD_HOME_DEVICE",
    # Timeouts and deadlines, in seconds. VLLM_PLE_OFFLOAD_READY_TIMEOUT
    # is not a compile factor: a launcher may set it (e.g. to 1800),
    # and hashing it would make every boot that changed it
    # miss the AOT cache for no reason.
    "VLLM_PLE_OFFLOAD_READY_TIMEOUT",
    "VLLM_PLE_PROGRESS_DEADLINE_S",
    # MERLIN_TOPK_EXACT_MEDIUM is deliberately NOT here: it is a capture
    # configuration factor and is hashed conservatively.
    # PLE offload process only (no CUDA context there): selects how the
    # host computes n-gram row IDs, not what any kernel or graph runs.
    "VLLM_E2_PLE_FAST_IDS",
    # Eager host ordering of the real-batch PLE request only: capture and
    # dummy forwards keep the late issue, so the compiled artifact and
    # every captured graph are the same with it on or off.
    "VLLM_E2_PLE_EARLY_REQUEST",
    # PLE offload process only: where its table bytes come from, never what
    # any kernel or graph runs.
    "VLLM_E2_PLE_STORE",
    "VLLM_E2_PLE_WARM",
    "VLLM_E2_PLE_DISK_THREADS",
    # Eager allocation size of the QSA prefill logits (rounded up, reused
    # across chunks). The op runs outside capture, the view handed to the
    # kernel has the same shape, strides and dtype, so no compiled artifact
    # or captured graph depends on it.
    "VLLM_E2_QSA_LOGITS_GRANULE_MB",
    # Step-keyed draws: AOT: no (the rejection
    # kernels and sample_draft are outside the compiled regions). CAPTURE: yes
    # (the FULL draft graphs record the STEP_KEYED draft sampler), but the flag
    # is read once at init, before capture, and graphs are never persisted, so
    # no cross-process artifact depends on it; hashing it would force an AOT
    # miss on every configuration for an RNG-keying A/B.
    "VLLM_SPEC_STEP_KEYED_RNG",
    # Init-frozen Triton constexpr, like RNG.
    # JIT specialization keys it; graphs are not persisted and AOT is unaffected.
    "VLLM_SPEC_STABLE_RESIDUAL",
    # Logging cadence for an already-compiled kernel's counters: it gates a
    # host-side timer and a snapshot op, and is read by no compiled code, so
    # hashing it would invalidate every compile cache for a log interval.
    "MERLIN_TOPK_EXACT_COUNTERS_LOG",
    # Host-side reader fallback (buffer length for a snapshot op outside
    # capture); dead on a build that answers the _slots query.
    "MERLIN_TOPK_COUNTER_SLOTS",
    # Marlin MoE per-shape block size / thread config. AOT: no. The
    # choice is made in the Python body of fused_marlin_moe, which runs inside
    # the opaque vllm.moe_forward custom op (moe_runner.py: "opaque custom ops is
    # a load-bearing assumption"), so the Inductor/AOT artifact contains no
    # trace of it, and the kernels it selects among are all compiled into the
    # extension already. CAPTURE: yes -- a cudagraph captured for a given M bakes
    # in the block size and launch config -- but the value is fixed for the
    # process, read on every call, and cudagraphs are never persisted, so there
    # is no cross-process artifact to key. Hashing it would only move the
    # release AOT cache key (even when unset) for no protection. Same argument
    # as VLLM_SPEC_STEP_KEYED_RNG above.
    "MERLIN_MARLIN_MOE_CFG",
    # Read inside opaque marlin_gemm;
    # only native launch parameters change, not persisted compiled code.
    # CUDA graphs are process-local; changing the table requires a restart.
    "MERLIN_MARLIN_DENSE_CFG",
    # Static cold-expert offload. AOT: no -- the expert tensors are read
    # only inside the opaque vllm.moe_forward custom op, which Inductor does not
    # trace, and the split changes where expert bytes live, not any compiled
    # graph. CAPTURE: the hot/cold tensors and maps are fixed before capture and
    # graphs are never persisted. Hashing these would give every placement arm
    # its own compile cache (the cross-cache class), which is what this
    # placement path exists to avoid.
    "MERLIN_COLD_EXPERTS",
    "MERLIN_COLD_EXPERTS_LAYERS",
    "MERLIN_COLD_EXPERTS_HOT_FRAC",
    # Repack release: synchronise + empty_cache after a module's
    # post-load processing; frees cached allocator blocks only, before any
    # compile or capture. Weights, kernels and graphs are unchanged.
    "MERLIN_REPACK_RELEASE",
    # Load-peak log: a diagnostic that brackets post-load processing
    # with memory reads and logs them; nothing compiled or captured reads it.
    "MERLIN_LOAD_PEAK_LOG",
    # Qwen3 parser literal spans: API-server output parsing only; the
    # model forward never sees it.
    "MERLIN_QWEN3_LITERAL_SPANS",
    # --- 27B lane knobs: ignored factors and their reasons. ---
    # RecoverSSM commit order (after vs before the MTP proposal): eager Python
    # call order after sampling; nothing compiled or captured depends on it.
    "VLLM_RECOVERSSM_COMMIT_AFTER_PROPOSE",
    # PP receive event split: stream order
    # after sampling; nothing compiled or captured depends on it.
    "VLLM_RECOVERSSM_PP_SPLIT_RECV",
    # RecoverSSM commit execution mode: read only after sampling, outside the
    # compiled forward and its capture; the commit graphs it selects are
    # in-process and bit-identical to the eager chain (see its declaration).
    "VLLM_RECOVERSSM_COMMIT_GRAPH",
    # Debug-only ownership assertion: a host-side check after the commit.
    "VLLM_RECOVERSSM_CHECK_OWNERSHIP",
    # GDN metadata host trims: host-side builder work only. The staged
    # buffers keep their addresses and get the same values, and nothing
    # inside the compiled forward or a captured graph reads the flags.
    "VLLM_GDN_SHARED_ALIGN_INDEX",
    "VLLM_GDN_DECODE_FASTPATH",
    # Selects a Triton launch
    # inside the opaque qwen_gdn_attention_core op; no compiled artefact
    # contains it, and registering it must not move the release AOT cache key.
    "VLLM_FLA_PACKED_DECODE_VERIFY_PARITY",
    # MTP draft bounded plan: decides on the host whether a draft build plans
    # FlashInfer from CPU bounds (and writes the page table with a Triton kernel
    # outside capture); nothing compiled or captured reads either flag, and
    # registering them must not move the release AOT cache key.
    "VLLM_DRAFT_BOUNDED_PLAN",
    "VLLM_DRAFT_BOUNDED_PLAN_VERIFY",
    # Target seq_lens async snapshot: host-side pinned D2H + event before the drafts; read by the
    # FlashInfer metadata build on the host only -- nothing compiled or captured reads it, and it must not
    # move the release AOT cache key (flags off/on compiled separate artefacts -> token drift).
    "VLLM_TARGET_SEQ_LENS_ASYNC",
    "VLLM_TARGET_SEQ_LENS_ASYNC_VERIFY",
    # Family-name aliases: resolved to architecture / model_type names at
    # config and registry time; no compiled or captured artifact depends on it.
    "VLLM_FLASHNEXT_FAMILY_ALIASES",
    "MERLIN_QWEN35_ARCH_ALIASES",
    # ------------------------------------------------------------------
    }
)


def add_merlin_compile_factors(factors: dict[str, object]) -> None:
    """Add the fork's derived compile factors to ``factors`` (in place)."""
    # Workload tile: a conservative capture factor only when on.
    if MERLIN_ENV["MERLIN_FLASHINFER_WORKLOAD_TILE"]():
        factors["MERLIN_FLASHINFER_WORKLOAD_TILE"] = True


# Retired MERLIN_* knobs a published release read and this one ignores
# (name -> reason). ``validate_environ`` names them retired instead of
# suggesting a misspelling. None in this release. The launcher keeps its own
# list of retired names it warns about before a boot; that list may hold
# VLLM_* names too, which the engine reports through vLLM's own
# unknown-variable check.
MERLIN_RETIRED: dict[str, str] = {}


# MERLIN_* names read by the Merlin command-line tools (launcher, setup,
# release scripts, container entrypoint), never by the engine. They may sit in
# the same shell as an engine boot, so the engine neither warns about them nor
# calls them misspelt. A CPU test checks that
# every name the tools read or the docs name is here or a registered knob.
MERLIN_TOOL_ONLY: frozenset[str] = frozenset({
    "MERLIN_ALLOW_SERVER_DEV_MODE",
    "MERLIN_CHECKPOINT_DIR",
    "MERLIN_EMULATE_VRAM_CLASS",
    "MERLIN_HOME",
    "MERLIN_LIBCUDA",
    "MERLIN_LOG_DIR",
    "MERLIN_MODEL",
    "MERLIN_PLE_PACK_DIR",
    "MERLIN_PLE_TEST_TABLE",
    "MERLIN_PROFILE",
    "MERLIN_PYTHON",
    "MERLIN_READY_BANNER",
    "MERLIN_RELEASE_VERSION",
    "MERLIN_SKIP_PORT_CHECK",
    "MERLIN_SKIP_RAM_CHECK",
    "MERLIN_UV_VERSION",
    "MERLIN_WHEEL",
    "MERLIN_WHEEL_URL",
})


def unknown_merlin_env_warning(name: str) -> str | None:
    """The boot warning for a MERLIN_* name no getter registers, or None for a
    name only the Merlin tools read (``MERLIN_TOOL_ONLY``)."""
    if name in MERLIN_TOOL_ONLY:
        return None
    why = MERLIN_RETIRED.get(name)
    if why is not None:
        return f"Retired Merlin environment variable detected: {name} is ignored ({why})"
    return (
        f"Unknown Merlin environment variable detected: {name} "
        "(not a registered knob; check the spelling)"
    )


def read_registered(name: str) -> Any:
    """Resolve one registered fork knob through its getter (no module cache)."""
    return MERLIN_ENV[name]()
