# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Fork startup guards that fail closed on a graph profile we never qualified.

These run at model construction, on every rank, for the target model and for
the MTP drafter, before profiling or capture.

Compile-profile guard: this tree restores torch.compile on the
NVIDIA Qwen4Exp path (reverting vLLM #55272) and removes Qwen4Exp from
``DEFAULT_BREAKABLE_CUDAGRAPH_ARCHITECTURES``. The ONE qualified profile is the
production one:

  * ``VLLM_USE_BREAKABLE_CUDAGRAPH`` explicitly pinned to exactly ``0``, and
    resolved OFF;
  * ``CompilationMode.VLLM_COMPILE``;
  * ``CUDAGraphMode.FULL_AND_PIECEWISE``;
  * every Qwen4Exp request-dependent custom op in ``splitting_ops``
    (:data:`REQUIRED_SPLITTING_OPS`);
  * a live ``@support_torch_compile`` submodule
    (:func:`assert_compiled_submodule`), which is the property
    ``has_compiled_submodule`` (vllm/v1/worker/gpu/cudagraph_utils.py) tests at capture.

Anything else -- eager, NONE + FULL, FULL_DECODE_ONLY, breakable PIECEWISE --
fails loud. There is no opt-out switch; the offline calibration harness
(tools/qsa_calibrate.py) replaces these functions inside its own worker
processes instead.

Timing limit: this runs AFTER ``VllmConfig.__post_init__`` resolved the
profile. It stops the boot before profiling and capture; it cannot recover
the originally requested profile if the config layer rewrote it.
"""

import os

from vllm import flashnext_family
from vllm.logger import init_logger

logger = init_logger(__name__)

BREAKABLE_ENV = "VLLM_USE_BREAKABLE_CUDAGRAPH"

#: The qualified profile (the compile-profile guard).
REQUIRED_COMPILATION_MODE_NAME = "VLLM_COMPILE"
REQUIRED_CUDAGRAPH_MODE_NAME = "FULL_AND_PIECEWISE"
#: The only raw VLLM_USE_BREAKABLE_CUDAGRAPH value accepted as the production
#: pin. Resolution semantics stay upstream's (_upstream_truth).
PRODUCTION_BREAKABLE_PIN = "0"

REQUIRED_SPLITTING_OPS = (
    "vllm::qwen4_exp_nv_qsa_with_output",
    "vllm::qwen4_exp_nv_ple_short_conv",
    "vllm::qwen4_exp_compute_ple_ngram_ids",
    "vllm::qwen_gdn_attention_core",
    "vllm::qwen_gdn_attention_core_fused_norm_packed",
    "vllm::ple_offload_wait_copy",
)

#: Upstream auto-enables breakable cudagraphs for these; this tree
#: does not (vllm/config/vllm.py).
OUR_AUTO_BREAKABLE_ARCHITECTURES = frozenset(flashnext_family.archs())


def _upstream_truth(raw: str | None) -> bool:
    """Upstream's exact semantics for the breakable switch.

    vllm/envs.py reads it as ``os.environ.get(..., "0") == "1"``. Nothing else is on.
    """
    return raw == "1"


def breakable_resolved() -> bool:
    """Whether breakable cudagraphs are on, as the engine will see it.

    The raw environment and the (cached) envs accessor are two readings of one
    source; either saying "on" means on. No exception is swallowed.
    """
    if _upstream_truth(os.environ.get(BREAKABLE_ENV)):
        return True
    import vllm.envs as envs

    return bool(envs.VLLM_USE_BREAKABLE_CUDAGRAPH)


def _name(value) -> str:
    return getattr(value, "name", str(value))


def _profile(
    vllm_config, role: str = "target"
) -> tuple[list[str], object, object, list[str] | None]:
    """Resolved (architectures, mode, cudagraph mode, splitting ops).

    For ``role="draft"`` the architectures are the DRAFT model's
    (``speculative_config.draft_model_config``): the drafter is constructed
    with the target's VllmConfig, whose model_config names the target.
    The graph profile is the engine's single CompilationConfig for both roles.
    """
    model_config = getattr(vllm_config, "model_config", None)
    if role == "draft":
        spec = getattr(vllm_config, "speculative_config", None)
        draft_config = getattr(spec, "draft_model_config", None)
        if draft_config is not None:
            model_config = draft_config
    if model_config is None:
        raise RuntimeError("e2-guard: vllm_config has no model_config to check")
    compilation_config = getattr(vllm_config, "compilation_config", None)
    if compilation_config is None:
        raise RuntimeError("e2-guard: vllm_config has no compilation_config to check")
    architectures = list(getattr(model_config, "architectures", None) or [])
    return (
        architectures,
        compilation_config.mode,
        compilation_config.cudagraph_mode,
        getattr(compilation_config, "splitting_ops", None),
    )


def assert_graph_profile_is_executable(vllm_config, *, role: str = "target") -> None:
    """Require the qualified profile: VLLM_COMPILE + FULL_AND_PIECEWISE.

    ``role`` is ``"target"`` or ``"draft"``; both are checked, because
    ``Qwen4ExpMTP`` builds ``Qwen4ExpMultiTokenPredictor`` rather than
    ``Qwen4ExpForCausalLM``, from a config rebuilt in ``mtp.py``.
    """
    architectures, mode, cudagraph_mode, splitting_ops = _profile(vllm_config, role)
    detail = (
        f"architectures={architectures} breakable={os.environ.get(BREAKABLE_ENV)} "
        f"compilation_mode={_name(mode)} cudagraph_mode={_name(cudagraph_mode)}"
    )

    # 1. The pin must be explicit, so a future upstream merge that re-adds us
    #    to DEFAULT_BREAKABLE_CUDAGRAPH_ARCHITECTURES cannot flip it silently.
    raw_breakable = os.environ.get(BREAKABLE_ENV)
    if raw_breakable is None:
        raise RuntimeError(
            f"e2-guard[{role}]: {BREAKABLE_ENV} is not set. Pin it in the "
            f"launcher: export {BREAKABLE_ENV}=0."
        )

    # 2. Breakable capture is not the qualified identity.
    if raw_breakable == "1" or breakable_resolved():
        raise RuntimeError(
            f"e2-guard[{role}]: {BREAKABLE_ENV} resolved ON ({detail}). The "
            "qualified profile is breakable-off VLLM_COMPILE + "
            f"FULL_AND_PIECEWISE. Export {BREAKABLE_ENV}=0."
        )

    if raw_breakable != PRODUCTION_BREAKABLE_PIN:
        raise RuntimeError(
            f"e2-guard[{role}]: {BREAKABLE_ENV}={raw_breakable!r} is not the "
            f"production pin; it must be exactly {PRODUCTION_BREAKABLE_PIN!r}. "
            "Upstream would read it as off, but the launch profile does not "
            f"say what was meant. Export {BREAKABLE_ENV}=0."
        )

    # 3. The profile itself: mode AND cudagraph mode, by name.
    if (
        _name(mode) != REQUIRED_COMPILATION_MODE_NAME
        or _name(cudagraph_mode) != REQUIRED_CUDAGRAPH_MODE_NAME
    ):
        raise RuntimeError(
            f"e2-guard[{role}]: resolved compilation_mode={_name(mode)} "
            f"cudagraph_mode={_name(cudagraph_mode)}; the qualified profile is "
            f"compilation_mode={REQUIRED_COMPILATION_MODE_NAME} "
            f"cudagraph_mode={REQUIRED_CUDAGRAPH_MODE_NAME} (compile-profile guard). Check "
            "-O / --compilation-config / --enforce-eager / TORCH_COMPILE_DISABLE."
        )

    # 4. The graph breaks that make PIECEWISE correct on this model.
    if splitting_ops is None:
        raise RuntimeError(
            f"e2-guard[{role}]: compilation_config.splitting_ops is unresolved; "
            "cannot confirm the Qwen4Exp graph breaks."
        )
    missing = [op for op in REQUIRED_SPLITTING_OPS if op not in splitting_ops]
    if missing:
        raise RuntimeError(
            f"e2-guard[{role}]: splitting_ops is missing {missing}. Each of "
            "these reads per-step request metadata (or forks/joins a side "
            "stream) and would be frozen into a piecewise CUDA graph."
        )

    logger.info(
        "e2-guard[%s]: architectures=%s breakable=OFF compilation_mode=%s "
        "cudagraph_mode=%s splitting_ops=+%d required. Recorded, not "
        "certified -- this checks startup state, not qualification.",
        role,
        architectures,
        _name(mode),
        _name(cudagraph_mode),
        len(REQUIRED_SPLITTING_OPS),
    )


def assert_compiled_submodule(module, vllm_config, *, role: str = "target") -> None:
    """Require a live ``@support_torch_compile`` wrapper on ``module``.

    This is the property ``has_compiled_submodule()`` (gpu/cudagraph_utils.py)
    checks before PIECEWISE capture. Checking it at
    construction turns a mid-capture failure into a startup one.
    """
    from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper

    compiled = isinstance(module, TorchCompileWithNoGuardsWrapper) and not getattr(
        module, "do_not_compile", True
    )
    if compiled:
        return
    detail = (
        f"{type(module).__name__} is_wrapper="
        f"{isinstance(module, TorchCompileWithNoGuardsWrapper)} "
        f"do_not_compile={getattr(module, 'do_not_compile', '<absent>')}"
    )
    raise RuntimeError(
        f"e2-guard[{role}]: {detail}: no live torch.compile wrapper, so "
        "PIECEWISE capture would raise in has_compiled_submodule "
        "(gpu/cudagraph_utils.py). The "
        "@support_torch_compile decorator (compile-profile guard) is missing or disabled."
    )


#: Target architectures whose V2 capture is checked. The construction guard
#: above lives in the NVIDIA Qwen4Exp model; this tree targets sm_86 only.
QWEN4EXP_TARGET_ARCHITECTURES = frozenset(flashnext_family.target_archs())

EXPECTED_RESOLVED_MODES = {
    "target": REQUIRED_CUDAGRAPH_MODE_NAME,
    "draft.prefill_cudagraph_manager": REQUIRED_CUDAGRAPH_MODE_NAME,
    "draft.decode_cudagraph_manager": "FULL_DECODE_ONLY",
    "draft.cudagraph_manager": "FULL_DECODE_ONLY",
    # DFlash/DFlash2 (DFlashSpeculator.init_cudagraph_manager): FULL decode
    # graphs, or NONE when the draft attention cannot run full graphs.
    "draft.query_cudagraph_manager": "FULL_DECODE_ONLY",
}
_DRAFT_MANAGER_ATTRS = (
    "prefill_cudagraph_manager",
    "decode_cudagraph_manager",
    "cudagraph_manager",
    "query_cudagraph_manager",
)


def assert_resolved_cudagraph_modes(
    vllm_config, target_manager, speculator=None
) -> None:
    """Fail loud if the V2 runner is about to capture a downgraded mode.

    Called from ``GPUModelRunner.capture_model`` (``vllm/v1/worker/gpu``)
    before any capture, including the ``profile_only`` capture and the
    "nothing to capture" early return, so a resolver downgrade to NONE cannot
    skip it. Applies only to a Qwen4Exp target; any other model is untouched.
    """
    model_config = getattr(vllm_config, "model_config", None)
    architectures = list(getattr(model_config, "architectures", None) or [])
    if not QWEN4EXP_TARGET_ARCHITECTURES & set(architectures):
        return

    observed: dict[str, str] = {}
    if target_manager is None:
        observed["target"] = "<no manager>"
    else:
        observed["target"] = _name(getattr(target_manager, "cudagraph_mode", None))
    if speculator is not None:
        found = False
        for attr in _DRAFT_MANAGER_ATTRS:
            if not hasattr(speculator, attr):
                continue
            found = True
            manager = getattr(speculator, attr)
            observed[f"draft.{attr}"] = (
                "<no manager>"
                if manager is None
                else _name(getattr(manager, "cudagraph_mode", None))
            )
        if not found:
            observed["draft"] = f"<unrecognised speculator {type(speculator).__name__}>"

    detail = " ".join(f"{key}={value}" for key, value in observed.items())
    wrong = {
        key: value
        for key, value in observed.items()
        if EXPECTED_RESOLVED_MODES.get(key) != value
    }
    if wrong:
        expected = {key: EXPECTED_RESOLVED_MODES.get(key, "<none>") for key in wrong}
        raise RuntimeError(
            f"e2-guard[runtime]: resolved cudagraph modes {wrong} differ from "
            f"the qualified profile {expected} ({detail}). The V2 runner's "
            "resolver downgraded the graph identity after the construction "
            "guard passed (attention CG support, spec-decode uniform batch, "
            "or no compiled submodule); see the resolver's warning above."
        )
    logger.info(
        "e2-guard[runtime]: resolved cudagraph modes before capture: %s.", detail
    )


UPSTREAM_UVA_ENV = "VLLM_PLE_CPU_OFFLOAD"


def check_upstream_uva_env_against_pull() -> None:
    """Fail loud if the old name would turn upstream's UVA path on.

    ``VLLM_PLE_CPU_OFFLOAD`` may be unset (the fork then supplies
    ``cpu_offload=False`` as the Engram default) or exactly ``"0"``. ``"1"``
    would enable upstream's pinned-host UVA backend beside ours: a double
    offload. Any other value is rejected rather than interpreted.
    """
    raw = os.environ.get(UPSTREAM_UVA_ENV)
    if raw is None or raw == "0":
        return
    if raw == "1":
        raise RuntimeError(
            f"e2-guard: PLE pull transport and {UPSTREAM_UVA_ENV}=1. On "
            f"upstream vLLM {UPSTREAM_UVA_ENV} is UPSTREAM's switch for the "
            "pinned-host UVA PLE backend (default on), not the E2 pull "
            "transport, so this launch asks for both transports at once. Unset "
            f"{UPSTREAM_UVA_ENV} or export {UPSTREAM_UVA_ENV}=0 "
            "(the pull transport replaces that switch)."
        )
    raise ValueError(
        f"e2-guard: PLE pull transport and {UPSTREAM_UVA_ENV}={raw!r}. "
        f'With the pull transport, {UPSTREAM_UVA_ENV} must be unset or '
        'exactly "0"; any other value is refused rather than guessed.'
    )


def pull_transport_cpu_offload_default() -> bool:
    """The Engram ``cpu_offload`` default, as this fork resolves it: the
    old-name check above, then ``False`` -- upstream's UVA backend is never
    the default beside the pull transport.
    """
    check_upstream_uva_env_against_pull()
    return False


def verify_engram_against_pull(engram_config) -> None:
    """Refuse an explicit ``cpu_offload=True`` beside the pull transport.

    ``--engram-config '{"cpu_offload": true}'`` takes precedence over every
    default, so it is checked on the resolved object, not the environment.
    """
    check_upstream_uva_env_against_pull()
    if engram_config is not None and bool(engram_config.cpu_offload):
        raise RuntimeError(
            "e2-guard: PLE pull transport but the resolved EngramConfig "
            "has cpu_offload=True (upstream's pinned-host UVA PLE backend). "
            "Both transports would own the PLE table. Drop cpu_offload from "
            "--engram-config (or set it false)."
        )


#: The effective PLE mode, as the guard names it (the pull transport is the
#: only one).
PLE_MODE_PULL = "pull"


def resolve_ple_mode(vllm_config) -> str:
    """The PLE transport this process will build (pull), or an error if the
    RESOLVED ``engram_config.cpu_offload`` asks for upstream's UVA backend as
    well. There is no fallback from one transport to another.
    """
    check_upstream_uva_env_against_pull()
    engram_config = getattr(vllm_config, "engram_config", None)
    if engram_config is not None and bool(engram_config.cpu_offload):
        raise RuntimeError(
            "e2-guard: PLE pull transport with a resolved Engram "
            "cpu_offload=True: the pull transport and upstream's UVA "
            "backend would both own the PLE table."
        )
    return PLE_MODE_PULL


def _role_ple_layer_ids(vllm_config, role: str) -> list[int]:
    """The 1-based PLE layer ids the role's own decoder layers carry.

    ``role="target"``: ``model_config.hf_text_config.ple_layer_ids``.
    ``role="draft"``: the drafter is constructed with the target's VllmConfig
    (as for the architectures), so the target's
    model_config names the TARGET's PLE layers. The drafter's decoder layers
    are ``num_hidden_layers + i`` for ``i < mtp_num_hidden_layers``
    (``Qwen4ExpMultiTokenPredictor``), and a layer builds a PLE block only if
    ``layer_idx + 1`` is in ``ple_layer_ids`` (``Qwen4ExpDecoderLayer``). The
    draft line reports exactly that intersection, read from
    ``speculative_config.draft_model_config`` -- empty for this checkpoint,
    whose ids must name backbone layers (``Qwen4ExpTextConfig``).
    """
    model_config = getattr(vllm_config, "model_config", None)
    if role == "draft":
        spec = getattr(vllm_config, "speculative_config", None)
        draft_config = getattr(spec, "draft_model_config", None)
        if draft_config is not None:
            model_config = draft_config
    text_config = getattr(model_config, "hf_text_config", None)
    ids = list(getattr(text_config, "ple_layer_ids", None) or [])
    if role != "draft":
        return ids
    num_hidden_layers = getattr(text_config, "num_hidden_layers", None)
    if num_hidden_layers is None:
        return ids
    num_mtp = int(getattr(text_config, "mtp_num_hidden_layers", 1))
    own = range(int(num_hidden_layers) + 1, int(num_hidden_layers) + num_mtp + 1)
    return [layer_id for layer_id in ids if layer_id in own]


def assert_ple_mode(vllm_config, *, role: str = "target") -> str:
    """Startup guard: resolve, assert and log the effective PLE mode.

    Runs beside ``assert_graph_profile_is_executable`` at model construction
    for ``role="target"`` and ``role="draft"`` (the MTP drafter carries the
    target's EngramConfig and the same process environment, so a mixed
    configuration fails there too).
    The logged ``ple_layer_ids`` are the role's own (:func:`_role_ple_layer_ids`).
    """
    mode = resolve_ple_mode(vllm_config)
    ple_layer_ids = _role_ple_layer_ids(vllm_config, role)
    engram_config = getattr(vllm_config, "engram_config", None)
    logger.info(
        "e2-guard[%s]: ple_mode=%s pull_transport=%d engram_cpu_offload=%s "
        "%s=%s ple_layer_ids=%s.",
        role,
        mode,
        1 if mode == PLE_MODE_PULL else 0,
        None if engram_config is None else bool(engram_config.cpu_offload),
        UPSTREAM_UVA_ENV,
        os.environ.get(UPSTREAM_UVA_ENV, "<unset>"),
        ple_layer_ids,
    )
    return mode


#: Back-compatible name for the earlier call sites.
assert_breakable_cudagraph_pinned = assert_graph_profile_is_executable
