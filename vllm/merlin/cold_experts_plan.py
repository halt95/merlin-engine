# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Static cold-expert offload: the profile file and the per-rank plan.

Standard library only, so the launcher and the CPU tests can import it without
torch. The engine side (tensors, the Marlin dispatch) is
``vllm/merlin/cold_experts.py``.

What the feature does
---------------------
For each MoE layer named by the profile, the rank's routed experts are split:

* the **hot** experts (the first ``H`` of the profile's per-layer ranking that this
  EP rank owns, ``H = round(hot_frac * local experts)``) get a GPU copy;
* the full master (all local experts) moves to pinned host RAM and is read through
  UVA, but only the **cold** experts are ever read from it.

The freed VRAM per layer is ``(local - H)`` experts' packed weights (scales stay on
the GPU). Placement only: the kernels read the same bytes.

Profile file (JSON, ``format: merlin-cold-experts/1``)
------------------------------------------------------
::

    {
      "format": "merlin-cold-experts/1",
      "num_layers": 48, "num_experts": 512, "top_k": 10,
      "checkpoint": {"name": "...", "config_sha256": "<hex>" | null},
      "source": {...provenance: probe run, segments, steps...},
      "default": {"layers": [15, 16, 19, 20], "hot_frac": 0.5},
      "ranking": {"0": [512 global expert ids, hottest first], ..., "47": [...]}
    }

``config_sha256`` is the sha256 of the checkpoint's ``config.json``; null marks a
placeholder profile that is not tied to a checkpoint (logged loudly, allowed).
A layer missing from ``ranking`` cannot be named.

Environment (all runtime-only, never compile factors; see vllm/merlin/envs.py)
------------------------------------------------------------------------------
``MERLIN_COLD_EXPERTS``            profile path; unset, blank or ``0`` = OFF (the
                                   default). :func:`profile_path` is the engine's
                                   reading of "off"; the launcher holds the same
                                   values (``COLD_EXPERTS_OFF``, kept equal by a test)
                                   so turning it off needs no engine source.
``MERLIN_COLD_EXPERTS_LAYERS``     optional override of ``default.layers``
                                   (comma-separated decoder layer indices).
``MERLIN_COLD_EXPERTS_HOT_FRAC``   optional override of ``default.hot_frac``
                                   (0 <= f < 1; 0 = whole-layer offload through the
                                   same path).

The two overrides only refine a profile: with the offload off they change
nothing, and :func:`ignored_overrides` names the ones that are set so the engine
can say so (one warning per engine). The launcher ignores them too, with a note,
and does not export them.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass

FORMAT = "merlin-cold-experts/1"
ENV_PROFILE = "MERLIN_COLD_EXPERTS"
ENV_LAYERS = "MERLIN_COLD_EXPERTS_LAYERS"
ENV_HOT_FRAC = "MERLIN_COLD_EXPERTS_HOT_FRAC"
ENV_NAMES = (ENV_PROFILE, ENV_LAYERS, ENV_HOT_FRAC)
# MERLIN_COLD_EXPERTS values that mean "off" (after stripping whitespace).
OFF_VALUES = ("", "0")


class ProfileError(ValueError):
    """A profile or knob that must not boot."""


@dataclass(frozen=True)
class Settings:
    path: str
    layers: tuple[int, ...]
    hot_frac: float
    profile: dict


def parse_layers(raw: str) -> tuple[int, ...]:
    """'15, 16,19' -> (15, 16, 19); sorted, unique; any non-integer fails."""
    out = []
    for part in raw.split(","):
        p = part.strip()
        if not p:
            continue
        if not (p.isascii() and p.isdigit()):
            raise ProfileError(f"{ENV_LAYERS}: {p!r} is not a layer index")
        out.append(int(p))
    if not out:
        raise ProfileError(f"{ENV_LAYERS}: no layer given")
    return tuple(sorted(set(out)))


def parse_hot_frac(raw) -> float:
    try:
        f = float(raw)
    except (TypeError, ValueError):
        raise ProfileError(f"hot_frac {raw!r} is not a number") from None
    if not (0.0 <= f < 1.0):
        raise ProfileError(f"hot_frac {f} must be in [0, 1)")
    return f


def validate(profile: dict) -> dict:
    """Check the structure; returns the profile. Every ranking must be a
    permutation of 0..num_experts-1."""
    if not isinstance(profile, dict) or profile.get("format") != FORMAT:
        raise ProfileError(f"not a {FORMAT} profile")
    try:
        L = int(profile["num_layers"])
        E = int(profile["num_experts"])
        ranking = profile["ranking"]
    except (KeyError, TypeError, ValueError) as e:
        raise ProfileError(f"profile field missing or bad: {e}") from None
    if L <= 0 or E <= 0 or not isinstance(ranking, dict) or not ranking:
        raise ProfileError("profile has no layers, experts or ranking")
    full = set(range(E))
    for k, ids in ranking.items():
        if not (isinstance(k, str) and k.isdigit() and int(k) < L):
            raise ProfileError(f"ranking key {k!r} is not a layer index below {L}")
        if not isinstance(ids, list) or len(ids) != E or set(ids) != full:
            raise ProfileError(f"ranking[{k}] is not a permutation of 0..{E - 1}")
    d = profile.get("default") or {}
    if "layers" in d:
        for layer in d["layers"]:
            if str(layer) not in ranking:
                raise ProfileError(f"default layer {layer} has no ranking")
    if "hot_frac" in d:
        parse_hot_frac(d["hot_frac"])
    ck = profile.get("checkpoint") or {}
    sha = ck.get("config_sha256")
    if sha is not None and not (isinstance(sha, str) and len(sha) == 64):
        raise ProfileError("checkpoint.config_sha256 must be a sha256 hex string or null")
    return profile


def load_profile(path: str) -> dict:
    if not path or not os.path.isfile(path):
        raise ProfileError(f"{ENV_PROFILE}={path!r}: no such file")
    try:
        with open(path, encoding="utf-8") as f:
            profile = json.load(f)
    except (OSError, ValueError) as e:
        raise ProfileError(f"{ENV_PROFILE}={path!r}: unreadable JSON ({e})") from None
    return validate(profile)


def config_sha256(checkpoint_dir: str) -> str | None:
    p = os.path.join(checkpoint_dir, "config.json")
    if not os.path.isfile(p):
        return None
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def check_checkpoint(profile: dict, checkpoint_dir: str | None) -> str:
    """'ok' / 'placeholder' (profile not tied to a checkpoint) / 'unknown' (no local
    config.json to hash). Raises on a mismatch."""
    want = (profile.get("checkpoint") or {}).get("config_sha256")
    if want is None:
        return "placeholder"
    got = config_sha256(checkpoint_dir) if checkpoint_dir else None
    if got is None:
        return "unknown"
    if got != want:
        raise ProfileError(
            f"profile is for config.json sha256 {want[:12]}..., the checkpoint has {got[:12]}...: "
            "build a profile for this checkpoint (tools/merlin/cold_expert_profile.py)"
        )
    return "ok"


def profile_path(environ=None) -> str | None:
    """The profile path ``MERLIN_COLD_EXPERTS`` names, stripped; None when the
    offload is off (unset, blank or ``0``)."""
    env = os.environ if environ is None else environ
    raw = (env.get(ENV_PROFILE) or "").strip()
    return None if raw in OFF_VALUES else raw


def ignored_overrides(environ=None) -> tuple[str, ...]:
    """The layer / hot-fraction overrides that are set (non-blank) while the
    offload is off: they have no profile to refine. Empty when it is on."""
    env = os.environ if environ is None else environ
    if profile_path(env) is not None:
        return ()
    return tuple(k for k in (ENV_LAYERS, ENV_HOT_FRAC) if (env.get(k) or "").strip())


def settings_from_env(environ=None) -> Settings | None:
    """None when off (see :func:`profile_path`). Raises ProfileError on a bad
    profile or knob."""
    env = os.environ if environ is None else environ
    path = profile_path(env)
    if path is None:
        return None
    profile = load_profile(path)
    d = profile.get("default") or {}
    raw_layers = (env.get(ENV_LAYERS) or "").strip()
    if raw_layers:
        layers = parse_layers(raw_layers)
    elif d.get("layers"):
        layers = tuple(sorted(set(int(x) for x in d["layers"])))
    else:
        raise ProfileError(f"no layers: set {ENV_LAYERS} or the profile's default.layers")
    for layer in layers:
        if str(layer) not in profile["ranking"]:
            raise ProfileError(f"layer {layer} has no ranking in {path}")
    raw_f = (env.get(ENV_HOT_FRAC) or "").strip()
    if raw_f:
        hot_frac = parse_hot_frac(raw_f)
    elif "hot_frac" in d:
        hot_frac = parse_hot_frac(d["hot_frac"])
    else:
        raise ProfileError(f"no hot fraction: set {ENV_HOT_FRAC} or the profile's default.hot_frac")
    return Settings(path, layers, hot_frac, profile)


@dataclass(frozen=True)
class LayerPlan:
    """One layer on one EP rank. Index spaces: ``global`` (router ids), ``local``
    (rows of the rank's master tensor), ``hot`` (rows of the GPU hot tensor)."""

    layer: int
    hot_local: tuple[int, ...]  # master rows copied to the GPU, ascending
    hot_map: tuple[int, ...]  # global -> hot row, -1 otherwise (len = global experts)
    cold_map: tuple[int, ...]  # global -> master row for local cold experts, -1 otherwise
    n_local: int

    @property
    def n_hot(self) -> int:
        return len(self.hot_local)

    @property
    def n_cold(self) -> int:
        return self.n_local - self.n_hot


def plan_layer(layer: int, ranking: list[int], local_of: list[int], hot_frac: float) -> LayerPlan:
    """``local_of[g]`` = the rank's master row of global expert g, -1 if another rank
    owns it (the EP map; identity without EP). Hot = the first H owned ids of the
    ranking, H = round(hot_frac * owned)."""
    G = len(local_of)
    if len(ranking) != G:
        raise ProfileError(f"layer {layer}: ranking has {len(ranking)} experts, the layer {G}")
    owned = [g for g in ranking if local_of[g] >= 0]
    n_local = len(owned)
    if n_local == 0:
        raise ProfileError(f"layer {layer}: this rank owns no expert")
    rows = sorted(local_of[g] for g in owned)
    if rows != list(range(n_local)):
        raise ProfileError(f"layer {layer}: the EP map's local rows are not 0..{n_local - 1}")
    H = int(round(hot_frac * n_local))
    H = max(0, min(H, n_local))
    hot_globals = owned[:H]
    hot_local = tuple(sorted(local_of[g] for g in hot_globals))
    pos = {r: i for i, r in enumerate(hot_local)}
    hot_set = set(hot_local)
    hot_map = tuple(pos[local_of[g]] if local_of[g] in hot_set and local_of[g] >= 0 else -1
                    for g in range(G))
    cold_map = tuple(local_of[g] if local_of[g] >= 0 and local_of[g] not in hot_set else -1
                     for g in range(G))
    return LayerPlan(layer, hot_local, hot_map, cold_map, n_local)


def freed_bytes(plans, bytes_per_expert: int) -> int:
    return sum(p.n_cold for p in plans) * bytes_per_expert
