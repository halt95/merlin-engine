#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Offline FP8 K/V scale calibration for the Qwen4Exp QSA attention layers.

Nothing here is imported by the engine: the harness boots a stock ``vllm serve`` with a small worker-side
hook injected through ``PYTHONPATH`` (a generated ``sitecustomize.py``), drives
calibration traffic, then max-merges the per-rank dumps into a
``VLLM_QSA_KV_SCALES`` sidecar. Collection, traffic and merge run from this one
tool, with the engine untouched:

* the worker hook wraps ``Qwen4ExpQSAAttention._project_qkv_gate`` and keeps a
  GPU-side running absmax of the pre-quantisation BF16 K and V per layer
  (``key``/``value`` are views of ``k``/``v``);
* the worker hook replaces the graph-profile guards in ``vllm.v1.e2_guards``
  with no-ops (calibration is eager by design, so the qualified-profile guards
  would refuse it). The PLE-mode guard is NOT touched;
* ``--enforce-eager`` is mandatory: the collector mutates state per step, which
  cudagraph capture would freeze.

Subcommands::

    qsa_calibrate.py serve   --model DIR --dump-dir EMPTY_DIR [--port 8171] [-- extra vllm args]
    qsa_calibrate.py traffic [--port 8171] [--model-name NAME] [--mark FILE]
    qsa_calibrate.py merge   DUMP_DIR OUT.json [--tp 2] [--pp 2] [--margin 1.10] [--after FILE]

``serve`` replaces itself with ``vllm serve``; run it under ``setsid`` and
SIGTERM the group after ``traffic``, then ``merge`` the dumps. Dump names are ``qsa_absmax_[pp<P>_]rank<R>.json`` (pp tag only
when PP > 1), the names merge expects. The engine never reads ``QSA_CALIBRATE_DIR``.
"""

from __future__ import annotations

import argparse
import importlib.abc
import importlib.machinery
import json
import math
import os
import re
import sys
import tempfile
from pathlib import Path

DIR_ENV = "QSA_CALIBRATE_DIR"  # not a VLLM_* name: the engine registry never sees it
E4M3_MAX = 448.0
QSA_MODULE = "vllm.models.qwen4_exp.nvidia.qsa"
GUARDS_MODULE = "vllm.v1.e2_guards"
GUARD_FUNCS = (
    "assert_graph_profile_is_executable",
    "assert_breakable_cudagraph_pinned",
    "assert_compiled_submodule",
    "assert_resolved_cudagraph_modes",
)
LAYERS = {
    f"language_model.model.layers.{n}.self_attn.attn"
    for n in (3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47)
}
DUMP_EVERY = 2000  # layer calls between periodic dumps


# ---------------------------------------------------------------- worker hook
class _PostImport(importlib.abc.MetaPathFinder):
    """Run ``fn(module)`` right after ``name`` finishes importing."""

    def __init__(self, name, fn):
        self.name, self.fn, self._busy = name, fn, False

    def find_spec(self, fullname, path=None, target=None):
        if fullname != self.name or self._busy:
            return None
        self._busy = True
        try:
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        finally:
            self._busy = False
        if spec is None or spec.loader is None:
            return None
        loader, fn = spec.loader, self.fn
        orig = loader.exec_module

        def exec_module(module, _orig=orig):
            _orig(module)
            fn(module)

        loader.exec_module = exec_module
        return spec


def _after_import(name, fn):
    mod = sys.modules.get(name)
    if mod is not None:
        fn(mod)
    else:
        sys.meta_path.insert(0, _PostImport(name, fn))


def _disarm_guards(mod):
    def _noop(*_a, **_k):
        return None

    for fn in GUARD_FUNCS:
        if hasattr(mod, fn):
            setattr(mod, fn, _noop)


class _Collector:
    def __init__(self, out_dir):
        self.out_dir = Path(out_dir)
        self.state = {}
        self.calls = 0
        self.ident = None

    def identity(self):
        if self.ident is not None:
            return self.ident
        try:
            from vllm.distributed import get_pp_group, get_tensor_model_parallel_rank

            pp = get_pp_group()
            tag = f"pp{pp.rank_in_group}_" if pp.world_size > 1 else ""
            self.ident = f"{tag}rank{get_tensor_model_parallel_rank()}"
        except Exception:
            return f"pid{os.getpid()}"  # not cached: retry once the groups exist
        return self.ident

    def dump(self):
        if not self.state:
            return
        out = {
            n: {"k_absmax": float(s[0].item()), "v_absmax": float(s[1].item())}
            for n, s in self.state.items()
        }
        path = self.out_dir / f"qsa_absmax_{self.identity()}.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(out, indent=1), encoding="utf-8")
        tmp.replace(path)

    def absmax(self, layer_name, key, value):
        import atexit

        import torch

        k = key.detach().abs().amax().float()
        v = value.detach().abs().amax().float()
        st = self.state.get(layer_name)
        if st is None:
            if not self.state:
                atexit.register(self.dump)
            self.state[layer_name] = [k, v]
        else:
            torch.maximum(st[0], k, out=st[0])
            torch.maximum(st[1], v, out=st[1])
        self.calls += 1
        if self.calls % DUMP_EVERY == 0:
            self.dump()


def _hook_qsa(collector):
    def patch(mod):
        cls = mod.Qwen4ExpQSAAttention
        orig = cls._project_qkv_gate
        orig_init = cls.__init__

        def project(self, qkv, positions):
            q, k, v, gate = orig(self, qkv, positions)
            collector.absmax(self.layer_name, k, v)
            return q, k, v, gate

        def init(self, *args, **kwargs):
            orig_init(self, *args, **kwargs)
            # vLLM 0.32 (vllm#57097) may fuse the main QK-norm/RoPE/gate and the
            # K/V cache write into the indexer launch, so _project_qkv_gate never
            # runs and the pre-quantisation K/V are never materialised, so the
            # harness forces the separate kernels.
            # The flag is read live by the forward and by QSAIndexer.forward.
            self.use_fused_qsa_prepare = False

        cls._project_qkv_gate = project
        cls.__init__ = init

    return patch


def install_worker_hook():
    """Called from the generated sitecustomize in every Python process."""
    out = os.environ.get(DIR_ENV, "").strip()
    if not out:
        return
    _after_import(GUARDS_MODULE, _disarm_guards)
    _after_import(QSA_MODULE, _hook_qsa(_Collector(out)))


# ---------------------------------------------------------------------- serve
def cmd_serve(a, extra):
    dump = Path(a.dump_dir)
    if not dump.is_dir() or any(dump.iterdir()):
        sys.exit(f"--dump-dir must be an existing EMPTY dir: {dump}")
    write_source(dump, a.model, a.max_model_len, extra)
    hookdir = Path(tempfile.mkdtemp(prefix="qsa-calib-hook-"))
    (hookdir / "sitecustomize.py").write_text(
        "import importlib.util as u\n"
        f"s = u.spec_from_file_location('qsa_calibrate', {str(Path(__file__).resolve())!r})\n"
        "m = u.module_from_spec(s); s.loader.exec_module(m); m.install_worker_hook()\n",
        encoding="utf-8",
    )
    env = dict(os.environ)
    env[DIR_ENV] = str(dump.resolve())
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(hookdir), env.get("PYTHONPATH", "")) if p
    )
    env.setdefault("VLLM_USE_BREAKABLE_CUDAGRAPH", "0")
    # a calibration boot never carries prod sidecars/counters
    for n in ("VLLM_QSA_KV_SCALES", "VLLM_QSA_KV_CLIP_COUNT"):
        env.pop(n, None)
    cmd = [
        a.vllm, "serve", a.model,
        "--served-model-name", a.name,
        "--host", "127.0.0.1", "--port", str(a.port),
        "--tensor-parallel-size", str(a.tp), "--pipeline-parallel-size", str(a.pp),
        "--max-model-len", str(a.max_model_len), "--max-num-seqs", "8",
        "--max-num-batched-tokens", "1024", "--kv-cache-dtype", "fp8_e4m3",
        "--enforce-eager", "--no-enable-prefix-caching",
        "--no-enable-flashinfer-autotune", "--shutdown-timeout", "60",
        *(["--enable-expert-parallel"] if a.tp * a.pp > 1 else []),
        *extra,
    ]
    print("QSA-CALIBRATE serve:", " ".join(cmd), flush=True)
    os.execvpe(cmd[0], cmd, env)


SOURCE = "calib_source.json"  # serve writes it into the dump dir; merge turns it into the sidecar's provenance file


def provenance_path(sidecar) -> Path:
    """``<sidecar stem>.provenance.json`` next to the sidecar: the engine's loader takes layer entries only, so the
    calibration source travels beside the scales, never inside them (the launcher reads it)."""
    p = Path(sidecar)
    return p.with_name(p.stem + ".provenance.json")


def _config_rope_is_yarn(cfg: Path) -> bool:
    """True when the checkpoint's own config.json already selects YaRN (rope_scaling / rope_parameters, top level or
    text_config), so a boot without a YaRN override still serves YaRN."""
    try:
        conf = json.loads(cfg.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    for scope in (conf, conf.get("text_config") if isinstance(conf, dict) else None):
        if not isinstance(scope, dict):
            continue
        for key in ("rope_scaling", "rope_parameters"):
            rope = scope.get(key)
            if isinstance(rope, dict) and "yarn" in (str(rope.get("rope_type", "")), str(rope.get("type", ""))):
                return True
    return False


def write_source(dump: Path, model: str, max_model_len: int, extra: list[str]) -> None:
    """What the dumps were calibrated on: the checkpoint's config.json hash and the RoPE the boot served."""
    import hashlib

    cfg = Path(model) / "config.json"
    src = {
        "model": str(Path(model).resolve()),
        "config_sha256": hashlib.sha256(cfg.read_bytes()).hexdigest() if cfg.is_file() else None,
        "max_model_len": max_model_len,
        "rope": "yarn" if any("yarn" in x for x in extra) or _config_rope_is_yarn(cfg) else "native",
    }
    (dump / SOURCE).write_text(json.dumps(src, indent=1), encoding="utf-8")


# -------------------------------------------------------------------- traffic
def cmd_traffic(a, _extra):
    import base64
    import io
    import random
    import time
    import urllib.request

    try:
        from PIL import Image
    except ImportError:
        sys.exit("Pillow is required: refusing to calibrate without the image arm")
    base = f"http://127.0.0.1:{a.port}/v1"
    words = (
        "harbour dawn quiet boats rest still granite ledger crimson orbit velvet "
        "thunder archive lantern mosaic quarry sable meridian copper glacier marble "
        "sextant bramble hollow zenith cinder plume rivet tundra ember lattice"
    ).split()
    rng = random.Random(1949)

    def text(n):
        return " ".join(rng.choice(words) for _ in range(n))

    def post(path, body):
        req = urllib.request.Request(
            f"{base}/{path}", data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        t0 = time.perf_counter()
        d = json.load(urllib.request.urlopen(req, timeout=3600))
        return d, time.perf_counter() - t0

    def completion(prompt, max_tokens=64, ignore_eos=False):
        d, dt = post("completions", {
            "model": a.model_name, "prompt": prompt, "max_tokens": max_tokens,
            "temperature": 0.8, "ignore_eos": ignore_eos,
        })
        print(f"  text: {d['usage']['prompt_tokens']} prompt tok, {dt:.1f}s", flush=True)
        return d["usage"]["prompt_tokens"]

    def image_chat(n):
        img = Image.new("RGB", (896, 896))
        px, r = img.load(), random.Random(7)
        for y in range(0, 896, 8):
            for x in range(0, 896, 8):
                c = (r.randrange(256), r.randrange(256), r.randrange(256))
                for dy in range(8):
                    for dx in range(8):
                        px[x + dx, y + dy] = c
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        d, dt = post("chat/completions", {
            "model": a.model_name, "max_tokens": 64, "temperature": 0.8,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": uri}},
                {"type": "text", "text": "Describe the image. Context: " + text(n)},
            ]}],
        })
        print(f"  image: {d['usage']['prompt_tokens']} prompt tok, {dt:.1f}s", flush=True)

    def w(tokens):  # the word list tokenises at ~1.26 tokens/word on this tokenizer
        return int(tokens / a.ratio)

    for label, t in (("2k", 2000), ("8k", 8000), ("32k", 32000)):
        print(f"[{label}]", flush=True)
        completion(text(w(t)))
    print("[image x2]", flush=True)
    image_chat(w(2000))
    image_chat(w(8000))
    deepest = 0
    for label, t in (("128k", 128000), ("200k", 200000), ("255k", 255000)):
        print(f"[{label}]", flush=True)
        deepest = max(deepest, completion(text(w(t))))
    if deepest < 250000:
        sys.exit(f"deepest arm reached only {deepest} prompt tokens (< 250000): fix --ratio")
    # Flush: dumps persist every DUMP_EVERY layer calls, not on a timer. 12 forced
    # 64-token completions cross at least four boundaries. The mark is written BEFORE
    # them (= the deepest request is done), so `merge --after` proves a dump landed
    # after the last real traffic; a post-flush mark could only be beaten by an exit
    # dump, which a SIGTERM'd worker never writes.
    if a.mark:
        Path(a.mark).write_text(str(time.time()), encoding="utf-8")
    print("[flush]", flush=True)
    for _ in range(12):
        completion(text(200), max_tokens=64, ignore_eos=True)
    print("TRAFFIC DONE", flush=True)


# ---------------------------------------------------------------------- merge
def cmd_merge(a, _extra):
    def die(s):
        sys.exit("FAIL: " + s)

    if not (math.isfinite(a.margin) and 1.0 <= a.margin <= 2.0):
        die(f"margin {a.margin} outside [1.0, 2.0]")
    d_dir, out = Path(a.dump_dir), Path(a.out)
    files = sorted(d_dir.glob("qsa_absmax_*.json"))
    want = (
        {f"qsa_absmax_pp{p}_rank{r}.json" for p in range(a.pp) for r in range(a.tp)}
        if a.pp > 1
        else {f"qsa_absmax_rank{r}.json" for r in range(a.tp)}
    )
    have = {f.name for f in files}
    if have != want:
        die(f"dump files {sorted(have)} != expected {sorted(want)} (pid-named or missing files are not mergeable)")
    stray = sorted(x.name for x in d_dir.iterdir() if x.name.endswith(".tmp"))
    if stray:
        die(f"unfinished dump temp files present: {stray}")
    mark = float(Path(a.after).read_text(encoding="utf-8").strip()) if a.after else None
    per_stage, merged = {}, {}
    for f in files:
        if mark is not None and os.path.getmtime(f) <= mark:
            die(f"{f.name} is not newer than the traffic mark (flush not persisted)")
        d = json.loads(f.read_text(encoding="utf-8"))
        if not d:
            die(f"empty dump {f.name}")
        m = re.fullmatch(r"qsa_absmax_(?:pp(\d+)_)?rank(\d+)\.json", f.name)
        stage = int(m.group(1) or 0)
        names = set(d)
        if names - LAYERS:
            die(f"{f.name}: unexpected layers {sorted(names - LAYERS)} (draft QSA layer? MTP must be off)")
        if per_stage.setdefault(stage, names) != names:
            die(f"{f.name}: layer set differs from the other TP rank(s) of stage {stage}")
        for name, st in d.items():
            k, v = float(st["k_absmax"]), float(st["v_absmax"])
            if not (math.isfinite(k) and math.isfinite(v) and k > 0 and v > 0):
                die(f"non-finite or non-positive absmax on {name} in {f.name}")
            z = merged.setdefault(name, {"k_absmax": 0.0, "v_absmax": 0.0})
            z["k_absmax"], z["v_absmax"] = max(z["k_absmax"], k), max(z["v_absmax"], v)
    stages = list(per_stage.values())
    for i in range(len(stages)):
        for j in range(i + 1, len(stages)):
            if stages[i] & stages[j]:
                die(f"PP stages overlap on {sorted(stages[i] & stages[j])}")
    if set(merged) != LAYERS:
        die(f"merged layers != the 12 QSA layers (missing {sorted(LAYERS - set(merged))})")
    sidecar = {
        n: {
            "k_scale": m["k_absmax"] * a.margin / E4M3_MAX,
            "v_scale": m["v_absmax"] * a.margin / E4M3_MAX,
        }
        for n, m in sorted(merged.items())
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    # no trailing newline: byte-compatible with the shipped sidecars
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(sidecar, indent=1, allow_nan=False))
    if set(json.loads(out.read_text(encoding="utf-8"))) != LAYERS:
        die("re-read of the written sidecar failed")
    src = d_dir / SOURCE
    if src.is_file():
        prov = dict(json.loads(src.read_text(encoding="utf-8")), margin=a.margin)
        provenance_path(out).write_text(json.dumps(prov, indent=1), encoding="utf-8")
        print(f"provenance: {provenance_path(out)}")
    else:
        print(f"note: no {SOURCE} in {d_dir} (dumps from an older serve): no provenance file; the launcher "
              "cannot check these scales against the checkpoint")
    print(
        f"PASS merged {len(files)} dump files (stages {sorted(per_stage)}), "
        f"{len(sidecar)} layers, margin {a.margin}: {out}"
    )


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    extra = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--model", required=True)
    s.add_argument("--dump-dir", required=True)
    s.add_argument("--name", default="qsa-calib")
    s.add_argument("--port", type=int, default=8171)
    s.add_argument("--tp", type=int, default=2)
    s.add_argument("--pp", type=int, default=2)
    s.add_argument("--max-model-len", type=int, default=262144)
    s.add_argument("--vllm", default="vllm")
    s.set_defaults(fn=cmd_serve)
    t = sub.add_parser("traffic")
    t.add_argument("--port", type=int, default=8171)
    t.add_argument("--model-name", default="qsa-calib")
    t.add_argument("--ratio", type=float, default=1.26)
    t.add_argument("--mark", help="write the epoch time after the deepest request (before the flush requests) here, for `merge --after`")
    t.set_defaults(fn=cmd_traffic)
    m = sub.add_parser("merge")
    m.add_argument("dump_dir")
    m.add_argument("out")
    m.add_argument("--tp", type=int, default=2)
    m.add_argument("--pp", type=int, default=2)
    m.add_argument("--margin", type=float, default=1.10)
    m.add_argument("--after")
    m.set_defaults(fn=cmd_merge)
    a = ap.parse_args(argv)
    a.fn(a, extra)


if __name__ == "__main__":
    main()
