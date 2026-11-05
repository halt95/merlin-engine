#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)

"""Head + MTP-drafter calibration capture: per-module input Hessians (H = X^T X) for the shared lm_head and the
MTP drafter's linears, taken from the SERVED philbert model. Calibration used public benchmark prompts
(ARC, MMLU-Pro, GPQA, GSM8K, HumanEval), template-generated synthetic tasks and a small hand-written
prompt set, together with the model's own responses to them. No calibration data is distributed.

vLLM general plugin (same shape as quant/kvcal.py): loaded in every vLLM process via
    PYTHONPATH=/path/to/mtpcal VLLM_PLUGINS=mtpcal MTPCAL_OUT=<dir>  vllm serve ... --enforce-eager --tensor-parallel-size 1
`register()` wraps `Qwen3_5MTP.__init__` to install forward-pre-hooks on the drafter's
    model.fc, model.layers.0.self_attn.qkv_proj / o_proj, model.layers.0.mlp.gate_up_proj / down_proj
and wraps `compute_logits` of BOTH `Qwen3_5MTP` (draft head input) and `Qwen3_5ForCausalLM`
(target head input), because ONE packed lm_head will serve both (eagle/utils shares it).

Hard requirement, asserted at boot: --enforce-eager (Python hooks do not run inside a replayed
CUDA graph). TP=2 is fine: column-parallel inputs are replicated on both ranks, and for the
row-parallel modules (o_proj, down_proj), whose input arrives already sliced per rank, the hook
all-gathers the slices along the feature dim first (both ranks execute the hook symmetrically),
so the Hessian is the FULL-input Hessian GPTQ needs. TP=1 OOMs on a 24 GB
card (19 GB target + 2.37 GiB draft head at construction). Only TP rank 0 flushes.
Hessians accumulate in fp32 on the GPU (~2.1 GB total: down_proj's 17408^2 dominates) and are
flushed to MTPCAL_OUT/hessians.pt every MTPCAL_FLUSH forwards and at exit, with the row count
per module so partial captures are still usable. Fused qkv / gate_up share one input each.
"""
import atexit, os, threading
import torch

OUT = os.environ.get("MTPCAL_OUT", "./mtpcal-out")
FLUSH = int(os.environ.get("MTPCAL_FLUSH", "200"))
_lock = threading.Lock()
_H: dict[str, torch.Tensor] = {}
_N: dict[str, int] = {}
_calls = [0]
_installed = [False]


def _acc(name: str, x: torch.Tensor) -> None:
    x = x.detach()
    if x.ndim > 2:
        x = x.reshape(-1, x.shape[-1])
    if x.shape[0] == 0:
        return
    x = x.to(torch.float32)
    with _lock:
        if name not in _H:
            _H[name] = torch.zeros(x.shape[1], x.shape[1], dtype=torch.float32, device=x.device)
            _N[name] = 0
        _H[name].addmm_(x.t(), x)
        _N[name] += int(x.shape[0])
    _calls[0] += 1
    if _calls[0] % FLUSH == 0:
        flush()


def _is_rank0() -> bool:
    try:
        from vllm.distributed import get_tensor_model_parallel_rank
        return get_tensor_model_parallel_rank() == 0
    except Exception:
        return True


def flush() -> None:
    if not _H or not _is_rank0():
        return
    os.makedirs(OUT, exist_ok=True)
    with _lock:
        payload = {"H": {k: v.cpu() for k, v in _H.items()}, "n": dict(_N)}
    tmp = os.path.join(OUT, "hessians.pt.tmp")
    torch.save(payload, tmp)
    os.replace(tmp, os.path.join(OUT, "hessians.pt"))
    print("[mtpcal] flushed %s (%s)" % (os.path.join(OUT, "hessians.pt"), {k: _N[k] for k in sorted(_N)}), flush=True)


ROW_PARALLEL = {"mtp.layers.0.self_attn.o_proj", "mtp.layers.0.mlp.down_proj"}


def _pre_hook(name):
    def hook(mod, args):
        x = args[0]
        if name in ROW_PARALLEL:
            from vllm.distributed import get_tensor_model_parallel_world_size, tensor_model_parallel_all_gather
            if get_tensor_model_parallel_world_size() > 1:
                x = tensor_model_parallel_all_gather(x.contiguous(), dim=-1)
        _acc(name, x)
    return hook


def _wrap_compute_logits(cls, name):
    orig = cls.compute_logits

    def compute_logits(self, hidden_states, *a, **kw):
        _acc(name, hidden_states)
        return orig(self, hidden_states, *a, **kw)
    cls.compute_logits = compute_logits


def register() -> None:
    if _installed[0]:
        return
    _installed[0] = True
    from vllm.model_executor.models import qwen3_5, qwen3_5_mtp

    orig_init = qwen3_5_mtp.Qwen3_5MTP.__init__

    def __init__(self, *a, **kw):
        orig_init(self, *a, **kw)
        from vllm.config import get_current_vllm_config
        cc = get_current_vllm_config().compilation_config
        if getattr(cc, "cudagraph_mode", None) is not None and str(cc.cudagraph_mode).upper().endswith("NONE") is False \
                and not get_current_vllm_config().model_config.enforce_eager:
            raise RuntimeError("mtpcal needs --enforce-eager (hooks do not run inside CUDA graphs)")
        m = self.model
        l0 = m.layers[0]
        m.fc.register_forward_pre_hook(_pre_hook("mtp.fc"))
        l0.self_attn.qkv_proj.register_forward_pre_hook(_pre_hook("mtp.layers.0.self_attn.qkv_proj"))
        l0.self_attn.o_proj.register_forward_pre_hook(_pre_hook("mtp.layers.0.self_attn.o_proj"))
        l0.mlp.gate_up_proj.register_forward_pre_hook(_pre_hook("mtp.layers.0.mlp.gate_up_proj"))
        l0.mlp.down_proj.register_forward_pre_hook(_pre_hook("mtp.layers.0.mlp.down_proj"))
        print("[mtpcal] hooks installed on the MTP drafter", flush=True)

    qwen3_5_mtp.Qwen3_5MTP.__init__ = __init__
    _wrap_compute_logits(qwen3_5_mtp.Qwen3_5MTP, "lm_head@draft")
    _wrap_compute_logits(qwen3_5.Qwen3_5ForCausalLM, "lm_head@target")
    atexit.register(flush)
    print("[mtpcal] registered (OUT=%s, flush every %d)" % (OUT, FLUSH), flush=True)
