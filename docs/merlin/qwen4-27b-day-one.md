# Qwen 4 dense 27B: day one

Sequence for serving a new dense Qwen 4 27B checkpoint on this engine before it has a recipe of its own. It
assumes the checkpoint is on the Qwen3.5 lineage (GDN + full attention, MTP head) under a new class name. If step 1
shows anything else, stop at that step: the remedy is code, not configuration.

Nothing below is qualified. A profile row made from it stays `untested` until the gates in step 6 pass on the target GPUs
and the run is recorded.

## 1. Read the architecture in config.json

Read `config.json` (and `generation_config.json`, `chat_template*`, `tokenizer_config.json`,
`model.safetensors.index.json`) before the weights. Compare with a Qwen3.8-27B checkpoint:

- `architectures` and `model_type` (top level and `text_config.model_type`);
- `text_config.layer_types`: only `linear_attention` and `full_attention`. Any other kind (for example
  `indexed_attention`, or QSA/HC/PLE fields such as `hc_*`, `ple_*`, `ngram_*`, `indexer_*`) means the Qwen4Exp
  lineage. That is out of scope here: it needs a dense FFN in the Qwen4Exp model;
- no `num_experts` (dense); `mtp_num_hidden_layers` (MTP head present);
- `vocab_size`, `max_position_embeddings`, `rope_parameters`, `vision_config`, `quantization_config`;
- the tensor index: `mtp.*` and `model.language_model.*` names as in Qwen3.8-27B. A renamed tensor fails the load or
  loads unquantised.

Also check that the installed transformers (`>= 5.20.0, < 5.21.0` in `requirements/common.txt`; the lock installs
5.20.2) knows the processor and
tokenizer classes named in the checkpoint. If they need a newer transformers, stop here.

## 2. Add an alias if the names are new

The Qwen3.5-lineage names live in one table, `vllm/qwen35_family.py`. Every Merlin gate reads it: the 27B sm_86
series patches, the Inductor reduction pin, RecoverSSM and its PP rule, the host-embed table, the GDN warm-up and the
MTP rewrite. With no env set the table holds today's `Qwen3_5*` names plus these Qwen 4 candidates:

- dense: `Qwen4ForCausalLM`, `Qwen4ForConditionalGeneration`;
- MoE: `Qwen4MoeForCausalLM`, `Qwen4MoeForConditionalGeneration`;
- model types: `qwen4`, `qwen4_text`, `qwen4_moe`, `qwen4_moe_text`.

Candidates only join the gates. The engine still has to know the class, so a new name needs an alias:

```bash
export MERLIN_QWEN35_ARCH_ALIASES="Qwen4ForConditionalGeneration=Qwen3_5ForConditionalGeneration,qwen4=qwen3_5"
```

- `alias=canonical` serves the alias with the canonical class. It adds the alias to the model registry and the
  config-hook map, and loads the aliased `model_type` with the Qwen3.5 HF config class. After loading, the
  `model_type` is renamed to the canonical one; the text config follows (`qwen4_text` becomes `qwen3_5_text`). A text
  config with an alias of its own is renamed to its target even when the parent is already canonical (a `qwen3_5`
  parent over `qwen4_text` needs `qwen4_text=qwen3_5_text`). The rename runs again after `--hf-overrides`, nested
  `text_config` overrides included, so an override cannot bring the alias back.
- Canonical names: `Qwen3_5ForCausalLM`, `Qwen3_5ForConditionalGeneration`, `Qwen3_5MoeForCausalLM`,
  `Qwen3_5MoeForConditionalGeneration`; model types `qwen3_5`, `qwen3_5_text`, `qwen3_5_moe`, `qwen3_5_moe_text`.
- A bare `Name` joins the gates only. Use it when the engine already registers the class, for example after porting an
  upstream PR.
- `-Name` drops a default candidate. Drop a candidate if a Qwen 4 class of that name turns out not to be on the
  Qwen3.5 lineage.
- Names that are all lower case are model types. Names with an upper-case letter are architectures.
- Refused at import: an alias that already names an architecture the engine registers (model registry or
  config-hook map; `Qwen3ForCausalLM=Qwen3_5ForCausalLM` would re-point real Qwen3) or a model type HF already maps;
  the same alias with two different targets; a model-type alias whose `_text` suffix differs from its target's
  (`qwen4_text=qwen3_5` is refused, `qwen4_text=qwen3_5_text` is right); a name both added and dropped
  (`Qwen4ForCausalLM=Qwen3_5ForCausalLM,-Qwen4ForCausalLM`); `--Name`; and `qwen3_5_mtp`, the MTP rewrite's output.

Flash-Next names win. A name in `VLLM_FLASHNEXT_FAMILY_ALIASES` (or a `Qwen4Exp*` name) is never a Qwen3.5 name, and
naming it here is refused at import. The env is read once when vLLM is imported, so export it before `merlin serve`.
The mp executor (our default) inherits it; Ray workers get it in their actor `runtime_env`, before they import vLLM.
It is not a compile-cache factor: after changing an entry for a checkpoint that has already booted, use a fresh
`CACHE_ROOT` / `VLLM_CACHE_ROOT`.

If the checkpoint ships its own `config.json` under the canonical names, no alias is needed.

## 3. Launcher smoke with the checkpoint's own template

`merlin serve <dir>` no longer guesses for a Qwen-4-named config outside the `qwen4_exp` lineage. It refuses unless
`QWEN4_ROUTE` names the recipe:

```bash
QWEN4_ROUTE=27b merlin serve /models/qwen4-27b --dry-run        # or: merlin serve 27b /models/qwen4-27b --dry-run
```

`QWEN4_ROUTE=flash-next` is the other value. Flash-Next routing is unchanged: it goes by the exact Flash-Next names
(`qwen4_exp`, `qwen4_exp_text`, `Qwen4ExpForCausalLM`, `Qwen4ExpForConditionalGeneration`, `Qwen4ExpMTP`), so a
renamed class that merely contains `exp` (say `Qwen4ExperimentalForConditionalGeneration`) still needs `QWEN4_ROUTE`.

`QWEN4_ROUTE` takes only `27b` or `flash-next`, also for a `qwen4_moe` checkpoint. To serve a Qwen 4 MoE checkpoint
with the Ornith 1.5 recipe (the Qwen3.5 MoE lineage), pass the checkpoint name instead of routing:
`merlin serve ornith1.5 /models/qwen4-moe`.

For such a checkpoint the launcher serves the checkpoint's own `chat_template` and `generation_config.json`. It drops
`--chat-template` (our Froggeric v22.5) and `--override-generation-config` (the Qwen3.8 sampling), and prints a note.
It does so only when the checkpoint ships a chat template: `chat_template.jinja`, `chat_template.json`, or a
`chat_template` entry in `tokenizer_config.json` or `processor_config.json`. Without one it keeps the recipe's template
and sampling and says so (vLLM would otherwise fail every chat request). `USE_CKPT_TEMPLATE=1` set by hand is honoured
either way, with a warning when there is no template and no `CHAT_TEMPLATE`. The switches:

| Variable | Unset | Empty | Value |
|---|---|---|---|
| `USE_CKPT_TEMPLATE` | on for an unrecognised Qwen 4 checkpoint that ships a chat template, off otherwise | as unset | `1`: the checkpoint's template and sampling; `0`: the recipe's |
| `CHAT_TEMPLATE` | the recipe's template (or none, see above) | the checkpoint's template | that template file |
| `GENERATION_OVERRIDE` | the recipe's sampling (or none, see above) | as unset | `none`: no `--override-generation-config` from the launcher; or a JSON object for `--override-generation-config` |
| `REASONING_PARSER` | the recipe's (`qwen3`; Super VL `nemotron_v3`) | as unset | `none`: no `--reasoning-parser`; or that vLLM parser |
| `TOOL_CALL_PARSER` | `qwen3_coder` | as unset | `none`: no `--tool-call-parser` and no `--enable-auto-tool-choice`; or that vLLM parser |

An empty value and `none` are each noted at start (`GENERATION_OVERRIDE=none` only when the recipe passes an
override, so with the checkpoint's template on there is nothing to note).

An explicit `CHAT_TEMPLATE` or `GENERATION_OVERRIDE` wins over `USE_CKPT_TEMPLATE`. The parsers stay `qwen3` /
`qwen3_coder` unless set. Check them against the model card. `REASONING_PARSER=hf TOOL_CALL_PARSER=hf` reads the
checkpoint's own `response_template`.

First boot on the target GPUs: TP4, `SERVED=qwen4-27b`, and `KV_BYTES=auto` (vLLM sizes the pool; no pin yet).

- Load must pass, then a short chat, a tool call and a thinking-on and thinking-off reply.
- Diff the rendered prompt against Froggeric v22.5. Keep Froggeric (`USE_CKPT_TEMPLATE=0`) only if the checkpoint
  template is unchanged in substance.
- The structured-decisions API (`vllm/entrypoints/openai/decisions/strategies.py`) picks its
  `READ_STRATEGIES` entry by the served architecture, which stays the checkpoint's own name. An aliased Qwen 4 name is
  not in that table, so the API refuses it; chat, completions and tool calls are unaffected.

## 4. MTP only

`SPEC` stays `mtp`, the default. The 27B recipe refuses any other value, DFlash2 included. There is no DFlash2 drafter
for a new checkpoint. The MTP rewrite comes from the table in step 2: `qwen3_5*` and the aliased or candidate model types
go to `Qwen3_5MTP` (`Qwen3_5MoeMTP` for MoE). `REDUCED_DRAFT_VOCAB` stays 0.

## 5. Profile row and pins

Clone a row in `tools/merlin/profiles/qwen38-27b.json`:

- `27b-tp2-3090` for the two-card lane, or `27b-tp4-3090` for four cards;
- new `id`, `status: "untested"`, `capacity.estimate: true`, with the same `launch_shape`;
- the profile registry checks must still pass after the edit.

Serve it with `--profile <new id>` and the `27b` checkpoint name (or `QWEN4_ROUTE=27b`). A permanent checkpoint row
needs more: a `CHECKPOINTS` entry in `tools/merlin/launch.py`, a `launch.checkpoints` entry in the profile JSON, a
`models/<name>/checkpoint.sha256`, and a `SETUP_WITHHELD` entry in `tools/merlin/setup_cli.py` until there is a release.

Re-measure on the target GPUs; none of these carries over:

- `serve.kv_cache_memory`, the whole-block KV pin. The TP4 block is 17 attention layers x 1,024 tokens x 512 B; a
  different attention-layer count or KV head shape changes it. Re-find it with a pool-only boot (`KV_BYTES=auto`), then
  a whole-block pin;
- `capacity.kv_blocks`, `pool_tokens`, `fit_262k`, `fit_131k`;
- `serve.block_size` (TP4 1024) and `max_num_batched_tokens` (TP2 2048 / TP4 4096; the `CHUNK` env), and `LPT`
  (TP2 832);
- the 8 MP image headroom at the new pin (`MAX_PIXELS`), if the checkpoint has a vision tower;
- `capture_sizes` only if `spec_k` changes from 4.

## 6. Gates

Run them on the target GPUs at the pinned row, T=0 unless stated, against the bar of the 27B release bench card
(`models/qwen38-27b/benchmarks/2026-10-20/BENCH-CARD.md`):

1. Load + template/parser smoke (step 3), plus a tokenizer round-trip that includes combining marks (`\p{M}`).
2. GSM8K-200, 5-shot, thinking off.
3. Needle panel at 32K, 131K and **262K (required)**: single-occurrence needles at varied depths, cold cache. Add
   the 262,144-token boundary row (a prompt ending just below and just above `max_model_len`): the upstream 262K crash
   vllm#62383 is open, and both fixes (#62409, #63432) are unmerged.
4. Residency at the pin: 3 x 262K and 6 x 131K (TP2), or 7 x 262K and 14 x 131K (TP4). Use distinct-start prompts
   (shared prefixes overstate residency) and require 0 preemptions.
5. Cached-prefix and longer-than-chunk prompts with FP8 KV.
6. Decode tok/s against Qwen3.8-27B on the same cards (same-card A/B only).

The row goes `tested` only with a recorded run of these gates.
