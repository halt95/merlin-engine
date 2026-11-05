#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)

"""Turn the published Qwen3.8-Flash-Next-W4A16-Merlin checkpoint into the serving copy.

The engine keys the FP8 PLE (n-gram embedding) table on the checkpoint config field
text_config.ple_embedding_dtype. The field is read from the model's *text* config
(`vllm_config.model_config.hf_text_config`), so it has to live inside the `"text_config"` object of
config.json, not at the top level. The serving copy differs from the published checkpoint in exactly that
one key; every weight file can stay a hard link of the published one. This script inserts the key as a
targeted text edit (the rest of the file is byte-preserved), re-parses the result to prove the key landed
where the engine reads it, writes a new file and renames it into place -- so a hard-linked config.json is
un-linked rather than mutated in place -- and keeps a backup.

    python3 models/flash-next/make-e1-config.py /path/to/Qwen3.8-Flash-Next-W4A16-Merlin
"""
import json, os, re, shutil, sys
KEY, VAL = "ple_embedding_dtype", "float8_e4m3fn"
d = sys.argv[1] if len(sys.argv) > 1 else sys.exit(__doc__)
p = os.path.join(d, "config.json")
if not os.path.isfile(p) or not os.path.isfile(os.path.join(d, "model.safetensors.index.json")):
    sys.exit(f"{d}: not a checkpoint directory (config.json + model.safetensors.index.json expected)")
raw = open(p, "rb").read()
cfg = json.loads(raw)
text = cfg.get("text_config")
if not isinstance(text, dict):
    sys.exit("config.json has no text_config object; this is not the Flash-Next checkpoint this script is for")
if KEY in text and text[KEY] != VAL:
    sys.exit(f"text_config.{KEY}={text[KEY]!r} already present; refusing to overwrite")
stale_top = KEY in cfg   # a top-level copy of the key, which the engine never reads
if text.get(KEY) == VAL and not stale_top:
    print("already set"); sys.exit(0)
if os.stat(p).st_nlink > 1:
    print(f"note: config.json has {os.stat(p).st_nlink} hard links; writing a new file so the other links stay unchanged")
s = raw.decode("utf-8")
if stale_top:
    # remove the top-level copy: the first
    # occurrence of the key line that sits before "text_config" is the top-level one
    tc = s.index('"text_config"')
    mm = None
    for cand in re.finditer(r'\r?\n[ \t]*"' + KEY + r'"\s*:\s*"[^"]*",?', s):
        if cand.start() < tc: mm = cand; break
    if mm is None: sys.exit(f"top-level {KEY} present but not in the expected position; remove it by hand")
    s = s[:mm.start()] + s[mm.end():]
    if KEY in json.loads(s): sys.exit("internal error: top-level key still present after removal; nothing written")
    print(f"note: stale top-level {KEY} (from the pre-fix script) removed")
m = re.search(r'"text_config"\s*:\s*\{', s)
if not m or s.count('"text_config"') != 1:
    sys.exit("could not locate a unique \"text_config\": { in config.json")
i = m.end()
nl = "\r\n" if "\r\n" in s[:400] else "\n"
# indentation of the first key inside text_config (the text after the opening brace up to the first quote)
after = s[i:i + 64]
indent = after[: len(after) - len(after.lstrip())].replace("\r", "").replace("\n", "") or "    "
new = s if text.get(KEY) == VAL else s[:i] + nl + indent + f'"{KEY}": "{VAL}",' + s[i:]
check = json.loads(new)  # must still parse ...
if check["text_config"].get(KEY) != VAL:  # ... and the key must be where the engine reads it
    sys.exit("internal error: key did not land in text_config; nothing written")
other = {k: v for k, v in check.items() if k != "text_config"}
if other != {k: v for k, v in cfg.items() if k not in ("text_config", KEY)} or \
   {k: v for k, v in check["text_config"].items() if k != KEY} != {k: v for k, v in text.items() if k != KEY}:
    sys.exit("internal error: something other than the one key changed; nothing written")
shutil.copy2(p, p + ".bak-pre-v2")
tmp = p + ".tmp-v2"
with open(tmp, "wb") as f: f.write(new.encode("utf-8"))
os.replace(tmp, p)
print(f"config.json: text_config.{KEY} = {VAL} in place, no other key (backup config.json.bak-pre-v2)")
