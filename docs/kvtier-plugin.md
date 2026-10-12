# The optional KV-cache tier plugin (`kvtier-dev-plugin`)

An **optional** out-of-tree plugin that puts a host-RAM and NVMe tier underneath this tree's KV cache and opens a
diagnostic surface on a running engine. It is not part of the Flash-Next release: nothing in `scripts/`, `patches/`,
`release/`, `Dockerfile` or `lxc/` depends on it, and the shipped serve command runs unchanged without it. Installing
it is a serving decision with a cost attached, so this file states the cost first.

Read [Flag limitations](#flag-limitations) before mounting anything. At the v2.5.1 geometry this tree **refuses to
boot** with the release's `--use-replayssm` (`scripts/serve-v2.5.sh:333`) and a KV connector both present: the last
clause of `vllm/config/vllm.py::validate_mamba_cached_kernel` raises on any KV-connector instance. So this is not a
package you can drop onto the shipped command line: the flag decision comes first, and it belongs to the release,
not to a plugin.

**Attribution.** *reference host* means this repository's own published measurements (`docs/reference.md`, cited by
line). *our box* means a different host serving the same fork at the same TP2 × PP2 + EP shape, the same
`--kv-cache-memory 4100000000` and the same `--block-size 4096`, with more host RAM than the reference host's 96 GB
allocation. *source* means a number read out of a file at the cited `file:line`. No number in this file was measured
on your hardware, and the tier sizes we ran are consequences of our host, not recommendations.

## What it is, and what it is not

`kvtier-dev-plugin` (Apache-2.0, `github.com/yarikdevcom/kvtier-dev-plugin`, 0.2.71 at the time of writing) is:

- **A package with two vLLM entry points**, `vllm.general_plugins` (loaded in every process) and
  `vllm.endpoint_plugins` (the API server, which is where the HTTP routes come from). Both are registered under the
  single name `kvtier_dev` (`kvtier-dev-plugin/pyproject.toml:32-40`). Nothing is edited in the vLLM tree.
- **A wrapper over the engine's own host-RAM tier**, the in-tree `SimpleCPUOffloadConnector` that
  `VLLM_USE_SIMPLE_KV_OFFLOAD` selects (`vllm/config/vllm.py:1052-1058`, read in our v2.5 tree checkout), plus an
  NVMe spill tier beneath it that the plugin implements itself, and the scheduler-side reads that make both legible.
- **Two fingerprint-pinned runtime patches to engine methods**: the upstream PR #56810 backport
  (`prefix_cacheable_group_ids`) and the 0.2.64 run-end anchor patch, which wraps
  `MambaManager.reachable_block_mask` (`single_type_kv_cache_manager.py:1610-1669`) so a cached range's last block is
  marked reachable (the plugin README's *Switches* row for `VLLM_E2_TIER_DEV_MAMBA_ANCHOR_PATCH`; cited by symbol
  rather than line number, because that file is restructured more often than the claim changes). Both are a refusal
  rather than a guess when the fingerprint drifts.
  The package also wraps a handful of tier seams without porting engine code (the store-decision path in
  `store_hook.py`, three handshake methods in `cpu_capacity.py`, the worker's writer). A separate four-hunk series is
  applied to the tree at rest; see [The plugin's patch series](#the-plugins-patch-series-and-how-it-relates-to-patches-here).
- **An HTTP surface under `/plugins/kvtier/`**: `status`, `snapshot`, `kvconfig`, `block`, `request`, `policy/reload`
  as reads, and `trace`, `guard`, `tier`, `lookups`, `blockdump`, `disk` as dev-state verbs that must be armed. It is
  a development window, not a serving API; see [Access model](#access-model-for-the-dev-routes).

It is **not** part of this release, not supported by it, not credentialled, and not free. At this geometry it costs
pinned host RAM, NVMe bytes, part of the GPU KV pool, a wait inside the engine step on each restore, and the flag
decision above.

## Install

From git, **non-editable**, into the venv that serves:

```bash
venv/bin/pip install "kvtier-dev-plugin @ git+https://github.com/yarikdevcom/kvtier-dev-plugin.git@v0.2.71"
venv/bin/pip check        # the package declares no dependencies; nothing else may move
```

- **Non-editable, deliberately.** The plugin wraps engine methods at import time, so an editable checkout means a
  `git pull` inside that checkout changes what the *next boot* imports. That is the point while you develop and the
  last thing you want in a venv that serves traffic.
- **Pin the ref.** `v0.2.71` is a tag, and the command above was run against the public repository to confirm it
  resolves (`pip download --no-deps` of that exact URL produced `kvtier-dev-plugin-0.2.71.zip`). This repository pins
  every other source it builds by commit and tree hash, so an un-pinned URL here would be the one line in the picture
  that drifts: it installs whatever the default branch holds at clone time, and for a package that wraps engine
  methods at import time that is a version nobody can reason about afterwards.
- **The pin set must not move.** The package declares `dependencies = []` precisely so that installing it into a live
  venv cannot churn this release's 196 pins (`release/v2.5/requirements-pinned.txt`). `pip check` and
  `pip list --outdated` are the check; a resolution that wants to touch `fastapi`, `starlette` or `torch` is wrong.
- **It registers through the entry points, and `VLLM_PLUGINS` decides whether vLLM loads them.** `VLLM_PLUGINS` is one
  allowlist for every plugin group, so naming only `kvtier_dev` also disables vLLM's own two general plugins
  (`lora_filesystem_resolver`, `lora_hf_hub_resolver`). If you use LoRA resolvers, name them too:
  `VLLM_PLUGINS=lora_filesystem_resolver,lora_hf_hub_resolver,kvtier_dev`.
- Nothing is active after the install: the routes answer 503 until `VLLM_E2_TIER_DEV=1` and the tier connector is on
  the command line.

## The environment a serving boot needs

Copy-pasteable, then the notes. Every name is the plugin's own (`kvtier_dev_plugin/config.py`, cited per line).

```bash
# the plugin and its routes
export VLLM_PLUGINS=kvtier_dev
export VLLM_E2_TIER_DEV=1                                     # unset or 0 = mounted but inert; routes answer 503
export VLLM_E2_TIER_DIAG=/var/tmp/kv-tier-diagnostics         # one JSON file per rank; the reader must see this dir
export VLLM_E2_TIER_DISK_JOURNAL_DIR=/var/lib/kv-tier/journal # persistent storage: the journal is not rebuildable

# the NVMe spill tier
export VLLM_E2_TIER_DISK_ROOT=/var/lib/kv-tier/store          # one file per spilled block; keep one root per tree
export VLLM_E2_TIER_DISK_BYTES=160000000000                   # the store's byte ceiling (source: config.py:342)
export KVTIER_DISK_BOOT_ENABLE=1                              # 0.2.33+: arm the spill tier at boot (the default)

# the engine's own host-RAM tier that the plugin instruments
export VLLM_USE_SIMPLE_KV_OFFLOAD=1
```

and two additions to the `vllm serve` line:

```bash
  --kv-offloading-size 32 \
  --kv-transfer-config '{"kv_connector":"SimpleCPUOffloadConnector","kv_role":"kv_both","kv_load_failure_policy":"recompute"}'
```

- **`LimitMEMLOCK=infinity` is not optional.** The host-RAM tier pins its arena. A process that inherits the usual
  8 MiB soft *and* hard memlock fails to pin and **nothing complains**. In our deployment that limit is a system
  unit's `LimitMEMLOCK=infinity` (`llm-api`'s `server/systemd/vllm-engine.service:139`); its header records that
  `/etc/security/limits.d/` provably does not apply on that host and a user-scope unit cannot set it, so do not plan
  on a `limits.d` drop-in. Verify after the boot from the engine's own view, not from `systemctl show`:
  `prlimit --pid <engine pid> --memlock`, or read the wrapper's post-start report if you have one.
- **`--kv-offloading-size` is server-wide, not per rank**: the connector divides `cpu_bytes_to_use` by the world size
  before each worker sizes its own arena
  (`vllm/distributed/kv_transfer/kv_connector/v1/simple_cpu_offload_connector.py:76-87`), so `32` pins ≈32 GiB in
  total across four ranks (≈8 GiB each) plus the plugin's own per-process buffers below.
- **Pass both flags or neither.** `VllmConfig._post_init_kv_transfer_config()` fills in `kv_connector`,
  `cpu_bytes_to_use` and `kv_role` only when a size was given; with `--kv-offloading-size` absent your JSON survives
  verbatim with **no** `cpu_bytes_to_use` and the connector boots at `DEFAULT_CPU_CAPACITY_BYTES`
  (same file, `:76`), a silently wrong-sized tier rather than no tier.
- **A failed read costs tokens, not the request** (plugin 0.2.72). The ENGINE default of
  `kv_load_failure_policy` is `"fail"` (`vllm/config/kv_transfer.py:103`), which turns an NVMe miss between the
  scheduler's lookup and the worker's read into a failed request instead of a slower-but-correct recompute. Since
  0.2.72 the plugin sets `"recompute"` itself **when your JSON omits the field**, and it never overrides an explicit
  value - including an explicit `"fail"`; the boot line's `LOAD_POLICY[what:who]` token reports the value in force
  and who chose it. Passing it explicitly, as the command above does, stays correct and is the clearest form.
- **`KVTIER_DISK_BOOT_ENABLE` is a disable switch, not an enable switch** (`config.py:753`): unset or `1` arms the
  spill tier at connector creation and raises the boot's serve posture with no POST; `0` restores the manual posture.
  It reads nothing persisted from an earlier boot, and `VLLM_E2_TIER_DEV_BOOT_INERT` (which arms nothing at all)
  outranks it.
- **The journal belongs on storage that survives a reboot** (`config.py:381`). The spill journal is the only map from
  a KV identity to a file, and it cannot be rebuilt; a journal in `/tmp` is gone at the next boot while the KV files
  it describes outlive it. The boot line reports the verdict as `journal_dir=tmpfs|disk|unset|unknown|unwritable`,
  and the boot creates a path you set that does not exist yet.
- **Two read-tuning names, and our box's values are ours.** `VLLM_E2_TIER_DISK_READ_THREADS` (`config.py:441`,
  default one thread, clamped to 1-16) and `VLLM_E2_TIER_DISK_READ_BUFFER_BYTES` (`config.py:466`, the bounce-buffer
  pool's byte ceiling **per process**). The pool, not the thread count, bounds in-flight work, so raising one without
  the other buys nothing. The pool default is `MAX_BUFFERS_BYTES` = 96 MiB (`kvtier_dev_plugin/disk_read.py:140`),
  which holds 8 units and therefore misses a 10-unit offer. Our box runs the thread count at the clamp ceiling (16) and a
  2 GiB pool per process (`VLLM_E2_TIER_DISK_READ_BUFFER_BYTES=2147483648`), which is ≈8 GiB of transient host memory across
  four ranks on top of the pinned arena; both are decisions about a 251 GiB host and tell you nothing about a 96 GB one.
- **`VLLM_E2_TIER_DISK_SETTLE_MS` (`config.py:425`) is a wait inside the engine step.** Since 0.2.32 it is a flat
  allowance, `min(SETTLE_MS, the load's remaining deadline)`, default `5000`; unset does **not** mean "never wait".
  It runs in `start_load_kv`, so an allowance spent there delays every request sharing that step: the trade is a
  longer step in exchange for a restore that lands. `0` means never wait and is reachable. **Our boot leaves it unset**, so
  it runs that 5,000 ms default - which is the point of the note: an unset value no longer means "never wait".
- **`--kv-offloading-size 32` is our arithmetic, not a reading of our box.** Our own serving boot passes
  `--kv-offloading-size 8` (server-wide, so ≈2 GiB per rank across four ranks) on a host with far more RAM than the
  reference host's 96 GB. The arena size is a consequence of the host, exactly like the pool numbers above.
- **Our boot reaches the connector through `--kv-transfer-config` and does not set `VLLM_USE_SIMPLE_KV_OFFLOAD`.** Both
  routes select `SimpleCPUOffloadConnector`: the env var is the shorthand, the JSON is the explicit form, and the JSON is
  what we run. Use one or the other. If the tier looks absent, check both spellings - the plugin's own posture text names a
  missing `VLLM_USE_SIMPLE_KV_OFFLOAD` as one candidate alongside a mis-set `KV_OFFLOAD_SIZE`.
- **Where a restore's destination rows come from: `arena` (the default) or `bounded`.**
  `VLLM_E2_TIER_DISK_RESTORE_SOURCE` (`config.py:686`) defaults to `arena`; our boot sets `bounded` with
  `VLLM_E2_TIER_DISK_RESTORE_STAGING_ROWS=86` (`config.py:707`, default `32`). The cap is per rank and reserves nothing: 86
  rows at the widest rank window (12,902,400 B) is ≈1.03 GiB per rank and ≈4.1 GiB across four ranks, which is **above** the
  ≈1.61 GiB the default's own arithmetic quotes for 32 rows. Treat 86 as a host-size decision, not a recommendation; the
  reason to run `bounded` at all is that it makes the ceiling a claim instead of the accident of an arena size, and
  `staging_cap`'s `binding` verdict is what tells you whether it needs growing.
- **Our boot also runs the byte witnesses and the load-pair prune ON, and the copy-paste block above deliberately omits
  them.** `VLLM_E2_TIER_DISK_RESTORE_VERIFY`, `VLLM_E2_TIER_DISK_GPU_DIRECT_WRITE`,
  `VLLM_E2_TIER_DISK_GPU_DIRECT_WRITE_DELIVERY`, `VLLM_E2_TIER_DISK_GPU_SHADOW_WRITES` and
  `VLLM_E2_TIER_DISK_PRUNE_RESTORED_LOADS` are OFF by default on purpose: they read or write real device memory, and the last
  is the only flag in the package whose failure mode is a **wrong answer** rather than a slower one (the fork pairs
  `load_cpu_blocks[i]` with `load_gpu_blocks[i]` positionally, so a one-sided prune shifts every later destination). They are
  out of the recommended block because a launcher must not be able to turn them on by accident, not because they are broken.
  Two spellings worth getting right: the delivery flag is `VLLM_E2_TIER_DISK_GPU_DIRECT_WRITE_DELIVERY` (with `DIRECT`) while
  its timeout is `VLLM_E2_TIER_DISK_GPU_WRITE_DELIVERY_TIMEOUT_SECONDS` (without it). We found the plugin's own README
  documenting the first name without `DIRECT` - a variable the plugin never reads - and corrected it there; where any prose
  and `kvtier_dev_plugin/config.py` disagree, the source file wins.

## v2.5.1 status, in one honest paragraph

On our v2.5.1 tree the disk read path has fired under real traffic: one request with `adopted_tokens` 8,192,
`ranks_reads_applied` 40, and the engine's own
`vllm:prompt_tokens_by_source_total{source="external_kv_transfer"}` moving by 8,192 to match. That is the entire
positive record at this geometry: one shape, small, and not a throughput claim. Everything larger we measured came
from the v2.2-era tree (a boot with `restorations=2`, `ranks_reads_applied=396`, and `external_kv_transfer` 278,400
exactly equal to `adopted_tokens`). The blocker at v2.5.1 is not the read path: it is the group gate and the flag
refusal under [Flag limitations](#flag-limitations), which keep the offers from being made at all. Treat the tier on
this release as experimental, keep it off traffic you care about, and do not accept a counter as evidence; run the
greedy-equality test in [What to read after a boot](#what-to-read-after-a-boot) on your own box.

## Flag limitations

This is the section that costs someone an afternoon. It is first because the first item is a boot failure, not a
tuning note.

### `--use-replayssm`: the engine refuses the combination, and where it does not, restore serves nothing

The shipped serve line carries `--block-size 4096 --use-replayssm` (`scripts/serve-v2.5.sh:333`). Two separate facts
apply to it, and they have different witnesses.

1. **On v2.5.1 the boot is refused.** `vllm/config/vllm.py::validate_mamba_cached_kernel` ends with a clause that
   raises `"--use-replayssm is incompatible with KV connectors (P/D disaggregation, KV cache offload)"` whenever
   `kv_transfer_config.is_kv_transfer_instance` is true, and that predicate is `kv_connector is not None and kv_role
   in {kv_producer, kv_consumer, kv_both}` (`vllm/config/kv_transfer.py:143`), so the `kv_role: "kv_both"` the tier
   needs fires it. The clause sits **outside** the `use_recoverssm` branch (so it vetoes the read-only plain
   ReplaySSM path too) and outside the architecture-specific blocks, and none of the four RecoverSSM commits in the
   v2.5 series (0071, 0072, 0085, 0091) states why a connector would break. All of this is read, not inferred: it is
   written up with the hashes in the plugin's `patches/drafts/DRAFT-0006-replayssm-connector-validation/patch.manifest`.
   Consequence: **with the release's flag set as shipped, a tier boot does not start.**
2. **Where the combination does run, the disk tier becomes write-only on this model.** The replayssm/merged group
   absorbs the anchor group (group 3), so a restore offer no longer covers a complete prefix-cacheable group set, and
   the plugin refuses it as `refused.incomplete_group_set`, on the offer side and on the spill side. Spill keeps
   working; restore serves zero tokens. The refusal is correct and should stay: the gate's own comment
   (`kvtier_dev_plugin/disk_restore.py:1773-1797`) explains that `get_num_new_matched_tokens` advances
   `num_computed_tokens` for the **whole** model, `update_state_after_alloc`'s `n_ext_g == 0 -> continue` is a silent
   no-op for a group that was never in the tuple, and nothing recomputes the claimed range. Offering the restorable
   subset instead of refusing it is how a mechanically perfect read (all units applied, no failures) served a
   garbage answer at `temperature=0` on our box, and the same file records that.

One clarification so the two statements below are not read as a contradiction: **our current v2.5.1 serving boot does not
carry this flag** - `--use-replayssm` occurs zero times in its `/proc/<pid>/cmdline`, so the refusal in item 1 has never
fired on it, and the write-only-tier behaviour in item 2 is the trap we stepped around rather than something we are
suffering. Both halves below remain true and both are deliberate: the release ships the flag, and dropping it is our
recommendation for a boot that reads from the tier - a decision we have already made for ourselves and are explicitly not
asking you to take in this pull request.

On the evidence above, dropping `--use-replayssm` is the only route to a tier that reads at this geometry, and it is a
serving decision, not a config tidy-up: RecoverSSM is one of the two v2.5 changes that take the pool from 806,792 to
924,993 (`docs/reference.md:317-319`; the side-cache layout alone measured 853,994 in its own commit) and v2.5.1's
follow-up-turn behaviour is the release's headline. That is why this pull request does not touch the launcher: a
one-line flag removal in `scripts/serve-v2.5.sh` is the maintainer's call, not a plugin's. The plugin also ships a
**DO-NOT-APPLY** draft of the validator relaxation
(`patches/drafts/DRAFT-0006-*`); its header carries an unanswered question — can a connector load ever target a row
RecoverSSM holds live scratch for? — and the failure mode it would open is a plausible continuation rather than an
exception. Do not apply it.

### `--prefix-cache-retention-interval` must be a multiple of the scheduler block size

The flag is upstream's own (`vllm/config/cache.py:148`, default `0` = retain only semantic checkpoints; positive
values additionally retain periodic checkpoints) and `scripts/serve-v2.5.sh` passes none, so nothing changes unless
you set it. The rule that bites is the validator: it requires the value to be non-negative and **a multiple of
`scheduler_block_size`**, and raises `ValueError` at boot otherwise
(`vllm/v1/core/kv_cache_coordinator.py:34-63`, both messages quoted there).

- On our v2.5.1 boots the scheduler block size is **4,096**. On the v2.2 line it was **3,200**. The number is printed
  in the engine's own refusal, so an illegal value tells you what your geometry demands, and `GET
  /plugins/kvtier/kvconfig` reports the geometry with a divisibility verdict per group
  (`kvtier_dev_plugin/kvconfig.py:189-216`).
- **9,600 is illegal here.** `9,600 % 4,096 = 1,408`, so a value carried over from the v2.2 line raises `ValueError`
  at boot; the three-block analogue at this geometry is **12,288**.
- The checkpoint slots this costs are, for a `T`-token conversation, `T / block + 4 × T / retention` per conversation
  for the Mamba / sliding-window checkpoints. At block 4,096, retention 12,288 and `T` = 200,000 that is 48.8 + 65.1
  = **113.9 slots per conversation**, × the concurrent-sequence cap. That arithmetic is ours, not the engine's published
  figure - and the multiplier has to be read carefully: **our own boot carries `--max-num-seqs` twice** (`8` first, then `4`),
  argparse is last-wins, so our number is × 4 = **456**, not × 8 = 911. Take the multiplier from the **last** occurrence of the
  flag on the command line (or from the engine's own config dump), and check your boot's own KV geometry and pool line.
- We have **not** established that a retention setting changes what the tier can restore. It is listed here because it
  governs the density of checkpoints that the align-mode contract reads, and because it is a block-multiple rule that
  fails a boot loudly.

### `--block-size` changes the store's file names

The spill file name carries a digest of the integrity fingerprint: model id and revision, cache and tensor dtype, the
slot width, **every group's block size, the scheduler and hash block sizes, which groups are prefix-cacheable**, the
hash algorithm, the `NONE_HASH` seed, the rank, the world size and the decode/context-parallel widths
(`kvtier_dev_plugin/disk_layout.py:78-84`). So `--block-size 4096` and any other value name different files, by
design: "a configuration change produces a different FILENAME, so stale bytes are un-findable rather than mis-read"
(same file, `:87-89`). Read literally, that is the whole safety property and it has two consequences:

- changing the block size does not corrupt an existing store; it makes it **unloadable**: restore serves zero from
  it until new spills land under the new digest, because a lookup for the new digest simply never finds the old name;
- the ceiling pass is what reclaims the old bytes, and it can refuse to. A `stale_geometry` name sheds **first**
  (`SHED_ORDER_BY_CLASSIFICATION`, `disk_layout.py:294`) but only when no live rank publishes that digest with
  positive evidence; when peer identities are not all known the name is attributed `unattributable` and fails closed,
  never deleted (`:1683-1695`, with the production reason recorded at `:276-292`). So an old-geometry store can hold
  gigabytes a boot will not reclaim on its own. A hand delete is the deterministic answer and is always safe: the
  store is a cache, so clearing it costs hit rate and never correctness.

### Two trees must not share one store root

Keep `VLLM_E2_TIER_DISK_ROOT` and `VLLM_E2_TIER_DISK_JOURNAL_DIR` **per tree** (per geometry, per `--block-size`, per
digest). One root under one `VLLM_E2_TIER_DISK_BYTES` ceiling means the capacity pass of one tree is free to evict the
other tree's data: the two are indistinguishable to a directory walk, and the ceiling is per root. The store is a
cache, so the fix in either direction is the same and is always safe: eviction, or a hand delete, costs hit rate and
never correctness.

## What to read after a boot

Three log lines, then two routes, then the counters that mean something.

```bash
grep -c 'kvtier-dev-plugin' /path/to/engine.log        # the plugin spoke at all
grep 'BOOT\[' engine.log | tail -1                     # the whole boot verdict, one line
grep 'disk boot default' engine.log | tail -1          # the spill tier's half, and how to switch it off
```

The boot verdict is one line at WARNING (`BOOT[auto] ... ram_tier=ARMED disk_tier=ARMED tier=ARMED faults=0
journal_dir=... disk_boot=ARMED MAMBA_ANCHOR[patched] restore=YES restore_gate=closed restore_ranks=0 ...`); the
token meanings are in the plugin's README (the `BOOT[auto]` line) and `DEVLOG.md`. Read `restore_gate` and
`restore_ranks` beside `disk_tier`: **arm, serve posture and worker proof are three separate gates**, so
`disk_tier=ARMED` next to `restore_gate=closed` is a legal boot, not a contradiction: the arm says bytes may be
written, the posture says requests may be served, the worker proof says a worker will fill the slots. Then:

```bash
curl -s localhost:8000/plugins/kvtier/tier?op=status | python3 -m json.tool | head -40
# expect install.installed true, armed true, boot.mode "auto", boot.tier true, patched_before_first_request true
curl -s localhost:8000/plugins/kvtier/status | python3 -m json.tool | head -20   # API-server process only
```

`/status` touches no engine (it answers even when the engine is wedged) and so is **not** the engine's version; each
payload says who is speaking via `plugin_version_scope`, and `plugin_version` is what the *process that built the
payload imported*, never installed metadata. A non-zero guard fault ring means the tier is alive and *declining*,
which is the point of arming it (`POST /plugins/kvtier/guard {"op":"captures"}`).

**Two rules about the counters, both paid for.**

- **`vllm:prompt_tokens_by_source_total{source="external_kv_transfer"}` is the only number that says tokens were
  served from the tier.** It is cumulative and it **resets at boot**: read it as a delta across the boot, or you will
  chart a huge negative. The per-rank `reads_applied` / `bytes_applied` in `$VLLM_E2_TIER_DIAG/rank-*.json` are the
  per-process witnesses (the EngineCore rollup's own spill and apply counters are structurally zero, because the
  workers write the per-rank files).
- **`offers` and `adopted` are promises, not proof.** They count what the scheduler said it would use. The test that a
  restore is *correct* is greedy output equality: one prompt with a planted code word, `temperature=0`, the flagged
  request's answer compared byte for byte against the unflagged recomputed answer, **and** the facts lined up behind
  it: `restorations > 0`, `ranks_reads_applied > 0`, and an `external_kv_transfer` delta that matches the adopted
  count. A reference shape from our box: `restorations=2`, `ranks_reads_applied=396`,
  `external_kv_transfer = 278,400 = adopted_tokens`. Output equality alone is not enough either (a synthetic
  repetition loop once produced a false "the restore is corrupt" conclusion against a control that had zero disk
  involvement), and neither counter alone is enough ever.

## The plugin's patch series, and how it relates to `patches/` here

Two unrelated patch sets share the word "patch" in this ecosystem. Do not confuse them.

| | this repository's series | the plugin's series |
|---|---|---|
| where | `patches/*.patch`, `patches/upstream/*.patch`, `release/v2.5/patches/0001-0097` | `kvtier-dev-plugin/patches/` (its canonical home) |
| what | the release's 97 commits over `v0.30.0`, plus the eight upstream-authored patches carried ahead of the base | four hunks on two files of the *installed* engine tree |
| how applied | `git am --keep-non-patch` (or the bundle) at build time, by `scripts/build-v2.5.sh` | `git apply -p1` against an existing checkout, by `patches/apply-kvtier-patches.sh` (`status\|apply\|revert`, `--dry-run`, `--expect-tree`) or by the SSH transport in the sibling tooling repo (`config/box/kvtier-patch.sh`) |
| gate | build asserts commit **and** tree hash | a per-patch `patch.manifest`: patch sha256, the target file's sha256 before and after, per-method fingerprints. **A stale fingerprint is a refusal naming both hashes; neither tool has `--force`** |

The four hunks (their `patches/README.md` is the provenance, and `--expect-tree` is the acceptance check):

| hunk | file | why |
|---|---|---|
| `0001-kvtier-pr56810` | `vllm/v1/simple_kv_offload/manager.py` | upstream PR #56810: honour `prefix_cacheable_group_ids`; without it a hybrid model whose scratch group block size does not divide the hash block size dies on `assert block_size % hash_block_size == 0` on the **first** store |
| `0002-kvtier-cpu-release-order` | same file | release CPU blocks tail-first, so a partly-reclaimed chain keeps its root |
| `0003-kvtier-disk-restore-meta` | `vllm/v1/simple_kv_offload/metadata.py` | declare `store_keys`, `store_fingerprint`, `load_paths` |
| `0004-kvtier-disk-restore` | `vllm/v1/simple_kv_offload/manager.py` | fill those fields on the scheduler side |

Order is a dependency (a hash chain on `manager.py`: 0001 → 0002 → 0004, plus a cross-file semantic dependency of 0004
on 0003), and the applier enforces it. The manifests' base and live tree hashes were pinned against the **v2.2**-era
tree; v2.5.1 records were added later, read on our box 2026-10-22, with `UNKNOWN` written where a value had not been
computed rather than a guess written in its place. On your tree the tool either matches a pinned pre-image or refuses,
so expect to re-pin. `patches/drafts/DRAFT-*` are explicitly **DO-NOT-APPLY** and live in a directory the applier and
the series test cannot see, so a draft can never disable the tool by being present.

Neither applier starts, stops, restarts or signals an engine, and neither sends it a request: a patched file is inert
until somebody boots it. Deploying the package itself is likewise not inert (a module the running process has not yet
first-imported loads new bytes beside old objects), so **install or upgrade the plugin in the same window as a
restart**.

## Access model for the dev routes

Stated plainly, because a reviewer must not misread this as a token-gated surface.

- **There is no credential, and as of 0.2.72 there is no address gate either.** Since 0.2.68 the plugin does not
  read `VLLM_E2_TIER_DEV_TOKEN` for access control at all; a value set there is ignored and the boot logs one
  `WARNING` saying so. Since 0.2.72 the **peer-address check is gone**, including `VLLM_E2_TIER_DEV_ALLOW`, whose
  name is now recognised purely so a boot can report that setting it does nothing. It was removed rather than
  tightened because it could not fail the way a gate must. The sanctioned reader of these routes reaches them by
  running `curl 127.0.0.1:8000` **on the box itself** (the proxy drives it over SSH), so the traffic that must get
  through presents as the loopback peer - the address it checks is the one it has to admit, and every other local
  process on the box (a sidecar, any unit running there) is admitted by the same rule. Meanwhile an address that is
  *not* local - the only kind an access control would actually filter - could only arrive by being put on the
  allowlist, which is the operator opening the port rather than a gate holding it shut. Headers were never consulted and still are not.
- **The one gate is an opt-in, not an access control.** `VLLM_E2_TIER_DEV` off ⇒ every route answers `503` and the
  surface does not exist. On ⇒ **every route answers every client that can reach the port, authenticated by
  nothing**, over the LAN or whatever that port is routed to. Two boot `WARNING`s say so and `/status` publishes it
  as `exposure`, `peer_gate`, `allow_env_gates: false` and `mutating_ops_refused_on_get`, so the posture is readable
  rather than remembered.
- **`/plugins/*` is outside vLLM's `--api-key`**, which covers `/v1`, `/v2`, `/inference` and `/cohere` only. A key
  in front of `/v1` protects nothing here. Everything is under `/plugins/kvtier`, never `/v1/*`, so an operator
  putting an authenticating proxy in front of the engine can deny this prefix explicitly and see that they did.
- **Network protection is the operator's job, not the plugin's.** Enable the surface only while you are working on
  it, and disable it when you are done; if the engine binds `0.0.0.0` (README, Quick start), an enabled dev surface
  is reachable from the network with no further check. The transport is plaintext HTTP with no TLS. Do not put
  uvicorn's proxy-headers middleware in front of it - not because a gate would read the rewritten peer, but
  because a surface with no credential should not be given a middleware that makes its own address ambiguous.
- **What the plugin does enforce is method, not identity.** Mutating verbs (`arm`, `disarm`, `offers`,
  `read-threads`, spill and reload verbs) are refused `405` on `GET`, from one list of truth
  (`route_policy.py`) rather than one answer per route - before 0.2.72 that question had one answer per route, and
  five of them answered it nowhere, so `GET /guard?op=arm` really did arm guard mode. `/disk` is the one route with
  **two** layers: the route's own refusal plus an engine-side `safe_method` check in `disk_store` that re-reads the
  same list, so a mutating op admitted by one half is caught by the other. Everywhere else there is one layer, and a
  property test asserts both halves read the same source so a verb cannot be added to a dispatch table without being
  named as mutating. **This is a defence against a mis-clicked `curl`, a browser's image request, and a route
  author's omission - it is not authentication and must never be described as one.**

## Open questions we have not answered

- Whether the 0.2.64 run-end anchor patch and this fork's own v2.5.1 retention work overlap: both mark or retain
  boundary state, and we have not compared their effects on this model's orphan rate.
- Whether a `--prefix-cache-retention-interval` changes what the tier can restore here, or only what the GPU pool
  keeps.
- Why mirror mode refused 776 of 5,484 spill offers on a live boot of ours, with the per-rank split naming `expired`
  (191) and `over_tick_cap` (1) rather than `ceiling` or `queue_full`. It is carried in our own work queue, not
  resolved here.

## The pool cost, which is the number to decide on

The GPU KV pool, three boots of ours at `--kv-cache-memory 4100000000`:

| boot | GPU KV pool | source |
|---|---|---|
| v2.2 line, tier mounted | 806,792 tokens | our box, at that line's own geometry; identical to this repository's published v2.2 figure (`docs/reference.md:315`) |
| **v2.5.1, tier mounted** | **764,843 tokens** | our box, at `--block-size 4096` |
| v2.5.1, tier off, `--use-replayssm` | 924,993 tokens | our box, on a **tier-off** boot with the flag set - not the tier-mounted serving boot two rows up; identical to the reference host's published figure (`docs/reference.md:192`, `:296`) |

The gate that closes the gap between the last two rows is `vllm/v1/core/kv_cache_utils.py::_side_cache_layout_enabled`,
which returns `False` whenever `kv_transfer_config is not None` — blanket, no connector-name check, no capability flag
— while `VLLM_E2_SIDE_CACHE_LAYOUT` only switches the layout **off**. There is no override, so a boot carrying a
connector can never run the side-cache layout, and that layout is the fork's own answer to the two side caches: its
own commit measures 806,792 → 853,994 at the same budget (`docs/reference.md:319`). So a boot that mounts the
tier at this geometry reports **17.3 % fewer pool tokens than the published v2.5.1 figure**.

Two honesty notes on that table, both of which we would rather state than smooth over:

- the last two rows hold the pin and the block size constant but **not** the flag: they are the two v2.5.1
  configurations that exist (connector on and flag off, connector off and flag on), so the 17.3 % is the price of the
  pair of choices, not of the connector alone. The layout half of it is attributed by reading the gate, not by an A/B.
- the first row is *not* a same-block comparison: 806,792 is the v2.2 line's published figure at its own geometry, so
  the −5.2 % between row one and row two belongs to v2.5.1's changes and we do not attribute it further here.

The plugin carries a draft (`patches/drafts/DRAFT-0005-side-cache-layout-for-kvtier/patch.manifest`) that would let its
connector opt into the layout **by name** while every foreign connector keeps the layout excluded exactly as written.
Its precondition (the connector must first recognise and refuse the layout's slot-packed ring group and its
mixed-replication merged group, or it would address a slot at the wrong width and report success) landed in 0.2.71
(`disk_restore.CONTRACT_SLOT_PACKED`, `CONTRACT_MIXED_REPLICATION`; refusals `slot_packed_group`,
`mixed_replication_group`). It is still a draft: unbuilt, ungraded, and by its own accounting it would buy at most the
fork's own 806,792 → 853,994 step. The remaining distance to 924,993 belongs to `--use-replayssm`, which is [the flag
decision above](#flag-limitations), and no plugin patch changes that.
