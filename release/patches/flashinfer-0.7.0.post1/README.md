# FlashInfer 0.7.0.post1: sm_86 FP8 small-q verify tile

[0001-sm8x-fp8-smallq-verify-tile-070.patch](0001-sm8x-fp8-smallq-verify-tile-070.patch)
packages the measured 27B MTP verify tile for `flashinfer-python==0.7.0.post1`.
Patch SHA-256: `8fd81372ec9a661faec3b0d2e08763dcb96cc7acb6559742b0265c20c0d4223f`.
This is the measured revision 2 patch.

The planner override is scoped to exactly sm_86, E4M3 KV, QK/VO head dimensions 256,
GQA group 6, and uniform query length 4 or 5 (MTP K=3 or K=4, 24 or 30 packed rows).
It selects CTA32 with a 2-Q-warp by 2-KV-warp layout. Other shapes retain the stock
planner. The 0.7.0 port includes the finite tile-path accumulator start required
by the upstream change to IEEE infinity.

Apply from the repository root:

```bash
bash release/apply-flashinfer-patch.sh /path/to/venv
```

The helper checks the installed version and import origin, refuses
`flashinfer-jit-cache`, and verifies whole-file SHA-256 for all five targets before
and after patching. An exclusive `flock` serializes classification, apply and
post-checks. Exactly patched trees pass an idempotent re-run; edited, partial and
rev-1 (`_sm8x_smallq_v5`) trees are refused. Reinstall pristine 0.7.0.post1 before
retrying a refused tree. GNU patch applies with `--forward --fuzz=0`; the finished
patch must also reverse cleanly. Patched FA2 modules use the revision-specific
JIT suffix `_sm8x_smallq_070_v1` to avoid reusing stock or earlier tile modules.

Installing the patch leaves the tile **OFF**. Only a launcher setting
`FLASHINFER_SM8X_FP8_SMALLQ=cta32` enables the release tile; unset, empty and `0`
retain stock behavior. The value is frozen on the first plan, so restart to change
it. The byte-identical patch also retains `cta16` as a q_len-4 experimental
discriminator; release launchers use `cta32`.

## Opt-in: workload-aware tile (default off)

This section describes the 0.7.0.post1 series kept for history; the 0.7.2.post1 installer refuses this wheel pin, and the current series is under [`../flashinfer-0.7.2.post1/`](../flashinfer-0.7.2.post1/README.md).

[0002-workload-smallq-plan-mode.patch](0002-workload-smallq-plan-mode.patch) applies
on top of 0001 and lets the 27B V2 builder choose the tile per plan: cta32 for a
proven pure verify batch, stock beside prefill. **The default install does not
apply it**: `release/install-env.sh` and `release/apply-flashinfer-patch.sh` above
install and verify 0001 only, byte for byte as before.

```bash
MERLIN_FLASHINFER_WORKLOAD_TILE=1 FLASHINFER_WHEEL=/path/to/flashinfer_python-0.7.0.post1-py3-none-any.whl \
  release/install-env.sh /path/to/new-venv
MERLIN_FLASHINFER_WORKLOAD_TILE=1 release/install-env.sh --verify /path/to/venv
```

- `release/apply-flashinfer-workload-patch.py` (run with the venv's Python) installs
  the ordered series 0001+0002 from the pinned wheel, SHA-256
  `c7adf826568d61fc1b7d3aadd4cae387a35a138bfc08e2deca2f34ecfa280716`. Without a wheel
  argument or `FLASHINFER_WHEEL` it downloads that exact wheel from PyPI. It accepts
  only complete pristine, v1 or v2 trees of the eight touched files, checks import
  origin, version and JIT-cache exclusions, and serializes with the same lock as
  the default helper. Edited or mixed trees fail closed.
- `--verify` is offline: it compares the eight installed files with the embedded
  v2 SHA-256 table and changes nothing. A venv that no release installer has
  patched fails with a clear message.
- v2 FA2 modules use the JIT suffix `_sm8x_smallq_070_v2`, so the first boot after
  switching series cold-JITs them. To go back to the default, reinstall
  flashinfer-python 0.7.0.post1 (or build a new venv) and run the default install.

Serving: `MERLIN_FLASHINFER_WORKLOAD_TILE=1` (registered in `vllm/merlin/envs.py`,
default 0) only changes anything when `FLASHINFER_SM8X_FP8_SMALLQ=cta32`. Then the
V2 builder plans cta32 only when the per-request query lengths and the original
scheduler flags prove pure uniform verify, and stock otherwise. With any other
process switch value the legacy path applies unchanged (unset, empty or `0` stock;
`cta16` the q_len-4 discriminator), with ordinary FULL graphs. The knob refuses the
V1 runner, ubatching/DBO, and a FlashInfer without patch 0002. With the knob unset
or 0 the engine's wrapper keys, graph keys and compile-cache factors are the same
as without the knob. Restart between configurations; never change either switch in a
serving process.

Mode -1 preserves legacy planning. Explicit mode 0/1 is carried in the FA2
plan/workspace ABI, decode-wrapper and plan-cache identity, and V2 FULL-graph
identity. Mode 0 uses an eager wrapper and cannot replay FULL graphs. PIECEWISE
keys are unchanged: attention runs eagerly outside those graphs. Each rank logs
`[flashinfer-workload-tile] <event> count=N` for the first and every 1000th plan
per mode, FULL fallback and new mode-keyed wrapper. The native build and serving are
qualified by GPU tests, not by shipped CI.

After a generation request with speculative decoding finishes, check the current
server boot's planner proof for every TP rank (q_len 5 is the default):

```bash
bash release/check-tile-proof.sh /path/to/server.log 5 --tp 2
```

The checker requires exactly one complete CTA32 proof line per visible CUDA device
0 through TP-1 after the last engine-start line. These lines prove planner choice;
they do not prove kernel execution or identify the installed patch revision.

Measured on 2026-10-15 with Qwen3.8-27B, vLLM 0.32 and RTX 3090s (sm_86):
ITL medians in ms over three repetitions per context depth:

| Configuration | 4K | 32K | 131K | 131K without tile | 0.30 reference, 131K |
|---|---:|---:|---:|---:|---:|
| TP2 | 19.99 | 21.41 | 25.27 | 27.27 | 25.3 |
| TP4 | 15.83 | 16.62 | 18.37 | 19.45 | 18.4 |

The kernel validation passed, including bitwise agreement with the shipped 0.6.18
tile on 44/44 cells and stock behavior with the knob off. OFF versus pristine
serving outputs were identical on 28/28 cases. The serving validation reported a cache-provenance mismatch
only for warm-versus-cold torch AOT compile-cache selection lines. This did not
establish a numerical mismatch. ON versus OFF changes reduction
order and was not bitwise identical (25 logprob differences, 3 token divergences).

GSM8K-200 with the tile on scored 192/193/191 across three runs, mean 192.0/200
versus the v1.3.0 reference of 196/200: -2.0 percentage points, 95% Newcombe CI
[-5.89, +1.61]. A same-day 0.32 tile-off quality run remains outstanding.
These are the 2026-10-15 measurements; packaging did not rerun GPU gates.
