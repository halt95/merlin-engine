# FlashInfer 0.7.2.post1: sm_86 FP8 small-q verify tile

This directory ports the 0.7.0.post1 series in `../flashinfer-0.7.0.post1/` to
`flashinfer-python==0.7.2.post1` (wheel SHA-256
`bfdc18b9e1e4d8d0730fe4343dc0e284cd0c160086de80f99888bc4867922d49`), for the vLLM 0.33 engine.

| Patch | SHA-256 | Touches |
|---|---|---|
| [0001-sm8x-fp8-smallq-verify-tile-072.patch](0001-sm8x-fp8-smallq-verify-tile-072.patch) | `6c5801f87d2a3d0f956448b1cfed86f51c793779807613e88e020559ec10eb7e` | 5 files |
| [0002-workload-smallq-plan-mode-072.patch](0002-workload-smallq-plan-mode-072.patch) | `12f6b4b9def352f7bbd420006e3f99376b2700bca000f34d806f852356a22689` | 6 files, on top of 0001 |

## What changed in the port

- The added and removed lines are the 0.7.0.post1 revision-1 series, unchanged, except:
  - the JIT suffixes are `_sm8x_smallq_072_v1` (0001) and `_sm8x_smallq_072_v2` (0001+0002), so
    no module built from the 0.7.0 sources is reused;
  - 0002 drops the 25 Merlin marker comment lines on the tile path, as was done for 0.7.0
    (comment-only);
  - two comments name 0.7.2.post1.
- Kernel device code (prefill.cuh) and the planner override (scheduler.cuh) are byte-for-byte the
  0.7.0 patch lines. Both patches apply with GNU patch `--fuzz=0` and no offsets, and reverse cleanly.
- Upstream changes between 0.7.0.post1 and 0.7.2.post1 in the touched files: the FA2 plan sizes
  `tmp_v`/`tmp_s` by `ceil(padded_batch_size * cta_tile_q / gqa_group_size)` rows with a runtime
  check, and allocates `block_valid_mask` whenever CUDA graphs are on; prefill.cuh drops the
  NVFP4 per-arch repack dispatch (FP8 E4M3 paths unchanged). The plan (20) and workspace (18)
  argument counts that 0002's adapter checks are unchanged. The tile override sets `cta_tile_q`
  before the tile counts, so the new bound holds for CTA32.

## Default off

Installing 0001 leaves the tile **off**: only `FLASHINFER_SM8X_FP8_SMALLQ=cta32` enables it.
0002 is opt-in (`MERLIN_FLASHINFER_WORKLOAD_TILE=1` for `install-env.sh`; engine knob default 0 in
`vllm/merlin/envs.py`); with the knob unset the FlashInfer plan mode stays -1, the legacy path.

## Measured and not yet measured

Measured: the JIT modules built on sm_86 and the live tile proof (`release/check-tile-proof.sh`)
passed on the 27B TP2 boot with the 0.33 stack (2026-11-04). Not yet run: the kernel validation
(bitwise against the 0.7.0 tile, stock with the knob off), the tile proof on every other TP layout,
and the 27B ITL and GSM8K cards on this port. The 0.7.0 figures in `../flashinfer-0.7.0.post1/README.md`
do not carry over.
