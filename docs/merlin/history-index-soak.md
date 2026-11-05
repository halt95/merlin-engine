# HistoryIndex slimming and how to read its memory

This document describes the smaller HistoryIndex and how to measure host memory over a long run with repeated, spaced
262K phases. It records no result.

## What changes

HistoryIndex is metrics-only. One insertion-ordered `{hash: in_prompt}` dict
replaces two OrderedDicts. Re-add refreshes the shared LRU and preserves an
earlier prompt flag. The existing geometry-derived capacity, 64-chain floor,
and four-times-cache margin stay unchanged. Capacity now bounds the union of
prompt and decode hashes.

Every removed key increments `history_truncated_full`; a removed key flagged
as prompt also increments `history_truncated`. Decode can now evict prompt
history before the old independent prompt cap would have filled. The prompt-history ceiling
stays sticky: after any prompt truncation, rho_bar, T*, E and E_tier are invalid.
A nonzero full-only truncation does not itself invalidate the prompt ceiling.
Do not compare those invalid metrics as performance regressions.

The retained-size regression counts index tables plus each retained 32-byte
key once against the old two-OrderedDict representation. It does not measure
process RSS, allocator fragmentation, or the long-run memory plateau.

## Measuring a long run

Record the exact commit, checkpoint revisions, host boot, GPU unlock and P2P state, profile, topology, hash unit, tier
size, retention and all explicit knobs. Suggested coverage is 27B TP2 (16-token hashes) and Flash-Next MTP with
`--swarm` (64-token hashes). Read back the Ready concurrency and the `SimpleCPU tier history` capacity line. A lane whose
run is too short to give a multi-hour plateau window has not been qualified; say so rather than declaring it done.

Keep the server processes alive across history fill and every phase. Continue
the 24-session workload with fresh, salted computed chains so this exercises
history turnover, not just repeated hits. Use at least three spaced
262K phases, at least 30 minutes apart, with at least two after history
saturation and at least one hour of late observation. The comparison baselines
must be at the same delay after each phase under comparable load. If the
history does not saturate, the window is too short, or a process restarts,
the plateau claim is incomplete.

Each phase includes the exact 262K row (261,888 prompt + 128 output tokens,
exact=true), a fresh salted cold chain with zero prefix hits checked, and a
return exercised from host using a fixed eviction sequence.
Save the GPU-hit G and host-return H outputs and their hashes, require H == G,
and confirm host restore with engine completion counters rather than a client
label alone. State thinking per row; the agents workload uses thinking off,
and reasoning rows use the served medium effort with engine and client read-back.
Record phase start and end times, fresh computed tokens, restores and valid rows.

### Readings to save

Collect these on one timestamped timeline, with run, phase and cycle IDs:

- Per-process samples at least every 10 minutes, plus immediately before and
  after each 262K phase: PID, process start time, role (API, EngineCore, each
  TP/PP worker/rank), VmRSS, VmLck, RssAnon, RssFile and RssShmem from
  `/proc/<pid>/status`. Enumerate every serving descendant, keep raw files and
  report missing samples and PID replacements. Continue through clean shutdown.
- Aggregate RSS and unlocked RSS (VmRSS minus VmLck) over exactly the same
  process set and denominator throughout. Show each PID separately.
  Summed worker RSS can count shared pages repeatedly; keep that series and add
  PSS from `smaps_rollup` when available to explain RssShmem steps, rather than
  changing the accounting mid-run.
- Raw tier and series lines and cumulative `history_truncated` and
  `history_truncated_full`, capacity and hash unit, fresh finished and computed tokens,
  outcome counts, adopted GPU and host tokens, completed and failed restores, and
  client error, timeout and stall counts. Record the first full and first prompt
  truncation timestamps. Rising full truncation proves the shared cap was
  reached; counters alone do not measure the history's bytes.
- First-262K and later-cycle RSS deltas for each PID, split into anon, file and shmem
  and locked and unlocked memory. Keep identical-delay post-phase baselines,
  phase-local peaks and recovery readings. Save output identity checks,
  tracebacks and fatal lines, and the shutdown audit.

### How to read the result

Report the second-half drift against the 0.1 percent/hour soak limit, plus first-half and late post-saturation
slopes. State each time window and baseline RSS explicitly: drift is 100 * RSS change / (baseline RSS * elapsed hours).
Also report unlocked drift and absolute MiB/hour per PID.

Separate (1) history-fill growth, (2) the first 262K high-water step and
(3) later-cycle increments. A plateau needs repeated late baselines that
stay within the limit while history truncation continues and
host restores really complete. A small whole-run slope without saturation,
or a single 262K phase, does not show a plateau. The smaller index is not
evidence that the earlier unexplained worker and API growth is fixed.

If workers continue growing after the first steps, save early and late
`smaps_rollup` readings. An otherwise identical fresh-process `MALLOC_ARENA_MAX=2`
comparison is the follow-up candidate; it is a separate arm, not a new default. A miss
against the limit, persistent late growth or an incomplete lane is a finding to report, not a pass.
