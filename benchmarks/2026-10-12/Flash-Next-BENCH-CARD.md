# v2.5.2 card — 2026-10-12 (against v2.5.1: decode ladder, prefill, idle power)

> **Historical switch.** `VLLM_E2_GUARD_IDLE` below belongs to the measured v2.x engines. It is no longer registered on
> vLLM 0.32, where idle parking is built in; setting it does not restore polling. See the [current running notes](../../models/flash-next/docs/reference.md#running-notes).

> **Provenance.** Maintainer measurements on the reference host (4× RTX 3090, 220 W, PCIe P2P, no NVLink; driver
> 610.43.02), public Merlin checkpoint. The decode, prefill and after-ladder idle readings come from one ladder boot per
> engine (10:19-10:54 UTC), each a fresh install through its own package launcher with defaults (`GUARD=warn`; v2.5.2 adds `VLLM_E2_GUARD_IDLE=1`). v2.5.1 = published source
> `96a0d13b6e` (fresh clone of the v2.5.1 release); v2.5.2 = `de6e2c2fe9` installed from the v2.5.2 bundle. Same frozen
> ladder as the v2.5.1 benchmark, section 1. The other two idle rows, the `VLLM_E2_GUARD_IDLE=0` control, the smokes and the
> tests come from separate v2.5.2 boots earlier the same morning (09:00-09:20 UTC). Streams are client-timed; drivers, raw streams and boot logs are not published.
>
> v2.5.2 changes only the bounds-guard reader. Single-stream decode and cold prefill read as v2.5.1's within run-to-run
> noise (cell medians within ±1.8 %; individual sends vary by more; median tokens per event match to within 0.01), and
> idle cards now drop to P8: about 110 W for four idle cards against about 446 W with the always-polling reader (one
> host, one 60 s window each). Concurrent load and cache behaviour were not re-measured.

## 1. Single-stream decode ladder and prefill

T=0 with a seed, needle prompt at depth, ≤ 256 tokens with thinking on and ≤ 1,024 with thinking off, 3 sends per cell;
each cell is the median of the 3 sends (one boot per engine). With MTP, read the median event interval (ITL) together
with tokens per event, not tok/s alone.

| prompt tokens | thinking | v2.5.2 tok/s | v2.5.1 tok/s | change | ITL ms (v2.5.2 / v2.5.1) | tokens/event (v2.5.2 / v2.5.1) |
|---|---|---|---|---|---|---|
| 4,096 | on | 162.0 | 164.9 | −1.8 % | 19.04 / 18.76 | 3.04 / 3.04 |
| 4,096 | off | 121.3 | 123.0 | −1.3 % | 19.06 / 18.85 | 2.30 / 2.30 |
| 32,768 | on | 164.7 | 162.9 | +1.1 % | 18.96 / 19.13 | 3.07 / 3.07 |
| 32,768 | off | 118.0 | 116.8 | +1.1 % | 18.92 / 19.11 | 2.21 / 2.21 |
| 131,072 | on | 168.0 | 171.0 | −1.8 % | 19.55 / 19.40 | 3.20 / 3.20 |
| 131,072 | off | 120.3 | 121.1 | −0.7 % | 19.55 / 19.51 | 2.33 / 2.34 |
| 261,120 | on | 177.2 | 177.8 | −0.3 % | 19.60 / 19.86 | 3.41 / 3.41 |
| 261,120 | off | 119.7 | 119.9 | −0.1 % | 19.60 / 19.46 | 2.33 / 2.33 |

| prefill | v2.5.2 | v2.5.1 | change |
|---|---|---|---|
| 10,187 prompt tokens | 5,020 tok/s (TTFT 2.03 s) | 5,013 tok/s (TTFT 2.03 s) | +0.1 % |
| 101,875 prompt tokens | 5,417 tok/s (TTFT 18.81 s) | 5,410 tok/s (TTFT 18.83 s) | +0.1 % |

Prefill is one send per depth per engine.

The ITL differences go both ways and stay under 0.3 ms. Boot to ready on the two ladder boots: 421 s (v2.5.2),
440 s (v2.5.1), both with a cold compile cache.

## 2. Idle power

`nvidia-smi` P-state and board power on all four cards, sampled every 5 s for 60 s after a 40 s settle;
the values are the window's last sample (the first sample differs by at most 2 W per card).

| state | v2.5.2 default (`VLLM_E2_GUARD_IDLE=1`) | v2.5.2 with `VLLM_E2_GUARD_IDLE=0` (v2.5.1 behaviour) |
|---|---|---|
| after the last response | P8 on all four; 31.3 / 22.7 / 24.2 / 32.2 W (110 W) | P2 on all four; 116.1 / 103.3 / 108.4 / 118.1 W (446 W) |
| after a resumed request (second turn) | P8 on all four; 31.1 / 22.6 / 24.1 / 32.1 W (110 W) | — |
| after the full decode ladder above | P8 on all four; 31.6 / 22.8 / 23.9 / 32.5 W (111 W) | — |

## 3. Checks on this build

- An idle-quiesce guard test file (not shipped in this repository), GPU tests included, on the four cards: 67 passed.
- Release smokes on the default boot: prefix-cache reuse on a repeated 8,414-token prompt (8,320 cached on the second
  send), tool call, reasoning and vision: all pass.
- 0 engine errors while serving on all six boots of the day (both ladder boots, the default and control boots, and two
  more v2.5.2 boots on other checkpoints).
- The v2.5.2 ladder boot logged the API server's `EngineDeadError` during the scripted drained shutdown: after every
  request had finished, after the guard's final drain reported no hits, and after all four workers had exited cleanly.
  The same shutdown message was seen on one v2.5.1 boot for the v2.5.1 card; this run's v2.5.1 boot did not log it.

## 4. One user on Hermes

Two coding sessions with the [hermes](https://github.com/NousResearch/hermes-agent) agent on v2.5.2, served through the
package launcher with defaults (a scripted read-and-edit workload, 88 requests over 18 minutes): 97.4 % and 95.5 % of
prompt tokens came from the prefix cache, by Hermes' own counters (cached / (cached + uncached) prompt tokens). One user,
not a controlled benchmark; the 8-conversation follow-up figure (92.5 %) was measured on v2.5.1.
