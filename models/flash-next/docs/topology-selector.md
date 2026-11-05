# Topology selector

For Merlin 3.0.2, built on vLLM 0.33.0. The controls below describe the selector and launcher;
synthetic-topology coverage does not qualify performance on NVLink or other unmeasured hardware.

> **Opt-in.** The selector runs only when you set `AUTO_TOPO=1` (or `AUTO_TOPO=strict`). With `AUTO_TOPO`
> unset or `0`, the default, `models/flash-next/serve.sh` never calls it and behaves exactly as it did without it: the
> same fixed settings, the same environment, the same vLLM command.
>
> Enable it if your host is not like the reference one: NVLink-bridged GPU pairs, more than four GPUs, or more than
> one CPU socket (NUMA node). On the reference layout (below) it picks the fixed settings anyway.
>
> ```
> AUTO_TOPO=1 models/flash-next/serve.sh /path/to/checkpoint
> ```

Since Merlin 3.0.0 the integration described here lives in the one launcher, `tools/merlin/launch.py` (ported line for
line from the v3.0.0 `serve.sh`; `models/flash-next/serve.sh` is a shim for it), and applies to every four-card
P2P-on Flash-Next-family row (a P2P-off row ignores `AUTO_TOPO`). `models/flash-next/serve.sh` runs the model on four GPUs: tensor parallel 2 x pipeline parallel 2. With `AUTO_TOPO=1`,
before it starts vLLM it calls `models/flash-next/topo_select.py`, which looks at the machine's GPUs and decides three things:

- **the GPU order.** The four GPUs form two tensor-parallel pairs, (1st, 2nd) and (3rd, 4th). The pipeline edges run
  from the 1st to the 3rd GPU and from the 2nd to the 4th. The selector keeps your order unless another order is
  strictly better for these pairs (for example, NVLink-bridged cards that are not next to each other in bus order).
- **the peer-to-peer (P2P) and NCCL settings**: `VLLM_SKIP_P2P_CHECK`, `NCCL_P2P_LEVEL` and `NCCL_PROTO`.
- **optionally, a NUMA binding.** With `AUTO_TOPO_BIND=1`, it prefixes the server with `numactl` to keep its CPUs
  and memory on the NUMA node the four GPUs are attached to.

On the reference layout (exactly four GPUs, one NUMA node, P2P working between every pair, no NVLink, no custom
order) the result is exactly the fixed settings earlier releases used: `CUDA_VISIBLE_DEVICES=0,1,2,3`,
`CUDA_DEVICE_ORDER=PCI_BUS_ID`, `VLLM_SKIP_P2P_CHECK=1`, `NCCL_P2P_LEVEL=SYS`, `NCCL_PROTO=Simple`, no prefix.

What it does **not** do:

- It never changes which GPUs you picked, only their order.
- It never creates a CUDA context and never allocates GPU memory. It reads NVML, `/sys` and `/proc` only.
- It measures nothing. P2P support comes from NVML's report. When the selector sets `VLLM_SKIP_P2P_CHECK=0`, vLLM
  runs its own P2P test once and caches the result.
- It never blocks a launch that works today. If anything is unknown, or the selector fails, times out or writes
  something unusable, serve prints one `topo: ...` line and uses the fixed settings above. The only exception is
  `AUTO_TOPO=strict`, below.

## Controls

All are environment variables read by `serve.sh` and the selector. Everything except `AUTO_TOPO` itself has
an effect only when the selector is on.

| variable | values | effect |
|---|---|---|
| `AUTO_TOPO` | unset or `0` (default) | do not run the selector. Serve behaves exactly as before the selector existed |
| | `1` | run the selector; fall back to the fixed settings on any doubt |
| | `strict` | run the selector, but refuse to start (exit 1) instead of guessing: when more than four GPUs are visible and `CUDA_VISIBLE_DEVICES` is not set (even with `GPU_SET=auto`), and when the selector cannot run (fails, times out, or writes an unusable file) |
| `AUTO_TOPO_NCCL` | unset or `legacy` (default) | keep `NCCL_PROTO=Simple` everywhere |
| | `default` | on a GPU set where no tensor-parallel pair has P2P, unset `NCCL_PROTO` and let NCCL choose. `Simple` has not been compared with NCCL's own choice without P2P |
| `AUTO_TOPO_BIND` | unset or `0` (default) | no binding |
| | `1` | bind the server to the GPU set's NUMA node when every precondition holds (see below) |
| `GPU_SET` | unset (default) | with more than four GPUs visible and `CUDA_VISIBLE_DEVICES` unset, use GPUs 0-3 in bus order and print a warning |
| | `auto` | in that same situation, pick the best four idle GPUs (no compute processes, at most 1 GiB used). A GPU that is idle when checked is not reserved |
| `CPU_SET` | a cpulist, e.g. `24-35,72-83` | further restricts the CPUs used for binding. Set but empty means no CPUs, so binding is skipped |
| `TOPO_TIMEOUT` | whole seconds, 1-600 (default 20) | how long discovery may take. Serve stops the selector 5 s later if it is still running, and kills it 5 s after that. Any other value prints a note and uses 20 |

Any other `AUTO_TOPO` value prints a `topo: note:` line and leaves the selector off. (Run by hand, the selector
itself treats an unset `AUTO_TOPO` as `1`; only serve's default is off.)

A `CUDA_VISIBLE_DEVICES` that is set but empty, or only whitespace, counts as unset, as it always did.

The selector writes a diagnostics JSON file for each decision: the GPU set, the link class of every pair, the top
candidate orders with their scores, and every variable's final value with where it came from (`caller`, `selector`
or `legacy`). Serve puts it under `$CACHE_ROOT/topology/`, or under `.vllm-cache/topology/` in the repository
when `CACHE_ROOT` is unset. That is the same cache root serve gives vLLM as `VLLM_CACHE_ROOT`. Run by itself, the
selector uses `$VLLM_CACHE_ROOT/topology`, then `$CACHE_ROOT/topology`, then `~/.cache/flash-next/topology`.
`--json-dir` overrides all of these.

## Behaviour changes with the selector on

None of these apply with `AUTO_TOPO` unset or `0`.

- **Your settings now win.** With the selector on, a value you set yourself for `NCCL_PROTO`,
  `NCCL_P2P_LEVEL`, `VLLM_SKIP_P2P_CHECK` or `CUDA_DEVICE_ORDER` is kept. Earlier releases overwrote them.
  Leaving `AUTO_TOPO` unset (or `0`) keeps the old behaviour exactly.
- **A reorder can change greedy output.** Greedy decoding is repeatable for a given GPU order and compile cache.
  A different order puts ranks on different physical cards and can change the order of floating-point reductions, so
  greedy output can differ between two orders. Compare runs only when they used the same `CUDA_VISIBLE_DEVICES`.
  The diagnostics file records it.
- **UUIDs on the reference host.** If you select the four GPUs by UUID, the GPU set is no longer the reference layout
  by definition. The selector then sets `VLLM_SKIP_P2P_CHECK=0`: the first boot runs vLLM's P2P test, and later boots
  use the cached result.

## Choosing a GPU set on a machine with more than four GPUs

List the GPUs with their UUIDs:

```
nvidia-smi -L
nvidia-smi topo -m
```

Pick four, ideally on one NUMA node and, if the machine has NVLink bridges, as two bridged pairs. Pass them by UUID
(this works with the selector off too; add `AUTO_TOPO=1` to let it check the order and the P2P settings):

```
CUDA_VISIBLE_DEVICES=GPU-aaaaaaaa-...,GPU-bbbbbbbb-...,GPU-cccccccc-...,GPU-dddddddd-... \
  models/flash-next/serve.sh /path/to/checkpoint
```

UUIDs are safer than numbers because they do not depend on enumeration order. Use full UUIDs: the selector accepts
a unique prefix and passes the full UUIDs on, but when it falls back to the fixed settings your value reaches vLLM
unchanged, and vLLM needs full UUIDs. Numeric
indices are read in PCI bus order (`CUDA_DEVICE_ORDER=PCI_BUS_ID`). If you set `CUDA_DEVICE_ORDER` to anything else,
the selector leaves the GPU set alone.

**NVLink.** You do not have to order the GPUs yourself. If two bridged cards are not adjacent in your list, the
selector reorders the list so that each bridged pair becomes a tensor-parallel pair, and prints `order reordered`.
This needs `AUTO_TOPO=1`; with the selector off (the default), your order is used as given. The pairing is derived
from the topology NVML reports and covered by unit tests on a synthetic bridged topology; it has not been run on
NVLink-bridged hardware.

## NUMA binding

`AUTO_TOPO_BIND=1` produces `numactl --physcpubind=<cpus> --preferred=<node> --` in front of the vLLM command, only
when all of these hold:

- all four GPUs report the same known NUMA node;
- `numactl` is installed;
- at least 8 CPUs remain after intersecting the process's allowed CPUs, that node's CPUs and `CPU_SET`;
- the node is in the process's allowed memory nodes (`Mems_allowed_list`);
- the node has at least 1.2 x 69 GiB free (the selector's existing heuristic; the approximately 69 GiB resident
  estimate is a **v3.0.0 (0.30) figure**, inherited from earlier releases and not re-measured on 0.32);
- a dry run, `numactl --physcpubind=<cpus> --preferred=<node> true`, succeeds.

Otherwise there is no prefix, and the diagnostics file names the first precondition that failed. Many containers deny
the memory-policy system call, so the dry run fails there and binding is skipped. Do not also pass vLLM's own
`--numa-bind` when a prefix is in use.

## Checking the decision without starting the server

With the selector on, serve takes a snapshot of your settings (`TOPO_CALLER_*` variables) before it applies any
defaults, and the selector reads your intent only from that snapshot. The simplest dry run is therefore through serve
itself:

```
AUTO_TOPO=1 SERVE_PRINT_ENV_AND_EXIT=1 models/flash-next/serve.sh /path/to/checkpoint
```

Without `AUTO_TOPO=1` the same command prints the fixed settings and does not run the selector.

In the container: `MODEL_DIR=/path/to/Qwen3.8-Flash-Next-W4A16-Merlin docker compose run --rm -e AUTO_TOPO=1 -e
SERVE_PRINT_ENV_AND_EXIT=1 flash-next`. The image does not include `numactl`, so `AUTO_TOPO_BIND=1` is always
skipped there.

This runs serve's usual checks and the selector, prints the final values and the exact launch prefix, and exits
without starting vLLM. It still writes a diagnostics file.

To run only the selector, with the venv's Python, and set the snapshot yourself. For example, to see what it would
do with a given GPU set:

```
env TOPO_CALLER_SET_CUDA_VISIBLE_DEVICES=1 TOPO_CALLER_CUDA_VISIBLE_DEVICES=GPU-aaaa...,GPU-bbbb...,GPU-cccc...,GPU-dddd... \
  ./venv/bin/python models/flash-next/topo_select.py --dry-run
```

`--dry-run` prints the summary line and the env file serve would source, and writes nothing. Variables without a
`TOPO_CALLER_SET_<NAME>=1` entry count as unset. The six snapshot variables are `CUDA_VISIBLE_DEVICES`,
`CUDA_DEVICE_ORDER`, `VLLM_SKIP_P2P_CHECK`, `NCCL_P2P_LEVEL`, `NCCL_PROTO` and `PLE_HOME`.

Run this way, the selector treats an unset `AUTO_TOPO` as `1`.

The selector exits 0 when it wrote a decision (selector or fixed settings), 2 on a usage error, 3 on a strict-mode
refusal, 4 on a discovery timeout and 5 on an internal error. Serve treats every non-zero code except 3 as "use the
fixed settings", unless `AUTO_TOPO=strict`.
