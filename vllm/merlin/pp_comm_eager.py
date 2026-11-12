# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Create the pipeline-parallel NCCL communicators at model-runner init.

torch creates two communicators lazily, on first use:
- the per-pair communicator of the unbatched hidden-state isend/irecv on the PP
  `device_group` (`isend_tensor_dict` / `irecv_tensor_dict`);
- the communicator of the `pp_broadcast` sibling group used by `PPHandler`.

The dummy profile run does no PP communication, so the first real step is the
first use of both, after the KV cache has been sized and allocated. Their
cudaMalloc then fails once the free memory is gone: 20 GB cards hit
`ncclUnhandledCudaError` at the first sample. Created here, after the worker's
init memory snapshot, their memory is in place before the profile and the KV
sizing, so it is accounted for like any other non-torch allocation.

The ops, groups and peers are the same as in the step, so torch caches and
reuses exactly these communicators. This has no numerical effect.
"""

from __future__ import annotations

import torch

from vllm.distributed.parallel_state import get_pp_group
from vllm.logger import init_logger

logger = init_logger(__name__)


def warm_pp_communicators(
    broadcast_group: torch.distributed.ProcessGroup,
    last_rank: int,
    device: torch.device,
) -> None:
    """Collective over the PP group: every PP rank calls it once, from
    `PPHandler.__init__`. `last_rank` is the global rank the step broadcasts
    sampled tokens from."""
    pp = get_pp_group()
    free_before = torch.cuda.mem_get_info(device)[0]
    buf = torch.zeros(1, dtype=torch.int64, device=device)
    handles = []
    # Hidden states flow from rank i to i + 1, as unbatched isend/irecv on
    # device_group with global peer ranks.
    if not pp.is_last_rank:
        handles.append(
            torch.distributed.isend(
                buf, dst=pp.ranks[pp.rank_in_group + 1], group=pp.device_group
            )
        )
    if not pp.is_first_rank:
        recv_buf = torch.empty_like(buf)
        handles.append(
            torch.distributed.irecv(
                recv_buf, src=pp.ranks[pp.rank_in_group - 1], group=pp.device_group
            )
        )
    for handle in handles:
        handle.wait()
    torch.distributed.broadcast(buf, src=last_rank, group=broadcast_group)
    torch.cuda.synchronize(device)
    free_after = torch.cuda.mem_get_info(device)[0]
    logger.info(
        "PP communicators created at init: %.1f MiB device memory",
        (free_before - free_after) / (1 << 20),
    )


def warm_tp_pcie_ipc_communicator(device: torch.device) -> None:
    """Create the TP `device_group` communicator that the FlashInfer PCIe IPC
    all-reduce setup uses, now. Called once per worker from `init_device`,
    after the model runner is built.

    `FlashInferPcieIpcAllReduce.setup` runs at kernel warm-up, after the KV
    cache is allocated. Its `PcieIpcAllReduceWorkspace` validates its arguments
    with `all_gather_object` on the TP `device_group`. That is the first torch
    collective on the group, so the group's NCCL communicator is created
    lazily right there and can fail for lack of memory (20 GB cards). This
    function issues the same all_gather on the same group, only when that
    backend is on, so other rows don't create a communicator they never use."""
    from vllm.distributed.parallel_state import get_tp_group

    tp = get_tp_group()
    if tp.world_size <= 1:
        return
    fi_comm = getattr(tp.device_communicator, "fi_pcie_ipc_ar_comm", None)
    if fi_comm is None or getattr(fi_comm, "disabled", False):
        return
    free_before = torch.cuda.mem_get_info(device)[0]
    local = torch.zeros(1, dtype=torch.int64, device=device)
    gathered = [torch.empty_like(local) for _ in range(tp.world_size)]
    torch.distributed.all_gather(gathered, local, group=tp.device_group)
    torch.cuda.synchronize(device)
    free_after = torch.cuda.mem_get_info(device)[0]
    logger.info(
        "TP PCIe IPC communicator created at init: %.1f MiB device memory",
        (free_before - free_after) / (1 << 20),
    )
