# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Base class for PLE layers that support cross-process CPU offload.

Synchronization between the offload process and the GPU main stream is
handled by :class:`CpuGpuSemaphore`, which uses CUDA's
``cuStreamWaitValue32`` / ``cuStreamWriteValue32`` stream memory operations.

Decode-step timeline, host-mapped PULL design
--------------------------------------------------------
Offload process (CPU, no CUDA)          GPU main stream (forward)
------------------------------          ------------------------------
forward_impl() on CPU                  ... other GPU ops ...
wait host flag == 0                    WaitValue32(host flag == 1)   <- in graph
rows -> shared pinned host slot         cuMemcpyHtoDAsync(slot -> GPU) <- in graph
publish fence; host flag = 1            return _gpu_output_buffer[:n]
                                        ... model consumes output ...
                                        WriteValue32(host flag = 0)   <- release_outputs

``release_outputs`` performs the sole reset. Safety requires serialized local
forwards on one ordered model stream: a new transaction cannot begin until the
prior release has been enqueued. If any prepared forward is abandoned or its
correspondence becomes uncertain, the worker must terminate; resetting,
retrying, or continuing could reuse stale rows or overwrite live input.
"""

import contextlib
import functools
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from typing import Any, cast

import torch
from cuda.bindings import driver as cuda_driver
from cuda.bindings.driver import CUstreamWaitValue_flags
from torch import nn

from vllm.utils.torch_utils import direct_register_custom_op

_offload_worker_flag = False

_ple_copy_capture = threading.local()


@contextlib.contextmanager
def observe_ple_copy_geometry() -> Iterator[list[tuple[str, int, int]]]:
    """Collect ``(layer, rows, bytes)`` from actual wait-copy executions."""
    if getattr(_ple_copy_capture, "records", None) is not None:
        raise RuntimeError("nested PLE copy-geometry observation is unsupported")
    records: list[tuple[str, int, int]] = []
    _ple_copy_capture.records = records
    try:
        yield records
    finally:
        del _ple_copy_capture.records


def _record_ple_copy_geometry(
    layer_name: str,
    num_tokens: int,
    row_bytes: int,
) -> None:
    records = getattr(_ple_copy_capture, "records", None)
    if records is not None:
        rows = int(num_tokens)
        records.append((str(layer_name), rows, rows * int(row_bytes)))


def is_offload_process() -> bool:
    """Return True inside the dedicated PLE CPU-offload subprocess."""
    return _offload_worker_flag


def mark_as_offload_worker() -> None:
    """Mark the current process as the dedicated PLE offload worker."""
    global _offload_worker_flag
    _offload_worker_flag = True


def _cuda_check(result: Any, operation: str) -> Any:
    """Check the ``(CUresult, ...)`` tuple returned by cuda-python calls."""
    error = result[0] if isinstance(result, tuple) else result
    if error.value != 0:
        raise RuntimeError(f"{operation} failed: {error}")
    return result


class CpuGpuSemaphore:
    """Cross-process semaphore backed by a one-element int32 CUDA tensor.

    The flag is stored in a regular GPU tensor that is shared through
    PyTorch's CUDA IPC mechanism. Both processes obtain device pointers that
    map to the same physical GPU memory, so a stream-memory wait in the GPU
    worker observes the write issued by the offload process.
    """

    RESET_VALUE = 0
    DONE_VALUE = 1

    def __init__(self, device: torch.device) -> None:
        self._flag_tensor = torch.zeros(1, dtype=torch.int32, device=device)
        self._flag_ptr: int | None = None

    @classmethod
    def from_host_mapped(
        cls, host_flag_tensor: torch.Tensor, device_ptr: int
    ) -> "CpuGpuSemaphore":
        """Semaphore on a page-locked, device-mapped shared host int32.

        ``host_flag_tensor`` keeps the shared storage alive; ``device_ptr`` is
        its device address (``cuMemHostGetDevicePointer``). Stream memory ops
        on it are executed by the GPU against host memory, so a CPU process
        with no CUDA context can signal it with a plain store.
        """
        semaphore = cls.__new__(cls)
        semaphore._flag_tensor = host_flag_tensor
        semaphore._flag_ptr = int(device_ptr)
        return semaphore

    @property
    def flag_tensor(self) -> torch.Tensor:
        """Return the tensor that backs the semaphore flag."""
        return self._flag_tensor

    @property
    def flag_ptr(self) -> int:
        """Device address used by the stream memory operations."""
        if self._flag_ptr is not None:
            return self._flag_ptr
        return self._flag_tensor.data_ptr()

    def reset(self, stream: torch.cuda.Stream | None = None) -> None:
        """Enqueue ``WriteValue32(flag=0)`` on ``stream``."""
        if stream is None:
            stream = torch.cuda.current_stream()
        _cuda_check(
            cuda_driver.cuStreamWriteValue32(
                cuda_driver.CUstream(stream.cuda_stream),
                cuda_driver.CUdeviceptr(self.flag_ptr),
                self.RESET_VALUE,
                0,
            ),
            "CpuGpuSemaphore.reset",
        )

    def signal(self, stream: torch.cuda.Stream | None = None) -> None:
        """Enqueue ``WriteValue32(flag=1)`` on ``stream``."""
        if stream is None:
            stream = torch.cuda.current_stream()
        _cuda_check(
            cuda_driver.cuStreamWriteValue32(
                cuda_driver.CUstream(stream.cuda_stream),
                cuda_driver.CUdeviceptr(self.flag_ptr),
                self.DONE_VALUE,
                0,
            ),
            "CpuGpuSemaphore.signal",
        )

    def wait_reset(self, stream: torch.cuda.Stream | None = None) -> None:
        """Enqueue ``WaitValue32(flag==0)`` on ``stream``."""
        if stream is None:
            stream = torch.cuda.current_stream()
        _cuda_check(
            cuda_driver.cuStreamWaitValue32(
                cuda_driver.CUstream(stream.cuda_stream),
                cuda_driver.CUdeviceptr(self.flag_ptr),
                self.RESET_VALUE,
                CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_EQ.value,
            ),
            "CpuGpuSemaphore.wait_reset",
        )




def _ple_offload_wait_impl(
    sem_flag_tensor: torch.Tensor,
    gpu_output_buffer: torch.Tensor,
    hidden_states: torch.Tensor,
) -> None:
    """Wait for the CPU result without releasing its output buffer."""
    stream = torch.cuda.current_stream()
    cuda_stream = cuda_driver.CUstream(stream.cuda_stream)
    dev_ptr = cuda_driver.CUdeviceptr(sem_flag_tensor.data_ptr())
    _cuda_check(
        cuda_driver.cuStreamWaitValue32(
            cuda_stream,
            dev_ptr,
            CpuGpuSemaphore.DONE_VALUE,
            CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_EQ.value,
        ),
        "cuStreamWaitValue32(done)",
    )


def _ple_offload_wait_fake(
    sem_flag_tensor: torch.Tensor,
    gpu_output_buffer: torch.Tensor,
    hidden_states: torch.Tensor,
) -> None:
    """Represent the side-effect-only wait during dynamo tracing."""
    pass


direct_register_custom_op(
    op_name="ple_offload_wait",
    op_func=_ple_offload_wait_impl,
    mutates_args=["gpu_output_buffer"],
    fake_impl=_ple_offload_wait_fake,
)




_PULL_LAYERS: dict[str, "PleOffloadLayer"] = {}


def _ple_offload_wait_copy_impl(
    gpu_output_buffer: torch.Tensor,
    hidden_states: torch.Tensor,
    num_tokens: int,
    layer_name: str,
) -> None:
    """Resolve this process's live pointers for ``layer_name``, then wait+DMA."""
    layer = _PULL_LAYERS.get(layer_name)
    if layer is None or layer._host_output_ptr is None:
        raise RuntimeError(
            f"PLE wait_copy: no armed pull layer named {layer_name!r} in this "
            f"process (armed: {sorted(_PULL_LAYERS)})"
        )
    _ple_offload_wait_copy_raw(
        gpu_output_buffer,
        hidden_states,
        layer._sem.flag_ptr,
        layer._host_output_ptr,
        layer._host_row_bytes,
        num_tokens,
        layer_name,
    )


def _ple_offload_wait_copy_raw(
    gpu_output_buffer: torch.Tensor,
    hidden_states: torch.Tensor,
    flag_ptr: int,
    host_ptr: int,
    row_bytes: int,
    num_tokens: int,
    layer_name: str,
) -> None:
    """Wait for the host-side result, then DMA it into the GPU buffer."""
    if not 0 <= int(num_tokens) <= gpu_output_buffer.shape[0] or int(row_bytes) != (
        gpu_output_buffer.shape[1] * gpu_output_buffer.element_size()
    ):
        raise RuntimeError(
            f"PLE wait_copy out of bounds: {num_tokens} rows x {row_bytes} B vs "
            f"GPU buffer {tuple(gpu_output_buffer.shape)} {gpu_output_buffer.dtype}"
        )
    nbytes = int(num_tokens) * int(row_bytes)
    _record_ple_copy_geometry(layer_name, num_tokens, row_bytes)
    stream = cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)
    _cuda_check(
        cuda_driver.cuStreamWaitValue32(
            stream,
            cuda_driver.CUdeviceptr(flag_ptr),
            CpuGpuSemaphore.DONE_VALUE,
            CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_EQ.value,
        ),
        "cuStreamWaitValue32(host flag done)",
    )
    if nbytes > 0:
        # host_ptr is this worker's host address of page-locked (registered)
        # memory, so the copy is a true async DMA and is capturable.
        _cuda_check(
            cuda_driver.cuMemcpyHtoDAsync(
                cuda_driver.CUdeviceptr(gpu_output_buffer.data_ptr()),
                host_ptr,
                nbytes,
                stream,
            ),
            "cuMemcpyHtoDAsync(PLE host -> GPU)",
        )


def _ple_offload_wait_copy_fake(
    gpu_output_buffer: torch.Tensor,
    hidden_states: torch.Tensor,
    num_tokens: int,
    layer_name: str,
) -> None:
    """Represent the side-effect-only wait + copy during dynamo tracing."""
    pass


direct_register_custom_op(
    op_name="ple_offload_wait_copy",
    op_func=_ple_offload_wait_copy_impl,
    mutates_args=["gpu_output_buffer"],
    fake_impl=_ple_offload_wait_copy_fake,
)


class PleOffloadLayer(nn.Module, ABC):
    """Base class for embedding-like PLE layers that can run on CPU.

    Subclasses implement :meth:`forward_impl`, which contains the regular
    on-device computation. In a GPU worker with PLE offload enabled, subclass
    constructors are skipped so their large weights are never allocated. The
    CPU process constructs a complete copy and owns those weights instead.
    """

    _is_cpu_offloaded = False
    _gpu_output_buffer: torch.Tensor
    _sem: CpuGpuSemaphore
    # Host-mapped pull design: host address of the shared output buffer and
    # the bytes per output row. ``None`` keeps the IPC push behaviour.
    _host_output_ptr: int | None = None
    _host_row_bytes: int = 0
    _ple_offload_name: str = "<unbound>"

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Skip subclass initialization in cross-process GPU-worker mode."""
        super().__init_subclass__(**kwargs)
        original_init_obj = cls.__dict__.get("__init__")
        if original_init_obj is None or not callable(original_init_obj):
            return
        original_init = cast(Callable[..., None], original_init_obj)

        @functools.wraps(original_init)
        def guarded_init(
            self: "PleOffloadLayer", *args: object, **kwargs: object
        ) -> None:
            if not is_offload_process():
                nn.Module.__init__(self)
                return
            original_init(self, *args, **kwargs)

        cls.__init__ = guarded_init  # type: ignore[method-assign, assignment]

    @classmethod
    def get_target_device(cls) -> torch.device:
        """Return CPU: the table lives in the CPU-only offload process."""
        return torch.device("cpu")

    @abstractmethod
    def forward_impl(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        *args: object,
        **kwargs: object,
    ) -> torch.Tensor:
        """Execute the actual embedding computation on the owning device."""
        raise NotImplementedError

    def get_offload_output_dtype(self, default_dtype: torch.dtype) -> torch.dtype:
        """Return the dtype used by cross-process output buffers."""
        return default_dtype

    def get_offload_output_dim(self, default_dim: int) -> int:
        """Return the last dimension used by cross-process output buffers."""
        return default_dim

    def initialize_dummy_offload_metadata(self, device: torch.device) -> None:
        """Initialize metadata needed by dummy offload execution."""

    def setup_cross_process_offload(
        self,
        gpu_output_buffer: torch.Tensor,
        semaphore: CpuGpuSemaphore,
        host_output_ptr: int | None = None,
        host_row_bytes: int = 0,
    ) -> None:
        """Configure the GPU-worker placeholder with its offload resources."""
        self._is_cpu_offloaded = True
        self._gpu_output_buffer = gpu_output_buffer
        self._sem = semaphore
        self._host_output_ptr = host_output_ptr
        self._host_row_bytes = int(host_row_bytes)
        if host_output_ptr is not None:
            # Guard: the registry key is the name the
            # traced graph carries, so it must be a real, unique layer name.
            name = self._ple_offload_name
            if name == "<unbound>":
                raise RuntimeError(
                    "PLE pull: setup_cross_process_offload on a layer with no "
                    "_ple_offload_name; the op would resolve nothing by name"
                )
            prior = _PULL_LAYERS.get(name)
            if prior is not None and prior is not self:
                raise RuntimeError(
                    f"PLE pull: two layers armed under the name {name!r}; the op "
                    "would resolve the wrong layer's pointers"
                )
            _PULL_LAYERS[name] = self

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        *args: object,
        **kwargs: object,
    ) -> torch.Tensor:
        """Wait for an offloaded result or delegate to ``forward_impl``."""
        if self._is_cpu_offloaded:
            if self._host_output_ptr is not None:
                torch.ops.vllm.ple_offload_wait_copy(
                    self._gpu_output_buffer,
                    hidden_states,
                    input_ids.shape[0],
                    self._ple_offload_name,
                )
            else:
                torch.ops.vllm.ple_offload_wait(
                    self._sem.flag_tensor,
                    self._gpu_output_buffer,
                    hidden_states,
                )
            return self._gpu_output_buffer[: input_ids.shape[0]]
        return self.forward_impl(hidden_states, input_ids, *args, **kwargs)

    def release_offloaded_output(
        self,
        stream: torch.cuda.Stream | None = None,
    ) -> None:
        """Mark the cross-process output buffer reusable on ``stream``."""
        if self._is_cpu_offloaded:
            self._sem.reset(stream)
