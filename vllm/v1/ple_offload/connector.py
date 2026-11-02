# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Exchange PLE data between a GPU worker and the CPU-offload process."""

import os
import queue
import threading
from dataclasses import dataclass
from multiprocessing.reduction import ForkingPickler
from typing import Any

import msgspec
import torch
import torch.nn as nn
import zmq
from cuda.bindings import driver as cuda_driver

from vllm.config import VllmConfig
from vllm.distributed.parallel_state import get_dp_group, get_pp_group, get_tp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.ple_offload_layer import (
    CpuGpuSemaphore,
    PleOffloadLayer,
    _cuda_check,
)
from vllm.v1.ple_offload.fatal import notify_parent_fatal, ple_fail_closed
from vllm.v1.ple_offload.progress import PleProgressMonitor
from vllm.v1.ple_offload.protocol import (
    PleOffloadRegistration,
    PleOffloadRequest,
)
from vllm.v1.ple_offload.test_fault import _FAULT
from vllm.v1.utils import record_function_or_nullcontext

logger = init_logger(__name__)

_PROCESS_LIFETIME_REGISTERED_STORAGE: list[torch.Tensor] = []


def _unlink_filename_shared_storage(
    tensor: torch.Tensor,
    reserved_filenames: set[bytes],
) -> None:
    """Reserve and unlink one already filename-shared CPU storage."""
    try:
        storage = tensor.untyped_storage()
        if storage.device.type != "cpu" or not storage.is_shared():
            return
        try:
            storage._get_shared_fd()
        except RuntimeError:
            pass
        else:
            return
        _, filename, _ = storage._share_filename_cpu_()
        if not isinstance(filename, bytes):
            filename = os.fsencode(filename)
        path = os.path.join("/dev/shm", os.fsdecode(filename).lstrip("/"))
    except Exception:
        logger.warning(
            "Failed to inspect a PLE shared-memory storage during teardown",
            exc_info=True,
        )
        return

    if filename in reserved_filenames:
        return
    try:
        storage._shared_incref()
    except Exception:
        logger.warning(
            "Failed to reserve the PLE shared-memory counter for %s",
            path,
            exc_info=True,
        )
        return
    reserved_filenames.add(filename)

    try:
        os.unlink(path)
    except FileNotFoundError:
        # Another teardown participant already removed the name. Retain the
        # reservation because this allocator must still avoid a later unlink.
        pass
    except OSError as error:
        # The name remains available, so restore the allocator's original
        # counter and permit a later release attempt to retry.
        try:
            storage._shared_decref()
        except Exception:
            # The reservation state is uncertain after a failed private call.
            # Keep it marked to avoid adding another reservation on retry.
            logger.warning(
                "Failed to roll back the PLE shared-memory counter for %s",
                path,
                exc_info=True,
            )
        else:
            reserved_filenames.discard(filename)
        logger.warning("Failed to unlink PLE shared memory %s: %s", path, error)


@dataclass(frozen=True)
class _PendingPleOffloadRequest:
    """Bind request metadata to its MRV2 D2H completion event."""

    request: PleOffloadRequest
    d2h_done_event: torch.cuda.Event
    eligible: bool = False
    transaction_id: str | None = None
    correlation_id: str | None = None


class PleOffloadConnector:
    """Connect a model runner V2 GPU worker to the PLE CPU worker."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        model: nn.Module,
        device: torch.device,
        ipc_addr: str,
        *,
        input_ids_source: torch.Tensor,
        query_start_loc_source: torch.Tensor,
        ngram_context_source: torch.Tensor | None,
    ) -> None:
        self.device = device
        self.rank = int(vllm_config.parallel_config.rank)
        self.dp_rank = get_dp_group().rank_in_group
        self.tp_rank = get_tp_group().rank_in_group
        self.pp_rank = get_pp_group().rank_in_group
        self._host_output_bufs: dict[str, torch.Tensor] = {}
        self._host_flags: dict[str, torch.Tensor] = {}
        self._reserved_host_storage_filenames: set[bytes] = set()
        self._gpu_output_bufs: dict[str, torch.Tensor] = {}
        self._layers = self._setup_layers(vllm_config, model)
        self._progress = PleProgressMonitor.from_environment(
            rank=self.rank,
            layer_names=tuple(self._layers),
        )

        # TP0 registers shared buffers with CUDA for asynchronous D2H copies.
        scheduler_config = vllm_config.scheduler_config
        self._input_ids_buf = torch.empty(
            scheduler_config.max_num_batched_tokens,
            dtype=torch.int32,
            device="cpu",
        ).share_memory_()
        self._query_start_loc_buf = torch.empty(
            scheduler_config.max_num_seqs + 1,
            dtype=torch.int32,
            device="cpu",
        ).share_memory_()
        self._ngram_context_buf = None
        config = vllm_config.model_config.hf_text_config
        ngram_context_len = int(config.ngram_size) - 1
        if ngram_context_len > 0:
            self._ngram_context_buf = torch.empty(
                scheduler_config.max_num_seqs,
                ngram_context_len,
                dtype=torch.int32,
                device="cpu",
            ).share_memory_()

        # Runner input allocations are address-stable, so bind them once and
        # pass only batch sizes through the per-forward request queue.
        self._input_ids_source = input_ids_source
        self._query_start_loc_source = query_start_loc_source
        self._ngram_context_source = ngram_context_source
        self._validate_input_sources()

        self._pinned_input_buffers: list[torch.Tensor] = []
        self._mapped_host_buffers: list[torch.Tensor] = []
        self._after_dummy = True
        self._request_queue: queue.Queue[_PendingPleOffloadRequest | None] = (
            queue.Queue(maxsize=vllm_config.max_concurrent_batches)
        )
        self._dropped_events: list[torch.cuda.Event] = []
        self._request_thread: threading.Thread | None = None
        self._request_thread_ready = threading.Event()
        self._zmq_ctx: zmq.Context | None = None
        self._registration_socket: zmq.Socket | None = None
        self._d2h_event_pool: queue.Queue[torch.cuda.Event] | None = None

        try:
            self._zmq_ctx = zmq.Context()
            self._registration_socket = self._zmq_ctx.socket(zmq.PUSH)
            self._registration_socket.connect(ipc_addr)
            payload = self._serialize_registration(vllm_config)

            with torch.accelerator.device_index(self.device.index):
                self._map_host_outputs(vllm_config)
                if self.tp_rank == 0:
                    self._pin_input_buffers()
            self._send_registration(payload, ipc_addr)

            if self.tp_rank == 0:
                with torch.accelerator.device_index(self.device.index):
                    self._d2h_event_pool = queue.Queue(
                        maxsize=vllm_config.max_concurrent_batches
                    )
                    for _ in range(vllm_config.max_concurrent_batches):
                        self._d2h_event_pool.put_nowait(torch.cuda.Event())
                self._start_request_thread(ipc_addr)
        except Exception:
            self.close()
            raise

    def _setup_layers(
        self,
        vllm_config: VllmConfig,
        model: nn.Module,
    ) -> dict[str, PleOffloadLayer]:
        """Attach output buffers and semaphores to GPU PLE placeholders."""
        layers = {
            name: module
            for name, module in model.named_modules()
            if isinstance(module, PleOffloadLayer)
        }
        if not layers:
            raise RuntimeError(
                "The PLE pull transport is on, but the model has no "
                "PleOffloadLayer"
            )

        config = vllm_config.model_config.hf_text_config
        max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        for name, layer in layers.items():
            if vllm_config.load_config.load_format == "dummy":
                layer.initialize_dummy_offload_metadata(self.device)
            out_dim = layer.get_offload_output_dim(int(config.ple_embed_dim))
            out_dtype = layer.get_offload_output_dtype(vllm_config.model_config.dtype)
            # The captured forward DMAs the host result into this device
            # buffer after the host flag is set; consumers read it as before.
            self._gpu_output_bufs[name] = torch.empty(
                max_num_tokens, out_dim, dtype=out_dtype, device=self.device
            )
            # Zero-initialised so dummy/profile forwards copy finite values.
            self._host_output_bufs[name] = torch.zeros(
                max_num_tokens, out_dim, dtype=out_dtype, device="cpu"
            ).share_memory_()
            # 16 x int32 = one 64-byte line; element 0 is the flag.
            self._host_flags[name] = torch.zeros(
                16, dtype=torch.int32, device="cpu"
            ).share_memory_()
        return layers

    def _map_host_outputs(self, vllm_config: VllmConfig) -> None:
        """Page-lock + device-map host outputs/flags and arm the placeholders."""
        for name, layer in self._layers.items():
            layer._ple_offload_name = name
            host_out = self._host_output_bufs[name]
            host_flag = self._host_flags[name]
            for buffer in (host_out, host_flag):
                self._register_host_buffer(buffer, device_map=True)
            flag_dev_ptr = _cuda_check(
                cuda_driver.cuMemHostGetDevicePointer(host_flag.data_ptr(), 0),
                "cuMemHostGetDevicePointer(PLE host flag)",
            )[1]
            out_dev_ptr = _cuda_check(
                cuda_driver.cuMemHostGetDevicePointer(host_out.data_ptr(), 0),
                "cuMemHostGetDevicePointer(PLE host output)",
            )[1]
            row_bytes = host_out.shape[1] * host_out.element_size()
            gpu_buf = self._gpu_output_bufs[name]
            if (
                gpu_buf.shape[0] != host_out.shape[0]
                or gpu_buf.shape[1] * gpu_buf.element_size() != row_bytes
            ):
                raise RuntimeError(f"PLE host/GPU buffer geometry mismatch for {name}")
            layer.setup_cross_process_offload(
                gpu_buf,
                CpuGpuSemaphore.from_host_mapped(host_flag, int(flag_dev_ptr)),
                host_output_ptr=host_out.data_ptr(),
                host_row_bytes=row_bytes,
            )
            logger.info(
                "PleOffload: host-mapped pull for %s (rows %d x %d B, flag dev "
                "0x%x, out dev 0x%x, host==dev %s)",
                name,
                host_out.shape[0],
                row_bytes,
                int(flag_dev_ptr),
                int(out_dev_ptr),
                int(out_dev_ptr) == host_out.data_ptr(),
            )

    def _pin_input_buffers(self) -> None:
        """Page-lock shared input allocations without replacing their storage."""
        buffers = [self._input_ids_buf, self._query_start_loc_buf]
        if self._ngram_context_buf is not None:
            buffers.append(self._ngram_context_buf)
        for buffer in buffers:
            self._register_host_buffer(buffer, device_map=False)

    def _register_host_buffer(self, buffer: torch.Tensor, device_map: bool) -> None:
        """Page-lock (and optionally device-map) a shared CPU allocation."""
        if buffer.device.type != "cpu" or not buffer.is_shared():
            raise RuntimeError("PLE host buffers must be shared CPU tensors")
        if not buffer.is_contiguous():
            raise RuntimeError("PLE host buffers must be contiguous")
        flags = cuda_driver.CU_MEMHOSTREGISTER_PORTABLE
        if device_map:
            flags |= cuda_driver.CU_MEMHOSTREGISTER_DEVICEMAP
        _cuda_check(
            cuda_driver.cuMemHostRegister(
                buffer.data_ptr(),
                buffer.numel() * buffer.element_size(),
                flags,
            ),
            "cuMemHostRegister(PLE host buffer)",
        )
        _PROCESS_LIFETIME_REGISTERED_STORAGE.append(buffer)
        if device_map:
            self._mapped_host_buffers.append(buffer)
        else:
            self._pinned_input_buffers.append(buffer)
        if not buffer.is_pinned():
            raise RuntimeError("CUDA did not page-lock a PLE host buffer")

    def _serialize_registration(self, vllm_config: VllmConfig) -> Any:
        """Build and serialise the registration (may move shared storages)."""
        # Each GPU worker owns distinct host output buffers and flags, while
        # TP0's shared inputs become the request source for its DP rank.
        registration = PleOffloadRegistration(
            worker_id=(
                self.dp_rank * vllm_config.parallel_config.world_size
                + vllm_config.parallel_config.rank
            ),
            tp_rank=self.tp_rank,
            dp_rank=self.dp_rank,
            host_output_buffers=dict(self._host_output_bufs),
            host_flag_tensors=dict(self._host_flags),
            input_ids_buf=self._input_ids_buf,
            query_start_loc_buf=self._query_start_loc_buf,
            ngram_context_buf=self._ngram_context_buf,
        )

        # ForkingPickler transmits tensors through shared-memory and CUDA IPC.
        import torch.multiprocessing as torch_mp

        original_strategy = torch_mp.get_sharing_strategy()
        torch_mp.set_sharing_strategy("file_system")
        try:
            payload = ForkingPickler.dumps(registration)
        finally:
            torch_mp.set_sharing_strategy(original_strategy)
        return payload

    def _send_registration(self, payload: Any, ipc_addr: str) -> None:
        """Publish the serialised registration (storages already mapped)."""
        assert self._registration_socket is not None
        self._registration_socket.send(payload)

        logger.info(
            "PleOffload: registered %d PleOffloadLayer(s) "
            "(dp_rank=%d, tp_rank=%d, ipc_addr=%s): %s",
            len(self._layers),
            self.dp_rank,
            self.tp_rank,
            ipc_addr,
            sorted(self._layers),
        )

    def _start_request_thread(self, ipc_addr: str) -> None:
        """Start the thread that publishes batches after inputs are ready."""
        self._request_thread = threading.Thread(
            target=self._request_loop,
            args=(ipc_addr,),
            name=f"ple-offload-dp{self.dp_rank}",
            daemon=True,
        )
        self._request_thread.start()
        if not self._request_thread_ready.wait(timeout=10):
            raise RuntimeError("Timed out starting the PLE request thread")

    def _request_loop(self, ipc_addr: str) -> None:
        """Wait for staged inputs, then notify the CPU worker."""
        socket: zmq.Socket | None = None
        try:
            if self._zmq_ctx is None:
                raise RuntimeError("PLE ZMQ context closed before thread startup")
            socket = self._zmq_ctx.socket(zmq.PUSH)
            socket.connect(ipc_addr)
            self._request_thread_ready.set()
            while True:
                request = self._request_queue.get()
                if request is None:
                    return
                self._process_request(request, socket)
        except Exception:
            logger.exception("PLE request thread failed")
            notify_parent_fatal()
            os._exit(1)
        finally:
            self._request_thread_ready.set()
            if socket is not None:
                socket.close(linger=0)

    def _process_request(
        self, pending: _PendingPleOffloadRequest, socket: zmq.Socket
    ) -> None:
        """Wait for one staged batch and publish its request."""
        request = pending.request
        event_pool = self._d2h_event_pool
        event = pending.d2h_done_event
        assert event_pool is not None, "PLE D2H event pool is not initialized"
        if _FAULT is not None and _FAULT.fires(
            "block_input_completion",
            role="gpu",
            unit="connector-request",
            eligible=pending.eligible,
            tp_rank=self.tp_rank,
            pp_rank=self.pp_rank,
            dp_rank=self.dp_rank,
            transaction_id=pending.transaction_id,
            correlation_id=pending.correlation_id,
        ):
            threading.Event().wait()

        with torch.accelerator.device_index(self.device.index):
            event.synchronize()
        event_pool.put_nowait(event)

        payload = msgspec.msgpack.encode(request)
        if _FAULT is None:
            socket.send(payload)
        else:
            # Armed runs only: correlation metadata rides as a second frame.
            metadata = msgspec.msgpack.encode(
                {
                    "serving": pending.eligible,
                    "transaction": pending.transaction_id,
                    "correlation": pending.correlation_id,
                }
            )
            socket.send_multipart([payload, metadata])

    def _validate_input_sources(self) -> None:
        """Validate fixed runner sources against shared input buffers."""
        sources = [
            ("input_ids", self._input_ids_source, self._input_ids_buf),
            (
                "query_start_loc",
                self._query_start_loc_source,
                self._query_start_loc_buf,
            ),
        ]
        if (self._ngram_context_source is None) != (self._ngram_context_buf is None):
            raise ValueError("PLE ngram_context source and buffer must match")
        if self._ngram_context_source is not None:
            assert self._ngram_context_buf is not None
            sources.append(
                (
                    "ngram_context",
                    self._ngram_context_source,
                    self._ngram_context_buf,
                )
            )

        for name, source, buffer in sources:
            if (
                source.device != self.device
                or source.dtype != buffer.dtype
                or source.ndim != buffer.ndim
                or source.shape[0] < buffer.shape[0]
                or source.shape[1:] != buffer.shape[1:]
            ):
                raise ValueError(f"PLE {name} source is incompatible")

    def _enqueue_cuda_inputs(
        self,
        request: PleOffloadRequest,
        d2h_done_event: torch.cuda.Event,
    ) -> None:
        """Stage inputs on the model stream and record D2H completion."""
        with (
            record_function_or_nullcontext("ple_offload: enqueue_cuda_inputs"),
            torch.accelerator.device_index(self.device.index),
        ):
            stream = torch.cuda.current_stream(self.device)
            self._input_ids_buf[: request.num_tokens].copy_(
                self._input_ids_source[: request.num_tokens],
                non_blocking=True,
            )
            self._query_start_loc_buf[: request.num_reqs + 1].copy_(
                self._query_start_loc_source[: request.num_reqs + 1],
                non_blocking=True,
            )
            if self._ngram_context_buf is not None:
                assert self._ngram_context_source is not None
                self._ngram_context_buf[: request.num_reqs].copy_(
                    self._ngram_context_source[: request.num_reqs],
                    non_blocking=True,
                )
            d2h_done_event.record(stream)

    def _launch(
        self,
        num_reqs: int,
        num_tokens: int,
        *,
        serving: bool = False,
        transaction_id: str | None = None,
        correlation_id: str | None = None,
    ) -> None:
        """Stage or queue one batch for request publication."""
        # Inputs are replicated across TP ranks. One request per DP rank drives
        # the CPU result fan-out to every registered TP output buffer.
        if self.tp_rank != 0:
            return

        request = PleOffloadRequest(
            dp_rank=self.dp_rank,
            num_tokens=num_tokens,
            num_reqs=num_reqs,
        )
        assert self._d2h_event_pool is not None, "PLE D2H event pool is not initialized"
        try:
            d2h_done_event = self._d2h_event_pool.get_nowait()
        except queue.Empty as exc:
            raise RuntimeError(
                "PLE has more requests than configured concurrent batches"
            ) from exc
        self._enqueue_cuda_inputs(request, d2h_done_event)
        if _FAULT is not None and _FAULT.fires(
            "drop_request",
            role="gpu",
            unit="connector-request",
            eligible=serving,
            tp_rank=self.tp_rank,
            pp_rank=self.pp_rank,
            dp_rank=self.dp_rank,
            transaction_id=transaction_id,
            correlation_id=correlation_id,
        ):
            # Deliberately retain the event: the fault is loss after D2H, not
            # premature event-pool reuse while that D2H may still be in flight.
            self._dropped_events.append(d2h_done_event)
            return
        self._request_queue.put_nowait(
            _PendingPleOffloadRequest(
                request,
                d2h_done_event,
                eligible=serving,
                transaction_id=transaction_id,
                correlation_id=correlation_id,
            )
        )

    def _fail_closed(self, where: str, error: BaseException) -> None:
        ple_fail_closed(
            where=where,
            rank=self.rank,
            layer=",".join(self._layers),
            sequence=self._progress.sequence,
            error=error,
        )

    @property
    def has_pending_completions(self) -> bool:
        return self._progress.has_pending_completions

    def begin_forward(self, *, dummy_run: bool, num_tokens: int) -> int:
        """Begin the bounded transaction before any runner fault boundary."""
        try:
            return self._progress.begin(
                startup=dummy_run,
                num_tokens=num_tokens,
            )
        except Exception as error:
            self._fail_closed("prepare_forward", error)

    def fault_transaction(self, sequence: int) -> str:
        """Read diagnostic state only when the caller has established arming."""
        return self._progress.describe_submission(sequence)

    def completion_is_pending(self, sequence: int) -> bool:
        """Return whether the completion for this exact sequence is still pending."""
        return self._progress.completion_is_pending(sequence)

    def prepare_forward(
        self,
        num_reqs: int,
        num_tokens: int,
        dummy_run: bool,
        *,
        consumer_num_tokens: int | None = None,
        serving: bool = False,
        transaction_id: str | None = None,
        correlation_id: str | None = None,
    ) -> None:
        """Prepare one PLE transaction; any ordinary exception is fatal."""
        try:
            self._prepare_forward_impl(
                num_reqs,
                num_tokens,
                dummy_run,
                consumer_num_tokens=consumer_num_tokens,
                serving=serving,
                transaction_id=transaction_id,
                correlation_id=correlation_id,
            )
        except Exception as error:
            self._fail_closed("prepare_forward", error)

    def _prepare_forward_impl(
        self,
        num_reqs: int,
        num_tokens: int,
        dummy_run: bool,
        *,
        consumer_num_tokens: int | None = None,
        serving: bool = False,
        transaction_id: str | None = None,
        correlation_id: str | None = None,
    ) -> None:
        """Submit real inputs or satisfy the PLE wait for a dummy forward."""
        if consumer_num_tokens is None:
            consumer_num_tokens = num_tokens
        if int(num_tokens) != int(consumer_num_tokens):
            raise RuntimeError(
                "PLE request/consumer token-count mismatch: "
                f"request={num_tokens}, consumer={consumer_num_tokens}"
            )
        if not 0 <= int(num_tokens) <= self._input_ids_buf.shape[0]:
            raise RuntimeError(
                f"PLE token count {num_tokens} exceeds connector capacity "
                f"{self._input_ids_buf.shape[0]}"
            )
        if not 0 <= int(num_reqs) < self._query_start_loc_buf.shape[0]:
            raise RuntimeError(
                f"PLE request count {num_reqs} exceeds connector capacity "
                f"{self._query_start_loc_buf.shape[0] - 1}"
            )
        if dummy_run:
            self._signal_dummy_outputs_impl(num_tokens)
            return
        if self._after_dummy:
            get_tp_group().barrier()
            self._after_dummy = False
            logger.info(
                "PleOffload: dummy->real transition (dp_rank=%d, tp_rank=%d): "
                "releases were synchronous, TP barrier passed",
                self.dp_rank,
                self.tp_rank,
            )
        self._launch(
            num_reqs,
            num_tokens,
            serving=serving,
            transaction_id=transaction_id,
            correlation_id=correlation_id,
        )

    def signal_dummy_outputs(self, num_tokens: int) -> None:
        """Locally satisfy PLE waits for dummy and capture forwards."""
        self._progress.begin(startup=True, num_tokens=num_tokens)
        self._signal_dummy_outputs_impl(num_tokens)

    def _signal_dummy_outputs_impl(self, num_tokens: int) -> None:
        """Signal dummy outputs after the progress transaction is armed."""
        # Dummy and capture forwards do not send CPU requests, but every PLE
        # placeholder still waits for a completed output semaphore.
        self._after_dummy = True
        stream = torch.cuda.current_stream(self.device)
        for layer in self._layers.values():
            layer._gpu_output_buffer[:num_tokens].zero_()
            layer._sem.signal(stream)

    def release_outputs(
        self,
        *,
        serving: bool = False,
        transaction_id: str | None = None,
        correlation_id: str | None = None,
    ) -> None:
        """Release one PLE transaction; any ordinary exception is fatal."""
        try:
            self._release_outputs_impl(
                serving=serving,
                transaction_id=transaction_id,
                correlation_id=correlation_id,
            )
            self._progress.complete()
        except Exception as error:
            self._fail_closed("release_outputs", error)

    def _release_outputs_impl(
        self,
        *,
        serving: bool = False,
        transaction_id: str | None = None,
        correlation_id: str | None = None,
    ) -> None:
        """Mark GPU output buffers reusable after the model consumes them."""
        if _FAULT is not None and _FAULT.fires(
            "exc_in_release",
            role="gpu",
            unit="connector-release",
            eligible=serving,
            tp_rank=self.tp_rank,
            pp_rank=self.pp_rank,
            dp_rank=self.dp_rank,
            transaction_id=transaction_id,
            correlation_id=correlation_id,
        ):
            raise RuntimeError("PLE test fault: exc_in_release")
        # Reset only after the consumer forward so the CPU worker cannot
        # overwrite an output that a GPU PLE placeholder may still read.
        stream = torch.cuda.current_stream(self.device)
        for layer in self._layers.values():
            layer.release_offloaded_output(stream)
        # Start the GPU completion budget after host submission (including cold
        # compilation). The monitor thread is the sole event query site.
        completion_event = torch.cuda.Event()
        completion_event.record(stream)
        self._progress.track_completion(completion_event)
        if self._after_dummy and not torch.cuda.is_current_stream_capturing():
            stream.synchronize()

    def release_host_storage(self) -> None:
        """Unlink host-storage names while retaining mappings and tensors."""
        buffers = [
            *self._host_output_bufs.values(),
            *self._host_flags.values(),
            self._input_ids_buf,
            self._query_start_loc_buf,
        ]
        if self._ngram_context_buf is not None:
            buffers.append(self._ngram_context_buf)
        for buffer in buffers:
            _unlink_filename_shared_storage(
                buffer,
                self._reserved_host_storage_filenames,
            )

    def close(self) -> None:
        """Stop transport and unlink names, retaining host mappings."""
        # The runner has drained the GPU while the producer is still alive.
        # Disarm before teardown can race a final event query or deadline.
        self._progress.close()
        request_thread = self._request_thread
        if request_thread is not None and request_thread.is_alive():
            try:
                self._request_queue.put(None, timeout=5)
            except queue.Full:
                logger.error("Timed out stopping the PLE request thread")
            request_thread.join(timeout=5)
        if request_thread is not None and request_thread.is_alive():
            logger.error("PLE request thread did not stop; deferring resource cleanup")
            return
        self._request_thread = None

        registered = len(self._pinned_input_buffers) + len(self._mapped_host_buffers)
        if registered:
            logger.debug(
                "PleOffload: %d registered host buffers retained until exit",
                registered,
            )
        self._d2h_event_pool = None
        if self._registration_socket is not None:
            self._registration_socket.close(linger=0)
            self._registration_socket = None
        if self._zmq_ctx is not None:
            self._zmq_ctx.term()
            self._zmq_ctx = None
        self.release_host_storage()
