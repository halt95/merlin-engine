# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""The fork's PLE pull transport, on upstream's split embedding modules.

Upstream's ``Qwen4ExpNGramEmbedding`` (``ngram_embedding.py``) has a device or
pinned-host UVA table and no offload.
This module adds the pull transport WITHOUT editing that upstream
class: ``Qwen4ExpNGramEmbeddingPull`` inherits it and ``PleOffloadLayer``.

* In a GPU worker, ``PleOffloadLayer.__init_subclass__`` skips the constructor:
  the module is an empty placeholder with no table. ``PleOffloadConnector``
  binds it to a host-mapped flag/buffer; its forward is the captured
  ``ple_offload_wait_copy`` op. Only the FP8 global scale is kept on the GPU.
* In the CPU-only offload process the constructor runs in full, on CPU. The
  table is upstream's ``Qwen4ExpPLEDeviceEmbedding`` allocated on CPU;
  ``forward_impl`` computes n-gram IDs and gathers raw (FP8) rows.

Always selected (the fork's only PLE transport;
upstream's device/UVA backends are never built).

Supported table formats: FP8 (global scale, the shipped table) and
unquantized -- the two ``Qwen4ExpPLEEmbeddingMethod`` families upstream itself
supports -- plus the fork's packed tables (``ple_formats.py``): INT6
g32 (the shipped opt-in) and INT8 per row (a reference table), whose rows cross
as U8 ``[tokens, ngram_heads * row_bytes]``
and are expanded on the GPU at ``dequantize_pulled``. NVFP4 tables are refused:
the pull transport does not support ``Qwen4ExpPLENvFp4EmbeddingMethod``.
"""

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import torch
from transformers import Qwen4ExpTextConfig

import vllm.envs as envs
from vllm.config import get_current_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.layers.ple_offload_layer import (
    PleOffloadLayer,
    is_offload_process,
)
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

from .ngram_embedding import (
    Qwen4ExpNGramEmbedding,
    Qwen4ExpPLEDeviceEmbedding,
    Qwen4ExpPLEEmbeddingMethod,
    Qwen4ExpPLEFp8EmbeddingMethod,
    Qwen4ExpPLEUnquantizedEmbeddingMethod,
)
from .ple_formats import (
    Qwen4ExpPLEInt6G32EmbeddingMethod,
    Qwen4ExpPLEPackedRowEmbeddingMethod,
    select_ple_embedding_method,
)

logger = init_logger(__name__)

#: The backend name this module reports in the PLE producer log line.
PULL_BACKEND_NAME = "E2PullTransport"

_FAST_IDS: bool = envs.VLLM_E2_PLE_FAST_IDS
_PLE_STORE: str = envs.VLLM_E2_PLE_STORE
_PLE_WARM: bool = envs.VLLM_E2_PLE_WARM
_PLE_DISK_THREADS: int = envs.VLLM_E2_PLE_DISK_THREADS


class Qwen4ExpPLEFileBackedEmbedding(Qwen4ExpPLEDeviceEmbedding):
    """The table's shape and dtype with no storage: rows come from the files."""

    def allocate_embedding_weight(
        self,
        num_embeddings: int,
        embedding_dim: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return torch.empty(num_embeddings, embedding_dim, dtype=dtype, device="meta")


@dataclass(frozen=True)
class _PreparedNGramIds:
    request_indices: torch.Tensor
    adjusted_columns: torch.Tensor
    shifted: list[torch.Tensor]
    ngram_size: int
    eos_token_id: int


@dataclass(frozen=True)
class _PreparedNGramIdsNp:
    """NumPy twin of ``_PreparedNGramIds`` (int64 arrays, same meaning)."""

    request_indices: np.ndarray
    adjusted_columns: np.ndarray
    shifted: tuple[np.ndarray, ...]
    ngram_size: int
    eos_token_id: int


def _cpu_int64(tensor: torch.Tensor) -> np.ndarray:
    """A CPU tensor as an int64 NumPy array (a copy when the dtype differs)."""
    if tensor.device.type != "cpu":
        raise ValueError("PLE fast n-gram IDs run on CPU tensors only")
    return tensor.detach().numpy().astype(np.int64, copy=False)


class Qwen4ExpNGramEmbeddingPull(PleOffloadLayer, Qwen4ExpNGramEmbedding):
    """Upstream n-gram embedding whose table lives in the offload process."""

    #: GPU placeholder: the method the offload process will use for the table,
    #: resolved with upstream's own selector (set by ``build``).
    _offload_quant_method: Qwen4ExpPLEEmbeddingMethod | None = None

    def __init__(
        self,
        config: Qwen4ExpTextConfig,
        embedding_dim: int,
        ple_dense_layer_id: int,
        max_total_tokens: int,
        *,
        data_parallel_rank: int,
        prefix: str,
        quant_config: QuantizationConfig | None = None,
        params_dtype: torch.dtype | None = None,
    ) -> None:
        # Runs only in the offload process (guarded by PleOffloadLayer).
        engram_config = get_current_vllm_config().engram_config
        if engram_config is not None and engram_config.cpu_offload:
            raise RuntimeError(
                "PLE pull transport: the offload process resolved "
                "cpu_offload=True; refusing to build upstream's UVA table here."
            )
        if _PLE_STORE == "disk":
            # Read by the table hook in Qwen4ExpNGramEmbedding.__init__.
            self.__dict__["_ple_table_cls"] = Qwen4ExpPLEFileBackedEmbedding
        Qwen4ExpNGramEmbedding.__init__(
            self,
            config,
            embedding_dim,
            ple_dense_layer_id,
            max_total_tokens,
            data_parallel_rank=data_parallel_rank,
            prefix=prefix,
            quant_config=quant_config,
            params_dtype=params_dtype,
        )
        if _PLE_STORE == "disk":
            self.__dict__["_ple_file_table"] = self._open_file_table(prefix)
        # Logged once per layer so the ID path is visible in the engine log.
        logger.info(
            "PLE offload n-gram IDs: %s path (VLLM_E2_PLE_FAST_IDS=%d)",
            "numpy" if _FAST_IDS else "torch",
            int(_FAST_IDS),
        )

    def _open_file_table(self, prefix: str) -> object:
        """The table's files, checked now so that a bad one fails at boot."""
        from vllm.model_executor.models.utils import extract_layer_index

        from .ple_file_table import PleFileTable, fp8_parts, int6_parts

        vllm_config = get_current_vllm_config()
        if str(vllm_config.load_config.load_format) == "dummy":
            raise RuntimeError(
                "--ple-store disk reads the table from checkpoint files; a dummy "
                "load has none"
            )
        embedding = self.ngram_embedding
        method = embedding.embedding_method
        if isinstance(method, Qwen4ExpPLEFp8EmbeddingMethod):
            parts = fp8_parts(embedding.weight.shape[1])
        elif isinstance(method, Qwen4ExpPLEInt6G32EmbeddingMethod):
            parts = int6_parts()
        else:
            raise RuntimeError(
                "--ple-store disk serves the FP8 and INT6 g32 tables only, not "
                f"{type(method).__name__}; use --ple-store ram"
            )
        table = PleFileTable(
            vllm_config.model_config.model,
            extract_layer_index(prefix),
            embedding.org_vocab_size,
            self.split_ngram_parts,
            parts,
            threads=_PLE_DISK_THREADS,
            warm=_PLE_WARM,
        )
        logger.info("%s", table.proof_line())
        return table

    @classmethod
    def build(
        cls,
        config: Qwen4ExpTextConfig,
        embedding_dim: int,
        ple_dense_layer_id: int,
        max_total_tokens: int,
        *,
        data_parallel_rank: int,
        prefix: str,
        quant_config: QuantizationConfig | None,
        params_dtype: torch.dtype | None,
        use_sequence_parallel: bool = False,
    ) -> "Qwen4ExpNGramEmbeddingPull":
        """Construct the placeholder (GPU) or the full CPU module (offload).

        ``use_sequence_parallel`` (GR/PLE SP, vllm#56322) selects
        ``Qwen4ExpNGramEmbeddingPullSP``, whose GPU forward returns this TP
        rank's token shard of the pulled rows. The offload process never
        shards: it publishes full-token rows either way.
        """
        module_cls = Qwen4ExpNGramEmbeddingPullSP if use_sequence_parallel else cls
        with torch.device(module_cls.get_target_device()):
            module = module_cls(
                config,
                embedding_dim,
                ple_dense_layer_id,
                max_total_tokens,
                data_parallel_rank=data_parallel_rank,
                prefix=prefix,
                quant_config=quant_config,
                params_dtype=params_dtype,
            )
        embedding_prefix = f"{prefix}.ngram_embedding"
        # The same selector the offload process's table runs (ngram_embedding.py).
        method = select_ple_embedding_method(
            quant_config,
            embedding_prefix,
            getattr(config, "ple_embedding_dtype", None),
        )
        if not isinstance(
            method,
            (
                Qwen4ExpPLEFp8EmbeddingMethod,
                Qwen4ExpPLEUnquantizedEmbeddingMethod,
                Qwen4ExpPLEPackedRowEmbeddingMethod,
            ),
        ):
            raise NotImplementedError(
                f"PLE pull transport does not support {type(method).__name__}"
            )
        if isinstance(method, Qwen4ExpPLEPackedRowEmbeddingMethod):
            # The GPU placeholder owns no table, so the pulled row width comes
            # from the config: ngram_heads x row_bytes per token.
            heads = (int(config.ngram_size) - 1) * int(config.heads_per_ngram)
            if heads <= 0 or embedding_dim != heads * method.HEAD_DIM:
                raise ValueError(
                    f"{method.DTYPE_NAME} needs ple_embed_dim == ngram_heads * "
                    f"{method.HEAD_DIM}, got {embedding_dim} with {heads} heads"
                )
            module._packed_out_dim = heads * method.ROW_BYTES
        module._offload_quant_method = method
        module._pull_params_dtype = params_dtype or torch.get_default_dtype()
        module._pull_embedding_prefix = embedding_prefix
        if not is_offload_process():
            logger.info(
                "Initialized PLE embedding %s: quantization_method=%s, "
                "weight_dtype=%s, weight_device=%s, pinned=%s, backend=%s",
                embedding_prefix,
                type(method).__name__,
                module.get_offload_output_dtype(module._pull_params_dtype),
                "cpu",
                False,
                PULL_BACKEND_NAME,
            )
        return module

    # ------------------------------------------------------------- formats

    def get_offload_output_dtype(self, default_dtype: torch.dtype) -> torch.dtype:
        """Rows cross the transport in their table storage dtype."""
        if isinstance(self._offload_quant_method, Qwen4ExpPLEPackedRowEmbeddingMethod):
            return torch.uint8
        embedding = getattr(self, "ngram_embedding", None)
        weight = getattr(embedding, "weight", None)
        if weight is not None:
            return weight.dtype
        if isinstance(self._offload_quant_method, Qwen4ExpPLEFp8EmbeddingMethod):
            return torch.float8_e4m3fn
        return getattr(self, "_pull_params_dtype", default_dtype)

    def get_offload_output_dim(self, default_dim: int) -> int:
        if isinstance(self._offload_quant_method, Qwen4ExpPLEPackedRowEmbeddingMethod):
            # ngram_heads * row_bytes per token, not ple_embed_dim values
            return self._packed_out_dim
        return default_dim

    def initialize_dummy_offload_metadata(self, device: torch.device) -> None:
        """Dummy loads skip checkpoint scales; give FP8 a unit scale."""
        if isinstance(self._offload_quant_method, Qwen4ExpPLEFp8EmbeddingMethod):
            self.register_buffer(
                "_offload_weight_scale",
                torch.ones((), dtype=torch.float32, device=device),
                persistent=False,
            )

    def dequantize_pulled(
        self, embeddings: torch.Tensor, output_dtype: torch.dtype
    ) -> torch.Tensor:
        """GPU-side dequantization of rows DMA'd from the offload process.

        Same arithmetic as upstream ``Qwen4ExpPLEFp8EmbeddingMethod.dequantize``.
        """
        method = self._offload_quant_method
        if isinstance(method, Qwen4ExpPLEUnquantizedEmbeddingMethod):
            return embeddings
        if isinstance(method, Qwen4ExpPLEPackedRowEmbeddingMethod):
            # U8 [n, ngram_heads * row_bytes] -> output_dtype [n, ple_embed_dim]
            return method.dequantize_rows(embeddings, output_dtype)
        weight_scale = getattr(self, "_offload_weight_scale", None)
        if weight_scale is None:
            raise RuntimeError(
                "FP8 PLE pull transport is missing its global scale on the GPU"
            )
        if weight_scale.device != embeddings.device:
            raise RuntimeError("FP8 PLE embedding scale must be on the output device")
        return embeddings.to(output_dtype) * weight_scale.to(output_dtype)

    # ------------------------------------------------------------- execution

    def start_prefetch(
        self,
        input_ids: torch.Tensor,
        query_start_loc: torch.Tensor,
        ngram_context: torch.Tensor,
    ) -> None:
        """No prefetch: rows are published by the offload process."""
        return None

    def _prepare_ngram_ids_offload(
        self,
        input_ids: torch.Tensor,
        query_start_loc: torch.Tensor,
        ngram_context: torch.Tensor,
    ) -> "_PreparedNGramIds | _PreparedNGramIdsNp":
        """Pack request tokens and compute the shifts shared by every PLE layer."""
        if _FAST_IDS:
            return self._prepare_ngram_ids_numpy(
                input_ids, query_start_loc, ngram_context
            )
        return self._prepare_ngram_ids_torch(
            input_ids, query_start_loc, ngram_context
        )

    def _prepare_ngram_ids_torch(
        self,
        input_ids: torch.Tensor,
        query_start_loc: torch.Tensor,
        ngram_context: torch.Tensor,
    ) -> _PreparedNGramIds:
        """The torch path (default): one CPU op per step of the packing."""
        input_ids = input_ids.reshape(-1).long()
        query_start_loc = query_start_loc.long()
        num_reqs = query_start_loc.numel() - 1
        num_tokens = input_ids.shape[0]
        if num_reqs <= 0:
            raise ValueError("PLE requires at least one request")
        max_seq_len = max(
            1,
            int((query_start_loc[1:] - query_start_loc[:-1]).max().item()),
        )
        num_valid_tokens = min(int(query_start_loc[-1].item()), num_tokens)

        positions = torch.arange(num_tokens, device=input_ids.device, dtype=torch.int64)
        packed = torch.full(
            (num_reqs, max_seq_len),
            self.eos_token_id,
            device=input_ids.device,
            dtype=torch.int64,
        )
        request_indices = torch.searchsorted(query_start_loc, positions, right=True) - 1
        request_indices.clamp_(max=num_reqs - 1)
        columns = (positions - query_start_loc[request_indices]).clamp(
            0, packed.shape[1] - 1
        )
        packed[request_indices[:num_valid_tokens], columns[:num_valid_tokens]] = (
            input_ids[:num_valid_tokens]
        )
        ngram_context = ngram_context[:num_reqs].to(
            device=input_ids.device, dtype=torch.long
        )

        context = torch.cat([ngram_context, packed], dim=-1)
        positions_2d, position_in_segment = self._shift_precompute(
            context, self.eos_token_id
        )
        shifted = [context]
        for shift in range(1, self.ngram_size):
            shifted.append(
                self._shift_apply(
                    context,
                    positions_2d,
                    position_in_segment,
                    shift,
                    self.eos_token_id,
                )
            )
        adjusted_columns = columns + self.ngram_size - 1
        return _PreparedNGramIds(
            request_indices,
            adjusted_columns,
            shifted,
            self.ngram_size,
            self.eos_token_id,
        )

    def _prepare_ngram_ids_numpy(
        self,
        input_ids: torch.Tensor,
        query_start_loc: torch.Tensor,
        ngram_context: torch.Tensor,
    ) -> _PreparedNGramIdsNp:
        """``_prepare_ngram_ids_torch`` step for step, in NumPy int64."""
        input_ids_np = _cpu_int64(input_ids.reshape(-1))
        query_start_loc_np = _cpu_int64(query_start_loc)
        num_reqs = query_start_loc_np.shape[0] - 1
        num_tokens = input_ids_np.shape[0]
        if num_reqs <= 0:
            raise ValueError("PLE requires at least one request")
        max_seq_len = max(
            1, int((query_start_loc_np[1:] - query_start_loc_np[:-1]).max())
        )
        num_valid_tokens = min(int(query_start_loc_np[-1]), num_tokens)
        eos = self.eos_token_id

        positions = np.arange(num_tokens, dtype=np.int64)
        packed = np.full((num_reqs, max_seq_len), eos, dtype=np.int64)
        request_indices = (
            np.searchsorted(query_start_loc_np, positions, side="right") - 1
        )
        np.minimum(request_indices, num_reqs - 1, out=request_indices)
        columns = np.clip(
            positions - query_start_loc_np[request_indices], 0, max_seq_len - 1
        )
        packed[request_indices[:num_valid_tokens], columns[:num_valid_tokens]] = (
            input_ids_np[:num_valid_tokens]
        )
        context = np.concatenate(
            [_cpu_int64(ngram_context[:num_reqs]), packed], axis=-1
        )

        # _shift_precompute
        seq_len = context.shape[1]
        positions_2d = np.arange(seq_len, dtype=np.int64)
        eos_positions = np.where(context == eos, positions_2d, -1)
        previous_eos_inclusive = np.maximum.accumulate(eos_positions, axis=1)
        previous_eos = np.concatenate(
            [
                np.full((num_reqs, 1), -1, dtype=np.int64),
                previous_eos_inclusive[:, :-1],
            ],
            axis=1,
        )
        position_in_segment = positions_2d[None, :] - previous_eos - 1

        # _shift_apply for every shift
        shifted = [context]
        for shift in range(1, self.ngram_size):
            source = positions_2d - shift
            gathered = context[:, np.maximum(source, 0)]
            valid = (source >= 0)[None, :] & (position_in_segment >= shift)
            shifted.append(np.where(valid, gathered, np.int64(eos)))
        adjusted_columns = columns + self.ngram_size - 1
        return _PreparedNGramIdsNp(
            request_indices,
            adjusted_columns,
            tuple(shifted),
            self.ngram_size,
            eos,
        )

    def _numpy_layer_tables(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """This layer's multipliers, head sizes and offsets as int64 arrays."""
        tables = self.__dict__.get("_np_layer_tables")
        if tables is None:
            tables = (
                _cpu_int64(self.layer_multipliers).copy(),
                _cpu_int64(self.ngram_heads_vocab_sizes).copy(),
                _cpu_int64(self.ngram_heads_offsets).copy(),
            )
            self.__dict__["_np_layer_tables"] = tables
        return tables

    def _compute_ngram_ids_from_common_numpy(
        self, common: _PreparedNGramIdsNp
    ) -> torch.Tensor:
        """``_compute_ngram_ids_from_common`` in NumPy; returns int64 torch."""
        multipliers, sizes, offsets = self._numpy_layer_tables()
        id_blocks = []
        for ngram in range(2, self.ngram_size + 1):
            start = (ngram - 2) * self.heads_per_ngram
            end = start + self.heads_per_ngram
            mixed = common.shifted[0] * multipliers[0]
            for index in range(1, ngram):
                mixed = np.bitwise_xor(
                    mixed, common.shifted[index] * multipliers[index]
                )
            ids = (
                np.remainder(mixed[..., None], sizes[start:end]) + offsets[start:end]
            )
            id_blocks.append(ids[common.request_indices, common.adjusted_columns])
        return torch.from_numpy(
            np.ascontiguousarray(np.concatenate(id_blocks, axis=-1))
        )

    def _compute_ngram_ids_from_common(
        self, common: "_PreparedNGramIds | _PreparedNGramIdsNp"
    ) -> torch.Tensor:
        """Apply this PLE layer's multipliers and vocabulary ranges."""
        if (
            common.ngram_size != self.ngram_size
            or common.eos_token_id != self.eos_token_id
        ):
            raise ValueError("PLE layers disagree on common n-gram configuration")
        if isinstance(common, _PreparedNGramIdsNp):
            return self._compute_ngram_ids_from_common_numpy(common)
        id_blocks = []
        for ngram in range(2, self.ngram_size + 1):
            start = (ngram - 2) * self.heads_per_ngram
            end = start + self.heads_per_ngram
            mixed = common.shifted[0] * self.layer_multipliers[0]
            for index in range(1, ngram):
                mixed = torch.bitwise_xor(
                    mixed, common.shifted[index] * self.layer_multipliers[index]
                )
            sizes = self.ngram_heads_vocab_sizes[start:end]
            offsets = self.ngram_heads_offsets[start:end]
            ids = torch.remainder(mixed.unsqueeze(-1), sizes) + offsets
            id_blocks.append(
                ids[common.request_indices, common.adjusted_columns]
            )
        return torch.cat(id_blocks, dim=-1)

    def _compute_ngram_ids_offload(
        self,
        input_ids: torch.Tensor,
        query_start_loc: torch.Tensor,
        ngram_context: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-layer IDs from the common request packing and shifts."""
        common = self._prepare_ngram_ids_offload(
            input_ids, query_start_loc, ngram_context
        )
        return self._compute_ngram_ids_from_common(common)

    def forward_impl(  # type: ignore[override]
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        query_start_loc: torch.Tensor,
        ngram_context: torch.Tensor,
        output_buffer: torch.Tensor | None = None,
        prepared_ngram_ids: "_PreparedNGramIds | _PreparedNGramIdsNp | None" = None,
    ) -> torch.Tensor:
        """Offload process only: gather raw table rows for these tokens."""
        del hidden_states
        if not is_offload_process():
            raise RuntimeError(
                "PLE pull transport: forward_impl ran outside the offload "
                "process -- the GPU placeholder was never bound by the connector"
            )
        if prepared_ngram_ids is None:
            ngram_ids = self._compute_ngram_ids_offload(
                input_ids, query_start_loc, ngram_context
            )
        else:
            ngram_ids = self._compute_ngram_ids_from_common(prepared_ngram_ids)
        num_tokens = input_ids.reshape(-1).shape[0]
        weight = self.ngram_embedding.weight
        file_table = self.__dict__.get("_ple_file_table")
        if file_table is not None:
            # The same rows in the same layout as the index_select paths below.
            width = weight.shape[1] * self.ngram_heads
            output = (
                output_buffer[:num_tokens, :width]
                if output_buffer is not None
                else torch.empty(num_tokens, width, dtype=weight.dtype, device="cpu")
            )
            rows = output.reshape(-1, weight.shape[1]).view(torch.uint8)
            file_table.gather(ngram_ids.reshape(-1), rows)
            return output
        if isinstance(self._offload_quant_method, Qwen4ExpPLEPackedRowEmbeddingMethod):
            # Packed rows: U8 [tokens, ngram_heads * row_bytes], gathered as bytes.
            row_bytes = self._offload_quant_method.ROW_BYTES
            width = self.ngram_heads * row_bytes
            if output_buffer is not None:
                output = output_buffer[:num_tokens, :width]
                torch.index_select(
                    weight,
                    0,
                    ngram_ids.reshape(-1),
                    out=output.reshape(-1, row_bytes),
                )
                return output
            return torch.index_select(weight, 0, ngram_ids.reshape(-1)).reshape(
                num_tokens, width
            )
        if output_buffer is not None:
            output = output_buffer[:num_tokens, : self.embedding_dim]
            torch.index_select(
                weight,
                0,
                ngram_ids.reshape(-1),
                out=output.reshape(-1, self.head_dim),
            )
            return output
        return torch.index_select(weight, 0, ngram_ids.reshape(-1)).reshape(
            num_tokens, self.embedding_dim
        )

    # ------------------------------------------------------------- loading

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Offload process: upstream loading. GPU worker: keep the FP8 scale.

        The GPU placeholder owns no table; every other checkpoint tensor for
        this subtree is consumed and dropped here (the offload process loads
        the table itself). Prefix grouping may call this more than once.
        """
        if is_offload_process():
            if "_ple_file_table" in self.__dict__:
                # The rows stay in the files, opened and checked at construction;
                # the rest (FP8's scale) loads as ever. Reported loaded for the
                # child's every-parameter-loaded check.
                loaded = Qwen4ExpNGramEmbedding.load_weights(
                    self,
                    (
                        (name, tensor)
                        for name, tensor in weights
                        if not name.startswith("ngram_embedding.shard_")
                    ),
                )
                loaded.add("ngram_embedding.weight")
                return loaded
            method = self.ngram_embedding.embedding_method
            if isinstance(method, Qwen4ExpPLEPackedRowEmbeddingMethod):
                weights = self._pair_packed_shards(method, weights)
            return Qwen4ExpNGramEmbedding.load_weights(self, weights)
        retained: set[str] = set()
        fp8 = isinstance(self._offload_quant_method, Qwen4ExpPLEFp8EmbeddingMethod)
        for name, loaded_weight in weights:
            if fp8 and name == "ngram_embedding.weight_scale":
                # The offload process validates its own copy; this one feeds
                # every pulled row, so a NaN, inf, zero or negative checkpoint
                # scale must refuse to load here too.
                usable = torch.isfinite(loaded_weight) & (loaded_weight > 0)
                if not bool(usable.all()):
                    raise ValueError("FP8 PLE checkpoint is missing its global scale")
                self.register_buffer(
                    "_offload_weight_scale",
                    loaded_weight.reshape(()).to(
                        device=torch.accelerator.current_accelerator(),
                        dtype=torch.float32,
                    ),
                    persistent=False,
                )
                retained.add(name)
        return retained

    def _pair_packed_shards(
        self,
        method: Qwen4ExpPLEPackedRowEmbeddingMethod,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> Iterable[tuple[str, torch.Tensor]]:
        """Interleave each shard's ``.codes`` / ``.scales`` into ``.weight`` rows.

        The paired U8 ``[rows, row_bytes]`` shard then takes upstream's shard path
        (format guard, shape check against the table's own row width, row-range
        copy). The pairer lives on the table module, so halves split across
        ``load_weights`` calls still pair, and ``process_weights_after_loading``
        refuses a shard left with one half.
        """
        shard_prefix = "ngram_embedding.shard_"
        # Built only when absent (setdefault would build one on every call).
        pairer = self.ngram_embedding.__dict__.get("_packed_pending")
        if pairer is None:
            pairer = method.make_pairer()
            self.ngram_embedding.__dict__["_packed_pending"] = pairer
        for name, tensor in weights:
            part = name.rsplit(".", 1)[-1]
            if name.startswith(shard_prefix) and part in ("codes", "scales"):
                shard_key = name[len(shard_prefix) : -len(part) - 1]
                rows = pairer.add(shard_key, part, tensor)
                if rows is not None:
                    yield f"{shard_prefix}{shard_key}.weight", rows
                continue
            yield name, tensor


class Qwen4ExpNGramEmbeddingPullSP(Qwen4ExpNGramEmbeddingPull):
    """Pull transport under GR/PLE sequence parallelism (vllm#56322).

    Upstream's SP contract (``Qwen4ExpNGramEmbedding.forward`` with
    ``use_sequence_parallel``): every TP rank receives the full-token
    ``input_ids`` but only its token shard of ``hidden_states`` --
    ``sp_shard`` order, rank ``r`` owns padded rows ``[r*c, (r+1)*c)`` with
    ``c = ceil(N / tp)`` -- and the embedding returns that same shard in the
    table storage dtype. Upstream combines the vocabulary owners with a
    reduce-scatter (ETP == TP) or an ETP all-reduce followed by ``sp_shard``.

    The offload process holds the whole table (TP1), so the rows it publishes
    are already complete: the full-token producer is unchanged and this
    consumer only selects its shard, through upstream's own
    ``_sp_shard_embeddings`` on its non-reducing branch (FP8 rows move as
    int8, as upstream does). A lookup has one vocabulary owner per element,
    so this is value-identical to upstream's reduce-scatter.

    Transport invariants are untouched: the wait/DMA is the base
    ``PleOffloadLayer.forward`` (same op, same row count, same pinned slot);
    the shard is a slice -- plus zero padding when ``N % tp != 0`` -- of the
    device buffer, with no host sync. No ``__init__`` here, so the
    ``PleOffloadLayer`` constructor guard is inherited unchanged.
    """

    #: The pulled rows are complete; summing them across ranks would multiply
    #: them by TP. Pins upstream's ``_sp_shard_embeddings`` to ``sp_shard``.
    use_reduce_scatter = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        *args: object,
        **kwargs: object,
    ) -> torch.Tensor:
        rows = super().forward(hidden_states, input_ids, *args, **kwargs)
        return self._sp_shard_embeddings(rows)


__all__ = [
    "PULL_BACKEND_NAME",
    "Qwen4ExpNGramEmbeddingPull",
    "Qwen4ExpNGramEmbeddingPullSP",
]
