# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modified by the Merlin project; see NOTICE.
"""Adapted from #68296 nvidia/engram.py:30-206.

The torch hash follows the pinned DeepSeek-V4.1 hash kernel's history and
signed-int64 behavior. No token-history cache survives a forward.
"""

import numpy as np
import torch
from torch import nn


def head_range(sizes, tp: int, rank: int) -> tuple[int, int]:
    if not 1 <= tp <= len(sizes) or not 0 <= rank < tp:
        raise ValueError("Engram requires one complete hash head per TP rank")
    prefix = np.cumsum([0, *sizes]).tolist()
    boundaries = [0]
    for r in range(1, tp):
        target = prefix[-1] * r / tp
        candidates = range(boundaries[-1] + 1, len(sizes) - (tp - r) + 1)
        boundaries.append(min(candidates, key=lambda i: abs(prefix[i] - target)))
    boundaries.append(len(sizes))
    return boundaries[rank], boundaries[rank + 1]


class FlashEngramLayout:
    def __init__(self, config):
        self.layer_ids = tuple(config.engram_layer_ids or ())
        self.max_ngram_size = config.engram_max_ngram_size
        self.n_heads = config.engram_n_head_per_ngram
        self.head_dim = config.engram_n_embed_per_ngram // self.n_heads
        self.pad_token_id = config.engram_pad_id
        self.vocab_size = config.vocab_size
        self.seed = config.engram_seed
        groups = config.engram_layer_head_vocab_sizes or {}
        self.primes = tuple(
            groups.get(str(layer), groups.get(layer)) for layer in self.layer_ids
        )
        flat = [[size for order in group for size in order] for group in self.primes]
        self.offsets = np.array(
            [np.cumsum([0, *sizes[:-1]]) for sizes in flat], dtype=np.int64
        )
        self.lookup_table = None

    def load_lookup(self, value: torch.Tensor) -> bool:
        if (
            value.dtype not in (torch.int32, torch.int64)
            or value.ndim != 1
            or not 0 < value.numel() <= self.vocab_size
        ):
            raise ValueError("Invalid frozen Engram token map")
        lookup = value.detach().cpu().numpy()
        if np.any(lookup < 0) or np.any(lookup > np.iinfo(np.int32).max):
            raise ValueError("Invalid frozen Engram token map")
        if self.lookup_table is not None:
            if not np.array_equal(self.lookup_table, lookup):
                raise ValueError("Engram layers must share the same token map")
            return False
        self.lookup_table = lookup.copy()
        self.compressed_vocab_size = int(lookup.max()) + 1
        self.token_map = lookup.tolist()
        self.token_map.extend(range(len(self.token_map), self.vocab_size))
        bound = max(1, (np.iinfo(np.int64).max // self.compressed_vocab_size) // 2)
        self.multipliers = np.stack(
            [
                np.random.default_rng(self.seed + 10007 * layer).integers(
                    0, bound, size=self.max_ngram_size, dtype=np.int64
                )
                * 2
                + 1
                for layer in self.layer_ids
            ]
        )
        return True


def batch_lookback_rows(lookback: torch.Tensor, query_start_loc: torch.Tensor):
    """The lookback rows of the batch ``query_start_loc`` describes: the leading
    ``query_start_loc.numel() - 1`` rows of the model state's persistent buffer
    (row r is batch request r; rows past the batch hold -1). Slicing reads the
    live request count at call time, so an eager section replayed by a breakable
    CUDA graph with its capture-time buffer argument still matches the replayed
    batch. A batch with more requests than the buffer has rows keeps the short
    slice, which the hash's layout check refuses."""
    return lookback[: max(query_start_loc.shape[0] - 1, 0)]


class FlashNgramHashState(nn.Module):
    def __init__(self, config, *, dummy: bool = False, rows=None):
        """``rows``: the checkpoint-global hash rows (indices into
        ``engram_layer_ids``) this PP stage hashes; None hashes every row.
        A row keeps its global multipliers/primes/offsets.
        """
        super().__init__()
        self.layout = FlashEngramLayout(config)
        self.lookback_depth = self.layout.max_ngram_size - 1
        count = len(self.layout.layer_ids)
        rows = list(range(count)) if rows is None else list(rows)
        if (
            not rows
            or len(set(rows)) != len(rows)
            or any(type(r) is not int or not 0 <= r < count for r in rows)
        ):
            raise ValueError("Engram hash rows must be distinct global row indices")
        self.rows = tuple(rows)
        self.register_buffer(
            "row_index", torch.tensor(rows, dtype=torch.int64), persistent=False
        )
        self.register_buffer(
            "token_map",
            torch.zeros(config.vocab_size, dtype=torch.int64),
            persistent=False,
        )
        self.register_buffer(
            "multipliers",
            torch.zeros(
                len(self.layout.layer_ids),
                self.layout.max_ngram_size,
                dtype=torch.int64,
            ),
            persistent=False,
        )
        self.register_buffer(
            "primes",
            torch.tensor(self.layout.primes, dtype=torch.int64),
            persistent=False,
        )
        self.register_buffer(
            "offsets",
            torch.tensor(self.layout.offsets, dtype=torch.int64),
            persistent=False,
        )
        self.ready = False
        if dummy:
            self.load_lookup(torch.arange(config.vocab_size, dtype=torch.int32))

    def load_lookup(self, value: torch.Tensor) -> None:
        if self.layout.load_lookup(value):
            self.token_map.copy_(
                torch.tensor(
                    self.layout.token_map,
                    dtype=torch.int64,
                    device=self.token_map.device,
                )
            )
            self.multipliers.copy_(
                torch.tensor(
                    self.layout.multipliers,
                    dtype=torch.int64,
                    device=self.multipliers.device,
                )
            )
        self.ready = True

    def forward(
        self,
        input_ids: torch.Tensor,
        lookback: torch.Tensor,
        query_start_loc: torch.Tensor,
        positions: torch.Tensor,
        num_actual_tokens: int,
    ) -> torch.Tensor:
        """Return [hashed rows, padded tokens, hash columns] int32.

        Row i is global Engram row ``self.rows[i]`` (all rows by default).

        lookback[r, 0] is the token immediately before request r's chunk.
        -1 means unavailable history; invalid positive token ids are dead.
        ``positions`` are the global 1-D token positions of the batch. They
        are the same on every DCP rank, unlike rank-local attention seq_lens.
        """
        layers = len(self.rows)
        multipliers = self.multipliers.index_select(0, self.row_index)
        primes_all = self.primes.index_select(0, self.row_index)
        offsets_all = self.offsets.index_select(0, self.row_index)
        tokens = input_ids.numel()
        columns = self.offsets.shape[1]
        num_requests = query_start_loc.numel() - 1
        if tokens == 0 or num_requests <= 0:
            return input_ids.new_full((layers, tokens, columns), -1, dtype=torch.int32)
        if not self.ready:
            raise ValueError("Missing checkpoint Engram token map")
        if (
            input_ids.ndim != 1
            or positions.ndim != 1
            or positions.numel() < tokens
            or lookback.shape != (num_requests, self.lookback_depth)
        ):
            raise ValueError("Invalid Flash Engram batch/lookback layout")
        rows = torch.arange(tokens, device=input_ids.device)
        requests = torch.searchsorted(query_start_loc[1:], rows, right=True)
        requests = requests.clamp_max(num_requests - 1).long()
        starts = query_start_loc[requests].long()
        logical = positions[:tokens].long()
        pad = self.token_map[self.layout.pad_token_id]
        rolling = torch.zeros(
            (layers, tokens), dtype=torch.int64, device=input_ids.device
        )
        blocked = torch.zeros(tokens, dtype=torch.bool, device=input_ids.device)
        hashes = []
        vocab = self.token_map.numel()
        safe_ids = input_ids.clamp(0, vocab - 1).long()
        for shift in range(self.layout.max_ngram_size):
            batch_rows = (rows - shift).clamp_min(0)
            raw_batch = input_ids[batch_rows]
            batch = self.token_map[safe_ids[batch_rows]]
            batch = torch.where((raw_batch < 0) | (raw_batch >= vocab), -1, batch)
            in_chunk = rows - shift >= starts
            column = shift - (rows - starts) - 1
            raw_history = lookback[
                requests, column.clamp(0, self.lookback_depth - 1).long()
            ]
            known = (column >= 0) & (column < self.lookback_depth) & (raw_history >= 0)
            history = self.token_map[raw_history.clamp(0, vocab - 1).long()]
            history = torch.where(raw_history >= vocab, -1, history)
            source = torch.where(in_chunk, batch, torch.where(known, history, pad))
            blocked |= (logical - shift < 0) | (source < 0)
            source = torch.where(blocked, pad, source)
            rolling = rolling ^ (source[None, :] * multipliers[:, shift, None])
            if shift:
                first = (shift - 1) * self.layout.n_heads
                offsets = offsets_all[:, first : first + self.layout.n_heads]
                primes = primes_all[:, shift - 1, :]
                hashes.append(
                    rolling[:, :, None].remainder(primes[:, None, :])
                    + offsets[:, None, :]
                )
        result = torch.cat(hashes, dim=-1).to(torch.int32)
        alive = (
            (rows < num_actual_tokens)
            & (rows < query_start_loc[-1])
            & (input_ids >= 0)
            & (input_ids < vocab)
            & (input_ids != self.layout.pad_token_id)
        )
        return result.masked_fill(~alive[None, :, None], -1)
