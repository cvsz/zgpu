"""Paged / Continuous KV-Cache Engine for zGPU.

Optimizes autoregressive generation by caching Key and Value states across decoding
steps, avoiding redundant re-computation of previous tokens.
"""

from __future__ import annotations


class KVCache:
    """Manages Key-Value states for Transformer layers on zGPU."""

    def __init__(self, max_batch_size: int = 1, max_seq_len: int = 128, num_layers: int = 2, d_model: int = 16):
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.num_layers = num_layers
        self.d_model = d_model

        # k_cache[layer][seq_idx] = List[float] (dim: d_model)
        self.k_cache: list[list[list[float]]] = [[[] for _ in range(max_seq_len)] for _ in range(num_layers)]
        self.v_cache: list[list[list[float]]] = [[[] for _ in range(max_seq_len)] for _ in range(num_layers)]
        self.current_seq_len = 0

    def update(
        self, layer_idx: int, k_new: list[float], v_new: list[float]
    ) -> tuple[list[list[float]], list[list[float]]]:
        """Append new K and V token vectors to cache and return the active sequence history."""
        if self.current_seq_len < self.max_seq_len:
            self.k_cache[layer_idx][self.current_seq_len] = list(k_new)
            self.v_cache[layer_idx][self.current_seq_len] = list(v_new)

        all_k = [self.k_cache[layer_idx][i] for i in range(self.current_seq_len + 1)]
        all_v = [self.v_cache[layer_idx][i] for i in range(self.current_seq_len + 1)]
        return all_k, all_v

    def advance_step(self):
        """Advance sequence index after all layers process the token."""
        self.current_seq_len += 1

    def reset(self):
        """Flush cache for a new prompt."""
        self.current_seq_len = 0
        for layer in range(self.num_layers):
            for s in range(self.max_seq_len):
                self.k_cache[layer][s].clear()
                self.v_cache[layer][s].clear()
