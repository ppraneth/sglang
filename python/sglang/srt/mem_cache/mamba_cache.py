# python/sglang/srt/mem_cache/mamba_cache.py

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project.
# Adapted from vllm/model_executor/models/mamba_cache.py

from dataclasses import dataclass

import torch

from sglang.srt.managers.schedule_batch import ModelWorkerBatch


@dataclass
class MambaCacheParams:
    """A data class to hold the tensors for a Mamba forward pass for all layers."""

    conv_states: torch.Tensor
    ssm_states: torch.Tensor
    slot_mapping: torch.Tensor

    def get_layer(self, layer_idx: int):
        """Return the cache parameters for a specific layer."""
        return MambaLayerCacheParams(
            conv_state=self.conv_states[layer_idx],
            ssm_state=self.ssm_states[layer_idx],
            slot_mapping=self.slot_mapping,
        )


@dataclass
class MambaLayerCacheParams:
    """A data class to hold the tensors for a single Mamba layer's forward pass."""

    conv_state: torch.Tensor
    ssm_state: torch.Tensor
    slot_mapping: torch.Tensor


class MambaCache:
    """A class to manage the memory for Mamba's convolutional and SSM states."""

    def __init__(
        self,
        max_batch_size: int,
        num_layers: int,
        dtype: torch.dtype,
        conv_state_shape: tuple[int, int],
        ssm_state_shape: tuple[int, int, int],
        device: str = "cuda",
    ):
        self.num_layers = num_layers
        self.max_batch_size = max_batch_size
        self.dtype = dtype
        self.device = device

        # Allocate the master tensors for the states
        # Shape: [num_layers, max_batch_size, ...]
        self.conv_states = torch.zeros(
            (num_layers, max_batch_size) + conv_state_shape, dtype=dtype, device=device
        )
        self.ssm_states = torch.zeros(
            (num_layers, max_batch_size) + ssm_state_shape, dtype=dtype, device=device
        )

    def prepare_mamba_inputs(self, batch: ModelWorkerBatch) -> MambaCacheParams:
        """
        Prepare the MambaCacheParams for the current forward pass.
        This involves gathering the states for the sequences in the current batch.
        """
        # req_pool_indices is the equivalent of slot_mapping or cache_indices.
        slot_mapping = batch.req_pool_indices.long()

        return MambaCacheParams(
            conv_states=self.conv_states,
            ssm_states=self.ssm_states,
            slot_mapping=slot_mapping,
        )

    def copy(self, src_slot: int, dst_slot: int):
        """Copy a cache slot from a source to a destination."""
        self.conv_states[:, dst_slot].copy_(self.conv_states[:, src_slot])
        self.ssm_states[:, dst_slot].copy_(self.ssm_states[:, src_slot])

    def fork(self, src_slot: int, dst_slot: int):
        """Alias for copy, used when forking a sequence."""
        self.copy(src_slot, dst_slot)
