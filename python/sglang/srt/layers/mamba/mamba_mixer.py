# python/sglang/srt/layers/mamba/mamba_mixer.py

# Adapted from transformers.models.mamba.modeling_mamba.MambaMixer
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn
from torch.nn.parameter import Parameter

from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from sglang.srt.layers.mamba.ops.mamba_ssm import (
    selective_scan_fn,
    selective_state_update,
)
from sglang.srt.managers.schedule_batch import ForwardMode, ModelWorkerBatch
from sglang.srt.mem_cache.mamba_cache import MambaLayerCacheParams


class MambaMixer(nn.Module):
    def __init__(self, config, tp_group):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.ssm_state_size = config.state_size
        self.conv_kernel_size = config.conv_kernel
        self.intermediate_size = config.intermediate_size
        self.time_step_rank = config.time_step_rank
        self.use_conv_bias = config.use_conv_bias
        self.use_bias = config.use_bias
        self.activation = "silu"
        self.tp_group = tp_group
        self.tp_size = tp_group.size() if tp_group is not None else 1

        # Projections
        self.in_proj = MergedColumnParallelLinear(
            self.hidden_size,
            [self.intermediate_size] * 2,
            bias=self.use_bias,
            tp_group=self.tp_group,
        )
        self.x_proj = RowParallelLinear(
            self.intermediate_size,
            self.time_step_rank + self.ssm_state_size * 2,
            bias=False,
            tp_group=self.tp_group,
        )
        self.dt_proj = ColumnParallelLinear(
            self.time_step_rank,
            self.intermediate_size,
            bias=True,
            tp_group=self.tp_group,
            skip_bias_add=True,
        )
        self.out_proj = RowParallelLinear(
            self.intermediate_size,
            self.hidden_size,
            bias=self.use_bias,
            tp_group=self.tp_group,
            input_is_parallel=True,
        )

        # 1D Convolution
        self.conv1d = nn.Conv1d(
            in_channels=self.intermediate_size,
            out_channels=self.intermediate_size,
            bias=self.use_conv_bias,
            kernel_size=self.conv_kernel_size,
            groups=self.intermediate_size,
            padding=self.conv_kernel_size - 1,
        )

        # A and D parameters (state-independent)
        self.A_log = nn.Parameter(
            torch.empty(self.intermediate_size // self.tp_size, self.ssm_state_size)
        )
        self.D = nn.Parameter(torch.empty(self.intermediate_size // self.tp_size))

    def forward(
        self,
        hidden_states: torch.Tensor,
        mamba_layer_cache: MambaLayerCacheParams,
        batch: ModelWorkerBatch,
    ):

        # 1. Gated MLP's linear projection
        projected_states, gate = self.in_proj(hidden_states)

        # 2. Convolution sequence transformation
        if batch.forward_mode == ForwardMode.PREFILL:
            # Varlen case for prefill: input is (total_tokens, dim)
            projected_states = causal_conv1d_fn(
                x=projected_states.t().contiguous(),
                weight=self.conv1d.weight.squeeze(1),
                bias=self.conv1d.bias,
                conv_states=mamba_layer_cache.conv_state,
                # A sequence has an initial state if its prefix length is > 0
                has_initial_state=(
                    torch.tensor(batch.extend_prefix_lens, device=hidden_states.device)
                    > 0
                ),
                cache_indices=mamba_layer_cache.slot_mapping,
                query_start_loc=batch.query_start_loc,
                activation=self.activation,
            ).t()
        else:  # DECODE
            # Decode case: input is (batch_size, dim)
            projected_states = causal_conv1d_update(
                x=projected_states,
                conv_state=mamba_layer_cache.conv_state,
                weight=self.conv1d.weight.squeeze(1),
                bias=self.conv1d.bias,
                activation=self.activation,
                conv_state_indices=mamba_layer_cache.slot_mapping,
            )

        # 3. State Space Model sequence transformation
        ssm_parameters = self.x_proj(projected_states)
        time_step, B, C = torch.split(
            ssm_parameters,
            [self.time_step_rank, self.ssm_state_size, self.ssm_state_size],
            dim=-1,
        )

        discrete_time_step, dt_bias = self.dt_proj(time_step)
        A = -torch.exp(self.A_log.float())
        D = self.D.float()

        if batch.forward_mode == ForwardMode.PREFILL:
            scan_outputs = selective_scan_fn(
                u=projected_states.t().contiguous(),
                ssm_states=mamba_layer_cache.ssm_state,
                delta=discrete_time_step.t().contiguous(),
                A=A,
                B=B.t().contiguous(),
                C=C.t().contiguous(),
                D=D,
                z=gate.t().contiguous(),
                delta_bias=dt_bias,
                delta_softplus=True,
                cache_indices=mamba_layer_cache.slot_mapping,
                has_initial_state=(
                    torch.tensor(batch.extend_prefix_lens, device=hidden_states.device)
                    > 0
                ),
                query_start_loc=batch.query_start_loc,
            ).t()
        else:  # DECODE
            scan_outputs = selective_state_update(
                state=mamba_layer_cache.ssm_state,
                x=projected_states,
                dt=discrete_time_step,
                A=A,
                B=B,
                C=C,
                D=D,
                z=gate,
                dt_bias=dt_bias,
                dt_softplus=True,
                state_batch_indices=mamba_layer_cache.slot_mapping,
            )

        # 4. Final linear projection
        contextualized_states = self.out_proj(scan_outputs)
        return contextualized_states
