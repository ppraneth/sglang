# python/sglang/srt/models/mamba.py

# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/mamba.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Iterable, Optional, Tuple

import torch
from torch import nn
from transformers import MambaConfig

from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import MergedColumnParallelLinear, RowParallelLinear
from sglang.srt.layers.logits_processor import LogitsProcessor, LogitsProcessorOutput
from sglang.srt.layers.mamba.mamba_mixer import MambaMixer
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.managers.schedule_batch import ModelWorkerBatch
from sglang.srt.mem_cache.mamba_cache import MambaCacheParams
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.utils import add_prefix, make_layers


class MambaDecoderLayer(nn.Module):
    def __init__(
        self,
        config: MambaConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        tp_group=None,
    ) -> None:
        super().__init__()
        self.config = config
        self.mixer = MambaMixer(config, tp_group=tp_group)
        self.norm = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        mamba_layer_cache: MambaCacheParams,
        batch: ModelWorkerBatch,
    ):
        if residual is None:
            residual = hidden_states
            hidden_states = self.norm(hidden_states)
        else:
            hidden_states, residual = self.norm(hidden_states, residual)

        hidden_states = self.mixer(hidden_states, mamba_layer_cache, batch)
        return hidden_states, residual


class MambaModel(nn.Module):
    def __init__(
        self,
        config: MambaConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        tp_group=None,
    ) -> None:
        super().__init__()
        self.config = config
        self.tp_group = tp_group

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("embed_tokens", prefix),
        )

        self.layers, self.start_layer, self.end_layer = make_layers(
            config.num_hidden_layers,
            lambda idx, p: MambaDecoderLayer(
                config=config, quant_config=quant_config, prefix=p, tp_group=tp_group
            ),
            prefix=add_prefix("layers", prefix),
            # SGLang does not yet support PP for Mamba, so we run all layers
            pp_rank=0,
            pp_size=1,
        )

        self.norm_f = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        batch: ModelWorkerBatch,
        mamba_cache_params: MambaCacheParams,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embed_tokens(input_ids)

        residual = None

        for i in range(self.config.num_hidden_layers):
            layer = self.layers[i]
            # Get the cache for the specific layer
            layer_cache_params = mamba_cache_params.get_layer(i)
            hidden_states, residual = layer(
                hidden_states=hidden_states,
                residual=residual,
                mamba_layer_cache=layer_cache_params,
                batch=batch,
            )

        hidden_states, _ = self.norm_f(hidden_states, residual)
        return hidden_states


class MambaForCausalLM(nn.Module):
    def __init__(
        self,
        config: MambaConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        tp_group=None,
    ) -> None:
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.tp_group = tp_group

        self.model = MambaModel(
            config,
            quant_config=quant_config,
            prefix=add_prefix("backbone", prefix),
            tp_group=tp_group,
        )

        if config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("lm_head", prefix),
            )

        self.logits_processor = LogitsProcessor(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        batch: ModelWorkerBatch,
        mamba_cache_params: MambaCacheParams,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> LogitsProcessorOutput:

        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            batch=batch,
            mamba_cache_params=mamba_cache_params,
            inputs_embeds=inputs_embeds,
        )

        return self.logits_processor(input_ids, hidden_states, self.lm_head, batch)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if "A_log" in name:
                # The original checkpoint stores A_log, but our parameter is named A
                name = name.replace("A_log", "A")

            # The conv1d weight is of shape (dim, 1, width) in checkpoints,
            # but our nn.Conv1d expects (dim, 1, width). We handle this during loading.
            if "conv1d.weight" in name and loaded_weight.dim() == 2:
                loaded_weight = loaded_weight.unsqueeze(1)

            if name not in params_dict:
                continue

            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)


# The entry class for SGLang model registration
EntryClass = MambaForCausalLM
