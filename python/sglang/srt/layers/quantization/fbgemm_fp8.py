# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/fbgemm_fp8.py

from typing import Any, Dict, List, Optional

import torch
from torch.nn import Module
from torch.nn.parameter import Parameter

from sglang.srt.layers.linear import LinearBase
from sglang.srt.layers.parameter import ChannelQuantScaleParameter, ModelWeightParameter
from sglang.srt.layers.quantization.base_config import (
    LinearMethodBase,
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.srt.layers.quantization.fp8_kernel import fp8_dtype, fp8_max, is_fp8_fnuz
from sglang.srt.layers.quantization.fp8_utils import normalize_e4m3fn_to_e4m3fnuz
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.layers.quantization.utils import is_layer_skipped


class FBGEMMFp8Config(QuantizationConfig):
    """Configuration class for FBGEMM FP8 quantization in SGLang."""

    def __init__(self, ignore_list: List[str], input_scale_ub: float):
        super().__init__()
        self.ignore_list = ignore_list if ignore_list else []
        self.input_scale_ub = input_scale_ub

    @classmethod
    def get_name(cls) -> str:
        return "fbgemm_fp8"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        # The fallback implementation does not require special capability.
        return 0

    @staticmethod
    def get_config_filenames() -> List[str]:
        return []

    def get_scaled_act_names(self) -> List[str]:
        return []

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "FBGEMMFp8Config":
        """Creates a config from the model's quantization_config dictionary."""
        ignore_list = cls.get_from_keys_or(config, ["modules_to_not_convert"], [])
        input_scale_ub = cls.get_from_keys(config, ["activation_scale_ub"])
        return cls(ignore_list=ignore_list, input_scale_ub=input_scale_ub)

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional["QuantizeMethodBase"]:
        if isinstance(layer, LinearBase):
            if is_layer_skipped(prefix, self.ignore_list):
                return UnquantizedLinearMethod()
            return FBGEMMFp8LinearMethod(self)
        return None


class FBGEMMFp8LinearMethod(LinearMethodBase):
    """
    Quantization method for FBGEMM FP8 linear layers.
    """

    def __init__(self, quant_config: FBGEMMFp8Config):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        del input_size, output_size
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")

        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype
        layer.logical_widths = output_partition_sizes

        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                dtype=torch.float8_e4m3fn,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        weight_scale = ChannelQuantScaleParameter(
            data=torch.empty((output_size_per_partition, 1), dtype=torch.float32),
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_scale", weight_scale)

        input_scale_ub = torch.nn.Parameter(
            torch.tensor(self.quant_config.input_scale_ub, dtype=torch.float32),
            requires_grad=False,
        )
        layer.register_parameter("input_scale_ub", input_scale_ub)

    def process_weights_after_loading(self, layer: Module) -> None:
        layer.weight = Parameter(layer.weight.data, requires_grad=False)
        layer.weight_scale = Parameter(layer.weight_scale.data, requires_grad=False)

        if is_fp8_fnuz():
            weight, weight_scale, _ = normalize_e4m3fn_to_e4m3fnuz(
                weight=layer.weight, weight_scale=layer.weight_scale, input_scale=None
            )
            layer.weight = Parameter(weight, requires_grad=False)
            layer.weight_scale = Parameter(weight_scale, requires_grad=False)

        layer.weight = Parameter(layer.weight.t(), requires_grad=False)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        output_shape = [*x.shape[:-1], layer.output_size_per_partition]
        input_2d = x.view(-1, layer.input_size_per_partition)

        # 1. Dynamically quantize activation.
        # Upcast to float32 for all scaling calculations to maximize precision.
        x_f32 = input_2d.to(torch.float32)
        x_amax = x_f32.abs().max(dim=-1, keepdim=True)[0]
        x_amax.clamp_(min=1e-12)

        x_scale = x_amax / fp8_max
        x_scale = torch.minimum(x_scale, layer.input_scale_ub)

        # Quantize the float32 tensor before casting to fp8.
        x_q = (x_f32 / x_scale).to(fp8_dtype)

        # 2. Perform scaled matrix multiplication using the reliable PyTorch fallback.
        device_identity = torch.ones(1, dtype=torch.float32, device=layer.weight.device)
        output_q = torch._scaled_mm(
            x_q,
            layer.weight,
            scale_a=device_identity,
            scale_b=device_identity,
            out_dtype=torch.float32,
        )[0]

        # 3. Dequantize the output.
        output = output_q * x_scale * layer.weight_scale.t()

        if bias is not None:
            output = output + bias

        output = output.to(x.dtype)

        return output.view(*output_shape)
