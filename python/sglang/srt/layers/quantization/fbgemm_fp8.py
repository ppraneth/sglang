# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project.
# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/fbgemm_fp8.py

import logging
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
from sglang.srt.layers.quantization.fp8_kernel import is_fp8_fnuz
from sglang.srt.layers.quantization.fp8_utils import (
    apply_fp8_linear,
    cutlass_fp8_supported,
    normalize_e4m3fn_to_e4m3fnuz,
)
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.layers.quantization.utils import is_layer_skipped
from sglang.srt.utils import get_bool_env_var, is_hip

try:
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (  # type: ignore
        apply_fp8_marlin_linear,
        prepare_fp8_layer_for_marlin,
    )

    MARLIN_FP8_AVAILABLE = True
except ImportError:
    MARLIN_FP8_AVAILABLE = False

    def dummy_func(*args, **kwargs):
        raise ImportError(
            "Marlin FP8 requires some operators from vLLM. Please install vLLm."
        )

    apply_fp8_marlin_linear = prepare_fp8_layer_for_marlin = dummy_func


logger = logging.getLogger(__name__)

_is_hip = is_hip()
_is_fp8_fnuz = is_fp8_fnuz()
_cutlass_fp8_supported = cutlass_fp8_supported()


class FBGEMMFp8Config(QuantizationConfig):
    """Config class for FBGEMM Fp8."""

    def __init__(
        self, ignore_list: Optional[list[str]] = None, input_scale_ub: float = 1.0
    ):
        super().__init__()
        self.ignore_list = ignore_list if ignore_list else []
        self.input_scale_ub = input_scale_ub

        # For GPUs that lack FP8 hardware support, we can leverage the Marlin
        # kernel for fast weight-only FP8 quantization.
        self.use_marlin = (
            get_bool_env_var("SGLANG_FORCE_FP8_MARLIN") and MARLIN_FP8_AVAILABLE
        )
        # Disable marlin for ROCm
        if _is_hip:
            self.use_marlin = False

    @classmethod
    def get_name(cls) -> str:
        return "fbgemm_fp8"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return []

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "FBGEMMFp8Config":
        ignore_list = cls.get_from_keys(config, ["modules_to_not_convert"])
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

    def get_scaled_act_names(self) -> List[str]:
        return []


class FBGEMMFp8LinearMethod(LinearMethodBase):
    """
    Linear method for FBGEMM-style FP8 quantization.
    This method uses per-channel static weight quantization and per-token
    dynamic activation quantization with an upper-bound scaling factor.
    """

    def __init__(self, quant_config: FBGEMMFp8Config):
        self.quant_config = quant_config
        self.out_dtype = torch.get_default_dtype()

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        weight_loader = extra_weight_attrs.get("weight_loader")
        del input_size, output_size
        output_size_per_partition = sum(output_partition_sizes)

        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype

        # WEIGHT
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

        # WEIGHT SCALE
        weight_scale = ChannelQuantScaleParameter(
            data=torch.empty((sum(output_partition_sizes), 1), dtype=torch.float32),
            output_dim=0,
            weight_loader=weight_loader,
        )
        weight_scale.data.fill_(torch.finfo(torch.float32).min)
        layer.register_parameter("weight_scale", weight_scale)

        # INPUT SCALE UPPER BOUND
        input_scale_ub = torch.nn.Parameter(
            torch.tensor(self.quant_config.input_scale_ub, dtype=torch.float32),
            requires_grad=False,
        )
        layer.input_scale_ub = input_scale_ub

    def process_weights_after_loading(self, layer: Module) -> None:
        # Required by torch.compile
        layer.weight = Parameter(layer.weight.data, requires_grad=False)
        layer.weight_scale = Parameter(layer.weight_scale.data, requires_grad=False)

        weight = layer.weight
        weight_scale = layer.weight_scale

        if _is_fp8_fnuz:
            weight, weight_scale, _ = normalize_e4m3fn_to_e4m3fnuz(
                weight=weight, weight_scale=weight_scale, input_scale=None
            )
            layer.weight_scale = Parameter(weight_scale, requires_grad=False)

        layer.weight = Parameter(weight.t(), requires_grad=False)

        if self.quant_config.use_marlin:
            prepare_fp8_layer_for_marlin(layer, size_k_first=False)
            # Activations are not quantized for Marlin.
            if hasattr(layer, "input_scale_ub"):
                del layer.input_scale_ub
        else:
            layer.weight = Parameter(layer.weight.t(), requires_grad=False)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        if self.quant_config.use_marlin:
            return apply_fp8_marlin_linear(
                input=x,
                weight=layer.weight,
                weight_scale=layer.weight_scale.view(-1),
                workspace=layer.workspace,
                size_n=layer.output_size_per_partition,
                size_k=layer.input_size_per_partition,
                bias=bias,
            )

        return apply_fp8_linear(
            input=x,
            weight=layer.weight,
            weight_scale=layer.weight_scale.view(-1),
            input_scale=None,
            input_scale_ub=layer.input_scale_ub,
            bias=bias,
            cutlass_fp8_supported=_cutlass_fp8_supported,
            use_per_token_if_dynamic=False,
        )
