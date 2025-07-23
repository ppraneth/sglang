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
        self.use_marlin = (
            get_bool_env_var("SGLANG_FORCE_FP8_MARLIN") and MARLIN_FP8_AVAILABLE
        )
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

    def get_scaled_act_names(self) -> List[str]:
        return []


class FBGEMMFp8LinearMethod(LinearMethodBase):
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
            data=torch.empty((sum(output_partition_sizes), 1), dtype=torch.float32),
            output_dim=0,
            weight_loader=weight_loader,
        )
        weight_scale.data.fill_(torch.finfo(torch.float32).min)
        layer.register_parameter("weight_scale", weight_scale)
        input_scale_ub = torch.nn.Parameter(
            torch.tensor(self.quant_config.input_scale_ub, dtype=torch.float32),
            requires_grad=False,
        )
        layer.input_scale_ub = input_scale_ub

    def process_weights_after_loading(self, layer: Module) -> None:
        layer.weight = Parameter(layer.weight.data, requires_grad=False)
        layer.weight_scale = Parameter(layer.weight_scale.data, requires_grad=False)

        if _is_fp8_fnuz:
            # For the e4m3fnuz FP8 format, both the weight and activation scales
            # must be doubled to maintain numerical equivalence with e4m3fn.

            # 1. Normalize the weight tensor itself for the fnuz format
            #    and get the correctly doubled weight_scale.
            new_weight, new_weight_scale = normalize_e4m3fn_to_e4m3fnuz(
                weight=layer.weight.data,
                weight_scale=layer.weight_scale.data,
            )
            layer.weight = Parameter(new_weight, requires_grad=False)
            layer.weight_scale = Parameter(new_weight_scale, requires_grad=False)

            # 2. Explicitly double the activation's upper-bound scale.
            #    This is done in-place for safety.
            if hasattr(layer, "input_scale_ub"):
                layer.input_scale_ub.data.mul_(2.0)

        if self.quant_config.use_marlin:
            # Marlin expects the original, non-transposed weight shape.
            prepare_fp8_layer_for_marlin(layer, size_k_first=False)
            if hasattr(layer, "input_scale_ub"):
                del layer.input_scale_ub
        else:
            # For the default (non-Marlin) path, transpose the weight.
            layer.weight = Parameter(layer.weight.data.t(), requires_grad=False)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # print(f"\n--- DEBUG: fbgemm_fp8.py | FBGEMMFp8LinearMethod.apply ---")
        # print(f"Input 'x': shape={x.shape}, dtype={x.dtype}, has_nan={torch.isnan(x).any()}, has_inf={torch.isinf(x).any()}")
        if self.quant_config.use_marlin:
            # The low-level Marlin kernel requires the weight_scale tensor
            # to be 2D (rank 2), which is its original shape [N, 1].
            return apply_fp8_marlin_linear(
                input=x,
                weight=layer.weight,
                weight_scale=layer.weight_scale,
                workspace=layer.workspace,
                size_n=layer.output_size_per_partition,
                size_k=layer.input_size_per_partition,
                bias=bias,
            )

        # The default kernel path might be more flexible. We flatten the
        # scale tensor to a 1D vector just in case.
        return apply_fp8_linear(
            input=x,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            input_scale=None,
            input_scale_ub=layer.input_scale_ub,
            bias=bias,
            cutlass_fp8_supported=_cutlass_fp8_supported,
            use_per_token_if_dynamic=False,
        )
