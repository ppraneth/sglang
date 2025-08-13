import logging
from typing import Any, List, Optional

import torch
from torch.nn import Module
from torch.nn.parameter import Parameter

from sglang.srt.layers.linear import LinearBase, UnquantizedLinearMethod
from sglang.srt.layers.parameter import (
    ChannelQuantScaleParameter,
    ModelWeightParameter,
    PerTensorScaleParameter,
)
from sglang.srt.layers.quantization.base_config import (
    LinearMethodBase,
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.srt.layers.quantization.fp8_kernel import is_fp8_fnuz
from sglang.srt.layers.quantization.fp8_utils import (
    Fp8LinearOp,
    normalize_e4m3fn_to_e4m3fnuz,
)
from sglang.srt.layers.quantization.utils import is_layer_skipped
from sglang.srt.utils import get_device_capability, is_cuda

try:
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
        apply_fp8_marlin_linear,
        prepare_fp8_layer_for_marlin,
    )

    MARLIN_FP8_AVAILABLE = True
except ImportError:
    MARLIN_FP8_AVAILABLE = False

    def apply_fp8_marlin_linear(*args, **kwargs):
        raise ImportError("vllm is not installed")

    def prepare_fp8_layer_for_marlin(*args, **kwargs):
        raise ImportError("vllm is not installed")


logger = logging.getLogger(__name__)


class FBGEMMFp8Config(QuantizationConfig):
    """Config class for FBGEMM Fp8."""

    def __init__(self, ignore_list: list[str], input_scale_ub: float):
        super().__init__()
        self.ignore_list = ignore_list if ignore_list else []
        self.input_scale_ub = input_scale_ub

        # For GPUs that lack FP8 hardware support (pre-Hopper), we can leverage
        # the Marlin kernel for fast weight-only FP8 quantization.
        use_marlin = False
        if is_cuda():
            major, _ = get_device_capability()
            # SM 89 is Ada Lovelace. Marlin is for pre-SM89 (pre-Ada) GPUs.
            if major < 8 or (major == 8 and _ == 0):  # Volta, Turing, Ampere
                use_marlin = True

        self.use_marlin = use_marlin and MARLIN_FP8_AVAILABLE

    @classmethod
    def get_name(cls) -> str:
        return "fbgemm_fp8"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "FBGEMMFp8Config":
        ignore_list = cls.get_from_keys(config, ["modules_to_not_convert"])
        input_scale_ub = cls.get_from_keys(config, ["activation_scale_ub"])
        return cls(ignore_list=ignore_list, input_scale_ub=input_scale_ub)

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional["QuantizeMethodBase"]:
        if isinstance(layer, LinearBase):
            # The vLLM original code used self.packed_modules_mapping which
            # is not standard in SGLang's QuantizationConfig.
            # Passing None is the safer default.
            if is_layer_skipped(
                prefix=prefix, ignored_layers=self.ignore_list, fused_mapping=None
            ):
                return UnquantizedLinearMethod()
            return FBGEMMFp8LinearMethod(self)
        return None


class FBGEMMFp8LinearMethod(LinearMethodBase):
    """
    SGLang implementation of the FBGEMM FP8 quantization method.
    """

    def __init__(self, quant_config: FBGEMMFp8Config):
        self.quant_config = quant_config
        # FBGEMM format uses dynamic, per-token activation quantization.
        # This translates to: static_input_quant=False, use_per_token_if_dynamic=True
        self.fp8_linear = Fp8LinearOp(
            static_input_quant=False, use_per_token_if_dynamic=True
        )
        self.out_dtype = torch.get_default_dtype()

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
        # NOTE: maybe_create_device_identity() is not needed in SGLang
        weight_loader = extra_weight_attrs.get("weight_loader")
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
        # SGLang uses PerTensorScaleParameter to handle scales of any shape.
        # This replaces vLLM's ChannelQuantScaleParameter.
        weight_scale = PerTensorScaleParameter(
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
        layer.register_parameter("input_scale_ub", input_scale_ub)

    def process_weights_after_loading(self, layer: Module) -> None:
        # Re-wrap parameters to be safe with torch.compile
        layer.weight_scale = Parameter(layer.weight_scale.data, requires_grad=False)
        layer.weight = Parameter(layer.weight.data, requires_grad=False)

        weight = layer.weight.data

        # Handle ROCm-specific FP8 format normalization
        if is_fp8_fnuz():
            weight, weight_scale, _ = normalize_e4m3fn_to_e4m3fnuz(
                weight=weight, weight_scale=layer.weight_scale.data, input_scale=None
            )
            layer.weight_scale = Parameter(weight_scale, requires_grad=False)

        # Transpose weight for GEMM
        layer.weight = Parameter(weight.t(), requires_grad=False)

        if self.quant_config.use_marlin:
            prepare_fp8_layer_for_marlin(layer)
            # Activations are not quantized for Marlin, so upper bound is not needed
            if hasattr(layer, "input_scale_ub"):
                del layer.input_scale_ub

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
                weight_scale=layer.weight_scale,
                workspace=layer.workspace,
                size_n=layer.output_size_per_partition,
                size_k=layer.input_size_per_partition,
                bias=bias,
            )

        return self.fp8_linear.apply(
            input=x,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            out_dtype=self.out_dtype,
            input_scale=None,  # Dynamic quantization, so scale is None
            input_scale_ub=layer.input_scale_ub,
            bias=bias,
        )
