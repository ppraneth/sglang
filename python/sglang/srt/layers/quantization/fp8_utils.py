from typing import Callable, List, Optional, Tuple

import torch

from sglang.srt.layers.quantization import deep_gemm_wrapper
from sglang.srt.layers.quantization.fp8_kernel import sglang_per_token_group_quant_fp8
from sglang.srt.layers.utils import is_sm100_supported

try:
    from vllm import _custom_ops as ops

    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

from sglang.srt.layers.quantization.fp8_kernel import (
    fp8_dtype,
    fp8_max,
    is_fp8_fnuz,
    per_token_group_quant_fp8,
    scaled_fp8_quant,
    sglang_per_token_quant_fp8,
    static_quant_fp8,
    w8a8_block_fp8_matmul_deepgemm,
    w8a8_block_fp8_matmul_triton,
)
from sglang.srt.utils import (
    align,
    get_bool_env_var,
    get_cuda_version,
    get_device_capability,
    is_cuda,
    is_flashinfer_available,
    is_hip,
)

_is_hip = is_hip()
_is_cuda = is_cuda()
_is_fp8_fnuz = is_fp8_fnuz()

_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip

if _use_aiter:
    import aiter
    from aiter import gemm_a8w8_blockscale, get_hip_quant

    aiter_per1x128_quant = get_hip_quant(aiter.QuantType.per_1x128)

if _is_cuda:
    from sgl_kernel import fp8_blockwise_scaled_mm, fp8_scaled_mm

use_vllm_cutlass_w8a8_fp8_kernel = get_bool_env_var("USE_VLLM_CUTLASS_W8A8_FP8_KERNEL")

# Input scaling factors are no longer optional in _scaled_mm starting
# from pytorch 2.5. Allocating a dummy tensor to pass as input_scale
TORCH_DEVICE_IDENTITY = None


def use_rowwise_torch_scaled_mm():
    _TORCH_VERSION = torch.__version__.split("+")[0]
    try:
        _TORCH_VERSION_TUPLE = tuple(map(int, _TORCH_VERSION.split(".")[:3]))
    except ValueError:
        _TORCH_VERSION_TUPLE = (0, 0, 0)
    if _is_hip:
        # The condition to determine if it is on a platform that supports
        # torch._scaled_mm rowwise feature.
        # The condition is determined once as the operations
        # are time consuming.
        return get_device_capability() >= (9, 4) and _TORCH_VERSION_TUPLE >= (2, 7, 0)
    return False


USE_ROWWISE_TORCH_SCALED_MM = use_rowwise_torch_scaled_mm()


def cutlass_fp8_supported():
    if not _is_cuda:
        return False
    major, minor = get_device_capability()
    cuda_version = get_cuda_version()
    if major >= 9:
        return cuda_version >= (12, 0)
    elif major == 8 and minor == 9:
        return cuda_version >= (12, 4)
    return False


def normalize_e4m3fn_to_e4m3fnuz(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    input_scale: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    assert weight.dtype == torch.float8_e4m3fn
    # The bits pattern 10000000(-128) represents zero in e4m3fn
    # but NaN in e4m3fnuz. So here we set it to 0.
    # https://onnx.ai/onnx/technical/float8.html
    weight_as_int8 = weight.view(torch.int8)
    ROCM_FP8_NAN_AS_INT = -128
    weight_as_int8[weight_as_int8 == ROCM_FP8_NAN_AS_INT] = 0
    weight = weight_as_int8.view(torch.float8_e4m3fnuz)

    # For the same bits representation, e4m3fnuz value is half of
    # the e4m3fn value, so we should double the scaling factor to
    # get the same dequantized value.
    # https://onnx.ai/onnx/technical/float8.html
    weight_scale = weight_scale * 2.0
    return weight, weight_scale


def cutlass_block_fp8_supported() -> bool:
    if not get_bool_env_var("SGLANG_SUPPORT_CUTLASS_BLOCK_FP8"):
        return False
    if _is_cuda:
        major, minor = torch.cuda.get_device_capability()
        sm_version = major * 10 + minor
        cuda_version = tuple(map(int, torch.version.cuda.split(".")))
        if cuda_version >= (12, 0) and sm_version >= 90:
            return True
    return False


CUTLASS_BLOCK_FP8_SUPPORTED = cutlass_block_fp8_supported()
ENABLE_FLASHINFER_GEMM = (
    get_bool_env_var("SGLANG_ENABLE_FLASHINFER_GEMM")
    and is_sm100_supported()
    and is_flashinfer_available()
)
if ENABLE_FLASHINFER_GEMM:
    from flashinfer.gemm import gemm_fp8_nt_groupwise


def dispatch_w8a8_block_fp8_linear() -> Callable:
    if ENABLE_FLASHINFER_GEMM:
        return flashinfer_gemm_w8a8_block_fp8_linear
    elif CUTLASS_BLOCK_FP8_SUPPORTED:
        return cutlass_w8a8_block_fp8_linear_with_fallback
    elif _use_aiter:
        return aiter_w8a8_block_fp8_linear
    elif deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM:
        return deepgemm_w8a8_block_fp8_linear_with_fallback
    else:
        return triton_w8a8_block_fp8_linear


def flashinfer_gemm_w8a8_block_fp8_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    block_size: List[int],
    weight_scale: torch.Tensor,
    input_scale: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    assert input_scale is None

    input_2d = input.view(-1, input.shape[-1])
    output_shape = [*input.shape[:-1], weight.shape[0]]

    q_input, x_scale = sglang_per_token_group_quant_fp8(
        input_2d, block_size[1], column_major_scales=False
    )

    output = gemm_fp8_nt_groupwise(
        q_input,
        weight,
        x_scale,
        weight_scale,
        scale_major_mode="K",
        out_dtype=input_2d.dtype,
    )

    if bias is not None:
        output += bias

    return output.to(dtype=input_2d.dtype).view(*output_shape)


def cutlass_w8a8_block_fp8_linear_with_fallback(
    input: torch.Tensor,
    weight: torch.Tensor,
    block_size: List[int],
    weight_scale: torch.Tensor,
    input_scale: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    assert input_scale is None

    # TODO: add more robust shape check here
    shape_supported = weight.shape[0] % 128 == 0 and weight.shape[1] % 128 == 0

    if not shape_supported:
        # fallback to triton
        return triton_w8a8_block_fp8_linear(
            input, weight, block_size, weight_scale, input_scale, bias
        )

    input_2d = input.view(-1, input.shape[-1])
    output_shape = [*input.shape[:-1], weight.shape[0]]

    q_input, x_scale = per_token_group_quant_fp8(
        input_2d, block_size[1], column_major_scales=True
    )
    output = fp8_blockwise_scaled_mm(
        q_input, weight.T, x_scale, weight_scale.T, out_dtype=input_2d.dtype
    )
    if bias is not None:
        output += bias
    return output.to(dtype=input_2d.dtype).view(*output_shape)


def deepgemm_w8a8_block_fp8_linear_with_fallback(
    input: torch.Tensor,
    weight: torch.Tensor,
    block_size: List[int],
    weight_scale: torch.Tensor,
    input_scale: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    assert input_scale is None

    output_dtype = input.dtype
    dtype_supported = output_dtype == torch.bfloat16

    # TODO: https://github.com/sgl-project/sglang/pull/6890#issuecomment-2943395737
    shape_supported = weight.shape[0] % 64 == 0 and weight.shape[1] % 128 == 0

    if not (shape_supported and dtype_supported):
        # fall back to triton
        return triton_w8a8_block_fp8_linear(
            input, weight, block_size, weight_scale, input_scale, bias
        )

    input_2d = input.view(-1, input.shape[-1])
    output_shape = [*input.shape[:-1], weight.shape[0]]

    q_input, x_scale = sglang_per_token_group_quant_fp8(
        input_2d,
        block_size[1],
        column_major_scales=True,
        scale_tma_aligned=True,
        scale_ue8m0=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
    )

    # NOTE(alcanderian): Useless when scale is packed to int32
    # if get_bool_env_var("SGLANG_W8A8_DEEPGEMM_SANITY_CHECK_UE8M0"):
    #     _check_ue8m0("x_scale", x_scale)
    #     _check_ue8m0("weight_scale", ws)

    output = w8a8_block_fp8_matmul_deepgemm(
        q_input, weight, x_scale, weight_scale, block_size, output_dtype=output_dtype
    )
    if bias is not None:
        output += bias
    return output.to(dtype=output_dtype).view(*output_shape)


def _check_ue8m0(name, x):
    x_ceil = ceil_to_ue8m0(x)
    assert torch.all(x == x_ceil), f"{name=} {x=} {x_ceil=}"


def aiter_w8a8_block_fp8_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    block_size: List[int],
    weight_scale: torch.Tensor,
    input_scale: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    assert input_scale is None
    input_2d = input.view(-1, input.shape[-1])
    output_shape = [*input.shape[:-1], weight.shape[0]]

    q_input, x_scale = aiter_per1x128_quant(input_2d, quant_dtype=aiter.dtypes.fp8)
    output = gemm_a8w8_blockscale(
        q_input, weight, x_scale, weight_scale, dtype=input.dtype
    )

    if bias is not None:
        output += bias

    return output.to(dtype=input_2d.dtype).view(*output_shape)


def triton_w8a8_block_fp8_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    block_size: List[int],
    weight_scale: torch.Tensor,
    input_scale: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    assert input_scale is None
    input_2d = input.view(-1, input.shape[-1])
    output_shape = [*input.shape[:-1], weight.shape[0]]

    q_input, x_scale = per_token_group_quant_fp8(
        input_2d, block_size[1], column_major_scales=False
    )
    output = w8a8_block_fp8_matmul_triton(
        q_input, weight, x_scale, weight_scale, block_size, output_dtype=input_2d.dtype
    )
    if bias is not None:
        output += bias
    return output.to(dtype=input_2d.dtype).view(*output_shape)


def input_to_float8(
    x: torch.Tensor, dtype: torch.dtype = fp8_dtype
) -> Tuple[torch.Tensor, torch.Tensor]:
    """This function quantizes input values to float8 values with tensor-wise quantization."""
    min_val, max_val = x.aminmax()
    amax = torch.maximum(min_val.abs(), max_val.abs()).float().clamp(min=1e-12)

    if _is_fp8_fnuz:
        dtype = fp8_dtype
        fp_max = fp8_max
    else:
        finfo = torch.finfo(dtype)
        fp_max = finfo.max

    scale = fp_max / amax
    x_scl_sat = (x.float() * scale).clamp(min=-fp_max, max=fp_max)
    return x_scl_sat.to(dtype).contiguous(), scale.float().reciprocal()


def block_quant_to_tensor_quant(
    x_q_block: torch.Tensor,
    x_s: torch.Tensor,
    block_size: List[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """This function converts block-wise quantization to tensor-wise quantization.
    The inputs are block-wise quantization tensor `x_q_block`, block-wise quantization scale
    and the block size.
    The outputs are tensor-wise quantization tensor and tensor-wise quantization scale.
    Note only float8 is supported for now.
    """
    block_n, block_k = block_size[0], block_size[1]
    n, k = x_q_block.shape
    n_tiles = (n + block_n - 1) // block_n
    k_tiles = (k + block_k - 1) // block_k
    assert n_tiles == x_s.shape[0]
    assert k_tiles == x_s.shape[1]

    x_dq_block = x_q_block.to(torch.float32)

    x_dq_block_tiles = [
        [
            x_dq_block[
                j * block_n : min((j + 1) * block_n, n),
                i * block_k : min((i + 1) * block_k, k),
            ]
            for i in range(k_tiles)
        ]
        for j in range(n_tiles)
    ]

    for i in range(k_tiles):
        for j in range(n_tiles):
            x_dq_block_tiles[j][i][:, :] = x_dq_block_tiles[j][i] * x_s[j][i]

    x_q_tensor, scale = (
        scaled_fp8_quant(x_dq_block)
        if _is_cuda
        else input_to_float8(x_dq_block, dtype=x_q_block.dtype)
    )
    return x_q_tensor, scale


def block_quant_dequant(
    x_q_block: torch.Tensor,
    x_s: torch.Tensor,
    block_size: List[int],
    dtype: torch.dtype,
) -> torch.Tensor:
    """This function converts block-wise quantization to unquantized.
    The inputs are block-wise quantization tensor `x_q_block`, block-wise quantization scale
    and the block size.
    The output is an unquantized tensor with dtype.
    """
    block_n, block_k = block_size[0], block_size[1]
    *_, n, k = x_q_block.shape

    # ... n_scale k_scale -> ... (n_scale block_n) (k_scale block_k)
    x_scale_repeat = x_s.repeat_interleave(block_n, dim=-2).repeat_interleave(
        block_k, dim=-1
    )
    x_scale_repeat = x_scale_repeat[..., :n, :k]

    return (x_q_block.to(torch.float32) * x_scale_repeat).to(dtype)


def requant_weight_ue8m0_inplace(weight, weight_scale_inv, weight_block_size):
    assert isinstance(weight, torch.nn.Parameter)
    assert isinstance(weight_scale_inv, torch.nn.Parameter)
    weight.data, weight_scale_inv.data = _requant_weight_ue8m0(
        weight, weight_scale_inv, weight_block_size
    )


def _requant_weight_ue8m0(
    weight: torch.Tensor,
    weight_scale_inv: torch.Tensor,
    weight_block_size: List[int],
):
    assert weight_block_size == [128, 128]

    *_, n, k = weight.shape

    weight_dequant = block_quant_dequant(
        weight,
        weight_scale_inv,
        weight_block_size,
        torch.bfloat16,
    )

    weight_dequant_flat = weight_dequant.view((-1, k))
    out_w_flat, out_s_flat = per_block_cast_to_fp8(weight_dequant_flat)

    out_w = out_w_flat.view(weight.shape)
    out_s = out_s_flat.view(weight_scale_inv.shape)

    # NOTE copy and modified from DeepGEMM
    def _transform_scale(sf, mn: int):
        import deep_gemm.utils.layout

        sf = sf.index_select(-2, torch.arange(mn, device=sf.device) // 128)
        sf = deep_gemm.utils.layout.get_col_major_tma_aligned_packed_tensor(sf)
        return sf

    out_s = _transform_scale(out_s, mn=out_w.shape[-2])

    return out_w, out_s


# COPIED FROM DeepGEMM
def per_block_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    m, n = x.shape
    x_padded = torch.zeros(
        (align(m, 128), align(n, 128)), dtype=x.dtype, device=x.device
    )
    x_padded[:m, :n] = x
    x_view = x_padded.view(-1, 128, x_padded.size(1) // 128, 128)
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    sf = ceil_to_ue8m0(x_amax / 448.0)
    x_scaled = (x_view * (1.0 / sf)).to(torch.float8_e4m3fn)
    return x_scaled.view_as(x_padded)[:m, :n].contiguous(), sf.view(
        x_view.size(0), x_view.size(2)
    )


# COPIED FROM DeepGEMM
def ceil_to_ue8m0(x: torch.Tensor):
    return torch.pow(2.0, torch.ceil(torch.log2(x.abs())))


def channel_quant_to_tensor_quant(
    x_q_channel: torch.Tensor,
    x_s: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    x_dq_channel = x_q_channel.to(torch.float32) * x_s
    x_q_tensor, scale = (
        scaled_fp8_quant(x_dq_channel)
        if _is_cuda
        else input_to_float8(x_dq_channel, dtype=x_q_channel.dtype)
    )
    return x_q_tensor, scale


def _process_scaled_mm_output(output, input_2d_shape, output_shape):
    if type(output) is tuple and len(output) == 2:
        output = output[0]
    return torch.narrow(output, 0, 0, input_2d_shape[0]).view(*output_shape)


def _apply_fallback_scaled_mm(
    qinput: torch.Tensor,
    weight: torch.Tensor,
    x_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    input_2d_shape: Tuple[int, int],
    output_shape: List[int],
    bias: Optional[torch.Tensor],
    input_dtype: torch.dtype,
) -> torch.Tensor:
    """
    A corrected fallback for FP8 linear layers. It dequantizes the inputs
    to the specified input_dtype and then performs a standard matrix multiplication.
    This is numerically correct but less performant than dedicated kernels.
    """
    # Dequantize the activation tensor (qinput).
    # qinput is shape [tokens, in_features], x_scale is [tokens, 1].
    dequant_input = qinput.to(input_dtype) * x_scale

    # Dequantize the weight tensor.
    # weight is shape [in_features, out_features], weight_scale is [out_features, 1].
    # We need to broadcast weight_scale correctly.
    dequant_weight = weight.to(input_dtype) * weight_scale.t()

    # Perform the standard matrix multiplication.
    output = torch.matmul(dequant_input, dequant_weight)

    # Add bias if it exists.
    if bias is not None:
        output += bias

    # Reshape the output to match the original input's dimensions.
    return output.view(*output_shape)


def apply_fp8_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    input_scale: Optional[torch.Tensor] = None,
    input_scale_ub: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    cutlass_fp8_supported: bool = cutlass_fp8_supported(),
    use_per_token_if_dynamic: bool = False,
    pad_output: Optional[bool] = None,
    compressed_tensor_quant: bool = False,
) -> torch.Tensor:
    # --- START DEBUGGING (INPUTS) ---
    # print(f"\n--- DEBUG: fp8_utils.py | apply_fp8_linear (INPUTS) ---")
    # print(f"Tensor 'input': shape={input.shape}, dtype={input.dtype}, has_nan={torch.isnan(input).any()}, has_inf={torch.isinf(input).any()}")

    # CORRECTED DEBUGGING FOR FP8 TENSORS
    # We must cast to float32 before checking for nan/inf
    # print(f"Tensor 'weight': shape={weight.shape}, dtype={weight.dtype}, has_nan={torch.isnan(weight.to(torch.float32)).any()}, has_inf={torch.isinf(weight.to(torch.float32)).any()}")

    # print(f"Tensor 'weight_scale': shape={weight_scale.shape}, dtype={weight_scale.dtype}, has_nan={torch.isnan(weight_scale).any()}, has_inf={torch.isinf(weight_scale).any()}")
    # if input_scale_ub is not None:
    # print(f"Tensor 'input_scale_ub': shape={input_scale_ub.shape}, dtype={input_scale_ub.dtype}, value={input_scale_ub.item()}, has_nan={torch.isnan(input_scale_ub).any()}, has_inf={torch.isinf(input_scale_ub).any()}")
    # --- END DEBUGGING ---

    if pad_output is None:
        pad_output = not get_bool_env_var("SGLANG_ENABLE_TORCH_COMPILE")

    input_2d = input.view(-1, input.shape[-1])
    output_shape = [*input.shape[:-1], weight.shape[1]]

    if input_scale_ub is not None:
        qinput, x_scale = static_quant_fp8(
            input_2d, input_scale_ub, repeat_scale=cutlass_fp8_supported
        )
    elif input_scale is not None:
        qinput, x_scale = static_quant_fp8(
            input_2d, input_scale, repeat_scale=cutlass_fp8_supported
        )
    else:
        if _is_cuda:
            qinput, x_scale = sglang_per_token_quant_fp8(input_2d)
        else:
            if _is_hip and weight_scale.numel() == 1:
                qinput, x_scale = ops.scaled_fp8_quant(
                    input_2d,
                    None,
                    use_per_token_if_dynamic=use_per_token_if_dynamic,
                )
            else:
                qinput, x_scale = per_token_group_quant_fp8(
                    input_2d, group_size=input_2d.shape[1]
                )

    is_per_channel = weight_scale.numel() == weight.shape[1]

    output = None
    if cutlass_fp8_supported and is_per_channel:
        try:
            if VLLM_AVAILABLE and use_vllm_cutlass_w8a8_fp8_kernel:
                output = ops.cutlass_scaled_mm(
                    qinput,
                    weight,
                    out_dtype=input.dtype,
                    scale_a=x_scale,
                    scale_b=weight_scale.view(-1),
                    bias=bias,
                )
            else:
                output = fp8_scaled_mm(
                    qinput,
                    weight,
                    x_scale,
                    weight_scale.view(-1),
                    out_dtype=input.dtype,
                    bias=bias,
                )
            output = output.view(*output_shape)
        except (ImportError, NameError, AttributeError):
            pass

    if output is None:
        per_tensor_weights = weight_scale.numel() == 1
        per_tensor_activations = x_scale.numel() == 1

        if per_tensor_weights and per_tensor_activations:
            output = torch._scaled_mm(
                qinput,
                weight,
                out_dtype=input.dtype,
                scale_a=x_scale,
                scale_b=weight_scale,
                bias=bias,
            )
            output = _process_scaled_mm_output(output, input_2d.shape, output_shape)
        else:
            output = _apply_fallback_scaled_mm(
                qinput,
                weight,
                x_scale,
                weight_scale,
                input_2d.shape,
                output_shape,
                bias,
                input.dtype,
            )

    # --- START DEBUGGING (OUTPUT) ---
    # print(f"\n--- DEBUG: fp8_utils.py | apply_fp8_linear (OUTPUT) ---")
    # print(f"Tensor 'output': shape={output.shape}, dtype={output.dtype}, has_nan={torch.isnan(output).any()}, has_inf={torch.isinf(output).any()}")
    # --- END DEBUGGING ---

    return output
