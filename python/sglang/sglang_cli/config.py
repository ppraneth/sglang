# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""The arguments of the server."""
import dataclasses
import json
import logging
import os
import random
import socket
import sys
import tempfile
from enum import Enum
from typing import List, Literal, Optional, Self, Union

import pydantic

from sglang.srt.connector import ConnectorType
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.hf_transformers_utils import check_gguf_file, get_config
from sglang.srt.lora.lora_registry import LoRARef
from sglang.srt.parser.reasoning_parser import ReasoningParser
from sglang.srt.utils import (
    LORA_TARGET_ALL_MODULES,
    SUPPORTED_LORA_TARGET_MODULES,
    configure_ipv6,
    get_device,
    get_device_memory_capacity,
    is_cuda,
    is_flashinfer_available,
    is_hip,
    is_npu,
    is_port_available,
    is_remote_url,
    is_sm90_supported,
    is_sm100_supported,
    is_triton_kernels_available,
    is_valid_ipv6_address,
    json_list_type,
    nullable_str,
    parse_connector_type,
)
from sglang.utils import is_in_ci

logger = logging.getLogger(__name__)


## Define Enums for all choice lists
class LoadFormat(str, Enum):
    AUTO = "auto"
    PT = "pt"
    SAFETENSORS = "safetensors"
    NPCACHE = "npcache"
    DUMMY = "dummy"
    SHARDED_STATE = "sharded_state"
    GGUF = "gguf"
    BITSANDBYTES = "bitsandbytes"
    LAYERED = "layered"
    REMOTE = "remote"
    REMOTE_INSTANCE = "remote_instance"


class QuantizationMethod(str, Enum):
    AWQ = "awq"
    FP8 = "fp8"
    GPTQ = "gptq"
    MARLIN = "marlin"
    GPTQ_MARLIN = "gptq_marlin"
    AWQ_MARLIN = "awq_marlin"
    BITSANDBYTES = "bitsandbytes"
    GGUF = "gguf"
    MODELOPT = "modelopt"
    MODELOPT_FP4 = "modelopt_fp4"
    PETIT_NVFP4 = "petit_nvfp4"
    W8A8_INT8 = "w8a8_int8"
    W8A8_FP8 = "w8a8_fp8"
    MOE_WNA16 = "moe_wna16"
    QOQ = "qoq"
    W4AFP8 = "w4afp8"
    MXFP4 = "mxfp4"


class AttentionBackend(str, Enum):
    TRITON = "triton"
    TORCH_NATIVE = "torch_native"
    FLEX_ATTENTION = "flex_attention"
    CUTLASS_MLA = "cutlass_mla"
    FA3 = "fa3"
    FA4 = "fa4"
    FLASHINFER = "flashinfer"
    FLASHMLA = "flashmla"
    TRTLLM_MLA = "trtllm_mla"
    TRTLLM_MHA = "trtllm_mha"
    DUAL_CHUNK_FLASH_ATTN = "dual_chunk_flash_attn"
    HYBRID_LINEAR_ATTN = "hybrid_linear_attn"
    AITER = "aiter"
    WAVE = "wave"
    INTEL_AMX = "intel_amx"
    ASCEND = "ascend"


class LoRABackend(str, Enum):
    TRITON = "triton"
    CSGMV = "csgmv"


class DisaggTransferBackend(str, Enum):
    MOONCAKE = "mooncake"
    NIXL = "nixl"
    ASCEND = "ascend"
    FAKE = "fake"


class GrammarBackend(str, Enum):
    XGRAMMAR = "xgrammar"
    OUTLINES = "outlines"
    LLGUIDANCE = "llguidance"
    NONE = "none"


class DeterministicAttentionBackend(str, Enum):
    FLASHINFER = "flashinfer"
    FA3 = "fa3"


class MoeA2ABackend(str, Enum):
    NONE = "none"
    DEEPEP = "deepep"


class MoeRunnerBackend(str, Enum):
    AUTO = "auto"
    TRITON = "triton"
    TRITON_KERNEL = "triton_kernel"
    FLASHINFER_TRTLLM = "flashinfer_trtllm"
    FLASHINFER_CUTLASS = "flashinfer_cutlass"
    FLASHINFER_MXFP4 = "flashinfer_mxfp4"


class FlashinferMxfp4MoePrecision(str, Enum):
    DEFAULT = "default"
    BF16 = "bf16"


class DeepEPMode(str, Enum):
    AUTO = "auto"
    NORMAL = "normal"
    LOW_LATENCY = "low_latency"


class EPDispatchAlgorithm(str, Enum):
    STATIC = "static"
    DYNAMIC = "dynamic"
    FAKE = "fake"


class ExpertDistributionRecorderMode(str, Enum):
    STAT = "stat"
    STAT_APPROX = "stat_approx"
    PER_PASS = "per_pass"
    PER_TOKEN = "per_token"


class DisaggregationMode(str, Enum):
    NULL = "null"
    PREFILL = "prefill"
    DECODE = "decode"


# Define constants
LOAD_FORMAT_CHOICES = [format.value for format in LoadFormat]
QUANTIZATION_CHOICES = [method.value for method in QuantizationMethod]
ATTENTION_BACKEND_CHOICES = [backend.value for backend in AttentionBackend]
LORA_BACKEND_CHOICES = [backend.value for backend in LoRABackend]
DISAGG_TRANSFER_BACKEND_CHOICES = [backend.value for backend in DisaggTransferBackend]
GRAMMAR_BACKEND_CHOICES = [backend.value for backend in GrammarBackend]
DETERMINISTIC_ATTENTION_BACKEND_CHOICES = [
    backend.value for backend in DeterministicAttentionBackend
]


# Allow external code to add more choices
def add_load_format_choices(choices):
    global LOAD_FORMAT_CHOICES
    LOAD_FORMAT_CHOICES.extend(choices)
    # Note: Dynamically extending an Enum is not supported for static type checking.
    # If new choices are added, the LoadFormat Enum must be updated manually.


def add_quantization_method_choices(choices):
    global QUANTIZATION_CHOICES
    QUANTIZATION_CHOICES.extend(choices)
    # Note: Dynamically extending an Enum is not supported for static type checking.
    # If new choices are added, the QuantizationMethod Enum must be updated manually.


def add_attention_backend_choices(choices):
    global ATTENTION_BACKEND_CHOICES
    ATTENTION_BACKEND_CHOICES.extend(choices)
    # Note: Dynamically extending an Enum is not supported for static type checking.
    # If new choices are added, the AttentionBackend Enum must be updated manually.


def add_disagg_transfer_backend_choices(choices):
    global DISAGG_TRANSFER_BACKEND_CHOICES
    DISAGG_TRANSFER_BACKEND_CHOICES.extend(choices)
    # Note: Dynamically extending an Enum is not supported for static type checking.
    # If new choices are added, the DisaggTransferBackend Enum must be updated manually.


def add_grammar_backend_choices(choices):
    global GRAMMAR_BACKEND_CHOICES
    GRAMMAR_BACKEND_CHOICES.extend(choices)
    # Note: Dynamically extending an Enum is not supported for static type checking.
    # If new choices are added, the GrammarBackend Enum must be updated manually.


class ServerConfig(pydantic.BaseModel):
    # Model and tokenizer
    model_path: str = pydantic.Field(
        ...,
        help="The path of the model weights. This can be a local folder or a Hugging Face repo ID.",
    )
    tokenizer_path: Optional[str] = pydantic.Field(
        default=None, help="The path of the tokenizer."
    )
    tokenizer_mode: str = pydantic.Field(
        default="auto",
        help="Tokenizer mode. 'auto' will use the fast tokenizer if available, and 'slow' will always use the slow tokenizer.",
    )
    tokenizer_worker_num: int = pydantic.Field(
        default=1, help="The worker num of the tokenizer manager."
    )
    skip_tokenizer_init: bool = pydantic.Field(
        default=False,
        help="If set, skip init tokenizer and pass input_ids in generate request.",
    )
    load_format: Literal[
        LoadFormat.AUTO,
        LoadFormat.PT,
        LoadFormat.SAFETENSORS,
        LoadFormat.NPCACHE,
        LoadFormat.DUMMY,
        LoadFormat.SHARDED_STATE,
        LoadFormat.GGUF,
        LoadFormat.BITSANDBYTES,
        LoadFormat.LAYERED,
        LoadFormat.REMOTE,
        LoadFormat.REMOTE_INSTANCE,
    ] = pydantic.Field(
        default=LoadFormat.AUTO, help="The format of the model weights to load."
    )
    model_loader_extra_config: str = pydantic.Field(
        default="{}",
        help="Extra config for model loader. This will be passed to the model loader corresponding to the chosen load_format.",
    )
    trust_remote_code: bool = pydantic.Field(
        default=False, help="Allow custom models defined on the Hub."
    )
    context_length: Optional[int] = pydantic.Field(
        default=None,
        help="The model's maximum context length. Defaults to None (will use the value from the model's config.json instead).",
    )
    is_embedding: bool = pydantic.Field(
        default=False, help="Enable for embedding models only"
    )
    enable_multimodal: Optional[bool] = pydantic.Field(
        default=None,
        help="Enable the multimodal functionality for the served model. If the model being served is not multimodal, nothing will happen",
    )
    revision: Optional[str] = pydantic.Field(
        default=None,
        help="The specific model version to use. It can be a branch name, a tag name, or a commit id. If unspecified, will use the default version.",
    )
    model_impl: str = pydantic.Field(
        default="auto",
        help='Which implementation of the model to use. "auto" will try to use the SGLang implementation if it exists and fall back to the Transformers implementation if no SGLang implementation is available.',
    )

    # HTTP server
    host: str = pydantic.Field(default="127.0.0.1", help="The host of the HTTP server.")
    port: int = pydantic.Field(default=30000, help="The port of the HTTP server.")
    skip_server_warmup: bool = pydantic.Field(
        default=False, help="If set, skip warmup."
    )
    warmups: Optional[str] = pydantic.Field(
        default=None,
        help="Specify custom warmup functions to run before server starts.",
    )
    nccl_port: Optional[int] = pydantic.Field(
        default=None,
        help="The port for NCCL distributed environment setup. Defaults to a random port.",
    )

    # Quantization and data type
    dtype: str = pydantic.Field(
        default="auto",
        help='Data type for model weights and activations. "auto" will use FP16 precision for FP32 and FP16 models, and BF16 precision for BF16 models.',
    )
    quantization: Optional[
        Literal[
            QuantizationMethod.AWQ,
            QuantizationMethod.FP8,
            QuantizationMethod.GPTQ,
            QuantizationMethod.MARLIN,
            QuantizationMethod.GPTQ_MARLIN,
            QuantizationMethod.AWQ_MARLIN,
            QuantizationMethod.BITSANDBYTES,
            QuantizationMethod.GGUF,
            QuantizationMethod.MODELOPT,
            QuantizationMethod.MODELOPT_FP4,
            QuantizationMethod.PETIT_NVFP4,
            QuantizationMethod.W8A8_INT8,
            QuantizationMethod.W8A8_FP8,
            QuantizationMethod.MOE_WNA16,
            QuantizationMethod.QOQ,
            QuantizationMethod.W4AFP8,
            QuantizationMethod.MXFP4,
        ]
    ] = pydantic.Field(
        default=None, help="The quantization method to use for the model weights."
    )
    quantization_param_path: Optional[str] = pydantic.Field(
        default=None,
        help="Path to the JSON file containing the KV cache scaling factors. This should generally be supplied, when KV cache dtype is FP8.",
    )
    kv_cache_dtype: str = pydantic.Field(
        default="auto",
        help='Data type for kv cache storage. "auto" will use model data type. "fp8_e5m2" and "fp8_e4m3" is supported for CUDA 11.8+.',
    )

    # Memory and scheduling
    mem_fraction_static: Optional[float] = pydantic.Field(
        default=None,
        help="The fraction of the memory used for static allocation (model weights and KV cache memory pool). Use a smaller value if you see out-of-memory errors.",
    )
    max_running_requests: Optional[int] = pydantic.Field(
        default=None, help="The maximum number of running requests."
    )
    max_queued_requests: Optional[int] = pydantic.Field(
        default=None,
        help="The maximum number of queued requests. This option is ignored when using disaggregation-mode.",
    )
    max_total_tokens: Optional[int] = pydantic.Field(
        default=None,
        help="The maximum number of tokens in the memory pool. If not specified, it will be automatically calculated based on the memory usage fraction.",
    )
    chunked_prefill_size: Optional[int] = pydantic.Field(
        default=None,
        help="The maximum number of tokens in a chunk for the chunked prefill. Setting this to -1 means disabling chunked prefill.",
    )
    max_prefill_tokens: int = pydantic.Field(
        default=16384,
        help="The maximum number of tokens in a prefill batch. The real bound will be the maximum of this value and the model's maximum context length.",
    )
    schedule_policy: str = pydantic.Field(
        default="fcfs", help="The scheduling policy of the requests."
    )
    enable_priority_scheduling: bool = pydantic.Field(
        default=False,
        help="Enable priority scheduling. Requests with higher priority integer values will be scheduled first by default.",
    )
    schedule_low_priority_values_first: bool = pydantic.Field(
        default=False,
        help="If specified with --enable-priority-scheduling, the scheduler will schedule requests with lower priority integer values first.",
    )
    priority_scheduling_preemption_threshold: int = pydantic.Field(
        default=10,
        help="Minimum difference in priorities for an incoming request to have to preempt running request(s).",
    )
    schedule_conservativeness: float = pydantic.Field(
        default=1.0,
        help="How conservative the schedule policy is. A larger value means more conservative scheduling. Use a larger value if you see requests being retracted frequently.",
    )
    page_size: Optional[int] = pydantic.Field(
        default=None, help="The number of tokens in a page."
    )
    hybrid_kvcache_ratio: Optional[float] = pydantic.Field(
        default=None,
        help="Mix ratio in [0,1] between uniform and hybrid kv buffers.",
    )
    swa_full_tokens_ratio: float = pydantic.Field(
        default=0.8,
        help="The ratio of SWA layer KV tokens / full layer KV tokens, regardless of the number of swa:full layers. It should be between 0 and 1.",
    )
    disable_hybrid_swa_memory: bool = pydantic.Field(
        default=False, help="Disable the hybrid SWA memory."
    )
    radix_eviction_policy: str = pydantic.Field(
        default="lru",
        help="The eviction policy of radix trees. 'lru' stands for Least Recently Used, 'lfu' stands for Least Frequently Used.",
    )
    # Runtime options
    device: Optional[str] = pydantic.Field(
        default=None,
        help="The device to use ('cuda', 'xpu', 'hpu', 'npu', 'cpu'). Defaults to auto-detection if not specified.",
    )
    tp_size: int = pydantic.Field(default=1, help="The tensor parallelism size.")
    pp_size: int = pydantic.Field(default=1, help="The pipeline parallelism size.")
    max_micro_batch_size: Optional[int] = pydantic.Field(
        default=None, help="The maximum micro batch size in pipeline parallelism."
    )
    stream_interval: int = pydantic.Field(
        default=1,
        help="The interval (or buffer size) for streaming in terms of the token length. A smaller value makes streaming smoother, while a larger value makes the throughput higher",
    )
    stream_output: bool = pydantic.Field(
        default=False, help="Whether to output as a sequence of disjoint segments."
    )
    random_seed: Optional[int] = pydantic.Field(default=None, help="The random seed.")
    constrained_json_whitespace_pattern: Optional[str] = pydantic.Field(
        default=None,
        help="(outlines backend only) Regex pattern for syntactic whitespaces allowed in JSON constrained output.",
    )
    watchdog_timeout: float = pydantic.Field(
        default=300,
        help="Set watchdog timeout in seconds. If a forward batch takes longer than this, the server will crash to prevent hanging.",
    )
    dist_timeout: Optional[int] = pydantic.Field(
        default=None, help="Set timeout for torch.distributed initialization."
    )
    download_dir: Optional[str] = pydantic.Field(
        default=None, help="Model download directory for huggingface."
    )
    base_gpu_id: int = pydantic.Field(
        default=0,
        help="The base GPU ID to start allocating GPUs from. Useful when running multiple instances on the same machine.",
    )
    gpu_id_step: int = pydantic.Field(
        default=1,
        help="The delta between consecutive GPU IDs that are used. For example, setting it to 2 will use GPU 0,2,4,...",
    )
    sleep_on_idle: bool = pydantic.Field(
        default=False, help="Reduce CPU usage when sglang is idle."
    )

    # Logging
    log_level: str = pydantic.Field(
        default="info", help="The logging level of all loggers."
    )
    log_level_http: Optional[str] = pydantic.Field(
        default=None,
        help="The logging level of HTTP server. If not set, reuse --log-level by default.",
    )
    log_requests: bool = pydantic.Field(
        default=False,
        help="Log metadata, inputs, outputs of all requests. The verbosity is decided by --log-requests-level",
    )
    log_requests_level: int = pydantic.Field(
        default=2,
        help="0: Log metadata (no sampling parameters). 1: Log metadata and sampling parameters. 2: Log metadata, sampling parameters and partial input/output. 3: Log every input/output.",
    )
    crash_dump_folder: Optional[str] = pydantic.Field(
        default=None,
        help="Folder path to dump requests from the last 5 min before a crash (if any). If not specified, crash dumping is disabled.",
    )
    show_time_cost: bool = pydantic.Field(
        default=False, help="Show time cost of custom marks."
    )
    enable_metrics: bool = pydantic.Field(
        default=False, help="Enable log prometheus metrics."
    )
    enable_metrics_for_all_schedulers: bool = pydantic.Field(
        default=False,
        help="Enable schedulers on all TP ranks (not just TP 0) to record request metrics separately.",
    )
    tokenizer_metrics_custom_labels_header: str = pydantic.Field(
        default="x-customer-labels",
        help="Specify the HTTP header for passing customer labels for tokenizer metrics.",
    )
    tokenizer_metrics_allowed_customer_labels: Optional[List[str]] = pydantic.Field(
        default=None, help="The customer labels allowed for tokenizer metrics."
    )
    bucket_time_to_first_token: Optional[List[float]] = pydantic.Field(
        default=None,
        help="The buckets of time to first token, specified as a list of floats.",
    )
    bucket_inter_token_latency: Optional[List[float]] = pydantic.Field(
        default=None,
        help="The buckets of inter-token latency, specified as a list of floats.",
    )
    bucket_e2e_request_latency: Optional[List[float]] = pydantic.Field(
        default=None,
        help="The buckets of end-to-end request latency, specified as a list of floats.",
    )
    collect_tokens_histogram: bool = pydantic.Field(
        default=False, help="Collect prompt/generation tokens histogram."
    )
    prompt_tokens_buckets: Optional[List[str]] = pydantic.Field(
        default=None, help="The buckets rule of prompt tokens."
    )
    generation_tokens_buckets: Optional[List[str]] = pydantic.Field(
        default=None, help="The buckets rule for generation tokens histogram."
    )
    decode_log_interval: int = pydantic.Field(
        default=40, help="The log interval of decode batch."
    )
    enable_request_time_stats_logging: bool = pydantic.Field(
        default=False, help="Enable per request time stats logging"
    )
    kv_events_config: Optional[str] = pydantic.Field(
        default=None,
        help="Config in json format for NVIDIA dynamo KV event publishing. Publishing will be enabled if this flag is used.",
    )
    gc_warning_threshold_secs: float = pydantic.Field(
        default=0.0,
        help="The threshold for long GC warning. If a GC takes longer than this, a warning will be logged. Set to 0 to disable.",
    )
    enable_trace: bool = pydantic.Field(
        default=False, help="Enable opentelemetry trace"
    )
    oltp_traces_endpoint: str = pydantic.Field(
        default="localhost:4317",
        help="Config opentelemetry collector endpoint if --enable-trace is set. format: <ip>:<port>",
    )

    # API related
    api_key: Optional[str] = pydantic.Field(
        default=None,
        help="Set API key of the server. It is also used in the OpenAI API compatible server.",
    )
    served_model_name: Optional[str] = pydantic.Field(
        default=None,
        help="Override the model name returned by the v1/models endpoint in OpenAI API server.",
    )
    weight_version: str = pydantic.Field(
        default="default",
        help="Version identifier for the model weights. Defaults to 'default' if not specified.",
    )
    chat_template: Optional[str] = pydantic.Field(
        default=None,
        help="The buliltin chat template name or the path of the chat template file. This is only used for OpenAI-compatible API server.",
    )
    completion_template: Optional[str] = pydantic.Field(
        default=None,
        help="The buliltin completion template name or the path of the completion template file. This is only used for OpenAI-compatible API server. only for code completion currently.",
    )
    file_storage_path: str = pydantic.Field(
        default="sglang_storage", help="The path of the file storage in backend."
    )
    enable_cache_report: bool = pydantic.Field(
        default=False,
        help="Return number of cached tokens in usage.prompt_tokens_details for each openai request.",
    )
    reasoning_parser: Optional[str] = pydantic.Field(
        default=None, help="Specify the parser for reasoning models."
    )
    tool_call_parser: Optional[str] = pydantic.Field(
        default=None, help="Specify the parser for handling tool-call interactions."
    )
    tool_server: Optional[str] = pydantic.Field(
        default=None,
        help="Either 'demo' or a comma-separated list of tool server urls to use for the model. If not specified, no tool server will be used.",
    )

    # Data parallelism
    dp_size: int = pydantic.Field(default=1, help="The data parallelism size.")
    load_balance_method: str = pydantic.Field(
        default="round_robin",
        help="The load balancing strategy for data parallelism.",
    )
    load_watch_interval: float = pydantic.Field(
        default=0.1, help="The interval of load watching in seconds."
    )
    # FIXME: remove this after dp rank scheduling is fully supported with PD-Disaggregation
    prefill_round_robin_balance: bool = pydantic.Field(
        default=False,
        help="Prefill is round robin balanced. This is used to promise decode server can get the correct dp rank.",
    )

    # Multi-node distributed serving
    dist_init_addr: Optional[str] = pydantic.Field(
        default=None,
        help="The host address for initializing distributed backend (e.g., `192.168.0.2:25000`).",
    )
    nnodes: int = pydantic.Field(default=1, help="The number of nodes.")
    node_rank: int = pydantic.Field(default=0, help="The node rank.")

    # Model override args in JSON
    json_model_override_args: str = pydantic.Field(
        default="{}",
        help="A dictionary in JSON string format used to override default model configurations.",
    )
    preferred_sampling_params: Optional[str] = pydantic.Field(
        default=None,
        help="json-formatted sampling settings that will be returned in /get_model_info",
    )

    # LoRA
    enable_lora: Optional[bool] = pydantic.Field(
        default=None, help="Enable LoRA support for the model."
    )
    max_lora_rank: Optional[int] = pydantic.Field(
        default=None,
        help="The maximum rank of LoRA adapters. If not specified, it will be automatically inferred from the adapters.",
    )
    lora_target_modules: Optional[Union[set[str], List[str]]] = pydantic.Field(
        default=None,
        help="The union set of all target modules where LoRA should be applied.",
    )
    lora_paths: Optional[
        Union[dict[str, str], List[dict[str, str]], List[str], List[LoRARef]]
    ] = pydantic.Field(default=None, help="The list of LoRA adapters to load.")
    max_loaded_loras: Optional[int] = pydantic.Field(
        default=None,
        help="If specified, it limits the maximum number of LoRA adapters loaded in CPU memory at a time.",
    )
    max_loras_per_batch: int = pydantic.Field(
        default=8,
        help="Maximum number of adapters for a running batch, include base-only request.",
    )
    lora_backend: Literal[LoRABackend.TRITON, LoRABackend.CSGMV] = pydantic.Field(
        default=LoRABackend.TRITON,
        help="The backend to use for LoRA computation. Options are 'triton' or 'csgmv'.",
    )
    max_lora_chunk_size: Optional[int] = pydantic.Field(
        default=16,
        help="Maximum chunk size for the ChunkedSGMV LoRA backend.",
    )

    # Kernel backend
    attention_backend: Optional[
        Literal[
            AttentionBackend.TRITON,
            AttentionBackend.TORCH_NATIVE,
            AttentionBackend.FLEX_ATTENTION,
            AttentionBackend.CUTLASS_MLA,
            AttentionBackend.FA3,
            AttentionBackend.FA4,
            AttentionBackend.FLASHINFER,
            AttentionBackend.FLASHMLA,
            AttentionBackend.TRTLLM_MLA,
            AttentionBackend.TRTLLM_MHA,
            AttentionBackend.DUAL_CHUNK_FLASH_ATTN,
            AttentionBackend.HYBRID_LINEAR_ATTN,
            AttentionBackend.AITER,
            AttentionBackend.WAVE,
            AttentionBackend.INTEL_AMX,
            AttentionBackend.ASCEND,
        ]
    ] = pydantic.Field(
        default=None,
        help="Choose the kernels for attention layers. If None, a default backend is selected based on hardware and model requirements.",
    )
    decode_attention_backend: Optional[str] = pydantic.Field(
        default=None,
        help="Choose the kernels for decode attention layers (have priority over --attention-backend).",
    )
    prefill_attention_backend: Optional[str] = pydantic.Field(
        default=None,
        help="Choose the kernels for prefill attention layers (have priority over --attention-backend).",
    )
    sampling_backend: Optional[str] = pydantic.Field(
        default=None, help="Choose the kernels for sampling layers."
    )
    grammar_backend: Optional[
        Literal[
            GrammarBackend.XGRAMMAR,
            GrammarBackend.LLGUIDANCE,
            GrammarBackend.OUTLINES,
            GrammarBackend.NONE,
        ]
    ] = pydantic.Field(
        default=None,
        help="The backend to use for grammar-constrained generation. If None, no grammar constraints are applied.",
    )
    mm_attention_backend: Optional[str] = pydantic.Field(
        default=None, help="Set multimodal attention backend."
    )

    # Speculative decoding
    speculative_algorithm: Optional[str] = pydantic.Field(
        default=None, help="Speculative algorithm."
    )
    speculative_draft_model_path: Optional[str] = pydantic.Field(
        default=None,
        help="The path of the draft model weights. This can be a local folder or a Hugging Face repo ID.",
    )
    speculative_draft_model_revision: Optional[str] = pydantic.Field(
        default=None, help="The specific draft model version to use."
    )
    speculative_num_steps: Optional[int] = pydantic.Field(
        default=None,
        help="The number of steps sampled from draft model in Speculative Decoding.",
    )
    speculative_eagle_topk: Optional[int] = pydantic.Field(
        default=None,
        help="The number of tokens sampled from the draft model in eagle2 each step.",
    )
    speculative_num_draft_tokens: Optional[int] = pydantic.Field(
        default=None,
        help="The number of tokens sampled from the draft model in Speculative Decoding.",
    )
    speculative_accept_threshold_single: float = pydantic.Field(
        default=1.0,
        help="Accept a draft token if its probability in the target model is greater than this threshold.",
    )
    speculative_accept_threshold_acc: float = pydantic.Field(
        default=1.0,
        help="The accept probability of a draft token is raised from its target probability p to min(1, p / threshold_acc).",
    )
    speculative_token_map: Optional[str] = pydantic.Field(
        default=None, help="The path of the draft model's small vocab table."
    )
    speculative_attention_mode: str = pydantic.Field(
        default="prefill",
        help="Attention backend for speculative decoding operations (both target verify and draft extend).",
    )
    # For lookahead only
    speculative_lookahead_min_match_window_size: int = pydantic.Field(
        default=1,
        help="The minimum window size for pattern matching in lookahead speculative decoding.",
    )
    speculative_lookahead_max_match_window_size: int = pydantic.Field(
        default=12,
        help="The maximum window size for pattern matching in lookahead speculative decoding.",
    )
    speculative_lookahead_min_bfs_breadth: int = pydantic.Field(
        default=1,
        help="The minimum breadth for BFS (Breadth-First Search) in lookahead speculative decoding.",
    )
    speculative_lookahead_max_bfs_breadth: int = pydantic.Field(
        default=10,
        help="The maximum breadth for BFS (Breadth-First Search) in lookahead speculative decoding.",
    )
    speculative_lookahead_match_type: Literal["BFS", "PROB"] = pydantic.Field(
        default="BFS", help="The match type for cache tree."
    )
    speculative_lookahead_branch_length: int = pydantic.Field(
        default=18, help="The branch length for lookahead speculative decoding."
    )
    speculative_lookahead_capacity: int = pydantic.Field(
        default=10 * 1000 * 1000,
        help="The cache capacity for lookahead speculative decoding.",
    )

    # Expert parallelism
    ep_size: int = pydantic.Field(
        default=1, help="The size of the expert parallelism group."
    )
    moe_a2a_backend: Literal[MoeA2ABackend.NONE, MoeA2ABackend.DEEPEP] = pydantic.Field(
        default=MoeA2ABackend.NONE, help="The backend for MoE all-to-all communication."
    )
    moe_runner_backend: Literal[
        MoeRunnerBackend.AUTO,
        MoeRunnerBackend.TRITON,
        MoeRunnerBackend.TRITON_KERNEL,
        MoeRunnerBackend.FLASHINFER_TRTLLM,
        MoeRunnerBackend.FLASHINFER_CUTLASS,
        MoeRunnerBackend.FLASHINFER_MXFP4,
    ] = pydantic.Field(
        default=MoeRunnerBackend.AUTO, help="The backend for MoE runner."
    )
    flashinfer_mxfp4_moe_precision: Literal[
        FlashinferMxfp4MoePrecision.DEFAULT, FlashinferMxfp4MoePrecision.BF16
    ] = pydantic.Field(
        default=FlashinferMxfp4MoePrecision.DEFAULT,
        help="The precision for flashinfer MXFP4 MoE.",
    )
    enable_flashinfer_allreduce_fusion: bool = pydantic.Field(
        default=False, help="Whether to enable flashinfer all-reduce fusion."
    )
    deepep_mode: Literal[DeepEPMode.AUTO, DeepEPMode.NORMAL, DeepEPMode.LOW_LATENCY] = (
        pydantic.Field(default=DeepEPMode.AUTO, help="The mode for DeepEP.")
    )
    ep_num_redundant_experts: int = pydantic.Field(
        default=0, help="Number of redundant experts for expert parallelism."
    )
    ep_dispatch_algorithm: Optional[
        Literal[
            EPDispatchAlgorithm.STATIC,
            EPDispatchAlgorithm.DYNAMIC,
            EPDispatchAlgorithm.FAKE,
        ]
    ] = pydantic.Field(
        default=None, help="The dispatch algorithm for expert parallelism."
    )
    init_expert_location: str = pydantic.Field(
        default="trivial", help="Initial expert location configuration."
    )
    enable_eplb: bool = pydantic.Field(
        default=False, help="Whether to enable expert parallelism load balancing."
    )
    eplb_algorithm: str = pydantic.Field(
        default="auto", help="The algorithm for expert parallelism load balancing."
    )
    eplb_rebalance_num_iterations: int = pydantic.Field(
        default=1000, help="Number of iterations for EPLB rebalancing."
    )
    eplb_rebalance_layers_per_chunk: Optional[int] = pydantic.Field(
        default=None, help="Number of layers per chunk for EPLB rebalancing."
    )
    eplb_min_rebalancing_utilization_threshold: float = pydantic.Field(
        default=1.0, help="Minimum utilization threshold for EPLB rebalancing."
    )
    expert_distribution_recorder_mode: Optional[
        Literal[
            ExpertDistributionRecorderMode.STAT,
            ExpertDistributionRecorderMode.STAT_APPROX,
            ExpertDistributionRecorderMode.PER_PASS,
            ExpertDistributionRecorderMode.PER_TOKEN,
        ]
    ] = pydantic.Field(default=None, help="Mode for expert distribution recorder.")
    expert_distribution_recorder_buffer_size: Optional[int] = pydantic.Field(
        default=None, help="Buffer size for expert distribution recorder."
    )
    enable_expert_distribution_metrics: bool = pydantic.Field(
        default=False, help="Whether to enable expert distribution metrics."
    )
    deepep_config: Optional[str] = pydantic.Field(
        default=None, help="Configuration for DeepEP."
    )
    moe_dense_tp_size: Optional[int] = pydantic.Field(
        default=None, help="Tensor parallelism size for MoE dense layers."
    )

    # Hierarchical cache
    enable_hierarchical_cache: bool = pydantic.Field(
        default=False, help="Enable hierarchical cache"
    )
    hicache_ratio: float = pydantic.Field(
        default=2.0,
        help="The ratio of the size of host KV cache memory pool to the size of device pool.",
    )
    hicache_size: int = pydantic.Field(
        default=0,
        help="The size of host KV cache memory pool in gigabytes, which will override the hicache_ratio if set.",
    )
    hicache_write_policy: str = pydantic.Field(
        default="write_through", help="The write policy of hierarchical cache."
    )
    hicache_io_backend: str = pydantic.Field(
        default="kernel",
        help="The IO backend for KV cache transfer between CPU and GPU",
    )
    hicache_mem_layout: str = pydantic.Field(
        default="layer_first",
        help="The layout of host memory pool for hierarchical cache.",
    )
    hicache_storage_backend: Optional[str] = pydantic.Field(
        default=None, help="The storage backend for hierarchical KV cache."
    )
    hicache_storage_prefetch_policy: str = pydantic.Field(
        default="best_effort",
        help="Control when prefetching from the storage backend should stop.",
    )
    hicache_storage_backend_extra_config: Optional[str] = pydantic.Field(
        default=None,
        help="A dictionary in JSON string format containing extra configuration for the storage backend.",
    )
    # LMCache
    enable_lmcache: bool = pydantic.Field(
        default=False,
        help="Using LMCache as an alternative hierarchical cache solution",
    )

    # Double Sparsity
    enable_double_sparsity: bool = pydantic.Field(
        default=False, help="Enable double sparsity attention"
    )
    ds_channel_config_path: Optional[str] = pydantic.Field(
        default=None, help="The path of the double sparsity channel config"
    )
    ds_heavy_channel_num: int = pydantic.Field(
        default=32, help="The number of heavy channels in double sparsity attention"
    )
    ds_heavy_token_num: int = pydantic.Field(
        default=256, help="The number of heavy tokens in double sparsity attention"
    )
    ds_heavy_channel_type: str = pydantic.Field(
        default="qk", help="The type of heavy channels in double sparsity attention"
    )
    ds_sparse_decode_threshold: int = pydantic.Field(
        default=4096, help="The type of heavy channels in double sparsity attention"
    )

    # Offloading
    cpu_offload_gb: int = pydantic.Field(
        default=0, help="How many GBs of RAM to reserve for CPU offloading."
    )
    offload_group_size: int = pydantic.Field(
        default=-1, help="Number of layers per group in offloading."
    )
    offload_num_in_group: int = pydantic.Field(
        default=1, help="Number of layers to be offloaded within a group."
    )
    offload_prefetch_step: int = pydantic.Field(
        default=1, help="Steps to prefetch in offloading."
    )
    offload_mode: str = pydantic.Field(default="cpu", help="Mode of offloading.")

    # Optimization/debug options
    disable_radix_cache: bool = pydantic.Field(
        default=False, help="Disable RadixAttention for prefix caching."
    )
    cuda_graph_max_bs: Optional[int] = pydantic.Field(
        default=None,
        help="Set the maximum batch size for cuda graph. It will extend the cuda graph capture batch size to this value.",
    )
    cuda_graph_bs: Optional[List[int]] = pydantic.Field(
        default=None, help="Set the list of batch sizes for cuda graph."
    )
    disable_cuda_graph: bool = pydantic.Field(default=False, help="Disable cuda graph.")
    disable_cuda_graph_padding: bool = pydantic.Field(
        default=False,
        help="Disable cuda graph when padding is needed. Still uses cuda graph when padding is not needed.",
    )
    enable_profile_cuda_graph: bool = pydantic.Field(
        default=False, help="Enable profiling of cuda graph capture."
    )
    enable_cudagraph_gc: bool = pydantic.Field(
        default=False,
        help="Enable garbage collection during CUDA graph capture. If disabled (default), GC is frozen during capture to speed up the process.",
    )
    enable_nccl_nvls: bool = pydantic.Field(
        default=False,
        help="Enable NCCL NVLS for prefill heavy requests when available.",
    )
    enable_symm_mem: bool = pydantic.Field(
        default=False, help="Enable NCCL symmetric memory for fast collectives."
    )
    disable_flashinfer_cutlass_moe_fp4_allgather: bool = pydantic.Field(
        default=False,
        help="Disables quantize before all-gather for flashinfer cutlass moe.",
    )
    enable_tokenizer_batch_encode: bool = pydantic.Field(
        default=False,
        help="Enable batch tokenization for improved performance when processing multiple text inputs.",
    )
    disable_outlines_disk_cache: bool = pydantic.Field(
        default=False,
        help="Disable disk cache of outlines to avoid possible crashes related to file system or high concurrency.",
    )
    disable_custom_all_reduce: bool = pydantic.Field(
        default=False,
        help="Disable the custom all-reduce kernel and fall back to NCCL.",
    )
    enable_mscclpp: bool = pydantic.Field(
        default=False,
        help="Enable using mscclpp for small messages for all-reduce kernel and fall back to NCCL.",
    )
    disable_overlap_schedule: bool = pydantic.Field(
        default=False,
        help="Disable the overlap scheduler, which overlaps the CPU scheduler with GPU model worker.",
    )
    enable_mixed_chunk: bool = pydantic.Field(
        default=False,
        help="Enabling mixing prefill and decode in a batch when using chunked prefill.",
    )
    enable_dp_attention: bool = pydantic.Field(
        default=False,
        help="Enabling data parallelism for attention and tensor parallelism for FFN.",
    )
    enable_dp_lm_head: bool = pydantic.Field(
        default=False,
        help="Enable vocabulary parallel across the attention TP group to avoid all-gather across DP groups.",
    )
    enable_two_batch_overlap: bool = pydantic.Field(
        default=False, help="Enabling two micro batches to overlap."
    )
    tbo_token_distribution_threshold: float = pydantic.Field(
        default=0.48,
        help="The threshold of token distribution between two batches in micro-batch-overlap.",
    )
    enable_torch_compile: bool = pydantic.Field(
        default=False,
        help="Optimize the model with torch.compile. Experimental feature.",
    )
    torch_compile_max_bs: int = pydantic.Field(
        default=32, help="Set the maximum batch size when using torch compile."
    )
    torchao_config: str = pydantic.Field(
        default="",
        help="Optimize the model with torchao. Experimental feature. Current choices are: int8dq, int8wo, int4wo-<group_size>, fp8wo, fp8dq-per_tensor, fp8dq-per_row",
    )
    enable_nan_detection: bool = pydantic.Field(
        default=False, help="Enable the NaN detection for debugging purposes."
    )
    enable_p2p_check: bool = pydantic.Field(
        default=False,
        help="Enable P2P check for GPU access, otherwise the p2p access is allowed by default.",
    )
    triton_attention_reduce_in_fp32: bool = pydantic.Field(
        default=False,
        help="Cast the intermediate attention results to fp32 to avoid possible crashes related to fp16.",
    )
    triton_attention_num_kv_splits: int = pydantic.Field(
        default=8,
        help="The number of KV splits in flash decoding Triton kernel. Larger value is better in longer context scenarios. The default value is 8.",
    )
    triton_attention_split_tile_size: Optional[int] = pydantic.Field(
        default=None,
        help="The size of split KV tile in flash decoding Triton kernel. Used for deterministic inference.",
    )
    num_continuous_decode_steps: int = pydantic.Field(
        default=1,
        help="Run multiple continuous decoding steps to reduce scheduling overhead.",
    )
    delete_ckpt_after_loading: bool = pydantic.Field(
        default=False, help="Delete the model checkpoint after loading the model."
    )
    enable_memory_saver: bool = pydantic.Field(
        default=False,
        help="Allow saving memory using release_memory_occupation and resume_memory_occupation",
    )
    allow_auto_truncate: bool = pydantic.Field(
        default=False,
        help="Allow automatically truncating requests that exceed the maximum input length instead of returning an error.",
    )
    enable_custom_logit_processor: bool = pydantic.Field(
        default=False,
        help="Enable users to pass custom logit processors to the server (disabled by default for security)",
    )
    flashinfer_mla_disable_ragged: bool = pydantic.Field(
        default=False,
        help="Not using ragged prefill wrapper when running flashinfer mla",
    )
    disable_shared_experts_fusion: bool = pydantic.Field(
        default=False,
        help="Disable shared experts fusion optimization for deepseek v3/r1.",
    )
    disable_chunked_prefix_cache: bool = pydantic.Field(
        default=False,
        help="Disable chunked prefix cache feature for deepseek, which should save overhead for short sequences.",
    )
    disable_fast_image_processor: bool = pydantic.Field(
        default=False,
        help="Adopt base image processor instead of fast image processor.",
    )
    keep_mm_feature_on_device: bool = pydantic.Field(
        default=False,
        help="Keep multimodal feature tensors on device after processing to save D2H copy.",
    )
    enable_return_hidden_states: bool = pydantic.Field(
        default=False, help="Enable returning hidden states with responses."
    )
    scheduler_recv_interval: int = pydantic.Field(
        default=1,
        help="The interval to poll requests in scheduler. Can be set to >1 to reduce the overhead of this.",
    )
    numa_node: Optional[List[int]] = pydantic.Field(
        default=None,
        help="Sets the numa node for the subprocesses. i-th element corresponds to i-th subprocess.",
    )

    # Dynamic batch tokenizer
    enable_dynamic_batch_tokenizer: bool = pydantic.Field(
        default=False,
        help="Enable async dynamic batch tokenizer for improved performance when multiple requests arrive concurrently.",
    )
    dynamic_batch_tokenizer_batch_size: int = pydantic.Field(
        default=32,
        help="[Only used if --enable-dynamic-batch-tokenizer is set] Maximum batch size for dynamic batch tokenizer.",
    )
    dynamic_batch_tokenizer_batch_timeout: float = pydantic.Field(
        default=0.002,
        help="[Only used if --enable-dynamic-batch-tokenizer is set] Timeout in seconds for batching tokenization requests.",
    )

    # Debug tensor dumps
    debug_tensor_dump_output_folder: Optional[str] = pydantic.Field(
        default=None, help="The output folder for dumping tensors."
    )
    debug_tensor_dump_input_file: Optional[str] = pydantic.Field(
        default=None, help="The input filename for dumping tensors"
    )
    debug_tensor_dump_inject: bool = pydantic.Field(
        default=False, help="Inject the outputs from jax as the input of every layer."
    )
    debug_tensor_dump_prefill_only: bool = pydantic.Field(
        default=False,
        help="Only dump the tensors for prefill requests (i.e. batch size > 1).",
    )

    # PD disaggregation: can be "null" (not disaggregated), "prefill" (prefill-only), or "decode" (decode-only)
    disaggregation_mode: Literal[
        DisaggregationMode.NULL, DisaggregationMode.PREFILL, DisaggregationMode.DECODE
    ] = pydantic.Field(
        default=DisaggregationMode.NULL,
        help="Disaggregation mode: 'null' (not disaggregated), 'prefill' (prefill-only), or 'decode' (decode-only).",
    )
    disaggregation_transfer_backend: Literal[
        DisaggTransferBackend.MOONCAKE,
        DisaggTransferBackend.NIXL,
        DisaggTransferBackend.ASCEND,
        DisaggTransferBackend.FAKE,
    ] = pydantic.Field(
        default=DisaggTransferBackend.MOONCAKE,
        help="The backend for disaggregation transfer.",
    )
    disaggregation_bootstrap_port: int = pydantic.Field(
        default=8998, help="Port for disaggregation bootstrap."
    )
    disaggregation_decode_tp: Optional[int] = pydantic.Field(
        default=None, help="Tensor parallelism size for decode disaggregation."
    )
    disaggregation_decode_dp: Optional[int] = pydantic.Field(
        default=None, help="Data parallelism size for decode disaggregation."
    )
    disaggregation_prefill_pp: Optional[int] = pydantic.Field(
        default=1, help="Pipeline parallelism size for prefill disaggregation."
    )
    disaggregation_ib_device: Optional[str] = pydantic.Field(
        default=None, help="InfiniBand device for disaggregation."
    )
    num_reserved_decode_tokens: int = pydantic.Field(
        default=512,
        help="Number of reserved decode tokens for decode KV cache offload in PD.",
    )

    # FIXME: hack to reduce ITL when decode bs is small
    disaggregation_decode_polling_interval: int = pydantic.Field(
        default=1, help="The interval to poll requests in decode server."
    )

    # For model weight update
    custom_weight_loader: Optional[List[str]] = pydantic.Field(
        default=None,
        help="The custom dataloader which used to update the model. Should be set with a valid import path.",
    )
    weight_loader_disable_mmap: bool = pydantic.Field(
        default=False, help="Disable mmap while loading weight using safetensors."
    )

    # Remote instance weight loading
    remote_instance_weight_loader_seed_instance_ip: Optional[str] = pydantic.Field(
        default=None,
        help="The ip of the seed instance for loading weights from remote instance.",
    )
    remote_instance_weight_loader_seed_instance_service_port: Optional[int] = (
        pydantic.Field(
            default=None,
            help="The service port of the seed instance for loading weights from remote instance.",
        )
    )
    remote_instance_weight_loader_send_weights_group_ports: Optional[List[int]] = (
        pydantic.Field(
            default=None,
            help="The communication group ports for loading weights from remote instance.",
        )
    )

    # For PD-Multiplexing
    enable_pdmux: bool = pydantic.Field(
        default=False, help="Enable PD-Multiplexing, PD running on greenctx stream."
    )
    sm_group_num: int = pydantic.Field(default=3, help="Number of sm partition groups.")

    # Mamba cache
    max_mamba_cache_size: Optional[int] = pydantic.Field(
        default=None, help="The maximum size of the mamba cache."
    )
    mamba_ssm_dtype: str = pydantic.Field(
        default="float32", help="The data type of the SSM states in mamba cache."
    )

    # For deterministic inference
    enable_deterministic_inference: bool = pydantic.Field(
        default=False,
        help="Enable deterministic inference mode with batch invariant ops.",
    )

    # Deprecated arguments
    enable_ep_moe: bool = pydantic.Field(
        default=False,
        help="(Deprecated) Enabling expert parallelism for moe. The ep size is equal to the tp size.",
    )
    enable_deepep_moe: bool = pydantic.Field(
        default=False,
        help="(Deprecated) Enabling DeepEP MoE implementation for EP MoE.",
    )
    enable_flashinfer_cutlass_moe: bool = pydantic.Field(
        default=False,
        help="(Deprecated) Enable FlashInfer CUTLASS MoE backend for modelopt_fp4 quant on Blackwell.",
    )
    enable_flashinfer_cutedsl_moe: bool = pydantic.Field(
        default=False,
        help="(Deprecated) Enable FlashInfer CuteDSL MoE backend for modelopt_fp4 quant on Blackwell.",
    )
    enable_flashinfer_trtllm_moe: bool = pydantic.Field(
        default=False,
        help="(Deprecated) Enable FlashInfer TRTLLM MoE backend on Blackwell.",
    )
    enable_triton_kernel_moe: bool = pydantic.Field(
        default=False, help="(Deprecated) Use triton moe grouped gemm kernel."
    )
    enable_flashinfer_mxfp4_moe: bool = pydantic.Field(
        default=False,
        help="(Deprecated) Enable FlashInfer MXFP4 MoE backend for modelopt_fp4 quant on Blackwell.",
    )
    _hf_config: Optional[pydantic.PrivateAttr] = pydantic.PrivateAttr(default=None)

    def _get_hf_config(self):
        """Helper method to get and cache the HuggingFace model config."""
        if self._hf_config is None:
            self._hf_config = get_config(
                self.model_path,
                trust_remote_code=self.trust_remote_code,
                revision=self.revision,
                model_override_args=json.loads(self.json_model_override_args),
            )
        return self._hf_config

    def _auto_choose_speculative_params(self):
        """
        Private helper method to automatically choose parameters for speculative decoding.
        """
        hf_config = self._get_hf_config()
        arch = hf_config.architectures[0]
        if self.speculative_algorithm == "STANDALONE":
            return (3, 1, 4)
        if arch in ["LlamaForCausalLM"]:
            return (5, 4, 8)
        elif arch in [
            "DeepseekV3ForCausalLM",
            "DeepseekV2ForCausalLM",
            "GptOssForCausalLM",
            "BailingMoeForCausalLM",
            "BailingMoeV2ForCausalLM",
        ]:
            return (3, 1, 4)
        elif arch in ["Grok1ForCausalLM", "Grok1VForCausalLM"]:
            return (5, 4, 8)
        else:
            return (5, 4, 8)

    def url(self) -> str:
        """Computes the server URL based on host and port."""
        # Note: You need to import `is_valid_ipv6_address`
        if is_valid_ipv6_address(self.host):
            return f"http://[{self.host}]:{self.port}"
        return f"http://{self.host}:{self.port}"

    def validate_buckets_rule(self, arg_name: str, buckets_rule: List[str]):
        if not buckets_rule:
            return

        assert len(buckets_rule) > 0, f"{arg_name} cannot be empty list"
        rule = buckets_rule[0]
        assert rule in [
            "tse",
            "default",
            "customer",
        ], f"Unsupported {arg_name} rule type: '{rule}'. Must be one of: 'tse', 'default', 'customer'"

        if rule == "tse":
            assert (
                len(buckets_rule) == 4
            ), f"{arg_name} TSE rule requires exactly 4 parameters: ['tse', middle, base, count], got {len(buckets_rule)}"
            try:
                middle = float(buckets_rule[1])
                base = float(buckets_rule[2])
                count = int(buckets_rule[3])
            except (ValueError, IndexError):
                assert (
                    False
                ), f"{arg_name} TSE rule parameters must be: ['tse', <float:middle>, <float:base>, <int:count>]"
            assert base > 1, f"{arg_name} TSE base must be larger than 1, got: {base}"
            assert count > 0, f"{arg_name} TSE count must be positive, got: {count}"
            assert middle > 0, f"{arg_name} TSE middle must be positive, got: {middle}"

        elif rule == "default":
            assert (
                len(buckets_rule) == 1
            ), f"{arg_name} default rule should only have one parameter: ['default'], got {len(buckets_rule)}"

        elif rule == "customer":
            assert (
                len(buckets_rule) >= 2
            ), f"{arg_name} customer rule requires at least one bucket value: ['customer', value1, ...]"
            try:
                bucket_values = [float(x) for x in buckets_rule[1:]]
            except ValueError:
                assert False, f"{arg_name} customer rule bucket values must be numeric"
            assert len(set(bucket_values)) == len(
                bucket_values
            ), f"{arg_name} customer rule bucket values should not contain duplicates"
            assert all(
                val >= 0 for val in bucket_values
            ), f"{arg_name} customer rule bucket values should be non-negative"

    def _validate_disagg_tp_size(self, prefill_tp: int, decode_tp: int):
        larger_tp = max(decode_tp, prefill_tp)
        smaller_tp = min(decode_tp, prefill_tp)
        if larger_tp % smaller_tp != 0:
            raise ValueError(
                "Different tp size is supported only when one tp is multiple of the other. "
                f"decode_tp={decode_tp}, prefill_tp={prefill_tp}"
            )

    def _model_specific_adjustments(self):
        hf_config = self._get_hf_config()
        model_arch = hf_config.architectures[0]
        if model_arch in ["GptOssForCausalLM"]:
            if self.attention_backend is None:
                if is_cuda() and is_sm100_supported():
                    self.attention_backend = "trtllm_mha"
                elif is_cuda() and is_sm90_supported():
                    self.attention_backend = "fa3"
                else:
                    self.attention_backend = "triton"
            supported_backends = ["triton", "trtllm_mha", "fa3"]
            logger.info(
                f"Use {self.attention_backend} as attention backend for GptOssForCausalLM"
            )
            assert (
                self.attention_backend in supported_backends
            ), f"GptOssForCausalLM requires one of {supported_backends} attention backend, but got '{self.attention_backend}'"

            if is_sm100_supported():
                if not self.enable_dp_attention:
                    self.enable_flashinfer_allreduce_fusion = True
                    logger.info(
                        "Enable FlashInfer AllReduce Fusion on sm100 for GptOssForCausalLM"
                    )
            quantization_config = getattr(hf_config, "quantization_config", None)
            is_mxfp4_quant_format = (
                quantization_config is not None
                and quantization_config.get("quant_method") == "mxfp4"
            )

            if is_sm100_supported() and is_mxfp4_quant_format:
                self.moe_runner_backend = "flashinfer_mxfp4"
                logger.warning(
                    "Detected SM100 and MXFP4 quantization format for GPT-OSS model, enabling FlashInfer MXFP4 MOE kernel."
                )
            else:
                if self.moe_runner_backend == "triton_kernel":
                    if self.ep_size != 1:
                        raise ValueError(
                            "Triton kernel MoE is only supported when ep_size == 1"
                        )
                if (
                    self.moe_runner_backend == "auto"
                    and self.ep_size == 1
                    and is_triton_kernels_available()
                ):
                    self.moe_runner_backend = "triton_kernel"
                    logger.warning(
                        "Detected GPT-OSS model, enabling triton_kernels MOE kernel."
                    )
            self.disable_hybrid_swa_memory = True
            if is_mxfp4_quant_format:
                # use bf16 for mxfp4 triton kernels
                self.dtype = "bfloat16"

        elif "Llama4" in model_arch and self.device != "cpu":
            if self.attention_backend not in {
                AttentionBackend.FA3,
                AttentionBackend.AITER,
                AttentionBackend.TRITON,
            }:
                raise ValueError("fa3, aiter, or triton is required for Llama4 model")
        elif model_arch in [
            "Gemma2ForCausalLM",
            "Gemma3ForCausalLM",
            "Gemma3ForConditionalGeneration",
            "Gemma3nForCausalLM",
            "Gemma3nForConditionalGeneration",
        ]:
            # FIXME: https://github.com/sgl-project/sglang/pull/7367 is not compatible with gemma2 model.
            # It failed at this test: https://github.com/sgl-project/sglang/actions/runs/16255155597/job/45890331952#step:4:736
            logger.warning(
                f"Disable hybrid SWA memory for {model_arch} as it is not yet supported."
            )
            self.disable_hybrid_swa_memory = True

    def adjust_mem_fraction_for_vlm(self, model_config):
        vision_config = getattr(model_config.hf_config, "vision_config", None)
        if vision_config is None:
            return

        # roughly reduce the mem_fraction_static base on params of Vit
        original_server_arg_mem_fraction = self.mem_fraction_static
        # a base mem_fraction_static factor for regular Vit
        base_mem_fraction_reduction_ratio = 0.95

        vit_num_layers = getattr(vision_config, "num_hidden_layers", 24)
        vit_hidden_size = getattr(vision_config, "hidden_size", 1024)

        # baseline ViT params (ViT-L/14)
        baseline_vit_layers = 24
        baseline_vit_hidden_size = 1024

        # weight params count
        current_complexity_score = vit_num_layers * (vit_hidden_size**2)
        baseline_complexity_score = baseline_vit_layers * (baseline_vit_hidden_size**2)
        complexity_ratio = (
            current_complexity_score / baseline_complexity_score
            if baseline_complexity_score > 0
            else 1.0
        )

        # every time the complexity grows 100%, adjust final factor for 10%
        sensitivity_scale = 0.1
        dynamic_adjustment_factor = 1.0 - sensitivity_scale * (complexity_ratio - 1.0)
        dynamic_adjustment_factor = max(0.8, min(1.05, dynamic_adjustment_factor))

        final_overall_factor = (
            base_mem_fraction_reduction_ratio * dynamic_adjustment_factor
        )
        self.mem_fraction_static = (
            original_server_arg_mem_fraction * final_overall_factor
        )

    @pydantic.model_validator(mode="after")
    def complete_and_validate_config(self) -> Self:
        # Step 1: Handle deprecated arguments.
        if self.enable_ep_moe:
            self.ep_size = self.tp_size
            print_deprecated_warning(
                "NOTE: --enable-ep-moe is deprecated. Please set `--ep-size` to the same value as `--tp-size` instead."
            )
        if self.enable_deepep_moe:
            self.moe_a2a_backend = MoeA2ABackend.DEEPEP
            print_deprecated_warning(
                "NOTE: --enable-deepep-moe is deprecated. Please set `--moe-a2a-backend` to 'deepep' instead."
            )
        if self.enable_triton_kernel_moe:
            self.moe_runner_backend = MoeRunnerBackend.TRITON_KERNEL
            print_deprecated_warning(
                "NOTE: --enable-triton-kernel-moe is deprecated. Please set `--moe-runner-backend` to 'triton_kernel' instead."
            )
        if self.enable_flashinfer_cutedsl_moe:
            self.moe_runner_backend = "flashinfer_cutedsl"
            print_deprecated_warning(
                "NOTE: --enable-flashinfer-cutedsl-moe is deprecated. Please set `--moe-runner-backend` to 'flashinfer_cutedsl' instead."
            )
        if self.enable_flashinfer_cutlass_moe:
            self.moe_runner_backend = MoeRunnerBackend.FLASHINFER_CUTLASS
            print_deprecated_warning(
                "NOTE: --enable-flashinfer-cutlass-moe is deprecated. Please set `--moe-runner-backend` to 'flashinfer_cutlass' instead."
            )
        if self.enable_flashinfer_trtllm_moe:
            self.moe_runner_backend = MoeRunnerBackend.FLASHINFER_TRTLLM
            print_deprecated_warning(
                "NOTE: --enable-flashinfer-trtllm-moe is deprecated. Please set `--moe-runner-backend` to 'flashinfer_trtllm' instead."
            )
        if self.enable_flashinfer_mxfp4_moe:
            self.moe_runner_backend = MoeRunnerBackend.FLASHINFER_MXFP4
            print_deprecated_warning(
                "NOTE: --enable-flashinfer-mxfp4-moe is deprecated. Please set `--moe-runner-backend` to 'flashinfer_mxfp4' instead."
            )

        # _handle_missing_default_values
        if self.tokenizer_path is None:
            self.tokenizer_path = self.model_path
        if self.served_model_name is None:
            self.served_model_name = self.model_path
        if self.device is None:
            self.device = get_device()
        if self.random_seed is None:
            self.random_seed = random.randint(0, 1 << 30)

        # _handle_mem_fraction_static
        gpu_mem = get_device_memory_capacity(self.device)
        if self.mem_fraction_static is None:
            if gpu_mem is not None:
                parallel_size = self.tp_size * self.pp_size
                if gpu_mem < 20 * 1024:
                    reserved_mem = (2.8 + parallel_size / 10) * 1024
                elif gpu_mem < 35 * 1024:
                    reserved_mem = (2.8 + parallel_size / 10) * 1024
                elif gpu_mem < 90 * 1024:
                    reserved_mem = (9.5 + parallel_size / 2) * 1024
                elif gpu_mem < 100 * 1024:
                    reserved_mem = (12 + parallel_size / 2) * 1024
                elif gpu_mem < 160 * 1024:
                    reserved_mem = (12 + parallel_size / 2) * 1024
                else:
                    reserved_mem = 32 * 1024

                if self.speculative_algorithm is not None:
                    if self.speculative_algorithm == "STANDALONE":
                        reserved_mem += 6 * 1024
                    elif self.speculative_algorithm != "LOOKAHEAD":
                        reserved_mem += 2 * 1024
                if self.enable_dp_attention:
                    reserved_mem += 4 * 1024

                self.mem_fraction_static = round((gpu_mem - reserved_mem) / gpu_mem, 3)
            else:
                self.mem_fraction_static = 0.88
            from sglang.srt.configs.model_config import ModelConfig

            model_config = ModelConfig.from_server_args(self)
            if model_config.is_multimodal:
                self.adjust_mem_fraction_for_vlm(model_config)

        # (from _handle_chunked_prefill_size)
        if self.chunked_prefill_size is None:
            if gpu_mem is not None:
                if gpu_mem < 35 * 1024:
                    self.chunked_prefill_size = 2048
                elif gpu_mem < 160 * 1024:
                    self.chunked_prefill_size = 8192
                else:
                    self.chunked_prefill_size = 16384
            else:
                self.chunked_prefill_size = 4096

        # (from _handle_cuda_graph_max_bs)
        if self.cuda_graph_max_bs is None:
            if gpu_mem is not None and gpu_mem < 35 * 1024:
                if self.tp_size < 4:
                    self.cuda_graph_max_bs = 8
                else:
                    self.cuda_graph_max_bs = 80

        # _handle_hpu_backends
        if self.device == "hpu":
            self.attention_backend = AttentionBackend.TORCH_NATIVE
            self.sampling_backend = "pytorch"

        # _handle_cpu_backends
        if self.device == "cpu":
            if self.attention_backend is None:
                self.attention_backend = AttentionBackend.INTEL_AMX
            self.sampling_backend = "pytorch"

        if parse_connector_type(self.model_path) != ConnectorType.INSTANCE:
            self._model_specific_adjustments()

        # _handle_sampling_backend
        if self.sampling_backend is None:
            self.sampling_backend = (
                "flashinfer" if is_flashinfer_available() else "pytorch"
            )

        # _handle_attention_backend_compatibility
        if (
            self.attention_backend == AttentionBackend.TORCH_NATIVE
            or self.attention_backend == "torch_native"
        ):
            logger.warning(
                "Cuda graph is disabled because of using torch native attention backend"
            )
            self.disable_cuda_graph = True
        if (
            self.attention_backend == AttentionBackend.FLEX_ATTENTION
            or self.attention_backend == "flex_attention"
        ):
            logger.warning(
                "Cuda graph is disabled because of using torch Flex Attention backend"
            )
            self.disable_cuda_graph = True
            assert (
                self.speculative_algorithm is None
            ), "Speculative decoding is currently not supported with Flex Attention backend"
        if is_npu() and self.attention_backend in [
            AttentionBackend.ASCEND,
            AttentionBackend.HYBRID_LINEAR_ATTN,
        ]:
            logger.warning(
                "At this moment Ascend attention backend only supports a page_size of 128, change page_size to 128."
            )
            self.page_size = 128
        if (
            self.attention_backend == AttentionBackend.FLASHMLA
            or self.decode_attention_backend == "flashmla"
        ):
            logger.warning(
                "FlashMLA only supports a page_size of 64, change page_size to 64."
            )
            self.page_size = 64
        if (
            self.attention_backend == AttentionBackend.CUTLASS_MLA
            or self.decode_attention_backend == "cutlass_mla"
        ):
            logger.warning(
                "Cutlass MLA only supports a page_size of 128, change page_size to 128."
            )
            self.page_size = 128

        if (
            self.attention_backend == AttentionBackend.TRTLLM_MLA
            or self.decode_attention_backend == "trtllm_mla"
        ):
            if not is_sm100_supported():
                raise ValueError(
                    "TRTLLM MLA backend is only supported on Blackwell GPUs (SM100). Please use a different backend."
                )

            if self.page_size not in [32, 64]:
                logger.warning(
                    f"TensorRT-LLM MLA only supports page_size of 32 or 64, changing page_size from {self.page_size} to 64."
                )
                self.page_size = 64

            if self.kv_cache_dtype not in ["fp8_e4m3", "auto"]:
                raise ValueError(
                    "TensorRT-LLM MLA backend only supports kv-cache-dtype of fp8_e4m3 or auto."
                )

        if (
            self.attention_backend == AttentionBackend.TRTLLM_MHA
            or self.decode_attention_backend == "trtllm_mha"
            or self.prefill_attention_backend == "trtllm_mha"
        ):
            if not is_sm100_supported():
                raise ValueError(
                    "TRTLLM MHA backend is only supported on Blackwell GPUs (SM100). Please use a different backend."
                )

            if self.page_size not in [16, 32, 64]:
                logger.warning(
                    f"TensorRT-LLM MHA only supports page_size of 16, 32 or 64, changing page_size from {self.page_size} to 64."
                )
                self.page_size = 64

        if self.attention_backend == AttentionBackend.DUAL_CHUNK_FLASH_ATTN:
            logger.warning(
                "Mixed chunk, radix cache, and cuda graphs are disabled because of using dual chunk flash attention backend"
            )
            self.enable_mixed_chunk = False
            self.disable_cuda_graph = True
            self.disable_radix_cache = True

        #  _handle_page_size
        if self.page_size is None:
            self.page_size = 1
        # _handle_amd_specifics
        #
        if is_hip():
            self.triton_attention_num_kv_splits = 16
        # _handle_grammar_backend

        if self.grammar_backend is None:
            self.grammar_backend = GrammarBackend.XGRAMMAR

        # _handle_data_parallelism
        if self.dp_size == 1:
            self.enable_dp_attention = False
            self.enable_dp_lm_head = False

        if self.enable_dp_attention:
            self.schedule_conservativeness = self.schedule_conservativeness * 0.3
            assert self.tp_size % self.dp_size == 0
            self.chunked_prefill_size = self.chunked_prefill_size // self.dp_size
            logger.warning(
                f"DP attention is enabled. The chunked prefill size is adjusted to {self.chunked_prefill_size} to avoid MoE kernel issues. "
            )

        if self.enable_dp_lm_head:
            assert (
                self.enable_dp_attention
            ), "Please enable dp attention when setting enable_dp_lm_head. "

        # _handle_moe_kernel_config
        if self.moe_runner_backend == MoeRunnerBackend.FLASHINFER_CUTLASS:
            assert (
                self.quantization == QuantizationMethod.MODELOPT_FP4
            ), "modelopt_fp4 quantization is required for Flashinfer MOE"
            assert self.ep_size in [
                1,
                self.tp_size,
            ], "The expert parallel size must be 1 or the same as the tensor parallel size"

        if self.moe_runner_backend == MoeRunnerBackend.FLASHINFER_TRTLLM:
            assert (
                self.quantization == QuantizationMethod.MODELOPT_FP4
                or self.quantization == QuantizationMethod.FP8
            ), "modelopt_fp4 quantization is required for Flashinfer TRTLLM MoE"
            self.disable_shared_experts_fusion = True
            logger.warning(
                "FlashInfer TRTLLM MoE is enabled. --disable-shared-experts-fusion is automatically set."
            )

        # _handle_deepep_moe
        if self.moe_a2a_backend == MoeA2ABackend.DEEPEP:
            if self.deepep_mode == DeepEPMode.NORMAL:
                logger.warning("Cuda graph is disabled because deepep_mode=`normal`")
                self.disable_cuda_graph = True
            self.ep_size = self.tp_size
            logger.warning(
                f"DeepEP MoE is enabled. The expert parallel size is adjusted to be the same as the tensor parallel size[{self.tp_size}]."
            )

        # _handle_eplb_and_dispatch
        if self.enable_eplb and (self.expert_distribution_recorder_mode is None):
            self.expert_distribution_recorder_mode = ExpertDistributionRecorderMode.STAT
            logger.warning(
                "EPLB is enabled. The expert_distribution_recorder_mode is automatically set."
            )

        if (self.enable_eplb or (self.init_expert_location is not None)) and (
            self.ep_dispatch_algorithm is None
        ):
            self.ep_dispatch_algorithm = EPDispatchAlgorithm.STATIC

        if self.enable_eplb:
            assert self.ep_size > 1

        # _handle_expert_distribution_metrics
        if self.enable_expert_distribution_metrics and (
            self.expert_distribution_recorder_mode is None
        ):
            self.expert_distribution_recorder_mode = ExpertDistributionRecorderMode.STAT

        if self.expert_distribution_recorder_buffer_size is None:
            if (x := self.eplb_rebalance_num_iterations) is not None:
                self.expert_distribution_recorder_buffer_size = x
            elif self.expert_distribution_recorder_mode is not None:
                self.expert_distribution_recorder_buffer_size = 1000

        # _handle_pipeline_parallelism
        if self.pp_size > 1:
            self.disable_overlap_schedule = True
            logger.warning(
                "Pipeline parallelism is incompatible with overlap schedule."
            )

        # check_server_args
        if (self.tp_size * self.pp_size) % self.nnodes != 0:
            raise ValueError(
                "tp_size * pp_size must be divisible by the number of nodes."
            )

        if self.pp_size > 1 and (
            not self.disable_overlap_schedule
            or self.speculative_algorithm is not None
            or self.enable_mixed_chunk
        ):
            raise ValueError(
                "Pipeline parallelism is not compatible with overlap schedule, speculative decoding, or mixed chunked prefill."
            )

        # check_lora_server_args
        if self.lora_paths:
            if self.max_loras_per_batch <= 0:
                raise ValueError("max_loras_per_batch must be positive.")
            if self.enable_lora is None:
                self.enable_lora = True
                logger.warning(
                    "--enable-lora is set to True because --lora-paths is provided."
                )
            elif self.enable_lora is False:
                logger.warning(
                    "--enable-lora is set to False, any provided lora_paths will be ignored."
                )
        if self.enable_lora:
            if isinstance(self.lora_paths, list):
                lora_paths = self.lora_paths
                self.lora_paths = []
                for lora_path in lora_paths:
                    if isinstance(lora_path, str):
                        if "=" in lora_path:
                            name, path = lora_path.split("=", 1)
                            lora_ref = LoRARef(
                                lora_name=name, lora_path=path, pinned=False
                            )
                        else:
                            lora_ref = LoRARef(
                                lora_name=lora_path, lora_path=lora_path, pinned=False
                            )
                    elif isinstance(lora_path, dict):
                        assert (
                            "lora_name" in lora_path and "lora_path" in lora_path
                        ), f"When providing LoRA paths as a list of dict, each dict should contain 'lora_name' and 'lora_path' keys. Got: {lora_path}"
                        lora_ref = LoRARef(
                            lora_name=lora_path["lora_name"],
                            lora_path=lora_path["lora_path"],
                            pinned=lora_path.get("pinned", False),
                        )
                    else:
                        raise ValueError(
                            f"Invalid type for item in --lora-paths list: {type(lora_path)}. "
                            "Expected a string or a dictionary."
                        )
                    self.lora_paths.append(lora_ref)
            elif isinstance(self.lora_paths, dict):
                self.lora_paths = [
                    LoRARef(lora_name=k, lora_path=v, pinned=False)
                    for k, v in self.lora_paths.items()
                ]
            elif self.lora_paths is None:
                self.lora_paths = []
            else:
                raise ValueError(
                    f"Invalid type for --lora-paths: {type(self.lora_paths)}. "
                    "Expected a list or a dictionary."
                )

            # Expand target modules
            if self.lora_target_modules:
                self.lora_target_modules = set(self.lora_target_modules)
                if "all" in self.lora_target_modules:
                    assert (
                        len(self.lora_target_modules) == 1
                    ), "If 'all' is specified in --lora-target-modules, it should be the only module specified."
                    self.lora_target_modules = set(SUPPORTED_LORA_TARGET_MODULES)

            # Ensure sufficient information is provided for LoRA initialization.
            assert self.lora_paths or (
                self.max_lora_rank and self.lora_target_modules
            ), "When no initial --lora-paths is provided, you need to specify both --max-lora-rank and --lora-target-modules for LoRA initialization."

            # Validate max_loaded_loras
            if self.max_loaded_loras is not None:
                assert self.max_loaded_loras >= self.max_loras_per_batch, (
                    "max_loaded_loras should be greater than or equal to max_loras_per_batch. "
                    f"max_loaded_loras={self.max_loaded_loras}, max_loras_per_batch={self.max_loras_per_batch}"
                )
                assert len(self.lora_paths) <= self.max_loaded_loras, (
                    "The number of LoRA paths should not exceed max_loaded_loras. "
                    f"max_loaded_loras={self.max_loaded_loras}, lora_paths={len(self.lora_paths)}"
                )

            if self.max_lora_chunk_size is not None:
                assert (
                    16 <= self.max_lora_chunk_size <= 128
                    and (self.max_lora_chunk_size & (self.max_lora_chunk_size - 1)) == 0
                ), "--max-lora-chunk-size must be a power of 2 between 16 and 128."

        # _handle_hicache
        if self.hicache_storage_backend == "mooncake":
            self.hicache_io_backend = "kernel"
            self.hicache_mem_layout = "page_first"

        if self.hicache_mem_layout == "page_first_direct":
            if self.hicache_io_backend != "direct":
                self.hicache_io_backend = "direct"
                logger.warning(
                    "Page first direct layout only support direct io backend"
                )

        # _handle_speculative_decoding
        if self.speculative_algorithm == "NEXTN":
            self.speculative_algorithm = "EAGLE"

        if self.speculative_algorithm in ("EAGLE", "EAGLE3", "STANDALONE"):
            if self.speculative_algorithm == "STANDALONE" and self.enable_dp_attention:
                # TODO: support dp attention for standalone speculative decoding
                raise ValueError(
                    "Currently standalone speculative decoding does not support dp attention."
                )
            if self.max_running_requests is None:
                self.max_running_requests = 48
            self.disable_overlap_schedule = True
            logger.warning(
                "Overlap scheduler is disabled because of using "
                "eagle speculative decoding."
            )
            if self.enable_mixed_chunk:
                self.enable_mixed_chunk = False
                logger.warning(
                    "Mixed chunked prefill is disabled because of using "
                    "eagle speculative decoding."
                )

            model_arch = self.get_hf_config().architectures[0]
            if model_arch in [
                "DeepseekV3ForCausalLM",
                "Glm4MoeForCausalLM",
                "BailingMoeForCausalLM",
                "BailingMoeV2ForCausalLM",
            ]:
                if self.speculative_draft_model_path is None:
                    self.speculative_draft_model_path = self.model_path
                else:
                    logger.warning(
                        "DeepSeek MTP does not require setting speculative_draft_model_path."
                    )

            if self.speculative_num_steps is None:
                assert (
                    self.speculative_eagle_topk is None
                    and self.speculative_num_draft_tokens is None
                )
                (
                    self.speculative_num_steps,
                    self.speculative_eagle_topk,
                    self.speculative_num_draft_tokens,
                ) = self._auto_choose_speculative_params(self)

            if (
                self.attention_backend == AttentionBackend.TRTLLM_MHA
                or self.decode_attention_backend == "trtllm_mha"
                or self.prefill_attention_backend == "trtllm_mha"
            ):
                if self.speculative_eagle_topk > 1:
                    raise ValueError(
                        "trtllm_mha backend only supports topk = 1 for speculative decoding."
                    )

            if (
                self.speculative_eagle_topk == 1
                and self.speculative_num_draft_tokens != self.speculative_num_steps + 1
            ):
                logger.warning(
                    "speculative_num_draft_tokens is adjusted to speculative_num_steps + 1 when speculative_eagle_topk == 1"
                )
                self.speculative_num_draft_tokens = self.speculative_num_steps + 1

            if (
                self.speculative_eagle_topk > 1
                and self.page_size > 1
                and self.attention_backend != AttentionBackend.FLASHINFER
            ):
                raise ValueError(
                    "speculative_eagle_topk > 1 with page_size > 1 is unstable and produces incorrect results for paged attention backends. This combination is only supported for the 'flashinfer' backend."
                )

        if self.speculative_algorithm == "LOOKAHEAD":
            if not self.device.startswith("cuda"):
                raise ValueError(
                    "Lookahead speculative decoding only supports CUDA device."
                )
            if self.max_running_requests is None:
                self.max_running_requests = 48
            self.disable_overlap_schedule = True
            self.enable_mixed_chunk = False
            self.speculative_eagle_topk = self.speculative_lookahead_max_bfs_breadth
            if self.speculative_num_draft_tokens is None:
                self.speculative_num_draft_tokens = (
                    self.speculative_lookahead_max_match_window_size
                )
            logger.warning(
                "The overlap scheduler and mixed chunked prefill are disabled because of "
                "using lookahead speculative decoding."
            )

            if (
                self.speculative_eagle_topk > 1
                and self.page_size > 1
                and self.attention_backend != AttentionBackend.FLASHINFER
            ):
                raise ValueError(
                    "speculative_eagle_topk > 1 with page_size > 1 is unstable and produces incorrect results for paged attention backends. This combination is only supported for the 'flashinfer' backend."
                )
            if self.enable_dp_attention:
                # TODO: support dp attention for lookahead speculative decoding
                raise ValueError(
                    "Currently lookahead speculative decoding does not support dp attention."
                )
        # _handle_load_format
        if (
            self.load_format == LoadFormat.AUTO or self.load_format == LoadFormat.GGUF
        ) and check_gguf_file(self.model_path):
            self.quantization = QuantizationMethod.GGUF
            self.load_format = LoadFormat.GGUF

        if is_remote_url(self.model_path):
            self.load_format = LoadFormat.REMOTE
        if self.custom_weight_loader is None:
            self.custom_weight_loader = []
        if self.load_format == LoadFormat.REMOTE_INSTANCE:
            if (
                self.remote_instance_weight_loader_seed_instance_ip is None
                or self.remote_instance_weight_loader_seed_instance_service_port is None
                or self.remote_instance_weight_loader_send_weights_group_ports is None
            ):
                self.load_format = LoadFormat.AUTO
        # _handle_disaggregation
        if self.disaggregation_mode == "decode":
            assert (
                self.disaggregation_decode_tp is None
            ), "Cannot set --disaggregation-decode-tp for the decode engine."

            assert (
                self.disaggregation_decode_dp is None
            ), "Cannot set --disaggregation-decode-dp for the decode engine."

            self.disable_radix_cache = True
            logger.warning("KV cache is forced as chunk cache for decode server")
            if self.dp_size > 1 and not is_in_ci():
                assert self.prefill_round_robin_balance, (
                    "Prefill round robin balance is required when dp size > 1. "
                    "Please make sure that the prefill instance is launched with `--load-balance-method round_robin`"
                    " and `--prefill-round-robin-balance` is set for decode server."
                )
        elif self.disaggregation_mode == "prefill":
            if self.disaggregation_decode_tp is None:
                self.disaggregation_decode_tp = self.tp_size
            if self.disaggregation_decode_dp is None:
                self.disaggregation_decode_dp = self.dp_size
            self.disaggregation_prefill_pp = self.pp_size
            self.validate_disagg_tp_size(self.tp_size, self.disaggregation_decode_tp)
            self.disable_cuda_graph = True
            logger.warning("Cuda graph is disabled for prefill server")
        # _handle_tokenizer_batching
        if self.enable_tokenizer_batch_encode and self.enable_dynamic_batch_tokenizer:
            raise ValueError(
                "Cannot enable both --enable-tokenizer-batch-encode and --enable-dynamic-batch-tokenizer. "
                "Please choose one tokenizer batching approach."
            )
        # _handle_environment_variables
        os.environ["SGLANG_ENABLE_TORCH_COMPILE"] = (
            "1" if self.enable_torch_compile else "0"
        )
        os.environ["SGLANG_MAMBA_SSM_DTYPE"] = self.mamba_ssm_dtype
        os.environ["SGLANG_DISABLE_OUTLINES_DISK_CACHE"] = (
            "1" if self.disable_outlines_disk_cache else "0"
        )
        os.environ["SGLANG_ENABLE_DETERMINISTIC_INFERENCE"] = (
            "1" if self.enable_deterministic_inference else "0"
        )

        # _handle_cache_compatibility
        if self.enable_hierarchical_cache and self.disable_radix_cache:
            raise ValueError(
                "The arguments enable-hierarchical-cache and disable-radix-cache are mutually exclusive "
                "and cannot be used at the same time. Please use only one of them."
            )

        # _handle_metrics_labels
        if (
            not self.tokenizer_metrics_custom_labels_header
            and self.tokenizer_metrics_allowed_customer_labels
        ):
            raise ValueError(
                "Please set --tokenizer-metrics-custom-labels-header when setting --tokenizer-metrics-allowed-customer-labels."
            )

        # _handle_deterministic_inference
        if self.enable_deterministic_inference:
            import importlib

            if not importlib.util.find_spec("batch_invariant_ops"):
                raise ValueError(
                    "batch_invariant_ops is not installed. Please install it from https://github.com/thinking-machines-lab/batch_invariant_ops/."
                )

            if self.attention_backend != AttentionBackend.FA3:
                self.disable_radix_cache = True
                logger.warning(
                    "Currently radix cache is disabled for deterministic inference. It will be supported in the future."
                )
            if self.attention_backend not in DETERMINISTIC_ATTENTION_BACKEND_CHOICES:
                raise ValueError(
                    f"Currently only {DETERMINISTIC_ATTENTION_BACKEND_CHOICES} attention backends are supported for deterministic inference."
                )

        return self


def print_deprecated_warning(message: str):
    logger.warning(f"\033[33m{message}\033[0m")
