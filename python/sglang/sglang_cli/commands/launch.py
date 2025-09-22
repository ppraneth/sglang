from typing import List, Literal, Optional, Self, Union

import pydantic
import typer

from sglang.sglang_cli.config import LoadFormat, ServerConfig, print_deprecated_warning
from sglang.srt.entrypoints.http_server import launch_server

app = typer.Typer()


@app.command(help="Launch the SGLang inference server.")
def launch(
    model_path: str = typer.Argument(
        ...,
        "--model-path",
        "--model",
        help="The path of the model weights. This can be a local folder or a Hugging Face repo ID.",
    ),
    tokenizer_path: str = typer.Option(
        None,
        "--tokenizer-path",
        help="The path of the tokenizer. Defaults to model_path.",
    ),
    tokenizer_mode: Literal["auto", "slow"] = typer.Option(
        "auto",
        "--tokenizer-mode",
        help="Tokenizer mode. 'auto' will use the fast tokenizer if available, and 'slow' will always use the slow tokenizer.",
    ),
    remote_instance_weight_loader_seed_instance_ip: Optional[str] = typer.Option(
        None,
        "--remote-instance-weight-loader-seed-instance-ip",
        help="The ip of the seed instance for loading weights from remote instance.",
    ),
    remote_instance_weight_loader_seed_instance_service_port: Optional[
        int
    ] = typer.Option(
        None,
        "--remote-instance-weight-loader-seed-instance-service-port",
        help="The service port of the seed instance for loading weights from remote instance.",
    ),
    remote_instance_weight_loader_send_weights_group_ports: Optional[
        str
    ] = typer.Option(
        None,
        "--remote-instance-weight-loader-send-weights-group-ports",
        help="The communication group ports for loading weights from remote instance.",
    ),
    tokenizer_worker_num: int = typer.Option(
        1, "--tokenizer-worker-num", help="The worker num of the tokenizer manager."
    ),
    skip_tokenizer_init: bool = typer.Option(
        False,
        "--skip-tokenizer-init",
        help="If set, skip init tokenizer and pass input_ids in generate request.",
    ),
    load_format: LoadFormat = typer.Option(
        LoadFormat.AUTO,
        "--load-format",
        help="The format of the model weights to load. "
        '"auto" will try to load the weights in the safetensors format '
        "and fall back to the pytorch bin format if safetensors format "
        "is not available. "
        '"pt" will load the weights in the pytorch bin format. '
        '"safetensors" will load the weights in the safetensors format. '
        '"npcache" will load the weights in pytorch format and store '
        "a numpy cache to speed up the loading. "
        '"dummy" will initialize the weights with random values, '
        "which is mainly for profiling."
        '"gguf" will load the weights in the gguf format. '
        '"bitsandbytes" will load the weights using bitsandbytes '
        "quantization."
        '"layered" loads weights layer by layer so that one can quantize a '
        "layer before loading another to make the peak memory envelope "
        "smaller.",
    ),
    model_loader_extra_config: str = typer.Option(
        "{}",
        "--model-loader-extra-config",
        help="Extra config for model loader. "
        "This will be passed to the model loader corresponding to the chosen load_format.",
    ),
    trust_remote_code: bool = typer.Option(
        False,
        "--trust-remote-code",
        help="Whether or not to allow for custom models defined on the Hub in their own modeling files.",
    ),
    context_length: Optional[int] = typer.Option(
        None,
        "--context-length",
        help="The model's maximum context length. Defaults to None (will use the value from the model's config.json instead).",
    ),
    is_embedding: bool = typer.Option(
        False, "--is-embedding", help="Whether to use a CausalLM as an embedding model."
    ),
    enable_multimodal: Optional[bool] = typer.Option(
        None,
        "--enable-multimodal",
        help="Enable the multimodal functionality for the served model. If the model being served is not multimodal, nothing will happen",
    ),
    revision: Optional[bool] = typer.Option(
        None,
        "--revision",
        help="The specific model version to use. It can be a branch "
        "name, a tag name, or a commit id. If unspecified, will use "
        "the default version.",
    ),
    model_impl: str = typer.Option(
        "auto",
        "--model-impl",
        help="Which implementation of the model to use.\n\n"
        '* "auto" will try to use the SGLang implementation if it exists '
        "and fall back to the Transformers implementation if no SGLang "
        "implementation is available.\n"
        '* "sglang" will use the SGLang model implementation.\n'
        '* "transformers" will use the Transformers model '
        "implementation.\n",
    ),
    # HTTP server
    host: str = typer.Option(
        "127.0.0.1",
        "--host",
        help="The host of the HTTP server.",
    ),
    port: int = typer.Option(30000, "--port", help="The port of the HTTP server."),
    skip_server_warmup: bool = typer.Option(
        False, "--skip-server-warmup", help="If set, skip warmup."
    ),
    warmups: Optional[str] = typer.Option(
        None,
        "--warmups",
        help="Specify custom warmup functions (csv) to run before server starts eg. --warmups=warmup_name1,warmup_name2 "
        "will run the functions `warmup_name1` and `warmup_name2` specified in warmup.py before the server starts listening for requests",
    ),
    nccl_port: Optional[int] = typer.Option(
        None,
        "--nccl-port",
        help="The port for NCCL distributed environment setup. Defaults to a random port.",
    ),
):
    cli_args = {k: v for k, v in locals().items() if v is not None}

    # Handle aliases manually since Typer doesn't pass them to Pydantic
    if "tensor_parallel_size" in cli_args:
        cli_args["tp_size"] = cli_args.pop("tensor_parallel_size")

    try:
        server_config = ServerConfig(**cli_args)
        server_config = server_config.complete_and_validate_config()

        print("🚀 Starting SGLang server with the following configuration:")
        print(server_config.model_dump_json(indent=2, exclude_none=True))

        launch_server(server_config)

    except pydantic.ValidationError as e:
        print(f"❌ Configuration Error:\n{e}")
        raise typer.Exit(code=1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise typer.Exit(code=1)
