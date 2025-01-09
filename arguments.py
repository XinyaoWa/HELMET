import argparse
import yaml
import ast
import os
import logging 
logger = logging.getLogger(__name__)

def parse_arguments():
    parser = argparse.ArgumentParser(description="evaluation on downstream tasks")
    parser.add_argument("--config", type=str, default=None, help="path to config file")
    parser.add_argument("--device", "-d", type=str, choices=["hpu","cpu","cuda"], help="Device to run", default="hpu")
    parser.add_argument("--tag", type=str, default="eval", help="tag to add to the output file")

    # model setting
    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument("--use_vllm", action="store_true", help="whether to use vllm engine")

    # data paths
    parser.add_argument("--datasets", type=str, default=None)
    parser.add_argument("--demo_files", type=str, default=None)
    parser.add_argument("--test_files", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None, help="path to save the predictions")
    parser.add_argument("--overwrite", action="store_true", help="whether to the saved file")
    parser.add_argument("--max_test_samples", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_depths", type=int, default=10)

    # dataset specific settings
    parser.add_argument("--popularity_threshold", type=int, default=3)

    # evaluation settings
    parser.add_argument("--shots", type=int, default=5, help="total number of demos (encoder + decoder)")
    parser.add_argument("--input_max_length", type=str, default='8192', help="the maximum number of tokens of the input, we truncate the end of the context; can be separated by comma to match the specified datasets")

    # generation settings
    parser.add_argument("--do_sample", type=ast.literal_eval, choices=[True, False], default=False, help="whether to use sampling (false is greedy), overwrites temperature")
    parser.add_argument("--generation_max_length", type=str, default='10', help="max number of tokens to generate, can be separated by comma to match the specified datasets")
    parser.add_argument("--generation_min_length", type=int, default=0, help="min number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0, help="generation temperature")
    parser.add_argument("--top_p", type=float, default=1.0, help="top-p parameter for nucleus sampling")
    parser.add_argument("--stop_newline", type=ast.literal_eval, choices=[True, False], default=False, help="whether to stop generation at newline")

    # model specific settings
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--no_cuda", action="store_true", help="disable cuda")
    parser.add_argument("--no_bf16", action="store_true", help="disable bf16 and use fp32")
    parser.add_argument("--no_torch_compile", action="store_true", help="disable cuda")
    parser.add_argument("--use_chat_template", type=ast.literal_eval, choices=[True, False], default=False, help="whether to use chat template")
    parser.add_argument("--rope_theta", type=int, default=None, help="override rope theta")

    # misc
    parser.add_argument("--debug", action="store_true", help="for debugging")
    parser.add_argument("--count_tokens", action="store_true", help="instead of running generation, just count the number of tokens (only for HF models not API)")

    #############below is all for optimum-habana#############
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Whether to perform generation in bf16 precision.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Number of tokens to generate.")
    parser.add_argument(
        "--max_input_tokens",
        type=int,
        default=0,
        help="If > 0 then pad and truncate the input sequences to this specified length of tokens. \
            if == 0, then truncate to 16 (original default) \
            if < 0, then do not truncate, use full input prompt",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Input batch size.")
    parser.add_argument("--warmup", type=int, default=3, help="Number of warmup iterations for benchmarking.")
    parser.add_argument("--n_iterations", type=int, default=5, help="Number of inference iterations for benchmarking.")
    parser.add_argument("--local_rank", type=int, default=0, metavar="N", help="Local process rank.")
    parser.add_argument(
        "--use_kv_cache",
        action="store_true",
        help="Whether to use the key/value cache for decoding. It should speed up generation.",
    )
    parser.add_argument(
        "--use_hpu_graphs",
        action="store_true",
        help="Whether to use HPU graphs or not. Using HPU graphs should give better latencies.",
    )
    parser.add_argument(
        "--dataset_name",
        default=None,
        type=str,
        help="Optional argument if you want to assess your model on a given dataset of the HF Hub.",
    )
    parser.add_argument(
        "--column_name",
        default=None,
        type=str,
        help="If `--dataset_name` was given, this will be the name of the column to use as prompts for generation.",
    )
    parser.add_argument(
        "--num_beams",
        default=1,
        type=int,
        help="Number of beams used for beam search generation. 1 means greedy search will be performed.",
    )
    parser.add_argument(
        "--top_k",
        default=None,
        type=int,
        help="Size of candidate set used for re-ranking in contrastive search. top_k > 1 enables contrastive search.",
    )
    parser.add_argument(
        "--penalty_alpha",
        default=None,
        type=float,
        help="Degeneration penalty for contrastive search. penalty_alpha > 0 enables contrastive search.",
    )
    parser.add_argument(
        "--trim_logits",
        action="store_true",
        help="Calculate logits only for the last token to save memory in the first step.",
    )
    parser.add_argument(
        "--profiling_warmup_steps",
        default=0,
        type=int,
        help="Number of steps to ignore for profiling.",
    )
    parser.add_argument(
        "--profiling_steps",
        default=0,
        type=int,
        help="Number of steps to capture for profiling.",
    )
    parser.add_argument(
        "--profiling_record_shapes",
        action="store_true",
        help="Record shapes when enabling profiling.",
    )
    parser.add_argument(
        "--bad_words",
        default=None,
        type=str,
        nargs="+",
        help="Optional argument list of words that are not allowed to be generated.",
    )
    parser.add_argument(
        "--force_words",
        default=None,
        type=str,
        nargs="+",
        help="Optional argument list of words that must be generated.",
    )
    parser.add_argument(
        "--assistant_model",
        default=None,
        type=str,
        help="Optional argument to give a path to a draft/assistant model for assisted decoding.",
    )
    parser.add_argument(
        "--peft_model",
        default=None,
        type=str,
        help="Optional argument to give a path to a PEFT model.",
    )
    parser.add_argument("--num_return_sequences", type=int, default=1)
    parser.add_argument(
        "--token",
        default=None,
        type=str,
        help="The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
        "generated when running `huggingface-cli login` (stored in `~/.huggingface`).",
    )
    parser.add_argument(
        "--model_revision",
        default="main",
        type=str,
        help="The specific model version to use (can be a branch name, tag name or commit id).",
    )
    parser.add_argument(
        "--attn_softmax_bf16",
        action="store_true",
        help="Whether to run attention softmax layer in lower precision provided that the model supports it and "
        "is also running in lower precision.",
    )
    parser.add_argument(
        "--bucket_size",
        default=-1,
        type=int,
        help="Bucket size to maintain static shapes. If this number is negative (default is -1) \
            then we use `shape = prompt_length + max_new_tokens`. If a positive number is passed \
            we increase the bucket in steps of `bucket_size` instead of allocating to max (`prompt_length + max_new_tokens`).",
    )
    parser.add_argument(
        "--bucket_internal",
        action="store_true",
        help="Split kv sequence into buckets in decode phase. It improves throughput when max_new_tokens is large.",
    )
    parser.add_argument(
        "--dataset_max_samples",
        default=-1,
        type=int,
        help="If a negative number is passed (default = -1) perform inference on the whole dataset, else use only `dataset_max_samples` samples.",
    )
    parser.add_argument(
        "--limit_hpu_graphs",
        action="store_true",
        help="Skip HPU Graph usage for first token to save memory",
    )
    parser.add_argument(
        "--show_graphs_count",
        action="store_true",
        help="Show statistics of HPU graph compilation.",
    )
    parser.add_argument(
        "--reuse_cache",
        action="store_true",
        help="Whether to reuse key/value cache for decoding. It should save memory.",
    )
    parser.add_argument("--verbose_workers", action="store_true", help="Enable output from non-master workers")
    parser.add_argument(
        "--simulate_dyn_prompt",
        default=None,
        type=int,
        nargs="*",
        help="If empty, static prompt is used. If a comma separated list of integers is passed, we warmup and use those shapes for prompt length.",
    )
    parser.add_argument(
        "--reduce_recompile",
        action="store_true",
        help="Preprocess on cpu, and some other optimizations. Useful to prevent recompilations when using dynamic prompts (simulate_dyn_prompt)",
    )
    parser.add_argument(
        "--use_flash_attention",
        action="store_true",
        help="Whether to enable Habana Flash Attention, provided that the model supports it.",
    )
    parser.add_argument(
        "--flash_attention_recompute",
        action="store_true",
        help="Whether to enable Habana Flash Attention in recompute mode on first token generation. This gives an opportunity of splitting graph internally which helps reduce memory consumption.",
    )
    parser.add_argument(
        "--flash_attention_causal_mask",
        action="store_true",
        help="Whether to enable Habana Flash Attention in causal mode on first token generation.",
    )
    parser.add_argument(
        "--flash_attention_fast_softmax",
        action="store_true",
        help="Whether to enable Habana Flash Attention in fast softmax mode.",
    )
    parser.add_argument(
        "--book_source",
        action="store_true",
        help="Whether to use project Guttenberg books data as input. Usefull for testing large sequence lenghts.",
    )
    parser.add_argument(
        "--torch_compile",
        action="store_true",
        help="Whether to use torch compiled model or not.",
    )
    parser.add_argument(
        "--ignore_eos",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Whether to disable stopping with eos token when calling `generate`. --no-ignore_eos to disable it",
    )
    parser.add_argument(
        "--const_serialization_path",
        "--csp",
        type=str,
        help="Path to serialize const params. Const params will be held on disk memory instead of being allocated on host memory.",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Whether to trust the execution of code from datasets/models defined on the Hub. This option should only be set to `True` for repositories you trust and in which you have read the code, as it will execute code present on the Hub on your local machine.",
    )
    parser.add_argument(
        "--parallel_strategy",
        type=str,
        choices=["tp", "ep", "none"],  # Add other strategies as needed
        default="none",
        help="Run multi card with the specified parallel strategy. Choices are 'tp' for Tensor Parallel Strategy or 'ep' for Expert Parallel Strategy or 'none'.",
    )
    parser.add_argument(
        "--input_embeds",
        action="store_true",
        help="Whether to enable inputs_embeds or not.",
    )
    parser.add_argument(
        "--sdp_on_bf16", action="store_true", help="Allow pyTorch to use reduced precision in the SDPA math backend"
    )

    parser.add_argument(
        "--buckets",
        type=int,
        nargs="+",
        help="Input length buckets to use with static_shapes",
        default=[16, 32, 64, 128, 189, 284, 384],
    )

    quant_parser_group = parser.add_mutually_exclusive_group()
    quant_parser_group.add_argument(
        "--load_quantized_model_with_autogptq",
        action="store_true",
        help="Load an AutoGPTQ quantized checkpoint using AutoGPTQ.",
    )
    quant_parser_group.add_argument(
        "--disk_offload",
        action="store_true",
        help="Whether to enable device map auto. In case no space left on cpu, weights will be offloaded to disk.",
    )
    quant_parser_group.add_argument(
        "--load_quantized_model_with_inc",
        action="store_true",
        help="Load a Huggingface quantized checkpoint using INC.",
    )
    quant_parser_group.add_argument(
        "--local_quantized_inc_model_path",
        type=str,
        default=None,
        help="Path to neural-compressor quantized model, if set, the checkpoint will be loaded.",
    )

    args = parser.parse_args()
    config = yaml.safe_load(open(args.config)) if args.config is not None else {}
    parser.set_defaults(**config)
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = f"output/{os.path.basename(args.model_name_or_path)}"

    if args.rope_theta is not None:
        args.output_dir = args.output_dir + f"-override-rope{args.rope_theta}"

    if args.torch_compile:
        args.use_hpu_graphs = False

    if not args.use_hpu_graphs:
        args.limit_hpu_graphs = False

    if args.use_flash_attention and not args.flash_attention_fast_softmax:
        args.flash_attention_fast_softmax = True

    args.quant_config = os.getenv("QUANT_CONFIG", "")
    if args.quant_config and args.load_quantized_model_with_autogptq:
        raise RuntimeError("Setting both quant_config and load_quantized_model_with_autogptq is unsupported. ")

    if args.quant_config == "" and args.disk_offload:
        logger.warning(
            "`--disk_offload` was tested only with fp8, it may not work with full precision. If error raises try to remove the --disk_offload flag."
        )

    return args
