import time
import torch
import argparse
from sglang import Engine
from transformers import AutoTokenizer

def is_musa():
    """Check if the current environment supports MUSA (Moore Threads Architecture)."""
    return hasattr(torch, "musa") and torch.musa.is_available()

def test_offline_infer_perf(args):
    """
    Evaluate Flash Attention performance using sglang.Engine for text generation.

    Args:
        args: Parsed command-line arguments containing test configuration.
    """
    # Validate the validity of the specified attention backend
    valid_backends = ["flashinfer", "fa3", "fa4", "triton", "trtllm_mla", "cutlass_mla"]
    if args.attention_backend is not None and args.attention_backend not in valid_backends:
        raise ValueError(f"Invalid attention backend: {args.attention_backend}. Valid options are: {valid_backends}")

    # Set attention backend status for logging
    attention_backend = args.attention_backend
    fa_status = attention_backend if attention_backend else "disabled"
    print(f"Loading model: {args.model_path} (Attention backend: {fa_status})")

    disable_radix_cache = args.disable_radix_cache
    disable_cuda_graph = args.disable_cuda_graph
    disable_overlap_schedule = args.disable_overlap_schedule
    cuda_graph_max_bs = args.cuda_graph_max_bs

    # # Initialize configuration from command-line arguments, with higher priority for MUSA environment defaults
    # if is_musa():
    #     if not disable_overlap_schedule:
    #         print("Please set disable_overlap_schedule -> True for MUSA")
    #         disable_overlap_schedule = True

    # Extract parameters passed from command line
    tp_size = args.tp_size
    mem_fraction_static = args.mem_fraction_static
    chunked_prefill_size = args.chunked_prefill_size

    # Initialize SGLang Engine with the configured parameters
    engine = Engine(
        model_path=args.model_path,
        tp_size=tp_size,
        mem_fraction_static=mem_fraction_static,  # Use static memory fraction configured from command line
        trust_remote_code=True,
        attention_backend=attention_backend,
        chunked_prefill_size=chunked_prefill_size,  # Use chunked prefill size configured from command line
        disable_radix_cache=disable_radix_cache,
        disable_cuda_graph=disable_cuda_graph,
        disable_overlap_schedule=disable_overlap_schedule,
        cuda_graph_max_bs=cuda_graph_max_bs,
    )
    print("Model loaded successfully")

    # Initialize tokenizer and set pad token if it's missing (use eos_token as fallback)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

    # Prepare batched prompts for inference testing
    if not args.prompt_text:
        # Default long prompt to stress test the attention computation performance
        args.prompt_text = (
            "Please detailedly describe the development history, core technological breakthroughs, "
            "and future application prospects of artificial intelligence in the field of natural language processing. "
            * 5
        )

    prompts = [args.prompt_text for _ in range(args.batch_size)]

    # Validate and get the input sequence length by tokenizing the prompt
    input_tokens = tokenizer(args.prompt_text, return_tensors="pt")
    input_seq_len = input_tokens.input_ids.shape[1]
    print(f"\nTest Configuration:")
    print(f"  Single prompt token length: {input_seq_len}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Max new tokens per sample: {args.max_new_tokens}")
    print(f"  Attention backend: {fa_status}")
    print(f"  Tensor Parallel (tp_size): {tp_size}")
    print(f"  Static Memory Fraction (mem_fraction_static): {mem_fraction_static}")
    print(f"  Chunked Prefill Size (chunked_prefill_size): {chunked_prefill_size}")
    print(f"  Warm-up runs: {args.warmup_runs} | Test runs: {args.test_runs}")
    print(f"  disable_radix_cache: {disable_radix_cache}")
    print(f"  disable_cuda_graph: {disable_cuda_graph}")
    print(f"  disable_overlap_schedule: {disable_overlap_schedule}")
    print(f"  cuda_graph_max_bs: {cuda_graph_max_bs}")

    # Define sampling parameters for text generation
    sampling_params = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": 0.7,
        "top_p": 0.95,
    }

    # Warm-up phase (eliminate the interference of initialization and CUDA cache)
    print(f"\nStarting warm-up runs ({args.warmup_runs} runs)...")
    for _ in range(args.warmup_runs):
        engine.generate(prompts, sampling_params=sampling_params)
    torch.cuda.synchronize()
    print("Warm-up completed")

    # Formal performance testing phase
    print(f"\nStarting performance testing ({args.test_runs} runs)...")
    total_time = 0.0
    total_tokens = 0.0

    for run_idx in range(args.test_runs):
        # Measure inference time with CUDA synchronization to ensure accuracy
        start_time = time.perf_counter()
        engine.generate(prompts, sampling_params=sampling_params)
        torch.cuda.synchronize()
        end_time = time.perf_counter()

        # Calculate metrics for the current run
        run_time = end_time - start_time
        run_tokens = args.batch_size * args.max_new_tokens

        # Accumulate total time and total generated tokens
        total_time += run_time
        total_tokens += run_tokens

        # Print per-run performance results
        throughput = run_tokens / run_time
        print(f"  Run {run_idx+1}/{args.test_runs}: {run_time:.4f}s | Throughput: {throughput:.2f} tokens/s")

    # Calculate and print summary performance metrics
    avg_time = total_time / args.test_runs
    avg_throughput = total_tokens / total_time

    print(f"\n===== Attention Backend ({fa_status}) Performance Summary =====")
    print(f"Average run time: {avg_time:.4f} seconds")
    print(f"Average throughput: {avg_throughput:.2f} tokens/second")
    print(f"Total generated tokens: {total_tokens:,}")
    print(f"Used Tensor Parallel Size (tp_size): {tp_size}")
    print(f"Used Static Memory Fraction: {mem_fraction_static}")
    print(f"Used Chunked Prefill Size: {chunked_prefill_size}")

    # Clean up allocated resources to release GPU memory
    del engine
    torch.cuda.empty_cache()
    print("\nTest completed | Resources cleaned up")


def parse_arguments():
    """Parse and validate command-line arguments for the performance test."""
    parser = argparse.ArgumentParser(
        description="Flash Attention Performance Test for LLMs using SGLang",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Core model configuration
    parser.add_argument(
        "--model_path",
        type=str,
        default="/mnt/si001226c4pl/default/triton_models/Qwen3-8B/",
        help="Path/name of pre-trained model (must support the specified attention backend)"
    )

    # Tensor parallelism configuration
    parser.add_argument(
        "--tp_size",
        type=int,
        default=4,
        choices=[1, 2, 4, 8, 16],
        help="Tensor parallelism size (number of GPUs used for tensor parallelism). Valid values: 1, 2, 4, 8, 16"
    )

    # Static memory fraction configuration (reserved for model weights)
    parser.add_argument(
        "--mem_fraction_static",
        type=float,
        default=0.75,
        help="Fraction of GPU memory to reserve for static model weights (range: 0.0 ~ 1.0). Higher values reduce dynamic memory overhead."
    )

    # Chunked prefill size configuration for long sequence optimization
    parser.add_argument(
        "--chunked_prefill_size",
        type=int,
        default=-1,
        help="Chunk size for the prefill phase (default: -1, disable chunked prefill). Use positive integers for large sequence prefill optimization."
    )

    # Attention backend configuration
    parser.add_argument(
        "--attention_backend",
        type=str,
        default="fa3",
        choices=["flashinfer", "fa3", "fa4", "triton", "trtllm_mla", "cutlass_mla", None],
        help="Specify the attention backend to use. Valid options: flashinfer, fa3, fa4, triton, trtllm_mla, cutlass_mla"
    )

    # Test prompt configuration
    parser.add_argument(
        "--prompt_text",
        type=str,
        default=None,
        help="Custom test prompt text (default: a long NLP AI-related prompt for stress testing)"
    )

    # Batch and generation settings
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Inference batch size (number of samples per inference run)"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Maximum number of new tokens to generate per sample"
    )

    # Test execution settings
    parser.add_argument(
        "--warmup_runs",
        type=int,
        default=3,
        help="Number of warm-up runs to eliminate initialization and cache overhead"
    )
    parser.add_argument(
        "--test_runs",
        type=int,
        default=10,
        help="Number of formal test runs for performance averaging"
    )

    # Optimization parameters configuration
    parser.add_argument(
        "--disable_radix_cache",
        action='store_true',
        default=False,
        help="Whether to disable radix cache (MUSA environment default: True, other environments default: False)"
    )
    parser.add_argument(
        "--disable_cuda_graph",
        action='store_true',
        default=False,
        help="Whether to disable CUDA Graph optimization (MUSA environment default: True, other environments default: False)"
    )
    parser.add_argument(
        "--disable_overlap_schedule",
        action='store_true',
        default=False,
        help="Whether to disable CPU-GPU overlap scheduling (MUSA environment default: True, other environments default: False)"
    )
    parser.add_argument(
        "--cuda_graph_max_bs",
        type=int,
        default=None,
        help="Maximum batch size supported by CUDA Graph (MUSA environment default: 128, other environments default: None)"
    )

    return parser.parse_args()


if __name__ == "__main__":
    # Parse command-line arguments and execute the performance test
    args = parse_arguments()
    test_offline_infer_perf(args)
