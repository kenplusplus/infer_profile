import time
import torch
import argparse
from sglang import Engine
from transformers import AutoTokenizer


def test_flash_attention_performance(args):
    """
    Evaluate Flash Attention performance using sglang.Engine for text generation

    Args:
        args: Parsed command-line arguments containing test configuration
    """
    # Validate attention backend value
    valid_backends = ["flashinfer", "fa3", "fa4", "triton", "trtllm_mla", "cutlass_mla"]
    if args.attention_backend is not None and args.attention_backend not in valid_backends:
        raise ValueError(f"Invalid attention backend: {args.attention_backend}. Valid options are: {valid_backends}")

    # Set attention backend status
    attention_backend = args.attention_backend
    fa_status = attention_backend if attention_backend else "disabled"
    print(f"Loading model: {args.model_path} (Attention backend: {fa_status})")

    engine = Engine(
        model_path=args.model_path,
        tp_size=1,
        trust_remote_code=True,
        attention_backend=attention_backend,  # Use specified backend or None to disable
        dtype="auto",
        quantization=None,
        kv_cache_dtype="auto",
    )
    print("Model loaded successfully")

    # Initialize tokenizer (set pad token if missing)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

    # Prepare batched prompts
    if not args.prompt_text:
        # Default long prompt to stress test attention computation
        args.prompt_text = (
            "Please detailedly describe the development history, core technological breakthroughs, "
            "and future application prospects of artificial intelligence in the field of natural language processing. "
            * 5
        )

    prompts = [args.prompt_text for _ in range(args.batch_size)]

    # Validate input sequence length
    input_tokens = tokenizer(args.prompt_text, return_tensors="pt")
    input_seq_len = input_tokens.input_ids.shape[1]
    print(f"\nTest Configuration:")
    print(f"  Single prompt token length: {input_seq_len}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Max new tokens per sample: {args.max_new_tokens}")
    print(f"  Attention backend: {fa_status}")
    print(f"  Warm-up runs: {args.warmup_runs} | Test runs: {args.test_runs}")

    # Sampling parameters for generation
    sampling_params = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": 0.7,
        "top_p": 0.95,
    }

    # Warm-up phase (eliminate initialization/CUDA cache interference)
    print(f"\nStarting warm-up runs ({args.warmup_runs} runs)...")
    for _ in range(args.warmup_runs):
        engine.generate(prompts, sampling_params=sampling_params)
    torch.cuda.synchronize()
    print("Warm-up completed")

    # Formal performance testing
    print(f"\nStarting performance testing ({args.test_runs} runs)...")
    total_time = 0.0
    total_tokens = 0.0

    for run_idx in range(args.test_runs):
        # Measure inference time with CUDA synchronization
        start_time = time.perf_counter()
        engine.generate(prompts, sampling_params=sampling_params)
        torch.cuda.synchronize()
        end_time = time.perf_counter()

        # Calculate run metrics
        run_time = end_time - start_time
        run_tokens = args.batch_size * args.max_new_tokens

        # Accumulate totals
        total_time += run_time
        total_tokens += run_tokens

        # Print per-run results
        throughput = run_tokens / run_time
        print(f"  Run {run_idx+1}/{args.test_runs}: {run_time:.4f}s | Throughput: {throughput:.2f} tokens/s")

    # Calculate and print summary metrics
    avg_time = total_time / args.test_runs
    avg_throughput = total_tokens / total_time

    print(f"\n===== Attention Backend ({fa_status}) Performance Summary =====")
    print(f"Average run time: {avg_time:.4f} seconds")
    print(f"Average throughput: {avg_throughput:.2f} tokens/second")
    print(f"Total generated tokens: {total_tokens:,}")

    # Clean up resources
    del engine
    torch.cuda.empty_cache()
    print("\nTest completed | Resources cleaned up")


def parse_arguments():
    """Parse command-line arguments for performance testing"""
    parser = argparse.ArgumentParser(
        description="Flash Attention Performance Test for LLMs using SGLang",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Core model configuration
    parser.add_argument(
        "--model_path",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Path/name of pre-trained model (must support specified attention backend)"
    )

    # Attention backend configuration (replace original enable_flash_attention)
    parser.add_argument(
        "--attention_backend",
        type=str,
        default="flashinfer",
        choices=["flashinfer", "fa3", "fa4", "triton", "trtllm_mla", "cutlass_mla", None],
        help="Specify attention backend to use (default: None/disabled). Valid options: flashinfer, fa3, fa4, triton, trtllm_mla, cutlass_mla"
    )

    # Test prompt configuration
    parser.add_argument(
        "--prompt_text",
        type=str,
        default=None,
        help="Custom test prompt text (default: long NLP AI prompt)"
    )

    # Batch and generation settings
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Inference batch size"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Maximum new tokens generated per sample"
    )

    # Test execution settings
    parser.add_argument(
        "--warmup_runs",
        type=int,
        default=3,
        help="Number of warm-up runs to eliminate initialization overhead"
    )
    parser.add_argument(
        "--test_runs",
        type=int,
        default=10,
        help="Number of formal test runs for performance averaging"
    )

    return parser.parse_args()


if __name__ == "__main__":
    # Parse command-line arguments and run test
    args = parse_arguments()
    test_flash_attention_performance(args)
