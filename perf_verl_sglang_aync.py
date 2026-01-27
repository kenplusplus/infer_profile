import time
import json
import numpy as np
from sglang import Engine
import os
import gc
import torch
import asyncio
import argparse  # Add command line argument parsing

from sglang.srt.server_args import ServerArgs

# ===================== Global Configuration =====================
# Test root directory
ROOT_OUTPUT_DIR = "./sglang_perf_test_async/"
# Performance test parameters (async batch test parameters) - set as default values, actual values are overridden by command line arguments
PROMPT_LENGTH = 512             # Input prompt length (number of tokens), not exposed to command line configuration temporarily

# Create root directory
os.makedirs(ROOT_OUTPUT_DIR, exist_ok=True)
GLOBAL_LOG_FILE = os.path.join(ROOT_OUTPUT_DIR, "global_test_log.txt")

def is_musa():
    return hasattr(torch, "musa") and torch.musa.is_available()

# ===================== General Utility Functions =====================
def log_info(msg):
    """Global logging function: record logs and print"""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    log_msg = f"[{timestamp}] {msg}"
    print(log_msg)
    with open(GLOBAL_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(log_msg + "\n")

def generate_test_prompt(length):
    """Generate test prompt with specified length"""
    return " ".join(["test"] * length)

def cleanup_engine(engine):
    """Clean up Engine resources and release GPU memory"""
    if engine is not None:
        engine.shutdown()
        del engine

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    elif torch.musa.is_available():
        torch.musa.empty_cache()
        torch.musa.ipc_collect()

    gc.collect()
    log_info("Engine resources cleaned up successfully")

# ===================== Async Performance Test Core Functions =====================
async def async_send_batch_requests(engine, prompt_batch, prompt_id_batch, generation_length):
    """
    Send batch requests asynchronously
    :param engine: SGLang Engine instance
    :param prompt_batch: List of batch prompts
    :param prompt_id_batch: Corresponding prompt ID list
    :param generation_length: Generated text length (number of tokens)
    :return: List of batch request results
    """
    sampling_params = {
        'max_new_tokens': generation_length,
        'temperature': 0.8,
        'top_p': 0.95,
        'repetition_penalty': 1.0,
        'top_k': 100,
        'presence_penalty': 0.0
    }
    
    try:
        batch_start_time = time.time()
        # Core: Asynchronous batch generation (Non-streaming Asynchronous Generation)
        outputs = await engine.async_generate(prompt_batch, sampling_params)
        batch_end_time = time.time()
        
        batch_results = []
        for idx, (prompt_id, output) in enumerate(zip(prompt_id_batch, outputs)):
            # Calculate single request latency (average total batch time, or use built-in output latency)
            single_latency = (batch_end_time - batch_start_time) / len(prompt_batch)
            
            # Compatible with return value formats
            if hasattr(output, 'text'):
                generated_text = output.text
            elif isinstance(output, dict) and 'text' in output:
                generated_text = output['text']
            elif isinstance(output, dict) and 'outputs' in output:
                generated_text = output['outputs'][0]['text']
            else:
                raise ValueError(f"Unsupported return value format: {type(output)}")
            
            generated_tokens = len(generated_text.strip().split()) if generated_text else 0
            
            batch_results.append({
                "prompt_id": prompt_id,
                "status": "success",
                "latency": single_latency,
                "generated_tokens": generated_tokens,
                "tokens_per_second": generated_tokens / single_latency if single_latency > 0 else 0
            })
        return batch_results
    except Exception as e:
        # Mark all prompts as error when batch request fails
        return [{
            "prompt_id": prompt_id,
            "status": "error",
            "error": str(e),
            "latency": 0,
            "generated_tokens": 0,
            "tokens_per_second": 0
        } for prompt_id in prompt_id_batch]

async def run_async_performance_test(test_mode, runtime_config, model_path, num_prompts, prompt_length, generation_length, async_batch_size):
    """
    Execute asynchronous performance test (Non-streaming Asynchronous Generation)
    :param test_mode: Test mode (original/optimized)
    :param runtime_config: Corresponding RuntimeConfig configuration
    :param model_path: Model path
    :param num_prompts: Total number of test requests
    :param prompt_length: Input prompt length
    :param generation_length: Generated text length
    :param async_batch_size: Asynchronous batch size
    :return: Test result dictionary
    """
    # Initialize test directory
    output_dir = os.path.join(ROOT_OUTPUT_DIR, test_mode)
    os.makedirs(output_dir, exist_ok=True)
    metrics_file = os.path.join(output_dir, "metrics.json")
    csv_file = os.path.join(output_dir, "performance.csv")
    log_info(f"\n========== Start {test_mode.upper()} parameter async test ==========")

    # Initialize Engine
    log_info(f"Initializing SGLang Engine with {test_mode} parameters...")
    try:
        engine = Engine(
            model_path=model_path,
            server_args=runtime_config,
        )
    except Exception as e:
        log_info(f"Engine initialization failed: {str(e)}")
        return None

    # Warm up (avoid first request latency affecting test results)
    log_info(f"{test_mode} Engine warming up...")
    warmup_prompts = [" ".join(["test"] * 10) for _ in range(2)]  # Batch warmup
    await engine.async_generate(
        warmup_prompts,
        {'max_new_tokens': 10}
    )
    log_info(f"{test_mode} Engine warmup completed")

    # Generate all test prompts and corresponding IDs
    all_prompts = [generate_test_prompt(prompt_length) for _ in range(num_prompts)]
    all_prompt_ids = list(range(num_prompts))

    # Split into batches (group by ASYNC_BATCH_SIZE)
    prompt_batches = [
        all_prompts[i:i + async_batch_size] 
        for i in range(0, num_prompts, async_batch_size)
    ]
    prompt_id_batches = [
        all_prompt_ids[i:i + async_batch_size] 
        for i in range(0, num_prompts, async_batch_size)
    ]

    # Execute async test
    log_info(f"{test_mode} async test started: {num_prompts} requests, split into {len(prompt_batches)} batches, {async_batch_size} requests per batch")
    total_start = time.time()
    all_results = []

    # Execute async requests batch by batch (can be modified to execute multiple batches concurrently as needed)
    for batch_idx, (prompt_batch, prompt_id_batch) in enumerate(zip(prompt_batches, prompt_id_batches)):
        log_info(f"Processing batch {batch_idx + 1}/{len(prompt_batches)}, containing {len(prompt_batch)} prompts")
        batch_results = await async_send_batch_requests(engine, prompt_batch, prompt_id_batch, generation_length)
        all_results.extend(batch_results)

    # Calculate total time consumption
    total_end = time.time()
    total_time = total_end - total_start

    # Statistics results
    success_results = [r for r in all_results if r["status"] == "success"]
    total_tokens = sum([r["generated_tokens"] for r in success_results])

    # Core performance metrics calculation
    success_rate = len(success_results) / num_prompts * 100 if num_prompts > 0 else 0
    avg_latency = np.mean([r["latency"] for r in success_results]) if success_results else 0
    avg_throughput = np.mean([r["tokens_per_second"] for r in success_results]) if success_results else 0
    total_throughput = total_tokens / total_time if total_time > 0 else 0

    # Build metrics dictionary
    metrics = {
        "test_config": {
            "num_prompts": num_prompts,
            "prompt_length": prompt_length,
            "generation_length": generation_length,
            "test_mode": test_mode,
            "execution_mode": "asynchronous",  # Mark as asynchronous execution
            "async_batch_size": async_batch_size,
            "model_path": model_path
        },
        "performance": {
            "total_requests": num_prompts,
            "successful_requests": len(success_results),
            "success_rate": success_rate,
            "total_time_seconds": total_time,
            "average_latency_seconds": avg_latency,
            "average_tokens_per_second": avg_throughput,
            "total_throughput_tokens_per_second": total_throughput,
            "total_tokens_generated": total_tokens
        },
        "detailed_results": all_results
    }

    # Save result files
    # 1. JSON summary
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)

    # 2. CSV detailed data
    with open(csv_file, "w", encoding="utf-8") as f:
        f.write("prompt_id,status,latency,generated_tokens,tokens_per_second,error\n")
        for r in all_results:
            error = r.get("error", "")
            f.write(
                f"{r['prompt_id']},{r['status']},{r.get('latency', 0):.4f},"
                f"{r.get('generated_tokens', 0)},{r.get('tokens_per_second', 0):.2f},"
                f"{error}\n"
            )

    # Print test summary
    log_info(f"\n========== {test_mode.upper()} parameter test summary ==========")
    log_info(f"Model path: {model_path}")
    log_info(f"Total requests: {num_prompts}")
    log_info(f"Successful requests: {len(success_results)}")
    log_info(f"Success rate: {success_rate:.2f}%")
    log_info(f"Total time consumption: {total_time:.2f} seconds")
    log_info(f"Average latency: {avg_latency:.2f} seconds")
    log_info(f"Average throughput: {avg_throughput:.2f} tokens/s")
    log_info(f"Total throughput: {total_throughput:.2f} tokens/s")
    log_info(f"Total generated tokens: {total_tokens}")
    log_info(f"{test_mode} test results saved to: {output_dir}")

    # Clean up resources
    cleanup_engine(engine)
    return metrics

# ===================== Build Test Configurations =====================
def build_runtime_configs(model_path, async_batch_size):
    """Build two RuntimeConfig configurations: original and optimized"""

    if is_musa():
        device = "musa"
    else:
        device = "cuda"

    # 1. Original parameter configuration
    original_config = ServerArgs(model_path=model_path)
    original_config.tokenizer_mode = "auto"
    original_config.tokenizer_worker_num = 1
    original_config.load_format = "safetensors"
    original_config.trust_remote_code = True
    original_config.dtype = "bfloat16"
    original_config.kv_cache_dtype = "auto"
    original_config.mem_fraction_static = 0.6
    original_config.max_running_requests = 48
    original_config.max_queued_requests = 9223372036854775807
    original_config.chunked_prefill_size = 8192
    original_config.max_prefill_tokens = 16384
    original_config.schedule_policy = "fcfs"
    original_config.schedule_conservativeness = 1.0
    original_config.page_size = 1
    original_config.hybrid_kvcache_ratio = None
    original_config.swa_full_tokens_ratio = 0.8
    original_config.disable_hybrid_swa_memory = False
    original_config.device = device  # Change to your actual device (cuda/musa)
    original_config.tp_size = 1
    original_config.pp_size = 1
    original_config.max_micro_batch_size = async_batch_size
    original_config.random_seed = 239081663
    original_config.attention_backend = "triton"
    original_config.sampling_backend = "flashinfer"
    original_config.triton_attention_num_kv_splits = 8
    original_config.num_continuous_decode_steps = 1
    original_config.disable_cuda_graph = True
    original_config.enable_torch_compile = False
    original_config.enable_mixed_chunk = False
    original_config.enable_two_batch_overlap = False
    original_config.enable_tokenizer_batch_encode = False
    original_config.enable_dynamic_batch_tokenizer = False
    original_config.cpu_offload_gb = 0
    original_config.enable_hierarchical_cache = False
    original_config.delete_ckpt_after_loading = False

    # 2. Optimized parameter configuration
    optimized_config = ServerArgs(model_path=model_path)
    optimized_config.__dict__.update(original_config.__dict__)

    optimized_config.mem_fraction_static = 0.7         # Optimization: Increase static memory ratio
    optimized_config.page_size = 64                    # Optimization: Increase KV cache page size
    optimized_config.hybrid_kvcache_ratio = 0.6        # Optimization: Enable hybrid KV cache
    optimized_config.attention_backend = "fa3"
    optimized_config.sampling_backend = "flashinfer"
    optimized_config.num_continuous_decode_steps = 4   # Optimization: Increase continuous decode steps
    optimized_config.disable_cuda_graph = False        # Optimization: Enable CUDA Graph

    return {
        "original": original_config,
        "optimized": optimized_config
    }

# ===================== Generate Comparison Report =====================
def generate_comparison_report(original_metrics, optimized_metrics):
    """Generate comparison report between original vs optimized parameters"""
    log_info("\n========== Final Comparison Report (Asynchronous Execution) ==========")

    # Extract core metrics
    orig_throughput = original_metrics["performance"]["total_throughput_tokens_per_second"]
    opt_throughput = optimized_metrics["performance"]["total_throughput_tokens_per_second"]
    orig_latency = original_metrics["performance"]["average_latency_seconds"]
    opt_latency = optimized_metrics["performance"]["average_latency_seconds"]
    orig_success = original_metrics["performance"]["success_rate"]
    opt_success = optimized_metrics["performance"]["success_rate"]

    # Calculate improvement rate
    throughput_improvement = ((opt_throughput - orig_throughput) / orig_throughput * 100) if orig_throughput > 0 else 0
    latency_change = ((opt_latency - orig_latency) / orig_latency * 100) if orig_latency > 0 else 0

    # Print comparison results
    log_info(f"Total throughput comparison:")
    log_info(f"  Original parameters: {orig_throughput:.2f} tokens/s")
    log_info(f"  Optimized parameters: {opt_throughput:.2f} tokens/s")
    log_info(f"  Improvement rate: {throughput_improvement:.2f}%")

    log_info(f"\nAverage latency comparison:")
    log_info(f"  Original parameters: {orig_latency:.2f} seconds")
    log_info(f"  Optimized parameters: {opt_latency:.2f} seconds")
    log_info(f"  Change rate: {latency_change:.2f}%")

    log_info(f"\nSuccess rate comparison:")
    log_info(f"  Original parameters: {orig_success:.2f}%")
    log_info(f"  Optimized parameters: {opt_success:.2f}%")

    # Save comparison report
    comparison_report = {
        "test_config": original_metrics["test_config"],
        "comparison": {
            "throughput": {
                "original": orig_throughput,
                "optimized": opt_throughput,
                "improvement_percent": throughput_improvement
            },
            "latency": {
                "original": orig_latency,
                "optimized": opt_latency,
                "change_percent": latency_change
            },
            "success_rate": {
                "original": orig_success,
                "optimized": opt_success
            }
        },
        "original_full_metrics": original_metrics,
        "optimized_full_metrics": optimized_metrics
    }

    report_file = os.path.join(ROOT_OUTPUT_DIR, "comparison_report_async.json")
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(comparison_report, f, indent=4, ensure_ascii=False)

    log_info(f"\nComplete comparison report saved to: {report_file}")
    log_info(f"All test results root directory: {ROOT_OUTPUT_DIR}")

# ===================== Async Main Function =====================
async def main(run_type, model_path, num_prompts, generation_length, async_batch_size):
    """Async main test flow (supports specified run_type and custom parameters)"""
    # Validate run_type parameter validity
    valid_run_types = ["original", "optimized", "both"]
    if run_type not in valid_run_types:
        log_info(f"Error: Invalid run_type parameter '{run_type}', valid values are: {valid_run_types}")
        return

    log_info(f"========== Start SGLang Offline Engine Performance Comparison Test (Asynchronous Non-streaming) ==========")
    log_info(f"Test type: {run_type.upper()}")
    log_info(f"Model path: {model_path}")
    log_info(f"Total test requests: {num_prompts}")
    log_info(f"Generated text length: {generation_length} tokens")
    log_info(f"Async batch size: {async_batch_size}")

    # 1. Build configurations (pass custom model path)
    configs = build_runtime_configs(model_path, async_batch_size)
    original_metrics = None
    optimized_metrics = None

    # 2. Execute corresponding tests according to run_type
    if run_type in ["original", "both"]:
        # Execute original parameter test
        original_metrics = await run_async_performance_test(
            "original", 
            configs["original"], 
            model_path, 
            num_prompts, 
            PROMPT_LENGTH, 
            generation_length, 
            async_batch_size
        )
        if original_metrics is None:
            log_info("Original parameter test failed")
            if run_type == "original":
                return
            else:  # Original test failed in both mode, terminate subsequent optimized test
                log_info("Original parameter test failed in both mode, terminating optimized parameter test")
                return

    if run_type in ["optimized", "both"]:
        # Execute optimized parameter test
        optimized_metrics = await run_async_performance_test(
            "optimized", 
            configs["optimized"], 
            model_path, 
            num_prompts, 
            PROMPT_LENGTH, 
            generation_length, 
            async_batch_size
        )
        if optimized_metrics is None:
            log_info("Optimized parameter test failed")
            return

    # 3. Generate comparison report only in both mode
    if run_type == "both" and original_metrics and optimized_metrics:
        generate_comparison_report(original_metrics, optimized_metrics)

    log_info("\n========== Test completed ==========")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="SGLang Asynchronous Performance Test Tool")
    parser.add_argument(
        "-r",
        "--run_type", 
        type=str, 
        default="both",
        choices=["original", "optimized", "both"],
        help="Test type: original(only original config) / optimized(only optimized config) / both(test both and compare)"
    )
    parser.add_argument(
        "-m",
        "--model_path", 
        type=str, 
        default="./Qwen3-1.7B",
        help="Model path, default value: ./Qwen3-1.7B"
    )
    parser.add_argument(
        "-g",
        "--generation_length", 
        type=int, 
        default=1024,
        help="Generated text length (number of tokens), default value: 1024"
    )
    parser.add_argument(
        "-b",
        "--async_batch_size", 
        type=int, 
        default=48,
        help="Asynchronous batch size (number of requests processed per batch), default value: 48"
    )
    parser.add_argument(
        "-n",
        "--num_prompts", 
        type=int, 
        default=96,
        help="Total number of test requests, default value: 96"
    )
    args = parser.parse_args()

    # Validate numerical parameter validity
    if args.generation_length <= 0:
        log_info("Error: generation_length must be a positive integer")
        exit(1)
    if args.async_batch_size <= 0:
        log_info("Error: async_batch_size must be a positive integer")
        exit(1)
    if args.num_prompts <= 0:
        log_info("Error: num_prompts must be a positive integer")
        exit(1)

    # Run async main function
    asyncio.run(main(
        args.run_type,
        args.model_path,
        args.num_prompts,
        args.generation_length,
        args.async_batch_size
    ))