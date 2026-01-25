import time
import json
import numpy as np
from sglang import Engine
import os
import gc
import torch
import asyncio  # 新增异步依赖

from sglang.srt.server_args import PortArgs, ServerArgs

# ===================== 全局配置 =====================
# 模型路径（请根据你的实际路径修改）
MODEL_PATH = "./Qwen3-1.7B"
# 测试根目录
ROOT_OUTPUT_DIR = "./sglang_perf_test_async/"
# 压测参数（异步批量测试参数）
NUM_PROMPTS = 10                # 测试请求总数
PROMPT_LENGTH = 512             # 输入prompt长度（token数）
GENERATION_LENGTH = 1024        # 生成文本长度（token数）
ASYNC_BATCH_SIZE = 5            # 异步批量大小（每批处理的请求数）

# 创建根目录
os.makedirs(ROOT_OUTPUT_DIR, exist_ok=True)
GLOBAL_LOG_FILE = os.path.join(ROOT_OUTPUT_DIR, "global_test_log.txt")

# ===================== 通用工具函数 =====================
def log_info(msg):
    """全局日志函数：记录日志并打印"""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    log_msg = f"[{timestamp}] {msg}"
    print(log_msg)
    with open(GLOBAL_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(log_msg + "\n")

def generate_test_prompt(length):
    """生成指定长度的测试Prompt"""
    return " ".join(["test"] * length)

def cleanup_engine(engine):
    """清理Engine资源，释放显存"""
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
    log_info("Engine资源已清理完成")

# ===================== 异步性能测试核心函数 =====================
async def async_send_batch_requests(engine, prompt_batch, prompt_id_batch):
    """
    异步发送批量请求
    :param engine: SGLang Engine实例
    :param prompt_batch: 批量prompt列表
    :param prompt_id_batch: 对应的prompt ID列表
    :return: 批量请求结果列表
    """
    sampling_params = {
        'max_new_tokens': GENERATION_LENGTH,
        'temperature': 0.8,
        'top_p': 0.95,
        'repetition_penalty': 1.0,
        'top_k': 100,
        'presence_penalty': 0.0
    }
    
    try:
        batch_start_time = time.time()
        # 核心：异步批量生成（Non-streaming Asynchronous Generation）
        outputs = await engine.async_generate(prompt_batch, sampling_params)
        batch_end_time = time.time()
        
        batch_results = []
        for idx, (prompt_id, output) in enumerate(zip(prompt_id_batch, outputs)):
            # 计算单个请求耗时（批量总耗时均分，或用output内置耗时）
            single_latency = (batch_end_time - batch_start_time) / len(prompt_batch)
            
            # 兼容返回值格式
            if hasattr(output, 'text'):
                generated_text = output.text
            elif isinstance(output, dict) and 'text' in output:
                generated_text = output['text']
            elif isinstance(output, dict) and 'outputs' in output:
                generated_text = output['outputs'][0]['text']
            else:
                raise ValueError(f"不支持的返回值格式: {type(output)}")
            
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
        # 批量请求失败时，标记所有prompt为错误
        return [{
            "prompt_id": prompt_id,
            "status": "error",
            "error": str(e),
            "latency": 0,
            "generated_tokens": 0,
            "tokens_per_second": 0
        } for prompt_id in prompt_id_batch]

async def run_async_performance_test(test_mode, runtime_config):
    """
    执行异步性能测试（Non-streaming Asynchronous Generation）
    :param test_mode: 测试模式（original/optimized）
    :param runtime_config: 对应的RuntimeConfig配置
    :return: 测试结果字典
    """
    # 初始化测试目录
    output_dir = os.path.join(ROOT_OUTPUT_DIR, test_mode)
    os.makedirs(output_dir, exist_ok=True)
    metrics_file = os.path.join(output_dir, "metrics.json")
    csv_file = os.path.join(output_dir, "performance.csv")
    log_info(f"\n========== 开始{test_mode.upper()}参数异步测试 ==========")

    # 初始化Engine
    log_info(f"初始化{test_mode}参数的SGLang Engine...")
    try:
        engine = Engine(
            model_path=MODEL_PATH,
            server_args=runtime_config,
        )
    except Exception as e:
        log_info(f"Engine初始化失败: {str(e)}")
        return None

    # 预热（避免首次请求耗时影响测试）
    log_info(f"{test_mode} Engine预热中...")
    warmup_prompts = [" ".join(["test"] * 10) for _ in range(2)]  # 批量预热
    await engine.async_generate(
        prompts=warmup_prompts,
        sampling_params={'max_new_tokens': 10}
    )
    log_info(f"{test_mode} Engine预热完成")

    # 生成所有测试prompt和对应的ID
    all_prompts = [generate_test_prompt(PROMPT_LENGTH) for _ in range(NUM_PROMPTS)]
    all_prompt_ids = list(range(NUM_PROMPTS))

    # 拆分批量（按ASYNC_BATCH_SIZE分组）
    prompt_batches = [
        all_prompts[i:i + ASYNC_BATCH_SIZE] 
        for i in range(0, NUM_PROMPTS, ASYNC_BATCH_SIZE)
    ]
    prompt_id_batches = [
        all_prompt_ids[i:i + ASYNC_BATCH_SIZE] 
        for i in range(0, NUM_PROMPTS, ASYNC_BATCH_SIZE)
    ]

    # 执行异步测试
    log_info(f"{test_mode} 异步测试开始：{NUM_PROMPTS}个请求，分{len(prompt_batches)}批处理，每批{ASYNC_BATCH_SIZE}个请求")
    total_start = time.time()
    all_results = []

    # 逐批执行异步请求（也可改为并发执行多批，根据需求调整）
    for batch_idx, (prompt_batch, prompt_id_batch) in enumerate(zip(prompt_batches, prompt_id_batches)):
        log_info(f"处理第{batch_idx + 1}/{len(prompt_batches)}批请求，包含{len(prompt_batch)}个prompt")
        batch_results = await async_send_batch_requests(engine, prompt_batch, prompt_id_batch)
        all_results.extend(batch_results)

    # 计算总耗时
    total_end = time.time()
    total_time = total_end - total_start

    # 统计结果
    success_results = [r for r in all_results if r["status"] == "success"]
    total_tokens = sum([r["generated_tokens"] for r in success_results])

    # 核心性能指标计算
    success_rate = len(success_results) / NUM_PROMPTS * 100 if NUM_PROMPTS > 0 else 0
    avg_latency = np.mean([r["latency"] for r in success_results]) if success_results else 0
    avg_throughput = np.mean([r["tokens_per_second"] for r in success_results]) if success_results else 0
    total_throughput = total_tokens / total_time if total_time > 0 else 0

    # 构建指标字典
    metrics = {
        "test_config": {
            "num_prompts": NUM_PROMPTS,
            "prompt_length": PROMPT_LENGTH,
            "generation_length": GENERATION_LENGTH,
            "test_mode": test_mode,
            "execution_mode": "asynchronous",  # 标记为异步执行
            "async_batch_size": ASYNC_BATCH_SIZE
        },
        "performance": {
            "total_requests": NUM_PROMPTS,
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

    # 保存结果文件
    # 1. JSON汇总
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)

    # 2. CSV详细数据
    with open(csv_file, "w", encoding="utf-8") as f:
        f.write("prompt_id,status,latency,generated_tokens,tokens_per_second,error\n")
        for r in all_results:
            error = r.get("error", "")
            f.write(
                f"{r['prompt_id']},{r['status']},{r.get('latency', 0):.4f},"
                f"{r.get('generated_tokens', 0)},{r.get('tokens_per_second', 0):.2f},"
                f"{error}\n"
            )

    # 打印测试汇总
    log_info(f"\n========== {test_mode.upper()}参数测试汇总 ==========")
    log_info(f"总请求数: {NUM_PROMPTS}")
    log_info(f"成功请求数: {len(success_results)}")
    log_info(f"成功率: {success_rate:.2f}%")
    log_info(f"总耗时: {total_time:.2f}秒")
    log_info(f"平均延迟: {avg_latency:.2f}秒")
    log_info(f"平均吞吐率: {avg_throughput:.2f} tokens/s")
    log_info(f"总吞吐率: {total_throughput:.2f} tokens/s")
    log_info(f"总生成Token数: {total_tokens}")
    log_info(f"{test_mode}测试结果保存至: {output_dir}")

    # 清理资源
    cleanup_engine(engine)
    return metrics

# ===================== 构建测试配置 =====================
def build_runtime_configs():
    """构建原始和优化两种RuntimeConfig配置"""
    # 1. 原始参数配置
    original_config = ServerArgs(model_path=MODEL_PATH)
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
    original_config.device = "musa"  # 改为你实际使用的设备（cuda/musa）
    original_config.tp_size = 1
    original_config.pp_size = 1
    original_config.max_micro_batch_size = 32
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

    # 2. 优化参数配置
    optimized_config = ServerArgs(model_path=MODEL_PATH)
    optimized_config.__dict__.update(original_config.__dict__)

    optimized_config.mem_fraction_static = 0.7         # 优化：增大静态显存比例
    optimized_config.page_size = 64                    # 优化：增大KV缓存页大小
    optimized_config.hybrid_kvcache_ratio = 0.6        # 优化：启用混合KV缓存
    optimized_config.attention_backend = "fa3"
    optimized_config.sampling_backend = "flashinfer"
    optimized_config.num_continuous_decode_steps = 4   # 优化：增加连续解码步数
    optimized_config.disable_cuda_graph = False        # 优化：启用CUDA Graph

    return {
        "original": original_config,
        "optimized": optimized_config
    }

# ===================== 生成对比报告 =====================
def generate_comparison_report(original_metrics, optimized_metrics):
    """生成原始vs优化参数的对比报告"""
    log_info("\n========== 最终对比报告（异步执行）==========")

    # 提取核心指标
    orig_throughput = original_metrics["performance"]["total_throughput_tokens_per_second"]
    opt_throughput = optimized_metrics["performance"]["total_throughput_tokens_per_second"]
    orig_latency = original_metrics["performance"]["average_latency_seconds"]
    opt_latency = optimized_metrics["performance"]["average_latency_seconds"]
    orig_success = original_metrics["performance"]["success_rate"]
    opt_success = optimized_metrics["performance"]["success_rate"]

    # 计算提升率
    throughput_improvement = ((opt_throughput - orig_throughput) / orig_throughput * 100) if orig_throughput > 0 else 0
    latency_change = ((opt_latency - orig_latency) / orig_latency * 100) if orig_latency > 0 else 0

    # 打印对比结果
    log_info(f"总吞吐率对比:")
    log_info(f"  原始参数: {orig_throughput:.2f} tokens/s")
    log_info(f"  优化参数: {opt_throughput:.2f} tokens/s")
    log_info(f"  提升率: {throughput_improvement:.2f}%")

    log_info(f"\n平均延迟对比:")
    log_info(f"  原始参数: {orig_latency:.2f} 秒")
    log_info(f"  优化参数: {opt_latency:.2f} 秒")
    log_info(f"  变化率: {latency_change:.2f}%")

    log_info(f"\n成功率对比:")
    log_info(f"  原始参数: {orig_success:.2f}%")
    log_info(f"  优化参数: {opt_success:.2f}%")

    # 保存对比报告
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

    log_info(f"\n完整对比报告已保存至: {report_file}")
    log_info(f"所有测试结果根目录: {ROOT_OUTPUT_DIR}")

# ===================== 异步主函数 =====================
async def main():
    """异步主测试流程"""
    log_info("========== 开始SGLang Offline Engine 性能对比测试（异步非流式）==========")

    # 1. 构建配置
    configs = build_runtime_configs()

    # 2. 执行原始参数测试
    original_metrics = await run_async_performance_test("original", configs["original"])
    if original_metrics is None:
        log_info("原始参数测试失败，终止测试")
        return

    # 3. 执行优化参数测试
    optimized_metrics = await run_async_performance_test("optimized", configs["optimized"])
    if optimized_metrics is None:
        log_info("优化参数测试失败")
        return

    # 4. 生成对比报告
    generate_comparison_report(original_metrics, optimized_metrics)

    log_info("\n========== 所有异步测试完成 ==========")

if __name__ == "__main__":
    # 运行异步主函数
    asyncio.run(main())