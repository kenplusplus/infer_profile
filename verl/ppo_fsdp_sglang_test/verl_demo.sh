#!/bin/bash

# ===================== 命令行参数解析 =====================
# 默认值配置（保持原有配置作为默认）
DEFAULT_TENSOR_MODEL_PARALLEL_SIZE=8
DEFAULT_N_GPUS_PER_NODE=8
DEFAULT_MODEL_PATH="/home/Qwen3-8B"

# 显示帮助信息
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo "VERL Training Script with Configurable Parameters"
    echo ""
    echo "Options:"
    echo "  -h, --help                  Show this help message and exit"
    echo "  -t, --tp-size NUM           Set tensor_model_parallel_size (default: $DEFAULT_TENSOR_MODEL_PARALLEL_SIZE)"
    echo "  -g, --gpus-per-node NUM     Set n_gpus_per_node (default: $DEFAULT_N_GPUS_PER_NODE)"
    echo "  -m, --model-path PATH       Set model path (default: $DEFAULT_MODEL_PATH)"
    echo ""
    echo "Examples:"
    echo "  # Use default values"
    echo "  $0"
    echo ""
    echo "  # Custom TP size and GPU count"
    echo "  $0 -t 4 -g 4"
    echo ""
    echo "  # Custom model path"
    echo "  $0 -m /home/Qwen3-1.7B"
    echo ""
    echo "  # Full custom configuration"
    echo "  $0 --tp-size 2 --gpus-per-node 2 --model-path /home/Qwen3-72B"
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            show_help
            exit 0
            ;;
        -t|--tp-size)
            TENSOR_MODEL_PARALLEL_SIZE="$2"
            shift 2
            ;;
        -g|--gpus-per-node)
            N_GPUS_PER_NODE="$2"
            shift 2
            ;;
        -m|--model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        *)
            echo "Error: Unknown option $1"
            show_help
            exit 1
            ;;
    esac
done

# 设置默认值（如果未通过命令行指定）
TENSOR_MODEL_PARALLEL_SIZE=${TENSOR_MODEL_PARALLEL_SIZE:-$DEFAULT_TENSOR_MODEL_PARALLEL_SIZE}
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-$DEFAULT_N_GPUS_PER_NODE}
MODEL_PATH=${MODEL_PATH:-$DEFAULT_MODEL_PATH}

# ===================== 环境配置 =====================
CURR_DIR=$(dirname $(realpath $0))
export SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK=True
export PYTHONPATH=/home/nanhu/nanhu-verl/:/home/nanhu/nanhu-sglang/:$PYTHONPATH
export VERL_LOGGING_LEVEL=DEBUG
export SGLANG_LOG_LEVEL=DEBUG
export SGLANG_LOG_VERBOSE=1
export LOG_LEVEL=DEBUG
export TRANSFORMERS_NO_FLASH_ATTENTION=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False,max_split_size_mb:512"

# 打印配置信息（方便调试）
echo "========================================"
echo "VERL Training Configuration"
echo "========================================"
echo "Tensor Model Parallel Size: $TENSOR_MODEL_PARALLEL_SIZE"
echo "GPUs per Node: $N_GPUS_PER_NODE"
echo "Model Path: $MODEL_PATH"
echo "========================================"
echo ""

# ===================== 启动训练 =====================
PYTHONUNBUFFERED=1 SGLANG_LOG_LEVEL=DEBUG LOG_LEVEL=DEBUG SGLANG_LOG_VERBOSE=1 python3 -m verl.trainer.main_ppo \
    data.train_files=/home/gsm8k/train.parquet \
    data.val_files=/home/gsm8k/test.parquet \
    data.train_batch_size=16 \
    data.max_prompt_length=512 \
    data.max_response_length=512 \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${TENSOR_MODEL_PARALLEL_SIZE} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    critic.optim.lr=1e-5 \
    critic.model.path=${MODEL_PATH} \
    critic.ppo_micro_batch_size_per_gpu=1 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.logger=console \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.total_epochs=15 \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.log_level="DEBUG" \
    +logging.disable_existing_loggers=false \
    +logging.root.level="DEBUG" \
    +logging.root.handlers="[console]" \
    +logging.loggers.verl.rollout.level="DEBUG" \
    +logging.loggers.verl.rollout.handlers="[console]" \
    +logging.loggers.verl.workers.rollout.level="DEBUG" \
    +logging.loggers.verl.workers.rollout.handlers="[console]" \
    +logging.loggers.verl.workers.rollout.propagate=true \
    2>&1 | tee verl_demo.log