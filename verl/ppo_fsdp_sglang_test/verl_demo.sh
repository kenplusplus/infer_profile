#!/bin/bash

CURR_DIR=$(dirname $(realpath $0))
export SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK=True
export PYTHONPATH=/home/nanhu/nanhu-verl/:/home/nanhu/nanhu-sglang/:$PYTHONPATH
export VERL_LOGGING_LEVEL=DEBUG
export SGLANG_LOG_LEVEL=DEBUG
export SGLANG_LOG_VERBOSE=1
export LOG_LEVEL=DEBUG
export TRANSFORMERS_NO_FLASH_ATTENTION=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False,max_split_size_mb:512"

PYTHONUNBUFFERED=1 SGLANG_LOG_LEVEL=DEBUG LOG_LEVEL=DEBUG SGLANG_LOG_VERBOSE=1 python3 -m verl.trainer.main_ppo \
    data.train_files=/home/gsm8k/train.parquet \
    data.val_files=/home/gsm8k/test.parquet \
    data.train_batch_size=256 \
    data.max_prompt_length=512 \
    data.max_response_length=512 \
    actor_rollout_ref.model.path=/home/Qwen3-8B \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=8 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    critic.optim.lr=1e-5 \
    critic.model.path=/home/Qwen3-8B \
    critic.ppo_micro_batch_size_per_gpu=4 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.logger=console \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=8 \
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
