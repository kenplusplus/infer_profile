#!/bin/bash
# verl-config-demo.sh

# ========== 1. ~N~C~O~X~G~O~E~M__ ==========
export SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK=True
export PYTHONPATH=/workspace/data/kenlu/nanhu-verl/:/workspace/data/kenlu/nanhu-sglang/:$PYTHONPATH
export VERL_LOGGING_LEVEL=DEBUG
export SGLANG_LOG_LEVEL=DEBUG
export SGLANG_LOG_VERBOSE=1
export LOG_LEVEL=DEBUG
export TRANSFORMERS_NO_FLASH_ATTENTION=1
export DATA_ROOT=/workspace/data/
export PYTHONUNBUFFERED=1

python3 -m verl.trainer.main_ppo \
  --config-name verl_ppo_config \
  --config-path /home/infer_profile/verl/ppo_fsdp_sglang_test \
  2>&1 | tee verl_demo.log
