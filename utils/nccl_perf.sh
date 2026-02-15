#!/bin/bash
#set -e  # Exit immediately if a command exits with a non-zero status

# ===================== Configurable Parameters =====================
# 1. Number of GPUs for testing (fixed to 8 cards)
GPU_NUM=8
# 2. Prefix directory for NCCL perf tools (customizable, default: /home/nccl-tests/build)
NCCL_TOOLS_DIR="/home/nccl-tests/build"
# 3. Get current date (format: year-month-day_hour-minute-second) as log file prefix
DATE_PREFIX=$(date +"%Y-%m-%d_%H-%M-%S")
# 4. List of NCCL perf tools (only tool names, path will be concatenated with NCCL_TOOLS_DIR)
NCCL_TOOLS=(
    gather_perf
    hypercube_perf
    reduce_perf
    reduce_scatter_perf
    scatter_perf
    sendrecv_perf
    all_gather_perf
    all_reduce_perf
    alltoall_perf
    broadcast_perf
)
# ===================== Core Logic =====================

# Check if the number of GPUs is 8
gpu_count=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits 2>/dev/null || echo 0)
#if [ $gpu_count -ne $GPU_NUM ]; then
#    echo "Error: The number of GPUs on the current machine is not 8 (detected $gpu_count cards)! Cannot execute tests."
#    exit 1
#fi

# Check if torchrun exists
if ! command -v torchrun &> /dev/null; then
    echo "Error: torchrun command not found! Please install PyTorch first."
    exit 1
fi

# Check if NCCL tools directory exists
if [ ! -d "$NCCL_TOOLS_DIR" ]; then
    echo "Warning: Configured NCCL tools directory $NCCL_TOOLS_DIR does not exist!"
    echo "Will try to find tools from system PATH..."
fi

# Check and get the full path of the tool
get_tool_path() {
    local tool_name=$1
    # First try to find from the configured directory
    local tool_path="${NCCL_TOOLS_DIR}/${tool_name}"
    if [ -x "$tool_path" ]; then
        echo "$tool_path"
        return 0
    fi
    # If not found in configured directory, find from system PATH
    if command -v $tool_name &> /dev/null; then
        echo "$(command -v $tool_name)"
        return 0
    fi
    # Not found in either location
    echo ""
    return 1
}

# Execute each NCCL perf test in a loop
for tool in "${NCCL_TOOLS[@]}"; do
    # Get the full path of the tool
    tool_path=$(get_tool_path $tool)
    if [ -z "$tool_path" ]; then
        echo "Warning: Tool $tool not found (directory: $NCCL_TOOLS_DIR / system PATH), skipping this test."
        continue
    fi

    # Define log file name: date_prefix_tool_name.log
    LOG_FILE="${DATE_PREFIX}_${tool}.log"
    echo "========================================"
    echo "Starting $tool test (path: $tool_path)"
    echo "Results will be saved to: $LOG_FILE"
    echo "========================================"

    # Launch 8-card test with torchrun
    $tool \
        -e 128M \
        -f 2 \
        -g $GPU_NUM \
        2>&1 | tee $LOG_FILE

    echo "✅ $tool test completed! Log file: $LOG_FILE"
    echo ""
done

echo "🎉 All NCCL performance tests completed!"
echo "List of log files:"
ls -l ${DATE_PREFIX}_*.log 2>/dev/null || echo "No log files generated (all tools may not be found)"
