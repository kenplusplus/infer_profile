#!/bin/bash
# 先将筛选出的PID保存到变量中
PIDS=$(mthreads-gmi | grep -E '^[0-9]+\s+[0-9]+\s+' | awk '{print $2}' | sort -u)

# 检查变量是否非空，非空时才执行kill
if [ -n "$PIDS" ]; then
    echo "正在终止进程: $PIDS"  # 可选：打印要终止的PID，方便排查
    echo "$PIDS" | xargs kill -9
else
    echo "未找到需要终止的进程"  # 可选：提示无进程，可根据需要删除
fi
